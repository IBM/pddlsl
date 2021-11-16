#!/usr/bin/env python

"""
For usage, run `plan.py -h` and read README.org
"""

import os
import argparse
import errno
import json
import numpy
import torch
import time
import math
import warnings
from typing import Optional, Type
from pytorch_lightning import seed_everything

from pyperplan.planner import _parse, _ground
from pyperplan.task import Task
from pyperplan.heuristics.heuristic_base import Heuristic

from pddlsl.heuristic_learner import (
    HeuristicLearner,
    SupervisedLearner
)

from pddlsl.model_wrapper import (
    NLMCarlosV2Wrapper,
    HGNWrapper,
    RRWrapper,
)

from pddlsl.custom_pyperplan.lm_cut import LmCutHeuristic
from pddlsl.custom_pyperplan.relaxation import hMaxHeuristic, hFFHeuristic
from pddlsl.custom_pyperplan.a_star import greedy_best_first_search
from pddlsl.custom_pyperplan.constant_heuristics import ZeroHeuristic, NinfHeuristic, InfHeuristic
from pyperplan.heuristics.blind import BlindHeuristic
from pddlsl.pyperplan_heuristic_nlm import NLMCarlosV2Heuristic
from pddlsl.pyperplan_heuristic_hgn import HGNHeuristic
from pddlsl.pyperplan_heuristic_gomoluch import LinearHeuristic

from util import *
from pddlsl.constants import COSTS

PYPERPLAN_HEURISTICS = {
    'lmcut': LmCutHeuristic,
    'hmax' : hMaxHeuristic,
    'ff' : hFFHeuristic,
    'zero' : ZeroHeuristic,
    'inf'  : InfHeuristic,
    'ninf' : NinfHeuristic,
    'blind' : BlindHeuristic,
}

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument('--force', '-f', action="store_true",
                        help="""If this flag is on, then we repeat those experiments that are already done.""")
    parser.add_argument('--verbose', '-v', action="store_true")
    parser.add_argument('--max-evaluations', type=int, default=float("inf"), help="maximum number of evaluations")

    parser.add_argument('--checkpoint', default="nll", choices=("nll","mse","last"), help="load the best v_mse/v_nll/last checkpoint.")

    parser.add_argument('-l', '--l-test', choices=COSTS, help="overrides --l-test specified in the training.")
    parser.add_argument('-u', '--u-test', choices=COSTS, help="overrides --u-test specified in the training.")
    parser.add_argument('-r', '--res-test', choices=COSTS, help="overrides --res-test specified in the training.")
    parser.add_argument('--elbo-prior-mu-test', choices=COSTS, help="overrides --elbo-prior-mu-test specified in the training.")
    parser.add_argument('--elbo-blind-l-test', choices=COSTS, help="overrides --elbo-blind-l-test specified in the training.")

    parser.add_argument('--discretize', action="store_true", help="discretize the predicted heuristic value.")
    parser.add_argument('--tiebreak', choices=PYPERPLAN_HEURISTICS.keys(), help="break ties with this heuristics.")

    parser.add_argument('heuristic',
                        help=("A JSON result file for the trained model, or "
                              f"a string name designating a heuristic, which is one of: "
                              f"{','.join(PYPERPLAN_HEURISTICS.keys())}."))

    parser.add_argument('problem', help="problem file")
    parser.add_argument('domain', nargs="?", help="domain file, but it is automatically deduced if missing.")

    args = parser.parse_args()

    if args.domain is None:
        args.domain = os.path.join(os.path.dirname(args.problem), "..", "domain.pddl")

    print(f"problem: {args.problem}")
    print(f"domain:  {args.domain}")
    if not os.path.isfile(args.problem):
        raise Exception(f"The problem file ({args.problem}) does not exist.")
    if not os.path.isfile(args.domain):
        raise Exception(f"The domain file ({args.domain}) does not exist.")

    return args


def find_heuristic(name:Optional[str]) -> Optional[Type]:
    if name is None:
        return None

    elif name in PYPERPLAN_HEURISTICS.keys():
        return PYPERPLAN_HEURISTICS[name]

    else:
        raise ValueError(f"heuristic {name} not found, should be one of: {tuple(PYPERPLAN_HEURISTICS.keys())}")


@idcache
def load_and_update_model_info(args:argparse.Namespace):
    """Load the model metadata from a heuristic json file, then supersede {l,u,res}_test."""
    # args.heuristic is a path to json
    with open(args.heuristic, 'r') as f:
        model_info = json.load(f)

    if args.l_test is not None:
        if args.verbose:
            print(f"overriding l_test: {model_info['l_test']} -> {args.l_test}")
        model_info['l_test'] = args.l_test
    if args.u_test is not None:
        if args.verbose:
            print(f"overriding u_test: {model_info['u_test']} -> {args.u_test}")
        model_info['u_test'] = args.u_test
    if args.res_test is not None:
        if args.verbose:
            print(f"overriding res_test: {model_info['res_test']} -> {args.res_test}")
        model_info['res_test'] = args.res_test
    if args.elbo_prior_mu_test is not None:
        if args.verbose:
            print(f"overriding elbo_prior_mu_test: {model_info['elbo_prior_mu_test']} -> {args.elbo_prior_mu_test}")
        model_info['elbo_prior_mu_test'] = args.elbo_prior_mu_test
    if args.elbo_blind_l_test is not None:
        if args.verbose:
            print(f"overriding elbo_blind_l_test: {model_info['elbo_blind_l_test']} -> {args.elbo_blind_l_test}")
        model_info['elbo_blind_l_test'] = args.elbo_blind_l_test

    if args.checkpoint == "mse":
        model_info['best_path'] = model_info['best_mse_path']
    elif args.checkpoint == "nll":
        model_info['best_path'] = model_info['best_nll_path']
    elif args.checkpoint == "last":
        model_info['best_path'] = model_info['last_path']
    else:
        raise "huh?"

    # Each model (NLM, RR, HGN) requires a different data functionality
    if model_info["model"] == "NLMCarlosV2":
        model_info["network_cls"] = NLMCarlosV2Wrapper
    elif model_info["model"] == "RR":
        model_info["network_cls"] = RRWrapper
    elif model_info["model"] == "HGN":
        model_info["network_cls"] = HGNWrapper
    else:
        raise "huh?"

    # Right now, "learner" must be "supervised"
    if model_info["learner"] == 'supervised':
        model_info["learner_cls"] = SupervisedLearner
    else:
        raise 'huh?'

    return model_info


def instantiate_heuristic(args:argparse.Namespace, task:Task) -> Heuristic:
    try:
        return find_heuristic(args.heuristic)(task)
    except:
        model_info = load_and_update_model_info(args)

        print(f"instantiating a heuristic from {args.heuristic}")

        if model_info["learner"] == "supervised":
            model_cls = SupervisedLearner
        else:
            raise "huh?"

        if model_info["model"] == "NLMCarlosV2":
            h_cls = NLMCarlosV2Heuristic
        elif model_info["model"] == "RR":
            h_cls = LinearHeuristic
        elif model_info["model"] == "HGN":
            h_cls = HGNHeuristic
        else:
            raise "huh?"

        h = h_cls(task, args.domain, args.problem,
                  model_cls.load_from_checkpoint(model_info['best_path'],map_location=torch.device('cpu')),
                  find_heuristic(model_info['l_test']),
                  find_heuristic(model_info['u_test']),
                  find_heuristic(model_info['res_test']),
                  args.discretize)

        if args.tiebreak is not None:
            h_tie = find_heuristic(args.tiebreak)(task)
            h2 = lambda node: (h(node), h_tie(node))
        else:
            h2 = h

        return h2



@idcache
def planning_json_path(args:argparse.Namespace):
    logdir = os.path.splitext(args.problem)[0]+".logs"
    os.makedirs(logdir, exist_ok=True)
    try:
        find_heuristic(args.heuristic) # raises an error when args.heuristic is a json file
        return os.path.join(logdir, f'{args.heuristic}.json')
    except:
        model_info = load_and_update_model_info(args)
        model_info_args = argparse.Namespace(**model_info)
        id = model_info["learner_cls"].id(model_info_args) + "_" + model_info["network_cls"].id(model_info_args)
        id += "-" + args.checkpoint
        if args.discretize:
            id += f"-{args.discretize}"
        if args.tiebreak:
            id += f"-{args.tiebreak}"
        return os.path.join(logdir, f'{id}.json')


def save_results(args, plan, plan_length, evaluated, elapsed):

    data = dict()

    try:
        find_heuristic(args.heuristic)
        data['model'] = args.heuristic
    except:
        # Note: this contains "domain" field already,
        # which overlaps the key in args.
        # However, args.domain is a filename, while
        # model_info["domain"] is a label like "blocksworld".
        data.update(load_and_update_model_info(args))

    # data["problem"] is inserted here.
    for k, v in vars(args).items():
        if k not in {"verbose", "force",
                     # respect the training-time configuration
                     "l_test",
                     "u_test",
                     "res_test",
                     "elbo_prior_mu_test",
                     "elbo_blind_l_test",}:
            data[k] = v

    if "learner_cls" in data:   # in ff + GBFS baseline, this key does not exist
        data["learner_cls"] = data["learner_cls"].__name__
    if "network_cls" in data:   # in ff + GBFS baseline, this key does not exist
        data["network_cls"] = data["network_cls"].__name__

    # re-insert the domain path. this is the correct one.
    data['domain'] = os.path.join(os.path.dirname(os.path.dirname(data['problem'])), "domain.pddl")
    # domain name comes in here.
    data['dname'] = os.path.basename(os.path.dirname(data['domain']))
    data['pname'] = basename_noext(args.problem)
    if plan is not None:
        data['plan'] = [op.name for op in plan]
    data['plan_length'] = plan_length
    data['evaluated']   = evaluated
    data['elapsed']     = elapsed
    data['max_evaluations'] = args.max_evaluations
    if args.verbose:
        print(f"writing to {planning_json_path(args)}")
    save_json(data, planning_json_path(args))
    pass


def main(args):

    seed_everything(42, workers=True)
    warnings.simplefilter("ignore")

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    def already_solved_or_failure_with_larger_resources(path):
        with open(path) as f:
            try:
                past_run = json.load(f)
            except json.decoder.JSONDecodeError:
                print(f"malformed json file: {path}")
                return False
        return \
            past_run["plan_length"] != -1 or \
            (past_run["plan_length"] == -1 and past_run["max_evaluations"] >= args.max_evaluations)

    if not target_required(
            planning_json_path(args),
            depends_on = args.heuristic if args.heuristic.endswith(".json") else None,
            force=args.force,
            verbose=args.verbose,
            dominated=already_solved_or_failure_with_larger_resources):
        return

    problem = _parse(args.domain, args.problem)
    task = _ground(problem,
                   remove_statics_from_initial_state = False,
                   remove_irrelevant_operators       = False)

    if args.verbose:
        print("> Loading heuristics --- ",end="")
    h = instantiate_heuristic(args, task)
    if args.verbose:
        print("done!")

    if args.verbose:
        print("--- Planning started ---")
    t1 = time.time()
    plan, evaluated = greedy_best_first_search(task, h, max_evaluations=args.max_evaluations)
    t2 = time.time()
    elapsed = t2-t1

    if plan is None:
        plan_length = -1
        evaluated = args.max_evaluations
    else:
        plan_length = len(plan)

    if args.verbose:
        print("--- Planning finished ---")
        print("> Plan:")
        if plan is not None:
            for i, op in enumerate(plan):
                print(f"> Step {i:03d}: {op.name}")
        print("> Plan length:", plan_length)
        print("> Evaluated:", evaluated)
        print("> Elapsed:", elapsed)

    save_results(args, plan, plan_length, evaluated, elapsed)

    pass


if __name__ == '__main__':
    args = parse_arguments()
    try:
        main(args)
    except:
        import pddlsl.stacktrace
        pddlsl.stacktrace.format(arraytypes={numpy.ndarray,torch.Tensor},include_self=False)
