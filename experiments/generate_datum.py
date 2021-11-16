#!/usr/bin/env python

"""
For usage, run `generate_datum.py -h` or read README.org
"""

import argparse
from time import time
import subprocess
import os
import errno
import re
import json
import glob
from typing import Optional, Literal
from collections import defaultdict

from lifted_pddl import Parser

from util import *
from pddlsl.constants import (
    PLANNER_CALL,
    OPTIMAL_SEARCH_OPTION,
    SATISFICING_SEARCH_OPTIONS,
    HEURISTICS,
    DOMAIN_TO_GENERATOR,
)

from pddlsl.relational_state import RelationalState, dense_to_sparse, sparse_to_dense
from pddlsl.pyperplan_heuristic_gomoluch import compute_gomoluch_features
from pddlsl.custom_pyperplan.relaxation import hFFHeuristic

from pyperplan.pddl.parser import Parser as PyParser
from pyperplan.grounding import ground
from pyperplan.search.searchspace import make_root_node





def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script generates a single problem and obtains all its related information.')

    parser.add_argument('-f','--force', action="store_true",
                        help="force overwriting an existing output.")
    parser.add_argument('-o', '--output', default="data",
                        help="root directory for storing the generated problem and data.")
    parser.add_argument('-r', '--retry', type=int, default=20,
                        help="""number of retries in case plan generation fails (due to timeouts or trivial problems).
                                If this number is reached and no valid plan has been generated, we raise an Exception""")
    parser.add_argument('-t', '--time-limit', type=int, default=300,
                        help="""time limit (in seconds) for the optimal planner (limit.sh). The time allocated for the satisficing
                                planner and heuristics calculation is 1/20 of this quantity.""")
    parser.add_argument('-m', '--mem-limit', type=int, default=8388608,
                        help="memory limit (in kB) for the planner (limit.sh).")
    parser.add_argument('--json', action="store_true",
                        help="If present, compute the metrics for training and evaluation.")

    parser.add_argument('stage',
                        help=(
                            "A subdirectory name under the domain directory. "
                            "For example, with 'train', it stores the results in 'data/blocksworld/train'. "
                            "Usual values are train, val, or test, but other names are also permitted. "))
    parser.add_argument('domain', choices=DOMAIN_TO_GENERATOR.keys(), help="domain name")
    parser.add_argument('seed', type=int,
                        help="seed (a random one is used if empty)")
    parser.add_argument('args', nargs='*',
                        help="all generator arguments, separated by whitespaces")

    return parser.parse_args()


def solve_problem(domain_path, problem_path, search_option, time_limit, memory_limit) -> Optional[tuple]:

    # Given the lines (given by file.readlines()) of a plan file (.plan, .plan.i), it returns whether
    # the plan is valid or not
    def is_plan_valid(lines):
        # A plan is valid if the last line corresponds to a comment (starts with character ';')
        return lines[-1][0] == ';'

    # Given an action encoded like "(stack b1 b2)", it returns it encoded like ("stack", ("b1", "b2"))
    def process_action(line):
        line = line[1:-1] # Remove '(' and ')'
        words = line.split()
        action_name = words[0]
        action_params = tuple(words[1:])

        return (action_name, action_params)


    planner_call_with_options = f"{PLANNER_CALL} {search_option}"
    subprocess.run(['planner-scripts/limit.sh', '-t', str(time_limit), '-m', str(memory_limit), '--', planner_call_with_options, '--', problem_path, domain_path],
                    shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # NEW
    # Obtain all the plan files (i.e., files in the form problem_name.plan*)
    # Some planners only obtain one plan, i.e., a single file problem_name.plan
    # Other planners (like LAMA) may obtain several plans like problem_name.plan.1, problem_name.plan.2,...
    plan_base_path = problem_path.replace('.pddl', '.plan')
    plan_paths = glob.glob(plan_base_path + '*')

    plan = None

    if len(plan_paths) > 0:
        # Read all the plans and get the shortest one (that is valid)
        for plan_path in plan_paths:
            with open(plan_path, 'r') as f:
                lines = f.readlines()

                # Check whether the plan is valid (it could happen that the planner is terminated while writing the plan)
                if is_plan_valid(lines):
                    # Parse the current plan
                    curr_plan = tuple([process_action(line.strip()) for line in lines if line.strip()[0] != ';']) # Ignore comments

                    # If the current plan is better than the one found so far, save it
                    if plan is None or len(curr_plan) < len(plan):
                        plan = curr_plan

            os.remove(plan_path) # Remove the .plan file

    return plan


def generate_problem(seed, args) -> tuple[str,str]:
    return subprocess.run(['pddl-generators',
                           '-d', os.path.join(args.output, args.domain, args.stage),
                           '-s', str(seed),
                           DOMAIN_TO_GENERATOR[args.domain],
                           '--', *args.args],
                          shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8') \
                     .split('\n')[:2] # The last element of the list is the empty string ''


def compute_optimal_plan(domain_path, problem_path, time_limit, memory_limit):
    return solve_problem(domain_path, problem_path, OPTIMAL_SEARCH_OPTION, time_limit, memory_limit)


PlanningResult = Literal["valid", "failed", "trivial"]
def validity(problem_path, plan) -> PlanningResult:
    problem_path_without_extension = os.path.splitext(problem_path)[0]
    log_path = problem_path_without_extension + '.log'
    err_path = problem_path_without_extension + '.err'

    # Check if the problem could not be solved
    if plan is None or not os.path.exists(err_path) or not os.path.exists(log_path):
        problem_solved = False
    else:
        with open(err_path, 'r') as f:
            # If the .err file contains something, then there was an error and we delete the problem
            problem_solved =  (f.read().strip(' \n') == '')

    if problem_solved:
        # check if its plan length was 0
        # Use regex to find the plan length in the .log file
        # Note: I think this could also be done by reading the .plan file, but I prefer to
        #       to do it this way just in case
        with open(log_path, 'r') as f:
            plan_length = int(re.search(r'Plan length: (\d+) step\(s\)\.', f.read()).group(1))

            trivial_problem = (plan_length == 0)

    ensure_file_removed(err_path)
    ensure_file_removed(log_path)

    if problem_solved:
        if trivial_problem:
            result = "trivial"
        else:
            result = "valid"
    else:
        result = "failed"

    if result != "valid":
        ensure_file_removed(problem_path)

    return result


def parse_problem_and_plan(domain_path, problem_path, plan_actions):
    """
    Parse the problem (i.e., encode its information in the lifted_pddl parser) and obtain the states
    associated with the plan actions. The first state corresponds to the problem initial state,
    whereas the last state is the goal state (obtained by executing the plan's last action)
    NOTE: the state associated with the returned parser (i.e., parser.atoms) does NOT correspond
          to the problem initial state but its goal. From this point onwards, only use plan_states
          to obtain the states associated with the plan (also, plan_states[0] is the problem initial state)
    """
    # Parse the problem file
    parser = Parser()
    parser.parse_domain(domain_path)
    parser.parse_problem(problem_path)

    # The first state of the plan equals the problem initial state
    plan_states = [parser.atoms]

    # Get a list of the states obtained by executing the plan actions
    for action_name, action_params in plan_actions:
        # Encode the action parameters as object indexes, instead of names
        action_param_inds = parser.get_object_indexes(action_params)

        # Obtain the next state
        next_state = parser.get_next_state(action_name, action_param_inds, check_action_applicability=True)

        # If the next state is the same as the current state, that means the action was not applicable
        if next_state == parser.atoms:
            raise Exception(f"The plan contains a non-applicable action!\n> Current state: \
                             {parser.atoms}\n> Action: {(action_name, action_params)}")

        # Append the new state to the plan states and assign it to the problem
        plan_states.append(next_state)
        parser.set_current_state(next_state)

    return parser, tuple(plan_states)


def compute_heuristics_on_init(domain_path, problem_path, time_limit, memory_limit) -> dict[str,int]:
    # eager_greedy([h1,h2...], bound=0) simply calculates the heuristics on the initial state and exists the search (due to bound=0)
    planner_call_heuristic = f"{PLANNER_CALL} '--search eager_greedy([{','.join(HEURISTICS.values())}],bound=0)'"

    subprocess.run(['planner-scripts/limit.sh', '-t', str(time_limit), '-m', str(memory_limit), '--', planner_call_heuristic, '--', problem_path, domain_path],
                    shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    log_path = problem_path.replace('.pddl', '.log')
    if not os.path.exists(log_path):
        raise Exception("The log file does not exist, so the heuristics could not be computed!!")

    with open(log_path, 'r') as f:
        planner_output = f.read()

    # note: for merge_and_shrink, options are spelled out.
    # [t=1.1288s, 52716 KB] Initial heuristic value for merge_and_shrink(shrink_strategy = shrink_bisimulation(greedy = false), merge_strategy = merge_sccs(order_of_sccs = topological, merge_selector = score_based_filtering(scoring_functions = list(goal_relevance, dfp, total_order))), label_reduction = exact(before_shrinking = true, before_merging = false), max_states = 50k, threshold_before_merge = 1): 7
    # [t=1.12881s, 52716 KB] Initial heuristic value for ff: 11
    parse_str = r"Initial heuristic value for {}(?:[^:]*): ([0-9]+)"

    results = dict()
    for h in HEURISTICS.keys():
        match = re.search(parse_str.format(h), planner_output)
        if match is None:
            raise Exception(f"Error when computing heuristic {h}: \n{planner_output}")
        results[h] = int(match.group(1))

    ensure_file_removed(problem_path.replace('.pddl', '.plan'))
    ensure_file_removed(problem_path.replace('.pddl', '.err'))
    ensure_file_removed(problem_path.replace('.pddl', '.log'))
    ensure_file_removed(problem_path.replace('.pddl', '.negative'))
    return results


def compute_ub_lb_hstar_on_path(parsed_problem, problem_path, domain_path, plan_states, time_limit, memory_limit) -> defaultdict[str,list[int]]:
    results : defaultdict[str,list[int]] = defaultdict(list)
    results["opt"] = list(range(len(plan_states)-1,-1,-1))
    for i, curr_plan_state in enumerate(plan_states):
        parsed_problem.set_current_state(curr_plan_state)

        curr_problem_path = append_to_name(problem_path, f"_temp_{i}")
        with open(curr_problem_path, 'w+') as f:
            f.write(parsed_problem.dump_pddl_problem(f"temp_problem_{i}"))

        for name, option in SATISFICING_SEARCH_OPTIONS.items():
            plan = solve_problem(domain_path, curr_problem_path, option, time_limit, memory_limit)
            if plan is None:
                raise Exception("The problem could not be solved with the satisficing planner!!")
            results[name].append(len(plan))
            ensure_file_removed(curr_problem_path.replace('.pddl', '.err'))
            ensure_file_removed(curr_problem_path.replace('.pddl', '.log'))

        for k, v in compute_heuristics_on_init(domain_path, curr_problem_path, time_limit, memory_limit).items():
            results[k].append(v)

        os.remove(curr_problem_path)
    return results


def compute_gomoluch_features_on_path(parsed_problem, problem_path, domain_path, plan_states):
    planning_features_list = []

    # For each state in @plan_states, calculate the planning features
    for i, curr_plan_state in enumerate(plan_states):
        # Assign the current plan state to the problem
        parsed_problem.set_current_state(curr_plan_state)

        # Save the current problem to disk
        curr_problem_path = append_to_name(problem_path, f"_temp_{i}")
        with open(curr_problem_path, 'w+') as f:
            f.write(parsed_problem.dump_pddl_problem(f"temp_problem_{i}"))

        # Create a pyperplan task associated with the current problem
        py_parser = PyParser(domain_path, curr_problem_path)
        py_domain = py_parser.parse_domain()
        py_problem = py_parser.parse_problem(py_domain)
        task = ground(py_problem, remove_statics_from_initial_state=False,
                    remove_irrelevant_operators=False)

        # Compute the planning features for the initial state of the task
        # (which corresponds to curr_plan_state)
        curr_planning_features = compute_gomoluch_features(task)
        planning_features_list.append(curr_planning_features)

        ensure_file_removed(curr_problem_path)

    return planning_features_list


def save_results(args, domain_path, problem_path, parsed_problem, plan_states, plan_actions, ub_lb_hstar, features_list):
    """
    Store all the problem (and plan) data in a compact form, in JSON, with the following properties:

    - initial_state : initial state of the problem, as a lis JSON does not support sets)
    - goal: goal of the problem, as a list of tt of atoms (note:he atoms that need to be true
    - plan_states : list with the states of the optimal plan. The first plan state equals the problem initial state
    - plan_actions : list with the actions of the optimal plan
    - features_list: list with the planning features used by the Ridge Regression model for each plan state

    The JSON file shares the same path as the problem (@problem_path) but has the '.json' extension instead of '.pddl'
    Note: we store the initial state, goal and plan states using a sparse tensor encoding, instead of PDDL.
    """
    data = dict()

    data['timelimit'] = args.time_limit
    data['memlimit'] = args.mem_limit
    data['result'] = "valid"
    data['plan_actions'] = plan_actions
    data.update(ub_lb_hstar)

    data['state_features'] = features_list

    # Shortcut names used to create the instances of RelationalState
    types, type_hierarchy, predicates, object_types = parsed_problem.types, parsed_problem.type_hierarchy, \
                                                      parsed_problem.predicates, parsed_problem.object_types

    # <Problem initial state>

    # The initial state atoms are not contained in @parsed_problem, but in @plan_states[0]
    init_rel_state = RelationalState(types, type_hierarchy, predicates, object_types, plan_states[0])

    # Encode the init state atoms as sparse tensors
    init_state_tensors, _, _ = dense_to_sparse(init_rel_state.atoms_nlm_encoding())
    data['initial_state'] = init_state_tensors

    # <Problem goal>

    # Encode the problem goal as the list of atoms that need to be true
    # goal[1:] -> Skip the "True" element in goal[0]
    goal_atoms = set([goal[1:] for goal in parsed_problem.goals])
    goal_rel_state = RelationalState(types, type_hierarchy, predicates, object_types, goal_atoms)

    # Encode the goal atoms as sparse tensors
    goal_atoms_tensors, _, _ = dense_to_sparse(goal_rel_state.atoms_nlm_encoding())
    data['goal'] = goal_atoms_tensors

    # <Plan states>

    # For each state in the plan, obtain its state atoms encoded as sparse tensors
    list_plan_states_tensors = []

    for curr_state in plan_states:
        curr_rel_state = RelationalState(types, type_hierarchy, predicates, object_types, curr_state)
        curr_state_tensors, _, _ = dense_to_sparse(curr_rel_state.atoms_nlm_encoding())
        list_plan_states_tensors.append(curr_state_tensors)

    data['plan_states']=list_plan_states_tensors


    save_json(data, problem_path.replace('.pddl', '.json'))
    pass


def save_failure(json_path, args, result):
    data = dict()

    data['timelimit'] = args.time_limit
    data['memlimit'] = args.mem_limit
    data['result'] = result

    save_json(data, json_path)
    pass


def main(args):

    result : PlanningResult = "failed"
    assert args.retry < 1000
    for i in range(args.retry):
        seed = args.seed * 1000 + i

        problem_path, domain_path = generate_problem(seed, args)
        if not args.json:
            print("> Not generating the training data metrics and the json file. To generate, give --json option")
            return
        json_path = problem_path.replace('.pddl', '.json')
        print("\n------ Generating problem data ------")
        print("> Problem:", problem_path)
        print("> Domain:", domain_path)
        print("> Seed:", seed)
        print("> Generator parameters:", *args.args)
        print("> Output:", json_path)
        if (not args.force) and os.path.exists(json_path):
            with open(json_path) as f:
                past_run = json.load(f)
                result = past_run["result"]
            if result == "valid":
                print(f"Data for problem {os.path.basename(problem_path)} exists, exiting.")
                return
            elif result == "trivial":
                print(f"Problem {os.path.basename(problem_path)} is trivial, retrying another seed.")
                continue
            elif past_run["timelimit"] < args.time_limit or past_run["memlimit"] < args.mem_limit:
                print(f"There is a failed attempt to solve {os.path.basename(problem_path)} with a smaller compute; attempting to solve it again.")
                print(f'(time, mem): old=({args.time_limit}, {args.mem_limit}), new=({past_run["timelimit"]}, {past_run["memlimit"]})')
            else:
                print(f"There is a failed attempt to solve {os.path.basename(problem_path)} with a larger compute; hopeless, retrying another seed.")
                print(f'(time, mem): old=({args.time_limit}, {args.mem_limit}), new=({past_run["timelimit"]}, {past_run["memlimit"]})')
                continue

        plan = compute_optimal_plan(domain_path, problem_path, args.time_limit, args.mem_limit)
        result = validity(problem_path, plan)
        print("> result:", result)
        if result == "valid":
            break
        else:
            save_failure(json_path,args,result)

    if result != "valid":
        print(f"Could not generate a valid problem with the given maximum number of retries ({args.retry})\nProblem path: {os.path.basename(problem_path)}")
        return

    parsed_problem, plan_states = parse_problem_and_plan(domain_path, problem_path, plan)

    # NOTE: @parsed_problem is modified by this method!!
    # The maximum planning time for calculating the satisficing plans and heuristics is the twentieth
    # part of the time for the optimal planner. We do this because, otherwise, LAMA can take a very long time
    ub_lb_hstar = compute_ub_lb_hstar_on_path(parsed_problem, problem_path,
                                              domain_path, plan_states, int(args.time_limit / 20), args.mem_limit)

    features_list = compute_gomoluch_features_on_path(parsed_problem, problem_path, domain_path, plan_states)

    save_results(args, domain_path, problem_path, parsed_problem, plan_states, plan, ub_lb_hstar, features_list)

    pass


if __name__ == '__main__':
    args = parse_arguments()
    try:
        main(args)
    except:
        import pddlsl.stacktrace
        pddlsl.stacktrace.format()
