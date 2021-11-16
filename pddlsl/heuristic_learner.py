"""
This module implements functionality for training a model (right now, either an NLM, RR or STRIPS-HGN) to predict a heuristic, using
either a gaussian or truncated gaussian
The HeuristicLearner class acts as the parent class
SupervisedLearner implements functionality for training a model when the ground-truth h* is known
"""

from typing import Literal
import argparse
from collections import defaultdict
import torch
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal # Gaussian distribution
from stable_trunc_gaussian import TruncatedGaussian # Truncated gaussian distribution
#from .truncated_gaussian import ParallelTruncatedGaussian as TruncatedGaussian
from torch.distributions.kl import kl_divergence

from .constants import inf, ninf, EPS, COSTS

def is_invalid(tensor):
    """
    Auxiliary function for debugging
    Returns True if the tensor passed as a parameter contains one or more invalid values
    A value is invalid if it is NaN, -inf or inf
    """
    return torch.any(torch.logical_or(torch.isinf(tensor), torch.isnan(tensor)))


def OpenBoundTruncatedGaussian(mu, sigma, a, b):
    return TruncatedGaussian(mu, sigma, a-EPS, b+EPS)


def smart(dist:TruncatedGaussian, mean, flip=False):
    mean_c = mean.ceil()
    mean_f = mean.floor()
    logp_mc = dist.log_prob(torch.clip(mean_c, min=dist.a+EPS, max=dist.b-EPS))
    logp_mf = dist.log_prob(torch.clip(mean_f, min=dist.a+EPS, max=dist.b-EPS))
    if flip:
        use_ceil = (logp_mc < logp_mf)
    else:
        use_ceil = (logp_mc > logp_mf)
    return (use_ceil * mean_c + ~use_ceil * mean_f)


class HeuristicLearner(pl.LightningModule):
    @classmethod
    def id(cls, args:argparse.Namespace):
        "Returns an id string used for saving the results."
        return (
            # truly common training options only.
            # options specifying the detail of the statistical models is inside SupervisedLearner.
            f"{args.domain}_{args.model}_{args.learner}"
            f"_{args.train_size}_{args.lr}_{args.decay}_{args.clip}_{args.batch_size}_{args.seed}"
        )

    @classmethod
    def add_parser_arguments(cls, parser:argparse.ArgumentParser):
        "adds model-dependent command-line options to parse (e.g., --breadth and --depth for NLM)"
        pass

    @classmethod
    def derive_arguments(cls, args:argparse.Namespace):
        "adds model-dependent derived arguments (e.g., args.out_features for NLM). Also, performs some checks"
        pass


    def __init__(self, args:argparse.Namespace):
        super().__init__()

        # This hyperparameter is used to later obtain the global_step of the saved checkpoint with the best val loss
        self.register_parameter('persistent_global_step', Parameter(torch.tensor(0, dtype=int), requires_grad=False))
        self.save_hyperparameters(args)

        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.decay)
        return optimizer

    def forward(self, batch, metrics=True) -> dict:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx=0):
        results = self.forward(batch)
        for k, v in results["metrics"].items():
            self.log(f"t_{k}", v, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, prog_bar=(k=="nll"))

        self.persistent_global_step += 1
        return results["metrics"]["nll"]

    def validation_step(self, batch, batch_idx=0):
        results = self.forward(batch)
        for k, v in results["metrics"].items():
            self.log(f"v_{k}", v, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, prog_bar=(k=="nll"))
        self.log(f"hp_metric", results["metrics"]["nll"], on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, prog_bar=False)
        pass

    def test_step(self, batch, batch_idx=0):
        results = self.forward(batch)
        for k, v in results["metrics"].items():
            self.log(f"T_{k}", v, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, prog_bar=(k=="nll"))
        pass

    def predict_step(self, batch, batch_idx=0):
        results = self.forward(batch, metrics=False)

        return results["prediction"]["heuristic"]


class SupervisedLearner(HeuristicLearner):
    @classmethod
    def id(cls, args:argparse.Namespace):
        "Returns an id string used for saving the results."
        return super().id(args) + (
            f"_{args.dist}"
            f"_{args.target}"
            f"_{args.sigma}"
            f"_{args.l_train}_{args.l_test}"
            f"_{args.u_train}_{args.u_test}"
            f"_{args.res_train}_{args.res_test}"
            f"_{args.l_as_input}"
        )

    @classmethod
    def add_parser_arguments(cls, parser:argparse.ArgumentParser):
        super().add_parser_arguments(parser)
        parser.set_defaults(learner="supervised")
        parser.add_argument('--sigma', default='learn', choices=("learn","fixed"))
        parser.add_argument('--target', default='opt', choices=COSTS)
        parser.add_argument('--l', choices=COSTS, help="If specified, overrides --l-train and --l-test.")
        parser.add_argument('--l-train', default='ninf', choices=COSTS)
        parser.add_argument('--l-test', default='ninf', choices=COSTS)
        parser.add_argument('--u', choices=COSTS, help="If specified, overrides --u-train and --u-test.")
        parser.add_argument('--u-train', default='inf', choices=COSTS)
        parser.add_argument('--u-test', default='inf', choices=COSTS)
        parser.add_argument('--res', choices=COSTS, help="If specified, overrides --res-train and --res-test.")
        parser.add_argument('--res-train', default='zero', choices=COSTS)
        parser.add_argument('--res-test', default='zero', choices=COSTS)
        parser.add_argument('--l-as-input', action="store_true", help="feed the lower bound as an additional nullary input.")
        pass

    @classmethod
    def derive_arguments(cls, args:argparse.Namespace):
        super().derive_arguments(args)
        if args.l is not None:
            args.l_train = args.l_test = args.l
        if args.u is not None:
            args.u_train = args.u_test = args.u
        if args.res is not None:
            args.res_train = args.res_test = args.res
        if args.l_as_input:
            assert args.l_train != "ninf", "--l-as-input requires a non-trivial training lower bound"
            assert args.l_test != "ninf", "--l-as-input requires a non-trivial testing lower bound"

        args.train_data_keys = [
            args.target    ,
            args.l_train   ,
            args.u_train   ,
            args.res_train ,
        ]
        args.test_data_keys = [
            args.target   ,
            args.l_test   ,
            args.u_test   ,
            args.res_test ,
        ]
        pass


    def __init__(self, args:argparse.Namespace):
        super().__init__(args)
        if isinstance(args, dict): # when the weights are loaded, recent lightning converts Namespace to a dict
            args = argparse.Namespace(**args)
        self.model = args.network_cls(args)


    def _extract_results(self, dist, dist_clip, pred_mu, pred_sigma, target, calculate_metrics):
        results = defaultdict(dict)
        mean = dist.mean
        results["prediction"]["mean"] = mean
        mean_clip = dist_clip.mean
        results["prediction"]["mean_clip"] = mean_clip

        if self.hparams.dist == "gaussian":
            results["prediction"]["heuristic"] = mean_clip # note : if clip value is ninf, then this is same as the mean
            results["prediction"]["heuristic_int"] = mean_clip.round()
        elif self.hparams.dist == "truncated":
            results["prediction"]["heuristic"] = mean
            results["prediction"]["heuristic_int"] = smart(dist, mean)
        else:
            raise "huh?"

        if calculate_metrics:
            results["metrics"]["nll"]      = -dist.log_prob(target).mean()
            results["metrics"]["nll_clip"] = -dist_clip.log_prob(target).mean()
            results["metrics"]["mse"]      = (mean - target).square().mean()
            results["metrics"]["mse_clip"] = (mean_clip - target).square().mean()
            results["metrics"]["mse_floor"]      = (mean.floor()      - target).square().mean()
            results["metrics"]["mse_floor_clip"] = (mean_clip.floor() - target).square().mean()
            results["metrics"]["mse_ceil"]       = (mean.ceil()       - target).square().mean()
            results["metrics"]["mse_ceil_clip"]  = (mean_clip.ceil()  - target).square().mean()
            results["metrics"]["mse_round"]      = (mean.round()      - target).square().mean()
            results["metrics"]["mse_round_clip"] = (mean_clip.round() - target).square().mean()

            if self.hparams.dist == "gaussian":
                results["metrics"]["mse_smart"]      = (mean.round()      - target).square().mean()
                results["metrics"]["mse_smart_clip"] = (mean_clip.round() - target).square().mean()
                results["metrics"]["mse_smart2"]      = (mean.round()      - target).square().mean()
                results["metrics"]["mse_smart2_clip"] = (mean_clip.round() - target).square().mean()
            elif self.hparams.dist == "truncated":
                results["metrics"]["mse_smart"]      = (smart(dist, mean)      - target).square().mean()
                results["metrics"]["mse_smart_clip"] = (smart(dist, mean_clip) - target).square().mean()
                results["metrics"]["mse_smart2"]      = (smart(dist, mean, True)      - target).square().mean()
                results["metrics"]["mse_smart2_clip"] = (smart(dist, mean_clip, True) - target).square().mean()
            else:
                raise "huh?"


            results["metrics"]["mu"]      = pred_mu.mean()
            results["metrics"]["sigma"]   = pred_sigma.mean()

        return results


    def forward(self, batch, metrics=True):
        model_specific_input, target, lb, ub, res = batch # The last two batch elements are only used for the ELBO

        delta_mu, sigma = self.model(model_specific_input, lb)

        mu = delta_mu + res # note: res(idual) is always added, but its default value is res == 0
        mu_clip = torch.clip(mu,min=lb,max=ub)

        if self.hparams.dist == "gaussian":
            dist = Normal(mu, sigma)
            dist_clip = Normal(mu_clip, sigma)

        elif self.hparams.dist == "truncated":
            dist = OpenBoundTruncatedGaussian(mu, sigma, lb, ub)
            dist_clip = OpenBoundTruncatedGaussian(mu_clip, sigma, lb, ub)
        else:
            raise "huh?"

        results = self._extract_results(dist, dist_clip, mu, sigma, target, metrics)
        return results

