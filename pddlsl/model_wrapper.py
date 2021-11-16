"""
This module serves as a wrapper around a particular Pytorch model using for predicting mu and sigma (right now, NLM, RR or STRIPS-HGN)
The wrapped model is used by the corresponding HeuristicLearner class to train the model using a truncated or gaussian distribution.
"""

import torch
import argparse
import os
from copy import deepcopy
from typing import Optional, Union

from .constants import inf, ninf, EPS

import lifted_pddl

# Abstract class for Model Wrapper
class ModelWrapper(torch.nn.Module):

    # Sigma (std) value to predict when sigma is not learned
    default_sigma = 0.5**(-0.5)

    def __init__(self, args:argparse.Namespace):
        super().__init__()

        # We store a copy of @args as a dictionary
        self.args = self._get_args_dict(args)

        if self.args['sigma'] == "learn":
            self.predict_sigma = True
        elif self.args['sigma'] == "fixed":
            self.predict_sigma = False
        else:
            raise 'huh?'

        self.l_as_input = self.args['l_as_input']


    @staticmethod
    def _get_args_dict(args:Union[argparse.Namespace, dict]):
        "Auxiliary method for copy and representing args as a dictionary (instead of argparse.Namespace)"
        args_dict = deepcopy(args) if type(args) == dict else deepcopy(vars(args))
        return args_dict

    @staticmethod
    def _clip_sigma(val, min=0.1, max=10):
        """
        Method for clipping the sigma value predicted by the model to some range [min,max] (approximately)
        To do so, we apply the function sigmoid(val)*max + min
        """
        return torch.sigmoid(val)*max + min

    def forward(self, model_specific_input, lb):
        pass


class NLMCommonWrapper(ModelWrapper):
    @classmethod
    def add_parser_arguments(cls, parser:argparse.ArgumentParser):
        parser.add_argument('--breadth', default=3, type=int, help="Arities to expand to in the hidden layers")
        parser.add_argument('--depth', default=5, type=int, help="Number of layers")
        parser.add_argument('--hidden-features', default=8, type=int, help="Feature size for the hidden layers")
        parser.add_argument('--mlp-hidden-features', default=0, type=int, help="Each element-wise NN can be a linear layer or a two-layer MLP. If positive, it uses a two-layer MLP with this hidden size.")

        parser.add_argument('--residual', choices=[None, "all", "input"], help="If specified, each layer is concatenated with the input of the first layer, or with the input of the first layer and all the outputs of the previous layers.")
        parser.add_argument('--batchnorm', default=False, type=eval, help="If True, use the batchnorm in the MLP (if each element-wise NN is a two-layer MLP), as well as before the activation except in the last layer.")
        parser.add_argument('--activation', default='sigmoid', choices=["sigmoid","relu"], help="activation function for each layer.")

    @classmethod
    def derive_arguments(cls, args:argparse.Namespace):

        # Assuming scalar outputs, we do not need some higher-arities near the output
        # because reductions are performed one arity at a time.
        # I.e., higher arities cannot not affect the output.
        # To save the compute, we set the number of output features to 0.

        # For example, with maximum arity 3 at the input and 5 layers,
        # the arity of each layer proceeds like 3->3->3->2->1->0 .

        args.hidden_features_list = []
        for d in range(args.depth-1):
            args.hidden_features_list.append([args.hidden_features]*(args.breadth+1))
            for b in range(args.depth - d, args.breadth+1):
                args.hidden_features_list[d][b] = 0

        if args.sigma == "learn":
            args.out_features = [2,0,0,0]
        elif args.sigma == "fixed":
            args.out_features = [1,0,0,0]
        else:
            raise "huh?"

    @classmethod
    def id(cls, args:argparse.Namespace):
        return (
            f"_{args.breadth}_{args.depth}_{args.hidden_features}_{args.mlp_hidden_features}"
            f"_{args.residual}"
            f"_{args.batchnorm}"
            f"_{args.activation}"
        )

from .nlm_carlos_v2 import NLMCarlosV2

class NLMCarlosV2Wrapper(NLMCommonWrapper):
    def __init__(self, args):
        super().__init__(args)

        self.model = NLMCarlosV2(self.args['hidden_features_list'],
                                 self.args['out_features'],
                                 self.args['mlp_hidden_features'],
                                 self.args['residual'],
                                 self.args['exclude_self'],
                                 self.args['batchnorm'],
                                 self.args['activation'],
                                 self.args['mask_value'])


    @classmethod
    def add_parser_arguments(cls, parser:argparse.ArgumentParser):
        super().add_parser_arguments(parser)
        parser.set_defaults(model="NLMCarlosV2")
        parser.add_argument('--exclude-self', default=False, type=eval, help="If True (default), the result of NLM expansion (2-or higher arities) omits argument combinations with duplicate arguments.")
        parser.add_argument('--mask-value', default=ninf, type=float)

    @classmethod
    def id(cls, args:argparse.Namespace):
        return super().id(args) + f"_{args.exclude_self}_{args.mask_value}"

    def forward(self, model_specific_input, lb):
        batch_tensors, num_objs_list = model_specific_input
        tensors_without_None = [t for t in batch_tensors if t is not None]
        B = tensors_without_None[0].shape[0] # batch dimension
        device = tensors_without_None[0].device

        if self.l_as_input:
            with torch.no_grad():
                lb_tensor = torch.full((B, 1), fill_value=lb, dtype=torch.float, device=device)

                # Append the lower bounds to the nullary predicates
                nullary_tensor : Optional[torch.Tensor] = batch_tensors[0]
                if nullary_tensor is None:
                    new_nullary_tensor = lb_tensor
                else:
                    new_nullary_tensor = torch.cat((nullary_tensor, lb_tensor), dim=-1)

                batch_tensors[0] = new_nullary_tensor

        model_output = self.model(batch_tensors, num_objs_list)

        nullary_output = model_output[0] # Obtain the nullary predicates

        mu = nullary_output[:,0].flatten()
        sigma = self._clip_sigma(nullary_output[:,1].flatten()) if self.predict_sigma else torch.full_like(mu, self.default_sigma)

        return mu, sigma


class RRWrapper(ModelWrapper):
    def __init__(self, args:argparse.Namespace):
        super().__init__(args)

        self.model = torch.nn.LazyLinear(self.args['num_output_features'])

    @classmethod
    def add_parser_arguments(cls, parser:argparse.ArgumentParser):
        parser.set_defaults(model="RR")

    @classmethod
    def derive_arguments(cls, args:argparse.Namespace):
        if args.sigma == "learn":
            args.num_output_features = 2
        elif args.sigma == "fixed":
            args.num_output_features = 1
        else:
            raise "huh?"

    @classmethod
    def id(cls, args:argparse.Namespace):
        # The RR model has no model-specific id, so we return the empty string
        return ""

    def forward(self, model_specific_input, lb):
        features_tensor = model_specific_input

        if self.l_as_input:
            features_tensor = torch.cat((features_tensor, lb), dim=1)

        model_output = self.model(features_tensor)

        mu = model_output[:,0]
        if self.predict_sigma:
            sigma = self._clip_sigma(model_output[:,1])
        else:
            sigma = torch.full_like(mu, self.default_sigma)

        return mu, sigma


from strips_hgn.features.global_features import NumberOfNodesAndEdgesGlobalFeatureMapper
from strips_hgn.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from strips_hgn.features.node_features import PropositionInStateAndGoal
from hypergraph_nets.models import EncodeProcessDecode
from hypergraph_nets.hypergraphs import HypergraphsTuple

class HGNWrapper(ModelWrapper):
    def __init__(self, args:argparse.Namespace):
        super().__init__(args)

        self.model = EncodeProcessDecode(
                        receiver_k = self.args['max_num_add_effects'],
                        sender_k = self.args['max_num_preconditions'],
                        hidden_size = self.args['hidden_size'],
                        edge_input_size=self.args['edge_input_size'],
                        node_input_size=self.args['node_input_size'],
                        global_input_size=self.args['global_input_size'],
                        global_output_size=self.args['global_output_size']
                    )

    @classmethod
    def add_parser_arguments(cls, parser:argparse.ArgumentParser):
        parser.set_defaults(model="HGN")
        parser.add_argument('--num-recursion-steps', type=int, default=3,)
        parser.add_argument('--max-num-add-effects', type=int, default=3,)
        parser.add_argument('--max-num-preconditions', type=int, default=7,)
        parser.add_argument('--hidden-size', type=int, default=32,)
        parser.add_argument('--edge-input-size', type=int, default=ComplexHyperedgeFeatureMapper.input_size(),)
        parser.add_argument('--node-input-size', type=int, default=PropositionInStateAndGoal.input_size(),)
        parser.add_argument('--global-input-size', type=int, default=NumberOfNodesAndEdgesGlobalFeatureMapper.input_size())

    @classmethod
    def derive_arguments(cls, args:argparse.Namespace):
        assert not args.l_as_input, "Currently, the STRIPS-HGN model is incompatible with the 'l_as_input' argument"

        if args.sigma == "learn":
            args.global_output_size = 2
        elif args.sigma == "fixed":
            args.global_output_size = 1
        else:
            raise "huh?"

    @classmethod
    def id(cls, args:argparse.Namespace):
        model_id = (
            f"_{args.num_recursion_steps}_{args.max_num_add_effects}_{args.max_num_preconditions}"
            f"_{args.hidden_size}_{args.edge_input_size}_{args.node_input_size}_{args.global_input_size}"
        )

        return model_id

    def forward(self, model_specific_input, lb):
        hypergraphs_tuple : HypergraphsTuple = model_specific_input

        # Forward pass
        # Since pred_mode=False, we obtain the intermediate outputs
        # model_output has shape (num_steps, num_graphs, global_size)
        model_output = \
            torch.stack(
                self.model(hypergraphs_tuple,
                           steps=self.args['num_recursion_steps'],
                           # When pred_mode=True, it returns only the last step of num_recursion_steps
                           # When pred_mode=False, it returns all steps.
                           # From internal testing, there seems no need for optimizing the output of all steps.
                           pred_mode=True),
                dim=0)

        mu = model_output[:,:,0]
        if self.predict_sigma:
            sigma = self._clip_sigma(model_output[:,:,1])
        else:
            sigma = torch.full_like(mu, self.default_sigma)

        return mu, sigma
