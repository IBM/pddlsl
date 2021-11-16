from typing import Sequence, Iterator, Literal, Optional, Union, Any
import os
import glob
import json
import numpy as np
import torch
import pytorch_lightning as pl
import random
import lifted_pddl

from .constants import inf, ninf, EPS

from torch.utils.data import Dataset, DataLoader, default_collate

from .relational_state import RelationalState, dense_to_sparse, sparse_to_dense

from strips_hgn.hypergraph.delete_relaxation import DeleteRelaxationHypergraphView
from strips_hgn.planning.strips import _PyperplanSTRIPSProblem
from strips_hgn.features.global_features import NumberOfNodesAndEdgesGlobalFeatureMapper
from strips_hgn.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from strips_hgn.features.node_features import PropositionInStateAndGoal
from strips_hgn.workflows.base_workflow import BaseFeatureMappingWorkflow
from strips_hgn.models.hypergraph_nets_adaptor import merge_hypergraphs_tuple

Path = str

def load_jsons(directory : Path, size : Optional[int] = None) -> Iterator[tuple[dict, Path]]:

    paths = glob.glob(os.path.join(directory,'*.json'))
    random.shuffle(paths)

    if size is None:
        size = len(paths)

    i = 0
    for path in paths:
        if i >= size:
            break
        with open(path, 'r') as f:
            data = json.load(f)
        if data["result"] == "valid":
            yield data, path
            i += 1
    pass

Cost = Optional[Union[int, float]]
Costs = tuple[Cost, ...]

Stage = Literal['train', 'val', 'test', 'predict']

class CommonDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage):
        args = self.args
        if stage == "fit":
            self.train_dataset = CommonDataset(self.create_dataset(load_jsons(os.path.join(args.output, args.domain, args.train_dir), args.train_size),
                                                                   args.train_data_keys))
            self.val_dataset = CommonDataset(self.create_dataset(load_jsons(os.path.join(args.output, args.domain, args.val_dir)),
                                                                 args.test_data_keys))
        elif stage == 'test':
            self.test_dataset = CommonDataset(self.create_dataset(load_jsons(os.path.join(args.output, args.domain, args.test_dir)),
                                                                  args.test_data_keys))
        elif stage == 'predict':
            self.predict_dataset = CommonDataset(self.create_dataset(load_jsons(os.path.join(args.output, args.domain, args.test_dir)),
                                                                     args.test_data_keys))
        else:
            raise Exception(f"Parameter @stage equals '{stage}' but it should be either 'fit', 'test' or 'predict'")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.args.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=0)
    pass


class CommonDataset(Dataset):
    def __init__(self, sample_list : list = []):
        self._dataset = sample_list

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def add_element(self, new_sample):
        self._dataset.append(new_sample)

    def del_element(self, idx):
        if idx < 0 or idx >= len(self):
            raise ValueError("Index out of range")

        del self._dataset[idx]
    pass



def get_costs(json_data, i, data_keys) -> Costs:
    def get_cost(key):
        if key in json_data:
            return json_data[key][i]
        elif key == "zero":
            return 0
        elif key == "inf":
            return inf
        elif key == "ninf":
            return ninf
        else:
            raise "huh?"

    return tuple([ get_cost(key) for key in data_keys ])



NLMState = list[Optional[torch.Tensor]]

def concatenate_nlm_states(state:NLMState, goal:NLMState) -> NLMState:
    return [
        torch.cat((s_t,g_t),dim=-1) if s_t is not None else None
        for s_t, g_t in zip(state, goal)
    ]

def pad_nlm_state(X:NLMState, N:int) -> NLMState:
    # We do not pad the nullary predicates X[0]
    # Each tensor is converted from shape [n*arity, P] to [N*arity, P]

    def _pad(x):
        if x is None:
            return None
        elif x.shape[-1] == 0:
            # Torch bug https://github.com/pytorch/pytorch/issues/71078
            # Padding a tensor with dimensions of size 0 causes an error.
            #
            # example:
            # a = T.zeros((5,5,0))
            # T.nn.functional.pad(a, pad=(0,0)+(1,1), mode="constant", value=1).shape -> error
            #
            # Avoiding the issue with reshaping.
            return x.reshape( (N, )*(x.dim()-1) + (0, ))
        else:
            return torch.nn.functional.pad(x,
                                           pad=(0,0)+(0,N-x.shape[0])*(x.dim()-1),
                                           mode='constant',
                                           value=0)


    padded_tensors = [X[0]] + [ _pad(x) for x in X[1:]]
    return padded_tensors


class NLMCarlosV2DataModule(CommonDataModule):
    def create_dataset(self, json_data_list: Iterator[tuple[dict, Path]], data_keys) \
        -> list[tuple[NLMState, int, Costs]]:
        dataset = []

        # Calculate the maximun number of objects N for all the samples in the dataset
        # This number N will be used for padding the tensors, so that they all have the same shape and
        # can be stacked in a single tensor of shape [B, N*arity, P]
        N = 0

        for json_data, json_path in json_data_list:
            parser = lifted_pddl.Parser()
            parser.parse_domain(os.path.join(os.path.dirname(json_path),"..","domain.pddl"))
            parser.parse_problem(json_path.replace(".json",".pddl"))

            max_pred_arity = len(json_data['goal'])-1 if self.args.breadth == -1 else self.args.breadth
            list_pred_arities = [len(p[1]) for p in parser.predicates] # Obtain a list with the arity of each predicate in the domain
            num_preds_each_arity = [list_pred_arities.count(r) for r in range(max_pred_arity+1)] # List where element r contains the number of predicates with arity r

            num_objs = len(parser.object_names)

            if num_objs > N:
                N = num_objs

            goal = sparse_to_dense(json_data['goal'],num_preds_each_arity,num_objs)
            goal = [torch.tensor(t) if t is not None else None for t in goal]

            for i in range(len(json_data['plan_states'])):
                plan_state = json_data['plan_states'][i]
                state = sparse_to_dense(plan_state,num_preds_each_arity,num_objs)
                state = [torch.tensor(t) if t is not None else None for t in state]

                dataset.append([concatenate_nlm_states(state, goal), num_objs, get_costs(json_data, i, data_keys)])

        # Zero pad all the tensors in the dataset so that they have shape [N*arity, P]
        for i in range(len(dataset)):
            nlm_state = dataset[i][0]
            nlm_state_padded = pad_nlm_state(nlm_state, N)
            dataset[i][0] = nlm_state_padded

        return dataset

    @staticmethod
    def collate_fn(batch : list[tuple[NLMState, int, Costs]]) \
        -> tuple[tuple[NLMState, list[int]], torch.Tensor, ...]:

        batch_tensors = [torch.stack([sample[0][r] for sample in batch], dim=0) if batch[0][0][r] is not None else None \
                         for r in range(len(batch[0][0]))]
        num_objs_list = [sample[1] for sample in batch]

        return (batch_tensors, num_objs_list), *default_collate([sample[2] for sample in batch])
    pass


class RRDataModule(CommonDataModule):
    def create_dataset(self, json_data_list: Iterator[tuple[dict, Path]], data_keys) -> list[tuple[list, Costs]]:
        dataset = []

        for json_data, _ in json_data_list:
            for i in range(len(json_data['plan_states'])):
                dataset.append((json_data['state_features'][i], get_costs(json_data, i, data_keys)))

        return dataset

    @staticmethod
    def collate_fn(batch : list[tuple[list, Costs]]) -> tuple[torch.Tensor, ...]:

        features = torch.tensor([sample[0] for sample in batch])

        return features, *default_collate([sample[1] for sample in batch])
    pass


class HGNDataModule(CommonDataModule):
    def create_dataset (self, json_data_list: Iterator[tuple[dict, Path]], data_keys) -> list[tuple[Any, Costs]]:
        dataset = []

        # Create encoder for encoding a (problem,state) pair as a single hypergraph tuple
        # This hypergraph tuple will then be input to the STRIPS-HGN in order to predict h
        state_hypergraph_encoder = BaseFeatureMappingWorkflow(
            global_feature_mapper_cls=NumberOfNodesAndEdgesGlobalFeatureMapper,
            node_feature_mapper_cls=PropositionInStateAndGoal,
            hyperedge_feature_mapper_cls=ComplexHyperedgeFeatureMapper,
            max_receivers=self.args.max_num_add_effects,
            max_senders=self.args.max_num_preconditions)

        # Iterate over the .json file corresponding to each problem
        for json_data, json_path in json_data_list:
            problem_path = json_path.replace(".json",".pddl")
            domain_path = os.path.join(os.path.dirname(json_path),"..","domain.pddl")
            # Parse problem with Lifted PDDL parser
            parser = lifted_pddl.Parser()
            parser.parse_domain(domain_path)
            parser.parse_problem(problem_path)

            # Obtain hypergraph representation of the problem (actually, of the pyperplan ground task)
            strips_problem = _PyperplanSTRIPSProblem(domain_path, problem_path)
            problem_hypergraph = DeleteRelaxationHypergraphView(strips_problem)

            # Obtain data for the problem initial state
            init_state_atoms = parser.encode_atoms_as_pddl(parser.atoms, 'str')
            init_state_hypergraph = state_hypergraph_encoder._get_input_hypergraphs_tuple(current_state=frozenset(init_state_atoms),
                                                                                          hypergraph=problem_hypergraph)

            dataset.append((init_state_hypergraph, get_costs(json_data, 0, data_keys)))

            # Encode each plan state as a hypergraphtuple
            # To obtain each plan state, successively apply each action in the plan
            for i, action in enumerate(json_data['plan_actions']):
                action_name = action[0]
                action_obj_inds = parser.get_object_indexes(action[1])

                # If the action is not applicable, then there is some bug with the code
                assert parser.is_action_applicable(action_name, action_obj_inds), \
                        f"The current action ({action}) is not applicable at the current state"

                # Apply the action to the current state in order to obtain the next state
                curr_state = parser.get_next_state(action_name, action_obj_inds)
                parser.set_current_state(curr_state) # Important to set the next state in the parser

                # Obtain data for the current state
                # The state ind is always one more than the action ind (i)
                curr_state_atoms = parser.encode_atoms_as_pddl(curr_state, 'str')
                curr_state_hypergraph = state_hypergraph_encoder._get_input_hypergraphs_tuple(current_state=frozenset(curr_state_atoms),
                                                                                              hypergraph=problem_hypergraph)
                dataset.append((curr_state_hypergraph, get_costs(json_data, i+1, data_keys)))

        return dataset

    @staticmethod
    def collate_fn(batch : list[tuple[Any, Costs]]) -> tuple[Any, torch.Tensor, ...]:

        hypergraphs_list = [sample[0] for sample in batch]
        hypergraphs_tuple = merge_hypergraphs_tuple(hypergraphs_list)

        return hypergraphs_tuple, *default_collate([sample[1] for sample in batch])
    pass
