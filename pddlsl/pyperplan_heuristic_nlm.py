from typing import Optional, Type
import numpy as np
import torch

from pyperplan.task import Task
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.search.searchspace import SearchNode

from .heuristic_learner import HeuristicLearner
from .datamodule import NLMCarlosV2DataModule

from .relational_state import RelationalState
import lifted_pddl


class NLMCarlosV1Heuristic(Heuristic):
    def __init__(self,
                 task:Task, domain_path:str, problem_path:str,
                 trained_model     : HeuristicLearner,
                 l_heuristic_class : Type[Heuristic],
                 u_heuristic_class : Type[Heuristic],
                 r_heuristic_class : Type[Heuristic],
                 discretize:bool):
        super().__init__()

        self._parser = lifted_pddl.Parser()
        self._parser.parse_domain(domain_path)
        self._parser.parse_problem(problem_path)

        self._goal = RelationalState(self._parser.types, self._parser.type_hierarchy,
                                     self._parser.predicates, self._parser.object_types,
                                     # Represent the goal atoms like ('on', (2,0))
                                     set([g[1:] for g in self._parser.goals]))

        self._l_heuristic = l_heuristic_class(task)
        self._u_heuristic = u_heuristic_class(task)
        self._r_heuristic = r_heuristic_class(task)
        self._trained_model = trained_model
        trained_model.eval()
        self.discretize = discretize
        pass

    def _node_state_to_rel_state(self, node:SearchNode) -> RelationalState:
        # Transform from '(pointing satellite1 star1)' to tuple('pointing', 'satellite', 'star1')
        state_atoms = [tuple(atom[1:-1].split()) for atom in node.state]
        # Transform from ('pointing', 'satellite', 'star1') to ('pointing', (5,9))
        state_atoms = set([(atom[0], tuple([self._parser.get_object_index(obj_name) \
                                            for obj_name in atom[1:]])) for atom in state_atoms])

        rel_state = RelationalState(self._parser.types, self._parser.type_hierarchy,
                                    self._parser.predicates, self._parser.object_types,
                                    state_atoms)

        return rel_state


    def _from_np_array_to_torch_tensor(self, state_goal_arrays:list[np.array])->list[torch.Tensor]:
        return [
            torch.tensor(arr) if arr is not None else None
            for arr in state_goal_arrays
        ]


    def __call__(self, node:SearchNode):
        with torch.no_grad():   # without this, .forward generates gradients WHICH ARE NOT FREED, causing memory leak
            state : RelationalState = self._node_state_to_rel_state(node)

            # Obtain the (dense) tensor representation associated with the (current state, goal) pair
            state_goal_arrays : list[np.array] = state.atoms_nlm_encoding_with_goal_state(
                self._goal,
                self._trained_model.hparams.breadth,
                add_object_types=False) # We do not add object types as the NLM was trained on state-goal tensors without object types

            state_goal_tensors = self._from_np_array_to_torch_tensor(state_goal_arrays)

            l = self._l_heuristic(node)
            u = self._u_heuristic(node)
            r = self._r_heuristic(node)

            batch = NLMCarlosV1DataModule.collate_fn([(state_goal_tensors, state.num_objects, (0.0, l, u, r))])

            results = self._trained_model.forward(batch, metrics=False)

        if self.discretize:
            return results["prediction"]["heuristic_int"]
        else:
            return results["prediction"]["heuristic"]

class NLMCarlosV2Heuristic(NLMCarlosV1Heuristic):
    def __call__(self, node:SearchNode):
        with torch.no_grad():   # without this, .forward generates gradients WHICH ARE NOT FREED, causing memory leak
            state : RelationalState = self._node_state_to_rel_state(node)

            # Obtain the (dense) tensor representation associated with the (current state, goal) pair
            state_goal_arrays : list[np.array] = state.atoms_nlm_encoding_with_goal_state(
                self._goal,
                self._trained_model.hparams.breadth,
                add_object_types=False) # We do not add object types as the NLM was trained on state-goal tensors without object types

            state_goal_tensors = self._from_np_array_to_torch_tensor(state_goal_arrays)

            l = self._l_heuristic(node)
            u = self._u_heuristic(node)
            r = self._r_heuristic(node)

            batch = NLMCarlosV2DataModule.collate_fn([(state_goal_tensors, state.num_objects, (0.0, l, u, r))])

            results = self._trained_model.forward(batch, metrics=False)

        if self.discretize:
            return results["prediction"]["heuristic_int"]
        else:
            return results["prediction"]["heuristic"]
