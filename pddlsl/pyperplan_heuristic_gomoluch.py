from typing import Optional, Type
import numpy as np
import torch

from pyperplan.task import Task
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.search.searchspace import SearchNode, make_root_node

from .heuristic_learner import HeuristicLearner
from .datamodule import RRDataModule

from .custom_pyperplan.relaxation import hFFHeuristic


"""
Function for computing the "planning features" as described in:

    Gomoluch, Pawel, et al. "Towards learning domain-independent planning heuristics." arXiv preprint arXiv:1707.06895 (2017).

This includes
    - Number of unsatisfied goals (hamming distance, goal-count heuristic)
    - Number of operators used in FF's relaxed plan (FF heuristic)
    - Total number of effects ignored in relaxed plan
    - Average number of effects ignored

"""
def compute_gomoluch_features(task, node=None, h_ff_instance=None):
    # If no h_ff instance was passed as a parameter, simply create one from scratch
    if h_ff_instance is None:
        h_ff_instance = hFFHeuristic(task)

    # If no node was passed as a parameter, we create a node out of the task initial state
    if node is None:
        node = make_root_node(task.initial_state)

    # Number of unsatisfied goals
    # It equals the number of goal facts which are not in the initial state
    goal_count_h = len(task.goals - task.initial_state)

    # Compute the relaxed plan obtained by the FF heuristic
    # The FF heuristic is equal to the length of this relaxed plan
    h_ff, ff_plan = h_ff_instance.calc_h_with_plan(node)

    if len(ff_plan) > 0: # Make sure the plan is not empty
        # Compute the number of effects each operator in the relaxed plan
        # ignores. This is equivalent to computing the number of del effects
        # of each operator in the relaxed plan

        # Dictionary which maps names to operators
        task_ops_dir = {op.name:op for op in task.operators}

        # List with the number of effects ignored for each operator in the relaxed plan
        num_ignored_effects_each_op = [len(task_ops_dir[op_name].del_effects) for op_name in ff_plan]

        # Calculate the total and mean number of ignored effects
        total_ignored_effects = np.sum(num_ignored_effects_each_op)
        mean_ignored_effects = np.mean(num_ignored_effects_each_op)

    else: # If the plan is empty, then the total and average number of ignored effects is both 0
        total_ignored_effects, mean_ignored_effects = 0, 0.0

    return goal_count_h, h_ff, int(total_ignored_effects), float(mean_ignored_effects)


class LinearHeuristic(Heuristic):
    def __init__(self,
                 task:Task, domain_path:str, problem_path:str,
                 trained_model     : HeuristicLearner,
                 l_heuristic_class : Type[Heuristic],
                 u_heuristic_class : Type[Heuristic],
                 r_heuristic_class : Type[Heuristic],
                 discretize:bool):
        super().__init__()
        self._task = task
        self._l_heuristic = l_heuristic_class(task)
        self._u_heuristic = u_heuristic_class(task)
        self._r_heuristic = r_heuristic_class(task)
        self._ff_heuristic = hFFHeuristic(task)
        self._trained_model = trained_model
        trained_model.eval()
        self.discretize = discretize
        pass

    def __call__(self, node:SearchNode):
        with torch.no_grad():   # without this, .forward generates gradients WHICH ARE NOT FREED, causing memory leak
            features = compute_gomoluch_features(self._task, node, self._ff_heuristic)

            l = self._l_heuristic(node)
            u = self._u_heuristic(node)
            r = self._r_heuristic(node)

            batch = RRDataModule.collate_fn([(features, (0.0, l, u, r))])

            results = self._trained_model.forward(batch, metrics=False)

        if self.discretize:
            return results["prediction"]["heuristic_int"]
        else:
            return results["prediction"]["heuristic"]
