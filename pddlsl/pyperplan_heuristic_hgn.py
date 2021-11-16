from typing import Optional, Type
import numpy as np
import torch

from pyperplan.task import Task
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.search.searchspace import SearchNode

from .heuristic_learner import HeuristicLearner
from .datamodule import HGNDataModule

from strips_hgn.workflows.base_workflow import BaseFeatureMappingWorkflow
from strips_hgn.hypergraph.delete_relaxation import DeleteRelaxationHypergraphView
from strips_hgn.planning.strips import _PyperplanSTRIPSProblem
from strips_hgn.features.global_features import NumberOfNodesAndEdgesGlobalFeatureMapper
from strips_hgn.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from strips_hgn.features.node_features import PropositionInStateAndGoal


class HGNHeuristic(Heuristic):
    def __init__(self,
                 task:Task, domain_path:str, problem_path:str,
                 trained_model     : HeuristicLearner,
                 l_heuristic_class : Type[Heuristic],
                 u_heuristic_class : Type[Heuristic],
                 r_heuristic_class : Type[Heuristic],
                 discretize:bool):
        super().__init__()
        # Encoder for transforming a (problem-state) pair into a hypergraph
        # Note: we use the default global, node and hyperedge feature mappers, since these have been
        # the ones used to create the datasets the model was trained on
        self._state_encoder = BaseFeatureMappingWorkflow(
            global_feature_mapper_cls=NumberOfNodesAndEdgesGlobalFeatureMapper,
            node_feature_mapper_cls=PropositionInStateAndGoal,
            hyperedge_feature_mapper_cls=ComplexHyperedgeFeatureMapper,
            max_receivers=trained_model.hparams.max_num_add_effects,
            max_senders=trained_model.hparams.max_num_preconditions)

        strips_problem = _PyperplanSTRIPSProblem(domain_path, problem_path)
        self._problem_hypergraph = DeleteRelaxationHypergraphView(strips_problem)

        self._l_heuristic = l_heuristic_class(task)
        self._u_heuristic = u_heuristic_class(task)
        self._r_heuristic = r_heuristic_class(task)
        self._trained_model = trained_model
        trained_model.eval()
        self.discretize = discretize
        pass

    def __call__(self, node:SearchNode):
        with torch.no_grad():   # without this, .forward generates gradients WHICH ARE NOT FREED, causing memory leak
            state_hypergraph = self._state_encoder._get_input_hypergraphs_tuple(
                current_state=node.state, hypergraph=self._problem_hypergraph)

            l = self._l_heuristic(node)
            u = self._u_heuristic(node)
            r = self._r_heuristic(node)

            batch = HGNDataModule.collate_fn([(state_hypergraph, (0.0, l, u, r))])

            results = self._trained_model.forward(batch, metrics=False)

        if self.discretize:
            return results["prediction"]["heuristic_int"]
        else:
            return results["prediction"]["heuristic"]
