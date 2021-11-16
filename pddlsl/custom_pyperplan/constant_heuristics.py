
from pyperplan.heuristics.heuristic_base import Heuristic
from pddlsl.datamodule import inf, ninf


class ConstantHeuristic(Heuristic):
    def __init__(self, task):
        pass

class ZeroHeuristic(ConstantHeuristic):
    def __call__(self, node):
        return 0.0

class NinfHeuristic(ConstantHeuristic):
    def __call__(self, node):
        return ninf

class InfHeuristic(ConstantHeuristic):
    def __call__(self, node):
        return inf




