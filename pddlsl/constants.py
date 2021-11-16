
# command line used to call fast-downward, using limit.sh
PLANNER_CALL = "planner-scripts/fd-latest-clean -o"

# Search option used to find the optimal plan
OPTIMAL_SEARCH_OPTION = "'--search astar(lmcut())'" # The string needs to be inside ''
OPTIMAL_SEARCH_OPTION_MANDS = "'--search astar(merge_and_shrink(shrink_strategy=shrink_bisimulation(greedy=false),merge_strategy=merge_sccs(order_of_sccs=topological,merge_selector=score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order])),label_reduction=exact(before_shrinking=true,before_merging=false),max_states=50k,threshold_before_merge=1))'" # The string needs to be inside ''

SATISFICING_SEARCH_OPTIONS = {
    "lama-first":"'--alias lama-first'",
}

HEURISTICS = {
    # relaxation-based admissible h
    'lmcut':'lmcut()',
    'hmax':'hmax()',
    # abstraction-based admissible h
    'merge_and_shrink':'merge_and_shrink(shrink_strategy=shrink_bisimulation(greedy=false),merge_strategy=merge_sccs(order_of_sccs=topological,merge_selector=score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order])),label_reduction=exact(before_shrinking=true,before_merging=false),max_states=50k,threshold_before_merge=1)',
    # relaxation-based inadmissible
    'ff':'ff()',
    'add':'add()',
    'cea':'cea()',
    # abstraction-based inadmissible
    # -- none found
    # blind
    'blind':'blind()',
}

COSTS = list(HEURISTICS.keys()) + [
    'opt',
    'zero',
    'ninf',
    'inf',
]

DOMAIN_TO_GENERATOR = {
    'blocksworld' : 'blocksworld-4ops',
    'logistics' : 'logistics',
    'satellite' : 'satellite',
    'ferry'     : 'ferry',
    'gripper'     : 'gripper',
    'miconic'     : 'miconic',
    'parking'     : 'parking',
    'visitall'     : 'visitall',
}

TRAINING_LOGS_DIR = 'training_logs'
TRAINING_CKPT_DIR = 'training_ckpt'
TRAINING_JSON_DIR = 'training_json'

# -- Truncated Gaussian constants --
inf = 1e5                       # large enough value
ninf = -1e5                     # small enough value
EPS = 0.1 # A small constant used to modify the bounds l,u so that the optimal
          # value h* is never equal to one of the bounds
