#!/bin/bash -e

# echo the command line calls to run.
tasks (){
    parallel echo python generate_datum.py -o data --json        train blocksworld ::: {1..38} ::: {5..16} # 38x12 = 456
    parallel echo python generate_datum.py -o data --json        val   blocksworld ::: {1..11} ::: {5..16} # 11x12 = 132
    parallel echo python generate_datum.py -o data --json        test  blocksworld ::: {1..11} ::: {5..16}

    parallel echo python generate_datum.py -o data --json        train logistics ::: {1..4} ::: {1..4} ::: {1..6} ::: {1..4} ::: {3..6} ::: 5 # 4x4x6x4x4x1=1536
    parallel echo python generate_datum.py -o data --json        val   logistics ::: 1      ::: {1..4} ::: {1..6} ::: {1..4} ::: {3..6} ::: 5 # 4x6x4x4x1=384
    parallel echo python generate_datum.py -o data --json        test  logistics ::: 1      ::: {1..4} ::: {1..6} ::: {1..4} ::: {3..6} ::: 5

    parallel echo python generate_datum.py -o data --json        train satellite ::: {1..4} ::: {1..6} ::: 3 ::: {3..6} ::: 7 10 15 ::: {3..6} # 4x6x4x3x4=1152
    parallel echo python generate_datum.py -o data --json        val   satellite ::: 1      ::: {1..6} ::: 3 ::: {3..6} ::: 7 10 15 ::: {3..6} # 6x4x3x4=288
    parallel echo python generate_datum.py -o data --json        test  satellite ::: 1      ::: {1..6} ::: 3 ::: {3..6} ::: 7 10 15 ::: {3..6}

    parallel echo python generate_datum.py -o data --json        train ferry ::: {1..16} ::: {2..6} ::: {2..6} # 16x5x5=400
    parallel echo python generate_datum.py -o data --json        val   ferry ::: {1..4}  ::: {2..6} ::: {2..6} #  4x5x5=100
    parallel echo python generate_datum.py -o data --json        test  ferry ::: {1..4}  ::: {2..6} ::: {2..6}

    parallel echo python generate_datum.py -o data --json        train gripper ::: {1..80} ::: -- ::: --randomize ::: {2..10..2} # 80x5=400
    parallel echo python generate_datum.py -o data --json        val   gripper ::: {1..20} ::: -- ::: --randomize ::: {2..10..2} # 20x5=100
    parallel echo python generate_datum.py -o data --json        test  gripper ::: {1..20} ::: -- ::: --randomize ::: {2..10..2}

    parallel echo python generate_datum.py -o data --json        train miconic ::: {1..50} ::: {2..4} ::: {2..4} # 50x3x3 = 450
    parallel echo python generate_datum.py -o data --json        val   miconic ::: {1..12} ::: {2..4} ::: {2..4} # 12x3x3 = 108
    parallel echo python generate_datum.py -o data --json        test  miconic ::: {1..12} ::: {2..4} ::: {2..4}

    # has action cost
    # parallel echo python generate_datum.py -o data --json        train parking ::: {1..16} ::: {2..6} ::: {2..6} # 16x5x5 = 400
    # parallel echo python generate_datum.py -o data --json        val   parking ::: {1..4}  ::: {2..6} ::: {2..6} #  4x5x5 = 100
    # parallel echo python generate_datum.py -o data --json        test  parking ::: {1..4}  ::: {2..6} ::: {2..6}

    parallel echo python generate_datum.py -o data --json        train visitall {1} {2} {2} {3} ::: {1..70} ::: {3..5} ::: 0.5 1.0 # 70x3x2 = 420
    parallel echo python generate_datum.py -o data --json        val   visitall {1} {2} {2} {3} ::: {1..17} ::: {3..5} ::: 0.5 1.0 # 17x3x2 = 102
    parallel echo python generate_datum.py -o data --json        test  visitall {1} {2} {2} {3} ::: {1..17} ::: {3..5} ::: 0.5 1.0

}

# larger instances.
tasks2 (){
    parallel echo python generate_datum.py -o data --json -t 1800 test2 blocksworld ::: {1..11} ::: {11..22}
    parallel echo python generate_datum.py -o data --json -t 1800 test2 logistics ::: 1      ::: {3..6} ::: {3..8} ::: {3..6} ::: {5..8} ::: 5
    parallel echo python generate_datum.py -o data --json -t 1800 test2 satellite ::: 1      ::: {2..7} ::: {3..4} ::: {4..7} ::: 8 11 16 ::: {4..7}
    parallel echo python generate_datum.py -o data --json -t 1800 test2 ferry ::: {1..4}  ::: {10..30..5} ::: {10..30..5}
    parallel echo python generate_datum.py -o data --json -t 1800 test2 gripper ::: {1..20} ::: -- ::: --randomize ::: {20..60..10}
    parallel echo python generate_datum.py -o data --json -t 1800 test2 miconic ::: {1..12} ::: {10..30..10} ::: {10..30..10}
    # parallel echo python generate_datum.py -o data --json -t 1800 test2 parking ::: {1..4}  ::: {10..25..5} ::: {10..25..5}
    parallel echo python generate_datum.py -o data --json -t 1800 test2 visitall {1} {2} {2} {3} ::: {1..17} ::: {6..8} ::: 0.5 1.0
}

# perform a list of commands in parallel.
tasks  | ./block-submission.py 4 12 -- jbsub -proj pddlsl-gen-$(date +%Y%m%d%H%M) -queue x86_1h -mem 32g -cores 4
tasks2 | ./block-submission.py 4 12 -- jbsub -proj pddlsl-gen-$(date +%Y%m%d%H%M) -queue x86_6h -mem 32g -cores 4


# to generate planning instances, run:
# cd data; ./copy-100-problems-to-plan-directory.sh test plan; ./copy-100-problems-to-plan-directory.sh test2 plan2
