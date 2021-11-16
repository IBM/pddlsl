#!/bin/bash

# See README for the description.

common_command="./train.py --mode test --verbose --test-dir test2 --json-dir training_json_large"

(
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: --decay ::: 0.0 0.01 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn fixed \
             ::: --res ::: zero ff \
             ::: --l ::: lmcut \
             ::: NLMCarlosV2 \
             ::: --hidden-features ::: 16 \
             ::: --exclude-self ::: True False
    # smaller dataset
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 100 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: --decay ::: 0.0 0.01 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: ff \
             ::: --l ::: lmcut \
             ::: NLMCarlosV2 \
             ::: --hidden-features ::: 16 \
             ::: --exclude-self ::: True False
    # different l
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: --decay ::: 0.0 0.01 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: ff \
             ::: --l ::: blind hmax \
             ::: NLMCarlosV2 \
             ::: --hidden-features ::: 16 \
             ::: --exclude-self ::: True False
    # different res
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: --decay ::: 0.0 0.01 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: lmcut \
             ::: --l ::: lmcut \
             ::: NLMCarlosV2 \
             ::: --hidden-features ::: 16 \
             ::: --exclude-self ::: True False


) | parallel jbsub -queue x86_1h -mem 32g -cores 1 -proj pddlsl-$(date +%Y%m%d%H%M)-test2-nlm

(
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.001 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn fixed \
             ::: --res ::: zero ff \
             ::: --l ::: lmcut \
             ::: HGN \
             ::: --num-recursion-steps ::: 4
    # smaller dataset
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 100 \
             ::: --lr ::: 0.001 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: ff \
             ::: --l ::: lmcut \
             ::: HGN \
             ::: --num-recursion-steps ::: 4
    # different l
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.001 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: ff \
             ::: --l ::: blind hmax \
             ::: HGN \
             ::: --num-recursion-steps ::: 4
    # different res
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.001 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: lmcut \
             ::: --l ::: lmcut \
             ::: HGN \
             ::: --num-recursion-steps ::: 4

) | parallel jbsub -queue x86_1h -mem 8g -cores 1 -proj pddlsl-$(date +%Y%m%d%H%M)-test2-hgn

(
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn fixed \
             ::: --res ::: zero ff \
             ::: --l ::: lmcut \
             ::: RR
    # smaller dataset
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 100 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: ff \
             ::: --l ::: lmcut \
             ::: RR
    # different l
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: ff \
             ::: --l ::: blind hmax \
             ::: RR
    # different res
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 40000 \
             ::: gaussian truncated \
             ::: blocksworld ferry gripper visitall \
             ::: supervised \
             ::: --sigma ::: learn \
             ::: --res ::: lmcut \
             ::: --l ::: lmcut \
             ::: RR

) | parallel jbsub -queue x86_1h -mem 4g -cores 1 -proj pddlsl-$(date +%Y%m%d%H%M)-test2-rr
