#!/bin/bash

# See README for the description.

common_command="./train.py --verbose --if-ckpt-does-not-exist create --if-ckpt-exists resume"

(
    parallel echo $common_command \
             ::: --seed ::: 42 100 200 300 400 \
             ::: --train-size ::: 400 \
             ::: --lr ::: 0.01 \
             ::: --steps ::: 100000 \
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
             ::: --steps ::: 100000 \
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
             ::: --steps ::: 100000 \
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
             ::: --steps ::: 100000 \
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


) | parallel jbsub -queue x86_6h -mem 4g -cores 1+1 -proj pddlsl-$(date +%Y%m%d%H%M)-train-nlm

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
    # comparison with crestein et al
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
             ::: --num-recursion-steps ::: 3

) | parallel jbsub -queue x86_24h -mem 8g -cores 1+1 -proj pddlsl-$(date +%Y%m%d%H%M)-train-hgn

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

) | parallel jbsub -queue x86_24h -mem 4g -cores 1+1 -proj pddlsl-$(date +%Y%m%d%H%M)-train-rr
