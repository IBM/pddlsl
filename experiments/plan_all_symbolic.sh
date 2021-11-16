#!/bin/bash

timestamp=$(date +%Y%m%d%H%M)
max_evaluations=10000

# echo the command line calls to run
tasks (){
    domain=$1
    # note: "'{1}'" protects paths that contain whitespaces.

    parallel echo ./plan.py -v --max-evaluations ${max_evaluations} \
             ::: ff lmcut \
             ::: data/${domain}/plan2/*.pddl

}

# tasks blocksworld
# tasks satellite
# tasks logistics

# pass the commands to a job scheduler like:
# ./plan_all.sh | parallel jbsub -queue x86_1h -mem 8g -cores 1 -proj pddlsl-plan-$(date +%Y%m%d%H%M)

# or, to limit the number of jobs to be submitted to a queue, group jobs together with ./block-submission.py
# ./plan_all.sh | ./block-submission.py 16 72 -- jbsub -proj pddlsl-plan-$(date +%Y%m%d%H%M) -queue x86_6h -mem 16g -cores 16

tasks blocksworld | ./block-submission.py 1 60 -- jbsub -proj pddlsl-plan-$(date +%Y%m%d%H%M)-sym-bw -queue x86_6h -mem 2g -cores 1
# tasks satellite   | ./block-submission.py 1 60 -- jbsub -proj pddlsl-plan-$(date +%Y%m%d%H%M)-sym-st -queue x86_6h -mem 2g -cores 1
# tasks logistics   | ./block-submission.py 1 60 -- jbsub -proj pddlsl-plan-$(date +%Y%m%d%H%M)-sym-lg -queue x86_6h -mem 2g -cores 1
tasks ferry       | ./block-submission.py 1 60 -- jbsub -proj pddlsl-plan-$(date +%Y%m%d%H%M)-sym-fe -queue x86_6h -mem 2g -cores 1
tasks gripper     | ./block-submission.py 1 60 -- jbsub -proj pddlsl-plan-$(date +%Y%m%d%H%M)-sym-gr -queue x86_6h -mem 2g -cores 1
tasks visitall    | ./block-submission.py 1 60 -- jbsub -proj pddlsl-plan-$(date +%Y%m%d%H%M)-sym-vi -queue x86_6h -mem 2g -cores 1
