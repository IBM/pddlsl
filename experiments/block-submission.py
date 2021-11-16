#!/bin/env python

"""

Submit a large number of commands to a job scheduler without flooding the
queue with short jobs by packing multiple commands into a single job.
This script is most useful when submitting a large number of short jobs.
Such short jobs are problematic because

* Some HPC environment has a limit on the number of jobs a single user can submit.
* Short jobs make the job handling cumbersome. For example, after noticing
  a bug, it takes some time to cancel the jobs if there are thousands.
* Short jobs harm the performance of the job scheduler.


It depends on GNU parallel, as the commands inside each job is parallelized by it.
You can specify the options for GNU parallel with -o flag.

Examples:

These examples run 128 echo commands in total.
It creates 4 jobs submitted to a 1 hour queue, where each job runs 32 tasks with 8 cores.
4 tasks are assigned to each core assuming that each task takes at most 15 minutes so that
the 4 tasks finished in an hour.

Running 1 hour queue, 4g memory, 8 cores on jbsub cluster:

  parallel echo echo ::: {1..128} | block-submission.py 8 4 -- jbsub -proj subgen -queue x86_1h -mem 4g -cores 8

Running 1 hour, 4g virtual memory per process, 8 cores on Torque/PBS cluster:

  parallel echo echo ::: {1..128} | block-submission.py 8 4 -- qsub -l ppn=8,cput=3600,pvmem=4g

General explanation of structure of above commands:

 - we use `parallel echo` to echo the 128 strings "echo <N>" (one string for each of the integers from 1 to 128 inclusive) into the block-submission.py script. It is not necessary to use parallel in this way with the block-submission.py script: you can also just generate a bunch of commands, store them in a file (one per line), and the `cat your_command_file.txt | ./block-submission.py <...>`.
 - arguments to block-submission.py must be separated from the job submission command (e.g. "jbsub") with the standard "--" delimiter.

"""

import sys
import os.path
import argparse
import subprocess
import tempfile

parser = argparse.ArgumentParser(
    formatter_class = argparse.RawDescriptionHelpFormatter,
    description = __doc__)

parser.add_argument("cores",
                    type=int,
                    help="Number of cores to assign to a job. "
                    "Note that the max memory usage of all cores must not exceed the max memory for the job.")

parser.add_argument("N",
                    type=int,
                    help="Number of processes to run sequentially in each core. "
                    "If each job has a runtime limit of T seconds, make sure each command ends in T/N seconds.")


parser.add_argument("commands",
                    nargs="*",
                    help="Job submission command for each job except the command to run. ")


parser.add_argument("-f",
                    metavar="FILE",
                    help="Takes the input from a file containing commands separated by newlines, "
                    "instead of from the standard input.")

parser.add_argument("-o","--options",
                    help="Options given to gnu parallel. "
                    "By default, the value is -j [cores].")


args = parser.parse_args()



if args.f is None:
    f = sys.stdin
else:
    f = open(args.f, mode="r")

per_job : int = (args.cores * args.N)

# todo: balance the reminder (e.g., if the reminder is 1 when per_job = 4, do
# not assing 1 job to the last job, but assign 3 tasks to the last 3 jobs)

d = tempfile.mkdtemp(prefix="tmp.",dir=".")
print(d)

options = f" --bar --shuf --line-buffer --halt never"
if args.options is not None:
    options += args.options

jobid = 0
EOF=False
while not EOF:
    filename = os.path.join(d,str(jobid))
    with open(filename, mode="w") as f2:
        for i in range(per_job):
            line=f.readline()
            f2.write(line)
            if line=="":
                EOF=True
    # if command file is empty, then do not submit a job for its non-existent commands
    if os.stat(filename).st_size == 0:
        continue
    subprocess.run([*args.commands, "bash", "-c", f"parallel {options} < {filename}",])
    jobid += 1
