#!/usr/bin/env python

import sys
import os
import argparse
import subprocess
import json
import pddlsl.stacktrace

parser = argparse.ArgumentParser()
parser.add_argument("json")


def main(args):

    jsonfile = args.json
    planfile = jsonfile.replace(".json",".plan")
    domfile = os.path.join(os.path.dirname(jsonfile),"..","..","domain.pddl")
    probfile = os.path.dirname(jsonfile).replace(".logs",".pddl")

    with open(jsonfile, "r") as f:
        data = json.load(f)

    if data["plan_length"] == -1:
        return 2

    with open(planfile, "w") as f:
        for action in data["plan"]:
            print(action, file=f)

    valfile = os.path.join(os.path.dirname(__file__), "..", "libs", "VAL", "build", "bin", "Validate")

    try:
        subprocess.run([valfile, domfile, probfile, planfile])
        return 0
    except:
        return 1


if __name__ == '__main__':
    args = parser.parse_args()
    status = None
    try:
        status = main(args)
    except:
        pddlsl.stacktrace.format()
    if status is not None:
        sys.exit(status)
