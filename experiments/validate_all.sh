#!/bin/bash

# don't do full shell expansion like data/*/plan/*.logs/*.json,
# it goes above the max argument limit for a shell (e.g. bash)

find data/*/plan/ -name "*.json" | parallel -v python ./validate.py
