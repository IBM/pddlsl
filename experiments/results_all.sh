#!/bin/bash -x

./results_planning.py data plan2 results_planning2.csv
./results_training.py training_json_large results_training2.csv
./results_planning.py data plan results_planning.csv
./results_training.py training_json results_training.csv
