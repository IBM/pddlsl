#!/usr/bin/env python3

import os
import subprocess
import argparse
import logging
import glob
import tqdm
import json
import pandas as pd
import multiprocessing as mp
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_info_columns', 500)

import pddlsl.stacktrace

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("root", nargs="?", default="training_json", help=f"directory containing json files.")
parser.add_argument("output", nargs="?", default="results_training.csv", help=f"output csv file name.")

args = parser.parse_args()

def load(logfile):
    with open(logfile,"r") as f:
        try:
            record = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"malformed json file: {logfile}")
            return None

    # temporary workaround to save csv file size
    for k in {"plan",
              "breadth",
	      "depth",
	      "hidden_features_list",
	      "reducers_list",
	      "out_features",
	      "num_output_features",
              "heuristic",
	      "problem",
              "best_mse_path",
	      "best_path",
              "best_nll_path",
              "global_output_size",
              "max_num_add_effects",
	      "max_num_preconditions",
	      "hidden_size",
	      "edge_input_size",
	      "node_input_size",
	      "global_input_size",
              "compute",
	      "mode",
	      "if_ckpt_exists",
	      "if_ckpt_does_not_exist",
              "force",
	      "verbose",
              "json_dir",
              "ckpt_dir",
              "logs_dir",
              "train_dir",
              "val_dir",
              "test_dir",
              "validation_epochs",
              "test_data_keys",
              "train_data_keys",
	      "elbo_prior_mu",
	      "elbo_prior_mu_train",
	      "elbo_prior_mu_test",
	      "elbo_blind_l",
	      "elbo_blind_l_train",
	      "elbo_blind_l_test",
	      "elbo_kl_coeff",}:
        if k in record:
            del record[k]
    record["model"] = translate(record["model"])
    return pd.Series(record)


def load_files(p):
    print("\nlisting logfiles...")
    logfiles = glob.glob(os.path.join(args.root, "*.json"))
    series_list = []
    # for logfile in tqdm.tqdm(logfiles):
    #     series_list.append(pd.Series(load(logfile)))
    print("\nloading logfiles...")
    with tqdm.tqdm(total=len(logfiles)) as pbar:
        for r in p.imap_unordered(load,logfiles,chunksize=mp.cpu_count()):
            if r is not None:
                series_list.append(r)
            pbar.update()

    df = pd.DataFrame(series_list)
    return df




def inc(value=None):
    if value is not None:
        inc.c = value-1
    inc.c += 1
    return inc.c
inc.c = 0

translations = {
    "NLMCarlosV2" : ("NLM",inc())
}

def translate(name):
    name = str(name)
    if name.endswith("-strips"):
        return name[:-7]
    elif name in translations:
        return translations[name][0]
    else:
        return name

def order(name):
    if isinstance(name, float):
        return name
    elif isinstance(name, int):
        return name
    name = str(name)
    if name in translations:
        return translations[name][1]
    else:
        return 0



if __name__ == "__main__":
    try:
        print(f"running with {len(os.sched_getaffinity(0))} processes")
        with mp.Pool(len(os.sched_getaffinity(0))) as p:
            df = load_files(p)
            df = df.reindex(sorted(df.columns), axis=1)
            df.to_csv(args.output)
    except:
        stacktrace.format()
