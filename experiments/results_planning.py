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

parser.add_argument("root", nargs="?", default="data", help=f"directory containing domain directories.")
parser.add_argument("stage", nargs="?", default="plan", help=f"subdirectory that containing problem instances. it should be under each domain directory.")
parser.add_argument("output", nargs="?", default="results_planning.csv", help=f"directory containing domain directories.")

args = parser.parse_args()

def load(logfile):
    with open(logfile,"r") as f:
        try:
            record = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"malformed json file: {logfile}")
            return None

    # record = {
    #     k:v
    #     for k, v in record.items()
    #     if not isinstance(v, list)
    # }
    if "plan" in record:
        record["solved"] = 1
    else:
        record["solved"] = 0

    # HACK! The seed is always 42 in the planning time
    # and it overwrites the seed during the training,
    # accidentally losing the information.
    # To recover the original seed, I parse the pathname.
    # if "best_path" in record:
    #     best_path = record["best_path"]
    #     name = os.path.splitext(os.path.basename(best_path))[0]
    #     name_without_tiebreak = name.split("-")[0]
    #     seed = name_without_tiebreak.split("_")[23]
    #     if name.endswith("-mse"):
    #         # seed = seed[:-4]
    #         print(f"we no longer use mse checkpoints: {logfile}")
    #         return None
    #     record["seed"] = int(seed)

    # filter records to reduce the csv file size
    record2 = dict()
    # definitely unused results
    blacklist = {
        "plan",
        # options that definitely do not affect results
            "output",
        "compute",
	"mode",
	"if_ckpt_exists",
	"if_ckpt_does_not_exist",
        "force",
	"verbose",
        "validation_epochs",
        "test_data_keys",
        "train_data_keys",
        # options that are not used now
	    "elbo_prior_mu",
	"elbo_prior_mu_train",
	"elbo_prior_mu_test",
	"elbo_blind_l",
	"elbo_blind_l_train",
	"elbo_blind_l_test",
	"elbo_kl_coeff",
        # paths
        "heuristic",
	"problem",
        "domain",
        # misc options / duplicates
        "learner_cls",
        "network_cls",
	"hidden_features_list",
	"reducers_list",
	"out_features",     # this is a list!
    }
    for k, v in record.items():
        if k in blacklist or \
           k.startswith("best_mse_T") or \
           k.startswith("best_nll_T") or \
           k.startswith("last_T") or \
           k.endswith("dir") or \
           k.endswith("path") :
            continue
        record2[k] = record[k]
    record2["model"] = translate(record["model"])
    return pd.Series(record2)


def load_files(p):
    wild = os.path.join(args.root, "*", args.stage, "*.logs", "*.json")
    print(f"\nlisting logfiles in {wild}...")
    logfiles = glob.glob(wild)
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
            print(df)
            df.info()

            metrics = {
	        "elapsed",
	        "evaluated",
	        "plan_length",
	    	"solved",
	    }

            options = set(df.columns) - metrics - {"seed"}
            agg_rule = dict()
            for name in metrics:
                agg_rule[name]=(name, "mean")
                agg_rule[name+"_std"]=(name, "std")
                agg_rule[name+"_min"]=(name, "min")
                agg_rule[name+"_max"]=(name, "max")
            df = df.drop("seed",axis=1).groupby(list(options),dropna=False).aggregate(**agg_rule).reset_index()
            df = df.reindex(sorted(df.columns), axis=1)
            print(df)
            df.info()
            df.to_csv(args.output)
    except:
        pddlsl.stacktrace.format()
