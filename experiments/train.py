#!/usr/bin/env python

"""
For usage, run `train.py -h` and read README.org
"""

import os
import shutil
import math
import argparse
import errno
import random
import numpy
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers.logger import Logger, DummyLogger
import numpy as np
import lifted_pddl
import json
import warnings

from typing import Optional

from util import *
from pddlsl.constants import (
    HEURISTICS,
    COSTS,
    DOMAIN_TO_GENERATOR,
    TRAINING_LOGS_DIR,
    TRAINING_CKPT_DIR,
    TRAINING_JSON_DIR,
)

from pddlsl.datamodule import (
    CommonDataModule,
    NLMCarlosV2DataModule,
    RRDataModule,
    HGNDataModule,
)

from pddlsl.heuristic_learner import (
    HeuristicLearner,
    SupervisedLearner
)

from pddlsl.model_wrapper import (
    NLMCarlosV2Wrapper,
    HGNWrapper,
    RRWrapper,
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
        description="It trains a model and stores the test losses into a JSON file.")
    parser.add_argument('-o', '--output', metavar="DIR", default="data", help="Root directory for storing the generated problem and data")
    parser.add_argument('--seed', default=42, type=int, metavar="N", help="Seed for the experiment")
    parser.add_argument('--train-size', type=int, metavar="N", default=400, help="The number of problems to use for training")

    parser.add_argument('--train-dir', metavar="DIR", default='train', help='The name of the subdirectory under the dataset directory for each domain, e.g., for logistics, it uses data/logistics/train/')
    parser.add_argument('--val-dir',   metavar="DIR", default='val',   help='The name of the subdirectory under the dataset directory for each domain, e.g., for logistics, it uses data/logistics/val/')
    parser.add_argument('--test-dir',  metavar="DIR", default='test',  help='The name of the subdirectory under the dataset directory for each domain, e.g., for logistics, it uses data/logistics/test/')
    parser.add_argument('--json-dir',  metavar="DIR", default=TRAINING_JSON_DIR, help='The directory to store the json files.')
    parser.add_argument('--logs-dir',  metavar="DIR", default=TRAINING_LOGS_DIR, help='The directory to store the tensorboard logs (event.out...) and hparams.yaml.')
    parser.add_argument('--ckpt-dir',  metavar="DIR", default=TRAINING_CKPT_DIR, help='The directory to store the checkpoint files (.ckpt).')

    parser.add_argument('-f', '--force', action="store_true", help="Overwrite an existing test result (json file).")
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--clip', type=float, default=0.1, help="Gradient clipping")
    parser.add_argument('--deterministic', action="store_true", help="Use slow deterministic training.")
    parser.add_argument('--batch-size', type=int, metavar="N", default=256, help="Batch size")
    parser.add_argument('--steps', type=int, metavar="N", default=10000, help="Maximum training update steps")
    parser.add_argument('--validation-epochs', type=int, metavar="N", default=10, help="Perform validation every N epochs.")
    parser.add_argument('--compute', choices=("auto","gpu","cpu"), default="auto",
                        help=("The compute mode. "
                              "'cpu' forces the CPU mode. "
                              "'gpu' forces the GPU mode. When no GPU is found, it raises an error and terminates. "
                              "'auto' tries to use a GPU, and falls back to the CPU mode when failed."))
    parser.add_argument('--mode', choices=("train","test","both"), default="both",
                        help=("The process to perform. "
                              "'train' and 'test' are self-explanatory. "
                              "'both' performs both."))
    parser.add_argument('--if-ckpt-exists',
                        choices=("supersede","resume","error","skip"),
                        default="skip",
                        help=("What to do when a checkpoint exists in the training mode. "
                              "'supersede': remove the existing checkpoint and create a new one. "
                              "'resume': resume the training from the existing checkpoint. "
                              "'error': quit immediately raising an error. "
                              "'skip': skip the training and proceed to the testing, if necessary. "))
    parser.add_argument('--if-ckpt-does-not-exist',
                        choices=("create","error"),
                        default="create",
                        help=("What to do when a checkpoint does not exist in the training mode. "
                              "'create': create a new one. "
                              "'error': quit immediately raising an error. "))
    parser.add_argument('dist', choices=('gaussian','truncated'))
    parser.add_argument('domain', choices=DOMAIN_TO_GENERATOR.keys())

    subparsers = parser.add_subparsers(title="learner", help="Specifies a learner class")

    for learner_cls, subparser in [(SupervisedLearner,
                                    subparsers.add_parser('supervised', help='Supervised learning specific options'))]:
        learner_cls.add_parser_arguments(subparser)

        subsubparsers = subparser.add_subparsers(title="model", help="model name and model-specific options")

        NLMCarlosV2Wrapper.add_parser_arguments(subsubparsers.add_parser('NLMCarlosV2', help='Neural Logic Machine specific options'))
        RRWrapper.add_parser_arguments(subsubparsers.add_parser('RR', help='Ridge Regression specific options'))
        HGNWrapper.add_parser_arguments(subsubparsers.add_parser('HGN', help='HyperGraph Network specific options'))

    args = parser.parse_args()

    # Each model (NLMCarlosV2, RR, HGN) requires a different data functionality
    if args.model == "NLMCarlosV2":
        args.dm_cls      = NLMCarlosV2DataModule
        args.network_cls = NLMCarlosV2Wrapper
    elif args.model == "RR":
        args.dm_cls      = RRDataModule
        args.network_cls = RRWrapper
    elif args.model == "HGN":
        args.dm_cls      = HGNDataModule
        args.network_cls = HGNWrapper
    else:
        raise "huh?"

    # Right now, learner needs to be "supervised"
    if args.learner == 'supervised':
        args.learner_cls = SupervisedLearner
    else:
        raise 'huh?'

    # Derive additional arguments specific to the network class / learner class (NLM, HGN, RR)
    args.learner_cls.derive_arguments(args)
    args.network_cls.derive_arguments(args)

    return args


def train(args, id, dm, training_ckpt_path : Optional[str]):
    if args.verbose:
        print("--- Training started ---")

    model = args.learner_cls(args)

    if args.compute == "cpu":
        print(f"running in the CPU mode (forced)")
        device_options = {"accelerator":"cpu"}
    elif args.compute == "auto":
        if pl.pytorch_lightning.accelerators.cuda.num_cuda_devices() <= 0:
            print(f"running in the CPU mode (auto: GPU not found)")
            device_options = {"accelerator":"cpu"}
        else:
            print("running in the GPU mode (auto: GPU found)")
            device_options = {"accelerator":"cuda","devices":find_available_gpu()}
    elif args.compute == "gpu":
        if pl.pytorch_lightning.accelerators.cuda.num_cuda_devices() <= 0:
            print(f"GPU required, but none found. Aborting")
            raise RuntimeError(f"GPU required, but none found. Aborting")
        else:
            print("running in the GPU mode (required: GPU found)")
            device_options = {"accelerator":"cuda","devices":find_available_gpu()}
    else:
        raise RuntimeError("huh?")

    # We save three checkpoints: the last one, the one with the best val NLL and the one with the best val MSE
    trainer = pl.Trainer(max_steps               = args.steps,
                         check_val_every_n_epoch = args.validation_epochs,
                         logger                  = TensorBoardLogger(args.logs_dir, name=id, version=""),
                         callbacks               = [DeviceStatsMonitor(),
                                                    ModelCheckpoint(monitor=None,
                                                                    enable_version_counter=False,
                                                                    every_n_epochs=1,
                                                                    dirpath=args.ckpt_dir,
                                                                    filename=id,
                                                                    save_on_train_epoch_end=True),
                                                    ModelCheckpoint(monitor='v_nll',
                                                                    enable_version_counter=False,
                                                                    every_n_epochs=1,
                                                                    dirpath=args.ckpt_dir,
                                                                    filename=id+"-nll",
                                                                    mode='min',
                                                                    save_top_k=1,
                                                                    save_on_train_epoch_end=False),
                                                    ModelCheckpoint(monitor='v_mse',
                                                                    enable_version_counter=False,
                                                                    every_n_epochs=1,
                                                                    dirpath=args.ckpt_dir,
                                                                    filename=id+"-mse",
                                                                    mode='min',
                                                                    save_top_k=1,
                                                                    save_on_train_epoch_end=False)],
                         deterministic           = args.deterministic,
                         gradient_clip_val       = args.clip,
                         **device_options)

    # note: this restores the training step. max_step works as expected.
    # note2: we always resume from the last checkpoint, not the checkpoint with best val NLL or MSE
    #		 (this is done so that logs work correctly)
    trainer.fit(model, datamodule=dm, ckpt_path=training_ckpt_path)
    if args.verbose:
        print("--- Training finished ---")
    pass


def load(args, training_ckpt_path):
    model = args.learner_cls.load_from_checkpoint(training_ckpt_path)

    if args.compute == "cpu" or pl.pytorch_lightning.accelerators.cuda.num_cuda_devices() <= 0:
        if args.compute == "cpu":
            print(f"running in the CPU mode (forced)")
        else:
            print(f"running in the CPU mode (GPU not found)")
        trainer = pl.Trainer(accelerator='cpu',
                             callbacks=[],
                             logger=DummyLogger(),
                             deterministic=args.deterministic)
    else: # Train on GPU
        print("running in the GPU mode")
        trainer = pl.Trainer(accelerator='cuda',
                             callbacks=[],
                             logger=DummyLogger(),
                             devices=find_available_gpu(),
                             deterministic=args.deterministic)

    return model, trainer


def test(args, dm, training_json_path, training_ckpt_path):
    if args.verbose:
        print("--- Test started ---")

    def fn(path):
        model, trainer = load(args, path)
        losses = trainer.test(model=model, datamodule=dm)[0]
        step = model.persistent_global_step.item()
        return losses, step

    last_path = training_ckpt_path
    nll_ckpt_path = append_to_name(training_ckpt_path, "-nll")
    mse_ckpt_path = append_to_name(training_ckpt_path, "-mse")
    last_losses, last_step = fn(training_ckpt_path)
    best_nll_losses, best_nll_step = fn(nll_ckpt_path)
    best_mse_losses, best_mse_step = fn(mse_ckpt_path)

    if args.verbose:
        print("--- Test finished ---")
        print("> Last training step:", last_step)
        print("> Its checkpoint path:", last_path)
        print("> Training step that achived the best validation NLL:", best_nll_step)
        print("> Its checkpoint path:", nll_ckpt_path)
        print("> Training step that achived the best validation MSE:", best_mse_step)
        print("> Its checkpoint path:", mse_ckpt_path)

    save_json(
        {
            **{k:v for k,v in vars(args).items() if k not in {"learner_cls","network_cls","dm_cls"}},
            **{'last_'+k:v for k,v in last_losses.items()},
            'last_step': last_step,
            'last_path': nll_ckpt_path,
            **{'best_nll_'+k:v for k,v in best_nll_losses.items()},
            'best_nll_step': best_nll_step,
            'best_nll_path': nll_ckpt_path,
            **{'best_mse_'+k:v for k,v in best_mse_losses.items()},
            'best_mse_step': best_mse_step,
            'best_mse_path': mse_ckpt_path,
        },
        training_json_path)

    if args.verbose:
        print("--- Test saved ---")


def main(args):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    seed_everything(args.seed, workers=True)
    warnings.simplefilter("ignore")
    id = args.learner_cls.id(args) + "_" + args.network_cls.id(args)

    training_json_path = os.path.join(args.json_dir, f'{id}.json')
    training_logs_path = os.path.join(args.logs_dir, f'{id}')
    training_ckpt_path = os.path.join(args.ckpt_dir, f'{id}.ckpt')

    dm = args.dm_cls(args)

    if args.mode in {"train", "both"}:
        if os.path.exists(training_ckpt_path):
            if args.if_ckpt_exists == "supersede":
                print(f"checkpoint exist, rerunning the training: {training_ckpt_path}")
                ensure_file_removed(training_ckpt_path)
                ensure_file_removed(append_to_name(training_ckpt_path, "-nll"))
                ensure_file_removed(append_to_name(training_ckpt_path, "-mse"))
                ensure_file_removed(training_json_path)
                ensure_file_removed(training_logs_path)
                train(args, id, dm, None)
            elif args.if_ckpt_exists == "resume":
                print(f"checkpoint exist, resuming the training: {training_ckpt_path}")
                train(args, id, dm, training_ckpt_path)
            elif args.if_ckpt_exists == "error":
                raise f"checkpoint should not exist: {training_ckpt_path}"
            elif args.if_ckpt_exists == "skip":
                print(f"checkpoint exist, skipping the training: {training_ckpt_path}")
            else:
                raise "huh?"
        else:
            if args.if_ckpt_does_not_exist == "create":
                train(args, id, dm, None)
            elif args.if_ckpt_does_not_exist == "error":
                raise f"checkpoint should exist: {training_ckpt_path}"
            else:
                raise "huh?"

    if args.mode in {"test", "both"} and \
       target_required(training_json_path,
                       depends_on=training_ckpt_path,
                       force=args.force,
                       verbose=args.verbose):

        test(args, dm,
             training_json_path,
             training_ckpt_path)

    pass


if __name__ == '__main__':
    args = parse_arguments()
    try:
        main(args)
    except:
        import pddlsl.stacktrace
        pddlsl.stacktrace.format(arraytypes={numpy.ndarray,torch.Tensor},include_self=False)
