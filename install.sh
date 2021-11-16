#!/bin/bash

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

env=pddlsl

# execute it in a subshell so that set -e stops on error, but does not exit the parent shell.
(
    set -e

    (conda activate >/dev/null 2>/dev/null)  || {
        echo "This script must be sourced, not executed. Run it like: source $0"
        exit 1
    }

    conda env create -n $env -f environment.yml || {
        echo "installation failed; cleaning up"
        conda env remove -n $env
        exit 1
    }

    conda activate $env

    pip install -r requirements.txt

    pip install -e .

    pre-commit install --allow-missing-config # install pre-commit hooks

    git submodule update --init --recursive

    ################################################################
    # reset my local variables so that they do not affect reproducibility

    echo "path variables that are set locally:"
    echo "-----------------"
    env | grep PATH | cut -d= -f1
    echo "-----------------"
    echo

    conda env config vars set LD_LIBRARY_PATH=
    conda env config vars set PYTHONPATH=
    conda env config vars set PKG_CONFIG_PATH=

    # these are required for building dependencies properly under conda
    conda env config vars set CPATH=${CONDA_PREFIX}/include:${CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/include:

    # note: "variables" field in yaml file introduced in conda 4.9 does not work because it does not expand shell variables

    # these are required for the variables to take effect
    conda deactivate ; conda activate $env

    echo $CPATH ; echo $LD_LIBRARY_PATH

    conda env config vars list

    ################################################################
    # build VAL validator

    (
        cd libs/VAL
        mkdir build
        cd build
        cmake ..
        make -j $(nproc)
    )

    ################################################################
    # build fast downward
    (
        cd libs/downward
        ./build.py -j $(nproc) release
    )

    ################################################################
    # install python submodules
    (
        cd libs/pddl-generators
        pip install -e .
    )
    (
        cd libs/strips_hgn
        pip install -e .
    )

) && {
    conda activate $env >/dev/null 2>/dev/null || echo "This script must be sourced, not executed. Run it like: source $0"
}
