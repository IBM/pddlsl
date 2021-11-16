

import os
import os.path
import shutil
import json

# Function that removes a file or directory in case it exists
# Sources: https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
#          https://stackoverflow.com/questions/6996603/how-can-i-delete-a-file-or-folder-in-python
def ensure_file_removed(path):
    # File
    if os.path.isfile(path) or os.path.islink(path):
        try:
            os.remove(path)
        except OSError as e: # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                raise # re-raise exception if a different error occurred
    # Folder
    elif os.path.isdir(path):
        shutil.rmtree(path) # Remove folder and contents


def ensure_directory_pathname(path):
    if path != '' and path[-1] != '/':
        return path + '/'
    else:
        return path


def append_to_name(path, string):
    root, ext = os.path.splitext(path)
    return root + string + ext


def basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]


def save_json(data:dict, path:str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w+") as f:
        json.dump(data, f, indent=2)


def target_required(path, *others, depends_on = None, force=False, verbose=True, dominated=None):
    if depends_on is None:
        depends_on = []
    elif isinstance(depends_on, tuple):
        depends_on = list(depends_on)
    elif isinstance(depends_on, list):
        pass
    else:
        depends_on = [depends_on]

    if verbose:
        print(f"> Checking if the target requires regeneration: {path}")

    def newer(path1, path2):
        return os.path.getmtime(path1) > os.path.getmtime(path2)

    def required():
        if not os.path.isfile(path):
            return "file not found"
        # path exists below.
        if any([newer(d,path) for d in depends_on]):
            return "file is outdated"
        if dominated is None:
            return False
        else:
            if dominated(path):
                return False
            else:
                return "file is not dominated"


    r = required()
    if r or force:
        if verbose:
            if r:
                print(f"> Target required (reason: {r}), regenerating: {path}")
            elif force:
                print(f"> Target not required, but forcing regeneration: {path}")
        for p in [path] + list(others):
            ensure_file_removed(p)
        return True
    else:
        if verbose:
            print(f"> Target not required, skipping: {path}")
        return False


def idcache(fn):
    """
    Custom decorator for caching functions.
    It stores the cache based on the built-in object id of the arguments.
    I'm not using functool.cache because Namespace is not hashable (which is stupid) and
    I didn't want to import frozendict just for this
    """
    cache = dict()
    def fn2(*args):
        key = tuple(id(arg) for arg in args)
        if key not in cache:
            cache[key] = fn(*args)
        return cache[key]

    fn2.fn = fn
    fn2.cache = cache

    return fn2


import torch.cuda
import time
def get_average_gpu_utilization(gpu_id, num_samples, sample_interval):
    # We run nvidia-smi several times because GPU util oscillates, so even though a GPU is busy, by pure chance
    # torch.cuda.utilization could show a util of 0 when it is called
    utilizations = []
    for _ in range(num_samples):
        utilization = torch.cuda.utilization(gpu_id)
        utilizations.append(utilization)
        time.sleep(sample_interval)

    if len(utilizations) > 0:
        average_utilization = sum(utilizations) / len(utilizations)
        return average_utilization
    else:
        return 0


# Returns the ID of the GPU with the lowest utilization (or the first GPU with 0 util)
def find_available_gpu(num_samples=5, sample_interval=0.2) -> list[int]:
    """
    Utility function that reliably obtains the ID of GPUs with the lowest utilization.
    pytorch_lightning.accelerators.find_usable_cuda_devices does not expect fluctuations of the GPU usage.
    """
    num_gpus = torch.cuda.device_count()
    available_gpu = None
    min_utilization = float('inf')

    for gpu_id in range(num_gpus):
        utilization = get_average_gpu_utilization(gpu_id, num_samples, sample_interval)
        if utilization == 0:
            return [gpu_id]  # Return the first available GPU with 0% utilization
        elif utilization < min_utilization:
            available_gpu = gpu_id
            min_utilization = utilization

    return [available_gpu] # We need to return [0] instead of e.g. 0 because pl.Trainer expects GPU indexes as lists
