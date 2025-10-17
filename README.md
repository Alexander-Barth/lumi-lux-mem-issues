
# Setup

This code will install all the needed software. This will need to be done only once:

```bash
git clone https://github.com/Alexander-Barth/lumi-lux-mem-issues
cd lumi-lux-mem-issues
module load Local-CSC julia/1.12.0 julia-amdgpu/1.1.3
julia --project=. --eval "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

For testing we will install all software in `~/.julia`. In my "production" runs, I have all software dependencies in a single tar file that is decompressed in `/tmp`.

# Run the script

## GPU memory issue


`gpu_mem_issue7.jl` is submitted to SLURM via:

```bash
sbatch --account=project_X --job-name=gpu_mem_issue7  --partition=small-g --time=48:00:00 --mem-per-cpu=30G --cpus-per-task=1  --ntasks=1 --nodes=1 --gpus=1  training.sh gpu_mem_issue7.jl
```

This script reproduces the `HSA_STATUS_ERROR_OUT_OF_RESOURCES` error:

```
[...]
(nn, lossval) = (9, 1.5727558f0)
(nn, lossval) = (10, 1.5775862f0)
:0:rocdevice.cpp            :2724: 10459743014355 us: [pid:2564  tid:0x150c4f3ff700] Callback: Queue 0x150b4de00000 Aborting with error : HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events. Code: 0x1008 Available Free mem : 0 MB
```


I got a consistent failure for this case (4 out of 4 tests).


## CPU memory "leak"


This script reproduces or memory leak (or memory fragmentation,...):

```bash
sbatch --account=project_X --job-name=oom --partition=small-g --time=48:00:00 --mem-per-cpu=30G --cpus-per-task=1  --ntasks=1 --nodes=1 --gpus=1  training.sh oom.jl
```

It will fail in about 4 out of 5 time with the error:

```
[...]
6855: Max. RSS:  29573.348 MiB
6856: Max. RSS:  29574.812 MiB
slurmstepd: error: Detected 1 oom_kill event in StepId=13631246.1. Some of the step tasks have been OOM Killed.
srun: error: nid007876: task 0: Out Of Memory
srun: Terminating StepId=13631246.1
```

# What I have tried:

* I tried to simplify the code, but unfortunately it seems to only reduce the probability of occurence of the failure (without solving it).
* I am now loading the module `julia-amdgpu/1.1.3` and reducing the `hard_memory_limit` and  `eager_gc` in `LocalPreferences.toml`.

```
[AMDGPU]
use_artifacts = false
hard_memory_limit = "60 %"
eager_gc = true
```

Previously reduced reproducers for `HSA_STATUS_ERROR_OUT_OF_RESOURCES` do not fail any more with these settings. But unfortunately, my primary application still fails with this error.
