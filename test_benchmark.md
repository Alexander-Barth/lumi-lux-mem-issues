
# Comparing pytorch/Lux.jl and MI250X/A100 for a 2D convolution

* 2D convolution with kernel size 3 of a matrix of size n x n x 64 x 128  (W x H x C x N) where `n` is the size parameter in the following tables following by a ReLU
* code: [bench_conv.jl](bench_conv.jl) and [bench_conv.py](bench_conv.py) for more information
* Absolute times in ms

|                                             **case** | **size 32** | **size 64** | **size 128** | **size 256** |
|-----------------------------------------------------:|------------:|------------:|-------------:|-------------:|
|                        CUDA.jl NVIDIA A100 SXM4 40GB |        0.29 |        0.77 |          2.7 |        10.32 |
|                        pytorch NVIDIA A100 SXM4 40GB |        0.39 |        1.31 |         4.19 |        16.41 |
|       exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |        0.81 |         2.7 |        10.46 |        41.45 |
|   exclusif-AMDGPU.jl non eagerGC AMD Instinct MI250X |        0.79 |        2.71 |        10.45 |        41.38 |
|   non exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |        0.88 |        4.32 |        17.52 |        43.17 |
| non exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |        0.86 |        4.31 |        17.57 |        71.07 |
|             non exclusif-pytorch AMD Instinct MI250X |        0.52 |        1.89 |         7.31 |        29.04 |


* Relative times compared to CUDA.jl NVIDIA A100:

|                                             **case** | **size 32** | **size 64** | **size 128** | **size 256** |
|-----------------------------------------------------:|------------:|------------:|-------------:|-------------:|
|                        CUDA.jl NVIDIA A100 SXM4 40GB |         1.0 |         1.0 |          1.0 |          1.0 |
|                        pytorch NVIDIA A100 SXM4 40GB |     1.34483 |      1.7013 |      1.55185 |      1.59012 |
|       exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |      2.7931 |     3.50649 |      3.87407 |      4.01647 |
|   exclusif-AMDGPU.jl non eagerGC AMD Instinct MI250X |     2.72414 |     3.51948 |      3.87037 |      4.00969 |
|   non exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |     3.03448 |     5.61039 |      6.48889 |      4.18314 |
| non exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |     2.96552 |      5.5974 |      6.50741 |      6.88663 |
|             non exclusif-pytorch AMD Instinct MI250X |      1.7931 |     2.45455 |      2.70741 |      2.81395 |

## Note
* We comparing a single graphical compute dice (GCD) to a full A100. AMD Instinct MI250X has two GCDs.
* `eagerGC` means `AMDGPU.EAGER_GC[] = true`, and noeagerGC means the default for AMDGPU.jl 2.1.2 (https://github.com/JuliaGPU/AMDGPU.jl/issues/844)
* exclusif mean we reserve a full AMD Instinct MI250X, i.e. two GCDs but use only one
* non-exclusif mean we reserve only a single GCD (the other GCD is potentially used by another user)
* LUMI (Instinct MI250X):
     * Python 3.12.11, pytorch 2.7.1, ROCm 6.2.4
     * julia 1.12.1, AMDGPU.jl 2.1.2, ROCm 6.2.2
* Lucia (A100):
     * Python 3.10.4, pytorch 1.12.0, CUDA 11.7.0 (PyTorch/1.12.0-foss-2022a-CUDA-11.7.0)
     * julia 1.12.1, CUDA.jl 5.9.3, CUDA 13.0.0
 

When trying to use a MI250X exclusivly, pytorch on Instinct MI250X fails with:

```Singularity> ROCR_VISIBLE_DEVICES=0 python bench_conv.py 
:0:rocdevice.cpp            :2986: 2909498332452 us: [pid:110488 tid:0x1517ee3ff700] Callback: Queue 0x1517ee000000 aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address. code: 0x29
Aborted
```
