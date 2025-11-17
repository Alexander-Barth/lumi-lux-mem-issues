
# Comparing pytorch/Lux.jl and MI250X/A100 for a 2D convolution

* 2D convolution with kernel size 3 of a matrix of size n x n x 64 x 128  (W x H x C x N) where `n` is the size parameter in the following tables following by a ReLU
* See the code: [bench_conv.jl](bench_conv.jl) and [bench_conv.py](bench_conv.py) for more information
* Absolute times in ms

|                                             **case** | **size 32** | **size 64** | **size 128** | **size 256** |
|-----------------------------------------------------:|------------:|------------:|-------------:|-------------:|
|                        CUDA.jl NVIDIA A100 SXM4 40GB |       0.292 |       0.769 |        2.696 |       10.323 |
|                        pytorch NVIDIA A100 SXM4 40GB |       0.394 |       1.307 |        4.187 |       16.415 |
|   non exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |       0.885 |       4.322 |       17.521 |        43.17 |
| non exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |       0.863 |       4.314 |       17.572 |       71.073 |
|             non exclusif-pytorch AMD Instinct MI250X |       0.523 |       1.891 |         7.31 |       29.035 |
|       exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |       0.811 |       2.698 |       10.462 |       41.452 |
|     exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |        0.79 |       2.706 |       10.446 |       41.379 |

The 71.073 ms for the 256x256 case, might be related to the fact that the MI250X is not used exclusively by the test.

* Relative times compared to CUDA.jl NVIDIA A100:
  
|                                             **case** | **size 32** | **size 64** | **size 128** | **size 256** |
|-----------------------------------------------------:|------------:|------------:|-------------:|-------------:|
|                        CUDA.jl NVIDIA A100 SXM4 40GB |         1.0 |         1.0 |          1.0 |          1.0 |
|                        pytorch NVIDIA A100 SXM4 40GB |        1.35 |         1.7 |         1.55 |         1.59 |
|   non exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |        3.03 |        5.62 |          6.5 |         4.18 |
| non exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |        2.95 |        5.61 |         6.52 |         6.88 |
|             non exclusif-pytorch AMD Instinct MI250X |        1.79 |        2.46 |         2.71 |         2.81 |
|       exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |        2.77 |        3.51 |         3.88 |         4.02 |
|     exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |         2.7 |        3.52 |         3.87 |         4.01 |


## Note
*  **AMD Instinct MI250X has two GCDs (Graphics Compute Die). We comparing a single GCD to a full A100.**
* `eagerGC` means `AMDGPU.EAGER_GC[] = true`, and  `noeagerGC` means the default GC for AMDGPU.jl 2.1.2 (https://github.com/JuliaGPU/AMDGPU.jl/issues/844)
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
