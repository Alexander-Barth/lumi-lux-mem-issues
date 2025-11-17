|                                             **case** | **size 32** | **size 64** | **size 128** | **size 256** |
|-----------------------------------------------------:|------------:|------------:|-------------:|-------------:|
|                        CUDA.jl NVIDIA A100 SXM4 40GB |        0.29 |        0.77 |          2.7 |        10.32 |
|                        pytorch NVIDIA A100 SXM4 40GB |        0.39 |        1.31 |         4.19 |        16.41 |
|       exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |        0.81 |         2.7 |        10.46 |        41.45 |
|   exclusif-AMDGPU.jl non eagerGC AMD Instinct MI250X |        0.79 |        2.71 |        10.45 |        41.38 |
|   non exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |        0.88 |        4.32 |        17.52 |        43.17 |
| non exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |        0.86 |        4.31 |        17.57 |        71.07 |
|             non exclusif-pytorch AMD Instinct MI250X |        0.52 |        1.89 |         7.31 |        29.04 |



|                                             **case** | **size 32** | **size 64** | **size 128** | **size 256** |
|-----------------------------------------------------:|------------:|------------:|-------------:|-------------:|
|                        CUDA.jl NVIDIA A100 SXM4 40GB |         1.0 |         1.0 |          1.0 |          1.0 |
|                        pytorch NVIDIA A100 SXM4 40GB |     1.34483 |      1.7013 |      1.55185 |      1.59012 |
|       exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |      2.7931 |     3.50649 |      3.87407 |      4.01647 |
|   exclusif-AMDGPU.jl non eagerGC AMD Instinct MI250X |     2.72414 |     3.51948 |      3.87037 |      4.00969 |
|   non exclusif-AMDGPU.jl eagerGC AMD Instinct MI250X |     3.03448 |     5.61039 |      6.48889 |      4.18314 |
| non exclusif-AMDGPU.jl noeagerGC AMD Instinct MI250X |     2.96552 |      5.5974 |      6.50741 |      6.88663 |
|             non exclusif-pytorch AMD Instinct MI250X |      1.7931 |     2.45455 |      2.70741 |      2.81395 |
