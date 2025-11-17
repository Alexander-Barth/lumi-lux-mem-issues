using Lux
using Random
using BenchmarkTools
using Statistics

if !isnothing(Sys.which("nvidia-smi"))
    import CUDA, cuDNN
    const GPU = CUDA
    gpu_name = CUDA.name(CUDA.device())
else
    import AMDGPU
#    AMDGPU.EAGER_GC[] = true
    const GPU = AMDGPU
    gpu_name = AMDGPU.HIP.name(AMDGPU.device())
end

device = Lux.gpu_device();

model = Conv((3,3),64 => 64,relu,cross_correlation=Lux.True())
#model = Conv((3,3),64 => 64,relu) # Warning: MIOpen supports only cross-correlation (flipkernel=true).
ps, st = device.(Lux.setup(Random.default_rng(), model));

sizes = 2 .^ (5:8) # 32 - 256
time_median = zeros(length(sizes))

fname = "$GPU.jl-" * replace(gpu_name," " => "-") * ".csv"
@show fname
f = open(fname,"w")
println(f,"size,median_time_second")

for i = 1:length(sizes)
    x = randn(Float32,sizes[i],sizes[i],64,128) |> device;
    print("size ",sizes[i]," ")
    bm = @benchmark (GPU.@sync model($x,$ps,$st)) samples=1000 seconds=60
    time_median[i] = time(median(bm)) / 1e9 # ns -> s
    println("time ",time_median[i] * 1e3," ms")
    println(f,sizes[i],",",time_median[i])   
end
close(f)
