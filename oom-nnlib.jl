using Printf
using Random
using NNlib

if !isnothing(Sys.which("nvidia-smi"))    
    using CUDA, cuDNN
    const device = cu
else
    using AMDGPU
    @show AMDGPU.devices()
    AMDGPU.versioninfo()
    const device = roc
end

if haskey(ENV,"SLURM_JOB_ID")
    cp(@__FILE__,"script-" * ENV["SLURM_JOB_ID"] * ".jl",force=true)
end

in_channels = 2
out_channels = 2

w = device(randn(Float32,3,3,in_channels,out_channels))

batchsize = 128

function inference(device, in_channels, out_channels, batchsize)
    sum_loss = 0.0
    println("in inference")
    for i = 1:1000^5
        x = randn(Float32,64,64,in_channels,batchsize) |> device
        y_hat = conv(x,w,flipped=true)
        #y_hat = conv(x,w) # also leaking

        sum_loss += sum(abs2,y_hat)

        # explictely running julia's garbage collector
        GC.gc()
        # print Max. RSS in MiB
        @printf "%d: Max. RSS:  %9.3f MiB\n" i Sys.maxrss()/2^20
        flush(stdout)
    end
    return sum_loss
end

println("about to call inference")
sum_loss = inference(device, in_channels, out_channels, batchsize)
@show sum_loss
