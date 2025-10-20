import Zygote
using Dates
using Lux
using Optimisers
using Printf
using Random
using Statistics
using Test

if !isnothing(Sys.which("nvidia-smi"))    
    import CUDA, cuDNN
    const GPU = CUDA
else
    import AMDGPU
    @show AMDGPU.devices()
    AMDGPU.versioninfo()
    const GPU = AMDGPU
end

if haskey(ENV,"SLURM_JOB_ID")
    cp(@__FILE__,"script-" * ENV["SLURM_JOB_ID"] * ".jl")
end

in_channels = 2
channels = (64,128,256)
out_channels = 2

model = Chain(
    Conv((3,3),in_channels=>64,relu,pad = SamePad(), cross_correlation=Lux.True()),
    Conv((1,1),64=>out_channels,pad = SamePad(), cross_correlation=Lux.True()),
)

batchsize = 32*4

device = gpu_device()
#device = cpu_device() # no leakage observed
learning_rate = 0.001

ps, st = device.(Lux.setup(Random.default_rng(), model));

opt = Adam(learning_rate);
opt_state = Optimisers.setup(opt, ps)
train_state = Training.TrainState(model, ps, st, opt)

function loss_function(model, ps, st, (x,y))
    y_hat,st = model(x,ps,st)
    loss = mean(abs2,y_hat - y)

    stat = NamedTuple()
    return (loss,st,stat)
end


function train!(train_state, device, in_channels, out_channels, batchsize)
    sum_loss = 0.0
    println("in train!")
    for i = 1:1000^5
        x = randn(Float32,64,64,in_channels,batchsize) |> device
        y = randn(Float32,64,64,out_channels,batchsize) |> device
        
        _, loss, _, train_state = Training.single_train_step!(
            AutoZygote(), loss_function, (x,y), train_state);

        sum_loss += loss

        # explicetly run julia's garbage collector
        GC.gc()
        # print Max. RSS in MiB
        @printf "%d: Max. RSS:  %9.3f MiB\n" i Sys.maxrss()/2^20
        flush(stdout)
    end
    return sum_loss
end

println("about to call train!")
sum_loss = train!(train_state, device, in_channels, out_channels, batchsize)
@show sum_loss
