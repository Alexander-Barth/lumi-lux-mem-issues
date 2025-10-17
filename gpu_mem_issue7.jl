parallel = get(ENV,"PARALLEL","false") == "true"


using ComponentArrays
using DIVAnd
using Dates
using JLD2
using LinearAlgebra
using Lux
using MLUtils: DataLoader
using Optimisers
using Printf
using Random
using Statistics
using Test
using Zygote

include("my_unet.jl")

if parallel
    import MPI
end

if !isnothing(Sys.which("nvidia-smi"))
    import CUDA, cuDNN
else
    import AMDGPU
    @show AMDGPU.devices()
    AMDGPU.versioninfo()
end


struct ConditionalFlowMatching{TC}  <: Lux.AbstractLuxLayer
    net::TC
    xin::Int
    nfrequencies::Int
end
export ConditionalFlowMatching

Lux.initialparameters(rng::AbstractRNG, nf::ConditionalFlowMatching) = Lux.initialparameters(rng, nf.net)
Lux.initialstates(rng::AbstractRNG, nf::ConditionalFlowMatching) = Lux.initialstates(rng, nf.net)


function tt(nf,t::Number,x::TA) where TA <: AbstractArray{T} where T
    sz = size(x)
    t_embed = similar(x,(sz[1:end-2]...,2*nf.nfrequencies,sz[end]))
    for i = 1:nf.nfrequencies
        selectdim(t_embed,ndims(x)-1,2i-1) .= cos.(T(pi)*t)
        selectdim(t_embed,ndims(x)-1,2i)   .= sin.(T(pi)*t)
    end
    return t_embed
end
Zygote.@nograd tt

function tt(nf,t,x::TA) where TA <: AbstractArray{T} where T
    sz = size(x)
    t_embed = similar(x,(sz[1:end-2]...,2*nf.nfrequencies,sz[end]))

    tt = reshape(t,(ntuple(i -> 1, length(sz[1:end-2]))...,length(t)))

    for i = 1:nf.nfrequencies
        selectdim(t_embed,ndims(x)-1,2i-1) .= cos.(T(pi)*tt)
        selectdim(t_embed,ndims(x)-1,2i)   .= sin.(T(pi)*tt)
    end
    return t_embed
end


function (nf::ConditionalFlowMatching)((t,x,aux...),ps,st)
    t_embed = tt(nf,t,x)
    return nf.net(cat(x,t_embed,aux...,dims=ndims(x)-1),ps,st)
end


function RK4!(f!,x,p,t,h,tmp)
    k1,k2,k3,k4 = tmp
    f!(k1, x, p, t)
    @. k1 *= h
    f!(k2, x + k1/2, p, t + h/2)
    @. k2 *= h
    f!(k3, x + k2/2, p, t + h/2)
    @. k3 *= h
    f!(k4, x + k3, p, t + h)
    @. k4 *= h
    @. x += (k1 + 2*k2 + 2*k3 + k4)/6
end

function f!(dx, x, (nf,ps,st,aux...), t)
    dx.x,st = nf((t,x.x,aux...),ps,st)
    if hasproperty(x,:J)
        for b = 1:length(x.J)
            auxb = ntuple(i -> aux[i][:,:,:,b],length(aux))
            jaco = Zygote.jacobian(x -> nf((t,x,auxb...),ps,st)[1][:],x.x[:,b:b])[1]
            dx.J[b] = tr(jaco)/100
        end
    end
end

function integ!(f!,x,tspan,(nf,ps,st,aux...); Nmax=20,xall = nothing)
    h = (tspan[2]-tspan[1])/Nmax

    dx = deepcopy(x)
    t = tspan[1]

    tmp = (deepcopy(x),deepcopy(x),deepcopy(x),deepcopy(x))
    for n = 1:Nmax
        RK4!(f!,x,(nf,ps,st,aux...),t,h,tmp)
        t += h

        if !isnothing(xall)
            selectdim(xall,ndims(x.x)+1,n) .= Array(x.x)
        end
    end

    return x
end

decode((nf,ps,st),x0,aux...; kwargs...) = integ!(f!,ComponentArray(x=copy(x0)),(0,1),(nf,ps,st,aux...); kwargs...).x

function loss_function(nf, ps, st, (t,z,x,aux,mask); σₘᵢₙ = 1f-4)
    y = @. (1 - (1 - σₘᵢₙ) * t) * z  +  t*x
    u = @. x - (1 - σₘᵢₙ) * z

    up,st = nf((t,y,aux...),ps,st)
    loss =
        if isnothing(mask)
            mean((up - u).^2)
        else
            mean(mask .* (up - u).^2)
        end

    stat = NamedTuple()
    return (loss,st,stat)
end


function _as_tuple(batch_cpu)
    if batch_cpu isa Tuple
        return batch_cpu
    else
        return (batch_cpu,)
    end
end

function train(fun::Function,device::Function,nf,dataloader,opt,nepochs;
               T = Float32,
               losses = T[],
               mask = nothing,
               backend = nothing,
               ddp = !isnothing(backend),
               )

    local_rank = if isnothing(backend)
        0
    else
        DistributedUtils.local_rank(backend)
    end

    ps, st = Lux.setup(Random.default_rng(), nf)

    if ddp
        println("use DDP (local rank $local_rank)")
        ps = DistributedUtils.synchronize!!(backend, ps)
        st = DistributedUtils.synchronize!!(backend, st)
        opt = DistributedUtils.DistributedOptimizer(backend, opt)
    end

    ps = ps |> device
    st = st |> device

    opt_state = Optimisers.setup(opt, ps)
    train_state = Training.TrainState(nf, ps, st, opt)

    if ddp
        opt_state = DistributedUtils.synchronize!!(backend, opt_state)
    end

    batch_cpu = first(dataloader)
    (x,aux...) = device(_as_tuple(batch_cpu))
    t = zeros(T,(ntuple(i -> 1,ndims(x)-1)...,size(x)[end])) |> device
    z = zeros(T,size(x)) |> device

    for epoch in 1:nepochs
        epoch_loss = T(0)
        epoch_count = 0

        for batch_cpu in dataloader
            (x,aux...) = device(_as_tuple(batch_cpu))
            @assert size(x)[end] == size(t)[end]
            rand!(t)
            randn!(z)

            _, loss, _, train_state = Training.single_train_step!(
                AutoZygote(), loss_function, (t,z,x,aux,mask), train_state);

            epoch_loss += loss
            epoch_count += 1
        end

        loss = epoch_loss/epoch_count
        push!(losses, loss)

        fun((epoch,nf,ps,train_state.states,train_state,loss))
    end
    return ps,train_state.states,losses,train_state
end


basedir = expanduser("~/Data/Biscay/")


varnames_x = ["chl","thetao"]
varnames_y = ["uo","vo"]


#---
local_rank = 0
if parallel
    const backend_type = MPIBackend
    DistributedUtils.initialize(backend_type)
    backend = DistributedUtils.get_distributed_backend(backend_type)
    local_rank = DistributedUtils.local_rank(backend)
else
    backend = nothing
end

@show parallel

timestamp = Dates.format(Dates.now(),"yyyy-mm-ddTHHMMSS")
outdir = joinpath(basedir,"Results-fm",timestamp)

mkpath(outdir)
T = Float32

tindex = 1:10000

println("output directory: ",outdir)
println("using $(Threads.nthreads()) thread(s)")

if local_rank == 0
    cp(@__FILE__,joinpath(outdir,basename(@__FILE__)),force=true)
    if haskey(ENV,"SLURM_JOB_ID")
        cp(@__FILE__,"script-" * ENV["SLURM_JOB_ID"] * ".jl",force=true)
    end
end

function scale(x,sx,mx,time)
    xa = copy(x)
    for n = 1:size(x,4)
        xa[:,:,:,n] .= (x[:,:,:,n] .- mx) ./ sx
    end
    return xa
end


function invscale(xa,sx,mx,time)
    x = copy(xa)
    for n = 1:size(x,4)
        x[:,:,:,n] .= xa[:,:,:,n] .* sx .+ mx
    end
    return x
end

function prep_data(varnames)
    time = 1:10000
    xa = rand(Float32,64,64,length(varnames),10000)
    mx = mean(xa,dims=4);
    sx = std(xa,dims=4);
    xa = scale(xa,sx,mx,time)
    return xa,sx,mx,time
end


function prep_test_data(varnames,sx,mx)
    time = 1:1000
    xa = rand(Float32,64,64,length(varnames),1000)
    xa = scale(xa,sx,mx,time)
    return xa,time
end

function validate(device,(model,ps,st),dataloader_test,my,sy,mask)
    st_test = Lux.testmode(st)

    sy_d = sy |> device
    my_d = my |> device

    Nens = 64;

    sumd = 0.0
    sumd2 = 0.0
    sumd_count = 0

    for batch in dataloader_test
        y,aux... = batch |> device;
        y_hat_ens = [decode((nf,ps,st_test),device(randn(T,size(y))),aux...) for k in 1:Nens]

        y_hat = mean(y_hat_ens);
        y_hat_std = std(y_hat_ens);

        y_ = invscale(y,sy_d,my_d,[]);
        y_hat_ = invscale(y_hat,sy_d,my_d,[]);

        y_isvalid = cpu(mask[:,:,:,ones(Int,size(y_,4))]);

        sumd += sum(y_hat_[y_isvalid] - y_[y_isvalid])
        sumd2 += sum(abs2,y_hat_[y_isvalid] - y_[y_isvalid])
        sumd_count += sum(y_isvalid)
    end
    bias = sumd / sumd_count
    RMSE = sqrt(sumd2 / sumd_count)
    return RMSE,bias
end


lon = 1:64
lat = 1:64

xa,sx,mx,time = prep_data(varnames_x);
ya,sy,my,time = prep_data(varnames_y);


xa_test,time_test = prep_test_data(varnames_x,sx,mx);
ya_test,time_test = prep_test_data(varnames_y,sy,my);

in_channels = size(ya,3) + size(xa,3)
channels = (64,128,256)
activation = selu
out_channels = size(ya,3)

nfrequencies = 3
in_channels += 2*nfrequencies

model = genmodel(;in_channels,
         channels,
         activation,
         out_channels,
         out_activation = identity,
         pool = MaxPool,
         head = [],
         )

nf = ConditionalFlowMatching(model, in_channels, nfrequencies)

batchsize = 64*2

mask = isfinite.(xa[:,:,:,1:1]);

buffer = true
partial = false

ya[isnan.(ya)] .= 0
ya_test[isnan.(ya_test)] .= 0

dataloader = DataLoader((ya,xa); batchsize, buffer, partial, shuffle = true);

dataloader_test = DataLoader((ya_test,xa_test); batchsize);



nepochs = 1000
#nepochs = 100
#nepochs = 10
device = gpu_device()
cpu = cpu_device()
learning_rate = 0.001
learning_rate = 1e-4
learning_rate = 9e-4
learning_rate = 5e-4
T = Float32
losses = []


@info "parameters" nepochs batchsize learning_rate channels activation nfrequencies

ps, st = device.(Lux.setup(Random.default_rng(), nf));

opt = AdamW(learning_rate);

y_batch,aux_batch... = device.(first(dataloader));

# test drive
out, = nf((0,y_batch,aux_batch...),ps,Lux.testmode(st));

size(out)

mask = mask |> device;

@show outdir
@show backend

error_stat = []

ps,st,losses = @time train(device,nf,dataloader,opt,nepochs; T, losses, mask, backend) do (nn,nf,ps,st,train_state,lossval)
    @show nn,lossval

    if (nn % 10 == 0 && local_rank == 0)
        model_fname = joinpath(outdir,"model-checkpoint-" * @sprintf("%05d",nn) * ".jld2")

        activation_str = string(activation);
        jldsave(model_fname;
                mx, sx, my, sy,
                losses, learning_rate, nepochs, batchsize,
                channels, in_channels, activation_str, out_channels,
                buffer, partial,
                ps = cpu(ps),
                st = cpu(st),
                )
    end

    if (nn % 10 == 0 && local_rank == 0)
        RMSE,bias = validate(device,(model,ps,st),dataloader_test,my,sy,mask)
        @show RMSE, bias
        push!(error_stat,(RMSE, bias))
    end
    
    if any(m -> m[1].name == "AMDGPU", Base.loaded_modules)
        AMDGPU.synchronize();
        GC.gc()
    end
end;
