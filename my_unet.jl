using Lux: True, False

function showsize(msg)
    function f(x)
        #@show msg,size(x)
        return x
    end
end

n_out_channels(::typeof(+),c1,c2) = c1

cat_channels(x,y) = cat(x,y,dims=3)
n_out_channels(::typeof(cat_channels),c1,c2) = c1+c2

@inline noskip(x,y) = x
n_out_channels(::typeof(noskip),c1,c2) = c1

function CatChannelwise(xs)
    xt = cat(xs...,dims=ndims(xs[1])-1)
    return xt
end

function DConv(ks,(in,out), σ = identity; kwargs...)
    [
        Conv(ks,in => out; use_bias = false, kwargs...),
        BatchNorm(out),
        σ,
        Conv(ks,out => out; use_bias = false, kwargs...),
        BatchNorm(out),
        σ,
    ]
end

function ResDConv(ks,(in,out), σ = identity; kwargs...)
    [
        Conv(ks,in => out; use_bias = false, kwargs...),
        BatchNorm(out),
        σ,
        SkipConnection(Chain(
            Conv(ks,out => out; use_bias = false, kwargs...),
            BatchNorm(out),
            σ),+),
    ]
end


function block(channels; ks = 3, activation=relu, connection = cat_channels, pool = MaxPool)
    conn,other_conn =
        if connection isa Tuple
            connection[1],connection[2:end]
        else
            connection,connection
        end

    if length(channels) == 2
        inner_block = []
    else
        inner_block = block(channels[2:end]; ks, activation, connection=other_conn, pool = pool)
    end

    nout = n_out_channels(conn,channels[1],channels[1])

    return [
        SkipConnection(
            Chain(
                pool((2,2)),
                ResDConv((ks,ks),channels[1]=>channels[2],activation,pad = SamePad(),cross_correlation=True())...,
                #showsize("before 3 $(channels[2])"),
                inner_block...,
                ConvTranspose((2,2),channels[2]=>channels[1],activation,pad=SamePad(),stride=2,cross_correlation=True()),
        ),
            conn),
        ResDConv((ks,ks),nout=>channels[1],activation,pad = SamePad(),cross_correlation=True())...,
    ]
end



function genmodel(;in_channels = 1,
                  channels = (64,128,256,512,1024),
                  activation = relu,
                  kernel_size = 3,
                  kernel_size_out = 1,
                  out_channels = 1,
                  connection = cat_channels,
                  out_activation = identity,
                  pool = MaxPool,
                  head = [CatChannelwise],
                  )

    ks = kernel_size
    inner_block = block(channels; activation, ks, connection, pool)

    model = Chain(
        head...,
        ResDConv((kernel_size,kernel_size),in_channels=>channels[1],activation,pad = SamePad(),cross_correlation=True())...,
        #showsize("before 0"),
        inner_block...,
        Conv((kernel_size_out,kernel_size_out),channels[1]=>out_channels,out_activation,pad = SamePad(),cross_correlation=True()),
    )
end


