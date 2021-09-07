include("./tabtransformer.jl")
function getdata(args,)
    # generate fake data
    n_obs = 1_000
    X_cont = rand(10, n_obs) |> Array{Float32}
    X_cat = rand(0:1, 10, n_obs) |> Array{Float32}

    y = rand(0:1, n_obs) |> Array{Float32}

    # One-hot-encode the labels
    y = Flux.onehotbatch(y, 0:1)

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((X_cont, X_cat, y), batchsize=args.batchsize, shuffle=true)

    return train_loader
end


function train(; kws...)
    args = TabTransfortmerArgs(; kws...) # collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Create test and train dataloaders
    train_loader = getdata(args)

    # Construct model
    model = tab_mlp(args) |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.lr)
    
    ## Training
    for epoch in 1:args.epochs
        println("Epoch: $epoch")
        for (X_cont, X_cat, y) in train_loader
            y = y |> device
            x = map(device, (X_cont, X_cat))
            gs = gradient(() -> logitcrossentropy(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
    end
end


train(; batchsize=128)