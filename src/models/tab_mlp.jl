# inspired by pytorch-widedeep
# https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tab_mlp.py

using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA

# Struct to define hyperparameters
@with_kw mutable struct Args
    lr::Float64 = 0.1		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers

    activation_function = relu
end


# set up continious, categorical inputs
function ContInput(;continious_variables=10::Int64,)
    return Chain(Dense(continious_variables, 32),
                BatchNorm(32, relu))
end

# TODO: Replace with categorical embedding
function CatInput(;ohe_size=10::Int64,)
    return Chain(Dense(ohe_size, 32, relu),
    )
end



# build model
function tab_mlp(; cont_var=10::Int64, cat_var=10::Int64, n_outputs=2::Int64)
    return Chain(
        Parallel(vcat,
            Chain(Dense(cont_var, 32),
                BatchNorm(32, relu)),
            Dense(cat_var, 32, relu)),
        Dropout(0.10),
        Dense(64, 64, relu),
        Dropout(0.10),
        Dense(64, n_outputs)
    )
end


test_cont = rand(10, 5)
test_cat = rand(0:1, 10, 5)
xs = map(cpu, (test_cont, test_cat))
tab_mlp()(xs)



function getdata(args,)
    # generate fake data
    n_obs = 5_000
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
    args = Args(; kws...) # collect options in a struct for convenience

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
    model = tab_mlp() |> device
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


train()