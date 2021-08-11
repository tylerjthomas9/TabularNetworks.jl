# dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
include("../../src/models/tab_mlp.jl")
include("./prepare_data.jl")
using CSV
using ProgressMeter

function train(; kws...)
    args = TabMLPArgs(; kws...) # collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Create test and train dataloaders
    train_loader, test_loader = getdata(args)

    # Construct model
    model = tab_mlp(args; cont_var=2, cat_var=114) |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.lr)
    
    ## Training
    for epoch in 1:args.epochs
        println("Epoch: $epoch")
        @showprogress 1 "Training..." for (X_cont, X_cat, y) in train_loader
            y = y |> device
            x = map(device, (X_cont, X_cat))
            gs = gradient(() -> logitcrossentropy(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
    end
end


train(; batchsize=1024, hidden_sizes=[128, 128])