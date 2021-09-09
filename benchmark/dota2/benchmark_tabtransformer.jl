# dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
include("../../src/models/tabtransformer.jl")
include("../../src/utils/metrics.jl")
include("./prepare_data.jl")
using CSV
using ProgressMeter
using Flux.Losses: logitcrossentropy


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
    train_loader, test_loader = getdata(args)

    # Construct model
    #model = TabTransfortmerArgs(args; cont_var=2, cat_var=114) |> device
    model = TabTransfortmerArgs(args; cont_var=10, cat_var=10) |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.lr)
    
    ## Training
    for epoch in 1:args.epochs
        println("Epoch: $epoch")
        @showprogress 1 "Training..." for (X_cont, X_cat, y) in train_loader
            y = y |> device
            X_cat = X_cat |> device
            X_cont = X_cont |> device
            gs = gradient(() -> logitcrossentropy(model(X_cat, X_cont), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # get train/test Losses
        loss_and_accuracy(train_loader, model, device; set="train")
        loss_and_accuracy(test_loader, model, device; set="test")
    end
end


train(; batchsize=2048, hidden_sizes=[256, 256], cat_dense_size=128)