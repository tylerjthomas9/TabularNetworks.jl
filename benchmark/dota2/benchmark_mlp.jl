# dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
include("../../src/models/mlp.jl")
include("../../src/utils/metrics.jl")
include("./prepare_data.jl")
using CSV
using ProgressMeter
using Flux.Losses: logitcrossentropy
using Random


function train(; kws...)
    # load hyperparameters
    args = MLPArgs(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU setup
    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    device=cpu

    # Create test and train dataloaders
    train_loader, test_loader = getdata(args)

    # Construct model
    #model = mlp(args) |> device
    model = MLP(args) |> device
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
            # model(X_cat, X_cont)
            pred = max.(model(X_cat, X_cont), 1e-6)
            print(pred)
            loss = logitcrossentropy(pred, y)
            println(loss)
            gs = gradient(() -> logitcrossentropy(pred, y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # get train/test Losses
        loss_and_accuracy(train_loader, model, device; set="train")
        loss_and_accuracy(test_loader, model, device; set="test")
    end
end


train(; batchsize=2048, hidden_dims=[256, 256],
    cont_input_dim=10, cat_input_dim=10, output_dim=2)