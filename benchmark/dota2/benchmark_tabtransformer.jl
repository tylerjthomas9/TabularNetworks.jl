include("../../src/models/tabtransformer.jl")
include("../../src/utils/metrics.jl")
include("./prepare_data.jl")
using CSV
using ProgressMeter
using Flux.Losses: logitcrossentropy
using Random
using CUDA


function train(; kws...)
    # Create test and train dataloaders
    train_loader, test_loader = getdata()

    # load hyperparameters
    cont_input_dim = size(train_loader.data[1], 1)
    cat_input_dim = size(train_loader.data[2], 1)
    args = TabTransfortmerArgs(; cont_input_dim=cont_input_dim, cat_input_dim=cat_input_dim, kws...)
    Random.seed!(args.seed)

    # GPU setup
    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        #CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    device = cpu


    # Construct model
    model = TabTransformer(args) |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.lr)
    
    ## Training
    for epoch in 1:args.epochs
        @info "Epoch: $epoch"
        @showprogress 1 "Training... " for (X_cont, X_cat, y) in train_loader
            y = y |> device
            X_cat = X_cat |> device
            X_cat_3d = zeros(args.cat_input_dim, 3, size(X_cat, 2)) |> Array{Float32} |> device
            for ix in 1:3
                X_cat_3d[:, ix, :] .= X_cat
            end
            println(size(X_cat_3d))
            X_cont = X_cont |> device
            model(X_cat, X_cont)
            gs = gradient(() -> logitcrossentropy(model(X_cat_3d, X_cont), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # get train/test losses
        loss_and_accuracy(train_loader, model, device; set="train")
        loss_and_accuracy(test_loader, model, device; set="test")
    end
end


train(; hidden_dims=[256, ])
