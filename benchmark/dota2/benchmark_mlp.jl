using CUDA
using FastAI
using Flux.Losses: logitcrossentropy
using Flux: DataLoader
using ProgressMeter
using Random
using Zygote
include("../../src/models/mlp.jl")
include("../../src/utils/metrics.jl")
include("./prepare_data.jl")
include("../../src/preprocessing/ohe_cat_features.jl")


function run_benchmark(; kws...)
    Random.seed!(42)

    # Create test and train dataloaders
    tabular_dataloaders = getdata()

    # get categorical cardinalities
    cardinalities = collect(map(col -> length(tabular_dataloaders.cat_dict[col]), 
                collect(1:length(tabular_dataloaders.cat_dict))))
    embedding_dims = FastAI.Models.get_emb_sz(cardinalities)

    # get continious variable input dims
    cont_input_dim = size(tabular_dataloaders.train_loader.data[2], 1)

    # load hyperparameters
    args = MLPArgs(; embedding_dims=embedding_dims, cont_input_dim=cont_input_dim, kws...)

    # GPU setup
    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end


    # Construct model
    model = MLP(args) |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.lr)
    
    ## Training
    for epoch in 1:args.epochs
        @info "Epoch: $epoch"
        progress = Progress(length(tabular_dataloaders.train_loader))
        for (y, X_cont, X_cat...) in tabular_dataloaders.train_loader
            # send batch  to device
            X_cat = X_cat |> device
            X_cont = X_cont |> device
            y = y |> device

            loss, back = Flux.pullback(() -> logitcrossentropy(model(X_cat, X_cont), y), ps) 
            gs = back(one(loss)) # calculate gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters

            next!(progress; showvalues=[(:loss, loss)])  # show loss
        end

        # get train/test losses
        loss_and_accuracy(tabular_dataloaders.train_loader, model, device; set="train")
        loss_and_accuracy(tabular_dataloaders.test_loader, model, device; set="test")
    end
end


a = run_benchmark(; hidden_dims=(256, 128))