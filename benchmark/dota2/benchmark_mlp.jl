using CUDA
using DataAugmentation: Categorify
using FastAI
using Flux.Losses: logitcrossentropy
using Flux: DataLoader
using ProgressMeter
using Random
include("../../src/models/mlp.jl")
include("../../src/utils/metrics.jl")
include("../../src/preprocessing/ohe_cat_features.jl")
include("./prepare_data.jl")


function run_benchmark(; kws...)
    Random.seed!(42)

    # Create test and train dataloaders
    train_loader, test_loader = getdata()

    # get categorical cardinalities
    cat_dict = get_category_dict(train_loader.data[1])
    cardinalities = collect(map(col -> length(cat_dict[col]), collect(1:length(cat_dict))))
    embedding_dims = FastAI.Models.get_emb_sz(cardinalities)

    # get continious variable input dims
    cont_input_dim = size(train_loader.data[2], 2)

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
        @showprogress 1 "Training... " for (X_cat, X_cont, y) in train_loader
            X_cat = ohe_cat_features(X_cat, cat_dict) |> device
            println(typeof(X_cat))
            X_cont = X_cont |> device
            y = y |> device
            model(X_cat, X_cont)
            # gs = gradient(() -> logitcrossentropy(model(X_cat, X_cont), y), ps) # compute gradient
            # Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # get train/test losses
        loss_and_accuracy(train_loader, cat_dict, model, device; set="train")
        loss_and_accuracy(test_loader, cat_dict, model, device; set="test")
    end
end


a = run_benchmark(; hidden_dims=(128, 64))
