include("../../src/models/mlp.jl")
include("../../src/utils/metrics.jl")
include("./prepare_data.jl")
using CUDA
using DataAugmentation: Categorify
using FastAI
using Flux.Losses: logitcrossentropy
using ProgressMeter
using Random

function train(; kws...)
    # Create test and train dataloaders
    train, test = getdata()

    # set column values
    target = :Column1
    cont_cols = (:Column2, :Column3)
    cat_cols = propertynames(train.table)[4:end] |> Tuple
    cat_dict = FastAI.gettransformdict(train, Categorify, cat_cols)

    # split data on target
    train_split = mapobs(row -> (row, row[target]), train)
    test_split = mapobs(row -> (row, row[target]), test)

    # get categorical cardinalities
    cardinalities = collect(map(col -> length(cat_dict[col]), cat_cols))
    embedding_sizes = FastAI.Models.get_emb_sz(cardinalities)

    # load hyperparameters
    cat_input_dim = size(train_loader.data[1], 1)
    cont_input_dim = size(train_loader.data[2], 1)
    args = MLPArgs(; cont_input_dim=cont_input_dim, cat_input_dim=cat_input_dim, kws...)
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


    # Construct model
    model = MLP(args) |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.lr)
    
    ## Training
    for epoch in 1:args.epochs
        @info "Epoch: $epoch"
        @showprogress 1 "Training... " for (X_cat, X_cont, y) in train_loader
            X_cat = X_cat |> device
            X_cont = X_cont |> device
            y = y |> device
            gs = gradient(() -> logitcrossentropy(model(X_cat, X_cont), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # get train/test losses
        loss_and_accuracy(train_loader, model, device; set="train")
        loss_and_accuracy(test_loader, model, device; set="test")
    end
end


train(; hidden_dims=[256, ])
