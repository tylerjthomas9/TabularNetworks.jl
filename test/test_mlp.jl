include("../src/models/mlp.jl")
using ProgressMeter
using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using StatsBase

function getdata(args,)
    train = zeros(5, 64)
    train[:, 33:end] .= 1.0


    y_train = zeros(64)
    y_train[33:end] .= 1.0

    X_train_cat = train[1:2, :] |> Array{Float32} 
    X_test_cat = X_train_cat

    X_train_cont = train[3:5, :] |> Array{Float32} 
    X_test_cont = X_train_cont


    # One-hot-encode the labels
    y_train = onehotbatch(y_train, [0, 1])
    y_test = onehotbatch(y_train, [0, 1])

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((X_train_cont, X_train_cat, y_train),  shuffle=true)
    test_loader = DataLoader((X_test_cont, X_test_cat, y_test), batchsize=args.batchsize, shuffle=false)

    return train_loader, test_loader
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (X_cont, X_cat, y) in data_loader
        y = y |> device
        x = map(device, (X_cont, X_cat))
        pred = model(x)
        ls += logitcrossentropy(pred, y, agg=sum)
        acc += sum(onecold(cpu(pred)) .== onecold(cpu(y)))
        num +=  size(y, 2)
    end
    return ls / num, acc / num
end


function train(; kws...)
    args = MLPArgs(; kws...) # collect options in a struct for convenience

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
    model = mlp(args; cont_var=2, cat_var=114) |> device
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

        # get train/test Losses
        train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
    end
end


train(; batchsize=16, hidden_sizes=[16, 8], cat_dense_size=16)