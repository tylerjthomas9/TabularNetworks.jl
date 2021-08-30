# dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
include("../../src/models/mlp.jl")
include("./prepare_data.jl")
using CSV
using ProgressMeter
using Flux: onecold
using Flux.Losses: logitcrossentropy


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


train(; batchsize=8192, hidden_sizes=[256, 256], cat_dense_size=128)