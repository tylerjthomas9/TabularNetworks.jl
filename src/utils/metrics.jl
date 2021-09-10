using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy


function loss_and_accuracy(data_loader::DataLoader, model, device; 
    set="train"::String, verbose=true)
    acc = 0
    ls = 0.0f0
    num = 0
    testmode!(model)
    for (X_cont, X_cat, y) in data_loader
        y = y |> device
        X_cat = X_cat |> device
        X_cont = X_cont |> device
        pred = model(X_cat, X_cont)
        ls += logitcrossentropy(pred, y, agg=sum)
        acc += sum(onecold(cpu(pred)) .== onecold(cpu(y)))
        num +=  size(y, 2)
    end
    testmode!(model, false)
    
    ls = ls / num
    acc = acc / num
    if verbose
        println("  $set loss = $ls, $set accuracy = $acc")
    end
    return ls, acc
end