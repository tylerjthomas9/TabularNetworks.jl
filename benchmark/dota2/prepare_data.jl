using CSV
using DataFrames
using Flux: onehotbatch
using Flux.Data: DataLoader


function getdata(args,)
    train = CSV.File("input/dota2Train.csv", delim=",", header=0) |> DataFrame
    test = CSV.File("input/dota2Test.csv", delim=",", header=0) |> DataFrame

    X_train_cat = select(train, Not([:Column1, :Column2, :Column3])) |> Array{Float32} |>  transpose
    X_test_cat = select(test, Not([:Column1, :Column2, :Column3])) |> Array{Float32} |> transpose

    X_train_cont = train[!, [:Column2, :Column3]]|> Array{Float32} |>  transpose
    X_test_cont = test[!, [:Column2, :Column3]] |> Array{Float32} |>  transpose

    # One-hot-encode the labels
    y_train = onehotbatch(train[!, :Column1], [-1, 1])
    y_test = onehotbatch(test[!, :Column1], [-1, 1])

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((X_train_cont, X_train_cat, y_train),  shuffle=true)
    test_loader = DataLoader((X_test_cont, X_test_cat, y_test), batchsize=args.batchsize, shuffle=false)

    return train_loader, test_loader
end