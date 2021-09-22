# dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
using CSV
using DataFrames
using Flux: onehotbatch
using Flux.Data: DataLoader
using StatsBase
include("../../src/preprocessing/ohe_cat_features.jl")


function getdata()
    train = CSV.File("input/dota2Train.csv", delim=",", header=0) |> DataFrame
    test = CSV.File("input/dota2Test.csv", delim=",", header=0) |> DataFrame

    X_train_cat = select(train, Not([:Column1, :Column2, :Column3])) |> Array{Float32} |> transpose |> copy
    X_test_cat = select(test, Not([:Column1, :Column2, :Column3])) |> Array{Float32} |> transpose |> copy

    X_train_cont = train[!, [:Column2, :Column3]]|> Array{Float32} |>  transpose
    X_test_cont = test[!, [:Column2, :Column3]] |> Array{Float32} |>  transpose

    # standardize continious data
    dt = fit(ZScoreTransform, X_train_cont, dims=2)
    X_train_cont = StatsBase.transform(dt, X_train_cont)
    X_test_cont = StatsBase.transform(dt, X_test_cont)

    # One-hot-encode the labels
    y_train = onehotbatch(train[!, :Column1], [-1.0, 1.0])
    y_test = onehotbatch(test[!, :Column1], [-1.0, 1.0])

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((X_train_cat, X_train_cont, y_train), batchsize=2048,  shuffle=true)
    test_loader = DataLoader((X_test_cat, X_test_cont, y_test), batchsize=2048, shuffle=false)

    return train_loader, test_loader
end