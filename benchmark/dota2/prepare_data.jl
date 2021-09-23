# dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
using CSV
using DataFrames
using Flux: onehotbatch
using Flux.Data: DataLoader
using Parameters: @with_kw
using StatsBase
using Statistics: var
include("../../src/preprocessing/ohe_cat_features.jl")

@with_kw struct TabularDataloaders
    train_loader
    test_loader
    cat_dict
end

function getdata()
    train = CSV.File("input/dota2Train.csv", delim=",", header=0) |> DataFrame
    test = CSV.File("input/dota2Test.csv", delim=",", header=0) |> DataFrame

    X_train_cat = select(train, Not([:Column1, :Column2, :Column3])) |> Array{Float32} |> transpose |> copy
    X_test_cat = select(test, Not([:Column1, :Column2, :Column3])) |> Array{Float32} |> transpose |> copy

    X_train_cont = train[!, [:Column2, :Column3]]|> Array{Float32} |>  transpose
    X_test_cont = test[!, [:Column2, :Column3]] |> Array{Float32} |>  transpose

    # drop cat variables with 1 category
    cat_var = var.(eachrow(X_train_cat))
    X_train_cat = X_train_cat[cat_var.!=0.0, :]
    X_test_cat = X_test_cat[cat_var.!=0.0, :]

    # get categorical dictionary
    cat_dict = get_category_dict(X_train_cat)

    # standardize continious data
    dt = fit(ZScoreTransform, X_train_cont, dims=2)
    X_train_cont = StatsBase.transform(dt, X_train_cont)
    X_test_cont = StatsBase.transform(dt, X_test_cont)

    # One-hot-encode the labels
    y_train = onehotbatch(train[!, :Column1], [-1.0, 1.0])
    y_test = onehotbatch(test[!, :Column1], [-1.0, 1.0])

    # ohe categorical columns
    # results in vector of ohe columns
    X_train_cat = ohe_cat_features(X_train_cat, cat_dict)
    X_test_cat = ohe_cat_features(X_test_cat, cat_dict)

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((y_train, X_train_cont, X_train_cat...), batchsize=2048,  shuffle=true)
    test_loader = DataLoader((y_test, X_test_cont, X_test_cat...), batchsize=2048, shuffle=false)

    tabular_dataloaders = TabularDataloaders(train_loader, test_loader, cat_dict)
    return tabular_dataloaders
end