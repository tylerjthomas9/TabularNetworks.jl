# dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
using CSV
using DataFrames
using FastAI: TableDataset

function getdata()
    train = CSV.File("input/dota2Train.csv", delim=",", header=0) |> DataFrame |> TableDataset
    test = CSV.File("input/dota2Test.csv", delim=",", header=0) |> DataFrame |> TableDataset
    return train, test
end