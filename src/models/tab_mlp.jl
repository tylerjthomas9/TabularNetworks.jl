# inspired by pytorch-widedeep
# https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tab_mlp.py

using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA

# Struct to define hyperparameters
@with_kw mutable struct Args
    lr::Float64 = 0.1		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_size::Vector{Int64} = [64, 64] # Size of hidden layers
    activation_function = relu
end

# build model
# TODO: variable number of dense layers
function tab_mlp(; cont_var=10::Int64, cat_var=10::Int64, n_outputs=2::Int64)
    return Chain(
        Parallel(vcat,
            Chain(Dense(cont_var, 32),
                BatchNorm(32, relu)),
            Dense(cat_var, 32, relu)),
        Dropout(0.10),
        Dense(64, 64, relu),
        Dropout(0.10),
        Dense(64, n_outputs)
    )
en