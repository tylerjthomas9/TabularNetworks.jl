# https://arxiv.org/abs/2012.06678

using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA

# Struct to define hyperparameters
@with_kw mutable struct TabTransfortmerArgs
    lr::Float64 = 0.1		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_sizes::Vector{Int64} = [64, 64] # Size of hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = relu
end