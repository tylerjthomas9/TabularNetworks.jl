# https://arxiv.org/abs/2012.06678

using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA
using NeuralAttentionlib
using NeuralAttentionlib: CausalMask, BiLengthMask, BatchedMask
using NNlib

# Struct to define hyperparameters
@with_kw mutable struct TabTransfortmerArgs
    cat_input_dim::Int64
    cont_input_dim::Int64
    cat_hidden_dim::Int64 = 32
    cont_hidden_dim::Int64 = 32
    lr::Float64 = 0.1		# learning rate
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_sizes::Vector{Int64} = [64, 64] # Size of hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = relu
    mha_heads::Int64 = 4
    mha_input_dims::Int64 = 7
    mha_head_dims::Int64 = 5
    mha_output_dims::Int64 = 8
end


struct TabTransformer
    cat_input
    cont_input
    dense
    output
end
@functor MLP

MLP(args::TabTransfortmerArgs) = TabTransformer(
    Dense(args.cat_input_dim, args.cat_hidden_dim, args.activation_function),
    Chain(
        Dense(args.cont_input_dim, args.cont_hidden_dim, args.activation_function),
        BatchNorm(args.cont_hidden_dim, )
        ),
    Chain([Dense(if ix==1 args.cat_hidden_dim + args.cont_hidden_dim else args.hidden_dims[ix-1] end, 
        args.hidden_dims[ix], args.activation_function) for ix in 1:size(args.hidden_dims, 1)]...),
    Dense(args.hidden_dims[end], args.output_dim, args.output_activation)
)

function (tab_transformer::TabTransformer)(cat, cont)
    h_cat = tab_transformer.cat_input(cat)
    h_cont = tab_transformer.cont_input(cont)
    h1 = vcat(h_cat, h_cont)
    h2 = tab_transformer.dense(h1)
    tab_transformer.output(h2)
end



