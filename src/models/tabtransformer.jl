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
include("../utils/mha.jl")

# Struct to define hyperparameters
@with_kw mutable struct TabTransfortmerArgs
    cat_input_dim::Int64
    cont_input_dim::Int64
    cat_hidden_dim::Int64 = 32
    cont_hidden_dim::Int64 = 32
    output_dim::Int64 = 2
    lr::Float64 = 0.1		# learning rate
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_dims::Vector{Int64} = [64, 64] # Size of hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = relu
    output_activation = sigmoid
    mha_heads::Int64 = 4
    mha_head_dims::Int64 = 5
    mha_output_dims::Int64 = 8
    mha_dropout::Float64 = 0.1
    transformer_dense_hidden_dim::Int64 = 128
    transformer_dense_dropout::Float64 = 0.1
    seed::Int64 = 42
end


struct Transformerlayer
    mha
    dense
end

@functor Transformerlayer

Transformerlayer(args::TabTransfortmerArgs) = Transformerlayer(
    MultiheadAttention(args.mha_heads, args.cat_input_dim, args.mha_head_dims, 
                                        args.mha_output_dims; future = true, pdrop = args.mha_dropout),
    Chain(Dense(args.cat_hidden_dim * args.mha_head_dims, args.transformer_dense_hidden_dim, args.activation_function), 
            Dropout(args.transformer_dense_dropout))
)

function (transformer_layer::Transformerlayer)(x)
    h1 = atten(transformer_layer.mha, x)
    h2 = transformer_layer.dense(h1)
    h3 = Softmax(vcat(h1, h2), dims=1)
    tab_transformer.output(h3)
end

struct TabTransformer
    input
    dense
    output
end
@functor TabTransformer

TabTransformer(args::TabTransfortmerArgs) = TabTransformer(
    Parallel(
        vcat,
        Transformerlayer(args),
        Chain(
            Dense(args.cont_input_dim, args.cont_hidden_dim, args.activation_function),
            BatchNorm(args.cont_hidden_dim, )
            ),
        ),
    Chain([Dense(if ix==1 args.transformer_dense_hidden_dim + args.cont_hidden_dim else args.hidden_dims[ix-1] end, 
        args.hidden_dims[ix], args.activation_function) for ix in 1:size(args.hidden_dims, 1)]...),
    Dense(args.hidden_dims[end], args.output_dim, args.output_activation)
)

function (tab_transformer::TabTransformer)(cat, cont)
    h1 = tab_transformer.input(cat, cont)
    h2 = tab_transformer.dense(h1)
    tab_transformer.output(h2)
end



