# https://arxiv.org/abs/2012.06678

using Flux
using Flux: @functor
using Parameters: @with_kw
include("../layers/transformer.jl")

# Struct to define hyperparameters
@with_kw mutable struct TabTransfortmerArgs
    cat_input_dim::Int64
    cont_input_dim::Int64
    cat_hidden_dim::Int64 = 32
    cont_hidden_dim::Int64 = 32
    output_dim::Int64 = 2
    lr::Float64 = 1e-3		# learning rate
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_dims::Vector{Int64} = [64, 64] # Size of hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = Flux.relu
    output_activation = sigmoid
    mha_heads::Int64 = 4
    mha_head_dims::Int64 = 5
    transformer_dropout::Float64 = 0.1
    transformer_dense_hidden_dim::Int64 = 64
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
        Chain(Transformer(args), Flux.flatten),
        Chain(
            Dense(args.cont_input_dim, args.cont_hidden_dim, args.activation_function),
            BatchNorm(args.cont_hidden_dim, )
            ),
        ),
    Chain([Dense(if ix==1 args.cat_input_dim + args.cont_hidden_dim else args.hidden_dims[ix-1] end, 
        args.hidden_dims[ix], args.activation_function) for ix in 1:size(args.hidden_dims, 1)]...),
    Dense(args.hidden_dims[end], args.output_dim, args.output_activation)
)

function (tab_transformer::TabTransformer)(X_cat, X_cont)
    h1 = tab_transformer.input(X_cat, X_cont)
    h2 = tab_transformer.dense(h1)
    tab_transformer.output(h2)
end



