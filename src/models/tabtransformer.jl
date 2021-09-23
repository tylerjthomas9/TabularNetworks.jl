# https://arxiv.org/abs/2012.06678

using Flux
using Flux: @functor
using Parameters: @with_kw
using FastAI
using FastAI.Models: TabularModel
import FastAI.Models.tabular_embedding_backbone
include("../layers/categorical_embeddings.jl")

# Struct to define hyperparameters
@with_kw mutable struct TabTransfortmerArgs
    embedding_dims::Vector{Tuple{Int64, Int64}} 
    cont_input_dim::Int64
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


"""
Create a series of dense layers following the
embedding and continious data vcat

https://github.com/FluxML/FastAI.jl/blob/master/src/models/tabularmodel.jl
"""
function dense_layers(args)
    cat_input_dim = sum([i[1] for i in args.embedding_dims])

    layers = []
    first_layer = linbndrop(cat_input_dim + args.cont_input_dim, first(args.hidden_dims); 
                    use_bn=args.batchnorm, p=args.dropout_rate, lin_first=args.linear_first, 
                    act=args.activation)
    push!(layers, first_layer)

    for (isize, osize) in zip(args.hidden_dims[1:(end-1)], args.hidden_dims[2:end])
        layer = linbndrop(isize, osize; use_bn=args.batchnorm, p=args.dropout_rate, 
                        lin_first=args.linear_first, act=args.activation)
        push!(layers, layer)
    end

    return Chain(layers...)

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



