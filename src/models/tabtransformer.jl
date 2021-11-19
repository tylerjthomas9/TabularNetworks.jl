# https://arxiv.org/abs/2012.06678

using Flux
using Flux: @functor
using Parameters: @with_kw
using FastAI
import FastAI.Models.tabular_embedding_backbone
include("../layers/transformer_block.jl")
include("../layers/categorical_embeddings.jl")

# Struct to define hyperparameters
@with_kw mutable struct TabTransfortmerArgs
    embedding_dims::Vector{Tuple{Int64, Int64}} 
    mha_input_dims::Int64
    cont_input_dim::Int64
    output_dim::Int64 = 2
    lr::Float64 = 3e-4		# learning rate
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_dims::Vector{Int64} = [128, 64] # Size of hidden layers
    embedding_dropout::Float64 = 0.10 # dropout for categorical embeddin
    batchnorm::Bool = true # batchnorm on dense layers
    linear_first::Bool = true # linear layer before or after batch norm
    activation = Flux.relu
    output_activation = sigmoid
    mha_heads::Int64 = 8 
    mha_head_dims::Int64 = 32
    transformer_dropout::Float64 = 0.1
    transformer_dense_hidden_dim::Int64 = 64
    transformer_blocks::Int64 = 6 
end


"""
Create a series of dense layers following the
embedding and continious data vcat

https://github.com/FluxML/FastAI.jl/blob/master/src/models/tabularmodel.jl
"""
function dense_layers(args)
    layers = []
    first_layer = linbndrop(args.mha_input_dims + args.cont_input_dim, first(args.hidden_dims); 
                    use_bn=args.batchnorm, p=args.dropout, lin_first=args.linear_first, 
                    act=args.activation)
    push!(layers, first_layer)

    for (isize, osize) in zip(args.hidden_dims[1:(end-1)], args.hidden_dims[2:end])
        layer = linbndrop(isize, osize; use_bn=args.batchnorm, p=args.dropout, 
                        lin_first=args.linear_first, act=args.activation)
        push!(layers, layer)
    end

    return Chain(layers...)

end

struct TabTransformer
    cat_embeddings
    cat_backbone
    cont_backbone
    dense
    output
end
@functor TabTransformer

TabTransformer(args::TabTransfortmerArgs) = TabTransformer(
    tabular_embedding_backbone(args.embedding_dims, args.embedding_dropout),
    Chain([TransformerBlock(args) for i in 1:args.transformer_blocks]..., 
            Flux.flatten),
    LayerNorm(args.cont_input_dim),
    dense_layers(args),
    Dense(args.hidden_dims[end], args.output_dim, args.output_activation)
)

function (t::TabTransformer)(X_cat, X_cont)
    h_cat = t.cat_embeddings(X_cat...)
    h_cat = t.cat_backbone(reshape(h_cat, (size(h_cat, 1), 1, size(h_cat, 2))))
    h_cont = t.cont_backbone(X_cont)
    h2 = t.dense(vcat(h_cat, h_cont))
    t.output(h2)
end



