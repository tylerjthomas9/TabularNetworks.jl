using Flux
using Flux: @functor
using Parameters: @with_kw
using FastAI
import FastAI.Models.tabular_embedding_backbone
include("../layers/categorical_embeddings.jl")


@with_kw mutable struct MLPArgs
    embedding_dims::Vector{Tuple{Int64, Int64}} 
    cont_input_dim::Int64
    output_dim::Int64 = 2
    lr::Float64 = 1e-2		# learning rate
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_dims::Tuple = (128, 64) # Size of dense hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    batchnorm::Bool = true # batchnorm on dense layers
    linear_first::Bool = true # linear layer before or after batch norm
    activation = Flux.relu
    output_activation = Flux.sigmoid
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


struct MLP
    cat_backbone
    cont_backbone
    dense
    output
end
@functor MLP

MLP(args::MLPArgs) = MLP(
    tabular_embedding_backbone(args.embedding_dims),
    BatchNorm(args.cont_input_dim),
    dense_layers(args),
    Dense(args.hidden_dims[end], args.output_dim, args.output_activation)
)

function (mlp::MLP)(X_cat, X_cont)
    h_cat = mlp.cat_backbone(X_cat...)
    h_cont = mlp.cont_backbone(X_cont)
    h2 = mlp.dense(vcat(h_cat, h_cont))
    mlp.output(h2)
end





