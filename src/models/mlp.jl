using Flux
using Flux: @functor
using Parameters: @with_kw
using FastAI
using FastAI.Models: TabularModel
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


struct MLP
    model
end
@functor MLP

MLP(args::MLPArgs) = MLP(
    TabularModel(
                tabular_embedding_backbone(args.embedding_dims),
                BatchNorm(args.cont_input_dim),
                Dense(args.hidden_dims[end], args.output_dim, args.output_activation);
                layersizes = args.hidden_dims,
                batchnorm = args.batchnorm,
                activation = args.activation,
                linear_first = args.linear_first
    )
)

function (mlp::MLP)(X_cat, X_cont)
    mlp.model(X_cat, X_cont)
end





