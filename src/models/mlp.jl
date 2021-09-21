using Flux
using Flux: @functor
using Parameters: @with_kw
using FastAI.Models: TabularModel, tabular_embedding_backbone
include("../layers/categorical_embeddings.jl")


@with_kw mutable struct MLPArgs
    embedding_cardinalities::Vector{Tuple{Int64, Int64}} 
    cont_columns::Tuple
    output_dim::Int64 = 2
    lr::Float64 = 1e-2		# learning rate
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_dims::Vector{Int64} = [64, 64] # Size of dense hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    batchnorm::Bool = true # batchnorm on dense layers
    activation_function = Flux.relu
    output_activation = Flux.sigmoid
    seed::Int64 = 42
end


struct MLP
    cat_backbone
    cont_backbone
    output
end
@functor MLP

MLP(args::MLPArgs) = MLP(
    tabular_embedding_backbone(args.embedding_cardinalities)
    BatchNorm(length(cont_columns))
    Dense(args.hidden_dims[end], args.output_dim, args.output_activation)
)

function (mlp::MLP)(cat, cont)
    TabularModel(cat_backbone, cont_backbone, output)
end





