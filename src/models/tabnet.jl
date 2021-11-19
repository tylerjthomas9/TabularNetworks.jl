# https://arxiv.org/abs/1908.07442


using Flux
using Flux: @functor
using Parameters: @with_kw
using FastAI
import FastAI.Models.tabular_embedding_backbone
include("../layers/dense_layers.jl")
include("../layers/transformer_block.jl")
include("../layers/categorical_embeddings.jl")

# Struct to define hyperparameters
@with_kw mutable struct TabNetArgs
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


struct FeatureTransformer

end

struct TabNetEncoder

end

struct TabNetDecoder

end


struct TabNet
    cat_embeddings
    cat_backbone
    cont_backbone
    dense
    output
end
@functor TabNet