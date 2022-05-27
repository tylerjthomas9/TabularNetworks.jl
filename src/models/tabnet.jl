# https://arxiv.org/abs/1908.07442


using Flux
using Flux: @functor
using Parameters: @with_kw
using FastAI
import FastAI.Models.tabular_embedding_backbone
include("../layers/transformer_block.jl")
include("../layers/categorical_embeddings.jl")

@with_kw mutable struct TabNetArgs
    mask_type::String = "sparsemax" #TODO: add entmax
    mha_input_dims::Int64
    momentum::Float64 = 0.01
    virtual_batch_size::Int64 = 128
end

"""
Ghost Batch Normalization
https://arxiv.org/abs/1705.08741

https://github.com/dreamquark-ai/tabnet/blob/40107a80b0be5ae865d945b85a52e5d99fc19a81/pytorch_tabnet/tab_network.py#L21
"""
struct GBN
    input_dim::Int64
    virtual_batch_size::Int64=128
    momentum::Float64::0.01

end

function (gbn::GBN)(x)
    bn = BatchNorm(input_dim)
    chunks = chunk(x, Int(round(size(x, 1) / gbn.virtual_batch_size)), dims=1)
    res = [bn(x_) for x_ in chunks]

    return cat(res, dims=1)
end


"""
Attentive Transformer

https://github.com/dreamquark-ai/tabnet/blob/40107a80b0be5ae865d945b85a52e5d99fc19a81/pytorch_tabnet/tab_network.py#L593
"""
