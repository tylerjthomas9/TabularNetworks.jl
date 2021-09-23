using Flux
using Flux: @functor
using NeuralAttentionlib
using NeuralAttentionlib: CausalMask, BiLengthMask, BatchedMask
using NNlib
include("./mha.jl")

#https://github.com/chengchingwen/Transformers.jl/blob/master/src/basic/transformer.jl
struct TransformerDense{input<:Dense, output<:Dense}
    din::input
    dout::output
end

@functor TransformerDense


"just a wrapper for two dense layer."
TransformerDense(size::Int, h::Int, act = relu) = TransformerDense(
    Dense(size, h, act),
    Dense(h, size)
)

function (pw::TransformerDense)(x::AbstractMatrix)
  # size(x) == (dims, seq_len)
  return pw.dout(pw.din(x))
end

function (d::TransformerDense)(x::A) where {T, N, A<:AbstractArray{T, N}}
  new_x = reshape(x, size(x, 1), :)
  y = d(new_x)
  return reshape(y, Base.setindex(size(x), size(y, 1), 1))
end


struct TransformerBlock
    mha
    ln
    dense
end

@functor TransformerBlock

"""
Generic transformer block
"""
TransformerBlock(args) = TransformerBlock(
    MultiheadAttention(args.mha_heads, sum([i[1] for i in args.embedding_dims]), args.mha_head_dims, 
    sum([i[1] for i in args.embedding_dims]); future = true, pdrop = args.transformer_dropout),
    LayerNorm(sum([i[1] for i in args.embedding_dims])),
    Chain(TransformerDense(sum([i[1] for i in args.embedding_dims]), args.transformer_dense_hidden_dim, args.activation), 
            Dropout(args.transformer_dropout))
)

function (t::TransformerBlock)(x)
    h1 = atten(t.mha, x)
    h2 = x + h1
    h2 = t.ln(h2)
    h3 = t.dense(h2)
    h4 = h2 + h3
    h4 = t.ln(h4)
    return h4
end
