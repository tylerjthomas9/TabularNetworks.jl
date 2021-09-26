using NeuralAttentionlib
using Test
using Flux
using MacroTools: @forward, @capture

# taken from
# https://github.com/chengchingwen/NeuralAttentionlib.jl/blob/master/test/mha.jl

const Abstract3DTensor{T} = AbstractArray{T, 3}

macro toNd(ex, outref::Int=1)
    if @capture ex f_(xs__; kw__)
        kwe = Expr(:parameters, kw...)
    else
        @capture ex f_(xs__)
    end
    rxs = map(xs) do x
        :(reshape($x, size($x, 1), :))
    end
    newcall = kw === nothing ? Expr(:call, f, rxs...) : Expr(:call, f, kwe, rxs...)
    :(reshape($newcall, :, Base.tail(size($(xs[outref])))...)) |> esc
end

abstract type AbstractAttention end

function atten(mha, x, mask=nothing)
    q = mha.iqproj(x)
    k = mha.ikproj(x)
    v = mha.ivproj(x)
    a = NeuralAttentionlib.multihead_qkv_attention(4, q, k, v, mask)
    return mha.oproj(a)
end



struct MultiheadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense, DP<:Dropout} <: AbstractAttention
    head::Int
    future::Bool
    iqproj::Q
    ikproj::K
    ivproj::V
    oproj::O
    drop::DP
end

Flux.functor(mh::MultiheadAttention) = (mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj), m -> MultiheadAttention(mh.head, mh.future, m..., mh.drop)

"""
    MultiheadAttention(head::Int, is::Int, hs::Int, os::Int;
                       future::Bool=true, pdrop = 0.1)
Multihead dot product Attention Layer, `head` is the number of head, 
`is` is the input size, `hs` is the hidden size of input projection layer of each head, 
`os` is the output size. When `future` is `false`, the k-th token can't see tokens at > k. 
`pdrop` is the dropout rate.
"""
MultiheadAttention(head::Int,
                   is::Int,
                   hs::Int,
                   os::Int;
                   future::Bool=true, pdrop = 0.1) = MultiheadAttention(head,
                                                                        future,
                                                                        Dense(is, hs*head),
                                                                        Dense(is, hs*head),
                                                                        Dense(is, hs*head),
                                                                        Dense(hs*head, os),
                                                                        Dropout(pdrop),
                                                                        )

function (mh::MultiheadAttention)(query::A1,
                                  key::A2,
                                  value::A3;
                                  mask=nothing) where {T,
                                                       A1 <: Abstract3DTensor{T},
                                                       A2 <: Abstract3DTensor{T},
                                                       A3 <: Abstract3DTensor{T}}
  qs = size(query)
  ks = size(key)
  vs = size(value)

  #size(ipq) == (h, q_seq_len, batch)
  ipq = @toNd mh.iqproj(query)
  ipk = @toNd mh.ikproj(key)
  ipv = @toNd mh.ivproj(value)

  h = size(ipq, 1)
  hs = div(h, mh.head)

  #size(ipq) == (hs, q_seq_len, head, batch)
  ipq = permutedims(reshape(ipq, hs, mh.head, qs[2], qs[3]), [1, 3, 2, 4])
  ipk = permutedims(reshape(ipk, hs, mh.head, ks[2], ks[3]), [1, 3, 2, 4])
  ipv = permutedims(reshape(ipv, hs, mh.head, vs[2], vs[3]), [1, 3, 2, 4])

  #size(ipq) == (hs, q_seq_len, head * batch)
  ipq = reshape(ipq, hs, qs[2], :)
  ipk = reshape(ipk, hs, ks[2], :)
  ipv = reshape(ipv, hs, vs[2], :)

  atten = attention(ipq,ipk,ipv,
                    mask,
                    mh.future,
                    mh.drop)

  atten = permutedims(reshape(atten, hs, qs[2], mh.head, qs[3]), [1, 3, 2, 4]) #size(atten) == (hs, head, ql, b)
  atten = reshape(atten, h, qs[2], qs[3]) #size(atten) == (h, ql, b)

  out = @toNd mh.oproj(atten)
  out #size(out) == (h, q_seq_len, batch)
end