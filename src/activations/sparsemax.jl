# https://github.com/FluxML/NNlib.jl/pull/364
"""
    sparsemax(x; dims = 1)

[Sparsemax](https://arxiv.org/abs/1602.02068) turns input array `x`
into sparse probability distributions that sum to 1 along the dimensions specified by `dims`.

Similar to softmax, each dimension is considered independent. For a matrix input `x` it will 
by default (`dims = 1`) treat it as a batch of vectors, with each column independent. Keyword 
`dims = 2` will instead treat rows independently.

# Examples

```jldoctest
julia> sparsemax(reshape(1:15, 5,3) ./ 2) # dims=1
5×3 Matrix{Float64}:
 0.0   0.0   0.0
 0.0   0.0   0.0
 0.0   0.0   0.0
 0.25  0.25  0.25
 0.75  0.75  0.75

julia> sparsemax(reshape(1:15, 5, 3) ./ 20) # Scale matters
5×3 Matrix{Float64}:
 0.1   0.1   0.1
 0.15  0.15  0.15
 0.2   0.2   0.2
 0.25  0.25  0.25
 0.3   0.3   0.3

 julia> sparsemax(reshape(1:15, 5, 3) ./ 20 .+ [-20 0 20])
 5×3 Matrix{Float64}:
  0.1   0.1   0.1
  0.15  0.15  0.15
  0.2   0.2   0.2
  0.25  0.25  0.25
  0.3   0.3   0.3
```
"""
function sparsemax(x::AbstractArray; dims::Integer=1)
    n_dims = length(size(x))
    if n_dims < dims
        error_msg = "Dims ($dims) is larger than the array dimensions ($n_dims)."
        throw(DimensionMismatch(error_msg))
    end

    if isa(x, AbstractVector)
        z = _sort_vector(x; dims=dims)
    else
        z = sort(float(x); dims=dims, rev=true) 
    end
    mask = _sparsemax_mask(z, dims)
    tausum = sum(z .* mask; dims) 
    kay = sum(mask; dims)
    z = _relu.(x  .- (tausum .- 1) ./ kay)
end

function _sort_vector(x::AbstractVector; dims::Integer=1, rev=true)
    if dims == 1
        z = sort(float(x); rev=rev)
    elseif dims == 2
        z = x
    end
    return z
end

function _sparsemax_mask(z::AbstractArray, dim::Integer)
    acc = cumsum(z; dims=dim)
    if dim == 1
        acc .= 1 .+ axes(z,1) .* z .> acc
    elseif dim == 2
        acc .= 1 .+ axes(z,2)' .* z .> acc
    else
        # This isn't type-stable. Writing into `acc` ensures the whole function still is:
        cnt = reshape(axes(x, dim), ntuple(_->1, dim-1)..., :)
        acc .= 1 .+ cnt .* z .> acc
    end
    acc
end

_ifelse(p, x, y) = ifelse(p, promote(x, y)...)
_relu(x) = _ifelse(x>0, x, false)  # different gradient at zero

function ∇sparsemax(Δ::AbstractArray, y::AbstractArray; dims = 1)
    nonzeros = Δ.!=0.0
    out = Δ .* y
    total = sum(out .* nonzeros; dims=dims) / sum(nonzeros; dims=dims)
    out = nonzeros .* (out .- total)
 end

function rrule(::typeof(sparsemax), xs; dims=1)
    y = sparsemax(xs; dims=dims)
    sparsemax_pullback(Δ) = (NoTangent(), ∇sparsemax(unthunk(Δ), y; dims = dims))
    return y, sparsemax_pullback
 end
 