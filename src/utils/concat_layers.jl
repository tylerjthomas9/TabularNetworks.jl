using Flux

struct Concat{T}
    catted::T
end
Concat(xs...) = Concat(xs)

Flux.@functor Concat

function (C::Concat)(x)
    mapreduce((f, x) -> f(x), vcat, C.catted, x)
end