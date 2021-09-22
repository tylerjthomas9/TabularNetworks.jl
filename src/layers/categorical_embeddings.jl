# https://github.com/FluxML/FastAI.jl/blob/master/src/models/tabularmodel.jl
function tabular_embedding_backbone(embedding_sizes, dropout_rate=0.)
    embedslist = [Flux.Embedding(ni, nf) for (ni, nf) in embedding_sizes]
    emb_drop = iszero(dropout_rate) ? identity : Dropout(dropout_rate)
    Chain(
        x -> tuple(eachrow(x)...), 
        Parallel(vcat, embedslist), 
        emb_drop
    )
end
