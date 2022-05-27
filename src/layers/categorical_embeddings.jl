using FastAI

"""
https://github.com/FluxML/FastAI.jl/blob/c3bb46a3b9ccc177e971fcfea020e91997b2a94e/src/Tabular/models.jl#L137
"""
function tabular_embedding_backbone(embedding_sizes, dropout_rate=0.)
    embedslist = [Flux.Embedding(ni, nf) for (nf, ni) in embedding_sizes]
    emb_drop = iszero(dropout_rate) ? identity : Dropout(dropout_rate)
    Chain(
        Parallel(vcat, embedslist), 
        emb_drop
    )
    Parallel(vcat, embedslist)
end

"""
https://github.com/FluxML/FastAI.jl/blob/c3bb46a3b9ccc177e971fcfea020e91997b2a94e/src/Vision/models/blocks.jl#L60
"""
function linbndrop(h_in, h_out; use_bn=true, p=0., act=identity, lin_first=false)
    bn = BatchNorm(lin_first ? h_out : h_in)
    dropout = p == 0 ? identity : Dropout(p)
    dense = Dense(h_in, h_out, act; bias=!use_bn)
    if lin_first
        return Chain(dense, bn, dropout)
    else
        return Chain(bn, dropout, dense)
    end
end
