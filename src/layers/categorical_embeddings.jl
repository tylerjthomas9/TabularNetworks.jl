using FastAI

# https://github.com/FluxML/FastAI.jl/blob/master/src/models/tabularmodel.jl
function tabular_embedding_backbone(embedding_sizes, dropout_rate=0.)
    embedslist = [Flux.Embedding(nf, ni) for (ni, nf) in embedding_sizes]
    emb_drop = iszero(dropout_rate) ? identity : Dropout(dropout_rate)
    Chain(
        Parallel(vcat, embedslist), 
        emb_drop
    )
    Parallel(vcat, embedslist)
end


# FROM: https://github.com/FluxML/FastAI.jl/blob/master/src/models/blocks.jl
# function was not importing, so I copied it over, for now.

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

"""
Create a series of dense layers following the
embedding and continious data vcat

https://github.com/FluxML/FastAI.jl/blob/master/src/models/tabularmodel.jl
"""
function dense_layers(args)
    cat_input_dim = sum([i[1] for i in args.embedding_dims])

    layers = []
    first_layer = linbndrop(cat_input_dim + args.cont_input_dim, first(args.hidden_dims); 
                    use_bn=args.batchnorm, p=args.dropout_rate, lin_first=args.linear_first, 
                    act=args.activation)
    push!(layers, first_layer)

    for (isize, osize) in zip(args.hidden_dims[1:(end-1)], args.hidden_dims[2:end])
        layer = linbndrop(isize, osize; use_bn=args.batchnorm, p=args.dropout_rate, 
                        lin_first=args.linear_first, act=args.activation)
        push!(layers, layer)
    end

    return Chain(layers...)

end