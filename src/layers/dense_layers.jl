"""
Create a series of dense layers following the
embedding and continious data vcat

https://github.com/FluxML/FastAI.jl/blob/master/src/models/tabularmodel.jl
"""
function dense_layers(args)
    layers = []
    first_layer = linbndrop(args.mha_input_dims + args.cont_input_dim, first(args.hidden_dims); 
                    use_bn=args.batchnorm, p=args.dropout, lin_first=args.linear_first, 
                    act=args.activation)
    push!(layers, first_layer)

    for (isize, osize) in zip(args.hidden_dims[1:(end-1)], args.hidden_dims[2:end])
        layer = linbndrop(isize, osize; use_bn=args.batchnorm, p=args.dropout, 
                        lin_first=args.linear_first, act=args.activation)
        push!(layers, layer)
    end

    return Chain(layers...)

end