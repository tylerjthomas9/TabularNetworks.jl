# https://arxiv.org/abs/2012.06678
# TODO: Update to NeuralAttentionlib.jl
using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Transformers
using Transformers.Basic
using Parameters: @with_kw
using CUDA
using WordTokenizers

@with_kw mutable struct TabTransfortmerArgs
    lr::Float64 = 0.1		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    dense_hidden_sizes::Vector{Int64} = [64, 64] # Size of hidden layers
    cont_dense_size::Int64 = 32 # size of hidden layer for continious input
    cat_dense_size::Int64 = 32 # size of hidden layer for categorical input
    dense_dropout::Float64 = 0.1 # dropout for dense layers
    transformer_dropout::Float64 = 0.1 # dropout for transformer
    activation_function = relu
    heads::Int64 = 4 # number of heads for multi-head attention
    head_dims::Int64 = 64
    transformer_hidden_size::Int64 = 512
    input_dims::Int64 
    output_dims::Int64 = 8
end



function tabtransformer_input(args; cont_var, cat_var)
    return Chain(Parallel(vcat,
        Chain(Dense(cont_var, args.cont_dense_size),
            BatchNorm(args.cont_dense_size, args.activation_function)),
        Chain(Dense(cat_var, args.cat_dense_size, args.activation_function),
        Stack(
            @nntopo(e → pe:(e, pe) → x → x → 2),
            PositionEmbedding(args.input_dims),
            (e, pe) -> e .+ pe,
            Dropout(0.1),
            [Transformer(args.input_dims, args.heads, args.head_dims, args.transformer_hidden_size; 
                    act=args.activation_function) for i = 1:4]...),
            Flux.flatten
            )
        )
    )
end

# TODO: elegant way to create a variable number of layers
function tabtransformer(args::TabTransfortmerArgs; cont_var=10::Int64, cat_var=10::Int64, n_outputs=2::Int64)

    input = tabtransformer_input(args; cont_var, cat_var)
    d1 = Dense(args.cont_dense_size + args.cat_dense_size, args.dense_hidden_sizes[1], args.activation_function)
    dropout = Dropout(args.dense_dropout)
    output = Dense(args.dense_hidden_sizes[end], n_outputs)

    if size(args.dense_hidden_sizes, 1) == 1
        return Chain(input, d1, dropout, output)
    elseif size(args.dense_hidden_sizes, 1) == 2
        d2 = Dense(args.dense_hidden_sizes[1], args.dense_hidden_sizes[2], args.activation_function)
        return Chain(input, d1, dropout, d2, dropout, output)
    elseif size(args.dense_hidden_sizes, 1) == 3
        d2 = Dense(args.dense_hidden_sizes[1], args.dense_hidden_sizes[2], args.activation_function)
        d3 = Dense(args.dense_hidden_sizes[2], args.dense_hidden_sizes[3], args.activation_function)
        return Chain(input, d1, dropout, d2, dropout, d3, dropout, output)
    end
end