using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA

@with_kw mutable struct MLPArgs
    lr::Float64 = 1e-2		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dense_dropout::Float64 = 0.10 # dropout from dense layers
    dense_hidden_sizes::Vector{Int64} = [64, 64] # Size of hidden layers
    cont_dense_size::Int64 = 32 # size of hidden layer for continious input
    cat_dense_size::Int64 = 32 # size of hidden layer for categorical input
    activation_function = relu
end

# build model
function mlp_input(args; cont_var, cat_var)
    return Chain(Parallel(vcat,
        Chain(Dense(cont_var, args.cont_dense_size),
            BatchNorm(args.cont_dense_size, args.activation_function)),
        Dense(cat_var, args.cat_dense_size, args.activation_function))
    )
end

# TODO: elegant way to create a variable number of layers
function mlp(args::MLPArgs; cont_var=10::Int64, cat_var=10::Int64, n_outputs=2::Int64)

    input = mlp_input(args; cont_var, cat_var)
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