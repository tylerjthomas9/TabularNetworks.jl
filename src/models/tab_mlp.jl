using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA

# Struct to define hyperparameters
@with_kw mutable struct TabMLPArgs
    lr::Float64 = 0.1		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_sizes::Vector{Int64} = [64, 64] # Size of hidden layers
    cont_dense_size::Int64 = 32 # size of hidden layer for continious input
    cat_dense_size::Int64 = 32 # size of hidden layer for categorical input
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = relu
end

# build model
function tab_mlp_input(args; cont_var, cat_var)
    return Chain(Parallel(vcat,
        Chain(Dense(cont_var, args.cont_dense_size),
            BatchNorm(args.cont_dense_size, args.activation_function)),
        Dense(cat_var, args.cat_dense_size, args.activation_function))
    )
end

# TODO: elegant way to create a variable number of layers
function tab_mlp(args::TabMLPArgs; cont_var=10::Int64, cat_var=10::Int64, n_outputs=2::Int64)

    input = tab_mlp_input(args; cont_var, cat_var)
    d1 = Dense(args.cont_dense_size + args.cat_dense_size, args.hidden_sizes[1], args.activation_function)
    dropout = Dropout(args.dropout_rate)
    output = Dense(args.hidden_sizes[end], n_outputs)

    if size(args.hidden_sizes, 1) == 1
        return Chain(input, d1, dropout, output)
    elseif size(args.hidden_sizes, 1) == 2
        d2 = Dense(args.hidden_sizes[1], args.hidden_sizes[2], args.activation_function)
        return Chain(input, d1, dropout, d2, dropout, output)
    elseif size(args.hidden_sizes, 1) == 3
        d2 = Dense(args.hidden_sizes[1], args.hidden_sizes[2], args.activation_function)
        d3 = Dense(args.hidden_sizes[2], args.hidden_sizes[3], args.activation_function)
        return Chain(input, d1, dropout, d2, dropout, d3, dropout, output)
    end
end