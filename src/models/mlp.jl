using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA

@with_kw mutable struct MLPArgs
    cat_input_dim::Int64
    cat_hidden_dim::Int64 = 32
    cont_input_dim::Int64
    cont_hidden_dim::Int64 = 32
    output_dim::Int64
    lr::Float64 = 1e-3		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_dims::Vector{Int64} = [64, 64] # Size of hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = tanh
    output_activation = sigmoid
    seed::Int64 = 42
end


struct MLP_Input
    categorical_input
    continious_input
end
@functor MLP_Input

MLP_Input(args::MLPArgs) = MLP_Input(
    Chain(Dense(args.cat_input_dim, args.cat_hidden_dim, args.activation_function),),
    Chain(
        BatchNorm(args.cont_input_dim, ),
        Dense(args.cont_input_dim, args.cont_hidden_dim, args.activation_function)
    )
)

function (mlp_input::MLP_Input)(cat, cont)
    Parallel(vcat, mlp_input.categorical_input(cat),
    mlp_input.continious_input(cont))
end

struct MLP
    input
    dense
    output
end
@functor MLP

MLP(args::MLPArgs) = MLP(
    Parallel(vcat, 
        Chain(Dense(args.cat_input_dim, args.cat_hidden_dim, args.activation_function),),
        Chain(Dense(args.cont_input_dim, args.cont_hidden_dim),
                BatchNorm(args.cont_hidden_dim, args.activation_function)
        )
    ),
    Dense(args.cat_hidden_dim + args.cont_hidden_dim, args.hidden_dims[1], args.activation_function),
    Dense(args.hidden_dims[1], args.output_dim, args.output_activation)
)

function (mlp::MLP)(cat, cont)
    h1 = mlp.input(cat, cont)
    h2 = mlp.dense(h1)
    mlp.output(h2)
end





