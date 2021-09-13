using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw


@with_kw mutable struct MLPArgs
    cat_input_dim::Int64
    cont_input_dim::Int64
    cat_hidden_dim::Int64 = 32
    cont_hidden_dim::Int64 = 32
    output_dim::Int64 = 2
    lr::Float64 = 1e-2		# learning rate
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_dims::Vector{Int64} = [64, 64] # Size of hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = relu
    output_activation = sigmoid
    seed::Int64 = 42
end



struct MLP
    cat_input
    cont_input
    dense
    output
end
@functor MLP

MLP(args::MLPArgs) = MLP(
    Dense(args.cat_input_dim, args.cat_hidden_dim, args.activation_function),
    Chain(
        Dense(args.cont_input_dim, args.cont_hidden_dim, args.activation_function),
        BatchNorm(args.cont_hidden_dim, )
        ),
    Chain([Dense(if ix==1 args.cat_hidden_dim + args.cont_hidden_dim else args.hidden_dims[ix-1] end, 
        args.hidden_dims[ix], args.activation_function) for ix in 1:size(args.hidden_dims, 1)]...),
    Dense(args.hidden_dims[end], args.output_dim, args.output_activation)
)

function (mlp::MLP)(cat, cont)
    h_cat = mlp.cat_input(cat)
    h_cont = mlp.cont_input(cont)
    h1 = vcat(h_cat, h_cont)
    h2 = mlp.dense(h1)
    mlp.output(h2)
end





