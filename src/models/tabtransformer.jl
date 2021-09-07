# https://arxiv.org/abs/2012.06678

using Flux
using Flux: @functor
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Parameters: @with_kw
using CUDA
using NeuralAttentionlib
using NeuralAttentionlib: CausalMask, BiLengthMask, BatchedMask
using NNlib

# Struct to define hyperparameters
@with_kw mutable struct TabTransfortmerArgs
    lr::Float64 = 0.1		# learning rate
    batchsize::Int64 = 16  # batch size
    epochs::Int64 = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
    dropout::Float64 = 0.10 # dropout from dense layers
    hidden_sizes::Vector{Int64} = [64, 64] # Size of hidden layers
    dropout_rate::Float64 = 0.10 # dropout for dense layers
    activation_function = relu
    mha_heads::Int64 = 4
    mha_input_dims::Int64 = 7
    mha_head_dims::Int64 = 5
    mha_output_dims::Int64 = 8
end



function tabtransformer_input(args; cont_var, cat_var)
    return Chain(Parallel(vcat,
        Chain(Dense(cont_var, args.cont_dense_size),
            BatchNorm(args.cont_dense_size, args.activation_function)),
        Chain(Dense(cat_var, args.cat_dense_size, args.activation_function),
            # TODO: TRANSFORMER BLOCK 
            )
        )
    )
end

# TODO: elegant way to create a variable number of layers
function tabtransformer(args; cont_var=10::Int64, cat_var=10::Int64, n_outputs=2::Int64)
    if size(args.hidden_sizes, 1) == 1
        return Chain(tabtransformer_input(args; cont_var, cat_var),
            Dense(args.cont_dense_size + args.cat_dense_size, args.hidden_sizes[1], args.activation_function),
            Dropout(args.dropout_rate),
            Dense(args.hidden_sizes[end], n_outputs)
        )
    elseif size(args.hidden_sizes, 1) == 2
        return Chain(tabtransformer_input(args; cont_var, cat_var),
            Dense(args.cont_dense_size + args.cat_dense_size, args.hidden_sizes[1], args.activation_function),
            Dropout(args.dropout_rate),
            Dense(args.hidden_sizes[1], args.hidden_sizes[2], args.activation_function),
            Dropout(args.dropout_rate),
            Dense(args.hidden_sizes[end], n_outputs)
        )
    elseif size(args.hidden_sizes, 1) == 2
        return Chain(tabtransformer_input(args; cont_var, cat_var),
            Dense(args.cont_dense_size + args.cat_dense_size, args.hidden_sizes[1], args.activation_function),
            Dropout(args.dropout_rate),
            Dense(args.hidden_sizes[1], args.hidden_sizes[2], args.activation_function),
            Dropout(args.dropout_rate),
            Dense(args.hidden_sizes[2], args.hidden_sizes[3], args.activation_function),
            Dropout(args.dropout_rate),
            Dense(args.hidden_sizes[end], n_outputs)
        )
    end

end