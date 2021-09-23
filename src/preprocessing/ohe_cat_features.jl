using Flux: onehotbatch, OneHotArray

function get_category_dict(X_cat::Array)
    category_dict = Dict{Float64, Vector{Float32}}()
    for variable in 1:size(X_cat, 1)
        category_dict[variable |> Float64] = unique(X_cat[variable, :])
    end
    return category_dict
end


"""
Build categorical feature embedding using 
one-hot encoding
"""
function ohe_cat_features(X_cat, category_dict)
    
    # loop over columns and build tuple of ohe matricies
    X_cat_ohe =  Any[]
    row_ix = 1
    @inbounds @views for row in eachrow(X_cat)
        push!(X_cat_ohe, onehotbatch(row, category_dict[row_ix]))
        row_ix += 1
    end
    
    return X_cat_ohe 
end

