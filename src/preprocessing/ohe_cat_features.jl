using Flux: onehotbatch, OneHotArray

function get_category_dict(X_cat::Array)
    category_dict = Dict{Int64, Vector{Float32}}()
    for variable in 1:size(X_cat, 1)
        category_dict[variable] = unique(X_cat[variable, :])
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
    for row_ix in 1:size(X_cat, 1)
        push!(X_cat_ohe, onehotbatch(X_cat[row_ix, :], category_dict[row_ix]))
    end
    
    return X_cat_ohe 
end

