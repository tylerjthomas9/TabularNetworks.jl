using Flux: onehotbatch, OneHotMatrix

"""
Build categorical feature embedding using 
one-hot encoding
"""
function ohe_cat_features(X_cat_train)
    
    # loop over columns and build ohe Array
    X_ohe_train =  Matrix{Bool}
    for row_ix in 1:size(X_cat_train, 1)
        if row_ix == 1
            X_ohe_train = onehotbatch(X_cat_train[row_ix, :], unique(X_cat_train[row_ix, :]))
        else
            X_ohe_train = vcat(X_ohe_train, onehotbatch(X_cat_train[row_ix, :], unique(X_cat_train[row_ix, :])))
        end
    end
    
    return X_ohe_train
end


function ohe_cat_features(X_cat_train, X_cat_test)
    
    # loop over columns and build ohe Array
    X_ohe_train =  Matrix{Bool}
    X_ohe_test =  Matrix{Bool}
    @assert size(X_cat_train, 1) == size(X_cat_test, 1)
    for row_ix in 1:size(X_cat_train, 1)
        if row_ix == 1
            X_ohe_train = onehotbatch(X_cat_train[row_ix, :], unique(X_cat_train[row_ix, :]))
            X_ohe_test = onehotbatch(X_cat_test[row_ix, :], unique(X_cat_train[row_ix, :]))
        else
            X_ohe_train = vcat(X_ohe_train, onehotbatch(X_cat_train[row_ix, :], unique(X_cat_train[row_ix, :])))
            X_ohe_test = vcat(X_ohe_test, onehotbatch(X_cat_test[row_ix, :], unique(X_cat_train[row_ix, :])))
        end
    end
    
    return X_ohe_train, X_ohe_test
end