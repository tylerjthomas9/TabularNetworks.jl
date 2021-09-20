# TabularNetworks.jl

Small project to learn Flux.jl/Deep learning in Julia. Inspired by pytorch-widedeep: https://github.com/jrzaurin/pytorch-widedeep


# Models

* Tabular MLP
  * Categorical features one-hot encoded, passed into dense layer
  * Continious features fed into dense layer with batchnorm1d

# Roadmap

## Features before v0.1
* Models
  * Tabular MLP
  * Tab Transformer
* Data preprocessing
  * General processing of continious, categorical features for all architectures 

## Long term goals
* Pre-training support for models like TabNet
* Comprehensive documentation on the architectures 
