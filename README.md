# TabularNetworks.jl

Small project to learn Flux.jl/Deep learning in Julia. Inspired by pytorch-widedeep: https://github.com/jrzaurin/pytorch-widedeep


# Models

* Tabular MLP
  * Categorical features one-hot encoded, passed into embedding layer
  * Continious features fed into dense layer with batchnorm1d
* TabTransformer (https://arxiv.org/abs/2012.06678)
  * Categorical features one-hot encoded, passed into embedding layer, then transformer block
  * Continious features fed into dense layer with batchnorm1d

# Roadmap

## Features before v0.1
* Models
  * Tabular MLP
  * Tab Transformer
  * TabNet
* Data preprocessing
  * General processing of continious, categorical features for all architectures 

## v0.2 goals
* Pre-training support for models like TabNet
* Comprehensive documentation on the architectures 
* SAINT implementation
