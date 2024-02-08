from .layers import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""


def affine_relu_forward(x, w, b):
    """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
  Backward pass for the affine-relu convenience layer
  """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Scale and shift parameters for the batch normalization layer
    - bn_param: Dictionary with parameters for the batch normalization layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out1, cache1 = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(out1, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (cache1, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivative
    - cache: Tuple of caches from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    - dgamma: Gradient with respect to gamma
    - dbeta: Gradient with respect to beta
    """
    cache1, bn_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(drelu, bn_cache)
    dx, dw, db = affine_backward(dbn, cache1)
    return dx, dw, db, dgamma, dbeta
