from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, A.dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, A.dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) * B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(0, A.dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i), B))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.const(0, A.dtype)
    D = tvm.compute(A.shape, lambda *i: tvm.expr.Select((A(*i) > C), B(*i), C))

    s = tvm.create_schedule(D.op)
    f = tvm.build(s, [A, B, D], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    A = tvm.placeholder((shapeA[0], shapeA[1]), dtype=dtype, name="A")
    B = tvm.placeholder((shapeB[0], shapeB[1]), dtype=dtype, name="B")

    if transposeA == False and transposeB == False:

        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.compute((shapeA[0], shapeB[1]), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k))

    elif transposeA == True and transposeB == False:

        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k))
    
    elif transposeA == False and transposeB == True:

        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))
    
    else:

        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.sum(A[k, i] * B[j, k], axis=k))

    s = tvm.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
    xk, yk = s[C].split(k, factor=8)
    s[C].reorder(xo, yo, xk, xi, yi, yk)
    s[C].parallel(xo)
    s[C].unroll(yk)
    #print(tvm.lower(s, [A, B, C], simple_mode=True))
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f



def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    Input = tvm.placeholder(shapeX, dtype=dtype, name="A")
    Filter = tvm.placeholder(shapeF, dtype=dtype, name="B")

    di = tvm.reduce_axis((0, R), name='di')
    dj = tvm.reduce_axis((0, S), name='dj')
    dc = tvm.reduce_axis((0, C), name='dc')

    Output = tvm.compute((N, M, H - R + 1, W - S + 1),
                        lambda n, m, i, j: tvm.sum(Input[n, dc, i + di, j + dj] * Filter[m, dc, di, dj], axis=[di, dj, dc]),
                        name='Output')
    s = tvm.create_schedule(Output.op)
    f = tvm.build(s, [Input, Filter, Output], tgt, target_host=tgt_host, name=func_name)
    return f

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """

    A = tvm.placeholder(shape, dtype=dtype, name="A")
    k = tvm.reduce_axis((0, shape[1]), name="k")
    max_A = tvm.compute((shape[0],), lambda i: tvm.max(A[i, k], axis=k), name="max_A")
    exp = tvm.compute(shape, lambda i, j: tvm.exp(A[i, j] - max_A[i]), name="exp")
    k1 = tvm.reduce_axis((0, shape[1]), name="k1")
    sum_exp = tvm.compute((shape[0],), lambda i: tvm.sum(exp[i, k1], axis=k1), name="sum_exp")
    B = tvm.compute(shape, lambda i, j: exp[i, j] / sum_exp[i], name="B")

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""

    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    k = tvm.reduce_axis((0, shape[1]), name="k")
    max_A = tvm.compute((shape[0],), lambda i: tvm.max(A[i, k], axis=k), name="max_A")
    exp = tvm.compute(shape, lambda i, j: tvm.exp(A[i, j] - max_A[i]), name="exp")
    k1 = tvm.reduce_axis((0, shape[1]), name="k1")
    sum_exp = tvm.compute((shape[0],), lambda i: tvm.sum(exp[i, k1], axis=k1), name="sum_exp")
    softmax = tvm.compute(shape, lambda i, j: exp[i, j] / sum_exp[i], name="softmax")

    log = tvm.compute(shape, lambda i, j: tvm.log(softmax[i, j]), name = "log")
    k2 = tvm.reduce_axis((0, shape[1]), name="k2")
    sum_softmax = tvm.compute((shape[0],), lambda i: tvm.sum(B[i, k2] * log[i, k2], axis = k2), name="sum_softmax")
    k3 = tvm.reduce_axis((0, shape[0]), name="k3")
    softmax_cross_entropy = tvm.compute((1,), lambda i: tvm.sum(-1 * sum_softmax[k3] / shape[0], axis = k3))

    s = tvm.create_schedule(softmax_cross_entropy.op)
    f = tvm.build(s, [A, B, softmax_cross_entropy], tgt, target_host=tgt_host, name=func_name)
    return f

def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f