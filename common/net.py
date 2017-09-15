import sys
import os

import chainer
import cupy
from chainer import function
from chainer import initializers
from chainer import utils
from chainer.utils import type_check
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np



class Floor(function.Function):
    @property
    def label(self):
        return 'floor'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        self.retain_inputs(())
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.floor(x[0]), x[0].dtype),

    def backward(self, x, grad_outputs):
        return grad_outputs


def floor(x):
    """Elementwise floor function.
    .. math::
       y_i = \\lfloor x_i \\rfloor
    Args:
        x (~chainer.Variable): Input variable.
    Returns:
        ~chainer.Variable: Output variable.
    """
    return Floor()(x)

def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if not chainer.config.train:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)


# differentiable backward functions

def backward_linear(x_in, x, l):
    y = F.matmul(x, l.W)
    return y


def backward_convolution(x_in, x, l):
    y = F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))
    return y


def backward_deconvolution(x_in, x, l):
    y = F.convolution_2d(x, l.W, None, l.stride, l.pad)
    return y


def backward_relu(x_in, x):
    y = (x_in.data > 0) * x
    return y


def backward_leaky_relu(x_in, x, a):
    y = (x_in.data > 0) * x + a * (x_in.data < 0) * x
    return y


def backward_sigmoid(x_in, g):
    y = F.sigmoid(x_in)
    return g * y * (1 - y)

class DCGANGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=16, ch=512, wscale=0.02,
                 z_distribution="uniform", hidden_activation=F.relu, output_activation=F.tanh, use_bn=True):
        super(DCGANGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * ch, initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, (1, 4), (1, 2), (0, 1), initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, (1, 4), (1, 2), (0, 1), initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, (1, 4), (1, 2), (0, 1), initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 1, (1, 3), 1, (0, 1), initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, z, n=None):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)),
                          (len(z), self.ch, -1, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            x = self.output_activation(self.dc4(h))
        else:
            h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                          (len(z), self.ch, -1, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))

        return x


class WGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=16, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        self.bottom_width = bottom_width
        self.ch = ch
        super(WGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(1, ch // 8, (1, 3), 1, (0, 1), initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, (1, 4), (1, 2), (0, 1), initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, (1, 3), 1, (0, 1), initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, (1, 4), (1, 2), (0, 1), initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, (1, 3), 1, (0, 1), initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, (1, 4), (1, 2), (0, 1), initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, (1, 3), 1, (0, 1), initialW=w)
            self.l4 = L.Linear(bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h0 = F.leaky_relu(self.c0(self.x))
        self.h1 = F.leaky_relu(self.c1(self.h0))
        self.h2 = F.leaky_relu(self.c1_0(self.h1))
        self.h3 = F.leaky_relu(self.c2(self.h2))
        self.h4 = F.leaky_relu(self.c2_0(self.h3))
        self.h5 = F.leaky_relu(self.c3(self.h4))
        self.h6 = F.leaky_relu(self.c3_0(self.h5))
        return self.l4(self.h6)
    
    def layer_output(self, x, layer):
        h = F.leaky_relu(self.c0(x))
        if layer == 'c0':
            return h
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.c1_0(h))
        if layer == 'c1':
            return h
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c2_0(h))
        if layer == 'c2':
            return h
        h = F.leaky_relu(self.c3(h))
        h = F.leaky_relu(self.c3_0(h))
        if layer == 'c3':
            return h
        return self.l4(h)
 

    def differentiable_backward(self, x):
        g = backward_linear(self.h6, x, self.l4)
        g = F.reshape(g, (x.shape[0], self.ch, -1, self.bottom_width))
        g = backward_leaky_relu(self.h6, g, 0.2)
        g = backward_convolution(self.h5, g, self.c3_0)
        g = backward_leaky_relu(self.h5, g, 0.2)
        g = backward_convolution(self.h4, g, self.c3)
        g = backward_leaky_relu(self.h4, g, 0.2)
        g = backward_convolution(self.h3, g, self.c2_0)
        g = backward_leaky_relu(self.h3, g, 0.2)
        g = backward_convolution(self.h2, g, self.c2)
        g = backward_leaky_relu(self.h2, g, 0.2)
        g = backward_convolution(self.h1, g, self.c1_0)
        g = backward_leaky_relu(self.h1, g, 0.2)
        g = backward_convolution(self.h0, g, self.c1)
        g = backward_leaky_relu(self.h0, g, 0.2)
        g = backward_convolution(self.x, g, self.c0)
        return g


class Alex(chainer.Chain):
    def __init__(self, z_dim=128):
        super(Alex, self).__init__()
        self.z_dim = z_dim
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 3, stride=1)
            self.conv2 = L.Convolution2D(None, 256, 3, pad=1)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, z_dim)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h

