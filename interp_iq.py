import argparse
import os
import sys
from functools import partial

import numpy as np
import chainer
import chainer.functions as F
from chainer import serializers
from chainer import Variable
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import common.net
import load_models

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def slerpn(p0, p1, pts=10, min_t=0.0, max_t=1.0):
    return np.array([slerp(np.squeeze(p0), np.squeeze(p1), i)
                     for i in np.linspace(min_t, max_t, pts)])


def make_hidden(n_hidden, batchsize):
    zs = np.random.randn(batchsize, n_hidden, 1, 1).astype(np.float32)
    ys = np.random.randint(0, num_classes, batchsize, dtype=np.int32)
    label_zs = F.embed_id(ys, np.identity(num_classes, dtype=np.float32)).data
    #label_zs[label_zs < 1] = -1
    label_zs = label_zs.reshape(label_zs.shape[0], label_zs.shape[1], 1, 1)
    return np.concatenate((zs, label_zs), axis=1).astype(np.float32)


parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

chainer.config.train = False
chainer.config.enable_backprop = False
sample_width = 128
num_classes = 11
n_hidden = 32
make_hidden_f = partial(make_hidden, n_hidden)
generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=make_hidden_f(1).shape[1],
                                      bottom_width=sample_width/8)
discriminator = common.net.LabeledDiscriminator(bottom_width=sample_width/8, n_labels=num_classes)
serializers.load_npz('results/rfmod_cond/DCGANGenerator_60000.npz', generator)
serializers.load_npz('results/rfmod_cond/LabeledDiscriminator_60000.npz', discriminator)
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()

xp = generator.xp
# interpolate between two randomly sampled points
for i in range(0, 11):
    np.random.seed(1)
    z = generator.make_hidden(64)
    np.random.seed(0)
    z_label = np.zeros((z.shape[0], 11))
    z_label[:, i] = 1.
    z[:, -11:] = z_label[:, :, np.newaxis, np.newaxis]
    z = slerpn(z[0], z[1], pts=128)
    z = Variable(xp.asarray(z))
    x = generator(z)
    _, yh_class = discriminator(x)
    yh_class = F.softmax(yh_class[0].reshape(1, -1)).data
    print ["{0:0.2f}".format(yhi) for yhi in yh_class.tolist()[0]]
    z = chainer.cuda.to_cpu(xp.asarray(z.data))
    x = chainer.cuda.to_cpu(x.data)
    x = x.reshape(x.shape[0], x.shape[2], x.shape[3])

    fig = plt.figure()
    ax = plt.axes()
    line1, = plt.plot(x[0, 0], '-x', animated=True)
    line2, = plt.plot(x[0, 1], '-+', animated=True)
    #plt.ylim((-3, 3))

    def init():
        line1.set_ydata(x[0, 0])
        line2.set_ydata(x[0, 1])
        return line1, line2

    def func(i):
        # 'pause' animation at beginning and end
        if i < 5:
            i = 0
        if i >= len(z):
            i = len(z) - 1

        line1.set_ydata(x[i, 0])
        line2.set_ydata(x[i, 1])
        return line1, line2


    if not os.path.exists('figures'):
        os.makedirs('figures')

    #plt.axis('off')
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, func, init_func=init, frames=np.arange(len(z))+10, interval=100, blit=True)
    ani.save('figures/rfmod/{}.gif'.format(i), dpi=80, writer='imagemagick')
    plt.close(fig)