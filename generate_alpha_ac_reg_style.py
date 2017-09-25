import numpy as np
import os, sys
import argparse
import copy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training

from common.dataset import Dataset, LabeledDataset
from chainer.training import extensions
from chainer.datasets import get_mnist
from chainer import Variable
from chainer import serializers
from sklearn import metrics

from functools import partial

import cupy

from common import dataset
import common.net
from common.net import Alex, VTCNN2
import utilities as util
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def slerpn(p0, p1, pts=10, min_t=0.0, max_t=1.0):
    return np.array([slerp(np.squeeze(p0), np.squeeze(p1), i)
                     for i in np.linspace(min_t, max_t, pts)])

def plotAnimation(x, outdir, outfile):
    fig = plt.figure()
    ax = plt.axes()
    line, = plt.plot(x[0,0,0,:], x[0,0,1,:], '-o', animated=True)
    plt.xlim([-2,2])
    plt.ylim([-2,2])

    def init():
        plt.plot(x_cir, y_cir, 'ro', markersize=14)
        line.set_xdata(x[0,0,0,:])
        line.set_ydata(x[0,0,1,:])
        return line,

    def func(i):
        print 'frame: ', i
        # 'pause' animation at beginning and end
        if i < 5:
            i = 0
        if i >= x.shape[0]:
            i = x.shape[0] - 1

        line.set_xdata(x[i,0,0,:])
        line.set_ydata(x[i,0,1,:])
        return line,

    # out_dir = "figures/constellations"
    # if not os.path.exists(out_dir):
        # os.makedirs(out_dir)

    #plt.axis('off')
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, func, init_func=init, frames=x.shape[0]+10, interval=100, blit=True)
    ani.save(os.path.join(outdir, outfile), dpi=50, writer='imagemagick')


x_cir = [2., 1.414, 0, -1.414, -2, -1.414, 0, 1.414]
y_cir = [0., 1.414, 2, 1.414, 0, -1.414, -2., -1.414]

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
# parser.add_argument('--model', '-m', default='result_classifier',
#                     help='Path to saved model')
parser.add_argument('--out', '-o', default='result_classifier',
                    help='Directory to output the result')
args = parser.parse_args()


if not os.path.exists(args.out):
    os.makedirs(args.out)


save_npzs = True

#######################################
###### SETUP DATASET AND MODEL ########
#######################################
noise_levels = [6,8,16,18]
style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=noise_levels, test=False)
train_max = np.max(np.abs(style_dataset.xs))

def make_hidden(n_hidden, batchsize):
    zs = np.random.randn(batchsize, n_hidden, 1, 1).astype(np.float32)
    snr_zs = np.random.uniform(6.0, 18.0, (batchsize, 1, 1, 1)).astype(np.float32)
    return np.concatenate((zs, snr_zs), axis=1)

chainer.config.train = False
chainer.config.enable_backprop = False
sample_width = 128
num_classes = 1
n_hidden = 32
make_hidden_f = partial(make_hidden, n_hidden)
generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=make_hidden_f(1).shape[1],
                                      bottom_width=sample_width/8)
discriminator = common.net.LabeledDiscriminator(bottom_width=sample_width/8, n_labels=make_hidden_f(1).shape[1])
serializers.load_npz('alphaac_reg_results_2/DCGANGenerator_10000.npz', generator)
serializers.load_npz('alphaac_reg_results_2/LabeledDiscriminator_10000.npz', discriminator)
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()

xp = generator.xp



#####################################################################
####### WRITE OUT PREDICTIONS FOR CLASSIFIER ACCURACY CHECK #########
######################################################################
evaluation_noise_levels = [6,8,10,12,14,16,18]
for noise_level in evaluation_noise_levels:

    style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=[noise_level], test=False)

    style_dataset.xs /= train_max

    query_image = style_dataset.xs[20:21]

    query_image = xp.array(query_image)
    _, s_vector = discriminator(query_image)

    print "True: %f, Predicted: %f" % (noise_level, s_vector.data[0][-1])
    # print s_vector.data[0]



########################################################################
####### GENERATE ORIGINAL AND ENCODED-RECONSTRUCTED IMAGES #############
########################################################################
evaluation_noise_levels = [6,12,18]
for noise_level in evaluation_noise_levels:
    style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=[noise_level], test=False)

    style_dataset.xs /= train_max

    query_image = style_dataset.xs[0:1]

    query_image = xp.array(query_image)
    _, s_vector = discriminator(query_image)
    query_image_h = generator(s_vector)

    x = query_image
    xh = query_image_h

    x = chainer.cuda.to_cpu(x)
    x = x*train_max

    xh = chainer.cuda.to_cpu(xh.data)
    xh = xh*train_max


    plt.figure()
    plt.plot(x.reshape(x.shape[2],x.shape[3]).T)
    plt.ylim([-3,3])
    plt.savefig(os.path.join(args.out, 'SNR%d_original.png' % (noise_level)))
    plt.figure()
    plt.plot(xh.reshape(xh.shape[2],xh.shape[3]).T)
    plt.ylim([-3,3])
    plt.savefig(os.path.join(args.out, 'SNR%d_recon.png' % (noise_level)))

    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    line, = plt.plot(x[0,0,0,:], x[0,0,1,:], '-o')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.plot(x_cir, y_cir, 'ro', markersize=14)
    plt.grid()
    plt.savefig(os.path.join(args.out, 'SNR%d_const.png' % (noise_level)))

    if save_npzs:
        np.savez(os.path.join(args.out, 'SNR%d_original.npz' % (noise_level)), x.reshape(x.shape[2],x.shape[3]).T)
        np.savez(os.path.join(args.out, 'SNR%d_recon.npz' % (noise_level)), xh.reshape(xh.shape[2],xh.shape[3]).T)


############################################################################################
####### GENERATE ANIMATION FIXED SNR INTERPOLATE IN Z SPACE BETWEEN TWO QUERY POINTS ####### 
############################################################################################
evaluation_noise_levels = [6,12,18]#range(6, 20, 2)
for noise_level in evaluation_noise_levels:
    style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=[noise_level], test=False)

    style_dataset.xs /= train_max

    query_image = style_dataset.xs[20:22]

    _, z = discriminator(xp.array(query_image))
    z = chainer.cuda.to_cpu(z.data)

    z = slerpn(z[0], z[1], pts=128)
    z = Variable(xp.asarray(z))
    x = generator(z)
    z = chainer.cuda.to_cpu(xp.asarray(z.data))
    x = chainer.cuda.to_cpu(x.data)
    data = copy.deepcopy(x)

    x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
    x = x*train_max

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


    plt.tight_layout()
    ani = animation.FuncAnimation(fig, func, init_func=init, frames=np.arange(len(z))+10, interval=100, blit=True)
    ani.save(os.path.join(args.out, 'SNR%d_animation.gif' % (noise_level)), dpi=80, writer='imagemagick')
    plt.close(fig)

    if save_npzs:
        np.savez(os.path.join(args.out, 'SNR%d_animation.npz' % (noise_level)), x)

    plotAnimation(data, args.out, 'SNR%d_iq_animation.npz' % (noise_level))


##########################################################################
###### GENERATE ANIMATION FIXED QUERY POINT INTERPOLATE OVER SNR #########
##########################################################################
style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=[18], test=False)

style_dataset.xs /= train_max

num_frames = 128

query_image = style_dataset.xs[20:21]
query_image = xp.array(query_image)
_, s_vector = discriminator(query_image)
s_vector = chainer.cuda.to_cpu(s_vector.data)
s_vector = s_vector.flatten()
s_vector = np.tile(s_vector,num_frames).reshape(-1,s_vector.shape[0])
z = s_vector
amp_real = np.linspace(18, -18, num_frames)
z[:, -1] = amp_real
z = z.reshape(z.shape[0], z.shape[1], 1, 1)
z = xp.asarray(z)
x = generator(Variable(z))

z = chainer.cuda.to_cpu(z)
x = chainer.cuda.to_cpu(x.data)
data = copy.deepcopy(x)
x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
x = x*train_max

fig = plt.figure()
ax = plt.axes()
line1, = plt.plot(x[0, 0], '-x', animated=True)
line2, = plt.plot(x[0, 1], '-+', animated=True)
plt.ylim((-1, 1))

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



#plt.axis('off')
plt.tight_layout()
ani = animation.FuncAnimation(fig, func, init_func=init, frames=np.arange(len(z))+10, interval=100, blit=True)
ani.save(os.path.join(args.out, 'SNR_interpolation.gif'), dpi=80, writer='imagemagick')
plt.close(fig)

if save_npzs:
        np.savez(os.path.join(args.out, 'SNR_interpolation.npz'), x)
    
plotAnimation(data, args.out, 'SNR_iq_animation.npz')
