import numpy as np
import os, sys
import argparse

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

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

## Alex net ##



model_map = {"AlexStock": Alex, "VTCNN2" : VTCNN2}

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='result_classifier',
                    help='Path to saved model')
parser.add_argument('--out', '-o', default='result_classifier',
                    help='Directory to output the result')
parser.add_argument('--unit', '-u', type=int, default=1000,
                    help='Number of units')
parser.add_argument('--model_type', '-t', type=str, default="AlexStock",
                    help='Which Model to run (AlexStock, VTCNN2)')
args = parser.parse_args()


###### SETUP DATASET #####
noise_levels = [6,8,16,18]
style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=noise_levels, test=False)
train_max = np.max(np.abs(style_dataset.xs))

noise_levels = [0]#range(6, 20, 2)
style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=noise_levels, test=False)

style_dataset.xs /= train_max

query_image = style_dataset.xs[20:21]
# print query_image.shape
# assert False


def make_hidden(n_hidden, batchsize):
    zs = np.random.randn(batchsize, n_hidden, 1, 1).astype(np.float32)
    ys = np.random.randint(0, num_classes, batchsize, dtype=np.int32)
    label_zs = F.embed_id(ys, np.identity(num_classes, dtype=np.float32)).data
    label_zs = label_zs.reshape(label_zs.shape[0], label_zs.shape[1], 1, 1)
    return np.concatenate((zs, label_zs), axis=1).astype(np.float32)



chainer.config.train = False
chainer.config.enable_backprop = False
sample_width = 128
num_classes = 1
n_hidden = 32
make_hidden_f = partial(make_hidden, n_hidden)
generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=make_hidden_f(1).shape[1],
                                      bottom_width=sample_width/8)
discriminator = common.net.LabeledDiscriminator(bottom_width=sample_width/8, n_labels=make_hidden_f(1).shape[1])
serializers.load_npz('alpha_results_3/alpha_rfmod_cond_2/DCGANGenerator_10000.npz', generator)
serializers.load_npz('alpha_results_3/alpha_rfmod_cond_2/LabeledDiscriminator_10000.npz', discriminator)
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()

xp = generator.xp


xp = np if args.gpu < 0 else cupy
query_image = xp.array(query_image)
_, s_vector = discriminator(query_image)
query_image_h = generator(s_vector)

x = query_image
xh = query_image_h


# query_image = chainer.cuda.to_cpu(query_image.data)
# query_image_h = chainer.cuda.to_cpu(query_image_h.data)

x = chainer.cuda.to_cpu(x)
x = x*train_max

xh = chainer.cuda.to_cpu(xh.data)
# xh = xh.reshape(xh.shape[0], xh.shape[2], xh.shape[3])
xh = xh*train_max

plt.figure()
plt.plot(x.reshape(x.shape[2],x.shape[3]).T)
plt.savefig('SNR0_original.png')
plt.figure()
plt.plot(xh.reshape(x.shape[2],x.shape[3]).T)
plt.savefig('SNR0_recon.png')
assert False




s_vector = chainer.cuda.to_cpu(s_vector._data[0])[0]

print "S_VECTPR: ", s_vector

input_to_gan = np.hstack((s_vector,np.array(1.)))
input_to_gan = np.reshape(input_to_gan, (1,len(input_to_gan)))
input_to_gan = input_to_gan.astype('float32')

print "Input to gan: ", input_to_gan.shape
# assert False




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


chainer.config.train = False
chainer.config.enable_backprop = False
sample_width = 128
num_classes = 1
n_hidden = 32
make_hidden_f = partial(make_hidden, n_hidden)
generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=make_hidden_f(1).shape[1],
                                      bottom_width=sample_width/8)
discriminator = common.net.LabeledDiscriminator(bottom_width=sample_width/8, n_labels=num_classes)
serializers.load_npz('results_style/rfmod_cond/DCGANGenerator_60000.npz', generator)
serializers.load_npz('results_style/rfmod_cond/LabeledDiscriminator_60000.npz', discriminator)
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()

xp = generator.xp

z = Variable(xp.asarray(input_to_gan))
print z.shape
x = generator(z)


import matplotlib.pyplot as plt

new_signal = chainer.cuda.to_cpu(x._data[0])
print new_signal
plt.plot(new_signal.reshape(x.shape[2],x.shape[3]).T)
plt.savefig('fig_0.png')


assert False



num_samples = 100000
zdim = 32

saved_xs = np.zeros((num_samples, 1, 2, 128))
saved_zs = np.zeros((num_samples, zdim))

batchsize = 1000

# interpolate between two randomly sampled points
for i in range(0, 100):
    # np.random.seed(1)
    z = generator.make_hidden(batchsize)
    # np.random.seed(0)
    z_label = np.zeros((z.shape[0], 1))
    z_label[:, -1] = 1.
    z[:, -1:] = z_label[:, :, np.newaxis, np.newaxis]
    z = z.reshape(z.shape[0], z.shape[1])
    # z = slerpn(z[0], z[1], pts=128)

    # z[:, -1] = z[0,-1]

    z = Variable(xp.asarray(z))
    x = generator(z)
    _, yh_class = discriminator(x)
    yh_class = F.softmax(yh_class[0].reshape(1, -1)).data
    print ["{0:0.2f}".format(yhi) for yhi in yh_class.tolist()[0]]
    z = chainer.cuda.to_cpu(xp.asarray(z.data))
    x = chainer.cuda.to_cpu(x.data)
    x = x.reshape(x.shape[0], x.shape[2], x.shape[3])

    print z
    print "Z: ", z.shape
    print "X: ", x.shape
    print "Xreshape: ", x.reshape(x.shape[0], 1, x.shape[1], x.shape[2]).shape

    z = z[:,:32]


    saved_xs[i*batchsize:(i+1)*batchsize,:] = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    saved_zs[i*batchsize:(i+1)*batchsize,:] = z


np.savez('generated_data_for_encoder_xs_and_zs.npz', x=saved_xs, z=saved_zs)
print "Saved!"



