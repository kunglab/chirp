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
style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=noise_levels, test=False, snr=True)
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

disc_gan = common.net.LabeledDiscriminatorJoined(bottom_width=sample_width/8, n_labels=make_hidden_f(1).shape[1])
disc = common.net.LabeledDiscriminatorJoined(bottom_width=sample_width/8)
#serializers.load_npz('results_regress_train_only/GAN_LabeledDiscriminatorJoined_60000.npz', disc_gan)
# serializers.load_npz('results_acgenerator_for_regression/LabeledDiscriminatorJoined_20000.npz', disc_gan)
# serializers.load_npz('results_regress_augmented_more500/LabeledDiscriminatorJoined_60000.npz', disc_gan)
serializers.load_npz(os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'], 'regressor_500/LabeledDiscriminatorJoined_20000.npz'), disc_gan)
serializers.load_npz(os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'], 'results_regress_train_only_8PSK_moredata/LabeledDiscriminatorJoined_40000.npz'), disc)
# serializers.load_npz('results_regress_golden/LabeledDiscriminatorJoined_40000.npz', disc)

if args.gpu >= 0:
    disc_gan.to_gpu()
    disc.to_gpu()

xp = disc.xp



#####################################################################
####### WRITE OUT PREDICTIONS FOR CLASSIFIER ACCURACY CHECK #########
######################################################################
error_l = []
evaluation_noise_levels = [6,8,10,12,14,16,18]
for noise_level in evaluation_noise_levels:

    style_dataset = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=[noise_level], test=True, snr=True)
    style_dataset.xs /= train_max

    query_image = xp.array(style_dataset.xs)

    _, y_hat_disc = disc(query_image)
    _, y_hat_gan = disc_gan(query_image)


    y_hat_gan = y_hat_gan.data[:, -1]
    y_hat_disc = y_hat_disc.data[:, -1]

    print "Noise Level: %f" % (noise_level)
    print "disc: "
    print y_hat_disc[:10]
    print "gan: "
    print y_hat_gan[:10]

    error_dis = F.mean_absolute_error(y_hat_disc, xp.ones(y_hat_disc.shape[0]).astype('float32')*noise_level)
    error_dis_gan = F.mean_absolute_error(y_hat_gan, xp.ones(y_hat_gan.shape[0]).astype('float32')*noise_level)

    error_l.append([chainer.cuda.to_cpu(error_dis.data), chainer.cuda.to_cpu(error_dis_gan.data)])

    print "Dis: %f, Gan: %f" %( error_dis.data, error_dis_gan.data)


error_l = np.array(error_l)

plt.figure()
plt.plot(evaluation_noise_levels,  error_l[:,0], 'r-o')
plt.plot(evaluation_noise_levels, error_l[:,1], 'b-o')
plt.legend(['Regression-Only Real', 'Regression+GAN Data'])
plt.savefig(os.path.join(os.environ['KUNGLAB_SHARE_FIGURES'], 'generator_improve_train.png'))
plt.close()
