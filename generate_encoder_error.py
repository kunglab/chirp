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

generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=make_hidden_f(1).shape[1],
                                      bottom_width=sample_width/8)
discriminator = common.net.LabeledDiscriminatorJoined(bottom_width=sample_width/8,
                                                      n_labels=1)

encoder = common.net.LabeledDiscriminatorJoined(bottom_width=sample_width/8,
                                                      n_labels=make_hidden_f(1).shape[1])

iteration_idx = 100000
serializers.load_npz(os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'], 'results_ged_hardlam10/DCGANGenerator_%d.npz' %(iteration_idx)), generator)
serializers.load_npz(os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'], 'results_ged_hardlam10/LabeledDiscriminatorJoined_%d.npz' %(iteration_idx)), discriminator)
serializers.load_npz(os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'], 'results_ged_hardlam10/EncoderLabeledDiscriminatorJoined_%d.npz' %(iteration_idx)), encoder)

if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()
    encoder.to_gpu()

xp = encoder.xp



#####################################################################
####### WRITE OUT PREDICTIONS FOR CLASSIFIER ACCURACY CHECK #########
######################################################################
nsamples = 16
evaluation_noise_levels = [6,8,10,12,14,16,18]
enc_l, disc_real_l, disc_enc_l, disc_hack_l = [], [], [], []
error_l = []

for noise_level in evaluation_noise_levels:

    data = dataset.RFModLabeled(class_set=['8PSK'], noise_levels=[noise_level], test=True, snr=True)
    data.xs /= train_max

    xs = xp.array(data.xs)

    _, z_real = encoder(xs)
    xh = generator(z_real)

    enc_pred_snr = np.mean(z_real.data[:,-1])

    yh_real, yh_real_label = discriminator(xs)
    zh_real_enc, zh_real_label = discriminator(xh)

    z_real = chainer.cuda.to_cpu(z_real.data)
    yh_real_label = chainer.cuda.to_cpu(yh_real_label.data)
    #z_real[:,-1] = yh_real_label.flatten()
    z_real[:,-1] = np.ones(z_real.shape[0])*18
    z_real = Variable(chainer.cuda.to_gpu(z_real))

    xh_hacked = generator(z_real)
    _, zh_hacked = discriminator(xh_hacked)

    xh = xh.reshape(xh.shape[0], xh.shape[2], xh.shape[3])
    xs = xs.reshape(xs.shape[0], xs.shape[2], xs.shape[3])
    xh_hacked = xh_hacked.reshape(xh_hacked.shape[0], xh_hacked.shape[2], xh_hacked.shape[3])

    z_real = chainer.cuda.to_cpu(z_real.data)
    xh = chainer.cuda.to_cpu(xh.data)
    xs = chainer.cuda.to_cpu(xs)
    # yh_real_label = chainer.cuda.to_cpu(yh_real_label.data)
    zh_real_label = chainer.cuda.to_cpu(zh_real_label.data)
    zh_hacked = chainer.cuda.to_cpu(zh_hacked.data)
    xh_hacked = chainer.cuda.to_cpu(xh_hacked.data)

    error_l.append(F.mean_absolute_error(xs, xh).data)


    print "True\tEncoder\tDiscr Real\tDisc EncGen\tDisc Hacked"
    enc_l.append(float(enc_pred_snr))
    disc_real_l.append(np.mean(yh_real_label))
    disc_enc_l.append(np.mean(zh_real_label))
    disc_hack_l.append(np.mean(zh_hacked))
    print "%d\t%f\t%f\t%f\t%f" % (noise_level, enc_pred_snr, np.mean(yh_real_label), np.mean(zh_real_label), np.mean(zh_hacked))

    fig = plt.figure(figsize=(15,15))
    for i in range(nsamples):
        ax = plt.subplot(4, 4, i+1)
        ax.plot(xs[i][0], '-')
        ax.plot(xh[i][0], '-')
        ax.plot(xh_hacked[i][0], '-')

    plt.title("Real vs Encoded I Data - SNR %d" % (noise_level))
    plt.tight_layout()
    plt.savefig(os.path.join(os.environ['KUNGLAB_SHARE_FIGURES'], 'recon_SNR_%d.png' % (noise_level)))





fig, ax = plt.subplots()

bar_width = 0.35/2.
opacity = 0.8

rects1 = plt.bar(np.arange(len(evaluation_noise_levels)), enc_l, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Encoder')

rects2 = plt.bar(np.arange(len(evaluation_noise_levels)) + bar_width, disc_real_l, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Disc - Real')

rects3 = plt.bar(np.arange(len(evaluation_noise_levels)) + 2*bar_width, disc_enc_l, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Disc(Enc)')

rects4 = plt.bar(np.arange(len(evaluation_noise_levels)) + 3*bar_width, disc_hack_l, bar_width,
                 alpha=opacity,
                 color='k',
                 label='Disc(Enc)')


plt.xlabel('SNR Level')
plt.ylabel('Predicted SNR')
plt.title('Encoder vs Discriminator SNR')
plt.xticks(np.arange(len(evaluation_noise_levels)) + bar_width / 2, evaluation_noise_levels)
plt.ylim([0, 20])
plt.yticks(range(0,22,2))
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.environ['KUNGLAB_SHARE_FIGURES'], 'SNR_encoder_error.png'))
plt.close('all')

plt.figure()
plt.plot(evaluation_noise_levels, error_l, '-bo')
plt.xlabel('Test SNR Level')
plt.ylabel('Average Reconstruction Error')
# plt.ylim([0, 2.5])
plt.tight_layout()
plt.savefig(os.path.join(os.environ['KUNGLAB_SHARE_FIGURES'], 'SNR_absolute_error_xs.png'))
plt.close()
