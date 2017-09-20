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
from chainer import serializers
from sklearn import metrics

import cupy

from common import dataset
import utilities as util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Alex net ##

class Alex(chainer.Chain):
    def __init__(self, output_dim):
        super(Alex, self).__init__()
        self.output_dim = output_dim
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, (1, 3), 1, (0, 1))
            self.conv2 = L.Convolution2D(None, 256, (1, 3), 1, (0, 1))
            self.conv3 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1))
            self.conv4 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1))
            self.conv5 = L.Convolution2D(None, 256, (1, 3), 1, (0, 1))
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, output_dim)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), (1,3), stride=1)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), (1,3), stride=1)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h



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
args = parser.parse_args()





###### SETUP DATASET #####
noise_levels = [-18, 0]
all_accs = []

for noise_level in noise_levels:



    RFdata_train = dataset.RFModLabeled(noise_levels=[noise_level], test=False)
    RFdata_test = dataset.RFModLabeled(noise_levels=[noise_level], test=True)

    print np.max(RFdata_train.xs)
    # RFdata_train.xs /= float(np.max(RFdata_train.xs))
    # RFdata_test.xs /= float(np.max(RFdata_test.xs))

    num_classes = np.unique(RFdata_train.ys).shape[0]

    RFdata_train = chainer.datasets.TupleDataset(RFdata_train.xs, RFdata_train.ys)
    RFdata_test = chainer.datasets.TupleDataset(RFdata_test.xs, RFdata_test.ys)


    # train model
    model = L.Classifier(Alex(num_classes))
    if args.gpu >= 0:
    	chainer.cuda.get_device_from_id(args.gpu).use()
    	model.to_gpu()


    serializers.load_npz(args.model, model)


    x, y = RFdata_test._datasets[0], RFdata_test._datasets[1]
    xp = np if args.gpu < 0 else cupy

    pred_ys = xp.zeros(y.shape)


    chainer.config.train = False
    for i in range(0, len(x), args.batchsize):
        x_batch = xp.array(x[i:i + args.batchsize])
        y_batch = xp.array(y[i:i + args.batchsize])
        y_pred = model.predictor(x_batch)
        acc = model.accfun(y_pred, y_batch)
        acc = chainer.cuda.to_cpu(acc.data)
        # print "Accuracy: ", acc
        pred_ys[i:i + args.batchsize] = np.argmax(y_pred._data[0], axis=1)
    chainer.config.train = True


    # np.savez(os.path.join(args.out,'pred_ys__model_%s__noise_%d.npz' % (args.model.replace('/','_'), noise_level)), pred_ys = chainer.cuda.to_cpu(pred_ys))

    cm = metrics.confusion_matrix(chainer.cuda.to_cpu(y), chainer.cuda.to_cpu(pred_ys))
    print cm
    util.graph_confusion_matrix(cm, os.path.join(args.out,'confusion_matrix__noiselevel%d.png' % (noise_level)), title='Confusion Matrix Noise Level %d' % (noise_level))



    cor = np.sum(np.diag(cm))
    ncor = np.sum(cm) - cor
    overall_acc = cor / float(cor+ncor)
    print "Overall Accuracy: ", overall_acc

    all_accs.append(overall_acc)


plt.plot(noise_levels, all_accs)
plt.xlabel('Evaluation SNR')
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy for Different Evaluation SNRs')
plt.savefig(os.path.join(args.out,'classification_acc_different_snrs_200epochs.png'))









