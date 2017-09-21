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
from common.net import Alex, VTCNN2
import utilities as util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
noise_levels = range(-18,20,2)
all_accs = []

for noise_level in noise_levels:

    RFdata_train = dataset.RFModLabeled(noise_levels=[noise_level], test=False)
    RFdata_test = dataset.RFModLabeled(noise_levels=[noise_level], test=True)

    num_classes = np.unique(RFdata_train.ys).shape[0]

    # train model
    model = L.Classifier(model_map[args.model_type](num_classes))
    if args.gpu >= 0:
    	chainer.cuda.get_device_from_id(args.gpu).use()
    	model.to_gpu()


    serializers.load_npz(args.model, model)


    x, y = RFdata_test.xs, RFdata_test.ys
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



    cm = metrics.confusion_matrix(chainer.cuda.to_cpu(y), chainer.cuda.to_cpu(pred_ys))
    print cm
    util.graph_confusion_matrix(cm, os.path.join(args.out,'confusion_matrix__noiselevel%d.png' % (noise_level)), title='Confusion Matrix Noise Level %d' % (noise_level))



    cor = np.sum(np.diag(cm))
    ncor = np.sum(cm) - cor
    overall_acc = cor / float(cor+ncor)
    print "Overall Accuracy: ", overall_acc

    all_accs.append(overall_acc)


plt.figure()
plt.plot(noise_levels, all_accs)
plt.xlabel('Evaluation SNR')
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy for Different Evaluation SNRs')
plt.ylim([0,1])
plt.grid()
plt.savefig(os.path.join(args.out,'classification_acc_VTCNN2_init.png'))
