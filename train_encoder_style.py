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
from chainer.dataset import dataset_mixin

from common.net import Alex, VTCNN2
import cupy

from common import dataset


class StyleEncoderDataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        self.xs = np.load('results_style_n18_to_18/rfmod_cond/generated_data_for_encoder_xs_and_zs_100k.npz')['x']
        self.ys = np.load('results_style_n18_to_18/rfmod_cond/generated_data_for_encoder_xs_and_zs_100k.npz')['z']

        print self.xs.shape


        train_size = .8
        if test:
            self.xs = self.xs[int(self.xs.shape[0]*train_size):]
            self.ys = self.ys[int(self.ys.shape[0]*train_size):]
        else:
            self.xs = self.xs[:int(self.xs.shape[0]*train_size)]
            self.ys = self.ys[:int(self.ys.shape[0]*train_size)]
        print("load labeled dataset.  shape: ", self.xs.shape)
        # np.random.seed()
        self.xs = self.xs.astype('float32')
        self.ys = self.ys.astype('float32')
        # self.ys = self.ys.astype('int32')

    def __len__(self):
        return self.xs.shape[0]

    def get_example(self, i):
        return self.xs[i], self.ys[i]




model_map = {"AlexStock": Alex, "VTCNN2" : VTCNN2}

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result_classifier',
                    help='Directory to output the result')
parser.add_argument('--unit', '-u', type=int, default=1000,
                    help='Number of units')
parser.add_argument('--model_type', '-t', type=str, default="AlexStock",
                    help='Which Model to run (AlexStock, VTCNN2)')
args = parser.parse_args()


if not os.path.exists(args.out):
    os.makedirs(args.out)

###### SETUP DATASET #####
# data = np.load('generated_data_for_encoder_xs_and_zs.npz')

# Xs = data['x']
# Zs = data['z']

# train = chainer.datasets.TupleDataset(Xs[:80000], Zs[:80000])
# test = chainer.datasets.TupleDataset(Xs[80000:], Zs[80000:])

RFdata_train = StyleEncoderDataset(test=False)
RFdata_test = StyleEncoderDataset(test=True)


num_classes = RFdata_train.ys.shape[1]

# train model
#model = L.Classifier(Alex(num_classes))
# model = L.Classifier(VTCNN2(num_classes))
model = L.Classifier(model_map[args.model_type](num_classes), lossfun=chainer.functions.mean_squared_error, accfun=chainer.functions.mean_squared_error)
if args.gpu >= 0:
	chainer.cuda.get_device_from_id(args.gpu).use()
	model.to_gpu()


#optimizer = chainer.optimizers.Adam(alpha=0.0001, beta1=0.0, beta2=.9)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
train_iter = chainer.iterators.SerialIterator(RFdata_train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(RFdata_test, args.batchsize,
                                             repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())
trainer.run()

serializers.save_npz(os.path.join(args.out, 'Encoder_from60000model_1.npz'), model)

'''
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


np.savez(os.path.join(args.out,'pred_ys__main_classifer_greaterthan10_regularized.npz'), pred_ys = chainer.cuda.to_cpu(pred_ys))

cm = metrics.confusion_matrix(chainer.cuda.to_cpu(y), chainer.cuda.to_cpu(pred_ys))
print cm

cor = np.sum(np.diag(cm))
ncor = np.sum(cm) - cor
print "Overall Accuracy: ", cor / float(cor+ncor)
'''


