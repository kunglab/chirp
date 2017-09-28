import argparse
import os
import sys
from functools import partial

import numpy as np
import chainer
from chainer import training
import chainer.functions as F
from chainer import Variable
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from dragan.updater import LabeledUpdater, RegressionUpdater
from common.dataset import Dataset, LabeledDataset, RFModLabeled
from common.evaluation import sample_generate, sample_generate_light
from common.record import record_setting
from chainer import serializers
import common.net
from tftb.generators import amgauss, fmlin
from common import dataset

def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--algorithm', '-a', type=str, default='dcgan', help='GAN algorithm')
parser.add_argument('--architecture', type=str, default='dcgan', help='Network architecture')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
parser.add_argument('--evaluation_interval', type=int, default=10000, help='Interval of evaluation')
parser.add_argument('--display_interval', type=int, default=100, help='Interval of displaying log to console')
parser.add_argument('--gamma', type=float, default=0.5, help='hyperparameter gamma')
parser.add_argument('--adv_lam', type=float, default=0.1, help='adversarial penalty')
parser.add_argument('--gp_lam', type=float, default=10, help='gradient penalty')
parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the discriminator (for cramer GAN)')

args = parser.parse_args()
results_output_dir = os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'], args.out) 
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir)
record_setting(results_output_dir)
report_keys = ['loss_dis']



noise_levels = [6,8,16,18]
RFdata_train = dataset.RFModLabeled(noise_levels=noise_levels, test=False, class_set=['8PSK'], snr=True)
RFdata_test = dataset.RFModLabeled(noise_levels=noise_levels, test=True, class_set=['8PSK'], snr=True)

train_max = np.max(np.abs(RFdata_train.xs))
RFdata_train.xs = RFdata_train.xs/train_max


augment = True

if augment:
    chainer.config.train = False
    chainer.config.enable_backprop = False
    def make_hidden(n_hidden, batchsize):
        zs = np.random.randn(batchsize, n_hidden, 1, 1).astype(np.float32)
        snr_zs = np.random.uniform(6.0, 18.0, (batchsize, 1, 1, 1)).astype(np.float32)
        return np.concatenate((zs, snr_zs), axis=1)

    sample_width = 128
    num_classes = 1
    n_hidden = 32
    make_hidden_f = partial(make_hidden, n_hidden)
    generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=make_hidden_f(1).shape[1],
                                        bottom_width=sample_width/8)
    

    serializers.load_npz(os.path.join(os.environ['KUNGLAB_SHARE_RESULTS'],'results_acgenerator_for_regression/DCGANGenerator_50000.npz'), generator)
    generator.to_gpu()
    xp = generator.xp

    num_per_snr = 500
    #snr_ranges = range(6,20,2)
    snr_ranges = [10,12,14]

    xs = []
    ys = []
    for snr in snr_ranges:
        z = generator.make_hidden(num_per_snr)
        z[:,-1,:,:] = snr
        z = Variable(xp.asarray(z))
        x = generator(z)
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape(x.shape[0], 1, x.shape[2], x.shape[3])
        ys.append([snr]*num_per_snr)
        xs.append(x)

    xs = np.vstack((xs))
    ys = np.hstack((ys))
    print ys.shape
    RFdata_train.xs = np.vstack((RFdata_train.xs, xs))
    print RFdata_train.ys.shape
    RFdata_train.ys = np.hstack((RFdata_train.ys, ys))
    print RFdata_train.ys.shape

chainer.config.train = True 
chainer.config.enable_backprop = True 

train_iter = chainer.iterators.SerialIterator(RFdata_train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(RFdata_test, args.batchsize,
                                             repeat=False, shuffle=False)

sample_width = 128
discriminator = common.net.LabeledDiscriminatorJoined(bottom_width=sample_width/8)

models = [discriminator]
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    print('use gpu {}'.format(args.gpu))
    for m in models:
        m.to_gpu()

updater = RegressionUpdater(**{
    'models': models,
    'optimizer': {
        'opt_dis': make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2),
    },
    'iterator': {'main': train_iter},
    'device': args.gpu,
})

trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=results_output_dir)

for m in models:
    trainer.extend(extensions.snapshot_object(
        m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))

# trainer.extend(extensions.Evaluator(test_iter, models[0], device=args.gpu))
trainer.extend(extensions.LogReport(keys=report_keys,
                                    trigger=(args.display_interval, 'iteration')))
trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
trainer.extend(extensions.ProgressBar(update_interval=10))


# Run the training
trainer.run()

