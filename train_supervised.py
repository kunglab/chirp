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

from dragan.updater import LabeledUpdater
from common.dataset import Dataset, LabeledDataset
from common.evaluation import rfmod_generate, rfmod_generate_light
from common.record import record_setting
import common.net
import dataset

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
record_setting(args.out)
report_keys = ['loss_dis', 'loss_gen', 'loss_gen_c', 'loss_dis_c', 'loss_gp']


noise_levels = range(-18, 20, 2)
train_dataset = dataset.RFModLabeled(noise_levels=noise_levels, test=False)
num_classes = np.unique(train_dataset.ys).shape[0]

train_max = np.max(np.abs(train_dataset.xs))
train_dataset.xs /= train_max
# make 1-hot (-1, 1)
train_dataset.ys = F.embed_id(train_dataset.ys, np.identity(num_classes, dtype=np.float32)).data
train_dataset.ys[train_dataset.ys < 1] = -1
train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

def make_hidden(n_hidden, batchsize):
    zs = np.random.randn(batchsize, n_hidden, 1, 1).astype(np.float32)
    ys = np.random.randint(0, num_classes, batchsize, dtype=np.int32)
    label_zs = F.embed_id(ys, np.identity(num_classes, dtype=np.float32)).data
    label_zs[label_zs < 1] = -1
    label_zs = label_zs.reshape(label_zs.shape[0], label_zs.shape[1], 1, 1)
    return np.concatenate((zs, label_zs), axis=1)

def loss_sigmoid_cross_entropy_with_logits(x, t):
    # print 'pred: ', x.data[0]
    # print 'real: ', t[0]
    # print
    return F.average(F.clip(x, 0.0, 1e10) - x*t + F.softplus(-x))

sample_width = train_dataset.xs.shape[3]
n_hidden = 32
make_hidden_f = partial(make_hidden, n_hidden)
generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=make_hidden_f(1).shape[1],
                                      bottom_width=sample_width/8)
discriminator = common.net.LabeledDiscriminator(bottom_width=sample_width/8, n_labels=num_classes)
models = []
models = [generator, discriminator]
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    print('use gpu {}'.format(args.gpu))
    for m in models:
        m.to_gpu()

updater = LabeledUpdater(**{
    'models': models,
    'optimizer': {
        'opt_gen': make_optimizer(generator, args.adam_alpha, args.adam_beta1, args.adam_beta2),
        'opt_dis': make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2),
    },
    'iterator': {'main': train_iter},
    'device': args.gpu,
    'gp_lam': args.gp_lam,
    'adv_lam': args.adv_lam,
    'class_error_f': loss_sigmoid_cross_entropy_with_logits,
    'n_labels': make_hidden_f(1).shape[1] - n_hidden
})
trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

for m in models:
    trainer.extend(extensions.snapshot_object(
        m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
trainer.extend(extensions.LogReport(keys=report_keys,
                                    trigger=(args.display_interval, 'iteration')))
trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
trainer.extend(extensions.ProgressBar(update_interval=10))

# visualization functions
trainer.extend(rfmod_generate(generator, discriminator, args.out, train_max=train_max),
               trigger=(args.evaluation_interval, 'iteration'),
               priority=extension.PRIORITY_WRITER)
trainer.extend(rfmod_generate_light(generator, discriminator, args.out, train_max=train_max),
               trigger=(args.evaluation_interval // 2, 'iteration'),
               priority=extension.PRIORITY_WRITER)

# Run the training
trainer.run()

