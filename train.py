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

from dragan.updater import Updater
from common.dataset import Cifar10Dataset, Dataset, RFModLabeled
from common.evaluation import sample_generate, sample_generate_light
from common.record import record_setting
import common.net
from tftb.generators import amgauss, fmlin

def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer



parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--algorithm', '-a', type=str, default="dcgan", help='GAN algorithm')
parser.add_argument('--architecture', type=str, default="dcgan", help='Network architecture')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
parser.add_argument('--evaluation_interval', type=int, default=10000, help='Interval of evaluation')
parser.add_argument('--display_interval', type=int, default=100, help='Interval of displaying log to console')
parser.add_argument('--n_dis', type=int, default=5, help='number of discriminator update per generator update')
parser.add_argument('--gamma', type=float, default=0.5, help='hyperparameter gamma')
parser.add_argument('--lam', type=float, default=10, help='gradient penalty')
parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the discriminator (for cramer GAN)')

args = parser.parse_args()
record_setting(args.out)
report_keys = ["loss_dis", "loss_gen", "loss_color"]


# Set up dataset
num_samp = 2**13
sample_width = 512
z = fmlin(num_samp, 0.01, .1)[0]
num_amps = 5.
amps = np.linspace(1./num_amps, 1., num_amps)
num_samp = 2**13
z = fmlin(num_samp, 0.01, .1)[0]
zs = np.array([z*amp for amp in amps])
#zr = np.array([zi.real for zi in z]).reshape(1, 1, 1, -1)

data_set = "rfradio-mod"
if data_set == "chirp":
    xs = []
    ys = []
    for i, z in enumerate(zs):
        zr = np.array([[zi.real, zi.imag] for zi in z]).T.reshape(1, 1, 2, -1)
        x = F.im2col(Variable(zr), ksize=(1, 256)).data
        x = x.transpose(3, 0, 2, 1)
        xs.append(x)
        ys.append(np.array([amps[i]]*x.shape[0]))
    xs = np.vstack((xs))
    ys = np.hstack((ys)) ## labels
else:  ## MOdulation RF dataset
    train_dataset_labeled = RFModLabeled(noise_level=10, class_set=['QPSK'])
    dataset_max = np.max(train_dataset_labeled.xs)
    train_dataset_labeled.xs = train_dataset_labeled.xs/np.max(train_dataset_labeled.xs)


train_dataset = Dataset(train_dataset_labeled.xs)
train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

# Setup algorithm specific networks and updaters
models = []
opts = {}
updater_args = {
    "iterator": {'main': train_iter},
    "device": args.gpu
}
sample_width=128
n_hidden=10
make_hidden_f = partial(common.net.standard_make_hidden, n_hidden)
generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=n_hidden, bottom_width=sample_width/8)
discriminator = common.net.WGANDiscriminator(bottom_width=sample_width/8)
models = [generator, discriminator]
report_keys.append("loss_gp")
updater_args["n_dis"] = args.n_dis
updater_args["lam"] = args.lam

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    print("use gpu {}".format(args.gpu))
    for m in models:
        m.to_gpu()

# Set up optimizers
opts["opt_gen"] = make_optimizer(generator, args.adam_alpha, args.adam_beta1, args.adam_beta2)
opts["opt_dis"] = make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2)

updater_args["optimizer"] = opts
updater_args["models"] = models

# Set up updater and trainer
updater = Updater(**updater_args)
trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

# Set up logging
for m in models:
    trainer.extend(extensions.snapshot_object(
        m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
trainer.extend(extensions.LogReport(keys=report_keys,
                                    trigger=(args.display_interval, 'iteration')))
trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
trainer.extend(sample_generate(generator, args.out, train_max=dataset_max), trigger=(args.evaluation_interval, 'iteration'),
               priority=extension.PRIORITY_WRITER)
trainer.extend(sample_generate_light(generator, args.out, train_max=dataset_max), trigger=(args.evaluation_interval // 10, 'iteration'),
               priority=extension.PRIORITY_WRITER)
trainer.extend(extensions.ProgressBar(update_interval=10))

# Run the training
trainer.run()

