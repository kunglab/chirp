import os
import sys
import math
sys.path.append(os.path.dirname(__file__))

import numpy as np
from PIL import Image
import scipy.linalg

import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers
import chainer.functions as F
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plot_markers = ['x', '+']

def encdis_generate_light(gen, dis, dst, train_max=1, nsamples=16, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(nsamples)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            np.random.seed(seed)
            x = gen(z)
            _, yh_fake_class = dis(x)
            xh = gen(yh_fake_class)
            np.random.seed()
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x*train_max
        xh = chainer.cuda.to_cpu(xh.data)
        xh = xh.reshape(xh.shape[0], xh.shape[2], xh.shape[3])
        xh = xh*train_max
        z = chainer.cuda.to_cpu(z.data)
        yh_fake_class = chainer.cuda.to_cpu(yh_fake_class.data)
        ones = np.arange(yh_fake_class.shape[1])

        fig = plt.figure(figsize=(30,15))
        for i in range(nsamples):
            ax = plt.subplot(4, 8, i*2+1)
            #for j, xi in enumerate(x[i]):
            ax.plot(x[i][0], '-x')
            ax.plot(xh[i][0], '-x')
            ax = plt.subplot(4, 8, i*2+2)
            ax.plot(ones[:20], z[i].flatten()[:20], 'g-x')
            ax.plot(ones[:20], yh_fake_class[i].flatten()[:20], 'r-+')
        plt.tight_layout()

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_latest.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path, dpi=100)
        plt.clf()
        plt.close(fig)

    return make_image

def encdis_generate(gen, dis, dst, train_max=1, nsamples=16, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(nsamples)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            np.random.seed(seed)
            x = gen(z)
            _, yh_fake_class = dis(x)
            xh = gen(yh_fake_class)
            np.random.seed()
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x*train_max
        xh = chainer.cuda.to_cpu(xh.data)
        xh = xh.reshape(xh.shape[0], xh.shape[2], xh.shape[3])
        xh = xh*train_max
        z = chainer.cuda.to_cpu(z.data)
        yh_fake_class = chainer.cuda.to_cpu(yh_fake_class.data)
        ones = np.arange(yh_fake_class.shape[1])

        fig = plt.figure(figsize=(30,15))
        for i in range(nsamples):
            ax = plt.subplot(4, 8, i*2+1)
            #for j, xi in enumerate(x[i]):
            #    ax.plot(xi, '-', marker=plot_markers[j])
            ax.plot(x[i][0], '-x')
            ax.plot(xh[i][0], '-x')
            ax = plt.subplot(4, 8, i*2+2)
            ax.plot(ones[:20], z[i].flatten()[:20], 'g-x')
            ax.plot(ones[:20], yh_fake_class[i].flatten()[:20], 'r-+')
        plt.tight_layout()

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path, dpi=100)
        plt.clf()
        plt.close(fig)

    return make_image


def rfmod_generate_light(gen, dis, dst, train_max=1, nsamples=16, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(nsamples)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            np.random.seed(seed)
            x = gen(z)
            _, yh_fake_class = dis(x)
            xh = gen(yh_fake_class)
            np.random.seed()
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x*train_max
        xh = chainer.cuda.to_cpu(xh.data)
        xh = xh.reshape(xh.shape[0], xh.shape[2], xh.shape[3])
        xh = xh*train_max
        z = chainer.cuda.to_cpu(z.data)
        yh_fake_class = chainer.cuda.to_cpu(yh_fake_class.data)
        ones = np.arange(yh_fake_class.shape[1])

        fig = plt.figure(figsize=(30,15))
        for i in range(nsamples):
            ax = plt.subplot(4, 8, i*2+1)
            #for j, xi in enumerate(x[i]):
            ax.plot(x[i][0], '-x')
            ax.plot(xh[i][0], '-x')
            ax = plt.subplot(4, 8, i*2+2)
            ax.plot(ones[:20], z[i].flatten()[:20], 'g-x')
            ax.plot(ones[:20], yh_fake_class[i].flatten()[:20], 'r-+')
        plt.tight_layout()

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_latest.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path, dpi=100)
        plt.clf()
        plt.close(fig)

    return make_image

def rfmod_generate(gen, dis, dst, train_max=1, nsamples=16, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(nsamples)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            np.random.seed(seed)
            x = gen(z)
            _, yh_fake_class = dis(x)
            xh = gen(yh_fake_class)
            np.random.seed()
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x*train_max
        xh = chainer.cuda.to_cpu(xh.data)
        xh = xh.reshape(xh.shape[0], xh.shape[2], xh.shape[3])
        xh = xh*train_max
        z = chainer.cuda.to_cpu(z.data)
        yh_fake_class = chainer.cuda.to_cpu(yh_fake_class.data)
        ones = np.arange(yh_fake_class.shape[1])

        fig = plt.figure(figsize=(30,15))
        for i in range(nsamples):
            ax = plt.subplot(4, 8, i*2+1)
            #for j, xi in enumerate(x[i]):
            #    ax.plot(xi, '-', marker=plot_markers[j])
            ax.plot(x[i][0], '-x')
            ax.plot(xh[i][0], '-x')
            ax = plt.subplot(4, 8, i*2+2)
            ax.plot(ones[:20], z[i].flatten()[:20], 'g-x')
            ax.plot(ones[:20], yh_fake_class[i].flatten()[:20], 'r-+')
        plt.tight_layout()

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path, dpi=100)
        plt.clf()
        plt.close(fig)

    return make_image

def sample_generate_light(gen, dst, train_max=1, nsamples=25, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(nsamples)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x*train_max
        np.random.seed()

        fig = plt.figure(figsize=(24,16))
        for i in range(nsamples):
            ax = plt.subplot(5, 5, i+1)
            for j, xi in enumerate(x[i]):
                # shift up by 2 to visualize
                ax.plot(xi + 2*j, '-', marker=plot_markers[j])
        plt.tight_layout()

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_latest.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path)
        plt.clf()

    return make_image


def sample_generate(gen, dst, train_max=1, nsamples=25, seed=0):
    """Visualization of rows*cols images randomly generated by the generator."""
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(nsamples)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x*train_max
        np.random.seed()

        fig = plt.figure(figsize=(24,16))
        for i in range(nsamples):
            ax = plt.subplot(5, 5, i+1)
            for j, xi in enumerate(x[i]):
                # shift up by 2 to visualize
                ax.plot(xi + 2*j, '-', marker=plot_markers[j])
        plt.tight_layout()

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path)
        plt.clf()

    return make_image


def get_mean_cov(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = model.xp

    print('Batch size:', batch_size)
    print('Total number of images:', n)
    print('Total number of batches:', n_batches)

    ys = xp.empty((n, 2048), dtype=xp.float32)

    for i in range(n_batches):
        print('Running batch', i + 1, '/', n_batches, '...')
        batch_start = (i * batch_size)
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = xp.asarray(ims_batch)  # To GPU if using CuPy
        ims_batch = Variable(ims_batch)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = model(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = y.data

    mean = xp.mean(ys, axis=0).get()
    # cov = F.cross_covariance(ys, ys, reduce="no").data.get()
    cov = np.cov(ys.get().T)

    return mean, cov
