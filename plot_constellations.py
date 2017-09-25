import argparse
import os
import sys

from functools import partial
import numpy as np
import chainer
import chainer.functions as F
from chainer import serializers
from chainer import Variable
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import common.net
import load_models

x_cir = [2., 1.414, 0, -1.414, -2, -1.414, 0, 1.414]
y_cir = [0., 1.414, 2, 1.414, 0, -1.414, -2., -1.414]

def plotConstellationData(data, mod_type='8PSK'):
    if mod_type == "8PSK":
        # x_cir = [2., 1.414, 0, -1.414, -2, -1.414, 0, 1.414]
        # y_cir = [0., 1.414, 2, 1.414, 0, -1.414, -2., -1.414]
        x = data[:,0]
        y = data[:,1]
        #fig = plt.figure(figsize=(16,12))
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.grid()

    else:
        print "No constellation for this modulation type"
        raise NotImplementedError



parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
args = parser.parse_args()


## Individual comparisons - orig - recon
recon_l = ['SNR6', 'SNR12', 'SNR18']
for r in recon_l:
    data_dir = "plot_data/data_turnto_consellation/" + r + "_original.npz"
    data = np.load(data_dir)['arr_0']
    x = data[:,0]
    y = data[:,1]
    fig = plt.figure(figsize=(16,12))
    ax = plt.subplot(2,2,1)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    line, = plt.plot(x, y, '-o')
    plt.plot(x_cir, y_cir, 'ro', markersize=14)
    plt.grid()
    plt.subplot(2,2,2)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.plot(x, '-+')
    plt.plot(y, '-x')
    plt.grid()
    plt.tight_layout()


    data_dir = "plot_data/data_turnto_consellation/" + r + "_recon.npz"
    data = np.load(data_dir)['arr_0']
    x = data[:,0]
    y = data[:,1]
    ax = plt.subplot(2,2,3)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    line, = plt.plot(x, y, '-o')
    plt.plot(x_cir, y_cir, 'ro', markersize=14)
    plt.grid()
    plt.subplot(2,2,4)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.plot(x, '-+')
    plt.plot(y, '-x')
    plt.grid()
    plt.tight_layout()
    plt.savefig('figures/const_' + r + '.png')

# assert False

# x_cir = [1., 0.707, 0, -0.707, -1, -0.707, 0, 0.707]
# y_cir = [0., 0.707, 1, 0.707, 0, -0.707, -1., -0.707]
file_l = ['SNR6_animation.npz', 'SNR_interpolation.npz', 'SNR12_animation.npz', 'SNR18_animation.npz']
for fh in file_l:
    # x = np.load("plot_data/data_turnto_consellation/SNR_interpolation.npz")['arr_0']
    x = np.load("plot_data/data_turnto_consellation/" + fh)['arr_0']
    x = x*2.25
    x = x[:64,:,:]
    fig = plt.figure()
    ax = plt.axes()
    line, = plt.plot(x[0,0,:], x[0,1,:], '-o', animated=True)
    plt.xlim([-2,2])
    plt.ylim([-2,2])

    def init():
        plt.plot(x_cir, y_cir, 'ro', markersize=14)
        line.set_xdata(x[0,0,:])
        line.set_ydata(x[0,1,:])
        return line,

    def func(i):
        print 'frame: ', i
        # 'pause' animation at beginning and end
        if i < 5:
            i = 0
        if i >= x.shape[0]:
            i = x.shape[0] - 1

        line.set_xdata(x[i,0,:])
        line.set_ydata(x[i,1,:])
        return line,

    out_dir = "figures/constellations"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #plt.axis('off')
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, func, init_func=init, frames=x.shape[0]+10, interval=100, blit=True)
    ani.save(os.path.join(out_dir, 'const_'+fh+'_.gif'), dpi=50, writer='imagemagick')