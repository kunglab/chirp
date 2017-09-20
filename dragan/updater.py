import numpy as np

import sys
import os

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.lam = kwargs.pop('lam')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp

        for it in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x = []
            for i in range(batchsize):
                x.append(np.asarray(batch[i]).astype("f"))
            x_real = (xp.asarray(x))
            std_x_real = xp.std(x_real, axis=0, keepdims=True)
            rnd_x = xp.random.uniform(0, 1, x_real.shape).astype("f")
            x_perturb = Variable(x_real + 0.5 * rnd_x * std_x_real)

            y_real = self.dis(x_real)

            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z)
            y_fake = self.dis(x_fake)

            loss_dis = F.sum(F.softplus(-y_real)) / batchsize
            loss_dis += F.sum(F.softplus(y_fake)) / batchsize

            y_mid = F.sigmoid(self.dis(x_perturb))
            dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))
            dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
            loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

            if it == 0:
                loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({
                    'loss_gen': loss_gen
                })
            x_fake.unchain_backward()

            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'loss_dis': loss_dis})
            chainer.reporter.report({'loss_gp': loss_gp})

class LabeledUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.gp_lam = kwargs.pop('gp_lam')
        self.adv_lam = kwargs.pop('adv_lam')
        self.class_error_f = kwargs.pop('class_error_f')
        super(LabeledUpdater, self).__init__(*args, **kwargs)
    
    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x, y = [], []
        for i in range(batchsize):
            xi, yi = batch[i]
            x.append(np.asarray(xi).astype("f"))
            y.append(yi)
        z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        x_fake = self.gen(z)
        x_real = (xp.asarray(x))
        std_x_real = xp.std(x_real, axis=0, keepdims=True)
        rnd_x = xp.random.uniform(0, 1, x_real.shape).astype('f')
        x_perturb = Variable(x_real + 0.5 * rnd_x * std_x_real)

        yh_fake, yh_fake_class = self.dis(x_fake)
        yh_real, yh_real_class = self.dis(x_real)

        y_fake_class = F.reshape(z[:, -1, 0, 0], (-1, 1))
        y_real_class = xp.asarray(y).astype('f').reshape(-1, 1)

        loss_g_class = self.class_error_f(y_fake_class, yh_fake_class)
        loss_gen = self.adv_lam*(F.sum(F.softplus(-yh_fake)) / batchsize) + loss_g_class

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        x_fake.unchain_backward()

        loss_d_class = self.class_error_f(y_real_class, yh_real_class)

        y_mid = F.sigmoid(self.dis(x_perturb)[0])
        dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))
        dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
        loss_gp = self.gp_lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

        loss_dis  = self.adv_lam*F.sum(F.softplus(-yh_real)) / batchsize 
        loss_dis += self.adv_lam*F.sum(F.softplus(yh_fake)) / batchsize
        loss_dis += loss_d_class
        loss_dis += loss_gp
 
        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({
            'loss_gen': loss_gen,
            'loss_gen_c': loss_g_class,
            'loss_dis': loss_dis,
            'loss_dis_c': loss_d_class,
            'loss_gp': loss_gp
        })
