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

            y_mid = self.dis(x_perturb)
            dydx = self.dis.differentiable_backward(y_mid)
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
        self.n_labels = kwargs.pop('n_labels')
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

        y_fake_class = F.reshape(z[:, -self.n_labels:, 0, 0], (-1, self.n_labels))
        y_real_class = xp.asarray(y).astype('f').reshape(-1, self.n_labels)

        loss_g_class = self.class_error_f(yh_fake_class, y_fake_class)
        loss_gen = self.adv_lam*(F.sum(F.softplus(-yh_fake)) / batchsize) + loss_g_class

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        x_fake.unchain_backward()

        loss_d_class = self.class_error_f(yh_real_class, y_real_class)

        yh_perturb, yh_perturb_class = self.dis(x_perturb)
        dydx = self.dis.differentiable_backward(xp.ones_like(yh_perturb), xp.ones_like(yh_perturb_class))
        loss_gp = self.gp_lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

        loss_dis  = self.adv_lam*(F.sum(F.softplus(-yh_real)) / batchsize)
        loss_dis += self.adv_lam*(F.sum(F.softplus(yh_fake))  / batchsize)
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


class AlphaUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.gp_lam = kwargs.pop('gp_lam')
        self.adv_lam = kwargs.pop('adv_lam')
        self.class_error_f = kwargs.pop('class_error_f')
        self.n_labels = kwargs.pop('n_labels')
        self.n_noise = kwargs.pop('n_noise')
        super(AlphaUpdater, self).__init__(*args, **kwargs)
    
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

        yh_fake, yh_fake_z = self.dis(x_fake)
        yh_real, yh_real_z = self.dis(x_real)
        xh_real = self.gen(yh_real_z)
        yh_fake_class = yh_fake_z[:, self.n_noise:]
        yh_fake_class.data = xp.ascontiguousarray(yh_fake_class.data)
        yh_fake_noise = yh_fake_z[:, :self.n_noise]
        yh_fake_noise.data = xp.ascontiguousarray(yh_fake_noise.data)
        yh_real_class = yh_real_z[:, self.n_noise:]
        yh_real_class.data = xp.ascontiguousarray(yh_real_class.data)
        yh_real_noise = yh_real_z[:, :self.n_noise]
        yh_real_noise.data = xp.ascontiguousarray(yh_real_noise.data)


        y_fake_noise = F.reshape(z[:, :self.n_noise, 0, 0], (-1, self.n_noise))
        y_fake_class = F.reshape(z[:, self.n_noise:, 0, 0], (-1, self.n_labels))
        y_real_class = xp.asarray(y).astype('f').reshape(-1, self.n_labels)

        loss_g_class = self.class_error_f(yh_fake_class, y_fake_class)
        loss_gen = self.adv_lam*(F.sum(F.softplus(-yh_fake)) / batchsize)
        #loss_gen = self.adv_lam*(F.sum(F.softplus(-yh_fake)) / batchsize) + loss_g_class

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        x_fake.unchain_backward()

        loss_d_class = self.class_error_f(yh_real_class, y_real_class)
        loss_noise = F.mean_absolute_error(yh_fake_noise, y_fake_noise)
        loss_recon = F.mean_absolute_error(x_real, xh_real)

        yh_perturb, yh_perturb_class = self.dis(x_perturb)
        dydx = self.dis.differentiable_backward(xp.ones_like(yh_perturb), xp.ones_like(yh_perturb_class))
        loss_gp = self.gp_lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

        loss_dis  = self.adv_lam*(F.sum(F.softplus(-yh_real)) / batchsize)
        loss_dis += self.adv_lam*(F.sum(F.softplus(yh_fake))  / batchsize)
        #loss_dis += loss_d_class
        loss_dis += loss_noise
        loss_dis += loss_recon
        loss_dis += loss_gp
 
        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({
            'loss_gen': loss_gen,
            'loss_gen_c': loss_g_class,
            'loss_dis': loss_dis,
            'loss_dis_c': loss_d_class,
            'loss_noise': loss_noise,
            'loss_recon': loss_recon,
            'loss_gp': loss_gp
        })


class AlphaACUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.gp_lam = kwargs.pop('gp_lam')
        self.adv_lam = kwargs.pop('adv_lam')
        self.class_error_f = kwargs.pop('class_error_f')
        self.n_labels = kwargs.pop('n_labels')
        self.n_noise = kwargs.pop('n_noise')
        super(AlphaACUpdater, self).__init__(*args, **kwargs)
    
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

        yh_fake, yh_fake_z = self.dis(x_fake)
        yh_real, yh_real_z = self.dis(x_real)
        xh_real = self.gen(yh_real_z)
        yh_fake_class = yh_fake_z[:, self.n_noise:]
        yh_fake_class.data = xp.ascontiguousarray(yh_fake_class.data)
        yh_fake_noise = yh_fake_z[:, :self.n_noise]
        yh_fake_noise.data = xp.ascontiguousarray(yh_fake_noise.data)
        yh_real_class = yh_real_z[:, self.n_noise:]
        yh_real_class.data = xp.ascontiguousarray(yh_real_class.data)
        yh_real_noise = yh_real_z[:, :self.n_noise]
        yh_real_noise.data = xp.ascontiguousarray(yh_real_noise.data)


        y_fake_noise = F.reshape(z[:, :self.n_noise, 0, 0], (-1, self.n_noise))
        y_fake_class = F.reshape(z[:, self.n_noise:, 0, 0], (-1, self.n_labels))
        y_real_class = xp.asarray(y).astype('f').reshape(-1, self.n_labels)

        loss_g_class = self.class_error_f(yh_fake_class, y_fake_class)
        # loss_gen = self.adv_lam*(F.sum(F.softplus(-yh_fake)) / batchsize)
        loss_gen = self.adv_lam*(F.sum(F.softplus(-yh_fake)) / batchsize) + loss_g_class

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        x_fake.unchain_backward()

        loss_d_class = self.class_error_f(yh_real_class, y_real_class)
        loss_noise = F.mean_absolute_error(yh_fake_noise, y_fake_noise)
        loss_recon = F.mean_absolute_error(x_real, xh_real)

        yh_perturb, yh_perturb_class = self.dis(x_perturb)
        dydx = self.dis.differentiable_backward(xp.ones_like(yh_perturb), xp.ones_like(yh_perturb_class))
        loss_gp = self.gp_lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

        loss_dis  = self.adv_lam*(F.sum(F.softplus(-yh_real)) / batchsize)
        loss_dis += self.adv_lam*(F.sum(F.softplus(yh_fake))  / batchsize)
        loss_dis += loss_d_class
        loss_dis += loss_noise
        loss_dis += loss_recon
        loss_dis += loss_gp
 
        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({
            'loss_gen': loss_gen,
            'loss_gen_c': loss_g_class,
            'loss_dis': loss_dis,
            'loss_dis_c': loss_d_class,
            'loss_noise': loss_noise,
            'loss_recon': loss_recon,
            'loss_gp': loss_gp
        })

### AKA the "jed" updater
class GEDUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.enc = kwargs.pop('models')
        self.gp_lam = kwargs.pop('gp_lam')
        self.adv_lam = kwargs.pop('adv_lam')
        self.class_error_f = kwargs.pop('class_error_f')
        self.n_labels = kwargs.pop('n_labels')
        self.n_noise = kwargs.pop('n_noise')
        super(GEDUpdater, self).__init__(*args, **kwargs)
    
    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        enc_optimizer = self.get_optimizer('opt_enc')
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

        yh_fake, yh_fake_label = self.dis(x_fake)
        yh_real, yh_real_label = self.dis(x_real)

        _, z_real = self.enc(x_real)
        xh_real = self.gen(z_real)
        _, z_fake = self.enc(x_fake)

        # spliting up z based on conditional format, noise | label 
        y_fake_noise = F.reshape(z[:, :self.n_noise, 0, 0], (-1, self.n_noise))
        y_fake_label = F.reshape(z[:, self.n_noise:, 0, 0], (-1, self.n_labels))

        y_real_label = xp.asarray(y).astype('f').reshape(-1, self.n_labels)

        # ENCODER LOSS
        loss_e_recon = F.mean_absolute_error(xh_real, x_real)
        loss_e_noise = F.mean_absolute_error(z_fake, z.reshape(z.shape[0], z.shape[1]))


        loss_enc = loss_e_recon + 10.*loss_e_noise


        # GENERATOR LOSS - condition piece
        loss_g_class = self.class_error_f(yh_fake_label, y_fake_label)
        # Adversarial Loss + loss_g (condition loss)
        loss_gen = self.adv_lam*(F.sum(F.softplus(-yh_fake)) / batchsize) + loss_g_class + loss_enc

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        x_fake.unchain_backward()

        self.enc.cleargrads()
        loss_enc.backward()
        enc_optimizer.update() 

        loss_d_class = self.class_error_f(yh_real_label, y_real_label)

        yh_perturb, yh_perturb_class = self.dis(x_perturb)
        dydx = self.dis.differentiable_backward(xp.ones_like(yh_perturb), xp.ones_like(yh_perturb_class))
        loss_gp = self.gp_lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

        loss_dis  = self.adv_lam*(F.sum(F.softplus(-yh_real)) / batchsize)
        loss_dis += self.adv_lam*(F.sum(F.softplus(yh_fake))  / batchsize)
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
            'loss_gp': loss_gp,
            'loss_e_recon': loss_e_recon,
            'loss_e_noise': loss_e_noise,
            'loss_enc': loss_enc
        })

class RegressionUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis = kwargs.pop('models')[0]
        super(RegressionUpdater, self).__init__(*args, **kwargs)
    
    def update_core(self):
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.dis.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x, y = [], []
        for i in range(batchsize):
            xi, yi = batch[i]
            x.append(np.asarray(xi).astype("f"))
            y.append(yi)

        y_real = xp.asarray(y).astype('f').reshape(-1, 1)
        x_real = (xp.asarray(x))
        _, y_hat = self.dis(x_real)
        loss_dis = F.mean_absolute_error(y_hat, y_real)
 
        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({
            'loss_dis': loss_dis,
        })