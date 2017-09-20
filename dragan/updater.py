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
        self.lam = kwargs.pop('lam')
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

        l_adv = 0.1
        loss_g_class = F.mean_absolute_error(y_fake_class, yh_fake_class)
        loss_gen = l_adv*(F.sum(F.softplus(-yh_fake)) / batchsize) + loss_g_class

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        chainer.reporter.report({
            'loss_gen': loss_gen,
            'loss_gen_c': loss_g_class
        })

        x_fake.unchain_backward()

        loss_d_class = F.mean_absolute_error(y_real_class, yh_real_class)

        y_mid = F.sigmoid(self.dis(x_perturb)[0])
        dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))
        dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
        loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

        loss_dis  = l_adv*F.sum(F.softplus(-yh_real)) / batchsize 
        loss_dis += l_adv*F.sum(F.softplus(yh_fake)) / batchsize
        loss_dis += loss_d_class
        loss_dis += loss_gp
 

        #print 'real: ', amp_real[0][0], amp_pred_real.data[0][0]
        #print 'fake: ', amp_fake.data[0][0], amp_pred_fake.data[0][0]

        self.dis.cleargrads()
        loss_dis.backward()
        #loss_gp.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'loss_dis_c': loss_d_class})
        chainer.reporter.report({'loss_gp': loss_gp})


class UpdaterWithGP(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0
        params = kwargs.pop('params')
        self._img_size = params['img_size']
        self._img_chan = params['img_chan']
        self._batch_size = params['batch_size']
        self._latent_len = params['latent_len']
        self._lambda_gp = params['lambda_gp']
        self._lambda_adv = params['lambda_adv']
        self._attr_len = params['attr_len']
        self._threshold = 0.75
        super(UpdaterWithGP, self).__init__(*args, **kwargs)

    def get_real_image_batch(self):
        xp = self.gen.xp
        batch= self.get_iterator('main').next()
        #print(batch)

        t_out = xp.zeros((self._batch_size, self._img_chan, self._img_size, self._img_size)).astype("f")
        tags = xp.zeros((self._batch_size, self._attr_len)).astype("f")
        for i in range(self._batch_size):
            t_out[i, :] = xp.asarray(batch[i][0])
            tags[i, :] = xp.asarray(batch[i][1])
        return t_out, tags

    def get_fake_tag(self):
        prob2 = np.random.rand(self._attr_len)
        tags = np.zeros((self._attr_len)).astype("f")
        tags[:] = -1.0
        tags[np.argmax(prob2[0:13])]=1.0
        tags[27 + np.argmax(prob2[27:])] = 1.0
        prob2[prob2<self._threshold] = -1.0
        prob2[prob2>=self._threshold] = 1.0
        for i in range(13, 27):
            tags[i] = prob2[i]
        return tags

    def get_fake_tag_batch(self):
        xp = self.gen.xp
        tags = xp.zeros((self._batch_size, self._attr_len)).astype("f")
        for i in range(self._batch_size):
            tags[i] = xp.asarray(self.get_fake_tag())
        return tags

    def get_fake_image_batch(self):
        z = self.get_latent_code_batch()
        tag = self.get_fake_tag_batch()
        x_out = self.gen(F.concat([Variable(z, volatile=True), Variable(tag, volatile=True)]), test=True).data
        return x_out, tag

    def get_latent_code_batch(self):
        xp = self.gen.xp
        z_in = xp.random.normal(size=(self._batch_size, self._latent_len)).astype("f")
        return z_in

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        opt_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        data_z = self.get_latent_code_batch()
        data_tag = self.get_fake_tag_batch()

        data_x, data_real_tag = self.get_real_image_batch()

        # x_fake
        x_fake = self.gen(F.concat([Variable(data_z),Variable(data_tag)]))

        # dis predictions for x_fake
        dis_fake, dis_g_class = self.dis(x_fake)
        data_tag[data_tag < 0] = 0.0

        # class loss for x_fake
        loss_g_class = loss_sigmoid_cross_entropy_with_logits(dis_g_class, data_tag)

        loss_gen = self._lambda_adv * loss_func_dcgan_dis_real(dis_fake) + loss_g_class
        chainer.report({'loss': loss_gen, 'loss_c': loss_g_class}, self.gen)

        opt_g.zero_grads()
        loss_gen.backward()
        opt_g.update()

        x_fake.unchain_backward()

        std_data_x = xp.std(data_x, axis=0, keepdims=True)
        rnd_x = xp.random.uniform(0, 1, data_x.shape).astype("f")
        x_perturbed = Variable(data_x + 0.5*rnd_x*std_data_x)

        x_real = Variable(data_x)
        dis_real, dis_d_class = self.dis(x_real)
        dis_perturbed, _ = self.dis(x_perturbed, retain_forward=True)
        g = Variable(xp.ones_like(dis_perturbed.data))
        grad = self.dis.differentiable_backward(g)

        grad_l2 = F.sqrt(F.sum(grad**2, axis=(1, 2, 3)))
        loss_gp = self._lambda_gp * loss_l2(grad_l2, 1.0)

        loss_d_class = loss_sigmoid_cross_entropy_with_logits(dis_d_class, data_real_tag)

        loss_dis = self._lambda_adv * ( loss_func_dcgan_dis_real(dis_real) + \
                    loss_func_dcgan_dis_fake(dis_fake) )+ \
                    loss_d_class + \
                    loss_gp

        opt_d.zero_grads()
        loss_dis.backward()
        opt_d.update()

        chainer.report({'loss': loss_dis, 'loss_gp': loss_gp, 'loss_c': loss_d_class}, self.dis)
