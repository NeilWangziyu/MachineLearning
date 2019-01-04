import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    封装了visdom的基本操作，但仍可以通过‘self.vis.function'
    or ‘self.function’调用原生接口
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10), t.arange(1, 11))
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # save 'loss' 23, which means the 23th point of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        modify visdom config
        :param env:
        :param kwargs:
        :return:
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self


    def plot_many(self, d):
        '''
        plot several fig
        @ parm d:dict(name, value) i.e. ('loss', 0.11)
        :param d:
        :return:
        '''
        for k, v in d.item():
            self.plot(k, v)


    def img_many(self, d):
        for k, v in d.item():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        :param name:
        :param y:
        :param kwargs:
        :return:
        '''
        x = self.index.get(name, 0)

        self.vis.line(Y= np.array([y]), X=np.array([x]),
        win=(name), opts = dict(title=name), update = None if x == 0 else 'append',
        **kwargs
        )

        self.index[name] = x + 1

    def img(self, name, img_,  **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))

        :param name:
        :param img_:
        :param kwargs:
        :return:
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        :param info:
        :param win:
        :return:
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),info=info
        ))
        self.vis.text(self.log_text, win)


    def __getattr__(self, name):
        return getattr(self.vis, name)















