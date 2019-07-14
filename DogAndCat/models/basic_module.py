import torch as t
import time

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module, 主要提供save and load
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        '''
        加载哦指定路径的模型
        :param path:
        :return:
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        save mode, name after "module name+time
        :param name:
        :return:
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name +'_'
            name = time.strftime(prefix + "%m%d_%H%M%S.pth")
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr = lr, weight_decay=weight_decay)

class Flat(t.nn.Module):
    '''
    把输入reshape 成batch_size, dim_length
    '''

    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



