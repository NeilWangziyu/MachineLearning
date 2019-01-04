import warnings
import torch as t

class DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    model = 'SqueezeNet'

    train_data_root = './DATASET_TRAIN/'
    test_data_root = './DATASET_TEST/'
    load_model_path = None

    batch_size = 32  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch


    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001
    lr_decay = 0.5

    weight_decay = 0e-5

    def _parse(self, kwags):
        '''
        根据字典kwargs 更新 config参数
        :param kwags:
        :return:
        '''
        for k, v in kwags.items():
            if not hasattr(self, k):
                warnings.warn("warning:opt has not attribut % s" %k)

            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k , getattr(self, k))

opt = DefaultConfig()
