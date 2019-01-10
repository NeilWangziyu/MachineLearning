import sys, os
import torch as t
from data import get_data
from model import PoetryModel
from torch import nn
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb

class Config(object):
    data_path = '/data'
    pickle_path= 'tang.npz'
    author = None       # 只学习某位作者的诗歌
    constrain = None    # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = False
    epoch = 20
    batch_size = 128
    maxlen = 125
    plot_every = 20
    env = 'poetry'
    max_gen_len = 200
    debug_file = '/tmp/debugp'
    model_path = None
    prefix_words = '细雨鱼儿出,微风燕子斜。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'
    acrostic = False
    model_prefix = 'checkpoints/tang'

opt = Config()


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """

    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：
    """

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>

    input = t.Tensor([word2ix['<START> ']]).view(1,1).long()
    if opt.use_gpu:input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output , hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1,1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
        生成藏头诗
        start_words : u'深度学习'
        生成：
        深木通中岳，青苔半日脂。
        度山分地险，逆浪到南巴。
        学道兵犹毒，当时燕不移。
        习根通古岸，开镜出清羸。
        """
    results = []
    start_word_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1,1).long())
    if opt.use_gpu:input = input.cuda()
    hidden = None

    index = 0   # 用来指示已经生成了多少句藏头诗
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)


    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topl(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}):
             # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型'
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1,1)
        else:
    # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    device = opt.device
    vis = Visualizer(env=opt.env)


         # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=1)

#     模型定义
    model = PoetryModel(len(word2ix), 128, 256)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(device)

    loss_meter = meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, data_ in  tqdm.tqdm(enumerate(dataloader)):
#             train
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

#             可视化
            if (1 + ii) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                # vis.plot('loss', loss_meter()[0])
                # 诗歌原文
                poetrys = [[ix2word[_word] for _word in data_[:, _iii].tolist()]
                           for _iii in range(data_.shape[1])][:16]
                # vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')
                print('</br>'.join([''.join(poetry) for poetry in poetrys]))

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                # vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')
                print('</br>'.join([''.join(poetry) for poetry in gen_poetries]))


    t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))


def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """
    for k, v in kwargs.items():
        setattr(opt, k, v)
    data, word2ix, ix2word = get_data(opt)










