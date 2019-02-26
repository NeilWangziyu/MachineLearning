# -*- coding: <utf-8> -*-
class DataLoader:
    def __init__(self):
        self.datafile = 'data/data.txt'
        self.dataset = self.load_data()

    '''加载数据集'''
    def load_data(self):
        dataset = []
        for line in open(self.datafile,encoding='utf-8'):
            line = line.strip().split(',')
            dataset.append([word for word in line[1].split(' ') if 'nbsp' not in word and len(word) < 11])
        return dataset

if __name__ == "__main__":
    dataset = DataLoader().dataset[:1000]
    print(dataset)