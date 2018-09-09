import gzip
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    # initialze your data, download, etc
    def __init__(self, filename="./data/shakespeare.txt.gz"):
        with gzip.open(filename, 'rt') as f:
            self.targetLines = [x.strip() for x in f if x.strip()]
            self.srcLines = [x.lower().replace(' ','')
                             for x in self.targetLines]
            self.len = len(self.srcLines)

    def __getitem__(self, index):
        return self.srcLines[index], self.targetLines[index]

    def __len__(self):
        return self.len

# Test the loader
if __name__ == "__main__":
    dataset = TextDataset()
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        num_workers=2
        # 多线程读取数据的线程数
    )

    for i, (src, target) in enumerate(train_loader):
        print(i, "data", src, target)

    # print(dataset.len)
