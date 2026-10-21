"""
bert模型数据加载与处理相关逻辑
"""
from torch.utils.data import Dataset, DataLoader
import time
import torch

def data_loader(data_dir):
    """
    给定数据集路径，返回事件列表和标签列表

    1. 打开文件
    2. 遍历，判断为空的情况
    3. 获取text，label
    4. 返回列表
    """
    titles = []
    labels = []
    # 1.
    # 打开文件
    with open(data_dir,'r',encoding='utf-8') as f:
        line_num = 0
        for line in f.readlines():
            line = line.strip()
            line_num += 1

            if line_num > 10000:
                break
            # 2.
            # 遍历，判断为空的情况
            if not line:
                continue
            # 3.
            # 获取text，label
            text,label = line.split('\t')
            titles.append(text)
            labels.append(int(label))
    # 4.
    # 返回列表
    return titles,labels

class NewsDataset(Dataset):
    def __init__(self,titles,labels):
        """
        输入：事件列表和标签列表
        """
        self.titles = titles
        self.labels = labels
    def __len__(self):
        return len(self.titles)
    def __getitem__(self, idx):
        # time.sleep(0.001)
        return self.titles[idx],self.labels[idx]



def build_dataloader(data_dir, batch_size, padding_size, tokenizer ):
    """
    构建数据迭代器

    1. 构建数据集
    2. 把数据集和批处理函数传入dataloader，得到可遍历对象
    """
    # local_func_calls = 0

    def collate_fn(batch, ):
        """
        批处理逻辑

        1. 拿到文本和标签
        2. 处理文本，得到词向量
        """
        # 1.
        # 拿到文本和标签
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # 2.
        # 处理文本，得到词向量
        tensors: dict = tokenizer(texts,truncation=True,
                                    padding=True,max_length=padding_size,
                                    return_tensors='pt')
        # nonlocal local_func_calls
        # local_func_calls += 1
        # print(f"局部函数调用次数：{local_func_calls}")
        return tensors, torch.tensor(labels,dtype=torch.long)

    dataset = NewsDataset(*data_loader(data_dir))
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,collate_fn=collate_fn)
    # print(f"初始的局部函数调用次数:{local_func_calls}")
    return dataloader


if __name__=='__main__':
    from config import Config
    from tqdm import tqdm
    from transformers import BertTokenizer


    config = Config()

    tokenizer = BertTokenizer.from_pretrained(config.bert_path)


    train_dataloader = build_dataloader(config.train_path,
                                        batch_size=config.batch_size,
                                        padding_size= config.pad_size,
                                        tokenizer=tokenizer)

    for data in tqdm(train_dataloader,desc="循环中"):
        print(data)
        pass
