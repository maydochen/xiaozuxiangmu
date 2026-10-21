"""
目标：快速走通bert模型的训练

步骤：
1. 数据读取
2. 数据向量化
3. 训练
"""

from config import Config
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# 实例化配置类
config = Config()

# 支持gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义模型
class BertClassifier(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)
    def forward(self,x):
        # 字典解包
        # output = self.bert(**x)
        _,pooler = self.bert(input_ids=x['input_ids'].to(device),attention_mask=x['attention_mask'].to(device),return_dict=False)
        # print(output.pooler_output)
        # 把池化层状态送入全连接层
        return self.fc(pooler)



model = BertClassifier().to(device)#
tokenizer = BertTokenizer.from_pretrained(config.bert_path)
# # 定义损失函数
criterion = torch.nn.CrossEntropyLoss()
# # 优化器
optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

def data_loader(data_path,batch_size):
    """
    遍历训练数据集中的batch

    思路：
    1. 读取数据
    2. 遍历事件和标签，保留batch
    """

    # 1. 读取数据
    data = pd.read_csv( data_path,sep='\t',names=['title','label'])
    titles = data['title'].tolist()
    labels = data['label'].tolist()
    # batch
    batch_num_max = 10000
    # 遍历
    for i in range(0,len(titles),batch_size):
        if i > batch_num_max:
            break
        yield titles[i:i+batch_size], labels[i:i+batch_size]
# iterator = data_loader(10)
# for i in iterator:
#   只会遍历一次
# for i in data_loader(10):
#   每次轮次都会创建一个新的可遍历对象

def data_processor(data,tokenizer):
    """
    把文本数据向量化
    """
    titles = data[0]
    labels = data[1]

    # 把文本送入词向量器
    ## padding: 是否填充, truncation: 是否截断, max_length: 句子的最大长度,return_tensors: 返回的张量类型
    ## input_ids 词向量, id_type词向量类型, mask掩码
    vector_dict = tokenizer(titles,padding=True, truncation=True, max_length=config.pad_size, return_tensors='pt')
    # print(vector_dict)
    # 把标签张量化
    labels = torch.tensor(labels,dtype=torch.long).to(device)

    # print(titles)
    # print(vector_dict)
    # print('-'*20)
    return vector_dict, labels

## 训练循环

for i in range(config.num_epochs):
    # 循环
    total_loss = 0
    batch_i = 0
    for batch in tqdm(data_loader(config.train_path,config.batch_size),desc=f"bert模型训练第{i+1}轮"):#
        # 拿到数据
        optimiser.zero_grad()
        # # 数据处理，拿到tensor
        input_vector, label = data_processor(batch,tokenizer)#
        # # 计算向前传播
        y = model(input_vector)#
        # print(y.shape)
        # # 计算损失
        loss = criterion(y,label)
        # print(loss.item())
        total_loss += loss.item()
        # # 反向传播
        loss.backward()

        # 更新参数
        optimiser.step()

        ## 打印平均损失
        batch_i += 1
        if batch_i % 20 == 1:
            print(f"当前平均损失：{total_loss/batch_i}")
    ## 查看准确率
    label_list = []
    pred_list = []
    for batch in data_loader(config.test_path,config.batch_size):#
        # 拿到数据
        with torch.no_grad():
            # # 数据处理，拿到tensor
            input_vector, label = data_processor(batch,tokenizer)#
            # # 计算向前传播
            y = model(input_vector)#
            # print(y.shape)  batch, 10
            #
            # # 计算模型实际输出
            pred_list.extend(torch.argmax(y, dim=1).tolist())
            label_list.extend(label.tolist())
    # print(pred_list)
    print(f"测试集准确率：{accuracy_score(label_list,pred_list)}")




