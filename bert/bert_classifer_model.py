import torch
from transformers import BertModel

# 支持gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义模型
class BertClassifier(torch.nn.Module):
    def __init__(self,bert_path,hidden_size,num_classes):
        """
        bert分类模型，需要给定bert预训练模型路径，隐藏层维度，和预测的类别数量
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    def forward(self,x):
        # 字典解包
        # output = self.bert(**x)
        _,pooler = self.bert(input_ids=x['input_ids'].to(device),attention_mask=x['attention_mask'].to(device),return_dict=False)
        # print(output.pooler_output)
        # 把池化层状态送入全连接层
        return self.fc(pooler)
