"""
目标：快速走通bert模型的训练

步骤：
1. 数据读取
2. 数据向量化
3. 训练
"""

import os

from config import Config
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score
from utils import build_dataloader
from model_io import save_model, build_model



## 训练循环
def train(i,train_path,batch_size,pad_size,
          tokenizer,model,criterion,optimiser,device):
    """
    模型训练
    """
    # 循环
    total_loss = 0
    batch_i = 0

    train_data = build_dataloader(train_path, batch_size,
                                  pad_size, tokenizer)
    for text_tensor,label in tqdm(train_data,desc=f"bert模型训练第{i+1}轮"):#
        # 拿到数据
        optimiser.zero_grad()
        # # 计算向前传播
        y = model(text_tensor)#
        # print(y.shape)
        # # 计算损失
        loss = criterion(y,label.to(device))
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

@torch.no_grad()
def evaluate(dev_path,
             batch_size,pad_size,
             tokenizer,model,):
    ## 查看准确率
    model.eval()   # 评估阶段关闭 dropout
    dev_data = build_dataloader(dev_path, batch_size,
                                  pad_size, tokenizer)

    label_list = []
    pred_list = []
    for input_vector,label in dev_data:#
        # 拿到数据
        # # 计算向前传播
        y = model(input_vector)#
        # print(y.shape)  batch, 10
        #
        # # 计算模型实际输出
        pred_list.extend(torch.argmax(y, dim=1).tolist())
        label_list.extend(label.tolist())
    # print(pred_list)
    return accuracy_score(label_list,pred_list), f1_score(label_list,pred_list,
                                                          average='micro')

def main():
    # 实例化配置类
    config = Config()

    # 支持gpu训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, device)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    # # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # # 优化器
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    ##
    best_f1 = 0
    for i in range(config.num_epochs):
        ## 执行训练
        model.train()
        train(i, config.train_path, config.batch_size,
              config.pad_size, tokenizer, model,
              criterion, optimiser, device)
        ## 测试
        acc, f1 = evaluate(config.dev_path, config.batch_size,
                           config.pad_size, tokenizer, model,
                           )
        print(f"当前轮次准确率：{acc}， f1:{f1}")

        ## 效果变好才保存，保证 save_models 里是最佳模型
        if f1 > best_f1:
            best_f1 = f1
            save_model(model, config.model_save_path)

    ## 训练结束再保存一次，避免 dev 集效果一直下降时没有任何模型文件
    if not os.path.exists(config.model_save_path):
        save_model(model, config.model_save_path)

    ## 用保存下来的模型在测试集上验证一次
    acc, f1 = evaluate(config.test_path, config.batch_size,
                       config.pad_size, tokenizer, model,
                       )
    print(f"测试集准确率：{acc}， f1:{f1}")


if __name__ == "__main__":
    main()


