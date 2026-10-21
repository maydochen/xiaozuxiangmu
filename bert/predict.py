"""
目标：
封装bert模型的预测能力

步骤：
1. 实例化bert模型
2. 导入模型参数
3. 模型预测
"""

import torch
from config import Config
import time

from transformers import BertTokenizer
from model_io import load_for_predict

# 步骤：
# 1. 实例化bert模型，use_quantised=True 时加载量化模型
config = Config()
# 2. 导入模型参数
bert_model = load_for_predict(config)
bert_model.eval()          # 预测阶段关闭 dropout 等随机行为
tokenizer = BertTokenizer.from_pretrained(config.bert_path)

# 导入标签定义
with open(config.class_path,'r',encoding='utf-8') as f:
    ind2label = [line.strip() for line in f if line.strip()]
# 3. 模型预测
def predict(data):
    """
    预测函数封装，给定输入字典（包含text字段），返回
    {
    "pred_class", # 预测结果
    "duration # 耗时
    }
    """
    t1 = time.time()

    # 2. 预测
    title = data["text"]
    # 预测
    ## 字典
    input_tensors = tokenizer([title], padding=True, truncation=True,
              max_length=config.pad_size,
              return_tensors="pt")
    with torch.no_grad():
        bert_output = bert_model(input_tensors)
    ind = int(torch.argmax(bert_output[0]).item())


    return {
        "text": data['text'],
        "pred_class": ind2label[ind],
        "duration": time.time() - t1
    }

def load_one_sample_per_class(data_path, num_classes):
    """
    从数据集中每个类别取一条真实样本，返回 {类别id: 文本}
    用于快速检查每一类的预测效果
    """
    samples = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t')
            label = int(label)
            if label not in samples:
                samples[label] = text
            if len(samples) >= num_classes:
                break
    return samples


if __name__ == "__main__":
    # 每个类别生成一条测试文本（取自测试集的真实样本），便于统计准确率
    samples = load_one_sample_per_class(config.test_path, config.num_classes)

    text_list = [(i, samples[i]) for i in range(config.num_classes) if i in samples]

    correct = 0
    for true_ind, text in text_list:
        result = predict({"text": text})
        true_label = ind2label[true_ind]
        pred_label = result["pred_class"]
        hit = pred_label == true_label
        correct += int(hit)
        flag = "√" if hit else "×"
        print(f"     文本：{text}")
        print(f"[{flag}] 真实：{true_label:<8s} 预测：{pred_label:<8s} "
              f"耗时：{result['duration']:.3f}s")

    total = len(text_list)
    print(f"\n共 {total} 个类别（每类 1 条），预测正确 {correct} 条，"
          f"准确率：{correct / total:.2%}")
