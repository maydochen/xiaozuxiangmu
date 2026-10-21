"""
目标：
测试量化前与量化后指标的差异：速度，占用空间，准确率

步骤：
1. 导入数据
2. 导入模型
3. 量化
4. 分别测试准确率
5. 保存量化模型（并在测试集上验证重新加载后的效果）
"""

import os
from config import Config
import torch
import bert_classifer_model
from utils import build_dataloader
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer
import time
from model_io import save_quantised_model, load_quantised_model, load_model

config = Config()
bert_classifer_model.device = torch.device("cpu")

# 步骤：
# 1. 导入数据

@torch.no_grad()

def evaluate(dev_path,
             batch_size,pad_size,
             tokenizer,model,):
    ## 查看准确率
    model.eval()
    dev_data = build_dataloader(dev_path, batch_size,
                                  pad_size, tokenizer)

    label_list = []
    pred_list = []
    t1 =  time.time()
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
    # print(list(model.parameters())[0].element_size())
    return accuracy_score(label_list,pred_list), (time.time()-t1)/len(dev_data), sum([p.numel()*p.element_size() for p in model.parameters()])/(1024**2)

# 2. 导入模型（量化只在 cpu 上做，这里直接按 cpu 载入）
model = load_model(config, config.model_save_path, "cpu")
model.eval()                      # 关键：关闭 dropout/bn 训练行为

tokenizer = BertTokenizer.from_pretrained(config.bert_path)
# 3. 量化
### 参1：model；参2：量化的部分 {nn.Linear};参3：精度
quantised_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype = torch.qint8
)
quantised_model.eval()

# 4. 分别测试准确率
orig_acc,orig_time, orig_size = evaluate(config.test_path, config.batch_size,config.pad_size,tokenizer,model)

q_acc,q_time, q_size = evaluate(config.test_path, config.batch_size,config.pad_size,tokenizer,quantised_model)

print(f"原模型准确率：{orig_acc}, 单样本耗时{orig_time},运行体积{orig_size}Mb")
print(f"量化模型准确率：{q_acc}, 单样本耗时{q_time}，运行体积{q_size}Mb")


print(f"速度提升{ (1/q_time/(1/orig_time) -1)*100:.2f}% ")
print(f"运行体积为原本模型的{q_size/orig_size * 100:.2f}%")
print(f"准确率变为原本模型的{q_acc/orig_acc*100:.2f}%")

# 5. 保存量化模型
q_file_size = save_quantised_model(quantised_model, config.quant_model_save_path)
orig_file_size = os.path.getsize(config.model_save_path) / (1024 ** 2) \
    if os.path.exists(config.model_save_path) else 0
if orig_file_size:
    print(f"原模型文件体积：{orig_file_size:.2f}Mb，"
          f"量化后文件为原本的{q_file_size / orig_file_size * 100:.2f}%")

# 6. 重新加载保存的量化模型，验证权重可用
#    这一轮会再跑一遍测试集，耗时与上面一次 evaluate 相当，不需要时设成 False
VERIFY_RELOAD = False
if VERIFY_RELOAD:
    reload_model = load_quantised_model(config, config.quant_model_save_path)
    r_acc, _, _ = evaluate(config.test_path, config.batch_size,
                           config.pad_size, tokenizer, reload_model)
    print(f"重新加载的量化模型准确率：{r_acc}"
          f"（与量化时{'一致' if abs(r_acc - q_acc) < 1e-6 else '不一致'}）")

