"""
目标：导入保存的模型，做预测

步骤：
1. 导入模型
2. 预测

运行方式：
    在本文件所在目录（02-fasttext）直接运行，PyCharm 里点运行按钮即可。
    运行前需要先运行 train.py 训练出模型文件，否则会提示找不到模型。
"""

# ------------------------------------------------------------------
# 【第 1 段】依赖导入
# 作用：引入文件路径处理、计时、模型加载、分词和项目配置
#   os       - 切换工作目录、取文件名
#   time     - 统计单条预测耗时
#   fasttext - 加载训练好的模型并预测
#   jieba    - 中文分词，必须与训练时用同一种方式
#   Config   - 项目路径配置
# ------------------------------------------------------------------
import os
import time

import fasttext
import jieba

from Config import Config

# 把工作目录切到本文件所在目录，保证配置里的相对路径（../data/、./models/）
# 在 PyCharm 里直接点运行按钮时也能找对
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------
# 【第 2 段】全局配置
# 作用：实例化配置类，并指定本次要使用哪个模型文件
#   MODEL_NAME 换一下就能切换模型：
#       jieba-no-autotune.bin   jieba 分词 + 普通训练
#       char-no-autotune.bin    char 分词  + 普通训练
#       jieba-autotune.bin      jieba 分词 + 自动调参（文件可能很大）
#       char-autotune.bin       char 分词  + 自动调参（文件可能很大）
# ------------------------------------------------------------------
config = Config()

MODEL_NAME = "char-autotune.bin"


# ------------------------------------------------------------------
# 【第 3 段】读取标签表
# 作用：把 class.txt 读成列表，下标就是标签 id
#   例如 ID2LABEL[17] 是「交通出行」
# ------------------------------------------------------------------
with open(config.label_path, 'r', encoding='utf-8') as f:
    ID2LABEL = [line.strip() for line in f if line.strip()]


# ------------------------------------------------------------------
# 【第 4 段】加载模型
# 作用：把训练好的 .bin 模型读进内存，并确定配套的分词方式
#   模型文件名以 char 开头，说明是字符级模型，预测时也要按字符切分
# ------------------------------------------------------------------
MODEL_PATH = config.model_path + MODEL_NAME

try:
    model = fasttext.load_model(MODEL_PATH)
except Exception:
    raise FileNotFoundError(f"模型文件不存在或无法读取: {MODEL_PATH}\n请先运行 train.py 完成训练")

SPLITTING_METHOD = 'char' if MODEL_NAME.startswith('char') else 'jieba'


# ------------------------------------------------------------------
# 【第 5 段】预测函数
# 作用：接收一条文本，返回中文类别、标签 id 和耗时
#   输入：{"text": "..."}
#   输出：{"text", "pred_class", "pred_id", "duration"}
# ------------------------------------------------------------------
def predict(data):
    t1 = time.time()

    # ---- 2. 预测 ----
    # 分词方式必须和训练时一致，否则模型看到的 token 全是没见过的词，结果会乱
    title = data['text']
    if SPLITTING_METHOD == 'char':
        text = ' '.join(list(title))
    else:
        text = ' '.join(jieba.lcut(title))

    # model.predict 返回标签元组，形如 ('__label__17',)，取第 0 个元素
    labels = model.predict(text)[0]
    # 标签字符串是 '__label__17'，去掉前缀得到 '17'，再转成整数 id
    label_id = int(labels[0][len('__label__'):])

    return {
        'text': data['text'],
        'pred_class': ID2LABEL[label_id],
        'pred_id': label_id,
        'duration': time.time() - t1,
    }


# ------------------------------------------------------------------
# 【第 6 段】程序入口
# 作用：准备一批待预测文本，逐条预测并打印结果，
#       最后统计总耗时和单条平均耗时
# ------------------------------------------------------------------
if __name__ == '__main__':
    # 待预测文本，取自测试集中的真实工单标题，可自行增删
    text_list = [
        "协调彻底解决管道堵塞问题",
        "核实外出务工补贴下发问题",
        "反映出租车司机危险驾驶相关问题",
        "核实危房重建补助政策",
        "协调解决职工养老保险缴纳问题",
        "求助解决噪音扰民问题（保障中高考）",
        "投诉某药店保健品夸大功效宣传",
        "咨询孩子转学需要办理哪些手续",
        "小区物业费上涨但服务质量下降",
        "咨询户口转移的办理流程",
    ]

    t1 = time.time()
    for text in text_list:
        result = predict({"text": text})
        print(f"{result['text']}  ->  {result['pred_class']}（标签 id {result['pred_id']}）")

    # 除以条数得到单条平均耗时，方便评估线上部署的响应速度
    duration = time.time() - t1
    print(f"\n共 {len(text_list)} 条，总耗时 {duration:.4f}s，平均 {duration / len(text_list) * 1000:.4f} ms/条")
