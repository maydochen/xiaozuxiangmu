"""
目标：把文本转换为fasttext接受的格式

步骤：
1. 读取数据
2. 转换格式
    2.1 标签保留数字 id
    2.2 分词（char 字符级 / jieba 词级）
3. 导出

运行方式：
    在本文件所在目录（02-fasttext）直接运行，PyCharm 里点运行按钮即可。
    运行后会在 processing/ 下生成 train-jieba.txt、train-char.txt 等 6 个文件。
"""

# ------------------------------------------------------------------
# 【第 1 段】依赖导入
# 作用：引入文件操作、数据处理、分词和项目配置
#   os       - 拼接路径、创建目录
#   pandas   - 读取制表符分隔的原始数据
#   jieba    - 中文分词
#   Config   - 项目路径配置
# ------------------------------------------------------------------
import os

import pandas as pd
import jieba

from Config import Config

# 把工作目录切到本文件所在目录，保证配置里的相对路径（../data/、processing/）
# 在 PyCharm 里直接点运行按钮时也能找对
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------
# 【第 2 段】全局配置
# 作用：实例化配置类，后面所有路径都从 config 里取
# ------------------------------------------------------------------
config = Config()


# ------------------------------------------------------------------
# 【第 3 段】分词函数
# 作用：把一句话变成 fasttext 需要的「空格分隔的 token 串」
#   char  - 字符级切分，每个汉字单独作为一个 token
#   jieba - 词级切分，按中文词语切开
# 两种方式训练出的模型效果不同，所以两种都生成一份数据
# ------------------------------------------------------------------
def tokenize(text, splitting_method='char'):
    # 先把制表符和换行替换成空格，否则会破坏 fasttext 一行的结构
    text = str(text).replace('\t', ' ').replace('\n', ' ').strip()
    if splitting_method == 'char':
        # 字符级：去掉空格后逐字切开，再拼成 "我 要 投 诉" 这种形式
        return ' '.join(list(text.replace(' ', '')))
    elif splitting_method == 'jieba':
        return ' '.join(jieba.lcut(text))
    else:
        raise ValueError(f"不支持的分词方式: {splitting_method}，可选 'char' | 'jieba'")


# ------------------------------------------------------------------
# 【第 4 段】数据转换函数
# 作用：读一个原始数据文件，转换成 fasttext 格式后写到输出路径
#       转换后的每一行形如：__label__3 小区 楼下 烧烤 店 深夜 营业
#       其中 3 就是原始数据里的标签 id
# ------------------------------------------------------------------
def preprocessing(input_data, output_data, splitting_method='char'):
    """
    根据输入文件，输出路径，和分词方式，产生fasttext格式的数据
    splitting_method: 'char'|'jieba'
    """
    # ---- 1. 读取数据 ----
    # 原始数据没有表头，第一列是文本，第二列是标签 id（用制表符分隔）
    data = pd.read_csv(
        input_data,
        sep='\t',
        names=['title', 'label'],
        encoding='utf-8',
        dtype=str,                  # 全部按字符串读，避免标签被当成数字产生 0.0
        keep_default_na=False,      # 空字符串保持为空，不变成 NaN
    )
    titles = data['title'].tolist()
    labels = data['label'].tolist()

    # 数一下 class.txt 有多少行，用于校验标签 id 是否越界
    with open(config.label_path, 'r', encoding='utf-8') as f:
        label_count = len([line for line in f if line.strip()])

    # ---- 2. 转换格式 ----
    output = []
    skipped = 0
    for title, label in zip(titles, labels):
        # 空文本没有训练价值，直接跳过
        if not title.strip():
            skipped += 1
            continue
        # 标签列如果不是整数，说明这行数据有问题，跳过
        try:
            label_id = int(label)
        except (TypeError, ValueError):
            skipped += 1
            continue
        # 标签 id 超出类别总数，同样跳过（说明数据和 class.txt 对不上）
        if not 0 <= label_id < label_count:
            skipped += 1
            continue
        output.append(f"__label__{label_id} {tokenize(title, splitting_method)}\n")

    # ---- 3. 导出 ----
    os.makedirs(os.path.dirname(os.path.abspath(output_data)), exist_ok=True)
    with open(output_data, 'w', encoding='utf-8') as f:
        f.writelines(output)

    print(f"[{splitting_method}] {os.path.basename(input_data)} -> {output_data}")
    print(f"    有效样本 {len(output)} 条，跳过 {skipped} 条")


# ------------------------------------------------------------------
# 【第 5 段】程序入口
# 作用：对训练集、测试集、验证集各处理一遍，
#       每种数据都生成 char 和 jieba 两个版本，共 6 个文件
# ------------------------------------------------------------------
if __name__ == '__main__':
    splitting_methods = ['jieba', 'char']

    # 原始数据路径 -> 输出路径的对应关系，集中写在这里便于维护
    data_dict = [
        {"输入路径": config.training_data_path,
         "输出路径": config.processing_training_data_path,
         "标签": "训练数据"},
        {"输入路径": config.testing_data_path,
         "输出路径": config.processing_testing_data_path,
         "标签": "测试数据"},
        {"输入路径": config.dev_data_path,
         "输出路径": config.processing_dev_data_path,
         "标签": "验证数据"},
    ]

    for splitting_method in splitting_methods:
        for data in data_dict:
            print(f"正在处理{data['标签']}（{splitting_method}）")
            # 把 processing/train.txt 变成 processing/train-jieba.txt 这种形式
            output_path = data['输出路径'].rsplit('.', 1)[0] + '-' + splitting_method + '.txt'
            preprocessing(data['输入路径'], output_path, splitting_method)
