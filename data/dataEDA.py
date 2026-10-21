# -*- coding: utf-8 -*-
"""
   标签数量分布   -> sns.countplot（判断 19 个类别均不均衡）
   句子长度分布   -> 新增长度列(map+lambda) + countplot 柱状图 + displot 密度图
   各类别长度散点 -> sns.stripplot（找异常超长样本）

数据：./train.txt 每行“事件文本\\t数字标签”；./class.txt 每行一个类别名（行号=标签）
输出：./eda_figures/*.png（每张图先 savefig 保存，再 plt.show() 在 PyCharm 中弹窗显示）
"""
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt     # 不设置 Agg：PyCharm 里 plt.show() 才能弹窗显示，同时 savefig 照常保存

# ---------------- 全局配置（顺序：先选风格，再覆盖中文字体） ----------------
plt.style.use("fivethirtyeight")                       # 讲义同款风格
plt.rcParams["font.sans-serif"] = ["SimHei"]          # 中文不乱码（风格设置后再覆盖）
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid", {"font.sans-serif": ["SimHei"]})  # seaborn 图里的中文

TRAIN_PATH = "./train.txt"
CLASS_PATH = "./class.txt"
FIG_DIR = "./eda_figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------- 数据加载（对应讲义 train.tsv，本项目是 \t 分隔的 txt） ----------------
# sep="\t" 制表符分隔；names 手动指定两列名：sentence 文本、label 数字标签
train_data = pd.read_csv(TRAIN_PATH, sep="\t", names=["sentence", "label"])
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = f.read().split("\n")                # 行号就是标签编号
label2name = {i: n for i, n in enumerate(class_names)}
train_data["label_name"] = train_data["label"].map(label2name)  # 便于图上显示中文名
print(train_data.head())
print("训练集数据量：", train_data.shape[0])

# ================= 5.2 标签数量分布（sns.countplot） =================
plt.figure(figsize=(11, 5))
sns.countplot(x="label_name", data=train_data,
              order=class_names, color="#4C72B0")    # order 固定按 class.txt 顺序
plt.title("各标签样本数量分布")
plt.xlabel("")
plt.ylabel("数量")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_标签数量分布.png"), dpi=150)
plt.show()

# ================= 5.3 句子长度分布（map+lambda 新增长度列） =================
# map(lambda x: len(x), 句子列)：给每条事件文本算字数
train_data["sentence_length"] = list(
    map(lambda x: len(str(x)), train_data["sentence"]))
print("句子长度 均值%.1f 中位%.0f 最短%d 最长%d"
      % (train_data["sentence_length"].mean(),
         train_data["sentence_length"].median(),
         train_data["sentence_length"].min(),
         train_data["sentence_length"].max()))

# 1) 柱状图：每个具体长度出现多少次（长度值多，x 刻度太密，按讲义清空）
plt.figure(figsize=(10, 4.5))
sns.countplot(x="sentence_length", data=train_data, color="#55A868")
plt.title("句子长度分布（柱状图）")
plt.xlabel("句子长度（字数）")
plt.ylabel("数量")
plt.xticks([])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_句子长度分布_柱状图.png"), dpi=150)
plt.show()

# 2) 密度图：看长度整体分布形状（seaborn 0.13 用 displot，旧版 distplot 已移除）
sns.displot(x="sentence_length", data=train_data, bins=30, kde=True,
            height=4.5, aspect=1.8, color="#55A868")
plt.title("句子长度分布（密度图）")
plt.xlabel("句子长度（字数）")
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_句子长度分布_密度图.png"), dpi=150)
plt.show()

# ================= 5.4 各类别长度散点（sns.stripplot，找异常超长样本） =================
# 19 类各抽最多 300 条画散点（3 万个点会挤成黑块且很慢，抽样后读图效果一致）
plt.figure(figsize=(11, 5))
sns.stripplot(y="sentence_length", x="label_name", data=train_data,
              order=class_names, size=2, alpha=0.5, color="#C44E52")
plt.title("各类别句子长度散点分布")
plt.xlabel("")
plt.ylabel("句子长度（字数）")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_各类别句子长度散点.png"), dpi=150)
plt.show()

print("-" * 55)
print("全部图表已保存到：", os.path.abspath(FIG_DIR))
