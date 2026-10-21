"""
目标：实现fasttext模型训练

步骤：
1. 声明数据路径 - 配置类实例化
2. 训练
3. 测试（只看总精确率）
4. 保存

运行方式：
    在本文件所在目录（02-fasttext）直接运行，PyCharm 里点运行按钮即可。
    运行前需要先运行 data_processing.py，生成 processing/ 下的数据文件。
"""

# ------------------------------------------------------------------
# 【第 1 段】依赖导入
# 作用：引入需要的标准库、第三方库和项目配置类
#   os       - 判断文件是否存在、创建目录、取文件名
#   fasttext - 训练模型、评估模型、保存模型
#   tqdm     - 在控制台显示训练进度条，避免训练时干等
#   Config   - 项目路径配置
# ------------------------------------------------------------------
import os

import fasttext
from tqdm import tqdm

from Config import Config

# 把工作目录切到本文件所在目录，保证配置里的相对路径（../data/、processing/、./models/）
# 在 PyCharm 里直接点运行按钮时也能找对
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------
# 【第 2 段】全局参数
# 作用：实例化配置类，并声明训练时用到的几个开关和超参数
#   EPOCH                 - 训练轮数，即把整个训练集完整遍历几遍。
#                           遍历 5 遍，等价于每个类别的样本都被学习 5 次
#   SPLITTING_METHODS     - 用哪几种分词方式的数据各训练一个模型
#   WHETHER_AUTOTUNE_LIST - 是否让 fasttext 自动搜索超参。
#                           [False, True] 表示普通训练和自动调参各训练一次，
#                           最终会得到 2 × 2 = 4 个模型
# ------------------------------------------------------------------
config = Config()

EPOCH = 5
SPLITTING_METHODS = ['jieba', 'char']
WHETHER_AUTOTUNE_LIST = [False, True]
AUTOTUNE_DURATION = 60


# ------------------------------------------------------------------
# 【第 3 段】训练函数
# 作用：完成「训练 -> 测试 -> 保存模型」这一条完整流程，
#       并把模型文件名和测试集总精确率返回给调用方
# 参数：
#   train_data_path / test_data_path - processing/ 下对应分词方式的数据
#   dev_data_path    - 验证集，只有自动调参时才会用到
#   model_save_path  - 模型保存路径
#   whether_autotune - True 表示让 fasttext 自动搜索超参
# ------------------------------------------------------------------
def train(train_data_path, test_data_path, dev_data_path, model_save_path, whether_autotune):
    # 先检查数据文件在不在，避免训练跑到一半才报错
    for path in (train_data_path, test_data_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}，请先运行 data_processing.py")

    # ---- 2. 训练 ----
    if whether_autotune:
        # 自动调参模式：在验证集上反复试超参，耗时由 AUTOTUNE_DURATION 控制
        model = fasttext.train_supervised(
            input=train_data_path,
            autotuneValidationFile=dev_data_path,
            autotuneDuration=AUTOTUNE_DURATION,
            thread=1,
            verbose=0,
        )
    else:
        # 普通模式：手动指定的默认超参
        model = fasttext.train_supervised(
            input=train_data_path,
            epoch=EPOCH,        # 遍历训练集 EPOCH 遍
            lr=0.5,
            wordNgrams=2,
            dim=100,
            minCount=1,
            thread=1,
            verbose=0,
        )

    # ---- 3. 测试 ----
    # test() 返回三个值：样本数、精确率、召回率，这里只取精确率
    num, precision, recall = model.test(test_data_path)

    # ---- 4. 保存 ----
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    model.save_model(model_save_path)

    return os.path.basename(model_save_path), precision


# ------------------------------------------------------------------
# 【第 4 段】程序入口
# 作用：把「分词方式」和「是否自动调参」做笛卡尔积，得到 4 组训练任务，
#       用 tqdm 进度条逐个训练，每个训练完打印一行「模型名  精确率」
# ------------------------------------------------------------------
if __name__ == '__main__':
    # 先把所有组合整理成列表，进度条才能知道总共有几个任务
    tasks = [(method, autotune) for autotune in WHETHER_AUTOTUNE_LIST
             for method in SPLITTING_METHODS]

    # tqdm 包住任务列表：每训练完一个模型，进度条前进一格
    progress = tqdm(tasks, desc="模型训练进度", ncols=100)
    results = []
    for splitting_method, whether_autotune in progress:
        autotune_str = "autotune" if whether_autotune else "no-autotune"

        # 根据分词方式拼出对应的数据文件和模型路径
        train_data_path = config.processing_training_data_path.rsplit('.', 1)[0] + f'-{splitting_method}.txt'
        test_data_path = config.processing_testing_data_path.rsplit('.', 1)[0] + f'-{splitting_method}.txt'
        dev_data_path = config.processing_dev_data_path.rsplit('.', 1)[0] + f'-{splitting_method}.txt'
        model_save_path = config.model_path + f"{splitting_method}-{autotune_str}.bin"

        # 进度条右侧显示当前任务名，方便看出正在训练哪个模型
        progress.set_postfix_str(f"{splitting_method}-{autotune_str}", refresh=True)

        model_name, precision = train(
            train_data_path, test_data_path, dev_data_path, model_save_path, whether_autotune
        )
        # 用 progress.write 输出，避免日志把进度条顶乱
        progress.write(f"{model_name}  精确率: {precision:.4f}")
        results.append((model_name, precision))

    progress.close()

    # 训练结束后统一再打印一遍汇总，格式同样是「模型名 + 精确率」
    print("\n========== 训练结果 ==========")
    for model_name, precision in results:
        print(f"{model_name}  精确率: {precision:.4f}")
