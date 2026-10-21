"""
目标：处理训练集，验证集，测试集，增加words字段，分词后用空格连接的句子

步骤：
1. 导入数据
2. 逐句做分词
3. 构建空格连接的句子，写入dataframe里面
4. 写入到输出文件
"""
import pandas as pd
import jieba
from config import Config
# 步骤：
# 1. 导入数据
config_obj = Config()

data_path = config_obj.training_data_path

def data_processing(input_dir,output_path):
    """
    给定输入文件路径和输出文件路径，把原始数据经过分词预处理后写入输出
    """
    data = pd.read_csv(input_dir, sep='\t', names=['title','label'])

    # 2. 逐句做分词
    # words_list = []
    # for title in data['title'].tolist():
    #     words_list.append(' '.join(jieba.lcut(title)))
    # data['words'] = words_list
    # def title_split(title):
    #     return " ".join(jieba.lcut(title))
    # 3. 构建空格连接的句子，写入dataframe里面
    data['words'] =data['title'].apply(lambda title: " ".join(jieba.lcut(title)[:30]))

    print(data)
    # 4. 写入到输出文件
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    # 数据与代码分离
    # 数据结构
    # data_paths = [
    #     [config_obj.training_data_path, config_obj.processing_training_data_path],
    #     [config_obj.testing_data_path, config_obj.processing_testing_data_path],
    #     [config_obj.dev_data_path, config_obj.processing_dev_data_path]
    # ]
    data_paths = [
        {"输入数据路径": config_obj.training_data_path,
         "输出数据路径": config_obj.processing_training_data_path},
        {"输入数据路径": config_obj.testing_data_path,
         "输出数据路径": config_obj.processing_testing_data_path},
        {"输入数据路径": config_obj.dev_data_path,
         "输出数据路径": config_obj.processing_dev_data_path},
    ]

    # df = pd.DataFrame()
    #
    # for i in range(len(df)):
    #     df.iloc[i,0]

    # 代码逻辑
    # for input_data_path,output_path in data_paths:
    #     data_processing(input_data_path,output_path)
    for data in data_paths:
        data_processing(data["输入数据路径"], data["输出数据路径"])