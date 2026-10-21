"""
随机森林基线模型配置文件
1. 训练，测试，验证数据集，标签数据路径
2. 训练，测试，验证数据集处理保存路径
3. 模型保存路径，预测结果保存路径，停用词数据
"""


class Config(object):
    def __init__(self):
        super().__init__()
        self.training_data_path = '../Random Forest/Datass/train.txt'
        self.testing_data_path = '../Random Forest/Datass/test.txt'
        self.dev_data_path = '../Random Forest/Datass/dev.txt'
        self.label_path = '../Random Forest/Datass/class.txt'

        self.orig_path_list = [self.training_data_path, self.testing_data_path, self.dev_data_path]

        ## 处理数据保存路径
        self.processing_training_data_path = "E:/python_PycharmProjects/Random Forest/processing/train.csv"
        self.processing_testing_data_path = "E:/python_PycharmProjects/Random Forest/processing/test.csv"
        self.processing_dev_data_path = "E:/python_PycharmProjects/Random Forest/processing/dev.csv"

        self.processing_data_list = [self.processing_training_data_path, self.processing_testing_data_path, self.processing_dev_data_path]


        # 模型保存路径
        self.model_path = "./models/"
        self.prediction_result_pah = "./results/"
        self.stop_words_path = "../Random Forest/Datass/stopwords.txt"


    def iterate_data_paths_and_processing(self):


        """
        遍历输入数据路径与预处理数据路径
        """
        for i in range((len(self.orig_path_list))):
            yield self.orig_path_list[i], self.processing_data_list[i]


if __name__ == "__main__":
    import pandas as pd
    config_obj = Config()
    pd.read_csv(config_obj.training_data_path, sep='\t')
    pd.read_csv(config_obj.testing_data_path, sep='\t')
    pd.read_csv(config_obj.dev_data_path,sep='\t')

    import os
    os.makedirs(config_obj.model_path, exist_ok=True)
    os.makedirs(config_obj.prediction_result_pah, exist_ok=True)

    for i in config_obj.iterate_data_paths_and_processing():
        print(i)