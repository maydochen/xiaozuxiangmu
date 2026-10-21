class Config(object):
    def __init__(self):
        super().__init__()
        self.training_data_path = '../data/train.txt'
        self.testing_data_path = '../data/test.txt'
        self.dev_data_path = '../data/dev.txt'
        self.label_path = '../data/class.txt'

        ## 处理数据保存路径
        self.processing_training_data_path = "processing/train.txt"
        self.processing_testing_data_path = "processing/test.txt"
        self.processing_dev_data_path = "processing/dev.txt"

        # 模型保存路径
        self.model_path = "./models/"
