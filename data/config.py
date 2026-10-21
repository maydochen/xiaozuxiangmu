"""
目标：配置类 - 配置
1. 训练集数据路径
2. 测试机数据路径
3. 验证集数据路径
4. 标签路径

"""

class Config(object):
    def __init__(self):
        self.training_data_path = './train.txt'
        self.testing_data_path = './test.txt'
        self.dev_data_path = './dev.txt'
        self.class_path = './class.txt'



if __name__ == '__main__':
    config_obj = Config()
    assert config_obj.training_data_path == './train.txt'
    assert config_obj.testing_data_path == "./test.txt"

    print("测试成功")
