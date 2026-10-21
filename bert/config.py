class Config(object):
    def __init__(self):
        self.train_path = "../data/train.txt"
        self.dev_path = "../data/dev.txt"
        self.test_path = "../data/test.txt"
        self.class_path = "../data/class.txt"
        self.bert_path = "../bert-base-chinese"
        self.model_save_path = "../save_models/bert_classifier.pt"
        self.quant_model_save_path = "../save_models/bert_classifier_quantised.pt"
        # 预测时是否使用量化模型
        self.use_quantised = True

        self.hidden_size = 768
        self.num_classes = 19
        self.num_epochs = 5
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        # self.learning_rate = 1
