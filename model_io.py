"""
模型权重的保存与加载（普通模型 + 动态量化模型）

train.py / quantisation.py / predict.py 统一从这里取，
避免各写一份导致量化权重加载方式不一致而报错。
"""

import os
import torch

import bert_classifer_model
from bert_classifer_model import BertClassifier


def _ensure_dir(save_path):
    """保证保存目录存在"""
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def _file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def _check_exists(save_path):
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"未找到模型文件：{os.path.abspath(save_path)}，"
            f"请先运行 train.py / quantisation.py 生成模型")


def build_model(config, device=None):
    """
    构造 BertClassifier。

    同时同步 bert_classifer_model 模块里的全局 device（forward 中要用），
    不传 device 时自动选 gpu
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)
    bert_classifer_model.device = device
    model = BertClassifier(config.bert_path, config.hidden_size, config.num_classes)
    model.to(device)
    return model


def save_model(model, save_path):
    """
    保存普通（float32）模型权重。

    统一转成 cpu 张量，这样训练用 gpu、预测在 cpu 上也能直接加载
    """
    _ensure_dir(save_path)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, save_path)
    size_mb = _file_size_mb(save_path)
    print(f"模型已保存到：{os.path.abspath(save_path)}（{size_mb:.2f}Mb）")
    return size_mb


def save_quantised_model(quantised_model, save_path):
    """
    保存动态量化后的 int8 权重
    """
    _ensure_dir(save_path)
    torch.save(quantised_model.state_dict(), save_path)
    size_mb = _file_size_mb(save_path)
    print(f"量化模型已保存到：{os.path.abspath(save_path)}（{size_mb:.2f}Mb）")
    return size_mb


def load_model(config, save_path, device=None):
    """
    加载普通模型权重
    """
    _check_exists(save_path)
    model = build_model(config, device)
    state_dict = torch.load(save_path, map_location=next(model.parameters()).device)
    model.load_state_dict(state_dict)
    return model


def load_quantised_model(config, save_path):
    """
    加载量化模型。

    注意：必须先按同样的配置做一次动态量化，再载入 int8 权重，
    直接把 int8 权重塞进普通的 float 线性层会报错
    """
    _check_exists(save_path)
    model = build_model(config, "cpu")
    model.eval()
    quantised_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantised_model.load_state_dict(torch.load(save_path, map_location="cpu"))
    quantised_model.eval()
    return quantised_model


def load_for_predict(config):
    """
    按配置选择加载普通模型或量化模型（供 predict.py 使用）
    """
    use_quantised = getattr(config, "use_quantised", False)
    if use_quantised:
        save_path = config.quant_model_save_path
        model = load_quantised_model(config, save_path)
    else:
        save_path = config.model_save_path
        model = load_model(config, save_path)
    print(f"已加载{ '量化' if use_quantised else '普通'}模型：{os.path.abspath(save_path)}")
    return model
