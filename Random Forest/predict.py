# -*- coding: utf-8 -*-
"""
工单文本分类 · 推理脚本

功能：
    1. 加载已保存的 TF-IDF 向量器与随机森林模型
    2. 提供单条预测接口 predict()
    3. 直接运行时逐条预测，并在末尾输出平均耗时

返回字段说明：
    text        输入原文
    pred_class  预测类别（中文，取自 class.txt）
    duration    单条推理耗时，单位秒
"""
from __future__ import annotations

import os
import pickle
import time

import jieba

from config import Config

# 与训练阶段保持一致：每句最多保留 30 个词
MAX_WORDS = 30

config = Config()

with open(os.path.join(config.model_path, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(os.path.join(config.model_path, "rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

# 标签表：下标即模型输出的类别编号，过滤空行避免错位
with open(config.label_path, "r", encoding="utf-8") as f:
    ind2label = [line.strip() for line in f if line.strip()]


def _tokenize(text: str) -> str:
    """分词并用空格连接，截断到 MAX_WORDS 个词"""
    return " ".join(jieba.lcut(text)[:MAX_WORDS])


def predict(data: dict) -> dict:
    """
    单条文本预测

    参数：
        data: 含 text 字段的字典，例如 {"text": "某小区垃圾堆积无人清理"}

    返回：
        {"text": 原文, "pred_class": 预测类别, "duration": 耗时（秒）}
    """
    if not isinstance(data, dict) or "text" not in data:
        raise ValueError("输入格式错误：需要包含 text 字段的字典")

    text = str(data["text"]).strip()
    if not text:
        raise ValueError("输入内容为空：text 字段不能是空字符串")

    start = time.perf_counter()
    vector = tfidf_vectorizer.transform([_tokenize(text)])
    label_id = int(model.predict(vector)[0])
    elapsed = time.perf_counter() - start

    return {
        "text": text,
        "pred_class": ind2label[label_id],
        "duration": round(elapsed, 6),
    }


def _print_result(index: int, item: dict) -> None:
    """按统一格式打印一条预测结果"""
    print(f"【第 {index:02d} 条】{item['text']}")
    print(f"    预测类别：{item['pred_class']}")
    print(f"    推理耗时：{item['duration'] * 1000:.2f} 毫秒")
    print()


if __name__ == "__main__":
    samples = [
        "某公园居民王先生提出，可将社区文化活动增加纳入志愿服务项目，降低长期运行成本",
        "解决排水管道破裂问题",
        "协调取消购买书籍填写身份信息问题",
        "某社区一名业主建议，针对某住宅小区规划公示内容不清晰组织多方协商并形成书面方案",
        "某主城区一名学生家长投诉，某培训机构无证开展学科培训，希望明确整改措施和完成时限",
        "2025年10月16日上午，某农产品收购点附近有农户反映，农产品收购商拖欠货款，希望提供法律或调解帮助",
        "某小区一处路边沟渠内漂浮枯枝杂草，影响排水和周边景观，希望安排清理",
        "2026年1月23日，某小区业主反映，物业以催缴装修保证金为由限制门禁或电梯使用，影响正常生活",
        "某街道一位居民反映，邻居一名重病患者近期因住房受灾陷入困境，日常吃饭和就医都遇到困难",
        "某农贸市场周边有5户居民反映，垃圾收集点污水外溢，周边生活受到影响",
        "协调村委会恢复河道",
        "2026年4月28日上午，某片区居民李女士反映，部分车辆长期违规停放并占用消防通道，希望加强执法",
        "某农贸市场周边居民李女士反映，社区卫生服务站医疗废物混投，已持续一年多，希望协调专业机构处置",
        "2026年2月27日下午，某社区有商户咨询，收到行政处罚告知后如何陈述申辩或申请听证)",
    ]

    print("=" * 72)
    print("工单文本分类 · 单条预测")
    print("=" * 72)

    total_seconds = 0.0
    for i, text in enumerate(samples, 1):
        result = predict({"text": text})
        _print_result(i, result)
        total_seconds += result["duration"]

    average_ms = total_seconds / len(samples) * 1000
    print(f"共 {len(samples)} 条，总耗时 {total_seconds * 1000:.2f} 毫秒，"
          f"平均每条 {average_ms:.2f} 毫秒")
