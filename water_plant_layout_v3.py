# -*- coding: utf-8 -*-
"""
供水厂设施布局优化算法 V3.0
基于剩余矩形排样算法与遗传算法结合 + 功能分区联合布置 + GPU加速

优化目标: 最小化供水厂总占地面积
约束条件:
1. 设施不重叠
2. 工艺流程相邻性约束
3. 差异化安全间距要求
4. 功能分区联合布置
"""

import sys
import io
# 设置输出编码为 UTF-8（避免 Windows console 编码问题）
if getattr(sys.stdout, "encoding", None) != 'utf-8' and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import random
import copy
import argparse
import csv
import json
from typing import List, Tuple, Dict
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from threading import Lock
warnings.filterwarnings('ignore')

# ==================== GPU 加速配置 ====================
GPU_AVAILABLE = False
GPU_NAME = "N/A"
cp = None

# 添加 NVIDIA CUDA DLL 路径 (Windows) - 必须在导入cupy之前
import os
import sys

# 尝试多个可能的CUDA路径
cuda_dll_paths = []

# 1. pip安装的nvidia包路径
for python_ver in ['Python313', 'Python312', 'Python311', 'Python310']:
    nvidia_path = os.path.join(os.environ.get('APPDATA', ''), 'Python', python_ver, 'site-packages', 'nvidia')
    if os.path.exists(nvidia_path):
        for subdir in os.listdir(nvidia_path):
            bin_path = os.path.join(nvidia_path, subdir, 'bin')
            if os.path.exists(bin_path):
                cuda_dll_paths.append(bin_path)

# 2. CUDA Toolkit安装路径
cuda_toolkit = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0')
if os.path.exists(os.path.join(cuda_toolkit, 'bin')):
    cuda_dll_paths.append(os.path.join(cuda_toolkit, 'bin'))

# 添加所有找到的路径到 PATH 和 add_dll_directory
for path in cuda_dll_paths:
    # 添加到 PATH 环境变量 (某些DLL依赖需要这个)
    if path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
    # 同时使用 add_dll_directory
    try:
        os.add_dll_directory(path)
    except Exception:
        pass

# 加载 CuPy (GPU加速)
try:
    import cupy as cp
    # 测试 GPU 是否真正可用
    test_arr = cp.array([1.0, 2.0, 3.0])
    _ = test_arr + test_arr  # 触发 CUDA 编译
    GPU_AVAILABLE = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props['name'].decode()
    GPU_MEM = props['totalGlobalMem'] / 1024**3
    print(f"✓ GPU 加速已启用: {GPU_NAME} ({GPU_MEM:.1f}GB)")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"  GPU 不可用，使用CPU模式")
    cp = None

# Numba JIT 加速 (CPU并行)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print(f"[OK] CPU并行加速已启用: {multiprocessing.cpu_count()} 核心")
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ==================== GPU 加速函数 ====================
# GPU使用阈值 - 只有数据量足够大时才用GPU，避免传输开销
GPU_THRESHOLD = 50  # 候选位置数量阈值
GPU_PLACED_THRESHOLD = 20  # 已放置设施数量阈值

def check_conflicts_gpu(candidates_x, candidates_y, new_w, new_h,
                        placed_x, placed_y, placed_w, placed_h, gaps):
    """GPU 并行检查所有候选位置的冲突 - 只在大批量时使用"""
    # 只有当候选位置多且已放置设施多时才用GPU
    if not GPU_AVAILABLE or cp is None:
        return None
    if len(candidates_x) < GPU_THRESHOLD or len(placed_x) < GPU_PLACED_THRESHOLD:
        return None  # 数据量小时CPU更快

    try:
        # 转移数据到 GPU
        cx = cp.asarray(candidates_x, dtype=cp.float32)
        cy = cp.asarray(candidates_y, dtype=cp.float32)
        px = cp.asarray(placed_x, dtype=cp.float32)
        py = cp.asarray(placed_y, dtype=cp.float32)
        pw = cp.asarray(placed_w, dtype=cp.float32)
        ph = cp.asarray(placed_h, dtype=cp.float32)
        g = cp.asarray(gaps, dtype=cp.float32)

        # 广播计算所有候选位置与所有已放置设施的碰撞
        # Shape: (n_cand, n_placed)
        cx_2d = cx[:, cp.newaxis]  # (n_cand, 1)
        cy_2d = cy[:, cp.newaxis]

        # 碰撞检测: 不重叠条件
        no_overlap = (
            (cx_2d >= px + pw + g) |  # 候选在右边
            (cx_2d + new_w + g <= px) |  # 候选在左边
            (cy_2d >= py + ph + g) |  # 候选在上边
            (cy_2d + new_h + g <= py)  # 候选在下边
        )

        # 如果所有已放置设施都不重叠，则该位置有效
        valid = cp.all(no_overlap, axis=1)

        return cp.asnumpy(valid)
    except Exception as e:
        return None  # GPU出错时回退到CPU

def evaluate_positions_gpu(candidates_x, candidates_y, new_w, new_h,
                           cur_max_x, cur_max_y, valid_mask):
    """GPU 并行评估所有有效位置的得分"""
    if not GPU_AVAILABLE or cp is None:
        return None

    try:
        cx = cp.asarray(candidates_x, dtype=cp.float32)
        cy = cp.asarray(candidates_y, dtype=cp.float32)
        mask = cp.asarray(valid_mask, dtype=cp.bool_)

        # 计算新边界
        new_max_x = cp.maximum(cur_max_x, cx + new_w)
        new_max_y = cp.maximum(cur_max_y, cy + new_h)

        # 面积得分
        scores = new_max_x * new_max_y

        # 长宽比惩罚
        aspect = cp.maximum(new_max_x, new_max_y) / cp.maximum(cp.minimum(new_max_x, new_max_y), 1.0)
        penalty = cp.where(aspect > 2.0, (aspect - 2.0) * 300.0, 0.0)
        scores = scores + penalty

        # 无效位置设为无穷大
        scores = cp.where(mask, scores, cp.inf)

        return cp.asnumpy(scores)
    except Exception as e:
        return None

# ==================== 设施类定义 ====================
class Facility:
    """设施类"""
    def __init__(self, name: str, name_cn: str, length: float, width: float,
                 category: str, safety_level: int = 1):
        self.name = name
        self.name_cn = name_cn
        self.length = length
        self.width = width
        self.area = length * width
        self.category = category
        self.safety_level = safety_level  # 安全等级 1-3
        self.x = 0
        self.y = 0
        self.rotated = False

    def get_dimensions(self):
        if self.rotated:
            return self.width, self.length
        return self.length, self.width

    def copy(self):
        f = Facility(self.name, self.name_cn, self.length, self.width,
                    self.category, self.safety_level)
        f.x, f.y, f.rotated = self.x, self.y, self.rotated
        return f

# ==================== 供水厂核心设施数据 ====================
# 格式: (代码, 中文名, 长度m, 宽度m, 功能分区, 安全等级1-3)
# 精细化10分区布局
FACILITIES_DATA = [
    # ===== 取水区 (intake) =====
    ("IPS", "取水泵站", 18, 12, "intake", 1),
    ("GC", "格栅间", 10, 6, "intake", 1),

    # ===== 预处理区 (pretreat) =====
    ("PS", "预沉池", 25, 15, "pretreat", 1),

    # ===== 常规处理区 (conventional) =====
    ("MB", "混凝池", 22, 14, "conventional", 1),
    ("FB", "絮凝池", 18, 12, "conventional", 1),
    ("SB", "沉淀池", 35, 18, "conventional", 1),
    ("FT", "滤池", 30, 16, "conventional", 1),

    # ===== 深度处理区 (advanced) =====
    ("OZG", "臭氧接触池", 20, 12, "advanced", 2),
    ("ACF", "活性炭滤池", 25, 15, "advanced", 1),
    ("OZR", "臭氧发生间", 12, 10, "advanced", 3),

    # ===== 消毒加药区 (chemical) =====
    ("CDR", "加药间", 14, 10, "chemical", 2),
    ("CLS", "加氯间", 10, 8, "chemical", 3),
    ("CST", "药剂仓库", 15, 10, "chemical", 2),

    # ===== 送配水区 (distribution) =====
    ("CWT", "清水池", 35, 25, "distribution", 1),
    ("BWT", "反冲洗水池", 15, 10, "distribution", 1),
    ("PH1", "一级泵房", 15, 10, "distribution", 1),
    ("PH2", "二级泵房", 20, 14, "distribution", 1),

    # ===== 动力区 (power) =====
    ("PDR", "配电室", 18, 12, "power", 2),
    ("TRF", "变压器室", 15, 10, "power", 3),

    # ===== 污泥处理区 (sludge) =====
    ("STK", "污泥浓缩池", 18, 12, "sludge", 1),
    ("SDR", "污泥脱水机房", 20, 12, "sludge", 1),
    ("SYD", "污泥堆场", 20, 15, "sludge", 1),

    # ===== 行政办公区 (admin) =====
    ("CR", "中控室", 18, 12, "admin", 1),
    ("OF", "综合办公楼", 25, 12, "admin", 1),
    ("GT", "门卫室", 6, 4, "admin", 1),
    ("LAB", "化验室", 12, 10, "admin", 1),

    # ===== 辅助设施区 (auxiliary) =====
    ("WH", "综合仓库", 18, 10, "auxiliary", 1),
    ("PK", "停车场", 20, 12, "auxiliary", 1),
    ("MW", "维修车间", 18, 12, "auxiliary", 1),

    # ===== 生活区 (living) =====
    ("DM1", "宿舍楼1", 15, 10, "living", 1),
    ("DM2", "宿舍楼2", 15, 10, "living", 1),
    ("DM3", "宿舍楼3", 15, 10, "living", 1),
    ("DM4", "宿舍楼4", 15, 10, "living", 1),
    ("SPF", "运动场", 30, 20, "living", 1),
]

# 精细分区列表（10个功能区）
CATEGORY_NAMES = {
    "intake": "取水区",
    "pretreat": "预处理区",
    "conventional": "常规处理区",
    "advanced": "深度处理区",
    "chemical": "消毒加药区",
    "distribution": "送配水区",
    "power": "动力区",
    "sludge": "污泥处理区",
    "admin": "行政办公区",
    "auxiliary": "辅助设施区",
    "living": "生活区",
}

# 分区颜色
CATEGORY_COLORS = {
    "intake": "#4ECDC4",       # 青色
    "pretreat": "#45B7D1",     # 浅蓝色
    "conventional": "#1E88E5", # 蓝色
    "advanced": "#9B59B6",     # 紫色
    "chemical": "#E74C3C",     # 红色
    "distribution": "#2ECC71", # 绿色
    "power": "#F39C12",        # 橙色
    "sludge": "#8B4513",       # 棕色
    "admin": "#95E1D3",        # 浅青绿
    "auxiliary": "#BDC3C7",    # 灰色
    "living": "#FFA07A",       # 浅橙色
}

# 安全间距矩阵 (米) - 紧凑设计
SAFETY_DISTANCES = {
    (1, 1): 2,   # 普通-普通
    (1, 2): 3,   # 普通-中危
    (1, 3): 5,   # 普通-高危
    (2, 2): 4,   # 中危-中危
    (2, 3): 6,   # 中危-高危
    (3, 3): 8,   # 高危-高危
}

# 工艺流程相邻性约束 (设施1, 设施2, 权重) - 权重越高约束越严格
ADJACENCY_REQUIRED = [
    # ===== 主工艺流程（必须严格相邻）=====
    ("IPS", "GC", 15),   # 取水泵站 → 格栅间
    ("GC", "PS", 15),    # 格栅间 → 预沉池
    ("PS", "MB", 15),    # 预沉池 → 混凝池
    ("MB", "FB", 15),    # 混凝池 → 絮凝池
    ("FB", "SB", 15),    # 絮凝池 → 沉淀池
    ("SB", "FT", 15),    # 沉淀池 → 滤池
    ("FT", "CWT", 15),   # 滤池 → 清水池
    ("CWT", "PH2", 12),  # 清水池 → 二级泵房

    # ===== 深度处理流程 =====
    ("FT", "OZG", 10),   # 滤池 → 臭氧接触池
    ("OZG", "ACF", 12),  # 臭氧接触池 → 活性炭滤池
    ("ACF", "CWT", 10),  # 活性炭滤池 → 清水池

    # ===== 化学加药（辅助流程）=====
    ("CDR", "MB", 8),    # 加药间 → 混凝池
    ("CDR", "FB", 6),    # 加药间 → 絮凝池
    ("CLS", "CWT", 8),   # 加氯间 → 清水池
    ("OZR", "OZG", 10),  # 臭氧发生间 → 臭氧接触池
    ("LAB", "CDR", 5),   # 化验室 → 加药间
    ("CST", "CDR", 6),   # 药剂仓库 → 加药间

    # ===== 污泥处理流程 =====
    ("SB", "STK", 8),    # 沉淀池 → 污泥浓缩池
    ("STK", "SDR", 10),  # 污泥浓缩池 → 污泥脱水机房
    ("SDR", "SYD", 8),   # 污泥脱水机房 → 污泥堆场
    ("CWT", "BWT", 6),   # 清水池 → 反冲洗水池

    # ===== 送配水 =====
    ("IPS", "PH1", 6),   # 取水泵站 → 一级泵房

    # ===== 动力设施 =====
    ("PDR", "PH1", 6),   # 配电室 → 一级泵房
    ("PDR", "PH2", 6),   # 配电室 → 二级泵房
    ("TRF", "PDR", 5),   # 变压器室 → 配电室

    # ===== 行政办公 =====
    ("CR", "OF", 5),     # 中控室 → 综合办公楼
    ("CR", "PDR", 4),    # 中控室 → 配电室
    ("GT", "OF", 4),     # 门卫室 → 综合办公楼

    # ===== 生活区（宿舍联合布置）=====
    ("DM1", "DM2", 12),   # 宿舍1-2必须相邻
    ("DM2", "DM3", 12),   # 宿舍2-3必须相邻
    ("DM3", "DM4", 12),   # 宿舍3-4必须相邻
    ("DM1", "SPF", 8),    # 宿舍靠近运动场
    ("DM4", "SPF", 8),    # 宿舍靠近运动场
    ("OF", "DM1", 5),     # 宿舍靠近办公楼
]

# ===== 工艺管线系统定义 =====
# 管线类型: (起点, 终点, 管线类型, 线宽, 颜色)
# 管线类型: "water"=原水/净水, "sludge"=污泥, "chemical"=药剂, "backwash"=反冲洗

PIPELINE_SYSTEM = {
    # ===== 主水处理管线（深蓝色实线）=====
    "main_water": {
        "color": "#1565C0",  # 深蓝色
        "width": 1.0,        # 更细的线
        "style": "-",
        "label": "主水处理管线",
        "pipes": [
            ("IPS", "GC"),   # 取水泵站 → 格栅间
            ("GC", "PS"),    # 格栅间 → 预沉池
            ("PS", "MB"),    # 预沉池 → 混凝池
            ("MB", "FB"),    # 混凝池 → 絮凝池
            ("FB", "SB"),    # 絮凝池 → 沉淀池
            ("SB", "FT"),    # 沉淀池 → 滤池
            ("FT", "CWT"),   # 滤池 → 清水池
            ("CWT", "PH2"),  # 清水池 → 二级泵房
        ]
    },

    # ===== 深度处理管线（紫色实线）=====
    "advanced_water": {
        "color": "#9B59B6",  # 紫色
        "width": 1.0,        # 更细的线
        "style": "-",
        "label": "深度处理管线",
        "pipes": [
            ("FT", "OZG"),   # 滤池 → 臭氧接触池
            ("OZG", "ACF"),  # 臭氧接触池 → 活性炭滤池
            ("ACF", "CWT"),  # 活性炭滤池 → 清水池
        ]
    },

    # ===== 污泥处理管线（棕色实线）=====
    "sludge": {
        "color": "#795548",  # 棕色
        "width": 1.0,        # 更细的线
        "style": "-",
        "label": "污泥管线",
        "pipes": [
            ("SB", "STK"),   # 沉淀池 → 污泥浓缩池
            ("STK", "SDR"),  # 污泥浓缩池 → 污泥脱水机房
            ("SDR", "SYD"),  # 污泥脱水机房 → 污泥堆场
        ]
    },

    # ===== 反冲洗管线（青色虚线）=====
    "backwash": {
        "color": "#00BCD4",  # 青色
        "width": 1.0,        # 更细的线
        "style": (0, (4, 2)),  # 虚线
        "label": "反冲洗管线",
        "pipes": [
            ("CWT", "BWT"),  # 清水池 → 反冲洗水池
            ("BWT", "FT"),   # 反冲洗水池 → 滤池
        ]
    },

    # ===== 加药管线（红色点划线）=====
    "chemical": {
        "color": "#E91E63",  # 品红色
        "width": 1.0,        # 更细的线
        "style": (0, (3, 1, 1, 1)),  # 点划线
        "label": "加药管线",
        "pipes": [
            ("CST", "CDR"),  # 药剂仓库 → 加药间
            ("CDR", "MB"),   # 加药间 → 混凝池
            ("CDR", "FB"),   # 加药间 → 絮凝池
            ("CLS", "CWT"),  # 加氯间 → 清水池
            ("OZR", "OZG"),  # 臭氧发生间 → 臭氧接触池
        ]
    },

    # ===== 电力线路（橙色短虚线）=====
    "power": {
        "color": "#FF9800",  # 橙色
        "width": 1.0,        # 更细的线
        "style": (0, (2, 2)),  # 短虚线
        "label": "电力线路",
        "pipes": [
            ("TRF", "PDR"),  # 变压器室 → 配电室
            ("PDR", "PH1"),  # 配电室 → 一级泵房
            ("PDR", "PH2"),  # 配电室 → 二级泵房
        ]
    },
}

PIPELINE_ROUTING_PARAMS = {
    "main_water": {"detour_factor": 1.08, "terminal_allowance": 4.0, "weight": 3.0},
    "advanced_water": {"detour_factor": 1.10, "terminal_allowance": 4.0, "weight": 2.4},
    "sludge": {"detour_factor": 1.18, "terminal_allowance": 5.0, "weight": 2.2},
    "backwash": {"detour_factor": 1.15, "terminal_allowance": 3.0, "weight": 1.5},
    "chemical": {"detour_factor": 1.25, "terminal_allowance": 2.5, "weight": 2.0},
    "power": {"detour_factor": 1.12, "terminal_allowance": 2.0, "weight": 1.0},
    "default": {"detour_factor": 1.15, "terminal_allowance": 3.0, "weight": 1.0},
}

PIPE_SEGMENT_TO_SYSTEM = {
    tuple(pipe): system_name
    for system_name, system_data in PIPELINE_SYSTEM.items()
    for pipe in system_data["pipes"]
}

# ==================== 竖向高程约束数据 ====================
# 格式: {设施代码: (地面高程m, 水深m, 类型)}
# 类型: "pump"=泵站(主动提升), "gravity"=重力流构筑物, "end"=终点/储存
# 基准地面参考高程: 100.0m (现状地面)
# 水处理高程均由取水泵站抬高后沿重力流递减

ELEVATION_DATA = {
    # ===== 主水处理流程 (Main Water Treatment) =====
    # IPS 取水泵站: 地面100m，泵后水面107.5m (扬程7.5m)
    "IPS":  {"ground": 100.0, "water_depth": 6.0,  "type": "pump",    "pump_head": 7.5,  "water_out": 107.5},
    # GC  格栅间: 地面103m，水损0.3m，出水107.2m
    "GC":   {"ground": 103.0, "water_depth": 3.5,  "type": "gravity", "loss": 0.30,      "water_out": 107.2},
    # PS  预沉池: 地面102.5m，水损0.5m，出水106.7m
    "PS":   {"ground": 102.5, "water_depth": 4.0,  "type": "gravity", "loss": 0.50,      "water_out": 106.7},
    # MB  混凝池: 地面102m，水损0.3m，出水106.4m
    "MB":   {"ground": 102.0, "water_depth": 3.5,  "type": "gravity", "loss": 0.30,      "water_out": 106.4},
    # FB  絮凝池: 地面102m，水损0.5m，出水105.9m
    "FB":   {"ground": 102.0, "water_depth": 3.5,  "type": "gravity", "loss": 0.50,      "water_out": 105.9},
    # SB  沉淀池: 地面101.5m，水损0.4m，出水105.5m
    "SB":   {"ground": 101.5, "water_depth": 4.0,  "type": "gravity", "loss": 0.40,      "water_out": 105.5},
    # FT  滤池: 地面101m，过滤水头损失2.0m（滤层阻力为主），出水103.5m
    "FT":   {"ground": 101.0, "water_depth": 5.5,  "type": "gravity", "loss": 2.00,      "water_out": 103.5},
    # OZG 臭氧接触池: 地面100.5m，水损0.6m，出水102.9m (深度处理)
    "OZG":  {"ground": 100.5, "water_depth": 5.0,  "type": "gravity", "loss": 0.60,      "water_out": 102.9},
    # ACF 活性炭滤池: 地面100m，水损1.5m，出水101.4m (深度处理)
    "ACF":  {"ground": 100.0, "water_depth": 5.0,  "type": "gravity", "loss": 1.50,      "water_out": 101.4},
    # CWT 清水池: 地面98m (半地埋)，进水101.4m，水面101.0m
    "CWT":  {"ground":  98.0, "water_depth": 5.0,  "type": "end",     "loss": 0.0,       "water_out": 101.0},
    # BWT 反冲洗水池: 地面99m，水面101.5m (由CWT高水位侧引)
    "BWT":  {"ground":  99.0, "water_depth": 4.0,  "type": "end",     "loss": 0.0,       "water_out": 101.5},
    # PH2 二级泵房: 地面100m，从CWT吸水加压送出
    "PH2":  {"ground": 100.0, "water_depth": 0.0,  "type": "pump",    "pump_head": 35.0, "water_out": 135.0},
    # PH1 一级泵房: 地面100m，从取水源输水到IPS
    "PH1":  {"ground": 100.0, "water_depth": 0.0,  "type": "pump",    "pump_head": 5.0,  "water_out": 105.0},

    # ===== 污泥处理流程 (Sludge Treatment) =====
    # SB  污泥出口(底部排泥): 地面101.5m，排泥口约99.5m
    # STK 污泥浓缩池: 地面97.5m，接收沉淀池排泥(99.5m)，水损0.3m
    "STK":  {"ground":  97.5, "water_depth": 3.5,  "type": "gravity", "loss": 0.30,      "water_out":  99.2},
    # SDR 污泥脱水机房: 地面96.5m，浓缩污泥由STK重力流入
    "SDR":  {"ground":  96.5, "water_depth": 0.0,  "type": "gravity", "loss": 0.20,      "water_out":  99.0},
    # SYD 污泥堆场: 地面95.5m（最低点，便于清运）
    "SYD":  {"ground":  95.5, "water_depth": 0.0,  "type": "end",     "loss": 0.0,       "water_out":  95.5},

    # ===== 深度处理辅助 =====
    "OZR":  {"ground": 100.0, "water_depth": 0.0,  "type": "gravity", "loss": 0.0,       "water_out": 100.0},
    "CDR":  {"ground": 100.0, "water_depth": 0.0,  "type": "gravity", "loss": 0.0,       "water_out": 100.0},
    "CLS":  {"ground": 100.0, "water_depth": 0.0,  "type": "gravity", "loss": 0.0,       "water_out": 100.0},
    "CST":  {"ground": 100.0, "water_depth": 0.0,  "type": "gravity", "loss": 0.0,       "water_out": 100.0},
}

# 相邻构筑物之间的沿程水头损失系数 (m/m) - 简化为均一坡度
# 实际 = 0.001 × 管长 + 0.3 (局部损失)
PIPE_HEADLOSS_PER_METER = 0.001   # 全局默认沿程损失系数 (m/m)，后备用
PIPE_LOCAL_LOSS = 0.30            # 全局默认局部损失 (m)
GRAVITY_FLOW_SAFETY = 0.20        # 全局默认安全余量 (m)

# 沉淀池底部排泥口高程（用于SB→STK污泥重力流计算）
SB_SLUDGE_OUTLET_ELEV = 99.5

# ===== 各管段精细水力参数 =====
# 格式: {(上游, 下游): (水力坡度i, 局部损失hm, 安全余量hs, 段惩罚权重w)}
# 清水管(低粘度): i=0.001~0.0015  |  污泥管(高粘度): i=0.003~0.006
# 段惩罚权重: 主工艺核心段 2.5, 深度处理分支 1.5, 污泥段 1.2, 次要 0.8
ELEV_SEGMENT_PARAMS = {
    # ===== 主水处理（清水 / 低粘度）=====
    ("IPS", "GC"):  (0.0010, 0.20, 0.20, 2.0),   # 泵后出水→格栅间
    ("GC",  "PS"):  (0.0010, 0.30, 0.20, 2.0),   # 格栅→预沉
    ("PS",  "MB"):  (0.0012, 0.30, 0.20, 2.5),   # 预沉→混凝  ★核心
    ("MB",  "FB"):  (0.0012, 0.25, 0.20, 2.5),   # 混凝→絮凝  ★核心
    ("FB",  "SB"):  (0.0010, 0.30, 0.20, 2.5),   # 絮凝→沉淀  ★核心
    ("SB",  "FT"):  (0.0010, 0.30, 0.20, 2.5),   # 沉淀→滤池  ★核心
    ("FT",  "CWT"): (0.0010, 0.25, 0.20, 2.0),   # 滤池→清水池
    # ===== 深度处理分支 =====
    ("FT",  "OZG"): (0.0012, 0.30, 0.20, 1.5),   # 滤池→臭氧接触池
    ("OZG", "ACF"): (0.0012, 0.30, 0.20, 1.5),   # 臭氧→活性炭滤池
    ("ACF", "CWT"): (0.0010, 0.25, 0.20, 1.5),   # 活性炭→清水池
    ("CWT", "BWT"): (0.0008, 0.20, 0.15, 0.8),   # 清水池→反冲洗水池（次要）
    # ===== 污泥处理（高粘度，需更大坡度）=====
    ("SB",  "STK"): (0.0040, 0.50, 0.30, 1.2),   # 沉淀排泥口→浓缩池（用排泥口高程）
    ("STK", "SDR"): (0.0050, 0.60, 0.30, 1.2),   # 浓缩池→脱水机房
    ("SDR", "SYD"): (0.0030, 0.40, 0.25, 0.8),   # 脱水机房→污泥堆场
}

# 主水处理重力流序列 (需依次满足高程衔接)
MAIN_WATER_ELEV_PATH = [
    ("IPS", "GC"),   # 泵后 → 格栅间
    ("GC",  "PS"),   # 格栅间 → 预沉池
    ("PS",  "MB"),   # 预沉池 → 混凝池
    ("MB",  "FB"),   # 混凝池 → 絮凝池
    ("FB",  "SB"),   # 絮凝池 → 沉淀池
    ("SB",  "FT"),   # 沉淀池 → 滤池
    ("FT",  "CWT"),  # 滤池 → 清水池 (常规路径)
    ("FT",  "OZG"),  # 滤池 → 臭氧接触池 (深度处理)
    ("OZG", "ACF"),  # 臭氧接触池 → 活性炭滤池
    ("ACF", "CWT"),  # 活性炭滤池 → 清水池
    ("CWT", "BWT"),  # 清水池 → 反冲洗水池
]

# 污泥处理重力流序列
SLUDGE_ELEV_PATH = [
    ("SB",  "STK"),  # 沉淀池排泥 → 污泥浓缩池
    ("STK", "SDR"),  # 污泥浓缩池 → 污泥脱水机房
    ("SDR", "SYD"),  # 污泥脱水机房 → 污泥堆场
]


def create_facilities() -> List[Facility]:
    return [Facility(name, name_cn, l, w, cat, sl)
            for name, name_cn, l, w, cat, sl in FACILITIES_DATA]

def get_safety_distance(f1: Facility, f2: Facility) -> float:
    """获取两个设施之间的安全间距 - 同分区内更紧凑"""
    # 同分区内间距更小
    if f1.category == f2.category:
        key = tuple(sorted([f1.safety_level, f2.safety_level]))
        base = SAFETY_DISTANCES.get(key, 2)
        return max(1, base - 1)  # 同分区减1米
    key = tuple(sorted([f1.safety_level, f2.safety_level]))
    return SAFETY_DISTANCES.get(key, 2)


def facility_center(f: Facility) -> Tuple[float, float]:
    """设施中心点坐标。"""
    w, h = f.get_dimensions()
    return f.x + w / 2, f.y + h / 2


def facility_edge_gap(f1: Facility, f2: Facility) -> Tuple[float, float]:
    """两个矩形设施边界之间的水平/竖向净距；重叠投影方向为0。"""
    w1, h1 = f1.get_dimensions()
    w2, h2 = f2.get_dimensions()
    gap_x = max(0.0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
    gap_y = max(0.0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))
    return gap_x, gap_y


def infer_pipeline_system(n1: str, n2: str, default: str = "default") -> str:
    """根据管段端点推断管线系统，方向不匹配时尝试反向。"""
    return (
        PIPE_SEGMENT_TO_SYSTEM.get((n1, n2)) or
        PIPE_SEGMENT_TO_SYSTEM.get((n2, n1)) or
        default
    )


def calculate_pipeline_segment(f1: Facility, f2: Facility,
                               system_name: str = "default") -> Dict[str, float]:
    """
    估算单段管线水平投影长度。

    相比中心点曼哈顿距离，这里用设施边界净距作为主体长度，并加入接入余量、
    转弯余量和管线系统绕行系数，更接近厂区正交管廊/管沟布置。
    """
    params = PIPELINE_ROUTING_PARAMS.get(system_name, PIPELINE_ROUTING_PARAMS["default"])
    cx1, cy1 = facility_center(f1)
    cx2, cy2 = facility_center(f2)
    center_manhattan = abs(cx2 - cx1) + abs(cy2 - cy1)

    gap_x, gap_y = facility_edge_gap(f1, f2)
    edge_clearance = gap_x + gap_y
    elbow_allowance = 1.5 if gap_x > 0 and gap_y > 0 else 0.0
    base_external = edge_clearance + params["terminal_allowance"] + elbow_allowance

    # 边界净距在相邻构筑物上可能接近0，保留最小外部接管长度。
    min_external = params["terminal_allowance"]
    routed_length = max(min_external, base_external) * params["detour_factor"]
    return {
        "center_manhattan_m": center_manhattan,
        "edge_clearance_m": edge_clearance,
        "routed_length_m": routed_length,
        "weighted_length_m": routed_length * params["weight"],
        "detour_factor": params["detour_factor"],
        "terminal_allowance_m": params["terminal_allowance"],
        "weight": params["weight"],
    }


def calculate_pipeline_network(placed: List[Facility],
                               systems: Dict = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """计算各管线系统和管段的长度明细。"""
    systems = systems or PIPELINE_SYSTEM
    fdict = {f.name: f for f in placed}
    rows = []
    for system_name, system_data in systems.items():
        for n1, n2 in system_data["pipes"]:
            if n1 not in fdict or n2 not in fdict:
                continue
            seg = calculate_pipeline_segment(fdict[n1], fdict[n2], system_name)
            rows.append({
                "system": system_name,
                "label": system_data.get("label", system_name),
                "from": n1,
                "to": n2,
                **seg,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "routed_length_m": 0.0,
            "weighted_length_m": 0.0,
            "center_manhattan_m": 0.0,
        }
    else:
        summary = {
            "routed_length_m": float(df["routed_length_m"].sum()),
            "weighted_length_m": float(df["weighted_length_m"].sum()),
            "center_manhattan_m": float(df["center_manhattan_m"].sum()),
        }
    return df, summary

# ==================== 功能分区联合布置器 ====================
class ZonePacker:
    """功能分区联合布置 - 先分区再排样"""

    def __init__(self, facilities: List[Facility]):
        self.facilities = facilities
        self.zones = self._create_zones()

    def _create_zones(self) -> Dict[str, List[Facility]]:
        """按功能分区分组"""
        zones = {}
        for f in self.facilities:
            if f.category not in zones:
                zones[f.category] = []
            zones[f.category].append(f)
        return zones

    def pack_zone(self, zone_facilities: List[Facility],
                  max_width: float) -> Tuple[List[Facility], float, float]:
        """对单个分区内的设施进行紧凑排列"""
        if not zone_facilities:
            return [], 0, 0

        # 按面积降序排列
        sorted_f = sorted(zone_facilities, key=lambda f: f.area, reverse=True)
        placed = []

        # 使用简单的行排列算法
        current_x, current_y = 0, 0
        row_height = 0
        max_x = 0

        for f in sorted_f:
            facility = f.copy()
            w, h = facility.get_dimensions()
            gap = 2  # 分区内部间距

            # 检查是否需要换行
            if current_x + w > max_width and current_x > 0:
                current_x = 0
                current_y += row_height + gap
                row_height = 0

            facility.x = current_x
            facility.y = current_y
            placed.append(facility)

            current_x += w + gap
            row_height = max(row_height, h)
            max_x = max(max_x, facility.x + w)

        zone_width = max_x
        zone_height = current_y + row_height

        return placed, zone_width, zone_height


# ==================== 高级智能优化系统 V4.0 ====================
# 融合: 多目标进化(MOEA) + 差分进化(DE) + 模拟退火(SA) + 深度强化学习引导

class AdvancedGA:
    """高级智能优化算法 - 多策略融合版 V4.1（深度优化）"""

    def __init__(self, facilities: List[Facility],
                 population_size: int = 200,   # 种群规模（平衡效率与多样性）
                 generations: int = 100,       # 正式优化代数
                 mutation_rate: float = 0.35,  # 更高变异率
                 random_seed: int = None,
                 enable_cache: bool = True):   # 适应度缓存，加速重复个体评估
        self.facilities = facilities
        self.n = len(facilities)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.total_area = sum(f.area for f in facilities)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_layout = None
        self.fitness_history = []
        self.enable_cache = enable_cache
        self._fitness_cache = {}
        self._fitness_cache_lock = Lock()
        self._fitness_cache_max_size = 50000

        # 分区信息
        self.categories = list(set(f.category for f in facilities))
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}

        # ========== 高级优化参数 ==========
        # 岛屿模型参数
        self.n_islands = 6  # 6个并行岛屿（增强多样性）
        self.migration_interval = 15  # 更频繁迁移
        self.migration_rate = 0.15  # 更多迁移比例

        # 差分进化参数
        self.de_F = 0.8  # 差分权重
        self.de_CR = 0.9  # 交叉概率

        # 模拟退火参数
        self.sa_T0 = 1000.0  # 初始温度
        self.sa_alpha = 0.995  # 降温系数
        self.sa_T_min = 1.0  # 最低温度

        # Pareto前沿
        self.pareto_front = []  # 非支配解集
        self.pareto_max_size = 100

        # 强化学习状态
        self.rl_q_table = {}  # Q表
        self.rl_alpha = 0.1  # 学习率
        self.rl_gamma = 0.9  # 折扣因子
        self.rl_epsilon = 0.2  # 探索率
        self.rl_actions = ['swap', 'insert', 'reverse', 'rotate', 'zone_shuffle', 'local_search']

        # 自适应控制器
        self.adaptive_controller = {
            'mutation_history': [],
            'crossover_history': [],
            'success_rates': {'mutation': 0.5, 'crossover': 0.5},
            'population_diversity': 1.0
        }

    def _repair_chromosome(self, chrom: Dict) -> Dict:
        """修复染色体，确保顺序、旋转、分区和策略字段始终有效。"""
        c = copy.deepcopy(chrom) if chrom is not None else {}

        raw_order = c.get('order', [])
        order = []
        used = set()
        for item in raw_order:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < self.n and idx not in used:
                order.append(idx)
                used.add(idx)
        missing = [idx for idx in range(self.n) if idx not in used]
        if missing:
            order.extend(missing)
        c['order'] = order[:self.n]

        raw_rotations = list(c.get('rotations', []))
        if len(raw_rotations) < self.n:
            raw_rotations.extend([False] * (self.n - len(raw_rotations)))
        c['rotations'] = [bool(v) for v in raw_rotations[:self.n]]

        raw_zone_order = c.get('zone_order', [])
        zone_order = []
        seen = set()
        for cat in raw_zone_order:
            if cat in self.categories and cat not in seen:
                zone_order.append(cat)
                seen.add(cat)
        for cat in self.categories:
            if cat not in seen:
                zone_order.append(cat)
        c['zone_order'] = zone_order

        try:
            strategy = int(c.get('strategy', 0))
        except (TypeError, ValueError):
            strategy = 0
        c['strategy'] = strategy if strategy in (0, 1, 2) else 0
        return c

    def _chromosome_key(self, chrom: Dict) -> Tuple:
        """生成适应度缓存键。"""
        return (
            tuple(chrom['order']),
            tuple(chrom['rotations']),
            tuple(chrom['zone_order']),
            chrom.get('strategy', 0),
        )

    def create_chromosome(self) -> Dict:
        """染色体: 设施顺序 + 旋转 + 分区排列顺序 + 布局策略"""
        # 按分区分组
        groups = {cat: [] for cat in self.categories}
        for i, f in enumerate(self.facilities):
            groups[f.category].append(i)

        # 每组内按面积排序，加入随机性
        order = []
        zone_order = self.categories.copy()

        # 分区顺序策略
        zone_strategy = random.random()
        if zone_strategy < 0.5:
            # 按工艺流程顺序 - 确保包含所有分区（生活区放最后）
            priority = ['intake', 'process_core', 'advanced', 'chemical',
                        'sludge', 'distribution', 'power', 'admin', 'auxiliary', 'living']
            # 先添加priority中存在的分区
            zone_order = [z for z in priority if z in self.categories]
            # 再添加任何遗漏的分区
            for cat in self.categories:
                if cat not in zone_order:
                    zone_order.append(cat)
        else:
            random.shuffle(zone_order)

        for cat in zone_order:
            group = groups[cat].copy()
            if random.random() < 0.7:
                group.sort(key=lambda i: self.facilities[i].area, reverse=True)
            else:
                random.shuffle(group)
            order.extend(group)

        # 验证order长度
        if len(order) != self.n:
            order = list(range(self.n))
            random.shuffle(order)

        rotations = [random.choice([True, False]) for _ in range(self.n)]

        # 随机选择布局策略 (0=工艺流程优先, 1=分区紧凑, 2=Skyline)
        strategy = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]

        return self._repair_chromosome({
            'order': order,
            'rotations': rotations,
            'zone_order': zone_order,
            'strategy': strategy
        })

    # ========== 高级优化方法 ==========

    def _calculate_multi_objective(self, chrom: Dict) -> Tuple[float, float, float, float, float]:
        """多目标评估：面积、管线长度、工艺流程、紧凑度、高程违规"""
        base_size = np.sqrt(self.total_area) * 1.8
        placed, bbox_w, bbox_h = self.decode_chromosome(chrom, base_size)

        if len(placed) < self.n:
            return float('inf'), float('inf'), float('inf'), 0.0, float('inf')

        # 目标1: 占地面积
        area = bbox_w * bbox_h

        # 目标2: 管线总长度
        pipe_length = self._calc_pipeline_length(placed)

        # 目标3: 工艺流程违规
        adj_penalty, _ = self._calc_adjacency_penalty_strict(placed)

        # 目标4: 空间利用率
        utilization = self.total_area / area if area > 0 else 0

        # 目标5: 竖向高程约束违规量(m) — 最小化，0为完全满足重力流
        elev_violation = self._calc_elevation_constraint(placed)

        return area, pipe_length, adj_penalty, utilization, elev_violation

    def _dominates(self, obj1: Tuple, obj2: Tuple) -> bool:
        """判断obj1是否支配obj2（Pareto支配）"""
        # obj1支配obj2：obj1在所有目标上都不差，且至少有一个更好
        # 目标1-3 最小化: area, pipe_length, adj_penalty
        # 目标4   最大化: utilization
        # 目标5   最小化: elev_violation (高程违规量)
        better_in_one = False
        for i in range(3):  # 前3个最小化
            if obj1[i] > obj2[i]:
                return False
            if obj1[i] < obj2[i]:
                better_in_one = True
        # 第4个最大化（利用率）
        if obj1[3] < obj2[3]:
            return False
        if obj1[3] > obj2[3]:
            better_in_one = True
        # 第5个最小化（高程违规，5目标可能不存在时兼容4目标）
        if len(obj1) > 4 and len(obj2) > 4:
            if obj1[4] > obj2[4]:
                return False
            if obj1[4] < obj2[4]:
                better_in_one = True
        return better_in_one

    def _update_pareto_front(self, chrom: Dict, objectives: Tuple):
        """更新Pareto前沿"""
        # 移除被新解支配的解
        self.pareto_front = [(c, o) for c, o in self.pareto_front
                            if not self._dominates(objectives, o)]

        # 检查新解是否被现有解支配
        is_dominated = any(self._dominates(o, objectives) for _, o in self.pareto_front)

        if not is_dominated:
            self.pareto_front.append((copy.deepcopy(chrom), objectives))
            # 限制大小
            if len(self.pareto_front) > self.pareto_max_size:
                # 移除最拥挤的解
                self.pareto_front = self._crowding_distance_selection(
                    self.pareto_front, self.pareto_max_size)

    def _crowding_distance_selection(self, front: List, n: int) -> List:
        """基于拥挤距离的选择"""
        if len(front) <= n:
            return front

        # 计算拥挤距离
        distances = [0.0] * len(front)
        num_objectives = len(front[0][1])   # 自动适应目标数量(4或5)

        for m in range(num_objectives):
            # 按第m个目标排序
            sorted_idx = sorted(range(len(front)), key=lambda i: front[i][1][m])

            # 边界解无穷大距离
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')

            # 中间解
            obj_range = front[sorted_idx[-1]][1][m] - front[sorted_idx[0]][1][m]
            if obj_range > 0:
                for i in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[i]] += (
                        front[sorted_idx[i+1]][1][m] - front[sorted_idx[i-1]][1][m]
                    ) / obj_range

        # 选择拥挤距离最大的n个
        selected_idx = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)[:n]
        return [front[i] for i in selected_idx]

    def _differential_evolution_mutate(self, target: Dict, pop: List) -> Dict:
        """差分进化变异：DE/rand/1策略"""
        target = self._repair_chromosome(target)
        pop = [self._repair_chromosome(c) for c in pop]
        # 随机选择3个不同个体
        candidates = [c for c in pop if c != target]
        if len(candidates) < 3:
            return self.mutate(target)

        a, b, c = random.sample(candidates, 3)

        # 对order进行差分变异（使用排列版本的DE）
        mutant = copy.deepcopy(target)

        # 基于位置的差分
        for i in range(self.n):
            if random.random() < self.de_CR:
                # 找到a中第i个元素在b和c中的位置差
                try:
                    pos_a = a['order'].index(i)
                    pos_b = b['order'].index(i)
                    pos_c = c['order'].index(i)
                    new_pos = int(pos_a + self.de_F * (pos_b - pos_c)) % self.n
                    # 交换到新位置
                    if 0 <= new_pos < self.n:
                        old_idx = mutant['order'].index(i)
                        mutant['order'][old_idx], mutant['order'][new_pos] = \
                            mutant['order'][new_pos], mutant['order'][old_idx]
                except ValueError:
                    pass

        # 旋转差分
        for i in range(self.n):
            if random.random() < self.de_CR:
                # 多数投票
                votes = [a['rotations'][i], b['rotations'][i], c['rotations'][i]]
                mutant['rotations'][i] = max(set(votes), key=votes.count)

        # 策略差分
        if random.random() < self.de_CR:
            strategies = [a.get('strategy', 0), b.get('strategy', 0), c.get('strategy', 0)]
            mutant['strategy'] = max(set(strategies), key=strategies.count)

        return self._repair_chromosome(mutant)

    def _simulated_annealing_local_search(self, chrom: Dict, T: float) -> Dict:
        """模拟退火局部搜索"""
        current = self._repair_chromosome(chrom)
        current_fit = self.calculate_fitness(current)

        # 多次邻域搜索
        for _ in range(5):
            # 生成邻域解
            neighbor = self._generate_neighbor(current)
            neighbor_fit = self.calculate_fitness(neighbor)

            # 接受准则
            delta = neighbor_fit - current_fit
            if delta < 0:  # 更好的解直接接受
                current = neighbor
                current_fit = neighbor_fit
            elif T > 0 and random.random() < np.exp(-delta / T):
                # 以一定概率接受较差解
                current = neighbor
                current_fit = neighbor_fit

        return self._repair_chromosome(current)

    def _generate_neighbor(self, chrom: Dict) -> Dict:
        """生成邻域解"""
        neighbor = self._repair_chromosome(chrom)

        # 随机选择变异类型
        mut_type = random.choice(['swap', 'insert', 'reverse', '2opt', 'block_swap'])

        if mut_type == 'swap':
            i, j = random.sample(range(self.n), 2)
            neighbor['order'][i], neighbor['order'][j] = neighbor['order'][j], neighbor['order'][i]

        elif mut_type == 'insert':
            i = random.randint(0, self.n - 1)
            j = random.randint(0, self.n - 1)
            item = neighbor['order'].pop(i)
            neighbor['order'].insert(j, item)

        elif mut_type == 'reverse':
            i, j = sorted(random.sample(range(self.n), 2))
            neighbor['order'][i:j+1] = reversed(neighbor['order'][i:j+1])

        elif mut_type == '2opt':
            # 2-opt局部优化
            i, j = sorted(random.sample(range(self.n), 2))
            neighbor['order'] = neighbor['order'][:i] + neighbor['order'][i:j+1][::-1] + neighbor['order'][j+1:]

        elif mut_type == 'block_swap':
            # 块交换：交换两个连续块
            block_size = random.randint(2, max(3, self.n // 8))
            i = random.randint(0, self.n - 2 * block_size)
            j = random.randint(i + block_size, self.n - block_size)
            block1 = neighbor['order'][i:i+block_size]
            block2 = neighbor['order'][j:j+block_size]
            neighbor['order'][i:i+block_size] = block2
            neighbor['order'][j:j+block_size] = block1

        # 有时也变异旋转
        if random.random() < 0.3:
            idx = random.randint(0, self.n - 1)
            neighbor['rotations'][idx] = not neighbor['rotations'][idx]

        return self._repair_chromosome(neighbor)

    def _rl_select_action(self, state: str) -> str:
        """强化学习选择动作"""
        if random.random() < self.rl_epsilon:
            return random.choice(self.rl_actions)

        # 贪婪选择
        if state not in self.rl_q_table:
            self.rl_q_table[state] = {a: 0.0 for a in self.rl_actions}

        return max(self.rl_actions, key=lambda a: self.rl_q_table[state][a])

    def _rl_update(self, state: str, action: str, reward: float, next_state: str):
        """Q-learning更新"""
        if state not in self.rl_q_table:
            self.rl_q_table[state] = {a: 0.0 for a in self.rl_actions}
        if next_state not in self.rl_q_table:
            self.rl_q_table[next_state] = {a: 0.0 for a in self.rl_actions}

        max_next_q = max(self.rl_q_table[next_state].values())
        self.rl_q_table[state][action] += self.rl_alpha * (
            reward + self.rl_gamma * max_next_q - self.rl_q_table[state][action]
        )

    def _get_rl_state(self, fitness: float, stagnation: int, diversity: float) -> str:
        """获取强化学习状态表示"""
        # 离散化状态
        fit_level = 'low' if fitness < 20000 else ('mid' if fitness < 30000 else 'high')
        stag_level = 'none' if stagnation < 10 else ('some' if stagnation < 30 else 'stuck')
        div_level = 'low' if diversity < 0.3 else ('mid' if diversity < 0.7 else 'high')
        return f"{fit_level}_{stag_level}_{div_level}"

    def _rl_guided_mutate(self, chrom: Dict, action: str) -> Dict:
        """根据RL动作执行变异"""
        mutant = self._repair_chromosome(chrom)

        if action == 'swap':
            i, j = random.sample(range(self.n), 2)
            mutant['order'][i], mutant['order'][j] = mutant['order'][j], mutant['order'][i]

        elif action == 'insert':
            i, j = random.randint(0, self.n-1), random.randint(0, self.n-1)
            item = mutant['order'].pop(i)
            mutant['order'].insert(j, item)

        elif action == 'reverse':
            i, j = sorted(random.sample(range(self.n), 2))
            mutant['order'][i:j+1] = reversed(mutant['order'][i:j+1])

        elif action == 'rotate':
            for i in range(self.n):
                if random.random() < 0.2:
                    mutant['rotations'][i] = not mutant['rotations'][i]

        elif action == 'zone_shuffle':
            random.shuffle(mutant['zone_order'])

        elif action == 'local_search':
            mutant = self._simulated_annealing_local_search(mutant, self.sa_T0 * 0.1)

        return self._repair_chromosome(mutant)

    def _calculate_diversity(self, pop: List) -> float:
        """计算种群多样性"""
        if len(pop) < 2:
            return 1.0

        total_diff = 0
        count = 0
        sample_size = min(20, len(pop))
        sample = random.sample(pop, sample_size)

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                # 计算顺序差异
                diff = sum(1 for k in range(self.n)
                          if sample[i]['order'][k] != sample[j]['order'][k])
                total_diff += diff / self.n
                count += 1

        return total_diff / count if count > 0 else 1.0

    def _adaptive_parameter_control(self, gen: int, stagnation: int, diversity: float):
        """自适应参数控制"""
        # 根据多样性调整变异率
        if diversity < 0.2:  # 多样性太低
            self.mutation_rate = min(0.6, self.mutation_rate * 1.2)
        elif diversity > 0.8:  # 多样性太高
            self.mutation_rate = max(0.1, self.mutation_rate * 0.9)

        # 根据停滞调整参数
        if stagnation > 30:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
            self.de_CR = min(0.95, self.de_CR + 0.02)
        elif stagnation < 5:
            self.mutation_rate = max(0.15, self.mutation_rate * 0.95)

        # 模拟退火温度衰减
        progress = gen / self.generations
        self.sa_T0 = max(self.sa_T_min, 1000.0 * (1 - progress) ** 2)

        # 强化学习探索率衰减
        self.rl_epsilon = max(0.05, 0.2 * (1 - progress))

    def decode_chromosome(self, chrom: Dict, container_size: float
                         ) -> Tuple[List[Facility], float, float]:
        """
        四大分区布局解码器
        分区: 10个精细化功能分区
        布局: 工艺流程驱动 + 紧凑排列
        """
        chrom = self._repair_chromosome(chrom)
        return self._decode_process_driven_layout(chrom, container_size)

    def _decode_process_driven_layout(self, chrom: Dict, container_size: float
                                     ) -> Tuple[List[Facility], float, float]:
        """
        工艺流程驱动布局 - 确保水处理工艺严格相邻
        布局策略:
        1. 主工艺流程"U"型或"L"型排列（严格相邻）
        2. 深度处理分支流程
        3. 辅助设施就近布置
        4. 非工艺设施紧凑填充
        """
        name_to_idx = {self.facilities[i].name: i for i in range(self.n)}
        placed = []
        placed_names = set()
        name_to_placed = {}

        # ========== 第一步：主工艺流程布局（U型折返，保持相邻）==========
        # 主水处理流程：取水→格栅→预沉→混凝→絮凝→沉淀→滤池→清水池
        main_process = ["IPS", "GC", "PS", "MB", "FB", "SB", "FT", "CWT"]

        # 计算主流程总长度，决定折返策略
        main_widths = []
        main_heights = []
        for name in main_process:
            if name in name_to_idx:
                idx = name_to_idx[name]
                f = self.facilities[idx].copy()
                f.rotated = chrom['rotations'][idx]
                w, h = f.get_dimensions()
                main_widths.append(w)
                main_heights.append(h)

        total_main_width = sum(main_widths) + (len(main_widths) - 1) * 2
        target_w = np.sqrt(self.total_area) * 1.4

        # 决定折返点：在第一行放置前半段，第二行放置后半段（反向）
        if total_main_width > target_w * 1.2:
            # 采用U型布局
            # 第一行：IPS→GC→PS→MB→FB→SB
            # 第二行：CWT←FT（折返）
            row1_names = main_process[:6]  # IPS到SB
            row2_names = main_process[6:]  # FT, CWT
        else:
            # 线性布局
            row1_names = main_process
            row2_names = []

        # 第一行布局 - 使用正确的安全距离
        current_x = 0
        row_y = 0
        row1_height = 0
        row1_end_x = 0
        last_facility = None

        for name in row1_names:
            if name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            f = self.facilities[idx].copy()
            f.rotated = chrom['rotations'][idx]
            w, h = f.get_dimensions()

            # 计算与上一个设施的安全距离
            if last_facility:
                gap = get_safety_distance(last_facility, f)
                current_x += gap  # 先加安全距离

            f.x = current_x
            f.y = row_y
            placed.append(f)
            placed_names.add(name)
            name_to_placed[name] = f
            row1_end_x = current_x + w
            current_x += w  # 再加设施宽度
            row1_height = max(row1_height, h)
            last_facility = f

        # 第二行布局（U型折返，从SB位置开始向左延伸）
        if row2_names:
            # 找最后一个第一行设施（SB）
            last_row1 = name_to_placed.get(row1_names[-1]) if row1_names else None
            if last_row1:
                # 计算SB到FT的安全距离
                sb_w, sb_h = last_row1.get_dimensions()
                first_row2_idx = name_to_idx.get(row2_names[0])
                if first_row2_idx is not None:
                    first_row2_f = self.facilities[first_row2_idx]
                    gap_to_row1 = get_safety_distance(last_row1, first_row2_f)
                else:
                    gap_to_row1 = 3
                row2_y = row1_height + gap_to_row1
                current_x = last_row1.x
            else:
                row2_y = row1_height + 3
                current_x = row1_end_x

            last_facility = None
            for name in row2_names:
                if name not in name_to_idx:
                    continue
                idx = name_to_idx[name]
                f = self.facilities[idx].copy()
                f.rotated = chrom['rotations'][idx]
                w, h = f.get_dimensions()

                # 计算与上一个设施的安全距离
                if last_facility:
                    gap = get_safety_distance(last_facility, f)
                    current_x += gap

                f.x = current_x
                f.y = row2_y
                placed.append(f)
                placed_names.add(name)
                name_to_placed[name] = f
                current_x += w
                last_facility = f

        # ========== 第二步：深度处理流程（分支工艺）==========
        # 深度处理分支：FT→OZG→ACF→(接回CWT)
        advanced_process = ["OZG", "ACF"]

        ft_facility = name_to_placed.get("FT")
        cwt_facility = name_to_placed.get("CWT")

        if ft_facility and cwt_facility:
            # 深度处理放在FT附近，需检查冲突
            for name in advanced_process:
                if name not in name_to_idx or name in placed_names:
                    continue
                idx = name_to_idx[name]
                f = self.facilities[idx].copy()
                f.rotated = chrom['rotations'][idx]

                # 使用带冲突检测的相邻位置查找
                target = name_to_placed.get("OZG") if name == "ACF" else ft_facility
                if target is None:
                    target = ft_facility

                pos = self._find_adjacent_position(placed, f, target)
                if pos:
                    f.x, f.y = pos
                    placed.append(f)
                    placed_names.add(name)
                    name_to_placed[name] = f

        # ========== 第三步：辅助设施就近布置（有序）==========
        # 按优先级排序辅助设施布置顺序 - 关键工艺设施优先
        auxiliary_list = [
            # 第一优先级：紧密工艺关联设施
            ("PH2", "CWT"),  # 二级泵房必须紧邻清水池
            ("PH1", "IPS"),  # 一级泵房紧邻取水泵站
            ("CDR", "MB"),   # 加药间紧邻混凝池（投药点）
            ("CLS", "CWT"),  # 加氯间紧邻清水池（消毒点）
            ("BWT", "FT"),   # 反冲洗水池紧邻滤池
            # 第二优先级：辅助支持设施
            ("OZR", "OZG"),  # 臭氧发生间靠近臭氧接触池
            ("CST", "CDR"),  # 药剂仓库靠近加药间
            ("STK", "SB"),   # 污泥浓缩池靠近沉淀池
            ("SDR", "STK"),  # 污泥脱水机房靠近浓缩池
            ("SYD", "SDR"),  # 污泥堆场靠近脱水机房
            # 第三优先级：动力设施
            ("PDR", "PH2"),  # 配电室靠近二级泵房
            ("TRF", "PDR"),  # 变压器室靠近配电室
            # 第四优先级：管理设施
            ("CR", "FT"),    # 中控室靠近滤池（便于监控核心工艺）
            ("LAB", "CR"),   # 化验室靠近中控室
        ]

        for aux_name, target_name in auxiliary_list:
            if aux_name not in name_to_idx or aux_name in placed_names:
                continue
            if target_name not in name_to_placed:
                continue

            idx = name_to_idx[aux_name]
            f = self.facilities[idx].copy()
            f.rotated = chrom['rotations'][idx]
            target = name_to_placed[target_name]

            # 在目标设施周围找位置
            pos = self._find_adjacent_position(placed, f, target)
            if pos:
                f.x, f.y = pos
                placed.append(f)
                placed_names.add(aux_name)
                name_to_placed[aux_name] = f

        # ========== 第四步：剩余设施紧凑布局 ==========
        remaining = []
        for idx in chrom['order']:
            f = self.facilities[idx]
            if f.name not in placed_names:
                fc = f.copy()
                fc.rotated = chrom['rotations'][idx]
                remaining.append(fc)

        # 按面积降序排列剩余设施
        remaining.sort(key=lambda f: -f.area)

        for f in remaining:
            pos = self._find_best_bl_position(placed, f, target_w * 1.5)
            if pos:
                f.x, f.y = pos
            else:
                max_y = max((p.y + p.get_dimensions()[1] for p in placed), default=0)
                f.x, f.y = 0, max_y + 2
            placed.append(f)
            placed_names.add(f.name)

        if not placed:
            return [], 0, 0

        # 门卫室放在入口位置，不与其他设施重叠
        gt_placed = False
        for f in placed:
            if f.name == "GT":
                gt_placed = True
                # 检查(0,0)是否可用
                if not self._check_conflict_at([p for p in placed if p.name != "GT"], f, 0, 0):
                    f.x = 0
                    f.y = 0
                else:
                    # 找一个不冲突的入口位置
                    # 尝试在左侧边缘找位置
                    for y_test in range(0, 100, 5):
                        if not self._check_conflict_at([p for p in placed if p.name != "GT"], f, 0, y_test):
                            f.x = 0
                            f.y = y_test
                            break
                break

        # ========== 最终检查：修复所有重叠和安全距离问题 ==========
        placed = self._fix_overlaps_and_safety(placed)

        # 计算边界
        max_x = max(f.x + f.get_dimensions()[0] for f in placed)
        max_y = max(f.y + f.get_dimensions()[1] for f in placed)

        return placed, max_x, max_y

    def _fix_overlaps_and_safety(self, placed: List[Facility]) -> List[Facility]:
        """修复所有重叠和安全距离问题 - 增强版V4（改进移动逻辑）"""
        max_iterations = 2000  # 更多迭代

        def check_position_valid(fac, x, y, others):
            """检查位置是否与其他设施无冲突"""
            w, h = fac.get_dimensions()
            for other in others:
                if other is fac:
                    continue
                ow, oh = other.get_dimensions()
                req_gap = get_safety_distance(fac, other)

                # 检查是否重叠或安全距离不足
                if x + w + req_gap <= other.x:
                    continue
                if other.x + ow + req_gap <= x:
                    continue
                if y + h + req_gap <= other.y:
                    continue
                if other.y + oh + req_gap <= y:
                    continue
                return False  # 有冲突
            return True

        def find_safe_position(fac, fixed, others):
            """在fixed周围找到不与others冲突的最紧凑位置"""
            tw, th = fac.get_dimensions()
            fw, fh = fixed.get_dimensions()
            gap = get_safety_distance(fac, fixed)

            # 候选位置优先级：尽量保持在原位置附近
            orig_x, orig_y = fac.x, fac.y

            # 计算当前布局边界
            max_x = max(f.x + f.get_dimensions()[0] for f in others if f is not fac)
            max_y = max(f.y + f.get_dimensions()[1] for f in others if f is not fac)

            # 首先尝试最小移动（在布局内）
            candidates = []
            for dx in [gap+1, -(gap+1), gap+fw+1, -(gap+tw+1)]:
                for dy in [0, gap+1, -(gap+1)]:
                    new_x, new_y = orig_x + dx, orig_y + dy
                    if new_x >= 0 and new_y >= 0 and new_x + tw <= max_x + 30:
                        candidates.append((new_x, new_y))

            # 然后尝试fixed周围的标准位置
            candidates.extend([
                (fixed.x + fw + gap, fixed.y),  # 右侧
                (fixed.x, fixed.y + fh + gap),  # 下方
                (max(0, fixed.x - tw - gap), fixed.y),  # 左侧
                (fixed.x, max(0, fixed.y - th - gap)),  # 上方
            ])

            # 过滤掉负坐标和超出边界太多的
            candidates = [(max(0, x), max(0, y)) for x, y in candidates
                         if x + tw <= max_x + 25 and y + th <= max_y + 25]

            # 按与原位置的距离排序
            candidates.sort(key=lambda p: abs(p[0] - orig_x) + abs(p[1] - orig_y))

            for x, y in candidates:
                if check_position_valid(fac, x, y, others):
                    return x, y

            # 如果基本候选都冲突，尝试小步增量（限制范围）
            for offset in [3, 5, 8, 12, 15]:
                test_positions = [
                    (orig_x + offset, orig_y),
                    (orig_x, orig_y + offset),
                    (max(0, orig_x - offset), orig_y),
                    (orig_x, max(0, orig_y - offset)),
                    (orig_x + offset, orig_y + offset),
                ]
                for x, y in test_positions:
                    if x + tw <= max_x + 20 and y + th <= max_y + 20:
                        if check_position_valid(fac, x, y, others):
                            return x, y

            # 保底：在布局边缘找最近的空位（限制扩展）
            for test_y in range(0, int(max_y), 5):
                if check_position_valid(fac, max_x + gap, test_y, others):
                    return max_x + gap, test_y

            return max_x + gap, orig_y if orig_y >= 0 else 0

        for iteration in range(max_iterations):
            has_conflict = False

            for i, f1 in enumerate(placed):
                w1, h1 = f1.get_dimensions()

                for j, f2 in enumerate(placed):
                    if j <= i:
                        continue
                    w2, h2 = f2.get_dimensions()
                    required_gap = get_safety_distance(f1, f2)

                    # 计算两个矩形之间的间距
                    if f1.x + w1 <= f2.x:
                        dx = f2.x - (f1.x + w1)
                    elif f2.x + w2 <= f1.x:
                        dx = f1.x - (f2.x + w2)
                    else:
                        dx = 0

                    if f1.y + h1 <= f2.y:
                        dy = f2.y - (f1.y + h1)
                    elif f2.y + h2 <= f1.y:
                        dy = f1.y - (f2.y + h2)
                    else:
                        dy = 0

                    physical_overlap = (dx == 0 and dy == 0)

                    if not physical_overlap:
                        if dx == 0:
                            actual_gap = dy
                        elif dy == 0:
                            actual_gap = dx
                        else:
                            actual_gap = (dx**2 + dy**2)**0.5
                        safety_violation = actual_gap < required_gap - 0.1
                    else:
                        safety_violation = True

                    if physical_overlap or safety_violation:
                        has_conflict = True
                        # 移动面积较小的设施
                        if f1.area <= f2.area:
                            to_move, fixed = f1, f2
                        else:
                            to_move, fixed = f2, f1

                        # 使用改进的位置查找
                        new_x, new_y = find_safe_position(to_move, fixed, placed)
                        to_move.x, to_move.y = new_x, new_y
                        break

                if has_conflict:
                    break

            if not has_conflict:
                break

        # 确保所有坐标非负
        min_x = min(f.x for f in placed)
        min_y = min(f.y for f in placed)
        if min_x < 0 or min_y < 0:
            for f in placed:
                if min_x < 0:
                    f.x -= min_x
                if min_y < 0:
                    f.y -= min_y

        return placed

    def _find_adjacent_position(self, placed: List[Facility], f: Facility,
                                target: Facility) -> Tuple[float, float]:
        """在目标设施周围找最近的可用位置 - 增强版"""
        w, h = f.get_dimensions()
        tw, th = target.get_dimensions()
        gap = get_safety_distance(f, target)

        # 候选位置优先级：右侧 > 上方 > 下方 > 左侧（水处理流程一般从左到右）
        candidates = [
            (target.x + tw + gap, target.y),                    # 右侧对齐
            (target.x + tw + gap, target.y + th/2 - h/2),       # 右侧居中
            (target.x, target.y + th + gap),                    # 正上方
            (target.x + tw/2 - w/2, target.y + th + gap),       # 上方居中
            (target.x + tw - w, target.y + th + gap),           # 上方右对齐
            (target.x, target.y - h - gap),                     # 正下方
            (target.x + tw/2 - w/2, target.y - h - gap),        # 下方居中
            (target.x - w - gap, target.y),                     # 左侧对齐
            (target.x - w - gap, target.y + th/2 - h/2),        # 左侧居中
            (target.x + tw + gap, target.y + th - h),           # 右下角
            (target.x - w - gap, target.y + th - h),            # 左下角
        ]

        for x, y in candidates:
            if x >= 0 and y >= 0:
                if not self._check_conflict_at(placed, f, x, y):
                    return (x, y)

        # 第二轮尝试：在目标周围扩大搜索范围
        search_range = [gap, gap + 5, gap + 10, gap + 15]
        for offset in search_range:
            for dx in [-w - offset, 0, tw + offset]:
                for dy in [-h - offset, 0, th + offset]:
                    if dx == 0 and dy == 0:
                        continue
                    x, y = target.x + dx, target.y + dy
                    if x >= 0 and y >= 0:
                        if not self._check_conflict_at(placed, f, x, y):
                            return (x, y)

        # 如果都不行，找最近的可用位置
        return self._find_best_bl_position(placed, f, 300)

    def _find_best_bl_position(self, placed: List[Facility], f: Facility,
                               max_w: float) -> Tuple[float, float]:
        """找最佳Bottom-Left位置"""
        w, h = f.get_dimensions()

        if not placed:
            return (0, 0)

        candidates = [(0, 0)]
        for pf in placed:
            pw, ph = pf.get_dimensions()
            gap = get_safety_distance(pf, f)
            candidates.extend([
                (pf.x + pw + gap, pf.y),
                (pf.x, pf.y + ph + gap),
                (0, pf.y + ph + gap),
                (pf.x + pw + gap, 0),
            ])

        best_pos = None
        best_score = float('inf')

        for x, y in candidates:
            if x < 0 or y < 0 or x + w > max_w:
                continue
            if self._check_conflict_at(placed, f, x, y):
                continue

            score = y * 1000 + x
            if score < best_score:
                best_score = score
                best_pos = (x, y)

        return best_pos

    def _pack_horizontal_compact(self, facilities: List[Facility]) -> Tuple[List[Facility], float, float]:
        """水平紧凑排列 - 超紧凑版，目标正方形"""
        if not facilities:
            return [], 0, 0

        # 估算目标宽度 (设施总面积开方 * 1.05 接近正方形)
        total_area = sum(f.area for f in facilities)
        target_w = np.sqrt(total_area) * 1.05

        # 按面积降序，然后高度降序
        sorted_f = sorted(facilities, key=lambda f: (-f.area, -max(f.get_dimensions())))

        placed = []
        rows = []  # [(y_start, row_height, row_width, [facilities])]

        for f in sorted_f:
            w, h = f.get_dimensions()
            w2, h2 = h, w  # 旋转版本

            # 尝试找现有行能放下的位置（不旋转和旋转都尝试）
            best_row = None
            best_gap = float('inf')
            use_rotated = False

            for row_idx, (row_y, row_h, row_w, row_facs) in enumerate(rows):
                # 不旋转
                if row_w + w + 1 <= target_w * 1.15:
                    gap = abs(h - row_h)  # 高度差距
                    if gap < best_gap:
                        best_gap = gap
                        best_row = row_idx
                        use_rotated = False

                # 旋转
                if row_w + w2 + 1 <= target_w * 1.15:
                    gap = abs(h2 - row_h)
                    if gap < best_gap:
                        best_gap = gap
                        best_row = row_idx
                        use_rotated = True

            if best_row is not None:
                row_y, row_h, row_w, row_facs = rows[best_row]
                if use_rotated:
                    f.rotated = not f.rotated
                    w, h = w2, h2
                f.x = row_w + 1
                f.y = row_y
                row_facs.append(f)
                rows[best_row] = (row_y, max(row_h, h), row_w + w + 1, row_facs)
                placed.append(f)
            else:
                # 开新行 - 选择更优的方向
                if rows:
                    last_row_y, last_row_h, _, _ = rows[-1]
                    new_row_y = last_row_y + last_row_h + 1
                else:
                    new_row_y = 0

                # 选择更宽的方向（更适合水平排列）
                if w2 > w:
                    f.rotated = not f.rotated
                    w, h = w2, h2

                f.x = 0
                f.y = new_row_y
                rows.append((new_row_y, h, w, [f]))
                placed.append(f)

        if not placed:
            return [], 0, 0

        max_x = max(p.x + p.get_dimensions()[0] for p in placed)
        max_y = max(p.y + p.get_dimensions()[1] for p in placed)
        return placed, max_x, max_y

    def _pack_skyline_compact(self, facilities: List[Facility]) -> Tuple[List[Facility], float, float]:
        """Skyline算法 - 超紧凑矩形装箱"""
        if not facilities:
            return [], 0, 0

        # 按高度降序，同高度按宽度降序
        sorted_f = sorted(facilities, key=lambda f: (-max(f.get_dimensions()), -min(f.get_dimensions())))

        # 估算容器宽度 (设施总面积开方 * 1.2)
        total_area = sum(f.area for f in facilities)
        container_w = np.sqrt(total_area) * 1.3

        # Skyline: [(x_start, x_end, height), ...]
        skyline = [(0, container_w, 0)]
        placed = []

        for f in sorted_f:
            w, h = f.get_dimensions()

            # 尝试旋转，选择更合适的方向
            w2, h2 = h, w
            use_rotated = False

            # 找最佳放置位置
            best_pos, best_waste = self._find_skyline_pos_v2(skyline, w, h, container_w)
            pos2, waste2 = self._find_skyline_pos_v2(skyline, w2, h2, container_w)

            if pos2 is not None and (best_pos is None or waste2 < best_waste):
                best_pos = pos2
                w, h = w2, h2
                use_rotated = True

            if best_pos is None:
                # 扩展容器宽度
                container_w *= 1.2
                skyline = [(0, container_w, 0)]
                for pf in placed:
                    pw, ph = pf.get_dimensions()
                    self._update_skyline_v2(skyline, pf.x, pw, pf.y + ph)
                best_pos, _ = self._find_skyline_pos_v2(skyline, w, h, container_w)
                if best_pos is None:
                    best_pos = (0, max(s[2] for s in skyline) if skyline else 0)

            f.x, f.y = best_pos
            if use_rotated:
                f.rotated = not f.rotated
            placed.append(f)

            # 更新skyline
            self._update_skyline_v2(skyline, f.x, w, f.y + h)

        if not placed:
            return [], 0, 0

        max_x = max(p.x + p.get_dimensions()[0] for p in placed)
        max_y = max(p.y + p.get_dimensions()[1] for p in placed)
        return placed, max_x, max_y

    def _find_skyline_pos_v2(self, skyline: List, w: float, h: float, container_w: float):
        """在skyline中找最佳位置 - 紧凑版"""
        best_pos = None
        best_waste = float('inf')

        for i, (x1, x2, sh) in enumerate(skyline):
            if x2 - x1 < w:
                continue
            if x1 + w > container_w:
                continue

            # 检查这个位置的实际高度 (需要检查所有被覆盖的skyline段)
            max_h = sh
            coverage_end = x1 + w
            for j in range(i, len(skyline)):
                sx1, sx2, sj_h = skyline[j]
                if sx1 >= coverage_end:
                    break
                max_h = max(max_h, sj_h)

            # 计算浪费空间
            waste = (max_h - sh) * w + (x2 - x1 - w) * h

            if waste < best_waste:
                best_waste = waste
                best_pos = (x1, max_h)

        return best_pos, best_waste

    def _update_skyline_v2(self, skyline: List, x: float, w: float, new_h: float):
        """更新skyline - 紧凑版"""
        x_end = x + w
        new_segments = []

        for sx1, sx2, sh in skyline:
            if sx2 <= x or sx1 >= x_end:
                # 不受影响的段
                new_segments.append((sx1, sx2, sh))
            else:
                # 被覆盖的段
                if sx1 < x:
                    new_segments.append((sx1, x, sh))
                if sx2 > x_end:
                    new_segments.append((x_end, sx2, sh))

        # 添加新段
        new_segments.append((x, x_end, new_h))

        # 排序并合并相邻同高度段
        new_segments.sort(key=lambda s: s[0])
        merged = []
        for seg in new_segments:
            if merged and abs(merged[-1][1] - seg[0]) < 0.01 and abs(merged[-1][2] - seg[2]) < 0.01:
                merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
            else:
                merged.append(seg)

        skyline.clear()
        skyline.extend(merged)

    def _pack_zone_compact(self, facilities: List[Facility]) -> Tuple[List[Facility], float, float]:
        """分区内紧凑排列 - 调用Skyline"""
        return self._pack_skyline_compact(facilities)

    def _decode_process_flow(self, chrom: Dict, container_size: float
                            ) -> Tuple[List[Facility], float, float]:
        """策略1: 工艺流程优先布局"""
        placed = []
        placed_names = set()

        # 工艺流程顺序 - 双线工艺排成两行
        process_line1 = ["IPS", "GC", "PS", "MB", "FB", "SB1", "FT1"]
        process_line2 = ["SB2", "FT2", "OZG", "ACF", "UF", "CWT", "BWT"]

        # 关联设施映射
        related_map = {
            "CDR": "MB", "CLS": "CWT", "FLD": "CWT", "OZR": "OZG",
            "ACT": "ACF", "PH1": "IPS", "PH2": "CWT", "STK": "SB1",
            "RWT": "MB", "PDR": "PH2", "TRF": "PDR",
        }

        name_to_idx = {self.facilities[i].name: i for i in range(self.n)}

        # 第一行工艺流程
        current_x, row1_height = 0, 0
        for name in process_line1:
            if name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            f = self.facilities[idx].copy()
            f.rotated = chrom['rotations'][idx]
            w, h = f.get_dimensions()
            f.x, f.y = current_x, 0
            placed.append(f)
            placed_names.add(name)
            current_x += w + 2
            row1_height = max(row1_height, h)

        row1_width = current_x

        # 第二行工艺流程（在第一行上方）
        current_x = 0
        row2_y = row1_height + 3
        for name in process_line2:
            if name not in name_to_idx:
                continue
            idx = name_to_idx[name]
            f = self.facilities[idx].copy()
            f.rotated = chrom['rotations'][idx]
            w, h = f.get_dimensions()
            f.x, f.y = current_x, row2_y
            placed.append(f)
            placed_names.add(name)
            current_x += w + 2

        name_to_placed = {f.name: f for f in placed}

        # 放置关联设施
        for related_name, target_name in related_map.items():
            if related_name not in name_to_idx or target_name not in name_to_placed:
                continue
            if related_name in placed_names:
                continue

            idx = name_to_idx[related_name]
            f = self.facilities[idx].copy()
            f.rotated = chrom['rotations'][idx]
            target = name_to_placed[target_name]
            tw, th = target.get_dimensions()
            gap = get_safety_distance(f, target)

            # 尝试多个位置：右侧、上方、下方
            positions = [
                (target.x + tw + gap, target.y),
                (target.x, target.y + th + gap),
                (target.x, target.y - f.get_dimensions()[1] - gap),
            ]

            for px, py in positions:
                if py >= 0 and not self._check_conflict_at(placed, f, px, py):
                    f.x, f.y = px, py
                    break
            else:
                # 找最近的空位
                best_pos = self._find_nearest_valid_pos(placed, f, target.x, target.y)
                f.x, f.y = best_pos

            placed.append(f)
            placed_names.add(related_name)
            name_to_placed[related_name] = f

        # 剩余设施用Bottom-Left填充
        self._place_remaining_bl(chrom, placed, placed_names, name_to_idx)

        if not placed:
            return [], 0, 0

        max_x = max(f.x + f.get_dimensions()[0] for f in placed)
        max_y = max(f.y + f.get_dimensions()[1] for f in placed)

        return placed, max_x, max_y

    def _decode_zone_compact(self, chrom: Dict, container_size: float
                            ) -> Tuple[List[Facility], float, float]:
        """策略2: 分区紧凑布局"""
        placed = []
        name_to_idx = {self.facilities[i].name: i for i in range(self.n)}

        # 按分区分组
        zone_facilities = {}
        for idx in chrom['order']:
            f = self.facilities[idx]
            if f.category not in zone_facilities:
                zone_facilities[f.category] = []
            fc = f.copy()
            fc.rotated = chrom['rotations'][idx]
            zone_facilities[f.category].append(fc)

        # 分区布局：采用网格式排列
        zone_layout = {}
        for cat, flist in zone_facilities.items():
            zone_placed, zw, zh = self._pack_zone_bl(flist)
            zone_layout[cat] = {'facilities': zone_placed, 'width': zw, 'height': zh}

        # 动态计算网格大小（适应分区数量）
        zone_order = chrom['zone_order']
        n_zones = len([z for z in zone_order if z in zone_layout])

        # 根据分区数量确定网格尺寸
        if n_zones <= 6:
            cols, rows = 3, 2
        elif n_zones <= 9:
            cols, rows = 3, 3
        elif n_zones <= 12:
            cols, rows = 4, 3
        else:
            cols, rows = 4, 4

        col_widths = [0] * cols
        row_heights = [0] * rows

        # 计算每个位置的分区
        zone_positions = {}
        zone_idx = 0
        for cat in zone_order:
            if cat in zone_layout:
                col, row = zone_idx % cols, zone_idx // cols
                if row < rows:  # 确保不越界
                    zone_positions[cat] = (col, row)
                    zl = zone_layout[cat]
                    col_widths[col] = max(col_widths[col], zl['width'])
                    row_heights[row] = max(row_heights[row], zl['height'])
                zone_idx += 1

        # 放置分区
        for cat, (col, row) in zone_positions.items():
            x_offset = sum(col_widths[:col]) + col * 5
            y_offset = sum(row_heights[:row]) + row * 5

            for f in zone_layout[cat]['facilities']:
                f.x += x_offset
                f.y += y_offset
                placed.append(f)

        if not placed:
            return [], 0, 0

        max_x = max(f.x + f.get_dimensions()[0] for f in placed)
        max_y = max(f.y + f.get_dimensions()[1] for f in placed)

        # 强制门卫室放置在边界
        placed = self._force_gate_to_boundary(placed, max_x, max_y)

        # 重新计算边界
        max_x = max(f.x + f.get_dimensions()[0] for f in placed)
        max_y = max(f.y + f.get_dimensions()[1] for f in placed)

        return placed, max_x, max_y

    def _force_gate_to_boundary(self, placed: List[Facility], bbox_w: float, bbox_h: float) -> List[Facility]:
        """强制门卫室放置在边界位置"""
        for f in placed:
            if f.name == "GT":  # 门卫室
                w, h = f.get_dimensions()
                # 检查是否已在边界
                on_boundary = (f.x < 3 or f.y < 3 or
                              abs(f.x + w - bbox_w) < 3 or
                              abs(f.y + h - bbox_h) < 3)

                if not on_boundary:
                    # 移动到左下角边界（入口位置）
                    # 找一个不冲突的边界位置
                    boundary_positions = [
                        (0, 0),  # 左下角
                        (0, bbox_h - h),  # 左上角
                        (bbox_w - w, 0),  # 右下角
                        (bbox_w - w, bbox_h - h),  # 右上角
                    ]

                    for bx, by in boundary_positions:
                        conflict = False
                        for other in placed:
                            if other.name == f.name:
                                continue
                            ow, oh = other.get_dimensions()
                            gap = get_safety_distance(f, other)
                            if not (bx >= other.x + ow + gap or bx + w + gap <= other.x or
                                    by >= other.y + oh + gap or by + h + gap <= other.y):
                                conflict = True
                                break

                        if not conflict:
                            f.x, f.y = bx, by
                            break
        return placed

    def _decode_skyline_bl(self, chrom: Dict, container_size: float
                          ) -> Tuple[List[Facility], float, float]:
        """策略3: Skyline + Bottom-Left混合算法"""
        placed = []
        skyline = [(0, 0, container_size * 2)]

        # 按染色体顺序放置
        for idx in chrom['order']:
            f = self.facilities[idx].copy()
            f.rotated = chrom['rotations'][idx]
            w, h = f.get_dimensions()

            best_pos = self._find_skyline_position(skyline, placed, f, w, h)
            f.x, f.y = best_pos
            placed.append(f)
            self._update_skyline(skyline, f.x, f.y, w, h)

        if not placed:
            return [], 0, 0

        max_x = max(f.x + f.get_dimensions()[0] for f in placed)
        max_y = max(f.y + f.get_dimensions()[1] for f in placed)

        return placed, max_x, max_y

    def _find_skyline_position(self, skyline: List, placed: List[Facility],
                               f: Facility, w: float, h: float) -> Tuple[float, float]:
        """在Skyline中找最佳位置"""
        best_pos = None
        best_score = float('inf')

        # 尝试Skyline段
        for sx, sy, sw in skyline:
            if sw >= w:
                if not self._check_conflict_at(placed, f, sx, sy):
                    score = sy * 100 + sx  # Bottom-Left优先
                    if score < best_score:
                        best_score = score
                        best_pos = (sx, sy)

        # 尝试已放置设施的角落
        for pf in placed:
            pw, ph = pf.get_dimensions()
            gap = get_safety_distance(pf, f)

            for x, y in [(pf.x + pw + gap, pf.y), (pf.x, pf.y + ph + gap),
                         (0, pf.y + ph + gap), (pf.x + pw + gap, 0)]:
                if x >= 0 and y >= 0 and not self._check_conflict_at(placed, f, x, y):
                    score = y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y)

        if best_pos is None:
            max_y = max((p.y + p.get_dimensions()[1] for p in placed), default=0)
            best_pos = (0, max_y + 3)

        return best_pos

    def _update_skyline(self, skyline: List, x: float, y: float, w: float, h: float):
        """更新Skyline"""
        new_top = y + h
        new_skyline = []

        for sx, sy, sw in skyline:
            if sx + sw <= x or sx >= x + w:
                new_skyline.append((sx, sy, sw))
            else:
                if sx < x:
                    new_skyline.append((sx, sy, x - sx))
                if sx + sw > x + w:
                    new_skyline.append((x + w, sy, sx + sw - x - w))

        new_skyline.append((x, new_top, w))
        new_skyline.sort(key=lambda s: s[0])

        # 合并相同高度的相邻段
        merged = []
        for seg in new_skyline:
            if merged and abs(merged[-1][1] - seg[1]) < 0.1 and abs(merged[-1][0] + merged[-1][2] - seg[0]) < 0.1:
                merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + seg[2])
            else:
                merged.append(seg)

        skyline.clear()
        skyline.extend(merged)

    def _find_nearest_valid_pos(self, placed: List[Facility], f: Facility,
                                target_x: float, target_y: float) -> Tuple[float, float]:
        """找到离目标最近的有效位置 - 优化版本"""
        w, h = f.get_dimensions()

        # 首先尝试目标位置
        if target_x >= 0 and target_y >= 0 and not self._check_conflict_at(placed, f, target_x, target_y):
            return (target_x, target_y)

        # 快速搜索：只尝试候选点
        candidates = [(0, 0)]
        for pf in placed:
            pw, ph = pf.get_dimensions()
            gap = get_safety_distance(pf, f)
            candidates.extend([
                (pf.x + pw + gap, pf.y),
                (pf.x, pf.y + ph + gap),
                (0, pf.y + ph + gap),
                (pf.x + pw + gap, pf.y + ph - h) if pf.y + ph > h else None,
            ])

        candidates = [c for c in candidates if c is not None and c[0] >= 0 and c[1] >= 0]

        # 按距离目标排序
        candidates.sort(key=lambda c: abs(c[0] - target_x) + abs(c[1] - target_y))

        for x, y in candidates:
            if not self._check_conflict_at(placed, f, x, y):
                return (x, y)

        max_y = max((p.y + p.get_dimensions()[1] for p in placed), default=0)
        return (0, max_y + 3)

    def _place_remaining_bl(self, chrom: Dict, placed: List[Facility],
                           placed_names: set, name_to_idx: Dict):
        """用Bottom-Left算法放置剩余设施"""
        for idx in chrom['order']:
            name = self.facilities[idx].name
            if name in placed_names:
                continue

            f = self.facilities[idx].copy()
            f.rotated = chrom['rotations'][idx]

            # 找同分区设施，优先靠近
            same_zone = [p for p in placed if p.category == f.category]
            best_pos = self._find_best_compact_pos(placed, f, same_zone)
            f.x, f.y = best_pos
            placed.append(f)
            placed_names.add(name)

    def _pack_zone_bl(self, facilities: List[Facility]
                     ) -> Tuple[List[Facility], float, float]:
        """分区内Bottom-Left紧凑排列"""
        if not facilities:
            return [], 0, 0

        # 按面积降序
        sorted_f = sorted(facilities, key=lambda f: f.area, reverse=True)
        placed = []

        for f in sorted_f:
            if not placed:
                f.x, f.y = 0, 0
            else:
                best_pos = self._find_bl_position(placed, f)
                f.x, f.y = best_pos
            placed.append(f)

        max_x = max(f.x + f.get_dimensions()[0] for f in placed)
        max_y = max(f.y + f.get_dimensions()[1] for f in placed)
        return placed, max_x, max_y

    def _find_bl_position(self, placed: List[Facility], f: Facility) -> Tuple[float, float]:
        """Bottom-Left算法找位置"""
        w, h = f.get_dimensions()
        candidates = [(0, 0)]

        for pf in placed:
            pw, ph = pf.get_dimensions()
            gap = get_safety_distance(pf, f)
            candidates.extend([
                (pf.x + pw + gap, pf.y),
                (pf.x, pf.y + ph + gap),
                (0, pf.y + ph + gap),
            ])

        best_pos = None
        best_score = float('inf')

        for x, y in candidates:
            if x < 0 or y < 0:
                continue
            if not self._check_conflict_at(placed, f, x, y):
                score = y * 1000 + x
                if score < best_score:
                    best_score = score
                    best_pos = (x, y)

        if best_pos is None:
            max_y = max(p.y + p.get_dimensions()[1] for p in placed)
            best_pos = (0, max_y + 3)

        return best_pos

    def _generate_compact_candidates(self, placed: List[Facility], new_f: Facility,
                                     same_zone: List[Facility],
                                     max_candidates: int = 320) -> List[Tuple[float, float]]:
        """生成剩余矩形/Bottom-Left候选点，保留核心排样策略并扩大有效搜索边界。"""
        if not placed:
            return [(0.0, 0.0)]

        candidates = {(0.0, 0.0)}

        # 全局边界点提供更紧凑的剩余矩形角点；过大时优先最近放置和同分区设施。
        anchors = placed if len(placed) <= 80 else placed[-80:]
        zone_anchors = same_zone[-20:] if same_zone else []
        for pf in list(anchors) + zone_anchors:
            pw, ph = pf.get_dimensions()
            gap = get_safety_distance(pf, new_f)
            right = pf.x + pw + gap
            top = pf.y + ph + gap
            candidates.update({
                (right, pf.y),
                (pf.x, top),
                (right, 0.0),
                (0.0, top),
                (right, top),
            })

        # 交叉投影：用已放置矩形的右边界/上边界组合生成剩余空间角点。
        x_frontiers = {0.0}
        y_frontiers = {0.0}
        for pf in anchors:
            pw, ph = pf.get_dimensions()
            gap = get_safety_distance(pf, new_f)
            x_frontiers.update({pf.x, pf.x + pw + gap})
            y_frontiers.update({pf.y, pf.y + ph + gap})

        xs = sorted(x for x in x_frontiers if x >= 0)[:45]
        ys = sorted(y for y in y_frontiers if y >= 0)[:45]
        current_limit_x = max(p.x + p.get_dimensions()[0] for p in placed) + new_f.get_dimensions()[0] + 10
        current_limit_y = max(p.y + p.get_dimensions()[1] for p in placed) + new_f.get_dimensions()[1] + 10
        for x in xs:
            for y in ys:
                if x <= current_limit_x and y <= current_limit_y:
                    candidates.add((float(x), float(y)))

        filtered = [(x, y) for x, y in candidates if x >= 0 and y >= 0]
        filtered.sort(key=lambda p: (p[1], p[0]))
        return filtered[:max_candidates]

    def _find_best_compact_pos(self, placed: List[Facility], new_f: Facility,
                               same_zone: List[Facility]) -> Tuple[float, float]:
        """找最佳紧凑位置 - GPU加速版本"""
        w, h = new_f.get_dimensions()

        if not placed:
            return (0, 0)

        candidates = self._generate_compact_candidates(placed, new_f, same_zone)

        if not candidates:
            max_y = max(f.y + f.get_dimensions()[1] for f in placed)
            return (0, max_y + 3)

        # 转换为 NumPy 数组
        candidates_x = np.array([c[0] for c in candidates], dtype=np.float64)
        candidates_y = np.array([c[1] for c in candidates], dtype=np.float64)

        placed_x = np.array([f.x for f in placed], dtype=np.float64)
        placed_y = np.array([f.y for f in placed], dtype=np.float64)
        placed_w = np.array([f.get_dimensions()[0] for f in placed], dtype=np.float64)
        placed_h = np.array([f.get_dimensions()[1] for f in placed], dtype=np.float64)
        gaps = np.array([get_safety_distance(pf, new_f) for pf in placed], dtype=np.float64)

        # 计算边界
        cur_max_x = max(placed_x[i] + placed_w[i] for i in range(len(placed)))
        cur_max_y = max(placed_y[i] + placed_h[i] for i in range(len(placed)))

        # 智能选择：只有数据量足够大时才用GPU
        use_gpu = (GPU_AVAILABLE and
                   len(candidates) >= GPU_THRESHOLD and
                   len(placed) >= GPU_PLACED_THRESHOLD)

        if use_gpu:
            # GPU 并行检查冲突
            valid_mask = check_conflicts_gpu(
                candidates_x, candidates_y, w, h,
                placed_x, placed_y, placed_w, placed_h, gaps
            )
            if valid_mask is not None:
                # GPU 并行评估得分
                scores = evaluate_positions_gpu(
                    candidates_x, candidates_y, w, h,
                    cur_max_x, cur_max_y, valid_mask
                )
                if scores is not None:
                    best_idx = np.argmin(scores)
                    if scores[best_idx] < np.inf:
                        return (candidates_x[best_idx], candidates_y[best_idx])

        # CPU向量化处理（小数据量或GPU失败时）
        cx_2d = candidates_x[:, np.newaxis]
        cy_2d = candidates_y[:, np.newaxis]
        no_overlap = (
            (cx_2d >= placed_x + placed_w + gaps) |
            (cx_2d + w + gaps <= placed_x) |
            (cy_2d >= placed_y + placed_h + gaps) |
            (cy_2d + h + gaps <= placed_y)
        )
        valid_mask = np.all(no_overlap, axis=1)

        if np.any(valid_mask):
            new_max_x = np.maximum(cur_max_x, candidates_x + w)
            new_max_y = np.maximum(cur_max_y, candidates_y + h)
            scores = new_max_x * new_max_y
            aspect = np.maximum(new_max_x, new_max_y) / np.maximum(np.minimum(new_max_x, new_max_y), 1.0)
            scores += np.where(aspect > 2.0, (aspect - 2.0) * 300.0, 0.0)
            scores += candidates_y * 0.20 + candidates_x * 0.05

            if same_zone:
                zone_centers = np.array([
                    [sf.x + sf.get_dimensions()[0] / 2, sf.y + sf.get_dimensions()[1] / 2]
                    for sf in same_zone
                ], dtype=np.float64)
                cand_centers = np.column_stack((candidates_x + w / 2, candidates_y + h / 2))
                zone_dist = np.abs(cand_centers[:, None, :] - zone_centers[None, :, :]).sum(axis=2).min(axis=1)
                scores += zone_dist * 0.08

            scores = np.where(valid_mask, scores, np.inf)
            best_idx = int(np.argmin(scores))
            if scores[best_idx] < np.inf:
                return (float(candidates_x[best_idx]), float(candidates_y[best_idx]))

        return (0, cur_max_y + 3)

    def _check_conflict_at(self, placed: List[Facility], new_f: Facility,
                           x: float, y: float) -> bool:
        """检查在指定位置是否有冲突"""
        w, h = new_f.get_dimensions()

        # 检查设施间冲突
        for pf in placed:
            pw, ph = pf.get_dimensions()
            gap = get_safety_distance(pf, new_f)

            if not (x >= pf.x + pw + gap or x + w + gap <= pf.x or
                    y >= pf.y + ph + gap or y + h + gap <= pf.y):
                return True

        return False

    def _point_dist(self, x: float, y: float, f: Facility) -> float:
        """点到设施中心的距离"""
        fw, fh = f.get_dimensions()
        cx, cy = f.x + fw/2, f.y + fh/2
        return abs(x - cx) + abs(y - cy)

    def calculate_fitness(self, chrom: Dict) -> float:
        """
        工艺流程驱动适应度函数 V4.1 - 优化版
        核心目标:
        1. 工艺流程严格相邻（最高权重）
        2. 面积利用率 > 65%
        3. 辅助设施就近
        4. 分区紧凑整齐
        """
        chrom = self._repair_chromosome(chrom)
        cache_key = None
        if self.enable_cache:
            cache_key = self._chromosome_key(chrom)
            with self._fitness_cache_lock:
                if cache_key in self._fitness_cache:
                    return self._fitness_cache[cache_key]

        base_size = np.sqrt(self.total_area) * 1.5

        placed, bbox_w, bbox_h = self.decode_chromosome(chrom, base_size)

        if len(placed) < self.n:
            fitness = float('inf')
            if self.enable_cache and cache_key is not None:
                with self._fitness_cache_lock:
                    self._fitness_cache[cache_key] = fitness
            return fitness

        # ========== 核心指标 ==========
        area = bbox_w * bbox_h
        util = self.total_area / area if area > 0 else 0

        # 1. 工艺流程相邻性 (最高权重 - 必须严格相邻)
        process_penalty = self._calc_process_adjacency_strict(placed)

        # 2. 面积惩罚（强化权重 - 核心目标）
        # 使用非线性惩罚：面积越大惩罚增长越快
        area_penalty = area * 6.5 + (area / 1000) ** 1.5 * 100

        # 3. 利用率奖励（大幅增强 - 追求紧凑，目标70%+）
        util_bonus = -util * 60000
        if util > 0.72:
            util_bonus -= 40000  # 极高利用率特大奖励
        elif util > 0.70:
            util_bonus -= 32000
        elif util > 0.68:
            util_bonus -= 25000
        elif util > 0.65:
            util_bonus -= 18000
        elif util > 0.62:
            util_bonus -= 12000
        elif util > 0.58:
            util_bonus -= 6000
        elif util > 0.55:
            util_bonus -= 3000
        elif util < 0.50:
            util_bonus += 10000  # 低利用率惩罚

        # 4. 长宽比惩罚 (适度)
        if min(bbox_w, bbox_h) > 0:
            aspect = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
            # 理想比例 2:1 到 3:1
            if aspect < 2.0:
                aspect_penalty = (2.0 - aspect) * 50  # 太方正也不好
            elif aspect > 3.0:
                aspect_penalty = (aspect - 3.0) * 120  # 太细长不好
            else:
                aspect_penalty = 0
        else:
            aspect_penalty = 0

        # 5. 管线总长度（提高权重）
        pipe_penalty = self._calc_pipeline_length(placed) * 0.7

        # 6. 分区紧凑性（提高权重）
        zone_compact = self._calc_zone_compactness(placed) * 1.2

        # 7. 宿舍联合布置
        dorm_penalty = self._calc_dormitory_cluster_penalty(placed) * 0.7

        # 8. 功能分区边界清晰度
        zone_boundary = self._calc_zone_boundary_clarity(placed) * 0.5

        # 9. 设施对齐度（提升整齐度）
        alignment = self._calc_alignment_bonus(placed) * 0.4

        # 10. 边界紧密度奖励（新增：奖励设施紧贴边界）
        boundary_bonus = self._calc_boundary_tightness(placed, bbox_w, bbox_h)

        # 11. 竖向高程约束惩罚 V2.0（加权幂次 + 盈余奖励）
        # _calc_elevation_constraint 已返回带权等效分，直接放大到与其他项同量级
        # 典型违规0.1m: ~0.1^1.2*2.5*400≈70分；违规1.0m: ~1.0^1.2*2.5*400=1120分
        elev_violation = self._calc_elevation_constraint(placed)
        elevation_penalty = elev_violation * 400.0

        fitness = (process_penalty + area_penalty + util_bonus +
                  aspect_penalty + pipe_penalty + zone_compact +
                  dorm_penalty + zone_boundary + alignment + boundary_bonus +
                  elevation_penalty)

        if self.enable_cache and cache_key is not None:
            with self._fitness_cache_lock:
                if len(self._fitness_cache) >= self._fitness_cache_max_size:
                    self._fitness_cache.clear()
                self._fitness_cache[cache_key] = fitness

        return fitness

    def _calc_process_adjacency_strict(self, placed: List[Facility]) -> float:
        """
        严格工艺流程相邻性计算 V2.0
        主工艺流程必须严格相邻（距离<5m），否则重罚
        同时考虑管线连接的实际路径
        """
        fdict = {f.name: f for f in placed}
        penalty = 0

        # 第一级：主工艺流程（必须严格相邻，最高优先级）
        critical_pairs = [
            ("IPS", "GC"), ("GC", "PS"), ("PS", "MB"), ("MB", "FB"),
            ("FB", "SB"), ("SB", "FT"), ("FT", "CWT"), ("CWT", "PH2"),  # 主流程
            ("FT", "OZG"), ("OZG", "ACF"), ("ACF", "CWT"),  # 深度处理分支
        ]

        for n1, n2 in critical_pairs:
            if n1 in fdict and n2 in fdict:
                f1, f2 = fdict[n1], fdict[n2]
                dist = self._calc_facility_distance(f1, f2)

                if dist <= 1:
                    penalty -= 2000  # 紧贴相邻超级奖励
                elif dist <= 2:
                    penalty -= 1600  # 紧密相邻大奖励
                elif dist <= 4:
                    penalty -= 1000  # 接近相邻奖励
                elif dist <= 8:
                    penalty -= 400   # 较近奖励
                elif dist <= 12:
                    penalty += dist * 100  # 中等惩罚
                else:
                    penalty += dist * 350  # 严重惩罚（加大）

        # 第二级：重要工艺配套设施（应该相邻）
        important_pairs = [
            ("CDR", "MB"), ("CDR", "FB"), ("CLS", "CWT"), ("OZR", "OZG"),  # 投加设施
            ("IPS", "PH1"), ("PH2", "PDR"), ("BWT", "FT"), ("BWT", "CWT"),  # 泵站配套
            ("SB", "STK"), ("STK", "SDR"), ("SDR", "SYD"),  # 污泥处理链
            ("TRF", "PDR"),  # 电力设施
        ]

        for n1, n2 in important_pairs:
            if n1 in fdict and n2 in fdict:
                f1, f2 = fdict[n1], fdict[n2]
                dist = self._calc_facility_distance(f1, f2)

                if dist <= 5:
                    penalty -= 400
                elif dist <= 10:
                    penalty -= 100
                elif dist > 25:
                    penalty += dist * 40

        # 第三级：一般设施相邻（可容忍一定距离）
        general_pairs = [
            ("CST", "CDR"), ("CR", "FT"), ("CR", "PDR"), ("LAB", "CR"),
            ("DM1", "DM2"), ("DM2", "DM3"), ("DM3", "DM4"),
            ("DM1", "SPF"), ("OF", "CR"), ("GT", "OF"),
        ]

        for n1, n2 in general_pairs:
            if n1 in fdict and n2 in fdict:
                f1, f2 = fdict[n1], fdict[n2]
                dist = self._calc_facility_distance(f1, f2)

                if dist <= 5:
                    penalty -= 150
                elif dist > 40:
                    penalty += dist * 15

        return penalty

    def _calc_facility_distance(self, f1: Facility, f2: Facility) -> float:
        """计算两个设施之间的最小距离"""
        w1, h1 = f1.get_dimensions()
        w2, h2 = f2.get_dimensions()

        # 计算边界距离
        dx = max(0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
        dy = max(0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))

        return np.sqrt(dx**2 + dy**2)

    def _calc_gate_position_penalty(self, placed: List[Facility], bbox_w: float, bbox_h: float) -> float:
        """门卫室位置惩罚 - 必须在入口边界处，不在主干道中间"""
        penalty = 0
        for f in placed:
            if f.name == "GT":
                w, h = f.get_dimensions()
                # 必须在左边界或下边界
                on_left = f.x < 2
                on_bottom = f.y < 2
                if not (on_left or on_bottom):
                    # 离边界的距离作为惩罚
                    dist_to_edge = min(f.x, f.y)
                    penalty += dist_to_edge * 50
        return penalty

    def _calc_boundary_constraint(self, placed: List[Facility], bbox_w: float, bbox_h: float) -> float:
        """计算边界约束惩罚"""
        penalty = 0
        for f in placed:
            if f.name == "GT":  # 门卫室
                w, h = f.get_dimensions()

                # 主干道与长边平行，门卫室在短边入口处
                if bbox_w >= bbox_h:
                    # 长边是X方向，主干道水平，门卫室在左边界(x=0)，y方向居中
                    if f.x > 2:  # 不在左侧边界
                        penalty += f.x * 50  # 严格惩罚

                    # 门卫室必须在左边界的y方向中点位置
                    center_y = bbox_h / 2
                    dist_to_center = abs(f.y + h/2 - center_y)
                    penalty += dist_to_center * 15  # 强制居中
                else:
                    # 长边是Y方向，主干道垂直，门卫室在底边界(y=0)，x方向居中
                    if f.y > 2:  # 不在底部边界
                        penalty += f.y * 50

                    center_x = bbox_w / 2
                    dist_to_center = abs(f.x + w/2 - center_x)
                    penalty += dist_to_center * 15

        return penalty

    def _calc_pipeline_length(self, placed: List[Facility]) -> float:
        """计算所有管线的加权总长度（用于优化）。"""
        _, summary = calculate_pipeline_network(placed)
        return summary["weighted_length_m"]

    # ==================== 竖向高程约束方法 ====================
    # ===== 公共辅助：计算两设施中心间的曼哈顿距离 =====
    @staticmethod
    def _segment_dist(fdict: dict, n1: str, n2: str, system_name: str = None) -> float:
        """计算两设施之间的管线路由长度估算。"""
        if n1 not in fdict or n2 not in fdict:
            return 0.0
        system_name = system_name or infer_pipeline_system(n1, n2)
        return calculate_pipeline_segment(fdict[n1], fdict[n2], system_name)["routed_length_m"]

    def _calc_elevation_constraint(self, placed: List[Facility]) -> float:
        """
        竖向高程约束计算 V2.0
        -----------------------------------------------
        改进项:
          1. 分段精细水力参数（清水管 vs 污泥管坡度不同）
          2. 修正 SB→STK 使用排泥口高程而非水侧出水位
          3. 加权幂次惩罚 weight * viol^1.2（大违规更重惩）
          4. 盈余水头奖励（引导算法追求稳健高程设计）
          5. 临界段额外惩罚（权重≥2.5且违规>0.5m时触发）
        返回: 加权等效惩罚分（越小越好，0 = 全段满足）
        """
        fdict = {f.name: f for f in placed}
        total_score = 0.0
        all_paths = MAIN_WATER_ELEV_PATH + SLUDGE_ELEV_PATH

        for (n_up, n_down) in all_paths:
            if n_up not in ELEVATION_DATA or n_down not in ELEVATION_DATA:
                continue
            if n_up not in fdict or n_down not in fdict:
                continue

            ed_up  = ELEVATION_DATA[n_up]
            ed_down = ELEVATION_DATA[n_down]

            # --- 获取本段精细水力参数（无则用全局默认）---
            seg_key = (n_up, n_down)
            if seg_key in ELEV_SEGMENT_PARAMS:
                i_coef, hm, hs, seg_w = ELEV_SEGMENT_PARAMS[seg_key]
            else:
                i_coef, hm, hs, seg_w = PIPE_HEADLOSS_PER_METER, PIPE_LOCAL_LOSS, GRAVITY_FLOW_SAFETY, 1.0

            # --- 管线水平距离 ---
            system_name = infer_pipeline_system(n_up, n_down)
            L = self._segment_dist(fdict, n_up, n_down, system_name)

            # --- 上游出水位（污泥特殊处理：SB用排泥口，非水侧出水位）---
            if n_up == "SB" and n_down == "STK":
                water_out_up = SB_SLUDGE_OUTLET_ELEV   # 底部排泥口 99.5m
            else:
                water_out_up = ed_up["water_out"]

            # --- 所需水头: 沿程 + 局部 + 安全余量 ---
            h_req = i_coef * L + hm + hs

            # --- 下游进水位 = 地面高程 + 进水超高（依构筑物类型取值）---
            # 泵站进水可降至地面以下（吸水管）；重力流构筑物留 0.30m 超高
            if ed_down["type"] == "pump":
                water_in_down = ed_down["ground"] - 0.50   # 泵坑液位
            else:
                water_in_down = ed_down["ground"] + 0.30

            # --- 可用水头 ---
            avail = water_out_up - water_in_down

            violation = h_req - avail   # >0 违规，<0 盈余

            if violation > 0:
                # 幂次惩罚（1.2次方）：对小违规宽松，大违规严厉
                total_score += seg_w * (violation ** 1.2)
                # 临界段额外惩罚（权重≥2.5，违规超过0.5m）
                if seg_w >= 2.5 and violation > 0.5:
                    total_score += seg_w * (violation - 0.5) * 2.0
            else:
                # 盈余水头奖励（上限2.0m，避免过度奖励降低约束强度）
                surplus = min(-violation, 2.0)
                total_score -= seg_w * surplus * 0.008

        return total_score

    def _get_elevation_report(self, placed: List[Facility]) -> dict:
        """
        高程衔接检查报告 V2.0（与 _calc_elevation_constraint V2.0 保持一致）
        返回: {(up, down): {water_out_up, water_in_down, pipe_L, i_coef,
                            h_req, available, surplus, seg_weight, feasible}}
        """
        fdict = {f.name: f for f in placed}
        report = {}
        all_paths = MAIN_WATER_ELEV_PATH + SLUDGE_ELEV_PATH

        for (n_up, n_down) in all_paths:
            if n_up not in ELEVATION_DATA or n_down not in ELEVATION_DATA:
                continue

            ed_up   = ELEVATION_DATA[n_up]
            ed_down = ELEVATION_DATA[n_down]

            seg_key = (n_up, n_down)
            if seg_key in ELEV_SEGMENT_PARAMS:
                i_coef, hm, hs, seg_w = ELEV_SEGMENT_PARAMS[seg_key]
            else:
                i_coef, hm, hs, seg_w = PIPE_HEADLOSS_PER_METER, PIPE_LOCAL_LOSS, GRAVITY_FLOW_SAFETY, 1.0

            system_name = infer_pipeline_system(n_up, n_down)
            L = self._segment_dist(fdict, n_up, n_down, system_name)
            h_req = i_coef * L + hm + hs

            if n_up == "SB" and n_down == "STK":
                water_out_up = SB_SLUDGE_OUTLET_ELEV
            else:
                water_out_up = ed_up["water_out"]

            if ed_down["type"] == "pump":
                water_in_down = ed_down["ground"] - 0.50
            else:
                water_in_down = ed_down["ground"] + 0.30

            avail    = water_out_up - water_in_down
            shortage = max(0.0, h_req - avail)
            surplus  = max(0.0, avail - h_req)

            report[(n_up, n_down)] = {
                "system":          system_name,
                "water_out_up":   round(water_out_up, 3),
                "water_in_down":  round(water_in_down, 3),
                "pipe_L":         round(L, 1),
                "i_coef":         i_coef,
                "h_req":          round(h_req, 3),
                "available_head": round(avail, 3),
                "surplus_m":      round(surplus, 3),
                "shortage_m":     round(shortage, 3),
                "seg_weight":     seg_w,
                "feasible":       shortage == 0.0,
            }
        return report

    def _calc_zone_compactness(self, placed: List[Facility]) -> float:
        """计算分区紧凑性奖励 - 增强版"""
        zones = {}
        for f in placed:
            if f.category not in zones:
                zones[f.category] = []
            zones[f.category].append(f)

        bonus = 0

        # 关键分区权重（处理区更重要）
        zone_weights = {
            "intake": 1.5,
            "pretreat": 1.5,
            "conventional": 2.0,
            "advanced": 2.0,
            "chemical": 1.5,
            "distribution": 1.8,
            "sludge": 1.2,
            "power": 1.5,
            "admin": 1.0,
            "auxiliary": 0.8,
            "living": 1.3,
        }

        for cat, flist in zones.items():
            if len(flist) < 2:
                continue

            weight = zone_weights.get(cat, 1.0)

            # 计算分区内设施之间的平均距离
            total_dist = 0
            count = 0
            for i, f1 in enumerate(flist):
                for f2 in flist[i+1:]:
                    total_dist += self._min_distance(f1, f2)
                    count += 1

            if count > 0:
                avg_dist = total_dist / count
                # 平均距离越小，奖励越多
                if avg_dist <= 5:
                    bonus -= 15 * weight * len(flist)
                elif avg_dist <= 10:
                    bonus -= 8 * weight * len(flist)
                elif avg_dist > 25:
                    bonus += avg_dist * 3 * weight

        return bonus

    def _calc_adjacency_penalty_strict(self, placed: List[Facility]) -> Tuple[float, float]:
        """严格的工艺流程相邻性惩罚"""
        fdict = {f.name: f for f in placed}
        penalty = 0
        critical_violations = 0

        # 主工艺流程中的关键连接（必须严格相邻）
        critical_pairs = [
            ("IPS", "GC"), ("GC", "PS"), ("PS", "MB"), ("MB", "FB"),
            ("FB", "SB"), ("SB", "FT"), ("FT", "CWT"), ("CWT", "PH2"),
            ("FT", "OZG"), ("OZG", "ACF"), ("ACF", "CWT"), ("STK", "SDR")
        ]

        for n1, n2, weight in ADJACENCY_REQUIRED:
            if n1 in fdict and n2 in fdict:
                f1, f2 = fdict[n1], fdict[n2]
                dist = self._min_distance(f1, f2)

                # 检查是否是关键工艺对
                is_critical = (n1, n2) in critical_pairs or (n2, n1) in critical_pairs

                if is_critical:
                    # 关键工艺必须相邻（距离<8m）
                    if dist > 8:
                        critical_violations += (dist - 8) * weight
                    elif dist > 3:
                        penalty += (dist - 3) * weight * 2
                else:
                    # 普通工艺约束
                    if dist > 15:
                        penalty += dist * weight
                    elif dist > 5:
                        penalty += (dist - 5) * weight * 0.5

        return penalty, critical_violations

    def _calc_adjacency_penalty(self, placed: List[Facility]) -> float:
        """相邻性惩罚（兼容旧接口）"""
        penalty, _ = self._calc_adjacency_penalty_strict(placed)
        return penalty

    def _calc_safety_penalty(self, placed: List[Facility]) -> float:
        """安全距离违规惩罚"""
        penalty = 0
        for i, f1 in enumerate(placed):
            for f2 in placed[i+1:]:
                required_gap = get_safety_distance(f1, f2)
                actual_dist = self._min_distance(f1, f2)
                if actual_dist < required_gap:
                    penalty += (required_gap - actual_dist) * 10
        return penalty

    def _min_distance(self, f1: Facility, f2: Facility) -> float:
        """最小距离"""
        w1, h1 = f1.get_dimensions()
        w2, h2 = f2.get_dimensions()
        dx = max(0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
        dy = max(0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))
        return np.sqrt(dx**2 + dy**2)

    def _calc_dormitory_cluster_penalty(self, placed: List[Facility]) -> float:
        """计算宿舍联合布置惩罚 - 4个宿舍必须紧密相邻形成宿舍区"""
        fdict = {f.name: f for f in placed}
        dorm_names = ["DM1", "DM2", "DM3", "DM4"]
        dorms = [fdict[n] for n in dorm_names if n in fdict]

        if len(dorms) < 2:
            return 0

        penalty = 0

        # 1. 宿舍之间必须紧密相邻（距离<5m）
        for i, d1 in enumerate(dorms):
            for d2 in dorms[i+1:]:
                dist = self._min_distance(d1, d2)
                if dist > 5:
                    # 距离超过5m则大惩罚
                    penalty += (dist - 5) * 15
                elif dist > 2:
                    # 轻微惩罚
                    penalty += (dist - 2) * 5

        # 2. 检查宿舍是否形成矩形集群（2x2布局最优）
        if len(dorms) == 4:
            # 计算宿舍区域包围盒
            min_x = min(d.x for d in dorms)
            max_x = max(d.x + d.get_dimensions()[0] for d in dorms)
            min_y = min(d.y for d in dorms)
            max_y = max(d.y + d.get_dimensions()[1] for d in dorms)

            cluster_w = max_x - min_x
            cluster_h = max_y - min_y
            total_dorm_area = sum(d.area for d in dorms)
            cluster_area = cluster_w * cluster_h

            # 集群紧凑度：面积利用率应>70%
            if cluster_area > 0:
                compactness = total_dorm_area / cluster_area
                if compactness < 0.7:
                    penalty += (0.7 - compactness) * 500

        # 3. 宿舍区应靠近运动场
        if "SPF" in fdict:
            spf = fdict["SPF"]
            dorm_center_x = sum(d.x + d.get_dimensions()[0]/2 for d in dorms) / len(dorms)
            dorm_center_y = sum(d.y + d.get_dimensions()[1]/2 for d in dorms) / len(dorms)
            spf_center_x = spf.x + spf.get_dimensions()[0]/2
            spf_center_y = spf.y + spf.get_dimensions()[1]/2

            dist_to_spf = np.sqrt((dorm_center_x - spf_center_x)**2 + (dorm_center_y - spf_center_y)**2)
            if dist_to_spf > 40:  # 宿舍区中心到运动场中心超过40m
                penalty += (dist_to_spf - 40) * 3

        return penalty

    def _calc_zone_boundary_clarity(self, placed: List[Facility]) -> float:
        """计算功能分区边界清晰度 - 同分区设施应集中，不同分区应有清晰边界"""
        zones = {}
        for f in placed:
            if f.category not in zones:
                zones[f.category] = []
            zones[f.category].append(f)

        penalty = 0

        for cat, flist in zones.items():
            if len(flist) < 2:
                continue

            # 计算分区包围盒
            min_x = min(f.x for f in flist)
            max_x = max(f.x + f.get_dimensions()[0] for f in flist)
            min_y = min(f.y for f in flist)
            max_y = max(f.y + f.get_dimensions()[1] for f in flist)

            zone_area = (max_x - min_x) * (max_y - min_y)
            facilities_area = sum(f.area for f in flist)

            # 分区填充率
            if zone_area > 0:
                fill_rate = facilities_area / zone_area
                # 填充率低于40%说明分区过于分散
                if fill_rate < 0.4:
                    penalty += (0.4 - fill_rate) * 200
                # 填充率高于50%给予奖励
                elif fill_rate > 0.5:
                    penalty -= (fill_rate - 0.5) * 100

        return penalty

    def _calc_alignment_bonus(self, placed: List[Facility]) -> float:
        """计算设施对齐度奖励 - 鼓励设施水平或垂直对齐"""
        bonus = 0

        # 按Y坐标分组（水平对齐）
        y_groups = {}
        for f in placed:
            # 四舍五入到5m精度
            y_key = round(f.y / 5) * 5
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(f)

        # 同一行有多个设施则奖励
        for y, flist in y_groups.items():
            if len(flist) >= 2:
                bonus -= len(flist) * 5  # 每对齐一个设施奖励5分

        # 按X坐标分组（垂直对齐）
        x_groups = {}
        for f in placed:
            x_key = round(f.x / 5) * 5
            if x_key not in x_groups:
                x_groups[x_key] = []
            x_groups[x_key].append(f)

        for x, flist in x_groups.items():
            if len(flist) >= 2:
                bonus -= len(flist) * 5

        return bonus

    def _calc_boundary_tightness(self, placed: List[Facility], bbox_w: float, bbox_h: float) -> float:
        """
        计算边界紧密度奖励 - 鼓励设施紧贴布局边界
        设施越靠近边界（0, 0）或填满边界区域，奖励越多
        """
        bonus = 0

        if not placed:
            return 0

        # 1. 原点吸引力：设施越靠近(0,0)越好
        for f in placed:
            dist_to_origin = np.sqrt(f.x**2 + f.y**2)
            # 距离原点每10m扣1分
            bonus += dist_to_origin * 0.1

        # 2. 边界填充度：检查布局边界是否被充分利用
        # 将布局分成网格，统计每个网格的填充情况
        grid_size = 20  # 20m网格
        n_cols = max(1, int(bbox_w / grid_size))
        n_rows = max(1, int(bbox_h / grid_size))

        grid_filled = [[False] * n_cols for _ in range(n_rows)]

        for f in placed:
            fw, fh = f.get_dimensions()
            # 计算设施覆盖的网格
            col_start = int(f.x / grid_size)
            col_end = min(n_cols, int((f.x + fw) / grid_size) + 1)
            row_start = int(f.y / grid_size)
            row_end = min(n_rows, int((f.y + fh) / grid_size) + 1)

            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    if 0 <= r < n_rows and 0 <= c < n_cols:
                        grid_filled[r][c] = True

        # 统计填充的网格数
        filled_count = sum(sum(row) for row in grid_filled)
        total_grids = n_rows * n_cols
        fill_ratio = filled_count / total_grids if total_grids > 0 else 0

        # 填充率越高奖励越多
        if fill_ratio > 0.8:
            bonus -= 500  # 填充率>80%大奖励
        elif fill_ratio > 0.6:
            bonus -= 200
        elif fill_ratio < 0.4:
            bonus += 300  # 填充率过低惩罚

        # 3. 边界边缘利用：设施是否紧贴左边和下边
        left_edge_count = sum(1 for f in placed if f.x < 3)
        bottom_edge_count = sum(1 for f in placed if f.y < 3)
        bonus -= (left_edge_count + bottom_edge_count) * 10

        return bonus

    def _calc_zone_side_constraint(self, placed: List[Facility], bbox_w: float, bbox_h: float) -> float:
        """
        计算分区主干道两侧约束惩罚
        污泥区(sludge)和生活区(living)必须分别放置在主干道两侧
        - 污泥区：放在主干道上方（远离入口）
        - 生活区：放在主干道下方（靠近入口）
        """
        penalty = 0

        # 确定主干道中心位置
        is_road_horizontal = (bbox_w >= bbox_h)
        road_center = bbox_h / 2 if is_road_horizontal else bbox_w / 2

        for f in placed:
            w, h = f.get_dimensions()
            f_center = (f.y + h / 2) if is_road_horizontal else (f.x + w / 2)

            if f.category == "sludge":
                # 污泥区应该在主干道上方（远离入口端）
                if is_road_horizontal:
                    # 水平主干道，污泥区应在上方（y > road_center）
                    if f_center < road_center:
                        penalty += (road_center - f_center) * 8
                else:
                    # 垂直主干道，污泥区应在右方（x > road_center）
                    if f_center < road_center:
                        penalty += (road_center - f_center) * 8

            elif f.category == "living":
                # 生活区应该在主干道下方（靠近入口端）
                if is_road_horizontal:
                    # 水平主干道，生活区应在下方（y < road_center）
                    if f_center > road_center:
                        penalty += (f_center - road_center) * 8
                else:
                    # 垂直主干道，生活区应在左方（x < road_center）
                    if f_center > road_center:
                        penalty += (f_center - road_center) * 8

        return penalty

    def selection(self, pop: List, fitness: List) -> Dict:
        """锦标赛选择"""
        tournament_size = min(5, len(pop))
        idx = random.sample(range(len(pop)), tournament_size)
        best = min(idx, key=lambda i: fitness[i])
        return copy.deepcopy(pop[best])

    def crossover(self, p1: Dict, p2: Dict) -> Tuple[Dict, Dict]:
        """交叉 - 安全版本"""
        p1 = self._repair_chromosome(p1)
        p2 = self._repair_chromosome(p2)
        size = self.n

        # 验证父代长度，如果不对则直接返回深拷贝
        if len(p1['order']) != size or len(p2['order']) != size:
            return self._repair_chromosome(p1), self._repair_chromosome(p2)

        # 以一定概率不交叉
        if random.random() > 0.8:
            return self._repair_chromosome(p1), self._repair_chromosome(p2)

        # 使用位置交叉(Position-based)
        def position_crossover(pa, pb):
            # 选择一些随机位置
            num_pos = max(1, size // 3)
            positions = random.sample(range(size), num_pos)
            child = [-1] * size
            used = set()

            # 从pa复制选中位置的元素
            for pos in positions:
                if pa[pos] not in used:
                    child[pos] = pa[pos]
                    used.add(pa[pos])

            # 从pb填充剩余位置
            pb_remaining = [x for x in pb if x not in used]
            idx = 0
            for i in range(size):
                if child[i] == -1:
                    if idx < len(pb_remaining):
                        child[i] = pb_remaining[idx]
                        idx += 1

            # 最终验证：确保没有-1和长度正确
            used_final = set(x for x in child if x != -1)
            missing = [x for x in range(size) if x not in used_final]
            for i in range(size):
                if child[i] == -1:
                    if missing:
                        child[i] = missing.pop()
                    else:
                        child[i] = i  # 紧急修复

            return child

        c1 = copy.deepcopy(p1)
        c2 = copy.deepcopy(p2)

        c1['order'] = position_crossover(p1['order'], p2['order'])
        c2['order'] = position_crossover(p2['order'], p1['order'])

        # 验证输出长度
        assert len(c1['order']) == size, f"c1 order length mismatch: {len(c1['order'])} != {size}"
        assert len(c2['order']) == size, f"c2 order length mismatch: {len(c2['order'])} != {size}"

        # 旋转交叉
        for i in range(size):
            if random.random() < 0.5:
                c1['rotations'][i], c2['rotations'][i] = c2['rotations'][i], c1['rotations'][i]

        # 分区顺序交叉
        if random.random() < 0.5:
            c1['zone_order'], c2['zone_order'] = c2['zone_order'], c1['zone_order']

        # 策略交叉
        if random.random() < 0.3:
            c1['strategy'], c2['strategy'] = c2.get('strategy', 0), c1.get('strategy', 0)

        return self._repair_chromosome(c1), self._repair_chromosome(c2)

    def mutate(self, chrom: Dict) -> Dict:
        """变异"""
        c = self._repair_chromosome(chrom)

        # 验证并修复染色体
        if len(c['order']) != self.n or len(set(c['order'])) != self.n:
            c['order'] = list(range(self.n))
            random.shuffle(c['order'])

        if len(c['rotations']) != self.n:
            c['rotations'] = [random.choice([True, False]) for _ in range(self.n)]

        # 顺序变异
        if random.random() < self.mutation_rate:
            # 多种变异方式
            mut_type = random.choice(['swap', 'insert', 'reverse'])
            if mut_type == 'swap':
                i, j = random.sample(range(self.n), 2)
                c['order'][i], c['order'][j] = c['order'][j], c['order'][i]
            elif mut_type == 'insert':
                i = random.randint(0, self.n - 1)
                j = random.randint(0, self.n - 1)
                item = c['order'].pop(i)
                c['order'].insert(j, item)
            else:  # reverse
                i, j = sorted(random.sample(range(self.n), 2))
                c['order'][i:j+1] = reversed(c['order'][i:j+1])

        # 旋转变异
        for i in range(self.n):
            if random.random() < 0.1:
                c['rotations'][i] = not c['rotations'][i]

        # 分区顺序变异
        if random.random() < 0.15:
            random.shuffle(c['zone_order'])

        # 策略变异
        if random.random() < 0.1:
            c['strategy'] = random.randint(0, 2)

        return self._repair_chromosome(c)

    def _knowledge_guided_mutate(self, chrom: Dict, knowledge_base: Dict) -> Dict:
        """基于知识库的智能变异"""
        c = self._repair_chromosome(chrom)

        # 验证并修复染色体
        if len(c['order']) != self.n or len(set(c['order'])) != self.n:
            c['order'] = list(range(self.n))
            random.shuffle(c['order'])

        # 从知识库学习：部分保留最优顺序
        if knowledge_base['best_orders'] and random.random() < 0.4:
            best_order = random.choice(knowledge_base['best_orders'])
            # 保留30-60%的最优顺序特征
            preserve_ratio = random.uniform(0.3, 0.6)
            preserve_count = int(self.n * preserve_ratio)
            preserve_positions = random.sample(range(self.n), preserve_count)

            new_order = [-1] * self.n
            used = set()

            # 复制保留位置
            for pos in preserve_positions:
                if pos < len(best_order):
                    val = best_order[pos]
                    if val not in used and val < self.n:
                        new_order[pos] = val
                        used.add(val)

            # 填充剩余位置（从当前染色体）
            remaining = [x for x in c['order'] if x not in used]
            random.shuffle(remaining)

            idx = 0
            for i in range(self.n):
                if new_order[i] == -1:
                    if idx < len(remaining):
                        new_order[i] = remaining[idx]
                        idx += 1

            # 确保有效
            if -1 not in new_order and len(set(new_order)) == self.n:
                c['order'] = new_order

        # 从知识库学习旋转模式
        if knowledge_base['best_rotations'] and random.random() < 0.3:
            best_rot = random.choice(knowledge_base['best_rotations'])
            # 部分继承旋转
            for i in range(min(self.n, len(best_rot))):
                if random.random() < 0.5:
                    c['rotations'][i] = best_rot[i]

        # 使用学习到的有效策略
        total_eff = sum(knowledge_base['effective_strategies'].values())
        if total_eff > 0:
            probs = [knowledge_base['effective_strategies'][s] / total_eff for s in [0, 1, 2]]
            c['strategy'] = random.choices([0, 1, 2], weights=probs)[0]

        # 额外变异
        return self.mutate(c)

    def _create_from_knowledge(self, knowledge_base: Dict) -> Dict:
        """基于知识库创建新个体"""
        chrom = self.create_chromosome()

        # 使用知识库中的最优模式
        if knowledge_base['best_orders']:
            best_order = random.choice(knowledge_base['best_orders'])
            # 继承50-80%的顺序
            preserve_ratio = random.uniform(0.5, 0.8)
            preserve_count = int(self.n * preserve_ratio)

            new_order = [-1] * self.n
            used = set()

            # 随机选择位置继承
            inherit_positions = random.sample(range(min(self.n, len(best_order))),
                                             min(preserve_count, len(best_order)))

            for pos in inherit_positions:
                val = best_order[pos]
                if val not in used and val < self.n:
                    new_order[pos] = val
                    used.add(val)

            # 填充剩余
            remaining = [x for x in range(self.n) if x not in used]
            random.shuffle(remaining)

            idx = 0
            for i in range(self.n):
                if new_order[i] == -1 and idx < len(remaining):
                    new_order[i] = remaining[idx]
                    idx += 1

            if -1 not in new_order and len(set(new_order)) == self.n:
                chrom['order'] = new_order

        # 继承最优分区顺序
        if knowledge_base['best_zone_orders'] and random.random() < 0.6:
            chrom['zone_order'] = copy.deepcopy(random.choice(knowledge_base['best_zone_orders']))

        # 继承旋转
        if knowledge_base['best_rotations'] and random.random() < 0.5:
            best_rot = random.choice(knowledge_base['best_rotations'])
            for i in range(min(self.n, len(best_rot))):
                if random.random() < 0.7:
                    chrom['rotations'][i] = best_rot[i]

        return self._repair_chromosome(chrom)

    def optimize(self) -> Tuple[List[Facility], float, float]:
        """
        高级智能优化算法 V4.0
        融合: 岛屿模型 + 差分进化 + 模拟退火 + 强化学习引导 + Pareto多目标
        """
        print("=" * 78)
        print("      🚀 供水厂设施布局优化系统 V4.0 - 高级智能优化")
        print("      融合: 岛屿并行 + 差分进化 + 模拟退火 + 强化学习 + Pareto前沿")
        print("=" * 78)
        print(f"种群: {self.population_size} × {self.n_islands}岛屿 = {self.population_size * self.n_islands} 总个体")
        print(f"最大代数: {self.generations}, 迁移间隔: {self.migration_interval}")
        print(f"⚡ CPU并行: {multiprocessing.cpu_count()} 核心")
        if GPU_AVAILABLE:
            print(f"🚀 GPU加速: {GPU_NAME}")
        print("-" * 78)

        # ========== 高级优化参数 ==========
        base_mutation_rate = self.mutation_rate
        learning_rate = 0.15
        momentum = 0.92

        # ========== 知识库（持续学习核心）==========
        knowledge_base = {
            'best_orders': [],
            'best_rotations': [],
            'best_zone_orders': [],
            'effective_strategies': {0: 1, 1: 1, 2: 1},
            'learned_patterns': [],
        }

        # 精英记忆池（扩大容量）
        elite_memory = []
        memory_size = 100  # 增大记忆池

        # ========== 全局最优保护 ==========
        global_best_fitness = float('inf')
        global_best_solution = None
        global_best_layout = None
        global_best_dims = (0, 0)

        # 学习状态
        stagnation_count = 0
        total_stagnation = 0
        restart_count = 0
        improvement_momentum = 0.1
        phase = "exploration"

        # 策略统计
        strategy_scores = {0: 0, 1: 0, 2: 0}
        strategy_usage = {0: 1, 1: 1, 2: 1}

        # ========== 岛屿模型初始化 ==========
        island_size = self.population_size // self.n_islands
        islands = []
        for island_id in range(self.n_islands):
            # 每个岛屿有不同的初始化策略
            island_pop = []
            for _ in range(island_size):
                chrom = self.create_chromosome()
                # 岛屿特化策略
                if island_id == 0:
                    chrom['strategy'] = 0  # 工艺流程优先
                elif island_id == 1:
                    chrom['strategy'] = 1  # 分区紧凑
                elif island_id == 2:
                    chrom['strategy'] = 2  # Skyline
                # 第4个岛屿随机混合
                island_pop.append(chrom)
            islands.append(island_pop)

        # 模拟退火温度
        sa_temperature = self.sa_T0

        n_workers = min(12, multiprocessing.cpu_count())  # 增加并行度

        # 确保初始变异率有效
        if np.isnan(self.mutation_rate) or self.mutation_rate <= 0:
            self.mutation_rate = 0.30

        print(f"\n{'阶段':<10} {'代数':>4} | {'面积':>10} | {'利用率':>7} | {'变异':>5} | {'温度':>7} | {'多样性':>6} | {'状态'}")
        print("-" * 85)

        for gen in range(self.generations):
            # ========== 岛屿并行进化 ==========
            all_fitness = []
            all_individuals = []

            for island_id, island_pop in enumerate(islands):
                # 并行计算岛屿适应度
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    island_fitness = list(executor.map(self.calculate_fitness, island_pop))

                all_fitness.extend(island_fitness)
                all_individuals.extend(island_pop)

            # ========== 更新全局最优 ==========
            min_fit = min(all_fitness)
            min_idx = all_fitness.index(min_fit)

            # 计算种群多样性
            diversity = self._calculate_diversity(all_individuals)

            if min_fit < self.best_fitness:
                improvement = (self.best_fitness - min_fit) / max(self.best_fitness, 1)
                self.best_fitness = min_fit
                self.best_solution = copy.deepcopy(all_individuals[min_idx])
                stagnation_count = 0
                improvement_momentum = momentum * improvement_momentum + learning_rate * improvement

                # RL奖励更新
                rl_state = self._get_rl_state(min_fit, stagnation_count, diversity)

                # 记录有效策略
                best_strategy = all_individuals[min_idx].get('strategy', 0)
                strategy_scores[best_strategy] += 1

                # ========== 更新全局最优（保护机制）==========
                if min_fit < global_best_fitness:
                    global_best_fitness = min_fit
                    global_best_solution = copy.deepcopy(all_individuals[min_idx])
                    size = np.sqrt(self.total_area) * 1.8
                    global_best_layout, gw, gh = self.decode_chromosome(global_best_solution, size)
                    global_best_dims = (gw, gh)

                    # ========== Pareto前沿更新 ==========
                    objectives = self._calculate_multi_objective(global_best_solution)
                    self._update_pareto_front(global_best_solution, objectives)

                    # ========== 知识学习：提取最优解特征 ==========
                    knowledge_base['best_orders'].append(copy.deepcopy(all_individuals[min_idx]['order']))
                    knowledge_base['best_rotations'].append(copy.deepcopy(all_individuals[min_idx]['rotations']))
                    knowledge_base['best_zone_orders'].append(copy.deepcopy(all_individuals[min_idx]['zone_order']))
                    knowledge_base['effective_strategies'][best_strategy] += 2

                    # 限制知识库大小
                    if len(knowledge_base['best_orders']) > 30:
                        knowledge_base['best_orders'] = knowledge_base['best_orders'][-30:]
                        knowledge_base['best_rotations'] = knowledge_base['best_rotations'][-30:]
                        knowledge_base['best_zone_orders'] = knowledge_base['best_zone_orders'][-30:]

                    print(f"  🎯 新全局最优! 面积: {gw*gh:.0f}㎡, 利用率: {self.total_area/(gw*gh)*100:.2f}%, Pareto解: {len(self.pareto_front)}")
            else:
                stagnation_count += 1
                total_stagnation += 1
                improvement_momentum *= 0.95

            self.fitness_history.append(self.best_fitness)

            # ========== 更新精英记忆池 ==========
            elite_idx = np.argsort(all_fitness)[:10]
            for idx in elite_idx:
                if len(elite_memory) < memory_size:
                    elite_memory.append((all_fitness[idx], copy.deepcopy(all_individuals[idx])))
                else:
                    worst_mem_idx = max(range(len(elite_memory)), key=lambda i: elite_memory[i][0])
                    if all_fitness[idx] < elite_memory[worst_mem_idx][0]:
                        elite_memory[worst_mem_idx] = (all_fitness[idx], copy.deepcopy(all_individuals[idx]))

            # ========== 自适应参数控制 ==========
            self._adaptive_parameter_control(gen, stagnation_count, diversity)

            # 模拟退火温度衰减
            sa_temperature *= self.sa_alpha
            sa_temperature = max(sa_temperature, self.sa_T_min)

            # ========== 自适应阶段切换 ==========
            if stagnation_count < 20:
                phase = "探索"
                adj_rate = min(2.0, 1.0 + 0.5 * improvement_momentum)
                self.mutation_rate = base_mutation_rate * adj_rate
            elif stagnation_count < 50:
                phase = "利用"
                self.mutation_rate = base_mutation_rate * 0.7
            else:
                phase = "精细"
                self.mutation_rate = base_mutation_rate * 0.3

            self.mutation_rate = max(0.05, min(0.6, self.mutation_rate))

            # ========== 输出进度 ==========
            if gen % 20 == 0 or stagnation_count == 0:
                size = np.sqrt(self.total_area) * 1.8
                placed, bw, bh = self.decode_chromosome(self.best_solution, size)
                util = self.total_area / (bw * bh) * 100 if bw * bh > 0 else 0
                status = "✓改进" if stagnation_count == 0 else f"停滞{stagnation_count}"
                print(f"{phase:<10} {gen:>4} | {bw*bh:>8.0f}㎡ | {util:>6.2f}% | {self.mutation_rate:>4.2f} | {sa_temperature:>6.1f} | {diversity:>5.2f} | {status}")

            # ========== 岛屿迁移（每隔一定代数）==========
            if gen > 0 and gen % self.migration_interval == 0:
                # 环形迁移：每个岛屿向下一个岛屿发送最优个体
                migrants = []
                for island_id in range(self.n_islands):
                    # 找到本岛屿最优个体
                    island_start = island_id * island_size
                    island_end = island_start + island_size
                    island_fit = all_fitness[island_start:island_end]
                    best_local_idx = island_start + np.argmin(island_fit)
                    n_migrants = max(1, int(island_size * self.migration_rate))
                    best_indices = np.argsort(island_fit)[:n_migrants]
                    migrants.append([copy.deepcopy(all_individuals[island_start + i]) for i in best_indices])

                # 迁移到下一个岛屿
                for island_id in range(self.n_islands):
                    target_island = (island_id + 1) % self.n_islands
                    # 替换目标岛屿的最差个体
                    for i, migrant in enumerate(migrants[island_id]):
                        if i < len(islands[target_island]):
                            islands[target_island][-1-i] = migrant

            # ========== 智能重启机制 ==========
            if stagnation_count > 80:
                restart_count += 1
                print(f"\n  🔄 智能重启 #{restart_count} (停滞{stagnation_count}代)")
                print(f"     已保护全局最优: {global_best_dims[0]:.0f}m×{global_best_dims[1]:.0f}m = {global_best_dims[0]*global_best_dims[1]:.0f}㎡")

                # 重新初始化所有岛屿，但保留精英
                for island_id in range(self.n_islands):
                    elite_n = max(5, island_size // 5)
                    island_start = island_id * island_size
                    island_fit = all_fitness[island_start:island_start + island_size]
                    elite_indices = np.argsort(island_fit)[:elite_n]
                    survivors = [copy.deepcopy(islands[island_id][i]) for i in elite_indices]

                    new_island = survivors.copy()

                    # 从全局最优和知识库生成
                    if global_best_solution:
                        for _ in range(island_size // 4):
                            mutant = self._knowledge_guided_mutate(global_best_solution, knowledge_base)
                            new_island.append(mutant)

                    # 差分进化生成
                    while len(new_island) < island_size - island_size // 5:
                        if len(survivors) >= 3:
                            target = random.choice(survivors)
                            mutant = self._differential_evolution_mutate(target, survivors)
                            new_island.append(mutant)
                        else:
                            new_island.append(self.create_chromosome())

                    # 随机新个体
                    while len(new_island) < island_size:
                        new_island.append(self.create_chromosome())

                    islands[island_id] = new_island[:island_size]

                stagnation_count = 0
                sa_temperature = self.sa_T0 * 0.5  # 重置温度
                self.mutation_rate = base_mutation_rate * 1.5
                phase = "探索"

                if restart_count >= 8:
                    print(f"\n达到最大重启次数，返回全局最优解")
                    break
                continue

            # ========== 收敛判断 ==========
            if stagnation_count > 120:
                print(f"\n第 {gen} 代达到收敛条件")
                break

            # ========== 岛屿内进化 ==========
            for island_id in range(self.n_islands):
                island_start = island_id * island_size
                island_pop = islands[island_id]
                island_fit = all_fitness[island_start:island_start + island_size]

                # 精英保留
                elite_n = max(3, island_size // 10)
                elite_indices = np.argsort(island_fit)[:elite_n]
                new_island = [copy.deepcopy(island_pop[i]) for i in elite_indices]

                # 从Pareto前沿注入
                if self.pareto_front and random.random() < 0.1:
                    pareto_chrom, _ = random.choice(self.pareto_front)
                    new_island.append(copy.deepcopy(pareto_chrom))

                # 根据策略效果调整策略分配
                total_score = sum(strategy_scores[s] / max(strategy_usage[s], 1) for s in strategy_scores)
                if total_score > 0:
                    strategy_probs = {s: (strategy_scores[s] / max(strategy_usage[s], 1)) / total_score
                                     for s in strategy_scores}
                else:
                    strategy_probs = {0: 0.5, 1: 0.3, 2: 0.2}

                # RL引导变异
                rl_state = self._get_rl_state(self.best_fitness, stagnation_count, diversity)
                rl_action = self._rl_select_action(rl_state)

                # 繁殖
                while len(new_island) < island_size:
                    # 混合策略：选择、交叉、变异
                    if random.random() < 0.3 and len(island_pop) >= 3:
                        # 差分进化
                        target = self.selection(island_pop, island_fit)
                        child = self._differential_evolution_mutate(target, island_pop)
                    elif random.random() < 0.5:
                        # 标准GA
                        p1 = self.selection(island_pop, island_fit)
                        p2 = self.selection(island_pop, island_fit)
                        child, _ = self.crossover(p1, p2)
                        child = self.mutate(child)
                    else:
                        # RL引导变异
                        parent = self.selection(island_pop, island_fit)
                        child = self._rl_guided_mutate(parent, rl_action)

                    # 模拟退火局部搜索（概率性）
                    if random.random() < 0.1 and sa_temperature > self.sa_T_min:
                        child = self._simulated_annealing_local_search(child, sa_temperature)

                    # 策略分配
                    if random.random() < 0.2:
                        child['strategy'] = random.choices([0, 1, 2],
                            weights=[strategy_probs[0], strategy_probs[1], strategy_probs[2]])[0]

                    strategy_usage[child.get('strategy', 0)] += 1
                    new_island.append(child)

                # 多样性注入
                if diversity < 0.3:
                    inject_n = island_size // 10
                    for _ in range(inject_n):
                        idx = random.randint(elite_n, len(new_island) - 1)
                        if knowledge_base['best_orders'] and random.random() < 0.5:
                            new_island[idx] = self._create_from_knowledge(knowledge_base)
                        else:
                            new_island[idx] = self.create_chromosome()

                islands[island_id] = new_island[:island_size]

                # RL更新
                new_fit = min(self.calculate_fitness(c) for c in new_island[:3])
                reward = (self.best_fitness - new_fit) / max(self.best_fitness, 1) * 100
                new_state = self._get_rl_state(new_fit, stagnation_count, diversity)
                self._rl_update(rl_state, rl_action, reward, new_state)

        # ========== 最终输出（使用全局最优）==========
        size = np.sqrt(self.total_area) * 1.8
        current_layout, cw, ch = self.decode_chromosome(self.best_solution, size)
        current_area = cw * ch

        if global_best_solution is not None:
            global_area = global_best_dims[0] * global_best_dims[1]
            if global_area < current_area and global_area > 0:
                final_layout = global_best_layout
                final_w, final_h = global_best_dims
                print("\n  ✅ 使用全局保护的最优解")
            else:
                final_layout = current_layout
                final_w, final_h = cw, ch
        else:
            final_layout = current_layout
            final_w, final_h = cw, ch

        print("\n" + "=" * 78)
        print("                    🏆 高级智能优化完成 🏆")
        print("=" * 78)
        print(f"  📐 最终尺寸: {final_w:.1f}m × {final_h:.1f}m = {final_w*final_h:.1f}㎡")
        print(f"  📊 空间利用率: {self.total_area/(final_w*final_h)*100:.2f}%")
        print(f"  🔄 智能重启: {restart_count}次 | 总停滞: {total_stagnation}代")
        print(f"  🧬 策略效果: 工艺流程={strategy_scores[0]}, 分区紧凑={strategy_scores[1]}, Skyline={strategy_scores[2]}")
        print(f"  📚 知识库: {len(knowledge_base['best_orders'])}个最优模式")
        print(f"  🎯 Pareto前沿: {len(self.pareto_front)}个非支配解")
        print(f"  🤖 RL状态数: {len(self.rl_q_table)}个")
        print("=" * 78)

        # ========== 锁定优化后的布局边界 ==========
        # 美化只在此边界内进行，不允许扩大总布局区域
        locked_w, locked_h = final_w, final_h
        print(f"\n🔒 锁定布局边界: {locked_w:.1f}m × {locked_h:.1f}m (利用率: {self.total_area/(locked_w*locked_h)*100:.2f}%)")

        # ========== 布局美化处理（在锁定边界内）==========
        print("\n🎨 正在进行布局美化处理（不改变总布局区域）...")
        final_layout = self._beautify_layout_constrained(final_layout, locked_w, locked_h)

        # 确保没有设施超出锁定边界
        for f in final_layout:
            fw, fh = f.get_dimensions()
            # 如果超出边界，拉回来
            if f.x + fw > locked_w:
                f.x = locked_w - fw
            if f.y + fh > locked_h:
                f.y = locked_h - fh
            if f.x < 0:
                f.x = 0
            if f.y < 0:
                f.y = 0

        # 修复可能的冲突（在边界内）- 执行两轮确保无冲突
        print("   🔧 修复布局冲突...")
        final_layout = self._fix_overlaps_constrained(final_layout, locked_w, locked_h)
        final_layout = self._fix_overlaps_constrained(final_layout, locked_w, locked_h)

        # 最终验证冲突
        conflict_count = 0
        for i, f1 in enumerate(final_layout):
            w1, h1 = f1.get_dimensions()
            for j, f2 in enumerate(final_layout):
                if j <= i:
                    continue
                w2, h2 = f2.get_dimensions()
                overlap_x = not (f1.x + w1 <= f2.x or f2.x + w2 <= f1.x)
                overlap_y = not (f1.y + h1 <= f2.y or f2.y + h2 <= f1.y)
                if overlap_x and overlap_y:
                    conflict_count += 1

        if conflict_count > 0:
            print(f"   ⚠️ 仍有 {conflict_count} 处冲突，执行紧急修复...")
            final_layout = self._emergency_fix_overlaps(final_layout, locked_w, locked_h)

        print(f"   美化完成，布局边界保持: {locked_w:.1f}m × {locked_h:.1f}m")
        print(f"   空间利用率: {self.total_area/(locked_w*locked_h)*100:.2f}%")

        self.best_layout = final_layout
        return final_layout, locked_w, locked_h

    def _emergency_fix_overlaps(self, layout: List[Facility], max_w: float, max_h: float) -> List[Facility]:
        """紧急冲突修复 - 暴力解决所有重叠"""
        # 按面积从大到小排序，大设施保持位置，小设施移动
        sorted_facilities = sorted(layout, key=lambda f: -f.area)

        fixed = [sorted_facilities[0]]  # 最大的保持不动

        for f in sorted_facilities[1:]:
            fw, fh = f.get_dimensions()

            # 检查当前位置是否有冲突
            has_conflict = False
            for placed in fixed:
                pw, ph = placed.get_dimensions()
                overlap_x = not (f.x + fw <= placed.x or placed.x + pw <= f.x)
                overlap_y = not (f.y + fh <= placed.y or placed.y + ph <= f.y)
                if overlap_x and overlap_y:
                    has_conflict = True
                    break

            if has_conflict:
                # 找到无冲突位置
                found = False
                for y in range(0, int(max_h - fh) + 1, 2):
                    for x in range(0, int(max_w - fw) + 1, 2):
                        valid = True
                        for placed in fixed:
                            pw, ph = placed.get_dimensions()
                            req_gap = get_safety_distance(f, placed)
                            if not (x >= placed.x + pw + req_gap or
                                    placed.x >= x + fw + req_gap or
                                    y >= placed.y + ph + req_gap or
                                    placed.y >= y + fh + req_gap):
                                valid = False
                                break
                        if valid:
                            f.x, f.y = x, y
                            found = True
                            break
                    if found:
                        break

            fixed.append(f)

        return layout

    def _beautify_layout_constrained(self, layout: List[Facility], max_w: float, max_h: float) -> List[Facility]:
        """
        约束版布局美化 - 在锁定的边界内进行美化，不扩大总布局区域
        只做不会导致设施超出边界的调整
        """
        fdict = {f.name: f for f in layout}

        # ===== 1. 宿舍楼整齐排列（仅在有空间时）=====
        dorm_names = ["DM1", "DM2", "DM3", "DM4"]
        dorms = [fdict[n] for n in dorm_names if n in fdict]

        if len(dorms) == 4:
            # 统一所有宿舍方向（全部不旋转）
            for d in dorms:
                d.rotated = False

            # 重新获取统一后的尺寸
            dorm_w = dorms[0].length  # 原始长度
            dorm_h = dorms[0].width   # 原始宽度
            gap = 3  # 宿舍间距
            dorm_block_w = dorm_w * 2 + gap
            dorm_block_h = dorm_h * 2 + gap

            # 找宿舍区当前的位置（使用原始位置作为参考）
            dorm_min_x = min(d.x for d in dorms)
            dorm_min_y = min(d.y for d in dorms)

            # 检查是否可以在当前位置附近形成2x2网格
            other_facilities = [f for f in layout if f.name not in dorm_names]

            # 尝试在当前位置附近找一个合适的2x2位置
            best_pos = None
            for offset_x in range(-10, 11, 2):
                for offset_y in range(-10, 11, 2):
                    test_x = max(0, dorm_min_x + offset_x)
                    test_y = max(0, dorm_min_y + offset_y)

                    # 检查是否超出边界
                    if test_x + dorm_block_w > max_w or test_y + dorm_block_h > max_h:
                        continue

                    # 检查2x2网格的每个位置是否与其他设施冲突
                    conflict = False
                    test_positions = [
                        (test_x, test_y),
                        (test_x + dorm_w + gap, test_y),
                        (test_x, test_y + dorm_h + gap),
                        (test_x + dorm_w + gap, test_y + dorm_h + gap)
                    ]

                    for tx, ty in test_positions:
                        for other in other_facilities:
                            ow, oh = other.get_dimensions()
                            req_gap = get_safety_distance(dorms[0], other)
                            if not (tx >= other.x + ow + req_gap or
                                    other.x >= tx + dorm_w + req_gap or
                                    ty >= other.y + oh + req_gap or
                                    other.y >= ty + dorm_h + req_gap):
                                conflict = True
                                break
                        if conflict:
                            break

                    if not conflict:
                        best_pos = (test_x, test_y)
                        break
                if best_pos:
                    break

            if best_pos:
                base_x, base_y = best_pos
                positions = [
                    (base_x, base_y),
                    (base_x + dorm_w + gap, base_y),
                    (base_x, base_y + dorm_h + gap),
                    (base_x + dorm_w + gap, base_y + dorm_h + gap)
                ]
                for i, d in enumerate(dorms):
                    d.x, d.y = positions[i]
                print(f"   ✓ 宿舍楼2×2网格布置于({base_x:.1f}, {base_y:.1f})")

        # ===== 2. 门卫室放在角落（如果可以）=====
        if "GT" in fdict:
            gt = fdict["GT"]
            gt_w, gt_h = gt.get_dimensions()

            # 检查左下角是否可用
            conflict = False
            for other in layout:
                if other.name == "GT":
                    continue
                ow, oh = other.get_dimensions()
                req_gap = get_safety_distance(gt, other)
                if not (0 >= other.x + ow + req_gap or
                        gt_w + req_gap <= other.x or
                        0 >= other.y + oh + req_gap or
                        gt_h + req_gap <= other.y):
                    conflict = True
                    break

            if not conflict:
                gt.x, gt.y = 0, 0
                print("   ✓ 门卫室已放置在入口位置(左下角)")

        # ===== 3. 设施对齐优化（提高整齐度）=====
        print("   🔲 正在优化设施对齐...")
        align_grid = 5  # 5米对齐网格
        align_count = 0

        for f in layout:
            fw, fh = f.get_dimensions()
            orig_x, orig_y = f.x, f.y

            # 尝试对齐到网格
            aligned_x = round(f.x / align_grid) * align_grid
            aligned_y = round(f.y / align_grid) * align_grid

            # 检查对齐后的位置是否有效（不超出边界且无冲突）
            if aligned_x >= 0 and aligned_y >= 0 and aligned_x + fw <= max_w and aligned_y + fh <= max_h:
                can_align = True
                for other in layout:
                    if other is f:
                        continue
                    ow, oh = other.get_dimensions()
                    req_gap = get_safety_distance(f, other)
                    if not (aligned_x >= other.x + ow + req_gap or
                            other.x >= aligned_x + fw + req_gap or
                            aligned_y >= other.y + oh + req_gap or
                            other.y >= aligned_y + fh + req_gap):
                        can_align = False
                        break

                if can_align:
                    f.x, f.y = aligned_x, aligned_y
                    if (f.x, f.y) != (orig_x, orig_y):
                        align_count += 1

        if align_count > 0:
            print(f"   ✓ {align_count}个设施完成对齐优化")

        # ===== 4. 同行/同列对齐（微调）=====
        # 找到Y坐标相近的设施组，统一Y坐标
        y_tolerance = 3  # 3m内视为同一行
        grouped = {}
        for f in layout:
            key = round(f.y / y_tolerance) * y_tolerance
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(f)

        row_align_count = 0
        for y_key, flist in grouped.items():
            if len(flist) >= 2:
                # 找到该组的平均Y坐标
                avg_y = np.mean([f.y for f in flist])
                target_y = round(avg_y / align_grid) * align_grid

                for f in flist:
                    fw, fh = f.get_dimensions()
                    if target_y >= 0 and target_y + fh <= max_h:
                        # 检查移动后是否冲突
                        can_move = True
                        for other in layout:
                            if other is f or other in flist:
                                continue
                            ow, oh = other.get_dimensions()
                            req_gap = get_safety_distance(f, other)
                            if not (f.x >= other.x + ow + req_gap or
                                    other.x >= f.x + fw + req_gap or
                                    target_y >= other.y + oh + req_gap or
                                    other.y >= target_y + fh + req_gap):
                                can_move = False
                                break

                        if can_move and abs(f.y - target_y) < 5:
                            f.y = target_y
                            row_align_count += 1

        if row_align_count > 0:
            print(f"   ✓ {row_align_count}个设施完成行对齐")

        # ===== 5. 收紧边界（只向内收紧，不能超出max_w, max_h）=====
        min_x = min(f.x for f in layout)
        min_y = min(f.y for f in layout)

        if min_x > 1:
            shift = min_x - 0.5
            for f in layout:
                f.x -= shift
            print(f"   ✓ X方向内部收紧 {shift:.1f}m")

        if min_y > 1:
            shift = min_y - 0.5
            for f in layout:
                f.y -= shift
            print(f"   ✓ Y方向内部收紧 {shift:.1f}m")

        return layout

    def _fix_overlaps_constrained(self, layout: List[Facility], max_w: float, max_h: float) -> List[Facility]:
        """
        约束版冲突修复 - 在锁定边界内修复重叠和安全距离问题
        不会将设施推出边界
        """
        max_iterations = 500

        def check_position_valid(fac, x, y, others, bound_w, bound_h):
            """检查位置是否在边界内且与其他设施无冲突"""
            w, h = fac.get_dimensions()

            # 检查边界
            if x < 0 or y < 0 or x + w > bound_w or y + h > bound_h:
                return False

            # 检查与其他设施的冲突
            for other in others:
                if other is fac:
                    continue
                ow, oh = other.get_dimensions()
                req_gap = get_safety_distance(fac, other)

                if not (x + w + req_gap <= other.x or
                        other.x + ow + req_gap <= x or
                        y + h + req_gap <= other.y or
                        other.y + oh + req_gap <= y):
                    return False
            return True

        def find_valid_position_in_bounds(fac, others, bound_w, bound_h):
            """在边界内找到一个有效位置"""
            fw, fh = fac.get_dimensions()
            orig_x, orig_y = fac.x, fac.y

            # 尝试在原位置附近找
            for offset in range(1, 20):
                candidates = [
                    (orig_x + offset, orig_y),
                    (orig_x - offset, orig_y),
                    (orig_x, orig_y + offset),
                    (orig_x, orig_y - offset),
                    (orig_x + offset, orig_y + offset),
                    (orig_x - offset, orig_y - offset),
                ]
                for x, y in candidates:
                    if check_position_valid(fac, x, y, others, bound_w, bound_h):
                        return x, y

            # 网格搜索
            for y in range(0, int(bound_h - fh), 3):
                for x in range(0, int(bound_w - fw), 3):
                    if check_position_valid(fac, x, y, others, bound_w, bound_h):
                        return x, y

            # 实在找不到，返回原位置
            return orig_x, orig_y

        for iteration in range(max_iterations):
            has_conflict = False

            for i, f1 in enumerate(layout):
                w1, h1 = f1.get_dimensions()

                for j, f2 in enumerate(layout):
                    if j <= i:
                        continue
                    w2, h2 = f2.get_dimensions()
                    required_gap = get_safety_distance(f1, f2)

                    # 检查重叠
                    overlap_x = not (f1.x + w1 <= f2.x or f2.x + w2 <= f1.x)
                    overlap_y = not (f1.y + h1 <= f2.y or f2.y + h2 <= f1.y)
                    physical_overlap = overlap_x and overlap_y

                    # 检查安全距离
                    if not physical_overlap:
                        dx = max(0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
                        dy = max(0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))
                        actual_gap = max(dx, dy)
                        safety_violation = actual_gap < required_gap - 0.1
                    else:
                        safety_violation = True

                    if physical_overlap or safety_violation:
                        has_conflict = True
                        # 移动面积较小的设施
                        if f1.area <= f2.area:
                            to_move = f1
                        else:
                            to_move = f2

                        new_x, new_y = find_valid_position_in_bounds(to_move, layout, max_w, max_h)
                        to_move.x, to_move.y = new_x, new_y
                        break

                if has_conflict:
                    break

            if not has_conflict:
                break

        return layout

    def _global_tighten(self, layout: List[Facility], max_rounds: int = 8) -> List[Facility]:
        """
        全局收紧 V2.0 - 增强版多策略收紧算法
        使用启发式方法通过多方向平移找到更紧凑的配置
        策略：向左、向下、对角线、间隙填充、大设施优先
        """
        print("   🔧 正在执行全局收紧 V2.0...")

        improved = True
        round_num = 0
        initial_area = max((f.x + f.get_dimensions()[0] for f in layout)) * \
                      max((f.y + f.get_dimensions()[1] for f in layout))

        def can_place_at(f, x, y, others):
            """检查设施是否可以放置在指定位置"""
            fw, fh = f.get_dimensions()
            if x < 0 or y < 0:
                return False
            for other in others:
                if other is f:
                    continue
                ow, oh = other.get_dimensions()
                req_gap = get_safety_distance(f, other)
                if not (x >= other.x + ow + req_gap or
                        other.x >= x + fw + req_gap or
                        y >= other.y + oh + req_gap or
                        other.y >= y + fh + req_gap):
                    return False
            return True

        while improved and round_num < max_rounds:
            improved = False
            round_num += 1

            # 策略1: 大设施优先向原点收紧（大设施移动效果最好）
            for f in sorted(layout, key=lambda x: -x.area):  # 按面积从大到小
                fw, fh = f.get_dimensions()
                best_x, best_y = f.x, f.y
                best_dist = f.x + f.y  # 到原点的曼哈顿距离

                # 尝试更靠近原点的位置
                for test_x in np.arange(0, f.x + 0.5, 2):  # 步进2m
                    for test_y in np.arange(0, f.y + 0.5, 2):
                        if can_place_at(f, test_x, test_y, layout):
                            dist = test_x + test_y
                            if dist < best_dist:
                                best_dist = dist
                                best_x, best_y = test_x, test_y

                if best_x != f.x or best_y != f.y:
                    f.x, f.y = best_x, best_y
                    improved = True

            # 策略2: 向左收紧 - 逐个尝试向左移动每个设施
            for f in sorted(layout, key=lambda x: x.x):  # 从左到右处理
                fw, fh = f.get_dimensions()
                best_x = f.x

                for test_x in range(int(f.x) - 1, -1, -1):
                    if can_place_at(f, test_x, f.y, layout):
                        best_x = test_x
                    else:
                        break

                if best_x < f.x:
                    f.x = best_x
                    improved = True

            # 策略3: 向下收紧 - 逐个尝试向下移动每个设施
            for f in sorted(layout, key=lambda x: x.y):  # 从下到上处理
                fw, fh = f.get_dimensions()
                best_y = f.y

                for test_y in range(int(f.y) - 1, -1, -1):
                    if can_place_at(f, f.x, test_y, layout):
                        best_y = test_y
                    else:
                        break

                if best_y < f.y:
                    f.y = best_y
                    improved = True

            # 策略4: 对角线收紧 - 同时向左下移动
            for f in sorted(layout, key=lambda x: x.x + x.y):
                fw, fh = f.get_dimensions()
                best_pos = (f.x, f.y)

                for offset in range(1, min(int(f.x), int(f.y)) + 1):
                    test_x, test_y = f.x - offset, f.y - offset
                    if can_place_at(f, test_x, test_y, layout):
                        best_pos = (test_x, test_y)
                    else:
                        break

                if best_pos != (f.x, f.y):
                    f.x, f.y = best_pos
                    improved = True

            # 策略5: 间隙填充 - 寻找内部空隙并填充小设施
            for f in sorted(layout, key=lambda x: x.area)[:len(layout)//3]:  # 只处理小设施
                fw, fh = f.get_dimensions()
                current_dist = f.x + f.y
                best_pos = (f.x, f.y)
                best_dist = current_dist

                # 在当前布局范围内寻找更好的位置
                max_x = max(of.x + of.get_dimensions()[0] for of in layout)
                max_y = max(of.y + of.get_dimensions()[1] for of in layout)

                for test_x in np.arange(0, max_x - fw + 1, 3):  # 步进3m
                    for test_y in np.arange(0, max_y - fh + 1, 3):
                        if can_place_at(f, test_x, test_y, layout):
                            dist = test_x + test_y
                            if dist < best_dist - 5:  # 至少改善5m
                                best_dist = dist
                                best_pos = (test_x, test_y)

                if best_pos != (f.x, f.y):
                    f.x, f.y = best_pos
                    improved = True

            # 计算当前面积
            current_area = max((f.x + f.get_dimensions()[0] for f in layout)) * \
                          max((f.y + f.get_dimensions()[1] for f in layout))
            improvement_percent = (initial_area - current_area) / initial_area * 100

            if improvement_percent > 0.1:
                print(f"      第 {round_num} 轮: 面积改善 {improvement_percent:.1f}%")
            else:
                improved = False

        # 最终边界修正
        min_x = min(f.x for f in layout)
        min_y = min(f.y for f in layout)
        if min_x > 0:
            for f in layout:
                f.x -= min_x
        if min_y > 0:
            for f in layout:
                f.y -= min_y

        return layout

    def _zone_group_move(self, layout: List[Facility]) -> List[Facility]:
        """
        分区整体移动 - 按功能分区将相邻的设施组作为整体移动
        避免单个设施被推到边缘，保持紧凑性
        """
        print("   🏘️  正在执行分区整体移动...")

        # 1. 按分区分组
        zones = {}
        for f in layout:
            if f.category not in zones:
                zones[f.category] = []
            zones[f.category].append(f)

        # 2. 对于每个分区，尝试整体紧凑移动
        for zone_name, facilities in zones.items():
            if len(facilities) <= 1:
                continue

            # 计算分区的边界
            zone_min_x = min(f.x for f in facilities)
            zone_min_y = min(f.y for f in facilities)
            zone_max_x = max(f.x + f.get_dimensions()[0] for f in facilities)
            zone_max_y = max(f.y + f.get_dimensions()[1] for f in facilities)

            zone_width = zone_max_x - zone_min_x
            zone_height = zone_max_y - zone_min_y

            # 计算其他分区的设施
            other_facilities = [f for f in layout if f.category != zone_name]

            if not other_facilities:
                continue

            # 尝试将整个分区向左向下移动
            best_offset_x = 0
            best_offset_y = 0

            # 尝试向左移动
            for offset_x in range(1, int(zone_min_x) + 1):
                can_move = True
                new_zone_min_x = zone_min_x - offset_x

                for f in facilities:
                    for other in other_facilities:
                        ow, oh = other.get_dimensions()
                        req_gap = get_safety_distance(f, other)
                        new_f_x = f.x - offset_x

                        # 检查碰撞
                        fw, fh = f.get_dimensions()
                        if not (new_f_x >= other.x + ow + req_gap or
                                other.x >= new_f_x + fw + req_gap or
                                f.y >= other.y + oh + req_gap or
                                other.y >= f.y + fh + req_gap):
                            can_move = False
                            break
                    if not can_move:
                        break

                if can_move:
                    best_offset_x = offset_x
                else:
                    break

            # 尝试向下移动
            for offset_y in range(1, int(zone_min_y) + 1):
                can_move = True
                new_zone_min_y = zone_min_y - offset_y

                for f in facilities:
                    for other in other_facilities:
                        ow, oh = other.get_dimensions()
                        req_gap = get_safety_distance(f, other)
                        new_f_y = f.y - offset_y

                        # 检查碰撞
                        fw, fh = f.get_dimensions()
                        if not (f.x >= other.x + ow + req_gap or
                                other.x >= f.x + fw + req_gap or
                                new_f_y >= other.y + oh + req_gap or
                                other.y >= new_f_y + fh + req_gap):
                            can_move = False
                            break
                    if not can_move:
                        break

                if can_move:
                    best_offset_y = offset_y
                else:
                    break

            # 应用移动
            if best_offset_x > 0 or best_offset_y > 0:
                for f in facilities:
                    f.x -= best_offset_x
                    f.y -= best_offset_y
                print(f"      {zone_name}: 向左移动 {best_offset_x:.0f}m, 向下移动 {best_offset_y:.0f}m")

        # 3. 最终边界修正
        min_x = min(f.x for f in layout)
        min_y = min(f.y for f in layout)
        if min_x > 0:
            for f in layout:
                f.x -= min_x
        if min_y > 0:
            for f in layout:
                f.y -= min_y

        return layout

    def _beautify_layout(self, layout: List[Facility], bw: float, bh: float) -> Tuple[List[Facility], float, float]:
        """
        布局美化处理 - 在不增加总面积的前提下优化视觉效果
        1. 宿舍楼整齐排列（2x2网格）
        2. 同类设施对齐
        3. 边界设施贴边
        4. 微调消除不必要的间隙
        """
        fdict = {f.name: f for f in layout}

        # ===== 1. 宿舍楼整齐排列（2×2网格，但优先保持布局紧凑） =====
        dorm_names = ["DM1", "DM2", "DM3", "DM4"]
        dorms = [fdict[n] for n in dorm_names if n in fdict]

        # 辅助函数：检查位置是否与其他设施冲突
        def check_rect_conflict(x, y, w, h, exclude_names, check_list):
            """检查矩形区域是否与其他设施冲突"""
            for other in check_list:
                if other.name in exclude_names:
                    continue
                ow, oh = other.get_dimensions()
                req_gap = 2.0  # 默认间隙
                # 简单矩形相交检测
                if not (x >= other.x + ow + req_gap or
                        x + w + req_gap <= other.x or
                        y >= other.y + oh + req_gap or
                        y + h + req_gap <= other.y):
                    return True  # 有冲突
            return False  # 无冲突

        if len(dorms) == 4:
            # 统一所有宿舍方向
            for d in dorms:
                d.rotated = False  # 统一为15×10的方向

            dorm_w, dorm_h = dorms[0].get_dimensions()  # 15×10
            gap = 2  # 宿舍间隙

            # 2x2网格总尺寸
            dorm_block_w = dorm_w * 2 + gap  # 32m
            dorm_block_h = dorm_h * 2 + gap  # 22m

            # 寻找宿舍区最佳位置（只在布局内部寻找，不扩展）
            other_facilities = [f for f in layout if f.name not in dorm_names]
            current_max_x = max(f.x + f.get_dimensions()[0] for f in other_facilities) if other_facilities else 100
            current_max_y = max(f.y + f.get_dimensions()[1] for f in other_facilities) if other_facilities else 50

            best_pos = None
            for test_y in range(0, int(current_max_y - dorm_block_h + 1), 3):
                for test_x in range(0, int(current_max_x - dorm_block_w + 1), 3):
                    block_conflict = check_rect_conflict(
                        test_x, test_y, dorm_block_w, dorm_block_h,
                        dorm_names, other_facilities
                    )

                    if not block_conflict:
                        best_pos = (test_x, test_y)
                        break
                if best_pos:
                    break

            if best_pos:
                base_x, base_y = best_pos
                # 2x2网格排列
                positions = [
                    (base_x, base_y),
                    (base_x + dorm_w + gap, base_y),
                    (base_x, base_y + dorm_h + gap),
                    (base_x + dorm_w + gap, base_y + dorm_h + gap)
                ]
                for i, d in enumerate(dorms):
                    d.x, d.y = positions[i]
                print(f"   ✓ 宿舍楼2×2网格布置于({base_x:.1f}, {base_y:.1f})")
            else:
                # 无法在布局内放置2×2，保持GA优化后的位置，只统一方向
                print("   ! 宿舍保持GA优化位置（布局内无空间）")

        # ===== 2. 同分区设施Y坐标对齐 （简化版，只做安全对齐）=====
        # 跳过Y坐标对齐，避免引入冲突
        # categories = {}
        # 直接进入下一步

        # ===== 4. 门卫室放在角落 =====
        if "GT" in fdict:
            gt = fdict["GT"]
            gt_w, gt_h = gt.get_dimensions()

            # 检查左下角是否可用
            conflict = False
            for other in layout:
                if other.name == "GT":
                    continue
                ow, oh = other.get_dimensions()
                req_gap = get_safety_distance(gt, other)
                if not (0 >= other.x + ow + req_gap or
                        gt_w + req_gap <= other.x or
                        0 >= other.y + oh + req_gap or
                        gt_h + req_gap <= other.y):
                    conflict = True
                    break

            if not conflict:
                gt.x, gt.y = 0, 0
                print("   ✓ 门卫室已放置在入口位置(左下角)")

        # ===== 5. 收紧边界 =====
        # 计算所有设施的实际边界
        if layout:
            min_x = min(f.x for f in layout)
            min_y = min(f.y for f in layout)

            # 如果有空白，平移所有设施
            if min_x > 2:
                for f in layout:
                    f.x -= (min_x - 1)
                print(f"   ✓ X方向收紧 {min_x - 1:.1f}m")

            if min_y > 2:
                for f in layout:
                    f.y -= (min_y - 1)
                print(f"   ✓ Y方向收紧 {min_y - 1:.1f}m")

            # ===== 6. 修复可能的冲突 =====
            layout = self._fix_overlaps_and_safety(layout)

            # ===== 7. 多次收紧边界（确保最紧凑）=====
            for _ in range(3):  # 多轮收紧
                min_x = min(f.x for f in layout)
                min_y = min(f.y for f in layout)
                if min_x > 0.5:
                    for f in layout:
                        f.x -= min_x
                if min_y > 0.5:
                    for f in layout:
                        f.y -= min_y

            # ===== 8. 最终修复（多轮确保无冲突）=====
            for repair_round in range(3):  # 最多3轮修复
                layout = self._fix_overlaps_and_safety(layout)

                # 验证是否还有冲突
                has_issue = False
                for i, f1 in enumerate(layout):
                    w1, h1 = f1.get_dimensions()
                    for j, f2 in enumerate(layout):
                        if j <= i:
                            continue
                        w2, h2 = f2.get_dimensions()
                        # 检查重叠
                        if not (f1.x >= f2.x + w2 or f2.x >= f1.x + w1 or
                                f1.y >= f2.y + h2 or f2.y >= f1.y + h1):
                            has_issue = True
                            break
                        # 检查安全距离
                        required_gap = get_safety_distance(f1, f2)
                        gap_x = max(0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
                        gap_y = max(0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))
                        actual_gap = min(gap_x, gap_y) if gap_x > 0 and gap_y > 0 else max(gap_x, gap_y)
                        if actual_gap < required_gap - 0.1:
                            has_issue = True
                            break
                    if has_issue:
                        break

                if not has_issue:
                    break

            # 最终收紧
            min_x = min(f.x for f in layout)
            min_y = min(f.y for f in layout)
            if min_x > 0:
                for f in layout:
                    f.x -= min_x
            if min_y > 0:
                for f in layout:
                    f.y -= min_y

            # 重新计算边界（紧贴设施）
            new_bw = max(f.x + f.get_dimensions()[0] for f in layout)
            new_bh = max(f.y + f.get_dimensions()[1] for f in layout)

            # 边界留1m余量
            new_bw = new_bw + 1
            new_bh = new_bh + 1

            return layout, new_bw, new_bh

        return layout, bw, bh


# ==================== 论文级可视化与结果输出 ====================

PUBLICATION_ZONE_COLORS = {
    "intake": "#6AAED6",
    "pretreat": "#9ECAE1",
    "conventional": "#2F6FA3",
    "advanced": "#9C89B8",
    "chemical": "#D9857A",
    "distribution": "#7DAF75",
    "power": "#D8A24A",
    "sludge": "#8C6D62",
    "admin": "#8DD3C7",
    "auxiliary": "#BDBDBD",
    "living": "#E7A977",
}

ZONE_LABELS_EN = {
    "intake": "Intake",
    "pretreat": "Pretreatment",
    "conventional": "Conventional",
    "advanced": "Advanced",
    "chemical": "Chemical",
    "distribution": "Distribution",
    "power": "Power",
    "sludge": "Sludge",
    "admin": "Administration",
    "auxiliary": "Auxiliary",
    "living": "Living",
}


def _apply_publication_style():
    """Nature-style matplotlib defaults with editable SVG/PDF text."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Microsoft YaHei", "SimHei"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _panel_label(ax, label: str, x: float = -0.08, y: float = 1.04):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", color="#222222")


def _save_publication_figure(fig, filename_base: str, dpi: int = 600):
    fig.savefig(f"{filename_base}.svg", bbox_inches="tight")
    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename_base}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")


def _facility_distance(f1: Facility, f2: Facility) -> float:
    w1, h1 = f1.get_dimensions()
    w2, h2 = f2.get_dimensions()
    dx = max(0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
    dy = max(0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))
    return float(np.sqrt(dx**2 + dy**2))


def _layout_metrics(facilities: List[Facility], bw: float, bh: float,
                    history: List[float]) -> Dict:
    total_facility_area = sum(f.area for f in facilities)
    layout_area = bw * bh
    fdict = {f.name: f for f in facilities}

    adjacency_rows = []
    for n1, n2, weight in ADJACENCY_REQUIRED:
        if n1 in fdict and n2 in fdict:
            dist = _facility_distance(fdict[n1], fdict[n2])
            adjacency_rows.append({
                "from": n1,
                "to": n2,
                "weight": weight,
                "distance_m": dist,
                "satisfied": dist < 5.0,
            })

    safety_rows = []
    for i, f1 in enumerate(facilities):
        for f2 in facilities[i + 1:]:
            required = get_safety_distance(f1, f2)
            actual = _facility_distance(f1, f2)
            safety_rows.append({
                "facility_1": f1.name,
                "facility_2": f2.name,
                "actual_gap_m": actual,
                "required_gap_m": required,
                "satisfied": actual + 0.1 >= required,
            })

    pipeline_segments, pipeline_summary = calculate_pipeline_network(facilities)
    if pipeline_segments.empty:
        pipeline_rows = []
    else:
        pipeline_rows = (
            pipeline_segments.groupby(["system", "label"], as_index=False)
            .agg(length_m=("routed_length_m", "sum"),
                 weighted_length_m=("weighted_length_m", "sum"),
                 segment_count=("from", "count"))
            .to_dict("records")
        )

    zone_rows = []
    for cat in CATEGORY_NAMES:
        zone_facilities = [f for f in facilities if f.category == cat]
        if not zone_facilities:
            continue
        area = sum(f.area for f in zone_facilities)
        zone_rows.append({
            "zone": cat,
            "zone_cn": CATEGORY_NAMES[cat],
            "zone_en": ZONE_LABELS_EN.get(cat, cat),
            "facility_count": len(zone_facilities),
            "facility_area_m2": area,
            "area_share_pct": area / total_facility_area * 100 if total_facility_area else 0,
        })

    return {
        "layout_area": layout_area,
        "facility_area": total_facility_area,
        "utilization": total_facility_area / layout_area * 100 if layout_area else 0,
        "adjacency_rows": adjacency_rows,
        "adjacency_rate": (
            sum(r["satisfied"] for r in adjacency_rows) / len(adjacency_rows) * 100
            if adjacency_rows else 0
        ),
        "safety_rows": safety_rows,
        "safety_rate": (
            sum(r["satisfied"] for r in safety_rows) / len(safety_rows) * 100
            if safety_rows else 0
        ),
        "pipeline_rows": pipeline_rows,
        "pipeline_segments": pipeline_segments.to_dict("records") if not pipeline_segments.empty else [],
        "pipeline_summary": pipeline_summary,
        "zone_rows": zone_rows,
        "best_fitness": min(history) if history else None,
        "final_fitness": history[-1] if history else None,
    }


def _write_csv(path: str, rows: List[Dict], fields: List[str]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_publication_tables(facilities: List[Facility], bw: float, bh: float,
                              history: List[float], metrics: Dict,
                              output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    facility_rows = []
    for f in sorted(facilities, key=lambda item: (item.category, item.name)):
        w, h = f.get_dimensions()
        facility_rows.append({
            "code": f.name,
            "name_cn": f.name_cn,
            "zone": f.category,
            "zone_cn": CATEGORY_NAMES.get(f.category, f.category),
            "x_m": round(f.x, 3),
            "y_m": round(f.y, 3),
            "width_m": round(w, 3),
            "height_m": round(h, 3),
            "area_m2": round(f.area, 3),
            "rotated": int(bool(f.rotated)),
            "safety_level": f.safety_level,
        })

    summary_rows = [
        {"metric": "layout_width_m", "value": round(bw, 3), "unit": "m"},
        {"metric": "layout_height_m", "value": round(bh, 3), "unit": "m"},
        {"metric": "layout_area_m2", "value": round(metrics["layout_area"], 3), "unit": "m2"},
        {"metric": "facility_area_m2", "value": round(metrics["facility_area"], 3), "unit": "m2"},
        {"metric": "space_utilization_pct", "value": round(metrics["utilization"], 3), "unit": "%"},
        {"metric": "adjacency_satisfaction_pct", "value": round(metrics["adjacency_rate"], 3), "unit": "%"},
        {"metric": "safety_satisfaction_pct", "value": round(metrics["safety_rate"], 3), "unit": "%"},
        {"metric": "pipeline_total_length_m", "value": round(metrics["pipeline_summary"]["routed_length_m"], 3), "unit": "m"},
        {"metric": "pipeline_weighted_length_m", "value": round(metrics["pipeline_summary"]["weighted_length_m"], 3), "unit": "weighted m"},
        {"metric": "optimization_generations", "value": len(history), "unit": "generation"},
        {"metric": "final_fitness", "value": metrics["final_fitness"], "unit": ""},
    ]

    _write_csv(os.path.join(output_dir, "facility_layout.csv"), facility_rows,
               ["code", "name_cn", "zone", "zone_cn", "x_m", "y_m", "width_m",
                "height_m", "area_m2", "rotated", "safety_level"])
    _write_csv(os.path.join(output_dir, "summary_metrics.csv"), summary_rows,
               ["metric", "value", "unit"])
    _write_csv(os.path.join(output_dir, "zone_summary.csv"), metrics["zone_rows"],
               ["zone", "zone_cn", "zone_en", "facility_count", "facility_area_m2", "area_share_pct"])
    _write_csv(os.path.join(output_dir, "adjacency_report.csv"), metrics["adjacency_rows"],
               ["from", "to", "weight", "distance_m", "satisfied"])
    _write_csv(os.path.join(output_dir, "pipeline_summary.csv"), metrics["pipeline_rows"],
               ["system", "label", "length_m", "weighted_length_m", "segment_count"])
    _write_csv(os.path.join(output_dir, "pipeline_segments.csv"), metrics["pipeline_segments"],
               ["system", "label", "from", "to", "center_manhattan_m", "edge_clearance_m",
                "routed_length_m", "weighted_length_m", "detour_factor", "terminal_allowance_m", "weight"])
    _write_csv(os.path.join(output_dir, "safety_report.csv"), metrics["safety_rows"],
               ["facility_1", "facility_2", "actual_gap_m", "required_gap_m", "satisfied"])

    metadata = {
        "figure_contract": {
            "core_conclusion": "The optimized water-treatment-plant layout is compact while satisfying safety-spacing checks and reporting process-flow adjacency explicitly.",
            "archetype": "asymmetric mixed-modality figure",
            "backend": "Python/matplotlib",
            "exports": ["svg", "pdf", "tiff", "png"],
        },
        "metrics": summary_rows,
    }
    with open(os.path.join(output_dir, "publication_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    caption = (
        "# Publication figure notes\n\n"
        "**Figure claim.** The hybrid remaining-rectangle/Bottom-Left and genetic-"
        "algorithm optimizer yields a compact facility layout while making safety, "
        "pipeline, and process-adjacency evidence auditable.\n\n"
        "**Panel map.** a, Optimized plant layout with functional zones and process "
        "pipelines. b, Genetic optimization convergence. c, Facility-area composition "
        "by functional zone. d, Process-adjacency satisfaction summary. e, Pipeline "
        "length by system.\n\n"
        "**Export QA.** SVG/PDF retain editable text; TIFF is exported at 600 dpi. "
        "Quantitative source tables are saved as CSV files in the same output folder.\n"
    )
    with open(os.path.join(output_dir, "figure_caption_and_qa.md"), "w", encoding="utf-8") as f:
        f.write(caption)


def _draw_layout_panel(ax, facilities: List[Facility], bw: float, bh: float,
                       show_legend: bool = True, show_pipelines: bool = True):
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    ax.add_patch(Rectangle((0, 0), bw, bh, lw=0.8, ec="#2F2F2F", fc="#FBFBF8", zorder=0))
    grid = max(10, int(round(min(bw, bh) / 5 / 10) * 10))
    for x in np.arange(0, bw + 0.1, grid):
        ax.plot([x, x], [0, bh], color="#E8E8E8", lw=0.35, zorder=0.2)
    for y in np.arange(0, bh + 0.1, grid):
        ax.plot([0, bw], [y, y], color="#E8E8E8", lw=0.35, zorder=0.2)

    if show_pipelines:
        centers = {
            f.name: (f.x + f.get_dimensions()[0] / 2, f.y + f.get_dimensions()[1] / 2)
            for f in facilities
        }
        for key, pipeinfo in PIPELINE_SYSTEM.items():
            color = pipeinfo["color"]
            for start, end in pipeinfo["pipes"]:
                if start not in centers or end not in centers:
                    continue
                x0, y0 = centers[start]
                x1, y1 = centers[end]
                ax.plot([x0, x1], [y0, y1], color=color, lw=1.0,
                        linestyle=pipeinfo["style"], alpha=0.62, zorder=1)

    for f in facilities:
        w, h = f.get_dimensions()
        color = PUBLICATION_ZONE_COLORS.get(f.category, "#BDBDBD")
        rect = Rectangle((f.x, f.y), w, h, lw=0.55, ec="#303030",
                         fc=color, alpha=0.88, zorder=2)
        ax.add_patch(rect)
        cx, cy = f.x + w / 2, f.y + h / 2
        text_color = "white" if f.category in {"conventional", "advanced", "sludge"} else "#222222"
        fs = 5.5 if min(w, h) < 9 else 6.2
        ax.text(cx, cy, f.name, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=text_color, zorder=3)

    ax.set_xlim(-2, bw + 2)
    ax.set_ylim(-2, bh + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    ax.tick_params(length=2.5, width=0.55)

    if show_legend:
        zone_handles = [
            mpatches.Patch(facecolor=PUBLICATION_ZONE_COLORS.get(cat, "#BDBDBD"),
                           edgecolor="#303030", linewidth=0.4,
                           label=ZONE_LABELS_EN.get(cat, cat))
            for cat in CATEGORY_NAMES
            if any(f.category == cat for f in facilities)
        ]
        pipe_handles = [
            Line2D([0], [0], color=data["color"], lw=1.2,
                   linestyle=data["style"], label=data["label"])
            for data in PIPELINE_SYSTEM.values()
        ]
        leg1 = ax.legend(handles=zone_handles, loc="upper left",
                         bbox_to_anchor=(1.01, 1.0), title="Functional zone",
                         borderaxespad=0.0, handlelength=1.0)
        ax.add_artist(leg1)
        ax.legend(handles=pipe_handles, loc="lower left",
                  bbox_to_anchor=(1.01, 0.0), title="Pipeline",
                  borderaxespad=0.0, handlelength=1.6)


def _create_publication_outputs(facilities: List[Facility], bw: float, bh: float,
                                history: List[float],
                                save_prefix: str = "academic_analysis") -> Dict:
    import matplotlib.patches as mpatches

    _apply_publication_style()
    output_dir = f"{save_prefix}_publication"
    os.makedirs(output_dir, exist_ok=True)
    metrics = _layout_metrics(facilities, bw, bh, history)

    # Figure 1: standalone layout plan.
    fig_layout, ax_layout = plt.subplots(figsize=(183 / 25.4, 112 / 25.4), dpi=150)
    _draw_layout_panel(ax_layout, facilities, bw, bh, show_legend=True, show_pipelines=True)
    _panel_label(ax_layout, "a")
    ax_layout.set_title(
        f"Optimized water-treatment-plant layout ({bw:.0f} m x {bh:.0f} m; "
        f"utilization {metrics['utilization']:.1f}%)",
        pad=4,
    )
    _save_publication_figure(fig_layout, os.path.join(output_dir, "fig1_layout_plan"))
    plt.close(fig_layout)

    # Figure 2: asymmetric mixed-modality evidence figure.
    fig = plt.figure(figsize=(183 / 25.4, 148 / 25.4), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.65, 1, 1], height_ratios=[1.05, 1, 1])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[2, 1:])

    _draw_layout_panel(ax_a, facilities, bw, bh, show_legend=False, show_pipelines=True)
    _panel_label(ax_a, "a")
    ax_a.set_title("Spatial arrangement and process links", pad=3)

    generations = np.arange(len(history))
    if len(history) > 0:
        ax_b.plot(generations, history, color="#2F6FA3", lw=1.4)
        ax_b.scatter([int(np.argmin(history))], [min(history)], s=20,
                     color="#B64342", zorder=3, label="best")
    ax_b.set_xlabel("Generation")
    ax_b.set_ylabel("Fitness (lower is better)")
    ax_b.grid(axis="y", color="#E5E5E5", lw=0.4)
    _panel_label(ax_b, "b")
    ax_b.set_title("Convergence", pad=3)

    zone_rows = metrics["zone_rows"]
    zone_rows_sorted = sorted(zone_rows, key=lambda r: r["facility_area_m2"], reverse=True)
    zone_labels = [r["zone_en"] for r in zone_rows_sorted]
    zone_values = [r["facility_area_m2"] for r in zone_rows_sorted]
    zone_colors = [PUBLICATION_ZONE_COLORS.get(r["zone"], "#BDBDBD") for r in zone_rows_sorted]
    y = np.arange(len(zone_rows_sorted))
    ax_c.barh(y, zone_values, color=zone_colors, edgecolor="#303030", lw=0.35)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(zone_labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Area (m2)")
    _panel_label(ax_c, "c")
    ax_c.set_title("Zone composition", pad=3)

    adj_ok = sum(r["satisfied"] for r in metrics["adjacency_rows"])
    adj_bad = len(metrics["adjacency_rows"]) - adj_ok
    safety_ok = sum(r["satisfied"] for r in metrics["safety_rows"])
    safety_bad = len(metrics["safety_rows"]) - safety_ok
    bars = np.array([[adj_ok, adj_bad], [safety_ok, safety_bad]], dtype=float)
    totals = bars.sum(axis=1)
    totals[totals == 0] = 1
    bars_pct = bars / totals[:, None] * 100
    ax_d.barh([0, 1], bars_pct[:, 0], color="#7DAF75", edgecolor="#303030", lw=0.35, label="Satisfied")
    ax_d.barh([0, 1], bars_pct[:, 1], left=bars_pct[:, 0], color="#D9857A",
              edgecolor="#303030", lw=0.35, label="Unsatisfied")
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(["Adjacency", "Safety"])
    ax_d.set_xlim(0, 100)
    ax_d.set_xlabel("Share (%)")
    ax_d.legend(loc="lower right")
    _panel_label(ax_d, "d")
    ax_d.set_title("Constraint checks", pad=3)

    pipe_rows = sorted(metrics["pipeline_rows"], key=lambda r: r["length_m"], reverse=True)
    pipe_labels = [r["label"] for r in pipe_rows]
    pipe_values = [r["length_m"] for r in pipe_rows]
    pipe_colors = [PIPELINE_SYSTEM[r["system"]]["color"] for r in pipe_rows]
    x = np.arange(len(pipe_rows))
    ax_e.bar(x, pipe_values, color=pipe_colors, edgecolor="#303030", lw=0.35)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(pipe_labels, rotation=20, ha="right")
    ax_e.set_ylabel("Length (m)")
    ax_e.grid(axis="y", color="#E5E5E5", lw=0.4)
    _panel_label(ax_e, "e")
    ax_e.set_title("Pipeline length by system", pad=3)

    fig.suptitle("Optimization evidence for a compact and auditable water-plant layout",
                 fontsize=9, fontweight="bold")
    _save_publication_figure(fig, os.path.join(output_dir, "fig2_publication_summary"))
    plt.close(fig)

    _write_publication_tables(facilities, bw, bh, history, metrics, output_dir)

    print("\n" + "-" * 78)
    print("  论文级图件与结果表已生成：")
    print(f"     {output_dir}\\fig1_layout_plan.svg/.pdf/.tiff/.png")
    print(f"     {output_dir}\\fig2_publication_summary.svg/.pdf/.tiff/.png")
    print(f"     {output_dir}\\facility_layout.csv")
    print(f"     {output_dir}\\summary_metrics.csv")
    print(f"     {output_dir}\\zone_summary.csv")
    print(f"     {output_dir}\\adjacency_report.csv")
    print(f"     {output_dir}\\pipeline_summary.csv")
    print(f"     {output_dir}\\pipeline_segments.csv")
    print(f"     {output_dir}\\safety_report.csv")
    print(f"     {output_dir}\\figure_caption_and_qa.md")
    print("-" * 78)

    metrics["output_dir"] = output_dir
    return metrics


# ==================== 论文级可视化入口 ====================
def visualize(facilities: List[Facility], bw: float, bh: float,
              history: List[float], save_path: str):
    """导出单张论文级布局图，保留旧调用接口。"""
    _apply_publication_style()
    if facilities:
        bw = max(f.x + f.get_dimensions()[0] for f in facilities)
        bh = max(f.y + f.get_dimensions()[1] for f in facilities)

    base, _ = os.path.splitext(save_path)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 112 / 25.4), dpi=150)
    _draw_layout_panel(ax, facilities, bw, bh, show_legend=True, show_pipelines=True)
    _panel_label(ax, "a")
    utilization = sum(f.area for f in facilities) / (bw * bh) * 100 if bw * bh else 0
    ax.set_title(
        f"Optimized facility layout ({bw:.0f} m x {bh:.0f} m; utilization {utilization:.1f}%)",
        pad=4,
    )
    _save_publication_figure(fig, base)
    plt.close(fig)
    print(f"论文级布局图已保存: {base}.svg/.pdf/.tiff/.png")


def comprehensive_academic_analysis(facilities: List[Facility], bw: float, bh: float,
                                    history: List[float], save_prefix: str = "academic_analysis"):
    """导出论文复合图、图注说明和可审查结果表。"""
    return _create_publication_outputs(facilities, bw, bh, history, save_prefix)


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="供水厂设施布局优化")
    parser.add_argument("--population", type=int, default=200, help="单代种群规模")
    parser.add_argument("--generations", type=int, default=150, help="优化代数")
    parser.add_argument("--mutation-rate", type=float, default=0.35, help="初始变异率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现实验")
    parser.add_argument("--no-cache", action="store_true", help="关闭适应度缓存")
    parser.add_argument("--skip-plots", action="store_true", help="跳过图片和学术分析图输出")
    parser.add_argument("--output-prefix", default="academic_analysis", help="论文图件与结果表输出前缀")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("         🚀 供水厂设施布局优化系统 V4.0 - 高级智能优化")
    print("         融合: 岛屿模型 + 差分进化 + 模拟退火 + 强化学习 + Pareto前沿")
    print("=" * 78 + "\n")

    facilities = create_facilities()

    # 统计信息
    print("设施清单:")
    print("-" * 70)
    print(f"{'代码':<6}{'名称':<10}{'尺寸':^12}{'面积':>6}{'分区':<12}{'安全等级'}")
    print("-" * 70)

    total_area = 0
    for f in facilities:
        sl_text = ["", "普通", "中危", "高危"][f.safety_level]
        print(f"{f.name:<6}{f.name_cn:<10}{f.length:>2.0f}m×{f.width:>2.0f}m  "
              f"{f.area:>5.0f}㎡  {CATEGORY_NAMES[f.category]:<10} {sl_text}")
        total_area += f.area

    print("-" * 70)
    print(f"总计: {len(facilities)} 个设施, 总面积 {total_area} ㎡\n")

    # 分区统计
    print("📊 功能分区统计:")
    cat_count = {}
    cat_area = {}
    for f in facilities:
        cat_count[f.category] = cat_count.get(f.category, 0) + 1
        cat_area[f.category] = cat_area.get(f.category, 0) + f.area

    for cat in CATEGORY_NAMES:
        if cat in cat_count:
            print(f"  {CATEGORY_NAMES[cat]}: {cat_count[cat]}个设施, {cat_area[cat]}㎡")
    print()

    # ========== 高级智能优化 ==========
    print("🧠 启动四大分区智能优化（深度优化模式）...\n")

    ga = AdvancedGA(
        facilities,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        random_seed=args.seed,
        enable_cache=not args.no_cache
    )
    best_layout, bw, bh = ga.optimize()
    best_area = bw * bh

    print("\n" + "=" * 78)
    print("                   🏆 最终优化结果 🏆")
    print("=" * 78)
    print(f"  布局尺寸: {bw:.1f}m × {bh:.1f}m")
    print(f"  占地面积: {best_area:.1f} ㎡")
    print(f"  设施面积: {total_area} ㎡")
    print(f"  空间利用率: {total_area/best_area*100:.2f}%")
    print("=" * 78)

    if not args.skip_plots:
        # 可视化
        visualize(best_layout, bw, bh, ga.fitness_history, 'water_plant_layout_v3.png')

        # 全面学术论文风格可视化分析
        comprehensive_academic_analysis(best_layout, bw, bh, ga.fitness_history,
                                        args.output_prefix)

    # 输出坐标
    print("\n📍 设施坐标:")
    print("-" * 75)
    for f in sorted(best_layout, key=lambda x: (x.category, x.y, x.x)):
        w, h = f.get_dimensions()
        rot = "✓" if f.rotated else ""
        print(f"  {f.name:<5} {f.name_cn:<10} ({f.x:>5.1f}, {f.y:>5.1f}) "
              f"{w:>4.0f}×{h:>3.0f}m  {CATEGORY_NAMES[f.category]:<10} {rot}")

    # 相邻性分析
    print("\n📊 工艺流程相邻性:")
    print("-" * 55)
    fdict = {f.name: f for f in best_layout}
    adj_ok = 0
    for n1, n2, _ in ADJACENCY_REQUIRED:
        if n1 in fdict and n2 in fdict:
            f1, f2 = fdict[n1], fdict[n2]
            w1, h1 = f1.get_dimensions()
            w2, h2 = f2.get_dimensions()
            dx = max(0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
            dy = max(0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 5:
                print(f"  {n1:>4}-{n2:<4}: ✓ 相邻")
                adj_ok += 1
            else:
                print(f"  {n1:>4}-{n2:<4}: {dist:.1f}m")

    print(f"\n  满足率: {adj_ok}/{len(ADJACENCY_REQUIRED)} = "
          f"{adj_ok/len(ADJACENCY_REQUIRED)*100:.1f}%")

    # 重叠和安全距离验证
    print("\n🔍 布局冲突检测:")
    print("-" * 55)
    conflicts = []
    safety_violations = []

    for i, f1 in enumerate(best_layout):
        w1, h1 = f1.get_dimensions()
        for j, f2 in enumerate(best_layout):
            if j <= i:
                continue
            w2, h2 = f2.get_dimensions()

            # 检测物理重叠
            overlap_x = not (f1.x >= f2.x + w2 or f2.x >= f1.x + w1)
            overlap_y = not (f1.y >= f2.y + h2 or f2.y >= f1.y + h1)

            if overlap_x and overlap_y:
                conflicts.append((f1.name, f2.name))
            else:
                # 检测安全距离违规
                dx = max(0, max(f1.x, f2.x) - min(f1.x + w1, f2.x + w2))
                dy = max(0, max(f1.y, f2.y) - min(f1.y + h1, f2.y + h2))
                actual_gap = max(dx, dy)  # 边界距离
                required_gap = get_safety_distance(f1, f2)

                if actual_gap < required_gap - 0.5:  # 允许0.5m误差
                    safety_violations.append((f1.name, f2.name, actual_gap, required_gap))

    if conflicts:
        print(f"  ❌ 发现 {len(conflicts)} 处物理重叠:")
        for n1, n2 in conflicts[:10]:  # 最多显示10个
            print(f"     {n1} 与 {n2} 重叠")
    else:
        print("  ✓ 无物理重叠")

    if safety_violations:
        print(f"  ⚠️ 发现 {len(safety_violations)} 处安全距离不足:")
        for n1, n2, actual, required in safety_violations[:10]:
            print(f"     {n1}-{n2}: 实际{actual:.1f}m < 要求{required:.1f}m")
    else:
        print("  ✓ 所有安全距离满足")

    return best_layout


if __name__ == "__main__":
    main()
