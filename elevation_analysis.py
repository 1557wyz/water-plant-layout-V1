# -*- coding: utf-8 -*-
"""
竖向高程约束分析与可视化脚本
==============================
功能:
  1. 输出主水处理和污泥处理各构筑物的高程一览
  2. 绘制水力纵剖面图（高程图）
  3. 判断各段重力流可行性（高程阈值判断）
  4. 输出高程合规性报告
"""

import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib import ticker
import matplotlib.transforms as mtransforms
import pandas as pd

# ---------- 学术风格全局设置 ----------
plt.rcParams.update({
    'font.sans-serif':    ['SimHei', 'Microsoft YaHei', 'Arial'],
    'axes.unicode_minus': False,
    'axes.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.minor.size':   2,
    'ytick.minor.size':   2,
    'grid.linewidth':     0.5,
    'grid.alpha':         0.35,
    'figure.dpi':         150,
})

# ---------- 学术配色方案 ----------
PALETTE = {
    'pump':      '#D84315',   # 深橘红 — 泵站
    'gravity':   '#1565C0',   # 深蓝   — 重力流
    'end':       '#2E7D32',   # 深绿   — 终点
    'sludge':    '#6D4C41',   # 棕色   — 污泥
    'hgl':       '#0277BD',   # HGL折线
    'hgl_sludge':'#8D6E63',   # 污泥HGL
    'ground':    '#78909C',   # 地面线
    'soil_fill': '#CFD8DC',   # 土层填充
    'water':     '#90CAF9',   # 水体
    'datum':     '#546E7A',   # 基准线
    'surplus':   '#43A047',   # 盈余
    'deficit':   '#E53935',   # 缺口
    'req':       '#FB8C00',   # 所需水头
    'panel_bg':  '#FAFAFA',   # 面板背景
    'fig_bg':    '#FFFFFF',   # 图纸背景
}

# ==================== 从主文件导入高程常量 ====================
# （与 water_plant_layout_v3.py 中的 ELEVATION_DATA 完全一致）
ELEVATION_DATA = {
    "IPS":  {"ground": 100.0, "water_depth": 6.0,  "type": "pump",    "pump_head": 7.5,  "water_out": 107.5, "name_cn": "取水泵站"},
    "GC":   {"ground": 103.0, "water_depth": 3.5,  "type": "gravity", "loss": 0.30,      "water_out": 107.2, "name_cn": "格栅间"},
    "PS":   {"ground": 102.5, "water_depth": 4.0,  "type": "gravity", "loss": 0.50,      "water_out": 106.7, "name_cn": "预沉池"},
    "MB":   {"ground": 102.0, "water_depth": 3.5,  "type": "gravity", "loss": 0.30,      "water_out": 106.4, "name_cn": "混凝池"},
    "FB":   {"ground": 102.0, "water_depth": 3.5,  "type": "gravity", "loss": 0.50,      "water_out": 105.9, "name_cn": "絮凝池"},
    "SB":   {"ground": 101.5, "water_depth": 4.0,  "type": "gravity", "loss": 0.40,      "water_out": 105.5, "name_cn": "沉淀池"},
    "FT":   {"ground": 101.0, "water_depth": 5.5,  "type": "gravity", "loss": 2.00,      "water_out": 103.5, "name_cn": "滤池"},
    "OZG":  {"ground": 100.5, "water_depth": 5.0,  "type": "gravity", "loss": 0.60,      "water_out": 102.9, "name_cn": "臭氧接触池"},
    "ACF":  {"ground": 100.0, "water_depth": 5.0,  "type": "gravity", "loss": 1.50,      "water_out": 101.4, "name_cn": "活性炭滤池"},
    "CWT":  {"ground":  98.0, "water_depth": 5.0,  "type": "end",     "loss": 0.0,       "water_out": 101.0, "name_cn": "清水池"},
    "BWT":  {"ground":  99.0, "water_depth": 4.0,  "type": "end",     "loss": 0.0,       "water_out": 101.5, "name_cn": "反冲洗水池"},
    "PH2":  {"ground": 100.0, "water_depth": 0.0,  "type": "pump",    "pump_head": 35.0, "water_out": 135.0, "name_cn": "二级泵房"},
    "STK":  {"ground":  97.5, "water_depth": 3.5,  "type": "gravity", "loss": 0.30,      "water_out":  99.2, "name_cn": "污泥浓缩池"},
    "SDR":  {"ground":  96.5, "water_depth": 0.0,  "type": "gravity", "loss": 0.20,      "water_out":  99.0, "name_cn": "污泥脱水机房"},
    "SYD":  {"ground":  95.5, "water_depth": 0.0,  "type": "end",     "loss": 0.0,       "water_out":  95.5, "name_cn": "污泥堆场"},
}

# 工艺流程路径
MAIN_WATER_PATH   = ["IPS", "GC", "PS", "MB", "FB", "SB", "FT", "CWT", "PH2"]
ADVANCED_PATH     = ["FT", "OZG", "ACF", "CWT"]  # 深度处理支路
SLUDGE_PATH       = ["SB", "STK", "SDR", "SYD"]  # 污泥处理（SB排泥口≈99.5m）

PIPE_HEADLOSS_PER_METER = 0.001   # 全局默认沿程损失系数
PIPE_LOCAL_LOSS         = 0.30
GRAVITY_FLOW_SAFETY     = 0.20

# SB 底部排泥口高程（与主文件 SB_SLUDGE_OUTLET_ELEV 一致）
SB_SLUDGE_OUTLET = 99.5

# 各管段精细参数（与主文件 ELEV_SEGMENT_PARAMS 同步）
ELEV_SEGMENT_PARAMS = {
    ("IPS", "GC"):  (0.0010, 0.20, 0.20, 2.0),
    ("GC",  "PS"):  (0.0010, 0.30, 0.20, 2.0),
    ("PS",  "MB"):  (0.0012, 0.30, 0.20, 2.5),
    ("MB",  "FB"):  (0.0012, 0.25, 0.20, 2.5),
    ("FB",  "SB"):  (0.0010, 0.30, 0.20, 2.5),
    ("SB",  "FT"):  (0.0010, 0.30, 0.20, 2.5),
    ("FT",  "CWT"): (0.0010, 0.25, 0.20, 2.0),
    ("FT",  "OZG"): (0.0012, 0.30, 0.20, 1.5),
    ("OZG", "ACF"): (0.0012, 0.30, 0.20, 1.5),
    ("ACF", "CWT"): (0.0010, 0.25, 0.20, 1.5),
    ("CWT", "BWT"): (0.0008, 0.20, 0.15, 0.8),
    ("SB",  "STK"): (0.0040, 0.50, 0.30, 1.2),
    ("STK", "SDR"): (0.0050, 0.60, 0.30, 1.2),
    ("SDR", "SYD"): (0.0030, 0.40, 0.25, 0.8),
}

# 典型管线水平间距假设（无实际布局时使用）
TYPICAL_DISTANCES = {
    ("IPS", "GC"):  30, ("GC",  "PS"):  25, ("PS",  "MB"):  20,
    ("MB",  "FB"):  15, ("FB",  "SB"):  18, ("SB",  "FT"):  15,
    ("FT",  "CWT"): 20, ("CWT", "PH2"): 15,
    ("FT",  "OZG"): 18, ("OZG", "ACF"): 15, ("ACF", "CWT"): 12,
    ("CWT", "BWT"): 10,
    ("SB",  "STK"): 20, ("STK", "SDR"): 15, ("SDR", "SYD"): 12,
}


def calc_pipe_loss(L: float, up: str = None, down: str = None) -> float:
    """按分段参数计算两构筑物之间总管道水头损失"""
    if up and down and (up, down) in ELEV_SEGMENT_PARAMS:
        i_coef, hm, hs, _ = ELEV_SEGMENT_PARAMS[(up, down)]
    else:
        i_coef, hm, hs = PIPE_HEADLOSS_PER_METER, PIPE_LOCAL_LOSS, GRAVITY_FLOW_SAFETY
    return i_coef * L + hm + hs


def check_gravity_flow(up: str, down: str, L: float = None) -> dict:
    """
    判断上下游构筑物之间是否满足重力流（V2.0 使用分段参数）
    """
    if L is None:
        L = TYPICAL_DISTANCES.get((up, down), TYPICAL_DISTANCES.get((down, up), 20))

    # 获取分段参数
    seg_key = (up, down)
    if seg_key in ELEV_SEGMENT_PARAMS:
        i_coef, hm, hs, seg_w = ELEV_SEGMENT_PARAMS[seg_key]
    else:
        i_coef, hm, hs, seg_w = PIPE_HEADLOSS_PER_METER, PIPE_LOCAL_LOSS, GRAVITY_FLOW_SAFETY, 1.0

    # 修正：SB→STK 用排泥口高程
    if up == "SB" and down == "STK":
        water_out_up = SB_SLUDGE_OUTLET
    else:
        water_out_up = ELEVATION_DATA[up]["water_out"]

    # 泵站下游允许液位在地面以下
    if ELEVATION_DATA[down]["type"] == "pump":
        water_in_down = ELEVATION_DATA[down]["ground"] - 0.50
    else:
        water_in_down = ELEVATION_DATA[down]["ground"] + 0.30

    required  = i_coef * L + hm + hs
    available = water_out_up - water_in_down
    shortage  = max(0.0, required - available)
    feasible  = shortage == 0.0

    verdict = "✅ 可重力自流" if feasible else f"❌ 缺水头 {shortage:.3f}m（需提升）"
    return {
        "upstream":       ELEVATION_DATA[up]["name_cn"],
        "downstream":     ELEVATION_DATA[down]["name_cn"],
        "water_out_up":   round(water_out_up, 3),
        "water_in_down":  round(water_in_down, 3),
        "pipe_L_m":       L,
        "i_coef":         i_coef,
        "required_head":  round(required, 3),
        "available_head": round(available, 3),
        "surplus_m":      round(max(0.0, available - required), 3),
        "shortage_m":     round(shortage, 3),
        "seg_weight":     seg_w,
        "feasible":       feasible,
        "verdict":        verdict,
    }


# ==================== 生成报告数据 ====================
def build_report():
    """汇总所有重力流段的高程检查"""
    rows = []
    # 主水处理（常规段）
    for i in range(len(MAIN_WATER_PATH) - 1):
        up, dn = MAIN_WATER_PATH[i], MAIN_WATER_PATH[i+1]
        if up in ELEVATION_DATA and dn in ELEVATION_DATA:
            rows.append(("主水处理", *check_gravity_flow(up, dn).values()))

    # 深度处理支路
    for i in range(len(ADVANCED_PATH) - 1):
        up, dn = ADVANCED_PATH[i], ADVANCED_PATH[i+1]
        if (up, dn) != ("FT", "OZG"):  # 避免重复
            continue  # 只检查OZG→ACF→CWT
    for i in range(1, len(ADVANCED_PATH) - 1):
        up, dn = ADVANCED_PATH[i], ADVANCED_PATH[i+1]
        if up in ELEVATION_DATA and dn in ELEVATION_DATA:
            rows.append(("深度处理", *check_gravity_flow(up, dn).values()))

    # FT→OZG
    rows.append(("深度处理", *check_gravity_flow("FT", "OZG").values()))

    # 污泥处理
    sludge_pairs = [("SB", "STK"), ("STK", "SDR"), ("SDR", "SYD")]
    for up, dn in sludge_pairs:
        if up in ELEVATION_DATA and dn in ELEVATION_DATA:
            rows.append(("污泥处理", *check_gravity_flow(up, dn).values()))

    cols = ["流程类型", "上游设施", "下游设施", "上游出水位(m)", "下游进水位(m)",
            "管线长度(m)", "坡度系数i", "所需水头(m)", "可用水头(m)", "盈余(m)", "水头缺口(m)", "段权重", "是否可行", "判定"]
    return pd.DataFrame(rows, columns=cols)


# ==================== 绘图函数 ====================

def _draw_structure(ax, x_center, ground, water_depth, water_out, ftype,
                    label_cn, label_code, bar_w=0.55):
    """绘制单个构筑物标准剖面（地面线 + 池体 + 水体）"""
    c_wall  = PALETTE[ftype]
    c_water = PALETTE['water']

    pool_top = ground + max(water_depth * 0.15, 0.3)   # 池顶略高于地面
    pool_bot = ground - max(water_depth * 0.85, 0.5)   # 池底

    # 构筑物外壳（空心矩形）
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_center - bar_w/2, pool_bot),
        bar_w, pool_top - pool_bot,
        boxstyle="square,pad=0",
        linewidth=1.6, edgecolor=c_wall,
        facecolor=c_wall + '18', zorder=3
    ))

    # 水体填充（到出水位）
    if water_depth > 0 and ftype != 'pump':
        w_top = min(water_out, pool_top)
        w_bot = pool_bot + 0.15
        if w_top > w_bot:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_center - bar_w/2 + 0.04, w_bot),
                bar_w - 0.08, w_top - w_bot,
                boxstyle="square,pad=0",
                linewidth=0, facecolor=c_water,
                alpha=0.55, zorder=4
            ))

    # 名称标注（池底下方）
    ax.text(x_center, pool_bot - 0.22,
            f"{label_cn}\n({label_code})",
            ha='center', va='top', fontsize=7.2,
            color='#263238', fontweight='bold', zorder=6)
    return pool_bot


def plot_main_water_elevation(ax):
    """主水处理水力纵剖面图 — 学术工程图样式 v2"""
    path   = MAIN_WATER_PATH
    n      = len(path)          # 9: IPS→GC→…→CWT→PH2
    xs     = list(range(n))
    Y_MIN, Y_MAX = 93.0, 113.5

    ax.set_facecolor(PALETTE['panel_bg'])
    ax.set_xlim(-0.7, n - 0.15)
    ax.set_ylim(Y_MIN, Y_MAX)

    # ── 土层背景 ──
    grounds = [ELEVATION_DATA[k]['ground'] for k in path]
    ax.fill_between(xs, Y_MIN, grounds,
                    color=PALETTE['soil_fill'], alpha=0.50,
                    hatch='////', linewidth=0, zorder=1)
    ax.plot(xs, grounds, color=PALETTE['ground'],
            lw=1.3, linestyle='--', zorder=2)

    # ── 构筑物剖面 ──
    pool_bots = []
    for i, code in enumerate(path):
        ed = ELEVATION_DATA[code]
        pb = _draw_structure(ax, i, ed['ground'], ed.get('water_depth', 0),
                             ed['water_out'], ed['type'], ed['name_cn'], code)
        pool_bots.append(pb)

    # ── 主 HGL（不含PH2，PH2出水位135m超出图框） ──
    water_outs = [ELEVATION_DATA[k]['water_out'] for k in path]
    ax.plot(xs[:-1], water_outs[:-1], '-o',
            color=PALETTE['hgl'], lw=2.4, ms=6,
            markerfacecolor='white', markeredgewidth=1.8,
            zorder=8, label='HGL（水力坡降线）')

    # HGL 数值标注（交错上下，避免重叠）
    # (x_offset_pt, y_offset_pt, ha)
    lbl_cfg = [
        ( 5,  13, 'left'),   # IPS  107.50
        ( 5, -17, 'left'),   # GC   107.20
        ( 5,  13, 'left'),   # PS   106.70
        ( 5, -17, 'left'),   # MB   106.40
        ( 5,  13, 'left'),   # FB   105.90
        ( 5, -17, 'left'),   # SB   105.50
        (-2,  12, 'right'),  # FT   103.50
        (-2, -17, 'right'),  # CWT  101.00
    ]
    for i, (code, wo) in enumerate(zip(path[:-1], water_outs[:-1])):
        ox, oy, ha = lbl_cfg[i]
        ax.annotate(f'{wo:.2f}',
                    xy=(xs[i], wo), xytext=(ox, oy),
                    textcoords='offset points',
                    fontsize=6.5, color=PALETTE['hgl'],
                    fontweight='bold', ha=ha)

    # ── 深度处理支路 HGL（OZG/ACF 插在 FT—CWT 段内侧） ──
    ft_x  = path.index('FT')    # 6
    cwt_x = path.index('CWT')   # 7
    d_xs  = [ft_x, ft_x + 0.32, ft_x + 0.68, cwt_x]
    d_ys  = [ELEVATION_DATA[k]['water_out'] for k in ['FT', 'OZG', 'ACF', 'CWT']]
    ax.plot(d_xs, d_ys, '--^',
            color='#7B1FA2', lw=1.7, ms=5,
            markerfacecolor='white', markeredgewidth=1.5,
            alpha=0.92, zorder=7, label='深度处理支路 HGL')
    # OZG 标注（上方）
    ax.text(d_xs[1] + 0.04, d_ys[1] + 0.55,
            f"臭氧  {d_ys[1]:.2f} m", ha='left', va='center',
            fontsize=6.2, color='#7B1FA2',
            bbox=dict(facecolor='white', edgecolor='#CE93D8',
                      alpha=0.88, boxstyle='round,pad=0.18', linewidth=0.6))
    # ACF 标注（下方）
    ax.text(d_xs[2] - 0.04, d_ys[2] - 0.65,
            f"活性炭  {d_ys[2]:.2f} m", ha='right', va='center',
            fontsize=6.2, color='#7B1FA2',
            bbox=dict(facecolor='white', edgecolor='#CE93D8',
                      alpha=0.88, boxstyle='round,pad=0.18', linewidth=0.6))

    # ── 泵站提升箭头 ──
    for i, code in enumerate(path):
        ed = ELEVATION_DATA[code]
        if ed['type'] != 'pump':
            continue
        ph  = ed.get('pump_head', 0)
        bot = pool_bots[i] + 0.4
        top = min(ed['water_out'], Y_MAX - 1.8)
        ax.annotate('',
                    xy=(xs[i], top), xytext=(xs[i], bot),
                    arrowprops=dict(
                        arrowstyle='->', color=PALETTE['pump'],
                        lw=2.2, mutation_scale=18
                    ), zorder=10)
        ax.text(xs[i] + 0.30, (top + bot) / 2,
                f'H={ph:.0f} m',
                fontsize=7.5, color=PALETTE['pump'],
                fontweight='bold', va='center')
        # PH2 截断符（锯齿）+ 出水位说明
        if code == 'PH2':
            for yb, sign in [(top - 0.05, 1), (top + 0.15, -1)]:
                xb = [xs[i]-0.14, xs[i]-0.05, xs[i]+0.05, xs[i]+0.14]
                ybs = [yb, yb + 0.12*sign, yb + 0.12*sign, yb]
                ax.plot(xb, ybs, color=PALETTE['pump'], lw=1.3, zorder=11)
            ax.text(xs[i], top + 0.35, '↑  135.0 m',
                    ha='center', va='bottom', fontsize=6.5,
                    color=PALETTE['pump'], fontweight='bold')

    # ── 水头损失 Δh 标注（只标重力流段，上下交错） ──
    # sides: +1=标注在折线上方, -1=下方, None=跳过
    # 对应段: IPS→GC, GC→PS, PS→MB, MB→FB, FB→SB, SB→FT, FT→CWT, CWT→PH2
    sides   = [None, +1, -1, +1, -1, +1, -1, None]
    for i, side in enumerate(sides):
        if side is None:
            continue
        wo1, wo2 = water_outs[i], water_outs[i + 1]
        dh = wo1 - wo2
        if 0 < dh < 8:
            mx = (xs[i] + xs[i+1]) / 2
            my = (wo1 + wo2) / 2 + side * 0.62
            ax.text(mx, my, f'Δh={dh:.2f} m',
                    ha='center', fontsize=5.8, color='#01579B',
                    bbox=dict(facecolor='white', edgecolor='#90CAF9',
                              alpha=0.88, boxstyle='round,pad=0.15',
                              linewidth=0.6), zorder=10)

    # ── 基准线 ──
    ax.axhline(100.0, color=PALETTE['datum'], lw=0.9,
               linestyle=':', alpha=0.8, zorder=2)
    ax.text(-0.65, 100.15, '±0.000\n(100.00 m)',
            fontsize=6.3, color=PALETTE['datum'], va='bottom')

    # ── 坐标轴 ──
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [ELEVATION_DATA[k]['name_cn'] for k in path],
        fontsize=8.5, rotation=20, ha='right'
    )
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.set_ylabel('Elevation  (m)', fontsize=10)
    ax.grid(axis='y', which='major', linestyle='--', color='#B0BEC5', alpha=0.4)
    ax.grid(axis='y', which='minor', linestyle=':', color='#CFD8DC', alpha=0.20)

    # ── 图例（左下角内嵌，2列，避开高HGL区域） ──
    legend_items = [
        mpatches.Patch(facecolor=PALETTE['soil_fill'], hatch='////',
                       alpha=0.55, label='土层'),
        Line2D([0],[0], linestyle='--', color=PALETTE['ground'],
               lw=1.2, label='GL（地面线）'),
        Line2D([0],[0], marker='o', color=PALETTE['hgl'], lw=2.2,
               markerfacecolor='white', markeredgewidth=1.6,
               label='HGL（水力坡降线）'),
        Line2D([0],[0], linestyle='--', marker='^', color='#7B1FA2', lw=1.5,
               markerfacecolor='white', markeredgewidth=1.2,
               label='深度处理支路 HGL'),
        mpatches.Patch(facecolor=PALETTE['water'], alpha=0.55, label='水体'),
        Line2D([0],[0], color=PALETTE['pump'], lw=2.0,
               marker=r'$\uparrow$', markersize=9, label='泵站提升'),
    ]
    ax.legend(handles=legend_items, fontsize=7.5,
              loc='upper center', bbox_to_anchor=(0.5, -0.16),
              framealpha=0.93, edgecolor='#B0BEC5',
              ncol=6, borderpad=0.6, handlelength=1.8,
              handletextpad=0.5, columnspacing=1.0)

    ax.set_title('(a)  主水处理工艺水力纵剖面图（含深度处理支路）',
                 fontsize=11, fontweight='bold', loc='left', pad=8)


def plot_sludge_elevation(ax):
    """污泥处理竖向纵剖面图 — 学术工程图样式"""
    path       = ['SB', 'STK', 'SDR', 'SYD']
    xs         = list(range(len(path)))
    Y_MIN, Y_MAX = 93.0, 104.0

    ax.set_facecolor(PALETTE['panel_bg'])
    ax.set_xlim(-0.8, len(path) - 0.2)
    ax.set_ylim(Y_MIN, Y_MAX)

    grounds    = [ELEVATION_DATA[k]['ground'] for k in path]
    # SB 使用排泥口高程（非水侧出水位）
    water_outs = [
        SB_SLUDGE_OUTLET,
        ELEVATION_DATA['STK']['water_out'],
        ELEVATION_DATA['SDR']['water_out'],
        ELEVATION_DATA['SYD']['water_out'],
    ]

    # ── 土层 ──
    ax.fill_between(xs, Y_MIN, grounds,
                    color='#D7CCC8', alpha=0.6,
                    hatch='xxxx', linewidth=0, zorder=1, label='土层')
    ax.plot(xs, grounds, color=PALETTE['ground'],
            lw=1.3, linestyle='--', zorder=2, label='GL（地面线）')

    # ── 构筑物剖面 ──
    bar_w = 0.52
    for i, code in enumerate(path):
        ed = ELEVATION_DATA[code]
        g  = ed['ground']
        wd = ed.get('water_depth', 0)
        pool_top = g + max(wd * 0.15, 0.3)
        pool_bot = g - max(wd * 0.85, 0.8)
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - bar_w/2, pool_bot), bar_w, pool_top - pool_bot,
            boxstyle='square,pad=0', linewidth=1.6,
            edgecolor=PALETTE['sludge'],
            facecolor=PALETTE['sludge'] + '18', zorder=3
        ))
        if wd > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (i - bar_w/2 + 0.04, pool_bot + 0.15), bar_w - 0.08,
                min(water_outs[i], pool_top) - pool_bot - 0.15,
                boxstyle='square,pad=0', linewidth=0,
                facecolor='#BCAAA4', alpha=0.5, zorder=4
            ))
        ax.text(i, pool_bot - 0.22,
                f"{ed['name_cn']}\n({code})",
                ha='center', va='top', fontsize=7.2,
                color='#3E2723', fontweight='bold', zorder=6)

    # ── 污泥 HGL ──
    ax.plot(xs, water_outs, '-s',
            color=PALETTE['hgl_sludge'], lw=2.2, ms=7,
            markerfacecolor='white', markeredgewidth=1.8,
            zorder=8, label='污泥流液位 (HGL)')

    for i, (wo, code) in enumerate(zip(water_outs, path)):
        note = '(排泥口)' if code == 'SB' else ''
        ax.annotate(f'{wo:.2f}{note}',
                    xy=(i, wo), xytext=(6, 8),
                    textcoords='offset points',
                    fontsize=7, color=PALETTE['hgl_sludge'],
                    fontweight='bold')

    # ── 水头差标注 ──
    for i in range(len(path) - 1):
        dz = water_outs[i] - water_outs[i+1]
        mx = (xs[i] + xs[i+1]) / 2
        my = (water_outs[i] + water_outs[i+1]) / 2 + 0.35
        ax.annotate('', xy=(xs[i+1], water_outs[i+1]),
                    xytext=(xs[i], water_outs[i]),
                    arrowprops=dict(arrowstyle='->', color=PALETTE['hgl_sludge'],
                                    lw=1.0, connectionstyle='arc3,rad=0.0'))
        ax.text(mx, my, f'Δh={dz:.2f} m',
                ha='center', fontsize=7, color='#4E342E',
                bbox=dict(facecolor='white', edgecolor='#A1887F',
                          alpha=0.88, boxstyle='round,pad=0.2', linewidth=0.7))

    # ── 基准线 ──
    ax.axhline(100.0, color=PALETTE['datum'], lw=0.9,
               linestyle=':', alpha=0.8, zorder=2)
    ax.text(-0.75, 100.12, '±0.000', fontsize=6.5, color=PALETTE['datum'])

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [ELEVATION_DATA[k]['name_cn'] for k in path],
        fontsize=9
    )
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylabel('Elevation  (m)', fontsize=10)
    ax.grid(axis='y', linestyle='--', color='#B0BEC5', alpha=0.4)
    ax.legend(fontsize=7.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.16),
              framealpha=0.93, edgecolor='#B0BEC5',
              ncol=3, borderpad=0.6, handlelength=1.8,
              handletextpad=0.5, columnspacing=1.0)
    ax.set_title('(b)  污泥处理工艺竖向纵剖面图（重力排泥）',
                 fontsize=11, fontweight='bold', loc='left', pad=8)


def plot_feasibility_bar(ax, df):
    """水头盈余瀑布图 — 学术样式（横向对比 + 盈余/缺口双色编码）"""
    labels   = [f"{r['上游设施']}→\n{r['下游设施']}" for _, r in df.iterrows()]
    avail    = df['可用水头(m)'].values
    req      = df['所需水头(m)'].values
    surplus  = avail - req          # 正=盈余, 负=缺口
    feasible = df['是否可行'].values
    weights  = df['段权重'].values
    n        = len(labels)
    y        = np.arange(n)

    ax.set_facecolor(PALETTE['panel_bg'])

    # ── 所需水头参考线（横向条） ──
    ax.barh(y, req, height=0.55, color='#ECEFF1',
            edgecolor='#90A4AE', linewidth=0.8, zorder=2, label='所需水头 $h_{req}$')

    # ── 可用水头（叠加显示到 req 位置之后） ──
    bar_colors = [PALETTE['surplus'] if f else PALETTE['deficit'] for f in feasible]
    ax.barh(y, avail, height=0.55, color=bar_colors,
            alpha=0.75, edgecolor='white', linewidth=0.6, zorder=3, label='可用水头 $h_{avl}$')

    # ── 盈余/缺口标注 ──
    for i, (s, f, w_seg) in enumerate(zip(surplus, feasible, weights)):
        color  = PALETTE['surplus'] if f else PALETTE['deficit']
        symbol = f'+{s:.3f} m' if f else f'{s:.3f} m'
        xpos   = avail[i] + 0.04
        ax.text(xpos, i, symbol,
                va='center', ha='left', fontsize=7.2,
                color=color, fontweight='bold')
        # 权重标记（核心段，放在轴右侧）
        if w_seg >= 2.5:
            trans = mtransforms.blended_transform_factory(
                ax.transAxes, ax.transData)
            ax.text(1.01, i, f'w={w_seg:.1f}',
                    transform=trans,
                    va='center', ha='left', fontsize=6.8,
                    color='#B71C1C', fontweight='bold')

    # ── 零基准竖线 ──
    ax.axvline(0, color='#455A64', lw=0.8, linestyle='-', zorder=5)

    # ── 需水头竖线 ──
    for i, r in enumerate(req):
        ax.vlines(r, i - 0.28, i + 0.28,
                  color=PALETTE['req'], lw=1.5, linestyle='-', zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.invert_yaxis()  # 上游在上
    ax.set_xlabel('水头  (m)', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.grid(axis='x', which='major', linestyle='--', color='#B0BEC5', alpha=0.4)
    ax.grid(axis='x', which='minor', linestyle=':', color='#CFD8DC', alpha=0.25)

    req_line  = Line2D([0],[0], color=PALETTE['req'], lw=1.5, label='所需水头阈值')
    sur_patch = mpatches.Patch(color=PALETTE['surplus'], alpha=0.75, label='盈余（重力流可行）')
    def_patch = mpatches.Patch(color=PALETTE['deficit'], alpha=0.75, label='缺口（需提升）')
    ax.legend(handles=[req_line, sur_patch, def_patch],
              fontsize=7.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.10),
              framealpha=0.93, edgecolor='#B0BEC5',
              ncol=3, borderpad=0.6, handlelength=1.8,
              handletextpad=0.5, columnspacing=1.0)
    ax.set_title('(c)  各段重力流可行性 — 水头盈余图',
                 fontsize=11, fontweight='bold', loc='left', pad=8)


def plot_elevation_table(ax, df):
    """高程参数汇总表 — 学术规范样式"""
    ax.axis('off')

    # ── 构筑物高程参数表 ──
    rows, row_colors = [], []
    TYPE_CN  = {'pump': '泵站↑', 'gravity': '重力流', 'end': '终点/储存'}
    TYPE_COL = {'pump': '#FFF3E0', 'gravity': '#E3F2FD', 'end': '#E8F5E9'}

    for code, ed in ELEVATION_DATA.items():
        extra = (f"H={ed['pump_head']:.0f} m" if ed['type'] == 'pump'
                 else f"Δh={ed.get('loss', 0):.2f} m")
        rows.append([
            ed['name_cn'], code,
            f"{ed['ground']:.2f}",
            f"{ed.get('water_depth', 0):.2f}",
            f"{ed['water_out']:.2f}",
            TYPE_CN[ed['type']], extra
        ])
        row_colors.append([TYPE_COL[ed['type']]] * 7)

    col_labels = [
        '构筑物名称', '代码',
        'GL\n(m)', 'WD\n(m)', 'HWL\n(m)',
        '工艺类型', '备注'
    ]

    tbl = ax.table(
        cellText=row_colors and rows,
        colLabels=col_labels,
        cellColours=row_colors,
        loc='upper center',
        cellLoc='center',
        bbox=[0.0, 0.12, 1.0, 0.85]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.0, 1.42)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', fontweight='bold', fontsize=8.2)
        cell.set_edgecolor('#263238')

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx > 0:
            cell.set_edgecolor('#B0BEC5')
            cell.set_linewidth(0.5)

    # ── 标题 & 图例 ──
    ax.set_title(
        '(d)  构筑物竖向高程参数汇总表\n'
        '注：GL=地面高程，WD=设计水深，HWL=出水位（高水位）；基准高程 ±0.000 = 100.00 m',
        fontsize=9.5, fontweight='bold', loc='left', pad=6
    )

    patches = [
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#D84315',
                       linewidth=0.8, label='泵站（主动提升）'),
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0',
                       linewidth=0.8, label='重力流构筑物'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32',
                       linewidth=0.8, label='终点/储存设施'),
    ]
    ax.legend(handles=patches, loc='lower center', fontsize=8,
              bbox_to_anchor=(0.5, 0.01), ncol=3,
              framealpha=0.92, edgecolor='#B0BEC5')


# ==================== 主程序 ====================
def main():
    # 1. 构建报告数据
    df = build_report()

    # 2. 控制台输出
    print("=" * 70)
    print("  供水厂竖向高程约束分析报告")
    print("  基准高程: ±0.000 = 100.00 m")
    print("=" * 70)

    # 各构筑物高程表
    print("\n【主水处理 + 污泥处理 — 构筑物高程一览】")
    print(f"{'代码':<6} {'名称':<10} {'地面高程(m)':>11} {'水深(m)':>8} {'出水位(m)':>10}  {'类型'}")
    print("-" * 58)
    for n, ed in ELEVATION_DATA.items():
        typ = {"pump": "泵站↑", "gravity": "重力流", "end": "终点/储存"}[ed["type"]]
        print(f"{n:<6} {ed['name_cn']:<10} {ed['ground']:>11.1f} "
              f"{ed.get('water_depth', 0):>8.1f} {ed['water_out']:>10.1f}  {typ}")

    # 重力流可行性判断
    print("\n【重力流可行性判断（高程阈值检查）V2.0 — 分段精细参数】")
    print(f"  全局默认沿程损失系数: {PIPE_HEADLOSS_PER_METER*1000:.1f}‰/m  "
          f"| 清水管实际: 1.0~1.2‰/m  | 污泥管实际: 3.0~5.0‰/m")
    print(f"  [修正] SB→STK 使用沉淀池底部排泥口 {SB_SLUDGE_OUTLET}m（非水侧出水位）")
    print()
    print(f"  {'上游':8} → {'下游':8}  {'坡度i':>8}  {'管长m':>6}  "
          f"{'所需m':>6}  {'可用m':>6}  {'盈余m':>6}  {'权重':>4}  判定")
    print("  " + "-" * 72)

    groups = df.groupby("流程类型")
    for gname, gdf in groups:
        print(f"  ── {gname} ──")
        for _, row in gdf.iterrows():
            status = "✅" if row["是否可行"] else f"❌ 缺{row['水头缺口(m)']:.3f}m"
            surplus_str = f"+{row['盈余(m)']:.3f}" if row["是否可行"] else "  0.000"
            print(f"  {row['上游设施']:8} → {row['下游设施']:8}  "
                  f"i={row['坡度系数i']*1000:.1f}‰  "
                  f"L={row['管线长度(m)']:>5.0f}  "
                  f"req={row['所需水头(m)']:.3f}  "
                  f"avl={row['可用水头(m)']:.3f}  "
                  f"sur={surplus_str}  "
                  f"w={row['段权重']:.1f}  {status}")
        print()

    total_shortage  = df[~df["是否可行"]]["水头缺口(m)"].sum()
    total_surplus   = df[df["是否可行"]]["盈余(m)"].sum()
    feasible_count  = df["是否可行"].sum()
    total_count     = len(df)
    print(f"  总计: {feasible_count}/{total_count} 段满足重力流  |  "
          f"总水头缺口 = {total_shortage:.3f}m  |  总盈余水头 = {total_surplus:.3f}m")

    # 3. 构建构筑物水深汇总表
    TYPE_CN  = {'pump': '泵站↑', 'gravity': '重力流', 'end': '终点/储存'}
    ZONE_MAP = {
        'IPS': '取水区',   'GC': '取水区',
        'PS':  '预处理区',
        'MB':  '常规处理区', 'FB': '常规处理区', 'SB': '常规处理区', 'FT': '常规处理区',
        'OZG': '深度处理区', 'ACF': '深度处理区',
        'CWT': '送配水区',  'BWT': '送配水区',  'PH2': '送配水区',
        'STK': '污泥处理区', 'SDR': '污泥处理区', 'SYD': '污泥处理区',
    }
    depth_rows = []
    for code, ed in ELEVATION_DATA.items():
        wd        = ed.get('water_depth', 0)
        ground    = ed['ground']
        water_out = ed['water_out']
        # 有效水深（液面到池底）
        effective_depth = wd if wd > 0 else '—'
        # 进水位（最高水面高程）
        if ed['type'] == 'pump':
            inlet_hwl = ground + wd if wd > 0 else ground
        else:
            inlet_hwl = round(water_out + ed.get('loss', 0), 3)

        # 池底高程
        if ed['type'] == 'pump':
            pool_bot_val = ground - wd if wd > 0 else '—'
        else:
            pool_bot_val = round(inlet_hwl - wd, 3) if wd > 0 else '—'

        # 埋深：池底低于地面的深度（正值=埋地, 0=齐平, 负值=高出地面）
        if isinstance(pool_bot_val, float):
            burial_depth = round(ground - pool_bot_val, 3)   # 正→地埋, 负→高出地面
        else:
            burial_depth = '—'

        # 超高：规范最小设计值 ≥ 0.30m（GB50013），此列显示设计规范值，非反算值
        freeboard = 0.30 if wd > 0 else '—'

        depth_rows.append({
            '代码':            code,
            '构筑物名称':      ed['name_cn'],
            '功能分区':        ZONE_MAP.get(code, '—'),
            '工艺类型':        TYPE_CN[ed['type']],
            '地面高程 GL(m)':  ground,
            '设计水深 WD(m)':  effective_depth,
            '进水位 HWL_in(m)':   inlet_hwl,
            '出水位 HWL_out(m)':  water_out,
            '池底高程(m)':        pool_bot_val,
            '埋深(m)\n正=地埋;负=高出地面': burial_depth,
            '内部水头损失(m)': ed.get('loss', 0) if ed['type'] != 'pump' else '—',
            '泵站扬程 H(m)':   ed.get('pump_head', '—') if ed['type'] == 'pump' else '—',
            '超高 fb(m)\n[GB50013规范值]': freeboard,
            '备注': (
                '排泥口 99.50m（见SB→STK段）' if code == 'SB' else
                '吸水坑低于地面 0.50m'         if ed['type'] == 'pump' and wd == 0 else
                f'半地埋，地面以下 {round(100.0 - ground, 1):.1f}m' if ground < 99.5 and ed['type'] == 'end' else
                '重力流终点' if ed['type'] == 'end' else ''
            )
        })
    df_depth = pd.DataFrame(depth_rows)

    # 4. 输出到Excel（多 Sheet）
    try:
        with pd.ExcelWriter("elevation_report.xlsx", engine="openpyxl") as writer:
            # Sheet1：重力流可行性报告
            df.to_excel(writer, sheet_name="重力流可行性报告", index=False)
            ws1 = writer.sheets["重力流可行性报告"]
            # 设置 Sheet1 列宽
            col_widths1 = [10, 12, 12, 14, 14, 10, 10, 12, 12, 8, 10, 6, 8, 20]
            for i, w in enumerate(col_widths1, 1):
                ws1.column_dimensions[
                    __import__('openpyxl').utils.get_column_letter(i)
                ].width = w

            # Sheet2：构筑物水深汇总表
            df_depth.to_excel(writer, sheet_name="构筑物水深汇总表", index=False)
            ws2 = writer.sheets["构筑物水深汇总表"]
            col_widths2 = [6, 14, 10, 10, 14, 12, 16, 16, 12, 16, 14, 14, 14, 30]
            for i, w in enumerate(col_widths2, 1):
                ws2.column_dimensions[
                    __import__('openpyxl').utils.get_column_letter(i)
                ].width = w

            # 冻结首行
            ws1.freeze_panes = "A2"
            ws2.freeze_panes = "A2"

            # 表头样式（深色背景 + 白字）
            from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
            header_fill  = PatternFill("solid", fgColor="37474F")
            header_font  = Font(color="FFFFFF", bold=True, size=10)
            center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
            thin_border  = Border(
                left=Side(style='thin', color='B0BEC5'),
                right=Side(style='thin', color='B0BEC5'),
                top=Side(style='thin', color='B0BEC5'),
                bottom=Side(style='thin', color='B0BEC5'),
            )
            # 分区配色
            ZONE_FILL = {
                '取水区':   'E3F2FD', '预处理区': 'E8F5E9',
                '常规处理区': 'FFF8E1', '深度处理区': 'F3E5F5',
                '送配水区':  'E0F7FA', '污泥处理区': 'FBE9E7',
            }

            for ws in [ws1, ws2]:
                for cell in ws[1]:
                    cell.fill      = header_fill
                    cell.font      = header_font
                    cell.alignment = center_align
                    cell.border    = thin_border
                ws.row_dimensions[1].height = 28

            # Sheet2 数据行：按功能分区染色 + 边框
            for row_idx, row in enumerate(ws2.iter_rows(min_row=2), start=2):
                zone_val = ws2.cell(row=row_idx, column=3).value  # 功能分区列
                fill_hex = ZONE_FILL.get(str(zone_val), 'FFFFFF')
                row_fill = PatternFill("solid", fgColor=fill_hex)
                for cell in row:
                    cell.fill      = row_fill
                    cell.alignment = center_align
                    cell.border    = thin_border
                ws2.row_dimensions[row_idx].height = 18

            # Sheet1 数据行：可行=绿，不可行=红
            ok_fill  = PatternFill("solid", fgColor="E8F5E9")
            err_fill = PatternFill("solid", fgColor="FFEBEE")
            for row_idx, row in enumerate(ws1.iter_rows(min_row=2), start=2):
                feasible_val = ws1.cell(row=row_idx, column=13).value  # 是否可行列
                row_fill = ok_fill if feasible_val else err_fill
                for cell in row:
                    cell.fill      = row_fill
                    cell.alignment = center_align
                    cell.border    = thin_border
                ws1.row_dimensions[row_idx].height = 18

        print("\n  → 已导出: elevation_report.xlsx（2个工作表）")
        print("       Sheet1: 重力流可行性报告（14段）")
        print("       Sheet2: 构筑物水深汇总表（15座构筑物）")
    except PermissionError:
        print("\n  ⚠ elevation_report.xlsx 被占用，跳过导出（请关闭 Excel 后重试）")

    # ================================================================
    # 4. 绘图 — 学术图版
    # 输出策略:
    #   • 4张独立子图（300 dpi，期刊单/双栏标准尺寸）
    #   • 1张合并总图（180 dpi，概览用）
    # ================================================================
    print("\n  正在生成高程图...")
    DPI_SINGLE = 300   # 学术期刊单图清晰度
    DPI_COMBINED = 180 # 合并总图

    # ── 学术字号约定（单图放大后字号需相应调小）──
    TITLE_FS  = 12
    LABEL_FS  = 11
    TICK_FS   = 9.5
    ANNOT_FS  = 8.5
    LEGEND_FS = 9

    # ─────────────────────────────────────────────
    # 图(a)：主水处理水力纵剖面  —  双栏全宽 190mm
    # ─────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(14, 6.0), facecolor=PALETTE['fig_bg'])
    fig_a.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.20)
    plot_main_water_elevation(ax_a)
    # 覆写字号（单图更大）
    ax_a.set_title(
        'Fig. 1(a)  Hydraulic Profile of Main Water Treatment Process\n'
        '主水处理工艺水力纵剖面图（含深度处理支路）',
        fontsize=TITLE_FS, fontweight='bold', loc='left', pad=8
    )
    ax_a.set_ylabel('Elevation  (m)', fontsize=LABEL_FS)
    ax_a.tick_params(axis='both', labelsize=TICK_FS)
    # 图例已由 plot_main_water_elevation() 内置，无需覆写
    # 期刊角注（右下）
    fig_a.text(0.97, 0.02,
               'Datum: ±0.000 = 100.00 m (Absolute)',
               ha='right', va='bottom', fontsize=8, color='#546E7A',
               style='italic')
    fig_a.savefig('elevation_fig_a_main_water.png',
                  dpi=DPI_SINGLE, bbox_inches='tight',
                  facecolor=PALETTE['fig_bg'])
    print('  → Fig.(a) 已保存: elevation_fig_a_main_water.png  (300 dpi)')
    plt.close(fig_a)

    # ─────────────────────────────────────────────
    # 图(b)：污泥处理竖向纵剖面  —  单栏 90mm
    # ─────────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(7, 5.5), facecolor=PALETTE['fig_bg'])
    fig_b.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.22)
    plot_sludge_elevation(ax_b)
    ax_b.set_title(
        'Fig. 1(b)  Sludge Treatment Profile\n'
        '污泥处理工艺竖向纵剖面图',
        fontsize=TITLE_FS, fontweight='bold', loc='left', pad=8
    )
    ax_b.set_ylabel('Elevation  (m)', fontsize=LABEL_FS)
    ax_b.tick_params(axis='both', labelsize=TICK_FS)
    # 图例已由 plot_sludge_elevation() 内置，无需覆写
    fig_b.text(0.97, 0.02,
               'Datum: ±0.000 = 100.00 m',
               ha='right', va='bottom', fontsize=8, color='#546E7A',
               style='italic')
    fig_b.savefig('elevation_fig_b_sludge.png',
                  dpi=DPI_SINGLE, bbox_inches='tight',
                  facecolor=PALETTE['fig_bg'])
    print('  → Fig.(b) 已保存: elevation_fig_b_sludge.png        (300 dpi)')
    plt.close(fig_b)

    # ─────────────────────────────────────────────
    # 图(c)：重力流可行性水头盈余图  —  单栏 90mm
    # ─────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(7, 7), facecolor=PALETTE['fig_bg'])
    fig_c.subplots_adjust(left=0.28, right=0.90, top=0.90, bottom=0.16)
    plot_feasibility_bar(ax_c, df)
    ax_c.set_title(
        'Fig. 1(c)  Gravity Flow Feasibility — Head Surplus\n'
        '各段重力流可行性水头盈余图',
        fontsize=TITLE_FS, fontweight='bold', loc='left', pad=8
    )
    ax_c.set_xlabel('Head  (m)', fontsize=LABEL_FS)
    ax_c.tick_params(axis='both', labelsize=TICK_FS)
    # 图例已由 plot_feasibility_bar() 内置，无需覆写
    fig_c.savefig('elevation_fig_c_feasibility.png',
                  dpi=DPI_SINGLE, bbox_inches='tight',
                  facecolor=PALETTE['fig_bg'])
    print('  → Fig.(c) 已保存: elevation_fig_c_feasibility.png   (300 dpi)')
    plt.close(fig_c)

    # ─────────────────────────────────────────────
    # 图(d)：构筑物高程参数汇总表  —  双栏全宽
    # ─────────────────────────────────────────────
    fig_d, ax_d = plt.subplots(figsize=(14, 6), facecolor=PALETTE['fig_bg'])
    fig_d.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.08)
    plot_elevation_table(ax_d, df)
    ax_d.set_title(
        'Fig. 1(d)  Vertical Elevation Parameters of Hydraulic Structures\n'
        '各构筑物竖向高程参数汇总表  (GL=地面高程, WD=设计水深, HWL=高水位；基准 ±0.000 = 100.00 m)',
        fontsize=TITLE_FS - 0.5, fontweight='bold', loc='left', pad=6
    )
    fig_d.savefig('elevation_fig_d_table.png',
                  dpi=DPI_SINGLE, bbox_inches='tight',
                  facecolor=PALETTE['fig_bg'])
    print('  → Fig.(d) 已保存: elevation_fig_d_table.png          (300 dpi)')
    plt.close(fig_d)

    # ─────────────────────────────────────────────
    # 合并总图（概览，180 dpi）
    # ─────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 24), facecolor=PALETTE['fig_bg'])
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.6, 2.0, 2.0],
        hspace=0.38,
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.945, bottom=0.04
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    plot_main_water_elevation(ax1)
    plot_sludge_elevation(ax2)
    plot_feasibility_bar(ax3, df)
    plot_elevation_table(ax4, df)

    fig.text(
        0.50, 0.977,
        'Vertical Elevation Constraint Analysis of Water Treatment Plant',
        ha='center', va='top', fontsize=14, fontweight='bold', color='#1A237E',
        fontfamily='Arial'
    )
    fig.text(
        0.50, 0.963,
        '供水厂竖向高程约束分析 — 主水处理及污泥处理工艺水力纵剖面图与重力流可行性评估',
        ha='center', va='top', fontsize=10, color='#37474F'
    )
    fig.add_artist(Line2D([0.05, 0.95], [0.955, 0.955],
                          transform=fig.transFigure,
                          color='#90A4AE', lw=0.8))

    fig.savefig('elevation_analysis.png', dpi=DPI_COMBINED, bbox_inches='tight',
                facecolor=PALETTE['fig_bg'])
    print('\n  → 合并总图已保存: elevation_analysis.png            (180 dpi)')
    plt.show()


if __name__ == "__main__":
    main()
