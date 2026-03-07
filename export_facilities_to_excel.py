import pandas as pd

# 设施参数数据
FACILITIES_DATA = [
    # 取水区
    ("IPS", "取水泵站", 18, 12, "intake", 1),
    ("GC", "格栅间", 10, 6, "intake", 1),
    # 预处理区
    ("PS", "预沉池", 25, 15, "pretreat", 1),
    # 常规处理区
    ("MB", "混凝池", 22, 14, "conventional", 1),
    ("FB", "絮凝池", 18, 12, "conventional", 1),
    ("SB", "沉淀池", 35, 18, "conventional", 1),
    ("FT", "滤池", 30, 16, "conventional", 1),
    # 深度处理区
    ("OZG", "臭氧接触池", 20, 12, "advanced", 2),
    ("ACF", "活性炭滤池", 25, 15, "advanced", 1),
    ("OZR", "臭氧发生间", 12, 10, "advanced", 3),
    # 消毒加药区
    ("CDR", "加药间", 14, 10, "chemical", 2),
    ("CLS", "加氯间", 10, 8, "chemical", 3),
    ("CST", "药剂仓库", 15, 10, "chemical", 2),
    # 送配水区
    ("CWT", "清水池", 35, 25, "distribution", 1),
    ("BWT", "反冲洗水池", 15, 10, "distribution", 1),
    ("PH1", "一级泵房", 15, 10, "distribution", 1),
    ("PH2", "二级泵房", 20, 14, "distribution", 1),
    # 动力区
    ("PDR", "配电室", 18, 12, "power", 2),
    ("TRF", "变压器室", 15, 10, "power", 3),
    # 污泥处理区
    ("STK", "污泥浓缩池", 18, 12, "sludge", 1),
    ("SDR", "污泥脱水机房", 20, 12, "sludge", 1),
    ("SYD", "污泥堆场", 20, 15, "sludge", 1),
    # 行政办公区
    ("CR", "中控室", 18, 12, "admin", 1),
    ("OF", "综合办公楼", 25, 12, "admin", 1),
    ("GT", "门卫室", 6, 4, "admin", 1),
    ("LAB", "化验室", 12, 10, "admin", 1),
    # 辅助设施区
    ("WH", "综合仓库", 18, 10, "auxiliary", 1),
    ("PK", "停车场", 20, 12, "auxiliary", 1),
    ("MW", "维修车间", 18, 12, "auxiliary", 1),
    # 生活区
    ("DM1", "宿舍楼1", 15, 10, "living", 1),
    ("DM2", "宿舍楼2", 15, 10, "living", 1),
    ("DM3", "宿舍楼3", 15, 10, "living", 1),
    ("DM4", "宿舍楼4", 15, 10, "living", 1),
    ("SPF", "运动场", 30, 20, "living", 1),
]

columns = ["代码", "中文名", "长度(m)", "宽度(m)", "功能分区", "安全等级"]
df = pd.DataFrame(FACILITIES_DATA, columns=columns)
df.to_excel("facilities.xlsx", index=False)
print("已导出为 facilities.xlsx")
