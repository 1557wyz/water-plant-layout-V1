from water_plant_layout_v3 import Facility, FACILITIES_DATA, AdvancedGA, PIPELINE_SYSTEM

# 创建设施对象
facilities = [Facility(*args) for args in FACILITIES_DATA]

# 初始化算法
ga = AdvancedGA(facilities, population_size=80, generations=50)

# 运行完整优化，保存最优解
best_fitness = float('inf')
best_chrom = None
for _ in range(ga.generations):
    chrom = ga.create_chromosome()
    area, pipe_length, adj_penalty, utilization = ga._calculate_multi_objective(chrom)
    fitness = area + pipe_length + adj_penalty - utilization * 100
    if fitness < best_fitness:
        best_fitness = fitness
        best_chrom = chrom
placed_best, w2, h2 = ga.decode_chromosome(best_chrom, (sum(f.area for f in facilities)) ** 0.5 * 1.8)

# 计算每种管线的长度
fdict = {f.name: f for f in placed_best}
centers = {f.name: (f.x + f.get_dimensions()[0]/2, f.y + f.get_dimensions()[1]/2) for f in placed_best}
pipe_lengths = {}
for system_name, system_data in PIPELINE_SYSTEM.items():
    total = 0
    for n1, n2 in system_data["pipes"]:
        if n1 in centers and n2 in centers:
            cx1, cy1 = centers[n1]
            cx2, cy2 = centers[n2]
            total += abs(cx2 - cx1) + abs(cy2 - cy1)
    pipe_lengths[system_name] = (system_data.get('label', system_name), total)


import pandas as pd
# 导出为Excel表格
df = pd.DataFrame([
    {'管线类型': label, '长度(m)': v}
    for label, v in pipe_lengths.values()
])
df.to_excel('pipeline_lengths.xlsx', index=False)
print('已导出为 pipeline_lengths.xlsx')
