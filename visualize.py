import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("开始加载 A5 数据（9.2M 行，可能需要 1-2 分钟）...")
t_start = time.time()

df = pd.read_csv('a5_drought_output.txt', sep='\t', header=None,
                 names=['station_id', 'record_date', 'precip', 'drought_level'],
                 dtype={'station_id': str})
print(f"A5 数据加载完成，共 {len(df):,} 行，耗时 {time.time()-t_start:.1f} 秒")

# 预处理
df['record_date'] = pd.to_datetime(df['record_date'])
df['year'] = df['record_date'].dt.year
df['month'] = df['record_date'].dt.month
df['is_severe'] = df['drought_level'].isin(['特旱', '重旱'])
df['year_month'] = df['record_date'].dt.to_period('M')

print("预处理完成")

# 加载 A4 数据
t_a4 = time.time()
a4_df = pd.read_csv('a4_drought_profile.txt', sep='\t', header=None,
                    names=['station_id', 'dry_event_count', 'max_dry_days', 'avg_dry_days', 'total_dry_days'])
print(f"A4 数据加载完成，共 {len(a4_df):,} 行，耗时 {time.time()-t_a4:.1f} 秒")

# 图1：干旱等级分布 + Top 20 最干旱站点
print("生成图1...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

level_order = ['特旱', '重旱', '中旱', '轻旱', '无旱']
level_colors = ['#8B0000', '#CD5C5C', '#F4A460', '#FFD700', '#90EE90']
level_counts = df['drought_level'].value_counts()
level_pct = level_counts.reindex(level_order) / len(df) * 100
axes[0].bar(level_order, level_pct, color=level_colors, edgecolor='black')
axes[0].set_ylabel('占比 (%)')
axes[0].set_title('A5 干旱等级分布')
for i, v in enumerate(level_pct):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
axes[0].axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='理论值 10%')
axes[0].axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='理论值 60%')
axes[0].legend()

severe_df = df[df['is_severe']]
if not severe_df.empty:
    station_severity = severe_df.groupby('station_id')['precip'].mean().sort_values().head(20)
    axes[1].barh(range(20), station_severity.values, color='firebrick', edgecolor='black')
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([str(i)[:12] for i in station_severity.index])
    axes[1].set_xlabel('特旱+重旱期间平均降水量 (mm)')
    axes[1].set_title('Top 20 最干旱站点')
    axes[1].invert_yaxis()
else:
    axes[1].text(0.5, 0.5, '没有特旱/重旱记录', transform=axes[1].transAxes, ha='center')
plt.tight_layout()
plt.savefig('a5_drought_distribution.png', dpi=150)
plt.close()

# 图2：年度干旱柱状图 + 月度干旱强度热力图
print("生成图2...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

yearly_level = df.groupby(['year', 'drought_level']).size().unstack(fill_value=0)
yearly_level = yearly_level.reindex(columns=level_order, fill_value=0)
yearly_level.plot(kind='bar', stacked=True, ax=axes[0], color=level_colors)
axes[0].set_xlabel('年份')
axes[0].set_ylabel('记录数')
axes[0].set_title('各年度干旱等级分布')
axes[0].legend(loc='upper right', ncol=5)

monthly_severe = df[df['is_severe']].groupby('year_month')['precip'].mean()
if not monthly_severe.empty:
    heat_data = monthly_severe.reset_index()
    heat_data['year'] = heat_data['year_month'].dt.year
    heat_data['month'] = heat_data['year_month'].dt.month
    pivot = heat_data.pivot(index='month', columns='year', values='precip')
    im = axes[1].imshow(pivot, aspect='auto', cmap='YlOrRd_r', interpolation='nearest')
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels(pivot.columns)
    axes[1].set_yticks(range(1,13))
    axes[1].set_yticklabels(range(1,13))
    axes[1].set_xlabel('年份')
    axes[1].set_ylabel('月份')
    axes[1].set_title('月度严重干旱（特旱+重旱）平均降水量 (mm)')
    plt.colorbar(im, ax=axes[1], label='平均降水量 (mm)')
plt.tight_layout()
plt.savefig('a5_temporal_drought.png', dpi=150)
plt.close()

# 图3：A4 vs A5 对比
print("生成图3...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

a4_dist = a4_df['max_dry_days'].value_counts().sort_index()
axes[0].bar(a4_dist.index[:15], a4_dist.values[:15], color='steelblue', edgecolor='black')
axes[0].set_xlabel('最长连续干旱天数')
axes[0].set_ylabel('站点数')
axes[0].set_title('A4：绝对阈值 (precip < 0.5mm)')
axes[0].set_yscale('log')
zero_cnt = a4_dist.get(0, 0)
zero_pct = zero_cnt / len(a4_df) * 100
axes[0].annotate(f'{zero_cnt} 站点无干旱\n占比 {zero_pct:.1f}%', xy=(0, zero_cnt),
                 xytext=(3, zero_cnt*0.8), arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)

station_extreme_pct = df.groupby('station_id')['drought_level'].apply(lambda x: (x == '特旱').mean()) * 100
axes[1].hist(station_extreme_pct, bins=40, color='firebrick', edgecolor='black', alpha=0.7)
axes[1].axvline(x=10, color='gold', linestyle='--', linewidth=2, label='理论值 10%')
axes[1].set_xlabel('特旱记录占比 (%)')
axes[1].set_ylabel('站点数')
axes[1].set_title('A5：相对阈值（基于站点百分位数）')
axes[1].legend()
plt.tight_layout()
plt.savefig('a4_vs_a5_comparison.png', dpi=150)
plt.close()

print("所有图表生成完毕！")
print("生成的文件：")
print("  - a5_drought_distribution.png")
print("  - a5_temporal_drought.png")
print("  - a4_vs_a5_comparison.png")