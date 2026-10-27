"""Create an auditable Chinese result summary from saved experiment outputs."""
import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
out=root/'results/basic_v2'
config=json.loads((out/'config.json').read_text())
m=pd.read_csv(out/'mechanism.csv')
v=pd.read_csv(out/'theory_validation.csv')
a=pd.read_csv(out/'array.csv')
s=pd.read_csv(out/'budget_selection.csv')
r=pd.read_csv(out/'reliability.csv')
pd.DataFrame([dict(parameter=k,value=json.dumps(value,ensure_ascii=False)) for k,value in config.items()]).to_csv(out/'parameters.csv',index=False)
start,end=m.iloc[0],m.iloc[-1]
mid=m.iloc[(m.eta-1).abs().argmin()]
gain=100*(1-mid.aoii/mid.aoii_direct)
p2=s[(s.problem=='P2')&(s.threshold==.05)].iloc[0]
small=r[r.Ns==200].set_index('method')
coverage=int(((v.simulation-v.theory).abs()<=v.halfwidth95).sum())
text=f'''# 基础实验结果与结论（basic_v2）

本报告由 summarize_basic.py 从 CSV 自动生成。原始数据：KU Leuven MaMIMO-UAV，doi:10.48804/0IMQDF；使用一段 50 秒 64 通道记录。500 个位置/CSI 对，空间留出 102 个 CSI，7 个训练 cell 波束，3 个公开场景。不是跨场景部署验证。

## 参数与复现

- 参数表：parameters.csv；完整配置：config.json；冻结模型与精确参考：empirical_model.npz。
- 完整执行：`python Experiments/run_basic_experiments.py`；摘要生成：`python Experiments/summarize_basic.py`。
- 每个源验证配置 {config['source_slots']:,} 个有效时隙，批长 {config['source_batch']:,}；主机制每场景每预算 {config['mechanism_N']:,} 样本。
- 主预算选择从共享物理扰动和 CSI 抽样的前 {config['Ns']:,} 个样本获得；不是从精确 q 直接选最优。独立确认用 {config['Nc']:,} 个等价 IID Bernoulli 样本的二项计数。
- 参考 SNR 仅用训练集能量标定。此版本取代 pilot 的测试集标度，所以绝对数值与 pilot 有变化，不应混用。

## 图组一：机制与理论验证

- eta 从 {start.eta:g} 增加至 {end.eta:g}，中断由 {start.q_exact:.6f} 降至 {end.q_exact:.6f}；AoII/T 由 {start.aoii:.6f} 降至 {end.aoii:.6f}。
- eta=1 下 MAP AoII/T={mid.aoii:.6f}，直接报告查询为 {mid.aoii_direct:.6f}，同预算相对下降 {gain:.3f}%。
- 强隐私端，MAP 平均中断={start.q_exact:.6f}，直接查询={start.q_direct:.6f}；但 MAP 平均 AoII={start.aoii:.6f} 小于直接查询的 {start.aoii_direct:.6f}。这是逐场景非线性聚合现象，不说明 MAP 普遍最优。
- {int((v.kind=='controlled').sum())} 组受控源验证最大相对偏差 {100*v[v.kind=='controlled'].relative_error.max():.3f}%；9 组实测模型边际验证最大相对偏差 {100*v[v.kind=='measured_marginal'].relative_error.max():.3f}%；全部配置最大绝对偏差 {abs(v.theory-v.simulation).max():.6f} 时隙。
- {coverage}/{len(v)} 个点落入各自点态 95% 批均值区间。这些区间不是 {len(v)} 点同时覆盖区间；个别偏离不能通过反复改种子消除。

## 图组二：阵列规模

采用单独的 75 位置几何 Rician 模型，绝不将未核实天线顺序的通道子集称为实测物理子阵列。半波长 UPA、kappa=10、路径损耗指数2；发射功率固定与参考对齐 SNR 固定分别比较。
'''
for mode in ['fixed_power','equal_aligned_snr']:
    b=a[(a['mode']==mode)&(a.eta==.1)].set_index('M')
    text+=f"\n- {mode}：eta=0.1 时 M=16 的 AoII/T={b.loc[16,'aoii']:.6f}，M=256 为 {b.loc[256,'aoii']:.6f}。\n"
text+=f'''
M=1 对预算不敏感；额外 Rayleigh 控制也方向无关。大阵列的准确对齐增益不意味着隐私约束下总占优势。几何结果的极低中断包含数值 CDF 下溢，不能称为实证零中断；MAP 误判参考含有限 Monte Carlo 分辨率。

## 图组三：预算选择与可靠性

- 四个 P1 隐私上限下均选择最大可行网格预算，与精确条件模型的网格最优一致；所有 G 和确认区间见 budget_selection.csv。不声称存在内点最优。
- P2 Dmax/T=0.05：选择 eta={p2.eta:.6f}，精确条件 AoII={p2.J_exact:.6f}，独立确认区间 [{p2.L_confirm:.6f}, {p2.U_confirm:.6f}]，状态={p2.status}。
- Ns=200、{config['repeats']} 次宏重复：点估计法违反率 {100*small.loc['point_estimate','violation_rate']:.1f}%；LAPIS 观测违反率 {100*small.loc['LAPIS','violation_rate']:.1f}%，但零次违反的双侧 95% 二项区间上端为 {100*small.loc['LAPIS','false_issue_upper95']:.3f}%，不是零风险。
- 对应平均预算：LAPIS={small.loc['LAPIS','mean_eta']:.6f}，点估计={small.loc['point_estimate','mean_eta']:.6f}。可靠性以额外披露为代价；样本量增加后两者趋近条件模型的可行边界。
- 可靠性实验用精确 q 的等价 Bernoulli 边际与独立预算流，以隔离区间选择效果。它不测量共享物理随机数的方差降低收益。两方法使用同一组选择估计，确认样本独立且不用于重选。

## 精确参考与适用范围

三维 Laplace 噪声在任意单位方向上的投影 t，其 CDF 在 t>=0 时为 1-(epsilon*t+2)*exp(-epsilon*t)/4，在负半轴对称。三个等先验共线位置的 MAP 边界为中点，故转移概率可解析积分。将其与有限 CSI bank 中逐场景/逐波束的经验中断率相乘，得到该条件经验模型的精确 q。

误差分层：MAP 积分在此布局下解析；有限 CSI bank 被固定为经验信道分布；Monte Carlo 区间只涉及这一分布下的随机抽样。真实 CSI 有时间相关性，独立 bootstrap 不验证原轨迹时序；GPS、空间 cell、图外查询及地图泛化误差不受 Hoeffding 区间覆盖。

当前结论可以用于会议论文的基础数值章节，但尚不能推广为任意 UAV 环境或经标定的真实吞吐/时延改善。下一阶段优先扩展独立航段与空间覆盖、校准信道功率、地图分辨率/先验敏感性，而不是增加动态预算或多基站。
'''
(out/'Results-Summary.md').write_text(text,encoding='utf-8')
print('Wrote',out/'Results-Summary.md')
