#!/usr/bin/env python3
"""按论文结论组织卫星证据核验；区分原始记录复算与当前环境重新执行。

本入口不修改原始模型，不读取 V8 进行特征选择，不将历史记录冒充本次搜索。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import traceback

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'evidence/satellite/raw_per_seed'
OUTPUT = ROOT / 'results/reproduced/satellite/verification'
PUBLISHED = {
    'power': {'D': 267, 'msf': .8054, 'random': .7222, 'delta': .0832},
    'thermal': {'D': 135, 'msf': .8807, 'random': .8343, 'delta': .0464},
    'attitude': {'D': 135, 'msf': .9195, 'random': .8814, 'delta': .0380},
    'data': {'D': 204, 'msf': .9511, 'random': .8245, 'delta': .1266},
}


def check_archived_table4():
    """使用实际逐种子 CSV 重算，而不是读取汇总表后直接宣布成功。"""
    rows, failed = [], []
    for subsystem, paper in PUBLISHED.items():
        p = DATA / subsystem / 'per_seed_results.csv'
        with p.open(encoding='utf-8-sig', newline='') as f:
            raw = list(csv.DictReader(f))
        seeds = [int(r['seed']) for r in raw]
        if len(raw) != 10 or sorted(seeds) != list(range(192, 202)):
            failed.append(f'{subsystem}: 应为192–201共10个唯一随机种子，实际{seeds}')
        msf = [float(r['msf_hv']) for r in raw]
        rnd = [float(r['random_hv']) for r in raw]
        deltas = [x - y for x, y in zip(msf, rnd)]
        wins = sum(d > 1e-12 for d in deltas)
        ties = sum(abs(d) <= 1e-12 for d in deltas)
        losses = sum(d < -1e-12 for d in deltas)
        if len(deltas) != 10 or (wins, ties, losses) != (10, 0, 0):
            failed.append(f'{subsystem}: 不是论文的10/0/0，实际{wins}/{ties}/{losses}')
        raw_p = float(wilcoxon(msf, rnd, alternative='two-sided', method='exact').pvalue)
        stat = {
            'subsystem': subsystem, 'candidateD': paper['D'], 'seeds': seeds,
            'msfMean': statistics.mean(msf), 'msfStd': statistics.stdev(msf),
            'randomMean': statistics.mean(rnd), 'randomStd': statistics.stdev(rnd),
            'deltaHV': statistics.mean(deltas), 'wins': wins, 'ties': ties,
            'losses': losses, 'wilcoxonPRaw': raw_p,
            'sourceCSV': os.path.relpath(p, ROOT),
            'sourceSHA256': hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for name, observed in [('msf', stat['msfMean']), ('random', stat['randomMean']), ('delta', stat['deltaHV'])]:
            if abs(observed - paper[name]) > .00015:
                failed.append(f'{subsystem}: {name}重算值{observed:.6f}不吻合论文{paper[name]:.4f}')
        rows.append(stat)
    # Holm 步进校正，原始 p 升序；确保全部记录都属于论文预设的四项检验族。
    ranked = sorted(enumerate(rows), key=lambda x: x[1]['wilcoxonPRaw'])
    prev = 0.0
    for rank, (index, row) in enumerate(ranked):
        prev = max(prev, min(1.0, (len(rows) - rank) * row['wilcoxonPRaw']))
        rows[index]['holmP'] = prev
        if abs(prev - .0078125) > 1e-9:
            failed.append(f"{row['subsystem']}: Holm p={prev}，论文目标0.0078125")
    return {'evidenceClass': '历史逐种子记录独立统计复算（不是重做40次优化搜索）',
            'sourceRecords': sum(len(r['seeds']) for r in rows),
            'all40Wins': sum(r['wins'] for r in rows) == 40,
            'allPass': not failed, 'rows': rows, 'errors': failed}


def check_frozen_model_hash():
    """把历史冻结模型的真实字节哈希与正式140特征配置绑定。"""
    p = ROOT / 'src/satellite/health-v2-service/core/models/subsystem-health-model-v6.json'
    conf = json.loads((ROOT/'configs/frozen_140_features.json').read_text(encoding='utf-8'))
    observed = hashlib.sha256(p.read_bytes()).hexdigest()
    expected = conf['freezeSourceSHA256']
    return {'evidenceClass': '原版冻结模型完整字节SHA-256核验', 'pass': observed == expected,
            'sourceSHA256': observed, 'expectedSHA256': expected}


def check_v8_dataset_manifest():
    """核对确认数据元信息，不将 manifest 当作重新生成数据的证明。"""
    p = ROOT / 'data/satellite/v8_confirmation/telemetry-msf-confirmation-v8.csv.manifest.json'
    x = json.loads(p.read_text(encoding='utf-8'))
    ok = (x.get('scenarios') == 411 and x.get('rows') == 9042 and x.get('featureCount') == 582)
    return {'evidenceClass': '数据集清单核查（非独立重算全部窗口）',
            'scenarios': x.get('scenarios'), 'windows': x.get('rows'),
            'features': x.get('featureCount'), 'pass': ok}


def execute(name, command, out_dir):
    """执行真实子程序并留存日志；失败不得写成通过。"""
    print(f'\n[卫星论文验证] 开始：{name}', flush=True)
    try:
        p = subprocess.run([str(x) for x in command], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src') + os.pathsep + os.environ.get('PYTHONPATH', '')},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        output = p.stdout or ''
        (out_dir / f'{name}.log').write_text(output, encoding='utf-8')
        if p.returncode != 0:
            print(output[-1800:], flush=True)
        else:
            print(f'  完成，退出码0；日志：{name}.log', flush=True)
        return {'status': '通过' if p.returncode == 0 else '失败',
                'exitCode': p.returncode, 'log': f'{name}.log'}
    except Exception as e:
        (out_dir / f'{name}.log').write_text(traceback.format_exc(), encoding='utf-8')
        return {'status': '失败', 'error': str(e), 'log': f'{name}.log'}


def check_v8_results(out_dir):
    """核对真实执行生成的 V8 故障确认报告，绝不使用历史 JSON 顶替。"""
    base = ROOT / 'results/reproduced/satellite/v8_confirmation/experiments'
    a = json.loads((base/'health_confirmation.json').read_text(encoding='utf-8'))
    b = json.loads((base/'trajectory_confirmation.json').read_text(encoding='utf-8'))
    checks = {
        '正常场景 30': a['normal']['scenarios'] == 30,
        '正常场景误报 2': a['normal']['falseAlarmScenarios'] == 2,
        '单故障识别 72/72': a['univariate']['scenarios'] == 72 and a['univariate']['detected'] == 72,
        '耦合故障识别 35/45': a['coupled']['scenarios'] == 45 and a['coupled']['detected'] == 35,
        '渐进退化提前预警 12/12': b['scenarios'] == 12 and b['earlyWarnings'] == 12,
    }
    return {'evidenceClass': '本次重跑 V8 路径适配版脚本（历史两脚本哈希与发布清单不一致）',
            'checks': checks, 'allPass': all(checks.values()),
            'falseAlarms': a['normal']['falseAlarmScenarios'],
            'univariateDetected': a['univariate']['detected'],
            'coupledDetected': a['coupled']['detected'],
            'earlyWarnings': b['earlyWarnings'],
            'outputFiles': [str((base/n).relative_to(ROOT)) for n in ('health_confirmation.json','trajectory_confirmation.json')]}


def write_report(report, out_dir):
    with (out_dir/'report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    t = report['table4']
    lines = ['# 卫星应用：论文结论核验报告', '', f"执行时间（UTC）：{report['generatedUTC']}", '',
        '## 论文 Table 4：MSF 与 Random 的四子系统配对', '',
        f"证据类型：**{t['evidenceClass']}**。", '',
        '| 子系统 | 维度 | MSF HV | Random HV | ΔHV | W/T/L | Holm p |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in t['rows']:
        lines.append(f"| {x['subsystem']} | {x['candidateD']} | {x['msfMean']:.4f} | {x['randomMean']:.4f} | +{x['deltaHV']:.4f} | {x['wins']}/{x['ties']}/{x['losses']} | {x['holmP']:.6f} |")
    lines += ['',f"40条逐种子记录统计复算：{'通过' if t['allPass'] else '失败'}。**本命令默认不重新执行40次搜索**。",
        f"历史冻结模型SHA-256：{'吻合' if report['frozenModel']['pass'] else '不吻合'}。", '',
        '## 原始 V6 冻结特征与在线部署', '']
    for name, result in report['runs'].items():
        lines.append(f"- **{name}**：{result['status']}（详见 `{result.get('log','')}`）")
    benchmark=report.get('newBenchmark')
    if benchmark:
        lines += ['', f"本次独立计时：582维中位数 {benchmark['full']['medianUs']:.2f} μs；140维 {benchmark['frozen']['medianUs']:.2f} μs；下降 {benchmark['medianReduction']*100:.2f}%；加速 {benchmark['medianSpeedup']:.2f} 倍。",'']
    lines += ['历史论文记载 556.11 → 223.58 μs/frame（59.8%），**属于历史环境数据，不要求新机器数值相同**。', '',
        '## V8 冻结健康确认', '']
    if 'v8' in report:
        lines.append('本轮实际运行了 V8 的**路径适配版**健康确认与轨迹脚本。历史冻结哈希中对应两脚本不匹配，因此不能表述为逐字节原版脚本核验。')
        for k, v in report['v8']['checks'].items():lines.append(f'- {k}：{"符合历史证据" if v else "不一致"}')
    else:
        lines.append('本轮未重跑 V8 健康确认；若需执行请加 `--with-v8`。V8 数据元信息清单不是健康确认指标的新运行证据。')
    lines += ['', '## 研究结论和边界', '',
        '- Table 4 统计复算来自历史40条配对记录；重新从零运行请使用 `python run.py satellite rerun-table4`。',
        '- 140/582 映射和对应特征计算由原版 V6 模型及 Node.js 实际程序验证；不借助 V8 重新选特征。',
        '- 新计时使用当次 CPU/Node 环境和原始逐次记录；实验通过不等于精确复现历史59.8%。',
        '- V8 场景为**合成遥测**，不能扩展解释为真实在轨精度。',
        '- 历史 V8 数据集清单为 411 场景、9042 窗口；V8 健康仿真确认的30/72/45案例与该训练/验证窗口库不同。',
        '- 完整从零重跑公开数据和40条卫星搜索需要另行逐项执行，状态不能自动升级为“已完成”。',
        '', f"总体：**{report['overallStatus']}**。", '']
    (out_dir/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description='统一核验论文卫星结论：Table4逐种子复算、冻结模型、在线联调、独立计时；可选V8')
    parser.add_argument('--repeats',type=int,default=200,help='本次独立计时次数（至少10）')
    parser.add_argument('--warmup',type=int,default=50,help='本次独立计时预热次数')
    parser.add_argument('--with-v8',action='store_true',help='额外重跑V8确认健康与渐进退化轨迹模拟，可能耗时较长')
    parser.add_argument('--archive-only',action='store_true',help='仅重算历史Table4与V8数据清单，不执行Node，适用于快速审计')
    parser.add_argument('--out',type=Path,default=OUTPUT,help='输出目录，默认 results/reproduced/satellite/verification/')
    a=parser.parse_args(argv)
    if a.repeats<10 or a.warmup<0:parser.error('repeats>=10 且 warmup>=0')
    if a.with_v8 and a.archive_only:parser.error('--archive-only 不可同时设置 --with-v8')
    out_dir=a.out.resolve();out_dir.mkdir(parents=True,exist_ok=True)
    report={'schema':'satellite-paper-claims-check-v1',
        'generatedUTC':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':sys.version.split()[0],'system':platform.platform()},
        'table4':check_archived_table4(), 'frozenModel':check_frozen_model_hash(),
        'v8DatasetManifest':check_v8_dataset_manifest(), 'runs':{},
        'scope':'历史统计复算' if a.archive_only else '历史统计复算+本机模型/功能/性能实测'}
    if not a.archive_only:
        report['runs']['冻结140特征值一致性'] = execute('frozen140',[ 'node',ROOT/'scripts/satellite/verify_frozen_features.js'],out_dir)
        report['runs']['V6在线服务91帧联调'] = execute('smoke91',['node',ROOT/'src/satellite/msf-training/integration-smoke-test-v6.js'],out_dir)
        bench_out = out_dir/'benchmark'
        report['runs']['582对140维独立计时'] = execute('benchmark',['node',ROOT/'scripts/satellite/benchmark_frozen_features.js',
            '--repeats',a.repeats,'--warmup',a.warmup,'--out',bench_out],out_dir)
        if report['runs']['582对140维独立计时']['status']=='通过':
            report['newBenchmark']=json.loads((bench_out/'summary.json').read_text(encoding='utf-8'))
        if a.with_v8:
            for key,script in [('V8健康故障确认','run_v8_health_confirmation.js'),('V8渐进退化确认','run_v8_trajectory_confirmation.js')]:
                report['runs'][key]=execute(key,['node',ROOT/'scripts/satellite'/script],out_dir)
            if all(report['runs'][k]['status']=='通过' for k in ('V8健康故障确认','V8渐进退化确认')):
                report['v8']=check_v8_results(out_dir)
    passall=report['table4']['allPass'] and report['frozenModel']['pass'] and report['v8DatasetManifest']['pass'] and all(r['status']=='通过' for r in report['runs'].values())
    if 'v8' in report:passall=passall and report['v8']['allPass']
    report['overallStatus']='本轮已执行项目通过（不代表所有论文搜索已全量重跑）' if passall else '发现不通过项目；请查看日志'
    write_report(report,out_dir)
    print('\nTable 4 历史配对统计复算：', '通过' if report['table4']['allPass'] else '失败', flush=True)
    print('历史冻结模型 SHA-256：', '通过' if report['frozenModel']['pass'] else '失败',flush=True)
    print('V8 确认数据清单：', '通过' if report['v8DatasetManifest']['pass'] else '失败', flush=True)
    print('汇总报告：',out_dir/'report.md',flush=True)
    print('机器可读：',out_dir/'report.json',flush=True)
    print('判定：',report['overallStatus'],flush=True)
    return 0 if passall else 1

if __name__=='__main__':sys.exit(main())
