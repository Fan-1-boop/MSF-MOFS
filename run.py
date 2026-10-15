#!/usr/bin/env python3
"""MSF-MOFS 全项目统一运行入口。

命令行只负责任务组织，不包含新的优化器公式；所有原始结果都只读保存。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
SAT=ROOT/'src/satellite'
NODE='node'

def call(args, *, cwd=ROOT):
    """执行子任务；任何子命令失败都向上传递退出码，禁止误报成功。"""
    cmd=[str(x) for x in args]
    print('执行：', ' '.join(cmd),flush=True)
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    subprocess.run(cmd,cwd=cwd,env=env,check=True)

def verify_model_hash():
    """核查历史模型 SHA 与冻结清单绑定关系。"""
    model_file=SAT/'health-v2-service/core/models/subsystem-health-model-v6.json'
    plan=json.loads((ROOT/'configs/frozen_140_features.json').read_text('utf-8'))
    observed=hashlib.sha256(model_file.read_bytes()).hexdigest()
    expected=plan['freezeSourceSHA256']
    if observed!=expected:raise RuntimeError(f'原始 V6 冻结模型哈希不匹配: {observed} != {expected}')
    print('原始 V6 冻结模型 SHA-256 核验通过:',observed)

def v6_workspace(dataset_file:Path):
    """在可写的 _local 中准备 V6 工作目录，不覆盖历史 JSON、模型或原始记录。"""
    target=ROOT/'results/_local/v6_pipeline/msf-training'
    target.mkdir(parents=True,exist_ok=True)
    source=SAT/'msf-training'
    for name in ('train_msf_mofs.py','run_multiseed_experiment.py','train_subsystem_health_model_v6.py',
                 'build_subsystem_datasets_v6.py','run_sparse_control_v6.py'):
        shutil.copy2(source/name,target/name)
    for file in source.glob('experiments/v6_final/*/experiment_results.json'):
        t=target/'experiments/v6_final'/file.parent.name/file.name
        t.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,t)
    # 训练输入必须由 V6 开发数据产生，不得替换成 V8 确认数据。
    call([sys.executable,target/'build_subsystem_datasets_v6.py','--data',dataset_file,'--out-dir',target])
    return target

def main(argv=None):
    parser=argparse.ArgumentParser(description='MSF-MOFS 可复现项目：统一入口（所有命令提供中文使用说明）')
    sub=parser.add_subparsers(dest='command',required=True)
    for command,desc in {
        'doctor':'查看 Python、Node.js 和关键文件是否齐全',
        'test':'执行 21 项 Python 单元测试及 V6 特征正确性测试',
        'verify':'运行测试、冻结模型哈希和公开数据历史证据审计',
        'satellite-smoke':'启动原版 V6 健康服务进行 91 帧在线联调',
        'benchmark':'新环境独立计时；输出写入 results/_local/',
        'v6-generate':'通过原始 V6 生成器重新生成 354 场景/7080 窗口',
        'v6-prepare':'从 V6 开发数据构造六子系统训练集，准备独立工作区',
        'v6-search':'对指定子系统执行 V6 MSF 与 Random 多随机种子搜索（较慢）',
        'v6-refit':'使用原始选择结果重新拟合模型到 _local（不会覆盖原模型）',
        'v8-health':'运行 V8 冻结健康确认评估（较慢，不参与选特征）',
        'v8-trajectory':'运行 V8 冻结退化轨迹确认评估（较慢）',
        'paper':'论文公开数据实验主入口（推荐）：查看子命令请运行 python run.py paper --help',
        'satellite':'卫星论文结论统一核验入口：verify / archive / rerun-table4',
        'public':'paper 的兼容别名：公开数据实验旧入口',
        'integrity':'核查最终交付文件哈希（建议新解压时运行）',
        'all':'完整快速回归：验证、91帧集成测试和短计时（不是全部优化搜索）',
    }.items():sub.add_parser(command,help=desc)
    for c in ('benchmark','public','paper','v6-search','satellite'):
        sub.choices[c].add_argument('args',nargs=argparse.REMAINDER,help='透传给子工具的参数')
    # 实验子工具拥有自己的 --seed / --repeats 等选项，不能让顶层解析器抢先拒绝。
    # 解析顶层命令名称后，把该命令后面的参数原样移交子命令。
    supplied=list(sys.argv[1:] if argv is None else argv)
    if supplied and supplied[0] in ('benchmark','public','paper','v6-search','satellite'):
        args=parser.parse_args([supplied[0]])
        args.args=supplied[1:]
    else:
        args=parser.parse_args(supplied)
    if args.command=='integrity':
        call([sys.executable,ROOT/'scripts/verify_integrity.py'])
    if args.command=='doctor':
        print('Python:',sys.version.split()[0]);call(['node','--version'])
        print('原始模型、Python 核心、论文实验数据已打包。')
        verify_model_hash()
    if args.command in ('test','verify','all'):
        verify_model_hash()
        call([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v'])
        call([NODE,ROOT/'scripts/satellite/verify_frozen_features.js'])
    if args.command in ('verify','all'):
        call([sys.executable,'-m','msf_repro','check'])
    if args.command in ('satellite-smoke','all'):
        call([NODE,SAT/'msf-training/integration-smoke-test-v6.js'])
    if args.command=='benchmark':
        call([NODE,ROOT/'scripts/satellite/benchmark_frozen_features.js',*args.args])
    if args.command=='all':
        call([NODE,ROOT/'scripts/satellite/benchmark_frozen_features.js','--repeats','200','--warmup','50'])
    if args.command=='v6-generate':
        out=ROOT/'results/_local/v6_pipeline/telemetry-msf-v6.csv'
        out.parent.mkdir(parents=True,exist_ok=True)
        call([NODE,SAT/'satellite-datastream-service/generate-training-data-v6.js','--out',out])
    if args.command in ('v6-prepare','v6-search','v6-refit'):
        local=ROOT/'results/_local/v6_pipeline/telemetry-msf-v6.csv'
        packaged=ROOT/'data/satellite/v6_development/v6_development_regenerated.csv'
        dataset=local if local.exists() else packaged
        if args.command=='v6-search' and not any(t in args.args for t in ['--subsystem']):
            raise SystemExit('v6-search 必须指定 --subsystem power|thermal|attitude|data|communication|propulsion')
        target=v6_workspace(dataset)
        if args.command=='v6-search':
            p=argparse.ArgumentParser()
            p.add_argument('--subsystem',required=True,choices=['power','thermal','attitude','data','communication','propulsion'])
            p.add_argument('--seed-start',type=int,default=92)
            p.add_argument('--seed-end',type=int,default=101)
            p.add_argument('--search-samples',type=int,default=2400)
            a=p.parse_args(args.args)
            if a.seed_end<a.seed_start:raise SystemExit('seed-end 必须 >= seed-start')
            call([sys.executable,target/'run_multiseed_experiment.py','--data',target/f'telemetry-msf-v6-{a.subsystem}.csv',
                  '--seed-start',a.seed_start,'--seed-end',a.seed_end,'--search-samples',a.search_samples,
                  '--out-dir',target/'experiments/v6_final'/f'{a.subsystem}_new_run'])
        elif args.command=='v6-refit':
            dest=ROOT/'results/_local/v6_pipeline/subsystem-health-model-v6-refit.json'
            call([sys.executable,target/'train_subsystem_health_model_v6.py','--root',target,'--out',dest])
            print('重训结果仅写入独立路径：',dest)
    if args.command in ('v8-health','v8-trajectory'):
        script='run_v8_health_confirmation.js' if args.command=='v8-health' else 'run_v8_trajectory_confirmation.js'
        verify_model_hash();call([NODE,ROOT/'scripts/satellite'/script])
    if args.command=='satellite':
        # 卫星应用统一验证入口：历史配对统计复算与当前真实重跑必须区分。
        action=args.args[0] if args.args else 'verify'
        extra=args.args[1:] if args.args else []
        if action in ('verify','paper-check'):
            verify_model_hash()
            call([sys.executable,ROOT/'scripts/satellite/paper_claims_check.py',*extra])
        elif action=='archive':
            call([sys.executable,ROOT/'scripts/satellite/paper_claims_check.py','--archive-only',*extra])
        elif action=='rerun-table4':
            if extra:
                call([sys.executable,'-m','msf_repro','satellite',*extra])
            else:
                call([sys.executable,'-m','msf_repro','satellite-all'])
        else:
            raise SystemExit('卫星用法：python run.py satellite verify [--with-v8] / archive / rerun-table4 [--subsystem power --start 192 --end 201]')
    if args.command in ('public','paper'):
        if not args.args:raise SystemExit('示例：python run.py paper status / paper check / paper demo --dataset hill_valley --seed 92')
        call([sys.executable,'-m','msf_repro',*args.args])

if __name__=='__main__':main()
