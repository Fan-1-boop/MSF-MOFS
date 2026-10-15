#!/usr/bin/env python3
"""对冻结交付包做独立内容哈希核验（不属于统计实验）。"""
from __future__ import annotations
from pathlib import Path
import csv,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'provenance/FILE_MANIFEST.csv'

def main():
    if not MANIFEST.exists(): raise SystemExit('缺少文件清单 provenance/FILE_MANIFEST.csv')
    failed=[];checked=0
    for row in csv.DictReader(MANIFEST.open(encoding='utf-8',newline='')):
        p=ROOT/row['path']
        if not p.is_file():failed.append((row['path'],'文件缺失'));continue
        size=p.stat().st_size
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if size!=int(row['bytes']) or sha!=row['sha256']:failed.append((row['path'],'字节大小或哈希不符'))
        checked+=1
    print(f'交付文件完整性检查：已校验 {checked} 个文件，错误 {len(failed)} 个')
    for a,b in failed[:30]:print('  ERROR:',a,b)
    if failed:sys.exit(1)
if __name__=='__main__':main()
