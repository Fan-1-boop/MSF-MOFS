# 在现有 MSF_MOFS_Reproducibility 项目根目录中执行。
# 建议把整个补充包放到项目的 analysis/pop30_g192/ 文件夹中。
$ErrorActionPreference = 'Stop'
$projectRoot = (Get-Location).Path
$experiment = Join-Path $projectRoot 'analysis\pop30_g192'
if (-not (Test-Path "$experiment\scripts\run_missing.py")) {
  throw '请先将实验补充包解压到项目的 analysis/pop30_g192 目录。'
}
python "$experiment\scripts\run_missing.py" --project-root "$projectRoot" --out "$experiment" --workers 1
if ($LASTEXITCODE -ne 0) {throw '存在运行失败的实验，检查 by_seed 中的 console.log'}
python "$experiment\scripts\verify_and_plot.py" --out "$experiment"
if ($LASTEXITCODE -ne 0) {throw '40组配对完整性检查未通过，拒绝生成正式图'}
Write-Host '全部实验完成；请查看 analysis/pop30_g192/analysis 和 figure。'
