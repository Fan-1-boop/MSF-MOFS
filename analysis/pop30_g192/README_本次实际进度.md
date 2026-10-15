# MSF-MOFS Pop=30 / G=192 中途存档（已验证）

2026-10-14 本次真实补跑完成 Hill-Valley seed 95、96；已完成 10/40 组、30/120 次搜索。

新完成两组均有 3 种算法各 193 个代数状态（0–192），每组 579 条逐代记录，数据格式和文件哈希已核验。余下 30 组未完成。

**需要注意**：会话计算环境可能被重置，已启动的后台进程不等于永久托管计算任务。

## 在自己的 Windows 电脑可靠地继续

1. 确保已解压 `MSF_MOFS_Reproducibility_Paper_Clean_CN_v1_2_2.zip` 并安装依赖。
2. 解压本包到主项目的 `analysis/pop30_g192/`（覆盖同名历史检查点，保留自己已生成的结果）。
3. 在项目根目录运行：

```powershell
python analysis/pop30_g192/scripts/run_missing.py --project-root . --out analysis/pop30_g192 --workers 1
```

程序自动跳过已经完成的种子。计算将持续至任务完成或电脑进程中断。完整正式图只在 40/40 组验证通过后制作。
