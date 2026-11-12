# SignalScope / Topic 2

《信号与系统》Topic 2 实验工程包含两个任务：交互式多域信号分析器，以及声学 2-FSK 复杂噪声通信实验。两个任务共用 `shared/` 中的信号处理模块，报告与演示文稿集中在 `documents/`。

[GitHub 仓库](https://github.com/PlyusWang/signals-and-systems-topic2) · [Task 1 Windows 便携版](https://github.com/PlyusWang/signals-and-systems-topic2/releases/tag/task1-v1.0.0)

## 共享与复用

完整复用包解压后，双击根目录的 `Start-SignalScope.cmd` 即可打开 SignalScope 和一份含噪样例，无需安装 Python。也可以直接运行 `task1\release\portable\SignalScope\SignalScope.exe`。保留 EXE 同目录的 `_internal\` 和 `samples\`，它们分别提供运行依赖和样例音频。

修改源码时，按下文建立 `.venv` 并安装 `requirements.txt`；原来的虚拟环境无需复制。报告源文件、文献和图片在 `documents\report\`，科研图在 `task2\figures\`，PPTX 及其生成程序在 `documents\presentation\`。TeX 会按相对路径引用科研图和 `shared\` 中的代码，因此应保留整个项目目录关系。编译报告需要 XeLaTeX、Biber、minted 及源文件指定的字体（包括 SimSun）；PPTX 可直接用 PowerPoint 编辑。

从完整工程重新打包：

```powershell
py -3.12 documents\scripts\build_reuse_bundle.py
```

生成 `release\SignalScope-Topic2-reusable-win-x64.zip` 及同名 `.sha256` 校验文件。包中保留源码、测试、完整样本与实验记录、报告/PPT 及完整的 Windows x64 便携程序目录；不包含 `.git`、本地虚拟环境、会话记录、编译缓存或旧审查 ZIP。包内的 `MANIFEST.sha256` 列出各文件的校验值。重新打包前，便携程序须已按下文的构建命令生成。

## 目录

```text
topic2/
├─ .codex/                    项目状态记录
├─ .venv*/                    本地 Python 环境
├─ .gitignore
├─ README.md
├─ requirements.txt           两个任务的运行与测试依赖
├─ conftest.py                共用测试配置
├─ shared/                    FFT、STFT、CWT、相关、小波、音频 I/O 及共用测试
├─ task1/
│  ├─ analyzer/               SignalScope 桌面程序
│  ├─ scripts/                启动、GUI 检查、CWT 基准和便携版构建
│  ├─ tests/
│  ├─ samples/               可直接打开的代表性 WAV
│  ├─ screenshots/           正式文档引用的 GUI 图片
│  ├─ SignalScope.spec
│  ├─ requirements-pack.txt
│  └─ release/                Windows 便携目录和 ZIP
├─ task2/
│  ├─ communication_experiment/  2-FSK、噪声模型与实验驱动
│  ├─ scripts/
│  ├─ tests/
│  ├─ audio/                  原始、加噪与处理 WAV
│  ├─ experiments/            CSV、JSON、重复接收数组及验证记录
│  └─ figures/                科研图表 PNG / PDF
└─ documents/
   ├─ report/                 XeLaTeX、文献、截图和报告 PDF
   ├─ presentation/           可编辑 PPTX、PDF 和生成程序
   ├─ scripts/                文档图片及可选审查包构建
   └─ REVIEW_README.md
```

## 运行

在项目根目录使用 64 位 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m task1.analyzer.app task1\samples\clean_plus_N1_N2.wav
.\.venv\Scripts\python.exe -m task2.communication_experiment.runner
```

也可以运行 `task1\scripts\run_analyzer.ps1` 和 `task2\scripts\run_experiment.ps1`，它们均使用根目录的 `.venv`。通信实验命令会重新生成实验结果，查看现有成品时无需执行。

SignalScope 的全局概览控制局部浏览窗口，局部左键拖动选择分析片段，Shift + 拖动临时平移，Esc 清除独立选区。FFT 支持动态 Hz/kHz 刻度和手动缩放保持；STFT 显示实际窗长、FFT 和重叠长度；STFT/CWT 使用相对峰值幅度（dB）。播放游标同时显示在局部波形和全局概览中。

“选择对比 WAV”可独立指定另一份音频，支持叠加或共用时间轴的上下分轨显示、不同采样率重采样和手动时间偏移。FFT 对比使用相同时间区间、窗函数及 dBFS 参考。播放和 WAV 导出使用主信号。小窗口提供折叠概览和滚动区域，保留主要绘图区高度。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest task1\tests task2\tests shared\tests -q
```

自动化测试共 82 项。已有实验记录包含 680 条单帧评价和 78 条重复接收记录。长音频 CWT、GUI 及便携版验证记录保存在 `task2\experiments\performance\`。

## Windows 便携版

```powershell
py -3.12 -m venv .venv-pack
.\.venv-pack\Scripts\python.exe -m pip install -r task1\requirements-pack.txt
.\task1\scripts\build_portable.ps1
```

成品位于 `task1\release\SignalScope-portable-win-x64.zip`，目录版位于 `task1\release\portable\SignalScope\SignalScope.exe`。解压后可直接运行，`samples\` 附有代表性 WAV，目标电脑无需 Python。

便携 ZIP 通过 GitHub Release 分发；源码仓库保留 `task1\samples\`、完整的 `task2\audio\` 和构建脚本。本地虚拟环境、编译缓存和发布二进制不提交到 Git。

## 报告与演示文稿

- 报告：`documents\report\topic2_report.pdf`，同目录保留 TeX、Biber 文献及截图。
- 演示文稿：`documents\presentation\topic2_presentation.pptx`，含 14 页、11 个 Office 原生数学公式；同目录保留生成程序。

报告在 `documents\report\` 内执行 XeLaTeX、Biber、XeLaTeX 两遍：

```powershell
Push-Location documents\report
xelatex -shell-escape -interaction=nonstopmode -halt-on-error topic2_report.tex
biber topic2_report
xelatex -shell-escape -interaction=nonstopmode -halt-on-error topic2_report.tex
xelatex -shell-escape -interaction=nonstopmode -halt-on-error topic2_report.tex
Pop-Location
```

在根目录执行 `.\.venv\Scripts\python.exe documents\presentation\build_presentation.py` 可重建 PPTX；生成程序保留已有原生公式。`--refresh-analysis` 只更新相关科研图与 SNR 公式，`--refresh-gui` 只更新 GUI 截图。需要单独审查包时可运行 `documents\scripts\build_review_bundle.ps1`；生成的 ZIP 不提交到 Git。
