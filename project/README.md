# XPBD Cloth Tearing Demo

Phase 1 is a CPU-first Python/Taichi research demo. The solver uses XPBD
compliance (`alpha = compliance / dt²`) for stretch, bend, and pin
constraints. Mesh fracture is handled between substeps by vertex splitting;
the solver itself has no fracture or coupling dependency.

## Environment and run

From the workspace root, install conda and create the environment. The
environment file is the single dependency manifest; its `pip` subsection is
used only for packages whose maintained Windows builds are distributed from
PyPI:

```powershell
conda env create -f environment.yml
conda activate XPBD_tear
```

If the shell is still using the base interpreter, use the explicit form for
every project command:

```powershell
conda run -n XPBD_tear python -m project.main --diagnostics --frames 80
```

NumPy 1.26.4 and matplotlib 3.8.4 are pinned in conda because this is the
verified Windows/Python 3.10 headless-diagnostics pair. Taichi and pyrender are
installed through the environment file's pip subsection because their current
Windows packages are better resolved by pip. `requirements.txt` remains as a
standalone pip update file for an already-created environment.

The verified demo defaults are stretch compliance `1e-6`, bend compliance
`2e-5`, damping `0.96`, eight substeps, and critical strain `0.15`. This gives
visible stretching and controlled fracture feedback; the extreme fracture
regression still uses its own low threshold/high pull-speed settings.

Run the interactive Taichi demo with:

```powershell
conda run -n XPBD_tear python -m project.main
```

If a Taichi compute backend is unavailable, the entry point falls back to CPU.
The previous Taichi GGUI window is still available, but this host's Vulkan
driver cannot initialize it; use the headless visualization commands below.

Controls are available in the right-side GUI: stretch/bend compliance,
critical strain, substeps, tearing toggle, reset, pause, and single-step.
The default scenario pins the top edge and slowly pulls the bottom edge.

## Headless diagnostics and scene export

Generate matplotlib plots and a JSON topology check without opening any window:

```powershell
conda run -n XPBD_tear python -m project.main --diagnostics --frames 80 --resolution 8 `
  --critical-strain 0.05 --pull-speed 1.0 --output-dir artifacts
```

This writes `strain_over_time.png`, `fracture_events.png`,
`topology_check.png`, and `topology_check.json`. The diagnostics include
maximum/mean and per-edge strain history, active constraint count, cumulative
fracture count, and triangle connected-component count.

Add `--render-scene` to export the final mesh:

```powershell
conda run -n XPBD_tear python -m project.main --diagnostics --render-scene --frames 80 --resolution 8 `
  --critical-strain 0.05 --pull-speed 1.0 --output-dir artifacts
```

`viz/scene_render.py` writes a deterministic matplotlib 3D PNG
(`artifacts/cloth_final.png`) with broken edges highlighted in red. On a host
with a known-good offscreen OpenGL context, set
`$env:XPBD_USE_PYRENDER=1` to opt into the pyrender backend; the default avoids
Windows driver initialization hangs in headless runs.

Create a short animated GIF demo:

```powershell
conda run -n XPBD_tear python -m project.main --gif `
  --gif-frames 60 --gif-fps 12 --resolution 8 `
  --gif-output artifacts/xpbd_tearing_demo.gif
```

## Validation

Run `conda run -n XPBD_tear python -m project.tests.test_simulation` from the workspace root for a
headless check covering finite dynamics, repeated tearing, vertex duplication,
disconnected triangle components, valid regenerated constraint references, and
the swappable always-false failure model. The graphical window requires a
working Vulkan/GUI driver on the host; headless plots and scene export do not.

## 中文说明

### 环境管理

本项目固定使用 conda 环境 `XPBD_tear`，不要使用 conda 主环境或工作区中的临时解释器运行项目。首次安装：

```powershell
conda env create -f environment.yml
conda activate XPBD_tear
```

如果当前 PowerShell 没有正确激活环境，使用显式命令运行：

```powershell
conda run -n XPBD_tear python -m project.tests.test_simulation
```

`environment.yml` 是唯一的环境清单：NumPy 和 Matplotlib 由 conda 管理，Taichi、pyrender 及其运行依赖通过该清单中的 pip 子段安装。Python 3.10、NumPy 1.26.4 和 Matplotlib 3.8.4 已针对 Windows 无窗口诊断流程固定验证。

### 运行与验收

交互演示：

```powershell
conda run -n XPBD_tear python -m project.main
```

无窗口诊断并生成图片：

```powershell
conda run -n XPBD_tear python -m project.main --diagnostics --render-scene `
  --frames 80 --resolution 8 --output-dir artifacts
```

诊断会生成应变曲线、断裂事件曲线、拓扑连通性图、拓扑 JSON 和最终布料 PNG。当前主机默认使用稳定的 Matplotlib 3D fallback；只有确认 OpenGL/EGL 离屏上下文可用时，才设置 `$env:XPBD_USE_PYRENDER=1` 使用 pyrender。

### GIF Demo

GIF 导出只调用已有的仿真步进和网格数据，不修改 XPBD 求解器或断裂拓扑逻辑：

```powershell
conda run -n XPBD_tear python -m project.main --gif `
  --gif-frames 60 --gif-fps 12 --resolution 8 `
  --gif-output artifacts/xpbd_tearing_demo.gif
```

默认输出约 5 秒，红色线段表示断裂边。`--gif-frames` 控制帧数，`--gif-fps` 控制播放速度，`--resolution` 控制布料网格密度。

### 当前已验证参数

默认物理参数为：拉伸 compliance `1e-6`、弯曲 compliance `2e-5`、阻尼 `0.96`、每帧 8 个子步、10 次迭代、临界应变 `0.15`。这组参数可以产生可见但受控的形变和断裂反馈。
