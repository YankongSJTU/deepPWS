# DeepPWS Clinical Studio

**首选交付已更新为 2026.10.26 DeepPWS 单机离线版。** 整体复制 `../faceBA_Offline_Windows/` 后，双击其中的 `Start_DeepPWS.bat` 即可使用；Python、依赖和全部模型随包提供，不连接服务器。新增颜色、侧别、神经分区及粗区域候选，以及重庆颜色与浙江神经分区适配选项；候选需核对确认，主疗效模型保留完整 3D/UV Ridge。操作说明见 [单机离线版说明](OFFLINE_README.md)，发行包为 `dist/DeepPWS_Offline_Windows_20261026_workbench1.zip`。下面保留早期 Web/GPUT6 版的部署说明供参考。

最新版增加年龄与病历 III/IV 标签估计：手填优先，留空使用带来源标记的照片估计，结果页可核对修正；性别、并发症和拍照间隔只手填。

一个可独立打包的中文影像评估工作台。支持临床资料、1–5 组治疗前后照片、完整冻结模型推理、病例归档、影像对比和报告导出。

## 现在使用

推荐使用已经配置好的 GPUT6：双击 **`start_gpu6_windows.bat`**，或运行：

```powershell
python connect_gpu6.py
```

浏览器打开 **http://127.0.0.1:8865**。连接脚本通过 Windows OpenSSH 将本机端口安全转发到 GPUT6 的 loopback 服务；需要当前用户具有 `kongyan@202.120.45.56:22236` 的 SSH 访问权限。包中没有 SSH 密钥或密码。

服务端目录：`/export/home/kongyan/project/faceBA/faceba_studio_20261023`。服务仅监听 `127.0.0.1:8765`，没有公开发布到互联网，也没有修改原有训练代码和实验输出。

若 GPUT6 服务重启后尚未运行：

```bash
cd /export/home/kongyan/project/faceBA/faceba_studio_20261023
/usr/bin/python3 start_gpu6_service.py
```

## 使用流程

1. 填写研究病例编号，补充年龄、性别、部位、治疗次数等资料。无需真实姓名。
2. 添加治疗前与治疗后照片。每阶段最多 5 张、每张不超过 20 MB；分析时两个阶段数量须相同，按相同视角顺序配对。
3. 点击“一键分析”。后台依次执行影像编码、面部配准、Lab 色彩变化、LAM 三维/UV 特征提取及固定 Ridge 模型推理。进度来自实际处理阶段。
4. 查看颜色、面积及综合评分，拖动滑块比较原始照片；导出 JSON 报告，或打开排版报告并通过浏览器“打印 / 另存为 PDF”。

可以只填写临床信息并先保存病例，随后从“分析记录”继续。更改参与模型的临床字段后，原评分失效并要求重新分析；补充性别、治疗次数、备注等记录字段可保留评分。更换照片时会建立新评估，保留旧记录。

## 临床字段边界

| 模型实际使用 | 仅保存到病例和报告 |
|---|---|
| 年龄、Fitzpatrick 肤型、病灶部位、三叉神经分区、颜色分型 | 性别、治疗次数、治疗方式、并发症、并发症说明、前后日期、临床备注 |

分类选项读取自模型保存的训练类别。年龄缺失时使用保存的训练中位数；未知分类按原训练编码规则编码并在结果中提示。新增加的记录字段不会被任意塞进原模型。评分范围 0–10，综合评分为颜色与面积的等权平均。固定分组由 3、6 两个阈值派生，不是独立的临床诊断。

## 内置模型与文件

- `models/fusion_source.npz`：校正队列源模型的拟合系数、截距和冻结预处理参数。推理时不反序列化 pickle。
- `models/model.json`：临床类别、参数指纹、模型版本、输入维度和字段用途。
- `runtime/`（完整包）：CLIP、FaRL、LAM、DINOv2 权重、面部配准资源及模型源码；不含患者照片、评分表、训练集特征缓存或 774 个患者折模型。
- `vendor/`：原特征提取函数的副本，仅适配运行路径，不修改原始实验代码。
- `backend/`：FastAPI 服务、病例与任务管理、评分和报告。
- `web/`：无外部 CDN、无前端编译依赖的界面。
- `data/`：运行后生成的 SQLite 病例库及每个病例的照片、特征和日志；**不进入发行包**。

模型源：`results/loocv_runs/00_baseline_corrected_813/shanghai_source.pkl`。上海 813 条记录、774 个患者组；源 LOOCV overall Spearman 为 0.757265。选择基线的校正队列参数，不使用旧 829 条队列模型。训练标识符和模型的训练键未导出到服务参数包。

## 在自己的 GPU 环境运行

建议 Linux + NVIDIA CUDA。完整提取依赖 PyTorch、匹配其版本的 PyTorch3D、CLIP/FaRL/LAM 资源以及 3DDFA；普通 CPU 环境仍可保存病例，但不会产生伪造评分。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-inference.txt
# 另行安装与本机 PyTorch/CUDA 匹配的 PyTorch3D。
python run.py --port 8765
```

完整包的权重默认从 `runtime/` 加载；可用 `FACEBA_RUNTIME_ROOT` 指向已有兼容资源目录。`runtime/torch_cache` 内含 LAM 使用的 DINOv2 checkpoint，避免推理时联网下载。Windows 下如已配置完整兼容 GPU 环境，可双击 `start_windows.bat`；一般优先使用 GPUT6 入口。

已验证的 GPUT6 复用原有 Python 3.10 / PyTorch 2.5.1 CUDA 12.1 / PyTorch3D 运行环境；Web 依赖安装在该应用自己的 `runtime/webdeps/`，不修改原有全局依赖。`requirements-inference.txt` 的 PyTorch 2.6 / torchvision 0.21 是新环境的起点，PyTorch3D 必须与所选 PyTorch/CUDA 匹配。

配置变量：

| 变量 | 用途 |
|---|---|
| `FACEBA_RUNTIME_ROOT` | 大模型与配准资源目录；默认 `runtime/` |
| `FACEBA_DATA_ROOT` | 病例数据目录；默认 `data/` |
| `FACEBA_API_TOKEN` | 共享网络部署的访问密钥，由管理员另行设置 |
| `FACEBA_WORKER_LAUNCHER` | 特殊 Python/GPU 环境的子进程启动脚本；GPUT6 已配置 |
| `CUDA_VISIBLE_DEVICES` | 选择 GPU；GPUT6 启动脚本默认物理 GPU 2 |

共享服务：设置 `FACEBA_API_TOKEN` 后才允许 `python run.py --host 0.0.0.0`。界面会要求输入密钥。公网部署还应由实际部署环境提供 HTTPS；本次交付的默认访问走 SSH 隧道。

## 验证与打包

```powershell
python -m pytest tests/test_app.py -q
node --check web/app.js
python build_release.py --full
```

`dist/faceBA_Studio_Connector_20261023.zip` 是轻量应用/连接包，含源模型参数，使用已配置 GPUT6 进行完整照片分析。

`dist/faceBA_Studio_Full_20261023.zip` 另含约 4.55 GB 的离线特征编码器和资源，可在兼容 GPU/Python 环境中独立运行。Python、CUDA 驱动及编译扩展不作为一个跨平台可执行环境打入 ZIP。

浏览器检查覆盖桌面、390px 手机布局、临床表单、保存草稿、记录页和删除操作。模型参数导出已与原始保存模型的推理路径比较，含缺失年龄、未知类别及非有限特征填补；匹配误差容限 1e-10。真实照片端到端验证摘要见 `gpu6_validation.json`（如验证完成后提供）。测试照片始终留在 GPUT6，不打入包内。

第三方模型和源码保留各自的许可约束，原有许可文件随相关源码保留。
