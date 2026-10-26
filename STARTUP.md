# 启动指南

本机实测环境：Windows + Python 3.14.10，项目位于 `E:\LLm\Project\face-mosaic-service`。

> **当前状态：环境已装好，可直接启动。** 不需要重建 venv，不需要重装依赖。

## 一、直接启动（当前机器）

```powershell
cd E:\LLm\Project\face-mosaic-service
.\.venv\Scripts\python.exe server.py
```

浏览器打开 <http://127.0.0.1:8000>，`Ctrl + C` 停止。

等价写法：

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

## 二、PowerShell 语法要点（踩坑记录）

PowerShell 不会在当前目录里搜索可执行文件，**必须加 `.\` 前缀**：

```powershell
.\.venv\Scripts\python.exe server.py     # 正确
.venv\Scripts\python server.py           # 报错：无法加载模块".venv"
```

在 cmd 里则相反，`.\.venv\Scripts\python.exe` 和 `.venv\Scripts\python.exe` 都能用。
文档里凡是以 `.\` 开头的命令都是 PowerShell 写法，cmd 下去掉 `.\` 即可。

另外本项目的 venv 是用 `--without-pip` 创建的（受沙箱限制），里面**没有 `pip.exe`**，
装包要用模块形式：

```powershell
.\.venv\Scripts\python.exe -m pip install <包名>
```

## 三、环境现状

- venv：`.venv`，45 个包，核心依赖已验证可导入
  （onnxruntime 1.31 / opencv 5.0 / uniface 4.1 / fastapi / imageio-ffmpeg）
- 模型权重：已缓存在 `.models\`，启动无需联网
  - `retinaface_mnet_v2.onnx`（人脸检测）
  - `2d_106.onnx`（106 点关键点）
  - `arcface_mnet.onnx`（人脸识别，选择性打码用）
  - `parsing_resnet18.onnx`（BiSeNet 皮肤掩膜，换脸融合用）
- ffmpeg：使用 imageio-ffmpeg 自带的二进制，无需系统安装

## 四、在全新机器上安装

```powershell
cd E:\LLm\Project\face-mosaic-service

# 建 venv。本机 ensurepip 会卡住或失败，务必加 --without-pip
python -m venv --without-pip .venv

# 用主 Python 的 pip 直接把依赖装进 venv（pip 26+ 的 --python 参数）
python -m pip --python .\.venv install -r requirements.txt
```

若第二条命令报 `PermissionError: ...\Temp\...\pip-unpack-...`
（pip 的临时解包目录被文件沙箱拦截），改用离线引导脚本：

```powershell
$env:TMP = "E:\LLm\Project\face-mosaic-service\.tmp"
$env:TEMP = $env:TMP

# 需用带 pip 的主 Python 运行，它会把包装进 .venv
python bootstrap.py
```

`bootstrap.py` 直接从 PyPI 解析依赖树、下载 wheel、解压进 site-packages，
已下载的 wheel 缓存在 `.wheels\`，重复运行只做补齐。

模型权重会在首次调用时自动下载；如果 GitHub / HuggingFace 都不通，
从镜像手动取文件放进 `.models\`：

```
https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/parsing_resnet18.onnx
```

缺失时程序自动回退（BiSeNet 缺失 → 凸包掩膜，效果略差）。

## 五、页面操作

1. 拖入视频（mp4 / mov / avi，≤500MB）
2. 三选一或组合：
   - 选遮挡方式（马赛克 / 高斯模糊 / 笑脸贴片 / 自定义贴片 / 椭圆模糊 / 中值模糊 / 纯黑 / 无遮挡）
   - 上传**换脸照片** → 视频中所有人脸替换为该照片中的脸
   - 上传**参考脸照片** → 选择性打码（只打此人 / 豁免此人）
3. 点「开始处理」，进度走完直接预览和下载

## 六、API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/upload` | multipart：`video`、`method`、`strength`，可选 `swap_face`、`ref_face`、`target_mode`、`ref_threshold`、`overlay`，返回 `{job_id, mode}` |
| GET | `/api/jobs/{id}` | `{status, processed, total, error}` |
| GET | `/api/download/{id}` | 成品 MP4 |

## 七、常见问题

- **端口被占用**：换端口启动 `--port 8001`，前端用相对路径，无需改动
- **检测不到人脸**：确认人脸清晰且不小于约 24px；极小脸场景可在 `server.py`
  调低 `BYTETracker` 的 `track_thresh` 或提高输入分辨率
- **`python -m venv .venv` 卡住不动或中断**：本机 `ensurepip` 有问题，先删掉残缺目录，
  再用 `python -m venv --without-pip .venv` 创建，然后按上面的装包步骤走
- **换脸有面具感**：这是关键点贴换方法的固有特性，表情取自照片、不跟随视频人物；
  需要表情级深度换脸请独立部署 facefusion / insightface inswapper
