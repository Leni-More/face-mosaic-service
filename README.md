# Face Mosaic Service — 视频人脸动态打码

基于 [yakhyo/uniface](https://github.com/yakhyo/uniface) 4.1.0 的上传式人脸打码服务。

每帧 RetinaFace 检测 + BYTETracker（Kalman 平滑、跨帧持久 ID）→ 遮挡框随人物动作动态跟随，
漏检若干帧时跟踪器仍输出预测框，马赛克不闪烁。

## 功能

- 浏览器上传视频（≤500MB），后台处理，实时进度
- **人脸替换（换脸）**：上传一张人脸照片，视频中所有人脸用 Landmark106
  关键点相似变换对齐 + 泊松融合替换为照片中的脸，随头部运动跟随；
  漏检帧按跟踪框位移外推。融合链包含三道处理：
  BiSeNet 解析生成皮肤掩膜（排除头发/背景，避免发际线色带）、
  变换矩阵 EMA 时序平滑（抑制帧间跳变）、
  融合前颜色迁移（对齐两人肤色，消除色差）。
  注意：表情来自照片本身，不跟随视频人物
  （无生成式模型）；需要表情级深度换脸请部署 facefusion / insightface inswapper
- 7 种遮挡方式：`pixelate` 马赛克 / `gaussian` 高斯模糊 / `emoji` 笑脸贴片 /
  `overlay` 自定义贴片图片 / `elliptical` 椭圆模糊 / `median` 中值模糊 /
  `blackout` 纯黑遮挡 / `none` 不遮挡（仅换脸）
- **选择性打码**：上传参考脸照片（ArcFace），`match` 只打码此人 /
  `exclude` 打码其他人豁免此人，阈值可调
- 强度可调（马赛克=块数，模糊=核大小，与人脸尺寸自适应缩放）
- 成品 H.264 MP4，自动回灌原视频音轨

## 运行

完整启动步骤（含本机沙箱环境的实测路径）见 [STARTUP.md](STARTUP.md)。

```bash
# 正常网络环境（推荐）
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python server.py
```

浏览器打开 `http://127.0.0.1:8000`。模型权重首次调用时自动下载（SHA-256 校验）。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/upload` | multipart：`video` 文件、`method`、`strength`，返回 `{job_id}` |
| GET | `/api/jobs/{id}` | `{status, processed, total, error}` |
| GET | `/api/download/{id}` | 成品 MP4 |

## 本目录特殊情况说明

`bootstrap.py` 是为本机沙箱环境准备的离线引导（pip 临时目录被拦截，
改为直接从 PyPI 解析依赖树、下载 wheel、解压进 site-packages）。普通环境直接用 pip。

## 注意

- uniface 为 MIT 协议；个别预训练权重有独立许可，商用前核对
  [Model licences](https://yakhyo.github.io/uniface/license-attribution/)。
