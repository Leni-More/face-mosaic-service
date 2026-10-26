"""人脸视频处理服务：上传视频 -> 检测+跟踪 -> 动态打码 / 人脸替换 -> 回传成品。

两条独立能力：
1. 打码（method）：每帧 RetinaFace 检测 + BYTETracker（Kalman 平滑，漏检帧持续
   输出预测框）-> 按 track 框打码，7 种方式，可上传自定义贴片图片。
2. 换脸（swap_face）：上传一张人脸照片，视频里所有人脸用 Landmark106 关键点
   相似变换对齐 + 泊松融合替换成照片中的脸，随头部运动跟随；漏检帧按跟踪框
   位移外推。表情不会跟随视频人物（无生成式模型），如需深度换脸用 facefusion。
3. 选择性打码（ref_face + target_mode）：按 ArcFace 相似度只打码参考脸本人或
   豁免参考脸本人。

输出统一转 H.264 MP4 并回灌原音轨。
"""
import math
import os
import subprocess
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 模型权重缓存固定到项目目录，避免依赖用户主目录写权限
os.environ.setdefault("UNIFACE_CACHE_DIR", str(Path(__file__).parent / ".models"))

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
WORK = ROOT / ".work"
for d in (UPLOADS, OUTPUTS, WORK):
    d.mkdir(exist_ok=True)

MAX_UPLOAD = 500 * 1024 * 1024
METHODS = {"pixelate", "gaussian", "blackout", "elliptical", "median",
           "emoji", "overlay", "none"}
TARGET_MODES = {"all", "match", "exclude"}

app = FastAPI(title="Face Mosaic Service")
JOBS: dict[str, dict] = {}
_model_lock = threading.Lock()
_models: dict = {}


def get_models():
    """懒加载检测 / 识别 / 关键点 / 解析模型，权重首次调用时自动下载。"""
    with _model_lock:
        if not _models:
            from uniface.detection import RetinaFace
            from uniface.landmark import Landmark106
            from uniface.parsing import BiSeNet
            from uniface.recognition import ArcFace

            _models.update(detector=RetinaFace(), landmarker=Landmark106(),
                           recognizer=ArcFace(), parser=BiSeNet())
    return _models


def clamp_box(box, w, h):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
    return x1, y1, x2, y2


def odd(n):
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def eye_angle(landmarks):
    """按 5 点 landmarks 的双眼连线估计人脸旋转角（弧度）。"""
    try:
        (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
        return math.atan2(ly - ry, lx - rx)
    except Exception:
        return 0.0


def draw_emoji(frame, box):
    """卡通笑脸贴片：黄色圆脸 + 眼睛 + 微笑弧线，随人脸框缩放。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = max(8, min(x2 - x1, y2 - y1) // 2)
    ink = (45, 45, 45)
    cv2.circle(frame, (cx, cy), r, (0, 200, 255), -1)
    cv2.circle(frame, (cx, cy), r, ink, max(2, r // 16))
    er = max(2, r // 6)
    cv2.circle(frame, (cx - r // 3, cy - r // 5), er, ink, -1)
    cv2.circle(frame, (cx + r // 3, cy - r // 5), er, ink, -1)
    cv2.ellipse(frame, (cx, cy + r // 6), (r // 2, max(2, r // 3)), 0, 25, 155,
                ink, max(2, r // 7))


def draw_overlay(frame, box, patch_bgra, angle):
    """把 RGBA 贴片缩放旋转后贴到人脸框上，alpha 混合。"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 6:
        return
    ph, pw = patch_bgra.shape[:2]
    scale = max(bw / pw, bh / ph) * 1.1
    nw, nh = max(2, int(pw * scale)), max(2, int(ph * scale))
    patch = cv2.resize(patch_bgra, (nw, nh), interpolation=cv2.INTER_AREA)
    if abs(angle) > 0.02:
        diag = int(math.hypot(nw, nh)) + 2
        canvas = np.zeros((diag, diag, 4), np.uint8)
        oy, ox = (diag - nh) // 2, (diag - nw) // 2
        canvas[oy:oy + nh, ox:ox + nw] = patch
        m = cv2.getRotationMatrix2D((diag / 2, diag / 2),
                                    math.degrees(angle), 1.0)
        patch = cv2.warpAffine(canvas, m, (diag, diag))
        nh, nw = diag, diag
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    px1, py1 = cx - nw // 2, cy - nh // 2
    h, w = frame.shape[:2]
    ix1, iy1 = max(0, px1), max(0, py1)
    ix2, iy2 = min(w, px1 + nw), min(h, py1 + nh)
    if ix2 <= ix1 or iy2 <= iy1:
        return
    sub = patch[iy1 - py1:iy2 - py1, ix1 - px1:ix2 - px1]
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    region = frame[iy1:iy2, ix1:ix2].astype(np.float32)
    frame[iy1:iy2, ix1:ix2] = (sub[:, :, :3].astype(np.float32) * alpha
                               + region * (1 - alpha)).astype(np.uint8)


def apply_method(frame, box, method, strength, patch=None, angle=0.0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return
    if method == "emoji":
        draw_emoji(frame, (x1, y1, x2, y2))
        return
    if method == "overlay" and patch is not None:
        draw_overlay(frame, (x1, y1, x2, y2), patch, angle)
        return
    if method == "none":
        return
    roi = frame[y1:y2, x1:x2]
    if method == "pixelate":
        blocks = max(3, int(strength))
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi, (max(1, rw // blocks), max(1, rh // blocks)),
                           interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh),
                                         interpolation=cv2.INTER_NEAREST)
        return
    if method == "blackout":
        frame[y1:y2, x1:x2] = 0
        return
    k = odd(max(3, strength * min(x2 - x1, y2 - y1) // 20))
    if method == "gaussian":
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return
    if method == "median":
        frame[y1:y2, x1:x2] = cv2.medianBlur(roi, min(k, 51))
        return
    if method == "elliptical":
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.ellipse(mask, (roi.shape[1] // 2, roi.shape[0] // 2),
                    (max(1, roi.shape[1] // 2 - 2), max(1, roi.shape[0] // 2 - 2)),
                    0, 0, 360, 255, -1)
        frame[y1:y2, x1:x2] = np.where(mask[..., None] == 255, blurred, roi)


# ---------------------------------------------------------------- 换脸

# 换脸 mask 保留的 BiSeNet 类别：皮肤/眉/眼/鼻/嘴/唇（排除头发、服装、背景）
FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)
# 变换矩阵时序平滑系数，越大越跟手、越小越平滑
SMOOTH_ALPHA = 0.65


def build_face_mask(lm, shape):
    """凸包掩膜（BiSeNet 不可用时的回退方案）。"""
    pts = lm.astype(np.float32)
    c = pts.mean(axis=0)
    hull = cv2.convexHull(pts.astype(np.int32))[:, 0, :].astype(np.float32)
    grown = (c + (hull - c) * 1.08).astype(np.int32)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, grown, 255)
    return cv2.GaussianBlur(mask, (41, 41), 0)


def build_skin_mask(img, bbox, lm):
    """用 BiSeNet 解析出皮肤+五官区域，精确排除头发与背景。

    解析在脸部 crop 上做（多留 15% 边距以覆盖发际线），结果贴回原图坐标。
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_box(bbox, w, h)
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    cls = get_models()["parser"].parse(crop)
    keep = np.isin(cls, FACE_CLASSES).astype(np.uint8) * 255
    # 闭运算补五官之间的缝隙，再从边界 floodFill 填内部小洞
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    ff = keep.copy()
    canvas = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, canvas, (0, 0), 255)
    keep = cv2.bitwise_or(keep, cv2.bitwise_not(ff))
    # 去掉零散小区域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = np.where(labels == big, 255, 0).astype(np.uint8)
    full = np.zeros((h, w), np.uint8)
    full[cy1:cy2, cx1:cx2] = keep
    return cv2.GaussianBlur(full, (41, 41), 0)


def prepare_swap_reference(path):
    """从换脸照片提取 106 点与皮肤融合 mask（照片坐标系）。"""
    m = get_models()
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("换脸照片读取失败")
    faces = m["detector"].detect(img)
    if not faces:
        raise RuntimeError("换脸照片中未检测到人脸")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    lm = np.asarray(m["landmarker"].get_landmarks(img, face.bbox), np.float32)
    mask = None
    try:
        mask = build_skin_mask(img, face.bbox, lm)
    except Exception:  # noqa: BLE001
        mask = None
    if mask is None or mask.max() == 0:
        mask = build_face_mask(lm, img.shape)
    return {"img": img, "lm": lm, "mask": mask}


def color_transfer(warped, frame, wmask, strength=0.75):
    """把 warp 后的参考脸颜色统计对齐到目标区域，消除两人的肤色色差。

    按通道做均值/方差对齐（Reinhard 方法），再按 strength 与原图混合，
    保留部分参考肤色的辨识度。
    """
    sel = wmask > 128
    if sel.sum() < 50:
        return warped
    src_px = warped[sel].reshape(-1, 3).astype(np.float32)
    dst_px = frame[sel].reshape(-1, 3).astype(np.float32)
    out = warped.astype(np.float32)
    for c in range(3):
        sm, ss = src_px[:, c].mean(), src_px[:, c].std() + 1e-6
        dm, ds = dst_px[:, c].mean(), dst_px[:, c].std() + 1e-6
        moved = (out[:, :, c] - sm) / ss * ds + dm
        out[:, :, c] = strength * moved + (1 - strength) * out[:, :, c]
    np.clip(out, 0, 255, out)
    return out.astype(np.uint8)


def swap_face_on_frame(frame, target_lm, ref, prev_m=None):
    """把参考脸按关键点相似变换 warp 到目标脸位置并泊松融合。

    prev_m 为上一帧变换矩阵，传入后对当前估计做 EMA 平滑以抑制帧间跳变。
    """
    h, w = frame.shape[:2]
    mres = cv2.estimateAffinePartial2D(ref["lm"], target_lm,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=5.0)
    m = mres[0] if mres else None
    if m is None:
        return frame, None
    if prev_m is not None:
        m = SMOOTH_ALPHA * m + (1 - SMOOTH_ALPHA) * prev_m
    center = tuple(int(v) for v in target_lm.mean(axis=0))
    warped = cv2.warpAffine(ref["img"], m, (w, h))
    wmask = cv2.warpAffine(ref["mask"], m, (w, h))
    # seamlessClone 会替换 mask 外接矩形内的全部像素：把矩形内、mask 外的
    # 参考图区域清零，避免参考照片的头发/背景梯度经泊松扩散到脸上
    warped = (warped.astype(np.float32) * (wmask[..., None] / 255.0)).astype(np.uint8)
    warped = color_transfer(warped, frame, wmask)
    try:
        out = cv2.seamlessClone(warped, frame, wmask, center, cv2.NORMAL_CLONE)
        return out, m
    except cv2.error:
        alpha = (wmask.astype(np.float32) / 255.0)[..., None]
        out = (warped.astype(np.float32) * alpha
               + frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return out, m


# ---------------------------------------------------------------- 工具

def load_patch(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def build_ref_embedding(path):
    m = get_models()
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = m["detector"].detect(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = m["recognizer"].get_normalized_embedding(img, face.landmarks)
    return np.asarray(emb, dtype=np.float32).ravel()


def face_center(face):
    x1, y1, x2, y2 = face.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


# ---------------------------------------------------------------- 主流程

def process_job(job_id):
    job = JOBS[job_id]
    src = UPLOADS / f"{job_id}{job['suffix']}"
    try:
        m = get_models()
        from uniface.tracking import BYTETracker

        tracker = BYTETracker(track_thresh=0.4, track_buffer=60, match_thresh=0.6)

        ref_emb = job.get("ref_emb")
        patch = load_patch(job["overlay_path"]) if job.get("overlay_path") else None
        if job["method"] == "overlay" and patch is None:
            raise RuntimeError("贴片图片读取失败，请改用 PNG/JPG")
        swap_ref = prepare_swap_reference(job["swap_path"]) \
            if job.get("swap_path") else None

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        job.update(total=total, processed=0)

        tmp_out = WORK / f"{job_id}.mp4"
        writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        n = 0
        track_wanted: dict[int, bool] = {}
        track_swap: dict[int, tuple] = {}  # tid -> (M, center)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = m["detector"].detect(frame)
            if faces:
                dets = np.array([[*f.bbox, f.confidence] for f in faces],
                                dtype=np.float32)
            else:
                dets = np.empty((0, 5), np.float32)
            tracks = tracker.update(dets)

            # 检测脸 -> track 关联（中心最近邻）
            pairs = []
            for t in tracks:
                box = t[:4]
                tcx, tcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                best, bd = None, 1e18
                for i, f in enumerate(faces):
                    fcx, fcy = face_center(f)
                    d = (tcx - fcx) ** 2 + (tcy - fcy) ** 2
                    if d < bd:
                        best, bd = i, d
                limit = (box[2] - box[0] + box[3] - box[1]) ** 2
                pairs.append((t, best if (best is not None and bd <= limit) else None))

            # 1) 换脸（优先于打码；换脸时 method 通常为 none）
            if swap_ref is not None:
                live_tids = set()
                for t, fi in pairs:
                    tid = int(t[4])
                    box = t[:4]
                    tcx, tcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    live_tids.add(tid)
                    if fi is not None:
                        lm = np.asarray(
                            m["landmarker"].get_landmarks(frame, faces[fi].bbox),
                            np.float32)
                        prev = track_swap.get(tid)
                        frame, mm = swap_face_on_frame(
                            frame, lm, swap_ref,
                            prev_m=prev[0] if prev else None)
                        if mm is not None:
                            track_swap[tid] = (mm, (tcx, tcy))
                    else:
                        # 漏检帧：用 Kalman 框中心位移外推上一帧变换
                        prev = track_swap.get(tid)
                        if prev:
                            pm, pc = prev
                            m2 = pm.copy()
                            m2[0, 2] += tcx - pc[0]
                            m2[1, 2] += tcy - pc[1]
                            warped = cv2.warpAffine(swap_ref["img"], m2, (w, h))
                            wmask = cv2.warpAffine(swap_ref["mask"], m2, (w, h))
                            warped = (warped.astype(np.float32)
                                      * (wmask[..., None] / 255.0)).astype(np.uint8)
                            warped = color_transfer(warped, frame, wmask)
                            try:
                                frame = cv2.seamlessClone(
                                    warped, frame, wmask,
                                    (int(tcx), int(tcy)), cv2.NORMAL_CLONE)
                                track_swap[tid] = (m2, (tcx, tcy))
                            except cv2.error:
                                pass
                for tid in list(track_swap):
                    if tid not in live_tids:
                        del track_swap[tid]

            # 2) 打码
            wanted_flags = []
            for f in faces:
                if ref_emb is None or swap_ref is not None:
                    wanted_flags.append(swap_ref is None)
                    continue
                emb = np.asarray(
                    m["recognizer"].get_normalized_embedding(frame, f.landmarks),
                    dtype=np.float32).ravel()
                sim = float(np.dot(emb, ref_emb))
                wanted_flags.append(sim >= job["ref_threshold"]
                                    if job["target_mode"] == "match"
                                    else sim < job["ref_threshold"])
            for t, fi in pairs:
                tid = int(t[4])
                if fi is not None:
                    track_wanted[tid] = wanted_flags[fi]
                    angle = eye_angle(faces[fi].landmarks)
                else:
                    angle = 0.0
                if track_wanted.get(tid, False):
                    apply_method(frame, t[:4], job["method"], job["strength"],
                                 patch=patch, angle=angle)
            writer.write(frame)
            n += 1
            job["processed"] = n
        cap.release()
        writer.release()
        if n == 0:
            raise RuntimeError("视频解码失败：未读到任何帧")

        out = OUTPUTS / f"{job_id}.mp4"
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-i", str(tmp_out), "-i", str(src),
               "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError("ffmpeg 转码失败: " + proc.stderr[-400:])
        tmp_out.unlink(missing_ok=True)
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/upload")
async def upload(
    video: UploadFile = File(...),
    method: str = Form("pixelate"),
    strength: float = Form(15),
    overlay: UploadFile | None = File(None),
    ref_face: UploadFile | None = File(None),
    swap_face: UploadFile | None = File(None),
    target_mode: str = Form("all"),
    ref_threshold: float = Form(0.5),
):
    if method not in METHODS:
        raise HTTPException(400, f"method 须为 {sorted(METHODS)} 之一")
    if target_mode not in TARGET_MODES:
        raise HTTPException(400, f"target_mode 须为 {sorted(TARGET_MODES)} 之一")
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{job_id}{suffix}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "文件超过 500MB 限制")
            f.write(chunk)

    job = {
        "id": job_id, "filename": video.filename, "suffix": suffix,
        "method": method, "strength": strength, "status": "queued",
        "processed": 0, "total": 0, "target_mode": target_mode,
        "ref_threshold": ref_threshold, "overlay_path": None,
        "ref_emb": None, "swap_path": None,
    }

    if swap_face is not None and swap_face.filename:
        sp = UPLOADS / f"{job_id}_swap{Path(swap_face.filename).suffix or '.png'}"
        with sp.open("wb") as f:
            f.write(await swap_face.read())
        job["swap_path"] = str(sp)  # prepare_swap_reference 校验人脸

    if ref_face is not None and ref_face.filename:
        rp = UPLOADS / f"{job_id}_ref{Path(ref_face.filename).suffix or '.jpg'}"
        with rp.open("wb") as f:
            f.write(await ref_face.read())
        emb = build_ref_embedding(rp)
        if emb is None:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "参考照片中未检测到人脸")
        job["ref_emb"] = emb

    if overlay is not None and overlay.filename and method == "overlay":
        ov = UPLOADS / f"{job_id}_ov{Path(overlay.filename).suffix or '.png'}"
        with ov.open("wb") as f:
            f.write(await overlay.read())
        job["overlay_path"] = str(ov)

    JOBS[job_id] = job
    threading.Thread(target=process_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "mode": "swap" if job["swap_path"] else "anonymize"}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return {k: job[k] for k in
            ("id", "filename", "method", "strength", "status",
             "processed", "total", "error") if k in job}


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    out = OUTPUTS / f"{job_id}.mp4"
    if not out.exists():
        raise HTTPException(404, "成品尚未生成")
    return FileResponse(out, media_type="video/mp4",
                        filename=f"mosaic_{job_id}.mp4")


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
