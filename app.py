import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
OWNER_NAME = "MOHD SAQIB"
MODEL_PATH = ROOT / "models" / "best.pt"
DATASET_DIR = ROOT / "dataset_videos"
DEFAULT_VIDEO = DATASET_DIR / "Project_3.mp4"
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "output"
METRICS_PATH = OUTPUT_DIR / "live_metrics.json"
HISTORY_PATH = OUTPUT_DIR / "queue_history.csv"
EVENTS_PATH = OUTPUT_DIR / "events.csv"
LATEST_FRAME_PATH = OUTPUT_DIR / "latest_frame.jpg"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return cleaned.strip("._") or "video.mp4"


def find_default_video() -> Path | None:
    if DEFAULT_VIDEO.exists():
        return DEFAULT_VIDEO

    if not DATASET_DIR.exists():
        return None

    videos = sorted(DATASET_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def parse_classes(value: str) -> list[int] | None:
    if value.strip().lower() in {"all", "none", ""}:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def write_json(path: Path, data: dict) -> None:
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reset_runtime_outputs(video_path: Path) -> None:
    ensure_runtime_dirs()
    for path in (HISTORY_PATH, EVENTS_PATH, LATEST_FRAME_PATH):
        if path.exists():
            path.unlink()

    write_json(
        METRICS_PATH,
        {
            "status": "ready",
            "video": str(video_path),
            "video_name": video_path.name,
            "started_at": None,
            "updated_at": now_text(),
            "message": "Ready to process",
            "frame": 0,
            "video_time": 0,
            "processing_fps": 0,
            "queue_length": 0,
            "avg_wait_time": 0,
            "max_wait_time": 0,
            "ema_service_time": None,
            "service_count": 0,
            "last_service_time": None,
            "accuracy_avg": None,
            "total_people_seen": 0,
            "active_tracks": [],
        },
    )


def draw_setup_overlay(frame, roi_points, direction_points, mode):
    temp = frame.copy()

    for point in roi_points:
        cv2.circle(temp, point, 6, (0, 0, 255), -1)

    if len(roi_points) >= 2:
        for idx in range(len(roi_points) - 1):
            cv2.line(temp, roi_points[idx], roi_points[idx + 1], (0, 255, 0), 2)

    if len(roi_points) == 4:
        cv2.line(temp, roi_points[3], roi_points[0], (0, 255, 0), 2)

    for idx, point in enumerate(direction_points):
        cv2.circle(temp, point, 6, (255, 0, 0), -1)
        label = "TAIL" if idx == 0 else "HEAD"
        cv2.putText(temp, label, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if len(direction_points) == 2:
        cv2.arrowedLine(temp, direction_points[0], direction_points[1], (255, 0, 0), 2)

    if mode == "roi":
        text = "Click 4 ROI points"
    elif mode == "direction":
        text = "Click TAIL -> HEAD"
    else:
        text = "Setup Complete"

    cv2.putText(temp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return temp


def collect_setup_points(frame):
    roi_points = []
    direction_points = []
    mode = {"value": "roi"}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if mode["value"] == "roi":
            roi_points.append((x, y))
            if len(roi_points) == 4:
                mode["value"] = "direction"
        elif mode["value"] == "direction":
            direction_points.append((x, y))
            if len(direction_points) == 2:
                mode["value"] = "done"

    cv2.namedWindow("Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Setup", 1000, 600)
    cv2.setMouseCallback("Setup", on_mouse)

    while True:
        temp = draw_setup_overlay(frame, roi_points, direction_points, mode["value"])
        cv2.imshow("Setup", temp)

        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), 27}:
            raise KeyboardInterrupt

        if mode["value"] == "done":
            cv2.imshow("Setup", temp)
            cv2.waitKey(800)
            break

    cv2.destroyWindow("Setup")
    return np.array(roi_points, dtype=np.int32), direction_points[0], direction_points[1]


def processor_metrics(
    *,
    args,
    status,
    message,
    frame_count=0,
    current_time=0,
    processing_fps=0,
    people=None,
    entry_time=None,
    ema_service_time=None,
    service_times=None,
    accuracies=None,
    seen_ids=None,
):
    people = people or []
    entry_time = entry_time or {}
    service_times = service_times or []
    accuracies = accuracies or []
    seen_ids = seen_ids or set()

    active_tracks = []
    waits = []
    for idx, (track_id, proj, cx, cy, predicted_wait) in enumerate(people):
        wait_time = max(0, current_time - entry_time.get(track_id, current_time))
        waits.append(wait_time)
        active_tracks.append(
            {
                "id": int(track_id),
                "position": idx + 1,
                "wait_time": round(wait_time, 1),
                "predicted_wait": predicted_wait,
                "x": int(cx),
                "y": int(cy),
            }
        )

    accuracy_avg = round(float(np.mean(accuracies)), 1) if accuracies else None
    data = {
        "status": status,
        "video": str(Path(args.video).resolve()),
        "video_name": Path(args.video).name,
        "started_at": getattr(args, "started_at", None),
        "updated_at": now_text(),
        "message": message,
        "frame": int(frame_count),
        "video_time": round(float(current_time), 2),
        "processing_fps": round(float(processing_fps), 2),
        "queue_length": len(people),
        "avg_wait_time": round(float(np.mean(waits)), 1) if waits else 0,
        "max_wait_time": round(float(np.max(waits)), 1) if waits else 0,
        "ema_service_time": round(float(ema_service_time), 1) if ema_service_time is not None else None,
        "service_count": len(service_times),
        "last_service_time": round(float(service_times[-1]), 1) if service_times else None,
        "accuracy_avg": accuracy_avg,
        "total_people_seen": len(seen_ids),
        "active_tracks": active_tracks,
    }
    write_json(METRICS_PATH, data)

    append_csv(
        HISTORY_PATH,
        [
            "updated_at",
            "video_name",
            "video_time",
            "queue_length",
            "avg_wait_time",
            "max_wait_time",
            "ema_service_time",
            "processing_fps",
            "accuracy_avg",
            "total_people_seen",
        ],
        {
            "updated_at": data["updated_at"],
            "video_name": data["video_name"],
            "video_time": data["video_time"],
            "queue_length": data["queue_length"],
            "avg_wait_time": data["avg_wait_time"],
            "max_wait_time": data["max_wait_time"],
            "ema_service_time": data["ema_service_time"],
            "processing_fps": data["processing_fps"],
            "accuracy_avg": data["accuracy_avg"],
            "total_people_seen": data["total_people_seen"],
        },
    )


def log_event(video_name: str, event: str, **values) -> None:
    row = {
        "updated_at": now_text(),
        "video_name": video_name,
        "event": event,
        "track_id": values.get("track_id"),
        "queue_position": values.get("queue_position"),
        "predicted_wait": values.get("predicted_wait"),
        "actual_wait": values.get("actual_wait"),
        "accuracy": values.get("accuracy"),
        "service_time": values.get("service_time"),
    }
    append_csv(EVENTS_PATH, list(row.keys()), row)


def process_video(args) -> None:
    ensure_runtime_dirs()
    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve() if args.video else find_default_video()

    if video_path is None:
        raise FileNotFoundError("No video found. Upload a video or keep Project_3.mp4 in dataset_videos.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    args.video = str(video_path)
    args.started_at = now_text()
    reset_runtime_outputs(video_path)
    processor_metrics(args=args, status="loading", message="Loading YOLO model")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_delay = max(1, int(1000 / fps))

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Video loaded but no frame could be read.")

    if args.no_window:
        h, w = frame.shape[:2]
        roi_polygon = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.int32)
        tail = (w // 2, h - 1)
        head = (w // 2, 0)
    else:
        processor_metrics(args=args, status="setup", message="Select ROI and queue direction in the OpenCV window")
        roi_polygon, tail, head = collect_setup_points(frame.copy())

    direction_vector = np.array(head) - np.array(tail)
    if float(np.linalg.norm(direction_vector)) == 0:
        raise RuntimeError("Direction tail and head cannot be the same point.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    entry_time = {}
    last_seen = {}
    prev_proj = {}
    recent_front_ids = {}
    service_times = []
    ema_service_time = None
    last_exit_time = None
    last_update = 0
    last_metrics_write = 0
    saved_prediction = {}
    prediction_time = {}
    prediction_position = {}
    accuracies = []
    seen_ids = set()
    cached_results = None
    start_perf = time.perf_counter()
    classes = parse_classes(args.classes)
    video_name = video_path.name

    if not args.no_window:
        cv2.namedWindow("Queue Monitoring", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Queue Monitoring", 1000, 600)

    processor_metrics(args=args, status="running", message="Processing video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time <= 0:
            current_time = frame_count / fps

        elapsed = max(time.perf_counter() - start_perf, 0.001)
        processing_fps = frame_count / elapsed
        original_h, original_w = frame.shape[:2]

        if frame_count % args.frame_skip == 0:
            resized_frame = cv2.resize(frame, (args.inference_width, args.inference_height))
            cached_results = model.track(
                resized_frame,
                persist=True,
                conf=args.conf,
                classes=classes,
                verbose=False,
            )

        raw_people = []
        results = cached_results

        if results is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            scale_x = original_w / args.inference_width
            scale_y = original_h / args.inference_height

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                if width < args.min_box_width or height < args.min_box_height:
                    continue

                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if cv2.pointPolygonTest(roi_polygon, (cx, cy), False) < 0:
                    continue

                point = np.array([cx, cy])
                proj = np.dot(point - np.array(tail), direction_vector)

                if track_id not in entry_time:
                    entry_time[track_id] = current_time
                    log_event(video_name, "entry", track_id=int(track_id))

                if current_time - entry_time[track_id] < args.min_track_time:
                    continue

                raw_people.append((track_id, proj, cx, cy))
                seen_ids.add(int(track_id))
                last_seen[track_id] = current_time
                prev_proj[track_id] = proj

        raw_people.sort(key=lambda item: item[1], reverse=True)

        for idx, (track_id, proj, cx, cy) in enumerate(raw_people):
            if idx + 1 <= 2:
                recent_front_ids[track_id] = current_time

        if current_time - last_update >= args.update_interval:
            current_ids = {person[0] for person in raw_people}

            for track_id in list(entry_time.keys()):
                if track_id in current_ids:
                    continue
                if track_id not in prev_proj:
                    continue
                if track_id not in recent_front_ids or current_time - recent_front_ids[track_id] > 3:
                    continue
                if current_time - last_seen.get(track_id, 0) < 0.3:
                    continue

                tracked_duration = current_time - entry_time[track_id]
                if tracked_duration < 1.5:
                    continue

                if last_exit_time is not None:
                    service_time = current_time - last_exit_time
                    valid = True

                    if len(service_times) >= 3:
                        recent_median = np.median(service_times[-3:])
                        if not 0.5 * recent_median <= service_time <= 1.5 * recent_median:
                            valid = False
                            log_event(video_name, "rejected_gap", track_id=int(track_id), service_time=round(service_time, 1))

                    if valid:
                        service_times.append(service_time)
                        log_event(video_name, "accepted_gap", track_id=int(track_id), service_time=round(service_time, 1))

                last_exit_time = current_time

                if track_id in saved_prediction:
                    actual_remaining = current_time - prediction_time[track_id]
                    predicted = saved_prediction[track_id]
                    error = abs(predicted - actual_remaining)
                    accuracy = max(0, (1 - (error / actual_remaining)) * 100) if actual_remaining > 0 else 0
                    accuracies.append(accuracy)
                    position = prediction_position[track_id]
                    log_event(
                        video_name,
                        "prediction_result",
                        track_id=int(track_id),
                        queue_position=int(position),
                        predicted_wait=round(float(predicted), 1),
                        actual_wait=round(float(actual_remaining), 1),
                        accuracy=round(float(accuracy), 1),
                    )

                log_event(video_name, "exit", track_id=int(track_id))
                entry_time.pop(track_id, None)
                last_seen.pop(track_id, None)
                prev_proj.pop(track_id, None)
                break

            if len(service_times) >= 2:
                recent = service_times[-5:]
                mean_recent = np.mean(recent)
                variation = np.std(recent)
                norm_var = variation / mean_recent if mean_recent > 0 else 0
                norm_var = min(norm_var, 1)
                alpha = max(0.4, min(0.9, 0.75 - 0.35 * norm_var))
                latest_service = recent[-1]

                if ema_service_time is None:
                    ema_service_time = latest_service
                else:
                    ema_service_time = alpha * ema_service_time + (1 - alpha) * latest_service

            last_update = current_time

        people = []
        for idx, (track_id, proj, cx, cy) in enumerate(raw_people):
            position = idx + 1
            real_wait = current_time - entry_time.get(track_id, current_time)
            wait_time = int(real_wait)

            if ema_service_time is None:
                predicted_wait = "Learning"
            elif position == 1:
                predicted_wait = 0
            else:
                predicted_wait = max(0, int((position - 1) * ema_service_time))

                if position >= 2 and track_id not in saved_prediction and len(service_times) >= 2:
                    saved_prediction[track_id] = predicted_wait
                    prediction_time[track_id] = current_time
                    prediction_position[track_id] = position
                    log_event(
                        video_name,
                        "prediction_saved",
                        track_id=int(track_id),
                        queue_position=int(position),
                        predicted_wait=int(predicted_wait),
                    )

            people.append((track_id, proj, cx, cy, predicted_wait))

            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            label = f"ID:{track_id} Pos:{position} W:{wait_time}s P:{predicted_wait}"
            if isinstance(predicted_wait, int):
                label += "s"
            cv2.putText(frame, label, (cx - 75, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.polylines(frame, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.arrowedLine(frame, tail, head, (255, 0, 0), 2)
        cv2.putText(frame, f"Queue Length: {len(people)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        service_text = f"EMA Service: {ema_service_time:.1f}s" if ema_service_time is not None else "EMA Learning"
        cv2.putText(frame, service_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {processing_fps:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if frame_count % 10 == 0:
            preview = cv2.resize(frame, (960, int(960 * original_h / original_w)))
            cv2.imwrite(str(LATEST_FRAME_PATH), preview)

        if current_time - last_metrics_write >= 0.75:
            processor_metrics(
                args=args,
                status="running",
                message="Processing video",
                frame_count=frame_count,
                current_time=current_time,
                processing_fps=processing_fps,
                people=people,
                entry_time=entry_time,
                ema_service_time=ema_service_time,
                service_times=service_times,
                accuracies=accuracies,
                seen_ids=seen_ids,
            )
            last_metrics_write = current_time

        if not args.no_window:
            cv2.imshow("Queue Monitoring", frame)
            if cv2.waitKey(frame_delay) & 0xFF in {ord("q"), 27}:
                break

    processor_metrics(
        args=args,
        status="finished",
        message="Processing finished",
        frame_count=frame_count,
        current_time=current_time if "current_time" in locals() else 0,
        processing_fps=processing_fps if "processing_fps" in locals() else 0,
        people=[],
        entry_time=entry_time,
        ema_service_time=ema_service_time,
        service_times=service_times,
        accuracies=accuracies,
        seen_ids=seen_ids,
    )

    cap.release()
    cv2.destroyAllWindows()


def build_processor_parser():
    parser = argparse.ArgumentParser(description=f"Vision-Based Queue Analytics - {OWNER_NAME}")
    parser.add_argument("--process", action="store_true", help="Run the OpenCV processor mode")
    parser.add_argument("--model", default=str(MODEL_PATH), help="YOLO model path")
    parser.add_argument("--video", default=None, help="Video path")
    parser.add_argument("--classes", default="0", help="Class ids to track, comma-separated. Use all for no filter.")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    parser.add_argument("--frame-skip", type=int, default=6, help="Run YOLO every N frames")
    parser.add_argument("--inference-width", type=int, default=960, help="Inference resize width")
    parser.add_argument("--inference-height", type=int, default=540, help="Inference resize height")
    parser.add_argument("--min-track-time", type=float, default=1.0, help="Seconds before a track is counted")
    parser.add_argument("--update-interval", type=float, default=1.5, help="Seconds between service updates")
    parser.add_argument("--min-box-width", type=float, default=40, help="Minimum detection box width")
    parser.add_argument("--min-box-height", type=float, default=80, help="Minimum detection box height")
    parser.add_argument("--no-window", action="store_true", help="Use full-frame ROI and do not open OpenCV windows")
    return parser


def save_uploaded_video(uploaded_file) -> Path:
    ensure_runtime_dirs()
    data = uploaded_file.getbuffer()
    digest = hashlib.sha256(data).hexdigest()[:12]
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}_{safe_name(Path(uploaded_file.name).stem)}{suffix}"
    path = UPLOAD_DIR / filename
    if not path.exists():
        path.write_bytes(data)
    return path


def launch_processor(video_path: Path, settings: dict) -> tuple[int, Path]:
    ensure_runtime_dirs()
    reset_runtime_outputs(video_path)
    log_path = OUTPUT_DIR / f"processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--process",
        "--video",
        str(video_path),
        "--model",
        settings["model"],
        "--conf",
        str(settings["conf"]),
        "--frame-skip",
        str(settings["frame_skip"]),
        "--classes",
        settings["classes"],
        "--min-track-time",
        str(settings["min_track_time"]),
    ]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    return process.pid, log_path


def stop_process(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=10)
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def read_history_dataframe():
    import pandas as pd

    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


def read_events_dataframe():
    import pandas as pd

    if not EVENTS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(EVENTS_PATH)
    except Exception:
        return pd.DataFrame()


def style_frontend(st):
    st.markdown(
        """
        <style>
        .stApp {
            background: #10131a;
            color: #f4f7fb;
        }
        .main .block-container {
            max-width: 1240px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #f8fafc;
            letter-spacing: 0;
        }
        p, label, span {
            color: #cfd7e3;
        }
        [data-testid="stSidebar"] {
            background: #151923;
            border-right: 1px solid #252c3a;
        }
        div[data-testid="stMetric"] {
            background: #171c27;
            border: 1px solid #2a3343;
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }
        div[data-testid="stMetric"] * {
            color: #f8fafc;
        }
        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid #2f3a4d;
            background: #1b2330;
            color: #f8fafc;
            font-weight: 650;
        }
        div.stButton > button[kind="primary"] {
            background: #0f8b8d;
            border-color: #0f8b8d;
            color: #ffffff;
        }
        [data-testid="stFileUploader"] section {
            background: #171c27;
            border: 1px dashed #56657d;
            border-radius: 8px;
        }
        .app-band {
            border: 1px solid #293244;
            border-radius: 8px;
            background: #171c27;
            padding: 1rem 1.1rem;
            margin: 0.4rem 0 1rem 0;
        }
        .chip {
            display: inline-block;
            border: 1px solid #314056;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            margin: 0 0.35rem 0.35rem 0;
            color: #d8e2ef;
            font-size: 0.8rem;
            background: #1b2330;
        }
        .muted {
            color: #93a4b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(st, metrics: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Queue Length", metrics.get("queue_length", 0))
    col2.metric("Average Wait", f"{metrics.get('avg_wait_time', 0)}s")
    fps = metrics.get("processing_fps", 0)
    col3.metric("Processing FPS", f"{fps:.1f}" if isinstance(fps, (int, float)) else fps)
    service = metrics.get("ema_service_time")
    col4.metric("EMA Service", "Learning" if service is None else f"{service}s")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("People Seen", metrics.get("total_people_seen", 0))
    col6.metric("Service Samples", metrics.get("service_count", 0))
    col7.metric("Prediction Accuracy", "N/A" if metrics.get("accuracy_avg") is None else f"{metrics['accuracy_avg']}%")
    col8.metric("Last Update", metrics.get("updated_at", "N/A"))


def render_dashboard(st, key_prefix: str):
    import plotly.express as px
    import plotly.graph_objects as go

    metrics = read_json(METRICS_PATH, {})
    status = metrics.get("status", "idle")

    st.markdown(
        f"""
        <div class="app-band">
            <strong>Vision-Based Queue Analytics</strong>
            <div class="muted">Prepared by {OWNER_NAME}</div>
            <div class="muted">Status: {status} | Video: {metrics.get("video_name", "No active video")}</div>
            <div style="margin-top:0.55rem">
                <span class="chip">YOLOv8 person detection</span>
                <span class="chip">Multi-object tracking</span>
                <span class="chip">Polygon ROI</span>
                <span class="chip">Adaptive EMA waiting time</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_cards(st, metrics)

    frame_col, table_col = st.columns([1.25, 1])
    with frame_col:
        st.subheader("Processed Video Preview")
        if LATEST_FRAME_PATH.exists():
            st.image(str(LATEST_FRAME_PATH), width="stretch")
        else:
            st.info("Start processing to see the annotated video preview.")

    with table_col:
        st.subheader("Active Queue")
        active_tracks = metrics.get("active_tracks", [])
        if active_tracks:
            st.dataframe(active_tracks, width="stretch", hide_index=True, key=f"{key_prefix}_active_tracks")
        else:
            st.info("No active tracks yet.")

    history = read_history_dataframe()
    if history.empty:
        st.info("Live charts will appear after processing starts.")
        return

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = px.line(history, x="video_time", y="queue_length", title="Queue Length Over Time")
        fig.update_traces(line_color="#0f8b8d", line_width=3)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#10131a", plot_bgcolor="#111722")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_queue_length_chart")

    with chart_col2:
        fig = px.line(history, x="video_time", y="avg_wait_time", title="Average Waiting Time")
        fig.update_traces(line_color="#f2a541", line_width=3)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#10131a", plot_bgcolor="#111722")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_avg_wait_chart")

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        fig = go.Figure()
        if "ema_service_time" in history:
            fig.add_trace(go.Scatter(x=history["video_time"], y=history["ema_service_time"], mode="lines", name="EMA Service"))
        fig.update_layout(
            title="Service Time Adaptation",
            template="plotly_dark",
            paper_bgcolor="#10131a",
            plot_bgcolor="#111722",
            xaxis_title="Video Time",
            yaxis_title="Seconds",
        )
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_service_time_chart")

    with chart_col4:
        fig = px.line(history, x="video_time", y="processing_fps", title="Processing FPS")
        fig.update_traces(line_color="#d95d39", line_width=3)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#10131a", plot_bgcolor="#111722")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_processing_fps_chart")


def render_process_page(st):
    st.subheader("Upload & Process Video")

    uploaded = st.file_uploader("Choose video file", type=["mp4", "avi", "mov", "mkv", "mpeg"])
    selected_video = None

    if uploaded is not None:
        selected_video = save_uploaded_video(uploaded)
        st.success(f"Uploaded: {selected_video.name}")
    else:
        sample = find_default_video()
        if sample is not None:
            selected_video = sample
            st.caption(f"Using sample video: {sample.name}")
        else:
            st.warning("Upload a video to begin.")

    settings = {
        "model": str(MODEL_PATH),
        "conf": st.session_state.get("conf", 0.5),
        "frame_skip": st.session_state.get("frame_skip", 6),
        "classes": st.session_state.get("classes", "0"),
        "min_track_time": st.session_state.get("min_track_time", 1.0),
    }

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start = st.button("Open Processing Window", type="primary", width="stretch", disabled=selected_video is None)
    with col2:
        stop = st.button("Stop Processor", width="stretch", disabled="processor_pid" not in st.session_state)
    with col3:
        clear = st.button("Clear Runtime Data", width="stretch")

    if start and selected_video is not None:
        pid, log_path = launch_processor(selected_video, settings)
        st.session_state.processor_pid = pid
        st.session_state.processor_log = str(log_path)
        st.success(f"Processor started. PID: {pid}")

    if stop:
        if stop_process(int(st.session_state.get("processor_pid", 0))):
            st.session_state.pop("processor_pid", None)
            st.warning("Processor stop requested.")

    if clear:
        reset_runtime_outputs(selected_video or find_default_video() or DEFAULT_VIDEO)
        st.info("Runtime dashboard data cleared.")

    if "processor_pid" in st.session_state:
        st.caption(f"Current processor PID: {st.session_state.processor_pid}")
    if "processor_log" in st.session_state:
        st.caption(f"Log: {st.session_state.processor_log}")

    st.divider()
    st.subheader("Dashboard")
    render_dashboard(st, "process")


def render_reports_page(st):
    events = read_events_dataframe()
    history = read_history_dataframe()

    st.subheader("Events")
    if events.empty:
        st.info("No events recorded yet.")
    else:
        st.dataframe(events.tail(200), width="stretch", hide_index=True)
        st.download_button("Download Events CSV", EVENTS_PATH.read_bytes(), file_name="queue_events.csv")

    st.subheader("History")
    if history.empty:
        st.info("No history recorded yet.")
    else:
        st.dataframe(history.tail(200), width="stretch", hide_index=True)
        st.download_button("Download History CSV", HISTORY_PATH.read_bytes(), file_name="queue_history.csv")


def run_frontend():
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    ensure_runtime_dirs()
    st.set_page_config(page_title=f"Vision-Based Queue Analytics | {OWNER_NAME}", layout="wide", initial_sidebar_state="expanded")
    st_autorefresh(interval=2500, key="queue_dashboard_refresh")
    style_frontend(st)

    st.sidebar.title("Controls")
    st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05, key="conf")
    st.sidebar.slider("Frame Skip", 1, 12, 6, 1, key="frame_skip")
    st.sidebar.slider("Minimum Track Time", 0.2, 3.0, 1.0, 0.1, key="min_track_time")
    st.sidebar.text_input("YOLO Class Filter", value="0", key="classes")
    st.sidebar.caption(f"Prepared by {OWNER_NAME}")
    st.sidebar.caption(f"Model: {MODEL_PATH.name}")

    st.title("Vision-Based Queue Analytics")
    st.caption(f"Prepared by {OWNER_NAME}")
    tabs = st.tabs(["Process Video", "Live Dashboard", "Reports"])

    with tabs[0]:
        render_process_page(st)
    with tabs[1]:
        render_dashboard(st, "live")
    with tabs[2]:
        render_reports_page(st)


def running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def launch_streamlit() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
    subprocess.run(cmd, cwd=str(ROOT))


def main() -> None:
    if "--process" in sys.argv:
        args = build_processor_parser().parse_args()
        try:
            process_video(args)
        except KeyboardInterrupt:
            write_json(METRICS_PATH, {"status": "stopped", "updated_at": now_text(), "message": "Stopped by user"})
            cv2.destroyAllWindows()
        except Exception as exc:
            ensure_runtime_dirs()
            write_json(METRICS_PATH, {"status": "error", "updated_at": now_text(), "message": str(exc)})
            cv2.destroyAllWindows()
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if running_inside_streamlit():
        run_frontend()
    else:
        launch_streamlit()


if __name__ == "__main__":
    main()
