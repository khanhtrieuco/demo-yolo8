import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory, url_for
from PIL import Image
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
TARGET_CLASS = os.environ.get("TARGET_CLASS", "bottle")
TOP_HALF_MIN_BOTTLES = int(os.environ.get("TOP_HALF_MIN_BOTTLES", 10))
BOTTOM_HALF_MAX_BOTTLES = int(os.environ.get("BOTTOM_HALF_MAX_BOTTLES", 0))
CENTER_LINE_RATIO = float(os.environ.get("CENTER_LINE_RATIO", 0.55))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.4))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


logger.info("Loading YOLO model from %s", MODEL_PATH)
model = YOLO(MODEL_PATH)
logger.info("Model loaded successfully")


app = Flask(__name__)


class ProcessingError(Exception):
    """Raised when an image cannot be processed."""


def _download_image(image_url: str, extra_headers: Optional[Dict[str, str]] = None) -> Image.Image:
    """Download an image from a URL and return a PIL image."""

    headers = DEFAULT_REQUEST_HEADERS.copy()
    if extra_headers:
        headers.update(extra_headers)

    try:
        response = requests.get(image_url, stream=True, timeout=15, headers=headers)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ProcessingError(f"Không thể tải ảnh từ URL: {exc}") from exc

    try:
        raw_data = BytesIO(response.content)
        image = Image.open(raw_data).convert("RGB")
        return image
    except Exception as exc:  # pylint: disable=broad-except
        raise ProcessingError("Dữ liệu tải về không phải là ảnh hợp lệ.") from exc


def _run_inference(image: Image.Image) -> Tuple[Dict[str, int], np.ndarray, Dict[str, bool]]:
    """Run YOLO inference and apply the business rules."""

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width, _ = image_cv.shape
    center_line_y = int(height * CENTER_LINE_RATIO)

    results = model.predict(source=image, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_idx = int(box.cls[0])
            names = model.names
            if isinstance(names, dict):
                class_name = names.get(class_idx, str(class_idx))
            elif isinstance(names, (list, tuple)) and 0 <= class_idx < len(names):
                class_name = names[class_idx]
            else:
                class_name = str(class_idx)
            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "box": [int(i) for i in box.xyxy[0]],
                }
            )

    bottles_in_top_half = 0
    bottles_in_bottom_half = 0

    for det in detections:
        if det["class"] != TARGET_CLASS:
            continue

        x1, y1, x2, y2 = det["box"]
        object_center_y = (y1 + y2) / 2

        if object_center_y < center_line_y:
            bottles_in_top_half += 1
        else:
            bottles_in_bottom_half += 1

    rule_results = {
        "rule1_passed": bottles_in_top_half >= TOP_HALF_MIN_BOTTLES,
        "rule2_passed": bottles_in_bottom_half <= BOTTOM_HALF_MAX_BOTTLES,
    }

    # Visualization
    viz_image = image_cv.copy()
    cv2.line(viz_image, (0, center_line_y), (width, center_line_y), (255, 255, 0), 2)
    cv2.putText(
        viz_image,
        "TOP HALF",
        (5, max(center_line_y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        viz_image,
        "BOTTOM HALF",
        (5, center_line_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class']} {det['confidence']:.2f}"

        color = (0, 0, 255)
        object_center_y = (y1 + y2) / 2
        if det["class"] == TARGET_CLASS and object_center_y < center_line_y:
            color = (0, 255, 0)

        cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            viz_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    counts = {
        "bottles_in_top_half": bottles_in_top_half,
        "bottles_in_bottom_half": bottles_in_bottom_half,
        "total_detections": len(detections),
        "image_width": width,
        "image_height": height,
    }

    return counts, viz_image, rule_results


def _save_visualization(image: np.ndarray) -> Path:
    filename = f"result_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), image)
    return output_path


@app.route("/process", methods=["POST"])
def process_image():
    payload = request.get_json(silent=True) or {}
    image_url = payload.get("image_url")
    raw_headers = payload.get("headers")
    request_headers: Dict[str, str] = {}

    if raw_headers is not None:
        if not isinstance(raw_headers, dict):
            return jsonify({"error": "Trường 'headers' phải là một object."}), 400
        request_headers = {str(k): str(v) for k, v in raw_headers.items() if v is not None}

    if not image_url:
        return jsonify({"error": "Thiếu trường 'image_url'."}), 400

    logger.info("Received image for processing: %s", image_url)

    try:
        image = _download_image(image_url, request_headers or None)
        counts, viz_image, rule_results = _run_inference(image)
        saved_path = _save_visualization(viz_image)
    except ProcessingError as exc:
        logger.exception("Image processing failed")
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error during processing")
        return jsonify({"error": "Lỗi không xác định khi xử lý ảnh."}), 500

    relative_path = saved_path.relative_to(OUTPUT_DIR)
    result_url = urljoin(request.host_url, url_for("get_result", filename=str(relative_path)))

    response = {
        "result_image_url": result_url,
        "counts": {
            "bottles_in_top_half": counts["bottles_in_top_half"],
            "bottles_in_bottom_half": counts["bottles_in_bottom_half"],
            "total_detections": counts["total_detections"],
        },
        "rules": rule_results,
        "image_dimensions": {
            "width": counts["image_width"],
            "height": counts["image_height"],
        },
    }

    logger.info(
        "Processing complete. Top: %s, Bottom: %s",
        counts["bottles_in_top_half"],
        counts["bottles_in_bottom_half"],
    )

    return jsonify(response), 200


@app.route("/results/<path:filename>")
def get_result(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/health", methods=["GET"])
def health() -> Tuple[str, int]:
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
