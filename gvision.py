
from google.cloud import vision
import io, cv2, numpy as np, json, math

# === HÀM 1: Dò vật thể bằng Vision AI ===
def detect_and_draw(image_path, output_prefix, client):
    print(f"🔍 Đang xử lý ảnh: {image_path}")
    with io.open(image_path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)

    # Phát hiện vật thể
    resp_obj = client.object_localization(image=image)
    objects = resp_obj.localized_object_annotations

    # Phát hiện nhãn hiệu (brand)
    resp_label = client.label_detection(image=image)
    labels = resp_label.label_annotations
    brand_labels = [l.description for l in labels if l.score > 0.7]

    # Chuẩn bị ảnh vẽ
    np_image = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    detected_items = []
    for obj in objects:
        vertices = [(int(v.x * w), int(v.y * h)) for v in obj.bounding_poly.normalized_vertices]
        cx, cy = sum(v[0] for v in vertices) / 4, sum(v[1] for v in vertices) / 4

        detected_items.append({
            "name": obj.name,
            "confidence": round(obj.score, 3),
            "center": {"x": cx, "y": cy},
            "bounding_box": [{"x": v[0], "y": v[1]} for v in vertices]
        })

        # Vẽ bounding box xanh cho vật thể phát hiện được
        pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.putText(img, f"{obj.name} ({obj.score:.2f})",
                    (vertices[0][0], vertices[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Lưu kết quả JSON + ảnh
    json_path, img_path = f"{output_prefix}.json", f"{output_prefix}.jpg"
    data = {"image_path": image_path, "detected_products": detected_items, "possible_brands": brand_labels}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    cv2.imwrite(img_path, img)

    print(f"✅ Đã lưu: {json_path}, {img_path}")
    return data, img


# === HÀM 2: So sánh hai ảnh, chỉ đánh dấu thiếu và thừa ===
def compare_and_mark(original, rearranged, img_test, tolerance=50):
    a_objs, b_objs = original["detected_products"], rearranged["detected_products"]
    matches, missing, extra = [], [], []
    used = set()

    # So sánh vật thể theo loại + vị trí gần nhất
    for a in a_objs:
        same_type = [b for b in b_objs if b["name"] == a["name"]]
        if not same_type:
            missing.append(a)
            continue
        nearest = min(same_type, key=lambda b: math.dist(
            (a["center"]["x"], a["center"]["y"]),
            (b["center"]["x"], b["center"]["y"])
        ))
        dist = math.dist((a["center"]["x"], a["center"]["y"]), (nearest["center"]["x"], nearest["center"]["y"]))
        if dist <= tolerance:
            matches.append(a)
        else:
            # xem như khác vị trí quá xa => mất hẳn
            missing.append(a)
        used.add(id(nearest))

    # Các vật thể test không match với gốc → thừa
    for b in b_objs:
        if id(b) not in used and not any(b["name"] == a["name"] for a in a_objs):
            extra.append(b)

    # === Vẽ bounding box đặc biệt ===
    def draw_box(item, color, label):
        pts = np.array([[v["x"], v["y"]] for v in item["bounding_box"]], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_test, [pts], True, color, 3)
        cv2.putText(img_test, f"{item['name']} {label}",
                    (item["bounding_box"][0]["x"], item["bounding_box"][0]["y"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Đỏ: vật bị thiếu (trong gốc nhưng không thấy ở test)
    for m in missing:
        draw_box(m, (0, 0, 255), "MISSING")

    # Xanh dương: vật thừa (chỉ xuất hiện trong test)
    for e in extra:
        draw_box(e, (255, 0, 0), "EXTRA")

    # Tính % giống nhau
    total = len(a_objs)
    score = round((len(matches) / total) * 100, 2) if total else 0

    cv2.imwrite("rearranged_compared.jpg", img_test)

    result = {
        "similarity_score": score,
        "matches": [m["name"] for m in matches],
        "missing": [{"name": m["name"], "center": m["center"], "bbox": m["bounding_box"]} for m in missing],
        "extra": [{"name": e["name"], "center": e["center"], "bbox": e["bounding_box"]} for e in extra],
        "summary": f"Độ giống nhau: {score}%. Thiếu {len(missing)}, thừa {len(extra)}."
    }
    return result


# === MAIN SCRIPT ===
if __name__ == "__main__":
    client = vision.ImageAnnotatorClient.from_service_account_json('gcv_key.json')

    original_image = "960x0-1577292486835725328574-1577292520144208949576.webp"
    rearranged_image = "960x0-1577292486835725328574-1577292520144208949576.png"

    # Phân tích 2 ảnh
    original_json, _ = detect_and_draw(original_image, "original_detected", client)
    rearranged_json, img_rearranged = detect_and_draw(rearranged_image, "rearranged_detected", client)

    # So sánh + khoanh khác biệt
    print("\n🔎 So sánh bố cục...")
    result = compare_and_mark(original_json, rearranged_json, img_rearranged)

    # Xuất JSON kết quả
    with open("comparison_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n🎯 KẾT QUẢ CUỐI CÙNG:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n🖼️ Ảnh khác biệt được lưu: rearranged_compared.jpg")
