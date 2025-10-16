from flask import Flask, request, jsonify
from providers.db_provider_postgres import DBProvider
from providers.ai_provider_vertex import AIProviderVertex
import os

app = Flask(__name__)
db = DBProvider()
ai = AIProviderVertex()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# === Upload reference ===
@app.route("/upload/reference", methods=["POST"])
def upload_reference():
    code = request.form.get("code")
    description = request.form.get("description", "")
    image = request.files.get("image")
    image_url = request.form.get("image_url")

    if not (image or image_url):
        return jsonify({"error": "Phải gửi file hoặc URL ảnh!"}), 400

    os.makedirs(os.path.join(UPLOAD_FOLDER, "reference"), exist_ok=True)
    path = image_url or os.path.join(UPLOAD_FOLDER, "reference", image.filename)
    if image:
        image.save(path)

    ref_id = db.insert_or_update_reference(code, path, description)
    return jsonify({
        "message": f"Đã cập nhật ảnh chuẩn {code}",
        "reference_id": ref_id,
        "path": path,
        "description": description
    })


# === Upload test ===
@app.route("/upload/test", methods=["POST"])
def upload_test():
    ref_code = request.form["reference_code"]
    image = request.files.get("image")
    image_url = request.form.get("image_url")

    if not (image or image_url):
        return jsonify({"error": "Phải gửi file hoặc URL ảnh!"}), 400

    test_path = image_url or os.path.join(UPLOAD_FOLDER, "test", image.filename)
    if image:
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        image.save(test_path)

    ref = db.get_reference_image_by_code(ref_code)
    if not ref:
        return jsonify({"error": "Không tìm thấy mẫu!"}), 404

    ref_path = ref.get("path") or ref.get("url")
    result = ai.analyze(ref_path, test_path, ref.get("description", ""))

    db.insert_check_result(ref["id"], test_path, result["score"], result["status"], result["summary"])

    return jsonify({
        "reference_code": ref_code,
        "reference_id": ref["id"],
        **result
    })


# === List all references ===
@app.route("/references", methods=["GET"])
def list_references():
    return jsonify(db.list_all_references())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
