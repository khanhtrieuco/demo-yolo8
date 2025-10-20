from flask import Flask, request, jsonify, render_template
from providers.db_provider_postgres import DBProvider
from providers.ai_provider_vertex import AIProviderVertex
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
db = DBProvider()
ai = AIProviderVertex()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/") 
def index(): 
    refs = db.list_all_references()
    return render_template('index.html', refs = refs)

@app.route("/check-history/<int:ref_id>", methods = ["GET"])
def get_check_history(ref_id):
    check_history = db.get_check_result(ref_id)
    return jsonify({
        "code": 200,
        "message": f"Lấy lịch sử check ref_id {ref_id} thành công",
        "data": check_history
    })
# === Upload reference ===
@app.route("/upload/reference", methods=["POST"])
def upload_reference():
    print('request form data:', request.form)
    code = request.form.get("code")
    token = request.form.get("token", "123456")
    description = request.form.get("description")
    image = request.files.get("image")
    image_url = request.form.get("image_url")
    if not token:
        return jsonify({
            "error": "Forbiden access",
            "code": 403,
            "data" : "null",
            "message": "You must provide token"
        })
    if not image_url:
        return jsonify({
            "error": "Bad request",
            "code": 400,
            "data" : "null",
            "message": "Vui lòng cung cấp link ảnh"
        })
    if not description:
        return jsonify({
            "error": "Bad request",
            "code": 400,
            "data" : "null",
            "message": "Vui lòng cung cấp mô tả"
        })
    if not code:
        return jsonify({
            "error": "Bad request",
            "code": 400,
            "data" : "null",
            "message": "Vui lòng cung cấp code"
        })
   
    os.makedirs(os.path.join(UPLOAD_FOLDER, "reference"), exist_ok=True)
    path = image_url or os.path.join(UPLOAD_FOLDER, "reference", image.filename)
    if image:
        image.save(path)

    ref_id = db.insert_or_update_reference(code, path, description)
    return jsonify({
        "code": 200,
        "message": f"Đã cập nhật ảnh chuẩn {code}",
        "data": {
            "reference_id": ref_id,
            "path": path,
            "description": description
        }
    })


# === Upload test ===
@app.route("/upload/test", methods=["POST"])
def upload_test():
    ref_code = request.form.get("reference_code")
    image = request.form.get("image")
    image_url = request.form.get("image_url_check")

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
        "code": 200,
        "message": "Check success",
        "data": {
            "reference_code": ref_code,
            "reference_id": ref["id"],
            **result
        }
    })


@app.route("/references", methods=["GET"])
def list_references():
    return jsonify(db.list_all_references())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5999, debug=True)
