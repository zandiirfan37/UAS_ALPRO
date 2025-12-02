import os
import io
import json
from datetime import datetime
from pathlib import Path
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw

# ------------------------------------------------------------
# COLOR LOGGING
# ------------------------------------------------------------
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

    class Dummy:
        RESET_ALL = ""

    class DummyFore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = ""

    Fore = DummyFore()
    Style = Dummy()


def log_info(msg: str):
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}")


def log_warn(msg: str):
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}")


def log_error(msg: str):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")


def log_success(msg: str):
    print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} {msg}")


log_info(f"app_yolo_server_pro.py dieksekusi, __name__ = {__name__}")

# ------------------------------------------------------------
# IMPORT YOLO
# ------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    log_success("ultralytics terdeteksi")
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    log_error("ultralytics belum terinstall. Jalankan: pip install ultralytics")

# ------------------------------------------------------------
# KONFIGURASI FLASK
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# DIREKTORI & ENV
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = Path(os.getenv("YOLO_SAVE_DIR", BASE_DIR / "detections"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
log_info(f"Folder penyimpanan hasil deteksi: {SAVE_DIR}")

# Threshold confidence
MIN_CONF_DET = float(os.getenv("YOLO_MIN_CONF_DET", "0.30"))
MIN_CONF_CLS = float(os.getenv("YOLO_MIN_CONF_CLS", "0.30"))

# untuk debug health
MODEL_PATH_USED: Path | None = None

# ------------------------------------------------------------
# DATA PENYAKIT KULIT (LABEL KELAS)
# ------------------------------------------------------------
SKIN_DISEASES = [
    {"label": "Acne", "description": "Peradangan pada folikel rambut dan kelenjar minyak yang menimbulkan komedo dan jerawat."},
    {"label": "Actinic Keratosis", "description": "Bercak kulit kasar dan bersisik akibat paparan sinar matahari jangka panjang, bisa menjadi prekanker."},
    {"label": "Benign Tumors", "description": "Kelainan kulit berupa tumor jinak yang tidak menyebar ke organ lain."},
    {"label": "Bullous", "description": "Kelompok penyakit kulit yang ditandai dengan gelembung atau lepuh berisi cairan."},
    {"label": "Candidiasis", "description": "Infeksi jamur Candida yang sering muncul di lipatan kulit yang lembap."},
    {"label": "Drug Eruption", "description": "Reaksi kulit terhadap obat, biasanya berupa ruam, kemerahan, atau gatal."},
    {"label": "Eczema", "description": "Peradangan kulit kronis yang menyebabkan kemerahan, gatal, dan kulit kering atau pecah."},
    {"label": "Infestations/Bites", "description": "Kelainan kulit akibat gigitan serangga atau infestasi parasit seperti tungau dan kutu."},
    {"label": "Lichen", "description": "Kelompok penyakit kulit dengan bercak atau papul kecil yang sering terasa gatal."},
    {"label": "Lupus", "description": "Penyakit autoimun yang dapat menimbulkan ruam khas pada kulit, termasuk di wajah."},
    {"label": "Moles", "description": "Tahi lalat atau nevus berpigmen yang biasanya jinak, tetapi perlu dipantau perubahannya."},
    {"label": "Psoriasis", "description": "Penyakit autoimun kronis yang menyebabkan plak merah bersisik pada kulit."},
    {"label": "Rosacea", "description": "Peradangan kulit wajah dengan kemerahan menetap, pembuluh darah tampak, dan kadang papul/pustul."},
    {"label": "Seborrheic Keratoses", "description": "Tumor jinak berwarna cokelat kehitaman dengan permukaan seperti tertempel di kulit."},
    {"label": "Skin Cancer", "description": "Kelainan kulit ganas yang dapat muncul sebagai benjolan, luka, atau bercak yang berubah bentuk/warna."},
    {"label": "Sun/Sunlight Damage", "description": "Kerusakan kulit karena paparan sinar matahari kronis, seperti keriput, bercak, atau perubahan warna."},
    {"label": "Tinea", "description": "Infeksi jamur dermatofit pada kulit, sering disebut kurap, dengan bercak melingkar."},
    {"label": "Unknown/Normal", "description": "Gambaran kulit yang tidak menunjukkan kelainan jelas atau dianggap dalam batas normal oleh model."},
    {"label": "Vascular Tumors", "description": "Lesi kulit yang berasal dari pembuluh darah, seperti hemangioma."},
    {"label": "Vasculitis", "description": "Peradangan pembuluh darah yang menimbulkan bercak atau purpura pada kulit."},
    {"label": "Vitiligo", "description": "Kehilangan pigmen kulit yang menyebabkan bercak putih tidak gatal."},
    {"label": "Warts", "description": "Kutil akibat infeksi virus HPV, sering muncul pada tangan, kaki, atau area lain."},
]
DESC_MAP = {d["label"]: d["description"] for d in SKIN_DISEASES}

# ------------------------------------------------------------
# LOAD YOLO MODEL (yolomodelbest3.pt di folder yang sama)
# ------------------------------------------------------------
def load_yolo_model():
    if not YOLO_AVAILABLE:
        log_error("ultralytics tidak tersedia, YOLO tidak bisa dipakai.")
        return None

    log_info("========== YOLO MODEL LOADER ==========")

    # Pakai file 'yolomodelbest3.pt' di folder yang sama dengan appp2.py
    model_path = Path(__file__).resolve().parent / "yolomodelbest3.pt"
    log_info(f"Mencari model di: {model_path}")

    if not model_path.exists():
        log_error(f"File model TIDAK DITEMUKAN di: {model_path}")
        return None

    try:
        log_info(f"Memuat YOLO model dari: {model_path}")
        model = YOLO(str(model_path))
        log_success("Model YOLO berhasil dimuat.")
        log_info(f"Daftar kelas model: {model.names}")
        log_info("========================================")
        return model
    except Exception as e:
        log_error(f"Gagal memuat YOLO dari {model_path}: {e}")
        return None

yolo_model = load_yolo_model()

# ------------------------------------------------------------
# FUNGSI: DETEKSI YOLO (DETEKSI + KLASIFIKASI)
# ------------------------------------------------------------
def run_detection_yolo(image_bytes):
    """
    Mengembalikan dict:
    {
        "detections": [ {...}, ... ],
        "best_detection": {...} atau None
    }
    """
    if yolo_model is None:
        raise RuntimeError(
            "YOLO model belum dimuat. Cek apakah file .pt ada dan path-nya benar."
        )

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = yolo_model(img, verbose=False)

    if not results:
        return {"detections": [], "best_detection": None}

    r = results[0]
    detections = []

    names = getattr(r, "names", {})

    # MODE DETEKSI (bbox)
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            conf = float(b.conf[0])
            if conf < MIN_CONF_DET:
                continue

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            cls_idx = int(b.cls[0])

            if isinstance(names, dict):
                label = names.get(cls_idx, str(cls_idx))
            else:
                label = names[cls_idx]

            desc = DESC_MAP.get(label, "")

            detections.append(
                {
                    "mode": "detection",
                    "box": [int(x1), int(y1), int(w), int(h)],
                    "label": label,
                    "confidence": round(conf, 4),
                    "description": desc,
                }
            )

    # MODE KLASIFIKASI (global probs)
    if hasattr(r, "probs") and r.probs is not None:
        probs = r.probs.data.tolist()
        for cls_idx, conf in enumerate(probs):
            conf = float(conf)
            if conf < MIN_CONF_CLS:
                continue

            if isinstance(names, dict):
                label = names.get(cls_idx, str(cls_idx))
            else:
                label = names[cls_idx]

            desc = DESC_MAP.get(label, "")

            detections.append(
                {
                    "mode": "classification",
                    "box": None,
                    "label": label,
                    "confidence": round(conf, 4),
                    "description": desc,
                }
            )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    best_detection = detections[0] if detections else None
    return {"detections": detections, "best_detection": best_detection}

# ------------------------------------------------------------
# GAMBAR BOUNDING BOX
# ------------------------------------------------------------
def draw_detections_on_image(image_bytes, detections):
    if not detections:
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        label = det.get("label", "")
        conf = det.get("confidence", 0.0)
        text = f"{label} {conf * 100:.1f}%"
        box = det.get("box")
        mode = det.get("mode", "detection")

        if box and mode == "detection":
            x, y, w, h = box
            x2, y2 = x + w, y + h

            draw.rectangle([x, y, x2, y2], outline=(189, 147, 249), width=4)

            text_y = max(0, y - 22)
            text_w = len(text) * 7 + 8
            draw.rectangle(
                [x, text_y, x + text_w, text_y + 20], fill=(89, 57, 161)
            )
            draw.text((x + 4, text_y + 3), text, fill=(255, 255, 255))
        else:
            # global classification tanpa bbox → hanya teks
            draw.text((10, 10), text, fill=(255, 0, 0))

    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    return buff.getvalue()

# ------------------------------------------------------------
# PROSES SATU GAMBAR
# ------------------------------------------------------------
def process_single_image_sync(image_bytes: bytes):
    """
    Jalankan deteksi, gambar bbox, simpan hasil, dan
    kembalikan dict untuk dikirim sebagai JSON response.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Simpan RAW image
    raw_path = SAVE_DIR / f"{ts}_raw.jpg"
    try:
        img_raw = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_raw.save(raw_path, format="JPEG")
        log_info(f"RAW image disimpan: {raw_path.name}")
    except Exception as e:
        log_warn(f"Gagal menyimpan RAW image: {e}")

    # Deteksi YOLO
    det_result = run_detection_yolo(image_bytes)
    detections = det_result["detections"]
    best_detection = det_result["best_detection"]

    # Gambar bbox
    processed_bytes = draw_detections_on_image(image_bytes, detections)

    # Simpan processed image
    detected_path = SAVE_DIR / f"{ts}_detected.jpg"
    try:
        img_det = Image.open(io.BytesIO(processed_bytes)).convert("RGB")
        img_det.save(detected_path, format="JPEG")
        log_info(f"Processed image disimpan: {detected_path.name}")
    except Exception as e:
        log_warn(f"Gagal menyimpan processed image: {e}")

    # Siapkan image_base64 (data URI)
    image_base64 = base64.b64encode(processed_bytes).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{image_base64}"

    # Simpan JSON hasil deteksi
    json_path = SAVE_DIR / f"{ts}_results.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "detections": detections,
                    "best_detection": best_detection,
                    "raw_image_file": raw_path.name,
                    "detected_image_file": detected_path.name,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        log_info(f"Hasil deteksi JSON disimpan: {json_path.name}")
    except Exception as e:
        log_warn(f"Gagal menyimpan JSON hasil deteksi: {e}")

    return {
        "success": True,
        "detections": detections,
        "best_detection": best_detection,
        "image_base64": image_data_uri,
        "timestamp": datetime.now().isoformat(),
        "raw_image_file": raw_path.name,
        "detected_image_file": detected_path.name,
        "results_json_file": json_path.name,
    }

# ------------------------------------------------------------
# ROUTE: SINGLE IMAGE
# ------------------------------------------------------------
@app.route("/api/kirimgambar", methods=["POST"])
def remote_detect():
    """
    Endpoint utama yang dipanggil dari app.py.
    Menerima satu gambar (upload/base64),
    memproses dan mengembalikan JSON.
    """
    try:
        image_bytes = None

        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return (
                    jsonify({"success": False, "error": "Tidak ada file yang dipilih"}),
                    400,
                )
            image_bytes = file.read()

        elif request.is_json and "image_base64" in request.json:
            base64_data = request.json["image_base64"]
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
        else:
            return (
                jsonify({"success": False, "error": "Tidak ada gambar yang diterima"}),
                400,
            )

        # Validasi gambar
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            return jsonify({"success": False, "error": "Format gambar tidak valid"}), 400

        result = process_single_image_sync(image_bytes)

        response = {
            "success": result["success"],
            "detections": result["detections"],
            "best_detection": result["best_detection"],
            "image_base64": result["image_base64"],
            "timestamp": result["timestamp"],
        }
        return jsonify(response), 200

    except RuntimeError as e:
        log_error(f"/api/remote-detect (model): {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        log_error(f"/api/remote-detect: {e}")
        return jsonify(
            {"success": False, "error": "Terjadi kesalahan internal server di server YOLO"},
            500,
        )

# ------------------------------------------------------------
# ROUTE: HEALTH CHECK
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "yolo_loaded": yolo_model is not None,
            "timestamp": datetime.now().isoformat(),
        }
    ), 200

# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    # ganti port default ke 7000 agar tidak diblok browser (6000 = unsafe port di Chrome/Brave)
    port = int(os.getenv("PORT", "7000"))
    log_info(f"Menjalankan YOLO SERVER di http://0.0.0.0:{port}")
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
