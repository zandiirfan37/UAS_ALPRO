from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import base64
import io
import random
from PIL import Image
from datetime import datetime
import os

print("DEBUG: app.py dieksekusi, __name__ =", __name__)

# ============================================
# IMPORT MYSQL CONNECTOR (dengan handling)
# ============================================
try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error
    print("DEBUG: mysql-connector-python terdeteksi")
except ImportError:
    mysql_connector = None
    Error = Exception
    print("WARNING: mysql-connector-python belum terinstall.")
    print("         Jalankan: pip install mysql-connector-python untuk mengaktifkan DB")

# ============================================
# INISIALISASI FLASK
# ============================================
# static_folder="." + static_url_path="" => bisa serve index.html di root
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ============================================
# KONFIGURASI DATABASE (XAMPP)
# ============================================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),          # default XAMPP
    "database": os.getenv("DB_NAME", "skindetect_db"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "connection_timeout": 10,
}

# ============================================
# DATA PENYAKIT KULIT (SIMULASI)
# ============================================
SKIN_DISEASES = [
    {"label": "Melanoma", "description": "Jenis kanker kulit yang berkembang dari sel melanosit"},
    {"label": "Psoriasis", "description": "Penyakit autoimun yang menyebabkan penumpukan sel kulit"},
    {"label": "Eczema", "description": "Kondisi kulit yang menyebabkan kulit menjadi merah dan gatal"},
    {"label": "Actinic Keratosis", "description": "Bercak kasar dan bersisik pada kulit akibat paparan sinar matahari"},
    {"label": "Basal Cell Carcinoma", "description": "Jenis kanker kulit yang paling umum"},
    {"label": "Benign Keratosis", "description": "Pertumbuhan kulit non-kanker yang umum"},
    {"label": "Dermatofibroma", "description": "Benjolan kulit jinak yang keras"},
    {"label": "Nevus", "description": "Tahi lalat atau tanda lahir pada kulit"},
]

# ============================================
# FUNGSI DATABASE
# ============================================
def get_db_connection():
    """Membuat koneksi ke database MySQL."""
    if mysql_connector is None:
        print("DEBUG: mysql-connector-python tidak tersedia, koneksi DB dinonaktifkan.")
        return None

    try:
        conn = mysql_connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"],
            connection_timeout=DB_CONFIG["connection_timeout"],
        )
        conn.autocommit = True
        if conn.is_connected():
            print("DEBUG: Berhasil terhubung ke MySQL")
        return conn
    except Exception as e:
        print(f"DEBUG: Error koneksi database: {e}")
        print("       Cek: service MySQL XAMPP, nama DB, user, dan password.")
        return None


def test_database_connection():
    """Testing koneksi saat startup (opsional)."""
    print("\n" + "=" * 60)
    print("TESTING KONEKSI DATABASE")
    print("=" * 60)

    conn = get_db_connection()
    if not conn:
        print("INFO: Koneksi DB gagal, API tetap bisa jalan tapi endpoint DB akan error.")
        print("=" * 60 + "\n")
        return

    try:
        cur = conn.cursor()
        cur.execute("SHOW TABLES;")
        tables = cur.fetchall()
        print(f"DEBUG: Jumlah tabel: {len(tables)}")
        for t in tables:
            print(f"   - {t[0]}")
        cur.close()
        conn.close()
        print("DEBUG: Database siap digunakan.")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"DEBUG: Error saat cek tabel: {e}")
        print("=" * 60 + "\n")

# ============================================
# SIMULASI YOLO
# ============================================
def load_yolo_model():
    print("DEBUG: Model YOLO simulasi berhasil dimuat")
    return {"status": "simulated", "version": "YOLOv8n"}


def run_detection(image_bytes):
    """
    Simulasi deteksi penyakit kulit:
    return list of dict: {box: [x,y,w,h], label, confidence, description}
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size

        # 70% kemungkinan ada deteksi
        if random.random() < 0.7:
            num_detections = random.randint(1, 3)
            results = []

            used_diseases = random.sample(
                SKIN_DISEASES,
                min(num_detections, len(SKIN_DISEASES)),
            )

            for disease in used_diseases:
                box_w = random.randint(int(img_width * 0.15), int(img_width * 0.4))
                box_h = random.randint(int(img_height * 0.15), int(img_height * 0.4))
                box_x = random.randint(0, img_width - box_w)
                box_y = random.randint(0, img_height - box_h)
                confidence = round(random.uniform(0.65, 0.98), 2)

                results.append(
                    {
                        "box": [box_x, box_y, box_w, box_h],
                        "label": disease["label"],
                        "confidence": confidence,
                        "description": disease["description"],
                    }
                )
            return results
        else:
            return []
    except Exception as e:
        print(f"DEBUG: Error dalam deteksi: {e}")
        return []


# load sekali di awal
model = load_yolo_model()

# ============================================
# ROUTE FRONTEND
# ============================================
@app.route("/")
def index():
    # asumsikan index.html ada di folder yang sama dengan app.py
    print("DEBUG: / dipanggil -> kirim index.html")
    return send_from_directory(".", "index.html")

# ============================================
# ROUTE API
# ============================================
@app.route("/api/detect", methods=["POST"])
def detect_disease():
    """
    Terima gambar, simpan ke DB, jalankan deteksi, kembalikan hasil.
    """
    try:
        image_bytes = None

        # Upload file (form-data)
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"success": False, "error": "Tidak ada file yang dipilih"}), 400
            image_bytes = file.read()

        # Data base64 (kamera)
        elif request.is_json and "image_base64" in request.json:
            base64_data = request.json["image_base64"]
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            image_bytes = base64.b64decode(base64_data)

        else:
            return jsonify({"success": False, "error": "Tidak ada gambar yang diterima"}), 400

        # Validasi gambar
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            return jsonify({"success": False, "error": "Format gambar tidak valid"}), 400

        # Koneksi DB
        conn = get_db_connection()
        if not conn:
            # Server tetap jalan, tapi beri info ke client
            return jsonify({"success": False, "error": "Gagal terhubung ke database"}), 500

        cur = conn.cursor()

        # Simpan gambar (pre-processing)
        insert_query = "INSERT INTO detection_history (image_data) VALUES (%s)"
        cur.execute(insert_query, (image_bytes,))
        conn.commit()
        image_id = cur.lastrowid

        # Jalankan deteksi
        detection_results = run_detection(image_bytes)

        # Simpan hasil deteksi
        results_json = json.dumps(detection_results)
        update_query = "UPDATE detection_history SET detection_results = %s WHERE id = %s"
        cur.execute(update_query, (results_json, image_id))
        conn.commit()

        # Update statistik
        if detection_results:
            cur.execute(
                """
                UPDATE detection_stats 
                SET total_detections = total_detections + 1,
                    total_diseases_found = total_diseases_found + %s
                """,
                (len(detection_results),),
            )
            conn.commit()

        cur.close()
        conn.close()

        # Kirim balik gambar base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "detections": detection_results,
                "image_id": image_id,
                "image_base64": f"data:image/jpeg;base64,{image_base64}",
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    except Exception as e:
        print(f"DEBUG: Error /api/detect: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "error": "Gagal terhubung ke database"}), 500

        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, detection_results, created_at "
            "FROM detection_history ORDER BY created_at DESC LIMIT 10"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        for r in rows:
            if r.get("detection_results"):
                try:
                    r["detection_results"] = json.loads(r["detection_results"])
                except Exception:
                    r["detection_results"] = None

        return jsonify({"success": True, "history": rows}), 200
    except Exception as e:
        print(f"DEBUG: Error /api/history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "error": "Gagal terhubung ke database"}), 500

        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM detection_stats LIMIT 1")
        stats = cur.fetchone()
        cur.close()
        conn.close()

        return jsonify({"success": True, "stats": stats}), 200
    except Exception as e:
        print(f"DEBUG: Error /api/stats: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }
    ), 200

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    test_database_connection()
    print("DEBUG: Menjalankan Flask API di http://127.0.0.1:5002")
    app.run(debug=True, host="127.0.0.1", port=5002)
