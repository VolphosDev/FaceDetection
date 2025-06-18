from flask import Flask, request, jsonify
import os, json, base64, mysql.connector, numpy as np, cv2
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import normalize
import threading
import subprocess
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from siamese_model import L2Norm, EuclideanDistance
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app)

DB = {"host":"123","user":"123","password":"123","database":"123"}
UMBRAL = 0.4  # Umbral de similitud para considerar que es la misma persona

model_lock = threading.Lock()
embedder = None
detector = MTCNN()

def load_embedder():
    global embedder
    with model_lock:
        if embedder is None:
            try:
                embedder = load_model("embedder.keras", custom_objects={
                "L2Norm": L2Norm
                })
                print("✅ Modelo embedder cargado")
            except Exception as e:
                print(f"[WARN] No se pudo cargar el modelo: {e}")

def check_and_run_data_insert():
    db, c = connect_db()
    c.execute("SELECT COUNT(*) FROM personas")
    total = c.fetchone()[0]
    c.close()
    db.close()

    if total == 0:
        print("[INFO] No hay usuarios. Ejecutando data_insert.py por primera vez...")
        try:
            subprocess.run(["python", "data_insert.py"], check=True)
        except Exception as e:
            print(f"[ERROR] Falló la ejecución automática de data_insert.py: {e}")
    else:
        print("[INFO] Datos ya existentes, no se ejecuta data_insert.py")

def connect_db():
    cfg = DB.copy()
    tmp = mysql.connector.connect(host=cfg["host"], user=cfg["user"], password=cfg["password"])
    cur = tmp.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {cfg['database']}")
    tmp.commit(); cur.close(); tmp.close()
    db = mysql.connector.connect(**cfg); c = db.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS personas (
        id INT AUTO_INCREMENT PRIMARY KEY,
        nombre VARCHAR(100), apellido_paterno VARCHAR(100),
        apellido_materno VARCHAR(100), correo VARCHAR(100),
        requisitoriado BOOLEAN)""")
    c.execute("""CREATE TABLE IF NOT EXISTS kp (
        id_kp INT AUTO_INCREMENT PRIMARY KEY,
        foto LONGBLOB, KP LONGTEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS personas_keypoints (
        id_persona INT, id_kp INT,
        FOREIGN KEY (id_persona) REFERENCES personas(id),
        FOREIGN KEY (id_kp) REFERENCES kp(id_kp))""")
    db.commit(); return db, c

def promedio_vectores(lista_vectores):
    vectores_np = np.array([json.loads(kp_json) for kp_json in lista_vectores])
    vectores_np = normalize(vectores_np)  # Normaliza cada uno
    prom = np.mean(vectores_np, axis=0)
    return normalize([prom])[0] 

def build_knn_model(personas_dict):
    X, y = [], []
    id_to_info = {}

    for pid, datos in personas_dict.items():
        try:
            vector_prom = promedio_vectores(datos["vectores"])
            X.append(vector_prom)
            y.append(pid)
            id_to_info[pid] = datos
        except Exception as e:
            continue

    if not X:
        return None, {}, []

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(X, y)
    return knn, id_to_info, X

def extract_face_vector(img):
    detections = detector.detect_faces(img)
    if not detections:
        raise ValueError("No se detectó rostro")

    x, y, w, h = detections[0]["box"]
    face_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)  # ← mantenemos color
    face_rgb = cv2.resize(face_rgb, (128, 128))
    face_rgb = face_rgb.astype("float32") / 255.0
    face_rgb = np.expand_dims(face_rgb, axis=0)

    load_embedder()
    emb = embedder.predict(face_rgb)
    return normalize(emb)[0]

@app.route("/reconocer", methods=["POST"])
def reconocer():
    if 'imagen' not in request.files:
        return jsonify(error="Falta imagen"), 400

    try:
        img = cv2.imdecode(np.frombuffer(request.files['imagen'].read(), np.uint8), cv2.IMREAD_COLOR)
        vector_entrada = extract_face_vector(img)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"Error al procesar la imagen: {str(e)}"), 500

    try:
        db, c = connect_db()
        c.execute("""
            SELECT pk.id_persona, k.KP, p.nombre, p.apellido_paterno, p.requisitoriado
            FROM personas_keypoints pk
            JOIN kp k ON pk.id_kp = k.id_kp
            JOIN personas p ON pk.id_persona = p.id
        """)
        registros = c.fetchall()
        c.close()
        db.close()
    except Exception as e:
        return jsonify(error=f"Error en base de datos: {str(e)}"), 500

    # Agrupar vectores por persona
    personas_dict = {}
    for pid, kp_json, nombre, apellido, requisitoriado in registros:
        if kp_json is None:
            continue
        if pid not in personas_dict:
            personas_dict[pid] = {
                "vectores": [],
                "nombre": nombre,
                "apellido": apellido,
                "requisitoriado": bool(requisitoriado)
            }
        personas_dict[pid]["vectores"].append(kp_json)

    # Entrenar modelo KNN
    knn, id_to_info, X = build_knn_model(personas_dict)
    if knn is None or not X:
        return jsonify(error="No hay suficientes vectores para comparar"), 500

    # Predecir ID más cercano
    pred_id = knn.predict([vector_entrada])[0]
    dist = knn.kneighbors([vector_entrada], return_distance=True)[0][0][0]

    print(f"[DEBUG] Predicción KNN → ID: {pred_id}, distancia: {dist}")

    if dist > UMBRAL or pred_id not in id_to_info:
        return jsonify(message="unknown", distancia=round(float(dist), 3)), 404

    persona = id_to_info[int(pred_id)]  # Aseguramos conversión aquí también
    return jsonify(
        id=int(pred_id),  # 👈 conversión para asegurar tipo nativo
        nombre=persona["nombre"],
        apellido_paterno=persona["apellido"],
        requisitoriado=bool(persona["requisitoriado"]),
        distancia=round(float(dist), 3)
    ), 200

@app.route("/usuarios", methods=["GET"])
def list_users():
    page, size = map(int, (request.args.get("page",1), request.args.get("size",10)))
    db, c = connect_db()
    c.execute("SELECT id,nombre,apellido_paterno,requisitoriado FROM personas LIMIT %s OFFSET %s", (size, (page-1)*size))
    data = [{"id":r[0], "nombre":r[1], "apellido_paterno":r[2], "requisitoriado":bool(r[3])} for r in c.fetchall()]
    c.execute("SELECT COUNT(*) FROM personas"); total = c.fetchone()[0]
    c.close(); db.close()
    return jsonify(page=page, size=size, total=total, data=data), 200

@app.route("/usuario/<int:id>", methods=["GET","DELETE"])
def user_detail(id):
    db, c = connect_db()
    if request.method=="GET":
        c.execute("SELECT id,nombre,apellido_paterno,apellido_materno,correo,requisitoriado FROM personas WHERE id=%s",(id,))
        r = c.fetchone()
        if not r: c.close(); db.close(); return jsonify(error="No existe"),404
        c.execute("SELECT k.id_kp,k.foto FROM kp k JOIN personas_keypoints pk ON k.id_kp=pk.id_kp WHERE pk.id_persona=%s",(id,))
        fotos=[{"id_kp":kid,"foto":foto} for kid, foto in c.fetchall()]
        c.close(); db.close()
        return jsonify(id=r[0], nombre=r[1], apellido_paterno=r[2], apellido_materno=r[3], correo=r[4], requisitoriado=bool(r[5]), fotos=fotos),200
    c.execute("DELETE p,k FROM personas p LEFT JOIN personas_keypoints pk ON p.id=pk.id_persona LEFT JOIN kp k ON pk.id_kp=k.id_kp WHERE p.id=%s",(id,))
    db.commit(); db.close()
    return jsonify(message="Usuario eliminado"),200

@app.route("/usuario/<int:id>/foto/<int:kid>", methods=["DELETE"])
def delete_photo(id, kid):
    db,c = connect_db()
    c.execute("DELETE FROM personas_keypoints WHERE id_persona=%s AND id_kp=%s",(id,kid))
    c.execute("DELETE FROM kp WHERE id_kp=%s",(kid,))
    db.commit(); c.close(); db.close()
    return jsonify(message="Foto eliminada"),200

@app.route("/registrar", methods=["POST"])
def registrar():
    data = request.form
    for key in ("nombre", "apellido_paterno", "apellido_materno", "correo", "requisitoriado"):
        if key not in data:
            return jsonify(error=f"Falta campo: {key}"), 400

    imagenes = request.files.getlist("imagen")
    if not imagenes:
        return jsonify(error="Se requieren imágenes"), 400

    db, c = connect_db()
    c.execute("SELECT id FROM personas WHERE correo=%s", (data["correo"],))
    row = c.fetchone()
    pid = row[0] if row else None

    if not pid:
        c.execute("""INSERT INTO personas(nombre, apellido_paterno, apellido_materno, correo, requisitoriado)
                     VALUES (%s, %s, %s, %s, %s)""",
                  (data["nombre"], data["apellido_paterno"], data["apellido_materno"],
                   data["correo"], bool(int(data["requisitoriado"]))))
        pid = c.lastrowid

    guardadas = 0

    for f in imagenes:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        detections = detector.detect_faces(img)
        if not detections:
            continue

        x, y, w, h = detections[0]["box"]
        x, y = max(x, 0), max(y, 0)
        cara_recortada = img[y:y+h, x:x+w]
        cara_recortada = cv2.resize(cara_recortada, (128, 128))

        try:
            emb = extract_face_vector(cara_recortada)
            guardadas += 1
        except Exception as e:
            print(f"[WARN] No se extrajo vector: {e}")
            emb = None

        foto_bytes = cv2.imencode(".jpg", cara_recortada)[1].tobytes()
        c.execute("INSERT INTO kp(foto, KP) VALUES (%s, %s)", (
            foto_bytes,
            json.dumps(emb.tolist()) if emb is not None else None
        ))
        id_kp = c.lastrowid  # 🔑 nuevo keypoint

        # Relación persona ↔ keypoint
        c.execute("INSERT INTO personas_keypoints(id_persona, id_kp) VALUES (%s, %s)", (pid, id_kp))

    db.commit()
    db.close()

    if guardadas == 0:
        return jsonify(error="No se pudo registrar ninguna imagen"), 400

    return jsonify(message=f"Registrado con {guardadas} imágenes", id=pid), 200

if __name__=="__main__":
    check_and_run_data_insert()
    app.run(host="0.0.0.0", port=5000, debug=True)