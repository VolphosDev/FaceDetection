from flask import Flask, request, jsonify
import os, json, base64, mysql.connector, numpy as np, cv2
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import normalize
import threading
import subprocess
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from siamese_model import L2Norm
from flask_cors import cross_origin


app = Flask(__name__)
CORS(app)

DB = {"host":"localhost","user":"root","password":"admin","database":"face_detection_db"}
UMBRAL = 0.4 

model_lock = threading.Lock()
embedder = None
detector = MTCNN()

def load_embedder():
    global embedder
    with model_lock:
        if embedder is None:
            try:
                embedder = load_model("embedder.keras", custom_objects={"L2Norm": L2Norm})
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

def extract_face_vector(img):
    detections = detector.detect_faces(img)
    if not detections:
        raise ValueError("No se detectó rostro")

    x, y, w, h = detections[0]["box"]
    face_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (128,128))
    face_rgb = face_rgb.astype("float32") / 255.0
    face_rgb = np.expand_dims(face_rgb, axis=0)

    load_embedder()
    emb = embedder.predict(face_rgb)
    return normalize(emb)[0]

@app.route("/reconocer", methods=["POST", "OPTIONS"])
@cross_origin()
def reconocer():
    if request.method == "OPTIONS":
        return '', 200

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

    mejor_distancia = float("inf")
    mejor_persona = None

    for pid, kp_json, nombre, apellido, requisitoriado in registros:
        if kp_json is None:
            continue
        emb = np.array(json.loads(kp_json))
        dist = cosine(vector_entrada, emb)
        if dist < mejor_distancia:
            mejor_distancia = dist
            mejor_persona = {
                "id": pid,
                "nombre": nombre,
                "apellido_paterno": apellido,
                "requisitoriado": bool(requisitoriado)
            }

    porcentaje = round((1 - mejor_distancia) * 100, 2)

    if porcentaje < 60.0 or mejor_persona is None:
        return jsonify(message="unknown", distancia=round(float(mejor_distancia), 3), porcentaje=porcentaje), 404

    return jsonify(
        id=mejor_persona["id"],
        nombre=mejor_persona["nombre"],
        apellido_paterno=mejor_persona["apellido_paterno"],
        requisitoriado=mejor_persona["requisitoriado"],
        distancia=round(float(mejor_distancia), 3),
        porcentaje=porcentaje
    ), 200

@app.route("/usuarios", methods=["GET", "OPTIONS"])
def list_users():
    page, size = map(int, (request.args.get("page",1), request.args.get("size",10)))
    db, c = connect_db()
    c.execute("SELECT id,nombre,apellido_paterno,requisitoriado FROM personas LIMIT %s OFFSET %s", (size, (page-1)*size))
    data = [{"id":r[0], "nombre":r[1], "apellido_paterno":r[2], "requisitoriado":bool(r[3])} for r in c.fetchall()]
    c.execute("SELECT COUNT(*) FROM personas"); total = c.fetchone()[0]
    c.close(); db.close()
    return jsonify(page=page, size=size, total=total, data=data), 200

@app.route("/usuario/<int:id>", methods=["GET", "DELETE", "OPTIONS"])
def user_detail(id):
    db, c = connect_db()
    if request.method == "GET":
        c.execute("SELECT id,nombre,apellido_paterno,apellido_materno,correo,requisitoriado FROM personas WHERE id=%s", (id,))
        r = c.fetchone()
        if not r:
            c.close(); db.close()
            return jsonify(error="No existe"), 404

        c.execute("SELECT k.id_kp, k.foto FROM kp k JOIN personas_keypoints pk ON k.id_kp = pk.id_kp WHERE pk.id_persona = %s", (id,))
        fotos = []
        for kid, foto in c.fetchall():
            foto_base64 = base64.b64encode(foto).decode("utf-8") if foto else None
            fotos.append({"id_kp": kid, "foto": foto_base64})

        c.close(); db.close()

        return jsonify(
            id=r[0],
            nombre=r[1],
            apellido_paterno=r[2],
            apellido_materno=r[3],
            correo=r[4],
            requisitoriado=bool(r[5]),
            fotos=fotos
        ), 200

    c.execute("DELETE p,k FROM personas p LEFT JOIN personas_keypoints pk ON p.id=pk.id_persona LEFT JOIN kp k ON pk.id_kp=k.id_kp WHERE p.id=%s", (id,))
    db.commit(); db.close()
    return jsonify(message="Usuario eliminado"), 200


@app.route("/usuarioedit/<int:id>", methods=["PUT", "OPTIONS"])
def editar_usuario(id):
    data = request.form
    db, c = connect_db()

    # Verificar si el usuario existe
    c.execute("SELECT id FROM personas WHERE id = %s", (id,))
    if not c.fetchone():
        c.close(); db.close()
        return jsonify(error="Usuario no encontrado"), 404

    # Actualizar solo los campos que vengan en el form
    campos_validos = ("nombre", "apellido_paterno", "apellido_materno", "correo", "requisitoriado")
    actualizaciones = []
    valores = []

    for campo in campos_validos:
        if campo in data:
            actualizaciones.append(f"{campo} = %s")
            if campo == "requisitoriado":
                valores.append(bool(int(data[campo])))
            else:
                valores.append(data[campo])

    if actualizaciones:
        query = f"UPDATE personas SET {', '.join(actualizaciones)} WHERE id = %s"
        valores.append(id)
        c.execute(query, valores)

    # Si se enviaron imágenes, procesarlas y añadirlas
    imagenes = request.files.getlist("imagen")
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
        id_kp = c.lastrowid
        c.execute("INSERT INTO personas_keypoints(id_persona, id_kp) VALUES (%s, %s)", (id, id_kp))

    db.commit()
    c.close(); db.close()

    return jsonify(message="Datos actualizados correctamente", imagenes_nuevas=guardadas), 200


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
        id_kp = c.lastrowid 

        c.execute("INSERT INTO personas_keypoints(id_persona, id_kp) VALUES (%s, %s)", (pid, id_kp))

    db.commit()
    db.close()

    if guardadas == 0:
        return jsonify(error="No se pudo registrar ninguna imagen"), 400

    return jsonify(message=f"Registrado con {guardadas} imágenes", id=pid), 200

if __name__=="__main__":
    check_and_run_data_insert()
    app.run(host="0.0.0.0", port=5000, debug=True)