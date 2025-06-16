import os
import cv2
import random
import uuid
import numpy as np
import json
from mtcnn import MTCNN  # ← NUEVO
from app import connect_db, extract_face_vector

PHOTOS_DIR = "photos"
detector = MTCNN()  # ← NUEVO

def cargar_imagen_rgb(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen: {ruta}")
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        print(f"[ERROR] Formato de imagen no compatible: {ruta}")
        return None

    return img

def obtener_datos_de_folder(folder_name):
    try:
        nombre, apellido = folder_name.strip().split(" ", 1)
        correo = f"{nombre.lower()}{apellido.lower().replace(' ', '')}@gmail.com"
        return nombre, apellido, correo
    except ValueError:
        print(f"[ERROR] Nombre inválido en carpeta: '{folder_name}'")
        return None, None, None

def procesar_y_guardar(img_bgr, nombre, apellido_p, correo, apellido_m=""):
    db, c = connect_db()

    # Insertar persona si no existe
    c.execute("SELECT id FROM personas WHERE correo=%s", (correo,))
    row = c.fetchone()
    pid = row[0] if row else None

    if not pid:
        c.execute("""INSERT INTO personas(nombre, apellido_paterno, apellido_materno, correo, requisitoriado)
                     VALUES (%s, %s, %s, %s, %s)""",
                  (nombre, apellido_p, apellido_m, correo, random.choice([0, 1])))
        pid = c.lastrowid

    # Detección de rostro con MTCNN
    detections = detector.detect_faces(img_bgr)
    if not detections:
        print(f"[WARN] No se encontró rostro en imagen de {nombre}")
        db.commit(); db.close()
        return

    x, y, w, h = detections[0]["box"]
    x, y = max(x, 0), max(y, 0)
    cara = img_bgr[y:y+h, x:x+w]
    cara = cv2.resize(cara, (128, 128))

    try:
        emb = extract_face_vector(img_bgr)
    except Exception as e:
        print(f"[WARN] No se pudo extraer vector: {e}")
        emb = None

    foto_bytes = cv2.imencode(".jpg", cara)[1].tobytes()

    c.execute("INSERT INTO kp(foto, KP) VALUES (%s, %s)", (
        foto_bytes,
        json.dumps(emb.tolist()) if emb is not None else None
    ))
    kid = c.lastrowid
    c.execute("INSERT INTO personas_keypoints(id_persona, id_kp) VALUES (%s, %s)", (pid, kid))

    db.commit()
    db.close()
    print(f"[✔️] Registrado: {nombre} {apellido_p}")

def procesar_directorio():
    for folder in os.listdir(PHOTOS_DIR):
        ruta_folder = os.path.join(PHOTOS_DIR, folder)
        if not os.path.isdir(ruta_folder):
            continue

        nombre, apellido, correo = obtener_datos_de_folder(folder)
        if not nombre:
            continue

        for archivo in os.listdir(ruta_folder):
            if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            ruta_img = os.path.join(ruta_folder, archivo)
            img = cargar_imagen_rgb(ruta_img)
            if img is None:
                continue

            procesar_y_guardar(img, nombre, apellido, correo)

if __name__ == "__main__":
    try:
        procesar_directorio()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción del usuario. Finalizando...")