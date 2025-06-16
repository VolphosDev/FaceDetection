import numpy as np
from siamese_model import build_siamese, contrastive_loss
from data_utils import make_pairs, load_faces_from_db
from sklearn.utils import shuffle
import cv2
import json
import mysql.connector

def conectar_db():
    return mysql.connector.connect(
        host="123",
        user="123",
        password="123",
        database="123"
    )

def insertar_embeddings(embedder, X, y):
    db = conectar_db()
    c = db.cursor()

    for img, persona_id in zip(X, y):
        emb = embedder.predict(np.expand_dims(img, axis=0))[0]
        _, img_encoded = cv2.imencode(".jpg", (img * 255).astype(np.uint8))
        foto_bytes = img_encoded.tobytes()

        try:
            c.execute("INSERT INTO kp(foto, KP) VALUES (%s, %s)",
                      (foto_bytes, json.dumps(emb.tolist())))
            id_kp = c.lastrowid

            c.execute("INSERT INTO personas_keypoints(id_persona, id_kp) VALUES (%s, %s)",
                      (int(persona_id), id_kp))
            print(f"[🆕] Nuevo embedding insertado para persona {persona_id}")

            db.commit()
        except mysql.connector.Error as e:
            print(f"[❌] Error con persona {persona_id}: {e}")

    c.close()
    db.close()
    print("✅ Todos los embeddings insertados correctamente")

# --- Cargar datos desde base de datos ---
print("[🚀] Cargando datos desde la base de datos...")
X, y = load_faces_from_db()
print(f"[✅] Imágenes cargadas: {len(X)}, etiquetas: {len(y)}")

if len(X) == 0:
    raise ValueError("❌ No se cargaron imágenes.")

X = X.astype("float32") / 255.0

print("[🔗] Generando pares de entrenamiento...")
pairs, labels = make_pairs(X, y)
print(f"[✅] Total de pares: {len(pairs)}")

if len(pairs) == 0:
    raise ValueError("❌ No se generaron pares.")

pairs, labels = shuffle(pairs, labels, random_state=42)

# --- Dividir ---
split = int(0.8 * len(pairs))
tr_p, va_p = pairs[:split], pairs[split:]
tr_l, va_l = labels[:split], labels[split:]

print(f"[📊] Entrenamiento: {len(tr_p)} pares | Validación: {len(va_p)} pares")

# --- Construir y compilar modelo ---
print("[🧠] Construyendo modelo siamese...")
siamese, embedder = build_siamese(input_shape=X.shape[1:], embed_dim=64)
siamese.compile(optimizer="adam", loss=contrastive_loss(margin=1.0))
print("[✅] Modelo compilado.")

# --- Entrenar ---
print("[🏋️‍♀️] Entrenando modelo...")
siamese.fit([tr_p[:, 0], tr_p[:, 1]], tr_l,
            validation_data=([va_p[:, 0], va_p[:, 1]], va_l),
            epochs=50, batch_size=16, verbose=1)

# --- Guardar modelo embedder ---
embedder.save("embedder.keras")
print("✅ Siamese y embedder listos")

# --- Insertar embeddings ---
print("[💾] Insertando embeddings en la base de datos...")
insertar_embeddings(embedder, X, y)
