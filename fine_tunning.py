from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
import numpy as np
import cv2
import json
import mysql.connector
from siamese_model import build_siamese, contrastive_loss, L2Norm, EuclideanDistance
from data_utils import make_pairs

# ===== Función para cargar imágenes de la base de datos =====
def load_images_from_db():
    db = mysql.connector.connect(
        host="123",
        user="123",
        password="123",
        database="123"
    )
    c = db.cursor()
    c.execute("""
        SELECT pk.id_persona, k.foto
        FROM personas_keypoints pk 
        JOIN kp k ON pk.id_kp = k.id_kp
    """)
    rows = c.fetchall()
    db.close()

    images, labels = [], []
    for pid, foto_bytes in rows:
        img = cv2.imdecode(np.frombuffer(foto_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (128, 128):
            continue
        img = img.astype("float32") / 255.0
        images.append(img)
        labels.append(pid)

    return np.array(images), np.array(labels)

# ===== Función para insertar embeddings en la base de datos =====
def insertar_embeddings_actualizados(model, db_images, labels):
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin",
        database="detection_face_db"
    )
    c = db.cursor()

    print("🧠 Reemplazando embeddings en la base de datos...")

    for i in range(len(db_images)):
        img = db_images[i]
        pid = int(labels[i])

        try:
            # Obtener todos los id_kp de esta persona
            c.execute("""
                SELECT pk.id_kp FROM personas_keypoints pk
                WHERE pk.id_persona = %s
            """, (pid,))
            id_kps = c.fetchall()

            if not id_kps:
                print(f"⚠️ No hay keypoints asociados a persona {pid}, se omite.")
                continue

            # Generar embedding
            emb = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            emb_json = json.dumps(emb.tolist())

            # Reemplazar embedding en cada kp asociado
            for (id_kp,) in id_kps:
                c.execute("UPDATE kp SET KP = %s WHERE id_kp = %s", (emb_json, id_kp))
                print(f"✅ Embedding reemplazado en id_kp {id_kp} para persona {pid}")

            db.commit()

        except Exception as e:
            print(f"❌ Error al reemplazar embedding para persona {pid}: {e}")
            db.rollback()

    c.close()
    db.close()
    print("🏁 Todos los embeddings fueron reemplazados correctamente.")

# ===== Código principal =====
if __name__ == "__main__":
    X, y = load_images_from_db()
    if len(X) < 2:
        print("❌ No hay suficientes imágenes válidas para fine-tuning.")
        exit()

    X, y = shuffle(X, y, random_state=42)
    pairs, labels = make_pairs(X, y)
    print(f"🔄 Se generaron {len(pairs)} pares.")

    embedder_net = load_model("embedder.keras", custom_objects={
        "L2Norm": L2Norm,
        "EuclideanDistance": EuclideanDistance
    })
    embedder_net.trainable = True

    siamese_net, _ = build_siamese(input_shape=(128, 128, 3), base_model=embedder_net)
    siamese_net.compile(optimizer="adam", loss=contrastive_loss(margin=1.0))

    siamese_net.fit([pairs[:, 0], pairs[:, 1]], labels,
                    validation_split=0.1,
                    epochs=10,
                    batch_size=16,
                    verbose=2)

    embedder_net.save("embedder.keras")
    print("✅ Fine-tuning completado y embedder.keras actualizado")

    insertar_embeddings_actualizados(embedder_net, X, y)