import mysql.connector
import numpy as np
import cv2

def load_faces_from_db():
    db = mysql.connector.connect(
        host="123",
        user="123",
        password="123",
        database="123"
    )
    c = db.cursor()
    c.execute("""
        SELECT k.foto, pk.id_persona
        FROM kp k
        JOIN personas_keypoints pk ON k.id_kp = pk.id_kp
    """)
    data = c.fetchall()
    db.close()

    X, y = [], []
    for foto_blob, id_persona in data:
        arr = np.frombuffer(foto_blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        X.append(img)
        y.append(id_persona)
    
    return np.array(X), np.array(y)

# 🔽 Agrega esto al final del archivo
def make_pairs(images, labels):
    pairs = []
    pair_labels = []
    label_to_indices = {}

    for idx, label in enumerate(labels):
        label = int(label)
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    for idx, img in enumerate(images):
        current_label = int(labels[idx])
        pos_candidates = label_to_indices[current_label]
        
        if len(pos_candidates) < 2:
            print(f"[⚠️] Etiqueta {current_label} tiene <2 imágenes. Se omite.")
            continue

        # Positiva
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = np.random.choice(pos_candidates)
        pos_img = images[pos_idx]
        pairs.append([img, pos_img])
        pair_labels.append(1)

        # Negativa
        neg_label = current_label
        while neg_label == current_label:
            neg_label = np.random.choice(list(label_to_indices.keys()))
        neg_idx = np.random.choice(label_to_indices[neg_label])
        neg_img = images[neg_idx]
        pairs.append([img, neg_img])
        pair_labels.append(0)

    return np.array(pairs), np.array(pair_labels)