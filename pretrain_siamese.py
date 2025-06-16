import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from siamese_model import build_siamese, contrastive_loss
from tqdm import tqdm
import tensorflow as tf

print("[⚙️] Dispositivos disponibles:")
print(tf.config.list_physical_devices())

# Opcional: limitar uso de memoria en GPU (evita que ocupe toda)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[✅] GPU habilitada y con crecimiento de memoria activado.")
    except RuntimeError as e:
        print("[❌] Error al configurar GPU:", e)
else:
    print("[🧱] No se detectó GPU, usando CPU.")

DATASET_PATH = "lfw-deepfunneled"
IMG_SIZE = (128, 128)

def load_dataset():
    images = []
    labels = []
    label_map = {}
    current_label = 0

    print("[📥] Cargando imágenes del dataset externo (solo personas con ≥2 fotos)...")
    for person_name in tqdm(os.listdir(DATASET_PATH)):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue

        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(image_files) < 2:
            continue  # ⚠️ Ignorar personas con solo una imagen

        for filename in image_files:
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img_to_array(img) / 255.0)

            if person_name not in label_map:
                label_map[person_name] = current_label
                current_label += 1
            labels.append(label_map[person_name])

    return np.array(images), np.array(labels)

def make_pairs(images, labels):
    print("[🔗] Generando pares positivos y negativos...")
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

        # 🔁 Par positivo (misma persona)
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = np.random.choice(label_to_indices[current_label])
        pos_img = images[pos_idx]
        pairs.append([img, pos_img])
        pair_labels.append(1)

        # ❌ Par negativo (persona diferente)
        neg_label = current_label
        while neg_label == current_label:
            neg_label = np.random.choice(list(label_to_indices.keys()))
        neg_idx = np.random.choice(label_to_indices[neg_label])
        neg_img = images[neg_idx]
        pairs.append([img, neg_img])
        pair_labels.append(0)

    return np.array(pairs), np.array(pair_labels)

def main():
    # --- Cargar y preparar datos ---
    X, y = load_dataset()
    print(f"[✅] Total imágenes cargadas: {len(X)} (personas válidas: {len(set(y))})")

    if len(X) < 10:
        print("[⚠️] Muy pocos datos. Revisa si hay suficientes personas con ≥2 fotos.")
        return

    pairs, labels = make_pairs(X, y)
    pairs, labels = shuffle(pairs, labels, random_state=42)

    # --- Separar en entrenamiento y validación ---
    split = int(0.8 * len(pairs))
    tr_p, va_p = pairs[:split], pairs[split:]
    tr_l, va_l = labels[:split], labels[split:]

    # --- Construir modelo siamés robusto ---
    print("[🧠] Entrenando modelo siamés...")
    siamese, embedder = build_siamese(input_shape=X.shape[1:], embed_dim=64, use_l2norm=True)
    siamese.compile(optimizer=Adam(learning_rate=1e-4), loss=contrastive_loss(margin=1.0))

    siamese.fit([tr_p[:, 0], tr_p[:, 1]], tr_l,
                validation_data=([va_p[:, 0], va_p[:, 1]], va_l),
                epochs=20, batch_size=32, verbose=1)

    # --- Guardar modelo entrenado ---
    embedder.save("embedder.keras")
    print("✅ Modelo preentrenado guardado como 'embedder.keras'")

if __name__ == "__main__":
    main()