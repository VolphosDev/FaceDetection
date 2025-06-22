import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from siamese_model import build_siamese, contrastive_loss
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

DATASET_PATH = "celeba_por_persona"
IMG_SIZE = (128,128)

def load_dataset():
    images = []
    labels = []
    label_map = {}
    current_label = 0

    max_personas = 4000
    personas_procesadas = 0

    print("[📥] Cargando imágenes del dataset externo (solo personas con ≥2 fotos)...")
    for person_name in tqdm(os.listdir(DATASET_PATH)):
        if personas_procesadas >= max_personas:
            break

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

        personas_procesadas += 1

    return np.array(images), np.array(labels)

def data_generator(images, labels, batch_size):
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    while True:
        pair_images_a = []
        pair_images_b = []
        pair_labels = []

        for _ in range(batch_size):
            idx = np.random.randint(0, len(images))
            img = images[idx]
            label = labels[idx]

            # Par positivo
            pos_idx = idx
            while pos_idx == idx:
                pos_idx = np.random.choice(label_to_indices[label])
            pos_img = images[pos_idx]
            pair_images_a.append(img)
            pair_images_b.append(pos_img)
            pair_labels.append(1)

            # Par negativo
            neg_label = label
            while neg_label == label:
                neg_label = np.random.choice(list(label_to_indices.keys()))
            neg_idx = np.random.choice(label_to_indices[neg_label])
            neg_img = images[neg_idx]
            pair_images_a.append(img)
            pair_images_b.append(neg_img)
            pair_labels.append(0)

        yield [np.array(pair_images_a), np.array(pair_images_b)], np.array(pair_labels)

def main():
    # --- Cargar y preparar datos ---
    X, y = load_dataset()
    print(f"[✅] Total imágenes cargadas: {len(X)} (personas válidas: {len(set(y))})")

    if len(X) < 10:
        print("[⚠️] Muy pocos datos. Revisa si hay suficientes personas con ≥2 fotos.")
        return

    # --- Construir modelo siamés robusto ---
    print("[🧠] Entrenando modelo siamés...")
    siamese, embedder = build_siamese(input_shape=X.shape[1:], embed_dim=128, use_l2norm=True)
    siamese.compile(optimizer=Adam(learning_rate=1e-4), loss=contrastive_loss(margin=1.0))

    # --- Callback: detener si no mejora ---
    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # --- Entrenamiento con generador ---
    batch_size = 16
    steps_per_epoch = len(X) // batch_size

    history = siamese.fit(
        data_generator(X, y, batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        callbacks=[early_stop],
        verbose=1
    )

    # --- Guardar modelo entrenado ---
    embedder.save("embedder.keras")
    print("✅ Modelo preentrenado guardado como 'embedder.keras'")

    # --- Graficar historial de pérdida ---
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.title('Curva de pérdida del modelo siamés')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_loss.png")
    plt.show()
    print("📊 Gráfico guardado como 'train_loss.png'")

if __name__ == "__main__":
    main()