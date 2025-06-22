import os
import shutil

# Configura estas rutas
RUTA_IMAGENES = "../Celeba/img_align_celeba"  # carpeta descomprimida del ZIP
ARCHIVO_IDENTIDADES = "identity_CelebA.txt"

# Crear carpeta destino fuera del directorio actual
DESTINO = os.path.abspath(os.path.join("..", "celeba_por_persona"))

# Leer asignaciones: imagen -> persona_id
persona_map = {}
with open(ARCHIVO_IDENTIDADES, "r") as f:
    for linea in f:
        nombre_img, persona_id = linea.strip().split()
        persona_map.setdefault(persona_id, []).append(nombre_img)

# Crear estructura por persona
os.makedirs(DESTINO, exist_ok=True)
contador_total = 0
contador_imagenes = 0
for persona_id, imagenes in persona_map.items():
    if len(imagenes) < 2:
        continue  # Saltar si no tiene al menos 2 fotos
    carpeta_persona = os.path.join(DESTINO, f"persona_{persona_id.zfill(5)}")
    os.makedirs(carpeta_persona, exist_ok=True)
    for img in imagenes:
        origen = os.path.join(RUTA_IMAGENES, img)
        destino = os.path.join(carpeta_persona, img)
        if os.path.exists(origen):
            shutil.copy(origen, destino)
            contador_imagenes += 1
    contador_total += 1

print(f"✅ Carpetas creadas: {contador_total}")
print(f"🖼️  Total de imágenes copiadas: {contador_imagenes}")