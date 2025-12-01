# =============================================================================
# Script para finetunear YOLOv8 con dataset de Roboflow en Google Colab (GPU A100)
# =============================================================================
# Instrucciones para Google Colab:
# 1. Sube tu modelo bestv2.pt a Google Drive o directamente a Colab
# 2. Ejecuta este script celda por celda o como un único bloque
# =============================================================================

# %% [markdown]
# ## 1. Instalación de dependencias

# %%
# Instalar dependencias necesarias
# !pip install roboflow ultralytics

# %% [markdown]
# ## 2. Imports y configuración

# %%
import os
import shutil
import random
import yaml
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

# Configuración de semilla para reproducibilidad
SEED = 42
random.seed(SEED)

# Configuración de división del dataset
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ruta base donde se guardará el dataset unificado
BASE_DIR = Path("/content")
DATASET_DIR = BASE_DIR / "dataset_unified"
ORIGINAL_DATASET_DIR = BASE_DIR / "Perrobot2-3"  # Nombre que Roboflow asigna al descargar

# %% [markdown]
# ## 3. Descargar dataset de Roboflow

# %%
def download_roboflow_dataset():
    """Descarga el dataset desde Roboflow"""
    print("📥 Descargando dataset de Roboflow...")
    rf = Roboflow(api_key="LLuSSCJQTRIpGVY3WsII")
    project = rf.workspace("perrobot").project("perrobot2-em7yi")
    version = project.version(3)
    dataset = version.download("yolov8", location=str(BASE_DIR / "Perrobot2-3"))
    print("✅ Dataset descargado correctamente")
    return dataset

# %% [markdown]
# ## 4. Unificar y redistribuir el dataset

# %%
def collect_all_images_and_labels(original_dir: Path):
    """
    Recolecta todas las imágenes y labels de train, valid y test
    
    Returns:
        list: Lista de tuplas (imagen_path, label_path)
    """
    all_data = []
    splits = ["train", "valid", "test"]
    
    for split in splits:
        images_dir = original_dir / split / "images"
        labels_dir = original_dir / split / "labels"
        
        if not images_dir.exists():
            print(f"⚠️ Directorio {images_dir} no existe, saltando...")
            continue
            
        for img_path in images_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                # Buscar el label correspondiente
                label_name = img_path.stem + ".txt"
                label_path = labels_dir / label_name
                
                if label_path.exists():
                    all_data.append((img_path, label_path))
                else:
                    print(f"⚠️ Label no encontrado para: {img_path.name}")
    
    print(f"📊 Total de imágenes recolectadas: {len(all_data)}")
    return all_data


def create_unified_dataset(all_data: list, output_dir: Path):
    """
    Crea el dataset unificado con la nueva división 70/15/15
    
    Args:
        all_data: Lista de tuplas (imagen_path, label_path)
        output_dir: Directorio de salida
    """
    # Mezclar datos aleatoriamente
    random.shuffle(all_data)
    
    # Calcular índices de división
    total = len(all_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        "train": all_data[:train_end],
        "valid": all_data[train_end:val_end],
        "test": all_data[val_end:]
    }
    
    print(f"\n📊 División del dataset:")
    print(f"   - Train: {len(splits['train'])} imágenes ({len(splits['train'])/total*100:.1f}%)")
    print(f"   - Valid: {len(splits['valid'])} imágenes ({len(splits['valid'])/total*100:.1f}%)")
    print(f"   - Test:  {len(splits['test'])} imágenes ({len(splits['test'])/total*100:.1f}%)")
    
    # Crear estructura de directorios y copiar archivos
    for split_name, split_data in splits.items():
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, label_path in split_data:
            # Copiar imagen
            shutil.copy2(img_path, images_dir / img_path.name)
            # Copiar label
            shutil.copy2(label_path, labels_dir / label_path.name)
    
    print(f"\n✅ Dataset unificado creado en: {output_dir}")


def get_class_names(original_dir: Path):
    """Obtiene los nombres de las clases del archivo data.yaml original"""
    yaml_path = original_dir / "data.yaml"
    
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('names', [])
    
    return []


def create_data_yaml(output_dir: Path, class_names: list):
    """Crea el archivo data.yaml para el entrenamiento"""
    data_yaml = {
        'path': str(output_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': class_names
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Archivo data.yaml creado: {yaml_path}")
    print(f"   Clases: {class_names}")
    
    return yaml_path

# %% [markdown]
# ## 5. Función principal de preparación del dataset

# %%
def prepare_dataset():
    """Función principal que prepara el dataset"""
    # Descargar dataset
    download_roboflow_dataset()
    
    # Limpiar dataset unificado anterior si existe
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    
    # Recolectar todos los datos
    all_data = collect_all_images_and_labels(ORIGINAL_DATASET_DIR)
    
    if len(all_data) == 0:
        raise ValueError("❌ No se encontraron datos en el dataset descargado")
    
    # Obtener nombres de clases
    class_names = get_class_names(ORIGINAL_DATASET_DIR)
    
    # Crear dataset unificado
    create_unified_dataset(all_data, DATASET_DIR)
    
    # Crear data.yaml
    yaml_path = create_data_yaml(DATASET_DIR, class_names)
    
    return yaml_path

# %% [markdown]
# ## 6. Entrenamiento del modelo

# %%
def train_model(data_yaml_path: Path, model_path: str = "bestv2.pt"):
    """
    Entrena el modelo YOLOv8
    
    Args:
        data_yaml_path: Ruta al archivo data.yaml
        model_path: Ruta al modelo pre-entrenado
    """
    print("\n🚀 Iniciando entrenamiento...")
    print(f"   Modelo base: {model_path}")
    print(f"   Dataset: {data_yaml_path}")
    
    # Cargar modelo
    model = YOLO(model_path)
    
    # Configuración optimizada para GPU A100
    results = model.train(
        data=str(data_yaml_path),
        epochs=100,                    # Número de épocas
        imgsz=640,                     # Tamaño de imagen
        batch=32,                      # Batch size (A100 permite batches grandes)
        patience=20,                   # Early stopping
        save=True,                     # Guardar checkpoints
        save_period=10,                # Guardar cada 10 épocas
        device=0,                      # GPU 0
        workers=8,                     # Workers para data loading
        project="runs/detect",         # Directorio de salida
        name="perrobot_finetuned",     # Nombre del experimento
        exist_ok=True,                 # Sobrescribir si existe
        pretrained=True,               # Usar pesos pre-entrenados
        optimizer="AdamW",             # Optimizador
        lr0=0.001,                     # Learning rate inicial
        lrf=0.01,                      # Learning rate final (lr0 * lrf)
        momentum=0.937,                # Momentum para SGD
        weight_decay=0.0005,           # Weight decay
        warmup_epochs=3,               # Épocas de warmup
        warmup_momentum=0.8,           # Momentum durante warmup
        warmup_bias_lr=0.1,            # Learning rate de bias durante warmup
        box=7.5,                       # Box loss gain
        cls=0.5,                       # Class loss gain
        dfl=1.5,                       # DFL loss gain
        hsv_h=0.015,                   # Augmentation: HSV-Hue
        hsv_s=0.7,                     # Augmentation: HSV-Saturation
        hsv_v=0.4,                     # Augmentation: HSV-Value
        degrees=0.0,                   # Augmentation: rotación
        translate=0.1,                 # Augmentation: traslación
        scale=0.5,                     # Augmentation: escala
        shear=0.0,                     # Augmentation: shear
        perspective=0.0,               # Augmentation: perspectiva
        flipud=0.0,                    # Augmentation: flip vertical
        fliplr=0.5,                    # Augmentation: flip horizontal
        mosaic=1.0,                    # Augmentation: mosaic
        mixup=0.0,                     # Augmentation: mixup
        copy_paste=0.0,                # Augmentation: copy-paste
        auto_augment="randaugment",    # Auto augment policy
        erasing=0.4,                   # Random erasing probability
        crop_fraction=1.0,             # Crop fraction for classification
        amp=True,                      # Automatic Mixed Precision
        fraction=1.0,                  # Fracción del dataset a usar
        profile=False,                 # Profile ONNX/TensorRT
        freeze=None,                   # Capas a congelar
        multi_scale=False,             # Multi-scale training
        overlap_mask=True,             # Overlap mask
        mask_ratio=4,                  # Mask ratio
        dropout=0.0,                   # Dropout
        val=True,                      # Validación durante entrenamiento
        plots=True,                    # Guardar plots
        rect=False,                    # Rectangular training
        cos_lr=False,                  # Cosine LR scheduler
        close_mosaic=10,               # Desactivar mosaic últimas N épocas
        resume=False,                  # Resumir entrenamiento
        seed=SEED,                     # Semilla
        verbose=True,                  # Verbose
    )
    
    print("\n✅ Entrenamiento completado!")
    print(f"   Mejor modelo guardado en: {results.save_dir}/weights/best.pt")
    
    return results

# %% [markdown]
# ## 7. Validación del modelo

# %%
def validate_model(model_path: str, data_yaml_path: Path):
    """Valida el modelo entrenado"""
    print("\n📊 Validando modelo...")
    
    model = YOLO(model_path)
    metrics = model.val(
        data=str(data_yaml_path),
        split='test',  # Usar split de test
        device=0,
        batch=32,
        imgsz=640,
    )
    
    print("\n📈 Métricas de validación:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    return metrics

# %% [markdown]
# ## 8. Ejecución principal

# %%
if __name__ == "__main__":
    # Preparar dataset
    data_yaml_path = prepare_dataset()
    
    # Entrenar modelo
    # NOTA: Asegúrate de subir bestv2.pt a /content/ en Colab
    # O modifica la ruta según donde hayas subido el modelo
    MODEL_PATH = "/content/bestv2.pt"
    
    results = train_model(data_yaml_path, MODEL_PATH)
    
    # Validar modelo final
    best_model_path = f"{results.save_dir}/weights/best.pt"
    validate_model(best_model_path, data_yaml_path)
    
    print("\n🎉 ¡Proceso completado!")
    print(f"   Modelo final: {best_model_path}")
    print("\n💡 Para descargar el modelo, usa:")
    print(f"   from google.colab import files")
    print(f"   files.download('{best_model_path}')")
