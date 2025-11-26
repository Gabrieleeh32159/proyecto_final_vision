from ultralytics import YOLO
import cv2
import numpy as np

class Model:
    def __init__(self, dataset_path: str, weights_path: str, base_model_path: str):
        """
        Inicializa el modelo YOLO.
        
        Args:
            dataset_path: Ruta al dataset (para entrenamiento, si aplica)
            weights_path: Ruta a los pesos entrenados personalizados
            base_model_path: Ruta al modelo base (best.pt)
        """
        self.dataset_path = dataset_path
        self.weights_path = weights_path
        self.base_model_path = base_model_path
        
        # Cargar el modelo
        try:
            if weights_path and weights_path != "<default_path>":
                print(f"[MODEL] Cargando modelo desde: {weights_path}")
                self.model = YOLO(weights_path)
            else:
                print(f"[MODEL] Cargando modelo base desde: {base_model_path}")
                self.model = YOLO(base_model_path)
        except Exception as e:
            print(f"[MODEL] Error al cargar modelo: {e}")
            raise
    
    def predict(self, image: np.ndarray, show: bool = False, conf: float = 0.7):
        """
        Realiza predicción sobre una imagen.
        
        Args:
            image: Imagen en formato numpy array (BGR)
            show: Si mostrar las detecciones
            conf: Umbral de confianza para las detecciones
            
        Returns:
            tuple: (imagen_anotada, mejor_detección)
                   mejor_detección es None si no hay detecciones
        """
        try:
            # Realizar predicción
            results = self.model(image, conf=conf, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return image, None
            
            # Obtener la primera detección (mayor confianza)
            result = results[0]
            
            # Imagen anotada
            annotated_img = result.plot()
            
            # Mejor detección (primera caja)
            best_box = result.boxes[0] if len(result.boxes) > 0 else None
            
            if show and best_box is not None:
                cv2.imshow("YOLO Detection", annotated_img)
                cv2.waitKey(1)
            
            return annotated_img, best_box
            
        except Exception as e:
            print(f"[MODEL] Error en predicción: {e}")
            return image, None
    
    def train(self, epochs: int = 50, imgsz: int = 640):
        """
        Entrena el modelo con el dataset especificado.
        
        Args:
            epochs: Número de épocas de entrenamiento
            imgsz: Tamaño de las imágenes
        """
        if not self.dataset_path or self.dataset_path == "<default_path>":
            print("[MODEL] No se especificó un dataset válido para entrenamiento")
            return
        
        try:
            print(f"[MODEL] Iniciando entrenamiento por {epochs} épocas...")
            results = self.model.train(
                data=self.dataset_path,
                epochs=epochs,
                imgsz=imgsz,
                patience=10,
                save=True,
                verbose=True
            )
            print("[MODEL] Entrenamiento completado")
            return results
        except Exception as e:
            print(f"[MODEL] Error en entrenamiento: {e}")
            raise
