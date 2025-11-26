#!/usr/bin/env python3
"""
Script para probar el modelo YOLO con videos de la carpeta Vision
"""
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Configuración
MODEL_PATH = "best.pt"
VIDEO_DIR = "Vision"

def test_video(video_path: str, model: YOLO, conf: float = 0.5):
    """
    Procesa un video con detección YOLO y muestra los resultados.
    
    Args:
        video_path: Ruta al archivo de video
        model: Modelo YOLO cargado
        conf: Umbral de confianza para detecciones
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[INFO] Procesando: {Path(video_path).name}")
    print(f"[INFO] FPS: {fps:.2f}, Frames totales: {total_frames}")
    print("[INFO] Presiona 'q' para salir, 'p' para pausar")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n[INFO] Video finalizado")
                break
            
            frame_count += 1
            
            # Realizar detección
            results = model(frame, conf=conf, verbose=False)
            
            # Anotar frame
            annotated_frame = results[0].plot()
            
            # Información en pantalla
            num_detections = len(results[0].boxes)
            info_text = f"Frame: {frame_count}/{total_frames} | Robots detectados: {num_detections}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow("YOLO Detection Test", annotated_frame)
        
        # Control de teclado
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n[INFO] Saliendo...")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Pausado' if paused else 'Reanudando'}")
    
    cap.release()
    cv2.destroyAllWindows()


def list_videos(video_dir: str):
    """Lista todos los videos disponibles en el directorio."""
    video_path = Path(video_dir)
    
    if not video_path.exists():
        print(f"[ERROR] No existe el directorio: {video_dir}")
        return []
    
    videos = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))
    return sorted(videos)


def main():
    # Cargar modelo
    print(f"[INFO] Cargando modelo YOLO: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print(f"[INFO] Modelo cargado exitosamente")
        print(f"[INFO] Clases detectables: {model.names}")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return
    
    # Listar videos disponibles
    videos = list_videos(VIDEO_DIR)
    
    if not videos:
        print(f"[ERROR] No se encontraron videos en {VIDEO_DIR}")
        return
    
    print(f"\n[INFO] Videos disponibles en {VIDEO_DIR}:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")
    
    # Seleccionar video
    if len(sys.argv) > 1:
        # Si se pasa argumento, usar ese número
        try:
            choice = int(sys.argv[1])
            if 1 <= choice <= len(videos):
                selected_video = videos[choice - 1]
            else:
                print(f"[ERROR] Número fuera de rango. Usando primer video.")
                selected_video = videos[0]
        except ValueError:
            print(f"[ERROR] Argumento inválido. Usando primer video.")
            selected_video = videos[0]
    else:
        # Usar el primer video por defecto
        selected_video = videos[0]
        print(f"\n[INFO] Usando video por defecto: {selected_video.name}")
        print(f"[INFO] Tip: Ejecuta 'python test_video.py <número>' para elegir otro video")
    
    # Procesar video
    test_video(str(selected_video), model, conf=0.5)
    
    print("\n[INFO] Proceso completado")


if __name__ == "__main__":
    main()
