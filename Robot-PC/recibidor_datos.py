import socket
import base64
import time

import numpy as np
import cv2
from ultralytics import YOLO

# ========= Configuración =========
ROBOT_IP   = "10.182.184.103"  # IP del TurtleBot4
ROBOT_PORT = 6000              # Debe coincidir con el nodo de telemetría

DESIRED_DOMAIN_ID = 10          # Debe coincidir con ROS_DOMAIN_ID del robot
PAIRING_CODE      = "ROBOT_A_100"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_10"  # por seguridad extra

# Modelo YOLO
MODEL_PATH = "../best.pt"  # Ruta al modelo entrenado

# ========= CONFIGURACIÓN =========
ROBOT_IP = "10.182.184.103"
ROBOT_PORT_TELEMETRY = 6000    # Recibir imágenes
ROBOT_PORT_CMD = 5007          # Enviar comandos

# ========= PARÁMETROS DE CONTROL =========
MAX_LINEAR_VEL = 0.26          # Velocidad lineal máxima (m/s)
MAX_ANGULAR_VEL = 1.0          # Velocidad angular máxima (rad/s)
KP_ANGULAR = 2.0               # Ganancia proporcional para giro
KP_LINEAR = 0.6                # Ganancia proporcional para avance
MIN_CONFIDENCE = 0.5           # Confianza mínima de detección
TARGET_AREA_RATIO = 0.15       # Área objetivo (15% del frame)
STOP_THRESHOLD = 0.05          # Umbral para considerar centrado
SEARCH_ANGULAR_VEL = 0.3       # Velocidad de búsqueda cuando no hay robot

class Sender:
    """Envía comandos de movimiento al robot"""
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        robot_addr = (ROBOT_IP, ROBOT_PORT_CMD)
        
        print("\n📡 Realizando handshake con comandos...")
        if not do_handshake(self.sock, robot_addr):
            raise Exception("Handshake de comandos fallido")
        
        print("[SENDER] Listo para enviar comandos")
    
    def send_cmd(self, vx, wz):
        """Envía comando de velocidad"""
        try:
            msg = f"CMD {vx:.4f} {wz:.4f}".encode("utf-8")
            self.sock.sendto(msg, (ROBOT_IP, ROBOT_PORT_CMD))
        except Exception as e:
            print(f"❌ Error enviando comando: {e}")
    
    def stop(self):
        """Detiene el robot"""
        self.send_cmd(0.0, 0.0)
    
    def close(self):
        self.stop()
        self.sock.close()


class RobotController:
    """Controla el movimiento del robot basado en detecciones de YOLO"""
    
    def __init__(self, sender):
        self.sender = sender
    
    def calculate_control(self, detections, img_width, img_height):
        """
        Calcula velocidades lineal y angular basadas en detecciones.
        
        Args:
            detections: Lista de detecciones de YOLO
            img_width: Ancho de la imagen
            img_height: Alto de la imagen
            
        Returns:
            tuple: (vx, wz, status)
        """
        if not detections:
            # Sin robot detectado: buscar girando
            return 0.0, SEARCH_ANGULAR_VEL, "BUSCANDO 🔄"
        
        # Seleccionar el robot más grande (más cercano)
        best_detection = max(detections, key=lambda d: d['area'])
        
        # Calcular error horizontal (centrado)
        cx = best_detection['center_x']
        frame_center_x = img_width / 2
        error_x = (cx - frame_center_x) / frame_center_x
        
        # Calcular error de distancia (área)
        area = best_detection['area']
        target_area = img_width * img_height * TARGET_AREA_RATIO
        error_area = (target_area - area) / target_area
        
        # Control proporcional
        wz = np.clip(KP_ANGULAR * error_x, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        vx = np.clip(KP_LINEAR * error_area, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        
        # Determinar estado
        if abs(error_x) < STOP_THRESHOLD and abs(error_area) < STOP_THRESHOLD:
            vx = 0.0
            wz = 0.0
            status = "EN POSICIÓN ✅"
        else:
            status = "PERSIGUIENDO 🤖"
        
        return vx, wz, status
    
    def execute_control(self, vx, wz):
        """Envía comandos de control al robot"""
        self.sender.send_cmd(vx, wz)


class DetectionProcessor:
    """Procesa detecciones de YOLO y extrae información relevante"""
    
    @staticmethod
    def extract_detections(results, class_filter='robots'):
        """
        Extrae información de detecciones filtradas por clase.
        
        Args:
            results: Resultados de YOLO
            class_filter: Clase a filtrar (default: 'robots')
            
        Returns:
            list: Lista de diccionarios con información de detecciones
        """
        detections = []
        
        for r in results:
            for box in r.boxes:
                # Obtener información de la caja
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                
                # Filtrar por clase
                if cls_name != class_filter:
                    continue
                
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Calcular propiedades
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                detection = {
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center_x': cx,
                    'center_y': cy,
                    'width': width,
                    'height': height,
                    'area': area
                }
                detections.append(detection)
        
        return detections
    
    @staticmethod
    def draw_detections(img, detections, vx, wz, status):
        """
        Dibuja detecciones y información de control en la imagen.
        
        Args:
            img: Imagen OpenCV
            detections: Lista de detecciones
            vx: Velocidad lineal
            wz: Velocidad angular
            status: Estado del sistema
            
        Returns:
            img: Imagen anotada
        """
        h, w = img.shape[:2]
        
        # Dibujar líneas de referencia (centro)
        cv2.line(img, (w//2, 0), (w//2, h), (255, 0, 0), 2)
        cv2.line(img, (0, h//2), (w, h//2), (255, 0, 0), 2)
        
        # Dibujar detecciones
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center_x'], det['center_y']
            conf = det['confidence']
            
            # Bounding box verde
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Centro del robot
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
            
            # Etiqueta con clase y confianza
            label = f"ROBOT {conf:.0%}"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # HUD con información
        y_offset = 40
        cv2.putText(img, f"Robots: {len(detections)}", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y_offset += 40
        cv2.putText(img, f"vx: {vx:.3f} m/s", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(img, f"wz: {wz:.3f} rad/s", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Estado del sistema
        if "PERSIGUIENDO" in status:
            color = (0, 255, 255)  # Amarillo
        elif "POSICIÓN" in status:
            color = (0, 255, 0)    # Verde
        else:  # BUSCANDO
            color = (0, 165, 255)  # Naranja
        
        cv2.putText(img, f"Estado: {status}", (15, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return img

def do_handshake(sock: socket.socket, robot_addr):
    sock.settimeout(1.0)
    print(f"[HANDSHAKE] Iniciando con {robot_addr}...")
    while True:
        # Enviar HELLO <domain> <pairing_code>
        msg = f"HELLO {DESIRED_DOMAIN_ID} {PAIRING_CODE}".encode("utf-8")
        sock.sendto(msg, robot_addr)

        try:
            data, addr = sock.recvfrom(4096)
            text = data.decode("utf-8").strip()
            parts = text.split()

            if len(parts) >= 3 and parts[0] == "ACK":
                domain_str = parts[1]
                robot_name = " ".join(parts[2:])

                print(f"[HANDSHAKE] Recibido: '{text}' desde {addr}")

                try:
                    domain_id = int(domain_str)
                except ValueError:
                    print("[HANDSHAKE] domain_id inválido, reintentando...")
                    continue

                if domain_id != DESIRED_DOMAIN_ID:
                    print(f"[HANDSHAKE] ROS_DOMAIN_ID no coincide "
                          f"(esperado={DESIRED_DOMAIN_ID}, recibido={domain_id}). Reintentando...")
                    continue

                if robot_name != EXPECTED_ROBOT_NAME:
                    print(f"[HANDSHAKE] robot_name no coincide "
                          f"(esperado={EXPECTED_ROBOT_NAME}, recibido={robot_name}). Reintentando...")
                    continue

                print(f"[HANDSHAKE] Emparejado con '{robot_name}' (domain {domain_id}).")
                sock.settimeout(None)  # sin timeout para recibir telemetría
                return
            else:
                print(f"[HANDSHAKE] Mensaje inesperado: '{text}', reintentando...")

        except socket.timeout:
            print("[HANDSHAKE] Timeout esperando ACK, reintentando...")

        except KeyboardInterrupt:
            print("[HANDSHAKE] Cancelado por el usuario.")
            raise


def handle_scan(parts):
    """
    parts: lista de strings del mensaje:
    SCAN <domain_id> <robot_name> <sec> <nsec> <angle_min> <angle_inc> <n> r1 ... rn
    """
    if len(parts) < 8:
        print("[SCAN] Mensaje demasiado corto.")
        return

    try:
        domain_id = int(parts[1])
        robot_name = parts[2]
        sec = int(parts[3])
        nsec = int(parts[4])
        angle_min = float(parts[5])
        angle_inc = float(parts[6])
        n = int(parts[7])

        ranges_str = parts[8:]
        if len(ranges_str) != n:
            print(f"[SCAN] n={n} pero llegaron {len(ranges_str)} rangos. Usando min(len, n).")
        n_effective = min(n, len(ranges_str))

        ranges = [float(r) for r in ranges_str[:n_effective]]

        # Aquí puedes hacer lo que quieras con el LIDAR.
        # Demo: imprimir algunos valores cada vez.
        print(f"[SCAN] robot={robot_name} domain={domain_id} "
              f"t={sec}.{nsec:09d} n={n_effective} "
              f"ejemplo={ranges[:5]}")

    except ValueError as e:
        print(f"[SCAN] Error parseando mensaje: {e}")


def handle_img(parts, model, controller):
    """
    Procesa imagen, detecta robots y controla el movimiento.
    
    Args:
        parts: lista de strings del mensaje IMG
        model: modelo YOLO para detección
        controller: RobotController para enviar comandos
    """
    if len(parts) < 6:
        print("[IMG] Mensaje demasiado corto.")
        return

    try:
        domain_id = int(parts[1])
        robot_name = parts[2]
        sec = int(parts[3])
        nsec = int(parts[4])

        # Decodificar imagen
        b64_str = " ".join(parts[5:])
        jpeg_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[IMG] Error al decodificar imagen.")
            return

        h, w = img.shape[:2]

        # ===== DETECCIÓN CON YOLO =====
        results = model(img, conf=MIN_CONFIDENCE, verbose=False)
        detections = DetectionProcessor.extract_detections(results, class_filter='robots')
        
        # ===== CALCULAR CONTROL =====
        vx, wz, status = controller.calculate_control(detections, w, h)
        
        # ===== EJECUTAR CONTROL =====
        controller.execute_control(vx, wz)
        
        # ===== LOG =====
        if detections:
            best = max(detections, key=lambda d: d['area'])
            print(f"[IMG] 🤖 {len(detections)} robot(s) | "
                  f"vx={vx:.3f} wz={wz:.3f} | {status}")
        else:
            print(f"[IMG] ❌ Sin robots | Buscando...")
        
        # ===== VISUALIZACIÓN =====
        annotated_img = DetectionProcessor.draw_detections(img, detections, vx, wz, status)
        cv2.imshow(f"🤖 Robot Follower - {robot_name}", annotated_img)
        cv2.waitKey(1)

    except Exception as e:
        print(f"[IMG] Error manejando imagen: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Función principal: inicializa componentes y ejecuta loop de telemetría.
    """
    print("\n" + "="*70)
    print("🤖 ROBOT FOLLOWER - Sistema de Persecución Autónomo")
    print("="*70)
    
    # ===== CARGAR MODELO YOLO =====
    print(f"\n[MAIN] Cargando modelo YOLO desde {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ [MAIN] Modelo cargado. Clases: {model.names}")
        
        # Verificar clase 'robots'
        if 'robots' not in model.names.values():
            print(f"⚠️  Advertencia: Clase 'robots' no encontrada.")
            print(f"   Clases disponibles: {list(model.names.values())}")
    except Exception as e:
        print(f"❌ [MAIN] Error al cargar modelo: {e}")
        return
    
    # ===== INICIALIZAR SENDER =====
    try:
        sender = Sender()
    except Exception as e:
        print(f"❌ [MAIN] Error inicializando Sender: {e}")
        return
    
    # ===== INICIALIZAR CONTROLLER =====
    controller = RobotController(sender)
    print("✅ [MAIN] Sistema de control inicializado")
    
    # ===== CONFIGURAR SOCKET TELEMETRÍA =====
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_addr = (ROBOT_IP, ROBOT_PORT)

    # ===== HANDSHAKE =====
    do_handshake(sock, robot_addr)

    print("\n" + "="*70)
    print("🟢 Sistema activo - Recibiendo telemetría y persiguiendo robots")
    print("Presiona Ctrl+C para salir")
    print("="*70 + "\n")
    
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            text = data.decode("utf-8", errors="ignore")
            parts = text.split()

            if not parts:
                continue

            msg_type = parts[0]

            if msg_type == "SCAN":
                handle_scan(parts)
            elif msg_type == "IMG":
                handle_img(parts, model, controller)
            else:
                print(f"[MAIN] Mensaje desconocido desde {addr}: '{msg_type}'")

    except KeyboardInterrupt:
        print("\n\n🛑 [MAIN] Interrupción del usuario...")
    except Exception as e:
        print(f"\n❌ [MAIN] Error en loop principal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 [MAIN] Deteniendo robot...")
        sender.stop()
        time.sleep(0.5)
        sender.close()
        sock.close()
        cv2.destroyAllWindows()
        print("✅ [MAIN] Sistema cerrado correctamente\n")


if __name__ == "__main__":
    main()