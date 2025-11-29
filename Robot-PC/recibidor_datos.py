import socket, struct
import base64
import time
import sys
import os
import math

import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Import tracker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracker import HybridRobotTracker

# ========= Configuración =========
DESIRED_DOMAIN_ID = 10
PAIRING_CODE = "ROBOT_A_100"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_10"

MODEL_PATH = "../best.pt"

ROBOT_IP = "192.168.0.220"
ROBOT_PORT_TELEMETRY = 6000
ROBOT_PORT_CMD = 5007

# ========= PARÁMETROS DE CONTROL AGRESIVOS =========
MAX_LINEAR_VEL = 2.0
MAX_ANGULAR_VEL = 2.0          # AUMENTADO para giros más rápidos
KP_ANGULAR = 2.5               # MUY AGRESIVO para seguir laterales
KP_LINEAR = 3.0                # MÁS AGRESIVO para acelerar
MIN_CONFIDENCE = 0.75          # 75% de confianza
TARGET_AREA_RATIO = 0.15       # Área objetivo más pequeña (más cerca)
STOP_THRESHOLD = 0.02          # MUY sensible para centrado
SEARCH_ANGULAR_VEL = 0.8       # Búsqueda más rápida
SEARCH_LINEAR_VEL = 0.5        # Movimiento más rápido en búsqueda
SEARCH_TIMEOUT = 10.0

# ========= PARÁMETROS KALMAN =========
KALMAN_PROCESS_NOISE = 0.005   # MÁS SENSIBLE
KALMAN_MEASUREMENT_NOISE = 0.05 # MÁS SENSIBLE
KALMAN_STATE_TRANSITION = 60    # Más frames de predicción


class KalmanFilterTracker:
    """Filtro Kalman para suavizar tracking"""
    
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # Estado: [cx, cy, width, height]
        self.kf.x = np.array([[img_width/2], [img_height/2], [100], [100]], dtype=float)
        
        # Matriz de transición (posición + velocidad)
        self.kf.F = np.array([[1., 0., 0.15, 0.],
                              [0., 1., 0., 0.15],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        
        # Matriz de medición (solo posición)
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        
        # Ruido de proceso
        self.kf.Q *= KALMAN_PROCESS_NOISE
        
        # Ruido de medición
        self.kf.R *= KALMAN_MEASUREMENT_NOISE
        
        # Covarianza inicial
        self.kf.P *= 1.0
        
        self.frames_without_detection = 0
        self.is_initialized = False
    
    def update(self, center_x, center_y, width, height):
        """Actualiza Kalman con nueva detección"""
        z = np.array([[center_x], [center_y]])
        
        if not self.is_initialized:
            self.kf.x = np.array([[center_x], [center_y], [width], [height]], dtype=float)
            self.is_initialized = True
        
        self.kf.predict()
        self.kf.update(z)
        self.frames_without_detection = 0
    
    def predict(self):
        """Predice posición sin detección"""
        self.kf.predict()
        self.frames_without_detection += 1
    
    def get_position(self):
        """Retorna posición predicha"""
        cx = float(self.kf.x[0])
        cy = float(self.kf.x[1])
        w = float(self.kf.x[2])
        h = float(self.kf.x[3])
        
        # Clampear a límites de imagen
        cx = np.clip(cx, 0, self.img_width)
        cy = np.clip(cy, 0, self.img_height)
        
        return cx, cy, w, h
    
    def is_valid(self):
        """Revisa si la predicción es válida"""
        return self.frames_without_detection < KALMAN_STATE_TRANSITION


class Sender:
    """Envía comandos de movimiento al robot"""
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_addr = (ROBOT_IP, ROBOT_PORT_CMD)
        
        print(f"\n📡 [SENDER] Configurado para enviar comandos a {self.robot_addr}")
        print("✅ [SENDER] Listo para enviar comandos")
    
    def send_cmd(self, vel: float, angular_vel: float):
        v = max(min(vel, 2.0), 0.0)
        w = max(min(angular_vel, 2.0), -2.0)

        payload = struct.pack('ff', w, v)
        try:
            self.sock.sendto(payload, self.robot_addr)
            print(f"[SENDER] 🎯 v={v:+.2f}m/s | w={w:+.2f}rad/s")
        except Exception as e:
            print(f"[SENDER][ERROR] {e}")
    
    def stop(self):
        self.send_cmd(0.0, 0.0)
    
    def close(self):
        self.stop()
        self.sock.close()


class RobotController:
    """Controla el movimiento del robot con Kalman AGRESIVO"""
    
    def __init__(self, sender):
        self.sender = sender
        self.time_lost_tracking = None
        self.search_start_time = None
        self.search_direction = 1
    
    def calculate_control(self, tracking_info, kalman_tracker, img_width, img_height):
        """Calcula velocidades AGRESIVAMENTE"""
        
        if tracking_info is not None:
            # Hay detección NUEVA - actualizar Kalman
            bbox = tracking_info['bbox']
            center = tracking_info['center']
            cx, cy = center['x'], center['y']
            width = bbox['x2'] - bbox['x1']
            height = bbox['y2'] - bbox['y1']
            
            kalman_tracker.update(cx, cy, width, height)
            self.time_lost_tracking = None
            self.search_start_time = None
            source_emoji = "🎯"
            
        else:
            # Sin detección - predecir con Kalman
            kalman_tracker.predict()
            
            if not kalman_tracker.is_valid():
                # Kalman expiró - entrar en modo búsqueda
                current_time = time.time()
                
                if self.time_lost_tracking is None:
                    self.time_lost_tracking = current_time
                    self.search_start_time = current_time
                
                time_without_robot = current_time - self.time_lost_tracking
                
                if time_without_robot > SEARCH_TIMEOUT:
                    # Timeout - resetear
                    self.time_lost_tracking = None
                    self.search_start_time = None
                    self.search_direction = 1
                
                # BÚSQUEDA AGRESIVA: Girar + avanzar
                return SEARCH_LINEAR_VEL, self.search_direction * SEARCH_ANGULAR_VEL, "🔄 BUSCANDO"
            
            source_emoji = "📡"
        
        # Usar posición de Kalman (suavizada)
        cx, cy, width, height = kalman_tracker.get_position()
        area = width * height
        
        # Calcular TODOS los errores
        frame_center_x = img_width / 2
        frame_center_y = img_height / 2
        
        # ERROR HORIZONTAL (para girar)
        error_x = (cx - frame_center_x) / frame_center_x
        
        # ERROR VERTICAL (para ajuste fino)
        error_y = (cy - frame_center_y) / frame_center_y
        
        target_area = img_width * img_height * TARGET_AREA_RATIO
        if target_area <= 0:
            target_area = 1.0
        
        area_ratio = area / target_area
        
        print(f"[CONTROL] 📍 Pos: ({cx:.0f},{cy:.0f}) | Error: X={error_x:+.3f} Y={error_y:+.3f} | Area_ratio={area_ratio:.3f}")
        
        # ========== CONTROL AGRESIVO ==========
        
        # GIRO AGRESIVO - Sigue lateralmente sin parar
        wz_base = KP_ANGULAR * error_x
        wz = np.clip(wz_base * 2.5, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)  # EXTRA agresivo
        
        # VELOCIDAD LINEAL AGRESIVA
        if area_ratio < 0.7:
            # Muy lejos - acelerar
            distance_error = 1.0 - area_ratio
            vx = np.clip(KP_LINEAR * distance_error * 1.5, 0.3, MAX_LINEAR_VEL)
        elif area_ratio < 1.0:
            # Lejos - avanzar normal
            distance_error = 1.0 - area_ratio
            vx = np.clip(KP_LINEAR * distance_error, 0.2, MAX_LINEAR_VEL)
        else:
            # Cerca o MUY cerca - frenar gradualmente
            vx = 0.05  # Mínimo para mantener presencia
        
        # ========== NUNCA DETENERSE COMPLETAMENTE ==========
        # Siempre hay velocidad angular para seguir
        if abs(error_x) < STOP_THRESHOLD:
            wz = 0.0
            status = "✅ CENTRADO"
        else:
            status = f"🔥 SIGUIENDO {source_emoji}"
        
        if abs(1.0 - area_ratio) < 0.03:
            vx = 0.0
            status = "📍 DISTANCIA PERFECTA"
        
        print(f"[CONTROL] 🚀 vx={vx:.2f} | wz={wz:+.2f} | {status}")
        
        return vx, wz, status
    
    def execute_control(self, vx, wz):
        self.sender.send_cmd(vx, wz)


def draw_control_hud(img, tracking_info, vx, wz, status, kalman_valid):
    """Dibuja HUD con toda la info"""
    h, w = img.shape[:2]
    
    # Línea de centro
    cv2.line(img, (w//2, 0), (w//2, h), (100, 100, 100), 2)
    cv2.line(img, (0, h//2), (w, h//2), (100, 100, 100), 2)
    
    # Velocidades en grande
    y_offset = 40
    cv2.putText(img, f"VELOCIDAD LINEAL: {vx:+.2f} m/s", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    y_offset += 40
    cv2.putText(img, f"VELOCIDAD ANGULAR: {wz:+.2f} rad/s", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    # Estado Kalman
    kalman_color = (0, 255, 0) if kalman_valid else (0, 0, 255)
    kalman_status = "✅ KALMAN ACTIVO" if kalman_valid else "⚠️ KALMAN EXPIRADO"
    y_offset += 40
    cv2.putText(img, kalman_status, (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, kalman_color, 2)
    
    # Estado principal
    if "SIGUIENDO" in status:
        color = (0, 255, 0)
    elif "CENTRADO" in status:
        color = (0, 255, 255)
    elif "BUSCANDO" in status:
        color = (0, 165, 255)
    elif "DISTANCIA" in status:
        color = (255, 0, 0)
    else:
        color = (200, 200, 200)
    
    cv2.putText(img, status, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    return img


def do_handshake(sock: socket.socket, robot_addr):
    sock.settimeout(1.0)
    print(f"[HANDSHAKE] Iniciando con {robot_addr}...")
    while True:
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
                    print(f"[HANDSHAKE] ROS_DOMAIN_ID no coincide")
                    continue

                if robot_name != EXPECTED_ROBOT_NAME:
                    print(f"[HANDSHAKE] robot_name no coincide")
                    continue

                print(f"[HANDSHAKE] ✅ Emparejado con '{robot_name}' (domain {domain_id}).")
                sock.settimeout(None)
                return
            else:
                print(f"[HANDSHAKE] Mensaje inesperado: '{text}'")

        except socket.timeout:
            print("[HANDSHAKE] Timeout esperando ACK...")

        except KeyboardInterrupt:
            print("[HANDSHAKE] Cancelado por usuario.")
            raise


def handle_img(parts, tracker, controller, kalman_tracker):
    """Procesa imagen con Kalman AGRESIVO"""
    if len(parts) < 6:
        print("[IMG] Mensaje demasiado corto.")
        return

    try:
        domain_id = int(parts[1])
        robot_name = parts[2]
        sec = int(parts[3])
        nsec = int(parts[4])

        b64_str = " ".join(parts[5:])
        jpeg_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[IMG] Error al decodificar imagen.")
            return

        h, w = img.shape[:2]
        
        if kalman_tracker.img_width != w or kalman_tracker.img_height != h:
            kalman_tracker = KalmanFilterTracker(w, h)

        annotated_frame, tracking_info = tracker.process_frame(img)
        
        vx, wz, status = controller.calculate_control(tracking_info, kalman_tracker, w, h)
        
        controller.execute_control(vx, wz)
        
        kalman_valid = kalman_tracker.is_valid()
        
        if tracking_info:
            conf = tracking_info['confidence']
            print(f"\n[IMG] 🎯 DETECTADO (conf={conf:.0%})")
        else:
            print(f"\n[IMG] 📡 PREDICIENDO CON KALMAN")
        
        annotated_frame = draw_control_hud(annotated_frame, tracking_info, vx, wz, status, kalman_valid)
        cv2.imshow(f"🤖 ROBOT TRACKER - SIGUIENDO AL PERRO", annotated_frame)
        cv2.waitKey(1)

    except Exception as e:
        print(f"[IMG] ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*80)
    print("🐕 🤖 ROBOT TRACKER - SIGUE AL PERRO CON TODO")
    print("="*80)
    print("✅ Confianza: 75%")
    print("✅ Kalman Filter activado")
    print("✅ Seguimiento 360°")
    print("✅ Búsqueda automática")
    print("="*80)
    
    try:
        tracker = HybridRobotTracker(
            target_labels=["robots"],
            conf_threshold_initial=0.75,
            conf_threshold_redetect=0.75,
            yolo_refresh_every=3,
            timeout_seconds=5.0,
            fps=30.0,
            imgsz=416
        )
        print(f"\n✅ Tracker inicializado")
    except Exception as e:
        print(f"❌ Error inicializando tracker: {e}")
        return
    
    try:
        sender = Sender()
    except Exception as e:
        print(f"❌ Error inicializando Sender: {e}")
        return
    
    controller = RobotController(sender)
    kalman_tracker = KalmanFilterTracker(640, 480)
    
    print("✅ Filtro Kalman inicializado")
    print("✅ Robot Controller listo\n")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_addr = (ROBOT_IP, ROBOT_PORT_TELEMETRY)

    do_handshake(sock, robot_addr)

    print("\n" + "="*80)
    print("🟢 SISTEMA ACTIVO - ¡ROBOT LISTO PARA SEGUIR AL PERRO!")
    print("="*80 + "\n")
    
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            text = data.decode("utf-8", errors="ignore")
            parts = text.split()

            if not parts:
                continue

            msg_type = parts[0]

            if msg_type == "IMG":
                handle_img(parts, tracker, controller, kalman_tracker)

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupción del usuario...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Deteniendo robot...")
        sender.stop()
        time.sleep(0.5)
        sender.close()
        sock.close()
        cv2.destroyAllWindows()
        print("✅ Sistema cerrado\n")


if __name__ == "__main__":
    main()