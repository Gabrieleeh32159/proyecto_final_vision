import socket, struct
import base64
import time
import sys
import os

import numpy as np
import cv2
from ultralytics import YOLO

# Import tracker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracker import HybridRobotTracker

# ========= Configuración =========
DESIRED_DOMAIN_ID = 10          # Debe coincidir con ROS_DOMAIN_ID del robot
PAIRING_CODE      = "ROBOT_A_100"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_10"  # por seguriad extra

# Modelo YOLO
MODEL_PATH = "../best.pt"  # Ruta al modelo entrenado

# ========= CONFIGURACIÓN =========
ROBOT_IP = "192.168.0.220"
ROBOT_PORT_TELEMETRY = 6000    # Recibir imágenes
ROBOT_PORT_CMD = 5007          # Enviar comandos

# ========= PARÁMETROS DE CONTROL =========
MAX_LINEAR_VEL = 2.0           # Velocidad lineal máxima (m/s) [0, 2] - Solo avance
MAX_ANGULAR_VEL = 1.0          # Velocidad angular máxima (rad/s) [-1, 1] - Limitado para giros suaves
KP_ANGULAR = 0.8               # Ganancia proporcional para giro (reducida para tracking suave)
KP_LINEAR = 2.0                # Ganancia proporcional para avance
MIN_CONFIDENCE = 0.5           # Confianza mínima de detección
TARGET_AREA_RATIO = 0.20       # Área objetivo (20% del frame) - robot se acerca más
STOP_THRESHOLD = 0.10          # Umbral para considerar centrado (más tolerante)
SEARCH_ANGULAR_VEL = 0.3       # Velocidad de búsqueda cuando no hay robot (rad/s)

class Sender:
    """Envía comandos de movimiento al robot"""
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_addr = (ROBOT_IP, ROBOT_PORT_CMD)
        
        print(f"\n📡 [SENDER] Configurado para enviar comandos a {self.robot_addr}")
        print("✅ [SENDER] Listo para enviar comandos")
    
    def send_cmd(self, vel: float, angular_vel: float):
        # Clamp velocidades: vx solo puede ser [0, 2], no retrocede
        v = max(min(vel, 2.0), 0.0)   # m/s - Solo avance
        w = max(min(angular_vel, 1.0), -1.0)  # rad/s - Giros suaves

        # IMPORTANTE: El robot espera (angular, lineal) en el protocolo UDP
        payload = struct.pack('ff', w, v)
        try:
            self.sock.sendto(payload, self.robot_addr)
            print(f"[SENDER] Enviado (v={v:+.3f}, w={w:+.3f})")
        except Exception as e:
            print(f"[SENDER][ERROR] {e}")
    
    def stop(self):
        """Detiene el robot"""
        self.send_cmd(0.0, 0.0)
    
    def close(self):
        self.stop()
        self.sock.close()


class RobotController:
    """Controla el movimiento del robot basado en tracking"""
    
    def __init__(self, sender, search_delay=1.5):
        self.sender = sender
        self.search_delay = search_delay  # Segundos antes de empezar a buscar
        self.time_lost_tracking = None  # Timestamp cuando se perdió el tracking
    
    def calculate_control(self, tracking_info, img_width, img_height):
        """
        Calcula velocidades lineal y angular basadas en tracking_info.
        
        Args:
            tracking_info: Dict con info del tracker o None si no hay robot
            img_width: Ancho de la imagen
            img_height: Alto de la imagen
            
        Returns:
            tuple: (vx, wz, status)
        """
        if tracking_info is None:
            # Sin robot trackeado
            current_time = time.time()
            
            if self.time_lost_tracking is None:
                # Recién perdimos el tracking
                self.time_lost_tracking = current_time
                return 0.0, 0.0, "ESPERANDO... ⏳"
            
            # Calcular tiempo sin tracking
            time_without_robot = current_time - self.time_lost_tracking
            
            if time_without_robot < self.search_delay:
                # Aún en periodo de espera
                remaining = self.search_delay - time_without_robot
                return 0.0, 0.0, f"ESPERANDO {remaining:.1f}s ⏳"
            else:
                # Ya pasó el delay, empezar a buscar
                return 0.0, SEARCH_ANGULAR_VEL, "BUSCANDO 🔄"
        
        # Robot detectado/trackeado - resetear timer
        self.time_lost_tracking = None
        
        # Extraer información del tracking
        bbox = tracking_info['bbox']
        center = tracking_info['center']
        cx = center['x']
        cy = center['y']
        
        # Calcular dimensiones del bbox
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        area = width * height
        
        # Calcular error horizontal (centrado)
        frame_center_x = img_width / 2
        error_x = (cx - frame_center_x) / frame_center_x
        
        # Calcular error de distancia (área)
        # Si area < target_area: robot está lejos, debe avanzar (vx > 0)
        # Si area >= target_area: robot está cerca, debe detenerse (vx = 0)
        target_area = img_width * img_height * TARGET_AREA_RATIO
        
        # Prevenir división por cero
        if target_area <= 0:
            target_area = 1.0
        
        area_ratio = area / target_area
        
        # Debug: mostrar errores calculados
        print(f"[CONTROL] error_x={error_x:.3f}, area_ratio={area_ratio:.3f}, area={area}, target={target_area:.0f}")
        
        # Control proporcional
        wz = np.clip(KP_ANGULAR * error_x, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        
        # vx proporcional a qué tan lejos está el robot (solo valores positivos)
        if area_ratio < 1.0:
            # Robot lejos, avanzar (más rápido si está muy lejos)
            distance_error = 1.0 - area_ratio  # Siempre positivo cuando area_ratio < 1.0
            vx_raw = KP_LINEAR * distance_error
            vx = np.clip(vx_raw, 0.0, MAX_LINEAR_VEL)
            print(f"[CONTROL] Robot lejos: distance_error={distance_error:.3f}, vx_raw={vx_raw:.3f}")
        else:
            # Robot cerca o muy cerca, detenerse
            vx = 0.0
            print(f"[CONTROL] Robot cerca: area_ratio={area_ratio:.3f} >= 1.0, deteniendo")
        
        print(f"[CONTROL] Calculado FINAL: vx={vx:.3f}, wz={wz:.3f}")
        
        # Determinar estado
        if abs(error_x) < STOP_THRESHOLD and abs(1.0 - area_ratio) < STOP_THRESHOLD:
            vx = 0.0
            wz = 0.0
            status = "EN POSICIÓN ✅"
        else:
            # Mostrar fuente del tracking
            source_emoji = "🎯" if tracking_info['source'] == "yolo" else "📡"
            status = f"TRACKING {source_emoji}"
        
        return vx, wz, status
    
    def execute_control(self, vx, wz):
        """Envía comandos de control al robot"""
        # Siempre enviar comandos (incluso 0,0 para detener)
        print(f"[CONTROLLER] Ejecutando control: vx={vx:.3f}, wz={wz:.3f}")
        self.sender.send_cmd(vx, wz)


def draw_control_hud(img, tracking_info, vx, wz, status):
    """Dibuja HUD minimalista con información de control"""
    h, w = img.shape[:2]
    
    # Línea de centro vertical
    cv2.line(img, (w//2, 0), (w//2, h), (100, 100, 100), 1)
    
    # HUD minimalista en la esquina
    y_offset = 30
    cv2.putText(img, f"vx: {vx:+.2f} m/s", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(img, f"wz: {wz:+.2f} rad/s", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Estado del sistema
    if "TRACKING" in status:
        color = (0, 255, 0)  # Verde
    elif "POSICIÓN" in status:
        color = (0, 255, 255)  # Amarillo
    elif "BUSCANDO" in status:
        color = (0, 165, 255)  # Naranja
    else:  # ESPERANDO
        color = (200, 200, 200)  # Gris
    
    cv2.putText(img, status, (10, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
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


def handle_img(parts, tracker, controller):
    """
    Procesa imagen con tracker híbrido y controla el movimiento.
    
    Args:
        parts: lista de strings del mensaje IMG
        tracker: HybridRobotTracker para tracking
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

        # ===== TRACKING HÍBRIDO =====
        annotated_frame, tracking_info = tracker.process_frame(img)
        
        # ===== CALCULAR CONTROL =====
        vx, wz, status = controller.calculate_control(tracking_info, w, h)
        
        # ===== EJECUTAR CONTROL =====
        controller.execute_control(vx, wz)
        
        # ===== LOG =====
        if tracking_info:
            conf = tracking_info['confidence']
            source = tracking_info['source']
            print(f"[IMG] 🤖 Robot tracked [{source}] conf={conf:.2f} | "
                  f"vx={vx:.2f} wz={wz:.2f} | {status}")
        else:
            print(f"[IMG] ❌ Sin tracking | {status}")
        
        # ===== VISUALIZACIÓN =====
        annotated_frame = draw_control_hud(annotated_frame, tracking_info, vx, wz, status)
        cv2.imshow(f"🤖 Robot Tracker - {robot_name}", annotated_frame)
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
    print("🤖 ROBOT TRACKER - Sistema de Persecución con Tracking Híbrido")
    print("="*70)
    
    # ===== INICIALIZAR TRACKER =====
    print(f"\n[MAIN] Inicializando Hybrid Robot Tracker...")
    try:
        tracker = HybridRobotTracker(
            target_labels=["robots"],
            conf_threshold_initial=0.7,
            conf_threshold_redetect=0.5,
            yolo_refresh_every=5,
            timeout_seconds=5.0,
            fps=30.0,
            imgsz=416  # Más rápido que 640, suficiente para robots
        )
        print(f"✅ [MAIN] Tracker inicializado (usando MPS si disponible)")
    except Exception as e:
        print(f"❌ [MAIN] Error inicializando tracker: {e}")
        return
    
    # ===== INICIALIZAR SENDER =====
    try:
        sender = Sender()
    except Exception as e:
        print(f"❌ [MAIN] Error inicializando Sender: {e}")
        return
    
    # ===== INICIALIZAR CONTROLLER =====
    controller = RobotController(sender, search_delay=1.5)
    print("✅ [MAIN] Sistema de control inicializado (delay: 1.5s)")
    
    # ===== CONFIGURAR SOCKET TELEMETRÍA =====
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_addr = (ROBOT_IP, ROBOT_PORT_TELEMETRY)

    # ===== HANDSHAKE =====
    do_handshake(sock, robot_addr)

    print("\n" + "="*70)
    print("🟢 Sistema activo - Tracking híbrido con delay de 1.5s")
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
                handle_img(parts, tracker, controller)
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