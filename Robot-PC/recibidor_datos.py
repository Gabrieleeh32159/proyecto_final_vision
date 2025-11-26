#!/usr/bin/env python3
"""
🤖 Robot Follower - Sistema Modular
Receiver: recibe cámara y detecta robots con YOLO
Sender: envía comandos de movimiento
"""

import socket
import base64
import time
import numpy as np
import cv2
from ultralytics import YOLO
import threading

# ========= CONFIGURACIÓN =========
ROBOT_IP = "10.182.184.103"
ROBOT_PORT_TELEMETRY = 6000    # Recibir imágenes
ROBOT_PORT_CMD = 5007          # Enviar comandos

DESIRED_DOMAIN_ID = 10
PAIRING_CODE = "ROBOT_A_100"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_10"

MODEL_PATH = "../best.pt"

# Parámetros de control
MAX_LINEAR_VEL = 0.26
MAX_ANGULAR_VEL = 1.0
KP_ANGULAR = 2.5
KP_LINEAR = 0.8
MIN_CONFIDENCE = 0.5
TARGET_AREA_RATIO = 0.15
STOP_THRESHOLD = 0.05

# Búsqueda girando
SEARCH_ANGULAR_VEL = 0.5  # Gira a esta velocidad cuando no hay robot


# ========= HELPER FUNCTIONS =========
def do_handshake(sock, robot_addr):
    """Realiza handshake con el robot"""
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
                
                try:
                    domain_id = int(domain_str)
                except ValueError:
                    continue
                
                if domain_id == DESIRED_DOMAIN_ID and robot_name == EXPECTED_ROBOT_NAME:
                    print(f"✅ [HANDSHAKE] Emparejado con '{robot_name}'")
                    sock.settimeout(None)
                    return True
        
        except socket.timeout:
            pass
        except KeyboardInterrupt:
            raise
    
    return False


# ========= RECEIVER =========
class Receiver:
    """Recibe imágenes del robot"""
    
    def __init__(self, model_path):
        print(f"🔄 Cargando modelo YOLO desde {model_path}...")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Modelo cargado. Clases: {self.model.names}")
            # Verificar que la clase 'robots' existe
            if 'robots' not in self.model.names.values():
                print(f"⚠️  Advertencia: Clase 'robots' no encontrada en el modelo")
                print(f"Clases disponibles: {list(self.model.names.values())}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model = None
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        robot_addr = (ROBOT_IP, ROBOT_PORT_TELEMETRY)
        
        print("\n📡 Realizando handshake con telemetría...")
        if not do_handshake(self.sock, robot_addr):
            raise Exception("Handshake fallido")
        
        print("[RECEIVER] Listo para recibir telemetría")
    
    def recv_frame(self):
        """Recibe un frame de cámara"""
        try:
            data, addr = self.sock.recvfrom(65535)
            text = data.decode("utf-8", errors="ignore")
            parts = text.split()
            
            if not parts or parts[0] != "IMG":
                return None
            
            if len(parts) < 6:
                return None
            
            try:
                b64_str = " ".join(parts[5:])
                jpeg_bytes = base64.b64decode(b64_str)
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
            except Exception as e:
                print(f"Error decodificando: {e}")
                return None
        
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error recibiendo: {e}")
            return None
    
    def detect_robots(self, img):
        """Detecta robots en la imagen usando YOLO"""
        if self.model is None or img is None:
            return []
        
        try:
            results = self.model(img, conf=MIN_CONFIDENCE, verbose=False)
            detections = []
            
            for r in results:
                for box in r.boxes:
                    # Obtener clase detectada
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    # Filtrar solo la clase 'robots'
                    if cls_name != 'robots':
                        continue
                    
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    area = width * height
                    
                    detection = {
                        'conf': conf,
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (cx, cy),
                        'area': area,
                        'width': width,
                        'height': height,
                        'class': cls_name
                    }
                    detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"Error en detección: {e}")
            return []
    
    def draw_detections(self, img, detections, vx, wz, status):
        """Dibuja detecciones y estado en la imagen"""
        h, w = img.shape[:2]
        
        # Dibujar detecciones
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = det['center']
            conf = det['conf']
            
            # Bounding box verde con mayor grosor
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Centro del robot
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
            # Etiqueta con clase y confianza
            cv2.putText(img, f"ROBOT {conf:.0%}", (x1, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Líneas de referencia (centro de la imagen)
        cv2.line(img, (w//2, 0), (w//2, h), (255, 0, 0), 2)
        cv2.line(img, (0, h//2), (w, h//2), (255, 0, 0), 2)
        
        # Información
        cv2.putText(img, f"Robots: {len(detections)}", (15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(img, f"vx={vx:.3f} m/s", (15, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"wz={wz:.3f} rad/s", (15, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Estado con colores
        if status == "SIGUIENDO":
            color = (0, 255, 255)
        elif status == "EN_OBJETIVO":
            color = (0, 255, 0)
        else:  # BUSCANDO
            color = (0, 165, 255)
        
        cv2.putText(img, f"ESTADO: {status}", (15, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return img
    
    def close(self):
        self.sock.close()


# ========= SENDER =========
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


# ========= CONTROLLER =========
class RobotFollower:
    """Lógica principal de seguimiento de robots"""
    
    def __init__(self, receiver, sender):
        self.receiver = receiver
        self.sender = sender
        self.last_cmd_time = time.time()
    
    def process_frame(self, img):
        """Procesa un frame y retorna comando + status"""
        
        # Detectar robots con YOLO
        detections = self.receiver.detect_robots(img)
        
        h, w = img.shape[:2]
        
        if not detections:
            # ❌ No hay robot: BUSCAR GIRANDO
            vx = 0.0
            wz = SEARCH_ANGULAR_VEL  # Girar en busca
            status = "BUSCANDO 🔄"
            print(f"❌ Sin robot detectado - GIRANDO para buscar")
        
        else:
            # ✅ Robot detectado: GIRAR Y AVANZAR HACIA ÉL
            best_robot = max(detections, key=lambda d: d['area'])
            cx, cy = best_robot['center']
            area = best_robot['area']
            
            # Error angular (horizontal)
            frame_center_x = w / 2
            error_x = (cx - frame_center_x) / frame_center_x
            
            # Error de distancia (invertido: si está lejos avanza, si está cerca retrocede)
            max_area = w * h * TARGET_AREA_RATIO
            error_area = (max_area - area) / max_area  # Invertido para lógica correcta
            
            # Control PID: girar para centrar y avanzar/retroceder según distancia
            wz = np.clip(KP_ANGULAR * error_x, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
            vx = np.clip(KP_LINEAR * error_area, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
            
            # Check si está en objetivo
            if abs(error_x) < STOP_THRESHOLD and abs(error_area) < STOP_THRESHOLD:
                vx = 0.0
                wz = 0.0
                status = "EN OBJETIVO ✅"
                print(f"✅ Robot en posición objetivo")
            else:
                status = "SIGUIENDO 🤖"
                print(f"🔄 SIGUIENDO ROBOT | error_x={error_x:.2f} error_area={error_area:.2f} | "
                      f"vx={vx:.3f} m/s | wz={wz:.3f} rad/s")
        
        # Enviar comando
        self.sender.send_cmd(vx, wz)
        
        # Dibujar en imagen
        img_annotated = self.receiver.draw_detections(img, detections, vx, wz, status)
        
        return img_annotated, status
    
    def run(self):
        """Loop principal"""
        print("\n🟢 Sistema iniciado - Siguiendo robots...")
        print("=" * 70)
        print("Presiona 'q' para salir")
        print("=" * 70 + "\n")
        
        try:
            while True:
                img = self.receiver.recv_frame()
                
                if img is None:
                    continue
                
                img_annotated, status = self.process_frame(img)
                
                cv2.imshow("🤖 Robot Follower", img_annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Saliendo...")
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupción del usuario")
        
        finally:
            print("\n🛑 Deteniendo robot...")
            self.sender.stop()
            time.sleep(0.5)
            self.sender.close()
            self.receiver.close()
            cv2.destroyAllWindows()
            print("✅ Sistema cerrado")


# ========= MAIN =========
def main():
    try:
        # Inicializar componentes
        receiver = Receiver(MODEL_PATH)
        sender = Sender()
        
        # Crear follower
        follower = RobotFollower(receiver, sender)
        
        # Ejecutar
        follower.run()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()