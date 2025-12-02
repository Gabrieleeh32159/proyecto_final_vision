"""
Robot Tracker v3.0 - Optimizado para MacBook M3 con MPS
Usa únicamente el tracker nativo de YOLO (ByteTrack) sin Kalman
"""

import socket
import struct
import base64
import time
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
from ultralytics import YOLO

# ========= Configuración de Conexión =========
DESIRED_DOMAIN_ID = 10
PAIRING_CODE = "ROBOT_A_100"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_10"

MODEL_PATH = "../best.pt"
TRACKER_CONFIG = os.path.join(os.path.dirname(__file__), "tracker_config.yaml")

ROBOT_IP = "192.168.0.220"
ROBOT_PORT_TELEMETRY = 6000
ROBOT_PORT_CMD = 5007


# ========= PARÁMETROS DE CONTROL =========
@dataclass
class ControlConfig:
    """Configuración del controlador"""
    # Velocidades
    max_linear_vel: float = 0.5         # Velocidad lineal al perseguir
    max_angular_vel: float = 2.0        # Velocidad angular máxima
    
    # Ganancias proporcionales
    kp_angular: float = 0.015           # Ganancia para centrado horizontal
    
    # Umbrales
    center_threshold: float = 0.05      # Error X aceptable (5% del ancho de imagen)
    min_confidence: float = 0.7        # Confianza mínima YOLO
    
    # Búsqueda cuando se pierde (gira 360° en una dirección)
    search_angular_vel: float = 0.15    # Velocidad de rotación en búsqueda (más lento)
    lost_frames_threshold: int = 120     # Frames sin detección para activar búsqueda
    
    # Regreso cuando detecta en borde derecho durante búsqueda
    edge_threshold: float = 0.75        # Porcentaje del ancho para considerar "borde derecho"
    return_angle_deg: float = 15.0      # Grados a girar de regreso
    return_angular_vel: float = search_angular_vel   # Velocidad de regreso (igual que búsqueda)


class SimplePController:
    """Controlador proporcional simple"""
    
    def __init__(self, kp: float, output_min: float, output_max: float):
        self.kp = kp
        self.output_min = output_min
        self.output_max = output_max
    
    def compute(self, error: float) -> float:
        """Calcula salida proporcional"""
        output = self.kp * error
        output = np.clip(output, self.output_min, self.output_max)
        return output


class Sender:
    """Envía comandos de movimiento al robot"""
    
    def __init__(self, robot_ip: str, robot_port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_addr = (robot_ip, robot_port)
        print(f"📡 [SENDER] Configurado para {self.robot_addr}")
    
    def send_cmd(self, vel: float, angular_vel: float):
        """Envía comando de velocidad
        
        Args:
            vel: Velocidad lineal (m/s)
            angular_vel: Velocidad angular (rad/s)
        """
        v = max(min(vel, 2.0), -0.5)  # Permitir pequeño retroceso
        w = max(min(angular_vel, 2.0), -2.0)
        
        # El protocolo espera: primero lineal, luego angular
        payload = struct.pack('ff', v, w)
        try:
            self.sock.sendto(payload, self.robot_addr)
            # Debug: descomentar para ver qué se envía
            # print(f"[CMD] v={v:.2f} w={w:.2f}")
        except Exception as e:
            print(f"[SENDER][ERROR] {e}")
    
    def stop(self):
        self.send_cmd(0.0, 0.0)
    
    def close(self):
        self.stop()
        self.sock.close()


class RobotTrackerV3:
    """
    Sistema de tracking simplificado para MacBook M3 con MPS
    Usa únicamente el tracker nativo de YOLO (ByteTrack)
    """
    
    def __init__(self, model_path: str, tracker_config: str, control_config: ControlConfig):
        self.control_config = control_config
        
        # Cargar modelo YOLO con MPS
        self.device = "mps"  # Forzar MPS para MacBook M3
        print(f"🖥️  Dispositivo: {self.device}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"✅ Modelo cargado: {model_path}")
        print(f"   Clases: {list(self.model.names.values())}")
        
        # Verificar archivo de configuración de tracker
        if os.path.exists(tracker_config):
            self.tracker_config = tracker_config
            print(f"✅ Tracker config: {tracker_config}")
        else:
            self.tracker_config = "bytetrack.yaml"
            print(f"⚠️  Usando tracker por defecto: bytetrack.yaml")
        
        # Controlador angular (solo este se necesita)
        self.controller_angular = SimplePController(
            control_config.kp_angular,
            -control_config.max_angular_vel,
            control_config.max_angular_vel
        )
        
        # Estado de tracking
        self.current_track_id: Optional[int] = None
        self.frames_without_detection = 0
        self.last_detection_time = time.time()
        self.search_direction = 1  # 1 = derecha, -1 = izquierda
        
        # Estado de regreso (cuando detecta en borde derecho durante búsqueda)
        self.was_searching = False
        self.is_returning = False
        self.return_start_time = 0.0
        self.return_duration = 0.0  # Se calcula dinámicamente
        
        # Métricas
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float, str]:
        """
        Procesa un frame y retorna comandos de control
        
        Returns:
            annotated_frame: Frame con anotaciones
            vx: Velocidad lineal
            wz: Velocidad angular
            status: Estado del tracking
        """
        start_time = time.time()
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # Ejecutar tracking con YOLO nativo
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.control_config.min_confidence,
            verbose=False,
            imgsz=640,
            device=self.device
        )
        
        # Procesar resultados
        annotated_frame = frame.copy()
        detection = self._extract_best_detection(results[0])
        
        # Lógica de estados
        if self.is_returning:
            # Estado: REGRESANDO (girando de vuelta)
            vx, wz, status = self._return_mode()
            
        elif detection is not None:
            # Tenemos detección
            cx, cy, bw, bh, conf, track_id = detection
            
            # Verificar si estaba buscando y detectó en borde derecho
            edge_x = w * self.control_config.edge_threshold
            if self.was_searching and cx > edge_x:
                # Detectó en borde derecho mientras buscaba -> activar regreso
                self._start_return()
                vx, wz, status = self._return_mode()
                print(f"[EDGE DETECT] cx={cx:.0f} > edge={edge_x:.0f} -> Regresando")
            else:
                # Detección normal -> perseguir
                self.current_track_id = track_id
                self.frames_without_detection = 0
                self.last_detection_time = time.time()
                self.was_searching = False
                
                vx, wz, status = self._calculate_control(cx, cy, bw, bh, w, h)
                
                # Debug
                print(f"[DETECTED] ID:{track_id} conf:{conf:.2f} -> vx={vx:.3f}, wz={wz:.3f} | {status}")
            
            # Dibujar detección
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            x2, y2 = int(cx + bw/2), int(cy + bh/2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(annotated_frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
            cv2.putText(annotated_frame, f"ID:{track_id} {conf:.0%}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Sin detección - modo búsqueda
            self.frames_without_detection += 1
            self.was_searching = True
            vx, wz, status = self._search_mode()
        
        # Dibujar HUD
        annotated_frame = self._draw_hud(annotated_frame, vx, wz, status, detection is not None)
        
        # Calcular FPS
        elapsed = time.time() - start_time
        self.fps_counter.append(elapsed)
        
        return annotated_frame, vx, wz, status
    
    def _extract_best_detection(self, result) -> Optional[Tuple]:
        """Extrae la mejor detección del resultado de YOLO"""
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        boxes = result.boxes
        
        # Si tenemos IDs de tracking, buscar nuestro track actual
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().numpy()
            
            # Si ya tenemos un track_id, priorizarlo
            if self.current_track_id is not None and self.current_track_id in track_ids:
                idx = np.where(track_ids == self.current_track_id)[0][0]
            else:
                # Tomar el de mayor confianza
                confs = boxes.conf.cpu().numpy()
                idx = np.argmax(confs)
            
            box = boxes.xywh[idx].cpu().numpy()
            cx, cy, bw, bh = box
            conf = float(boxes.conf[idx])
            track_id = int(track_ids[idx])
            
            return cx, cy, bw, bh, conf, track_id
        
        # Sin tracking IDs - usar la mejor confianza
        confs = boxes.conf.cpu().numpy()
        idx = np.argmax(confs)
        box = boxes.xywh[idx].cpu().numpy()
        cx, cy, bw, bh = box
        conf = float(confs[idx])
        
        return cx, cy, bw, bh, conf, -1
    
    def _calculate_control(self, cx: float, cy: float, bw: float, bh: float,
                          img_w: int, img_h: int) -> Tuple[float, float, str]:
        """Calcula comandos de control: velocidad lineal máxima + control angular"""
        cfg = self.control_config
        
        # Error horizontal en píxeles (positivo = perro a la derecha, negativo = a la izquierda)
        error_x = cx - img_w / 2
        
        # Threshold de centrado dinámico (porcentaje del ancho de imagen)
        center_threshold_px = img_w * cfg.center_threshold
        
        # SIEMPRE avanzar a velocidad máxima cuando detecta
        vx = cfg.max_linear_vel
        
        # Control angular (proporcional) - girar hacia donde está el perro
        # Si error_x > 0 (perro a la derecha) -> wz < 0 (girar derecha)
        # Si error_x < 0 (perro a la izquierda) -> wz > 0 (girar izquierda)
        wz = -self.controller_angular.compute(error_x)
        
        # Si está muy centrado, reducir giro
        if abs(error_x) < center_threshold_px:
            wz *= 0.2
            status = "✅ CENTRADO - PERSIGUIENDO"
        else:
            status = "🎯 SIGUIENDO PERRO"
        
        # Debug
        print(f"[CONTROL] error_x={error_x:.1f}px -> vx={vx:.2f}, wz={wz:.3f}")
        
        return vx, wz, status
    
    def _search_mode(self) -> Tuple[float, float, str]:
        """Modo búsqueda: gira continuamente en una dirección (360°) hasta encontrar el perro"""
        cfg = self.control_config
        
        # Activar búsqueda solo después de threshold de frames perdidos
        if self.frames_without_detection < cfg.lost_frames_threshold:
            # Aún no activar búsqueda, solo detener
            return 0.0, 0.0, "⏸️ ESPERANDO..."
        
        time_lost = time.time() - self.last_detection_time
        
        # Girar siempre en la misma dirección (sin cambiar)
        # search_direction se mantiene constante para hacer giro 360°
        wz = self.search_direction * cfg.search_angular_vel
        vx = 0.0  # NO avanzar mientras busca
        
        print(f"[SEARCH] Girando 360°: vx={vx:.2f}, wz={wz:.2f}")
        
        return vx, wz, f"🔄 BUSCANDO 360° [{time_lost:.1f}s]"
    
    def _start_return(self):
        """Inicia el modo de regreso"""
        cfg = self.control_config
        self.is_returning = True
        self.return_start_time = time.time()
        # Calcular duración: tiempo = ángulo (rad) / velocidad (rad/s)
        angle_rad = np.radians(cfg.return_angle_deg)
        self.return_duration = angle_rad / cfg.return_angular_vel
        print(f"[RETURN] Iniciando regreso: {cfg.return_angle_deg}° en {self.return_duration:.2f}s")
    
    def _return_mode(self) -> Tuple[float, float, str]:
        """Modo regreso: gira en dirección opuesta para compensar overshoot"""
        cfg = self.control_config
        
        elapsed = time.time() - self.return_start_time
        
        if elapsed >= self.return_duration:
            # Terminó el regreso -> detenerse y evaluar
            self.is_returning = False
            self.was_searching = False
            self.frames_without_detection = 0  # Resetear para dar oportunidad de detectar
            print(f"[RETURN] Regreso completado")
            return 0.0, 0.0, "⏸️ EVALUANDO..."
        
        # Girar en dirección opuesta a la búsqueda
        wz = -self.search_direction * cfg.return_angular_vel
        vx = 0.0
        
        remaining = self.return_duration - elapsed
        print(f"[RETURN] Regresando: wz={wz:.2f}, restante={remaining:.2f}s")
        
        return vx, wz, f"REGRESANDO [{remaining:.1f}s]"
    
    def _draw_hud(self, frame: np.ndarray, vx: float, wz: float, 
                  status: str, has_detection: bool) -> np.ndarray:
        """Dibuja información en pantalla"""
        h, w = frame.shape[:2]
        
        # Líneas de centro
        cv2.line(frame, (w//2, 0), (w//2, h), (80, 80, 80), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (80, 80, 80), 1)
        
        # Panel de info
        y = 30
        cv2.putText(frame, f"VEL LINEAL: {vx:+.2f} m/s", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30
        cv2.putText(frame, f"VEL ANGULAR: {wz:+.2f} rad/s", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Indicador de estado
        y += 30
        if has_detection:
            cv2.putText(frame, "PERRO DETECTADO", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SIN DETECCION", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # FPS
        if self.fps_counter:
            avg_time = np.mean(list(self.fps_counter))
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dispositivo MPS
        cv2.putText(frame, "MPS", (w - 60, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        
        # Estado principal
        color = (0, 255, 0) if "OBJETIVO" in status or "SIGUIENDO" in status else (0, 165, 255)
        if "BUSCANDO" in status:
            color = (0, 0, 255)
        cv2.putText(frame, status, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame
    
    def reset(self):
        """Resetea el estado del tracker"""
        self.current_track_id = None
        self.frames_without_detection = 0


def do_handshake(sock: socket.socket, robot_addr: Tuple[str, int]):
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

                if domain_id != DESIRED_DOMAIN_ID:
                    continue

                if robot_name != EXPECTED_ROBOT_NAME:
                    continue

                print(f"[HANDSHAKE] ✅ Emparejado con '{robot_name}'")
                sock.settimeout(None)
                return

        except socket.timeout:
            print("[HANDSHAKE] Esperando ACK...")
        except KeyboardInterrupt:
            raise


def main():
    print("\n" + "="*70)
    print("🤖 ROBOT TRACKER v3.0 - Optimizado para MacBook M3 con MPS")
    print("="*70)
    print("✅ Tracker nativo de YOLO (ByteTrack)")
    print("✅ Aceleración MPS (Metal Performance Shaders)")
    print("✅ Control proporcional simplificado")
    print("✅ Rotación de búsqueda automática")
    print("="*70 + "\n")
    
    # Configuración
    control_config = ControlConfig()
    
    # Inicializar componentes
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "best.pt")
        tracker = RobotTrackerV3(model_path, TRACKER_CONFIG, control_config)
    except Exception as e:
        print(f"❌ Error inicializando tracker: {e}")
        import traceback
        traceback.print_exc()
        return
    
    sender = Sender(ROBOT_IP, ROBOT_PORT_CMD)
    
    # Conexión UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_addr = (ROBOT_IP, ROBOT_PORT_TELEMETRY)
    
    do_handshake(sock, robot_addr)
    
    print("\n" + "="*70)
    print("🟢 SISTEMA ACTIVO - Presiona 'q' para salir")
    print("="*70 + "\n")
    
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            text = data.decode("utf-8", errors="ignore")
            parts = text.split()

            if not parts:
                continue

            if parts[0] == "IMG":
                try:
                    # Decodificar imagen
                    b64_str = " ".join(parts[5:])
                    jpeg_bytes = base64.b64decode(b64_str)
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if img is None:
                        continue

                    # Procesar frame
                    annotated, vx, wz, status = tracker.process_frame(img)
                    
                    # Debug: ver valores enviados
                    if "BUSCANDO" in status or "ESPERANDO" in status:
                        print(f"[DEBUG] Estado: {status} -> vx={vx:.3f}, wz={wz:.3f}")
                    
                    # Enviar comando
                    sender.send_cmd(vx, wz)
                    
                    # Mostrar
                    cv2.imshow("Robot Tracker v3.0 - MPS", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    print(f"[ERROR] {e}")
                    import traceback
                    traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupción del usuario...")
    finally:
        print("🛑 Deteniendo robot...")
        sender.stop()
        time.sleep(0.3)
        sender.close()
        sock.close()
        cv2.destroyAllWindows()
        print("✅ Sistema cerrado\n")


if __name__ == "__main__":
    main()
