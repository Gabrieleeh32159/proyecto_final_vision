"""
Robot Tracker v2.0 - Optimizado para MacBook Air M3
Usa ByteTrack integrado en YOLO + Kalman Filter mejorado para predicción
"""

import socket
import struct
import base64
import time
import sys
import os
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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


# ========= PARÁMETROS DE CONTROL PID =========
@dataclass
class ControlConfig:
    """Configuración del controlador PID"""
    # Velocidades máximas
    max_linear_vel: float = 2.0
    max_angular_vel: float = 2.0
    
    # PID Angular (para centrar horizontalmente)
    kp_angular: float = 2.5
    ki_angular: float = 0.05
    kd_angular: float = 0.8
    
    # PID Lineal (para mantener distancia)
    kp_linear: float = 2.5
    ki_linear: float = 0.02
    kd_linear: float = 0.5
    
    # Área objetivo (ratio del frame)
    target_area_ratio: float = 0.12
    
    # Umbrales
    center_threshold: float = 0.05      # Error X aceptable
    distance_threshold: float = 0.08    # Error de área aceptable
    min_confidence: float = 0.5         # Confianza mínima YOLO
    
    # Búsqueda cuando se pierde
    search_angular_vel: float = 0.6
    search_linear_vel: float = 0.3
    search_timeout: float = 8.0
    
    # Suavizado de comandos
    command_smoothing: float = 0.3      # 0 = sin suavizado, 1 = máximo suavizado


# ========= PARÁMETROS KALMAN MEJORADO =========
@dataclass  
class KalmanConfig:
    """Configuración del filtro Kalman"""
    process_noise_pos: float = 0.1      # Ruido de proceso para posición
    process_noise_vel: float = 0.5      # Ruido de proceso para velocidad
    process_noise_size: float = 0.05    # Ruido de proceso para tamaño
    measurement_noise: float = 0.5      # Ruido de medición
    max_frames_predict: int = 45        # Máximo frames de predicción pura (1.5s a 30fps)


class AdaptiveKalmanFilter:
    """
    Filtro Kalman mejorado con estado [cx, cy, vx, vy, w, h]
    Adaptativo: ajusta ruido según velocidad del objeto
    """
    
    def __init__(self, img_width: int, img_height: int, config: KalmanConfig):
        self.img_width = img_width
        self.img_height = img_height
        self.config = config
        
        # Estado: [cx, cy, vx, vy, w, h]
        self.dim_x = 6  # Dimensión del estado
        self.dim_z = 4  # Dimensión de medición [cx, cy, w, h]
        
        # Estado inicial
        self.x = np.array([
            [img_width / 2],   # cx
            [img_height / 2],  # cy
            [0.0],             # vx
            [0.0],             # vy
            [100.0],           # w
            [100.0]            # h
        ], dtype=np.float64)
        
        # Covarianza inicial
        self.P = np.eye(self.dim_x) * 100.0
        
        # Matriz de transición de estado (dt = 1 frame)
        self.F = np.array([
            [1, 0, 1, 0, 0, 0],  # cx = cx + vx
            [0, 1, 0, 1, 0, 0],  # cy = cy + vy
            [0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 1, 0],  # w = w
            [0, 0, 0, 0, 0, 1]   # h = h
        ], dtype=np.float64)
        
        # Matriz de observación
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Observamos cx
            [0, 1, 0, 0, 0, 0],  # Observamos cy
            [0, 0, 0, 0, 1, 0],  # Observamos w
            [0, 0, 0, 0, 0, 1]   # Observamos h
        ], dtype=np.float64)
        
        # Ruido de proceso (se ajusta adaptativamente)
        self._update_process_noise()
        
        # Ruido de medición
        self.R = np.eye(self.dim_z) * config.measurement_noise
        
        # Tracking de frames sin detección
        self.frames_without_detection = 0
        self.is_initialized = False
        self.velocity_history = deque(maxlen=10)
    
    def _update_process_noise(self, velocity_factor: float = 1.0):
        """Actualiza Q según velocidad estimada"""
        q_pos = self.config.process_noise_pos * velocity_factor
        q_vel = self.config.process_noise_vel * velocity_factor
        q_size = self.config.process_noise_size
        
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_size, q_size])
    
    def predict(self) -> Tuple[float, float, float, float]:
        """Predicción del siguiente estado"""
        # Predicción de estado
        self.x = self.F @ self.x
        
        # Predicción de covarianza
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.frames_without_detection += 1
        
        return self._get_position()
    
    def update(self, cx: float, cy: float, w: float, h: float):
        """Actualiza con nueva medición"""
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)
        
        if not self.is_initialized:
            # Primera medición - inicializar estado
            self.x = np.array([[cx], [cy], [0.0], [0.0], [w], [h]], dtype=np.float64)
            self.P = np.eye(self.dim_x) * 10.0
            self.is_initialized = True
            self.frames_without_detection = 0
            return
        
        # Calcular velocidad observada
        old_cx, old_cy = float(self.x[0]), float(self.x[1])
        observed_vx = cx - old_cx
        observed_vy = cy - old_cy
        velocity_magnitude = math.sqrt(observed_vx**2 + observed_vy**2)
        self.velocity_history.append(velocity_magnitude)
        
        # Adaptar ruido de proceso según velocidad
        if len(self.velocity_history) >= 3:
            avg_velocity = np.mean(list(self.velocity_history))
            velocity_factor = 1.0 + avg_velocity / 50.0  # Normalizar
            self._update_process_noise(min(velocity_factor, 3.0))
        
        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Actualización de estado
        y = z - self.H @ self.x  # Innovación
        self.x = self.x + K @ y
        
        # Actualización de covarianza
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
        
        self.frames_without_detection = 0
    
    def _get_position(self) -> Tuple[float, float, float, float]:
        """Retorna posición actual con límites"""
        cx = float(np.clip(self.x[0], 0, self.img_width))
        cy = float(np.clip(self.x[1], 0, self.img_height))
        w = float(max(self.x[4], 10))
        h = float(max(self.x[5], 10))
        return cx, cy, w, h
    
    def get_position(self) -> Tuple[float, float, float, float]:
        """Retorna última posición conocida"""
        return self._get_position()
    
    def get_velocity(self) -> Tuple[float, float]:
        """Retorna velocidad estimada"""
        return float(self.x[2]), float(self.x[3])
    
    def is_prediction_valid(self) -> bool:
        """Verifica si la predicción es confiable"""
        return self.frames_without_detection < self.config.max_frames_predict
    
    def reset(self):
        """Resetea el filtro"""
        self.is_initialized = False
        self.frames_without_detection = 0
        self.velocity_history.clear()
        self.P = np.eye(self.dim_x) * 100.0


class PIDController:
    """Controlador PID con anti-windup y suavizado"""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_min: float, output_max: float,
                 integral_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def compute(self, error: float, dt: float = 1/30) -> float:
        """Calcula salida PID"""
        # Proporcional
        p_term = self.kp * error
        
        # Integral con anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivativo
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        
        self.prev_error = error
        
        # Salida total
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_min, self.output_max)
        
        return output
    
    def reset(self):
        """Resetea el controlador"""
        self.integral = 0.0
        self.prev_error = 0.0


class CommandSmoother:
    """Suaviza comandos de velocidad para evitar movimientos bruscos"""
    
    def __init__(self, smoothing_factor: float = 0.3):
        self.smoothing = smoothing_factor
        self.prev_vx = 0.0
        self.prev_wz = 0.0
    
    def smooth(self, vx: float, wz: float) -> Tuple[float, float]:
        """Aplica suavizado exponencial"""
        smooth_vx = self.smoothing * self.prev_vx + (1 - self.smoothing) * vx
        smooth_wz = self.smoothing * self.prev_wz + (1 - self.smoothing) * wz
        
        self.prev_vx = smooth_vx
        self.prev_wz = smooth_wz
        
        return smooth_vx, smooth_wz
    
    def reset(self):
        self.prev_vx = 0.0
        self.prev_wz = 0.0


class Sender:
    """Envía comandos de movimiento al robot"""
    
    def __init__(self, robot_ip: str, robot_port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_addr = (robot_ip, robot_port)
        print(f"📡 [SENDER] Configurado para {self.robot_addr}")
    
    def send_cmd(self, vel: float, angular_vel: float):
        """Envía comando de velocidad"""
        v = max(min(vel, 2.0), -0.5)  # Permitir pequeño retroceso
        w = max(min(angular_vel, 2.0), -2.0)
        
        payload = struct.pack('ff', w, v)
        try:
            self.sock.sendto(payload, self.robot_addr)
        except Exception as e:
            print(f"[SENDER][ERROR] {e}")
    
    def stop(self):
        self.send_cmd(0.0, 0.0)
    
    def close(self):
        self.stop()
        self.sock.close()


class RobotTrackerV2:
    """
    Sistema de tracking optimizado para MacBook M3
    Usa ByteTrack integrado + Kalman adaptativo + Control PID
    """
    
    def __init__(self, model_path: str, tracker_config: str,
                 control_config: ControlConfig, kalman_config: KalmanConfig):
        self.control_config = control_config
        self.kalman_config = kalman_config
        
        # Cargar modelo YOLO
        self.device = self._get_device()
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
        
        # Componentes de control
        self.kalman: Optional[AdaptiveKalmanFilter] = None
        self.pid_angular = PIDController(
            control_config.kp_angular,
            control_config.ki_angular, 
            control_config.kd_angular,
            -control_config.max_angular_vel,
            control_config.max_angular_vel
        )
        self.pid_linear = PIDController(
            control_config.kp_linear,
            control_config.ki_linear,
            control_config.kd_linear,
            -0.3,  # Permitir pequeño retroceso
            control_config.max_linear_vel
        )
        self.smoother = CommandSmoother(control_config.command_smoothing)
        
        # Estado de tracking
        self.current_track_id: Optional[int] = None
        self.frames_without_track = 0
        self.last_detection_time = time.time()
        self.search_direction = 1
        self.is_searching = False
        
        # Métricas
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
    
    def _get_device(self) -> str:
        """Detecta mejor dispositivo disponible"""
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
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
        
        # Inicializar Kalman si es necesario
        if self.kalman is None:
            self.kalman = AdaptiveKalmanFilter(w, h, self.kalman_config)
        
        # Ejecutar tracking con ByteTrack
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.control_config.min_confidence,
            verbose=False,
            imgsz=416,  # Optimizado para velocidad
            half=True if self.device in ["mps", "cuda"] else False
        )
        
        # Procesar resultados
        annotated_frame = frame.copy()
        detection = self._extract_best_detection(results[0], w, h)
        
        if detection is not None:
            # Tenemos detección - actualizar Kalman y calcular control
            cx, cy, bw, bh, conf, track_id = detection
            self.kalman.update(cx, cy, bw, bh)
            self.current_track_id = track_id
            self.frames_without_track = 0
            self.last_detection_time = time.time()
            self.is_searching = False
            
            vx, wz, status = self._calculate_control(cx, cy, bw, bh, w, h, "YOLO")
            
            # Dibujar detección
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            x2, y2 = int(cx + bw/2), int(cy + bh/2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(annotated_frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
            cv2.putText(annotated_frame, f"ID:{track_id} {conf:.0%}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif self.kalman.is_initialized and self.kalman.is_prediction_valid():
            # Sin detección pero Kalman válido - usar predicción
            cx, cy, bw, bh = self.kalman.predict()
            self.frames_without_track += 1
            
            vx, wz, status = self._calculate_control(cx, cy, bw, bh, w, h, "KALMAN")
            
            # Dibujar predicción
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            x2, y2 = int(cx + bw/2), int(cy + bh/2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.circle(annotated_frame, (int(cx), int(cy)), 6, (0, 165, 255), -1)
            
            remaining = self.kalman_config.max_frames_predict - self.frames_without_track
            cv2.putText(annotated_frame, f"PREDICCION [{remaining}f]",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        else:
            # Sin detección y Kalman expirado - modo búsqueda
            vx, wz, status = self._search_mode()
            self.frames_without_track += 1
        
        # Suavizar comandos
        vx, wz = self.smoother.smooth(vx, wz)
        
        # Dibujar HUD
        annotated_frame = self._draw_hud(annotated_frame, vx, wz, status, detection is not None)
        
        # Calcular FPS
        elapsed = time.time() - start_time
        self.fps_counter.append(elapsed)
        
        return annotated_frame, vx, wz, status
    
    def _extract_best_detection(self, result, img_w: int, img_h: int) -> Optional[Tuple]:
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
                          img_w: int, img_h: int, source: str) -> Tuple[float, float, str]:
        """Calcula comandos de control usando PIDs"""
        cfg = self.control_config
        
        # Error horizontal normalizado [-1, 1]
        error_x = (cx - img_w / 2) / (img_w / 2)
        
        # Error de distancia (basado en área)
        current_area = bw * bh
        target_area = img_w * img_h * cfg.target_area_ratio
        area_ratio = current_area / target_area if target_area > 0 else 1.0
        error_distance = 1.0 - area_ratio  # Positivo = muy lejos, negativo = muy cerca
        
        # Control angular (PID)
        wz = self.pid_angular.compute(error_x)
        
        # Control lineal (PID)
        vx = self.pid_linear.compute(error_distance)
        
        # Si está muy centrado, reducir giro
        if abs(error_x) < cfg.center_threshold:
            wz *= 0.3
        
        # Si está a buena distancia, reducir velocidad
        if abs(error_distance) < cfg.distance_threshold:
            vx *= 0.3
        
        # Determinar estado
        if abs(error_x) < cfg.center_threshold and abs(error_distance) < cfg.distance_threshold:
            status = "✅ OBJETIVO LOGRADO"
        elif abs(error_x) < cfg.center_threshold:
            status = f"📍 CENTRADO | dist:{error_distance:+.2f}"
        elif abs(error_distance) < cfg.distance_threshold:
            status = f"📏 DISTANCIA OK | x:{error_x:+.2f}"
        else:
            status = f"🎯 SIGUIENDO [{source}]"
        
        return vx, wz, status
    
    def _search_mode(self) -> Tuple[float, float, str]:
        """Modo búsqueda cuando se pierde el robot"""
        cfg = self.control_config
        
        if not self.is_searching:
            self.is_searching = True
            self.search_direction = 1 if np.random.random() > 0.5 else -1
        
        time_lost = time.time() - self.last_detection_time
        
        # Timeout - resetear
        if time_lost > cfg.search_timeout:
            self.kalman.reset()
            self.pid_angular.reset()
            self.pid_linear.reset()
            self.current_track_id = None
            self.search_direction *= -1
            self.last_detection_time = time.time()
        
        # Patrón de búsqueda: girar + avanzar lento
        wz = self.search_direction * cfg.search_angular_vel
        vx = cfg.search_linear_vel
        
        return vx, wz, f"🔄 BUSCANDO [{time_lost:.1f}s]"
    
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
        
        # Indicador de fuente
        y += 30
        if has_detection:
            cv2.putText(frame, "🟢 YOLO ACTIVO", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif self.kalman and self.kalman.is_prediction_valid():
            cv2.putText(frame, "🟡 KALMAN PREDICIENDO", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "🔴 BUSCANDO", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # FPS
        if self.fps_counter:
            avg_time = np.mean(list(self.fps_counter))
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Estado principal
        color = (0, 255, 0) if "OBJETIVO" in status or "SIGUIENDO" in status else (0, 165, 255)
        if "BUSCANDO" in status:
            color = (0, 0, 255)
        cv2.putText(frame, status, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame
    
    def reset(self):
        """Resetea todos los componentes"""
        if self.kalman:
            self.kalman.reset()
        self.pid_angular.reset()
        self.pid_linear.reset()
        self.smoother.reset()
        self.current_track_id = None
        self.frames_without_track = 0
        self.is_searching = False


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
    print("🤖 ROBOT TRACKER v2.0 - Optimizado para MacBook M3")
    print("="*70)
    print("✅ ByteTrack integrado en YOLO")
    print("✅ Filtro Kalman Adaptativo (6 estados)")
    print("✅ Control PID completo (P+I+D)")
    print("✅ Suavizado de comandos")
    print("✅ Aceleración MPS (Metal)")
    print("="*70 + "\n")
    
    # Configuraciones
    control_config = ControlConfig()
    kalman_config = KalmanConfig()
    
    # Inicializar componentes
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "best.pt")
        tracker = RobotTrackerV2(model_path, TRACKER_CONFIG, control_config, kalman_config)
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
                    
                    # Enviar comando
                    sender.send_cmd(vx, wz)
                    
                    # Mostrar
                    cv2.imshow("Robot Tracker v2.0", annotated)
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
