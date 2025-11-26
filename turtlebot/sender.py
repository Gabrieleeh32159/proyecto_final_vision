import socket, struct
from turtlebot.config import config

class Sender:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        print(f"[SENDER] UDP inicializado hacia {config.ROBOT_IP}:{config.SEND_PORT}")

    def send(self, vel: float, angular_vel: float):
        # Clamp similar al teleop original (ajusta si quieres)
        v = max(min(vel,  2.0), -2.0)   # m/s
        w = max(min(angular_vel, 6.0), -6.0)  # rad/s

        payload = struct.pack('ff', v, w)
        try:
            self.sock.sendto(payload, (config.ROBOT_IP, config.SEND_PORT))
            print(f"[SENDER] Enviado (v={v:+.3f}, w={w:+.3f})")
        except Exception as e:
            print(f"[SENDER][ERROR] {e}")
