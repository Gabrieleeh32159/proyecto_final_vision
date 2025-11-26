import socket
from turtlebot.config import config
from turtlebot.conexion import do_handshake
from enum import Enum
from turtlebot.handlers.recv_image import handle_img

class MessageType(Enum):
    SCAN = "SCAN"
    IMG = "IMG"

class ReceiverResponse:
    def __init__(self, img):
        self.img = img

class Receiver:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    
        self.robot_addr = (config.ROBOT_IP, config.ROBOT_PORT)
        do_handshake(self.sock, self.robot_addr)
        print("[RECEIVER] Recibiendo telemetría.")

    def recv(self):
        data, addr = self.sock.recvfrom(65535)
        text = data.decode("utf-8", errors="ignore")
        parts = text.split()
        if not parts:
            return

        msg_type = parts[0]

        if msg_type == MessageType.SCAN.value:
            return
        elif msg_type == MessageType.IMG.value:
            img = handle_img(parts)
        else:
            print(f"[MAIN] Mensaje desconocido desde {addr}: '{msg_type}'")
        if img is not None:
            return {
                "image": img
            }
        return {}