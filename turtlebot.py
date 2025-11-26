from email.mime import image
from enum import Enum
from turtlebot.receiver import Receiver
import cv2
from models.model import Model
from turtlebot.config import config
from turtlebot.controller import Controller
from turtlebot.sender import Sender

class State(Enum):
    SCAN = 0
    TRACK = 1
    PREDICT = 2

class Turtlebot:
    def __init__(self):
        self.state = State.SCAN
        self.receiver = Receiver()
        self.model = Model("", config.WEIGHTS_MODEL_PATH, config.BASE_MODEL_PATH)
        self.controller = Controller()
        self.sender = Sender()

    def recv(self):
        response = self.receiver.recv()
        return response
        
    def update(self):
        while True:
            print("[TURTLEBOT] Estado actual:", self.state)
            res = self.recv()
            if res is None:
                continue

            image = res.get("image", None)
            if image is None:
                continue

            print("[TURTLEBOT] Imagen recibida con forma:", image.shape)

            # Detect
            img, box = self.model.predict(image, show=False)
            cv2.imshow("Deteccion YOLO", image)
            cv2.waitKey(1)

            if box is not None:
                # Puppy detectado → TRACK
                self.state = State.TRACK
                vel_x, vel_w = self.controller.compute_movement(box, img.shape)
                print(f"[TRACK] moviendo: v={vel_x:.2f}, w={vel_w:.2f}")
                self.sender.send(vel_x, vel_w)
            else:
                # No se ve el Puppy → SCAN
                self.state = State.SCAN
                print("[SCAN] No detectado, girando buscando...")
                self.sender.send(0.0, 0.8)   # giro suave buscando

