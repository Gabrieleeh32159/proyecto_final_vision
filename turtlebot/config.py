import os
from dotenv import load_dotenv

load_dotenv()
    
class Config:
    # ========= Configuración =========
    ROBOT_IP   = "10.153.100.101"  # IP del TurtleBot4
    ROBOT_PORT = 6000              # Debe coincidir con el nodo de telemetría
    SEND_PORT = 5007

    DESIRED_DOMAIN_ID = 4          # Debe coincidir con ROS_DOMAIN_ID del robot
    PAIRING_CODE      = "ROBOT_A_42"
    EXPECTED_ROBOT_NAME = "turtlebot4_lite_1"  # por seguridad extra

    DATASET_PATH: str = os.getenv("DATASET_PATH", "<default_path>")
    WEIGHTS_MODEL_PATH: str = os.getenv("WEIGHTS_MODEL_PATH", "<default_path>")
    BASE_MODEL_PATH: str = "best.pt"

config = Config()