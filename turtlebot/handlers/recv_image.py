import base64
import numpy as np
import cv2

def handle_img(parts):
    """
    parts: lista de strings del mensaje:
    IMG <domain_id> <robot_name> <sec> <nsec> <base64_jpeg>
    Como base64 puede tener espacios si algo raro pasa, juntamos desde índice 5.
    """
    if len(parts) < 6:
        print("[IMG] Mensaje demasiado corto.")
        return

    try:
        domain_id = int(parts[1])
        robot_name = parts[2]
        sec = int(parts[3])
        nsec = int(parts[4])

        b64_str = " ".join(parts[5:])  # el resto del mensaje
        jpeg_bytes = base64.b64decode(b64_str)

        # Decodificar JPEG a imagen OpenCV
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[IMG] Error al decodificar imagen.")
            return

        # Mostrar con el mismo estilo del código antiguo
        cv2.imshow("Camara (JPEG)", img)
        cv2.waitKey(1)

    except Exception as e:
        print(f"[IMG] Error manejando imagen: {e}")