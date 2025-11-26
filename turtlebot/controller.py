class Controller:
    def __init__(self):
        # Ganancias (ajustables)
        self.K_ang = 1.2
        self.max_w = 1.5  # velocidad angular máxima
        self.max_v = 0.8  # velocidad lineal máxima

    def compute_movement(self, box, image_shape):
        if box is None:
            return 0.0, 0.0

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2         # centro del Puppy
        w = x2 - x1                 # ancho del bounding box

        frame_center = image_shape[1] // 2
        error_x = cx - frame_center

        # Control proporcional
        angular_z = -(error_x / frame_center) * self.K_ang
        angular_z = max(-self.max_w, min(self.max_w, angular_z))

        # Avance proporcional al tamaño del Puppy (si está cerca, baja velocidad)
        closeness = w / image_shape[1]          # 0.0 lejos — 1.0 cerca
        linear_x = (1.0 - closeness) * self.max_v
        linear_x = max(0.0, min(self.max_v, linear_x))

        return linear_x, angular_z
