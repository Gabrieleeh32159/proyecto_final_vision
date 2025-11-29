import cv2
import numpy as np

# -------------------------------------
# CONFIGURACIÓN
# -------------------------------------
CAM_INDEX = 0  # cámara por defecto

MIN_AREA = 800         # área mínima del contorno
ASPECT_MIN = 1.0       # relación de aspecto mínima (w/h)
ASPECT_MAX = 5.0       # relación de aspecto máxima
SOLIDITY_MAX = 0.9     # solidez máxima para descartar objetos muy compactos

SEARCH_SIZE = 200      # radio de la ventana de re-búsqueda alrededor de la última posición
SHOW_MASK = True       # mostrar o no la máscara de movimiento

# -------------------------------------
# CREACIÓN DE TRACKER CON FALLBACK
# -------------------------------------
def create_tracker():
    """
    Intenta crear un tracker CSRT.
    Si no existe, prueba legacy.CSRT, luego KCF y luego MOSSE.
    """
    tracker = None
    # CSRT moderno
    try:
        tracker = cv2.TrackerCSRT_create()
        print("[INFO] Usando TrackerCSRT_create()")
        return tracker
    except AttributeError:
        pass

    # CSRT legacy
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
        print("[INFO] Usando legacy.TrackerCSRT_create()")
        return tracker
    except AttributeError:
        pass

    # KCF fallback
    try:
        tracker = cv2.TrackerKCF_create()
        print("[INFO] Usando TrackerKCF_create() (fallback)")
        return tracker
    except AttributeError:
        pass

    try:
        tracker = cv2.legacy.TrackerKCF_create()
        print("[INFO] Usando legacy.TrackerKCF_create() (fallback)")
        return tracker
    except AttributeError:
        pass

    # MOSSE fallback
    try:
        tracker = cv2.TrackerMOSSE_create()
        print("[INFO] Usando TrackerMOSSE_create() (fallback)")
        return tracker
    except AttributeError:
        pass

    try:
        tracker = cv2.legacy.TrackerMOSSE_create()
        print("[INFO] Usando legacy.TrackerMOSSE_create() (fallback)")
        return tracker
    except AttributeError:
        pass

    raise RuntimeError("No se pudo crear ningún tracker (CSRT/KCF/MOSSE). Revisa tu instalación de OpenCV-contrib.")


# -------------------------------------
# DETECCIÓN POR MOVIMIENTO + FORMA
# -------------------------------------
bg = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=500, detectShadows=False)


def detect_puppybot(gray_frame, roi=None):
    """
    Detecta al Puppybot usando:
    - background subtraction
    - filtrado morfológico
    - filtros por área, aspecto y solidez

    Si roi no es None, asume tupla (x1, y1, x2, y2) y recorta ahí.
    Devuelve bounding box (x, y, w, h) en coordenadas globales.
    """
    if roi is not None:
        x1, y1, x2, y2 = roi
        sub_frame = gray_frame[y1:y2, x1:x2]
        mask = bg.apply(sub_frame)
    else:
        x1, y1 = 0, 0
        sub_frame = gray_frame
        mask = bg.apply(sub_frame)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity > SOLIDITY_MAX:
            continue

        # convertir coords locales a globales
        gx = x + x1
        gy = y + y1

        candidates.append((area, (gx, gy, w, h)))

    if len(candidates) == 0:
        return None, mask

    _, box = max(candidates, key=lambda x: x[0])
    return box, mask


# -------------------------------------
# KALMAN FILTER PARA PREDICCIÓN DE POSICIÓN
# -------------------------------------
def create_kalman():
    """
    Kalman simple con estado [x, y, vx, vy]
    """
    kf = cv2.KalmanFilter(4, 2)  # 4 estados, 2 mediciones

    # Matriz de transición de estado
    # [x]   [1 0 1 0][x]
    # [y] = [0 1 0 1][y]
    # [vx]  [0 0 1 0][vx]
    # [vy]  [0 0 0 1][vy]
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

    # Matriz de medida: solo medimos x, y
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)

    # Ruido de proceso (ajusta si vibra mucho)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

    # Ruido de medición
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    # Error inicial estimado
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf


kalman = create_kalman()
kalman_initialized = False


def kalman_update(cx, cy, have_measurement=True):
    """
    Actualiza o predice con Kalman:
    - Si have_measurement=True: corrige con medición (cx, cy).
    - Si False: solo predice.
    Devuelve (px, py) = posición estimada.
    """
    global kalman_initialized

    if not kalman_initialized and have_measurement:
        # inicializar con la primera medición
        kalman.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        kalman_initialized = True

    if have_measurement:
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        kalman.correct(measurement)

    prediction = kalman.predict()
    px, py = int(prediction[0]), int(prediction[1])
    return px, py


# -------------------------------------
# MAIN LOOP
# -------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    tracker = None
    tracking = False
    manual_mode = False
    last_position = None

    print("[INFO] Controles:")
    print("   s : seleccionar objeto manualmente")
    print("   a : forzar detección automática global")
    print("   ESC : salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame no leído, saliendo...")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key = cv2.waitKey(1) & 0xFF

        # --------------------------
        # MODO SELECCIÓN MANUAL
        # --------------------------
        if key == ord('s'):
            print("[INFO] Selecciona el objeto con el mouse y presiona ENTER.")
            cv2.imshow("frame", frame)
            box = cv2.selectROI("frame", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            if box != (0, 0, 0, 0):
                tracker = create_tracker()
                tracker.init(frame, box)
                tracking = True
                manual_mode = True

                x, y, w, h = [int(v) for v in box]
                cx = x + w // 2
                cy = y + h // 2
                kalman_update(cx, cy, have_measurement=True)
                last_position = (cx, cy)

            continue

        # --------------------------
        # FORZAR DETECCIÓN AUTOMÁTICA GLOBAL
        # --------------------------
        if key == ord('a'):
            print("[INFO] Detección automática global forzada.")
            box, mask = detect_puppybot(frame_gray, roi=None)
            if box is not None:
                tracker = create_tracker()
                tracker.init(frame, box)
                tracking = True
                manual_mode = False

                x, y, w, h = [int(v) for v in box]
                cx = x + w // 2
                cy = y + h // 2
                kalman_update(cx, cy, have_measurement=True)
                last_position = (cx, cy)

        # --------------------------
        # DETECCIÓN AUTOMÁTICA SOLO SI NO ES MANUAL Y NO ESTAMOS TRACKING
        # --------------------------
        if not tracking and not manual_mode:
            box, mask = detect_puppybot(frame_gray, roi=None)
            if box is not None:
                tracker = create_tracker()
                tracker.init(frame, box)
                tracking = True

                x, y, w, h = [int(v) for v in box]
                cx = x + w // 2
                cy = y + h // 2
                kalman_update(cx, cy, have_measurement=True)
                last_position = (cx, cy)

        # --------------------------
        # TRACKING + RECUPERACIÓN
        # --------------------------
        if tracking:
            ok, box = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in box]
                cx = x + w // 2
                cy = y + h // 2
                last_position = (cx, cy)

                # actualizar Kalman con medición
                px, py = kalman_update(cx, cy, have_measurement=True)

                # dibujar caja + centro
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # dibujar predicción Kalman
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Kalman", (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            else:
                # TRACKER PERDIDO
                tracking = False
                manual_mode = False
                print("[WARN] Tracker perdido. Intentando re-detección alrededor de la última posición...")

                if last_position is not None:
                    # usar solo predicción de Kalman si está inicializado
                    px, py = kalman_update(last_position[0], last_position[1], have_measurement=False)
                    lx, ly = px, py
                else:
                    # si nunca tuvimos posición, usamos el centro de la imagen
                    h_img, w_img = frame_gray.shape
                    lx, ly = w_img // 2, h_img // 2

                # definir región de búsqueda
                x1 = max(0, lx - SEARCH_SIZE)
                y1 = max(0, ly - SEARCH_SIZE)
                x2 = min(frame.shape[1], lx + SEARCH_SIZE)
                y2 = min(frame.shape[0], ly + SEARCH_SIZE)
                roi = (x1, y1, x2, y2)

                box, mask = detect_puppybot(frame_gray, roi=roi)
                if box is not None:
                    print("[INFO] Objeto recuperado automáticamente.")
                    tracker = create_tracker()
                    tracker.init(frame, box)
                    tracking = True

                    x, y, w, h = [int(v) for v in box]
                    cx = x + w // 2
                    cy = y + h // 2
                    kalman_update(cx, cy, have_measurement=True)
                    last_position = (cx, cy)
                else:
                    print("[WARN] No se pudo recuperar el objeto en la ventana de búsqueda.")

        # --------------------------
        # VISUALIZACIÓN
        # --------------------------
        cv2.putText(frame, "s: seleccionar  a: auto-detect  ESC: salir",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("frame", frame)

        if SHOW_MASK:
            # mostrar la última máscara generada por bg.apply
            # para eso llamamos detect_puppybot con un ROI nulo solo para visualización
            _, mask_vis = detect_puppybot(frame_gray, roi=None)
            cv2.imshow("mask", mask_vis)

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
