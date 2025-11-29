# udp_teleop_windows.py
import socket
import struct
import time
import msvcrt  # Solo Windows

ROBOT_IP   = "10.182.184.108"  # <-- pon aquí la IP de tu Turtlebot
ROBOT_PORT = 5007

MAX_LIN = 2.0   # Debe calzar con tu nodo (max_linear)
MAX_ANG = 6.0   # Debe calzar con tu nodo (max_angular)

STEP_LIN = 0.1
STEP_ANG = 0.2


def clamp(x, mn, mx):
    return max(mn, min(mx, x))


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    v = 0.0
    w = 0.0

    print("=== UDP Teleop Windows -> Turtlebot ===")
    print("Controles:")
    print("  W/S : acelerar adelante/atrás")
    print("  A/D : girar izquierda/derecha")
    print("  X   : stop (v=0, w=0)")
    print("  Q   : salir")
    print("---------------------------------------")

    try:
        while True:
            # Leer teclas sin bloquear
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    key = ch.decode('utf-8').lower()
                except UnicodeDecodeError:
                    key = ''

                if key == 'w':
                    v += STEP_LIN
                elif key == 's':
                    v -= STEP_LIN
                elif key == 'a':
                    w += STEP_ANG
                elif key == 'd':
                    w -= STEP_ANG
                elif key == 'x':
                    v = 0.0
                    w = 0.0
                elif key == 'q':
                    print("Saliendo...")
                    # enviar stop antes de irnos
                    payload = struct.pack('ff', 0.0, 0.0)
                    sock.sendto(payload, (ROBOT_IP, ROBOT_PORT))
                    break

                # Clamp de seguridad (igual que en el nodo)
                v = clamp(v, -MAX_LIN, MAX_LIN)
                w = clamp(w, -MAX_ANG, MAX_ANG)

                print(f"v={v:.2f} m/s, w={w:.2f} rad/s")

            # Enviar continuamente el último comando
            payload = struct.pack('ff', float(v), float(w))
            try:
                sock.sendto(payload, (ROBOT_IP, ROBOT_PORT))
            except Exception as e:
                print(f"Error enviando UDP: {e}")
                break

            time.sleep(0.05)  # 20 Hz aprox

    finally:
        sock.close()


if __name__ == "__main__":
    main()
