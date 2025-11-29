#!/usr/bin/env python3
"""
movedor_mac.py - Control manual del TurtleBot para macOS

Controles:
  W/S : acelerar adelante/atrás (impulsos)
  A/D : girar izquierda/derecha (impulsos)
  X   : stop inmediato
  Q   : salir

Modo: impulsos (presiona tecla → envía comando → para automáticamente)
"""
import socket
import struct
import time
import sys
import select
import termios
import tty

ROBOT_IP = "192.168.0.220"  # Cambia a la IP real del TurtleBot si usas robot físico
ROBOT_PORT = 5007

# Velocidades para impulsos
IMPULSE_LIN = 0.3   # velocidad lineal por impulso
IMPULSE_ANG = 0.5   # velocidad angular por impulso
IMPULSE_DURATION = 0.2  # duración del impulso en segundos


def clamp(x, mn, mx):
    return max(mn, min(mx, x))


class NonBlockingInput:
    """Clase para leer teclas sin bloquear en macOS/Linux"""
    
    def __init__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def kbhit(self):
        """Devuelve True si hay una tecla presionada"""
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    
    def getch(self):
        """Lee un caracter sin bloquear"""
        if self.kbhit():
            return sys.stdin.read(1)
        return None


def send_command(sock, robot_addr, v, w, duration=None):
    """Envía comando UDP. Si duration está especificado, envía por ese tiempo y luego para."""
    payload = struct.pack('ff', float(v), float(w))
    
    if duration is None:
        # Envío único
        try:
            sock.sendto(payload, robot_addr)
            print(f"Enviado: v={v:.2f} m/s, w={w:.2f} rad/s")
        except Exception as e:
            print(f"Error enviando UDP: {e}")
    else:
        # Envío por duración especificada
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                sock.sendto(payload, robot_addr)
                time.sleep(0.02)  # 50 Hz mientras dura el impulso
            
            # Enviar stop al final del impulso
            stop_payload = struct.pack('ff', 0.0, 0.0)
            sock.sendto(stop_payload, robot_addr)
            print(f"Impulso: v={v:.2f} m/s, w={w:.2f} rad/s durante {duration}s → STOP")
        except Exception as e:
            print(f"Error enviando impulso: {e}")


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_addr = (ROBOT_IP, ROBOT_PORT)

    print("=== UDP Teleop macOS (Modo Impulsos) -> Turtlebot ===")
    print("Controles:")
    print("  W/S : impulso adelante/atrás")
    print("  A/D : impulso izquierda/derecha") 
    print("  X   : stop inmediato")
    print("  Q   : salir")
    print("  ESC : salir")
    print("---------------------------------------")
    print("Cada tecla envía un impulso y para automáticamente")
    print("Presiona una tecla para empezar...")

    try:
        with NonBlockingInput() as nb_input:
            while True:
                # Leer teclas sin bloquear
                key = nb_input.getch()
                
                if key is not None:
                    key = key.lower()
                    
                    if key == 'w':
                        send_command(sock, robot_addr, IMPULSE_LIN, 0.0, IMPULSE_DURATION)
                    elif key == 's':
                        send_command(sock, robot_addr, -IMPULSE_LIN, 0.0, IMPULSE_DURATION)
                    elif key == 'a':
                        send_command(sock, robot_addr, 0.0, IMPULSE_ANG, IMPULSE_DURATION)
                    elif key == 'd':
                        send_command(sock, robot_addr, 0.0, -IMPULSE_ANG, IMPULSE_DURATION)
                    elif key == 'x':
                        send_command(sock, robot_addr, 0.0, 0.0)
                        print("STOP inmediato")
                    elif key == 'q' or ord(key) == 27:  # ESC
                        print("\nSaliendo...")
                        # enviar stop antes de irnos
                        send_command(sock, robot_addr, 0.0, 0.0)
                        break

                time.sleep(0.01)  # pequeña pausa para no saturar CPU

    except KeyboardInterrupt:
        print("\nInterrumpido por Ctrl+C")
    finally:
        # Enviar stop final
        try:
            send_command(sock, robot_addr, 0.0, 0.0)
        except Exception:
            pass
        sock.close()
        print("Conexión cerrada.")


if __name__ == "__main__":
    main()