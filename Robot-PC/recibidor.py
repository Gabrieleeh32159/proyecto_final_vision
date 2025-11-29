#!/usr/bin/env python3
import socket
import struct
import time
import traceback

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class UdpTeleopReceiver(Node):
    def __init__(self):
        super().__init__('udp_teleop_receiver')

        # ====== Parámetros (se pueden remapear con ros2 run/launch) ======
        self.declare_parameter('listen_ip', '0.0.0.0')
        self.declare_parameter('listen_port', 5007)
        # Publicamos en /cmd_vel por defecto, como teleop_twist_keyboard stamped
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('timeout_sec', 0.5)   # Si no llegan cmds, se frena
        self.declare_parameter('max_linear', 2.0)    # Seguridad extra
        self.declare_parameter('max_angular', 6.0)

        listen_ip   = self.get_parameter('listen_ip').get_parameter_value().string_value
        listen_port = self.get_parameter('listen_port').get_parameter_value().integer_value
        self.timeout_sec = self.get_parameter('timeout_sec').get_parameter_value().double_value
        self.max_linear  = self.get_parameter('max_linear').get_parameter_value().double_value
        self.max_angular = self.get_parameter('max_angular').get_parameter_value().double_value
        cmd_topic  = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        # ====== Publisher a /cmd_vel (TwistStamped) ======
        self.pub_cmd = self.create_publisher(TwistStamped, cmd_topic, 10)

        # ====== Socket UDP no bloqueante ======
        self.get_logger().info(
            f"[INIT] Creando socket UDP en {listen_ip}:{listen_port}..."
        )
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((listen_ip, listen_port))
            self.sock.setblocking(False)
        except Exception as e:
            self.get_logger().error(f"[INIT] Error creando/bindeando socket: {e}")
            self.get_logger().error(traceback.format_exc())
            raise

        try:
            local_ip, local_port = self.sock.getsockname()
            self.get_logger().info(
                f"[INIT] Socket realmente escuchando en {local_ip}:{local_port}"
            )
        except Exception as e:
            self.get_logger().warn(f"[INIT] No se pudo obtener getsockname(): {e}")

        self.get_logger().info(
            f"🛰️  Escuchando UDP en {listen_ip}:{listen_port}, publicando TwistStamped en '{cmd_topic}'\n"
            f"     timeout_sec={self.timeout_sec}, max_linear={self.max_linear}, max_angular={self.max_angular}"
        )

        # Última vez que llegó un comando
        self.last_msg_time = time.time()
        self.last_v = 0.0
        self.last_w = 0.0
        self.already_stopped = True

        # Contadores de diagnóstico
        self.timer_calls = 0
        self.no_data_cycles = 0

        # Timer a ~50 Hz para leer socket y publicar TwistStamped
        self.timer = self.create_timer(0.02, self.timer_callback)

    def timer_callback(self):
        """Lee todos los paquetes disponibles y publica el último recibido como TwistStamped."""
        self.timer_calls += 1
        updated = False

        # Leer todos los datagramas pendientes (nos quedamos con el último)
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                # No hay más datos en este ciclo
                break
            except Exception as e:
                self.get_logger().warn(f"[UDP] Error recibiendo: {e}")
                self.get_logger().warn(traceback.format_exc())
                break

            data_len = len(data)
            self.get_logger().info(
                f"[UDP] Recibidos {data_len} bytes desde {addr}"
            )

            if data_len < 8:
                self.get_logger().warn(
                    f"[UDP] Paquete muy corto ({data_len} bytes) desde {addr}, se ignora"
                )
                continue

            # Loguear primeros bytes en hex para depurar formato
            hex_preview = data[:16].hex()
            self.get_logger().info(
                f"[UDP] Datos (hasta 16 bytes) en hex: {hex_preview}"
            )

            try:
                v, w = struct.unpack('ff', data[:8])
            except struct.error as e:
                self.get_logger().warn(f"[UDP] Error desempaquetando struct: {e}")
                continue

            # Clamp de seguridad
            v_clamped = max(-self.max_linear, min(self.max_linear, v))
            w_clamped = max(-self.max_angular, min(self.max_angular, w))

            if v != v_clamped or w != w_clamped:
                self.get_logger().info(
                    f"[UDP] Valores fuera de rango, clampeados: "
                    f"v={v:.3f}->{v_clamped:.3f}, w={w:.3f}->{w_clamped:.3f}"
                )

            self.last_v = v_clamped
            self.last_w = w_clamped
            self.last_msg_time = time.time()
            updated = True

            self.get_logger().info(
                f"[UDP] Actualizado comando: v={self.last_v:.3f}, w={self.last_w:.3f}, "
                f"timestamp={self.last_msg_time:.3f}"
            )

        # Diagnóstico si no llega nada
        if not updated:
            self.no_data_cycles += 1
            # Cada ~1 segundo (50 ciclos aprox) avisa que no llega nada
            if self.no_data_cycles % 50 == 0:
                self.get_logger().info(
                    f"[UDP] {self.no_data_cycles} ciclos sin recibir datos "
                    f"(~{self.no_data_cycles * 0.02:.1f} s)"
                )
        else:
            # Si este ciclo sí hubo datos, reseteamos el contador
            self.no_data_cycles = 0

        now = time.time()

        # Construimos TwistStamped (como teleop_twist_keyboard con stamped:=true)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""

        # Si pasó demasiado tiempo sin nuevos comandos, frenar
        if now - self.last_msg_time > self.timeout_sec:
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0

            if not self.already_stopped:
                self.get_logger().info(
                    f"⏹️  Timeout sin comandos: frenando robot "
                    f"(dt={now - self.last_msg_time:.3f} s)"
                )
                self.already_stopped = True
        else:
            # Publicar el último comando recibido
            msg.twist.linear.x = float(self.last_v)
            msg.twist.angular.z = float(self.last_w)
            if self.timer_calls % 25 == 0:  # log cada ~0.5s para no spamear
                self.get_logger().info(
                    f"[CMD_VEL] Publicando TwistStamped: v={msg.twist.linear.x:.3f}, "
                    f"w={msg.twist.angular.z:.3f}, dt={now - self.last_msg_time:.3f} s"
                )
            self.already_stopped = False

        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UdpTeleopReceiver()
    node.get_logger().info("[MAIN] Nodo UdpTeleopReceiver iniciado, esperando paquetes UDP...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[MAIN] KeyboardInterrupt, cerrando nodo...")
    except Exception as e:
        node.get_logger().error(f"[MAIN] Excepción no controlada: {e}")
        node.get_logger().error(traceback.format_exc())
    finally:
        node.get_logger().info("Cerrando nodo UDP teleop receiver...")
        try:
            node.sock.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()