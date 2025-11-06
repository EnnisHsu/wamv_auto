from os import close
import rclpy
from rclpy.node import Node

import time
import threading
import subprocess
import numpy as np
from enum import Enum


from std_msgs.msg import Float64
from ros_gz_interfaces.msg import Contacts
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration


from wamv_auto.map import MapBase
from wamv_auto.rs_astar import RSAStarPlanner
import wamv_auto.clothoid as clothoid
from wamv_auto.pid import PID

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


class Auto_sys_status(Enum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    DOCKED = 3
    ERROR = 4

class wamv_state:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        self.vx = 0.0
        self.vy = 0.0
        self.v = 0.0
        self.omega = 0.0
        
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        self.last_time = 0.0

class wamv(Node):
    def __init__(self):
        super().__init__('wamv')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.get_logger().info('WAMV Node has been started.')
        
        # wamv state
        self.state = wamv_state()

        self.collision_flag = False
        self.odom_ready = False
        self.sub_update_time = 0.0

        # Thruster commands
        self.left_pos = 0.0
        self.right_pos = 0.0
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        
        self.pos_pid = PID(kp=1.0, ki=0.0, kd=0.1, output_limits=(-0.78, 0.78), death_zone=(-0.01, 0.01))
        self.thrust_pid = PID(kp=2000.0, ki=0.0, kd=0.0, output_limits=(-3000.0, 3000.0), death_zone=(-0.02, 0.02))
        
        self.left_pos_pub = self.create_publisher(Float64, '/wamv/thrusters/left/pos', 10)
        self.right_pos_pub = self.create_publisher(Float64, '/wamv/thrusters/right/pos', 10)
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.odom_sub = self.create_subscription(Odometry, '/wamv/odom', self.odom_callback, 10)
        self.collision_sub = self.create_subscription(Contacts, '/vrx/contacts', self.collision_callback, 10)

        self.set_wamv_sub = self.create_subscription(Point, '/wamv/set_wamv_state', self.set_wamv_callback, 10)
        self.set_pos_pid_target_sub = self.create_subscription(Float64, '/wamv/pos_pid/set_target', self.set_pos_pid_target_callback, 10)
        self.set_thrust_pid_target_sub = self.create_subscription(Float64, '/wamv/thrust_pid/set_target', self.set_thrust_pid_target_callback, 10)
        
        # PID 监控话题发布器（用于 rqt_plot 可视化）
        self.pos_target_pub = self.create_publisher(Float64, '/wamv/pos_pid/target', 10)
        self.pos_current_pub = self.create_publisher(Float64, '/wamv/pos_pid/current', 10)
        self.thrust_target_pub = self.create_publisher(Float64, '/wamv/thrust_pid/target', 10)
        self.thrust_current_pub = self.create_publisher(Float64, '/wamv/thrust_pid/current', 10)
        
        self.pid_thread = threading.Thread(target=self.pid_control_thread)
        self.pid_thread.start()

    def collision_callback(self, msg):
        for contact in msg.contacts:
            if 'wamv' in contact.collision1.name or 'wamv' in contact.collision2.name:
                self.collision_flag = True
                self.get_logger().warn('Collision detected with wamv!')
                return self.collision_flag
        self.collision_flag = False
        return self.collision_flag
    
    def odom_callback(self, msg):
        self.state.x = msg.pose.pose.position.x
        self.state.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        self.state.yaw = yaw
        self.odom_ready = True
        
        current_time = time.time()
        if self.state.last_time != 0.0:
            dt = current_time - self.state.last_time
            if dt > 0:
                # 计算世界坐标系下的速度分量
                self.state.vx = (self.state.x - self.state.last_x) / dt
                self.state.vy = (self.state.y - self.state.last_y) / dt
                
                # 转换到船体坐标系，计算有符号的纵向速度
                # v > 0: 前进, v < 0: 后退
                cos_yaw = np.cos(self.state.yaw)
                sin_yaw = np.sin(self.state.yaw)
                self.state.v = self.state.vx * cos_yaw + self.state.vy * sin_yaw
                
                self.thrust_pid.update_current(self.state.v)

                dyaw = self.state.yaw - self.state.last_yaw
                # Normalize dyaw to [-pi, pi]
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                self.state.omega = dyaw / dt
                self.pos_pid.update_current(self.state.yaw)

                # self.get_logger().debug(f'Odom updated: time={current_time} dt={dt} x={self.state.x}, y={self.state.y}, yaw={self.state.yaw}, v={self.state.v}, omega={self.state.omega}')

        self.state.last_x = self.state.x
        self.state.last_y = self.state.y
        self.state.last_yaw = self.state.yaw
        self.state.last_time = current_time
        return self.odom_ready
    
    def set_wamv_callback(self, msg):
        self.get_logger().info(f'WAMV state set to x: {msg.x}, y: {msg.y}, yaw: {msg.z}')
        self.set_wamv(msg.x, msg.y, msg.z)
        
        
    def set_wamv(self, x, y, yaw):
        qz = np.sin((yaw) / 2.0)
        qw = np.cos((yaw) / 2.0)

        try:
            result = subprocess.run([
                "gz", "service",
                "--reqtype", "gz.msgs.Pose",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "2000",
                "--service","/world/docking_world/set_pose",
                "--req", f'name: "wamv" position: {{ x: {x} y: {y} z: 0.0 }} orientation: {{ x: 0.0 y: 0.0 z: {qz} w: {qw} }}'
            ], check=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.get_logger().info(f'WAMV pose ({x}, {y}, {yaw}) set successfully.')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to set WAMV pose: {e.stderr}')
            
    def set_pos_pid_target_callback(self, msg):
        self.get_logger().info(f'Setting Position PID target to: {msg.data}')
        self.pos_pid.update_target(msg.data)
        
    def set_thrust_pid_target_callback(self, msg):
        self.get_logger().info(f'Setting Thrust PID target to: {msg.data}')
        self.thrust_pid.update_target(msg.data)

    def pid_control_thread(self):
        self.get_logger().info('Starting PID control thread for WAMV thrusters.')
        rate = self.create_rate(10)  # 10 Hz
        while rclpy.ok():    
            if self.odom_ready:
                # self.get_logger().debug(f'WAMV State: x={self.state.x}, y={self.state.y}, yaw={self.state.yaw}, v={self.state.v}, omega={self.state.omega}')
                thrust_cmd = self.thrust_pid.compute()
                self.left_thrust_pub.publish(Float64(data=thrust_cmd))
                self.right_thrust_pub.publish(Float64(data=thrust_cmd))
                if thrust_cmd > 0:
                    pos_cmd = - 1.0 * self.pos_pid.compute()
                else:
                    pos_cmd = self.pos_pid.compute()
                self.left_pos_pub.publish(Float64(data=pos_cmd))
                self.right_pos_pub.publish(Float64(data=pos_cmd))
                self.get_logger().debug(f'PID Control: thrust(current={self.state.v} - target={self.thrust_pid.target})_cmd={thrust_cmd}, pos_cmd(current={self.state.yaw} - target={self.pos_pid.target})={pos_cmd}')
            
                # 5. 发布 PID 监控数据供 rqt_plot 使用
                self.pos_target_pub.publish(Float64(data=self.pos_pid.target))
                self.pos_current_pub.publish(Float64(data=self.pos_pid.current))
                self.thrust_target_pub.publish(Float64(data=self.thrust_pid.target))
                self.thrust_current_pub.publish(Float64(data=self.thrust_pid.current))
            else:
                self.get_logger().warn('Odometry not ready, stop wamv.')
                self.pos_pid.update_target(self.state.yaw)
                self.thrust_pid.update_target(0.0)
                pos_cmd = self.pos_pid.compute()
                thrust_cmd = self.thrust_pid.compute()
                self.left_pos_pub.publish(Float64(data=-pos_cmd))
                self.right_pos_pub.publish(Float64(data=-pos_cmd))
                self.left_thrust_pub.publish(Float64(data=thrust_cmd))
                self.right_thrust_pub.publish(Float64(data=thrust_cmd))

            rate.sleep()


class wamv_auto_sys(Node):
    def __init__(self, wamv):
        super().__init__('wamv_auto_sys')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.get_logger().info('WAMV Auto System Node has been started.')

        self.wamv = wamv
        self.auto_sys_status = Auto_sys_status.IDLE

        self.map = MapBase("./SIM.png", enable_dual_inflation=True)
        self.astar_planner = RSAStarPlanner(self.map,
                                            max_curvature=0.05,
                                            step_len=3.0,
                                            yaw_bins=24,
                                            substeps=3,
                                            allow_reverse=True,
                                            reverse_penalty=1.2,
                                            rs_connect_alpha=5.0,
                                            viz_live=False,
                                            penalty_weight=0.3
                                            )   
        # self.clothoid_planner = clothoid.fit_and_speed()
        self.target: wamv_state = wamv_state()
        self.target_received = False
        self.path = []
        
        # Path visualization publisher
        self.path_marker_pub = self.create_publisher(MarkerArray, '/wamv/path_markers', 10)
        self.target_sub = self.create_subscription(Point, '/wamv/target_point', self.target_callback, 10)
        
        self.thread = threading.Thread(target=self.wamv_auto_sys_thread)
        self.thread.start()
        
        self.control_thread = threading.Thread(target=self.wamv_control_sys_thread)
        self.control_thread.start()
        
        # Pure Pursuit 参数
        self.lookahead_distance_base = 2.0  # 基础前瞻距离（米）
        self.arrival_threshold = 2.0  # 到达阈值（米）
        self.min_lookahead = 2.0  # 最小前瞻距离
        self.max_lookahead = 5.0  # 最大前瞻距离
        self.lookahead_gain = 1.0  # 前瞻距离与速度的增益系数
        
        # 接近目标的特殊处理参数
        self.approach_zone_distance = 5.0  # 接近区域：5米内启用特殊处理
        self.min_approach_speed = 0.3  # 接近时的最低速度
        self.path_progress_threshold = 0.8  # 路径点推进阈值（米）
        
        # 超时保护
        self.near_target_start_time = None
        self.near_target_timeout = 15.0  # 15秒超时
        
        # 路径推进状态
        self.current_target_idx = 0  # 当前追踪的目标点索引

    def target_callback(self, msg):
        self.target.x = msg.x
        self.target.y = msg.y
        self.target.yaw = msg.z
        self.get_logger().info(f'Target point received: ({self.target.x:.2f}, {self.target.y:.2f}, {np.degrees(self.target.yaw):.1f}°)')
        self.target_received = True
        

    def visualize_target_docking_zone(self):
        """
        使用 Gazebo Marker 可视化目标停泊区域：
        透明红色框：长度 = 载具长度 + 2×arrival_threshold
                    宽度 = 载具宽度 + 2×arrival_threshold
        """
        try:
            import subprocess
            
            # 获取载具尺寸（从 map.vehicle 获取）
            vehicle_length = self.map.vehicle.length if hasattr(self.map, 'vehicle') else 10.0
            vehicle_width = self.map.vehicle.width if hasattr(self.map, 'vehicle') else 5.0
            
            # 计算红色框尺寸
            box_length = vehicle_length + 2.0 * self.arrival_threshold
            box_width = vehicle_width + 2.0 * self.arrival_threshold
            
            # 绘制透明红色框
            box_marker_cmd = [
                "gz", "service",
                "--reqtype", "gz.msgs.Marker",
                "--reptype", "gz.msgs.Empty",
                "--timeout", "1000",
                "--service", "/marker",
                "--req",
                f'ns: "target_zone", '
                f'id: 1001, '
                f'action: ADD_MODIFY, '
                f'type: BOX, '
                f'pose: {{ '
                f'  position: {{ x: {self.target.x} y: {self.target.y} z: 0.5 }} '
                f'  orientation: {{ '
                f'    x: 0.0 y: 0.0 '
                f'    z: {np.sin(self.target.yaw / 2.0)} '
                f'    w: {np.cos(self.target.yaw / 2.0)} '
                f'  }} '
                f'}}, '
                f'scale: {{ x: {box_length} y: {box_width} z: 1.0 }}, '
                f'material: {{ ambient: {{r: 1.0 g: 0.0 b: 0.0 a: 0.3}} '
                f'diffuse: {{r: 1.0 g: 0.0 b: 0.0 a: 0.3}} }}'
            ]
            
            # 执行命令
            subprocess.run(box_marker_cmd, check=False, capture_output=True)
            
            self.get_logger().info(
                f'Target docking zone visualized: '
                f'box size=({box_length:.1f}m × {box_width:.1f}m) at ({self.target.x:.1f}, {self.target.y:.1f}, {np.degrees(self.target.yaw):.1f}°)'
            )
            
        except Exception as e:
            self.get_logger().error(f'Failed to visualize target docking zone: {e}')
    
    def clear_target_docking_zone(self):
        """
        清除目标停泊区域的可视化标记
        """
        try:
            import subprocess
            clear_cmd = [
                "gz", "service",
                "--reqtype", "gz.msgs.Marker",
                "--reptype", "gz.msgs.Empty",
                "--timeout", "1000",
                "--service", "/marker",
                "--req",
                f'ns: "target_zone", id: 1001, action: DELETE_MARKER'
            ]
            subprocess.run(clear_cmd, check=False, capture_output=True)
            self.get_logger().info('Target docking zone marker cleared')
        except Exception as e:
            self.get_logger().error(f'Failed to clear target docking zone: {e}')

    def calculate_path(self, start : wamv_state, goal : wamv_state):
        self.get_logger().info(f'Calculating path from ({start.x}, {start.y}, {start.yaw}) to ({goal.x}, {goal.y}, {goal.yaw})')
        astar_path = self.astar_planner.plan([start.x, start.y, start.yaw], [goal.x, goal.y, goal.yaw])
        if astar_path is None:
            self.get_logger().error('No path found!')
            return None
        clothoid_path = clothoid.fit_and_speed(astar_path,
                                          resample_step=2.0,
                                          smooth_win=5,
                                          v_max=3.0,
                                          a_long=0.6,
                                          a_brake=0.6,
                                          a_lat_max=10.0,
                                          r_max=0.78,
                                          kappa_max=1.0/20.0,
                                          enforce_zero_at_switch=True,
                                          auto_infer_if_constant=True
                                          )
        self.get_logger().debug(f'Clothoid path: {clothoid_path}.')
        # self.path = clothoid.pack_points_with_velocity(clothoid_path)
        resample_path = clothoid.resample_uniform_time(clothoid_path, dt=0.1)
        self.get_logger().debug(f'Resampled path: {resample_path}.')
        self.path = clothoid.pack_points_with_velocity(resample_path)
        return self.path

    def show_path(self):
        """
        在 VRX 仿真场景中通过箭头显示泊船路径
        使用 ROS 2 Marker 消息来可视化路径
        """
        if not self.path or len(self.path) == 0:
            self.get_logger().warn('No path to display!')
            return
        
        marker_array = MarkerArray()
        
        # 创建路径线段 Marker
        line_marker = Marker()
        line_marker.header.frame_id = "world"  # 或者使用 "map" 根据你的坐标系
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "path_line"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        # 设置线条属性
        line_marker.scale.x = 0.3  # 线宽
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0  # 绿色
        line_marker.color.b = 0.0
        line_marker.color.a = 0.8  # 透明度
        
        # 设置持久时间（0 表示永久显示直到被删除）
        line_marker.lifetime = Duration(sec=0, nanosec=0)
        
        # 添加路径点到线段
        for point_data in self.path:
            p = Point()
            # 假设 path 中每个点是 [x, y, yaw] 或者 (x, y, yaw) 格式
            if isinstance(point_data, (list, tuple)) and len(point_data) >= 2:
                p.x = float(point_data[0])
                p.y = float(point_data[1])
                p.z = 0.5  # 设置高度，使其在水面上方可见
            else:
                self.get_logger().warn(f'Invalid path point format: {point_data}')
                continue
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # 创建箭头 Markers 显示方向
        arrow_interval = max(1, len(self.path) // 20)  # 每隔一定间隔显示一个箭头，避免过于密集
        
        for i in range(0, len(self.path), arrow_interval):
            point_data = self.path[i]
            
            if not isinstance(point_data, (list, tuple)) or len(point_data) < 3:
                continue
                
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "world"
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = "path_arrows"
            arrow_marker.id = i + 1
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            # 箭头位置
            arrow_marker.pose.position.x = float(point_data[0])
            arrow_marker.pose.position.y = float(point_data[1])
            arrow_marker.pose.position.z = 0.5
            
            # 箭头方向（使用路径点的 yaw）
            yaw = float(point_data[2])
            # 将 yaw 转换为四元数
            arrow_marker.pose.orientation.x = 0.0
            arrow_marker.pose.orientation.y = 0.0
            arrow_marker.pose.orientation.z = np.sin(yaw / 2.0)
            arrow_marker.pose.orientation.w = np.cos(yaw / 2.0)
            
            # 箭头尺寸
            arrow_marker.scale.x = 1.5  # 箭头长度
            arrow_marker.scale.y = 0.3  # 箭头宽度
            arrow_marker.scale.z = 0.3  # 箭头高度
            
            # 箭头颜色（蓝色）
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 0.5
            arrow_marker.color.b = 1.0
            arrow_marker.color.a = 0.9
            
            arrow_marker.lifetime = Duration(sec=0, nanosec=0)
            
            marker_array.markers.append(arrow_marker)
        
        # 发布 Marker Array
        self.path_marker_pub.publish(marker_array)
        self.get_logger().info(f'Published path with {len(marker_array.markers)} markers ({len(self.path)} points)')
    
    def clear_path_markers(self):
        """
        清除所有路径标记
        """
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.path_marker_pub.publish(marker_array)
        self.get_logger().info('Cleared all path markers')
    
    def show_path_gz_sim(self):
        """
        使用 gz service 在 Gazebo Sim 中显示泊船路径
        通过调用 /marker 服务直接在仿真场景中绘制路径线段和箭头
        """
        if not self.path or len(self.path) == 0:
            self.get_logger().warn('No path to display in Gazebo Sim!')
            return
        
        # 确保 path 是可迭代的列表类型
        if not isinstance(self.path, (list, tuple, np.ndarray)):
            self.get_logger().error(f'Path is not a list/tuple/array, it is: {type(self.path)}')
            return
        
        self.get_logger().info(f'Displaying path in Gazebo Sim with {len(self.path)} points')
        
        # 1. 清除所有旧的标记
        # try:
        #     subprocess.run([
        #         "gz", "service",
        #         "-s", "/marker",
        #         "--reqtype", "gz.msgs.Marker",
        #         "--reptype", "gz.msgs.Empty",
        #         "--timeout", "1000",
        #         "--req", "action: DELETE_ALL"
        #     ], check=False, capture_output=True, text=True)
        #     self.get_logger().debug('Cleared old markers in Gazebo Sim')
        # except Exception as e:
        #     self.get_logger().warn(f'Failed to clear old markers: {e}')
        
        # 2. 构建路径线段的点列表
        points_list = []
        for point_data in self.path:
            if isinstance(point_data, (list, tuple, np.ndarray)) and len(point_data) >= 2:
                x = float(point_data[0])
                y = float(point_data[1])
                z = 0.5  # 设置高度，使其在水面上方可见
                points_list.append(f"{{x: {x}, y: {y}, z: {z}}}")
            else:
                self.get_logger().warn(f'Invalid path point format: {point_data}')
        
        if len(points_list) == 0:
            self.get_logger().error('No valid points to display')
            return
        
        points_str = ", ".join(points_list)
        
        # 3. 添加路径线段 (绿色)
        line_cmd = [
            "gz", "service",
            "-s", "/marker",
            "--reqtype", "gz.msgs.Marker",
            "--reptype", "gz.msgs.Empty",
            "--timeout", "2000",
            "--req", 
            f"action: ADD_MODIFY, "
            f"ns: 'wamv_path_line', "
            f"id: 0, "
            f"type: LINE_STRIP, "
            f"point: [{points_str}], "
            f"material: {{ambient: {{r: 0.0, g: 1.0, b: 0.0, a: 0.9}}, diffuse: {{r: 0.0, g: 1.0, b: 0.0, a: 0.9}}}}"
        ]
        
        try:
            result = subprocess.run(line_cmd, check=True, capture_output=True, text=True)
            self.get_logger().info(f'Path line displayed in Gazebo Sim with {len(points_list)} points')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to display path line in Gazebo Sim: {e.stderr}')
            return
        
        # 4. 添加箭头标记显示方向 (蓝色)
        arrow_interval = max(1, len(self.path) // 20)  # 每隔一定间隔显示一个箭头
        arrow_count = 0
        
        for i in range(0, len(self.path), arrow_interval):
            point_data = self.path[i]
            
            if not isinstance(point_data, (list, tuple, np.ndarray)) or len(point_data) < 3:
                continue
            
            x = float(point_data[0])
            y = float(point_data[1])
            z = 1.5  # 提高高度，确保明显可见
            yaw = float(point_data[2])
            
            # # 使用 BOX（立方体）代替 ARROW，更容易看到
            # # 将 yaw 转换为四元数
            # qz = np.sin(yaw / 2.0)
            # qw = np.cos(yaw / 2.0)
            
            # # 方法1：使用 BOX 显示方向（更明显）
            # box_cmd = [
            #     "gz", "service",
            #     "-s", "/marker",
            #     "--reqtype", "gz.msgs.Marker",
            #     "--reptype", "gz.msgs.Empty",
            #     "--timeout", "100",
            #     "--req",
            #     f"action: ADD_MODIFY, "
            #     f"ns: 'wamv_path_boxes', "
            #     f"id: {i + 1}, "
            #     f"type: BOX, "
            #     f"pose: {{position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}}}, "
            #     f"scale: {{x: 2.0, y: 0.5, z: 0.5}}, "  # 长方体：长2米，宽和高0.5米
            #     f"material: {{ambient: {{r: 1.0, g: 0.0, b: 0.0, a: 1.0}}, diffuse: {{r: 1.0, g: 0.0, b: 0.0, a: 1.0}}}}"  # 红色
            # ]
            
            # 方法2：使用 TRIANGLE_LIST 绘制平面三角形箭头
            # 计算三角形箭头的三个顶点
            arrow_length = 1.5  # 箭头长度
            arrow_width = 0.8   # 箭头底边宽度
            
            # 箭头尖端（最前方的点）
            tip_x = x + arrow_length * np.cos(yaw)
            tip_y = y + arrow_length * np.sin(yaw)
            
            # 箭头底边的两个点（垂直于前进方向）
            # 计算垂直方向的单位向量
            perp_x = -np.sin(yaw)
            perp_y = np.cos(yaw)
            
            # 左底点
            left_x = x + perp_x * arrow_width / 2.0
            left_y = y + perp_y * arrow_width / 2.0
            
            # 右底点
            right_x = x - perp_x * arrow_width / 2.0
            right_y = y - perp_y * arrow_width / 2.0
            
            # 构建 TRIANGLE_LIST，需要指定三个顶点
            triangle_cmd = [
                "gz", "service",
                "-s", "/marker",
                "--reqtype", "gz.msgs.Marker",
                "--reptype", "gz.msgs.Empty",
                "--timeout", "100",
                "--req",
                f"action: ADD_MODIFY, "
                f"ns: 'wamv_path_arrows', "
                f"id: {i * 2 + 2}, "
                f"type: TRIANGLE_LIST, "
                f"point: ["
                f"{{x: {tip_x}, y: {tip_y}, z: {z}}}, "     # 尖端
                f"{{x: {left_x}, y: {left_y}, z: {z}}}, "   # 左底点
                f"{{x: {right_x}, y: {right_y}, z: {z}}}"   # 右底点
                f"], "
                f"material: {{ambient: {{r: 1.0, g: 0.8, b: 0.0, a: 1.0}}, diffuse: {{r: 1.0, g: 0.8, b: 0.0, a: 1.0}}}}"  # 金黄色箭头
            ]
            
            try:
                # subprocess.run(box_cmd, check=True, capture_output=True, text=True)
                subprocess.run(triangle_cmd, check=True, capture_output=True, text=True)
                arrow_count += 1
            except subprocess.CalledProcessError as e:
                self.get_logger().warn(f'Failed to display arrow {i}: {e.stderr}')
        
        self.get_logger().info(f'Displayed {arrow_count} arrows in Gazebo Sim')
    
    def clear_path_gz_sim(self):
        """
        清除 Gazebo Sim 中的所有路径标记
        """
        try:
            subprocess.run([
                "gz", "service",
                "-s", "/marker",
                "--reqtype", "gz.msgs.Marker",
                "--reptype", "gz.msgs.Empty",
                "--timeout", "1000",
                "--req", "action: DELETE_ALL"
            ], check=True, capture_output=True, text=True)
            self.get_logger().info('Cleared all path markers in Gazebo Sim')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to clear markers in Gazebo Sim: {e.stderr}')
    
    def find_closest_path_point(self, current_x: float, current_y: float) -> int:
        """
        找到路径上离当前位置最近的点的索引
        
        Args:
            current_x: 当前 x 坐标
            current_y: 当前 y 坐标
        
        Returns:
            最近点的索引
        """
        if not self.path or len(self.path) == 0:
            return 0
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(self.path):
            dx = point[0] - current_x
            dy = point[1] - current_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def pure_pursuit_control(self, lookahead_distance: float) -> tuple:
        """
        Pure Pursuit 路径跟踪算法
        
        Args:
            lookahead_distance: 前瞻距离（米）
        
        Returns:
            (target_yaw, target_speed, lookahead_point_idx): 目标航向、速度和前瞻点索引
        """
        if not self.path or len(self.path) == 0:
            return self.wamv.state.yaw, 0.0, 0
        
        # 1. 找到最近点
        closest_idx = self.find_closest_path_point(self.wamv.state.x, self.wamv.state.y)
        
        # 2. 在最近点之后寻找与前瞻距离匹配的点
        lookahead_point = None
        lookahead_idx = closest_idx
        
        for i in range(closest_idx, len(self.path)):
            dx = self.path[i][0] - self.wamv.state.x
            dy = self.path[i][1] - self.wamv.state.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist >= lookahead_distance:
                lookahead_point = self.path[i]
                lookahead_idx = i
                break
        
        # 如果没找到（说明已经接近路径终点），使用最后一个点
        if lookahead_point is None:
            lookahead_point = self.path[-1]
            lookahead_idx = len(self.path) - 1
        
        # 3. 计算目标航向（从当前位置指向前瞻点）
        dx = lookahead_point[0] - self.wamv.state.x
        dy = lookahead_point[1] - self.wamv.state.y
        target_yaw = np.arctan2(dy, dx)
        
        # 4. 目标速度从路径点获取（假设路径点格式为 [x, y, yaw, u, v_lat, r]）
        if len(lookahead_point) > 3:
            target_speed = lookahead_point[3]
        else:
            target_speed = 0.0
        
        return target_yaw, target_speed, lookahead_idx
    
    def get_adaptive_lookahead_distance(self) -> float:
        """
        根据速度和换向点自适应调整前瞻距离
        
        Returns:
            自适应的前瞻距离（米）
        """
        # 前瞻距离 = 基础距离 + 速度增益 * 当前速度
        base_lookahead = self.lookahead_distance_base + self.lookahead_gain * abs(self.wamv.state.v)
        
        # 限制在合理范围内
        base_lookahead = np.clip(base_lookahead, self.min_lookahead, self.max_lookahead)
        
        # 检查前方是否有换向点
        if hasattr(self, 'current_target_idx') and self.current_target_idx < len(self.path) - 1:
            current_speed = self.path[self.current_target_idx][3] if len(self.path[self.current_target_idx]) > 3 else 0.0
            
            # 查找最近的换向点（在前方 20 个点内）
            for i in range(self.current_target_idx + 1, min(self.current_target_idx + 20, len(self.path))):
                next_speed = self.path[i][3] if len(self.path[i]) > 3 else 0.0
                if current_speed * next_speed < 0:  # 速度符号变化
                    # 计算到换向点的距离
                    dx = self.path[i][0] - self.wamv.state.x
                    dy = self.path[i][1] - self.wamv.state.y
                    dist_to_switch = np.sqrt(dx**2 + dy**2)
                    
                    # 如果换向点在前瞻距离内，缩短前瞻距离
                    if dist_to_switch < base_lookahead:
                        adjusted_lookahead = max(self.min_lookahead, dist_to_switch * 0.7)
                        self.get_logger().debug(
                            f'Reduced lookahead near reverse switch: '
                            f'{base_lookahead:.2f}m → {adjusted_lookahead:.2f}m (switch at {dist_to_switch:.2f}m)'
                        )
                        return adjusted_lookahead
                    break  # 只考虑最近的换向点
        
        return base_lookahead
    
    def handle_low_speed_approach(self, target_speed: float, dist_to_end: float) -> float:
        """
        处理接近目标时的低速情况
        
        Args:
            target_speed: 路径规划的目标速度
            dist_to_end: 到终点的距离
        
        Returns:
            调整后的目标速度
        """
        # 只在接近区域内处理
        if dist_to_end > self.approach_zone_distance:
            return target_speed
        
        # 如果目标速度太低（接近或在死区内）
        if abs(target_speed) < self.min_approach_speed:
            # 根据距离动态调整速度
            # 距离越近，速度越慢，但不低于最低速度
            if dist_to_end > self.arrival_threshold:
                # 线性插值：从最低速度到更低的速度
                speed_ratio = (dist_to_end - self.arrival_threshold) / (self.approach_zone_distance - self.arrival_threshold)
                adjusted_speed = self.min_approach_speed * (0.5 + 0.5 * speed_ratio)  # 0.5x 到 1.0x 之间
                
                # 保持原来的方向（正负号）
                if target_speed < 0:
                    adjusted_speed = -adjusted_speed
                
                self.get_logger().debug(
                    f'Low speed approach: original={target_speed:.3f}, '
                    f'adjusted={adjusted_speed:.3f}, dist={dist_to_end:.2f}m'
                )
                return adjusted_speed
        
        return target_speed
    
    def check_arrival_timeout(self, dist_to_end: float) -> bool:
        """
        检查是否在接近区域停留过久（超时）
        
        Args:
            dist_to_end: 到终点的距离
        
        Returns:
            True 如果超时应该强制到达
        """
        if dist_to_end < self.approach_zone_distance:
            if self.near_target_start_time is None:
                # 第一次进入接近区域
                self.near_target_start_time = time.time()
                self.get_logger().info(f'Entered approach zone ({dist_to_end:.2f}m), starting timeout counter')
                return False
            else:
                # 检查是否超时
                elapsed = time.time() - self.near_target_start_time
                if elapsed > self.near_target_timeout:
                    self.get_logger().warn(
                        f'Approach timeout ({elapsed:.1f}s > {self.near_target_timeout}s) at {dist_to_end:.2f}m. '
                        f'Declaring arrival.'
                    )
                    return True
        else:
            # 离开接近区域，重置计时器
            self.near_target_start_time = None
        
        return False
    
    def detect_reverse_switch_point(self, current_idx: int) -> tuple:
        """
        [DEPRECATED] 检测当前是否接近倒车切换点
        
        此方法已弃用，换向点检测现在已集成到 progressive_pure_pursuit_control() 中。
        保留此方法仅为向后兼容，不建议使用。
        
        Args:
            current_idx: 当前目标点索引
        
        Returns:
            (is_switch_point, distance_to_switch, prev_speed): 
            是否是切换点、到切换点的距离、切换前的速度
        """
        import warnings
        warnings.warn(
            "detect_reverse_switch_point() is deprecated. "
            "Reverse point detection is now handled in progressive_pure_pursuit_control().",
            DeprecationWarning,
            stacklevel=2
        )
        
        if current_idx <= 0 or current_idx >= len(self.path):
            return False, float('inf'), 0.0
        
        prev_speed = self.path[current_idx - 1][3]
        curr_speed = self.path[current_idx][3]
        
        # 检测速度符号变化（倒车点）
        is_switch = (prev_speed * curr_speed < 0)
        
        if is_switch:
            switch_point = self.path[current_idx]
            dx = switch_point[0] - self.wamv.state.x
            dy = switch_point[1] - self.wamv.state.y
            dist_to_switch = np.sqrt(dx**2 + dy**2)
            return True, dist_to_switch, prev_speed
        
        return False, float('inf'), 0.0
    
    def progressive_pure_pursuit_control(self, lookahead_distance: float) -> tuple:
        """
        渐进式 Pure Pursuit：防止前瞻点跨越换向点
        
        Returns:
            (target_yaw, target_speed, lookahead_idx, current_target_idx)
        """
        if not self.path or len(self.path) == 0:
            return self.wamv.state.yaw, 0.0, 0, 0
        
        closest_idx = self.find_closest_path_point(self.wamv.state.x, self.wamv.state.y)
        if closest_idx > self.current_target_idx:
            self.current_target_idx = closest_idx
            self.get_logger().debug(f'Updated current_target_idx to closest point: {self.current_target_idx}/{len(self.path)}')
        
        # # 1. 检查是否到达当前目标点，如果是则推进
        # if self.current_target_idx < len(self.path) - 1:
        #     current_target = self.path[self.current_target_idx]
        #     dx = current_target[0] - self.wamv.state.x
        #     dy = current_target[1] - self.wamv.state.y
        #     dist_to_current = np.sqrt(dx**2 + dy**2)
            
        #     if dist_to_current < self.path_progress_threshold:
        #         self.current_target_idx += 1
        #         self.get_logger().debug(
        #             f'Progressed to waypoint {self.current_target_idx}/{len(self.path)} '
        #             f'(reached within {dist_to_current:.2f}m)'
        #         )
        
        # 2. 关键修改：查找换向点（速度符号变化的点）
        reverse_switch_idx = None
        if self.current_target_idx < len(self.path):
            current_speed = self.path[self.current_target_idx][3] if len(self.path[self.current_target_idx]) > 3 else 0.0
            
            for i in range(self.current_target_idx + 1, len(self.path)):
                next_speed = self.path[i][3] if len(self.path[i]) > 3 else 0.0
                if current_speed * next_speed < 0:  # 速度符号变化
                    reverse_switch_idx = i
                    self.get_logger().debug(f'Detected reverse switch at index {i}')
                    break
        
        # 3. 确定搜索前瞻点的最大索引（不能超过换向点）
        max_search_idx = reverse_switch_idx if reverse_switch_idx is not None else len(self.path)
        
        # 4. 在限定范围内寻找前瞻点
        lookahead_point = None
        lookahead_idx = self.current_target_idx
        
        for i in range(self.current_target_idx, max_search_idx):
            dx = self.path[i][0] - self.wamv.state.x
            dy = self.path[i][1] - self.wamv.state.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist >= lookahead_distance:
                lookahead_point = self.path[i]
                lookahead_idx = i
                break
        
        # 如果没找到，使用搜索范围内的最后一个点
        if lookahead_point is None:
            if max_search_idx > self.current_target_idx:
                lookahead_point = self.path[max_search_idx - 1]
                lookahead_idx = max_search_idx - 1
            else:
                lookahead_point = self.path[self.current_target_idx]
                lookahead_idx = self.current_target_idx
        
        
        # 6. 关键修改：使用前瞻点的速度，确保航向和速度一致
        target_speed = lookahead_point[3] if len(lookahead_point) > 3 else 0.0
        
        # 5. 计算目标航向（基于前瞻点）
        dx = lookahead_point[0] - self.wamv.state.x
        dy = lookahead_point[1] - self.wamv.state.y
        if target_speed >= 0:
        # 前进：船头指向前瞻点
            target_yaw = np.arctan2(dy, dx)
        else:
            # 倒车：船尾指向前瞻点，即船头背向前瞻点
            # 方法1：使用路径点的 psi（推荐）
            target_yaw = lookahead_point[2] if len(lookahead_point) > 2 else self.wamv.state.yaw
        
        # 7. 如果接近换向点，减速
        if reverse_switch_idx is not None:
            switch_point = self.path[reverse_switch_idx]
            dx_switch = switch_point[0] - self.wamv.state.x
            dy_switch = switch_point[1] - self.wamv.state.y
            dist_to_switch = np.sqrt(dx_switch**2 + dy_switch**2)
            
            if dist_to_switch < 2.0:  # 距离换向点 2 米内减速
                speed_scale = max(0.3, dist_to_switch / 2.0)
                original_speed = target_speed
                target_speed = target_speed * speed_scale
                self.get_logger().debug(
                    f'Slowing near reverse switch: {original_speed:.2f} → {target_speed:.2f} m/s '
                    f'(dist={dist_to_switch:.2f}m)'
                )
        
        return target_yaw, target_speed, lookahead_idx, self.current_target_idx
            
    def get_docking_path(self, target):
        self.get_logger().info(f'Calculating docking path to target: {target}')
        
        docking_path = self.calculate_path(self.wamv.state, target)
        if docking_path:
            self.get_logger().info(f'Docking path calculated with {len(docking_path)} points.')
            self.clear_path_markers()
            self.show_path()
        
    def wamv_auto_sys_thread(self):
        """
        WAMV 自动系统主线程
        """
        self.get_logger().info('WAMV Auto System thread started.')
        while rclpy.ok():
            
            # 在这里添加自动系统的主要逻辑
            if self.target_received:
                if self.auto_sys_status == Auto_sys_status.IDLE:
                    self.get_logger().info('Starting path planning...')
                    self.auto_sys_status = Auto_sys_status.PLANNING
                    # 重置路径跟踪状态
                    self.current_target_idx = 0
                    self.near_target_start_time = None
                    # 清除旧路径并显示目标区域
                    self.clear_path_gz_sim()
                    self.visualize_target_docking_zone()
                    docking_path = self.calculate_path(self.wamv.state, self.target)
                    if docking_path:
                        self.get_logger().info(f'Docking path calculated with {len(docking_path)} points.')
                        self.get_logger().debug(f'Path points: {docking_path}')
                        # self.clear_path_markers()
                        # self.show_path()
                        self.show_path_gz_sim()
                        self.auto_sys_status = Auto_sys_status.NAVIGATING
                    else:
                        self.get_logger().error('Path planning failed, returning to IDLE state.')
                        self.auto_sys_status = Auto_sys_status.IDLE
                    self.target_received = False
                else:
                    self.get_logger().info(f'System currently busy({self.auto_sys_status}), cannot process new target.')
            else:
                if self.auto_sys_status == Auto_sys_status.DOCKED:
                    self.get_logger().info('WAMV is docked at target. Awaiting new target...')
                    self.current_target_idx = 0
                    self.near_target_start_time = None
                    self.auto_sys_status = Auto_sys_status.IDLE
            #     if self.auto_sys_status == Auto_sys_status.NAVIGATING:
            #         self.get_logger().info('Navigating to target... TBA....')
            #         self.get_logger().info('Navigation complete, returning to IDLE state.')
            #         self.auto_sys_status = Auto_sys_status.IDLE
            time.sleep(0.1)
            
    def wamv_control_sys_thread(self):
        """
        WAMV 控制系统线程 - 使用改进的 Pure Pursuit 算法
        包括低速接近处理、超时保护、倒车切换点检测
        """
        self.get_logger().info('WAMV Control System (Enhanced Pure Pursuit) thread started.')
        
        while rclpy.ok():
            if self.auto_sys_status == Auto_sys_status.NAVIGATING:
                if not self.path or len(self.path) == 0:
                    self.get_logger().error('Path is empty!')
                    self.auto_sys_status = Auto_sys_status.IDLE
                    # 停船
                    self.wamv.pos_pid.update_target(self.wamv.state.yaw)
                    self.wamv.thrust_pid.update_target(0.0)
                    continue
                
                # 1. 计算到终点的距离
                dx_end = self.path[-1][0] - self.wamv.state.x
                dy_end = self.path[-1][1] - self.wamv.state.y
                dist_to_end = np.sqrt(dx_end**2 + dy_end**2)
                
                # 2. 超时检查（优先级高于正常到达检查）
                if self.check_arrival_timeout(dist_to_end):
                    self.get_logger().warn(f'Forcing arrival due to timeout at {dist_to_end:.2f}m')
                    # 强制到达
                    self.wamv.pos_pid.update_target(self.path[-1][2])
                    self.wamv.thrust_pid.update_target(0.0)
                    self.wamv.thrust_pid.enable_death_zone = True
                    self.auto_sys_status = Auto_sys_status.DOCKED
                    continue
                
                # 3. 正常到达检查
                if dist_to_end < self.arrival_threshold:
                    self.get_logger().info(f'Arrived at destination! Distance: {dist_to_end:.2f}m')
                    # 停船：保持目标航向，推力设为0
                    self.wamv.pos_pid.update_target(self.path[-1][2])
                    self.wamv.thrust_pid.update_target(0.0)
                    # 恢复死区设置
                    self.wamv.thrust_pid.enable_death_zone = True
                    self.auto_sys_status = Auto_sys_status.DOCKED
                    continue
                
                # 4. 动态调整推力PID死区
                if self.arrival_threshold < dist_to_end < self.approach_zone_distance:
                    # 接近区域：禁用死区，确保能够微调
                    if self.wamv.thrust_pid.enable_death_zone:
                        self.wamv.thrust_pid.enable_death_zone = False
                        self.get_logger().info(
                            f'Entering approach zone ({dist_to_end:.2f}m), disabling thrust death zone'
                        )
                
                # 5. 根据当前速度自适应调整前瞻距离
                lookahead_distance = self.get_adaptive_lookahead_distance()
                
                # 6. 渐进式 Pure Pursuit 控制：基于弧长推进而非时间
                # 换向点检测和限制已经在 progressive_pure_pursuit_control 内部处理
                target_yaw, target_speed, lookahead_idx, current_target_idx = \
                    self.progressive_pure_pursuit_control(lookahead_distance)
                
                # 7. 低速接近处理：确保有足够推力到达目标
                adjusted_speed = self.handle_low_speed_approach(target_speed, dist_to_end)
                
                # 8. 更新 PID 控制器目标
                self.wamv.pos_pid.update_target(target_yaw)
                self.wamv.thrust_pid.update_target(adjusted_speed)
                
                # 9. 日志输出（每秒输出一次）
                current_time = time.time()
                if not hasattr(self, '_last_log_time') or (current_time - self._last_log_time) >= 1.0:
                    self._last_log_time = current_time
                    
                    # 计算横向误差
                    closest_idx = self.find_closest_path_point(self.wamv.state.x, self.wamv.state.y)
                    closest_point = self.path[closest_idx]
                    dx_closest = closest_point[0] - self.wamv.state.x
                    dy_closest = closest_point[1] - self.wamv.state.y
                    cross_track_error = np.sqrt(dx_closest**2 + dy_closest**2)
                    
                    # 计算航向误差
                    heading_error = target_yaw - self.wamv.state.yaw
                    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                    
                    # 计算剩余超时时间
                    timeout_info = ""
                    if self.near_target_start_time is not None:
                        elapsed = time.time() - self.near_target_start_time
                        remaining = self.near_target_timeout - elapsed
                        timeout_info = f", timeout in {remaining:.1f}s"
                    
                    self.get_logger().info(
                        f'Enhanced Pure Pursuit Status:\n'
                        f'  Progress: waypoint {current_target_idx}/{len(self.path)}, '
                        f'lookahead {lookahead_idx}/{len(self.path)} '
                        f'({100.0*current_target_idx/len(self.path):.1f}%)\n'
                        f'  Distance to end: {dist_to_end:.2f}m{timeout_info}\n'
                        f'  Lookahead distance: {lookahead_distance:.2f}m\n'
                        f'  Current: pos=({self.wamv.state.x:.2f}, {self.wamv.state.y:.2f}), '
                        f'yaw={np.degrees(self.wamv.state.yaw):.1f}°, speed={self.wamv.state.v:.2f}m/s\n'
                        f'  Target: yaw={np.degrees(target_yaw):.1f}°, '
                        f'speed={target_speed:.2f}→{adjusted_speed:.2f}m/s\n'
                        f'  Errors: cross_track={cross_track_error:.2f}m, heading={np.degrees(heading_error):.1f}°\n'
                        f'  Death zone: {"DISABLED" if not self.wamv.thrust_pid.enable_death_zone else "ENABLED"}'
                    )
            
            elif self.auto_sys_status == Auto_sys_status.DOCKED:
                # 保持停船状态
                self.wamv.thrust_pid.update_target(0.0)
        
            time.sleep(0.1)
            
def main(args=None):
    rclpy.init(args=args)
    
    wamv_node = wamv()

    wamv_auto_sys_node = wamv_auto_sys(wamv=wamv_node)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(wamv_auto_sys_node)
    executor.add_node(wamv_node)
    executor.spin()
    # rclpy.spin(wamv_auto_sys_node)
    # wamv_auto_sys_node.destroy_node()
    # rclpy.shutdown()
    
if __name__ == '__main__':
    main()