from os import close
import nav_msgs
import nav_msgs.msg
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

from nav_msgs.msg import Path
from rclpy.action import ActionClient
from nav2_msgs.action import FollowPath
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_regulated_pure_pursuit_controller import RegulatedPurePursuitController
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
        
        # self.pid_thread = threading.Thread(target=self.pid_control_thread)
        # self.pid_thread.start()

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


class wamv_nav2_sys(Node):
    def __init__(self, wamv):
        super().__init__('wamv_nav2_sys')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.get_logger().info('WAMV Navigation2 System Node has been started.')

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
        self.nav2_path : Path = []
        
        # Path visualization publisher
        self.path_marker_pub = self.create_publisher(MarkerArray, '/wamv/path_markers', 10)
        self.target_sub = self.create_subscription(Point, '/wamv/target_point', self.target_callback, 10)
        
        self.ac = ActionClient(self, FollowPath, 'follow_path')
        self._active_goal_handle = None
        

        self.thread = threading.Thread(target=self.wamv_nav2_sys_thread)
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
    
            
    def convert_path_to_nav2(self, path):
        """
        将路径转换为 Nav2 可用的格式（PoseStamped 列表）
        
        Args:
        """
        nav2_path : nav_msgs.msg.Path = []

        for point in path:
            pose = nav_msgs.msg.PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.2
            pose.pose.orientation = self.get_quaternion_from_yaw(point[2])
            nav2_path.poses.append(pose)

        return nav2_path

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
                    # 清除旧路径并显示目标区域
                    self.clear_path_gz_sim()
                    self.visualize_target_docking_zone()
                    docking_path = self.calculate_path(self.wamv.state, self.target)
                    self.nav2_path = self.convert_path_to_nav2(docking_path)
                    if self.nav2_path:
                        self.get_logger().info(f'Docking path calculated with {len(self.nav2_path)} points.')
                        self.get_logger().debug(f'Path points: {self.nav2_path}')
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
                self.send_active_goal(self.nav2_path)
            
            elif self.auto_sys_status == Auto_sys_status.DOCKED:
                # 保持停船状态
                self.wamv.thrust_pid.update_target(0.0)
        
            time.sleep(0.1)
            
    def cancel_active_goal(self):
        if self._active_goal_handle is not None:
            try:
                cancel_future = self._active_goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
                self.get_logger().info('Active goal cancelled successfully.')
            except Exception as e:
                self.get_logger().error(f'Failed to cancel active goal: {e}')
            self._active_goal_handle = None
            
    def send_active_goal(self, path: Path):
        if not self.ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available!')
            return
        
        self.cancel_active_goal()
        
        path.header.stamp = self.get_clock().now().to_msg()
        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = "FollowPath"
        
        self.get_logger().info('Sending new goal to FollowPath action server...')
        send_goal_future = self.ac.send_goal_async(goal_msg, feedback_callback=self.on_feedback)

        send_goal_future.add_done_callback(self.on_goal_response)
        
    def on_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'Feedback received: {feedback}')
        self.get_logger().info(f'Received feedback: {feedback.current_waypoint}/{feedback.total_waypoints} waypoints reached.')

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            self._active_goal_handle = None
            return

        self.get_logger().info('Goal accepted :)')
        self._active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_get_result)
            
    def on_get_result(self, future):
        try:
            result = future.result()
            status = result.status
            self.get_logger().info(f'Goal result received with status: {status}')
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.auto_sys_status = Auto_sys_status.DOCKED
        finally:
            self._active_goal_handle = None

def main(args=None):
    rclpy.init(args=args)
    
    wamv_node = wamv()

    wamv_nav2_sys_node = wamv_nav2_sys(wamv=wamv_node)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(wamv_nav2_sys_node)
    executor.add_node(wamv_node)
    executor.spin()
    # rclpy.spin(wamv_auto_sys_node)
    # wamv_auto_sys_node.destroy_node()
    # rclpy.shutdown()
    
if __name__ == '__main__':
    main()