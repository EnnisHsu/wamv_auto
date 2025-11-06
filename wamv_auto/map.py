from typing import List, Optional
import cv2  # 用于形态学膨胀
import matplotlib
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, LinearRing, box
from shapely.ops import unary_union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import random

matplotlib.use('tkAgg')  # 避免与 pygame 冲突

class Vehicle:
    def __init__(self, length=10.0, width=5.0):
        """
        length: 船体长度（单位：cm）
        width: 船体宽度（单位：cm）
        """
        self.length = length
        self.width = width
        
    def set_pose(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.box = self.create_vehicle_box(x, y, yaw)
        
    def create_vehicle_box(self, x, y, yaw):
        rect = Polygon([
            (-self.length / 2, - self.width / 2),
            (self.length / 2, - self.width / 2),
            (self.length / 2, self.width / 2),
            (-self.length / 2, self.width / 2)
        ])
        
        rotated_rect = rotate(rect, angle=yaw * 180 / np.pi, origin=(0, 0))
        rotated_rect = translate(rotated_rect, x, y)
        
        return LinearRing(rotated_rect.exterior.coords)
        

class MapBase:
    def __init__(self, image_path,  transform_width: float = 100.0, transform_height: float = 100.0,
                 enable_dual_inflation: bool = False):
        """
        image_path: 图像路径
        transform_width: 图像转换后的宽度（单位：cm）
        transform_height: 图像转换后的高度（单位：cm）
        """

        # 基础图与尺寸
        self.image_path = image_path
        self.image = Image.open(self.image_path).convert('L')
        self.grid = np.flipud(np.array(self.image))
        # self.grid = np.array(self.image)

        self.pixel_height, self.pixel_width = self.grid.shape
        self.transform_width = transform_width
        self.transform_height = transform_height
        self.pixel_resolution_x = self.transform_width / self.pixel_width
        self.pixel_resolution_y = self.transform_height / self.pixel_height

        # 车辆模型
        self.vehicle = Vehicle(5.0, 3.0)

        # 功能开关：双层膨胀掩码
        self.enable_dual_inflation = bool(enable_dual_inflation)

        # 障碍几何
        self.obstacle_polygons = self._extract_obstacles()

        # 双层膨胀掩码（严禁通行区/需碰撞校验区）
        self.strict_forbid_mask = None  # True=严禁通行
        self.soft_check_mask = None     # True=需额外碰撞检测
        # 软限制惩罚图（像素级）：值随接近障碍而增大
        self.soft_penalty_map = None  # numpy.ndarray[H,W], float32
        if self.enable_dual_inflation:
            self._init_dual_inflation_masks()

        # 交互绘图句柄
        self._fig = None
        self._ax = None
        self._path_line = None
        self._path_coll = None  # LineCollection 渐变路径
        self._veh_patches = []  # 实时所有姿态的多边形补丁句柄
        self._veh_patch = None  # 末端隐形补丁（用于箭头定位）
        self._arrow = None
        self._arrows = []  # 全程箭头句柄列表（实时）
        self._path_cbar = None
        # 全局粗路径（下采样）覆盖图层句柄
        self._global_line = None
        self._global_scatter = None

    def _extract_obstacles(self):
        obstacle_coords = np.argwhere(self.grid == 0)
        # obstacle_polygons = [
        #     box(
        #         x * self.pixel_resolution_x,
        #         (self.pixel_height - y - 1) * self.pixel_resolution_y,
        #         (x + 1) * self.pixel_resolution_x,
        #         (self.pixel_height - y) * self.pixel_resolution_y
        #     )
        #     for y, x in obstacle_coords
        # ]
        obstacle_polygons = [
            box(
                x * self.pixel_resolution_x,
                y * self.pixel_resolution_y,  # 直接使用 y 坐标
                (x + 1) * self.pixel_resolution_x,
                (y + 1) * self.pixel_resolution_y
            )
            for y, x in obstacle_coords
        ]
        print(f"Extracted {len(obstacle_polygons)} obstacle polygons from the map.")
        # print(f"Obstacle polygons: {obstacle_polygons}")
        return unary_union(obstacle_polygons)
    
    def visualize_obstacles(self, block=True):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            self.grid,
            cmap='gray',
            extent=[0, self.transform_width, 0, self.transform_height],
            origin='lower'
        )
    
        # 绘制障碍区域
        if self.obstacle_polygons:
            geom = self.obstacle_polygons
            if getattr(geom, 'geom_type', None) == 'Polygon':
                x, y = geom.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='red', ec='red', label='Obstacle')
            elif getattr(geom, 'geom_type', None) in ('MultiPolygon', 'GeometryCollection'):
                for polygon in getattr(geom, 'geoms', []):
                    if getattr(polygon, 'geom_type', None) == 'Polygon':
                        x, y = polygon.exterior.xy
                        ax.fill(x, y, alpha=0.5, fc='red', ec='red', label='Obstacle')
    
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Obstacle Visualization")
        plt.legend()
        plt.tight_layout()
        if block:
            plt.show()
    
    def check_collision(self, vehicle_ring: LinearRing) -> bool:
        """返回 True 表示发生碰撞或越界。
        规则：
        - 车辆多边形任意部分超出 [0,W]×[0,H] 边界即视为碰撞（禁止出界）。
        - 与障碍几何有相交也视为碰撞。
        - 允许贴边：即刚好贴在边界线上不视为越界。
        """
        vehicle_poly = Polygon(vehicle_ring)
        # 越界判定（允许贴边）：
        minx, miny, maxx, maxy = vehicle_poly.bounds
        if minx < 0.0 or miny < 0.0 or maxx > float(self.transform_width) or maxy > float(self.transform_height):
            return True
        # 障碍碰撞
        return vehicle_poly.intersects(self.obstacle_polygons)

    def _is_polygon_within_bounds(self, vehicle_ring: LinearRing) -> bool:
        """仅做边界包络检查：车辆多边形必须完全位于 [0,W]×[0,H] 内（贴边允许）。"""
        poly = Polygon(vehicle_ring)
        minx, miny, maxx, maxy = poly.bounds
        return (minx >= 0.0 and miny >= 0.0 and
                maxx <= float(self.transform_width) and maxy <= float(self.transform_height))

    def show_map_with_vehicle(self, vehicle_ring: LinearRing, grid_spacing=0.1):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            self.grid,
            cmap='gray',
            extent=[0, self.transform_width, 0, self.transform_height],
            origin='lower'
        )

        # 网格设置
        ax.set_xticks(np.arange(0, self.transform_width + grid_spacing, grid_spacing))
        ax.set_yticks(np.arange(0, self.transform_height + grid_spacing, grid_spacing))
        ax.grid(which='both', color='lightblue', linestyle='--', linewidth=0.5)

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Map with Grid")
        
        if vehicle_ring is not None:
            x, y = vehicle_ring.xy
            ax.fill(x, y, alpha=0.5, ec='blue' , label='Vehicle')
            dx = 0.01 * np.cos(0)
            dy = 0.01 * np.sin(0)
            center_x, center_y = vehicle_ring.centroid
            ax.arrow(center_x, center_y, dx, dy, head_width=0.005, head_length=0.01, fc='red', ec='red')
        
        plt.tight_layout()
        plt.show()
        
    def _init_dual_inflation_masks(self):
        """
        构建两层基于障碍的膨胀掩码：
        - 严禁通行区：以内层半径 R1 = 车辆宽度/2 进行膨胀，任何通过均禁止。
        - 软限制区：外层半径 R2 = 车辆对角线/2，与严禁通行区的差集，区域内允许，但使用时需要做载具碰撞检测。
        掩码均为与 grid 同尺寸的 bool 数组。
        """
        # 原始占用（grid==0 视为障碍）
        occ = (self.grid == 0).astype(np.uint8)
        H, W = occ.shape
        # 计算像素半径（cm -> px）
        L = float(getattr(self.vehicle, 'length', 10.0))
        Wv = float(getattr(self.vehicle, 'width', 5.0))
        r1_cm = 0.5 * Wv
        r2_cm = 0.5 * float(np.hypot(L, Wv))
        # 使用较小的分辨率换算，保证覆盖
        px_size_r1 = max(1, int(np.ceil(r1_cm / max(1e-6, min(self.pixel_resolution_x, self.pixel_resolution_y)))))
        px_size_r2 = max(px_size_r1, int(np.ceil(r2_cm / max(1e-6, min(self.pixel_resolution_x, self.pixel_resolution_y)))))

        # 构造结构元素并膨胀
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px_size_r1 + 1, 2 * px_size_r1 + 1))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px_size_r2 + 1, 2 * px_size_r2 + 1))
        dil1 = cv2.dilate(occ, k1, iterations=1)
        dil2 = cv2.dilate(occ, k2, iterations=1)

        strict = dil1.astype(bool)
        soft = (dil2.astype(bool) & (~strict))
        self.strict_forbid_mask = strict
        self.soft_check_mask = soft

        # 构建软限制惩罚图：严格区=超大惩罚；软区=随距离递减；外部=0
        try:
            obs = (self.grid == 0).astype(np.uint8)  # 1=障碍
            dist_px = cv2.distanceTransform(1 - obs, distanceType=cv2.DIST_L2, maskSize=5)
            px_to_cm = float(min(self.pixel_resolution_x, self.pixel_resolution_y))
            dist_cm = dist_px * px_to_cm

            BIG = 1e6
            penalty = np.zeros_like(dist_cm, dtype=np.float32)
            # 严禁区：直接给超大惩罚
            penalty[strict] = BIG

            # 软区：从 r1 到 r2 内按距离线性（或幂次）递减
            r1 = r1_cm
            r2 = r2_cm
            denom = max(1e-6, (r2 - r1))
            idx = np.where(soft)
            dvals = dist_cm[idx]
            # t=1 表示靠近 r1（更危险），t=0 靠近 r2（较安全）
            t = 1.0 - np.clip((dvals - r1) / denom, 0.0, 1.0)
            gamma = 1.0  # >1 更靠近 r1 时更高
            k = 50.0     # 强度
            pvals = (k * (t ** gamma)).astype(np.float32)
            penalty[idx] = pvals

            self.soft_penalty_map = penalty
        except Exception:
            print("Error occurred while initializing soft penalty map.")
            self.soft_penalty_map = None

    def show_soft_penalty(self, vmax: float = 60.0, cmap_name: str = 'Reds', viz: bool = True, block: bool = True):
        """可视化软限制惩罚图（值大=靠近障碍；红色单色渐变），colorbar 刻度显示 penalty 数值。"""
        if self.soft_penalty_map is None:
            raise RuntimeError("soft_penalty_map 未生成；请确保启用双层膨胀或已调用 _init_dual_inflation_masks().")
        fig, ax = plt.subplots(figsize=(6, 6))
        ext = [0, self.transform_width, 0, self.transform_height]
        ax.imshow(self.grid, cmap='gray', extent=ext, origin='lower')
        ax.grid(which='both', color='lightblue', linestyle='--', linewidth=0.5)
        # 自动生成 colorbar 刻度（基于当前 vmax）
        im = ax.imshow(self.soft_penalty_map, extent=ext, origin='lower', cmap=cmap_name, alpha=0.6, vmin=0.0, vmax=vmax)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='soft penalty')
        try:
            ticks = np.linspace(0.0, float(vmax), 6)
            cb.set_ticks(ticks)
            labels = [f"{t:.0f}" if t >= 1.0 else f"{t:.2f}" for t in ticks]
            cb.set_ticklabels(labels)
        except Exception:
            pass
        ax.set_title('Soft Penalty Map (higher=closer to obstacles)')
        ax.set_xlabel('East (cm)')
        ax.set_ylabel('North (cm)')
        plt.tight_layout()
        if viz:
            plt.show(block=block)
        else:
            plt.close(fig)

    def world_to_rc(self, x: float, y: float) -> tuple[int, int]:
        """世界坐标(cm) -> 栅格行列索引(r,c)"""
        r = int(np.clip(y / self.pixel_resolution_y, 0, self.pixel_height - 1))
        c = int(np.clip(x / self.pixel_resolution_x, 0, self.pixel_width - 1))
        return r, c

    def classify_xy(self, x: float, y: float) -> str:
        """
        将点(x,y)分类：
        - 'forbid' 严禁通行区
        - 'soft'   需碰撞检测区
        - 'free'   完全自由区
        """
        if not getattr(self, 'enable_dual_inflation', False):
            return 'free'
        if self.strict_forbid_mask is None or self.soft_check_mask is None:
            self._init_dual_inflation_masks()
        r, c = self.world_to_rc(x, y)
        if self.strict_forbid_mask[r, c]:
            return 'forbid'
        if self.soft_check_mask[r, c]:
            return 'soft'
        return 'free'

    def is_pose_traversable(self, x: float, y: float, yaw: float) -> bool:
        """
        基于双层掩码判断姿态是否可通行：
        - 严禁区：直接 False
        - 软限制区：需要生成车辆多边形，进行碰撞检测，通过则 True
        - 自由区：True
        """
        # 若未启用双层膨胀，则回退为普通的车辆碰撞检测
        if not getattr(self, 'enable_dual_inflation', False):
            try:
                ring = self.vehicle.create_vehicle_box(x, y, yaw)
            except Exception:
                return False
            return not self.check_collision(ring)
        # 启用时，走严格/软限制/自由逻辑
        cls = self.classify_xy(x, y)
        if cls == 'forbid':
            return False
        if cls == 'soft':
            try:
                ring = self.vehicle.create_vehicle_box(x, y, yaw)
            except Exception:
                return False
            return not self.check_collision(ring)
        # 自由区：至少需满足边界完整包络（不出界）
        try:
            ring = self.vehicle.create_vehicle_box(x, y, yaw)
        except Exception:
            return False
        return self._is_polygon_within_bounds(ring)

    def show_dual_inflation_masks(self, overlay: bool = True, alpha_forbid: float = 0.35, alpha_soft: float = 0.25,
                                  grid_spacing: float = 5.0, save_path: Optional[str] = None,
                                  viz: bool = True, block: bool = True) -> None:
        """
        以颜色叠加显示双层膨胀：
        - 红色：严禁通行区 strict_forbid_mask
        - 橙色：软限制区 soft_check_mask
        叠加顺序：先软再严禁，使严禁优先显示。
        """
        if self.strict_forbid_mask is None or self.soft_check_mask is None:
            self._init_dual_inflation_masks()
        ext = [0, self.transform_width, 0, self.transform_height]
        fig, ax = plt.subplots(figsize=(6, 6))
        if overlay:
            ax.imshow(self.grid, cmap='gray', extent=ext, origin='lower')
        # 软限制区（橙色）
        ax.imshow(self.soft_check_mask.astype(np.uint8), cmap='Oranges', alpha=alpha_soft,
                  extent=ext, origin='lower', vmin=0, vmax=1)
        # 严禁区（红色）
        ax.imshow(self.strict_forbid_mask.astype(np.uint8), cmap='Reds', alpha=alpha_forbid,
                  extent=ext, origin='lower', vmin=0, vmax=1)
        # 网格
        ax.set_xticks(np.arange(0, self.transform_width + grid_spacing, grid_spacing))
        ax.set_yticks(np.arange(0, self.transform_height + grid_spacing, grid_spacing))
        ax.grid(which='both', color='lightblue', linestyle='--', linewidth=0.5)
        ax.set_xlabel("East (cm)")
        ax.set_ylabel("North (cm)")
        ax.set_title("Dual-Inflation Masks: forbid (red), soft (orange)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if viz:
            plt.show(block=block)
        else:
            plt.close(fig)

    def show_map(self, trajectory: List, grid_spacing=5, viz: bool = True, save_fig: bool = False, fig_name: str = "map.png"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            self.grid,
            cmap='gray',
            extent=[0, self.transform_width, 0, self.transform_height],
            origin='lower'
        )

        

        # 网格设置
        ax.set_xticks(np.arange(0, self.transform_width + grid_spacing, grid_spacing))
        ax.set_yticks(np.arange(0, self.transform_height + grid_spacing, grid_spacing))
        ax.grid(which='both', color='lightblue', linestyle='--', linewidth=0.5)

        ax.set_xlabel("East (cm)")
        ax.set_ylabel("North (cm)")
        ax.set_title("Map with Grid")
        
        if trajectory is not None and len(trajectory) > 0:
            print(f"Trajectory length: {len(trajectory)}, Trajectory: {trajectory}")

            # 1) 折线路径：按时间渐变蓝色
            xs = [p[0] for p in trajectory]
            ys = [p[1] for p in trajectory]
            if len(trajectory) >= 2:
                points = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # 将起点颜色调深：使用 0.35~0.95 的范围，避免接近白色
                t_min, t_max = 0.35, 0.95
                tvals_line = np.linspace(t_min, t_max, len(segments))
                lc = LineCollection(segments, cmap='Blues', norm=plt.Normalize(t_min, t_max))
                lc.set_array(tvals_line)
                lc.set_linewidth(2.0)
                lc.set_alpha(0.9)
                ax.add_collection(lc)
                # 为路径渐变添加颜色条
                cb = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label('Path')
                cb.set_ticks([t_min, t_max])
                cb.set_ticklabels(['Start', 'End'])

            # 标记起点与终点
            # 起点使用更深的蓝色，并加黑色描边增强可见性
            start_color = cm.get_cmap('Blues')(0.9)
            ax.scatter(xs[0], ys[0], c=[start_color], s=40, zorder=4, label='Start', edgecolors='k', linewidths=0.6)
            ax.scatter(xs[-1], ys[-1], c='red', s=30, zorder=3, label='Goal')
            
            # 2) 船体矩形：按时间索引渐变蓝色（可视化姿态演化）
            n = len(trajectory)
            # 船体渐变也同步加深起点颜色
            tvals = np.linspace(0.35, 0.95, n)
            cmap = cm.get_cmap('Blues')
            alpha = 0.35
            for i, pose in enumerate(trajectory):
                vehicle_ring = self.vehicle.create_vehicle_box(pose[0], pose[1], pose[2])
                x, y = vehicle_ring.xy
                color = cmap(tvals[i])
                # 船体填充
                ax.fill(x, y, fc=color, ec=color, alpha=alpha)
                # 在每个姿态绘制方向箭头
                cx, cy = vehicle_ring.centroid.x, vehicle_ring.centroid.y
                dx = 3 * np.cos(pose[2])
                dy = 3 * np.sin(pose[2])
                # 使用较深蓝的箭头，并稍微半透明，避免遮挡过重
                ax.arrow(cx, cy, dx, dy, head_width=1, head_length=2, fc='red', ec='red', alpha=0.8)
            
            # 注意：颜色条已在上面基于 lc 创建
            
        plt.tight_layout()
        if viz:
            plt.show()
        if save_fig:
            plt.savefig(fig_name, bbox_inches='tight')
        
    def _draw_static_bg(self, ax, grid_spacing=5):
        ax.imshow(
            self.grid,
            cmap='gray',
            extent=[0, self.transform_width, 0, self.transform_height],
            origin='lower'
        )
        ax.set_xticks(np.arange(0, self.transform_width + grid_spacing, grid_spacing))
        ax.set_yticks(np.arange(0, self.transform_height + grid_spacing, grid_spacing))
        ax.grid(which='both', color='lightblue', linestyle='--', linewidth=0.5)
        ax.set_xlabel("East (cm)")
        ax.set_ylabel("North (cm)")
        ax.set_title("Map with Grid")

        # 障碍只绘一次
        if self.obstacle_polygons:
            geom = self.obstacle_polygons
            if getattr(geom, 'geom_type', None) == 'Polygon':
                x, y = geom.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='red', ec='red')
            elif getattr(geom, 'geom_type', None) in ('MultiPolygon', 'GeometryCollection'):
                for polygon in getattr(geom, 'geoms', []):
                    if getattr(polygon, 'geom_type', None) == 'Polygon':
                        x, y = polygon.exterior.xy
                        ax.fill(x, y, alpha=0.5, fc='red', ec='red')

    def show_map_live_begin(self, grid_spacing=5, backend=None, viz: bool = True):
        """
        启动实时绘制（非阻塞）
        """
        if backend:
            plt.switch_backend(backend)  # 可传 'TkAgg' 或 'Qt5Agg'
        if viz and not plt.isinteractive():
            plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6, 6))
        self._draw_static_bg(self._ax, grid_spacing=grid_spacing)
        # 确保窗口非阻塞显示并立即绘制
        try:
            plt.show(block=False)
        except Exception:
            pass
        self._fig.canvas.draw_idle()
        try:
            self._fig.canvas.flush_events()
        except Exception:
            pass
        try:
            plt.pause(0.001)
        except Exception:
            pass

    def show_map_live_update(self, trajectory: List):
        """
        更新轨迹（非阻塞）
        - 轨迹使用蓝色渐变折线（LineCollection）
        - 载具矩形按时间索引使用蓝色渐变填充（与 show_map 一致）
        - 箭头：全程在每个姿态绘制红色箭头
        """
        # 若未初始化，自动启动实时绘制
        if self._fig is None or self._ax is None:
            self.show_map_live_begin(viz=True)
        if not plt.isinteractive():
            plt.ion()
        if not trajectory:
            return

        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        # 若之前画过旧的简单折线，移除它（改用渐变折线）
        if getattr(self, '_path_line', None) is not None:
            try:
                self._path_line.remove()
            except Exception:
                pass
            self._path_line = None

        # 更新渐变折线（LineCollection）
        if getattr(self, '_path_coll', None) is not None:
            try:
                self._path_coll.remove()
            except Exception:
                pass
            self._path_coll = None

        if len(trajectory) >= 2:
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t_min, t_max = 0.35, 0.95  # 起点颜色偏深，避免接近白色
            tvals_line = np.linspace(t_min, t_max, len(segments))
            lc = LineCollection(segments, cmap='Blues', norm=plt.Normalize(t_min, t_max))
            lc.set_array(tvals_line)
            lc.set_linewidth(2.0)
            lc.set_alpha(0.9)
            self._path_coll = self._ax.add_collection(lc)

            if getattr(self, '_path_cbar', None) is None:
                self._path_cbar = self._fig.colorbar(self._path_coll, ax=self._ax, fraction=0.046, pad=0.04)
                self._path_cbar.set_label('Path Progress')
                self._path_cbar.set_ticks([0.35, 0.95])
                self._path_cbar.set_ticklabels(['Start', 'End'])
        # 清理旧的所有载具矩形补丁
        if getattr(self, '_veh_patches', None):
            for p in self._veh_patches:
                try:
                    p.remove()
                except Exception:
                    pass
        self._veh_patches = []

        # 载具矩形按时间渐变
        n = len(trajectory)
        cmap = cm.get_cmap('Blues')
        tvals = np.linspace(0.35, 0.95, n)
        alpha = 0.35
        for i, pose in enumerate(trajectory):
            ring = self.vehicle.create_vehicle_box(pose[0], pose[1], pose[2])
            vx, vy = ring.xy
            color = cmap(tvals[i])
            patch = self._ax.fill(vx, vy, fc=color, ec=color, alpha=alpha)[0]
            self._veh_patches.append(patch)

        # 更新末端船体描绘（保持一份；可选）
        last = trajectory[-1]
        ring = self.vehicle.create_vehicle_box(last[0], last[1], last[2])
        vx, vy = ring.xy
        if self._veh_patch is not None:
            try:
                self._veh_patch.remove()
            except Exception:
                pass
            self._veh_patch = None
        self._veh_patch = self._ax.fill(vx, vy, alpha=0.0)[0]  # 隐形，仅作为末端姿态引用

        # 清理旧的全程箭头
        if getattr(self, '_arrows', None):
            for a in self._arrows:
                try:
                    a.remove()
                except Exception:
                    pass
        self._arrows = []

        # 在每个姿态绘制红色箭头
        for pose in trajectory:
            dx = 3 * np.cos(pose[2])
            dy = 3 * np.sin(pose[2])
            ring_i = self.vehicle.create_vehicle_box(pose[0], pose[1], pose[2])
            cx, cy = ring_i.centroid.x, ring_i.centroid.y
            arrow = self._ax.arrow(cx, cy, dx, dy, head_width=1, head_length=2, fc='red', ec='red', alpha=0.85)
            self._arrows.append(arrow)

        # 刷新 GUI 事件队列，确保实时可见
        self._fig.canvas.draw_idle()
        try:
            self._fig.canvas.flush_events()
        except Exception:
            pass
        try:
            plt.pause(0.001)  # 关键：非阻塞刷新
        except Exception:
            pass

    def show_map_live_end(self, block=True):
        """
        结束实时绘制
        """
        if self._fig is None:
            return
        plt.ioff()
        if block:
            plt.show()
        # 清理句柄（可选）
        # self._fig, self._ax = None, None
        # self._path_line = None
        # self._veh_patch = None
        # self._arrow = None
        
    def show_free_mask(self, free_mask, overlay: bool = True, grid_spacing: float = 5.0,
                       alpha: float = 0.35, save_path: Optional[str] = None,
                       viz: bool = True, block: bool = True) -> None:
        """
        显示/保存膨胀后的可行区域（free_mask=True）。
        - overlay=True: 叠加在原始地图灰度图上
        - alpha: 可行区域透明度
        - save_path: 保存到文件（若提供）
        - viz: 是否前台显示；False 仅保存不显示
        - block: 显示时是否阻塞（plt.show(block)）
        """
        ext = [0, self.transform_width, 0, self.transform_height]
        fig, ax = plt.subplots(figsize=(6, 6))
        if overlay:
            ax.imshow(self.grid, cmap='gray', extent=ext, origin='lower')
        # 以绿色覆盖可行区域
        ax.imshow(free_mask.astype(np.uint8),
                  cmap='gray', alpha=alpha, extent=ext, origin='lower',
                  vmin=0, vmax=1)
        # 网格与坐标轴
        ax.set_xticks(np.arange(0, self.transform_width + grid_spacing, grid_spacing))
        ax.set_yticks(np.arange(0, self.transform_height + grid_spacing, grid_spacing))
        ax.grid(which='both', color='lightblue', linestyle='--', linewidth=0.5)
        ax.set_xlabel("East (cm)")
        ax.set_ylabel("North (cm)")
        ax.set_title("Inflated Free Area (C-space)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if viz:
            plt.show(block=block)
        else:
            plt.close(fig)
    
    def show_global_coarse_path(self, coarse_xy: List[tuple], color: str = 'orange',
                                linewidth: float = 1.8, markersize: float = 28,
                                alpha: float = 0.85) -> None:
        """
        在实时窗口上叠加显示全局规划（已下采样）的路径：
        - 使用实线连线
        - 用圆点标注每个下采样点
        该覆盖层在 show_map_live_update 的刷新中不会被清理，可持续显示。
        """
        if self._fig is None or self._ax is None:
            self.show_map_live_begin(viz=True)
        pts = list(coarse_xy or [])
        if len(pts) == 0:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # 清理旧的覆盖层
        if self._global_line is not None:
            try:
                self._global_line.remove()
            except Exception:
                pass
            self._global_line = None
        if self._global_scatter is not None:
            try:
                self._global_scatter.remove()
            except Exception:
                pass
            self._global_scatter = None
        # 新增连线与散点
        self._global_line = self._ax.plot(xs, ys, color=color, linewidth=linewidth,
                                          alpha=alpha, zorder=2, linestyle='-')[0]
        self._global_scatter = self._ax.scatter(xs, ys, c=color, s=markersize,
                                                edgecolors='k', linewidths=0.5,
                                                alpha=alpha, zorder=3)
        # 刷新
        self._fig.canvas.draw_idle()
        try:
            self._fig.canvas.flush_events()
        except Exception:
            pass
        try:
            plt.pause(0.001)
        except Exception:
            pass
    
    def clear_global_coarse_path(self) -> None:
        """清除全局粗路径覆盖层。"""
        if self._global_line is not None:
            try:
                self._global_line.remove()
            except Exception:
                pass
            self._global_line = None
        if self._global_scatter is not None:
            try:
                self._global_scatter.remove()
            except Exception:
                pass
            self._global_scatter = None
        
    def save_fig(self, filename):
        """
        保存当前图像
        """
        if self._fig is None:
            raise RuntimeError("Figure not initialized. Call show_map_live_begin() first.")
        self._fig.savefig(filename, bbox_inches='tight')
        print(f"Figure saved to {filename}")
        
    def close_all_windows(self):
        """
        关闭所有 matplotlib 窗口
        """
        plt.close('all')
        self._fig, self._ax = None, None
        # 清理实时绘图句柄
        self._path_line = None
        if getattr(self, '_path_coll', None) is not None:
            try:
                self._path_coll.remove()
            except Exception:
                pass
        self._path_coll = None
        if getattr(self, '_veh_patches', None):
            for p in self._veh_patches:
                try:
                    p.remove()
                except Exception:
                    pass
        self._veh_patches = []
        self._veh_patch = None
        self._arrow = None
        if getattr(self, '_arrows', None):
            for a in self._arrows:
                try:
                    a.remove()
                except Exception:
                    pass
        self._arrows = []


class PoseSampler:
    def __init__(self, map: MapBase, max_attempts=1000, goal_from_list: bool = False):
        """
        vehicle_length/width: 船体矩形尺寸（单位：米）
        max_attempts: 最大尝试次数（避免死循环）
        """
        self.map = map
        self.max_attempts = max_attempts
        self.map_w = self.map.transform_width
        self.map_h = self.map.transform_height
        self.goal_from_list = goal_from_list
        if self.goal_from_list:
            self.goal_list:List = [[70, 90, np.pi], [70, 80, np.pi], [70, 70, np.pi],\
                [70, 60, np.pi], [70, 52, np.pi], [70, 42, np.pi], [70, 42, 0],\
                [70, 52, 0], [70, 60, 0], [70, 70, 0], [70, 80, 0], [70, 90, 0],\
                [55, 90, -np.pi / 2], [55, 78, -np.pi / 2], [55, 66, -np.pi / 2],\
                [55, 90, np.pi / 2], [55, 78, np.pi / 2], [55, 66, np.pi / 2],
            ]

    def _sample_pose(self):
        for _ in range(self.max_attempts):
            x = random.uniform(0, self.map_w)
            y = random.uniform(0, self.map_h)
            yaw = random.uniform(-np.pi, np.pi)
            ring = self.map.vehicle.create_vehicle_box(x, y, yaw)
            if not self.map.check_collision(ring):
                return (x, y, yaw)
        raise RuntimeError("Failed to sample a valid pose within limit")

    def sample_start_and_goal(self):
        start = self._sample_pose()
        if self.goal_from_list:
            goal = random.choice(self.goal_list)
            if isinstance(goal, list):
                goal = (goal[0], goal[1], goal[2])
        else:
            goal = self._sample_pose()
        return start, goal

if __name__ == "__main__":
    # 实例化并展示地图
    map = MapBase("./SIM.png", enable_dual_inflation=True)
    map.show_dual_inflation_masks()
    map.show_soft_penalty()
    # map = MapBase("./frame.png", real_resolution=10.0)
    sampler = PoseSampler(map, goal_from_list=True)
    (start, goal) = sampler.sample_start_and_goal()
    # start = [55, 66, - np.pi / 2]  # 船体起始位置
    # goal = [55, 54, - np.pi / 2]    # 船体目标位置
    trajectory :List = [start, goal]
    map.show_map(trajectory)
    # map.visualize_obstacles()
