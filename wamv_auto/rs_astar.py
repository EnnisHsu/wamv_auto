import math
import heapq
import time
from typing import List, Tuple, Dict, Optional

from wamv_auto.map import MapBase
from wamv_auto.reeds_shepp import plan_rs_path


class RSAStarPlanner:
    """
    Hybrid A* with Reeds–Shepp terminal connection.
    单位沿用 MapBase（厘米）。
    - 使用粗粒度运动原语扩展（前/后+转向弧段）。
    - 启发优先用 RS 长度，失败回退欧氏距离。
    - 近端触发 RS 终端连接，做碰撞校验后快速收敛。
    """

    def __init__(
        self,
        map_obj: MapBase,
        max_curvature: float = 1.0,
        step_len: float = 2.0,
        yaw_bins: int = 36,
        substeps: int = 5,
        allow_reverse: bool = True,
        reverse_penalty: float = 1.2,
        rs_connect_alpha: float = 10.0,
        viz_live: bool = False,
        viz_every: int = 60,
        timeout: int = 60,
        use_rs_heuristic: bool = False,
        rs_heuristic_trigger_ratio: float = 2.0,   # 相对 alpha*Rmin 的比例
        rs_connect_every: int = 10,                # 远端每N步触发一次RS连接
        goal_xy_tol: Optional[float] = None,        # 位置收敛容差（None -> 1.5*step_len）
        goal_yaw_tol: float = math.radians(10.0),   # 航向收敛容差
        max_turn_per_step_rad: float = 0.35,        # 单步最大转角限幅（~20°）
        penalty_weight: float = 0.0,                # 软限制惩罚的权重（cm*权重 叠加到代价）
    ) -> None:
        # 基本参数
        self.map = map_obj
        self.kappa_max = float(max_curvature)
        self.step_len = float(step_len)
        self.substeps = max(2, int(substeps))
        self.allow_reverse = bool(allow_reverse)
        self.reverse_penalty = float(reverse_penalty)
        self.alpha = float(rs_connect_alpha)
        self.viz_live = bool(viz_live)
        self.viz_every = max(1, int(viz_every))
        self.timeout = int(timeout)
        self.use_rs_heuristic = bool(use_rs_heuristic)
        self.rs_h_trigger_ratio = float(rs_heuristic_trigger_ratio)
        self.rs_connect_every = max(1, int(rs_connect_every))

        # 航向量化
        self.yaw_bins = max(8, int(yaw_bins))
        self.dtheta = 2.0 * math.pi / self.yaw_bins

        # 地图尺寸
        self.W = float(getattr(self.map, "transform_width", 100.0))
        self.H = float(getattr(self.map, "transform_height", 100.0))

        # 启发缓存（按量化键存储）
        self._h_cache: Dict[Tuple[int, int, int], float] = {}

        # 终止与原语限幅
        self.goal_xy_tol = (1.5 * self.step_len) if goal_xy_tol is None else float(goal_xy_tol)
        self.goal_yaw_tol = float(goal_yaw_tol)
        # 默认按曲率给出理论值后限幅
        self.dtheta_max = min(self.kappa_max * self.step_len, float(max_turn_per_step_rad))

        # 软惩罚
        self.penalty_weight = float(penalty_weight)
        self._penalty_map = getattr(self.map, 'soft_penalty_map', None)

    # ---------- 基础工具 ----------
    def _normalize_angle(self, th: float) -> float:
        return (th + math.pi) % (2.0 * math.pi) - math.pi

    def _yaw_to_bin(self, yaw: float) -> int:
        t = (yaw % (2.0 * math.pi)) / self.dtheta
        return int(round(t)) % self.yaw_bins

    def _in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.W and 0.0 <= y <= self.H

    def _is_pose_free(self, x: float, y: float, yaw: float) -> bool:
        # 优先用 MapBase.is_pose_traversable，若无则回退碰撞检测
        if hasattr(self.map, 'is_pose_traversable'):
            try:
                return bool(self.map.is_pose_traversable(x, y, yaw))
            except Exception:
                pass
        try:
            # 回退：直接检查车辆盒与障碍是否相交
            ring = self.map.vehicle.create_vehicle_box(x, y, yaw)
            return not self.map.check_collision(ring)
        except Exception:
            return False

    def _is_segment_free(self, x: float, y: float, yaw: float, dtheta: float, travel: float) -> bool:
        # 沿弧段做子步采样
        for i in range(1, self.substeps + 1):
            t = i / self.substeps
            yaw_i = yaw + t * dtheta
            xi = x + t * travel * math.cos(yaw + 0.5 * dtheta)
            yi = y + t * travel * math.sin(yaw + 0.5 * dtheta)
            if not self._in_bounds(xi, yi):
                return False
            if not self._is_pose_free(xi, yi, yaw_i):
                return False
        return True

    def _propagate(self, x: float, y: float, yaw: float, dtheta: float, travel: float) -> Tuple[float, float, float]:
        nx = x + travel * math.cos(yaw + 0.5 * dtheta)
        ny = y + travel * math.sin(yaw + 0.5 * dtheta)
        nyaw = self._normalize_angle(yaw + dtheta)
        return nx, ny, nyaw

    # ---------- 软惩罚采样 ----------
    def _penalty_at(self, x: float, y: float) -> float:
        if self._penalty_map is None:
            return 0.0
        if not hasattr(self.map, 'world_to_rc'):
            return 0.0
        try:
            r, c = self.map.world_to_rc(x, y)
            H, W = self._penalty_map.shape
            if r < 0 or c < 0 or r >= H or c >= W:
                return 0.0
            return float(self._penalty_map[r, c])
        except Exception:
            return 0.0

    def _segment_penalty(self, x: float, y: float, yaw: float, dtheta: float, travel: float) -> float:
        if self.penalty_weight <= 1e-9:
            return 0.0
        acc = 0.0
        n = self.substeps
        for i in range(1, n + 1):
            t = i / n
            xi = x + t * travel * math.cos(yaw + 0.5 * dtheta)
            yi = y + t * travel * math.sin(yaw + 0.5 * dtheta)
            acc += self._penalty_at(xi, yi)
        avg_p = acc / n
        # 乘以段长度使之同 path length 同量纲
        return self.penalty_weight * abs(travel) * avg_p

    def _heuristic(self, s: Tuple[float, float, float], g: Tuple[float, float, float]) -> float:
        # 使用缓存（对连续状态做简单量化）
        ix = int(round(s[0] / max(1.0, self.step_len)))
        iy = int(round(s[1] / max(1.0, self.step_len)))
        it = self._yaw_to_bin(s[2])
        key = (ix, iy, it)
        if key in self._h_cache:
            return self._h_cache[key]

        dist = math.hypot(g[0] - s[0], g[1] - s[1])
        h = dist
        if self.use_rs_heuristic:
            # 仅在足够接近时才调用昂贵的 RS
            rmin = 1.0 / max(1e-6, self.kappa_max)
            trigger = max(self.alpha * rmin * self.rs_h_trigger_ratio, 3.0 * rmin)
            if dist < trigger:
                try:
                    rs = plan_rs_path(self.map, s, g, maxc=self.kappa_max, step_size=self.step_len, penalty_weight=self.penalty_weight)
                    if rs and len(rs) >= 2:
                        h = min(h, sum(math.hypot(x2 - x1, y2 - y1)
                                       for (x1, y1, _), (x2, y2, _) in zip(rs[:-1], rs[1:])))
                except Exception:
                    pass
        self._h_cache[key] = h
        return h

    def _attempt_rs_connect(self, cur: Tuple[float, float, float], goal: Tuple[float, float, float], step_idx: int = 0) -> Optional[List[Tuple[float, float, float]]]:
        # 距离触发
        dist = math.hypot(goal[0] - cur[0], goal[1] - cur[1])
        rmin = 1.0 / max(1e-6, self.kappa_max)
        near_thresh = self.alpha * rmin
        if dist > near_thresh and (step_idx % self.rs_connect_every != 0):
            return None
        try:
            rs_path = plan_rs_path(self.map, cur, goal, maxc=self.kappa_max, step_size=self.step_len, penalty_weight=self.penalty_weight)
        except Exception:
            return None
        if not rs_path:
            return None
        # 碰撞校验
        for x, y, yaw in rs_path:
            if not self._in_bounds(x, y) or not self._is_pose_free(x, y, yaw):
                return None
        return rs_path

    # ---------- 主规划 ----------
    def plan(self, start: Tuple[float, float, float], goal: Tuple[float, float, float], return_with_direction: bool = False) -> Optional[List[Tuple[float, float, float]]]:
        sx, sy, syaw = start
        gx, gy, gyaw = goal
        if not self._in_bounds(sx, sy) or not self._in_bounds(gx, gy):
            return None
        if not self._is_pose_free(sx, sy, syaw):
            return None
        yaw_set = [-self.dtheta_max, 0.0, self.dtheta_max]
        start_node = {'state': start, 'g': 0.0, 'h': self._heuristic(start, goal), 'parent': None, 'dir': 0}
        # open 用最小堆（f, cnt, node_dict）
        open_heap: List[Tuple[float, int, dict]] = []
        cnt = 0
        heapq.heappush(open_heap, (start_node['g'] + start_node['h'], cnt, start_node))
        closed: set[Tuple[int, int, int]] = set()
        # g-cost 剪枝
        g_cost: Dict[Tuple[int, int, int], float] = {}

        def key_of(s: Tuple[float, float, float]) -> Tuple[int, int, int]:
            xb = int(round(s[0]))
            yb = int(round(s[1]))
            tb = self._yaw_to_bin(s[2])
            return xb, yb, tb

        steps = 0
        st_time = time.time()
        while open_heap:
            if (time.time() - st_time) > self.timeout:
                print(f"Timeout after {self.timeout} seconds and {steps} expansions")
                return None
            _, _, node = heapq.heappop(open_heap)
            s = node['state']
            k = key_of(s)
            if k in closed:
                continue
            closed.add(k)
            g_cost[k] = node['g']

            # 直接到达判据（不依赖 RS 终端连接）
            dist_goal = math.hypot(gx - s[0], gy - s[1])
            yaw_diff = abs((s[2] - gyaw + math.pi) % (2.0 * math.pi) - math.pi)
            if dist_goal <= self.goal_xy_tol and yaw_diff <= self.goal_yaw_tol:
                rec = self._reconstruct(node, with_dir=return_with_direction)
                print(f"Goal reached after {steps} expansions, path len: {len(rec)}")
                return rec

            # 近端 RS 尝试
            rs_conn = self._attempt_rs_connect(s, goal, steps)
            if rs_conn:
                base_path = self._reconstruct(node, with_dir=return_with_direction)
                if return_with_direction:
                    # 追加 RS 连接段，统一标记为前进(+1)（Reeds–Shepp 反向段未细分；可后续扩展）
                    rs_ext = [(x, y, yaw, +1) for (x, y, yaw) in rs_conn[1:]]  # 避免重复当前节点
                    full_path = base_path + rs_ext
                else:
                    full_path = base_path + rs_conn[1:]
                print(f"RS connection succeeded after {steps} expansions, path len: {len(full_path)}")
                return full_path

            # 扩展
            for dyaw in yaw_set:
                # 前/后
                for_signs = (1, -1) if self.allow_reverse else (1,)
                for sign in for_signs:
                    travel = sign * self.step_len
                    # dtheta ≈ kappa * ds，绑定方向符号，使后退保持转向一致（临近策略）
                    dtheta = math.copysign(abs(dyaw), travel)
                    if not self._is_segment_free(s[0], s[1], s[2], dtheta, travel):
                        continue
                    ns = self._propagate(s[0], s[1], s[2], dtheta, travel)
                    if not self._in_bounds(ns[0], ns[1]) or not self._is_pose_free(*ns):
                        continue
                    nk = key_of(ns)
                    if nk in closed:
                        continue
                    base_cost = abs(travel)
                    if travel < 0:
                        base_cost *= self.reverse_penalty
                    # 仅将软限制惩罚与路径长度相加，不受后退惩罚缩放
                    penalty_cost = self._segment_penalty(s[0], s[1], s[2], dtheta, travel)
                    step_cost = base_cost + penalty_cost
                    new_g = node['g'] + step_cost
                    # g-cost 剪枝：若非更优则丢弃
                    old_g = g_cost.get(nk, float('inf'))
                    if new_g >= old_g:
                        continue
                    child = {
                        'state': ns,
                        'g': new_g,
                        'h': self._heuristic(ns, goal),
                        'parent': node,
                        'dir': sign,
                    }
                    cnt += 1
                    heapq.heappush(open_heap, (child['g'] + child['h'], cnt, child))

            steps += 1
            if self.viz_live and hasattr(self.map, 'show_map_live_update') and steps % self.viz_every == 0:
                try:
                    self.map.show_map_live_update(self._reconstruct(node))
                except Exception:
                    pass

        return None

    def _reconstruct(self, node: dict, with_dir: bool = False) -> List[Tuple[float, float, float]]:
        """回溯路径；with_dir=True 时返回 (x,y,yaw,dir)。起点 dir 可能为 0（未知），将前向传播修正为首个非零方向。"""
        raw: List[Tuple[float, float, float, int]] = []
        n = node
        while n is not None:
            d = n.get('dir', 0)
            sx, sy, syaw = n['state']
            raw.append((sx, sy, syaw, d))
            n = n['parent']
        raw.reverse()
        if not with_dir:
            return [(x, y, yaw) for (x, y, yaw, _) in raw]
        # 方向修正：若起点为0，用后续第一个非零方向填充；其余仍为0的延续前一方向
        first_non_zero = next((d for (_, _, _, d) in raw if d != 0), +1)
        fixed: List[Tuple[float, float, float, int]] = []
        cur_dir = first_non_zero
        for (x, y, yaw, d) in raw:
            if d != 0:
                cur_dir = d
            fixed.append((x, y, yaw, cur_dir))
        return fixed  # type: ignore[return-value]


def rs_astar(map_obj: MapBase, start: Tuple[float, float, float], goal: Tuple[float, float, float],
             max_curvature: float = 1.0, step_size: float = 2.0, alpha: float = 10.0,
             yaw_bins: int = 36, substeps: int = 5, allow_reverse: bool = True,
             reverse_penalty: float = 1.2, viz_live: bool = False, viz_every: int = 60,
             penalty_weight: float = 0.0) -> Optional[List[Tuple[float, float, float]]]:
    """便捷函数：保持与旧版 API 兼容。
    - 参数名基本沿用旧 rs_astar()；内部转为 RSAStarPlanner 并调用 plan。
    """
    planner = RSAStarPlanner(
        map_obj,
        max_curvature=max_curvature,
        step_len=step_size,
        yaw_bins=yaw_bins,
        substeps=substeps,
        allow_reverse=allow_reverse,
        reverse_penalty=reverse_penalty,
        rs_connect_alpha=alpha,
        viz_live=viz_live,
    viz_every=viz_every,
    penalty_weight=penalty_weight,
    )
    return planner.plan(start, goal, return_with_direction=False)


def rs_astar_with_dir(map_obj: MapBase, start: Tuple[float, float, float], goal: Tuple[float, float, float],
                      max_curvature: float = 1.0, step_size: float = 2.0, alpha: float = 10.0,
                      yaw_bins: int = 36, substeps: int = 5, allow_reverse: bool = True,
                      reverse_penalty: float = 1.2, viz_live: bool = False, viz_every: int = 60, timeout: int = 60,
                      penalty_weight: float = 0.0) -> Optional[List[Tuple[float, float, float, int]]]:
    """与 rs_astar 相同，但返回带方向符号的路径列表 (x,y,yaw,dir)。dir=+1 前进，-1 倒车。"""
    planner = RSAStarPlanner(
        map_obj,
        max_curvature=max_curvature,
        step_len=step_size,
        yaw_bins=yaw_bins,
        substeps=substeps,
        allow_reverse=allow_reverse,
        reverse_penalty=reverse_penalty,
        rs_connect_alpha=alpha,
        viz_live=viz_live,
        viz_every=viz_every,
        timeout=timeout,
        penalty_weight=penalty_weight,
    )
    return planner.plan(start, goal, return_with_direction=True)


if __name__ == "__main__":
    # 轻量测试入口：与 astar.py 的 demo 风格一致
    from map import PoseSampler

    mp = MapBase("./SIM.png",enable_dual_inflation=True)
    sampler = PoseSampler(mp, goal_from_list=True)
    # 为保证可运行性，若 PoseSampler 未实现，则使用一组静态点
    try:
        start, goal = sampler.sample_start_and_goal()
    except Exception:
        start, goal = (51.0, 25.0, -1.40), (55.0, 90.0, 1.57)
    print(f"Start: {start}, Goal: {goal}")
    
    planner = RSAStarPlanner(mp, max_curvature=0.05, step_len=2.0, viz_live=True, viz_every=40,
                             penalty_weight=0.5)
    mp.show_map_live_begin(viz=True) if hasattr(mp, 'show_map_live_begin') else None
    st_time = time.time()
    path = planner.plan(start, goal, return_with_direction=True)
    if path:
        print(f"RSA* path len: {len(path)}, time_cost: {time.time() - st_time:.2f} s")
        print(f"RSA* path: {path}")
        if hasattr(mp, 'show_map_live_update'):
            try:
                mp.show_map_live_update(path)
            except Exception:
                pass
        if hasattr(mp, 'save_fig'):
            try:
                mp.save_fig("./img/rs_astar_demo.png")
            except Exception:
                pass
    else:
        print("RSA* failed to find a path")
        if hasattr(mp, 'show_map'):
            try:
                mp.show_map([start, goal])
            except Exception:
                pass
    if hasattr(mp, 'show_map_live_end'):
        try:
            mp.show_map_live_end()
        except Exception:
            pass