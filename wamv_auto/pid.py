import time
import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target=0.0, death_zone=(None, None), output_limits=(None, None), sample_time=0.01):
        self.kp = kp                # 比例系数
        self.ki = ki                # 积分系数
        self.kd = kd                # 微分系数
        self.target = target        # 目标值
        self.current = 0.0

        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = None
        
        self.enable_death_zone = True
        self.death_zone = death_zone  # 死区范围 (最小值, 最大值)
        self.output_limits = output_limits  # (最小值, 最大值)
        self.sample_time = sample_time      # 最小时间间隔（秒）

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = None
        
    def update_target(self, target):
        self.target = target
        
    def update_current(self, current):
        self.current = current
        
    def comput_error(self, old_data, new_data):
        return new_data - old_data

    def compute(self) -> float:
        current_time = time.time()
        error = self.comput_error(self.current, self.target)

        if self._last_time is None:
            self._last_time = current_time
            return 0.0

        dt = current_time - self._last_time
        if dt < self.sample_time:
            return 0.0  # 等待足够的采样间隔

        # 死区判断：误差在死区范围内时，输出为 0
        if self.enable_death_zone:
            min_dz, max_dz = self.death_zone
            if min_dz is not None and max_dz is not None:
                if min_dz <= error <= max_dz:
                    # 在死区内，不累积积分，直接返回 0
                    self._prev_error = error
                    self._last_time = current_time
                    return 0.0
            elif min_dz is not None:
                if error >= min_dz:
                    self._prev_error = error
                    self._last_time = current_time
                    return 0.0
            elif max_dz is not None:
                if error <= max_dz:
                    self._prev_error = error
                    self._last_time = current_time
                    return 0.0

        # 积分项
        self._integral += error * dt

        # 微分项
        derivative = self.comput_error(self._prev_error, error) / dt if dt > 0 else 0.0

        # PID 输出
        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        # 限制输出
        min_out, max_out = self.output_limits
        if min_out is not None:
            output = max(min_out, output)
        if max_out is not None:
            output = min(max_out, output)

        # 更新状态
        self._prev_error = error
        self._last_time = current_time

        return output

class POS_PID(PID):
    def comput_error(self, old_data, new_data):
        # 位置环误差计算
        error = new_data - old_data
        return np.arctan2(np.sin(error), np.cos(error))  # 归一化到 [-π, π]