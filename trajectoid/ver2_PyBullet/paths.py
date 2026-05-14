"""
경로 입력 모듈
==============
2D 주기 경로를 생성하고 호장(arc-length) 매개변수화 + 곡률 κ(s) 계산.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


class PathSpec:
    """
    호장(arc-length) 매개변수화된 2D 주기 경로.

    Attributes
    ----------
    xy : (N, 2) ndarray — 경로 점 좌표
    s  : (N,) ndarray   — 누적 호장 매개변수
    kappa : (N,) ndarray — 부호 있는 곡률 κ(s)
    psi : (N,) ndarray   — 접선 방향 각도 ψ(s)
    L  : float           — 한 주기 총 호장
    """

    def __init__(self, xy: np.ndarray):
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be (N, 2) array")
        if len(xy) < 4:
            raise ValueError("Need at least 4 points for curvature computation")

        self.xy = xy
        self._compute_arclength()
        self._compute_tangent_and_curvature()

    # ── 내부 계산 ─────────────────────────────────────────────

    def _compute_arclength(self):
        diffs = np.diff(self.xy, axis=0)
        ds = np.linalg.norm(diffs, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.L = self.s[-1]

    def _compute_tangent_and_curvature(self):
        x, y = self.xy[:, 0], self.xy[:, 1]
        n = len(x)

        # Savitzky-Golay 필터로 안정적인 미분 (최소 window 5)
        win = min(max(n // 10 | 1, 5), n - (1 if n % 2 == 0 else 0))
        if win % 2 == 0:
            win -= 1
        win = max(win, 5)
        if win > n:
            win = n if n % 2 == 1 else n - 1

        dx = savgol_filter(x, window_length=win, polyorder=3, deriv=1,
                           delta=self.L / (n - 1))
        dy = savgol_filter(y, window_length=win, polyorder=3, deriv=1,
                           delta=self.L / (n - 1))
        ddx = savgol_filter(x, window_length=win, polyorder=3, deriv=2,
                            delta=self.L / (n - 1))
        ddy = savgol_filter(y, window_length=win, polyorder=3, deriv=2,
                            delta=self.L / (n - 1))

        # 접선 각도 ψ(s)
        self.psi = np.arctan2(dy, dx)

        # 부호 있는 곡률: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        speed_sq = dx**2 + dy**2
        speed_32 = np.power(speed_sq, 1.5)
        speed_32[speed_32 < 1e-30] = 1e-30  # 0 나누기 방지
        self.kappa = (dx * ddy - dy * ddx) / speed_32

    # ── 보간된 곡률 함수 ───────────────────────────────────────

    def kappa_func(self):
        """s → κ(s) 보간 함수 반환 (scipy CubicSpline)."""
        return CubicSpline(self.s, self.kappa, extrapolate=True)

    def psi_func(self):
        """s → ψ(s) 보간 함수 반환."""
        return CubicSpline(self.s, self.psi, extrapolate=True)

    # ── 등간격 호장 리샘플링 ───────────────────────────────────

    def resample(self, n_points: int) -> 'PathSpec':
        """등간격 호장으로 리샘플링."""
        s_new = np.linspace(0, self.L, n_points)
        cs_x = CubicSpline(self.s, self.xy[:, 0])
        cs_y = CubicSpline(self.s, self.xy[:, 1])
        xy_new = np.stack([cs_x(s_new), cs_y(s_new)], axis=1)
        return PathSpec(xy_new)

    # ── 회전수(turning number) ────────────────────────────────

    def turning_number(self) -> float:
        """I_T = (1/2π) ∫₀ᴸ κ(s) ds"""
        return np.trapz(self.kappa, self.s) / (2 * np.pi)

    # ── 팩토리 메서드 ─────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str, scale: float = 1.0,
                    n_points: int = 2000) -> 'PathSpec':
        """
        사전 정의된 경로 생성.

        Parameters
        ----------
        name : 'sinusoid' | 'figure8' | 'lemniscate' | 'circle' | 'zigzag'
        scale : 경로 크기 스케일
        n_points : 점 수
        """
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        if name == 'sinusoid':
            x = np.linspace(-scale, scale, n_points)
            y = scale * 0.6 * np.sin(3 * np.pi * x / scale)

        elif name == 'figure8':
            x = scale * np.sin(t)
            y = scale * np.sin(t) * np.cos(t)

        elif name == 'lemniscate':
            a = scale
            den = 1 + np.sin(t)**2
            x = a * np.cos(t) / den
            y = a * np.sin(t) * np.cos(t) / den

        elif name == 'circle':
            r = scale * 0.85
            x = r * np.cos(t)
            y = r * np.sin(t)

        elif name == 'zigzag':
            x = np.linspace(-scale, scale, n_points)
            y = scale * 0.4 * np.abs(np.mod(x / scale * 4 + 1, 2) - 1) - scale * 0.2

        else:
            raise ValueError(f"Unknown preset: '{name}'")

        return cls(np.stack([x, y], axis=1))

    @classmethod
    def from_numpy(cls, xy: np.ndarray) -> 'PathSpec':
        """(N, 2) numpy 배열로부터 생성."""
        return cls(xy)
