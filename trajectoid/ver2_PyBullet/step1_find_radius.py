"""
Step 1: Darboux Frame ODE 적분 + 최적 구 반지름 R* 탐색
========================================================

v1의 이산 회전행렬 곱(rotation_to_origin)을 연속 ODE 적분으로 대체.
Gauss-Bonnet 정리를 이용한 구면 면적 계산 + Brent 법으로 R* 결정.

핵심 수식 (trajectoid_algorithm_v2.md §2, §3):
    dp/ds = t̂
    dt̂/ds = κ_g(s)·N̂ − (1/R)·n̂
    dN̂/ds = −κ_g(s)·t̂
    S(R) = R²(2π − 2πI_T − α_A^ext − α_M^ext)
    존재 조건: S(R)/R² = ±2π/n
"""

import logging
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from tqdm import tqdm

from .config import (
    ODE_METHOD, ODE_RTOL, ODE_ATOL,
    R_SCAN_MIN, R_SCAN_MAX, R_SCAN_N,
    BRENT_XTOL, BRENT_RTOL,
)
from .paths import PathSpec

log = logging.getLogger(__name__)


# ================================================================
#  데이터 클래스
# ================================================================
@dataclass
class DarbouxSolution:
    """Darboux ODE 적분 결과."""
    s: np.ndarray          # (M,)   호장 매개변수
    p: np.ndarray          # (M, 3) 구면 곡선 위치
    t_hat: np.ndarray      # (M, 3) 접선 벡터
    N_hat: np.ndarray      # (M, 3) 측지 법선 벡터
    R: float               # 구 반지름


# ================================================================
#  Gram-Schmidt 재직교화
# ================================================================
def _orthonormalize_frame(t, N, p, R):
    """Darboux frame을 재직교화하고 구면 구속을 강제."""
    # n̂ = p/|p| (구면 구속)
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-15:
        return t, N, p
    n = p / p_norm
    # p를 정확히 구면 위에 놓기
    p = n * R

    # t̂: n̂에 수직하게 만들기
    t = t - np.dot(t, n) * n
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-15:
        return t, N, p
    t = t / t_norm

    # N̂ = n̂ × t̂ (자동으로 정규직교)
    N = np.cross(n, t)
    N_norm = np.linalg.norm(N)
    if N_norm < 1e-15:
        return t, N, p
    N = N / N_norm

    return t, N, p


# ================================================================
#  Darboux ODE 우변
# ================================================================
def _darboux_rhs(s, state, kappa_interp, R):
    """
    9성분 상태벡터 [p(3), t̂(3), N̂(3)]의 시간 도함수.

    n̂ = p/|p| 는 적분 변수가 아닌 유도량.
    """
    px, py, pz = state[0], state[1], state[2]
    tx, ty, tz = state[3], state[4], state[5]
    Nx, Ny, Nz = state[6], state[7], state[8]

    p_vec = np.array([px, py, pz])
    t_vec = np.array([tx, ty, tz])
    N_vec = np.array([Nx, Ny, Nz])

    p_norm = np.linalg.norm(p_vec)
    if p_norm < 1e-15:
        n_vec = np.array([0.0, 0.0, -1.0])
    else:
        n_vec = p_vec / p_norm

    kg = float(kappa_interp(s))

    dp = t_vec
    dt = kg * N_vec - (1.0 / R) * n_vec
    dN = -kg * t_vec

    return np.concatenate([dp, dt, dN])


# ================================================================
#  Darboux ODE 적분
# ================================================================
def integrate_darboux(path: PathSpec, R: float,
                      psi0: float = 0.0,
                      n_eval: int = 0) -> DarbouxSolution:
    """
    Darboux Frame ODE를 [0, L]에서 적분.

    Parameters
    ----------
    path : PathSpec
    R : 구 반지름
    psi0 : 초기 접선 방향 (rad)
    n_eval : dense_output 평가점 수 (0이면 솔버 자체 스텝만 사용)

    Returns
    -------
    DarbouxSolution
    """
    kappa_interp = path.kappa_func()
    L = path.L

    # 초기 조건: 구의 남극에서 출발
    p0 = np.array([0.0, 0.0, -R])
    t0 = np.array([np.cos(psi0), np.sin(psi0), 0.0])
    N0 = np.array([-np.sin(psi0), np.cos(psi0), 0.0])
    y0 = np.concatenate([p0, t0, N0])

    if n_eval > 0:
        t_eval = np.linspace(0, L, n_eval)
    else:
        t_eval = None

    sol = solve_ivp(
        fun=lambda s, y: _darboux_rhs(s, y, kappa_interp, R),
        t_span=(0, L),
        y0=y0,
        method=ODE_METHOD,
        rtol=ODE_RTOL,
        atol=ODE_ATOL,
        t_eval=t_eval,
        dense_output=(n_eval == 0),
        max_step=L / 100,  # 최대 스텝 제한으로 정밀도 보장
    )

    if not sol.success:
        raise RuntimeError(f"Darboux ODE 적분 실패: {sol.message}")

    s_arr = sol.t
    y_arr = sol.y.T  # (M, 9)

    # 프레임 재직교화
    p_arr = np.zeros((len(s_arr), 3))
    t_arr = np.zeros((len(s_arr), 3))
    N_arr = np.zeros((len(s_arr), 3))

    for i in range(len(s_arr)):
        p_i = y_arr[i, 0:3]
        t_i = y_arr[i, 3:6]
        N_i = y_arr[i, 6:9]
        t_i, N_i, p_i = _orthonormalize_frame(t_i, N_i, p_i, R)
        p_arr[i] = p_i
        t_arr[i] = t_i
        N_arr[i] = N_i

    return DarbouxSolution(s=s_arr, p=p_arr, t_hat=t_arr, N_hat=N_arr, R=R)


# ================================================================
#  대원호(great circle arc) 접선 계산
# ================================================================
def _great_circle_tangent(A, B):
    """A에서 B로 향하는 대원호의 단위 접선 벡터."""
    # A에서 B까지의 방향을 A의 접평면에 사영
    t = B - np.dot(A, B) * A
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-15:
        return np.zeros(3)
    return t / t_norm


# ================================================================
#  외각 계산
# ================================================================
def _exterior_angle(t_curve, t_arc, n_at_point):
    """
    구면 위 한 점에서 곡선 접선과 대원호 접선 사이의 외각.

    접평면에서의 부호 있는 각도를 계산한 후 외각으로 변환.
    """
    # 두 벡터가 접평면에 있으므로 내적으로 각도 계산
    cos_ang = np.clip(np.dot(t_curve, t_arc), -1, 1)
    # 부호: 법선과의 외적으로 결정
    cross = np.cross(t_curve, t_arc)
    sin_ang = np.dot(cross, n_at_point)
    angle = np.arctan2(sin_ang, cos_ang)  # 내각 (접선에서 호까지)
    # 외각 = π - 내각
    return np.pi - angle


# ================================================================
#  Gauss-Bonnet 구면 면적 계산
# ================================================================
def compute_gauss_bonnet_area(sol: DarbouxSolution,
                              path: PathSpec) -> float:
    """
    S(R) = R²(2π − 2πI_T − α_A^ext − α_M^ext)

    §3.2: 곡선 + 대원호로 둘러싸인 구면 영역의 부호 면적.
    I_T는 회전수(turning number).
    """
    R = sol.R
    A = sol.p[0]    # 시작점
    M = sol.p[-1]   # 끝점

    # 회전수
    I_T = path.turning_number()
    I_T_rounded = round(I_T)

    # A와 M이 너무 가까우면 면적 ≈ 0
    AM_dist = np.linalg.norm(A - M)
    if AM_dist < 1e-12 * R:
        return R**2 * (2 * np.pi - 2 * np.pi * I_T_rounded)

    # 대원호의 접선 (A에서, M에서)
    t_arc_at_A = _great_circle_tangent(A / R, M / R)  # 단위 구면에서 계산
    t_arc_at_M = _great_circle_tangent(M / R, A / R)  # M→A 방향

    # 곡선의 접선 (A에서, M에서)
    t_curve_at_A = sol.t_hat[0]
    t_curve_at_M = sol.t_hat[-1]

    # 법선 (단위 구면)
    n_A = A / R
    n_M = M / R

    # 외각
    alpha_A = _exterior_angle(t_curve_at_A, t_arc_at_A, n_A)
    alpha_M = _exterior_angle(-t_arc_at_M, -t_curve_at_M, n_M)

    S = R**2 * (2 * np.pi - 2 * np.pi * I_T_rounded - alpha_A - alpha_M)
    return S


# ================================================================
#  면적 잔차 함수
# ================================================================
def _area_residual(R, path, n, sign=+1):
    """
    f(R) = S(R)/R² − sign·2π/n

    이 함수의 근이 period-n trajectoid의 존재 조건.
    """
    try:
        sol = integrate_darboux(path, R, psi0=0.0)
        S = compute_gauss_bonnet_area(sol, path)
        return S / R**2 - sign * 2 * np.pi / n
    except Exception:
        return np.nan


# ================================================================
#  R* 탐색
# ================================================================
def find_optimal_radius(path: PathSpec, n: int = 2,
                        R_min: float = R_SCAN_MIN,
                        R_max: float = R_SCAN_MAX,
                        n_scan: int = R_SCAN_N) -> float:
    """
    Brent 법으로 최적 구 반지름 R*를 찾는다.

    1) [R_min, R_max]에서 f(R)를 스캔하여 부호 변화 구간 탐지
    2) 첫 번째 부호 변화 구간에서 brentq로 정밀 근 결정

    Parameters
    ----------
    path : PathSpec
    n : period (2 = TPT, ≥3 = MPT)
    R_min, R_max : 스캔 범위
    n_scan : 스캔 샘플 수

    Returns
    -------
    R_star : float — 최적 반지름
    """
    log.info(f"R* 탐색 시작: n={n}, R∈[{R_min:.3f}, {R_max:.3f}], 스캔 {n_scan}개")

    R_values = np.linspace(R_min, R_max, n_scan)
    f_values = np.full(n_scan, np.nan)

    # 양/음 부호 모두 시도
    for sign in [+1, -1]:
        for i in tqdm(range(n_scan), desc=f'  R 스캔 (sign={sign:+d})',
                      unit='pt', ncols=72):
            f_values[i] = _area_residual(R_values[i], path, n, sign)

        # 부호 변화 구간 찾기
        valid = ~np.isnan(f_values)
        for i in range(n_scan - 1):
            if valid[i] and valid[i + 1]:
                if f_values[i] * f_values[i + 1] < 0:
                    R_a, R_b = R_values[i], R_values[i + 1]
                    log.info(f"  부호 변화 발견: R∈[{R_a:.6f}, {R_b:.6f}]"
                             f" (sign={sign:+d})")
                    try:
                        R_star = brentq(
                            lambda r: _area_residual(r, path, n, sign),
                            R_a, R_b,
                            xtol=BRENT_XTOL,
                            rtol=BRENT_RTOL,
                        )
                        log.info(f"  ★ R* = {R_star:.10f} (sign={sign:+d})")
                        return R_star
                    except ValueError:
                        continue

    raise RuntimeError(
        f"R* 를 찾지 못했습니다. "
        f"R∈[{R_min}, {R_max}] 범위를 확장하거나 경로를 확인하세요."
    )


# ================================================================
#  편의 함수: 모든 R에 대한 면적 스캔 (디버깅/시각화용)
# ================================================================
def scan_area_function(path: PathSpec, n: int = 2,
                       R_min: float = R_SCAN_MIN,
                       R_max: float = R_SCAN_MAX,
                       n_scan: int = 200):
    """
    디버깅용: f(R) = S(R)/R² − 2π/n 을 스캔하여 반환.

    Returns
    -------
    R_values, f_plus, f_minus : 각각 (n_scan,) ndarray
    """
    R_values = np.linspace(R_min, R_max, n_scan)
    f_plus = np.full(n_scan, np.nan)
    f_minus = np.full(n_scan, np.nan)

    for i in tqdm(range(n_scan), desc='  면적 함수 스캔', ncols=72):
        f_plus[i] = _area_residual(R_values[i], path, n, +1)
        f_minus[i] = _area_residual(R_values[i], path, n, -1)

    return R_values, f_plus, f_minus
