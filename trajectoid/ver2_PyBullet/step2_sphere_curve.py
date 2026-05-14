"""
Step 2: R*에서 구면 곡선 전체 프로파일 생성
===========================================

Step 1에서 결정된 R*로 Darboux ODE를 밀도 높게 재적분하여
p(s), t̂(s), N̂(s)의 전체 프로파일을 획득.

v1 대체 대상: compute_trajectoid.py:trace_on_sphere() (342행)
"""

import logging
import numpy as np

from .config import N_S_SAMPLES
from .paths import PathSpec
from .step1_find_radius import DarbouxSolution, integrate_darboux

log = logging.getLogger(__name__)


def generate_sphere_curve(path: PathSpec, R_star: float,
                          psi0: float = 0.0,
                          n_points: int = N_S_SAMPLES) -> DarbouxSolution:
    """
    R = R*에서 밀도 높은 Darboux ODE 적분 수행.

    Parameters
    ----------
    path : PathSpec — 입력 경로
    R_star : float — 최적 반지름 (Step 1 결과)
    psi0 : float — 초기 접선 방향 (rad)
    n_points : int — 출력 점 수

    Returns
    -------
    DarbouxSolution — 구면 곡선 전체 프로파일
    """
    log.info(f"구면 곡선 생성: R* = {R_star:.8f}, {n_points} 점")

    sol = integrate_darboux(path, R_star, psi0=psi0, n_eval=n_points)

    # 검증: 모든 점이 반지름 R 구면 위에 있는지
    radii = np.linalg.norm(sol.p, axis=1)
    max_dev = np.max(np.abs(radii - R_star))
    log.info(f"  구면 구속 최대 편차: {max_dev:.2e}")
    if max_dev > 1e-6 * R_star:
        log.warning(f"  경고: 구면 편차가 큼 ({max_dev:.2e}). "
                    f"ODE 정밀도를 높이세요.")

    # 검증: 프레임 직교성
    t_dot_N = np.abs(np.sum(sol.t_hat * sol.N_hat, axis=1))
    max_non_orth = np.max(t_dot_N)
    log.info(f"  프레임 직교성 최대 편차: {max_non_orth:.2e}")

    # 끝점 정보
    A = sol.p[0]
    M = sol.p[-1]
    AM_dist = np.linalg.norm(A - M)
    AM_arc = np.arccos(np.clip(np.dot(A, M) / R_star**2, -1, 1)) * R_star
    log.info(f"  A = {A}")
    log.info(f"  M = {M}")
    log.info(f"  |AM| 유클리드 = {AM_dist:.6f}, 대원호 = {AM_arc:.6f}")

    return sol
