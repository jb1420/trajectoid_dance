"""
Step 4: Ruled Surface 점 구름 생성
===================================

폐합된 구면 곡선의 각 점에서 측지 법선 N̂ 방향으로 모선(ruling line)을
확장하여 trajectoid 표면의 점 구름을 생성.

핵심 수식 (trajectoid_algorithm_v2.md §5):
    X(s, λ) = p(s) + λ · N̂(s)
    |λ| ≤ λ_max = √(R_shell² − R²)

v1에는 없는 완전 신규 접근법 (v1은 boolean subtraction 사용).
"""

import logging
import numpy as np

from .config import N_S_SAMPLES, N_LAMBDA_SAMPLES, R_SHELL_RATIO
from .step3_close_curve import ClosedCurve

log = logging.getLogger(__name__)


def generate_ruled_surface(closed: ClosedCurve,
                           R_shell_ratio: float = R_SHELL_RATIO,
                           n_s: int = N_S_SAMPLES,
                           n_lambda: int = N_LAMBDA_SAMPLES) -> np.ndarray:
    """
    Ruled surface 점 구름 생성.

    Parameters
    ----------
    closed : ClosedCurve — 폐합된 구면 곡선 + frame
    R_shell_ratio : float — R_shell / R 비율 (기본 1.25)
    n_s : int — 곡선 방향 샘플 수
    n_lambda : int — 모선 방향 샘플 수

    Returns
    -------
    points : (n_s × n_lambda, 3) ndarray — 점 구름
    """
    R = closed.R
    R_shell = R * R_shell_ratio

    lambda_max = np.sqrt(R_shell**2 - R**2)
    log.info(f"Ruled surface 생성: R={R:.6f}, R_shell={R_shell:.6f}, "
             f"λ_max={lambda_max:.6f}")
    log.info(f"  샘플: {n_s} × {n_lambda} = {n_s * n_lambda:,} 점")

    # 곡선 리샘플링 (등간격)
    M = len(closed.p)
    if M >= n_s:
        # 다운샘플
        indices = np.linspace(0, M - 1, n_s, dtype=int)
    else:
        # 이미 충분한 점이 있으면 전부 사용
        indices = np.arange(M)
        n_s = M

    p_sampled = closed.p[indices]       # (n_s, 3)
    N_sampled = closed.N_hat[indices]   # (n_s, 3)

    # 모선 방향 λ 값
    lambdas = np.linspace(-lambda_max, lambda_max, n_lambda)

    # 점 구름 생성: X(s, λ) = p(s) + λ · N̂(s)
    # 벡터화 계산
    # p_sampled: (n_s, 3), lambdas: (n_lambda,)
    # -> points: (n_s, n_lambda, 3)
    points = (p_sampled[:, np.newaxis, :]
              + lambdas[np.newaxis, :, np.newaxis] * N_sampled[:, np.newaxis, :])

    points = points.reshape(-1, 3)

    log.info(f"  점 구름 생성 완료: {len(points):,} 점")
    log.info(f"  범위: x∈[{points[:,0].min():.4f}, {points[:,0].max():.4f}]"
             f" y∈[{points[:,1].min():.4f}, {points[:,1].max():.4f}]"
             f" z∈[{points[:,2].min():.4f}, {points[:,2].max():.4f}]")

    return points


def generate_ruled_surface_with_caps(closed: ClosedCurve,
                                     R_shell_ratio: float = R_SHELL_RATIO,
                                     n_s: int = N_S_SAMPLES,
                                     n_lambda: int = N_LAMBDA_SAMPLES,
                                     n_cap: int = 50) -> np.ndarray:
    """
    Ruled surface + 구면 캡 점을 추가하여 convex hull 품질 향상.

    구면 곡선에서 먼 영역(극 근처)에 추가 점을 배치하여
    convex hull이 구면 형상을 더 잘 보존하도록 함.
    """
    R = closed.R
    R_shell = R * R_shell_ratio

    # 기본 ruled surface 점
    points = generate_ruled_surface(closed, R_shell_ratio, n_s, n_lambda)

    # 구면 캡: R_shell 크기의 구면 위에 균일 분포 점 추가
    # 이코사면체 기반 구면 점 생성
    phi_cap = np.linspace(0, 2 * np.pi, n_cap, endpoint=False)
    theta_cap = np.linspace(0.1, np.pi - 0.1, n_cap // 2)
    cap_points = []
    for theta in theta_cap:
        for phi in phi_cap:
            x = R_shell * np.sin(theta) * np.cos(phi)
            y = R_shell * np.sin(theta) * np.sin(phi)
            z = R_shell * np.cos(theta)
            cap_points.append([x, y, z])
    cap_points = np.array(cap_points)

    combined = np.vstack([points, cap_points])
    log.info(f"  캡 포함 총 점: {len(combined):,}")

    return combined
