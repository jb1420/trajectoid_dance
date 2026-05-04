"""
Step 3: 구면 곡선 폐합 (TPT / MPT)
====================================

한 주기 구면 곡선을 닫힌 곡선으로 만든다.

TPT (Two-Period Trajectoid, n=2):
    대원호의 법선축으로 180° 회전 반사하여 두 번째 반쪽 생성.
    Gauss-Bonnet 조건에 의해 접선 연속성 자동 보장.

MPT (Multi-Period Trajectoid, n≥3):
    꼭짓점 Λ를 찾고 2πi/n 회전 복사.

v1 대체 대상: compute_trajectoid.py:double_the_path()
"""

import logging
import numpy as np
from dataclasses import dataclass

from .step1_find_radius import DarbouxSolution

log = logging.getLogger(__name__)


@dataclass
class ClosedCurve:
    """폐합된 구면 곡선 + Darboux frame."""
    s: np.ndarray          # (M_total,) 호장
    p: np.ndarray          # (M_total, 3) 위치
    t_hat: np.ndarray      # (M_total, 3) 접선
    N_hat: np.ndarray      # (M_total, 3) 측지 법선
    R: float               # 구 반지름
    n_periods: int         # 주기 수


# ================================================================
#  TPT: Period-2 폐합
# ================================================================
def close_tpt(sol: DarbouxSolution) -> ClosedCurve:
    """
    Two-Period Theorem에 의한 폐합.

    대칭축 c = (A × M) / |A × M| 기준 180° 회전:
        R_180 = 2·c·cᵀ − I

    두 번째 반쪽:
        p₂(s) = R_180 · p₁(L − s)
        t̂₂(s) = −R_180 · t̂₁(L − s)  (방향 반전)
        N̂₂(s) = R_180 · N̂₁(L − s)
    """
    R = sol.R
    A = sol.p[0]     # 시작점
    M = sol.p[-1]    # 끝점

    log.info("TPT (period-2) 폐합 시작")

    # 대칭축
    c = np.cross(A, M)
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-12 * R:
        log.warning("  A와 M이 대척점 또는 동일점 — 대원호 폐합 퇴화")
        # 퇴화 경우: 임의 수직 축 사용
        arbitrary = np.array([1, 0, 0]) if abs(A[0]) < 0.9 * R else np.array([0, 1, 0])
        c = np.cross(A, arbitrary)
        c_norm = np.linalg.norm(c)
    c = c / c_norm

    # 180° 회전 행렬: R_180 = 2·c·cᵀ − I
    R_180 = 2.0 * np.outer(c, c) - np.eye(3)

    # 첫 번째 반쪽 (원본)
    p1 = sol.p
    t1 = sol.t_hat
    N1 = sol.N_hat
    s1 = sol.s

    # 두 번째 반쪽 (반전 + 180° 회전)
    p1_rev = p1[::-1]     # 역순
    t1_rev = t1[::-1]
    N1_rev = N1[::-1]

    p2 = (R_180 @ p1_rev.T).T
    t2 = -(R_180 @ t1_rev.T).T   # 방향 반전
    N2 = (R_180 @ N1_rev.T).T

    # 호장 연결
    L = s1[-1]
    s2 = L + s1  # 두 번째 반쪽 호장은 L부터 2L까지

    # 합치기 (두 번째 반쪽의 첫 점 = 첫 번째 반쪽의 마지막 점과 동일하므로 제거)
    p_closed = np.vstack([p1, p2[1:]])
    t_closed = np.vstack([t1, t2[1:]])
    N_closed = np.vstack([N1, N2[1:]])
    s_closed = np.concatenate([s1, s2[1:]])

    # 접선 연속성 검증 (접합점 A, M에서)
    _check_junction(p1, t1, p2, t2, "M", R)
    _check_junction_end(p_closed, t_closed, "폐합점", R)

    log.info(f"  폐합 완료: {len(s_closed)} 점, 총 호장 = {s_closed[-1]:.6f}")

    return ClosedCurve(
        s=s_closed, p=p_closed,
        t_hat=t_closed, N_hat=N_closed,
        R=R, n_periods=2,
    )


# ================================================================
#  MPT: Period-n 폐합
# ================================================================
def close_mpt(sol: DarbouxSolution, n: int) -> ClosedCurve:
    """
    Multi-Period Theorem에 의한 폐합 (n ≥ 3).

    꼭짓점 Λ를 찾고 각 주기를 2π/n 만큼 회전 복사.

    Λ 조건:
        - A, M으로부터 등거리
        - ∠MΛA = 2π/n
        - 구면 거리 AM ≤ 2πR/n
    """
    R = sol.R
    A = sol.p[0]
    M = sol.p[-1]

    log.info(f"MPT (period-{n}) 폐합 시작")

    # A, M의 단위 구면 좌표
    a = A / R
    m = M / R

    # 대원호 AM의 각거리
    cos_am = np.clip(np.dot(a, m), -1, 1)
    theta_am = np.arccos(cos_am)

    if theta_am > 2 * np.pi / n:
        raise ValueError(
            f"구면 거리 AM = {theta_am:.4f} rad > 2π/{n} = {2*np.pi/n:.4f} rad. "
            f"Period-{n} trajectoid 불가."
        )

    # Λ 결정: AM 중점에서 수직 대원 위, ∠MΛA = 2π/n 만족하는 점
    # AM 중점
    mid = a + m
    mid_norm = np.linalg.norm(mid)
    if mid_norm < 1e-12:
        raise ValueError("A와 M이 대척점 — MPT 불가")
    mid = mid / mid_norm

    # AM에 수직이고 구면 위의 방향
    perp = np.cross(a, m)
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-12:
        raise ValueError("A = M — MPT 불가 (이미 닫힘)")
    perp = perp / perp_norm

    # Λ = mid·cos(φ) + perp×mid·sin(φ)의 형태로 탐색
    # ∠MΛA = 2π/n 을 만족하는 φ 결정
    half_am = theta_am / 2
    target_angle = 2 * np.pi / n

    # 이등변 구면삼각형에서: cos(AM/2) = cos(ΛM)·sin(∠MΛA/2) / sin(보조각)
    # 수치적으로 해결
    from scipy.optimize import brentq

    mid_perp = np.cross(perp, mid)  # mid에 수직인 접선 방향
    mid_perp = mid_perp / np.linalg.norm(mid_perp)

    def angle_at_lambda(phi):
        """φ에 대한 Λ 위치에서의 ∠MΛA 계산."""
        lam = mid * np.cos(phi) + mid_perp * np.sin(phi)
        lam = lam / np.linalg.norm(lam)
        # Λ에서 A, M까지의 구면 접선 벡터
        ta = a - np.dot(a, lam) * lam
        ta_n = np.linalg.norm(ta)
        tm = m - np.dot(m, lam) * lam
        tm_n = np.linalg.norm(tm)
        if ta_n < 1e-15 or tm_n < 1e-15:
            return 0.0
        ta = ta / ta_n
        tm = tm / tm_n
        return np.arccos(np.clip(np.dot(ta, tm), -1, 1))

    # φ ∈ (0, π) 에서 탐색
    try:
        phi_star = brentq(
            lambda phi: angle_at_lambda(phi) - target_angle,
            0.01, np.pi - 0.01,
            xtol=1e-12,
        )
    except ValueError:
        raise ValueError(
            f"Λ를 찾지 못함. 구면 거리 AM = {theta_am:.4f} 가 "
            f"2π/{n} = {2*np.pi/n:.4f} 보다 너무 클 수 있음."
        )

    Lambda = mid * np.cos(phi_star) + mid_perp * np.sin(phi_star)
    Lambda = Lambda / np.linalg.norm(Lambda) * R

    log.info(f"  Λ = {Lambda}")
    log.info(f"  ∠MΛA = {angle_at_lambda(phi_star):.6f} rad "
             f"(목표: {target_angle:.6f})")

    # 회전축: Λ 방향
    rot_axis = Lambda / R

    # 각 주기를 2πi/n 만큼 회전하여 합치기
    angle_step = 2 * np.pi / n
    all_p = []
    all_t = []
    all_N = []
    all_s = []

    L_period = sol.s[-1]

    for i in range(n):
        angle = i * angle_step
        # Rodrigues 회전 행렬
        K = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0],
        ])
        R_rot = (np.eye(3)
                 + np.sin(angle) * K
                 + (1 - np.cos(angle)) * (K @ K))

        p_i = (R_rot @ sol.p.T).T
        t_i = (R_rot @ sol.t_hat.T).T
        N_i = (R_rot @ sol.N_hat.T).T
        s_i = sol.s + i * L_period

        if i > 0:
            # 첫 점 제거 (이전 주기 마지막 점과 동일)
            p_i = p_i[1:]
            t_i = t_i[1:]
            N_i = N_i[1:]
            s_i = s_i[1:]

        all_p.append(p_i)
        all_t.append(t_i)
        all_N.append(N_i)
        all_s.append(s_i)

    p_closed = np.vstack(all_p)
    t_closed = np.vstack(all_t)
    N_closed = np.vstack(all_N)
    s_closed = np.concatenate(all_s)

    log.info(f"  폐합 완료: {len(s_closed)} 점, 총 호장 = {s_closed[-1]:.6f}")

    return ClosedCurve(
        s=s_closed, p=p_closed,
        t_hat=t_closed, N_hat=N_closed,
        R=R, n_periods=n,
    )


# ================================================================
#  폐합 편의 함수
# ================================================================
def close_curve(sol: DarbouxSolution, n: int = 2) -> ClosedCurve:
    """n에 따라 TPT 또는 MPT 선택."""
    if n == 2:
        return close_tpt(sol)
    elif n >= 3:
        return close_mpt(sol, n)
    else:
        raise ValueError(f"n must be ≥ 2, got {n}")


# ================================================================
#  검증 헬퍼
# ================================================================
def _check_junction(p1, t1, p2, t2, name, R):
    """접합점에서의 위치·접선 연속성 검사."""
    pos_gap = np.linalg.norm(p1[-1] - p2[0])
    if pos_gap > 1e-6 * R:
        log.warning(f"  {name} 접합점 위치 불연속: {pos_gap:.2e}")

    # 접선 방향 차이 (방향이 같아야 함)
    t_end = t1[-1]
    t_start = t2[0]
    cos_angle = np.clip(np.dot(t_end, t_start), -1, 1)
    angle_deg = np.degrees(np.arccos(abs(cos_angle)))
    if angle_deg > 1.0:
        log.warning(f"  {name} 접합점 접선 불연속: {angle_deg:.2f}°")


def _check_junction_end(p_closed, t_closed, name, R):
    """폐합 시작/끝점 검사."""
    pos_gap = np.linalg.norm(p_closed[0] - p_closed[-1])
    if pos_gap > 1e-4 * R:
        log.warning(f"  {name} 위치 갭: {pos_gap:.2e}")
