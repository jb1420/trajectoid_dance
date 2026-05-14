"""
Trajectoid v2 — 전체 파이프라인 실행기
========================================

경로 → R* 탐색 → 구면 곡선 → 폐합 → Ruled Surface → Convex Hull → 시뮬레이션

사용법:
    python -m ver2_PyBullet.run_pipeline
    python -m ver2_PyBullet.run_pipeline --path sinusoid --n 2 --sim inclined
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectoid.ver2_PyBullet.config import OUTPUT_DIR, R_SHELL_RATIO, N_S_SAMPLES, N_LAMBDA_SAMPLES
from trajectoid.ver2_PyBullet.paths import PathSpec
from trajectoid.ver2_PyBullet.step1_find_radius import find_optimal_radius, scan_area_function
from trajectoid.ver2_PyBullet.step2_sphere_curve import generate_sphere_curve
from trajectoid.ver2_PyBullet.step3_close_curve import close_curve
from trajectoid.ver2_PyBullet.step4_ruled_surface import generate_ruled_surface_with_caps
from trajectoid.ver2_PyBullet.step5_convex_hull_mesh import generate_mesh
from trajectoid.ver2_PyBullet.verification import run_all_checks


def run_pipeline(
    path_name: str = 'sinusoid',
    path_scale: float = 1.0,
    path_npoints: int = 2000,
    n_periods: int = 2,
    output_dir: str = OUTPUT_DIR,
    run_sim: bool = True,
    sim_mode: str = 'inclined',
    slope_deg: float = 5.0,
    gui: bool = True,
    sim_duration: float = 30.0,
    scan_only: bool = False,
):
    """
    전체 파이프라인 실행.

    Parameters
    ----------
    path_name : 경로 이름 ('sinusoid', 'figure8', 'lemniscate', 'circle', 'zigzag')
    path_scale : 경로 크기
    path_npoints : 경로 점 수
    n_periods : trajectoid 주기 수 (2=TPT, ≥3=MPT)
    output_dir : 출력 디렉토리
    run_sim : 시뮬레이션 실행 여부
    sim_mode : 'inclined' | 'tilting'
    slope_deg : 경사각 (inclined 모드)
    gui : PyBullet GUI 표시
    sim_duration : 시뮬레이션 시간 (s)
    scan_only : R 스캔만 수행 (디버깅용)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Trajectoid v2 Pipeline")
    print("=" * 60)
    print(f"  경로: {path_name} (scale={path_scale}, {path_npoints} pts)")
    print(f"  주기: n={n_periods}")
    print(f"  출력: {output_dir}/")
    print("=" * 60)

    # ── Step 0: 경로 생성 ────────────────────────────────────
    print("\n[0/6] 경로 생성 …")
    path = PathSpec.from_preset(path_name, scale=path_scale, n_points=path_npoints)
    print(f"  호장 L = {path.L:.6f}")
    print(f"  회전수 I_T = {path.turning_number():.4f}")
    print(f"  곡률 범위: [{path.kappa.min():.4f}, {path.kappa.max():.4f}]")

    # 경로 시각화 저장
    _plot_path(path, os.path.join(output_dir, 'path.png'))

    # ── Step 1: R* 탐색 ──────────────────────────────────────
    print("\n[1/6] 최적 반지름 R* 탐색 …")

    if scan_only:
        R_vals, f_plus, f_minus = scan_area_function(path, n=n_periods)
        _plot_area_scan(R_vals, f_plus, f_minus, n_periods,
                        os.path.join(output_dir, 'area_scan.png'))
        print("  스캔 완료 (scan_only=True). 파이프라인 종료.")
        return

    R_star = find_optimal_radius(path, n=n_periods)
    print(f"  ★ R* = {R_star:.10f}")

    # ── Step 2: 구면 곡선 생성 ────────────────────────────────
    print("\n[2/6] 구면 곡선 생성 (R = R*) …")
    sol = generate_sphere_curve(path, R_star, n_points=N_S_SAMPLES)
    print(f"  곡선 점 수: {len(sol.s)}")

    # 구면 곡선 시각화 저장
    _plot_sphere_curve(sol, os.path.join(output_dir, 'sphere_curve.png'))

    # ── Step 3: 곡선 폐합 ────────────────────────────────────
    print(f"\n[3/6] 곡선 폐합 (period-{n_periods}) …")
    closed = close_curve(sol, n=n_periods)
    print(f"  폐합 곡선 점 수: {len(closed.s)}")

    # ── Step 4: Ruled surface 생성 ────────────────────────────
    print("\n[4/6] Ruled surface 점 구름 생성 …")
    point_cloud = generate_ruled_surface_with_caps(
        closed,
        R_shell_ratio=R_SHELL_RATIO,
        n_s=N_S_SAMPLES,
        n_lambda=N_LAMBDA_SAMPLES,
    )
    print(f"  점 구름: {len(point_cloud):,} 점")

    # 점 구름 저장
    np.save(os.path.join(output_dir, 'point_cloud.npy'), point_cloud)

    # ── Step 5: Convex hull 메쉬 ──────────────────────────────
    print("\n[5/6] Convex hull 메쉬 생성 …")
    mesh = generate_mesh(point_cloud, output_dir=output_dir)
    print(f"  메쉬: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # ── Step 6: 검증 ──────────────────────────────────────────
    print("\n[6/6] 검증 …")
    mesh_path = os.path.join(output_dir, 'trajectoid.obj')
    verification = run_all_checks(mesh, R_star)

    # ── 시뮬레이션 (옵션) ─────────────────────────────────────
    if run_sim:
        print("\n[SIM] PyBullet 시뮬레이션 시작 …")
        from trajectoid.ver2_PyBullet.step6_pybullet_sim import (
            TrajectoidSimulation, PathFollowController,
        )

        sim = TrajectoidSimulation(
            mesh_path=mesh_path,
            mode=sim_mode,
            slope_deg=slope_deg,
            gui=gui,
        )

        controller = None
        if sim_mode == 'tilting':
            controller = PathFollowController(path.xy)

        try:
            history = sim.run(
                duration=sim_duration,
                tilt_controller=controller,
                realtime=gui,
            )

            # 접촉 궤적 시각화
            contact_trace = sim.get_contact_trace()
            if len(contact_trace) > 0:
                _plot_contact_trace(contact_trace, path.xy,
                                    os.path.join(output_dir, 'contact_trace.png'))

            # 시뮬레이션 포함 재검증
            verification_with_sim = run_all_checks(
                mesh, R_star, history=history, target_path=path.xy
            )

        finally:
            sim.close()

    print("\n" + "=" * 60)
    print("  파이프라인 완료!")
    print(f"  출력 디렉토리: {os.path.abspath(output_dir)}")
    print("=" * 60)


# ================================================================
#  시각화 헬퍼
# ================================================================

def _plot_path(path: PathSpec, save_path: str):
    """2D 경로 + 곡률 시각화."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(path.xy[:, 0], path.xy[:, 1], 'b-', linewidth=1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Input Path')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)

    ax2.plot(path.s, path.kappa, 'r-', linewidth=1)
    ax2.set_title('Curvature κ(s)')
    ax2.set_xlabel('Arc length s')
    ax2.set_ylabel('κ')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  경로 시각화 저장: {save_path}")


def _plot_sphere_curve(sol, save_path: str):
    """3D 구면 곡선 시각화."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 구면 와이어프레임
    R = sol.R
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1, linewidth=0.3)

    # 구면 곡선
    ax.plot(sol.p[:, 0], sol.p[:, 1], sol.p[:, 2],
            'b-', linewidth=2, label='Sphere curve')
    ax.scatter(*sol.p[0], color='green', s=50, label='A (start)')
    ax.scatter(*sol.p[-1], color='red', s=50, label='M (end)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'Sphere Curve (R = {R:.4f})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  구면 곡선 시각화 저장: {save_path}")


def _plot_area_scan(R_vals, f_plus, f_minus, n, save_path: str):
    """면적 잔차 함수 f(R) 스캔 시각화."""
    fig, ax = plt.subplots(figsize=(12, 5))
    valid_p = ~np.isnan(f_plus)
    valid_m = ~np.isnan(f_minus)
    ax.plot(R_vals[valid_p], f_plus[valid_p], 'b-', label='f(R) [+2π/n]')
    ax.plot(R_vals[valid_m], f_minus[valid_m], 'r-', label='f(R) [-2π/n]')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('R')
    ax.set_ylabel('f(R) = S(R)/R² ∓ 2π/n')
    ax.set_title(f'Area Residual Scan (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  면적 스캔 시각화 저장: {save_path}")


def _plot_contact_trace(trace: np.ndarray, target_path: np.ndarray,
                        save_path: str):
    """접촉 궤적 vs 목표 경로 비교."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(target_path[:, 0], target_path[:, 1], 'b-',
            linewidth=2, alpha=0.5, label='Target path')
    ax.plot(trace[:, 0], trace[:, 1], 'r.', markersize=1,
            alpha=0.3, label='Contact trace')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Contact Trace vs Target Path')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  접촉 궤적 시각화 저장: {save_path}")


# ================================================================
#  CLI 진입점
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Trajectoid v2 Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--path', default='sinusoid',
                        choices=['sinusoid', 'figure8', 'lemniscate',
                                 'circle', 'zigzag'],
                        help='경로 프리셋')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='경로 크기')
    parser.add_argument('--npoints', type=int, default=2000,
                        help='경로 점 수')
    parser.add_argument('--n', type=int, default=2,
                        help='Trajectoid 주기 수 (2=TPT, ≥3=MPT)')
    parser.add_argument('--output', default=OUTPUT_DIR,
                        help='출력 디렉토리')
    parser.add_argument('--no-sim', action='store_true',
                        help='시뮬레이션 건너뛰기')
    parser.add_argument('--sim-mode', default='inclined',
                        choices=['inclined', 'tilting'],
                        help='시뮬레이션 모드')
    parser.add_argument('--slope', type=float, default=5.0,
                        help='경사각 (도, inclined 모드)')
    parser.add_argument('--no-gui', action='store_true',
                        help='GUI 비활성화')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='시뮬레이션 시간 (초)')
    parser.add_argument('--scan-only', action='store_true',
                        help='R 스캔만 수행 (디버깅용)')

    args = parser.parse_args()

    run_pipeline(
        path_name=args.path,
        path_scale=args.scale,
        path_npoints=args.npoints,
        n_periods=args.n,
        output_dir=args.output,
        run_sim=not args.no_sim,
        sim_mode=args.sim_mode,
        slope_deg=args.slope,
        gui=not args.no_gui,
        sim_duration=args.duration,
        scan_only=args.scan_only,
    )


if __name__ == '__main__':
    main()
