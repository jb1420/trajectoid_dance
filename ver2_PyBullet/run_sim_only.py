"""
시뮬레이션만 실행 스크립트
===========================

이미 생성된 trajectoid.obj 메쉬를 사용하여 PyBullet 시뮬레이션만 수행.

사용법:
    python -m ver2_PyBullet.run_sim_only
    python -m ver2_PyBullet.run_sim_only --mesh output_v2/trajectoid.obj --mode inclined --duration 60
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ver2_PyBullet.config import BALL_R, SIM_DURATION
from ver2_PyBullet.step6_pybullet_sim import (
    TrajectoidSimulation, PathFollowController,
)
from ver2_PyBullet.paths import PathSpec
from ver2_PyBullet.verification import check_no_slip


def run_sim_only(
    mesh_path: str,
    mode: str = 'inclined',
    slope_deg: float = 5.0,
    duration: float = SIM_DURATION,
    gui: bool = True,
    path_name: str = 'sinusoid',
    path_scale: float = 1.0,
    output_dir: str = 'output_v2',
):
    """
    시뮬레이션 단독 실행.

    Parameters
    ----------
    mesh_path : trajectoid 메쉬 파일 경로
    mode : 'inclined' | 'tilting'
    slope_deg : 경사각 (도)
    duration : 시뮬 시간 (초)
    gui : GUI 표시
    path_name : 경로 (tilting 모드일 때만 사용)
    path_scale : 경로 크기
    output_dir : 결과 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"메쉬 파일 없음: {mesh_path}")

    print("=" * 60)
    print("  Trajectoid v2 — Simulation Only")
    print("=" * 60)
    print(f"  메쉬: {mesh_path}")
    print(f"  모드: {mode}")
    if mode == 'inclined':
        print(f"  경사각: {slope_deg}°")
    print(f"  시뮬 시간: {duration}s")
    print("=" * 60)

    # 경로 로드 (tilting 모드)
    path = None
    if mode == 'tilting':
        print("\n경로 로드 …")
        path = PathSpec.from_preset(path_name, scale=path_scale)
        print(f"  경로: {path_name} (L={path.L:.4f})")

    # 시뮬레이션 환경 생성
    print("\nPyBullet 환경 초기화 …")
    sim = TrajectoidSimulation(
        mesh_path=mesh_path,
        mode=mode,
        slope_deg=slope_deg,
        gui=gui,
    )

    # 제어기 생성 (tilting 모드)
    controller = None
    if mode == 'tilting' and path is not None:
        controller = PathFollowController(path.xy)
        print(f"경로 추종 제어기 활성화")

    # 시뮬레이션 실행
    print(f"\n시뮬레이션 실행 ({duration}s) …")
    try:
        history = sim.run(
            duration=duration,
            tilt_controller=controller,
            realtime=gui,
        )
        print("시뮬레이션 완료!")

        # 미끄러짐 검증
        print("\n미끄러짐 분석 …")
        slip_result = check_no_slip(history, BALL_R)
        print(f"  mean_slip: {slip_result['mean_slip']:.6f}")
        print(f"  max_slip: {slip_result['max_slip']:.6f}")
        print(f"  p95_slip: {slip_result['p95_slip']:.6f}")

        # 결과 시각화
        print(f"\n결과 시각화 …")
        _plot_results(history, output_dir)

        # 접촉 궤적 저장
        if mode == 'tilting' and path is not None:
            contact_trace = sim.get_contact_trace()
            if len(contact_trace) > 0:
                _plot_contact_trace(contact_trace, path.xy,
                                    os.path.join(output_dir, 'contact_trace.png'))

    finally:
        sim.close()

    print("\n" + "=" * 60)
    print(f"결과 저장: {os.path.abspath(output_dir)}/")
    print("=" * 60)


def _plot_results(history: dict, output_dir: str):
    """시뮬레이션 결과 시각화."""
    time_arr = np.array(history['time'])
    pos_arr = np.array(history['ball_pos'])
    vel_arr = np.array(history['ball_lin_vel'])
    ang_vel_arr = np.array(history['ball_ang_vel'])

    speeds = np.linalg.norm(vel_arr, axis=1)

    # 1) 위치
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(time_arr, pos_arr[:, 0], 'b-')
    axes[0, 0].set_ylabel('x [m]')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Position')

    axes[0, 1].plot(time_arr, pos_arr[:, 1], 'g-')
    axes[0, 1].set_ylabel('y [m]')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time_arr, pos_arr[:, 2], 'r-')
    axes[1, 0].set_ylabel('z [m]')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].grid(True, alpha=0.3)

    # 2) 속도
    axes[1, 1].plot(time_arr, speeds, 'k-', linewidth=1.5)
    axes[1, 1].set_ylabel('Speed [m/s]')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Linear Velocity Magnitude')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory.png'), dpi=150)
    plt.close()

    # 3) 각속도
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_arr, ang_vel_arr[:, 0], label='ωx', alpha=0.7)
    ax.plot(time_arr, ang_vel_arr[:, 1], label='ωy', alpha=0.7)
    ax.plot(time_arr, ang_vel_arr[:, 2], label='ωz', alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angular_velocity.png'), dpi=150)
    plt.close()

    log.info(f"  궤적 그래프 저장: trajectory.png")
    log.info(f"  각속도 그래프 저장: angular_velocity.png")


def _plot_contact_trace(trace: np.ndarray, target_path: np.ndarray,
                        save_path: str):
    """접촉 궤적 vs 목표 경로."""
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
    log.info(f"  접촉 궤적 그래프 저장: contact_trace.png")


def main():
    parser = argparse.ArgumentParser(
        description='Trajectoid v2 — Simulation Only',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mesh', default='output_v2/trajectoid.obj',
                        help='trajectoid 메쉬 파일 경로')
    parser.add_argument('--mode', default='inclined',
                        choices=['inclined', 'tilting'],
                        help='시뮬레이션 모드')
    parser.add_argument('--slope', type=float, default=5.0,
                        help='경사각 (도, inclined 모드)')
    parser.add_argument('--duration', type=float, default=SIM_DURATION,
                        help='시뮬레이션 시간 (초)')
    parser.add_argument('--no-gui', action='store_true',
                        help='GUI 비활성화')
    parser.add_argument('--path', default='sinusoid',
                        choices=['sinusoid', 'figure8', 'lemniscate',
                                 'circle', 'zigzag'],
                        help='경로 (tilting 모드)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='경로 스케일')
    parser.add_argument('--output', default='output_v2',
                        help='출력 디렉토리')

    args = parser.parse_args()

    run_sim_only(
        mesh_path=args.mesh,
        mode=args.mode,
        slope_deg=args.slope,
        duration=args.duration,
        gui=not args.no_gui,
        path_name=args.path,
        path_scale=args.scale,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
