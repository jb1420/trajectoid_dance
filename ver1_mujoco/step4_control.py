"""
Step 4: Control Logic & Main Simulation Loop
============================================

두 가지 제어 모드:

  [Sphere]
    - 현재 공 위치와 경로 상의 Look-ahead 점 사이 오차를 계산
    - simple-pid 2개(X/Y)로 평판 Pitch/Roll 각도를 제어
    - 순수 추종(Pure Pursuit) 방식으로 경로 따라가기

  [Trajectoid]
    - 기울기 방향 = 경로 접선 φ
    - 기울기 크기 = PID(target_speed - current_speed)  ← Y축만 PID 제어
    - roll_y  =  tilt_mag · cos(φ)
    - pitch_x = -tilt_mag · sin(φ)

실행:
    python step4_control.py

필수:
    pip install mujoco simple-pid
    (Trajectoid 모드는 step1 + step2/3 먼저 실행)
"""

import os
import sys
import time
import numpy as np

try:
    from simple_pid import PID
except ImportError:
    sys.exit("[오류] simple-pid 가 없습니다.\n  pip install simple-pid")

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("[오류] mujoco 가 없습니다.\n  pip install mujoco")

# step2_3에서 환경 클래스와 상수 임포트
sys.path.insert(0, os.path.dirname(__file__))
from ver1_mujoco.step2_3_mujoco_env import (
    BallPlateEnv,
    PLATE_HALF, PLATE_THICK, PLATE_H,
    BALL_R, CORE_R, PLATE_MAX_TILT, TIMESTEP,
)


# ================================================================
#  SETTINGS
# ================================================================
OBJECT_TYPE    = 'trajectoid'  # 'sphere' | 'trajectoid'

# ── 경로 ──────────────────────────────────────────────────────
PATH_TYPE      = 'sinusoid'      # 'sinusoid' | 'figure8' | 'lemniscate' | 'circle'
PATH_SCALE     = 0.15          # 경로 최대 진폭 (m) — PLATE_HALF(0.25)보다 작게

# ── Sphere PID ────────────────────────────────────────────────
PID_KP         = 1.8           # 비례 게인
PID_KI         = 0.04          # 적분 게인 (작을수록 안정)
PID_KD         = 0.60          # 미분 게인 (진동 억제)
LOOKAHEAD_M    = 0.035         # Look-ahead 거리 (m) — 너무 짧으면 진동
CTRL_LIMIT_RAD = PLATE_MAX_TILT * 0.8   # PID 출력 상한 (± rad)

# ── Trajectoid 속도 PID ────────────────────────────────────────
TARGET_SPEED   = 0.25         # m/s  목표 구름 속도
SPEED_KP       = 2.0           # 속도 PID 비례 게인
SPEED_KI       = 0.05          # 속도 PID 적분 게인
SPEED_KD       = 0.30          # 속도 PID 미분 게인
TILT_MIN       = 0.005         # rad  최소 기울기 (정지 방지)
TILT_MAX       = PLATE_MAX_TILT * 0.8   # rad  최대 기울기

# ── 경로 저장 (Step 1과 동기화) ───────────────────────────────
SAVE_PATH_DATA = True          # True 이면 경로를 path_data.npy 로 저장
PATH_DATA_FILE = 'output/path_data.npy'  # 저장 경로 (step1 과 공유)
PATH_N         = 800           # 경로 점 수 (step1 정밀도에 영향)

# ── 시뮬레이션 ────────────────────────────────────────────────
MAX_SIM_TIME   = 500.0         # 최대 시뮬레이션 시간 (s)
SUBSTEPS       = 4             # mj_step 반복 횟수 / 제어 주기
LOG_INTERVAL   = 1000           # 터미널 출력 간격 (step 수)


# ================================================================
#  경로 생성
# ================================================================
def make_path(path_type: str, scale: float, n: int = 800) -> np.ndarray:
    """
    평판 로컬 좌표 (x, y) [m] 로 경로를 생성합니다.

    Returns
    -------
    path : (n, 2) ndarray — 각 행이 (x, y) [m]
    """
    t = np.linspace(0, 2 * np.pi, n)

    if path_type == 'sinusoid':
        x = np.linspace(-scale, scale, n)
        y = scale * 0.6 * np.sin(3 * np.pi * x / scale)

    elif path_type == 'figure8':
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)

    elif path_type == 'lemniscate':
        a    = scale
        den  = 1 + np.sin(t) ** 2
        x    = a * np.cos(t) / den
        y    = a * np.sin(t) * np.cos(t) / den

    elif path_type == 'circle':
        r = scale * 0.85
        x = r * np.cos(t)
        y = r * np.sin(t)

    else:
        raise ValueError(f"알 수 없는 PATH_TYPE: '{path_type}'")

    return np.stack([x, y], axis=1)


def path_tangent_angles(path: np.ndarray) -> np.ndarray:
    """각 경로 점에서의 접선 방향 각도 (rad) 계산."""
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    return np.arctan2(dy, dx)


# ================================================================
#  경로 추종 제어기
# ================================================================
class PathFollowController:
    """
    경로 추종 제어기.

    Sphere  모드: Look-ahead + 2축 독립 PID
    Trajectoid 모드: 경로 접선 방향 기울기 유지

    Parameters
    ----------
    object_type : 'sphere' | 'trajectoid'
    path        : (N, 2) ndarray — 평판 로컬 좌표 경로 [m]
    dt          : 제어 주기 (s)
    """

    def __init__(
        self,
        object_type: str,
        path: np.ndarray,
        dt: float = TIMESTEP * SUBSTEPS,
    ):
        self.object_type = object_type
        self.path        = path
        self.tangents    = path_tangent_angles(path)
        self._prev_idx   = 0   # 경로 진행 인덱스 (뒤로 돌아가지 않음)

        # ── 호 길이 누적 (look-ahead 계산용) ──────────────────
        diffs            = np.diff(path, axis=0)
        seg_lens         = np.linalg.norm(diffs, axis=1)
        self._arclens    = np.concatenate([[0.0], np.cumsum(seg_lens)])

        # ── PID (sphere only) ──────────────────────────────────
        if object_type == 'sphere':
            self._pid_x = PID(
                Kp=PID_KP, Ki=PID_KI, Kd=PID_KD,
                setpoint=0.0,
                sample_time=dt,
                output_limits=(-CTRL_LIMIT_RAD, CTRL_LIMIT_RAD),
            )
            self._pid_y = PID(
                Kp=PID_KP, Ki=PID_KI, Kd=PID_KD,
                setpoint=0.0,
                sample_time=dt,
                output_limits=(-CTRL_LIMIT_RAD, CTRL_LIMIT_RAD),
            )

        # ── 로그용 ────────────────────────────────────────────
        self.nearest_pt  = path[0].copy()
        self.lookahead_pt = path[0].copy()
        self.cross_err   = 0.0

    # ── 내부: 현재 인덱스 + Look-ahead 인덱스 탐색 ────────────
    def _advance(self, ball_xy: np.ndarray) -> tuple[int, int]:
        """
        현재 공 위치에서 가장 가까운 경로 인덱스를 찾고,
        그보다 LOOKAHEAD_M 앞에 있는 인덱스를 반환합니다.

        단방향 진행 (뒤로 되돌아가지 않음) — 반복 경로에 강인.
        """
        n     = len(self.path)
        # 이전 인덱스 근방에서 검색 (윈도우 크기 60)
        lo    = max(0, self._prev_idx - 5)
        hi    = min(n, lo + 80)
        dists = np.linalg.norm(self.path[lo:hi] - ball_xy, axis=1)
        nearest_idx = int(np.argmin(dists)) + lo
        self._prev_idx = nearest_idx

        # Look-ahead: 호 길이 기준으로 전진
        arc_target = self._arclens[nearest_idx] + LOOKAHEAD_M
        # 호 길이가 arc_target 을 넘는 첫 인덱스
        lookahead_idx = int(
            np.searchsorted(self._arclens, arc_target, side='left')
        )
        lookahead_idx = min(lookahead_idx, n - 1)

        return nearest_idx, lookahead_idx

    # ── 횡방향 오차 (Cross-Track Error) ──────────────────────
    def _cross_track_error(self, ball_xy: np.ndarray, nearest_idx: int) -> float:
        """경로 접선에 수직 방향(법선) 오차 [m]"""
        tang   = self.tangents[nearest_idx]
        normal = np.array([-np.sin(tang), np.cos(tang)])   # 법선 벡터
        diff   = ball_xy - self.path[nearest_idx]
        return float(np.dot(diff, normal))

    # ── 메인 제어 계산 ────────────────────────────────────────
    def compute(self, obs: dict) -> tuple[float, float]:
        """
        관측값을 받아 (roll_y_target, pitch_x_target) 반환.

        Parameters
        ----------
        obs : BallPlateEnv.get_obs() 딕셔너리

        Returns
        -------
        roll_y  : float — Y축 목표 회전각 (rad)
        pitch_x : float — X축 목표 회전각 (rad)
        """
        ball_xy = obs['ball_pos_plate']   # 평판 로컬 (x, y)

        if self.object_type == 'sphere':
            return self._sphere_control(ball_xy)
        else:
            return self._trajectoid_control(ball_xy)

    def _sphere_control(self, ball_xy: np.ndarray) -> tuple[float, float]:
        """
        Sphere PID 제어.

        Look-ahead 점을 목표로 두고 X/Y 독립 PID로 평판 각도 계산.

          roll_y  = PID_x(error_x)      [+X 방향 경사]
          pitch_x = -PID_y(error_y)     [부호: +Y 경사는 pitch 음수]
        """
        nearest_idx, lookahead_idx = self._advance(ball_xy)
        lookahead_pt = self.path[lookahead_idx]

        self.nearest_pt   = self.path[nearest_idx].copy()
        self.lookahead_pt = lookahead_pt.copy()
        self.cross_err    = self._cross_track_error(ball_xy, nearest_idx)

        # PID: setpoint = lookahead 좌표, 입력 = 현재 공 좌표
        self._pid_x.setpoint = float(lookahead_pt[0])
        self._pid_y.setpoint = float(lookahead_pt[1])

        roll_y  = float(self._pid_x(float(ball_xy[0])))
        pitch_x = -float(self._pid_y(float(ball_xy[1])))

        return roll_y, pitch_x

    def _trajectoid_control(self, ball_xy: np.ndarray) -> tuple[float, float]:
        """
        Trajectoid 접선 기울기 제어.

        접선 방향 φ 로 일정 기울기를 유지:
          roll_y  =  TRAJ_TILT_MAG · cos(φ)
          pitch_x = -TRAJ_TILT_MAG · sin(φ)

        이 제어로 평판이 경로 방향 아래를 향하므로,
        트레젝토이드 형태가 자연스럽게 경로를 따라가도록 함.
        """
        nearest_idx, _ = self._advance(ball_xy)
        phi = float(self.tangents[nearest_idx])

        self.nearest_pt  = self.path[nearest_idx].copy()
        self.cross_err   = self._cross_track_error(ball_xy, nearest_idx)

        roll_y  = TRAJ_TILT_MAG * np.cos(phi)
        pitch_x = -TRAJ_TILT_MAG * np.sin(phi)

        return roll_y, pitch_x

    def reset(self) -> None:
        """PID 적분 초기화, 경로 인덱스 리셋."""
        self._prev_idx = 0
        if self.object_type == 'sphere':
            self._pid_x.reset()
            self._pid_y.reset()


# ================================================================
#  Trajectoid 전용 속도 제어기
# ================================================================
class TrajectoidSpeedController:
    """
    Trajectoid 속도 제어기.

    기울기 방향  = 경로 접선 φ
    기울기 크기  = PID(target_speed - current_speed)

    roll_y  =  tilt_mag · cos(φ)
    pitch_x = -tilt_mag · sin(φ)

    Parameters
    ----------
    path         : (N, 2) ndarray — 평판 로컬 좌표 경로 [m]
    target_speed : 목표 구름 속도 [m/s]
    dt           : 제어 주기 [s]
    """

    def __init__(self, path: np.ndarray, target_speed: float, dt: float):
        self.path      = path
        self.tangents  = path_tangent_angles(path)
        self._prev_idx = 0

        diffs           = np.diff(path, axis=0)
        self._arclens   = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])

        self.pid_speed  = PID(
            Kp=SPEED_KP, Ki=SPEED_KI, Kd=SPEED_KD,
            setpoint=target_speed,
            sample_time=dt,
            output_limits=(TILT_MIN, TILT_MAX),
        )

        # 로그·마커용
        self.nearest_pt   = path[0].copy()
        self.lookahead_pt = path[0].copy()
        self.tilt_mag     = TILT_MIN
        self.cross_err    = 0.0

    def _advance(self, ball_xy: np.ndarray) -> tuple[int, int]:
        """nearest idx 탐색 (단방향, 윈도우 80) — PathFollowController 와 동일."""
        n     = len(self.path)
        lo    = max(0, self._prev_idx - 5)
        hi    = min(n, lo + 80)
        dists = np.linalg.norm(self.path[lo:hi] - ball_xy, axis=1)
        nearest_idx = int(np.argmin(dists)) + lo
        self._prev_idx = nearest_idx

        arc_target    = self._arclens[nearest_idx] + LOOKAHEAD_M
        lookahead_idx = int(np.searchsorted(self._arclens, arc_target, side='left'))
        lookahead_idx = min(lookahead_idx, n - 1)
        return nearest_idx, lookahead_idx

    def _cross_track_error(self, ball_xy: np.ndarray, nearest_idx: int) -> float:
        tang   = self.tangents[nearest_idx]
        normal = np.array([-np.sin(tang), np.cos(tang)])
        diff   = ball_xy - self.path[nearest_idx]
        return float(np.dot(diff, normal))

    def compute(self, obs: dict) -> tuple[float, float]:
        """
        관측값을 받아 (roll_y_target, pitch_x_target) 반환.

        obs 에 'ball_vel_plate' (평판 로컬 2D 속도) 가 있어야 합니다.
        """
        ball_xy = obs['ball_pos_plate']
        speed   = float(np.linalg.norm(obs['ball_vel_plate']))

        nearest_idx, lookahead_idx = self._advance(ball_xy)
        phi = float(self.tangents[nearest_idx])

        self.tilt_mag     = float(self.pid_speed(speed))
        self.nearest_pt   = self.path[nearest_idx].copy()
        self.lookahead_pt = self.path[lookahead_idx].copy()
        self.cross_err    = self._cross_track_error(ball_xy, nearest_idx)

        roll_y  = self.tilt_mag * np.cos(phi)
        pitch_x = -self.tilt_mag * np.sin(phi)
        return roll_y, pitch_x

    def reset(self) -> None:
        """PID 적분 초기화, 경로 인덱스 리셋."""
        self._prev_idx = 0
        self.pid_speed.reset()


# ================================================================
#  뷰어 커스텀 마커 (Look-ahead 점 시각화)
# ================================================================
def _draw_markers(
    viewer,
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    lookahead_world: np.ndarray | None = None,
    nearest_world:   np.ndarray | None = None,
) -> None:
    """
    MuJoCo 뷰어 user_scn 에 마커 geom 을 추가합니다.
    - 노란 구  : Look-ahead 목표점
    - 흰색 구  : 현재 가장 가까운 경로점
    """
    try:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            idx = 0

            def _add_sphere(pos, rgba, size=0.007):
                nonlocal idx
                if idx >= viewer.user_scn.maxgeom:
                    return
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[idx],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([size, 0.0, 0.0]),
                    pos.astype(np.float64),
                    np.eye(3).flatten().astype(np.float64),
                    np.array(rgba, dtype=np.float32),
                )
                idx += 1
                viewer.user_scn.ngeom = idx

            if lookahead_world is not None:
                _add_sphere(lookahead_world, [1.0, 0.95, 0.0, 0.95])   # 노란색
            if nearest_world is not None:
                _add_sphere(nearest_world,   [1.0, 1.0,  1.0, 0.80])   # 흰색
    except Exception:
        pass   # 마커 실패해도 시뮬레이션 계속


def _plate_local_to_world(
    local_xy: np.ndarray,
    model: mujoco.MjModel,
    data:  mujoco.MjData,
) -> np.ndarray:
    """평판 로컬 (x, y) 좌표를 World 3D 좌표로 변환."""
    body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'plate_inner')
    origin   = data.xpos[body_id]
    rot_mat  = data.xmat[body_id].reshape(3, 3)
    z_offset = PLATE_THICK / 2 + 0.003
    local_3d = np.array([local_xy[0], local_xy[1], z_offset])
    return origin + rot_mat @ local_3d


# ================================================================
#  메인 시뮬레이션 루프
# ================================================================
def run_simulation() -> None:
    print("=" * 58)
    print("  Ball-and-Plate Control Simulation  (Step 4)")
    print("=" * 58)
    print(f"  OBJECT_TYPE  : {OBJECT_TYPE}")
    print(f"  PATH_TYPE    : {PATH_TYPE}")
    if OBJECT_TYPE == 'sphere':
        print(f"  PID gains    : Kp={PID_KP}  Ki={PID_KI}  Kd={PID_KD}")
        print(f"  Look-ahead   : {LOOKAHEAD_M*1000:.0f} mm")
    else:
        print(f"  Target speed : {TARGET_SPEED*1000:.1f} mm/s")
        print(f"  Speed PID    : Kp={SPEED_KP}  Ki={SPEED_KI}  Kd={SPEED_KD}")
        print(f"  Tilt range   : [{np.degrees(TILT_MIN):.2f}°, {np.degrees(TILT_MAX):.1f}°]")
    print("=" * 58)

    # ── 경로 생성 ────────────────────────────────────────────
    path = make_path(PATH_TYPE, PATH_SCALE, n=PATH_N)
    print(f"\n  경로: {len(path)} pts  ({PATH_TYPE})")

    # ── Step 1 용 path_data.npy 저장 (선택) ─────────────────
    if SAVE_PATH_DATA:
        os.makedirs(os.path.dirname(PATH_DATA_FILE) or '.', exist_ok=True)
        # 미터 → 알고리즘 단위 (1 unit = CORE_R) 변환 후 저장
        path_algo = path / CORE_R
        np.save(PATH_DATA_FILE, path_algo)
        print(f"  경로 저장 완료: {PATH_DATA_FILE}  "
              f"(scale: {PATH_SCALE}m / {CORE_R*1000:.1f}mm = "
              f"{PATH_SCALE/CORE_R:.1f} algo units)")

    # ── 환경 생성 ────────────────────────────────────────────
    env = BallPlateEnv(
        object_type=OBJECT_TYPE,
        path_2d=path,
        render=True,
    )

    # ── 제어기 생성 ──────────────────────────────────────────
    if OBJECT_TYPE == 'trajectoid':
        ctrl = TrajectoidSpeedController(
            path=path,
            target_speed=TARGET_SPEED,
            dt=TIMESTEP * SUBSTEPS,
        )
    else:
        ctrl = PathFollowController(
            object_type=OBJECT_TYPE,
            path=path,
            dt=TIMESTEP * SUBSTEPS,
        )

    # ── 초기화 ───────────────────────────────────────────────
    obs = env.reset()
    ctrl.reset()

    print("\n  시뮬레이션 시작 (뷰어 창 q=종료)\n")
    if OBJECT_TYPE == 'trajectoid':
        print(
            f"  {'step':>6s}  {'t(s)':>6s}  "
            f"{'ball_x(mm)':>10s}  {'ball_y(mm)':>10s}  "
            f"{'speed(mm/s)':>11s}  {'tilt(°)':>7s}  "
            f"{'cross_err(mm)':>13s}"
        )
        print("  " + "-" * 82)
    else:
        print(
            f"  {'step':>6s}  {'t(s)':>6s}  "
            f"{'ball_x(mm)':>10s}  {'ball_y(mm)':>10s}  "
            f"{'cross_err(mm)':>13s}  "
            f"{'roll_y(°)':>9s}  {'pitch_x(°)':>10s}"
        )
        print("  " + "-" * 78)

    step      = 0
    t_sim     = 0.0
    total_err = 0.0
    max_err   = 0.0

    while env.is_running and t_sim < MAX_SIM_TIME:

        # ── 제어 계산 ──────────────────────────────────────────
        roll_y, pitch_x = ctrl.compute(obs)

        # ── 시뮬레이션 스텝 ────────────────────────────────────
        obs = env.step(roll_y, pitch_x, n_substep=SUBSTEPS)

        # ── 뷰어 마커 (Look-ahead 점) ──────────────────────────
        if env._viewer and env._viewer.is_running():
            la_world = _plate_local_to_world(ctrl.lookahead_pt, env.model, env.data)
            nr_world = _plate_local_to_world(ctrl.nearest_pt,   env.model, env.data)
            _draw_markers(env._viewer, env.model, env.data, la_world, nr_world)

        # ── 오차 통계 ──────────────────────────────────────────
        abs_err    = abs(ctrl.cross_err)
        total_err += abs_err
        if abs_err > max_err:
            max_err = abs_err

        # ── 터미널 로그 ────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            bp = obs['ball_pos_plate'] * 1000  # → mm
            if OBJECT_TYPE == 'trajectoid':
                speed_mm = float(np.linalg.norm(obs['ball_vel_plate'])) * 1000
                print(
                    f"  {step:>6d}  {obs['time']:>6.2f}  "
                    f"{bp[0]:>+10.1f}  {bp[1]:>+10.1f}  "
                    f"{speed_mm:>11.1f}  "
                    f"{np.degrees(ctrl.tilt_mag):>7.2f}  "
                    f"{ctrl.cross_err*1000:>+13.1f}"
                )
            else:
                print(
                    f"  {step:>6d}  {obs['time']:>6.2f}  "
                    f"{bp[0]:>+10.1f}  {bp[1]:>+10.1f}  "
                    f"{ctrl.cross_err*1000:>+13.1f}  "
                    f"{np.degrees(obs['plate_roll']):>+9.1f}  "
                    f"{np.degrees(obs['plate_pitch']):>+10.1f}"
                )

        # ── 평판 이탈 감지 → 리셋 ──────────────────────────────
        ball_dist = float(np.linalg.norm(obs['ball_pos_plate']))
        if ball_dist > PLATE_HALF * 0.93:
            print(f"\n  [!] 공이 평판 가장자리 이탈 (r={ball_dist*1000:.0f} mm) → reset\n")
            obs = env.reset()
            ctrl.reset()

        t_sim += TIMESTEP * SUBSTEPS
        step  += 1

    # ── 결과 요약 ────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  시뮬레이션 종료")
    print(f"  총 스텝     : {step}")
    print(f"  시뮬 시간   : {t_sim:.1f} s")
    if step > 0:
        print(f"  평균 횡오차 : {total_err/step*1000:.2f} mm")
        print(f"  최대 횡오차 : {max_err*1000:.2f} mm")
    print("=" * 58)

    env.close()


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == '__main__':
    run_simulation()
