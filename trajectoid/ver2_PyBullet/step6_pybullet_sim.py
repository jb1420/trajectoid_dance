"""
Step 6: PyBullet 물리 시뮬레이션
=================================

Trajectoid를 경사면 또는 2축 기울기 판 위에서 굴리는 시뮬레이션.

v1 대체 대상:
    - step2_3_mujoco_env.py (MuJoCo 환경)
    - step_rolling_inclined.py (경사면 시뮬레이션)
    - step4_control.py (제어 로직)

v1 미끄러짐 문제 해결:
    1. 볼록 메쉬: collision = visual 정확 일치 (CoACD 불필요)
    2. enableConeFriction: 정확한 마찰 원뿔
    3. 높은 솔버 반복 (150회)
    4. 작은 타임스텝 (1ms)
"""

import os
import time
import logging
import tempfile
import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError("pybullet이 설치되지 않았습니다.\n  pip install pybullet")

from .config import (
    BALL_R, CORE_R, MESH_SCALE,
    SHELL_DENSITY, BEARING_DENSITY,
    SIM_TIMESTEP, SOLVER_ITERATIONS, GRAVITY,
    SLOPE_DEG,
    LATERAL_FRICTION, SPINNING_FRICTION, ROLLING_FRICTION,
    RESTITUTION, CONTACT_STIFFNESS, CONTACT_DAMPING,
    RAMP_HALF_LEN, RAMP_HALF_WID, RAMP_HALF_THICK, RAMP_POS_Z,
    PLATE_HALF, PLATE_THICK, PLATE_H, PLATE_MAX_TILT,
    PLATE_KP, PLATE_KV,
    SIM_DURATION, SIM_SUBSTEPS,
)

log = logging.getLogger(__name__)


# ================================================================
#  복합 질량·관성 계산 (PLA 외피 + 강철 볼베어링)
# ================================================================
def compute_inertial(R_outer=BALL_R, R_core=CORE_R):
    """
    트레젝토이드 전체 질량(kg)과 관성 모멘트(kg·m²).

    모델: 속이 빈 PLA 외피 (홈 30% 부피 감소) + 강철 볼베어링
    """
    V_outer = 4 / 3 * np.pi * R_outer**3
    V_inner = 4 / 3 * np.pi * R_core**3
    V_shell = (V_outer - V_inner) * 0.70
    m_shell = V_shell * SHELL_DENSITY
    R_avg = (R_outer + R_core) / 2
    I_shell = 2 / 3 * m_shell * R_avg**2

    m_core = V_inner * BEARING_DENSITY
    I_core = 2 / 5 * m_core * R_core**2

    mass = m_shell + m_core
    inertia = I_shell + I_core
    return mass, [inertia, inertia, inertia]


# ================================================================
#  URDF 생성 (기울기 판용)
# ================================================================
def _generate_tilting_plate_urdf():
    """2축 기울기 판 URDF 문자열 생성."""
    half = PLATE_HALF
    thick = PLATE_THICK
    h = PLATE_H
    limit = PLATE_MAX_TILT
    # 판 관성 (균일 밀도 직육면체)
    mass = 2.0
    ixx = mass / 12 * ((2 * half)**2 + thick**2)
    iyy = mass / 12 * ((2 * half)**2 + thick**2)
    izz = mass / 12 * ((2 * half)**2 + (2 * half)**2)

    return f"""<?xml version="1.0"?>
<robot name="tilting_plate">
  <link name="world"/>

  <!-- Y축 회전 (roll) -->
  <joint name="roll_y" type="revolute">
    <parent link="world"/>
    <child link="roll_frame"/>
    <origin xyz="0 0 {h}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{-limit}" upper="{limit}" effort="500" velocity="10"/>
    <dynamics damping="0.5"/>
  </joint>
  <link name="roll_frame">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- X축 회전 (pitch) -->
  <joint name="pitch_x" type="revolute">
    <parent link="roll_frame"/>
    <child link="plate"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="{-limit}" upper="{limit}" effort="500" velocity="10"/>
    <dynamics damping="0.5"/>
  </joint>
  <link name="plate">
    <visual>
      <geometry>
        <box size="{2*half} {2*half} {thick}"/>
      </geometry>
      <material name="plate_mat">
        <color rgba="0.45 0.60 0.75 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="{2*half} {2*half} {thick}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>"""


# ================================================================
#  시뮬레이션 환경
# ================================================================
class TrajectoidSimulation:
    """
    PyBullet 기반 trajectoid 구름 시뮬레이션.

    Parameters
    ----------
    mesh_path : str — trajectoid OBJ/STL 파일 경로
    mode : 'inclined' | 'tilting'
    slope_deg : float — 경사각 (inclined 모드)
    gui : bool — GUI 표시 여부
    """

    def __init__(self, mesh_path: str, mode: str = 'inclined',
                 slope_deg: float = SLOPE_DEG, gui: bool = True):
        self.mesh_path = mesh_path
        self.mode = mode
        self.slope_deg = slope_deg
        self.slope_rad = np.radians(slope_deg)
        self.gui = gui
        self._sim_time = 0.0

        # 데이터 기록
        self.history = {
            'time': [],
            'ball_pos': [],
            'ball_orn': [],
            'ball_lin_vel': [],
            'ball_ang_vel': [],
            'contact_points': [],
        }

        self._setup()

    def _setup(self):
        """PyBullet 초기화."""
        if self.gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(SIM_TIMESTEP)
        p.setPhysicsEngineParameter(
            numSolverIterations=SOLVER_ITERATIONS,
            numSubSteps=SIM_SUBSTEPS,
            enableConeFriction=True,
        )

        # 바닥면
        self.ground_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.ground_id, -1, lateralFriction=0.5)

        if self.mode == 'inclined':
            self._create_inclined_plane()
            log.info(f"경사면 생성: {self.slope_deg}° 경사")
        elif self.mode == 'tilting':
            self._create_tilting_plate()
            log.info(f"기울기 판 생성")
        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

        self._create_trajectoid()
        self._configure_friction()

        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self._reset_camera()

        log.info(f"PyBullet 환경 초기화 완료 (mode={self.mode})")

    # ── 카메라 설정 ────────────────────────────────────────

    def _reset_camera(self):
        """카메라를 trajectoid가 보이는 위치로 설정."""
        if not self.gui:
            return
        start_pos = self._compute_start_position()
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,          # 물체에서 0.5m 거리
            cameraYaw=45,                # 수평 회전
            cameraPitch=-20,             # 위에서 내려다보기
            cameraTargetPosition=start_pos,
        )
        log.info(f"  카메라 리셋: target={[round(x,3) for x in start_pos]}")

    # ── 경사면 생성 ─────────────────────────────────────────

    def _create_inclined_plane(self):
        """고정 경사면 생성."""
        half_extents = [RAMP_HALF_LEN, RAMP_HALF_WID, RAMP_HALF_THICK]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.45, 0.60, 0.75, 1.0],
        )
        orn = p.getQuaternionFromEuler([0, -self.slope_rad, 0])

        self.plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, RAMP_POS_Z],
            baseOrientation=orn,
        )

    # ── 기울기 판 생성 ─────────────────────────────────────

    def _create_tilting_plate(self):
        """2축 기울기 판 (URDF)."""
        urdf_str = _generate_tilting_plate_urdf()
        # 임시 URDF 파일 저장
        self._urdf_tmp = tempfile.NamedTemporaryFile(
            suffix='.urdf', mode='w', delete=False
        )
        self._urdf_tmp.write(urdf_str)
        self._urdf_tmp.close()

        self.plate_id = p.loadURDF(
            self._urdf_tmp.name,
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )

        # 조인트 인덱스 확인
        self.joint_roll_y = None
        self.joint_pitch_x = None
        for i in range(p.getNumJoints(self.plate_id)):
            info = p.getJointInfo(self.plate_id, i)
            name = info[1].decode('utf-8')
            if name == 'roll_y':
                self.joint_roll_y = i
            elif name == 'pitch_x':
                self.joint_pitch_x = i

        log.info(f"  기울기 판 조인트: roll_y={self.joint_roll_y}, "
                 f"pitch_x={self.joint_pitch_x}")

        # 경사면 ID 대용 (마찰 설정용)
        self.plane_id = self.plate_id

    # ── Trajectoid 생성 ────────────────────────────────────

    def _create_trajectoid(self):
        """Trajectoid 메쉬 로드 + 물리 바디 생성."""
        import os

        # 절대 경로로 변환
        mesh_abs = os.path.abspath(self.mesh_path)
        if not os.path.exists(mesh_abs):
            raise FileNotFoundError(f"메쉬 파일 없음: {mesh_abs}")

        log.info(f"  메쉬 로드: {mesh_abs}")

        # 메쉬가 볼록체이므로 직접 convex collision shape 사용
        try:
            col_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=mesh_abs,
                meshScale=[MESH_SCALE] * 3,
            )
            log.info(f"    충돌 형상 생성 성공")
        except Exception as e:
            log.error(f"    충돌 형상 생성 실패: {e}")
            # 폴백: 구(sphere)로 대체
            log.warning("    구로 폴백합니다")
            col_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=BALL_R,
            )

        vis_shape = -1
        try:
            vis_shape = p.createVisualShape(
                p.GEOM_MESH,
                fileName=mesh_abs,
                meshScale=[MESH_SCALE] * 3,
                rgbaColor=[0.85, 0.35, 0.18, 1.0],
            )
            if vis_shape < 0:
                raise RuntimeError("createVisualShape returned -1")
            log.info(f"    시각 형상 생성 성공")
        except Exception as e:
            log.error(f"    시각 형상 생성 실패: {e}")
            # 폴백: 구로 대체
            vis_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=BALL_R,
                rgbaColor=[0.85, 0.35, 0.18, 1.0],
            )
            log.warning(f"    구(sphere)로 폴백")

        mass, inertia = compute_inertial()

        # 시작 위치: 경사면 위 (모드별 계산)
        start_pos = self._compute_start_position()

        self.ball_id = p.createMultiBody(
            baseMass=mass,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=start_pos,
            baseOrientation=[0, 0, 0, 1],
        )
        # localInertiaDiagonal 오버라이드 제거 — PyBullet이 메쉬에서 자동 계산하게 둠
        # 수동 오버라이드는 질량/관성 불일치를 유발하여 솔버 동결을 일으킬 수 있음

        log.info(f"  Trajectoid 생성: mass={mass:.4f} kg, "
                 f"I(expected)={inertia[0]:.2e} kg·m²")
        log.info(f"  시작 위치: {start_pos}")

    def _compute_start_position(self):
        """모드별 초기 위치 계산.

        경사면(inclined):
            경사면 박스 중심 = [0, 0, RAMP_POS_Z], Y축 -slope_rad 회전
            박스 상단 표면의 z 좌표 = 경사면 중심 z + RAMP_HALF_THICK * cos(slope)
            x 위치에서의 z 오프셋 = x * sin(slope)
        """
        if self.mode == 'inclined':
            x_start = RAMP_HALF_LEN * 0.5   # 경사면 길이의 절반 지점 (중앙)
            # 경사면 박스 상단 표면: z = RAMP_POS_Z + RAMP_HALF_THICK + x*sin(slope)
            z_surface = (RAMP_POS_Z
                         + RAMP_HALF_THICK * np.cos(self.slope_rad)
                         + x_start * np.sin(self.slope_rad))
            z_ball = z_surface + BALL_R + 0.01   # 표면 바로 위
            log.info(f"    경사면 표면 z={z_surface:.4f}, 공 중심 z={z_ball:.4f}")
            return [x_start, 0.0, z_ball]
        else:
            # 기울기 판 중앙 위
            z_ball = PLATE_H + PLATE_THICK / 2 + BALL_R + 0.01
            log.info(f"    판 표면 z={PLATE_H + PLATE_THICK/2:.4f}, 공 중심 z={z_ball:.4f}")
            return [0.0, 0.0, z_ball]

    # ── 마찰 설정 ──────────────────────────────────────────

    def _configure_friction(self):
        """경사면/판 + trajectoid 양쪽에 마찰 설정."""
        bodies = [self.ball_id]
        if self.mode == 'inclined':
            bodies.append(self.plane_id)
        elif self.mode == 'tilting':
            bodies.append(self.plate_id)

        for body in bodies:
            if isinstance(body, tuple):
                bid, lid = body
            else:
                bid, lid = body, -1

            try:
                p.changeDynamics(
                    bid, lid,
                    lateralFriction=LATERAL_FRICTION,
                    spinningFriction=SPINNING_FRICTION,
                    rollingFriction=ROLLING_FRICTION,
                    restitution=RESTITUTION,
                    # contactStiffness와 contactDamping은 선택적
                    # contactStiffness=CONTACT_STIFFNESS,
                    # contactDamping=CONTACT_DAMPING,
                )
                log.info(f"    마찰 설정 적용: lateral={LATERAL_FRICTION}, "
                         f"rolling={ROLLING_FRICTION}")
            except Exception as e:
                log.warning(f"    마찰 설정 실패: {e}")

    # ── 기울기 판 제어 ──────────────────────────────────────

    def set_plate_tilt(self, roll_y: float, pitch_x: float):
        """기울기 판 각도 설정 (tilting 모드)."""
        if self.mode != 'tilting':
            return
        roll_y = np.clip(roll_y, -PLATE_MAX_TILT, PLATE_MAX_TILT)
        pitch_x = np.clip(pitch_x, -PLATE_MAX_TILT, PLATE_MAX_TILT)

        if self.joint_roll_y is not None:
            p.setJointMotorControl2(
                self.plate_id, self.joint_roll_y,
                p.POSITION_CONTROL,
                targetPosition=roll_y,
                force=500,
                positionGain=PLATE_KP / 1000,
                velocityGain=PLATE_KV / 1000,
            )
        if self.joint_pitch_x is not None:
            p.setJointMotorControl2(
                self.plate_id, self.joint_pitch_x,
                p.POSITION_CONTROL,
                targetPosition=pitch_x,
                force=500,
                positionGain=PLATE_KP / 1000,
                velocityGain=PLATE_KV / 1000,
            )

    # ── 시뮬레이션 스텝 ─────────────────────────────────────

    def step(self):
        """한 제어 주기 진행."""
        for _ in range(SIM_SUBSTEPS):
            p.stepSimulation()
        self._sim_time += SIM_TIMESTEP * SIM_SUBSTEPS
        self._record()

    def _debug_info(self) -> str:
        """디버그 정보 출력."""
        pos, orn = p.getBasePositionAndOrientation(self.ball_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.ball_id)
        contacts = p.getContactPoints(self.ball_id, self.plane_id if self.mode == 'inclined' else self.plate_id)
        return (f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                f"vel=({lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}) "
                f"contacts={len(contacts)}")

    def _record(self):
        """현재 상태 기록."""
        pos, orn = p.getBasePositionAndOrientation(self.ball_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.ball_id)

        self.history['time'].append(self._sim_time)
        self.history['ball_pos'].append(np.array(pos))
        self.history['ball_orn'].append(np.array(orn))
        self.history['ball_lin_vel'].append(np.array(lin_vel))
        self.history['ball_ang_vel'].append(np.array(ang_vel))

        # 접촉점 기록
        if self.mode == 'inclined':
            contacts = p.getContactPoints(self.ball_id, self.plane_id)
        else:
            contacts = p.getContactPoints(self.ball_id, self.plate_id)

        if contacts:
            # positionOnB = 경사면/판 위의 접촉점
            self.history['contact_points'].append(
                np.array([c[6] for c in contacts])  # positionOnB
            )
        else:
            self.history['contact_points'].append(np.array([]).reshape(0, 3))

    # ── 관찰값 ──────────────────────────────────────────────

    def get_observation(self) -> dict:
        """현재 상태 딕셔너리."""
        pos, orn = p.getBasePositionAndOrientation(self.ball_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.ball_id)
        return {
            'time': self._sim_time,
            'ball_pos': np.array(pos),
            'ball_orn': np.array(orn),
            'ball_lin_vel': np.array(lin_vel),
            'ball_ang_vel': np.array(ang_vel),
        }

    def get_contact_trace(self) -> np.ndarray:
        """누적 접촉 궤적 (경사면/판 좌표)."""
        all_pts = []
        for pts in self.history['contact_points']:
            if len(pts) > 0:
                all_pts.extend(pts.tolist() if pts.ndim > 1 else [pts.tolist()])
        if not all_pts:
            return np.array([]).reshape(0, 3)
        return np.array(all_pts)

    # ── 메인 실행 루프 ──────────────────────────────────────

    def run(self, duration: float = SIM_DURATION,
            tilt_controller=None,
            realtime: bool = True) -> dict:
        """
        시뮬레이션 메인 루프.

        Parameters
        ----------
        duration : 시뮬레이션 시간 (s)
        tilt_controller : callable(obs) -> (roll_y, pitch_x) 또는 None
        realtime : GUI 실시간 동기화 여부

        Returns
        -------
        history : 기록된 상태 딕셔너리
        """
        n_steps = int(duration / (SIM_TIMESTEP * SIM_SUBSTEPS))
        log.info(f"시뮬레이션 시작: {duration}s, {n_steps} 제어 스텝")

        step_dt = SIM_TIMESTEP * SIM_SUBSTEPS
        wall_start = time.time()

        for i in range(n_steps):
            # 기울기 판 제어
            if tilt_controller is not None and self.mode == 'tilting':
                obs = self.get_observation()
                roll_y, pitch_x = tilt_controller(obs)
                self.set_plate_tilt(roll_y, pitch_x)

            self.step()

            # 실시간 동기화
            if realtime and self.gui:
                elapsed = time.time() - wall_start
                target = (i + 1) * step_dt
                if target > elapsed:
                    time.sleep(target - elapsed)

            # 주기적 상태 출력
            if (i + 1) % 500 == 0 or i < 5:  # 처음 몇 스텝 + 주기적
                obs = self.get_observation()
                pos = obs['ball_pos']
                speed = np.linalg.norm(obs['ball_lin_vel'])
                if self.mode == 'inclined':
                    contacts = p.getContactPoints(self.ball_id, self.plane_id)
                else:
                    contacts = p.getContactPoints(self.ball_id, self.plate_id)
                log.info(f"  step={i+1:6d} t={self._sim_time:.2f}s  "
                         f"pos=({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f})  "
                         f"speed={speed:.4f} m/s  contacts={len(contacts)}")

            # 공이 바닥으로 떨어졌으면 중단
            pos = p.getBasePositionAndOrientation(self.ball_id)[0]
            if pos[2] < -2.0:  # 낙하 조건 완화
                log.warning(f"  공 낙하 감지 (z={pos[2]:.3f}), 시뮬 중단")
                break

        log.info(f"시뮬레이션 완료: {self._sim_time:.2f}s")

        return {k: np.array(v) if k != 'contact_points' else v
                for k, v in self.history.items()}

    # ── 정리 ────────────────────────────────────────────────

    def close(self):
        """PyBullet 연결 해제."""
        p.disconnect(self.client)
        if hasattr(self, '_urdf_tmp') and os.path.exists(self._urdf_tmp.name):
            os.unlink(self._urdf_tmp.name)


# ================================================================
#  기울기 판 제어기 (경로 추종)
# ================================================================
class PathFollowController:
    """
    경로 접선 방향으로 기울기를 유지하는 제어기.

    Parameters
    ----------
    path : (N, 2) ndarray — 경로 [m]
    target_speed : float — 목표 속도 (m/s)
    """

    def __init__(self, path: np.ndarray,
                 target_speed: float = 0.25,
                 kp: float = 2.0, ki: float = 0.05, kd: float = 0.3,
                 tilt_min: float = 0.005, tilt_max: float = 0.28):
        self.path = path
        self.target_speed = target_speed
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max

        # 접선 각도 사전 계산
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        self.tangent_angles = np.arctan2(dy, dx)

        self._integral = 0.0
        self._prev_error = 0.0
        self._path_idx = 0

    def __call__(self, obs: dict) -> tuple:
        """
        관찰값으로부터 (roll_y, pitch_x) 각도 계산.

        기울기 방향 = 현재 가장 가까운 경로점의 접선 방향
        기울기 크기 = PID(target_speed - current_speed)
        """
        ball_pos = obs['ball_pos'][:2]  # xy
        ball_vel = obs['ball_lin_vel'][:2]
        speed = np.linalg.norm(ball_vel)

        # 가장 가까운 경로점 찾기
        dists = np.linalg.norm(self.path - ball_pos, axis=1)
        self._path_idx = np.argmin(dists)

        # 접선 방향
        phi = self.tangent_angles[self._path_idx]

        # 속도 PID
        dt = SIM_TIMESTEP * SIM_SUBSTEPS
        error = self.target_speed - speed
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0
        self._prev_error = error

        tilt_mag = (self.kp * error
                    + self.ki * self._integral
                    + self.kd * derivative)
        tilt_mag = np.clip(tilt_mag, self.tilt_min, self.tilt_max)

        roll_y = tilt_mag * np.cos(phi)
        pitch_x = -tilt_mag * np.sin(phi)

        return roll_y, pitch_x
