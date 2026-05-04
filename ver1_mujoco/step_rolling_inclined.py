"""
5도 경사면 구름 물리 시뮬레이션
================================

Step 1에서 생성된 STL/OBJ 파일을 5도 기울기 평면에서 굴리는 MuJoCo 시뮬레이션.

물리 조건:
  - 중력       : 9.81 m/s² (−Z 방향)
  - 경사각     : 5° (기본값, SLOPE_DEG 로 변경 가능)
  - 미끄럼 없는 구름(rolling without slipping) 조건:
      condim=6, 높은 접선 마찰계수(FRIC_SLIDE ≥ 1.0) 적용
      이론 조건: μ > (2/7) * tan(5°) ≈ 0.063 이상이면 순수 구름

경사면 구성:
  - 상단(+X) → 하단(-X) 방향으로 경사
  - euler="0 -{SLOPE_DEG} 0" : Y축 기준 음의 회전 → +X 끝이 위쪽(uphill)
  - 공은 상단에서 시작해 −X 방향으로 굴러내려감

실행:
    python step_rolling_inclined.py

필수 라이브러리:
    pip install mujoco trimesh coacd
"""

import io
import os
import sys
import time
import textwrap
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit(
        "[오류] mujoco 가 설치되지 않았습니다.\n"
        "  pip install mujoco\n"
        "설치 후 다시 실행하세요."
    )


# ================================================================
#  SETTINGS
# ================================================================
OBJECT_TYPE      = 'trajectoid'       # 'sphere' | 'trajectoid'
TRAJECTOID_MESH  = 'output/trajectoid.obj'

SLOPE_DEG        = 5.0                # 경사각 (degrees)
SLOPE_RAD        = np.radians(SLOPE_DEG)

# 경사면 치수 (m)
RAMP_LEN_HALF    = 1.5                # 경사면 반길이 → 총 3.0 m
RAMP_WID_HALF    = 0.4                # 경사면 반너비
RAMP_THICK_HALF  = 0.05              # 경사면 반두께
RAMP_POS_Z       = 0.55              # 경사면 중심 World-Z

# 공 시작 위치: 경사면 상단에서 이 비율만큼 내측
BALL_START_RATIO = 0.80               # 상단 끝에서 80% 위치

# 객체 치수 (m) — step2_3 과 동일
BALL_R           = 0.015875           # outer sphere 반지름 (15.875 mm)
CORE_R           = 0.0127             # 볼베어링 반지름 (12.7 mm)
MESH_SCALE       = CORE_R

# 밀도 (kg/m³)
SHELL_DENSITY    = 1240.0             # PLA
BEARING_DENSITY  = 7874.0             # 강철

# ── 마찰 파라미터 (미끄럼 방지 핵심) ────────────────────────────
# 5도 경사에서 구름 조건: μ_slide > 2/7 * tan(5°) ≈ 0.063
# FRIC_SLIDE = 5.0 은 충분히 크므로 슬립 없이 구름
FRIC_SLIDE       = 5.0                # 접선 미끄럼 마찰계수 (높을수록 미끄럼 억제)
FRIC_SPIN        = 0.1                # 수직축 스핀 마찰계수
FRIC_ROLL        = 0.005             # 구름(rolling) 마찰계수 (너무 크면 멈춤)

# CoACD 파라미터 (trajectoid 충돌 분해)
COACD_THRESHOLD  = 0.05
COACD_MAX_CONVEX = 32

# Solver
TIMESTEP         = 0.002              # 시뮬레이션 time step (s)
SOLVER_ITER      = 100                # 반복 횟수 (구름 정밀도 향상)


# 재생 배속 (1.0 = 실시간, 10.0 = 10배속, 0 = 최대 속도)
# 5도 경사는 매우 느리므로 10배속 이상 권장
SIM_SPEED        = 10.0


# 시뮬레이션 지속 시간 (s)
SIM_DURATION     = 100.0 * SIM_SPEED


# ================================================================
#  헬퍼 함수
# ================================================================

def _fric(s=FRIC_SLIDE, sp=FRIC_SPIN, r=FRIC_ROLL) -> str:
    return f"{s} {sp} {r}"


def _quat_to_euler_deg(quat_wxyz: np.ndarray) -> tuple:
    """
    쿼터니언 (w, x, y, z) → 내재적 X-Y-Z 오일러 각도 (도 단위).

    반환값:
      roll_x  : X축 회전 (도) — 좌우 기울기
      pitch_y : Y축 회전 (도) — 경사면 구름 방향 (핵심 값)
      yaw_z   : Z축 회전 (도) — 수직축 회전
    """
    w, qx, qy, qz = quat_wxyz

    # Roll (X축)
    sinr = 2.0 * (w * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx**2 + qy**2)
    roll_x = np.degrees(np.arctan2(sinr, cosr))

    # Pitch (Y축) — 경사면 방향 구름에 해당
    sinp = np.clip(2.0 * (w * qy - qz * qx), -1.0, 1.0)
    pitch_y = np.degrees(np.arcsin(sinp))

    # Yaw (Z축)
    siny = 2.0 * (w * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw_z = np.degrees(np.arctan2(siny, cosy))

    return roll_x, pitch_y, yaw_z


def _compute_inertial() -> tuple:
    """트레젝토이드 전체 질량(kg)과 등방 관성 모멘트(kg·m²) 계산."""
    V_outer = 4/3 * np.pi * BALL_R**3
    V_inner = 4/3 * np.pi * CORE_R**3
    V_shell = (V_outer - V_inner) * 0.70
    m_shell = V_shell * SHELL_DENSITY
    R_avg   = (BALL_R + CORE_R) / 2
    I_shell = 2/3 * m_shell * R_avg**2

    m_core  = V_inner * BEARING_DENSITY
    I_core  = 2/5 * m_core * CORE_R**2

    return (m_shell + m_core), (I_shell + I_core)


def _coacd_geoms(mesh_bytes: bytes) -> tuple:
    """
    .obj 바이트 → CoACD 볼록 분해 → (geoms_xml, assets_dict)

    - 시각 메쉬 (contype=0) + N개 볼록 충돌 조각 (condim=6)
    - MESH_SCALE 적용 (알고리즘 단위 → 미터)
    """
    try:
        import trimesh as _trimesh
    except ImportError:
        raise ImportError("trimesh 없음.  pip install trimesh")
    try:
        import coacd
    except ImportError:
        raise ImportError("coacd 없음.  pip install coacd")

    print("  [CoACD] 메쉬 로딩 및 스케일 적용 중 …")
    mesh = _trimesh.load(io.BytesIO(mesh_bytes), file_type='obj', force='mesh')
    mesh.apply_scale(MESH_SCALE)
    _trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        print("  [CoACD] 경고: 메쉬가 watertight 하지 않음 — 결과가 부정확할 수 있음")

    print(f"  [CoACD] 볼록 분해 시작 … "
          f"(threshold={COACD_THRESHOLD}, max_parts={COACD_MAX_CONVEX})")
    m     = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        m,
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_CONVEX,
    )
    print(f"  [CoACD] {len(parts)} 개 볼록 조각 생성")

    assets: dict = {}
    geom_lines: list = []

    # 시각 메쉬 (충돌 없음)
    bio_vis = io.BytesIO()
    mesh.export(bio_vis, file_type='obj')
    assets['traj_visual.obj'] = bio_vis.getvalue()
    geom_lines.append(
        '<geom name="visual" type="mesh" mesh="traj_visual" '
        'contype="0" conaffinity="0" '
        'rgba="0.85 0.35 0.18 0.90"/>'
    )

    # 볼록 충돌 조각
    fric_str = _fric()
    for i, (verts, faces) in enumerate(parts):
        key  = f'traj_cvx_{i:03d}.obj'
        part = _trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
        bio  = io.BytesIO()
        part.export(bio, file_type='obj')
        assets[key] = bio.getvalue()

        rgba = "0.45 0.75 1.0 0.12" if i == 0 else "0 0 0 0"
        geom_lines.append(
            f'<geom name="cvx_{i:03d}" type="mesh" mesh="traj_cvx_{i:03d}" '
            f'condim="6" friction="{fric_str}" '
            f'solref="0.005 1" solimp="0.99 0.999 0.001 0.5 2" '
            f'rgba="{rgba}"/>'
        )

    return "\n          ".join(geom_lines), assets


# ================================================================
#  공 초기 위치 계산
# ================================================================

def _compute_ball_start_pos() -> tuple:
    """
    경사면 상단 위 공의 초기 World 위치 계산.

    경사면 설정:
      body pos = (0, 0, RAMP_POS_Z),  euler = "0 -SLOPE_DEG 0"

    R_y(-θ) 행렬:
      [cos θ   0  -sin θ]
      [  0     1    0   ]
      [sin θ   0   cos θ]

    → 로컬 +X 방향이 World에서 (cos θ, 0, sin θ) → 위쪽(uphill) ✓
    """
    c = np.cos(SLOPE_RAD)
    s = np.sin(SLOPE_RAD)

    # R_y(-θ) 행렬 (로컬 → 월드 변환)
    Ry_neg = np.array([
        [ c, 0., -s],
        [0., 1., 0.],
        [ s, 0.,  c],
    ])

    # 경사면 상단 로컬 좌표 (상단 표면)
    x_local = RAMP_LEN_HALF * BALL_START_RATIO
    z_local = RAMP_THICK_HALF

    local_surface = np.array([x_local, 0.0, z_local])
    world_surface = Ry_neg @ local_surface + np.array([0., 0., RAMP_POS_Z])

    # 표면 법선 (로컬 +Z → 월드)
    normal_world = Ry_neg @ np.array([0., 0., 1.])

    # 공 중심 = 표면 점 + BALL_R * 법선
    ball_center = world_surface + BALL_R * normal_world

    return float(ball_center[0]), float(ball_center[1]), float(ball_center[2])


# ================================================================
#  MJCF XML 생성
# ================================================================

def build_mjcf(object_type: str) -> tuple:
    """
    5도 경사면 + 구르는 물체 MJCF XML 생성.

    경사면 구조:
      - body name="ramp"  pos="0 0 {RAMP_POS_Z}"  euler="0 -{SLOPE_DEG} 0"
        - geom type="box"  (정적 고정 바디, 관절 없음)
      - body name="ball"  (freejoint)

    미끄럼 방지:
      - condim=6  (접선 + 스핀 + 구름 마찰 모두 활성화)
      - FRIC_SLIDE >> min 요구치  →  슬립 없이 구름
    """
    ball_x, ball_y, ball_z = _compute_ball_start_pos()
    print(f"  공 초기 위치 (World): ({ball_x:.4f}, {ball_y:.4f}, {ball_z:.4f})")

    assets: dict = {}
    fric  = _fric()
    slope = SLOPE_DEG

    # ── 물체별 블록 ──────────────────────────────────────────────
    if object_type == 'sphere':
        asset_block  = ""
        object_block = textwrap.dedent(f"""\
            <!-- ── 단순 구 (sphere) ── -->
            <body name="ball" pos="{ball_x:.6f} {ball_y:.6f} {ball_z:.6f}">
              <freejoint name="ball_joint"/>
              <geom name="ball_geom"
                    type="sphere"
                    size="{BALL_R:.6f}"
                    density="2000"
                    friction="{fric}"
                    condim="6"
                    solref="0.005 1"
                    solimp="0.99 0.999 0.001 0.5 2"
                    rgba="0.85 0.25 0.20 1.0"/>
            </body>""")

    elif object_type == 'trajectoid':
        mesh_path = os.path.abspath(TRAJECTOID_MESH)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(
                f"Trajectoid 메쉬를 찾을 수 없습니다: {mesh_path}\n"
                "먼저 step1_boolean_subtraction.py 를 실행하세요."
            )
        with open(mesh_path, 'rb') as f:
            mesh_bytes = f.read()

        geoms_xml, assets = _coacd_geoms(mesh_bytes)
        mass_kg, I_kgm2   = _compute_inertial()
        I_str = f"{I_kgm2:.3e}"

        asset_lines = []
        for fname in assets:
            name = fname.replace('.obj', '').replace('.', '_')
            asset_lines.append(f'    <mesh name="{name}" file="{fname}"/>')
        asset_block = "  <asset>\n" + "\n".join(asset_lines) + "\n  </asset>"

        object_block = textwrap.dedent(f"""\
            <!-- ── Trajectoid (CoACD 충돌 분해) ── -->
            <body name="ball" pos="{ball_x:.6f} {ball_y:.6f} {ball_z:.6f}">
              <freejoint name="ball_joint"/>
              <inertial pos="0 0 0" mass="{mass_kg:.6f}"
                        diaginertia="{I_str} {I_str} {I_str}"/>
              {geoms_xml}
            </body>""")

    else:
        raise ValueError(f"OBJECT_TYPE 은 'sphere' 또는 'trajectoid': '{object_type}'")

    # ── MJCF 본문 ────────────────────────────────────────────────
    xml = textwrap.dedent(f"""\
        <mujoco model="rolling_inclined_{object_type}">

          <!-- ── 물리 설정 ── -->
          <option timestep="{TIMESTEP}"
                  gravity="0 0 -9.81"
                  integrator="implicitfast"
                  iterations="{SOLVER_ITER}"
                  tolerance="1e-10"/>

          <compiler autolimits="true"/>

          {asset_block}

          <!-- ── 전역 기본값: condim=6 (3D 마찰 원추 + 구름·스핀 마찰) ── -->
          <!-- FRIC_SLIDE=5.0 >> min 요구치(0.025) → 미끄럼 없는 순수 구름 보장 -->
          <default>
            <geom condim="6" friction="{fric}"
                  solref="0.005 1" solimp="0.99 0.999 0.001 0.5 2"/>
          </default>

          <worldbody>

            <!-- 바닥 평면: 공이 경사면에서 굴러떨어지면 받아줌 -->
            <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
                  friction="{fric}"
                  condim="6"
                  rgba="0.22 0.22 0.22 0.7"/>

            <!-- ┌────────────────────────────────────────────────────┐ -->
            <!-- │  5도 경사면 (정적 고정 — 관절 없음)               │ -->
            <!-- │  euler="0 -{slope:.1f} 0"                          │ -->
            <!-- │    → R_y(-{slope:.1f}°): 로컬 +X = 위쪽(uphill)   │ -->
            <!-- │    → 공은 상단(+X)에서 출발, -X 방향으로 굴러내려감 │ -->
            <!-- └────────────────────────────────────────────────────┘ -->
            <body name="ramp" pos="0 0 {RAMP_POS_Z:.4f}" euler="0 -{slope:.1f} 0">
              <geom name="ramp_surface"
                    type="box"
                    size="{RAMP_LEN_HALF:.4f} {RAMP_WID_HALF:.4f} {RAMP_THICK_HALF:.5f}"
                    friction="{fric}"
                    condim="6"
                    rgba="0.45 0.60 0.75 1.0"/>

              <!-- 경사면 하단 끝 표시 (시각화 전용) -->
              <site name="ramp_bottom" pos="-{RAMP_LEN_HALF:.4f} 0 {RAMP_THICK_HALF:.5f}"
                    size="0.015" type="sphere" rgba="1.0 0.3 0.3 0.9"/>
              <!-- 경사면 상단 끝 표시 (시각화 전용) -->
              <site name="ramp_top" pos="{RAMP_LEN_HALF:.4f} 0 {RAMP_THICK_HALF:.5f}"
                    size="0.015" type="sphere" rgba="0.3 1.0 0.3 0.9"/>
            </body>

            <!-- ── 구르는 물체 ── -->
            {object_block}

          </worldbody>

          <!-- ── 센서: 위치 / 선속도 / 각속도 ── -->
          <sensor>
            <framepos    name="ball_pos"    objtype="body" objname="ball"/>
            <framelinvel name="ball_vel"    objtype="body" objname="ball"/>
            <frameangvel name="ball_angvel" objtype="body" objname="ball"/>
          </sensor>

        </mujoco>""")

    return xml, assets


# ================================================================
#  RollingInclinedEnv 클래스
# ================================================================

class RollingInclinedEnv:
    """
    5도 고정 경사면 위에서 구 / 트레젝토이드가 구르는 MuJoCo 환경.

    물리 특징:
      - 경사면은 정적 강체 (관절 없음)
      - 공은 freejoint (6-DOF)
      - condim=6 + 높은 FRIC_SLIDE → 미끄럼 없는 순수 구름
    """

    def __init__(self, object_type: str = OBJECT_TYPE, render: bool = True):
        self.object_type = object_type
        self.render_flag = render

        print(f"[RollingInclinedEnv] MJCF 생성 중 … (object_type={object_type})")
        xml, assets = build_mjcf(object_type)

        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data  = mujoco.MjData(self.model)

        # ball body id
        self._ball_bid  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball'
        )
        # freejoint qpos 주소
        self._jadr_ball = int(self.model.joint('ball_joint').qposadr)

        # 초기 공 위치 (World)
        self._init_pos = _compute_ball_start_pos()

        # ── 회전 추적 변수 ───────────────────────────────────────
        # 각속도 센서를 적분해서 누적 회전각(rad) 보관
        self._theta_acc  = np.zeros(3)   # [θx, θy, θz] 누적 (rad)
        self._disp_acc   = 0.0           # 경사면 하강 방향 누적 이동거리 (m)
        self._last_time  = 0.0           # 이전 step 시각

        # viewer
        self._viewer = None
        if render:
            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            # 경사면을 옆에서 + 약간 위에서 보는 시점
            self._viewer.cam.azimuth   = 100.0
            self._viewer.cam.elevation = -18.0
            self._viewer.cam.distance  = 4.5
            self._viewer.cam.lookat[:] = [0.0, 0.0, RAMP_POS_Z + 0.1]

        mujoco.mj_resetData(self.model, self.data)
        self._place_ball()
        mujoco.mj_forward(self.model, self.data)

        print(f"[RollingInclinedEnv] 초기화 완료")
        print(f"  nq={self.model.nq}  nv={self.model.nv}  nsensor={self.model.nsensor}")

    # ── 공 배치 ─────────────────────────────────────────────────
    def _place_ball(self) -> None:
        """공을 경사면 상단에 배치 (속도=0, 정지 상태)."""
        x, y, z = self._init_pos
        qadr = self._jadr_ball
        self.data.qpos[qadr:qadr + 3] = [x, y, z]
        self.data.qpos[qadr + 3]      = 1.0    # quaternion w (단위 회전)
        self.data.qpos[qadr + 4:qadr + 7] = [0.0, 0.0, 0.0]
        # 속도 초기화
        self.data.qvel[:] = 0.0
        # 추적 변수 초기화
        self._theta_acc[:]  = 0.0
        self._disp_acc      = 0.0
        self._last_time     = 0.0

    # ── reset ────────────────────────────────────────────────────
    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self._place_ball()
        mujoco.mj_forward(self.model, self.data)
        if self._viewer and self._viewer.is_running():
            self._viewer.sync()
        return self.get_obs()

    # ── step ─────────────────────────────────────────────────────
    def step(self, n_substep: int = 1) -> dict:
        """n_substep 만큼 시뮬레이션 진행."""
        for _ in range(n_substep):
            mujoco.mj_step(self.model, self.data)
        if self._viewer and self._viewer.is_running():
            self._viewer.sync()
        return self.get_obs()

    # ── get_obs ──────────────────────────────────────────────────
    def get_obs(self) -> dict:
        """
        현재 관측값 반환.

        Keys
        ----
        pos               : (3,)  World 위치 [m]
        vel               : (3,)  World 선속도 [m/s]
        angvel            : (3,)  World 각속도 [rad/s]
        euler_deg         : (3,)  오일러 각도 [도] (roll_x, pitch_y, yaw_z)
        theta_acc         : (3,)  누적 회전각 [도] (angvel 적분)
        theta_expected_deg: float 이동거리 기준 기대 회전각 [도] (순수 구름 가정)
        roll_ratio        : float 실제/기대 구름 비율 (1=순수 구름, 0=완전 미끄럼)
        v_down            : float 하강 방향 속도 [m/s]
        v_roll            : float 순수 구름 예측 속도 [m/s]
        slip              : float 순간 미끄럼량 [m/s]
        slip_ratio        : float 순간 미끄럼 비율 (0=구름, 1=미끄럼)
        disp_acc          : float 누적 이동거리 [m]
        time              : float 시뮬레이션 시간 [s]
        on_ramp           : bool  경사면 위 여부
        """
        now    = self.data.time
        dt     = now - self._last_time
        self._last_time = now

        pos    = self.data.sensor('ball_pos').data.copy()
        vel    = self.data.sensor('ball_vel').data.copy()
        angvel = self.data.sensor('ball_angvel').data.copy()

        # ── 경사면 방향 벡터 ─────────────────────────────────────
        c = np.cos(SLOPE_RAD)
        s = np.sin(SLOPE_RAD)
        downhill = np.array([-c, 0.0, -s])

        # ── 선속도 & 순간 미끄럼 ─────────────────────────────────
        v_down  = float(np.dot(vel, downhill))
        omega_y = float(angvel[1])
        v_roll  = -omega_y * BALL_R   # 순수 구름: v_down = -omega_y * BALL_R
        slip    = abs(v_down - v_roll)
        slip_ratio = slip / max(abs(v_down) + abs(v_roll), 1e-6) * 2.0

        # ── 누적 회전각 & 이동거리 적분 ──────────────────────────
        if dt > 0:
            self._theta_acc += angvel * dt      # [rad]
            self._disp_acc  += abs(v_down) * dt  # [m]

        # 이동거리 기준 기대 Y 회전각 (순수 구름 가정)
        theta_expected_rad = self._disp_acc / BALL_R
        theta_expected_deg = np.degrees(theta_expected_rad)

        # roll_ratio: 실제 누적 Y 회전 / 기대 회전 (1이면 순수 구름)
        theta_y_actual = abs(self._theta_acc[1])
        roll_ratio = (theta_y_actual / theta_expected_rad
                      if theta_expected_rad > 1e-4 else 1.0)

        # ── 쿼터니언 → 오일러 각도 ──────────────────────────────
        qadr     = self._jadr_ball
        quat     = self.data.qpos[qadr + 3: qadr + 7]   # (w, qx, qy, qz)
        euler_deg = np.array(_quat_to_euler_deg(quat))

        # ── 경사면 위 여부 ────────────────────────────────────────
        Ry_pos = np.array([[ c, 0.,  s],
                            [0., 1., 0.],
                            [-s, 0.,  c]])
        ramp_center = np.array([0., 0., RAMP_POS_Z])
        local_pos   = Ry_pos @ (pos - ramp_center)
        on_ramp = (
            abs(local_pos[0]) < RAMP_LEN_HALF * 1.05 and
            abs(local_pos[1]) < RAMP_WID_HALF * 1.05
        )

        return {
            'pos':                pos,
            'vel':                vel,
            'angvel':             angvel,
            'euler_deg':          euler_deg,
            'theta_acc':          np.degrees(self._theta_acc),
            'theta_expected_deg': theta_expected_deg,
            'roll_ratio':         roll_ratio,
            'v_down':             v_down,
            'v_roll':             v_roll,
            'slip':               slip,
            'slip_ratio':         slip_ratio,
            'disp_acc':           self._disp_acc,
            'time':               now,
            'on_ramp':            on_ramp,
        }

    # ── viewer 상태 ──────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        if self._viewer is None:
            return True
        return self._viewer.is_running()

    # ── close ────────────────────────────────────────────────────
    def close(self) -> None:
        if self._viewer and self._viewer.is_running():
            self._viewer.close()


# ================================================================
#  물리 정보 출력
# ================================================================

def print_physics_info(model: mujoco.MjModel) -> None:
    """시뮬레이션 물리 파라미터 요약."""
    print("\n" + "-" * 56)
    print("  물리 모델 정보")
    print("-" * 56)

    print("  [Bodies]")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "(world)"
        mass = model.body_mass[i]
        if mass > 0:
            print(f"    {i:2d}  {name:<24s}  mass={mass:.5f} kg")

    print("\n  [Geoms (condim & friction)]")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "?"
        fric = model.geom_friction[i]
        cond = model.geom_condim[i]
        print(f"    {i:2d}  {name:<24s}  condim={cond}  "
              f"fric=[slide={fric[0]:.1f}, spin={fric[1]:.2f}, roll={fric[2]:.4f}]")

    print("-" * 56)


# ================================================================
#  이론 물리 계산
# ================================================================

def print_theoretical_physics() -> None:
    """순수 구름 이론값 출력."""
    g     = 9.81
    theta = SLOPE_RAD

    # 속이 찬 구 (I = 2/5 * m * r²)
    a_solid_sphere   = (5/7)  * g * np.sin(theta)
    mu_min_sphere    = (2/7)  * np.tan(theta)

    # 속이 빈 구껍질 (I = 2/3 * m * r²)
    a_hollow_sphere  = (3/5)  * g * np.sin(theta)
    mu_min_hollow    = (2/5)  * np.tan(theta)

    # 원통 (I = 1/2 * m * r²)
    a_cylinder       = (2/3)  * g * np.sin(theta)
    mu_min_cylinder  = (1/3)  * np.tan(theta)

    print("\n" + "=" * 56)
    print(f"  이론 물리 (경사각 {SLOPE_DEG}도)")
    print("=" * 56)
    print(f"  g·sin({SLOPE_DEG}°) = {g*np.sin(theta):.5f} m/s²  (중력 경사 성분)")
    print(f"\n  물체별 순수 구름 가속도:")
    print(f"    속이 찬 구   (I=2/5mr²): a = {a_solid_sphere:.5f} m/s²  "
          f"μ_min={mu_min_sphere:.4f}")
    print(f"    속이 빈 구껍질(I=2/3mr²): a = {a_hollow_sphere:.5f} m/s²  "
          f"μ_min={mu_min_hollow:.4f}")
    print(f"    원통         (I=1/2mr²): a = {a_cylinder:.5f} m/s²  "
          f"μ_min={mu_min_cylinder:.4f}")
    print(f"\n  설정된 FRIC_SLIDE={FRIC_SLIDE:.1f}  (min 요구치 대비 {FRIC_SLIDE/mu_min_sphere:.0f}배 이상)")
    print(f"  -> 미끄럼 없는 순수 구름 조건 충족")
    print("=" * 56)


# ================================================================
#  메인 시뮬레이션
# ================================================================

def run_simulation() -> None:
    """
    5도 경사면 구름 시뮬레이션 실행.

    - 공을 경사면 상단에 정지 상태로 놓음
    - 중력에 의해 하강 방향으로 굴러내려감
    - 미끄럼량(slip) 실시간 모니터링
    - MuJoCo viewer 로 시각화
    """
    print("=" * 56)
    print("  5도 경사면 구름 시뮬레이션  (Rolling without Slipping)")
    print("=" * 56)
    print(f"  물체 타입    : {OBJECT_TYPE}")
    print(f"  경사각       : {SLOPE_DEG}° ({SLOPE_RAD:.5f} rad)")
    print(f"  경사면 크기  : {RAMP_LEN_HALF*2:.1f} m × {RAMP_WID_HALF*2:.1f} m")
    print(f"  미끄럼마찰   : {FRIC_SLIDE}  (구름 보장 기준 ≈ 0.063 이상)")
    print(f"  구름마찰     : {FRIC_ROLL}")
    print(f"  시뮬 시간    : {SIM_DURATION} s")
    print(f"  재생 배속    : {SIM_SPEED:.0f}x  (SIM_SPEED 로 조정)")
    print("=" * 56)

    print_theoretical_physics()

    env = RollingInclinedEnv(object_type=OBJECT_TYPE, render=True)
    print_physics_info(env.model)

    obs = env.reset()
    print(f"\n  초기 공 위치: {obs['pos']}")
    print(f"  뷰어 창이 열리면 마우스로 시점 조절 가능 (q 또는 창 닫기 = 종료)\n")

    # ── 배속 설정 ────────────────────────────────────────────
    TARGET_FPS = 60
    if SIM_SPEED > 0:
        N_SUBSTEP = max(1, int(SIM_SPEED / (TIMESTEP * TARGET_FPS)))
    else:
        N_SUBSTEP = 50
    FRAME_TIME  = (TIMESTEP * N_SUBSTEP) / max(SIM_SPEED, 1e-9)
    PRINT_EVERY = max(1, int(2.0 / (TIMESTEP * N_SUBSTEP)))  # 2 s 마다 출력
    step        = 0
    off_ramp    = False

    print(f"  [배속 {SIM_SPEED:.0f}x]  N_SUBSTEP={N_SUBSTEP} "
          f"({TIMESTEP*N_SUBSTEP*1000:.1f} ms/frame)\n")

    # ── 로그 헤더 ────────────────────────────────────────────
    # 【회전각 해석】
    #   euler pitch_y : 현재 쿼터니언 기준 Y축 오일러 회전각 (도)
    #   theta_y(acc)  : angvel 적분 누적 Y축 회전각 (도)  ← 구름량 직접 지표
    #   theta_expect  : 이동거리 기준 기대 회전각 (도)    ← 순수 구름이면 theta_y 와 일치
    #   roll_ratio    : theta_y(acc) / theta_expect       ← 1.0=순수 구름, 0=완전 미끄럼
    #   slip_ratio    : 순간 미끄럼 비율                  ← 0=구름, 1=미끄럼
    print(f"  {'시간':>6}  {'위치X':>7}  {'pitch_y':>8}  "
          f"{'θ_acc_Y':>9}  {'θ_expect':>9}  "
          f"{'roll_r':>7}  {'slip_r':>7}  {'slip(m/s)':>9}  상태")
    print(f"  {'(s)':>6}  {'(m)':>7}  {'(deg)':>8}  "
          f"{'(deg)':>9}  {'(deg)':>9}  "
          f"{'(0~1)':>7}  {'(0~1)':>7}  {'':>9}")
    print(f"  {'-'*85}")

    while env.is_running:
        t_frame_start = time.perf_counter()

        obs = env.step(n_substep=N_SUBSTEP)
        t   = obs['time']

        # ── 콘솔 로그 ────────────────────────────────────────
        if step % PRINT_EVERY == 0:
            on_str = "경사면" if obs['on_ramp'] else "탈출  "
            rr = obs['roll_ratio']
            sr = obs['slip_ratio']
            # roll_ratio 판정
            if   rr > 0.90:  roll_judge = "구름OK"
            elif rr > 0.50:  roll_judge = "부분구름"
            else:             roll_judge = "<<미끄럼>>"

            print(
                f"  {t:6.1f}  "
                f"{obs['pos'][0]:+7.3f}  "
                f"{obs['euler_deg'][1]:+8.2f}  "
                f"{obs['theta_acc'][1]:+9.2f}  "
                f"{obs['theta_expected_deg']:+9.2f}  "
                f"{rr:7.3f}  "
                f"{sr:7.3f}  "
                f"{obs['slip']:9.5f}  "
                f"{on_str} {roll_judge}"
            )

        # 경사면 탈출 감지
        if not obs['on_ramp'] and not off_ramp:
            off_ramp = True
            print(f"\n  [!] 공이 경사면 탈출 (t={t:.1f}s, 이동={obs['disp_acc']:.3f}m)\n")

        if t >= SIM_DURATION:
            print(f"\n  시뮬레이션 완료 (t={t:.1f}s)")
            break

        step += 1

        if SIM_SPEED > 0:
            elapsed   = time.perf_counter() - t_frame_start
            remaining = FRAME_TIME - elapsed
            if remaining > 0.0005:
                time.sleep(remaining)

    env.close()
    print("\n  종료.")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == '__main__':
    run_simulation()
