"""
Steps 2 & 3: MuJoCo Ball-and-Plate Environment + Physical Models
=================================================================

Step 2: 2축 Hinge(Pitch / Roll)로 제어되는 평판 환경
Step 3: Sphere / Trajectoid 복합 질량체 물리 모델

실행:
    python step2_3_mujoco_env.py

필수 라이브러리 설치:
    pip install mujoco simple-pid

설계 개요:
    World
     └─ plate_outer  (body at PLATE_H)
           ├─ joint: roll_y   (Y축, 좌우 기울기)
           └─ plate_inner  (body)
                 ├─ joint: pitch_x  (X축, 앞뒤 기울기)
                 ├─ geom: plate     (충돌·시각 박스, 높은 마찰)
                 └─ site: path_???  (경로 시각화 마커, N개)

    Ball body  (freejoint — 6 DOF)
     ├─ [sphere mode]    geom: sphere  (단순 구)
     └─ [trajectoid mode] geom: shell  (trajectoid mesh)
                          geom: core   (볼베어링, 충돌 없음, 질량만)

제어:
    data.ctrl[0] = target_roll_y   (rad)   → Y축 회전 목표각
    data.ctrl[1] = target_pitch_x  (rad)   → X축 회전 목표각
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


# ===================================================== ===========
#  SETTINGS
# ================================================================
OBJECT_TYPE      = 'trajectoid'       # 'sphere' | 'trajectoid'
TRAJECTOID_MESH  = 'output/trajectoid.obj'

# 평판 치수 (m)
PLATE_HALF       = 0.25    # 반너비 (0.25 → 500 mm × 500 mm 정사각형)
PLATE_THICK      = 0.005   # 두께
PLATE_H          = 0.5     # 평판 중심 World-Z 높이
PLATE_MASS       = 2.0     # 평판 질량 (kg)
PLATE_MAX_TILT   = 0.35    # 최대 기울기 (rad ≈ 20°)

# 객체 치수 (실물 기준, m)
BALL_R           = 0.015875   # outer sphere 반지름 (15.875 mm)
CORE_R           = 0.0127     # 볼베어링 반지름 (12.7 mm = 0.5 inch)
MESH_SCALE       = CORE_R     # 알고리즘 1단위 = CORE_R [m]

# 밀도 (kg/m³)
SHELL_DENSITY    = 1240.0     # PLA 플라스틱 외피
BEARING_DENSITY  = 7874.0     # 강철 볼베어링

# 마찰 파라미터 (미끄럼 방지 핵심)
FRIC_SLIDE       = 3.0        # 접선 미끄럼 마찰계수
FRIC_SPIN        = 0.1        # 수직축 스핀 마찰계수
FRIC_ROLL        = 0.02       # 구름(rolling) 마찰계수

# CoACD 파라미터 (trajectoid 충돌 분해)
COACD_THRESHOLD  = 0.05       # 볼록 근사 허용 오차 (낮을수록 조각↑, 정밀↑)
COACD_MAX_CONVEX = 32         # 최대 볼록 조각 수

# Solver
TIMESTEP         = 0.002      # 시뮬레이션 time step (s)
SOLVER_ITER      = 50         # 최대 반복 횟수

# 경로 시각화 마커 수
PATH_MARKER_N    = 80

# 액추에이터 PD 게인
ACT_KP           = 400        # 위치 게인
ACT_KV           = 40         # 속도(댐핑) 게인


# ================================================================
#  MJCF XML 생성 헬퍼
# ================================================================

def _fric(s=FRIC_SLIDE, sp=FRIC_SPIN, r=FRIC_ROLL) -> str:
    return f"{s} {sp} {r}"


# ================================================================
#  복합 질량·관성 계산 (PLA 외피 + 강철 볼베어링)
# ================================================================
def _compute_inertial() -> tuple[float, float]:
    """
    트레젝토이드 전체 질량(kg)과 구 관성 모멘트(kg·m²) 계산.

    모델: 속이 빈 PLA 외피 (홈으로 30% 부피 감소) + 강철 볼베어링 내부 구

    Returns (mass_kg, I_kgm2)
    """
    V_outer = 4/3 * np.pi * BALL_R**3
    V_inner = 4/3 * np.pi * CORE_R**3
    V_shell = (V_outer - V_inner) * 0.70     # 홈으로 30% 감소
    m_shell = V_shell * SHELL_DENSITY
    R_avg   = (BALL_R + CORE_R) / 2
    I_shell = 2/3 * m_shell * R_avg**2

    m_core  = V_inner * BEARING_DENSITY
    I_core  = 2/5 * m_core * CORE_R**2

    return (m_shell + m_core), (I_shell + I_core)


# ================================================================
#  CoACD 볼록 분해 (trajectoid 충돌용)
# ================================================================
def _coacd_geoms(mesh_bytes: bytes) -> tuple[str, dict]:
    """
    .obj 바이트 → CoACD 분해 → (geoms_xml, assets_dict)

    시각 메쉬 (contype=0) + N개 볼록 충돌 조각 (condim=6).
    모든 메쉬는 MESH_SCALE 적용 후 미터 단위로 저장됩니다.
    """
    try:
        import trimesh as _trimesh
    except ImportError:
        raise ImportError("trimesh 가 없습니다.  pip install trimesh")
    try:
        import coacd
    except ImportError:
        raise ImportError("coacd 가 없습니다.  pip install coacd")

    print("  [CoACD] 메쉬 로딩 및 스케일 적용 중 …")
    mesh = _trimesh.load(io.BytesIO(mesh_bytes), file_type='obj', force='mesh')
    mesh.apply_scale(MESH_SCALE)                # 알고리즘 단위 → 미터
    _trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        print("  [CoACD] 경고: 메쉬가 watertight 하지 않음 — 결과가 부정확할 수 있습니다")

    print(f"  [CoACD] 볼록 분해 시작 … "
          f"(threshold={COACD_THRESHOLD}, max_parts={COACD_MAX_CONVEX})")
    m     = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        m,
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_CONVEX,
    )
    print(f"  [CoACD] {len(parts)} 개 볼록 조각 생성")

    assets: dict[str, bytes] = {}
    geom_lines: list[str]    = []

    # ① 시각 메쉬 (충돌 없음) — 스케일된 미터 단위로 저장
    bio_vis = io.BytesIO()
    mesh.export(bio_vis, file_type='obj')
    assets['traj_visual.obj'] = bio_vis.getvalue()
    geom_lines.append(
        '<geom name="visual" type="mesh" mesh="traj_visual" '
        'contype="0" conaffinity="0" '
        'rgba="0.85 0.35 0.18 0.90"/>'
    )

    # ② 볼록 충돌 조각들
    for i, (verts, faces) in enumerate(parts):
        key  = f'traj_cvx_{i:03d}.obj'
        part = _trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
        bio  = io.BytesIO()
        part.export(bio, file_type='obj')
        assets[key] = bio.getvalue()

        rgba = "0.45 0.75 1.0 0.12" if i == 0 else "0 0 0 0"
        geom_lines.append(
            f'<geom name="cvx_{i:03d}" type="mesh" mesh="traj_cvx_{i:03d}" '
            f'condim="6" friction="{_fric()}" '
            f'solref="0.005 1" solimp="0.99 0.999 0.001 0.5 2" '
            f'rgba="{rgba}"/>'
        )

    return "\n          ".join(geom_lines), assets


def _path_sites_xml(n: int) -> str:
    """평판 로컬 프레임 안에 N개의 경로 시각화 site 생성."""
    z = PLATE_THICK / 2 + 0.0008   # 평판 표면 약간 위
    lines = []
    for i in range(n):
        lines.append(
            f'        <site name="path_{i:04d}" pos="0 0 {z:.5f}" '
            f'size="0.0015" type="sphere" rgba="0.15 0.95 0.35 0.85"/>'
        )
    return "\n".join(lines)


def _sphere_body_xml(start_z: float) -> str:
    """단순 구 body XML 블록."""
    return textwrap.dedent(f"""\
        <!-- ── Sphere (단순 구) ── -->
        <body name="ball" pos="0 0 {start_z:.6f}">
          <freejoint name="ball_joint"/>
          <geom name="ball_geom"
                type="sphere"
                size="{BALL_R:.6f}"
                density="2000"
                friction="{_fric()}"
                condim="6"
                solref="0.005 1"
                solimp="0.99 0.999 0.001 0.5 2"
                rgba="0.85 0.25 0.20 1.0"/>
        </body>""")


def _trajectoid_body_xml(start_z: float) -> str:
    """
    복합 질량체 Trajectoid body XML 블록.

    구조:
      - shell geom : 충돌·시각 메쉬  (PLA 밀도, condim=6)
      - core  geom : 볼베어링 구      (강철 밀도, 충돌 없음 → contype/conaffinity=0)
    """
    scale_str = f"{MESH_SCALE:.6f} {MESH_SCALE:.6f} {MESH_SCALE:.6f}"
    return textwrap.dedent(f"""\
        <!-- ── Trajectoid (복합 질량체) ── -->
        <body name="ball" pos="0 0 {start_z:.6f}">
          <freejoint name="ball_joint"/>

          <!-- 외피: 충돌·시각 메쉬 -->
          <geom name="shell"
                type="mesh"
                mesh="traj_shell"
                density="{SHELL_DENSITY:.1f}"
                friction="{_fric()}"
                condim="6"
                solref="0.005 1"
                solimp="0.99 0.999 0.001 0.5 2"
                rgba="0.80 0.35 0.20 0.85"/>

          <!-- 내부 볼베어링: 질량·관성 기여 (충돌 없음) -->
          <geom name="core_bearing"
                type="sphere"
                size="{CORE_R:.6f}"
                density="{BEARING_DENSITY:.1f}"
                contype="0"
                conaffinity="0"
                rgba="0.70 0.70 0.75 0.50"/>
        </body>""")


def _trajectoid_body_xml_v2(
    start_z: float,
    geoms_xml: str,
    mass_kg: float,
    I_kgm2: float,
) -> str:
    """CoACD 분해 메쉬를 사용하는 Trajectoid body XML (v2)."""
    I_str = f"{I_kgm2:.3e}"
    return textwrap.dedent(f"""\
        <!-- ── Trajectoid v2 (CoACD 충돌 분해) ── -->
        <body name="ball" pos="0 0 {start_z:.6f}">
          <freejoint name="ball_joint"/>
          <inertial pos="0 0 0" mass="{mass_kg:.6f}"
                    diaginertia="{I_str} {I_str} {I_str}"/>
          {geoms_xml}
        </body>""")


def build_mjcf(object_type: str, path_2d: np.ndarray | None = None) -> tuple[str, dict]:
    """
    환경 MJCF XML 문자열을 생성합니다.

    Parameters
    ----------
    object_type : 'sphere' | 'trajectoid'
    path_2d     : (N, 2) array — 평판 로컬 좌표 경로 (시각화용, 선택)

    Returns
    -------
    xml    : str          — MJCF XML 문자열
    assets : dict[str, bytes]  — 메쉬 바이너리 {파일명: 바이트}
    """
    # ── 평판 top face z, 공 시작 z ──────────────────────────
    plate_top_z  = PLATE_H + PLATE_THICK / 2
    ball_start_z = plate_top_z + BALL_R + 0.0005   # 약간 gap

    # ── 객체별 블록 ──────────────────────────────────────────
    assets: dict[str, bytes] = {}

    if object_type == 'sphere':
        asset_block  = ""
        object_block = _sphere_body_xml(ball_start_z)

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
        mass_kg, I_kgm2  = _compute_inertial()

        asset_lines = []
        for fname in assets:
            name = fname.replace('.obj', '').replace('.', '_')
            asset_lines.append(f'    <mesh name="{name}" file="{fname}"/>')
        asset_block = "  <asset>\n" + "\n".join(asset_lines) + "\n  </asset>"
        object_block = _trajectoid_body_xml_v2(ball_start_z, geoms_xml, mass_kg, I_kgm2)

    else:
        raise ValueError(
            f"OBJECT_TYPE 은 'sphere' 또는 'trajectoid' 이어야 합니다: '{object_type}'"
        )

    # ── 경로 site 위치 주입 ──────────────────────────────────
    path_sites_xml = _path_sites_xml(PATH_MARKER_N)

    # ── MJCF 본문 ────────────────────────────────────────────
    fric  = _fric()
    xml = textwrap.dedent(f"""\
        <mujoco model="ball_plate_{object_type}">

          <!-- 솔버 & 물리 설정 -->
          <option timestep="{TIMESTEP}"
                  gravity="0 0 -9.81"
                  integrator="implicitfast"
                  iterations="{SOLVER_ITER}"
                  tolerance="1e-10"/>

          <compiler autolimits="true"/>

          {asset_block}

          <!-- 전역 기본값: 모든 geom에 condim=6 적용 -->
          <default>
            <geom condim="6" friction="{fric}"
                  solref="0.005 1" solimp="0.99 0.999 0.001 0.5 2"/>
            <joint damping="0.5"/>
          </default>

          <worldbody>

            <!-- 바닥 참조면 (시각화 전용, 충돌 없음) -->
            <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0"
                  rgba="0.28 0.28 0.28 0.6" contype="0" conaffinity="0"/>

            <!-- ┌─────────────────────────────────────────┐ -->
            <!-- │  평판 어셈블리: 2축 짐벌                 │ -->
            <!-- │  outer  joint: roll_y  (Y축, 좌우 틸트)  │ -->
            <!-- │  inner  joint: pitch_x (X축, 앞뒤 틸트)  │ -->
            <!-- └─────────────────────────────────────────┘ -->
            <body name="plate_outer" pos="0 0 {PLATE_H:.4f}">
              <!-- plate_outer 자체 질량 (MuJoCo: 관절 있는 body는 질량 > 0 필수) -->
              <inertial pos="0 0 0" mass="0.05"
                        diaginertia="1e-4 1e-4 1e-4"/>
              <joint name="roll_y"
                     type="hinge"
                     axis="0 1 0"
                     range="-{PLATE_MAX_TILT:.4f} {PLATE_MAX_TILT:.4f}"
                     damping="2.0"
                     armature="0.01"/>

              <body name="plate_inner">
                <joint name="pitch_x"
                       type="hinge"
                       axis="1 0 0"
                       range="-{PLATE_MAX_TILT:.4f} {PLATE_MAX_TILT:.4f}"
                       damping="2.0"
                       armature="0.01"/>

                <!-- 평판 geom -->
                <geom name="plate"
                      type="box"
                      size="{PLATE_HALF:.4f} {PLATE_HALF:.4f} {PLATE_THICK/2:.5f}"
                      friction="{fric}"
                      condim="6"
                      mass="{PLATE_MASS:.2f}"
                      rgba="0.42 0.52 0.78 1.0"/>

                <!-- 경로 시각화 markers (평판 로컬 프레임) -->
            {path_sites_xml}

              </body>
            </body>

            <!-- ┌─────────────────────────────┐ -->
            <!-- │  Rolling object body        │ -->
            <!-- └─────────────────────────────┘ -->
            {object_block}

          </worldbody>

          <!-- ┌─────────────────────────────────────────────┐ -->
          <!-- │  Actuators: PD 위치 제어                     │ -->
          <!-- │  ctrl[0] = target roll_y  (rad)              │ -->
          <!-- │  ctrl[1] = target pitch_x (rad)              │ -->
          <!-- └─────────────────────────────────────────────┘ -->
          <actuator>
            <position name="act_roll_y"
                       joint="roll_y"
                       kp="{ACT_KP}"
                       kv="{ACT_KV}"
                       ctrlrange="-{PLATE_MAX_TILT:.4f} {PLATE_MAX_TILT:.4f}"/>
            <position name="act_pitch_x"
                       joint="pitch_x"
                       kp="{ACT_KP}"
                       kv="{ACT_KV}"
                       ctrlrange="-{PLATE_MAX_TILT:.4f} {PLATE_MAX_TILT:.4f}"/>
          </actuator>

          <!-- 센서: 공 위치·속도 모니터링용 -->
          <sensor>
            <framepos   name="ball_pos"  objtype="body" objname="ball"/>
            <framexaxis name="ball_xax"  objtype="body" objname="ball"/>
            <framelinvel name="ball_vel" objtype="body" objname="ball"/>
          </sensor>

        </mujoco>""")

    return xml, assets


# ================================================================
#  BallPlateEnv 클래스
# ================================================================

class BallPlateEnv:
    """
    2축 Hinge 평판 위에서 구 / 트레젝토이드가 구르는 MuJoCo 환경.

    사용 예:
        env = BallPlateEnv('sphere')
        obs = env.reset()
        for _ in range(1000):
            obs = env.step(roll_target, pitch_target)
        env.close()
    """

    # ctrl 인덱스 상수
    CTRL_ROLL_Y  = 0
    CTRL_PITCH_X = 1

    def __init__(
        self,
        object_type: str = OBJECT_TYPE,
        path_2d: np.ndarray | None = None,
        render: bool = True,
    ):
        """
        Parameters
        ----------
        object_type : 'sphere' | 'trajectoid'
        path_2d     : (N, 2) float array — 경로 (평판 로컬 좌표, m)
        render      : MuJoCo viewer 표시 여부
        """
        self.object_type = object_type
        self.render_flag = render

        print(f"[BallPlateEnv] MJCF 생성 중 … (object_type={object_type})")
        xml, assets = build_mjcf(object_type, path_2d)

        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data  = mujoco.MjData(self.model)

        # joint qpos 주소 캐시 (MuJoCo 3.x: qposadr가 배열 → int 변환 필수)
        self._jadr_roll  = int(self.model.joint('roll_y').qposadr)
        self._jadr_pitch = int(self.model.joint('pitch_x').qposadr)

        # freejoint qpos 주소 (공)
        self._jadr_ball  = int(self.model.joint('ball_joint').qposadr)

        # 공 body id
        self._ball_bid   = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball'
        )

        # 경로 site id 목록 (있으면 업데이트)
        self._site_ids   = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'path_{i:04d}')
            for i in range(PATH_MARKER_N)
        ]

        # 경로 마커 설정
        if path_2d is not None:
            self.set_path(path_2d)

        # viewer
        self._viewer = None
        if render:
            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self._viewer.cam.azimuth   = 45
            self._viewer.cam.elevation = -25
            self._viewer.cam.distance  = 1.5

        # 초기 상태 저장
        mujoco.mj_resetData(self.model, self.data)
        self._place_ball_at_center()
        mujoco.mj_forward(self.model, self.data)

        print(f"[BallPlateEnv] 초기화 완료")
        print(f"  nq={self.model.nq}  nv={self.model.nv}  "
              f"nu={self.model.nu}  nsensor={self.model.nsensor}")

    # ── 경로 마커 설정 ───────────────────────────────────────
    def set_path(self, path_2d: np.ndarray) -> None:
        """
        경로 시각화 마커를 평판 위에 업데이트.

        Parameters
        ----------
        path_2d : (N, 2) — 평판 로컬 (x, y) 좌표 [m]
        """
        n = len(self._site_ids)
        # 경로를 n개 점으로 샘플링
        idx = np.linspace(0, len(path_2d) - 1, n).astype(int)
        sampled = path_2d[idx]

        z_on_plate = PLATE_THICK / 2 + 0.0008
        for i, sid in enumerate(self._site_ids):
            if sid < 0:
                continue
            x, y = sampled[i]
            self.model.site_pos[sid] = [x, y, z_on_plate]

    # ── 공 초기 위치 ─────────────────────────────────────────
    def _place_ball_at_center(self) -> None:
        """공을 평판 중심 위에 배치."""
        plate_top_z  = PLATE_H + PLATE_THICK / 2
        start_z      = plate_top_z + BALL_R + 0.0005
        qadr = self._jadr_ball
        self.data.qpos[qadr:qadr + 3] = [0.0, 0.0, start_z]
        self.data.qpos[qadr + 3]      = 1.0   # w
        self.data.qpos[qadr + 4:qadr + 7] = [0.0, 0.0, 0.0]   # x y z

    # ── reset ────────────────────────────────────────────────
    def reset(self) -> dict:
        """환경 초기화. 공을 평판 중심 위에 놓고 obs 반환."""
        mujoco.mj_resetData(self.model, self.data)
        self._place_ball_at_center()
        mujoco.mj_forward(self.model, self.data)
        if self._viewer and self._viewer.is_running():
            self._viewer.sync()
        return self.get_obs()

    # ── step ─────────────────────────────────────────────────
    def step(
        self,
        roll_y:  float,
        pitch_x: float,
        n_substep: int = 1,
    ) -> dict:
        """
        평판 목표 각도를 설정하고 시뮬레이션 1스텝(n_substep) 진행.

        Parameters
        ----------
        roll_y   : 목표 Y축 회전각 (rad)  — 양수 = 공이 +X 방향으로 구름
        pitch_x  : 목표 X축 회전각 (rad)  — 양수 = 공이 -Y 방향으로 구름
        n_substep: 한 번의 step() 호출에서 적분 횟수 (기본 1)
        """
        roll_y  = float(np.clip(roll_y,  -PLATE_MAX_TILT, PLATE_MAX_TILT))
        pitch_x = float(np.clip(pitch_x, -PLATE_MAX_TILT, PLATE_MAX_TILT))

        self.data.ctrl[self.CTRL_ROLL_Y]  = roll_y
        self.data.ctrl[self.CTRL_PITCH_X] = pitch_x

        for _ in range(n_substep):
            mujoco.mj_step(self.model, self.data)

        if self._viewer and self._viewer.is_running():
            self._viewer.sync()

        return self.get_obs()

    # ── get_obs ──────────────────────────────────────────────
    def get_obs(self) -> dict:
        """
        현재 관측값 딕셔너리 반환.

        Keys
        ----
        ball_pos   : (3,) World 좌표 [m]
        ball_pos_plate : (2,) 평판 로컬 (x, y) [m]
        ball_vel   : (3,) World 선속도 [m/s]
        plate_roll : float — roll_y 현재각 [rad]
        plate_pitch: float — pitch_x 현재각 [rad]
        time       : float — 시뮬레이션 시간 [s]
        """
        ball_pos = self.data.xpos[self._ball_bid].copy()
        ball_vel = self.data.sensor('ball_vel').data.copy()

        roll_cur  = float(self.data.qpos[self._jadr_roll])
        pitch_cur = float(self.data.qpos[self._jadr_pitch])

        # 평판 로컬 좌표: 평판 body의 역회전으로 World 좌표 변환
        plate_body_xmat = self.data.xmat[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'plate_inner')
        ].reshape(3, 3)
        plate_origin = self.data.xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'plate_inner')
        ]
        local = plate_body_xmat.T @ (ball_pos - plate_origin)
        ball_pos_plate  = local[:2]
        ball_vel_plate  = (plate_body_xmat.T @ ball_vel)[:2]   # 평판 로컬 2D 속도

        return {
            'ball_pos':        ball_pos,
            'ball_pos_plate':  ball_pos_plate,
            'ball_vel':        ball_vel,
            'ball_vel_plate':  ball_vel_plate,
            'plate_roll':      roll_cur,
            'plate_pitch':     pitch_cur,
            'time':            self.data.time,
        }

    # ── viewer 상태 ──────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        if self._viewer is None:
            return True
        return self._viewer.is_running()

    # ── close ────────────────────────────────────────────────
    def close(self) -> None:
        if self._viewer and self._viewer.is_running():
            self._viewer.close()


# ================================================================
#  물리 모델 정보 출력 (Step 3 확인용)
# ================================================================
def print_physics_model_info(model: mujoco.MjModel) -> None:
    """생성된 MuJoCo 모델의 물리 파라미터를 요약 출력합니다."""
    print("\n" + "─" * 52)
    print("  물리 모델 정보")
    print("─" * 52)

    # Body 목록
    print("  [Bodies]")
    for i in range(model.nbody):
        name  = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "(world)"
        mass  = model.body_mass[i]
        ixx   = model.body_inertia[i][0]
        print(f"    {i:2d}  {name:<20s}  mass={mass:.5f} kg  Ixx={ixx:.2e}")

    # Geom 목록
    print("\n  [Geoms]")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "?"
        fric = model.geom_friction[i]
        cond = model.geom_condim[i]
        print(f"    {i:2d}  {name:<20s}  condim={cond}  "
              f"fric=[{fric[0]:.1f},{fric[1]:.2f},{fric[2]:.3f}]")

    # Joint 목록
    print("\n  [Joints]")
    for i in range(model.njnt):
        name  = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "?"
        jtype = model.jnt_type[i]
        rng   = model.jnt_range[i]
        jtype_str = ['free','ball','slide','hinge'][jtype]
        print(f"    {i:2d}  {name:<20s}  type={jtype_str:<6s}  "
              f"range=[{np.degrees(rng[0]):.1f}°, {np.degrees(rng[1]):.1f}°]")

    print("─" * 52)


# ================================================================
#  DEMO
# ================================================================
def run_demo() -> None:
    """
    데모: 평판을 사인파로 틸트하면서 공이 구르는 모습 시각화.
    - sphere 모드: 틸트 방향을 따라 직선 구름
    - trajectoid 모드: 경로를 따라 구름 (Step 4 제어 필요)

    뷰어 창이 열리면 마우스/키보드로 시점 조절 가능.
    'q' 또는 창 닫기로 종료.
    """
    print("=" * 55)
    print("  Ball-and-Plate Demo  (Step 2 & 3)")
    print("=" * 55)
    print(f"  object_type : {OBJECT_TYPE}")
    print(f"  timestep    : {TIMESTEP} s")
    print(f"  plate size  : {PLATE_HALF*2*1000:.0f} mm × {PLATE_HALF*2*1000:.0f} mm")
    print(f"  ball radius : {BALL_R*1000:.3f} mm")
    print("=" * 55)

    # 샘플 경로 (sin파, 평판 로컬 좌표)
    t_path  = np.linspace(0, 2 * np.pi, 400)
    path_2d = np.stack([
        np.linspace(-0.18, 0.18, 400),
        0.08 * np.sin(3 * t_path),
    ], axis=1)

    env = BallPlateEnv(
        object_type=OBJECT_TYPE,
        path_2d=path_2d,
        render=True,
    )
    print_physics_model_info(env.model)

    obs = env.reset()
    print(f"\n  초기 공 위치 (World): {obs['ball_pos']}")
    print("  뷰어 창이 열리면 마우스로 시점 조절 가능 (q=종료)\n")

    step   = 0
    t_sim  = 0.0

    # ── 제어 루프 ────────────────────────────────────────────
    # Sphere demo: Y축 roll을 천천히 사인파 → 공이 좌우로 구름
    # Trajectoid demo: 틸트 방향이 천천히 회전 (실제 트레젝토이드 메커니즘)
    while env.is_running:
        if OBJECT_TYPE == 'sphere':
            # 사인파 틸트 — 공이 좌우로 굴러다님
            roll_target  = 0.15 * np.sin(2 * np.pi * t_sim / 4.0)
            pitch_target = 0.10 * np.sin(2 * np.pi * t_sim / 6.0 + 0.5)

        else:  # trajectoid
            # 틸트 방향이 일정 각속도로 회전 (트레젝토이드 핵심 메커니즘)
            tilt_mag = 0.08             # 일정 기울기 크기 (rad)
            omega    = 2 * np.pi / 12  # 1회전 주기 12 s
            phi      = omega * t_sim
            roll_target  = tilt_mag * np.cos(phi)
            pitch_target = tilt_mag * np.sin(phi)

        obs = env.step(roll_target, pitch_target, n_substep=4)

        # 상태 출력 (0.5s 마다)
        if step % 250 == 0:
            bp = obs['ball_pos_plate']
            print(
                f"  t={obs['time']:6.2f}s  "
                f"ball_plate=({bp[0]:+.3f}, {bp[1]:+.3f})  "
                f"roll={np.degrees(obs['plate_roll']):+.1f}°  "
                f"pitch={np.degrees(obs['plate_pitch']):+.1f}°"
            )

        # 공이 평판 밖으로 나가면 리셋
        if np.linalg.norm(obs['ball_pos_plate']) > PLATE_HALF * 0.9:
            print("  [!] 공이 평판 가장자리 근처 → reset")
            obs = env.reset()

        t_sim += TIMESTEP * 4
        step  += 1

        # 실시간 배속 조절 (시뮬 시간 ≈ 실제 시간)
        time.sleep(TIMESTEP * 0.5)

    env.close()
    print("\n  종료.")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == '__main__':
    run_demo()
