"""
test_tilt.py  (v3)
==================
Trajectoid 을 기울어진 평판 위에서 굴리는 테스트.
접촉점을 빨간 흔적으로 표시.

충돌 방식 (COLLISION_METHOD):
  'coacd'        - CoACD 볼록 분해 → .obj 자체가 직접 충돌  (권장)
                   pip install coacd
  'mathematical' - path_data.npy로부터 cutting box를 수학적으로 재계산
                   → Boolean 빼기 → CoACD  (가장 정확, 시간 소요)
                   pip install coacd manifold3d
  'proxy_sphere' - 구(sphere)를 충돌 대리로 사용  (빠름, 정확도 낮음)

왜 직접 mesh 충돌이 안 되는가:
  MuJoCo mesh geom = 내부적으로 convex hull(볼록 껍데기)만 사용
  trajectoid = 구에서 홈을 파낸 비볼록(non-convex) 형상
  → convex hull ≈ 원래 구  → 홈 정보 손실 → 슬라이딩

CoACD / Mathematical 방식:
  비볼록 메쉬를 N개의 볼록 조각으로 분해
  → 각 조각은 MuJoCo에서 정확히 충돌 가능
  → 합집합 ≈ 원래 형상
"""

import io, os, sys, time
import numpy as np

try:
    import mujoco, mujoco.viewer
except ImportError:
    sys.exit("pip install mujoco")

try:
    import trimesh
except ImportError:
    sys.exit("pip install trimesh")

# ================================================================
#  SETTINGS
# ================================================================
COLLISION_METHOD = 'coacd'         # 'coacd' | 'mathematical' | 'proxy_sphere'

MESH_FILE        = 'output/trajectoid.obj'
PATH_DATA_FILE   = 'trajectoids/examples/little-prince-2/folder_for_path/path_data.npy'

CORE_R           = 0.0127          # 볼베어링 반지름 12.7 mm  (= 1 algo unit)
OUTER_R          = 0.015875        # outer sphere 반지름 15.875 mm  (= 1.25 algo units)
MESH_SCALE       = CORE_R          # 알고리즘 1단위 → 0.0127 m

# CoACD 파라미터 (coacd / mathematical 모드)
COACD_THRESHOLD      = 0.05        # 볼록 근사 허용 오차 (낮을수록 조각↑, 정밀↑)
COACD_MAX_CONVEX     = 32          # 최대 볼록 조각 수

# Mathematical 모드 전용
MATH_CORE_RADIUS     = 1.0         # 알고리즘 core_radius (변경 금지)
MATH_CUT_SIZE        = 10          # 알고리즘 cut_size (변경 금지)
MATH_OUTER_RADIUS    = 1.25        # 알고리즘 outer_radius
MATH_SPHERE_SUBDIV   = 5           # icosphere 세분화 횟수

# 평판
TILT_DEG         = 20.0
TILT_AXIS        = 'x'             # 'x' | 'y' | 'both'
PLATE_HALF       = 0.30
PLATE_THICK      = 0.005
PLATE_Z          = 0.0

# 시뮬레이션
SIM_DURATION     = 300.0
TIMESTEP         = 0.001

# 흔적 마커
TRAIL_EVERY      = 8
TRAIL_MAX        = 1500


# ================================================================
#  질량·관성 해석 계산  (composite: PLA 외피 + 강철 볼베어링)
# ================================================================
def _compute_inertial() -> tuple[float, float]:
    """
    트레젝토이드 전체 질량(kg)과 구 관성 모멘트(kg·m²) 계산.

    모델: 속이 빈 구형 PLA 외피 + 내부 강철 볼베어링 고체 구

    Returns (mass_kg, I_kgm2)
    """
    # PLA 외피 (hollow sphere, 홈으로 약 30% 부피 감소)
    V_outer  = 4/3 * np.pi * OUTER_R**3
    V_inner  = 4/3 * np.pi * CORE_R**3
    V_shell  = (V_outer - V_inner) * 0.70      # 30% 홈에 의한 감소
    m_shell  = V_shell * 1240.0                # PLA 1240 kg/m³
    R_avg    = (OUTER_R + CORE_R) / 2
    I_shell  = 2/3 * m_shell * R_avg**2

    # 강철 볼베어링 (solid sphere)
    m_core   = V_inner * 7874.0                # 강철 7874 kg/m³
    I_core   = 2/5 * m_core * CORE_R**2

    return (m_shell + m_core), (I_shell + I_core)


# ================================================================
#  Method A: CoACD 볼록 분해  (.obj → 볼록 조각들)
# ================================================================
def _coacd_from_bytes(mesh_bytes: bytes) -> tuple[str, dict]:
    """
    .obj 바이트 → CoACD 분해 → (geoms_xml, assets_dict)

    assets_dict 에는 시각 메쉬 + 각 볼록 조각 OBJ 바이트가 들어있음.
    geoms_xml 은 ball body 안에 삽입할 <geom ...> 문자열.
    """
    try:
        import coacd
    except ImportError:
        raise ImportError(
            "coacd 가 설치되지 않았습니다.\n"
            "  pip install coacd\n"
            "또는 COLLISION_METHOD = 'proxy_sphere' 로 변경하세요."
        )

    print("  [CoACD] 메쉬 로딩 중 …")
    mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type='obj', force='mesh')
    mesh.apply_scale(MESH_SCALE)          # 알고리즘 단위 → 미터

    # 메쉬 수리 (CoACD 입력은 watertight 권장)
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        print("  [CoACD] 경고: 메쉬가 watertight 하지 않음 — 결과가 부정확할 수 있습니다")

    print(f"  [CoACD] 볼록 분해 시작 … "
          f"(threshold={COACD_THRESHOLD}, max_parts={COACD_MAX_CONVEX})")
    m      = coacd.Mesh(mesh.vertices, mesh.faces)
    parts  = coacd.run_coacd(
        m,
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_CONVEX,
    )
    print(f"  [CoACD] {len(parts)} 개 볼록 조각 생성")

    assets: dict[str, bytes] = {}
    geom_lines: list[str]    = []

    # ① 시각 메쉬 (no collision)
    assets['traj_visual.obj'] = mesh_bytes
    geom_lines.append(
        '<geom name="visual" type="mesh" mesh="traj_visual" '
        'contype="0" conaffinity="0" '
        'rgba="0.85 0.35 0.18 0.90"/>'
    )

    # ② 볼록 조각들 (collision only)
    for i, (verts, faces) in enumerate(parts):
        key = f'traj_cvx_{i:03d}.obj'
        part = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
        bio  = io.BytesIO()
        part.export(bio, file_type='obj')
        assets[key] = bio.getvalue()

        # 첫 조각만 반투명으로 표시, 나머지 완전 투명
        rgba = "1 0.75 0.45 0.85" if i == 0 else "0 0 0 0"
        geom_lines.append(
            f'<geom name="cvx_{i:03d}" type="mesh" mesh="traj_cvx_{i:03d}" '
            f'condim="6" friction="3.0 0.1 0.02" '
            f'solref="0.004 1" solimp="0.99 0.999 0.001 0.5 2" '
            f'rgba="{rgba}"/>'
        )

    return "\n      ".join(geom_lines), assets


# ================================================================
#  Method B: 수학적 재계산  (path_data → cutting boxes → Boolean → CoACD)
# ================================================================
def _rotation_step(point: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """구름 운동 한 스텝의 회전 행렬 (4×4 homogeneous)."""
    v     = prev - point
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return trimesh.transformations.identity_matrix()
    axis  = [v[1], -v[0], 0.0]
    return trimesh.transformations.rotation_matrix(-theta, axis, [0,0,0])


def _math_decompose() -> tuple[str, dict]:
    """
    path_data.npy → cutting boxes 재계산 → Boolean 빼기 → CoACD
    → (geoms_xml, assets_dict)
    """
    try:
        import coacd
    except ImportError:
        raise ImportError("pip install coacd manifold3d")

    if not os.path.exists(PATH_DATA_FILE):
        raise FileNotFoundError(
            f"경로 데이터 없음: {PATH_DATA_FILE}\n"
            "먼저 examples/little-prince-2 의 path_data.npy 를 생성하세요."
        )

    print("  [Mathematical] path_data 로드 중 …")
    from tqdm import tqdm
    path_data = np.load(PATH_DATA_FILE)
    print(f"  [Mathematical] 경로 점 수: {len(path_data)}")

    # ── Cutting boxes 생성 (step1 과 동일 로직) ───────────────
    r   = MATH_CORE_RADIUS
    cs  = MATH_CUT_SIZE
    box_size = cs * r
    base_box = trimesh.creation.box(
        extents=[box_size, box_size, box_size],
        transform=trimesh.transformations.translation_matrix([0, 0, -r - box_size/2]),
    )

    print("  [Mathematical] 회전 행렬 계산 중 …")
    rotations = [trimesh.transformations.identity_matrix()]
    for i in tqdm(range(1, len(path_data)), ncols=60):
        R = _rotation_step(path_data[i], path_data[i-1])
        rotations.append(
            trimesh.transformations.concatenate_matrices(rotations[-1], R)
        )

    print("  [Mathematical] Cutting boxes 생성 중 …")
    boxes = []
    for i in tqdm(range(len(path_data)), ncols=60):
        b = base_box.copy()
        b.apply_transform(rotations[i])
        boxes.append(b)

    # ── Boolean ───────────────────────────────────────────────
    print("  [Mathematical] Boolean union(boxes) …")
    boxes_union = trimesh.boolean.union(boxes, engine='manifold')

    print("  [Mathematical] Boolean difference(sphere − boxes) …")
    sphere = trimesh.creation.icosphere(MATH_SPHERE_SUBDIV, MATH_OUTER_RADIUS)
    traj   = trimesh.boolean.difference([sphere, boxes_union], engine='manifold')

    # 실제 크기로 스케일
    traj.apply_scale(MESH_SCALE)

    # ── CoACD ─────────────────────────────────────────────────
    print(f"  [Mathematical] CoACD 분해 …")
    m     = coacd.Mesh(traj.vertices, traj.faces)
    parts = coacd.run_coacd(m, threshold=COACD_THRESHOLD,
                             max_convex_hull=COACD_MAX_CONVEX)
    print(f"  [Mathematical] {len(parts)} 개 볼록 조각 생성")

    assets: dict[str, bytes] = {}
    geom_lines: list[str]    = []

    # 시각 메쉬
    bio_vis = io.BytesIO()
    traj.export(bio_vis, file_type='obj')
    assets['traj_visual.obj'] = bio_vis.getvalue()
    geom_lines.append(
        '<geom name="visual" type="mesh" mesh="traj_visual" '
        'contype="0" conaffinity="0" rgba="0.85 0.35 0.18 0.90"/>'
    )

    # 볼록 조각들
    for i, (verts, faces) in enumerate(parts):
        key  = f'traj_cvx_{i:03d}.obj'
        part = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
        bio  = io.BytesIO()
        part.export(bio, file_type='obj')
        assets[key] = bio.getvalue()
        rgba = "0.45 0.75 1.0 0.12" if i == 0 else "0 0 0 0"
        geom_lines.append(
            f'<geom name="cvx_{i:03d}" type="mesh" mesh="traj_cvx_{i:03d}" '
            f'condim="6" friction="3.0 0.1 0.02" '
            f'solref="0.004 1" solimp="0.99 0.999 0.001 0.5 2" '
            f'rgba="{rgba}"/>'
        )

    return "\n      ".join(geom_lines), assets


# ================================================================
#  Method C: 대리 구  (fallback)
# ================================================================
def _proxy_sphere(mesh_bytes: bytes) -> tuple[str, dict]:
    """구를 충돌 대리로 사용. 시각만 mesh."""
    assets = {'traj_visual.obj': mesh_bytes}
    geoms  = (
        f'<!-- 충돌: 구 대리 -->\n'
        f'      <geom name="coll_sphere" type="sphere" size="{OUTER_R:.6f}" '
        f'condim="6" friction="3.0 0.1 0.02" '
        f'solref="0.004 1" solimp="0.99 0.999 0.001 0.5 2" '
        f'rgba="0.45 0.75 1.0 0.12"/>\n'
        f'      <!-- 시각: mesh -->\n'
        f'      <geom name="visual" type="mesh" mesh="traj_visual" '
        f'contype="0" conaffinity="0" rgba="0.85 0.35 0.18 0.90"/>'
    )
    return geoms, assets


# ================================================================
#  MJCF 빌드
# ================================================================
def build_mjcf(geoms_xml: str, assets: dict,
               mass_kg: float, I_kgm2: float) -> tuple[str, dict]:
    """
    전체 MJCF XML 문자열 + assets dict 반환.

    <inertial> 로 질량/관성을 명시 → geom 의 density 는 무시됨 (geometry만 사용).
    assets 에 있는 각 파일명이 <mesh file="..."> 에서 참조됨.
    """
    tilt_rad = np.radians(TILT_DEG)
    if TILT_AXIS == 'x':
        euler = f"{TILT_DEG:.4f} 0 0"
    elif TILT_AXIS == 'y':
        euler = f"0 {TILT_DEG:.4f} 0"
    else:
        a = TILT_DEG / np.sqrt(2)
        euler = f"{a:.4f} {a:.4f} 0"

    ball_z = PLATE_Z + PLATE_THICK/2 + OUTER_R + 0.001

    # <asset> 내 mesh 선언들 (모두 scale="1 1 1" — 이미 미터 단위)
    asset_lines = []
    for fname in assets:
        name = fname.replace('.obj', '').replace('.', '_')
        asset_lines.append(f'    <mesh name="{name}" file="{fname}"/>')
    asset_block = "\n".join(asset_lines)

    xml = f"""<mujoco model="tilt_test">

  <option timestep="{TIMESTEP}" gravity="0 0 -9.81" integrator="implicitfast"/>

  <asset>
{asset_block}
  </asset>

  <default>
    <geom condim="6" friction="3.0 0.1 0.02"
          solref="0.004 1" solimp="0.99 0.999 0.001 0.5 2"/>
  </default>

  <worldbody>

    <geom type="plane" size="2 2 0.1" pos="0 0 -0.5"
          rgba="0.22 0.22 0.22 0.5" contype="0" conaffinity="0"/>

    <!-- 기울어진 평판 (정적, 각도 고정) -->
    <body name="plate" pos="0 0 {PLATE_Z:.4f}" euler="{euler}">
      <geom name="plate_geom" type="box"
            size="{PLATE_HALF:.4f} {PLATE_HALF:.4f} {PLATE_THICK/2:.5f}"
            rgba="0.40 0.55 0.80 1.0" mass="5.0"/>
    </body>

    <!-- Trajectoid (freejoint 6DOF) -->
    <body name="ball" pos="0 0 {ball_z:.6f}">
      <freejoint name="ball_joint"/>

      <!--
        <inertial> 를 명시하면 MuJoCo는 geom density를 무시하고
        이 값만 질량/관성에 사용.  (geom 은 geometry/충돌만 담당)
        mass  = {mass_kg*1000:.1f} g
        I_diag = {I_kgm2:.3e} kg·m²
      -->
      <inertial pos="0 0 0" mass="{mass_kg:.6f}"
                diaginertia="{I_kgm2:.3e} {I_kgm2:.3e} {I_kgm2:.3e}"/>

      {geoms_xml}

    </body>

  </worldbody>

</mujoco>"""

    return xml, assets


# ================================================================
#  접촉 흔적
# ================================================================
trail_pts: list[np.ndarray] = []

def get_contact_point(data, ball_id, plate_id) -> np.ndarray:
    ball_pos   = data.xpos[ball_id].copy()
    plate_norm = data.xmat[plate_id].reshape(3,3)[:, 2]
    return ball_pos - OUTER_R * plate_norm

def draw_trail(viewer) -> None:
    if not trail_pts:
        return
    max_g      = viewer.user_scn.maxgeom
    pts        = trail_pts[-max_g:]
    n          = len(pts)
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        for i, pt in enumerate(pts):
            age   = i / max(n-1, 1)
            r     = 0.25 + 0.75 * age
            alpha = 0.35 + 0.60 * age
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.0009, 0.0, 0.0]),
                pt.astype(np.float64),
                np.eye(3).flatten().astype(np.float64),
                np.array([r, 0.0, 0.0, alpha], dtype=np.float32),
            )
            viewer.user_scn.ngeom += 1


# ================================================================
#  MAIN
# ================================================================
def main():
    # ── 1. 메쉬 파일 로드 ────────────────────────────────────
    mesh_abs = os.path.abspath(MESH_FILE)
    if not os.path.exists(mesh_abs) and COLLISION_METHOD != 'mathematical':
        sys.exit(f"[오류] 메쉬 없음: {mesh_abs}\n먼저 step1_boolean_subtraction.py 실행")

    mesh_bytes = b''
    if os.path.exists(mesh_abs):
        with open(mesh_abs, 'rb') as f:
            mesh_bytes = f.read()

    # ── 2. 충돌 geom 구성 ────────────────────────────────────
    print("=" * 55)
    print(f"  Trajectoid 기울기 테스트  (method={COLLISION_METHOD})")
    print("=" * 55)
    print(f"  기울기   : {TILT_DEG}° ({TILT_AXIS}축)")
    print(f"  outer R  : {OUTER_R*1000:.3f} mm")
    print(f"  core  R  : {CORE_R*1000:.3f} mm")
    print(f"  scale    : {MESH_SCALE*1000:.3f} mm/unit")

    if COLLISION_METHOD == 'coacd':
        geoms_xml, assets = _coacd_from_bytes(mesh_bytes)
    elif COLLISION_METHOD == 'mathematical':
        geoms_xml, assets = _math_decompose()
    else:
        print("  [proxy_sphere] 구 대리 사용 (정밀도 낮음)")
        geoms_xml, assets = _proxy_sphere(mesh_bytes)

    # ── 3. 질량·관성 ─────────────────────────────────────────
    mass_kg, I_kgm2 = _compute_inertial()
    print(f"\n  총 질량  : {mass_kg*1000:.1f} g")
    print(f"  관성 모멘트: {I_kgm2:.3e} kg·m²")

    # ── 4. MJCF 빌드 + 모델 로드 ─────────────────────────────
    xml, assets = build_mjcf(geoms_xml, assets, mass_kg, I_kgm2)
    model  = mujoco.MjModel.from_xml_string(xml, assets)
    data   = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    ball_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'plate')

    print(f"\n  geom 수  : {model.ngeom}")
    print(f"  모델 질량 확인: {model.body_mass[ball_id]*1000:.1f} g")
    print("\n  뷰어 조작: 마우스=회전/줌, q=종료\n")

    # ── 5. 시뮬레이션 루프 ───────────────────────────────────
    log_int = int(1.0 / TIMESTEP)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.azimuth   = 30
        viewer.cam.elevation = -20
        viewer.cam.distance  = 0.30
        viewer.cam.lookat[:] = [0.0, 0.0, PLATE_Z + PLATE_THICK/2 + OUTER_R]

        t, step = 0.0, 0
        while viewer.is_running() and t < SIM_DURATION:
            mujoco.mj_step(model, data)

            # 흔적 기록
            if step % TRAIL_EVERY == 0:
                trail_pts.append(get_contact_point(data, ball_id, plate_id))
                if len(trail_pts) > TRAIL_MAX:
                    trail_pts.pop(0)

            draw_trail(viewer)

            if step % log_int == 0:
                pos = data.xpos[ball_id]
                vel = np.linalg.norm(data.cvel[ball_id][:3])
                print(
                    f"  t={t:5.1f}s  "
                    f"({pos[0]*1000:+6.1f}, {pos[1]*1000:+6.1f}, {pos[2]*1000:+6.1f}) mm  "
                    f"v={vel*1000:.1f} mm/s  trail={len(trail_pts)}"
                )

            viewer.sync()
            t    += TIMESTEP
            step += 1

    print("\n  완료.")


if __name__ == '__main__':
    main()
