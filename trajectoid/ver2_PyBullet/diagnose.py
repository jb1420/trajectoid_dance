"""
PyBullet 단계별 진단
====================
A) 구(sphere) 먼저 굴림 → 경사면 자체가 맞는지 확인
B) 메쉬 로딩 확인 + 강제 힘으로 작동 여부 확인
C) 메쉬 정보 출력
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    sys.exit("pip install pybullet")

# ── 공통 파라미터 ────────────────────────────────────────────────
MESH_PATH = os.path.abspath('output_v2/trajectoid.obj')
SLOPE_DEG = 5.0
SLOPE_RAD = np.radians(SLOPE_DEG)
BALL_R    = 0.015875    # outer sphere radius (m)
MESH_SCALE = 0.0127     # 알고리즘 1단위 → m

# 경사면 파라미터
RAMP_HALF_LEN   = 1.5
RAMP_HALF_WID   = 0.4
RAMP_HALF_THICK = 0.05
RAMP_POS_Z      = 0.55

def make_ramp(client):
    """경사면 + 바닥 생성, ramp_id 반환."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        floor_id = p.loadURDF("plane.urdf")
    except Exception:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 0.01])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[5, 5, 0.01],
                                  rgbaColor=[0.6, 0.6, 0.6, 1])
        floor_id = p.createMultiBody(0, col, vis, [0, 0, -0.01])

    col = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[RAMP_HALF_LEN, RAMP_HALF_WID,
                                               RAMP_HALF_THICK])
    vis = p.createVisualShape(p.GEOM_BOX,
                               halfExtents=[RAMP_HALF_LEN, RAMP_HALF_WID,
                                            RAMP_HALF_THICK],
                               rgbaColor=[0.3, 0.5, 0.8, 1.0])
    orn = p.getQuaternionFromEuler([0, -SLOPE_RAD, 0])
    ramp_id = p.createMultiBody(0, col, vis, [0, 0, RAMP_POS_Z],
                                 baseOrientation=orn)
    p.changeDynamics(ramp_id, -1, lateralFriction=0.5, rollingFriction=0.001)
    return ramp_id

def start_pos_on_ramp():
    """경사면 위 공 중심 위치."""
    x = RAMP_HALF_LEN * 0.5
    z_surf = RAMP_POS_Z + RAMP_HALF_THICK * np.cos(SLOPE_RAD) + x * np.sin(SLOPE_RAD)
    return [x, 0.0, z_surf + BALL_R + 0.005]

def run_sim(ball_id, ramp_id, steps=3000, label=""):
    """steps 동안 시뮬, 500스텝마다 출력."""
    print(f"\n  --- {label} 시뮬 ---")
    for i in range(steps):
        p.stepSimulation()
        time.sleep(0.001)
        if i % 500 == 0:
            pos, _ = p.getBasePositionAndOrientation(ball_id)
            vel, _ = p.getBaseVelocity(ball_id)
            spd    = np.linalg.norm(vel)
            cont   = p.getContactPoints(ball_id, ramp_id)
            print(f"    step={i:4d}  pos=({pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f})"
                  f"  speed={spd:.4f}  contacts={len(cont)}")

# ════════════════════════════════════════════════════════════════════
# TEST A — 순수 구 (sphere)
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST A: 구(Sphere) 굴림 테스트  ← 경사면 자체가 맞는지")
print("=" * 60)

cA = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(0.001)
p.setPhysicsEngineParameter(numSolverIterations=50, enableConeFriction=True)
p.resetDebugVisualizerCamera(0.6, 45, -20, start_pos_on_ramp())

ramp_A = make_ramp(cA)

# 구 생성
col_sph = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R)
vis_sph = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R,
                               rgbaColor=[0.2, 0.8, 0.2, 1])
mass_sph = 0.05   # 50g
sph_id = p.createMultiBody(mass_sph, col_sph, vis_sph,
                            basePosition=start_pos_on_ramp())
p.changeDynamics(sph_id, -1,
                 lateralFriction=0.3, rollingFriction=0.001,
                 spinningFriction=0.001, restitution=0.0)

run_sim(sph_id, ramp_A, steps=3000, label="Sphere")

pos_final, _ = p.getBasePositionAndOrientation(sph_id)
rolled = pos_final[0] < start_pos_on_ramp()[0] - 0.01
print(f"\n  결과: {'✓ 구가 굴러 내려감 (경사면 정상)' if rolled else '✗ 구가 움직이지 않음!'}")

p.disconnect(cA)

# ════════════════════════════════════════════════════════════════════
# TEST B — 트레젝토이드 메쉬
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST B: Trajectoid 메쉬 테스트")
print("=" * 60)

if not os.path.exists(MESH_PATH):
    print(f"  ❌ 메쉬 없음: {MESH_PATH}")
    print("  먼저 run_pipeline.py를 실행하세요.")
    sys.exit(0)

# 메쉬 정보 출력
try:
    import trimesh as _tm
    mesh = _tm.load(MESH_PATH)
    print(f"\n  메쉬 정보:")
    print(f"    vertices : {len(mesh.vertices):,}")
    print(f"    faces    : {len(mesh.faces):,}")
    print(f"    watertight: {mesh.is_watertight}")
    bbox = mesh.bounding_box.extents
    print(f"    bbox(알고리즘 단위): {bbox[0]:.3f} × {bbox[1]:.3f} × {bbox[2]:.3f}")
    print(f"    bbox(미터): {bbox[0]*MESH_SCALE:.4f} × {bbox[1]*MESH_SCALE:.4f} × {bbox[2]*MESH_SCALE:.4f}")
except Exception as e:
    print(f"  trimesh 로드 실패: {e}")

# ── 메쉬가 너무 크면 단순화 ──────────────────────────────────────
SIMPLE_PATH = os.path.abspath('output_v2/trajectoid_simple.obj')
try:
    import trimesh as _tm
    mesh = _tm.load(MESH_PATH)
    if len(mesh.vertices) > 2000:
        print(f"\n  메쉬 단순화: {len(mesh.vertices):,} → ~500 버텍스")
        simple = mesh.simplify_quadric_decimation(500)
        simple.export(SIMPLE_PATH)
        print(f"  ✓ 단순화 메쉬 저장: {SIMPLE_PATH}")
        USE_MESH = SIMPLE_PATH
    else:
        USE_MESH = MESH_PATH
except Exception as e:
    print(f"  단순화 실패: {e}")
    USE_MESH = MESH_PATH

cB = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(0.001)
p.setPhysicsEngineParameter(numSolverIterations=50, enableConeFriction=True)
p.resetDebugVisualizerCamera(0.6, 45, -20, start_pos_on_ramp())

ramp_B = make_ramp(cB)

# 충돌 형상 생성
try:
    col_traj = p.createCollisionShape(
        p.GEOM_MESH, fileName=USE_MESH, meshScale=[MESH_SCALE] * 3)
    print(f"\n  충돌 형상 id={col_traj}  (메쉬: {os.path.basename(USE_MESH)})")
except Exception as e:
    print(f"  ❌ 충돌 형상 실패: {e}")
    col_traj = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R)
    print(f"  ⚠ 구로 폴백")

try:
    vis_traj = p.createVisualShape(
        p.GEOM_MESH, fileName=MESH_PATH, meshScale=[MESH_SCALE] * 3,
        rgbaColor=[0.85, 0.35, 0.18, 1.0])
except Exception as e:
    vis_traj = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R,
                                   rgbaColor=[0.85, 0.35, 0.18, 1.0])

# 명시적 관성 설정 (PyBullet 자동 계산 대신)
# 구 근사: I = (2/5)*m*R² (과소 추정이지만 동역학적으로 안정)
mass_traj = 0.075   # 75g (현실적)
I_sphere = (2/5) * mass_traj * BALL_R**2   # ≈ 4.76e-6 kg·m²

traj_id = p.createMultiBody(
    baseMass=mass_traj,
    baseCollisionShapeIndex=col_traj,
    baseVisualShapeIndex=vis_traj,
    basePosition=start_pos_on_ramp(),
)
p.changeDynamics(traj_id, -1,
                 localInertiaDiagonal=[I_sphere, I_sphere, I_sphere],
                 lateralFriction=0.3,
                 rollingFriction=0.001,
                 spinningFriction=0.001,
                 restitution=0.0)

# PyBullet이 계산한 관성 확인
dyn = p.getDynamicsInfo(traj_id, -1)
print(f"\n  PyBullet 동역학 정보:")
print(f"    mass     = {dyn[0]:.5f} kg")
print(f"    inertia  = {dyn[2]}")
print(f"    설정한 I = ({I_sphere:.3e}, {I_sphere:.3e}, {I_sphere:.3e})")

run_sim(traj_id, ramp_B, steps=1500, label="Trajectoid (자유 낙하)")

# 여전히 안 움직이면 → 강제 힘 적용
pos_now, _ = p.getBasePositionAndOrientation(traj_id)
if abs(pos_now[0] - start_pos_on_ramp()[0]) < 0.005:
    print(f"\n  ⚠ 자유 낙하로는 안 움직임 → 힘 0.5N 적용 테스트")
    for i in range(1500):
        # 경사 방향 힘 (-x 방향)
        p.applyExternalForce(traj_id, -1,
                             [-0.5, 0, 0], [0, 0, 0], p.WORLD_FRAME)
        p.stepSimulation()
        time.sleep(0.001)
        if i % 500 == 0:
            pos, _ = p.getBasePositionAndOrientation(traj_id)
            vel, _ = p.getBaseVelocity(traj_id)
            print(f"    step={i:4d}  pos=({pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f})"
                  f"  speed={np.linalg.norm(vel):.4f}")

p.disconnect(cB)

print("\n" + "=" * 60)
print("  진단 완료")
print("  TEST A: 구 굴림 여부 확인 → 경사면 OK/NG 판단")
print("  TEST B: 메쉬 자유 낙하 후 힘 적용 → 물리 작동 확인")
print("=" * 60)
