"""
Step 1: Trajectoid Mesh Boolean Subtraction (Automation)
=========================================================

3ds Max 수동 작업을 자동화:
  - outer sphere에서 cutting boxes를 순차적으로 빼어 (Boolean Difference)
    최종 trajectoid.obj 메쉬를 생성합니다.

사용법:
  python step1_boolean_subtraction.py

주요 파라미터 (SETTINGS 섹션에서 조정):
  INPUT_MODE     : 'load_boxes' (기존 .obj 불러오기) or 'generate' (경로에서 실시간 생성)
  PATH_DATA_FILE : 경로 데이터 .npy 파일 경로  (INPUT_MODE='generate' 일 때)
  BOXES_FOLDER   : cutting boxes .obj 파일 폴더  (INPUT_MODE='load_boxes' 일 때)
  CORE_RADIUS    : 알고리즘 내부 정규화 반지름 (= 1, 변경 금지)
  OUTER_RADIUS   : outer sphere 반지름 (> CORE_RADIUS, 실물 기준 15.875/12.7 ≈ 1.25)
  CUT_SIZE       : cutting box 한 변 길이 / CORE_RADIUS  (원본 알고리즘과 동일하게 10)
  OUTPUT_FILE    : 결과 메쉬 저장 경로

의존 라이브러리:
  trimesh          (pip install trimesh)
  manifold3d       (pip install manifold3d)  ← Boolean 연산 백엔드
  numpy, tqdm

Notes on geometry (기하학적 관계):
  - outer sphere radius R > core_radius r = 1
  - cutting box 윗면이 z = -r 에 위치하므로, R > r 이면 outer sphere 표면을 파고듦
  - 3D 프린팅 실물 치수: r = 12.7 mm (1인치 볼베어링), R = 15.875 mm  → R/r ≈ 1.25
"""

import os
import sys
import glob
import logging
import numpy as np
import trimesh
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)


# ================================================================
#  SETTINGS
# ================================================================
INPUT_MODE      = 'generate'   # 'load_boxes' | 'generate'

# INPUT_MODE = 'load_boxes' 일 때
BOXES_FOLDER    = 'trajectoids/examples/little-prince-2/cut_meshes'

# INPUT_MODE = 'generate' 일 때
# step4_control.py 의 SAVE_PATH_DATA=True 로 먼저 저장된 경로를 사용하거나,
# 직접 path_data.npy 를 지정하세요.
PATH_DATA_FILE  = 'output/path_data.npy'
KX              = 1.0    # x 스케일 (경로 생성 시)
KY              = 1.0    # y 스케일 (경로 생성 시)

# 공통 기하학 파라미터
CORE_RADIUS     = 1.0    # 알고리즘 내부 정규화 반지름 (변경 금지)
OUTER_RADIUS    = 1.25   # outer sphere 반지름 (실물 15.875mm / 12.7mm ≈ 1.25)
CUT_SIZE        = 10     # cutting box 크기 / CORE_RADIUS (원본과 동일)

SPHERE_SUBDIVISIONS = 5  # icosphere 세분화 횟수 (클수록 정밀, 5 ≈ 10K faces)

OUTPUT_DIR      = 'output'
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, 'trajectoid.obj')

BOOLEAN_ENGINE  = 'manifold'   # 'manifold' (권장) | 'blender' | 'scad'


# ================================================================
#  rolling-without-slipping 회전 행렬 계산
#  (compute_trajectoid.py 의 rotation_to_origin 을 iterative 로 재구현)
# ================================================================
def _rotation_step(point: np.ndarray, prev_point: np.ndarray) -> np.ndarray:
    """
    구가 prev_point → point 방향으로 구를 때의 회전 행렬 (4×4 homogeneous).

    핵심 물리:
      - 미끄러짐 없이 구르므로 arc_length = rotation_angle
      - 회전 축은 이동 방향에 수직인 수평 축
    """
    v = prev_point - point                     # point 에서 prev_point 로의 벡터
    theta = float(np.linalg.norm(v))           # 이동 거리 = 회전 각도
    if theta < 1e-12:
        return trimesh.transformations.identity_matrix()

    axis = np.array([v[1], -v[0], 0.0])        # 수직 축 (xy 평면 내)
    return trimesh.transformations.rotation_matrix(
        angle=-theta,
        direction=axis,
        point=[0.0, 0.0, 0.0],
    )


def compute_cumulative_rotations(path_data: np.ndarray) -> list:
    """
    경로 각 점에 대한 누적 회전 행렬 목록 반환.

    Returns
    -------
    rotations : list of (4, 4) ndarray, 길이 = len(path_data)
        rotations[i] = 구가 origin 에서 path_data[i] 까지 굴렀을 때의 누적 회전
    """
    n = len(path_data)
    rotations = [trimesh.transformations.identity_matrix()]

    for i in tqdm(range(1, n), desc='  회전 행렬 계산', unit='pts', ncols=72):
        R_step = _rotation_step(path_data[i], path_data[i - 1])
        R_cum  = trimesh.transformations.concatenate_matrices(rotations[-1], R_step)
        rotations.append(R_cum)

    return rotations


# ================================================================
#  cutting box 생성 (in-memory)
# ================================================================
def generate_cutting_boxes(
    path_data: np.ndarray,
    core_radius: float = CORE_RADIUS,
    cut_size: float = CUT_SIZE,
    kx: float = 1.0,
    ky: float = 1.0,
) -> list:
    """
    compute_shape() 로직을 in-memory 로 재현.
    각 경로 점마다 cutting box를 생성하여 리스트로 반환.

    Returns
    -------
    boxes : list of trimesh.Trimesh
    """
    data = path_data.copy()
    data[:, 0] *= kx
    data[:, 1] *= ky

    log.info(f"  경로 점 수: {len(data)}")

    # base box: cutting box 의 원형 (아직 회전 없음)
    #   - 크기 : cut_size * core_radius 의 정육면체
    #   - 중심 : [0, 0, -core_radius - cut_size*core_radius/2]
    #             (top face 가 z = -core_radius 에 닿음)
    box_size   = cut_size * core_radius
    box_center_z = -core_radius - 0.5 * box_size
    base_box = trimesh.creation.box(
        extents=[box_size, box_size, box_size],
        transform=trimesh.transformations.translation_matrix(
            [0.0, 0.0, box_center_z]
        ),
    )

    # 누적 회전 행렬 계산
    rotations = compute_cumulative_rotations(data)

    # 각 점의 cutting box 생성
    boxes = []
    for i in tqdm(range(len(data)), desc='  Cutting boxes 생성', unit='box', ncols=72):
        box = base_box.copy()
        box.apply_transform(rotations[i])
        boxes.append(box)

    return boxes


# ================================================================
#  기존 .obj 파일에서 cutting box 로드
# ================================================================
def load_cutting_boxes(folder: str) -> list:
    """
    folder 안의 test_0.obj … test_N.obj 를 순서대로 로드.

    Returns
    -------
    boxes : list of trimesh.Trimesh
    """
    pattern  = os.path.join(folder, 'test_*.obj')
    all_paths = glob.glob(pattern)

    if not all_paths:
        raise FileNotFoundError(
            f"'{folder}' 에서 test_*.obj 파일을 찾을 수 없습니다.\n"
            f"먼저 compute_shape() 로 cutting boxes 를 생성하세요."
        )

    # test_0, test_1, … 순서로 정렬
    def sort_key(p):
        name = os.path.splitext(os.path.basename(p))[0]  # 'test_42'
        return int(name.split('_')[1])

    all_paths.sort(key=sort_key)
    log.info(f"  {len(all_paths)} 개의 cutting box .obj 파일 발견")

    boxes = []
    for p in tqdm(all_paths, desc='  Cutting boxes 로드', unit='box', ncols=72):
        m = trimesh.load(p, force='mesh', process=False)
        boxes.append(m)

    return boxes


# ================================================================
#  Boolean subtraction  (핵심)
# ================================================================
def boolean_subtract_all(
    outer_sphere: trimesh.Trimesh,
    cutting_boxes: list,
    engine: str = BOOLEAN_ENGINE,
) -> trimesh.Trimesh:
    """
    outer_sphere 에서 cutting_boxes 전체를 뺀 결과 메쉬를 반환.

    전략:
      1. union(cutting_boxes) → boxes_union  (하나의 복합 메쉬)
      2. difference(outer_sphere, boxes_union) → trajectoid

    이 방식이 sequential difference (상자 하나씩 빼기) 보다
    수치적으로 안정적이고 빠릅니다.
    """
    # ── manifold3d 설치 확인 ──────────────────────────────────
    if engine == 'manifold':
        try:
            import manifold3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "manifold3d 가 설치되지 않았습니다.\n"
                "  pip install manifold3d\n"
                "또는 BOOLEAN_ENGINE = 'blender' 로 변경하세요 "
                "(Blender 가 PATH에 있어야 함)."
            )

    n = len(cutting_boxes)
    log.info(f"  Boolean union: {n} 개의 cutting box 합산 중 …")

    # 1단계: 박스들 전체를 union
    #   trimesh.boolean.union(meshes) 은 모든 메쉬를 하나로 합칩니다.
    boxes_union = trimesh.boolean.union(cutting_boxes, engine=engine)

    if not isinstance(boxes_union, trimesh.Trimesh):
        raise RuntimeError(
            "cutting boxes union 이 유효한 메쉬를 반환하지 않았습니다. "
            "입력 박스들이 manifold(물샐틈없음) 한지 확인하세요."
        )

    log.info(
        f"  Union 결과: {len(boxes_union.vertices):,} vertices, "
        f"{len(boxes_union.faces):,} faces"
    )

    # 2단계: outer sphere - boxes_union
    log.info("  Boolean difference: sphere − boxes_union …")
    # trimesh 최신 버전: difference([base, subtractor], engine=...)  형식
    result = trimesh.boolean.difference([outer_sphere, boxes_union], engine=engine)

    if not isinstance(result, trimesh.Trimesh):
        raise RuntimeError(
            "Boolean difference 가 유효한 메쉬를 반환하지 않았습니다.\n"
            "OUTER_RADIUS 가 CORE_RADIUS 보다 큰지, "
            "메쉬가 watertight 한지 확인하세요."
        )

    return result


# ================================================================
#  메쉬 품질 리포트
# ================================================================
def mesh_report(mesh: trimesh.Trimesh, name: str = '') -> None:
    label = f"[{name}] " if name else ''
    log.info(f"  {label}vertices : {len(mesh.vertices):,}")
    log.info(f"  {label}faces    : {len(mesh.faces):,}")
    log.info(f"  {label}watertight: {mesh.is_watertight}")
    log.info(f"  {label}volume   : {mesh.volume:.4f}")
    bb = mesh.bounding_box.extents
    log.info(f"  {label}bbox     : {bb[0]:.3f} × {bb[1]:.3f} × {bb[2]:.3f}")


# ================================================================
#  MAIN
# ================================================================
def main() -> None:
    print("=" * 60)
    print("  Step 1: Trajectoid Boolean Subtraction")
    print("=" * 60)
    print(f"  입력 모드    : {INPUT_MODE}")
    print(f"  core_radius  : {CORE_RADIUS}")
    print(f"  outer_radius : {OUTER_RADIUS}  (실물 15.875 / 12.7 mm ≈ 1.25)")
    print(f"  cut_size     : {CUT_SIZE}")
    print(f"  Boolean 엔진 : {BOOLEAN_ENGINE}")
    print(f"  출력 파일    : {OUTPUT_FILE}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Cutting boxes 준비 ────────────────────────────────
    print("\n[1/4] Cutting boxes 준비 중 …")
    if INPUT_MODE == 'load_boxes':
        cutting_boxes = load_cutting_boxes(BOXES_FOLDER)

    elif INPUT_MODE == 'generate':
        if not os.path.exists(PATH_DATA_FILE):
            raise FileNotFoundError(
                f"경로 데이터 파일을 찾을 수 없습니다: {PATH_DATA_FILE}\n"
                "INPUT_MODE='load_boxes' 로 변경하거나 올바른 경로를 지정하세요."
            )
        path_data = np.load(PATH_DATA_FILE)
        log.info(f"  경로 데이터 로드: {PATH_DATA_FILE}  ({len(path_data)} pts)")
        cutting_boxes = generate_cutting_boxes(path_data, CORE_RADIUS, CUT_SIZE, KX, KY)

    else:
        raise ValueError(f"INPUT_MODE 는 'load_boxes' 또는 'generate' 이어야 합니다: '{INPUT_MODE}'")

    print(f"  → Cutting boxes 준비 완료: {len(cutting_boxes)} 개")

    # ── 2. Outer sphere 생성 ─────────────────────────────────
    print("\n[2/4] Outer sphere 생성 중 …")
    # icosphere: 균일하게 삼각형이 분포되어 Boolean 연산에 적합
    outer_sphere = trimesh.creation.icosphere(
        subdivisions=SPHERE_SUBDIVISIONS,
        radius=OUTER_RADIUS,
    )
    mesh_report(outer_sphere, 'outer_sphere')
    print(f"  → Outer sphere 완료  (R = {OUTER_RADIUS})")

    # ── 3. Boolean subtraction ───────────────────────────────
    print(f"\n[3/4] Boolean subtraction ({len(cutting_boxes)} boxes) …")
    print("  (시간이 소요될 수 있습니다 — manifold3d 기준 약 1~5분)")
    trajectoid_mesh = boolean_subtract_all(outer_sphere, cutting_boxes, BOOLEAN_ENGINE)
    mesh_report(trajectoid_mesh, 'trajectoid')
    print("  → Boolean subtraction 완료")

    # ── 4. 결과 저장 ─────────────────────────────────────────
    print(f"\n[4/4] 메쉬 저장 중: {OUTPUT_FILE}")
    trajectoid_mesh.export(OUTPUT_FILE)

    # 추가로 .stl 도 저장 (MuJoCo / 3D 프린팅에 유용)
    stl_path = OUTPUT_FILE.replace('.obj', '.stl')
    trajectoid_mesh.export(stl_path)
    log.info(f"  STL도 함께 저장: {stl_path}")

    print("\n" + "=" * 60)
    print(f"  완료!  결과 파일:")
    print(f"    OBJ : {os.path.abspath(OUTPUT_FILE)}")
    print(f"    STL : {os.path.abspath(stl_path)}")
    print(f"  메쉬 통계:")
    mesh_report(trajectoid_mesh)
    print("=" * 60)


if __name__ == '__main__':
    main()
