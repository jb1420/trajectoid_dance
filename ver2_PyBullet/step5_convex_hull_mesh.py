"""
Step 5: Convex Hull → 최종 메쉬 출력
======================================

Ruled surface 점 구름으로부터 3D convex hull을 계산하여
trajectoid의 최종 삼각형 메쉬를 생성.

v1 대체 대상: step1_boolean_subtraction.py 전체
    (manifold3d boolean 연산 → convex hull로 대체)

핵심 장점:
    - boolean 연산 완전 제거 (manifold3d, CoACD 의존성 제거)
    - 결과가 태생적으로 볼록체 → PyBullet에서 직접 사용 가능
"""

import os
import logging
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from .config import OUTPUT_DIR

log = logging.getLogger(__name__)


def generate_mesh(point_cloud: np.ndarray,
                  output_dir: str = OUTPUT_DIR,
                  name: str = 'trajectoid') -> trimesh.Trimesh:
    """
    점 구름으로부터 convex hull 메쉬를 생성하고 파일로 저장.

    Parameters
    ----------
    point_cloud : (N, 3) ndarray — ruled surface 점 구름
    output_dir : str — 출력 디렉토리
    name : str — 파일 이름 접두어

    Returns
    -------
    mesh : trimesh.Trimesh — 최종 trajectoid 메쉬
    """
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Convex hull 계산: {len(point_cloud):,} 점 입력")

    # 퇴화 방지: 미세 지터 추가
    jitter = np.random.default_rng(42).normal(0, 1e-10, point_cloud.shape)
    hull = ConvexHull(point_cloud + jitter)

    log.info(f"  Hull 결과: {len(hull.vertices)} vertices, "
             f"{len(hull.simplices)} faces")

    # trimesh 메쉬 생성
    mesh = trimesh.Trimesh(
        vertices=point_cloud[hull.vertices],
        faces=_remap_faces(hull.simplices, hull.vertices),
        process=True,
    )

    # 법선 방향 수정 (바깥 방향으로)
    trimesh.repair.fix_normals(mesh)

    # 메쉬 품질 보고
    log.info(f"  최종 메쉬: {len(mesh.vertices)} vertices, "
             f"{len(mesh.faces)} faces")
    log.info(f"  Watertight: {mesh.is_watertight}")
    log.info(f"  Volume: {mesh.volume:.6f}")
    bb = mesh.bounding_box.extents
    log.info(f"  Bounding box: {bb[0]:.4f} × {bb[1]:.4f} × {bb[2]:.4f}")

    # 파일 저장
    obj_path = os.path.join(output_dir, f'{name}.obj')
    stl_path = os.path.join(output_dir, f'{name}.stl')

    mesh.export(obj_path)
    mesh.export(stl_path)

    log.info(f"  저장: {obj_path}")
    log.info(f"  저장: {stl_path}")

    return mesh


def _remap_faces(simplices, hull_vertices):
    """
    ConvexHull.simplices는 원본 점 구름의 인덱스를 참조.
    trimesh에서는 hull.vertices 배열 기준의 인덱스가 필요.
    """
    # hull_vertices[i] = 원본 인덱스 → i = 새 인덱스
    inv_map = {orig: new for new, orig in enumerate(hull_vertices)}
    return np.array([[inv_map[v] for v in face] for face in simplices])


def mesh_report(mesh: trimesh.Trimesh, name: str = '') -> dict:
    """메쉬 품질 지표 딕셔너리 반환."""
    label = f"[{name}] " if name else ''
    info = {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'watertight': mesh.is_watertight,
        'volume': mesh.volume,
        'bbox': mesh.bounding_box.extents.tolist(),
        'is_convex': mesh.is_convex if hasattr(mesh, 'is_convex') else None,
    }
    for k, v in info.items():
        log.info(f"  {label}{k}: {v}")
    return info
