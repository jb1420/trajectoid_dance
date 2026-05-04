"""
검증 모듈
==========
Trajectoid 메쉬 품질 + 시뮬레이션 결과 검증.

검증 항목:
    1. 내접구: 모든 면-원점 거리 ≥ R
    2. 볼록성: 모든 면 법선이 바깥 방향
    3. 미끄러짐 없음: slip_ratio ≈ 0
    4. 경로 추종: cross-track error
"""

import logging
import numpy as np
import trimesh

log = logging.getLogger(__name__)


# ================================================================
#  메쉬 검증
# ================================================================

def check_inscribed_sphere(mesh: trimesh.Trimesh, R: float,
                           tol: float = 1e-4) -> dict:
    """
    모든 면에서 원점까지의 거리가 R 이상인지 확인.

    내접구 조건: d(face, origin) ≥ R for all faces.
    """
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center

    # 면-원점 거리 = |face_center · face_normal|
    # (원점이 중심이므로 signed distance = face_center · face_normal)
    distances = np.sum(face_centers * face_normals, axis=1)

    min_dist = np.min(distances)
    mean_dist = np.mean(distances)
    n_violation = np.sum(distances < R - tol)

    passed = min_dist >= R - tol

    result = {
        'passed': passed,
        'min_distance': min_dist,
        'mean_distance': mean_dist,
        'target_R': R,
        'n_violations': int(n_violation),
    }

    if passed:
        log.info(f"  ✓ 내접구 검증 통과: min_dist={min_dist:.6f} ≥ R={R:.6f}")
    else:
        log.warning(f"  ✗ 내접구 검증 실패: min_dist={min_dist:.6f} < R={R:.6f} "
                    f"({n_violation} 면 위반)")

    return result


def check_convexity(mesh: trimesh.Trimesh) -> dict:
    """
    모든 면의 법선이 중심(원점)에서 바깥을 향하는지 확인.
    """
    centroid = mesh.centroid
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    # 면 중심에서 centroid로의 벡터와 법선의 내적이 양수여야 바깥 방향
    outward = face_centers - centroid
    dots = np.sum(outward * face_normals, axis=1)
    n_inward = np.sum(dots < 0)

    passed = n_inward == 0

    result = {
        'passed': passed,
        'n_inward_faces': int(n_inward),
        'total_faces': len(mesh.faces),
    }

    if passed:
        log.info(f"  ✓ 볼록성 검증 통과: 모든 {len(mesh.faces)} 면 법선 바깥 방향")
    else:
        log.warning(f"  ✗ 볼록성 검증 실패: {n_inward}/{len(mesh.faces)} 면 안쪽 방향")

    return result


def check_watertight(mesh: trimesh.Trimesh) -> dict:
    """메쉬가 물샐틈없는(watertight)지 확인."""
    result = {
        'passed': mesh.is_watertight,
        'is_watertight': mesh.is_watertight,
        'euler_number': mesh.euler_number,
    }
    if result['passed']:
        log.info(f"  ✓ Watertight 검증 통과 (Euler number = {mesh.euler_number})")
    else:
        log.warning(f"  ✗ Watertight 검증 실패 (Euler number = {mesh.euler_number})")
    return result


# ================================================================
#  시뮬레이션 검증
# ================================================================

def check_no_slip(history: dict, R: float) -> dict:
    """
    미끄러짐 없는 구름(rolling without slipping) 검증.

    조건: v = ω × r  (여기서 r = 구 중심 → 접촉점 벡터)
    slip_ratio = |v - ω × r| / |v|  → 0에 가까울수록 미끄러짐 없음

    Parameters
    ----------
    history : 시뮬레이션 기록 딕셔너리
    R : 구 반지름

    Returns
    -------
    dict with slip_ratio statistics
    """
    ball_pos = np.array(history['ball_pos'])
    ball_lin_vel = np.array(history['ball_lin_vel'])
    ball_ang_vel = np.array(history['ball_ang_vel'])

    slip_ratios = []

    for i in range(len(ball_pos)):
        v = ball_lin_vel[i]
        omega = ball_ang_vel[i]
        speed = np.linalg.norm(v)

        if speed < 1e-6:
            continue

        # 접촉점까지의 벡터 (근사: 중력 방향 하방으로 R)
        r_contact = np.array([0, 0, -R])

        # rolling without slipping: v = ω × r
        v_predicted = np.cross(omega, r_contact)
        slip = np.linalg.norm(v - v_predicted)
        slip_ratio = slip / speed
        slip_ratios.append(slip_ratio)

    if not slip_ratios:
        return {'passed': True, 'mean_slip': 0.0, 'max_slip': 0.0,
                'note': 'No motion detected'}

    slip_arr = np.array(slip_ratios)
    mean_slip = np.mean(slip_arr)
    max_slip = np.max(slip_arr)
    p95_slip = np.percentile(slip_arr, 95)

    threshold = 0.05  # 5% 이하면 통과
    passed = mean_slip < threshold

    result = {
        'passed': passed,
        'mean_slip': float(mean_slip),
        'max_slip': float(max_slip),
        'p95_slip': float(p95_slip),
        'n_samples': len(slip_ratios),
    }

    if passed:
        log.info(f"  ✓ 미끄러짐 검증 통과: mean={mean_slip:.4f}, "
                 f"max={max_slip:.4f}, p95={p95_slip:.4f}")
    else:
        log.warning(f"  ✗ 미끄러짐 검증 실패: mean={mean_slip:.4f} > {threshold}")

    return result


def compare_path(contact_trace: np.ndarray,
                 target_path: np.ndarray) -> dict:
    """
    접촉 궤적과 목표 경로의 cross-track error 계산.

    Parameters
    ----------
    contact_trace : (M, 3) 또는 (M, 2) — 접촉점 궤적
    target_path : (N, 2) — 목표 2D 경로

    Returns
    -------
    dict with error statistics
    """
    if len(contact_trace) == 0:
        return {'passed': False, 'note': 'No contact trace'}

    # 2D로 변환
    trace_2d = contact_trace[:, :2] if contact_trace.shape[1] >= 2 else contact_trace

    # 각 접촉점에서 목표 경로까지의 최소 거리
    errors = []
    for pt in trace_2d:
        dists = np.linalg.norm(target_path - pt, axis=1)
        errors.append(np.min(dists))

    errors = np.array(errors)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    std_err = np.std(errors)

    result = {
        'mean_error': float(mean_err),
        'max_error': float(max_err),
        'std_error': float(std_err),
        'n_points': len(errors),
    }

    log.info(f"  경로 추종 오차: mean={mean_err:.6f}, max={max_err:.6f}, "
             f"std={std_err:.6f}")

    return result


# ================================================================
#  종합 검증
# ================================================================

def run_all_checks(mesh: trimesh.Trimesh, R: float,
                   history: dict = None,
                   target_path: np.ndarray = None) -> dict:
    """모든 검증을 실행하고 결과를 통합 반환."""
    log.info("=" * 50)
    log.info("  검증 시작")
    log.info("=" * 50)

    results = {}

    # 메쉬 검증
    results['inscribed_sphere'] = check_inscribed_sphere(mesh, R)
    results['convexity'] = check_convexity(mesh)
    results['watertight'] = check_watertight(mesh)

    # 시뮬레이션 검증 (옵션)
    if history is not None:
        from .config import BALL_R
        results['no_slip'] = check_no_slip(history, BALL_R)

        if target_path is not None:
            contact_trace = []
            for pts in history.get('contact_points', []):
                if isinstance(pts, np.ndarray) and len(pts) > 0:
                    if pts.ndim == 1:
                        contact_trace.append(pts)
                    else:
                        contact_trace.extend(pts)
            if contact_trace:
                results['path_tracking'] = compare_path(
                    np.array(contact_trace), target_path
                )

    # 종합 판정
    all_passed = all(
        r.get('passed', True) for r in results.values()
        if isinstance(r, dict) and 'passed' in r
    )
    results['overall_passed'] = all_passed

    log.info("=" * 50)
    log.info(f"  종합 결과: {'✓ 통과' if all_passed else '✗ 실패'}")
    log.info("=" * 50)

    return results
