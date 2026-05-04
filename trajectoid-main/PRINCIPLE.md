# Trajectoids 원리 정리

이 문서는 [trajectoid-main/](.) 코드베이스의 동작 원리를 한국어로 정리한 것입니다. 사용자가 그린 2D 곡선으로부터, 평면 위를 굴렸을 때 그 곡선을 따라 정확히 굴러가는 3D 입체("trajectoid")를 생성·시뮬레이션하는 알고리즘이 핵심입니다.

> 이론 출처: Sobolev et al., *"Solid-body trajectoids shaped to roll along desired pathways"*, Nature 620, 310–315 (2023). 코드 레퍼런스: [yaroslavsobolev/trajectoids](https://github.com/yaroslavsobolev/trajectoids).

---

## 1. 큰 그림

전체 파이프라인은 다음과 같습니다.

```
사용자가 2D 곡선을 그림
        │
        ▼
[curve_editor.py]  freehand / Bezier / polyline 입력
        │   smooth · resample · scale · rotate · translate
        ▼
[trajectoids_adapter.py]
   ① 경로 검증 + 정규화
   ② 굴림(rolling)에 의한 회전 누적 계산   ← 핵심 수학
   ③ 스케일 자동 추정 (시작/끝 자세 일치)
   ④ 각 시점의 "지면 평면" 법선 추출
   ⑤ 구(sphere) ∩ 반공간들의 교집합 → implicit field
   ⑥ marching cubes로 메쉬 추출
        │
        ▼
[app.py] 3D 뷰어(OpenGL/Matplotlib) + 굴림 시뮬레이션 + STL 내보내기
```

핵심 아이디어는 **"3D 물체가 평면을 미끄럼 없이 굴러갈 때, 물체에 고정된 한 점이 그리는 평면 궤적은 회전의 누적으로 결정된다"** 는 것입니다. 이 관계를 거꾸로 풀어, 원하는 평면 곡선을 만들어내는 입체의 형상을 깎아냅니다.

---

## 2. 2D 곡선 입력 단계 ([curve_editor.py](curve_editor.py))

`CurveEditorWidget`은 Qt 위젯으로, 사용자가 곡선을 만드는 다섯 가지 도구를 제공합니다.

| 도구 | 동작 |
|------|------|
| `FREEHAND` | 마우스 이동 거리 1.6 px 이상마다 점 추가 ([curve_editor.py:323](curve_editor.py#L323)) |
| `BEZIER` | 제어점들을 추가, scipy `splprep`/`splev`로 3차 스플라인 보간 ([curve_editor.py:175-191](curve_editor.py#L175-L191)) |
| `POLYLINE` | 클릭마다 정점 추가, 더블클릭으로 종료 |
| `ERASER` | 반경 14 px 이내 가장 가까운 점 삭제 |
| `SELECT` | 제어점 드래그 |

곡선은 항상 `np.ndarray (N,2)`로 저장되고, 변환(스무딩·리샘플·스케일·회전·이동)이나 undo/redo가 가능합니다. `_closed_hint`는 "닫힌 곡선으로 해석할지" 힌트이며, Bezier 스플라인의 `per` 파라미터와 후속 검증·메쉬 생성 단계로 전달됩니다.

스무딩은 단순한 **chaikin-style corner cutting** 변형으로 구현되어 있습니다 ([trajectoids_adapter.py:163-186](trajectoids_adapter.py#L163-L186)):

```
p_i, p_{i+1}  →  0.75·p_i + 0.25·p_{i+1},  0.25·p_i + 0.75·p_{i+1}
```

리샘플링(`resample_uniform`)은 누적 호 길이를 기준으로 균등한 호 길이 간격으로 점을 다시 뽑습니다 — 이후 회전 누적 계산이 호 길이 기반이라 균등 간격이 수치적으로 안전합니다.

---

## 3. 핵심 수학: 굴림 회전의 누적 ([trajectoids_adapter.py](trajectoids_adapter.py))

### 3.1 한 스텝의 회전

평면 위에서 단위 구가 미끄럼 없이 굴러간다고 가정합시다. 점 `p_{i-1}`에서 `p_i`로 이동하는 미소 변위 벡터를 `Δ = p_{i-1} - p_i`라 할 때, 굴림 회전은:

- **회전축** = `Δ`를 평면 안에서 90° 돌린 벡터 `(Δ_y, -Δ_x, 0)` (구의 접선 평면 내, 진행 방향에 수직)
- **회전각** = `|Δ|`  (반지름 1인 구이므로 호 길이 = 회전각)

이 둘을 Rodrigues 공식 ([trajectoids_adapter.py:241-257](trajectoids_adapter.py#L241-L257))에 넣어 `_rotation_from_point_to_point`가 한 스텝의 회전 행렬을 만듭니다 ([trajectoids_adapter.py:260-267](trajectoids_adapter.py#L260-L267)).

> 부호가 `-theta`인 이유: 코드 컨벤션상 "현재 점 기준에서 이전 점이 어디 있었는가"를 거꾸로 풀어 누적하기 때문입니다. 결과적으로 시작 점 자세에서 누적 곱을 곱하면 i번째 점에 도달했을 때의 자세가 됩니다.

### 3.2 누적 회전과 구면 위 자취

`rotations_to_origin(path)`는 0번 점에서 항등행렬로 출발해 매 스텝의 회전을 오른쪽에 누적합니다:

```
R_0 = I
R_i = R_{i-1} @ step(p_i, p_{i-1})
```

`trace_on_sphere`는 구의 "맨 아래 점" `(0, 0, -r)`을 시작점으로 두고, 각 `R_i`를 곱해 구면 위 자취를 얻습니다 ([trajectoids_adapter.py:287-291](trajectoids_adapter.py#L287-L291)):

```
trace[i] = R_i · (0, 0, -r)
```

이 자취가 의미하는 바: **"평면 곡선을 따라 단위 구를 굴렸을 때, 처음에 지면에 닿아 있던 점이 구 표면을 따라 그리는 곡선"** 입니다.

### 3.3 닫힘 조건과 스케일 추정

곡선을 그대로 굴리면 시작/끝의 자세가 일치하지 않을 수 있습니다(즉, 한 바퀴 돌고 나서 입체가 처음 자세로 돌아오지 못함). 이를 정량화한 것이 **mismatch angle** — 누적 회전 행렬의 trace로부터 회전각을 뽑아냅니다 ([trajectoids_adapter.py:281-284, 294-297](trajectoids_adapter.py#L281-L297)).

코드는 곡선 전체에 곱해질 **스케일 s** 를 자유 파라미터로 두고, scipy `minimize_scalar` (bounded)로 다음 목적함수를 최소화합니다 ([trajectoids_adapter.py:300-334](trajectoids_adapter.py#L300-L334)):

```
objective(s) = mismatch_angle(s · path) + 0.75 · |trace_end - trace_start|
```

탐색 구간은 `2π / path_length`를 기준값으로 `[0.2× , 5×]`. 즉, "한 바퀴 굴림 = 곡선 한 주기" 근방에서 시작·끝 자세와 위치가 모두 일치하도록 곡선을 적절히 키우거나 줄이는 것입니다.

생성기는 mismatch가 65°를 넘으면 실패 처리하고 사용자에게 스무딩/리샘플/단순화를 권합니다 ([trajectoids_adapter.py:528-532](trajectoids_adapter.py#L528-L532)).

---

## 4. 메쉬 생성: 평면 절단 + 마칭 큐브 ([trajectoids_adapter.py:337-401](trajectoids_adapter.py#L337-L401))

### 4.1 절단 평면들

각 시점 `i`에서 구면 위 접점 위치는 `trace[i]`이고, 그 점에서의 외향 법선은 `-trace[i] / r` 입니다 (구의 안쪽을 가리키는 방향). 이 법선을 모아 `_compute_normals`가 반환합니다.

만약 입체가 "구 ∩ 모든 시점의 반공간(접평면 안쪽)"으로 정의된다면, 매 시점에서 입체는 항상 그 시점의 접평면에 정확히 닿게 됩니다. 즉, 굴림 운동 동안 어느 순간에도 평면(지면)을 뚫지 않으면서 그 평면과 접하게 되어, 결과적으로 원하는 자취를 따라 굴러갑니다.

### 4.2 implicit field

`_implicit_field`는 3D 격자 위에서 두 가지 항을 계산합니다:

- **sphere_term** = `‖p‖ − R_outer`  (구 바깥이면 양수)
- **plane_term** = `max_k ( −(n_k · p + r_core) )`  (어떤 절단 평면이라도 위반하면 양수)

최종 SDF-like field는 두 항의 `max`로, 0 이하인 영역이 입체의 내부에 해당합니다. 이것을 `skimage.measure.marching_cubes`에 넣어 `level=0` 등위면 메쉬를 추출합니다 ([trajectoids_adapter.py:393-401](trajectoids_adapter.py#L393-L401)).

격자 해상도는 GUI의 `Grid` 값(기본 96)이며, 메모리 절약을 위해 chunk 단위(`40000` 포인트, 법선은 `96`개씩 배치)로 나누어 BLAS 행렬곱을 돌립니다.

### 4.3 핵심 자료구조

`generate_trajectoid_mesh`가 반환하는 `GenerationResult`:

| 필드 | 의미 |
|------|------|
| `vertices`, `faces` | 메쉬 (STL 내보내기와 렌더링에 사용) |
| `scale` | 자동 추정된 스케일 |
| `mismatch_angle` | 라디안 단위 시작/끝 자세 차 |
| `endpoint_gap` | 구면 자취의 시작·끝 거리 |
| `resampled_points` | 스케일 적용 후의 평면 곡선 (시뮬레이션 입력) |
| `normals` | 절단 평면 법선들 |

---

## 5. 굴림 시뮬레이션 ([trajectoids_adapter.py:427-503](trajectoids_adapter.py#L427-L503))

`build_roll_simulation`은 메쉬 생성과 별도로, **3D 입체가 평면 위를 굴러가는 애니메이션 프레임**을 만듭니다. (메쉬 생성은 입체의 형상을 결정하고, 시뮬레이션은 그 입체의 시간에 따른 자세를 결정합니다.)

알고리즘은 본질적으로 §3과 같지만 출력이 다릅니다:

1. 평면 곡선을 호 길이로 누적해 `cumulative` 만들기
2. `target_roll_angle_rad × core_radius`만큼의 호 길이를 `n_frames` (기본 240)로 등분 → `arc_samples`
3. 닫힌 모드면 `mod cycle_len`으로 감싸서 `local_samples` 만들고, 곡선 위 위치 `centers_xy`를 보간
4. 각 프레임의 회전은 §3.1과 같은 공식으로 누적: 축 = `(Δ_y, −Δ_x, 0)`, 각 = `|Δ| / r`
5. translation은 `(x, y, r)` — 구의 중심이 항상 평면 위 `r` 높이에 있음

`MainWindow._simulate_last_shape`에서 `target_roll_angle_rad = trajectory_length / core_radius`로 설정하므로, 시뮬레이션은 정확히 곡선을 한 번 다 굴러간 만큼 진행됩니다 ([app.py:1940-1957](app.py#L1940-L1957)).

---

## 6. 애플리케이션 셸 ([app.py](app.py))

### 6.1 3D 뷰어

두 가지 백엔드를 제공합니다:

- **GPU 경로** (`pyqtgraph.opengl`): 커스텀 GLSL 셰이더 `_build_stainless_gpu_shader` ([app.py:87-141](app.py#L87-L141))로 스테인리스/금속 재질 표현 (key + fill + Fresnel rim)
- **CPU 폴백** (`MatplotlibTrajectoidViewerWidget`): GPU 의존성이 없을 때 사용, LOD(`_lod_max_faces=520`)로 인터랙션 시 메쉬를 다운샘플해 응답성 유지

두 뷰어 모두 회전/줌/팬, 와이어프레임 토글, 시뮬레이션 재생, 카메라 리셋·핏을 지원하고, 시뮬레이션 중에는 카메라를 45° 고정합니다.

### 6.2 좌표계 주의

[app.py:1830-1831](app.py#L1830-L1831):

```python
curve_math = curve.copy()
curve_math[:, 1] *= -1.0
```

스크린 좌표(Y-down)와 수학 좌표(Y-up)를 변환합니다. 이걸 빼먹으면 시뮬레이션에서 입체가 거꾸로 굴러갑니다.

### 6.3 STL 내보내기 ([trajectoids_adapter.py:61-133](trajectoids_adapter.py#L61-L133))

binary STL 포맷을 직접 작성합니다 (외부 라이브러리 의존 없이 `struct` + numpy structured dtype 사용). 면 법선은 `(v1-v0) × (v2-v0)`로 계산하여 정규화하고, 80바이트 헤더 + 4바이트 면 개수 + facet 배열 형태로 기록합니다.

---

## 7. 한 줄 요약

> **사용자가 그린 평면 곡선을 적절한 스케일로 키우고, 단위 구를 그 곡선을 따라 미끄럼 없이 굴린다고 가정해 매 순간 지면과 닿는 접평면을 모은 다음, "구 ∩ 모든 접평면의 안쪽 반공간"의 교집합을 마칭 큐브로 추출한 입체** — 이것이 그 곡선을 따라 굴러가는 trajectoid입니다.

수학적 핵심은 [§3](#3-핵심-수학-굴림-회전의-누적-trajectoids_adapterpy)의 회전 누적과 [§3.3](#33-닫힘-조건과-스케일-추정)의 스케일 최적화이고, 기하학적 핵심은 [§4](#4-메쉬-생성-평면-절단--마칭-큐브-trajectoids_adapterpy337-401)의 반공간 교집합입니다.
