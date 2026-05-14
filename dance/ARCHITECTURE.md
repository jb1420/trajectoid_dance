# dance/ — 파일별 기능 정리

Claude 가 코드 수정 전에 빠르게 구조를 파악하기 위한 참고용 문서. 코드가 바뀌면 같이 갱신할 것.

## 데이터 흐름 한눈에

```
사용자 입력 (preset 선택 / freehand 그리기)
        │
        ▼
   curve_xy (Nx2)         ← Dancer.curve_xy
        │
        ▼ generate_dancer()
   prepare_curve → validate → generate_trajectoid_mesh → build_roll_simulation
        │                              │                          │
        ▼                              ▼                          ▼
   resampled curve              GenerationResult            RollSimulationResult
                                (vertices, faces, …)        (translations, rotations)
                                            │                          │
                                            └──────────┬───────────────┘
                                                       ▼
                                                  Viewer 렌더링
                                              (3D mesh + 굴러가는 모션)
```

핵심 모듈은 `trajectoids_adapter.py` (수학·메쉬 생성)와 `viewer.py` (시각화), 그 사이를 잇는 `dancer.py` (데이터 모델).

---

## 파일별 역할

### `app.py` — 메인 GUI 진입점
- `MainWindow`: 좌(roster + playback) / 중(`DancerEditorPanel`) / 우(viewer) 3패널 스플릿.
- `DancerEditorPanel`: 선택된 dancer의 curve source / motion / cycles 편집 + Generate Mesh 버튼.
- 시그널 흐름:
  - `dancerChanged` → motion 만 갱신, 뷰어 transform 재바인딩.
  - `generateRequested` → 무거운 mesh + sim 재생성.
  - `nameChanged` / `colorChanged` → roster UI만 갱신.
- Playback 컨트롤: duration, loop, wireframe, **shader 콤보** (`MESH_SHADERS`), **opacity 슬라이더**, Play/Stop, Reset View, Export STLs.
- File 메뉴: New / Open / Save / Save As (`scene_io` 사용). dirty 상태 추적.
- 진입점: `python app.py`.

### `viewer.py` — 다중 dancer 3D 뷰어
- 두 개의 백엔드를 가진 위젯을 제공하고, `make_viewer()`가 환경에 맞는 백엔드를 골라 반환.
  - **`_MultiTrajectoidGLViewer`** (GPU, pyqtgraph.opengl) — 메인 경로.
  - **`_MultiTrajectoidMplViewer`** (Matplotlib 폴백) — pyqtgraph 가 없을 때만.
- `_DancerState` (slots dataclass): per-dancer 캐시 (vertices/faces/lod/translations/rotations/trajectory + GL items).
- 핵심 함수
  - `_make_animation_mesh(...)`: GPU 백엔드는 face 전체를 그대로 사용 (decimation 안 함). 과거 stride 샘플링 때문에 mesh 가 점/구멍처럼 보이던 버그를 제거.
  - `_make_mpl_mesh(...)`: Matplotlib 전용 보수적 decimator (stride ≤ 3).
  - `_hex_to_rgba(...)`: dancer color hex → rgba.
- 뷰어 API (백엔드 공통):
  - `add_or_update_dancer(d)` / `remove_dancer(id)` / `clear_dancers()`
  - `start_play(duration, loop)` / `stop_play()` / `is_playing` / `playFinished` 시그널
  - `reset_view()`, `set_wireframe(bool)`, `set_shader(name)`, `set_opacity(float)`
- 상수: `MESH_SHADERS` (Shaded / Normal colors / Balloon / Edge highlight).
- 프레임 변환식: `world = (lod_vertices @ rot.T) + (translation + start_offset)`.
- 캐시 디렉터리(`XDG_CACHE_HOME`, `MPLCONFIGDIR`)를 import 시점에 자동 생성.

### `trajectoids_adapter.py` — 수학·메쉬 생성 코어
업스트림 `yaroslavsobolev/trajectoids/compute_trajectoid.py` 의 알고리즘 직접 포팅.

- 데이터 클래스
  - `ValidationResult` (errors / suggestions / 길이 / closure_gap / 간격비)
  - `GenerationResult` (vertices, faces, scale, mismatch_angle, endpoint_gap, resampled_points, normals, surface_contact_curve)
  - `RollSimulationResult` (translations_xyz, rotations, trajectory_xy, achieved_roll_angle_rad, completed_target, message)
- 곡선 유틸: `path_length`, `resample_uniform`, `smooth_path`, `curvature_profile`, `validate_path`.
- 회전·롤링 코어
  - `_rodrigues_rotation_matrix`
  - `_rotation_from_point_to_point` (no-slip step rotation)
  - `rotations_to_origin` (path →누적 회전 행렬)
  - `trace_on_sphere`, `mismatch_angle`, `_objective_for_scale`, `estimate_scale`
- 메쉬 생성 파이프라인
  1. `_compute_normals` (구면 trace → half-space 평면 법선)
  2. `_implicit_field` (chunked: sphere ∩ 모든 half-space → SDF 그리드)
  3. `_field_to_mesh` (skimage `marching_cubes`)
  - 통합 진입점: `generate_trajectoid_mesh(...)` → `GenerationResult`.
- 굴림 시뮬레이션
  - `_clean_path`, `_sample_open_polyline_batch`, `_tile_open_path` (open path 의 자연스러운 무한 연장)
  - `build_roll_simulation(path_xy, target_roll_angle_rad, closed, n_frames, core_radius)` → `RollSimulationResult`.
  - **부호 규약 (project memory 참고)**: 롤 축은 `(-dy, dx, 0)`. 절대 뒤집지 말 것.
- 출력: `export_binary_stl(vertices, faces, path)`.

### `dancer.py` — Dancer / DanceScene 모델
- `COLOR_PALETTE` (8색).
- `Dancer` dataclass: id, name, curve_source(`"preset:circle"` / `"freehand"`), curve_xy, color_hex, start_offset_xy, phase_offset, speed_multiplier, n_cycles, closed, 캐시 결과(`gen_result`, `sim_result`, `cycle_arc_length`).
  - `Dancer.new(...)` 팩토리.
- `DanceScene`: `dancers` 리스트 + `duration_seconds`, `loop`, `global_ticks`. `next_color()`, `next_name()` 헬퍼.
- `prepare_curve(raw, source)`: freehand 면 Y-flip (Qt 화면→수학좌표), recenter, smooth, resample.
- `_resample_translations`, `_resample_rotations_nearest`, `normalize_sim`: sim 결과를 `global_ticks` 길이로 리샘플 (모든 dancer 의 공통 timeline).
- `generate_dancer(d, *, resolution, core_radius)`: prepare → validate → generate_trajectoid_mesh → build_roll_simulation 까지 한 번에. `gen_result`/`sim_result`/`cycle_arc_length` 채워서 반환. ValueError 로 실패 사유 전달.

### `curve_editor.py` — Freehand 커브 그리기 위젯
- `Tool` 상수: FREEHAND / BEZIER / POLYLINE / ERASER / SELECT.
- `CurveEditorWidget` (QWidget): 마우스로 곡선 입력, undo/redo (`_history`/`_redo`).
- 외부에서 사용하는 API: `set_tool`, `set_closed_hint`, `apply_smooth(passes)`, `apply_resample(n_points)`, `clear_curve`, `sampled_points(n_samples)`, `curveChanged` 시그널.
- 좌표는 **Qt 화면 좌표(y-down)**. `prepare_curve()` 가 수학좌표로 변환.
- (선택) scipy `splprep/splev` 가 있으면 베지어 보간에 사용.

### `presets.py` — 프리셋 닫힌 곡선
- 모든 프리셋은 닫힌 둘레 ≈ `2π` 가 되도록 정규화 (단위 구의 둘레와 동일 → mismatch 작아짐, auto-scale 안정).
- 헬퍼: `_normalize_length`, `_finalize` (recenter + smooth + resample + normalize).
- 프리셋 함수: `circle`, `figure_eight`, `heart`, `star_5`, `peanut`, `clover_3`.
- 매핑: `PRESETS` (key → 함수), `PRESET_LABELS` (key → 표시이름).
- `get_preset(name)` 으로 호출.

### `scene_io.py` — `.tdance` 파일 직렬화
- 포맷: zip 아카이브
  - `scene.json` — 메타·스칼라.
  - `arrays.npz` — `{dancer_id}__{role}` 키의 numpy 배열 (curve_xy / gen vertices·faces·… / sim translations·rotations·…).
- `save_scene(scene, path)`: 임시파일에 쓰고 atomic rename.
- `load_scene(path)`: 버전 체크 (`TDANCE_VERSION = 1`) → `_reconstruct_scene`.
- 캐시된 mesh/sim 도 함께 저장하므로 재생성 없이 즉시 재현 가능.

### `roll_test.py` — 롤링 디버그 페이지 (독립 실행)
- 메인 앱과 분리해 굴림 키네매틱스만 검증.
- 두 body 모드: **sphere** (analytic ground truth), **trajectoid** (실제 메쉬).
- preset 경로: `line`(open), `circle`, `square`, `figure_eight`.
- 시간 스크럽 + Play/Pause, contact-point 마커, body-frame axes, HUD (center xyz, min vert z 등).
- `python roll_test.py` 로 실행. GPU 백엔드 필수 (matplotlib 폴백 미지원).
- 용도: 부호·축 버그 의심될 때 격리 검증. `MEMORY.md` 의 "Rolling kinematics sign convention" 참고.

### `make_examples.py` — 예제 씬 생성기
- `examples/` 폴더에 세 개의 `.tdance` 시드 파일을 만든다: `trio.tdance`, `orbit.tdance`, `canon.tdance`.
- 한 번 실행해 두면 앱의 Open 으로 즉시 데모 가능.

### `examples/` — 사전 생성된 `.tdance` 파일들 (`make_examples.py` 산출물)

### `requirements.txt` — 의존성
PySide6, numpy, scipy, scikit-image, matplotlib, pyqtgraph, PyOpenGL.

### `__pycache__/` — Python 바이트코드 캐시 (gitignore 대상)

---

## 자주 하는 작업과 진입점

| 하고 싶은 것 | 손볼 파일 |
|---|---|
| 메쉬가 점·와이어로 보이는 문제 | `viewer._make_animation_mesh`, `_DancerState.lod_*` |
| 메쉬 색·셰이딩·투명도 | `viewer.MESH_SHADERS`, `set_shader`/`set_opacity`, `app.py` playback form |
| 굴림 방향이 거꾸로 | `trajectoids_adapter.build_roll_simulation` 의 axis = `(-dy, dx, 0)` |
| 새 프리셋 추가 | `presets.py` 에 함수 추가 + `PRESETS`/`PRESET_LABELS` 등록 |
| 새 export 포맷 | `trajectoids_adapter.export_binary_stl` 옆에 추가 + `app._on_export_*` 훅 |
| `.tdance` 스키마 변경 | `scene_io.TDANCE_VERSION` 올리고 `_build_*`/`_reconstruct_*` 동시 수정 |
| Curve 편집기 도구 | `curve_editor.Tool`, `CurveEditorWidget` 메서드 |
| 멀티 dancer 타임라인 | `dancer.normalize_sim`, `viewer.GLOBAL_TICKS`, `_draw_frame` |

## 좌표·시간 규약 (실수 방지)

- **2D 곡선 좌표계**: 내부는 모두 수학좌표(y-up). Qt freehand 입력만 `prepare_curve` 에서 y-flip.
- **롤 축**: 항상 `(-dy, dx, 0)`. 부호 뒤집으면 굴림 방향이 시각적으로 반대로 보임 (회귀 시 `roll_test.py` 로 검증).
- **공통 타임라인**: 모든 dancer 의 sim 결과는 `normalize_sim` 으로 `global_ticks` 길이로 맞춘 뒤 `phase_offset` + `speed_multiplier` 로 개별 위상/속도 조절.
- **트랜스폼**: 렌더 시 `world = (verts @ rot.T) + translation + start_offset`. 순서를 바꾸면 회전 중심이 어긋남.
