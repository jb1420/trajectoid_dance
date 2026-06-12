# dance/ — 파일별 기능 정리

Claude 가 코드 수정 전에 빠르게 구조를 파악하기 위한 참고용 문서. 코드가 바뀌면 같이 갱신할 것.

## 프로젝트 트리

```
dance/
├── app.py                  # GUI 진입점 (MainWindow + DancerEditorPanel, tab 기반 에디터)
├── viewer.py               # 3D 뷰어 (pyqtgraph.opengl 우선, matplotlib 폴백)
├── layout_canvas.py        # 2D top-down 레이아웃 캔버스 (드래그/회전/스케일)  ← 신규
├── curve_editor.py         # Freehand 커브 입력 위젯 (bezier/polyline/freehand/eraser/select)
├── trajectoids_adapter.py  # 수학·메쉬 생성 코어 (구면 trace → SDF → marching cubes → roll sim)
├── doubling.py             # 2주기(doubling) 생성 코어 — 닫힌곡선의 수학적 정확화 (π-회전 스케일)  ← 신규
├── dancer.py               # Dancer / DanceScene 데이터 모델 + generate_dancer 파이프라인
├── presets.py              # 프리셋 곡선 (Param/PresetSpec 기반, fixed + parametric)
├── scene_io.py             # `.tdance` (zip: scene.json + arrays.npz) 직렬화 + legacy 마이그레이션
├── roll_test.py            # 롤링 디버그 페이지 (독립 실행)
├── make_examples.py        # examples/ 폴더에 시드 .tdance 생성
├── make_trefoil_trajectoid.py  # trefoil 프리셋 → 2주기 trajectoid STL + 멀티뷰 미리보기 (독립 실행)
├── make_printable_trajectoid.py # 3D 프린팅용 STL (중앙 쇠구슬 캐비티) 생성 CLI (독립 실행)  ← 신규
├── examples/               # 사전 생성된 .tdance (trio / orbit / canon / four_infinity)
├── output/                 # 생성 산출물 (예: output/trefoil/*.stl, output/printable/<name>/*.stl) ← gitignore 권장
├── requirements.txt        # PySide6, numpy, scipy, scikit-image, matplotlib, pyqtgraph, PyOpenGL
├── ARCHITECTURE.md         # ← 이 문서
└── __pycache__/            # gitignore 대상
```

## 데이터 흐름 한눈에

```
사용자 입력 (preset 선택 / freehand 그리기 / 레이아웃 캔버스 회전·스케일)
        │
        ▼
   curve_xy (Nx2)         ← Dancer.curve_xy (+ Dancer.curve_params, parametric preset 일 때)
        │
        ▼ generate_dancer()
   prepare_curve → validate → generate_two_period_trajectoid_mesh → build_roll_simulation
                              (닫힌곡선은 doubling 으로 2주기 정확 생성)
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

핵심 모듈은 `trajectoids_adapter.py` (수학·메쉬 생성)와 `viewer.py` (시각화), 그 사이를 잇는 `dancer.py` (데이터 모델). 2D 배치는 `layout_canvas.py` 가 담당.

---

## 파일별 역할

### `app.py` — 메인 GUI 진입점
- `MainWindow`: 좌(roster + mode/playback) / **스테이지** 2패널 스플릿 + 좌측 Scenes 도크.
  - **스테이지(`QStackedWidget`)**: 에디터(`DancerEditorPanel`)와 viewer 를 **상호배타**로 담아 한 번에 하나만 표시(`self._stage`, index0=editor / index1=viewer). 덕분에 수정 모드에선 Curve/Layout 캔버스가, 실행 모드에선 3D 뷰가 패널 전체 폭을 차지.
- `DancerEditorPanel`: **3개 탭** 구성
  - **Curve**: source 콤보(프리셋/Freehand) + `Parameters` 그룹(parametric preset 의 슬라이더, 동적 생성) + Freehand 일 때만 보이는 `Drawing tool` 팔레트(Bezier 기본) + `CurveEditorWidget` + Smooth/Resample/Clear/Generate Mesh 버튼.
  - **Motion**: Start X/Y, Phase offset, Speed, Cycles per build. **Mixed-state sentinel** (`-50.5`, `-0.05`, `0.20`, `0` → `setSpecialValueText("—")`) 로 멀티선택 시 "값이 다름" 표기.
  - **Layout**: `LayoutCanvasWidget` (2D top-down, `Fit view` 버튼).
- 시그널 흐름:
  - `dancerChanged` → motion 만 갱신, 뷰어 transform 재바인딩.
  - `generateRequested` → 무거운 mesh + sim 재생성.
  - `nameChanged` / `colorChanged` → roster UI만 갱신.
  - 캔버스: `dancerTranslated` (드래그) → motion 핸들러로 라우팅, `dancerCurveModified` (회전·스케일) → 뷰어에서 stale mesh 제거 + dirty 처리, `selectionChanged` → roster 와 양방향 동기화 (`_suppress_roster_signal` 가드).
- 좌측 패널
  - **Roster**: `ExtendedSelection` (멀티선택 가능). + Add / Duplicate / Remove / Clear All.
  - **Mode (수정/실행 피벗 버튼)**: 파란 `_generate_all_btn` 하나가 **에디터↔뷰어 모드 전환을 겸함** (`_on_mode_button`). 수정 모드 라벨은 빌드 상태 따라 `Generate All Meshes`(0명, 비활성) / `▶ Run`(모두 최신) / `Generate & Run (n dirty)`(dirty 있음). 후자는 `_build_dirty` 로 일괄 빌드 후 자동으로 실행 모드 진입. 실행 모드에선 녹색 `✎ Edit` 로 바뀌어 다시 수정 모드로. 상태 갱신 = `_update_mode_button`, 뷰 전환 = `_enter_run_view`/`_enter_edit_view`(후자는 `_on_stop` 먼저).
  - **Playback**: duration, loop, wireframe, **shader 콤보** (`MESH_SHADERS`), **opacity 슬라이더**, Play/Stop, Reset View, Export STLs. **실행 모드 = 자동 재생**: 피벗으로 실행 진입 시 `_on_play` 가 바로 시작. `Play All` 을 수정 모드에서 눌러도 `_on_play` 가 `_enter_run_view` 로 전환 후 재생.
- **Scenes 도크** (`QDockWidget`, 좌측 도킹): `examples/*.tdance` 를 **GUI 내**에서 열고/저장(별도 파일탐색기 팝업 없이). `_build_scene_library` 가 구성 — 파일 리스트(더블클릭/Open 으로 로드, 현재 파일은 bold+선택 강조), Refresh, 파일명 입력 + Save(덮어쓰기 시 확인). `EXAMPLES_DIR = Path(__file__).parent/"examples"` 한 폴더만 탐색. `_refresh_scene_list`(스캔/강조), `_open_scene_item`(discard 가드 후 `_load_from_path`), `_save_scene_from_panel`(이름 sanitize→`_save_to_path`). load/save 흐름이 항상 리스트를 재동기화.
- File 메뉴: New / Open / Save / Save As (`scene_io` 사용). dirty 상태 추적, close 시 confirm. **Open…/Save As… 는 이제 `examples/` 밖 파일용 폴백** (네이티브 `QFileDialog`). View 메뉴의 **Scenes** 토글로 도크 표시/숨김.
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
  - `_cluster_decimate(vertices, faces, cells=18)`: **Simple 모드용** 그리드-클러스터 데시메이션. 정점을 voxel 셀(최대 `cells³`)로 묶어 대표점으로 스냅 → degenerate face 제거. O(V+F), dancer 당 1회. 면 수 ~70% 감소(예: 8.6k→2.6k)하면서 셸 형태 유지.
  - `_pose_matrix(rot, trans)` (GL 뷰어 staticmethod): 3×3 회전 + 평행이동을 **column-vector GL 변환행렬**(`QMatrix4x4`)로 묶음.
- 뷰어 API (백엔드 공통):
  - `add_or_update_dancer(d)` / `remove_dancer(id)` / `clear_dancers()`
  - `start_play(duration, loop)` / `stop_play()` / `is_playing` / `playFinished` 시그널
  - `reset_view()`, `set_wireframe(bool)`, `set_shader(name)`, `set_opacity(float)`
- 상수: `MESH_SHADERS` = **Shaded / Simple (fast) / Normal colors / Edge highlight**. **반드시 pyqtgraph 내장 셰이더만 사용** — import 시점에 `shaders.Shaders` 에 append 한 커스텀 셰이더는 GL 컨텍스트 생성 시 `initShaders()` 가 리스트를 재초기화하며 지워져 이름 조회 실패 → 메쉬가 안 그려진다(과거 `danceShaded` 가 안 보이던 원인). 기본값 `DEFAULT_MESH_SHADER="shaded"`.
- **Simple (fast) 모드** (`SIMPLE_MODE_KEY="simple"`): 콤보에서 고르면 `set_shader` 가 `_simple_mode=True` 로 전환. `_mode_mesh(st)` 가 각 dancer 를 **데시메이트된 저폴리 셸 + 값싼 `balloon` 셰이더**(조명 계산 거의 없음)로 바꿔 `setMeshData` 재업로드. GPU 가 래스터하는 삼각형 수를 크게 줄여 컴퓨팅 자원을 아낀다. 다른 모드로 돌아가면 full-res 메쉬 복원.
- **프레임 변환 (입체감 핵심)**: 메쉬는 **body frame 정점 그대로** GL 아이템에 올리고, 프레임마다 `setTransform(_pose_matrix(rot, trans+start_offset))` 로 포즈만 갱신. 정점에 회전을 굽지 않으므로 OpenGL `gl_NormalMatrix` 가 매 프레임 법선을 재정렬 → 굴러갈 때 셰이딩이 형상을 따라가 입체감이 산다. (과거엔 `verts = lod_vertices @ rot.T` 로 정점을 굽고 `computeNormals=False` 라 법선이 frame 0 에 고정 → 평평하게 보이던 버그.)
- 캐시 디렉터리(`XDG_CACHE_HOME`, `MPLCONFIGDIR`)를 import 시점에 자동 생성.

### `layout_canvas.py` — 2D top-down 레이아웃 캔버스 (신규)
- `LayoutCanvasWidget` (QWidget): 모든 dancer 의 `curve_xy + start_offset_xy` 를 월드 평면에 그림. pan(MMB) / zoom(휠) / fit(F 키 또는 더블클릭).
- 트랜스폼 기즈모 (선택된 dancer 에만 표시)
  - **드래그(curve 위 클릭)**: `start_offset_xy` 만 변경 → mesh 재생성 불필요. `dancerTranslated` 시그널.
  - **모서리 핸들 4개**: 곡선 중심 기준 균일 스케일 → `curve_xy` 자체가 변하므로 `gen_result`/`sim_result` 무효화. release 시 `dancerCurveModified` 발사.
  - **상단 회전 핸들**: 곡선 중심 기준 회전 → 동일하게 캐시 무효화.
- 시그널: `dancerTranslated(id)`, `dancerCurveModified(id)` (regen 필요), `selectionChanged(list[id])`.
- 시각 규약
  - dancer 색으로 곡선 stroke. `gen_result is None` 이면 **dashed line + 이름 옆 `⟳`** 표기 → 재생성 필요 표시.
  - 그리드: 화면에 200 단위 이상 보이면 그리드 생략 (zoom-out 시 노이즈 방지).
- 좌표 규약: 월드는 수학좌표(y-up), Qt 화면은 y-down. 모든 변환은 `_world_to_screen_xy` / `_screen_to_world` 통과.
- Ctrl+클릭으로 멀티선택 토글, 빈 곳 클릭은 선택 해제.

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
  - 통합 진입점: `generate_trajectoid_mesh(...)` → `GenerationResult` (단일 주기. 닫힌곡선은 `doubling.py` 가 대체).
- 굴림 시뮬레이션
  - `_clean_path`, `_sample_open_polyline_batch`, `_tile_open_path` (open path 의 자연스러운 무한 연장)
  - `build_roll_simulation(path_xy, target_roll_angle_rad, closed, n_frames, core_radius)` → `RollSimulationResult`.
  - **부호 규약 (project memory 참고)**: 롤 축은 `(-dy, dx, 0)`. 절대 뒤집지 말 것.
- 출력: `export_binary_stl(vertices, faces, path)`.

### `doubling.py` — 2주기(doubling) 생성 코어 (신규)
닫힌곡선을 **수학적으로 정확한 2주기 trajectoid** 로 생성. `trajectoids_adapter` 의 코어(`rotations_to_origin`, `trace_on_sphere`, `_rotation_angle`, `_compute_normals`, `_implicit_field`, `_field_to_mesh`, `GenerationResult` 등)를 그대로 재사용하고, **스케일 탐색 + 2회 순회 경로 조립**만 추가.

- **왜 2주기인가**: 한 바퀴 후 자세 복귀(net rotation `= I ∈ SO(3)`)는 조건 3개인데 자유 파라미터(스케일)는 1개라 단일 주기는 과결정 → 일반 폐곡선에서 근사일 뿐. 두 바퀴면 net rotation `= R²` 이고 **`R²=I` ⟺ `R`이 π-회전**(축 자유). π-회전 집합은 SO(3)에서 여유차원 1이라 스케일 하나로 항상 도달 가능(IVT). 이게 임의 폐곡선에 대해 2주기 trajectoid 가 **항상 존재**하는 이유. (Sobolev et al., Nature 620, 2023)
- 함수
  - `find_pi_rotation_scale(loop_xy, ...)` → `PiScaleResult`. 한 바퀴 회전각 `θ(s)` 를 스캔해 **처음 π 가 되는 스케일 `s*`** 를 찾음(첫 국소최대 = 첫 π-터치 → `minimize_scalar` 로 정밀화). 범위 내 못 찾으면 1회 확장 후 `ValueError`. `θ(s)` 는 `core_radius` 무관(r=1 규약).
  - `_doubled_closed_path(loop_xy)` → 두 바퀴 닫힌 폴리라인 `(2N+1, 2)` (정확히 2N 세그먼트). 각 바퀴의 마감 세그먼트 포함.
  - `generate_two_period_trajectoid_mesh(loop_xy, ...)` → `GenerationResult`. 전처리(recenter/smooth/resample)는 단일 주기와 동일, 스케일만 `s*` 사용. **2회 순회 trace 의 접평면**으로 구를 깎음.
    - `resampled_points` = 단일 루프 × `s*` (시뮬레이션이 보이는 한 바퀴를 굴리도록). `surface_contact_curve` 만 2회 순회 trace. `mismatch_angle` = `R²` 회전각(≈0).
  - **65° 단일-주기 검사는 미적용** — 2주기는 mismatch≈0 목표라, `find_pi_rotation_scale` 의 수용 검사(refined `θ ≈ π`)가 실패 게이트.
- 비고: 대칭 프리셋(원·lemniscate 등)은 기존 단일주기(둘레 area≈2π)보다 **작은 area≈π 형상**으로 바뀜. 한 주기 = 두 바퀴(한 바퀴 후엔 π 회전, 두 바퀴 후 원자세 복귀).

### `dancer.py` — Dancer / DanceScene 모델
- `COLOR_PALETTE` (8색).
- `Dancer` dataclass: **`dancer_id`** (uuid hex), name, `curve_source` (`"preset:circle"` / `"freehand"` / 등), `curve_xy`, `color_hex`, `start_offset_xy`, `phase_offset`, `speed_multiplier`, `n_cycles`, `closed`, **`curve_params: dict`** (parametric preset 의 슬라이더 값. 프리셋 전환 시 `_on_source_changed` 가 spec 기본값으로 채움), 캐시 결과(`gen_result`, `sim_result`, `cycle_arc_length`).
  - `Dancer.new(...)` 팩토리 (uuid 자동 부여).
- `DanceScene`: `dancers` 리스트 + `duration_seconds`, `loop`, `global_ticks=480`. `add()`, `remove(dancer_id)`, `find(dancer_id)`, `next_color()`, `next_name()` 헬퍼.
- `prepare_curve(raw, source)`: freehand 면 Y-flip (Qt 화면→수학좌표), recenter, smooth(1 pass), resample(320pt). closed=True 강제.
- `_resample_translations`, `_resample_rotations_nearest`, `normalize_sim`: sim 결과를 `global_ticks` 길이로 리샘플 (모든 dancer 의 공통 timeline).
- `generate_dancer(d, *, resolution=96, core_radius=1.0)`: prepare → validate → **`generate_two_period_trajectoid_mesh`**(닫힌곡선 항상 2주기) → **WYSIWYG 리스케일** → build_roll_simulation 까지 한 번에. 굴림은 최소 2바퀴(`laps = max(2, n_cycles)`)로 한 주기를 채워 자세가 원위치로 복귀. ValueError 로 실패 사유 전달.
  - **WYSIWYG 리스케일 (레이아웃 = source of truth)**: doubling 이 돌려준 트라젝토이드는 `pts × s*`(`s* = gen.scale`, π-회전 스케일) 위를 `core_radius` 로 구른다. `s*` 는 π-회전 정규화라 **곡선 모양만으로 정해져** 입력 크기와 무관 → 보정 없으면 3D 궤적이 레이아웃보다 `s*` 배 크고, 레이아웃에서 곡선을 키워도 3D 크기가 안 변함. 트라젝토이드는 **스케일 공변**(몸체·경로·구름반지름을 같은 배율로 키워도 여전히 유효)이므로 전체를 `1/s*` 로 스케일 → 굴림 궤적이 그린 곡선과 정확히 일치. 그 결과 **공 반지름 = `core_radius / s*`** (모양마다 조금씩 다름), 굴림 시뮬레이션도 이 `effective_radius` 로 수행. `gen.scale` 은 진단용으로 여전히 `s*` 를 보관(info 라벨 표시).

### `curve_editor.py` — Freehand 커브 그리기 위젯
- `Tool` 상수: FREEHAND / BEZIER / POLYLINE / ERASER / SELECT.
- `CurveEditorWidget` (QWidget): 마우스로 곡선 입력, undo/redo (`_history`/`_redo`).
- 외부에서 사용하는 API: `set_tool`, `set_closed_hint`, `apply_smooth(passes)`, `apply_resample(n_points)`, `clear_curve`, `sampled_points(n_samples)`, `curveChanged` 시그널.
- 좌표는 **Qt 화면 좌표(y-down)**. `prepare_curve()` 가 수학좌표로 변환.
- (선택) scipy `splprep/splev` 가 있으면 베지어 보간에 사용.
- `app.py` 의 기본 도구는 `BEZIER` (자유곡선보다 깨끗한 결과).

### `presets.py` — 프리셋 닫힌 곡선 (parametric 지원)
- 모든 프리셋은 닫힌 둘레 ≈ `2π` 가 되도록 정규화 (단위 구의 둘레와 동일 → mismatch 작아짐, auto-scale 안정).
- 데이터 구조
  - `Param(name, label, min, max, default, kind=float, step=0.05)` — UI 슬라이더 한 칸.
  - `PresetSpec(key, label, params: tuple[Param,...], generator)` — 한 프리셋 = 메타 + 생성함수.
- 헬퍼: `_normalize_length`, `_finalize` (recenter + smooth + resample + normalize), `_polygon_perimeter`.
- 생성 함수
  - **Fixed**: `gen_circle`, `gen_figure_eight`, `gen_infinity` (Bernoulli lemniscate), `gen_heart`, `gen_cardioid`, `gen_peanut`.
  - **Parametric**: `gen_ellipse(aspect)`, `gen_polygon(n_sides)`, `gen_star(n_points, inner_ratio)`, `gen_star_polygon(n_points, step)` (자기교차 펜타그램형 {n/k} — 꼭짓점을 `step` 칸씩 건너뛰며 잇는 한 획 별. n=5,step=2 가 고전 펜타그램, K5 − 5각형), `gen_rose(k)`, `gen_lissajous(a, b)`, `gen_clover(n_leaves)`.
- 레지스트리
  - `PRESET_SPECS` (튜플) — 위 모든 spec.
  - `PRESETS_BY_KEY` (dict, key → spec).
  - `PRESET_LABELS` (dict, key → label) — 호환용.
  - `LEGACY_KEY_MIGRATION`: `{"star_5": ("star", {n_points:5, inner_ratio:0.45}), "clover_3": ("clover", {n_leaves:3})}` — 옛 키 자동 마이그레이션.
- 진입점: `get_preset(name, **params)` (legacy 키도 받음), `get_preset_spec(name)`.

### `scene_io.py` — `.tdance` 파일 직렬화
- 포맷: zip 아카이브
  - `scene.json` — 메타·스칼라 (각 dancer 의 `curve_params` 포함).
  - `arrays.npz` — `{dancer_id}__{role}` 키의 numpy 배열 (curve_xy / gen vertices·faces·… / sim translations·rotations·…).
- `save_scene(scene, path)`: 임시파일에 쓰고 atomic rename.
- `load_scene(path)`: 버전 체크 (`TDANCE_VERSION = 1`) → `_reconstruct_scene`.
  - 로드 시 `curve_source` 가 `LEGACY_KEY_MIGRATION` 에 있으면 새 키 + 기본 params 로 자동 마이그레이션 (저장된 params 가 우선).
- 캐시된 mesh/sim 도 함께 저장하므로 재생성 없이 즉시 재현 가능.

### `roll_test.py` — 롤링 디버그 페이지 (독립 실행)
- 메인 앱과 분리해 굴림 키네매틱스만 검증.
- 두 body 모드: **sphere** (analytic ground truth), **trajectoid** (실제 메쉬).
- preset 경로: `line`(open), `circle`, `square`, `figure_eight`.
- 시간 스크럽 + Play/Pause, contact-point 마커, body-frame axes, HUD (center xyz, min vert z 등).
- `python roll_test.py` 로 실행. GPU 백엔드 필수 (matplotlib 폴백 미지원).
- 용도: 부호·축 버그 의심될 때 격리 검증. `MEMORY.md` 의 "Rolling kinematics sign convention" 참고.

### `make_examples.py` — 예제 씬 생성기
- `examples/` 폴더에 시드 `.tdance` 파일을 만든다 (`trio`, `orbit`, `canon`, `four_infinity`).
- 한 번 실행해 두면 앱의 Open 으로 즉시 데모 가능.

### `make_trefoil_trajectoid.py` — trefoil trajectoid 생성기 (독립 실행)
- 앱과 동일한 닫힌곡선 파이프라인(`Dancer` + `generate_dancer`, 즉 2주기 doubling)을 `trefoil` 프리셋에 적용해 산출물을 `output/trefoil/` 에 쓴다.
  - `trefoil_trajectoid.stl` — 굴림 솔리드 (binary STL, full-res). `export_binary_stl` 사용.
  - `trefoil_trajectoid.png` — 검증/미리보기. 좌상단에 **그린 경로 vs 굴림 궤적**(겹쳐야 정상), 나머지 5개 패널에 **여러 시점(front/side/three-quarter/top/underside)** 의 3D 메쉬.
- 시각 규약: 높이(z) 기준 **옅은 무지개 컬러맵**(`turbo` 를 흰색 쪽으로 블렌딩한 `_pale_rainbow_cmap`) + `LightSource` 음영으로 입체감. 색 범위(`vmin/vmax`)는 전 패널 공통.
- 진단 출력: `s*`(π-회전 스케일), 2바퀴 mismatch(≈0), 구면 endpoint gap, 정점/면 수, bbox, `effective_radius = core_radius/s*`.
- 진입점: `python make_trefoil_trajectoid.py` (`dance/` 에서 실행). matplotlib `Agg` 백엔드라 GUI 불필요.

### `make_printable_trajectoid.py` — 3D 프린팅용 STL 생성기 (독립 실행, 신규)
실제로 굴러가는 **물리 trajectoid** 를 FDM 으로 뽑기 위한 CLI. 핵심 통찰: trajectoid 가 설계 경로대로 구르려면 **무게중심이 구 중심에 있어야** 하는데(이론은 균질 구 가정), 접평면들이 플라스틱을 비대칭으로 깎아내므로 플라스틱-only 셸은 CoM 이 치우친다. 해결책(원논문 · `ver2_PyBullet/config.py` 와 동일)은 **중앙에 밀도 높은 쇠구슬**을 박는 것 — 구슬 질량이 지배해 CoM 을 중심에 고정.
- 그래서 trajectoid 를 **중앙 구형 캐비티가 있는 셸**로 생성:
  `solid = {‖p‖ ≤ R_outer} ∩ {nₖ·p ≥ -R_core} \ {‖p‖ < R_cavity}` (outer 구 ∩ 접평면 컷 ∖ 구슬 포켓).
- **경로 두 종류** (`load_curve` → `(curve, name, closed)`):
  - **닫힌 루프**(`--preset` dance 프리셋, 기본 trefoil) → `build_normals(closed=True)` 가 **정식 period-2(doubling)** 사용: `doubling.find_pi_rotation_scale` + `_doubled_closed_path`.
  - **열린 주기 경로**(`--periodic sinusoid|zigzag`, `--periods`/`--amp`) → `build_normals(closed=False)` 가 **단일주기**(`estimate_scale` + 1회 trace) 사용. 대칭 주기 경로는 한 주기에 자세가 자연히 복귀(mismatch≈0)해 그대로 앞으로 타일링. (`--path-file` + `--open` 으로 임의 열린 경로도 가능.)
  - 분기 결과는 `ScaleInfo(scale, mismatch_rad, period_label)` 로 통일해 리포트에 전달.
- **수학 코어 재사용**: 위 두 경로 모두 `trajectoids_adapter._compute_normals`/`_implicit_field`/`_field_to_mesh` 를 그대로 호출. 추가한 것은 **캐비티 항과 반쪽 분할 항**뿐.
- `hollow_fields()`: 비싼 `_implicit_field` 를 1회만 호출해 solid 필드를 얻고, `max(solid, R_cavity−‖p‖)` 로 hollow(셸), `max(hollow, ∓z)` 로 top/bottom 반쪽 필드를 값싸게 파생. **z=0 절단면은 marching cubes 가 자동으로 평면 캡** → 각 반쪽이 watertight (boolean 불필요).
- **물리 치수 파생** (`PrintGeometry`, 단위 mm): `--ball-mm`(구슬 지름)에서 `cavity_r = ball/2 + clearance`, `core_r = cavity_r + wall`(=알고리즘 단위 1.0 의 물리 크기 → `scale_mm`), `outer_r = core_r × shell_ratio`. 알고리즘 코어 구를 1.0 에 고정하므로 `scale_mm = core_r_mm`. **`--scale`(`size_scale`)** 은 알고리즘-단위 비율은 그대로 두고 `scale_mm`·표시 치수만 곱하는 **균일 확대**(예: 3 = 3배 인쇄).
- **3개 모드**: `split`(기본, 두 반쪽 + 조립 참고 셸) / `inplace`(단일 폐쇄 셸, 프린트 일시정지 삽입) / `solid`(**캐비티 없이 꽉 채운** 순수 trajectoid — 균질 구가 아니라 CoM 이 치우쳐 단독으로는 정확히 안 구름. 리포트가 그 centroid 오프셋을 명시). split 의 bottom 반쪽은 X축 180° 회전 후 바닥(z=0)에 평평히 안착시켜 export.
- **출력**: `output/printable/<name>/` 에 `<stem>_*.stl`(`stem = name` + `_x{scale}` (≠1배일 때)) — `*_half_top.stl`/`*_half_bottom.stl`/`*_assembled_ref.stl`(split) 또는 `*_shell.stl`(inplace) / `*_solid.stl`(solid) + `*_preview.png`(입력 경로 / 단면(+쇠구슬) / 외곽 셸 3D).
- **진단 리포트**: 치수, 질량(PLA 1240 / 강철 7874 kg/m³), (캐비티 모드)구슬 질량비·**CoM 오프셋**(외경의 5% 초과 시 경고) / (solid 모드)centroid 오프셋, `ScaleInfo` scale·mismatch, watertight 여부.
- 진입점 예: `python make_printable_trajectoid.py --preset trefoil --ball-mm 19.05` (닫힌·쇠구슬) / `--periodic sinusoid` (열린 주기·쇠구슬) / `--preset trefoil --mode solid --scale 3` (3배·솔리드). matplotlib `Agg`. 콘솔이 cp949 라도 깨지지 않게 import 시점에 stdout/stderr 를 UTF-8 로 `reconfigure`.

### `examples/` — 사전 생성된 `.tdance` 파일들
`trio.tdance`, `orbit.tdance`, `canon.tdance`, `four_infinity.tdance`.

### `requirements.txt` — 의존성
PySide6, numpy, scipy, scikit-image, matplotlib, pyqtgraph, PyOpenGL.

### `__pycache__/` — Python 바이트코드 캐시 (gitignore 대상)

---

## 자주 하는 작업과 진입점

| 하고 싶은 것 | 손볼 파일 |
|---|---|
| 메쉬가 점·와이어로 보이는 문제 | `viewer._make_animation_mesh`, `_DancerState.lod_*` |
| 굴러갈 때 평평·입체감 없음 | `viewer._draw_frame` 는 `setTransform(_pose_matrix(...))` 로 포즈만 갱신(정점 굽지 말 것) → 법선이 형상 따라감 |
| 특정 셰이더에서 오브젝트가 안 보임 | 커스텀 셰이더 금지(내장만). `MESH_SHADERS` 값은 pyqtgraph 내장 이름이어야 함 |
| 저사양·성능 모드 | Simple (fast) = `_cluster_decimate` 저폴리 + `balloon`. `viewer._mode_mesh` / `set_shader` 의 `_simple_mode` |
| 메쉬 색·셰이딩·투명도 | `viewer.MESH_SHADERS`, `set_shader`/`set_opacity`, `app.py` playback form |
| 굴림 방향이 거꾸로 | `trajectoids_adapter.build_roll_simulation` 의 axis = `(-dy, dx, 0)` |
| 2주기 스케일/π-회전 탐색 조정 | `doubling.find_pi_rotation_scale` (`lo_factor`/`hi_factor`/`coarse_tol`/`fine_tol`) |
| 레이아웃과 3D 궤적 크기/위치가 안 맞음 | `dancer.generate_dancer` 의 **WYSIWYG 리스케일**(`1/s*`) + `effective_radius`. 끄면 `s*` 배 차이 재발 |
| 닫힌곡선 메쉬가 근사로만 나옴 / mismatch 큼 | `doubling.generate_two_period_trajectoid_mesh` (2주기 정확 생성). 단일주기는 `trajectoids_adapter.generate_trajectoid_mesh` |
| 새 프리셋 추가 (fixed) | `presets.py` 에 `gen_xxx` 함수 + `PresetSpec(..., (), gen_xxx)` 를 `PRESET_SPECS` 에 등록 |
| 새 프리셋 추가 (parametric) | `gen_xxx(param=...)` + `PresetSpec(..., (Param(...),...), gen_xxx)`. UI 슬라이더는 자동 생성 |
| 옛 프리셋 키 호환 | `presets.LEGACY_KEY_MIGRATION` 에 매핑 추가 (`scene_io` + `get_preset` 둘 다 사용) |
| 새 export 포맷 | `trajectoids_adapter.export_binary_stl` 옆에 추가 + `app._on_export_*` 훅 |
| 3D 프린팅용 STL(쇠구슬 캐비티) | `make_printable_trajectoid.py`. 캐비티/반쪽은 `hollow_fields` 의 `max()` 항, 치수는 `PrintGeometry`. `--mode split/inplace/solid`, 크기 `--scale` |
| 열린 주기 경로 trajectoid | `make_printable_trajectoid.py --periodic sinusoid/zigzag` (`PERIODIC_PATHS`, 단일주기 `build_normals(closed=False)`) |
| `.tdance` 스키마 변경 | `scene_io.TDANCE_VERSION` 올리고 `_build_*`/`_reconstruct_*` 동시 수정 |
| examples 씬을 GUI에서 열고/저장 | `app.MainWindow._build_scene_library`, `_refresh_scene_list`, `_save_scene_from_panel` (좌측 Scenes 도크) |
| 수정/실행 모드(에디터↔뷰어) 동작 | `app.MainWindow._on_mode_button`, `_update_mode_button`, `_enter_run_view`/`_enter_edit_view`, `self._stage`(`QStackedWidget`) |
| Curve 편집기 도구 | `curve_editor.Tool`, `CurveEditorWidget` 메서드 |
| 멀티 dancer 타임라인 | `dancer.normalize_sim`, `viewer.GLOBAL_TICKS`, `_draw_frame` |
| 2D 배치/회전/스케일 UX | `layout_canvas.LayoutCanvasWidget` (`_begin_*`/`_update_*` 패밀리) |
| Editor 멀티선택 동작 | `app.DancerEditorPanel._apply_bulk_state`, `_set_float_field` (sentinel 처리) |

## 좌표·시간 규약 (실수 방지)

- **2D 곡선 좌표계**: 내부는 모두 수학좌표(y-up). Qt freehand 입력만 `prepare_curve` 에서 y-flip. 레이아웃 캔버스도 y-up 월드를 사용하고 `_world_to_screen_xy` 에서만 y 를 뒤집는다.
- **롤 축**: 항상 `(-dy, dx, 0)`. 부호 뒤집으면 굴림 방향이 시각적으로 반대로 보임 (회귀 시 `roll_test.py` 로 검증).
- **공통 타임라인**: 모든 dancer 의 sim 결과는 `normalize_sim` 으로 `global_ticks`(=480) 길이로 맞춘 뒤 `phase_offset` + `speed_multiplier` 로 개별 위상/속도 조절.
- **트랜스폼**: 렌더 시 `world = (verts @ rot.T) + translation + start_offset`. 순서를 바꾸면 회전 중심이 어긋남.
- **Cache 무효화**: `curve_xy` 가 바뀌면(레이아웃 회전·스케일, freehand 수정, parametric 슬라이더, source 전환) `gen_result = sim_result = None` 으로 비워야 함. 레이아웃 캔버스에서는 dashed stroke + `⟳` 마크로 시각적 피드백.
- **2주기(period-2)**: 닫힌곡선 trajectoid 의 한 주기는 **두 바퀴**다. 한 바퀴 후엔 π 회전 자세, 두 바퀴 후 원자세 복귀(`R²=I`). 굴림은 `laps = max(2, n_cycles)` 로 돌린다. 메쉬는 한 바퀴 trace 가 아니라 **2회 순회 접평면**으로 깎이므로, 같은 곡선이라도 단일주기 형상보다 작다(area≈π vs 2π). `gen_result` 캐시가 있는 옛 `.tdance` 는 재생성 전까지 단일주기 형상 유지.
- **WYSIWYG 스케일**: `generate_dancer` 가 doubling 결과를 `1/s*` 로 균일 스케일하므로 **3D 굴림 궤적 = 레이아웃에 그린 `curve_xy`** (위치·크기 모두 일치). 대가로 **공 반지름이 dancer 마다 `core_radius/s*`** 로 달라지고, 굴림은 그 `effective_radius` 로 돈다. 레이아웃 캔버스는 별도 변환 없이 `curve_xy + start_offset_xy` 만 그려도 3D 와 맞는다 (단 freehand 는 `curve_xy` 가 Qt y-down 화면좌표라 레이아웃에서 상하반전됨 — 별개 이슈). 곡선 크기를 바꾸려면 레이아웃에서 스케일 후 재생성하면 3D 도 같은 비율로 커진다.
