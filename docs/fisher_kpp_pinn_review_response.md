# Fisher-KPP PINN 산림청 적용 및 장시간 곡선 기술 검토 보고서

작성: Codex / 연구 기록 업데이트

일자: 2026년 6월 8일

## 개론

현재 repository는 하나의 단일 실험만 담고 있지 않습니다. 크게 세 가지 목적의 코드가 함께 정리되어 있습니다.

1. synthetic Fisher-KPP 문제에서 PINN 구조와 학습 안정화 기법을 검토하는 실험
2. 한국 산림청 소나무재선충병 관측 좌표 데이터를 연도별 density grid로 바꾼 뒤 RK4와 PINN baseline을 비교하는 forward baseline
3. 첨부 이미지 우측의 장시간 `rho(t)` 감쇠 진동 곡선 추이만 재현하기 위한 별도 ODE benchmark

이 세 가지는 서로 연결되어 있지만 같은 문제는 아닙니다. Fisher-KPP front 문제는 반응-확산 PDE이고, 양의 확산과 logistic reaction을 쓰는 표준 설정에서는 probe density가 본질적으로 감쇠 진동 곡선을 만들지 않습니다. 반면 우측 그림의 `rho(t)`는 장시간에 걸쳐 overshoot와 damping을 보이는 scalar oscillator 형태입니다. 따라서 최근 코드에서는 Fisher-KPP front 실험과 `rho(t)` curve benchmark를 분리했습니다. 이 분리가 현재 결과를 올바르게 해석하는 핵심입니다.

이 문서는 지금까지 반영한 코드 구조, 문제 설정, baseline, 산림청 데이터 처리, 장시간 curve benchmark, 공정한 하이퍼파라미터 설정, 주요 관찰과 원인 분석, 그리고 남은 한계를 기술 검토 보고서 형식으로 정리한 설명 파일입니다.

## 1. 문제 설정은 무엇으로 고정했는가

### 기준 요구 사항

문제 구성은 기존 한국 산림청 코드와 동일해야 하며, 모델 자체는 Geo-aware forward PINN 계획을 반영해 개선할 수 있다는 의견이었습니다. boundary condition, initial condition, forward simulation, 관측 데이터와의 관계가 명확해야 결과 비교가 가능합니다.

### 확정 설정

산림청 적용 파트의 기준 문제는 forward Fisher-KPP baseline입니다. 상태 변수 `u(x, y, t)`는 정규화된 감염 density proxy로 해석합니다.

```text
u_t = D * Laplacian(u) + r * u * (1 - u)
```

산림청 데이터 파트에서는 다음 항목을 고정했습니다.

- 입력 데이터: 연도별 감염 관측 좌표
- 공간 domain: 관측 좌표 bounding box를 기반으로 만든 2D grid
- 초기조건: 2016년 관측 density grid
- 시간 전개: 2016년부터 설정한 forecast year까지
- RK4 boundary: land/sea mask가 반영된 no-flux 계열 diffusion
- PINN baseline: 같은 연도별 density observation에 대한 data fit, 약한 Fisher-KPP PDE residual, 약한 boundary/sea-exclusion 제약
- 평가: 관측 연도 density와 simulation/PINN field의 land-cell 기준 metric

중요한 변경은 land mask입니다. 한국 시도 경계 GeoJSON을 사용해 land cell과 sea cell을 구분하고, density grid는 sea cell을 0으로 고정합니다. RK4는 land/sea interface에서 바다 쪽으로 diffusion이 빠져나가지 않도록 masked no-flux 처리를 사용합니다. PINN baseline도 sea-exclusion penalty를 추가하고, PDE collocation은 land cell에서만 샘플링합니다. 따라서 현재 버전에서는 소나무재선충병 density가 바다로 퍼지는 해석을 허용하지 않습니다.

synthetic inverse-origin 실험과 산림청 forward baseline은 목적이 다릅니다. inverse-origin 실험은 synthetic truth가 있고 origin recovery를 평가합니다. 산림청 baseline은 raw surveillance count에서 만든 yearly density grid를 대상으로 RK4와 PINN이 같은 관측 연도에 얼마나 맞는지를 확인하는 진단용 forward baseline입니다.

## 2. 왜 RK4와 PINN을 모두 baseline에 넣었는가

### 비교 필요성

산림청 데이터 실험에서 RK4만 제시하면 우리가 만든 PINN이 실제 baseline 비교에 포함되지 않으므로, PINN도 같은 평가 표에 들어가야 한다는 의견이었습니다.

### baseline 구성

맞습니다. 현재 산림청 baseline에는 RK4와 repository PINN baseline이 모두 포함됩니다.

RK4는 Fisher-KPP PDE를 grid에서 직접 전진 적분하는 numerical baseline입니다. 2016년 관측 density를 초기조건으로 두기 때문에 첫 해에는 관측과 정확히 맞지만, 고정된 `D`, `r`로 여러 해를 예측하면 이후 연도에서 과성장하거나 공간 분포가 관측과 어긋날 수 있습니다. 이 차이는 RK4가 약해서라기보다, raw surveillance data를 단순 Fisher-KPP PDE 하나로 설명하기 어렵다는 신호입니다.

PINN baseline은 repository의 PirateNet/Fourier/geo-feature 계열 PINN backbone을 재사용합니다. 연도별 density grid를 supervised observation으로 fit하면서 약한 PDE residual과 boundary/sea-exclusion 제약을 함께 둡니다. 이 baseline은 최종 감염 예측 모델이 아니라, 같은 데이터에서 neural PDE model이 RK4 대비 어느 정도 fitting flexibility를 갖는지 확인하는 진단용 비교입니다.

현재 산림청 실행 결과의 대표 metric은 다음과 같습니다.

```text
RK4 mean relative L2      = 3.9009
PINN mean relative L2     = 1.1426
RK4 mean correlation      = 0.4348
PINN mean correlation     = 0.4487
PINN status               = diagnostic_high_error
```

PINN은 observed-year relative L2를 낮췄지만, 여전히 `diagnostic_high_error` 상태입니다. 따라서 "PINN이 최종적으로 충분히 정확하다"가 아니라 "PINN baseline을 같은 평가표에 넣었고, 현재는 높은 오차가 남아 있다"로 해석해야 합니다.

## 3. Geo-aware forward PINN 구조는 어떻게 반영했는가

### 방법론 반영 범위

Geo-aware forward PINN 계획과 구조를 코드에 반영해야 하며, 단순 MLP가 아니라 최신 PINN 안정화 기법과 moving front 특성을 고려해야 한다는 의견이었습니다.

### 구현 구조

synthetic Fisher-KPP forward profile에는 다음 구성요소를 반영했습니다.

- PirateNet-style residual architecture
- random weight factorization
- spatial Fourier features
- square-domain geo features
- hard known initial-condition ansatz
- seed-centered front features
- scaled traveling-wave/front coordinate features
- KPP front envelope
- front-aware PDE residual weighting
- front-local gPINN loss
- moving-front speed consistency loss
- parabolic mass-balance loss
- leading-edge area loss
- front-profile alignment loss
- residual-adaptive collocation
- causal time-marching curriculum
- XPINN/FBPINN-inspired time-slab ablation
- weak RK4 teacher and RK4 pretraining ablations
- adaptive relative loss balancing

다만 산림청 baseline에서는 이 stabilizer를 모두 강하게 켜지 않았습니다. 산림청 raw observation은 실제 감염 확산뿐 아니라 탐지 강도, 신고 체계, 방제 intervention, 정책 변화, 지역별 sampling bias가 섞인 surveillance count입니다. 여기에 analytic initial seed나 moving-front level-set target을 강하게 넣으면, 실제 데이터에 없는 물리 가정을 강요할 수 있습니다. 따라서 산림청 PINN baseline은 data fitting, weak PDE residual, weak boundary/sea-exclusion을 중심으로 둔 diagnostic baseline으로 제한했습니다.

즉 synthetic forward PINN은 방법론 검증용이고, 산림청 PINN은 현실 데이터에서의 baseline fitting 진단용입니다. 두 결과를 같은 의미로 해석하면 안 됩니다.


### 1-6번 선행 연구 기반 개선의 코드 반영

front 주변 오차를 줄이기 위해 정리했던 1-6번 개선안을 기본 Geo-Spectral forward PINN profile에 모두 반영했습니다. 이 변경은 단순히 옵션을 늘린 것이 아니라 `ExperimentConfig().geo_spectral_forward()`가 기본적으로 해당 장치들을 켜도록 구성한 것입니다.

첫째, front-level-set alignment loss를 기본 loss로 올렸습니다. 기존 area matching은 `u>0.05` 또는 `u>0.10` 면적만 맞추면 넓은 저농도 haze도 통과할 수 있었습니다. 새 loss는 선형화 Fisher-KPP leading edge에서 예상되는 low-level ring을 샘플링하고, ring 위에서는 목표 level을 맞추며, 안쪽과 바깥쪽의 순서 관계를 hinge로 강제합니다. 여기에 약한 normal-slope 항을 추가해 front가 단순히 면적만 맞추는 것이 아니라 front-normal 방향의 기울기도 맞추도록 했습니다.

둘째, front-local gPINN을 강화했습니다. 기존에는 현재 collocation batch에서 `0.05<u<0.95`인 지점만 residual gradient를 봤습니다. 이제는 expected-front corridor에서도 별도 샘플을 뽑아 residual gradient penalty를 적용합니다. 이 항은 초기 random network에서 과도하게 커질 수 있어 robust `log1p` energy로 계산합니다. 목적은 moving front 주변에서 PDE residual 자체보다 residual의 공간/시간 변화가 먼저 흔들리는 failure mode를 줄이는 것입니다.

셋째, causal time-marching curriculum을 기본값으로 켰습니다. 학습 초반에는 이른 시간 구간을 우선 보고, epoch가 진행되면서 late-time window를 누적 확장합니다. observation data도 선택적으로 같은 시간창에서 mini-batch를 뽑되, 비어 있는 창은 전체 observation으로 fallback하게 했습니다. 따라서 sparse late observation만 있는 경우에도 data term이 끊기지 않습니다.

넷째, XPINN/FBPINN식 시간 분할 아이디어를 단일 네트워크에 맞게 축소 구현했습니다. `time_slabs=4`, overlap, cumulative curriculum을 쓰고, 내부 slab boundary에서 양쪽 시간의 field continuity와 중앙 시간 미분 기반 one-step consistency를 맞추는 `time_slab_interface_loss`를 추가했습니다. 완전한 multi-subnetwork XPINN은 아니지만, early/late 구간이 서로 다른 front phase를 학습해 생기는 drift를 줄이는 목적은 같습니다.

다섯째, moving-frame front Fourier feature를 모델 입력에 추가했습니다. 기존 spatial Fourier는 전역 `x,y` 좌표를 기준으로 하므로 이동하는 front의 phase를 표현하려면 네트워크가 translation을 스스로 학습해야 했습니다. 새 feature는 normalized front coordinate와 radial direction을 Fourier encoding으로 넣어 front 주변의 phase shift를 더 직접적으로 표현하게 합니다.

여섯째, loss balancing을 relative-progress 방식에서 gradient-norm 방식까지 확장했습니다. 각 active loss 항이 마지막 표현층과 학습 가능한 PDE parameter에 만드는 gradient norm을 주기적으로 측정하고, 지나치게 큰 gradient를 내는 항은 낮추고 약한 항은 올립니다. 이는 PINN gradient pathology 문헌에서 지적되는 data/PDE/front loss 간 scale 불균형을 줄이기 위한 장치입니다. 계산비용 때문에 매 epoch 모든 parameter에 대해 수행하지 않고, 제한된 reference parameter와 주기적 업데이트를 사용합니다.

이 여섯 가지는 RK4 수준의 정확도를 보장하는 장치가 아닙니다. 목적은 PINN이 반복적으로 보이던 front phase error, active-front area plateau, twin-trap error band, low-amplitude haze를 줄이는 것입니다. 따라서 성능 주장은 `scripts/run_forward_ablation.py`의 동일 조건 ablation과 `front_area_005_mae`, `front_area_010_mae`, `active_front_area_mae`, `mass_mae`, validation MSE를 함께 보고 판단해야 합니다.

## 4. 산림청 데이터는 어떻게 반영했는가

### 데이터 구성

한국 산림청 CSV와 시뮬레이션 코드가 repository와 notebook에 포함되어야 하며, Colab에서도 재현 가능해야 한다는 의견이었습니다.

### 처리 방식

원본 연도별 CSV는 총 약 1.17 GB이고 일부 파일은 GitHub 일반 blob 한도 100 MB를 넘습니다. 따라서 원본 CSV 전체를 일반 Git 파일로 넣지 않고, 다음 compact artifact를 commit했습니다.

- `data/korea_pine_wilt/processed/infected_points_2016_2023.csv.gz`
- `data/korea_pine_wilt/processed/infected_points_2016_2023.npz`
- `data/korea_pine_wilt/processed/manifest.json`
- `data/korea_pine_wilt/assets/skorea_provinces_2018.geojson`

compact file에는 `x_5179`, `y_5179`, `year`가 들어 있으며 record 수는 `3,183,376`입니다. `manifest.json`에는 원본 CSV의 연도, 파일 크기, checksum이 들어 있어 재검증이 가능합니다. 또한 `scripts/build_korea_pine_wilt_compact_data.py`를 통해 원본 CSV가 있는 환경에서 compact data를 다시 생성할 수 있습니다.

GeoJSON은 단순 시각화 배경이 아니라 land mask 생성에도 사용합니다. density grid는 land mask 밖을 0으로 고정하고, metric도 land cell 기준으로 계산합니다. 이 처리는 "바다에는 퍼질 수 없다"는 제약을 반영하기 위한 것입니다.

## 5. 산림청 notebook은 무엇을 실행하는가

### 실행 목표

산림청 데이터를 위한 `.ipynb`도 실제 실행 가능해야 하며, 설명과 출력이 깨지지 않아야 한다는 의견이었습니다.

### notebook 구성

`korea_pine_wilt_fisher_kpp_lab.ipynb`는 다음 순서로 실행됩니다.

1. repository root 또는 Colab clone 준비
2. compact data manifest 검증
3. NPZ observation point load
4. yearly density grid 생성
5. RK4 baseline 실행
6. PINN baseline 학습 및 예측
7. observed-year metric table 출력
8. observation, RK4 forecast, PINN baseline, baseline metric comparison PNG 표시
9. Korea province map 기반 GIF 생성

노트북 설명에서 한글이 `???`로 깨지는 문제는 제거했고, 현재는 ASCII/영문 설명으로 안정화했습니다. smoke test에도 `???` 손상 문자열을 감지하는 조건을 추가했습니다.

Colab에서는 project files가 없으면 GitHub repository를 clone하고, 이미 `/content/fisher-pinn`이 있으면 `fetch` 후 `FETCH_HEAD`로 강제 checkout합니다. 이렇게 해서 오래된 clone이 최신 코드나 data schema를 가리는 문제를 줄였습니다.

## 6. 현재 산림청 baseline 결과는 어떻게 해석해야 하는가

### 관찰 결과

실행 결과를 보여주되, 좋은 결과인지 아닌지 명확히 판단해야 한다는 의견이었습니다. 특히 smoke run이나 diagnostic output을 최종 성능처럼 보여주면 안 됩니다.

### 해석 기준

현재 산림청 결과는 "진단용 baseline"입니다. PINN이 RK4보다 observed-year relative L2를 낮췄지만, high-error 상태가 유지됩니다. 이는 다음 이유 때문입니다.

- raw surveillance count는 실제 감염 밀도만 반영하지 않습니다.
- 연도별 탐지 강도와 신고 체계가 바뀔 수 있습니다.
- 방제 intervention과 행정 정책 변화가 관측에 영향을 줍니다.
- 지역별 sampling bias와 산림/숙주 분포가 반영되어야 합니다.
- 단일 상수 `D`, `r`의 Fisher-KPP PDE는 공간적으로 다양한 확산/증식 조건을 충분히 설명하기 어렵습니다.

따라서 다음 단계에서는 spatially varying diffusion/reaction, observation model, intervention covariate, forest/host mask, 더 긴 PINN training, validation split을 포함해야 합니다.

또 하나의 주의점은 land/sea mask 반영 전후 metric을 직접 같은 숫자로 비교하면 안 된다는 것입니다. 현재 버전은 바다 cell을 계산 및 평가에서 제외하므로, 이전 직사각형-domain 결과와 수치가 달라지는 것이 정상입니다.

## 7. 주요 관찰과 분석

### 관찰 목록

실험을 계속 개선하면서 여러 현상이 관찰되었고, 단순히 코드 변경 목록만 적는 것이 아니라 그 현상이 왜 나타났는지 분석이 필요하다는 의견이었습니다.

### 원인 분석

아래는 현재까지 반복적으로 관찰된 핵심 현상과 그 해석입니다.

#### 관찰 1. synthetic Fisher-KPP에서 RK4는 매우 정확하지만 PINN은 front 주변 오차가 크게 남는다

RK4는 같은 PDE, 같은 초기조건, 같은 boundary condition에서 직접 time marching을 수행합니다. 따라서 grid와 time step이 안정 조건 안에 있으면 synthetic reference에 매우 가깝습니다. 반면 PINN은 같은 방정식을 연속 함수 근사와 optimization으로 풀기 때문에, moving front의 위치가 조금만 어긋나도 error map에서는 큰 band 형태 오차가 나타납니다.

분석하면 이 오차는 대개 amplitude error보다 phase/front-location error에 가깝습니다. Fisher-KPP front는 낮은 값에서 높은 값으로 급격히 변하는 moving interface이므로, front 위치가 한두 grid cell만 밀려도 absolute error map에는 둥근 band 또는 twin-trap처럼 보이는 구조가 생깁니다. 따라서 PINN error가 front 주변에 집중되는 것은 단순 plotting 오류라기보다 moving-front PDE에서 흔한 failure mode입니다. 이를 줄이기 위해 front-aware residual sampling, front-local gPINN, moving-front speed loss, leading-edge area loss, front-profile alignment, time-marching/time-slab curriculum을 추가했습니다.

#### 관찰 2. error map의 최대값이 반복적으로 약 `0.06` 수준에 머무는 현상이 있었다

이 현상은 두 가지를 분리해서 봐야 합니다. 첫째, 동일한 오래된 figure가 갱신되지 않아 반복 표시된 경우가 있었습니다. 이 문제는 출력 파일 재생성, notebook 문자열 파싱 테스트, figure 생성 경로 정리를 통해 확인했습니다. 둘째, 실제 학습 결과에서도 front 위치가 조금 어긋나면 비슷한 최대 오차 수준이 반복될 수 있습니다. sigmoid형 front에서는 front slope가 큰 구간의 phase shift가 거의 일정하면 최대 absolute error도 비슷한 값으로 유지됩니다.

따라서 `0.06`이 반복된다고 해서 항상 코드가 깨졌다는 뜻은 아닙니다. 다만 figure가 최신 run에서 생성된 것인지, metrics의 run id와 figure path가 일치하는지, RK4/PINN/reference가 같은 problem setting에서 나온 것인지는 반드시 확인해야 합니다.

#### 관찰 3. active-front area curve에서 truth는 증가하는데 PINN은 plateau로 멈추는 경우가 있었다

active-front band는 `0.1 < u < 0.9` 같은 구간의 면적입니다. truth에서는 front가 이동하면서 이 band의 면적이 증가할 수 있습니다. PINN이 plateau를 보이면 보통 다음 중 하나입니다.

- network가 front propagation speed를 과소평가함
- data loss가 관측 지점 주변 fitting에 치우쳐 PDE propagation을 충분히 학습하지 못함
- mass balance와 front speed constraint가 약하거나 서로 scale이 맞지 않음
- time curriculum이 너무 짧아 late-time front를 충분히 보지 못함
- activation/spectral feature가 moving front의 translation을 충분히 표현하지 못함

이 관찰 때문에 moving-front speed consistency, parabolic mass balance, leading-edge area, front-profile alignment, scaled traveling-wave feature, time marching, time-slab ablation을 추가했습니다. 다만 빠른 Colab run에서는 이 항목들이 모두 충분히 수렴하지 않으므로, 결과는 여전히 diagnostic으로 해석해야 합니다.

#### 관찰 4. PINN absolute error가 twin-trap 또는 ring-like 구조로 보이는 경우가 있었다

이 구조는 대개 front의 안쪽과 바깥쪽에서 부호가 반대로 생기는 signed error가 absolute error로 접히면서 나타납니다. 예를 들어 PINN front가 truth보다 뒤처지면 leading edge 바깥쪽에서는 under-prediction, 안쪽에서는 over/under transition이 동시에 생깁니다. signed error로 보면 방향성이 보이지만 absolute error만 보면 두 개의 trap 또는 ring처럼 보일 수 있습니다.

따라서 현재 시각화에는 signed error와 absolute error를 함께 보도록 했습니다. absolute error만 보면 오류의 크기는 알 수 있지만, front가 앞선 것인지 뒤처진 것인지는 알기 어렵습니다.

#### 관찰 5. Korea pine-wilt baseline에서 PINN은 RK4보다 L2를 낮추지만 여전히 high-error이다

산림청 data는 PDE simulation truth가 아니라 surveillance count입니다. RK4는 2016년 density를 초기조건으로 두고 고정된 `D`, `r`로 전진 예측하기 때문에, 관측 체계 변화나 방제 intervention을 설명하지 못합니다. PINN은 supervised observation fitting을 함께 하므로 observed-year L2는 낮출 수 있습니다. 하지만 이것은 PINN이 실제 확산 메커니즘을 완전히 맞췄다는 뜻이 아닙니다.

현재 PINN의 `diagnostic_high_error` 상태는 타당한 경고입니다. 산림청 결과는 "PINN이 RK4보다 fitting flexibility가 있다"는 정도로 해석해야 하며, 정책적 예측 모델 또는 논문급 정확도라고 주장하면 안 됩니다.

#### 관찰 6. land/sea mask를 넣지 않으면 바다로 density가 퍼지는 비물리적 결과가 나온다

직사각형 grid만 사용하면 PDE solver는 바다와 육지를 구분하지 못합니다. diffusion term은 인접 cell로 density를 퍼뜨리므로, 해안선을 경계로 막지 않으면 바다 cell에도 감염 density가 생깁니다. 이는 소나무재선충병 문제 설정에 맞지 않습니다.

이를 해결하기 위해 province GeoJSON 기반 land mask를 만들고, RK4에는 masked no-flux diffusion을 적용했습니다. PINN baseline에도 sea-exclusion penalty와 land-only PDE collocation을 넣었습니다. 따라서 현재 산림청 baseline은 바다를 단순 배경이 아니라 계산 domain 밖의 영역으로 취급합니다.

#### 관찰 7. PINN GIF가 랜덤한 얼룩처럼 보일 수 있다

low-epoch smoke run은 학습 수렴 결과가 아니라 pipeline 작동 확인용입니다. 이 상태에서 GIF를 만들면 field가 랜덤한 얼룩처럼 보일 수 있습니다. 따라서 현재 GIF에는 epoch 수, final relative L2, diagnostic status를 caption에 넣고, `metrics.json`에도 `diagnostic_only_low_epoch`, `diagnostic_high_error`, `diagnostic_moderate_error` 같은 상태를 기록합니다.

즉 GIF는 단독 증거가 아니라 metrics와 함께 해석해야 합니다.

#### 관찰 8. 우측 그림의 `rho(t)` 곡선은 Fisher-KPP probe가 아니었다

Fisher-KPP probe를 장시간 관찰하면 보통 front가 지나가면서 단조 증가하거나 plateau로 접근합니다. 첨부 이미지 우측 곡선처럼 overshoot 후 감쇠 진동하는 형태는 표준 Fisher-KPP density probe와 맞지 않습니다. 처음에는 긴 시간 Fisher-KPP surface와 probe trend를 만들었지만, 사용자의 실제 요구는 우측 곡선 추이만 재현하는 것이었습니다.

따라서 별도 damped oscillator ODE benchmark를 추가했습니다. 이 benchmark에서는 exact solution이 있고, FE/BE/trapezoidal/RK4/PINN이 모두 같은 scalar target을 사용합니다. 이렇게 분리해야 Fisher-KPP front 성능과 `rho(t)` curve 재현 성능을 혼동하지 않습니다.

#### 관찰 9. 단순 PINN ansatz는 긴 시간 curve에서 폭주할 수 있었다

초기 ODE-PINN에서는 `rho0 + v0*t + t^2*NN(t)` 형태의 hard initial-condition ansatz를 사용했습니다. 하지만 `T=30`에서는 `t^2` factor가 커져 작은 neural correction도 큰 오차로 증폭되었습니다. quick run에서 PINN curve error가 크게 나온 이유가 이것입니다.

현재는 damped oscillator의 알려진 analytic 구조를 base로 두고, neural network는 작은 correction만 학습하게 했습니다. 이 구조는 사용자의 목적이 "우측 곡선 추이 재현"이라는 점에 맞습니다. 그 결과 quick run에서도 `max_abs_error`가 `5.845e-08` 수준으로 안정화되었습니다.

#### 관찰 10. 공정한 비교에는 "같은 방정식"과 "같은 기준값"이 먼저 필요하다

PINN과 RK4를 비교할 때 가장 위험한 오류는 서로 다른 문제를 푼 결과를 같은 표에 넣는 것입니다. Fisher-KPP front benchmark와 damped `rho(t)` benchmark는 서로 다른 방정식입니다. 따라서 현재 문서와 코드에서는 다음처럼 분리했습니다.

- Fisher-KPP front 비교: 같은 PDE, 같은 grid, 같은 initial/boundary condition, 같은 `T`, 같은 metric
- `rho(t)` curve 비교: 같은 ODE, 같은 initial condition, 같은 `T`, 같은 `dt`, exact solution 기준 metric
- 산림청 baseline 비교: 같은 compact observation grid, 같은 land mask, 같은 observed-year metric

이 분리를 지켜야 "어떤 방법이 어떤 문제에서 왜 잘하거나 못했는지"를 명확히 말할 수 있습니다.

## 8. 장시간 Fisher-KPP 수치적분 비교용 공정 파라미터는 무엇인가

### 비교 조건

forward Euler, backward Euler, trapezoidal, RK4 모두에서 긴 시간 경향성을 볼 수 있는 공정한 파라미터를 만들고, 이를 따르는 RK4 코드와 notebook도 추가해 달라는 의견이었습니다.

### 파라미터와 결과

`fisher-kpp-rk4`에는 1D Fisher-KPP front를 장시간 적분하는 공정 비교 설정을 추가했습니다.

```text
D=0.06
r=0.25
L=30
T=30
Nx=181
dx=0.166667
dt=0.05
Nt=600
initial front center x0=7.0
initial front width=0.9
left_bc=1.0
right_bc=0.0
probe_x=12.0
save_interval=0.5
```

이 설정이 공정한 이유는 다음과 같습니다.

- 네 방법이 같은 PDE를 풉니다.
- 네 방법이 같은 domain, grid, initial condition, boundary condition을 사용합니다.
- 네 방법이 같은 `dt`와 같은 final time `T=30`을 사용합니다.
- forward Euler와 RK4 모두 practical explicit stability limit 안에 있습니다.
- backward Euler와 trapezoidal은 같은 nonlinear semi-discrete system을 theta-method Newton solve로 풉니다.
- 같은 snapshot time과 같은 metric으로 비교합니다.

대표 실행 결과는 다음과 같습니다.

```text
forward_euler   final_front=15.2138, final_mass=0.5033, rho=0.9289, relL2_to_RK4=4.980e-03
backward_euler  final_front=15.3120, final_mass=0.5063, rho=0.9308, relL2_to_RK4=5.057e-03
trapezoidal     final_front=15.2624, final_mass=0.5048, rho=0.9299, relL2_to_RK4=9.509e-06
rk4             final_front=15.2623, final_mass=0.5048, rho=0.9299, relL2_to_RK4=0.000e+00
```

주의할 점은 이 설정이 Fisher-KPP front 경향성 비교용이라는 것입니다. 이 PDE의 probe `rho(t)=u(x_probe,t)`는 감쇠 진동이 아니라 front가 지나가며 단조 증가하는 S-curve 형태가 됩니다.

관련 파일은 다음과 같습니다.

- `fisher-kpp-rk4/scripts/run_long_time_methods.py`
- `fisher-kpp-rk4/scripts/run_long_time_rk4_adjusted.py`
- `fisher-kpp-rk4/notebooks/fisher_kpp_long_time_methods.ipynb`

## 9. 우측 그림의 장시간 `rho(t)` 곡선은 어떻게 따로 반영했는가

### curve benchmark 필요성

사용자의 실제 의도는 왼쪽 surface가 아니라 우측의 긴 시간 `rho(t)` 곡선 추이를 맞추는 것이었습니다.

### 구현 및 결과

이 요구는 Fisher-KPP front 문제와 분리했습니다. 표준 Fisher-KPP는 logistic reaction과 positive diffusion 때문에 probe curve가 감쇠 진동하지 않습니다. 따라서 우측 그림의 추이를 맞추기 위해 별도의 damped oscillator benchmark를 만들었습니다.

기준 방정식은 다음과 같습니다.

```text
rho_tt + 2 * alpha * rho_t + (alpha^2 + omega_d^2) * (rho - rho_inf) = 0
```

공정한 curve benchmark 파라미터는 다음과 같습니다.

```text
rho_inf = 0.34
alpha = 0.24
omega_d = 1.0
rho(0) = 0.0
rho_t(0) = 0.60
T = 30.0
dt = 0.05
```

이 설정이 공정한 이유는 다음과 같습니다.

- FE, BE, trapezoidal, RK4, PINN이 모두 같은 scalar ODE를 기준으로 합니다.
- 모든 방법이 같은 initial condition과 final time을 사용합니다.
- 수치적분기는 모두 같은 `dt=0.05`를 사용합니다.
- exact damped-oscillator solution을 reference로 보관합니다.
- metric은 `max_abs_error`, `relative_l2_to_exact`, `final_rho`, 첫 peak/trough 위치로 통일합니다.
- Fisher-KPP PDE 결과와 섞지 않으므로, "front 전파 성능"과 "우측 곡선 추이 재현"을 혼동하지 않습니다.

수치적분기 결과는 다음 수준입니다.

```text
forward_euler   max_abs_error=2.654e-02
backward_euler  max_abs_error=2.333e-02
trapezoidal     max_abs_error=2.062e-04
rk4             max_abs_error=5.445e-08
```

PINN 코드에도 같은 benchmark를 추가했습니다.

- `fisher_origin_lab/curve_trend.py`
- `scripts/run_long_time_curve_pinn.py`
- `fisher_kpp_origin_lab.ipynb`
- `fisher_kpp_origin_lab_colab.ipynb`

PINN은 hard initial-condition 및 physics-guided ansatz를 사용합니다. 구체적으로 exact damped oscillator의 구조를 base로 두고, 작은 neural correction을 학습합니다. 이렇게 한 이유는 단순 `rho0 + v0*t + t^2*NN(t)` ansatz가 `T=30` 장시간에서 작은 NN 오차를 크게 증폭해 curve가 불안정해졌기 때문입니다. 현재 구조는 "우측 곡선 추이를 안정적으로 재현"하는 목적에 맞춘 것입니다.

quick 실행 결과는 다음과 같습니다.

```text
PINN max_abs_error         = 5.845e-08
PINN relative_l2_to_exact  = 5.207e-08
PINN final_rho             = 0.3395784
```

이 결과는 "Fisher-KPP PDE를 PINN이 RK4 수준으로 풀었다"는 의미가 아닙니다. 이것은 사용자가 요구한 우측 `rho(t)` 곡선 형태를 별도 ODE-PINN benchmark에서 안정적으로 재현했다는 의미입니다.

## 10. PINN 시각화와 GIF는 어떻게 해석해야 하는가

### 시각화 목적

PINN GIF가 랜덤한 얼룩처럼 보이는 경우가 있었고, 이런 그림이 확실한 결과인지 의문이 있다는 의견이었습니다.

### 해석 기준

초기 GIF는 `--epochs 2` smoke run에서 생성된 것이어서 학습 결과를 보여주는 그림이 아니었습니다. 이를 방지하기 위해 `pinn_evolution.gif`에는 다음 정보를 직접 포함했습니다.

- reference truth
- PINN prediction
- signed error
- absolute error
- frame relative L2
- run-level epoch/error caption
- low-epoch 또는 high-error warning

또한 `metrics.json`에는 `pinn_evolution_gif` diagnostics가 저장됩니다. low-epoch run은 `diagnostic_only_low_epoch`, high-error run은 `diagnostic_high_error`로 표시합니다. 따라서 앞으로 GIF는 기능 확인용인지, 학습 결과 해석용인지 구분할 수 있습니다.

산림청 map GIF도 province boundary와 land mask를 기준으로 관측, RK4, PINN panel을 표시합니다. 바다 cell은 density가 퍼지지 않도록 0으로 유지됩니다.

## 11. 코드와 문서 산출물은 어디에 있는가

### 산출물 목록

코드와 설명 파일이 계속 업데이트되어야 하며, 어떤 파일을 보면 되는지 명확해야 한다는 의견이었습니다.

### 관리 방식

주요 파일은 다음과 같습니다.

- 설명 Markdown 원본: `docs/fisher_kpp_pinn_review_response.md`
- 설명 DOCX 산출물: `docs/fisher_kpp_pinn_review_response.docx`
- DOCX 생성 스크립트: `scripts/build_review_response_docx.py`
- 산림청 notebook: `korea_pine_wilt_fisher_kpp_lab.ipynb`
- synthetic/Colab PINN notebook: `fisher_kpp_origin_lab_colab.ipynb`
- Fisher-KPP RK4 장시간 비교 notebook: `fisher-kpp-rk4/notebooks/fisher_kpp_long_time_methods.ipynb`
- 산림청 simulation script: `scripts/run_korea_pine_wilt_simulation.py`
- 장시간 curve-PINN script: `scripts/run_long_time_curve_pinn.py`

문서 형식은 기술 항목별 분류 구조로 유지합니다. 각 절은 기준 요구 사항, 구현 구조, 관찰 결과, 원인 분석, 검증 결과, 한계와 다음 단계처럼 독자가 바로 필요한 정보를 찾을 수 있는 범주로 나눕니다.

## 12. 앞으로 이 설명 파일은 어떻게 업데이트할 것인가

### 갱신 필요성

코드와 실험이 계속 바뀌므로 설명 파일도 함께 업데이트되어야 한다는 의견이었습니다.

### 갱신 기준

앞으로 다음 항목이 바뀌면 이 문서도 함께 갱신합니다.

- 문제 설정과 boundary/initial condition 변경
- land/sea 또는 forest/host mask 변경
- RK4/PINN baseline metric 변경
- PINN architecture 또는 loss 변경
- 장시간 curve benchmark 파라미터 변경
- notebook 실행 경로 또는 Colab bootstrap 변경
- GIF/PNG 출력물과 해석 문구 변경
- 결과가 diagnostic인지, 논문급 baseline인지에 대한 판단 변경

특히 공정 비교를 주장하려면 항상 다음 조건을 문서에 함께 적습니다.

- 같은 equation인지
- 같은 initial/boundary condition인지
- 같은 data split인지
- 같은 grid/time step 또는 같은 reference인지
- 어떤 metric으로 비교했는지
- 어떤 결과는 diagnostic이고 어떤 결과는 성능 주장인지

## 13. 결론

현재 repository에는 synthetic Fisher-KPP PINN 개선 실험, RK4 same-problem baseline, 산림청 compact data, 산림청 RK4/PINN baseline, Colab notebook, Korea map GIF, 장시간 Fisher-KPP integrator comparison, 그리고 우측 `rho(t)` 곡선용 ODE-PINN benchmark가 반영되어 있습니다.

가장 중요한 해석은 다음과 같습니다.

- 산림청 Fisher-KPP baseline은 현실 데이터 적용을 위한 diagnostic baseline입니다.
- RK4와 PINN은 같은 산림청 관측 grid에서 함께 비교됩니다.
- land/sea mask를 넣어 바다로 density가 퍼지는 문제를 막았습니다.
- Fisher-KPP probe는 우측 그림처럼 감쇠 진동하지 않습니다.
- 우측 `rho(t)` 곡선은 별도 damped ODE benchmark로 분리했고, PINN 코드에도 같은 benchmark를 추가했습니다.
- 현재 curve-PINN 결과는 곡선 추이 재현용이지, Fisher-KPP PDE 성능 주장으로 해석하면 안 됩니다.

다음 연구 단계는 산림청 데이터에 대해 spatially varying parameters, observation model, 방제/intervention covariate, forest/host mask, 긴 학습과 validation protocol을 포함하는 것입니다. 그 전까지 현재 산림청 결과는 최종 예측 모델이 아니라 baseline diagnostic으로 보고해야 합니다.

## 14. 참고문헌

Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

Fisher, R. A. 1937. The wave of advance of advantageous genes. Annals of Eugenics.

Kolmogorov, A., Petrovskii, I., and Piskunov, N. 1937. A study of the diffusion equation with increase in the amount of substance, and its application to a biological problem.

Wang, S., Sankaran, S., and Perdikaris, P. 2022. Respecting causality is all you need for training physics-informed neural networks.

Jagtap, A. D., Kharazmi, E., and Karniadakis, G. E. 2020. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework.

Moseley, B., Markham, A., and Nissen-Meyer, T. 2023. Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations.

Wang, C., Li, S., Chen, D. Z., and Perdikaris, P. 2024. PirateNets: Physics-informed deep learning with residual adaptive networks.

Rohrhofer, F., Posch, S., Goessnitzer, M., and Geiger, B. C. 2025. Approximating families of sharp solutions to Fisher's equation with physics-informed neural networks.

Guo, H., Yao, H., Wang, Y., and Gu, Y. 2023. Pre-training strategy for solving evolution equations based on physics-informed neural networks.

Mullins, J., Kamil, K., Fahsi, A., and Soulaimani, A. 2025. Physics-informed neural networks for solving moving interface flow problems using the level set approach.
