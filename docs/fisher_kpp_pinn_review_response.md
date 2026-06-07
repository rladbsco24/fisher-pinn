# Fisher-KPP PINN 산림청 적용 검토의견 회신

작성: Codex / 연구 기록 초안

일자: 2026년 6월 7일

## 개론

현재 파이프라인의 핵심은 Fisher-KPP 반응-확산 방정식을 기준 문제로 두고, 같은 문제 설정에서 numerical solver(RK4)와 우리가 구성한 PINN baseline을 함께 비교하는 것입니다. 초기 synthetic inverse-origin 실험에서는 PINN의 이동 front 재현, known initial condition, causal/time-slab curriculum, front-aware residual sampling, RK4 teacher regularization 등을 검토했습니다. 이후 한국 산림청 소나무재선충병 관측 좌표 데이터에 대해 compact observation dataset을 만들고, 이를 연도별 density grid로 변환하여 같은 forward Fisher-KPP 문제의 baseline 평가로 확장했습니다.

이 문서는 현재 코드와 노트북의 의도, 반영된 방법론, baseline 구성, 산림청 데이터 처리 방식, 그리고 아직 남아 있는 한계를 검토의견 회신서 형식으로 정리한 설명 파일입니다. 이후 실험 결과나 코드 구조가 바뀔 때는 이 문서를 함께 업데이트하는 것을 기준으로 둡니다.

## 1. 문제 설정은 무엇으로 고정했는가

### 말씀하신 내용 요약

문제 구성은 기존 한국 산림청 코드와 동일하게 유지하되, 모델 자체는 개선할 수 있어야 한다는 의견입니다. 특히 boundary condition, initial condition, forward simulation, 산림청 관측 데이터와의 대응 관계가 불분명하면 결과 비교가 의미를 잃을 수 있습니다.

### 답변

현재 산림청 적용 단계에서는 문제를 forward Fisher-KPP baseline으로 정리했습니다. 상태 변수 `u(x, y, t)`는 정규화된 감염 밀도 또는 관측 density proxy이고, 지배 방정식은 다음 구조입니다.

```text
u_t = D * Laplacian(u) + r * u * (1 - u)
```

공간 domain은 산림청 관측 좌표의 bounding box를 정규화한 2D grid로 사용합니다. RK4 baseline은 2016년 관측 density grid를 초기조건으로 두고 no-flux에 가까운 Neumann boundary 처리를 적용하여 2030년까지 전진 적분합니다. PINN baseline은 같은 grid observation을 사용하되, 관측 연도 density에 대한 data loss와 Fisher-KPP PDE residual, 약한 Neumann boundary loss를 함께 사용합니다.

기존 inverse-origin PINN 실험과 산림청 forward baseline은 목적이 다릅니다. inverse-origin 실험은 synthetic truth가 있고 origin recovery를 평가합니다. 산림청 baseline은 raw surveillance count에서 만든 yearly density grid를 대상으로 RK4와 PINN이 같은 관측 연도에 얼마나 맞는지 비교하는 진단용 forward baseline입니다.

## 2. 왜 RK4와 PINN을 모두 baseline에 넣었는가

### 말씀하신 내용 요약

산림청 데이터 실험에서 RK4만 제시하면 우리가 만든 PINN이 실제 baseline 비교에 들어가지 않으므로, PINN도 같은 평가 표에 포함해야 한다는 의견입니다.

### 답변

맞습니다. 현재 코드에는 산림청 전용 baseline으로 RK4와 PINN을 모두 포함했습니다.

RK4는 같은 Fisher-KPP 방정식을 grid solver로 직접 푸는 numerical baseline입니다. 2016년 관측 density를 초기조건으로 두기 때문에 2016년에는 관측과 완전히 일치하지만, 고정된 `D`, `r`로 이후 연도를 예측하면서 2020년 이후 과성장하는 경향이 나타납니다.

PINN baseline은 repository의 PirateNet/Fourier/geo-feature 계열 PINN backbone을 재사용합니다. 산림청 yearly density grid를 supervised observation으로 fit하고, 동시에 약한 Fisher-KPP residual과 Neumann boundary penalty를 둡니다. 이 baseline은 raw surveillance data를 완전히 설명하는 최종 모델이 아니라, 같은 데이터에서 neural PDE model이 RK4 대비 어느 정도 fitting flexibility를 갖는지 확인하는 진단용 비교입니다.

현재 실행 결과에서 RK4의 observed-year mean relative L2는 `3.9009`, PINN의 observed-year mean relative L2는 `1.1426`입니다. PINN이 RK4보다 관측 연도 L2는 낮췄지만, status는 `diagnostic_high_error`로 표시됩니다. 즉 PINN baseline을 추가했지만, 아직 논문급 최종 성능이라고 주장할 수준은 아닙니다.

## 3. Geo-aware forward PINN 구조는 어떻게 반영했는가

### 말씀하신 내용 요약

Geo-aware forward PINN 계획과 구조를 코드에 반영해야 하며, 단순 MLP가 아니라 최신 PINN 안정화 기법과 moving front 특성을 고려해야 한다는 의견입니다.

### 답변

synthetic Fisher-KPP forward profile에는 다음 구성요소가 반영되어 있습니다.

- PirateNet-style residual architecture와 random weight factorization
- spatial Fourier features
- square-domain geo features
- hard known initial condition ansatz
- traveling-wave/front coordinate features
- KPP front envelope
- front-aware PDE residual weighting
- front-local gPINN loss
- moving-front speed consistency loss
- parabolic mass-balance loss
- leading-edge area and front-profile losses
- residual-adaptive collocation
- causal/time-slab curriculum
- weak RK4 teacher ablation

산림청 baseline에서는 전체 synthetic stabilizer를 그대로 모두 켜지는 않았습니다. 산림청 raw observation은 실제 감염 과정뿐 아니라 탐지, 방제, 신고, 정책 변화가 섞인 surveillance count이므로, analytic initial seed나 정확한 moving-front level-set target을 강하게 넣으면 오히려 잘못된 물리 가정을 강요할 수 있습니다. 따라서 산림청 PINN baseline은 data fitting, weak PDE residual, weak Neumann boundary를 중심으로 둔 diagnostic baseline으로 제한했습니다.

## 4. 산림청 데이터는 어떻게 반영했는가

### 말씀하신 내용 요약

한국 산림청 CSV와 시뮬레이션도 코드와 노트북에 커밋되어야 하며, Colab에서도 돌아가야 한다는 의견입니다.

### 답변

원본 연도별 CSV는 총 약 1.17 GB이고 일부 파일이 GitHub 일반 blob 한도 100 MB를 넘습니다. 따라서 원본 CSV 전체를 일반 Git 파일로 커밋하지 않고, 다음 compact artifact를 커밋했습니다.

- `data/korea_pine_wilt/processed/infected_points_2016_2023.csv.gz`
- `data/korea_pine_wilt/processed/infected_points_2016_2023.npz`
- `data/korea_pine_wilt/processed/manifest.json`
- `data/korea_pine_wilt/assets/skorea_provinces_2018.geojson`

compact file에는 `x_5179`, `y_5179`, `year`가 들어 있으며 총 record 수는 `3,183,376`입니다. `manifest.json`에는 원본 CSV의 연도, 파일 크기, sha256 checksum이 들어 있어 원자료 검증이 가능합니다. 또한 `scripts/build_korea_pine_wilt_compact_data.py`를 통해 원본 CSV가 있는 환경에서는 compact data를 다시 생성할 수 있습니다.

Colab 노트북 `korea_pine_wilt_fisher_kpp_lab.ipynb`는 project files가 없으면 GitHub repository를 clone하고, 이미 `/content/fisher-pinn`이 있으면 `fetch` 후 `FETCH_HEAD`로 강제 checkout하여 오래된 clone을 재사용하지 않도록 수정했습니다.

## 5. 산림청 노트북은 무엇을 실행하는가

### 말씀하신 내용 요약

산림청 데이터를 위한 `.ipynb`도 실제 실행 가능해야 하며, 설명과 출력이 깨지지 않아야 한다는 의견입니다.

### 답변

현재 산림청 노트북은 다음 순서로 실행됩니다.

1. repository root 또는 Colab clone 준비
2. compact data manifest 검증
3. NPZ observation point load
4. yearly density grid 생성
5. RK4 baseline 실행
6. PINN baseline 학습 및 예측
7. observed-year metric table 출력
8. observation, RK4 forecast, PINN baseline, baseline metric comparison PNG 표시

노트북 설명에서 한글이 `???`로 깨졌던 문제는 제거했고, 현재는 ASCII/영문 설명으로 안정화했습니다. smoke test에도 `???` 손상 문자열을 감지하는 조건을 추가했습니다.

## 6. 현재 산림청 baseline 결과는 어떻게 해석해야 하는가

### 말씀하신 내용 요약

실행 결과를 보여주되, 좋은 결과인지 아닌지 명확하게 판단해야 한다는 의견입니다. 특히 단순 smoke run이나 diagnostic output을 최종 성능처럼 보여서는 안 됩니다.

### 답변

현재 산림청 notebook 실행 결과는 다음과 같습니다.

```text
RK4 mean relative L2      = 3.9009
PINN mean relative L2     = 1.1426
RK4 mean correlation      = 0.4348
PINN mean correlation     = 0.4487
PINN status               = diagnostic_high_error
```

PINN은 RK4보다 observed-year relative L2를 낮췄습니다. 하지만 `diagnostic_high_error` 상태이므로, 이 결과는 최종 예측 모델의 성능이 아니라 baseline diagnostic으로 해석해야 합니다. 특히 2016년 관측 density를 초기조건으로 한 fixed-parameter RK4는 후반부에 감염 density가 과성장하고, PINN은 관측 연도 supervised fitting으로 L2를 낮추지만 spatial correlation 개선 폭은 제한적입니다.

이 현상은 Fisher-KPP 하나로 raw surveillance count를 설명하기 어렵기 때문입니다. 실제 산림청 데이터에는 감염 확산 외에도 탐지 강도, 신고 체계, 방제 intervention, 지역별 sampling bias, 행정 정책 변화, 지형·산림 mask, 숙주 분포가 섞여 있습니다. 따라서 다음 단계에서는 spatially varying reaction/diffusion, observation model, intervention covariate, land/forest mask를 모델에 넣어야 합니다.

## 7. PINN 시각화와 GIF는 어떻게 해석해야 하는가

### 말씀하신 내용 요약

PINN GIF가 랜덤 얼룩처럼 보이는 경우가 있었고, 이런 그림이 확실한 결과인지 의문이 있다는 의견입니다.

### 답변

초기 GIF는 `--epochs 2` smoke run에서 생성된 것이어서 학습 품질을 보여주는 결과가 아니었습니다. 이를 방지하기 위해 `pinn_evolution.gif`는 이제 다음 정보를 직접 포함합니다.

- reference truth
- PINN prediction
- signed error
- absolute error
- frame relative L2
- run-level epoch/error caption
- low-epoch 또는 high-error warning

또한 `metrics.json`에는 `pinn_evolution_gif` diagnostics가 저장됩니다. low-epoch run은 `diagnostic_only_low_epoch`, high-error run은 `diagnostic_high_error`로 표시됩니다. 따라서 앞으로 GIF는 기능 확인용인지, 학습 결과 해석용인지 구분할 수 있습니다.

## 8. 앞으로 이 설명 파일은 어떻게 업데이트할 것인가

### 말씀하신 내용 요약

코드와 실험이 계속 바뀌므로 설명 파일도 같이 업데이트되어야 한다는 의견입니다. 양식은 첨부 회신서 형식을 따르는 것이 좋다는 의견입니다.

### 답변

이 설명 파일의 Markdown 원본은 `docs/fisher_kpp_pinn_review_response.md`입니다. DOCX 산출물은 `docs/fisher_kpp_pinn_review_response.docx`입니다. 이후 코드나 결과가 바뀔 때는 다음 항목을 함께 갱신합니다.

- 문제 설정 변화
- 새 loss 또는 architecture 반영 여부
- RK4/PINN baseline metric
- 산림청 compact data 또는 preprocessing 변화
- 노트북 실행 결과
- 결과 해석에서 과장될 수 있는 부분
- 남은 한계와 다음 개선 항목

문서 형식은 참고 회신서처럼 각 항목을 `말씀하신 내용 요약`과 `답변`으로 나누어 유지합니다. 이 방식은 검토의견별로 어떤 코드를 왜 바꿨고, 어떤 결과가 나왔으며, 아직 무엇이 부족한지를 추적하기 쉽습니다.

## 9. 결론

현재 repository에는 synthetic Fisher-KPP PINN 개선 실험, RK4 same-problem baseline, 산림청 compact data, 산림청 RK4/PINN baseline, Colab notebook, GIF diagnostics가 반영되어 있습니다. 산림청 노트북은 실제 실행 가능하고, PINN baseline도 RK4와 같은 metric table에 포함됩니다.

다만 현재 산림청 결과는 여전히 diagnostic baseline입니다. PINN이 RK4보다 L2를 낮추긴 했지만 high-error 상태이고, raw surveillance count를 Fisher-KPP PDE만으로 설명하기에는 관측 bias와 intervention factor가 큽니다. 다음 개선은 산림/지형 mask, spatially varying parameters, observation model, control/intervention covariates, 더 긴 PINN training 및 validation split을 포함해야 합니다.

## 10. 참고문헌

Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

Fisher, R. A. 1937. The wave of advance of advantageous genes. Annals of Eugenics.

Kolmogorov, A., Petrovskii, I., and Piskunov, N. 1937. A study of the diffusion equation with increase in the amount of substance, and its application to a biological problem.

Wang, S., Sankaran, S., and Perdikaris, P. 2022. Respecting causality is all you need for training physics-informed neural networks.

Jagtap, A. D., Kharazmi, E., and Karniadakis, G. E. 2020. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework.

Moseley, B., Markham, A., and Nissen-Meyer, T. 2023. Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations.

Wang, C., Li, S., Chen, D. Z., and Perdikaris, P. 2024. PirateNets: Physics-informed deep learning with residual adaptive networks.

Rohrhofer, F., Posch, S., Gößnitzer, M., and Geiger, B. C. 2025. Approximating families of sharp solutions to Fisher's equation with physics-informed neural networks.

Guo, H., Yao, H., Wang, Y., and Gu, Y. 2023. Pre-training strategy for solving evolution equations based on physics-informed neural networks.

Mullins, J., Kamil, K., Fahsi, A., and Soulaimani, A. 2025. Physics-informed neural networks for solving moving interface flow problems using the level set approach.
