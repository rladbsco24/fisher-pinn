# Fisher-KPP PINN 및 한국 산림청 확장 모델 기술 설명서

작성: Codex. 업데이트: 2026년 6월 9일. 범위: 기본 Fisher-KPP PINN, Ablowitz-Zeppetella travelling-wave 검증, RK4 및 수치해석 baseline, 한국 산림청 pine-wilt 관측자료 확장, 시각화 산출물.

## 1. 문서 목적과 현재 결론

이 문서는 현재 저장소에 반영된 PINN, RK4, 한국 산림청 확장 코드를 하나의 기술 설명으로 정리한다. 이전 문서의 질문-답변식 구성과 불릿 중심 서술은 제거하고, 문제 설정, 모델 구조, 실험 protocol, 주요 관찰, 원인 분석, 산출물 관리 기준을 짧은 절 단위의 문단으로 압축했다. 본문은 최종 성능을 과장하지 않고, synthetic Fisher-KPP 검증과 실제 산림청 관측자료 baseline을 분리해서 해석한다.

현재 가장 중요한 결론은 세 가지다. 첫째, 기본 Fisher-KPP 검증은 임의 Gaussian reference가 아니라 문헌에 알려진 Ablowitz-Zeppetella travelling-wave 해를 기준으로 정리되었다. 이 regime에서 PINN full run은 1200 epoch로 수행되었고 final-time relative L2는 1.7608e-3, final-time max absolute error는 2.6630e-2였다. 같은 문제를 직접 time marching으로 푸는 RK4는 final-time relative L2 8.7850e-5, max absolute error 3.1818e-4를 보였다. 따라서 PINN은 이전의 0.06 수준 반복 오류에서 개선되었지만, 같은 PDE와 같은 해를 직접 적분하는 RK4에는 아직 미치지 못한다.

둘째, 한국 산림청 pine-wilt 코드는 실제 관측 count를 density grid로 바꾼 diagnostic forward baseline이다. 2016년 density를 초기조건으로 쓰고, 2016년부터 2023년까지의 observed-year metric으로 RK4와 PINN을 함께 비교한다. 현재 full run에서 RK4 mean relative L2는 2.5024이고 PINN mean relative L2는 0.6883이며, correlation은 RK4 0.6602, PINN 0.6985이다. 이 결과는 PINN이 관측자료에 더 유연하게 맞는다는 의미이지, 실제 확산 메커니즘을 완전히 식별했다는 의미는 아니다.

셋째, 첨부 이미지의 장시간 감쇠 진동 곡선은 Fisher-KPP density probe가 아니라 별도의 damped oscillator benchmark로 분리했다. Fisher-KPP는 양의 diffusion과 logistic reaction을 갖는 front propagation 문제이므로 일반적으로 probe curve가 감쇠 진동하지 않는다. 따라서 장시간 곡선 추세 재현은 별도 ODE-PINN benchmark에서 확인하고, Fisher-KPP front 정확도는 travelling-wave 또는 관측 grid 기준으로 평가한다.

## 2. 문제 설정과 공정 비교 기준

기본 PDE는 u_t = D Δu + r u(1-u) 형태의 Fisher-KPP 반응-확산 방정식이다. 기본 synthetic 검증은 1차원 travelling wave를 2차원 ridge 형태로 확장한 Ablowitz-Zeppetella 해를 사용한다. 물리 좌표는 x in [-20, 20], 정규화 domain은 [0, 1]^2이며, D=1/40^2=0.000625, r=1, T=10으로 맞추었다. 이 설정은 exact solution이 있어 PINN, RK4, observation fit, front metric이 모두 같은 기준에서 비교된다.

RK4와 기타 수치해석 방법의 공정 비교 기준은 같은 equation, 같은 initial condition, 같은 boundary condition, 같은 grid, 같은 time step, 같은 final time, 같은 metric을 고정하는 것이다. explicit method는 안정성 조건을 만족하는 dt를 사용하고, backward Euler와 trapezoidal은 같은 semi-discrete system을 implicit theta-method로 푼다. RK4가 매우 정확한 이유는 같은 PDE를 grid 위에서 직접 time marching하며, synthetic reference도 같은 equation에서 나온 exact 해이기 때문이다. 반대로 PINN은 continuous function approximation과 nonconvex optimization을 통해 PDE와 데이터를 동시에 만족시켜야 하므로 front 위치의 작은 phase error가 absolute error band로 증폭된다.

한국 산림청 baseline의 공정 비교 기준은 synthetic 검증과 다르다. 여기서는 exact truth가 없으므로 2016년 관측 density를 공통 초기조건으로 사용하고, 같은 land mask, 같은 observed-year metric, 같은 density normalization을 적용한다. 바다 cell은 물리적으로 확산 대상이 아니므로 RK4에서는 masked no-flux diffusion과 sea-zero projection을 쓰고, PINN에서는 sea-exclusion penalty와 land-only collocation을 둔다. 이 제약을 넣지 않으면 직사각형 grid에서 density가 바다로 퍼지는 비물리적 결과가 생긴다.

## 3. 기본 PINN 모델 구조

현재 기본 PINN은 grid를 직접 이산화해서 unknown을 두는 finite-difference 모델이 아니다. 입력은 연속 좌표 (x, y, t)이고 출력은 u(x, y, t)이다. PDE residual의 u_t, u_xx, u_yy는 PyTorch autograd로 계산하며, collocation point는 Sobol sampling, residual-adaptive refinement, front-band sampling으로 뽑는다. grid는 평가, figure, metric 계산을 위해서만 사용된다.

backbone은 PirateNet 계열 residual adaptive network, Fourier feature, random weight factorization, geo feature를 조합한다. Fisher-KPP moving front는 고주파 공간 오차보다 front phase 오차가 더 중요한 경우가 많으므로, front-aware residual weighting, front-local gPINN, causal time marching, time-slab curriculum, level-set alignment loss를 추가했다. 여기서 level-set은 PDE를 level-set 방정식으로 바꾸는 것이 아니라, reaction-diffusion 해가 만드는 front contour의 위치와 기울기를 보조 loss로 고정하는 장치다. travelling wave reference가 있는 synthetic regime에서는 이 선택이 타당하며, 실제 산림청 자료에서는 강한 analytic front target 대신 약한 물리 anchor와 관측 fit 중심으로 제한한다.

front-level-set alignment loss는 이전 area matching의 약점을 보완한다. 단순히 u>0.05 또는 u>0.10 면적만 맞추면 넓은 저농도 haze가 생겨도 metric이 좋아질 수 있다. 새 loss는 expected low-level ring 근처에서 목표 level을 맞추고, ring 안쪽과 바깥쪽의 순서 관계를 hinge로 강제하며, 약한 normal-slope 항으로 front-normal 방향의 기울기까지 맞춘다. 이 때문에 active-front area plateau, twin-trap absolute-error band, low-amplitude haze가 줄어드는 방향으로 학습 압력이 생긴다.

기본 AZ benchmark에서는 D와 r을 고정한다. 이유는 reference가 D=1, r=1인 travelling wave에서 유도되었고, 목적이 inverse coefficient 식별이 아니라 solver/PINN 구조 검증이기 때문이다. 코드 자체는 learn_diffusion, learn_reaction, spatial coefficient field를 지원하지만, D와 r을 동시에 자유롭게 열면 Fisher-KPP leading-edge speed c=2 sqrt(D r)의 비식별성이 생긴다. 따라서 계수 추정은 한국 산림청 확장처럼 물리 prior와 anchor가 있는 경우에 더 자연스럽다.

## 4. 한국 산림청 pine-wilt 확장

한국 산림청 확장은 기존 zip 코드의 원리를 이어받아 normalized coefficient와 physical coefficient를 함께 관리한다. 기존 구조에서 D_phys = D_tilde S_km^2 / ΔT, r_phys = r_tilde / ΔT로 해석하던 방식을 유지하되, 현재 코드는 length scale, grid anisotropy, land mask, coefficient anchor, optional spatial coefficient field를 추가했다. full run의 물리 prior는 D=15.5 km^2/year, r=0.70 1/year, implied front speed 6.5879 km/year이다. 정규화 scalar diffusion은 4.5646e-5이고, x방향 diffusion은 8.2659e-5, y방향 diffusion은 4.5646e-5로 기록된다.

PINN 쪽에서는 scalar diffusion과 reaction을 prior 주변에서 학습할 수 있고, spatial coefficient field가 켜지면 작은 공간 변동도 허용한다. 현재 full run에서 학습된 값은 diffusion 4.5530e-5, reaction 0.70016이며, physical scale로는 diffusion 15.4606 km^2/year, reaction 0.70016 1/year이다. 이 값들이 prior와 가까운 것은 강한 물리 식별이 완료되었다는 뜻이 아니라, physics anchor가 과도한 drift를 막고 관측자료에 맞는 작은 보정만 허용했다는 뜻이다.

실제 관측자료는 병 확산의 완전한 truth가 아니라 신고, 조사, 방제, 행정 경계, 조사 강도 변화가 섞인 surveillance count다. 따라서 단일 Fisher-KPP PDE만으로 모든 observed-year field를 설명하기 어렵다. 현재 산림청 결과에서 PINN은 RK4보다 relative L2와 mass error를 줄였지만 support false negative가 여전히 높다. 이는 모델이 관측된 고농도 핵심 영역은 맞추되, 약한 외곽 support를 충분히 넓히지 못한다는 신호다. 다음 단계에서는 host forest mask, intervention covariate, observation model, spatially varying D/r의 더 엄밀한 검증이 필요하다.

## 5. 주요 관찰과 원인 분석

가장 반복적으로 나타난 현상은 PINN absolute error가 front 주변에 ring 또는 twin-trap 형태로 나타나는 것이다. 이는 대부분 amplitude error가 아니라 front-location 또는 phase error다. sigmoid형 travelling front에서는 위치가 한두 grid cell만 어긋나도 truth의 steep transition과 PINN transition이 서로 다른 위치에서 생기며, absolute error map에는 안쪽과 바깥쪽에 두 개의 띠가 생긴다. signed error를 함께 보면 어느 방향으로 front가 밀렸는지 확인할 수 있고, absolute error만 보면 오류가 더 기하학적으로 보인다.

이전의 0.06 수준 max error가 반복된 이유는 두 가지로 정리된다. 하나는 figure cache와 notebook path가 오래된 run을 가리켜 같은 error map을 계속 보여준 문제였고, 다른 하나는 실제로 moving front phase가 조금 어긋날 때 max absolute error가 비슷한 값으로 포화되는 문제였다. 현재 full run figure는 metrics.json과 같은 run directory에서 생성되며, 기본 AZ regime의 최신 final-time max absolute error는 0.02663이다. 따라서 현재 기본 regime은 더 이상 0.06이라고 보는 것이 맞지 않다.

RK4와 PINN 사이의 차이가 큰 이유는 방법론적이다. RK4는 exact PDE를 time step마다 직접 적분하고 boundary condition도 정확히 적용한다. PINN은 모든 시간과 공간을 하나의 neural function으로 압축해 표현하며, data loss, PDE residual, boundary loss, initial-condition loss, front loss의 scale을 동시에 조율해야 한다. 이 구조는 sparse data나 inverse 문제에는 장점이 있지만, exact forward benchmark에서 고해상도 수치해석 solver를 절대오차 기준으로 따라잡기는 어렵다. 따라서 논문 baseline급 비교에서는 PINN의 장점을 inverse setting, sparse observation, irregular domain, coefficient learning, differentiable surrogate 관점에서 제시하고, forward accuracy는 RK4 같은 classical solver와 별도로 보고해야 한다.

산림청 결과에서 PINN이 RK4보다 낮은 relative L2를 보이는 이유도 같은 맥락이다. RK4는 2016년 초기조건과 고정 D/r로 순수 forward propagation을 한다. PINN은 observed-year data를 supervised term으로 직접 본다. 따라서 산림청 baseline에서 PINN의 우위는 “관측자료를 포함한 neural PDE fit이 순수 forward RK4보다 관측 grid에 잘 맞는다”는 뜻이지, RK4보다 Fisher-KPP PDE를 더 정확히 푼다는 뜻이 아니다.

## 6. 산출물과 검증 상태

기본 PINN full run 산출물은 runs/ablowitz_zeppetella_pinn 아래에 있다. 핵심 figure는 reconstruction.png, spacetime_error.png, residual_front_diagnostics.png, pinn_vs_rk4_comparison.png, training_diagnostics.png, pinn_evolution.gif, pinn_error.gif이다. metrics.json에는 full run epoch 1200, validation observation MSE 7.7896e-6, PINN final-time relative L2 1.7608e-3, PINN max absolute error 2.6630e-2, RK4 final-time relative L2 8.7850e-5, RK4 max absolute error 3.1818e-4가 기록되어 있다.

한국 산림청 full run 산출물은 runs/korea_pine_wilt_full 아래에 있다. 주요 figure는 observed_density_by_year.png, rk4_forecast_timeline.png, observed_vs_simulated_metrics.png, baseline_metric_comparison.png, pinn_baseline_observed_years.png, korea_map_baselines.gif, korea_error_baselines.gif이다. summary JSON에는 compact data record 3,183,376개, land cell 3,609개, sea cell 5,607개, grid size 96, end year 2030, PINN epoch 1200이 기록된다.

Notebook은 Colab 실행을 고려해 구성되어 있다. fisher_kpp_origin_lab_colab.ipynb는 기본 Fisher-KPP/AZ 검증을 실행하고, korea_pine_wilt_fisher_kpp_lab.ipynb는 Colab에서 GitHub repository를 clone 또는 refresh한 뒤 compact data manifest를 확인하고 산림청 simulation을 수행한다. fisher-kpp-rk4/notebooks/fisher_kpp_long_time_methods.ipynb는 forward Euler, backward Euler, trapezoidal, RK4의 장시간 수치해석 비교를 다룬다. 모든 notebook 설명은 깨진 한글 문자열을 피하고, Colab path와 local path를 모두 처리하도록 맞추었다.

## 7. 해석 기준과 다음 개선 방향

현재 저장소는 논문 제출용 최종 모델이라기보다 논문 baseline 수준으로 가기 위한 재현 가능한 실험 bundle에 가깝다. 기본 AZ benchmark는 exact reference를 통해 PINN 구조와 front loss가 정상 동작하는지 확인하는 역할을 한다. 한국 산림청 확장은 실제 관측자료에서 land mask, physical scaling, D/r prior, PINN baseline, RK4 baseline, GIF 시각화를 한 workflow로 묶은 진단용 pipeline이다.

다음 개선의 우선순위는 관측 모델과 계수 식별성이다. Fisher-KPP front speed는 leading edge에서 2 sqrt(D r)에 의해 크게 결정되므로, front 위치만으로는 D와 r을 안정적으로 분리하기 어렵다. 따라서 시간별 total mass, local growth, host density, 방제 intervention, 관측 강도 차이를 함께 모델링해야 한다. PINN 구조 측면에서는 full XPINN/FBPINN domain decomposition, mixture-of-experts front model, monotone-front 또는 positivity-preserving output transform, Bayesian 또는 ensemble uncertainty가 다음 후보가 된다.

성능을 보고할 때는 세 가지 문장을 분리해야 한다. 기본 Fisher-KPP exact benchmark에서 PINN의 forward solver accuracy는 RK4보다 낮다. 산림청 observed-year baseline에서는 PINN이 RK4보다 관측 grid에 더 잘 맞지만 diagnostic baseline이다. 장시간 감쇠 진동 곡선은 Fisher-KPP PDE가 아니라 별도 scalar ODE benchmark에서 검증된 결과다. 이 구분을 지키면 figure, metric, 문헌 claim 사이의 충돌을 피할 수 있다.

## 8. 참고문헌

Fisher, R. A. 1937. “The Wave of Advance of Advantageous Genes.” Annals of Eugenics 7 (4): 355-369. https://doi.org/10.1111/j.1469-1809.1937.tb02153.x.

Kolmogorov, A. N., I. G. Petrovskii, and N. S. Piskunov. 1937. “A Study of the Equation of Diffusion with Increase in the Quantity of Matter, and Its Application to a Biological Problem.” Moscow University Bulletin of Mathematics 1: 1-25.

Ablowitz, Mark J., and Anthony Zeppetella. 1979. “Explicit Solutions of Fisher’s Equation for a Special Wave Speed.” Bulletin of Mathematical Biology 41: 835-840. https://doi.org/10.1007/BF02462380.

Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. 2019. “Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.” Journal of Computational Physics 378: 686-707. https://doi.org/10.1016/j.jcp.2018.10.045.

Karniadakis, George Em, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. 2021. “Physics-Informed Machine Learning.” Nature Reviews Physics 3: 422-440. https://doi.org/10.1038/s42254-021-00314-5.

Wang, Sifan, Yujun Teng, and Paris Perdikaris. 2021. “Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks.” SIAM Journal on Scientific Computing 43 (5): A3055-A3081. https://doi.org/10.1137/20M1318043.

Wang, Sifan, Xinling Yu, and Paris Perdikaris. 2022. “When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective.” Journal of Computational Physics 449: 110768. https://doi.org/10.1016/j.jcp.2021.110768.

Krishnapriyan, Aditi S., Amir Gholami, Shandian Zhe, Robert M. Kirby, and Michael W. Mahoney. 2021. “Characterizing Possible Failure Modes in Physics-Informed Neural Networks.” In Advances in Neural Information Processing Systems 34, 26548-26560.

Yu, Jeremy, Lu Lu, Xuhui Meng, and George Em Karniadakis. 2022. “Gradient-Enhanced Physics-Informed Neural Networks for Forward and Inverse PDE Problems.” Computer Methods in Applied Mechanics and Engineering 393: 114823. https://doi.org/10.1016/j.cma.2022.114823.

Jagtap, Ameya D., and George Em Karniadakis. 2020. “Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations.” Communications in Computational Physics 28 (5): 2002-2041. https://doi.org/10.4208/cicp.OA-2020-0164.

Moseley, Ben, Andrew Markham, and Tarje Nissen-Meyer. 2023. “Finite Basis Physics-Informed Neural Networks (FBPINNs): A Scalable Domain Decomposition Approach for Solving Differential Equations.” Advances in Computational Mathematics 49: 62. https://doi.org/10.1007/s10444-023-10065-9.

Wang, Sifan, Bowen Li, Yuhan Chen, and Paris Perdikaris. 2024. “PirateNets: Physics-Informed Deep Learning with Residual Adaptive Networks.” Journal of Machine Learning Research 25 (402): 1-51. https://jmlr.org/papers/v25/24-0313.html.

Yang, Yu, Qihong Yang, Yangtao Deng, and Qiaolin He. 2024. “Moving Sampling Physics-Informed Neural Networks Induced by Moving Mesh PDE.” Neural Networks 180: 106706. https://doi.org/10.1016/j.neunet.2024.106706.

Rohrhofer, Franz M., Stefan Posch, Clemens Goessnitzer, and Bernhard C. Geiger. 2025. “Approximating Families of Sharp Solutions to Fisher’s Equation with Physics-Informed Neural Networks.” Computer Physics Communications 307: 109422. https://doi.org/10.1016/j.cpc.2024.109422.
