# TabPFN 스터디 노트

Claude Code와 나눈 대화를 정리한 문서입니다. (2026-10-29 ~ 2026-11-03)

- 논문: Hollmann et al., *"Accurate predictions on small data with a tabular foundation model"*, **Nature 637, 2025** (CC BY 4.0)
- 코드: Prior Labs `tabpfn` 저장소 (Apache 2.0). 로컬 설치 버전 `tabpfn 9.2.0`

목차
1. [교수님 지시사항](#1-교수님-지시사항)
2. [핵심 개념](#2-핵심-개념)
3. [코드 구조](#3-코드-구조)
4. [실행 흐름: fit → predict](#4-실행-흐름-fit--predict)
5. [전처리와 앙상블](#5-전처리와-앙상블)
6. [모델 아키텍처 (v2, v3.5)](#6-모델-아키텍처-v2-v35)
7. [회귀: bar distribution](#7-회귀-bar-distribution)
8. [부가 기능](#8-부가-기능)
9. [실행 환경 설정](#9-실행-환경-설정)
10. [논문 원문: 향후 연구 방향과 한계](#10-논문-원문-향후-연구-방향과-한계)
11. [실험 계획](#11-실험-계획)
12. [직접 구현 로드맵](#12-직접-구현-로드맵)
13. [코드 공부 순서와 디버깅 방법](#13-코드-공부-순서와-디버깅-방법)
14. [원격 접속 (아이패드)](#14-원격-접속-아이패드)
15. [할 일](#15-할-일)

---

## 1. 교수님 지시사항

1. TabPFN 논문의 **기본적인 코드를 직접 구현**해 볼 것
2. 저자가 말한 **Potential Future Direction**을 보고, 무엇을 더 해 보며 성능을 테스트할지 고민할 것
3. **직접 고른 주요 데이터**(구매 데이터, 공정 설비 데이터 등)로 실험할 것

---

## 2. 핵심 개념

TabPFN은 **Prior-Data Fitted Network(PFN)** 입니다.

- 수많은 **합성 표 데이터셋**으로 Transformer를 **미리 한 번만** 학습합니다.
- 새 데이터에서는 학습(경사하강)을 하지 않습니다. **train 데이터 전체를 입력(컨텍스트)으로 넣고**, test 행의 예측을 **forward 한 번**으로 냅니다. LLM의 in-context learning(ICL)과 같은 방식입니다.
- 이론적으로는 합성 데이터 분포(prior)에 대한 **베이즈 사후 예측분포**를 근사합니다 (Müller et al., 2022).

그래서 코드에서 `fit()`은 학습하지 않습니다. 하는 일은 다음 세 가지입니다.
- 사전학습된 체크포인트 불러오기
- 데이터 전처리
- (선택) train 행의 KV 캐시 만들기

실제 계산은 `predict()`에서 `[train 행 + test 행]`을 함께 모델에 넣을 때 일어납니다.

---

## 3. 코드 구조

| 경로 | 역할 |
|---|---|
| [classifier.py](../src/tabpfn/classifier.py), [regressor.py](../src/tabpfn/regressor.py) | 사용자 API. sklearn 스타일 `TabPFNClassifier`, `TabPFNRegressor` |
| [base.py](../src/tabpfn/base.py) | 공통 초기화: 모델 로딩, 정밀도 결정, 추론 엔진 생성 |
| [model_loading.py](../src/tabpfn/model_loading.py) | HuggingFace 다운로드, 캐시, 모델 빌드, fitted 모델 저장/로드 |
| [constants.py](../src/tabpfn/constants.py) | `ModelVersion` (v2, v2.5, v2.6, v3, v3.5, v3.5-fast) 등 상수 |
| [inference_config.py](../src/tabpfn/inference_config.py) | 버전별 추론 하이퍼파라미터 |
| [inference.py](../src/tabpfn/inference.py) | **추론 엔진**: 앙상블 멤버마다 모델 호출 |
| [preprocessing/](../src/tabpfn/preprocessing/) | 데이터 정제, 타입 감지, 앙상블별 전처리 (CPU용 sklearn, GPU용 torch) |
| [architectures/](../src/tabpfn/architectures/) | PyTorch 모델. 버전마다 단일 파일 |
| [architectures/shared/](../src/tabpfn/architectures/shared/) | attention 백엔드, 회귀용 `BarDistribution`, 청크 평가 등 |
| [inference_tuning.py](../src/tabpfn/inference_tuning.py) | softmax temperature, 결정 임계값 튜닝 |
| [downsample_correction.py](../src/tabpfn/downsample_correction.py) | 다수 클래스 다운샘플링으로 생긴 prior shift 보정 |
| [finetuning/](../src/tabpfn/finetuning/) | 특정 데이터셋에 맞춘 파인튜닝 (역전파) |
| [examples/](../examples/) | 사용 예제 |

> **중요**: 이 저장소에는 **추론 코드만** 있습니다. 합성 데이터 생성기(prior)와 사전학습 루프는 공개되어 있지 않습니다. 직접 구현 과제에서 이 두 부분을 만들어야 합니다.

---

## 4. 실행 흐름: fit → predict

분류기 기준입니다.

### `fit()` — [classifier.py:959](../src/tabpfn/classifier.py#L959)
1. `_initialize_model_variables()`: 체크포인트 로드, 디바이스(CUDA/MPS/CPU)와 정밀도 결정
2. `_initialize_dataset_preprocessing()`: 입력 검증, 날짜/텍스트 확장, 범주형 추론, 라벨 인코딩, **앙상블 설정 생성**
3. `TabPFNEnsemblePreprocessor`: 멤버마다 전처리 파이프라인 fit
4. `_compute_downsample_correction()`: 샘플을 줄였다면 클래스 prior 보정 가중치 계산
5. `_maybe_calibrate_temperature_and_tune_decision_thresholds()`: `tuning_config`가 있으면 holdout으로 튜닝
6. `create_inference_engine()` — [base.py:317](../src/tabpfn/base.py#L317): `fit_mode`에 맞는 엔진 생성

### `predict_proba()` → `_raw_predict()` → `forward()`
1. 테스트 X에 fit 때와 같은 정제 적용
2. `executor_.iter_outputs(X)`: 앙상블 멤버마다 `[train; test]`를 모델에 넣어 logit 생성
3. 멤버마다 섞었던 **클래스 순서를 원래대로 복원** — [classifier.py:2035 부근](../src/tabpfn/classifier.py#L2035)
4. `logits_to_probabilities()` — [classifier.py:1831](../src/tabpfn/classifier.py#L1831): temperature → softmax와 멤버 평균 → downsample 보정 → (선택) 클래스 균형화
5. `predict()`는 argmax 후 라벨을 원래 값으로 복원

---

## 5. 전처리와 앙상블

모델 하나를 `n_estimators`번 서로 다른 "관점"으로 돌려 평균합니다. 멤버마다 달라지는 것은 다음과 같습니다 — [ensemble.py:1307](../src/tabpfn/preprocessing/ensemble.py#L1307).

- **feature shift**: 컬럼 순서 회전/셔플. 위치 의존성 제거
- **class permutation**: 클래스 번호 셔플. 인덱스 편향 제거
- **preprocessor config**: 분포 변환 방식(quantile, power, squashing scaler 등) 교대
- **feature/sample subsampling**: 피처나 행이 많을 때 부분집합 사용 (LightGBM 중요도 기반 선택 포함)

멤버 하나의 파이프라인 — [pipeline_factory.py](../src/tabpfn/preprocessing/pipeline_factory.py):

```
(다항 피처) → 상수 피처 제거 → 분포 재형성(Quantile 등) → 범주형 인코딩
→ (SVD 피처 추가) → (fingerprint 피처) → 피처 셔플
```

### 추론 엔진 (`fit_mode`) — [inference.py](../src/tabpfn/inference.py)

| fit_mode | 엔진 | 특징 |
|---|---|---|
| `low_memory` | `InferenceEngineOnDemand` | 예측할 때마다 전처리. 메모리 최소 |
| `fit_preprocessors` (기본) | `InferenceEngineCachePreprocessing` | 전처리 결과 캐시 |
| `fit_with_cache` | `InferenceEngineExplicitKVCache` | train 행의 attention K/V를 미리 계산. 반복 예측이 빠름 |
| `batched` | `InferenceEngineBatchedNoPreprocessing` | 파인튜닝용. 그래디언트가 흐름 |

---

## 6. 모델 아키텍처 (v2, v3.5)

모든 버전은 [interface.py](../src/tabpfn/architectures/interface.py)의 `Architecture` 인터페이스를 따르고, [architectures/\_\_init\_\_.py](../src/tabpfn/architectures/__init__.py)의 `ARCHITECTURES`에 등록됩니다.

### TabPFN v2 — [tabpfn_v2.py](../src/tabpfn/architectures/tabpfn_v2.py) (논문 모델, 공부 기준)

입력 텐서는 `(Batch, Rows, Columns, Embedding)` 형태의 **셀 단위 2D 격자**입니다. 기본값: emsize 192, 12층, 6 heads.

1. **임베딩**: 피처를 2개씩 묶어 정규화. NaN/Inf는 평균으로 채우고 지시자(-2, 2, 4)를 붙여 선형 투영. 타깃 y도 **추가 컬럼 하나**로 임베딩 (test 행의 y 칸은 빈 임베딩)
2. **`TabPFNBlock` × 12** — [tabpfn_v2.py:321](../src/tabpfn/architectures/tabpfn_v2.py#L321)
   - **AlongRowAttention**: 한 행 안 피처끼리 attention
   - **AlongColumnAttention** — [tabpfn_v2.py:152](../src/tabpfn/architectures/tabpfn_v2.py#L152): 한 컬럼 안 행끼리 attention. **train 행은 train 행만, test 행도 train 행만 봄.** test끼리는 서로 안 봄 (ICL의 핵심, 마스크 없이 슬라이싱으로 구현). test 행은 KV head 1개만 쓰는 multi-query attention
   - MLP. 세 서브블록 모두 post-LayerNorm
3. **디코딩**: test 행의 타깃 컬럼 임베딩 → 출력 head → logit

### TabPFN v3.5 — [tabpfn_v3_5.py](../src/tabpfn/architectures/tabpfn_v3_5.py) (현재 기본값)

셀 격자 전체에 attention을 거는 v2의 비용 문제를, **단계를 나눠 행 단위 토큰으로 압축**해서 해결합니다.

- **Stage 0, 셀 임베딩**: Fourier feature + 컬럼 내 **ECDF 순위**. 피처 3개씩 그룹
- **Stage 1, `FeatureDistributionEmbedder`**: 컬럼마다 induced self-attention(inducing point 128개)으로 train 분포 요약
- **Stage 2, `ColumnAggregator`**: CLS 토큰 8개 + Transformer(RoPE)로 **행 하나를 고정 크기 벡터로 압축** (컬럼 수와 무관)
- **Stage 3, ICL Transformer (24층)**: 행 토큰끼리 attention. train 행에 y 임베딩 더함, test 행은 train 행만 봄
- **출력 head**: 분류는 `ManyClassDecoder` (test가 train 임베딩에 attention해 라벨을 모아 옴), 회귀는 MLP로 bucket logit
- 같은 인스턴스가 `task_type`으로 분류와 회귀를 모두 처리

---

## 7. 회귀: bar distribution

회귀는 값을 직접 내지 않고 **y 범위를 수천 개 bucket으로 나눈 분류 문제**로 풉니다 — [bar_distribution.py](../src/tabpfn/architectures/shared/bar_distribution.py).

- bucket마다 logit → 히스토그램 형태의 확률분포
- `FullSupportBarDistribution`: 양 끝에 half-normal 꼬리를 붙여 실수 전체 지원
- y는 z-정규화 공간에서 예측 후 원래 스케일로 복원
- `predict(output_type=...)` — [regressor.py:1426](../src/tabpfn/regressor.py#L1426): `mean`, `median`, `mode`, `quantiles`, `full`. **불확실성 추정이 기본 제공**

---

## 8. 부가 기능

- **Inference tuning** — [inference_tuning.py](../src/tabpfn/inference_tuning.py): K-fold holdout으로 temperature와 이진 분류 임계값을 `eval_metric`에 맞춰 최적화
- **Downsample correction** — [downsample_correction.py](../src/tabpfn/downsample_correction.py): 다수 클래스를 줄여 컨텍스트에 넣으면 소수 클래스를 과대평가하므로, label-shift 가정 아래 prior 비율로 보정
- **Fine-tuning** — [finetuning/](../src/tabpfn/finetuning/): `FinetunedTabPFNClassifier/Regressor`. 실제 가중치 학습. AMP, DDP, early stopping
- **Differentiable input**: `fit_with_differentiable_input` (예제: [input_gradients.py](../examples/input_gradients.py))
- **저장/로드**: `save_fitted_tabpfn_model`, `load_fitted_tabpfn_model`
- **임베딩 추출**: `get_embeddings()`

---

## 9. 실행 환경 설정

### 이 맥의 상황
- 시스템 Python이 3.15뿐 (TabPFN 지원 범위는 3.10~3.14) → **uv로 Python 3.12 가상환경** 사용
- Apple Silicon(arm64): GPU 가속(MPS) 자동 사용
- 가상환경은 **iCloud 폴더 밖** `~/venvs/tabpfn`에 둠 (PyTorch 수 GB가 iCloud로 동기화되는 것 방지)

### 설치 (완료됨)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv 설치
source ~/.local/bin/env                              # 설치 직후 같은 터미널에서 uv 인식
uv venv ~/venvs/tabpfn --python 3.12
source ~/venvs/tabpfn/bin/activate
cd "/Users/chan/Library/Mobile Documents/com~apple~CloudDocs/Desktop/대학원/dxlab/20260921_Technical Study Paper/TabPNF Study/TabPFN"
uv pip install -e .     # editable 설치: src/tabpfn 수정이 바로 반영됨
```

### 매번 실행할 때
```bash
source ~/venvs/tabpfn/bin/activate
cd "/Users/chan/Library/Mobile Documents/com~apple~CloudDocs/Desktop/대학원/dxlab/20260921_Technical Study Paper/TabPNF Study/TabPFN"
python examples/tabpfn_for_binary_classification.py   # v3.5 (기본)
python study/run_v2.py                                 # v2 분류 + 회귀
```

VS Code 인터프리터: `Cmd+Shift+P` → "Python: Select Interpreter" → `~/venvs/tabpfn/bin/python`

### 모델 버전과 라이선스

| 모델 | 라이선스 | 동의 절차 | 비용 |
|---|---|---|---|
| **v2** | Prior Labs License (Apache 2.0 + 출처 표기) | 없음 | 무료, 상업적 사용 가능 |
| **v2.5 ~ v3.5** | 비상업적 라이선스 | Prior Labs 로그인 + 동의 (처음 한 번) | 연구·학습 무료, 상업적 사용은 유료 |

```python
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

clf = TabPFNClassifier()                                            # v3.5
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)  # v2
```

- v3.5 라이선스 동의: **완료** (API 키는 `~/.cache/tabpfn/auth_token`에 저장)
- 모델 파일 캐시: `~/Library/Caches/tabpfn/` (v3.5 약 876MB, v2 다운로드 완료)

### 실행 결과 기록

유방암 데이터 (569행, 30피처, test 33%):

| | v2 | v3.5 |
|---|---|---|
| ROC AUC | 0.997 | 0.9952 |
| Accuracy | 0.9734 | 0.9734 |

캘리포니아 집값 1000행 회귀 (v2): R² 0.8835, 80% 예측 구간 출력 확인

### 문제 해결 기록
- **`zsh: command not found: uv`**: uv 설치 전에 열린 터미널. `source ~/.local/bin/env` 실행 또는 터미널 재시작
- **라이선스 대기 화면에서 멈춤**: 정상. 브라우저에서 로그인/동의하거나, API 키를 붙여 넣고 Enter. 아이패드 터널로 실행할 때는 API 키 붙여 넣기 방식만 동작
- **`Could not reach the license server ... HTTP 400`**: 저장된 API 키 파일이 망가진 경우 (`API key:` 입력칸에 키가 아닌 짧은 문자가 들어가 저장됨). 코드가 400을 "서버 연결 실패"로 잘못 처리함.
  ```bash
  wc -c ~/.cache/tabpfn/auth_token   # 2~3바이트처럼 작으면 망가진 것
  rm ~/.cache/tabpfn/auth_token      # 지우고 다시 실행
  ```
- **`Warning: unauthenticated requests to the HF Hub`**: 무시해도 됨

---

## 10. 논문 원문: 향후 연구 방향과 한계

### Conclusion (PDF 7쪽)

> "Potential future directions include **scaling to larger datasets**⁴⁸, **handling data drift**⁴⁹, **investigating fine-tuning abilities across related tabular tasks**⁵⁰ and **understanding the theoretical foundations** of our approach⁵¹. Future work could also explore **creating specialized priors** to handle data types such as **time series**⁵² and **multi-modal data**, or specialized modalities such as ECG, neuroimaging data⁵³ and genetic data."

| # | 방향 | 참고문헌 (먼저 읽을 논문) |
|---|---|---|
| ① | 대규모 데이터로 확장 | [48] Feuer et al., *TuneTables: context optimization for scalable prior-data fitted networks* (NeurIPS 2024) |
| ② | 데이터 드리프트 대응 | [49] Helli et al., *Drift-resilient TabPFN: In-context learning temporal distribution shifts on tabular data* (NeurIPS 2024) |
| ③ | 관련 과제 파인튜닝 | [50] Thomas et al., *Retrieval & fine-tuning for in-context tabular models* (ICML 2024 ICL Workshop) |
| ④ | 이론적 기반 | [51] Nagler, *Statistical foundations of prior-data fitted networks* (ICML 2023) |
| ⑤ | 특화 prior (시계열, 멀티모달, ECG 등) | [52] Dooley et al., *ForecastPFN* (NeurIPS 2023) |

④는 이론 연구라 실험 과제 후보에서 제외. 밀도 추정, 데이터 생성, 임베딩, 파인튜닝 시연(Fig. 6)은 **이미 시연된 기능**이지 향후 방향이 아님.

### Limitations of TabPFN (Methods, PDF 9쪽)
1. 추론 속도가 CatBoost 같은 최적화된 방법보다 느릴 수 있음
2. 메모리 사용량이 데이터 크기에 선형으로 증가
3. 평가는 **10,000행, 500피처**까지. 더 큰 데이터는 추가 연구 필요

User guide: 10,000행·500피처 이하에서 강함. 더 큰 데이터나 매우 non-smooth한 회귀는 CatBoost, XGBoost, AutoGluon이 나을 가능성이 높음.

### 학습 설정 (Methods, PDF 10쪽)
- 약 **200만 step × 배치 64 = 약 1억 3천만 개 합성 데이터셋**
- **RTX 2080 Ti 8장으로 약 2주**
- 데이터셋당 train 최대 2,048행 (균등 샘플링), 검증 128행, 피처 1~160개 (beta 분포), 테이블 크기 75,000셀 이하
- Adam, linear warmup + cosine annealing, 학습률 0.0001~0.0005

→ **논문 규모 재현은 불가능.** 직접 구현은 "작은 규모(GPU 1장, 몇 시간)로 같은 원리 재현"이 목표.

### Prior 구조 (Fig. 2, PDF 3~4쪽)
1. 상위 하이퍼파라미터 샘플링 (데이터 크기, 피처 수, 노드 수, 그래프 복잡도)
2. **랜덤 DAG(구조적 인과 모델, SCM)** 생성
3. 루트 노드에 랜덤 노이즈(정규/균등 분포, 샘플 간 비독립성 다양) 입력 후 그래프를 따라 전파
4. 각 엣지 연결 유형: **작은 신경망**(sigmoid, ReLU, modulo, sine 등), **이산화**(범주형 생성), **결정 트리**(규칙 기반). 엣지마다 가우시안 노이즈 추가
5. 랜덤 노드 위치에서 피처(F)와 타깃(T) 추출
6. 후처리, 양자화, warping. 결측치 등 현실적 문제 추가

---

## 11. 실험 계획

### 방향 × 데이터 조합

| 방향 | 구매 데이터 | 공정·설비 데이터 | 난이도 |
|---|---|---|---|
| ① 대규모 | 1만 행 이상이 흔해 잘 맞음 | 피처 500개 이상이면 잘 맞음 | 낮음 (subsampling 옵션으로 바로 실험) |
| ② **드리프트** | 계절성, 프로모션 | **설비 노후화, 센서 드리프트, 레시피 변경** | 중간 (시간순 분할) |
| ③ 파인튜닝 | 매장/상품군이 여러 개일 때 | 같은 종류 설비/라인이 여러 개일 때 | 중간 ([finetuning/](../src/tabpfn/finetuning/)) |
| ⑤ 특화 prior | 고객 행동 패턴 | 공정 데이터처럼 생긴 합성 데이터 | 높음 (직접 구현과 연결) |

### 추천안: 공정·설비 데이터 + ② 드리프트
- 공정 데이터는 시간에 따라 분포가 바뀌는 게 자연스러움. "작년 데이터로 학습한 모델이 올해도 통하는가"는 현장의 실제 질문
- 실험 설계가 단순함: **랜덤 분할 vs 시간순 분할**의 성능 차이 = 드리프트의 영향. 그 차이를 줄이는 방법 비교
  - 예: 최근 데이터만 컨텍스트로 쓰기, 시간 피처 추가, 최근 데이터 가중 컨텍스트 샘플링
- 비교할 선행 연구 존재: Drift-resilient TabPFN [49]
- 확장: 직접 구현한 작은 TabPFN의 prior에 "시간에 따라 변하는 SCM"을 넣는 ⑤ 실험 → 직접 구현을 한 사람만 할 수 있는 차별점

### 대안: 구매 데이터 + ① 대규모
1만 행을 살짝 넘는 데이터에서 컨텍스트 구성 방식(랜덤 / 층화 / kNN 검색) 비교. 가장 빨리 결과가 나오는 안전한 선택.

### 데이터 후보 (크기와 구성은 내려받은 뒤 확인)

| 데이터 | 분야 | 대략 크기 | 특징 | 맞는 방향 |
|---|---|---|---|---|
| UCI SECOM | 반도체 공정 | 약 1,567행 × 590피처, 불량 약 7% | 타임스탬프 포함, 결측 많음 | ②, ① (피처) |
| UCI AI4I 2020 Predictive Maintenance | 설비 고장 | 10,000행 × 약 14피처, 고장 약 3% | 깔끔함, 시간 정보 제한적 | ③, 불균형 |
| Bosch Production Line (Kaggle) | 생산 라인 | 100만 행 이상, 수천 피처 | 매우 큼 | ① 극단 사례 |
| UCI Online Shoppers Purchasing Intention | 구매 | 12,330행 × 18피처, 구매 약 15% | 1만 행 한계를 살짝 넘음 | ①, ③ |
| UCI Online Retail | 구매 | 50만 건 이상 거래 | 고객 단위 집계 필요 | ②, ① |

연구실/협력 기업의 실제 데이터가 가장 좋음. 공개 가능 여부는 교수님께 확인.

### 일정 (8주)

| 주차 | 내용 | 결과물 |
|---|---|---|
| 1 | 논문 정독 + 참고문헌 [49] 또는 [48] + 데이터 확정 | 실험 계획 1장 |
| 2~4 | **과제 1**: 작은 TabPFN 직접 구현 | 구현 코드, 공식 TabPFN/XGBoost 비교표 |
| 5 | **과제 3**: 고른 데이터로 기준선 (TabPFN v2/v3.5, XGBoost, CatBoost) | 랜덤 vs 시간순 분할 성능표 |
| 6~7 | **과제 2**: 개선 방법 2~3가지 실험 | 개선 효과 그래프 |
| 8 | 정리, 발표 | 발표 자료 |

---

## 12. 직접 구현 로드맵

목표: **v2 구조를 아주 작게 만들고, 합성 데이터로 처음부터 학습**하기.

### 1단계: 합성 데이터 생성기 (prior) — 가장 중요
`sample_dataset(n_rows, n_features, n_classes) -> (X, y)`
1. 선형 / 가우시안 프로세스 prior (회귀부터)
2. 무작위 MLP prior: 랜덤 가중치 MLP의 출력을 y로
3. **SCM prior**: 랜덤 DAG, 노드 = 부모의 랜덤 함수 + 노이즈. 일부 노드를 X로, 하나를 y로 (논문 Fig. 2)
4. 분류 변환: 연속 y를 분위수로 잘라 클래스로
5. 현실성: 결측치, 범주형 피처, 무의미한 피처, 다양한 분포

검증: 생성한 데이터셋에 sklearn 모델을 돌려 난이도가 다양한지 확인

### 2단계: 모델 (v2 축소판, 200~300줄)

| 구성요소 | 참고 | 단순화 |
|---|---|---|
| 셀 임베딩 | `_embed_features` | 피처마다 `Linear(1→E)`, 그룹핑 없음 |
| y 임베딩 | `_embed_targets` | train: `Linear(y)` 또는 class embedding, test: 학습되는 빈칸 벡터 |
| 피처 간 attention | `AlongRowAttention` | `nn.MultiheadAttention` |
| 행 간 attention | `AlongColumnAttention` | query는 전체 행, key/value는 train 행만 |
| 블록 | `TabPFNBlock` | 행 attention → 열 attention → MLP (post-norm) |
| 디코더 | `_decode` | test 행 y 칸 → `Linear(E→n_classes)` |

핵심 코드:
```python
# x: (B*C, R, E),  N = train 행 수
q = x                      # 모든 행이 질문하고
k = v = x[:, :N]           # train 행에만 답을 구함 (test끼리는 서로 안 봄)
out = attn(q, k, v)
```

검증: shape 확인. **test 행 하나를 바꿔도 다른 test 행 출력이 변하지 않는지** 테스트.

### 3단계: 사전학습 루프
```python
for step in range(num_steps):
    X, y = batch_of_synthetic_datasets()       # 매 step 새 데이터셋
    N = random_split_point()                   # train/test 분할도 랜덤
    logits = model(X, y[:N])
    loss = cross_entropy(logits, y[N:])
    loss.backward(); opt.step()
```
- 시작 규모: emsize 128, 4~6층, 행 ≤ 500, 피처 ≤ 20, 클래스 ≤ 5
- 검증: loss 감소 확인. 실제 데이터(iris, breast_cancer, wine)에서 로지스틱 회귀, RandomForest와 비교

### 4단계: 추론 래퍼 + 앙상블
- sklearn 스타일 `fit`(저장만)/`predict_proba`
- 전처리: 결측치, 표준화/quantile, 범주형 인코딩
- 앙상블: 피처 순서, 클래스 순서, 전처리를 바꿔 K번 예측 → 클래스 순서 복원 후 평균
- 검증: K=1 vs K=8 성능 비교

### 5단계: 회귀 (bar distribution)
- z-정규화 후 prior의 y 분위수로 bucket 경계 결정
- bucket 분류로 학습 (bucket 폭으로 나눠 밀도로 맞춤)
- 예측 분포에서 mean/median/quantile 계산
- 검증: 데이터가 없는 구간에서 분포가 넓어지는지 확인

### 6단계 (선택)
- KV 캐시, v3.5식 CLS 행 압축, 공식 모델과 비교 분석
- 참고: 교육용 최소 구현(예: nanoTabPFN)이 공개되어 있음. 직접 짜 본 뒤 비교 권장

### 배경 논문
- Müller et al., 2022, *Transformers Can Do Bayesian Inference* (ICLR) — PFN 원리
- Hollmann et al., 2023, *TabPFN* v1 (ICLR) — prior 설계 상세
- Hollmann et al., 2025, Nature — v2 구조

---

## 13. 코드 공부 순서와 디버깅 방법

### 읽는 순서
1. [examples/tabpfn_for_binary_classification.py](../examples/tabpfn_for_binary_classification.py) — 사용법
2. [classifier.py](../src/tabpfn/classifier.py)의 `fit`, `forward`, `logits_to_probabilities` — 전체 흐름
3. **[tabpfn_v2.py](../src/tabpfn/architectures/tabpfn_v2.py)** — 논문과 1:1 대응, 가장 짧음
4. [ensemble.py](../src/tabpfn/preprocessing/ensemble.py), [pipeline_factory.py](../src/tabpfn/preprocessing/pipeline_factory.py) — 앙상블
5. [bar_distribution.py](../src/tabpfn/architectures/shared/bar_distribution.py) — 회귀 출력
6. [tabpfn_v3_5.py](../src/tabpfn/architectures/tabpfn_v3_5.py)의 `forward`, `_stages_0_to_2` (청크/캐시 분기는 처음엔 건너뛰기)

### 디버거로 흐름 따라가기
1. 인터프리터를 `~/venvs/tabpfn/bin/python`으로 선택
2. 중단점 위치:

| 순서 | 위치 | 볼 것 |
|---|---|---|
| ① | [classifier.py:959](../src/tabpfn/classifier.py#L959) `fit` | fit이 실제로 하는 일 |
| ② | [classifier.py:1657](../src/tabpfn/classifier.py#L1657) `predict` | 예측의 출발점 |
| ③ | [tabpfn_v2.py:800](../src/tabpfn/architectures/tabpfn_v2.py#L800) `forward` | 입력 `x`, `y`의 shape |
| ④ | [tabpfn_v2.py:166](../src/tabpfn/architectures/tabpfn_v2.py#L166) 열 attention | train/test 분리 (`single_eval_pos`) |

3. [run_v2.py](run_v2.py)를 열고 **F5** → "Python Debugger: Current File"
4. Variables 창에서 shape 메모. F10 = 다음 줄, F11 = 함수 안으로

### 첫 주 계획
| 일 | 내용 |
|---|---|
| 1 | 논문 초록, Fig. 1·2, Methods 아키텍처 |
| 2 | `run_v2.py`로 train 행 수, `n_estimators` 바꿔 보며 결과 표 작성 |
| 3~4 | 디버거로 fit → predict 흐름과 shape 추적 |
| 5 | 입력 → 전처리/앙상블 → 임베딩 → 블록 × 12 → 출력 그림 한 장 정리 |

---

## 14. 원격 접속 (아이패드)

### VS Code Remote Tunnels
아이패드에서 고치는 파일이 **이 맥의 바로 그 파일**이고, 실행도 맥에서 됨.

1. 맥 VS Code 왼쪽 아래 Accounts → **Turn on Remote Tunnel Access** → GitHub 로그인
2. 터널 이름: **`seunchanui-macbookpr`**
3. 아이패드 Safari: **https://vscode.dev/tunnel/seunchanui-macbookpr** → 같은 GitHub 계정(GTigerPapa)으로 로그인
4. 맥 잠자기 방지: 시스템 설정 → 배터리 → 옵션 → "전원 어댑터 연결 시 자동 잠자기 방지". 덮개는 열어 둘 것

현재 상태 (2026-11-02 확인):
- 터널은 맥에서 실행 중이지만, **서비스 설치는 실패**해서 "이 세션에서만" 모드. 맥 VS Code를 닫으면 터널도 꺼짐
- 아이패드에서 **"Opening Remote…"에서 멈춤**. 맥 로그에 접속 기록 없음 → 요청이 맥까지 오지 않음
- 시도할 것: 주소의 터널 이름 확인, 아이패드 설정 → Safari → "크로스 사이트 추적 방지" 끄기, 또는 Chrome/Edge 앱으로 접속

### SSH
다른 기기에서 맥 터미널에 접속하는 방법. 아이패드에서는 VS Code 화면 없이 터미널만 쓰게 됨. 같은 와이파이 안에서는 쉽지만(시스템 설정 → 공유 → 원격 로그인), 외부 접속은 포트포워딩이나 Tailscale이 필요함. 현재 목적에는 Remote Tunnels가 더 적합.

### 이 노트 보기
- 프로젝트 폴더가 iCloud Drive 안에 있으므로 **아이패드 "파일" 앱**에서 바로 열람 가능
- Claude Code 대화 기록 자체는 맥의 `~/.claude/projects/`에만 있고 git에는 포함되지 않음

---

## 15. 할 일

- [ ] GitHub push 권한 해결 후 업로드 (https://github.com/GTigerPapa/tabPFN-study)
  - 로컬 커밋은 완료 (`main` 브랜치). push가 403으로 거부됨 → 키체인의 옛 토큰 삭제 후 새 Personal Access Token(`repo` 권한 또는 해당 저장소 Contents 쓰기 권한)으로 재시도
    ```bash
    printf "protocol=https\nhost=github.com\n" | git credential-osxkeychain erase
    git push -u origin main
    ```
  - 논문 PDF와 `study/` 폴더를 커밋에 포함할지 결정 (논문은 CC BY 4.0)
- [ ] 아이패드 원격 터널 접속 문제 해결
- [ ] 실험 방향(드리프트 / 대규모)과 데이터(공정 / 구매) 확정, 교수님과 상의
- [ ] 참고문헌 [49] Drift-resilient TabPFN 또는 [48] TuneTables 읽기
- [ ] 5주차 기준선 스크립트 작성 (랜덤 vs 시간순 분할, TabPFN vs XGBoost)
- [ ] 2~4주차 직접 구현 뼈대 작성 (`prior.py`, `model.py`, `train.py`)

---

## 16. 첫 실험 결과 — SECOM 베이스라인 (2026-11-04)

스크립트: [study/secom_baseline.py](secom_baseline.py) · 데이터: `study/data/` (UCI SECOM, 1567행 x 474센서(상수 제거 후), 불량 6.6%, 2008-07-19 ~ 10-17)

| 분할 | 모델 | ROC AUC | PR AUC |
|---|---|---|---|
| 무작위 (70/30, stratified, seed 0) | TabPFN v3.5 | 0.761 | 0.207 |
| 무작위 | HistGradBoost | 0.693 | 0.195 |
| 시간순 (앞 70% → 뒤 30%) | TabPFN v3.5 | 0.516 | 0.058 |
| 시간순 | HistGradBoost | 0.574 | 0.071 |

- 무작위 분할에선 TabPFN 우세, **시간순 분할에선 두 모델 모두 찍기 수준(0.5)으로 붕괴** → data drift 연구 주제의 근거.
- 주의: seed 1개, test 불량 26~31개뿐이라 숫자 변동이 큼. 다음 단계에서 여러 seed / 여러 시간 구간으로 확인 필요.
- XGBoost는 Mac에서 `brew install libomp` 필요해서 sklearn HistGradientBoosting으로 대체.

---

## 17. 데이터 결정 — 날씨 → 태양광 발전량 (2026-11-04)

데이터 고민 경로: SECOM(해석 불가) → 나라장터(데이터 과다, TabPFN 필요성 약함) → 콘크리트(변화 없음) → **날씨 + 태양광** (적은 데이터 + 계절 drift + 발전소 간 전이)

- 연구 질문(안): "신규 태양광 발전소의 첫 2주~1달 데이터 + 날씨만으로 발전량을 얼마나 예측할 수 있나? 다른 발전소 데이터를 빌려 오면 나아지나?"
  → 논문 future direction 중 data drift, related tabular tasks 전이와 연결
- API (공공데이터포털, 키는 study/.env 의 G2B_KEY, git 제외됨)
  - 한국남동발전 시간대별 태양광 발전실적: `apis.data.go.kr/B551893/solar-power-by-hour/list` (응답 키가 `reponse` 오타)
  - 기상청 ASOS 일자료 `1360000/AsosDalyInfoService`, 시간자료 `1360000/AsosHourlyInfoService` (`getWthrDataList`)
  - (참고) 나라장터 공공데이터개방표준서비스 `1230000/ao/PubDataOpnStdService` 도 승인됨
- 수집: [study/collect_solar_weather.py](collect_solar_weather.py) → `study/data/solar/solar_daily.csv` (14,958행, 발전기 21대, 2023~2024), `solar_hourly.csv`
- 관측소 매칭: 구미·남해 관측소는 일사량 미측정 → 구미→대구(143), 삼천포→진주(192). 탑선태양광은 위치 불명 + 발전량 0이라 제외.
- 품질: 발전량-일사량 일별 상관 0.51~0.86. 하루치 시간 프로필에서 일부 발전기(영동, 삼천포#4~6, 영흥#3)가 오후에 급등하는 이상 패턴 → 시간 단위 실험 전 점검 필요.
