"""SECOM(반도체 공정 센서 데이터)으로 TabPFN vs 그래디언트 부스팅 비교.

데이터: UCI SECOM — 1567개 웨이퍼, 590개 센서값, 불량(1) 약 6.6%, 생산 시각 포함.
    study/data/secom.data, study/data/secom_labels.data

두 가지 방식으로 train/test를 나눠서 비교한다.
  1) 무작위 분할  : 보통 논문/대회에서 쓰는 방식
  2) 시간순 분할  : 앞 70% 기간으로 학습, 뒤 30% 기간을 예측 (실제 공장 상황)
두 점수 차이가 크면 → 시간이 지나며 데이터 성질이 바뀐다(data drift)는 신호.

실행:
    source ~/venvs/tabpfn/bin/activate
    python study/secom_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

DATA = Path(__file__).parent / "data"

# ---- 1. 데이터 읽기 ----
X = pd.read_csv(DATA / "secom.data", sep=" ", header=None)
labels = pd.read_csv(DATA / "secom_labels.data", sep=" ", header=None)
y = (labels[0] == 1).astype(int).to_numpy()  # 원본: -1=정상, 1=불량
time = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S")

# 값이 하나뿐인(정보 없는) 센서 제거. 결측치(NaN)는 두 모델 모두 그대로 처리 가능.
X = X.loc[:, X.nunique() > 1].to_numpy()
print(f"데이터: {X.shape[0]}행 x {X.shape[1]}센서, 불량률 {y.mean():.1%}")
print(f"기간  : {time.min():%Y-%m-%d} ~ {time.max():%Y-%m-%d}\n")


def evaluate(X_train, X_test, y_train, y_test):
    """두 모델을 학습/예측하고 (ROC AUC, PR AUC)를 돌려준다."""
    models = {
        "TabPFN": TabPFNClassifier(),  # 기본값 = v3.5
        "GradBoost": HistGradientBoostingClassifier(random_state=0),
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        scores[name] = (roc_auc_score(y_test, p), average_precision_score(y_test, p))
    return scores


def show(title, scores, n_test_fail):
    print(f"[{title}]  (test 불량 {n_test_fail}개)")
    for name, (roc, pr) in scores.items():
        print(f"  {name:10s} ROC AUC={roc:.3f}   PR AUC={pr:.3f}")
    print()


# ---- 2. 무작위 분할 ----
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)
show("무작위 분할", evaluate(X_tr, X_te, y_tr, y_te), y_te.sum())

# ---- 3. 시간순 분할 ----
order = np.argsort(time.to_numpy())
cut = int(len(order) * 0.7)
tr, te = order[:cut], order[cut:]
print(f"시간순: 학습 ~{time.iloc[tr].max():%m-%d}, 예측 {time.iloc[te].min():%m-%d}~")
show("시간순 분할", evaluate(X[tr], X[te], y[tr], y[te]), y[te].sum())

print("읽는 법: ROC AUC 0.5=찍기, 1.0=완벽 / PR AUC 기준선=불량률(약 0.07)")
