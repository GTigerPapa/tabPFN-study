"""TabPFN v2로 분류/회귀를 돌려 보는 스터디용 스크립트 (라이선스 동의 불필요).

실행:
    source ~/venvs/tabpfn/bin/activate
    python study/run_v2.py
"""

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

# ---- 분류: 유방암 데이터 (569행, 30피처, 2클래스) ----
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
clf.fit(X_train, y_train)  # 첫 실행 시 v2 가중치 다운로드
proba = clf.predict_proba(X_test)
print("[분류] ROC AUC :", round(roc_auc_score(y_test, proba[:, 1]), 4))
print("[분류] Accuracy:", round(accuracy_score(y_test, clf.predict(X_test)), 4))

# ---- 회귀: 캘리포니아 집값 (CPU/MPS에서 빠르도록 1000행만 사용) ----
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X[:1000], y[:1000], test_size=0.33, random_state=42
)

reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
reg.fit(X_train, y_train)
print("[회귀] R^2     :", round(r2_score(y_test, reg.predict(X_test)), 4))

# 회귀는 분포를 예측하므로 분위수(불확실성)도 얻을 수 있음
q10, q90 = reg.predict(X_test[:3], output_type="quantiles", quantiles=[0.1, 0.9])
for i in range(3):
    print(f"  샘플{i}: 실제={y_test[i]:.2f}, 80% 구간=[{q10[i]:.2f}, {q90[i]:.2f}]")
