# -*- coding: utf-8 -*-
"""
Pretraga hiperparametara Random Forest-a (korak D4).

Izvrsava se kao zasebna skripta jer GridSearchCV unutar DataSpell-a
premasuje raspolozivu memoriju. Rezultat se snima u modeli/pretraga_rf.joblib,
odakle ga notebook ucitava.

Pokretanje:  python pretraga_rf.py
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "employee_promotion_prediction.csv"
MIN_WORKING_AGE = 15
MAX_MONTHLY_HOURS = 48 * 52 / 12

MODELI = Path("modeli")
TABELE = Path("tabele")
MODELI.mkdir(exist_ok=True)
TABELE.mkdir(exist_ok=True)

# ---------- iste pripreme kao u fazama A i C ----------
df_raw = pd.read_csv(DATA_PATH)
df = df_raw[(df_raw["age"] - df_raw["years_at_company"] >= MIN_WORKING_AGE)
            & (df_raw["avg_monthly_hours"] <= MAX_MONTHLY_HOURS)].copy()

TARGET = "promoted"
X = df.drop(columns=["employee_id", TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

num_kolone = X_train.select_dtypes(include=np.number).columns.tolist()
kat_kolone = X_train.select_dtypes(include="object").columns.tolist()
ATRIBUTI_STABLA = num_kolone + kat_kolone

pretprocesor = ColumnTransformer([
    ("num", StandardScaler(), num_kolone),
    ("kat", OneHotEncoder(handle_unknown="ignore", drop="first"), kat_kolone),
])

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"accuracy": "accuracy", "precision": "precision",
           "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

# ---------- pretraga ----------
mreza_rf = {
    "model__max_depth": [10, 20, None],
    "model__min_samples_leaf": [1, 20, 50],
    "model__class_weight": [None, "balanced_subsample"],
}

print(f"Trening: {len(X_train):,} redova | mreza: 18 kombinacija x 5 segmenata = 90 treniranja")
print("Broj stabala fiksiran na 200 (Breiman, 2001).\n")

t0 = time.time()
pretraga_rf = GridSearchCV(
    Pipeline([
        ("pre", pretprocesor),
        ("model", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                         random_state=RANDOM_STATE)),
    ], memory="kes_pipeline"),
    param_grid=mreza_rf,
    scoring=SCORING,
    refit="f1",
    cv=CV,
    n_jobs=1,
    verbose=2,
)
pretraga_rf.fit(X_train[ATRIBUTI_STABLA], y_train)
trajanje = (time.time() - t0) / 60

# ---------- snimanje ----------
joblib.dump(pretraga_rf, MODELI / "pretraga_rf.joblib", compress=3)

rez_rf = pd.DataFrame(pretraga_rf.cv_results_)[[
    "param_model__max_depth", "param_model__min_samples_leaf",
    "param_model__class_weight",
    "mean_test_accuracy", "mean_test_precision", "mean_test_recall",
    "mean_test_f1", "mean_test_roc_auc",
]].sort_values("mean_test_f1", ascending=False).round(4)
rez_rf.columns = ["max_depth", "min_samples_leaf", "class_weight",
                  "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
rez_rf.to_csv(TABELE / "tabela_10_pretraga_rf.csv", index=False, encoding="utf-8-sig")

print(f"\n{'='*62}")
print(f"Trajanje: {trajanje:.1f} min")
print("Najbolji parametri:",
      {k.replace('model__', ''): v for k, v in pretraga_rf.best_params_.items()})
print(f"Najbolji F1 iz unakrsne validacije: {pretraga_rf.best_score_:.4f}")
print(f"\nSacuvano: modeli/pretraga_rf.joblib")
print(f"Sacuvano: tabele/tabela_10_pretraga_rf.csv")
print(f"{'='*62}")
print(rez_rf.head(10).to_string(index=False))
