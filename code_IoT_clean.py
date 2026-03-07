import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from river import tree, preprocessing

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA (IoT_clean.csv)
# ─────────────────────────────────────────────
df = pd.read_csv("/Users/rakeshry/Library/CloudStorage/OneDrive-UniversitetetiAgder/Desktop/Paper_ML/IoT_clean.csv")

target_col = "label"

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode string target labels to integers if needed
if y.dtype == object:
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)

# Train/test split — done BEFORE any fitting to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Dataset: IoT_clean.csv")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Classes: {y.nunique()}")

# ─────────────────────────────────────────────
# 2. BUILD RIVER MODELS (with/without normalization)
# ─────────────────────────────────────────────

def make_hoeffding_tree(normalize=False):
    m = tree.HoeffdingTreeClassifier(grace_period=200, delta=1e-7, tau=0.05, leaf_prediction="mc")
    return preprocessing.StandardScaler() | m if normalize else m

def make_efdt(normalize=False):
    m = tree.ExtremelyFastDecisionTreeClassifier(grace_period=200, delta=1e-7, tau=0.05, min_samples_reevaluate=20, leaf_prediction="mc")
    return preprocessing.StandardScaler() | m if normalize else m

def make_hoeffding_adaptive_tree(normalize=False):
    m = tree.HoeffdingAdaptiveTreeClassifier(grace_period=200, delta=1e-7, tau=0.05, leaf_prediction="mc")
    return preprocessing.StandardScaler() | m if normalize else m

def make_sgt_classifier(normalize=False):
    m = tree.SGTClassifier(grace_period=200, delta=1e-7)
    return preprocessing.StandardScaler() | m if normalize else m

# ─────────────────────────────────────────────
# 3. TRAIN ON TRAINING SET ONLY
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, model_name):
    print(f"\nTraining {model_name}...")
    train_df = X_train.copy()
    train_df["__target__"] = y_train.values
    for i, (_, row) in enumerate(train_df.iterrows()):
        x = row.drop("__target__").to_dict()
        label = int(row["__target__"])
        model.learn_one(x, label)
        if (i + 1) % 10_000 == 0:
            print(f"  {i+1}/{len(train_df)} samples processed")
    print(f"  Done training {model_name}.")
    return model

# ─────────────────────────────────────────────
# 4. EVALUATE ON HELD-OUT TEST SET (no further learning)
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    preds = []
    for _, row in X_test.iterrows():
        x = row.to_dict()
        p = model.predict_one(x)
        if isinstance(p, bool):
            p = int(p)
        preds.append(p)
    return accuracy_score(y_test, preds), preds

models_config = [
    (make_hoeffding_tree, "Hoeffding Tree"),
    (make_efdt, "EFDT"),
    (make_hoeffding_adaptive_tree, "Hoeffding Adaptive Tree"),
    (make_sgt_classifier, "SGT"),
]

results_with = {}
results_without = {}

for make_fn, name in models_config:
    # With normalization
    model_norm = train_model(make_fn(normalize=True), X_train, y_train, f"{name} (normalized)")
    acc_norm, preds_norm = evaluate_model(model_norm, X_test, y_test, name)
    results_with[name] = acc_norm
    print(f"\n{name} [WITH normalization] — Test Accuracy: {acc_norm:.4f}")
    print(classification_report(y_test, preds_norm))

for make_fn, name in models_config:
    # Without normalization
    model_raw = train_model(make_fn(normalize=False), X_train, y_train, f"{name} (raw)")
    acc_raw, preds_raw = evaluate_model(model_raw, X_test, y_test, name)
    results_without[name] = acc_raw
    print(f"\n{name} [WITHOUT normalization] — Test Accuracy: {acc_raw:.4f}")
    print(classification_report(y_test, preds_raw))

# ─────────────────────────────────────────────
# 5. SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY — With vs Without Normalization (IoT_clean.csv)")
print("="*55)
print(f"{'Model':<30} {'With norm':>12} {'Without norm':>12} {'Best':>10}")
print("-"*55)
for name in [n for _, n in models_config]:
    w, wo = results_with[name], results_without[name]
    best = "Normalized" if w > wo else "Raw"
    print(f"{name:<30} {w:>12.4f} {wo:>12.4f} {best:>10}")
