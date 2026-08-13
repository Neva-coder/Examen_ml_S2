import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def find_best_threshold(y_true, y_probs):
    """Recherche le seuil de décision qui maximise le F1-score sur la classe 1."""
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        score = f1_score(y_true, (y_probs >= thresh).astype(int), pos_label=1)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1

def main():
    print("Démarrage...\n")

    # 1. CHARGEMENT DES DONNÉES
    print("1. Chargement des données...")
    train = pd.read_csv("reservations_train.csv")
    test = pd.read_csv("reservations_test.csv")

    # 2. IMPUTATION DES VALEURS MANQUANTES (Sans Data Leakage)
    print("2. Nettoyage et imputation des valeurs manquantes...")

    # agent_id -> 0
    train["agent_id"] = train["agent_id"].fillna(0)
    test["agent_id"] = test["agent_id"].fillna(0)

    # Médiane du Train pour les variables numériques
    num_impute_cols = ["enfants", "prix_moyen_nuit_eur", "demandes_speciales"]
    for col in num_impute_cols:
        if col in train.columns:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            test[col] = test[col].fillna(med)

    # Mode du Train pour la variable catégorielle
    if "marche_origine" in train.columns:
        mode_val = train["marche_origine"].mode()[0]
        train["marche_origine"] = train["marche_origine"].fillna(mode_val)
        test["marche_origine"] = test["marche_origine"].fillna(mode_val)

    # 3. FEATURE ENGINEERING TEMPOREL
    print("3. Feature engineering temporel...")
    for df in [train, test]:
        df["date_reservation"] = pd.to_datetime(df["date_reservation"])
        df["date_arrivee"] = pd.to_datetime(df["date_arrivee"])

        # Création des nouvelles variables
        df["delai_reservation"] = (
            df["date_arrivee"] - df["date_reservation"]
        ).dt.days
        df["mois_arrivee"] = df["date_arrivee"].dt.month
        df["jour_semaine_arrivee"] = df["date_arrivee"].dt.dayofweek

    # 4. VALIDATION TEMPORELLE (Split 80% / 20%)
    print("4. Préparation du découpage chronologique (Validation Temporelle)...")
    train_sorted = train.sort_values(by="date_reservation").reset_index(
        drop=True
    )

    split_idx = int(len(train_sorted) * 0.8)
    train_val = train_sorted.iloc[:split_idx].copy()
    val_val = train_sorted.iloc[split_idx:].copy()

    ignore_cols = [
        "reservation_id",
        "date_reservation",
        "date_arrivee",
        "reservation_annulee",
    ]
    features = [c for c in train.columns if c not in ignore_cols]

    X_tr = train_val[features].copy()
    y_tr = train_val["reservation_annulee"].copy()

    X_va = val_val[features].copy()
    y_va = val_val["reservation_annulee"].copy()

    # Définition propre des colonnes numériques et catégorielles
    num_cols = X_tr.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_tr.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if "agent_id" in X_tr.columns and "agent_id" not in cat_cols:
        cat_cols.append("agent_id")
        if "agent_id" in num_cols:
            num_cols.remove("agent_id")

    # Homogénéisation des types en string pour éviter les erreurs d'encodage
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype(str)
        X_va[col] = X_va[col].astype(str)

    # 5. PRÉTRAITEMENT SCIKIT-LEARN
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    # 6. BASELINE (RÉGRESSION LOGISTIQUE)
    print("\n--- ÉVALUATION BASELINE (RÉGRESSION LOGISTIQUE) ---")
    baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )

    baseline_model.fit(X_tr, y_tr)
    y_pred_base = baseline_model.predict(X_va)
    f1_base = f1_score(y_va, y_pred_base, pos_label=1)
    print(f"F1-Score Baseline (Seuil 0.5) : {f1_base:.4f}")

    # 7. MODÈLE AVANCÉ (XGBOOST + OPTIMISATION DU SEUIL)
    print("\n--- ÉVALUATION MODÈLE AVANCÉ (XGBOOST) ---")
    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_va_proc = preprocessor.transform(X_va)

    # Ratio de déséquilibre pour scale_pos_weight
    ratio = (len(y_tr) - sum(y_tr)) / sum(y_tr)

    xgb_val_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=ratio,
        random_state=42,
    )

    xgb_val_model.fit(X_tr_proc, y_tr)
    probs_xgb = xgb_val_model.predict_proba(X_va_proc)[:, 1]

    best_thresh, best_f1 = find_best_threshold(y_va, probs_xgb)
    print(f"Seuil optimal trouvé sur validation : {best_thresh:.2f}")
    print(f"Meilleur F1-Score XGBoost : {best_f1:.4f}\n")

    # 8. ENTRAÎNEMENT FINAL SUR 100% DU TRAIN ET PRÉDICTIONS TEST
    print("8. Entraînement final du modèle sur l'ensemble des données...")
    X_train_full = train[features].copy()
    y_train_full = train["reservation_annulee"].copy()
    X_test_final = test[features].copy()

    for col in cat_cols:
        X_train_full[col] = X_train_full[col].astype(str)
        X_test_final[col] = X_test_final[col].astype(str)

    X_train_proc = preprocessor.fit_transform(X_train_full)
    X_test_proc = preprocessor.transform(X_test_final)

    ratio_full = (len(y_train_full) - sum(y_train_full)) / sum(y_train_full)

    final_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=ratio_full,
        random_state=42,
    )

    final_model.fit(X_train_proc, y_train_full)

    # Prédictions sur le test
    test_probs = final_model.predict_proba(X_test_proc)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    # 9. EXPÉRIMENTATION ET CRÉATION DE SUBMISSION.CSV
    submission = pd.DataFrame(
        {
            "reservation_id": test["reservation_id"],
            "probabilite_annulation": np.round(test_probs, 4),
            "reservation_annulee": test_preds,
        }
    )

    submission.to_csv("submission.csv", index=False)

    print("submission.csv GÉNÉRÉ")
    print(f"Total lignes générées : {len(submission)} (Attendu : 2000)")
    print("\nRépartition des prédictions finales :")
    print(submission["reservation_annulee"].value_counts())
    print("\nAperçu du fichier :")
    print(submission.head())

if __name__ == "__main__":
    main()
