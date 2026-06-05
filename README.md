# Prédiction du Churn Client — Télécommunications

**Cours :** Techniques de sélection et évaluation des modèles  
**Enseignant :** Ing. Abdel-Hamid MOHAMED CHEIKH, SupNum  
**Dataset :** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Structure du projet

```
feautureEng/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Données brutes (7 043 × 21)
│   └── processed/
│       ├── X_train.csv / X_test.csv            ← Features prétraitées (29)
│       ├── X_train_selected.csv / X_test_selected.csv  ← Features sélectionnées (24)
│       ├── y_train.csv / y_test.csv
│       ├── scaler.pkl                          ← StandardScaler sauvegardé
│       ├── selected_features.json              ← Shortlist consensus
│       └── experiment_results.csv             ← Résultats 12 expériences
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   ← EDA, dictionnaire variables, visualisations
│   ├── 02_preprocessing.ipynb      ← Imputation, outliers, encodage, scaling
│   ├── 03_feature_selection.ipynb  ← Filter, Wrapper, Embedded
│   └── 04_modeling_experiments.ipynb ← 12 expériences, ROC, validation croisée
│
├── reports/
│   ├── rapport.pdf                 
│   
│   └── *.png                       ← 32 graphiques générés
│
├── .gitignore
└── README.md
```

## Pipeline résumé

```
Données brutes
  └─ 1. Imputation (4 stratégies comparées → constante 0 retenue)
  └─ 2. Outliers   (Z-score | IQR | Isolation Forest → Winsorisation)
  └─ 3. Encodage   (Label | One-Hot | Target | Frequency comparés)
  └─ 4. Scaling    (Min-Max vs Standardisation → Standardisation retenue)
  └─ 5. Feature Selection
        ├─ Filter   : Variance Threshold, Pearson, Spearman, Chi², ANOVA
        ├─ Wrapper  : Forward, Backward, RFE×3 (LR, SVM, RF)
        └─ Embedded : Lasso, Elastic Net, RF importance, GB importance
  └─ 6. Modélisation (LR, RF, GB, SVM × 3 pipelines = 12 expériences)
```

## Résultats

| Modèle | Baseline | Prétraitement | Prép+Sélection |
|--------|----------|---------------|----------------|
| **Logistic Regression** | 0.839 | 0.847 | **0.847** |
| Gradient Boosting | 0.844 | 0.842 | 0.843 |
| SVM | 0.734 | **+0.097 →** 0.832 | 0.832 |
| Random Forest | 0.824 | 0.820 | 0.822 |

**Meilleur modèle :** Logistic Regression + pipeline complet → AUC=0.847, Recall=0.797

## Environnement

```bash
conda activate churn_project
jupyter notebook notebooks/
```

Dépendances : `pandas numpy matplotlib seaborn scikit-learn scipy category_encoders jupyter`
