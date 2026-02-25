# HumanForYou -- Prediction d'attrition des employes

Projet de Machine Learning visant a predire le depart volontaire des employes de l'entreprise pharmaceutique HumanForYou (~4000 employes, ~16% d'attrition annuelle).

**Modele final** : XGBoost, 23 features, seuil optimise a 0.46 par validation croisee.  
**CV-F1 : 0.94** | Test-F1 : 0.99 (voir [Limites connues](#limites-connues) pour l'interpretation).

---

## Quickstart (3 commandes)

```bash
# 1. Cloner et entrer dans le projet
git clone https://github.com/aymen-elo/Human-For-You.git && cd Human-For-You

# 2. Installer l'environnement
# Windows :
setup.bat
# Linux / macOS :
chmod +x setup.sh && ./setup.sh

# 3. Lancer Jupyter et executer les notebooks dans l'ordre (00 -> 05)
jupyter notebook notebooks/
```

**Ou manuellement :**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name humanforyou --display-name "Python (HumanForYou)"
jupyter notebook notebooks/
```

---

## Pipeline d'analyse

Les notebooks doivent etre executes **dans l'ordre** :

| #  | Notebook | Description |
|----|----------|-------------|
| 00 | Environment Check | Verification des dependances et des fichiers de donnees |
| 01 | Data Validation Pipeline | Validation des schemas, audit qualite, fusion des 4 sources, extraction de 3 features badge (H1 2015) |
| 02 | EDA Explorer | Distributions, tests statistiques (Mann-Whitney, Chi2), correlations, analyse d'equite initiale |
| 03 | Feature Engineering | Feature engineering, encodage, imputation (KNNImputer post-split), selection de 23 features, scaling |
| 04 | Model Benchmark | Comparaison de 9 algorithmes avec `class_weight="balanced"`, courbes ROC/PR, validation croisee |
| 05 | Model Optimization | Tuning, calibration de seuil par CV, SHAP, LIME, equite multi-metriques, robustesse, integrite du modele |

Les scripts dans `scripts/` reproduisent le pipeline independamment des notebooks :
- `feature_selection.py` : selection hybride (correlation + Gini) avec audit complet
- `run_ablation.py` : 5 scenarios d'ablation pour quantifier l'apport de chaque groupe de features
- `run_production_validation.py` : pipeline de bout en bout (preprocessing -> benchmark -> tuning -> export)

---

## Choix techniques

| Decision | Justification |
|----------|---------------|
| **class_weight="balanced" au lieu de SMOTE** | Produit des resultats de CV plus stables et evite l'inflation artificielle du train set. |
| **Calibration du seuil par CV** (cross_val_predict) | Evite le data leakage : le seuil est optimise sur des predictions out-of-fold, pas sur le test set. |
| **Badge restreint a H1 2015** (jan-jun) | Evite la contamination temporelle : les departs ont lieu en 2016, le badge H2 pourrait refleter un desengagement pre-depart. |
| **XGBoost comme modele final** | Meilleur CV-F1 apres tuning (0.94 vs 0.93 pour RF). Meilleur controle de l'imbalance via scale_pos_weight. |
| **23 features (whitelist)** | Validee par tests d'ablation : la reduction de 41 a 23 features ne degrade le CV-F1 que de 0.004. |
| **KNNImputer post-split** | Imputation fit sur le train uniquement, evite le data leakage. |

---

## Considerations ethiques (ALTAI / AI Act)

Le notebook 05 integre 14 sections couvrant :
- **Equite** : Disparate Impact, Equal Opportunity, Predictive Parity sur genre, statut marital, age
- **Mitigation du biais** : seuils differencies par statut marital (calibres par CV)
- **Detection de proxies** : correlation des top features avec les variables sensibles
- **Explainabilite** : SHAP (global) + LIME (local, top 5 employes a risque)
- **Robustesse** : injection de bruit gaussien sur features continues (5 niveaux, 10 seeds)
- **Integrite** : hash SHA-256 du modele, metadata completes, classification AI Act
- **RGPD** : section de minimisation des donnees prete a activer

---

## Limites connues

1. **Performance elevee (F1 = 0.99 sur test)** : Le dataset source (IBM HR Analytics via Kaggle) est semi-synthetique. Les frontieres de decision sont artificiellement nettes. En production reelle, on anticipe F1 = 0.50-0.70 (benchmark industrie). Les tests d'ablation sur 5 scenarios et 10 random seeds prouvent qu'il n'y a pas de data leakage.

2. **Biais MaritalStatus** : Le Disparate Impact pour le statut marital est < 0.80 (regle des 4/5). La mitigation par seuils differencies est tentee mais peut echouer. Remediation recommandee : suppression des features MaritalStatus ou utilisation de fairlearn.

3. **Donnees mono-annuelles** : Le modele est entraine sur 2015 uniquement. La generalisation temporelle n'est pas verifiee.

---

## Donnees

Source : [HR Analytics Case Study](https://www.kaggle.com/vjchoudhary7/hr-analytics-case-study) (Kaggle)

| Fichier | Description | Taille |
|---------|-------------|--------|
| general_data.csv | 24 variables RH par employe (4410 lignes) | 0.5 MB |
| employee_survey_data.csv | Satisfaction (environnement, travail, equilibre) | < 0.1 MB |
| manager_survey_data.csv | Evaluation manageriale (implication, performance) | < 0.1 MB |
| in_time.csv / out_time.csv | Horaires de badge sur 2015 | ~22 MB chacun |

Voir `data/README.md` pour le dictionnaire complet des variables.

---

## Algorithmes evalues

| Algorithme | class_weight | F1 (test) | Note |
|------------|:---:|:---:|------|
| XGBoost | scale_pos_weight | **0.989** | Modele final retenu |
| Random Forest | balanced | 0.989 | CV-F1 inferieur apres tuning |
| MLP | - | 0.986 | Pas de gestion native de l'imbalance |
| Decision Tree | balanced | 0.975 | |
| SVM (RBF) | balanced | 0.724 | |
| Logistic Regression | balanced | 0.573 | |
| Gradient Boosting | - | 0.573 | Pas de class_weight natif |
| KNN | - | 0.498 | |
| AdaBoost | - | 0.404 | |

Note : les algorithmes sans `class_weight` sont desavantages dans ce benchmark (donnees desequilibrees 84/16).

---

## Technologies

- **Python 3.10+**
- pandas, numpy, scipy
- matplotlib, seaborn, plotly
- scikit-learn, XGBoost
- SHAP, LIME, fairlearn
- Jupyter Notebook