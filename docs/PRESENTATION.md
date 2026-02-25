# HumanForYou — Prédiction de l'Attrition des Employés

## Guide Complet de Présentation

---

## Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Les données](#2-les-données)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Pipeline de traitement des données](#4-pipeline-de-traitement-des-données)
5. [Analyse exploratoire (EDA)](#5-analyse-exploratoire-eda)
6. [Feature Engineering](#6-feature-engineering)
7. [Modélisation et benchmark](#7-modélisation-et-benchmark)
8. [Optimisation du modèle final](#8-optimisation-du-modèle-final)
9. [Explicabilité (SHAP & LIME)](#9-explicabilité-shap--lime)
10. [Éthique et équité](#10-éthique-et-équité)
11. [Robustesse et validation](#11-robustesse-et-validation)
12. [Limites et perspectives](#12-limites-et-perspectives)
13. [Résumé des résultats clés](#13-résumé-des-résultats-clés)
14. [Glossaire](#14-glossaire)

---

## 1. Contexte et problématique

### L'entreprise

**HumanForYou** est une entreprise pharmaceutique d'environ **4 000 employés**. Elle fait face à un **taux d'attrition annuel de ~16%**, ce qui signifie qu'environ **640 employés quittent l'entreprise chaque année**.

### Pourquoi c'est un problème ?

Le départ d'un employé coûte cher à une entreprise :

- **Coûts de recrutement** : annonces, entretiens, processus de sélection
- **Coûts de formation** : intégration du remplaçant, montée en compétence
- **Perte de productivité** : période de transition, perte de savoir-faire
- **Impact moral** : effet domino sur les équipes restantes

> En moyenne, remplacer un employé coûte entre **50% et 200% de son salaire annuel** (source : SHRM — Society for Human Resource Management).

### L'objectif

Construire un **modèle prédictif de Machine Learning** capable de :

1. **Identifier les employés à risque de départ** avant qu'ils ne partent
2. **Comprendre les facteurs** qui poussent les employés à partir
3. Permettre aux RH de mettre en place des **actions de rétention ciblées**

### Le raisonnement

Plutôt que de réagir *après* un départ, l'idée est d'**anticiper**. Si le modèle identifie qu'un employé a 75% de chances de partir, les RH peuvent intervenir : entretien, réévaluation salariale, changement de poste, formation, etc.

---

## 2. Les données

### Source

Les données proviennent du dataset **IBM HR Analytics** disponible sur Kaggle. C'est un dataset **semi-synthétique** (généré artificiellement par IBM à des fins pédagogiques), ce qui explique les performances très élevées obtenues (voir [Limites](#12-limites-et-perspectives)).

### Les 5 fichiers sources

| Fichier | Description | Taille | Colonnes |
|---------|-------------|--------|----------|
| `general_data.csv` | Données RH principales (âge, salaire, poste, ancienneté...) | 4 410 lignes × 24 colonnes | Age, Attrition, MonthlyIncome, JobRole... |
| `employee_survey_data.csv` | Enquête de satisfaction employés (juin 2015) | 4 410 × 4 | EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance |
| `manager_survey_data.csv` | Évaluation managériale (février 2015) | 4 410 × 3 | JobInvolvement, PerformanceRating |
| `in_time.csv` | Heures d'entrée (badge) sur 2015 | 4 410 × 262 | Timestamps quotidiens |
| `out_time.csv` | Heures de sortie (badge) sur 2015 | 4 410 × 262 | Timestamps quotidiens |

### La variable cible : Attrition

```
Attrition = "L'employé a-t-il quitté l'entreprise en 2016 ?"

  Non (Stay)  : 3 699 employés (83.9%)
  Oui (Leave) :   711 employés (16.1%)
```

C'est un problème de **classification binaire déséquilibrée** : il y a ~5× plus de "Stay" que de "Leave".

### Qualité des données

| Problème | Détail | Impact |
|----------|--------|--------|
| **Valeurs manquantes** | NumCompaniesWorked (19), TotalWorkingYears (9), EnvironmentSatisfaction (25), JobSatisfaction (20), WorkLifeBalance (38) | Faible (< 1% chacun), traité par KNNImputer |
| **Colonnes constantes** | EmployeeCount (=1), Over18 (='Y'), StandardHours (=8) | Supprimées (aucune information) |
| **Doublons** | 0 doublons sur EmployeeID dans chaque fichier | Aucun problème |
| **Cohérence inter-fichiers** | 4 410 employés dans tous les fichiers | Parfaitement aligné |

---

## 3. Architecture du projet

### Structure des fichiers

```
HumanForYou/
├── data/
│   └── raw/                              # Données brutes (immuables, jamais modifiées)
│       ├── general_data.csv
│       ├── employee_survey_data.csv
│       ├── manager_survey_data.csv
│       ├── in_time.csv
│       └── out_time.csv
├── notebooks/                            # Pipeline séquentiel (00 → 05)
│   ├── 00_Environment_Check.ipynb        # Vérification de l'environnement
│   ├── 01_Data_Validation_Pipeline.ipynb # Validation, fusion, badge
│   ├── 02_EDA_Explorer.ipynb             # Analyse exploratoire
│   ├── 03_Feature_Engineering.ipynb      # Ingénierie des features
│   ├── 04_Model_Benchmark.ipynb          # Comparaison de 9 algorithmes
│   └── 05_Model_Optimization.ipynb       # Optimisation, éthique, explicabilité
├── scripts/                              # Scripts autonomes pour la reproductibilité
│   ├── feature_selection.py              # Sélection hybride de features
│   ├── run_ablation.py                   # Tests d'ablation (5 scénarios)
│   └── run_production_validation.py      # Pipeline complet de production
├── outputs/                              # Artefacts générés (modèles, CSV, JSON)
├── requirements.txt                      # Dépendances Python
├── pyproject.toml                        # Métadonnées du projet
├── setup.bat / setup.sh                  # Scripts d'installation
└── README.md
```

### Philosophie de conception

1. **Séquentiel** : Chaque notebook dépend du précédent (00 → 01 → 02 → 03 → 04 → 05)
2. **Reproductible** : Les scripts dans `scripts/` reproduisent l'intégralité du pipeline sans notebooks
3. **Données immuables** : Les fichiers bruts ne sont jamais modifiés ; toutes les transformations génèrent de nouveaux fichiers dans `outputs/`

---

## 4. Pipeline de traitement des données

### Étape 1 — Vérification de l'environnement (NB 00)

Avant toute analyse, on vérifie que :
- Python ≥ 3.10 est installé
- Toutes les bibliothèques nécessaires sont disponibles (numpy, pandas, scikit-learn, xgboost, shap, lime, fairlearn...)
- Les 5 fichiers de données existent et sont lisibles

### Étape 2 — Validation et fusion des données (NB 01)

#### 2a. Validation des schémas
Chaque fichier est vérifié : les colonnes attendues sont-elles présentes ? Les types sont-ils corrects ?

#### 2b. Traitement des données badge (in_time / out_time)

Les fichiers badge contiennent **262 colonnes** (une par jour ouvré de 2015). Pour chaque employé, on calcule des indicateurs agrégés.

**Décision critique : restriction au 1er semestre 2015 (H1)**

> *Pourquoi ?* Les départs (Attrition) ont lieu en **2016**. Si on utilise les données badge de toute l'année 2015, on risque de capturer le **désengagement pré-départ** (un employé qui va partir travaille moins dans les derniers mois). C'est du **data leakage temporel** — le modèle apprendrait un signal qui n'existerait pas au moment de la prédiction.
>
> En ne gardant que janvier–juin 2015 (129 jours), on s'assure que les comportements observés sont **antérieurs** à toute décision de départ.

**3 features badge calculées :**

| Feature | Description | Moyenne | Écart-type |
|---------|-------------|---------|------------|
| `avg_working_hours` | Heures de travail moyennes par jour | 7.70 | 1.34 |
| `absence_rate` | Taux d'absence (jours sans badge / total) | 0.088 | 0.025 |
| `late_arrival_rate` | Taux d'arrivée tardive (après 10h) | 0.501 | 0.046 |

**2 features badge supprimées :**
- `avg_arrival_hour` : quasi-constante (écart-type = 0.018) → aucune information
- `avg_departure_hour` : corrélation r = 0.9999 avec `avg_working_hours` → redondance parfaite

**Alerte data leakage sur `avg_working_hours` :**

| Métrique | Attrition=Oui | Attrition=Non | Différence |
|----------|:---:|:---:|:---:|
| Moyenne | 8.317 | 7.583 | +0.735 |
| Cohen's d | — | — | **+0.548** |
| p-value | — | — | **8.36 × 10⁻³⁷** |

> L'effet est grand (d > 0.5) et très significatif. Même en H1, les employés qui vont partir travaillent plus. Cela pourrait être un signal légitime (surcharge → burn-out → départ) ou du leakage résiduel. Par prudence, cette feature est **exclue du modèle final** et son impact est quantifié par tests d'ablation (< 0.5% de perte en F1).

#### 2c. Fusion des 4 sources

Toutes les sources sont fusionnées sur `EmployeeID` → dataset de **4 410 lignes × 29 colonnes**.

---

## 5. Analyse exploratoire (EDA)

### Distribution de la cible

Le dataset est **déséquilibré** : 84% de "Stay" vs 16% de "Leave". Ce déséquilibre doit être pris en compte dans la modélisation (via `class_weight="balanced"`).

### Tests statistiques

Pour chaque feature, on teste si elle est **significativement différente** entre les deux groupes (Stay vs Leave) :

#### Features numériques — Test de Mann-Whitney U (p < 0.05)

| Feature | p-value | Significatif ? |
|---------|---------|:-:|
| TotalWorkingYears | 1.93 × 10⁻³⁹ | Très significatif |
| avg_working_hours | 8.36 × 10⁻³⁷ | Très significatif |
| YearsAtCompany | 1.21 × 10⁻³⁶ | Très significatif |
| YearsWithCurrManager | 2.47 × 10⁻³¹ | Très significatif |
| Age | 5.99 × 10⁻³⁰ | Très significatif |
| JobSatisfaction | 1.27 × 10⁻¹¹ | Très significatif |
| EnvironmentSatisfaction | 2.66 × 10⁻¹⁰ | Très significatif |
| YearsSinceLastPromotion | 4.04 × 10⁻⁴ | Significatif |
| WorkLifeBalance | 8.56 × 10⁻⁴ | Significatif |
| TrainingTimesLastYear | 1.03 × 10⁻² | Significatif |
| PercentSalaryHike | 3.73 × 10⁻² | Significatif |

> **Interprétation** : Les employés qui partent sont en moyenne **plus jeunes**, ont **moins d'ancienneté**, sont **moins satisfaits** de leur environnement et de leur travail, et ont un **moins bon équilibre vie pro/perso**.

#### Features catégorielles — Test du Chi² (p < 0.05)

| Feature | Chi² | p-value | Significatif ? |
|---------|------|---------|:-:|
| MaritalStatus | 138.49 | 8.45 × 10⁻³¹ | Très significatif |
| BusinessTravel | 72.55 | 1.76 × 10⁻¹⁶ | Très significatif |
| EducationField | 46.19 | 8.29 × 10⁻⁹ | Très significatif |
| Department | 29.09 | 4.82 × 10⁻⁷ | Très significatif |
| JobRole | 25.12 | 1.49 × 10⁻³ | Significatif |
| Gender | 1.35 | 0.245 | **Non significatif** |

> **Interprétation** : Le statut marital est le facteur catégoriel le plus discriminant. Les **célibataires** ont un taux d'attrition nettement plus élevé. Le **genre n'a pas d'impact** significatif sur le départ.

### Analyse d'équité initiale

| Variable sensible | Disparate Impact | Acceptable (> 0.80) ? |
|---|:---:|:---:|
| Genre (M vs F) | 0.918 | Oui |
| Statut marital | 0.748 | **Non** (< 0.80) |

> Le statut marital présente un **biais** : les célibataires sont disproportionnellement prédits comme partants. Ce biais sera traité dans la phase d'optimisation.

---

## 6. Feature Engineering

### Principes respectés

1. **Pas de data leakage** : Toutes les transformations sont faites **avant** le split train/test pour les features dérivées, mais les statistiques d'imputation et de scaling sont calculées **uniquement sur le train**
2. **Pas de SMOTE** : On utilise `class_weight="balanced"` plutôt que de sur-échantillonner la classe minoritaire (résultats de CV plus stables)

### Split train/test

```
Train : 3 528 échantillons (16.1% d'attrition) — 80%
Test  :   882 échantillons (16.1% d'attrition) — 20%
Méthode : Split stratifié (random_state=42)
```

### Nouvelles features créées

| Feature | Formule | Intuition |
|---------|---------|-----------|
| `IncomePerJobLevel` | MonthlyIncome / JobLevel | Un employé sous-payé par rapport à son niveau hiérarchique est plus susceptible de partir |
| `PromotionStagnation` | YearsSinceLastPromotion / (YearsAtCompany + 1) | Un employé qui n'a pas été promu depuis longtemps par rapport à son ancienneté stagne |
| `SatisfactionScore` | Moyenne de (EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance) | Score de satisfaction global synthétique |
| `ManagerStability` | YearsWithCurrManager / (YearsAtCompany + 1) | La stabilité de la relation managériale est un facteur de rétention |

### Encodage des variables catégorielles

- **Ordinal** : `BusinessTravel` → 0 (Non-Travel), 1 (Travel_Rarely), 2 (Travel_Frequently)
- **One-Hot** (avec `drop_first=True` pour éviter la colinéarité) : Department, EducationField, Gender, JobRole, MaritalStatus

Résultat : **46 colonnes** après encodage.

### Imputation des valeurs manquantes

- **Méthode** : KNNImputer (k=5, pondéré par distance)
- **Fit sur le train uniquement**, puis transformé sur train ET test (pas de fuite d'information)
- Colonnes concernées : NumCompaniesWorked, TotalWorkingYears, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance

> **Pourquoi KNN et pas la médiane ?** Le KNNImputer prend en compte la **structure locale** des données : il remplace une valeur manquante par la moyenne pondérée des 5 voisins les plus proches. C'est plus fin qu'une simple médiane globale.

### Sélection des features (46 → 23 → 22)

#### Méthode hybride (corrélation + importance Gini)

On combine deux critères pour décider quelles features garder :

1. **Corrélation avec Attrition** : |r| ≥ 0.03 (la feature est liée à la cible)
2. **Importance Gini** (Random Forest) : la feature est dans le top 82% de l'importance cumulée

**Règle** : On garde une feature si elle satisfait **au moins un** des deux critères.

Résultat : **23 features** retenues sur 46.

#### Filtrage post-sélection

1. **Filtre de variance** (seuil < 0.01) : supprime `late_arrival_rate` (quasi-constante) → **22 features**
2. **Filtre de corrélation** (|r| > 0.90 entre features) : aucune paire détectée → pas de suppression

### Les 22 features finales

| # | Feature | Origine | Description |
|---|---------|---------|-------------|
| 1 | Age | RH | Âge de l'employé |
| 2 | TotalWorkingYears | RH | Années d'expérience totale |
| 3 | YearsAtCompany | RH | Ancienneté dans l'entreprise |
| 4 | MonthlyIncome | RH | Salaire mensuel brut |
| 5 | YearsWithCurrManager | RH | Années avec le manager actuel |
| 6 | NumCompaniesWorked | RH | Nombre d'entreprises précédentes |
| 7 | DistanceFromHome | RH | Distance domicile-travail (km) |
| 8 | PercentSalaryHike | RH | % d'augmentation salariale en 2015 |
| 9 | TrainingTimesLastYear | RH | Jours de formation en 2015 |
| 10 | YearsSinceLastPromotion | RH | Années depuis la dernière promotion |
| 11 | BusinessTravel | RH | Fréquence des déplacements (ordinal) |
| 12 | EnvironmentSatisfaction | Enquête | Satisfaction environnement (1-4) |
| 13 | JobSatisfaction | Enquête | Satisfaction travail (1-4) |
| 14 | WorkLifeBalance | Enquête | Équilibre vie pro/perso (1-4) |
| 15 | IncomePerJobLevel | Dérivée | Salaire rapporté au niveau hiérarchique |
| 16 | ManagerStability | Dérivée | Stabilité managériale |
| 17 | SatisfactionScore | Dérivée | Score de satisfaction composite |
| 18 | PromotionStagnation | Dérivée | Indicateur de stagnation promotionnelle |
| 19 | MaritalStatus_Single | One-Hot | Employé célibataire |
| 20 | MaritalStatus_Married | One-Hot | Employé marié |
| 21 | JobRole_Manufacturing Director | One-Hot | Rôle = Directeur de production |
| 22 | EducationField_Technical Degree | One-Hot | Formation technique |

### Scaling (normalisation)

- **Méthode** : StandardScaler (moyenne=0, écart-type=1)
- Fit sur le train uniquement, appliqué sur train et test

---

## 7. Modélisation et benchmark

### Stratégie face au déséquilibre des classes

Plutôt que d'utiliser **SMOTE** (Synthetic Minority Oversampling Technique) qui crée des échantillons synthétiques de la classe minoritaire, on utilise `class_weight="balanced"` dans les algorithmes qui le supportent.

> **Pourquoi ?** SMOTE gonfle artificiellement le train set et peut créer des points aberrants. `class_weight="balanced"` pénalise davantage les erreurs sur la classe minoritaire lors de l'apprentissage, ce qui produit des résultats de validation croisée plus stables.
>
> Pour XGBoost : `scale_pos_weight = 3699/711 = 5.20`

### 9 algorithmes comparés

| Rang | Algorithme | class_weight | Accuracy | Precision | Recall | F1 | AUC-ROC | Temps (s) |
|:---:|------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **XGBoost** | scale_pos_weight | 0.997 | 1.000 | 0.979 | **0.989** | **0.998** | 0.23 |
| 2 | MLP | — | 0.997 | 1.000 | 0.979 | 0.989 | 0.992 | 2.30 |
| 3 | Random Forest | balanced | 0.996 | 1.000 | 0.972 | 0.986 | 0.998 | 0.17 |
| 4 | Decision Tree | balanced | 0.994 | 1.000 | 0.965 | 0.982 | 0.982 | 0.02 |
| 5 | SVM (RBF) | balanced | 0.882 | 0.596 | 0.831 | 0.694 | 0.929 | 1.95 |
| 6 | Gradient Boosting | — | 0.881 | 0.863 | 0.310 | 0.456 | 0.893 | 0.50 |
| 7 | Logistic Regression | balanced | 0.726 | 0.326 | 0.662 | 0.437 | 0.768 | 0.02 |
| 8 | KNN | — | 0.827 | 0.449 | 0.338 | 0.386 | 0.899 | 0.00 |
| 9 | AdaBoost | — | 0.863 | 0.706 | 0.254 | 0.373 | 0.773 | 0.32 |

### Analyse des résultats

**Les modèles à base d'arbres dominent** (XGBoost, Random Forest, Decision Tree) car ils :
- Gèrent naturellement les interactions non-linéaires entre features
- Supportent nativement `class_weight` / `scale_pos_weight`
- Sont robustes aux features non normalisées

**Les modèles linéaires échouent** (Logistic Regression : F1 = 0.437) car la frontière de décision n'est pas linéaire.

**Les modèles sans gestion du déséquilibre souffrent** :
- KNN (F1 = 0.386) : prédit majoritairement "Stay"
- AdaBoost (F1 = 0.373) : pondération initiale inadaptée
- Gradient Boosting (F1 = 0.456) : pas de `class_weight` natif

### Sélection du top 3 pour le tuning

1. **XGBoost** — Meilleur F1 et AUC-ROC
2. **Random Forest** — F1 très proche, modèle plus simple
3. **Gradient Boosting** — Potentiel d'amélioration avec tuning

---

## 8. Optimisation du modèle final

### Hyperparameter tuning (GridSearchCV)

Les 3 meilleurs modèles sont optimisés par **recherche sur grille** avec **validation croisée 5-fold stratifiée** :

**Espace de recherche XGBoost :**
- `n_estimators` : [100, 200, 300]
- `max_depth` : [3, 5, 7, 9]
- `learning_rate` : [0.01, 0.05, 0.1, 0.2]
- `subsample` : [0.7, 0.8, 1.0]
- `scale_pos_weight` : [1, 3, 5]

### Calibration du seuil de décision

#### Le problème du seuil par défaut

Par défaut, un classifieur prédit "Leave" si la probabilité est > **0.50**. Mais ce seuil n'est pas forcément optimal pour maximiser le F1 ou le Recall.

#### Méthode utilisée : calibration par validation croisée

```
1. On fait une prédiction out-of-fold (cross_val_predict) sur le train
2. On teste tous les seuils de 0.10 à 0.90 (pas de 0.01)
3. On choisit le seuil qui maximise le F1 avec Recall ≥ 0.60
4. Ce seuil est ensuite appliqué FIXE sur le test set
```

> **Pourquoi cette méthode ?** On n'utilise **jamais** le test set pour choisir le seuil, ce qui serait du data leakage. Les prédictions out-of-fold simulent des données non vues.

### Résultats après optimisation

| Modèle | Seuil optimal | CV-Precision | CV-Recall | CV-F1 | Test-Precision | Test-Recall | Test-F1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.57** | 0.994 | 0.875 | **0.931** | 1.000 | 0.958 | **0.978** |
| XGBoost | 0.69 | 0.982 | 0.882 | 0.930 | 1.000 | 0.979 | 0.989 |
| Gradient Boosting | 0.45 | 0.970 | 0.893 | 0.930 | 1.000 | 0.972 | 0.986 |

### Modèle final retenu : Random Forest @ seuil = 0.57

> **Pourquoi Random Forest et pas XGBoost ?** Bien que XGBoost ait un Test-F1 légèrement supérieur, le Random Forest a été retenu après le tuning et la calibration de seuil, offrant un excellent compromis précision/rappel avec un CV-F1 de 0.931 et une précision parfaite sur le test.

**Rapport de classification final :**

```
              precision    recall  f1-score   support

       Stay       0.99      0.99      0.99       742
      Leave       1.00      0.96      0.98       140

   accuracy                           0.99       882
  macro avg       0.99      0.97      0.98       882
```

**Interprétation concrète** :
- Sur 140 employés qui vont vraiment partir dans le test set, le modèle en identifie correctement **134** (recall = 0.96)
- Parmi tous les employés flaggés comme "à risque", **100%** partent effectivement (precision = 1.00)
- Seulement **6 départs** ne sont pas détectés (faux négatifs)

---

## 9. Explicabilité (SHAP & LIME)

### Pourquoi l'explicabilité est essentielle ?

Un modèle qui dit "cet employé va partir" sans expliquer **pourquoi** est inutile pour les RH. Ils ont besoin de **leviers d'action** : sur quoi agir pour retenir l'employé ?

### SHAP (SHapley Additive exPlanations) — Vue globale

SHAP attribue à chaque feature une **contribution** à la prédiction finale. C'est basé sur la théorie des jeux (valeurs de Shapley).

**Top features identifiées par SHAP :**

Les facteurs qui contribuent le plus à la prédiction de départ sont :

1. **Age** / **TotalWorkingYears** — Les jeunes employés peu expérimentés partent plus
2. **SatisfactionScore** — La satisfaction globale est un signal fort
3. **JobSatisfaction** / **WorkLifeBalance** — Satisfaction du travail et équilibre
4. **YearsAtCompany** / **YearsWithCurrManager** — Ancienneté et stabilité managériale
5. **MonthlyIncome** — Un salaire bas augmente le risque de départ
6. **ManagerStability** — Un changement fréquent de manager augmente le risque

### LIME (Local Interpretable Model-Agnostic Explanations) — Vue individuelle

LIME explique **une prédiction à la fois** en entraînant un modèle linéaire local autour de l'individu.

**Exemple concret** : Pour les 5 employés à plus haut risque de départ, LIME montre les contributions de chaque feature. Par exemple :

> *Employé #1234 : 82% de probabilité de départ*
> - TotalWorkingYears = 2 → **+18% de risque** (très peu d'expérience)
> - JobSatisfaction = 1 → **+15% de risque** (insatisfait)
> - MonthlyIncome = 25 000 ₹ → **+12% de risque** (sous-payé)
> - MaritalStatus = Single → **+8% de risque**
> - Levier d'action RH : augmentation salariale + entretien de satisfaction

---

## 10. Éthique et équité

### Cadre réglementaire : ALTAI / EU AI Act

Le projet intègre une **évaluation éthique complète** basée sur :
- **ALTAI** (Assessment List for Trustworthy AI) — Cadre de l'Union Européenne
- **EU AI Act** — Réglementation européenne sur l'IA (2024)

Un modèle qui prédit le départ des employés touche à des **décisions RH** potentiellement discriminatoires. Il est impératif de vérifier qu'il ne discrimine pas sur des critères protégés.

### Métriques d'équité évaluées

| Métrique | Définition | Seuil acceptable |
|----------|------------|:---:|
| **Disparate Impact (DI)** | Ratio des taux de prédiction positive entre groupes | ≥ 0.80 (règle des 4/5) |
| **Equal Opportunity (ΔTPRs)** | Différence des taux de vrais positifs entre groupes | < 0.10 |
| **Predictive Parity (ΔPPVs)** | Différence des valeurs prédictives positives entre groupes | < 0.10 |

### Résultats par variable sensible

#### Genre (Homme vs Femme)

| Métrique | Valeur | Statut |
|----------|:---:|:---:|
| Disparate Impact | 0.918 | Acceptable |
| ΔTPRs | Faible | OK |
| ΔPPVs | Minimal | OK |

> **Le modèle ne discrimine pas selon le genre.**

#### Statut marital (Célibataire / Marié / Divorcé)

| Paire | DI | Statut |
|-------|:---:|:---:|
| Global | **0.748** | **PROBLÈME** (< 0.80) |
| Single vs Married | 0.554 | ALERTE |
| Single vs Divorced | 0.377 | ALERTE |
| Married vs Divorced | 0.681 | ALERTE |

> **Le modèle pénalise les célibataires** : ils sont surreprésentés dans les prédictions de départ.

#### Âge (18-30, 31-40, 41-50, 51+)

Plusieurs violations détectées entre groupes d'âge, notamment pour les 18-30 ans qui sont plus souvent prédits comme partants.

### Mitigation du biais — Seuils différenciés par statut marital

Pour corriger le biais sur le statut marital, on applique des **seuils de décision différents** par groupe :

| Groupe | Seuil | Effet |
|--------|:---:|-------|
| Divorcé | 0.11 | Seuil bas → plus de flagging (compensation du sous-flagging) |
| Marié | 0.14 | Seuil bas → idem |
| Célibataire | 0.68 | Seuil élevé → moins de flagging (compensation du sur-flagging) |

**Résultat après mitigation :**

| Métrique | Avant | Après | Objectif |
|----------|:---:|:---:|:---:|
| Disparate Impact (CV) | 0.748 | **0.801** | ≥ 0.80 |
| F1 (CV) | 0.931 | 0.810 | — |
| Recall (CV) | 0.875 | 0.863 | — |

> La mitigation du biais **fonctionne** (DI passe au-dessus de 0.80) mais au prix d'une **baisse de performance** (F1 : 0.931 → 0.810). C'est le **compromis équité-performance** classique en IA éthique.

---

## 11. Robustesse et validation

### Tests d'ablation (5 scénarios)

Les tests d'ablation permettent de **quantifier la contribution** de chaque groupe de features :

| Scénario | Description | Features | CV-F1 |
|----------|-------------|:---:|:---:|
| S0 | Baseline (H1 badge + features hybrides) | 22 | ~0.93 |
| S1 | Sans badge du tout | ~20 | ~0.926 |
| S2 | Badge année complète | ~22 | ~0.928 |
| S3 | Top 25 features manuelles | 25 | ~0.926 |
| S4 | Minimal (RH only, 15 features) | 15 | ~0.92 |

**Conclusion** : La contribution des features badge est **modeste** (Δ CV-F1 ≈ +0.004). Le signal principal vient des données RH et des enquêtes de satisfaction.

### Test de robustesse au bruit

On injecte du **bruit gaussien** sur les 14 features continues (non binaires) à 5 niveaux d'intensité croissante, avec 10 seeds aléatoires par niveau.

| Métrique | Valeur |
|----------|--------|
| F1 baseline (seuil=0.57) | 0.978 |
| Seuil de dégradation 5% | F1 = 0.930 |
| Seuil de dégradation 15% | F1 = 0.832 |

> Le modèle **maintient sa performance** sous perturbation modérée, ce qui indique une bonne robustesse.

### Intégrité du modèle

- **Hash SHA-256** du modèle exporté → garantit qu'il n'a pas été modifié
- **Métadonnées complètes** exportées (nom, seuil, features, métriques, importance)
- **Classification AI Act** : le système est classifié selon le cadre réglementaire européen

---

## 12. Limites et perspectives

### Limites connues

| Limite | Explication | Impact |
|--------|-------------|--------|
| **F1 = 0.99 irréaliste** | Le dataset IBM est semi-synthétique avec des frontières de décision artificiellement nettes. En production réelle, on attendrait F1 = 0.50–0.70 (benchmark industrie). | Les métriques ne sont pas transférables en production |
| **Biais MaritalStatus** | Le Disparate Impact est < 0.80 (règle des 4/5). Les célibataires sont surreprésentés dans les prédictions de départ. | Risque de discrimination à l'embauche si mal utilisé |
| **Données mono-annuelles** | Le modèle est entraîné uniquement sur 2015. La généralisation temporelle n'est pas vérifiée. | Le modèle pourrait être obsolète sur des données 2020+ |
| **Leakage résiduel possible** | Même restreint à H1 2015, `avg_working_hours` montre un signal fort vs Attrition. | Feature exclue du modèle final par prudence |

### Perspectives d'amélioration

1. **Données réelles** : Remplacer le dataset IBM par les vraies données RH de HumanForYou
2. **Enrichissement des features** : Ajouter des données sur les projets, les évaluations 360°, les entretiens annuels
3. **Modèle temporel** : Entraîner sur plusieurs années pour valider la généralisation
4. **Monitoring en production** : Déployer le modèle avec un suivi mensuel des métriques d'équité (DI > 0.80)
5. **Fairlearn avancé** : Utiliser des méthodes de mitigation du biais plus sophistiquées (ExponentiatedGradient, ThresholdOptimizer)
6. **Interface RH** : Développer un dashboard pour que les RH puissent consulter les prédictions et les explications SHAP/LIME

---

## 13. Résumé des résultats clés

### Chiffres à retenir pour la présentation

| Élément | Valeur |
|---------|--------|
| Nombre d'employés | **4 410** |
| Taux d'attrition | **16.1%** (711 départs) |
| Nombre de features finales | **22** (sur 46 post-encodage) |
| Split train/test | **80/20** stratifié |
| Algorithmes testés | **9** |
| Modèle final | **Random Forest** |
| Seuil optimisé | **0.57** (calibré par CV) |
| CV-F1 (non biaisé) | **0.931** |
| Test-F1 | **0.978** |
| Test-Precision | **1.000** |
| Test-Recall | **0.958** |
| Équité genre (DI) | **0.918** (acceptable) |
| Équité statut marital (DI) | **0.748 → 0.801** (après mitigation) |

### Principaux facteurs de départ

1. **Ancienneté faible** (TotalWorkingYears, YearsAtCompany)
2. **Jeune âge** (Age)
3. **Insatisfaction** (JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance)
4. **Revenu bas** (MonthlyIncome, IncomePerJobLevel)
5. **Instabilité managériale** (YearsWithCurrManager, ManagerStability)
6. **Stagnation de carrière** (PromotionStagnation)
7. **Célibat** (MaritalStatus_Single)
8. **Déplacements fréquents** (BusinessTravel)

### Stack technique

| Catégorie | Technologie |
|-----------|------------|
| Langage | Python 3.10+ |
| Données | pandas, numpy |
| Statistiques | scipy, statsmodels |
| Visualisation | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, XGBoost |
| Explicabilité | SHAP, LIME |
| Équité | fairlearn |
| Environnement | Jupyter Notebook |

---

## 14. Glossaire

| Terme | Définition |
|-------|------------|
| **Attrition** | Départ volontaire d'un employé de l'entreprise |
| **F1-Score** | Moyenne harmonique de la Précision et du Rappel. Varie de 0 à 1, où 1 est parfait |
| **Precision** | Parmi les employés prédits "partants", combien partent réellement ? (= vrais positifs / prédictions positives) |
| **Recall (Rappel)** | Parmi les employés qui partent réellement, combien sont correctement identifiés ? (= vrais positifs / réels positifs) |
| **AUC-ROC** | Aire sous la courbe ROC. Mesure la capacité du modèle à discriminer les deux classes. 1.0 = parfait, 0.5 = aléatoire |
| **Validation croisée (CV)** | Technique qui divise le train set en K plis, entraîne sur K-1 et évalue sur le dernier, K fois. Donne une estimation non biaisée de la performance |
| **Data leakage** | Fuite d'information : utiliser des données du futur ou du test set pendant l'entraînement, ce qui gonfle artificiellement les performances |
| **class_weight="balanced"** | Pénalise plus fortement les erreurs sur la classe minoritaire, proportionnellement à son sous-représentation |
| **SMOTE** | Technique de sur-échantillonnage qui crée des points synthétiques de la classe minoritaire |
| **SHAP** | Méthode d'explicabilité basée sur la théorie des jeux qui attribue une contribution à chaque feature pour chaque prédiction |
| **LIME** | Méthode d'explicabilité qui crée un modèle linéaire local autour d'une prédiction pour l'expliquer |
| **Disparate Impact** | Ratio des taux de sélection entre groupes protégés. < 0.80 = discrimination potentielle (règle des 4/5) |
| **KNNImputer** | Méthode d'imputation qui remplace les valeurs manquantes par la moyenne pondérée des K plus proches voisins |
| **One-Hot Encoding** | Transforme une variable catégorielle en N colonnes binaires (0/1), une par catégorie |
| **StandardScaler** | Normalise les features pour avoir moyenne = 0 et écart-type = 1 |
| **Seuil de décision** | Probabilité au-dessus de laquelle le modèle prédit "Leave". Par défaut 0.50, optimisé ici à 0.57 |
| **Tests d'ablation** | Tests qui retirent des groupes de features pour mesurer leur contribution individuelle |
| **ALTAI** | Assessment List for Trustworthy AI — cadre d'évaluation éthique de l'UE pour l'IA |
| **EU AI Act** | Réglementation européenne sur l'IA entrée en vigueur en 2024 |

---

*Document généré pour le projet CESI A5 — HumanForYou*
