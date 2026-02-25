# Modèles et Méthodes — Guide Pédagogique Complet

## Document explicatif pour le projet HumanForYou

---

## Table des matières

1. [Prétraitement des données](#1-prétraitement-des-données)
   - 1.1 [Traitement des données badge](#11-traitement-des-données-badge)
   - 1.2 [Imputation des valeurs manquantes (KNNImputer)](#12-imputation-des-valeurs-manquantes---knnimputer)
   - 1.3 [Encodage des variables catégorielles](#13-encodage-des-variables-catégorielles)
   - 1.4 [Normalisation (StandardScaler)](#14-normalisation---standardscaler)
2. [Sélection des features](#2-sélection-des-features)
   - 2.1 [Corrélation avec la cible](#21-corrélation-avec-la-cible)
   - 2.2 [Importance Gini (Random Forest)](#22-importance-gini---random-forest)
   - 2.3 [Méthode hybride](#23-méthode-hybride)
   - 2.4 [Filtres post-sélection](#24-filtres-post-sélection)
3. [Gestion du déséquilibre des classes](#3-gestion-du-déséquilibre-des-classes)
4. [Les 9 algorithmes de classification](#4-les-9-algorithmes-de-classification)
   - 4.1 [Régression Logistique](#41-régression-logistique)
   - 4.2 [K-Nearest Neighbors (KNN)](#42-k-nearest-neighbors-knn)
   - 4.3 [Support Vector Machine (SVM)](#43-support-vector-machine-svm)
   - 4.4 [Arbre de Décision](#44-arbre-de-décision)
   - 4.5 [Random Forest](#45-random-forest)
   - 4.6 [Gradient Boosting](#46-gradient-boosting)
   - 4.7 [XGBoost](#47-xgboost)
   - 4.8 [AdaBoost](#48-adaboost)
   - 4.9 [MLP (Réseau de neurones)](#49-mlp-réseau-de-neurones)
5. [Optimisation des hyperparamètres](#5-optimisation-des-hyperparamètres)
   - 5.1 [GridSearchCV](#51-gridsearchcv)
   - 5.2 [RandomizedSearchCV](#52-randomizedsearchcv)
6. [Calibration du seuil de décision](#6-calibration-du-seuil-de-décision)
7. [Validation croisée stratifiée](#7-validation-croisée-stratifiée)
8. [Métriques d'évaluation](#8-métriques-dévaluation)
9. [Méthodes d'explicabilité](#9-méthodes-dexplicabilité)
   - 9.1 [SHAP](#91-shap)
   - 9.2 [LIME](#92-lime)
10. [Métriques d'équité](#10-métriques-déquité)
11. [Tests de robustesse](#11-tests-de-robustesse)
12. [Résumé des configurations exactes](#12-résumé-des-configurations-exactes)

---

## 1. Prétraitement des données

### 1.1 Traitement des données badge

#### Le problème

Les fichiers `in_time.csv` et `out_time.csv` contiennent **262 colonnes** (une par jour de 2015) avec des timestamps bruts. Il faut transformer ces milliers de valeurs en quelques indicateurs exploitables par employé.

#### La restriction au 1er semestre (H1 2015)

```
Fenêtre temporelle : 1er janvier → 30 juin 2015 (129 jours)
```

**Pourquoi ?** Les départs (Attrition) ont lieu en **2016**. Si on utilise le badge du 2e semestre 2015, on risque de capturer le **désengagement pré-départ** : un employé qui sait qu'il va partir travaille moins, arrive en retard, s'absente davantage. Ce signal ne serait **pas disponible** au moment de la prédiction réelle (on voudrait prédire *avant* que le désengagement commence).

C'est un cas classique de **data leakage temporel** : le modèle apprendrait un signal du futur.

#### Les 3 features calculées

**1. `avg_working_hours`** — Heures de travail moyennes par jour
```
Pour chaque jour valide :
    heures_travail = (heure_sortie + minute_sortie/60) - (heure_entrée + minute_entrée/60)

avg_working_hours = moyenne(heures_travail) sur tous les jours où l'employé a badgé
```
> *Intuition* : Un employé surchargé (heures élevées) peut être en burn-out ; un employé qui travaille peu peut être désengagé.

**2. `absence_rate`** — Taux d'absence
```
absence_rate = (jours_sans_badge) / (nombre_total_de_jours_ouvrés)
```
> *Intuition* : Un taux d'absence élevé peut signaler un désengagement ou des problèmes personnels.

**3. `late_arrival_rate`** — Taux d'arrivée tardive
```
late_arrival_rate = (jours_où_arrivée >= 10h00) / (jours_où_l'employé_a_badgé)
```
> *Intuition* : Des arrivées tardives fréquentes peuvent indiquer une démotivation.

#### Features badge supprimées

| Feature | Raison de suppression |
|---------|----------------------|
| `avg_arrival_hour` | Quasi-constante (écart-type = 0.018). Tous les employés arrivent à peu près à la même heure → aucune information discriminante |
| `avg_departure_hour` | Corrélation r = 0.9999 avec `avg_working_hours`. C'est une **redondance parfaite** : si on sait l'heure d'arrivée (constante) et les heures travaillées, on connaît l'heure de départ |
| `avg_working_hours` | Exclue du modèle final : Cohen's d = 0.548 vs Attrition (p < 10⁻³⁶). Effet trop large même en H1 → suspicion de leakage résiduel. Tests d'ablation confirment < 0.5% d'impact sur F1 |

---

### 1.2 Imputation des valeurs manquantes — KNNImputer

#### Le problème

5 colonnes ont des valeurs manquantes (< 1% chacune) :

| Colonne | Manquants | % |
|---------|:---------:|:---:|
| WorkLifeBalance | 38 | 0.9% |
| EnvironmentSatisfaction | 25 | 0.6% |
| JobSatisfaction | 20 | 0.5% |
| NumCompaniesWorked | 19 | 0.4% |
| TotalWorkingYears | 9 | 0.2% |

#### Pourquoi pas simplement la médiane ?

Remplacer par la médiane globale ignore la **structure des données**. Un employé de 25 ans, junior, célibataire a probablement un profil de satisfaction différent d'un directeur de 55 ans marié. La médiane globale "écrase" ces différences.

#### Comment fonctionne KNNImputer ?

```
Configuration : KNNImputer(n_neighbors=5, weights="distance")
```

**Principe** : Pour chaque valeur manquante, l'algorithme :

1. **Cherche les 5 employés les plus proches** (sur toutes les autres features non manquantes)
2. **Calcule la moyenne pondérée** de leurs valeurs pour la feature manquante
3. La **pondération par distance** (`weights="distance"`) donne plus de poids aux voisins les plus proches

**Exemple concret** :
```
Employé A : Age=28, Salary=30k, JobSatisfaction=?

Voisin 1 (distance=0.2) : JobSatisfaction = 3
Voisin 2 (distance=0.5) : JobSatisfaction = 2
Voisin 3 (distance=0.8) : JobSatisfaction = 4
Voisin 4 (distance=1.0) : JobSatisfaction = 3
Voisin 5 (distance=1.2) : JobSatisfaction = 2

Imputation = Σ(1/distance × valeur) / Σ(1/distance)
           = (5×3 + 2×2 + 1.25×4 + 1×3 + 0.83×2) / (5 + 2 + 1.25 + 1 + 0.83)
           ≈ 2.82
```

#### Protection contre le data leakage

```
imputer.fit(X_train)        ← Apprend les voisinages sur le train uniquement
imputer.transform(X_train)  ← Applique sur le train
imputer.transform(X_test)   ← Applique sur le test (sans re-fit !)
```

Si on faisait `fit(X_train + X_test)`, les voisins du test "verraient" des données d'entraînement et vice versa → fuite d'information.

---

### 1.3 Encodage des variables catégorielles

Les algorithmes de ML ne comprennent que des **nombres**. Il faut transformer les variables textuelles en valeurs numériques.

#### Encodage ordinal (pour les variables ordonnées)

Quand les catégories ont un **ordre naturel**, on attribue un entier croissant :

```
BusinessTravel :
    Non-Travel       → 0   (jamais)
    Travel_Rarely    → 1   (rarement)
    Travel_Frequently → 2   (fréquemment)
```

> L'ordre est important : "fréquemment" > "rarement" > "jamais". Un encodage ordinal préserve cette relation.

#### One-Hot Encoding (pour les variables nominales)

Quand les catégories n'ont **pas d'ordre** (ex: "Sales" n'est pas "supérieur" à "R&D"), on crée une colonne binaire par catégorie :

```
MaritalStatus = {Single, Married, Divorced}

Avec drop_first=True :
    MaritalStatus_Married  MaritalStatus_Single
              0                     1          ← Single
              1                     0          ← Married
              0                     0          ← Divorced (référence implicite)
```

**`drop_first=True`** : On supprime la 1ère catégorie (ici "Divorced") qui devient la **catégorie de référence**. Si Married=0 et Single=0, c'est forcément Divorced. Cela évite la **colinéarité parfaite** entre les colonnes.

**Variables encodées en One-Hot** : Department, EducationField, Gender, JobRole, MaritalStatus

**Résultat** : 29 colonnes brutes → **46 colonnes** après encodage.

---

### 1.4 Normalisation — StandardScaler

#### Le problème

Les features ont des échelles très différentes :
- `Age` : 18–60
- `MonthlyIncome` : 10 000–200 000 ₹
- `JobSatisfaction` : 1–4

Un algorithme comme KNN ou SVM traiterait MonthlyIncome comme beaucoup plus important que Age simplement parce que ses valeurs sont 1 000× plus grandes.

#### Comment fonctionne StandardScaler ?

```
Pour chaque feature :
    x_normalisé = (x - moyenne) / écart_type
```

Après transformation :
- **Moyenne = 0** pour toutes les features
- **Écart-type = 1** pour toutes les features

**Exemple** :
```
Age originale : [25, 30, 35, 40, 45]
Moyenne = 35, Écart-type = 7.07

Age normalisée : [-1.41, -0.71, 0.0, +0.71, +1.41]
```

#### Protection contre le data leakage

```
scaler.fit(X_train)        ← Calcule moyenne et écart-type sur le train
scaler.transform(X_train)  ← Normalise le train
scaler.transform(X_test)   ← Normalise le test avec les stats du train
```

> Les modèles à base d'arbres (Random Forest, XGBoost, Decision Tree) sont **invariants au scaling** — ils découpent sur des seuils, pas sur des distances. Le scaling est fait ici par cohérence et pour les algorithmes qui en ont besoin (SVM, KNN, MLP, Logistic Regression).

---

## 2. Sélection des features

### Pourquoi sélectionner ?

Sur 46 features post-encodage, beaucoup sont **non-informatives** (pas de lien avec Attrition) ou **redondantes** (corrélées entre elles). Garder trop de features :
- Ajoute du **bruit** (le modèle apprend des patterns aléatoires)
- Augmente le **risque de surapprentissage** (overfitting)
- Rend le modèle **moins interprétable**

### 2.1 Corrélation avec la cible

On calcule la **corrélation de Pearson** (r) entre chaque feature et la variable Attrition (encodée 0/1) :

```
r = corrélation de Pearson entre feature X et Attrition Y

Si |r| >= 0.03 → La feature est potentiellement utile
Si |r| < 0.03 → La feature n'a presque aucun lien linéaire avec la cible
```

> Le seuil de 0.03 est volontairement **bas** : on veut être inclusif à cette étape et ne pas perdre de signal.

**Limite** : La corrélation de Pearson ne capture que les **relations linéaires**. Une feature peut avoir une relation non-linéaire forte avec la cible mais une corrélation de Pearson faible.

### 2.2 Importance Gini — Random Forest

Pour compenser la limite de la corrélation linéaire, on utilise l'**importance Gini** d'un Random Forest entraîné temporairement :

```python
rf_temp = RandomForestClassifier(
    n_estimators=300,          # 300 arbres pour une estimation stable
    class_weight="balanced",   # Gestion du déséquilibre
    random_state=42
)
rf_temp.fit(X_train_scaled, y_train)
importance_gini = rf_temp.feature_importances_
```

**Comment ça marche ?**

Dans un arbre de décision, chaque noeud divise les données sur une feature. L'**importance Gini** d'une feature mesure la **réduction totale de l'impureté** (entropie/Gini) apportée par cette feature dans tous les noeuds de tous les arbres.

```
Impureté Gini d'un noeud = 1 - Σ(p_i²)

où p_i = proportion de la classe i dans le noeud

Noeud pur (100% Stay) : Gini = 1 - 1² = 0
Noeud 50/50 :           Gini = 1 - (0.5² + 0.5²) = 0.5
```

Plus une feature réduit l'impureté quand elle est utilisée pour diviser, plus elle est importante.

**Critère de sélection** : On garde les features qui représentent le **top 82% de l'importance cumulée** (triée par importance décroissante).

### 2.3 Méthode hybride

On combine les deux critères avec un **OU logique** :

```
Garder la feature si :
    |corrélation avec Attrition| >= 0.03
    OU
    la feature est dans le top 82% d'importance Gini cumulée
```

> L'union (OU) est choisie plutôt que l'intersection (ET) pour ne pas perdre de features qui seraient utiles pour un seul des deux critères. Par exemple, une feature avec une relation non-linéaire forte mais une corrélation de Pearson faible serait sauvée par le critère Gini.

**Résultat** : 46 features → **23 features** retenues.

### 2.4 Filtres post-sélection

Deux filtres supplémentaires sont appliqués après la sélection hybride :

#### Filtre de variance (seuil < 0.01)

```
Si la variance d'une feature < 0.01 → Suppression

Variance faible = la feature est quasi-constante = pas d'information
```

**Feature supprimée** : `late_arrival_rate` (variance trop faible sur les données non scalées)

#### Filtre de multicolinéarité (|r| > 0.90)

```
Si deux features ont une corrélation > 0.90 entre elles :
    → On supprime celle avec la plus faible importance Gini
```

**Résultat** : Aucune paire avec |r| > 0.90 détectée → pas de suppression.

**Bilan** : 46 → 23 (hybride) → 22 (variance) → **22 features finales**.

---

## 3. Gestion du déséquilibre des classes

### Le problème

```
Stay  : 3 699 employés (83.9%)
Leave :   711 employés (16.1%)
```

Un classifieur naïf qui prédit **toujours** "Stay" aurait une accuracy de 84% mais un recall de 0% (ne détecte aucun départ). L'accuracy seule est **trompeuse** sur des données déséquilibrées.

### Deux approches possibles

#### Option 1 : SMOTE (Synthetic Minority Oversampling) — NON RETENUE

SMOTE crée des **échantillons synthétiques** de la classe minoritaire en interpolant entre des voisins existants :

```
Pour chaque échantillon "Leave" :
    1. Trouver ses K voisins "Leave" les plus proches
    2. Choisir un voisin aléatoirement
    3. Créer un point synthétique entre les deux
```

**Pourquoi non retenu ?**
- Gonfle artificiellement le train set (de 3 528 à ~6 000+)
- Peut créer des points aberrants (interpolation dans des zones "Stay")
- Résultats de validation croisée **moins stables**

#### Option 2 : `class_weight="balanced"` — RETENUE

Plutôt que de modifier les données, on modifie la **fonction de perte** de l'algorithme :

```
Poids de chaque classe = n_total / (2 × n_classe)

Poids "Stay"  = 4410 / (2 × 3699) = 0.596
Poids "Leave" = 4410 / (2 × 711)  = 3.101
```

Chaque erreur sur un "Leave" est pénalisée **~5× plus** qu'une erreur sur un "Stay". L'algorithme est incité à bien classer la classe minoritaire.

**Pour XGBoost**, le paramètre équivalent est :
```
scale_pos_weight = n_négatifs / n_positifs = 3699 / 711 = 5.20
```

**Avantages** :
- Pas de modification des données
- Résultats CV plus stables
- Moins de risque de surapprentissage

---

## 4. Les 9 algorithmes de classification

### 4.1 Régression Logistique

#### Principe

Malgré son nom, c'est un algorithme de **classification** (pas de régression). Il modélise la **probabilité** qu'un employé parte via une fonction sigmoïde :

```
P(Leave) = 1 / (1 + e^(-z))

où z = β₀ + β₁×Age + β₂×Salary + ... + βₙ×Feature_n
```

La sigmoïde transforme n'importe quel score z en une probabilité entre 0 et 1.

#### Configuration utilisée

```python
LogisticRegression(
    max_iter=1000,           # Nombre max d'itérations pour converger
    random_state=42,         # Reproductibilité
    class_weight="balanced"  # Gestion du déséquilibre
)
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| Simple, rapide, interprétable | Ne capture que les relations **linéaires** |
| Donne des probabilités calibrées | Échoue si la frontière de décision est complexe |
| Coefficients = importance des features | Sensible aux features corrélées |

#### Résultat dans le projet

```
F1 = 0.437 | AUC = 0.768
```

> **Performance faible** : La frontière de décision entre Stay et Leave n'est pas linéaire dans ce dataset. La régression logistique ne peut pas capturer les interactions complexes entre features.

---

### 4.2 K-Nearest Neighbors (KNN)

#### Principe

KNN ne "apprend" rien — il mémorise toutes les données d'entraînement. Pour classer un nouvel employé :

1. **Calculer la distance** entre cet employé et tous les employés du train set
2. **Sélectionner les K plus proches** voisins
3. **Vote majoritaire** : la classe la plus fréquente parmi les K voisins gagne

```
Nouvel employé → Chercher les 5 voisins les plus proches
                → 3 sont "Stay", 2 sont "Leave"
                → Prédiction : "Stay"
```

#### Configuration utilisée

```python
KNeighborsClassifier(
    n_neighbors=5,   # 5 voisins
    n_jobs=-1        # Parallélisation
)
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| Aucun entraînement (lazy learning) | **Très sensible** au scaling des features |
| Capture des frontières non-linéaires | Lent en prédiction (calcule toutes les distances) |
| Simple à comprendre | Souffre de la **malédiction de la dimensionnalité** |
| | **Pas de class_weight** natif |

#### Résultat dans le projet

```
F1 = 0.386 | AUC = 0.899
```

> **Performance faible** : KNN n'a pas de mécanisme natif pour gérer le déséquilibre des classes. Avec 84% de "Stay" parmi les voisins, il prédit presque toujours "Stay".

---

### 4.3 Support Vector Machine (SVM)

#### Principe

SVM cherche l'**hyperplan** (une frontière linéaire en haute dimension) qui sépare au mieux les deux classes avec la **marge maximale**.

Avec un noyau **RBF** (Radial Basis Function), SVM projette les données dans un espace de dimension supérieure où elles deviennent linéairement séparables :

```
Espace original (2D) :          Espace projeté (3D) :
                                     /\
  x x x x                          / x \
 o o o x x        ──────→        /  x  \
 o o o o                        o o ─── o
                                 (séparables !)
```

Le noyau RBF mesure la **similarité gaussienne** entre deux points :
```
K(x, x') = exp(-γ × ||x - x'||²)
```

#### Configuration utilisée

```python
SVC(
    kernel="rbf",              # Noyau gaussien (non-linéaire)
    probability=True,          # Permet d'obtenir des probabilités
    random_state=42,
    class_weight="balanced"    # Gestion du déséquilibre
)
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| Excellent pour les frontières complexes | **Lent** sur les grands datasets (O(n²) à O(n³)) |
| Robuste au surapprentissage (marge maximale) | Sensible au scaling |
| Efficace en haute dimension | Les hyperparamètres (C, γ) sont critiques |

#### Résultat dans le projet

```
F1 = 0.694 | AUC = 0.929
```

> **Performance moyenne** : SVM avec paramètres par défaut. Un tuning de C et γ pourrait améliorer significativement les résultats, mais les modèles à base d'arbres restent plus adaptés ici.

---

### 4.4 Arbre de Décision

#### Principe

Un arbre de décision pose une **séquence de questions binaires** sur les features pour arriver à une prédiction :

```
                  Age < 30 ?
                /           \
            Oui               Non
           /                    \
  Satisfaction < 2 ?        YearsAtCompany < 3 ?
     /        \                /           \
  Leave     Stay           Leave          Stay
```

À chaque noeud, l'algorithme choisit la feature et le seuil qui **maximisent la séparation** entre les classes (réduction de l'impureté Gini).

#### Configuration utilisée

```python
DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)
```

> Aucune limitation de profondeur (`max_depth=None`), ce qui permet à l'arbre de croître jusqu'à ce que toutes les feuilles soient pures. Cela mène souvent au **surapprentissage** (l'arbre mémorise le train set).

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| **Très interprétable** (visualisable) | **Surapprentissage** fort sans élagage |
| Pas besoin de scaling | Instable (un petit changement de données = arbre différent) |
| Gère les non-linéarités | Biaisé vers les features avec beaucoup de valeurs |
| Rapide (train : 0.02s) | Frontières de décision rectangulaires |

#### Résultat dans le projet

```
F1 = 0.982 | AUC = 0.982
```

> **Bonne performance** : L'arbre profond capture bien les patterns, mais le risque de surapprentissage est élevé.

---

### 4.5 Random Forest

#### Principe

Random Forest corrige les faiblesses de l'arbre de décision en entraînant **plusieurs arbres** et en combinant leurs prédictions par **vote majoritaire** :

```
          Données d'entraînement
          /        |         \
   Échantillon 1  Éch. 2   Éch. 3   ... (bootstrap sampling)
        |           |         |
    Arbre 1     Arbre 2   Arbre 3    ... (100 arbres)
        \          |         /
         Vote majoritaire
              ↓
         Prédiction finale
```

**Deux sources de diversité** :
1. **Bagging** : Chaque arbre est entraîné sur un échantillon bootstrap (tirage aléatoire avec remise)
2. **Feature sampling** : À chaque noeud, seul un sous-ensemble aléatoire de features est considéré (√n features par défaut)

#### Configuration utilisée (benchmark)

```python
RandomForestClassifier(
    n_estimators=100,          # 100 arbres
    random_state=42,
    n_jobs=-1,                 # Parallélisation
    class_weight="balanced"    # Gestion du déséquilibre
)
```

#### Configuration après tuning (modèle final)

```python
# Grille de recherche :
{
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
# → 144 combinaisons testées (GridSearchCV)

# Meilleurs paramètres trouvés :
{
    "n_estimators": 100,       # 100 arbres suffisent
    "max_depth": None,         # Pas de limite de profondeur
    "min_samples_split": 2,    # Minimum 2 échantillons pour diviser
    "min_samples_leaf": 1      # Minimum 1 échantillon par feuille
}
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| **Réduit le surapprentissage** vs arbre unique | Moins interprétable qu'un arbre seul |
| Robuste aux outliers et au bruit | Peut être lent si beaucoup d'arbres |
| Feature importance intégrée | Biais vers les features à haute cardinalité |
| **Supporte class_weight** | |

#### Résultat dans le projet

```
Benchmark : F1 = 0.986 | AUC = 0.998
Après tuning (seuil=0.57) : CV-F1 = 0.931 | Test-F1 = 0.978  ← MODÈLE FINAL RETENU
```

---

### 4.6 Gradient Boosting

#### Principe

Contrairement au Random Forest qui entraîne les arbres **en parallèle**, le Gradient Boosting les entraîne **séquentiellement**. Chaque nouvel arbre corrige les **erreurs** de l'ensemble précédent :

```
Arbre 1 → Prédictions → Erreurs (résidus)
                              ↓
                         Arbre 2 → Corrige les erreurs de l'Arbre 1
                              ↓
                         Arbre 3 → Corrige les erreurs restantes
                              ↓
                            ...
```

Mathématiquement, chaque arbre est entraîné sur le **gradient négatif** de la fonction de perte :

```
Prédiction finale = Arbre_1(x) + η × Arbre_2(x) + η × Arbre_3(x) + ...

η = learning_rate (taux d'apprentissage, contrôle la "vitesse" de correction)
```

#### Configuration utilisée (benchmark)

```python
GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
    # PAS de class_weight natif !
)
```

#### Configuration après tuning

```python
# Grille de recherche :
{
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_split": [2, 5, 10],
    "subsample": [0.7, 0.8, 1.0]
}
# → 243 combinaisons, 50 testées (RandomizedSearchCV)

# Meilleurs paramètres trouvés :
{
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_split": 2,
    "subsample": 0.7             # 70% des données par arbre (stochastic GB)
}
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| Très performant (corrige itérativement) | Entraînement séquentiel → **non parallélisable** |
| Contrôle fin via learning_rate | Sensible au surapprentissage si trop d'arbres |
| subsample ajoute de la régularisation | **Pas de class_weight natif** (problème ici) |

#### Résultat dans le projet

```
Benchmark : F1 = 0.456 | AUC = 0.893
Après tuning (seuil=0.45) : CV-F1 = 0.930 | Test-F1 = 0.986
```

> **Amélioration spectaculaire après tuning** : Le benchmark sans `class_weight` était catastrophique (F1 = 0.456). Après tuning avec `subsample` et des hyperparamètres optimisés, le modèle rivalise avec les meilleurs.

---

### 4.7 XGBoost

#### Principe

**eXtreme Gradient Boosting** est une version optimisée du Gradient Boosting avec des améliorations clés :

1. **Régularisation L1/L2** intégrée (contrôle la complexité des arbres)
2. **Élagage intelligent** (arrête de faire grandir un arbre quand ça ne sert plus)
3. **Parallélisation** au niveau des noeuds (contrairement au GB classique)
4. **Gestion native des valeurs manquantes**
5. **`scale_pos_weight`** pour le déséquilibre des classes

```
XGBoost = Gradient Boosting
        + Régularisation (évite le surapprentissage)
        + Optimisations système (rapidité)
        + scale_pos_weight (déséquilibre)
```

#### Configuration utilisée (benchmark)

```python
XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric="logloss",       # Fonction de perte : log-loss (cross-entropy)
    verbosity=0,
    scale_pos_weight=5.20        # Ratio négatifs/positifs (3699/711)
)
```

#### Configuration après tuning

```python
# Grille de recherche :
{
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "scale_pos_weight": [1, 3, 5]
}
# → 432 combinaisons, 50 testées (RandomizedSearchCV)

# Meilleurs paramètres trouvés :
{
    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.2,        # Taux d'apprentissage élevé
    "subsample": 0.8,            # 80% des données par arbre
    "scale_pos_weight": 1        # Pas de pondération supplémentaire (déjà gérée)
}
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| **État de l'art** pour les données tabulaires | Plus de paramètres à régler que RF |
| Rapide grâce aux optimisations système | Peut surapprendred si mal configuré |
| `scale_pos_weight` natif | Moins interprétable qu'un arbre unique |
| Régularisation intégrée | |
| Gère les valeurs manquantes | |

#### Résultat dans le projet

```
Benchmark : F1 = 0.989 | AUC = 0.998
Après tuning (seuil=0.69) : CV-F1 = 0.930 | Test-F1 = 0.989
```

---

### 4.8 AdaBoost

#### Principe

**Adaptive Boosting** est un algorithme de boosting qui se concentre itérativement sur les **exemples mal classés** :

```
Itération 1 : Entraîner un classifieur faible (stump) avec des poids uniformes
            → Identifier les exemples mal classés
            → Augmenter le poids de ces exemples

Itération 2 : Entraîner un nouveau classifieur sur les données repondérées
            → Le classifieur se concentre sur les cas difficiles
            → Augmenter encore le poids des erreurs

... (100 itérations)

Prédiction finale = Vote pondéré de tous les classifieurs
```

#### Configuration utilisée

```python
AdaBoostClassifier(
    n_estimators=100,
    random_state=42
    # PAS de class_weight natif
)
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| Se concentre sur les cas difficiles | **Très sensible** au bruit et aux outliers |
| Réduit le biais itérativement | Pas de class_weight natif |
| Combine des classifieurs faibles | Peut surapprendreed sur les outliers |

#### Résultat dans le projet

```
F1 = 0.373 | AUC = 0.773
```

> **Performance la plus faible** : Sans gestion du déséquilibre, AdaBoost se concentre sur les erreurs de la classe majoritaire et ignore la classe minoritaire.

---

### 4.9 MLP (Multi-Layer Perceptron)

#### Principe

Le MLP est un **réseau de neurones artificiels** à couches entièrement connectées :

```
Couche d'entrée     Couche cachée 1    Couche cachée 2    Sortie
(22 features)         (64 neurones)      (32 neurones)    (2 classes)

  x₁ ─────┐
  x₂ ──┐  ├──→ h₁ ──┐
  x₃ ──┼──┤         ├──→ h'₁ ──┐
  ...  ├──┤   h₂ ──┤         ├──→ P(Stay)
  x₂₂ ┘  └──→ ...  └──→ ... ──┘──→ P(Leave)
            (64)        (32)
```

Chaque neurone calcule :
```
sortie = activation(Σ(poids × entrées) + biais)
```

L'activation **ReLU** (Rectified Linear Unit) est utilisée par défaut :
```
ReLU(x) = max(0, x)
```

L'entraînement se fait par **rétro-propagation** : on calcule le gradient de l'erreur par rapport à chaque poids et on ajuste dans la direction qui réduit l'erreur.

#### Configuration utilisée

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 couches cachées : 64 puis 32 neurones
    max_iter=500,                  # Max 500 époques d'entraînement
    random_state=42
    # PAS de class_weight natif
)
```

#### Forces et faiblesses

| Forces | Faiblesses |
|--------|------------|
| Capture des patterns très complexes | **Boîte noire** (peu interprétable) |
| Universel (peut approximer toute fonction) | Sensible au scaling et à l'initialisation |
| Performant sur de grands datasets | Pas de class_weight natif |
| | Lent à entraîner (2.3s vs 0.17s pour RF) |

#### Résultat dans le projet

```
F1 = 0.989 | AUC = 0.992
```

> **Excellente performance** même sans gestion du déséquilibre, grâce à la capacité du réseau à capturer des frontières complexes. Mais son manque d'interprétabilité et l'absence de class_weight font qu'il n'est pas retenu comme modèle final.

---

## 5. Optimisation des hyperparamètres

### 5.1 GridSearchCV

#### Principe

GridSearchCV teste **toutes les combinaisons** possibles de paramètres dans une grille prédéfinie :

```
Grille :
    n_estimators = [100, 200, 300]    → 3 valeurs
    max_depth    = [10, 15, 20, None] → 4 valeurs
    min_samples_split = [2, 5, 10]    → 3 valeurs
    min_samples_leaf  = [1, 2, 4]     → 3 valeurs

Total : 3 × 4 × 3 × 3 = 108 combinaisons
Avec 5-fold CV : 108 × 5 = 540 entraînements
```

Pour chaque combinaison, on fait une **validation croisée 5-fold** et on retient celle avec le meilleur F1 moyen.

#### Quand l'utiliser ?

Quand l'espace de recherche est **petit** (< 200 combinaisons). Au-delà, c'est trop lent.

**Utilisé pour** : Random Forest (144 combinaisons)

### 5.2 RandomizedSearchCV

#### Principe

Quand la grille est trop grande, RandomizedSearchCV teste un **sous-ensemble aléatoire** de combinaisons :

```
Grille :
    n_estimators    = [100, 200, 300]         → 3
    max_depth       = [3, 5, 7, 9]            → 4
    learning_rate   = [0.01, 0.05, 0.1, 0.2]  → 4
    subsample       = [0.7, 0.8, 1.0]         → 3
    scale_pos_weight = [1, 3, 5]              → 3

Total : 3 × 4 × 4 × 3 × 3 = 432 combinaisons
→ On en teste 50 aléatoirement (n_iter=50)
```

#### Quand l'utiliser ?

Quand l'espace de recherche est **grand** (> 200 combinaisons).

**Utilisé pour** : XGBoost (432 combinaisons), Gradient Boosting (243 combinaisons)

#### Pourquoi ça marche ?

Des études (Bergstra & Bengio, 2012) montrent que la recherche aléatoire est **souvent aussi efficace** que la recherche exhaustive, car la plupart des paramètres n'ont qu'un impact marginal. Tester 50 combinaisons aléatoires couvre bien l'espace des paramètres importants.

---

## 6. Calibration du seuil de décision

### Le problème du seuil par défaut

Par défaut, un classifieur prédit "Leave" si `P(Leave) > 0.50`. Mais ce seuil n'est pas forcément optimal :

- **Seuil trop bas** (0.30) → Beaucoup de faux positifs (employés flaggés à tort) → les RH perdent du temps
- **Seuil trop haut** (0.80) → Beaucoup de faux négatifs (départs manqués) → on perd des employés

Le seuil optimal dépend du **compromis précision/rappel** souhaité.

### Méthode utilisée : Calibration par validation croisée

```
Étapes :
1. Faire des prédictions out-of-fold (cross_val_predict, 5 folds)
   → Chaque échantillon du train est prédit par un modèle qui ne l'a pas vu

2. Tester tous les seuils de 0.10 à 0.90 (pas de 0.01)
   → Pour chaque seuil, calculer Precision, Recall et F1

3. Critère de sélection :
   → Maximiser F1
   → Sous contrainte : Recall ≥ 0.60 (on veut détecter au moins 60% des départs)

4. Appliquer le seuil choisi FIXE sur le test set
```

#### Pourquoi cross_val_predict et pas directement sur le test set ?

Si on optimisait le seuil sur le test set, on **adapterait** notre modèle à des données censées être "non vues". C'est du **data leakage** : les métriques de test seraient gonflées artificiellement.

Les prédictions out-of-fold simulent des données non vues car chaque prédiction est faite par un modèle qui **n'a pas été entraîné** sur cet échantillon.

### Seuils optimaux trouvés

| Modèle | Seuil par défaut | Seuil optimal | Gain F1 (CV) |
|--------|:---:|:---:|:---:|
| Random Forest | 0.50 | **0.57** | +0.02 |
| XGBoost | 0.50 | **0.69** | +0.01 |
| Gradient Boosting | 0.50 | **0.45** | +0.05 |

---

## 7. Validation croisée stratifiée

### Principe

La validation croisée (CV) divise le train set en **K plis** (folds) et entraîne K fois :

```
Fold 1 : [TEST] [Train] [Train] [Train] [Train]
Fold 2 : [Train] [TEST] [Train] [Train] [Train]
Fold 3 : [Train] [Train] [TEST] [Train] [Train]
Fold 4 : [Train] [Train] [Train] [TEST] [Train]
Fold 5 : [Train] [Train] [Train] [Train] [TEST]

Score final = Moyenne des 5 scores
```

Le mot **stratifié** signifie que chaque fold maintient le **même ratio de classes** que le dataset complet (16% Leave / 84% Stay). Sans stratification, un fold pourrait avoir 0% de Leave par hasard.

### Configuration utilisée

```python
StratifiedKFold(
    n_splits=5,       # 5 plis
    shuffle=True,     # Mélanger les données avant de diviser
    random_state=42   # Reproductibilité
)
```

### Pourquoi c'est important ?

Le CV-F1 est une estimation **non biaisée** de la performance réelle du modèle. Le Test-F1 peut être optimiste ou pessimiste selon la chance du split. Le CV-F1 est plus fiable car il moyenne sur 5 évaluations différentes.

---

## 8. Métriques d'évaluation

### Matrice de confusion

```
                      Prédit Stay    Prédit Leave
Réel Stay (négatif)      VN             FP          ← Faux Positif (fausse alarme)
Réel Leave (positif)     FN             VP          ← Faux Négatif (départ manqué)
```

### Les 5 métriques utilisées

#### 1. Accuracy (exactitude)
```
Accuracy = (VP + VN) / (VP + VN + FP + FN)
         = Prédictions correctes / Total
```
> **Piège** : 84% d'accuracy en prédisant toujours "Stay". Inutile seule sur des données déséquilibrées.

#### 2. Precision (précision)
```
Precision = VP / (VP + FP)
          = "Parmi ceux que je dis Leave, combien partent vraiment ?"
```
> **Haute precision** = Peu de fausses alarmes → les RH ne perdent pas de temps sur des cas non pertinents.

#### 3. Recall (rappel / sensibilité)
```
Recall = VP / (VP + FN)
       = "Parmi ceux qui partent vraiment, combien ai-je détectés ?"
```
> **Haut recall** = Peu de départs manqués → on n'est pas surpris par un départ non anticipé.

#### 4. F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Moyenne harmonique de Precision et Recall
```
> La **moyenne harmonique** pénalise les écarts : si Precision=1.0 et Recall=0.1, F1=0.18 (pas 0.55). Elle force un **équilibre** entre les deux.

#### 5. AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
```
AUC = Aire sous la courbe ROC

Courbe ROC : Taux de Vrais Positifs (Recall) vs Taux de Faux Positifs
             pour tous les seuils possibles de 0 à 1
```

| AUC | Interprétation |
|:---:|----------------|
| 1.0 | Discrimination parfaite |
| 0.9+ | Excellente |
| 0.8-0.9 | Bonne |
| 0.7-0.8 | Acceptable |
| 0.5 | Aléatoire (pas mieux qu'un pile ou face) |

---

## 9. Méthodes d'explicabilité

### 9.1 SHAP (SHapley Additive exPlanations)

#### Origine : Théorie des jeux

Les **valeurs de Shapley** viennent de la théorie des jeux coopératifs. Elles répondent à la question : "Quelle est la contribution **équitable** de chaque joueur au résultat ?"

Transposé au ML : "Quelle est la contribution de chaque **feature** à la prédiction ?"

#### Comment ça fonctionne ?

Pour une prédiction donnée, SHAP calcule la contribution marginale de chaque feature en la retirant de toutes les combinaisons possibles de features :

```
Prédiction complète (22 features) : P(Leave) = 0.82
Sans Age :                          P(Leave) = 0.75
Sans Age et Salary :                P(Leave) = 0.60
Sans Age mais avec Salary :         P(Leave) = 0.78

Contribution de Age = Moyenne de toutes les contributions marginales
                    = Moyenne(0.82-0.75, 0.78-0.60, ...) ≈ +0.09
```

> En pratique, calculer toutes les combinaisons est exponentiel (2²² pour 22 features). **TreeExplainer** utilise des propriétés des arbres pour un calcul exact en temps polynomial.

#### Configuration utilisée

```python
explainer = shap.TreeExplainer(best_model)          # Exploite la structure des arbres
X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_sample)       # Valeurs SHAP pour 500 individus

shap.summary_plot(shap_vals, X_sample, max_display=20)  # Top 20 features
```

#### Ce que SHAP produit

- **Summary plot** : Vue globale — chaque point est un individu, la couleur représente la valeur de la feature (rouge=élevée, bleu=faible), la position horizontale représente l'impact SHAP
- **Force plot** : Vue individuelle — comment chaque feature pousse la prédiction vers "Stay" ou "Leave"
- **Importance globale** : Moyenne des |SHAP| par feature → ranking des features

### 9.2 LIME (Local Interpretable Model-Agnostic Explanations)

#### Principe

LIME explique **une prédiction à la fois** en créant un **modèle linéaire local** :

```
1. Prendre un individu à expliquer (ex: Employé #1234)
2. Générer des "voisins" artificiels en perturbant ses features
3. Prédire ces voisins avec le modèle complexe (Random Forest)
4. Entraîner un modèle LINÉAIRE sur ces voisins
5. Les coefficients de ce modèle linéaire = les contributions locales
```

```
    Modèle complexe (RF)              LIME : Modèle linéaire local
    ┌─────────────────┐               autour de l'individu
    │   /\    /\      │
    │  /  \  /  \     │     →     y ≈ β₁×Age + β₂×Salary + ...
    │ /    \/    \    │
    └─────────────────┘             β₁ = -0.18 → Age pousse vers Leave
                                     β₂ = +0.12 → Salary pousse vers Stay
```

#### Configuration utilisée

```python
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_test.columns),
    class_names=["Stay", "Leave"],
    mode="classification",
    discretize_continuous=True,      # Discrétise les features continues pour la lisibilité
    random_state=42
)

exp = explainer_lime.explain_instance(
    X_test.iloc[idx].values,
    best_model.predict_proba,
    num_features=10,                 # Afficher les 10 features les plus influentes
    top_labels=1
)
```

#### SHAP vs LIME — Comparaison

| Critère | SHAP | LIME |
|---------|------|------|
| **Scope** | Global + local | **Local uniquement** |
| **Théorie** | Fondé mathématiquement (Shapley) | Approximation heuristique |
| **Fidélité** | Exacte (pour les arbres) | Approximative |
| **Vitesse** | Rapide (TreeExplainer) | Plus lent (rééchantillonnage) |
| **Usage** | Ranking global des features | Explication d'un cas individuel |
| **Avantage clé** | Propriétés théoriques (additivité, symétrie, nullité) | Agnostique au modèle (fonctionne avec tout) |

---

## 10. Métriques d'équité

### Pourquoi l'équité est essentielle ?

Un modèle qui prédit le départ des employés peut être utilisé pour des **décisions RH** : qui retenir, qui promouvoir, qui surveiller. Si le modèle discrimine sur le genre, l'âge ou le statut marital, il amplifie les biais existants et peut violer la loi.

### 3 métriques évaluées

#### 1. Disparate Impact (DI)

```
DI = (Taux de sélection du groupe défavorisé) / (Taux de sélection du groupe favorisé)

Exemple :
    Taux de prédiction "Leave" pour les femmes : 16%
    Taux de prédiction "Leave" pour les hommes : 17.4%
    DI = 16% / 17.4% = 0.918
```

| DI | Interprétation |
|:---:|---|
| ≥ 0.80 | Acceptable (règle des 4/5 du droit du travail américain) |
| < 0.80 | **Biais potentiel** → investigation nécessaire |
| = 1.00 | Parité parfaite |

#### 2. Equal Opportunity (Égalité des chances)

```
ΔTPR = |Recall(groupe A) - Recall(groupe B)|

"Le modèle détecte-t-il aussi bien les départs dans chaque groupe ?"
```

| ΔTPR | Interprétation |
|:---:|---|
| < 0.10 | Acceptable |
| ≥ 0.10 | **Inéquité** → un groupe est moins bien détecté |

#### 3. Predictive Parity (Parité prédictive)

```
ΔPPV = |Precision(groupe A) - Precision(groupe B)|

"Quand le modèle dit 'Leave', est-ce aussi fiable pour chaque groupe ?"
```

| ΔPPV | Interprétation |
|:---:|---|
| < 0.10 | Acceptable |
| ≥ 0.10 | **Inéquité** → les alertes sont moins fiables pour un groupe |

### Résultats dans le projet

| Variable | DI | Equal Opportunity | Predictive Parity |
|----------|:---:|:---:|:---:|
| Genre | 0.918 | OK | OK |
| Statut marital | **0.748** | Violations | OK |
| Âge | Violations multiples | Violations (jusqu'à 0.115) | OK |

### Mitigation : Seuils différenciés

Pour corriger le biais sur le statut marital, on applique des **seuils de décision différents** par groupe, calibrés par validation croisée :

```
Divorcé  → seuil = 0.11 (seuil bas = plus de flagging pour compenser le sous-flagging)
Marié    → seuil = 0.14
Célibataire → seuil = 0.68 (seuil élevé = moins de flagging pour compenser le sur-flagging)
```

**Résultat** : DI passe de 0.748 à **0.801** (au-dessus du seuil de 0.80), mais le F1 baisse de 0.931 à 0.810. C'est le **compromis équité-performance**.

---

## 11. Tests de robustesse

### Tests d'ablation

On retire des **groupes de features** et on mesure l'impact sur la performance :

| Scénario | Description | CV-F1 | Δ vs S0 |
|----------|-------------|:---:|:---:|
| S0 | Baseline (22 features, badge H1) | ~0.930 | — |
| S1 | Sans features badge | ~0.926 | -0.004 |
| S2 | Badge année complète (au lieu de H1) | ~0.928 | -0.002 |
| S3 | Top 25 features manuelles | ~0.926 | -0.004 |
| S4 | Minimal (15 features RH uniquement) | ~0.920 | -0.010 |

> **Conclusion** : Les features badge contribuent modestement (+0.4% de F1). Le signal principal vient des données RH et des enquêtes de satisfaction.

### Injection de bruit gaussien

On perturbe les features continues avec du bruit croissant pour tester si le modèle **résiste aux imprécisions** des données :

```
Pour chaque niveau de bruit σ ∈ {0.05, 0.10, 0.15, 0.20, 0.25} :
    Pour chaque seed ∈ {1, 2, ..., 10} :
        X_perturbed = X_test + N(0, σ)    (bruit gaussien)
        Calculer F1(X_perturbed)
    F1_moyen(σ) = Moyenne des 10 seeds
```

| Métrique | Valeur |
|----------|--------|
| F1 baseline | 0.978 |
| F1 à 5% de dégradation | 0.930 |
| F1 à 15% de dégradation | 0.832 |

> Le modèle maintient sa performance sous **perturbation modérée**, ce qui est un signe de robustesse.

---

## 12. Résumé des configurations exactes

### Prétraitement

| Composant | Configuration exacte |
|-----------|---------------------|
| Badge cutoff | `pd.Timestamp("2015-06-30")` — H1 2015 uniquement |
| Seuil arrivée tardive | `in_hours >= 10` (10h00) |
| KNNImputer | `n_neighbors=5, weights="distance"` |
| StandardScaler | Défaut (`mean=0, std=1`) |
| Train/test split | `test_size=0.20, stratify=y, random_state=42` |

### Sélection de features

| Paramètre | Valeur |
|-----------|--------|
| Seuil corrélation (cible) | `|r| >= 0.03` |
| Seuil importance Gini cumulée | `82%` |
| Logique de combinaison | Union (OU) |
| Seuil de variance | `< 0.01` → suppression |
| Seuil de multicolinéarité | `|r| > 0.90` → suppression |
| RF pour Gini | `n_estimators=300, class_weight="balanced", random_state=42` |

### Modèles (benchmark)

| Modèle | Paramètres exacts |
|--------|-------------------|
| Logistic Regression | `max_iter=1000, random_state=42, class_weight="balanced"` |
| KNN | `n_neighbors=5, n_jobs=-1` |
| SVM (RBF) | `kernel="rbf", probability=True, random_state=42, class_weight="balanced"` |
| Decision Tree | `random_state=42, class_weight="balanced"` |
| Random Forest | `n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"` |
| Gradient Boosting | `n_estimators=100, random_state=42` |
| XGBoost | `n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0, scale_pos_weight=5.20` |
| AdaBoost | `n_estimators=100, random_state=42` |
| MLP | `hidden_layer_sizes=(64, 32), max_iter=500, random_state=42` |

### Optimisation

| Paramètre | Valeur |
|-----------|--------|
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| GridSearchCV (RF) | 144 combinaisons, scoring=F1 |
| RandomizedSearchCV (XGB, GB) | `n_iter=50`, scoring=F1 |
| Seuil search range | `np.arange(0.10, 0.91, 0.01)` |
| Contrainte recall | `>= 0.60` |

### Explicabilité

| Composant | Configuration exacte |
|-----------|---------------------|
| SHAP | `TreeExplainer(best_model)`, sample=500, `max_display=20` |
| LIME | `discretize_continuous=True`, `num_features=10`, `top_labels=1` |

### Modèle final retenu

| Paramètre | Valeur |
|-----------|--------|
| Algorithme | **Random Forest** |
| Seuil | **0.57** |
| CV-F1 | **0.931** |
| Test-F1 | **0.978** |
| Precision (test) | **1.000** |
| Recall (test) | **0.958** |

---

*Document technique — Projet HumanForYou, CESI A5*
