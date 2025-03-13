
# Fiche Récap – TD Machine Learning en Python

Cette fiche résume les étapes du TD : de la préparation des données à l'entraînement de plusieurs modèles.

---

## 1. Chargement des bibliothèques

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## 2. Chargement et exploration des données

```python
df = pd.read_csv("fichier.csv")
df.head()
df.describe()
df.info()
df["colonne"].value_counts()
```

- Moyenne : `df.mean(numeric_only=True)`
- Écart-type : `df.std(numeric_only=True)`

---

## 3. Nettoyage et préparation des données

```python
# Supprimer des colonnes
df = df.drop(columns=["id", "nom_colonne"])

# Imputation des valeurs manquantes
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(df.drop(columns=["target"]))

# Encodage d'une variable catégorielle
enc = LabelEncoder()
y = enc.fit_transform(df["target"])
```

---

## 4. Séparation Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## 5. Standardisation

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Entraînement de modèles

```python
# Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Autres modèles :

```python
LogisticRegression(), KNeighborsClassifier(), MLPClassifier()
```

---

## 7. Évaluation

```python
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Matrice de confusion

|               | Prédit 0 | Prédit 1 |
|---------------|----------|----------|
| Réel 0        | VN       | FP       |
| Réel 1        | FN       | VP       |

---

## 8. Optimisation

```python
from sklearn.model_selection import GridSearchCV

params = {"n_estimators": [50, 100], "max_depth": [None, 5]}
grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## 9. Visualisations utiles

```python
sns.histplot(df["colonne"])
sns.boxplot(x="groupe", y="valeur", data=df)
sns.heatmap(df.corr(), annot=True)
sns.pairplot(df, hue="target")
```

---
