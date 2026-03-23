import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

# 1. Carregar o dataset obrigatório 
df = pd.read_csv('student_records_missing.csv')

# 2. Definir as colunas
numeric_features = ['study_hours', 'sleep_hours', 'caffeine_intake_mg', 'mental_health_score']
nominal_features = ['gender', 'academic_level']
ordinal_features = ['internet_quality']
targets = ['Focus Index', 'Burnout Level', 'Productivity Score', 'Exam Score']

# 3. Separar X e os dois y (Regressão e Classificação) [cite: 14, 16]
X = df.drop(columns=targets)
y_reg = df['Exam Score']
y_clf = df['Burnout Level']

# 4. Construir o Pipeline de Pré-processamento 
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, nominal_features + ordinal_features)
])

# 5. Executar as Baselines com Validação Cruzada [cite: 22, 43]
# Regressão Linear (Target: Exam Score)
pipe_reg = Pipeline(steps=[('pre', preprocessor), ('model', LinearRegression())])
cv_reg = cross_validate(pipe_reg, X, y_reg, cv=5, scoring=['neg_mean_absolute_error', 'r2'])

# Árvore de Decisão (Target: Burnout Level)
pipe_clf = Pipeline(steps=[('pre', preprocessor), ('model', DecisionTreeClassifier(max_depth=3, random_state=42))])
cv_clf = cross_validate(pipe_clf, X, y_clf, cv=5, scoring=['accuracy', 'f1_weighted'])

# 6. Mostrar Resultados
print(f"--- RESULTADOS BASELINE ---")
print(f"Regressão (MAE): {-cv_reg['test_neg_mean_absolute_error'].mean():.3f}")
print(f"Classificação (F1-Score): {cv_clf['test_f1_weighted'].mean():.3f}")