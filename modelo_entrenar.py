# modelo_entrenar.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ------------ 1. Cargar datos ------------
DATA_PATH = "data_filtrada_imputada.csv"  # Ajusta si tu ruta es distinta
df = pd.read_csv(DATA_PATH)

# ------------ 2. Definir variables ------------
# Asegúrate que 'formalidad' sea binaria (0 = informal, 1 = formal)
df['formalidad'] = df['formalidad'].map({0: 0, 1: 1})

# Selección de variables predictoras
predictoras = [
    'Sexo',
    'Edad',
    'Mayor nivel educativo alcanzado',
    'Estado civil',
    'AREA',
    'Tipo de ocupación',
    'Para la ocupación que desempeña, ¿tiene algún tipo de contrato',
    '¿Es propietario de una o varias propiedades inmuebles?',
    'Tiempo de desplazamiento hasta su sitio de trabajo',
    'Meses que estuvo sin empleo entre su trabajo actual y el anterior'
]

# Eliminar filas con valores nulos en variables seleccionadas
df_modelo = df[predictoras + ['formalidad']].dropna()

X = pd.get_dummies(df_modelo[predictoras], drop_first=True)
y = df_modelo['formalidad']

# ------------ 3. Dividir en entrenamiento y prueba ------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------ 4. Entrenar modelo ------------
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# ------------ 5. Evaluación ------------
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

print("\n📊 MÉTRICAS DEL MODELO")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))

# ------------ 6. Coeficientes del modelo ------------
print("\n📌 COEFICIENTES DEL MODELO")
coeficientes = pd.Series(modelo.coef_[0], index=X.columns)
print(coeficientes.sort_values(ascending=False))

# ------------ 7. Guardar modelo ------------
joblib.dump(modelo, "modelo_formalidad.pkl")
joblib.dump(X.columns, "columnas_modelo.pkl")  # Para usarlo en la app luego
print("\n✅ Modelo y columnas guardados exitosamente.")
