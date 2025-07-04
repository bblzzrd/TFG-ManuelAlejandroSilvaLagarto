import pickle
import pandas as pd
from river import tree

# Cargar datos
df = pd.read_csv('data/ev_demand_train_2020_2022_corr.csv', parse_dates=['timestamp'])

# Añadir features adicionales directamente en el DataFrame
df['is_charging_hour'] = df['hour'].between(9, 11).astype(int)
df['day_type'] = df.apply(
    lambda row: 'holiday' if row['is_holiday'] else ('weekend' if row['is_weekend'] else 'working_day'),
    axis=1
)

# Convertimos 'day_type' en variables dummy (one-hot)
df = pd.get_dummies(df, columns=['day_type'], drop_first=False)

# Definir función de extracción de features
def row_to_features(row):
    return {
        'hour': row.hour,
        'day_of_week': row.day_of_week,
        'month': row.month,
        'is_charging_hour': row.is_charging_hour,
        'day_type_holiday': row.day_type_holiday,
        'day_type_weekend': row.day_type_weekend,
        'day_type_working_day': row.day_type_working_day
    }

# Configurar un árbol Hoeffding más propenso a dividir
modelo = tree.HoeffdingTreeRegressor(
    grace_period=10,
    leaf_prediction='mean',
    max_depth=None,
    min_samples_split=5,  
    delta=1e-7             
)

print("Entrenando modelo con todo el dataset...")

# Entrenamiento eficiente usando itertuples
for row in df.itertuples(index=False):
    x = row_to_features(row)
    y = row.energia_kWh
    modelo.learn_one(x, y)

print("Entrenamiento completado.")

# Guardar modelo entrenado
with open("modelo_hoeffding_tree.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("Modelo guardado en 'modelo_hoeffding_tree.pkl'")