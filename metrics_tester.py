import pandas as pd
import numpy as np
from datetime import datetime
from river import linear_model, preprocessing, metrics, tree
from river.utils import rolling
import holidays

# Cargar datos simulados
df_future = pd.read_csv("data/ev_demand_train_2015_2024_strict.csv", parse_dates=["timestamp"])

# Preparar días festivos para España
es_holidays = holidays.Spain(years=range(2025, 2036))

# Función para extraer características
def extract_features(row):
    dt = row["timestamp"]
    return {
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "is_weekend": dt.weekday() >= 5,
        "is_holiday": dt.date() in es_holidays,
        "day_of_year": dt.timetuple().tm_yday
    }

# Aplicar extracción de características
df_future["features"] = df_future.apply(extract_features, axis=1)

model = preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()
metric = metrics.MAE()

predictions = []
errors = []
saved_cycles = 0
threshold_kWh = 144  

previous_prediction = 0
accumulated_diff = 0

# Entrenar en línea con datos simulados y guardar predicciones por día
df_future["date"] = df_future["timestamp"].dt.date
daily_pred = {}
for i, row in df_future.iterrows():
    x = row["features"]
    y = row["energia_kWh"]

    y_pred = model.predict_one(x) or 0
    model.learn_one(x, y)

    predictions.append(y_pred)
    errors.append(abs(y - y_pred))
    metric.update(y, y_pred)

    diff = y_pred - previous_prediction
    accumulated_diff += abs(diff)
    previous_prediction = y_pred

    # Acumular predicción por día
    fecha = row["date"]
    if fecha not in daily_pred:
        daily_pred[fecha] = 0
    daily_pred[fecha] += y_pred

# Agregar predicciones y errores al dataframe
df_future["prediction"] = predictions
df_future["abs_error"] = errors

# Calcular ciclos ahorrados al final de cada día
saved_cycles = 0
dates = sorted(daily_pred.keys())
for idx in range(1, len(dates)):
    gasto_hoy = daily_pred[dates[idx]]
    gasto_ayer = df_future[df_future["date"] == dates[idx - 1]]["energia_kWh"].sum()
    diff = gasto_ayer - gasto_hoy
    if diff > 0:
        saved_cycles += diff / threshold_kWh

# Resultados
mae = metric.get()
total_saved_cycles = saved_cycles

# Mostrar resultados clave
print(f"MAE: {mae:.2f} kWh")
daily_true = df_future.groupby("date")["energia_kWh"].sum()
daily_pred_series = pd.Series(daily_pred)
daily_mae = np.mean(np.abs(daily_true - daily_pred_series))
print(f"MAE diario: {daily_mae:.2f} kWh")
print(f"Total saved cycles: {total_saved_cycles:.2f}")
# Calcular gasto medio por día laborable
laborable_days = [date for date in daily_pred if datetime.strptime(str(date), "%Y-%m-%d").weekday() < 5 and date not in es_holidays]
if laborable_days:
    mean_laborable_consumption = np.mean([daily_pred[date] for date in laborable_days])
    print(f"Gasto medio por día laborable: {mean_laborable_consumption:.2f} kWh")
else:
    print("No hay días laborables en el periodo analizado.")