import pickle
import asyncio
from datetime import datetime, timedelta, date
from river import tree
import pandas as pd
import os
from river.drift import ADWIN
from pymodbus.client import AsyncModbusTcpClient
import logging

MODBUS_HOST = 'localhost'
MODBUS_PORT = 5020
REGISTER_ADDR_SOC = 0  # Registro inicio para SoC, hora y día simulados
REGISTER_ADDR_RED = 3  # Registro conexión red (escritura)
SOC_MAX_KWH = 240  # 100% SoC en kWh

# Ruta al modelo preentrenado
MODEL_PATH = "modelo_hoeffding_tree.pkl"

# Para transformar filas en features River
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

class ClienteModbus:
    def __init__(self):
        self.estado_red = False
        self.modelo = None
        self.df = pd.read_csv('data/ev_demand_test_2023_corr.csv', parse_dates=['timestamp'])
        self.current_day_simulado = None
        self.prediccion_demanda_dia_siguiente = 0
        self.detector_drift = ADWIN()
        self.historico_csv_dir = 'data/historico_dias'
        os.makedirs(self.historico_csv_dir, exist_ok=True)
    async def conectar(self):
        self.client = AsyncModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
        await self.client.connect()

    async def cerrar(self):
        await self.client.close()

    async def toggle_red(self, conectar):
        if conectar != self.estado_red:
            self.estado_red = conectar
            valor = 1 if conectar else 0
            await self.client.write_register(REGISTER_ADDR_RED, valor)
            print(f"Red {'conectada' if conectar else 'desconectada'}")

    async def leer_soc_hora_dia(self):
        rr = await self.client.read_holding_registers(REGISTER_ADDR_SOC, 7)
        if rr.isError():
            return None, None, None
        regs = rr.registers
        soc = regs[0]
        hora_simulada = regs[4]
        fecha_high = regs[5]
        fecha_low = regs[6]
        fecha_int = (fecha_high << 16) + fecha_low
        fecha_str = str(fecha_int).zfill(8)  #8 dígitos, ej. '20250607'

        try:
            año = int(fecha_str[:4])
            mes = int(fecha_str[4:6])
            dia = int(fecha_str[6:8])
            fecha = date(año, mes, dia)
        except Exception:
            fecha = None

        return soc, hora_simulada, fecha  # ya devuelve date directamente

    async def entrenar_y_predecir(self, dia):
        logging.basicConfig(
            filename='modbus_client.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Guardamos los datos del día en un CSV
        df_dia = self.df[self.df['timestamp'].dt.date == dia]
        if not df_dia.empty:
            df_dia.to_csv(f"{self.historico_csv_dir}/{dia}.csv", index=False)

        # Si hay drift, hacemos soft reset entrenando solo con los últimos 30 días
        if self.detector_drift.drift_detected:
            
            logging.info(f"Drift detectado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("⚠️ DRIFT DETECTADO: reentrenando modelo con últimos 30 días")
            modelo = tree.HoeffdingTreeRegressor()
            ultimos_30 = self.df[self.df['timestamp'].dt.date > (dia - timedelta(days=30))]
            for _, row in ultimos_30.iterrows():
                x = row_to_features(row)
                y = row['energia_kWh']
                modelo.learn_one(x, y)
            self.detector_drift = ADWIN()  # Reiniciamos el detector
        else:
            # Cargamos modelo existente si no hay drift
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    modelo = pickle.load(f)
            else:
                modelo = tree.HoeffdingTreeRegressor()

            # Entrenamos con el día actual
            for _, row in df_dia.iterrows():
                x = row_to_features(row)
                y = row['energia_kWh']
                modelo.learn_one(x, y)
           
            
        # Predecimos la demanda del día siguiente
        dia_siguiente = dia + timedelta(days=1)
        df_siguiente = self.df[self.df['timestamp'].dt.date == dia_siguiente]

        demanda_total_real = df_siguiente['energia_kWh'].sum()
        demanda_total_predicha = 0

        for _, row in df_siguiente.iterrows():
            x = row_to_features(row)
            y_pred = modelo.predict_one(x)
            demanda_total_predicha += y_pred or 0

        # Evaluamos el error y lo pasamos al detector de drift
        if demanda_total_real != 0:
            error_relativo = abs(demanda_total_real - demanda_total_predicha) / demanda_total_real
        else:
            error_relativo = 0  # Evitamos división por cero

        self.detector_drift.update(error_relativo)
        # Guardamos el modelo actualizado
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(modelo, f)

        self.modelo = modelo
        self.prediccion_demanda_dia_siguiente = demanda_total_predicha
        print(f"Demanda real: {demanda_total_real:.2f} kWh")
        print(f"Demanda estimada para {dia_siguiente}: {demanda_total_predicha:.2f} kWh")
        print(f"Error relativo: {error_relativo:.2%}")
        
        print(f"Entrenando modelo con datos del día {dia}")
        
        logging.info(f"Demanda real: {demanda_total_real:.2f} kWh")
        logging.info(f"Demanda estimada para {dia_siguiente}: {demanda_total_predicha:.2f} kWh")
        logging.info(f"Error relativo: {error_relativo:.2%}")
    async def run(self):
        await self.conectar()
        print("Cliente Modbus iniciado")

        while True:
            soc, hora_simulada, fecha_simulada = await self.leer_soc_hora_dia()
            if soc is None or hora_simulada is None or fecha_simulada is None:
                print("Error leyendo SoC, hora simulada o fecha simulada")
                await asyncio.sleep(3)
                continue

            # Usamos la fecha simulada que viene del servidor directamente
            if self.current_day_simulado != fecha_simulada:
                if self.current_day_simulado is not None:
                    await self.entrenar_y_predecir(self.current_day_simulado)
                self.current_day_simulado = fecha_simulada

            if self.prediccion_demanda_dia_siguiente == 0:
                await self.toggle_red(False)
            else:
                soc_kwh = soc / 100 * SOC_MAX_KWH
                conectar_red = (soc_kwh < self.prediccion_demanda_dia_siguiente + (0.15*SOC_MAX_KWH) and (hora_simulada < 8) and soc_kwh < SOC_MAX_KWH*0.9)
                await self.toggle_red(conectar_red)

            print(f"Fecha simulada: {fecha_simulada} - Hora simulada: {hora_simulada}:00 - SoC: {soc}% - Red {'ON' if self.estado_red else 'OFF'}")
            await asyncio.sleep(2)  # 2 segundos = 1 hora simulada

if __name__ == "__main__":
    cliente = ClienteModbus()
    asyncio.run(cliente.run())