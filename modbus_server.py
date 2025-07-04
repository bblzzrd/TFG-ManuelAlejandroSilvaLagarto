import asyncio
import pandas as pd
from pymodbus.server import ModbusTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock
from pymodbus.device import ModbusDeviceIdentification

# Carga el CSV de demanda
df = pd.read_csv('data/ev_demand_test_2023_corr.csv', parse_dates=['timestamp'])

# Parámetros
SOC_MAX_KWH = 240  # 100% SoC en kWh
SIM_SECONDS_PER_HOUR = 2  # 2 segundos = 1 hora simulada

# Inicializa la base de datos: 10 registros holding
# hr[0]: SoC en %; hr[1]: demanda en kW; hr[3]: conexión red (0/1)
store = ModbusSlaveContext(
    hr=ModbusSequentialDataBlock(0, [50, 0, 0, 1] + [0]*6),  # SoC inicial 50%
)
context = ModbusServerContext(slaves=store, single=True)

# Información del dispositivo
identity = ModbusDeviceIdentification()
identity.VendorName = 'SimuladorESS'
identity.ProductCode = 'ESS123'
identity.ProductName = 'BatteryESS'
identity.ModelName = 'ESSv1'
identity.MajorMinorRevision = '1.0'

async def update_registers():
    print("Servidor Modbus iniciado correctamente.")
    dia_actual = None

    for index, row in df.iterrows():
        timestamp = row["timestamp"]
        energia_kWh = row["energia_kWh"]
        demanda_kw = energia_kWh
        is_holiday = bool(row["is_holiday"])

        # Mostrar nuevo día si cambia
        fecha = timestamp.date()
        if fecha != dia_actual:
            dia_actual = fecha
            festivo_str = "Festivo" if is_holiday else "Laborable"
            print(f"\n== Día {fecha} ({festivo_str}) ==\n")

        # Obtener estado actual
        soc_percent = context[0x00].getValues(3, 0, count=1)[0]
        soc_kWh = soc_percent / 100 * SOC_MAX_KWH
        hr = context[0x00].getValues(3, 0, count=6)
        red_conectada = hr[3] == 1

        # Cálculo del nuevo SoC
        carga = 45 if red_conectada else 0
        soc_kWh = min(SOC_MAX_KWH, max(0, soc_kWh + carga - energia_kWh))
        soc_percent = soc_kWh / SOC_MAX_KWH * 100

        # Actualiza registros: SoC, demanda, reservado, red conectada, hora simulada, día simulada
        hour_index = timestamp.hour
        fecha_int = int(fecha.strftime("%Y%m%d"))  # ej: 20250607
        # División en dos registros de 16 bits
        fecha_high = (fecha_int >> 16) & 0xFFFF
        fecha_low = fecha_int & 0xFFFF
        context[0x00].setValues(3, 0, [
            int(soc_percent),
            int(demanda_kw),
            0,
            int(red_conectada),
            hour_index % 24,
            fecha_high,
            fecha_low
        ])

        hora_simulada = timestamp.strftime("%H:%M")
        print(f"[{hora_simulada}] SoC: {soc_percent:.1f}% ({soc_kWh:.1f} kWh), "
              f"Demanda: {demanda_kw:.1f} kW, Red conectada: {red_conectada}, Día: {fecha_int}")

        await asyncio.sleep(SIM_SECONDS_PER_HOUR)

async def main():
    server = ModbusTcpServer(context, identity=identity, address=("localhost", 5020))
    print("Servidor Modbus iniciado en localhost:5020")
    await asyncio.gather(
        server.serve_forever(),
        update_registers()
    )

if __name__ == "__main__":
    asyncio.run(main())