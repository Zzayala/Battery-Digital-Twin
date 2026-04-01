import sys
import os
sys.path.append(r"C:\Temp_Analisis\Bateria12V")
from Simulador_Degradacion12V import simular_endurance

# Configuración 3S2P (95-35)
print("Ejecutando 3S2P (95% -> 35%)...")
resultados_3s2p = simular_endurance(
    n_series=3, 
    n_paralelo=2, 
    soc_inicial=95.0, 
    soc_minimo=35.0, 
    soh_inicial=100.0, 
    temp_amb=42.5
)
print(f"3S2P - Ciclos estimados: {resultados_3s2p['ciclos_estimados']}")

# Configuración 4S2P (80-20)
print("\nEjecutando 4S2P (80% -> 20%)...")
resultados_4s2p = simular_endurance(
    n_series=4, 
    n_paralelo=2, 
    soc_inicial=80.0, 
    soc_minimo=20.0, 
    soh_inicial=100.0, 
    temp_amb=42.5
)
print(f"4S2P - Ciclos estimados: {resultados_4s2p['ciclos_estimados']}")
