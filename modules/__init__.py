"""
Batería Lab - Paquete de Módulos
================================
Este paquete contiene los módulos del simulador de degradación de baterías.
"""

from .utils import (
    get_data_path,
    time_formatter,
    aplicar_estilo_dark,
    mostrar_kpi_html,
    aplicar_estilos_globales,
    Colores
)

from .data_manager import (
    cargar_db_modelos,
    guardar_db_modelos,
    cargar_db_packs,
    guardar_db_packs,
    inicializar_session_state
)

from .physics import (
    obtener_funcion_ocv_polinomica,
    generar_perfil_potencia_unificado,
    calcular_soc_final_para_umbral,
    simular_con_umbral_adaptativo,
    CONSTANTES_VEHICULO,
    CONSTANTES_ELECTRICAS
)

from .optimization import (
    buscar_umbral_optimo
)

__all__ = [
    # Utils
    'get_data_path', 'time_formatter', 'aplicar_estilo_dark', 
    'mostrar_kpi_html', 'aplicar_estilos_globales', 'Colores',
    # Data Manager
    'cargar_db_modelos', 'guardar_db_modelos', 'cargar_db_packs', 
    'guardar_db_packs', 'inicializar_session_state',
    # Physics
    'obtener_funcion_ocv_polinomica', 'generar_perfil_potencia_unificado',
    'calcular_soc_final_para_umbral', 'simular_con_umbral_adaptativo',
    'CONSTANTES_VEHICULO', 'CONSTANTES_ELECTRICAS',
    # Optimization
    'buscar_umbral_optimo'
]
