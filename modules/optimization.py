"""
Módulo de Optimización
======================
Contiene la lógica del auto-ajuste y búsqueda de umbrales óptimos.

Funciones:
- buscar_umbral_optimo(): Encuentra el umbral de aceleración que maximiza
  el uso de la batería llegando exactamente al SOC mínimo.
"""

import numpy as np

from .physics import calcular_soc_final_para_umbral


# =============================================================================
# BÚSQUEDA DE UMBRAL ÓPTIMO
# =============================================================================

def buscar_umbral_optimo(soc_minimo, acc_telem, v_ms, n_vueltas, params):
    """
    Busca el umbral más bajo posible que mantenga SOC final >= soc_minimo.
    
    Objetivo: Gastar toda la batería permitida, llegando justo al SOC mínimo.
    
    Lógica:
    - Umbral BAJO = más consumo = SOC final MÁS BAJO
    - Umbral ALTO = menos consumo = SOC final MÁS ALTO
    
    Buscamos el umbral más bajo donde SOC final >= soc_minimo
    
    Args:
        soc_minimo: SOC mínimo objetivo (%)
        acc_telem: Array de aceleraciones de telemetría (m/s²)
        v_ms: Array de velocidades (m/s)
        n_vueltas: Número de vueltas a simular
        params: Diccionario con todos los parámetros físicos necesarios
    
    Returns:
        tuple: (umbral_optimo, soc_final_optimo, log_pruebas)
               El log es una lista de dicts con info de debug para UI
    """
    UMBRAL_MINIMO = 0.0
    UMBRAL_MAXIMO = 13.0
    
    # Rango objetivo: [SOC_Min, SOC_Min + 2%]
    soc_target_min = soc_minimo
    soc_target_max = soc_minimo + 2.0
    
    # Log para debug
    log_pruebas = []
    
    # --- 1. Verificar extremos primero ---
    
    # Caso A: ¿Qué pasa si ayudamos SIEMPRE (Umbral 0)? (Máximo consumo posible)
    soc_con_umbral_min = calcular_soc_final_para_umbral(
        UMBRAL_MINIMO, acc_telem, v_ms, n_vueltas, **params
    )
    # Si gastando al máximo cumplimos el mínimo (nos sobra o estamos justo), el 0.0 es el óptimo (máx performance)
    if soc_con_umbral_min >= soc_target_min:
         return UMBRAL_MINIMO, soc_con_umbral_min, log_pruebas
    
    # Caso B: ¿Qué pasa si NO ayudamos NUNCA (Umbral 13)? (Mínimo consumo posible)
    soc_con_umbral_max = calcular_soc_final_para_umbral(
        UMBRAL_MAXIMO, acc_telem, v_ms, n_vueltas, **params
    )
    
    # Caso C: Si ahorrando al máximo no llegamos al mínimo -> Devolver MAX (Mejor esfuerzo para sobrevivir)
    if soc_con_umbral_max < soc_target_min:
         return UMBRAL_MAXIMO, soc_con_umbral_max, log_pruebas
    
    # --- 2. Bisección con Rastreo del Mejor Válido ---
    # Si estamos aquí, la solución cruza el objetivo en algún punto entre 0 y 13.
    # Inicializamos el "mejor conocido" con el caso seguro (MAXIMO), que sabemos que cumple (soc_max >= target_min)
    mejor_umbral = UMBRAL_MAXIMO
    mejor_soc = soc_con_umbral_max
    
    umbral_bajo, umbral_alto = UMBRAL_MINIMO, UMBRAL_MAXIMO
    max_iter = 20 # Iteraciones suficientes para precisión
    
    for i in range(max_iter):
        umbral_test = (umbral_bajo + umbral_alto) / 2.0
        soc_final = calcular_soc_final_para_umbral(
            umbral_test, acc_telem, v_ms, n_vueltas, **params
        )
        
        decision = ""
        es_valido = False
        
        # Analizar resultado
        if soc_target_min <= soc_final <= soc_target_max:
            # ¡Dimos en el clavo! (Ventana perfecta)
            return round(umbral_test, 2), soc_final, log_pruebas
            
        elif soc_final < soc_target_min:
            # DEFICIT: Nos falta batería -> Consumo Excesivo -> SUBIR Umbral
            umbral_bajo = umbral_test
            decision = 'DEFICIT_SUBIR'
            es_valido = False # Este umbral rompe la restricción de SOC mínimo
            
        else: # soc_final > soc_target_max
            # EXCESO: Nos sobra batería -> Consumo Bajo -> BAJAR Umbral
            umbral_alto = umbral_test
            decision = 'EXCESO_BAJAR'
            es_valido = True # Cumple SOC >= Min, es un candidato válido
        
        # Actualizar "Mejor Candidato" si es válido y menor (más agresivo) que el anterior
        # Nota: En la lógica de EXCESO_BAJAR, siempre vamos hacia umbrales menores.
        # Si es válido, este umbral_test es mejor (más bajo) que el mejor_umbral anterior (que venía de arriba).
        if es_valido:
            mejor_umbral = umbral_test
            mejor_soc = soc_final
        
        log_pruebas.append({
            'iter': i + 1, 
            'umbral': round(umbral_test, 2),
            'soc': round(soc_final, 2),
            'decision': decision
        })
        
        # Tolerancia de convergencia
        if (umbral_alto - umbral_bajo) < 0.05:
            break
            
    # Si salimos del bucle sin "EXITO" exacto, devolvemos el MEJOR VALIDO encontrado
    # Esto garantiza que nunca devolvemos un umbral que deje tirado al coche (SOC < Min)
    return round(mejor_umbral, 2), mejor_soc, log_pruebas


# =============================================================================
# FUNCIONES AUXILIARES DE OPTIMIZACIÓN
# =============================================================================

def preparar_parametros_optimizacion(
    cap_celda, r_interna_mohm, n_s, n_p, soc_max,
    i_descarga_cont, i_descarga_pico, t_descarga_pico,
    v_nom_celda, v_max_celda, v_min_celda,
    masa_vehiculo, F_aero, F_roll, eta_total,
    p_media_aux_w, dt_sim, P_bat_input,
    cl, area, dist_peso,
    activar_limite_motor=False,
    p_motor_max_kw=0.0
):
    """
    Prepara el diccionario de parámetros para la función de optimización.
    
    Args:
        cap_celda, r_interna_mohm, n_s, n_p, soc_max: Params Pila/Pack
        i_descarga_cont, i_descarga_pico, t_descarga_pico: Límites Pila
        v_nom_celda, v_max_celda, v_min_celda: Voltajes Pila
        masa_vehiculo, F_aero, F_roll, eta_total: Vehiculo
        p_media_aux_w, dt_sim, P_bat_input: Simulacion
        cl, area, dist_peso: Dinámica
        activar_limite_motor, p_motor_max_kw: Extra
    
    Returns:
        dict: Parámetros listos para buscar_umbral_optimo
    """
    from .physics import CONSTANTES_VEHICULO
    
    return {
        'masa': masa_vehiculo,
        'F_aero': F_aero,
        'F_roll': F_roll,
        'eta': eta_total,
        
        # Parámetros directos (ya procesados por SOH o snapshot)
        'I_cont': i_descarga_cont,
        'I_pico': i_descarga_pico,
        't_pico_max': t_descarga_pico,
        'v_nom_celda': v_nom_celda,
        'v_max_celda': v_max_celda,
        'v_min_celda': v_min_celda,
        'cap_celda': cap_celda,
        'r_interna_mohm': r_interna_mohm,
        'n_s': n_s,
        'n_p': n_p,
        'soc_max': soc_max,
        
        # Parámetros adicionales
        'p_aux': p_media_aux_w,
        'dt': dt_sim,
        'mu': CONSTANTES_VEHICULO.MU_NEUMATICOS,
        'rho': CONSTANTES_VEHICULO.RHO_AIRE,
        'cl': cl,
        'area': area,
        'dist_peso': dist_peso,
        'P_regen_vector': P_bat_input,
        'activar_limite_motor': activar_limite_motor,
        'p_motor_max_kw': p_motor_max_kw
    }