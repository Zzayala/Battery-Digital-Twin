"""
Módulo de física: Cálculos de potencia, SOC, regeneración y simulación.
"""

import numpy as np
from scipy.optimize import minimize_scalar
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Mock de jit si no está disponible para no romper lógica
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# FUNCIONES JIT (NUMBA) - CORE CÁLCULO
# =============================================================================

@jit(nopython=True)
def _simular_soc_numba(
    P_elec_pack,
    dt,
    n_vueltas,
    soc_inicial,
    cap_pack_as,
    r_pack_ohm,
    n_s,
    ocv_coeffs  # Coeficientes del polinomio OCV (suponiendo orden 3 o 5)
):
    """
    Núcleo de cálculo de evolución de SOC acelerado por Numba.
    No puede usar objetos ni diccionarios, solo arrays y escalares.
    """
    soc_actual = soc_inicial

    # Pre-extraer coeffs para velocidad (suponemos polyval manual o np.polyval si numba lo soporta)
    # Numba soporta np.polyval parcialmente, pero mejor implementarlo inline para máxima compatibilidad
    # Coeffs orden: [p3, p2, p1, p0] por ejemplo

    # Bucle vueltas
    for _ in range(n_vueltas):
        # Bucle time-steps
        for p_inst in P_elec_pack:
            # 1. Calcular OCV con Polinomio (manual para velocidad extrema)
            # np.polyval(coeffs, x) -> c0*x^N + c1*x^N-1 ... + cN
            # Implementamos Horner's method
            v_cell_ocv = 0.0
            for c in ocv_coeffs:
                v_cell_ocv = v_cell_ocv * soc_actual + c

            v_pack_ocv = v_cell_ocv * n_s

            # 2. Calcular Corriente
            if abs(r_pack_ohm) < 1e-5:
                if v_pack_ocv > 0.1:
                    i_pack = p_inst / v_pack_ocv
                else:
                    i_pack = 0.0
            else:
                # I^2*R - OCV*I + P = 0
                term_c = p_inst
                term_b = -v_pack_ocv
                term_a = r_pack_ohm

                discr = term_b*term_b - 4*term_a*term_c

                if discr < 0:
                    discr = 0.0 # Colapso de tensión

                # Solución cuadrática (-b - sqrt(discr)) / 2a (la otra raíz dispara I)
                # Ojo signo: Queremos la menor corriente en magnitud que satisfaga la potencia
                # Descarga (P>0): OCV - I*R = V_term. P=V_term*I.
                # I = (OCV - sqrt(OCV^2 - 4RP))/2R
                i_pack = (-term_b - np.sqrt(discr)) / (2 * term_a)

            # 3. Actualizar SOC
            d_soc = -(i_pack * dt) / cap_pack_as * 100.0
            soc_actual = soc_actual + d_soc

            # Clamp limits
            if soc_actual < 0.0: soc_actual = 0.0
            if soc_actual > 100.0: soc_actual = 100.0

    return soc_actual


# =============================================================================
# CONSTANTES DEL VEHÍCULO
# =============================================================================

class CONSTANTES_VEHICULO:
    """Constantes físicas del vehículo."""
    G = 9.81                        # Aceleración gravitatoria (m/s²)
    RHO_AIRE = 1.225                # Densidad del aire (kg/m³)
    CD = 0.40                       # Coeficiente de arrastre
    AREA_FRONTAL = 1.24             # Área frontal (m²)
    C_ROLL = 0.015                  # Coeficiente de rodadura
    MU_NEUMATICOS = 1.2             # Coeficiente de fricción neumáticos
    CL_DOWNFORCE = 2.5              # Coeficiente de sustentación (downforce)
    DISTRIBUCION_PESO_DELANT = 0.35 # Distribución de peso en eje delantero


class CONSTANTES_ELECTRICAS:
    """Constantes del sistema eléctrico."""
    EFICIENCIA_MOTOR = 0.92         # Eficiencia del motor
    EFICIENCIA_INVERSOR = 0.95      # Eficiencia del inversor
    R_CONN_PACK = 0.002             # Resistencia de conexiones del pack (Ω)
    R_CONN_PACK = 0.002             # Resistencia de conexiones del pack (Ω)


# =============================================================================
# BASE DE DATOS DE DEGRADACIÓN
# =============================================================================




# =============================================================================
# CÁLCULO DE DINÁMICA VEHICULAR
# =============================================================================

def calcular_dinamica_vehiculo(v_ms, acc_m2, masa_kg, cd=0.40, area_frontal=1.24):
    """
    Calcula fuerzas y potencias mecánicas básicas del vehículo.

    Args:
        v_ms: Array de velocidades (m/s)
        acc_m2: Array de aceleraciones (m/s²)
        masa_kg: Masa total del vehículo (kg)
        cd: Coeficiente de arrastre
        area_frontal: Área frontal (m²)

    Returns:
        dict: {
            'F_aero': Fuerza aerodinámica (N),
            'F_roll': Fuerza de rodadura (N),
            'F_frenos': Fuerza de frenado mecánico (N),
            'P_frenos_total': Potencia total de frenado (W) antes de regen
        }
    """
    CV = CONSTANTES_VEHICULO

    # Fuerzas Resistivas
    F_aero = 0.5 * cd * area_frontal * CV.RHO_AIRE * (v_ms**2)
    F_roll = masa_kg * CV.G * CV.C_ROLL

    # Fuerza Neta Necesaria (F = m*a)
    F_necesaria = np.abs(acc_m2) * masa_kg

    # Frenado Mecánico (solo cuando acc < 0)
    F_frenos = np.zeros_like(v_ms)
    mask_frenando = acc_m2 < 0

    F_frenos_raw = F_necesaria - F_aero - F_roll
    F_frenos[mask_frenando] = np.maximum(0, F_frenos_raw[mask_frenando])

    P_frenos_total = F_frenos * v_ms

    return {
        'F_aero': F_aero,
        'F_roll': F_roll,
        'F_frenos': F_frenos,
        'P_frenos_total': P_frenos_total
    }


# =============================================================================
# CÁLCULO DE DEGRADACIÓN (ARRHENIUS + WÖHLER)
# =============================================================================

def calcular_factores_degradacion(
    I_cell_array, cap_celda, temps_full, soc_max, soc_min,
    cerebro_ia=None, factor_dano_regen=4.0, factor_pouch=2.0
):
    """
    Calcula los factores de degradación de la batería usando modelos físicos.

    Modelos aplicados:
    - Arrhenius: Factor térmico (duplica degradación cada 10°C sobre 25°C)
    - Wöhler: Factor de fatiga por DoD (ciclos profundos degradan más)
    - Stress SOC: Penalización por operar lejos del 50% SOC

    Args:
        I_cell_array: Array de corrientes por celda (A)
        cap_celda: Capacidad nominal de la celda (Ah)
        temps_full: Array de temperaturas durante simulación (°C)
        soc_max: SOC máximo operativo (%)
        soc_min: SOC mínimo operativo (%)
        cerebro_ia: Instancia de CerebroDegradacion (opcional, para predicción IA)
        factor_dano_regen: Factor multiplicador de daño por regeneración
        factor_pouch: Factor de formato de celda (pouch vs cilíndrica)

    Returns:
        dict: {
            'ciclos_vida': Ciclos estimados hasta 80% capacidad,
            'deg_por_ciclo': Degradación por ciclo (Ah),
            'c_rate_efectivo': C-rate RMS efectivo,
            'metodo': 'IA' o 'BD' (base de datos),
            'factores': {f_temp, factor_fatiga, factor_stress}
        }
    """
    # --- Cálculo de C-rate efectivo (RMS ponderado) ---
    corrientes_desc = I_cell_array[I_cell_array >= 0]
    corrientes_regen = I_cell_array[I_cell_array < 0]

    suma_sq_desc = np.sum(corrientes_desc**2) if len(corrientes_desc) > 0 else 0
    suma_sq_regen = (np.sum(corrientes_regen**2) * factor_dano_regen) if len(corrientes_regen) > 0 else 0
    mean_sq = (suma_sq_desc + suma_sq_regen) / len(I_cell_array) if len(I_cell_array) > 0 else 0
    rms_current = np.sqrt(mean_sq)

    c_rate_efectivo = rms_current / cap_celda if cap_celda > 0 else 0

    # --- Parámetros operativos ---
    temp_promedio = np.mean(temps_full)
    dod_pct = soc_max - soc_min
    dod_decimal = max(dod_pct, 1.0) / 100.0
    soc_promedio = (soc_max + soc_min) / 2.0

    # --- Ley de Arrhenius (Factor Térmico) ---
    # Duplica la degradación por cada 10°C sobre 25°C
    f_temp = 2.0 ** ((temp_promedio - 25.0) / 10.0)

    # --- Ley de Wöhler (Factor de Fatiga por DoD) ---
    # Ciclos profundos causan más daño (exponente típico 1.5-2.0)
    factor_fatiga = dod_decimal ** 1.6

    # --- Factor de Stress por SOC medio ---
    # Operar cerca de 0% o 100% causa más estrés que cerca de 50%
    factor_stress = 1.0 + (0.0027 * (abs(soc_promedio - 50.0) ** 2))

    # --- Predicción de degradación ---
    metodo_degradacion = "BD"  # Por defecto: Base de Datos
    deg_por_ciclo = 0.0
    delta_r_ohm_por_ciclo = 0.0
    motivo_fallback = None  # Razón por la que no se usó IA

    # Intentar usar módulo de inferencia IA primero
    # Intentar usar módulo de inferencia IA V2
    if cerebro_ia is not None:
        try:
            # V2 devuelve: deg_cap_base, metric_res_norm, info
            tasa_cap_v2, metrica_res_v2, info_ia = cerebro_ia.predecir_degradacion(
                c_rate_efectivo, temp_promedio, dod_pct
            )

            if tasa_cap_v2 is not None:
                # 1. Aplicar factores correctores a la capacidad
                deg_por_ciclo = (tasa_cap_v2 * cap_celda) * factor_pouch * f_temp * factor_fatiga * factor_stress

                # 2. CÁLCULO NUEVO V2: Crecimiento de Resistencia (Delta R)
                # Según la documentación de V2: Delta_R (Ohm) = Métrica / Capacidad_Tu_Celda
                delta_r_ohm_por_ciclo = (metrica_res_v2 / cap_celda) * f_temp

                metodo_degradacion = "IA"
            else:
                motivo_fallback = info_ia if info_ia else "Datos fuera del rango experimental"
        except Exception as e:
            motivo_fallback = f"Error en módulo IA V2: {str(e)}"
    else:
        motivo_fallback = "Módulo IA no disponible"

    # Fallback: Usar base de datos genérica
    if metodo_degradacion == "BD":
        from scipy.interpolate import interp1d
        # Usamos un modelo simplificado basado en C-rate
        c_rates_ref = [0.5, 1.0, 2.0, 3.0, 5.0]
        deg_base_ref = [0.0001, 0.0002, 0.0004, 0.0007, 0.0012]  # % por ciclo aproximado
        f_interp = interp1d(c_rates_ref, deg_base_ref, fill_value="extrapolate", bounds_error=False)
        deg_base = float(f_interp(c_rate_efectivo))
        deg_por_ciclo = ((deg_base * cap_celda) / 2000) * factor_pouch * f_temp * factor_fatiga * factor_stress

    # --- Cálculo de vida útil ---
    cap_perdida_lim = cap_celda * 0.20  # 20% = fin de vida
    ciclos_vida = cap_perdida_lim / deg_por_ciclo if deg_por_ciclo > 1e-15 else 500000

    return {
        'ciclos_vida': ciclos_vida,
        'deg_por_ciclo': deg_por_ciclo,
        'delta_r_ohm_por_ciclo': delta_r_ohm_por_ciclo,
        'c_rate_efectivo': c_rate_efectivo,
        'temp_promedio': temp_promedio,
        'metodo': metodo_degradacion,
        'motivo_fallback': motivo_fallback,  # None si se usó IA, razón si se usó BD
        'factores': {
            'f_temp': f_temp,
            'factor_fatiga': factor_fatiga,
            'factor_stress': factor_stress
        }
    }


# =============================================================================
# CÁLCULO DE KPIs DE CARRERA
# =============================================================================

def calcular_kpis_carrera(
    P_elec_pack, I_cell, v_ms, soc_full, dt,
    n_vueltas, p_aux, i_descarga_cont,
    v_nom_celda, n_s, n_p, cap_celda, peso_celda_g,
    soc_max, soc_min, r_interna_mohm
):
    """
    Calcula todos los KPIs de energía y rendimiento de la carrera.

    Args:
        P_elec_pack: Array de potencia eléctrica por vuelta (W)
        I_cell: Array de corriente por celda (A)
        v_ms: Array de velocidades (m/s)
        soc_full: Array de SOC durante toda la carrera (%)
        dt: Delta de tiempo (s)
        n_vueltas: Número de vueltas
        p_aux: Potencia auxiliar (W)
        i_descarga_cont: Corriente continua máxima (A)
        v_nom_celda: Tensión nominal de celda (V)
        n_s, n_p: Topología del pack
        cap_celda: Capacidad de celda (Ah)
        peso_celda_g: Peso de celda (g)
        soc_max, soc_min: Límites operativos de SOC (%)

    Returns:
        dict: KPIs calculados
    """
    # --- Energía de Consumo (Salida del Pack) ---
    P_consumo_v = np.maximum(P_elec_pack, 0)
    E_consumo_vuelta = np.sum(P_consumo_v) * dt / 3600.0  # Wh
    E_consumo_total = E_consumo_vuelta * n_vueltas

    # --- Energía Disipada (Pérdidas Térmicas) ---
    # P_loss = I_cell^2 * R_cell * N_cells (aprox, ignorando R_conn por ahora o incluyéndola)
    # R_cell efectiva ya incluye factor transitorio
    # R_cell efectiva ya incluye factor transitorio
    r_cell_eff = (r_interna_mohm / 1000.0)
    n_cells_total = n_s * n_p

    # P_loss_total (W) = (I_cell^2 * R_cell_eff * n_cells_total)
    # Nota: I_cell ya tiene signo, al cuadrado se vuelve positivo siempre
    P_loss_t = (I_cell ** 2) * r_cell_eff * n_cells_total

    E_termica_vuelta = np.sum(P_loss_t) * dt / 3600.0 # Wh
    E_termica_total = E_termica_vuelta * n_vueltas



    # --- Energía de Regeneración NETA (Entrada al Pack) ---
    # Esto es lo que realmente recarga el SOC.
    # Diferente a la bruta de motores que se gasta en auxiliares antes de entrar.
    P_regen_v_neta = np.abs(np.minimum(P_elec_pack, 0))
    E_regen_vuelta = np.sum(P_regen_v_neta) * dt / 3600.0  # Wh
    E_regen_total = E_regen_vuelta * n_vueltas

    # --- Capacidad del Pack ---
    cap_pack_ah = cap_celda * n_p
    V_pack_nom = v_nom_celda * n_s
    E_pack_total = V_pack_nom * cap_pack_ah  # Wh (Energía Nominal Total)

    # E_real_disp: Energía teórica disponible en el rango SOC config (a voltaje nominal)
    E_real_disp = E_pack_total * (soc_max - soc_min) / 100.0

    # E_virtual: Energía REAL disponible para descargar (Lo que había + Lo que entró)
    # Ahora sí debería ser comparable con E_consumo_total si acabamos al SOC mínimo.
    E_virtual = E_real_disp + E_regen_total

    # --- Tiempos en Pico ---
    # Descarga: I_cell > 0 y supera límite continuo
    mask_pico_desc = I_cell > i_descarga_cont
    t_pico_desc = np.sum(mask_pico_desc) * dt

    # Carga (regeneración): I_cell < 0 y |I_cell| supera un umbral
    # Límite de carga típico es menor que descarga (baterías aceptan menos corriente de carga)
    # Usamos 25% del límite de descarga como umbral de "pico de carga"
    i_carga_cont = i_descarga_cont * 0.25  # Límite conservador para carga
    I_cell_carga = np.where(I_cell < 0, np.abs(I_cell), 0)
    mask_pico_carga = I_cell_carga > i_carga_cont
    t_pico_carga = np.sum(mask_pico_carga) * dt

    # --- Potencias Máximas ---
    p_max_desc_kw = np.max(P_elec_pack) / 1000.0
    p_max_carga_kw = np.abs(np.min(P_elec_pack)) / 1000.0

    # --- SOC Final ---
    soc_final_real = soc_full[-1]
    soc_disponible = soc_final_real - soc_min

    # --- Distancia ---
    distancia_vuelta_m = np.trapz(v_ms, dx=dt)
    distancia_total_km = (distancia_vuelta_m * n_vueltas) / 1000.0

    # --- Peso del Pack ---
    peso_pack_kg = (peso_celda_g * n_s * n_p) / 1000.0

    return {
        # Energías
        'E_regen_vuelta': E_regen_vuelta,
        'E_regen_total': E_regen_total,
        'E_consumo_vuelta': E_consumo_vuelta,
        'E_consumo_total': E_consumo_total,
        'E_termica_vuelta': E_termica_vuelta,
        'E_termica_total': E_termica_total,

        'E_pack_total': E_pack_total,
        'E_real_disp': E_real_disp,
        'E_virtual': E_virtual,

        # Tiempos pico
        't_pico_desc': t_pico_desc,
        't_pico_carga': t_pico_carga,

        # Potencias
        'p_max_desc_kw': p_max_desc_kw,
        'p_max_carga_kw': p_max_carga_kw,

        # SOC
        'soc_final_real': soc_final_real,
        'soc_disponible': soc_disponible,

        # Distancia y Peso
        'distancia_vuelta_m': distancia_vuelta_m,
        'distancia_total_km': distancia_total_km,
        'peso_pack_kg': peso_pack_kg,

        # Extras útiles
        'cap_pack_ah': cap_pack_ah,
        'V_pack_nom': V_pack_nom
    }


# =============================================================================
# FUNCIÓN OCV POLINÓMICA
# =============================================================================

@st.cache_resource
def obtener_funcion_ocv_polinomica(v_min, v_nom, v_max):
    """
    Genera una función OCV(SOC) basada en un polinomio de GRADO 9 de alta fidelidad.

    La curva base se escala linealmente para encajar en los límites [v_min, v_max]
    definidos por el usuario.

    Args:
        v_min: Tensión mínima de celda (V) - corresponde a SOC 0%
        v_nom: Tensión nominal (V) - NO SE USA para deformar, solo referencia (se mantiene la forma base)
        v_max: Tensión máxima de celda (V) - corresponde a SOC 100%

    Returns:
        función: f(soc) → voltage
    """

    def f_ocv_base(soc):
        # Polinomio base (aprox 2.75V a 4.2V)
        soc = np.clip(soc, 0, 100)
        ocv = (2.17851508e-15 * soc**9
              - 1.06187295e-12 * soc**8
              + 2.19502650e-10 * soc**7
              - 2.50522387e-08 * soc**6
              + 1.72171213e-06 * soc**5
              - 7.29079506e-05 * soc**4
              + 1.87251524e-03 * soc**3
              - 2.77050498e-02 * soc**2
              + 2.18531709e-01 * soc
              + 2.75626590e+00)
        return ocv

    # Calibración dinámica: Adaptar curva base a límites v_min y v_max
    v_base_0 = f_ocv_base(0.0)      # ~2.756 V
    v_base_100 = f_ocv_base(100.0)  # ~4.19 V

    rango_base = v_base_100 - v_base_0
    rango_user = v_max - v_min

    def f_ocv_hibrida(soc):
        v_raw = f_ocv_base(soc)

        # 1. Normalizar la curva base a [0, 1]
        v_norm = (v_raw - v_base_0) / rango_base

        # 2. Escalar al rango del usuario
        v_scaled = v_min + (v_norm * rango_user)

        # Opcional: Podríamos aplicar un offset adicional fino si v_nom difiere
        # significativamente del centro, pero escalar a min/max es lo más robusto
        # para evitar voltajes fuera de rango.

        return v_scaled

    # --- Parche para Numba ---
    # f_ocv_hibrida es una función compleja, no un polinomio simple.
    # Para que Numba funcione, necesitamos aproximarla a un polinomio y exponer .coeffs
    # Generamos puntos para ajustar un polinomio sustituto
    x_test = np.linspace(0, 100, 20)
    y_test = [f_ocv_hibrida(x) for x in x_test] # La función escalar v_min correcta

    # Ajuste polinómico de grado 5 (suficiente para OCV)
    # y = c[0]*x^5 + ... (numpy polyfit devuelve coeficiente mayor grado primero)
    coeffs_approx = np.polyfit(x_test, y_test, 5)

    # Adjuntamos al objeto funcion
    f_ocv_hibrida.coeffs = coeffs_approx

    return f_ocv_hibrida


# =============================================================================
# GENERADOR DE PERFIL DE POTENCIA UNIFICADO
# =============================================================================

def generar_perfil_potencia_unificado(
    acc_telem, v_ms, dt,
    masa, F_aero, F_roll, eta,
    I_cont, I_pico, t_pico_max,
    v_nom_est, n_s, n_p,
    p_aux, P_regen_vector,
    mu, rho, cl, area, dist_peso,
    acc_umbral=4.0,
    margen_traccion=0.90,
    activar_limite_motor=False,
    p_motor_max_kw=0.0
):
    """
    Genera el perfil de potencia eléctrica del pack para una vuelta.

    Args:
        acc_telem: Array de aceleraciones (m/s²)
        v_ms: Array de velocidades (m/s)
        dt: Paso temporal (s)
        masa: Masa del vehículo (kg)
        F_aero: Array de fuerza aerodinámica (N)
        F_roll: Fuerza de rodadura (N)
        eta: Eficiencia total del tren motriz
        I_cont: Corriente continua máxima (A)
        I_pico: Corriente pico máxima (A)
        t_pico_max: Tiempo máximo en corriente pico (s)
        v_nom_est: Tensión nominal estimada de celda (V)
        n_s: Número de celdas en serie
        n_p: Número de celdas en paralelo
        p_aux: Potencia auxiliar (W)
        P_regen_vector: Vector de potencia de regeneración disponible (W)
        mu: Coeficiente de fricción
        rho: Densidad del aire (kg/m³)
        cl: Coeficiente de sustentación
        area: Área frontal (m²)
        dist_peso: Distribución de peso eje delantero
        acc_umbral: Umbral de aceleración para activar tracción (m/s²)
        margen_traccion: Margen de seguridad para tracción (0-1)
        activar_limite_motor: Boolean, si True limita la potencia mecánica
        p_motor_max_kw: Potencia mecánica máxima de motor (kW)

    Returns:
        np.array: Potencia eléctrica del pack (W) para cada punto de la vuelta
    """
    n_puntos = len(acc_telem)
    P_elec_pack = np.zeros(n_puntos)

    tiempo_acum_pico = 0.0

    # Márgenes de seguridad
    I_cont_safe = I_cont * 0.95
    I_pico_safe = I_pico * 0.95
    V_pack_nom = v_nom_est * n_s

    for i in range(len(acc_telem)):
        acc_i = acc_telem[i]
        vel_i = v_ms[i]

        # 1. ZONA DE NO-TRACCIÓN (BAJO UMBRAL)
        if acc_i <= acc_umbral:
            # Enfriamos el fusible térmico virtual
            tiempo_acum_pico = max(0, tiempo_acum_pico - dt * 0.5)

            # Aplicamos Regen (si frenada) o solo Auxiliar
            if acc_i < 0:
                P_elec_pack[i] = -P_regen_vector[i] + p_aux
            else:
                P_elec_pack[i] = p_aux
            continue

        # 2. ZONA DE TRACCIÓN (SOBRE UMBRAL)
        F_inert = masa * acc_i
        F_tot = F_inert + F_aero[i] + F_roll
        P_mech = F_tot * vel_i

        # Limitar Potencia Mecánica por Motor (si aplica)
        if activar_limite_motor:
             P_max_mech = p_motor_max_kw * 1000.0
             P_mech = min(P_mech, P_max_mech)

        # Límites Eléctricos Dinámicos
        I_disp = I_pico_safe if tiempo_acum_pico < t_pico_max else I_cont_safe
        P_bat_max = V_pack_nom * I_disp * n_p

        # Límite Grip
        F_down = 0.5 * rho * vel_i**2 * cl * area
        Peso_front = masa * 9.81 * dist_peso
        F_trac_max = (Peso_front + F_down * dist_peso) * mu
        P_grip = (F_trac_max * vel_i) / eta * margen_traccion

        # Selección de Potencia Real
        P_dem = P_mech / eta if P_mech > 0 else 0
        P_real = min(P_dem, P_bat_max, P_grip)

        P_elec_pack[i] = P_real + p_aux

        # Gestión del tiempo de pico
        v_ref_safe = V_pack_nom if V_pack_nom > 1.0 else 1.0
        corriente_estimada = P_real / v_ref_safe / n_p
        if corriente_estimada > I_cont_safe:
            tiempo_acum_pico += dt
        else:
            tiempo_acum_pico = max(0, tiempo_acum_pico - dt * 0.2)

    return P_elec_pack


# =============================================================================
# CÁLCULO DE SOC FINAL
# =============================================================================

def calcular_soc_final_para_umbral(
    umbral_test,
    acc_telem, v_ms, n_vueltas,
    masa, F_aero, F_roll, eta,
    I_cont, I_pico, t_pico_max,
    v_nom_celda, v_max_celda, v_min_celda,
    n_s, n_p, cap_celda,
    soc_max, r_interna_mohm,
    p_aux, P_regen_vector,
    mu, rho, cl, area, dist_peso,
    dt,
    margen_traccion=0.90,
    activar_limite_motor=False,
    p_motor_max_kw=0.0
):
    """
    Calcula el SOC final simulando con la curva OCV real.

    Args:
        umbral_test: Umbral de aceleración a probar (m/s²)
        ... (resto de parámetros recibidos via **params)

    Returns:
        float: SOC final después de n_vueltas (%)
    """
    # 1. Generar polinomio OCV
    f_ocv = obtener_funcion_ocv_polinomica(v_min_celda, v_nom_celda, v_max_celda)

    # 2. Generar perfil de potencia del pack (W)
    P_elec_pack = generar_perfil_potencia_unificado(
        acc_telem, v_ms, dt,
        masa, F_aero, F_roll, eta,
        I_cont, I_pico, t_pico_max,
        v_nom_celda, n_s, n_p,
        p_aux, P_regen_vector,
        mu, rho, cl, area, dist_peso,
        acc_umbral=umbral_test,
        margen_traccion=margen_traccion,
        activar_limite_motor=activar_limite_motor,
        p_motor_max_kw=p_motor_max_kw
    )

    # 3. Integración temporal (Acelerada con Numba si es posible)
    # Extraer coeficientes del polinomio f_ocv para pasarlos a Numba
    # f_ocv es un numpy.poly1d
    ocv_coeffs = f_ocv.coeffs

    # Pre-cálculo de constantes
    cap_pack_as = cap_celda * n_p * 3600.0
    r_pack_ohm = (r_interna_mohm / 1000.0 * n_s) / n_p

    # Asegurar tipos float64 para numba
    P_elec_pack = P_elec_pack.astype(np.float64)
    dt = float(dt)
    n_vueltas = int(n_vueltas)
    soc_max = float(soc_max)
    cap_pack_as = float(cap_pack_as)
    r_pack_ohm = float(r_pack_ohm)
    ocv_coeffs = ocv_coeffs.astype(np.float64)
    n_s = float(n_s)

    soc_actual = _simular_soc_numba(
        P_elec_pack,
        dt,
        n_vueltas,
        soc_max,
        cap_pack_as,
        r_pack_ohm,
        n_s,
        ocv_coeffs
    )

    return soc_actual


# =============================================================================
# CÁLCULO DE REGENERACIÓN
# =============================================================================

def calcular_regeneracion(
    P_frenos_total, v_ms, soc_actual,
    v_max_celda, r_interna_mohm, n_s, n_p,
    f_ocv_func, eta_total, cap_celda=5.0,
    aplicar_limite_crate=True  # Si False, solo limita por tensión
):
    """
    Calcula la potencia de regeneración basada en el Pack completo.

    Lógica Pack:
    - R_pack = (R_celda * 1.25 * n_s) / n_p
    - V_pack_max = V_celda_max * n_s
    - V_ocv_pack = f_ocv(soc) * n_s
    - I_max_crate_pack = (Cap_celda * n_p) * 9C (si aplicar_limite_crate=True)

    Args:
        P_frenos_total: Potencia mecánica de frenado (W)
        aplicar_limite_crate: Si True, aplica límite de 9C. Si False, solo límite por tensión.
        ...
    """
    # Parámetros del PACK
    R_cell_eff = (r_interna_mohm / 1000.0)
    R_pack = (R_cell_eff * n_s) / n_p

    V_pack_max = v_max_celda * n_s
    Cap_pack = cap_celda * n_p

    # Límite de corriente por C-rate (Pack)
    if aplicar_limite_crate:
        I_max_crate_pack = Cap_pack * 9.0
    else:
        I_max_crate_pack = float('inf')  # Sin límite de C-rate

    P_regen_mech_utilizada = np.zeros_like(v_ms)
    P_bat_input = np.zeros_like(v_ms)

    soc_is_scalar = np.isscalar(soc_actual)

    for i in range(len(P_frenos_total)):
        P_freno_mec_i = float(P_frenos_total[i])
        if P_freno_mec_i <= 0:
            continue

        soc_i = soc_actual if soc_is_scalar else soc_actual[i]

        # OCV del Pack
        V_ocv_pack = f_ocv_func(soc_i) * n_s

        # Delta V Pack
        Delta_V_pack = V_pack_max - V_ocv_pack

        if Delta_V_pack <= 0:
            continue

        # Corriente admisible por resistencia (Ohmica)
        if R_pack > 1e-6:
            I_ohmica_pack = Delta_V_pack / R_pack
        else:
            I_ohmica_pack = 999999.0 # Infinita teórica

        # Selección de corriente de carga real (limitada por química 4C)
        I_charge_pack = min(I_ohmica_pack, I_max_crate_pack)

        # Potencia eléctrica que el pack puede tragar (a V_max de carga)
        # P = V * I. Asumimos carga a voltaje terminal máximo por seguridad o CV.
        P_elec_max_abs = V_pack_max * I_charge_pack

        # Potencia mecánica equivalente requerida (considerando pérdidas)
        # Si quiero meter 10kW a la batería, necesito absorber 10/eta kW de la rueda.
        P_mech_max_abs = P_elec_max_abs / eta_total

        # Lo que realmente regeneramos es el mínimo entre lo que frenamos y lo que podemos absorber
        P_regen_real_mech = min(P_freno_mec_i, P_mech_max_abs)

        P_regen_mech_utilizada[i] = P_regen_real_mech
        P_bat_input[i] = P_regen_real_mech * eta_total

    return P_regen_mech_utilizada, P_bat_input


# =============================================================================
# SIMULACIÓN CON UMBRAL ADAPTATIVO
# =============================================================================

def simular_con_umbral_adaptativo(
    acc_telem, v_ms, dt, n_vueltas,
    masa, F_aero, F_roll, eta,
    I_cont, I_pico, t_pico_max,
    v_nom_celda, v_max_celda, v_min_celda,
    n_s, n_p, cap_celda,
    soc_max, soc_min, r_interna_mohm,
    p_aux, P_regen_vector,
    mu, rho, cl, area, dist_peso,
    temp_amb, refrigeracion,
    peso_celda_g=100.0,
    umbral_inicial=4.0,
    intervalo_adaptacion_m=50.0,
    margen_soc_final=0.5,
    margen_traccion=0.90
):
    """
    Simulación completa con CONTROL ADAPTATIVO del umbral de aceleración.
    ...
    """
    # Constantes
    UMBRAL_MIN = 0.0
    UMBRAL_MAX = 13.0
    UMBRAL_MAX = 13.0

    # Preparación
    f_ocv = obtener_funcion_ocv_polinomica(v_min_celda, v_nom_celda, v_max_celda)
    r_operative_cell = (r_interna_mohm / 1000.0)
    r_pack_ohm = (r_interna_mohm / 1000.0 * n_s) / n_p
    cap_pack_as = cap_celda * n_p * 3600.0
    V_pack_nom = v_nom_celda * n_s

    # Calcular distancia total de la prueba
    distancia_vuelta = np.sum(v_ms * dt)
    distancia_total = distancia_vuelta * n_vueltas

    # SOC objetivo (mínimo + margen)
    soc_objetivo = soc_min + margen_soc_final

    # Márgenes eléctricos
    I_cont_safe = I_cont * 0.95
    I_pico_safe = I_pico * 0.95

    # Arrays de salida
    n_puntos_vuelta = len(acc_telem)
    n_puntos_total = n_puntos_vuelta * n_vueltas

    t_vuelta = n_puntos_vuelta * dt
    t_full = np.zeros(n_puntos_total)
    soc_full = np.zeros(n_puntos_total)
    temps_full = np.zeros(n_puntos_total)
    umbral_full = np.zeros(n_puntos_total)
    P_elec_full = np.zeros(n_puntos_total)
    distancia_full = np.zeros(n_puntos_total)

    # Estado inicial
    soc_actual = soc_max
    T_actual = temp_amb
    umbral_actual = umbral_inicial
    distancia_acumulada = 0.0
    ultima_adaptacion_m = 0.0
    tiempo_acum_pico = 0.0

    # Constantes térmicas (MODELO DE PACK COMPLETO)
    n_celdas_total = n_s * n_p
    mass_celda_kg = peso_celda_g / 1000.0
    mass_pack_kg = mass_celda_kg * n_celdas_total
    cp = 1000  # J/(kg·K) - Calor específico de celdas LiPo
    r_operative_cell = (r_interna_mohm / 1000.0)

    idx_global = 0

    for vuelta in range(n_vueltas):
        for i in range(n_puntos_vuelta):
            acc_i = acc_telem[i]
            vel_i = v_ms[i]

            # Actualizar distancia
            distancia_acumulada += vel_i * dt

            # ============================================
            # CONTROL ADAPTATIVO (cada intervalo_adaptacion_m metros)
            # ============================================
            if distancia_acumulada - ultima_adaptacion_m >= intervalo_adaptacion_m:
                ultima_adaptacion_m = distancia_acumulada

                # Calcular ratios
                distancia_restante = distancia_total - distancia_acumulada
                soc_disponible = soc_actual - soc_objetivo
                soc_inicial_disponible = soc_max - soc_objetivo

                if distancia_restante > 0 and soc_inicial_disponible > 0:
                    # Ratio de energía restante (0 = agotada, 1 = completa)
                    ratio_energia = max(0, soc_disponible / soc_inicial_disponible)
                    # Ratio de carrera restante (0 = terminada, 1 = inicio)
                    ratio_distancia = distancia_restante / distancia_total

                    # Diferencia: positivo = sobra energía, negativo = falta energía
                    diferencia = ratio_energia - ratio_distancia

                    # Ajuste proporcional del umbral
                    # diferencia > 0: sobra energía → bajar umbral (más agresivo)
                    # diferencia < 0: falta energía → subir umbral (más conservador)
                    K_adaptacion = 0.7  # Ganancia del controlador (reducida)
                    delta_umbral_raw = -diferencia * K_adaptacion

                    # === SLEW RATE LIMITER ===
                    # Limitar cambio máximo por actualización a ±0.5 m/s²
                    SLEW_RATE_MAX = 0.5  # m/s² por intervalo
                    delta_umbral_limited = np.clip(delta_umbral_raw, -SLEW_RATE_MAX, SLEW_RATE_MAX)

                    # === FILTRO EMA (Media Móvil Exponencial) ===
                    # Suaviza la transición: nuevo = α×calculado + (1-α)×anterior
                    ALPHA_EMA = 0.3  # Factor de suavizado (0.3 = 30% nuevo, 70% anterior)
                    umbral_calculado = umbral_actual + delta_umbral_limited
                    umbral_suavizado = ALPHA_EMA * umbral_calculado + (1 - ALPHA_EMA) * umbral_actual

                    umbral_actual = np.clip(umbral_suavizado, UMBRAL_MIN, UMBRAL_MAX)

            # ============================================
            # CÁLCULO DE POTENCIA (igual que antes)
            # ============================================
            if acc_i <= umbral_actual:
                # Zona de no-tracción
                tiempo_acum_pico = max(0, tiempo_acum_pico - dt * 0.5)
                if acc_i < 0:
                    P_inst = -P_regen_vector[i] + p_aux
                else:
                    P_inst = p_aux
            else:
                # Zona de tracción
                F_inert = masa * acc_i
                F_tot = F_inert + F_aero[i] + F_roll
                P_mech = F_tot * vel_i

                I_disp = I_pico_safe if tiempo_acum_pico < t_pico_max else I_cont_safe
                P_bat_max = V_pack_nom * I_disp * n_p

                F_down = 0.5 * rho * vel_i**2 * cl * area
                Peso_front = masa * 9.81 * dist_peso
                F_trac_max = (Peso_front + F_down * dist_peso) * mu
                P_grip = (F_trac_max * vel_i) / eta * margen_traccion

                P_dem = P_mech / eta if P_mech > 0 else 0
                P_real = min(P_dem, P_bat_max, P_grip)
                P_inst = P_real + p_aux

                corriente_est = P_real / V_pack_nom / n_p if V_pack_nom > 0 else 0
                if corriente_est > I_cont_safe:
                    tiempo_acum_pico += dt
                else:
                    tiempo_acum_pico = max(0, tiempo_acum_pico - dt * 0.2)

            # ============================================
            # MODELO ELÉCTRICO (SOC)
            # ============================================
            v_ocv_pack = f_ocv(soc_actual) * n_s

            if abs(r_pack_ohm) < 1e-6:
                i_pack = P_inst / v_ocv_pack if v_ocv_pack > 0 else 0
            else:
                discr = v_ocv_pack**2 - 4 * r_pack_ohm * P_inst
                if discr < 0:
                    discr = 0
                i_pack = (v_ocv_pack - np.sqrt(discr)) / (2 * r_pack_ohm)

            d_soc = -(i_pack * dt) / cap_pack_as * 100.0
            soc_actual = np.clip(soc_actual + d_soc, 0, 100)

            # ============================================
            # MODELO TÉRMICO (Pack completo)
            # ============================================
            i_cell = i_pack / n_p
            # Calor generado: todas las celdas contribuyen (pérdidas Joule)
            Q_gen_pack = n_celdas_total * (i_cell**2) * r_operative_cell  # [W]

            # Calor disipado: capacidad total del sistema de refrigeración
            Q_out_pack = refrigeracion * (T_actual - temp_amb)  # [W]

            # Variación de temperatura: masa del pack completo
            dT = ((Q_gen_pack - Q_out_pack) / (mass_pack_kg * cp)) * dt  # [°C]
            T_actual += dT

            # ============================================
            # GUARDAR RESULTADOS
            # ============================================
            t_full[idx_global] = vuelta * t_vuelta + i * dt
            soc_full[idx_global] = soc_actual
            temps_full[idx_global] = T_actual
            umbral_full[idx_global] = umbral_actual
            P_elec_full[idx_global] = P_inst
            distancia_full[idx_global] = distancia_acumulada

            idx_global += 1

    return {
        't_full': t_full,
        'soc_full': soc_full,
        'temps_full': temps_full,
        'umbral_full': umbral_full,
        'P_elec_full': P_elec_full,
        'distancia_full': distancia_full,
        'soc_final': soc_actual,
        'umbral_final': umbral_actual,
        'distancia_total': distancia_total
    }


# =============================================================================
# SIMULACIÓN CON UMBRAL FIJO (MIGRADO DESDE APP.PY)
# =============================================================================

def simular_modo_fijo(
    P_elec_pack_vuelta, v_ms_vuelta, dt, n_vueltas,
    soc_max,
    cap_celda, n_s, n_p,
    r_interna_mohm,
    temp_amb, refrigeracion, peso_celda_g,
    f_ocv,
    acc_umbral
):
    """
    Simula la carrera paso a paso con un umbral de potencia fijo.
    Reemplaza el bucle manual que existía en app.py.

    Args:
        P_elec_pack_vuelta: Array de potencia eléctrica para una vuelta (W)
        v_ms_vuelta: Array de velocidades para una vuelta (m/s)
        dt: Delta de tiempo (s)
        n_vueltas: Número de vueltas
        ... resto de parámetros físicos ...

    Returns:
        dict: Resultados completos (t_full, soc_full, etc.)
    """
    # Constantes
    FACTOR_R_TRANSITORIO = 1.25
    cp = 1000  # J/(kg·K) - Calor específico

    # Preparación de parámetros
    r_operative_cell = (r_interna_mohm / 1000.0) * FACTOR_R_TRANSITORIO
    r_pack_ohm = (r_operative_cell * n_s) / n_p
    cap_pack_as = cap_celda * n_p * 3600.0

    n_celdas_total = n_s * n_p
    mass_pack_kg = (peso_celda_g / 1000.0) * n_celdas_total

    # Arrays de salida
    n_puntos_vuelta = len(P_elec_pack_vuelta)
    n_puntos_total = n_puntos_vuelta * n_vueltas

    t_full = np.zeros(n_puntos_total)
    soc_full = np.zeros(n_puntos_total)
    temps_full = np.zeros(n_puntos_total)
    umbral_full = np.full(n_puntos_total, acc_umbral)
    P_elec_full = np.zeros(n_puntos_total)
    distancia_full = np.zeros(n_puntos_total)

    # Estado inicial
    soc_curr = soc_max
    T_curr = temp_amb
    distancia_acumulada = 0.0

    idx_global = 0
    t_ciclo = dt * n_puntos_vuelta  # Duración aproximada de una vuelta para la base de tiempo

    for vuelta in range(n_vueltas):
        t_inicio_vuelta = vuelta * t_ciclo

        for i in range(n_puntos_vuelta):
            p_inst = P_elec_pack_vuelta[i]
            vel_i = v_ms_vuelta[i]

            # --- MODELO ELÉCTRICO ---
            v_ocv_pack = f_ocv(soc_curr) * n_s

            if abs(r_pack_ohm) < 1e-6:
                i_pack = p_inst / v_ocv_pack if v_ocv_pack > 0 else 0
            else:
                discr = v_ocv_pack**2 - 4 * r_pack_ohm * p_inst
                if discr < 0: discr = 0
                i_pack = (v_ocv_pack - np.sqrt(discr)) / (2 * r_pack_ohm)

            # Actualizar SOC
            d_soc = -(i_pack * dt) / cap_pack_as * 100.0
            soc_curr = np.clip(soc_curr + d_soc, 0, 100)

            # --- MODELO TÉRMICO ---
            i_cell = i_pack / n_p
            Q_gen = n_celdas_total * (i_cell**2) * r_operative_cell
            Q_out = refrigeracion * (T_curr - temp_amb)
            dT = ((Q_gen - Q_out) / (mass_pack_kg * cp)) * dt
            T_curr += dT

            # Actualizar Distancia
            distancia_acumulada += vel_i * dt

            # Guardar datos
            t_full[idx_global] = t_inicio_vuelta + i * dt
            soc_full[idx_global] = soc_curr
            temps_full[idx_global] = T_curr
            P_elec_full[idx_global] = p_inst
            distancia_full[idx_global] = distancia_acumulada

            idx_global += 1

    return {
        't_full': t_full,
        'soc_full': soc_full,
        'temps_full': temps_full,
        'umbral_full': umbral_full,
        'P_elec_full': P_elec_full,
        'distancia_full': distancia_full,
        'soc_final': soc_curr,
        'umbral_final': acc_umbral,
        'distancia_total': distancia_acumulada
    }
