"""
Batería Lab - Simulador de Degradación de Baterías
===================================================
Aplicación principal Streamlit para simular y analizar
el comportamiento de baterías en vehículos de competición.

Autor: Batería Lab Team
Versión: 2.0 (Modularizado)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import time
import pandas as pd

# --- CONFIGURACIÓN DE PÁGINA (DEBE SER LO PRIMERO) ---
st.set_page_config(
    page_title="Simulador Degradación Batería", 
    layout="wide", 
    page_icon="🔋"
)

# --- IMPORTAR MÓDULOS PROPIOS ---
from modules.utils import (
    get_data_path, time_formatter, aplicar_estilo_dark, 
    mostrar_kpi_html, aplicar_estilos_globales, Colores
)
from modules.data_manager import (
    cargar_db_modelos, guardar_db_modelos, 
    cargar_db_packs, guardar_db_packs,
    cargar_telemetria_csv, cargar_potencia_base,
    save_benchmark_result, list_benchmark_circuits, list_benchmark_files, 
    load_benchmark_file, delete_benchmark_file,
    inicializar_session_state, save_last_config
)
from modules.physics import (
    obtener_funcion_ocv_polinomica, generar_perfil_potencia_unificado,
    calcular_soc_final_para_umbral, calcular_regeneracion,
    simular_con_umbral_adaptativo, simular_modo_fijo,
    calcular_factores_degradacion, calcular_kpis_carrera,
    calcular_dinamica_vehiculo,
    CONSTANTES_VEHICULO, CONSTANTES_ELECTRICAS
)
from modules.optimization import (
    buscar_umbral_optimo, preparar_parametros_optimizacion
)
from modules.Modulo_Inferencia import CerebroDegradacion


# =============================================================================
# INICIALIZACIÓN
# =============================================================================

# Aplicar estilos CSS
aplicar_estilos_globales()

# Inicializar Session State
inicializar_session_state()

# Recuparación preventiva de estado antes de pintar widgets (para Auto-Ajuste)
# ELIMINADO: Lógica de snapshot antigua que causaba conflictos con inputs del usuario.
# if st.session_state.get('auto_ajuste_pendiente', False):
#     snap_recov = st.session_state.get('snapshot_inputs', {})
#     if 'p_motor_max_kw' in snap_recov:
#          st.session_state.p_motor_max_kw = snap_recov['p_motor_max_kw']

# Constantes rápidas
CV = CONSTANTES_VEHICULO
CE = CONSTANTES_ELECTRICAS

# Cargar módulo de inferencia IA (cached)
@st.cache_resource
def cargar_cerebro_ia():
    """Carga el modelo de inferencia V2."""
    from modules.utils import get_data_path
    # Asegúrate de que el archivo físico se llame Modulo_Inferencia.py 
    # o actualiza el import al nombre del nuevo archivo:
    from modules.Modulo_Inferencia import CerebroDegradacion 
    
    ruta_csv = get_data_path("Resultado_Analisis_Bateria.csv")
    return CerebroDegradacion(ruta_csv)

cerebro_ia = cargar_cerebro_ia()


# =============================================================================
# FUNCIONES AUXILIARES DE CALLBACKS (UI Helpers)
# =============================================================================

def actualizar_inputs_desde_modelo():
    """
    Callback UI: Actualiza los inputs del formulario cuando se selecciona un modelo.
    
    Esta función manipula st.session_state (estado visual de la UI),
    por eso vive en app.py y no en data_manager.py.
    """
    if 'selector_modelo' in st.session_state and 'db_models' in st.session_state:
        seleccion = st.session_state.selector_modelo
        if seleccion in st.session_state.db_models['models']:
            data = st.session_state.db_models['models'][seleccion]
            st.session_state.form_nombre = seleccion
            st.session_state.form_cap = data['cap']
            st.session_state.form_vnom = data['v_nom']
            st.session_state.form_vmax = data['v_max']
            st.session_state.form_vmin = data.get('v_min', 2.5)
            st.session_state.form_peso = data['peso']
            st.session_state.form_rint = data['r_int']
            st.session_state.form_ac_imp = data.get('ac_imp', 0.0) # Cargar AC Impedance si existe
            st.session_state.form_icont = data.get('i_cont', 20.0)
            st.session_state.form_ipico = data.get('i_pico', 40.0)
            st.session_state.form_tpico = data.get('t_pico', 10.0)
            st.session_state.db_models['last_used'] = seleccion
            guardar_db_modelos(st.session_state.db_models)

def aplicar_config_pack():
    """Callback: Carga configuración de pack seleccionado y ejecuta auto-ajuste."""
    seleccion = st.session_state.selector_pack
    
    # Ignorar selección vacía
    if seleccion == "-- Seleccionar --":
        return
    
    # Protección contra recargas innecesarias
    last = st.session_state.get('last_loaded_pack', None)
    if last == seleccion:
        return
    st.session_state.last_loaded_pack = seleccion

    # Extraer nombre del pack
    if " [" in seleccion and seleccion.endswith("]"):
        nombre_pack = seleccion.rsplit(" [", 1)[0]
    else:
        nombre_pack = seleccion
    
    if nombre_pack in st.session_state.db_packs['packs']:
        st.toast(f"📥 Cargando: {nombre_pack}", icon="📦")
        data = st.session_state.db_packs['packs'][nombre_pack]
        
        # Cargar modelo de celda asociado
        if data['modelo_celda'] in st.session_state.db_models['models']:
            st.session_state.selector_modelo = data['modelo_celda']
            actualizar_inputs_desde_modelo()
        
        # Cargar topología
        st.session_state.pack_ns = data['ns']
        st.session_state.pack_np = data['np']
        st.session_state.pack_soc_max = data['soc_max']
        st.session_state.pack_soc_min = data['soc_min']
        st.session_state.nombre_pack_input = nombre_pack
        
        # Cargar configuración de motor (si existe en el pack, sino defaults)
        st.session_state.activar_limite_motor = data.get('activar_limite_motor', False)
        st.session_state.p_motor_max_kw = data.get('p_motor_max_kw', 10.0)
        
        save_last_config(nombre_pack) # PERSISTENCIA
        
        # === AUTO-AJUSTE AUTOMÁTICO AL CARGAR PACK ===
        st.session_state.auto_ajuste_pendiente = True
        st.toast("🎯 Ejecutando auto-ajuste...", icon="⚙️")



def decrementar_umbral():
    st.session_state.slider_acc = round(max(0.0, st.session_state.slider_acc - 0.01), 2)


def incrementar_umbral():
    st.session_state.slider_acc = round(min(13.0, st.session_state.slider_acc + 0.01), 2)

def actualizar_celda_pack():
    """Callback: Actualiza celda cuando cambia en config pack."""
    seleccion = st.session_state.selector_celda_pack
    if seleccion in st.session_state.db_models['models']:
        data = st.session_state.db_models['models'][seleccion]
        st.session_state.form_nombre = seleccion
        st.session_state.form_cap = data['cap']
        st.session_state.form_vnom = data['v_nom']
        st.session_state.form_vmax = data['v_max']
        st.session_state.form_vmin = data.get('v_min', 2.5)
        st.session_state.form_peso = data['peso']
        st.session_state.form_rint = data['r_int']
        st.session_state.form_ac_imp = data.get('ac_imp', 0.0)
        st.session_state.form_icont = data.get('i_cont', 20.0)
        st.session_state.form_ipico = data.get('i_pico', 40.0)
        st.session_state.form_tpico = data.get('t_pico', 10.0)
        st.session_state.db_models['last_used'] = seleccion
        guardar_db_modelos(st.session_state.db_models)

def activar_auto_ajuste():
    """Callback simple: Marca que se debe recalcular el umbral."""
    st.session_state.auto_ajuste_pendiente = True

def eliminar_pack_seleccionado():
    """Callback: Marca pack para eliminación."""
    seleccion = st.session_state.get('selector_pack', '-- Seleccionar --')
    if seleccion != "-- Seleccionar --":
        if " [" in seleccion and seleccion.endswith("]"):
            nombre = seleccion.rsplit(" [", 1)[0]
        else:
            nombre = seleccion
        
        if nombre in st.session_state.db_packs['packs']:
            del st.session_state.db_packs['packs'][nombre]
            guardar_db_packs(st.session_state.db_packs)
            st.session_state.last_loaded_pack = None
            st.session_state.pack_eliminado = nombre  # Flag para mostrar mensaje


# =============================================================================
# BARRA LATERAL - CONFIGURACIÓN
# =============================================================================

with st.sidebar:
    st.title("🔋 Batería Lab")
    
    # --- DINÁMICA DEL VEHÍCULO ---
    # --- DINÁMICA DEL VEHÍCULO ---
    with st.expander("🏎️ Pre-set Vehículo", expanded=True):
        masa_vehiculo = st.number_input("Masa Vehículo (kg)", min_value=100.0, value=320.0, step=10.0, format="%.1f")
        cd_vehiculo = st.number_input("Coeficiente de Arrastre (Cd)", min_value=0.1, value=0.40, step=0.01, format="%.2f")
        area_frontal = st.number_input("Área Frontal (m²)", min_value=0.1, value=1.24, step=0.01, format="%.2f")
        cl_downforce = st.number_input("Coeficiente Downforce (Cl)", min_value=0.0, value=2.50, step=0.1, format="%.2f")
        dist_peso_front = st.number_input("Dist. Peso Delantero", min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.2f")

        opciones_circuito = ["Germany 2012", "Germany 2010", "Austria 2012"]
        circuito_seleccionado = st.selectbox("Seleccionar Circuito", opciones_circuito)
    
    # --- SELECCIÓN DEL PACK ---
    with st.expander("📦 Selección del Pack", expanded=True):
        st.caption("Pack base para el Análisis Individual")
        
        # Mostrar mensaje si se eliminó un pack
        if 'pack_eliminado' in st.session_state:
            st.success(f"Pack '{st.session_state.pack_eliminado}' eliminado")
            del st.session_state.pack_eliminado
        
        packs_formateados = []
        for nombre_pack in sorted(st.session_state.db_packs['packs'].keys()):
            data_pack = st.session_state.db_packs['packs'][nombre_pack]
            celda = data_pack.get('modelo_celda', '?')
            packs_formateados.append(f"{nombre_pack} [{celda}]")
        
        # Si el pack seleccionado ya no existe, resetear
        current_selection = st.session_state.get('selector_pack', '-- Seleccionar --')
        if current_selection != '-- Seleccionar --':
            if " [" in current_selection and current_selection.endswith("]"):
                nombre_actual = current_selection.rsplit(" [", 1)[0]
            else:
                nombre_actual = current_selection
            if nombre_actual not in st.session_state.db_packs['packs']:
                st.session_state.selector_pack = '-- Seleccionar --'
        
        st.selectbox(
            "📂 Cargar Pack:", 
            ["-- Seleccionar --"] + packs_formateados, 
            key="selector_pack", 
            on_change=aplicar_config_pack
        )
        
        # Ficha técnica visual del pack seleccionado
        if st.session_state.selector_pack != "-- Seleccionar --":
            # Obtener nombre real del pack
            sel = st.session_state.selector_pack
            nombre_pack = sel.rsplit(" [", 1)[0] if " [" in sel else sel
            
            if nombre_pack in st.session_state.db_packs['packs']:
                p_data = st.session_state.db_packs['packs'][nombre_pack]
                modelo_celda = p_data.get('modelo_celda', 'Desconocido')
                
                # Intentar obtener datos de la celda para cálculos
                cap_celda = 0
                v_nom_celda = 0
                peso_celda = 0
                if modelo_celda in st.session_state.db_models['models']:
                    c_data = st.session_state.db_models['models'][modelo_celda]
                    cap_celda = c_data.get('cap', 0)
                    v_nom_celda = c_data.get('v_nom', 0)
                    peso_celda = c_data.get('peso', 0)
                
                # Cálculos rápidos
                ns, np_val = p_data.get('ns', 0), p_data.get('np', 0)
                soc_min_p, soc_max_p = p_data.get('soc_min', 0), p_data.get('soc_max', 100)
                v_pack = v_nom_celda * ns
                e_pack_kwh = (v_pack * cap_celda * np_val) / 1000.0
                peso_pack_kg = (peso_celda * ns * np_val) / 1000.0
                
                st.markdown(f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 5px; border: 1px solid #444; font-size: 0.85em; margin-bottom: 10px;">
                    <strong style="color: #4ade80;">🔋 DATOS PACK ACTIVO:</strong><br>
                    • <strong>Celda:</strong> {modelo_celda}<br>
                    • <strong>Config:</strong> {ns}S {np_val}P<br>
                    • <strong>Energía:</strong> {e_pack_kwh:.2f} kWh<br>
                    • <strong>Tensión:</strong> {v_pack:.1f} V<br>
                    • <strong>SOC:</strong> {soc_min_p}% - {soc_max_p}%<br>
                    • <strong>Peso:</strong> {peso_pack_kg:.1f} kg
                </div>
                """, unsafe_allow_html=True)

        # Botón Eliminar Pack seleccionado (usa callback)
        if st.session_state.selector_pack != "-- Seleccionar --":
            st.button("🗑️ Eliminar Pack Seleccionado", type="primary", 
                     use_container_width=True, key="eliminar_pack_btn",
                     on_click=eliminar_pack_seleccionado)
    
    # --- ESTRATEGIA DE CONSUMO ---
    with st.expander("🎯 Estrategia de Consumo", expanded=False):
        st.caption("Umbral de activación de tracción eléctrica")
        
        # Aplicar resultado de auto-ajuste pendiente (si existe de un rerun anterior)
        if 'auto_ajuste_resultado_pendiente' in st.session_state:
            st.session_state.slider_acc = st.session_state.auto_ajuste_resultado_pendiente
            del st.session_state.auto_ajuste_resultado_pendiente
        
        # Aplicar resultado de auto-ajuste pendiente antiguo (retrocompatibilidad)
        if 'auto_ajuste_resultado' in st.session_state:
            st.session_state.slider_acc = st.session_state.auto_ajuste_resultado
            del st.session_state.auto_ajuste_resultado
        
        # Botones de ajuste fino
        col_minus, col_valor, col_plus = st.columns([1, 2, 1])
        with col_minus:
            st.button("➖ 0.01", key="acc_minus", use_container_width=True, on_click=decrementar_umbral)
        with col_valor:
            st.markdown(f"<h3 style='text-align:center; margin:0;'>{st.session_state.slider_acc:.2f} m/s²</h3>", unsafe_allow_html=True)
        with col_plus:
            st.button("➕ 0.01", key="acc_plus", use_container_width=True, on_click=incrementar_umbral)
        
        # Slider principal
        st.slider(
            "Aceleración Umbral (m/s²)", 
            min_value=0.0, max_value=13.0, step=0.01,
            format="%.2f", key="slider_acc"
        )
        if st.session_state.get('flag_auto_ajuste_completado', False):
             val = st.session_state.slider_acc
             if val <= 0.01:
                 st.warning(f"⚠️ Saturación Inferior: Ajustado a {val:.2f} m/s². Incluso con máxima asistencia, podría sobrar batería.")
             elif val >= 12.99:
                 st.error(f"⚠️ Saturación Superior: Ajustado a {val:.2f} m/s². Incluso con asistencia mínima, podría faltar batería.")
             else:
                 st.success(f"✅ Optimizado a {val:.2f} m/s²")
             del st.session_state.flag_auto_ajuste_completado

        acc_umbral = st.session_state.slider_acc
        
        # === MODO ADAPTATIVO ===
        st.markdown("---")
        modo_adaptativo = st.toggle("🔄 Modo Adaptativo", value=False, key="modo_adaptativo",
                                     help="Ajusta el umbral automáticamente durante la carrera para llegar exactamente al SOC mínimo")
        
        if modo_adaptativo:
            st.success("✅ El umbral se ajustará cada 50m para optimizar el consumo en tiempo real")
            intervalo_adaptacion = st.slider("Intervalo de adaptación (m)", 25, 200, 50, 25, key="intervalo_adaptacion")
            margen_soc_objetivo = st.slider("Margen sobre SOC mínimo (%)", 0.0, 5.0, 0.5, 0.1, key="margen_soc_objetivo")
        else:
            intervalo_adaptacion = 50
            margen_soc_objetivo = 0.5
            # Botón Auto-Ajuste (solo en modo fijo)
            if st.button("🎯 Auto-Ajustar Umbral Inicial", key="auto_ajuste", use_container_width=True):
                activar_auto_ajuste()
                st.rerun()
            st.caption("Calcula el umbral fijo óptimo para toda la carrera")
        
        st.info(f"💡 Umbral inicial = {acc_umbral:.2f} m/s²" + (" (se adaptará)" if modo_adaptativo else " (fijo)"))

    st.markdown("---")
    st.header("📦 Diseño del Pack")

    # --- BIBLIOTECA DE CELDAS ---
    with st.expander("🔋 Biblioteca de Celdas", expanded=False):
        modelos_disponibles = list(st.session_state.db_models['models'].keys())
        last_used = st.session_state.db_models.get('last_used', modelos_disponibles[0])
        if last_used not in modelos_disponibles:
            last_used = modelos_disponibles[0]
        
        try:
            idx_def = modelos_disponibles.index(last_used)
        except:
            idx_def = 0
        
        st.selectbox("Modelo Activo:", modelos_disponibles, index=idx_def, 
                     key='selector_modelo', on_change=actualizar_inputs_desde_modelo)

        st.markdown("---")
        st.caption("Editor de Parámetros")
        st.text_input("Nombre del Modelo", key="form_nombre")
        
        col_a, col_b = st.columns(2)
        col_a.number_input("Capacidad (Ah)", 0.1, 200.0, step=0.1, key="form_cap", format="%.2f")
        col_b.number_input("Tensión Nom (V)", 0.0, 10.0, step=0.05, key="form_vnom", format="%.2f")
        col_b.number_input("Tensión Max (V)", 0.0, 10.0, step=0.05, key="form_vmax", format="%.2f")
        col_b.number_input("Tensión Min (V)", 0.0, 10.0, step=0.05, key="form_vmin", format="%.2f")
        col_a.number_input("Peso (g)", 1.0, 5000.0, step=1.0, key="form_peso", format="%.1f")
        
        # --- Lógica RDC / AC Impedance ---
        # Si el usuario escribe en AC Impedance, calculamos RDC automáticamente.
        # Si escribe en RDC, borramos AC Impedance para evitar confusión.
        
        def on_change_ac():
            if st.session_state.form_ac_imp > 0:
                st.session_state.form_rint = st.session_state.form_ac_imp * 1.35

        def on_change_rdc():
             # Si edita RDC manualmente, no hacemos nada especial (manda RDC)
             pass

        col_b.number_input("RDC. Interna (mΩ)", 0.01, 100.0, step=0.1, key="form_rint", format="%.2f", on_change=on_change_rdc, help="Resistencia en corriente directa (DC). Es el valor que usa la simulación.")
        col_a.number_input("ACImpedance (mΩ) [1kHz]", 0.0, 100.0, step=0.1, key="form_ac_imp", format="%.2f", on_change=on_change_ac, help="Si introduces este valor, se calculará RDC automáticamente (* 1.35).")
        
        st.markdown("##### Límites de Descarga")
        col_c, col_d = st.columns(2)
        col_c.number_input("I Continua (A)", 1.0, 200.0, step=1.0, key="form_icont", format="%.1f")
        col_d.number_input("I Pico (A)", 1.0, 300.0, step=1.0, key="form_ipico", format="%.1f")
        col_c.number_input("Tiempo Pico (s)", 0.5, 60.0, step=0.5, key="form_tpico", format="%.1f")
        
        col_save, col_del = st.columns([1,1])
        with col_save:
            if st.button("💾 Guardar Celda", use_container_width=True):
                # Calcular RDC final si aplica (doble check por si no saltó callback)
                if st.session_state.form_ac_imp > 0:
                     # Prioridad a la conversión si hay valor AC visiblemente activo y RDC no fue modificado despues?
                     # Simplificación: El valor visual de form_rint es el que manda al guardar.
                     pass 

                nuevo_nombre = st.session_state.form_nombre
                nueva_data = {
                    "cap": st.session_state.form_cap, 
                    "v_nom": st.session_state.form_vnom,
                    "v_max": st.session_state.form_vmax, 
                    "v_min": st.session_state.form_vmin,
                    "peso": st.session_state.form_peso,
                    "r_int": st.session_state.form_rint,
                    "ac_imp": st.session_state.form_ac_imp, # Guardamos también el valor AC por referencia
                    "i_cont": st.session_state.form_icont,
                    "i_pico": st.session_state.form_ipico,
                    "t_pico": st.session_state.form_tpico
                }
                st.session_state.db_models['models'][nuevo_nombre] = nueva_data
                st.session_state.db_models['last_used'] = nuevo_nombre
                guardar_db_modelos(st.session_state.db_models)
                st.success("Celda guardada.")
                st.rerun()
        with col_del:
            if st.button("🗑️ Eliminar", type="primary", use_container_width=True):
                nombre_borrar = st.session_state.form_nombre
                if nombre_borrar in st.session_state.db_models['models'] and len(st.session_state.db_models['models']) > 1:
                    del st.session_state.db_models['models'][nombre_borrar]
                    keys = list(st.session_state.db_models['models'].keys())
                    st.session_state.db_models['last_used'] = keys[0]
                    guardar_db_modelos(st.session_state.db_models)
                    st.rerun()

    # Variables de salida
    nombre_modelo = st.session_state.form_nombre
    cap_celda = st.session_state.form_cap
    v_nom_celda = st.session_state.form_vnom
    v_max_celda = st.session_state.form_vmax
    v_min_celda = st.session_state.form_vmin
    peso_celda_g = st.session_state.form_peso
    r_interna_mohm = st.session_state.form_rint
    i_descarga_cont = st.session_state.form_icont
    i_descarga_pico = st.session_state.form_ipico
    t_descarga_pico = st.session_state.form_tpico

    # --- CONFIGURACIÓN DEL PACK ---
    with st.expander("⚙️ Configuración del Pack", expanded=True):
        celdas_disponibles_pack = list(st.session_state.db_models['models'].keys())
        celda_actual = st.session_state.get('form_nombre', celdas_disponibles_pack[0])
        try:
            idx_celda = celdas_disponibles_pack.index(celda_actual)
        except ValueError:
            idx_celda = 0
        
        celda_seleccionada_pack = st.selectbox(
            "🔋 Celda Base:", celdas_disponibles_pack, index=idx_celda,
            key="selector_celda_pack", on_change=actualizar_celda_pack
        )
        
        st.markdown("---")
        n_s = st.slider("Series (S)", 7, 18, key="pack_ns")
        n_p = st.slider("Paralelo (P)", 1, 6, key="pack_np")
        soc_max = st.slider("SOC Máximo (%)", 0, 100, key="pack_soc_max")
        soc_min = st.slider("SOC Mínimo (%)", 0, 100, key="pack_soc_min")
        
        # Leer valores actualizados del session_state (si el callback los modificó)
        soc_max = st.session_state.pack_soc_max
        soc_min = st.session_state.pack_soc_min
        
        st.markdown("---")
        nombre_pack_input = st.text_input("Nombre del Pack", key="nombre_pack_input")
        
        if st.button("💾 Guardar Pack", use_container_width=True):
            data_pack = {
                "modelo_celda": celda_seleccionada_pack,
                "ns": n_s, "np": n_p,
                "soc_max": soc_max, "soc_min": soc_min
            }
            st.session_state.db_packs['packs'][nombre_pack_input] = data_pack
            guardar_db_packs(st.session_state.db_packs)
            save_last_config(nombre_pack_input) # Guardar como última config
            st.success("Pack Guardado")
            st.rerun()

    # --- CONDICIONES DE CONTORNO ---
    # --- CONDICIONES DE CONTORNO ---
    with st.expander("🌡️ Condiciones de Contorno", expanded=False):
        temp_amb = st.slider("Temp. Ambiente (°C)", 0.0, 60.0, 25.0, 0.5, on_change=activar_auto_ajuste)
        refrigeracion = st.slider("Refrigeración del Pack (W/K)", 0.0, 100.0, 20.0, 1.0,
                                   help="Capacidad total del sistema de refrigeración del pack. Valores típicos: 10-50 W/K", on_change=activar_auto_ajuste)
        
        st.markdown("---")
        st.caption("Estado de Salud (SOH)")
        
        # Slider invertido (100% a la izquierda -> 80% a la derecha)
        # Usamos select_slider porque st.slider obliga a min < max
        opciones_soh = list(range(100, 79, -1))
        soh_inicial = st.select_slider(
            "SOH Inicial (%)", 
            options=opciones_soh, 
            value=100, 
            key="soh_inicial",
            help="Simula la batería envejecida. Reduce capacidad y aumenta resistencia.",
            on_change=activar_auto_ajuste
        )
        st.caption("⚠️ **EOL (Fin de Vida)** se considera al **80% de SOH**.")
        
        st.markdown("---")

        activar_limite_motor = True
        p_motor_max_kw = st.slider("Potencia Máxima Motor (kW)", 6.0, 20.0, step=0.5, key="p_motor_max_kw", help="Limitación fija: Potencia mecánica máxima disponible a la rueda.", on_change=activar_auto_ajuste)

    guardar_benchmark_btn = st.sidebar.button("💾 Guardar Resultado", use_container_width=True, help="Guarda CSV + Metadatos para Benchmark Offline.")


# Validación SOC
if soc_min >= soc_max:
    soc_min = soc_max - 1
    st.sidebar.warning("SOC Mínimo ajustado automáticamente.")


# =============================================================================
# LÓGICA DE SIMULACIÓN
# =============================================================================

# Cargar datos necesarios
f_ocv_user = obtener_funcion_ocv_polinomica(v_min_celda, v_nom_celda, v_max_celda)
t_aux, p_aux, p_media_aux_w, i_media_aux_a = cargar_potencia_base()
t_telem, v_kmh, acc_telem = cargar_telemetria_csv(circuito_seleccionado)

# Configuración por circuito
VUELTAS_POR_CIRCUITO = {"Austria 2012": 20, "Germany 2010": 26, "Germany 2012": 12}
n_vueltas = VUELTAS_POR_CIRCUITO.get(circuito_seleccionado, 1)

MAPAS_CIRCUITO = {
    "Germany 2012": "mapa_germany12.png",
    "Germany 2010": "mapa_germany10.png",
    "Austria 2012": "mapa_austria12.png",
}

if t_telem is not None and len(t_telem) > 1:
    t_base = t_telem
    v_ms = v_kmh / 3.6
    
    # === APLICACIÓN DE ENVEJECIMIENTO (SOH) ===
    # Si SOH < 100%, degradamos capacidad y aumentamos resistencia
    factor_soh = soh_inicial / 100.0
    cap_celda_sim = cap_celda * factor_soh
    
    # Ajuste de límites de corriente (se degradan con la capacidad)
    i_descarga_cont_sim = i_descarga_cont * factor_soh
    i_descarga_pico_sim = i_descarga_pico * factor_soh
    
    # Cálculo de aumento de Resistencia
    # Usamos la IA para estimar el ratio de degradación típico en condiciones estándar
    r_interna_sim = r_interna_mohm
    
    if soh_inicial < 100.0 and cerebro_ia is not None:
        try:
            # Estimación rápida de C-rate promedio de la vuelta para consultar IA
            # P_mech ~ Masa * Acc * V. 
            # Aproximación muy gruesa: Potencia promedio de tracción positiva
            acc_pos = acc_telem[acc_telem > 0]
            v_pos = v_ms[acc_telem > 0]
            if len(acc_pos) > 0:
                p_mech_est = np.mean(masa_vehiculo * acc_pos * v_pos)
                p_elec_est = p_mech_est / 0.9  # Eficiencia aprox
                v_pack_est = v_nom_celda * n_s
                i_pack_est = p_elec_est / v_pack_est
                c_rate_est = i_pack_est / (cap_celda * n_p)
            else:
                c_rate_est = 1.0 # Default
                
            # Consultar IA para condiciones base
            deg_cap_base, metric_res_norm, _ = cerebro_ia.predecir_degradacion(float(c_rate_est), temp_amb, 80.0)
            
            if deg_cap_base is not None and deg_cap_base > 0:
                loss_pct = 1.0 - factor_soh
                # Ciclos equivalentes para perder ese % de capacidad
                ciclos_eq = loss_pct / deg_cap_base
                
                # Aumento de resistencia para esos ciclos
                # Delta_R = (Metric / Cap) * Ciclos
                # Nota: Usamos cap_celda original para desnormalizar, porque la métrica se refiere a la celda nueva
                delta_r_ohm = (metric_res_norm / cap_celda) * ciclos_eq
                
                r_interna_sim = r_interna_mohm + (delta_r_ohm * 1000.0) # Convertir a mOhm
                
                st.info(f"📉 **SOH {soh_inicial}%:** Simulado con {int(ciclos_eq)} ciclos equivalentes. R_int: {r_interna_mohm:.1f} → {r_interna_sim:.1f} mΩ")
        except Exception as e:
            st.warning(f"No se pudo estimar resistencia envejecida: {e}")

    # Dinámica Vehicular (usa función centralizada)
    dinamica = calcular_dinamica_vehiculo(
        v_ms, acc_telem, masa_vehiculo,
        cd=cd_vehiculo, area_frontal=area_frontal
    )
    F_aero = dinamica['F_aero']
    F_roll = dinamica['F_roll']
    P_frenos_total = dinamica['P_frenos_total']
    
    eta_total = CE.EFICIENCIA_MOTOR * CE.EFICIENCIA_INVERSOR
    
    # Regeneración - Usar SOC promedio del rango operativo
    # Esto permite regeneración incluso con SOC_max=100%
    # ya que durante la carrera el SOC bajará y tendrá margen
    soc_promedio_operativo = (soc_max + soc_min) / 2.0
    P_regen_mech, P_bat_input = calcular_regeneracion(
        P_frenos_total, v_ms, soc_promedio_operativo,
        v_max_celda, r_interna_sim, n_s, n_p,
        f_ocv_user, eta_total, cap_celda_sim
    )
    
    # === AUTO-AJUSTE ===
    # === AUTO-AJUSTE SIMPLE ===
    if st.session_state.get('auto_ajuste_pendiente', False):
        st.session_state.auto_ajuste_pendiente = False
        
        with st.spinner("🎯 Buscando umbral óptimo..."):
            dt_sim = t_telem[1] - t_telem[0] if len(t_telem) > 1 else 0.1
            
            # Recopilar parámetros directamente del entorno vivo
            params_opt = preparar_parametros_optimizacion(
                cap_celda=cap_celda_sim,        # Ya incluye degradación SOH
                r_interna_mohm=r_interna_sim,   # Ya incluye degradación SOH
                n_s=st.session_state.pack_ns, 
                n_p=st.session_state.pack_np, 
                soc_max=st.session_state.pack_soc_max,
                i_descarga_cont=i_descarga_cont_sim, 
                i_descarga_pico=i_descarga_pico_sim,
                t_descarga_pico=t_descarga_pico,
                v_nom_celda=v_nom_celda, 
                v_max_celda=v_max_celda, 
                v_min_celda=v_min_celda,
                masa_vehiculo=masa_vehiculo, 
                F_aero=F_aero, F_roll=F_roll, eta_total=eta_total,
                p_media_aux_w=p_media_aux_w, 
                dt_sim=dt_sim, 
                P_bat_input=P_bat_input,
                cl=cl_downforce, 
                area=area_frontal, 
                dist_peso=dist_peso_front,
                activar_limite_motor=st.session_state.get('activar_limite_motor', True),
                p_motor_max_kw=st.session_state.get('p_motor_max_kw', 10.0)
            )
            
            # Buscar el nuevo umbral
            umbral_opt, soc_final_opt, log_opt = buscar_umbral_optimo(
                st.session_state.pack_soc_min, # Objetivo: SOC Mínimo config
                acc_telem, v_ms, n_vueltas, params_opt
            )
            
            # Aplicar resultado y notificar
            st.session_state.auto_ajuste_resultado_pendiente = round(umbral_opt, 2)
            st.session_state.flag_auto_ajuste_completado = True
            st.rerun() # Refrescar UI con nuevo valor

    # Perfil de potencia
    dt_telem = t_telem[1] - t_telem[0] if len(t_telem) > 1 else 0.1
    MARGEN_TRACCION = 0.90
    
    P_elec_pack = generar_perfil_potencia_unificado(
        acc_telem=acc_telem, v_ms=v_ms, dt=dt_telem, acc_umbral=acc_umbral,
        masa=masa_vehiculo, F_aero=F_aero, F_roll=F_roll, eta=eta_total,
        I_cont=i_descarga_cont_sim, I_pico=i_descarga_pico_sim, t_pico_max=t_descarga_pico,
        v_nom_est=v_nom_celda, n_s=n_s, n_p=n_p,
        p_aux=p_media_aux_w, P_regen_vector=P_bat_input,
        mu=CV.MU_NEUMATICOS, rho=CV.RHO_AIRE, cl=cl_downforce,
        area=area_frontal, dist_peso=dist_peso_front,
        margen_traccion=MARGEN_TRACCION,
        activar_limite_motor=activar_limite_motor,
        p_motor_max_kw=p_motor_max_kw
    )
    
    # Límite de grip para visualización
    F_downforce = 0.5 * CV.RHO_AIRE * v_ms**2 * cl_downforce * area_frontal
    Peso_eje_del = masa_vehiculo * CV.G * dist_peso_front
    Carga_total = Peso_eje_del + F_downforce * dist_peso_front
    F_traccion_max = Carga_total * CV.MU_NEUMATICOS
    P_mecanica_grip = F_traccion_max * v_ms
    P_limite_grip = (P_mecanica_grip / eta_total) * MARGEN_TRACCION

    dt_telem = t_base[1] - t_base[0] if len(t_base) > 1 else 0.1
    dt = dt_telem  # Alias para compatibilidad
    
    # --- CÁLCULO ELÉCTRICO REFINADO (2 PASOS) ---
    
    # PASO 1: Estimación inicial con V_nominal (constante)
    v_pack_nom_total = v_nom_celda * n_s
    I_pack_est = np.divide(P_elec_pack, v_pack_nom_total, out=np.zeros_like(P_elec_pack), where=v_pack_nom_total!=0)
    
    # Resistencias
    r_nom_cell = r_interna_mohm / 1000.0
    r_operative_cell = r_nom_cell
    r_pack_total = (r_operative_cell * n_s / n_p) + CE.R_CONN_PACK
    
    # Estimar SOC y Voltaje con corriente inicial
    cap_pack_ah = cap_celda * n_p
    ah_consumed_est = np.cumsum(I_pack_est * (dt_telem / 3600.0))
    soc_est = soc_max - (ah_consumed_est / cap_pack_ah) * 100.0
    
    # OCV estimado
    ocv_est = f_ocv_user(np.clip(soc_est, 0, 100))
    v_ocv_pack = ocv_est * n_s
    
    # V_pack estimado = V_ocv - I*R_total
    v_drop_est = I_pack_est * r_pack_total
    v_pack_est = v_ocv_pack - v_drop_est
    
    # Límite físico de voltaje para el recalculo
    v_pack_max_abs = v_max_celda * n_s
    v_pack_est = np.clip(v_pack_est, 0.1, v_pack_max_abs * 1.05) # Evitar div por 0
    
    # PASO 2: Recalcular Corriente con Voltaje Estimado (Más preciso)
    I_pack_final = np.divide(P_elec_pack, v_pack_est, out=np.zeros_like(P_elec_pack), where=v_pack_est!=0)
    I_cell = I_pack_final / n_p
    
    # Recalcular SOC con corriente final
    ah_consumed = np.cumsum(I_pack_final * (dt_telem / 3600.0))
    soc_t = soc_max - (ah_consumed / cap_pack_ah) * 100.0
    
    # Recalcular Voltaje final preciso
    ocv_final = f_ocv_user(np.clip(soc_t, 0, 100))
    v_pack_t = (ocv_final * n_s) - (I_pack_final * r_pack_total)
    
    # Asignaciones para compatibilidad futura
    I_base_pack = I_pack_final
    mass_celda_kg = peso_celda_g / 1000.0
    mass_pack_kg = mass_celda_kg * n_s * n_p
    


    
    # Límites físicos de tensión (no puede superar V_max ni bajar de V_min)
    v_pack_max = v_max_celda * n_s  # Máximo durante carga
    v_pack_min = 2.5 * n_s          # Mínimo seguro (corte inferior)
    v_pack_t = np.clip(v_pack_t, v_pack_min, v_pack_max)
    
    p_pack_t = v_pack_t * I_base_pack
    
    # --- SIMULACIÓN TÉRMICA N VUELTAS ---
    t_vuelta = t_base[-1] - t_base[0] if len(t_base) > 1 else 60.0
    
    # Verificar si modo adaptativo está activo
    modo_adaptativo = st.session_state.get('modo_adaptativo', False)
    intervalo_adaptacion = st.session_state.get('intervalo_adaptacion', 50)
    margen_soc_objetivo = st.session_state.get('margen_soc_objetivo', 0.5)
    
    if modo_adaptativo:
        # === SIMULACIÓN CON UMBRAL ADAPTATIVO ===
        resultado_sim = simular_con_umbral_adaptativo(
            acc_telem=acc_telem, v_ms=v_ms, dt=dt_telem, n_vueltas=n_vueltas,
            masa=masa_vehiculo, F_aero=F_aero, F_roll=F_roll, eta=eta_total,
            I_cont=i_descarga_cont_sim, I_pico=i_descarga_pico_sim, t_pico_max=t_descarga_pico,
            v_nom_celda=v_nom_celda, v_max_celda=v_max_celda, v_min_celda=v_min_celda,
            n_s=n_s, n_p=n_p, cap_celda=cap_celda_sim,
            soc_max=soc_max, soc_min=soc_min, r_interna_mohm=r_interna_sim,
            p_aux=p_media_aux_w, P_regen_vector=P_bat_input,
            mu=CV.MU_NEUMATICOS, rho=CV.RHO_AIRE, cl=CV.CL_DOWNFORCE,
            area=CV.AREA_FRONTAL, dist_peso=CV.DISTRIBUCION_PESO_DELANT,
            temp_amb=temp_amb, refrigeracion=refrigeracion,
            peso_celda_g=peso_celda_g,
            umbral_inicial=acc_umbral,
            intervalo_adaptacion_m=float(intervalo_adaptacion),
            margen_soc_final=margen_soc_objetivo,
            margen_traccion=MARGEN_TRACCION
        )
        
        t_full = resultado_sim['t_full']
        soc_full = resultado_sim['soc_full']
        temps_full = resultado_sim['temps_full']
        umbral_full = resultado_sim['umbral_full']
        P_elec_full = resultado_sim['P_elec_full']
        distancia_full = resultado_sim['distancia_full']
        r_pack_full = resultado_sim.get('r_pack_full', np.zeros_like(t_full))
        umbral_final = resultado_sim['umbral_final']
        
    else:
        # === SIMULACIÓN CON UMBRAL FIJO (usa función centralizada) ===
        resultado_sim = simular_modo_fijo(
            P_elec_pack_vuelta=P_elec_pack,
            v_ms_vuelta=v_ms,
            dt=dt,
            n_vueltas=n_vueltas,
            soc_max=soc_max,
            cap_celda=cap_celda_sim,
            n_s=n_s,
            n_p=n_p,
            r_interna_mohm=r_interna_sim,
            temp_amb=temp_amb,
            refrigeracion=refrigeracion,
            peso_celda_g=peso_celda_g,
            f_ocv=f_ocv_user,
            acc_umbral=acc_umbral
        )
        
        t_full = resultado_sim['t_full']
        soc_full = resultado_sim['soc_full']
        temps_full = resultado_sim['temps_full']
        umbral_full = resultado_sim['umbral_full']
        r_pack_full = resultado_sim.get('r_pack_full', np.zeros_like(t_full))
        umbral_final = resultado_sim['umbral_final']
    
    t_max = np.max(temps_full)
    
    # --- DEGRADACIÓN (usa función centralizada) ---
    resultado_deg = calcular_factores_degradacion(
        I_cell_array=I_cell,
        cap_celda=cap_celda,
        temps_full=temps_full,
        soc_max=soc_max,
        soc_min=soc_min,
        cerebro_ia=cerebro_ia
    )
    
    ciclos = resultado_deg['ciclos_vida']
    metodo_degradacion = resultado_deg['metodo']
    motivo_fallback = resultado_deg.get('motivo_fallback')
    texto_vida = "> 90.000" if ciclos > 90000 else f"{int(ciclos)}"
    
    # Guardar info del método para mostrar en UI
    st.session_state.degradacion_metodo = metodo_degradacion
    st.session_state.degradacion_motivo_fallback = motivo_fallback
    
    # Alertas
    v_status_msg = "VOLTAJE OK"
    v_status_color = Colores.OK
    if np.any(v_pack_t > 60.0):
        v_status_msg = "FALLO: V > 60V"
        v_status_color = Colores.ERROR
    elif np.any(v_pack_t < 9.5):
        v_status_msg = "FALLO: V < 9.5V"
        v_status_color = Colores.ERROR
    
    # --- CÁLCULO DE KPIs (usa función centralizada) ---
    kpis = calcular_kpis_carrera(
        P_elec_pack=P_elec_pack,
        I_cell=I_cell,
        v_ms=v_ms,
        soc_full=soc_full,
        dt=dt_telem,
        n_vueltas=n_vueltas,
        p_aux=p_media_aux_w,
        i_descarga_cont=i_descarga_cont,
        v_nom_celda=v_nom_celda,
        n_s=n_s,
        n_p=n_p,
        cap_celda=cap_celda,
        peso_celda_g=peso_celda_g,
        soc_max=soc_max,
        soc_min=soc_min,
        r_interna_mohm=r_interna_mohm
    )
    
    # Extraer valores para visualización
    E_regen_vuelta = kpis['E_regen_vuelta']
    E_regen_total = kpis['E_regen_total']
    E_consumo_vuelta = kpis['E_consumo_vuelta']
    E_consumo_total = kpis['E_consumo_total']
    E_termica_vuelta = kpis['E_termica_vuelta']
    E_termica_total = kpis['E_termica_total']

    E_real_disp = kpis['E_real_disp']
    E_virtual = kpis['E_virtual']
    t_pico_desc = kpis['t_pico_desc']
    t_pico_carga = kpis['t_pico_carga']
    p_max_desc_kw = kpis['p_max_desc_kw']
    p_max_carga_kw = kpis['p_max_carga_kw']
    soc_final_real = kpis['soc_final_real']
    soc_disponible = kpis['soc_disponible']
    distancia_total_km = kpis['distancia_total_km']
    peso_pack_kg = kpis['peso_pack_kg']
    cap_pack_ah = kpis['cap_pack_ah']


    # =============================================================================
    # VISUALIZACIÓN PRINCIPAL
    # =============================================================================
    
    col_titulo, col_mapa = st.columns([3, 1])
    with col_titulo:
        st.title("🔋 Simulador de Baterías")
        st.markdown(f"Circuito: **{circuito_seleccionado}** | Monitorización de perfiles eléctricos y térmicos.")
    with col_mapa:
        mapa_archivo = MAPAS_CIRCUITO.get(circuito_seleccionado, None)
        if mapa_archivo:
            mapa_ruta = get_data_path(mapa_archivo, "mapas")
            if os.path.exists(mapa_ruta):
                st.image(mapa_ruta, use_container_width=True)

    # --- PESTAÑAS PRINCIPALES ---
    tab_single, tab_bench = st.tabs(["🏎️ Análisis Individual", "📊 Benchmark Comparativo"])

    with tab_single:
        # KPIs Fila 0
        col_pack, col_soc_range, col_topo = st.columns(3)
        with col_pack:
            nombre_pack_disp = st.session_state.get('nombre_pack_input', 'Pack')
            mostrar_kpi_html("Pack Activo", f"{nombre_pack_disp} [{nombre_modelo}]", "", "#6366f1")
        with col_soc_range:
            mostrar_kpi_html("Intervalo SOC", f"{soc_max}% → {soc_min}%", f"ΔDoD: {soc_max - soc_min}%", "#22d3ee")
        with col_topo:
            v_pack_nom = n_s * v_nom_celda
            mostrar_kpi_html("Configuración", f"{n_s}S {n_p}P", f"V_nom: {v_pack_nom:.1f}V", "#a855f7")
        
        # KPIs Fila 1
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: mostrar_kpi_html("Estado Voltaje", v_status_msg, "", v_status_color)
        with col2: mostrar_kpi_html("Temp. Máxima", f"{t_max:.1f}", "°C", Colores.ERROR if t_max > 60 else Colores.OK)
        with col3: mostrar_kpi_html("Vida Estimada", texto_vida, f"Ciclos ({metodo_degradacion})", Colores.INFO)
        with col4:
            r_cell_mohm = r_interna_mohm
            r_pack_mohm = r_pack_total * 1000
            mostrar_kpi_html("R Pack/Celda", f"{r_pack_mohm:.1f} / {r_cell_mohm:.1f}", "mΩ", Colores.WARNING)
        with col5: mostrar_kpi_html("Peso Pack", f"{peso_pack_kg:.1f}", "kg", "#9c27b0")

        # KPIs Fila 2
        col6, col7, col_term, col8, col9 = st.columns(5)
        with col6: mostrar_kpi_html("E. Regen Total", f"{E_regen_total:.0f}", "Wh", Colores.OK)
        with col7: mostrar_kpi_html("Consumo Total", f"{E_consumo_total:.0f}", "Wh", Colores.POTENCIA_DESC)
        with col_term: mostrar_kpi_html("Pérdidas Totales", f"{E_termica_total:.0f}", "Wh", "#ff5722")
        E_virtual_real = E_virtual - E_termica_total
        with col8: mostrar_kpi_html("E. Real / Virtual I / Virtual R", f"{E_real_disp:.0f} / {E_virtual:.0f} / {E_virtual_real:.0f}", "Wh", Colores.INFO)
        with col9: mostrar_kpi_html("T. Pico (Push/Carga)", f"{t_pico_desc:.1f} / {t_pico_carga:.1f}", "s", "#e83e8c")

        # KPIs Fila 3
        col10, col11, col12 = st.columns(3)
        with col10: mostrar_kpi_html("P. Max", f"{p_max_desc_kw:.1f} / {p_max_carga_kw:.1f}", "kW", "#fd7e14")
        with col11:
            color_soc = Colores.OK if soc_disponible >= 0 else Colores.ERROR
            mostrar_kpi_html("SOC Disp/Rest", f"{soc_disponible:.1f}% / {soc_final_real:.1f}%", "", color_soc)
        with col12: mostrar_kpi_html("Distancia", f"{distancia_total_km:.1f}", f"km ({n_vueltas} Vueltas)", "#6610f2")
        # --- INFO MÉTODO DE DEGRADACIÓN ---
        metodo = st.session_state.get('degradacion_metodo', 'BD')
        fallback_msg = st.session_state.get('degradacion_motivo_fallback', None)
        
        if metodo == "IA":
            st.success("🤖 Degradación calculada con **Inteligencia Artificial** (Red Neuronal)")
        else:
            msg = f"⚠️ Degradación calculada con **Modelo Genérico (Base de Datos)**. "
            if fallback_msg:
                msg += f"Motivo: *{fallback_msg}*"
            st.warning(msg)

        # --- GRÁFICOS ---
        tab1, tab2, tab3 = st.tabs(["⚡ Eléctrico", "🌡️ Térmico & SOC", "🏎️ Dinámico"])

        with tab1:
            st.subheader("Tensión, Corriente y Potencia")
            fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            
            ax1.plot(t_base, v_pack_t, color=Colores.TENSION, linewidth=1)
            ax1.fill_between(t_base, v_pack_t, alpha=0.2, color=Colores.TENSION)
            aplicar_estilo_dark(ax1, "Tensión del Pack", "", "Tensión (V)")
            ax1.set_ylim(35, 65)
            
            ax2.plot(t_base, I_base_pack, color=Colores.CORRIENTE, linewidth=1)
            ax2.fill_between(t_base, I_base_pack, alpha=0.2, color=Colores.CORRIENTE)
            aplicar_estilo_dark(ax2, "Corriente del Pack", "", "Corriente (A)")
            
            P_pack_kw = (v_pack_t * I_base_pack) / 1000.0
            P_desc = np.maximum(P_pack_kw, 0)
            P_carga = np.minimum(P_pack_kw, 0)
            ax3.fill_between(t_base, P_desc, color=Colores.POTENCIA_DESC, alpha=0.5)
            ax3.plot(t_base, P_desc, color=Colores.POTENCIA_DESC, linewidth=0.8)
            ax3.fill_between(t_base, P_carga, color=Colores.POTENCIA_CARGA, alpha=0.5)
            ax3.plot(t_base, P_carga, color=Colores.POTENCIA_CARGA, linewidth=0.8)
            ax3.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
            aplicar_estilo_dark(ax3, "Potencia del Pack", "Tiempo", "Potencia (kW)")
            ax3.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            
            plt.tight_layout()
            plt.tight_layout()
            st.pyplot(fig1)

            # --- NUEVO: GRÁFICO DE RESISTENCIA DEL PACK (1 VUELTA) ---
            st.markdown("---")
            st.subheader("📊 Resistencia Interna Pack (1 Vuelta)")
            
            fig_r_lap, ax_r_lap = plt.subplots(figsize=(12, 3))
            
            # Tomar solo la primera vuelta (la longitud de t_base)
            n_puntos_vuelta = len(t_base)
            # Asegurar que r_pack_full tenga datos (si no se corrió simulación, puede estar vacío o ceros)
            if 'r_pack_full' in locals() and len(r_pack_full) >= n_puntos_vuelta:
                r_lap_mohm = r_pack_full[:n_puntos_vuelta] * 1000.0
                
                ax_r_lap.plot(t_base, r_lap_mohm, color='#ff9800', linewidth=1.2)
                ax_r_lap.fill_between(t_base, r_lap_mohm, alpha=0.2, color='#ff9800')
                
                # Referencia base
                r_base_pack_mohm = (r_interna_mohm * n_s / n_p)
                ax_r_lap.axhline(r_base_pack_mohm, color='white', linestyle='--', alpha=0.5, label='R Base')
                
                ax_r_lap.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
            
            ax_r_lap.set_ylabel("Resistencia (mΩ)", color='#ff9800')
            ax_r_lap.tick_params(axis='y', labelcolor='#ff9800')
            aplicar_estilo_dark(ax_r_lap, "Dinámica de Impedancia (Vuelta 1)", "Tiempo", "")
            ax_r_lap.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            
            plt.tight_layout()
            st.pyplot(fig_r_lap)
            
            # --- PERFIL AUXILIAR (TAB 1) ---
            st.markdown("---")
            st.subheader("🔌 Perfil de Potencia Auxiliar (Sistemas)")
            st.markdown(f"**Potencia Media:** {p_media_aux_w:.1f} W | **Corriente Media:** {i_media_aux_a:.1f} A")
            
            fig_aux, ax_aux = plt.subplots(figsize=(10, 3))
            
            # Si hay perfil real cargado
            if t_aux is not None and len(t_aux) > 0:
                ax_aux.step(t_aux, p_aux, where='post', color='#00bcd4', linewidth=1.5)
                ax_aux.fill_between(t_aux, p_aux, step='post', alpha=0.2, color='#00bcd4')
                # Línea de valor medio
                ax_aux.axhline(p_media_aux_w, color='white', linestyle='--', linewidth=1, alpha=0.7, label=f"Media: {p_media_aux_w:.1f}W")
                ax_aux.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
                aplicar_estilo_dark(ax_aux, "Consumo Auxiliares (Base)", "Tiempo (s)", "Potencia (W)")
            else:
                ax_aux.axhline(p_media_aux_w, color='#00bcd4', linestyle='-', linewidth=2)
                ax_aux.text(t_base[-1]/2, p_media_aux_w + 10, f"Constante: {p_media_aux_w} W", color='#00bcd4', ha='center')
                aplicar_estilo_dark(ax_aux, "Consumo Auxiliares (Constante)", "Tiempo", "Potencia (W)")
                ax_aux.set_xlim(0, t_base[-1])
            
            st.pyplot(fig_aux)
            
            # --- CURVA OCV vs SOC ---
            st.markdown("---")
            st.subheader("📈 Curva OCV vs SOC")
            
            # Generar curva OCV para todo el rango de SOC
            soc_range = np.linspace(0, 100, 101)
            ocv_celda = f_ocv_user(soc_range)
            ocv_pack = ocv_celda * n_s
            
            fig_ocv, ax_ocv = plt.subplots(figsize=(12, 4))
            
            # Eje principal: Tensión del Pack
            ax_ocv.plot(soc_range, ocv_pack, color=Colores.TENSION, linewidth=2, label='Pack')
            ax_ocv.fill_between(soc_range, ocv_pack, alpha=0.2, color=Colores.TENSION)
            ax_ocv.set_ylabel(f"Tensión Pack (V) - {n_s}S", color=Colores.TENSION)
            ax_ocv.tick_params(axis='y', labelcolor=Colores.TENSION)
            
            # Líneas de referencia SOC
            ax_ocv.axvline(soc_max, color=Colores.OK, linestyle='--', alpha=0.7, label=f'SOC Max ({soc_max}%)')
            ax_ocv.axvline(soc_min, color=Colores.WARNING, linestyle='--', alpha=0.7, label=f'SOC Min ({soc_min}%)')
            
            # Eje secundario: Tensión de celda
            ax_cell = ax_ocv.twinx()
            ax_cell.plot(soc_range, ocv_celda, color='#9c27b0', linewidth=1.5, linestyle=':', alpha=0.7, label='Celda')
            ax_cell.set_ylabel("Tensión Celda (V)", color='#9c27b0')
            ax_cell.tick_params(axis='y', labelcolor='#9c27b0')
            
            aplicar_estilo_dark(ax_ocv, "Open Circuit Voltage vs State of Charge", "SOC (%)", "")
            ax_ocv.set_xlim(100, 0)  # Invertido: 100 a 0 (izquierda a derecha)
            ax_ocv.legend(loc='lower left', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
            
            plt.tight_layout()
            st.pyplot(fig_ocv)


        with tab2:
            st.subheader(f"Perfil Térmico y SOC ({n_vueltas} vueltas)")
            fig_t, ax_soc = plt.subplots(figsize=(12, 5))
            ax_temp = ax_soc.twinx()
            
            # Eje Izquierdo: SOC (%)
            ax_soc.plot(t_full, soc_full, color=Colores.SOC, linewidth=1.5)
            ax_soc.fill_between(t_full, soc_full, alpha=0.15, color=Colores.SOC)
            ax_soc.axhline(soc_max, color=Colores.OK, linestyle='--', alpha=0.6)
            ax_soc.axhline(soc_min, color=Colores.OK, linestyle='--', alpha=0.6)
            ax_soc.set_ylabel("SOC (%)", color=Colores.SOC)
            ax_soc.tick_params(axis='y', labelcolor=Colores.SOC)
            ax_soc.set_ylim(0, 100)
            
            # Eje Derecho: Temperatura (°C)
            ax_temp.plot(t_full, temps_full, color=Colores.TEMPERATURA, linewidth=1.5)
            ax_temp.fill_between(t_full, temps_full, alpha=0.2, color=Colores.TEMPERATURA)
            ax_temp.axhline(60, color='red', linestyle='--', alpha=0.5)
            ax_temp.set_ylabel("Temperatura (°C)", color=Colores.TEMPERATURA)
            ax_temp.tick_params(axis='y', labelcolor=Colores.TEMPERATURA)
            ax_temp.set_ylim(20, max(70, np.max(temps_full) + 5))
            
            for v in range(1, n_vueltas):
                ax_soc.axvline(x=v * t_vuelta, color='white', linestyle=':', alpha=0.2)
            
            aplicar_estilo_dark(ax_soc, f"SOC vs Temp - {circuito_seleccionado}", "Tiempo", "")
            ax_temp.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            
            plt.tight_layout()
            st.pyplot(fig_t)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temp. Máxima", f"{np.max(temps_full):.1f} °C")
            c2.metric("Temp. Promedio", f"{np.mean(temps_full):.1f} °C")
            c3.metric("SOC Inicial", f"{soc_max:.0f} %")
            c4.metric("SOC Final", f"{soc_full[-1]:.1f} %")


            
            # === GRÁFICO DE UMBRAL ADAPTATIVO (solo si está activo) ===
            
            # --- NUEVO: GRÁFICO DE RESISTENCIA DEL PACK ---
            st.markdown("---")
            st.subheader("📊 Evolución de Resistencia Interna (Pack)")
            
            fig_r, ax_r = plt.subplots(figsize=(12, 3))
            
            # Convertir a mOhms para visualización más clara
            r_pack_mohm_full = r_pack_full * 1000.0
            
            ax_r.plot(t_full, r_pack_mohm_full, color='#ff9800', linewidth=1.2)
            ax_r.fill_between(t_full, r_pack_mohm_full, alpha=0.2, color='#ff9800')
            ax_r.set_ylabel("Resistencia Pack (mΩ)", color='#ff9800')
            ax_r.tick_params(axis='y', labelcolor='#ff9800')
            
            # Línea de referencia base (estática)
            r_base_pack_mohm = (r_interna_mohm * n_s / n_p)
            ax_r.axhline(r_base_pack_mohm, color='white', linestyle='--', alpha=0.5, label='R Base (Estática)')
            ax_r.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
            
            for v in range(1, n_vueltas):
                ax_r.axvline(x=v * t_vuelta, color='white', linestyle=':', alpha=0.2)
                
            aplicar_estilo_dark(ax_r, "Dinámica de Impedancia Interna", "Tiempo", "")
            ax_r.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            
            plt.tight_layout()
            st.pyplot(fig_r)

            # === GRÁFICO DE UMBRAL ADAPTATIVO (solo si está activo) ===
            if modo_adaptativo:
                st.markdown("---")
                st.subheader("🔄 Evolución del Umbral Adaptativo")
                
                fig_umbral, ax_umb = plt.subplots(figsize=(12, 3))
                ax_umb.plot(t_full, umbral_full, color='#ffd700', linewidth=1.5)
                ax_umb.fill_between(t_full, umbral_full, alpha=0.2, color='#ffd700')
                ax_umb.axhline(acc_umbral, color='white', linestyle='--', alpha=0.5, label='Umbral inicial')
                ax_umb.set_ylabel("Umbral (m/s²)", color='#ffd700')
                ax_umb.tick_params(axis='y', labelcolor='#ffd700')
                ax_umb.set_ylim(1.5, 12.5)
                
                for v in range(1, n_vueltas):
                    ax_umb.axvline(x=v * t_vuelta, color='white', linestyle=':', alpha=0.2)
                
                aplicar_estilo_dark(ax_umb, "Control Adaptativo del Umbral", "Tiempo", "")
                ax_umb.xaxis.set_major_formatter(FuncFormatter(time_formatter))
                
                plt.tight_layout()
                st.pyplot(fig_umbral)
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Umbral Inicial", f"{acc_umbral:.2f} m/s²")
                col_b.metric("Umbral Final", f"{umbral_final:.2f} m/s²")
                col_c.metric("Variación", f"{umbral_final - acc_umbral:+.2f} m/s²")

        with tab3:
            st.subheader("🏎️ Dinámica del Vehículo (1 Vuelta)")
            
            # --- PERFIL DE ACELERACIÓN ---
            fig_acc, ax_acc = plt.subplots(figsize=(12, 3))
            acc_pos = np.maximum(acc_telem, 0)
            acc_neg = np.minimum(acc_telem, 0)
            ax_acc.fill_between(t_telem, acc_pos, color=Colores.ACELERACION, alpha=0.5)
            ax_acc.plot(t_telem, acc_pos, color=Colores.ACELERACION, linewidth=1)
            ax_acc.fill_between(t_telem, acc_neg, color=Colores.FRENADA, alpha=0.5)
            ax_acc.plot(t_telem, acc_neg, color=Colores.FRENADA, linewidth=1)
            ax_acc.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
            aplicar_estilo_dark(ax_acc, "Perfil de Aceleración", "Tiempo", "Aceleración (m/s²)")
            ax_acc.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            
            plt.tight_layout()
            st.pyplot(fig_acc)
            
            # --- FRENADA REGENERATIVA ---
            st.markdown("---")
            st.subheader("⚓ Análisis de Frenada Regenerativa")
            
            # Recalcular regeneración detallada para 1 vuelta
            soc_vis = (soc_max + soc_min) / 2
            P_regen_mech_vis, P_bat_input_vis = calcular_regeneracion(
                P_frenos_total, v_ms, soc_vis,
                v_max_celda, r_interna_mohm, n_s, n_p,
                f_ocv_user, eta_total, cap_celda
            )
            
            fig_regen, ax_reg = plt.subplots(figsize=(10, 4))
            ax_reg.fill_between(t_base, P_frenos_total/1000, 0, color='gray', alpha=0.3, label="Frenada Total (Mecánica)")
            P_recup_kw = P_regen_mech_vis / 1000 
            ax_reg.fill_between(t_base, P_recup_kw, 0, color=Colores.POTENCIA_CARGA, alpha=0.8, label="Regeneración (Recuperada)")
            ax_reg.plot(t_base, P_recup_kw, color='#4caf50', linewidth=1)
            aplicar_estilo_dark(ax_reg, "Distribución de Frenada", "Tiempo", "Potencia (kW)")
            ax_reg.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
            ax_reg.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            st.pyplot(fig_regen)
            
            c_r1, c_r2, c_r3 = st.columns(3)
            e_freno_total = np.trapz(P_frenos_total, t_base) / 3600
            e_recuperada = np.trapz(P_regen_mech_vis, t_base) / 3600
            c_r1.metric("Energía Total", f"{e_freno_total:.1f} Wh")
            c_r2.metric("Energía Recuperada", f"{e_recuperada:.1f} Wh")
            val_ef = (e_recuperada/e_freno_total)*100 if e_freno_total > 0 else 0
            c_r3.metric("Eficiencia", f"{val_ef:.1f} %")
            
            st.markdown("---")
            
            fig_dyn, ax_dyn = plt.subplots(figsize=(12, 5))
            ax_power = ax_dyn.twinx()
            
            ax_dyn.plot(t_telem, v_kmh, color=Colores.GRIS, linewidth=1, alpha=0.6)
            ax_dyn.fill_between(t_telem, v_kmh, color=Colores.GRIS, alpha=0.1)
            ax_dyn.set_ylabel("Velocidad (km/h)", color=Colores.GRIS)
            ax_dyn.tick_params(axis='y', labelcolor=Colores.GRIS)
            ax_dyn.set_ylim(bottom=0)
            
            P_elec_kw = P_elec_pack / 1000.0
            P_trac_kw = np.maximum(P_elec_kw, 0)
            P_reg_kw = np.minimum(P_elec_kw, 0)
            
            ax_power.plot(t_telem, P_limite_grip / 1000.0, color=Colores.GRIP, linestyle='--', linewidth=1.5, alpha=0.8)
            ax_power.fill_between(t_telem, P_trac_kw, color=Colores.POTENCIA_DESC, alpha=0.5)
            ax_power.fill_between(t_telem, P_reg_kw, color=Colores.POTENCIA_CARGA, alpha=0.5)
            ax_power.set_ylabel("Potencia Motor (kW)", color=Colores.POTENCIA_DESC)
            ax_power.tick_params(axis='y', labelcolor=Colores.POTENCIA_DESC)
            ax_power.axhline(y=0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
            
            aplicar_estilo_dark(ax_dyn, "Dinámica del Vehículo", "Tiempo", "")
            ax_dyn.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            
            plt.tight_layout()
            st.pyplot(fig_dyn)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Velocidad Máx", f"{np.max(v_kmh):.1f} km/h")
            c2.metric("Velocidad Media", f"{np.mean(v_kmh):.1f} km/h")
            c3.metric("Pot. Máx Tracción", f"{np.max(P_elec_kw):.1f} kW")
            c4.metric("Pot. Máx Regen", f"{-np.min(P_elec_kw):.1f} kW")

    # === NUEVO BENCHMARK OFFLINE ===
    with tab_bench:
        st.subheader("📊 Benchmark Comparativo (Offline)")
        st.markdown("""
        **Modo Offline:** Visualiza y compara simulaciones guardadas anteriormente.
        Selecciona un circuito y los archivos que deseas analizar.
        """)
        
        # 1. Selector de Circuito
        circuitos_disp = list_benchmark_circuits()
        
        if not circuitos_disp:
            st.info("📂 No hay simulaciones guardadas. Ve a 'Análisis Individual' -> 'Guardar Resultado' para añadir datos.")
        else:
            circuito_bench = st.selectbox("📂 Seleccionar Circuito Comparativo", circuitos_disp)
            
            # Layout: Mapa + Selector Archivos
            col_map_bench, col_sel_bench = st.columns([1, 2])
            
            with col_map_bench:
                # Mostrar Mapa si existe
                mapa_archivo_bench = MAPAS_CIRCUITO.get(circuito_bench, None)
                if mapa_archivo_bench:
                     path_map = get_data_path(mapa_archivo_bench, "mapas")
                     if os.path.exists(path_map):
                         st.image(path_map, use_container_width=True, caption=f"Circuito: {circuito_bench}")
                else:
                    st.markdown(f"**{circuito_bench}**")

            with col_sel_bench:
                # 2. Selector de Archivos CSV
                files_disp = list_benchmark_files(circuito_bench)
                if not files_disp:
                    st.warning("Carpeta vacía.")
                else:
                    files_sel = st.multiselect(
                        "Seleccionar Simulaciones:",
                        files_disp,
                        default=files_disp if len(files_disp) <= 3 else files_disp[:3]
                    )
                    
                    # Botón de Eliminar
                    if st.button("🗑️ Eliminar Archivos Seleccionados", type="primary", key="btn_del_bench"):
                        if files_sel:
                            borrados = 0
                            for f_del in files_sel:
                                ok, msg = delete_benchmark_file(circuito_bench, f_del)
                                if ok: borrados += 1
                            st.success(f"Se eliminaron {borrados} archivos.")
                            st.rerun()
                        else:
                            st.warning("Selecciona archivos para eliminar.")

            # 3. Visualización y Tabla
            if 'files_sel' in locals() and files_sel:
                st.markdown("---")
                
                # Gráfico Comparativo
                fig_bench, ax_bench = plt.subplots(figsize=(12, 6))
                
                resultados_meta = []
                
                for f_name in files_sel:
                    t_b, temp_b, meta_b = load_benchmark_file(circuito_bench, f_name)
                    
                    if t_b is not None and len(t_b) > 0:
                        # Limpiar nombre (quitar .csv)
                        label_clean = f_name.replace(".csv", "")
                        ax_bench.plot(t_b, temp_b, linewidth=2, label=label_clean)
                        
                        # Guardar metadatos para tabla
                        row = {'Nombre': label_clean}
                        # Añadir metadatos clave con nombres amigables
                        for k, v in meta_b.items():
                            row[k] = v
                        resultados_meta.append(row)
                
                aplicar_estilo_dark(ax_bench, f"Evolución Térmica - {circuito_bench}", "Tiempo (s)", "Temperatura (°C)")
                ax_bench.legend()
                # Línea de límite térmico general
                ax_bench.axhline(60, color='red', linestyle='--', alpha=0.5, label="Límite 60°C")
                
                st.pyplot(fig_bench)
                
                # Scatter Plot: Vida Estimada vs E. Virtual R
                if resultados_meta:
                    st.subheader("Vida Estimada vs E. Virtual R")
                    
                    fig_sc, ax_sc = plt.subplots(figsize=(10, 6))
                    
                    x_vals = []
                    y_vals = []
                    labels = []
                    
                    for row in resultados_meta:
                        try:
                            # Parse Vida: "2350 (Ciclos)" -> 2350.0
                            v_raw = str(row.get('Vida_Estimada', '0')).split()[0]
                            # Limpiar caracteres no numéricos si los hubiera
                            v_clean = "".join([c for c in v_raw if c.isdigit() or c == '.'])
                            vid = float(v_clean) if v_clean else 0.0
                            
                            # Parse Energía: "12300.50" -> 12300.5
                            e_raw = str(row.get('Energia_Virtual_Real_Wh', '0'))
                            e_clean = "".join([c for c in e_raw if c.isdigit() or c == '.'])
                            eng = float(e_clean) if e_clean else 0.0
                            
                            x_vals.append(vid)
                            y_vals.append(eng)
                            labels.append(row.get('Nombre', '?'))
                        except:
                            pass
                    
                    if x_vals:
                        # Plot Scatter
                        ax_sc.scatter(x_vals, y_vals, s=150, c='#00bcd4', alpha=0.8, edgecolors='white', zorder=3)
                        
                        # Etiquetas
                        for i, txt in enumerate(labels):
                            ax_sc.annotate(txt, (x_vals[i], y_vals[i]), 
                                           xytext=(0, 10), textcoords='offset points', 
                                           ha='center', fontsize=9, color='white', fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#444", alpha=0.8))
                        
                        aplicar_estilo_dark(ax_sc, "Pareto: Vida vs Energía", "Vida Estimada (Ciclos)", "Energía Virtual Real (Wh)")
                        ax_sc.grid(True, linestyle=':', alpha=0.3)
                        st.pyplot(fig_sc)
                    else:
                        st.warning("No se encontraron datos numéricos válidos en los metadatos.")
            
            elif 'files_sel' in locals():
                st.info("Selecciona al menos una simulación para ver la comparativa.")

    # Bloque Legacy (Desactivado)
    if False: # with tab_bench:
        st.subheader("📊 Benchmark Comparativo")
        st.markdown("""
        **Objetivo:** Identificar el diseño (Pack + Celda) más eficiente y con menor degradación.
        Selecciona varios candidatos de tu biblioteca de packs para simularlos simultáneamente.
        """)
        
        # Selector Multi-Select de Packs
        packs_benchmark = []
        mapa_nombres = {}
        
        for nombre_pack, data_pack in st.session_state.db_packs['packs'].items():
            celda = data_pack.get('modelo_celda', '?')
            etiqueta = f"{nombre_pack} [{celda}]"
            packs_benchmark.append(etiqueta)
            mapa_nombres[etiqueta] = nombre_pack
        
        candidatos = st.multiselect(
            "Seleccionar Candidatos:",
            options=packs_benchmark,
            placeholder="Elige 2 o más packs para comparar..."
        )
        
        if candidatos:
            if st.button("🚀 Ejecutar Comparativa", type="primary", key="btn_benchmark"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                resultados = []
                masa_base = masa_vehiculo - (peso_celda_g/1000 * n_s * n_p)  # Chasis sin pack
                
                for i, etiqueta in enumerate(candidatos):
                    status_text.text(f"Simulando {i+1}/{len(candidatos)}: {etiqueta}...")
                    
                    nombre = mapa_nombres[etiqueta]
                    pack = st.session_state.db_packs['packs'][nombre]
                    
                    # Configuración del candidato
                    modelo_celda = pack['modelo_celda']
                    ns_c, np_c = pack['ns'], pack['np']
                    soc_max_c, soc_min_c = pack['soc_max'], pack['soc_min']
                    
                    if modelo_celda not in st.session_state.db_models['models']:
                        st.warning(f"Celda '{modelo_celda}' no encontrada, omitiendo {nombre}")
                        continue
                    
                    celda = st.session_state.db_models['models'][modelo_celda]
                    cap_c = celda['cap']
                    v_nom_c = celda['v_nom']
                    v_max_c = celda['v_max']
                    peso_c = celda['peso'] / 1000.0
                    r_int_c = celda['r_int']
                    i_cont_c = celda.get('i_cont', 20.0)
                    i_pico_c = celda.get('i_pico', 40.0)
                    
                    # Recalcular dinámica con masa específica del candidato
                    peso_pack_c = peso_c * ns_c * np_c
                    masa_c = masa_base + peso_pack_c
                    
                    # Dinámica Vehicular para este candidato (masa distinta)
                    dinamica_c = calcular_dinamica_vehiculo(v_ms, acc_telem, masa_c, cd=cd_vehiculo, area_frontal=area_frontal)
                    F_roll_c = dinamica_c['F_roll']
                    F_aero_c = dinamica_c['F_aero']  # Igual para todos, pero lo sacamos del dict
                    P_frenos_total_c = dinamica_c['P_frenos_total']
                    
                    # Regeneración para este candidato
                    v_min_c = celda.get('v_min', 2.5)
                    f_ocv_c = obtener_funcion_ocv_polinomica(v_min_c, v_nom_c, v_max_c)
                    P_regen_c, P_bat_c = calcular_regeneracion(
                        P_frenos_total=P_frenos_total_c,
                        v_ms=v_ms, soc_actual=soc_max_c, v_max_celda=v_max_c,
                        r_interna_mohm=r_int_c, n_s=ns_c, n_p=np_c,
                        f_ocv_func=f_ocv_c, eta_total=eta_total
                    )
                    
                    # Parámetros para auto-ajuste
                    params_c = {
                        'masa_vehiculo': masa_c, 'F_aero': F_aero_c, 'F_roll': F_roll_c,
                        'eta_total': eta_total, 'i_descarga_cont': i_cont_c,
                        'i_descarga_pico': i_pico_c, 't_descarga_pico': 10.0,
                        'v_nom_celda': v_nom_c, 'n_s': ns_c, 'n_p': np_c,
                        'cap_celda': cap_c, 'soc_max': soc_max_c,
                        'v_max_celda': v_max_c, 'v_min_celda': celda.get('v_min', 2.5), 
                        'p_media_aux_w': p_media_aux_w, 'dt': dt_telem,
                        'mu': CV.MU_NEUMATICOS, 'rho': CV.RHO_AIRE,
                        'cl': cl_downforce, 'area': area_frontal,
                        'dist_peso': dist_peso_front,
                        'P_regen_vector': P_bat_c, 'r_interna_mohm': r_int_c,
                        'activar_limite_motor': activar_limite_motor,
                        'p_motor_max_kw': p_motor_max_kw
                    }
                    
                    # Auto-Ajuste
                    umbral_c, _, _ = buscar_umbral_optimo(soc_min_c, acc_telem, v_ms, n_vueltas, params_c)
                    
                    # Generar perfil de potencia
                    P_elec_c = generar_perfil_potencia_unificado(
                        acc_telem=acc_telem, v_ms=v_ms, dt=dt_telem, acc_umbral=umbral_c,
                        masa=masa_c, F_aero=F_aero_c, F_roll=F_roll_c, eta=eta_total,
                        I_cont=i_cont_c, I_pico=i_pico_c, t_pico_max=10.0,
                        v_nom_est=v_nom_c, n_s=ns_c, n_p=np_c,
                        p_aux=p_media_aux_w, P_regen_vector=P_bat_c,
                        mu=CV.MU_NEUMATICOS, rho=CV.RHO_AIRE, cl=cl_downforce,
                        area=area_frontal, dist_peso=dist_peso_front
                    )
                    
                    # Simulación térmica N vueltas (usa función centralizada)
                    resultado_sim_c = simular_modo_fijo(
                        P_elec_pack_vuelta=P_elec_c,
                        v_ms_vuelta=v_ms,
                        dt=dt_telem,
                        n_vueltas=n_vueltas,
                        soc_max=soc_max_c,
                        cap_celda=cap_c,
                        n_s=ns_c,
                        n_p=np_c,
                        r_interna_mohm=r_int_c,
                        temp_amb=temp_amb,
                        refrigeracion=refrigeracion,
                        peso_celda_g=peso_c,
                        f_ocv=f_ocv_c,
                        acc_umbral=umbral_c
                    )
                    
                    t_full_c = resultado_sim_c['t_full']
                    temps_c = resultado_sim_c['temps_full']
                    socs_c = resultado_sim_c['soc_full']
                    
                    # Calcular métricas usando funciones centralizadas
                    # Estimar I_cell desde el perfil de potencia
                    V_pack_nom_c = v_nom_c * ns_c
                    I_pack_c = P_elec_c / V_pack_nom_c
                    I_cell_c = I_pack_c / np_c
                    
                    # Calcular degradación usando función centralizada
                    resultado_deg_c = calcular_factores_degradacion(
                        I_cell_array=I_cell_c,
                        cap_celda=cap_c,
                        temps_full=temps_c,
                        soc_max=soc_max_c,
                        soc_min=soc_min_c
                    )
                    ciclos_c = resultado_deg_c['ciclos_vida']
                    
                    # Energía virtual
                    E_nom_c = v_nom_c * ns_c * cap_c * np_c
                    E_util_c = E_nom_c * (soc_max_c - soc_min_c) / 100.0
                    E_regen_c = np.sum(np.abs(np.minimum(P_elec_c - p_media_aux_w, 0))) * dt_telem / 3600 * n_vueltas
                    E_virtual_c = E_util_c + E_regen_c
                    
                    resultados.append({
                        'etiqueta': etiqueta, 'umbral': umbral_c,
                        'tiempo': t_full_c, 'temp': temps_c, 'soc': socs_c,
                        'soc_final': socs_c[-1], 'energia_virtual': E_virtual_c,
                        'ciclos_vida': ciclos_c, 'temp_max': np.max(temps_c),
                        'peso_pack': peso_pack_c
                    })
                    
                    progress_bar.progress((i + 1) / len(candidatos))
                
                status_text.success("✅ Comparativa finalizada!")
                
                # === VISUALIZACIONES ===
                st.markdown("---")
                st.markdown("### 🌡️ Evolución de Temperatura")
                
                fig_temp, ax_temp = plt.subplots(figsize=(12, 5))
                for res in resultados:
                    lbl = f"{res['etiqueta']} (Umb: {res['umbral']:.2f})"
                    ax_temp.plot(res['tiempo'], res['temp'], linewidth=1.5, label=lbl)
                ax_temp.axhline(60, color='red', linestyle='--', alpha=0.5, label='Límite 60°C')
                aplicar_estilo_dark(ax_temp, "Comparativa Térmica", "Tiempo", "Temperatura (°C)")
                ax_temp.xaxis.set_major_formatter(FuncFormatter(time_formatter))
                ax_temp.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
                st.pyplot(fig_temp)
                
                st.markdown("### 📍 SOH vs Energía Virtual")
                fig_soh, ax_soh = plt.subplots(figsize=(10, 5))
                colores = plt.cm.tab10(np.linspace(0, 1, len(resultados)))
                
                for res, color in zip(resultados, colores):
                    ax_soh.scatter(res['ciclos_vida'], res['energia_virtual'], s=200, c=[color], edgecolors='white', linewidths=2)
                    ax_soh.annotate(res['etiqueta'].split('[')[0].strip(), 
                                   (res['ciclos_vida'], res['energia_virtual']),
                                   textcoords="offset points", xytext=(10, 5), fontsize=9, color='white',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
                
                aplicar_estilo_dark(ax_soh, "SOH vs Energía Virtual", "Ciclos de Vida", "Energía Virtual (Wh)")
                ax_soh.set_xlim(left=0)
                ax_soh.set_ylim(bottom=0)
                ax_soh.grid(True, alpha=0.2)
                st.pyplot(fig_soh)
                
                # Tabla resumen
                st.markdown("### 🏆 Resumen de Rendimiento")
                tabla = []
                for res in resultados:
                    tabla.append({
                        "Candidato": res['etiqueta'],
                        "Umbral (m/s²)": f"{res['umbral']:.2f}",
                        "Temp Máx (°C)": f"{res['temp_max']:.1f}",
                        "SOC Final (%)": f"{res['soc_final']:.1f}",
                        "E. Virtual (Wh)": f"{res['energia_virtual']:.0f}",
                        "Ciclos Vida": f"{res['ciclos_vida']:.0f}",
                        "Peso Pack (kg)": f"{res['peso_pack']:.1f}"
                    })
                st.table(tabla)
        else:
            st.info("👆 Selecciona al menos 2 packs para comenzar el análisis comparativo.")

    # =============================================================================
    # GUARDAR RESULTADO BENCHMARK (OFFLINE)
    # =============================================================================
    # Ejecutado al final para tener acceso a todas las variables calculadas
    if guardar_benchmark_btn:
        try:
            # Recuperar variables clave de la sesión actual
            # Nombre: Pack [Celda]
            nombre_pack_actual = st.session_state.get('nombre_pack_input', 'Pack')
            # nombre_modelo debe estar definido en el bloque inicial de carga
            label_full = f"{nombre_pack_actual} [{nombre_modelo}]"
            
            # Metadatos solicitados
            # Vida estimada y Energía Virtual Real deben haber sido calculadas en el bloque de análisis
            # Si no, valor por defecto
            
            vida_str = texto_vida if 'texto_vida' in locals() else "N/A"
            energia_vr_str = f"{E_virtual_real:.2f}" if 'E_virtual_real' in locals() else "0.0"
            
            meta_data = {
                'Vida_Estimada': vida_str,
                'Energia_Virtual_Real_Wh': energia_vr_str,
                'Circuito': circuito_seleccionado,
                'Tiempo_Total_s': f"{t_full[-1]:.2f}" if 't_full' in locals() else "0"
            }
            
            # Guardar
            if 't_full' in locals() and 'temps_full' in locals():
                ok, path_res = save_benchmark_result(
                    circuito_seleccionado,
                    label_full,
                    t_full, temps_full,
                    meta_data
                )
                if ok:
                    st.sidebar.success(f"✅ Guardado: {os.path.basename(path_res)}")
                else:
                    st.sidebar.error(f"❌ Error al guardar: {path_res}")
            else:
                st.sidebar.warning("⚠️ No hay datos de simulación disponibles para guardar.")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error inesperado: {str(e)}")

else:
    st.error("No se pudieron cargar los datos del circuito. Verifica los archivos CSV.")
