"""
Módulo de Gestión de Datos
==========================
Contiene funciones para cargar/guardar datos JSON y gestionar el Session State.

Funciones:
- cargar_db_modelos(): Carga la base de datos de modelos de celdas
- guardar_db_modelos(): Guarda la base de datos de modelos
- cargar_db_packs(): Carga la base de datos de packs
- guardar_db_packs(): Guarda la base de datos de packs
- inicializar_session_state(): Inicializa el estado de la sesión
"""

import streamlit as st
import json
import os
import numpy as np
import pandas as pd
from .utils import get_data_path



# =============================================================================
# RUTAS DE ARCHIVOS DE DATOS
# =============================================================================

def get_config_path(filename="last_config.json"):
    return get_data_path(filename, "db")


def save_benchmark_result(circuit_name, pack_label, t_data, temp_data, metadata):
    """
    Guarda los resultados de simulación para benchmark offline.
    
    Args:
        circuit_name: Nombre del circuito (crea subcarpeta)
        pack_label: Nombre del archivo (Pack [Celda])
        t_data: Array de tiempos
        temp_data: Array de temperaturas
        metadata: Dict con KPIs escalares (Vida, Energía, etc)
    """
    # Ruta base: data/benchmark
    # Asumimos que data_manager.py está en modules/, subir un nivel -> root -> data/benchmark
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(root_dir, 'data', 'benchmark')
    
    circuit_dir = os.path.join(base_dir, circuit_name)
    os.makedirs(circuit_dir, exist_ok=True)
    
    # Sanitize filename (windows friendly)
    safe_label = "".join([c for c in pack_label if c.isalnum() or c in (' ', '-', '_', '[', ']')]).strip()
    filename = f"{safe_label}.csv"
    csv_path = os.path.join(circuit_dir, filename)
    
    # Guardar CSV con metadatos en cabecera
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(f"# Benchmark: {pack_label} @ {circuit_name}\n")
            for k, v in metadata.items():
                f.write(f"# META:{k}={v}\n")
            f.write("Time_s,Temp_C\n")
            for t, temp in zip(t_data, temp_data):
                f.write(f"{t:.2f},{temp:.2f}\n")
        return True, csv_path
    except Exception as e:
        return False, str(e)



def _validate_benchmark_path(circuit_name, filename=None):
    """
    Valida que la ruta del archivo esté dentro del directorio de benchmarks.
    Retorna la ruta absoluta si es válida, o None si no lo es.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.abspath(os.path.join(root_dir, 'data', 'benchmark'))

    if filename:
        target_path = os.path.abspath(os.path.join(base_dir, circuit_name, filename))
    else:
        target_path = os.path.abspath(os.path.join(base_dir, circuit_name))

    # Check that target_path starts with base_dir using commonpath
    try:
        if os.path.commonpath([base_dir, target_path]) == base_dir:
            return target_path
    except ValueError:
        # Can happen if paths are on different drives
        return None

    return None


def list_benchmark_circuits():
    """Devuelve lista de carpetas de circuitos con benchmarks."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(root_dir, 'data', 'benchmark')
    if not os.path.exists(base_dir):
        return []
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def list_benchmark_files(circuit_name):
    """Devuelve lista de archivos CSV en la carpeta del circuito."""
    target_dir = _validate_benchmark_path(circuit_name)
    if not target_dir or not os.path.exists(target_dir):
        return []
    return [f for f in os.listdir(target_dir) if f.endswith('.csv')]

def load_benchmark_file(circuit_name, filename):
    """Carga CSV de benchmark y extrae datos + metadata."""
    path = _validate_benchmark_path(circuit_name, filename)
    
    if not path:
        return None, None, {'Error': 'Ruta de archivo inválida'}

    t_data = []
    temp_data = []
    metadata = {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        start_data = False
        for line in lines:
            line = line.strip()
            if not line: continue
            
            if line.startswith("# META:"):
                # Parse metadata key=value
                parts = line.replace("# META:", "").split("=", 1)
                if len(parts) == 2:
                    metadata[parts[0].strip()] = parts[1].strip()
            elif line.startswith("Time_s,Temp_C"):
                start_data = True
                continue
            elif line.startswith("#"):
                continue
            elif start_data:
                # Parse Result Data
                try:
                    vals = line.split(',')
                    t_data.append(float(vals[0]))
                    temp_data.append(float(vals[1]))
                except:
                    pass
        
        return np.array(t_data), np.array(temp_data), metadata
        
    except Exception as e:
        return None, None, {'Error': str(e)}

def delete_benchmark_file(circuit_name, filename):
    """Borra un archivo de benchmark."""
    path = _validate_benchmark_path(circuit_name, filename)
    if not path:
        return False, "Ruta de archivo inválida o intento de acceso no autorizado"

    try:
        if os.path.exists(path):
            os.remove(path)
            return True, "Borrado con éxito"
        return False, "Archivo no existe"
    except Exception as e:
        return False, str(e)


def save_last_config(pack_name):
    """Guarda el nombre del último pack seleccionado y su modelo."""
    try:
        path = get_config_path()
        data = {'last_pack': pack_name}
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error guardando config: {e}")


def load_last_config():
    """Carga el nombre del último pack seleccionado."""
    try:
        path = get_config_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get('last_pack', "-- Seleccionar --")
    except:
        pass
    return "-- Seleccionar --"

ARCHIVO_DB_CELDAS = get_data_path("modelos_celdas_db.json")
ARCHIVO_DB_PACKS = get_data_path("modelos_packs_db.json")


# =============================================================================
# FUNCIONES DE CARGA/GUARDADO - MODELOS DE CELDAS
# =============================================================================

def cargar_db_modelos():
    """
    Carga la base de datos de modelos de celdas desde JSON.
    
    IMPORTANTE: Esta función NO crea datos por defecto.
    Si el archivo no existe o está corrupto, lanza un error explícito.
    
    Returns:
        dict: Diccionario con 'last_used' y 'models'
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el archivo tiene sintaxis JSON inválida
        Exception: Cualquier otro error de lectura
    """
    if not os.path.exists(ARCHIVO_DB_CELDAS):
        st.error(f"❌ ERROR CRÍTICO: No se encuentra el archivo de modelos de celdas:\n`{ARCHIVO_DB_CELDAS}`")
        st.info("💡 Debes crear el archivo manualmente o restaurarlo desde un backup.")
        st.stop()
    
    try:
        with open(ARCHIVO_DB_CELDAS, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validar estructura mínima
        if 'models' not in data:
            st.error("❌ ERROR: El archivo de celdas no tiene la estructura correcta (falta 'models').")
            st.stop()
            
        if len(data['models']) == 0:
            st.error("❌ ERROR: El archivo de celdas está vacío. No hay modelos definidos.")
            st.stop()
            
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"❌ ERROR DE SINTAXIS JSON en `{ARCHIVO_DB_CELDAS}`:\n```\n{e}\n```")
        st.info("💡 Revisa el archivo con un validador JSON online.")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ ERROR al leer el archivo de celdas: {e}")
        st.stop()


def guardar_db_modelos(data):
    """
    Guarda la base de datos de modelos de celdas en JSON.
    
    Args:
        data: Diccionario con los modelos a guardar
    """
    try:
        with open(ARCHIVO_DB_CELDAS, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error al guardar DB celdas: {e}")


# =============================================================================
# FUNCIONES DE CARGA/GUARDADO - PACKS
# =============================================================================

def cargar_db_packs():
    """
    Carga la base de datos de configuraciones de packs desde JSON.
    
    Si el archivo no existe, crea uno vacío (es válido empezar sin packs).
    Si el archivo existe pero está corrupto, lanza error explícito.
    
    Returns:
        dict: Diccionario con 'packs'
    
    Raises:
        json.JSONDecodeError: Si el archivo tiene sintaxis JSON inválida
    """
    # Si no existe, crear archivo vacío (es válido)
    if not os.path.exists(ARCHIVO_DB_PACKS):
        data_vacio = {"packs": {}}
        try:
            with open(ARCHIVO_DB_PACKS, 'w', encoding='utf-8') as f:
                json.dump(data_vacio, f, indent=4, ensure_ascii=False)
            st.info(f"📄 Se ha creado el archivo de packs vacío: `{ARCHIVO_DB_PACKS}`")
        except Exception as e:
            st.error(f"❌ ERROR al crear archivo de packs: {e}")
            st.stop()
        return data_vacio
    
    # Archivo existe, intentar leerlo
    try:
        with open(ARCHIVO_DB_PACKS, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validar estructura mínima
        if 'packs' not in data:
            st.error("❌ ERROR: El archivo de packs no tiene la estructura correcta (falta 'packs').")
            st.stop()
            
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"❌ ERROR DE SINTAXIS JSON en `{ARCHIVO_DB_PACKS}`:\n```\n{e}\n```")
        st.info("💡 Revisa el archivo con un validador JSON online, o elimínalo para crear uno nuevo.")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ ERROR al leer el archivo de packs: {e}")
        st.stop()


def guardar_db_packs(data):
    """
    Guarda la base de datos de packs en JSON.
    
    Args:
        data: Diccionario con los packs a guardar
    """
    try:
        with open(ARCHIVO_DB_PACKS, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error al guardar DB packs: {e}")


# =============================================================================
# FUNCIONES DE CARGA DE DATOS - TELEMETRÍA Y PERFILES
# =============================================================================

def cargar_telemetria_csv(circuito_name):
    """
    Carga los datos de telemetría de un circuito desde CSV.
    
    Args:
        circuito_name: Nombre del circuito ("Germany 2012", "Germany 2010", "Austria 2012")
    
    Returns:
        tuple: (t_out, v_out, a_out) - Arrays de tiempo, velocidad (km/h) y aceleración (m/s²)
    """
    mapa_archivos = {
        "Germany 2012": "DATOS - Germany 2012.csv",
        "Germany 2010": "DATOS - Germany 2010.csv",
        "Austria 2012": "DATOS - Austria 2012.csv"
    }
    
    if circuito_name not in mapa_archivos:
        return None, None, None
    
    path = get_data_path(mapa_archivos[circuito_name], "telemetria")
    
    try:
        times, speeds, accels = [], [], []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line in lines[1:]:  # Skip header
            parts = line.split(';')
            if len(parts) >= 3:
                try:
                    v_str = parts[0].replace(',', '.').strip()
                    t_str = parts[1].replace(',', '.').strip()
                    a_str = parts[2].replace(',', '.').strip()
                    
                    v_val = float(v_str)
                    t_val = float(t_str)
                    a_val = float(a_str)
                    
                    if np.isnan(v_val) or np.isnan(t_val) or np.isnan(a_val):
                        continue
                    
                    times.append(t_val)
                    speeds.append(v_val)
                    accels.append(a_val)
                except:
                    continue
        
        # Ordenar por tiempo
        arr = sorted(zip(times, speeds, accels))
        t_out = np.array([x[0] for x in arr])
        v_out = np.array([x[1] for x in arr])
        a_out = np.array([x[2] for x in arr])
        
        return t_out, v_out, a_out
    except Exception as e:
        st.warning(f"Error cargando telemetría: {e}")
        return None, None, None


def cargar_potencia_base(filename="potencia_base.txt"):
    """
    Carga el perfil de potencia auxiliar (sistemas del vehículo).
    
    Args:
        filename: Nombre del archivo de potencia base
    
    Returns:
        tuple: (t_out, p_out, p_media_w, i_media_a)
    """
    full_path = get_data_path(filename)
    
    if not os.path.exists(full_path):
        return None, None, 0.0, 0.0
    
    try:
        # Load data using pandas
        # errors='ignore' in open() maps to encoding_errors='ignore'
        df = pd.read_csv(full_path, on_bad_lines='skip', encoding='utf-8', encoding_errors='ignore')
        
        required_cols = ['t_inicio_s', 't_fin_s', 'duracion_s', 'corriente_total_A', 'potencia_total_W']
        
        # Check if required columns are present
        if not all(col in df.columns for col in required_cols):
            return None, None, 0.0, 0.0

        # Convert to numeric and drop invalid rows (mimicking try-except float conversion)
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=required_cols, inplace=True)

        if df.empty:
            return np.array([]), np.array([]), 0.0, 0.0

        durations = df['duracion_s']
        suma_dur = durations.sum()
        
        if suma_dur > 0:
            p_media_w = (df['potencia_total_W'] * durations).sum() / suma_dur
            i_media_a = (df['corriente_total_A'] * durations).sum() / suma_dur
        else:
            p_media_w = 0.0
            i_media_a = 0.0

        # Generate t_out and p_out
        # Vectorized steps calculation
        steps = np.maximum(1, ((df['t_fin_s'] - df['t_inicio_s']) * 10).astype(int))
        
        # Repeat power values
        p_out = np.repeat(df['potencia_total_W'].values, steps)

        # Generate time values
        t_list = [np.linspace(start, end, n, endpoint=False)
                  for start, end, n in zip(df['t_inicio_s'], df['t_fin_s'], steps)]

        if t_list:
            t_out = np.concatenate(t_list)
        else:
            t_out = np.array([])

        return t_out, p_out, p_media_w, i_media_a
    except Exception:
        return None, None, 0.0, 0.0


# =============================================================================
# GESTIÓN DEL SESSION STATE
# =============================================================================

def inicializar_session_state():
    """
    Inicializa todas las variables del Session State de forma robusta.
    Debe llamarse al inicio de la aplicación.
    """
    # Cargar bases de datos
    if 'db_models' not in st.session_state:
        st.session_state.db_models = cargar_db_modelos()
    
    if 'db_packs' not in st.session_state:
        st.session_state.db_packs = cargar_db_packs()
        
    last_pack_saved = load_last_config()
    
    # Defaults de Modelo de Celda
    first_model_name = "Generic_Cell"
    if 'models' in st.session_state.db_models and len(st.session_state.db_models['models']) > 0:
        first_model_name = list(st.session_state.db_models['models'].keys())[0]
        
    # Si tenemos un pack guardado, intentamos usar su modelo
    default_model = first_model_name
    if last_pack_saved != "-- Seleccionar --" and last_pack_saved in st.session_state.db_packs['packs']:
             pack_data = st.session_state.db_packs['packs'][last_pack_saved]
             if pack_data['modelo_celda'] in st.session_state.db_models['models']:
                 default_model = pack_data['modelo_celda']
    
    last_model_name = st.session_state.get('selector_modelo', default_model)
    
    # Verificar modelo a usar para defaults
    if last_model_name not in st.session_state.db_models['models']:
        if st.session_state.db_models['models']:
            last_model_name = list(st.session_state.db_models['models'].keys())[0]
        else:
            last_model_name = 'Celda Generica'
            st.session_state.db_models['models'] = {
                'Celda Generica': {
                    'cap': 5.0, 'v_nom': 3.7, 'v_max': 4.2, 'v_min': 2.5,
                    'peso': 70.0, 'r_int': 10.0,
                    'i_cont': 20.0, 'i_pico': 40.0, 't_pico': 10.0
                }
            }
    
    data_init = st.session_state.db_models['models'][last_model_name]
    
    # Defaults para todas las variables
    defaults = {
        # Variables de Formulario (Celda Activa)
        'form_nombre': last_model_name,
        'form_cap': data_init['cap'],
        'form_vnom': data_init['v_nom'],
        'form_vmax': data_init['v_max'],
        'form_vmin': data_init.get('v_min', 2.5),
        'form_peso': data_init['peso'],
        'form_rint': data_init['r_int'],
        'form_icont': data_init.get('i_cont', 20.0),
        'form_ipico': data_init.get('i_pico', 40.0),
        'form_tpico': data_init.get('t_pico', 10.0),
        
        # Variables de Pack
        'pack_ns': 12,
        'pack_np': 2,
        'pack_soc_max': 100,  # Default: rango completo
        'pack_soc_min': 0,    # Default: rango completo
        
        # Variables de Lógica
        'auto_ajuste_pendiente': False,
        'snapshot_inputs': {},
        'slider_acc': 4.0,
        'slider_acc': 4.0,
        'nombre_pack_input': "Mi Pack Personalizado",
        'selector_pack': last_pack_saved,
        
        # Variables de Motor
        'activar_limite_motor': True,
        'p_motor_max_kw': 10.0
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

