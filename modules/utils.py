"""
Módulo de Utilidades
====================
Contiene funciones de visualización, estilos y utilidades del sistema.

Funciones:
- get_data_path(): Obtiene rutas a archivos de datos
- time_formatter(): Formatea tiempo para gráficos
- aplicar_estilo_dark(): Aplica tema oscuro a Matplotlib
- mostrar_kpi_html(): Genera tarjetas KPI estilizadas
- aplicar_estilos_globales(): CSS global para Streamlit
"""

import streamlit as st
import os
import sys
import html


# =============================================================================
# FUNCIONES DE RUTAS Y SISTEMA
# =============================================================================

def get_data_path(filename, subfolder=None):
    """
    Obtiene la ruta absoluta de un archivo en la carpeta data/.
    
    Args:
        filename: Nombre del archivo
        subfolder: Subcarpeta opcional (ej: 'telemetria', 'mapas')
        
    Returns:
        Ruta absoluta completa al archivo
        
    Ejemplo:
        >>> get_data_path('modelos_celdas_db.json')
        'C:/Temp_Analisis/Bateria_Lab_Project/data/modelos_celdas_db.json'
        
        >>> get_data_path('DATOS - Germany 2012.csv', 'telemetria')
        'C:/Temp_Analisis/Bateria_Lab_Project/data/telemetria/DATOS - Germany 2012.csv'
    """
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        base = os.getcwd()
    
    if subfolder:
        return os.path.join(base, "data", subfolder, filename)
    return os.path.join(base, "data", filename)


def setup_python_path():
    """Agrega el directorio del proyecto al path de Python."""
    try:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        project_dir = os.getcwd()
    
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)


def time_formatter(x, pos):
    """
    Formateador de tiempo para ejes de Matplotlib.
    Convierte segundos a formato MM:SS.
    
    Args:
        x: Valor en segundos
        pos: Posición (requerido por FuncFormatter)
        
    Returns:
        String formateado "MM:SS"
    """
    m = int(x // 60)
    s = int(x % 60)
    return f"{m:02d}:{s:02d}"


# =============================================================================
# ESTILOS CSS
# =============================================================================

def aplicar_estilos_globales():
    """Aplica los estilos CSS globales a la aplicación Streamlit."""
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
        }
        .stApp {
            background-color: #0e1117;
        }
        h1, h2, h3 {
            color: #00bcd4 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN - GRÁFICOS
# =============================================================================

def aplicar_estilo_dark(ax, titulo="", x_label="", y_label=""):
    """
    Aplica el estilo oscuro de Streamlit a un eje de Matplotlib.
    
    Args:
        ax: Eje de Matplotlib
        titulo: Título del gráfico (opcional)
        x_label: Etiqueta eje X (opcional)
        y_label: Etiqueta eje Y (opcional)
    """
    # Fondo transparente y oscuro
    ax.set_facecolor('#0e1117')
    fig = ax.get_figure()
    fig.patch.set_alpha(0)
    
    # Textos y Ejes en Blanco
    ax.tick_params(axis='both', colors='white', labelsize=8)
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.set_title(titulo, loc='left', color='white', fontsize=10, fontweight='bold', pad=10)
    
    # Etiquetas
    if x_label: 
        ax.set_xlabel(x_label, color='white', fontsize=9)
    if y_label: 
        ax.set_ylabel(y_label, color='white', fontsize=9)
    
    # Bordes (Spines) y Rejilla
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, linestyle='--', color='white')


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN - KPIs
# =============================================================================

def mostrar_kpi_html(label, valor, unidad="", color_borde="#007acc"):
    """
    Genera y muestra una tarjeta KPI con estilo CSS personalizado.
    
    Args:
        label: Etiqueta descriptiva del KPI
        valor: Valor principal a mostrar
        unidad: Unidad de medida (opcional)
        color_borde: Color del borde izquierdo (hex)
    """
    # Escapar inputs para prevenir XSS
    label_esc = html.escape(str(label))
    valor_esc = html.escape(str(valor))
    unidad_esc = html.escape(str(unidad))
    color_borde_esc = html.escape(str(color_borde))

    html_code = f"""
    <div style="
        background-color: #1e1e1e; 
        padding: 10px; 
        border-radius: 6px; 
        border-left: 5px solid {color_borde_esc};
        margin-bottom: 10px;">
        <p style="margin:0; color: #aaa; font-size: 11px; text-transform: uppercase;">{label_esc}</p>
        <p style="margin:0; font-size: 20px; font-weight: bold; color: white;">
            {valor_esc} <span style="font-size: 12px; color: #ccc;">{unidad_esc}</span>
        </p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


# =============================================================================
# CONSTANTES DE COLORES
# =============================================================================

class Colores:
    """Paleta de colores estándar para gráficos y KPIs."""
    
    # Estados
    OK = "#28a745"
    WARNING = "#ffc107"
    ERROR = "#dc3545"
    INFO = "#00d2ff"
    
    # Gráficos
    TENSION = "#d946ef"
    CORRIENTE = "#00d2ff"
    TEMPERATURA = "#ff0055"
    SOC = "#00d2ff"
    POTENCIA_DESC = "#ff6b35"
    POTENCIA_CARGA = "#28a745"
    ACELERACION = "#a6e22e"
    FRENADA = "#f92672"
    GRIP = "#d05ce3"
    
    # Neutrales
    GRIS = "#6c757d"
    FONDO = "#0e1117"
    FONDO_CARD = "#1e1e1e"
