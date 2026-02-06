import pandas as pd
import numpy as np
import os
import sys

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_EXCEL = os.path.join(DIRECTORIO_ACTUAL, "Pouch cell_summary.xlsx")
RUTA_DATOS_RAW = os.path.join(DIRECTORIO_ACTUAL, "Battery raw data")

print(f"--- INICIANDO ANÁLISIS V4.0 (ARQUITECTURA DE BATCHES) ---")

# ==========================================
# 2. CARGA DEL EXCEL (C-RATES POR BATCH)
# ==========================================
# Estructura DB: { 'P462': {4: 6.0, 5: 6.0 ...}, 'P531': {...} }
DB_BATCHES = {}

if os.path.exists(RUTA_EXCEL):
    try:
        print(f"📂 Cargando Excel de metadatos...")
        xls = pd.ExcelFile(RUTA_EXCEL)
        
        for sheet in xls.sheet_names:
            # Identificador del batch (ej. P462)
            batch_id = sheet.split(' ')[0].strip() # "P462 (high load)" -> "P462"
            
            # Buscar cabecera dinámicamente
            df_preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
            header_idx = -1
            for i, row in df_preview.iterrows():
                txt = row.astype(str).str.lower().str.strip().tolist()
                if any('cell' in x for x in txt) and any(('rate' in x or 'char' in x) for x in txt):
                    header_idx = i
                    break
            
            if header_idx != -1:
                df = pd.read_excel(xls, sheet_name=sheet, header=header_idx)
                df.columns = [str(c).lower().strip() for c in df.columns]
                
                c_id = next((c for c in df.columns if 'cell' in c or 'no' in c), None)
                c_rate = next((c for c in df.columns if 'rate' in c or 'char' in c), None)
                
                if c_id and c_rate:
                    if batch_id not in DB_BATCHES: DB_BATCHES[batch_id] = {}
                    
                    for _, row in df.iterrows():
                        try:
                            cid = int(row[c_id])
                            val = str(row[c_rate]).upper().replace('C','').strip()
                            DB_BATCHES[batch_id][cid] = float(val)
                        except: continue
        
        print(f"📊 Metadatos cargados: {len(DB_BATCHES)} batches identificados.")
        
    except Exception as e:
        print(f"⚠️ Alerta Excel: {e}")
else:
    print("⚠️ No se encontró el Excel. Se usará 4.0C por defecto.")

def obtener_c_rate(batch_name, cell_num):
    # Intentar coincidencia exacta o parcial del batch
    # batch_name viene de la carpeta (ej "P462_NMC532...")
    for key in DB_BATCHES:
        if key in batch_name: # Si "P462" está en "P462_NMC..."
            return DB_BATCHES[key].get(cell_num, 4.0)
    return 4.0

# ==========================================
# 3. LECTURA ULTRA-ROBUSTA (CONTENIDO)
# ==========================================

def leer_csv_ninja(ruta, nombre_dato, opcional=False):
    if not os.path.exists(ruta):
        if opcional: return None
        raise FileNotFoundError(f"Falta: {os.path.basename(ruta)}")

    try:
        df = pd.read_csv(ruta)
    except:
        # Fallback: intentar leer sin cabecera si falla
        try:
            df = pd.read_csv(ruta, header=None)
        except:
            raise ValueError("Archivo corrupto")

    # --- DETECCIÓN DE COLUMNA CYCLE POR CONTENIDO ---
    col_cycle = None
    
    # 1. Por nombre
    cols_lower = {c: str(c).lower().strip() for c in df.columns}
    for orig, limpia in cols_lower.items():
        if limpia in ['cycle', 'cyc', 'index']:
            col_cycle = orig
            break
            
    # 2. Por contenido (Si parece un contador 1, 2, 3...)
    if not col_cycle:
        for c in df.columns:
            # Debe ser numérico
            if pd.api.types.is_numeric_dtype(df[c]):
                # Debe ser positivo y creciente
                series = df[c].dropna()
                if len(series) > 5 and series.iloc[0] < series.iloc[-1]:
                    # Heurística simple: si empieza cerca de 0/1 y crece
                    if 0 <= series.iloc[0] <= 100: 
                        col_cycle = c
                        break
    
    if not col_cycle:
        raise ValueError(f"Imposible detectar columna Cycle en {os.path.basename(ruta)}")

    df.rename(columns={col_cycle: 'Cycle'}, inplace=True)

    # --- DETECCIÓN DE COLUMNA DE DATOS ---
    col_dato = None
    
    # 1. Por nombre (si existe)
    if nombre_dato in df.columns:
        col_dato = nombre_dato
    
    # 2. Por eliminación (La columna numérica que NO es Cycle ni Unnamed)
    if not col_dato:
        candidates = []
        for c in df.columns:
            if c == 'Cycle': continue
            if 'unnamed' in str(c).lower(): continue
            if pd.api.types.is_numeric_dtype(df[c]):
                candidates.append(c)
        
        if len(candidates) > 0:
            # Tomamos la última columna candidata (suele ser el dato en formato Capacity)
            col_dato = candidates[-1]

    if not col_dato:
        raise ValueError(f"Imposible detectar dato '{nombre_dato}'")
        
    df.rename(columns={col_dato: nombre_dato}, inplace=True)
    
    # Limpieza final: quitar duplicados de ciclo si los hubiera
    df = df.drop_duplicates(subset=['Cycle'])
    
    return df[['Cycle', nombre_dato]]

# ==========================================
# 4. ESCANEO Y EJECUCIÓN
# ==========================================
datos = []

print("="*60)
print(f"🔎 ESCANEANDO TODOS LOS ARCHIVOS EN: {RUTA_DATOS_RAW}")
print("="*60)

# 1. Encontrar todos los Capacity_*.csv
archivos_encontrados = []
for root, dirs, files in os.walk(RUTA_DATOS_RAW):
    for f in files:
        if f.startswith("Capacity_") and f.endswith(".csv"):
            # Extraer info
            # f = Capacity_Cell04.csv -> cell_str = Cell04
            cell_str = f.replace("Capacity_", "").replace(".csv", "")
            try:
                cell_num = int(cell_str.replace("Cell", "").replace("cell", ""))
            except: continue
            
            folder_name = os.path.basename(root)
            
            archivos_encontrados.append({
                'cell_num': cell_num,
                'cell_str': cell_str,
                'folder': folder_name,
                'path_cap': os.path.join(root, f),
                'root': root
            })

print(f"📦 Archivos encontrados: {len(archivos_encontrados)}")

# 2. Procesar cada archivo
for item in archivos_encontrados:
    unique_id = f"{item['folder']}_{item['cell_str']}"
    print(f"Procesando {unique_id}...", end=" ")
    
    try:
        # Rutas hermanas
        path_ce = os.path.join(item['root'], f"CE_{item['cell_str']}.csv")
        path_eod = os.path.join(item['root'], f"EOD_{item['cell_str']}.csv")
        
        # C-Rate (Batch aware)
        c_rate = obtener_c_rate(item['folder'], item['cell_num'])
        
        # Leer
        df_cap = leer_csv_ninja(item['path_cap'], 'Capacity')
        df_ce = leer_csv_ninja(path_ce, 'Coulombic_Eff', True)
        df_eod = leer_csv_ninja(path_eod, 'EODV', True)
        
        if df_eod is None: raise ValueError("Falta EOD")
        
        # Merge
        df = df_cap
        if df_ce is not None: df = pd.merge(df, df_ce, on='Cycle')
        if df_eod is not None: df = pd.merge(df, df_eod, on='Cycle')
        
        if len(df) < 5: raise ValueError("Pocos datos")
        
        # Métricas
        slope_cap = np.polyfit(df['Cycle'], df['Capacity'], 1)[0]
        slope_eod = np.polyfit(df['Cycle'], df['EODV'], 1)[0]
        ce_avg = df['Coulombic_Eff'].mean() if 'Coulombic_Eff' in df.columns else 0
        
        datos.append({
            'Cell_ID': unique_id, # ID Único (Batch + Cell)
            'Batch': item['folder'],
            'Cell_Num': item['cell_num'],
            'C_rate': c_rate,
            'Cap_Max': df['Capacity'].max(),
            'Slope_Capacity': slope_cap,
            'Slope_EODV': slope_eod,
            'Avg_CE': ce_avg,
            'Cycles_Total': len(df)
        })
        print(f"✅ OK ({c_rate}C)")
        
    except Exception as e:
        print(f"❌ {e}")

# Guardar
if datos:
    df_fin = pd.DataFrame(datos)
    out = os.path.join(DIRECTORIO_ACTUAL, "Resultado_Analisis_Bateria_Completo.csv")
    df_fin.to_csv(out, index=False)
    print(f"\n✨ HECHO. {len(df_fin)} perfiles generados.")
    print(f"📂 Archivo: {out}")
else:
    print("\n💀 ERROR: No se generaron datos.")