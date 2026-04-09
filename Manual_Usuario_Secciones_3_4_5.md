# 3. Descripción de la Interfaz de Usuario (UI)

El Simulador de la Batería de Tracción cuenta con una interfaz gráfica intuitiva dividida principalemente en dos grandes bloques de interacción: el panel de control lateral y el visor de resultados central.

### 3.1. Panel Lateral de Configuración (Inputs)
Ubicado en la parte izquierda de la pantalla, este panel agrupa todas las variables de entrada necesarias para configurar el estado inicial de la batería y el entorno físico de la simulación. Está diseñado mediante menús desplegables y controles deslizantes (*sliders*) para evitar la introducción de valores fuera de los rangos físicos permitidos.

*   **Bloque "Macro Configuración":** Permite cargar configuraciones predefinidas o guardar el set actual de parámetros para agilizar ensayos repetitivos sin tener que introducir los datos uno a uno.
*   **Gestor de Archivos:** Sección habilitada para cargar el perfil temporal de intensidad (el ciclo de carga/descarga o conducción) en formato `.csv` o `.txt`.

### 3.2. Visor Central y Pestañas de Navegación
La zona principal de la pantalla está dedicada a la visualización de datos tras el cálculo. Una vez ejecutada la simulación, el usuario interactúa con los resultados a través de un sistema de pestañas temáticas ubicadas en la zona superior de la gráfica:

*   **Pestaña ⚡ Eléctrico:** Dedicada exclusivamente al análisis del comportamiento eléctrico puro (Tensiones, corrientes y la evolución de la resistencia interna de la celda respecto al pack).
*   **Pestaña 🌡️ Térmico & SOC:** Focalizada en el seguimiento del estado de carga (%) y la evolución de la temperatura de la batería en función de las pérdidas por efecto Joule y su capacidad de disipación.

---

# 4. Configuración de la Simulación (Inputs de Entrada)

Para garantizar la precisión de los modelos físicos (basados en Arrhenius y estimaciones dinámicas de la resistencia), el usuario debe cumplimentar rigurosamente los siguientes apartados antes de ejecutar cualquier ensayo.

### 4.1. Parámetros de la Celda Unitaria
*   **Química y Formato:** Obligatorio para definir las curvas características (OCV). El modelo toma por defecto el formato geométrico *Pouch* (bolsa), lo que activa internamente el factor geométrico de disipación térmica ($f_{pouch} = 1.5$).
*   **Capacidad Nominal ($C_n$):** Capacidad de la celda individual expresada en Amperios-hora (Ah).
*   **Resistencia Interna Base ($R_{0}$):** Valor nominal de resistencia a 25°C y SOC 50%. Este valor será recalculado dinámicamente por el motor de física de la simulación en función del SOC y la Temperatura.

### 4.2. Arquitectura del Pack de Baterías
La matriz geométrica define la potencia y capacidad total del sistema de tracción.
*   **Celdas en Serie (Ns):** Determina el voltaje nominal y máximo del pack de tracción.
*   **Ramas en Paralelo (Np):** Multiplica la corriente máxima admisible y la capacidad total ($Ah_{pack}$).

### 4.3. Condiciones Frontera (Térmicas y Estado Inicial)
Estos valores definen el punto de partida temporal ($t=0$) de la simulación:
*   **SOC Inicial (%):** Nivel de carga remanente en el instante cero (0% = Vacía; 100% = Llena).
*   **Temperatura Inicial ($T_{bat}$):** Temperatura interna de las celdas al arrancar el ensayo, expresada en grados Celsius (°C).
*   **Temperatura Ambiente ($T_{env}$):** Temperatura del entorno exterior. Marca la asíntota térmica hacia la que el pack tenderá a enfriarse (o calentarse) si no hay corriente.
*   **Coeficiente Global de Transferencia de Calor ($h \cdot A$):** Capacidad del sistema de refrigeración para extraer calor del pack hacia el exterior. Afecta directamente a la velocidad con la que la temperatura del pack diverge de la temperatura ambiente.

---

# 5. Ejecución e Interpretación de Resultados

Una vez configurados los parámetros de los pasos anteriores y cargado el perfil de ciclo de conducción o test (vector de Intensidad vs Tiempo), el usuario hará clic en el botón de **"Ejecutar Simulación"**.

Dependiendo de la resolución temporal del ciclo y los cálculos de resistencia cruzada acoplada a la temperatura, el proceso durará unos instantes. Al finalizar, la vista renderizará las gráficas dinámicas.

### 5.1. Análisis de la Pestaña "⚡ Eléctrico"
Esta visualización devuelve tres sub-gráficas fundamentales correspondientes al bloque eléctrico:

1.  **Perfil de Corriente Demandada ($I$):** Verificación de que el simulador ha leído correctamente el archivo de entrada. Por convención del software, valores positivos denotan **descarga** (tracción), mientras que valores negativos denotan **carga** (freno regenerativo o conexión a cargador).
2.  **Variación de Voltaje ($V_{pack}$):** Curva de tensión resultante tras sumar la Tensión a Circuito Abierto (OCV) y la caída óhmica dinámica. El operador debe vigilar que la tensión no cruce los límites de *Cut-off Voltage* o sobrecarga de diseño de las celdas escogidas durante picos extremos de demanda.
3.  **Análisis de la Resistencia Interna ($R$):** Gráfica vital que muestra la evolución cruzada de la resistencia.
    *   **Línea continua:** Muestra el comportamiento resistivo del pack global.
    *   **Línea punteada:** Muestra el comportamiento resistivo referenciado unívocamente a una de sus celdas asiladas.
    *   *Nota analítica:* La resistencia cambiará dinámicamente. Verás que la resistencia del sistema cae a medida que aumenta la temperatura interna de las celdas, reflejando el modelo de aumento de la conductividad de los electrolitos (Ley de Arrhenius).

### 5.2. Análisis de la Pestaña "🌡️ Térmico & SOC"
Esta vista agrupa los dos factores resultantes más importantes para monitorizar el estrato de seguridad y rendimiento.

1.  **Estado de Carga (SOC):** Evolución temporal del porcentaje de carga, calculado internamente mediante recuento de Coulomb (Coulomb Counting). Demuestra la pérdida neta de energía durante el perfil temporal ensayado.
2.  **Evolución Térmica ($T_{pack}$):** Representa el aumento progresivo de la temperatura del banco de baterías producto de la radiación de calor interno por Efecto Joule ($I^2 \cdot R_{dinámica}$).
    *   Si el algoritmo observa que el gradiente térmico de la celda incrementa más rápido que lo que el sistema externo disipa ($h \cdot A$), veremos una rampa continua de temperatura.
    *   *Nota operativa:* El diseñador debe utilizar esta curva para certificar que, en régimen exigente, el pack no alcanza su zona de límite térmico (*Thermal Runaway*) para el ciclo de uso estipulado.
