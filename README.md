# 🔋 Formula Student Battery Digital Twin & Adaptive Strategy Tool

### High-Fidelity Thermal Simulation & Energy Management System (EMS)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Prototype-success)

## 🎯 Project Overview
In Formula Student (and Formula E), the limiting factor is often not peak power, but the **thermal bottleneck** and **energy management**. This application serves as a **Digital Twin** for the Formula Gades High-Voltage Accumulator.

It allows the engineering team to:
1.  **Initialize** realistic cell parameters using Machine Learning (KNN).
2.  **Simulate** thermal behavior under real race profiles.
3.  **Optimize** the Energy Management Strategy using an **Adaptive PD Controller**.
4.  **Benchmark** multiple pack configurations.

---

## ⚙️ 1. System Parametrization & Hybrid Modeling
The tool combines physics-based equations with data-driven estimation for maximum fidelity.

### Vehicle & Circuit Physics
Input parameters for longitudinal dynamics (mass, drag coefficient $C_d$, frontal area) and circuit selection to generate the power demand profile.

<img width="339" height="581" alt="Captura de pantalla 2026-02-09 011940" src="https://github.com/user-attachments/assets/97067a92-1d91-4694-ac7b-ddb0a85e4a5b" />

### 🤖 AI-Powered Initialization (KNN)
Baseline parameters are not hard-coded. A **K-Nearest Neighbors (KNN)** algorithm queries a database of experimental cell data to find the closest "Digital Twin" match.
* **Input:** Capacity, Voltage, Form Factor.
* **Output:** Baseline **Internal Resistance ($R_{int}$)** and **State of Health (SOH)**.
* **Result:** A highly accurate starting point for the physics simulation.

### Electro-Chemical Library
Create your custom database. Key parameters include internal resistance ($R_{int}$), capacity, and thermal limits.
<p align="center">
  <img width="344" height="961" alt="Captura de pantalla 2026-02-09 012038" src="https://github.com/user-attachments/assets/cfcdc918-1772-400c-9b18-5f0404484b02" />
  <img width="339" height="762" alt="Captura de pantalla 2026-02-09 012049" src="https://github.com/user-attachments/assets/2ab39865-3368-4b91-8a29-d20d63387b04" />

*Left: Cell Parameter Editor | Right: Full Pack Assembler (Series/Parallel config)*

---

## 🧠 2. Adaptive Control Strategy (The "Brain")
A static acceleration threshold for electric boost is inefficient. I implemented a **Real-time Adaptive Controller** (PD Logic) that adjusts the deployment threshold based on the remaining State of Charge (SOC) and remaining laps.

* **Goal:** Deplete the battery exactly as the race finishes (SOC $\approx$ 5%), maximizing performance without "bonking" or finishing with unused energy.
* **Mechanism:** If SOC > Target, the threshold lowers (more boost). If SOC < Target, the threshold raises (conservation mode).

| Adaptive Settings | Control Response |
| :---: | :---: |
|<img width="1200" height="614" alt="Captura de pantalla 2026-02-09 012906" src="https://github.com/user-attachments/assets/78bbeeaa-6bfb-45f4-9984-1bc45734e38c" /> | <img width="1561" height="666" alt="Captura de pantalla 2026-02-09 013222" src="https://github.com/user-attachments/assets/4ef4dfe0-d8b9-4aef-b5a8-928eb6f004f9" /><img width="1125" height="444" alt="Captura de pantalla 2026-02-09 013328" src="https://github.com/user-attachments/assets/ffd5f2c4-d3d1-4fe4-a630-fe4d3622a305" />
| *Initial Threshold* | *Real-time Threshold adjustment over 12 laps* |

---

## 📊 3. Simulation Results & Telemetry
The Digital Twin generates comprehensive telemetry for the Endurance event.

### Thermal & SOC Analysis
Prediction of cell temperature rise based on Joule Heating ($I^2R$) and configurable cooling capacity ($W/K$).
<img width="918" height="899" alt="Captura de pantalla 2026-02-09 013912" src="https://github.com/user-attachments/assets/ce676bc9-3b81-4a2f-9632-215196947360" />
*Note the inverse correlation between SOC depletion and Temperature rise.*

### Electrical Performance & Dynamic Impedance
Detailed breakdown of Voltage sag, Bus Current, and Power delivery. Crucially, it models the **Dynamic Internal Resistance** evolution per lap, which increases as the pack heats up/ages.
<img width="1003" height="1088" alt="Captura de pantalla 2026-02-09 013752" src="https://github.com/user-attachments/assets/881b66b0-8811-4487-8871-caba91ad675f" />

### Regenerative Braking Efficiency
Analysis of mechanical braking vs. energy recovered via the motor (Regen).

<img width="932" height="471" alt="Captura de pantalla 2026-02-09 014006" src="https://github.com/user-attachments/assets/34758c66-1382-4d07-94ec-cefa33c415ca" />
(Remember: 3Kg battery)

---

## 📈 4. Benchmark & Comparative Analysis
The true power of this tool lies in the **Multi-Pack Comparison Module**. Instead of testing a single configuration, the engineer can simulate multiple battery architectures (e.g., changing cell chemistry, series/parallel config, or cooling mass) under identical boundary conditions to perform a fair trade-off analysis.

<img width="928" height="284" alt="Captura de pantalla 2026-02-09 014327" src="https://github.com/user-attachments/assets/b28f60ad-74fe-4fe0-8ffd-2ff68fe24276" />


### Thermal Performance Comparison
Direct visualization of thermal inertia and steady-state temperatures across different packs. This helps identify designs that might be lighter but suffer from thermal runaway risk.

<img width="908" height="499" alt="Captura de pantalla 2026-02-09 014456" src="https://github.com/user-attachments/assets/2e427af2-1768-46de-98df-d27882dbce21" />

### The "Golden Ratio": Life vs. Effective Energy
I developed a custom metric called **"Real Virtual Energy"** to measure the true utility of the pack during a race, accounting for efficiency and regeneration:
$$E_{Virtual} = E_{Nominal} + E_{Regen} - E_{ThermalLosses}$$

The scatter plot below serves as a **Pareto Front** for decision-making:
* **X-Axis:** Estimated Cycle Life (Durability).
* **Y-Axis:** Real Virtual Energy (Performance).
* **Goal:** Select the pack in the top-right quadrant (High Energy + Long Life).

<img width="909" height="544" alt="Captura de pantalla 2026-02-09 014527" src="https://github.com/user-attachments/assets/9d6b4687-12f0-412a-b113-f9a71aa1c0eb" />

---

## 🛠️ Technology Stack
* **Core Logic:** Python (NumPy, Pandas for time-series handling).
* **UI/Visualization:** Streamlit & Plotly (Interactive charts).
* **Physics Model:**
    * $P_{loss} = I^2 \cdot R_{int}(T, SOC)$
    * $Q_{cool} = h \cdot A \cdot (T_{cell} - T_{amb})$
    * Longitudinal Dynamics for power demand.

---
*Author: [Álvaro Ayala Martín] [LinkedIn](https://www.linkedin.com/in/ayalamartin/)*
