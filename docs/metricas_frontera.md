# Métricas para el Análisis de Polarización en Redes Signadas

## 1. Métricas de Posicionamiento Estructural

Estas métricas caracterizan la ubicación, inclinación e influencia de cada nodo dentro de la arquitectura global del conflicto, revelando el rol estructural que cada actor desempeña en la dinámica de polarización.

### 1.1. Sesgo de Equilibrio ($S_{eq}$)

Definición: cuantifica la inclinación o lealtad estructural de un nodo hacia uno de los dos polos, basada en el balance total de sus conexiones.

Fórmula: $S_{eq}(v) = \frac{d_{C1}(v) - d_{C2}(v)}{d_{C1}(v) + d_{C2}(v)}$

donde d_Ci(v) representa el grado total (conexiones) del nodo v hacia el clúster i.

Interpretación:
- S_eq(v) = +1: lealtad estructural total al Clúster 1
- S_eq(v) = −1: lealtad estructural total al Clúster 2  
- S_eq(v) = 0: nodo perfectamente balanceado o fronterizo
- |S_eq(v)| < 0.2: zona de alta ambigüedad estructural

### 1.2. Centralidad de Autovector del Núcleo

Definición: mide la prominencia e influencia de un actor dentro del núcleo polarizado del conflicto (excluye actores neutrales). Captura que un actor es central si está fuertemente conectado a otros actores centrales.

Contexto del vector de asignación: se parte de c ∈ {−1, 0, 1}^n que asigna cada nodo a su clúster:
- c_i = +1: nodo en Clúster 1
- c_i = −1: nodo en Clúster 2  
- c_i = 0: nodo neutral (excluido del núcleo polarizado)

Fórmula (cociente de Rayleigh): $x_1 = \arg\max_{x \neq 0} \frac{x^T A_p x}{x^T x}$

donde A_p ∈ R^{m×m} es la submatriz de adyacencia signada que contiene únicamente las interacciones entre los m = |V_p| nodos del núcleo polarizado (c_i ≠ 0).

Interpretación:
- Signo (+/−): polaridad del actor en la dinámica del conflicto
- Magnitud |x₁(v)|: nivel de influencia y cohesión dentro de su polo
- Valores cercanos a cero: actores estructuralmente ambiguos en la frontera del clivaje
- Relación con c: el signo de x₁(v) suele coincidir con c_v, su magnitud revela centralidad real

---

## 2. Métricas de Comportamiento Anómalo y Agencia de Frontera

Enfocadas en detectar y cuantificar comportamientos que desafían la división estricta de la red, identificando actores con potencial transformador o de mediación.

### 2.1. Conexiones Anómalas (cálculo auxiliar)

Definición: estimación del número de conexiones dirigidas hacia el clúster minoritario para un nodo dado. Base para métricas de frontera.

Fórmula: $\mathrm{Conexiones\_Anom}(v) = d_{total}(v) \cdot \frac{1 - |S_{eq}(v)|}{2}$

Interpretación:
- Máximo cuando S_eq(v) = 0 (nodo perfectamente fronterizo)
- Mínimo (0) cuando |S_eq(v)| = 1 (lealtad total a un polo)

### 2.2. Índice de Frontera

Definición: métrica compuesta que combina proporción y volumen absoluto de conexiones anómalas para identificar puentes significativos entre clústeres.

Fórmula: $\mathrm{Indice\_Frontera}(v) = \left( \frac{\mathrm{Conexiones\_Anom}(v)}{d_{total}(v)} \right) \cdot \ln\big(1 + \mathrm{Conexiones\_Anom}(v)\big)$

Interpretación:
- Valores altos: candidatos robustos a elementos de frontera con agencia transformadora
- Componente proporcional: normaliza por la actividad total del nodo
- Componente logarítmica: penaliza nodos con pocas conexiones anómalas absolutas

### 2.3. Grado Anómalo Externo ($d_{anom}$)

Definición: cuenta directa del número de conexiones positivas (afinidad/apoyo) que un actor mantiene con miembros del clúster opuesto.

Fórmula: $d_{anom}(v) = d_{inter}^{+}(v)$

donde d_inter⁺(v) son las aristas positivas inter-clúster del nodo v.

Interpretación:
- Medida absoluta de “puentes de afinidad” hacia el bando contrario
- Cuantifica directamente el comportamiento no polarizado
- Valor 0 indica polarización perfecta en relaciones externas

### 2.4. Proporción de Anomalía Externa ($P_{anom}$)

Definición: normalización de la métrica anterior que calcula el porcentaje de conexiones inter-clúster que son positivas.

Fórmula: $P_{anom}(v) = \frac{d_{inter}^{+}(v)}{d_{inter}(v)}$

Interpretación:
- P_anom(v) = 1: anomalía total (todas las conexiones externas son de afinidad)
- P_anom(v) = 0: polarización perfecta (todas las conexiones externas son de conflicto)  
- P_anom(v) = 0.5: máxima ambigüedad relacional externa
- P_anom(v) > 0.7: comportamiento fuertemente mediador

---

## 3. Síntesis: Tipología de Actores

Combinando estas métricas se identifican arquetipos clave:

- Actores nucleares: alto |x₁|, |S_eq| > 0.8, P_anom < 0.2  
- Actores de frontera: |S_eq| < 0.3, alto Índice de Frontera  
- Mediadores activos: d_anom > 0, P_anom > 0.5  
- Puentes estructurales: alto grado externo, S_eq moderado

Esta tipología facilita identificar actores clave para entender la reproducción y la transformación potencial de la estructura de clivaje.
