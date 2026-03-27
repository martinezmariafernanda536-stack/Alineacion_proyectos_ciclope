# CICLOPE v18.0 — APC Colombia
## Sistema de Alineación Estratégica

### Cambios v18.0 (visibles en la plataforma del usuario)

| ID | Cambio |
|----|--------|
| v18-1 | **Palabras Nucleares Estratégicas** — panel editable en sidebar, mayor ponderación semántica (15%) |
| v18-2 | **Selector de modelo** — incluye BGE-M3 para comparar con MPNet |
| v18-3 | **Tokenización corregida** — solo palabras completas (límites `\b`), sin cortes en términos compuestos |
| v18-4 | **Checkbox PMI** — activa/desactiva el filtro PMI desde el sidebar |
| v18-5 | **PMI universal** — aplica a CUALQUIER tipo de proyecto (sin restricción FARC) |
| v18-6 | **Keywords manuales** — el usuario puede agregar términos de búsqueda en tiempo real |
| v18-7 | **Resultados directos** — sin procesos intermedios innecesarios |
| v18-8 | **Scoring mejorado** — alineado con palabras nucleares + objetivos + similitud semántica real |

### Pesos del sistema v18

| Señal | Sin histórico | Con histórico |
|-------|:---:|:---:|
| 🧠 Semántica | 50% | 44% |
| 🔑 Keywords | 35% | 30% |
| 💎 Nucleares | **15%** | **18%** |
| 📚 Histórico | 0% | 8% |

### Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar Streamlit
streamlit run app.py
```

### Archivos de datos
- `data/marcos_estrategicos.xlsx` — marcos estratégicos (ODS, PND, ENCI, PMI, CAD, SECTORES)
- `data/Proyectos_alienados_2025.xlsx` — histórico de proyectos (activa modo con_referencia)

### IMPORTANTE — Caché
La carpeta `cache/embeddings/` se genera automáticamente en el primer arranque.
Si el sistema se comporta de manera inesperada, eliminar la carpeta `cache/` y reiniciar.

---
*CICLOPE v18.0 — 2026-03-04*
