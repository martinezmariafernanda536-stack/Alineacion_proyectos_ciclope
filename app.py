"""

CICLOPE v22.0 — Interfaz Principal
APC COLOMBIA | Sistema de Alineación Estratégica

CAMBIOS v22.0 (visibles en la plataforma del usuario):
  v18-1 → Panel de Palabras Nucleares: editable en sidebar, mayor peso semántico
  v18-2 → Selector de modelo (incluye BGE-M3)
  v18-3 → Tokenización corregida (palabras completas en NLPProcessor)
  v18-4 → Checkbox PMI: activa/desactiva el filtro PMI desde el sidebar
  v18-5 → PMI sin restricción FARC: aplica a cualquier tipo de proyecto
  v18-6 → Motor de búsqueda: campo para agregar keywords manuales
  v18-7 → Resultados finales directos, sin procesos intermedios innecesarios
  v18-8 → Biblioteca unificada: histórico + documental integrado
"""

import io
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

from utils.data_loader import DataLoader
from utils.nlp_processor import NLPProcessor
from utils.alignment_engine import AlignmentEngine
from utils.ai_enrichment import (alinear_proyecto_con_ia,
    construir_resultados_desde_ia, extraer_keywords_ia)
from config import (
    CONFIG, MODELOS_DISPONIBLES,
    PALABRAS_NUCLEARES_DEFAULT, PMI_ACTIVO_DEFAULT,
    W_SEMANTICO_SIN_REF, W_NUCLEARES_SIN_REF,
    W_KEYWORDS_SIN_REF,
)

st.set_page_config(
    page_title="CICLOPE · APC Colombia",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Fondo general ── */
.stApp {
    background-color: #F5F7FA;
}

/* ── Header institucional ── */
.main-header {
    font-family: 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    text-align: center;
    background: linear-gradient(135deg, #0A3161 0%, #0891B2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px;
    padding-top: 8px;
}
.sub-header {
    text-align: center;
    color: #5B6E8C;
    font-size: 0.97rem;
    font-weight: 400;
    letter-spacing: 0.5px;
    margin-bottom: 1.8rem;
}
.header-divider {
    border: none;
    border-top: 2px solid;
    border-image: linear-gradient(90deg, #0A3161, #0891B2) 1;
    margin: 0 auto 1.6rem auto;
    width: 80px;
    opacity: 0.8;
}

/* ── Score badges ── */
.score-badge {
    display: inline-block;
    padding: 4px 13px;
    border-radius: 20px;
    color: white;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.3px;
}
.score-excelente {
    background: #0A5C2E;
    box-shadow: 0 0 8px rgba(10, 92, 46, 0.45);
}
.score-bueno {
    background: #1A7340;
    box-shadow: 0 0 8px rgba(26, 115, 64, 0.40);
}
.score-moderado {
    background: #A85C00;
    box-shadow: 0 0 8px rgba(168, 92, 0, 0.40);
}
.score-bajo {
    background: #8B1A1A;
    box-shadow: 0 0 8px rgba(139, 26, 26, 0.40);
}

/* ── Signal pills ── */
.signal-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
    letter-spacing: 0.1px;
}
.signal-sem  { background: #DBEAFE; color: #1E3A8A; }
.signal-kw   { background: #DCFCE7; color: #14532D; }
.signal-nuc  { background: #EDE9FE; color: #4C1D95; }
.signal-ref  { background: #FEF9C3; color: #713F12; }

/* ── Tags de palabras clave y nucleares ── */
.kw-tag {
    display: inline-block;
    background: #EDE9FE;
    color: #4C1D95;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.79rem;
    font-weight: 500;
    margin: 2px;
}
.nuc-tag {
    display: inline-block;
    background: #DBEAFE;
    color: #1E3A8A;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.79rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Texto justificado e info de tarjeta ── */
.justify-text {
    font-size: 0.86rem;
    color: #4A5568;
    font-style: italic;
    line-height: 1.5;
}
.card-meta {
    font-size: 0.78rem;
    color: #718096;
    letter-spacing: 0.1px;
}

/* ── Badges de modo ── */
.badge-modo {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 6px;
    font-size: 0.77rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.badge-ref    { background: #DCFCE7; color: #14532D; }
.badge-noref  { background: #FEF3C7; color: #92400E; }

/* ── Sección nuclear ── */
.nuclear-section {
    background: #F5F3FF;
    border-left: 3px solid #6D28D9;
    padding: 9px 14px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
}

/* ── Expanders: evitar truncación de títulos ── */
.stExpander p, .stExpander strong {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word;
}

/* ── Separadores ── */
hr {
    border: none !important;
    border-top: 1px solid #CBD5E0 !important;
    margin: 1.2rem 0 !important;
    opacity: 0.7;
}

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #5B6E8C !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0A3161 !important;
}

/* ── Botón principal ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0A3161 0%, #0E7490 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.2px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(10, 49, 97, 0.30) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(10, 49, 97, 0.42) !important;
}

/* ── Expanders: sombra + borde izquierdo + fade-in ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2EBF5 !important;
    border-left: 4px solid #0891B2 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(8, 49, 97, 0.07);
    margin-bottom: 10px;
    animation: fadeIn 0.35s ease forwards;
}

/* ── Sidebar: acento superior cyan ── */
[data-testid="stSidebar"] {
    background-color: #EEF2F7;
    border-right: 1px solid #D1DCEA;
    border-top: 3px solid #0891B2;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #0A3161;
    font-weight: 700;
    letter-spacing: -0.2px;
    border-left: 3px solid #0891B2;
    padding-left: 8px;
}

/* ── Focus en campos de texto ── */
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
.stTextInput input:focus,
textarea:focus {
    border-color: #0891B2 !important;
    box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.18) !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── session_state ──
for key, default in [("resultados", None), ("proyecto_actual", {}),
                     ("modelo_seleccionado", CONFIG["MODELO_NLP"]),
                     ("pmi_activo", PMI_ACTIVO_DEFAULT),
                     ("palabras_nucleares", list(PALABRAS_NUCLEARES_DEFAULT)),
                     ("keywords_adicionales", []),
                     ("pesos_custom", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource(show_spinner=False)
def inicializar_sistema(modelo_nombre: str):
    """
    Inicializa el sistema con el modelo seleccionado.
    Cachea por nombre de modelo: cambiar modelo → reinicializa.
    """
    with st.spinner(f"🔄 Cargando {modelo_nombre.split('/')[-1]}... esto puede tardar 1-2 min la primera vez"):
        data_loader = DataLoader(CONFIG["RUTA_EXCEL_MARCOS"])
        catalogos   = data_loader.cargar_todos_los_marcos()
        if not catalogos:
            st.error("❌ No se pudieron cargar los marcos estratégicos.")
            st.stop()

        nlp = NLPProcessor(modelo_nombre)

        ref_db     = None
        modo_motor = "sin_referencia"

        # Cargar BD histórica si el usuario la activó y el archivo existe
        if st.session_state.get("usar_ref_db", False):
            from pathlib import Path as _Path
            for _ruta in [
                _Path("data") / "Proyectos_alienados_2025.xlsx",
                _Path(__file__).parent / "data" / "Proyectos_alienados_2025.xlsx",
            ]:
                if _ruta.exists():
                    try:
                        from utils.reference_db import ReferenceDB
                        ref_db = ReferenceDB(str(_ruta), nlp)
                        if ref_db.cargado:
                            modo_motor = "con_referencia"
                            print(f"✅ BD referencia cargada: {_ruta}")
                        else:
                            ref_db = None
                    except Exception as _e:
                        print(f"⚠️ ReferenceDB: {_e}")
                        ref_db = None
                    break

        engine = AlignmentEngine(
            nlp, catalogos,
            reference_db=ref_db,
            modo=modo_motor,
            palabras_nucleares=st.session_state.get("palabras_nucleares", PALABRAS_NUCLEARES_DEFAULT),
            activar_pmi=st.session_state.get("pmi_activo", PMI_ACTIVO_DEFAULT),
        )

    return data_loader, nlp, engine, modo_motor, ref_db


def mostrar_sidebar(catalogos, nlp, engine, modo_motor, ref_db=None):
    with st.sidebar:
        st.markdown("## 🎯 CICLOPE v22.0")
        st.caption("Alineación Estratégica · APC Colombia")

        # ── Modo activo ──
        if modo_motor == "con_referencia":
            st.markdown(
                "<span class='badge-modo badge-ref'>📚 Modo: CON referencia histórica</span>",
                unsafe_allow_html=True
            )
            # ── Opción Base de Datos Histórica (opcional) ────────────────
            st.markdown("---")
            usar_ref = st.checkbox(
                "📁 Usar BD histórica (Proyectos_alienados_2025.xlsx)",
                value=st.session_state.get("usar_ref_db", False),
                key="usar_ref_db",
                help="Coloca el archivo en data/ para activar el modo histórico",
            )
            if usar_ref:
                from pathlib import Path
                ruta = Path("data") / "Proyectos_alienados_2025.xlsx"
                if ruta.exists():
                    st.success("✅ BD histórica encontrada")
                else:
                    st.warning("⚠️ No se encontró data/Proyectos_alienados_2025.xlsx")
            if ref_db and ref_db.cargado:
                stats_ref = ref_db.obtener_estadisticas()
                st.caption(
                    f"📂 {stats_ref['proyectos']} proyectos históricos · "
                    f"{stats_ref['entradas_indice']} alineaciones indexadas"
                )
        else:
            st.markdown(
                "<span class='badge-modo badge-noref'>⚡ Modo: SIN referencia histórica</span>",
                unsafe_allow_html=True
            )
            st.caption("Coloca 'Proyectos_alienados_2025.xlsx' en data/ para modo histórico.")

        st.markdown("---")

        # ── v18-2: Selector de Modelo ──
        st.markdown("### 🤖 Modelo de Embeddings")
        nombre_actual = next(
            (k for k, v in MODELOS_DISPONIBLES.items()
             if v == st.session_state.modelo_seleccionado),
            list(MODELOS_DISPONIBLES.keys())[0]
        )
        modelo_elegido = st.selectbox(
            "Modelo semántico",
            options=list(MODELOS_DISPONIBLES.keys()),
            index=list(MODELOS_DISPONIBLES.keys()).index(nombre_actual),
            help="MPNet: mejor precisión (recomendado). LaBSE: robusto bilingüe. MiniLM: más rápido en equipos lentos."
        )
        nuevo_modelo = MODELOS_DISPONIBLES[modelo_elegido]
        if nuevo_modelo != st.session_state.modelo_seleccionado:
            st.session_state.modelo_seleccionado = nuevo_modelo
            st.cache_resource.clear()
            st.rerun()

        info_modelo = nlp.obtener_info_modelo()
        st.caption(
            f"Activo: `{info_modelo.get('modelo','').split('/')[-1]}` · "
            f"dim: {info_modelo.get('dimension_embeddings', 0)}"
        )

        st.markdown("---")

        # ── v18-4/5: Filtro PMI ──
        st.markdown("### 📋 Filtro PMI")
        pmi_activo = st.checkbox(
            "✅ Activar marco PMI",
            value=st.session_state.pmi_activo,
            help=(
                "v18: El marco PMI ahora aplica a CUALQUIER tipo de proyecto, "
                "no solo a proyectos relacionados con FARC/posconflicto. "
                "Desactiva esta opción si no es relevante para tu proyecto."
            )
        )
        if pmi_activo != st.session_state.pmi_activo:
            st.session_state.pmi_activo = pmi_activo
            engine.actualizar_pmi(pmi_activo)

        if pmi_activo:
            st.caption("PMI activo: aplica a cualquier tipo de proyecto.")
        else:
            st.caption("PMI desactivado: se excluye del análisis.")

        st.markdown("---")

        # ── v18-1: Palabras Nucleares ──
        st.markdown("### 🧠 Palabras Nucleares Estratégicas")
        st.caption(
            "Términos centrales que orientan la lógica semántica. "
            "Su presencia en los marcos aumenta el score de alineación."
        )
        nucleares_texto = st.text_area(
            "Palabras nucleares (una por línea)",
            value="\n".join(st.session_state.palabras_nucleares),
            height=150,
            help="Edita estas palabras según el enfoque estratégico del proyecto. "
                 "Se aplican como ponderación adicional en el ranking."
        )
        col_nuc1, col_nuc2 = st.columns(2)
        # ── v28: Enriquecimiento IA ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Enriquecimiento con IA")
        usar_ia = st.toggle(
            "Activar análisis IA",
            value=st.session_state.get("usar_ia", False),
            key="toggle_ia",
            help="Claude analiza el proyecto en profundidad antes del scoring. "
                 "Mejora significativamente la precisión de alineación."
        )
        st.session_state["usar_ia"] = usar_ia
        if usar_ia:
            st.success("✅ IA activada — el análisis tardará ~5s más")
            st.caption("Gemini evalúa los candidatos del motor y elimina falsos positivos.")


        if col_nuc1.button("💾 Aplicar", use_container_width=True, key="btn_nucleares"):
            nuevas = [p.strip() for p in nucleares_texto.split("\n") if p.strip()]
            st.session_state.palabras_nucleares = nuevas
            engine.actualizar_palabras_nucleares(nuevas)
            st.success(f"✅ {len(nuevas)} palabras nucleares aplicadas.")
        if col_nuc2.button("↺ Restaurar", use_container_width=True, key="btn_nuc_reset"):
            st.session_state.palabras_nucleares = list(PALABRAS_NUCLEARES_DEFAULT)
            engine.actualizar_palabras_nucleares(PALABRAS_NUCLEARES_DEFAULT)
            st.rerun()

        st.markdown("---")

        # ── Keywords Adicionales — v23: bugs de limpieza corregidos ──
        st.markdown("### 🔍 Keywords Adicionales")
        st.caption("Agrega términos manuales para enriquecer la búsqueda semántica.")

        # BUG-FIX: usamos un key derivado de un contador para poder limpiar el campo
        if "_kw_input_count" not in st.session_state:
            st.session_state["_kw_input_count"] = 0

        kw_nuevas = st.text_input(
            "Agregar keyword (Enter para confirmar)",
            placeholder="Ej: gestión del riesgo, acción humanitaria...",
            key=f"input_nueva_kw_{st.session_state._kw_input_count}"
        )

        if kw_nuevas and kw_nuevas.strip():
            nuevas_kw = [k.strip() for k in kw_nuevas.replace(";", ",").split(",") if k.strip()]
            todas_kw  = list(dict.fromkeys(st.session_state.keywords_adicionales + nuevas_kw))
            if nuevas_kw:                                           # siempre agregar y limpiar
                st.session_state.keywords_adicionales = todas_kw
                nlp.set_keywords_adicionales(todas_kw)
                # BUG-FIX: incrementar contador fuerza un widget nuevo → campo vacío
                st.session_state["_kw_input_count"] += 1
                st.rerun()

        if st.session_state.keywords_adicionales:
            st.markdown(
                " ".join(
                    f'<span class="kw-tag">{kw}</span>'
                    for kw in st.session_state.keywords_adicionales
                ),
                unsafe_allow_html=True
            )
            if st.button("🗑️ Limpiar keywords", key="btn_limpiar_kw"):
                st.session_state.keywords_adicionales = []
                nlp.set_keywords_adicionales([])
                # BUG-FIX: limpiar también el campo de texto rotando el key
                st.session_state["_kw_input_count"] += 1
                st.rerun()

        st.markdown("---")

        # ── Estado del Sistema ──
        st.markdown("### 📊 Estado del Sistema")
        total = sum(len(df) for df in catalogos.values())
        c1, c2 = st.columns(2)
        c1.metric("Marcos",    len(catalogos))
        c2.metric("Registros", total)

        c3, c4 = st.columns(2)
        c3.metric("Keywords",   info_modelo.get("total_keywords", 0))
        c4.metric("Nucleares",  info_modelo.get("palabras_nucleares", 0))

        # ── BD Histórica — siempre visible ───────────────────────────
        st.markdown("---")
        usar_ref = st.checkbox(
            "📁 Usar BD proyectos históricos",
            value=st.session_state.get("usar_ref_db", False),
            key="usar_ref_db",
            help="Activa Proyectos_alienados_2025.xlsx si está en data/",
        )
        if usar_ref:
            from pathlib import Path
            ruta = Path("data") / "Proyectos_alienados_2025.xlsx"
            if ruta.exists():
                st.success("✅ BD histórica detectada")
            else:
                st.warning("⚠️ No se encontró data/Proyectos_alienados_2025.xlsx")

        st.markdown("---")
        st.markdown("### ⚖️ Pesos del Sistema (editables)")
        st.caption("Ajusta los pesos — deben sumar 100%. Los cambios se aplican al próximo análisis.")

        # Sin base de datos histórica activa, w_referencia = 0
        # Pesos distribuidos en 3 componentes para sumar siempre 100%
        usar_ref_db = st.session_state.get("usar_ref_db", False)
        max_ref = 20 if usar_ref_db else 0

        w_sem  = st.slider("🧠 Semántica (%)",  0, 100,
                           int(round(engine.w_semantico * 100)), step=5,
                           key="slider_sem")
        w_nuc  = st.slider("💎 Nucleares (%)",  0, 100,
                           int(round(engine.w_nucleares * 100)), step=5,
                           key="slider_nuc")
        w_kw   = st.slider("🔑 Keywords (%)",   0, 100,
                           int(round(engine.w_keywords  * 100)), step=5,
                           key="slider_kw")
        if usar_ref_db:
            w_ref = st.slider("📚 Histórico (%)", 0, 20,
                              int(round(engine.w_referencia * 100)), step=1,
                              key="slider_ref")
        else:
            w_ref = 0

        total_pesos = w_sem + w_nuc + w_kw + w_ref
        if total_pesos != 100:
            diff = 100 - total_pesos
            signo = "+" if diff > 0 else ""
            st.warning(
                f"⚠️ Suma: **{total_pesos}%** — ajusta Semántica "
                f"a **{max(0, w_sem + diff)}%** ({signo}{diff}%)"
            )
        else:
            st.success(f"✅ Pesos: {total_pesos}%")
            nuevos_pesos = {
                "semantico":  w_sem  / 100,
                "nucleares":  w_nuc  / 100,
                "keywords":   w_kw   / 100,
                "referencia": w_ref  / 100,
            }
            if nuevos_pesos != st.session_state.get("pesos_custom"):
                st.session_state.pesos_custom = nuevos_pesos
                engine.actualizar_pesos(nuevos_pesos)

        st.markdown("---")
        st.markdown("### 🎨 Niveles de Confianza")
        st.markdown("""
- 🟢 **Excelente** ≥ 68%
- 🟡 **Buena** ≥ 50%
- 🟠 **Moderada** ≥ 33%
- 🔴 **Débil** < 33%
""")
        st.markdown("---")
        st.caption("v22.0 — Ajustes técnicos y funcionales aplicados")


def formulario_proyecto():
    st.markdown("### 📋 Información del Proyecto")

    c1, c2 = st.columns([3, 1])
    with c1:
        nombre = st.text_input(
            "Nombre del Proyecto *",
            placeholder="Ej: Programa de Desminado Humanitario en territorios rurales"
        )
    with c2:
        top_n = st.number_input(
            "Máx. por marco",
            min_value=2, max_value=10,
            value=2,
        )

    objetivo = st.text_area(
        "Objetivo General del Proyecto *",
        height=150,
        placeholder=(
            "Describa el objetivo con detalle: población objetivo, zona geográfica, "
            "tipo de intervención y resultados esperados. Mínimo 20 palabras.\n\n"
            "Incluya términos técnicos específicos: ej. 'desminado humanitario', "
            "'VIH/SIDA', 'economía circular', 'primera infancia', etc."
        )
    )

    descripcion = st.text_area(
        "Descripción adicional (opcional — mejora la precisión)",
        height=100,
        placeholder="Componentes, actividades, socios, indicadores, financiadores, ODS esperados..."
    )

    return nombre, objetivo, descripcion, top_n


def ejecutar_analisis(nombre, objetivo, descripcion, top_n, engine):
    if not nombre.strip() or not objetivo.strip():
        st.error("⚠️ Debe ingresar nombre y objetivo del proyecto.")
        return None
    if len(objetivo.split()) < 5:
        st.warning("⚠️ El objetivo es muy breve. Se recomienda al menos 20 palabras para mayor precisión.")

    texto_completo = f"{nombre}. {objetivo}"
    if descripcion.strip():
        texto_completo += f" {descripcion}"

    umbral = CONFIG.get("UMBRAL_CONFIANZA_MINIMO", 25) / 100
    analisis_ia = {}

    usar_ia = st.session_state.get("usar_ia", False)

    # Motor semántico siempre corre primero
    with st.spinner("🔍 Analizando alineación estratégica..."):
        resultados = engine.alinear_proyecto(texto_completo, umbral_minimo=umbral, top_n=top_n)

    # Si IA activa: Gemini evalúa catálogo completo y reemplaza los marcos que cubre
    if usar_ia:
        with st.spinner("🤖 Gemini alineando con los marcos normativos..."):
            selecciones_ia = alinear_proyecto_con_ia(
                nombre, objetivo, descripcion, engine.catalogos,
                max_por_marco=top_n,
            )
        if "error" not in selecciones_ia:
            resultados_ia = construir_resultados_desde_ia(
                selecciones_ia, engine.catalogos, max_por_marco=top_n
            )
            # Aplicar filtro PMI igual que en modo mecánico
            pmi_activo_actual = st.session_state.get("pmi_activo", True)
            if not pmi_activo_actual and "PMI" in resultados_ia:
                del resultados_ia["PMI"]
            # Reemplazar resultados del motor con los de Gemini donde aplique
            for marco, alineaciones in resultados_ia.items():
                resultados[marco] = alineaciones
            # Si PMI desactivado, eliminarlo también de los resultados del motor
            if not pmi_activo_actual and "PMI" in resultados:
                del resultados["PMI"]
            # Garantizar marcos no cubiertos por IA con al menos 1 resultado
            for marco_oblig in ["CAD", "SECTORES"]:
                if not resultados.get(marco_oblig):
                    res_f = engine.alinear_proyecto(texto_completo, umbral_minimo=0, top_n=1)
                    if res_f.get(marco_oblig):
                        resultados[marco_oblig] = res_f[marco_oblig][:1]
            st.session_state["ultimo_analisis_ia"] = selecciones_ia
            # Actualizar _meta para indicar fuente IA y estado PMI
            if "_meta" in resultados:
                resultados["_meta"]["fuente_ia"] = True
                resultados["_meta"]["pmi_activo"] = pmi_activo_actual
        else:
            st.warning(f"⚠️ IA no disponible: {selecciones_ia.get('error','?')} — usando análisis estándar")

    # Garantizar que CAD y SECTORES siempre tengan al menos 1 resultado
    if not resultados.get("_meta", {}).get("fuente_ia"):
        for marco_oblig in ["CAD", "SECTORES"]:
            if not resultados.get(marco_oblig):
                res_forzado = engine.alinear_proyecto(
                    texto_completo, umbral_minimo=0, top_n=1
                )
                top1 = res_forzado.get(marco_oblig, [])
                if top1:
                    resultados[marco_oblig] = top1[:1]

    st.session_state.proyecto_actual = {
        "nombre":      nombre,
        "objetivo":    objetivo,
        "descripcion": descripcion,
        "fecha":       datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    return resultados


def mostrar_resultados(resultados, engine):
    st.markdown("---")
    st.markdown("## 📊 Resultados de Alineación Estratégica")

    meta  = resultados.get("_meta", {})
    stats = engine.obtener_estadisticas(resultados)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alineaciones",      stats.get("total_alineaciones", 0))
    c2.metric("Score Promedio",          f"{stats.get('promedio_score', 0):.1f}%")
    c3.metric("Score Máximo",            f"{stats.get('max_score', 0):.1f}%")
    marcos_activos = sum(1 for v in stats.get("por_marco", {}).values() if v.get("cantidad", 0) > 0)
    c4.metric("Marcos con Alineaciones", marcos_activos)

    # v19: mostrar sector detectado
    sector = meta.get("sector_detectado", "")
    sim_sect = meta.get("similitud_sector", 0)
    if sector and sim_sect > 40:
        st.info(f"🏛️ Sector detectado: **{sector.replace('_', ' ').title()}** — similitud {sim_sect:.0f}%")

    # v18-4: indicar si PMI está activo
    if not meta.get("pmi_activo", True):
        st.info("ℹ️ Marco PMI desactivado en este análisis.")

    # v28: mostrar resumen del análisis IA si está disponible
    analisis_ia = st.session_state.get("ultimo_analisis_ia", {})
    if analisis_ia and "error" not in analisis_ia:
        n_alin = sum(len(v) for v in analisis_ia.values() if isinstance(v, list))
        n_marcos = len([k for k in analisis_ia if isinstance(analisis_ia[k], list)])
        st.success(f"🤖 Gemini alineó directamente — {n_alin} alineaciones en {n_marcos} marcos")

    # v18-1: mostrar palabras nucleares detectadas
    nucleares = meta.get("nucleares_detectadas", [])
    keywords  = meta.get("keywords_detectadas", [])

    col_det1, col_det2 = st.columns(2)
    with col_det1:
        if nucleares:
            with st.expander(f"💎 Palabras nucleares detectadas ({len(nucleares)})", expanded=True):
                st.markdown(
                    " ".join(f'<span class="nuc-tag">💎 {n}</span>' for n in nucleares),
                    unsafe_allow_html=True
                )
        else:
            st.info("💎 No se detectaron palabras nucleares en el objetivo. "
                    "Considera enriquecer la descripción con términos estratégicos.")

    with col_det2:
        if keywords:
            with st.expander(f"🔍 Términos técnicos detectados ({len(keywords)})", expanded=False):
                st.markdown(
                    " ".join(f'<span class="kw-tag">{kw}</span>' for kw in keywords),
                    unsafe_allow_html=True
                )

    if stats.get("por_marco"):
        activos = {k: v for k, v in stats["por_marco"].items() if v.get("cantidad", 0) > 0}
        if activos:
            fig = px.bar(
                x=list(activos.keys()),
                y=[v["max"] for v in activos.values()],
                labels={"x": "Marco", "y": "Score Máximo (%)"},
                title="Score Máximo por Marco Estratégico",
                color=[v["max"] for v in activos.values()],
                color_continuous_scale="RdYlGn", range_color=[30, 100],
            )
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    for marco, alineaciones in resultados.items():
        if marco.startswith("_"):
            continue
        n = len(alineaciones)

        # v18-4: indicar PMI desactivado
        label_extra = ""
        if marco == "PMI" and not meta.get("pmi_activo", True):
            label_extra = " *(desactivado)*"

        with st.expander(
            f"📌 **{marco}**{label_extra} — {n} alineación{'es' if n!=1 else ''}",
            expanded=(n > 0)
        ):
            if not alineaciones:
                if marco == "PMI" and not meta.get("pmi_activo", True):
                    st.caption("Marco PMI desactivado. Actívalo en el sidebar para incluirlo.")
                else:
                    st.caption("No se encontraron alineaciones sobre el umbral mínimo.")
                continue

            for i, alin in enumerate(alineaciones, 1):
                score = alin["score"]
                clase = (
                    "score-excelente" if score >= 68 else
                    "score-bueno"     if score >= 50 else
                    "score-moderado"  if score >= 33 else "score-bajo"
                )
                st.markdown(
                    f"**{i}. {alin['titulo']}**",
                    unsafe_allow_html=False
                )
                st.markdown(
                    f"<span class='score-badge {clase}'>{score:.1f}% {alin.get('nivel_confianza','')}</span>",
                    unsafe_allow_html=True
                )
                st.caption(alin["descripcion"])

                # v18-1: mostrar score nuclear + sem + kw + ref
                cs, cn, ck, cref = st.columns(4)
                cs.markdown(
                    f"<span class='signal-pill signal-sem'>🧠 Sem: {alin.get('score_semantico',0):.1f}%</span>",
                    unsafe_allow_html=True
                )
                cn.markdown(
                    f"<span class='signal-pill signal-nuc'>💎 Nuc: {alin.get('score_nuclear',0):.1f}%</span>",
                    unsafe_allow_html=True
                )
                ck.markdown(
                    f"<span class='signal-pill signal-kw'>🔑 KW: {alin.get('score_keywords',0):.1f}%</span>",
                    unsafe_allow_html=True
                )
                score_ref = alin.get("score_referencia", 0)
                score_sect = alin.get("score_sectorial", 0)
                if score_ref > 0:
                    cref.markdown(
                        f"<span class='signal-pill signal-ref'>📚 Hist: {score_ref:.1f}%</span>",
                        unsafe_allow_html=True
                    )
                elif score_sect > 0:
                    cref.markdown(
                        f"<span class='signal-pill signal-ref'>🏛️ Sect: +{score_sect:.1f}%</span>",
                        unsafe_allow_html=True
                    )

                if alin.get("justificacion"):
                    st.markdown(
                        f"<p class='justify-text'>💡 {alin['justificacion']}</p>",
                        unsafe_allow_html=True
                    )
                if alin.get("metadata"):
                    st.markdown(
                        f"<span class='card-meta'>📋 {alin['metadata']}</span>",
                        unsafe_allow_html=True
                    )
                st.markdown("---")


def _guardar_en_exports(nombre_b, ts, xlsx_bytes):
    """Guarda copia automática en exports/ (solo Excel). No interrumpe la UI si falla."""
    try:
        carpeta = Path("exports")
        carpeta.mkdir(exist_ok=True)
        (carpeta / f"alineacion_{nombre_b}_{ts}.xlsx").write_bytes(xlsx_bytes)
        return True
    except Exception:
        return False


def mostrar_exportacion(resultados, engine):
    st.markdown("---")
    st.markdown("### 💾 Exportar Resultados")

    df = engine.exportar_resultados(resultados, st.session_state.proyecto_actual)
    if df.empty:
        st.info("Sin resultados para exportar.")
        return

    ts       = st.session_state.get("_export_ts", datetime.now().strftime("%Y%m%d_%H%M%S"))
    nombre_b = st.session_state.proyecto_actual.get("nombre", "proyecto").replace(" ", "_")[:35]

    col_csv, col_xlsx = st.columns(2)

    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    with col_csv:
        st.download_button(
            "📥 Descargar CSV",
            data=csv_bytes,
            file_name=f"alineacion_{nombre_b}_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_xlsx:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Resultados", index=False)

            if "Marco" in df.columns:
                resumen = (
                    df.groupby("Marco")
                    .agg(Cantidad=("Score_%", "count"),
                         Score_Promedio=("Score_%", "mean"),
                         Score_Máximo=("Score_%", "max"))
                    .round(1)
                )
                resumen.to_excel(writer, sheet_name="Resumen_por_Marco")

            df_top = df[df["Score_%"] >= 50]
            if not df_top.empty:
                df_top.to_excel(writer, sheet_name="Alta_Confianza", index=False)

            cols_auditoria = [c for c in [
                "Marco", "Título_Meta", "Score_%",
                "Score_Semántico_%", "Score_Keywords_%",
                "Score_Nuclear_%",          # v18 nuevo
                "Score_Histórico_%",
                "Multiplicador_Precisión", "Boost_Marco", "Score_Verificación_%",
            ] if c in df.columns]
            if cols_auditoria:
                df[cols_auditoria].to_excel(writer, sheet_name="Auditoría_Matemática", index=False)

            for sn in writer.sheets:
                ws = writer.sheets[sn]
                for col in ws.columns:
                    max_w = max(
                        (len(str(cell.value)) for cell in col if cell.value), default=10
                    )
                    ws.column_dimensions[col[0].column_letter].width = min(max_w + 2, 60)

        buf.seek(0)
        xlsx_bytes = buf.read()
        buf.seek(0)
        st.download_button(
            "📊 Descargar Excel",
            data=buf,
            file_name=f"alineacion_{nombre_b}_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    if not st.session_state.get("_export_guardado", False):
        guardado = _guardar_en_exports(nombre_b, ts, xlsx_bytes)
        if guardado:
            st.session_state._export_guardado = True
            st.caption(f"Copia guardada en exports/alineacion_{nombre_b}_{ts}.xlsx")
        else:
            st.caption("No se pudo guardar la copia en exports/ (verifica permisos de escritura).")
    else:
        st.caption(f"Copia guardada en exports/alineacion_{nombre_b}_{ts}.xlsx")

    st.markdown("#### 🔍 Vista previa")
    cols_preview = [c for c in [
        "Marco", "Título_Meta", "Score_%", "Nivel_Confianza",
        "Score_Semántico_%", "Score_Keywords_%", "Score_Nuclear_%",
        "Multiplicador_Precisión", "Boost_Marco",
    ] if c in df.columns]

    st.table(df[cols_preview].round(2).reset_index(drop=True))
    st.caption(
        f"{len(df)} alineaciones · Ordenadas por Score descendente · "
        f"Excel incluye hoja 'Auditoría_Matemática' y 'Score_Nuclear_%'."
    )


def main():
    st.markdown("<h1 class='main-header'>CICLOPE</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='header-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Sistema de Alineación Estratégica &nbsp;·&nbsp; APC Colombia</p>",
        unsafe_allow_html=True,
    )

    modelo_activo = st.session_state.modelo_seleccionado

    try:
        data_loader, nlp, engine, modo_motor, ref_db = inicializar_sistema(modelo_activo)
    except Exception as e:
        st.error(f"❌ Error al inicializar el sistema: {e}")
        st.stop()

    # Sincronizar estado de sesión con engine (por si el usuario cambió algo)
    engine.actualizar_pmi(st.session_state.pmi_activo)
    engine.actualizar_palabras_nucleares(st.session_state.palabras_nucleares)
    nlp.set_keywords_adicionales(st.session_state.keywords_adicionales)

    mostrar_sidebar(engine.catalogos, nlp, engine, modo_motor, ref_db)
    nombre, objetivo, descripcion, top_n = formulario_proyecto()

    if st.button("🔍 Analizar Alineación Estratégica", type="primary", use_container_width=True):
        resultados = ejecutar_analisis(nombre, objetivo, descripcion, top_n, engine)
        if resultados:
            st.session_state.resultados = resultados
            st.session_state._export_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state._export_guardado = False

    if st.session_state.resultados:
        mostrar_resultados(st.session_state.resultados, engine)
        mostrar_exportacion(st.session_state.resultados, engine)


if __name__ == "__main__":
    main()
