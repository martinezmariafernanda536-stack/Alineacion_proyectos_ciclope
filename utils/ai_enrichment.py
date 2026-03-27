"""
CICLOPE v23.1 — ai_enrichment.py
APC COLOMBIA | Módulo de Enriquecimiento con IA

CAMBIOS v23.1 sobre v23:
  - top_n respetado en modo IA: construir_resultados_ia corta a max_por_marco
  - alinear_con_ia / alinear_proyecto_con_ia reciben top_n y lo pasan al prompt
  - Prompt reforzado con "EXACTAMENTE {top_n} entradas como MÁXIMO por marco"
  - Contexto jerárquico en catálogo: [Eje/Cat] para PND, [ODS N] para ODS, [Obj] para ENCI
  - texto_completo preferido sobre titulo+descripcion en filtro TF-IDF
  - Pesos alineados con alignment_engine (SEM=0.80, NUC=0.10, KW=0.10)
  - _boost_marco inyectado desde BOOST_MARCOS del config
"""

import json
import re
import os
import requests
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────── CONFIG ─────────────────

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)

TIMEOUT = 50

# Pesos alineados con W_SEMANTICO_SIN_REF / W_NUCLEARES_SIN_REF / W_KEYWORDS_SIN_REF
W_SEM = 0.80
W_NUC = 0.10
W_KW  = 0.10

# Boost por marco — importado de config.py para evitar desincronización
from config import BOOST_MARCOS as _BOOST_MARCOS_DEFAULT

# ───────────────── PROMPT PRINCIPAL ─────────────────

PROMPT_TEMPLATE = """\
Eres un experto en cooperación internacional y marcos normativos de Colombia.

PROYECTO
Nombre: {nombre}
Objetivo: {objetivo}
Descripción: {descripcion}

CATÁLOGO (formato: indice | id [contexto jerárquico] | titulo — descripcion_corta)
{catalogo}

═══════════════════════════════════════════════════════
INSTRUCCIONES PRECISAS

Paso 1 — Identifica los 5 temas principales del proyecto. Ten en cuenta que
este sistema es usado por APC Colombia para alinear proyectos de cooperación
internacional. Los tipos válidos de proyectos incluyen: desminado, VIH/salud,
género/VBG, medio ambiente/clima, agricultura/seguridad alimentaria,
gobernanza/transparencia, paz/posconflicto/reincorporación, migración/
venezolanos, niñez/primera infancia, agua/saneamiento, educación,
ciencia/tecnología/innovación, trabajo/empleo, comercio, cultura, y
cooperación sur-sur/triangular/fortalecimiento institucional.

Paso 2 — Para cada marco (ODS, PND, ENCI, PMI, CAD, SECTORES):
  a) Revisa TODAS las entradas de ese marco en el catálogo.
  b) Selecciona MÁXIMO {max_por_marco} entradas — las MÁS alineadas.
  c) Asigna 3 sub-scores independientes (cada uno 0-100):
     - score_semantico: similitud temática general entre proyecto y meta
     - score_nuclear:   coincidencia de conceptos clave/términos nucleares
     - score_keywords:  presencia de términos técnicos del dominio sectorial

LÍMITE ESTRICTO: devuelve MÁXIMO {max_por_marco} entradas por marco. No más.

REGLAS CRÍTICAS POR MARCO:
- ODS: La Meta específica (ej. 6.1, 16.3) debe ser coherente con el proyecto,
       no solo el ODS padre. Penaliza con score_nuclear < 50 si la meta es genérica.
       Proyectos de cooperación internacional → considera ODS 17.x (alianzas).
       Proyectos de gobernanza → ODS 16.6/16.7. Agricultura → ODS 2.
       Trabajo/empleo → ODS 8. Ciencia/tecnología → ODS 9.
- PND: Usa el [Eje] y [Cat] del catálogo para verificar coherencia. Proyecto de paz
       → eje Paz Total. Proyecto ambiental → eje Ordenamiento del Agua. score_nuclear
       alto (> 65) solo si el catalizador usa exactamente los mismos conceptos.
- ENCI: La Estrategia Nacional de Cooperación Internacional tiene 4 objetivos:
        (1) Fortalecer la cooperación recibida para el desarrollo sostenible,
        (2) Proyectar a Colombia como oferente de cooperación sur-sur y triangular,
        (3) Focalizar la cooperación en el desarrollo territorial y la paz,
        (4) Cooperación para el medio ambiente y cambio climático.
        score_nuclear alto (> 65) si el proyecto es explícitamente de cooperación
        internacional, sur-sur, triangular o fortalecimiento institucional.
- CAD: Asigna categorías OCDE/CAD según el sector temático del proyecto.
       Gobernanza/instituciones → "gobierno y sociedad civil".
       Cooperación técnica → "cooperacion para el desarrollo".
       Agricultura → "agricultura, silvicultura, pesca".
       Educación → "educacion". Salud → "salud y poblacion".
       Medio ambiente → "medio ambiente". Género → "igualdad de genero".
       RESTRICCIÓN ESTRICTA — entradas CAD de VIH/SIDA ("mitigacion social del vih",
       "lucha contra ets", "lucha contra enfermedades de transmision"): SOLO incluir
       si el proyecto menciona explícitamente al menos una de estas palabras:
       VIH, SIDA, HIV, AIDS, antirretroviral, epidemia, tuberculosis.
       Si ninguna de estas palabras aparece en el proyecto → excluir esas entradas CAD.
       Un proyecto de género/VBG, trata, desminado o gobernanza NO califica.
- SECTORES: Identifica el ministerio/entidad colombiana más relevante.
            Cooperación → APC Colombia. Medio ambiente → Ministerio de Ambiente.
            Salud → Ministerio de Salud. Agricultura → Ministerio de Agricultura.
            Paz/reincorporación → Presidencia / Ministerio de Defensa.
            Migración → Cancillería. Niñez → Presidencia (ICBF).
            Gobernanza → Función Pública / DNP.
- PMI: Solo si el proyecto tiene vínculo directo con el Acuerdo de Paz. Si no → [].
- Si un marco NO tiene alineación real (score_semantico < 40) → devuelve lista vacía [].
- Descarta entradas donde score_semantico < 40.

═══════════════════════════════════════════════════════
FORMATO — SOLO JSON VÁLIDO, SIN TEXTO ADICIONAL:

{{
  "ODS": [
    {{
      "indice": 12,
      "id": "ODS_12",
      "score_semantico": 85,
      "score_nuclear": 70,
      "score_keywords": 65,
      "razon": "La Meta 6.1 corresponde directamente con el acceso a agua potable"
    }}
  ],
  "PND": [...],
  "ENCI": [],
  "PMI": [],
  "CAD": [...],
  "SECTORES": [...]
}}
"""

# ───────────────── API KEY ─────────────────

def cargar_api_keys() -> list:
    """
    Carga todas las API keys de Gemini disponibles en orden de prioridad.
    Busca: GEMINI_API_KEY → GEMINI_API_KEY_2 → GEMINI_API_KEY_3
    en st.secrets y variables de entorno.
    Retorna lista de keys válidas (len > 20).
    """
    nombres = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
    secrets = {}
    try:
        import streamlit as st
        for nombre in nombres:
            val = st.secrets.get(nombre, "").strip()
            if len(val) > 20:
                secrets[nombre] = val
    except Exception:
        pass

    keys = []
    for nombre in nombres:
        if nombre in secrets:
            keys.append(secrets[nombre])
        else:
            val = os.environ.get(nombre, "").strip()
            if len(val) > 20:
                keys.append(val)
    return keys


def cargar_api_key() -> str:
    """Compatibilidad: retorna la primera key disponible."""
    keys = cargar_api_keys()
    return keys[0] if keys else ""


# ───────────────── NORMALIZACIÓN ─────────────────

def _normalizar(texto):
    if not texto:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


# ───────────────── FILTRO SEMÁNTICO PRE-IA ─────────────────

def filtrar_catalogo(nombre, objetivo, descripcion, catalogos, top_n=15):
    """
    Pre-filtra con TF-IDF para reducir cada marco a top_n candidatos.
    Usa texto_completo si está disponible (incluye keywords de dominio enriquecidos).
    """
    texto_proyecto = f"{nombre} {objetivo} {descripcion}".lower()
    catalogos_filtrados = {}

    for marco, df in catalogos.items():
        if df is None or df.empty:
            catalogos_filtrados[marco] = df
            continue

        # Preferir texto_completo enriquecido (ejes PND, catalizadores, objetivos ENCI)
        if "texto_completo" in df.columns:
            textos = df["texto_completo"].fillna("").str.lower().tolist()
        else:
            textos = (
                df["titulo"].fillna("") + " " + df["descripcion"].fillna("")
            ).str.lower().tolist()

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        try:
            matriz      = vectorizer.fit_transform([texto_proyecto] + textos)
            similitudes = cosine_similarity(matriz[0:1], matriz[1:]).flatten()
        except Exception:
            catalogos_filtrados[marco] = df
            continue

        df_temp = df.copy()
        df_temp["_score_tfidf"] = similitudes
        df_temp = df_temp.sort_values("_score_tfidf", ascending=False).head(top_n)
        catalogos_filtrados[marco] = df_temp

    return catalogos_filtrados


# ───────────────── CATÁLOGO CON CONTEXTO JERÁRQUICO ─────────────────

def construir_catalogo(catalogos):
    """
    Construye el catálogo para el prompt con contexto jerárquico por marco:
      PND  → [Eje: ...] [Cat: ...]
      ODS  → [ODS N]
      ENCI → [Obj.N: ...]
      PMI  → [Pilar: ...]
    Ayuda a Gemini a razonar sobre coherencia de eje/catalizador.
    """
    lineas = []
    for marco in ["ODS", "PND", "ENCI", "PMI", "CAD", "SECTORES"]:
        df = catalogos.get(marco)
        if df is None or df.empty:
            continue
        lineas.append(f"\n[{marco}]")
        for idx, row in df.iterrows():
            id_orig = row.get("id_original", idx)
            titulo  = str(row.get("titulo",      "")).strip()
            desc    = str(row.get("descripcion", "")).strip()[:90]

            # Contexto jerárquico según marco
            contexto = ""
            if marco == "PND":
                eje = str(row.get("eje",          "")).strip()
                cat = str(row.get("catalizador",  "")).strip()
                if eje and eje.lower() not in ("nan", ""):
                    contexto = f" [Eje: {eje[:50]}]"
                if cat and cat.lower() not in ("nan", ""):
                    contexto += f" [Cat: {cat[:40]}]"
            elif marco == "ODS":
                ods_n = str(row.get("ods", row.get("ods_nombre", ""))).strip()
                if ods_n and ods_n.lower() not in ("nan", ""):
                    contexto = f" [ODS {ods_n[:30]}]"
            elif marco == "ENCI":
                obj   = str(row.get("objetivo",    "")).strip()
                eje_n = str(row.get("eje_numero",  "")).strip()
                if obj and obj.lower() not in ("nan", ""):
                    contexto = f" [Obj.{eje_n}: {obj[:40]}]"
            elif marco == "PMI":
                pilar = str(row.get("pilar", "")).strip()
                if pilar and pilar.lower() not in ("nan", ""):
                    contexto = f" [Pilar: {pilar[:40]}]"

            lineas.append(f"{idx} | {id_orig}{contexto} | {titulo} — {desc}")
    return "\n".join(lineas)


# ───────────────── LIMPIEZA JSON ─────────────────

def limpiar_json(texto):
    texto = re.sub(r"```json", "", texto)
    texto = re.sub(r"```",    "", texto)
    texto = texto.strip()
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        texto = match.group(0)
    texto = re.sub(r",\s*([}\]])", r"\1", texto)
    return texto


# ───────────────── VALIDACIÓN DE SCORES ─────────────────

def _validar_score(val, default=0.0):
    try:
        v = float(val)
        return max(0.0, min(100.0, v))
    except (TypeError, ValueError):
        return float(default)


def _calcular_score_final(s_sem, s_nuc, s_kw):
    """80% semántico + 10% nuclear + 10% keywords — igual que alignment_engine."""
    return round(W_SEM * s_sem + W_NUC * s_nuc + W_KW * s_kw, 1)


# ───────────────── LLAMADA GEMINI ─────────────────

def llamar_gemini(prompt):
    """
    Llama a Gemini rotando entre todas las API keys disponibles.
    - Rota automáticamente si una key falla con 429 (cuota) o 403 (inválida).
    - Solo retorna error si TODAS las keys fallan.
    """
    api_keys = cargar_api_keys()
    if not api_keys:
        return {"error": "API key no configurada"}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.10,
            "topP": 0.90,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json",
        },
    }

    ultimo_error = "Error desconocido"
    for i, api_key in enumerate(api_keys, 1):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=TIMEOUT,
            )

            # Rotar en cuota agotada o key inválida — no es error fatal
            if response.status_code in (429, 403):
                ultimo_error = f"Key {i} agotada/inválida (HTTP {response.status_code})"
                continue

            if response.status_code != 200:
                ultimo_error = f"Gemini API error {response.status_code}"
                continue

            data  = response.json()
            texto = ""
            try:
                for part in data["candidates"][0]["content"]["parts"]:
                    texto += part.get("text", "")
            except Exception:
                ultimo_error = "Respuesta inesperada de Gemini"
                continue

            if not texto.strip():
                ultimo_error = "Respuesta vacía"
                continue

            texto = limpiar_json(texto)
            return json.loads(texto)

        except json.JSONDecodeError:
            ultimo_error = "JSON inválido en respuesta de Gemini"
            continue
        except requests.exceptions.Timeout:
            ultimo_error = f"Timeout (key {i})"
            continue
        except Exception as e:
            ultimo_error = str(e)
            continue

    return {"error": f"Todas las keys fallaron — último error: {ultimo_error}"}


# ───────────────── API PRINCIPAL ─────────────────

def alinear_con_ia(nombre, objetivo, descripcion, catalogos, max_por_marco=2):
    """
    Flujo completo:
    1) Filtro TF-IDF (top 20 candidatos por marco)
    2) Llamada a Gemini con prompt de sub-scores + límite max_por_marco
    3) Reintento con prompt de reparación si falla
    """
    catalogos_filtrados = filtrar_catalogo(
        nombre, objetivo, descripcion, catalogos
    )
    catalogo = construir_catalogo(catalogos_filtrados)

    prompt = PROMPT_TEMPLATE.format(
        nombre        = nombre        or "",
        objetivo      = objetivo      or "",
        descripcion   = descripcion   or "",
        catalogo      = catalogo,
        max_por_marco = max_por_marco,
    )

    resultado = llamar_gemini(prompt)
    if "error" not in resultado:
        return resultado

    # No reintentar si el error es de cuota/keys — el repair_prompt no ayudaría
    error_msg = resultado.get("error", "")
    if "429" in error_msg or "agotada" in error_msg or "Todas las keys" in error_msg:
        return resultado

    repair_prompt = (
        "El siguiente JSON tiene errores. Corrígelo y devuelve SOLO JSON válido "
        "con la misma estructura (ODS, PND, ENCI, PMI, CAD, SECTORES), "
        "cada entrada con: indice, id, score_semantico, score_nuclear, "
        "score_keywords, razon.\n\n"
        + json.dumps(resultado)
    )
    return llamar_gemini(repair_prompt)


# ───────────────── RESOLUCIÓN ROBUSTA DE ÍNDICE ─────────────────

def _resolver_fila(df, sel):
    """
    Busca la fila en orden de confiabilidad:
    1. id_original exacto   (campo 'id' del JSON)
    2. índice pandas (loc)
    3. posición (iloc)
    4. coincidencia parcial de título (fallback)
    """
    idx    = sel.get("indice")
    id_str = str(sel.get("id", "")).strip()

    # 1. Por id_original
    if id_str and "id_original" in df.columns:
        mask = df["id_original"].astype(str).str.strip() == id_str
        if mask.any():
            fila = df[mask].iloc[0]
            return fila, df[mask].index[0]

    # 2. df.loc
    try:
        if idx is not None and idx in df.index:
            return df.loc[idx], idx
    except Exception:
        pass

    # 3. df.iloc
    try:
        if idx is not None:
            pos = int(idx)
            if 0 <= pos < len(df):
                return df.iloc[pos], df.index[pos]
    except Exception:
        pass

    # 4. Coincidencia por título
    razon_norm = _normalizar(sel.get("razon", ""))
    if "titulo" in df.columns:
        for i, row in df.iterrows():
            titulo_norm = _normalizar(row.get("titulo", ""))
            if titulo_norm and (
                titulo_norm in razon_norm or razon_norm[:40] in titulo_norm
            ):
                return row, i

    return None, None


# ───────────────── CONSTRUCCIÓN RESULTADOS ─────────────────

def construir_resultados_ia(selecciones, catalogos, max_por_marco=2):
    """
    Convierte la respuesta de Gemini al formato interno de CICLOPE.

    CLAVE: el parámetro max_por_marco se aplica SIEMPRE como corte duro,
    independientemente de cuántas entradas devuelva Gemini.
    """
    if not selecciones or "error" in selecciones:
        return {}

    resultados = {}

    for marco, lista in selecciones.items():
        if not isinstance(lista, list) or len(lista) == 0:
            continue

        df = catalogos.get(marco)
        if df is None or df.empty:
            continue

        boost_marco  = _BOOST_MARCOS_DEFAULT.get(marco, 1.0)
        alineaciones = []

        for sel in lista:
            row, idx_real = _resolver_fila(df, sel)
            if row is None:
                continue

            s_sem = _validar_score(sel.get("score_semantico", sel.get("score", 0)))
            s_nuc = _validar_score(sel.get("score_nuclear",  0))
            s_kw  = _validar_score(sel.get("score_keywords", 0))

            # Fallback legado: solo 'score' sin sub-scores
            if s_sem == 0 and s_nuc == 0 and s_kw == 0:
                score_raw = _validar_score(sel.get("score", 0))
                s_sem = score_raw
                s_nuc = max(0.0, score_raw - 15.0)
                s_kw  = max(0.0, score_raw - 20.0)

            score_base  = _calcular_score_final(s_sem, s_nuc, s_kw)
            # Boost aditivo sobre el espacio libre hasta 97 — evita que scores
            # altos se acumulen siempre en el cap exacto de 97%
            headroom    = max(0.0, 97.0 - score_base)
            score_final = round(min(97.0, score_base + (boost_marco - 1.0) * headroom), 1)

            nivel = "Alto" if score_base >= 75 else "Medio" if score_base >= 50 else "Bajo"
            razon = str(sel.get("razon", "")).strip()

            alineaciones.append({
                "titulo":              str(row.get("titulo",      "")).strip(),
                "descripcion":         str(row.get("descripcion", "")).strip()[:300],
                "score":               score_final,
                "score_semantico":     round(s_sem, 1),
                "score_nuclear":       round(s_nuc, 1),
                "score_keywords":      round(s_kw,  1),
                "score_semantico_raw": round(s_sem, 1),
                "nivel_confianza":     nivel,
                "id_original":         row.get("id_original", idx_real),
                "razon_ia":            razon,
                "justificacion":       razon,
                "fuente":              "IA",
                "_boost_marco":        boost_marco,
                "_precision_mult":     1.0,
            })

        alineaciones.sort(key=lambda x: x["score"], reverse=True)

        # ── CORTE DURO: máximo max_por_marco resultados por marco ──
        if alineaciones:
            resultados[marco] = alineaciones[:max_por_marco]

    return resultados


# ───────────────── EXTRACCIÓN DE KEYWORDS ─────────────────

def extraer_keywords_ia(texto, max_keywords=10):
    if not texto:
        return []
    texto    = texto.lower()
    palabras = re.findall(r"\b[a-záéíóúñ]{4,}\b", texto)
    stopwords = {
        "para", "este", "esta", "estos", "estas", "desde", "hasta",
        "sobre", "entre", "mediante", "donde", "cuando", "porque",
        "tambien", "ademas", "proyecto", "programa", "desarrollo",
        "fortalecer", "promover", "cooperacion", "internacional",
        "objetivo", "descripcion", "actividad", "acciones",
        "implementacion", "resultado", "nacional", "colombia",
        "poblacion", "comunidades",
    }
    palabras = [p for p in palabras if p not in stopwords]
    frecuencia = {}
    for p in palabras:
        frecuencia[p] = frecuencia.get(p, 0) + 1
    ordenadas = sorted(frecuencia.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in ordenadas[:max_keywords]]


# ───────────────── COMPATIBILIDAD CON APP ─────────────────

def alinear_proyecto_con_ia(nombre, objetivo, descripcion, catalogos, max_por_marco=2):
    """Alias para mantener compatibilidad con app.py."""
    return alinear_con_ia(nombre, objetivo, descripcion, catalogos, max_por_marco)


def construir_resultados_desde_ia(selecciones, catalogos, max_por_marco=2):
    """Alias para mantener compatibilidad con app.py."""
    return construir_resultados_ia(selecciones, catalogos, max_por_marco)
