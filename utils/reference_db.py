import json
import re
import os
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ───────────────── CONFIG ─────────────────

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)

TIMEOUT = 40

# modelo embeddings
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# caches globales
FAISS_INDEX = {}
FAISS_DATA = {}

# ───────────────── PROMPT ─────────────────

PROMPT_TEMPLATE = """
Eres un experto en cooperación internacional y marcos normativos de Colombia.

Analiza el siguiente proyecto.

PROYECTO
Nombre: {nombre}
Objetivo: {objetivo}
Descripción: {descripcion}

CATÁLOGO
{catalogo}

INSTRUCCIONES

Paso 1
Identifica los 5 temas principales del proyecto.

Paso 2
Relaciona esos temas con el catálogo.

Paso 3
Selecciona las 2 entradas MÁS alineadas de cada marco.

CRITERIOS
- relación directa con el objetivo
- coherencia con cooperación internacional
- impacto esperado
- pertinencia temática

Si un marco no tiene alineación clara, no lo incluyas.

Devuelve SOLO JSON válido.

Formato:

{{
"ODS":[{{"indice":0,"score":85,"razon":"..."}}],
"PND":[{{"indice":0,"score":80,"razon":"..."}}],
"ENCI":[{{"indice":0,"score":75,"razon":"..."}}],
"PMI":[{{"indice":0,"score":70,"razon":"..."}}],
"CAD":[{{"indice":0,"score":60,"razon":"..."}}],
"SECTORES":[{{"indice":0,"score":65,"razon":"..."}}]
}}
"""

# ───────────────── API KEY ─────────────────


def cargar_api_key():

    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "").strip()
        if len(key) > 20:
            return key
    except Exception:
        pass

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if len(key) > 20 else ""


# ───────────────── FAISS ─────────────────


def construir_indices_faiss(catalogos):

    global FAISS_INDEX, FAISS_DATA

    for marco, df in catalogos.items():

        if df is None or df.empty:
            continue

        textos = (
            df["titulo"].fillna("") + " " + df["descripcion"].fillna("")
        ).tolist()

        embeddings = modelo_embeddings.encode(textos)

        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)

        index.add(np.array(embeddings).astype("float32"))

        FAISS_INDEX[marco] = index
        FAISS_DATA[marco] = df.reset_index(drop=False)


def buscar_marcos_similares(nombre, objetivo, descripcion, top_n=15):

    texto_proyecto = f"{nombre} {objetivo} {descripcion}"

    embedding_query = modelo_embeddings.encode([texto_proyecto])

    catalogos_filtrados = {}

    for marco, index in FAISS_INDEX.items():

        D, I = index.search(
            np.array(embedding_query).astype("float32"), top_n
        )

        df = FAISS_DATA[marco]

        resultados = df.iloc[I[0]].copy()

        catalogos_filtrados[marco] = resultados

    return catalogos_filtrados


# ───────────────── CATÁLOGO TEXTO ─────────────────


def construir_catalogo(catalogos):

    lineas = []

    for marco in ["ODS", "PND", "ENCI", "PMI", "CAD", "SECTORES"]:

        df = catalogos.get(marco)

        if df is None or df.empty:
            continue

        lineas.append(f"\n[{marco}]")

        for idx, row in df.iterrows():

            titulo = str(row.get("titulo", "")).strip()
            desc = str(row.get("descripcion", "")).strip()[:120]

            lineas.append(f"{idx} | {titulo} — {desc}")

    return "\n".join(lineas)


# ───────────────── LIMPIEZA JSON ─────────────────


def limpiar_json(texto):

    texto = re.sub(r"```json", "", texto)
    texto = re.sub(r"```", "", texto)

    texto = texto.strip()

    match = re.search(r"\{.*\}", texto, re.DOTALL)

    if match:
        texto = match.group(0)

    texto = re.sub(r",\s*([}\]])", r"\1", texto)

    return texto


# ───────────────── GEMINI ─────────────────


def llamar_gemini(prompt):

    api_key = cargar_api_key()

    if not api_key:
        return {"error": "API key no configurada"}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.15,
            "topP": 0.95,
            "maxOutputTokens": 1500,
            "responseMimeType": "application/json",
        },
    }

    try:

        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT,
        )

        if response.status_code != 200:

            return {
                "error": f"Gemini API error {response.status_code}"
            }

        data = response.json()

        texto = ""

        try:

            for part in data["candidates"][0]["content"]["parts"]:
                texto += part.get("text", "")

        except Exception:

            return {"error": "Respuesta inesperada"}

        if not texto.strip():
            return {"error": "Respuesta vacía"}

        texto = limpiar_json(texto)

        return json.loads(texto)

    except json.JSONDecodeError:

        return {"error": "JSON inválido"}

    except requests.exceptions.Timeout:

        return {"error": "Timeout de Gemini"}

    except Exception as e:

        return {"error": str(e)}


# ───────────────── API PRINCIPAL ─────────────────


def alinear_con_ia(nombre, objetivo, descripcion):

    catalogos_filtrados = buscar_marcos_similares(
        nombre, objetivo, descripcion
    )

    catalogo = construir_catalogo(catalogos_filtrados)

    prompt = PROMPT_TEMPLATE.format(
        nombre=nombre or "",
        objetivo=objetivo or "",
        descripcion=descripcion or "",
        catalogo=catalogo,
    )

    resultado = llamar_gemini(prompt)

    if "error" not in resultado:
        return resultado

    repair_prompt = (
        "Corrige y devuelve JSON válido:\n"
        + json.dumps(resultado)
    )

    return llamar_gemini(repair_prompt)


# ───────────────── RESULTADOS ─────────────────


def construir_resultados_ia(selecciones, catalogos):

    if not selecciones or "error" in selecciones:
        return {}

    resultados = {}

    for marco, lista in selecciones.items():

        df = catalogos.get(marco)

        if df is None:
            continue

        alineaciones = []

        for sel in lista:

            idx = sel.get("indice")
            score = float(sel.get("score", 0))
            razon = sel.get("razon", "")

            try:
                row = df.loc[idx]
            except Exception:
                try:
                    row = df.iloc[int(idx)]
                except Exception:
                    continue

            alineaciones.append(
                {
                    "titulo": row.get("titulo", ""),
                    "descripcion": row.get("descripcion", "")[:300],
                    "score": score,
                    "score_semantico": score,
                    "score_nuclear": 0,
                    "score_keywords": 0,
                    "nivel_confianza": (
                        "Alto"
                        if score >= 70
                        else "Medio"
                        if score >= 45
                        else "Bajo"
                    ),
                    "id_original": row.get("id_original", idx),
                    "razon_ia": razon,
                    "fuente": "IA",
                }
            )

        alineaciones.sort(key=lambda x: x["score"], reverse=True)

        resultados[marco] = alineaciones

    return resultados
