"""
CICLOPE v23.0 - Alignment Engine
APC COLOMBIA | Sistema de Alineación Estratégica

CAMBIOS v19.0:
  v19-1 → Scoring híbrido: 60% sem / 25% nuclear / 10% kw / 5% histórico
  v19-2 → SectorProfileEngine: perfiles vectoriales por sector de gobierno
  v19-3 → Boost sectorial directo al score cuando proyecto coincide con perfil
  v19-4 → Bloqueos CAD duros: desminado/biosfera (×0.18), VIH/tráfico (×0.30)
  v19-5 → ODS boost VIH→3.3 ×1.55 aplicado al precision_mult con efecto real
  v19-6 → Normalización vectores antes del cálculo coseno (consistencia dimensional)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    W_SEMANTICO_CON_REF, W_KEYWORDS_CON_REF, W_REFERENCIA_CON_REF,
    W_SEMANTICO_SIN_REF, W_KEYWORDS_SIN_REF,
    W_NUCLEARES_SIN_REF, W_NUCLEARES_CON_REF,
    SIM_MIN_CALIBRACION, SIM_MAX_CALIBRACION,
    UMBRAL_SEMANTICO_MINIMO, UMBRAL_SEMANTICO_SECTORES,
    UMBRAL_EXCELENTE, UMBRAL_BUENO, UMBRAL_MODERADO,
    BOOST_MARCOS,
    ODS_BOOST_ESPECIFICO,
    ODS_PENALIZAR_SIN_CONTEXTO,
    ODS_PENALIZACION_SIN_CONTEXTO_FACTOR,
    ODS_BLOQUEOS_TEMATICOS,
    ODS_BLOQUEO_FACTOR,
    MARCO_PENALIZACIONES_TEMATICAS,
    NUCLEAR_BOOST_FACTOR,
    NUCLEAR_PENALTY_FACTOR,
    NUCLEAR_MIN_MATCHES,
    PALABRAS_NUCLEARES_DEFAULT,
    SECTOR_PROFILES,
    SECTOR_PROFILE_BOOST,
    SECTOR_PROFILE_THRESHOLD,
    ENCI_BOOST_ESPECIFICO, PND_BOOST_ESPECIFICO,
    CAD_BOOST_ESPECIFICO, SECTORES_BOOST_ESPECIFICO,
    CAD_WHITELIST_ENTRIES,
)

MODOS_VALIDOS = {"con_referencia", "sin_referencia"}
CACHE_VERSION = "v26"

_DESC_VENTANA_PRIMARIA = 200   # ampliada a 200 para capturar más contexto


# ══════════════════════════════════════════════════════════════════════════
# TÉRMINOS ULTRA-ESPECÍFICOS
# ══════════════════════════════════════════════════════════════════════════
_ULTRA_ESPECIFICOS: frozenset = frozenset({
    "vih", "sida", "hiv", "aids",
    "arv", "tarv", "antiretroviral", "antirretroviral",
    "prep", "pep", "carga viral", "cd4", "pvvih",
    "its", "ets", "hsh",
    "mitigacion social del vih",
    "respuesta al vih", "prevencion del vih",
    "prueba rapida de vih", "diagnostico vih",
    "coinfeccion vih", "poblaciones clave vih",
    "hombres que tienen relaciones sexuales",
    "personas transgenero",
    "trabajo sexual", "trabajadoras sexuales",
    "usuarios drogas inyectables",
    "desminado", "aicma", "antipersonal",
    "artefactos explosivos", "municiones sin explotar",
    "accion contra minas",
    "victimas de minas",
    "remanentes explosivos", "desminado humanitario",
    "minas antipersonal",
    "liberacion de tierras",
    "despeje humanitario", "estudio no tecnico",
    "equipos multitarea",
    "acueducto", "alcantarillado",
    "planta de tratamiento de agua", "agua potable",
    "primera infancia", "desarrollo infantil",
    "explotacion infantil", "trabajo infantil",
    "reclutamiento de menores", "maltrato infantil",
    "reincorporacion", "acuerdo de paz", "paz total",
    "pnis", "posconflicto", "sustitucion de cultivos",
    "excombatientes", "firmantes de paz",
    "pdet", "zomac", "apc colombia",
    "cooperacion sur sur", "cooperacion triangular",
    "feminicidio", "vbg",
    "violencia basada en genero", "lgbti",
    "mujeres rurales", "empoderamiento femenino",
    "reforma agraria", "agricultura familiar",
    "agroemprendedor", "paisajes forestales",
    "adaptacion al cambio climatico",
    "mitigacion cambio climatico",
    "soluciones basadas en la naturaleza",
    # v27: Nuevos términos ultra-específicos
    "asistencia humanitaria", "ayuda humanitaria",
    "cluster humanitario", "accion humanitaria",
    "trata de personas", "trata de ninos",
    "explotacion sexual comercial infantil", "escnna",
    "violencia intrafamiliar", "violencia sexual",
    "violencia domestica",
    "entornos protectores", "proteccion infantil",
    "ninez adolescencia",
    "educacion riesgo minas", "erm",
    "atencion victimas minas",
    "wash humanitario", "saneamiento emergencia",
    "medios de vida", "seguridad alimentaria emergencia",
    "desnutricion aguda", "muac",
    "generaciones sin violencias",
    "masculinidades", "cambio de normas sociales",
})

_DOMINIOS_SUAVES: frozenset = frozenset({
    "economia circular", "plastico", "embalaje", "reciclaje",
    "reutilizacion", "ecodiseno", "produccion limpia",
})

_DOMINIOS_CERRADOS: frozenset = frozenset({
    "vih", "sida", "hiv", "aids", "arv", "tarv", "prep", "pep",
    "desminado", "aicma", "antipersonal", "minas antipersonal",
    "artefactos explosivos", "accion contra minas",
    "liberacion de tierras",
    "acueducto", "alcantarillado", "agua potable",
    "planta de tratamiento de agua",
    # v27: nuevos dominios cerrados
    "asistencia humanitaria", "cluster humanitario",
    "trata de personas", "escnna",
    "explotacion sexual comercial infantil",
})

_SECTOR_TERMINOS: dict = {
    "salud":       {"vih", "sida", "hiv", "aids", "arv", "tarv", "prep", "pep",
                    "its", "ets", "hsh", "carga viral", "cd4", "pvvih",
                    "antirretroviral", "salud publica", "epidemia",
                    "tuberculosis", "hepatitis", "malaria", "dengue",
                    "salud mental", "nutricion", "mortalidad materna",
                    "agua potable", "acueducto", "alcantarillado"},
    "defensa":     {"desminado", "aicma", "antipersonal", "artefactos explosivos",
                    "minas antipersonal", "accion contra minas",
                    "liberacion de tierras", "despeje humanitario",
                    "municiones sin explotar", "remanentes explosivos",
                    "equipos multitarea", "estudio no tecnico",
                    "victimas de minas", "map", "muse",
                    "campaña contra minas", "contaminacion artefactos"},
    "interior":    {"paz total", "posconflicto", "reincorporacion",
                    "acuerdo de paz", "excombatientes", "firmantes de paz",
                    "pdet", "sustitucion de cultivos", "pnis"},
    "ambiente":    {"adaptacion al cambio climatico", "mitigacion cambio climatico",
                    "economia circular", "plastico", "reciclaje", "biodiversidad",
                    "deforestacion", "bosques", "ecosistemas", "paramos",
                    "soluciones basadas en la naturaleza"},
    "comercio":    {"economia circular", "plastico", "embalaje", "pymes",
                    "exportaciones", "produccion limpia", "competitividad",
                    "cadenas de valor", "internacionalizacion"},
    "agricultura": {"agroemprendedor", "agricultura climaticamente inteligente",
                    "reforma agraria", "agroecologia", "agricultura familiar",
                    "campesinos", "campesinado", "pequenos productores",
                    "seguridad alimentaria"},
    "vivienda":    {"acueducto", "alcantarillado", "agua potable",
                    "planta de tratamiento de agua", "saneamiento basico",
                    "wash", "infraestructura hidrica"},
    "exterior":    {"cooperacion sur sur", "cooperacion triangular",
                    "cooperacion internacional", "apc colombia"},
    "presidencia": {"paz total", "posconflicto", "pdet",
                    "reincorporacion"},
    "educacion":   {"educacion", "aprendizaje", "escolar", "formacion docente",
                    "primera infancia", "inclusion educativa"},
    "justicia":    {"maltrato infantil", "trabajo infantil",
                    "explotacion infantil", "vbg", "feminicidio",
                    "trata de personas", "violencia intrafamiliar",
                    "violencia basada en genero", "lgbti",
                    "reclutamiento de menores"},
    "icbf":        {"primera infancia", "ninez", "adolescencia", "familia",
                    "proteccion infantil", "trabajo infantil",
                    "maltrato infantil", "adopcion"},
    "cancilleria": {"cooperacion internacional", "relaciones exteriores",
                    "diplomacia", "migración", "venezolanos",
                    "refugiados", "asilo", "consulados"},
    "prosperidad": {"asistencia humanitaria", "poblaciones vulnerables",
                    "desplazados", "inclusion social", "superacion pobreza",
                    "transferencias monetarias", "familias en accion"},
}

# ═══ v27: Bloqueos temáticos de sector — penaliza matches claramente incorrectos ═══
_SECTOR_BLOQUEOS: dict = {
    # Cuando hay términos claros de desminado, penalizar sectores no relacionados
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal",
               "artefactos explosivos"}): {
        "salud", "educacion", "ambiente", "vivienda", "icbf",
    },
    # Cuando hay términos claros de VIH, penalizar sectores no relacionados
    frozenset({"vih", "sida", "hiv", "aids", "antirretroviral", "tarv", "pvvih"}): {
        "defensa", "agricultura", "ambiente", "comercio",
    },
    # Cuando hay términos claros de medio ambiente, penalizar defensa/salud
    frozenset({"biodiversidad", "deforestacion", "ecosistemas",
               "economia circular"}): {
        "defensa", "justicia",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# SECTOR PROFILE ENGINE — v19 NUEVO
# ══════════════════════════════════════════════════════════════════════════

class SectorProfileEngine:
    """
    Precalcula embeddings representativos por sector de gobierno.
    Permite comparar un proyecto no solo contra metas individuales,
    sino también contra el vector promedio del sector relevante.
    """

    def __init__(self, nlp_processor):
        self.nlp = nlp_processor
        self._perfiles: dict = {}   # {sector: embedding_array}
        self._precalcular()

    def _precalcular(self):
        print("🏛️  Precalculando perfiles sectoriales...")
        for sector, texto in SECTOR_PROFILES.items():
            emb = self.nlp.generar_embedding(texto)
            norm = np.linalg.norm(emb)
            self._perfiles[sector] = emb / norm if norm > 0 else emb
        print(f"   ✅ {len(self._perfiles)} perfiles sectoriales listos")

    def detectar_sector(self, embedding_proyecto: np.ndarray) -> tuple[str, float]:
        """
        Retorna el sector más similar al proyecto y su score de similitud.
        El embedding_proyecto debe estar normalizado.
        """
        if not self._perfiles:
            return "", 0.0
        mejor_sector = ""
        mejor_sim    = -1.0
        for sector, emb_perfil in self._perfiles.items():
            sim = float(np.dot(embedding_proyecto, emb_perfil))
            if sim > mejor_sim:
                mejor_sim    = sim
                mejor_sector = sector
        return mejor_sector, max(0.0, mejor_sim)

    def calcular_boost_sectorial(self, embedding_proyecto_norm: np.ndarray,
                                  texto_marco_lower: str,
                                  sector_detectado: str) -> float:
        """
        v19-3: Aplica boost al score cuando:
          - El proyecto tiene similitud alta con un perfil sectorial.
          - El texto del marco coincide con ese sector.
        Retorna boost adicional [0, SECTOR_PROFILE_BOOST].
        """
        if sector_detectado not in self._perfiles:
            return 0.0
        sim_sector = float(np.dot(embedding_proyecto_norm,
                                   self._perfiles[sector_detectado]))
        if sim_sector < SECTOR_PROFILE_THRESHOLD:
            return 0.0
        # Verificar si el marco tiene contenido del sector
        palabras_sector = set(SECTOR_PROFILES[sector_detectado].lower().split())
        palabras_marco  = set(texto_marco_lower.split())
        overlap = len(palabras_sector & palabras_marco)
        if overlap >= 3:
            return SECTOR_PROFILE_BOOST * min(1.0, (sim_sector - SECTOR_PROFILE_THRESHOLD) / 0.20)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# ALIGNMENT ENGINE
# ══════════════════════════════════════════════════════════════════════════

class AlignmentEngine:
    """
    Motor de alineación multi-señal v19.
    Pipeline (11 pasos):
      1.  Similitud coseno semántica → calibrada [0, 0.97]
      2.  Score keywords del dominio → [0, 1]
      3.  Score palabras nucleares   → [0, 1]
      4.  Score histórico (modo con_referencia) → [0, 1]
      5.  Combinación ponderada (60/25/10/5)
      6.  Boost sectorial (SectorProfileEngine) → +0..12 pts
      7.  Penalización texto corto
      8.  Precision multiplier (boosts/penalties ultra-específicos)
      9.  Reglas ODS específicas
      10. Penalizaciones temáticas ENCI/PMI/CAD
      11. Boost marco + cap 97%
    """

    def __init__(self, nlp_processor, catalogos, reference_db=None,
                 modo: str = "sin_referencia",
                 palabras_nucleares: list = None,
                 activar_pmi: bool = True,
                 pesos_custom: dict = None):

        if modo not in MODOS_VALIDOS:
            raise ValueError(f"modo debe ser uno de {MODOS_VALIDOS}")

        self.nlp              = nlp_processor
        self.catalogos        = catalogos
        self.ref_db           = reference_db
        self.modo             = modo
        self.activar_pmi      = activar_pmi
        self.embeddings_cache = {}

        self.palabras_nucleares = palabras_nucleares or PALABRAS_NUCLEARES_DEFAULT
        self.nlp.set_palabras_nucleares(self.palabras_nucleares)

        self._aplicar_pesos(modo, pesos_custom)

        # v19-2: inicializar perfiles sectoriales
        self.sector_engine = SectorProfileEngine(nlp_processor)

        self._precalcular_embeddings()

    def _aplicar_pesos(self, modo: str, pesos_custom: dict = None):
        """Aplica pesos estándar o custom (del sidebar)."""
        if pesos_custom:
            self.w_semantico  = pesos_custom.get("semantico",  W_SEMANTICO_SIN_REF)
            self.w_nucleares  = pesos_custom.get("nucleares",  W_NUCLEARES_SIN_REF)
            self.w_keywords   = pesos_custom.get("keywords",   W_KEYWORDS_SIN_REF)
            self.w_referencia = pesos_custom.get("referencia", 0.0)
        elif modo == "con_referencia":
            self.w_semantico  = W_SEMANTICO_CON_REF
            self.w_nucleares  = W_NUCLEARES_CON_REF
            self.w_keywords   = W_KEYWORDS_CON_REF
            self.w_referencia = W_REFERENCIA_CON_REF
        else:
            self.w_semantico  = W_SEMANTICO_SIN_REF
            self.w_nucleares  = W_NUCLEARES_SIN_REF
            self.w_keywords   = W_KEYWORDS_SIN_REF
            self.w_referencia = 0.0
            self.ref_db       = None

    def actualizar_pmi(self, activar: bool):
        self.activar_pmi = activar

    def actualizar_palabras_nucleares(self, palabras: list):
        self.palabras_nucleares = [p.strip().lower() for p in palabras if p.strip()]
        self.nlp.set_palabras_nucleares(self.palabras_nucleares)

    def actualizar_pesos(self, pesos: dict):
        """v19-6: actualización dinámica de pesos desde el sidebar."""
        self.w_semantico  = pesos.get("semantico",  self.w_semantico)
        self.w_nucleares  = pesos.get("nucleares",  self.w_nucleares)
        self.w_keywords   = pesos.get("keywords",   self.w_keywords)
        self.w_referencia = pesos.get("referencia", self.w_referencia)

    @staticmethod
    def _calibrar_similitud(sim_raw: float) -> float:
        cal = (sim_raw - SIM_MIN_CALIBRACION) / (SIM_MAX_CALIBRACION - SIM_MIN_CALIBRACION)
        return max(0.0, min(0.97, cal))

    @staticmethod
    def _normalizar(emb: np.ndarray) -> np.ndarray:
        """v19-5: normalización explícita para consistencia dimensional."""
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _cache_key_marco(self, marco: str) -> Path:
        """v20: cache incluye modelo y dimension para evitar mezclas."""
        modelo_slug = (self.nlp.modelo_cargado or "unknown").replace("/","_").replace("-","_")
        dim = self.nlp.modelo.get_sentence_embedding_dimension() if self.nlp.modelo else 768
        return Path(f"cache/embeddings/{marco}_{modelo_slug}_d{dim}.pkl")

    def _precalcular_embeddings(self):
        print("Precalculando embeddings de marcos estratégicos...")
        dim_actual = self.nlp.modelo.get_sentence_embedding_dimension() if self.nlp.modelo else 768
        for marco, df in self.catalogos.items():
            ruta_cache = self._cache_key_marco(marco)
            embeddings = self.nlp.cargar_embeddings_cache(ruta_cache)
            # v20: validar dimension de cache
            if (embeddings is not None and len(embeddings) == len(df)
                    and (embeddings.ndim < 2 or embeddings.shape[1] == dim_actual)):
                print(f"  📦 Cache {marco}: {len(embeddings)} embeddings dim={dim_actual}")
            else:
                if embeddings is not None:
                    print(f"  ⚠️  Cache {marco} dim mismatch → regenerando")
                print(f"  Calculando {marco} ({len(df)} elementos)...")
                textos     = df["texto_completo"].tolist()
                embeddings = self.nlp.generar_embeddings_batch(textos)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms
                self.nlp.guardar_embeddings_cache(embeddings, ruta_cache)
                print(f"  ✅ {marco}: {len(embeddings)} embeddings normalizados dim={dim_actual}")
            self.embeddings_cache[marco] = embeddings

    def _detectar_ultra_proyecto(self, texto: str) -> frozenset:
        if not texto:
            return frozenset()
        txt = texto.lower()
        encontrados = {t for t in _ULTRA_ESPECIFICOS if t in txt}
        encontrados |= {
            a.lower()
            for a in re.findall(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto)
            if a.lower() in _ULTRA_ESPECIFICOS
        }
        return frozenset(encontrados)

    def _detectar_suaves_proyecto(self, texto: str) -> frozenset:
        if not texto:
            return frozenset()
        txt = texto.lower()
        return frozenset(t for t in _DOMINIOS_SUAVES if t in txt)

    def alinear_proyecto(self, texto_proyecto: str,
                         umbral_minimo: float = 0.30,
                         top_n: int = 5) -> dict:
        emb_raw      = self.nlp.generar_embedding(texto_proyecto)
        emb_norm     = self._normalizar(emb_raw)   # normalizado para sector engine

        contexto_historico = {}
        if self.ref_db and self.ref_db.cargado:
            contexto_historico = self.ref_db.obtener_contexto_referencia(emb_raw)

        keywords_detectadas  = self.nlp.analizar_keywords_en_texto(texto_proyecto)
        # Pasar el set default para distinguir nucleares del usuario vs sistema
        _nucleares_default_set = set(n.strip().lower() for n in PALABRAS_NUCLEARES_DEFAULT)
        nucleares_detectadas = self.nlp.detectar_nucleares_en_texto(
            texto_proyecto, self.palabras_nucleares,
            nucleares_default=_nucleares_default_set
        )

        # v19-2: detectar sector del proyecto
        sector_detectado, sim_sector = self.sector_engine.detectar_sector(emb_norm)

        resultados: dict = {}
        for marco, df in self.catalogos.items():
            if marco == "PMI" and not self.activar_pmi:
                resultados[marco] = []
                continue
            resultados[marco] = self._calcular_alineaciones(
                texto_proyecto, emb_raw, emb_norm,
                marco, df, umbral_minimo, top_n,
                sector_detectado,
            )

        resultados["_meta"] = {
            "keywords_detectadas":  keywords_detectadas,
            "nucleares_detectadas": nucleares_detectadas,
            "contexto_historico":   contexto_historico,
            "modo":                 self.modo,
            "pmi_activo":           self.activar_pmi,
            "sector_detectado":     sector_detectado,
            "similitud_sector":     round(sim_sector * 100, 1),
            "pesos": {
                "semantico":  round(self.w_semantico  * 100),
                "nucleares":  round(self.w_nucleares  * 100),
                "keywords":   round(self.w_keywords   * 100),
                "referencia": round(self.w_referencia * 100),
            },
            "total_alineaciones": sum(
                len(v) for k, v in resultados.items() if not k.startswith("_")
            ),
        }
        return resultados

    def _calcular_alineaciones(self, texto_proyecto, emb_raw, emb_norm,
                                marco, df, umbral_minimo, top_n,
                                sector_detectado):
        embeddings_marco = self.embeddings_cache.get(marco)
        if embeddings_marco is None:
            return []

        palabras_proyecto = len(texto_proyecto.split())
        boost_marco       = BOOST_MARCOS.get(marco, 1.0)

        ultra_proyecto  = self._detectar_ultra_proyecto(texto_proyecto)
        suaves_proyecto = self._detectar_suaves_proyecto(texto_proyecto)

        ods_boosts_activos   = []
        ods_bloqueos_activos = set()
        if marco == "ODS":
            ods_boosts_activos, ods_bloqueos_activos = self._preparar_reglas_ods(
                ultra_proyecto, suaves_proyecto
            )

        penalizaciones_marco_activas = []
        if marco in {"ENCI", "PMI", "CAD", "PND", "ODS", "SECTORES"}:
            penalizaciones_marco_activas = self._preparar_penalizaciones_marco(
                marco, ultra_proyecto
            )

        candidatos = []

        for i in range(len(df)):
            fila        = df.iloc[i]
            texto_marco = str(fila.get("texto_completo", ""))

            # v19-5: similitud con embeddings ya normalizados → producto punto = coseno
            sim_raw = float(np.dot(emb_norm, embeddings_marco[i]))
            sim_raw = max(0.0, min(1.0, sim_raw))

            if sim_raw < UMBRAL_SEMANTICO_MINIMO:
                continue
            if marco == "SECTORES" and sim_raw < UMBRAL_SEMANTICO_SECTORES:
                continue

            sim_cal  = self._calibrar_similitud(sim_raw)
            score_kw = self.nlp.calcular_score_keywords_dominio(texto_proyecto, texto_marco)

            score_nuclear = self.nlp.calcular_score_nuclear(
                texto_proyecto, texto_marco, self.palabras_nucleares
            )

            score_ref = 0.0
            if self.ref_db and self.ref_db.cargado and self.w_referencia > 0:
                titulo_marco = str(fila.get("titulo", ""))
                score_ref = self.ref_db.obtener_voto_referencia(
                    emb_raw, marco, titulo_marco
                )

            # v19-1: scoring híbrido 60/25/10/5
            score = (
                self.w_semantico  * sim_cal      +
                self.w_nucleares  * score_nuclear +
                self.w_keywords   * score_kw      +
                self.w_referencia * score_ref
            )

            # v19-2: boost sectorial directo al score
            boost_sect = self.sector_engine.calcular_boost_sectorial(
                emb_norm, texto_marco.lower(), sector_detectado
            )
            score = min(0.97, score + boost_sect)

            # Penalizar textos muy cortos
            if palabras_proyecto <= 8:
                score *= 0.92
            elif palabras_proyecto <= 15:
                score *= 0.97

            # ─── PRECISION BOOST ──────────────────────────────────────
            titulo_lower = str(fila.get("titulo",      "")).lower()
            # v25: normalizar tildes para que los bloqueos temáticos funcionen
            titulo_lower_norm = titulo_lower
            for _a, _b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
                titulo_lower_norm = titulo_lower_norm.replace(_a, _b)
            desc_raw     = str(fila.get("descripcion", ""))
            desc_prim    = desc_raw[:_DESC_VENTANA_PRIMARIA].lower()

            ultra_title = sum(1 for t in ultra_proyecto if t in titulo_lower)
            ultra_desc  = sum(1 for t in ultra_proyecto if t in desc_prim)

            precision_mult = 1.0

            if ultra_title >= 2:   precision_mult = 1.18
            elif ultra_title == 1: precision_mult = 1.12

            if precision_mult == 1.0:
                if ultra_desc >= 2:    precision_mult = 1.10
                elif ultra_desc == 1:  precision_mult = 1.06

            suave_title = sum(1 for t in suaves_proyecto if t in titulo_lower)
            suave_desc  = sum(1 for t in suaves_proyecto if t in desc_prim)
            if suave_title >= 1 or suave_desc >= 1:
                precision_mult = max(precision_mult, 1.08)

            if marco == "SECTORES" and ultra_proyecto:
                # v27: boost más fuerte para sectores correctos
                for sector_kw, sector_ultras in _SECTOR_TERMINOS.items():
                    if sector_kw in titulo_lower:
                        hits = sum(1 for u in ultra_proyecto if u in sector_ultras)
                        hits += sum(1 for u in suaves_proyecto if u in sector_ultras)
                        if hits >= 3:
                            precision_mult = max(precision_mult, 1.55)
                        elif hits >= 2:
                            precision_mult = max(precision_mult, 1.40)
                        elif hits >= 1:
                            precision_mult = max(precision_mult, 1.25)
                        break
                # v27: penalizar sectores claramente incorrectos
                for bloqueo_ultras, sectores_bloqueados in _SECTOR_BLOQUEOS.items():
                    if bloqueo_ultras & ultra_proyecto:
                        for sector_bloq in sectores_bloqueados:
                            if sector_bloq in titulo_lower:
                                precision_mult = min(precision_mult, 0.45)
                                break

            es_cabecera = (
                desc_raw.strip().startswith("[Pilar:") or
                desc_raw.strip().startswith("[Capítulo:") or
                desc_raw.strip().startswith("[Cap")
            )
            if es_cabecera and ultra_title == 0:
                penalty = 0.82 if ultra_desc == 0 else 0.90
                precision_mult = min(precision_mult, penalty)

            tiene_ultra_cerrado = bool(ultra_proyecto & _DOMINIOS_CERRADOS)
            if tiene_ultra_cerrado and ultra_title == 0 and ultra_desc == 0:
                precision_mult = min(precision_mult, 0.78)

            # v19-4: palabras nucleares en título del marco
            nucleares_en_titulo = sum(
                1 for n in self.palabras_nucleares if n.lower() in titulo_lower
            )
            if nucleares_en_titulo >= 2:
                precision_mult = max(precision_mult,
                                     min(precision_mult * NUCLEAR_BOOST_FACTOR, 1.45))
            elif nucleares_en_titulo == 1:
                precision_mult = max(precision_mult,
                                     min(precision_mult * 1.15, 1.28))

            # ─── ODS REGLAS ──────────────────────────────────────────
            if marco == "ODS":
                for substr, factor in ods_boosts_activos:
                    if substr in titulo_lower:
                        precision_mult = max(precision_mult, factor)
                        break

                for bloqueo_frag in ods_bloqueos_activos:
                    if bloqueo_frag in titulo_lower or bloqueo_frag in desc_prim:
                        precision_mult = min(precision_mult, ODS_BLOQUEO_FACTOR)
                        break

                for ods17_substr, kw_just in ODS_PENALIZAR_SIN_CONTEXTO.items():
                    if ods17_substr in titulo_lower:
                        tiene_contexto = any(kw in texto_proyecto.lower() for kw in kw_just)
                        if not tiene_contexto:
                            precision_mult = min(precision_mult,
                                                 ODS_PENALIZACION_SIN_CONTEXTO_FACTOR)
                        break

            # ─── PENALIZACIONES MARCO ────────────────────────────────
            if penalizaciones_marco_activas:
                for substr_titulo_pen, factor_pen in penalizaciones_marco_activas:
                    if substr_titulo_pen in titulo_lower_norm:
                        precision_mult = min(precision_mult, factor_pen)
                        break

            # ─── v23: BOOST ENCI ESPECIFICO ──────────────────────────
            if marco == "ENCI":
                for kw_set, boosts in ENCI_BOOST_ESPECIFICO.items():
                    if kw_set & ultra_proyecto or kw_set & suaves_proyecto:
                        for entry_substr, factor in boosts:
                            if entry_substr in titulo_lower_norm or entry_substr in desc_prim:
                                precision_mult = max(precision_mult, factor)
                                break

            # ─── v23: BOOST PND ESPECIFICO ───────────────────────────
            if marco == "PND":
                for kw_set, boosts in PND_BOOST_ESPECIFICO.items():
                    if kw_set & ultra_proyecto or kw_set & suaves_proyecto:
                        for catalizador_substr, factor in boosts:
                            if catalizador_substr in titulo_lower_norm:
                                precision_mult = max(precision_mult, factor)
                                break

            # ─── v29: BOOST CAD ESPECIFICO ───────────────────────────
            if marco == "CAD":
                for kw_set, boosts in CAD_BOOST_ESPECIFICO.items():
                    if kw_set & ultra_proyecto or kw_set & suaves_proyecto:
                        for entry_substr, factor in boosts:
                            if entry_substr in titulo_lower_norm or entry_substr in desc_prim:
                                precision_mult = max(precision_mult, factor)
                                break

            # ─── v29: BOOST SECTORES ESPECIFICO ──────────────────────
            if marco == "SECTORES":
                for kw_set, boosts in SECTORES_BOOST_ESPECIFICO.items():
                    if kw_set & ultra_proyecto or kw_set & suaves_proyecto:
                        for entry_substr, factor in boosts:
                            if entry_substr in titulo_lower_norm or entry_substr in desc_prim:
                                precision_mult = max(precision_mult, factor)
                                break

            score = min(0.97, score * precision_mult * boost_marco)

            # ─── v30: CAD WHITELIST — bloqueo absoluto post-boosts ────
            # Se aplica al score FINAL y resetea precision_mult a 1.0
            # para que NUCLEAR_BOOST y otros boosts no inflen el bloqueo
            if marco == "CAD":
                texto_proy_norm = texto_proyecto.lower()
                for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u")]:
                    texto_proy_norm = texto_proy_norm.replace(a, b)
                for cad_substr, wl_config in CAD_WHITELIST_ENTRIES.items():
                    if cad_substr in titulo_lower_norm:
                        kw_req      = wl_config["keywords_requeridas"]
                        factor_bloq = wl_config["factor_bloqueo"]
                        tiene_kw    = any(kw in texto_proy_norm for kw in kw_req)
                        if not tiene_kw:
                            # Recalcular score desde cero ignorando boosts
                            score_base = min(0.97, (sim_cal * self.w_semantico
                                            + score_kw * self.w_keywords
                                            + (score_nuclear / 100) * self.w_nucleares * 100
                                            + score_ref * self.w_referencia))
                            score = score_base * factor_bloq
                        break

            candidatos.append({
                "indice":         i,
                "score":          score,
                "sim_raw":        sim_raw,
                "sim_cal":        sim_cal,
                "score_kw":       score_kw,
                "score_nuclear":  score_nuclear,
                "score_ref":      score_ref,
                "boost_sect":     boost_sect,
                "ultra_title":    ultra_title,
                "ultra_desc":     ultra_desc,
                "precision_mult": precision_mult,
                "boost_marco":    boost_marco,
            })

        candidatos.sort(key=lambda x: x["score"], reverse=True)

        alineaciones = []
        for c in candidatos[: top_n * 4]:
            if c["score"] < umbral_minimo:
                continue
            if len(alineaciones) >= top_n:
                break

            idx  = c["indice"]
            fila = df.iloc[idx]
            alineaciones.append({
                "marco":               marco,
                "titulo":              str(fila.get("titulo",      "")),
                "descripcion":         str(fila.get("descripcion", "")),
                "score":               round(c["score"] * 100, 2),
                "similitud":           round(c["score"], 4),
                "score_semantico":     round(c["sim_cal"]      * 100, 1),
                "score_semantico_raw": round(c["sim_raw"]      * 100, 1),
                "score_keywords":      round(c["score_kw"]     * 100, 1),
                "score_nuclear":       round(c["score_nuclear"] * 100, 1),
                "score_referencia":    round(c["score_ref"]     * 100, 1),
                "score_sectorial":     round(c["boost_sect"]    * 100, 1),
                "nivel_confianza":     _clasificar_confianza(c["score"]),
                "id_original":         fila.get("id_original", idx + 1),
                "metadata":            _extraer_metadata(fila),
                "justificacion":       _generar_justificacion(c, self.modo),
                "_precision_mult":     round(c["precision_mult"], 4),
                "_boost_marco":        round(c["boost_marco"],     4),
            })

        return alineaciones

    def _preparar_reglas_ods(self, ultra_proyecto, suaves_proyecto):
        boosts_activos   = []
        bloqueos_activos = set()
        for ultra_set, lista_boosts in ODS_BOOST_ESPECIFICO.items():
            if ultra_set & (ultra_proyecto | suaves_proyecto):
                boosts_activos.extend(lista_boosts)
        for ultra_set, set_frags in ODS_BLOQUEOS_TEMATICOS.items():
            if ultra_set & ultra_proyecto:
                bloqueos_activos |= set_frags
        boosts_activos.sort(key=lambda x: x[1], reverse=True)
        return boosts_activos, bloqueos_activos

    def _preparar_penalizaciones_marco(self, marco, ultra_proyecto):
        penalizaciones = []
        for ultra_set, lista_pens in MARCO_PENALIZACIONES_TEMATICAS.items():
            if ultra_set & ultra_proyecto:
                for marco_afectado, substr_titulo, factor in lista_pens:
                    if marco_afectado == marco:
                        penalizaciones.append((substr_titulo, factor))
        return penalizaciones

    def exportar_resultados(self, resultados: dict, info_proyecto: dict) -> pd.DataFrame:
        registros = []
        for marco, alineaciones in resultados.items():
            if marco.startswith("_"):
                continue
            for alin in alineaciones:
                registros.append({
                    "Proyecto":              info_proyecto.get("nombre",      ""),
                    "Objetivo_Proyecto":     info_proyecto.get("objetivo",    ""),
                    "Descripción_Proyecto":  info_proyecto.get("descripcion", ""),
                    "Fecha_Análisis":        info_proyecto.get("fecha",       ""),
                    "Marco":                 marco,
                    "ID_Meta":               alin.get("id_original", ""),
                    "Título_Meta":           alin["titulo"],
                    "Descripción_Meta":      alin["descripcion"],
                    "Score_%":               round(alin["score"], 2),
                    "Nivel_Confianza":       alin["nivel_confianza"],
                    "Score_Semántico_%":     alin["score_semantico"],
                    "Score_Semántico_RAW_%": alin.get("score_semantico_raw", 0),
                    "Score_Nuclear_%":       alin.get("score_nuclear", 0),
                    "Score_Keywords_%":      alin["score_keywords"],
                    "Score_Sectorial_%":     alin.get("score_sectorial", 0),
                    "Score_Histórico_%":     alin.get("score_referencia", 0),
                    "Razón_IA":               alin.get("razon_ia", ""),
                    "Fuente":                 alin.get("fuente", "Motor"),
                    "Multiplicador_Precisión": alin.get("_precision_mult", 1.0),
                    "Boost_Marco":             alin.get("_boost_marco",     1.0),
                    "Score_Verificación_%":  round(
                        min(97,
                            alin.get("score_semantico", alin.get("score", 0)) * self.w_semantico
                            + alin.get("score_nuclear",   0) / 100 * self.w_nucleares * 100
                            + alin.get("score_keywords",  0) * self.w_keywords
                            + alin.get("score_referencia",0) * self.w_referencia
                        ) * alin.get("_precision_mult", 1.0)
                        * alin.get("_boost_marco", 1.0),
                        2
                    ),
                    "Justificación": alin.get("justificacion", alin.get("razon_ia", "")),
                    "Metadata":      alin.get("metadata", ""),
                })
        df = pd.DataFrame(registros)
        if not df.empty:
            df = df.sort_values("Score_%", ascending=False).reset_index(drop=True)
        return df

    def obtener_estadisticas(self, resultados: dict) -> dict:
        stats = {
            "total_alineaciones": 0,
            "promedio_score":     0.0,
            "max_score":          0.0,
            "min_score":          100.0,
            "por_marco":          {},
        }
        todos_scores = []
        for marco, alineaciones in resultados.items():
            if marco.startswith("_") or not alineaciones:
                continue
            scores = [a["score"] for a in alineaciones]
            stats["por_marco"][marco] = {
                "cantidad": len(alineaciones),
                "promedio": round(np.mean(scores), 1),
                "max":      round(max(scores),     1),
                "min":      round(min(scores),     1),
            }
            todos_scores.extend(scores)
        if todos_scores:
            stats["total_alineaciones"] = len(todos_scores)
            stats["promedio_score"]     = round(np.mean(todos_scores), 1)
            stats["max_score"]          = round(max(todos_scores),     1)
            stats["min_score"]          = round(min(todos_scores),     1)
        return stats


# ══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE APOYO
# ══════════════════════════════════════════════════════════════════════════

def _clasificar_confianza(score: float) -> str:
    if   score >= UMBRAL_EXCELENTE: return "🟢 Excelente"
    elif score >= UMBRAL_BUENO:     return "🟡 Buena"
    elif score >= UMBRAL_MODERADO:  return "🟠 Moderada"
    else:                           return "🔴 Débil"


def _extraer_metadata(fila) -> str:
    excluir = {"titulo", "descripcion", "texto_completo", "id_original"}
    partes  = []
    for col in fila.index:
        if col in excluir:
            continue
        valor = str(fila.get(col, "")).strip()
        if valor and valor.lower() not in {"nan", "no aplica", "", "none"}:
            partes.append(f"{col.upper()}: {valor}")
    return " | ".join(partes[:5])


def _generar_justificacion(c: dict, modo: str) -> str:
    partes  = []
    sem_cal = c["sim_cal"]      * 100
    sem_raw = c["sim_raw"]      * 100
    kw      = c["score_kw"]     * 100
    nuclear = c.get("score_nuclear", 0) * 100
    ref     = c["score_ref"]    * 100
    sect    = c.get("boost_sect", 0) * 100
    u_title = c.get("ultra_title",    0)
    p_mult  = c.get("precision_mult", 1.0)

    if sem_cal >= 70:
        partes.append(f"alta similitud semántica ({sem_cal:.0f}% / coseno {sem_raw:.0f}%)")
    elif sem_cal >= 50:
        partes.append(f"similitud semántica moderada-alta ({sem_cal:.0f}%)")
    elif sem_cal >= 35:
        partes.append(f"similitud semántica moderada ({sem_cal:.0f}%)")

    if nuclear >= 50:
        partes.append(f"fuerte coherencia temática nuclear ({nuclear:.0f}%)")
    elif nuclear >= 30:
        partes.append(f"alineación con objetivos estratégicos nucleares ({nuclear:.0f}%)")

    if kw >= 40:
        partes.append(f"vocabulario técnico especializado ({kw:.0f}%)")
    elif kw >= 20:
        partes.append(f"términos técnicos del dominio ({kw:.0f}%)")

    if sect >= 5:
        partes.append(f"refuerzo perfil sectorial (+{sect:.0f}%)")

    if u_title >= 1 and p_mult >= 1.10:
        partes.append("términos técnicos específicos coinciden en el título del marco")

    if p_mult > 1.12:
        partes.append(f"boost temático aplicado (×{p_mult:.2f})")
    elif p_mult < 0.85:
        partes.append(f"penalización por dominio ajeno (×{p_mult:.2f})")

    if ref >= 25 and modo == "con_referencia":
        partes.append(f"respaldado por precedentes históricos ({ref:.0f}%)")

    return (
        "Alineación justificada por: " + ", ".join(partes) + "."
        if partes else
        "Alineación basada en análisis semántico del marco."
    )
