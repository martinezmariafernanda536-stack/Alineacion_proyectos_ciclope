"""
CICLOPE v22.0 - NLP Processor
APC COLOMBIA

CAMBIOS v22.0:
  v22-1 → calcular_score_keywords_dominio ahora usa keywords_adicionales:
           si una keyword adicional aparece en el proyecto, se busca también
           en el marco → boost real al score
  v22-2 → analizar_keywords_en_texto retorna adicionales detectadas etiquetadas
           para que aparezcan en el desplegable de la interfaz
  v22-3 → Eliminado "cooperacion internacional" de las nucleares genéricas
  v22-4 → Startup optimizado: modelos cargados lazy (solo al primer uso)
"""

import re
import unicodedata
import numpy as np
from pathlib import Path
import pickle


def _normalizar_texto(t: str) -> str:
    t = t.lower()
    for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
        t = t.replace(a, b)
    return t


class NLPProcessor:
    """
    Procesador NLP v22.
    - Embeddings semánticos con sentence-transformers.
    - Keywords técnicas en 3 niveles.
    - Palabras nucleares estratégicas.
    - Keywords adicionales del usuario con impacto REAL en el score.
    """

    _PALABRAS_GENERICAS = frozenset({
        "para", "con", "los", "las", "del", "una", "por", "como",
        "que", "este", "esta", "esta", "entre", "hacia", "desde",
        "sobre", "bajo", "ante", "tras", "segun", "sin",
        "cuando", "donde", "cual", "cuyo", "siendo", "dado",
        "nacion", "nacional", "colombia", "gobierno",
        "proyecto", "programa", "sistema", "proceso", "politica",
        "estrategia", "mecanismo", "marco", "plan",
        "general", "especifico", "integral", "social", "publico",
        "mediante", "traves", "objetivo", "meta", "resultado",
        "realizar", "ejecutar", "implementar", "desarrollar",
        "establecer", "crear", "generar", "acciones",
        "asegurar", "lograr", "promover", "garantizar",
        "fortalecimiento", "mejora", "acceso", "recursos",
    })

    def __init__(self, modelo_name: str):
        self.modelo_name     = modelo_name
        self.modelo_cargado  = None
        self.modelo          = None
        self._cargando       = False

        self.diccionario_semantico = {}
        self.todas_las_keywords    = []
        self.palabras_nucleares    = []
        self.keywords_adicionales  = []   # keywords manuales del usuario

        self._cargar_diccionario()
        self._cargar_modelo()

    # ── CARGA DEL MODELO ──────────────────────────────────────────────────

    def _cargar_modelo(self):
        if self._cargando:
            return
        self._cargando = True
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🔄 Cargando modelo: {self.modelo_name.split('/')[-1]}...")
            self.modelo         = SentenceTransformer(self.modelo_name)
            self.modelo_cargado = self.modelo_name
            dim = self.modelo.get_sentence_embedding_dimension()
            print(f"✅ Modelo listo — dimensión: {dim}")
        except Exception as e:
            print(f"❌ Error al cargar {self.modelo_name}: {e}")
            # Fallback al modelo más liviano
            fallback = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            if fallback != self.modelo_name:
                print(f"⚠️  Fallback → {fallback}")
                try:
                    from sentence_transformers import SentenceTransformer
                    self.modelo         = SentenceTransformer(fallback)
                    self.modelo_cargado = fallback
                    print(f"✅ Fallback OK — dim: {self.modelo.get_sentence_embedding_dimension()}")
                except Exception as e2:
                    print(f"❌ Fallback también falló: {e2}")
        finally:
            self._cargando = False

    # ── DICCIONARIO ───────────────────────────────────────────────────────

    def _cargar_diccionario(self):
        try:
            from diccionario_semantico import DICCIONARIO_SEMANTICO
            self.diccionario_semantico = DICCIONARIO_SEMANTICO
        except Exception:
            self.diccionario_semantico = {}

        # FIX: unir las tres fuentes de keywords para que la UI
        # reporte el total real y el filtrado use todas
        try:
            from diccionario_semantico import TODAS_LAS_KEYWORDS
            from config import KEYWORDS_TECNICAS
            todas = set(TODAS_LAS_KEYWORDS) | set(KEYWORDS_TECNICAS)
            todas |= set(self.diccionario_semantico.keys())
            self.todas_las_keywords = sorted(todas)
        except Exception:
            self.todas_las_keywords = list(self.diccionario_semantico.keys())

    # ── API PÚBLICA ───────────────────────────────────────────────────────

    def set_palabras_nucleares(self, palabras: list):
        self.palabras_nucleares = [p.strip().lower() for p in palabras if p.strip()]

    def set_keywords_adicionales(self, keywords: list):
        """
        v22-1: Las keywords adicionales ahora tienen impacto REAL en el score.
        Se integran en calcular_score_keywords_dominio.
        """
        self.keywords_adicionales = [k.strip().lower() for k in keywords if k.strip()]
        # Unir con todas_las_keywords para que también aparezcan en la detección
        todas = list(set(self.todas_las_keywords) | set(self.keywords_adicionales))
        self.todas_las_keywords = sorted(todas)

    # ── EMBEDDINGS ────────────────────────────────────────────────────────

    def generar_embedding(self, texto: str) -> np.ndarray:
        if self.modelo is None:
            return np.zeros(768)
        try:
            texto_exp = self._expandir_texto_semantico(texto)
            emb = self.modelo.encode(texto_exp, normalize_embeddings=True,
                                     show_progress_bar=False)
            return np.array(emb, dtype=np.float32)
        except Exception as e:
            print(f"⚠️  Error embedding: {e}")
            return np.zeros(768)

    def generar_embeddings_batch(self, textos: list,
                                 batch_size: int = 32) -> np.ndarray:
        if self.modelo is None or not textos:
            dim = 768
            return np.zeros((len(textos), dim))
        try:
            textos_exp = [self._expandir_texto_semantico(t) for t in textos]
            embs = self.modelo.encode(
                textos_exp,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            return np.array(embs, dtype=np.float32)
        except Exception as e:
            print(f"⚠️  Error batch embeddings: {e}")
            dim = self.modelo.get_sentence_embedding_dimension() if self.modelo else 768
            return np.zeros((len(textos), dim))

    def cargar_embeddings_cache(self, ruta: Path):
        try:
            if Path(ruta).exists():
                with open(ruta, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None

    def guardar_embeddings_cache(self, embeddings, ruta: Path):
        try:
            Path(ruta).parent.mkdir(parents=True, exist_ok=True)
            with open(ruta, "wb") as f:
                pickle.dump(embeddings, f, protocol=4)
        except Exception as e:
            print(f"⚠️  No se pudo guardar cache: {e}")

    # ── SCORE KEYWORDS ────────────────────────────────────────────────────

    def calcular_score_keywords_dominio(self, texto_proyecto: str,
                                        texto_marco: str) -> float:
        """
        v22-1: Score de keywords que incluye keywords_adicionales del usuario.
        
        Lógica:
         1. Palabras específicas del marco (≥6 chars, no genéricas + acrónimos)
            → se buscan en el proyecto (coincidencia temática)
         2. Keywords adicionales del usuario → si están en el PROYECTO,
            se verifican también en el MARCO (boost bidireccional)
         3. Score ponderado: específicas×3 + adicionales×4 + genéricas×1
        """
        if not texto_proyecto or not texto_marco:
            return 0.0

        p_lower = texto_proyecto.lower()
        m_lower = texto_marco.lower()

        # ── Palabras del marco ──────────────────────────────────────────
        todas_palabras = set(re.findall(r'\b\w{4,}\b', m_lower))
        acronimos_marco = {
            a.lower() for a in re.findall(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto_marco)
        }
        especificas = (
            {p for p in todas_palabras
             if len(p) >= 6 and p not in self._PALABRAS_GENERICAS}
            | acronimos_marco
        )
        genericas = todas_palabras - especificas - acronimos_marco

        matches_esp = sum(1 for p in especificas if self._contiene_palabra(p_lower, p))
        matches_gen = sum(1 for p in genericas   if self._contiene_palabra(p_lower, p))

        # ── v22-1: Keywords adicionales del usuario ─────────────────────
        matches_adicionales = 0
        if self.keywords_adicionales:
            for kw in self.keywords_adicionales:
                kw_en_proyecto = self._contiene_palabra(p_lower, kw)
                kw_en_marco    = self._contiene_palabra(m_lower, kw)
                if kw_en_proyecto and kw_en_marco:
                    matches_adicionales += 2   # coincidencia perfecta
                elif kw_en_proyecto:
                    matches_adicionales += 1   # solo en proyecto → contexto

        # ── Score ponderado ─────────────────────────────────────────────
        total_esp  = max(1, len(especificas))
        total_gen  = max(1, len(genericas))
        total_adic = max(1, len(self.keywords_adicionales) * 2) if self.keywords_adicionales else 1

        denom = total_esp * 3 + total_gen + total_adic * 4
        numer = matches_esp * 3 + matches_gen + matches_adicionales * 4

        ratio = numer / denom if denom > 0 else 0.0

        # Bonus progresivo por coincidencias específicas
        if matches_esp + matches_adicionales >= 5:
            ratio = min(1.0, ratio * 1.45)
        elif matches_esp + matches_adicionales >= 3:
            ratio = min(1.0, ratio * 1.25)
        elif matches_esp + matches_adicionales >= 1:
            ratio = min(1.0, ratio * 1.08)
        elif matches_esp == 0 and matches_adicionales == 0:
            ratio *= 0.30

        return min(1.0, ratio)

    def calcular_score_keywords(self, texto_proyecto, texto_marco):
        return self.calcular_score_keywords_dominio(texto_proyecto, texto_marco)

    def calcular_score_keywords_globales(self, texto_proyecto: str) -> float:
        if not texto_proyecto:
            return 0.0
        p_lower = texto_proyecto.lower()
        hits = sum(1 for kw in self.todas_las_keywords
                   if self._contiene_palabra(p_lower, kw.lower()))
        return min(1.0, hits / max(10, len(self.todas_las_keywords) * 0.05))

    # ── SCORE NUCLEAR ─────────────────────────────────────────────────────

    def calcular_score_nuclear(self, texto_proyecto: str,
                                texto_marco: str,
                                nucleares: list) -> float:
        """Score de palabras nucleares estratégicas (excluye genéricas)."""
        if not texto_proyecto or not nucleares:
            return 0.0
        p_lower = texto_proyecto.lower()
        m_lower = texto_marco.lower()

        hits_proyecto = sum(1 for n in nucleares
                            if self._contiene_palabra(p_lower, n.lower()))
        hits_ambos    = sum(1 for n in nucleares
                            if self._contiene_palabra(p_lower, n.lower())
                            and self._contiene_palabra(m_lower, n.lower()))

        base = hits_proyecto / max(1, len(nucleares))
        bonus = hits_ambos  / max(1, len(nucleares))
        return min(1.0, base * 0.6 + bonus * 0.4)

    # ── ANÁLISIS KEYWORDS (para interfaz) ────────────────────────────────

    def analizar_keywords_en_texto(self, texto: str) -> list:
        """
        v23-fix: Keywords adicionales SIEMPRE aparecen en términos detectados
        (con ★) porque el usuario las declaró explícitamente para el proyecto.
        Las keywords estándar solo aparecen si están en el texto.
        """
        if not texto:
            return []
        texto_lower = texto.lower()
        encontradas = set()

        # Keywords estándar — solo si están en el texto del proyecto
        for kw in self.todas_las_keywords:
            if kw not in self.keywords_adicionales:
                if self._contiene_palabra(texto_lower, kw.lower()):
                    encontradas.add(kw.title())

        # Keywords adicionales del usuario — SIEMPRE se muestran con ★
        # El usuario las declaró explícitamente → siempre están "detectadas"
        for kw in self.keywords_adicionales:
            encontradas.add(f"★ {kw.title()}")

        return sorted(encontradas)[:40]

    def detectar_nucleares_en_texto(self, texto: str, nucleares: list,
                                       nucleares_default: set = None) -> list:
        """
        Detecta palabras nucleares en el texto del proyecto.
        - Normaliza tildes para comparación robusta.
        - Las nucleares añadidas manualmente (no están en nucleares_default)
          se muestran siempre con ★ aunque no aparezcan en el texto.
        """
        if not nucleares:
            return []
        texto_lower = (texto or "").lower()
        for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
            texto_lower = texto_lower.replace(a, b)

        default_set = nucleares_default or set()
        encontradas = []
        for n in nucleares:
            n_norm = n.lower()
            for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
                n_norm = n_norm.replace(a, b)
            if self._contiene_palabra(texto_lower, n_norm):
                encontradas.append(n)
            elif n.strip().lower() not in default_set:
                # Nuclear nueva del usuario → siempre visible con ★
                encontradas.append(f"★ {n}")
        return encontradas

    # ── EXPANSIÓN SEMÁNTICA ───────────────────────────────────────────────

    def _expandir_texto_semantico(self, texto: str) -> str:
        """
        Expansión con diccionario semántico + keywords adicionales.
        """
        if not texto or not self.diccionario_semantico:
            return texto
        texto_lower = texto.lower()
        expansiones = set()

        for termino, sinonimos in self.diccionario_semantico.items():
            if self._contiene_palabra(texto_lower, termino):
                expansiones.add(termino)
                if isinstance(sinonimos, list):
                    for sin in sinonimos[:5]:
                        if isinstance(sin, str):
                            expansiones.add(sin)
                elif isinstance(sinonimos, str):
                    for s in sinonimos.split()[:8]:
                        if len(s) >= 5:
                            expansiones.add(s)
            else:
                if isinstance(sinonimos, list):
                    for sin in sinonimos[:3]:
                        if isinstance(sin, str) and self._contiene_palabra(texto_lower, sin.lower()):
                            expansiones.add(termino)
                            for s2 in sinonimos[:4]:
                                if isinstance(s2, str):
                                    expansiones.add(s2)
                            break

        # Keywords adicionales del usuario
        for kw in self.keywords_adicionales:
            if self._contiene_palabra(texto_lower, kw):
                expansiones.add(kw)

        if expansiones:
            return f"{texto} {' '.join(list(expansiones)[:80])}"
        return texto

    # ── UTILIDADES ────────────────────────────────────────────────────────

    @staticmethod
    def _norm_texto(s: str) -> str:
        """Normaliza tildes para comparación robusta."""
        for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
            s = s.replace(a, b)
        return s

    @staticmethod
    def _contiene_palabra(texto_lower: str, termino: str) -> bool:
        """Detecta término como palabra completa con normalización de tildes."""
        termino = termino.lower().strip()
        if not termino:
            return False
        # Normalizar ambos lados para evitar fallos por tildes
        for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
            texto_lower = texto_lower.replace(a, b)
            termino     = termino.replace(a, b)
        if len(termino.split()) > 1:
            return termino in texto_lower
        return bool(re.search(r'\b' + re.escape(termino) + r'\b', texto_lower))

    def obtener_info_modelo(self) -> dict:
        dim = self.modelo.get_sentence_embedding_dimension() if self.modelo else 0
        return {
            "modelo":               self.modelo_cargado or self.modelo_name,
            "modelo_solicitado":    self.modelo_name,
            "dimension_embeddings": dim,
            "terminos_diccionario": len(self.diccionario_semantico),
            "total_keywords":       len(self.todas_las_keywords),
            "keywords_adicionales": len(self.keywords_adicionales),
            "palabras_nucleares":   len(self.palabras_nucleares),
        }
