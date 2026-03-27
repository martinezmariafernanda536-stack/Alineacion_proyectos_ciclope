"""
CICLOPE v23.0 - CONFIGURACION
APC COLOMBIA | Sistema de Alineacion Estrategica

CAMBIOS v22.0:
  v20-1 → Solo modelos 768-dim validados: MPNet, LaBSE, MiniLM (sin mezclas)
  v20-2 → Cache incluye nombre+dim del modelo → invalida automaticamente al cambiar
  v20-3 → Semantica reforzada: pesos 65/20/10/5
  v20-4 → Diccionario semantico ampliado: 200+ terminos nuevos por sector
  v20-5 → Startup optimizado: sector profiles lazy, cache agresivo
"""

import torch


def _seleccionar_modelo() -> tuple:
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 8:
            return "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 16
    return "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 8


MODELO_AUTO, BATCH_AUTO = _seleccionar_modelo()

# ══════════════════════════════════════════════════════════════
# MODELOS — solo 768-dim validados (sin mezclas dimensionales)
# ══════════════════════════════════════════════════════════════

MODELOS_DISPONIBLES = {
    "MPNet Multilingual (recomendado)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "MiniLM Multilingual (ligero)":     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "LaBSE (bilingüe robusto)":         "sentence-transformers/LaBSE",
}

DIMENSION_MODELOS = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/LaBSE":                                  768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}

# ══════════════════════════════════════════════════════════════
# PESOS v20 — semantica reforzada (65%)
# ══════════════════════════════════════════════════════════════

W_SEMANTICO_SIN_REF  = 0.80
W_NUCLEARES_SIN_REF  = 0.10
W_KEYWORDS_SIN_REF   = 0.10
W_REFERENCIA_SIN_REF = 0.00

W_SEMANTICO_CON_REF  = 0.60
W_NUCLEARES_CON_REF  = 0.18
W_KEYWORDS_CON_REF   = 0.17
W_REFERENCIA_CON_REF = 0.05

# ══════════════════════════════════════════════════════════════
# PERFILES SECTORIALES
# ══════════════════════════════════════════════════════════════

SECTOR_PROFILES = {
    "salud": (
        "VIH sida hiv aids antirretroviral tarv tratamiento prevencion pruebas diagnostico "
        "salud publica epidemia poblaciones clave migrantes vulnerables carga viral "
        "cd4 prep pep hsh trabajo sexual personas transgenero hepatitis tuberculosis "
        "atencion medica cobertura sanitaria sistemas de salud vacunacion "
        "prueba rapida de vih prevencion del vih respuesta nacional al vih "
        "salud sexual reproductiva planificacion familiar anticoncepcion "
        "enfermedades tropicales malaria dengue leishmaniasis chagas "
        "nutricion desnutricion seguridad alimentaria salud mental psicosocial"
    ),
    "defensa_paz": (
        "desminado humanitario minas antipersonal aicma accion contra minas "
        "artefactos explosivos liberacion de tierras equipos multitarea "
        "victimas de minas posconflicto acuerdo de paz reincorporacion "
        "excombatientes pdet zonas de consolidacion seguridad territorial "
        "despeje estudio no tecnico educacion riesgo minas "
        "paz total firma acuerdo transicion justicia "
        "reintegracion social excombatientes comunidades conflicto armado"
    ),
    "agricultura": (
        "agricultura familiar agroemprendedor campesinado economia campesina "
        "reforma agraria seguridad alimentaria pequenos productores "
        "agricultura climaticamente inteligente cadenas de valor agropecuarias "
        "asistencia tecnica rural productividad agropecuaria agroecologia "
        "riego sistemas produccion cosecha semillas fertilizantes "
        "mercados campesinos comercializacion produccion organica "
        "ganaderia pesca acuicultura apicultura caficultura cacao"
    ),
    "ambiente": (
        "cambio climatico adaptacion mitigacion biodiversidad deforestacion "
        "ecosistemas economia circular reciclaje plasticos residuos "
        "soluciones basadas en la naturaleza paisajes forestales "
        "carbono neutralidad emisiones sostenibilidad ambiental "
        "agroemprendedor andino amazonico cuencas hidrograficas "
        "bosques humedales paramos arrecifes corales manglares "
        "energia renovable solar eolica hidraulica eficiencia energetica"
    ),
    "educacion": (
        "educacion formacion docente aprendizaje escolar primera infancia "
        "inclusion educativa desercion escolar acceso a la educacion "
        "calidad educativa competencias lectoescritura pedagogia "
        "educacion superior universitaria vocacional tecnica "
        "becas programas educativos interculturalidad bilingue "
        "habilidades digitales stem ciencia tecnologia innovacion"
    ),
    "genero": (
        "igualdad de genero empoderamiento de la mujer feminicidio vbg "
        "violencia basada en genero lgbti mujeres rurales brecha de genero "
        "autonomia economica mujer agenda de paz y mujer "
        "derechos sexuales reproductivos violencia sexual intrafamiliar "
        "paridad politica liderazgo femenino mujeres indigenas afro"
    ),
    "agua_saneamiento": (
        "acueducto alcantarillado agua potable saneamiento basico "
        "planta de tratamiento suministro de agua comunidades rurales "
        "infraestructura hidrica agua segura potabilizacion "
        "lavado de manos higiene wash sistemas de riego "
        "cuencas hidrograficas gestion del agua recurso hidrico"
    ),
    "cooperacion": (
        "cooperacion internacional cooperacion sur-sur cooperacion triangular "
        "ayuda oficial al desarrollo apc colombia donantes receptores "
        "asistencia tecnica internacional alianzas estrategicas "
        "movilizacion de recursos financiamiento externo cooperacion descentralizada "
        "coherencia politicas desarrollo agenda 2030 ods objetivos desarrollo sostenible "
        "partenariado asociaciones publico privadas snci sistema nacional cooperacion "
        "cooperacion bilateral multilateral cooperacion regional "
        "transferencia de conocimiento intercambio de experiencias buenas practicas "
        "capacidades institucionales fortalecimiento institucional "
        "gestion de la cooperacion planificacion seguimiento evaluacion "
        "eficacia de la ayuda apropiacion alineacion armonizacion "
        "cooperacion tecnica no reembolsable donacion aporte "
        "agencia de cooperacion organizacion internacional naciones unidas bid bm "
        "union europea usaid giz jica koica aecid acdi "
        "marco de asociacion pais estrategia cooperacion rendicion de cuentas"
    ),
    # v27: Nuevos perfiles sectoriales
    "justicia": (
        "acceso a la justicia sistema judicial defensa publica "
        "proteccion a la infancia trabajo infantil trata de personas "
        "explotacion sexual comercial menores derechos de la ninez "
        "reparacion de victimas victimas del conflicto armado "
        "violencia intrafamiliar violencia basada en genero feminicidio "
        "medidas de proteccion ley de victimas restitución de tierras "
        "ICBF Bienestar Familiar proteccion derechos humanos "
        "sistema penitenciario carcelario rehabilitacion resocializacion"
    ),
    "cancilleria": (
        "relaciones exteriores politica exterior diplomacia "
        "cooperacion internacional acuerdos bilaterales multilaterales "
        "venezolanos migrantes refugiados regularizacion migratoria "
        "consulados embajadas cancilleria colombia "
        "tratados internacionales organizaciones internacionales "
        "cooperacion sur-sur cooperacion triangular"
    ),
    "prosperidad_social": (
        "superacion de pobreza extrema transferencias monetarias "
        "familias en accion jovenes en accion ingreso solidario "
        "asistencia humanitaria desplazados victimas "
        "inclusion social poblaciones vulnerables "
        "programas sociales subsidios atencion integral "
        "comunidades vulnerables en riesgo de pobreza"
    ),
    "vicepresidencia": (
        "accion integral contra minas antipersonal aicma "
        "coordinacion desminado humanitario educacion riesgo minas "
        "atencion victimas minas antipersonal reintegracion "
        "territorios liberados de minas paz territorial "
        "estrategia integral contra minas estándares desminado"
    ),
    # v30: Nuevos perfiles sectoriales
    "trabajo": (
        "empleo trabajo decente desempleo generacion de empleo ingresos "
        "formacion para el trabajo capacitacion laboral empleabilidad "
        "mercado laboral derechos laborales sindicatos negociacion colectiva "
        "economia popular solidaria emprendimiento pymes mipymes "
        "jovenes neet inclusion laboral poblaciones vulnerables "
        "migrantes venezolanos fuerza laboral productividad "
        "trabajo informal formalizacion laboral proteccion social"
    ),
    "comercio": (
        "comercio exterior exportaciones importaciones aranceles "
        "cadenas globales de valor integracion economica "
        "facilitacion del comercio aduanas barreras no arancelarias "
        "acuerdos comerciales tratados libre comercio tlc "
        "competitividad empresarial pymes exportadoras "
        "proexport procolombia marca colombia "
        "comercio justo economia solidaria mercados internacionales"
    ),
    "gobernanza": (
        "gobernanza transparencia rendicion de cuentas anticorrupcion "
        "fortalecimiento institucional capacidades del estado "
        "participacion ciudadana democracia descentralizacion "
        "servicios publicos gestion publica reforma del estado "
        "estado de derecho acceso a la informacion "
        "planificacion publica presupuesto por resultados "
        "gobierno abierto datos abiertos e-gobierno gobierno digital "
        "veeduria ciudadana control social organismos control"
    ),
    "cultura": (
        "cultura patrimonio cultural diversidad cultural identidad "
        "industrias culturales creativas artes musica cine "
        "turismo cultural sostenible artesanias "
        "comunidades indigenas afrodescendientes etnias "
        "lenguas nativas interculturalidad salvaguardia "
        "diplomacia cultural intercambio cultural "
        "derechos culturales memoria historica"
    ),
    "ciencia_tecnologia": (
        "ciencia tecnologia innovacion investigacion desarrollo "
        "transferencia tecnologica i+d sistemas de innovacion "
        "universidades centros investigacion spin-off startups "
        "propiedad intelectual patentes conocimiento "
        "transformacion digital inteligencia artificial "
        "habilidades digitales brecha digital conectividad "
        "stem vocaciones cientificas becas investigacion "
        "colciencias minciencias ecosistema innovacion"
    ),
}

SECTOR_PROFILE_BOOST     = 0.10
SECTOR_PROFILE_THRESHOLD = 0.50

# ══════════════════════════════════════════════════════════════
# PALABRAS NUCLEARES — ampliadas
# ══════════════════════════════════════════════════════════════

PALABRAS_NUCLEARES_DEFAULT = [
    # ── COOPERACION INTERNACIONAL ──
    "desarrollo sostenible",
    "impacto social", "transformacion territorial",
    "agenda 2030", "financiamiento externo",
    # ── PAZ Y POSCONFLICTO ──
    "construccion de paz", "posconflicto", "reintegracion",
    "acuerdo de paz", "consolidacion territorial",
    "excombatientes", "firmantes de paz", "reincorporacion integral",
    "pdet", "reforma rural integral", "sustitucion de cultivos",
    "justicia transicional", "victimas conflicto",
    # ── POBLACIONES VULNERABLES ──
    "poblacion vulnerable", "comunidades rurales",
    "territorios priorizados", "inclusion social", "derechos humanos",
    "migrantes venezolanos", "refugiados", "desplazados",
    "indigenas", "afrodescendientes", "comunidades negras",
    "personas con discapacidad", "adultos mayores",
    # ── SALUD PUBLICA ──
    "vih", "sida", "hiv", "aids", "salud publica",
    "poblaciones clave", "prevencion vih", "tratamiento vih",
    "antirretroviral", "tarv", "arv", "carga viral", "cd4",
    "prueba rapida vih", "salud sexual", "migrantes vulnerables",
    "tuberculosis", "hepatitis", "malaria", "its", "ets",
    "prep", "pep", "pvvih", "hsh", "trabajo sexual",
    "salud mental", "nutricion", "mortalidad materna",
    # ── AMBIENTE ──
    "cambio climatico", "sostenibilidad ambiental", "economia circular",
    "biodiversidad", "adaptacion climatica", "deforestacion",
    "ecosistemas", "energia renovable", "transicion energetica",
    "soluciones basadas en la naturaleza", "paramos", "bosques",
    # ── GENERO ──
    "igualdad de genero", "empoderamiento de la mujer",
    "violencia basada en genero", "feminicidio",
    "lgbti", "personas transgenero", "diversidad sexual",
    "mujeres rurales", "autonomia economica mujer",
    # ── ECONOMIA Y AGRICULTURA ──
    "competitividad", "desarrollo economico",
    "cadenas de valor", "seguridad alimentaria", "agricultura familiar",
    "pequenos productores", "reforma agraria",
    "emprendimiento", "pymes", "economia popular",
    # ── DESMINADO ──
    "desminado humanitario", "accion contra minas", "minas antipersonal",
    "liberacion de tierras", "victimas de minas", "aicma",
    "artefactos explosivos", "municiones sin explotar",
    # ── AGUA Y SANEAMIENTO ──
    "agua potable", "saneamiento basico", "acueducto", "alcantarillado",
    "wash", "acceso agua segura",
    # ── EDUCACION ──
    "educacion", "primera infancia", "inclusion educativa",
    "calidad educativa", "formacion tecnica", "acceso educacion",
    # ── GOBERNANZA ──
    "fortalecimiento institucional", "gobernanza",
    "participacion ciudadana", "transparencia",
    "politica publica", "capacidades locales",
]

NUCLEAR_BOOST_FACTOR   = 1.45
NUCLEAR_PENALTY_FACTOR = 0.85
NUCLEAR_MIN_MATCHES    = 1

# ══════════════════════════════════════════════════════════════
# PMI
# ══════════════════════════════════════════════════════════════

PMI_ACTIVO_DEFAULT           = True
PMI_APLICA_SOLO_POSCONFLICTO = False
PMI_BOOST_UNIVERSAL          = 1.12

# ══════════════════════════════════════════════════════════════
# CALIBRACION Y UMBRALES
# ══════════════════════════════════════════════════════════════

SIM_MIN_CALIBRACION       = 0.18
SIM_MAX_CALIBRACION       = 0.95
UMBRAL_SEMANTICO_MINIMO   = 0.18
UMBRAL_SEMANTICO_SECTORES = 0.28
UMBRAL_LONGITUD_ACRONIMO  = 3

UMBRAL_EXCELENTE = 0.68
UMBRAL_BUENO     = 0.50
UMBRAL_MODERADO  = 0.33

# ══════════════════════════════════════════════════════════════
# BOOST POR MARCO
# ══════════════════════════════════════════════════════════════

BOOST_MARCOS = {
    "PMI":      PMI_BOOST_UNIVERSAL,
    "ENCI":     1.20,
    "PND":      1.08,
    "ODS":      1.05,
    "CAD":      1.07,
    "SECTORES": 1.02,
}

# ══════════════════════════════════════════════════════════════
# ODS REGLAS
# ══════════════════════════════════════════════════════════════

ODS_BOOST_ESPECIFICO = {
    frozenset({"vih", "sida", "hiv", "aids", "arv", "tarv", "pvvih"}): [
        ("3.3", 1.55), ("vih", 1.50), ("sida", 1.50), ("sexual", 1.18), ("3.", 1.12),
    ],
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal",
               "artefactos explosivos", "municiones sin explotar"}): [
        ("16.1", 1.38), ("16.2", 1.32), ("16.a", 1.22), ("3.2", 1.15), ("16.", 1.20),
    ],
    frozenset({"primera infancia", "maltrato infantil", "trabajo infantil",
               "explotacion infantil", "reclutamiento de menores", "abuso infantil"}): [
        ("16.2", 1.32), ("16.1", 1.22), ("4.", 1.15),
    ],
    frozenset({"acueducto", "alcantarillado", "agua potable",
               "planta de tratamiento de agua", "agua y saneamiento"}): [
        ("6.", 1.38), ("wash", 1.28), ("3.9", 1.18),
    ],
    # v30: proyectos humanitarios/conflicto → boost ODS 17.2 cooperación internacional
    frozenset({"asistencia humanitaria", "ayuda humanitaria", "emergencia humanitaria",
               "desplazamiento forzado", "desplazados", "confinamiento",
               "cluster humanitario", "accion humanitaria", "respuesta humanitaria"}): [
        ("17.2", 1.45),  # compromisos cooperación países desarrollados
        ("17.6", 1.30),  # cooperación regional e internacional
        ("16.1", 1.25),  # reducir violencia y muertes
        ("1.5",  1.20),  # resiliencia pobres y vulnerables
    ],
    frozenset({"economia circular", "plastico", "reciclaje", "reutilizacion"}): [
        ("12.", 1.32), ("9.4", 1.20), ("11.6", 1.14),
    ],
    frozenset({"educacion", "formacion docente", "aprendizaje", "inclusion educativa"}): [
        ("4.", 1.30), ("4.1", 1.25), ("4.5", 1.18),
    ],
    frozenset({"feminicidio", "vbg", "violencia basada en genero", "lgbti"}): [
        ("5.", 1.30), ("10.3", 1.18),
    ],
    frozenset({"paisajes forestales", "biodiversidad", "deforestacion",
               "ecosistemas", "soluciones basadas en la naturaleza"}): [
        ("15.", 1.30), ("13.", 1.22), ("6.6", 1.14),
    ],
    # Agricultura / seguridad alimentaria → ODS 2
    frozenset({"agricultura", "seguridad alimentaria", "hambre",
               "produccion agropecuaria", "campesinado", "soberania alimentaria"}): [
        ("2.",   1.45),  # ODS 2 — Hambre Cero
        ("2.3",  1.40),  # productividad pequeños agricultores
        ("2.4",  1.35),  # agricultura sostenible
        ("15.2", 1.18),  # gestión sostenible bosques/suelos
    ],
    # Trabajo / empleo / desarrollo económico → ODS 8
    frozenset({"empleo", "trabajo decente", "desempleo", "generacion de empleo",
               "ingresos", "actividad economica", "desarrollo economico"}): [
        ("8.",   1.40),  # ODS 8 — Trabajo Decente
        ("8.5",  1.38),  # empleo pleno y productivo
        ("8.6",  1.32),  # jóvenes NEET
        ("1.1",  1.18),  # pobreza extrema
    ],
    # Gobernanza / transparencia / instituciones → ODS 16.6
    frozenset({"gobernanza", "transparencia", "instituciones", "rendicion de cuentas",
               "anticorrupcion", "participacion ciudadana", "estado de derecho"}): [
        ("16.6", 1.45),  # instituciones eficaces y transparentes
        ("16.7", 1.40),  # toma de decisiones inclusiva
        ("17.9", 1.30),  # fortalecimiento capacidades
        ("16.",  1.22),
    ],
    # Ciencia / tecnología / innovación → ODS 9
    frozenset({"ciencia", "tecnologia", "innovacion", "investigacion",
               "transferencia tecnologica", "i+d", "digital"}): [
        ("9.5",  1.45),  # investigación e innovación
        ("9.b",  1.40),  # desarrollo tecnológico países en desarrollo
        ("17.6", 1.30),  # cooperación ciencia y tecnología
        ("9.",   1.22),
    ],
}

ODS_PENALIZAR_SIN_CONTEXTO = {
    "17.14": {"coherencia", "politicas", "alineacion", "agenda 2030",
              "cooperacion", "marco normativo", "estrategia nacional"},
    "17.15": {"normativo", "liderazgo", "soberania",
              "cooperacion", "politica publica", "autonomia"},
    "17.16": {"alianza", "financiamiento", "datos", "monitoreo",
              "cooperacion", "sur sur", "triangular", "multilateral"},
    "17.17": {"alianza", "publico", "privado", "sociedad civil",
              "cooperacion", "asociacion", "articulacion"},
    "17.6":  {"tecnologia", "conocimiento", "ciencia",
              "transferencia", "innovacion", "investigacion"},
    "17.3":  {"financiamiento", "recurso financiero", "movilizacion",
              "cooperacion", "aod", "fondo", "donacion"},
    "17.2":  {"pais desarrollado", "aod", "ayuda oficial",
              "cooperacion internacional", "donante", "receptor"},
}
ODS_PENALIZACION_SIN_CONTEXTO_FACTOR = 0.78

ODS_BLOQUEOS_TEMATICOS = {
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal"}): {
        "3.6", "transito", "transporte vial", "seguridad vial",
    },
    # v30: proyectos humanitarios/conflicto no deben alinearse con seguridad vial
    frozenset({"asistencia humanitaria", "ayuda humanitaria", "emergencia humanitaria",
               "desplazamiento forzado", "desplazados", "confinamiento",
               "cluster humanitario", "accion humanitaria"}): {
        "3.6", "transito", "transporte vial", "seguridad vial",
        "accidentes de transito",
    },
    frozenset({"economia circular", "plastico"}): {
        "armas", "conflicto armado", "terrorismo",
    },
}
ODS_BLOQUEO_FACTOR = 0.55

MARCO_PENALIZACIONES_TEMATICAS = {
    # v24: penalizar PND "mujeres" cuando el proyecto no es de género
    frozenset({"vih", "sida", "hiv", "aids", "antirretroviral"}): [
        ("PND", "el cambio es con las mujeres: hacia una politica exterior feminista", 0.40),
        ("PND", "el cambio es con las mujeres: garantia de los derechos en salud", 0.52),
        # v27: penalizar entradas CAD no relacionadas con salud/VIH
        ("CAD", "desminado",                      0.15),
        ("CAD", "accion contra minas",            0.15),
        ("CAD", "biosfera",                       0.15),
        ("CAD", "desarrollo rural",               0.35),
        ("CAD", "agricultura",                    0.35),
        ("CAD", "infraestructura",                0.38),
        ("CAD", "transporte",                     0.25),
        ("CAD", "energia",                        0.30),
        ("CAD", "telecomunicaciones",             0.30),
        # v27: penalizar sectores de gobierno incorrectos para VIH
        ("SECTORES", "defensa",     0.22),
        ("SECTORES", "agricultura", 0.30),
        ("SECTORES", "ambiente",    0.30),
        ("SECTORES", "comercio",    0.28),
    ],
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal"}): [
        ("PND",  "el cambio es con las mujeres", 0.42),
        # v27: penalizar sectores incorrectos para desminado
        ("SECTORES", "salud",      0.25),
        ("SECTORES", "educacion",  0.30),
        ("SECTORES", "ambiente",   0.28),
        ("SECTORES", "vivienda",   0.22),
        ("SECTORES", "icbf",       0.22),
        ("SECTORES", "comercio",   0.20),
    ],
    frozenset({"cambio climatico", "adaptacion climatica", "biodiversidad",
               "paisajes forestales", "deforestacion"}): [
        ("PND",  "el cambio es con las mujeres: mujeres como motor", 0.45),
        ("PND",  "el cambio es con las mujeres: hacia una politica exterior feminista", 0.40),
        # v27: penalizar sectores incorrectos para medio ambiente
        ("SECTORES", "defensa",   0.30),
        ("SECTORES", "justicia",  0.30),
    ],
    frozenset({"vih", "sida", "hiv", "aids"}): [
        ("ENCI", "trata de personas", 0.60),
        ("ENCI", "trata de ninos",    0.60),
        ("ENCI", "esclavitud",        0.65),
        ("PMI",  "estupefacientes",   0.70),
    ],
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal",
               "artefactos explosivos", "map", "municiones sin explotar"}): [
        ("PMI",  "desarrollo social: salud",   0.55),
        ("ENCI", "salud mental",                0.65),
        # CAD: bloquear entradas no relacionadas con acción contra minas
        ("CAD",  "inundaciones",                0.18),
        ("CAD",  "estupefacientes",             0.18),
        ("CAD",  "biosfera",                    0.10),
        ("CAD",  "ozono",                       0.10),
        ("CAD",  "marina",                      0.10),
        ("CAD",  "contaminacion del aire",      0.10),
        ("CAD",  "mitigacion social del vih",   0.06),
        ("CAD",  "lucha contra ets",            0.06),
        ("CAD",  "formacion de personal para",  0.12),
        ("CAD",  "planificacion familiar",      0.12),
        ("CAD",  "salud mental",                0.12),
        ("CAD",  "educacion y formacion",       0.22),
        ("CAD",  "suministro de energia",       0.18),
        ("CAD",  "desarrollo agricola",         0.15),
        ("CAD",  "agricultura",                 0.15),
        # v29: bloquear agua/saneamiento que no aplica a desminado
        ("CAD",  "abastecimiento de agua",      0.12),
        ("CAD",  "agua potable",                0.12),
        ("CAD",  "saneamiento basico",          0.12),
        ("CAD",  "agua y saneamiento",          0.12),
        ("CAD",  "sistemas de agua",            0.12),
        ("CAD",  "infraestructura de agua",     0.12),
        ("CAD",  "gestion de residuos",         0.18),
        ("CAD",  "residuos solidos",            0.18),
        ("CAD",  "vivienda",                    0.20),
        ("CAD",  "transporte",                  0.20),
        ("CAD",  "energia renovable",           0.18),
        # ODS: bloquear metas de salud/epidemias no relevantes
        ("ODS",  "epidemias del sida",          0.10),
        ("ODS",  "poner fin a las epidemias",   0.10),
        ("ODS",  "salud sexual y reproductiva", 0.30),
        ("ODS",  "reducir sustancialmente el numero de muertes y enfermedades", 0.38),
        # PND: penalizar capítulos no relacionados con desminado técnico
        ("PND",  "la cultura de paz en la cotidianidad", 0.45),
        ("PND",  "el dialogo: un camino",                0.42),
        # SECTORES: penalizar sectores sin relación con desminado
        ("SECTORES", "salud",      0.20),
        ("SECTORES", "educacion",  0.22),
        ("SECTORES", "ambiente",   0.22),
        ("SECTORES", "vivienda",   0.18),
        ("SECTORES", "icbf",       0.18),
        ("SECTORES", "comercio",   0.15),
        ("SECTORES", "hacienda",   0.20),
        ("SECTORES", "agricultura", 0.18),
    ],
    # v27: Asistencia humanitaria — no es desminado ni VIH
    frozenset({"asistencia humanitaria", "ayuda humanitaria", "emergencia humanitaria",
               "cluster humanitario"}): [
        ("CAD",  "mitigacion social del vih",   0.20),
        ("CAD",  "lucha contra ets",            0.20),
        ("CAD",  "biosfera",                    0.20),
        ("CAD",  "accion contra minas",         0.35),
        ("ODS",  "epidemias del sida",          0.20),
    ],
    # v27: Niñez/Entornos protectores — trata y protección infantil
    frozenset({"primera infancia", "ninez", "maltrato infantil", "trabajo infantil",
               "explotacion infantil", "trata de personas", "reclutamiento de menores"}): [
        ("CAD",  "mitigacion social del vih",   0.25),
        ("CAD",  "lucha contra ets",            0.25),
        ("CAD",  "control estupefacientes",     0.40),
        ("SECTORES", "defensa",                 0.25),
        ("SECTORES", "ambiente",                0.25),
    ],
    # v27: VBG/Género — no es desminado ni VIH puro
    frozenset({"violencia basada en genero", "vbg", "feminicidio",
               "violencia intrafamiliar", "violencia de genero"}): [
        ("CAD",  "biosfera",               0.20),
        ("CAD",  "control estupefacientes", 0.35),
        ("CAD",  "accion contra minas",    0.40),
        ("SECTORES", "defensa",            0.25),
        ("SECTORES", "ambiente",           0.25),
    ],
}


# ══════════════════════════════════════════════════════════════════════════
# v26: CAD WHITELIST — entradas que requieren keywords explícitas del proyecto
# Si el proyecto NO contiene ninguna de las keywords de la lista,
# se aplica un factor de bloqueo fuerte a esas entradas CAD.
# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# CAD_BOOST_ESPECIFICO — boost para entradas CAD por tema del proyecto
# Formato: {frozenset_keywords}: [(substr_titulo_cad, factor), ...]
# ══════════════════════════════════════════════════════════════
CAD_BOOST_ESPECIFICO = {
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal",
               "accion contra minas", "artefactos explosivos", "map", "muse",
               "liberacion de tierras", "municiones sin explotar"}): [
        ("accion contra minas",              1.65),  # categoría CAD exacta
        ("paz y seguridad",                  1.50),
        ("conflicto armado",                 1.45),
        ("construccion de paz",              1.40),
        ("seguridad publica",                1.35),
        ("prevencion de conflictos",         1.35),
        ("rehabilitacion",                   1.30),
        ("victimas",                         1.28),
        ("asistencia humanitaria de emergencia", 1.20),
        ("ayuda humanitaria",                1.18),
    ],
    frozenset({"agua potable", "acueducto", "saneamiento", "alcantarillado"}): [
        ("abastecimiento de agua",           1.55),
        ("agua potable",                     1.55),
        ("saneamiento basico",               1.50),
        ("sistemas de agua",                 1.45),
        ("gestion del agua",                 1.40),
    ],
    frozenset({"violencia basada en genero", "vbg", "feminicidio", "mujer"}): [
        ("empoderamiento de la mujer",       1.50),
        ("igualdad de genero",               1.45),
        ("violencia contra la mujer",        1.40),
        ("derechos de la mujer",             1.35),
    ],
    # Medio ambiente / cambio climático
    frozenset({"clima", "cambio climatico", "paramo", "biodiversidad",
               "ecosistema", "deforestacion", "carbono", "reforestacion"}): [
        ("medio ambiente",                   1.55),
        ("cambio climatico",                 1.50),
        ("biodiversidad",                    1.48),
        ("bosques",                          1.45),
        ("gestion ambiental",                1.40),
        ("energia renovable",                1.35),
    ],
    # Agricultura / seguridad alimentaria
    frozenset({"agricultura", "seguridad alimentaria", "campesino",
               "cultivos", "produccion agropecuaria", "rural"}): [
        ("agricultura",                      1.55),
        ("seguridad alimentaria",            1.52),
        ("desarrollo rural",                 1.48),
        ("produccion de alimentos",          1.45),
        ("pesca",                            1.35),
    ],
    # Reincorporación / paz total / posconflicto
    frozenset({"reincorporacion", "excombatientes", "posconflicto",
               "acuerdo de paz", "farc", "paz total", "reintegracion"}): [
        ("construccion de paz",              1.60),
        ("reintegracion",                    1.55),
        ("conflicto armado",                 1.50),
        ("prevencion de conflictos",         1.45),
        ("paz y seguridad",                  1.40),
        ("reconciliacion",                   1.38),
    ],
    # Asistencia humanitaria (proyectos no-desminado)
    frozenset({"humanitario", "desplazamiento", "emergencia humanitaria",
               "refugiados", "proteccion civil", "victimas"}): [
        ("asistencia humanitaria de emergencia", 1.55),
        ("ayuda humanitaria",                1.50),
        ("personas desplazadas",             1.48),
        ("proteccion de civiles",            1.45),
        ("prevencion de desastres",          1.35),
    ],
    # Cooperación internacional / sur-sur / triangular
    frozenset({"cooperacion internacional", "sur sur", "triangular",
               "cooperacion tecnica", "apc", "agenda 2030"}): [
        ("cooperacion para el desarrollo",   1.55),
        ("cooperacion tecnica",              1.52),
        ("asistencia oficial al desarrollo", 1.48),
        ("coordinacion de la ayuda",         1.45),
    ],
    # Migración / venezolanos / movilidad humana
    frozenset({"migracion", "venezolanos", "migrantes", "refugiados",
               "movilidad humana", "integracion migrantes"}): [
        ("personas desplazadas",             1.55),
        ("asistencia a refugiados",          1.52),
        ("migracion",                        1.50),
        ("integracion social",               1.40),
        ("proteccion de civiles",            1.35),
    ],
    # Niñez / primera infancia / juventud
    frozenset({"ninez", "primera infancia", "infancia", "juventud",
               "adolescentes", "cero a siempre"}): [
        ("educacion basica",                 1.50),
        ("salud y poblacion",                1.45),
        ("nutricion basica",                 1.48),
        ("proteccion social",                1.42),
        ("derechos de la infancia",          1.55),
    ],
}

# ══════════════════════════════════════════════════════════════
# SECTORES_BOOST_ESPECIFICO — boost para entradas SECTORES por tema
# Formato: {frozenset_keywords}: [(substr_titulo_sector, factor), ...]
# ══════════════════════════════════════════════════════════════
SECTORES_BOOST_ESPECIFICO = {
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal",
               "accion contra minas", "artefactos explosivos", "map", "muse",
               "liberacion de tierras", "municiones sin explotar"}): [
        ("defensa",      1.65),  # Ministerio de Defensa — sector correcto desminado
        ("apc",          1.50),  # APC Colombia — cooperación internacional
        ("presidencia",  1.35),  # Presidencia — paz y posconflicto
        ("interior",     1.30),  # Ministerio Interior — territorios
        ("justicia",     1.25),  # Justicia transicional
    ],
    frozenset({"violencia basada en genero", "vbg", "feminicidio", "mujer"}): [
        ("equidad de genero",  1.55),
        ("mujer",              1.50),
        ("consejeria presidencial", 1.35),
    ],
    frozenset({"agua potable", "acueducto", "saneamiento", "alcantarillado"}): [
        ("vivienda",     1.50),  # Minvivienda — agua y saneamiento
        ("ambiente",     1.35),
    ],
    # Medio ambiente / cambio climático
    frozenset({"clima", "cambio climatico", "paramo", "biodiversidad",
               "ecosistema", "deforestacion", "carbono"}): [
        ("ambiente",          1.65),  # Ministerio de Ambiente
        ("agricultura",       1.35),  # MADR — ecosistemas rurales
        ("apc",               1.28),
    ],
    # Agricultura / seguridad alimentaria
    frozenset({"agricultura", "seguridad alimentaria", "campesino",
               "cultivos", "rural", "produccion agropecuaria"}): [
        ("agricultura",       1.65),  # Ministerio de Agricultura
        ("prosperidad",       1.35),  # DPS — poblaciones rurales vulnerables
        ("apc",               1.28),
    ],
    # Reincorporación / posconflicto
    frozenset({"reincorporacion", "excombatientes", "posconflicto",
               "acuerdo de paz", "paz total", "reintegracion"}): [
        ("presidencia",       1.60),  # Alto Comisionado de Paz
        ("defensa",           1.50),  # Ministerio Defensa — transición
        ("justicia",          1.45),  # Justicia transicional
        ("interior",          1.40),
        ("apc",               1.35),
    ],
    # Cooperación internacional / sur-sur / triangular
    frozenset({"cooperacion internacional", "sur sur", "triangular",
               "cooperacion tecnica", "apc", "agenda 2030"}): [
        ("apc",               1.70),  # APC Colombia — cooperación internacional
        ("cancilleria",       1.55),  # Cancillería — relaciones exteriores
        ("presidencia",       1.35),
    ],
    # Migración / venezolanos / movilidad humana
    frozenset({"migracion", "venezolanos", "migrantes",
               "movilidad humana", "refugiados"}): [
        ("cancilleria",       1.60),  # Cancillería — migración
        ("prosperidad",       1.50),  # DPS — integración social
        ("interior",          1.40),
        ("apc",               1.35),
    ],
    # Niñez / primera infancia / juventud
    frozenset({"ninez", "primera infancia", "infancia", "juventud",
               "adolescentes", "cero a siempre"}): [
        ("presidencia",       1.55),  # ICBF bajo Presidencia
        ("educacion",         1.50),  # MEN
        ("salud",             1.45),  # Minsalud — nutrición/salud infantil
        ("prosperidad",       1.40),  # DPS — protección social
    ],
}


CAD_WHITELIST_ENTRIES = {
    # CAD VIH/SIDA y ETS — solo deben aparecer si el proyecto habla de salud/VIH
    "mitigacion social del vih":  {
        # Solo términos explícitamente relacionados con VIH — NO "salud sexual" sola
        # porque aparece en proyectos de VBG/género y activaría falsamente esta entrada
        "keywords_requeridas": {"vih", "sida", "hiv", "aids",
                                 "antirretroviral", "epidemia", "tuberculosis"},
        "factor_bloqueo": 0.02,  # bloqueo casi total post-boosts
    },
    "lucha contra ets": {
        "keywords_requeridas": {"vih", "sida", "hiv", "aids",
                                 "antirretroviral", "epidemia", "tuberculosis"},
        "factor_bloqueo": 0.02,
    },
    "lucha contra enfermedades de transmision": {
        "keywords_requeridas": {"vih", "sida", "hiv", "aids", "its", "ets",
                                 "enfermedades transmisibles", "salud sexual",
                                 "antirretroviral", "epidemia", "tuberculosis",
                                 "malaria", "hepatitis"},
        "factor_bloqueo": 0.10,
    },
    "formacion de personal para poblacion y salud reproductiva": {
        "keywords_requeridas": {"salud reproductiva", "salud sexual", "planificacion familiar",
                                 "anticoncepcion", "vih", "sida", "its", "poblacion",
                                 "salud materna", "mortalidad materna"},
        "factor_bloqueo": 0.15,
    },
    "planificacion familiar": {
        "keywords_requeridas": {"salud reproductiva", "planificacion familiar",
                                 "anticoncepcion", "embarazo", "vih", "sida",
                                 "salud sexual", "salud materna"},
        "factor_bloqueo": 0.18,
    },
    # Control estupefacientes — solo para proyectos de drogas
    "control estupefacientes": {
        "keywords_requeridas": {"estupefacientes", "drogas", "narcotrafico",
                                 "sustancias psicoactivas", "coca", "coca ilicita",
                                 "cultivos ilicitos", "sustancias sicoactivas"},
        "factor_bloqueo": 0.18,
    },
    # Biosfera — solo medio ambiente
    "proteccion de la biosfera": {
        "keywords_requeridas": {"biosfera", "biodiversidad", "ecosistemas",
                                 "cambio climatico", "medio ambiente", "paramos",
                                 "deforestacion", "bosques"},
        "factor_bloqueo": 0.10,
    },
    # v27: Entradas CAD de educación — solo para proyectos de educación
    "educacion y formacion": {
        "keywords_requeridas": {"educacion", "formacion", "aprendizaje", "escolar",
                                 "docente", "pedagogia", "curriculum", "becas",
                                 "capacitacion", "competencias"},
        "factor_bloqueo": 0.22,
    },
    # v27: Salud mental — solo para proyectos de salud mental/psicosocial
    "salud mental": {
        "keywords_requeridas": {"salud mental", "psicosocial", "psicologia",
                                 "trastorno mental", "depresion", "ansiedad",
                                 "psiquiatria", "bienestar emocional", "trauma",
                                 "apoyo psicologico", "atencion psicosocial"},
        "factor_bloqueo": 0.20,
    },
    # v27: Energía — solo para proyectos de energía
    "suministro de energia": {
        "keywords_requeridas": {"energia", "electricidad", "electrificacion",
                                 "energia renovable", "solar", "eolica",
                                 "hidroelectrica", "red electrica", "acceso energia"},
        "factor_bloqueo": 0.22,
    },
    # v27: Agricultura en CAD — no debe aparecer en desminado/VIH
    "desarrollo agricola": {
        "keywords_requeridas": {"agricultura", "agricola", "cultivos", "cosecha",
                                 "campesino", "ganaderia", "produccion agropecuaria",
                                 "semillas", "fertilizantes", "campo"},
        "factor_bloqueo": 0.22,
    },
    # v30: Salud general y básica — solo para proyectos de salud real
    "salud, general": {
        "keywords_requeridas": {"salud", "atencion medica", "hospital", "clinica",
                                 "medico", "enfermedad", "tratamiento", "paciente",
                                 "sistema de salud", "vih", "sida", "nutricion",
                                 "vacunacion", "mortalidad", "morbilidad"},
        "factor_bloqueo": 0.20,
    },
    "salud basica": {
        "keywords_requeridas": {"salud", "atencion medica", "hospital", "clinica",
                                 "medico", "enfermedad", "tratamiento", "paciente",
                                 "sistema de salud", "vih", "sida", "nutricion",
                                 "vacunacion", "mortalidad", "morbilidad"},
        "factor_bloqueo": 0.20,
    },
}

PARAMETROS_SIMILITUD = {
    "boost_coincidencias_exactas":  1.55,
    "boost_terminos_tecnicos":      1.55,
    "boost_mega_match":             1.70,
    "boost_poblaciones_clave":      1.30,
    "penalizacion_texto_muy_corto": 0.82,
    "min_palabras_penalizacion":    18,
    "penalizacion_texto_corto":     0.90,
    "min_palabras_texto":           35,
    "normalizacion_texto_largo":    0.96,
    "max_palabras_normalizacion":   180,
    "boost_multiples_terminos":     1.22,
    "min_terminos_bonus":           3,
    "boost_muchos_terminos":        1.35,
    "min_terminos_bonus_alto":      5,
    "penalizacion_sin_terminos":    0.62,
    "penalizacion_match_generico":  0.78,
}

KEYWORDS_TIER1_CRITICAS = {
    "desminado", "desminado humanitario", "accion contra minas",
    "map", "muse", "minas antipersonal", "artefactos explosivos",
    "municiones sin explotar", "aicma", "victimas de minas",
    "liberacion de tierras", "despeje humanitario",
    "vih", "sida", "vih/sida", "hiv", "aids",
    "arv", "tar", "tarv", "antiretroviral", "antirretroviral",
    "prep", "pep", "carga viral", "cd4", "pvvih",
    "personas viviendo con vih", "its", "ets", "hsh",
    "trabajadoras sexuales", "trabajo sexual",
    "usuarios drogas inyectables",
    "prueba rapida de vih", "diagnostico vih", "poblaciones clave vih",
    "migrantes venezolanos",
    "pdet", "zomac", "reincorporacion", "excombatientes",
    "acuerdo de paz", "posconflicto",
}

KEYWORDS_TIER2_ESPECIFICAS = {
    "campesinos", "campesinado", "economia campesina",
    "agricultura familiar", "pequenos productores",
    "reforma agraria", "seguridad alimentaria",
    "violencia basada en genero", "vbg", "feminicidio",
    "empoderamiento de mujeres", "lgbti",
    "cambio climatico", "adaptacion climatica",
    "deforestacion", "biodiversidad", "ecosistemas",
    "economia circular", "plastico", "reciclaje",
    "acueducto", "saneamiento basico", "agua potable",
    "educacion", "formacion docente", "aprendizaje",
    "primera infancia", "nutricion infantil",
    "salud mental", "psicosocial",
    "energia renovable", "eficiencia energetica",
    "turismo sostenible", "patrimonio cultural",
}

KEYWORDS_TIER3 = {
    "cooperacion internacional", "sur-sur", "cooperacion triangular",
    "aod", "ayuda oficial al desarrollo",
    "indigenas", "afrodescendientes", "comunidades negras",
    "discapacidad", "migrantes", "refugiados", "desplazados",
    "tuberculosis", "hepatitis", "malaria", "dengue",
    "salud publica", "salud sexual", "salud reproductiva",
    "pymes", "emprendimiento", "cadenas de valor",
    "desarrollo sostenible", "desarrollo territorial",
    "gobernanza", "participacion ciudadana",
    "fortalecimiento institucional", "capacidades locales",
}

KEYWORDS_TECNICAS = KEYWORDS_TIER1_CRITICAS | KEYWORDS_TIER2_ESPECIFICAS | KEYWORDS_TIER3

STOPWORDS_PERSONALIZADAS = {
    "mediante", "traves", "objetivo", "meta", "resultado",
    "realizar", "ejecutar", "implementar", "desarrollar", "establecer",
    "crear", "generar", "proceso", "nacional", "colombia",
    "proyecto", "programa", "actividad", "estrategia", "accion",
    "acciones", "politica", "politicas",
    "sistema", "acceso", "recursos", "apoyo", "fortalecimiento",
    "mejora", "mejoramiento", "identificar", "promover", "garantizar",
    "asegurar", "lograr", "periodo", "marco", "general",
    "especifico", "integral", "social", "publico",
}

CONFIG = {
    "RUTA_EXCEL_MARCOS":         "data/marcos_estrategicos.xlsx",
    "RUTA_PROYECTOS_REFERENCIA": "data/Proyectos_alienados_2025.xlsx",
    "RUTA_EXPORTACIONES":        "exports/",
    "RUTA_CACHE_EMBEDDINGS":     "cache/embeddings_v26/",

    "MODELO_NLP":            MODELO_AUTO,
    "BATCH_SIZE_EMBEDDINGS": BATCH_AUTO,

    "UMBRAL_CONFIANZA_MINIMO": 25,
    "TOP_N_RESULTADOS":         5,

    "PESO_SEMANTICO_SIN_REF":  W_SEMANTICO_SIN_REF,
    "PESO_NUCLEARES_SIN_REF":  W_NUCLEARES_SIN_REF,
    "PESO_KEYWORDS_SIN_REF":   W_KEYWORDS_SIN_REF,
    "PESO_SEMANTICO_CON_REF":  W_SEMANTICO_CON_REF,
    "PESO_NUCLEARES_CON_REF":  W_NUCLEARES_CON_REF,
    "PESO_KEYWORDS_CON_REF":   W_KEYWORDS_CON_REF,
    "PESO_REFERENCIA_CON_REF": W_REFERENCIA_CON_REF,

    "PALABRAS_NUCLEARES":    PALABRAS_NUCLEARES_DEFAULT,
    "PMI_ACTIVO_DEFAULT":    PMI_ACTIVO_DEFAULT,

    "CACHE_EMBEDDINGS":       True,
    "LOG_LEVEL":              "INFO",
    "USAR_GPU_SI_DISPONIBLE": True,

    "HOJAS_EXCEL": {
        "ODS":      "ODS",
        "PND":      "PND",
        "ENCI":     "ENCI",
        "PMI":      "PMI",
        "CAD":      "CAD",
        "SECTORES": "SECTORES",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# v23: BOOST ESPECIFICO PARA ENCI Y PND
# Si el proyecto tiene estas keywords, impulsa estas entradas específicas
# ══════════════════════════════════════════════════════════════════════════

ENCI_BOOST_ESPECIFICO = {
    frozenset({"vih", "sida", "hiv", "aids", "antirretroviral", "tarv", "pvvih"}): [
        ("2.2.2", 1.45),   # migrantes + VIH
        ("3.3.7", 1.22),   # prevencion atencion justicia
        ("3.2.10", 1.18),  # VBG (coinfeccion VIH/VBG)
    ],
    frozenset({"desminado", "aicma", "antipersonal", "minas antipersonal",
               "accion contra minas", "liberacion de tierras", "artefactos explosivos",
               "map", "muse", "municiones sin explotar"}): [
        ("3.1.1",  1.58),  # reforma rural + desminado humanitario
        ("3.1.5",  1.55),  # reincorporacion integral + desminadores
        ("3.3.10", 1.50),  # reparacion colectiva victimas minas
        ("3.1.6",  1.40),  # JEP verdad reparacion no repeticion
        ("3.1.3",  1.35),  # pdet sustitucion territorios minados
        ("3.1.2",  1.28),  # planeacion territorial paz
        ("3.1.9",  1.22),  # derechos humanos
    ],
    frozenset({"violencia basada en genero", "vbg", "feminicidio", "mujer"}): [
        ("3.2.10", 1.50),  # violencias basadas en genero
        ("3.3.6", 1.40),   # violencia politica mujeres
        ("3.3.7", 1.35),   # plan integral prevencion atencion
        ("3.3.1", 1.22),   # iniciativas productivas mujeres
    ],
    frozenset({"migrante", "venezolano", "refugiado", "retornado"}): [
        ("2.2.2", 1.55),   # atencion integracion migrantes
        ("2.2.3", 1.25),   # coordinacion sectorial migracion
    ],
    frozenset({"agua potable", "acueducto", "saneamiento", "alcantarillado"}): [
        ("1.2.4", 1.50),   # obras acueducto saneamiento
        ("4.3.4", 1.15),   # buenas practicas territoriales
    ],
    frozenset({"cambio climatico", "adaptacion", "clima", "biodiversidad"}): [
        ("1.3.1", 1.40),   # acuerdo paris NDC
        ("1.3.2", 1.35),   # resiliencia climatica
        ("1.2.3", 1.25),   # mercado carbono dialogos
    ],
    frozenset({"primera infancia", "ninos", "nutricion", "desnutricion"}): [
        ("2.1.3", 1.50),   # nutricion NNA
        ("2.1.4", 1.45),   # centros recuperacion nutricional
        ("3.2.5", 1.40),   # primera infancia ninez
    ],
    frozenset({"economia campesina", "campesino", "agricultura familiar", "pequenos productores"}): [
        ("2.3.3", 1.45),   # economias campesinas solidarias
        ("2.1.1", 1.40),   # cadenas agricolas alimentacion
        ("2.3.5", 1.30),   # asociatividad solidaria paz
    ],
    frozenset({"reincorporacion", "excombatiente", "firmantes", "paz total"}): [
        ("3.1.5", 1.55),   # reincorporacion integral
        ("3.1.1", 1.30),   # reforma rural + reincorporacion
        ("3.1.3", 1.22),   # pnis pdet sustitucion
    ],
    frozenset({"pdet", "sustitucion de cultivos", "coca", "pnis"}): [
        ("3.1.3", 1.55),   # PNIS pdet sustitucion
        ("3.1.1", 1.30),   # reforma rural
        ("3.1.2", 1.20),   # planeacion territorial paz
    ],
    frozenset({"cooperacion internacional", "sur-sur", "snci", "apc colombia"}): [
        ("4.2.2", 1.40),   # observatorio cooperacion
        ("4.2.4", 1.35),   # fortalecimiento capacidades gestion
        ("4.3.4", 1.30),   # buenas practicas territoriales
        ("4.4.3", 1.25),   # sesiones anuales sur-sur triangular
    ],
    # v27: Asistencia humanitaria
    frozenset({"asistencia humanitaria", "ayuda humanitaria", "cluster humanitario",
               "accion humanitaria", "desplazados", "desplazamiento forzado"}): [
        ("3.3.10", 1.50),  # reparacion colectiva victimas
        ("2.2.2",  1.48),  # atencion e integracion migrantes/desplazados
        ("3.3.7",  1.42),  # plan integral prevencion atencion
        ("2.1.3",  1.35),  # nutricion NNA en emergencia
        ("3.1.9",  1.25),  # derechos humanos
    ],
    # v27: Entornos protectores / niñez y trata
    frozenset({"trata de personas", "escnna", "explotacion sexual comercial infantil",
               "explotacion infantil", "primera infancia", "maltrato infantil"}): [
        ("4.5.6", 1.58),  # marco regulatorio trata personas
        ("3.2.5", 1.52),  # primera infancia ninez
        ("3.3.7", 1.40),  # plan integral prevencion atencion justicia
        ("3.2.10",1.35),  # violencias basadas en genero (coocurrencia)
        ("3.1.9", 1.25),  # derechos humanos
    ],
    # v27: Violencia intrafamiliar y generaciones sin violencias
    frozenset({"violencia intrafamiliar", "violencia domestica",
               "violencia basada en genero", "vbg", "feminicidio",
               "generaciones sin violencias"}): [
        ("3.2.10", 1.55),  # violencias basadas en genero
        ("3.3.7",  1.48),  # plan integral prevencion atencion
        ("3.3.6",  1.42),  # violencia politica mujeres
        ("3.3.1",  1.25),  # iniciativas productivas mujeres
    ],
}

PND_BOOST_ESPECIFICO = {
    frozenset({"vih", "sida", "hiv", "aids", "antirretroviral", "tarv", "pvvih",
               "salud publica", "epidemia", "poblaciones clave"}): [
        ("superacion de privaciones", 1.45),
        ("expansion de capacidades",  1.40),
        ("sistema de proteccion",     1.35),
        ("el cambio es con las mujeres", 1.22),
        ("jovenes con derechos",      1.20),
    ],
    frozenset({"desminado", "minas antipersonal", "aicma", "artefactos explosivos",
               "accion contra minas", "liberacion de tierras", "map", "muse"}): [
        ("defensa integral",                        1.58),
        ("reparacion efectiva",                     1.55),
        ("territorios que se transforman",          1.50),
        ("reivindicacion de los derechos",          1.42),
        ("reforma rural",                           1.35),
        ("justicia transicional",                   1.30),
        ("victimas del conflicto",                  1.28),
        ("reincorporacion integral",                1.18),
        # Excluidos intencionalmente: "la cultura de paz en la cotidianidad"
        # y "el dialogo: un camino" — demasiado genéricos para proyectos técnicos de desminado
    ],
    frozenset({"migrante", "venezolano", "refugiado", "regularizacion"}): [
        ("fortalecimiento de vinculos", 1.55),
        ("superacion de privaciones",   1.35),
        ("expansion de capacidades",    1.30),
    ],
    frozenset({"cambio climatico", "adaptacion", "biodiversidad", "ecosistemas"}): [
        ("transicion economica",     1.45),
        ("naturaleza viva",          1.40),
        ("transicion energetica",    1.25),
    ],
    frozenset({"agua potable", "acueducto", "saneamiento basico"}): [
        ("el agua, la biodiversidad", 1.45),
        ("superacion de privaciones", 1.25),
    ],
    frozenset({"reincorporacion", "excombatiente", "firmantes"}): [
        ("reivindicacion de los derechos", 1.50),
        ("territorios que se transforman",  1.40),
        ("dialogos de paz",                 1.30),
    ],
    frozenset({"violencia basada en genero", "vbg", "feminicidio"}): [
        ("el cambio es con las mujeres", 1.50),
        ("colombia igualitaria",        1.35),
        ("actores diferenciales",       1.22),
    ],
    frozenset({"primera infancia", "nutricion", "desnutricion"}): [
        ("crece la generacion",       1.50),
        ("infraestructura social",    1.35),
        ("superacion de privaciones", 1.25),
    ],
    frozenset({"agricultura familiar", "campesino", "reforma agraria"}): [
        ("campesinado colombiano",    1.45),
        ("derecho humano a la alimentacion", 1.40),
        ("reforma rural",             1.30),
    ],
    # v27: Asistencia humanitaria en PND
    frozenset({"asistencia humanitaria", "ayuda humanitaria", "cluster humanitario",
               "desplazados", "desplazamiento forzado"}): [
        ("reparacion efectiva",             1.48),
        ("superacion de privaciones",       1.45),
        ("sistema de proteccion",           1.40),
        ("territorios que se transforman",  1.30),
        ("actores diferenciales",           1.22),
    ],
    # v27: Niñez/trata/entornos protectores en PND
    frozenset({"trata de personas", "escnna", "explotacion sexual comercial infantil",
               "primera infancia", "maltrato infantil", "reclutamiento de menores"}): [
        ("crece la generacion",           1.55),
        ("superacion de privaciones",     1.48),
        ("reivindicacion de los derechos", 1.40),
        ("infraestructura social",        1.35),
        ("actores diferenciales",         1.28),
    ],
    # v27: Violencia intrafamiliar / generaciones sin violencias
    frozenset({"violencia intrafamiliar", "violencia domestica",
               "generaciones sin violencias", "prevencion violencia"}): [
        ("el cambio es con las mujeres", 1.52),
        ("colombia igualitaria",         1.45),
        ("actores diferenciales",        1.35),
        ("seguridad humana",             1.28),
        ("superacion de privaciones",    1.22),
    ],
}

VERSION = "29.0"
FECHA   = "2026-03-05"

print(f"\n🎯 CICLOPE v{VERSION} — Alineación de Alta Precisión")
print(f"⚙️  Sem {W_SEMANTICO_SIN_REF*100:.0f}% / Nuc {W_NUCLEARES_SIN_REF*100:.0f}% / KW {W_KEYWORDS_SIN_REF*100:.0f}%")
print(f"🔧 {len(PALABRAS_NUCLEARES_DEFAULT)} nucleares | CAD Whitelist ampliada | Sectores enriquecidos\n")
