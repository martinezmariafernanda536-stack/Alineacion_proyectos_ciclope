"""CICLOPE v23.0 - Data Loader con Enriquecimiento Estructural Completo - APC COLOMBIA"""
import pandas as pd
from pathlib import Path
import re

_PND_EJES = {
    "ORDENAMIENTO DEL TERRITORIO ALREDEDOR DEL AGUA Y JUSTICIA AMBIENTAL": (
        "ordenamiento territorial agua justicia ambiental biodiversidad ecosistemas "
        "paramos humedales bosques cuencas hidrograficas catastro multipropósito "
        "tenencia tierra acceso tierra catastro rural formalizacion predial "
        "gestion riesgo desastres alerta temprana resiliencia "
        "escazu acuerdos ambientales internacionales acuerdo paris "
        "cambio climatico adaptacion mitigacion "
        "acueducto alcantarillado agua potable saneamiento basico "
        "conservacion naturaleza servicios ecosistemicos "
        "gobernanza ambiental institucionalidad ambiental "
        "determinantes ambientales planificacion territorial"
    ),
    "SEGURIDAD HUMANA Y JUSTICIA SOCIAL": (
        "seguridad humana justicia social proteccion social derechos "
        "VIH SIDA HIV AIDS antirretroviral TARV ARV prep pep pvvih carga viral CD4 "
        "poblaciones clave hsh trabajo sexual personas transgenero migrantes venezolanos "
        "tuberculosis hepatitis malaria ITS ETS salud publica epidemia prevencion tratamiento "
        "salud mental nutricion primera infancia mortalidad materna vacunacion "
        "educacion cobertura inclusion calidad primera infancia formacion docentes "
        "vivienda infraestructura social centros desarrollo integral CDI "
        "defensa soberania fuerzas militares policia seguridad "
        "justicia servicios juridicos acceso justicia victimas "
        "transicion justicia JEP verdad reparacion no repeticion "
        "transparencia anticorrupcion institucionalidad "
        "desplazados migrantes refugiados retornados integracion atencion "
        "lgbti diversidad sexual genero discriminacion "
        "proteccion ninez adolescentes primera infancia crianza "
        "jovenes oportunidades empleo educacion cultura paz "
        "adultos mayores pension vejez envejecimiento "
        "personas discapacidad inclusion garantias mundo sin barreras "
        "pobreza extrema vulnerabilidad superacion privaciones dignidad humana "
        "renta ciudadana transferencias condicionadas familias "
        "bienestar social calidad vida inclusion productiva "
        "salud sexual reproductiva planificacion familiar anticoncepcion"
    ),
    "DERECHO HUMANO A LA ALIMENTACIÓN": (
        "seguridad alimentaria soberania alimentaria derecho alimentacion hambre cero "
        "disponibilidad alimentos acceso fisico alimentos "
        "produccion agropecuaria transformacion sector agropecuario "
        "cadenas suministro logistica agricola comercializacion mercados "
        "alimentos sanos seguros inocuidad alimentaria calidad "
        "practicas alimentacion saludable nutricion curso vida "
        "desnutricion malnutricion recuperacion nutricional "
        "unidades centros recuperacion nutricional ZRN "
        "comunidades indigenas NARP modelos atencion integral propios "
        "riego distritos rehabilitacion pequena escala "
        "cadenas agricolas derecho alimentacion "
        "pequenos productores economia campesina agricultura familiar "
        "gobernanza multinivel politicas alimentarias"
    ),
    "TRANSFORMACIÓN PRODUCTIVA, INTERNACIONALIZACIÓN Y ACCIÓN CLIMÁTICA": (
        "transformacion productiva internacionalizacion accion climatica "
        "cambio climatico adaptacion mitigacion acuerdo paris NDC "
        "biodiversidad CBD marco sendai reduccion riesgo "
        "resiliencia climatica comunidades enfoque comunitario adaptacion "
        "energia renovable FNCER solar eolica hidro eficiencia energetica "
        "descarbonizacion transicion energetica combustibles fosiles "
        "ciudades resilientes habitat sostenible "
        "reindustrializacion bioeconomia economia verde economia circular reciclaje "
        "produccion limpia innovacion tecnologica ciencia I+D+i "
        "comercio exterior internacionalizacion politica comercial TLC "
        "cooperacion regulatoria acuerdos comerciales internacionales "
        "investigacion desarrollo innovacion misiones estrategicas "
        "biotecnologia ecosistemas naturales territorios "
        "infraestructura TIC digital conectividad "
        "financiamiento desarrollo mecanismos habilitantes "
        "conservacion naturaleza programa naturaleza viva "
        "carbono neutralidad territorios resilientes "
        "cadenas valor exportaciones competitividad pymes "
        "agroemprendedor andino amazonico biologico "
        "paisajes forestales deforestacion bosques amazonia "
        "economia productiva modelos productivos sostenibles"
    ),
    "CONVERGENCIA REGIONAL": (
        "convergencia regional ciudades territorios vivienda habitat "
        "ciudad construida participativa incluyente "
        "servicios sociales equipamientos urbanos colectivos "
        "modelos desarrollo supramunicipales vinculos urbano rurales "
        "acueducto alcantarillado agua saneamiento urbano rural "
        "vivienda digna deficit habitacional subsidios vivienda VIS VIP "
        "gestion suelo ordenamiento urbano planificacion "
        "barrios paz territorios humanos mecanismos acceso "
        "infraestructura vial conectividad transporte logistica "
        "productividad sistemas regionales innovacion "
        "fortalecimiento institucional motor cambio confianza "
        "dispositivos democraticos participacion ciudadana dialogo "
        "reparacion victimas reparacion colectiva colectiva "
        "campesinado tierra territorialidades campesinas educacion rural "
        "discapacidad garantias inclusion mundo sin barreras "
        "paz territorios implementacion acuerdos PDET "
        "ELN dialogos paz conversaciones politicas "
        "desescalamiento conflicto paz cotidianidad"
    ),
    "ACTORES DIFERENCIALES PARA EL CAMBIO": (
        "actores diferenciales cambio mujeres jovenes etnias campesinado discapacidad "
        "mujeres motor desarrollo economico sostenible protectoras vida ambiente agroemprendedoras "
        "mujeres politica vida paz liderazgo femenino participacion politica paridad "
        "salud plena mujeres derechos sexuales reproductivos VIH SIDA "
        "vida libre violencias feminicidio VBG violencia basada genero "
        "sociedad libre estereotipos gobernanza genero observatorios casas mujeres "
        "politica exterior feminista liderazgo internacional genero "
        "jovenes transformaciones vida oportunidades empleo bienestar "
        "salud bienestar jovenes ITS enfermedades transmisibles drogas salud mental "
        "juventudes artistas cultura paz protagonistas cambio "
        "pueblos indigenas afrodescendientes comunidades negras NARP raizales palenqueros "
        "territorios colectivos autonomia resguardos consejo comunitario "
        "igualdad oportunidades garantias etnias derechos territoriales "
        "colombia igualitaria diversa libre discriminacion "
        "lgbti construccion tejido social diverso orientacion sexual "
        "ninos ninas adolescentes protegidos primera infancia CDI ICBF "
        "crece generacion vida paz proteccion trayectoria vital "
        "personas discapacidad mundo sin barreras inclusion "
        "campesinado tierra educacion pertinencia participacion "
        "reivindicacion derechos grupos afectados personas dejan armas "
        "reintegracion excombatientes acuerdo paz "
        "colombianos exterior migrantes proteccion vinculos consulares"
    ),
}

_PND_CATALIZADORES = {
    "el agua": "ciclo agua cuencas paramos humedales agua potable saneamiento acueducto",
    "naturaleza viva": "conservacion naturaleza biodiversidad restauracion ecosistemas areas protegidas",
    "sistema de proteccion social": "VIH SIDA salud publica poblaciones vulnerables transferencias renta ciudadana migrantes",
    "superacion de privaciones": "salud garantista cobertura universal VIH SIDA pobreza extrema dignidad",
    "expansion de capacidades": "bienestar VIH SIDA lgbti discapacidad migrantes educacion empleo inclusion",
    "infraestructura social": "CDI primera infancia establecimientos educativos centros desarrollo",
    "transicion economica": "carbono neutralidad descarbonizacion clima bioeconomia economia baja carbono",
    "transicion energetica": "energia renovable solar eolica electrificacion eficiencia descarbonizacion",
    "cambio es con las mujeres": "mujeres rurales VIH SIDA salud reproductiva violencias liderazgo genero",
    "jovenes con derechos": "jovenes VIH SIDA ITS salud sexual empleo educacion cultura paz drogas",
    "crece la generacion": "ninos ninas adolescentes primera infancia nutricion CDI salud proteccion",
    "campesinado colombiano": "campesinado tierra economia campesina reforma agraria educacion rural",
    "reparacion efectiva": "victimas reparacion restitucion tierra vida seguridad desminado minas",
    "territorios que se transforman": "acuerdo paz pdet reincorporacion excombatientes comunidades",
    "dialogos de paz": "ELN dialogos paz conversaciones politicas cese fuego",
    "defensa integral": "defensa soberania fuerzas militares desminado humanitario minas antipersonal AICMA MAP MUSE artefactos explosivos liberacion tierras",
    "la cultura de paz en la cotidianidad": "desescalamiento violencia paz cotidianidad territorios comunidades cese fuego acuerdo paz desminado minas riesgo artefactos",
    "el dialogo: un camino": "dialogos paz ELN FARC grupos armados cese fuego negociacion desescalamiento acuerdos verificacion",
    "territorios que se transforman": "acuerdo paz teatro colon implementacion PDET reforma rural desminado minas victimas reincorporacion",
    "reparacion efectiva": "victimas conflicto reparacion restitucion tierra desminado minas antipersonal MAP liberacion tierras",
    "reivindicacion de los derechos": "excombatientes personas dejan armas tejido social reintegracion victimas conflicto desminado",
    "justicia transicional": "JEP verdad justicia reparacion no repeticion victimas reconciliacion",
    "reivindicacion": "excombatientes reintegracion paz total tejido social inclusion productiva",
    "reforma rural": "reforma rural integral mujer rural tierra catastro desminado retorno",
}

_ENCI_OBJETIVOS = {
    "1": (
        "cambio climatico justicia ambiental transformacion productiva "
        "ordenamiento territorial riesgo desastres alerta temprana "
        "catastro multipropósito SAT sistema administracion territorio "
        "saberes culturales tradicionales deforestacion crimen ambiental "
        "dialogos internacionales conservacion ecosistemas estrategicos "
        "mercado carbono justo equitativo incluyente "
        "acueducto saneamiento basico agua potable obras infraestructura "
        "acuerdo paris convenio diversidad biologica marco sendai NDC "
        "resiliencia climatica comunidades enfoque comunitario adaptacion "
        "energia renovable FNCER fuentes no convencionales transicion energetica "
        "descarbonizacion flota infraestructura abastecimiento energetico "
        "comercio exterior internacionalizacion politica comercial TLC "
        "cooperacion regulatoria acuerdos internacionales armonizacion "
        "investigacion innovacion bioeconomia ecosistemas naturales territorios "
        "infraestructura investigacion desarrollo tecnologico I+D+i "
        "industria digital tecnologias digitales adopcion "
        "economia circular reciclaje produccion limpia sostenible "
        "biodiversidad paramos bosques humedales cuencas hidrograficas "
        "adaptacion mitigacion cambio climatico paisajes forestales amazonia "
        "reindustrializacion cadenas valor competitividad exportaciones"
    ),
    "2": (
        "hambre atencion humanitaria seguridad alimentaria soberania alimentaria "
        "cadenas agricolas derecho alimentacion produccion local mercados campesinos "
        "riego distritos pequena escala rehabilitacion areas "
        "nutricion ninos ninas adolescentes NNA desnutricion malnutricion "
        "unidades centros recuperacion nutricional comunitaria ZRN "
        "comunidades indigenas NARP modelos atencion integral propios "
        "migrantes venezolanos refugiados retornados integracion atencion servicios "
        "VIH SIDA atencion migrantes venezolanos vulnerables salud "
        "regularizacion migratoria acceso servicios educacion salud trabajo vivienda "
        "lucha trata personas esclavitud moderna explotacion laboral sexual "
        "economia popular comunitaria solidaria rural campesina "
        "acopio distribucion comercializacion productos campesinos "
        "redes logisticas asociatividad cooperativas redes "
        "economias campesinas comunitarias solidarias rurales unidades productivas "
        "instrumentos financieros iniciativas economia popular credito "
        "asociatividad solidaria paz ASPP "
        "inclusion social productiva mujeres reincorporados discapacidad campesinos "
        "generacion ingresos trabajo decente emprendimiento "
        "seguridad alimentaria nutricion pobreza hambre acceso alimentos "
        "agricultura familiar pequenos productores produccion agropecuaria"
    ),
    "3": (
        "paz justicia desigualdad posconflicto acuerdo paz implementacion "
        "reforma rural integral mujer rural acceso tierra catastro fondo tierras "
        "desminado humanitario liberacion territorios minados AICMA "
        "accion contra minas antipersonal MAP MUSE artefactos explosivos "
        "retorno seguro comunidades reintegracion desminadores excombatientes "
        "empleo posconflicto construccion paz "
        "instrumentos planeacion territorial paz PNS planes nacionales sectoriales "
        "PNIS pdet zonas priorizadas planes sustitucion cultivos ilicitos "
        "vias regionales secundarias terciarias conectividad rural "
        "reincorporacion integral excombatientes firmantes paz ARN "
        "SIVJRNR JEP verdad justicia reparacion no repeticion "
        "sistema defensoria derechos humanos lideres sociales "
        "seguimiento verificacion garantias cumplimiento acuerdo paz "
        "acceso prioritario hogares oferta institucional transferencias "
        "infraestructura dotacion social equipamientos comunitarios "
        "liderazgos juveniles participacion social paz resolucion conflictos "
        "primera infancia ninez jovenes adultos mayores ciclo vital "
        "conectividad digital gobierno ciberseguridad telecomunicaciones "
        "bibliotecas museos archivos cultura patrimonio diversidad "
        "emergencias seguridad sistema integrado prevencion "
        "violencias basadas genero VBG feminicidio prevencion atencion ruta "
        "jurisdiccion especial indigena pueblos "
        "responsabilidad penal adolescente jovenes "
        "practicas culturales ancestrales interculturalidad lenguas nativas "
        "iniciativas productivas mujeres generacion ingresos "
        "participacion ciudadana electoral condiciones capacidades institucionales "
        "violencia politica mujeres diversas rurales etnicas lgbtiq discapacidad "
        "plan integral prevencion atencion acceso justicia VBG "
        "ciudades territorios seguros convivencia "
        "oferta cultural artistica deportiva recreativa inclusion "
        "planes reparacion colectiva victimas minas antipersonal rehabilitacion "
        "eficiencia institucional acuerdos comunidades etnias indigenas "
        "lgbti diversidad genero derechos humanos no discriminacion "
        "indigenas afrodescendientes etnias derechos territoriales territorios colectivos"
    ),
    "4": (
        "cooperacion internacional fortalecimiento institucional SNCI APC Colombia "
        "gestion datos sistemas informacion fondos cooperacion entidades "
        "herramientas innovacion digital georeferenciacion variables reportes "
        "sistemas registro informacion entidades territoriales captura datos "
        "analisis informacion cooperacion internacional universidades centros pensamiento "
        "plataformas datos abiertos acceso informacion publica "
        "difusion herramientas incentivos reporte ecosistema actores nacionales "
        "planes trabajo intersectoriales territoriales articulacion "
        "observatorio cooperacion internacional gestion conocimiento SNCI "
        "mecanismos comunicacion actores SNCI innovadora "
        "fortalecimiento capacidades gestion cooperacion actores ecosistema "
        "intercambios nacionales internacionales gestion eficaz cooperacion "
        "agendas regionales proyectos estrategicos concurrencia niveles gobierno "
        "coordinacion territorial estrategia regional llegada territorio "
        "participacion gobiernos locales iniciativas cooperacion generacion iniciativas "
        "buenas practicas territoriales cooperacion desarrollo pais "
        "visibilizacion gobiernos locales objetivos ENCI 2023 2026 "
        "enfoques derechos genero diferenciales formulacion aprobacion seguimiento "
        "fortalecimiento capacidades actores sociales privados formulacion "
        "acceso cooperacion sociedad civil especial enfasis comunidades indigenas "
        "sesiones comites SNCI plan accion "
        "actores tomadores decisiones recursos cooperacion 30 porciento "
        "sesiones anuales cooperacion sur-sur sur sur triangular bilateral "
        "fortalecimiento institucional confianza ciudadana recuperar "
        "lucha corrupcion entidades publicas nacionales territoriales "
        "seguimiento evaluacion contratacion compra publica mejoramiento "
        "interoperabilidad sistemas informacion tecnologias disruptivas "
        "coordinacion interinstitucional preventivo control investigacion sancion "
        "marco regulatorio denunciantes corrupcion proteccion "
        "acompanamiento asesoria asistencia tecnica territorial mejorar "
        "competencias servidores colaboradores publicos gestion compras publica"
    ),
}


# ── v23: Enriquecimiento por título de entrada ENCI (nivel fino) ──────────
# Resuelve que entradas genéricas como "3.2.5 Primera infancia y niñez"
# no deben puntuar alto para proyectos de VIH, desminado, etc.
_ENCI_TITULO_KEYWORDS = {
    # Objetivo 1 — Ambiente
    "1.1.1": "gestion riesgo administracion riesgo alerta temprana planificacion desastres",
    "1.1.2": "capacidades gobiernos locales ordenamiento planificacion territorial",
    "1.1.3": "informacion catastral capacidades territoriales cumplimiento",
    "1.1.4": "sistema administrativo territorio SAT catastro",
    "1.1.5": "reconocimiento capacidades metodologia",
    "1.2.1": "saberes culturales tradicionales deforestacion crimenes ambientales",
    "1.2.3": "dialogos internacionales cooperacion ecosistemas carbono justo",
    "1.2.4": "acueducto saneamiento basico agua potable obras infraestructura hidrica",
    "1.3.1": "acuerdo paris convenio biodiversidad marco sendai NDC metas clima",
    "1.3.2": "resiliencia climatica comunidades adaptacion cambio climatico",
    "1.3.3": "energia renovable FNCER transformacion energetica solar eolica",
    "1.3.4": "comercio exterior internacionalizacion politica comercial reindustrializacion",
    "1.3.5": "cooperacion regulatoria internacional IRC armonizacion normativa",
    "1.3.6": "investigacion innovacion bioeconomia ecosistemas I+D+i ciencia",
    "1.3.7": "infraestructura investigacion desarrollo tecnologico innovacion SNCI",
    "1.3.8": "politica espacial colombiana satelite tecnologia",
    "1.3.10": "industria digital tecnologias digitales adopcion TIC conectividad",
    "1.3.11": "descarbonizacion flota infraestructura combustibles fosiles",
    # Objetivo 2 — Humanitario / Alimentación / Migración
    "2.1.1": "cadenas agricolas derecho alimentacion produccion agropecuaria campesinos",
    "2.1.2": "riego rehabilitacion areas distritos pequena escala agricultura",
    "2.1.3": "nutricion ninos ninas adolescentes desnutricion malnutricion",
    "2.1.4": "unidades centros recuperacion nutricional comunitaria NNA",
    "2.1.5": "zonas recuperacion nutricional ZRN alimentacion ninos",
    "2.1.6": "modelos atencion integral propios comunidades indigenas NARP",
    "2.2.2": "migrantes venezolanos refugiados retornados integracion atencion VIH SIDA salud acceso servicios regularizacion",
    "2.2.3": "coordinacion sectorial nacional regional entidades territoriales",
    "2.2.5": "trata personas lucha explotacion esclavitud moderna trafico",
    "2.3.1": "acopio distribucion comercializacion economia popular mercados campesinos",
    "2.3.2": "redes logisticas asociatividad cooperativas redes campesinas",
    "2.3.3": "economias campesinas comunitarias solidarias rurales unidades productivas",
    "2.3.4": "instrumentos financieros economia popular credito emprendimiento",
    "2.3.5": "asociatividad solidaria paz ASPP cooperativas campesinas",
    "2.3.6": "inclusion social productiva mujeres reincorporados discapacidad campesinos ingresos",
    # Objetivo 3 — Paz / Justicia / Género
    "3.1.1": "reforma rural integral mujer rural tierra catastro desminado liberacion territorios",
    "3.1.2": "planeacion territorial paz planes nacionales sectoriales PNS",
    "3.1.3": "PNIS pdet zonas priorizadas sustitucion cultivos ilicitos coca",
    "3.1.4": "vias regionales secundarias terciarias conectividad rural",
    "3.1.5": "reincorporacion integral excombatientes firmantes paz ARN desminado empleo",
    "3.1.6": "SIVJRNR JEP verdad justicia reparacion no repeticion memoria",
    "3.1.9": "defensoria derechos humanos lideres sociales proteccion",
    "3.1.11": "seguimiento verificacion garantias cumplimiento acuerdo paz",
    "3.2.1": "acceso prioritario hogares oferta institucional transferencias servicios",
    "3.2.3": "infraestructura dotacion social equipamientos comunitarios rural",
    "3.2.4": "liderazgos juveniles participacion social paz resolucion conflictos",
    "3.2.5": "primera infancia ninez jovenes adultos mayores ciclo vital CDI ICBF proteccion ninos",
    "3.2.7": "conectividad digital gobierno ciberseguridad telecomunicaciones",
    "3.2.8": "bibliotecas museos archivos cultura patrimonio casas cultura",
    "3.2.9": "emergencias seguridad sistema integrado prevencion",
    "3.2.10": "violencias basadas genero VBG feminicidio prevencion atencion ruta justicia mujer",
    "3.2.11": "jurisdiccion especial indigena pueblos indigenas derecho propio",
    "3.2.12": "responsabilidad penal adolescente sistema adolescentes infractores",
    "3.2.13": "practicas culturales ancestrales interculturalidad lenguas nativas identidad",
    "3.3.1": "iniciativas productivas mujeres generacion ingresos rural urbano emprendimiento",
    "3.3.5": "participacion ciudadana electoral condiciones capacidades institucionales",
    "3.3.6": "violencia politica mujeres rurales etnicas lgbtiq discapacidad prevencion",
    "3.3.7": "plan integral prevencion atencion acceso justicia VBG genero mujer",
    "3.3.8": "ciudades territorios seguros convivencia seguridad ciudadana",
    "3.3.9": "oferta cultural artistica deportiva recreativa fisica inclusion",
    "3.3.10": "planes reparacion colectiva victimas minas antipersonal rehabilitacion fisica psicosocial",
    "3.3.11": "eficiencia institucional acuerdos comunidades etnias indigenas cumplimiento",
    # Objetivo 4 — Cooperación
    "4.1.1": "gestion datos sistemas informacion fondos cooperacion entidades",
    "4.1.2": "herramientas innovacion digital georeferenciacion variables reportes",
    "4.1.3": "sistemas registro informacion entidades territoriales captura datos",
    "4.1.4": "analisis informacion cooperacion internacional universidades centros pensamiento",
    "4.1.5": "plataformas datos abiertos acceso informacion publica transparencia",
    "4.1.6": "difusion herramientas incentivos reporte ecosistema actores",
    "4.2.1": "planes trabajo intersectoriales territoriales articulacion coordinacion",
    "4.2.2": "observatorio cooperacion internacional gestion conocimiento SNCI",
    "4.2.3": "mecanismos comunicacion actores SNCI innovadora",
    "4.2.4": "fortalecimiento capacidades gestion cooperacion actores ecosistema",
    "4.2.5": "intercambios nacionales internacionales gestion eficaz cooperacion sur-sur",
    "4.3.1": "agendas regionales proyectos estrategicos concurrencia niveles gobierno",
    "4.3.2": "coordinacion territorial estrategia regional llegada territorio mecanismo",
    "4.3.3": "participacion gobiernos locales iniciativas cooperacion generacion iniciativas",
    "4.3.4": "buenas practicas territoriales cooperacion desarrollo modalidades",
    "4.3.5": "visibilizacion gobiernos locales objetivos ENCI 2023 2026",
    "4.3.6": "enfoques derechos genero diferenciales formulacion aprobacion seguimiento",
    "4.3.7": "fortalecimiento capacidades actores sociales privados formulacion implementacion",
    "4.3.8": "acceso cooperacion sociedad civil indigenas afro mujeres jovenes",
    "4.4.1": "sesiones comites SNCI plan accion cooperacion",
    "4.4.2": "actores decisiones recursos cooperacion 30 por ciento",
    "4.4.3": "sesiones anuales cooperacion sur-sur triangular bilateral",
    "4.5.1": "fortalecimiento institucional confianza ciudadana recuperar motor cambio",
    "4.5.2": "lucha corrupcion entidades publicas nacionales territoriales cumplimiento",
    "4.5.3": "seguimiento evaluacion contratacion compra publica mejoramiento",
    "4.5.4": "interoperabilidad sistemas informacion tecnologias disruptivas",
    "4.5.5": "coordinacion interinstitucional preventivo control investigacion sancion",
    "4.5.6": "marco regulatorio denunciantes corrupcion proteccion",
    "4.5.7": "acompanamiento asesoria asistencia tecnica territorial mejorar servicios",
    "4.5.8": "competencias servidores colaboradores publicos gestion compras publica",
}

_SECTOR_KEYWORDS = {
    "Salud y Protección Social": (
        "Ministerio Salud Proteccion Social MSPS supersalud INVIMA INS IETS "
        "VIH SIDA HIV AIDS antirretroviral TARV ARV prep pep pvvih carga viral CD4 "
        "poblaciones clave hsh trabajo sexual transgenero udi migrantes venezolanos "
        "prevencion vih prueba rapida diagnostico salud publica epidemia "
        "tuberculosis hepatitis malaria ITS ETS enfermedades transmisibles "
        "cobertura universal salud mental nutricion mortalidad materna vacunacion "
        "plan nacional respuesta vih salud sexual reproductiva "
        "atencion medica enfermedades cronicas hospitales EPS IPS aseguramiento"
    ),
    "Defensa": (
        "Ministerio Defensa Nacional fuerzas militares ejercito policia nacional armada "
        "seguridad defensa soberania territorial orden publico "
        "desminado humanitario minas antipersonal MAP MUSE AICMA "
        "accion contra minas territorios minados liberacion tierras equipos multitarea "
        "posconflicto construccion paz grupos armados crimen organizado narcotrafico"
    ),
    "Interior": (
        "Ministerio Interior ley orden publico gobernabilidad autonomia territorial "
        "comunidades indigenas afrodescendientes etnias NARP territorios colectivos "
        "derechos humanos participacion ciudadana democratica "
        "victimas conflicto construccion paz convivencia "
        "migracion desplazamiento forzado retorno integracion"
    ),
    "Agricultura y Desarrollo Rural": (
        "Ministerio Agricultura MADR ADR UPRA AGROSAVIA banco agrario credito rural "
        "reforma agraria acceso tierra campesinos fondo tierras catastro rural "
        "seguridad alimentaria soberania produccion agropecuaria agricola pecuaria "
        "agricultura familiar pequenos productores cadenas valor agropecuarias "
        "cafe cacao panela ganaderia pesca acuicultura apicultura frutas hortalizas "
        "riego distritos agroecologia produccion sostenible mercados campesinos"
    ),
    "Ambiente y Desarrollo Sostenible": (
        "Ministerio Ambiente MADS CAR corporaciones autonomas regionales SINAP PNN "
        "biodiversidad ecosistemas paramos bosques humedales corales manglares "
        "cambio climatico adaptacion mitigacion NDC acuerdo paris "
        "economia circular reciclaje residuos produccion limpia cero contaminacion "
        "cuencas hidrograficas agua saneamiento gobernanza agua "
        "areas protegidas deforestacion restauracion REDD carbono"
    ),
    "Vivienda, Ciudad y Territorio": (
        "Ministerio Vivienda Ciudad Territorio MVT FNA "
        "acueducto alcantarillado agua potable saneamiento basico WASH "
        "vivienda digna deficit habitacional subsidios VIS VIP titulacion "
        "urbanismo servicios publicos comunidades rurales "
        "planta tratamiento potabilizacion agua segura "
        "ordenamiento urbano gestion suelo habitat"
    ),
    "Educación": (
        "Ministerio Educacion Nacional MEN ICETEX universidades IES "
        "educacion preescolar basica media superior calidad cobertura "
        "inclusion educativa primera infancia CDI desercion escolar docentes "
        "SENA formacion tecnica tecnologica aprendices certificacion "
        "becas creditos educativos competencias STEM ciencia tecnologia"
    ),
    "Trabajo": (
        "Ministerio Trabajo empleo formal informal desempleo OIT SENA "
        "derechos laborales seguridad social pension cesantias ARL UGPP "
        "economia popular trabajo digno migrantes venezolanos inclusion laboral "
        "discapacidad mujeres jovenes emprendimiento ingresos COLPENSIONES"
    ),
    "Igualdad y Equidad": (
        "Ministerio Igualdad Equidad Colombia genero lgbti diversidad sexual "
        "empoderamiento mujer derechos humanos no discriminacion "
        "etnias afrodescendientes indigenas discapacidad VIH SIDA vulnerables"
    ),
    "Relaciones Exteriores": (
        "Ministerio Relaciones Exteriores Cancilleria politica exterior diplomacia "
        "cooperacion internacional bilateral multilateral AOD donantes "
        "colombianos exterior diaspora migracion consulares retorno "
        "acuerdos tratados ONU organismos multilaterales BID PNUD USAID UE APC"
    ),
    "Comercio, Industria y Turismo": (
        "Ministerio Comercio MINCIT exportaciones internacionalizacion TLC "
        "competitividad pymes emprendimiento iNNpulsa ProColombia "
        "turismo sostenible ecoturismo economia circular cadenas valor clusters"
    ),
    "Justicia y del Derecho": (
        "Ministerio Justicia USPEC INPEC acceso justicia derechos "
        "sistema penitenciario rehabilitacion victimas "
        "justicia transicional reparacion lucha drogas narcotrafico "
        "defensoria publica juridica notariado registro civil"
    ),
    "Cultura": (
        "Ministerio Cultura colombia patrimonio cultural expresiones "
        "diversidad identidad pueblos indigenas afrodescendientes "
        "saberes ancestrales lenguas nativas interculturalidad "
        "artistas creadores cine musica danza teatro biblioteca"
    ),
    "Presidencia": (
        "Presidencia Republica paz total coordinacion interinstitucional "
        "posconflicto reincorporacion excombatientes DAPRE "
        "consejeria derechos humanos consejeria genero"
    ),
    "Sector Planeación": (
        "DNP planeacion politica publica evaluacion presupuesto inversion "
        "SINERGIA cooperacion politicas coordinacion estadistica"
    ),
    "Sector Inclusión Social": (
        "DPS Prosperidad Social ICBF Colombia Mayor Ingreso Social "
        "pobreza Familias en Accion Jovenes Accion transferencias "
        "VIH SIDA migrantes victimas vulnerabilidad PAPSIVI Unidad Victimas"
    ),
    "Ciencia, Tecnología e Innovación": (
        "Minciencias CTI investigacion I+D+i becas doctorados "
        "ecosistema ciencia tecnologia innovacion biotecnologia"
    ),
    "Minas y Energía": (
        "Ministerio Minas Energia MME mineria energia electrica gas "
        "transicion energetica renovables UPME CREG Ecopetrol ISA"
    ),
    "Transporte": (
        "Ministerio Transporte INVIAS ANI vias transporte infraestructura "
        "carreteras conectividad territorial logistica aeropuertos puertos"
    ),
    "Hacienda y Crédito Público": (
        "Ministerio Hacienda DIAN impuestos tributos presupuesto "
        "financiamiento deuda publica banca multilateral BID BM"
    ),
    "Deporte": (
        "Ministerio Deporte actividad fisica recreacion "
        "deporte escolar comunitario inclusion discapacidad "
        "infraestructura deportiva centros recreacion"
    ),
    "Estadística": (
        "DANE estadistica datos informacion censos "
        "indicadores medicion encuestas registros administrativos"
    ),
    "Inteligencia Estratégica": (
        "inteligencia contrainteligencia seguridad Estado "
        "informacion estrategica DNI"
    ),
    "Función Pública": (
        "funcion publica servicio civil empleo publico "
        "reforma institucional capacidades servidores "
        "gestion publica modernizacion Estado"
    ),
}


class DataLoader:
    """Carga y procesa marcos estratégicos con enriquecimiento estructural completo v22."""

    def __init__(self, ruta_excel):
        self.ruta_excel = Path(ruta_excel)

    def cargar_todos_los_marcos(self):
        from config import CONFIG
        catalogos = {}
        if not self.ruta_excel.exists():
            print(f"❌ No se encontró: {self.ruta_excel.absolute()}")
            return catalogos
        try:
            excel_file = pd.ExcelFile(self.ruta_excel)
            hojas = excel_file.sheet_names
            print(f"📂 Hojas disponibles en marcos: {hojas}")
            for marco, nombre_hoja in CONFIG["HOJAS_EXCEL"].items():
                hoja_real = self._encontrar_hoja(nombre_hoja, hojas)
                if hoja_real:
                    df_raw = pd.read_excel(self.ruta_excel, sheet_name=hoja_real)
                    df = self._limpiar_dataframe(df_raw, marco)
                    if not df.empty:
                        catalogos[marco] = df
                        print(f"✅ {marco}: {len(df)} registros cargados")
                        self._diagnostico_calidad(df, marco)
                    else:
                        print(f"⚠️  {marco}: Sin datos válidos")
                else:
                    print(f"❌ {marco}: Hoja '{nombre_hoja}' no encontrada. Disponibles: {hojas}")
            return catalogos
        except Exception as e:
            print(f"❌ Error cargando Excel: {e}")
            import traceback; traceback.print_exc()
            return catalogos

    def _encontrar_hoja(self, nb, hojas):
        def n(s):
            s=s.lower().strip()
            for a,b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u")]:
                s=s.replace(a,b)
            return s
        nb2=n(nb)
        for h in hojas:
            if n(h)==nb2 or nb2 in n(h): return h
        return None

    def _norm(self, s):
        s=s.lower()
        for a,b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
            s=s.replace(a,b)
        return s

    def _limpiar_dataframe(self, df, marco):
        df=df.dropna(how="all").copy()
        df.columns=df.columns.str.strip().str.lower().str.replace(r"\s+"," ",regex=True)
        col_titulo=self._encontrar_columna(df,["titulo","nombre","meta","objetivo",
                                               "indicador","descripcion breve","enunciado","item"])
        if col_titulo is None:
            print(f"⚠️  {marco}: sin columna título. Cols: {df.columns.tolist()}")
            return pd.DataFrame()
        col_desc=self._encontrar_columna(df,["descripcion","descripción","detalle","texto","contenido"])
        df_l=pd.DataFrame()
        df_l["titulo"]=df[col_titulo].astype(str).str.strip()
        # Limpiar BOM y caracteres invisibles
        df_l["titulo"]=df_l["titulo"].str.replace(r'^[\ufeff\u200b\s]+','',regex=True).str.strip()
        if col_desc and col_desc!=col_titulo:
            df_l["descripcion"]=df[col_desc].astype(str).str.strip()
        else:
            df_l["descripcion"]=df_l["titulo"]
        # Columnas contextuales
        ctx_keys={"ods","meta ods","pilar","eje","catalizador","sector","objetivo",
                  "linea_estrategica","eje_numero","subcatalizador","resultado","componente","linea"}
        for col in df.columns:
            if col not in {col_titulo, col_desc}:
                df_l[col]=df[col].fillna("").astype(str).str.strip()
        # Texto completo base
        partes=[df_l["titulo"],df_l["descripcion"]]
        for col in df_l.columns:
            if col in {"titulo","descripcion","texto_completo","id_original"}: continue
            if any(cx in col.lower() for cx in ctx_keys):
                vals=df_l[col].astype(str)
                ok=vals[(vals.str.len()>3)&(~vals.str.lower().str.match(r"^nan$|^$"))]
                if len(ok)>len(df_l)*0.15: partes.append(df_l[col])
        df_l["texto_completo"]=(
            pd.concat(partes,axis=1)
            .apply(lambda row:" ".join(p for p in row if p and str(p).lower() not in {"nan",""}),axis=1)
        )
        # Enriquecimiento
        df_l["texto_completo"]=df_l.apply(lambda row:self._enriquecer(row,marco),axis=1)
        df_l=df_l[(df_l["texto_completo"].str.len()>10)&(~df_l["texto_completo"].str.lower().str.match(r"^nan\s*$"))].copy()
        df_l["id_original"]=range(1,len(df_l)+1)
        return df_l.reset_index(drop=True)

    def _enriquecer(self, row, marco):
        base=str(row.get("texto_completo",""))
        extras=[]
        if marco=="PND":
            eje=str(row.get("eje","")).upper().strip()
            for eje_key,kw in _PND_EJES.items():
                if eje_key in eje or self._norm(eje_key[:25]) in self._norm(eje):
                    extras.append(kw); break
            cat=self._norm(str(row.get("catalizador","")))
            for ck,kw in _PND_CATALIZADORES.items():
                if ck in cat: extras.append(kw); break
        elif marco=="ENCI":
            obj=str(row.get("objetivo","")).strip()
            obj_num=obj[0] if obj else ""
            if obj_num in _ENCI_OBJETIVOS:
                extras.append(_ENCI_OBJETIVOS[obj_num])
            else:
                # fallback por título
                tn=self._norm(str(row.get("titulo","")))
                if any(k in tn for k in ["cambio climatico","clima","ambiente","energia","agua","biodiversidad"]):
                    extras.append(_ENCI_OBJETIVOS["1"])
                elif any(k in tn for k in ["hambre","aliment","nutric","migrant","humanitar","campesino"]):
                    extras.append(_ENCI_OBJETIVOS["2"])
                elif any(k in tn for k in ["paz","justicia","genero","desminado","victimas","reincorporacion","cultivos"]):
                    extras.append(_ENCI_OBJETIVOS["3"])
                else:
                    extras.append(_ENCI_OBJETIVOS["4"])
            # v23: enriquecimiento fino por número de entrada
            titulo_raw=str(row.get("titulo",""))
            import re as _re
            m=_re.search(r'(\d+\.\d+\.\d+)', titulo_raw)
            if m:
                entry_num=m.group(1).strip()
                if entry_num in _ENCI_TITULO_KEYWORDS:
                    extras.append(_ENCI_TITULO_KEYWORDS[entry_num])
        elif marco=="SECTORES":
            titulo=str(row.get("titulo",""))
            for sn,kw in _SECTOR_KEYWORDS.items():
                if self._norm(sn[:15]) in self._norm(titulo) or any(
                    self._norm(p) in self._norm(titulo) for p in sn.split() if len(p)>5
                ):
                    extras.append(kw); break
        if extras:
            return base+" | TEMAS: "+" ".join(extras)
        return base

    def _encontrar_columna(self, df, candidatas):
        for n in candidatas:
            for col in df.columns:
                if n in col.lower(): return col
        return None

    def _diagnostico_calidad(self, df, marco):
        ne=df["texto_completo"].str.contains("TEMAS:",na=False).sum()
        lm=df["texto_completo"].str.len().mean()
        print(f"   📊 {marco} | {len(df)} regs | long.media: {lm:.0f} | enriquecidos: {ne}/{len(df)}")
