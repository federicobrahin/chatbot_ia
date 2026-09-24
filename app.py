import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import google.generativeai as genai
import re

# ==========================================
# 1. MOTOR DE INFERENCIA DIFUSO (SED)
# ==========================================
@st.cache_resource
def crear_sistema_difuso():
    tiempo = ctrl.Antecedent(np.arange(0, 121, 1), 'tiempo') 
    habilidad = ctrl.Antecedent(np.arange(1, 11, 1), 'habilidad') 
    ingredientes = ctrl.Antecedent(np.arange(1, 11, 1), 'ingredientes') 
    complejidad = ctrl.Consequent(np.arange(0, 101, 1), 'complejidad') 

    tiempo.automf(names=['poco', 'medio', 'mucho'])
    habilidad.automf(names=['novato', 'aficionado', 'experto'])
    ingredientes.automf(names=['pocos', 'varios', 'muchos'])
    complejidad.automf(names=['basica', 'elaborada', 'gourmet'])

    reglas = [
        ctrl.Rule(tiempo['poco'] & habilidad['novato'] & ingredientes['pocos'], complejidad['basica']),
        ctrl.Rule(tiempo['poco'] & habilidad['novato'] & ingredientes['varios'], complejidad['basica']),
        ctrl.Rule(tiempo['poco'] & habilidad['novato'] & ingredientes['muchos'], complejidad['basica']),
        ctrl.Rule(tiempo['poco'] & habilidad['aficionado'] & ingredientes['pocos'], complejidad['basica']),
        ctrl.Rule(tiempo['poco'] & habilidad['aficionado'] & ingredientes['varios'], complejidad['basica']),
        ctrl.Rule(tiempo['poco'] & habilidad['aficionado'] & ingredientes['muchos'], complejidad['elaborada']),
        ctrl.Rule(tiempo['poco'] & habilidad['experto'] & ingredientes['pocos'], complejidad['basica']),
        ctrl.Rule(tiempo['poco'] & habilidad['experto'] & ingredientes['varios'], complejidad['elaborada']),
        ctrl.Rule(tiempo['poco'] & habilidad['experto'] & ingredientes['muchos'], complejidad['elaborada']),
        
        ctrl.Rule(tiempo['medio'] & habilidad['novato'] & ingredientes['pocos'], complejidad['basica']),
        ctrl.Rule(tiempo['medio'] & habilidad['novato'] & ingredientes['varios'], complejidad['basica']),
        ctrl.Rule(tiempo['medio'] & habilidad['novato'] & ingredientes['muchos'], complejidad['elaborada']),
        ctrl.Rule(tiempo['medio'] & habilidad['aficionado'] & ingredientes['pocos'], complejidad['basica']),
        ctrl.Rule(tiempo['medio'] & habilidad['aficionado'] & ingredientes['varios'], complejidad['elaborada']),
        ctrl.Rule(tiempo['medio'] & habilidad['aficionado'] & ingredientes['muchos'], complejidad['elaborada']),
        ctrl.Rule(tiempo['medio'] & habilidad['experto'] & ingredientes['pocos'], complejidad['elaborada']),
        ctrl.Rule(tiempo['medio'] & habilidad['experto'] & ingredientes['varios'], complejidad['elaborada']),
        ctrl.Rule(tiempo['medio'] & habilidad['experto'] & ingredientes['muchos'], complejidad['gourmet']),
        
        ctrl.Rule(tiempo['mucho'] & habilidad['novato'] & ingredientes['pocos'], complejidad['basica']),
        ctrl.Rule(tiempo['mucho'] & habilidad['novato'] & ingredientes['varios'], complejidad['elaborada']),
        ctrl.Rule(tiempo['mucho'] & habilidad['novato'] & ingredientes['muchos'], complejidad['elaborada']),
        ctrl.Rule(tiempo['mucho'] & habilidad['aficionado'] & ingredientes['pocos'], complejidad['elaborada']),
        ctrl.Rule(tiempo['mucho'] & habilidad['aficionado'] & ingredientes['varios'], complejidad['elaborada']),
        ctrl.Rule(tiempo['mucho'] & habilidad['aficionado'] & ingredientes['muchos'], complejidad['gourmet']),
        ctrl.Rule(tiempo['mucho'] & habilidad['experto'] & ingredientes['pocos'], complejidad['elaborada']),
        ctrl.Rule(tiempo['mucho'] & habilidad['experto'] & ingredientes['varios'], complejidad['gourmet']),
        ctrl.Rule(tiempo['mucho'] & habilidad['experto'] & ingredientes['muchos'], complejidad['gourmet'])
    ]

    sistema_control = ctrl.ControlSystem(reglas)
    return ctrl.ControlSystemSimulation(sistema_control)

# ==========================================
# 2. INTERFAZ Y CONEXIÓN A GEMINI
# ==========================================
st.set_page_config(page_title="Sistema Experto Culinario", page_icon="⚙️")

# --- BARRA LATERAL CON INFORMACIÓN DEL PROYECTO ---
with st.sidebar:
    st.subheader("📚 Inteligencia Artificial")
    st.markdown("**Profesor:**")
    st.markdown("Mag. Ing. Mario Marcelo Figueroa de la Cruz")
    
    st.markdown("**Integrantes del grupo:**")
    st.markdown("""
    - Antúnez Ruiz Huidobro, Facundo
    - Brahin, Federico Tomás
    - Cáceres Prado, Martín
    - Matos Villalba, Luis Humberto
    - Rodríguez Marat, Martín
    """)
    st.caption("Universidad del Norte Santo Tomás de Aquino (UNSTA)")

# --- TÍTULO PRINCIPAL ---
st.title("⚙️ Sistema Experto Difuso Integrado con LLM")
st.markdown("Determinación de complejidad culinaria mediante Lógica Difusa y generación de recetas asistida por Inteligencia Artificial.")
st.divider() 

# Configuración de estado inicial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": "Bienvenido al Sistema Experto. Por favor, ingrese los ingredientes disponibles (separados por coma o espacio):"}]
if "estado_chat" not in st.session_state:
    st.session_state.estado_chat = "pidiendo_ingredientes"
if "datos" not in st.session_state:
    st.session_state.datos = {}

# Mostrar historial
for msj in st.session_state.mensajes:
    with st.chat_message(msj["role"]):
        st.markdown(msj["content"])

prompt = st.chat_input("Ingrese los datos solicitados...")

if prompt:
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.estado_chat == "pidiendo_ingredientes":
        st.session_state.datos['ingredientes_lista'] = [i.strip() for i in re.split(r',|\sy\s|\s+', prompt) if i.strip()]
        st.session_state.datos['cant_ingredientes'] = len(st.session_state.datos['ingredientes_lista'])
        
        respuesta = "Datos registrados. A continuación, ingrese el tiempo disponible para la preparación (en minutos):"
        st.session_state.estado_chat = "pidiendo_tiempo"

    elif st.session_state.estado_chat == "pidiendo_tiempo":
        try:
            st.session_state.datos['tiempo'] = int(prompt.strip())
            respuesta = "Correcto. Por último, indique su nivel de habilidad culinaria en una escala del 1 al 10 (1 = Principiante, 10 = Experto):"
            st.session_state.estado_chat = "pidiendo_habilidad"
        except ValueError:
            respuesta = "Error de validación: Por favor, ingrese únicamente un valor numérico entero."

    elif st.session_state.estado_chat == "pidiendo_habilidad":
        try:
            st.session_state.datos['habilidad'] = int(prompt.strip())
            
            # --- EVALUACIÓN DIFUSA ---
            simulador = crear_sistema_difuso()
            simulador.input['tiempo'] = st.session_state.datos['tiempo']
            simulador.input['habilidad'] = st.session_state.datos['habilidad']
            simulador.input['ingredientes'] = min(st.session_state.datos['cant_ingredientes'], 10) 
            
            simulador.compute()
            puntaje = simulador.output['complejidad']
            
            if puntaje < 35: categoria = "Baja (Receta Básica)"
            elif puntaje < 70: categoria = "Media (Receta Elaborada)"
            else: categoria = "Alta (Alta Cocina / Desafiante)"

            # --- LLAMADA A GEMINI CON SECRETS, SPINNER Y TIMEOUT EXTENDIDO ---
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=api_key)
                
                modelo = genai.GenerativeModel('gemini-3.6-flash')
                
                with st.spinner('Procesando inferencia lógica y comunicando con el modelo de lenguaje...'):
                    prompt_gemini = f"""
                    Actúe como un asistente culinario profesional. El usuario solicita una receta con los siguientes ingredientes: {', '.join(st.session_state.datos['ingredientes_lista'])}.
                    
                    REGLA DE VALIDACIÓN ESTRICTA: Analice si TODOS los elementos ingresados son ingredientes culinarios reales y comestibles. 
                    Si el usuario ingresó objetos no comestibles (ej. herramientas, materiales de construcción, tóxicos) o cadenas sin sentido:
                    Indique formalmente que el sistema experto solo procesa ingredientes alimenticios y solicite al usuario que reinicie la consulta. No genere ninguna receta.

                    Si los ingredientes son válidos, proceda bajo estas directivas:
                    El motor de inferencia difuso determinó que la receta debe tener una complejidad de categoría: {categoria} (Valor de defusificación: {puntaje}/100).
                    Tiempo disponible: {st.session_state.datos['tiempo']} minutos. Nivel de habilidad del usuario: {st.session_state.datos['habilidad']}/10.
                    Redacte una receta formal, clara y estructurada que cumpla con estos parámetros. Utilice formato Markdown con viñetas y títulos adecuados.
                    """
                    
                    # Timeout extendido a 60 segundos para evitar el error 499
                    respuesta_gemini = modelo.generate_content(prompt_gemini, request_options={"timeout": 60})
                    receta = respuesta_gemini.text
                    
            except KeyError:
                receta = "❌ Error interno: No se ha configurado la variable de entorno GEMINI_API_KEY en los secretos del servidor."
            except Exception as e:
                receta = f"❌ Error de comunicación con la API externa: La operación tardó demasiado o fue interrumpida. Detalle técnico: {e}"

            respuesta = f"""
            🧠 **Resultados del Motor de Inferencia Difuso:**
            - **Valor de salida (Defusificación):** {puntaje:.2f}/100
            - **Categoría asignada:** {categoria}
            
            👨‍🍳 **Resolución del Modelo de Lenguaje (LLM):**
            {receta}
            
            *(Ingrese la palabra "reiniciar" para ejecutar una nueva consulta)*
            """
            st.session_state.estado_chat = "terminado"
            
        except ValueError:
            respuesta = "Error de validación: Por favor, ingrese un número entero del 1 al 10."
            
    elif st.session_state.estado_chat == "terminado" and prompt.lower() == "reiniciar":
        st.session_state.mensajes = [{"role": "assistant", "content": "Sistema reiniciado. Por favor, ingrese los ingredientes disponibles (separados por coma o espacio):"}]
        st.session_state.estado_chat = "pidiendo_ingredientes"
        st.rerun()
    else:
        respuesta = "Comando no reconocido. Ingrese 'reiniciar' para iniciar un nuevo proceso."

    if st.session_state.estado_chat != "terminado" or prompt.lower() != "reiniciar":
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)
