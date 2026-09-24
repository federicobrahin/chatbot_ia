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
st.set_page_config(page_title="ChefBot Inteligente", page_icon="👨‍🍳")

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
st.title("👨‍🍳 ChefBot Experto Difuso con IA")
st.markdown("Calculo la complejidad del plato mediante Lógica Difusa y genero la receta paso a paso utilizando Inteligencia Artificial.")
st.divider() 

# Configuración de estado inicial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": "¡Hola! Bienvenido a ChefBot. ¿Qué ingredientes tenés disponibles? (Podés separarlos por coma o espacios, ej: pollo cebolla papa)"}]
if "estado_chat" not in st.session_state:
    st.session_state.estado_chat = "pidiendo_ingredientes"
if "datos" not in st.session_state:
    st.session_state.datos = {}

# Mostrar historial
for msj in st.session_state.mensajes:
    with st.chat_message(msj["role"]):
        st.markdown(msj["content"])

prompt = st.chat_input("Escribe tu respuesta aquí...")

if prompt:
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.estado_chat == "pidiendo_ingredientes":
        st.session_state.datos['ingredientes_lista'] = [i.strip() for i in re.split(r',|\sy\s|\s+', prompt) if i.strip()]
        st.session_state.datos['cant_ingredientes'] = len(st.session_state.datos['ingredientes_lista'])
        
        respuesta = "¡Excelente! Ahora decime, ¿cuánto **tiempo libre** tenés para cocinar hoy? (en minutos, ej: 30)"
        st.session_state.estado_chat = "pidiendo_tiempo"

    elif st.session_state.estado_chat == "pidiendo_tiempo":
        try:
            st.session_state.datos['tiempo'] = int(prompt.strip())
            respuesta = "Perfecto. Por último, del 1 al 10, ¿qué tan **hábil** sos en la cocina? (1 = Principiante, 10 = Experto)"
            st.session_state.estado_chat = "pidiendo_habilidad"
        except ValueError:
            respuesta = "Por favor, ingresá únicamente un número entero (ej: 45)."

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
            
            if puntaje < 35: categoria = "Básica y rápida 🟢"
            elif puntaje < 70: categoria = "Elaborada 🟡"
            else: categoria = "Gourmet / Desafiante 🔴"

            # --- LLAMADA A GEMINI OPTIMIZADA PARA VELOCIDAD ---
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=api_key)
                
                modelo = genai.GenerativeModel('gemini-3.6-flash')
                
                with st.spinner('👨‍🍳 El chef está analizando los datos y creando tu receta... Un momento por favor.'):
                    prompt_gemini = f"""
                    Sos un chef profesional. El usuario te pidió cocinar con estos ingredientes: {', '.join(st.session_state.datos['ingredientes_lista'])}.
                    
                    REGLA ESTRICTA: Analizá si TODOS los ingredientes son comestibles y reales. 
                    Si el usuario ingresó objetos no comestibles (ej: tornillos, maderas):
                    Respondé educadamente que el sistema solo procesa alimentos y pedí que reinicie. No des ninguna receta.

                    Si los ingredientes SON comestibles:
                    El sistema experto determinó una complejidad: {categoria} (Puntaje: {puntaje}/100).
                    Tiempo disponible: {st.session_state.datos['tiempo']} minutos. Habilidad: {st.session_state.datos['habilidad']}/10.
                    
                    Dame una receta paso a paso que cumpla con estos criterios. Sé directo, amigable y usá viñetas.
                    MUY IMPORTANTE: Sé BREVE y conciso. Evitá textos muy largos.
                    """
                    
                    # Forzamos a la IA a responder más rápido limitando la longitud de la respuesta
                    respuesta_gemini = modelo.generate_content(
                        prompt_gemini,
                        generation_config={"max_output_tokens": 500, "temperature": 0.5},
                        request_options={"timeout": 90}
                    )
                    receta = respuesta_gemini.text
                    
            except KeyError:
                receta = "❌ Error: No se encontró la API Key en los secretos de Streamlit (st.secrets)."
            except Exception as e:
                receta = f"❌ Error de conexión: La inteligencia artificial tardó demasiado en responder. (Detalle: {e})"

            respuesta = f"""
            🧠 **Diagnóstico del Sistema Experto Difuso:**
            - Complejidad: **{puntaje:.2f}/100** ({categoria})
            
            👨‍🍳 **Receta Generada por IA:**
            {receta}
            
            *(Escribí "reiniciar" si querés probar con otra consulta)*
            """
            st.session_state.estado_chat = "terminado"
            
        except ValueError:
            respuesta = "Por favor, ingresá un número del 1 al 10."
            
    elif st.session_state.estado_chat == "terminado" and prompt.lower() == "reiniciar":
        st.session_state.mensajes = [{"role": "assistant", "content": "¡Vamos de nuevo! ¿Qué cocinamos hoy? ¿Qué ingredientes tenés disponibles?"}]
        st.session_state.estado_chat = "pidiendo_ingredientes"
        st.rerun()
    else:
        respuesta = "Comando no reconocido. Escribí 'reiniciar' para volver a empezar."

    if st.session_state.estado_chat != "terminado" or prompt.lower() != "reiniciar":
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)
