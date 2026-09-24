import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import google.generativeai as genai
import re  # Agregamos esta librería para leer mejor los ingredientes

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
st.set_page_config(page_title="Chefbot inteligente", page_icon="👨‍🍳")

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

# --- TÍTULO PRINCIPAL PROLIJO ---
st.title("👨‍🍳 Chefbot experto difuso con IA")
st.markdown("Calculo la complejidad del plato mediante lógica difusa y genero la receta paso a paso utilizando inteligencia artificial.")
st.divider() 

# Configuración de estado inicial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": "¡Hola! ¿Qué cocinamos hoy? Decime, ¿qué ingredientes tenés en la heladera? Podés separarlos por coma o espacios (Ej: pollo cebolla papa)"}]
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
        # Usamos RegEx para separar por comas, espacios o la letra "y"
        st.session_state.datos['ingredientes_lista'] = [i.strip() for i in re.split(r',|\sy\s|\s+', prompt) if i.strip()]
        st.session_state.datos['cant_ingredientes'] = len(st.session_state.datos['ingredientes_lista'])
        
        respuesta = "¡Anotado! Ahora decime, ¿cuánto **tiempo libre** tenés para cocinar hoy? (minutos, ej: 30)"
        st.session_state.estado_chat = "pidiendo_tiempo"

    elif st.session_state.estado_chat == "pidiendo_tiempo":
        try:
            st.session_state.datos['tiempo'] = int(prompt.strip())
            respuesta = "Perfecto. Por último, del 1 al 10, ¿qué tan **hábil** sos en la cocina? (1=quemo el agua, 10=MasterChef)"
            st.session_state.estado_chat = "pidiendo_habilidad"
        except ValueError:
            respuesta = "Por favor, poné solo un número (ej: 45)."

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

            # --- LLAMADA A GEMINI CON SECRETS ---
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=api_key)
                
                modelo = genai.GenerativeModel('gemini-3.6-flash')
                
                prompt_gemini = f"""
                Sos un chef profesional. El usuario te pidió cocinar con estos ingredientes: {', '.join(st.session_state.datos['ingredientes_lista'])}.
                
                REGLA ESTRICTA: Primero, analizá si TODOS los ingredientes ingresados son comestibles y reales. 
                Si el usuario ingresó objetos no comestibles (ej: tornillos, piedras, madera, veneno, etc.) o cosas sin sentido:
                NO des ninguna receta. Respondé con un tono gracioso diciendo que sos un Chef, no un ferretero ni un mago, y decile que escriba "reiniciar" para intentar con comida de verdad. Ignorá el resto de las instrucciones.

                Si los ingredientes SON comestibles, seguí estas instrucciones:
                Un sistema experto difuso determinó que la receta debe tener una complejidad: {categoria} (Puntaje: {puntaje}/100).
                El usuario tiene {st.session_state.datos['tiempo']} minutos libres y un nivel de habilidad de {st.session_state.datos['habilidad']}/10.
                Dame una receta paso a paso que cumpla con estos criterios. Sé directo, amigable y estructurá la respuesta con títulos y viñetas.
                """
                
                respuesta_gemini = modelo.generate_content(prompt_gemini)
                receta = respuesta_gemini.text
            except KeyError:
                receta = "❌ Error: No se encontró la API Key en los secretos de Streamlit (st.secrets)."
            except Exception as e:
                receta = f"❌ Error al conectar con Gemini: {e}"

            respuesta = f"""
            🧠 **Diagnóstico del sistema experto difuso:**
            - Complejidad: **{puntaje:.2f}/100** ({categoria})
            
            👨‍🍳 **Receta sugerida:**
            {receta}
            
            *(Escribí "reiniciar" si querés probar con otra cosa)*
            """
            st.session_state.estado_chat = "terminado"
            
        except ValueError:
            respuesta = "Poné solo un número del 1 al 10 por favor."
            
    elif st.session_state.estado_chat == "terminado" and prompt.lower() == "reiniciar":
        st.session_state.mensajes = [{"role": "assistant", "content": "¡Vamos de nuevo! ¿Qué cocinamos hoy? ¿Qué ingredientes tenés?"}]
        st.session_state.estado_chat = "pidiendo_ingredientes"
        st.rerun()
    else:
        respuesta = "Escribí 'reiniciar' para volver a empezar."

    if st.session_state.estado_chat != "terminado" or prompt.lower() != "reiniciar":
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)
