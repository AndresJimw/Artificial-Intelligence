import streamlit as st
import pandas as pd
from openai import OpenAI


st.set_page_config(
    page_title="Chatbot de Posgrados en Línea – Pichincha",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Chatbot – Posgrados en Línea (Pichincha)")
st.write("Sube tu CSV filtrado y haz preguntas sobre la oferta académica.")


with st.sidebar:
    st.header("Configuración")

    api_key = st.text_input("🔑 OpenAI API key", type="password")

    model_name = st.selectbox(
        "Modelo",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    temperature = st.slider("Creatividad", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📄 Sube el archivo CSV de posgrados filtrado",
        type=["csv"],
    )


if not api_key:
    st.info("Introduce tu API key para continuar.")
    st.stop()

if uploaded_file is None:
    st.info("Sube el CSV filtrado para continuar.")
    st.stop()


try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error cargando el CSV: {e}")
    st.stop()

for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

data_context = df.to_csv(index=False)

client = OpenAI(api_key=api_key)


SYSTEM_PROMPT = f"""
Eres un asistente especializado en la oferta de programas de POSGRADO EN LÍNEA
de la provincia de Pichincha (Ecuador).

Tu conocimiento proviene EXCLUSIVAMENTE de la siguiente tabla CSV:

```csv
{data_context}
```

REGLAS IMPORTANTES:
1. Solo responde preguntas sobre los programas presentes en el CSV.
2. Si la pregunta está fuera de este contexto, responde exactamente:
   "Lo siento, solo puedo responder preguntas sobre la oferta de posgrados en línea de Pichincha en esta base de datos."
3. No inventes información adicional.
4. Responde siempre en español.
"""


if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_prompt = st.chat_input("Escribe tu pregunta...")

if user_prompt:
    st.session_state.chat.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(st.session_state.chat)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)

        st.session_state.chat.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Error al llamar a OpenAI: {e}")
