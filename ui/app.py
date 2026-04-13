import time
import requests
import streamlit as st


st.set_page_config(page_title="DGA Classifier", page_icon="🌐", layout="centered")

DEFAULT_ENDPOINT = "http://192.168.1.60:8085/serve/dga-classifier/7"

st.title("DGA Domain Classifier")
st.write("Введите доменное имя и получите предсказание от ClearML Serving endpoint.")

with st.sidebar:
    st.header("Settings")
    endpoint_url = st.text_input("Serving endpoint URL", value=DEFAULT_ENDPOINT)

domain = st.text_input("Domain", placeholder="e.g. google.com")

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if not domain.strip():
        st.warning("Введите доменное имя.")
    else:
        payload = {"domain": domain.strip()}
        try:
            start = time.perf_counter()
            response = requests.post(
                endpoint_url,
                json=payload,
                timeout=10,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            response.raise_for_status()
            result = response.json()

            label = result.get("label")

            st.success("Prediction received")
            col1, col2 = st.columns(2)
            col1.metric("Label", str(label))
            col2.metric("Latency", f"{latency_ms:.2f} ms")

            with st.expander("Response JSON"):
                st.json(result)

        except requests.exceptions.RequestException as e:
            st.error("Endpoint недоступен или вернул ошибку.")
            st.code(str(e))
        except ValueError:
            st.error("Не удалось разобрать JSON-ответ от endpoint.")
