import streamlit as st
from neuron import Neuron

st.set_page_config(layout = "wide")
st.image("img/neurona_transparente.png", width=300)

# Title
st.title("Simulador de Neurona")

# Contenido en función de la pestaña seleccionada
    # Personalizar el tamaño del contenido de la pestaña 1
st.markdown("<style>h1 { font-size: 28px; }</style>", unsafe_allow_html=True)
st.title("Una neurona con una entrada y un peso")
    
E = st.slider("Elige el número de entradas/pesos que tendrá la neurona", 1, 10, key="s1")

st.subheader("Pesos")

colW = st.columns(E)
w = []
x = []

for i in range(0,E):
    with colW[i]:
        w.append(st.number_input(f"w{i}", key=f"nw{i}"))
st.text(f"w = {w}")

st.subheader("Entradas")

colX = st.columns(E)

for i in range(0,E):
    with colX[i]:
        x.append(st.number_input(f"x{i}", key=f"nx{i}"))
st.text("x = {x}")

col1, col2= st.columns(2)

with col1:
    st.subheader("Sesgo")
    b = st.number_input("Introduzca el valor del sesgo", key="n3")

    
with col2:
    st.subheader("Función de activación")
    f = st.selectbox("Elige la función de activación",("Sigmoide", "ReLU", "Tangente Hiperbólica")) 

if st.button('Calcular la salida', key='b1'):
    n1 = Neuron(weights=w, bias=b, func=f)
    output = n1.run(input_data=x)
    st.text(f"La salida de la neurona es: {output}")