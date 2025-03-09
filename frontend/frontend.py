import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Detecção com YOLO", page_icon="🤖")

st.title("Detecção de objetos com YOLO")
st.write("Faça o upload da imagem para detecção de objetos:")



# Configuração do backend

# Cria um campo de entrada de texto do link do seu back-end
backend_URL = st.text_input("SEU_BACKEND_URL:") # http://127.0.0.1:8000/upload/
st.write("Exemplo: http://127.0.0.1:8000/upload/")
API_URL = backend_URL # Substitua pelo seu backend
 
# Opção para selecionar o modelo YOLO
model_options = {
    "YOLOv8 Nano (Rápido)": "yolov8n",
    "YOLOv8 Small (Médio)": "yolov8s",
    "YOLOv8 Medium (Boa precisão)": "yolov8m",
    "YOLOv8 Large (Alta precisão)": "yolov8l",
    "YOLOv8 X (Máxima precisão)": "yolov8x"
}
# Seção de configurações avançadas
with st.expander("⚙️ Configurações Opcionais"):

    st.write("🤖 Configurações do modelo")
    model_name = st.selectbox("Escolha o modelo YOLO:", list(model_options.keys()))
    selected_model = model_options[model_name]
    # Ajuste do nível de confiança mínima
    confidence = st.slider("Nível de confiança mínima:", 0.0, 1.0, 0.25, 0.05)

    st.write("🎨 Configurações da Borda")
    border_size = st.slider("Espessura da borda", min_value=0, max_value=100, value=50)
    border_color = st.color_picker("Escolha a cor da borda", "#323232")


    st.write("📝 Configurações do Texto")
    font_scale = st.slider("Tamanho da fonte do texto", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    font_thickness = st.slider("Espessura da fonte", min_value=1, max_value=5, value=2)
    text_color = st.color_picker("Cor do texto", "#FFFFFF")

    background_color = st.color_picker("Cor do fundo do texto", "#000000")
    background_alpha = st.slider("Transparência do fundo do texto", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Upload da imagem
uploaded_image = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])


# Converter cores para formato (R,G,B)
def hex_to_rgb(hex_color):
    return ",".join(map(str, Image.new("RGB", (1, 1), hex_color).getpixel((0, 0))))

border_color_rgb = hex_to_rgb(border_color)
text_color_rgb = hex_to_rgb(text_color)
background_color_rgb = hex_to_rgb(background_color)

# Mostrar a imagem carregada
if uploaded_image is not None:
    st.write("Imagem carregada")
    st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)

# Botão para enviar
if st.button("Processar imagem 🤖"):
    if uploaded_image:
        files = {"file": uploaded_image.getvalue()}
        params = {"confidence": confidence, "model_name": selected_model,
            "border_size": border_size,
            "border_color": border_color_rgb,
            "font_scale": font_scale,
            "font_thickness": font_thickness,
            "text_color": text_color_rgb,
            "background_color": background_color_rgb,
            "background_alpha": background_alpha
        }

        response = requests.post(API_URL, files=files, params=params)

        if response.status_code == 200:
            # Exibir a imagem processada
            result_image = Image.open(io.BytesIO(response.content))
            st.write("Imagem processada")
            st.image(result_image, caption="Imagem Processada", use_column_width=True)
        else:
            st.error("Erro ao processar a imagem. Verifique se o backend está rodando.")

"""
## Instruções para configurar o ambiente e executar a aplicação

### 1. Criar e ativar o ambiente virtual
```bash
python -m venv venv  # Criar ambiente virtual
source venv/bin/activate  # Ativar no Linux/macOS
venv\Scripts\activate  # Ativar no Windows
```

### 2. Instalar dependências
é sempre bom veriricar
```bash
python.exe -m pip install --upgrade pip
```

Front-end
```bash
pip install streamlit requests
```

Back-end
```bash
pip install torch fastapi uvicorn opencv-python pillow numpy ultralytics
```

### 3. Entrar na pasta da frontend
```bash
cd frontend
```

### 4. Executar o streamlit
```bash
streamlit run frontend.py
```

### 5. Entrar na pasta da API
```bash
cd backend
pip install "fastapi[standard]"
```

### 6. Executar a API
```bash
venv\Scripts\activate
fastapi dev backend.py
```

A página streamlit estará disponível em `http://localhost:8501/`
A API estará disponível em `http://127.0.0.1:8000/docs` para testes via Swagger UI.
"""
