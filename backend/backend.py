from fastapi import FastAPI, File, UploadFile, Query
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
from fastapi.responses import StreamingResponse
import torch

app = FastAPI()

# Dicionário com os caminhos dos modelos disponíveis
MODEL_PATHS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt"
}

# Carregar modelo padrão ao iniciar a API
current_model_name = "yolov8n"
model = YOLO(MODEL_PATHS[current_model_name])

# Rota para alterar o modelo de detecção
@app.post("/set_model/")
async def set_model(version: str = Query("yolov8n", enum=list(MODEL_PATHS.keys()))):
    global model, current_model_name

    if version in MODEL_PATHS and version != current_model_name:
        model = YOLO(MODEL_PATHS[version])  # Carrega o novo modelo selecionado
        current_model_name = version
        return {"message": f"Modelo alterado para {version}"}
    
    return {"message": f"O modelo já está definido como {version}"}

# Rota para processar a imagem
@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...), 
    model_name: str = Query("yolov8n", enum=list(MODEL_PATHS.keys())), # Adicionado model_name para garantir que seja recebido
    conf_threshold: float = Query(0.25, description="Confiança mínima"),
    border_size: int = Query(50, description="Espessura da borda"),
    border_color: str = Query("50,50,50", description="Cor da borda (R,G,B)"),
    font_scale: float = Query(0.7, description="Tamanho da fonte"),
    font_thickness: int = Query(2, description="Espessura da fonte"),
    text_color: str = Query("255,255,255", description="Cor do texto (R,G,B)"),
    background_color: str = Query("0,0,0", description="Cor do fundo do texto (R,G,B)"),
    background_alpha: float = Query(0.5, description="Transparência do fundo do texto (0 a 1)")
):
    """
    Processa uma imagem com YOLO e retorna a imagem anotada.
    Permite configurar borda, texto e fundo do texto.
    """
    global model, current_model_name

    try:
        # Se o modelo selecionado for diferente do atual, carregar o modelo correto
        if model_name != current_model_name:
            model = YOLO(MODEL_PATHS[model_name])
            current_model_name = model_name

        # Ler a imagem enviada pelo usuário
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Processar a imagem com YOLO
        results = model(image_cv)

        # Converter strings de cores para tuplas (R, G, B)
        border_color = tuple(map(int, border_color.split(",")))
        text_color = tuple(map(int, text_color.split(",")))
        background_color = tuple(map(int, background_color.split(",")))

        # Criar a borda ao redor da imagem
        border_cv = cv2.copyMakeBorder(
            image_cv, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=border_color
        )

        # Processar as detecções
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{model.names[int(box.cls[0])]} {conf:.2f}"

                    # Ajustar coordenadas devido à borda
                    x1 += border_size
                    y1 += border_size
                    x2 += border_size
                    y2 += border_size

                    # Desenhar a caixa ao redor do objeto
                    cv2.rectangle(border_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                   
                    # Configuração do texto
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    text_x, text_y = x1, y1 - 10

                    # Criar fundo para o texto
                    overlay = border_cv.copy()
                    cv2.rectangle(overlay, 
                                  (text_x, text_y - text_size[1] - 5), 
                                  (text_x + text_size[0] + 5, text_y + 5), 
                                  background_color, -1)
                    
                    cv2.addWeighted(overlay, background_alpha, border_cv, 1 - background_alpha, 0, border_cv)
                    # Adiciona configurações do texto
                    cv2.putText(border_cv, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Converter para JPEG e retornar como resposta
        _, encoded_image = cv2.imencode(".jpg", border_cv)
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}

     # Método Main
if __name__ == "main":
# if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port="8000",
        log_level="info"
    )

"""
## Instruções para configurar o ambiente e executar a aplicação

### 1. Criar e ativar o ambiente virtual
```bash
python -m venv venv  # Criar ambiente virtual
source venv/bin/activate  # Ativar no Linux/macOS
venv\Scripts\activate  # Ativar no Windows
```

### 2. Instalar dependências
```bash
é sempre bom veriricar
python.exe -m pip install --upgrade pip

pip install torch fastapi uvicorn opencv-python pillow numpy ultralytics
pip install "fastapi[standard]"
```

### 3. Entrar na pasta da API
```bash
cd backend
pip install "fastapi[standard]"
```

### 4. Executar a API
```bash
venv\Scripts\activate
fastapi dev backend.py
```

A API estará disponível em `http://127.0.0.1:8000/docs` para testes via Swagger UI.
"""
