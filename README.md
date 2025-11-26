# Detección de Emociones en Tiempo Real para Dispositivos de Bajos Recursos

## Introducción

Este proyecto tiene como objetivo desarrollar un sistema robusto y eficiente para la detección de emociones faciales en tiempo real, con un enfoque específico en la optimización del rendimiento para su implementación en dispositivos de bajos recursos (como Raspberry Pi 5, microcontroladores avanzados o teléfonos móviles).

Se utiliza el dataset FER2013 y se exploran tres arquitecturas de Redes Neuronales Convolucionales (CNN) diferentes para encontrar el equilibrio ideal entre precisión, velocidad de inferencia y tamaño del modelo.

---

## Arquitecturas Implementadas

Se han desarrollado, entrenado y comparado tres modelos de visión por computadora para abordar la tarea de clasificación de 7 emociones:

### 1. CNN Personalizada (Desde Cero)

**Descripción:**  
Red Neuronal Convolucional optimizada y construida específicamente para este proyecto. Se centra en tener un número mínimo de capas y parámetros para asegurar una rápida inferencia en la CPU/GPU de un dispositivo de bajo consumo.

**Ventajas:**

- Modelo extremadamente ligero  
- Control total sobre la arquitectura  
- Menor riesgo de sobreajuste (overfitting) inicial  

**Uso:**  
Sirve como la línea base de rendimiento ligero frente a las otras arquitecturas.

---

### 2. ResNet50 con Transferencia de Aprendizaje

**Descripción:**  
Se utiliza la arquitectura ResNet50 (una red profunda con conexiones residuales) precargada con pesos de ImageNet. Se aplica la técnica de Transferencia de Aprendizaje, congelando las capas iniciales y reentrenando las capas superiores (clasificadoras).

**Técnicas utilizadas:**

- Transferencia de Aprendizaje  
- Fine-Tuning (ajuste fino) de las últimas capas convolucionales  

**Ventajas:**

- Alta precisión potencial debido a la capacidad de la arquitectura profunda  
- Aprovechamiento de pesos preentrenados en un gran dataset de imágenes  

**Uso:**  
Sirve como el modelo de referencia de alta precisión, contra el cual se comparan los otros modelos.

---

### 3. MobileNetV2

**Descripción:**  
Arquitectura de CNN diseñada específicamente por Google para aplicaciones móviles y embebidas. Utiliza convoluciones separables en profundidad (Depthwise Separable Convolutions) para reducir drásticamente el número de parámetros y operaciones de cálculo.

**Énfasis:**

- Latencia baja  
- Tamaño de modelo pequeño  

**Uso:**  
Es el modelo más prometedor para la implementación final en tiempo real en dispositivos con recursos limitados (por ejemplo, Raspberry Pi 5 o teléfonos móviles).

---

## Dataset

**Nombre:** FER2013 (Facial Expression Recognition 2013)  

**Descripción:**  
Contiene imágenes en escala de grises de rostros humanos de tamaño \(48 \times 48\) píxeles, etiquetadas con una de siete categorías de emociones:

- 0: Angry (Enojo)  
- 1: Disgust (Asco)  
- 2: Fear (Miedo)  
- 3: Happy (Felicidad)  
- 4: Sad (Tristeza)  
- 5: Surprise (Sorpresa)  
- 6: Neutral (Neutral)  

Este dataset se utiliza tanto para el entrenamiento como para la validación de los tres modelos.

---

## Tecnologías y Requisitos

### Software

- Python: \(\ge 3.8\)  
- TensorFlow / Keras (para desarrollo y entrenamiento de modelos)  
- OpenCV (para preprocesamiento de imágenes y captura de video en tiempo real)  
- NumPy  
- Scikit-learn  

### Requisitos de Hardware (para implementación en el borde / Edge)

- Raspberry Pi 5  
- Teléfono inteligente Android/iOS (a través de un wrapper como TensorFlow Lite)  
- Webcam o cámara CSI conectada al dispositivo  

---

## Estructura del Proyecto

```bash
/emotion-detector
├── models/                       # Modelos entrenados (H5, saved_model, TFLite)
│   ├── cnn_custom.h5
│   ├── resnet50_finetuned.h5
│   └── mobilenetv2_optimized.h5
├── data/                         # Scripts para descarga y preprocesamiento de FER2013
│   └── fer2013.csv
├── notebooks/                    # Jupyter notebooks para experimentación
│   ├── 1_data_exploration.ipynb
│   ├── 2_cnn_training.ipynb
│   └── 3_transfer_learning.ipynb
├── scripts/                      # Scripts de entrenamiento y conversión
│   ├── train_models.py
│   └── tflite_converter.py       # Conversión a formato TFLite
└── src/                          # Código fuente para inferencia en tiempo real
    └── real_time_detector.py     # Script principal para la Raspberry Pi
```

Ajusta los nombres de archivos y rutas según la estructura real de tu repositorio.

---

## Resultados Clave

Resultados esperados para cada arquitectura (valores de ejemplo, sustituir por los obtenidos tras el entrenamiento):

| Arquitectura          | Precisión (Validación) | Tamaño del Modelo (MB) | FPS Estimado (Raspberry Pi 5) |
|-----------------------|------------------------|------------------------|--------------------------------|
| CNN Personalizada     | 90.2 %                 | 9 MB                   | ≥ 30 FPS                       |
| ResNet50 (Fine-Tuned) | 77.64 %                | 100 MB                 |   ≈ 5 FPS                        |
| MobileNetV2           | 78 %                   | 40 MB                  | ≥ 20 FPS                       |

**Nota:** Sustituir `XX.XX %` y `X.X MB` por los valores reales obtenidos tras el entrenamiento y la evaluación.

---

## Ejecución del Proyecto

### 1. Entrenamiento

Para entrenar y guardar los modelos (se recomienda disponer de GPU):

```bash
python scripts/train_models.py
```

Este script debe:

- Cargar y preprocesar el dataset FER2013  
- Entrenar la CNN personalizada  
- Aplicar transferencia de aprendizaje y fine-tuning a ResNet50  
- Entrenar y optimizar MobileNetV2  
- Guardar los modelos entrenados en la carpeta `models/`  

---

### 2. Optimización para Dispositivos de Bajos Recursos

El modelo MobileNetV2 se convierte a TensorFlow Lite (TFLite) para una máxima eficiencia en CPU y microcontroladores:

```bash
python scripts/tflite_converter.py --model_path models/mobilenetv2_optimized.h5
```

Este script debe:

- Cargar el modelo en formato H5  
- Aplicar la conversión a TFLite (opcionalmente con cuantización)  
- Guardar el archivo `.tflite` en la carpeta `models/`  

---

### 3. Detección en Tiempo Real

Ejecuta el detector usando la cámara del dispositivo (por ejemplo, en Raspberry Pi 5):

```bash
python src/real_time_detector.py --model models/mobilenetv2_optimized.tflite
```

Este script debe:

- Cargar el modelo TFLite  
- Inicializar la cámara (webcam o CSI)  
- Detectar el rostro, preprocesar la imagen y realizar la inferencia en tiempo real  
- Mostrar la emoción predicha en la ventana de video o en la interfaz correspondiente  

---

## Próximas Mejoras

Algunas posibles extensiones del proyecto:

- Implementar cuantización post-entrenamiento más agresiva (por ejemplo, int8) para reducir aún más el tamaño del modelo.  
- Integrar un sistema de seguimiento de rostros para mejorar la estabilidad de la detección en video.  
- Desplegar el modelo como servicio en dispositivos móviles mediante TensorFlow Lite o frameworks nativos.  
- Añadir una interfaz gráfica ligera para usuarios finales.

---

