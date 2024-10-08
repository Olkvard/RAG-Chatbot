# Sistema de Extracción de Información de Documentos mediante Retrieval Augmented Generation (RAG)

## Descripción General
Este proyecto está enfocado en construir un **sistema de extracción de información** utilizando **Retrieval Augmented Generation (RAG)**. El sistema integra la recuperación de documentos con **modelos open-source** para generar respuestas contextuales a partir de un conjunto de documentos. Su objetivo es ofrecer una solución eficiente al aprovechar modelos de lenguaje más pequeños que pueden ejecutarse en hardware convencional, sin necesidad de recursos computacionales extensos.

Las principales tecnologías usadas en este proyecto incluyen:
- **LangChain** para gestionar el modelo de lenguaje y las cadenas RAG.
- **Mistral 7B**, un modelo de lenguaje open-source optimizado a través de cuantización.
- **Chroma** como base de datos vectorial para almacenar y gestionar los embeddings de los documentos.
- **Gradio** para crear una interfaz intuitiva y fácil de usar para la interacción con el usuario.

## Características
- **Recuperación Eficiente de Documentos**: Utiliza recuperación basada en vectores para obtener documentos de alta precisión.
- **Generación de Respuestas Contextualizadas**: Combina los documentos recuperados con un modelo generativo para producir respuestas precisas basadas en contexto.
- **Ejecución Ligera**: Capaz de ejecutarse en ordenadores comerciales gracias al uso de modelos más pequeños y cuantizados.
- **Seguridad y Privacidad**: Al ejecutarse de forma local, no se envían datos a servidores externos, haciéndolo adecuado para proyectos confidenciales.

## Instalación

### Requisitos Previos
Asegúrate de tener instalados los siguientes elementos en tu sistema:
- Python 3.8+
- Entorno virtual (recomendado)

### Guía Paso a Paso

1. Clonar el repositorio:
    ```bash
    git clone https://github.com/tuusuario/sistema-extraccion-informacion-RAG.git
    cd sistema-extraccion-informacion-RAG
    ```

2. Crear y activar un entorno virtual:
    ```bash
    python3 -m venv env
    source env/bin/activate  # En Windows usa `env\Scripts\activate`
    ```

3. Instalar las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

4. Descargar los pesos del modelo preentrenado:
    Descarga el modelo **Mistral 7B** y coloca los pesos en el directorio `models/`.

5. Ejecutar el sistema de forma local:
    ```bash
    python app.py
    ```

6. Acceder a la interfaz Gradio:
   El sistema se iniciará localmente y podrás acceder a la interfaz Gradio en tu navegador en `http://localhost:7860/`.

## Uso

Una vez que el sistema esté funcionando, puedes:
- Subir un conjunto de documentos en formato PDF o texto.
- Hacer preguntas al sistema relacionadas con el contenido de los documentos.
- El sistema recuperará los documentos relevantes y generará una respuesta detallada.

## Estructura del Proyecto
```bash
├── app.py                    # Punto de entrada principal de la aplicación
├── models/                   # Directorio para almacenar los pesos del modelo
├── data/                     # Directorio para almacenar los documentos subidos
├── embeddings/               # Directorio para almacenar los embeddings en Chroma
├── README.md                 # Introducción al proyecto
├── requirements.txt          # Dependencias de Python
└── src/                      # Código fuente para la implementación de RAG
```

## Tecnologías Utilizadas

LangChain: Orquesta la pipeline de RAG.
Mistral 7B: Modelo de lenguaje open-source optimizado para este proyecto.
Chroma: Base de datos vectorial para una recuperación eficiente de documentos.
Gradio: Interfaz para la interacción con el sistema.

## Futuras Mejoras

Soporte Multimodal: El sistema podría expandirse para procesar imágenes o audio además de texto.
Mejora de la Precisión: A medida que los modelos open-source mejoren, se podrán integrar modelos más pequeños con mejor precisión en el sistema.

