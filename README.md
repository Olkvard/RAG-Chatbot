# Sistema de Extracción de Información de Documentos mediante Retrieval Augmented Generation (RAG)

## Descripción General
Este proyecto está enfocado en construir un **sistema de extracción de información** utilizando **Retrieval Augmented Generation (RAG)**. El sistema integra la recuperación de documentos con **modelos open-source** para generar respuestas contextuales a partir de un conjunto de documentos. Su objetivo es ofrecer una solución eficiente al aprovechar modelos de lenguaje más pequeños que pueden ejecutarse en hardware convencional, sin necesidad de recursos computacionales extensos.

Las principales tecnologías usadas en este proyecto incluyen:
- **LangChain** para gestionar el modelo de lenguaje y las cadenas RAG.
- **Mistral 7B**, un modelo de lenguaje open-source optimizado a través de cuantización.
- **Chroma** como base de datos vectorial para almacenar y gestionar los embeddings de los documentos.
- **Gradio** para crear una interfaz intuitiva y fácil de usar para la interacción con el usuario.
- **Ollama** para ejecutar el modelo de lenguaje de forma sencilla.

## Características
- **Recuperación Eficiente de Documentos**: Utiliza recuperación basada en vectores para obtener documentos de alta precisión.
- **Generación de Respuestas Contextualizadas**: Combina los documentos recuperados con un modelo generativo para producir respuestas precisas basadas en contexto.
- **Ejecución Ligera**: Capaz de ejecutarse en ordenadores comerciales gracias al uso de modelos más pequeños y cuantizados.
- **Seguridad y Privacidad**: Al ejecutarse de forma local, no se envían datos a servidores externos, haciéndolo adecuado para proyectos confidenciales.

## Instalación

### Requisitos Previos
Asegúrate de tener instalados los siguientes elementos en tu sistema:
- Python 3.8+
- Ollama
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

4. Descargar el modelo preentrenado:         
    Descarga el modelo **mistral:instruct** y el modelo **nomic-embed-text** desde la interfaz de Ollama
   
5. Crear la base de datos:
           
   Para preparar el código para un proyecto nuevo el primer paso es decidir qué información se usará en el proyecto, este paso es crucial pues si no se seleccionan los más relevantes puede haber perdidas de precisión en las respuestas generadas. 

   Lo más recomendable sería buscar diferentes temas dentro de la base de datos, por ejemplo: Diseño, Requisitos, Tests, ... Y dividir los documentos según en que categoría caen. 

   Una vez obtenidos los documentos deben colocarse en carpetas divididas según el tema elegido y MUY IMPORTANTE que no tengan subcarpetas. 

   Cuando se ha completado este paso ya puede empezar el proceso de ajuste del código.

   El archivo database.py se encarga de la creación de las bases de datos que utilizará el LLM para responder a las preguntas, tan solo hay que cambiar dos líneas de código.

   ```bash
   DATA_PATH = "data"
   CHROMA_PATH = "dataDB"
   ```
   En la variable DATA_PATH se debe escribir la ruta de la carpeta donde se encuentran los documentos, y en la variable CHROMA_PATH la ruta de la base de datos que se va a crear. 

   Una vez realizados los cambios se ejecuta el programa, con `python database.py`, y se creará la carpeta con la base de datos, este proceso se debe repetir con todas las categorías seleccionadas anteriormente. 

7. Ajustar el código del main.py:                                                  
   Tras la creación de las bases de datos se deben actualizar los accesos que realiza el LLM. Para ello el primer paso es modificar las variables globales que indican la dirección de las bases de datos. 

    Interfaz de usuario gráfica, Texto

    Descripción generada automáticamente 


    En estas líneas se debe crear una variable global por cada base de datos creada y asignarle la ruta de dicha base de datos, lo recomendable es seleccionar cada una de estas y hacer click derecho, luego seleccionar la opción de cambiar todas las ocurrencias de esta forma te aseguras de que no haya problemas más adelante. Tal y como se muestra en la Figura 3. 
    ```bash
    python main.py
    ```
    
8. Acceder a la interfaz Gradio:              
   El sistema se iniciará localmente y podrás acceder a la interfaz Gradio en tu navegador en `http://localhost:7860/`.

## Uso

Una vez que el sistema esté funcionando, puedes:
- Hacer preguntas al sistema relacionadas con el contenido de los documentos.
- El sistema recuperará los documentos relevantes y generará una respuesta detallada.

## Estructura del Proyecto
```bash
├── main.py                   # Punto de entrada principal de la aplicación
├── data/                     # Directorio para almacenar los documentos subidos
├── dataDB/                   # Directorio para almacenar los embeddings en Chroma
├── README.md                 # Introducción al proyecto
└── requirements.txt          # Dependencias de Python
```

## Tecnologías Utilizadas

LangChain: Orquesta la pipeline de RAG.                
Mistral 7B: Modelo de lenguaje open-source optimizado para este proyecto.                   
Chroma: Base de datos vectorial para una recuperación eficiente de documentos.                 
Gradio: Interfaz para la interacción con el sistema.                   
Ollama: Facilita la ejecución del modelo de lenguaje de forma local.              

## Futuras Mejoras

Soporte Multimodal: El sistema podría expandirse para procesar imágenes o audio además de texto.        
Mejora de la Precisión: A medida que los modelos open-source mejoren, se podrán integrar modelos más pequeños con mejor precisión en el sistema.      

