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
    ```bash
   DATA_DB = "dataDB"
   DATA2_DB = "data2DB"
   ```
    En estas líneas se debe crear una variable global por cada base de datos creada y asignarle la ruta de dicha base de datos. Tras este proceso debemos ir a la clase `Route` aqui debemos cambiar las opciones dentro de los corchetes de `Literal`, y poner el tema de cada uno da nuestras bases de datos. Se obtienen mejores resultados si los nombres están en inglés.
   ```bash
   class Route(BaseModel):
    """En esta clase se definen las diferentes bases de datos a las que tiene acceso el LLM para su direccionamiento"""
    datasource: Literal["Quantum Phisics", "Tema 2"]= Field(
        ...,
        description="Given a user question choose wich datasource would be most relevant for answering their question, you must choose one"
    )
   ```
   Ahora bajamos a la función `get_llm_response` y aquí debemos modificar varias cosas, primero debemos crear las variables que se encarguen de extraer los documentos, tan solo hay que cambiar dos nombres.
   ```bash
   Phisics_DB = Chroma(persist_directory=DATA_DB, embedding_function=EMBEDDING)
   Data2_DB = Chroma(persist_directory=DATA2_DB, embedding_function=EMBEDDING)
   ```
    Una vez cambiado eso debemos cambiar la selección de base de datos a buscar.
   ```bash
   if "Quantum Phisics" in most_similar.datasource:
       results = phisics_DB.similarity_search_with_score(contextualized_query, k=7, filter=filter)
       context_text = "\n\n---\n\n".join([doc[0].page_content for doc in results])
   elif "Tema 2" in most_similar.datasource:
       filter = json.loads(query_construction(contextualized_query, DateFilter))
       results = Data2_DB.similarity_search_with_score(contextualized_query, k=7)
       context_text = "\n\n---\n\n".join([doc[0].page_content for doc in results])
   ```
   En este caso hay dos tipos de bases de datos, las que necesitan filtrado por fecha o no, por ejemplo si tienes una base de datos que almacena los reportes mensuales sobre un proyecto y quieres buscar una reunión en concreto, entonces en esa base de datos será necesaria la linea de filtro, a parte de eso tan solo tienes que asegurarte que los nombres entre " " coincidan con los nombres de la clase `Route` y que se acceda a la variable correcta.
   Por último para asegurarnos que la función de validación de respuestas funciona como es debido debemos cambiar los nombres en la función `vote`
    ```bash
    if "Quantum Phisics" in most_similar.datasource:
        with open("data/RespuestasCorrectas.txt", 'a') as archivo:
            archivo.write(data.value + '\n\n')

        generate_data_store(chunk_size=1000, data_path="data", chroma_path=DATA_DB)
            
    elif "Tema 2" in most_similar.datasource:
        with open("data2/RespuestasCorrectas.txt", 'a') as archivo:
            archivo.write(data.value + '\n\n')
            
        generate_data_store(chunk_size=1000, data_path="data2", chroma_path=DATA2_DB)
    ```
    En estas líneas hay que cambiar los nombres entre " " para que se ajusten a los que hemos puesto anteriormente. Y tras todas estas modificaciones ya podemos ejecutar el programa.
   ```bash
   python main.py
   ```
    
9. Acceder a la interfaz Gradio:              
   El sistema se iniciará localmente y podrás acceder a la interfaz Gradio en tu navegador en `http://localhost:7860/`.

## Uso

Una vez que el sistema esté funcionando, puedes:
- Hacer preguntas al sistema relacionadas con el contenido de los documentos.
- El sistema recuperará los documentos relevantes y generará una respuesta detallada.

## Estructura del Proyecto
```bash
├── data/                     # Directorio para almacenar los documentos subidos
├── README.md                 # Introducción al proyecto
├── database.py               # Creación de las bases de datos
├── main.py                   # Punto de entrada principal de la aplicación
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

