import json
import gradio as gr
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pydantic import ValidationError
import datetime
from typing import Optional, Literal, List, Tuple, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
from database import generate_data_store

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

REPORTS_DB = "DB"
DESIGN_DB = "DesignDB"
SPECS_DB = "specsDB"
SOURCES = ""
EMBEDDING = OllamaEmbeddings(model = "nomic-embed-text")
LLM = Ollama(model = "mistral:instruct", temperature = 0.5, callbacks=callback_manager)
STRUCTURED_LLM = OllamaFunctions(model = "mistral:instruct", temperature = 0, callbacks=callback_manager)
History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

#Estas son las diferentes prompts que se pasan al LLM

RAG_TEMPLATE = """You are a powerful asistant, your role is to answer the questions based on the provided context.

Please respect the following rules to answer the question:
- The answer to the question should be found inside the provided context
- The answer must be in this language: Spanish

Here is the context: {context}.
You will return the answer to the question only based exclusively on the provided context. If you dont know the answer simply respond "I dont know".
Here is the question: {question}."""

HyDE_TEMPLATE = """Please write a brief text to answer the next question. \
Question: '''${question}'''"""

ROUTING_TEMPLATE = """You are an expert at routing a user question to the appropriate data source.\
Based on the topic of the question, route it to the relevant datasource, you must choose one."""

METADATA_TEMPLATE = """You are an expert in transforming user questions. \
Given a question, return the optimized query to get the most relevant results.\
If there is an acronim or some words you are not familiar with, don't try to refrase them."""

FILTER_QUERY = """You are an expert in transforming user prompts into filters.
Given a set of parameters create a filter to filter the results by those paramenters
The parameters to select from are:
total_pages, ModDate.
Only use the parameters that match the ones in the input to complete the filter:
    {{"$and": [{{"ModDate": {{"$eq": 2021-05-01}}}},{{"total_pages": {{"$gte":1}}}}]}}.
It is an obligation to include always this parameter in between the filter: {{"total_pages": {{"$gte":1}}}}
It is your obligation to submit only the filter, no words or explanation needed."""

RERANKING_TEMPLATE = """You are a helpfull assistant that generates multiple search queries based on a single input query. \
Generate multiple search queries related to: '''${question}'''\
Output (4 queries):"""

CONTEXTUALIZE_RAG_TEMPLATE = """Given a history chat and the last user question \
wich might reference the context on the history chat, reformulate the question in spanish so \
it includes the part or parts of the context that are relevant to the question. Do not answer the question, \
you only need to reformulate the question in spanish, if there is no context return the question exactly the same as it has been formulated. \
If it refers to a specific report make sure to include the date of that report. \
History: '''${history}''' \
Question: '''${question}'''"""

class Reports(BaseModel):
    """Esta clase se utiliza para detectar los parámetros de búsqueda dentro de los metadatos de los reportes."""
    ModDate: Optional[datetime.date] = Field(
        None,
        description="Date filter it has to be in the following format always yyyy-mm-dd. Only use if explicitly specified.",
    )
    def pretty_print(self) -> None:
        """Imprime la salida del LLM de forma estructurada"""
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                return(f"{field}: {getattr(self, field)}")

class Route(BaseModel):
    """En esta clase se definen las diferentes bases de datos a las que tiene acceso el LLM para su direccionamiento"""
    datasource: Literal["Monthly Reports", "Design and test plan", "Requisites of the project"]= Field(
        ...,
        description="Given a user question choose wich datasource would be most relevant for answering their question, you must choose one"
    )

def reciprocal_rank_fusion(results: list[list], k=60):
    """Ordena los resultados por relevancia. Actualmente no se utiliza.

    Args: 
        results: Una lista de todos los documentos.

    Returns:
        reranked_results: Una lista de los documentos ordenados."""
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1/(rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key= lambda x: x[1], reverse= True)
    ]
    return reranked_results

def query_reranking(retriever):
    """Crea una cadena para llamar al LLM y que ordene los resultados segun la relevancia, hace una llamada a la función reciprocal_rank_fusion. Actualmente no se usa."""

    prompt = ChatPromptTemplate.from_template(RERANKING_TEMPLATE)
    generate_queries = (
        prompt
        | LLM
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    return generate_queries | retriever.map() | reciprocal_rank_fusion

def query_translation(query):
    """Transforma la query a un documento hipotetico esperando que mejore la extracción de los documentos más relevantes. De momento no se han obtenido muy buenos resultados.

    Args: 
        query: Consulta original en str

    Return: 
        HyDE_query: Consulta modificada con forma de documento en str"""

    prompt = ChatPromptTemplate.from_template(HyDE_TEMPLATE)
    generate_docs_for_retrieval = (
        prompt | LLM | StrOutputParser()
    )
    HyDE_query=generate_docs_for_retrieval.invoke({"question":query})
    return HyDE_query

def query_context(history, query):
    """Otorga al LLM memoría conversacional poniendo las consultas en contexto con las anteriores. Actualmente no implementada

    Args: 
        history: Una lista de listas con todas las consultas del chat hasta el momento.
        query: La última consulta hecha por el usuario.
        
    Returns:
        contextualized_query: Si es la primera consulta que se hace se devuelve la query tal cual esta,
        si no es la primera consulta se devuelve la query junto con la consulta anterior."""
    
    prompt = ChatPromptTemplate.from_template(CONTEXTUALIZE_RAG_TEMPLATE)

    chain = (
        prompt | LLM | StrOutputParser()
    )
    contextualized_query = chain.invoke({"history": history, "question": query})

    return contextualized_query

def query_construction(query, clase):
    """Extrae de la query original los metadatos mediante los que se quiere realizar la búsqueda y crea un filtro para permitir esa búsqueda. Quiza debería dividirse en dos para aumentar la modularidad y reducir el acoplamiento.
    
    Args: 
        query: Un str que representa la consulta del usuario.
        clase: La clase de documento en el que se quiere buscar.
        
    Returns:
        filter: Un objeto tipo Dict con los parámetros de búsqueda específicados."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", METADATA_TEMPLATE),
            ("human", "{question}"),
        ]
    )
    structured_llm = STRUCTURED_LLM.with_structured_output(clase)
    query_analyzer = prompt | structured_llm
    metadata_query = query_analyzer.invoke({"question":query}).pretty_print()
    metadata_query = str(metadata_query)

    filter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FILTER_QUERY),
            ("human", "{question}"),
        ]
    )
    filtered_query = filter_prompt | LLM
    filter = filtered_query.invoke({"question": metadata_query})
    return filter

def query_router(query): 
    """Direcciona al LLM a la base de datos más relevante dependiendo de la consulta que se realiza.
    
    Args:
        query: La consulta realizada por el usuario.
        
    Returns:
        route: En formato JSON el nombre de la base de datos más similar según el LLM.
        
    Raises: 
        ValidationError: Si result no es del tipo indicado.
        Exception: Si hay algún otro error."""
    
    structured_llm = STRUCTURED_LLM.with_structured_output(Route)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTING_TEMPLATE),
            ("human", "{question}"),
        ]
    )
    router = prompt | structured_llm

    try:
        result = router.invoke({"question": query})
        
        # Comprueba si result ya es un objeto de tipo Route
        if isinstance(result, Route):
            route = result
        else:
            # Asumiendo que resul es un diccionario, se valida contra el modelo Route
            route = Route(**result)
        
        return route
        
    except ValidationError as e:
        print("Validation Error:", e)
        print("Result:", result)
        raise e
    except Exception as e:
        print("Unexpected error:", e)
        raise e

def create_chain():
    """Crea una cadena de langchain para hacer la pregunta al LLM"""

    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )
    return chain

def get_llm_response(query, history):
    """Esta es la función principal para obtener las respuestas del LLM, dependiendo de la base de datos a la que se le direcciona realiza la búsqueda de una forma u otra.
    
    Args:
        query: La consulta del usuario.
        history: Una lista con todos los mensajes del chat hasta el momento.
        
    Returns:
        formatted_response: La respuesta del LLM preparada para tener la siguiente forma:
        Respuesta: 'respuesta'
        Fuentes: [documentId],...
        
    Raises: 
        Exception: Si hay cualquier problema al obtener la respuesta"""

    Reports_DB = Chroma(persist_directory=REPORTS_DB, embedding_function=EMBEDDING)
    Design_DB = Chroma(persist_directory=DESIGN_DB, embedding_function=EMBEDDING)
    Specs_DB = Chroma(persist_directory=SPECS_DB, embedding_function=EMBEDDING)

    #Esto junta el historial del chat y lo transforma en un string. Se usa para darle al chatbot memoria a corto plazo, por el tamaño del modelo se ha ajustado a máximo 10 preguntas
    historia = []
    if len(history)<10:
        for item in history:
            historia.append(''.join(item))
        historia = ". \n".join(historia)
    else:
        historial = history[-10]
        for item in historial:
            historia.append(''.join(item))
        historia = ". \n".join(historia)
    if historia:
        contextualized_query = query_context(historia, query)
    else:
        contextualized_query= query

    try:
        # Determina la base de datos y el flujo a seguir dependiendo de la pregunta
        most_similar= query_router(contextualized_query)
        if "Monthly Reports" in most_similar.datasource:
            filter = json.loads(query_construction(contextualized_query, Reports))
            results = Reports_DB.similarity_search_with_score(contextualized_query, k=7, filter=filter)
            context_text = "\n\n---\n\n".join([doc[0].page_content for doc in results])
        elif "Design and test plan" in most_similar.datasource:
            results = Design_DB.similarity_search_with_score(contextualized_query, k=7)
            context_text = "\n\n---\n\n".join([doc[0].page_content for doc in results])
        elif "Requisites of the project" in most_similar.datasource:
            results = Specs_DB.similarity_search_with_score(contextualized_query, k=7)
            context_text = "\n\n---\n\n".join([doc[0].page_content for doc in results])

        chain = create_chain()
        response_stream = chain.stream({"context": context_text, "question": contextualized_query})
        sources = [doc[0].metadata.get("id", None) for doc in results]

        full_response = most_similar.datasource + "\n"
        for chunk in response_stream:
            full_response += chunk 
            yield chunk, history, sources

    except Exception as e:
        print("Unexpected error in get_llm_response:", e)
        raise e
    
def view_context_files (history, sources):
    history.append(sources)
    return history

def clear_session() -> History:
    return '', [], ''

def retry(chat_history):
    last_input = ""
    # Aquí puedes agregar lógica para reintentar la última pregunta
    if chat_history:
        last_input = chat_history[-1][0]  # Obtiene la última pregunta
        chat_history[-1]= (last_input, "")
    return gr.update, gr.update

def vote(data: gr.LikeData, history):
    if data.liked:
        most_similar = query_router(data.value)

        if "Monthly Reports" in most_similar.datasource:
            with open("data/RespuestasCorrectas.txt", 'a') as archivo:
                archivo.write(data.value + '\n\n')

            generate_data_store(chunk_size=1000, data_path="data", chroma_path=REPORTS_DB)
            
        elif "Design and test plan" in most_similar.datasource:
            with open("DesignData/RespuestasCorrectas.txt", 'a') as archivo:
                archivo.write(data.value + '\n\n')
            
            generate_data_store(chunk_size=1000, data_path="DesignData", chroma_path=DESIGN_DB)

        elif "Requisites of the project" in most_similar.datasource:
            with open("specificationData/RespuestasCorrectas.txt", 'a') as archivo:
                archivo.write(data.value + '\n\n')

            generate_data_store(chunk_size=1000, data_path="specificationData", chroma_path=SPECS_DB)

        mesage = "La respuesta se ha añadido a la base de datos: "+ most_similar.datasource

        history.append(("Buena respuesta", mesage))
    else:
        mesage = "Pruebe a realizar la pregunta de forma más detallada"
        history.append(("No me ha gustado la respuesta", mesage))

    return history
        
def main():
    """Crea una interfaz de usuario para permitir el uso de la aplicación como un chatbot conversacional"""

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=7><b>RAG Chatbot</b></font></center>""")

        # Componentes del chatbot
        chatbot = gr.Chatbot(label='Mistral:Instruct', height=560)
        sources = gr.TextArea(label='Fuentes', lines=3)
        textbox = gr.Textbox(lines=1, label='Pregunta')

        with gr.Row():
            try_again = gr.Button("Intentar de nuevo")
            clear_history = gr.Button("Borrar historial")
            submit = gr.Button("Enviar")

        
        def stream_response(query, history):
            """Función para manejar el streaming de respuestas en un solo mensaje."""
    
            # Añadir la pregunta al historial inicialmente
            history.append((query, ""))  # Añadir la pregunta con una respuesta vacía inicialmente
            
            # Llama a get_llm_response con streaming de la respuesta
            response_so_far = ""
            for response_chunk, updated_history, updated_sources in get_llm_response(query, history):
                response_so_far += response_chunk  # Acumular respuesta progresivamente
                # Actualizar el último mensaje del historial con la respuesta acumulada
                history[-1] = (query, response_so_far)  # Sobreescribir la última respuesta con la nueva parcial
                # Devolver el historial actualizado y las fuentes (que se pueden vaciar al principio)
                yield "", history, updated_sources
        

        # Evento para el envío de la pregunta (con streaming)
        textbox.submit(stream_response,
                       inputs=[textbox, chatbot],
                       outputs=[textbox, chatbot, sources],
                       concurrency_limit=40)

        submit.click(stream_response,
                     inputs=[textbox, chatbot],
                     outputs=[textbox, chatbot, sources],
                     concurrency_limit=40)

        # Evento para limpiar el historial
        clear_history.click(fn=clear_session,  # Reiniciar historial y textbox
                            inputs=[],
                            outputs=[textbox, chatbot, sources],
                            concurrency_limit=10)

        # Evento para reintentar la última pregunta
        try_again.click(fn=retry,
                        inputs=[chatbot],
                        outputs=[textbox, chatbot],
                        concurrency_limit=10)

        try_again.click(stream_response,
                        inputs=[textbox, chatbot],
                        outputs=[textbox, chatbot, sources],
                        concurrency_limit=40)
        
        chatbot.like(vote, inputs=[chatbot], outputs=[chatbot])

    demo.launch()


if __name__ == "__main__":
    main()
