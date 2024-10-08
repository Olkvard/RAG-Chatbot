import argparse
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os
import shutil

#Aquí se especifica la dirección de la base de datos y de donde se extrae la información, además del tamaño de los chunks
CHUNK_SIZE = 1000
DATA_PATH = "specificationData"
CHROMA_PATH = "specsDB"

def load_documents(documents, data_path):
    """Esta función carga todos los documentos presentes en el DATA_PATH.
    
    Args:
        documents: Es un array en el que se van cargando los documentos
        
    Returns:
        documents: El mismo array con los documentos cargados"""
    
    for file in os.listdir(data_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(data_path, file)
            loader = PDFPlumberLoader(pdf_path)
            documents.extend(loader.load())
        if file.endswith('.txt'):
            txt_path = os.path.join(data_path, file)
            loader = TextLoader(txt_path)
            documents.extend(loader.load())
    return documents

def split_text(documents: list[Document], chunk_size):
    """Esta función divide el texto de los documentos en chunks del tamaño específicado por el usuario.
    
    Args:
        documents: Un array con los documentos cargados
        chunk_size: Un integer que indica el tamaño de los chunks
        
    Returns:
        chunks: Un array con los chunks ya divididos."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

def calculate_chunk_ids(chunks):
    """Esta función calcula los ids de los chunks para que no haya dos iguales en la base de datos.
    
    Args:
        chunks: Un array con los chunks sin ids.
        
    Returns:
        chunks: Un array con los chunks con ids."""
    
    # Esto creará IDs como el siguiente "data/nombre_documento.pdf:6:2"
    # Nombre del documento : Número de página : Indice del chunk

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def adjust_dates(chunks):
    """Esta función modifica la fecha de los documentos para que tengan el formato adecuado.
    
    Args:
        chunks: Un array con los chunks y la fecha mal puesta
        
    Returns:
        chunks: Un array con los chunks y la fecha bien puesta"""
    
    for chunk in chunks:
        mod_date = chunk.metadata.get("ModDate", "")
        formatted_date = f"{mod_date[2:6]}-{mod_date[6:8]}-01"
        chunk.metadata["ModDate"] = formatted_date
    
    return chunks

def save_to_chroma(chunks: list[Document], chroma_path):
    """Esta función codifica y guarda la información dentro de la base de datos.
    
    Args:
        chunks: Un array con los chunks procesados"""
    
    # Crea una nueva base de datos vectorial.
    db = Chroma(
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"), persist_directory=chroma_path
    )

    chunks_with_ids = calculate_chunk_ids(adjust_dates(chunks))

    # Añade o actualiza los documentos.
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Solo añade los documentos que no esten presentes en la base de datos.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")

def clear_database(chroma_path):
    """Elimina toda la base de datos."""
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

def generate_data_store(chunk_size, data_path, chroma_path):
    """Esta función genera toda la información llamando a las funciones auxiliares.
    
    Args:
        chunk_size: Integer que indica el tamaño que van a tener los chunks"""
    documents=[]
    documents = load_documents(documents, data_path)
    chunks = split_text(documents, chunk_size)
    save_to_chroma(chunks, chroma_path)
    print(chunks[0].metadata)

def main():
    """Funcion principal. Para ejecutarse debe pasarse el tamaño de los chunks y si desea borrar la anterior base de datos o no."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("Clearing Database")
        clear_database()
    
    generate_data_store(CHUNK_SIZE, DATA_PATH, CHROMA_PATH)

if __name__ == "__main__":
    main()