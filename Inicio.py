import streamlit as st
import pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from typing import List
import time

# Configuración de la página
st.set_page_config(
    page_title="PDF Query App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialización de variables de sesión
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'api_status' not in st.session_state:
    st.session_state.api_status = None

def validate_api_keys(openai_key: str, pinecone_key: str, pinecone_env: str) -> bool:
    """Valida las API keys intentando realizar operaciones básicas."""
    try:
        # Validar OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key, model="gpt-4o-mini")
        _ = embeddings.embed_query("test")
        
        # Validar Pinecone
        pinecone.init(api_key=pinecone_key, environment=pinecone_env)
        _ = pinecone.list_indexes()
        
        return True
    except Exception as e:
        st.error(f"Error al validar las API keys: {str(e)}")
        return False

def initialize_credentials():
    """Configura y valida las credenciales de API."""
    with st.sidebar:
        st.header("📝 Configuración de APIs")
        
        # Campos para las API keys
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Ingresa tu API key de OpenAI. Puedes obtenerla en: https://platform.openai.com/account/api-keys"
        )
        
        pinecone_api_key = st.text_input(
            "Pinecone API Key",
            type="password",
            help="Ingresa tu API key de Pinecone. Puedes obtenerla en la consola de Pinecone."
        )
        
        pinecone_environment = st.text_input(
            "Pinecone Environment",
            help="El ambiente de Pinecone (ejemplo: us-east1-gcp)"
        )
        
        index_name = st.text_input(
            "Nombre del índice de Pinecone",
            help="El nombre del índice donde se almacenarán los embeddings"
        )
        
        # Botón para validar credenciales
        if st.button("Validar y Conectar"):
            if all([openai_api_key, pinecone_api_key, pinecone_environment, index_name]):
                with st.spinner("Validando credenciales..."):
                    if validate_api_keys(openai_api_key, pinecone_api_key, pinecone_environment):
                        # Guardar credenciales en variables de entorno temporales
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                        
                        # Inicializar servicios
                        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
                        
                        # Verificar si el índice existe
                        if index_name not in pinecone.list_indexes():
                            st.error(f"El índice '{index_name}' no existe en tu cuenta de Pinecone")
                            return
                        
                        st.session_state.index = pinecone.Index(index_name)
                        st.session_state.embeddings = OpenAIEmbeddings(model="gpt-4o-mini")
                        st.session_state.initialized = True
                        st.success("✅ ¡Conexión exitosa!")
                        st.session_state.api_status = "connected"
            else:
                st.error("Por favor, completa todos los campos")
        
        # Mostrar estado actual
        if st.session_state.api_status == "connected":
            st.sidebar.success("Estado: Conectado ✅")
        elif st.session_state.api_status is None:
            st.sidebar.warning("Estado: No conectado ⚠️")

def extract_text_from_pdf(pdf_file) -> str:
    """Extrae el texto de un archivo PDF."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        # Barra de progreso para la extracción
        progress_bar = st.progress(0)
        for i, page in enumerate(pdf_reader.pages):
            text += page.extract_text()
            progress_bar.progress((i + 1) / total_pages)
        
        return text
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        return ""

def split_text(text: str) -> List[str]:
    """Divide el texto en chunks manejables."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed_and_upload_to_pinecone(chunks: List[str], namespace: str):
    """Genera embeddings y los sube a Pinecone."""
    embeddings = st.session_state.embeddings
    batch_size = 100
    
    with st.spinner("Procesando documento..."):
        progress_bar = st.progress(0)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            
            try:
                embed = embeddings.embed_documents(batch)
                vectors = list(zip(ids, embed, batch))
                st.session_state.index.upsert(vectors=vectors, namespace=namespace)
                
                # Actualizar progreso
                progress = min((i + batch_size) / len(chunks), 1.0)
                progress_bar.progress(progress)
                
                # Pausa para evitar rate limits
                time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Error en el lote {i//batch_size + 1}: {str(e)}")
                return False
    
    return True

def query_pinecone(query: str, namespace: str, k: int = 5):
    """Realiza consultas en Pinecone."""
    try:
        if not query.strip():
            return []
        
        with st.spinner("Buscando resultados..."):
            query_embedding = st.session_state.embeddings.embed_query(query)
            results = st.session_state.index.query(
                vector=query_embedding,
                namespace=namespace,
                top_k=k,
                include_metadata=True
            )
        
        return results
    except Exception as e:
        st.error(f"Error al realizar la consulta: {str(e)}")
        return []

def main():
    st.title("🔍 Consulta de PDFs con Pinecone")
    st.markdown("""
    Esta aplicación te permite:
    1. Cargar documentos PDF
    2. Procesarlos y almacenarlos en Pinecone
    3. Realizar consultas semánticas sobre su contenido
    """)
    
    # Inicializar credenciales
    initialize_credentials()
    
    if not st.session_state.initialized:
        st.warning("⚠️ Por favor, configura y valida tus credenciales en la barra lateral para comenzar.")
        return
    
    # Tabs para separar funcionalidades
    tab1, tab2 = st.tabs(["📤 Subir Documento", "🔎 Realizar Consultas"])
    
    with tab1:
        st.header("Subir y Procesar PDF")
        uploaded_file = st.file_uploader("Selecciona tu archivo PDF", type="pdf")
        namespace = st.text_input(
            "Namespace para Pinecone",
            value="default_namespace",
            help="Un identificador único para agrupar los documentos relacionados"
        )
        
        if uploaded_file is not None:
            if st.button("Procesar PDF"):
                # Extraer texto
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.info(f"📄 Texto extraído: {len(text)} caracteres")
                    
                    # Dividir en chunks
                    chunks = split_text(text)
                    st.info(f"📑 Documento dividido en {len(chunks)} chunks")
                    
                    # Generar embeddings y subir a Pinecone
                    if embed_and_upload_to_pinecone(chunks, namespace):
                        st.success("✅ ¡Documento procesado y subido exitosamente!")
    
    with tab2:
        st.header("Realizar Consultas")
        query = st.text_input("💭 ¿Qué deseas consultar?")
        k = st.slider("Número de resultados", min_value=1, max_value=10, value=5)
        
        if st.button("🔍 Buscar"):
            results = query_pinecone(query, namespace, k)
            
            if results and hasattr(results, 'matches'):
                for i, match in enumerate(results.matches, 1):
                    with st.expander(f"Resultado {i} (Score: {match.score:.4f})"):
                        st.markdown(match.metadata.get('text', 'No se encontró texto'))

if __name__ == "__main__":
    main()
