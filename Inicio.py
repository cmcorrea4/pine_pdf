import streamlit as st
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time

# Configuración de la página
st.set_page_config(page_title="PDF Query App", layout="wide")
st.title("Sistema de Registro y Consulta de PDFs con Pinecone")

# Función para obtener índices de Pinecone
def get_pinecone_indexes(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        current_indexes = pc.list_indexes().names()
        st.write("Índices disponibles:", current_indexes)
        return list(current_indexes)
    except Exception as e:
        st.error(f"Error al obtener índices: {str(e)}")
        return []

# Función para limpiar estados
def clear_all_states():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# Inicialización de estados
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar para configuración
with st.sidebar:
    st.markdown("### Configuración de Credenciales")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Introduce tu API key de OpenAI"
    )
    
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        type="password",
        help="Introduce tu API key de Pinecone"
    )
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Actualizar"):
            st.experimental_rerun()
    with col2:
        if st.button("🗑️ Limpiar Caché"):
            clear_all_states()
    
    # Verificar conexión con Pinecone
    if pinecone_api_key:
        try:
            st.markdown("### Estado de Conexión")
            available_indexes = get_pinecone_indexes(pinecone_api_key)
            if available_indexes:
                st.success("✅ Conectado a Pinecone")
                selected_index = st.selectbox(
                    "Selecciona un índice",
                    options=available_indexes
                )
                
                if selected_index:
                    pc = Pinecone(api_key=pinecone_api_key)
                    index = pc.Index(selected_index)
                    stats = index.describe_index_stats()
                    st.info(f"Índice seleccionado: {selected_index}")
                    st.write("Estadísticas:", stats)
            else:
                st.warning("⚠️ No hay índices disponibles")
                selected_index = None
        except Exception as e:
            st.error(f"Error de conexión: {str(e)}")
            selected_index = None
    else:
        selected_index = None

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        return ""

def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_and_upload_to_pinecone(chunks, namespace: str, embedding_model, index):
    try:
        with st.spinner("Procesando y subiendo documentos..."):
            texts = [chunk for chunk in chunks]
            embeddings = embedding_model.embed_documents(texts)
            
            # Preparar vectores para Pinecone
            vectors = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                vectors.append({
                    "id": f"doc_{i}",
                    "values": embedding,
                    "metadata": {"text": text}
                })
            
            # Subir en lotes
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                time.sleep(0.5)  # Evitar rate limits
                
            st.success(f"✅ {len(vectors)} vectores subidos exitosamente")
            return True
    except Exception as e:
        st.error(f"Error al procesar y subir documentos: {str(e)}")
        return False

# Interface principal
if pinecone_api_key and openai_api_key and selected_index:
    tabs = st.tabs(["📤 Subir PDF", "🔍 Consultar"])
    
    with tabs[0]:
        st.header("Subir y Procesar PDF")
        uploaded_file = st.file_uploader("Selecciona un PDF", type="pdf")
        namespace = st.text_input("Namespace", value="default")
        
        if uploaded_file and st.button("Procesar PDF"):
            try:
                # Inicializar servicios
                embedding_model = OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    model="text-embedding-ada-002"
                )
                
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index(selected_index)
                
                # Procesar PDF
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.info(f"Texto extraído: {len(text)} caracteres")
                    chunks = split_text(text)
                    st.info(f"Documento dividido en {len(chunks)} chunks")
                    
                    if process_and_upload_to_pinecone(chunks, namespace, embedding_model, index):
                        st.success("¡Documento procesado y subido exitosamente!")
                        st.session_state.system_initialized = True
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tabs[1]:
        st.header("Realizar Consultas")
        if st.session_state.system_initialized:
            query = st.text_input("Tu pregunta:")
            if query and st.button("Buscar"):
                try:
                    # Inicializar servicios para búsqueda
                    embedding_model = OpenAIEmbeddings(
                        openai_api_key=openai_api_key,
                        model="text-embedding-ada-002"
                    )
                    pc = Pinecone(api_key=pinecone_api_key)
                    index = pc.Index(selected_index)
                    
                    # Realizar búsqueda
                    query_embedding = embedding_model.embed_query(query)
                    results = index.query(
                        vector=query_embedding,
                        namespace=namespace,
                        top_k=5,
                        include_metadata=True
                    )
                    
                    # Mostrar resultados
                    st.markdown("### Resultados encontrados:")
                    for i, match in enumerate(results['matches'], 1):
                        with st.expander(f"Resultado {i} (Score: {match['score']:.4f})"):
                            st.write(match['metadata']['text'])
                
                except Exception as e:
                    st.error(f"Error en la búsqueda: {str(e)}")
        else:
            st.warning("Por favor, primero procesa un documento PDF.")
else:
    st.info("👈 Configura las credenciales en el panel lateral para comenzar.")

# Información en el sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Sobre esta aplicación")
    st.write("""
    Esta aplicación te permite:
    1. Cargar documentos PDF
    2. Procesarlos y almacenarlos en Pinecone
    3. Realizar búsquedas semánticas sobre su contenido
    """)
