# PDF Query App con Pinecone

Una aplicación de Streamlit que permite cargar documentos PDF, procesarlos y realizar búsquedas semánticas utilizando Pinecone como base de datos vectorial y OpenAI para la generación de embeddings.

## 🚀 Características

- Carga y procesamiento de archivos PDF
- Almacenamiento de vectores en Pinecone
- Búsqueda semántica en el contenido de los documentos
- Interfaz intuitiva con Streamlit
- Manejo de múltiples namespaces
- Visualización de estadísticas del índice

## 📋 Prerrequisitos

- Python 3.8 o superior
- Una cuenta en OpenAI con API key
- Una cuenta en Pinecone con API key
- Un índice en Pinecone con dimensionalidad 1536

## 🔧 Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd pdf-query-app
```

2. Crear un entorno virtual:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuración de Pinecone

1. Crear una cuenta en [Pinecone](https://www.pinecone.io/)
2. Obtener tu API key desde la consola de Pinecone
3. Crear un nuevo índice con las siguientes especificaciones:
   - Dimensionalidad: 1536
   - Métrica: cosine
   - Pod type: p1 o superior

## 🔑 Configuración de OpenAI

1. Crear una cuenta en [OpenAI](https://platform.openai.com)
2. Obtener tu API key desde la configuración de tu cuenta
3. Asegurarte de tener créditos disponibles

## 📦 Requisitos del Sistema

```txt
# Framework web
streamlit>=1.32.0

# Procesamiento de PDFs
PyPDF2>=3.0.0

# LangChain y componentes
langchain>=0.1.0
langchain-community>=0.0.16
langchain-openai>=0.0.5
langchain-pinecone>=0.0.3

# Base de datos vectorial
pinecone-client>=3.0.0

# OpenAI
openai>=1.12.0

# Utilidades
python-dotenv>=0.19.0
typing>=3.7.4
tqdm>=4.65.0
```

## 🚀 Ejecución de la Aplicación

1. Activar el entorno virtual si no está activado:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. Ejecutar la aplicación:
```bash
streamlit run app.py
```

3. Abrir el navegador en `http://localhost:8501`

## 📝 Uso de la Aplicación

1. **Configuración Inicial:**
   - Ingresar la API key de OpenAI
   - Ingresar la API key de Pinecone
   - Seleccionar el índice de Pinecone

2. **Cargar Documentos:**
   - Ir a la pestaña "Subir PDF"
   - Seleccionar un archivo PDF
   - Especificar un namespace (opcional)
   - Hacer clic en "Procesar PDF"

3. **Realizar Consultas:**
   - Ir a la pestaña "Consultar"
   - Escribir tu pregunta
   - Hacer clic en "Buscar"
   - Ver los resultados ordenados por relevancia

## ⚠️ Solución de Problemas Comunes

1. **Error de dimensionalidad:**
   - Asegurarse de que el índice en Pinecone tenga dimensión 1536
   - Crear un nuevo índice si es necesario

2. **Errores de API key:**
   - Verificar que las API keys sean válidas
   - Comprobar que tengas créditos disponibles en OpenAI

3. **Errores de procesamiento de PDF:**
   - Verificar que el PDF no esté corrupto
   - Asegurarse de que el PDF sea legible y no esté escaneado

## 🔐 Seguridad

- Las API keys se manejan de forma segura y no se almacenan
- Los datos se procesan localmente antes de ser enviados
- Se utilizan conexiones seguras para todas las API

## 📈 Limitaciones

- El tamaño máximo de archivo PDF depende de la memoria disponible
- La velocidad de procesamiento depende de la conexión a internet
- Los costos dependen del uso de las APIs de OpenAI y Pinecone

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría hacer.

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE.md](LICENSE.md) para más detalles.
