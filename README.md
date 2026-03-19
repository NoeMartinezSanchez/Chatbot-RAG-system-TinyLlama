---
title: Chatbot RAG Prepa en Línea SEP
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Tu README existente...

# 🤖 Chatbot RAG para Prepa en Línea SEP

Sistema de asistencia educativa inteligente con **Retrieval-Augmented Generation (RAG)** basado en **TinyLlama**, diseñado para proporcionar soporte 24/7 a estudiantes de Prepa en Línea SEP.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![TinyLlama](https://img.shields.io/badge/TinyLlama-1.1B-FF6B6B.svg)](https://huggingface.co/TinyLlama)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4ECDC4.svg)](https://faiss.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-27ae60.svg)]()

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura](#-arquitectura)
- [Tecnologías](#-tecnologías)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [API](#-api)
- [Configuración](#-configuración)
- [Problemas Conocidos](#-problemas-conocidos)
- [Roadmap](#-roadmap)
- [Licencia](#-licencia)

## 🎯 Descripción

Sistema de asistencia educativa que combina RAG con el modelo **TinyLlama-1.1B-Chat** para generar respuestas contextualizadas basadas en la documentación oficial de Prepa en Línea SEP.

### Características

- **Pipeline RAG** con embeddings multilingües Sentence Transformers
- **Vector Store FAISS** para búsqueda semántica rápida
- **TinyLlama 1.1B** como modelo generativo (ejecución en CPU)
- **API REST** con FastAPI
- **Interfaz Web** responsiva

## 🏗️ Arquitectura

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Usuario   │────▶│  FastAPI    │────▶│ RAG System  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                   ┌─────────────┐     ┌─────────────┐
                   │  Interfaz   │     │  TinyLlama  │
                   │     Web     │◀────│  Generator  │
                   └─────────────┘     └─────────────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │   FAISS     │
                                      │  Retriever  │
                                      └─────────────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │  Embedding  │
                                      │   Model     │
                                      └─────────────┘
```

### Componentes Principales

| Componente | Tecnología |
|------------|------------|
| **Modelo LLM** | TinyLlama-1.1B-Chat (Hugging Face) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | FAISS CPU |
| **API** | FastAPI + Uvicorn |
| **Frontend** | HTML5/CSS3/JavaScript |

## 🛠️ Tecnologías

- **FastAPI** - Framework web moderno
- **TinyLlama** - Modelo generativo ligero (1.1B parámetros)
- **Hugging Face Transformers** - Carga e inferencia del modelo
- **FAISS** - Búsqueda vectorial
- **Sentence Transformers** - Embeddings multilingües
- **PyTorch** - Backend de deep learning

## 📁 Estructura del Proyecto

```
ChatBot_5_TinyLlama/
├── api/
│   ├── main.py              # Aplicación FastAPI
│   └── endpoints.py         # Endpoints REST
├── rag/
│   ├── core.py              # Sistema RAG principal
│   ├── generator.py         # Generador TinyLlama
│   ├── retriever.py        # Vector Store FAISS
│   └── embeddings.py       # Modelo de embeddings
├── models/
│   ├── tinyllama_wrapper.py # Wrapper del modelo
│   └── __init__.py
├── config/
│   ├── settings.py          # Configuración centralizada
│   └── models.py           # Modelos Pydantic
├── static/
│   └── index.html          # Interfaz web
├── data/
│   └── vector_store/       # Índices FAISS
├── scripts/                 # Utilidades
├── tests/                  # Pruebas
├── .env                    # Variables de entorno
├── requirements.txt        # Dependencias
└── README.md              # Este archivo
```

## 🚀 Instalación

### Requisitos

- Python 3.12+
- 8GB RAM disponible
- 4GB+ espacio en disco

### Pasos

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd ChatBot_5_TinyLlama

# 2. Crear entorno virtual
python -m venv tinyllama_env
tinyllama_env\Scripts\activate  # Windows
# source tinyllama_env/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar entorno
cp .env.example .env
# Editar .env con configuración deseada

# 5. Iniciar servidor
python -m api.main
```

### Dependencias Principales

```
fastapi>=0.115.0
uvicorn[standard]>=0.24.0
torch>=2.0.0
transformers>=4.40.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
loguru>=0.7.0
```

## ⚡ Uso

### Iniciar el Servidor

```bash
python -m api.main
```

### Acceder a la Interfaz

- **Web**: http://localhost:8000
- **Swagger**: http://localhost:8000/api/docs
- **Health**: http://localhost:8000/health

### Ejemplo de Consulta cURL

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cómo cambio mi correo electrónico?"}'
```

## 📡 API

### Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/chat` | Consultar al chatbot |
| `GET` | `/health` | Verificar estado |
| `GET` | `/stats` | Estadísticas del sistema |
| `GET` | `/` | Interfaz web |

### Formato de Request (POST /chat)

```json
{
  "message": "¿Cómo cambio mi correo?",
  "conversation_id": "user_123",
  "user_id": "estudiante_456"
}
```

### Formato de Response

```json
{
  "response": "Para cambiar tu correo...",
  "sources": [...],
  "confidence": 0.85,
  "is_rag_response": true
}
```

## ⚙️ Configuración

### Variables de Entorno

```env
# API
API_HOST=0.0.0.0
API_PORT=8000

# RAG
TOP_K_RESULTS=3
SIMILARITY_THRESHOLD=0.7

# TinyLlama
TEMPERATURE=0.15
MAX_NEW_TOKENS=150
REPETITION_PENALTY=1.5
```

### Parámetros de Generación

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `temperature` | 0.15 | Creatividad vs deterministic |
| `top_p` | 0.85 | Nucleus sampling |
| `repetition_penalty` | 1.5 | Reduce repeticiones |
| `max_new_tokens` | 150 | Longitud máxima de respuesta |
| `no_repeat_ngram_size` | 3 | Previene bucles |

## ⚠️ Problemas Conocidos

### Estado: En Optimización (Marzo 2026)

| Problema | Estado | Descripción |
|----------|--------|-------------|
| **Alucinaciones** | 🔄 En proceso | TinyLlama menciona universidades/carreras fuera de Prepa en Línea |
| **Formato inconsistente** | 🔄 En proceso | Respuestas varían entre guiones, números y párrafos |
| **Bucles de repetición** | 🔄 En proceso | En ciertas preguntas se encicla |

### Acciones Realizadas

- Prompt simplificado en `rag/generator.py`
- Parámetros ajustados (temperature, repetition_penalty, no_repeat_ngram_size)
- Limpieza de metadatos del contexto antes de pasarlo al modelo

## 🗺️ Roadmap

### ✅ Completado

- [x] Migración de BERT a TinyLlama
- [x] Wrapper del modelo en `models/tinyllama_wrapper.py`
- [x] Integración RAG funcionando
- [x] API con endpoint /chat
- [x] Interfaz web operativa

### 🔄 En Desarrollo

- [ ] Optimización de prompts para reducir alucinaciones
- [ ] Formato consistente de respuestas
- [ ] Prevención de bucles de repetición
- [ ] Mejora de calidad de respuestas

### 📋 Próximos Pasos

- [ ] Testing con usuarios reales
- [ ] Métricas de satisfacción
- [ ] Documentación de casos de uso

## 🚀 Despliegue en Hugging Face Spaces

El chatbot está desplegado y disponible públicamente en Hugging Face Spaces, lo que permite su acceso 24/7 sin depender de infraestructura local.

### 📊 Especificaciones del Despliegue

| Característica | Detalle |
|----------------|---------|
| **Plataforma** | Hugging Face Spaces (Tier Gratuito) |
| **SDK** | Docker |
| **Hardware** | CPU básico (2 vCPU · 16 GB RAM) |
| **Almacenamiento** | ~50 GB SSD |
| **Modelo Principal** | TinyLlama-1.1B-Chat (cargado desde Hugging Face Hub) |
| **Vector Store** | FAISS con 89 vectores de conocimiento |
| **Git LFS** | Utilizado para archivos de índice (.pkl, .bin) |

### 🔧 Configuración del Entorno en la Nube

```yaml
# Variables de entorno configuradas en HF Spaces
API_HOST: 0.0.0.0
API_PORT: 7860
TEMPERATURE: 0.15
MAX_NEW_TOKENS: 150
TOP_K_RESULTS: 3
SIMILARITY_THRESHOLD: 0.75
DEBUG: False
```

## 📄 Licencia

MIT License - ver archivo [LICENSE](LICENSE) para más detalles.

---

**🔄 Última Actualización**: Marzo 2026

**🏷️ Versión**: 2.0.0 (TinyLlama)
