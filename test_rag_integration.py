"""Test script for RAG + TinyLlama integration."""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rag_pipeline():
    """Test the complete RAG pipeline with TinyLlama."""
    print("=== Probando integración RAG + TinyLlama ===\n")

    from rag.generator import TinyLlamaGenerator

    print("Inicializando TinyLlamaGenerator...")
    start = time.time()

    try:
        generator = TinyLlamaGenerator(use_quantization=False)
        print(f"Tiempo de carga: {time.time() - start:.1f} segundos\n")
    except Exception as e:
        print(f"Error inicializando generator: {e}")
        return

    print("--- Prueba 1: Generación sin contexto ---")
    query = "¿Qué es TinyLlama?"
    start = time.time()
    response = generator.generate(query, context="")
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Tiempo: {time.time() - start:.1f} segundos\n")

    print("--- Prueba 2: Generación con contexto simulado ---")
    context = """
    TinyLlama es un modelo de lenguaje compacto con 1.1 mil millones de parámetros.
    Fue desarrollado para ser eficiente y ejecutarse en dispositivos con recursos limitados.
    Es ideal para aplicaciones que requieren respuestas rápidas.
    """
    query = "¿Cuántos parámetros tiene TinyLlama?"
    start = time.time()
    response = generator.generate(query, context)
    print(f"Context: {context[:100]}...")
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Tiempo: {time.time() - start:.1f} segundos\n")

    print("--- Prueba 3: Pipeline RAG completo ---")
    try:
        from rag.core import RAGSystem
        from rag.retriever import VectorStoreFAISS

        print("Inicializando sistema RAG completo...")
        rag = RAGSystem()

        query = "¿Qué modelos usa el chatbot?"
        start = time.time()

        response, es_rag, confidence, sources = rag.process_query(query)

        print(f"Query: {query}")
        print(f"Es RAG: {es_rag}")
        print(f"Confianza: {confidence:.2f}")
        print(f"Response: {response}")
        print(f"Fuentes: {len(sources)}")
        print(f"Tiempo: {time.time() - start:.1f} segundos\n")

    except Exception as e:
        print(f"Error en pipeline RAG: {e}")
        import traceback
        traceback.print_exc()

    print("=== Pruebas completadas ===")


if __name__ == "__main__":
    test_rag_pipeline()