"""Test script for improved RAG generation."""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.generator import TinyLlamaGenerator


def test_mejorado():
    """Test the improved prompt."""
    print("=== Probando prompt mejorado ===\n")

    print("Inicializando TinyLlamaGenerator...")
    gen = TinyLlamaGenerator(use_quantization=False)

    contexto = "Para retomar estudios, el estudiante debe contactar a su tutor y entregar actividades pendientes durante el periodo de recuperación de 10 días hábiles."
    pregunta = "Quiero retomar mis estudios, ¿qué debo hacer?"

    print(f"\nContexto: {contexto}")
    print(f"Pregunta: {pregunta}\n")

    start = time.time()
    respuesta = gen.generate_with_context(contexto, pregunta)
    elapsed = time.time() - start

    print(f"=== Respuesta mejorada ({elapsed:.1f}s) ===")
    print(respuesta)


if __name__ == "__main__":
    test_mejorado()