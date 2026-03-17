# test_extractor.py
from rag.generator import TinyLlamaGenerator
gen = TinyLlamaGenerator()

contexto = "¿Qué pasa si no tengo mi certificado? Tienes 6 meses para presentarlo. Durante ese periodo puedes continuar tus estudios bajo carta compromiso. Si no lo presentas en el plazo, tu inscripción será cancelada."
pregunta = "¿Qué pasa si no tengo mi certificado?"

respuesta = gen.generate_with_context(contexto, pregunta)
print(f"EXTRACCIÓN:\n{respuesta}")