from models import TinyLlamaWrapper
import time

start = time.time()
print('Cargando modelo (esto tomará 20-40 segundos)...')
wrapper = TinyLlamaWrapper(use_quantization=False)

print('Generando respuesta...')
respuesta = wrapper.generate('Hola, ¿cómo estás?')
print(f'Respuesta: {respuesta}')
print(f'Tiempo total: {time.time()-start:.1f} segundos')