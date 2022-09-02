# Widaug

Widaug es un sistema de aumento de datos para aplicaciones de Reconocimiento de Entidades.

## WikidataFunctionalities

Los siguientes archivos permiten manejar Wikidata para el objetivo de Widaug:

- **entity_searcher.py**: Utiliza el archivo "trie.tsv" (tabla que contiene todas las entradas de wikidata para acelerar la búsqueda y manejo de archivos) para hacer una conversión entre nombre e id de las diferentes entradas de Wikidata.

	- `get_id (file, wordlist)`: Dada una lista de nombres de entidades, devuelve una lista con la id correspondiente a cada palabra.
		- file: el archivo de trie.tsv (abierto).
		- wordlist: la lista de palabras.

	- `get_name (file, idlist)`: Dada una lista de id's de Wikidata, devuelve una lista con los nombres correspondientes.
		- file: el archivo de trie.tsv (abierto).
		- idlist: la lista de id's.

-  **wikiquery</span>.py**: Utiliza Wikidata para obtener entidades relevantes para Widaug.

	- `search_sib(entity, relation, lang, limit)`: Obtiene "entidades hermanas" (entidades con el mismo padre) de una entidad.
		- entity: La entidad de origen (dada por su id de Wikidata).
		- relation: La relación correspondiente (dada por su id de Wikidata).
		- lang: El idioma (es, en, fr...) de la consulta.
		- limit: El número de entidades máximo a generar.

	- `search_child(entity, relation, lang, limit)`: Obtiene entidades hijas de una entidad.
		- entity: La entidad de origen (dada por su id de Wikidata).
		- relation: La relación correspondiente (dada por su id de Wikidata).
		- lang: El idioma (es, en, fr...) de la consulta.
		- limit: El número de entidades máximo a generar.

- **test.ipynb**: Contiene un ejemplo de uso de las dos librerias anteriores.

## FinalVersion

Los siguientes archivos permiten implementar los sistemas de aumento de datos propuestos y evaluar su rendimiento en modelos de *Transformers*.

- **DatabaseProcessing.ipynb**: Permite manipular el *dataset* original del reto Profner para implementar los diferentes sistemas de aumento de datos con Wikidata y Word Embeddings.

- **ModelTraining.ipynb**: Permite entrenar modelos de *Transformers* de reconocimiento de entidades para evaluar su rendimiento con los diferentes *datasets*.