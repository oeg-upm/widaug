# Widaug

Widaug is a data augmentation system for entity recognition using Wikdiata.

## WikidataFunctionalities
¡
The following files are used for the management of Wikidata:

- **entity_searcher.py**: Uses the file "trie.tsv" (table with all Wikidata entries to speed up the search of entities) to do a conversion between name and id of the different Wikidata entries.

	- `get_id (file, wordlist)`: Given a list of entity names, returns a list with the id corresponding to each word.
		- file: the trie.tsv file (open).
		- wordlist: the list of words.

	- `get_name (file, idlist)`: Given a list of Wikidata id's, return a list of corresponding names.
		- file: the trie.tsv file (open).
		- idlist: the list of id's.

-  **wikiquery</span>.py**: Use Wikidata to get entities relevant to Widaug.

	- `search_sib(entity, relation, lang, limit)`: Gets "sibling entities" (entities with the same parent) of an entity.
		- entity: The source entity (given by its Wikidata id).
		- relation: The corresponding relation (given by its Wikidata id).
		- lang: The language (es, en, fr...) of the query.
		- limit: The maximum number of entities to generate.

	- `search_child(entity, relation, lang, limit)`: Gets child entities of an entity.
		- entity: The source entity (given by its Wikidata id).
		- relation: The corresponding relation (given by its Wikidata id).
		- lang: The language (es, en, fr...) of the query.
		- limit: The maximum number of entities to generate.

- **test.ipynb**: An example of the use of the two previous libraries.

## FinalVersion

The following files allow to implement the proposed data augmentation systems and evaluate their performance in Transformer models.

- **DatabaseProcessing.ipynb**: It allows manipulating the original dataset of the Profner challenge to implement the different data augmentation systems with Wikidata and Word Embeddings.

- **ModelTraining.ipynb**: It allows the training of Transformer models for entity recognition to evaluate their performance with the different datasets.

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