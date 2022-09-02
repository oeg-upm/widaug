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