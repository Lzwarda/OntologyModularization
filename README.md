# Modularisation d'Ontologie MMC

Ce projet implémente une technique de partitionnement d'une ontologie basé sur un Modèle de Markov Caché (MMC).

## Description

Ce projet permet de rendre modulaire les ontologies en utilisant une approche basée sur les Modèles de Markov Cachés. 
Il est décomposé en deux fichiers : 
	- "chargement_fichier.py" : pour l'extraction des triplets RDF à partir des requêtes SPARQL que une ontologie.
	- "modularisation_Ontologie_MMC.py" : pour l'exploitation des triplets RDF afin de générer les partitions de l'ontologie.

## Fonctionnalités principales

- Extraction des triplets RDF d'une ontolpogie
- Traitement de fichiers de triplets RDF
- Construction de chaînes de Markov pour représenter les relations
- Initialisation et entraînement des Modèles de Markov Cachés
- Construction des vecteurs et partitionnement des points 
- Calcul de distances intra-module et inter-module

## Prérequis

- Python 3.x
- Les dépendances suivantes sont nécessaires :
  - numpy
  - scikit-learn
  - matplotlib
  - owlready2

## Installation

1. Cloner le dépôt
3. Mettre à jours les fichers en indiquant les différents chemins des fichiers
2. Installer les dépendances via pip :
	"""bash
	pip install numpy scikit-learn matplotlib
	"""

## Utilisation

1. Télécharger votre ontologie à modulariser
1. Préparer les requêtes SPARQL à exécuter (éventuellement)
2. Exécuter le script : "chargement_fichier.py"
3. Exécuter le script : "modularisation_Ontologie_MMC.py"

Le script "modularisation_Ontologie_MMC.py" générera  :
	- les graphiques (courbe d'elbow et nuage des points)
	- le fichier de sortie contenant les étiquettes des concepts et relations, les nombres d'éléments par partitions, les distances intra-module et inter-module ainsi que le temps d'exécution

## Auteur

Ce projet a été développé par Warda Lazarre.

