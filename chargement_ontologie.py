from owlready2 import *

#Fonction qui extrait les triplets et les range dans un fichier txt en séparant les éléments par des tabulations
def execution_SPARQL(fichier_onto):
    
    #Extraction et sauvegarde des triplets basés sur la subsomption des concepts
    trip = list(default_world.sparql("""SELECT ?s  ?o WHERE {?s rdfs:subClassOf ?o . ?s a owl:Class . ?o a owl:Class .}"""))
    for t in trip:
        ch1 = str(t[0])+ "\t" + "http://www.w3.org/2000/01/rdf-schema#subClassOf\t" + str(t[1]) +"\n"
        fichier_onto.write(ch1)
    #Extraction et sauvegarde des triplets basés sur les relations de type ObjectProperty
    trip2 = list(default_world.sparql("""SELECT ?s ?p ?o WHERE {?s a owl:Class . ?o a owl:Class . ?p a owl:ObjectProperty . ?p rdfs:domain ?s . ?p rdfs:range ?o .}"""))
    for t in trip2:
        ch2 = str(t[0])+ "\t" + str(t[1])+ "\t" + str(t[2]) + "\n"
        fichier_onto.write(ch2)
    
# Fonction principale
ontologie = r"Chemin\vers\ontologie\owl"
triplets = ontologie[:len(ontologie) - 4]
triplets = triplets + ".txt"
ontos = get_ontology(ontologie).load()
fichier = (open(triplets, "x"))
ontos = get_ontology(ontologie).load()
print("Nom de l'ontologie : \t" + ontologie)
        
execution_SPARQL(fichier)                      # Exécution des requêtes SPARQL sur l'ontologie
fichier.close()