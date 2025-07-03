import numpy as np                                 # Gestion des tableaux
import timeit                                      # Gestion du temps d'exécution
import re                                          # Recherche dans les chaines de caractères
import matplotlib.pyplot as plt                    # Gestion des courbes et graphiques
from sklearn.cluster import KMeans                 # Partitionnement
from sklearn.decomposition import PCA              # Réduction de dimension
from sklearn.metrics import silhouette_score       # Calcul des métriques


#Corps de la fonction qui prend un fichier .txt et génère le tableau des concepts, 
#le tableau des relations et l'ensemble des triplets étiquétés

def traiter_fichier(nom_fichier):

    tableau_concepts = []         # Contient les concepts (les chaines de caractères)
    tableau_relations = []        # Contient les relations (les chaines de caractères)
    triplets_etiquetes = []       # Contient les triplets issus du fichier d'entrée sous forme des numéros issus des tableaux précédents

    # Dictionnaires pour stocker les index des concepts et relations
    index_concepts = {}
    index_relations = {}
    with open(nom_fichier, "r", encoding="utf-8", errors="ignore") as fichier:
        for ligne in fichier:
                        
            if re.search(r"some ", ligne) or re.search(r"some\(", ligne) or re.search(r"Not\(", ligne) or re.search(r"Not ", ligne) or re.search(r"\|", ligne) or re.search(r"&", ligne) :
                print(f"⚠ Ligne ignorée (format incorrect) : {ligne.strip()}")
            else:
                elements = ligne.strip().split("\t")  # Séparer par tabulation
                sujet, relation, objet = elements  # Récupération des colonnes
                
                # Ajout des concepts (1ère et 3e colonne) dans Tableau_Concepts
                for concept in [sujet, objet]:
                    if concept not in index_concepts:
                        index_concepts[concept] = len(tableau_concepts)  # Assigner un index
                        tableau_concepts.append(concept)

                # Ajout des relations (2e colonne) dans Tableau_Relations
                if relation not in index_relations:
                    index_relations[relation] = len(tableau_relations)  # Assigner un index
                    tableau_relations.append(relation)

                # Création de la liste Triplets_Etiquetes avec les index
                triplet = [index_concepts[sujet], index_relations[relation], index_concepts[objet]]
                triplets_etiquetes.append(triplet)

    return tableau_concepts,tableau_relations,triplets_etiquetes


# Corps de la fonction qui construit les séquences valides 
# à partir de l'ensemble des triplets étiquetés reçu en paramètres


def construction_chaines_Markov(tableau_etiquetes, N, M):

    # Initialisation de la liste finale des chaînes de Markov
    chaines_Markov = []

    # Vérification si tableau est vide
    if not tableau_etiquetes:
        return chaines_Markov

    # Ajouter le premier triplet dans le premier groupe
    chaines_Markov.append([tableau_etiquetes[0]])

    # Parcourir les triplets à partir du deuxième élément
    for triplet in tableau_etiquetes[1:]:
        sujet, relation, objet = triplet
        ajout = False
        
        # Vérification si le triplet doit être ajouté en début ou en fin d'une chaîne existante
        for chaine in chaines_Markov:
            # Si le sujet correspond à l'objet du dernier triplet de la chaîne, on l'ajoute à la fin
            if chaine[-1][2] == sujet:
                chaine.append(triplet)
                ajout = True
                break  
            # Si l'objet correspond au sujet du premier triplet de la chaîne, on l'ajoute au début
            elif chaine[0][0] == objet:
                chaine.insert(0, triplet)
                ajout = True
                break      
        
        # Si aucune chaîne ne correspond, créer une nouvelle chaîne avec ce triplet
        if not ajout:
            chaines_Markov.append([triplet])

    for i in range(len(chaines_Markov)):
        triplet = [chaines_Markov[i][len(chaines_Markov[i])-1][2], M-1, N-1]
        chaines_Markov[i].append(triplet)
    
    return chaines_Markov

# Corps de la fonction qui initialise les paramètres du MMC
# Elle utilise les chaines de Markov et les nombres de concepts et de relations de l'ontologie

def initilisationMMC(chaines_Markov, N, M, epsilon):
    
    # Initialisation des paramètres du MMC à zéros
    A = np.zeros((N, N))
    B = np.zeros((N, M))
    pi = np.zeros(N)

    K = len(chaines_Markov)
    # Initialisation de la matrice des transitions d'états
    for i in range(N):
        for j in range(N):
            num, denom = 0.0, 0.0
            for k in range(K):
                for o in range(len(chaines_Markov[k])):
                    if(chaines_Markov[k][o][0] == i):
                        if(chaines_Markov[k][o][2] == j):
                            num = num + 1.0
                        denom = denom + 1.0
            
            if(num != 0.0):
                A[i][j] = num / (denom + epsilon) 
    # Normalisation des coefficients de la matrice des transitions d'états
    for i in range(N):
        for j in range(N):
            A[i][j] += (1.0 - sum(A[i])) / N
    # Initialisation de la matrice d'observation des symboles
    for i in range(N):
        for j in range(M):
            num, denom = 0.0, 0.0
            for k in range(K):
                for o in range(len(chaines_Markov[k])):
                    if(chaines_Markov[k][o][0] == i):
                        if(chaines_Markov[k][o][1] == j):
                            num = num + 1.0
                        denom = denom + 1.0
                
            if(num != 0.0):
                B[i][j] = num / (denom + epsilon)
    # Normalisation de la matrice d'observation des symboles
    for i in range(N):
        for j in range(M):
            B[i][j] += (1.0 - sum(B[i])) / M
    # Corps de la fonction qui initalise le vecteur d'états initiaux
    for i in range(N):
        num = 0.0
        for k in range(K):
            if(chaines_Markov[k][0][0] == i):
                num = num + 1.0
        pi[i] = num / (K + epsilon)
    som = sum(pi)
    # Normalisation du vecteur des états initiaux
    for i in range(N):
        pi[i] += (1.0 - som) / N

    return A, B, pi

# Corps de la fonction qui calcule les distances intra-module, inter-module et la silhouette
# Elle prend en entrée les vecteurs, les étiquettes et les centres des clusters

def calculate_distances(X, labels, centroids):

    # Calcul des distances intra-cluster
    intra_distances = []
    for i in range(len(centroids)):
        # Récupérer les points du cluster i
        cluster_points = X[labels == i]
        # Calcul de la distance entre chaque point et son centroïde
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        intra_distances.append(np.mean(distances))  # Moyenne des distances pour ce cluster

    # Moyenne des distances intra-cluster
    D_intra = np.mean(intra_distances)
    
    # Calcul des distances inter-cluster
    inter_distances = []
    K = len(centroids)
    for i in range(K):
        for j in range(i+1, K):
            # Calcul de la distance entre les centroïdes des clusters i et j
            inter_distances.append(np.linalg.norm(centroids[i] - centroids[j]))
    
    # Moyenne des distances inter-cluster
    D_inter = np.mean(inter_distances)

    # Calcul du silhouette score
    silhouette_avg = silhouette_score(X, labels)
    
    return D_intra, D_inter, silhouette_avg


# Définition de la classe qui gère le MMC
# Elle possède entre autres les méthodes liées à l'initialisation 
# de ses attributs et celles d'entrainement du MMC

class MMC:
    # Corps du consttructeur de la classe, elle initialise les paramètres 
    # du MMC avec ceux reçus lors de l'instanciation de la classe
    def __init__(self, A, B, pi):
        self.A_init = A
        self.B_init = B
        self.pi_init = pi

        self.A_train = A
        self.B_train = B
        self.pi_train = pi

        self.N = len(self.A_init[0])
        self.M = len(self.B_init[0])

    # Corps de la fonction forward, elle permet de calculer les coefficients 
    # de la variable Alpha utilisée lors de l'entrainement du MMC
    def forward(self, A, B, pi, O):

        T = len(O)
        alpha = np.zeros((T, self.N))

        for i in range(self.N):
            alpha[0][i] = pi[i] * B[i][O[0]]

        for t in range(1, T):
            for j in range(self.N):
                som = 0.0
                for i in range(self.N):
                    som += alpha[t-1][i] * A[i][j]
                alpha[t][j] = som * B[j][O[t]]

        return alpha
    
    # Corps de la fonction backward, elle permet de claculer les coefficients 
    # de la variable Beta utilisée lors de l'entrainement du MMC
    
    def backward(self, A, B, O):

        T = len(O)
        beta = np.zeros((T, self.N))

        for i in range(self.N):
            beta[T-1][i] = 1.0

        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t][i] = 0.0
                for j in range(self.N):
                    beta[t][i] += A[i][j] * B[j][O[t+1]] * beta[t+1][j]

        return beta
    
    # Corps de la fonction xi_calcul, elle permet de calculer les coefficients 
    # des variables Xi et Gamma utilisées lors de l'entrainement du MMC
    
    def xi_calcul(self, alpha, beta, A, B, O):

        T = len(O)
        xi = np.zeros((T, self.N, self.N))
        gamma = np.zeros((T, self.N))
        
        for t in range(T-1):
            denom = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    denom += alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
        
            #print("Dénominateur : \t \t ", denom)
            for i in range(self.N):
                for j in range(self.N):
                    if(denom != 0.0):
                        xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]) / denom
                
            denom2 = 0.0   
            for i in range(self.N):
                denom2 += alpha[t][i] * beta[t][i]
            
            for i in range(self.N):
                if(denom2 != 0.0): 
                    gamma[t][i] = (alpha[t][i] * beta[t][i]) / denom2
        
        return xi, gamma
    
    # Corps de la fonction forwardBackward, elle permet de calculer 
    # la probabilité d'observer une séquence d'observations

    def forwardBackward(self, alpha):
        prob = 0.0
        for i in range(self.N):
            prob += alpha[len(alpha)-1][i]
        return prob
    
    # Corps de la fonction reestimation, elle permet de mettre à jour les 
    # paramètres du MMC en réévaluant leurs coefficients, elle utilise 
    # les variables Xi, Gamma et les séquences d'observations

    def reestimation(self, xi_K, gamma_K, sequences):

        A_t = np.zeros((self.N, self.N))
        B_t = np.zeros((self.N, self.M))
        pi_t = np.zeros((self.N))

        K = len(sequences)
        for i in range(self.N):
            # Mise à jour de chaque coefficient de Pi
            pii = 0.0
            for k in range(K):
                pii += gamma_K[k][0][i]
            pi_t[i] = pii / K

            # Mise à jour de chaque ligne de la matrice A
            for j in range(self.N):
                ai = 0.0
                aid = 0.0
                for k in range(K):
                    for t in range(len(sequences[k])-1):
                        ai += xi_K[k][t][i][j]
                        aid += gamma_K[k][t][i]

                if (aid != 0.0):
                    A_t[i][j] = ai / aid
            
            # Mise à jour de chaque ligne de la matrice B
            for j in range(self.M):
                bi = 0.0
                bid = 0.0
                for k in range(K):
                    for t in range(len(sequences[k])-1):
                        if(j == sequences[k][t]):
                            bi += gamma_K[k][t][i]
                        bid += gamma_K[k][t][i]
                if(bid != 0.0):
                    B_t[i][j] = bi / bid

        return A_t, B_t, pi_t

    # Corps de la fonction baum_welch_multisequence. 
    # Elle permet d'entrainer le MMC, c'est-à-dire mettre à jour les paramètres du MMC 
    # jusqu'à satisfaction des contraintes. Elle fait appel à plusieurs autres fonctions.
    
    def baum_welch_multisequence(self, sequences, max_iter = 100, seuil = 0.0001):
        K = len(sequences)
        # Initialisation du MMC initial
        prob_init = np.zeros((K))
        A1 = self.A_init.copy()
        B1 = self.B_init.copy()
        pi1 = self.pi_init.copy()

        # Calcul des probabilités initiales des séquences en utilisant la fonction forwardaBackward
        for k in range(K):
            prob_init[k] = self.forwardBackward(self.forward(A1, B1, pi1, sequences[k]))

        iter = 0
        xi_K = []
        gamma_K = []

        # Boucle sur le nombre d'itérations pour l'entrainement
        while(iter <= max_iter):
            iter += 1
            # Calcul des coefficients des varaibles alpha, beta, xi et gamma de chaque séquence
            for k in range(K):
                alpha = self.forward(A1, B1, pi1, sequences[k])
                beta = self.backward(A1, B1, sequences[k])
                x, g = self.xi_calcul(alpha, beta, A1, B1, sequences[k])
                xi_K.append(x)
                gamma_K.append(g)

            # Réévaluation des paramètres du MMC
            A1, B1, pi1 = self.reestimation(xi_K, gamma_K, sequences)

            prob_cal = np.zeros(K)
            # Calcul des nouvelles probabilités d'observer les séquences avec les nouveaux paramètres du MMC
            for k in range(K):
                prob_cal[k] = self.forwardBackward(self.forward(A1, B1, pi1, sequences[k]))

            p = 0
            for k in range(K):
                if((prob_cal[k] - prob_init[k]) > seuil):
                    p += 1
            # Arrêt de la boucle s'il n'y a pas grande amélioration. Dans le cas présent, s'il y a amélioration des probabilités 
            # de la moitié des séquence, on continue, sinon on arrête la boucle. On peut toutefois changer cette condition
            if(p < K/2):
                return prob_init, iter

            # Mise à jour du MMC et des probabilités d'observer les séquences
            self.A_train = A1.copy()
            self.B_train = B1.copy()
            self.pi_train = pi1.copy()
            prob_init = prob_cal.copy()

        return prob_init, iter
    
# Début de la fonction principale

temps_debut = timeit.default_timer()                                           #Initialisation du temps d'exécution
ontologie = "Nom_Ontologie"                                                 
triplets_rdf = "Chemin/Vers/LeDossier/" + ontologie + ".txt"                   #Fichier txt contenant les triplets 
etiquettes = "Chemin/Vers/LeDossier/" + ontologie + "_sorties.txt"             #Fichier txt qui va contenir les éléments de sorties du programme
vect = "Chemin/Vers/LeDossier/" + ontologie + ".csv"                           #Fichier csv qui va contenir les vecteurs pour le K-Means
elbow = "Chemin/Vers/LeDossier/" + ontologie + "_elbow.png"                    #Fichier png qui va contenir la courbe d'elbow pour le clustering
nuage = "Chemin/Vers/LeDossier/" + ontologie + "_nuage.png"                    #Fichier png qui va contenir les clusters sous 3D

Tableau_Concepts,Tableau_Relations,Triplets_Etiquetes = traiter_fichier(triplets_rdf)  # Génération des triplets étiquettés et des étiquettes

N = len(Tableau_Concepts) + 1
M = len(Tableau_Relations) + 1
chaines_Markov = construction_chaines_Markov(Triplets_Etiquetes, N, M)                 # Construction des chaines de Markov
Tableau_Relations.append("subClassOfThing")                                            # Ajout de la relation fictive
Tableau_Concepts.append("OwlThing")                                                    # Ajout du concept fictif

sortie = open(etiquettes, "w")
# Ecriture de la liste des concepts et des relations
sortie.write(f"Liste des Concepts de l'ontologie {ontologie}\n\n")
for i, el in enumerate(Tableau_Concepts):
    sortie.write(f"\t{i}: \t{el}\n")
sortie.write(f"\n\n *************************************\nListe des Relations de l'ontologie {ontologie}\n\n")
for i, el in enumerate(Tableau_Relations):
    sortie.write(f"\t{i}: \t{el}\n")

A, B, pi = initilisationMMC(chaines_Markov, N, M, 0.5)                                # Initialisation des paramètres du MMC avec les chaines de Markov

sequences = [[triplet[1] for triplet in seq] for seq in chaines_Markov]               # Extraction des séquences d'observations dans les chaines de Markov

mmc = MMC(A, B, pi)                                                                   # Instanciation de la classe MMC
prob, iter = mmc.baum_welch_multisequence(sequences, 100, 1/(10**(-N/10)))            # Entrainement du MMC

print("Probabilité après entrainement : \n", prob)
print("Iterations : ", iter)
print("\n***********************\n")

vecteurs = np.hstack((mmc.A_train[:-1,:-1], mmc.B_train[:-1,:-1]))                   # Construction des vecteurs pour le partitionnement
    
np.savetxt(vect, vecteurs, delimiter="\t", fmt="%f")                                 # Enregistrement des vecteurs




inertias = []                                                                        # Stocke les inerties pour chaque valeur de k
K_range = range(1, int(len(vecteurs)/10))                                            # Tester de 1 à 10 clusters
# recherche du nombre de clusters optimal avec la méthode d'elbow
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(vecteurs)
    inertias.append(kmeans.inertia_)  # Ajouter l'inertie du modèle

# Affiche de la courbe pour lecture graphique
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', markersize=8, label="Inertie")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie intra-cluster")
plt.title(ontologie)
plt.legend()
plt.grid(True)
plt.savefig(elbow, dpi=300)
plt.show()

optimal_k = int(input("Entrez le nombre optimal de clusters basé sur le graphique : "))

pca = PCA(n_components=2)                                                            # Rédutcion à 2 dimensions pour visualiser
vecteurs_2D = pca.fit_transform(vecteurs)                                            # Projection en 2D

kmeans_final = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)               # Exécution du Kmeans avec le nombre optimal de clusters
clusters = kmeans_final.fit_predict(vecteurs)
centroids = kmeans_final.cluster_centers_                                            # Récupération des centroïdes (centres des clusters)

plt.figure(figsize=(8, 6))                                                           

for i in range(optimal_k):
    plt.scatter(vecteurs_2D[clusters == i, 0], vecteurs_2D[clusters == i, 1], label=f"Cluster {i}")

# Calcul des distances intra-cluster, inter-cluster et silhouette score
D_intra, D_inter, silhouette_avg = calculate_distances(vecteurs, kmeans.labels_, kmeans.cluster_centers_)
sortie.write("\n\n***********************\n\nSorties\n\n")
sortie.write(f"\tDistance intra-cluster : {D_intra}\n")
sortie.write(f"\tDistance inter-cluster : {D_inter}\n")
sortie.write(f"\tSilhouette score : {silhouette_avg}")

# Affichage du nuage de points avec les clusters
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title(f"{ontologie} avec {optimal_k} clusters")
plt.legend()
plt.grid(True)
plt.savefig(nuage, dpi=300)
plt.show()

unique, counts = np.unique(clusters, return_counts=True)                               # Décompte des éléments par cluster
cluster_counts = dict(zip(unique, counts))
sortie.write("\n***********************\n")
nb_vecteurs = len(vecteurs)
sortie.write(f"Nombre total d'éléments : {nb_vecteurs} \n")
sortie.write("Nombre d'éléments par cluster : \n")
maximum_clusters = 0
for cluster_id, count in cluster_counts.items():
    sortie.write(f"\tCluster {cluster_id} : {count} éléments\n")
    if count > maximum_clusters:
        maximum_clusters = count

sortie.write("\n***********************\n")
sortie.write(f"Pourcentage du maximum : {maximum_clusters*100.0 / nb_vecteurs}\n")    # Calcul du pourcentage du plus grand cluster par rapport au nombre total des éléments
temps_execution = timeit.default_timer() - temps_debut                                # Calcul du temps d'exécution de tout le script
sortie.write("\n***********************\n")
sortie.write(f"Temps d'exécution : {temps_execution:.4f} secondes")                   # Affichage du temps d'exécution
sortie.close()