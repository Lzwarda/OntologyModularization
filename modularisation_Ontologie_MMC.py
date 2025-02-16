import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


#Corps de la fonction qui prend un fichier .txt et génère le tableau des concepts, 
#le tableau des relations et l'ensemble des triplets étiquétés
#ELle correspond à l'algorithme d'étiquettage des triplets

def traiter_fichier(nom_fichier):

    tableau_concepts = []         # Contient les concepts (les chaines de caractères)
    tableau_relations = []        # Contient les relations (les chaines de caractères)
    triplets_etiquetes = []       # Contient les triplets issus du fichier d'entrée sous forme des numéros issus des tableaux précédents

    # Dictionnaires pour stocker les index des concepts et relations
    index_concepts = {}
    index_relations = {}

    with open(nom_fichier, "r", encoding="utf-8", errors="ignore") as fichier:
        for ligne in fichier:
            elements = ligne.strip().split("\t")  # Séparer par tabulation
            
            if len(elements) != 3:
                print(f"⚠ Ligne ignorée (format incorrect) : {ligne.strip()}")
                continue

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
# à partir de l'ensemble des séquences reçu en paramètres

def ajustement_chaines_Markov(chaine, M):
    chaines_correctes = []                    # Contirndra les chaines qui ont au moins deux triplets
    chaines_uniques = []                      # Contiendra les chaines qui n'ont qu'un seul triplet ou de longueur 1
    uniques = []                              # Contiendra la fusion des chaines ayant un seul triplet

    for ch in chaine :                        # Collection des séquences ayant un seul triplet
        if len(ch) == 1:                      
            chaines_uniques.append(ch)        
        else:
            chaines_correctes.append(ch)

    uniques.append(chaines_uniques[0][0])

    for i in range(1, len(chaines_uniques)):             # Assemblage des séquences de longueur 1 pour former une seule séquence
        triplet = [chaines_uniques[i-1][0][2], M, chaines_uniques[i][0][0]]
        uniques.append(triplet)
        uniques.append(chaines_uniques[i][0])

    chaines_correctes.append(uniques)

    return chaines_correctes


# Corps de la fonction qui prend en entrée le tableau 
# des triplets étiquettés et génère les chaines de Markov

def construction_chaines_Markov(tableau_etiquetes):

    # Initialisation de la liste finale des chaînes de Markov
    chaines_Markov = []

    # Vérification si tableau est vide
    if not tableau_etiquetes:
        return chaines_Markov

    # Ajout du premier triplet dans le premier groupe
    chaines_Markov.append([tableau_etiquetes[0]])

    # Parcours les triplets à partir du deuxième élément
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
    
    return chaines_Markov


# Corps de la fonction qui prend en entrée les les chaines de Markov correctes, le nombre de concepts, 
# le nombre de relation et le biais pour calculer les composantes de deux matrices (A et B) et 
# d'un vecteur (pi) qui seront considérés comme les paramètres d'un MMC

def initilisationMMC(chaines_Markov, N, M, epsilon):
    
    # Initialisation des matrices et vecteur avec des valeurs nulles
    A = np.zeros((N, N))
    B = np.zeros((N, M))
    pi = np.zeros(N)

    K = len(chaines_Markov)                 # Calcul du nombre de chaines de Markov

    #Initialisation de la matrice A
    for i in range(N):
        for j in range(N):
            num, denom = 0.0, 0.0
            for k in range(K):
                for o in range(len(chaines_Markov[k])):
                    if(chaines_Markov[k][o][0] == i):        # Lorsque le sujet du triplet courant est i
                        if(chaines_Markov[k][o][2] == j):    # Lorsque l'objet du triplet courant est j
                            num = num + 1.0                  
                        denom = denom + 1.0
            
            if(num != 0.0):
                A[i][j] = num / (denom + epsilon)            # Calcul de la composante A[i][j] en ajoutant le biais au dénominateur

    for i in range(N):
        for j in range(N):
            A[i][j] += (1.0 - sum(A[i])) / N                 # Partage égale de la différence entre 1 et la somme des éléments d'une ligne

    #Initialisation de la matrice B
    for i in range(N):
        for j in range(M):
            num, denom = 0.0, 0.0
            for k in range(K):
                for o in range(len(chaines_Markov[k])):
                    if(chaines_Markov[k][o][0] == i):        # Lorsque le sujet du triplet courant est i
                        if(chaines_Markov[k][o][1] == j):    # Lorsque le prédicat du triplet courant est j
                            num = num + 1.0
                        denom = denom + 1.0
                
            if(num != 0.0):
                B[i][j] = num / (denom + epsilon)            # Calcul de la composante B[i][j] en ajoutant le biais au dénominateur

    for i in range(N):
        for j in range(M):
            B[i][j] += (1.0 - sum(B[i])) / M                 # Partage égale de la différence entre 1 et la somme des éléments d'une ligne   

    for i in range(N):
        num = 0.0
        for k in range(K):
            if(chaines_Markov[k][0][0] == i):                # Lorsque le sujet du premier triplet de la chaine est i
                num = num + 1.0
        pi[i] = num / (K + epsilon)                          # Calcul de la composante pi[i] en ajoutant le biais au dénominateur
    som = sum(pi)
    for i in range(N):
        pi[i] += (1.0 - som) / N                             # Partage égale de la différence entre 1 et la somme des éléments

    return A, B, pi


#Corps de la fonction qui prend en entrées les centroides ( centroids), 
# les ensembles de points (X) et les numéro des clusters des points (labels) 
# puis calcule les distances intra-cluster (intra-module), inter-cluster(inter-module) 
# et la silhouette score après l'exécution des K-Moyennes

def calculate_distances(X, labels, centroids):

    # Calcul des distances intra-cluster pour chaque cluster
    intra_distances = []
    for i in range(len(centroids)):
        cluster_points = X[labels == i]                                   # Récupérer les points du cluster i
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1) # Calcul de la distance entre chaque point et son centroïde
        intra_distances.append(np.mean(distances))                        # Moyenne des distances pour ce cluster
    D_intra = np.mean(intra_distances)                                    # Moyenne des distances intra-cluster
    
    # Calcul des distances inter-cluster
    inter_distances = []
    K = len(centroids)
    for i in range(K):
        for j in range(i+1, K):
            inter_distances.append(np.linalg.norm(centroids[i] - centroids[j]))     # Calcul de la distance entre les centroïdes des clusters i et j
    D_inter = np.mean(inter_distances)                                              # Moyenne des distances inter-cluster

    # Calcul du silhouette score
    silhouette_avg = silhouette_score(X, labels)
    
    return D_intra, D_inter, silhouette_avg


#Classe qui gère le MMC et implémente les méthodes sur le MMC

class MMC:
    #Corps du constructeur de la classe, il prend en entrée deux matrices (A et B) et un vecteur (pi)
    def __init__(self, A, B, pi):
        #Initialisation des paramètres initiaux du MMC
        self.A_init = A
        self.B_init = B
        self.pi_init = pi
        #Initialisation des paramètres entrainés, on peut aussi ommettre cette étape
        self.A_train = A
        self.B_train = B
        self.pi_train = pi

        self.N = len(self.A_init[0])       # Calcul du nombre d'états du MMC
        self.M = len(self.B_init[0])       # Calcul du nombre de symboles du MMC

    # Corps de la fonction forward() qui prend en entrée les paramètres du MMC et 
    # une séquences d'observations et calcule les variables alpha pour l'entrainement

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
    
    # Corps de la fonction backward() qui prend en entrée les paramètres du MMC et 
    # une séquence d'observations et calcule les variables beta pour l'entrainement
    
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
    
    # Corps de la fonction qui prend les variables alpha, beta, 
    # les paramètres du MMC et une séquence et calcule les composantes des variables Xi et Gamma
    
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
    
    #Corps de la fonction forwardBackward() qui calcule la probabilité d'observer la séquence
    
    def forwardBackward(self, alpha):
        prob = 0.0
        for i in range(self.N):
            prob += alpha[len(alpha)-1][i]
        return prob
    
    # Corps de la fonction qui réestime les paramètres d'un MMC avec les variables Xi, Gamma et les séquences

    def reestimation(self, xi_K, gamma_K, sequences):

        A_t = np.zeros((self.N, self.N))
        B_t = np.zeros((self.N, self.M))
        pi_t = np.zeros((self.N))

        K = len(sequences)
        for i in range(self.N):
            pii = 0.0
            for k in range(K):
                pii += gamma_K[k][0][i]
            pi_t[i] = pii / K

            for j in range(self.N):
                ai = 0.0
                aid = 0.0
                for k in range(K):
                    for t in range(len(sequences[k])-1):
                        ai += xi_K[k][t][i][j]
                        aid += gamma_K[k][t][i]

                if (aid != 0.0):
                    A_t[i][j] = ai / aid

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

    # Corps de la fonction Baum_welch_Multisequence qui fait l'entrainement du MMC sur les séquences

    def baum_welch_multisequence(self, sequences, max_iter = 100, seuil = 0.0001):
        K = len(sequences)
        prob_init = np.zeros((K))
        A1 = self.A_init.copy()
        B1 = self.B_init.copy()
        pi1 = self.pi_init.copy()

        # Calcul des probabilités initiales
        for k in range(K):
            prob_init[k] = self.forwardBackward(self.forward(A1, B1, pi1, sequences[k]))

        iter = 0
        xi_K = []
        gamma_K = []

        # Debut des itérations
        while(iter <= max_iter):
            iter += 1
            # Parcours des séquences et calcul des variables xi et gamma pour toutes les séquences
            for k in range(K):
                alpha = self.forward(A1, B1, pi1, sequences[k])
                beta = self.backward(A1, B1, sequences[k])
                x, g = self.xi_calcul(alpha, beta, A1, B1, sequences[k])
                xi_K.append(x)
                gamma_K.append(g)

            A1, B1, pi1 = self.reestimation(xi_K, gamma_K, sequences)

            prob_cal = np.zeros(K)
            # Calcul des nouvelles probabilités
            for k in range(K):
                prob_cal[k] = self.forwardBackward(self.forward(A1, B1, pi1, sequences[k]))

            p = 0
            for k in range(K):
                if((prob_cal[k] - prob_init[k]) > seuil):
                    p += 1
            if(p < K/2):
                return prob_init, iter            # Arret après vérification des seuils de probabilités

            self.A_train = A1.copy()
            self.B_train = B1.copy()
            self.pi_train = pi1.copy()

            prob_init = prob_cal.copy()

        return prob_init, iter
    
# Debut de la fonction principale
temps_debut = timeit.default_timer()             # Capture du temps de debut

triplets_rdf = r"Chemin\fichier\triplets\txt"    # Le fichier issu de la sortie du code cchargement_ontologie.py 
# Appel de la fonction traiter_fichier() et recupération 
# des tableaux de concepts, relations et triplets étiquetés
Tableau_Concepts,Tableau_Relations,Triplets_Etiquetes = traiter_fichier(triplets_rdf) 
N = len(Tableau_Concepts)        # Nombre d'états du modèle
M = len(Tableau_Relations)       # Nimbre de symboles du modèle
# Appel de la fonction construction_chaines_Markov() pour construire les chaines de Markov
chaines_Markov = construction_chaines_Markov(Triplets_Etiquetes)
# Appel de la fonction ajustement_chaines_Markov() pour reorganiser les chaines de Markov
chaines = ajustement_chaines_Markov(chaines_Markov, M)
Tableau_Relations.append("fictif")       # Ajout d'une relation "fictif" à la liste des relations
# Appel de la foncrion initilisationMMC() pour initialiser les matrices A et B et le vecteur pi
A, B, pi = initilisationMMC(chaines, N, M+1, 0.5)
# Extraction des prédicats dans les triplets pour former les séquences de symboles
sequences = [[triplet[1] for triplet in seq] for seq in chaines]
# Création de l'objet mmc comme instance de la classe MMC
mmc = MMC(A, B, pi)
# Appel de la fonction baum_welch_multisequence() pour entrainer l'objet mmc et récupérer 
# la probabilité finale et le nombre d'itérations effectuées
prob, iter = mmc.baum_welch_multisequence(sequences, 100, 0.000001)
# Affichage de la probabilité et du nombre d'itérations
print("Prob : ", prob)
print("Iterations : ", iter)
# Fusion des lignes des matrices A et B du mmc entrainé pour former les vecteurs d'entrée des K-Moyennes
vecteurs = np.hstack((mmc.A_train, mmc.B_train))
# Sauvegarde des vecteurs dans un fichier .csv 
np.savetxt(r"Chemin\saubegarde\.csv", vecteurs, delimiter="\t", fmt="%f")

inertias = []  # Stocke les inerties pour chaque valeur de k
K_range = range(1, int(len(vecteurs)/10))  # Tester de 1 à la moitié du nombre de vecteurs
# Variation du nombre de clusters pour chercher celui qui est optimal
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(vecteurs)
    inertias.append(kmeans.inertia_)  # Ajouter l'inertie du modèle

#  Affichage de la courbe de la méthode Elbow 
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', markersize=8, label="Inertie")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie intra-cluster")
plt.title("Méthode Elbow pour choisir k")
plt.legend()
plt.grid()
plt.show()

# Choix du nombre optimal de clusters
optimal_k = int(input("Entrez le nombre optimal de clusters basé sur le graphique : "))
# Réduction de la dimenstion des vecteurs à 2 pour visualiser les clusters
pca = PCA(n_components=2)                              # Reduction à 2 dimensions 
vecteurs_2D = pca.fit_transform(vecteurs)              # Projection en 2D"
# Application des K-Moyennes (K-means) avec le nombre optimal de clusters
kmeans_final = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
clusters = kmeans_final.fit_predict(vecteurs)
centroids = kmeans_final.cluster_centers_              # Récupération des centroïdes

# Affichage du nuage de points avec les clusters 
plt.figure(figsize=(8, 6))

# Traçage de chaque point avec une couleur correspondant à son cluster
for i in range(optimal_k):
    plt.scatter(vecteurs_2D[clusters == i, 0], vecteurs_2D[clusters == i, 1], label=f"Cluster {i}")

# Calcul des distances intra-cluster, inter-cluster et silhouette score avec la fonction calculate_distances()
D_intra, D_inter, silhouette_avg = calculate_distances(vecteurs, kmeans.labels_, kmeans.cluster_centers_)

# Affichage des résultats des distances et silhouette score
print(f"Distance intra-cluster : {D_intra}")
print(f"Distance inter-cluster : {D_inter}")
print(f"Silhouette score : {silhouette_avg}")
# Affichage du nuage des points
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title(f"Clustering K-Means avec {optimal_k} clusters")
plt.legend()
plt.grid()
plt.show()

# Comptage des éléments par cluster 
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Affichage des nombres d'éléments par cluster
print("Nombre d'éléments par cluster :")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id} : {count} éléments")
# Calcul et affichage du temps d'exécution
temps_execution = timeit.default_timer() - temps_debut
print(f"Temps d'exécution : {temps_execution:.4f} secondes")