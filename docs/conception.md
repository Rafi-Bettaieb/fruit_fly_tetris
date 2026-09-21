# Une mouche qui joue à Tetris — connectome MaleCNS v1.0 entraîné par imitation

**Document de conception — septembre 2026.** Ce document ne contient pas de code. Toutes les décisions y sont prises : il dit précisément ce qui sera fait, avec quels réglages et dans quel ordre.

- « Attendu » : valeur que la phase 0 doit produire. **Aucun chiffre de jeu n'a encore été mesuré : le dépôt ne contient aucun code.** Les chiffres de la section 12 sont des attentes, pas des mesures, tant que la phase 0 n'a pas tourné.
- « Estimation » : valeur prévue pour le matériel. La mesure réelle remplace l'estimation dès que le matériel a été mesuré (étape D, section 17).
- « À vérifier » : chiffre repris d'une source extérieure, à confronter aux données.

## Sommaire

1. Résumé du projet
2. Les projets existants et la recette retenue
3. Honnêteté scientifique
4. Outils, données et matériel
5. Le jeu : règles retenues
6. Le moteur de jeu (faits mesurés)
7. Le connectome : données, signes, populations
8. Le modèle neuronal
9. Entrée, sortie et protocole temporel
10. Entraînement
11. Évaluation et contrôles
12. Points de référence attendus — première expérience du projet
13. Interface 3D et démo
14. Performance attendue sur le matériel
15. Structure du projet
16. Qualité, reproductibilité, MLOps
17. Plan de réalisation
18. Risques et parades
19. Récapitulatif des décisions
20. Hors périmètre
21. Références

Annexes : glossaire, check-list de démarrage.

---

## 1. Résumé du projet

**Le projet.** Faire jouer à Tetris un réseau de neurones dont le câblage est celui du système nerveux central complet d'une drosophile mâle, mesuré au microscope électronique. Le jeu de données est le MaleCNS v1.0 : 165 122 neurones tracés et 25 563 197 connexions entre neurones. Le projet suit la méthode des projets communautaires de septembre 2026 qui ont entraîné ce même connectome à conduire une voiture simulée.

**Le principe en une phrase.** Pour chaque pièce, le programme calcule toutes les façons de la poser ; la mouche regarde chaque grille qui en résulterait et lui donne une note ; la pièce est posée là où la note est la meilleure.

**Les six principes de la recette :**

1. **Câblage figé.** Aucune connexion n'est ajoutée, retirée ou déplacée.
2. **Forces apprises.** Seuls s'entraînent un gain par connexion, une fuite par neurone et une température pour les notes, soit environ 25,7 millions de paramètres.
3. **Interfaces figées.** La grille entre par 4 114 neurones visuels et la note sort de 708 neurones moteurs, via des projections aléatoires tirées une fois pour toutes au début d'un run. Le seul chemin entre la grille et la note passe par le câblage mesuré.
4. **Temps de réflexion.** Chaque grille est présentée pendant K mises à jour du réseau, K étant calculé pour que le signal atteigne les neurones moteurs.
5. **Imitation, puis DAgger.** La mouche imite d'abord un expert. Elle joue ensuite seule pendant que l'expert corrige les situations où elle s'est mise.
6. **Contrôles.** La même recette est appliquée à une mouche au câblage recâblé au hasard. La mouche est aussi comparée à un juge de grilles linéaire et à deux témoins (graphe coupé, entrée aveuglée).

**Pourquoi Tetris :**

- **Le talent se mesure.** Un joueur au hasard complète environ 0,1 ligne par partie ; l'expert en complète environ 195 sur 500 pièces (attendu, section 12). Chaque progrès de la mouche se voit.
- **Le jeu est visuel.** La grille entre par les neurones visuels, comme l'image de la route dans le projet de référence.
- **Le professeur existe déjà.** C'est une heuristique classique de Tetris, forte et rapide.
- **La démo parle d'elle-même.** Tout le monde voit si la mouche joue bien.
- **Le sujet est libre.** Aucun projet « mouche + Tetris » n'a été trouvé dans la vague actuelle.

**Deux livrables, définis par leur contenu et non par une date :**

1. **v0.1 :** la mouche entraînée sur une graine, la mouche recâblée, le juge linéaire et les deux témoins, chaque condition évaluée sur les 100 parties du protocole final ; la page publique en ligne, en mode secours, puisque le GPU est encore pris par les entraînements.
2. **v1.0 :** les résultats sur trois graines, le direct public allumé — la mouche joue en continu et tout le monde regarde la même partie — et le mode duel en local.

**Objectifs chiffrés** (plafond de 500 pièces) :

- **minimum :** dépasser le juge de grilles linéaire, **mesuré à 32,7 lignes par partie** ;
- **cible :** dépasser 62,6 lignes par partie, score attendu d'un MLP juge de grilles à 128 neurones cachés. Ce MLP n'est pas entraîné dans la v0.1 : le seuil reste un repère de la littérature tant qu'il n'est pas mesuré ;

La barre minimale est **plus haute que prévu** : le juge linéaire mesuré fait 32,7 lignes, et non les 13,8 attendues (section 12). C'est ce chiffre-là qui fait foi.

- **dans tous les cas :** mesurer l'écart entre la mouche et la mouche recâblée, avec intervalles de confiance.

Ces deux seuils sont les chiffres de la section 12. Les modèles simples étant réentraînés en phase 2 avec le protocole final, les seuils qui font foi pour les décisions sont leurs scores réentraînés, mesurés dans le même protocole que la comparaison (section 12).

Le résultat est publié tel qu'il est, qu'il atteigne ces objectifs ou non.

**Faisabilité (estimation).** Tout tourne sur la RTX 3050 (4 Go) :

- environ 2,3 Go de mémoire à l'entraînement ;
- environ 7 h 30 par run (mesuré) ;
- environ 0,3 s de réflexion par pièce en jeu.

Ces valeurs supposent K = 8. La mesure de D tranche ; à K = 16, tout ce qui passe par le graphe double (section 14.3).

### Ce qui change par rapport au projet Uno

- **Le moteur.** Un moteur Tetris maison, simple et entièrement maîtrisé, remplace RLCard.
- **La sortie.** Une note par grille candidate, au lieu de 61 scores d'action.
- **La mémoire.** Aucune mémoire entre les pièces : la grille contient toute l'information utile.
- **La mesure.** Des lignes complétées au lieu d'un pourcentage de victoires, avec une marge énorme entre le hasard, les modèles simples et l'expert.
- **Le coût.** Une décision par pièce, et chaque décision évalue jusqu'à 34 grilles.

---

## 2. Les projets existants et la recette retenue

### 2.1 Trois familles d'approches

La publication du MaleCNS v1.0 en septembre 2026 a déclenché une vague de projets. La mouche « joue » à Doom, Mario 64, Beat Saber, Minecraft, Fruit Ninja, aux échecs et à Othello, ou conduit une voiture. Derrière la même image se cachent trois méthodes très différentes.

| Famille | Principe | Exemples | Atouts | Limites |
|---|---|---|---|---|
| A. Connectome figé + lecture | Le réseau n'apprend rien ; on programme ou on entraîne seulement la lecture de son activité | Minecraft (des groupes de neurones choisissent parmi des actions programmées) ; Fruit Ninja (circuit visuo-moteur de 4 386 neurones, lecture descendante ajustée) ; fly-craftax (lecture par neurones descendants entraînée par PPO) ; Fly Chess Lab (lecture de coups programmée, non entraînée) | Simple, peu coûteux | Le connectome n'est qu'un générateur de caractéristiques ; un contrôle « entrée directe » a déjà fait aussi bien dans au moins un de ces projets (référence exacte **à sourcer**, section 21) |
| B. Câblage figé, forces apprises | Topologie imposée par le connectome ; on apprend la force des connexions et quelques paramètres par neurone | flyhard (volant tourné par la patte d'une mouche simulée) ; fly-self-driving (conduite dans une rue) ; BeatTheFly (Othello et Pong sur des sous-circuits, poids entraînés sous le masque du connectome, loi de Dale respectée pour Othello) ; hunting-fly (entraînement limité à ce qu'un vrai cerveau pourrait ajuster) | Résultats les mieux documentés, contrôles publiés | Coûteux : rétropropagation à travers des millions de connexions |
| C. Plasticité « biologique » | Règles locales de type STDP, modulées par un signal dopaminergique de récompense | Doom (les dégâts stimulent deux neurones dopaminergiques PPL101) ; Fly-Driving-Car ; FlyPong | Récit biologique fort | Résultats rarement démontrés ; FlyPong publie des résultats négatifs |

**Tetris.** Aucun projet connu n'a été trouvé. Le précédent le plus proche est la conduite (famille B), autre tâche à entrée visuelle.

**Choix :**

- **famille B :** c'est la recette du projet ;
- **famille A :** elle sert uniquement de contrôle (le « réservoir ») ;
- **famille C :** exclue.

### 2.2 La référence : fly-self-driving (recette flyhard)

| Aspect | Ce que fait la référence |
|---|---|
| Graphe | 165 122 neurones tracés, 25 563 197 connexions dirigées ; chaque neurone porte un état de taux signé |
| Ce qui apprend | Un gain par connexion et une fuite par neurone, soit 25 728 319 paramètres ; aucune connexion ajoutée, retirée ou déplacée |
| Entrée (figée) | Chacun des 4 114 neurones sensoriels du lobe optique lit un pixel tiré au hasard d'une image 64 × 32, avec un signe aléatoire |
| Sortie (figée) | Un vecteur aléatoire fixe sur les 708 neurones moteurs de la corde nerveuse ventrale donne l'angle de braquage |
| Temps | 4 mises à jour du graphe par décision de 50 ms ; état conservé d'une décision à l'autre. Ce seul changement fait passer la première tâche de 3 à 17 routes réussies sur 20 |
| Entraînement | Imitation pure d'un expert, avec du bruit injecté dans les démonstrations pour enregistrer des récupérations ; événements rares surpondérés (dépassements ×3) ; rétropropagation dans le temps tronquée sur 16 décisions, lots de 4, Adam, taux 0,04, 1 200 mises à jour ; puis 3 tours de DAgger de 500 mises à jour |
| Coût | ~35 min par run sur H100, ~100 min sur Mac mini M4 ; projet bouclé en ~3 jours pour moins de 25 $ de GPU loué |
| Contrôles | Modèle linéaire (1 153 paramètres) ; MLP à 40 neurones cachés ; « mouche recâblée », entraînée avec la même recette (indices présynaptiques permutés sur toutes les connexions : chaque neurone garde son degré entrant et le nombre total de connexions est inchangé, mais qui-parle-à-qui est détruit). Une seule graine par condition, d'où des écarts rapportés plutôt que des verdicts |
| Résultats (20 rues jamais vues, avec trafic) | Expert 20 ; linéaire 12 ; MLP 18 ; mouche par imitation seule 12 ; mouche + DAgger 20, 19 et 19 sur trois jeux indépendants ; mouche recâblée 16 |

### 2.3 Ce que ce projet reprend et ce qu'il adapte

**Repris tel quel :**

- le graphe et le principe des paramètres appris ;
- les interfaces aléatoires figées, et les mêmes populations d'entrée (4 114 neurones) et de sortie (708 neurones) ;
- l'imitation suivie de 3 tours de DAgger ;
- le contrôle recâblé ;
- l'enregistrement de l'activité pour la démo.

**Adapté à Tetris :**

- **la décision :** la mouche note chaque grille résultante (formulation « juge de grilles »), au lieu de produire une commande continue ;
- **l'état :** chaque grille est évaluée à partir d'un état remis à zéro, pendant K mises à jour ;
- **la perte :** une softmax sur les notes des candidats, qui imite le choix de l'expert ;
- **le budget de mises à jour :** un pas d'entraînement porte ici 4 termes de perte, contre 64 dans la référence, qui rétropropageait à travers 16 décisions. À nombre de pas égal, le signal de gradient serait 16 fois plus faible ; le clonage est donc porté à 2 400 mises à jour (section 10.3) ;
- **la mesure :** des lignes par partie, sur des séquences de pièces identiques pour toutes les conditions.

---

## 3. Honnêteté scientifique

- **Aucune correspondance biologique.** Tetris est une tâche visuelle, mais aucune fonction connue du cerveau de la mouche n'y correspond. La grille est envoyée à des neurones visuels par convention ; ce n'est pas un modèle d'œil.
- **Le programme fait une partie du travail.** Il énumère les placements possibles et calcule la grille résultante de chacun. La mouche ne fait que juger des grilles. C'est aussi ce que font la plupart des IA de Tetris, et la démo le dit clairement. Règle absolue : les candidats ne sont jamais pré-filtrés par l'expert, sinon c'est lui qui jouerait.
- **La manette est une image, et la démo le dit.** La mouche choisit une position finale, elle n'appuie sur aucun bouton. La suite de touches affichée est reconstituée après coup par le moteur (13.2). C'est le seul endroit du projet où l'illustration va au-delà de ce que fait le modèle, et il porte sa mention.
- **Ce que mesure le projet.** À recette identique, le câblage mesuré note-t-il mieux les grilles qu'un câblage recâblé ? Grâce à la grande marge (de 0,1 à 195 lignes), la réponse est mesurable.
- **Les modèles simples placent la barre haut.** Un juge de grilles linéaire, soit 206 paramètres entraînés en quelques secondes, est attendu autour de 13,8 lignes ; un MLP de 128 neurones cachés autour de 62,6. Si la mouche et ses 25,7 millions de paramètres font moins bien, ce résultat est publié tel quel, avec son analyse.
- **Un choix de conception n'est pas vérifié ici.** La formulation « juge de grilles » est retenue parce qu'elle est donnée pour environ trente fois meilleure que le choix direct (§12). Les élèves à choix direct ne sont pas entraînés dans ce projet : cet écart reste donc une hypothèse reprise de la littérature, pas une mesure du dépôt, et c'est écrit comme tel dans les résultats.
- **Le budget de chaque condition est publié.** Nombre de situations vues, nombre de mises à jour, nombre de paramètres. Un écart de score entre deux modèles entraînés sur des budgets différents n'est pas un écart de capacité, et le tableau de résultats doit permettre de le voir.
- **Vocabulaire.**
  - On dit : « un réseau contraint par le câblage mesuré du système nerveux central d'une drosophile mâle, dont seules les forces des connexions ont été ajustées, complète X lignes par partie en moyenne ».
  - On ne dit pas : « une vraie mouche », « un cerveau téléversé », « la mouche comprend Tetris », « la mouche est consciente ».
- **Traçabilité.** Un registre indique ce qui est mesuré, inféré, figé ou appris (section 8.4), comme le demande la charte de flyhard.

---

## 4. Outils, données et matériel

### 4.1 Logiciels

| Rôle | Outil retenu |
|---|---|
| Langage | Python, dans la version exigée par flyhard |
| Calcul | PyTorch avec CUDA, dans la version exigée par flyhard |
| Base du cerveau | flyhard (préparation du graphe et modèle) + correctif CUDA de fly-self-driving (rétropropagation avec graphe transposé en cache, environ 2,5 fois plus rapide) ; licence MIT, créditée dans le dépôt. Les deux dépôts sont épinglés à un commit précis et recopiés dans le dépôt dès le départ |
| Traitement des données | pyarrow, pandas, NumPy, SciPy |
| Jeu | Moteur Tetris écrit pour le projet, en Python pur, sans dépendance de jeu |
| Suivi des expériences | MLflow |
| Configurations | Un fichier YAML par expérience |
| Tests | pytest |
| Intégration continue | GitHub Actions |
| Serveur | FastAPI + WebSocket, servi par Uvicorn |
| Démo web | Three.js, site statique |
| Déploiement | Docker Compose, 3 services (api, web, mlflow) ; NVIDIA Container Toolkit pour donner le GPU au service api |

### 4.2 Données (MaleCNS v1.0, licence CC-BY 4.0)

| Fichier | Taille | Contenu | Usage |
|---|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 Mo | Annotations des neurones (classes, types, côtés…), **sans** les neurotransmetteurs | Neurones tracés, populations d'entrée et de sortie, positions |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 Mo | Prédictions agrégées de neurotransmetteur par neurone | Signes des connexions |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,1 Go | Forces de connexion entre **tous** les segments, fragments compris (~152 millions de lignes) | Graphe |

- **Source :** https://male-cns.janelia.org/download/ — liens directs, sans authentification :
  `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<nom du fichier>`
- **Sélection des neurones — vérifiée sur les données.** Le critère « tracé » est la colonne **`status`**, et non `statusLabel`. Les deux existent et ne disent pas la même chose : `statusLabel` est fin (« Roughly traced », « Reviewed », « Prelim Roughly traced »…) et aucune combinaison de ses valeurs ne redonne les chiffres de la référence. `status == "Traced"` les donne exactement, tous les trois :

  | | Attendu | Mesuré |
  |---|---|---|
  | Neurones tracés | 165 122 | **165 122** |
  | `superclass == "ol_sensory"` ∩ tracés | 4 114 | **4 114** |
  | `superclass == "vnc_motor"` ∩ tracés | 708 | **708** |

  Les deux populations sont disjointes, comme l'exige §9.3, et 140 024 neurones tracés portent une position de soma — largement de quoi en tirer les 20 000 de la visualisation (§7.5).
- **Quand.** Le téléchargement se lance en fond dès le départ — 1,2 Go, quelques minutes — et la préparation tourne ensuite toute seule. Ni le moteur ni les repères de la section 12 n'en ont besoin : c'est ce qui permet de commencer à coder sans attendre. Les données servent à partir de l'étape D (section 17).
- **Ne pas télécharger le reste.** Les quatre jeux écartés ci-dessous pèsent environ 22 Go et n'entrent dans aucun calcul du projet. L'erreur est facile à faire, la page de téléchargement les proposant côte à côte.
- **Non téléchargés :** les points synaptiques (12,7 Go), les paires de partenaires (6,8 Go), les neurotransmetteurs par synapse (2,7 Go) et les volumes d'imagerie.
- **Lecture du gros fichier.** Les 152 millions de lignes sont lues par morceaux avec pyarrow et filtrées sur les neurones tracés avant toute matérialisation en mémoire. C'est le seul moment du projet où les 24 Go de RAM sont en jeu.
- **Précision sur les chiffres.** Les « 125 millions de synapses » annoncées dans la presse sont bien incluses : ce sont les poids des 25,6 millions de connexions entre neurones tracés, qui totalisent environ 124 millions de contacts.
- **Attribution** dans le dépôt et dans la démo : FlyEM (HHMI Janelia), Université de Cambridge, MRC LMB, Google Research.
- **Licences du projet.** Le code est publié sous licence MIT. Les fichiers dérivés du MaleCNS — graphe préparé, positions de soma, enregistrements de parties qui contiennent l'activité des neurones — restent sous CC-BY 4.0 et portent l'attribution ci-dessus, y compris ceux que sert la page publique. Le crédit de flyhard et de fly-self-driving (MIT) figure dans le dépôt et sur la page.

### 4.3 Matériel

- **Machine :** i7 de 11e génération, 24 Go de RAM, RTX 3050 avec 4 Go de VRAM. Tout le projet tourne sur cette machine, sans GPU loué.
- **Disque :** 20 Go réservés. Détail : données brutes 1,2 Go ; graphe préparé quelques centaines de Mo ; démonstrations 65 Mo (10.2) ; points de contrôle 4 × 0,31 Go par run (10.3) ; enregistrements de parties (13.5).
- **Ménage.** Un run terminé ne conserve que son meilleur point de contrôle. Sans cette règle, les 9 runs de la v1.0 occupent à eux seuls environ 11 Go.

---

## 5. Le jeu : règles retenues

### 5.1 Rappel : Tetris moderne

- **Grille :** 10 colonnes × 20 lignes.
- **Pièces :** 7 tétrominos (I, O, T, S, Z, J, L), tirés par « sacs de 7 » : chaque sac contient une pièce de chaque type, dans un ordre aléatoire.
- **Déroulement :** les pièces tombent avec une gravité ; le joueur les déplace, les fait tourner et peut en garder une en réserve. Une ligne pleine disparaît.
- **Fin de partie :** quand la pile atteint le haut de la grille.

### 5.2 Version retenue

| Point | Tetris moderne | Version retenue | Raison |
|---|---|---|---|
| Décision | Commandes en temps réel (gauche, droite, rotation, chute) | Choix de la position finale (rotation + colonne) | C'est la formulation qui apprend (section 12) |
| Chute | Glissements sous un surplomb, rotations spéciales (T-spin) | Chute verticale depuis le haut, sans glissement | Moteur simple ; au plus 34 candidats par pièce |
| Tirage des pièces | Sacs de 7 | Sacs de 7 | Standard du jeu |
| Aperçu | Plusieurs pièces suivantes | 1 pièce suivante, affichée au joueur humain, non fournie à la mouche | L'expert ne s'en sert pas |
| Réserve | Oui | Non | Simplicité |
| Temps | Gravité, délai de verrouillage | Aucun : tour par tour | La réflexion de la mouche ne dépend pas d'une horloge |
| Fin de partie | Pile qui atteint le haut | Pièce impossible à poser dans les 20 lignes, ou plafond de pièces atteint | Parties de durée maîtrisée |
| Score | Points (bonus pour 4 lignes, combos) | Lignes complétées | Mesure directe du talent |

**Un seul moteur.** Le moteur Python est l'unique moteur du projet : il sert à l'entraînement, à l'évaluation, aux enregistrements et au serveur. Le navigateur n'a aucun moteur de jeu : il affiche ce que lui envoient le serveur ou les enregistrements.

---

## 6. Le moteur de jeu

*Les nombres de placements sont des faits de combinatoire, vérifiables sans rien exécuter. Les vitesses et les durées de partie sont des attentes, confirmées par la phase 0.*

- **Placements possibles par pièce (grille vide) :**
  - O : 9 ;
  - I, S et Z : 17 ;
  - T, J et L : 34.

  Il y a donc au plus 34 candidats par pièce, et 23,1 en moyenne sur les sept pièces.
- **Numérotation.** Rotation (0 à 3) × colonne (0 à 9) donne 40 identifiants, dont au plus 34 sont valides. Les rotations identiques à une précédente sont éliminées.
- **Ce que fournit le moteur.** Pour chaque pièce, la liste des placements valides et, pour chacun, la grille résultante et le nombre de lignes complétées.
- **Graines.** Une graine fixe toute la séquence de pièces : toutes les conditions jouent exactement les mêmes séquences (évaluation appariée). Le départage des égalités tire sur un générateur distinct, sans quoi deux joueurs différents ne verraient pas la même séquence.
- **Vitesse (mesurée).** L'énumération des candidats coûte 0,055 ms par pièce, et l'expert décide en 0,33 ms : 100 parties de 500 pièces prennent 16 s. Deux choix y suffisent — les sommets des colonnes sont calculés une fois par pièce et non pour chacun des 34 candidats, et seules les lignes que la pièce vient de toucher sont examinées pour la disparition.
- **Durée des parties :**
  - le joueur au hasard perd après 25 pièces (mesuré) ;
  - l'expert atteint le plafond de 500 pièces dans 100 % des parties ;
  - avec un plafond de 10 000 pièces, l'expert fait en moyenne environ 3 000 lignes et atteint le plafond dans 30 % des parties (sur 10 parties).
- **Interface.** Trois opérations : lire l'état, obtenir les candidats, jouer un candidat. La mouche est une fonction qui renvoie une note par grille candidate.

---

## 7. Le connectome : données, signes, populations

### 7.1 Préparation du graphe

- **Méthode :** les scripts d'acquisition et de préparation de flyhard. Ils téléchargent les données, gardent les neurones au statut « tracé », construisent le graphe et le vérifient par empreinte SHA-256.
- **Résultat attendu :** 165 122 neurones et 25 563 197 connexions, exactement comme la référence.
- **Neurones exclus :** le jeu de données complet compte environ 166 700 neurones. Les quelque 1 600 neurones qui n'ont pas le statut « tracé » sont exclus, car leur reconstruction est jugée moins complète.
- **Vérifications faites à la préparation :**
  - l'empreinte correspond à celle publiée par flyhard ;
  - les nombres de neurones et de connexions sont exacts ;
  - aucune connexion ne touche un neurone exclu ;
  - le schéma des trois fichiers est archivé dans le dépôt.

### 7.2 Orientation

Les connexions vont du neurone présynaptique au neurone postsynaptique. L'entrée d'un neurone est la somme sur ses connexions entrantes.

- **Stockage :** le graphe est rangé par neurone postsynaptique, ce qui rend ce calcul efficace, avec une copie transposée pour la rétropropagation.
- **Test :** comme une erreur d'orientation ne produit aucun message d'erreur, elle est testée sur un graphe jouet (section 16).

### 7.3 Signes (loi de Dale)

Chaque neurone présynaptique a un seul signe. Les signes sont recalculés à partir du fichier de neurotransmetteurs, avec la convention ci-dessous, puis appliqués au graphe de flyhard.

| Neurotransmetteur prédit | Signe |
|---|---|
| Acétylcholine | Excitateur |
| GABA | Inhibiteur |
| Glutamate | Inhibiteur |
| Histamine | Inhibitrice |
| Dopamine, sérotonine, octopamine, tyramine | Excitateurs |
| Inconnu | Excitateur |

- **GABA et glutamate :** convention de Shiu et al. (Nature, 2024), qui ne rendent inhibiteurs que ces deux neurotransmetteurs.
- **Histamine :** ajoutée comme inhibitrice, car c'est le neurotransmetteur des photorécepteurs, qui agit par des canaux chlorure.
- **Colonne retenue :** `consensus_nt` du fichier de neurotransmetteurs, la prédiction agrégée par neurone.
- **Tous les autres cas :** excitateurs, comme dans les modèles du cerveau entier. La valeur `unclear` du fichier entre dans ce cas.

**Ce que la convention donne réellement, mesuré sur les données.**

| Population | Composition | Part inhibitrice |
|---|---|---|
| Les 165 122 neurones tracés | ACh 62,8 % · glutamate 17,7 % · GABA 13,4 % · histamine 3,6 % · inconnu 2,2 % | **34,7 %** |
| Les 4 114 neurones d'entrée | histamine 100 % | **100 %** |
| Les 708 neurones moteurs | inconnu 54,9 % · glutamate 42,7 % · ACh 2,0 % · GABA 0,3 % | **42,9 %** |

**Les neurones d'entrée sont tous inhibiteurs.** Ce sont des photorécepteurs, et la convention ci-dessus les rend inhibiteurs sans exception : **toute** l'information sur la grille entre dans le graphe par de l'inhibition. Ce n'est pas une erreur, et cela ne perd pas d'information — la projection d'entrée (§9.2) donne à chaque neurone sensoriel un signe aléatoire ±1, de sorte que la moitié d'entre eux s'activent sur une case occupée et l'autre moitié sur une case vide. Le signal reste donc bidirectionnel en aval.

Mais cela donne à ce tirage aléatoire un rôle que le document ne lui prêtait pas : **c'est lui seul qui empêche l'entrée d'être un biais systématique**. À retenir si le témoin « entrée aveuglée » se comportait un jour de façon inattendue.

**Plus de la moitié des signes de sortie sont inférés par défaut.** 54,9 % des neurones moteurs ont un neurotransmetteur `unclear` et deviennent excitateurs par convention, non par mesure. Le registre de §8.4 doit le dire : pour cette population, le signe est majoritairement un choix de ce document.

### 7.4 Poids de base, normalisation et facteur global

- **Poids de base :** signe × nombre de synapses, divisé par le nombre total de synapses reçues par le neurone postsynaptique. À l'initialisation, la somme des **valeurs absolues** des poids entrants de chaque neurone vaut donc 1.
- **Neurone sans connexion entrante :** son total de synapses reçues vaut 0. Aucune division n'est faite et son entrée synaptique reste nulle. Leur nombre est compté et rapporté avec les mesures du graphe (7.6).
- **Ordre.** Cette procédure a besoin de grilles candidates : elle vient donc **après** les démonstrations (étape B de la section 17), qui coûtent moins d'une minute de CPU. Aucune raison de la faire avant.
- **Facteur global — mesuré : 24.** Un multiplicateur commun à toutes les connexions, fixé une fois pour toutes avant le premier entraînement, selon la procédure suivante :
  1. on essaie 8, 12, 16, 24, 32 et 64 ;
  2. pour chaque valeur, on présente 100 grilles candidates tirées des démonstrations pendant K mises à jour ;
  3. on écarte les valeurs qui saturent plus de 20 % des neurones (état supérieur à 0,99 en valeur absolue), même seuil qu'en section 18 ;
  4. parmi celles qui restent, on retient la plus petite pour laquelle **au moins 80 %** des 708 neurones moteurs ont une activité supérieure à 0,01 en valeur absolue.

  **Ce que la mesure donne**, sur le connectome complet, K = 7, 100 grilles :

  | Facteur | Neurones moteurs actifs | Saturés | Participation |
  |---|---|---|---|
  | 8 | **0,0 %** | 0,0 % | 58,0 % |
  | 12 | 10,6 % | 0,1 % | 66,5 % |
  | 16 | 50,8 % | 0,1 % | 76,6 % |
  | **24** | **92,4 %** | 0,2 % | 93,4 % |
  | 32 | 97,7 % | 0,2 % | 94,7 % |
  | 64 | 99,6 % | 0,2 % | 93,8 % |

  **La grille d'origine {1, 2, 4, 8} ne pouvait pas fonctionner.** À 8 — la valeur que le document retenait par défaut — **aucun** neurone moteur n'est actif, et la note de sortie vaut 4 × 10⁻⁷. La mouche aurait été incapable d'apprendre quoi que ce soit, et l'échec se serait présenté comme la parade « Pas d'apprentissage » (§18) sans que la cause réelle soit visible. Le repli prévu prolongeait la grille **vers le bas**, c'est-à-dire dans la mauvaise direction.

  **Pourquoi le signal s'éteint.** La normalisation fait que les poids entrants d'un neurone somment à 1 en valeur absolue. Mais les états de ses présynaptiques sont à peu près indépendants, et leur somme pondérée vaut donc environ 1/√degré. Le degré entrant médian étant de 100, cela fait une division par 10 à chaque saut, soit un facteur 1 000 sur les D = 3 sauts qui séparent l'entrée de la sortie. Un facteur global d'environ 10 est nécessaire rien que pour compenser cette dilution — d'où 24, et non 8.

  **Pourquoi 80 % et non « la moitié ».** Le seuil d'origine tombe exactement sur une falaise : 16 donne 50,8 % de neurones moteurs actifs, c'est-à-dire que le critère se joue à 0,8 point près et qu'un autre échantillon de grilles pourrait le faire basculer. Comme la saturation ne dépasse jamais 0,2 %, même à 64, elle n'est pas le danger que « la plus petite valeur » était censée écarter. On prend donc la plus petite valeur à marge confortable.
- **Vérifications à l'initialisation :**
  - l'activité reste bornée ;
  - deux grilles différentes produisent des activités différentes.
- **Portée.** Le facteur global ne multiplie que l'entrée synaptique, pas l'entrée sensorielle, qui vaut ±1 (8.1). Il règle donc le poids du graphe **par rapport à** l'entrée, et non le niveau général d'activité.

### 7.5 Populations

| Population | Rôle | Taille |
|---|---|---|
| Neurones sensoriels du lobe optique | Entrée : la grille | 4 114 |
| Neurones moteurs de la corde nerveuse ventrale | Sortie : la note | 708 |
| Sous-ensemble de neurones positionnés (soma), tiré une fois avec la graine du run | Visualisation | 20 000 |

### 7.6 Mesures à faire sur le graphe

- **Descriptif — mesuré :** 165 122 neurones, 25 563 197 connexions, **38,3 % de connexions inhibitrices**, **659 neurones sans connexion entrante** (7.4). Le graphe préparé pèse 587 Mo.
- **Composition des interfaces.** Part de neurones inhibiteurs parmi les 4 114 neurones d'entrée et parmi les 708 neurones moteurs. Ce chiffre n'est pas neutre : si les neurones d'entrée retenus sont des photorécepteurs, la convention de 7.3 les rend **tous** inhibiteurs et la totalité du signal entre dans le graphe par de l'inhibition. Cela ne change aucun réglage, mais c'est nécessaire pour lire les résultats et pour interpréter le facteur global.
- **Positions.** Nombre de neurones tracés qui portent une position de soma. S'il est inférieur à 20 000, le sous-ensemble de visualisation (7.5) est réduit d'autant et le chiffre réel remplace 20 000 partout.
- **Distance D — mesurée : D = 3.** C'est le plus petit nombre de sauts qui permet d'atteindre au moins 90 % des 708 neurones moteurs en partant des 4 114 neurones d'entrée (parcours en largeur sur le graphe orienté). **100 % des neurones moteurs sont atteignables**, et le cas de repli ci-dessous ne se présente donc pas. D fixe le temps de réflexion K (section 9.4).
- **Si 90 % ne sont pas atteignables :** on rapporte la part réellement atteignable ; D devient la profondeur qui atteint 90 % des neurones moteurs *atteignables* ; les neurones moteurs hors d'atteinte restent dans le vecteur de sortie, leur état reste nul et leur coefficient ne sert à rien. Le cas est signalé dans les résultats, car il réduit d'autant la dimension utile de la lecture.
- **Participation.** C'est la part des neurones dont l'écart-type d'activité, mesuré sur 1 000 grilles candidates tirées des démonstrations, dépasse 0,01. C'est le chiffre honnête derrière « tous les neurones travaillent » ; il est mesuré avant et après l'entraînement.

**Ce que D décidait :** s'il avait dépassé 8, donc K = 16, tous les temps de la section 14.3 auraient doublé et la parade « run de plus de 6 heures » se serait déclenchée. **Mesuré à 3**, il donne K = 7 : le budget de calcul tient, et même avec un peu de marge.

---

## 8. Le modèle neuronal

### 8.1 Dynamique

- **État.** Chaque neurone porte un état compris entre −1 et 1. Il est nul au départ de chaque évaluation.
- **Mise à jour.** À chaque mise à jour, le nouvel état d'un neurone vaut :
  - (1 − fuite) × son ancien état,
  - plus fuite × tangente hyperbolique de la somme de ses entrées.
- **Entrées.** Deux sortes d'entrées s'additionnent :
  - **synaptiques :** pour chaque connexion entrante, l'état du neurone présynaptique × son signe × le poids de base normalisé × le gain appris × le facteur global ;
  - **sensorielle :** uniquement pour les 4 114 neurones d'entrée (section 9.2).
- **Code.** Le code de flyhard sert de base ; il est adapté pour respecter exactement cette spécification.
- **Pas de simulation à impulsions (LIF).** Un modèle à taux est dérivable, donc entraînable par rétropropagation. Il demande quelques mises à jour par évaluation au lieu de milliers de pas de 0,1 ms, et c'est ce qui a fonctionné dans la référence.

### 8.2 Paramètres appris

| Paramètre | Nombre | Contrainte | Valeur initiale |
|---|---|---|---|
| Gain par connexion | 25 563 197 | Positif (fonction softplus), pour ne jamais inverser un signe | 1 (paramètre brut ≈ 0,5413) |
| Fuite par neurone | 165 122 | Entre 0 et 1 (fonction sigmoïde) | 0,5 (paramètre brut 0) |
| Température des notes | 1 | Positive (fonction softplus) | 1 (paramètre brut ≈ 0,5413) |

Total : 25 563 197 + 165 122 + 1 = **25 728 320**, soit environ 25,7 millions de paramètres. La référence en annonce 25 728 319 : la différence est la température, qu'elle n'a pas.

Les trois contraintes sont obtenues par reparamétrage, jamais par troncature après coup : aucun gain ne peut devenir négatif, donc aucun signe ne peut s'inverser, quelle que soit la taille du pas d'apprentissage.

**La température ne change pas le jeu.** Multiplier toutes les notes d'une même pièce par un nombre positif ne change pas leur classement. La température n'agit donc que sur la perte pendant l'entraînement et sur les probabilités affichées dans la démo (13.2) ; en jeu, l'argmax est identique avec ou sans elle.

### 8.3 Éléments figés

- la topologie ;
- les signes ;
- les nombres de synapses et la normalisation ;
- le facteur global ;
- la projection d'entrée ;
- le vecteur de sortie.

**Figés pendant tout un run.** La projection d'entrée et le vecteur de sortie sont tirés une fois, au début d'un run, à partir de la graine de ce run, puis ne bougent plus. Ils ne sont jamais entraînés. La v0.1 utilise la graine 0 ; les runs 1 et 2 de la v1.0 les retirent avec leur propre graine, de sorte que les intervalles à trois graines couvrent aussi le tirage des interfaces, qui est le seul vrai tirage aléatoire de la recette (section 10.5).

### 8.4 Registre mesuré / inféré / figé / appris (tenu dans le dépôt)

| Élément | Statut | Source |
|---|---|---|
| Qui est connecté à qui | Mesuré | MaleCNS |
| Nombre de synapses par connexion | Mesuré | MaleCNS |
| Signe de chaque neurone | Inféré | Prédiction de neurotransmetteur + convention (7.3). Pour 2,2 % des neurones tracés — et **54,9 % des neurones moteurs** — la prédiction est `unclear` : le signe vient alors de la convention seule, pas d'une mesure |
| Gains, fuites, température | Appris | Entraînement |
| Normalisation, facteur global, tangente hyperbolique, K | Choisi | Ce document |
| Projection d'entrée et vecteur de sortie | Figé, tiré au hasard une fois par run | Graine du run |

---

## 9. Entrée, sortie et protocole temporel

### 9.1 Encodage d'une grille candidate : 205 entrées

| Bloc | Contenu | Codage | Taille |
|---|---|---|---|
| Grille résultante | Les 200 cases après la pose et après disparition des lignes pleines | Occupée +1, vide −1 | 200 |
| Lignes complétées par ce placement | 0 à 4 | Une case à +1, les autres à −1 | 5 |

Rien d'autre n'est fourni : ni la pièce suivante, ni le marquage de la pièce posée. C'est exactement l'encodage mesuré avec les modèles simples (section 12). Les 5 dernières entrées sont indispensables : une fois les lignes disparues, la grille seule ne dit plus combien il y en avait.

### 9.2 Projection d'entrée (figée)

- **Affectation.** Chacun des 4 114 neurones sensoriels du lobe optique est affecté à une seule des 205 entrées. L'affectation est aléatoire mais équilibrée : chaque entrée est lue par 20 ou 21 neurones. Exactement : 4 114 = 205 × 20 + 14, donc 14 entrées tirées au hasard sont lues par 21 neurones et les 191 autres par 20.
- **Signe.** Chaque neurone reçoit un signe aléatoire, + ou − avec une chance sur deux.
- **Intensité.** L'entrée sensorielle d'un neurone vaut son signe × la valeur de l'entrée qu'il lit (+1 ou −1).
- **Tirage.** Il est fait une fois par run, avec la graine du run, et enregistré à côté du point de contrôle.

C'est le principe des pixels de la référence : la mouche « regarde » la grille.

### 9.3 Sortie (figée)

- **La note** est la somme, sur les 708 neurones moteurs, de leur état multiplié par un coefficient fixe, le tout multiplié par la température apprise.
- **Les coefficients** sont tirés une fois avec la graine du run, selon une loi normale de moyenne 0 et d'écart-type 1/√708.
- **En jeu,** la pièce est posée sur le candidat le mieux noté. Les égalités sont départagées au hasard, avec un générateur distinct de celui des pièces : la séquence de pièces d'une graine reste donc rigoureusement identique d'une condition à l'autre, ce dont dépend toute l'évaluation appariée (11.1).
- **En entraînement,** les notes des candidats d'une même pièce passent dans une softmax.

**Règle de conception : aucun élément programmé ne peut transporter le signal seul.**

- La projection d'entrée et le vecteur de sortie sont aléatoires et figés.
- Les populations d'entrée et de sortie sont disjointes.
- Tout le chemin passe par le graphe.
- Les témoins « graphe coupé » et « entrée aveuglée » (section 11) doivent retomber au niveau du hasard, soit environ 0,1 ligne par partie.

### 9.4 Protocole temporel

**Une évaluation par candidat.** Pour chaque pièce et chaque candidat :

1. l'état du réseau est remis à zéro ;
2. la grille est présentée aux neurones d'entrée pendant K mises à jour ;
3. la note est lue après la K-ième mise à jour.

**Règles :**

- **En parallèle.** Les candidats d'une même pièce (34 au plus) sont évalués en même temps, comme un lot.
- **Valeur de K — mesurée et confirmée : K = 7.** K = D + 4, où D = 3 est la distance mesurée sur le graphe (section 7.6). Le plafond de 16 ne joue pas, et les estimations de temps de la section 14.3, faites pour K = 8, sont donc légèrement conservatrices.
- **Le « + 4 » a été vérifié, pas supposé — et il tient.** C'est le réglage le plus arbitraire du modèle, et s'il est trop petit le signal n'a pas le temps de ressortir. Le run de mise au point a donc été passé trois fois, 200 mises à jour chacun, sur le connectome complet :

  | K | Accord au départ | Accord après 200 pas | Gain | Durée |
  |---|---|---|---|---|
  | D + 2 = 5 | 6,4 % | 43,8 % | +37,4 | 24 min |
  | **D + 4 = 7** | 9,6 % | **46,6 %** | +37,0 | 31 min |
  | D + 8 = 11 | 11,2 % | 44,0 % | +32,8 | 44 min |

  **La règle du document est la meilleure des trois.** Et K = 11 est plus mauvais que K = 7, ce qui compte autant : allonger le temps de réflexion ne rend pas la mouche meilleure, il existe un optimum, et il tombe là où le document l'avait placé.

  Une nuance à citer avec ces chiffres : les points de départ diffèrent. Avant tout apprentissage, à gains tous égaux à 1, le réseau est déjà au-dessus du hasard à K élevé — 11,2 % à K = 11 contre 6,4 % à K = 5. Une partie de l'écart entre les trois colonnes ne vient donc pas de l'apprentissage. Mesuré en **gain**, K = 5 et K = 7 sont à égalité et K = 11 décroche seul.
- **Pas de mémoire entre les pièces.** La grille contient toute l'information utile.
- **Conséquence.** Avec l'état remis à zéro, la note d'un candidat est une fonction déterministe de ses 205 entrées, sans état caché : la mouche est un noteur de grilles à 25,7 millions de paramètres. L'évaluation en lot ne doit donc strictement rien changer à la note d'un candidat, ni son rang dans le lot. C'est un test (section 16), et c'est la seule façon de détecter une fuite d'information entre candidats dans le noyau creux.

---

## 10. Entraînement

### 10.1 L'expert (le professeur)

L'expert note chaque grille résultante avec 4 critères, puis joue le placement le mieux noté. Il n'anticipe pas la pièce suivante.

| Critère | Poids |
|---|---|
| Hauteur totale des colonnes | −0,510066 |
| Lignes complétées | +0,760666 |
| Trous (cases vides sous un bloc) | −0,35663 |
| Bosses (différences de hauteur entre colonnes voisines) | −0,184483 |

- **Origine :** ce sont les critères de l'IA de Tetris de Yiyuan Lee.
- **Performance mesurée :** 195 lignes sur un maximum de 200 avec un plafond de 500 pièces ; environ 3 000 lignes en moyenne avec un plafond de 10 000.
- **Information :** l'expert voit exactement la même grille que la mouche. Aucun professeur ne triche avec des informations cachées.

### 10.2 Démonstrations

- **Parties.** 410 parties de l'expert (graines 0 à 409), plafonnées à 300 pièces.
- **Bruit.** 5 % des coups sont joués au hasard, en gardant l'étiquette de l'expert. On enregistre ainsi des situations dégradées dont il faut savoir sortir, comme le bruit de braquage de la référence.
- **Volume attendu.** Environ 100 000 situations. Le calcul : avec 5 % de bruit, une partie de l'expert dure en moyenne 244 pièces (chiffre déduit de la section 12), donc 410 × 244 ≈ 100 000. Chaque situation enregistre toutes les grilles candidates et le choix de l'expert.
- **Volume réellement consommé par la mouche.** Le clonage voit 2 400 × 4 = 9 600 situations et les trois tours de DAgger 3 × 500 × 4 = 6 000, soit 15 600 situations, environ 16 % du jeu. Les modèles simples, eux, en font 10 époques. Cet écart est inhérent au coût d'un pas sur 25,6 millions de connexions ; il est publié à côté des scores (section 3).
- **Situations de test.** 20 parties sans bruit (graines 500 à 519), tenues à l'écart de tout entraînement. Sans bruit, l'expert atteint le plafond : environ 6 000 situations et 139 000 candidats.
- **Stockage.** Grilles en bits, soit environ 650 octets par situation (jusqu'à 34 grilles de 200 bits) et environ 65 Mo au total. Variante si le disque manque : n'enregistrer que la grille d'avant, la pièce et le choix de l'expert, soit une trentaine d'octets, et recalculer les candidats au chargement — le moteur les produit en 0,5 ms.

### 10.3 Clonage de comportement

| Réglage | Valeur |
|---|---|
| Perte | Softmax sur les notes des candidats ; la cible est le choix de l'expert |
| Candidats par situation | 10 : celui de l'expert + 9 autres tirés au hasard sans remise (tous s'il y en a moins) |
| Lot | 4 situations |
| Optimiseur | Adam, taux 0,04 |
| Nombre de mises à jour | 2 400 |
| Rétropropagation | À travers les K mises à jour de chaque évaluation, jamais d'une pièce à l'autre |
| Points de contrôle | Toutes les 200 mises à jour et à la fin de la phase ; on garde les 3 derniers et le meilleur. Un point de contrôle contient les paramètres, l'état d'Adam et les interfaces tirées, soit environ 0,31 Go |
| Choix du meilleur | Accord avec l'expert sur un sous-échantillon fixe de 500 situations de test, tiré une fois avec la graine 0 |

**Pourquoi 2 400 et non 1 200.** Un pas d'entraînement porte ici 4 termes de perte, un par situation du lot. Dans la référence, un pas portait 64 décisions, car elle rétropropageait à travers 16 décisions consécutives. À nombre de pas égal, le signal de gradient serait donc environ 16 fois plus faible. Doubler le nombre de pas est le compromis retenu entre ce constat et le budget de calcul : au-delà, le run dépasse les 6 heures de la parade 14.5.

**Pourquoi un sous-échantillon pour choisir le meilleur point de contrôle.** Les 20 parties de test contiennent environ 6 000 situations et 139 000 candidats : les noter toutes coûte une demi-heure de GPU, donc 6 heures pour les 12 points de contrôle d'un clonage, ce qui doublerait le budget de la section 14.3. Les 500 situations du sous-échantillon coûtent environ 3 minutes. L'accord sur les 6 000 situations n'est mesuré qu'une fois par condition, pour le tableau final (11.1).

**Règle de divergence.** Si la perte devient invalide (NaN) ou augmente de plus de 50 % sur 100 mises à jour, l'entraînement repart du dernier point de contrôle avec un taux divisé par 4. **Trois relances au maximum** : au-delà, le run est déclaré échoué, arrêté, et publié comme tel avec ses courbes. Sans ce plafond, un run instable tourne en rond des heures et occupe la carte pour rien.

**Suivi dans MLflow :**

- la perte ;
- l'accord avec l'expert sur le sous-échantillon de test ;
- la norme des gradients ;
- l'activité moyenne ;
- la part de neurones saturés (état supérieur à 0,99 en valeur absolue).

**Deux corrections à l'essai.** Le premier run en recette du document plafonne entre 52 et 55 % d'accord dès le pas 800, alors que la perte oscille sans tendance depuis le pas 400. Deux défauts de la recette, sans coût de calcul, sont mis à l'essai :

- **La perte vise un placement tiré au sort, l'accord en accepte plusieurs.** Mesuré sur les 101 017 situations : l'expert a plusieurs optimaux à égalité dans **16,4 %** d'entre elles, et dans **10,2 %** des situations d'un lot, un autre optimal tiré parmi les concurrents est puni comme une erreur — un placement que l'accord compte juste. Correction : la perte devient − log de la probabilité totale donnée aux optimaux. Là où l'expert n'a qu'un optimal, elle est identique à l'entropie croisée.
- **Le taux ne décroît jamais.** 0,04 constant, avec des lots de 4 situations : le modèle tourne autour d'un optimum sans s'y poser. Correction : décroissance en cosinus de 0,04 à 0,004 sur la phase. Le taux ne dépend que du numéro de pas, donc une reprise de séance le retrouve exactement.

**Protocole.** À budget égal — 800 pas, même graine — contre le meilleur point des 800 premiers pas du run en recette du document, extrait avant que les pas suivants ne le remplacent. Les deux modèles sont mesurés sur les 6 000 situations de test, situation par situation, et l'intervalle de l'écart est tiré **par parties entières** : les situations d'une même partie ne sont pas indépendantes (`mouche comparer-recettes`). L'échantillon de 500 ne peut pas trancher : son bruit est de ± 2 points.

**Règle d'adoption.** Adoptée seulement si l'intervalle exclut zéro, et alors pour **toutes** les conditions — la mouche recâblée en premier, sans quoi la comparaison qui porte la question du projet ne tient plus. Une séance commencée dans une recette la garde jusqu'au bout : le fichier de séance l'enregistre et refuse d'en changer.

### 10.4 DAgger

- **Trois tours.** Graines 2000–2029, puis 2100–2129, puis 2200–2229.
- **Un tour :**
  1. La mouche joue seule 30 parties plafonnées à 300 pièces, en choisissant toujours le candidat le mieux noté.
  2. L'expert étiquette chaque situation réellement atteinte.
  3. On réentraîne pendant 500 mises à jour, avec les mêmes réglages que le clonage.
- **Composition des lots.** La moitié de chaque lot est tirée dans les situations de DAgger accumulées, l'autre moitié dans les démonstrations d'origine. Un tirage uniforme sur l'ensemble donnerait aux situations de DAgger 18 % du poids (21 900 situations contre 100 000), contre 47 % dans les mesures de la section 12, qui n'utilisaient que 100 parties de démonstration : le gain mesuré là-bas ne serait pas transposable ici.
- **Chaque condition produit ses propres données.** La mouche, la mouche recâblée et le juge linéaire jouent chacun leurs 3 × 30 parties. Les graines sont communes, les situations atteintes ne le sont pas — c'est le principe même de DAgger.

**Pourquoi c'est indispensable.** À Tetris, une erreur crée un trou, puis d'autres : la mouche arrive vite dans des grilles abîmées que l'expert ne rencontre jamais. C'est l'équivalent de la voiture qui dérive.

**Gains mesurés avec les modèles simples :**

- MLP : de 51,5 à 62,6 lignes par partie ;
- linéaire : de 6,3 à 13,8 lignes par partie.

### 10.5 Graines

- **v0.1 :** graine 0 pour toutes les conditions.
- **v1.0 :** graines 0, 1 et 2 pour la mouche et la mouche recâblée.

**Trois familles de graines, à ne jamais confondre.** Le mot « graine » désigne trois choses différentes dans ce document ; chaque famille a son propre générateur.

| Famille | Ce qu'elle tire | Valeurs |
|---|---|---|
| Run | Projection d'entrée, vecteur de sortie, permutation du recâblage, sous-ensemble de visualisation, ordre des lots, tirage des 9 autres candidats | v0.1 : 0 · v1.0 : 0, 1, 2 |
| Parties | Séquence de pièces | Démonstrations 0–409 · test 500–519 · évaluation finale 1000–1099 · évaluation de développement 1500–1529 · DAgger 2000–2029, 2100–2129, 2200–2229 · direct public 3000 et au-delà |
| Départage | Choix entre candidats de note égale | Tirée du numéro de partie, indépendante du joueur |

Les plages de la famille « parties » sont disjointes. La graine du run tire **aussi** les interfaces : les trois runs de la v1.0 ne sont donc pas trois répétitions d'un même modèle, mais trois tirages de la même recette. Les intervalles à trois graines couvrent ainsi le seul vrai aléa de la construction, et c'est de lui que dépend la conclusion « mouche contre mouche recâblée ». La mouche et sa recâblée d'une même graine partagent évidemment les mêmes interfaces.

---

## 11. Évaluation et contrôles

### 11.1 Protocole de mesure

| Usage | Parties | Graines | Plafond |
|---|---|---|---|
| Développement | 30 | 1500 à 1529 | 300 pièces |
| Résultats finaux | 100 | 1000 à 1099 | 500 pièces |

Les deux plages sont disjointes : tous les arbitrages du projet (les trois points d'arrêt de la section 17, les parades de la section 18) se prennent sur les parties de développement, et les 100 parties finales ne sont regardées qu'au moment du tableau de résultats.

**Mesures rapportées :**

- les lignes par partie, en moyenne et en médiane, avec un intervalle de confiance à 95 % obtenu par rééchantillonnage (bootstrap, 10 000 tirages) ;
- les pièces posées ;
- la part des parties qui atteignent le plafond, car la moyenne est tronquée par ce plafond : l'expert y est à 100 %, ses 195,2 lignes sont donc un plancher et non une moyenne ;
- les trous créés par pièce ;
- l'accord avec l'expert sur les 6 000 situations de test (même candidat choisi), et le niveau du hasard correspondant, soit environ 4 % (une chance sur 23,1 candidats en moyenne) ;
- le budget d'entraînement : situations vues, mises à jour, paramètres.

Toutes les conditions jouent les mêmes séquences de pièces. Le joueur au hasard et l'expert sont mesurés dans les mêmes conditions, comme repères.

**Bootstrap apparié.** Puisque les conditions jouent les mêmes séquences, l'écart entre deux conditions se mesure partie par partie. Le rééchantillonnage tire des **graines de parties**, et les mêmes graines servent aux deux conditions à chaque tirage ; on publie l'intervalle de la différence moyenne. Le tirage lui-même part d'une graine fixée, comme tout le reste du projet : un intervalle publié doit pouvoir être retrouvé au chiffre près. Une conclusion « écart démontré » exige que cet intervalle exclue 0. Deux intervalles calculés séparément qui ne se recouvrent pas forment un critère à la fois plus sévère et moins informatif : ils restent affichés dans les tableaux, mais ne servent pas de test.

**Ce que cet intervalle mesure, et ce qu'il ne mesure pas.** Il mesure le bruit de partie à partie, **à graine fixée**. Il ne dit rien de la variabilité d'une graine à l'autre, c'est-à-dire du tirage des interfaces et de l'optimisation (10.5). Avec trois graines, cette seconde variabilité est *vue*, pas estimée. Le facteur limitant de toute conclusion est donc le nombre de graines, pas le nombre de parties : passer de 100 à 500 parties resserrerait un intervalle qui n'est déjà pas le bon. Les tableaux affichent les deux — l'intervalle apparié par graine, et l'écart des trois graines — et les conclusions se formulent sur le second. Un écart qui change de signe d'une graine à l'autre n'est pas un effet, quelle que soit la finesse des intervalles.

### 11.2 Contrôles

| Contrôle | Réalisation | Question posée |
|---|---|---|
| Mouche recâblée | Indices présynaptiques permutés sur toutes les connexions (graine du run) ; le signe suit le nouveau neurone présynaptique ; même recette complète que la mouche | Le câblage précis compte-t-il ? |
| Recâblage à signes conservés (v1.0, conditionnel) | Permutation des indices présynaptiques **à l'intérieur de chaque classe de signe** : chaque connexion garde son signe, chaque neurone garde son équilibre excitation / inhibition, seule l'identité de la source change | Est-ce la topologie qui compte, ou l'équilibre local des signes ? |
| Réservoir — **reporté** | Connectome figé (gains à 1, fuites à 0,5) ; seuls les 708 coefficients de sortie et la température s'entraînent | Qu'apporte l'apprentissage des gains ? Question secondaire, remise à après la v0.1 |
| Linéaire | La note est une combinaison linéaire des 205 entrées | Un modèle simple fait-il aussi bien ? |
| MLP 128 — **reporté** | Une couche cachée de 128 neurones (ReLU) | Un modèle simple non linéaire fait-il aussi bien ? Une vingtaine de lignes une fois la boucle d'entraînement écrite |
| Graphe coupé | Gain **effectif** forcé à 0 sur toutes les connexions, c'est-à-dire la sortie de la softplus et non le paramètre brut ; tout le reste inchangé | Y a-t-il un raccourci caché ? (attendu : environ 0,1 ligne) |
| Entrée aveuglée | Toutes les entrées sensorielles à 0 | Même question |

**Précisions sur les contrôles de câblage.**

- **Recâblée.** La permutation conserve, pour chaque neurone postsynaptique, son degré entrant et le multiensemble des nombres de synapses qu'il reçoit : la normalisation de 7.4 est donc inchangée, et les deux mouches ont exactement les mêmes poids de base en valeur absolue. Elle ne conserve pas les degrés sortants, qui deviennent à peu près poissonniens. Elle crée aussi quelques boucles sur soi et quelques doublons (une même paire tirée deux fois) : ils sont **conservés**, pour que le nombre de connexions reste exactement 25 563 197, et leur nombre est rapporté. Enfin, comme le signe suit le nouveau neurone présynaptique, l'équilibre excitation / inhibition de chaque neurone se rapproche de la moyenne du réseau : la mouche recâblée perd donc deux choses en même temps, le câblage précis **et** l'équilibre local des signes.
- **Recâblage à signes conservés.** Il sépare ces deux effets, et il n'est lancé qu'en v1.0, et seulement si la v0.1 montre un écart entre la mouche et la mouche recâblée — sinon la question ne se pose pas. Coût quand il se déclenche : un run et une évaluation finale par graine.
- **Réservoir.** C'est la seule condition où le vecteur de sortie s'entraîne, ce qui est la définition même d'un réservoir et déroge donc volontairement à 8.3. Cela ne crée pas de raccourci : ces 709 paramètres ne lisent que les 708 états produits par le graphe, et les populations d'entrée et de sortie restent disjointes. Le budget de mises à jour est identique à celui de la mouche, pour que la comparaison porte sur ce qui est appris et non sur la durée d'entraînement.

**Entraînement des trois modèles simples :**

- mêmes démonstrations, même perte, 10 candidats par situation ;
- Adam, taux 0,001, lots de 64 situations, 10 époques ;
- puis les mêmes 3 tours de DAgger, avec 3 époques par tour et la même règle de composition des lots (10.4) ;
- ces 10 époques représentent bien plus de situations vues que les 2 400 mises à jour de la mouche : chaque modèle est entraîné jusqu'à son mieux, et le budget de chacun est publié avec son score (section 3).

**Repères en protocole de développement.** Les modèles simples, le joueur au hasard et l'expert sont aussi évalués en protocole de développement (30 parties, graines 1500–1529, plafond 300) en même temps que le tableau des repères. Sans cela, le troisième point d'arrêt comparerait une mouche mesurée sur 30 parties à plafond 300 à un chiffre obtenu sur 100 parties à plafond 500.

### 11.3 Lire les résultats

| Résultat observé | Conclusion |
|---|---|
| Mouche ≈ mouche recâblée (intervalle de la différence contenant 0) | Le câblage précis n'apporte rien à Tetris. C'est un résultat valable et publié comme tel |
| Mouche > mouche recâblée, intervalle de la différence excluant 0 sur les 3 graines | Effet du câblage, publié avec ses intervalles et la mention « à répliquer ». Rappeler que ce contrôle détruit aussi l'équilibre local des signes, et lancer le recâblage à signes conservés (11.2) |
| Mouche recâblée > mouche, intervalle excluant 0 | Le câblage mesuré dessert cette tâche. Résultat publié tel quel, après vérification que les deux conditions ont bien eu le même facteur global, le même K et le même budget |
| Mouche au-dessus du linéaire, sous la cible de 62,6 | La recette apprend sans atteindre la cible. Publié tel quel ; l'écart mouche / recâblée reste la mesure principale. Entraîner alors le MLP pour mesurer la cible au lieu de la citer |
| Mouche < juge linéaire | Problème d'entraînement : la procédure de la section 18 s'applique avant toute conclusion |

Dans tous les cas, les écarts sont rapportés avec leurs intervalles, et les échecs sont publiés.

---

## 12. Points de référence attendus — première expérience du projet

**Deux lignes sur six sont mesurées.** Les bornes du tableau — le hasard et l'expert — sont produites par le moteur du dépôt, 100 parties sur les graines 1000 à 1099, plafond 500. Elles tombent sur les valeurs attendues : 0,1 ligne et 25,0 pièces pour le hasard ; 195,7 lignes de moyenne et le plafond atteint dans 100 % des parties pour l'expert.

**Une seule ligne d'élève sera mesurée : le juge linéaire.** Le MLP et les deux modèles à choix direct ne sont pas entraînés dans ce projet — leurs valeurs restent des repères de la littérature, jamais des mesures du dépôt, et sont citées comme telles partout où elles servent.

Conséquence à assumer : **l'écart de trente entre les deux formulations n'est pas vérifié ici.** C'est l'argument qui justifie la formulation « juge de grilles », donc le cœur de la conception, et il est repris sans être remesuré. Le projet gagne une semaine et perd cette vérification ; les résultats doivent le dire.

**Conditions de mesure :**

- le moteur du projet, une fois écrit et passé par ses tests (section 16) ;
- 100 parties sur les graines 1000 à 1099 ;
- plafond de 500 pièces, soit au plus 200 lignes ;
- mêmes séquences de pièces pour tous les joueurs.

| Joueur | Lignes par partie (moyenne / médiane) | Pièces posées |
|---|---|---|
| **Hasard — mesuré** | **0,1 / 0** | **25,0** |
| Choix direct, linéaire (imitation → + DAgger) | 0,7 → 0,7 | 35 → 36 |
| Choix direct, MLP de 256 (imitation → + DAgger) | 1,5 → 2,2 (médiane 1 → 2) | 40 → 43 |
| **Juge de grilles, linéaire (imitation → + DAgger) — mesuré** | **32,6 → 30,3 (final : 32,7 [30,4 ; 35,1], médiane 32)** | **123,9** |
| Juge de grilles, MLP de 128 (imitation → + DAgger) | 51,5 → 62,6 (médiane 46 → 60) | — |
| **Expert (heuristique à 4 critères) — mesuré** | **195,7 / 197** | **500 (plafond atteint à 100 %)** |

**Données utilisées :**

- **choix direct :** 57 237 démonstrations (200 parties, 5 % de bruit), puis 3 tours de DAgger de 60 parties ;
- **juge de grilles :** 24 380 situations (100 parties plafonnées à 300 pièces, 5 % de bruit), puis 3 tours de DAgger de 30 parties.

Le tout tourne sur CPU, en quelques secondes à une minute par condition : la phase 0 est courte, mais elle n'est pas facultative.

**Ce que la phase 0 décide.** Ces chiffres ne sont pas de la décoration, ils portent quatre choses du document :

| Ce qui en dépend | Si la mesure confirme | Si elle infirme |
|---|---|---|
| Les objectifs de la section 1 (> 13,8 puis > 62,6) | Les seuils sont ceux qui sortent de la phase 0 | Les seuils sont remplacés par les valeurs mesurées, sans discussion |
| Le choix « juge de grilles » plutôt que choix direct, cœur de la conception (2.3, 20) | Le choix est fondé | **Toute la formulation est à rouvrir** avant d'écrire la moindre ligne sur le connectome |
| La marge hasard → expert, qui rend l'effet du câblage mesurable | Le projet a une question mesurable | Si la marge est étroite, le projet change de question |
| Les volumes de la section 10.2 (410 parties ≈ 100 000 situations, dilution de DAgger) | Les volumes tiennent | Ils sont recalculés à partir de la durée de partie réellement observée |

**Leçons attendues, à confirmer :**

- **La marge est énorme.** Du hasard (0,1) à l'expert (195), il y a de la place pour mesurer l'effet du câblage, contrairement au Uno.
- **La formulation « juge de grilles » est décisive.** À taille comparable, elle serait environ 30 fois meilleure que le choix direct, avec deux fois moins de données. C'est l'hypothèse sur laquelle repose la conception, et c'est donc la première à vérifier.
- **DAgger n'a rien apporté au juge linéaire** : 32,6 lignes après clonage, 30,3 après les trois tours, en protocole de développement. Le document attendait un doublement. Trois explications restent ouvertes et non départagées : un modèle à 206 paramètres était peut-être déjà à son plafond après le clonage ; le rééquilibrage moitié-moitié des lots (§10.4) change la donne par rapport aux mesures citées ; ou 4 700 mises à jour par tour sont trop peu. **C'est un résultat, pas une anomalie à corriger**, et il est publié tel quel. Il ne préjuge pas de ce que DAgger fera pour la mouche, dont la capacité est tout autre.

**Ce qu'il faut écrire pour produire ce tableau :**

- le moteur, l'expert, le joueur au hasard et l'évaluation appariée — c'est le bloc 1, et il ne dépend de rien d'autre ;
- le juge de grilles linéaire, avec ses démonstrations et son DAgger sur CPU. Les trois autres lignes du tableau — les deux élèves à choix direct et le MLP — ne sont pas produites : elles restent des repères de la littérature, et le document le dit partout où elles servent.

---

## 13. Interface 3D et démo

### 13.1 La scène

- **Décor.** Une drosophile est assise devant un téléviseur à tube qui affiche la grille, une manette posée entre ses pattes avant : croix directionnelle à gauche, quatre boutons d'action en losange à droite, deux gâchettes sur la tranche, deux boutons centraux. C'est l'image du projet, et c'est elle qui décide du reste de la scène — pas de borne d'arcade, pas de joystick.
- **Pas de marque.** La manette et la console sont des formes reconnaissables, sans logo, sans nom de marque et sans reprise d'une charte graphique existante. Le site est public et attribué au MaleCNS (13.8) ; il ne s'adosse à aucune autre marque.
- **La mouche.** Modèle simple, pattes avant sur la manette, animées par la séquence de touches reconstituée (13.2). Les quatre autres pattes et les ailes sont au repos ; aucune animation ne suggère qu'elle réagit à l'écran en temps réel.
- **Thèmes.** Clair et sombre, selon le réglage de l'appareil.
- **Animations.** Le réglage « réduire les animations » est respecté.
- **Sur mobile.** La grille est en haut de l'écran, les commandes en bas.

### 13.2 Montrer la mouche qui réfléchit

- **Pièces fantômes.** Les 3 meilleurs candidats s'affichent en transparence, chacun avec sa probabilité en pourcentage (softmax des notes).
- **Nuage de neurones.** Les neurones de visualisation (section 7.5) sont affichés à la position de leur soma : 20 000 en local, 4 000 en public (13.3). Chaque point est coloré par son activité : orange si positive, bleu si négative, gris au repos ; chaque neurone est normalisé sur sa propre plage.
- **Halo.** Son intensité suit la probabilité du candidat choisi.

**La manette : une reconstitution, pas une commande.** La mouche choisit une position finale, jamais une suite de touches. Le moteur reconstitue donc après coup, pour l'affichage seul, la suite de touches qui mène à cette position :

1. le nombre de rotations, dans le sens le plus court (au plus deux appuis) ;
2. les appuis gauche ou droite nécessaires pour passer de la colonne d'apparition à la colonne choisie (au plus neuf) ;
3. un appui bas pour la chute.

Soit une douzaine d'appuis au plus. Ils s'allument sur la manette au rythme de 230 ms, et la patte avant du côté du bouton va l'enfoncer : la gauche pour la croix, la droite pour A et B.

**La pièce suit les appuis.** Elle apparaît en haut de la grille, non tournée et centrée ; chaque appui sur A ou B la fait tourner d'un quart de tour sous les yeux du spectateur, chaque ◀ ▶ la décale d'une colonne, et ▼ la pose. Un contour marque dès l'apparition la case d'arrivée — la décision est prise avant le premier appui, et l'image ne le cache pas. Les positions successives sont calculées par le moteur (`touches.trajectoire`) et transmises à la page, qui ne connaît pas la forme des pièces ; un test vérifie que la dernière, laissée tomber, est exactement le placement choisi. Le coup suivant attend la fin des appuis : une séquence longue prend près de deux secondes, et à cadence fixe la rotation n'aurait jamais été visible.

**Ce que la démo écrit noir sur blanc, à côté de la manette :** « La mouche note toutes les façons de poser la pièce — jusqu'à 34 — et choisit la meilleure. La suite de touches est reconstituée après coup, pour l'image. » Sans cette phrase, l'animation raconterait une mouche qui joue en temps réel, ce que le projet ne fait pas et exclut (section 20).

**Coût.** Affichage seul : aucune conséquence sur le moteur, le modèle, l'entraînement ou les mesures. La reconstitution est une fonction du moteur qui prend un candidat et rend une liste de touches ; elle est testée comme le reste (rejouer la séquence reconstituée doit redonner exactement la position choisie).

### 13.3 Les modes

| Mode | Contenu | Version | Où il tourne |
|---|---|---|---|
| **Direct public** | La mouche joue, maintenant, et tout le monde regarde la même partie | v1.0 | Diffusé depuis la machine à RTX 3050 |
| Secours | Dernière partie enregistrée, affichée quand la machine est éteinte ou occupée | v0.1 | Site statique, sans GPU |
| Duel | Humain contre mouche, même séquence de 100 pièces | v1.0 | Localement, sur la machine à RTX 3050 |

**Une seule partie, diffusée à tous.** C'est la décision qui rend le direct public possible. Le serveur ne fait pas tourner une partie par visiteur : il en fait tourner **une**, en continu, partie après partie, et le WebSocket envoie le même état à tous les navigateurs connectés. Le coût GPU est donc constant — 0,3 s par pièce — que la page ait un spectateur ou mille. Un visiteur qui arrive au milieu d'une partie reçoit l'état courant et les dix dernières pièces, puis suit le flux.

**La partie diffusée est une partie du protocole, pas une démonstration à part.**

- **Quel modèle.** Celui des trois graines de la v1.0 qui a le meilleur score de **développement**. Jamais celui qui a le meilleur score final : on ne choisit pas un modèle sur le jeu de parties qui sert à publier ses résultats (11.1). La graine retenue est affichée sur la page.
- **Quelles graines de parties.** Une plage dédiée à partir de 3000, jamais utilisée ailleurs, et chaque partie diffusée est journalisée avec la sienne. N'importe quelle partie vue en public peut donc être rejouée à l'identique — y compris celle où la mouche a fait n'importe quoi.
- **Quel plafond.** 500 pièces, comme le protocole final, pour que les lignes affichées à l'écran se comparent directement au tableau de résultats.

**Ce que coûte un spectateur.** Par pièce : la grille en bits (25 octets), la pièce, les trois candidats fantômes et leurs probabilités (une cinquantaine d'octets), et le nuage de 4 000 neurones (4 000 octets). Soit environ 4 Ko par pièce, une pièce toutes les 0,3 s, donc **14 Ko/s par spectateur**. Vingt spectateurs tiennent dans 2,2 Mbit/s de débit montant, deux cents dans 22 Mbit/s.

**Dégradation, dans l'ordre :**

1. au-delà de 50 spectateurs simultanés, le nuage passe à 1 000 neurones (3,5 Ko/s par spectateur) ;
2. au-delà de 200, ou si le débit montant mesuré ne suit pas, la page bascule tout le monde sur le mode secours ;
3. le nombre de spectateurs et le débit sont mesurés en continu et affichés dans MLflow.

**Le mode secours n'est pas une option.** La machine est un poste de travail : elle est éteinte la nuit une partie de l'année, et elle est occupée par les runs de la v1.0. Sans repli, la page publique est morte la moitié du temps. Le secours rejoue donc la dernière partie enregistrée, avec un bandeau qui ne ment pas : « *hors ligne — dernière partie du 12 octobre, 143 lignes* », ou « *la mouche est à l'entraînement — dernière partie du…* ». Le bandeau du direct dit, lui, « *en direct — pièce 212, 47 lignes* ».

**Règles du duel :**

- **Mêmes règles que la mouche.** Le joueur humain déplace et fait tourner une pièce fantôme, puis valide ; la pièce tombe tout droit, sans gravité ni glissement.
- **Gagnant.** Celui qui a complété le plus de lignes après les 100 pièces.
- **Fin anticipée.** Un joueur bloqué avant la fin garde les lignes déjà complétées.

### 13.4 Architecture

- **Moteur.** Le moteur Python fait autorité.
- **Serveur.** Le serveur FastAPI + WebSocket fait tourner le moteur et la mouche. Une boucle unique joue la partie publique en continu ; les connexions WebSocket ne font que s'y abonner. Le duel, lui, ouvre une partie par joueur, en local seulement.
- **Navigateur.** Il affiche l'état reçu et envoie les coups du joueur humain. Il ne calcule rien : pas de moteur côté navigateur (5.2).
- **Messages.** Chaque message porte un identifiant de partie et un numéro de séquence ; tout message d'une ancienne partie est ignoré.
- **Exposition.** La machine n'est pas ouverte sur internet : le serveur est publié par un tunnel sortant (type Cloudflare Tunnel ou Tailscale Funnel), sans port entrant ni adresse fixe. Le tunnel sert aussi de coupe-circuit : on le ferme et la page retombe sur le mode secours.
- **Le direct public est en lecture seule.** Aucun visiteur ne peut agir sur la partie diffusée, ni lancer un calcul. Le seul mode qui accepte des coups est le duel, et il reste local.
- **Jamais l'entraînement et la diffusion en même temps.** 1 Go d'inférence plus 2,3 Go d'entraînement font 3,3 Go : la carte en a 4, moins ce que prend l'affichage du bureau. Cela passerait peut-être, et un dépassement de mémoire tuerait un run de quatre heures. Les deux usages s'excluent donc par un verrou sur le GPU :
  - un seul des deux le détient à la fois ;
  - **l'entraînement est prioritaire**, c'est lui qui produit les résultats. Lancer un run coupe la diffusion, qui rend le verrou et laisse la page retomber sur le mode secours, bandeau « la mouche est à l'entraînement » ;
  - le serveur reprend la main tout seul à la fin du run. Aucune intervention manuelle, dans un sens comme dans l'autre.
- **Conséquence.** Le direct public ne s'allume qu'une fois la série de la v1.0 terminée (étape K, section 17). Ensuite, le régime normal s'inverse : la mouche joue en public, et la diffusion s'interrompt d'elle-même chaque fois qu'un nouveau run demande la carte.

### 13.5 Enregistrement des parties

**Par pièce, on enregistre :**

- la grille ;
- la pièce ;
- les candidats avec leurs notes et leurs probabilités ;
- le choix de la mouche ;
- l'état final des 20 000 neurones de visualisation pour le candidat choisi, à raison d'un octet par neurone avec une échelle propre à chaque neurone (comme la référence).

**Taille.** Environ 20 Ko par pièce, soit environ 10 Mo pour une partie de 500 pièces.

La séquence de touches n'est pas enregistrée : elle se recalcule à l'affichage à partir du candidat choisi, qui l'est déjà.

**Parties vitrines.** 3 parties de 100 pièces enregistrent en plus toutes les K mises à jour du candidat choisi, soit 100 × K × 20 Ko : 16 Mo par partie à K = 8, 32 Mo à K = 16. Elles servent à animer la réflexion de la mouche.

**À quoi servent encore les enregistrements**, maintenant que la page publique diffuse du direct :

- **au mode secours** (13.3), qui rejoue la dernière partie quand la machine est occupée ou éteinte. Une partie récente est publiée à chaque fois que le direct s'éteint proprement ;
- **aux parties vitrines**, qui montrent les K mises à jour du réseau — impossible en direct, cela multiplierait le débit par K ;
- **à l'analyse**, puisque toute partie du tableau de résultats doit pouvoir être rejouée coup par coup.

**Budget du site public.** Une partie de 500 pièces pèse 10 Mo et les trois vitrines 48 à 96 Mo : c'est trop pour un site statique. Dans les fichiers publiés, le nuage est donc ramené à 4 000 neurones, tirés dans les 20 000 avec la graine du run, et les fichiers sont compressés — environ 2 Mo par partie de 500 pièces et 10 à 20 Mo pour les trois vitrines. Les 20 000 neurones restent affichés dans le mode duel, qui tourne en local, et les enregistrements complets restent dans le dossier de travail, hors git.

### 13.6 Accessibilité

- Chaque pièce a une couleur **et** sa lettre (I, O, T, S, Z, J, L), pour les daltoniens.
- Toutes les commandes fonctionnent au clavier et au toucher.
- Tous les boutons ont un libellé.
- Avec « réduire les animations », les pattes de la mouche ne bougent plus : les touches de la séquence reconstituée s'allument sans mouvement, et la mention qui l'accompagne reste affichée.

### 13.7 Latence (estimation)

Environ 0,3 s par pièce sur la RTX 3050 (34 candidats évalués en parallèle, K mises à jour). C'est le rythme du direct public : une pièce toutes les trois dixièmes de seconde, ce qui se regarde bien — on voit la mouche réfléchir.

**Mémoire en diffusion :** environ 1 Go. L'inférence n'a besoin ni de la copie transposée du graphe, ni des gradients, ni des états d'Adam, ni des états gardés pour la rétropropagation. D'où la règle de 13.4 : 1 Go de diffusion plus 2,3 Go d'entraînement ne tiennent pas dans 4 Go, donc jamais les deux en même temps.

### 13.8 Mentions

- L'attribution CC-BY du MaleCNS.
- Un encart honnête reprenant le vocabulaire de la section 3.

---

## 14. Performance attendue sur le matériel

*Machine : i7 de 11e génération, 24 Go de RAM, RTX 3050 avec 4 Go de VRAM. Les valeurs de cette section sont des estimations, remplacées par les mesures de l'étape D (section 17). Elles supposent **K = 8** (donc D = 4) et un lot de 4 situations × 10 candidats = 40 évaluations.*

### 14.1 Mémoire GPU pour l'entraînement

| Mesure sur la RTX 3050 | Valeur |
|---|---|
| VRAM réellement disponible | **3,68 Gio**, et non 4,00 |
| Modèle seul (CSR, poids, gains, interfaces) | 0,58 Gio |
| Jeu, 34 candidats en parallèle | **1,36 Gio** |
| **Entraînement, 40 évaluations, K = 7** | **2,15 Gio** |

L'estimation d'origine annonçait 2,3 Go pour l'entraînement : **elle était juste**. En revanche la carte n'offre que 3,68 Gio et non 4,00 — la marge réelle est de 1,5 Gio, pas de 1,7.

**Un réglage a dû être adapté.** Le correctif de fly-self-driving découpe la rétropropagation en tranches de 8 388 608 arêtes, taille calibrée pour la carte de 48 Go sur laquelle il a été écrit. Sur 4 Go avec 40 évaluations, une telle tranche demande 8,4 M × 40 × 4 octets = **1,34 Gio d'un coup** : c'était elle, et non le graphe, qui faisait dépasser la mémoire. La tranche est donc calculée à partir du lot pour tenir dans un budget de 128 Mo. **Cela ne change rien aux mathématiques** — même gradient, seul le découpage diffère — et rien à la vitesse : mesuré, passer le budget à 384 Mo ne gagne pas un centième de seconde, la rétropropagation étant limitée par la bande passante mémoire et non par le lancement des noyaux.

### 14.2 Le piège à éviter

- **Ne jamais conserver le message de chaque connexion.** Avec 40 évaluations en parallèle, un seul tableau de messages par connexion pèse 25,6 millions × 40 × 4 octets, soit environ 4 Go, pour une seule mise à jour. C'est impossible, même une fois.
- **Le calcul passe par un noyau creux** qui calcule les entrées sans ces messages et accumule le gradient des gains au fil de la rétropropagation.
- **Ce noyau est celui de flyhard, avec le correctif de fly-self-driving.** On n'en écrit pas un nouveau.

### 14.3 Temps de calcul

| Étape | Mesuré sur la RTX 3050 |
|---|---|
| Une pièce en jeu (34 candidats en parallèle) | **0,20 s** |
| Un pas d'entraînement (40 évaluations, K = 7) | **5,2 s** |
| Accord sur le sous-échantillon de 500 situations | ≈ 1 min |
| Clonage (2 400 pas) | ≈ 3 h 30 |
| Points de contrôle du clonage (12) | ≈ 12 min |
| Un tour de DAgger (30 parties + 500 pas) | ≈ 1 h 15 |
| **Un run complet (clonage + 3 tours de DAgger)** | **≈ 7 h 30** |
| Évaluation de développement d'une condition (30 parties, plafond 300) | ≈ 30 min |
| Évaluation finale d'une condition (100 parties, plafond 500) | ≈ 2 h 50 |
| Évaluation finale des témoins coupé et aveuglé | ≈ 5 min les deux |

**Le jeu est conforme à l'estimation, l'entraînement ne l'est pas.** Une pièce coûte 0,20 s au lieu des 0,3 s annoncées, et une évaluation finale 2 h 50 au lieu de 4 h. Mais un pas d'entraînement coûte **5,2 s au lieu de 1 s** : la rétropropagation à travers 25,6 millions d'arêtes est vingt-cinq fois plus chère que le passage avant, alors que le document tablait sur un rapport de trois. Le run complet passe donc de 3 h 30 estimées à 7 h 30 mesurées.

**Conséquence assumée : le seuil de la parade « run trop long » passe de 6 à 10 heures.** Ce seuil servait à garantir qu'un run tienne dans une nuit, à une époque où l'estimation était de 3 heures. Un run de 7 h 30 lancé le soir se relève le matin ; le déclencher à 6 heures reviendrait à ramener le clonage à 1 200 pas pour satisfaire une limite calibrée sur une estimation fausse, alors que le raisonnement de §2.3 sur le signal de gradient tient toujours. C'est un arbitrage explicite, pas un oubli.

### 14.4 RAM et disque

- La préparation des données par les scripts de flyhard est le seul moment critique pour la RAM (4.2).
- Une fois préparé, le graphe fait quelques centaines de Mo.
- Le disque est décrit en 4.3.

### 14.5 Procédure si la machine ne suffit pas

Les mesures s'appliquent dans l'ordre, jusqu'à ce que le problème disparaisse.

**Mémoire GPU insuffisante :**

1. 6 candidats par situation au lieu de 10 ;
2. poids de base en demi-précision ;
3. lots de 2 situations au lieu de 4.

**Run de plus de 10 heures :**

1. clonage ramené à 1 200 mises à jour ;
2. tours de DAgger de 20 parties au lieu de 30 ;
3. série de la v1.0 réduite aux graines 0 et 1.

**Interdit dans tous les cas :** réduire le nombre de candidats en les pré-filtrant avec l'expert.

### 14.6 Découper un run en séances

Un run de 7 h 30 immobilise la machine une nuit entière, et une coupure à la sixième heure perdait tout. Le run se découpe donc en **séances** : `mouche entrainer-mouche --duree 3` travaille trois heures, enregistre son état et rend la carte ; la même commande, relancée plus tard, reprend exactement là.

**Ce qui se découpe est le temps, pas la recette.** 2 400 pas de clonage, trois tours de DAgger, mêmes réglages : le budget est celui de 10.3 et 10.4, quel que soit le nombre de séances. Réduire le budget reste possible (`--pas`), mais donne un autre modèle, publié avec son propre budget à côté des autres (3) — ce n'est pas un découpage.

**La reprise est exacte, et vérifiée bit pour bit.** Un run en plusieurs séances produit les mêmes poids et la même suite de pertes que le même run d'une traite, DAgger compris (`tests/test_reprise.py`, sur CPU). Sur la carte, cette égalité ne peut pas être bit pour bit, et ce n'est pas le fait des séances : les produits creux de cuSPARSE ne sont pas déterministes, et deux runs d'une traite de même graine divergent déjà dans les derniers chiffres — mesuré à la première séance, 46,2 % puis 45,2 % d'accord au pas 200. La reprise restaure l'état exact ; elle n'ajoute aucun écart à celui-là. Il faut pour cela enregistrer, en plus des paramètres :

| Ce qui est enregistré | Ce qui se passerait sans |
|---|---|
| Les moments d'Adam, taux courant compris | Chaque séance repartirait de moyennes nulles, et un taux divisé par la règle de divergence reviendrait à sa valeur d'origine |
| L'état du générateur du run | Chaque séance rejouerait les lots du début du run. Rien ne le signale : le modèle obtenu est simplement moins entraîné qu'annoncé |
| Le meilleur état et son accord | Le point de contrôle retenu (10.3) ne serait choisi que parmi ceux de la dernière séance |
| Les situations collectées par DAgger | La collecte d'un tour, environ une demi-heure, serait refaite à chaque reprise |

Le test vérifie aussi le cas contraire : une reprise qui ne recharge que les paramètres donne une autre trajectoire. Sans ce contre-exemple, le test d'égalité ne prouverait pas que l'état du générateur sert à quelque chose.

**Une séance s'arrête à un point de contrôle, jamais au milieu.** À cet instant l'état de secours de la règle de divergence coïncide avec l'état courant : il n'y a rien de plus à retenir. La granularité est donc de 200 pas, environ 18 minutes, et la séance s'arrête *avant* la limite demandée — elle n'entame pas une fenêtre qu'elle ne finirait pas, ni une collecte DAgger s'il reste moins de 30 minutes. En contrepartie, une séance avance toujours d'au moins une fenêtre, même avec une limite plus courte.

**Le fichier de séance** (`data/points_de_controle/seance-mouche.pt`) pèse environ 400 Mio. Il est réécrit **à chaque point de contrôle**, pas seulement en fin de séance : un plantage ou une coupure de courant ne coûte que la fenêtre en cours, soit 18 minutes au plus, et la commande relancée reprend au dernier point de contrôle. Il est écrit à côté puis renommé : une coupure pendant l'écriture laisse intact l'état précédent. Entre deux séances, la carte est libre — le direct peut tourner, avec le meilleur modèle atteint jusque-là.

**La mise en veille tue une séance, et le fait en silence.** Constaté à la première séance : capot rabattu, portable mis en veille, et au réveil un processus qui attendait indéfiniment un signal que la carte n'enverrait plus — à 100 % d'un cœur, sans erreur, sans une ligne de journal, verrou GPU conservé. Trois parades en découlent :

- la séance bloque la veille **et la fermeture du capot** (`systemd-inhibit --what=sleep:idle:handle-lid-switch`). Bloquer la seule veille ne suffit pas : sous Ubuntu, le capot passe outre par défaut (`LidSwitchIgnoreInhibited=yes`) ;
- un chien de garde arrête le processus, pile d'appels à l'appui, si un pas dure plus de 15 minutes au lieu de 5 secondes ; la séance suivante reprend au dernier point de contrôle ;
- l'entraînement refuse de démarrer sans CUDA. Après une veille, le pilote NVIDIA perd souvent CUDA jusqu'au rechargement de `nvidia_uvm`, et le code passait alors au processeur sans rien dire — trois heures pour quelques dizaines de pas.

Estimé à partir des temps mesurés de 14.3 :

| Découpage | Séances | Ce que fait chacune, environ |
|---|---|---|
| D'une traite | 1 | Tout, 7 h 30 |
| `--duree 3` | 3 | 1 800 pas de clonage · fin du clonage, DAgger 1, début de DAgger 2 · fin de DAgger 2, DAgger 3, évaluation de développement |
| `--duree 2` | 4 | 1 200 pas · 1 200 pas · DAgger 1, début de DAgger 2 · fin de DAgger 2, DAgger 3, évaluation |

---

## 15. Structure du projet

### 15.1 Disposition

Un seul paquet Python sous `src/`, et à côté ce qui n'est pas du code.

| Chemin | Rôle |
|---|---|
| `src/mouche_tetris/graines.py` | Les trois familles de graines (§10.5) ; **seul endroit du projet autorisé à créer un générateur aléatoire** |
| `src/mouche_tetris/notation.py` | L'interface `Noteur`, commune aux dix conditions comparées, et le budget publié avec chaque score |
| `src/mouche_tetris/encodage.py` | Les 205 entrées (§9.1) ; seule chose que le moteur et les modèles partagent |
| `src/mouche_tetris/tetris/` | Le moteur : pièces, sacs de 7, candidats, grilles résultantes, lignes, fin de partie, reconstitution des touches |
| `src/mouche_tetris/expert/` | Heuristique à 4 critères |
| `src/mouche_tetris/connectome/` | Acquisition, empreintes, signes, graphe, modèle, variantes de contrôle |
| `src/mouche_tetris/modeles/` | Le joueur au hasard et le juge de grilles linéaire. Le MLP et le réservoir y viendront s'ils sont repris |
| `src/mouche_tetris/entrainement/` | Démonstrations, clonage, DAgger, points de contrôle, séances (14.6) |
| `src/mouche_tetris/evaluation/` | Séries appariées, bootstrap, rapports |
| `src/mouche_tetris/enregistrement/` | Captures de parties : grilles, candidats, notes, activité |
| `src/mouche_tetris/serveur/` | Diffusion du direct public, duel, verrou GPU |
| `src/mouche_tetris/cli.py` | Une commande par étape du plan (§17) |
| `configs/` | Un fichier YAML par expérience |
| `data/` | Hors git : brut, préparé, démonstrations, points de contrôle, enregistrements. Seules l'arborescence et `empreintes.json` sont versionnées |
| `web/` | Scène 3D, direct public, mode secours, duel |
| `tests/` | Tests unitaires et de fumée, dont le graphe jouet |
| `docs/` | Ce document ; résultats avec contrôles et échecs ; registre de la section 8.4 |

### 15.2 Trois règles de dépendance

Elles ne sont pas cosmétiques : chacune empêche une classe de bug ou protège une contrainte du document.

1. **`tetris` et `expert` n'importent ni PyTorch, ni le connectome, ni CUDA.** C'est ce qui permet à la phase 0 de tourner sur n'importe quelle machine, et à l'intégration continue de tourner sans télécharger le MaleCNS (§16). La frontière est `encodage` : le moteur produit 205 réels, les modèles les consomment, et ni l'un ni l'autre ne connaît la représentation de son voisin.
2. **Toutes les conditions implémentent `Noteur`.** On leur donne les candidats d'une pièce, elles rendent une note par candidat. L'évaluation, l'enregistrement et le serveur ne parlent qu'à cette interface. Conséquence mécanique : les dix conditions sont mesurées par exactement le même code, ce dont dépend toute la comparabilité de la section 11.
3. **Aucun module ne construit un générateur aléatoire.** On en demande un à `graines`, qui sépare le flux des pièces, celui du départage et celui du run. C'est la seule protection contre le bug le plus coûteux du projet — un départage qui décale la séquence de pièces et rend fausses toutes les comparaisons appariées, sans lever la moindre erreur.

### 15.3 Ce que cette disposition corrige

Quatre choix qui diffèrent d'une arborescence à plat, chacun pour une raison précise :

- **`evaluation` et non `eval`** : `eval` est une fonction intégrée de Python, et un paquet qui la masque finit toujours par surprendre quelqu'un ;
- **`data/` n'est plus un module** : le code d'acquisition vit dans `connectome`, et `data/` ne contient que des octets — 1,2 Go de brut qui n'ont rien à faire dans un paquet importable ;
- **disposition `src/`** : sans elle, les tests importent le dossier de travail plutôt que le paquet installé, et une erreur d'empaquetage ne se voit qu'au déploiement ;
- **les trois modules de frontière à la racine** : `graines`, `notation` et `encodage` ne sont pas des utilitaires, ce sont les coutures. Les enterrer dans des sous-paquets rendrait invisible ce qui tient l'architecture.

---

## 16. Qualité, reproductibilité, MLOps

- **Graines.** Toutes les graines sont fixées : Python, NumPy, PyTorch et moteur Tetris. Les valeurs sont celles de la section 10.5.
- **Données.** Chaque fichier préparé a une empreinte SHA-256, vérifiée à chaque chargement.
- **Configuration.** Un fichier YAML versionné par expérience. Chaque run est tracé dans MLflow : configuration, métriques, empreinte du graphe, graine.
- **Tests du moteur :**
  - disparition des lignes pleines, y compris quatre d'un coup ;
  - fin de partie ;
  - chaque sac contient les 7 pièces ;
  - nombre de candidats sur grille vide : 9, 17 ou 34 selon la pièce ;
  - une même graine donne la même séquence, **quel que soit le joueur** : le départage des égalités doit tirer sur un autre générateur (9.3), sans quoi l'évaluation appariée est fausse ;
  - l'expert retrouve son score sur trois graines fixées (test de non-régression) ;
  - séquence de touches reconstituée : la rejouer sur la grille d'avant redonne exactement le candidat choisi, pour les 34 candidats de chaque pièce (13.2).
- **Tests du modèle :**
  - orientation du graphe, sur un graphe jouet de 20 neurones ;
  - loi de Dale : aucun gain effectif négatif, et le signe de chaque connexion est bien celui de son neurone présynaptique ;
  - normalisation : la somme des valeurs absolues des poids entrants vaut 1 pour tout neurone qui a au moins une connexion entrante, et 0 pour les autres ;
  - couverture de la projection d'entrée : chaque entrée est lue par 20 ou 21 neurones, et chaque neurone d'entrée lit une seule entrée ;
  - **invariance au lot :** la note d'un candidat est la même qu'il soit évalué seul ou dans un lot de 34, et l'ordre des candidats dans le lot ne change rien ;
  - **remise à zéro :** deux évaluations successives du même candidat donnent la même note ;
  - graphe coupé et entrée aveuglée : sur le graphe jouet, tous les candidats reçoivent exactement la même note, donc le choix est purement aléatoire. Le niveau du hasard en lignes par partie (environ 0,1) est une mesure de la section 11.2, pas un test d'intégration continue : il demande le graphe complet et un GPU.
- **Tests des données :** l'empreinte SHA-256 de chaque fichier préparé est vérifiée au chargement, et un test la compare à la valeur publiée par flyhard.
- **Intégration continue.** GitHub Actions lance tous les tests à chaque envoi de code. Ils tournent sur le graphe jouet, sans télécharger le MaleCNS.
- **Déploiement.** Docker Compose avec 3 services :
  - api, avec accès au GPU ;
  - web, le site statique ;
  - mlflow, l'interface de suivi.
- **Publication.** Les résultats avec leurs intervalles, les contrôles et les échecs, comme le fait la référence.

---

## 17. Ordre de réalisation

**Pas de calendrier.** Le travail se fait au fil de l'eau. Ce qui compte n'est pas *quand*, mais **ce qui débloque quoi** : chaque étape a besoin de la sortie d'une autre, et les prendre dans le désordre revient à travailler sur des chiffres qu'on ne peut pas encore avoir.

| Étape | Ce qu'elle produit | Ce qu'elle exige d'abord |
|---|---|---|
| **A. Moteur** | Moteur, expert, joueur au hasard, évaluation appariée, tests | Rien. Tourne sur CPU, sans les données |
| **B. Repères** | Les démonstrations, la boucle d'entraînement, et le juge de grilles linéaire — la barre à dépasser | A |
| **C. Données** | Graphe préparé, empreinte vérifiée, 165 122 neurones et 25 563 197 connexions | Rien non plus — se lance en fond dès le départ, en parallèle de A et B |
| **D. Matériel** | Mémoire et temps par mise à jour mesurés sur la RTX 3050 | C |
| **E. Graphe** | D, facteur global, participation, composition des interfaces, K provisoire | B (pour les grilles candidates) et C |
| **F. Mise au point** | Chaîne complète validée, témoins au niveau du hasard, K arrêté | D et E |
| **G. La mouche** | Clonage, DAgger, évaluation de développement, graine 0 — en une ou plusieurs séances (14.6) | F |
| **H. Contrôles** | Recâblée, graphe coupé, entrée aveuglée, graine 0 | G — même recette exactement |
| **I. v0.1** | Évaluations finales de toutes les conditions, enregistrements, scène 3D, page publique en mode secours | H |
| **J. v1.0** | Graines 1 et 2, tableau final avec intervalles ; recâblage à signes conservés si un écart est apparu | I |
| **K. Direct** | Serveur de diffusion, tunnel, direct public allumé, mode duel | J — la carte ne peut pas diffuser pendant un entraînement (13.4) |

**Le GPU n'est pas partagé.** Les étapes D à J l'occupent entièrement, l'une après l'autre. Ce n'est pas une contrainte d'emploi du temps mais un verrou (13.4) : un seul usage à la fois, l'entraînement prioritaire.

### Trois points d'arrêt

Ce ne sont pas des jalons de calendrier. Ce sont les trois endroits où continuer sans vérifier coûte cher, et ils se prennent tous sur les parties de développement, jamais sur les 100 parties finales (11.1).

**1. Après le juge linéaire (B) — la barre est-elle en place.**

- *Le juge linéaire apprend et se place nettement au-dessus du hasard :* la boucle d'entraînement fonctionne, et la mouche héritera de la même. On enchaîne.
- *Il n'apprend pas :* le problème est dans la boucle, la perte ou les démonstrations, pas dans le connectome. Le corriger ici coûte des minutes de CPU ; le découvrir après quatre heures de GPU coûte une nuit.

**2. Après le run court (F) — est-ce que ça apprend.**

- *Accord avec l'expert d'au moins 10 %, contre environ 4 % au hasard :* lancer le clonage complet.
- *Moins de 10 % :* appliquer la parade « Pas d'apprentissage » (section 18) avant de lancer quoi que ce soit de long.

**3. Après le DAgger de la mouche (G) — est-ce qu'on dépasse la barre.**

- *Au moins le score du juge linéaire, mesuré dans le même protocole de développement :* on continue vers les contrôles.
- *En dessous :* appliquer « Pas d'apprentissage », refaire le run complet **une seule fois**, puis publier le résultat obtenu avec son analyse. Les contrôles et la v0.1 suivent de toute façon : un résultat négatif se publie comme un autre.

---

## 18. Risques et parades

| Risque | Symptôme | Parade, appliquée dans l'ordre |
|---|---|---|
| Activité qui s'emballe ou s'éteint | Plus de 20 % de neurones saturés, ou moins de la moitié des neurones moteurs actifs | Refaire la procédure du facteur global (7.4), qui écarte les valeurs saturantes et peut descendre sous 1 |
| Pas d'apprentissage | Accord avec l'expert inférieur à 10 % après 200 mises à jour, alors que le hasard est à environ 4 % | 1. Recalculer D et vérifier K ; 2. vérifier les témoins coupé et aveuglé ; 3. diviser le taux par 4 |
| Mouche sous le juge linéaire après DAgger | Moins que le repère linéaire mesuré dans le même protocole | Appliquer « Pas d'apprentissage », refaire une seule fois le run complet, puis publier le résultat obtenu avec son analyse |
| Mémoire GPU insuffisante | Erreur de mémoire | Procédure 14.5 |
| Calcul trop long | Run de plus de 6 heures | Procédure 14.5 |
| Dépôt de référence indisponible ou modifié | flyhard ou fly-self-driving change, disparaît, ou ne s'installe pas | Les deux dépôts sont épinglés à un commit précis dès le départ et recopiés dans le dépôt, noyau CUDA compris (licence MIT, crédit conservé) |
| La tâche de référence ne tourne pas sur 4 Go | Échec à l'étape D, qui bloque tout ce qui suit | Mesurer la mémoire et le temps par mise à jour directement sur le modèle Tetris, avec un lot de la forme prévue (40 évaluations, K mises à jour). La reproduction de la conduite devient facultative |
| Lecture des poids impossible | Le fichier de 1,1 Go et ses 152 millions de lignes saturent les 24 Go de RAM | Lecture par morceaux avec pyarrow et filtrage sur les neurones tracés avant toute matérialisation |
| Tricherie involontaire | Performances trop belles pour être vraies | Témoins coupé et aveuglé repassés à chaque version ; aucun pré-filtrage par l'expert ; test d'invariance au lot (16) |
| Sur-interprétation | Titres du genre « la mouche comprend Tetris » | Vocabulaire de la section 3 ; contrôles publiés avec les résultats |
| Données modifiées | Empreinte différente de celle de flyhard | Arrêt ; retour à la version des données dont l'empreinte est publiée |
| Direct public indisponible | Machine éteinte, GPU pris par un run, tunnel coupé | Le mode secours prend le relais automatiquement, avec un bandeau daté (13.3). La page n'affiche jamais un direct figé en prétendant qu'il est vivant |
| Direct public saturé | Plus de 50 spectateurs simultanés, ou débit montant insuffisant | Dégradation en trois temps : nuage à 1 000 neurones, puis bascule générale sur le secours, puis fermeture du tunnel (13.3) |
| Retard | Jalon manqué | Le calendrier de la dernière section fait foi : un jalon manqué décale la suite, il ne supprime aucune étape de la v0.1 |

---

## 19. Récapitulatif des décisions

| Domaine | Décision |
|---|---|
| Jeu | Grille 10 × 20, sacs de 7, choix de la position finale, chute verticale, pas de réserve, tour par tour, score en lignes |
| Moteur | Moteur Python écrit pour le projet, en phase 0 ; seul moteur du projet |
| Formulation | Juge de grilles : une note par grille résultante, le candidat le mieux noté est joué |
| Graphe | Préparé par les scripts de flyhard : 165 122 neurones tracés, 25 563 197 connexions, empreinte vérifiée |
| Signes | Acétylcholine +, GABA −, glutamate −, histamine −, tous les autres + |
| Normalisation | Poids divisés par le total des synapses reçues par chaque neurone ; entrée synaptique nulle si aucune connexion entrante |
| Facteur global | La plus petite valeur parmi 1, 2, 4, 8 qui active au moins la moitié des neurones moteurs sans saturer plus de 20 % du réseau ; grille prolongée vers le bas si aucune ne convient (8 par défaut) |
| Dynamique | Modèle à taux, état entre −1 et 1, tangente hyperbolique, fuite par neurone |
| Paramètres appris | Gains (softplus, départ 1), fuites (sigmoïde, départ 0,5), température (softplus, départ 1) ; 25 728 320 au total |
| Entrée | 205 entrées ; 4 114 neurones du lobe optique, 20 ou 21 par entrée, signes aléatoires, graine du run |
| Sortie | 708 neurones moteurs, coefficients normaux d'écart-type 1/√708, graine du run |
| Temps | État remis à zéro pour chaque candidat ; K = D + 4, plafonné à 16, vérifié au run court contre D + 2 et D + 8 ; candidats en lot |
| Expert | Heuristique à 4 critères (poids en 10.1), sans anticipation |
| Démonstrations | 410 parties, plafond 300, 5 % de bruit, environ 100 000 situations ; test : 20 parties sans bruit |
| Clonage | 10 candidats par situation, lots de 4, Adam 0,04, 2 400 mises à jour, points de contrôle toutes les 200 |
| DAgger | 3 tours de 30 parties (plafond 300) et 500 mises à jour ; lots tirés moitié dans DAgger, moitié dans les démonstrations |
| Évaluation | Développement : 30 parties, graines 1500–1529, plafond 300 ; final : 100 parties, graines 1000–1099, plafond 500 ; bootstrap apparié à 95 % sur les différences partie par partie |
| Contrôles | Recâblée, juge linéaire, graphe coupé, entrée aveuglée ; réservoir, MLP et recâblage à signes conservés reportés après la v0.1 |
| Graines | v0.1 : graine 0 ; v1.0 : graines 0, 1, 2, qui retirent aussi les interfaces |
| Démo | Mouche devant un téléviseur à tube, manette entre les pattes avant, séquence de touches reconstituée et annoncée comme telle ; 3 pièces fantômes ; 20 000 neurones affichés en local et 4 000 en public |
| Direct public | Une seule partie jouée en continu, diffusée à tous les spectateurs depuis la RTX 3050, en lecture seule, par tunnel sortant ; mode secours obligatoire quand la machine est occupée ou éteinte. Page en ligne en v0.1 (secours seul), direct allumé en v1.0 |
| Outillage | MLflow, YAML, pytest, GitHub Actions, Docker Compose à 3 services |
| Matériel | RTX 3050 uniquement, pas de GPU loué ; verrou GPU, l'entraînement prioritaire sur la diffusion |
| Divergence | Trois relances au maximum, puis le run est déclaré échoué et publié comme tel |
| Licences | Code sous MIT ; fichiers dérivés du MaleCNS sous CC-BY 4.0, attribution portée jusque sur la page publique |
| Ordre | Pas de calendrier : onze étapes enchaînées par leurs dépendances, trois points d'arrêt (section 17) |

---

## 20. Hors périmètre

Ces éléments sont exclus du projet ; ils ne seront pas réalisés.

| Élément exclu | Raison |
|---|---|
| Choix direct du placement par la mouche | 30 fois moins bon que le juge de grilles (section 12) |
| Commandes en temps réel (gauche, droite, rotation) | Demande un autre expert et une autre formulation. La manette de la démo n'en est pas une : elle rejoue une séquence reconstituée après coup, et l'écran le dit (13.2) |
| Réserve, glissements, rotations spéciales | Complexité du moteur sans gain pour la question posée |
| Pièce suivante ou marquage de la pièce posée en entrée | L'encodage mesuré suffit ; une seule variable à la fois |
| Mémoire entre les pièces | La grille contient toute l'information |
| Expert plus fort | L'expert actuel laisse déjà une marge énorme |
| Apprentissage par renforcement | L'imitation suivie de DAgger suffit et coûte bien moins |
| Simulation à impulsions (LIF) | Non entraînable par rétropropagation dans le budget de calcul |
| Plasticité « biologique » (famille C) | Résultats rarement démontrés |
| Autres entrées ou sorties (voie olfactive, neurones descendants) | Les populations de la référence sont reprises telles quelles |
| Sous-circuit de mise au point | Inutile : un run court sur le graphe complet ne coûte que quelques minutes |
| GPU loué | Tout tourne sur la RTX 3050, y compris la diffusion du direct public (13.3) |
| Une partie par visiteur dans le direct public | Un GPU, une partie : tout le monde regarde la même. Une partie par visiteur demanderait un parc de cartes |
| Corps de mouche simulé (NeuroMechFly) | Hors sujet pour un jeu tour par tour |

---

## 21. Références

**Données**

- MaleCNS v1.0 : https://male-cns.janelia.org/download/
- Exploration interactive sur neuPrint : https://neuprint.janelia.org

**Projets de référence**

- flyhard : https://github.com/MarkUnthank/flyhard
- fly-self-driving : https://github.com/suanmiao/fly-self-driving. La méthode est décrite dans docs/method.md, les résultats et les échecs dans docs/results.md.
- Site de fly-self-driving : https://fly-brain-drives.kylon.app

**Tetris**

- Critères de l'heuristique experte : IA de Tetris de Yiyuan Lee (hauteur totale, lignes complétées, trous, bosses).

**Méthodes et modèles**

- DAgger : Ross, Gordon et Bagnell, 2011, https://arxiv.org/abs/1011.0686
- Convention de signes et modèle du cerveau entier : Shiu et al., Nature, 2024, https://www.nature.com/articles/s41586-024-07763-9
- Réservoirs à base de connectomes : Suárez et al., « Connectome-based reservoir computing with the conn2res toolbox », Nature Communications, 2024 ; https://github.com/estefanysuarez/conn2res

**Autres projets cités**

- Sous-circuits entraînés avec loi de Dale (Othello) : https://huggingface.co/phclab/MushroomBody_Othello
- Lecture par neurones descendants, entraînement borné : https://github.com/Duck-luv-pie/hunting-fly
- Graphe jouet pour les tests : https://github.com/AntonioCoppe/flyciv

**Listes de projets**

- https://github.com/townie/awesome-fruit-fly
- https://github.com/cobanov/awesome-fly

**À sourcer.** Plusieurs projets cités dans le tableau de la section 2.1 n'ont pas de lien ici : Doom, Minecraft, Fruit Ninja, fly-craftax, Fly Chess Lab, BeatTheFly, Fly-Driving-Car, FlyPong. Leurs liens sont à retrouver dans les deux listes ci-dessus, en particulier celui du projet où un contrôle « entrée directe » a égalé le connectome : c'est l'argument qui justifie d'écarter la famille A, et il ne peut pas rester sans source.

---

## Annexe A — Glossaire

| Terme | Définition |
|---|---|
| Connectome | Carte complète des neurones et de leurs connexions |
| Neurone tracé | Neurone entièrement reconstruit, par opposition aux fragments |
| Loi de Dale | Un neurone a le même effet (excitateur ou inhibiteur) sur toutes ses cibles |
| Modèle à taux | Chaque neurone est représenté par une activité continue, pas par des impulsions |
| Rétropropagation | Calcul qui indique comment ajuster chaque réglage pour réduire l'erreur |
| Clonage de comportement | Apprentissage supervisé qui imite les choix d'un expert |
| DAgger | L'élève joue, l'expert étiquette les situations atteintes, et l'on réentraîne sur l'ensemble des données |
| Juge de grilles | Formulation où le modèle note chaque grille résultante, au lieu de choisir directement un placement |
| Candidat | Une façon possible de poser la pièce courante (rotation + colonne) |
| Sac de 7 | Tirage où chaque série de 7 pièces contient une pièce de chaque type |
| Trou | Case vide sous un bloc, dans la même colonne |
| Bosses | Somme des différences de hauteur entre colonnes voisines |
| Distance D | Nombre de sauts nécessaires pour atteindre 90 % des neurones moteurs depuis les neurones d'entrée |
| Saturation | État d'un neurone supérieur à 0,99 en valeur absolue : il ne transmet plus de variation |
| Mouche recâblée | Même nombre de connexions et même degré entrant, mais connexions redistribuées au hasard |
| Réservoir | Connectome figé dont seule la lecture de sortie est entraînée |
| Bootstrap apparié | Rééchantillonnage qui tire des graines de parties, les mêmes pour les deux conditions comparées, et donne l'intervalle de leur différence |
| Neurones descendants | Neurones qui transmettent les commandes du cerveau à la corde nerveuse ventrale |
| Corde nerveuse ventrale | Équivalent de la moelle épinière chez la mouche |
| Intervalle de confiance à 95 % | Plage qui contient la vraie valeur dans 95 % des cas |

## Annexe B — Check-list de démarrage

1. Sur CPU, sans rien télécharger : squelette du dépôt, moteur, expert, joueur au hasard, tests, évaluation appariée, puis élèves simples. Produire le tableau de la section 12 et le geler comme repère du projet (étapes A et B, section 17).
2. Cloner flyhard et fly-self-driving, **épingler les deux commits**, lancer l'acquisition et la préparation du graphe, puis vérifier l'empreinte.
3. Lancer un court entraînement de la tâche de référence sur la RTX 3050, et noter la mémoire utilisée et le temps par mise à jour. En cas d'échec, mesurer directement sur le modèle Tetris (section 18).
4. Mesurer D, fixer K et le facteur global, mesurer la participation et la composition des interfaces (sections 7.4 et 7.6).
5. Écrire les tests du moteur et du modèle (section 16).
6. Générer les 100 000 situations de démonstration, puis entraîner le juge linéaire et l'évaluer dans les deux protocoles (section 11.2).
