"""Le moteur — l'unique moteur du projet (conception.md §5.2, §6).

Il sert à l'entraînement, à l'évaluation, aux enregistrements et au serveur. Le
navigateur n'en a aucun : il affiche ce qu'on lui envoie.

Modules prévus :

- `pieces`    — les 7 tétrominos et leurs rotations distinctes
- `grille`    — grille 10 × 20, pose, disparition des lignes, fin de partie
- `sacs`      — tirage par sacs de 7
- `candidats` — énumération des placements : O 9, I/S/Z 17, T/J/L 34, jamais plus de 34
- `partie`    — boucle de jeu, plafond de pièces, comptage des lignes
- `touches`   — reconstitution de la séquence de touches d'un candidat, pour la
                démo seulement (§13.2) ; la mouche n'appuie sur aucun bouton

Dépendances interdites ici : PyTorch, CUDA, le connectome. Ce paquet doit rester
testable sur une machine nue.

Deux pièges, tous deux silencieux :

1. le départage des égalités doit tirer sur `graines.generateur_departage`, jamais
   sur celui des pièces — sinon deux joueurs ne voient plus la même séquence et
   toute l'évaluation appariée est fausse ;
2. la grille rendue pour un candidat est celle **d'après** la disparition des
   lignes, accompagnée du nombre de lignes disparues. C'est exactement ce que
   `encodage` attend.
"""
