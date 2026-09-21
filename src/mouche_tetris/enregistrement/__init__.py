"""Capture des parties pour la démo et pour l'analyse (conception.md §13.5).

Par pièce : la grille, la pièce, les candidats avec leurs notes et leurs
probabilités, le choix de la mouche, et l'état des neurones de visualisation
pour le candidat choisi — un octet par neurone, échelle propre à chacun.

Environ 20 Ko par pièce, 10 Mo pour une partie de 500 pièces.

Trois usages, maintenant que la page publique diffuse du direct :

- le **mode secours**, qui rejoue la dernière partie quand la machine est
  occupée ou éteinte ;
- les **parties vitrines** : 3 parties de 100 pièces qui enregistrent en plus
  chacune des K mises à jour du candidat choisi. Impossible en direct, cela
  multiplierait le débit par K ;
- l'**analyse** : toute partie du tableau de résultats doit pouvoir être rejouée
  coup par coup.

La séquence de touches n'est pas enregistrée : elle se recalcule à l'affichage à
partir du candidat choisi (`tetris.touches`).

Dans les fichiers publiés, le nuage est ramené à 4 000 neurones et compressé —
environ 2 Mo par partie. Les 20 000 restent en local.
"""
