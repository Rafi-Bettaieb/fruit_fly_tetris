"""Les modèles de comparaison (conception.md §11.2).

Ce sont eux qui placent la barre, et elle est haute : un MLP de 128 neurones
cachés, entraîné en une minute, est attendu autour de 62,6 lignes par partie.

- `hasard`     — notes tirées au sort ; le plancher, environ 0,1 ligne
- `lineaire`   — combinaison linéaire des 205 entrées
- `mlp`        — une couche cachée de 128 (ReLU)
- `convolutif` — 2 couches 3 × 3 à 32 filtres (ReLU, taille 20 × 10 conservée),
                 puis les 5 entrées « lignes complétées » ajoutées, une couche de
                 128 (ReLU) et la note
- `direct`     — élèves à choix direct, écrits **une seule fois**, en phase 0, pour
                 mesurer l'écart qui justifie la formulation « juge de grilles ».
                 Ensuite hors périmètre (§20)

Tous implémentent `Noteur` : ils sont donc mesurés par le même code que la mouche.

Entraînement : mêmes démonstrations, même perte, 10 candidats par situation ;
Adam 0,001, lots de 64, 10 époques ; puis 3 tours de DAgger à 3 époques.
Ces 10 époques représentent bien plus de situations vues que les 2 400 mises à
jour de la mouche : l'écart est assumé, mais le budget de chacun est publié à
côté de son score (voir `notation.Budget`).
"""
