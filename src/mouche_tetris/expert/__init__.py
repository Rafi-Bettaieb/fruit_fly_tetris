"""L'expert : l'heuristique à 4 critères qui sert de professeur (conception.md §10.1).

Quatre critères sur la grille résultante, quatre poids fixes, un argmax :

| Critère                                      | Poids      |
|----------------------------------------------|------------|
| Hauteur totale des colonnes                  | −0,510066  |
| Lignes complétées                            | +0,760666  |
| Trous (cases vides sous un bloc)             | −0,35663   |
| Bosses (différences de hauteur voisines)     | −0,184483  |

Origine : l'IA de Tetris de Yiyuan Lee.

Deux règles :

- l'expert n'anticipe pas la pièce suivante, et voit exactement la même grille
  que la mouche. Aucun professeur ne triche avec des informations cachées ;
- **l'expert ne filtre jamais les candidats.** C'est la règle absolue du projet
  (§3, §14.5) : s'il réduit la liste des placements, c'est lui qui joue. Interdit
  même comme parade à un manque de mémoire ou de temps.

Il est aussi un `Noteur` (voir `notation`), ce qui lui permet d'être mesuré par
exactement le même code que toutes les autres conditions.
"""
