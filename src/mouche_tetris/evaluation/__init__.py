"""Parties appariées, intervalles de confiance, rapports (conception.md §11.1).

Nommé `evaluation` et non `eval` : `eval` est une fonction intégrée de Python,
et un paquet qui la masque est une mauvaise surprise garantie.

- `parties`   — séries appariées. Développement : 30 parties, graines 1500–1529,
                plafond 300. Final : 100 parties, graines 1000–1099, plafond 500.
                Les deux plages sont disjointes : aucune décision du projet n'est
                prise sur les parties qui serviront à publier
- `bootstrap` — rééchantillonnage **apparié**, 10 000 tirages : on tire des
                graines de parties, les mêmes pour les deux conditions comparées,
                et on publie l'intervalle de leur différence
- `rapport`   — tableau de résultats, budgets, échecs

Toutes les conditions jouent exactement les mêmes séquences. C'est la méthode
centrale du projet, et elle repose entièrement sur la séparation des générateurs
de `graines`.

Ce que l'intervalle mesure, et ce qu'il ne mesure pas : il mesure le bruit de
partie à partie, **à graine fixée**. Il ne dit rien de la variabilité d'une
graine à l'autre. Avec trois graines, celle-ci est vue, pas estimée. Le facteur
limitant d'une conclusion est donc le nombre de graines, pas le nombre de
parties — et un écart qui change de signe d'une graine à l'autre n'est pas un
effet, aussi fins que soient les intervalles.
"""
