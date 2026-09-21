"""L'expert : quatre critères, quatre poids, un argmax (conception.md §10.1).

Les poids viennent de l'IA de Tetris de Yiyuan Lee. Ils ne sont pas ajustés ici,
et ne le seront pas : un expert plus fort est explicitement hors périmètre (§20),
parce que celui-ci laisse déjà une marge énorme entre le hasard et lui.

Deux règles d'honnêteté, et elles ne sont pas décoratives :

- **il voit exactement la même chose que la mouche** — la même liste de
  candidats, la même grille résultante. Aucun professeur ne triche avec des
  informations cachées, et il n'anticipe pas la pièce suivante ;
- **il ne filtre jamais les candidats.** Il note, il ne réduit pas. S'il
  restreignait la liste, ce serait lui qui jouerait, et la mesure ne voudrait
  plus rien dire (§3).
"""

from __future__ import annotations

import numpy as np

from ..tetris.grille import mesures

POIDS_HAUTEUR = -0.510066
POIDS_LIGNES = 0.760666
POIDS_TROUS = -0.35663
POIDS_BOSSES = -0.184483


class Expert:
    """Le professeur. Implémente `Noteur` (voir `notation`)."""

    nom = "expert"

    def noter(self, candidats) -> np.ndarray:
        notes = np.empty(len(candidats), dtype=np.float64)
        for indice, candidat in enumerate(candidats):
            mesure = mesures(candidat.grille)
            notes[indice] = (
                POIDS_HAUTEUR * mesure.hauteur_totale
                + POIDS_LIGNES * candidat.lignes_completees
                + POIDS_TROUS * mesure.trous
                + POIDS_BOSSES * mesure.bosses
            )
        return notes
