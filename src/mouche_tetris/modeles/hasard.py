"""Le joueur au hasard : le plancher du projet, environ 0,1 ligne par partie.

Il ne tire rien lui-même. Il donne **la même note à tous les candidats**, et
c'est le départage des égalités (§9.3) qui choisit uniformément. Un tirage de
moins, et surtout : le joueur au hasard devient exactement ce que sont les deux
témoins de §11.2.

Le graphe coupé met tous les gains effectifs à 0, l'entrée aveuglée met toutes
les entrées sensorielles à 0 ; dans les deux cas les 708 neurones moteurs
restent à zéro, toutes les notes sont égales, et le choix redevient purement
aléatoire. Les trois conditions doivent donc rendre le même score, et tout écart
signale un raccourci caché. C'est le témoin le plus important du projet, et il
se réduit à ce fichier.
"""

from __future__ import annotations

import numpy as np


class Hasard:
    """Implémente `Noteur`. Aucune préférence, jamais."""

    nom = "hasard"

    def noter(self, candidats) -> np.ndarray:
        return np.zeros(len(candidats), dtype=np.float64)
