"""Les séries de parties appariées (conception.md §11.1).

Deux protocoles, et deux seulement :

| Usage | Parties | Graines | Plafond |
|---|---|---|---|
| Développement | 30 | 1500–1529 | 300 pièces |
| Résultats finaux | 100 | 1000–1099 | 500 pièces |

Les deux plages sont **disjointes**. Tous les arbitrages du projet — les trois
points d'arrêt de la section 17, les parades de la section 18 — se prennent sur
les parties de développement, et les 100 parties finales ne sont regardées qu'au
moment du tableau de résultats. Sans cette séparation, on choisirait un modèle
sur le jeu de parties qui sert ensuite à publier ses performances.

Toutes les conditions jouent exactement les mêmes séquences, ce qui rend les
différences comparables partie par partie (voir `bootstrap`).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..tetris.partie import Partie, jouer


class Protocole(NamedTuple):
    nom: str
    graines: range
    plafond: int

    @property
    def parties(self) -> int:
        return len(self.graines)


DEVELOPPEMENT = Protocole("développement", range(1500, 1530), 300)
FINAL = Protocole("final", range(1000, 1100), 500)


class Serie(NamedTuple):
    """Ce qu'une condition a fait sur un protocole, partie par partie."""

    nom: str
    protocole: Protocole
    parties: tuple[Partie, ...]

    @property
    def lignes(self) -> np.ndarray:
        """Les lignes de chaque partie, dans l'ordre des graines.

        C'est ce tableau que le bootstrap apparié rééchantillonne : l'indice `i`
        désigne la même séquence de pièces pour toutes les conditions.
        """
        return np.array([partie.lignes for partie in self.parties], dtype=np.float64)

    @property
    def moyenne(self) -> float:
        return float(self.lignes.mean())

    @property
    def mediane(self) -> float:
        return float(np.median(self.lignes))

    @property
    def pieces_moyennes(self) -> float:
        return float(np.mean([partie.pieces_posees for partie in self.parties]))

    @property
    def part_plafond(self) -> float:
        """Part des parties qui atteignent le plafond.

        À publier avec la moyenne : quand elle vaut 1, la moyenne est tronquée
        par le plafond et devient un plancher, pas une performance (§11.1).
        """
        return float(np.mean([partie.plafond_atteint for partie in self.parties]))

    @property
    def trous_par_piece(self) -> float:
        posees = sum(partie.pieces_posees for partie in self.parties)
        crees = sum(partie.trous_crees for partie in self.parties)
        return crees / posees if posees else 0.0


def serie(noteur, protocole: Protocole = FINAL, enregistrer_coups: bool = False) -> Serie:
    """Fait jouer un noteur sur toutes les parties d'un protocole."""
    return Serie(
        nom=noteur.nom,
        protocole=protocole,
        parties=tuple(
            jouer(noteur, graine=graine, plafond=protocole.plafond,
                  enregistrer_coups=enregistrer_coups)
            for graine in protocole.graines
        ),
    )
