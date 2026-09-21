"""La boucle de jeu : tour par tour, sans gravité ni horloge (conception.md §5.2).

Une partie confronte un `Noteur` à une séquence de pièces fixée par une graine.
À chaque tour : énumérer les candidats, les faire noter, jouer le mieux noté,
recommencer. Fin quand la pièce ne tient plus dans les 20 lignes, ou quand le
plafond de pièces est atteint.

**Le même code joue toutes les conditions.** La mouche, la mouche recâblée, le
réservoir, les élèves simples, les témoins, l'expert et le joueur au hasard
passent tous par ici, sans aucune branche particulière. C'est ce qui rend leurs
scores comparables (§11).

**Le départage des égalités tire sur son propre flux** (§9.3). Si deux candidats
ont exactement la même note — ce qui arrive tout le temps chez l'expert par
symétrie, et systématiquement chez les témoins, dont toutes les notes sont
égales — on en choisit un au hasard, sans jamais toucher au flux des pièces.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .. import graines
from . import sacs
from .candidats import Candidat, enumerer
from .grille import GRILLE_VIDE, Grille, trous


class Coup(NamedTuple):
    """Ce qui s'est passé pour une pièce. Conservé sur demande seulement."""

    grille_avant: Grille
    lettre: str
    candidats: tuple[Candidat, ...]
    choisi: int
    notes: np.ndarray


class Partie(NamedTuple):
    """Le résultat d'une partie, tel qu'il alimente le tableau de la section 11.1."""

    lignes: int
    pieces_posees: int
    plafond_atteint: bool
    trous_crees: int
    graine: int
    coups: tuple[Coup, ...]

    @property
    def trous_par_piece(self) -> float:
        return self.trous_crees / self.pieces_posees if self.pieces_posees else 0.0


def choisir(notes: np.ndarray, departage: np.random.Generator) -> int:
    """L'indice du meilleur candidat, égalités départagées au hasard.

    L'égalité se teste exactement, pas à une tolérance près : chez l'expert, des
    positions symétriques donnent la même note au bit près, et chez les témoins
    toutes les notes sont identiques — ce qui fait de ce départage le seul
    mécanisme de décision, donc un joueur uniformément au hasard (§11.2).
    """
    meilleures = np.flatnonzero(notes == notes.max())
    if len(meilleures) == 1:
        return int(meilleures[0])
    return int(meilleures[departage.integers(0, len(meilleures))])


def jouer(
    noteur,
    graine: int,
    plafond: int,
    enregistrer_coups: bool = False,
) -> Partie:
    """Joue une partie complète et rend son résultat.

    `enregistrer_coups` conserve chaque situation avec ses candidats et ses
    notes : indispensable pour les démonstrations et les enregistrements
    (§10.2, §13.5), inutile et coûteux pour une simple évaluation.
    """
    grille = GRILLE_VIDE
    flux = sacs.sequence(graine)
    departage = graines.generateur_departage(graine)

    lignes = 0
    posees = 0
    trous_crees = 0
    trous_avant = 0
    coups: list[Coup] = []

    while posees < plafond:
        lettre = next(flux)
        liste = enumerer(grille, lettre)
        if not liste:
            break  # la pièce ne tient plus : fin de partie (§5.2)

        notes = np.asarray(noteur.noter(liste), dtype=np.float64)
        if notes.shape != (len(liste),):
            raise ValueError(
                f"{noteur.nom} a rendu {notes.shape} notes pour {len(liste)} candidats"
            )
        indice = choisir(notes, departage)

        if enregistrer_coups:
            coups.append(Coup(grille, lettre, liste, indice, notes))

        choisi = liste[indice]
        grille = choisi.grille
        lignes += choisi.lignes_completees
        posees += 1

        trous_apres = trous(grille)
        trous_crees += max(0, trous_apres - trous_avant)
        trous_avant = trous_apres

    return Partie(
        lignes=lignes,
        pieces_posees=posees,
        plafond_atteint=posees >= plafond,
        trous_crees=trous_crees,
        graine=graine,
        coups=tuple(coups),
    )
