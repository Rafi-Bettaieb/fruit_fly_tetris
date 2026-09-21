"""DAgger : l'élève joue, l'expert corrige (conception.md §10.4).

Trois tours. À chaque tour, le modèle joue seul 30 parties, l'expert étiquette
**chaque situation réellement atteinte**, et l'on réentraîne sur l'ensemble
accumulé.

**Pourquoi c'est indispensable ici.** À Tetris, une erreur creuse un trou, puis
un autre : l'élève arrive vite dans des grilles abîmées que l'expert, qui joue
bien, ne rencontre jamais. Le clonage seul l'entraîne donc sur une distribution
de grilles qu'il ne verra pas. C'est l'équivalent exact de la voiture qui dérive
hors de la trajectoire experte.

**La composition des lots n'est pas neutre.** La moitié de chaque lot vient des
situations de DAgger, l'autre moitié des démonstrations d'origine. Un tirage
uniforme sur l'ensemble accumulé donnerait aux situations de DAgger 18 % du
poids — 21 900 contre 101 017 — là où les mesures de la littérature qui
justifient DAgger en donnaient 47 %, avec quatre fois moins de démonstrations.
Sans ce rééquilibrage, le gain attendu ne serait pas transposable.

**Chaque condition produit ses propres données.** Les graines sont communes,
les situations atteintes ne le sont pas : c'est le principe même de la méthode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import graines
from ..expert.criteres import Expert
from ..tetris.partie import choisir, jouer
from .clonage import Journal, Reglages, entrainer
from .demonstrations import PLAFOND, Situation

TOURS = graines.DAGGER


def collecter(noteur, graines_parties, plafond: int = PLAFOND) -> list[Situation]:
    """Fait jouer le modèle, puis fait étiqueter par l'expert ce qu'il a atteint.

    L'élève joue toujours son meilleur candidat, sans exploration : ce sont ses
    vraies erreurs qu'on veut corriger, pas des erreurs ajoutées.
    """
    expert = Expert()
    situations: list[Situation] = []
    for graine in graines_parties:
        partie = jouer(noteur, graine=graine, plafond=plafond, enregistrer_coups=True)
        departage = graines.generateur_departage(graine)
        for coup in partie.coups:
            notes = expert.noter(coup.candidats)
            choix = choisir(notes, departage)
            meilleurs = tuple(int(i) for i in np.flatnonzero(notes == notes.max()))
            situations.append(
                Situation(
                    grille_avant=coup.grille_avant,
                    lettre=coup.lettre,
                    candidats=coup.candidats,
                    choix_expert=choix,
                    meilleurs_expert=meilleurs,
                    joue=coup.choisi,
                )
            )
    return situations


@dataclass
class ResultatDagger:
    tours: list[Journal]
    situations_collectees: list[int]

    @property
    def total_collecte(self) -> int:
        return sum(self.situations_collectees)


def executer(
    modele,
    adaptateur,
    demonstrations: list[Situation],
    situations_de_test: list[Situation],
    reglages: Reglages,
    nom: str = "modèle",
    graine_run: int = 0,
    plafond: int = PLAFOND,
    tours=TOURS,
    peripherique: str = "cpu",
):
    """Les trois tours de DAgger, à partir d'un modèle déjà cloné.

    `adaptateur` transforme le modèle en `Noteur` pour qu'il joue — pour un
    modèle PyTorch, `NoteurTorch`.
    """
    accumule: list[Situation] = []
    resultat = ResultatDagger(tours=[], situations_collectees=[])

    for numero, graines_du_tour in enumerate(tours, start=1):
        nouvelles = collecter(adaptateur(modele), graines_du_tour, plafond)
        accumule.extend(nouvelles)
        resultat.situations_collectees.append(len(nouvelles))

        modele, journal = entrainer(
            modele,
            demonstrations,
            situations_de_test,
            reglages,
            graine_run=graine_run + numero,
            nom=f"{nom} · DAgger {numero}",
            peripherique=peripherique,
            situations_dagger=accumule,
        )
        resultat.tours.append(journal)

    return modele, resultat
