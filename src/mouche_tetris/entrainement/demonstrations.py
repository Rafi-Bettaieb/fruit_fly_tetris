"""Les démonstrations de l'expert (conception.md §10.2).

410 parties, graines 0 à 409, plafond 300 pièces, **5 % des coups joués au
hasard en gardant l'étiquette de l'expert**. Environ 100 000 situations.

**Pourquoi le bruit.** Sans lui, l'expert ne visite que de belles grilles, et
l'élève n'apprend jamais à sortir d'une grille abîmée — alors que c'est
exactement là qu'il se retrouvera dès qu'il jouera seul. Les 5 % de coups au
hasard produisent des situations dégradées **avec la bonne réponse à côté** :
ce que l'expert aurait joué, pas ce qui a été joué. C'est l'équivalent du bruit
de braquage de la référence, et c'est ce qui rend le clonage utilisable avant
même DAgger.

**Chaque situation garde tous ses candidats.** Pas seulement le choix de
l'expert : la perte du clonage est une softmax sur plusieurs candidats (§10.3),
il lui faut donc les perdants autant que le gagnant.

**Jeu de test.** 20 parties sans bruit, graines 500 à 519, tenues à l'écart de
tout entraînement. Elles servent à choisir le meilleur point de contrôle et à
mesurer l'accord avec l'expert.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .. import graines
from ..expert.criteres import Expert
from ..tetris.candidats import Candidat, enumerer
from ..tetris.grille import GRILLE_VIDE, Grille
from ..tetris.partie import choisir
from ..tetris.sacs import sequence

PLAFOND = 300
BRUIT = 0.05


class Situation(NamedTuple):
    """Une décision de l'expert, avec tout ce qu'il avait sous les yeux."""

    grille_avant: Grille
    lettre: str
    candidats: tuple[Candidat, ...]
    choix_expert: int
    """Indice dans `candidats` — la cible de la perte, **toujours** le choix de
    l'expert, même quand un coup au hasard a été joué à la place."""

    meilleurs_expert: tuple[int, ...]
    """Tous les candidats que l'expert note au maximum, `choix_expert` compris.

    L'expert départage ses égalités au hasard, et sur une grille symétrique il
    en a souvent. Mesurer l'accord contre son tirage précis punirait un modèle
    qui choisit l'autre placement aussi bon : l'accord se mesure donc contre cet
    ensemble (voir `evaluation.accord`)."""

    joue: int
    """Indice réellement joué. Diffère de `choix_expert` sur 5 % des coups."""

    @property
    def bruite(self) -> bool:
        return self.joue != self.choix_expert


def generer_partie(graine: int, bruit: float = BRUIT, plafond: int = PLAFOND) -> list[Situation]:
    """Une partie de l'expert, bruitée, avec toutes ses situations étiquetées."""
    expert = Expert()
    grille = GRILLE_VIDE
    flux = sequence(graine)
    departage = graines.generateur_departage(graine)
    # Le bruit tire sur le flux du run, pas sur celui des pièces : la séquence
    # de pièces d'une graine reste la même, bruit ou pas (§10.5).
    alea = graines.generateur_run(graine)

    situations: list[Situation] = []
    for _ in range(plafond):
        lettre = next(flux)
        liste = enumerer(grille, lettre)
        if not liste:
            break

        notes = expert.noter(liste)
        choix_expert = choisir(notes, departage)
        meilleurs = tuple(int(i) for i in np.flatnonzero(notes == notes.max()))

        if bruit and alea.random() < bruit:
            joue = int(alea.integers(0, len(liste)))
        else:
            joue = choix_expert

        situations.append(Situation(grille, lettre, liste, choix_expert, meilleurs, joue))
        grille = liste[joue].grille

    return situations


def generer(
    graines_parties: range,
    bruit: float = BRUIT,
    plafond: int = PLAFOND,
) -> list[Situation]:
    """Toutes les situations d'un ensemble de parties."""
    situations: list[Situation] = []
    for graine in graines_parties:
        situations.extend(generer_partie(graine, bruit=bruit, plafond=plafond))
    return situations


def jeu_d_entrainement() -> list[Situation]:
    """Les 410 parties bruitées, environ 100 000 situations (§10.2)."""
    return generer(graines.DEMONSTRATIONS, bruit=BRUIT)


def jeu_de_test() -> list[Situation]:
    """Les 20 parties sans bruit, tenues à l'écart de tout entraînement."""
    return generer(graines.TEST, bruit=0.0)


def echantillon_de_test(situations: list[Situation], combien: int = 500) -> list[Situation]:
    """Le sous-échantillon fixe qui sert à choisir le meilleur point de contrôle.

    Noter les 6 000 situations de test à chaque point de contrôle coûterait une
    demi-heure de GPU, soit six heures par run (§10.3). 500 suffisent, tirées
    une fois avec la graine 0 et jamais retirées.
    """
    generateur = graines.generateur_analyse(0)
    indices = generateur.choice(len(situations), size=min(combien, len(situations)), replace=False)
    return [situations[i] for i in sorted(indices)]


def tirer_candidats(
    situation: Situation, combien: int, generateur: np.random.Generator
) -> tuple[list[Candidat], int]:
    """Le choix de l'expert, plus `combien − 1` autres tirés au hasard (§10.3).

    Rend la liste mélangée et l'indice de la bonne réponse dedans. Le mélange
    compte : si la cible était toujours en position 0, un modèle pourrait
    apprendre la position plutôt que la grille.
    """
    total = len(situation.candidats)
    if total <= combien:
        retenus = list(range(total))
    else:
        autres = [i for i in range(total) if i != situation.choix_expert]
        tires = generateur.choice(len(autres), size=combien - 1, replace=False)
        retenus = [situation.choix_expert] + [autres[i] for i in tires]
        generateur.shuffle(retenus)
    return [situation.candidats[i] for i in retenus], retenus.index(situation.choix_expert)


def tirer_difficiles(
    situation: Situation, combien: int, notes: np.ndarray, generateur: np.random.Generator
) -> tuple[list[Candidat], int]:
    """Le choix de l'expert, plus les `combien − 1` autres que le **modèle** note le plus haut.

    Les concurrents tirés au hasard sont souvent des placements absurdes que le
    modèle écarte déjà : ils n'apprennent presque rien. L'accord, lui, se joue
    entre le choix de l'expert et les meilleurs rivaux du modèle — ses propres
    confusions. On les lui montre donc à chaque pas.

    **Ce n'est pas un pré-filtrage par l'expert** (§3) : c'est le modèle qui
    choisit ses concurrents, l'expert ne fournit que la bonne réponse, comme
    avant. En jeu et en évaluation, le modèle voit toujours tous les candidats.
    """
    total = len(situation.candidats)
    if total <= combien:
        retenus = list(range(total))
    else:
        autres = [i for i in range(total) if i != situation.choix_expert]
        # Tri stable sur la note, du plus haut au plus bas : à notes égales,
        # l'ordre d'énumération départage, et le tirage reste reproductible.
        autres.sort(key=lambda i: -notes[i])
        retenus = [situation.choix_expert] + autres[: combien - 1]
    generateur.shuffle(retenus)
    return [situation.candidats[i] for i in retenus], retenus.index(situation.choix_expert)
