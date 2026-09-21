"""L'accord avec l'expert (conception.md §11.1).

Part des situations de test où le noteur jouerait le même candidat que
l'expert. C'est la métrique qui sert à deux choses :

- **choisir le meilleur point de contrôle** pendant un entraînement, sur un
  sous-échantillon fixe de 500 situations (§10.3) ;
- **le point d'arrêt après le run court** : moins de 10 % d'accord après
  200 mises à jour déclenche la parade « Pas d'apprentissage » (§18).

**Le niveau du hasard est d'environ 4 %**, soit une chance sur les 22,5 candidats
que compte une situation en moyenne. Sans ce repère, le seuil de 10 % ne veut
rien dire : il n'est que deux fois et demie le hasard, et c'est volontaire —
c'est un test de « est-ce que ça apprend du tout », pas de « est-ce que c'est
bon ».

**On mesure contre la préférence de l'expert, pas contre son tirage.** L'expert
départage ses égalités au hasard, et sur une grille symétrique il en a souvent.
Compter comme une erreur le fait de jouer l'autre placement — que l'expert note
exactement pareil — serait injuste et ferait tomber l'expert sous 100 % d'accord
avec lui-même. L'accord est donc : *le candidat choisi fait-il partie de ceux
que l'expert note au maximum ?*

Les égalités du **noteur**, elles, sont départagées comme en jeu, sur un flux
fixé : la mesure reste reproductible, et un noteur qui donne la même note à tout
— le joueur au hasard, le graphe coupé, l'entrée aveuglée — retombe bien sur le
hasard au lieu d'être flatté par un argmax qui choisirait toujours le premier.
"""

from __future__ import annotations

import numpy as np

from .. import graines
from ..tetris.partie import choisir


def reussites(noteur, situations, graine: int = 0) -> np.ndarray:
    """Pour chaque situation, vrai si le noteur y joue un optimal de l'expert.

    C'est ce vecteur, et non sa moyenne, qui permet de comparer deux modèles
    situation par situation : les mêmes situations des deux côtés, et le même
    flux de départage.
    """
    departage = graines.generateur_analyse(graine)
    justes = np.zeros(len(situations), dtype=bool)
    for indice, situation in enumerate(situations):
        notes = np.asarray(noteur.noter(situation.candidats), dtype=np.float64)
        justes[indice] = choisir(notes, departage) in situation.meilleurs_expert
    return justes


def accord(noteur, situations, graine: int = 0) -> float:
    """Part des situations où le noteur joue un candidat que l'expert juge optimal."""
    if not situations:
        return 0.0
    return float(reussites(noteur, situations, graine).mean())


def niveau_du_hasard(situations) -> float:
    """L'accord qu'obtiendrait un noteur sans préférence, à comparer au reste.

    C'est la moyenne du rapport « candidats optimaux / candidats », et non
    1 / 22,5 : les situations à peu de candidats, ou à plusieurs optima, sont
    plus faciles à deviner et pèsent dans la moyenne.
    """
    if not situations:
        return 0.0
    return float(np.mean([len(s.meilleurs_expert) / len(s.candidats) for s in situations]))
