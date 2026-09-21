"""Les trois familles de graines du projet (conception.md §10.5).

Le mot « graine » désigne trois choses différentes dans ce projet, et les confondre
casse l'évaluation appariée **sans lever la moindre erreur** :

- si le départage des égalités tire sur le générateur des pièces, deux joueurs
  différents ne voient plus la même séquence, et toutes les comparaisons
  partie par partie deviennent fausses ;
- si les interfaces sont tirées sur le générateur des lots, changer la graine
  d'entraînement change aussi la projection d'entrée sans qu'on l'ait voulu.

Ce module est donc le **seul** endroit du projet autorisé à construire un générateur
aléatoire. Partout ailleurs, on en demande un ici.
"""

from __future__ import annotations

import numpy as np

# --- Plages de graines de parties, disjointes par construction (§10.5) ---
DEMONSTRATIONS = range(0, 410)
TEST = range(500, 520)
EVALUATION_FINALE = range(1000, 1100)
EVALUATION_DEVELOPPEMENT = range(1500, 1530)
DAGGER = (range(2000, 2030), range(2100, 2130), range(2200, 2230))
DIRECT_PUBLIC_DEBUT = 3000

# Décalages appliqués à une même graine de partie pour en dériver des flux
# indépendants. Deux flux issus d'une même partie ne doivent jamais se croiser.
_FLUX_PIECES = 0
_FLUX_DEPARTAGE = 1_000_000_007


def generateur_pieces(graine_partie: int) -> np.random.Generator:
    """Le flux qui fixe la séquence de pièces, et rien d'autre.

    Il ne dépend que du numéro de partie : toutes les conditions qui jouent la
    graine N voient exactement la même séquence, quel que soit leur talent.
    C'est ce qui rend l'évaluation appariée possible (§11.1).
    """
    return np.random.default_rng(graine_partie + _FLUX_PIECES)


def generateur_departage(graine_partie: int) -> np.random.Generator:
    """Le flux qui départage les candidats de note égale, et rien d'autre (§9.3).

    Séparé du précédent pour que tirer au sort entre deux placements ne décale
    jamais la séquence de pièces.
    """
    return np.random.default_rng(graine_partie + _FLUX_DEPARTAGE)


def generateur_run(graine_run: int) -> np.random.Generator:
    """Le flux d'un entraînement : interfaces, ordre des lots, tirage des candidats.

    Il tire aussi la projection d'entrée, le vecteur de sortie, la permutation du
    recâblage et le sous-ensemble de visualisation : en v1.0, changer de graine
    change donc le modèle **et** ses interfaces, ce qui est voulu (§8.3, §10.5).
    """
    return np.random.default_rng(graine_run)


def generateur_analyse(graine: int = 0) -> np.random.Generator:
    """Le flux du bootstrap, et de lui seul.

    Ce n'est pas une quatrième famille : il ne touche ni au modèle, ni au jeu, et
    ne change aucun résultat. Il est ici parce qu'un intervalle publié doit
    pouvoir être retrouvé au chiffre près (§11.1), et parce que ce module est le
    seul autorisé à construire un générateur.
    """
    return np.random.default_rng(graine)


def verifier_plages_disjointes() -> None:
    """Garde-fou : aucune graine de partie ne doit servir à deux usages.

    Appelé par les tests. Une collision ici voudrait dire qu'on évalue sur des
    parties vues à l'entraînement.
    """
    plages = {
        "demonstrations": set(DEMONSTRATIONS),
        "test": set(TEST),
        "evaluation_finale": set(EVALUATION_FINALE),
        "evaluation_developpement": set(EVALUATION_DEVELOPPEMENT),
        "dagger": set().union(*DAGGER),
    }
    noms = sorted(plages)
    for i, a in enumerate(noms):
        for b in noms[i + 1 :]:
            commun = plages[a] & plages[b]
            if commun:
                raise AssertionError(f"graines partagées entre {a} et {b} : {sorted(commun)[:5]}")
