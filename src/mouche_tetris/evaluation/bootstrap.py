"""Intervalles de confiance par rééchantillonnage (conception.md §11.1).

10 000 tirages, méthode des percentiles, graine fixée pour qu'un intervalle
publié puisse être retrouvé au chiffre près.

**Apparié, pas indépendant.** Toutes les conditions jouent les mêmes séquences
de pièces. L'écart entre deux d'entre elles se mesure donc partie par partie :
le rééchantillonnage tire des **indices de parties**, et les mêmes indices
servent aux deux conditions à chaque tirage. On publie l'intervalle de la
différence moyenne, et « écart démontré » exige qu'il exclue 0.

Deux intervalles calculés séparément qui ne se recouvrent pas forment un critère
à la fois plus sévère et moins informatif : ils restent affichés, mais ne
servent pas de test.

**Ce que cet intervalle ne mesure pas.** Il mesure le bruit de partie à partie,
*à graine de run fixée*. Il ne dit rien de la variabilité d'une graine à l'autre
— celle du tirage des interfaces et de l'optimisation. Avec trois graines, cette
seconde variabilité est vue, pas estimée. Le facteur limitant d'une conclusion
est donc le nombre de graines, pas le nombre de parties : passer de 100 à 500
parties resserrerait un intervalle qui n'est déjà pas le bon.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .. import graines

TIRAGES = 10_000
NIVEAU = 0.95


class Intervalle(NamedTuple):
    valeur: float
    bas: float
    haut: float

    @property
    def exclut_zero(self) -> bool:
        """Le critère de conclusion du projet, pour une différence."""
        return self.bas > 0 or self.haut < 0

    def __str__(self) -> str:
        return f"{self.valeur:.1f} [{self.bas:.1f} ; {self.haut:.1f}]"


def _percentiles(echantillons: np.ndarray, niveau: float) -> tuple[float, float]:
    marge = (1 - niveau) / 2 * 100
    return float(np.percentile(echantillons, marge)), float(
        np.percentile(echantillons, 100 - marge)
    )


def intervalle(
    valeurs: np.ndarray, tirages: int = TIRAGES, niveau: float = NIVEAU, graine: int = 0
) -> Intervalle:
    """Intervalle de la moyenne d'une condition, une valeur par partie."""
    valeurs = np.asarray(valeurs, dtype=np.float64)
    generateur = graines.generateur_analyse(graine)
    indices = generateur.integers(0, len(valeurs), size=(tirages, len(valeurs)))
    moyennes = valeurs[indices].mean(axis=1)
    bas, haut = _percentiles(moyennes, niveau)
    return Intervalle(float(valeurs.mean()), bas, haut)


def difference_appariee(
    valeurs_a: np.ndarray,
    valeurs_b: np.ndarray,
    tirages: int = TIRAGES,
    niveau: float = NIVEAU,
    graine: int = 0,
) -> Intervalle:
    """Intervalle de la différence moyenne A − B, partie par partie.

    Les deux séries doivent avoir joué les mêmes graines, dans le même ordre :
    l'indice `i` désigne la même séquence de pièces des deux côtés. C'est cet
    appariement qui retire du calcul la variabilité due aux séquences elles-mêmes,
    et qui rend l'écart bien plus net qu'une comparaison de deux moyennes
    indépendantes.
    """
    valeurs_a = np.asarray(valeurs_a, dtype=np.float64)
    valeurs_b = np.asarray(valeurs_b, dtype=np.float64)
    if valeurs_a.shape != valeurs_b.shape:
        raise ValueError(
            f"séries non appariées : {valeurs_a.shape} contre {valeurs_b.shape}. "
            "Les deux conditions doivent avoir joué le même protocole."
        )
    ecarts = valeurs_a - valeurs_b
    generateur = graines.generateur_analyse(graine)
    indices = generateur.integers(0, len(ecarts), size=(tirages, len(ecarts)))
    moyennes = ecarts[indices].mean(axis=1)
    bas, haut = _percentiles(moyennes, niveau)
    return Intervalle(float(ecarts.mean()), bas, haut)


def difference_appariee_par_groupes(
    valeurs_a: np.ndarray,
    valeurs_b: np.ndarray,
    groupes: np.ndarray,
    tirages: int = TIRAGES,
    niveau: float = NIVEAU,
    graine: int = 0,
) -> Intervalle:
    """Différence moyenne A − B sur des mesures regroupées, rééchantillonnée par groupe.

    Pour l'accord : une valeur par situation, mais les situations d'une même
    partie ne sont pas indépendantes — une grille abîmée en produit d'autres.
    Rééchantillonner les situations une à une ferait comme si les 6 000 étaient
    indépendantes, et donnerait un intervalle trop étroit. On tire donc des
    **parties entières**, et la différence est recalculée sur toutes leurs
    situations : l'intervalle a la largeur que justifient 20 parties, pas celle
    que suggéreraient 6 000 situations.
    """
    valeurs_a = np.asarray(valeurs_a, dtype=np.float64)
    valeurs_b = np.asarray(valeurs_b, dtype=np.float64)
    groupes = np.asarray(groupes)
    if not (valeurs_a.shape == valeurs_b.shape == groupes.shape):
        raise ValueError("valeurs et groupes doivent avoir la même forme")
    noms, rangs = np.unique(groupes, return_inverse=True)
    sommes = np.bincount(rangs, weights=valeurs_a - valeurs_b, minlength=len(noms))
    tailles = np.bincount(rangs, minlength=len(noms)).astype(np.float64)
    generateur = graines.generateur_analyse(graine)
    tires = generateur.integers(0, len(noms), size=(tirages, len(noms)))
    moyennes = sommes[tires].sum(axis=1) / tailles[tires].sum(axis=1)
    bas, haut = _percentiles(moyennes, niveau)
    return Intervalle(float(sommes.sum() / tailles.sum()), bas, haut)
