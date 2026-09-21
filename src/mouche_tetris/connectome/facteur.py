"""Le facteur global et la participation (conception.md §7.4, §7.6).

Le **facteur global** est un multiplicateur commun à toutes les connexions, fixé
une fois pour toutes avant le premier entraînement. Il ne multiplie que l'entrée
synaptique, jamais l'entrée sensorielle, qui vaut ±1 : il règle donc le poids du
graphe **par rapport à** l'entrée, et non le niveau général d'activité.

La procédure, dans l'ordre :

1. essayer 8, 12, 16, 24, 32, 64 ;
2. pour chaque valeur, présenter 100 grilles candidates pendant K mises à jour ;
3. **écarter** les valeurs qui saturent plus de 20 % des neurones ;
4. parmi celles qui restent, garder la plus petite pour laquelle au moins 80 %
   des neurones moteurs ont une activité supérieure à 0,01.

Le point 3 n'est pas décoratif. Sans lui, la parade « activité qui s'emballe »
de §18 consisterait à refaire une procédure déterministe qui redonnerait
forcément la même valeur : elle ne pourrait rien corriger.

**Mesuré sur le MaleCNS v1.0, K = 7, 100 grilles :**

| Facteur | Moteurs actifs | Saturés | Participation |
|---|---|---|---|
| 8 | **0,0 %** | 0,0 % | 58,0 % |
| 12 | 10,6 % | 0,1 % | 66,5 % |
| 16 | 50,8 % | 0,1 % | 76,6 % |
| **24** | **92,4 %** | 0,2 % | 93,4 % |
| 32 | 97,7 % | 0,2 % | 94,7 % |
| 64 | 99,6 % | 0,2 % | 93,8 % |

**Pourquoi le signal s'éteint sans un grand facteur.** La normalisation fait que
les poids entrants d'un neurone somment à 1 en valeur absolue, mais les états de
ses présynaptiques sont à peu près indépendants : leur somme pondérée vaut donc
environ 1/√degré. Le degré entrant médian étant de 100, cela divise le signal par
10 à chaque saut, soit 1 000 sur les D = 3 sauts qui séparent l'entrée de la
sortie. Un facteur d'environ 10 est nécessaire rien que pour compenser cette
dilution — d'où 24, et non 8.

La **participation** est la part des neurones dont l'activité varie d'une grille
à l'autre. C'est le chiffre honnête derrière « tous les neurones travaillent » :
il est mesuré avant et après l'entraînement, et publié tel quel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graphe import Graphe
from .modele import Mouche

# Grille mesurée sur le connectome complet : à 8, aucun neurone moteur n'est
# actif. La grille {1, 2, 4, 8} du document d'origine ne pouvait pas fonctionner,
# et son repli prolongeait vers le bas — dans la mauvaise direction.
VALEURS = (8.0, 12.0, 16.0, 24.0, 32.0, 64.0)
VALEURS_ETENDUES = (128.0, 256.0)
PAR_DEFAUT = 24.0
MESURE_MALECNS = 24.0
"""Valeur retenue sur le MaleCNS v1.0 avec K = 7 : 92,4 % de neurones moteurs
actifs, 0,2 % de saturés, participation de 93,4 % (§7.4)."""

SEUIL_ACTIVITE = 0.01
SEUIL_SATURATION = 0.99
PART_SATUREE_MAX = 0.20
PART_MOTEURS_ACTIFS = 0.8
"""Le seuil d'origine — « au moins la moitié » — tombait sur une falaise : 16
donne 50,8 % de neurones moteurs actifs, à 0,8 point du critère. Comme la
saturation ne dépasse jamais 0,2 %, elle n'est pas le danger que « la plus
petite valeur » devait écarter : on prend la plus petite à marge confortable."""


@dataclass
class Diagnostic:
    """Ce qu'une valeur candidate donne, pour publication."""

    facteur: float
    part_moteurs_actifs: float
    part_saturee: float

    @property
    def sature(self) -> bool:
        return self.part_saturee > PART_SATUREE_MAX

    @property
    def convient(self) -> bool:
        return not self.sature and self.part_moteurs_actifs >= PART_MOTEURS_ACTIFS


@torch.no_grad()
def diagnostiquer(graphe: Graphe, encodages: torch.Tensor, k: int, facteur: float) -> Diagnostic:
    """Mesure l'activité du réseau à gains initiaux, pour une valeur du facteur."""
    mouche = Mouche(graphe, k=k, facteur_global=facteur)
    etats = mouche.etats(encodages)
    moteurs = etats[:, mouche.sorties]
    return Diagnostic(
        facteur=facteur,
        part_moteurs_actifs=float((moteurs.abs() > SEUIL_ACTIVITE).float().mean()),
        part_saturee=float((etats.abs() > SEUIL_SATURATION).float().mean()),
    )


def choisir_facteur_global(
    graphe: Graphe, encodages: torch.Tensor, k: int
) -> tuple[float, list[Diagnostic]]:
    """Applique la procédure de §7.4 et rend la valeur retenue avec ses mesures."""
    diagnostics = [diagnostiquer(graphe, encodages, k, valeur) for valeur in VALEURS]
    for diagnostic in diagnostics:
        if diagnostic.convient:
            return diagnostic.facteur, diagnostics

    for valeur in VALEURS_ETENDUES:
        diagnostic = diagnostiquer(graphe, encodages, k, valeur)
        diagnostics.append(diagnostic)
        if diagnostic.convient:
            return diagnostic.facteur, diagnostics

    return PAR_DEFAUT, diagnostics


@torch.no_grad()
def participation(mouche: Mouche, encodages: torch.Tensor) -> float:
    """Part des neurones dont l'écart-type d'activité dépasse 0,01 (§7.6).

    Mesuré sur des grilles candidates tirées des démonstrations. Un neurone qui
    ne varie pas d'une grille à l'autre ne participe pas à la décision, même
    s'il est actif : c'est la variation qui porte l'information.
    """
    etats = mouche.etats(encodages)
    return float((etats.std(dim=0) > SEUIL_ACTIVITE).float().mean())


def composition_des_interfaces(graphe: Graphe) -> dict[str, float]:
    """Part d'inhibiteurs à l'entrée et à la sortie (§7.6).

    Ce chiffre n'est pas neutre : si les neurones d'entrée retenus sont des
    photorécepteurs, la convention de §7.3 les rend **tous** inhibiteurs, et la
    totalité du signal entre dans le graphe par de l'inhibition. Cela ne change
    aucun réglage, mais c'est nécessaire pour lire les résultats.
    """
    return {
        "entrees_inhibitrices": float(np.mean(graphe.signes[graphe.entrees] < 0)),
        "sorties_inhibitrices": float(np.mean(graphe.signes[graphe.sorties] < 0)),
        "connexions_inhibitrices": graphe.part_inhibitrice,
    }
