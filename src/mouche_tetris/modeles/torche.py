"""Le pont entre un modèle PyTorch et l'interface `Noteur` (conception.md §11.2).

Un modèle entraînable est un `nn.Module` qui prend des encodages et rend une
note par encodage. Pour jouer, il faut un `Noteur` : c'est ce que fait
l'adaptateur ci-dessous, sans gradient et en NumPy.

**Le même module sert aux deux usages.** Un modèle entraîné puis adapté joue
exactement ce qu'il a appris : il n'y a pas un chemin d'entraînement et un
chemin de jeu qui pourraient diverger. La mouche passera par le même adaptateur
que le juge linéaire.

**Un modèle ne lit du candidat que son encodage** (§9.3). Il reçoit un tenseur
de 205 réels, jamais la rotation, la colonne ou la pièce d'origine.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..encodage import TAILLE_ENCODAGE, encoder_lot


class NoteurTorch:
    """Fait jouer un `nn.Module`. Implémente `Noteur`."""

    def __init__(self, modele: nn.Module, nom: str, peripherique: str = "cpu") -> None:
        self.modele = modele.to(peripherique).eval()
        self.nom = nom
        self.peripherique = peripherique

    @torch.no_grad()
    def noter(self, candidats) -> np.ndarray:
        encodages = torch.from_numpy(encoder_lot(candidats)).to(self.peripherique)
        # `reshape(-1)` et non `squeeze` : avec un seul candidat, un squeeze
        # supprimerait aussi la dimension du lot et rendrait un scalaire.
        return self.modele(encodages).reshape(-1).cpu().numpy().astype(np.float64)


def verifier_forme(modele: nn.Module) -> None:
    """Un modèle doit rendre une note par encodage, et une seule.

    Se vérifie sur un lot factice : (n, 205) doit donner (n,) ou (n, 1). Une
    erreur de forme ici se traduirait plus tard par un message obscur au milieu
    d'un entraînement de plusieurs heures.
    """
    factice = torch.zeros(3, TAILLE_ENCODAGE)
    sortie = modele(factice)
    if sortie.shape not in ((3,), (3, 1)):
        raise ValueError(
            f"{type(modele).__name__} rend {tuple(sortie.shape)}, attendu (3,) ou (3, 1)"
        )
