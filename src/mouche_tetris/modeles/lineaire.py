"""Le juge de grilles linéaire : la barre à dépasser (conception.md §11.2).

Une combinaison linéaire des 205 entrées, soit **206 paramètres**. Attendu
autour de 13,8 lignes par partie après DAgger.

C'est la seule assurance du projet contre un résultat gênant : si la mouche et
ses 25 728 320 paramètres font moins bien que ces 206-là, il faut pouvoir le
dire, et pour le dire il faut l'avoir mesuré. C'est aussi lui qui valide la
boucle d'entraînement avant qu'elle ne coûte des heures de GPU — il apprend en
quelques secondes de CPU.

Rapport de taille entre les deux modèles : **124 894 pour 1**.
"""

from __future__ import annotations

from torch import nn

from ..encodage import TAILLE_ENCODAGE


class JugeLineaire(nn.Module):
    """Une note par encodage, par combinaison linéaire des 205 entrées."""

    def __init__(self) -> None:
        super().__init__()
        self.couche = nn.Linear(TAILLE_ENCODAGE, 1)

    def forward(self, encodages):
        return self.couche(encodages).squeeze(-1)
