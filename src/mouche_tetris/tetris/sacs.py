"""Le tirage des pièces par sacs de 7 (conception.md §5.1, §5.2).

Chaque sac contient une pièce de chaque type, dans un ordre tiré au sort. C'est
le standard du jeu, et il a une propriété utile ici : la fréquence des pièces ne
dérive pas. Sur 500 pièces, toutes les conditions reçoivent exactement le même
nombre de I, de O et de T — l'écart entre deux joueurs ne peut donc pas venir
d'un tirage plus généreux.

**La séquence ne dépend que du numéro de partie.** Elle est tirée sur le flux
`graines.generateur_pieces`, jamais sur celui du départage des égalités : deux
joueurs de talents différents voient rigoureusement la même suite de pièces,
ce dont dépend toute l'évaluation appariée (§11.1).
"""

from __future__ import annotations

from collections.abc import Iterator

from .. import graines
from . import pieces


def sequence(graine_partie: int) -> Iterator[str]:
    """La suite infinie des pièces d'une partie, sac de 7 après sac de 7."""
    generateur = graines.generateur_pieces(graine_partie)
    nombre = len(pieces.LETTRES)
    while True:
        for indice in generateur.permutation(nombre):
            yield pieces.LETTRES[indice]


def premieres(graine_partie: int, combien: int) -> tuple[str, ...]:
    """Les `combien` premières pièces d'une partie, pour les tests et la démo."""
    flux = sequence(graine_partie)
    return tuple(next(flux) for _ in range(combien))
