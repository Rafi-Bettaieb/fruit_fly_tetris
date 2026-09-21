"""Les 7 tétrominos et leurs rotations distinctes (conception.md §6).

C'est la première brique du projet, et la seule dont l'exactitude se démontre
sans rien exécuter d'autre : sur une grille vide, une pièce de largeur `l` se
pose à `10 − l + 1` colonnes, ce qui donne 9 placements pour le O, 17 pour le I,
le S et le Z, 34 pour le T, le J et le L. Jamais plus de 34.

**Pourquoi éliminer les rotations identiques.** Le O est le même carré dans les
quatre orientations, le I n'a que deux formes. Garder les quatre ferait deux
dégâts : on évaluerait jusqu'à 40 grilles au lieu de 34, pour rien, et surtout
des candidats en double se partageraient la masse de la softmax pendant
l'entraînement (§10.3) — un placement présent quatre fois pèserait quatre fois
moins qu'il ne doit. Les doublons ne sont donc pas une inélégance, ils
fausseraient la perte.

Une pièce est un ensemble de cases `(ligne, colonne)`, ligne 0 en haut, et
chaque rotation est normalisée pour toucher la ligne 0 et la colonne 0.
"""

from __future__ import annotations

from typing import NamedTuple

LETTRES: tuple[str, ...] = ("I", "O", "T", "S", "Z", "J", "L")

Case = tuple[int, int]

# Forme de départ de chaque pièce. L'orientation choisie ici n'a pas
# d'importance : la liste des rotations distinctes est la même quelle que soit
# celle par laquelle on commence.
_FORMES_INITIALES: dict[str, tuple[Case, ...]] = {
    "I": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "O": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "S": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "Z": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "J": ((0, 0), (1, 0), (1, 1), (1, 2)),
    "L": ((0, 2), (1, 0), (1, 1), (1, 2)),
}


class Rotation(NamedTuple):
    """Une orientation d'une pièce, collée en haut à gauche de son cadre."""

    cases: tuple[Case, ...]
    largeur: int
    hauteur: int

    def colonnes_de_la_case_la_plus_basse(self) -> dict[int, int]:
        """Pour chaque colonne occupée, la ligne la plus basse de la pièce.

        C'est ce dont la chute verticale a besoin : la pièce descend jusqu'à ce
        que l'une de ces cases touche la pile ou le fond.
        """
        plus_basse: dict[int, int] = {}
        for ligne, colonne in self.cases:
            if colonne not in plus_basse or ligne > plus_basse[colonne]:
                plus_basse[colonne] = ligne
        return plus_basse


def _normaliser(cases: tuple[Case, ...]) -> tuple[Case, ...]:
    """Colle la forme en haut à gauche et range ses cases dans un ordre stable.

    Sans cet ordre stable, deux rotations identiques auraient deux
    représentations différentes et la déduplication laisserait passer des
    doublons.
    """
    ligne_min = min(ligne for ligne, _ in cases)
    colonne_min = min(colonne for _, colonne in cases)
    return tuple(sorted((ligne - ligne_min, colonne - colonne_min) for ligne, colonne in cases))


def _tourner(cases: tuple[Case, ...]) -> tuple[Case, ...]:
    """Quart de tour dans le sens des aiguilles d'une montre."""
    ligne_max = max(ligne for ligne, _ in cases)
    return _normaliser(tuple((colonne, ligne_max - ligne) for ligne, colonne in cases))


def _rotations_distinctes(forme: tuple[Case, ...]) -> tuple[Rotation, ...]:
    distinctes: list[tuple[Case, ...]] = []
    courante = _normaliser(forme)
    for _ in range(4):
        if courante not in distinctes:
            distinctes.append(courante)
        courante = _tourner(courante)
    return tuple(
        Rotation(
            cases=cases,
            largeur=max(colonne for _, colonne in cases) + 1,
            hauteur=max(ligne for ligne, _ in cases) + 1,
        )
        for cases in distinctes
    )


ROTATIONS: dict[str, tuple[Rotation, ...]] = {
    lettre: _rotations_distinctes(forme) for lettre, forme in _FORMES_INITIALES.items()
}


def rotations(lettre: str) -> tuple[Rotation, ...]:
    """Les orientations distinctes d'une pièce : 1 pour le O, 2 ou 4 pour les autres."""
    return ROTATIONS[lettre]


def placements_sur_grille_vide(lettre: str, colonnes: int = 10) -> int:
    """Nombre de placements d'une pièce sur une grille vide.

    Attendu pour 10 colonnes : O 9 ; I, S, Z 17 ; T, J, L 34 (§6).
    """
    return sum(colonnes - rotation.largeur + 1 for rotation in rotations(lettre))
