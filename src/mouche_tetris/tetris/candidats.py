"""L'énumération des placements : la fonction centrale du moteur (conception.md §6).

Pour une grille et une pièce, tous les placements valides, chacun accompagné de
sa grille résultante et du nombre de lignes qu'il fait disparaître. Sur grille
vide : 9 pour le O, 17 pour le I, le S et le Z, 34 pour le T, le J et le L.

**Tout le projet passe par ici.** Les démonstrations de l'expert, les parties de
DAgger, l'évaluation appariée, les enregistrements et le serveur de diffusion
appellent tous cette fonction. La mouche, elle, ne fait que noter ce qu'elle rend.

**Aucun filtrage, jamais.** C'est la règle absolue du projet (§3, §14.5) : la
liste rendue ici est complète. Si l'expert la réduisait, c'est lui qui jouerait,
et la mesure ne voudrait plus rien dire. Les parades mémoire de §14.5 réduisent
le nombre de candidats **échantillonnés pour la perte**, jamais le nombre de
candidats énumérés.

**L'ordre est stable** : rotation croissante, puis colonne croissante. Le
départage des égalités tire un indice dans cette liste (§9.3) ; si l'ordre
changeait d'une exécution à l'autre, deux runs de même graine divergeraient.
"""

from __future__ import annotations

from typing import NamedTuple

from . import pieces
from .grille import COLONNES, Grille, poser, sommets


class Candidat(NamedTuple):
    """Une façon de poser la pièce courante, et ce qu'elle donne."""

    rotation: int
    """Indice dans `pieces.rotations(lettre)` — 0 à 3 selon la pièce."""

    colonne: int
    """Colonne de la case la plus à gauche de la pièce."""

    grille: Grille
    """La grille après la pose **et** après disparition des lignes pleines."""

    lignes_completees: int
    """Entre 0 et 4. Sans ce nombre, la grille ci-dessus ne dit plus combien de
    lignes ont disparu : c'est pourquoi l'encodage a 205 entrées et non 200."""

    @property
    def identifiant(self) -> int:
        """Numérotation de §6 : rotation × 10 + colonne, donc 0 à 39.

        Sert aux élèves à choix direct, écrits une seule fois pour mesurer
        l'écart qui justifie la formulation « juge de grilles » (§12).
        """
        return self.rotation * COLONNES + self.colonne


def enumerer(grille: Grille, lettre: str) -> tuple[Candidat, ...]:
    """Tous les placements valides de `lettre` sur `grille`.

    Rend un tuple vide s'il n'y en a aucun : la pièce ne tient plus dans les
    20 lignes, et la partie est finie (§5.2).
    """
    plafonds = sommets(grille)
    trouves = []
    for indice_rotation, rotation in enumerate(pieces.rotations(lettre)):
        for colonne in range(COLONNES - rotation.largeur + 1):
            resultat = poser(grille, rotation, colonne, plafonds)
            if resultat is None:
                continue
            resultante, completees = resultat
            trouves.append(Candidat(indice_rotation, colonne, resultante, completees))
    return tuple(trouves)


def partie_perdue(grille: Grille, lettre: str) -> bool:
    """Vrai si la pièce ne peut être posée nulle part."""
    return not enumerer(grille, lettre)
