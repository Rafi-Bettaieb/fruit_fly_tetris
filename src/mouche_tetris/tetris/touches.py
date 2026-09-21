"""La manette : reconstituer la suite de touches d'un placement (§13.2).

**La mouche n'appuie sur aucun bouton.** Elle note jusqu'à 34 grilles et choisit
la meilleure ; le moteur pose la pièce. Il n'existe donc aucune pression de
bouton à afficher, et la démo en fabrique une **après coup**, pour l'image.

C'est le seul endroit du projet où l'illustration va au-delà de ce que fait le
modèle, et il porte sa mention : la page affiche, à côté de la manette,
« *La mouche note toutes les façons de poser la pièce — jusqu'à 34 — et choisit
la meilleure. La suite de touches est reconstituée après coup, pour l'image.* »
Sans cette phrase, l'animation raconterait une mouche qui joue en temps réel, ce
que le projet ne fait pas et exclut explicitement (§20).

La reconstitution, dans l'ordre :

1. les rotations, dans le sens le plus court — deux appuis au plus ;
2. les déplacements depuis la colonne d'apparition — neuf au plus ;
3. un appui bas pour la chute.

Soit une douzaine d'appuis au plus, allumés à 60 ms, donc moins d'une seconde
pendant que la pièce descend.

**Convention d'apparition.** Le moteur ne fait pas apparaître les pièces : un
candidat désigne directement sa rotation et sa colonne. Pour l'affichage, on
convient qu'une pièce apparaît centrée dans sa rotation finale, et l'on compte
les déplacements à partir de là. C'est une convention d'image, pas une règle du
jeu — le moteur reste seul juge (§5.2).

Rien de tout cela ne touche au moteur, au modèle, à l'entraînement ni aux
mesures. Mais c'est testé comme le reste : rejouer la séquence doit redonner
**exactement** le placement choisi.
"""

from __future__ import annotations

from enum import Enum

from .grille import COLONNES
from .pieces import Case, rotations


class Touche(Enum):
    """Les boutons d'une manette à croix directionnelle et boutons d'action."""

    GAUCHE = "gauche"
    DROITE = "droite"
    ROTATION_HORAIRE = "A"
    ROTATION_ANTIHORAIRE = "B"
    CHUTE = "bas"


def colonne_d_apparition(lettre: str, rotation: int) -> int:
    """La colonne où l'on convient que la pièce apparaît, centrée."""
    largeur = rotations(lettre)[rotation].largeur
    return (COLONNES - largeur) // 2


def reconstituer(lettre: str, rotation: int, colonne: int) -> tuple[Touche, ...]:
    """La suite de touches qui mène au placement choisi."""
    nombre = len(rotations(lettre))
    horaires = rotation % nombre
    antihoraires = (nombre - horaires) % nombre

    if horaires <= antihoraires:
        touches = [Touche.ROTATION_HORAIRE] * horaires
    else:
        touches = [Touche.ROTATION_ANTIHORAIRE] * antihoraires

    ecart = colonne - colonne_d_apparition(lettre, rotation)
    touches += [Touche.DROITE if ecart > 0 else Touche.GAUCHE] * abs(ecart)
    touches.append(Touche.CHUTE)
    return tuple(touches)


def trajectoire(lettre: str, rotation: int, colonne: int) -> tuple[tuple[Case, ...], ...]:
    """Les positions de la pièce en haut de la grille, touche après touche.

    Rend l'apparition, puis la position après chaque touche **sauf la chute** :
    autant de positions que de touches. La page les affiche au rythme des appuis
    — la pièce apparaît, tourne à chaque A ou B, glisse à chaque ◀ ▶ — et la
    chute l'amène aux cases que rend `_cellules_posees`, que seul le moteur sait
    calculer puisqu'elles dépendent de la pile.

    Calculé ici plutôt que dans le navigateur : la forme de chaque rotation est
    une règle du jeu, et le moteur Python en est le seul juge (§5.2).
    """
    formes = rotations(lettre)
    touches = reconstituer(lettre, rotation, colonne)

    def position(r: int, c: int) -> tuple[Case, ...]:
        return tuple((ligne, c + colonne_piece) for ligne, colonne_piece in formes[r].cases)

    courante, colonne_courante = 0, colonne_d_apparition(lettre, 0)
    positions = [position(courante, colonne_courante)]
    for touche in touches[:-1]:
        if touche is Touche.ROTATION_HORAIRE or touche is Touche.ROTATION_ANTIHORAIRE:
            pas = 1 if touche is Touche.ROTATION_HORAIRE else -1
            courante = (courante + pas) % len(formes)
            # Comme dans `rejouer` : la pièce reste centrée pendant qu'elle
            # tourne, et les déplacements se comptent depuis ce centre.
            colonne_courante = colonne_d_apparition(lettre, courante)
        elif touche is Touche.DROITE:
            colonne_courante += 1
        elif touche is Touche.GAUCHE:
            colonne_courante -= 1
        positions.append(position(courante, colonne_courante))
    return tuple(positions)


def rejouer(lettre: str, touches: tuple[Touche, ...]) -> tuple[int, int]:
    """Applique une suite de touches et rend le placement (rotation, colonne).

    L'inverse exact de `reconstituer` : c'est ce qui permet de vérifier que la
    séquence affichée mène bien où la mouche a joué, et pas ailleurs.
    """
    nombre = len(rotations(lettre))
    rotation = 0
    deplacements = 0
    for touche in touches:
        if touche is Touche.ROTATION_HORAIRE:
            rotation = (rotation + 1) % nombre
        elif touche is Touche.ROTATION_ANTIHORAIRE:
            rotation = (rotation - 1) % nombre
        elif touche is Touche.DROITE:
            deplacements += 1
        elif touche is Touche.GAUCHE:
            deplacements -= 1
    return rotation, colonne_d_apparition(lettre, rotation) + deplacements
