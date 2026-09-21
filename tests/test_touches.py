"""La reconstitution de la séquence de touches (§13.2, §16).

Le test qui compte est l'aller-retour : rejouer la séquence affichée doit
redonner **exactement** le placement choisi, pour les 34 candidats de chaque
pièce. Une séquence qui mènerait ailleurs ferait mentir la démo sur ce que la
mouche a joué — et c'est précisément ce qu'on ne veut pas ici.
"""

from __future__ import annotations

import pytest

from mouche_tetris.tetris import candidats as c
from mouche_tetris.tetris import grille as g
from mouche_tetris.tetris import pieces
from mouche_tetris.tetris.touches import (
    Touche,
    colonne_d_apparition,
    reconstituer,
    rejouer,
    trajectoire,
)


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_aller_retour_sur_tous_les_candidats(lettre):
    """Le test prescrit en §16, pour les 34 candidats de chaque pièce."""
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        touches = reconstituer(lettre, candidat.rotation, candidat.colonne)
        assert rejouer(lettre, touches) == (candidat.rotation, candidat.colonne)


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_toute_sequence_finit_par_la_chute(lettre):
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        touches = reconstituer(lettre, candidat.rotation, candidat.colonne)
        assert touches[-1] is Touche.CHUTE
        assert touches.count(Touche.CHUTE) == 1


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_au_plus_deux_rotations(lettre):
    """Le sens le plus court : jamais trois appuis là où un suffit."""
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        touches = reconstituer(lettre, candidat.rotation, candidat.colonne)
        rotations = sum(
            1 for t in touches if t in (Touche.ROTATION_HORAIRE, Touche.ROTATION_ANTIHORAIRE)
        )
        assert rotations <= 2


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_une_douzaine_d_appuis_au_plus(lettre):
    """Douze appuis à 60 ms font moins d'une seconde : la pièce a le temps de
    descendre pendant que la manette s'allume."""
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        assert len(reconstituer(lettre, candidat.rotation, candidat.colonne)) <= 12


def test_le_o_n_a_jamais_besoin_de_tourner():
    """Une seule rotation distincte : aucun appui de rotation n'a de sens."""
    for candidat in c.enumerer(g.GRILLE_VIDE, "O"):
        touches = reconstituer("O", candidat.rotation, candidat.colonne)
        assert Touche.ROTATION_HORAIRE not in touches
        assert Touche.ROTATION_ANTIHORAIRE not in touches


def test_le_i_tourne_en_un_seul_appui():
    """Deux rotations distinctes : horaire et antihoraire mènent au même endroit,
    donc un appui suffit toujours."""
    for candidat in c.enumerer(g.GRILLE_VIDE, "I"):
        touches = reconstituer("I", candidat.rotation, candidat.colonne)
        rotations = sum(
            1 for t in touches if t in (Touche.ROTATION_HORAIRE, Touche.ROTATION_ANTIHORAIRE)
        )
        assert rotations == (1 if candidat.rotation else 0)


def test_le_sens_le_plus_court_est_choisi():
    """Pour une pièce à 4 rotations, aller à la rotation 3 se fait en un appui
    antihoraire, jamais en trois horaires."""
    touches = reconstituer("T", 3, colonne_d_apparition("T", 3))
    assert touches.count(Touche.ROTATION_ANTIHORAIRE) == 1
    assert Touche.ROTATION_HORAIRE not in touches


def test_aucun_deplacement_a_la_colonne_d_apparition():
    for lettre in pieces.LETTRES:
        for rotation in range(len(pieces.rotations(lettre))):
            colonne = colonne_d_apparition(lettre, rotation)
            touches = reconstituer(lettre, rotation, colonne)
            assert Touche.GAUCHE not in touches
            assert Touche.DROITE not in touches


def test_les_deplacements_vont_dans_le_bon_sens():
    depart = colonne_d_apparition("T", 0)
    assert Touche.DROITE in reconstituer("T", 0, depart + 2)
    assert Touche.GAUCHE in reconstituer("T", 0, depart - 2)


# --- La trajectoire affichée ---------------------------------------------


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_une_position_par_touche(lettre):
    """L'apparition, puis une position par touche hors chute : la page en
    affiche une à chaque appui, la chute menant aux cases posées."""
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        touches = reconstituer(lettre, candidat.rotation, candidat.colonne)
        assert len(trajectoire(lettre, candidat.rotation, candidat.colonne)) == len(touches)


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_la_trajectoire_mene_au_placement_choisi(lettre):
    """La dernière position, laissée tomber tout droit, est la pièce posée.

    Sinon la page montrerait la pièce tourner et glisser vers un endroit, puis
    atterrir ailleurs — exactement le mensonge que l'aller-retour interdit.
    """
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        derniere = trajectoire(lettre, candidat.rotation, candidat.colonne)[-1]
        forme = pieces.rotations(lettre)[candidat.rotation]
        assert set(derniere) == {(ligne, candidat.colonne + col) for ligne, col in forme.cases}


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_la_piece_apparait_non_tournee_et_centree(lettre):
    premiere = trajectoire(lettre, 0, colonne_d_apparition(lettre, 0))[0]
    forme = pieces.rotations(lettre)[0]
    depart = colonne_d_apparition(lettre, 0)
    assert set(premiere) == {(ligne, depart + col) for ligne, col in forme.cases}


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_la_trajectoire_reste_dans_la_grille(lettre):
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        for position in trajectoire(lettre, candidat.rotation, candidat.colonne):
            assert all(0 <= col < g.COLONNES for _, col in position)


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_chaque_appui_fait_exactement_ce_qu_il_annonce(lettre):
    """A ou B change la forme ; ◀ ou ▶ décale d'une colonne sans la changer."""
    for candidat in c.enumerer(g.GRILLE_VIDE, lettre):
        touches = reconstituer(lettre, candidat.rotation, candidat.colonne)
        positions = trajectoire(lettre, candidat.rotation, candidat.colonne)
        for touche, avant, apres in zip(touches[:-1], positions[:-1], positions[1:],
                                        strict=True):
            if touche is Touche.DROITE:
                assert set(apres) == {(ligne, col + 1) for ligne, col in avant}
            elif touche is Touche.GAUCHE:
                assert set(apres) == {(ligne, col - 1) for ligne, col in avant}
            else:
                assert touche in (Touche.ROTATION_HORAIRE, Touche.ROTATION_ANTIHORAIRE)
                assert lettre != "O"
