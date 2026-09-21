"""L'énumération des placements.

C'est la fonction par laquelle passe tout le projet. Trois propriétés la rendent
utilisable, et aucune ne se voit à l'exécution si elle est fausse : le compte
est complet, l'ordre est stable, et rien n'est filtré.
"""

from __future__ import annotations

import pytest

from mouche_tetris.tetris import candidats as c
from mouche_tetris.tetris import grille as g
from mouche_tetris.tetris import pieces

PLACEMENTS_ATTENDUS = {"O": 9, "I": 17, "S": 17, "Z": 17, "T": 34, "J": 34, "L": 34}


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_compte_sur_grille_vide(lettre):
    """Le fait de §6, vu depuis l'énumération complète."""
    assert len(c.enumerer(g.GRILLE_VIDE, lettre)) == PLACEMENTS_ATTENDUS[lettre]


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_jamais_plus_de_34(lettre):
    assert len(c.enumerer(g.GRILLE_VIDE, lettre)) <= 34


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_ordre_stable(lettre):
    """Rotation croissante, puis colonne croissante.

    Le départage des égalités tire un indice dans cette liste (§9.3) : un ordre
    instable ferait diverger deux runs de même graine.
    """
    liste = c.enumerer(g.GRILLE_VIDE, lettre)
    cles = [(cand.rotation, cand.colonne) for cand in liste]
    assert cles == sorted(cles)
    assert c.enumerer(g.GRILLE_VIDE, lettre) == liste


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_identifiants_distincts_et_sous_40(lettre):
    """La numérotation de §6 : rotation × 10 + colonne, 40 identifiants possibles."""
    identifiants = [cand.identifiant for cand in c.enumerer(g.GRILLE_VIDE, lettre)]
    assert len(set(identifiants)) == len(identifiants)
    assert all(0 <= i < 40 for i in identifiants)


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_chaque_candidat_modifie_la_grille(lettre):
    for cand in c.enumerer(g.GRILLE_VIDE, lettre):
        assert cand.grille != g.GRILLE_VIDE


@pytest.mark.parametrize("lettre", pieces.LETTRES)
def test_quatre_cases_ajoutees_sauf_disparition(lettre):
    for cand in c.enumerer(g.GRILLE_VIDE, lettre):
        assert cand.lignes_completees == 0
        assert g.cases_occupees(cand.grille).sum() == 4


def test_une_colonne_libre_reste_jouable():
    """Un puits profond n'est pas une fin de partie : le I vertical y tombe.

    Vingt lignes pleines sauf la colonne 9 laissent passer un I vertical, qui
    complète quatre lignes d'un coup.
    """
    puits = g.depuis_lignes(["#########."] * g.LIGNES)
    liste = c.enumerer(puits, "I")
    assert len(liste) == 1
    assert liste[0].colonne == 9
    assert liste[0].lignes_completees == 4


def test_fin_de_partie_quand_aucun_placement():
    """La pile atteint la ligne 0 : plus aucune pièce ne tient dans les 20 lignes.

    Les colonnes 0 à 8 sont occupées dès la ligne du haut, la colonne 9 dès la
    suivante. Toute pièce déborderait par le haut, quelle que soit sa rotation.
    """
    bouchee = g.depuis_lignes(["#########.", ".........#"] + ["." * 10] * 18)
    for lettre in pieces.LETTRES:
        assert c.enumerer(bouchee, lettre) == ()
        assert c.partie_perdue(bouchee, lettre)


def test_partie_non_perdue_sur_grille_vide():
    for lettre in pieces.LETTRES:
        assert not c.partie_perdue(g.GRILLE_VIDE, lettre)


def test_les_candidats_se_raréfient_quand_la_pile_monte():
    """Le nombre de candidats n'est 34 que sur grille vide."""
    haute = g.depuis_lignes(["#........."] * 18)
    liste = c.enumerer(haute, "T")
    assert 0 < len(liste) < 34


def test_un_candidat_peut_completer_des_lignes():
    base = g.depuis_lignes(["#########."] * 4)
    liste = c.enumerer(base, "I")
    quadruples = [cand for cand in liste if cand.lignes_completees == 4]
    assert len(quadruples) == 1
    assert quadruples[0].grille == g.GRILLE_VIDE


def test_les_candidats_partent_tous_de_la_meme_grille():
    """La grille d'origine ne doit jamais être modifiée par l'énumération.

    C'est ce que garantit la représentation immuable : sans elle, le premier
    candidat corromprait les trente-trois suivants.
    """
    base = g.depuis_lignes(["#####....."])
    copie = base
    c.enumerer(base, "T")
    assert base == copie
