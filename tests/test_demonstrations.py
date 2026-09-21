"""Les démonstrations de l'expert.

Deux propriétés sont critiques et silencieuses : l'étiquette reste **toujours**
le choix de l'expert même quand un coup au hasard est joué, et le bruit ne doit
pas décaler la séquence de pièces.
"""

from __future__ import annotations

from mouche_tetris import graines
from mouche_tetris.entrainement import demonstrations as d
from mouche_tetris.tetris.sacs import premieres


def test_une_partie_produit_des_situations():
    situations = d.generer_partie(0, plafond=50)
    assert 0 < len(situations) <= 50
    for situation in situations:
        assert len(situation.candidats) > 0
        assert 0 <= situation.choix_expert < len(situation.candidats)
        assert 0 <= situation.joue < len(situation.candidats)


def test_le_bruit_ne_decale_pas_la_sequence_de_pieces():
    """Le bruit tire sur le flux du run, jamais sur celui des pièces (§10.5)."""
    avec = d.generer_partie(0, bruit=0.5, plafond=40)
    sans = d.generer_partie(0, bruit=0.0, plafond=40)
    commun = min(len(avec), len(sans))
    assert [s.lettre for s in avec[:commun]] == [s.lettre for s in sans[:commun]]
    assert [s.lettre for s in sans[:commun]] == list(premieres(0, commun))


def test_sans_bruit_l_expert_joue_toujours_son_choix():
    for situation in d.generer_partie(0, bruit=0.0, plafond=60):
        assert situation.joue == situation.choix_expert
        assert not situation.bruite


def test_le_bruit_change_le_coup_joue_pas_l_etiquette():
    """Le cœur de §10.2 : on enregistre la situation dégradée **et** la bonne réponse."""
    situations = d.generer_partie(0, bruit=1.0, plafond=60)
    bruitees = [s for s in situations if s.bruite]
    assert bruitees, "avec un bruit de 100 %, des coups doivent différer"
    for situation in bruitees:
        # L'étiquette reste ce que l'expert aurait joué, pas ce qui a été joué :
        # c'est toute l'astuce, et sans elle le bruit n'apprendrait rien.
        assert situation.choix_expert != situation.joue
        attendu = situation.candidats[situation.choix_expert]
        assert attendu is not situation.candidats[situation.joue]


def test_taux_de_bruit_proche_de_5_pour_cent():
    situations = d.generer(range(0, 12), bruit=d.BRUIT, plafond=200)
    part = sum(s.bruite for s in situations) / len(situations)
    # Un coup au hasard peut retomber sur le choix de l'expert : la part observée
    # est donc légèrement sous les 5 % tirés.
    assert 0.02 < part < 0.06


def test_les_parties_bruitees_durent_moins_longtemps():
    sans = len(d.generer_partie(0, bruit=0.0, plafond=300))
    avec = len(d.generer_partie(0, bruit=0.3, plafond=300))
    assert avec < sans


def test_reproductible():
    a = d.generer_partie(3, plafond=80)
    b = d.generer_partie(3, plafond=80)
    assert [(s.lettre, s.choix_expert, s.joue) for s in a] == [
        (s.lettre, s.choix_expert, s.joue) for s in b
    ]


def test_jeu_de_test_sans_bruit_et_hors_entrainement():
    assert set(graines.TEST) & set(graines.DEMONSTRATIONS) == set()
    situations = d.generer(range(500, 502), bruit=0.0)
    assert all(not s.bruite for s in situations)


def test_echantillon_de_test_fixe():
    situations = d.generer(range(500, 502), bruit=0.0)
    a = d.echantillon_de_test(situations, 50)
    b = d.echantillon_de_test(situations, 50)
    assert [s.grille_avant for s in a] == [s.grille_avant for s in b]
    assert len(a) == 50


# --- Tirage des candidats pour la perte -----------------------------------


def test_le_tirage_garde_toujours_la_cible():
    situations = d.generer_partie(0, plafond=30)
    generateur = graines.generateur_run(0)
    for situation in situations:
        tires, cible = d.tirer_candidats(situation, 10, generateur)
        assert len(tires) == min(10, len(situation.candidats))
        assert tires[cible] is situation.candidats[situation.choix_expert]


def test_le_tirage_melange_la_position_de_la_cible():
    """Sans mélange, un modèle apprendrait la position au lieu de la grille."""
    situations = [s for s in d.generer_partie(0, plafond=120) if len(s.candidats) > 10]
    generateur = graines.generateur_run(0)
    positions = {d.tirer_candidats(s, 10, generateur)[1] for s in situations}
    assert len(positions) > 3


def test_tirage_quand_il_y_a_moins_de_candidats_que_demande():
    situations = d.generer_partie(0, plafond=30)
    generateur = graines.generateur_run(0)
    situation = min(situations, key=lambda s: len(s.candidats))
    tires, cible = d.tirer_candidats(situation, 100, generateur)
    assert len(tires) == len(situation.candidats)
    assert tires[cible] is situation.candidats[situation.choix_expert]
