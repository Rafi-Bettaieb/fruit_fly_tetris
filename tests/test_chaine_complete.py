"""La chaîne complète : un connectome réduit qui apprend vraiment à Tetris.

C'est la mise au point de l'étape F (§17), faite ici sur 1 200 neurones au lieu
de 165 122. Elle valide tout l'enchaînement — encodage, projection d'entrée
figée, K mises à jour du graphe, lecture motrice, perte softmax, Adam,
rétropropagation à travers les K mises à jour, jeu — **avant** que la même
chaîne ne coûte des heures de GPU sur le connectome complet.

**Pourquoi pas le graphe jouet de 20 neurones.** Il n'a que 4 neurones
sensoriels pour 205 entrées : il ne lit donc que 4 cases de la grille et ne
distingue quasiment aucun candidat. Aucun réglage ne le ferait apprendre, et
échouer là ne prouverait rien. La seule contrainte qui compte pour ce test est
d'avoir **au moins 205 neurones d'entrée** ; `petit_connectome` en a 410.

Ce qui est testé ici n'est pas la performance — 1 200 neurones ne feront pas
195 lignes — mais que **le signal passe et que le gradient revient**.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouche_tetris.connectome import controles as ctrl
from mouche_tetris.connectome.facteur import (
    choisir_facteur_global,
    composition_des_interfaces,
    participation,
)
from mouche_tetris.connectome.graphe import petit_connectome
from mouche_tetris.connectome.modele import Mouche
from mouche_tetris.encodage import encoder_lot
from mouche_tetris.entrainement import clonage
from mouche_tetris.entrainement import demonstrations as d
from mouche_tetris.evaluation import parties as ev
from mouche_tetris.evaluation.accord import accord, niveau_du_hasard
from mouche_tetris.modeles.hasard import Hasard
from mouche_tetris.modeles.torche import NoteurTorch


@pytest.fixture(scope="module")
def reduit():
    """Un connectome réduit : 1 200 neurones, 410 à l'entrée — assez pour lire
    les 205 cases de la grille, ce dont le graphe jouet de 20 est incapable."""
    return petit_connectome()


@pytest.fixture(scope="module")
def situations():
    return d.generer(range(0, 4), plafond=90)


@pytest.fixture(scope="module")
def test_situations():
    return d.generer(range(500, 501), bruit=0.0, plafond=90)


@pytest.fixture(scope="module")
def grilles(situations):
    """Cent grilles candidates tirées des démonstrations, comme en §7.4."""
    tirees = [c for s in situations[:40] for c in s.candidats][:100]
    return torch.from_numpy(encoder_lot(tuple(tirees)))


# --- Facteur global ------------------------------------------------------


def test_le_facteur_global_est_choisi_dans_la_grille(reduit, grilles):
    facteur, diagnostics = choisir_facteur_global(reduit, grilles, k=6)
    assert facteur in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert len(diagnostics) >= 4


def test_le_facteur_ecarte_les_valeurs_saturantes(reduit, grilles):
    """Sans cette règle, la parade « activité qui s'emballe » de §18 ne
    pourrait rien corriger : elle redonnerait la même valeur."""
    facteur, diagnostics = choisir_facteur_global(reduit, grilles, k=6)
    retenu = next((di for di in diagnostics if di.facteur == facteur), None)
    if retenu is not None and retenu.convient:
        assert not retenu.sature


def test_participation_et_composition(reduit, grilles):
    mouche = Mouche(reduit, k=6, facteur_global=4.0)
    part = participation(mouche, grilles)
    assert 0.0 <= part <= 1.0
    composition = composition_des_interfaces(reduit)
    assert set(composition) == {
        "entrees_inhibitrices",
        "sorties_inhibitrices",
        "connexions_inhibitrices",
    }
    assert all(0.0 <= valeur <= 1.0 for valeur in composition.values())


# --- La chaîne complète --------------------------------------------------


@pytest.fixture(scope="module")
def entrainement(reduit, situations, test_situations):
    """Entraîne une fois, mesure plusieurs fois.

    Sans ce partage, chaque test refait les mêmes 400 mises à jour : trente
    secondes chacun pour exactement le même modèle.
    """
    torch.manual_seed(0)
    depart = Mouche(reduit, k=6, facteur_global=4.0, graine_run=0)
    avant = accord(NoteurTorch(depart, "mouche réduite"), test_situations)
    mouche, journal = clonage.entrainer(
        depart,
        situations,
        test_situations,
        clonage.Reglages(lot=16, taux=0.02, mises_a_jour=400, point_de_controle_tous_les=200),
        nom="mouche réduite",
    )
    return mouche, journal, avant


@pytest.mark.lent
def test_une_mouche_reduite_apprend(entrainement, test_situations):
    """Le test le plus important du dépôt avant l'arrivée du MaleCNS.

    Si le signal ne traverse pas le graphe, ou si le gradient ne revient pas à
    travers les K mises à jour, l'accord avec l'expert reste au niveau du
    hasard. C'est le même symptôme que la parade « Pas d'apprentissage » de §18
    — et il se diagnostique ici en quelques secondes plutôt qu'après un run.
    """
    mouche, journal, avant = entrainement
    hasard = niveau_du_hasard(test_situations)
    apres = accord(NoteurTorch(mouche, "mouche réduite"), test_situations)

    assert not journal.echoue
    assert avant == pytest.approx(hasard, abs=0.10), "au départ, le réseau ne sait rien"
    assert apres > avant + 0.05, f"aucun apprentissage : {avant:.1%} → {apres:.1%}"
    assert np.mean(journal.pertes[-50:]) < np.mean(journal.pertes[:50])


@pytest.mark.lent
def test_la_mouche_reduite_joue_mieux_que_le_hasard(entrainement):
    """Elle doit poser des pièces et compléter des lignes, pas seulement noter."""
    mouche, _, _ = entrainement
    court = ev.Protocole("court", range(1500, 1510), 200)
    apprise = ev.serie(NoteurTorch(mouche, "mouche réduite"), court)
    au_hasard = ev.serie(Hasard(), court)
    assert apprise.pieces_moyennes > au_hasard.pieces_moyennes
    assert apprise.lignes.sum() > au_hasard.lignes.sum()


# --- Les témoins, sur la chaîne complète ---------------------------------


def test_le_graphe_coupe_joue_comme_le_hasard(reduit):
    """Le témoin d'intégrité le plus important du projet (§11.2).

    Gains effectifs à 0 : les neurones moteurs, disjoints des neurones d'entrée,
    ne reçoivent plus rien. Toutes les notes sont égales, le départage choisit
    uniformément, et le score doit être **exactement** celui du joueur au
    hasard. S'il fait mieux, un raccourci transporte le signal hors du graphe.
    """
    coupe = ctrl.couper_le_graphe(Mouche(reduit, k=6, facteur_global=4.0))
    court = ev.Protocole("court", range(1500, 1510), 300)
    serie_coupee = ev.serie(NoteurTorch(coupe, "graphe coupé"), court)
    serie_hasard = ev.serie(Hasard(), court)
    assert serie_coupee.lignes.tolist() == serie_hasard.lignes.tolist()


def test_l_entree_aveuglee_joue_comme_le_hasard(reduit):
    aveugle = ctrl.aveugler_l_entree(Mouche(reduit, k=6, facteur_global=4.0))
    court = ev.Protocole("court", range(1500, 1510), 300)
    serie_aveugle = ev.serie(NoteurTorch(aveugle, "entrée aveuglée"), court)
    serie_hasard = ev.serie(Hasard(), court)
    assert serie_aveugle.lignes.tolist() == serie_hasard.lignes.tolist()


@pytest.mark.lent
def test_la_mouche_recablee_suit_la_meme_recette(reduit, situations, test_situations):
    """Le contrôle qui porte la question du projet doit passer par exactement la
    même chaîne — même boucle, mêmes réglages, même budget."""
    recable = ctrl.recabler(reduit, graine=0)
    torch.manual_seed(0)
    mouche, journal = clonage.entrainer(
        Mouche(recable, k=6, facteur_global=4.0, graine_run=0),
        situations,
        test_situations,
        clonage.Reglages(lot=16, taux=0.02, mises_a_jour=200,
                         point_de_controle_tous_les=100),
        nom="recâblée",
    )
    assert not journal.echoue
    assert len(journal.accords) == 2
