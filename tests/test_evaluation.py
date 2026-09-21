"""Séries appariées, bootstrap, rapport.

Le test qui compte ici est `test_l_appariement_resserre_l_intervalle` : il
montre *pourquoi* le projet apparie, et pas seulement qu'il le fait.
"""

from __future__ import annotations

import numpy as np
import pytest

from mouche_tetris.evaluation import bootstrap as b
from mouche_tetris.evaluation import parties as ev
from mouche_tetris.evaluation import rapport
from mouche_tetris.expert.criteres import Expert
from mouche_tetris.modeles.hasard import Hasard

# --- Protocoles ----------------------------------------------------------


def test_les_deux_protocoles_sont_disjoints():
    """Aucune décision ne se prend sur les parties qui serviront à publier."""
    assert set(ev.DEVELOPPEMENT.graines) & set(ev.FINAL.graines) == set()
    assert ev.DEVELOPPEMENT.parties == 30
    assert ev.FINAL.parties == 100
    assert ev.DEVELOPPEMENT.plafond == 300
    assert ev.FINAL.plafond == 500


def test_une_serie_joue_toutes_les_graines_du_protocole():
    court = ev.Protocole("court", range(1500, 1505), 40)
    resultat = ev.serie(Expert(), court)
    assert len(resultat.parties) == 5
    assert [p.graine for p in resultat.parties] == list(range(1500, 1505))
    assert resultat.part_plafond == 1.0
    assert resultat.pieces_moyennes == 40


# --- Bootstrap -----------------------------------------------------------


def test_intervalle_encadre_la_moyenne():
    valeurs = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0, 10.0])
    resultat = b.intervalle(valeurs, tirages=2000)
    assert resultat.bas < resultat.valeur < resultat.haut
    assert resultat.valeur == pytest.approx(valeurs.mean())


def test_intervalle_reproductible():
    """Un intervalle publié doit pouvoir être retrouvé au chiffre près."""
    valeurs = np.arange(30, dtype=float)
    assert b.intervalle(valeurs) == b.intervalle(valeurs)


def test_intervalle_se_resserre_avec_les_parties():
    generateur = np.random.default_rng(0)
    petit = b.intervalle(generateur.normal(50, 10, 20))
    grand = b.intervalle(generateur.normal(50, 10, 2000))
    assert (grand.haut - grand.bas) < (petit.haut - petit.bas)


def test_difference_refuse_des_series_non_appariees():
    with pytest.raises(ValueError, match="appariées"):
        b.difference_appariee(np.zeros(10), np.zeros(9))


def test_l_appariement_resserre_l_intervalle():
    """Pourquoi le projet apparie, et pas seulement qu'il le fait.

    Deux conditions jouent les mêmes séquences : une séquence facile les avantage
    toutes les deux. L'appariement retire cette variabilité commune du calcul,
    et l'écart devient bien plus net qu'en comparant deux moyennes séparément.
    """
    generateur = np.random.default_rng(0)
    difficulte = generateur.normal(50, 15, 100)  # commune aux deux conditions
    a = difficulte + generateur.normal(3, 1, 100)
    ref = difficulte + generateur.normal(0, 1, 100)

    apparie = b.difference_appariee(a, ref)
    largeur_appariee = apparie.haut - apparie.bas
    largeur_separee = (b.intervalle(a).haut - b.intervalle(a).bas) + (
        b.intervalle(ref).haut - b.intervalle(ref).bas
    )
    assert largeur_appariee < largeur_separee / 3
    assert apparie.exclut_zero


def test_exclut_zero():
    assert b.Intervalle(5.0, 1.0, 9.0).exclut_zero
    assert b.Intervalle(-5.0, -9.0, -1.0).exclut_zero
    assert not b.Intervalle(0.5, -2.0, 3.0).exclut_zero


# --- Un vrai écart, mesuré -----------------------------------------------


def test_l_expert_bat_le_hasard_avec_un_intervalle_qui_exclut_zero():
    court = ev.Protocole("court", range(1500, 1515), 120)
    expert = ev.serie(Expert(), court)
    hasard = ev.serie(Hasard(), court)
    ecart = b.difference_appariee(expert.lignes, hasard.lignes)
    assert ecart.exclut_zero
    assert ecart.valeur > 10


def test_le_rapport_produit_un_tableau_markdown():
    court = ev.Protocole("court", range(1500, 1503), 60)
    series = [ev.serie(Hasard(), court), ev.serie(Expert(), court)]
    texte = rapport.tableau(series, budgets={"expert": "aucun — heuristique"})
    assert texte.startswith("| Condition |")
    assert "hasard" in texte and "expert" in texte
    assert "aucun — heuristique" in texte
    assert len(texte.splitlines()) == 4  # entête, séparateur, deux conditions

    ecarts = rapport.ecarts_apparies(series[0], [series[1]])
    assert "expert − hasard" in ecarts


# --- Écart d'accord, apparié et rééchantillonné par parties ---------------


def test_par_groupes_singletons_egale_l_appariement_simple():
    """Une situation par partie : les deux méthodes tirent les mêmes indices
    et doivent rendre exactement le même intervalle."""
    from mouche_tetris import graines

    generateur = graines.generateur_analyse(7)
    a = generateur.random(200) < 0.55
    b_ = generateur.random(200) < 0.50
    simple = b.difference_appariee(a, b_)
    groupes = b.difference_appariee_par_groupes(a, b_, np.arange(200))
    assert groupes == pytest.approx(simple)


def test_par_groupes_la_valeur_est_l_ecart_sur_toutes_les_situations():
    a = np.array([1, 1, 1, 0, 0, 1], dtype=float)
    b_ = np.array([0, 1, 0, 0, 0, 1], dtype=float)
    groupes = np.array([0, 0, 0, 1, 1, 1])
    assert b.difference_appariee_par_groupes(a, b_, groupes).valeur == pytest.approx(2 / 6)


def test_des_situations_liees_elargissent_l_intervalle():
    """Le cas réel : les situations d'une partie se ressemblent. Les traiter
    comme indépendantes donnerait un intervalle faussement étroit."""
    from mouche_tetris import graines

    generateur = graines.generateur_analyse(3)
    # 20 parties de 300 situations ; l'écart dépend surtout de la partie.
    par_partie = generateur.normal(0.03, 0.08, size=20)
    ecarts = np.repeat(par_partie, 300) + generateur.normal(0, 0.01, size=6000)
    groupes = np.repeat(np.arange(20), 300)
    zeros = np.zeros_like(ecarts)
    independant = b.difference_appariee(ecarts, zeros)
    par_parties = b.difference_appariee_par_groupes(ecarts, zeros, groupes)
    assert (par_parties.haut - par_parties.bas) > 3 * (independant.haut - independant.bas)


def test_les_reussites_redonnent_l_accord():
    from mouche_tetris.entrainement import demonstrations as d
    from mouche_tetris.evaluation.accord import accord, reussites

    situations = d.generer(range(500, 502), bruit=0.0, plafond=60)
    justes = reussites(Hasard(), situations)
    assert justes.dtype == bool and len(justes) == len(situations)
    assert justes.mean() == pytest.approx(accord(Hasard(), situations))
    assert reussites(Expert(), situations).all()
