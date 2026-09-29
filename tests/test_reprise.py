"""Un entraînement découpé en séances doit donner exactement le même modèle.

C'est la seule chose qui compte pour ce module (§14.6). Une reprise approximative
ne lève aucune erreur : elle produit un modèle médiocre qu'on croit entraîné
pendant 2 400 pas, et l'on publierait un budget faux à côté d'un score vrai.

Ces tests tournent sur le juge linéaire, sur CPU, en quelques secondes — le même
raisonnement que pour la boucle de clonage : on valide avant que cela ne coûte
des heures de GPU.
"""

from __future__ import annotations

import pytest
import torch

from mouche_tetris.entrainement import clonage, reprise
from mouche_tetris.entrainement import demonstrations as d
from mouche_tetris.modeles.lineaire import JugeLineaire
from mouche_tetris.modeles.torche import NoteurTorch

REGLAGES = clonage.Reglages(lot=8, taux=0.01, mises_a_jour=40, point_de_controle_tous_les=10)


@pytest.fixture(scope="module")
def situations():
    return d.generer(range(0, 4), plafond=80)


@pytest.fixture(scope="module")
def test_situations():
    return d.generer(range(500, 501), bruit=0.0, plafond=80)


def _modele() -> JugeLineaire:
    torch.manual_seed(0)
    return JugeLineaire()


def _entrainer(modele, situations, test_situations, **extra):
    return clonage.entrainer(
        modele, situations, test_situations, REGLAGES, nom="juge", bavard=False, **extra
    )


def _poids(modele) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in modele.parameters()])


# --- L'exactitude de la reprise ------------------------------------------


@pytest.fixture(scope="module")
def une_traite(situations, test_situations):
    modele, journal = _entrainer(_modele(), situations, test_situations)
    return _poids(modele), list(journal.pertes)


@pytest.fixture(scope="module")
def en_deux_seances(situations, test_situations):
    """Première séance arrêtée au premier point de contrôle, puis reprise."""
    session = reprise.Session(budget_clonage=REGLAGES.mises_a_jour)
    # Une limite nulle : la boucle s'arrête au premier point de contrôle, qui est
    # le plus tôt où elle a le droit de s'arrêter.
    _entrainer(_modele(), situations, test_situations,
               session=session, limite_secondes=1e-9)
    premiere = session.pas

    modele, journal = _entrainer(_modele(), situations, test_situations, session=session)
    return _poids(modele), list(journal.pertes), premiere


def test_la_premiere_seance_s_arrete_a_un_point_de_controle(en_deux_seances):
    _, _, premiere = en_deux_seances
    assert premiere == REGLAGES.point_de_controle_tous_les
    assert premiere < REGLAGES.mises_a_jour


def test_deux_seances_donnent_exactement_le_meme_modele(une_traite, en_deux_seances):
    """Le test principal du module : bit pour bit, pas « à peu près »."""
    attendus, _ = une_traite
    obtenus, _, _ = en_deux_seances
    assert torch.equal(obtenus, attendus)


def test_la_trajectoire_entiere_est_la_meme(une_traite, en_deux_seances):
    """Comparer les seuls poids finaux laisserait passer deux trajectoires
    différentes qui se rejoignent ; on compare donc les 40 pertes."""
    _, attendues = une_traite
    _, obtenues, _ = en_deux_seances
    assert obtenues == attendues


def test_le_nombre_de_pas_est_bien_celui_du_budget(en_deux_seances):
    _, pertes, _ = en_deux_seances
    assert len(pertes) == REGLAGES.mises_a_jour


def test_sans_l_etat_du_generateur_la_reprise_est_fausse(
    situations, test_situations, une_traite
):
    """Ce que le test précédent démontre, et qui ne se voit pas autrement.

    Une reprise « évidente » — recharger les paramètres et repartir — rejoue
    exactement les lots du début du run. Elle ne lève rien, elle ne plante pas,
    et le modèle obtenu n'est pas celui qu'on croit.
    """
    session = reprise.Session(budget_clonage=REGLAGES.mises_a_jour)
    _entrainer(_modele(), situations, test_situations,
               session=session, limite_secondes=1e-9)

    naive = reprise.Session(budget_clonage=REGLAGES.mises_a_jour)
    naive.parametres = session.parametres  # les poids, et rien d'autre
    _, journal = _entrainer(_modele(), situations, test_situations, session=naive)

    _, attendues = une_traite
    assert len(journal.pertes) == REGLAGES.mises_a_jour
    assert journal.pertes != attendues


# --- Le fichier de séance ------------------------------------------------


def test_l_aller_retour_par_le_disque_ne_perd_rien(
    situations, test_situations, une_traite, tmp_path
):
    session = reprise.Session(budget_clonage=REGLAGES.mises_a_jour)
    _entrainer(_modele(), situations, test_situations,
               session=session, limite_secondes=1e-9)
    reprise.sauvegarder(session, tmp_path / "seance.pt")
    relue = reprise.charger(tmp_path / "seance.pt")

    modele, journal = _entrainer(_modele(), situations, test_situations, session=relue)
    attendus, attendues = une_traite
    assert torch.equal(_poids(modele), attendus)
    assert journal.pertes == attendues


class _Plantage(Exception):
    pass


def test_un_plantage_ne_coute_que_la_fenetre_en_cours(
    situations, test_situations, une_traite, tmp_path
):
    """L'état part sur le disque à chaque point de contrôle, pas en fin de séance.

    Sans cela, une coupure de courant à la troisième heure d'une séance de
    trois heures perdrait la séance entière — exactement ce que le découpage
    devait éviter.
    """
    chemin = tmp_path / "seance.pt"

    def sauver_puis_planter(session):
        reprise.sauvegarder(session, chemin)
        if session.pas == 20:
            raise _Plantage

    with pytest.raises(_Plantage):
        _entrainer(_modele(), situations, test_situations,
                   session=reprise.Session(budget_clonage=REGLAGES.mises_a_jour),
                   sauvegarde=sauver_puis_planter)

    relue = reprise.charger(chemin)
    assert relue.pas == 20
    modele, journal = _entrainer(_modele(), situations, test_situations, session=relue)
    attendus, attendues = une_traite
    assert torch.equal(_poids(modele), attendus)
    assert journal.pertes == attendues


def test_un_pas_bloque_arrete_le_processus_avec_sa_pile(tmp_path):
    """Le chien de garde, dans un processus à part puisqu'il le tue.

    Simule le cas réel : un appel qui ne rend jamais la main, comme une attente
    du GPU après une mise en veille. Sans garde, le processus tournerait à vide
    pour toujours ; avec, il s'arrête et dit où il était bloqué.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import time
        from mouche_tetris.entrainement import clonage
        clonage.armer_la_garde(1)
        def attente_du_gpu_qui_ne_viendra_jamais():
            while True:
                time.sleep(0.05)
        attente_du_gpu_qui_ne_viendra_jamais()
    """)
    resultat = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=60)
    assert resultat.returncode != 0
    assert "attente_du_gpu_qui_ne_viendra_jamais" in resultat.stderr


def test_la_garde_est_levee_meme_sur_exception():
    """Oubliée armée, elle tuerait un quart d'heure plus tard le processus qui
    a rattrapé l'exception — la suite de tests, par exemple.

    Une garde d'une seconde, une exception pendant l'entraînement, puis trois
    secondes d'attente : si la garde avait survécu, le processus serait mort.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import time
        from mouche_tetris.entrainement import clonage, reprise
        from mouche_tetris.entrainement import demonstrations as d
        from mouche_tetris.modeles.lineaire import JugeLineaire
        clonage.GARDE_SECONDES = 1
        situations = d.generer(range(0, 1), plafond=30)
        def planter(_session):
            raise RuntimeError("plantage simulé")
        try:
            clonage.entrainer(
                JugeLineaire(), situations, situations,
                clonage.Reglages(lot=4, mises_a_jour=10, point_de_controle_tous_les=5),
                bavard=False, session=reprise.Session(), sauvegarde=planter)
        except RuntimeError:
            pass
        time.sleep(3)
        print("vivant")
    """)
    resultat = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=120)
    assert resultat.returncode == 0, resultat.stderr[-2000:]
    assert "vivant" in resultat.stdout


def test_pas_de_fichier_provisoire_apres_ecriture(tmp_path):
    """L'écriture passe par un fichier temporaire, puis un renommage atomique :
    400 Mio écrits en place seraient perdus par une coupure de courant."""
    chemin = tmp_path / "seance.pt"
    reprise.sauvegarder(reprise.Session(), chemin)
    assert chemin.exists()
    assert list(tmp_path.iterdir()) == [chemin]


def test_charger_sans_fichier_rend_none(tmp_path):
    assert reprise.charger(tmp_path / "rien.pt") is None


# --- L'enchaînement des phases -------------------------------------------


def test_les_phases_s_enchainent_dans_l_ordre_du_document():
    session = reprise.Session()
    vues = [session.phase]
    while session.phase != "fini":
        session.meilleur_etat = {}
        reprise._passer_a_la_phase_suivante(session, clonage.Journal())
        vues.append(session.phase)
    assert vues == ["clonage", "dagger-1", "dagger-2", "dagger-3", "fini"]


def test_sans_dagger_le_clonage_termine_le_run():
    session = reprise.Session(avec_dagger=False)
    session.meilleur_etat = {}
    reprise._passer_a_la_phase_suivante(session, clonage.Journal())
    assert session.phase == "fini"
    assert session.terminee


def test_une_nouvelle_phase_repart_du_meilleur_etat_avec_un_optimiseur_neuf():
    """Un run d'une traite construit un Adam par phase (§10.4) : la reprise
    doit faire pareil, sinon les moments du clonage contamineraient DAgger."""
    meilleur = {"marque": torch.zeros(1)}
    session = reprise.Session()
    session.meilleur_etat = meilleur
    session.optimiseur, session.generateur = {"faux": 1}, {"faux": 2}
    session.pas, session.meilleur_accord = 2400, 0.5
    reprise._passer_a_la_phase_suivante(session, clonage.Journal())

    assert session.parametres is meilleur
    assert session.optimiseur is None
    assert session.generateur is None
    assert session.pas == 0
    assert session.meilleur_accord == -1.0


MINI_TOURS = (range(2000, 2001), range(2100, 2101), range(2200, 2201))
MINI_DAGGER = clonage.Reglages(lot=8, taux=0.01, mises_a_jour=20, point_de_controle_tous_les=10)


def _poursuivre(session, situations, test_situations, chemin, limite=None):
    return reprise.poursuivre(
        _modele(), lambda m: NoteurTorch(m, "juge"), situations, test_situations, session,
        reglages=clonage.Reglages(lot=8, taux=0.01, mises_a_jour=20,
                                  point_de_controle_tous_les=10),
        reglages_dagger=MINI_DAGGER, limite_secondes=limite, nom="juge",
        chemin=chemin, tours=MINI_TOURS, plafond=40,
    )


@pytest.mark.lent
def test_le_run_complet_en_seances_egale_le_run_d_une_traite(
    situations, test_situations, tmp_path
):
    """Clonage puis trois tours de DAgger, d'une traite ou par petits bouts.

    Chaque séance repart d'un modèle neuf et du seul fichier sur disque, comme
    le lendemain matin : c'est la situation réelle, et c'est ce qui prouve que
    rien d'indispensable n'est resté en mémoire.
    """
    d_une_traite = reprise.Session(budget_clonage=20)
    modele, d_une_traite = _poursuivre(d_une_traite, situations, test_situations,
                                       tmp_path / "une.pt")
    assert d_une_traite.phase == "fini"

    chemin = tmp_path / "seances.pt"
    reprise.sauvegarder(reprise.Session(budget_clonage=20), chemin)
    seances = 0
    while not (session := reprise.charger(chemin)).terminee:
        morceaux, session = _poursuivre(session, situations, test_situations, chemin, limite=1e-9)
        seances += 1
        assert seances < 50, "une séance n'a rien fait avancer"

    assert session.phase == "fini"
    assert seances > 4, "le découpage n'a rien découpé"
    assert torch.equal(_poids(morceaux), _poids(modele))
    assert [p for p, _ in session.historique] == ["clonage", "dagger-1", "dagger-2", "dagger-3"]
    for (_, attendu), (_, obtenu) in zip(d_une_traite.historique, session.historique,
                                         strict=True):
        assert obtenu.pertes == attendu.pertes
    assert len(session.dagger_collecte) == len(d_une_traite.dagger_collecte)


def test_la_collecte_dagger_survit_a_un_changement_de_phase():
    """Elle coûte une demi-heure par tour : la refaire à chaque reprise ferait
    d'une séance de trois heures une séance de deux."""
    session = reprise.Session()
    session.meilleur_etat = {}
    session.dagger_collecte = ["situation"]
    session.dagger_collecte_faite = True
    reprise._passer_a_la_phase_suivante(session, clonage.Journal())
    assert session.dagger_collecte == ["situation"]
    assert not session.dagger_collecte_faite


# --- La recette des corrections, en séances -------------------------------

CORRIGEE = clonage.Reglages(lot=8, taux=0.01, mises_a_jour=40, point_de_controle_tous_les=10,
                            perte="optimaux", taux_final=0.001)


def test_la_recette_corrigee_se_reprend_aussi_exactement(situations, test_situations):
    """Le taux décroissant dépend du numéro de pas : une reprise doit retrouver
    exactement la même suite de taux, donc la même trajectoire."""
    def entrainer(session=None, limite=None):
        return clonage.entrainer(_modele(), situations, test_situations, CORRIGEE, nom="juge",
                                 bavard=False, session=session, limite_secondes=limite)

    modele, journal = entrainer()
    session = reprise.Session(budget_clonage=40, perte="optimaux", taux_final=0.001)
    entrainer(session, limite=1e-9)
    assert session.pas == 10
    repris, journal_repris = entrainer(session)
    assert torch.equal(_poids(repris), _poids(modele))
    assert journal_repris.pertes == journal.pertes


def test_on_ne_change_pas_de_recette_en_route(situations, test_situations, tmp_path):
    session = reprise.Session(budget_clonage=20)  # recette du document
    with pytest.raises(ValueError, match="recette"):
        reprise.poursuivre(_modele(), lambda m: NoteurTorch(m, "juge"), situations,
                           test_situations, session, reglages=CORRIGEE,
                           chemin=tmp_path / "s.pt")


DIFFICILES = clonage.Reglages(lot=8, taux=0.01, mises_a_jour=40, point_de_controle_tous_les=10,
                              perte="optimaux", taux_final=0.001, tirage="difficiles")


def test_le_tirage_difficile_se_reprend_aussi_exactement(situations, test_situations):
    """Les concurrents dépendent des notes du modèle, donc de ses paramètres :
    une reprise exacte des paramètres doit redonner exactement les mêmes lots."""
    def entrainer(session=None, limite=None):
        return clonage.entrainer(_modele(), situations, test_situations, DIFFICILES, nom="juge",
                                 bavard=False, session=session, limite_secondes=limite)

    modele, journal = entrainer()
    session = reprise.Session(budget_clonage=40, perte="optimaux", taux_final=0.001,
                              tirage="difficiles")
    entrainer(session, limite=1e-9)
    repris, journal_repris = entrainer(session)
    assert torch.equal(_poids(repris), _poids(modele))
    assert journal_repris.pertes == journal.pertes
