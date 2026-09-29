"""Une commande par étape du plan (conception.md §17).

    mouche telecharger          # les 3 fichiers MaleCNS (1,2 Go) dans data/brut
    mouche inspecter            # schéma des fichiers, sans rien charger
    mouche preparer-graphe      # construit le graphe et vérifie le contrat de §7.1
    mouche demonstrations       # 410 parties de l'expert, mises en cache
    mouche juge-lineaire        # clonage + DAgger + évaluation du juge linéaire
    mouche run-court            # mise au point sur le connectome complet (étape F, §17)
    mouche entrainer-mouche     # le run complet : clonage + 3 tours de DAgger
    mouche avancement           # où en est le run, sans toucher à la carte
    mouche comparer-recettes    # les deux corrections de §10.3 contre la recette du document
    mouche enregistrer --modele M  # une partie de M, avec l'activité de ses neurones
    mouche page                 # met la dernière partie enregistrée dans web/index.html
    mouche diffuser             # la mouche joue EN DIRECT, sur http://localhost:8000
    mouche reperes              # hasard et expert sur le protocole final

Le run complet dure 7 h 30 (§14.3) et se découpe en séances :

    mouche entrainer-mouche --duree 3    # ce soir
    mouche entrainer-mouche --duree 3    # demain, reprend exactement là
    mouche entrainer-mouche              # jusqu'au bout

Le budget reste celui du document — ce qui se découpe est le temps d'occupation
de la carte, pas la recette (§14.6).

Chaque commande écrit ce qu'elle trouve, avec ses intervalles. Les réglages qui
comptent sont ceux du document : ils ne se passent pas en ligne de commande, car
ce qui doit être reproductible est le fichier, pas la mémoire de celui qui tape.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# Doit être posé avant que PyTorch n'initialise CUDA, donc avant tout import de
# torch — que les commandes font paresseusement, plus bas. Sur 3,68 Gio, la
# fragmentation de l'allocateur suffit à faire échouer une allocation de 128 Mio
# alors que 300 Mio sont réservés mais inutilisés ; les segments extensibles
# suppriment ce cas.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CACHE = Path("data/demonstrations")


def _charger_ou_generer(nom: str, fabrique):
    """Les démonstrations coûtent une minute de CPU : on les garde sur disque."""
    chemin = CACHE / f"{nom}.pkl"
    if chemin.exists():
        with chemin.open("rb") as fichier:
            return pickle.load(fichier)
    debut = time.perf_counter()
    situations = fabrique()
    CACHE.mkdir(parents=True, exist_ok=True)
    with chemin.open("wb") as fichier:
        pickle.dump(situations, fichier)
    print(f"  {nom} : {len(situations)} situations générées en {time.perf_counter()-debut:.0f} s")
    return situations


def jeux_de_situations():
    from .entrainement import demonstrations as d

    entrainement = _charger_ou_generer("entrainement", d.jeu_d_entrainement)
    test = _charger_ou_generer("test", d.jeu_de_test)
    return entrainement, test, d.echantillon_de_test(test)


def commande_telecharger(_) -> int:
    """Récupère les trois fichiers du MaleCNS et inscrit leurs empreintes.

    Les trois seulement : la page de téléchargement propose aussi les points
    synaptiques, les paires de partenaires et les volumes d'imagerie, soit
    environ 22 Go dont le projet n'utilise rien — et le disque n'en a que 20.
    """
    import urllib.request

    from .connectome import acquisition as acq
    from .connectome import empreintes as emp

    acq.BRUT.mkdir(parents=True, exist_ok=True)
    for nom in acq.FICHIERS:
        destination = acq.chemin(nom)
        if destination.exists():
            print(f"  {nom} : déjà là ({destination.stat().st_size / 1e6:.0f} Mo)")
        else:
            attendu = acq.TAILLES_ATTENDUES_MO[nom]
            print(f"  {nom} : téléchargement, environ {attendu} Mo…", flush=True)
            debut = time.perf_counter()
            urllib.request.urlretrieve(acq.url(nom), destination)  # noqa: S310
            taille = destination.stat().st_size / 1e6
            print(f"    {taille:.0f} Mo en {time.perf_counter() - debut:.0f} s")
        emp.inscrire(nom, emp.empreinte(destination))
    print(f"empreintes inscrites dans {emp.MANIFESTE}")
    print("Ce sont les empreintes de VOS fichiers : les remplacer par celles que")
    print("publie flyhard, sinon la vérification ne vérifie que sa propre copie.")
    return 0


def commande_inspecter(_) -> int:
    from .connectome import acquisition as acq

    for schema in acq.inspecter_tout():
        print(f"  {schema}")
    fichier = acq.chemin("annotations")
    if not fichier.exists():
        return 0
    neurones = acq.lire_les_neurones()
    print()
    for nom, obtenu, attendu in (
        ("neurones tracés", len(neurones["identifiants"]), acq.NEURONES_ATTENDUS),
        ("neurones d'entrée", len(neurones["entrees"]), acq.ENTREES_ATTENDUES),
        ("neurones moteurs", len(neurones["sorties"]), acq.SORTIES_ATTENDUES),
    ):
        marque = "✓" if obtenu == attendu else f"✗ attendu {attendu}"
        print(f"  {nom:<20} {obtenu:>7}  {marque}")
    avec_soma = sum(1 for soma in neurones["somas"] if soma)
    print(f"  {'avec soma':<20} {avec_soma:>7}  (§7.5 en tire 20 000 pour la démo)")
    return 0


def commande_preparer_graphe(_) -> int:
    """Construit le graphe depuis les trois fichiers et vérifie le contrat de §7.1."""
    import pickle

    from .connectome import acquisition as acq
    from .connectome.construction import PREPARE_GRAPHE, construire_le_graphe

    manquants = [nom for nom in acq.FICHIERS if not acq.chemin(nom).exists()]
    if manquants:
        print(f"fichiers manquants : {manquants} — lancer d'abord « mouche telecharger »")
        return 1

    debut = time.perf_counter()
    graphe = construire_le_graphe()
    print(f"graphe construit en {time.perf_counter() - debut:.0f} s")
    print(f"  {graphe.n_neurones} neurones, {graphe.n_connexions} connexions")
    print(f"  {graphe.part_inhibitrice:.1%} de connexions inhibitrices")
    print(f"  {graphe.sans_connexion_entrante()} neurones sans connexion entrante")

    debut = time.perf_counter()
    distance, part = graphe.distance_aux_sorties()
    print(f"  D = {distance} — {part:.1%} des neurones moteurs atteignables "
          f"({time.perf_counter() - debut:.0f} s)")
    print(f"  donc K = min(D + 4, 16) = {min(distance + 4, 16)}")

    PREPARE_GRAPHE.parent.mkdir(parents=True, exist_ok=True)
    with PREPARE_GRAPHE.open("wb") as fichier:
        pickle.dump(graphe, fichier)
    print(f"  enregistré dans {PREPARE_GRAPHE}")
    return 0


def commande_demonstrations(_) -> int:
    entrainement, test, echantillon = jeux_de_situations()
    bruitees = sum(s.bruite for s in entrainement)
    candidats = sum(len(s.candidats) for s in entrainement) / len(entrainement)
    print(
        f"entraînement : {len(entrainement)} situations, "
        f"{bruitees / len(entrainement):.1%} bruitées, {candidats:.1f} candidats en moyenne"
    )
    print(f"test         : {len(test)} situations, sous-échantillon de {len(echantillon)}")
    return 0


def commande_reperes(_) -> int:
    from .evaluation import parties as ev
    from .evaluation import rapport
    from .expert.criteres import Expert
    from .modeles.hasard import Hasard

    series = [ev.serie(noteur, ev.FINAL) for noteur in (Hasard(), Expert())]
    print(rapport.tableau(series, budgets={"hasard": "aucun", "expert": "aucun — 4 poids figés"}))
    return 0


def commande_juge_lineaire(args) -> int:
    from .entrainement import clonage, dagger
    from .evaluation import accord as ac
    from .evaluation import parties as ev
    from .evaluation import rapport
    from .modeles.lineaire import JugeLineaire
    from .modeles.torche import NoteurTorch

    entrainement, _, echantillon = jeux_de_situations()
    print(f"hasard sur l'échantillon de test : {ac.niveau_du_hasard(echantillon):.1%} d'accord")

    adapter = lambda modele: NoteurTorch(modele, "juge linéaire")  # noqa: E731

    debut = time.perf_counter()
    modele, journal = clonage.entrainer(
        JugeLineaire(), entrainement, echantillon,
        clonage.REGLAGES_LINEAIRE, nom="juge linéaire", graine_run=args.graine,
    )
    accords = " → ".join(f"{valeur:.1%}" for _, valeur in journal.accords[-4:])
    print(f"clonage : {time.perf_counter()-debut:.0f} s, accord {accords}")
    apres_clonage = ev.serie(adapter(modele), ev.DEVELOPPEMENT)
    print(f"  développement après clonage : {apres_clonage.moyenne:.1f} lignes")

    debut = time.perf_counter()
    modele, resultat = dagger.executer(
        modele, adapter, entrainement, echantillon,
        clonage.REGLAGES_LINEAIRE_DAGGER, nom="juge linéaire", graine_run=args.graine,
    )
    print(f"DAgger : {time.perf_counter()-debut:.0f} s, "
          f"{resultat.total_collecte} situations collectées {resultat.situations_collectees}")
    apres_dagger = ev.serie(adapter(modele), ev.DEVELOPPEMENT)
    print(f"  développement après DAgger : {apres_dagger.moyenne:.1f} lignes")

    finale = ev.serie(adapter(modele), ev.FINAL)
    pas = clonage.REGLAGES_LINEAIRE.mises_a_jour + 3 * clonage.REGLAGES_LINEAIRE_DAGGER.mises_a_jour
    budget = f"{pas} pas · 206 paramètres"
    print()
    print(rapport.tableau([finale], budgets={"juge linéaire": budget}))
    return 0


def commande_run_court(args) -> int:
    """La mise au point de l'étape F : 200 mises à jour, trois valeurs de K, témoins.

    C'est le deuxième point d'arrêt du projet (§17). Il répond à une seule
    question — est-ce que ça apprend du tout — avant d'engager sept heures et
    demie de GPU. Un accord d'au moins 10 % contre environ 4 % au hasard suffit.
    """
    import pickle

    import torch

    from .connectome import controles as ctrl
    from .connectome import noyau
    from .connectome.construction import PREPARE_GRAPHE
    from .connectome.facteur import MESURE_MALECNS
    from .connectome.modele import Mouche
    from .entrainement import clonage
    from .evaluation.accord import accord, niveau_du_hasard
    from .modeles.torche import NoteurTorch

    peripherique = "cuda" if torch.cuda.is_available() else "cpu"
    entrainement, _, echantillon = jeux_de_situations()
    with PREPARE_GRAPHE.open("rb") as fichier:
        graphe = pickle.load(fichier)
    distance, _ = graphe.distance_aux_sorties()
    hasard = niveau_du_hasard(echantillon)
    print(f"{graphe.n_neurones} neurones, {graphe.n_connexions} connexions, D = {distance}")
    print(f"facteur global {MESURE_MALECNS} · {peripherique} · hasard {hasard:.1%} d'accord\n")

    pas = args.pas if args.pas is not None else 200
    reglages = clonage.Reglages(
        mises_a_jour=pas, point_de_controle_tous_les=max(50, pas // 2)
    )
    resultats = {}
    for k in (distance + 2, distance + 4, distance + 8):
        mouche = Mouche(graphe, k=k, facteur_global=MESURE_MALECNS, graine_run=args.graine)
        depart = accord(NoteurTorch(mouche, "mouche", peripherique), echantillon)
        debut = time.perf_counter()
        mouche, journal = clonage.entrainer(
            mouche, entrainement, echantillon, reglages,
            graine_run=args.graine, nom=f"mouche K={k}", peripherique=peripherique,
        )
        arrivee = accord(NoteurTorch(mouche, "mouche", peripherique), echantillon)
        resultats[k] = arrivee
        print(f"K = {k:>2} : accord {depart:.1%} → {arrivee:.1%} "
              f"| perte {journal.pertes[0]:.3f} → {sum(journal.pertes[-20:]) / 20:.3f} "
              f"| {time.perf_counter() - debut:.0f} s")
        del mouche
        noyau.vider_le_cache()
        if peripherique == "cuda":
            torch.cuda.empty_cache()

    meilleur = max(resultats, key=resultats.get)
    print(f"\nmeilleur K : {meilleur} ({resultats[meilleur]:.1%})")
    print(f"K du document = D + 4 = {distance + 4} ({resultats[distance + 4]:.1%})")

    print("\ntémoins — doivent rendre exactement la même note à tous les candidats :")
    for nom, fabrique in (("graphe coupé", ctrl.couper_le_graphe),
                          ("entrée aveuglée", ctrl.aveugler_l_entree)):
        temoin = fabrique(Mouche(graphe, k=distance + 4, facteur_global=MESURE_MALECNS))
        notes = NoteurTorch(temoin, nom, peripherique).noter(echantillon[0].candidats)
        egales = bool((abs(notes - notes[0]) < 1e-6).all())
        print(f"  {nom:<16} notes toutes égales : {egales} | accord "
              f"{accord(NoteurTorch(temoin, nom, peripherique), echantillon[:200]):.1%}")
        del temoin
        noyau.vider_le_cache()
        if peripherique == "cuda":
            torch.cuda.empty_cache()

    obtenu = resultats[distance + 4]
    print(f"\n{'APPREND' if obtenu >= 0.10 else 'N APPREND PAS'} : "
          f"{obtenu:.1%} contre {hasard:.1%} au hasard, seuil à 10 %")
    return 0


def _chemin_de_seance(args) -> Path:
    from .entrainement import reprise

    return Path(args.seance) if args.seance else reprise.CHEMIN


def commande_avancement(args) -> int:
    """Où en est le run, sans toucher à la carte."""
    from .entrainement import reprise

    chemin = _chemin_de_seance(args)
    session = reprise.charger(chemin)
    if session is None:
        print(f"aucune séance en cours ({chemin})")
        return 0
    print(f"{chemin} ({chemin.stat().st_size / 2**20:.0f} Mio)")
    print(f"  {session.avancement()}")
    for phase, journal in session.historique:
        suite = " → ".join(f"{v:.1%}" for _, v in journal.accords[-4:])
        print(f"  {phase:<10} terminée · accord {suite}"
              + (f" · {journal.relances} relance(s)" if journal.relances else ""))
    if session.dagger_collecte:
        print(f"  DAgger : {len(session.dagger_collecte)} situations accumulées")
    return 0


def commande_entrainer_mouche(args) -> int:
    """Le run complet de l'étape G : clonage, puis trois tours de DAgger (§17).

    En une fois, ou en plusieurs séances. `--duree 3` arrête la séance au dernier
    point de contrôle qui tient dans trois heures, enregistre tout, et rend la
    carte ; relancer la même commande reprend exactement là. **Le budget du
    document ne change pas** : c'est le temps d'occupation de la carte qui se
    découpe, pas la recette (§14.6).

    `--pas` reste possible pour un run plus court — un modèle à 600 pas ne vaut
    pas le run complet, mais il joue vraiment, et son score est publié avec son
    budget comme les autres.
    """
    import pickle

    import torch

    from .connectome.construction import PREPARE_GRAPHE
    from .connectome.facteur import MESURE_MALECNS
    from .connectome.modele import Mouche
    from .enregistrement.capture import capturer, ecrire
    from .entrainement import clonage, reprise
    from .evaluation import parties as ev
    from .evaluation.accord import niveau_du_hasard
    from .modeles.torche import NoteurTorch
    from .serveur import verrou_gpu
    from .tetris.partie import jouer

    if not torch.cuda.is_available():
        # Sur le processeur, un pas prend des minutes au lieu de 5 s : la séance
        # tournerait trois heures pour quelques dizaines de pas. Mieux vaut
        # refuser que passer au CPU en silence, comme le faisait ce code.
        print("CUDA indisponible : l'entraînement de la mouche ne tourne pas sur le processeur.")
        print("  Après une mise en veille, le pilote NVIDIA perd souvent CUDA ; pour le rétablir :")
        print("    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm")
        print("  (sinon, redémarrer la machine)")
        return 1

    pas = args.pas if args.pas is not None else clonage.REGLAGES_MOUCHE.mises_a_jour
    chemin_seance = _chemin_de_seance(args)
    if args.recommencer:
        Path(chemin_seance).unlink(missing_ok=True)

    session = reprise.charger(chemin_seance)
    if session is None:
        session = reprise.Session(graine_run=args.graine, budget_clonage=pas,
                                  avec_dagger=not args.sans_dagger,
                                  perte=args.perte or "choix", taux_final=args.taux_final,
                                  tirage=args.tirage or "hasard")
    elif args.pas is not None and session.budget_clonage != pas:
        # Un run à moitié fait sous 2 400 pas et terminé sous 1 200 ne serait ni
        # l'un ni l'autre, et son budget publié serait faux.
        print(f"la séance en cours porte un budget de {session.budget_clonage} pas de "
              f"clonage, pas {pas}.")
        print("  reprendre sans --pas, ou --recommencer pour repartir de zéro")
        return 1
    elif ((args.perte is not None and args.perte != session.perte)
          or (args.taux_final is not None and args.taux_final != session.taux_final)
          or (args.tirage is not None and args.tirage != session.tirage)):
        print(f"la séance en cours suit la recette {reprise.recette(session)} : "
              f"on ne change pas de recette en route.")
        print("  reprendre sans --perte, --taux-final ni --tirage, ou --recommencer")
        return 1
    if session.terminee:
        print(session.avancement())
        print("  --recommencer pour repartir de zéro")
        return 0

    # L'entraînement est prioritaire : il évince la diffusion, qui est faite pour
    # être interrompue (§13.4). Entre deux séances, la carte est libre.
    try:
        verrou_gpu.prendre("entrainement", prioritaire=True)
    except verrou_gpu.GpuOccupe as erreur:
        print(erreur)
        return 1

    try:
        peripherique = "cuda" if torch.cuda.is_available() else "cpu"
        entrainement, _, echantillon = jeux_de_situations()
        with PREPARE_GRAPHE.open("rb") as fichier:
            graphe = pickle.load(fichier)
        distance, _ = graphe.distance_aux_sorties()
        k = min(distance + 4, 16)
        print(f"K = {k} · facteur global {MESURE_MALECNS} · {peripherique} · "
              f"hasard {niveau_du_hasard(echantillon):.1%} d'accord")

        mouche = Mouche(graphe, k=k, facteur_global=MESURE_MALECNS, graine_run=args.graine)
        print(f"{mouche.n_parametres:,} paramètres".replace(",", " "))
        print(f"séance {session.seances + 1} · {session.avancement()}")
        if args.duree:
            print(f"  limite de séance : {args.duree:g} h — arrêt au dernier point de "
                  f"contrôle qui tient dedans")
        print()

        recette = reprise.recette(session)
        if recette != {"perte": "choix", "taux_final": None, "tirage": "hasard"}:
            print(f"  recette : {recette}")
        reglages = clonage.Reglages(
            mises_a_jour=session.budget_clonage,
            # 12 points de contrôle pour les 2 400 pas du document (§14.3), donc
            # une séance qui s'arrête à moins de 20 minutes près.
            point_de_controle_tous_les=min(200, max(50, session.budget_clonage // 6)),
            **recette,
        )
        reglages_dagger = clonage.Reglages(
            mises_a_jour=clonage.REGLAGES_MOUCHE_DAGGER.mises_a_jour, **recette)
        adapter = lambda modele: NoteurTorch(modele, "mouche", peripherique)  # noqa: E731

        debut = time.perf_counter()
        mouche, session = reprise.poursuivre(
            mouche, adapter, entrainement, echantillon, session,
            reglages=reglages, reglages_dagger=reglages_dagger,
            peripherique=peripherique, nom="mouche", chemin=chemin_seance,
            limite_secondes=args.duree * 3600 if args.duree else None,
        )
        print(f"\nséance close en {(time.perf_counter() - debut) / 60:.0f} min · "
              f"{session.avancement()}")

        suffixe = "complet" if session.phase == "fini" else session.etiquette
        chemin = Path("data/points_de_controle") / f"mouche-{suffixe}-graine{args.graine}.pt"
        chemin.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clonage.capturer_parametres(mouche), chemin)
        print(f"  modèle sauvegardé : {chemin.name} "
              f"({chemin.stat().st_size / 2**20:.0f} Mio)")

        if not session.terminee:
            print("\n  relancer la même commande pour la séance suivante ;")
            print("  « mouche diffuser » entre-temps, la carte est libre.")
            return 0
        if session.phase == "echoue":
            print("\nrun échoué (règle de divergence) — publié comme tel (§10.3)")
            return 1

        # Le run est fini : l'évaluation de développement et la partie
        # enregistrée tournent aussi sur la carte, donc sous le même verrou.
        noteur = NoteurTorch(mouche, "mouche", peripherique)
        debut = time.perf_counter()
        dev = ev.serie(noteur, ev.DEVELOPPEMENT)
        print(f"\ndéveloppement : {dev.moyenne:.1f} lignes, {dev.pieces_moyennes:.0f} pièces "
              f"({time.perf_counter() - debut:.0f} s)")

        partie = jouer(noteur, graine=1000, plafond=args.plafond or 300,
                       enregistrer_coups=True)
        titre = f"mouche · {session.budget_clonage} pas"
        capture = capturer(partie, f"{titre} + DAgger" if session.avec_dagger else titre)
        ecrire(capture, Path("data/enregistrements/mouche.json.gz"))
        print(f"partie enregistrée : {partie.lignes} lignes, {partie.pieces_posees} pièces")
    finally:
        verrou_gpu.rendre()
    return 0


REFERENCE_RECETTE = Path("data/points_de_controle/reference-recette-document-800pas.pt")


def commande_comparer_recettes(args) -> int:
    """Un essai de recette contre un modèle de référence, à budget égal (§10.3).

    **À budget égal.** La référence est un meilleur point déjà entraîné — par
    défaut celui des 800 premiers pas du run en recette du document, 55,0 %
    d'accord sur l'échantillon. L'essai s'entraîne autant de pas, même graine,
    dans la recette donnée par `--perte`, `--taux-final` et `--tirage`, et son
    meilleur point est choisi de la même façon, sur le même échantillon, aux
    mêmes pas. Chaque recette essayée a ses propres fichiers : un essai ne
    reprend ni n'écrase jamais celui d'une autre.

    **Mesuré sur tout le jeu de test**, environ 6 000 situations, et situation par
    situation : les deux modèles voient les mêmes grilles. L'échantillon de 500
    sert à choisir les points de contrôle ; avec ± 2 points de bruit, il ne peut
    pas trancher un écart de quelques points. L'intervalle de l'écart est tiré
    par parties entières (`difference_appariee_par_groupes`).

    L'essai s'entraîne en séance, comme le run : interrompu, il reprend.
    """
    import json

    import numpy as np
    import torch

    from . import graines
    from .connectome.construction import PREPARE_GRAPHE
    from .connectome.facteur import MESURE_MALECNS
    from .connectome.modele import Mouche
    from .entrainement import clonage, reprise
    from .entrainement import demonstrations as d
    from .evaluation.accord import reussites
    from .evaluation.bootstrap import difference_appariee_par_groupes
    from .modeles.torche import NoteurTorch
    from .serveur import verrou_gpu

    if not torch.cuda.is_available():
        print("CUDA indisponible — voir « mouche entrainer-mouche » pour le rétablir.")
        return 1
    chemin_reference = Path(args.reference) if args.reference else REFERENCE_RECETTE
    if not chemin_reference.exists():
        print(f"{chemin_reference} : référence absente, rien à comparer")
        return 1
    reference = torch.load(chemin_reference, map_location="cpu", weights_only=False)
    pas = reference["pas"]
    recette = {"perte": args.perte or "optimaux",
               "taux_final": args.taux_final if args.taux_final is not None else 0.004,
               "tirage": args.tirage or "hasard"}
    nom_essai = f"{recette['perte']}-{recette['taux_final']}-{recette['tirage']}"
    chemin_seance = (Path(args.seance) if args.seance
                     else Path(f"data/points_de_controle/seance-essai-{nom_essai}.pt"))
    chemin_resultat = Path(f"data/resultats/comparaison-{nom_essai}.json")

    try:
        verrou_gpu.prendre("entrainement", prioritaire=True)
    except verrou_gpu.GpuOccupe as erreur:
        print(erreur)
        return 1

    try:
        peripherique = "cuda"
        entrainement, test, echantillon = jeux_de_situations()
        with PREPARE_GRAPHE.open("rb") as fichier:
            graphe = pickle.load(fichier)
        distance, _ = graphe.distance_aux_sorties()
        mouche = Mouche(graphe, k=min(distance + 4, 16), facteur_global=MESURE_MALECNS,
                        graine_run=reference["graine_run"])

        session = reprise.charger(chemin_seance) or reprise.Session(
            graine_run=reference["graine_run"], budget_clonage=pas, avec_dagger=False,
            **recette)
        print(f"référence : {reference['recette']} — {pas} pas, "
              f"{reference['accord_echantillon']:.1%} sur l'échantillon")
        print(f"essai     : {reprise.recette(session)} — {session.avancement()}\n")

        if not session.terminee:
            # Les mêmes points de contrôle que la référence, pas ceux que donnerait
            # le budget de l'essai : la référence, tirée d'un run de 2 400 pas, a
            # choisi son meilleur point parmi 200, 400, 600 et 800. L'intervalle
            # d'un run de 800 pas serait de 133 — sept candidats contre quatre,
            # donc plus de chances de tomber sur un bon point par hasard.
            intervalle = reference["accords"][0][0]
            reglages = clonage.Reglages(
                mises_a_jour=pas, point_de_controle_tous_les=intervalle,
                **reprise.recette(session))
            print(f"  points de contrôle tous les {intervalle} pas, comme la référence : "
                  f"{[p for p, _ in reference['accords']]}")
            mouche, session = reprise.poursuivre(
                mouche, lambda m: NoteurTorch(m, "essai", peripherique),
                entrainement, echantillon, session, reglages=reglages,
                peripherique=peripherique, nom="essai", chemin=chemin_seance,
                limite_secondes=args.duree * 3600 if args.duree else None)
            if not session.terminee:
                print(f"\nséance close · {session.avancement()} — relancer la même commande")
                return 0
        if session.phase == "echoue":
            print("essai échoué (règle de divergence) — publié comme tel")
            return 1
        torch.save({
            "parametres": session.parametres, "pas": pas, "graine_run": session.graine_run,
            "accord_echantillon": max(a for _, a in session.historique[-1][1].accords),
            "accords": session.historique[-1][1].accords,
            "recette": f"essai {reprise.recette(session)}",
        }, Path("data/points_de_controle") / f"essai-{nom_essai}-{pas}pas.pt")

        # Les parties du jeu de test, pour tirer l'intervalle par parties entières.
        par_partie = [len(d.generer_partie(g, bruit=0.0)) for g in graines.TEST]
        if sum(par_partie) != len(test):
            raise RuntimeError("le jeu de test en cache ne correspond plus aux parties 500 à 519")
        groupes = np.repeat(np.arange(len(par_partie)), par_partie)

        print(f"\nmesure sur tout le jeu de test : {len(test)} situations, "
              f"{len(par_partie)} parties")
        justes = {}
        for nom, parametres in (("reference", reference["parametres"]),
                                ("essai", session.parametres)):
            clonage.restaurer_parametres(mouche, parametres)
            debut = time.perf_counter()
            justes[nom] = reussites(NoteurTorch(mouche, nom, peripherique), test)
            print(f"  {nom:<9} accord {justes[nom].mean():.1%} "
                  f"({(time.perf_counter() - debut) / 60:.0f} min)", flush=True)

        ecart = difference_appariee_par_groupes(justes["essai"], justes["reference"], groupes)
        if ecart.bas > 0:
            conclusion = "gain démontré : l'intervalle exclut zéro"
        elif ecart.haut < 0:
            conclusion = "perte démontrée : l'intervalle exclut zéro"
        else:
            conclusion = "pas d'écart démontré : l'intervalle contient zéro"
        print(f"\nécart essai − référence : {ecart.valeur * 100:+.1f} points "
              f"[{ecart.bas * 100:+.1f} ; {ecart.haut * 100:+.1f}], IC à 95 % par parties")
        print(f"  {conclusion}")

        chemin_resultat.parent.mkdir(parents=True, exist_ok=True)
        chemin_resultat.write_text(json.dumps({
            "pas": pas,
            "graine_run": reference["graine_run"],
            "recettes": {
                "reference": f"{reference['recette']} ({chemin_reference.name})",
                "essai": f"{reprise.recette(session)}, taux 0,04 → final en cosinus",
            },
            "accord_echantillon_500": {
                "reference": reference["accords"],
                "essai": session.historique[-1][1].accords if session.historique else [],
            },
            "jeu_de_test": {"situations": len(test), "parties": len(par_partie)},
            "accord_test_complet": {nom: float(v.mean()) for nom, v in justes.items()},
            "ecart_points": {"valeur": ecart.valeur * 100, "bas": ecart.bas * 100,
                             "haut": ecart.haut * 100},
            "conclusion": conclusion,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  résultat écrit dans {chemin_resultat}")
    finally:
        verrou_gpu.rendre()
    return 0


def commande_page(args) -> int:
    """Met une partie enregistrée dans la page, à la place de la précédente.

    La page s'ouvre seule, sans serveur : la partie y est intégrée, sur la ligne
    `let P = …`. La remplacer à la main, c'était recoller 20 000 caractères de
    JSON dans un fichier de 3 Mo ; cette commande le fait, et complète au passage
    une capture ancienne de sa trajectoire (`capture.completer`).
    """
    import json

    from .enregistrement.capture import completer, lire

    source = Path(args.partie or "data/enregistrements/mouche.json.gz")
    if not source.exists():
        print(f"{source} : aucune partie enregistrée")
        return 1
    capture = completer(lire(source))

    page = Path("web/index.html")
    lignes = page.read_text(encoding="utf-8").split("\n")
    rangs = [i for i, ligne in enumerate(lignes) if ligne.startswith("let P = ")]
    if len(rangs) != 1:
        print(f"{page} : {len(rangs)} lignes « let P = » au lieu d'une, rien n'est modifié")
        return 1
    # « </ » fermerait la balise <script> si une chaîne le contenait.
    donnees = json.dumps(capture, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    lignes[rangs[0]] = f"let P = {donnees};"

    # Les positions du nuage de neurones, pour la partie intégrée comme pour le
    # direct : la page s'ouvre en fichier local, où elle ne peut rien charger.
    import base64

    from .connectome import nuage as nu

    description = capture.get("nuage") or {}
    fichier_nuage = nu.chemin(int(description.get("graine_run", args.graine)))
    if fichier_nuage.exists():
        nuage = nu.charger(fichier_nuage)
        if description and description["empreinte"] != nuage.empreinte():
            print(f"{source} a été enregistrée avec un autre nuage que {fichier_nuage} : "
                  f"rien n'est modifié")
            return 1
        balise = ('<script id="nuage" type="text/plain">'
                  + base64.b64encode(nu.pour_la_page(nuage)).decode("ascii") + "</script>")
        existante = [i for i, ligne in enumerate(lignes) if ligne.startswith('<script id="nuage"')]
        if existante:
            lignes[existante[0]] = balise
        else:
            maillage = next(i for i, ligne in enumerate(lignes)
                            if ligne.startswith('<script id="maillage"'))
            lignes.insert(maillage + 1, balise)
        print(f"  nuage : {nuage.n} neurones {nuage.composition()} · empreinte {nuage.empreinte()}")
    else:
        print(f"  pas de nuage ({fichier_nuage} absent) : la page s'affichera sans lui")

    provisoire = page.with_name(page.name + ".provisoire")
    provisoire.write_text("\n".join(lignes), encoding="utf-8")
    provisoire.replace(page)
    print(f"{page} : « {capture['nom']} », {len(capture['coups'])} coups, "
          f"{capture['lignes']} lignes — depuis {source} "
          f"({page.stat().st_size / 2**20:.1f} Mio)")
    return 0


def commande_diffuser(args) -> int:
    """Fait jouer la mouche en direct et sert la page (§13.3).

    Une seule partie tourne, diffusée à tous les navigateurs connectés : le coût
    GPU est constant, qu'il y ait un spectateur ou mille.
    """

    import torch
    import uvicorn

    from .serveur import verrou_gpu
    from .serveur.diffusion import construire

    modele = Path(args.modele) if args.modele else max(
        Path("data/points_de_controle").glob("mouche-*.pt"),
        key=lambda f: f.stat().st_mtime, default=None)
    if modele is None or not modele.exists():
        print("aucun modèle entraîné — lancer d'abord « mouche entrainer-mouche »")
        return 1

    try:
        verrou_gpu.prendre("diffusion")
    except verrou_gpu.GpuOccupe as erreur:
        print(erreur)
        return 1

    try:
        peripherique = "cuda" if torch.cuda.is_available() else "cpu"
        noteur, nuage = _mouche_qui_montre_son_activite(modele, peripherique)
        print(f"modèle : {modele.name} · {peripherique} · K = {noteur.modele.k}")
        print(f"nuage : {nuage.n} neurones {nuage.composition()} · empreinte {nuage.empreinte()}")
        print(f"\n  ouvrez http://{args.hote}:{args.port}\n")
        uvicorn.run(construire(noteur, Path("web"), modele.stem),
                    host=args.hote, port=args.port, log_level="warning")
    finally:
        verrou_gpu.rendre()
    return 0


def _mouche_qui_montre_son_activite(modele: Path, peripherique: str):
    """Charge un modèle entraîné et l'équipe de son nuage de neurones (§13.2).

    Rend un `NoteurAvecNuage` : il note comme n'importe quel noteur, et sait en
    plus rendre l'activité des neurones affichés pour le candidat choisi. La
    plage de chaque neurone est calibrée sur 64 grilles du jeu de test — les
    choix de l'expert, étalés sur les 20 parties.
    """
    import pickle

    import torch

    from .connectome import nuage as nu
    from .connectome.construction import PREPARE_GRAPHE
    from .connectome.facteur import MESURE_MALECNS
    from .connectome.modele import Mouche
    from .entrainement import demonstrations as d
    from .entrainement.clonage import restaurer_parametres

    etat = torch.load(modele, map_location="cpu", weights_only=False)
    # Deux formats : les paramètres seuls, ou — pour les modèles des essais de
    # recette — un dictionnaire qui les porte avec leur recette et leur graine.
    avec_details = "parametres" in etat
    graine_run = int(etat.get("graine_run", 0)) if avec_details else 0

    with PREPARE_GRAPHE.open("rb") as fichier:
        graphe = pickle.load(fichier)
    distance, _ = graphe.distance_aux_sorties()
    mouche = Mouche(graphe, k=min(distance + 4, 16), facteur_global=MESURE_MALECNS,
                    graine_run=graine_run)
    restaurer_parametres(mouche, etat["parametres"] if avec_details else etat)
    mouche = mouche.to(peripherique)

    nuage = nu.obtenir(graphe, graine_run)
    test = _charger_ou_generer("test", d.jeu_de_test)
    calibration = [s.candidats[s.choix_expert] for s in test[:: max(1, len(test) // 64)][:64]]
    activite = nu.Activite(mouche, nuage, calibration, peripherique)
    return nu.NoteurAvecNuage(mouche, modele.stem, peripherique, activite), nuage


def commande_enregistrer(args) -> int:
    """Fait jouer une partie au modèle, avec l'activité de son nuage, pour la page.

    Graine 1000, comme la partie enregistrée en fin de run. Par défaut 60 pièces
    au plus : chacune pèse 56 Ko d'activité (8 000 neurones × 7 mises à jour), et
    la page embarque la partie entière pour s'ouvrir sans serveur.
    """
    import torch

    from .enregistrement.capture import capturer, ecrire
    from .serveur import verrou_gpu
    from .tetris.partie import jouer

    if not args.modele or not Path(args.modele).exists():
        print("--modele : le point de contrôle à faire jouer")
        return 1
    if not torch.cuda.is_available():
        print("CUDA indisponible — voir « mouche entrainer-mouche » pour le rétablir.")
        return 1
    try:
        verrou_gpu.prendre("enregistrement")
    except verrou_gpu.GpuOccupe as erreur:
        print(erreur)
        return 1
    try:
        modele = Path(args.modele)
        noteur, nuage = _mouche_qui_montre_son_activite(modele, "cuda")
        plafond = args.plafond or 60
        debut = time.perf_counter()
        partie = jouer(noteur, graine=1000, plafond=plafond, enregistrer_coups=True)
        capture = capturer(partie, f"{modele.stem} · graine 1000", noteur)
        chemin = ecrire(capture, Path("data/enregistrements/mouche.json.gz"))
        print(f"partie : {partie.lignes} ligne(s), {partie.pieces_posees} pièces"
              + (f" (plafond de {plafond} atteint)" if partie.plafond_atteint else "")
              + f" · {time.perf_counter() - debut:.0f} s")
        print(f"  {chemin} ({chemin.stat().st_size / 2**20:.1f} Mio) · nuage {nuage.n} neurones")
        print("  « mouche page » pour la mettre dans la page")
    finally:
        verrou_gpu.rendre()
    return 0


COMMANDES = {
    "telecharger": commande_telecharger,
    "inspecter": commande_inspecter,
    "preparer-graphe": commande_preparer_graphe,
    "demonstrations": commande_demonstrations,
    "reperes": commande_reperes,
    "juge-lineaire": commande_juge_lineaire,
    "run-court": commande_run_court,
    "entrainer-mouche": commande_entrainer_mouche,
    "avancement": commande_avancement,
    "comparer-recettes": commande_comparer_recettes,
    "enregistrer": commande_enregistrer,
    "page": commande_page,
    "diffuser": commande_diffuser,
}


def main(argv: list[str] | None = None) -> int:
    analyseur = argparse.ArgumentParser(
        prog="mouche", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyseur.add_argument("commande", choices=sorted(COMMANDES))
    analyseur.add_argument("--graine", type=int, default=0, help="graine du run (§10.5)")
    analyseur.add_argument("--pas", type=int, default=None,
                           help="mises à jour de clonage (défaut : 2 400 du document)")
    analyseur.add_argument("--duree", type=float, default=None,
                           help="durée de la séance en heures ; sans elle, va jusqu'au bout")
    analyseur.add_argument("--seance", default=None,
                           help="fichier de séance (défaut : data/points_de_controle)")
    analyseur.add_argument("--recommencer", action="store_true",
                           help="efface la séance en cours et repart de zéro")
    analyseur.add_argument("--sans-dagger", action="store_true",
                           help="s'arrêter après le clonage, sans les trois tours")
    analyseur.add_argument("--perte", choices=("choix", "optimaux"), default=None,
                           help="cible de la perte (défaut : « choix », recette du document)")
    analyseur.add_argument("--taux-final", type=float, default=None,
                           help="décroissance du taux jusqu'à cette valeur (défaut : constant)")
    analyseur.add_argument("--tirage", choices=("hasard", "difficiles"), default=None,
                           help="concurrents du choix de l'expert (défaut : « hasard »)")
    analyseur.add_argument("--reference", default=None,
                           help="comparer-recettes : modèle de référence "
                                "(défaut : recette du document)")
    analyseur.add_argument("--plafond", type=int, default=None,
                           help="plafond de pièces enregistrées")
    analyseur.add_argument("--modele", help="point de contrôle à diffuser")
    analyseur.add_argument("--partie", help="partie enregistrée à mettre dans la page")
    analyseur.add_argument("--hote", default="127.0.0.1", help="adresse d'écoute")
    analyseur.add_argument("--port", type=int, default=8000, help="port d'écoute")
    arguments = analyseur.parse_args(argv)
    return COMMANDES[arguments.commande](arguments)


if __name__ == "__main__":
    sys.exit(main())
