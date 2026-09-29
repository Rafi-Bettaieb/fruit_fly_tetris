"""Un run en plusieurs séances : découper le temps, pas la recette (§14.6).

Un run complet — 2 400 mises à jour de clonage puis trois tours de DAgger — dure
7 h 30 sur la RTX 3050 (§14.3). Ce n'est pas un problème de calcul mais
d'organisation : il faut immobiliser la machine une nuit entière, et une coupure
à la sixième heure perdait tout.

**Le budget du document ne bouge pas.** 2 400 pas, trois tours, mêmes réglages.
Le raccourcir donnerait un autre modèle — légitime, mais qu'il faudrait publier
comme tel, avec son budget, à côté des autres (§3). Ce qui se découpe ici est le
temps d'occupation de la carte : trois heures ce soir, trois demain, jusqu'au
bout de la même recette.

**Ce qu'il faut enregistrer pour que la reprise soit exacte.** Les paramètres ne
suffisent pas. Sans l'état de l'optimiseur, Adam repart de moyennes nulles à
chaque séance. Sans l'état du générateur, chaque séance rejoue les mêmes lots que
le début du run, qui serait alors bien plus court en information qu'en
apparence — et cette erreur-là ne lève aucune exception : elle produit un modèle
médiocre qu'on croit entraîné. Les deux états sont dans le fichier de séance.

**Une séance s'arrête à un point de contrôle, jamais au milieu.** À cet instant
l'état de secours de la règle de divergence coïncide avec l'état courant, donc il
n'y a rien de plus à retenir. La granularité de l'arrêt est celle du point de
contrôle — 200 pas, environ 18 minutes — et la boucle s'arrête un peu avant la
limite demandée plutôt qu'un peu après.

**Ce qui distingue une séance d'une phase.** La phase appartient à la recette :
clonage, puis trois tours de DAgger. La séance appartient à l'emploi du temps.
Une séance peut finir une phase et en entamer une autre ; une phase peut prendre
trois séances. Le fichier enregistre la phase et l'avancement dans la phase, pas
le découpage en séances, qui n'a aucune conséquence sur le résultat.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from . import clonage, dagger
from .clonage import Journal, Reglages
from .demonstrations import Situation

CHEMIN = Path("data/points_de_controle/seance-mouche.pt")

PHASES_DAGGER = ("dagger-1", "dagger-2", "dagger-3")

COLLECTE_ESTIMEE = 30 * 60
"""Ce qu'on réserve pour une collecte DAgger avant de l'entamer (§14.3 : un tour
coûte 1 h 15, dont environ 45 min pour les 500 pas). Une collecte ne s'arrête pas
en cours de route : lancée à cinq minutes de la limite, elle ferait déborder la
séance d'une demi-heure."""


@dataclass
class Session:
    """L'état d'un run entre deux séances.

    Tout ce qui s'y trouve est nécessaire à une reprise exacte. Les gros
    tenseurs sont sur le CPU : le fichier pèse environ 400 Mio — les paramètres
    courants, les deux moments d'Adam, et le meilleur état retenu.
    """

    graine_run: int = 0
    budget_clonage: int = 0
    """Le nombre de mises à jour de clonage. Enregistré pour refuser une reprise
    sous un autre budget : un run à moitié fait à 2 400 pas et fini à 1 200 ne
    serait ni l'un ni l'autre."""
    avec_dagger: bool = True
    perte: str = "choix"
    taux_final: float | None = None
    tirage: str = "hasard"
    """La recette (`Reglages.perte`, `taux_final`, `tirage`), pour la même raison
    que le budget : une reprise sous une autre recette donnerait un modèle qui
    n'aurait suivi aucune des deux. Des valeurs par défaut simples, et non des
    fabriques : un fichier de séance écrit avant l'ajout de ces champs les
    retrouve ainsi à la relecture, avec la recette du document."""

    phase: str = "clonage"
    """« clonage », « dagger-1 » à « dagger-3 », puis « fini » — ou « echoue »
    si la règle de divergence a rendu les armes (§10.3)."""
    pas: int = 0

    parametres: dict | None = None
    """Les paramètres à poursuivre. Avec `optimiseur` à None, ils marquent un
    début de phase : la boucle repart d'eux avec un optimiseur neuf."""
    optimiseur: dict | None = None
    generateur: dict | None = None
    journal: Journal = field(default_factory=Journal)
    meilleur_etat: dict | None = None
    meilleur_accord: float = -1.0
    reference_perte: float | None = None

    dagger_collecte: list[Situation] = field(default_factory=list)
    """Les situations accumulées par DAgger, tous tours confondus (§10.4)."""
    dagger_collecte_faite: bool = False
    """Vrai quand la collecte du tour courant est faite. Elle coûte une
    demi-heure : on l'enregistre aussitôt, pour ne jamais la refaire."""

    phase_achevee: bool = False
    secondes: float = 0.0
    seances: int = 0
    historique: list[tuple[str, Journal]] = field(default_factory=list)

    @property
    def terminee(self) -> bool:
        return self.phase in ("fini", "echoue")

    @property
    def etiquette(self) -> str:
        if self.terminee:
            return self.phase
        return f"{self.phase}-{self.pas}pas"

    def avancement(self) -> str:
        """Une ligne lisible : où en est le run, et ce qui reste."""
        if self.phase == "fini":
            return f"run terminé · {self.seances} séances · {self.secondes / 3600:.1f} h"
        if self.phase == "echoue":
            return "run échoué (règle de divergence) — publié comme tel"
        if self.phase == "clonage":
            fait, total = self.pas, self.budget_clonage
        else:
            fait, total = self.pas, clonage.REGLAGES_MOUCHE_DAGGER.mises_a_jour
        accord = (f" · meilleur accord {self.meilleur_accord:.1%}"
                  if self.meilleur_accord >= 0 else "")
        return (f"{self.phase} {fait}/{total} pas{accord} · "
                f"{self.secondes / 3600:.1f} h de carte sur {self.seances} séances")


def recette(session: Session) -> dict:
    """Les réglages de recette d'une séance, prêts pour `Reglages(**…)`."""
    return {"perte": session.perte, "taux_final": session.taux_final, "tirage": session.tirage}


def sauvegarder(session: Session, chemin: Path = CHEMIN) -> Path:
    """Écrit la séance, en deux temps.

    Le fichier fait 400 Mio : une coupure de courant au milieu de l'écriture
    laisserait un fichier tronqué là où se trouvait le seul état du run. On écrit
    donc à côté, puis on renomme — un renommage est atomique, l'écriture ne
    l'est pas.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    provisoire = chemin.with_name(chemin.name + ".provisoire")
    torch.save(session, provisoire)
    os.replace(provisoire, chemin)
    return chemin


def charger(chemin: Path = CHEMIN) -> Session | None:
    """Relit une séance, ou None s'il n'y en a pas.

    `weights_only=False` est indispensable : le fichier contient un journal et
    des situations de DAgger, pas seulement des tenseurs. Il n'est lu que depuis
    `data/`, écrit par ce module.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        return None
    return torch.load(chemin, map_location="cpu", weights_only=False)


def _phase_suivante(session: Session) -> str:
    if session.phase == "clonage":
        return PHASES_DAGGER[0] if session.avec_dagger else "fini"
    rang = PHASES_DAGGER.index(session.phase)
    return PHASES_DAGGER[rang + 1] if rang + 1 < len(PHASES_DAGGER) else "fini"


def _passer_a_la_phase_suivante(session: Session, journal: Journal) -> None:
    """Clôt la phase courante et prépare la suivante.

    La phase suivante part du **meilleur** état de la précédente, avec un
    optimiseur, un générateur et un journal neufs : c'est exactement ce que fait
    un run d'une traite, où chaque appel à `entrainer` construit son propre Adam.
    """
    session.historique.append((session.phase, journal))
    session.phase = _phase_suivante(session)
    session.pas = 0
    session.parametres = session.meilleur_etat
    session.optimiseur = None
    session.generateur = None
    session.journal = Journal()
    session.meilleur_etat = None
    session.meilleur_accord = -1.0
    session.reference_perte = None
    session.dagger_collecte_faite = False


def poursuivre(
    modele: nn.Module,
    adaptateur,
    demonstrations: list[Situation],
    situations_de_test: list[Situation],
    session: Session,
    *,
    reglages: Reglages | None = None,
    reglages_dagger: Reglages = clonage.REGLAGES_MOUCHE_DAGGER,
    limite_secondes: float | None = None,
    peripherique: str = "cpu",
    nom: str = "mouche",
    chemin: Path = CHEMIN,
    tours=dagger.TOURS,
    plafond: int = dagger.PLAFOND,
) -> tuple[nn.Module, Session]:
    """Avance le run d'une séance et rend la main quand le temps est écoulé.

    Sans `limite_secondes`, va jusqu'au bout — c'est le run d'une traite. Avec,
    s'arrête au dernier point de contrôle qui tient dans le temps imparti, après
    avoir tout enregistré.
    """
    reglages = reglages or Reglages(mises_a_jour=session.budget_clonage, **recette(session))
    fournie = {"perte": reglages.perte, "taux_final": reglages.taux_final,
               "tirage": reglages.tirage}
    if fournie != recette(session):
        raise ValueError(
            f"recette de la séance : {recette(session)} ; réglages fournis : {fournie}")
    debut_seance = time.perf_counter()
    session.seances += 1

    def reste() -> float | None:
        if limite_secondes is None:
            return None
        return limite_secondes - (time.perf_counter() - debut_seance)

    # Le premier tour d'une séance avance toujours d'un cran, quelle que soit la
    # limite : sans cette règle, une séance trop courte ne ferait rien, et la
    # suivante non plus. Une séance dure donc au moins une fenêtre de points de
    # contrôle — environ 18 minutes sur le connectome complet.
    premier_tour = True
    while not session.terminee:
        restant = reste()
        if not premier_tour and restant is not None and restant <= 0:
            break
        if session.phase == "clonage":
            reglages_phase, graine = reglages, session.graine_run
            accumule = None
        else:
            numero = PHASES_DAGGER.index(session.phase) + 1
            reglages_phase, graine = reglages_dagger, session.graine_run + numero
            if not session.dagger_collecte_faite:
                if not premier_tour and restant is not None and restant < COLLECTE_ESTIMEE:
                    break
                # La collecte part du meilleur état de la phase précédente, comme
                # dans `dagger.executer` où le modèle passé au tour suivant est
                # celui que `entrainer` a rendu.
                if session.parametres is not None:
                    clonage.restaurer_parametres(modele, session.parametres)
                debut = time.perf_counter()
                # Une collecte prend une demi-heure ; bloquée après une mise en
                # veille, elle tournerait à vide indéfiniment (voir GARDE_SECONDES).
                clonage.armer_la_garde(4 * COLLECTE_ESTIMEE)
                try:
                    nouvelles = dagger.collecter(adaptateur(modele), tours[numero - 1], plafond)
                finally:
                    clonage.lever_la_garde()
                session.dagger_collecte.extend(nouvelles)
                session.dagger_collecte_faite = True
                sauvegarder(session, chemin)
                print(f"  [{nom} · {session.phase}] {len(nouvelles)} situations collectées "
                      f"en {(time.perf_counter() - debut) / 60:.0f} min "
                      f"({len(session.dagger_collecte)} au total)", flush=True)
            accumule = session.dagger_collecte

        premier_tour = False
        restant = reste()
        modele, journal = clonage.entrainer(
            modele, demonstrations, situations_de_test, reglages_phase,
            graine_run=graine, nom=f"{nom} · {session.phase}", peripherique=peripherique,
            situations_dagger=accumule, session=session, limite_secondes=restant,
            sauvegarde=lambda etat: sauvegarder(etat, chemin),
        )
        sauvegarder(session, chemin)

        if journal.echoue:
            session.historique.append((session.phase, journal))
            session.phase = "echoue"
            sauvegarder(session, chemin)
            break
        if not session.phase_achevee:
            break  # temps de séance écoulé, l'état est sur le disque
        _passer_a_la_phase_suivante(session, journal)
        sauvegarder(session, chemin)

    return modele, session
