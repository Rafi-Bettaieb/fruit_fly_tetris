"""Le clonage de comportement : la boucle que partagent tous les modèles.

Conception.md §10.3. Perte softmax sur les candidats d'une pièce, la cible étant
le choix de l'expert. Adam. Points de contrôle réguliers, le meilleur choisi sur
l'accord avec l'expert.

**Une seule boucle pour toutes les conditions.** Le juge linéaire et la mouche
passent par ici avec les mêmes réglages de structure et des budgets différents.
C'est ce qui rend leur comparaison honnête : ce qui diffère est le modèle, pas
la manière de l'entraîner.

**Réglages du document :**

| | Mouche | Juge linéaire |
|---|---|---|
| Candidats par situation | 10 | 10 |
| Lot | 4 situations | 64 situations |
| Adam | 0,04 | 0,001 |
| Mises à jour | 2 400 | 10 époques |

Le juge linéaire voit donc bien plus de situations que la mouche : l'écart est
assumé — chaque modèle est entraîné jusqu'à son mieux — mais il est publié à
côté des scores (§3).

**Pourquoi 2 400 mises à jour pour la mouche et non 1 200 comme la référence.**
Un pas porte ici 4 termes de perte, un par situation du lot. Dans la référence,
un pas en portait 64, car elle rétropropageait à travers 16 décisions
consécutives. À nombre de pas égal, le signal de gradient serait seize fois plus
faible (§2.3).

**Règle de divergence.** Perte invalide, ou en hausse de plus de 50 % sur
100 mises à jour : reprise du dernier point de contrôle avec un taux divisé par
4. Trois relances au maximum, ensuite le run est déclaré échoué et publié comme
tel — sans ce plafond, un run instable tourne en rond des heures et occupe la
carte pour rien.

**L'entraînement peut se dérouler en plusieurs séances.** `session` et
`limite_secondes` permettent d'arrêter la boucle à un point de contrôle et de la
reprendre exactement là, une autre fois. Le budget de mises à jour ne change
pas : c'est le temps d'occupation de la carte qui se découpe, pas la recette
(§14.6, module `reprise`).
"""

from __future__ import annotations

import faulthandler
import functools
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .reprise import Session

from .. import graines
from ..encodage import TAILLE_ENCODAGE, encoder_lot
from ..evaluation.accord import accord
from ..modeles.torche import NoteurTorch
from .demonstrations import Situation, tirer_candidats, tirer_difficiles


@dataclass
class Reglages:
    """Les réglages d'un entraînement. Un jeu par condition (§19)."""

    candidats_par_situation: int = 10
    lot: int = 4
    taux: float = 0.04
    mises_a_jour: int = 2400
    point_de_controle_tous_les: int = 200
    # Règle de divergence
    hausse_max: float = 0.5
    fenetre_divergence: int = 100
    diviseur_du_taux: int = 4
    relances_max: int = 3
    # Les deux corrections à l'essai (§10.3). Par défaut, la recette du document.
    perte: str = "choix"
    """« choix » : la cible est le placement que l'expert a tiré au sort parmi
    ses optimaux. « optimaux » : la cible est l'ensemble de ses optimaux. Dans
    16,4 % des situations l'expert en a plusieurs, et avec « choix » un autre
    optimal tiré parmi les concurrents — 10,2 % des situations d'un lot — est
    puni comme une erreur, alors que l'accord le compte juste."""
    taux_final: float | None = None
    """None : taux constant. Sinon, décroissance en cosinus de `taux` à
    `taux_final` sur les `mises_a_jour` pas de la phase."""
    tirage: str = "hasard"
    """« hasard » : les concurrents du choix de l'expert sont tirés au hasard.
    « difficiles » : ce sont ceux que le modèle note le plus haut
    (`demonstrations.tirer_difficiles`). Coût : un passage avant sans gradient
    sur tous les candidats du lot, environ un dixième d'un pas."""

    def __post_init__(self) -> None:
        if self.perte not in ("choix", "optimaux"):
            raise ValueError(f"perte « {self.perte} » inconnue : « choix » ou « optimaux »")
        if self.tirage not in ("hasard", "difficiles"):
            raise ValueError(f"tirage « {self.tirage} » inconnu : « hasard » ou « difficiles »")


def taux_du_pas(reglages: Reglages, pas: int, relances: int = 0) -> float:
    """Le taux d'apprentissage du pas `pas` (compté à partir de 0).

    Fonction du seul numéro de pas et du nombre de relances : une reprise après
    une séance interrompue retrouve donc exactement le même taux, sans rien
    enregistrer de plus. Chaque relance de la règle de divergence le divise par
    `diviseur_du_taux`, avec ou sans décroissance.
    """
    reduction = reglages.diviseur_du_taux ** relances
    if reglages.taux_final is None:
        return reglages.taux / reduction
    avancee = min(1.0, pas / max(1, reglages.mises_a_jour))
    cosinus = 0.5 * (1 + math.cos(math.pi * avancee))
    return (reglages.taux_final + (reglages.taux - reglages.taux_final) * cosinus) / reduction


GARDE_SECONDES = 15 * 60
"""Au-delà, un pas n'est plus lent : il est bloqué.

Un pas prend 5,2 s, un point de contrôle environ une minute. Le cas qui a fixé
cette règle : le capot rabattu, le portable mis en veille, et au réveil un
contexte CUDA perdu. Le processus attendait alors indéfiniment un signal que la
carte n'enverrait plus — à 100 % d'un cœur, sans erreur, sans une ligne dans le
journal, et en gardant le verrou GPU. Le chien de garde transforme ce silence en
une pile d'appels dans le journal et un arrêt net ; la séance suivante reprend
au dernier point de contrôle."""


def armer_la_garde(secondes: float | None = None) -> None:
    """Arrête le processus, pile d'appels à l'appui, si rien ne la réarme d'ici là."""
    flux = sys.__stderr__
    if flux is not None:
        faulthandler.dump_traceback_later(secondes or GARDE_SECONDES, exit=True, file=flux)


def lever_la_garde() -> None:
    faulthandler.cancel_dump_traceback_later()


def _sous_garde(fonction):
    """Lève la garde à la sortie, exception comprise : oubliée armée, elle
    tuerait un quart d'heure plus tard un processus qui fait autre chose."""

    @functools.wraps(fonction)
    def gardee(*args, **kwargs):
        try:
            return fonction(*args, **kwargs)
        finally:
            lever_la_garde()

    return gardee


REGLAGES_MOUCHE = Reglages()
REGLAGES_MOUCHE_DAGGER = Reglages(mises_a_jour=500)

# 10 époques sur les 101 017 situations, par lots de 64 : 101017 / 64 × 10 ≈ 15 800.
REGLAGES_LINEAIRE = Reglages(
    lot=64, taux=0.001, mises_a_jour=15_800, point_de_controle_tous_les=1_000
)
# 3 époques par tour de DAgger, même calcul.
REGLAGES_LINEAIRE_DAGGER = Reglages(
    lot=64, taux=0.001, mises_a_jour=4_700, point_de_controle_tous_les=1_000
)


@dataclass
class Journal:
    """Ce qu'on suit pendant l'entraînement, et qui part dans MLflow (§10.3)."""

    pertes: list[float] = field(default_factory=list)
    accords: list[tuple[int, float]] = field(default_factory=list)
    relances: int = 0
    echoue: bool = False

    @property
    def meilleur_accord(self) -> float:
        return max((valeur for _, valeur in self.accords), default=0.0)


def _sur_cpu(valeur):
    """Copie récursive d'un état vers la RAM, tenseurs compris."""
    if torch.is_tensor(valeur):
        return valeur.detach().to("cpu", copy=True)
    if isinstance(valeur, dict):
        return {cle: _sur_cpu(sous) for cle, sous in valeur.items()}
    if isinstance(valeur, (list, tuple)):
        return type(valeur)(_sur_cpu(sous) for sous in valeur)
    return valeur


def capturer_parametres(modele: nn.Module) -> dict[str, torch.Tensor]:
    """Un point de contrôle ne garde que ce qui s'entraîne, et le garde en RAM.

    Copier le `state_dict` complet sur le GPU coûtait 587 Mio par copie, dont
    489 de **buffers figés** — la topologie CSR, les poids mesurés, les
    interfaces — qui ne changent jamais. Avec deux copies conservées, cela
    faisait 1,2 Gio de VRAM gelée avant la première mise à jour, sur une carte
    qui en a 3,68. L'entraînement dépassait la mémoire dès le premier pas.
    """
    return {nom: p.detach().to("cpu", copy=True) for nom, p in modele.named_parameters()}


def restaurer_parametres(modele: nn.Module, etat: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for nom, parametre in modele.named_parameters():
            parametre.copy_(etat[nom].to(parametre.device))


@torch.no_grad()
def noter_tous_les_candidats(modele: nn.Module, situations: list[Situation],
                             peripherique: str = "cpu") -> list[np.ndarray]:
    """Les notes du modèle sur tous les candidats de chaque situation, sans gradient.

    Une situation à la fois : 34 candidats au plus, la mémoire d'une pièce en
    jeu. Tout le lot d'un coup — jusqu'à 136 évaluations — dépasserait ce que la
    carte peut ajouter à un pas d'entraînement.
    """
    return [
        modele(torch.from_numpy(encoder_lot(s.candidats)).to(peripherique))
        .reshape(-1).float().cpu().numpy()
        for s in situations
    ]


def preparer_lot(
    situations: list[Situation],
    reglages: Reglages,
    generateur: np.random.Generator,
    peripherique: str = "cpu",
    notes: list[np.ndarray] | None = None,
):
    """Construit un lot : encodages, masque et cibles.

    Les situations n'ont pas toutes le même nombre de candidats — une grille
    haute en laisse moins qu'une grille vide. On complète donc jusqu'à la plus
    grande et l'on masque le reste : les places inventées reçoivent −∞ avant la
    softmax, et ne peuvent donc ni être choisies ni recevoir de gradient.
    """
    if reglages.tirage == "difficiles":
        if notes is None:
            raise ValueError("tirage « difficiles » : il faut les notes du modèle sur le lot")
        lots = [tirer_difficiles(s, reglages.candidats_par_situation, n, generateur)
                for s, n in zip(situations, notes, strict=True)]
    else:
        lots = [tirer_candidats(s, reglages.candidats_par_situation, generateur)
                for s in situations]
    largeur = max(len(candidats) for candidats, _ in lots)

    encodages = np.zeros((len(lots), largeur, TAILLE_ENCODAGE), dtype=np.float32)
    masque = np.zeros((len(lots), largeur), dtype=bool)
    cibles = np.empty(len(lots), dtype=np.int64)
    optimaux = np.zeros((len(lots), largeur), dtype=bool)

    for indice, ((candidats, cible), situation) in enumerate(zip(lots, situations, strict=True)):
        encodages[indice, : len(candidats)] = encoder_lot(candidats)
        masque[indice, : len(candidats)] = True
        cibles[indice] = cible
        # Un placement se reconnaît à sa rotation et sa colonne : deux candidats
        # d'une même situation ne peuvent pas partager les deux.
        optimaux_expert = {(situation.candidats[j].rotation, situation.candidats[j].colonne)
                           for j in situation.meilleurs_expert}
        for position, candidat in enumerate(candidats):
            optimaux[indice, position] = (candidat.rotation, candidat.colonne) in optimaux_expert

    return (
        torch.from_numpy(encodages).to(peripherique),
        torch.from_numpy(masque).to(peripherique),
        torch.from_numpy(cibles).to(peripherique),
        torch.from_numpy(optimaux).to(peripherique),
    )


def perte_du_lot(modele: nn.Module, encodages, masque, cibles, optimaux=None,
                 perte: str = "choix"):
    """Softmax sur les candidats d'une pièce.

    « choix » : entropie croisée vers le placement tiré par l'expert.
    « optimaux » : − log de la probabilité totale donnée aux optimaux de
    l'expert. Avec un seul optimal — 83,6 % des situations — les deux sont
    identiques ; avec plusieurs, seule la seconde laisse le modèle choisir
    librement entre des placements que l'expert note exactement pareil.
    """
    notes = modele(encodages.reshape(-1, TAILLE_ENCODAGE)).reshape(masque.shape)
    notes = notes.masked_fill(~masque, float("-inf"))
    if perte == "choix":
        return F.cross_entropy(notes, cibles)
    log_probabilites = F.log_softmax(notes, dim=1)
    return -torch.logsumexp(log_probabilites.masked_fill(~optimaux, float("-inf")), dim=1).mean()


@_sous_garde
def entrainer(
    modele: nn.Module,
    situations: list[Situation],
    situations_de_test: list[Situation],
    reglages: Reglages = REGLAGES_MOUCHE,
    graine_run: int = 0,
    nom: str = "modèle",
    peripherique: str = "cpu",
    situations_dagger: list[Situation] | None = None,
    bavard: bool = True,
    session: Session | None = None,
    limite_secondes: float | None = None,
    sauvegarde: Callable[[Session], None] | None = None,
) -> tuple[nn.Module, Journal]:
    """Entraîne un modèle par clonage et rend sa meilleure version.

    « Meilleure » au sens de l'accord avec l'expert sur les situations de test
    fournies — jamais au sens de la perte, qui peut continuer à baisser pendant
    que le modèle joue moins bien.

    Avec `session`, l'état complet de la boucle est déposé à chaque point de
    contrôle, et `sauvegarde` l'écrit aussitôt sur le disque ; avec
    `limite_secondes`, la boucle rend la main au premier point de contrôle dont le
    suivant ne tiendrait plus dans le temps imparti. Relancer la fonction avec la
    même session reprend exactement où elle s'était arrêtée.
    """
    modele = modele.to(peripherique)
    depart = time.perf_counter()
    generateur = graines.generateur_run(graine_run)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=reglages.taux)
    journal = Journal()

    meilleur_etat = capturer_parametres(modele)
    meilleur_accord = -1.0
    dernier_point = (meilleur_etat, _sur_cpu(optimiseur.state_dict()))
    reference_perte: float | None = None
    pas = 0

    if session is not None and session.parametres is not None:
        restaurer_parametres(modele, session.parametres)
        if session.optimiseur is not None:
            # Reprise en cours de phase. Les paramètres seuls ne suffiraient pas :
            # sans l'état d'Adam, les moyennes mobiles repartent de zéro et le
            # premier pas de la séance est un pas à l'aveugle ; sans l'état du
            # générateur, la séance rejoue exactement les lots du début, et le run
            # serait plus court en information qu'en apparence.
            optimiseur.load_state_dict(session.optimiseur)
            generateur.bit_generator.state = session.generateur
            journal, pas = session.journal, session.pas
            meilleur_etat = session.meilleur_etat or meilleur_etat
            meilleur_accord = session.meilleur_accord
            reference_perte = session.reference_perte
            # Une séance s'arrête toujours à un point de contrôle : l'état de
            # secours de la règle de divergence y coïncide avec l'état courant.
            dernier_point = (session.parametres, session.optimiseur)
    pas_initial = pas
    interrompue = False

    while pas < reglages.mises_a_jour:
        armer_la_garde()
        if situations_dagger:
            # Moitié DAgger, moitié démonstrations (§10.4). Un tirage uniforme
            # sur l'union noierait les situations de DAgger, quatre fois moins
            # nombreuses que les démonstrations.
            moitie = reglages.lot // 2
            lot = [situations[i] for i in generateur.integers(0, len(situations),
                                                              size=reglages.lot - moitie)]
            lot += [situations_dagger[i] for i in generateur.integers(0, len(situations_dagger),
                                                                      size=moitie)]
        else:
            lot = [situations[i] for i in generateur.integers(0, len(situations),
                                                              size=reglages.lot)]
        notes = (noter_tous_les_candidats(modele, lot, peripherique)
                 if reglages.tirage == "difficiles" else None)
        encodages, masque, cibles, optimaux = preparer_lot(
            lot, reglages, generateur, peripherique, notes)

        perte = perte_du_lot(modele, encodages, masque, cibles, optimaux, reglages.perte)
        optimiseur.zero_grad(set_to_none=True)
        perte.backward()
        if reglages.taux_final is not None:
            for groupe in optimiseur.param_groups:
                groupe["lr"] = taux_du_pas(reglages, pas, journal.relances)
        optimiseur.step()

        valeur = float(perte.item())
        journal.pertes.append(valeur)
        pas += 1

        # La toute première perte sert de référence à la première fenêtre. Sans
        # elle, un entraînement qui explose dès les premiers pas — de 2,3 à
        # 3 × 10⁹ avec un taux absurde — n'aurait rien à quoi se comparer au
        # premier relevé, et la divergence passerait inaperçue.
        if reference_perte is None and math.isfinite(valeur):
            reference_perte = valeur

        # --- Divergence ---
        # On compare la **moyenne** de la dernière fenêtre à celle de la
        # précédente. Comparer deux pertes isolées, prises tous les cent pas,
        # laisserait passer un entraînement qui oscille entre deux valeurs
        # énormes sans jamais monter de 50 % pile au moment du relevé.
        diverge = math.isnan(valeur) or math.isinf(valeur)
        if not diverge and pas % reglages.fenetre_divergence == 0:
            fenetre = journal.pertes[-reglages.fenetre_divergence :]
            moyenne = float(np.mean(fenetre))
            if reference_perte is not None and moyenne > reference_perte * (
                1 + reglages.hausse_max
            ):
                diverge = True
            reference_perte = moyenne
        if diverge:
            if journal.relances >= reglages.relances_max:
                journal.echoue = True
                break
            journal.relances += 1
            if bavard:
                print(f"  [{nom}] divergence au pas {pas} — relance "
                      f"{journal.relances}/{reglages.relances_max}, taux divisé par "
                      f"{reglages.diviseur_du_taux}", flush=True)
            etat_modele, etat_optimiseur = dernier_point
            restaurer_parametres(modele, etat_modele)
            optimiseur.load_state_dict(etat_optimiseur)
            for groupe in optimiseur.param_groups:
                groupe["lr"] /= reglages.diviseur_du_taux
            reference_perte = None
            continue

        # --- Point de contrôle ---
        if pas % reglages.point_de_controle_tous_les == 0 or pas == reglages.mises_a_jour:
            dernier_point = (capturer_parametres(modele), _sur_cpu(optimiseur.state_dict()))
            score = accord(NoteurTorch(modele, nom, peripherique), situations_de_test)
            modele.train()
            journal.accords.append((pas, score))
            if score > meilleur_accord:
                meilleur_accord = score
                meilleur_etat = capturer_parametres(modele)

            ecoule = time.perf_counter() - depart
            # Le rythme se mesure sur la séance en cours, pas depuis le premier
            # pas du run : sur une reprise, `pas` porte l'histoire des séances
            # précédentes et donnerait une vitesse fantaisiste.
            par_pas = ecoule / max(1, pas - pas_initial)
            if bavard:
                # Un run complet dure des heures : sans cette ligne, on ne sait
                # pas s'il progresse, s'il diverge, ni quand il finira.
                perte_recente = float(np.mean(journal.pertes[-50:]))
                print(
                    f"  [{nom}] {pas}/{reglages.mises_a_jour} · accord {score:.1%} · "
                    f"perte {perte_recente:.3f} · {ecoule / 60:.0f} min de séance, "
                    f"~{par_pas * (reglages.mises_a_jour - pas) / 60:.0f} min "
                    f"jusqu'au bout de la phase",
                    flush=True,
                )

            if session is not None:
                session.pas = pas
                session.parametres, session.optimiseur = dernier_point
                session.generateur = generateur.bit_generator.state
                session.journal = journal
                session.meilleur_etat = meilleur_etat
                session.meilleur_accord = meilleur_accord
                session.reference_perte = reference_perte
                if sauvegarde is not None:
                    # À chaque point de contrôle, pas seulement en fin de séance :
                    # un plantage ou une coupure à la troisième heure ne coûte
                    # alors que la fenêtre en cours, pas la séance entière.
                    sauvegarde(session)

            if limite_secondes is not None and pas < reglages.mises_a_jour:
                # On s'arrête **avant** de dépasser, pas après : entamer une
                # fenêtre qu'on ne finira pas la ferait recommencer entièrement
                # à la séance suivante.
                if ecoule + par_pas * reglages.point_de_controle_tous_les > limite_secondes:
                    interrompue = True
                    if bavard:
                        print(f"  [{nom}] séance close à {pas}/{reglages.mises_a_jour} pas "
                              f"— la suite reprendra ici", flush=True)
                    break

    if session is not None:
        session.phase_achevee = not interrompue
        session.secondes += time.perf_counter() - depart
    restaurer_parametres(modele, meilleur_etat)
    return modele, journal
