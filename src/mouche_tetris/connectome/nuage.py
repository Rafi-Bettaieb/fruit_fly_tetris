"""Le nuage de neurones de la démo : ce qui s'active quand la mouche décide (§7.5, §13.2).

Pour chaque pièce, la mouche note chaque grille possible en la faisant traverser
par le réseau, en K = 7 mises à jour. Le nuage montre ces mises à jour pour la
grille qu'elle a choisie : la vague part des neurones sensoriels, envahit le
cerveau, puis atteint les neurones moteurs de la corde nerveuse ventrale, dont
la note est lue. Mesuré sur le modèle en recette corrigée : 2,5 % des neurones
actifs à la première mise à jour, 37 % à la troisième, 94 % à la septième ; les
neurones moteurs ne s'allument qu'à la cinquième.

**Ce qui est affiché, et comment c'est choisi.** 8 000 neurones, tirés une fois
avec la graine du run (`graines.generateur_visualisation`) :

- les 703 neurones moteurs qui ont un soma — la sortie, dont la note est lue ;
  5 des 708 n'en ont pas dans les annotations ;
- 1 000 des 4 114 neurones sensoriels du lobe optique — l'entrée, qui lit la
  grille ;
- les 6 297 autres, tirés au hasard parmi les neurones placés.

La composition n'est pas celle d'un tirage uniforme, qui ne garderait que 86
neurones moteurs et une centaine de sensoriels : on ne verrait ni où le signal
entre, ni où il est lu. Elle est affichée telle quelle dans la légende.

**Les positions sont celles du soma**, en voxels de 8 nm dans le repère du
MaleCNS, pour 140 024 des 165 122 neurones. Les neurones sensoriels du lobe
optique n'en ont pas — 28 sur 4 114 : leur corps cellulaire est dans l'œil,
hors du volume imagé. Ils sont placés au **centre de leurs cibles**, pondéré par
le nombre de synapses, ce qui les met dans le lobe optique où arrivent leurs
axones. C'est une position calculée, pas mesurée, et la page le dit.

**Chaque neurone est rapporté à sa propre plage** (§13.2), mesurée une fois sur
des grilles de calibration : sans cela, les quelques neurones très actifs
écraseraient les autres et le nuage paraîtrait éteint. Un octet par neurone et
par mise à jour.

**Ce que ce n'est pas.** L'état d'un modèle à taux contraint par le câblage
mesuré — pas une mesure d'activité sur une vraie mouche.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import graines
from ..encodage import encoder_lot
from ..modeles.torche import NoteurTorch
from .graphe import Graphe

TAILLE = 8_000
ENTREES_AFFICHEES = 1_000
AUTRE, ENTREE, SORTIE = 0, 1, 2
UM_PAR_VOXEL = 0.008
SEUIL_DE_SILENCE = 0.02
"""Un neurone dont l'état ne dépasse jamais ce seuil en calibration est tenu pour
silencieux : on ne l'étire pas sur toute l'échelle des couleurs, sinon son bruit
de fond serait peint comme une activité franche."""

ENTETE = b"NUAGE1"


def chemin(graine_run: int) -> Path:
    return Path(f"data/prepare/nuage-graine{graine_run}.npz")


@dataclass
class Nuage:
    indices: np.ndarray
    """Indices des neurones affichés dans le graphe, triés."""
    positions: np.ndarray
    """(N, 3) en micromètres, repère du MaleCNS : x latéral, y dorso-ventral,
    z antéro-postérieur — le cerveau à l'avant, la corde nerveuse à l'arrière."""
    roles: np.ndarray
    """AUTRE, ENTREE ou SORTIE, pour chaque neurone affiché."""
    graine_run: int

    @property
    def n(self) -> int:
        return len(self.indices)

    def empreinte(self) -> str:
        """Identifie le tirage : la page et le serveur vérifient qu'ils parlent
        des mêmes neurones, dans le même ordre, avant d'afficher quoi que ce soit."""
        return hashlib.sha256(np.asarray(self.indices, dtype="<i8").tobytes()).hexdigest()[:12]

    def composition(self) -> dict[str, int]:
        return {"entrees": int((self.roles == ENTREE).sum()),
                "sorties": int((self.roles == SORTIE).sum()),
                "autres": int((self.roles == AUTRE).sum())}


def positions_des_neurones(graphe: Graphe, somas: list) -> np.ndarray:
    """(n_neurones, 3) en micromètres, NaN pour un neurone sans position.

    `somas` est aligné sur les indices du graphe — c'est l'ordre de
    `acquisition.lire_les_neurones`, celui qui fonde les indices denses. Les
    neurones d'entrée sans soma sont placés au centre de leurs cibles, pondéré
    par le nombre de synapses.
    """
    positions = np.full((graphe.n_neurones, 3), np.nan)
    for indice, soma in enumerate(somas):
        if soma is not None and len(soma) == 3:
            positions[indice] = soma
    positions *= UM_PAR_VOXEL

    placees = ~np.isnan(positions).any(axis=1)
    a_placer = np.zeros(graphe.n_neurones, dtype=bool)
    a_placer[graphe.entrees] = True
    a_placer &= ~placees
    arete = a_placer[graphe.pre] & placees[graphe.post]
    source = graphe.pre[arete]
    poids = graphe.synapses[arete].astype(np.float64)
    cibles = positions[graphe.post[arete]]
    totaux = np.bincount(source, weights=poids, minlength=graphe.n_neurones)
    calculees = a_placer & (totaux > 0)
    for axe in range(3):
        somme = np.bincount(source, weights=poids * cibles[:, axe], minlength=graphe.n_neurones)
        positions[calculees, axe] = somme[calculees] / totaux[calculees]
    return positions


def construire(graphe: Graphe, positions: np.ndarray, graine_run: int,
               taille: int = TAILLE, entrees: int = ENTREES_AFFICHEES) -> Nuage:
    """Tire les neurones affichés, une fois pour toutes, avec la graine du run."""
    generateur = graines.generateur_visualisation(graine_run)
    placees = ~np.isnan(positions).any(axis=1)

    sorties = graphe.sorties[placees[graphe.sorties]]
    entrees_placees = graphe.entrees[placees[graphe.entrees]]
    entrees_tirees = generateur.choice(entrees_placees, size=min(entrees, len(entrees_placees)),
                                       replace=False)
    exclus = np.zeros(graphe.n_neurones, dtype=bool)
    exclus[graphe.entrees] = True
    exclus[graphe.sorties] = True
    autres_placees = np.flatnonzero(placees & ~exclus)
    reste = max(0, taille - len(sorties) - len(entrees_tirees))
    autres = generateur.choice(autres_placees, size=min(reste, len(autres_placees)), replace=False)

    indices = np.sort(np.concatenate([sorties, entrees_tirees, autres]).astype(np.int64))
    roles = np.full(len(indices), AUTRE, dtype=np.uint8)
    roles[np.isin(indices, graphe.entrees)] = ENTREE
    roles[np.isin(indices, graphe.sorties)] = SORTIE
    return Nuage(indices=indices, positions=positions[indices].astype(np.float32),
                 roles=roles, graine_run=graine_run)


def sauvegarder(nuage: Nuage, fichier: Path | None = None) -> Path:
    fichier = Path(fichier or chemin(nuage.graine_run))
    fichier.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fichier, indices=nuage.indices, positions=nuage.positions, roles=nuage.roles,
             graine_run=nuage.graine_run)
    return fichier


def charger(fichier: Path) -> Nuage:
    with np.load(fichier) as donnees:
        return Nuage(indices=donnees["indices"], positions=donnees["positions"],
                     roles=donnees["roles"], graine_run=int(donnees["graine_run"]))


def obtenir(graphe: Graphe, graine_run: int) -> Nuage:
    """Le nuage du run : relu s'il existe, sinon tiré et enregistré.

    Le tirer à chaque lancement donnerait le même résultat, mais relire les
    annotations coûte une dizaine de secondes ; et surtout, le fichier est ce
    qui garantit que la page et le serveur montrent les mêmes neurones.
    """
    fichier = chemin(graine_run)
    if fichier.exists():
        nuage = charger(fichier)
        if nuage.indices.max(initial=0) < graphe.n_neurones:
            return nuage
    from .acquisition import lire_les_neurones

    nuage = construire(graphe, positions_des_neurones(graphe, lire_les_neurones()["somas"]),
                       graine_run)
    sauvegarder(nuage, fichier)
    return nuage


def pour_la_page(nuage: Nuage) -> bytes:
    """Le nuage en binaire pour la page : positions sur 16 bits, rôles sur 8.

    Format `NUAGE1`, petit-boutiste : en-tête, nombre de neurones, empreinte
    (12 caractères), centre et demi-étendue en micromètres ; puis les positions
    rapportées à [−1, 1] sur la plus grande demi-étendue — les proportions du
    système nerveux sont conservées ; enfin un octet de rôle par neurone.
    """
    positions = nuage.positions.astype(np.float64)
    centre = (positions.max(axis=0) + positions.min(axis=0)) / 2
    demi = float(np.abs(positions - centre).max()) or 1.0
    reduites = np.round((positions - centre) / demi * 32767).astype("<i2")
    return b"".join([
        ENTETE,
        struct.pack("<I", nuage.n),
        nuage.empreinte().encode("ascii"),
        struct.pack("<4f", *centre, demi),
        reduites.tobytes(),
        nuage.roles.astype(np.uint8).tobytes(),
    ])


def quantifier(valeurs: np.ndarray) -> np.ndarray:
    """[−1, 1] → un octet : 128 au repos, 255 au maximum positif, 1 au négatif."""
    return np.clip(np.round(valeurs * 127) + 128, 0, 255).astype(np.uint8)


class Activite:
    """L'activité des neurones affichés, rapportée à leur plage propre."""

    def __init__(self, modele, nuage: Nuage, calibration: list, peripherique: str = "cpu",
                 lot: int = 16) -> None:
        """`calibration` : des candidats dont les grilles fixent la plage de chaque neurone."""
        self.modele = modele
        self.nuage = nuage
        self.peripherique = peripherique
        self.indices = torch.as_tensor(nuage.indices, device=peripherique)
        maximum = torch.zeros(nuage.n, device=peripherique)
        for debut in range(0, len(calibration), lot):
            encodages = torch.from_numpy(encoder_lot(tuple(calibration[debut:debut + lot])))
            trajet = modele.trajectoire(encodages.to(peripherique), self.indices)
            maximum = torch.maximum(maximum, trajet.abs().amax(dim=(0, 1)))
        self.echelles = torch.where(maximum < SEUIL_DE_SILENCE, torch.ones_like(maximum), maximum)

    @property
    def k(self) -> int:
        return self.modele.k

    def etapes(self, candidat) -> np.ndarray:
        """(K, N) octets : chaque mise à jour, pour la grille de ce candidat."""
        encodage = torch.from_numpy(encoder_lot((candidat,))).to(self.peripherique)
        trajet = self.modele.trajectoire(encodage, self.indices)[0] / self.echelles
        return quantifier(trajet.float().cpu().numpy())


class NoteurAvecNuage(NoteurTorch):
    """Un `NoteurTorch` qui sait aussi dire ce qui s'est activé.

    `activite` est une capacité facultative de l'interface `Noteur` : le serveur
    et l'enregistrement la demandent si elle existe, et s'en passent sinon —
    l'expert, le hasard et le juge linéaire n'ont pas d'état interne à montrer.
    """

    def __init__(self, modele, nom: str, peripherique: str, activite: Activite) -> None:
        super().__init__(modele, nom, peripherique)
        self._activite = activite

    def description_du_nuage(self) -> dict:
        nuage = self._activite.nuage
        return {"n": nuage.n, "k": self._activite.k, "empreinte": nuage.empreinte(),
                "graine_run": nuage.graine_run, "composition": nuage.composition()}

    def activite(self, candidat) -> dict:
        octets = self._activite.etapes(candidat)
        return {"k": int(octets.shape[0]), "n": int(octets.shape[1]),
                "etapes": base64.b64encode(octets.tobytes()).decode("ascii")}
