"""Lecture des fichiers MaleCNS v1.0 (conception.md §4.2, §7.1).

Trois fichiers, 1,2 Go au total, licence CC-BY 4.0, téléchargeables directement
depuis `storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`.

| Fichier | Taille | Ce qu'on en tire |
|---|---|---|
| `body-annotations-…` | 13 Mo | Neurones tracés, populations, positions de soma |
| `body-neurotransmitters-…` | 42 Mo | Signes (loi de Dale) |
| `connectome-weights-…` | 1,1 Go | Le graphe |

**Le critère « tracé » est la colonne `status`, pas `statusLabel`.** Les deux
existent et ne disent pas la même chose. `statusLabel` est fin — « Roughly
traced », « Reviewed », « Prelim Roughly traced »… — et aucune combinaison de
ses valeurs ne redonne les chiffres de la référence. `status` est grossier, et
`status == "Traced"` donne exactement les trois nombres du document :

| | Attendu | Vérifié |
|---|---|---|
| Neurones tracés | 165 122 | ✓ |
| `superclass == "ol_sensory"` ∩ tracés | 4 114 | ✓ |
| `superclass == "vnc_motor"` ∩ tracés | 708 | ✓ |

Les deux populations sont bien disjointes, comme l'exige §9.3.

**Le seul moment critique pour la RAM.** Le fichier de poids contient environ
152 millions de lignes, fragments compris, alors qu'on n'en garde que 25,6
millions — celles dont les deux extrémités sont des neurones tracés. Chargé d'un
coup dans pandas, il peut saturer les 24 Go de la machine. On lit donc par
morceaux et on filtre avant toute matérialisation.

**Le résultat attendu est un contrat, pas une indication.** Si la préparation
donne autre chose, quelque chose a changé en amont, et le document est clair
(§18) : on s'arrête plutôt que de continuer sur un graphe qui n'est pas celui de
la référence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

NEURONES_ATTENDUS = 165_122
CONNEXIONS_ATTENDUES = 25_563_197
ENTREES_ATTENDUES = 4_114
SORTIES_ATTENDUES = 708

BRUT = Path("data/brut")
PREPARE = Path("data/prepare")
RACINE_DISTANTE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
)

FICHIERS = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmetteurs": "body-neurotransmitters-male-cns-v1.0.feather",
    "poids": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
TAILLES_ATTENDUES_MO = {"annotations": 13, "neurotransmetteurs": 42, "poids": 1100}

# Colonnes vérifiées sur le fichier réel.
COLONNE_IDENTIFIANT = "bodyId"
COLONNE_STATUT = "status"
COLONNE_POPULATION = "superclass"
COLONNE_SOMA = "somaLocation"

STATUT_TRACE = "Traced"
POPULATION_ENTREE = "ol_sensory"
POPULATION_SORTIE = "vnc_motor"


class DonneesIncoherentes(RuntimeError):
    """Le graphe préparé ne correspond pas à celui de la référence."""


@dataclass
class Schema:
    """Ce qu'on a trouvé dans un fichier, sans charger ses données."""

    fichier: str
    lignes: int
    colonnes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.fichier} : {self.lignes} lignes, {len(self.colonnes)} colonnes"


def url(nom: str) -> str:
    return f"{RACINE_DISTANTE}/{FICHIERS[nom]}"


def chemin(nom: str, racine: Path = BRUT) -> Path:
    return Path(racine) / FICHIERS[nom]


def inspecter(chemin_fichier: Path) -> Schema:
    """Lit les métadonnées d'un fichier Feather sans charger ses données."""
    import pyarrow.feather as feather

    table = feather.read_table(chemin_fichier, memory_map=True)
    return Schema(
        fichier=Path(chemin_fichier).name,
        lignes=table.num_rows,
        colonnes=list(table.column_names),
    )


def inspecter_tout(racine: Path = BRUT) -> list[Schema]:
    schemas = []
    for nom in FICHIERS:
        fichier = chemin(nom, racine)
        if fichier.exists():
            schemas.append(inspecter(fichier))
        else:
            schemas.append(Schema(fichier=f"{nom} — absent : {fichier}", lignes=0))
    return schemas


def lire_les_neurones(racine: Path = BRUT):
    """Les neurones tracés, avec leurs populations et leurs positions de soma.

    Rend un dictionnaire de tableaux NumPy alignés sur `identifiants`.
    """
    import numpy as np
    import pyarrow.feather as feather

    table = feather.read_table(chemin("annotations", racine), memory_map=True)
    statut = np.asarray(table.column(COLONNE_STATUT).to_pylist(), dtype=object).astype(str)
    traces = statut == STATUT_TRACE

    population = np.asarray(table.column(COLONNE_POPULATION).to_pylist(), dtype=object)
    identifiants = np.asarray(table.column(COLONNE_IDENTIFIANT).to_pylist(), dtype=np.int64)
    somas = table.column(COLONNE_SOMA).to_pylist()

    return {
        "identifiants": identifiants[traces],
        "entrees": identifiants[traces & (population == POPULATION_ENTREE)],
        "sorties": identifiants[traces & (population == POPULATION_SORTIE)],
        "somas": [soma for soma, garde in zip(somas, traces, strict=True) if garde],
    }


def verifier_le_graphe(n_neurones: int, n_connexions: int, n_entrees: int, n_sorties: int) -> None:
    """Le contrôle sans appel de §7.1.

    Quatre nombres, tous connus d'avance. S'ils ne tombent pas, on ne continue
    pas : les résultats du projet se comparent à ceux de la référence, et une
    comparaison n'a de sens que sur le même graphe.
    """
    ecarts = []
    for nom, obtenu, attendu in (
        ("neurones", n_neurones, NEURONES_ATTENDUS),
        ("connexions", n_connexions, CONNEXIONS_ATTENDUES),
        ("neurones d'entrée", n_entrees, ENTREES_ATTENDUES),
        ("neurones moteurs", n_sorties, SORTIES_ATTENDUES),
    ):
        if obtenu != attendu:
            ecarts.append(f"  {nom} : {obtenu} au lieu de {attendu}")
    if ecarts:
        raise DonneesIncoherentes(
            "Le graphe préparé n'est pas celui de la référence :\n"
            + "\n".join(ecarts)
            + "\nArrêt (§18). Vérifier l'empreinte des fichiers bruts et la version "
            "des scripts de flyhard avant toute autre chose."
        )
