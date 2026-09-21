"""Empreintes SHA-256 des fichiers préparés (conception.md §16, §18).

**Vérifiées à chaque chargement, pas une fois pour toutes.** Une empreinte
différente de celle publiée par flyhard signifie que les données ne sont pas
celles de la référence — et la règle du document est alors sans appel : on
s'arrête, on revient à la version publiée, on ne continue pas « pour voir ».

La raison est simple : tous les résultats du projet se comparent à ceux de
fly-self-driving, et une comparaison n'a de sens que sur le même graphe. Un
run lancé sur des données silencieusement différentes produirait des chiffres
d'apparence normale et sans aucune valeur.

Le manifeste `data/empreintes.json` est versionné dans git, contrairement aux
données elles-mêmes : c'est lui qui porte la promesse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFESTE = Path("data/empreintes.json")
TAILLE_DE_BLOC = 1 << 20


class EmpreinteInvalide(RuntimeError):
    """Levée quand un fichier ne correspond pas à son empreinte attendue."""


def empreinte(chemin: Path) -> str:
    """SHA-256 d'un fichier, lu par blocs — le graphe pèse 1,1 Go."""
    condense = hashlib.sha256()
    with Path(chemin).open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(TAILLE_DE_BLOC), b""):
            condense.update(bloc)
    return condense.hexdigest()


def attendues(manifeste: Path = MANIFESTE) -> dict[str, str | None]:
    if not Path(manifeste).exists():
        return {}
    donnees = json.loads(Path(manifeste).read_text(encoding="utf-8"))
    return {cle: valeur for cle, valeur in donnees.items() if not cle.startswith("_")}


def verifier(nom: str, chemin: Path, manifeste: Path = MANIFESTE) -> str:
    """Compare l'empreinte d'un fichier à celle du manifeste.

    Une entrée à `null` signifie « pas encore renseignée » : l'empreinte est
    alors calculée et rendue, mais **rien n'est validé**. C'est le cas au
    premier chargement, et c'est le moment d'inscrire la valeur publiée par
    flyhard dans le manifeste plutôt que celle qu'on vient de calculer soi-même
    — sinon la vérification ne vérifie que sa propre copie.
    """
    reelle = empreinte(chemin)
    attendue = attendues(manifeste).get(nom)
    if attendue is None:
        return reelle
    if reelle != attendue:
        raise EmpreinteInvalide(
            f"{chemin} ne correspond pas à l'empreinte attendue pour « {nom} ».\n"
            f"  attendue : {attendue}\n"
            f"  obtenue  : {reelle}\n"
            "Arrêt : revenir à la version des données dont l'empreinte est publiée (§18)."
        )
    return reelle


def inscrire(nom: str, valeur: str, manifeste: Path = MANIFESTE) -> None:
    """Renseigne une empreinte dans le manifeste versionné."""
    chemin = Path(manifeste)
    donnees = json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}
    donnees[nom] = valeur
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(donnees, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
