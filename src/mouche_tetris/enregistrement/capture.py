"""Enregistrement d'une partie pour la démo (conception.md §13.5).

Format JSON compressé, destiné à être lu par la page web. Une partie de
500 pièces pèse environ 2 Mo avec le nuage réduit à 4 000 neurones.

**Ce qui est enregistré.** Les probabilités des meilleurs candidats, parce
qu'elles montrent la décision. Et, pour l'image, la séquence de touches et la
trajectoire de la pièce — apparition, rotation à chaque A ou B, glissement à
chaque ◀ ▶ — calculées ici par le moteur : le navigateur ne connaît pas la forme
des pièces et ne doit pas la connaître (§5.2). Toutes deux se déduisent du seul
placement choisi ; `completer` les ajoute à une capture qui ne les a pas.

**Le nuage de neurones est facultatif.** Un noteur qui n'a pas d'état interne —
l'expert, le joueur au hasard, le juge linéaire — n'a rien à en dire, et le
champ reste vide. La page s'y adapte : elle affiche la grille et la manette sans
le nuage, plutôt que de mentir avec des points inventés.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..tetris.grille import COLONNES, LIGNES, cases_occupees
from ..tetris.partie import Partie
from ..tetris.touches import reconstituer, trajectoire

FANTOMES = 3
"""Nombre de candidats affichés en transparence (§13.2)."""


def _trajectoire(lettre: str, rotation: int, colonne: int) -> list[list[list[int]]]:
    """La trajectoire du moteur, en listes JSON."""
    return [[list(case) for case in position]
            for position in trajectoire(lettre, rotation, colonne)]


def completer(capture: dict[str, Any]) -> dict[str, Any]:
    """Ajoute séquence de touches et trajectoire aux coups qui n'en ont pas.

    Les captures faites avant la trajectoire restent ainsi utilisables : les
    deux se recalculent depuis la pièce et le placement choisi, sans rejouer la
    partie ni recharger le modèle.
    """
    for coup in capture["coups"]:
        lettre, choix = coup["piece"], coup["choix"]
        coup.setdefault("touches", [t.value for t in
                                    reconstituer(lettre, choix["rotation"], choix["colonne"])])
        coup.setdefault("trajectoire", _trajectoire(lettre, choix["rotation"], choix["colonne"]))
    return capture


def _probabilites(notes: np.ndarray) -> np.ndarray:
    """Softmax des notes, telle que la démo l'affiche en pourcentages."""
    stables = notes - notes.max()
    exponentielles = np.exp(stables)
    return exponentielles / exponentielles.sum()


def _grille_en_lignes(grille) -> list[str]:
    """La grille en 20 chaînes de 10 caractères, lisible et compacte en JSON."""
    cases = cases_occupees(grille)
    return ["".join("#" if cases[y, x] else "." for x in range(COLONNES)) for y in range(LIGNES)]


def _cellules_posees(grille_avant, lettre: str, candidat) -> list[list[int]]:
    """Les quatre cases que la pièce vient d'occuper, avant disparition des lignes.

    La page en a besoin pour dessiner la pièce fantôme à l'endroit où elle
    tomberait. Les recalculer côté navigateur demanderait d'y réimplémenter la
    chute — donc un second moteur, ce que §5.2 interdit.
    """
    from ..tetris.grille import poser, sommets
    from ..tetris.pieces import rotations

    rotation = rotations(lettre)[candidat.rotation]
    plafonds = sommets(grille_avant)
    depart = LIGNES
    for colonne_piece, ligne_basse in rotation.colonnes_de_la_case_la_plus_basse().items():
        depart = min(depart, plafonds[candidat.colonne + colonne_piece] - 1 - ligne_basse)
    assert poser(grille_avant, rotation, candidat.colonne, plafonds) is not None
    return [
        [depart + ligne, candidat.colonne + colonne] for ligne, colonne in rotation.cases
    ]


def capturer(partie: Partie, nom: str, etats_moteurs=None) -> dict[str, Any]:
    """Transforme une partie jouée en structure prête pour la page web.

    `partie` doit avoir été jouée avec `enregistrer_coups=True`.
    """
    if not partie.coups:
        raise ValueError("partie jouée sans enregistrer_coups=True : rien à capturer")

    pieces = []
    for indice, coup in enumerate(partie.coups):
        probabilites = _probabilites(coup.notes)
        meilleurs = np.argsort(-coup.notes)[:FANTOMES]
        choisi = coup.candidats[coup.choisi]
        pieces.append(
            {
                "grille": _grille_en_lignes(coup.grille_avant),
                "piece": coup.lettre,
                "candidats": len(coup.candidats),
                "choix": {
                    "rotation": choisi.rotation,
                    "colonne": choisi.colonne,
                    "cellules": _cellules_posees(coup.grille_avant, coup.lettre, choisi),
                },
                "lignes": choisi.lignes_completees,
                "touches": [touche.value for touche in
                            reconstituer(coup.lettre, choisi.rotation, choisi.colonne)],
                "trajectoire": _trajectoire(coup.lettre, choisi.rotation, choisi.colonne),
                "fantomes": [
                    {
                        "rotation": coup.candidats[int(i)].rotation,
                        "colonne": coup.candidats[int(i)].colonne,
                        "probabilite": round(float(probabilites[int(i)]), 4),
                        "cellules": _cellules_posees(
                            coup.grille_avant, coup.lettre, coup.candidats[int(i)]
                        ),
                    }
                    for i in meilleurs
                ],
                "neurones": (
                    _quantifier(etats_moteurs[indice]) if etats_moteurs is not None else None
                ),
            }
        )

    return {
        "nom": nom,
        "graine": partie.graine,
        "lignes": partie.lignes,
        "pieces_posees": partie.pieces_posees,
        "plafond_atteint": partie.plafond_atteint,
        "trous_par_piece": round(partie.trous_par_piece, 3),
        "avertissement": (
            "La mouche note toutes les façons de poser la pièce — jusqu'à 34 — et "
            "choisit la meilleure. La suite de touches est reconstituée après coup, "
            "pour l'image."
        ),
        "coups": pieces,
    }


def _quantifier(etats: np.ndarray) -> list[int]:
    """Un octet par neurone, échelle propre à chacun (§13.5).

    Chaque neurone est normalisé sur sa propre plage : sans cela, les quelques
    neurones très actifs écraseraient tous les autres et le nuage paraîtrait
    éteint.
    """
    etats = np.asarray(etats, dtype=np.float64)
    amplitude = np.abs(etats).max() or 1.0
    return np.clip(np.round(etats / amplitude * 127) + 128, 0, 255).astype(int).tolist()


def ecrire(capture: dict[str, Any], chemin: Path) -> Path:
    """Écrit la capture en JSON compressé. La page la lit telle quelle."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(chemin, "wt", encoding="utf-8") as fichier:
        json.dump(capture, fichier, ensure_ascii=False, separators=(",", ":"))
    return chemin


def lire(chemin: Path) -> dict[str, Any]:
    with gzip.open(Path(chemin), "rt", encoding="utf-8") as fichier:
        return json.load(fichier)
