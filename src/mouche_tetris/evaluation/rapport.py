"""Mise en forme du tableau de résultats (conception.md §11.1, §11.3).

Le tableau est produit en Markdown, pour tomber tel quel dans `docs/resultats.md`.

Trois colonnes sont là par honnêteté plutôt que par tradition :

- **la part de parties au plafond**, parce qu'une moyenne tronquée par le
  plafond est un plancher et non une performance : l'expert y est à 100 % ;
- **le budget**, parce qu'un écart entre deux modèles entraînés sur des budgets
  différents se lit comme un écart de capacité, ce qu'il n'est pas (§3) ;
- **l'intervalle**, toujours, y compris quand il est large.
"""

from __future__ import annotations

from .bootstrap import Intervalle, difference_appariee, intervalle
from .parties import Serie

ENTETES = (
    "Condition",
    "Lignes (IC 95 %)",
    "Médiane",
    "Pièces",
    "Plafond",
    "Trous/pièce",
    "Budget",
)


def _ligne(cellules) -> str:
    return "| " + " | ".join(str(cellule) for cellule in cellules) + " |"


def tableau(series: list[Serie], budgets: dict[str, str] | None = None) -> str:
    """Le tableau de résultats, une ligne par condition."""
    budgets = budgets or {}
    lignes = [_ligne(ENTETES), _ligne(["---"] * len(ENTETES))]
    for serie in series:
        lignes.append(
            _ligne(
                (
                    serie.nom,
                    intervalle(serie.lignes),
                    f"{serie.mediane:.0f}",
                    f"{serie.pieces_moyennes:.1f}",
                    f"{serie.part_plafond * 100:.0f} %",
                    f"{serie.trous_par_piece:.3f}",
                    budgets.get(serie.nom, "—"),
                )
            )
        )
    return "\n".join(lignes)


def ecarts_apparies(reference: Serie, autres: list[Serie]) -> str:
    """Les différences appariées par rapport à une condition de référence.

    C'est ici que se lit la question du projet : l'écart entre la mouche et la
    mouche recâblée, avec son intervalle. Un intervalle qui contient 0 n'est pas
    un demi-résultat, c'est une réponse — et elle se publie (§11.3).
    """
    lignes = [
        _ligne(("Écart", "Différence (IC 95 %)", "Conclusion")),
        _ligne(["---"] * 3),
    ]
    for serie in autres:
        ecart: Intervalle = difference_appariee(serie.lignes, reference.lignes)
        verdict = "écart démontré" if ecart.exclut_zero else "compatible avec 0"
        lignes.append(_ligne((f"{serie.nom} − {reference.nom}", ecart, verdict)))
    return "\n".join(lignes)
