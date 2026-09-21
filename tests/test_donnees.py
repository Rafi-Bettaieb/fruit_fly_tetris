"""Empreintes et contrôles d'intégrité des données.

Ces tests ne téléchargent rien (§16) : ils portent sur le mécanisme de
vérification, pas sur le MaleCNS lui-même.
"""

from __future__ import annotations

import json

import pytest

from mouche_tetris.connectome import acquisition as acq
from mouche_tetris.connectome import empreintes as emp


@pytest.fixture
def fichier(tmp_path):
    chemin = tmp_path / "graphe.bin"
    chemin.write_bytes(b"un graphe qui n'en est pas un")
    return chemin


@pytest.fixture
def manifeste(tmp_path):
    chemin = tmp_path / "empreintes.json"
    chemin.write_text('{"_commentaire": "essai", "graphe": null}\n', encoding="utf-8")
    return chemin


def test_empreinte_stable(fichier):
    assert emp.empreinte(fichier) == emp.empreinte(fichier)
    assert len(emp.empreinte(fichier)) == 64


def test_empreinte_change_avec_le_contenu(fichier, tmp_path):
    autre = tmp_path / "autre.bin"
    autre.write_bytes(b"un graphe qui n'en est pas un.")  # un point de plus
    assert emp.empreinte(fichier) != emp.empreinte(autre)


def test_une_entree_nulle_ne_valide_rien(fichier, manifeste):
    """Au premier chargement, on calcule sans valider — et l'on inscrit ensuite
    la valeur **publiée par flyhard**, pas celle qu'on vient de calculer."""
    valeur = emp.verifier("graphe", fichier, manifeste)
    assert valeur == emp.empreinte(fichier)


def test_les_commentaires_ne_sont_pas_des_empreintes(manifeste):
    assert "_commentaire" not in emp.attendues(manifeste)


def test_une_empreinte_differente_arrete_tout(fichier, manifeste):
    emp.inscrire("graphe", "0" * 64, manifeste)
    with pytest.raises(emp.EmpreinteInvalide, match="Arrêt"):
        emp.verifier("graphe", fichier, manifeste)


def test_une_empreinte_correcte_passe(fichier, manifeste):
    emp.inscrire("graphe", emp.empreinte(fichier), manifeste)
    assert emp.verifier("graphe", fichier, manifeste) == emp.empreinte(fichier)


def test_inscrire_preserve_le_manifeste(manifeste):
    emp.inscrire("graphe", "a" * 64, manifeste)
    donnees = json.loads(manifeste.read_text(encoding="utf-8"))
    assert donnees["_commentaire"] == "essai"
    assert donnees["graphe"] == "a" * 64


# --- Contrat sur le graphe préparé ---------------------------------------


def test_le_graphe_de_la_reference_passe():
    acq.verifier_le_graphe(165_122, 25_563_197, 4_114, 708)


@pytest.mark.parametrize(
    "arguments, attendu",
    [
        ((165_121, 25_563_197, 4_114, 708), "neurones"),
        ((165_122, 25_000_000, 4_114, 708), "connexions"),
        ((165_122, 25_563_197, 4_000, 708), "entrée"),
        ((165_122, 25_563_197, 4_114, 700), "moteurs"),
    ],
)
def test_tout_ecart_arrete_le_projet(arguments, attendu):
    """Quatre nombres connus d'avance : s'ils ne tombent pas, on ne continue pas."""
    with pytest.raises(acq.DonneesIncoherentes, match=attendu):
        acq.verifier_le_graphe(*arguments)


def test_le_message_dit_quoi_faire():
    with pytest.raises(acq.DonneesIncoherentes, match="empreinte"):
        acq.verifier_le_graphe(1, 1, 1, 1)


def test_les_fichiers_attendus_sont_nommes():
    assert set(acq.FICHIERS) == {"annotations", "neurotransmetteurs", "poids"}
    assert all(nom.endswith(".feather") for nom in acq.FICHIERS.values())


def test_inspecter_signale_un_fichier_absent(tmp_path):
    schemas = acq.inspecter_tout(tmp_path)
    assert len(schemas) == 3
    assert all("absent" in schema.fichier for schema in schemas)
