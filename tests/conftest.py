"""Fixtures communes.

Contrainte structurante (conception.md §16) : **les tests ne téléchargent jamais
le MaleCNS**. Tout ce qui touche au connectome tourne sur un graphe jouet de
20 neurones, construit en mémoire et déterministe.

Les tests qui exigeraient vraiment les données portent la marque `connectome`,
et l'intégration continue les exclut explicitement. La marque n'est pas une
convention de politesse : c'est ce qui garantit qu'un test coûteux ne se glisse
pas dans la CI par inadvertance.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def graphe_jouet():
    """20 neurones, connexions fixées, signes fixés.

    Sert à tester ce qui ne produit aucun message d'erreur quand c'est faux :
    l'orientation du graphe (présynaptique → postsynaptique), la normalisation
    par neurone postsynaptique, et le fait que les témoins « graphe coupé » et
    « entrée aveuglée » donnent à tous les candidats la même note.
    """
    pytest.skip("phase 1 — fixture à écrire avec le paquet connectome")
