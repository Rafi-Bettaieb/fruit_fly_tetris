"""Une mouche qui joue à Tetris — connectome MaleCNS v1.0 entraîné par imitation.

Toutes les décisions du projet sont dans `docs/conception.md`. Ce paquet ne
décide rien : il applique. En cas de désaccord entre un commentaire d'ici et le
document, c'est le document qui a raison — ou alors il faut le corriger, pas le
contourner.

Trois modules à la racine tiennent l'architecture, parce qu'ils sont les
frontières entre les morceaux :

- `graines`   — les trois familles de graines, seul endroit qui crée un générateur
- `notation`  — l'interface `Noteur`, commune aux dix conditions comparées
- `encodage`  — les 205 entrées, seule chose que le moteur et les modèles partagent

Règle de dépendance, à ne pas enfreindre : `tetris` et `expert` ne connaissent ni
PyTorch, ni le connectome, ni CUDA. C'est ce qui permet à la phase 0 et à
l'intégration continue de tourner sur n'importe quelle machine.
"""

__version__ = "0.0.0"
