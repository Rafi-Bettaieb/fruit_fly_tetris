/* Le nuage de neurones : ce qui s'active quand la mouche décide (conception.md §13.2).
 *
 * 8 000 neurones du MaleCNS, chacun à la position de son soma, dessinés comme
 * des points. Pour chaque pièce, la page reçoit l'état de ces neurones après
 * chacune des K mises à jour du réseau, pour la grille que la mouche a choisie,
 * et l'anime : la vague part des neurones sensoriels, envahit le cerveau, puis
 * atteint les neurones moteurs de la corde nerveuse ventrale, où la note est lue.
 *
 * Les positions et les rôles viennent du dépôt (`connectome/nuage.py`,
 * format NUAGE1) ; le navigateur ne calcule rien d'autre que des couleurs.
 *
 * Orientation : le repère du MaleCNS a x latéral, y dorso-ventral et z
 * antéro-postérieur. On dessine z vers le bas — le cerveau en haut, la corde
 * nerveuse en dessous —, comme sur les images de référence du MaleCNS.
 */
(function (global) {
  "use strict";

  const REPOS = [0.17, 0.19, 0.18];
  const POSITIF = [1.0, 0.64, 0.16];
  const NEGATIF = [0.30, 0.64, 1.0];
  const TAILLES = [0.026, 0.040, 0.048];   // autres, entrées, sorties
  const AUTRE = 0, ENTREE = 1, SORTIE = 2;

  function octetsDepuisBase64(texte) {
    const brut = atob(texte);
    const octets = new Uint8Array(brut.length);
    for (let i = 0; i < brut.length; i++) octets[i] = brut.charCodeAt(i);
    return octets;
  }

  function decoder(texte) {
    const octets = octetsDepuisBase64(texte);
    const entete = String.fromCharCode.apply(null, octets.subarray(0, 6));
    if (entete !== "NUAGE1") throw new Error("format de nuage inconnu : " + entete);
    const vue = new DataView(octets.buffer);
    const n = vue.getUint32(6, true);
    const empreinte = String.fromCharCode.apply(null, octets.subarray(10, 22));
    const debut = 38;
    const attendu = debut + n * 7;
    if (octets.length !== attendu) {
      throw new Error(`nuage tronqué : ${octets.length} octets au lieu de ${attendu}`);
    }
    // Copier plutôt que viser dans le tampon : un Int16Array exige un décalage pair.
    const positions = new Int16Array(octets.slice(debut, debut + n * 6).buffer);
    const roles = octets.slice(debut + n * 6, attendu);
    return { n, empreinte, positions, roles };
  }

  /* Une pastille ronde et douce : un point carré ferait un nuage de pixels. */
  function texturePastille() {
    const toile = document.createElement("canvas");
    toile.width = toile.height = 64;
    const ctx = toile.getContext("2d");
    const degrade = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    degrade.addColorStop(0, "rgba(255,255,255,1)");
    degrade.addColorStop(0.45, "rgba(255,255,255,0.85)");
    degrade.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = degrade;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(toile);
  }

  global.SceneNuage = function (hote, options) {
    const opts = options || {};
    const d = decoder(opts.donnees);
    const lent = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const largeur = hote.clientWidth || 600;
    const hauteur = opts.hauteur || 360;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, largeur / hauteur, 0.01, 50);
    const rendu = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    rendu.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendu.setSize(largeur, hauteur);
    rendu.setClearColor(0x0e1110, 1);
    hote.appendChild(rendu.domElement);
    rendu.domElement.setAttribute("role", "img");
    rendu.domElement.setAttribute("aria-label",
      "Nuage de neurones du système nerveux central : cerveau en haut, corde nerveuse ventrale en bas");

    // Un groupe de points par rôle, pour donner une taille différente aux
    // neurones d'entrée et de sortie — ceux qui disent où le signal entre et
    // où il est lu.
    const pastille = texturePastille();
    const groupes = [AUTRE, ENTREE, SORTIE].map(role => {
      const membres = [];
      for (let i = 0; i < d.n; i++) if (d.roles[i] === role) membres.push(i);
      const positions = new Float32Array(membres.length * 3);
      membres.forEach((i, j) => {
        const x = d.positions[3 * i] / 32767, y = d.positions[3 * i + 1] / 32767;
        const z = d.positions[3 * i + 2] / 32767;
        positions[3 * j] = x;
        positions[3 * j + 1] = -z;       // antéro-postérieur vers le bas
        positions[3 * j + 2] = y;
      });
      const couleurs = new Float32Array(membres.length * 3);
      for (let j = 0; j < membres.length; j++) couleurs.set(REPOS, 3 * j);
      const geometrie = new THREE.BufferGeometry();
      geometrie.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometrie.setAttribute("color", new THREE.BufferAttribute(couleurs, 3));
      const materiau = new THREE.PointsMaterial({
        size: TAILLES[role], vertexColors: true, map: pastille, transparent: true,
        alphaTest: 0.05, depthWrite: false, sizeAttenuation: true,
      });
      const points = new THREE.Points(geometrie, materiau);
      scene.add(points);
      return { role, membres, couleurs, geometrie };
    });

    const composition = {
      autres: groupes[0].membres.length,
      entrees: groupes[1].membres.length,
      sorties: groupes[2].membres.length,
    };

    // --- Cadrage : tout le système nerveux, sous tous les angles ---
    // Le nuage tourne autour de l'axe vertical : sa largeur à l'écran peut
    // valoir, selon l'angle, son rayon le plus grand dans le plan horizontal.
    // Une distance fixe le rognait dès que le panneau était plus étroit que haut.
    let rayonHorizontal = 0, rayonVertical = 0;
    for (const groupe of groupes) {
      const p = groupe.geometrie.attributes.position.array;
      for (let j = 0; j < p.length; j += 3) {
        rayonHorizontal = Math.max(rayonHorizontal, Math.hypot(p[j], p[j + 2]));
        rayonVertical = Math.max(rayonVertical, Math.abs(p[j + 1]));
      }
    }
    function distanceDeCadrage() {
      const t = Math.tan(camera.fov / 2 * Math.PI / 180);
      return 1.08 * Math.max(rayonVertical / t, rayonHorizontal / (t * camera.aspect));
    }

    // --- Caméra : rotation lente autour de l'axe vertical, et à la main ---
    let angle = 0.6, hauteurVue = 0.12, distance = distanceDeCadrage(), tourne = !lent;
    function placerCamera() {
      camera.position.set(
        distance * Math.sin(angle) * Math.cos(hauteurVue),
        distance * Math.sin(hauteurVue),
        distance * Math.cos(angle) * Math.cos(hauteurVue));
      camera.lookAt(0, 0, 0);
    }
    let saisi = false, dernierX = 0, dernierY = 0;
    const toile = rendu.domElement;
    toile.style.cursor = "grab";
    toile.addEventListener("pointerdown", e => {
      saisi = true; tourne = false; dernierX = e.clientX; dernierY = e.clientY;
      toile.setPointerCapture(e.pointerId); toile.style.cursor = "grabbing";
    });
    toile.addEventListener("pointermove", e => {
      if (!saisi) return;
      angle -= (e.clientX - dernierX) * 0.008;
      hauteurVue = Math.max(-1.2, Math.min(1.2, hauteurVue + (e.clientY - dernierY) * 0.006));
      dernierX = e.clientX; dernierY = e.clientY;
      placerCamera();
    });
    for (const fin of ["pointerup", "pointercancel", "pointerleave"]) {
      toile.addEventListener(fin, () => { saisi = false; toile.style.cursor = "grab"; });
    }

    // --- Activité -------------------------------------------------------
    // `etapes` : K × N octets, 128 au repos. On interpole d'une mise à jour à
    // la suivante pour que la vague se voie avancer au lieu de sauter.
    let etapes = null, k = 0, pas = 150, depart = 0, derniereEtape = -1;
    const valeurs = new Float32Array(d.n);

    function valeurDe(octet) { return (octet - 128) / 127; }

    function peindre() {
      for (const groupe of groupes) {
        const c = groupe.couleurs;
        groupe.membres.forEach((i, j) => {
          const v = valeurs[i];
          const cible = v >= 0 ? POSITIF : NEGATIF;
          // Racine douce : une activité faible reste visible sans être criarde.
          const a = Math.pow(Math.min(1, Math.abs(v)), 0.7);
          c[3 * j] = REPOS[0] + (cible[0] - REPOS[0]) * a;
          c[3 * j + 1] = REPOS[1] + (cible[1] - REPOS[1]) * a;
          c[3 * j + 2] = REPOS[2] + (cible[2] - REPOS[2]) * a;
        });
        groupe.geometrie.attributes.color.needsUpdate = true;
      }
    }

    function eteindre() {
      etapes = null;
      valeurs.fill(0);
      peindre();
      derniereEtape = -1;
      if (opts.surEtape) opts.surEtape(-1, 0);
    }

    function afficher(activite, pasMs) {
      if (!activite || activite.n !== d.n) { eteindre(); return false; }
      etapes = octetsDepuisBase64(activite.etapes);
      k = activite.k;
      pas = pasMs || 150;
      depart = performance.now();
      derniereEtape = -1;
      if (lent) {
        for (let i = 0; i < d.n; i++) valeurs[i] = valeurDe(etapes[(k - 1) * d.n + i]);
        peindre();
        if (opts.surEtape) opts.surEtape(k, k);
      }
      return true;
    }

    function avancer(maintenant) {
      if (!etapes || lent) return;
      const avancee = Math.min(k, Math.max(0, (maintenant - depart) / pas));
      const rang = Math.min(k - 1, Math.floor(avancee));
      const fraction = avancee >= k ? 1 : avancee - rang;
      const courant = rang * d.n, precedent = (rang - 1) * d.n;
      for (let i = 0; i < d.n; i++) {
        const avant = rang === 0 ? 0 : valeurDe(etapes[precedent + i]);
        valeurs[i] = avant + (valeurDe(etapes[courant + i]) - avant) * fraction;
      }
      peindre();
      const etape = avancee >= k ? k : Math.floor(avancee) + (fraction > 0 ? 1 : 0);
      if (etape !== derniereEtape) {
        derniereEtape = etape;
        if (opts.surEtape) opts.surEtape(etape, k);
      }
    }

    function redimensionner() {
      const l = hote.clientWidth || largeur;
      camera.aspect = l / hauteur;
      camera.updateProjectionMatrix();
      rendu.setSize(l, hauteur);
      distance = distanceDeCadrage();
      placerCamera();
    }
    window.addEventListener("resize", redimensionner);

    placerCamera();
    (function boucle() {
      requestAnimationFrame(boucle);
      if (tourne) { angle += 0.0022; placerCamera(); }
      avancer(performance.now());
      rendu.render(scene, camera);
    })();

    return { afficher, eteindre, empreinte: d.empreinte, n: d.n, composition };
  };
})(window);
