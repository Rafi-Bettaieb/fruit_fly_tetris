/* Scène 3D : une drosophile posée sur une manette (conception.md §13.1).
 *
 * Construite à partir de formes plutôt qu'à partir d'un maillage scanné :
 * corps ellipsoïdes, yeux composés sphériques, ailes translucides, six pattes
 * à trois segments. Les proportions suivent celles de Drosophila melanogaster —
 * abdomen plus long que le thorax, yeux occupant la moitié de la tête, ailes
 * repliées vers l'arrière au repos.
 *
 * Les touches de la manette s'allument depuis la séquence reconstituée par le
 * moteur (§13.2) : la mouche n'appuie sur rien, c'est une image.
 */
(function (global) {
  "use strict";

  const AMBRE = 0xc8963c, AMBRE_SOMBRE = 0x8a6425, ROUGE = 0xcf2a17;

  function ellipsoide(rx, ry, rz, materiau, segments) {
    const g = new THREE.SphereGeometry(1, segments || 28, segments || 22);
    g.scale(rx, ry, rz);
    return new THREE.Mesh(g, materiau);
  }

  /* Décode le maillage NeuroMechFly et rebâtit sa hiérarchie.
   *
   * Format MOUCHE3 : un en-tête par segment — nom, parent, position,
   * quaternion, groupe, tailles — puis les géométries concaténées. Chaque
   * segment devient un Object3D attaché à son parent, si bien que faire
   * tourner un fémur entraîne le tibia et les cinq tarses qui le suivent.
   *
   * Le scan est en Z vers le haut, comme MuJoCo ; Three.js est en Y vers le
   * haut, d'où le quart de tour appliqué à la racine.
   */
  function construireMouche(base64) {
    const brut = atob(base64);
    const octets = new Uint8Array(brut.length);
    for (let i = 0; i < brut.length; i++) octets[i] = brut.charCodeAt(i);
    const vue = new DataView(octets.buffer);

    let o = 7;
    const nSegments = vue.getUint32(o, true); o += 4;
    const decodeur = new TextDecoder();
    const entetes = [];
    for (let s = 0; s < nSegments; s++) {
      const taille = vue.getUint8(o); o += 1;
      const nom = decodeur.decode(octets.subarray(o, o + taille)); o += taille;
      const parent = vue.getInt16(o, true); o += 2;
      const pos = [vue.getFloat32(o, true), vue.getFloat32(o + 4, true),
                   vue.getFloat32(o + 8, true)]; o += 12;
      const quat = [vue.getFloat32(o, true), vue.getFloat32(o + 4, true),
                    vue.getFloat32(o + 8, true), vue.getFloat32(o + 12, true)]; o += 16;
      const groupe = vue.getUint8(o); o += 1;
      const nSommets = vue.getUint32(o, true); o += 4;
      const nIndices = vue.getUint32(o, true); o += 4;
      entetes.push({ nom, parent, pos, quat, groupe, nSommets, nIndices });
    }

    const attendu = o + entetes.reduce((s, e) => s + e.nSommets * 24 + e.nIndices * 4, 0);
    if (attendu !== octets.length) {
      throw new Error(`en-tête incohérent : ${attendu} octets attendus, ${octets.length} reçus`);
    }

    const materiaux = [
      new THREE.MeshStandardMaterial({ color: 0xb8863a, roughness: 0.44, metalness: 0.12 }),
      new THREE.MeshStandardMaterial({ color: 0xcf2a17, roughness: 0.25, metalness: 0.18 }),
      new THREE.MeshPhysicalMaterial({
        color: 0xf2f0ea, transparent: true, opacity: 0.26, roughness: 0.1,
        side: THREE.DoubleSide, depthWrite: false,
      }),
      new THREE.MeshStandardMaterial({ color: 0x7e5a26, roughness: 0.55, metalness: 0.08 }),
    ];

    const articulations = {};
    const noeuds = entetes.map(e => {
      const noeud = new THREE.Object3D();
      noeud.name = e.nom;
      noeud.position.set(e.pos[0], e.pos[1], e.pos[2]);
      // MuJoCo range le quaternion en (w, x, y, z) ; Three.js en (x, y, z, w).
      noeud.quaternion.set(e.quat[1], e.quat[2], e.quat[3], e.quat[0]);
      noeud.userData.repos = noeud.quaternion.clone();
      articulations[e.nom] = noeud;
      return noeud;
    });

    entetes.forEach((e, i) => {
      const nS = e.nSommets, nI = e.nIndices;
      if (nS) {
        const positions = new Float32Array(octets.buffer.slice(o, o + nS * 12)); o += nS * 12;
        const normales = new Float32Array(octets.buffer.slice(o, o + nS * 12)); o += nS * 12;
        const indices = new Uint32Array(octets.buffer.slice(o, o + nI * 4)); o += nI * 4;
        const geometrie = new THREE.BufferGeometry();
        geometrie.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometrie.setAttribute("normal", new THREE.BufferAttribute(normales, 3));
        geometrie.setIndex(new THREE.BufferAttribute(indices, 1));
        noeuds[i].add(new THREE.Mesh(geometrie, materiaux[e.groupe]));
      }
      if (e.parent >= 0) noeuds[e.parent].add(noeuds[i]);
    });

    // Le bout de la patte n'est pas l'origine du dernier tarse mais l'extrémité
    // de sa griffe : on place un repère au point le plus éloigné de sa géométrie,
    // sans quoi la patte viserait juste avec son avant-dernière articulation.
    for (const cote of ["L", "R"]) {
      const dernier = articulations[cote + "FTarsus5"];
      if (!dernier || !dernier.children.length) continue;
      const geometrie = dernier.children[0].geometry;
      const pos = geometrie.getAttribute("position");
      let loin = new THREE.Vector3(), max = -1;
      for (let i = 0; i < pos.count; i++) {
        const v = new THREE.Vector3().fromBufferAttribute(pos, i);
        if (v.lengthSq() > max) { max = v.lengthSq(); loin = v; }
      }
      const pointe = new THREE.Object3D();
      pointe.position.copy(loin);
      dernier.add(pointe);
      articulations[cote + "Pointe"] = pointe;
    }

    const mouche = new THREE.Group();
    const racine = noeuds[0];
    racine.rotation.x = -Math.PI / 2;
    mouche.add(racine);

    const boite = new THREE.Box3().setFromObject(mouche);
    const taille = new THREE.Vector3(), centre = new THREE.Vector3();
    boite.getSize(taille); boite.getCenter(centre);
    racine.position.sub(centre);
    mouche.scale.setScalar(2.6 / Math.max(taille.x, taille.y, taille.z));
    mouche.articulations = articulations;
    return mouche;
  }

  /* La manette SNES européenne : silhouette en os, croix grise foncée à
   * gauche, quatre boutons colorés en losange sur une plaque violette à
   * droite, deux boutons obliques au centre, deux gâchettes sur la tranche.
   *
   * Le corps est extrudé depuis un contour 2D plutôt qu'assemblé à partir
   * d'ellipsoïdes : c'est la seule façon d'obtenir le profil pincé au milieu
   * et les deux lobes arrondis aux extrémités.
   */
  function construireManette() {
    const manette = new THREE.Group();

    const LOBE = 0.62, ECART = 1.15;
    const contour = new THREE.Shape();
    contour.moveTo(-ECART, -LOBE);
    contour.absarc(-ECART, 0, LOBE, -Math.PI / 2, Math.PI / 2, true);
    contour.quadraticCurveTo(-ECART * 0.45, 0.40, 0, 0.44);
    contour.quadraticCurveTo(ECART * 0.45, 0.40, ECART, LOBE);
    contour.absarc(ECART, 0, LOBE, Math.PI / 2, -Math.PI / 2, true);
    contour.quadraticCurveTo(ECART * 0.45, -0.40, 0, -0.44);
    contour.quadraticCurveTo(-ECART * 0.45, -0.40, -ECART, -LOBE);

    const coque = new THREE.MeshStandardMaterial({
      color: 0xd6d3cf, roughness: 0.58, metalness: 0.04,
    });
    // Avec un biseau, l'extrusion déborde de `bevelThickness` de chaque côté :
    // elle occupe donc [−0,06 ; 0,26] et non [0 ; 0,20]. Oublier ce débord
    // enfouissait tous les boutons dans la coque, et la manette paraissait nue.
    const EPAISSEUR = 0.20, BISEAU = 0.06;
    const corps = new THREE.Mesh(new THREE.ExtrudeGeometry(contour, {
      depth: EPAISSEUR, bevelEnabled: true, bevelSize: BISEAU, bevelThickness: BISEAU,
      bevelSegments: 4, curveSegments: 26,
    }), coque);
    corps.rotation.x = -Math.PI / 2;
    corps.position.y = BISEAU;
    manette.add(corps);
    const DESSUS = EPAISSEUR + 2 * BISEAU;   // la face supérieure, à 0,32

    // Les gâchettes, sur la tranche arrière des deux lobes.
    for (const cote of [-1, 1]) {
      const gachette = new THREE.Mesh(
        new THREE.BoxGeometry(0.52, 0.16, 0.20), coque);
      gachette.position.set(cote * ECART, DESSUS * 0.55, -LOBE - 0.04);
      manette.add(gachette);
    }

    // La plaque violette sous les quatre boutons d'action.
    const plaque = new THREE.Mesh(
      new THREE.CylinderGeometry(0.46, 0.46, 0.045, 32),
      new THREE.MeshStandardMaterial({ color: 0x8878b0, roughness: 0.5, metalness: 0.05 }));
    plaque.position.set(ECART, DESSUS + 0.012, 0.02);
    plaque.rotation.y = 0.5;
    manette.add(plaque);

    const touches = {}, mailles = {};
    function bouton(nom, x, z, rayon, couleur, hauteur, angle) {
      const materiau = new THREE.MeshStandardMaterial({
        color: couleur, roughness: 0.28, metalness: 0.05,
        emissive: couleur, emissiveIntensity: 0.12,
      });
      // L'illumination d'un appui remplace cette teinte : on garde la valeur de
      // repos, sinon éteindre un bouton le rendrait noir au lieu de le rendre
      // à sa couleur.
      materiau.userData.repos = materiau.emissive.clone();
      const maille = new THREE.Mesh(
        new THREE.CylinderGeometry(rayon, rayon * 0.92, hauteur || 0.11, 24), materiau);
      maille.position.set(x, DESSUS + 0.03 + (hauteur || 0.10) / 2, z);
      if (angle) maille.rotation.y = angle;
      manette.add(maille);
      touches[nom] = materiau;
      mailles[nom] = maille;
    }

    // La croix : quatre branches distinctes, pour que « gauche » et « droite »
    // soient deux cibles différentes quand la patte vient les chercher.
    const SOMBRE = 0x45454d;
    const BRAS = 0.145;
    for (const [nom, dx, dz] of [
      ["gauche", -1, 0], ["droite", 1, 0], ["haut", 0, -1], ["bas", 0, 1],
    ]) {
      const materiau = new THREE.MeshStandardMaterial({
        color: SOMBRE, roughness: 0.55, metalness: 0.05, emissive: 0x000000,
      });
      materiau.userData.repos = materiau.emissive.clone();
      const maille = new THREE.Mesh(
        new THREE.BoxGeometry(dx ? 0.30 : 0.19, 0.09, dz ? 0.30 : 0.19), materiau);
      maille.position.set(-ECART + dx * 0.20, DESSUS + 0.045, 0.02 + dz * 0.20);
      manette.add(maille);
      touches[nom] = materiau;
      mailles[nom] = maille;
    }
    const centre = new THREE.Mesh(
      new THREE.BoxGeometry(0.19, 0.085, 0.19),
      new THREE.MeshStandardMaterial({ color: SOMBRE, roughness: 0.55 }));
    centre.position.set(-ECART, DESSUS + 0.042, 0.02);
    manette.add(centre);

    // Les quatre boutons d'action, aux couleurs de la console européenne.
    bouton("X", ECART, -0.24, 0.135, 0x3f3fc8);
    bouton("A", ECART + 0.26, 0.02, 0.135, 0xd63c3c);
    bouton("Y", ECART - 0.26, 0.02, 0.135, 0x14a05c);
    bouton("B", ECART, 0.28, 0.135, 0xedaa22);

    // Select et Start, obliques au centre.
    bouton("select", -0.22, 0.10, 0.075, 0x55555d, 0.06, -0.42);
    bouton("start", 0.14, 0.10, 0.075, 0x55555d, 0.06, -0.42);
    mailles.select.scale.set(1, 1, 1.9);
    mailles.start.scale.set(1, 1, 1.9);

    // Le câble, qui sort du haut.
    const cable = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.035, 1.5, 10),
      new THREE.MeshStandardMaterial({ color: 0x2a2a2e, roughness: 0.7 }));
    cable.position.set(0, DESSUS * 0.6, -0.92);
    cable.rotation.x = Math.PI / 2.3;
    manette.add(cable);

    manette.touches = touches;
    manette.mailles = mailles;
    return manette;
  }

  global.SceneMouche = function (conteneur, options) {
    const opts = options || {};
    const scene = new THREE.Scene();
    const largeur = conteneur.clientWidth || 480;
    const hauteur = opts.hauteur || 300;

    const camera = new THREE.PerspectiveCamera(38, largeur / hauteur, 0.1, 100);
    const rendu = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    rendu.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendu.setSize(largeur, hauteur);
    rendu.shadowMap.enabled = true;
    rendu.shadowMap.type = THREE.PCFSoftShadowMap;
    conteneur.appendChild(rendu.domElement);
    rendu.domElement.style.cursor = "grab";
    rendu.domElement.setAttribute("role", "img");
    rendu.domElement.setAttribute("aria-label",
      "Drosophile posée sur une manette ; les touches s'allument au rythme du placement joué");

    scene.add(new THREE.HemisphereLight(0xfff2e0, 0x30251c, 0.75));
    const cle = new THREE.DirectionalLight(0xfff4e2, 1.5);
    cle.position.set(3.0, 6.0, 4.0);
    cle.castShadow = true;
    cle.shadow.mapSize.set(1024, 1024);
    cle.shadow.camera.left = cle.shadow.camera.bottom = -6;
    cle.shadow.camera.right = cle.shadow.camera.top = 6;
    scene.add(cle);
    const contre = new THREE.DirectionalLight(0x9ec8ff, 0.5);
    contre.position.set(-4, 2, -3.5);
    scene.add(contre);

    // Un sol : sans lui la mouche flotte, et les ombres n'ont rien où se poser.
    const sol = new THREE.Mesh(
      new THREE.PlaneGeometry(26, 26),
      new THREE.MeshStandardMaterial({ color: 0x2a2320, roughness: 0.95, metalness: 0 }));
    sol.rotation.x = -Math.PI / 2;
    sol.position.y = -1.62;
    sol.receiveShadow = true;
    scene.add(sol);

    // La manette est posée à plat devant la mouche, inclinée vers elle.
    const manette = construireManette();
    manette.position.set(0, -1.28, -1.15);
    manette.rotation.x = -0.20;
    manette.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    scene.add(manette);

    const mouche = construireMouche(opts.maillage);

    // Pose de repos : les deux pattes avant tendues vers la manette, les
    // quatre autres en appui. Le scan est figé dans la posture de la
    // simulation ; c'est ici qu'on l'assied devant sa console.
    // Le scan est figé dans la posture de marche de la simulation. La posture
    // de jeu — les deux pattes avant tendues vers la manette, les quatre
    // autres en appui au sol — n'existe dans aucune donnée : elle est réglée
    // ici, articulation par articulation.
    const REPOS = {
      LFCoxa: [-0.45, 0, -0.30], LFFemur: [-0.55, 0, 0.20], LFTibia: [0.35, 0, 0],
      RFCoxa: [-0.45, 0, 0.30], RFFemur: [-0.55, 0, 0.20], RFTibia: [0.35, 0, 0],
      LMCoxa: [0.25, 0, -0.30], LMFemur: [-0.20, 0, 0],
      RMCoxa: [0.25, 0, 0.30], RMFemur: [-0.20, 0, 0],
      LHCoxa: [0.05, 0, -0.25], LHFemur: [-0.15, 0, 0],
      RHCoxa: [0.05, 0, 0.25], RHFemur: [-0.15, 0, 0],
    };
    for (const [nom, angles] of Object.entries(REPOS)) {
      const a = mouche.articulations[nom];
      if (!a) continue;
      a.rotation.x += angles[0];
      a.rotation.y += angles[1];
      a.rotation.z += angles[2];
      a.userData.repos = a.quaternion.clone();
    }
    mouche.position.set(0, -0.95, 1.05);
    mouche.rotation.y = Math.PI / 2;   // le museau vers la manette
    mouche.rotation.z = -0.10;         // l'avant du corps légèrement relevé
    mouche.traverse(o => { if (o.isMesh) o.castShadow = true; });
    scene.add(mouche);

    const cible = new THREE.Vector3(0, -0.95, -0.15);
    let angle = 0.95, hauteurVue = 0.30, distance = 7.4, tourne = true;
    function placerCamera() {
      camera.position.set(
        cible.x + distance * Math.cos(angle) * Math.cos(hauteurVue),
        cible.y + distance * Math.sin(hauteurVue),
        cible.z + distance * Math.sin(angle) * Math.cos(hauteurVue));
      camera.lookAt(cible);
    }

    let saisi = false, dernierX = 0, dernierY = 0;
    const toile = rendu.domElement;
    toile.addEventListener("pointerdown", e => {
      saisi = true; tourne = false; dernierX = e.clientX; dernierY = e.clientY;
      toile.setPointerCapture(e.pointerId); toile.style.cursor = "grabbing";
    });
    toile.addEventListener("pointermove", e => {
      if (!saisi) return;
      angle -= (e.clientX - dernierX) * 0.008;
      hauteurVue = Math.max(-0.25, Math.min(1.25, hauteurVue + (e.clientY - dernierY) * 0.006));
      dernierX = e.clientX; dernierY = e.clientY;
      placerCamera();
    });
    for (const fin of ["pointerup", "pointercancel", "pointerleave"]) {
      toile.addEventListener(fin, () => { saisi = false; toile.style.cursor = "grab"; });
    }

    const lent = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let minuteurs = [];

    // La croix directionnelle est à gauche de la manette, les boutons d'action
    // à droite : chaque touche mobilise la patte avant de son côté.
    const COTE = {
      gauche: "L", droite: "L", haut: "L", bas: "L", croix: "L",
      A: "R", B: "R", X: "R", Y: "R", select: "R", start: "R",
    };

    /* Cinématique inverse par descente cyclique.
     *
     * Une flexion d'angle fixe ne pose la patte nulle part : elle plie du même
     * nombre de degrés que la touche soit sous la griffe ou à l'autre bout de
     * la manette. On résout donc la chaîne — coxa, fémur, tibia — pour que la
     * pointe atteigne le bouton visé.
     *
     * La descente cyclique remonte la chaîne depuis la pointe : chaque
     * articulation tourne du strict nécessaire pour rapprocher la pointe de la
     * cible, et l'on recommence. Quelques passes suffisent, et la méthode ne
     * demande ni longueurs de segments ni cas particuliers — ce qui compte
     * pour une patte à huit articulations dont les axes ne sont pas alignés.
     */
    function resoudre(chaine, pointe, cible, passes) {
      const posArticulation = new THREE.Vector3();
      const posPointe = new THREE.Vector3();
      const quatParent = new THREE.Quaternion();
      const quatMonde = new THREE.Quaternion();
      for (let passe = 0; passe < passes; passe++) {
        for (let i = chaine.length - 1; i >= 0; i--) {
          const articulation = chaine[i];
          if (!articulation) continue;
          articulation.getWorldPosition(posArticulation);
          pointe.getWorldPosition(posPointe);
          const versPointe = posPointe.clone().sub(posArticulation);
          const versCible = cible.clone().sub(posArticulation);
          if (versPointe.lengthSq() < 1e-9 || versCible.lengthSq() < 1e-9) continue;
          const rotation = new THREE.Quaternion().setFromUnitVectors(
            versPointe.normalize(), versCible.normalize());
          // Un pas entier par passe ferait osciller la chaîne : on n'en prend
          // qu'une fraction, ce qui converge plus doucement et plus sûrement.
          rotation.slerp(new THREE.Quaternion(), 0.55);
          articulation.getWorldQuaternion(quatMonde);
          articulation.parent.getWorldQuaternion(quatParent);
          articulation.quaternion.copy(
            quatParent.invert().multiply(rotation).multiply(quatMonde));
          articulation.updateMatrixWorld(true);
        }
      }
    }

    // Pose de repos et pose d'appui, calculées une fois par touche : résoudre à
    // chaque image coûterait cher pour un résultat identique.
    const SEGMENTS_PATTE = ["Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2"];
    const posesDAppui = {};

    function chaineDe(cote) {
      return SEGMENTS_PATTE.map(s => mouche.articulations[cote + "F" + s]);
    }

    function calculerPoses() {
      const cible = new THREE.Vector3();
      for (const [nom, maille] of Object.entries(manette.mailles)) {
        const cote = COTE[nom] || "R";
        const chaine = chaineDe(cote);
        const pointe = mouche.articulations[cote + "Pointe"];
        if (!pointe || chaine.some(a => !a)) continue;

        chaine.forEach(a => a.quaternion.copy(a.userData.repos));
        mouche.updateMatrixWorld(true);
        maille.getWorldPosition(cible);
        cible.y += 0.06; // le dessus du bouton, pas son centre
        resoudre(chaine, pointe, cible, 14);
        posesDAppui[nom] = chaine.map(a => a.quaternion.clone());
        chaine.forEach(a => a.quaternion.copy(a.userData.repos));
      }
      mouche.updateMatrixWorld(true);
    }

    const gestes = [];
    function animerPattes(maintenant) {
      const actives = new Set();
      for (let i = gestes.length - 1; i >= 0; i--) {
        const g = gestes[i];
        const t = (maintenant - g.instant) / g.duree;
        if (t >= 1) { gestes.splice(i, 1); continue; }
        if (t < 0) continue;
        actives.add(g.cote);
        // Aller-retour amorti : la patte descend, touche, remonte.
        const avancee = Math.sin(Math.PI * t) ** 1.6;
        g.chaine.forEach((a, j) => {
          a.quaternion.copy(a.userData.repos).slerp(g.pose[j], avancee);
        });
      }
      for (const cote of ["L", "R"]) {
        if (actives.has(cote)) continue;
        chaineDe(cote).forEach(a => { if (a) a.quaternion.copy(a.userData.repos); });
      }
      // Une patte avant qui se tend déplace le poids : les quatre autres se
      // calent. Sans ce contrepoids la mouche a l'air posée sur un socle, et
      // seules ses deux pattes avant semblent lui appartenir.
      const appui = gestes.reduce((max, g) => {
        const t = (maintenant - g.instant) / g.duree;
        return t > 0 && t < 1 ? Math.max(max, Math.sin(Math.PI * t)) : max;
      }, 0);
      for (const cote of ["L", "R"]) {
        for (const rang of ["M", "H"]) {
          const sens = cote === "L" ? 1 : -1;
          appliquerCale(mouche.articulations[cote + rang + "Coxa"],
                        appui * 0.09 * sens * (rang === "M" ? 1 : -0.6));
          appliquerCale(mouche.articulations[cote + rang + "Femur"], appui * -0.07);
        }
      }
    }

    const axeCale = new THREE.Vector3(0, 0, 1);
    const rotationCale = new THREE.Quaternion();
    function appliquerCale(articulation, angle) {
      if (!articulation || !articulation.userData.repos) return;
      rotationCale.setFromAxisAngle(axeCale, angle);
      articulation.quaternion.copy(articulation.userData.repos).multiply(rotationCale);
    }

    /* `pas` est la durée d'un appui, la même que celle de la grille : la page
     * fait tourner ou glisser la pièce à l'instant où la patte touche le
     * bouton, c'est-à-dire à mi-geste. Aux vitesses ×2 et ×4, la page le
     * raccourcit, et tout le geste se raccourcit avec lui. */
    function allumer(sequence, couleurVive, pas) {
      pas = pas || 230;
      minuteurs.forEach(clearTimeout);
      minuteurs = [];
      gestes.length = 0;
      const vive = new THREE.Color(couleurVive || 0xf0a830);
      Object.values(manette.touches).forEach(m => m.emissive.copy(m.userData.repos));
      const depart = performance.now();
      sequence.forEach((nom, i) => {
        const m = manette.touches[nom];
        if (!m) return;
        if (lent) { m.emissive.copy(vive); return; }
        minuteurs.push(setTimeout(() => m.emissive.copy(vive), i * pas + pas * 0.48));
        minuteurs.push(setTimeout(() => m.emissive.copy(m.userData.repos), i * pas + pas * 1.43));
        const cote = COTE[nom] || "R";
        const pose = posesDAppui[nom];
        if (pose) {
          gestes.push({ cote, chaine: chaineDe(cote), pose,
                        instant: depart + i * pas, duree: pas * 1.74 });
        }
      });
    }

    function redimensionner() {
      const l = conteneur.clientWidth || largeur;
      camera.aspect = l / hauteur;
      camera.updateProjectionMatrix();
      rendu.setSize(l, hauteur);
    }
    window.addEventListener("resize", redimensionner);

    calculerPoses();
    placerCamera();
    (function boucle() {
      requestAnimationFrame(boucle);
      if (tourne && !lent) { angle += 0.0016; placerCamera(); }
      if (!lent) animerPattes(performance.now());
      rendu.render(scene, camera);
    })();

    return { allumer, redimensionner, scene, mouche, manette };
  };
})(window);
