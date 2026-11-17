// =============================================================================
//  KI-Klassifikation · Verborgene Schicht · nicht linear trennbar (XOR)
//  Ein kleines neuronales Netz 2 → h → 1 lernt Klassen, die keine gerade
//  Linie trennen kann – vier Beispiele zur Auswahl:
//    - Logik: XOR (wie MemBrain nlsP)
//    - Pneumatikzylinder: genau ein Endlagensensor darf melden (XOR mit Rauschen)
//    - Bohrung: Toleranzkreis
//    - Spritzguss: Prozessfenster (Rechteck)
// =============================================================================
//
//  Dateien
//    basis.js        gemeinsame Grundlage (Layout, Farben, Kopfzeile, UI-Helfer)
//    daten.js        Daten der vier Beispiele
//    netz.js         das Netz (Vorwärtsrechnung, Backpropagation, Training)
//    ui.js           Knöpfe, Regler, Auswahllisten
//    sketch.js       Ablauf, Zeichnen, Erklärungen (diese Datei)
//
//  Bedienung
//    →         eine Epoche weiter
//    E         100 Epochen weiter
//    Leertaste trainieren / anhalten
//    R         neu starten
//    Klick     auf einen Punkt: im Netz nachrechnen
// =============================================================================

const APP_TITEL = "KI-Klassifikation";

const P_DATEN = { x: 24, y: 84, w: 820, h: 800, titel: "" };
const P_ERKL = { x: 860, y: 84, w: 716, h: 244, titel: "" };
const P_EINST = { x: 860, y: 344, w: 716, h: 132, titel: "Einstellungen" };
const P_NETZ = { x: 860, y: 492, w: 716, h: 184, titel: "" };
const P_KURVE = { x: 860, y: 692, w: 350, h: 192, titel: "Fehler (MSE) je Epoche" };
const P_KONF = { x: 1226, y: 692, w: 350, h: 192, titel: "" };

const PLOT = { x: 104, y: 134, w: 700, h: 612 };
const ZELLE = 5; // Kantenlänge einer Zelle der Ausgabekarte (Pixel)
const RASTER = 14; // Abstand der Stützpunkte für die Entscheidungsgrenze
// Tempo beim Trainieren: Epochen pro Bild bzw. Pause zwischen zwei Epochen
const TEMPO = { langsam: { pause: 260, je: 1 }, mittel: { pause: 0, je: 1 }, schnell: { pause: 0, je: 8 } };

// -----------------------------------------------------------------------------
// Zustand
// -----------------------------------------------------------------------------
let szenario = "xor";
let startwert = 0;
let daten = [];

let neuronen = 4;
let neuronenVorschau = null; // Wert des Reglers, solange er gezogen wird
let aktivierung = "sigmoid";
let lernrate = 1;
let tempo = "mittel";
let gewichtStart = 1;

let training = null; // vorberechnete Epochen
let epoche = 0;
let laufAuto = false;
let letzterSchritt = 0;
let stoppBeimZiel = true;

let linienAn = true;
let falscheAn = true;
let auswahlIndex = 0;

// Zwischenspeicher der Ausgabekarte
let karteBild = null;
let karteSchluessel = "";

// =============================================================================
// Start
// =============================================================================
function setup() {
  ladeSchriften();
  const [cw, ch] = zeichenflaeche();
  cnv = createCanvas(cw, ch);
  S = width / W;

  szenarioLaden("xor");
  erstelleUi();
  cnv.parent(huelle);
  layoutUi();
}

function nachFehler() {
  laufAuto = false;
}

// =============================================================================
// Hauptschleife
// =============================================================================
function draw() {
  try {
    logik();
    zeichneRahmen(() => {
      zeichneStatusZeile(statusText());
      zeichneDatenPanel();
      zeichneErklaerung();
      zeichneEinstellungen();
      zeichneNetz();
      zeichneFehlerKurve();
      zeichneKonfusion();
      zeichneTooltip();
    });
    aktualisiereUi();
  } catch (err) {
    behandleFehler(err);
  }
}

function logik() {
  if (!laufAuto) {
    return;
  }
  const t = TEMPO[tempo];
  if (millis() - letzterSchritt < t.pause) {
    return;
  }
  letzterSchritt = millis();
  for (let i = 0; i < t.je; i++) {
    if (epoche >= maxEpoche()) {
      break;
    }
    epoche++;
    if (stoppBeimZiel && training.gelernt > 0 && epoche === training.gelernt) {
      laufAuto = false;
      zeigeHinweis(`Ziel erreicht: alle ${daten.length} ${SZ.wort.mehrzahl} richtig, Fehler unter ${fKomma(GRENZE_MSE, 2)}. Mit „Trainieren“ geht es weiter.`);
      break;
    }
  }
  if (epoche >= maxEpoche()) {
    laufAuto = false;
  }
}

// =============================================================================
// Daten und Ablauf
// =============================================================================
function szenarioLaden(id) {
  szenario = id;
  setzeSzenario(id);
  startwert = SZ.startwert || 0;
  daten = erzeugeDaten(startwert);
  for (const p of daten) {
    p.xn = normiere(p);
  }
  neuronen = SZ.neuronen;
  lernrate = SZ.lernrate;
  aktivierung = "sigmoid";
  gewichtStart = 1;
  auswahlIndex = 0;
  neuStarten();
}

function szenarioWechseln(id) {
  if (laufAuto || id === szenario) {
    return;
  }
  szenarioLaden(id);
}

function neueMessdaten() {
  if (SZ.fest) {
    zeigeHinweis("XOR hat feste Daten. Für zufällige Messdaten ein anderes Beispiel wählen.");
    return;
  }
  startwert = Math.floor(Math.random() * 1e9);
  daten = erzeugeDaten(startwert);
  for (const p of daten) {
    p.xn = normiere(p);
  }
  auswahlIndex = 0;
  neuStarten();
}

function neueStartgewichte() {
  if (laufAuto) {
    return;
  }
  gewichtStart++;
  neuStarten();
  zeigeHinweis("Neue zufällige Startgewichte. Lernt das Netz wieder – und sieht die Grenze gleich aus?");
}

function neuStarten() {
  laufAuto = false;
  epoche = 0;
  const seed = 4242 + 101 * gewichtStart + Object.keys(SZENARIEN).indexOf(szenario) * 7 + neuronen * 13;
  training = trainiere(daten, neuronen, aktivierung, lernrate, seed, SZ.maxEpochen);
  karteSchluessel = "";
}

function maxEpoche() {
  return training ? training.stand.length - 1 : 0;
}

function schritt() {
  if (epoche < maxEpoche()) {
    epoche++;
  }
}

function springe(n) {
  if (laufAuto) {
    return;
  }
  epoche = Math.min(maxEpoche(), epoche + n);
}

function autoUmschalten() {
  if (laufAuto) {
    laufAuto = false;
    return;
  }
  if (epoche >= maxEpoche()) {
    return;
  }
  laufAuto = true;
  letzterSchritt = 0;
}

function stand() {
  return training.stand[epoche];
}

function netz() {
  return stand().netz;
}

// =============================================================================
// Zahlen und Texte
// =============================================================================
function fW(v) {
  return fKomma(v, 2);
}

function klassenName(k) {
  return SZ.klassen[k].name;
}

function statusText() {
  const s = stand();
  return (
    `${daten.length} ${SZ.wort.mehrzahl} · Netz 2 → ${neuronen} → 1 · Aktivierung ${AKTIVIERUNGEN[aktivierung].name} · η = ${fKomma(lernrate, 1)} · ` +
    `Epoche ${epoche} · Fehler (MSE) ${fKomma(s.mse, 3)} · richtig ${s.richtig} von ${daten.length}`
  );
}

// =============================================================================
// Diagramm – Umrechnung
// =============================================================================
function zuPlot(x, y) {
  const [m1, m2] = MERKMALE;
  return [PLOT.x + ((x - m1.min) / (m1.max - m1.min)) * PLOT.w, PLOT.y + PLOT.h - ((y - m2.min) / (m2.max - m2.min)) * PLOT.h];
}

function nZuPlot(a, b) {
  return [PLOT.x + ((a + 1) / 2) * PLOT.w, PLOT.y + PLOT.h - ((b + 1) / 2) * PLOT.h];
}

function plotZuN(px, py) {
  return [((px - PLOT.x) / PLOT.w) * 2 - 1, ((PLOT.y + PLOT.h - py) / PLOT.h) * 2 - 1];
}

function imPlot(px, py) {
  return px >= PLOT.x && px <= PLOT.x + PLOT.w && py >= PLOT.y && py <= PLOT.y + PLOT.h;
}

function punktUnterMaus() {
  const mx = mausX();
  const my = mausY();
  if (!imPlot(mx, my)) {
    return -1;
  }
  let bester = -1;
  let besterAbstand = 11;
  daten.forEach((p, i) => {
    const [px, py] = zuPlot(p.x, p.y);
    const d = dist(mx, my, px, py);
    if (d < besterAbstand) {
      besterAbstand = d;
      bester = i;
    }
  });
  return bester;
}

// =============================================================================
// Zeichnen – Formen
// =============================================================================
function zeichneForm(form, px, py, r) {
  if (form === "dreieck") {
    triangle(px, py - r * 1.15, px + r, py + r * 0.7, px - r, py + r * 0.7);
  } else {
    circle(px, py, r * 2);
  }
}

function zeichneDatenpunkt(p, px, py, alpha, r) {
  const kl = SZ.klassen[p.k];
  stroke(...FARBE.flaeche);
  strokeWeight(1.4);
  fill(kl.farbe[0], kl.farbe[1], kl.farbe[2], alpha);
  zeichneForm(kl.form, px, py, r || 6.5);
}

// =============================================================================
// Ausgabekarte: Farbe = Ausgabe ŷ des Netzes an jeder Stelle
// =============================================================================
function aktualisiereKarte() {
  const schluessel = [szenario, startwert, neuronen, aktivierung, lernrate, gewichtStart, epoche, dunkelModus].join("|");
  if (schluessel === karteSchluessel && karteBild) {
    return;
  }
  const nx = Math.ceil(PLOT.w / ZELLE);
  const ny = Math.ceil(PLOT.h / ZELLE);
  if (!karteBild) {
    karteBild = createGraphics(nx, ny);
    karteBild.pixelDensity(1);
  }
  const g = karteBild;
  const n = netz();
  const c0 = SZ.klassen[0].farbe;
  const c1 = SZ.klassen[1].farbe;
  const pap = FARBE.papier;
  g.loadPixels();
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const [a, b] = plotZuN(PLOT.x + (i + 0.5) * ZELLE, PLOT.y + (j + 0.5) * ZELLE);
      const y = vorwaerts(n, [a, b], aktivierung).y;
      const c = y >= 0.5 ? c1 : c0;
      const anteil = 0.08 + Math.abs(y - 0.5) * 0.5;
      const k = 4 * (j * nx + i);
      g.pixels[k] = pap[0] + (c[0] - pap[0]) * anteil;
      g.pixels[k + 1] = pap[1] + (c[1] - pap[1]) * anteil;
      g.pixels[k + 2] = pap[2] + (c[2] - pap[2]) * anteil;
      g.pixels[k + 3] = 255;
    }
  }
  g.updatePixels();
  karteSchluessel = schluessel;
}

// Entscheidungsgrenze ŷ = 0,5 (Marching Squares auf einem groben Raster)
function zeichneGrenze() {
  const n = netz();
  const nx = Math.ceil(PLOT.w / RASTER);
  const ny = Math.ceil(PLOT.h / RASTER);
  const wert = [];
  for (let i = 0; i <= nx; i++) {
    wert.push([]);
    for (let j = 0; j <= ny; j++) {
      const [a, b] = plotZuN(PLOT.x + i * RASTER, PLOT.y + j * RASTER);
      wert[i].push(vorwaerts(n, [a, b], aktivierung).z);
    }
  }
  stroke(...FARBE.text);
  strokeWeight(2.6);
  const zwischen = (x1, y1, v1, x2, y2, v2) => {
    const t = v1 / (v1 - v2);
    return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t];
  };
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const x0 = PLOT.x + i * RASTER;
      const y0 = PLOT.y + j * RASTER;
      const x1 = x0 + RASTER;
      const y1 = y0 + RASTER;
      const a = wert[i][j];
      const b = wert[i + 1][j];
      const c = wert[i + 1][j + 1];
      const d = wert[i][j + 1];
      const pkt = [];
      if (a > 0 !== b > 0) {
        pkt.push(zwischen(x0, y0, a, x1, y0, b));
      }
      if (b > 0 !== c > 0) {
        pkt.push(zwischen(x1, y0, b, x1, y1, c));
      }
      if (c > 0 !== d > 0) {
        pkt.push(zwischen(x1, y1, c, x0, y1, d));
      }
      if (d > 0 !== a > 0) {
        pkt.push(zwischen(x0, y1, d, x0, y0, a));
      }
      if (pkt.length === 2) {
        line(pkt[0][0], pkt[0][1], pkt[1][0], pkt[1][1]);
      } else if (pkt.length === 4) {
        line(pkt[0][0], pkt[0][1], pkt[1][0], pkt[1][1]);
        line(pkt[2][0], pkt[2][1], pkt[3][0], pkt[3][1]);
      }
    }
  }
}

// =============================================================================
// Datenfeld
// =============================================================================
function zeichneDatenPanel() {
  P_DATEN.titel = SZ.titel;
  panel(P_DATEN);
  zeichneLegende();

  const [m1, m2] = MERKMALE;
  const n = netz();
  const gross = daten.length <= 12;

  stroke(...FARBE.linie);
  strokeWeight(1);
  fill(...FARBE.papier);
  rect(PLOT.x, PLOT.y, PLOT.w, PLOT.h);

  clipRechteck(PLOT.x, PLOT.y, PLOT.w, PLOT.h);

  aktualisiereKarte();
  drawingContext.imageSmoothingEnabled = false;
  image(karteBild, PLOT.x, PLOT.y, Math.ceil(PLOT.w / ZELLE) * ZELLE, Math.ceil(PLOT.h / ZELLE) * ZELLE);
  drawingContext.imageSmoothingEnabled = true;

  // Gitter
  const anzX = Math.round((m1.max - m1.min) / m1.gitter);
  const anzY = Math.round((m2.max - m2.min) / m2.gitter);
  stroke(...FARBE.text, 18);
  strokeWeight(1);
  for (let i = 0; i <= anzX; i++) {
    const [gx] = zuPlot(m1.min + i * m1.gitter, m2.min);
    line(gx, PLOT.y, gx, PLOT.y + PLOT.h);
  }
  for (let i = 0; i <= anzY; i++) {
    const [, gy] = zuPlot(m1.min, m2.min + i * m2.gitter);
    line(PLOT.x, gy, PLOT.x + PLOT.w, gy);
  }

  // Linien der versteckten Neuronen (zⱼ = 0)
  if (linienAn) {
    for (let j = 0; j < n.W.length; j++) {
      const l = neuronLinie(n.W[j][0], n.W[j][1], n.b[j]);
      if (!l) {
        continue;
      }
      const [ax, ay] = nZuPlot(l[0][0], l[0][1]);
      const [bx, by] = nZuPlot(l[1][0], l[1][1]);
      stroke(...FARBE.senf);
      strokeWeight(1.8);
      gestrichelt(true);
      line(ax, ay, bx, by);
      gestrichelt(false);
      // Nummer an der Stelle, die am weitesten im Bild liegt (Viertel der Linie)
      const lx = ax + (bx - ax) * (0.18 + 0.1 * (j % 3));
      const ly = ay + (by - ay) * (0.18 + 0.1 * (j % 3));
      noStroke();
      fill(...FARBE.senf);
      circle(lx, ly, 20);
      fill(255);
      textFont(MONO);
      textStyle(BOLD);
      textSize(11);
      textAlign(CENTER, CENTER);
      text(String(j + 1), lx, ly);
      textStyle(NORMAL);
    }
  }

  // Entscheidungsgrenze ŷ = 0,5
  zeichneGrenze();

  // Punkte
  daten.forEach((p, i) => {
    const [px, py] = zuPlot(p.x, p.y);
    const y = vorwaerts(n, p.xn, aktivierung).y;
    const falsch = (y >= 0.5 ? 1 : 0) !== p.k;
    zeichneDatenpunkt(p, px, py, 240, gross ? 10 : 6.5);
    if (falscheAn && falsch) {
      noFill();
      stroke(...FARBE.senf);
      strokeWeight(2.4);
      circle(px, py, gross ? 34 : 24);
    }
    if (gross) {
      noStroke();
      fill(...FARBE.text);
      textFont(SCHRIFT);
      textSize(12);
      const rechts = px > PLOT.x + PLOT.w - 190;
      textAlign(rechts ? RIGHT : LEFT, CENTER);
      text(`${p.name} → ŷ = ${fW(y)}`, rechts ? px - 24 : px + 24, py + 1);
    }
    if (i === auswahlIndex) {
      noFill();
      stroke(...FARBE.text);
      strokeWeight(2);
      gestrichelt(true);
      circle(px, py, gross ? 44 : 34);
      gestrichelt(false);
    }
  });

  clipEnde();

  // Achsen
  noStroke();
  fill(...FARBE.grau);
  textFont(SCHRIFT);
  textSize(12);
  for (let i = 0; i <= anzX; i++) {
    const v = m1.min + i * m1.gitter;
    const [gx] = zuPlot(v, m2.min);
    textAlign(CENTER, TOP);
    text(Number(v.toFixed(3)).toString().replace(".", ","), gx, PLOT.y + PLOT.h + 6);
  }
  for (let i = 0; i <= anzY; i++) {
    const v = m2.min + i * m2.gitter;
    const [, gy] = zuPlot(m1.min, v);
    textAlign(RIGHT, CENTER);
    text(Number(v.toFixed(3)).toString().replace(".", ","), PLOT.x - 6, gy);
  }
  fill(...FARBE.text);
  textStyle(BOLD);
  textSize(13);
  textAlign(CENTER, TOP);
  text(`x₁ = ${m1.name}${m1.einheit ? " in " + m1.einheit : ""}`, PLOT.x + PLOT.w / 2, PLOT.y + PLOT.h + 26);
  push();
  translate(P_DATEN.x + 18, PLOT.y + PLOT.h / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, TOP);
  text(`x₂ = ${m2.name}${m2.einheit ? " in " + m2.einheit : ""}`, 0, 0);
  pop();
  textStyle(NORMAL);

  if (epoche === 0 && !laufAuto) {
    banner("Startgewichte zufällig · mit „Trainieren“ lernt das Netz", PLOT.x + PLOT.w / 2, PLOT.y + 14, false);
  }

  noStroke();
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(12);
  textAlign(LEFT, CENTER);
  text("Farbe = Ausgabe ŷ · schwarze Linie: ŷ = 0,5", 556, 838);
  text("Klick auf einen Punkt: im Netz nachrechnen", 556, 856);

  cursor(punktUnterMaus() >= 0 ? HAND : ARROW);
}

function zeichneLegende() {
  textFont(SCHRIFT);
  textSize(12);
  const y = P_DATEN.y + 17;
  const eintraege = SZ.klassen.map((k, i) => ({ name: `${k.name} (t = ${i})`, k: i }));
  eintraege.push({ name: "verstecktes Neuron", k: -1 });
  eintraege.push({ name: "Grenze ŷ = 0,5", k: -2 });
  let breite = 0;
  for (const e of eintraege) {
    breite += textWidth(e.name) + 34;
  }
  let x = P_DATEN.x + P_DATEN.w - 16 - breite;
  for (const e of eintraege) {
    if (e.k >= 0) {
      zeichneDatenpunkt({ k: e.k }, x + 6, y, 235);
    } else if (e.k === -1) {
      stroke(...FARBE.senf);
      strokeWeight(2);
      gestrichelt(true);
      line(x - 2, y, x + 14, y);
      gestrichelt(false);
    } else {
      stroke(...FARBE.text);
      strokeWeight(2.6);
      line(x - 2, y, x + 14, y);
    }
    noStroke();
    fill(...FARBE.text);
    textFont(SCHRIFT);
    textSize(12);
    textAlign(LEFT, CENTER);
    text(e.name, x + 20, y + 1);
    x += textWidth(e.name) + 34;
  }
}

// =============================================================================
// Erklärfeld
// =============================================================================
function erklaerung() {
  const s = stand();
  const nDat = daten.length;
  const g = training.gelernt;
  const formeln = [
    `verborgen   hⱼ = f(wⱼ₁·x₁ + wⱼ₂·x₂ + bⱼ)    f = ${AKTIVIERUNGEN[aktivierung].name}   (j = 1 … ${neuronen})`,
    "Ausgang     ŷ = sig(v₁·h₁ + … + vₕ·hₕ + c)",
    "Lernen      Backpropagation: Fehler t − ŷ rückwärts auf alle Gewichte verteilen",
  ];

  if (epoche === 0) {
    return {
      titel: "Verborgene Schicht · mehrere Linien statt einer",
      text:
        `Ein einzelnes Neuron zieht nur eine gerade Linie – damit lassen sich diese ${SZ.wort.mehrzahl} nicht trennen. ` +
        `Hier gibt es ${neuronen} versteckte Neuronen: Jedes zieht eine eigene Linie (gestrichelt), das Ausgangsneuron kombiniert sie zu einer gebogenen Grenze.`,
      zeilen: 4,
      formeln,
    };
  }
  if (g > 0 && epoche >= g) {
    return {
      titel: `Gelernt nach ${g} Epochen ✓`,
      text:
        `Alle ${nDat} ${SZ.wort.mehrzahl} liegen auf der richtigen Seite der schwarzen Grenze. Die gestrichelten Linien zeigen, wo die versteckten Neuronen „umschalten“ – zusammen formen sie die Grenze. ` +
        (neuronen > 2 ? "Wie wenige Neuronen reichen auch?" : "Mit so wenigen Neuronen klappt das nicht bei jedem Start."),
      zeilen: 4,
      formeln: [`Epoche ${epoche}:  Fehler (MSE) = ${fKomma(s.mse, 4)}   richtig ${s.richtig} von ${nDat}`, ...formeln.slice(0, 2)],
    };
  }
  if (epoche >= maxEpoche()) {
    let tipp;
    if (neuronen === 1) {
      tipp = "Mit einem versteckten Neuron gibt es nur eine Linie – das ist nicht mehr als ein einzelnes Neuron. Mehr Neuronen nehmen!";
    } else if (neuronen === 2 && szenario !== "xor" && szenario !== "zylinder") {
      tipp = "Zwei Linien reichen für einen Kreis oder ein Rechteck nicht. Mehr versteckte Neuronen nehmen!";
    } else {
      tipp = "Das Netz ist in einer Sackgasse hängen geblieben (lokales Minimum). Neue Startgewichte, eine andere Lernrate oder mehr Neuronen probieren.";
    }
    return {
      titel: `Nach ${maxEpoche()} Epochen nicht am Ziel ✗`,
      text: tipp,
      zeilen: 4,
      formeln: [`Epoche ${epoche}:  Fehler (MSE) = ${fKomma(s.mse, 4)}   richtig ${s.richtig} von ${nDat}`],
    };
  }
  return {
    titel: `Epoche ${epoche} · das Netz lernt`,
    text:
      `In jeder Epoche kommen alle ${nDat} ${SZ.wort.mehrzahl} einmal dran. Bei jedem Punkt wird der Fehler t − ŷ rückwärts durch das Netz geschickt und jedes Gewicht ein kleines Stück verbessert. ` +
      "Beobachte, wie die gestrichelten Linien wandern und die Grenze sich biegt.",
    zeilen: 4,
    formeln: [`Fehler (MSE) = ${fKomma(s.mse, 4)}   richtig ${s.richtig} von ${nDat}`, ...formeln.slice(0, 2)],
  };
}

function zeichneErklaerung() {
  const e = erklaerung();
  P_ERKL.titel = e.titel;
  panel(P_ERKL);
  zeichneErklaerText(P_ERKL, e);
}

function zeichneEinstellungen() {
  const p = P_EINST;
  panel(p);
  noStroke();
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(12);
  textAlign(LEFT, TOP);
  let t;
  if (neuronenVorschau !== null) {
    t = `Loslassen: Das Netz wird mit ${neuronenVorschau} versteckten Neuronen neu trainiert.`;
  } else if (aktivierung === "relu") {
    t = "ReLU: gibt negative Summen als 0 weiter – die Grenze bekommt Ecken.";
  } else if (aktivierung === "tanh") {
    t = "tanh: wie Sigmoid, aber von −1 bis +1 – lernt hier meist schneller.";
  } else {
    t = "Sigmoid: weicher Übergang von 0 bis 1 – nicht linear, darum können sich Linien zu Kurven verbinden.";
  }
  text(t, p.x + 16, p.y + p.h - 22);
}

// =============================================================================
// Netz-Feld: Rechnung für einen Punkt
// =============================================================================
function zeichneNetz() {
  const p = P_NETZ;
  const pkt = daten[begrenzen(auswahlIndex, 0, daten.length - 1)];
  const n = netz();
  const r = vorwaerts(n, pkt.xn, aktivierung);
  const klasse = r.y >= 0.5 ? 1 : 0;
  p.titel = `Netz 2 → ${neuronen} → 1 · ${pkt.name}`;
  panel(p);

  const ix = p.x + 74;
  const hx = p.x + 250;
  const ox = p.x + 420;
  const oben = p.y + KOPF_H + 18;
  const unten = p.y + p.h - 24;
  const iy = [p.y + 82, p.y + 134];
  const hn = n.W.length;
  const abstand = hn > 1 ? Math.min(34, (unten - oben) / (hn - 1)) : 0;
  const hStart = (oben + unten) / 2 - (abstand * (hn - 1)) / 2;
  const hy = [];
  for (let j = 0; j < hn; j++) {
    hy.push(hStart + j * abstand);
  }
  const oy = (oben + unten) / 2;
  const rh = hn > 6 ? 8 : 10;

  // Verbindungen: Farbe = Vorzeichen, Dicke = Betrag des Gewichts
  const kante = (x1, y1, x2, y2, w) => {
    const f = w >= 0 ? FARBE.gruen : FARBE.akzent;
    stroke(f[0], f[1], f[2], 90 + Math.min(150, Math.abs(w) * 40));
    strokeWeight(Math.min(5, 0.6 + Math.abs(w) * 0.8));
    line(x1, y1, x2, y2);
  };
  for (let j = 0; j < hn; j++) {
    kante(ix + 18, iy[0], hx - rh, hy[j], n.W[j][0]);
    kante(ix + 18, iy[1], hx - rh, hy[j], n.W[j][1]);
    kante(hx + rh, hy[j], ox - 22, oy, n.v[j]);
  }

  // Eingänge
  const namen = ["x₁", "x₂"];
  for (let i = 0; i < 2; i++) {
    stroke(...FARBE.linie);
    strokeWeight(1.2);
    fill(...FARBE.flaeche2);
    circle(ix, iy[i], 36);
    noStroke();
    fill(...FARBE.text);
    textFont(MONO);
    textSize(11.5);
    textAlign(CENTER, CENTER);
    text(fW(pkt.xn[i]), ix, iy[i]);
    textFont(SCHRIFT);
    textStyle(BOLD);
    textSize(13);
    textAlign(RIGHT, CENTER);
    text(namen[i], ix - 24, iy[i]);
    textStyle(NORMAL);
  }

  // versteckte Neuronen: Füllung zeigt die Aktivierung
  const hMax = aktivierung === "relu" ? Math.max(1, ...r.hh) : 1;
  for (let j = 0; j < hn; j++) {
    const a = aktivierung === "tanh" ? (r.hh[j] + 1) / 2 : r.hh[j] / hMax;
    stroke(...FARBE.senf);
    strokeWeight(1.6);
    fill(...FARBE.flaeche);
    circle(hx, hy[j], rh * 2);
    noStroke();
    fill(FARBE.senf[0], FARBE.senf[1], FARBE.senf[2], 40 + 200 * begrenzen(a, 0, 1));
    circle(hx, hy[j], rh * 2 - 4);
    textFont(MONO);
    textSize(10.5);
    const lab = `h${j + 1} = ${fW(r.hh[j])}`;
    fill(...FARBE.flaeche, 225);
    rect(hx + rh + 3, hy[j] - 7, textWidth(lab) + 6, 14, 3);
    fill(...FARBE.text);
    textAlign(LEFT, CENTER);
    text(lab, hx + rh + 6, hy[j]);
  }

  // Ausgang
  const kf = SZ.klassen[klasse].farbe;
  stroke(kf[0], kf[1], kf[2]);
  strokeWeight(2.4);
  fill(...FARBE.flaeche);
  circle(ox, oy, 44);
  noStroke();
  fill(kf[0], kf[1], kf[2]);
  textFont(MONO);
  textStyle(BOLD);
  textSize(13);
  textAlign(CENTER, CENTER);
  text(fW(r.y), ox, oy);
  textStyle(NORMAL);
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(11);
  textAlign(CENTER, BOTTOM);
  text("ŷ", ox, oy - 26);

  // rechts: Ergebnis in Worten
  const rx = p.x + 486;
  stroke(...FARBE.linie);
  strokeWeight(1);
  line(rx - 14, p.y + KOPF_H + 10, rx - 14, p.y + p.h - 12);
  const [m1, m2] = MERKMALE;
  noStroke();
  textFont(SCHRIFT);
  textAlign(LEFT, TOP);
  textSize(11.5);
  fill(...FARBE.blass);
  text("Messwerte", rx, p.y + KOPF_H + 10);
  fill(...FARBE.text);
  textSize(12);
  text(`${m1.kurz} = ${wertText(pkt.x, m1)}`, rx, p.y + KOPF_H + 26);
  text(`${m2.kurz} = ${wertText(pkt.y, m2)}`, rx, p.y + KOPF_H + 42);
  fill(...FARBE.blass);
  textSize(11.5);
  text("Ergebnis", rx, p.y + KOPF_H + 64);
  fill(kf[0], kf[1], kf[2]);
  textStyle(BOLD);
  textSize(13);
  text(`ŷ = ${fW(r.y)} → ${klassenName(klasse)}`, rx, p.y + KOPF_H + 80);
  textStyle(NORMAL);
  const richtig = klasse === pkt.k;
  fill(...(richtig ? FARBE.gruen : FARBE.akzent));
  textSize(12);
  text(richtig ? `richtig ✓ (t = ${pkt.k})` : `falsch ✗ (t = ${pkt.k})`, rx, p.y + KOPF_H + 98);
  fill(...FARBE.blass);
  textSize(10.5);
  text("Linien: grün = positives Gewicht,", rx, p.y + p.h - 30);
  text("rot = negatives, dick = groß", rx, p.y + p.h - 17);
}

// =============================================================================
// Fehler (MSE) je Epoche
// =============================================================================
function zeichneFehlerKurve() {
  const p = P_KURVE;
  panel(p);
  const gx = p.x + 50;
  const gy = p.y + KOPF_H + 14;
  const gw = p.w - 66;
  const gh = p.h - KOPF_H - 62;
  const st = training.stand;
  const maxE = Math.max(10, maxEpoche());
  let maxM = 0.05;
  for (let e = 0; e < st.length; e++) {
    maxM = Math.max(maxM, st[e].mse);
  }
  maxM = Math.min(1, maxM * 1.1);
  stroke(...FARBE.blass);
  strokeWeight(1);
  line(gx, gy + gh, gx + gw, gy + gh);
  line(gx, gy, gx, gy + gh);
  // Zielgrenze
  const zy = gy + gh - (GRENZE_MSE / maxM) * gh;
  stroke(...FARBE.gruen, 160);
  gestrichelt(true);
  line(gx, zy, gx + gw, zy);
  gestrichelt(false);
  // Verlauf bis zur aktuellen Epoche
  noFill();
  stroke(...FARBE.akzent);
  strokeWeight(2);
  beginShape();
  const schritt = Math.max(1, Math.floor(epoche / 400));
  for (let e = 0; e <= epoche; e += schritt) {
    vertex(gx + (e / maxE) * gw, gy + gh - (st[e].mse / maxM) * gh);
  }
  vertex(gx + (epoche / maxE) * gw, gy + gh - (st[epoche].mse / maxM) * gh);
  endShape();
  noStroke();
  fill(...FARBE.akzent);
  circle(gx + (epoche / maxE) * gw, gy + gh - (st[epoche].mse / maxM) * gh, 7);
  // Achsen
  fill(...FARBE.grau);
  textFont(SCHRIFT);
  textSize(10.5);
  textAlign(RIGHT, CENTER);
  text(fKomma(maxM, 2), gx - 4, gy + 4);
  text("0", gx - 4, gy + gh);
  textAlign(CENTER, TOP);
  text("0", gx, gy + gh + 3);
  text(String(maxE), gx + gw, gy + gh + 3);
  fill(...FARBE.text);
  textSize(11.5);
  textAlign(LEFT, CENTER);
  text(`Epoche ${epoche} von ${maxEpoche()}: MSE ${fKomma(st[epoche].mse, 4)}`, p.x + 16, p.y + p.h - 11);
  fill(...FARBE.gruen);
  textAlign(RIGHT, CENTER);
  text(`Ziel < ${fKomma(GRENZE_MSE, 2)}`, p.x + p.w - 16, p.y + p.h - 11);
}

// =============================================================================
// Konfusionsmatrix (Schwelle ŷ = 0,5, alle Daten)
// =============================================================================
function zeichneKonfusion() {
  const p = P_KONF;
  p.titel = `Konfusionsmatrix · Epoche ${epoche}`;
  panel(p);
  const n = netz();
  const m = [
    [0, 0],
    [0, 0],
  ];
  for (const q of daten) {
    m[q.k][vorwaerts(n, q.xn, aktivierung).y >= 0.5 ? 1 : 0]++;
  }
  const richtig = m[0][0] + m[1][1];
  const zw = 62;
  const zh = 38;
  const x0 = p.x + 120;
  const y0 = p.y + KOPF_H + 34;
  noStroke();
  fill(...FARBE.blass);
  textFont(SCHRIFT);
  textSize(10.5);
  textAlign(CENTER, BOTTOM);
  text("vorhergesagt", x0 + zw, y0 - 15);
  push();
  translate(p.x + 16, y0 + zh);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text("tatsächlich", 0, 0);
  pop();
  for (let i = 0; i < 2; i++) {
    const kl = SZ.klassen[i];
    fill(kl.farbe[0], kl.farbe[1], kl.farbe[2]);
    textSize(10.5);
    textAlign(CENTER, BOTTOM);
    text(kurz(kl.name, 11), x0 + i * zw + zw / 2, y0 - 2);
    textAlign(RIGHT, CENTER);
    text(kurz(kl.name, 12), x0 - 6, y0 + i * zh + zh / 2);
  }
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const x = x0 + j * zw;
      const y = y0 + i * zh;
      stroke(...FARBE.linie);
      strokeWeight(1);
      fill(...(i === j ? FARBE.gruenFlaeche : m[i][j] > 0 ? FARBE.rotFlaeche : FARBE.papier));
      rect(x, y, zw, zh);
      noStroke();
      fill(...(i === j ? FARBE.gruen : m[i][j] > 0 ? FARBE.akzent : FARBE.blass));
      textFont(MONO);
      textStyle(BOLD);
      textSize(16);
      textAlign(CENTER, CENTER);
      text(String(m[i][j]), x + zw / 2, y + zh / 2);
      textStyle(NORMAL);
    }
  }
  noStroke();
  fill(...FARBE.text);
  textFont(SCHRIFT);
  textSize(12);
  textAlign(LEFT, CENTER);
  text(`Genauigkeit: ${richtig} von ${daten.length} = ${prozent(richtig / daten.length, 1)}`, p.x + 16, p.y + p.h - 28);
  fill(...FARBE.blass);
  textSize(10.5);
  text("Schwelle: ŷ ≥ 0,5 → Klasse 1", p.x + 16, p.y + p.h - 12);
}

function kurz(s, n) {
  n = n || 9;
  return s.length > n ? s.slice(0, n - 1) + "." : s;
}

// -----------------------------------------------------------------------------
// Tooltip
// -----------------------------------------------------------------------------
function zeichneTooltip() {
  const [m1, m2] = MERKMALE;
  const i = punktUnterMaus();
  if (i >= 0) {
    const p = daten[i];
    const y = vorwaerts(netz(), p.xn, aktivierung).y;
    tooltip([
      `${p.name} · ${klassenName(p.k)} (t = ${p.k}) · ${wertText(p.x, m1)} · ${wertText(p.y, m2)}`,
      `ŷ = ${fW(y)} → ${klassenName(y >= 0.5 ? 1 : 0)} ${(y >= 0.5 ? 1 : 0) === p.k ? "✓" : "✗"}`,
    ]);
    return;
  }
  const mx = mausX();
  const my = mausY();
  if (imPlot(mx, my)) {
    const y = vorwaerts(netz(), plotZuN(mx, my), aktivierung).y;
    tooltip(`Netz an dieser Stelle: ŷ = ${fW(y)} → ${klassenName(y >= 0.5 ? 1 : 0)}`);
  }
}

// =============================================================================
// Eingaben
// =============================================================================
function mousePressed() {
  const i = punktUnterMaus();
  if (i >= 0) {
    auswahlIndex = i;
  }
}

function keyPressed() {
  if (!tastaturFrei()) {
    return true;
  }
  if (key === "ArrowRight") {
    if (!laufAuto) {
      schritt();
    }
    return false;
  }
  if (key === " ") {
    autoUmschalten();
    return false;
  }
  if ((key === "r" || key === "R") && !laufAuto) {
    neuStarten();
    return false;
  }
  if ((key === "e" || key === "E") && !laufAuto) {
    springe(100);
    return false;
  }
  return true;
}

// =============================================================================
// Daten als CSV speichern (z. B. für Orange, MemBrain oder Python)
// =============================================================================
function csvSpeichern() {
  const [m1, m2] = MERKMALE;
  const zeilen = [["nr", "name", m1.csv, m2.csv, "klasse", "zielwert"].join(";")];
  for (const p of daten.slice().sort((a, b) => a.id - b.id)) {
    zeilen.push([p.id, p.name, p.x.toFixed(m1.stellen), p.y.toFixed(m2.stellen), klassenName(p.k), p.k].join(";"));
  }
  ladeHerunter(SZ.csvDatei, zeilen.join("\n") + "\n");
}
