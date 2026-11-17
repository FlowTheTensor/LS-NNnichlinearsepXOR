// =============================================================================
//  daten.js – Daten für die Beispiele (zwei Merkmale, jeweils MIT Klasse)
// =============================================================================
//  Keines dieser Beispiele lässt sich mit einer einzigen geraden Linie trennen
//  (nicht linear separierbar). Ein Netz mit verborgener Schicht schafft es:
//  Jedes versteckte Neuron zieht eine eigene Linie, das Ausgangsneuron
//  kombiniert sie.
//
//  Beispiel "xor" (Standard):
//    Logisches XOR mit vier Mustern – wie in der Praxiswerkstatt MemBrain (nlsP).
//
//  Beispiel "zylinder":
//    Pneumatikzylinder mit zwei Endlagensensoren. Genau einer muss melden.
//    Beide oder keiner = Störung. Das ist ein XOR mit Messrauschen.
//
//  Beispiel "bohrung":
//    Prüfstation Bohrbild: i. O. nur innerhalb eines Toleranzkreises.
//
//  Beispiel "fenster":
//    Spritzgießmaschine: Gutteile nur in einem Prozessfenster aus Temperatur
//    und Druck (Rechteck mit vier Seiten → mindestens vier Linien).
// =============================================================================

const SZENARIEN = {
  xor: {
    name: "Logik · XOR",
    titel: "Wahrheitstabelle · logisches XOR",
    untertitel: "wie in der Praxiswerkstatt MemBrain (nlsP) · verborgene Schicht · nicht linear trennbar",
    merkmale: [
      { name: "Eingang A", kurz: "A", einheit: "", min: -0.25, max: 1.25, gitter: 0.25, stellen: 0, csv: "a" },
      { name: "Eingang B", kurz: "B", einheit: "", min: -0.25, max: 1.25, gitter: 0.25, stellen: 0, csv: "b" },
    ],
    klassen: [
      { name: "Ausgang 0", farbe: FARBE.blau, form: "kreis" },
      { name: "Ausgang 1", farbe: FARBE.akzent, form: "dreieck" },
    ],
    fest: [
      [0, 0, 0, "0 XOR 0"],
      [0, 1, 1, "0 XOR 1"],
      [1, 0, 1, "1 XOR 0"],
      [1, 1, 0, "1 XOR 1"],
    ],
    neuronen: 4,
    lernrate: 1,
    maxEpochen: 3000,
    wort: { einzahl: "Muster", mehrzahl: "Muster" },
    csvDatei: "logik_xor.csv",
  },

  zylinder: {
    name: "Zylinder · Endlagen",
    titel: "Sensordaten · Pneumatikzylinder",
    untertitel: "Plausibilitätsprüfung · genau ein Endlagensensor darf melden (XOR mit Rauschen)",
    startwert: 20261121,
    merkmale: [
      { name: "Sensor hinten", kurz: "S_hinten", einheit: "V", min: 0, max: 24, gitter: 4, stellen: 1, csv: "sensor_hinten_v" },
      { name: "Sensor vorne", kurz: "S_vorne", einheit: "V", min: 0, max: 24, gitter: 4, stellen: 1, csv: "sensor_vorne_v" },
    ],
    klassen: [
      { name: "Störung", farbe: FARBE.blau, form: "kreis" },
      { name: "plausibel", farbe: FARBE.akzent, form: "dreieck" },
    ],
    // Punktwolken an den vier Ecken: [Mittelwert x, Mittelwert y, Streuung, Anzahl, Klasse]
    ecken: [
      [3.5, 3, 1.3, 9, 0],
      [20.5, 3, 1.3, 10, 1],
      [3.5, 21, 1.3, 10, 1],
      [20.5, 21, 1.3, 9, 0],
    ],
    neuronen: 4,
    lernrate: 0.5,
    maxEpochen: 800,
    wort: { einzahl: "Messung", mehrzahl: "Messungen" },
    csvDatei: "sensordaten_zylinder.csv",
  },

  bohrung: {
    name: "Bohrung · Toleranzkreis",
    titel: "Messdaten · Prüfstation Bohrbild",
    untertitel: "Qualitätsprüfung · i. O. nur innerhalb des Toleranzkreises",
    startwert: 20261122,
    merkmale: [
      { name: "Abweichung x", kurz: "Δx", einheit: "µm", min: -50, max: 50, gitter: 10, stellen: 0, csv: "abweichung_x_um" },
      { name: "Abweichung y", kurz: "Δy", einheit: "µm", min: -50, max: 50, gitter: 10, stellen: 0, csv: "abweichung_y_um" },
    ],
    klassen: [
      { name: "i. O.", farbe: FARBE.blau, form: "kreis" },
      { name: "n. i. O.", farbe: FARBE.akzent, form: "dreieck" },
    ],
    kreis: { r: 28, n: 70, rand: 4 },
    neuronen: 5,
    lernrate: 0.5,
    maxEpochen: 800,
    wort: { einzahl: "Bohrung", mehrzahl: "Bohrungen" },
    csvDatei: "messdaten_bohrbild.csv",
  },

  fenster: {
    name: "Spritzguss · Prozessfenster",
    titel: "Prozessdaten · Spritzgießmaschine",
    untertitel: "Qualitätsprüfung · Gutteile nur in einem Fenster aus Temperatur und Druck",
    startwert: 20261123,
    merkmale: [
      { name: "Werkzeugtemperatur", kurz: "T", einheit: "°C", min: 180, max: 260, gitter: 10, stellen: 0, csv: "temperatur_c" },
      { name: "Einspritzdruck", kurz: "p", einheit: "bar", min: 600, max: 1400, gitter: 100, stellen: 0, csv: "druck_bar" },
    ],
    klassen: [
      { name: "Gutteil", farbe: FARBE.blau, form: "kreis" },
      { name: "Ausschuss", farbe: FARBE.akzent, form: "dreieck" },
    ],
    rechteck: { x0: 205, x1: 242, y0: 880, y1: 1250, n: 80, randX: 3, randY: 30 },
    neuronen: 6,
    lernrate: 0.5,
    maxEpochen: 800,
    wort: { einzahl: "Bauteil", mehrzahl: "Bauteile" },
    csvDatei: "prozessdaten_spritzguss.csv",
  },
};

// aktives Beispiel – wird mit setzeSzenario() umgeschaltet
let SZ = SZENARIEN.xor;
let MERKMALE = SZ.merkmale;

function setzeSzenario(schluessel) {
  SZ = SZENARIEN[schluessel];
  MERKMALE = SZ.merkmale;
}

function runde(v, m) {
  return Number(v.toFixed(m.stellen));
}

function erzeugeDaten(startwert) {
  const [m1, m2] = MERKMALE;
  const liste = [];
  const neu = (x, y, k, name) => {
    liste.push({
      x: runde(begrenzen(x, m1.min, m1.max), m1),
      y: runde(begrenzen(y, m2.min, m2.max), m2),
      k,
      name,
    });
  };
  if (SZ.fest) {
    for (const [x, y, k, name] of SZ.fest) {
      neu(x, y, k, name);
    }
  } else {
    const rnd = zufallsGenerator(startwert);
    if (SZ.ecken) {
      for (const [mx, my, s, n, k] of SZ.ecken) {
        for (let i = 0; i < n; i++) {
          neu(mx + gaussZahl(rnd) * s, my + gaussZahl(rnd) * s, k);
        }
      }
    } else if (SZ.kreis) {
      const kr = SZ.kreis;
      let n = 0;
      while (n < kr.n) {
        const x = (rnd() * 2 - 1) * 48;
        const y = (rnd() * 2 - 1) * 48;
        const d = Math.hypot(x, y);
        // nahe am Rand weglassen, damit die Klassen klar getrennt sind
        if (Math.abs(d - kr.r) < kr.rand) {
          continue;
        }
        neu(x, y, d < kr.r ? 0 : 1);
        n++;
      }
    } else if (SZ.rechteck) {
      const r = SZ.rechteck;
      let n = 0;
      while (n < r.n) {
        const x = m1.min + rnd() * (m1.max - m1.min);
        const y = m2.min + rnd() * (m2.max - m2.min);
        const drinX = x > r.x0 && x < r.x1;
        const drinY = y > r.y0 && y < r.y1;
        const nahX = Math.abs(x - r.x0) < r.randX || Math.abs(x - r.x1) < r.randX;
        const nahY = Math.abs(y - r.y0) < r.randY || Math.abs(y - r.y1) < r.randY;
        if ((nahX && (drinY || nahY)) || (nahY && (drinX || nahX))) {
          continue;
        }
        neu(x, y, drinX && drinY ? 0 : 1);
        n++;
      }
    }
    // in zufälliger Reihenfolge (wie gemessen)
    for (const p of liste) {
      p.r = rnd();
    }
    liste.sort((a, b) => a.r - b.r);
  }
  for (let i = 0; i < liste.length; i++) {
    liste[i].id = (SZ.fest ? 1 : 101) + i;
    if (!liste[i].name) {
      liste[i].name = `${SZ.wort.einzahl} ${liste[i].id}`;
    }
    delete liste[i].r;
  }
  return liste;
}

// Merkmale auf −1 … +1 umrechnen (normieren) – damit rechnet das Netz
function normiere(p) {
  const [m1, m2] = MERKMALE;
  return [(2 * (p.x - m1.min)) / (m1.max - m1.min) - 1, (2 * (p.y - m2.min)) / (m2.max - m2.min) - 1];
}

function wertText(v, m) {
  return v.toFixed(m.stellen).replace(".", ",").replace("-", "−") + (m.einheit ? " " + m.einheit : "");
}

if (typeof module !== "undefined") {
  module.exports = { SZENARIEN, setzeSzenario, erzeugeDaten, normiere };
}
