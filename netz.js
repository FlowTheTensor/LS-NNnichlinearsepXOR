// =============================================================================
//  netz.js – kleines neuronales Netz 2 → h → 1 (reine Logik, ohne p5)
// =============================================================================
//  Eingänge x₁, x₂ (normiert auf −1 … +1)
//  verborgene Schicht (hidden layer) mit h Neuronen:
//      zⱼ = wⱼ₁·x₁ + wⱼ₂·x₂ + bⱼ        hⱼ = f(zⱼ)      f = Sigmoid, tanh oder ReLU
//  Ausgangsneuron:
//      z = Σ vⱼ·hⱼ + c                  ŷ = sig(z)      (Wahrscheinlichkeit für Klasse 1)
//
//  Training mit Fehlerrückführung (Backpropagation), Punkt für Punkt:
//      Fehler        e = t − ŷ
//      Ausgang       δ = e · ŷ·(1 − ŷ)
//      verborgen     δⱼ = δ · vⱼ · f'(zⱼ)
//      Anpassung     vⱼ += η·δ·hⱼ    c += η·δ
//                    wⱼᵢ += η·δⱼ·xᵢ   bⱼ += η·δⱼ
//
//  Alle Epochen werden vorab berechnet (trainiere). Die Oberfläche blättert
//  nur durch die gespeicherten Zwischenstände.
// =============================================================================

const AKTIVIERUNGEN = {
  sigmoid: {
    name: "Sigmoid",
    f: (z) => 1 / (1 + Math.exp(-z)),
    df: (z, h) => h * (1 - h),
  },
  tanh: {
    name: "tanh",
    f: (z) => Math.tanh(z),
    df: (z, h) => 1 - h * h,
  },
  relu: {
    name: "ReLU",
    f: (z) => (z > 0 ? z : 0),
    df: (z) => (z > 0 ? 1 : 0),
  },
};

function sig(z) {
  return 1 / (1 + Math.exp(-z));
}

// Startgewichte (zufällig, aber mit festem Startwert reproduzierbar)
//   W[j] = [wⱼ₁, wⱼ₂], b[j], v[j], c
function startNetz(h, rnd) {
  const z = () => rnd() * 2 - 1;
  const netz = { W: [], b: [], v: [], c: z() * 0.5 };
  for (let j = 0; j < h; j++) {
    netz.W.push([z() * 1.5, z() * 1.5]);
    netz.b.push(z());
    netz.v.push(z());
  }
  return netz;
}

function kopiereNetz(n) {
  return { W: n.W.map((w) => [w[0], w[1]]), b: n.b.slice(), v: n.v.slice(), c: n.c };
}

// Vorwärtsrechnung: liefert alle Zwischenwerte
function vorwaerts(netz, x, aktivierung) {
  const f = AKTIVIERUNGEN[aktivierung].f;
  const zh = [];
  const hh = [];
  let z = netz.c;
  for (let j = 0; j < netz.W.length; j++) {
    const zj = netz.W[j][0] * x[0] + netz.W[j][1] * x[1] + netz.b[j];
    const hj = f(zj);
    zh.push(zj);
    hh.push(hj);
    z += netz.v[j] * hj;
  }
  return { zh, hh, z, y: sig(z) };
}

// ein Lernschritt (Backpropagation) für einen Punkt – verändert netz
function lerne(netz, x, t, eta, aktivierung) {
  const df = AKTIVIERUNGEN[aktivierung].df;
  const r = vorwaerts(netz, x, aktivierung);
  const e = t - r.y;
  const delta = e * r.y * (1 - r.y);
  for (let j = 0; j < netz.W.length; j++) {
    const dj = delta * netz.v[j] * df(r.zh[j], r.hh[j]);
    netz.v[j] += eta * delta * r.hh[j];
    netz.W[j][0] += eta * dj * x[0];
    netz.W[j][1] += eta * dj * x[1];
    netz.b[j] += eta * dj;
  }
  netz.c += eta * delta;
  return e;
}

function bewerte(netz, daten, aktivierung) {
  let mse = 0;
  let richtig = 0;
  for (const p of daten) {
    const y = vorwaerts(netz, p.xn, aktivierung).y;
    mse += (p.k - y) * (p.k - y);
    if ((y >= 0.5 ? 1 : 0) === p.k) {
      richtig++;
    }
  }
  return { mse: mse / daten.length, richtig };
}

// Training vorab: Zwischenstand nach jeder Epoche
//   ergebnis.stand[e] = {netz, mse, richtig}  (e = 0: Startgewichte)
//   ergebnis.gelernt  = erste Epoche, ab der alle richtig sind und mse < GRENZE_MSE (oder −1)
const GRENZE_MSE = 0.02;

function trainiere(daten, h, aktivierung, eta, startwert, maxEpochen) {
  const rnd = zufallsGenerator(startwert);
  const netz = startNetz(h, rnd);
  const reihenfolge = daten.map((_, i) => i);
  const stand = [{ netz: kopiereNetz(netz), ...bewerte(netz, daten, aktivierung) }];
  let gelernt = -1;
  for (let ep = 1; ep <= maxEpochen; ep++) {
    // jede Epoche in neuer, aber reproduzierbarer Reihenfolge
    for (let i = reihenfolge.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const tmp = reihenfolge[i];
      reihenfolge[i] = reihenfolge[j];
      reihenfolge[j] = tmp;
    }
    for (const i of reihenfolge) {
      lerne(netz, daten[i].xn, daten[i].k, eta, aktivierung);
    }
    const b = bewerte(netz, daten, aktivierung);
    stand.push({ netz: kopiereNetz(netz), ...b });
    if (gelernt < 0 && b.richtig === daten.length && b.mse < GRENZE_MSE) {
      gelernt = ep;
    }
  }
  return { stand, gelernt };
}

// Linie zⱼ = 0 eines versteckten Neurons im normierten Quadrat [−1, 1]²
function neuronLinie(w1, w2, b) {
  if (Math.abs(w1) < 1e-12 && Math.abs(w2) < 1e-12) {
    return null;
  }
  const pkt = [];
  const add = (x, y) => {
    if (x >= -1 - 1e-9 && x <= 1 + 1e-9 && y >= -1 - 1e-9 && y <= 1 + 1e-9) {
      for (const q of pkt) {
        if (Math.abs(q[0] - x) < 1e-9 && Math.abs(q[1] - y) < 1e-9) {
          return;
        }
      }
      pkt.push([x, y]);
    }
  };
  if (Math.abs(w2) > 1e-12) {
    add(-1, (-b + w1) / w2);
    add(1, (-b - w1) / w2);
  }
  if (Math.abs(w1) > 1e-12) {
    add((-b + w2) / w1, -1);
    add((-b - w2) / w1, 1);
  }
  return pkt.length >= 2 ? [pkt[0], pkt[1]] : null;
}

if (typeof module !== "undefined") {
  module.exports = { AKTIVIERUNGEN, sig, startNetz, vorwaerts, lerne, bewerte, trainiere, neuronLinie, GRENZE_MSE };
}
