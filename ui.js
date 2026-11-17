// =============================================================================
//  ui.js – HTML-Bedienelemente über der Zeichenfläche
// =============================================================================
//  Positionen im logischen 1600×900-Raster (Helfer in basis.js).
// =============================================================================

function erstelleUi() {
  erstelleHuelle();
  erstelleBeispielReiter(szenarioWechseln);

  ui.neueDaten = knopf(1230, 12, 150, "Neue Messdaten", neueMessdaten, "kopf");
  ui.csv = knopf(1388, 12, 148, "CSV speichern", csvSpeichern, "kopf");

  // --- unter dem Diagramm ----------------------------------------------------
  ui.linien = knopf(40, 830, 300, "Linien der versteckten Neuronen", () => {
    linienAn = !linienAn;
  });
  ui.falsche = knopf(350, 830, 192, "Fehler markieren", () => {
    falscheAn = !falscheAn;
  });

  // --- Ablauf (Erklärfeld) ---------------------------------------------------
  ui.schritt = knopf(876, 284, 190, "Epoche ▶ (→)", schritt, "dunkel");
  ui.auto = knopf(1074, 284, 190, "▶ Trainieren", autoUmschalten, "primaer");
  ui.neu = knopf(1272, 284, 190, "↺ Neu starten (R)", neuStarten);
  ui.sprung = knopf(1470, 284, 92, "+100 (E)", () => springe(100));
  ui.sprung.attribute("title", "100 Epochen weiter");

  // --- Einstellungen ---------------------------------------------------------
  platziere(createSpan("Versteckte Neuronen"), 876, 380, 156);
  ui.neuronen = platziere(createSlider(1, 8, neuronen, 1), 1032, 384, 150);
  ui.neuronen.input(() => {
    neuronenVorschau = Number(ui.neuronen.value());
  });
  ui.neuronen.changed(() => {
    neuronen = Number(ui.neuronen.value());
    neuronenVorschau = null;
    neuStarten();
    ui.neuronen.elt.blur();
  });
  ui.neuronenWert = platziere(createSpan(""), 1190, 380, 30);

  platziere(createSpan("Aktivierung"), 1236, 380, 92);
  ui.aktivierung = auswahl(
    1328,
    380,
    120,
    Object.keys(AKTIVIERUNGEN).map((k) => [AKTIVIERUNGEN[k].name, k]),
    aktivierung,
    (v) => {
      aktivierung = v;
      neuStarten();
    }
  );

  platziere(createSpan("η"), 1462, 380, 20);
  ui.lernrate = auswahl(
    1484,
    380,
    80,
    ["0.1", "0.5", "1", "2"].map((v) => [v.replace(".", ","), v]),
    String(lernrate),
    (v) => {
      lernrate = parseFloat(v);
      neuStarten();
    }
  );

  platziere(createSpan("Tempo"), 876, 418, 56);
  ui.tempo = auswahl(
    932,
    418,
    110,
    ["langsam", "mittel", "schnell"].map((v) => [v, v]),
    tempo,
    (v) => {
      tempo = v;
    }
  );
  ui.stopp = kaestchen(1060, 418, 230, "beim Ziel anhalten", stoppBeimZiel, (an) => {
    stoppBeimZiel = an;
  });
  ui.start = knopf(1300, 414, 262, "🎲 Neue Startgewichte", neueStartgewichte);
}

// wird in jedem Frame aufgerufen
function aktualisiereUi() {
  const ende = epoche >= maxEpoche();
  beschrifte(ui.auto, laufAuto ? "■ Anhalten" : "▶ Trainieren");
  klasse(ui.auto, "primaer", !laufAuto);
  klasse(ui.auto, "dunkel", laufAuto);
  beschrifte(ui.schritt, ende ? "Ende" : epoche === 0 ? "1. Epoche ▶ (→)" : "Epoche ▶ (→)");
  sperre(ui.schritt, laufAuto || ende);
  sperre(ui.auto, ende);
  sperre(ui.sprung, laufAuto || ende);

  beschrifte(ui.linien, linienAn ? "Linien der Neuronen ausblenden" : "Linien der versteckten Neuronen");
  klasse(ui.linien, "kopf", true);
  klasse(ui.linien, "aktiv", linienAn);
  beschrifte(ui.falsche, falscheAn ? "Fehler ausblenden" : "Fehler markieren");
  klasse(ui.falsche, "kopf", true);
  klasse(ui.falsche, "aktiv", falscheAn);

  beschrifte(ui.neuronenWert, `<b>${neuronenVorschau || neuronen}</b>`);
  if (neuronenVorschau === null && Number(ui.neuronen.value()) !== neuronen) {
    ui.neuronen.value(neuronen);
  }
  if (ui.aktivierung.value() !== aktivierung) {
    ui.aktivierung.selected(aktivierung);
  }
  if (ui.lernrate.value() !== String(lernrate)) {
    ui.lernrate.selected(String(lernrate));
  }

  aktualisiereBeispielReiter(szenario, laufAuto);
  sperre(ui.neueDaten, laufAuto || !!SZ.fest);
  for (const el of [ui.neuronen, ui.aktivierung, ui.lernrate, ui.neu, ui.start]) {
    sperre(el, laufAuto);
  }
}
