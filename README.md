# KI-Klassifikation · Verborgene Schicht · nicht linear trennbar (XOR)

Ein kleines neuronales Netz zum Zuschauen und Ausprobieren: Zwei Eingänge, eine **verborgene Schicht** mit 1 bis 8 Neuronen und ein Ausgangsneuron lernen Klassen, die sich mit **keiner geraden Linie** trennen lassen – allen voran das logische XOR. Jedes versteckte Neuron zieht eine eigene Linie, das Ausgangsneuron kombiniert sie zu einer gebogenen Grenze. Gelernt wird mit **Backpropagation**, Epoche für Epoche.

Erstellt für den Unterricht an der Jakob-Preh-Schule ([bsnes.de](https://www.bsnes.de)) im Fach Konzepte der KI (KoKI), Skript Kap. 5.6 Neuronale Netze, Lerngebiet 1 und 2. Gegenstück zur Lernsituation **KI-Klassifikation: Ein Neuron** (LS-NNLineareseparation).

**Live-Version:** [Hier direkt anschauen](https://flowthetensor.github.io/LS-NNnichlinearsepXOR/)

---

## Worum geht es?

Ein einzelnes Neuron trennt zwei Klassen nur durch eine gerade Linie. Beim XOR liegen die beiden Klassen über Kreuz – eine Linie reicht nicht. Die Lösung ist eine Schicht dazwischen:

```
2 Eingänge (x₁, x₂)  →  h versteckte Neuronen  →  1 Ausgang ŷ

verborgen   hⱼ = f(wⱼ₁·x₁ + wⱼ₂·x₂ + bⱼ)        f = Sigmoid, tanh oder ReLU
Ausgang     ŷ  = sig(v₁·h₁ + … + vₕ·hₕ + c)      Wahrscheinlichkeit für Klasse 1
```

- Jedes versteckte Neuron hat seine eigene Linie zⱼ = 0 (gestrichelt, nummeriert). Dort „schaltet“ es um.
- Weil die Aktivierungsfunktion **nicht linear** ist, kann das Ausgangsneuron diese Linien zu einer gebogenen Grenze verbinden (schwarze Linie: ŷ = 0,5).
- Die Hintergrundfarbe zeigt die Ausgabe ŷ an jeder Stelle: blau ≈ Klasse 0, rot ≈ Klasse 1, je kräftiger, desto sicherer.

Die Merkmale werden vorher auf −1 … +1 umgerechnet (**normiert**). Die Zielwerte sind t = 0 und t = 1.

In der roten **Beispiel-Leiste** stehen vier Beispiele zur Verfügung:

| Beispiel | Merkmale | Klassen | Form der Grenze | Standard |
|---|---|---|---|---|
| **Logik · XOR** | Eingang A, B (0 oder 1) | Ausgang 0 / 1 | über Kreuz, 4 Muster wie in MemBrain (nlsP) | 4 Neuronen, η = 1 |
| **Zylinder · Endlagen** | Sensor hinten, Sensor vorne (0 … 24 V) | Störung / plausibel | XOR mit Messrauschen: genau ein Endlagensensor darf melden | 4 Neuronen, η = 0,5 |
| **Bohrung · Toleranzkreis** | Abweichung Δx, Δy (−50 … +50 µm) | i. O. / n. i. O. | Kreis mit Radius 28 µm | 5 Neuronen, η = 0,5 |
| **Spritzguss · Prozessfenster** | Werkzeugtemperatur (180 … 260 °C), Einspritzdruck (600 … 1400 bar) | Gutteil / Ausschuss | Rechteck 205 … 242 °C und 880 … 1250 bar | 6 Neuronen, η = 0,5 |

> Die Industrie-Werte sind simuliert und an eine typische 24-V-Modellanlage für Industrie 4.0 angelehnt. Punkte direkt an der Grenze werden weggelassen, damit die Klassen sauber getrennt sind.

Der **Pneumatikzylinder** ist ein XOR aus der Praxis: Fährt der Zylinder sauber, meldet genau einer der beiden Endlagensensoren. Melden beide oder keiner, stimmt etwas nicht (Sensor verstellt, Kabelbruch, Zylinder hängt in der Mitte).

## Training mit Backpropagation

In jeder **Epoche** kommen alle Trainingspunkte einmal dran, in zufälliger Reihenfolge. Für jeden Punkt:

```
Fehler        e  = t − ŷ
Ausgang       δ  = e · ŷ·(1 − ŷ)
verborgen     δⱼ = δ · vⱼ · f'(zⱼ)
Anpassung     vⱼ ← vⱼ + η·δ·hⱼ       c  ← c  + η·δ
              wⱼᵢ ← wⱼᵢ + η·δⱼ·xᵢ     bⱼ ← bⱼ + η·δⱼ
```

Das Feld **„Fehler (MSE) je Epoche“** zeigt den mittleren quadratischen Fehler. Das Ziel ist erreicht, wenn alle Punkte richtig sind und der Fehler unter 0,02 liegt. Mit „beim Ziel anhalten“ stoppt das Training dort.

## Bedienung

| Aktion | Wirkung |
|---|---|
| Beispiel-Leiste (Reiter 1–4) | XOR, Zylinder, Bohrung oder Spritzguss |
| **→** oder „Epoche“ | eine Epoche weiter |
| **E** oder „+100“ | 100 Epochen weiter |
| **Leertaste** oder „Trainieren“ | Training laufen lassen / anhalten |
| **R** oder „Neu starten“ | wieder mit den Startgewichten beginnen |
| Regler „Versteckte Neuronen“ | 1 … 8; beim Loslassen wird neu trainiert |
| Aktivierung | Sigmoid, tanh oder ReLU |
| η | Lernrate 0,1 … 2 |
| „Neue Startgewichte“ | andere zufällige Startgewichte |
| Klick auf einen Punkt | Punkt im Netz-Feld nachrechnen (Werte aller versteckten Neuronen und ŷ) |
| Maus über dem Diagramm | Ausgabe ŷ des Netzes an dieser Stelle |
| Mond / Sonne oben rechts | dunkles oder helles Farbschema; die Wahl gilt für alle Lernsituationen |

Unter dem Diagramm lassen sich die Linien der versteckten Neuronen und die Markierung der Fehler (gelbe Ringe) ein- und ausblenden. „Neue Messdaten“ erzeugt neue Daten (nicht bei XOR), „CSV speichern“ lädt die Daten herunter.

## Ideen für den Unterricht

- **Warum braucht es die Schicht?** Erst in der LS „Ein Neuron“ das XOR versuchen, dann hier mit 1 verstecktem Neuron. Was ist der Unterschied zwischen 1 verstecktem Neuron und einem einzelnen Neuron?
- **Wie viele Neuronen reichen?** Für XOR, Kreis und Rechteck jeweils die kleinste Zahl versteckter Neuronen suchen, mit der das Netz zuverlässig lernt. Warum braucht das Rechteck mehr als das XOR?
- **Linien lesen:** Nach dem Training die gestrichelten Linien ansehen. Welche Linie gehört zu welcher Seite der Grenze? Gibt es Neuronen, die (fast) nichts beitragen?
- **Startgewichte:** Mehrmals „Neue Startgewichte“ mit 2 versteckten Neuronen beim XOR. Warum klappt es manchmal nicht (lokales Minimum)?
- **Aktivierungsfunktionen:** Sigmoid, tanh und ReLU vergleichen. Welche lernt am schnellsten? Wie sieht die Grenze bei ReLU aus?
- **Lernrate:** η = 0,1 und η = 2 vergleichen. Was passiert mit der Fehlerkurve?
- **Brücke zu MemBrain (nlsP):** Dasselbe Netz in MemBrain bauen und die Fehlerkurven vergleichen.
- **Praxis Zylinder:** Welche Fehlerbilder stecken hinter „beide Sensoren melden“ und „keiner meldet“? Warum ist hier ein KI-Modell eigentlich übertrieben – und wann wäre es sinnvoll?
- **Ausblick:** Dasselbe Prinzip mit mehr Schichten und mehr Neuronen steckt in CNNs und Sprachmodellen (LS Neuronales Netz, Notebook NN3 mit TensorFlow).

## Dateien

| Datei | Inhalt |
|---|---|
| `index.html` | Seite, bindet Schriften, p5.js und die Skripte ein |
| `style.css` | Gestaltung im Corporate Design der Schule |
| `basis.js` | gemeinsame Grundlage der KI-Lernsituationen (Layout, Farben, Kopfzeile, Beispiel-Leiste, UI-Helfer) |
| `daten.js` | die vier Beispiele (Merkmale, Klassen, Daten, Standardeinstellungen), der Datengenerator und die Normierung |
| `netz.js` | das Netz – Vorwärtsrechnung, Backpropagation, Training aller Epochen vorab – reines JavaScript, ohne p5 |
| `ui.js` | Knöpfe, Regler, Auswahllisten, Kästchen |
| `sketch.js` | Ablauf, Zeichnen (Ausgabekarte, Grenze, Linien, Netz), Erklärtexte |
| `assets/logo-jps.png` | Logo der Jakob-Preh-Schule |
| `assets/logo-jps-hell.png` | Logo in Weiß für das dunkle Farbschema |
| `libraries/p5.min.js` | p5.js (lokal, damit die Seite auch offline läuft) |

Die Dateien `matrix.js`, `nn.js` und `libraries/p5.sound.min.js` stammen aus der ersten Fassung und werden nicht mehr gebraucht.

## Ein eigenes Beispiel ergänzen

In `daten.js` einen Eintrag in `SZENARIEN` hinzufügen. Er erscheint automatisch als weiterer Reiter:

```js
mein: {
  name: "Mein Beispiel",
  titel: "Meine Messdaten",
  untertitel: "Worum es geht · verborgene Schicht",
  merkmale: [
    { name: "Merkmal 1", kurz: "m1", einheit: "V", min: 0, max: 10, gitter: 1, stellen: 1, csv: "m1_v" },
    { name: "Merkmal 2", kurz: "m2", einheit: "V", min: 0, max: 10, gitter: 1, stellen: 1, csv: "m2_v" },
  ],
  klassen: [
    { name: "Klasse A", farbe: FARBE.blau, form: "kreis" },
    { name: "Klasse B", farbe: FARBE.akzent, form: "dreieck" },
  ],
  fest: [
    [2, 2, 0, "Teil 1"],
    [8, 2, 1, "Teil 2"],
  ],
  neuronen: 4,
  lernrate: 0.5,
  maxEpochen: 800,
  wort: { einzahl: "Teil", mehrzahl: "Teile" },
  csvDatei: "meine_daten.csv",
},
```

Statt `fest` gehen auch Punktwolken an Ecken (`ecken`, mit `startwert`), ein Kreis (`kreis`) oder ein Rechteck (`rechteck`), siehe die vorhandenen Beispiele.

## Lokal starten

Am einfachsten in VS Code mit der Erweiterung **Live Server**: Rechtsklick auf `index.html` → „Open with Live Server“.

## Technische Hinweise

- Läuft mit **p5.js 1.x** (lokal mitgeliefert) und **p5.js 2.x**. Dafür verzichtet der Code bewusst auf Funktionen, die sich zwischen den Versionen geändert haben (z. B. `preload`, `loadFont`).
- Die Schriften (Quicksand, Inter, JetBrains Mono) kommen von Google Fonts. Ohne Internet weicht die Seite automatisch auf Systemschriften aus.
- Alle Schleifen haben geschweifte Klammern: Der p5-Web-Editor baut in Schleifen eine Endlosschleifen-Sicherung ein, die bei Schleifen ohne Klammern fehlschlägt.
- **Vorberechnung:** Beim Start (und bei jeder Änderung der Einstellungen) werden alle Epochen mit festem Startwert vorab trainiert (`trainiere` in `netz.js`). Die Oberfläche blättert nur durch die gespeicherten Zwischenstände – „Epoche“ und „+100“ sind deshalb sofort da, und jeder Lauf ist wiederholbar.
- Die Ausgabekarte wird pixelweise berechnet und zwischengespeichert, die Grenze ŷ = 0,5 mit Marching Squares auf einem groben Raster gezeichnet.
- Fehler in `draw()` werden abgefangen und als Hinweis angezeigt, statt das Programm einzufrieren.
- `basis.js` und `style.css` sind dieselben wie in den anderen KI-Lernsituationen; das Farbschema hell/dunkel wird über den gemeinsamen `localStorage`-Schlüssel `jps-lernsituationen-farbschema` für alle Lernsituationen gemerkt.
- Vereinfachungen: Das Ausgangsneuron hat immer Sigmoid, der Fehler ist der quadratische Fehler (wie im Notebook „Eisverkäufer“), Training Punkt für Punkt ohne Momentum.
- Die erste Fassung beruhte auf dem *Toy-Neural-Network-JS* von Daniel Shiffman (The Coding Train); diese Fassung ist neu geschrieben.

## Lizenz

[![CC BY 4.0](https://img.shields.io/badge/Lizenz-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/deed.de)

**KI-Klassifikation · Verborgene Schicht · nicht linear trennbar (XOR)** von **Christoph Rothaug, Jakob-Preh-Schule Bad Neustadt**, steht unter der Lizenz [Creative Commons Namensnennung 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.de). Der vollständige Lizenztext steht in der Datei [`LICENSE`](LICENSE).

Du darfst das Material (Quellcode, Texte, Aufgaben und Grafiken) kopieren, weitergeben und verändern, auch kommerziell. Dafür musst du

- den Urheber nennen: *Christoph Rothaug, Jakob-Preh-Schule Bad Neustadt*,
- auf die Lizenz verlinken: <https://creativecommons.org/licenses/by/4.0/deed.de>,
- angeben, ob du etwas verändert hast.

Vorschlag für die Namensnennung: *„KI-Klassifikation · Verborgene Schicht · nicht linear trennbar (XOR)“ von Christoph Rothaug, Jakob-Preh-Schule Bad Neustadt, CC BY 4.0, https://github.com/FlowTheTensor/LS-NNnichlinearsepXOR*

**Ausgenommen** von dieser Lizenz sind:

| Bestandteil | Rechte |
|---|---|
| `libraries/p5.min.js` | [p5.js](https://p5js.org), GNU LGPL 2.1 |
| `assets/logo-jps.png`, `assets/logo-jps-hell.png` | Logo der Jakob-Preh-Schule – alle Rechte bei der Schule |
| Schriften Quicksand, Inter, JetBrains Mono | über Google Fonts eingebunden, SIL Open Font License 1.1 |
