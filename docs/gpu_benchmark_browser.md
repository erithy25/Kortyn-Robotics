# GPU-Messung im Browser – Schritt für Schritt

Diese Anleitung ist für Menschen ohne Programmierkenntnisse. Du brauchst nur einen Browser und ein kostenloses
Konto bei **Google** (für Colab) oder **Kaggle**. Kosten: keine. Zeitaufwand: etwa eine Stunde, davon rund fünf
Minuten Klicken; den Rest rechnet die GPU allein.

**Worum es geht:** Kortyn hat bisher nur auf normalen Prozessoren gerechnet und die Rechenzeit einer
Grafikkarte (GPU) angenommen. Das Notebook misst diese Zeit jetzt echt: Wie lange braucht das Robotermodell
SmolVLA auf einer NVIDIA-GPU für eine Anfrage, für 2, 4, 8, 16 oder 32 gleichzeitige Anfragen, in mehreren
Rechengenauigkeiten. Heraus kommt eine kleine Datei (JSON), die du ins Repository hochlädst. Mit ihr rechnet
Kortyn die vorab festgelegte Entscheidungsregel: Bedient diese GPU mindestens 10 Roboter so, dass 99 % der
Antworten in höchstens 150 ms beim Roboter sind?

---

## Variante A: Google Colab (empfohlen)

**Voraussetzung:** ein Google-Konto.

1. **Notebook öffnen.** Klicke auf diesen Link:
   <https://colab.research.google.com/github/erithy25/Kortyn-Robotics/blob/phase-5-publish/notebooks/gpu_benchmark.ipynb>
   - Colab zeigt eventuell „Warnung: Dieses Notebook wurde nicht von Google erstellt“. Das ist normal für jedes
     fremde Notebook. Klicke **„Trotzdem ausführen“**, wenn danach gefragt wird.
   - Falls der Link nicht geht: Öffne die Datei auf GitHub
     (<https://github.com/erithy25/Kortyn-Robotics/blob/phase-5-publish/notebooks/gpu_benchmark.ipynb>),
     klicke rechts oben auf das Download-Symbol („Download raw file“) und lade die Datei in Colab über
     **Datei → Notebook hochladen**.
2. **GPU einschalten.** Oben im Menü **„Laufzeit“ → „Laufzeittyp ändern“**. Unter „Hardwarebeschleuniger“
   **„T4-GPU“** wählen und **„Speichern“** klicken.
3. **Starten.** Menü **„Laufzeit“ → „Alle ausführen“**.
4. **Warten.** Unter jedem Schritt erscheint, was gerade passiert:
   - Schritt 1 meldet „GPU gefunden: Tesla T4 …“.
   - Schritt 2 installiert alles (etwa 5 Minuten).
   - Schritt 3 misst (etwa 30–60 Minuten). Zeilen wie `batch  4: Median 120.5 ms` zeigen den Fortschritt.

   Lass den Browser-Tab offen und den Computer wach (kein Ruhezustand). Du musst nichts tun.
5. **Herunterladen.** Schritt 4 startet den Download selbst. Falls der Browser fragt, ob die Seite Dateien
   herunterladen darf: **erlauben**. Die Datei heißt etwa `smolvla_gpu_teslat4_colab_20261028-0930.json`.
   - Kein Download? Links auf das **Ordnersymbol** klicken, die Datei unter `/content` mit der rechten Maustaste
     anklicken → **„Herunterladen“**.
6. **Ins Repository hochladen:** siehe [unten](#datei-ins-repository-hochladen).

---

## Variante B: Kaggle

**Voraussetzung:** ein Kaggle-Konto mit **Telefon-Verifizierung**. Ohne sie gibt Kaggle weder GPU noch Internet
frei. Einmalig unter deinem Profilbild → **Settings → Phone verification**.

1. **Notebook importieren.** Auf <https://www.kaggle.com> → **„Create“ → „New Notebook“**. Dann im Menü
   **„File“ → „Import Notebook“** und diesen Link einfügen:
   <https://github.com/erithy25/Kortyn-Robotics/blob/phase-5-publish/notebooks/gpu_benchmark.ipynb>
   (oder die heruntergeladene `.ipynb`-Datei hochladen, siehe Variante A, Schritt 1).
2. **GPU und Internet einschalten.** Rechts im Bereich **„Session options“**: bei **„Accelerator“**
   **„GPU T4 x2“** wählen, bei **„Internet“** einschalten.
3. **Starten.** Oben **„Run All“**.
4. **Warten** wie bei Colab (etwa eine Stunde).
5. **Herunterladen.** Rechts unter **„Output“** den Ordner `/kaggle/working` öffnen, auf die Datei
   `smolvla_gpu_….json` zeigen → **„⋮“ → „Download“**.
6. **Ins Repository hochladen:** siehe unten.

---

## Datei ins Repository hochladen

1. Melde dich auf <https://github.com> an.
2. Öffne den Zielordner:
   <https://github.com/erithy25/Kortyn-Robotics/tree/phase-5-publish/results/phase5/gpu>
   Oben links muss als Branch **`phase-5-publish`** stehen.
3. Rechts oben **„Add file“ → „Upload files“**. Die heruntergeladene Datei ins Fenster ziehen.
4. Unten bei „Commit changes“ eine kurze Beschreibung eintragen, z. B. `GPU-Messung T4 (Colab)`. Die Option
   **„Commit directly to the `phase-5-publish` branch“** auswählen und **„Commit changes“** klicken.

Fertig. Kortyn rechnet die Entscheidungsregel danach mit einem einzigen Befehl:

```bash
uv run python -m kortyn.phase5_steps gpu-decision
```

Mehrere Messungen (zum Beispiel Colab **und** Kaggle) sind willkommen: einfach jede Datei hochladen. Jede wird
einzeln ausgewertet.

---

## Wenn etwas nicht klappt

| Meldung oder Lage | Was tun |
|---|---|
| **„Keine GPU aktiv“** in Schritt 1 | GPU einschalten (Colab: Schritt 2, Kaggle: Schritt 2) und erneut „Alle ausführen“ bzw. „Run All“. |
| Colab: „Es ist derzeit keine GPU verfügbar“ oder Nutzungslimit erreicht | Die kostenlosen GPUs sind gerade belegt. Später noch einmal versuchen oder Kaggle nehmen. |
| Kaggle: GPU oder Internet lässt sich nicht einschalten | Telefon-Verifizierung fehlt (siehe Voraussetzung). |
| „Installation fehlgeschlagen bei: …“ | Noch einmal „Alle ausführen“. Bleibt der Fehler, den roten Text kopieren und schicken. |
| „PyTorch sieht die GPU nicht“ | Menü „Laufzeit → Laufzeit neu starten“ (Kaggle: „Restart Session“), dann erneut „Alle ausführen“. |
| Die Sitzung bricht während der Messung ab (Tab geschlossen, Zeitlimit) | Das Notebook neu starten. Bei Colab und Kaggle ist die virtuelle Maschine danach leer; die Messung beginnt von vorn. |
| Eine Konfiguration meldet „Fehler, Konfiguration übersprungen“ | Kein Problem: Das Notebook misst die anderen weiter, und der Fehler steht in der Datei. |

---

## Was das Notebook genau tut (für Interessierte)

Vorab festgelegt in [`reports/phase5_preregistration.md`](../reports/phase5_preregistration.md), Abschnitt 4.

1. **GPU prüfen** mit `nvidia-smi`. Ohne GPU hält es hier mit einer Anleitung an.
2. **Installation** in einer eigenen Python-3.12-Umgebung mit festen Versionen: LeRobot 0.6.1, PyTorch 2.11.0
   (CUDA 12.6), dieselben Bibliotheken wie Kortyns CPU-Messungen.
3. **Messung** mit dem Modell `lerobot/smolvla_libero` (fest gepinnte Version) und acht echten Bildern aus der
   Simulation LIBERO (zwei Kameras, Roboterzustand, Aufgabentext):
   - Genauigkeiten: wie geladen (bf16/fp32 gemischt), fp16, fp32, dazu die schnellste davon kompiliert
     (`torch.compile`).
   - Vorher eine Rechenkontrolle gegen fp32: Eine Genauigkeit zählt nur, wenn ihre Aktionen höchstens 25 % so
     stark abweichen wie zwei Läufe des Modells mit verschiedenem Zufall.
   - Je Batchgröße 10 Aufwärmläufe und 100 Messläufe; Median, p90, p95, p99, alle Einzelwerte und der
     GPU-Speicher.
4. **Datei schreiben** (nach jeder Genauigkeit neu, damit nichts verloren geht) und zum Download anbieten.

## Datenschutz und Kosten

- **Kostenlos:** Colab und Kaggle stellen die T4-GPU kostenlos mit einem Zeitkontingent bereit. Du musst nichts
  kaufen, kein Abo abschließen.
- **Keine Telemetrie:** Das Notebook schaltet die Nutzungsstatistik von Hugging Face ab und nutzt kein
  Experiment-Tracking. Heruntergeladen werden nur Python-Pakete (PyPI, PyTorch) und das Modell (Hugging Face).
- **Was in der Datei steht:** Messwerte, GPU-Name, Treiber- und Softwareversionen, Plattform (Colab/Kaggle) und
  Uhrzeit der Messung. Keine persönlichen Daten.
