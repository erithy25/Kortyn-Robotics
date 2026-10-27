# Kortyn Phase 5 – Präregistrierung: RTC mit kurzer Führung und GPU-Entscheidungsregel

_Committet am 27.10.2026 vor dem ersten neuen Lauf dieser Phase (LIBERO-Rollout, Flotten-Simulation oder
GPU-Messung). Danach bleibt diese Datei unverändert. Jede Abweichung steht mit Zeitpunkt und Begründung im
Report (`paper/report.md`, Abschnitt „Deviations from the pre-registration“)._

**Entscheidung nach Phase 4 (Vorgabe):** Kortyn wird als Startup-Idee in der jetzigen Form nicht
weiterverfolgt. Hauptweg: technischer Bericht und Open-Source-Werkzeug. Gemessen werden nur noch zwei Dinge:

1. die echte Rechenzeit von SmolVLA auf einer NVIDIA-GPU, einzeln und gebatcht (Abschnitte 4–5);
2. RTC mit kurzer Führung, diesmal mit bestandener 0-ms-Prüfung vor der Dosis-Wirkung (Abschnitt 3).

**Was vor diesem Commit schon gelaufen ist** (keine Erfolgsmessung, kein Rollout):

- Acht LIBERO-Beobachtungen direkt nach dem Reset (Aufgaben 8 und 0, Init-States 0–3) als realistische
  Eingaben des GPU-Notebooks. Kein Policy-Schritt, keine Episode.
- Eine numerische Kalibrierung auf der CPU: ein Batch dieser acht Beobachtungen durch SmolVLA in drei
  Präzisionen (Abschnitt 4.2). Nur Aktionswerte, kein Erfolg.
- Die Power-Simulation des Gates (Abschnitt 3.5, `results/phase5/power.json`, `kortyn/power_phase5.py`).
- Auswertungen vorhandener Daten: Paarung zwischen Bedingungen (φ), Durchsatz der Phase-4-Blöcke.

Über den Erfolg von RTC mit kurzer Führung ist nichts bekannt.

## 0. Fragen und Antwortlogik

| Frage | Messgröße | Entscheidung |
|---|---|---|
| G: Passt RTC mit kurzer Führung bei 0 ms zur synchronen Referenz? | gepaarter Vergleich mit `sync25` | Abschnitt 3.3; **Stopp-Regel** |
| H1: Streckt RTC mit kurzer Führung die Dosis-Wirkungs-Kurve? (nur wenn G besteht) | mittlerer Erfolgsgewinn im Budget gegen die Phase-4-Kurve ohne RTC | Abschnitt 3.7 |
| GPU: Bedient die gemessene GPU ≥ 10 Roboter mit p99 der Ende-zu-Ende-Latenz ≤ 150 ms? | Flotten-Simulation mit der Messdatei | Abschnitt 5; **Geschäftsentscheidung** |

## 1. Arbeitspunkt (unverändert aus Phase 4)

- **Policy:** `lerobot/smolvla_libero`, Revision `a24f1f380461b665158d52acd140152c79e8b36d`, wie von LeRobot
  geladen (VLM und Aktionsexperte bf16, Projektionen fp32), 10 Flow-Schritte, Chunk 50.
- **Aufgaben:** LIBERO-Long, Aufgaben 8 und 0 × Init-States 0–49. Seeds 1000–1049 = Aufgabe 8, 1050–1099 =
  Aufgabe 0. Seeds 1100–1199 spielen dieselben Init-States und Simulator-Seeds noch einmal, mit anderem
  Policy-Rauschen (das Rauschen hängt am Episoden-Seed, `request_seed`).
- **Agent:** 20 Hz, eine Anfrage, sobald ≤ 24 Aktionen übrig sind (alle 25 Takte), Budget 1 250 ms,
  Kaltstart, ohne Plan steht der Arm still. Bilder verlustfrei (zstd Stufe 9, bitgleiche Modelleingabe).
- **Rechnen:** CPU, Simulationszeit, 4 Worker, ein Torch-Thread je Worker, `--share-model`.
- **Fester Code-Stand:** Alle neuen Rollouts laufen in einem eigenen Git-Worktree auf dem Commit dieser
  Präregistrierung (er enthält die Laufkonfigurationen). Die Weiterentwicklung (Notebook, Werkzeug, Bericht)
  berührt die laufenden Experimente nicht.

## 2. Befunde aus vorhandenen Daten, die das Design bestimmen

**2.1 Warum Phase 4 die 0-ms-Prüfung verfehlte.** RTC bei 0 ms erreichte 42 % gegen 53 % für `sync25`
(−11 pp [−23; +1], McNemar p = 0,13; Kriterium 2 verfehlt). Zwei Effekte addierten sich:

- Asynchron ohne RTC liegt bei 0 ms schon 6–7 pp unter `sync25` (Phase 3: 46 %, Phase 4: 47 %). Der neue
  Plan greift einen Takt später: Im Anfragetakt läuft noch die 26. Aktion des alten Plans, dann die Aktionen
  1–25 des neuen. `sync25` führt die Aktionen 0–24 des neuen Plans aus. Beide Vergleiche nutzen dieselbe
  einzelne `sync25`-Messung aus Phase 3; ob der Abstand echt ist, ist offen.
- RTC kostete bei 0–200 ms weitere 5–10 pp, fast nur in Aufgabe 0. Vermutete Ursache: Die präregistrierte
  Führung umfasste bis zu 25 Aktionen. Bei d = 1 Takt zog sie 24 Aktionen des neuen Plans zum alten Plan,
  der 1,25 s früher beobachtet wurde.

**2.2 Paarung ist schwach.** Gleiche Seeds unter verschiedenen Bedingungen hängen nur mäßig zusammen:
φ = 0,17 (RTC gegen ohne RTC bei 0 ms) bis 0,58 (dieselbe Bedingung in Phase 3 und 4); 21–41 % der Paare
sind diskordant. Mit 100 Paaren ist das 95-%-KI einer Differenz ±11–13 pp breit. Das Phase-4-Kriterium
„untere Grenze ≥ −15 pp“ bestand deshalb selbst bei echter Gleichheit nur mit 68–88 % Wahrscheinlichkeit
(Abschnitt 3.5).

**2.3 Laufreihenfolge.** LIBERO trägt Zustand zwischen Episoden eines Prozesses. Die Reihenfolge ändert
einzelne Episoden, aber nicht die Quote (Phase 4, Frischprozess-Prüfung). Bedingungen, die verglichen
werden, laufen deshalb im selben Block, je Seed direkt nacheinander.

## 3. RTC mit kurzer Führung

### 3.1 Einstellung

- **Neu: Führungslänge H = D + 4** (`max_overlap_steps: 4` statt 24). D ist die erwartete Verzögerung, mit
  der der Agent plant (Phase-4-Schätzer: Median + 1,645 · Streuung der Latenzen der Episode). Die ersten D
  Aktionen der Führung laufen sicher noch (Gewicht 1, `inference_delay`), danach führen höchstens 4 Aktionen
  weich (EXP-Gewichte). Bei fester Latenz L ist D = d(L) bei 0 ms und d(L) + 1 bei 200, 500 und 1 000 ms
  (Sicherheitsaufschlag 1,645 · 5 ms); in den ersten beiden geführten Anfragen einer Episode ist D wegen der
  Startstreuung von 50 ms um höchstens einen Takt größer. Das gilt genauso für die Phase-4-Läufe.
- **Begründung:** LeRobot empfiehlt einen Ausführungshorizont von 8–12 Aktionen. Bei d = 1 ergibt D + 4 = 5,
  bei 500 ms 15. Die Führung bleibt damit dort, wo sie Phase 4 half (große d: die sicheren Aktionen), und
  zieht bei kleinem d nur noch 4 statt 24 Aktionen zum alten Plan.
- **Unverändert:** LeRobots RTC ohne Änderung (`RTCConfig(enabled, EXP, max_guidance_weight=10)`), dieselbe
  Eigenheit von LeRobot 0.6.1 (Korrektur ohne Vektor-Jacobi-Produkt), Kaltstart ohne Führung, Startrauschen
  wie ohne RTC. Es wird nur ein Parameter geändert, damit ein Unterschied zu Phase 4 der Führungslänge
  zuzuschreiben ist.
- **Strategie** `rtc_kurz` in `kortyn/experiments/phase5/libero_rtc_kurz.yaml`.

### 3.2 Gate-Block G

Drei Bedingungen bei 0 ms, je **200 Episoden** (Seeds 1000–1199, also jeder Init-State zweimal mit
verschiedenem Rauschen):

| Lauf | Bedingung | Rolle |
|---|---|---|
| `sync25` | synchron, 25 Aktionen je Plan | Referenz des Gates, neu gemessen |
| `async_0ms` | asynchron ohne RTC (Strategie „halten“) | Kontrolle (Abschnitt 3.4) |
| `rtc_kurz_0ms` | asynchron mit RTC, H = D + 4 | Prüfling |

- **Gleicher Block, gleicher Prozess:** Alle drei laufen im RTC-Profil, damit ein Worker sie je Seed direkt
  nacheinander spielt (`run_virtual` arbeitet ein Profil Seed für Seed ab). Ohne Führung ist die Abtastung
  im RTC-Profil bitgleich zum Profil ohne RTC (Phase-4-Prüfung 2).
- **Warum `sync25` neu:** Die Phase-3-Messung lief in einem anderen Laufkontext; Phase 4 fand dort eine
  Stufe, die sich in frischen Prozessen nicht reproduzierte (Aufgabe 8, 500 ms: 26 % gegen 46 %). Die
  Phase-3-Messung wird nur berichtet.
- **Warum 200 Episoden:** Abschnitt 3.5. LIBERO hat 50 Init-States je Aufgabe; mehr unabhängige
  Init-States gibt es nicht. Die zweite Episode je Init-State unterscheidet sich nur im Rauschen; die
  Auswertung behandelt beide als Cluster.

### 3.3 Gate-Kriterium (Stopp-Regel)

Vergleich `rtc_kurz_0ms` gegen `sync25` aus Block G, gepaart je Seed (200 Paare in 100 Clustern
Aufgabe × Init-State). Das Gate ist **bestanden**, wenn

1. McNemar exakt über alle 200 Paare p ≥ 0,05 **und**
2. die untere Grenze des 95-%-KI der Differenz (RTC − sync25) ≥ −15 pp.

Das KI kommt aus einem Cluster-Bootstrap: Init-States mit beiden Episoden werden je Aufgabe mit Zurücklegen
gezogen (10 000 Ziehungen, Seed 5000). Das ist das Kriterium R-RTC aus Phase 4; nur das KI berücksichtigt die
Cluster. Ist RTC signifikant **besser** als `sync25` (p < 0,05, Differenz > 0), gilt das Gate ebenfalls als
bestanden und wird so ausgewiesen: Das Gate soll Schaden bei Latenz null ausschließen. Code:
`kortyn/rtc_gate.py`.

**Folge:**

- **Bestanden:** Block D läuft (Abschnitt 3.6).
- **Nicht bestanden:** Keine weitere RTC-Messung. Der Bericht hält fest, dass die Frage, ob RTC SmolVLA
  verzögerungstoleranter macht, offen bleibt, mit Zahlen und Kontrollen aus Block G. Keine zweite
  Einstellung, kein Nachmessen, keine andere Referenz.

### 3.4 Kontrollen (ohne Einfluss auf das Gate)

Mit demselben Verfahren wie 3.3 berichtet:

- **K1 `async_0ms` gegen `sync25`:** Misst, ob schon die asynchrone Ausführung ohne RTC zur synchronen
  Referenz passt. Fällt K1 durch, misst das Gate auch die Architektur, nicht nur RTC.
- **K2 `rtc_kurz_0ms` gegen `async_0ms`:** Der reine RTC-Effekt bei 0 ms.
- **K3 Reproduktion:** `async_0ms` (Seeds 1000–1099) gegen Phase-4-Block-A `fest_0ms` (47 %) und
  `sync25` (Seeds 1000–1099) gegen Phase 3 (53 %); Quote und gepaarter Vergleich.
- **Je Aufgabe:** alle Vergleiche zusätzlich getrennt für Aufgabe 8 und 0 (Phase 4: RTC-Verlust in Aufgabe 0).

### 3.5 Teststärke des Gates

Simulation mit genau dem Code des Gates (`kortyn/power_phase5.py`, 2 000 Datensätze je Zelle, 2 000
Bootstrap-Ziehungen). Erfolg der Referenz 53 %. Gaußsche Kopula: ein latenter Effekt je Init-State (beide
Episoden, beide Bedingungen) und einer je Episode (beide Bedingungen). ρ ist die latente Korrelation beider
Bedingungen auf einem Seed, je zur Hälfte auf dem Init-State; 0,3–0,7 deckt die gemessenen φ von 0,17–0,58 ab.

| Wahre Quote RTC kurz | 100 Paare (Phase-4-Design), ρ = 0,3 / 0,5 / 0,7 | **200 Paare (dieses Design)**, ρ = 0,3 / 0,5 / 0,7 |
|---|---|---|
| 53 % (wie `sync25`) | 68 / 76 / 88 % | **93 / 97 / 98 %** |
| 47 % (asynchron ohne RTC, −6 pp) | 31 / 38 / 48 % | 54 / 63 / 64 % |
| 43 % (−10 pp) | 16 / 18 / 22 % | 22 / 25 / 23 % |
| 38 % (−15 pp, Fehl-Bestehen) | 2,5 / 3,4 / 3,0 % | 3,0 / 3,4 / 1,7 % |

- Mit 200 Paaren besteht ein gleichwertiges RTC in 93–98 % der Fälle; die Rate des Fehl-Bestehens bei
  −15 pp bleibt bei ≤ 3,4 %.
- **Offenes Risiko:** Liegt die asynchrone Ausführung selbst 6 pp unter `sync25` (Abschnitt 2.1), besteht
  selbst ein RTC ohne jeden eigenen Schaden nur mit 54–64 %. Die Stopp-Regel gilt trotzdem; K1 macht diesen
  Fall im Bericht sichtbar.

### 3.6 Dosis-Block D (nur wenn das Gate besteht)

`rtc_kurz` bei **200, 500 und 1 000 ms**, je 100 Episoden (Seeds 1000–1099), Konfiguration
`kortyn/experiments/phase5/libero_rtc_kurz_dose.yaml`. Die 0-ms-Stufe der Kurve sind die Seeds 1000–1099 von
`rtc_kurz_0ms` aus Block G.

### 3.7 Auswertung von Block D

Kurvenmodell, Bootstrap und Schwellen wie Phase 4 (`kortyn/dose_curve.py`; log-logistisch je Aufgabe über
die wirksame Latenz, Anpassungstest mit Ausweichkurve, Schwanzregel ab 1 250 ms, gepaarter Bootstrap über
Seeds je Aufgabe, 2 000 Ziehungen, Seed 4242):

- **Vergleichskurve ohne RTC:** die Phase-4-Kurve unverändert (Block A: 0, 100, 200, 300, 500, 750, 1 000,
  1 250 ms; 1 500 und 2 000 ms aus Phase 3), dieselben Seeds 1000–1099.
- **H1-kurz (primär):** mittlerer Erfolgsgewinn der RTC-kurz-Kurve gegenüber der Kurve ohne RTC über die
  25 Takte im Budget (wirksame Latenz 50–1 250 ms), gepaartes 95-%-KI. **Bestätigt**, wenn die untere Grenze
  > 0; **„RTC schadet“**, wenn die obere Grenze < 0; sonst **nicht nachweisbar**. Gleiche Messgröße wie
  Phase-4-H1 (−1,6 pp [−6,1; +3,4]).
- **Je Stufe (sekundär):** gepaarte Differenz RTC kurz − ohne RTC bei 0, 200, 500 und 1 000 ms (Block A
  bzw. Block G für 0 ms), McNemar und Bootstrap-KI, je Aufgabe.
- **Beschreibend:** L* (≤ 3 pp Verlust gegenüber 0 ms derselben Bedingung) mit konservativer und oberer
  Grenze, ΔL* gegen ohne RTC, Vergleich mit RTC lang aus Phase 4. Vier Stufen lösen L* gröber auf als die
  sechs Stufen von Phase 4.
- **Kapazität** (Phase-4-Kriterium: ≤ 1 % verpasste Anfragen und schlechtester Roboter ≤ 3 pp unter dem
  Einzelroboter) mit der RTC-kurz-Kurve auf der vorhandenen Stufe-1-Simulation mit RTC-Mehraufwand
  (`results/phase4/fleet/scan_rtc.jsonl`, RTX 3060 Ti 222,5 ms). Die kürzere Führung ändert die Rechenzeit
  nicht nennenswert (die Korrektur ist ein gewichteter Fehler über H Aktionen).

**Keine Teststärke-Simulation für H1-kurz.** Phase 4 hatte für H1 75–100 % bei einer Streckung der
Latenzachse um 1,5 (sechs RTC-Stufen). Mit vier Stufen ist die Kurve weniger festgelegt; die Teststärke ist
niedriger und wird nicht beziffert.

## 4. GPU-Messung im Browser (Notebook)

### 4.1 Messprotokoll (`notebooks/gpu_benchmark.ipynb`)

Der Nutzer führt das Notebook unverändert in Google Colab oder Kaggle aus und lädt die JSON-Datei ins Repo.

- **Software:** eigene Python-3.12-Umgebung mit `uv`, gepinnt `lerobot[smolvla]==0.6.1` und PyTorch 2.11.0
  (wie die CPU-Läufe) aus dem CUDA-12.6-Index von PyTorch: Er braucht nur einen Treiber ≥ 525 und enthält
  Kernel für ältere GPUs. Empfohlen ist eine T4 (Colab, Kaggle „GPU T4 x2“). Keine Telemetrie
  (`HF_HUB_DISABLE_TELEMETRY=1`, kein W&B).
- **Eingaben:** acht echte LIBERO-Long-Beobachtungen (Aufgaben 8 und 0, Init-States 0–3: zwei Kameras
  360 × 360, 8-D-Zustand, die Sprachanweisung der Aufgabe), im Notebook eingebettet als JPEG q90. Batch b
  nimmt die Beobachtungen reihum. Startrauschen aus einem festen CPU-Generator.
- **Gemessen** wie bisher (`kortyn/bench_batch.py`): Vorverarbeitung (Normalisierung, Tokenisierung) +
  `predict_action_chunk` + Nachverarbeitung + Kopie auf die CPU, synchronisiert. Das ist die Servicezeit des
  Brains für einen Batch.
- **Batchgrößen** 1, 2, 4, 8, 16, 32. Je Größe 10 Aufwärm-Läufe, dann **100 gemessene Läufe**. Berichtet:
  Median, p90, p95, p99, Mittel, Streuung, Minimum, Maximum, alle 100 Einzelwerte, Spitzen-GPU-Speicher
  (belegt und reserviert). Kein Speicher mehr: diese und größere Größen entfallen (`oom`). Würden die 100
  Läufe nach den Aufwärm-Läufen länger als 5 Minuten dauern: diese und größere Größen entfallen (`skipped`).
- **Konfigurationen**, in dieser Reihenfolge:
  1. `standard`: wie von LeRobot geladen (bf16/fp32 gemischt), eager. Der Arbeitspunkt aller Kurven.
  2. `fp16`: alle bf16-Tensoren als fp16, fp32-Teile bleiben, eager.
  3. `fp32`: alles fp32, eager, TF32 aus.
  4. `compiled`: die schnellste numerisch gültige eager-Konfiguration (Median bei Batch 1) mit LeRobots
     eigenem Schalter `compile_model` (`torch.compile`, Modus `default`). Kann die Plattform nicht
     kompilieren (z. B. P100 ohne Triton-Unterstützung), steht der Fehler in der Datei.

  Fehlt einer Konfiguration die Hardware-Unterstützung (z. B. bf16 auf einer T4), wird gemessen, was
  PyTorch tut, oder der Fehler gespeichert. Eine fehlerhafte Konfiguration bricht das Notebook nicht ab.

### 4.2 Numerische Gültigkeit

Vor der Zeitmessung rechnet jede Konfiguration einen Batch der acht Beobachtungen mit festem Rauschen.

- **Referenz:** fp32 auf derselben GPU. **Streuung s:** RMS-Abstand zweier fp32-Läufe mit verschiedenem
  Rauschen, das eigene Zufallsrauschen der Policy.
- **Gültig**, wenn alle Aktionen endlich sind und der RMS-Abstand zur fp32-Referenz bei gleichem Rauschen
  **≤ 25 % von s** ist.
- **Kalibrierung auf der CPU** (vor diesem Commit, dieselben Eingaben): s = 0,0166 (Aktionen im Mittel
  0,42); `standard` weicht um 6,9 % von s ab, `fp16` um 0,8 %. Die Schwelle lässt `standard` (Arbeitspunkt
  aller Kurven) mit dreifachem Abstand zu und schließt numerisch defekte Läufe (Überlauf, NaN) aus.

### 4.3 Dateiformat

`kortyn-gpu-benchmark-v2` (JSON): Gerät, Software-Versionen, Eingaben, Einstellungen, je Konfiguration
Präzision, Kompilierung, Status, numerische Prüfung und die Zeilen je Batchgröße mit allen Einzelwerten.
Kortyn liest die Datei direkt (`kortyn/gpu_benchmark.py`, Flottenmodell und Kapazitätswerkzeug); das alte
Format `kortyn-batch-service-v1` bleibt lesbar.

## 5. GPU-Entscheidungsregel (Geschäftsfrage)

**Frage (Vorgabe):** Kann die gemessene GPU ≥ 10 Roboter mit einer Ende-zu-Ende-Latenz (Büro-WLAN +
Warteschlange + Servicezeit) von p99 ≤ 150 ms bedienen? **Ja:** Die Geschäftsidee wird ein zweites Mal
geprüft. **Nein:** endgültig verworfen. Das Ergebnis geht unverändert in den Bericht.

**Rechnung**, sobald eine Messdatei in `results/phase5/gpu/` liegt (Simulationszeit, kein Rollout):

- **Servicezeit:** je Batchgröße die empirische Verteilung der 100 Messwerte. Zwischen gemessenen Größen
  wird quantilweise linear interpoliert (gleiches Quantil u bei beiden Nachbarn, linear in b). Größere
  Batches als die größte gültige gemessene Größe gibt es nicht.
- **Flotte:** Stufe-1-Simulation wie Phase 4 (`kortyn/fleet.py`): Agent wie Abschnitt 1 (Client-Overhead
  1 ms, Transport-Overhead 5 ms), Episodenlängen aus Phase 3 `fest_0ms`, Startstreuung 10 s, Horizont 630 s,
  Messfenster ab 30 s, 5 Wiederholungen, Seed 3000 (gemeinsame Zufallszahlen).
- **Netz:** Büro-WLAN wie in allen Phasen: RTT 20 ± 10 ms gleichverteilt, keine Stalls, keine
  Bandbreitengrenze; Rückweg-Marge des Brains 15 ms.
- **Strategien:** a (FIFO), b (FIFO + Batching, ≤ 16, ≤ 25 ms Warten), d (EDF + Batching, ≤ 16), e (d +
  Zugangskontrolle). Bei e müssen alle Roboter zugelassen sein.
- **Ende-zu-Ende-Latenz** einer Anfrage: Ankunft des Plans am Roboter minus Beobachtungszeitpunkt.
- **p99:** über alle Anfragen aller Roboter im Messfenster, gepoolt über die 5 Wiederholungen. Gerechnet aus
  den 10-ms-Histogrammen: p99 ≤ 150 ms genau dann, wenn ≥ 99 % der Anfragen unter 150 ms liegen.
- **Latenz-Kapazität K₁₅₀** je Konfiguration und Strategie: größte Flotte N, für die N und jede kleinere
  Flotte p99 ≤ 150 ms erfüllen (Scan N = 1, 2, … bis zum ersten Verstoß, höchstens 32).
- **Urteil:** **Ja**, wenn für mindestens eine numerisch gültige Konfiguration (4.2) und eine Strategie
  K₁₅₀ ≥ 10 gilt. Sonst **Nein**.
- **Mehrere Messdateien** (z. B. Colab und Kaggle): Urteil je Datei; die Geschäftsfrage ist Ja, sobald eine
  gültige Messung Ja ergibt. Der Bericht nennt alle Messungen und die GPU.
- **Ungültige Messung** (kein Gerät erkannt, keine gültige Konfiguration, < 100 Messungen bei Batch 1):
  kein Urteil, bis eine gültige Datei vorliegt.

**Zusätzlich berichtet, ohne Einfluss auf das Urteil:**

1. Vorhergesagter Erfolg (Phase-4-Kurve ohne RTC) bei N = 10 und bei K₁₅₀: Mittel und schlechtester Roboter.
2. Kapazität nach dem Phase-4-Kriterium (≤ 1 % verpasst, schlechtester ≤ 3 pp unter Einzelroboter) mit der
   gemessenen Servicezeit.
3. K₁₅₀ mit geteiltem WLAN-Medium 100 und 200 Mbit/s und zstd-Bildern (292 kB je Anfrage, Phase-4-Modell):
   Die Übertragungszeit zählt bei einem 150-ms-Ziel spürbar mit.
4. Gemessene Servicezeit bei Batch 1 gegen die Annahmen der Phasen 3–4 (RTX 3060 Ti 220 ms).

## 6. Laufzeitschätzung

_Stand 27.10.2026, 21:00 UTC, vor dem ersten Lauf._

**Gemessener Durchsatz** dieser Maschine (4 vCPU, 13,4 GB Speichergrenze): Phase-4-Block A 800 Episoden
in ≈ 4,2 h, Block B 600 Episoden in 3,0 h, also ≈ 190–200 Episoden je Stunde. Episoden mit hoher Latenz
laufen öfter bis zum Ende (520 Schritte) und dauern länger.

| Block | Umfang | Wandzeit |
|---|---|---|
| G: Gate, 3 Bedingungen × 200 Episoden bei 0 ms | 600 Episoden | ≈ 3,0–3,2 h |
| Gate-Auswertung | – | ≈ 0,1 h |
| D: Dosis 200 / 500 / 1 000 ms (nur wenn G besteht) | 300 Episoden | ≈ 1,6–1,8 h |
| Kurven, H1, Kapazität | – | ≈ 0,2 h |
| **Summe RTC-Messung** | | **≈ 4,9–5,3 h**, mit paralleler Entwicklungsarbeit bis ≈ 5,8 h |

Notebook, Werkzeug, Bericht und README entstehen parallel. Die Schätzung liegt deutlich unter 10 Stunden;
es ist keine Kürzung vorgesehen. Bricht der Container ab, wird fortgesetzt (Läufe überspringen vorhandene
Episoden).

**GPU-Notebook** (beim Nutzer, nicht auf dieser Maschine): Installation ≈ 5 min, Laden ≈ 2 min, Messung
≈ 30–60 min je nach GPU.

## 7. Übersicht der Entscheidungen

| Nr. | Frage | Kriterium | Abschnitt |
|---|---|---|---|
| G | RTC kurz bei 0 ms wie `sync25` | McNemar p ≥ 0,05 und untere KI-Grenze ≥ −15 pp (oder signifikant besser) | 3.3 |
| H1 | RTC kurz streckt die Kurve | untere KI-Grenze des mittleren Gewinns im Budget > 0 | 3.7 |
| N | Konfiguration numerisch gültig | endlich und RMS-Abstand zu fp32 ≤ 25 % der Rauschstreuung | 4.2 |
| GPU | Geschäftsidee zweite Prüfung | K₁₅₀ ≥ 10 für eine gültige Konfiguration und Strategie | 5 |

## 8. Abnahme (Vorgabe)

1. pytest, ruff und mypy grün.
2. Notebook und Anleitung fertig; das Notebook ist auf der CPU bis zur GPU-Prüfung getestet.
3. Die RTC-Frage ist mit bestandener 0-ms-Prüfung neu gemessen oder ehrlich als offen dokumentiert.
4. `paper/report.md`, Lizenz, Kapazitätswerkzeug und README sind fertig.
5. Diese Präregistrierung ist vor dem ersten neuen Lauf committet.
