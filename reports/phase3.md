# Kortyn Phase 3 – Flotten-Brain: Wie viele Roboter bedient ein Brain?

**Zentrale Frage:** Wie viele Roboter kann ein Brain bedienen, ohne dass einer sein Latenzbudget reißt, und
welche Scheduling-Strategie maximiert diese Zahl?

## 0. Vorab festgelegt (Präregistrierung)

_Geschrieben am 25.10.2026 um 10:25 UTC, vor jedem Messlauf von Phase 3, und so committet. Dieser Abschnitt
wird nach den Läufen nicht mehr geändert. Jede Abweichung steht im Abschnitt „Abweichungen von der
Präregistrierung“ am Ende des Reports, mit Zeitpunkt und Grund._

### 0.1 Arbeitspunkt und Standards

| Größe | Festlegung | Begründung |
|---|---|---|
| Policy | `lerobot/smolvla_libero`, Revision `a24f1f38`, bf16 wie von LeRobot geladen, 10 Flow-Schritte, Chunk 50 | Vorgabe; einziger LeRobot-Checkpoint für LIBERO |
| Aufgabenbasis | LIBERO-Long (`libero_10`), Umgebung exakt wie `lerobot-eval`: 360×360-Kameras, feste Init-States, 10 Einschwing-Schritte, relative Endeffektor-Steuerung, 20 Hz, max. 520 Schritte | LeRobots Evaluationsprotokoll |
| Anfrage-Takt | Neue Anfrage, sobald ≤ 24 Aktionen übrig sind: alle 25 Takte (1,25 s), solange Pläne rechtzeitig kommen; höchstens eine Anfrage unterwegs | Standard aus Phase 3a (LeRobot Async: Schwelle 50 %) |
| **Latenzbudget B** | (24 + 1) × 50 ms = **1 250 ms** | Zeit, bis die Queue leerläuft |
| Starvation | Arm steht still (Endeffektor-Verschiebung 0, Greifer hält seinen Befehl), Zeitschritt-Ausrichtung | wie Phase 3a |
| **Episodenstart** | **Kaltstart**: Der erste Plan kommt mit derselben Latenz wie jeder andere; bis dahin steht der Arm | neuer Standard (Vorgabe) |
| Bildübertragung | roh, bis Schritt 1.5 entscheidet | Vorgabe |
| Stichprobe | 100 Episoden je Bedingung: 2 Aufgaben × Init-States 0–49, Seeds 1000–1099, gepaart über alle Bedingungen | Standard ab Phase 3a |
| Timeout einer Anfrage | 10 s (nur bei Überlast relevant) | – |

Der Takt von 25 Aktionen je Plan ist ein Arbeitspunkt, kein Optimum. LeRobot empfiehlt für diesen Checkpoint
10 Aktionen je Plan (höhere Erfolgsquote, aber 2,5-mal so viele Anfragen). Mit 2,8 s CPU-Inferenz je Plan
sprengt dieser Takt das Laufzeitbudget. Der Pilot misst deshalb, wie viel Erfolg der Arbeitspunkt kostet,
und die Warteschlangen-Simulation rechnet die Kapazität zusätzlich für 10 Aktionen je Plan.

### 0.2 Schritt 1: Aufgaben, Referenz, Kaltstart, JPEG

**1.1 Pilot und Reproduktion.**
- **Pilot:** alle zehn LIBERO-Long-Aufgaben × Init-States 0–9 = 100 Episoden. Synchron mit 50 (`sync50`,
  LeRobot-Protokoll wie ausgeliefert), 25 (`sync25`, Referenz-Takt der Latenz-Experimente) und 10 Aktionen
  je Plan (`sync10`, LeRobots Empfehlung).
- **LeRobots eigene Evaluation:** `lerobot-eval` auf denselben 100 Episoden (je Aufgabe 10 Episoden,
  Batch 1, Seed 1000, `n_action_steps` aus dem Checkpoint = 50, 360×360, ein CPU-Thread).
- **Reproduktionskriterium R1** (Gate für Abnahmekriterium 2): `sync50` gegen `lerobot-eval`, gepaart über
  die 100 Episoden. Reproduziert, wenn
  1. McNemar exakt p ≥ 0,05 **und**
  2. das 95-%-KI der gepaarten Differenz (Bootstrap, 10 000 Ziehungen) vollständig in [−15, +15] pp liegt.

  Gilt nur 1., lautet das Urteil „nicht unterscheidbar, Präzision unzureichend“. Gilt 1. nicht, stoppe ich
  die Hauptexperimente und kläre die Abweichung, bevor es weitergeht.
- **Publizierte Werte** (beschreibend, ohne Gate; für diese Checkpoint-Revision gibt es keinen
  publizierten LIBERO-Long-Wert):
  - SmolVLA-Paper: LIBERO-Long 71 % (eigenes Training).
  - Paper-Ablation: Mittel über die vier Suiten 51,8 % bei 50 und 82,8 % bei 10 Aktionen je Plan.
  - Community-Reproduktionen von `HuggingFaceVLA/smolvla_libero` auf LIBERO-Long mit `lerobot-eval`:
    43–56 %; Leaderboard: 60 %.

  Abweichungen werden erklärt.

**1.2 Aufgabenwahl** (Regel, angewandt auf `sync25` im Pilot, je 10 Episoden):
1. Zulässig sind Aufgaben mit 3–9 Erfolgen von 10.
2. Unter den zulässigen nehme ich die zwei mit der längsten medianen Episodenlänge der erfolgreichen
   Episoden (langer Horizont ist der Zweck). Bei Gleichstand gewinnt die höhere Erfolgsquote.
3. Gibt es weniger als zwei zulässige Aufgaben, erweitere ich das Band auf 2–10, dann auf 1–10 Erfolge.

**1.3 Kalt- gegen Warmstart:** bei fester Latenz 1 000 ms (0,8 B), 100 gepaarte Episoden. Berichtet werden
Differenz mit KI und McNemar; das Ergebnis entscheidet nichts.

**1.4 JPEG q90:** `sync25` roh gegen `sync25` mit JPEG q90 (beide Kameras), 100 gepaarte Episoden. JPEG q90
wird Standard für Schritt 2–4 genau dann, wenn
1. McNemar exakt p ≥ 0,05 **und**
2. die untere Grenze des 95-%-KI der Differenz (q90 − roh) ≥ −10 pp.

Sonst bleibt roh Standard.

### 0.3 Schritt 2: Dosis-Wirkungs-Kurven

- **Feste Ende-zu-Ende-Latenzen**, Kaltstart, ein Roboter, kein Netz: L ∈ {0, 250, 500, 1 000, 1 250,
  1 500, 2 000} ms, also bis 1,6 B.
- **Jitter-Variante:** L je Anfrage unabhängig gleichverteilt in [500, 1 500] ms (Mittel 1 000 ms).
- **Umfang:** je 100 Episoden auf denselben Seeds; Bildübertragung nach Schritt 1.4.
- **Ergebnis:** Erfolgsquote je Stufe und Aufgabe mit Wilson-95-%-KI, gepoolt und je Aufgabe. Dazu
  McNemar gegen 0 ms mit Holm-Korrektur über die sieben übrigen Bedingungen sowie Planalter, Starvation,
  Episodenlänge und Zeit bis zum Erfolg.
- LIBERO vergibt Reward nur bei Erfolg (spärlich). „Reward“ ist hier daher gleich Erfolg; als zusätzliches
  Maß berichte ich die Zeit bis zum Erfolg.

### 0.4 Schritt 3: Brain-Modell mit Batching

- **Servicezeit** eines Batches der Größe b auf einer GPU: s(b) = s₁ · g(b) · ε, mit ε lognormal
  (Mittel 1, Variationskoeffizient 5 %) je Batch.
  - s₁ ist die Einzel-Inferenzzeit der GPU für die LIBERO-Eingabe (zwei Kameras).
  - g(1) = 1; g kommt aus einem benannten Profil („kein Batching-Gewinn“ g(b) = b, „pessimistisch“,
    „realistisch“, „optimistisch“).
- **Festlegung der Profile:** Die Zahlen der Profile und s₁ je GPU lege ich nach der Recherche und vor der
  ersten Warteschlangen-Simulation fest, mit eigenem Commit und Zeitstempel. Das sind Modell-Eingaben,
  keine Erfolgskriterien.
- **Messtabellen** (JSON aus dem Benchmark-Skript) ersetzen ein Profil: lineare Interpolation zwischen
  gemessenen Batchgrößen, keine Extrapolation über die größte gemessene hinaus. Batchgrößen jenseits davon
  sind für die Scheduler gesperrt.

### 0.5 Schritt 4: Strategien und Flotten-Simulation

**Strategien** (Parameter fest):

| | Strategie | Parameter |
|---|---|---|
| a | FIFO ohne Batching (Baseline) | – |
| b | FIFO mit Batching | max. Batch 16, max. Wartezeit 25 ms ab der ältesten wartenden Anfrage |
| c | Deadline (EDF) ohne Batching | Priorität = Zeitpunkt, an dem die Queue des Roboters leerläuft (aus Füllstand bei der Anfrage) |
| d | EDF mit deadline-bewusstem Batching | max. Batch 16; größter Batch in EDF-Reihenfolge, der keine erreichbare Deadline reißt und die nächste wartende Anfrage nicht verdrängt |
| e | d + Zugangskontrolle | Neuer Roboter nur, wenn die Worst-Case-Vorhersage alle Budgets einhält (unten) |

**Einzelheiten der Strategien:**
- **Deadline-Schätzung des Brains:**
  - Deadline der Anfrage minus Rückweg-Marge (Einweg-Verzögerung des Netzes: Mittel + halbe Jitter-Breite;
    Büro-WLAN 15 ms, schlecht 60 ms).
  - Servicezeit-Schätzung für Entscheidungen: p90 des Profils.
- **Zugangskontrolle (e):** Worst Case = alle aufgenommenen Roboter fragen gleichzeitig an. Das Brain
  bedient sie in EDF-Batches zu höchstens 16. Ein neuer Roboter wird aufgenommen, wenn
  1. der letzte Batch innerhalb von B − Hinweg-Marge − Rückweg-Marge fertig ist (p90-Servicezeiten) **und**
  2. die Auslastung bei vollen Batches ≤ 0,9 beträgt.
- **Episodenstart-Anfragen** (leere Queue): EDF-Priorität = Anfragezeit, also sofort fällig.

**Flotten:**
- Größen 1, 2, 4, 8, 16 und 32; Startzeitpunkte gleichverteilt in den ersten 10 s.
- Je Konfiguration 10 Wiederholungen mit eigenen Startzeiten und Netz-Seeds.
- Messfenster 30 s bis 630 s nach dem ersten Start.
- Netz: Büro-WLAN (Standard), schlecht (Sensitivität).
- GPU-Profile: RTX 3060 Ti (Schätzung, bis eine Messung vorliegt), RTX 3090, RTX 4090.
- Batching-Profile: alle vier.

**Methode (zweistufig):**
1. **Warteschlange allein:**
   - Ereignisgesteuerte Zeit-Roboter mit genau den Anfrage-, Merge- und Starvation-Regeln des Agenten.
     Gleichheit mit dem echten Agenten prüft ein Test: gleiche Latenz-Ziehungen ergeben gleiche Anfrage- und
     Lieferzeiten.
   - Episodenlängen werden aus der 0-ms-Stufe von Schritt 2 gezogen; Reset-Zeit 0.
   - Dieselben Scheduler-Klassen wie in den echten Rollouts.
2. **Vorhersage je Roboter:** Mittel über die Latenzen seiner Anfragen (im Messfenster) von f(L), mit
   linearer Interpolation der Dosis-Wirkungs-Kurve zwischen den Stufen.
   - Jenseits von 2 000 ms bleibt f auf dem Wert der letzten Stufe (markiert als Extrapolation).
   - Die zwei Aufgaben gehen mit gleichem Gewicht ein.
   - Die Unsicherheit kommt aus einem gepaarten Bootstrap über die 100 Seeds der Kurve.
3. **Validierung:**
   - **Konfiguration:** RTX 3060 Ti (Schätzung), Batching „realistisch“, Büro-WLAN.
   - **Strategien:** die beste und die schlechteste nach Kapazität in dieser Konfiguration. Gleichstand
     entscheidet die niedrigere bzw. höhere Budget-Verletzungsrate bei der größten gemeinsamen Flotte.
   - **Flottengrößen:** n₁ = größte Rastergröße, bei der die schlechteste Strategie Kriterium (1) erfüllt;
     n₂ = nächste Rastergröße. Beide höchstens 8 (Arbeitsspeicher: eine Simulation je Flotte); erfüllt die
     schlechteste Strategie (1) noch bei 8, gilt n₁ = 4, n₂ = 8.
   - **Umfang:** je Fall etwa 100 Episoden (n Roboter × ⌈100/n⌉ Episoden) auf den Seeds von Schritt 2; die
     Seeds werden reihum verteilt; Startzeiten wie Wiederholung 0 der Warteschlangen-Simulation.
   - **Kriterium:** In allen vier Fällen enthält das Bonferroni-adjustierte 98,75-%-KI der mittleren
     gepaarten Differenz (gemessener Erfolg − vorhergesagter Erfolg derselben Episode) die Null.
     Familienweise ergibt das 95 %.
   - **Vorhersage einer Episode:** Mittel über die Latenzen ihres Roboters aus Stufe 1 von y(L), wobei y die
     zwischen den Stufen linear interpolierten Ausgänge desselben Seeds in Schritt 2 sind.
   - Verfehlt ein Fall das Kriterium, stoppe ich und erkläre.

### 0.6 Kapazität (Definition der Vorgabe, Schwellen unverändert)

Kapazität = größte Flottengröße n ∈ {1, 2, 4, 8, 16, 32}, bei der für n und alle kleineren Rastergrößen gilt:
1. **Budget-Verletzungen:** Höchstens 1 % aller Anfragen (aller Roboter und Wiederholungen, im Messfenster)
   reißen ihr Budget.
   - Eine Anfrage reißt ihr Budget, wenn ihr Plan nach dem Takt ankommt, an dem die Queue des Roboters
     leerläuft.
   - Für Episodenstart-Anfragen gilt das nominelle Budget B ab der Beobachtung.
2. **Schlechtester Roboter:** Seine vorhergesagte Erfolgsquote liegt nicht signifikant unter der eines
   einzelnen Roboters (n = 1) auf derselben Hardware.
   - Schlechtester Roboter = je Wiederholung der Roboter mit der niedrigsten Vorhersage, gemittelt über die
     Wiederholungen.
   - Signifikant darunter = die obere Grenze des 95-%-KI (Bootstrap über die Seeds der Kurve) der
     Differenz liegt unter 0.

Ergänzend, ohne Einfluss auf die Kapazität im Raster: dieselbe Rechnung für jede Flottengröße von 1 bis 128
(nur Warteschlange). Sie zeigt, wo zwischen den Rasterpunkten die Grenze liegt.

### 0.7 Wirtschaftlichkeit

- **Rechnung:** Hardwarekosten je Roboter = (Preis der GPU + Anteil des Rechners) / Kapazität, je Strategie
  und GPU-Profil. Dagegen steht ein Rechner im Roboter, der dasselbe Modell im Budget ausführen kann; seine
  Latenz geht über die Dosis-Wirkungs-Kurve in den Erfolg ein.
- **Preise und Abschreibung:** Alle Preise mit Quelle und Abrufdatum, als Annahmen gekennzeichnet.
  Abschreibung über 36 Monate; Strom mit ausgewiesenem Preis je kWh.

### 0.8 Abnahme (Vorgabe)

1. pytest, ruff und mypy grün (beide Umgebungen: Standard mit Meta-World und LIBERO).
2. LIBERO-Long reproduziert die Referenz (R1), oder die Abweichung ist erklärt.
3. Dosis-Wirkungs-Kurven liegen vor.
4. Die Validierung besteht, oder ich habe gestoppt und erklärt.
5. `reports/phase3.md`, README und die Benchmark-Anleitung sind fertig.

## Laufzeitschätzung und Kürzungen

_Stand 25.10.2026, 11:07 UTC (Commit-Zeit), nach dem Start des Piloten und vor allen Hauptläufen._

**Gemessene Kosten:** Eine Inferenz von SmolVLA auf der LIBERO-Eingabe kostet auf einem CPU-Kern 3,7 s,
einschließlich Simulator. Ein zweiter Torch-Thread beschleunigt sie nicht (4,0 s je Anfrage). Die Rechenzeit
skaliert also nur über Prozesse, und deren Zahl begrenzt der Arbeitsspeicher. Das Limit liegt bei 13,4 GB
(cgroup), nicht bei den 16 GB der Maschine.

| Block | Episoden | Anfragen (≈) | CPU-h |
|---|---:|---:|---:|
| Pilot `sync25` + `sync50` (Rest) | 186 | 2 450 | 2,5 |
| Hauptstichprobe `sync25` roh + JPEG q90 | 200 | 3 400 | 3,5 |
| Warmstart bei 1 000 ms | 100 | 1 600 | 1,6 |
| Dosis-Wirkung, 8 Bedingungen | 800 | 12 800 | 13,0 |
| Validierung, 4 Fälle | 400 | 6 800 | 6,9 |
| **Summe Kortyn-Rollouts** | **1 686** | **27 050** | **27,5** |
| `lerobot-eval` (Rest, parallel, ein Prozess) | ≈ 88 | – | ≈ 3,8 |

**Wandzeit:** Bis `lerobot-eval` fertig ist (≈ 13:45 UTC), laufen 2 Kortyn-Worker, danach 3–4. Das ergibt
≈ 8–9,5 h ab jetzt, also ≈ 11–12,5 h für Phase 3 insgesamt (Start 09:42 UTC). Der Pilot mit `sync10` käme mit
≈ 4,2 CPU-h (≈ 1,4 h) dazu. Damit läge die Gesamtzeit über 12 h.

**Kürzungen:**

| | Kürzung | Ersparnis | Was verloren geht | Status |
|---|---|---|---|---|
| 1 | Pilot ohne `sync10` | ≈ 1,4 h | Gemessener Erfolg bei 10 Aktionen je Plan. Der Report nennt stattdessen den Literaturwert; die Warteschlangen-Kapazität für 10 Aktionen je Plan bleibt erhalten, ohne Erfolgsvorhersage | **umgesetzt** |
| 2 | Dosis-Wirkung ohne 250 ms (6 Stufen + Jitter, Vorgabe ≥ 5) | ≈ 0,5 h | Ein Stützpunkt im flachen Bereich unter dem Budget | **umgesetzt** (siehe Abweichungen) |
| 3 | Validierung mit ≈ 60 statt ≈ 100 Episoden je Fall | ≈ 0,9 h | Teststärke der Validierung (breitere KI) | Reserve |

Kürzungen 2 und 3 setze ich nur ein, wenn der Zeitplan um mehr als eine Stunde rutscht. Sie werden dann unten
als Abweichung mit Zeitpunkt ausgewiesen, bevor die betroffenen Läufe starten.

## Abweichungen und Präzisierungen der Präregistrierung

Alle Einträge sind vor der Auswertung der jeweils betroffenen Ergebnisse geschrieben und committet.

| Zeit (UTC) | Punkt | Abweichung oder Präzisierung | Grund | Auswirkung |
|---|---|---|---|---|
| 11:07 | 1.1 Pilot | `sync10` entfällt (Kürzung 1) | Laufzeit; `sync10` entscheidet nichts | Kein Gate betroffen |
| 11:07 | Durchführung | Parallelität 2–4 Worker statt 4. Das Modell wird im Hauptprozess geladen und per Fork an die Worker vererbt (`--share-model`); jeder Worker gibt freien Heap nach dem Laden und nach jedem Seed zurück | Speicherlimit 13,4 GB; drei Abbrüche durch den OOM-Killer | Keine auf Ergebnisse: Fork-Worker liefern bit-gleiche Episoden wie Spawn-Worker (Test mit PushT, Stichprobe LIBERO Seed 1000 `sync50`: 1/1 Episoden-, 11/11 Anfrage-Records identisch) |
| 11:07 | Durchführung | Alle Kortyn-Rollouts mit genau einem Torch-Thread je Prozess | Fließkomma-Ergebnisse hängen von der Threadzahl ab (Seed 1001 `sync50` endet mit 2 Threads anders als mit 1) | Gepaarte Vergleiche bleiben sauber gepaart |
| 11:07 | 0.5 Strategie d | „Größter Batch, der keine erreichbare Deadline reißt und die nächste wartende Anfrage nicht verdrängt“ ist umgesetzt als gierige Vorausschau: Gewählt wird der größte EDF-Präfix, nach dem die übrigen wartenden Anfragen in EDF-Reihenfolge (jeweils größter passender Batch, p90-Servicezeit, ohne neue Ankünfte) keine Deadline reißen, die jetzt noch erreichbar ist. Passt keine Größe (Überlast), nimmt der Scheduler den größten Präfix, dessen eigene erreichbare Mitglieder rechtzeitig fertig werden | Präzisierung vor der ersten Warteschlangen-Simulation: Die wörtliche Regel prüft nur die nächste Anfrage und bedient unter Überlast weniger Roboter rechtzeitig als möglich | Gilt für d und e |
| 11:07 | 0.6 Kapazität, Strategie e | Ein abgewiesener Roboter gilt als nicht bedient (vorhergesagter Erfolg 0). Kapazität von e ist damit die größte Flotte, die vollständig aufgenommen wird und (1) und (2) erfüllt. Die Zahl der aufgenommenen Roboter wird zusätzlich berichtet | Die Definition ließ offen, wie abgewiesene Roboter zählen | Nur Strategie e |
| 11:07 | 0.5 Validierung | KI der mittleren gepaarten Differenz: Perzentil-Bootstrap über die Episoden eines Falls, 10 000 Ziehungen, Niveau 98,75 % | Die Methode war offen | – |
| 11:07 | 0.5 Episodenlängen | Episodenlängen der Warteschlangen-Simulation: alle Episoden der 0-ms-Stufe beider Aufgaben, gleich gewichtet gezogen | Präzisierung | – |
| 11:21 | 0.5 Validierung, Flottengrößen | Validierungsflotten höchstens 4 statt 8 Roboter: n₁ = größte Rastergröße ≤ 4, bei der die schlechteste Strategie Kriterium (1) erfüllt, n₂ = nächste Rastergröße; erfüllt sie (1) noch bei 4, gilt n₁ = 2, n₂ = 4. Die Policy-Inferenz der Validierung läuft in per Fork geteilten Worker-Prozessen (Ergebnis je Anfrage unabhängig vom rechnenden Prozess) | Gemessen: Eine LIBERO-Umgebung belegt ≈ 2,3 GB (Szene und Modell ≈ 1,1 GB, Software-Renderer ≈ 1 GB, Texturen 185 MB); 8 Roboter bräuchten ≈ 20 GB bei 13,4 GB Limit. Festgelegt vor der ersten Warteschlangen-Simulation | Die Validierung deckt Flotten bis 4 Roboter ab; größere Flotten bleiben unvalidierte Vorhersage (im Report so gekennzeichnet) |
| 11:21 | Durchführung | Bis `lerobot-eval` fertig ist, rechnet ein Kortyn-Worker statt zwei | Zwei Worker neben `lerobot-eval` liegen am Speicherlimit (zwei weitere OOM-Abbrüche um 11:14) | Nur Laufzeit |
| 11:43 | 0.5 Validierung, Konfiguration | Die Validierung läuft auf einem eigenen GPU-Profil „Validierung“ mit s₁ = 420 ms statt auf „RTX 3060 Ti (Schätzung)“; Batching „realistisch“ und Büro-WLAN bleiben | Die Recherche (unten) senkt die 3060-Ti-Schätzung von den eingeplanten ≈ 420 ms auf 220 ms. Damit überlastet keine Strategie das Brain mit ≤ 4 Robotern, und die Validierung prüfte nur den flachen Teil der Kurve. 420 ms ist die ursprünglich eingeplante Schätzung (Skalierung mit der Rechenleistung) und erzeugt bei 4 Robotern ohne Batching Überlast. Festgelegt vor der ersten Warteschlangen-Simulation | Die Validierung prüft die Methode (Warteschlange × Dosis-Wirkung), nicht eine bestimmte GPU |
| 11:43 | 0.5 GPU-Profile | Zusätzliches Profil „RTX 4090 kompiliert“ (`torch.compile`) als Sensitivität | Eager-PyTorch ist bei Batch 1 vom Kernel-Start-Overhead dominiert; kompiliert sinkt s₁ auf etwa ein Viertel. Zeigt, wie stark die Kapazität von der Software abhängt | Ein weiteres Modell-Profil, keine geänderte Schwelle |
| 11:47 | Transparenz | Vor der Festlegung der Profile lief ein Geschwindigkeitstest des Simulators (9 Läufe mit Platzhalterwerten s₁ = 160 ms, β = 0,25, Strategien a, b, d). Er diente nur der Laufzeitplanung und floss nicht in die Profile ein | Vollständigkeit | Keine |
| 11:47 | Durchführung | Zeitangaben dieses Abschnitts sind die Commit-Zeiten (vorher geschätzte Uhrzeiten korrigiert) | Genauigkeit | Keine |
| 11:51 | 0.6 Ergänzung 1–128 | Der Scan über alle Flottengrößen 1–128 läuft nur im Standardnetz Büro-WLAN, mit 3 Wiederholungen je Größe, ohne das Validierungsprofil, und bricht je Konfiguration ab, sobald Kriterium (1) bei 3 aufeinanderfolgenden Größen verletzt ist. Das Raster 1–32 mit 10 Wiederholungen bleibt unverändert | Laufzeit: ohne Abbruch wären es > 16 000 Läufe | Ergänzung ohne Einfluss auf die Kapazität im Raster |
| 11:52 | 0.1 Kapazität bei 10 Aktionen je Plan | Nur Kriterium (1), Büro-WLAN, 3 Wiederholungen je Rastergröße; Episodenlängen der 0-ms-Stufe bei 25 Aktionen je Plan | Ohne `sync10` gibt es keine Dosis-Wirkungskurve für diesen Takt; die Episodenlängen sind eine Näherung | Sensitivität, ohne Einfluss auf die Hauptergebnisse |
| 11:54 | 0.3 Dosis-Wirkung | Kürzung 2 umgesetzt: Stufen 0, 500, 1 000, 1 250, 1 500, 2 000 ms und Jitter (6 feste Stufen statt 7; Vorgabe ≥ 5); die Vorhersage interpoliert zwischen 0 und 500 ms linear. Holm-Korrektur über 6 statt 7 Vergleiche | Neue Laufzeitrechnung mit den gemessenen 2,3 GB je LIBERO-Umgebung: ≈ 13,8 h statt ≈ 12 h für Phase 3 (höchstens 3 Worker, Validierung nur sequenziell) | Ein Stützpunkt weniger im Bereich unter dem Budget |
| 12:07 | Transparenz | Probelauf der Warteschlangen-Simulation mit den festgelegten Profilen, aber Platzhalter-Episodenlängen (40 Läufe: RTX 3090 und Validierungsprofil, beide Netze, Batching realistisch, 2 und 16 Roboter, 1 Wiederholung), um Netz mit Stalls, alle Strategien und die Zugangskontrolle vor den echten Läufen zu prüfen. Befund: Die präregistrierte Zugangskontrolle (e) nimmt bei Überlast zu viele Roboter auf (Validierungsprofil, 16 Interessenten: 12 aufgenommen, 99,6 % ihrer Anfragen über Budget), weil ihr Worst Case gleichzeitige Anfragen in vollen Batches annimmt; tatsächlich kommen die Anfragen verteilt, warten auf einen laufenden, nicht unterbrechbaren Batch und werden in kleinen, teuren Batches bedient. Strategie e bleibt unverändert; der Befund wird berichtet. Die Ergebnisse des Probelaufs fließen nirgends ein | Vollständigkeit | Keine auf Profile, Schwellen oder Strategien |
| 13:27 | 0.7 Wirtschaftlichkeit | Ergänzung: Rechner im Roboter neben dem Brain (Raspberry Pi 5 4 GB, 114,62 € netto, 6 W) als zusätzliche Spalte und zusätzlicher Break-even. Die präregistrierte Rechnung bleibt unverändert und wird zuerst berichtet | Mit einem geteilten Brain braucht jeder Roboter weiterhin einen Rechner für Kameras, Netz und Motorbefehle; im Onboard-Fall erledigt das der Jetson selbst. Ohne diese Zeile wäre der Vergleich zugunsten des Brains verzerrt | Nur Wirtschaftlichkeit; geschrieben vor jeder Kapazitätsberechnung |

## Festlegung der Brain-Profile (vor der ersten Warteschlangen-Simulation)

_Geschrieben am 25.10.2026 um 11:43 UTC (Commit-Zeit) nach der Recherche, vor jeder Warteschlangen-Simulation, mit eigenem
Commit. Modell-Eingaben, keine Erfolgskriterien._

**Was die Recherche ergab:**
- **SmolVLA:** Für SmolVLA selbst gibt es keine publizierte Kurve Servicezeit gegen Batchgröße. SmolVLA-Paper,
  LeRobot-Paper, LeRobot-Issues und vla.cpp messen nur Batch 1.
- **Eager-PyTorch ist bei Batch 1 vom Start-Overhead der Kernels dominiert, nicht von der GPU-Rechenzeit.**
  - LeRobot-Issue #4633 (RTX 4090, eine Kamera): eager 194 ms, davon 181 ms in den 10 Flow-Schritten;
    kompiliert 39,8 ms.
  - π0.5 eager: 150,5 ms auf einer RTX 4090, 333,8 ms auf einer RTX PRO 6000 (vLLM-Omni PR #9237).
  - Deshalb unterscheiden sich GPUs bei Batch 1 weniger, als ihre Rechenleistung erwarten ließe.
- **Rechenaufwand je Roboter:** etwa 590 GFLOP (zweimal SigLIP-512, Sprachmodell-Präfix, Action-Expert).
  Das sind ≈ 6–8 ms GPU-Zeit auf einer RTX 4090, ≈ 12–15 ms auf einer RTX 3090 und ≈ 27–34 ms auf einer
  RTX 3060 Ti. Solange die Rechenzeit klein gegen den Overhead ist, kostet jeder weitere Roboter im Batch
  wenig.
- **Gemessenes Batching vergleichbarer VLAs** (β = Zusatzkosten je weiterer Anfrage relativ zu einer
  Einzelinferenz):
  - GR00T N1.7 auf L40S: 0,07–0,10 (Armory, arXiv 2608.00337).
  - Graph-optimierte VLAs: 0,16–0,35 (GR00T/π0.5 auf H200, Rosa, arXiv 2607.01088; GR00T auf RTX 5090,
    PhyAI, arXiv 2608.03682).
  - π0.5 auf L40S, Hopper und Thor: 0,42–0,50 (Armory; PhyAI).
  - OpenVLA mit Stichproben-Schleife: ≈ 1 (RoboMonkey, arXiv 2506.17811).

**Servicezeit einer Einzelinferenz s₁** (LIBERO-Eingabe mit zwei Kameras, 10 Flow-Schritte, Chunk 50, eager
PyTorch, einschließlich Vor- und Nachverarbeitung):

| Profil | s₁ | Grundlage |
|---|---:|---|
| RTX 4090 | 115 ms | LeRobot-Paper (arXiv 2602.22818, Tab. 3): 99,2 ms fp32, Eingabe nicht angegeben; plus ≈ 15 ms für die zweite Kamera und die Vorverarbeitung. Eine zweite Messung (Issue #4633) ergab 194 ms; die Spanne zeigt die Host-Abhängigkeit |
| RTX 3090 | 180 ms | vla.cpp (arXiv 2606.08094 v2, Tab. IV): 174,1 ms eager, Benchmark LIBERO, ohne Tokenisierung und Serialisierung; plus ≈ 6 ms Vorverarbeitung |
| RTX 3060 Ti (Schätzung) | 220 ms | Keine Messung gefunden. Overhead wie RTX 3090, GPU-Rechenzeit ≈ 2,2-mal so lang (32,4 statt 71 BF16-TFLOPS, 448 statt 936 GB/s): ≈ 200 ms, aufgerundet für WSL2. Wird durch die Benchmark-Messung ersetzt |
| RTX 4090 kompiliert | 50 ms | Issue #4633: 39,8 ms mit `torch.compile` (eine Kamera); plus zweite Kamera und Vorverarbeitung |
| Validierung | 420 ms | Testprofil der Validierung (siehe Abweichungen): erzeugt Überlast schon bei 4 Robotern |

**Batching-Profile** s(b) = s₁ · (1 + β · (b − 1)) · ε, ε lognormal mit 5 % Variationskoeffizient:

| Profil | β | Begründung |
|---|---:|---|
| optimistisch | 0,05 | Die GPU-Rechenzeit je weiterem Roboter versteckt sich unter dem Overhead, bis b · g ≈ s₁; gemessen bis 0,07–0,10 (GR00T, L40S) |
| realistisch | 0,15 | β ≈ g / s₁ für eager: 0,06 (4090), 0,07 (3090), 0,14 (3060 Ti); aufgerundet, damit auch kompilierte Ausführung grob abgedeckt ist |
| pessimistisch | 0,45 | Obere Grenze gemessener VLA-Batching-Kurven (π0.5 auf L40S, Hopper, Thor) |
| kein Batching-Gewinn | 1,0 | b Anfragen kosten b Inferenzen, z. B. bei einer Schleife über die Stichproben |

Alle Profile gelten bis Batch 32 (`batch_limit`). Die Scheduler b, d und e begrenzen Batches auf 16.
Messtabellen aus `kortyn/bench_batch.py` ersetzen ein Profil vollständig.
