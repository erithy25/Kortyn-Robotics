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
