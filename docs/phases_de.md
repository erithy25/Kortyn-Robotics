> **Archiv (Deutsch):** Dies war das README der Phasen 1–4. Es enthält die Ergebnisübersichten und die Reproduktionsbefehle jeder Phase. Das aktuelle README ist englisch: [`../README.md`](../README.md); der Gesamtbericht steht in [`../paper/report.md`](../paper/report.md).

# Kortyn – ausgelagerte Roboter-Policy unter schlechtem Netz

Kortyn lagert die KI eines Roboters (die *Policy*) auf einen Server im selben Netz aus („**Kortyn Brain**“)
und misst, wie zuverlässig der Roboter („**Kortyn Agent**“) weiterarbeitet, wenn das Netz schlecht wird.
Eine vortrainierte LeRobot-Policy löst eine simulierte Aufgabe, der Roboter läuft in **echter Echtzeit** (er
wartet nie auf das Modell), und ein Netzwerk-Störproxy verschlechtert die Verbindung stufenweise. Seit Phase 3a
führt ein deterministischer **Simulationszeit-Modus** dieselbe Regelschleife auf einer virtuellen Uhr aus.

| Phase | Frage | Kurzantwort | Report |
|---|---|---|---|
| **1 – Baseline** | Wie stark leidet die ausgelagerte Policy unter Latenz, Jitter und Stalls? | Echtzeit kostet nichts, solange das Latenzbudget hält; danach sinkt die Qualität mit dem Planalter. | [`reports/phase1.md`](../reports/phase1.md) |
| **2 – Freshness Engine** | Holt ein jüngeres Planalter die Qualität zurück, ohne unnötige Brain-Last? | **Nein.** Häufigere Anfragen senken das Planalter, bringen aber keine Qualität und kosten bis zu 2× Brain-Last. Keine Strategie schlägt die Baseline signifikant. | [`reports/phase2.md`](../reports/phase2.md) |
| **3a – Simulationszeit + SmolVLA** | Lässt sich die Echtzeit-Messung durch eine deterministische Simulation ersetzen? Schadet Latenz einer VLA-Policy (SmolVLA) mehr, besonders bei dynamischen Aufgaben? | **Ja**, validiert gegen Phase 1, deterministisch reproduzierbar, ≈ 6× weniger Seeds für dieselbe Präzision. SmolVLA ist in Millisekunden empfindlicher (Budget 312 statt 700 ms), relativ zum Budget nicht. Nicht die Dynamik der Aufgabe entscheidet, sondern ihre Präzisionsanforderung (über Budget: push −30 pp, coffee-push −7 pp). | [`reports/phase3a.md`](../reports/phase3a.md) |
| **3 – Flotten-Brain** | Wie viele Roboter bedient ein Brain im Budget, welche Strategie maximiert das? | **Präregistriert: einer**, egal mit welcher Strategie, weil SmolVLA auf LIBERO-Long schon unter dem Budget steil Erfolg verliert (46 % → 20 % bei 500 ms). Ergänzend: Batching vervierfacht bis verfünffacht die Flotte im Budget; bei gleichem Erfolg wie ein Jetson Orin Nano Super kostet ein Roboter am Brain ≈ halb so viel. | [`reports/phase3.md`](../reports/phase3.md) |
| **4 – Real-Time Chunking** | Macht RTC SmolVLA so verzögerungstolerant, dass ein Brain viele Roboter ohne nennenswerten Qualitätsverlust bedient? | **Nein.** LeRobots RTC streckt die Kurve nicht, es kappt sie: mittlerer Gewinn im Budget −1,6 pp (nicht nachweisbar). Eine RTX 3060 Ti bedient 2–3 Roboter ohne und 5 mit RTC; absolut bringt RTC etwa einen Roboter. Bei gleichem Erfolg ist das Brain trotzdem 2–3,5-mal günstiger je Roboter als langsame Jetsons, den Erfolg eines Thor erreicht es nicht. Die korrigierte Validierung mit 5 und 10 Robotern besteht, liegt aber gepoolt ≈ 10 pp unter der Vorhersage; mit diesem Abschlag bleibt gegen den Orin Nano Super nur Faktor 1–1,5. | [`reports/phase4.md`](../reports/phase4.md) |

- Aufgabe Phase 1–2: **PushT** (LeRobot/gym-pusht, 10 Hz, max. 300 Schritte)
- Policy Phase 1–2: **`lerobot/diffusion_pusht`** (Diffusion Policy, Apache-2.0), lokal ins aktuelle
  LeRobot-Format migriert
- Phase 3a: **Meta-World** (drawer-close, push, coffee-push; MuJoCo, 80 Hz, max. 500 Schritte) mit
  **`lerobot/smolvla_metaworld`** (SmolVLA, Apache-2.0)
- Phase 3: **LIBERO-Long** (LeRobot/LIBERO, 20 Hz, max. 520 Schritte, zwei Kameras) mit
  **`lerobot/smolvla_libero`** (SmolVLA, Apache-2.0); LIBERO läuft in einer eigenen Umgebung (`.venv-libero`)
- Phase 4: wie Phase 3, dazu LeRobots **Real-Time Chunking** (RTC, unverändert eingebunden) und verlustfreie
  Bildübertragung (zstd)
- Alles Open Source, keine Cloud, keine Telemetrie. Läuft unter Linux (CPU oder NVIDIA-GPU mit CUDA) und unter
  Windows 10/11 mit WSL2 (Ubuntu); die Messungen entstanden in einer Linux-Cloud-Umgebung ohne GPU

---

## Phase 4: Real-Time Chunking – Ergebnisse auf einen Blick

Messung vom 26./27.10.2026 in der Cloud (4-vCPU-Xeon, keine GPU, Simulationszeit), vor dem ersten Lauf
präregistriert ([`reports/phase4_preregistration.md`](../reports/phase4_preregistration.md)). GPU-Servicezeiten sind
benannte Annahmen. Vollständiger Report: **[`reports/phase4.md`](../reports/phase4.md)**, Rohtabellen in
[`reports/phase4_data.md`](../reports/phase4_data.md).

**Frage:** Kann Real-Time Chunking (RTC) SmolVLA so verzögerungstolerant machen, dass ein Brain viele Roboter
ohne nennenswerten Qualitätsverlust bedient (schlechtester Roboter höchstens 3 pp unter einem einzelnen)?

**Antwort: Nein. LeRobots RTC streckt die Kurve nicht, es kappt sie.**

| Latenz (ein Roboter, 100 Episoden je Stufe) | 0 ms | 200 ms | 300 ms | 500 ms | 1 000 ms |
|---|---:|---:|---:|---:|---:|
| ohne RTC | 47 % | 47 % | 30 % | 29 % | 5 % |
| mit RTC | 42 % | 38 % | 33 % | 30 % | 5 % |

| RTX 3060 Ti, Batching realistisch | ohne RTC | mit RTC |
|---|---:|---:|
| L* (≤ 3 pp Verlust gegenüber 0 ms derselben Bedingung) | 200 ms | 450 ms |
| Mittlerer Erfolgsgewinn im Budget (H1) | – | −1,6 pp [−6,1; +3,4] |
| Roboter je Brain (Kapazität, beste Strategie) | 3 | 5 |
| Erfolg des schlechtesten Roboters dabei | 35,9 % | 35,7 % |
| Kosten je Roboter und Monat (inkl. Raspberry Pi) | 25,66 € | 18,80 € |

- **Absolut bringt RTC etwa einen Roboter:** Den Erfolg, den das Brain mit RTC bei 5 Robotern hält, hält es
  ohne RTC bei 4. Der Sprung von 3 auf 5 kommt vor allem daher, dass das 3-pp-Kriterium relativ misst und RTC
  den Einzelroboter senkt.
- **Engpass:** Bis zur Kapazität begrenzt die Kurve, nicht die Hardware. Nach dem Budget allein trägt die Karte
  22–23 Roboter. Das WLAN bremst mit dem neuen verlustfreien Standard-Codec (zstd, 292 statt 778 kB je Anfrage,
  bitgleich) erst unter 100 Mbit/s Nutzrate.
- **Wirtschaftlichkeit:** Bei gleichem Erfolg ist das Brain 2-mal günstiger je Roboter als ein Jetson Orin Nano
  Super (16–17 Roboter) und 2–3,5-mal günstiger als ein AGX Orin (1–2 Roboter). Den Erfolg eines Thor (45 %)
  erreicht es nicht. Präregistrierte Regel: **„trägt“** (mit RTC 5 Roboter, 18,80 € gegen 118,83 € für den
  AGX Orin).
- **Validierung:** Beide präregistrierten Fälle (Strategie d mit 5 und 10 Robotern) erfüllen das Kriterium:
  32 % gemessen gegen 43 % vorhergesagt und 38 % gegen 45 % (10 Roboter mit 50 statt 60 Episoden,
  Speichergrenze). Gepoolt liegt die Messung ≈ 10 pp unter der Vorhersage, etwa zur Hälfte Glättung der Kurve.
  Im ungünstigsten Fall schrumpft der Kostenvorteil gegen den Orin Nano Super auf Faktor 1–1,5, und das
  AGX-Orin-Niveau erreicht das Brain nicht mehr.
- **Renderfehler gefunden und behoben:** Mehrere LIBERO-Simulatoren in einem Prozess löschten sich gegenseitig
  Render-Ressourcen. Das verfälschte die Flotten-Validierung aus Phase 3 und den ersten Lauf dieser Phase;
  Einzelroboter-Läufe waren nicht betroffen (Commit `4c83bac`, Report Abschnitt 5.4).
- **Nächster Schritt:** die Servicezeit messen und senken (kompilierte Laufzeit auf der RTX 3060 Ti), RTC mit
  kurzer Führung erneut messen.

---

## Phase 3: Flotten-Brain – Ergebnisse auf einen Blick

Messung vom 25.10.2026 in der Cloud (4-vCPU-Xeon, keine GPU, Simulationszeit). GPU-Servicezeiten sind
benannte Annahmen aus der Literatur, bis das Benchmark-Skript auf einer echten Karte läuft. Vollständiger Report:
**[`reports/phase3.md`](../reports/phase3.md)**, Rohtabellen in [`reports/phase3_data.md`](../reports/phase3_data.md),
Anleitung für die Messung auf deiner RTX 3060 Ti: [`docs/benchmark_rtx3060ti.md`](benchmark_rtx3060ti.md).

**Frage:** Wie viele Roboter bedient ein Brain, ohne dass einer sein Latenzbudget reißt, und welche
Scheduling-Strategie maximiert diese Zahl? Aufgabe: LIBERO-Long (zwei Aufgaben × 50 Init-States) mit
`lerobot/smolvla_libero`, ein Plan alle 1,25 s, Budget 1 250 ms, Kaltstart.

**Referenz:** Kortyn reproduziert LeRobots eigene Evaluation auf denselben 100 Episoden (54 % gegen 52 %,
McNemar p = 0,82).

**Dosis-Wirkung (ein Roboter, feste Latenz):**

| Latenz | 0 ms | 500 ms | 1 000 ms | 1 250 ms | ≥ 1 500 ms |
|---|---:|---:|---:|---:|---:|
| Erfolg | 46 % | 20 % | 6 % | 2 % | 0 % |

SmolVLA verliert schon weit unter dem Budget den größten Teil seines Erfolgs. Ursache ist sehr wahrscheinlich
der Planwechsel (ein Plan startet erst L nach seiner Beobachtung), nicht Stillstand.

**Kapazität (präregistriert): 1 Roboter je Brain**, für alle Strategien, GPUs und Batching-Annahmen. Der zweite
Roboter verlängert die mittlere Wartezeit um einige Dutzend Millisekunden. Bei dieser Kurve ist das ein
signifikanter Verlust, und das Kriterium hat keine Toleranz.

**Ergänzungen (nicht präregistriert, Vorhersage):**

| | ohne Batching (a, c) | mit Batching (b, d, e) |
|---|---:|---:|
| Roboter im Budget, RTX 3060 Ti (Kriterium 1; Raster, Scan) | 4; 5 | 16; 23–25 |
| Roboter im Budget, RTX 4090 kompiliert (Raster, Scan) | 16; 24 | 32; 110–122 |
| Erfolg des schlechtesten Roboters, RTX 3060 Ti, 16 Roboter | – | 19 % (1 Roboter: 33 %) |
| Kosten je Roboter und Monat bei gleichem Erfolg wie ein Jetson Orin Nano Super (18 %, 18,16 €) | 21,95 € (4 Roboter) | **9,22 €** (16 Roboter) |

- **Batching ist der Hebel, nicht die Reihenfolge:** EDF ohne Batching bringt nichts, mit Batching vervier- bis
  verfünffacht sich die Flotte. Die Batching-Annahme allein verschiebt die Zahl um den Faktor 8–10 (RTX 3060 Ti,
  d: 5 bis 50 Roboter im Scan).
- **Validierung:** Echte Flotten mit 2 und 4 Robotern bestätigen das Warteschlangenmodell auf wenige
  Millisekunden und erfüllen das präregistrierte Kriterium in allen vier Fällen. Der Erfolg liegt aber durchgängig
  5–7 pp unter der Vorhersage (je Fall nicht signifikant). **Nachtrag aus Phase 4:** Diese Flotten-Rollouts waren
  durch einen Renderfehler verfälscht (behoben in `4c83bac`); die 5–7 pp sind kein Befund über die Vorhersage.
- **Nächster Schritt:** Real-Time Chunking für SmolVLA (macht die Kurve flacher und damit jede Kapazität größer),
  parallel das Benchmark-Skript auf der RTX 3060 Ti.

---

## Phase 3a: Simulationszeit-Modus und SmolVLA-Baseline – Ergebnisse auf einen Blick

Messung vom 24./25.10.2026 auf derselben Maschine (4-vCPU-Xeon, keine GPU). Vollständiger Report mit allen
Tabellen, Tests und Abbildungen: **[`reports/phase3a.md`](../reports/phase3a.md)**, Rohtabellen in
[`reports/phase3a_data.md`](../reports/phase3a_data.md).

**Was neu ist:**
- **Simulationszeit-Modus** (`python -m kortyn.run_virtual`): Discrete-Event-Simulation mit virtueller Uhr.
  - Der Agent läuft Anweisung für Anweisung wie in Echtzeit; die Policy rechnet weiterhin echt.
  - *Wann* ein Plan ankommt, bestimmt ein Latenzmodell oder der Replay geloggter Phase-1-Latenzen. Das Modell
    besteht aus dem Netzmodell des Proxys mit denselben Seeds, einem FIFO-Brain mit austauschbarem Scheduler
    und der Servicezeit je Hardware-Profil.
  - Zwei Läufe derselben Konfiguration sind identisch, unabhängig von Worker-Zahl und Reihenfolge: bei
    Meta-World bitgleich, bei PushT bis auf die letzten Float-Bits der Physik-Engine (pymunk).
  - Der Echtzeit-Modus (`python -m kortyn.run_experiment`) bleibt unverändert.
- **SmolVLA auf Meta-World:**
  - Die Sprachanweisung reist im Protokoll mit.
  - Bilder lassen sich mit wählbarer Auflösung und Kompression übertragen.
  - Die Meta-World-Umgebung ist exakt wie in LeRobots Evaluation aufgebaut.

**Validierung gegen alle 550 Echtzeit-Episoden aus Phase 1** (gepaart über Seeds; Erfolg, Ø Max-Reward,
Planalter, Starvation):

| Latenzquelle | Vergleiche | KI enthält 0 | Holm-signifikant | Bewertung |
|---|---:|---:|---:|---|
| Replay der geloggten Latenzen | 44 | 42 | 2 (Planalter +0,05 ms) | ✅ alle 550 Episoden reproduziert, 441 bitgleich, der Rest bis auf die letzten Float-Bits |
| Modell, episodengenaue Servicezeit | 44 | 41 | 0 | ✅ |
| Modell, gepooltes Hardware-Profil | 52 | 49 | 2 (Planalter +14/+17 ms) | ✅ Abweichungen kommen aus der Drift der Phase-1-Maschine |

**Messpräzision:**
- Gepaarte Vergleiche naher Bedingungen streuen 2,5-mal weniger als in Echtzeit. Für dieselbe Präzision
  reichen damit ≈ 6-mal weniger Seeds.
- Die Power-Analyse ergab 100 Seeds je Bedingung. Nachweisbar sind damit Erfolgsunterschiede ab 15 pp (push)
  bzw. 10 pp (coffee-push).

**SmolVLA: Erfolgsverlust in pp gegen die synchrone Referenz** (100 Seeds je Bedingung, gepaart, ein Roboter;
Latenz = publizierte GPU-Inferenzzeit + Netz-Szenario aus Phase 1):

| GPU-Profil (Inferenz) | Netz: gut / Büro-WLAN / schlecht / sehr schlecht | drawer-close (Ref. 100 %) | push (Ref. 69 %) | coffee-push (Ref. 66 %) |
|---|---|---:|---:|---:|
| RTX 4090 (99 ms) | 4 Szenarien | ±0 / ±0 / ±0 / ±0 | −6 / −7 / −10 / **−15** | −1 / +1 / +1 / −6 |
| RTX 3090 (174 ms) | 4 Szenarien | ±0 / ±0 / ±0 / ±0 | −8 / −8 / **−19** / **−26** | −1 / +3 / −5 / −7 |
| Apple M4 (311 ms) | 4 Szenarien | ±0 / ±0 / ±0 / ±0 | **−23** / **−28** / **−32** / **−43** | −7 / **−8** / −7 / **−8** |
| Apple M1 (722 ms, Grenzfall) | nur „gut“ | **−100** | **−62** | **−40** |

Fett = 95-%-KI schließt die Null aus.

**Abgleich der Sync-Referenz:**
- **Publizierte Werte** (je 30 Episoden), auf den Seeds 1000–1029: drawer-close 100 gegen 100 %, push 77 gegen
  70 %, coffee-push 73 gegen 90 %.
- **LeRobots `lerobot-eval`**, gleiche 50 Seeds: 60 % gegen Kortyn 66 % (McNemar p = 0,45).

**Antwort auf die Kernfrage:**
- **In Millisekunden schadet Latenz SmolVLA mehr, relativ zum Budget nicht.**
  - Die Queue überbrückt bei 80 Hz nur 312,5 ms statt 700 ms.
  - Ø Verlust im/über Budget: push −10/−30 pp, coffee-push −1/−7 pp, drawer-close ±0; PushT −4/−44 pp.
- **Vorbehalt:** Meta-World-Episoden sind nach 54–79 Takten (< 1 s) entschieden.
  - Der latenzfreie Start-Plan deckt davon 50 Takte ab; Latenz trifft nur ein bis zwei Pläne, bei PushT rund 19.
  - Je betroffenem Plan ist SmolVLA-push deshalb empfindlicher als PushT.
- **„Dynamisch“ laut Experten-Test heißt nicht empfindlicher:** coffee-push verliert über dem Budget 23 pp
  weniger als push. Entscheidend ist, wie viel der Start-Plan allein löst und wie präzise die folgenden Pläne
  sein müssen.

**Konsequenzen für Phase 3 (mehrere Roboter an einem Brain):**
- **Kapazität (Rechnung):**
  - RTX 4090 ohne Batching: zwei Roboter mit Reserve; ein dritter nur mit gestaffelten Anfragen.
  - RTX 3090: ein Roboter.
  - Apple M4: schon ein Roboter liegt über dem Budget.
- **Scheduler:** Deadline (Budget − Rückweg) statt FIFO testen, mit Robotern in unterschiedlichen Netzen. Die
  Anfragefrequenz sollte je Aufgabe wählbar sein, denn push profitiert von Frische (Sync-25 gegen Sync-50:
  +12 pp).
- **Bilder:** JPEG q90 als Standard: 1,1 statt 17,7 Mbit/s je Roboter, ohne signifikanten Verlust.
  Herunterskalieren auf 240 px kostet 20 pp.
- **Aufgaben:** eine Aufgabe mit längerem Horizont ergänzen (LIBERO-Long, nur Linux, oder verkettete
  Meta-World-Aufgaben).
- **Keine reine CPU-Inferenz:** 0,9–1,7 s je Plan auf dieser Maschine.

**Laufzeit:**
- Validierung ≈ 6 h: gepoolt 2 h 21 min, episodengenau 1 h 34 min, Replay 2 h 13 min.
- SmolVLA ≈ 4,7 h in Vier-Worker-Äquivalent (Schätzung vorab: 4,7–12,7 h).
- Vieles lief parallel; insgesamt ≈ 11 h Wanduhr.

---

## Phase 2: Freshness Engine – Ergebnisse auf einen Blick

Messung vom 24.10.2026 auf derselben Maschine wie Phase 1, mit identischem Checkpoint, Brain-Profil `kalibriert`
(DDIM-4) und denselben Seeds. Vollständiger Report mit allen Tabellen, gepaarten Tests und Abbildungen:
**[`reports/phase2.md`](../reports/phase2.md)**.

**Stufe 1 – Screening:** alle 9 Strategien, 20 Seeds; Δ Reward = gepaarte Differenz zur frisch gemessenen
Baseline.

| Strategie | Anfragen/s (gut / sehr schlecht) | Auslastung Brain (gut / sehr schlecht) | Ø Planalter (gut / sehr schlecht) | Δ Reward gut | Δ Reward sehr schlecht |
|---|---:|---:|---:|---:|---:|
| Baseline (Phase 1: Schwelle 6, letzte Aktion wiederholen) | 1,23 / 1,16 | 41 % / 38 % | 0,73 / 0,98 s | – | – |
| Schwelle 6 + Notfall-Halt | 1,23 / 1,16 | 40 % / 38 % | 0,72 / 0,85 s | +0,043 | +0,002 |
| Schwelle 8 + Notfall-Halt | 1,64 / 1,50 | 54 % / 50 % | 0,63 / 0,78 s | +0,016 | −0,056 |
| Schwelle 10 + Notfall-Halt | 2,46 / 1,70 | 80 % / 56 % | 0,53 / 0,75 s | −0,064 | −0,033 |
| Adaptiv (Ziel-Planalter 0,7 s) + Notfall-Halt | 1,40 / 1,68 | 46 % / 56 % | 0,68 / 0,75 s | +0,015 | −0,063 |
| … jeweils mit Inpainting | wie oben | wie oben | wie oben | −0,069 … +0,038 | −0,092 … −0,013 |

Kein Unterschied ist nach Holm-Korrektur signifikant. Die vorab festgelegte Regel (größte mittlere
Reward-Differenz, bei < 0,01 weniger Anfragen/s) wählte `schwelle6` und `schwelle6_inpaint` für Stufe 2: das
Phase-1-Timing mit Notfall-Halt, einmal ohne und einmal mit Inpainting.

**Stufe 2 – Bestätigung:** 50 Seeds, alle Netz-Szenarien; Baseline = die Phase-1-Läufe; Δ gepaart, alle
Holm-p ≥ 0,92.

| Szenario | Baseline | Schwelle 6 + Notfall-Halt | Schwelle 6 + Inpainting |
|---|---:|---:|---:|
| gut | 62 % | 64 % (+2 pp) | 50 % (−12 pp) |
| Büro-WLAN | 70 % | 54 % (−16 pp) | 58 % (−12 pp) |
| schlecht | 46 % | 56 % (+10 pp, p = 0,36) | 44 % (−2 pp) |
| sehr schlecht | 42 % | 46 % (+4 pp, p = 0,81) | 34 % (−8 pp) |
| schlecht ohne Stall | 50 % | 66 % (+16 pp) | 48 % (−2 pp) |
| sehr schlecht ohne Stall | 50 % | 50 % (±0 pp) | 42 % (−8 pp) |
| **über alle 6 gepoolt** (300 Paare) | 53 % | 56 % (+3 pp), Reward +0,011 (p = 0,41) | 46 % (−7 pp), Reward −0,032 (p = 0,03; Holm 0,06) |

1. **Das Planalter ist nicht der Hebel.**
   - Häufigere Anfragen senken das Ø-Planalter, z. B. Schwelle 10 in „gut“ von 0,73 auf 0,53 s. Qualität bringen sie
     aber nicht zurück; tendenziell kosten sie welche.
   - Innerhalb eines Szenarios ist der Zusammenhang flach bis umgekehrt: +0,031 Reward je 0,1 s (KI −0,002 bis
     +0,073). In Phase 1, zwischen den Szenarien, fiel der Reward dagegen um −0,037 je 0,1 s. Der Unterschied ist
     signifikant (+0,067, 95 %-KI +0,025 bis +0,119).
   - Das Planalter aus Phase 1 ist damit ein Symptom von Latenz und Stalls, keine eigenständige Stellschraube.
   - Häufigere Anfragen ändern nicht die Verzögerung bis zur ersten Aktion eines Plans (Reaktionszeit p50 bleibt
     400 bzw. 500 ms). Sie erhöhen nur die Zahl der Planwechsel.
2. **Auch ohne Latenz sind 8 Aktionen je Plan das Optimum** (Zusatz-Ablation im Sync-Modus, 50 Seeds).
   - Bei 4 statt 8 Aktionen sinkt der Erfolg von 60 % auf 38 % (nominal p = 0,05, nach Holm n. s.), bei 6 auf 48 %,
     bei 12 auf 56 %. Der Reward ist bei 8 am höchsten, so wurde die Policy auch evaluiert.
   - Kürzere Pläne sind genau das, was häufigere Anfragen in Echtzeit bewirken. Sie helfen also auch dann nicht, wenn
     keine Latenz im Spiel ist.
3. **Inpainting (gradientenfrei) hilft nicht.**
   - Über alle Szenarien liegt es bei −0,032 Reward und −7 pp Erfolg, nominal signifikant, nach Holm knapp nicht.
   - Im weichen Überlappungsbereich mischt es alte und neue Vorhersage; bei 4 DDIM-Schritten bleibt zu wenig Raum, das
     auf einen gültigen Plan zurückzuziehen.
4. **Der Notfall-Halt ist neutral.**
   - Er kostet keine Qualität: gepoolt +0,011 Reward, n. s. Dafür macht er Budgetverletzungen sichtbar: 22,6
     Notfälle/min in „sehr schlecht“. Die Baseline führt dort ohne Meldung Pläne aus, die bis zu 1,9 s alt sind
     (p95).
5. **Brain-Last:** Ein Roboter belegt mit dem Phase-1-Timing ≈ 40 % des Inferenz-Threads (1,2 Anfragen/s).
   - Schwelle 10 verdoppelt das auf 80 %.
   - Die adaptive Strategie kostet +15 bis +45 % und erreicht ihr Ziel-Planalter in schlechten Netzen trotzdem nicht.
6. **Messrauschen ist so groß wie die gesuchten Effekte.**
   - Zwei identische Phase-1-Läufe unterscheiden sich um Δ Reward −0,031.
   - Die A/A-Kontrolle in Stufe 2 schwankt bis −16 pp: In „Büro-WLAN“ greift der Notfall-Halt nie, Schwelle 6 ist
     dort faktisch die Baseline.
   - Effekte unter ≈ 0,03 Reward sind mit 50 Seeds nicht belastbar.

**Konsequenzen für Phase 3 (mehrere Roboter an einem Brain):**
- **Anfrage-Timing der Baseline beibehalten:** Refill bei ≤ 6, also 8 Aktionen je Plan. Freshness-Mehrlast kauft
  keine Qualität, sie kostet nur Brain-Kapazität.
- **Kapazität:** ≈ 40 % Auslastung je Roboter heißt rechnerisch ≈ 2,5 Roboter je Brain-Instanz bei Volllast
  (DDIM-4, 3 CPU-Kerne). Mit Reserve gegen Wartezeit eher 2.
- **Der Engpass mit mehreren Robotern ist die Latenz bis zur ersten Aktion**, also Wartezeit plus Stalls. Gegenmittel:
  - Micro-Batching mehrerer Roboter in einem Denoising-Durchlauf
  - Deadline-Scheduling im Brain nach dem Queue-Füllstand der Roboter
  - Admission Control
- **Notfall-Halt** als Sicherheitsnetz pro Roboter behalten und loggen.
- **Latenzschätzer:** Median + 1,645·MAD deckt nur 83–89 % der Anfragen ab. Für Deadlines das empirische p95 des
  Fensters nehmen.
- **Messdesign:** Varianten immer zeitgleich verschränkt messen und ≥ 100 Seeds einplanen, wenn Effekte unter 0,05
  Reward zählen sollen.

**Konsequenzen für einen Wechsel auf ein modernes Modell wie SmolVLA:**
- **Echtes RTC:** SmolVLA (Flow Matching, Chunk 50, 10 Flow-Schritte) bringt LeRobots RTC mit, also Guidance per
  Autograd statt unseres gradientenfreien Inpaintings. Damit lässt sich die Freshness-Frage neu stellen: RTC soll
  genau das leisten, was hier scheiterte, nämlich häufiger umplanen ohne Konsistenzverlust.
- **Hardware:** RTC kostet einen Backward-Pass pro Flow-Schritt. Das Brain braucht dafür eine GPU und muss
  vorher kalibriert werden; das CPU-Brain von hier reicht nicht.
- **Payload:** 512²-Bilder statt 96² sind ≈ 28× mehr Pixel pro Kamera. Bandbreite und Serialisierung werden relevant:
  Bildkompression im Protokoll und Messungen mit dem Bandbreitenlimit des Proxys einplanen.
- **Neue Baseline nötig:** Für PushT gibt es keinen offiziellen SmolVLA-Checkpoint. Zwei Wege:
  - SmolVLA auf PushT feinjustieren; dann bleibt die Aufgabe vergleichbar, es kostet aber Rechenzeit auf eigener
    Hardware.
  - Auf Meta-World wechseln: offizieller Checkpoint `smolvla_metaworld`, MuJoCo, läuft ohne GPU. Der
    80-Hz-Steuertakt verschärft dort allerdings das Latenzbudget.

**Laufzeit:** Stufe 1 2 h 48 min, Stufe 2 4 h 25 min, Sync-Ablation 33 min, also 7 h 46 min statt der angepeilten
≈ 6 h. Das Protokoll (50 Seeds × 6 Szenarien) wurde für die Vergleichbarkeit mit Phase 1 nicht gekürzt.

---

## Phase 1: Ergebnisse auf einen Blick

Messung vom 23./24.10.2026 auf der Entwicklungsmaschine: Linux, 4-vCPU-Xeon, **CPU statt GPU**, Brain auf
3 Kernen. Je Zeile 50 Episoden mit identischen Seeds. Vollständiger Report mit Konfidenzintervallen, Tests und
Abbildungen: **[`reports/phase1.md`](../reports/phase1.md)**.

| Szenario | Netz (RTT) | `kalibriert` (DDIM-4, Inferenz 0,34 s): Erfolg · verhungert | `ddim10` (DDIM-10, Inferenz 0,78 s): Erfolg · verhungert |
|---|---|---:|---:|
| Referenz (synchron, Sim wartet) | – | 60 % · – | 58 % · – |
| `gut` | 0 ms | 62 % · 0 % | 20 % · 18,8 % |
| `buero_wlan` | 20 ± 10 ms | 70 % · 0 % | 22 % · 23,6 % |
| `schlecht` | 80 ± 40 ms + 300 ms Stall alle 10 s | 46 % · 0,1 % | 6 % · 36,1 % |
| `sehr_schlecht` | 150 ± 80 ms + 1 s Stall alle 10 s | 42 % · 10,7 % | 4 % · 54,0 % |
| Ablation `schlecht_ohne_stall` | 80 ± 40 ms | 50 % · 0 % | – |
| Ablation `sehr_schlecht_ohne_stall` | 150 ± 80 ms | 50 % · 0,03 % | – |

1. **Das Latenzbudget entscheidet.** Der Refill bei ≤ 6 Aktionen gibt dem Roboter 700 ms Reserve.
   - Bleibt die Ende-zu-Ende-Latenz darunter, kostet Echtzeit nichts: `kalibriert` „gut“ 62 % gegenüber 60 % synchron
     (McNemar p = 1,0).
   - Liegt schon die Inferenz darüber, bricht die Erfolgsquote selbst bei perfektem Netz ein: `ddim10` mit 0,78 s
     fällt von 58 % auf 20 % (p < 0,001). Der Roboter verhungert dann in jedem Zyklus.
2. **Der Roboter verhungert, sobald Inferenz + Netz + Stall länger als 700 ms dauern.** Bei `kalibriert` passiert das nur in Stalls:
   - 300-ms-Stalls kratzen am Budget: E2E-p99 708 ms, 0,1 % der Zeit verhungert.
   - 1-s-Stalls reißen es: p99 1,5 s, 10,7 % der Zeit, 8,6 Ereignisse/min.
3. **Hauptursache des Qualitätsverlusts ist das Alter der ausgeführten Pläne, nicht der Stillstand.**
   - Ohne Stalls verhungert der Roboter praktisch nie (≤ 0,03 % der Zeit) und verliert trotzdem fast genauso viel. Bei 150 ms RTT liegt der Max-Reward gegenüber „gut“ bei −0,092 ohne und −0,105 mit Stalls.
   - Bis ~0,75 s Planalter gibt es keinen messbaren Effekt (synchron 0,35 s ≈ „gut“ 0,74 s). Darüber fallen Erfolg und Reward stetig, egal ob Netz oder Rechenzeit das Alter verursachen. Siehe `reports/phase1_aktionsalter.png`.
4. **Büro-WLAN ist unkritisch:** 70 %, kein Unterschied zu „gut“ (p = 0,45).
5. **Statistik:** Bei n = 50 hat die Erfolgsquote ein 95 %-KI von ±14 pp.
   - Signifikant auf der Erfolgsquote sind der Echtzeit-Einbruch von `ddim10`, „sehr schlecht“ (`kalibriert`)
     sowie „schlecht“ und „sehr schlecht“ (`ddim10`).
   - Auf dem trennschärferen Max-Reward kommen bei `kalibriert` auch „schlecht“ und „sehr schlecht ohne Stall“ hinzu.
   - Das Rauschen zwischen zwei identischen Läufen beträgt ~0,03 Reward.

**Einordnung:**
- **Plausibilität:** Der Checkpoint erreicht publiziert 65,4 % (500 Episoden, DDPM-100, GPU). Unsere synchronen und
  „gut“-Werte von 58–62 % liegen im Konfidenzintervall. Die Migration ist über `lerobot-eval` validiert
  (DDPM-100: Ø Max-Reward 0,963 gegenüber 0,961 publiziert).
- **Untergrenze:** PushT ist quasistatisch, Stillstand kostet hier nur Zeit. Für dynamische Aufgaben sind die
  gemessenen Verluste eine Untergrenze.
- **Mit GPU:** Auf einer NVIDIA-GPU (z. B. RTX 3060 Ti unter WSL2) ist die Inferenz schneller, das verschiebt die
  Budget-Grenze. Vorher `kortyn.calibrate` ausführen.

**Konsequenzen für Phase 2 (Reflex-Modus) und Phase 3 (mehrere Roboter):**
- **Phase 2:**
  - Ein reiner Stillstands-Ersatz reicht nicht. Das mittlere Planalter ist Latenz + (Anfrageperiode − 1)/2. Der
    Agent muss es aktiv begrenzen: Latenz online messen und bei schlechtem Netz öfter anfragen (adaptive
    Refill-Schwelle). Damit wächst zugleich der Puffer gegen Stalls.
  - Chunk-Übergänge glätten, z. B. mit Real-Time Chunking/Inpainting.
  - Für echte Budget-Verletzungen (Stalls) braucht der Agent einen lokalen Reflex-Fallback.
- **Phase 3:** Ein Brain mit einem Inferenz-Thread ist auf dieser CPU bei ~2 Robotern gesättigt (0,34 s Inferenz
  pro 0,8 s Anfrageperiode). Häufigere Anfragen aus Phase 2 verschärfen das. Gebraucht werden Batching mit
  Latenz-Obergrenze, deadline-basiertes Scheduling nach dem Queue-Füllstand der Roboter und eine Admission
  Control.

> **Nachtrag nach Phase 2:** Die Empfehlung „bei schlechtem Netz öfter anfragen“ hat Phase 2 widerlegt.
> Ein so gesenktes Planalter bringt keine Qualität zurück und kostet Brain-Last; gradientenfreies Inpainting hilft
> nicht. Der Notfall-Halt ist umgesetzt, ohne lokales Modell (siehe „Phase 2“ oben).

---

## Schnellstart (≈ 5 Minuten + Downloads)

Voraussetzungen: Linux x86_64 oder Windows 10/11 mit WSL2 (Ubuntu; Einrichtung Schritt für Schritt in
[`docs/benchmark_rtx3060ti.md`](benchmark_rtx3060ti.md), Schritte 1–5), [`uv`](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), Git, ~4 GB freier Platz (LIBERO für Phase 3: ~15 GB).

```bash
git clone https://github.com/erithy25/Kortyn-Robotics.git
cd Kortyn-Robotics
uv sync                     # installiert Python 3.12 + alle gepinnten Pakete (uv.lock)
uv run pytest               # ~2 min, offline, ohne Modell-Download
```

Ein echter End-to-End-Lauf (Brain + Proxy + Agent, 2 Episoden im Szenario „gut“). Beim ersten Start lädt
das Brain den Checkpoint (~1 GB) von Hugging Face und migriert ihn nach `checkpoints/`:

```bash
uv run python -m kortyn.run_experiment --profile kalibriert --run gut --episodes 2 --results-dir results_smoke
uv run python -m kortyn.report --results-dir results_smoke --reports-dir reports_smoke
xdg-open reports_smoke/phase1.md   # unter WSL2: explorer.exe reports_smoke
```

## Vollständige Reproduktion

```bash
# 1. (empfohlen) Brain-Geschwindigkeit auf deiner Maschine prüfen – siehe „Kalibrierung“
uv run python -m kortyn.calibrate

# 2. Alle Szenarien × 50 Episoden für beide Brain-Profile (fortsetzbar, Abbruch mit Ctrl-C ist sicher)
uv run python -m kortyn.run_experiment

# 3. Referenzen mit LeRobots offiziellem Eval-Skript (synchron, gleiche Seeds)
uv run python -m kortyn.lerobot_reference --profile kalibriert
uv run python -m kortyn.lerobot_reference --profile ddim10
uv run python -m kortyn.lerobot_reference --scheduler DDPM --num-inference-steps 100 --name ddpm100   # ~1 h CPU

# 4. Phase 2: Stufe 1 (alle Strategien × 20 Seeds), Auswahl, Stufe 2 (2 Beste × 50 Seeds × 6 Szenarien)
uv run python -m kortyn.run_experiment --config kortyn/experiments/phase2/screening.yaml
uv run python -m kortyn.report_phase2 --select          # Rangliste nach der vorab festgelegten Regel
uv run python -m kortyn.run_experiment --config kortyn/experiments/phase2/confirmation.yaml
# Zusatz-Ablation: 4 / 6 / 12 statt 8 Aktionen je Plan im Sync-Modus
for n in 4 6 12; do uv run python -m kortyn.run_experiment --config kortyn/experiments/phase2/ablation_sync_$n.yaml; done

# 5. Reports: reports/phase1.md und reports/phase2.md (--phase 1|2 für nur einen)
uv run python -m kortyn.report

# 6. Phase 3a: Validierung des Simulationszeit-Modus gegen Phase 1 (braucht die Phase-1-Ergebnisse)
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/validation_model.yaml
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/validation_replay.yaml
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/validation_model.yaml --profile kalibriert \
    --run modell__gut --run modell__sehr_schlecht --episodes 10 --workers 1 --results-dir results/phase3a/determinism
# Aufgabenwahl (Totzeit-Toleranz der Meta-World-Experten) und SmolVLA (Checkpoint ≈ 0,9 GB beim ersten Lauf).
# MuJoCo rendert ohne Display über EGL (Linux, WSL2), überschreibbar mit MUJOCO_GL.
uv run python -m kortyn.expert_delay
for task in drawer_close push coffee_push; do
  uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/smolvla_${task}.yaml
done
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/smolvla_transport.yaml
uv run python -m kortyn.lerobot_reference --config kortyn/experiments/phase3a/smolvla_coffee_push.yaml \
    --profile smolvla --episodes 50
uv run python -m kortyn.bench_inference --config kortyn/experiments/phase3a/smolvla_push.yaml
uv run python -m kortyn.report_phase3a      # reports/phase3a_data.md + reports/phase3a_*.png

# 7. Phase 3: LIBERO-Long und Flotten-Brain. LIBERO braucht MuJoCo < 3.9 und läuft deshalb in einer eigenen
#    Umgebung (.venv-libero); Assets (gepinnte Revision) lädt die Umgebung beim ersten Start.
export UV_PROJECT_ENVIRONMENT=.venv-libero
uv sync --no-group metaworld --group libero
L="uv run --no-group metaworld --group libero python -m"
$L kortyn.run_virtual --config kortyn/experiments/phase3/libero_pilot.yaml --run sync25 --run sync50 \
    --workers 2 --share-model                       # Pilot: 10 Aufgaben x 10 Init-States
for ids in 0,1,2,3,4 5,6,7,8,9; do                  # LeRobots eigene Evaluation, dieselben Episoden
  $L kortyn.lerobot_reference --config kortyn/experiments/phase3/libero_pilot.yaml --profile smolvla_libero \
      --threads 1 --task-ids $ids --name smolvla_libero_tasks${ids//,/}
done
unset UV_PROJECT_ENVIRONMENT
uv run python -m kortyn.phase3_steps select-tasks --pilot-config kortyn/experiments/phase3/libero_pilot.yaml
export UV_PROJECT_ENVIRONMENT=.venv-libero          # Hauptstichprobe: JPEG, Kaltstart, Dosis-Wirkung
$L kortyn.run_virtual --config kortyn/experiments/phase3/libero_main.yaml --workers 3 --share-model
unset UV_PROJECT_ENVIRONMENT
uv run python -m kortyn.fleet --config kortyn/experiments/phase3/fleet.yaml          # Stufe 1: Warteschlange
uv run python -m kortyn.fleet --config kortyn/experiments/phase3/fleet.yaml --scan --networks buero_wlan \
    --out results/phase3/fleet/scan.jsonl                                            # jede Größe 1..128
uv run python -m kortyn.fleet --config kortyn/experiments/phase3/fleet_p10.yaml      # 10 Aktionen je Plan
uv run python -m kortyn.fleet_predict --config kortyn/experiments/phase3/fleet.yaml  # Stufe 2: Kapazität
uv run python -m kortyn.fleet_predict --config kortyn/experiments/phase3/fleet.yaml \
    --stage1 results/phase3/fleet/scan.jsonl --out results/phase3/fleet/capacity_scan.json
uv run python -m kortyn.fleet_predict --config kortyn/experiments/phase3/fleet_p10.yaml --queue-only \
    --out results/phase3/fleet/capacity_p10.json
uv run python -m kortyn.fleet_rollout --config kortyn/experiments/phase3/fleet.yaml --select   # 4 Fälle
export UV_PROJECT_ENVIRONMENT=.venv-libero          # Stufe 3: echte Flotten-Rollouts je Fall (hier: a/e x 2/4)
for case in "a 2" "a 4" "e 2" "e 4"; do set -- $case
  $L kortyn.fleet_rollout --config kortyn/experiments/phase3/fleet.yaml --strategy $1 --robots $2 --workers 3
done
unset UV_PROJECT_ENVIRONMENT
uv run python -m kortyn.fleet_rollout --config kortyn/experiments/phase3/fleet.yaml --evaluate
uv run python -m kortyn.report_phase3       # reports/phase3_data.md + reports/phase3_*.png

# 8. Phase 4: Real-Time Chunking (RTC), Kapazität mit 3-pp-Toleranz, WLAN, Wirtschaftlichkeit. Vorab
#    festgelegt in reports/phase4_preregistration.md; Laufzeiten und Verlauf in reports/phase4.md, Abschnitt 1.
uv run python -m kortyn.power_phase4                 # Power-Analyse (vor jedem Lauf) -> results/phase4/power.json
export UV_PROJECT_ENVIRONMENT=.venv-libero
uv sync --no-group metaworld --group libero          # neu in Phase 4: zstandard, Pillow, scipy
L="uv run --no-group metaworld --group libero python -m"
$L kortyn.run_virtual --config kortyn/experiments/phase4/libero_repro.yaml --workers 4 --share-model  # R0
$L kortyn.run_virtual --config kortyn/experiments/phase4/libero_dose.yaml --workers 4 --share-model  # A: ohne RTC
$L kortyn.bench_rtc --config kortyn/experiments/phase4/libero_rtc.yaml      # RTC-Mehraufwand r (ruhige CPU)
$L kortyn.bench_codec --config kortyn/experiments/phase4/libero_dose.yaml   # verlustfreier Codec (ruhige CPU)
$L kortyn.run_virtual --config kortyn/experiments/phase4/libero_rtc.yaml --workers 4 --share-model   # B: mit RTC
for s in $(seq 1000 1049); do                        # Ergänzung: Aufgabe 8 bei 500 ms, frischer Prozess je Episode
  $L kortyn.run_virtual --config kortyn/experiments/phase4/libero_fresh.yaml --seed-start $s --episodes 1 \
      --workers 1 --results-dir results/phase4/fresh/stream$((s % 4))
done
unset UV_PROJECT_ENVIRONMENT
F="uv run python -m kortyn.fleet --config kortyn/experiments/phase4/fleet.yaml --scan --scan-replications 5 --scan-stop-on-rejection"
$F --gpus rtx3060ti_schaetzung --networks buero_wlan --out results/phase4/fleet/scan.jsonl          # ohne RTC
$F --gpus rtx3060ti_schaetzung_rtc --networks buero_wlan --out results/phase4/fleet/scan_rtc.jsonl  # mit RTC
$F --strategies d --batching realistisch \
    --networks wlan_25,wlan_50,wlan_100,wlan_200,wlan_400,wlan_25_roh,wlan_50_roh,wlan_100_roh,wlan_200_roh,wlan_400_roh \
    --out results/phase4/fleet/scan_wlan.jsonl                                                       # geteiltes WLAN
for step in curves retest sensitivity fresh; do uv run python -m kortyn.phase4_steps $step; done  # L*, H1/H2, R-RTC, ...
C="--curve rtx3060ti_schaetzung=ohne_rtc --curve rtx3060ti_schaetzung_rtc=mit_rtc"
uv run python -m kortyn.fleet_capacity --stage1 results/phase4/fleet/scan.jsonl results/phase4/fleet/scan_rtc.jsonl \
    $C --out results/phase4/fleet/capacity.json
uv run python -m kortyn.fleet_capacity --stage1 results/phase4/fleet/scan_wlan.jsonl $C \
    --out results/phase4/fleet/capacity_wlan.json
export UV_PROJECT_ENVIRONMENT=.venv-libero           # Validierung: beste Strategie mit RTC (d), 5 und 10 Roboter
for n in 5 10; do
  $L kortyn.fleet_rollout --config kortyn/experiments/phase4/fleet.yaml --strategy d --robots $n --workers 4
done
unset UV_PROJECT_ENVIRONMENT
uv run python -m kortyn.phase4_steps validate
uv run python -m kortyn.economics_phase4 --capacity results/phase4/fleet/capacity.json \
    --rtc-overhead results/phase4/rtc_overhead.json
uv run python -m kortyn.report_phase4                # reports/phase4_data.md + reports/phase4_*.png
```

Laufzeiten auf der Messmaschine (4-vCPU-Xeon, keine GPU): Kalibrierung ≈ 10 min, alle Läufe ≈ 4 h 40 min (650
Episoden: `kalibriert` 400, `ddim10` 250), LeRobot-Referenz DDPM-100 ≈ 1 h; Phase 2: Stufe 1 ≈ 2 h 50 min,
Stufe 2 ≈ 4 h 25 min, Sync-Ablation ≈ 35 min; Phase 3a: Validierung ≈ 6 h, SmolVLA ≈ 4,7 h mit 4 Workern
(Aufschlüsselung in [`reports/phase3a.md`](../reports/phase3a.md), Abschnitt 9); Phase 3 ≈ 13 h mit 1–3 Workern
(Pilot und `lerobot-eval` ≈ 3,7 h, Dosis-Wirkung ≈ 5,3 h, Validierung ≈ 2,6 h; Verlauf in
[`reports/phase3.md`](../reports/phase3.md), Abschnitt „Tatsächlicher Verlauf“); Phase 4 ≈ 12,4 h mit 4 Workern
(Block A ≈ 4,3 h, Block B ≈ 3,0 h, Validierung ≈ 1,7 h einschließlich des ungültigen ersten Laufs; Verlauf in
[`reports/phase4.md`](../reports/phase4.md), Abschnitt 1). Mit einer NVIDIA-GPU ist die
Inferenz schneller; die Echtzeit-Episoden dauern trotzdem ihre 10–30 s Simulationszeit.

> **Hinweise für Windows (WSL2):** Den aktuellen NVIDIA-Treiber unter Windows installieren, in Ubuntu keinen;
> `nvidia-smi` muss in Ubuntu die Karte zeigen. Das Brain wählt automatisch `cuda` (Fallback CPU). Für lange Läufe
> den Ruhezustand in den Windows-Energieeinstellungen ausschalten. CPU-Pinning ist nur unter Linux aktiv.

---

## Architektur

```
run_experiment (Agent-Prozess, ohne torch)
   │ startet                      │ startet (einmal pro Lauf)
   ▼                              ▼
Kortyn Brain :50051  ◄─ TCP/HTTP2 ─  netem_proxy :50052  ◄─  Kortyn Agent (PushT, 10 Hz Echtzeit)
(gRPC, LeRobot-Policy)              (Control :50053)
```

| Modul | Aufgabe | Start |
|---|---|---|
| `kortyn/brain.py` | gRPC-Server: lädt die Policy (LeRobot-Klassen), beantwortet `Infer` mit Action-Chunks (optional mit Inpainting-Guide), ein Inferenz-Thread, Server-Timings, Last-Zähler (`GetStats`) | `python -m kortyn.brain --profile kalibriert` |
| `kortyn/policy_runner.py` | Beobachtungshistorie → Action-Chunk, exakt LeRobot-Eval-Semantik (Golden-Test) | – |
| `kortyn/inpainting.py` | Phase 2: gradientenfreies Inpainting von Diffusion-Chunks (eingefrorener + weich geführter Teil) | – |
| `kortyn/agent.py` | Simulierter Roboter: fester Takt, Action-Queue, Starvation-Zählung, Metriken; `sync`-Referenzmodus; Phase 2: Strategien, Planalter, Notfall-Halt | `python -m kortyn.agent --episodes 3` |
| `kortyn/latency.py` | Phase 2: robuste Online-Latenzschätzung (Median, 1,4826·MAD) | – |
| `kortyn/freshness.py` | Phase 2: Anfrage-Strategien (Schwelle, adaptiv), Ausrichtung, Inpainting-Guides, Planalter-Grenze | – |
| `kortyn/netem_proxy.py` | TCP-Proxy: Delay, Jitter, Bandbreite, Stalls; per YAML und Control-Port steuerbar | `python -m kortyn.netem_proxy --scenario schlecht` |
| `kortyn/metrics.py` | Records, Definitionen, Statistik (Perzentile, Wilson, McNemar, Bootstrap, Permutationstest, Holm) | – |
| `kortyn/run_experiment.py` | Startet Brain + Proxy, Szenarien seed-weise verschränkt, JSONL nach `results/` | `python -m kortyn.run_experiment` |
| `kortyn/report.py` | `reports/phase1.md` + PNG-Abbildungen (ruft auch den Phase-2-Report auf) | `python -m kortyn.report` |
| `kortyn/report_phase2.py` | `reports/phase2.md`: Tabellen, gepaarte Tests, Auswahlregel, Abbildungen | `python -m kortyn.report_phase2 --select` |
| `kortyn/calibrate.py` | Wählt die DDIM-Schrittzahl, bei der „gut“ nicht verhungert | `python -m kortyn.calibrate` |
| `kortyn/lerobot_reference.py` | Offizielles `lerobot-eval` als Referenz | `python -m kortyn.lerobot_reference --profile ddim10` |
| `kortyn/checkpoint.py` | Download (gepinnte Revision) + Migration ins Processor-Format | – |
| `kortyn/proto/kortyn_brain.proto` | Wire-Protokoll (Protobuf, kein Pickle); Phase 3a: Sprachanweisung, komprimierte Bilder | – |
| `kortyn/virtual_time.py` | Phase 3a: Discrete-Event-Simulation (virtuelle Uhr), Latenzmodelle (Netz = Proxy-Modell, Brain-Warteschlange mit austauschbarem Scheduler, Servicezeit je Hardware-Profil), Replay geloggter Latenzen | – |
| `kortyn/virtual_agent.py` | Phase 3a: derselbe Agent auf der virtuellen Uhr; Policy rechnet echt im Prozess (`LocalBrain`) | – |
| `kortyn/run_virtual.py` | Phase 3a: Simulationszeit-Läufe, parallel und bitgleich reproduzierbar, fortsetzbar (`--stop-file`) | `python -m kortyn.run_virtual --config …` |
| `kortyn/metaworld_env.py` | Phase 3a: Meta-World wie LeRobots Evaluation, Bilder nur bei Bedarf gerendert | – |
| `kortyn/expert_delay.py` | Phase 3a: Totzeit-Toleranz der skriptierten Meta-World-Experten (Aufgabenwahl) | `python -m kortyn.expert_delay` |
| `kortyn/bench_inference.py` | Phase 3a: echte CPU-Inferenzzeit der Policy | `python -m kortyn.bench_inference --config …` |
| `kortyn/env_determinism.py` | Phase 3a: Ist die PushT-Physik bitgenau reproduzierbar? | `python -m kortyn.env_determinism` |
| `kortyn/report_phase3a.py` | `reports/phase3a_data.md` + Abbildungen: Validierung, Rauschen/Power, SmolVLA | `python -m kortyn.report_phase3a` |
| `kortyn/libero_env.py` | Phase 3: LIBERO wie LeRobots Evaluation, Kameras nur bei Bedarf gerendert; mehrere Simulatoren je Prozess (Flotte) | – |
| `kortyn/fleet.py` | Phase 3/4: Warteschlangen-Simulation der Flotte (Stufe 1), Scan über alle Flottengrößen; Phase 4: geteiltes WLAN | `python -m kortyn.fleet --config …` |
| `kortyn/fleet_brain.py` | Phase 3: Brain einer Flotte in Simulationszeit: Batching, Scheduler a–e, Zugangskontrolle | – |
| `kortyn/fleet_predict.py` | Phase 3: Erfolg je Roboter aus seiner Latenzverteilung, Kapazität (Stufe 2) | `python -m kortyn.fleet_predict --config …` |
| `kortyn/fleet_rollout.py` | Phase 3/4: Validierung mit echten LIBERO-Flotten an einem Brain (Stufe 3) | `python -m kortyn.fleet_rollout --config … --strategy d --robots 5` |
| `kortyn/economics.py` | Phase 3: Kosten je Roboter, Brain gegen Rechner im Roboter | – |
| `kortyn/phase3_steps.py` | Phase 3: präregistrierte Entscheidungsregeln (Aufgabenwahl, JPEG, Kaltstart) | `python -m kortyn.phase3_steps …` |
| `kortyn/report_phase3.py` | `reports/phase3_data.md` + Abbildungen | `python -m kortyn.report_phase3` |
| `kortyn/bench_codec.py` | Phase 4: verlustfreie Codecs auf echten LIBERO-Bildern: Bitgleichheit, Größe, Zeit | `python -m kortyn.bench_codec --config …` |
| `kortyn/bench_rtc.py` | Phase 4: Rechen-Mehraufwand von RTC je Anfrage | `python -m kortyn.bench_rtc --config …` |
| `kortyn/dose_curve.py` | Phase 4: glatte, monotone Dosis-Wirkungs-Kurven (log-logistisch, Ausweichkurve PCHIP), gepaarter Bootstrap, L* | – |
| `kortyn/power_phase4.py` | Phase 4: Power-Analyse vor dem ersten Lauf | `python -m kortyn.power_phase4` |
| `kortyn/phase4_steps.py` | Phase 4: Kurven, L*, H1/H2, R-RTC, Re-Test, Validierung, Ergänzungen | `python -m kortyn.phase4_steps curves` |
| `kortyn/fleet_capacity.py` | Phase 4: Kapazität mit 3-pp- und 1-%-Kriterium aus Stufe 1 und glatter Kurve | `python -m kortyn.fleet_capacity --stage1 … --curve … --out …` |
| `kortyn/wlan.py` | Phase 4: geteiltes Funkmedium der Flotten-Simulation | – |
| `kortyn/economics_phase4.py` | Phase 4: Kosten je Roboter bei gleichem Erfolg, Entscheidungsregel | `python -m kortyn.economics_phase4 --capacity … --rtc-overhead …` |
| `kortyn/report_phase4.py` | `reports/phase4_data.md` + Abbildungen | `python -m kortyn.report_phase4` |

### Warum ein eigener Brain-Server statt LeRobots Async Inference?

LeRobots `PolicyServer`/`RobotClient` sind für echte Hardware gebaut und mit einer Simulation nicht sauber
nutzbar: Der Client verlangt einen Hardware-`Robot` und tut bei leerer Queue nichts (keine Starvation-Zählung),
der Server ist Single-Client, filtert Beobachtungen per hart kodiertem Ähnlichkeitsschwellwert, schläft
künstlich, liefert keine Server-Timings, nutzt `pickle` über das Netz – und übergibt der Policy **nur einen
Frame**, obwohl Diffusion Policy zwei Frames Historie braucht. Kortyn übernimmt die Konzepte (Chunks mit
Timesteps, Verwerfen veralteter Aktionen, Refill-Schwelle) und baut auf LeRobots Policy-Klassen auf, ohne
LeRobot zu verändern.

---

## Messmethodik

**Echtzeit-Vertrag des Agents** (`kortyn/agent.py`): absolute Deadlines (`t0 + k·dt`, kein Drift), genau ein
`env.step` pro Takt. Pro Takt k:
1. eingetroffene Chunks übernehmen – ein Chunk aus der Beobachtung von Schritt t enthält Aktionen für
   t…t+14; Schritte < k sind veraltet und werden verworfen, ein neuerer Chunk ersetzt überlappende Schritte;
2. Aktion für k ausführen – fehlt sie, **wiederholt der Roboter die letzte Aktion** (PushT: hält das letzte
   Positionsziel, also still) und der Takt zählt als **Starvation**;
3. bei ≤ 6 verbleibenden Aktionen asynchron eine neue Anfrage mit der Historie (o_{k−1}, o_k) senden
   (max. eine gleichzeitig, blockiert den Takt nie) → alle 8 Takte eine Anfrage, dieselben Beobachtungspaare
   wie LeRobots Eval;
4. Simulation einen Schritt weiter.

Beobachtung o_k wird erst zu ihrer physikalischen Zeit (Taktbeginn k) freigegeben. Vor Episodenbeginn holt
der Roboter seinen ersten Chunk („Startlinie“, nicht in der Statistik; `agent.start_mode: cold` schaltet das ab).

**Szenarien** (`kortyn/experiments/*.yaml`, Werte = Round-Trip-Zeit, je zur Hälfte pro Richtung, Jitter
gleichverteilt wie netem-Default, Stall = Link friert in beide Richtungen ein, Phase pro Episode aus dem Seed):

| Szenario | RTT | Stall |
|---|---|---|
| `gut` | 0 ms | – |
| `buero_wlan` | 20 ± 10 ms | – |
| `schlecht` | 80 ± 40 ms | 300 ms alle 10 s |
| `sehr_schlecht` | 150 ± 80 ms | 1000 ms alle 10 s |
| `schlecht_ohne_stall` (Ablation, nur `kalibriert`) | 80 ± 40 ms | – |
| `sehr_schlecht_ohne_stall` (Ablation, nur `kalibriert`) | 150 ± 80 ms | – |
| `referenz_sync` | – (Simulation wartet, LeRobot-Eval-Semantik) | – |

**Reproduzierbarkeit:** Seeds 1000–1049 (= Seeds der veröffentlichten Checkpoint-Evaluation), identische
Startzustände in allen Szenarien, Netz-Zufall und Diffusionsrauschen pro Episode/Beobachtungsschritt geseedet.
Zeitliches Rest-Rauschen ist in Echtzeitsystemen unvermeidbar und wird durch den Wiederholungslauf
`gut_wiederholung` quantifiziert.

**Messhygiene:** Brain auf Kernen 0..n−2, Agent + Proxy auf dem letzten Kern (Linux), Episoden mit
Takt-Verspätung > 50 ms, Proxy-Verspätung > 20 ms, Reconnect, CPU-Steal > 20 % oder Infrastrukturfehler
gelten als ungültig und werden wiederholt.

**Metriken** (Definitionen in `kortyn/metrics.py`): E2E-Latenz, Reaktionszeit, Netz-RTT, Jitter (RFC 5481),
Payload (Protobuf und Wire-Bytes), Queue-Füllstand pro Takt, Starvation-Takte/-Ereignisse inkl. Ursache
(Compute vs. Netz), Aktionsalter, Erfolg, Ø Max-Reward (LeRobot-Metrik, kontinuierlich) und Dauer pro Episode.
Statistik: Wilson-KI, exakter McNemar-Test und gepaarter Bootstrap über dieselben Seeds.
Phase 2 ergänzt das Planalter jeder ausgeführten Aktion (Freigabe der Beobachtung → Ausführung), Notfälle pro
Takt (Queue leer oder Plan älter als die Grenze) und die Brain-Last (Anfragen/s, Auslastung des Inferenz-Threads,
Wartezeit; gegengeprüft mit den Brain-Zählern per `GetStats`), außerdem einen gepaarten Permutationstest und die
Holm-Korrektur.
Rohdaten: `results/<profil>/<lauf>/*.jsonl`, Metadaten aller Messsitzungen in `results/<profil>/run_meta.json`.

### Kalibrierung

Diffusion Policy rechnet standardmäßig 100 DDPM-Schritte – auf der CPU ~6 s pro Chunk, nicht echtzeitfähig.
`kortyn.calibrate` misst für mehrere DDIM-Schrittzahlen, ob „gut“ ohne Starvation läuft (p99-Latenz ≤ 5 Takte).
Das Profil `kalibriert` nutzt die größte bestehende Schrittzahl (Eintrag in `kortyn/experiments/base.yaml`),
das Profil `ddim10` die im Diffusion-Policy-Paper für Echtzeit verwendete Einstellung. Für vergleichbare Zahlen
auf deinem Rechner dieselben Profile verwenden; `calibrate` zeigt dir, ob dein Brain schneller ist.

### Checkpoint-Migration

`lerobot/diffusion_pusht` liegt im Format vor LeRobot 0.4 vor. LeRobots offizielles Migrationsskript bricht
in 0.6.1 bei diesem Checkpoint ab (`Couldn't encode 84`). `kortyn/checkpoint.py` nutzt dieselben
LeRobot-Hilfsfunktionen mit korrekt dekodierter Konfiguration und lädt die Gewichte strikt.

---

## Konfiguration

Alles steht in `kortyn/experiments/base.yaml` (Ports, Pfade, Brain-Profile, Agent-/Proxy-Parameter, Seeds,
Suiten) und den Szenario-Dateien. Unbekannte Schlüssel und falsche Typen sind Fehler. CLI-Overrides:

```bash
uv run python -m kortyn.run_experiment --profile ddim10 --run schlecht --episodes 20 --seed-start 1000
uv run python -m kortyn.run_experiment --device mps --no-pin-cpus --results-dir results_mac
uv run python -m kortyn.netem_proxy --scenario sehr_schlecht --upstream 127.0.0.1:50051 --port 50052
```

Eigene Szenarien: neue YAML-Datei in `kortyn/experiments/` (Feld `name` = Dateiname), z. B. mit
`bandwidth_mbps: 5` für ein Bandbreitenlimit, dann `--run <name>`.

**Phase-2-Strategien** stehen unter `strategies:` und werden pro Lauf gewählt. Experiment-Dateien erben per
`extends:` von `base.yaml`; Dicts werden zusammengeführt, Listen ersetzt.

```yaml
extends: ../base.yaml
strategies:
  meine_strategie:
    request: {mode: adaptive, target_age_s: 0.7, latency_quantile_z: 1.645, safety_margin_ticks: 1}
    smoothing: {mode: inpaint, schedule: exp, max_overlap_steps: 10}   # none | inpaint
    guard: {mode: hold, max_plan_age_s: 1.2}                           # repeat_last (Phase 1) | hold
    alignment: timestep                                                # shift = naiver Gegenentwurf
experiment:
  results_dir: results/meine_strategie
  suites:
    - profile: kalibriert
      runs:
        - {scenario: schlecht, strategy: meine_strategie}   # Ergebnis: <results_dir>/kalibriert/meine_strategie__schlecht/
        - {scenario: schlecht, strategy: baseline}          # baseline = exakt Phase 1
```

**Phase-3a-Simulationszeit** (`python -m kortyn.run_virtual`): Läufe mit `timing:` statt Echtzeit. Ein Timing
beschreibt, wie Latenz entsteht: `latency: model` setzt sie aus Client-Overhead, dem Proxy-Netzmodell des
Szenarios, Transport-Overhead, einer Brain-Warteschlange (`brain_servers`, `scheduler`) und der Servicezeit eines
Hardware-Profils zusammen; `latency: replay` spielt die geloggten Latenzen eines Phase-1-Laufs ab. Verteilungen:
`constant`, `uniform`, `normal`, `lognormal` oder `empirical` (aus geloggten Läufen, optional `per_seed`).

```yaml
hardware_profiles:
  rtx4090:
    service:
      smolvla: {kind: lognormal, mean_ms: 99.2, std_ms: 5.0}
timings:
  schnell:
    latency: model
    hardware: rtx4090
    client_overhead: {kind: constant, value_ms: 1.0}
    transport_overhead: {kind: constant, value_ms: 5.0}
experiment:
  suites:
    - profile: smolvla
      runs:
        - {scenario: schlecht, timing: schnell, strategy: halten}   # Ergebnis: <results_dir>/smolvla/schnell__halten__schlecht/
```

**Phase 4: Real-Time Chunking und verlustfreie Bilder.** Ein Brain-Profil schaltet LeRobots RTC mit `rtc:` ein
(LeRobot bleibt unverändert, Kortyn setzt nur dessen Konfiguration), die Strategie schickt den Rest des
laufenden Plans als Führung mit (`smoothing: {mode: rtc}`), und `image_transport` wählt je Lauf die
Bildübertragung (`raw`, `jpeg`, `png`, `webp`, `zstd`). Ohne diese Angaben laufen alle bisherigen Modi
unverändert (Rohbilder, keine Führung).

```yaml
profiles:
  smolvla_libero_rtc:
    num_inference_steps: 10
    image_size: [360, 360]
    rtc: {prefix_attention_schedule: exp, max_guidance_weight: 10.0}
strategies:
  rtc:
    guard: {mode: hold}
    smoothing: {mode: rtc, schedule: exp, max_overlap_steps: 24}
experiment:
  suites:
    - profile: smolvla_libero_rtc
      runs:
        - {scenario: gut, timing: fest_300ms, strategy: rtc, image_transport: {codec: zstd, level: 9}, label: rtc_300ms}
```

## Entwicklung

```bash
uv run pytest                 # ~500 Tests: Proxy, Metriken, Codec, Migration, Brain, Golden-Tests, Echtzeit-Agent,
                              #   Latenzschätzer, Anfrage-Strategien, Inpainting, virtuelle Uhr, Latenzmodelle,
                              #   Replay, Determinismus, Meta-World, SmolVLA (mit lokalem Checkpoint), Reports,
                              #   Flotte, verlustfreie Codecs, RTC, Dosis-Kurven, Kapazität, WLAN, Kosten
uv run ruff check . && uv run ruff format --check .
uv run mypy kortyn            # strict
# Protokoll geändert? Python-Module neu erzeugen:
uv run python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. kortyn/proto/kortyn_brain.proto
```
