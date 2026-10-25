# Kortyn – ausgelagerte Roboter-Policy unter schlechtem Netz

Kortyn lagert die KI eines Roboters (die *Policy*) auf einen Server im selben Netz aus („**Kortyn Brain**“)
und misst, wie zuverlässig der Roboter („**Kortyn Agent**“) weiterarbeitet, wenn das Netz schlecht wird.
Eine vortrainierte LeRobot-Policy löst eine simulierte Aufgabe, der Roboter läuft in **echter Echtzeit** (er
wartet nie auf das Modell), und ein Netzwerk-Störproxy verschlechtert die Verbindung stufenweise.

| Phase | Frage | Kurzantwort | Report |
|---|---|---|---|
| **1 – Baseline** | Wie stark leidet die ausgelagerte Policy unter Latenz, Jitter und Stalls? | Echtzeit kostet nichts, solange das Latenzbudget hält; danach sinkt die Qualität mit dem Planalter. | [`reports/phase1.md`](reports/phase1.md) |
| **2 – Freshness Engine** | Holt ein jüngeres Planalter die Qualität zurück, ohne unnötige Brain-Last? | **Nein.** Häufigere Anfragen senken das Planalter, bringen aber keine Qualität und kosten bis zu 2× Brain-Last. Keine Strategie schlägt die Baseline signifikant. | [`reports/phase2.md`](reports/phase2.md) |

- Aufgabe: **PushT** (LeRobot/gym-pusht, 10 Hz, max. 300 Schritte)
- Policy: **`lerobot/diffusion_pusht`** (Diffusion Policy, Apache-2.0), lokal ins aktuelle LeRobot-Format
  migriert
- Alles Open Source, keine Cloud, keine Telemetrie, läuft auf macOS (Apple Silicon, MPS) und Linux (CPU)

---

## Phase 2: Freshness Engine – Ergebnisse auf einen Blick

Messung vom 24.10.2026 auf derselben Maschine wie Phase 1, mit identischem Checkpoint, Brain-Profil `kalibriert`
(DDIM-4) und denselben Seeds. Vollständiger Report mit allen Tabellen, gepaarten Tests und Abbildungen:
**[`reports/phase2.md`](reports/phase2.md)**.

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
- **Hardware:** RTC kostet einen Backward-Pass pro Flow-Schritt. Das Brain braucht dafür GPU oder MPS (Mac) und muss
  vorher kalibriert werden; das CPU-Brain von hier reicht nicht.
- **Payload:** 512²-Bilder statt 96² sind ≈ 28× mehr Pixel pro Kamera. Bandbreite und Serialisierung werden relevant:
  Bildkompression im Protokoll und Messungen mit dem Bandbreitenlimit des Proxys einplanen.
- **Neue Baseline nötig:** Für PushT gibt es keinen offiziellen SmolVLA-Checkpoint. Zwei Wege:
  - SmolVLA auf PushT feinjustieren; dann bleibt die Aufgabe vergleichbar, es kostet aber Rechenzeit auf eigener
    Hardware.
  - Auf Meta-World wechseln: offizieller Checkpoint `smolvla_metaworld`, MuJoCo, läuft auch auf macOS. Der
    80-Hz-Steuertakt verschärft dort allerdings das Latenzbudget.

**Laufzeit:** Stufe 1 2 h 48 min, Stufe 2 4 h 25 min, Sync-Ablation 33 min, also 7 h 46 min statt der angepeilten
≈ 6 h. Das Protokoll (50 Seeds × 6 Szenarien) wurde für die Vergleichbarkeit mit Phase 1 nicht gekürzt.

---

## Phase 1: Ergebnisse auf einen Blick

Messung vom 23./24.10.2026 auf der Entwicklungsmaschine: Linux, 4-vCPU-Xeon, **CPU statt MPS**, Brain auf
3 Kernen. Je Zeile 50 Episoden mit identischen Seeds. Vollständiger Report mit Konfidenzintervallen, Tests und
Abbildungen: **[`reports/phase1.md`](reports/phase1.md)**.

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
- **Auf dem Mac:** Mit MPS ist die Inferenz schneller, das verschiebt die Budget-Grenze. Vorher `kortyn.calibrate`
  ausführen.

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

Voraussetzungen: macOS mit Apple Silicon oder Linux x86_64, [`uv`](https://docs.astral.sh/uv/)
(`brew install uv` bzw. `curl -LsSf https://astral.sh/uv/install.sh | sh`), Git, ~4 GB freier Platz.

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
open reports_smoke/phase1.md   # macOS; unter Linux: xdg-open
```

## Vollständige Reproduktion

```bash
# 1. (empfohlen) Brain-Geschwindigkeit auf deiner Maschine prüfen – siehe „Kalibrierung“
uv run python -m kortyn.calibrate

# 2. Alle Szenarien × 50 Episoden für beide Brain-Profile (fortsetzbar, Abbruch mit Ctrl-C ist sicher)
caffeinate -i uv run python -m kortyn.run_experiment        # macOS: caffeinate verhindert App Nap/Sleep
# Linux: uv run python -m kortyn.run_experiment

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
# MuJoCo rendert ohne Display: Standard EGL (Linux) bzw. CGL (macOS), überschreibbar mit MUJOCO_GL.
uv run python -m kortyn.expert_delay
for task in drawer_close push coffee_push; do
  uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/smolvla_${task}.yaml
done
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/smolvla_transport.yaml
uv run python -m kortyn.lerobot_reference --config kortyn/experiments/phase3a/smolvla_coffee_push.yaml \
    --profile smolvla --episodes 50
uv run python -m kortyn.bench_inference --config kortyn/experiments/phase3a/smolvla_push.yaml
uv run python -m kortyn.report_phase3a      # reports/phase3a_data.md + reports/phase3a_*.png
```

Laufzeiten auf der Messmaschine (4-vCPU-Xeon, keine GPU): Kalibrierung ≈ 10 min, alle Läufe ≈ 4 h 40 min (650
Episoden: `kalibriert` 400, `ddim10` 250), LeRobot-Referenz DDPM-100 ≈ 1 h; Phase 2: Stufe 1 ≈ 2 h 50 min,
Stufe 2 ≈ 4 h 25 min, Sync-Ablation ≈ 35 min. Auf Apple Silicon mit MPS ist die
Inferenz schneller; die Echtzeit-Episoden dauern trotzdem ihre 10–30 s Simulationszeit.

> **Mac-Hinweise:** Netzteil anschließen, `caffeinate -i` verwenden (sonst drosselt macOS lange Läufe),
> Terminal nicht minimieren. Das Brain wählt automatisch `mps` (Fallback CPU); `PYTORCH_ENABLE_MPS_FALLBACK=1`
> wird gesetzt. CPU-Pinning ist nur unter Linux aktiv und wird auf macOS ignoriert.

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
auf deinem Mac dieselben Profile verwenden; `calibrate` zeigt dir, ob dein Brain schneller ist.

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

## Entwicklung

```bash
uv run pytest                 # ~270 Tests: Proxy, Metriken, Codec, Migration, Brain, Golden-Tests, Echtzeit-Agent,
                              #   Latenzschätzer, Anfrage-Strategien, Inpainting, virtuelle Uhr, Latenzmodelle,
                              #   Replay, Determinismus, Meta-World, SmolVLA (mit lokalem Checkpoint), Reports
uv run ruff check . && uv run ruff format --check .
uv run mypy kortyn            # strict
# Protokoll geändert? Python-Module neu erzeugen:
uv run python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. kortyn/proto/kortyn_brain.proto
```
