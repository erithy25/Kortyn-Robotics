# Kortyn – Phase 1: Baseline-Messung

Kortyn lagert die KI eines Roboters (die *Policy*) auf einen Server im selben Netz aus („**Kortyn Brain**“)
und misst, wie zuverlässig der Roboter („**Kortyn Agent**“) weiterarbeitet, wenn das Netz schlecht wird.
Phase 1 ist die **Baseline**: eine vortrainierte LeRobot-Policy löst eine simulierte Aufgabe, der Roboter
läuft in **echter Echtzeit** (er wartet nie auf das Modell), und ein Netzwerk-Störproxy verschlechtert die
Verbindung stufenweise.

- Aufgabe: **PushT** (LeRobot/gym-pusht, 10 Hz, max. 300 Schritte)
- Policy: **`lerobot/diffusion_pusht`** (Diffusion Policy, Apache-2.0), lokal ins aktuelle LeRobot-Format
  migriert
- Alles Open Source, keine Cloud, keine Telemetrie, läuft auf macOS (Apple Silicon, MPS) und Linux (CPU)

Ergebnisse: **[`reports/phase1.md`](reports/phase1.md)** (Tabellen und Abbildungen).

---

## Schnellstart (≈ 5 Minuten + Downloads)

Voraussetzungen: macOS mit Apple Silicon oder Linux x86_64, [`uv`](https://docs.astral.sh/uv/)
(`brew install uv` bzw. `curl -LsSf https://astral.sh/uv/install.sh | sh`), Git, ~4 GB freier Platz.

```bash
git clone https://github.com/erithy25/Kortyn-Robotics.git
cd Kortyn-Robotics
uv sync                     # installiert Python 3.12 + alle gepinnten Pakete (uv.lock)
uv run pytest               # ~1 min, offline, ohne Modell-Download
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

# 4. Report
uv run python -m kortyn.report
```

Laufzeiten auf der Messmaschine (4-vCPU-Xeon, keine GPU): kompletter Lauf ≈ 3,5 h. Auf Apple Silicon mit
MPS ist die Inferenz schneller; die Echtzeit-Episoden dauern trotzdem ihre 10–30 s Simulationszeit.

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
| `kortyn/brain.py` | gRPC-Server: lädt die Policy (LeRobot-Klassen), beantwortet `Infer` mit Action-Chunks, ein Inferenz-Thread, Server-Timings | `python -m kortyn.brain --profile kalibriert` |
| `kortyn/policy_runner.py` | Beobachtungshistorie → Action-Chunk, exakt LeRobot-Eval-Semantik (Golden-Test) | – |
| `kortyn/agent.py` | Simulierter Roboter: fester Takt, Action-Queue, Starvation-Zählung, Metriken; `sync`-Referenzmodus | `python -m kortyn.agent --episodes 3` |
| `kortyn/netem_proxy.py` | TCP-Proxy: Delay, Jitter, Bandbreite, Stalls; per YAML und Control-Port steuerbar | `python -m kortyn.netem_proxy --scenario schlecht` |
| `kortyn/metrics.py` | Records, Definitionen, Statistik (Perzentile, Wilson, McNemar, Bootstrap) | – |
| `kortyn/run_experiment.py` | Startet Brain + Proxy, Szenarien seed-weise verschränkt, JSONL nach `results/` | `python -m kortyn.run_experiment` |
| `kortyn/report.py` | `reports/phase1.md` + PNG-Abbildungen | `python -m kortyn.report` |
| `kortyn/calibrate.py` | Wählt die DDIM-Schrittzahl, bei der „gut“ nicht verhungert | `python -m kortyn.calibrate` |
| `kortyn/lerobot_reference.py` | Offizielles `lerobot-eval` als Referenz | `python -m kortyn.lerobot_reference --profile ddim10` |
| `kortyn/checkpoint.py` | Download (gepinnte Revision) + Migration ins Processor-Format | – |
| `kortyn/proto/kortyn_brain.proto` | Wire-Protokoll (Protobuf, kein Pickle) | – |

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
(Compute vs. Netz), Aktionsalter, Erfolg und Dauer pro Episode. Rohdaten: `results/<profil>/<lauf>/*.jsonl`.

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

## Entwicklung

```bash
uv run pytest                 # 100+ Tests: Proxy, Metriken, Codec, Migration, Brain, Golden-Test, Echtzeit-Agent
uv run ruff check . && uv run ruff format --check .
uv run mypy kortyn            # strict
# Protokoll geändert? Python-Module neu erzeugen:
uv run python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. kortyn/proto/kortyn_brain.proto
```
