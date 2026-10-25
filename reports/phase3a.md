# Kortyn Phase 3a – Simulationszeit-Modus und SmolVLA-Baseline

_Stand 25.10.2026. Messmaschine: Linux-Container mit 4 vCPU (Xeon, 2,1 GHz), keine GPU. Alle Tabellen stammen
aus [`phase3a_data.md`](phase3a_data.md) (erzeugt mit `python -m kortyn.report_phase3a`), die Rohdaten aus
`results/phase3a/`._

**Fragen:**
1. Lässt sich die Echtzeit-Messung aus Phase 1 durch eine deterministische Simulation in virtueller Zeit
   ersetzen, ohne die Ergebnisse zu verändern? Wie viel präziser wird die Messung dadurch?
2. Wie stark schadet Latenz einer aktuellen VLA-Policy (SmolVLA) auf Meta-World, je Aufgabe, Netz-Szenario und
   GPU-Klasse? Schadet sie dort mehr als der Diffusion Policy auf PushT, und mehr bei dynamischen Aufgaben?

## Kurzfassung

1. **Der Simulationszeit-Modus reproduziert Phase 1.** Validiert gegen alle 550 Echtzeit-Episoden aus Phase 1
   (2 Brain-Profile, 11 Läufe, je 50 Seeds):
   - **Replay der geloggten Latenzen:** Alle 550 Episoden werden reproduziert, 441 davon bitgleich. Die übrigen
     109 weichen nur in den letzten Float-Bits der PushT-Abdeckung ab (pymunk-Physik); keine kippt.
   - **Latenzmodell mit episodengenauer Servicezeit:** Keine der 44 gepaarten Differenzen ist nach Holm
     signifikant.
   - **Latenzmodell mit gepooltem Hardware-Profil** (die Form, die Phase 3 nutzt): 2 von 52 Differenzen sind
     signifikant. Beide betreffen das Planalter (+14/+17 ms) und gehen auf die Drift der Phase-1-Messmaschine
     zwischen zwei Sitzungen zurück.
2. **Präziser als Echtzeit:**
   - Jede Episode ist je Seed deterministisch, unabhängig von Worker-Zahl und Reihenfolge; Maschinenrauschen
     und Drift fallen weg. Zwei Ausführungen sind bei Meta-World bitgleich (identischer Digest). Bei PushT
     weichen sie nur in den letzten Float-Bits der Abdeckung ab, einer Eigenschaft der pymunk-Physik.
   - Gepaarte Vergleiche naher Bedingungen streuen 2,5-mal weniger als in Echtzeit, dank gemeinsamer
     Zufallszahlen (SD der Reward-Differenz „gut“ gegen „Büro-WLAN“: 0,068 statt 0,167). Für dieselbe
     Präzision reichen damit rund 6-mal weniger Seeds.
   - Die Power-Analyse ergab 100 Seeds je Bedingung. Geplant war, Erfolgsunterschiede ab 17 pp mit 80 % Power
     nachzuweisen; realisiert wurden 15 pp (push) und 10 pp (coffee-push).
3. **Die SmolVLA-Pipeline entspricht LeRobots eigener Evaluation.**
   - Auf denselben 50 Seeds erreicht `lerobot-eval` bei coffee-push 60 %, Kortyns synchrone Referenz 66 %.
     43 von 50 Seeds enden gleich (McNemar p = 0,45).
   - Gegen die publizierten Werte (30 Episoden je Aufgabe) liegen auf den ersten 30 Evaluations-Seeds alle
     drei Aufgaben im Konfidenzintervall: drawer-close 100 gegen 100 %, push 77 gegen 70 %, coffee-push 73
     gegen 90 %.
   - Über 100 Seeds liegt coffee-push mit 60 % signifikant unter dem publizierten Wert. Die späteren Seeds sind
     schwerer, und das Messprotokoll der Referenz ist nicht dokumentiert.
4. **In Millisekunden schadet Latenz SmolVLA mehr, relativ zum Latenzbudget nicht; der Vergleich ist aber
   durch die kurzen Meta-World-Episoden verzerrt.**

   | Ø Erfolgsverlust gegen die eigene Sync-Referenz | Latenz im Budget | Latenz über Budget |
   |---|---:|---:|
   | SmolVLA drawer-close | ±0 pp | ±0 pp |
   | SmolVLA push | −10 pp | −30 pp |
   | SmolVLA coffee-push | −1 pp | −7 pp |
   | PushT, Phase 1 (kalibriert im Budget, ddim10 darüber) | −4 pp | −44 pp |

   - **Kleineres Budget:** Bei 80 Hz und einer Refill-Schwelle von 24 Aktionen überbrückt die Queue 312,5 ms,
     bei PushT 700 ms. Latenzen von 320–470 ms kosten push 23–43 pp; PushT verliert bei 340–500 ms höchstens
     16 pp.
   - **Vorbehalt:** Meta-World-Episoden entscheiden sich in der ersten Sekunde.
     - Erfolge kommen im Median nach 54–79 Takten; davon deckt der latenzfreie Start-Plan 50 ab.
     - Latenz trifft bis zum Erfolg also nur ein bis zwei Pläne, bei PushT rund 19.
     - Dass push trotzdem bis zu 43 pp verliert, spricht je betroffenem Plan für eine *höhere* Empfindlichkeit
       als bei PushT.
     - Für den Dauerbetrieb braucht ein fairer Vergleich Aufgaben mit längerem Horizont.
5. **„Dynamisch“ im Sinne der Experten-Totzeit sagt die Empfindlichkeit nicht voraus.**
   - Für den skriptierten Experten ist coffee-push die empfindlichste der drei Aufgaben (−27 pp schon bei
     50 ms Totzeit). Mit SmolVLA verliert sie über dem Budget trotzdem signifikant weniger als push
     (Unterschied +23 pp [+12, +35]).
   - Entscheidend ist, wie viel der Start-Plan allein erledigt und wie präzise die nächsten ein bis zwei Pläne
     sein müssen.
   - Bei push wirkt das Planalter auch ohne Latenz: Sync-50 (Ø Planalter 0,30 s) verliert gegen Sync-25
     (0,15 s) 12 pp [−23, −1].
6. **Für Phase 3 (mehrere Roboter) entscheidet das Latenzbudget je Roboter.**
   - **GPU-Kapazität:** Ein RTX-4090-Brain trägt rechnerisch ohne Batching zwei SmolVLA-Roboter mit Reserve
     (63 % Auslastung). Ein dritter (95 %) passt nur mit gestaffelten Anfragen.
   - **Bildübertragung:** JPEG q90 senkt die Bandbreite je Roboter von 17,7 auf 1,1 Mbit/s, ohne
     signifikanten Erfolgsverlust (−4 pp [−12, +4], n = 50).
   - **Keine reine CPU-Inferenz:** Auf dieser Maschine dauert ein Plan 0,9–1,7 s, länger als ein ganzer Chunk
     (625 ms).

## Ergebnisse auf einen Blick

**Was die Tabelle zeigt:**
- Erfolgsverlust je Bedingung gegenüber der synchronen Referenz mit 25 Aktionen je Plan (Sync-25:
  drawer-close 100 %, push 69 %, coffee-push 66 %).
- Gepaart über 100 Seeds; 95-%-KI per Bootstrap über Seeds.
- Ein Roboter; Latenz = publizierte GPU-Inferenzzeit + Netz-Szenario aus Phase 1.
- Letzte Spalte: ✓ = KI enthält die Null.

| Bedingung | Δ drawer-close-v3 | Δ push-v3 | Δ coffee-push-v3 | coffee-push − push [KI] |
|---|---:|---:|---:|---:|
| RTX 4090 (99 ms), gut | +0 pp [+0, +0] | -6 pp [-15, +2] | -1 pp [-8, +6] | ✓ +5 pp [-6, +17] |
| RTX 4090 (99 ms), Büro-WLAN | +0 pp [+0, +0] | -7 pp [-16, +2] | +1 pp [-6, +8] | ✓ +8 pp [-4, +20] |
| RTX 4090 (99 ms), schlecht | +0 pp [+0, +0] | -10 pp [-19, +0] | +1 pp [-6, +7] | ✓ +11 pp [-1, +23] |
| RTX 4090 (99 ms), sehr schlecht | +0 pp [+0, +0] | -15 pp [-26, -4] | -6 pp [-13, +0] | ✓ +9 pp [-4, +22] |
| RTX 3090 (174 ms), gut | +0 pp [+0, +0] | -8 pp [-17, +1] | -1 pp [-9, +6] | ✓ +7 pp [-5, +19] |
| RTX 3090 (174 ms), Büro-WLAN | +0 pp [+0, +0] | -8 pp [-17, +1] | +3 pp [-4, +10] | ✓ +11 pp [+0, +23] |
| RTX 3090 (174 ms), schlecht | +0 pp [+0, +0] | -19 pp [-29, -9] | -5 pp [-12, +2] | ✗ +14 pp [+1, +26] |
| RTX 3090 (174 ms), sehr schlecht | +0 pp [+0, +0] | -26 pp [-36, -16] | -7 pp [-14, +0] | ✗ +19 pp [+7, +32] |
| Apple M4 (311 ms), gut | +0 pp [+0, +0] | -23 pp [-34, -12] | -7 pp [-14, +0] | ✗ +16 pp [+3, +29] |
| Apple M4 (311 ms), Büro-WLAN | +0 pp [+0, +0] | -28 pp [-39, -17] | -8 pp [-15, -1] | ✗ +20 pp [+7, +33] |
| Apple M4 (311 ms), schlecht | +0 pp [+0, +0] | -32 pp [-42, -21] | -7 pp [-14, +0] | ✗ +25 pp [+13, +38] |
| Apple M4 (311 ms), sehr schlecht | +0 pp [+0, +0] | -43 pp [-54, -32] | -8 pp [-15, -1] | ✗ +35 pp [+22, +48] |
| Apple M1 (722 ms), gut | -100 pp [-100, -100] | -62 pp [-71, -52] | -40 pp [-50, -30] | ✗ +22 pp [+8, +36] |

![SmolVLA: Erfolg je Aufgabe, GPU-Profil und Netz-Szenario](phase3a_smolvla.png)

---

## 1. Der Simulationszeit-Modus

### Prinzip

Im Echtzeit-Modus von Phase 1 tickt die Wanduhr. Jede Zahl trägt deshalb das Rauschen der Maschine:
schwankende Inferenzzeit, CPU-Steal, Takt-Jitter. Der Simulationszeit-Modus (`kortyn/virtual_time.py`,
`kortyn/virtual_agent.py`) ersetzt die Wanduhr durch eine **virtuelle Uhr** in einer Discrete-Event-Simulation:

1. **Ereignisse** liegen in einer Prioritätswarteschlange und laufen nach (Zeit, Priorität,
   Einfügereihenfolge).
   - Fallen mehrere auf denselben Zeitpunkt, kommen Lieferungen an den Roboter vor Brain-Ereignissen und
     diese vor Kontrolltakten.
   - Ein Plan, der exakt zum Taktbeginn ankommt, wird also in diesem Takt übernommen, wie in Echtzeit.
2. **Der Agent** ist dieselbe Regelschleife wie in Echtzeit, Anweisung für Anweisung. Takt *k* beginnt bei
   t0 + k·dt (plus einer modellierten oder wiederholten Verspätung). In jedem Takt:
   1. Er übernimmt alle bis dahin gelieferten Pläne, zeitschrittgenau; veraltete Aktionen fallen weg.
   2. Er führt die Aktion für Schritt *k* aus. Ist die Queue leer (Starvation), überbrückt er nach der
      Strategie des Laufs: Phase 1 wiederholt die letzte Aktion (PushT: Positionsziel halten), SmolVLA steht
      still.
   3. Er entscheidet über eine neue Anfrage und macht genau einen Simulationsschritt.

   Auf das Brain wartet er nie.
3. **Eine Anfrage**, die zum Zeitpunkt t gestellt wird, liefert ihren Plan bei t + L.
   - Der Inhalt des Plans kommt von der echten Policy im selben Prozess (`LocalBrain`: derselbe
     Validierungs- und Inferenzpfad wie das gRPC-Brain).
   - Simuliert wird nur, *wann* der Plan ankommt.
4. **Gemeinsamer Code:** Request-Bau, Records, Starvation-Zuordnung und Gültigkeitsprüfung liegen in
   `AgentCore`; Echtzeit- und Simulationszeit-Agent nutzen sie identisch.

### Der Echtzeit-Modus bleibt unverändert

- `python -m kortyn.run_experiment` mit den Phase-1/2-Konfigurationen läuft wie bisher. Der
  Simulationszeit-Modus hat einen eigenen Einstiegspunkt (`python -m kortyn.run_virtual`).
- Die Echtzeit-Schleife wurde nur verschoben, nicht geändert: Ein AST-Vergleich vor und nach der Verschiebung
  ist gleich, alle Phase-1/2-Tests sind grün.
- Neue Konfigurationsfelder haben das alte Verhalten als Standard: Bildübertragung `raw`, Starvation
  `repeat_last`, Servicezeit-Quelle `per_seed: false`. Ältere gespeicherte Läufe gelten bei der
  Metadaten-Prüfung weiter als gleich konfiguriert.

### Zusammensetzung der Latenz L (Modus `model`)

| Baustein | Modell | Quelle in der Validierung |
|---|---|---|
| Client-Overhead (Beobachtung → Versand) | Verteilung | empirisch: geloggte `t_send − t_obs` aus Phase 1 (Median 0,3 ms) |
| Uplink | Netzmodell des Proxys (`NetworkState`: Delay, Jitter, Stalls, Bandbreite, Reihenfolge) mit denselben Seeds wie der Proxy, auf der virtuellen Uhr | Szenario-YAML |
| Transport-Overhead (gRPC, Loopback, Proxy) | Verteilung | empirisch: Netzanteil im Szenario „gut“ (Median 3,3 ms) |
| Zustell-Verspätung des Proxys | konstant 1 ms je zurückgehaltener Nachricht und Richtung | Der Proxy-Timer (asyncio) wacht ≈ 1 ms zu spät auf; der Netzanteil liegt in allen gestörten Szenarien 1,8–2,1 ms über RTT + „gut“ |
| Brain-Warteschlange | `VirtualBrain`: *n* Server (hier 1), Scheduler austauschbar (hier FIFO); Anfragen, deren Deadline abgelaufen ist, werden übersprungen | – |
| Servicezeit | Hardware-Profil je Brain-Profil | empirisch: geloggte Serverzeiten aus Phase 1, gepoolt oder episodengenau |
| Downlink | wie Uplink | Szenario-YAML |
| Takt-Verspätung | Verteilung (Standard 0 = idealer Takt) | – |

Phase 3 kann mehrere Roboter (je eigene Umgebung, eigener Link) an einen `VirtualBrain` hängen und den
Scheduler tauschen (FIFO → EDF, Round-Robin, Batching), ohne den Agenten zu ändern.

### Replay (Modus `replay`)

Der Replay spielt einen geloggten Echtzeitlauf nach:
- Die Simulation läuft auf der Episodenuhr des Laufs (t0 = 0); jeder Takt beginnt exakt so spät wie geloggt.
- Hat der geloggte Lauf im selben Schritt angefragt, wird die Anfrage zum geloggten Zeitpunkt beantwortet,
  mit geloggter Ende-zu-Ende-Latenz und Serverzeiten.
- Divergiert die Episode (Anfrage in anderen Schritten), leiht sich die Anfrage die Latenz der
  nächstgelegenen geloggten Anfrage und zählt als Fallback.

### Determinismus

- **Zufall:** Jede Zufallsgröße kommt aus einem eigenen Strom, per SHA-256 abgeleitet aus Episoden-Seed,
  Timing-Seed, Roboter, Komponente und Beobachtungsschritt.
  - Eine Latenz hängt damit nicht davon ab, wie viele Anfragen vorher gezogen wurden.
  - Bedingungen mit gleichem Seed teilen ihre Zufallszahlen (Common Random Numbers).
- **Policy:** Bei fester Threadzahl liefert die Policy für gleiche Eingaben bitgleiche Pläne.
  - Das Rauschen kommt aus einem pro Anfrage geseedeten CPU-Generator.
  - Die Threadzahl ist Teil der Konfiguration: 1 statt 3 Threads ändert die letzten Float-Bits.
- **Unabhängig von Parallelität:** Request-IDs beginnen in jeder Episode bei 1. Der Inferenz-Cache (Memoisierung
  nach Anfrage-Inhalt) ändert keine Ergebnisse.
- **Geprüft per Test:** zwei identische Läufe und ein dritter mit anderer Worker-Zahl, der halb fortgesetzt
  wird (auch nach einem geordneten Stopp per `--stop-file`). Der Digest über alle Records ist identisch.
- **Geprüft per Wiederholung echter Läufe** mit anderer Worker-Zahl (Abschnitt 2):
  - PushT: zwei Validierungsläufe, dreimal ausgeführt.
  - SmolVLA/Meta-World: zwei Echtzeit-Bedingungen, zweimal ausgeführt.
- **Grenze:** Die PushT-Physik (pymunk) ist selbst nicht bitgenau reproduzierbar.
  - Dieselbe Aktionsfolge auf demselben Seed ergibt je nach Env-Instanz und Prozess Abdeckungen, die in der
    letzten Stelle abweichen (≤ 2·10⁻¹⁶, `python -m kortyn.env_determinism`).
  - Zeitpunkte, Takt-Serien, Erfolg und Länge bleiben davon unberührt. In einer chaotischen Kontaktdynamik
    kann ein solches Bit im Prinzip aber eine Episode kippen.
  - MuJoCo (Meta-World) ist bitgenau reproduzierbar.

---

## 2. Validierung gegen Phase 1

### Design

Verglichen wird mit Phase 1 unter gleichen Bedingungen:
- gleiche Seeds (1000–1049), gleiche Policy;
- gleiche Brain-Profile (kalibriert = DDIM-4, ddim10 = DDIM-10);
- alle Echtzeit-Szenarien, einschließlich der beiden Stall-Ablationen.

Jede Phase-1-Messung wird ihrem Gegenstück in Simulationszeit gegenübergestellt, **gepaart über die Seeds**:
- Differenz und 95-%-KI per Bootstrap über Seeds;
- p-Werte per McNemar (Erfolg) bzw. Vorzeichen-Permutation (übrige Metriken), Holm-korrigiert.

Die Latenz kommt aus drei Quellen, die jeweils mehr vom echten Lauf übernehmen:

| Quelle | Netz | Servicezeit des Brains | Takt | Was sie prüft |
|---|---|---|---|---|
| **Modell (gepoolt)** | Proxy-Modell, neu gezogen | Verteilung aller Phase-1-Läufe des Profils (Hardware-Profil `phase1_cpu`) | ideal | das, was Phase 3 tatsächlich nutzt: Szenario + Hardware-Profil |
| **Modell (episodengenau)** | Proxy-Modell, neu gezogen | Verteilung der Serverzeiten *derselben* Phase-1-Episode (gleicher Lauf, gleicher Seed) | ideal | DES-Mechanik + Netzmodell, ohne die Drift der Messmaschine |
| **Replay** | geloggte Ende-zu-Ende-Latenz je Anfrage | geloggt | geloggte Verspätung je Takt | DES-Mechanik allein: Gleiche Latenzen müssen gleiche Episoden ergeben |

In keine Quelle fließen Ergebnisse aus Phase 1 ein (Erfolg, Reward), nur Zeiten. Die episodengenaue Quelle
zieht die Servicezeiten neu und in anderer Reihenfolge als geloggt. Sie übernimmt nur, wie schnell die
Maschine während dieser Episode war.

**Warum drei Quellen?** Die Phase-1-Maschine war nicht stationär:
- Ihre Inferenz wurde im Lauf einer Sitzung schneller.
- Die Stall-Ablationen liefen in einer späteren, schnelleren Sitzung (Tabellen unten).
- Ein gepooltes Hardware-Profil kann diese Drift nicht abbilden.
- Bei ddim10 liegen die Latenzen zudem nahe an den Taktgrenzen (Chunk-Periode 800 ms, Latenz ≈ 780–950 ms).
  Dort verschieben schon 10 ms Drift die Starvation messbar.

Die episodengenaue Quelle trennt diesen Hardware-Effekt vom Simulationsfehler.

### Bestehenskriterium

Festgelegt, nachdem die gepoolten Läufe Abweichungen zeigten, die mit der Drift zusammenfielen, und bevor
die episodengenauen Läufe gerechnet waren:
1. **Modell (episodengenau) und Replay:** Die gepaarten 95-%-KI der Differenz für Erfolg, Ø Max-Reward,
   Planalter und Starvation enthalten die Null.
   - Bei rund 45 Vergleichen je Quelle sind zwei bis drei Ausreißer allein durch Zufall zu erwarten (5 %).
   - Nach Holm-Korrektur darf keine Differenz signifikant sein.
2. **Modell (gepoolt):** dasselbe, außer wo die Abweichung nachweislich aus der Hardware-Drift kommt, also in
   der episodengenauen Quelle verschwindet.
3. **Determinismus:** Zwei Ausführungen derselben Konfiguration sind bitgleich.

### Ergebnis: bestanden

| Quelle | Vergleiche | KI der Differenz enthält 0 | Wert im KI der Echtzeitmessung | Holm-signifikant |
|---|---:|---:|---:|---:|
| Simulationszeit: Modell (gepoolt) | 52 | 49 | 48 | 2 |
| Simulationszeit: Modell (episodengenau) | 44 | 41 | 40 | 0 |
| Simulationszeit: Replay | 44 | 42 | 44 | 2 |

„Wert im KI der Echtzeitmessung“ ist das schwächere, ungepaarte Kriterium: Der Simulationswert liegt im
95-%-KI des Echtzeit-Werts.

**Modell (gepoolt), 52 Vergleiche:** 49 Intervalle enthalten die Null. Die drei Ausnahmen betreffen alle das
Planalter bei kalibriert, und alle drei folgen aus der Hardware-Drift:
- gut −3 ms [−6, −0]
- schlecht o. Stall +14 ms [+10, +17] (Holm-signifikant)
- sehr schlecht o. Stall +17 ms [+14, +20] (Holm-signifikant)

Erklärung und Verlauf:
- Die beiden Ablationen liefen in Phase 1 in einer späteren Sitzung, in der die Maschine 15 ms schneller
  inferierte (Median 324 statt 339 ms).
- Das gepoolte Profil mischt beide Sitzungen. Die Simulation wird dadurch bei den Ablationen zu langsam und
  bei den Läufen der Hauptsitzung leicht zu schnell.
- ddim10 liegt mit allen 50 Seeds vollständig in den Intervallen. Nach 20 Seeds war das noch nicht so (sehr
  schlecht +25 pp Erfolg, −2,6 pp Starvation), weil die echte Maschine in den ersten Seeds 10–20 ms langsamer
  war. Über den Rest der Sitzung mittelt sich das heraus.

**Modell (episodengenau), 44 Vergleiche:** 41 Intervalle enthalten die Null, keine Differenz ist nach Holm
signifikant. Die Drift-Abweichungen verschwinden wie erwartet:

| Planalter, Δ zur Echtzeit (kalibriert) | gepoolt | episodengenau |
|---|---:|---:|
| gut | −3 ms [−6, −0] | −1 ms [−3, +1] |
| schlecht o. Stall | +14 ms [+10, +17] | +1 ms [−2, +4] |
| sehr schlecht o. Stall | +17 ms [+14, +20] | +4 ms [+1, +6] |

Es bleiben zwei Arten von Ausreißern:
- **Büro-WLAN, Erfolg −20 pp [−34, −6]:**
  - Der empfindlichere Reward zeigt dort keine Differenz (+0,015 [−0,018, +0,065]).
  - Der Phase-1-Lauf war mit 70 % ohnehin der höchste aller Echtzeit-Läufe (gut 62 %, Wiederholung 60 %).
  - Die Simulation (50 %) liegt auf der anderen Seite desselben Rauschens: Episoden knapp an der
    Erfolgsschwelle (Abdeckung 95 %) kippen mit der Latenz-Ziehung.
- **Planalter +3 bis +4 ms** (Büro-WLAN, sehr schlecht o. Stall):
  - Systematisch, aber nur 0,4 % des Planalters und weit unter jeder Wirkung auf Erfolg oder Reward. Sichtbar
    wird es nur, weil das Planalter auf ±3 ms genau gemessen ist.
  - Wahrscheinliche Quelle ist die Netznachbildung im Millisekundenbereich: Der Proxy liest eine Nachricht in
    mehreren TCP-Bursts (≈ 1,2 je Nachricht) mit je eigenem Delay, das Modell überträgt sie als Ganzes. Die
    Zustell-Verspätung des Proxys ist als konstante 1 ms modelliert.

**Replay, 44 Vergleiche:**
- **Treue:** Alle 550 Phase-1-Episoden werden reproduziert: 441 bitgleich, 109 gleich bis auf die letzten
  Float-Bits der Abdeckung (pymunk, siehe Determinismus), keine weicht ab. Keine Anfrage musste auf eine
  Ersatz-Latenz zurückgreifen. Erfolg, Länge, alle Takt-Serien, Starvation und Reward sind identisch.
- **Zwei Vergleiche wörtlich signifikant:** das Planalter in „gut“ (kalibriert +0,045 ms, ddim10 +0,062 ms im
  Mittel je Aktion; 99 % der Aktionen liegen innerhalb ±0,7 ms).
  - Das sind 0,006 % des Planalters. Signifikant werden sie nur, weil das Planalter auf Mikrosekunden genau
    gemessen wird und die Differenz bei identischen Episoden kaum streut.
  - Ursache ist die Zeitbasis: Echtzeit misst Freigabe und Ausführung direkt mit der Wanduhr. Der Replay setzt
    beide aus Taktnummer und geloggter Takt-Verspätung zusammen; die Verarbeitungszeit innerhalb des Takts
    fehlt darin.
  - Mit einer Äquivalenzgrenze von ±1 ms (ein Tausendstel des Planalters) besteht auch dieser Vergleich.
- **Bewertung:** bestanden, mit diesen zwei Vergleichen offen ausgewiesen als Abweichung vom wörtlichen
  Kriterium.

![Validierung](phase3a_validation.png)

### Alle Vergleiche

**Profil kalibriert** (DDIM-4): Echtzeit gegen Simulationszeit, erst die Niveaus, dann die gepaarten
Differenzen (✓ = KI enthält die Null).

| Szenario | Messung | Erfolg [95-%-KI] | Ø Max-Reward [KI] | Planalter [KI] | Starvation [KI] |
|---|---|---:|---:|---:|---:|
| **Referenz (sync)** | Echtzeit (Phase 1) | 60 % [46–72] | 0.952 [0.914–0.983] | 0.348 s [0.347–0.348] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (gepoolt) | 58 % [44–71] | 0.944 [0.894–0.982] | 0.348 s [0.347–0.348] | 0.0 % [0.0–0.0] |
| **gut** | Echtzeit (Phase 1) | 62 % [48–74] | 0.969 [0.938–0.994] | 0.743 s [0.740–0.746] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (gepoolt) | 60 % [46–72] | 0.958 [0.918–0.987] | 0.740 s [0.738–0.742] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (episodengenau) | 62 % [48–74] | 0.960 [0.926–0.988] | 0.742 s [0.739–0.745] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Replay | 62 % [48–74] | 0.969 [0.938–0.994] | 0.743 s [0.740–0.746] | 0.0 % [0.0–0.0] |
| **gut (Wiederholung)** | Echtzeit (Phase 1) | 60 % [46–72] | 0.937 [0.879–0.980] | 0.742 s [0.739–0.744] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (gepoolt) | 60 % [46–72] | 0.958 [0.918–0.987] | 0.740 s [0.738–0.742] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (episodengenau) | 56 % [42–69] | 0.947 [0.905–0.981] | 0.743 s [0.740–0.745] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Replay | 60 % [46–72] | 0.937 [0.879–0.980] | 0.742 s [0.739–0.744] | 0.0 % [0.0–0.0] |
| **Büro-WLAN** | Echtzeit (Phase 1) | 70 % [56–81] | 0.946 [0.891–0.985] | 0.748 s [0.745–0.751] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (gepoolt) | 62 % [48–74] | 0.954 [0.911–0.987] | 0.747 s [0.745–0.749] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (episodengenau) | 50 % [37–63] | 0.961 [0.927–0.987] | 0.752 s [0.748–0.756] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Replay | 70 % [56–81] | 0.946 [0.891–0.985] | 0.748 s [0.745–0.751] | 0.0 % [0.0–0.0] |
| **schlecht** | Echtzeit (Phase 1) | 46 % [33–60] | 0.876 [0.798–0.939] | 0.832 s [0.828–0.836] | 0.1 % [0.1–0.2] |
|  | Simulationszeit: Modell (gepoolt) | 52 % [39–65] | 0.900 [0.822–0.962] | 0.830 s [0.826–0.832] | 0.2 % [0.1–0.3] |
|  | Simulationszeit: Modell (episodengenau) | 48 % [35–61] | 0.914 [0.838–0.973] | 0.833 s [0.828–0.837] | 0.2 % [0.1–0.3] |
|  | Simulationszeit: Replay | 46 % [33–60] | 0.876 [0.798–0.939] | 0.832 s [0.828–0.836] | 0.1 % [0.1–0.2] |
| **sehr schlecht** | Echtzeit (Phase 1) | 42 % [29–56] | 0.863 [0.790–0.926] | 1.004 s [0.996–1.013] | 10.7 % [10.2–11.2] |
|  | Simulationszeit: Modell (gepoolt) | 42 % [29–56] | 0.885 [0.818–0.944] | 1.003 s [0.995–1.011] | 10.7 % [10.1–11.2] |
|  | Simulationszeit: Modell (episodengenau) | 44 % [31–58] | 0.858 [0.781–0.928] | 1.012 s [1.001–1.023] | 11.0 % [10.5–11.6] |
|  | Simulationszeit: Replay | 42 % [29–56] | 0.863 [0.790–0.926] | 1.004 s [0.996–1.013] | 10.7 % [10.2–11.2] |
| **schlecht o. Stall** | Echtzeit (Phase 1) | 50 % [37–63] | 0.904 [0.835–0.962] | 0.804 s [0.801–0.807] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (gepoolt) | 54 % [40–67] | 0.890 [0.808–0.955] | 0.818 s [0.815–0.821] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (episodengenau) | 52 % [39–65] | 0.936 [0.886–0.976] | 0.805 s [0.802–0.808] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Replay | 50 % [37–63] | 0.904 [0.835–0.962] | 0.804 s [0.801–0.807] | 0.0 % [0.0–0.0] |
| **sehr schlecht o. Stall** | Echtzeit (Phase 1) | 50 % [37–63] | 0.876 [0.791–0.944] | 0.864 s [0.861–0.867] | 0.0 % [0.0–0.1] |
|  | Simulationszeit: Modell (gepoolt) | 34 % [22–48] | 0.896 [0.826–0.950] | 0.881 s [0.878–0.883] | 0.0 % [0.0–0.1] |
|  | Simulationszeit: Modell (episodengenau) | 46 % [33–60] | 0.880 [0.801–0.946] | 0.868 s [0.864–0.871] | 0.0 % [0.0–0.1] |
|  | Simulationszeit: Replay | 50 % [37–63] | 0.876 [0.791–0.944] | 0.864 s [0.861–0.867] | 0.0 % [0.0–0.1] |

| Szenario | Quelle | Δ Erfolg [KI] | Δ Ø Max-Reward [KI] | Δ Planalter [KI] | Δ Starvation [KI] |
|---|---|---:|---:|---:|---:|
| Referenz (sync) | Modell | ✓ -2 pp [-6, +0] | ✓ -0.008 [-0.024, +0.000] | ✓ -0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| gut | Modell | ✓ -2 pp [-18, +14] | ✓ -0.011 [-0.043, +0.019] | ✗ -3 ms [-6, -0] | ✓ +0.0 pp [+0.0, +0.0] |
| gut | Modell episodengenau | ✓ +0 pp [-16, +16] | ✓ -0.008 [-0.029, +0.010] | ✓ -1 ms [-3, +1] | ✓ +0.0 pp [+0.0, +0.0] |
| gut | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [+0.000, +0.000] | ✗ +0 ms [+0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| gut (Wiederholung) | Modell | ✓ +0 pp [-16, +16] | ✓ +0.020 [-0.000, +0.048] | ✓ -2 ms [-4, +1] | ✓ +0.0 pp [+0.0, +0.0] |
| gut (Wiederholung) | Modell episodengenau | ✓ -4 pp [-18, +10] | ✓ +0.009 [-0.029, +0.058] | ✓ +1 ms [-1, +3] | ✓ +0.0 pp [+0.0, +0.0] |
| gut (Wiederholung) | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.000] | ✓ +0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| Büro-WLAN | Modell | ✓ -8 pp [-24, +10] | ✓ +0.008 [-0.033, +0.062] | ✓ -1 ms [-4, +2] | ✓ +0.0 pp [+0.0, +0.0] |
| Büro-WLAN | Modell episodengenau | ✗ -20 pp [-34, -6] | ✓ +0.015 [-0.018, +0.065] | ✗ +3 ms [+0, +6] | ✓ +0.0 pp [+0.0, +0.0] |
| Büro-WLAN | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [+0.000, +0.000] | ✓ -0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| schlecht | Modell | ✓ +6 pp [-10, +20] | ✓ +0.024 [-0.026, +0.077] | ✓ -2 ms [-7, +2] | ✓ +0.0 pp [-0.0, +0.1] |
| schlecht | Modell episodengenau | ✓ +2 pp [-12, +18] | ✓ +0.038 [-0.011, +0.090] | ✓ +1 ms [-2, +4] | ✓ +0.0 pp [-0.0, +0.1] |
| schlecht | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.000] | ✓ -0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| sehr schlecht | Modell | ✓ +0 pp [-18, +18] | ✓ +0.022 [-0.042, +0.093] | ✓ -1 ms [-12, +9] | ✓ -0.1 pp [-0.7, +0.5] |
| sehr schlecht | Modell episodengenau | ✓ +2 pp [-18, +22] | ✓ -0.006 [-0.075, +0.060] | ✓ +7 ms [-2, +17] | ✓ +0.3 pp [-0.2, +0.9] |
| sehr schlecht | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [+0.000, +0.000] | ✓ +0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| schlecht o. Stall | Modell | ✓ +4 pp [-16, +24] | ✓ -0.014 [-0.091, +0.063] | ✗ +14 ms [+10, +17] | ✓ +0.0 pp [+0.0, +0.0] |
| schlecht o. Stall | Modell episodengenau | ✓ +2 pp [-18, +20] | ✓ +0.032 [-0.035, +0.103] | ✓ +1 ms [-2, +4] | ✓ +0.0 pp [+0.0, +0.0] |
| schlecht o. Stall | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [+0.000, +0.000] | ✓ -0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| sehr schlecht o. Stall | Modell | ✓ -16 pp [-32, +0] | ✓ +0.019 [-0.037, +0.082] | ✗ +17 ms [+14, +20] | ✓ +0.0 pp [-0.1, +0.1] |
| sehr schlecht o. Stall | Modell episodengenau | ✓ -4 pp [-24, +16] | ✓ +0.003 [-0.063, +0.083] | ✗ +4 ms [+1, +6] | ✓ +0.0 pp [+0.0, +0.1] |
| sehr schlecht o. Stall | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.000] | ✓ +0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |

**Profil ddim10** (DDIM-10):

| Szenario | Messung | Erfolg [95-%-KI] | Ø Max-Reward [KI] | Planalter [KI] | Starvation [KI] |
|---|---|---:|---:|---:|---:|
| **Referenz (sync)** | Echtzeit (Phase 1) | 58 % [44–71] | 0.934 [0.891–0.970] | 0.348 s [0.347–0.348] | 0.0 % [0.0–0.0] |
|  | Simulationszeit: Modell (gepoolt) | 58 % [44–71] | 0.934 [0.891–0.970] | 0.348 s [0.347–0.348] | 0.0 % [0.0–0.0] |
| **gut** | Echtzeit (Phase 1) | 20 % [11–33] | 0.793 [0.704–0.871] | 1.173 s [1.166–1.179] | 18.8 % [18.0–19.7] |
|  | Simulationszeit: Modell (gepoolt) | 18 % [10–31] | 0.759 [0.676–0.832] | 1.173 s [1.169–1.177] | 18.9 % [18.4–19.4] |
|  | Simulationszeit: Modell (episodengenau) | 12 % [6–24] | 0.780 [0.689–0.860] | 1.175 s [1.168–1.182] | 19.1 % [18.1–20.0] |
|  | Simulationszeit: Replay | 20 % [11–33] | 0.793 [0.704–0.871] | 1.173 s [1.166–1.179] | 18.8 % [18.0–19.7] |
| **Büro-WLAN** | Echtzeit (Phase 1) | 22 % [13–35] | 0.760 [0.674–0.841] | 1.207 s [1.200–1.213] | 23.6 % [22.8–24.5] |
|  | Simulationszeit: Modell (gepoolt) | 18 % [10–31] | 0.707 [0.614–0.788] | 1.211 s [1.207–1.214] | 24.1 % [23.6–24.6] |
|  | Simulationszeit: Modell (episodengenau) | 16 % [8–29] | 0.770 [0.678–0.850] | 1.206 s [1.198–1.213] | 23.4 % [22.5–24.4] |
|  | Simulationszeit: Replay | 22 % [13–35] | 0.760 [0.674–0.841] | 1.207 s [1.200–1.213] | 23.6 % [22.8–24.5] |
| **schlecht** | Echtzeit (Phase 1) | 6 % [2–16] | 0.651 [0.545–0.744] | 1.307 s [1.301–1.313] | 36.1 % [35.4–36.7] |
|  | Simulationszeit: Modell (gepoolt) | 8 % [3–19] | 0.679 [0.581–0.769] | 1.307 s [1.302–1.311] | 36.1 % [35.6–36.6] |
|  | Simulationszeit: Modell (episodengenau) | 14 % [7–26] | 0.661 [0.562–0.749] | 1.310 s [1.303–1.317] | 36.5 % [35.7–37.2] |
|  | Simulationszeit: Replay | 6 % [2–16] | 0.651 [0.545–0.744] | 1.307 s [1.301–1.313] | 36.1 % [35.4–36.7] |
| **sehr schlecht** | Echtzeit (Phase 1) | 4 % [1–13] | 0.591 [0.501–0.680] | 1.584 s [1.565–1.604] | 54.0 % [53.2–54.8] |
|  | Simulationszeit: Modell (gepoolt) | 10 % [4–21] | 0.592 [0.493–0.687] | 1.590 s [1.573–1.607] | 53.7 % [53.1–54.4] |
|  | Simulationszeit: Modell (episodengenau) | 0 % [0–7] | 0.623 [0.528–0.713] | 1.597 s [1.578–1.617] | 54.2 % [53.4–55.0] |
|  | Simulationszeit: Replay | 4 % [1–13] | 0.591 [0.501–0.680] | 1.584 s [1.565–1.604] | 54.0 % [53.2–54.8] |

| Szenario | Quelle | Δ Erfolg [KI] | Δ Ø Max-Reward [KI] | Δ Planalter [KI] | Δ Starvation [KI] |
|---|---|---:|---:|---:|---:|
| Referenz (sync) | Modell | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.001] | ✓ +0 ms [+0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| gut | Modell | ✓ -2 pp [-16, +14] | ✓ -0.035 [-0.107, +0.031] | ✓ +0 ms [-8, +8] | ✓ +0.1 pp [-1.0, +1.1] |
| gut | Modell episodengenau | ✓ -8 pp [-22, +6] | ✓ -0.013 [-0.086, +0.055] | ✓ +2 ms [-2, +6] | ✓ +0.2 pp [-0.3, +0.7] |
| gut | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.000] | ✗ +0 ms [+0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| Büro-WLAN | Modell | ✓ -4 pp [-18, +12] | ✓ -0.053 [-0.151, +0.045] | ✓ +4 ms [-4, +12] | ✓ +0.5 pp [-0.5, +1.5] |
| Büro-WLAN | Modell episodengenau | ✓ -6 pp [-20, +8] | ✓ +0.009 [-0.087, +0.103] | ✓ -1 ms [-6, +4] | ✓ -0.2 pp [-0.9, +0.4] |
| Büro-WLAN | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.000] | ✓ +0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| schlecht | Modell | ✓ +2 pp [-4, +8] | ✓ +0.028 [-0.039, +0.094] | ✓ -0 ms [-7, +6] | ✓ +0.0 pp [-0.8, +0.8] |
| schlecht | Modell episodengenau | ✓ +8 pp [-2, +18] | ✓ +0.010 [-0.073, +0.095] | ✓ +3 ms [-2, +9] | ✓ +0.4 pp [-0.2, +1.0] |
| schlecht | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.000] | ✓ +0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |
| sehr schlecht | Modell | ✓ +6 pp [-4, +16] | ✓ +0.000 [-0.063, +0.063] | ✓ +5 ms [-22, +31] | ✓ -0.2 pp [-1.4, +0.8] |
| sehr schlecht | Modell episodengenau | ✓ -4 pp [-10, +0] | ✓ +0.032 [-0.048, +0.111] | ✓ +12 ms [-2, +28] | ✓ +0.2 pp [-0.3, +0.7] |
| sehr schlecht | Replay | ✓ +0 pp [+0, +0] | ✓ +0.000 [-0.000, +0.000] | ✓ +0 ms [-0, +0] | ✓ +0.0 pp [+0.0, +0.0] |

**Drift der Messmaschine in Phase 1:** Serverzeit je Anfrage in ms, als Median je Lauf und Mittel je Block
aus zehn Seeds. Diese Drift ist der Grund für die episodengenaue Quelle.

| Lauf | Median | Seeds 1000–1009 | Seeds 1010–1019 | Seeds 1020–1029 | Seeds 1030–1039 | Seeds 1040–1049 |
|---|---:|---:|---:|---:|---:|---:|
| gut | 340 | 353 | 344 | 344 | 342 | 344 |
| gut (Wiederholung) | 340 | 346 | 351 | 346 | 341 | 344 |
| Büro-WLAN | 341 | 351 | 350 | 342 | 344 | 338 |
| schlecht | 339 | 347 | 341 | 341 | 342 | 340 |
| sehr schlecht | 338 | 355 | 341 | 339 | 338 | 342 |
| schlecht o. Stall | 325 | 327 | 329 | 328 | 327 | 326 |
| sehr schlecht o. Stall | 324 | 326 | 329 | 329 | 327 | 327 |

| Lauf | Median | Seeds 1000–1009 | Seeds 1010–1019 | Seeds 1020–1029 | Seeds 1030–1039 | Seeds 1040–1049 |
|---|---:|---:|---:|---:|---:|---:|
| gut | 777 | 795 | 784 | 775 | 778 | 778 |
| Büro-WLAN | 778 | 791 | 787 | 774 | 780 | 773 |
| schlecht | 782 | 791 | 789 | 782 | 790 | 779 |
| sehr schlecht | 782 | 795 | 789 | 781 | 785 | 774 |

**Latenzverteilungen** (gepooltes Modell gegen Echtzeit):

| Profil | Szenario | E2E p50 Echtzeit / Modell | E2E p95 Echtzeit / Modell | Netzanteil Ø Echtzeit / Modell |
|---|---|---:|---:|---:|
| kalibriert | gut | 344 ms / 339 ms | 405 ms / 394 ms | 3.5 / 3.4 ms |
| kalibriert | Büro-WLAN | 366 ms / 361 ms | 419 ms / 414 ms | 25.4 / 25.4 ms |
| kalibriert | schlecht | 428 ms / 424 ms | 548 ms / 564 ms | 97.2 / 98.3 ms |
| kalibriert | sehr schlecht | 503 ms / 500 ms | 1307 ms / 1308 ms | 227.8 / 227.7 ms |
| kalibriert | schlecht o. Stall | 412 ms / 422 ms | 456 ms / 482 ms | 85.4 / 85.1 ms |
| kalibriert | sehr schlecht o. Stall | 482 ms / 494 ms | 548 ms / 569 ms | 155.3 / 155.2 ms |
| ddim10 | gut | 781 ms / 784 ms | 857 ms / 857 ms | 3.3 / 3.4 ms |
| ddim10 | Büro-WLAN | 805 ms / 805 ms | 872 ms / 875 ms | 25.4 / 25.5 ms |
| ddim10 | schlecht | 870 ms / 869 ms | 965 ms / 960 ms | 91.6 / 92.8 ms |
| ddim10 | sehr schlecht | 948 ms / 946 ms | 1766 ms / 1845 ms | 230.7 / 234.8 ms |

![Latenz](phase3a_latency.png)

**Replay-Treue je Lauf:**

| Profil | Szenario | Episoden | bitgleich | gleich bis auf letzte Float-Bits | abweichend | Fallbacks |
|---|---|---:|---:|---:|---:|---:|
| kalibriert | gut | 50 | 37 | 13 | 0 | 0 |
| kalibriert | gut (Wiederholung) | 50 | 44 | 6 | 0 | 0 |
| kalibriert | Büro-WLAN | 50 | 41 | 9 | 0 | 0 |
| kalibriert | schlecht | 50 | 41 | 9 | 0 | 0 |
| kalibriert | sehr schlecht | 50 | 40 | 10 | 0 | 0 |
| kalibriert | schlecht o. Stall | 50 | 41 | 9 | 0 | 0 |
| kalibriert | sehr schlecht o. Stall | 50 | 35 | 15 | 0 | 0 |
| ddim10 | gut | 50 | 39 | 11 | 0 | 0 |
| ddim10 | Büro-WLAN | 50 | 38 | 12 | 0 | 0 |
| ddim10 | schlecht | 50 | 42 | 8 | 0 | 0 |
| ddim10 | sehr schlecht | 50 | 43 | 7 | 0 | 0 |
| **Summe** | | **550** | **441** | **109** | **0** | **0** |

### Determinismus

Wiederholte Ausführungen derselben Läufe, mit anderer Worker-Zahl, anderer Reihenfolge der Seeds und frischen
Prozessen. „Bitgleich“ heißt: Erfolg, Länge, alle Takt-Serien, Abdeckung und Reward sind identisch.

| Vergleich | Profil | Lauf | Episoden | bitgleich | gleich bis auf letzte Float-Bits | abweichend | Anfragen identisch |
|---|---|---|---:|---:|---:|---:|---:|
| 1. gegen 2. Ausführung (4 statt 1 Worker; Code-Stand der Validierung gegen aktuellen) | kalibriert | modell__gut | 10 | 6 | 4 | 0 | 248 / 248 |
| 1. gegen 2. Ausführung (4 statt 1 Worker; Code-Stand der Validierung gegen aktuellen) | kalibriert | modell__sehr_schlecht | 10 | 8 | 2 | 0 | 318 / 318 |
| 2. gegen 3. Ausführung (1 statt 2 Worker, gleicher Code) | kalibriert | modell__gut | 10 | 7 | 3 | 0 | 248 / 248 |
| 2. gegen 3. Ausführung (1 statt 2 Worker, gleicher Code) | kalibriert | modell__sehr_schlecht | 10 | 7 | 3 | 0 | 318 / 318 |
| SmolVLA/Meta-World: 1 gegen 2 Worker (gleicher Code) | smolvla | langsam__sehr_schlecht | 3 | 3 | 0 | 0 | 11 / 11 |
| SmolVLA/Meta-World: 1 gegen 2 Worker (gleicher Code) | smolvla | mittel__gut | 3 | 3 | 0 | 0 | 12 / 12 |

- **Anfragen:** Alle 1 155 wiederholten Anfragen sind bitgleich, ebenso alle Takt-Serien und Ausgänge.
- **Meta-World:** Die Episoden sind vollständig bitgleich (MuJoCo). Der SHA-256-Digest über alle Records ist
  für beide Ausführungen identisch (`python -m kortyn.run_virtual --digest …`):
  - `mittel__gut`: `db8bec2f…0ac1929e`
  - `langsam__sehr_schlecht`: `23d8b7e4…eccb5edd`
- **PushT:** Bei 3 von 10 Episoden weicht die Abdeckung in der letzten Stelle ab. Ursache ist die Physik von
  PushT selbst, nicht der Simulationszeit-Modus:
  - `python -m kortyn.env_determinism` führt eine feste, kontaktreiche Aktionsfolge in mehreren
    Env-Instanzen und Prozessen aus; die Abdeckungen weichen um bis zu 2·10⁻¹⁶ ab.
  - Mit einem skriptierten Schieber, der weniger Kontakte erzeugt, bleiben sie identisch.
- **Code-Stände:** Die ersten beiden Zeilen vergleichen zusätzlich zwei Code-Stände, denn die
  Validierungsläufe stammen vom Stand vor der SmolVLA-Integration. Auch dort ist alles bitgleich bis auf
  dieselben letzten Bits.

---

## 3. Restrauschen, Power und Seed-Zahl

### Woher die Streuung kommt

- **Deterministisch je Seed:** Dieselbe Konfiguration liefert dieselbe Episode. Ein A/A-Vergleich mit
  identischem Timing-Seed hat die Differenz exakt null.
- **Prozessrauschen:** Was zwischen zwei *Bedingungen* streut, kommt aus der Aufgabe, nicht aus der Messung.
  Andere Latenzen lassen die Kontakt-Trajektorien auseinanderlaufen, und ab da entscheidet die Policy anders.
  Echtzeit hat dieses Rauschen auch, dazu Maschinenrauschen (Inferenzzeit-Schwankung, Takt-Jitter, Drift),
  das sich nicht wiederholen lässt.
- **Common Random Numbers:** Jede Zufallsgröße kommt aus einem eigenen Strom (Seed, Komponente,
  Beobachtungsschritt).
  - Zwei Bedingungen auf demselben Seed teilen alle Ziehungen, in denen sie übereinstimmen (Servicezeit,
    Client-Overhead). Ihr Unterschied ist dann *nur* der Unterschied der Bedingungen.
  - Das senkt die Varianz gepaarter Vergleiche dort, wo die Bedingungen nahe beieinander liegen.

### Messung

| Vergleich (gleiche Seeds) | n | Erfolg A / B | diskordante Paare | SD der Reward-Differenz | identische Episoden |
|---|---:|---:|---:|---:|---:|
| Echtzeit A/A: gut vs. gut (Wiederholung), kalibriert | 50 | 62 % / 60 % | 26 % | 0.149 | 9 |
| Simulationszeit A/A, neue Latenzen: gut, kalibriert | 50 | 60 % / 60 % | 32 % | 0.099 | 7 |
| Simulationszeit A/A, neue Latenzen: sehr schlecht, kalibriert | 50 | 42 % / 50 % | 48 % | 0.235 | 2 |
| Simulationszeit A/A, neue Latenzen: gut, ddim10 | 50 | 18 % / 22 % | 28 % | 0.258 | 0 |
| Echtzeit: gut vs. Büro-WLAN, kalibriert | 50 | 62 % / 70 % | 32 % | 0.167 | 3 |
| Simulationszeit: gut vs. Büro-WLAN, kalibriert | 50 | 60 % / 62 % | 26 % | 0.068 | 9 |
| Echtzeit: Referenz (sync) vs. gut, kalibriert | 50 | 60 % / 62 % | 42 % | 0.132 | 0 |
| Simulationszeit: Referenz (sync) vs. gut, kalibriert | 50 | 58 % / 60 % | 38 % | 0.146 | 0 |
| Echtzeit: gut vs. sehr schlecht, kalibriert | 50 | 62 % / 42 % | 32 % | 0.270 | 0 |
| Simulationszeit: gut vs. sehr schlecht, kalibriert | 50 | 60 % / 42 % | 46 % | 0.259 | 0 |

Lesart:
- **Gleiche Bedingung, neue Latenz-Ziehung (A/A):** Simulationszeit „gut“, kalibriert: 32 % diskordante
  Paare, SD 0,099. Echtzeit: 26 %, SD 0,149. Das ist dieselbe Größenordnung; das Prozessrauschen dominiert,
  und beim Reward kommt in Echtzeit die Maschinenstreuung dazu.
- **Nahe Bedingungen** (gut gegen Büro-WLAN, 20 ms RTT Unterschied): Simulationszeit SD 0,068, Echtzeit
  0,167.
  - Bei gleicher Präzision braucht die Simulation also ≈ 6-mal weniger Seeds (Varianzverhältnis
    (0,167/0,068)² ≈ 6).
  - Hier wirken die Common Random Numbers voll.
- **Ferne Bedingungen** (gut gegen sehr schlecht): 0,259 gegen 0,270. Kein Gewinn, weil die Trajektorien
  ohnehin auseinanderlaufen.
- **Bedingung gegen Sync-Referenz** (der Vergleich, den Schritt 3 für jede Latenzbedingung macht): 38 %
  diskordant, SD 0,146. Das ist der Planungswert.

### Power-Analyse und Seed-Zahl

MDE (kleinster nachweisbarer Effekt) = kleinster Effekt, den ein gepaarter Test mit 80 % Power bei α = 5 %
(zweiseitig) erkennt:
- **Erfolg:** McNemars Stichprobenformel (Connor 1987) mit dem gemessenen Anteil diskordanter Paare.
- **Reward:** gepaarte Normalapproximation mit der gemessenen SD der Differenzen.

| Rauschmodell | diskordant | SD Reward-Diff. | MDE Erfolg (n = 50 | n = 100 | n = 200) | MDE Ø Max-Reward (n = 50 | n = 100 | n = 200) |
|---|---:|---:|---|---|
| Echtzeit (Phase 1) (identische Bedingung zweimal gemessen) | 26 % | 0.149 | 20 % / 14 % / 10 % | 0.059 / 0.042 / 0.029 |
| Simulationszeit, Latenzen neu gezogen (Mittel der A/A-Replikationen) | 36 % | 0.209 | 23 % / 17 % / 12 % | 0.083 / 0.059 / 0.041 |
| Simulationszeit, Bedingung vs. Sync-Referenz (Planungswert für Schritt 3 (Referenz vs. gut)) | 38 % | 0.146 | 24 % / 17 % / 12 % | 0.058 / 0.041 / 0.029 |

![Power](phase3a_power.png)

**Entscheidung: 100 Seeds je Bedingung für Schritt 3.**
- Bei 50 Seeds erkennt der Vergleich gegen die Sync-Referenz erst Erfolgsverluste ab 24 pp, bei 100 Seeds ab
  17 pp, bei 200 Seeds ab 12 pp.
- 200 Seeds hätten die Rechenzeit verdoppelt: SmolVLA war vor dem Start auf 4,7–12,7 h geschätzt, mit 200
  Seeds also 9–25 h. Das hätte die 10-Stunden-Grenze gerissen.
- 17 pp sind kleiner als die Effekte, um die es geht:
  - In Phase 1 kostete „sehr schlecht“ 20 pp.
  - Die skriptierten Meta-World-Experten brechen bei dynamischen Aufgaben schon bei 100 ms Totzeit um 30
    bis über 90 pp ein.
- Kleinere Effekte weist Schritt 3 nicht nach, sondern grenzt sie nach oben ab (Konfidenzintervalle in den
  Tabellen).

### Realisiert in Schritt 3

SmolVLA streut weniger als PushT, weil viele Seeds in jeder Bedingung gleich enden. Mit 100 Seeds erkennt der
Vergleich gegen Sync-25 Erfolgsunterschiede ab 15 pp (push) bzw. 10 pp (coffee-push).

| Aufgabe | Bedingungen | Seeds | diskordante Paare Ø [min–max] | SD Reward-Diff. Ø | MDE Erfolg | MDE Ø Max-Reward |
|---|---:|---:|---:|---:|---:|---:|
| drawer-close-v3 | 12 | 100 | 0 % [0 %–0 %] | 0.00 | – | 0.00 |
| push-v3 | 12 | 100 | 31 % [18 %–49 %] | 3.91 | 15 % | 1.09 |
| coffee-push-v3 | 12 | 100 | 14 % [12 %–16 %] | 1.51 | 10 % | 0.42 |

---

## 4. SmolVLA auf Meta-World

### Recherche: Modell, Checkpoint, Referenzwerte

| Frage | Befund | Quelle |
|---|---|---|
| Welches Modell? | SmolVLA 0,45 B (SmolVLM2-500M, 16 VLM-Schichten, Flow-Matching-Action-Expert), Chunk 50 Aktionen, 10 Flow-Schritte | SmolVLA-Paper (arXiv 2506.01844) |
| Welcher Checkpoint? | `lerobot/smolvla_metaworld`, Revision `8d82bdf3…` gepinnt; MT50 (alle 50 Aufgaben), Fine-Tuning von `smolvla_base` mit 5 000 Schritten (Batch 32). Eingaben: ein Kamerabild 480 × 480, Zustand 4-D (Endeffektor xyz + Greifer), Sprachanweisung. Ausgabe 4-D (Endeffektor-Verschiebung + Greifer) | Model Card / `config.json` |
| Wie evaluiert LeRobot? | `metaworld==3.0.0` (v3-Aufgaben), 80 Hz, max. 500 Schritte, Erfolg beendet die Episode, Kamera `corner2`, Seeds ab 1000; der Agent führt alle 50 Aktionen eines Chunks aus | LeRobot `envs/metaworld.py`, `lerobot-eval` |
| Publizierte Erfolgsquoten *dieses* Checkpoints | je Aufgabe, 30 Episoden, von einer Heatmap abgelesen: drawer-close 100 %, push 70 %, coffee-push 90 % | Early-Exits-Paper (arXiv 2609.29382v2, Abb. 4) |
| Publizierte Werte des SmolVLA-Papers | eigenes Modell (100 k Schritte), Neuplanung nach jeder Aktion, 10 Versuche je Aufgabe: easy 82,5 %, medium 41,8 %, hard 45,0 %, very hard 60,0 % | SmolVLA-Paper, Tab. 2 |
| Inferenzzeit je Chunk (10 Schritte) | RTX 4090 99,2 ms (fp32), A100 278,8 ms, Apple M1 MPS 721,8 ms, M1 CPU 2 028 ms | LeRobot-Paper (arXiv 2602.22818, Tab. 3) |
| | RTX 3090 eager PyTorch 174,1 ms (torch.compile 65,9 ms), Apple M4 Metal P90 330,6 ms, i9-14900HX CPU 2 141 ms | vla.cpp (arXiv 2606.08094v2) |

Der Wert des SmolVLA-Papers taugt für diesen Checkpoint nur als Größenordnung:
- anderes Training (100 k statt 5 k Schritte);
- anderes Ausführungsprotokoll (1 statt 50 Aktionen je Plan);
- nur 10 Versuche je Aufgabe.

Der Abgleich der Sync-Referenz läuft deshalb gegen die Werte je Aufgabe aus dem Early-Exits-Paper. Diese
stammen vom selben Checkpoint, sind aber von einer Heatmap abgelesen, und das Ausführungsprotokoll ist dort
nicht angegeben.

### Aufgabenwahl: Wie viel Verzögerung verträgt eine Aufgabe?

Die Auswahl beruht auf einer Policy-unabhängigen Messung (`kortyn/expert_delay.py`, 30 Seeds je Stufe):
- Der skriptierte Meta-World-Experte jeder Aufgabe bekommt seine Beobachtung *k* Takte verspätet und handelt
  trotzdem sofort (Closed-Loop mit Totzeit).
- Gemessen wird, ab welcher Totzeit die Aufgabe kippt.
- Eine Aufgabe gilt als **dynamisch**, wenn schon kleine Totzeiten den Erfolg kosten: Objekte bewegen sich nach
  dem Kontakt weiter, oder der Greifer überschießt.

| Aufgabe | 0 ms | 50 ms | 100 ms | 200 ms | 300 ms | 500 ms |
|---|---:|---:|---:|---:|---:|---:|
| **drawer-close-v3** | 100 % | 100 % | 100 % | 100 % | 97 % | 83 % |
| hammer-v3 | 100 % | 100 % | 70 % | 10 % | 17 % | 33 % |
| plate-slide-v3 | 100 % | 100 % | 100 % | 13 % | 0 % | 0 % |
| **push-v3** | 100 % | 93 % | 97 % | 3 % | 3 % | 0 % |
| door-open-v3 | 93 % | 90 % | 83 % | 3 % | 0 % | 3 % |
| pick-place-v3 | 100 % | 90 % | 27 % | 27 % | 13 % | 10 % |
| **coffee-push-v3** | 100 % | 73 % | 43 % | 0 % | 3 % | 0 % |
| stick-push-v3 | 100 % | 97 % | 10 % | 7 % | 3 % | 0 % |
| soccer-v3 | 97 % | 50 % | 13 % | 13 % | 23 % | 7 % |
| sweep-into-v3 | 87 % | 43 % | 10 % | 10 % | 10 % | 10 % |
| basketball-v3 | 97 % | 33 % | 3 % | 0 % | 0 % | 0 % |
| button-press-v3 | 100 % | 0 % | 0 % | 0 % | 0 % | 0 % |

Gewählt sind drei Aufgaben (fett) mit verschiedener Schwierigkeit und Totzeit-Toleranz:
- **drawer-close** (easy) ist statisch: Der Experte schafft noch 83 % bei 500 ms Totzeit. Das ist die
  Kontrollaufgabe.
- **push** (hard) kippt zwischen 100 und 200 ms (97 % → 3 %).
- **coffee-push** (medium) ist die dynamischste der drei: Schon 50 ms kosten 27 pp, bei 100 ms bleiben 43 %.
  Der Checkpoint schafft sie laut Referenz trotzdem zu 90 %, sie ist also messbar.

Nicht gewählt:
- **dynamische Aufgaben, die der Checkpoint schon ohne Latenz selten löst** (publiziert: basketball 10 %,
  soccer 17 %, sweep-into 23 %). Dort bliebe kaum Erfolg, dessen Verlust man messen könnte.
- **button-press:** Der Experte fällt schon bei 50 ms auf 0 %, obwohl sich der Knopf nicht von selbst bewegt.
  Das spricht für eine Eigenheit des skriptierten Reglers, nicht für eine dynamische Aufgabe.
- **stick-push** (very hard, 70 % publiziert, Experte 97 % → 10 % zwischen 50 und 100 ms) wäre eine vierte
  Kandidatin gewesen. Sie fiel dem Laufzeitbudget zum Opfer: Eine Aufgabe kostet 1,5–4 h Rechenzeit.

### Integration

- **Brain:**
  - SmolVLA läuft mit LeRobots eigenen Klassen und Pre-/Postprozessoren (unverändert, kein Fork).
  - Die Sprachanweisung reist im Protokoll mit (`InferRequest.task`); der Tokenizer läuft im Brain.
  - Das Rauschen des Flow-Matchings kommt wie bei der Diffusion Policy aus einem je Anfrage geseedeten
    CPU-Generator.
  - Ein Golden-Test (`tests/test_smolvla.py`) prüft, dass der Plan bei gleichem Anfangsrauschen bitgleich zu
    LeRobots Evaluationspfad ist (`preprocess_observation` → Präprozessor → `predict_action_chunk` →
    Postprozessor).
- **Umgebung:** `kortyn/metaworld_env.py` baut die Umgebung exakt wie LeRobots `MetaworldEnv` (Aufgabe, Kamera,
  Seeds, Bildorientierung).
  - Ein Test vergleicht Schritt für Schritt mit LeRobots Umgebung: Bilder, Zustände, Rewards und Erfolg sind
    identisch.
  - Gerendert wird nur, wenn eine Anfrage ein Bild braucht; die Physik läuft jeden Takt.
- **Agent:** derselbe Simulationszeit-Agent wie für PushT, mit Chunk 50 und Refill-Schwelle 24.
  - Solange die Latenz unter 312,5 ms bleibt, geht alle 25 Takte eine Anfrage raus (LeRobots Async-Inferenz:
    Schwelle 50 %).
  - Ohne Plan steht der Arm still: Strategie `halten`, also Verschiebung null, und der Greifer hält seinen
    Befehl. Begründung in Abschnitt 8.
- **Sync-Referenzen:**
  - **Sync-50** führt alle 50 Aktionen eines Plans aus, wie LeRobots Evaluation. Das ist der Abgleich mit
    publizierten Werten.
  - **Sync-25** fragt mit denselben Beobachtungsschritten an wie der Echtzeit-Agent (alle 25 Takte), wartet
    aber auf jeden Plan. Das ist die gepaarte Referenz, gegen die jede Latenzbedingung gemessen wird. Die
    Differenz zu Sync-25 ist damit reiner Latenzeffekt.
- **Bildübertragung:** wählbar je Lauf (`agent.image_transport`).
  - Codecs: roh (Standard, wie LeRobots Evaluation), JPEG oder PNG mit einstellbarer Qualität, dazu optional
    Herunterskalieren beim Roboter.
  - Das Brain dekodiert und skaliert auf die Eingangsgröße der Policy zurück.
  - Gemessen werden Nutzlast je Anfrage und Bandbreite je Roboter.
- **GPU-Profile:** Die Inferenz rechnet auf der CPU dieser Maschine; *wann* der Plan ankommt, bestimmt das
  Hardware-Profil.
  - Ankunft = publizierte Inferenzzeit (lognormal, 5 % Streuung angenommen) + 1 ms Client-Overhead + 5 ms
    Transport-Overhead + Netz-Szenario.
  - **schnell:** RTX 4090 (99 ms).
  - **mittel:** RTX 3090 (174 ms).
  - **langsam:** Apple M4 (311 ms Mittel, aus dem publizierten P90 von 330,6 ms).
  - **Grenzfall:** Apple M1 (722 ms, länger als ein ganzer Chunk von 625 ms).

### Sync-Referenz gegen publizierte Werte und LeRobots Evaluation

| Aufgabe | Gruppe | Kortyn Sync-50 (LeRobot-Protokoll) | Kortyn Sync-25 | publiziert (dieser Checkpoint) | Δ Sync-50 − publiziert [KI] | Sync-50, Seeds 1000–1029 | Δ auf diesen Seeds [KI] |
|---|---|---:|---:|---:|---:|---:|---:|
| drawer-close-v3 | easy | 100 % [96–100] | 100 % [96–100] | 100 % (n = 30) | ✓ +0 pp [-4, +11] | 30/30 (100 %) | ✓ +0 pp [-11, +11] |
| push-v3 | hard | 57 % [47–66] | 69 % [59–77] | 70 % (n = 30) | ✓ -13 pp [-30, +7] | 23/30 (77 %) | ✓ +7 pp [-15, +28] |
| coffee-push-v3 | medium | 60 % [50–69] | 66 % [56–75] | 90 % (n = 30) | ✗ -30 pp [-42, -12] | 22/30 (73 %) | ✓ -17 pp [-36, +3] |

| Aufgabe | Seeds | lerobot-eval (diese Maschine) | Kortyn Sync-50 (gleiche Seeds) | gleicher Ausgang | nur lerobot-eval / nur Kortyn | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| coffee-push-v3 | 1000–1049 | 60 % (30/50) | 66 % (33/50) | 43/50 | 2 / 5 | 0,453 |

- **drawer-close** passt exakt (100 %).
- **push** passt: Über 100 Seeds liegt Sync-50 bei 57 % (Δ −13 pp [−30, +7]), auf den Seeds 1000–1029 bei
  77 % gegen 70 % publiziert.
- **coffee-push** passt über 100 Seeds nicht: 60 % gegen 90 % (Δ −30 pp [−42, −12]).

Der coffee-push-Abstand liegt nicht an Kortyn. Belege:
1. **LeRobots eigene Evaluation** (`lerobot-eval`, gleiche Maschine, Seeds 1000–1049) kommt auf **60 %**,
   Kortyns Sync-50 auf denselben Seeds auf 66 %.
   - 43 von 50 Seeds enden gleich, die übrigen 7 verteilen sich 2 : 5 (McNemar p = 0,45).
   - Abweichungen sind erwartbar: LeRobot zieht das Flow-Rauschen aus dem globalen Zufallsgenerator, Kortyn
     je Anfrage geseedet.
2. **Auf den Seeds einer Standard-Evaluation mit 30 Episoden** (1000–1029) liegt Kortyn bei 73 %. Die
   Differenz zu 90 % ist mit −17 pp [−36, +3] nicht signifikant.
   - Die späteren Seeds sind bei coffee-push schwerer (1030–1099: 54 %).
   - Die Startkonfigurationen streuen stark, und 30 Episoden sind eine kleine Stichprobe davon.
3. **Ausgeschlossen:**
   - Sprachanweisung: identisch mit den Trainingsdaten (`lerobot/metaworld_mt50`, Stand vor dem Training).
   - Kamera-Slots: `empty_cameras = 0` wie im Training.
   - Ausführungsprotokoll: 50 Aktionen je Chunk wie in `config.json` und `train_config.json`.
   - Präzision: Die Gewichte sind wie bei LeRobot üblich bf16, auf GPU und CPU gleich; `use_amp` ist aus.
   - Seeding: Vor LeRobot-PR #3727 (27.07.2026) waren die Objektpositionen ungeseedet, aber gleich verteilt.
4. **Offen** bleibt, mit welchem Protokoll die Referenz gemessen hat.
   - Das Early-Exits-Paper nennt 30 Episoden je Aufgabe, 10 Flow-Schritte und eine A100, aber weder Seeds noch
     Aktionen je Plan noch Versionen.
   - coffee-push streut schon zwischen den Varianten des Papers stark (77–93 % bei kleinen Rechenänderungen).

**Fazit:** Kortyns SmolVLA-Pipeline reproduziert LeRobots eigene Evaluation. Die publizierten Werte erreicht
sie auf vergleichbaren Seeds innerhalb der Konfidenzintervalle. Maßstab für alle Latenz-Vergleiche ist die
eigene, gepaarte Sync-Referenz auf 100 Seeds, nicht der publizierte Wert.

**Sync-25 gegen Sync-50:** Neuplanung alle 25 statt 50 Takte ist bei push um 12 pp besser (p = 0,05), bei
coffee-push um 6 pp (p = 0,18). Das passt zur Ablation im SmolVLA-Paper: Weniger ausgeführte Aktionen je Chunk
erhöhen den Erfolg (LIBERO: 50 → 30 Aktionen, +19 pp).

### Echte Inferenzzeit auf dieser Maschine

| Policy | Sampler | CPU | Threads | Median | p90 | Minimum |
|---|---|---|---:|---:|---:|---:|
| smolvla | flow-10 | Intel(R) Xeon(R) Processor @ 2.10GHz | 1 | 1744 ms | 1849 ms | 1674 ms |
| smolvla | flow-10 | Intel(R) Xeon(R) Processor @ 2.10GHz | 2 | 1178 ms | 1291 ms | 1034 ms |
| smolvla | flow-10 | Intel(R) Xeon(R) Processor @ 2.10GHz | 4 | 907 ms | 1085 ms | 787 ms |

- **Messung:** SmolVLA (bf16, 10 Flow-Schritte) braucht auf dieser CPU 1,7 s mit einem Thread und 0,9 s mit
  vier.
- **Einordnung:** Das liegt in der Größenordnung publizierter CPU-Werte (M1 CPU 2,0 s, i9-14900HX 2,1 s).
- **Folge:** Ein reines CPU-Brain läge über der Dauer eines ganzen Chunks (625 ms) und damit im
  Grenzfall-Regime, in dem jeder Plan nach dem ersten veraltet ankommt.
- **Deshalb GPU-Profile:** Die Experimente simulieren GPU-Profile. Gerechnet wird trotzdem auf der CPU; nur
  der Ankunftszeitpunkt des Plans folgt dem Profil.

---

## 5. Ergebnisse je Aufgabe, Netz-Szenario und GPU-Profil

**Legende der Tabellen:**
- 100 Seeds je Zeile, gepaart zur Sync-25-Referenz.
- **E2E** = Beobachtung bis Plan beim Roboter.
- **Planalter** = Ø Alter der ausgeführten Aktionen.
- **Starvation** = Anteil der Takte ohne Aktion (der Arm steht still).
- **Bandbreite** = Anfragen und Antworten je Roboter bei Rohbildern.

**drawer-close** (easy, statisch):

| Bedingung | Erfolg [95-%-KI] | Δ vs. Sync-25 [KI] (p) | Ø Max-Reward | E2E p50 / p95 | Planalter | Starvation | Bandbreite/Roboter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sync, 50 Aktionen/Plan (LeRobot-Eval) | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | – | 0.26 s | – | 11.2 Mbit/s |
| Sync, 25 Aktionen/Plan (Referenz) | 100 % [96–100] | – | 10.00 | – | 0.14 s | – | 21.9 Mbit/s |
| RTX 4090 (99 ms), gut | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 105 ms / 114 ms | 0.23 s | 0.0 % | 16.8 Mbit/s |
| RTX 4090 (99 ms), Büro-WLAN | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 125 ms / 137 ms | 0.24 s | 0.0 % | 16.7 Mbit/s |
| RTX 4090 (99 ms), schlecht | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 187 ms / 225 ms | 0.29 s | 0.8 % | 16.4 Mbit/s |
| RTX 4090 (99 ms), sehr schlecht | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 256 ms / 373 ms | 0.33 s | 9.2 % | 15.2 Mbit/s |
| RTX 3090 (174 ms), gut | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 181 ms / 196 ms | 0.28 s | 0.0 % | 16.7 Mbit/s |
| RTX 3090 (174 ms), Büro-WLAN | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 201 ms / 218 ms | 0.29 s | 0.0 % | 16.6 Mbit/s |
| RTX 3090 (174 ms), schlecht | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 261 ms / 313 ms | 0.33 s | 2.0 % | 16.1 Mbit/s |
| RTX 3090 (174 ms), sehr schlecht | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 332 ms / 937 ms | 0.36 s | 17.3 % | 13.5 Mbit/s |
| Apple M4 (311 ms), gut | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 317 ms / 346 ms | 0.36 s | 4.0 % | 15.6 Mbit/s |
| Apple M4 (311 ms), Büro-WLAN | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 338 ms / 364 ms | 0.36 s | 9.3 % | 14.5 Mbit/s |
| Apple M4 (311 ms), schlecht | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 399 ms / 446 ms | 0.37 s | 25.2 % | 12.7 Mbit/s |
| Apple M4 (311 ms), sehr schlecht | 100 % [96–100] | +0 pp [+0, +0] (1,000) | 10.00 | 467 ms / 1233 ms | 0.38 s | 48.0 % | 10.3 Mbit/s |
| Apple M1 (722 ms), gut | 0 % [0–4] | -100 pp [-100, -100] (< 0,001) | 0.51 | 730 ms / 789 ms | 0.31 s | 90.0 % | 7.7 Mbit/s |

**push** (hard):

| Bedingung | Erfolg [95-%-KI] | Δ vs. Sync-25 [KI] (p) | Ø Max-Reward | E2E p50 / p95 | Planalter | Starvation | Bandbreite/Roboter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sync, 50 Aktionen/Plan (LeRobot-Eval) | 57 % [47–66] | -12 pp [-23, -1] (0,050) | 6.75 | – | 0.30 s | – | 9.5 Mbit/s |
| Sync, 25 Aktionen/Plan (Referenz) | 69 % [59–77] | – | 7.43 | – | 0.15 s | – | 18.6 Mbit/s |
| RTX 4090 (99 ms), gut | 63 % [53–72] | -6 pp [-15, +2] (0,238) | 7.05 | 105 ms / 114 ms | 0.25 s | 0.0 % | 16.4 Mbit/s |
| RTX 4090 (99 ms), Büro-WLAN | 62 % [52–71] | -7 pp [-16, +2] (0,210) | 6.76 | 125 ms / 136 ms | 0.27 s | 0.0 % | 16.4 Mbit/s |
| RTX 4090 (99 ms), schlecht | 59 % [49–68] | -10 pp [-19, +0] (0,064) | 6.67 | 186 ms / 221 ms | 0.33 s | 1.6 % | 16.4 Mbit/s |
| RTX 4090 (99 ms), sehr schlecht | 54 % [44–63] | -15 pp [-26, -4] (0,014) | 6.92 | 258 ms / 326 ms | 0.39 s | 10.4 % | 15.3 Mbit/s |
| RTX 3090 (174 ms), gut | 61 % [51–70] | -8 pp [-17, +1] (0,134) | 6.94 | 180 ms / 195 ms | 0.32 s | 0.0 % | 16.4 Mbit/s |
| RTX 3090 (174 ms), Büro-WLAN | 61 % [51–70] | -8 pp [-17, +1] (0,152) | 7.02 | 200 ms / 216 ms | 0.34 s | 0.0 % | 16.4 Mbit/s |
| RTX 3090 (174 ms), schlecht | 50 % [40–60] | -19 pp [-29, -9] (0,001) | 6.56 | 261 ms / 299 ms | 0.40 s | 3.1 % | 16.5 Mbit/s |
| RTX 3090 (174 ms), sehr schlecht | 43 % [34–53] | -26 pp [-36, -16] (< 0,001) | 6.30 | 333 ms / 404 ms | 0.44 s | 23.3 % | 14.2 Mbit/s |
| Apple M4 (311 ms), gut | 46 % [37–56] | -23 pp [-34, -12] (< 0,001) | 6.49 | 317 ms / 345 ms | 0.44 s | 6.6 % | 16.4 Mbit/s |
| Apple M4 (311 ms), Büro-WLAN | 41 % [32–51] | -28 pp [-39, -17] (< 0,001) | 6.28 | 337 ms / 362 ms | 0.45 s | 15.7 % | 15.7 Mbit/s |
| Apple M4 (311 ms), schlecht | 37 % [28–47] | -32 pp [-42, -21] (< 0,001) | 5.99 | 398 ms / 446 ms | 0.46 s | 40.9 % | 13.1 Mbit/s |
| Apple M4 (311 ms), sehr schlecht | 26 % [18–35] | -43 pp [-54, -32] (< 0,001) | 5.55 | 469 ms / 1129 ms | 0.46 s | 63.4 % | 10.6 Mbit/s |
| Apple M1 (722 ms), gut | 7 % [3–14] | -62 pp [-71, -52] (< 0,001) | 4.54 | 730 ms / 789 ms | 0.31 s | 89.6 % | 7.7 Mbit/s |

**coffee-push** (medium, laut Experten-Test die dynamischste der drei Aufgaben):

| Bedingung | Erfolg [95-%-KI] | Δ vs. Sync-25 [KI] (p) | Ø Max-Reward | E2E p50 / p95 | Planalter | Starvation | Bandbreite/Roboter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sync, 50 Aktionen/Plan (LeRobot-Eval) | 60 % [50–69] | -6 pp [-13, +1] (0,180) | 2.90 | – | 0.30 s | – | 9.8 Mbit/s |
| Sync, 25 Aktionen/Plan (Referenz) | 66 % [56–75] | – | 3.18 | – | 0.15 s | – | 18.7 Mbit/s |
| RTX 4090 (99 ms), gut | 65 % [55–74] | -1 pp [-8, +6] (1,000) | 2.90 | 105 ms / 113 ms | 0.25 s | 0.0 % | 16.6 Mbit/s |
| RTX 4090 (99 ms), Büro-WLAN | 67 % [57–75] | +1 pp [-6, +8] (1,000) | 2.89 | 125 ms / 136 ms | 0.27 s | 0.0 % | 16.6 Mbit/s |
| RTX 4090 (99 ms), schlecht | 67 % [57–75] | +1 pp [-6, +7] (1,000) | 2.91 | 185 ms / 219 ms | 0.32 s | 1.4 % | 16.6 Mbit/s |
| RTX 4090 (99 ms), sehr schlecht | 60 % [50–69] | -6 pp [-13, +0] (0,146) | 2.88 | 256 ms / 326 ms | 0.38 s | 10.5 % | 15.5 Mbit/s |
| RTX 3090 (174 ms), gut | 65 % [55–74] | -1 pp [-9, +6] (1,000) | 2.93 | 180 ms / 194 ms | 0.31 s | 0.0 % | 16.6 Mbit/s |
| RTX 3090 (174 ms), Büro-WLAN | 69 % [59–77] | +3 pp [-4, +10] (0,581) | 2.93 | 200 ms / 215 ms | 0.33 s | 0.0 % | 16.6 Mbit/s |
| RTX 3090 (174 ms), schlecht | 61 % [51–70] | -5 pp [-12, +2] (0,267) | 2.89 | 260 ms / 302 ms | 0.39 s | 3.1 % | 16.6 Mbit/s |
| RTX 3090 (174 ms), sehr schlecht | 59 % [49–68] | -7 pp [-14, +0] (0,118) | 2.79 | 332 ms / 405 ms | 0.42 s | 22.4 % | 14.2 Mbit/s |
| Apple M4 (311 ms), gut | 59 % [49–68] | -7 pp [-14, +0] (0,118) | 2.82 | 319 ms / 348 ms | 0.43 s | 6.9 % | 16.4 Mbit/s |
| Apple M4 (311 ms), Büro-WLAN | 58 % [48–67] | -8 pp [-15, -1] (0,077) | 2.79 | 338 ms / 363 ms | 0.43 s | 15.2 % | 15.7 Mbit/s |
| Apple M4 (311 ms), schlecht | 59 % [49–68] | -7 pp [-14, +0] (0,092) | 2.74 | 398 ms / 447 ms | 0.44 s | 38.2 % | 13.1 Mbit/s |
| Apple M4 (311 ms), sehr schlecht | 58 % [48–67] | -8 pp [-15, -1] (0,057) | 2.71 | 469 ms / 1207 ms | 0.43 s | 58.1 % | 10.6 Mbit/s |
| Apple M1 (722 ms), gut | 26 % [18–35] | -40 pp [-50, -30] (< 0,001) | 2.41 | 731 ms / 790 ms | 0.31 s | 87.1 % | 7.6 Mbit/s |

### Bildübertragung (push, Sync-25)

| Bildübertragung | KB pro Anfrage | Bandbreite/Roboter bei 3,2 Anfragen/s | Erfolg [KI] | Δ vs. roh [KI] (p) |
|---|---:|---:|---:|---:|
| roh, 480×480 (LeRobot-Eval) | 691.4 | 17.7 Mbit/s | 76 % [63–86] | – |
| JPEG q75, 240×240 (im Brain auf 480×480 skaliert) | 7.7 | 0.2 Mbit/s | 56 % [42–69] | -20 pp [-36, -4] (0,031) |
| JPEG q90, 480×480 | 41.3 | 1.1 Mbit/s | 72 % [58–83] | -4 pp [-12, +4] (0,625) |

- **JPEG q90:** spart 94 % der Nutzlast, ohne signifikanten Erfolgsverlust (−4 pp [−12, +4]). Bei zehn
  Robotern sind das 11 statt 177 Mbit/s.
- **240 × 240 Pixel + JPEG q75:** kostet signifikant Erfolg (−20 pp). Die Policy braucht Bilddetails, die beim
  Herunterskalieren verloren gehen.
- **Stichprobe:** n = 50 Seeds; ein Verlust bis 12 pp ist für q90 nicht ausgeschlossen.

---

## 6. Schadet Latenz bei SmolVLA und dynamischen Aufgaben mehr?

![Latenzwirkung](phase3a_latency_effect.png)

**So ist gerechnet:**
- Mittlerer gepaarter Erfolgsverlust gegen die eigene synchrone Referenz, gemittelt über die Bedingungen einer
  Aufgabe, ohne den M1-Grenzfall. Das KI kommt aus einem Bootstrap über Seeds.
- **Budget** = Latenz, die die Action-Queue ohne Starvation überbrückt: Refill-Schwelle + 1 Takt. Bei PushT
  sind das 7 × 100 ms = 700 ms, bei SmolVLA 25 × 12,5 ms = 312,5 ms.
- Eine Bedingung zählt als „über Budget“, wenn ihre mediane Latenz darüber liegt.

| Aufgabe | Bedingungen | Anzahl | Seeds | Sync-Referenz | Ø Δ Erfolg gegen Sync [95-%-KI] |
|---|---|---:|---:|---:|---:|
| PushT kalibriert | alle Bedingungen | 4 | 50 | 58 % | ✓ -4 pp [-17, +9] |
| PushT kalibriert | Latenz im Budget | 4 | 50 | 58 % | ✓ -4 pp [-17, +9] |
| PushT ddim10 | alle Bedingungen | 4 | 50 | 58 % | ✗ -44 pp [-57, -31] |
| PushT ddim10 | Latenz über Budget | 4 | 50 | 58 % | ✗ -44 pp [-57, -31] |
| drawer-close-v3 | alle Bedingungen | 12 | 100 | 100 % | ✓ +0 pp [+0, +0] |
| drawer-close-v3 | Latenz im Budget | 7 | 100 | 100 % | ✓ +0 pp [+0, +0] |
| drawer-close-v3 | Latenz über Budget | 5 | 100 | 100 % | ✓ +0 pp [+0, +0] |
| push-v3 | alle Bedingungen | 12 | 100 | 69 % | ✗ -19 pp [-26, -11] |
| push-v3 | Latenz im Budget | 7 | 100 | 69 % | ✗ -10 pp [-18, -3] |
| push-v3 | Latenz über Budget | 5 | 100 | 69 % | ✗ -30 pp [-39, -21] |
| coffee-push-v3 | alle Bedingungen | 12 | 100 | 66 % | ✓ -4 pp [-10, +2] |
| coffee-push-v3 | Latenz im Budget | 7 | 100 | 66 % | ✓ -1 pp [-8, +5] |
| coffee-push-v3 | Latenz über Budget | 5 | 100 | 66 % | ✗ -7 pp [-14, -0] |

| Unterschied der Verluste (zweite minus erste Aufgabe) | Bedingungen | Δ [95-%-KI] |
|---|---|---:|
| coffee-push-v3 − drawer-close-v3 | alle Bedingungen | ✓ -4 pp [-10, +3] |
| push-v3 − drawer-close-v3 | alle Bedingungen | ✗ -19 pp [-26, -11] |
| coffee-push-v3 − push-v3 | alle Bedingungen | ✗ +15 pp [+5, +25] |
| coffee-push-v3 − drawer-close-v3 | Latenz im Budget | ✓ -1 pp [-8, +5] |
| push-v3 − drawer-close-v3 | Latenz im Budget | ✗ -10 pp [-18, -3] |
| coffee-push-v3 − push-v3 | Latenz im Budget | ✓ +9 pp [-1, +20] |
| coffee-push-v3 − drawer-close-v3 | Latenz über Budget | ✗ -7 pp [-14, -0] |
| push-v3 − drawer-close-v3 | Latenz über Budget | ✗ -30 pp [-40, -21] |
| coffee-push-v3 − push-v3 | Latenz über Budget | ✗ +23 pp [+12, +35] |

### SmolVLA gegen PushT

- **Relativ zum Budget verliert SmolVLA nicht mehr als die Diffusion Policy:**
  - Im Budget: PushT kalibriert −4 pp [−17, +9]; SmolVLA push −10 pp [−18, −3], coffee-push −1 pp,
    drawer-close ±0.
  - Über dem Budget: PushT ddim10 −44 pp [−57, −31]; SmolVLA push −30 pp [−39, −21], coffee-push −7 pp
    [−14, −0], drawer-close ±0.
  - push bewegt sich in derselben Größenordnung wie PushT, die anderen beiden Aufgaben deutlich darunter.
- **In absoluter Zeit verliert SmolVLA mehr, weil das Budget kleiner ist:**
  - PushT plant mit 10 Hz und 15 Aktionen je Plan; die Reserve von 7 Takten überbrückt 700 ms.
  - SmolVLA steuert mit 80 Hz und 50 Aktionen je Plan; die Reserve von 25 Takten überbrückt nur 312,5 ms.
  - Latenzen um 320–470 ms, die PushT kalibriert noch im Budget hält (+4 bis −16 pp bei 340–500 ms), liegen
    bei SmolVLA darüber und kosten push 23–43 pp.
  - Schon eine Apple-M4-GPU (311 ms Inferenz) sprengt das Budget ohne jedes Netz.
- **Die Mechanik ist dieselbe wie bei PushT:** Ein Plan, der nach L Takten ankommt, hat nur noch 50 − L
  nutzbare Aktionen, weil veraltete Aktionen verworfen werden.
  - Über dem Budget fällt die Queue in jedem Zyklus leer; im Gleichgewicht steht der Arm (2L − 50)/L der Zeit
    still.
  - Bei M4 + sehr schlecht sind das 48–63 %, beim M1-Grenzfall (L > 50 Takte) 87–90 %.

### Der Vorbehalt: Meta-World-Episoden entscheiden sich in der ersten Sekunde

| | PushT (Phase 1, kalibriert) | drawer-close | push | coffee-push |
|---|---:|---:|---:|---:|
| Takt | 10 Hz | 80 Hz | 80 Hz | 80 Hz |
| Erfolg im Median nach (synchrone Referenz) | 157 Takte (15,7 s) | 79 Takte (0,99 s) | 61 Takte (0,76 s) | 54 Takte (0,68 s) |
| davon deckt der latenzfreie Start-Plan ab | 15 Takte (10 %) | 50 Takte (63 %) | 50 Takte (82 %) | 50 Takte (93 %) |
| neue Pläne (mit Latenz) bis dahin | ≈ 19 | 2 | 1–2 | 1 |

- **Start-Plan:** Der erste Plan entsteht wie in Phase 1 vor Episodenbeginn (Warmstart) und ist in allen
  Bedingungen identisch und latenzfrei.
- **Kurze Erfolge, keine Erholung:** Bei Meta-World erledigt dieser Plan den größten Teil der Arbeit. Die
  Latenz wirkt nur auf die ein bis zwei Pläne danach, also auf das letzte Stück bis zum Erfolg. Misserfolge
  laufen bei allen drei Aufgaben bis zum Zeitlimit von 500 Takten (1 501 von 1 501 gescheiterten Episoden):
  Wer im ersten Anlauf scheitert, erholt sich nicht mehr.
- **Folge für den Vergleich:** Die Budget-relativen Zahlen oben vergleichen daher ungleiche Dinge.
  - Bei PushT wirkt Latenz auf rund 19 Pläne je Erfolg, bei SmolVLA auf ein bis zwei.
  - Dass push trotzdem bis zu 43 pp verliert, spricht je betroffenem Plan für eine *höhere* Empfindlichkeit
    als bei PushT.
- **Stalls:** Der erste Stall beginnt zufällig in den ersten 10 s einer Episode. Vor dem Median-Erfolg
  (0,7–1,0 s) liegt er nur in rund 7–10 % der Episoden. Für Stalls ist diese Baseline daher wenig empfindlich.
- **Offene Frage:** Wie stark Latenz SmolVLA im Dauerbetrieb schadet, beantwortet diese Baseline nicht. Dafür
  braucht es Aufgaben mit längerem Horizont (Abschnitt 7).

### Statisch gegen dynamisch

Die Rangfolge der Policy ist eine andere als die der skriptierten Experten:

| Aufgabe | Experte: Totzeit bis zum Einbruch | SmolVLA: Ø Verlust über Budget | M1-Grenzfall: Erfolg nur mit dem Start-Plan |
|---|---|---:|---:|
| drawer-close | robust bis 300 ms (97 %) | ±0 pp | 0 % (von 100 %) |
| push | robust bis 100 ms, bei 200 ms 3 % | −30 pp | 7 % (von 69 %) |
| coffee-push | −27 pp schon bei 50 ms | −7 pp | 26 % (von 66 %) |

Der Unterschied coffee-push − push über dem Budget ist mit +23 pp [+12, +35] signifikant, aber in der
Gegenrichtung der Experten-Erwartung. Zwei Eigenschaften der Aufgabe erklären das besser als ihre Dynamik:

1. **Wie viel der Start-Plan allein löst.** Im M1-Grenzfall ist der Start-Plan der einzige nutzbare Plan, denn
   jeder spätere kommt vollständig veraltet an.
   - coffee-push gelingt dann noch in 26 % der Seeds (39 % der synchronen Erfolge), push in 7 % (10 %),
     drawer-close nie.
   - Bei coffee-push ist ein großer Teil der Arbeit getan, bevor Latenz überhaupt wirkt.
2. **Wie präzise die folgenden Pläne sein müssen.**
   - drawer-close braucht den zweiten Plan (Median 79 Takte bis zum Erfolg), aber nicht dessen Präzision:
     Jede Schließbewegung reicht, auch ein veralteter Plan schiebt in die richtige Richtung. Deshalb scheitert
     drawer-close im M1-Grenzfall immer und verliert sonst nie.
   - Bei push muss der Puck auf einen Zielpunkt geführt werden. Ein veralteter Plan korrigiert die letzten
     Zentimeter zu spät.

### Planalter oder Starvation?

Tabelle je Bedingung im Anhang.
- **coffee-push:** Der Verlust folgt dem Planalter.
  - Bis 0,33 s Ø-Planalter gibt es keinen Verlust (−1 bis +3 pp).
  - Ab 0,38 s fehlen 5–8 pp, egal ob der Arm 3 % oder 58 % der Zeit stillsteht.
- **push:** Das Planalter wirkt schon im Budget.
  - −6 bis −19 pp bei 0,25–0,40 s Planalter, gegenüber 0,15 s in der Sync-25-Referenz.
  - Dasselbe zeigt die Referenz ohne Latenz: Sync-50 hat 0,30 s Planalter und verliert 12 pp.
  - Anders als PushT in Phase 2 profitiert SmolVLA-push also von frischeren Plänen. Dort brachten weniger
    Aktionen je Plan keinen Gewinn.
- **push über dem Budget:** Zusätzlich wirkt die Starvation.
  - Bei fast gleichem Planalter (0,44–0,46 s) wächst der Verlust mit der Starvation von −23 pp (7 % Stillstand)
    auf −43 pp (63 %).
  - Erfolgreiche Episoden dauern dann länger (M4 + sehr schlecht: Median 139 statt 61 Takte), bleiben aber
    weit unter dem Zeitlimit. Der Stillstand kostet also nicht die Zeit, sondern die Qualität der ausgeführten
    Aktionen.
  - Über dem Budget führt der Arm von jedem Plan nur noch das Ende aus (bei L = 38 Takten die Aktionen 38–49
    von 50), also die am weitesten vorausschauenden und unsichersten Vorhersagen, und steht dazwischen still.

---

## 7. Konsequenzen für Phase 3 (mehrere Roboter an einem Brain)

1. **Ziel ist das Latenzbudget je Roboter, nicht die mittlere Latenz.**
   - Für SmolVLA bei 80 Hz sind es 312,5 ms für Netz, Warteschlange und Inferenz zusammen.
   - Der `VirtualBrain` hat für andere Scheduler bereits die austauschbare Schnittstelle.
   - Bei gleichartigen Robotern ist EDF (Earliest Deadline First) gleich FIFO. Gewinn entsteht erst bei
     unterschiedlichen Budgets oder Netzen, mit der Deadline = Anfragezeit + Budget − erwarteter Rückweg.
   - Genau das sollte Phase 3 testen: Roboter mit gutem und schlechtem Netz an einem Brain.
2. **Kapazität: zwei Roboter je RTX 4090 ohne Batching** (Rechnung, keine Messung).
   - Ein Roboter fragt 3,2-mal pro Sekunde an; bei 99 ms Inferenz ist das Brain mit zwei Robotern zu 63 %
     ausgelastet, mit drei zu 95 %.
   - **Pessimistisch** (zufällige Ankunft, M/D/1): Zwei Roboter warten im Mittel ≈ 86 ms, Ende-zu-Ende also
     ≈ 190 ms plus Netz, im Budget bis zum Szenario „schlecht“. Drei Roboter warten ≈ 1 s.
   - **Periodisch** (die Anfragen kommen tatsächlich im festen 25-Takt-Raster): Selbst bei gleichzeitigen
     Anfragen wartet der dritte Roboter nur 198 ms, landet mit Netz aber am Budget. Drei Roboter gehen also nur
     mit gestaffelten Anfragen, und das ist eine Scheduler-Aufgabe.
   - Eine RTX 3090 trägt einen Roboter (56 % Auslastung, zwei überlasten sie). Ein Apple M4 liegt schon mit
     einem Roboter über dem Budget.
   - **Batching über Roboter** ist der Hebel für mehr. Dafür fehlen publizierte Batch-Zeiten von SmolVLA;
     Phase 3 braucht eine GPU-Messung oder eine ausgewiesene Annahme.
3. **Frische kostet Kapazität, und bei präzisen Aufgaben lohnt sie sich.**
   - push gewinnt durch Neuplanung alle 25 statt 50 Takte 12 pp. Häufigere Anfragen erhöhen aber die
     Brain-Last je Roboter.
   - Das ist der Zielkonflikt aus Phase 2, mit anderem Ausgang: Bei PushT lohnte Frische nicht, bei
     SmolVLA-push schon.
   - Ein Phase-3-Scheduler sollte die Anfragefrequenz je Aufgabe wählen können.
   - push gehört als empfindlicher Benchmark in jede Phase-3-Messung, drawer-close dient als Negativkontrolle.
4. **Aufgaben mit längerem Horizont aufnehmen.** Meta-World-Episoden entscheiden sich in der ersten Sekunde,
   der Start-Plan verdeckt dort den Großteil der Latenzwirkung. Optionen:

   | Option | Nutzen | Kosten |
   |---|---|---|
   | LIBERO-Long mit `lerobot/smolvla_libero` | lange, mehrstufige Aufgaben, publizierte Referenz | nur unter Linux paketiert, nicht auf dem Mac reproduzierbar |
   | verkettete Meta-World-Aufgaben | gleicher Checkpoint und Stack, läuft überall | eigenes Protokoll, keine publizierte Referenz |
   | Kaltstart statt Warmstart | erster Plan trägt ebenfalls Latenz | misst eher Anlaufverhalten als Dauerbetrieb |
5. **Bilder komprimieren:** JPEG q90 als Standard senkt die Last auf 1,1 Mbit/s je Roboter (zehn Roboter: 11
   statt 177 Mbit/s), ohne signifikanten Verlust. Stärkeres Herunterskalieren kostet Erfolg.
6. **Keine reine CPU-Inferenz für VLAs dieser Größe:** 0,9–1,7 s je Plan liegen jenseits eines ganzen Chunks.
7. **Methodik übernehmen:**
   - Phase 3 läuft in Simulationszeit mit 100 Seeds je Bedingung und gemeinsamen Zufallszahlen.
   - Hardware-Profile werden, wo es echte Messungen gibt, episoden- oder sitzungsgenau kalibriert, damit
     Hardware-Drift nicht als Effekt erscheint.

---

## 8. Einschränkungen: was der Simulationszeit-Modus und diese Baseline nicht abbilden

**Rechenzeit und Hardware**
- **Unabhängige Servicezeiten:** Sie werden je Anfrage unabhängig gezogen. Echte Inferenzzeiten sind
  autokorreliert (Takt, Thermik, Hintergrundlast) und driften zwischen Sitzungen (Phase 1: 339 → 324 ms).
  Bursts langsamer Inferenzen und Drift bildet der Modus nur ab, wenn man sie als Profil anlegt.
- **GPU-Profile sind publizierte Werte**, ergänzt um eine angenommene Streuung von 5 %. Sie stammen aus
  verschiedenen Laufzeiten (PyTorch eager, vla.cpp), Präzisionen (fp32, bf16) und Bildgrößen. Es sind
  Größenordnungen, keine Messung auf eigener Hardware.
- **Plan-Inhalte von der CPU:** Die Pläne stammen von SmolVLA in bf16 auf der CPU. Auf einer GPU (andere
  Kernels) unterscheiden sie sich in den letzten Bits; statistisch gleichwertig, aber nicht bitgleich.
- **Keine Rechenkonkurrenz** zwischen Agent, Simulation und Brain: Simulation und Rendering kosten in
  virtueller Zeit nichts. Auf einem echten Roboter-Rechner verzögern sie den Takt.

**Netz**
- **Proxy-Modell aus Phase 1:** Delay und Jitter je Nachricht, periodische Stalls, Reihenfolgetreue.
  Paketverlust, TCP-Retransmits und Staukontrolle fehlen, ebenso geteilte Bandbreite zwischen Robotern und
  WLAN-Airtime.
- **Bandbreite unbegrenzt:** In den vier Szenarien beeinflusst die Nutzlast die Latenz nicht. Die Bandbreite
  je Roboter wird gemessen, aber nicht als Engpass simuliert.
- **Keine Fragmentierung:** Ein Request ist eine Nachricht. Phase 1 zeigte ≈ 1,2 Lesevorgänge je Nachricht;
  die Aufteilung auf mehrere Bursts mit eigenem Delay ist nicht modelliert.

**Takt und Robotik**
- **Idealer Takt:** keine Verspätung, außer im Replay oder mit einem Takt-Jitter-Profil.
- **Kein Aktuator-Modell:** Die Aktion wirkt ab dem Takt, in dem sie ausgeführt wird; zwischen Agent und
  Motoren gibt es keine Totzeit.
- **Starvation heißt in beiden Umgebungen stillstehen.**
  - Bei PushT wiederholt der Agent wie in Phase 1 das letzte absolute Positionsziel.
  - Bei Meta-World sendet er eine Null-Verschiebung, und der Greifer behält seinen Befehl (Strategie
    `halten`). Das tut ein Roboter-Treiber ohne neue Kommandos, und das bewirkt auch LeRobots Async-Client bei
    leerer Queue.
  - Die Phase-1-Regel „letzte Aktion wiederholen“ hätte bei Verschiebungen den Arm weiterfahren lassen. Das
    wäre eine andere, härtere Starvation, die den Vergleich mit PushT verzerrt.
- **Veraltete Aktionen fallen weg** (Ausrichtung nach Zeitschritt, wie LeRobots Async-Inferenz).
  - Ein Plan, der nach L Takten ankommt, hat nur noch 50 − L nutzbare Aktionen.
  - Ab L > 25 Takten (312,5 ms) steigt die Starvation deshalb steil, im Gleichgewicht auf (2L − 50)/L.
  - Ab L ≥ 50 (625 ms) ist jeder Plan bei Ankunft vollständig veraltet.
  - Andere Ausrichtungen (veraltete Aktionen trotzdem ausführen, Phase 2: `shift`) sind nicht gemessen.
- **Warmstart und kurzer Horizont:** Der erste Plan entsteht vor Episodenbeginn (wie in Phase 1).
  - Bei Meta-World deckt er 50 der 54–79 Takte bis zum Median-Erfolg ab; Latenz wirkt nur auf ein bis zwei
    Pläne (Abschnitt 6).
  - Die Baseline unterschätzt daher die Latenzwirkung im Dauerbetrieb, und Stalls fallen nur in 7–10 % der
    Episoden in die entscheidende Phase.

**Aufgaben und Policy**
- **Eng gefasst:** drei Aufgaben aus einem Benchmark, ein Checkpoint (MT50, 5 000 Trainingsschritte), eine
  Kamera, ein Roboter. Ein anders trainierter Checkpoint kann anders auf das Planalter reagieren.
- **Schwache Referenz:** Die publizierte Referenz beruht auf 30 Episoden je Aufgabe, von einer Heatmap
  abgelesen, mit unbekanntem Protokoll.

**Statistik**
- **Prozessrauschen bestimmt die Power:** Ergebnisse sind je Seed deterministisch. Die verbleibende Streuung
  zwischen Bedingungen ist Prozessrauschen: Kleine Timing-Unterschiede lassen Kontakt-Trajektorien
  auseinanderlaufen. Das ist eine Eigenschaft der Aufgabe, nicht der Messung.
- **Geltungsbereich:** Ein Simulationszeit-Ergebnis gilt für das gewählte Latenzmodell. Die Validierung zeigt
  Übereinstimmung mit der Phase-1-Maschine, nicht mit jeder Hardware.

---

## 9. Laufzeit

Gemessen auf der 4-vCPU-Maschine. Ein Worker ist ein Prozess mit einem Torch-Thread; der Replay nutzt drei
Threads wie das Brain in Phase 1.

| Schritt | Episoden | Laufzeit |
|---|---:|---:|
| Validierung, gepooltes Modell (kalibriert + ddim10, einschließlich A/A-Wiederholungen) | 750 | 47 + 94 min (4 Worker) |
| Validierung, episodengenaues Modell | 550 | 94 min (4 Worker) |
| Validierung, Replay | 550 | 133 min (1 Worker, 3 Threads) |
| Determinismus (PushT 2 × 20, SmolVLA 2 × 6) | 52 | 18 min (parallel zu anderen Läufen) |
| `lerobot-eval` coffee-push | 50 | 45 min (1 Thread) |
| SmolVLA Sync-Referenzen (3 Aufgaben × 2 × 100) | 600 | 117 min (1 Worker) |
| SmolVLA Bildübertragung | 100 | 32 min (1 Worker) |
| SmolVLA Echtzeit-Bedingungen (3 × 13 × 100) | 3 900 | drawer-close 103 min (1 Worker) + 6 min (4 Worker), coffee-push 91 min, push 122 min (je 4 Worker) |
| CPU-Benchmark | – | 2 min |

**Schätzung gegen Ist bei SmolVLA:**
- Vor dem Start waren 4,7–12,7 h geschätzt (realistisch 6–8 h), nach den Sync-Referenzen 5–5,5 h plus
  Puffer.
- Tatsächlich waren es ≈ 4,7 h in Vier-Worker-Äquivalent; der vorbereitete Kürzungsplan (M4 und M1 auf 50
  Seeds) wurde nicht gebraucht.
- Geholfen hat der Inferenz-Cache: Die ersten Anfragen einer Episode sind in allen 13 Bedingungen identisch und
  werden nur einmal gerechnet.

**Verluste:**
- Eine Wartekette hing etwa 1 h, weil ihr Prozessmuster auf ihren eigenen Startbefehl passte. Ketten warten
  jetzt auf Prozess-IDs.
- Ein Lauf brach nach 9 min an der Metadaten-Prüfung ab, als ein neues Konfigurationsfeld dazukam. Die Prüfung
  akzeptiert jetzt explizit nachgetragene Felder mit Altverhalten.

## 10. Abweichungen vom Plan

- **Dritte Validierungsquelle:** Die episodengenaue Servicezeit kam hinzu, nachdem die gepoolten Läufe
  Abweichungen zeigten, die mit der Drift der echten Maschine zusammenfielen. Bestehenskriterium und Quelle
  waren festgelegt, bevor diese Läufe rechneten.
- **Replay-Planalter:** Zwei Replay-Vergleiche sind wörtlich signifikant (+0,05 ms). Bewertet sind sie mit
  einer Toleranz von ±1 ms und hier offen ausgewiesen.
- **Starvation-Verhalten:** Für SmolVLA heißt Starvation Stillstand statt Wiederholung der letzten Aktion, weil
  Meta-World-Aktionen Verschiebungen sind. Festgelegt vor allen SmolVLA-Echtzeit-Läufen.
- **Reihenfolge:**
  - Die SmolVLA-Sync-Referenzen liefen parallel zur Validierung; sie hängen nicht vom Simulationszeit-Modus ab.
  - Die Echtzeit-Experimente starteten erst, nachdem die Modellvalidierung bestanden war.
  - Der Replay lief parallel zu ihnen als zusätzliche Bestätigung und bestand ebenfalls.
- **Zusatzanalyse nach den Läufen:** Die Episodenlänge bis zum Erfolg (Abschnitt 6) war nicht vorab geplant.
  Sie kam hinzu, weil sie die Deutung der Latenzwirkung ändert.

## 11. Reproduktion

```bash
uv sync
# Validierung gegen Phase 1: gepooltes und episodengenaues Modell, Replay (braucht results/kalibriert, results/ddim10)
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/validation_model.yaml
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/validation_replay.yaml
# Determinismus: dieselben Läufe erneut mit anderer Worker-Zahl (PushT, Seeds 1000–1009)
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/validation_model.yaml --profile kalibriert \
    --run modell__gut --run modell__sehr_schlecht --episodes 10 --workers 1 --results-dir results/phase3a/determinism
uv run python -m kortyn.env_determinism          # PushT-Physik allein
# ... und für SmolVLA/Meta-World mit 1 und 2 Workern, danach die Digests vergleichen
for variant in a:1 b:2; do
  uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/smolvla_drawer_close.yaml \
      --run mittel__gut --run langsam__sehr_schlecht --episodes 3 --workers ${variant#*:} \
      --results-dir results/phase3a/smolvla_determinism_${variant%%:*}
done
uv run python -m kortyn.run_virtual --digest results/phase3a/smolvla_determinism_{a,b}/smolvla/mittel__gut
# Aufgabenwahl: Totzeit-Toleranz der Meta-World-Experten
# (MuJoCo rendert ohne Display: Standard EGL unter Linux, CGL unter macOS, überschreibbar mit MUJOCO_GL)
uv run python -m kortyn.expert_delay
# SmolVLA (Checkpoint ≈ 0,9 GB wird beim ersten Lauf geladen)
for task in drawer_close push coffee_push; do
  uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/smolvla_${task}.yaml
done
uv run python -m kortyn.run_virtual --config kortyn/experiments/phase3a/smolvla_transport.yaml
uv run python -m kortyn.lerobot_reference --config kortyn/experiments/phase3a/smolvla_coffee_push.yaml \
    --profile smolvla --episodes 50
uv run python -m kortyn.bench_inference --config kortyn/experiments/phase3a/smolvla_push.yaml
# Tabellen und Abbildungen: reports/phase3a_data.md + reports/phase3a_*.png
uv run python -m kortyn.report_phase3a
```

Hinweise:
- Die Läufe sind fortsetzbar, und `--stop-file <pfad>` beendet einen Lauf geordnet nach dem laufenden Seed.
- Der Digest zweier Ausführungen derselben Konfiguration ist identisch, wenn sie mit derselben Threadzahl
  liefen. PushT kann in den letzten Float-Bits der Abdeckung abweichen (Abschnitt 2).

---

## Anhang: Latenzwirkung je Bedingung

Sortiert nach Aufgabe und medianer Latenz. „relativ zur Sync-Referenz“ = Erfolg der Bedingung geteilt durch den
Erfolg der Referenz auf denselben Seeds.

| Aufgabe | Bedingung | Mediane Latenz | Latenz / Budget | Ø Planalter | Starvation | Erfolg | relativ zur Sync-Referenz | Δ Erfolg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PushT ddim10 | gut | 784 ms | 1.12 | 1.17 s | 18.9 % | 18 % | 31 % | -40 pp |
| PushT ddim10 | Büro-WLAN | 805 ms | 1.15 | 1.21 s | 24.1 % | 18 % | 31 % | -40 pp |
| PushT ddim10 | schlecht | 869 ms | 1.24 | 1.31 s | 36.1 % | 8 % | 14 % | -50 pp |
| PushT ddim10 | sehr schlecht | 946 ms | 1.35 | 1.59 s | 53.7 % | 10 % | 17 % | -48 pp |
| PushT kalibriert | gut | 339 ms | 0.48 | 0.74 s | 0.0 % | 60 % | 103 % | +2 pp |
| PushT kalibriert | Büro-WLAN | 361 ms | 0.52 | 0.75 s | 0.0 % | 62 % | 107 % | +4 pp |
| PushT kalibriert | schlecht | 424 ms | 0.61 | 0.83 s | 0.2 % | 52 % | 90 % | -6 pp |
| PushT kalibriert | sehr schlecht | 500 ms | 0.71 | 1.00 s | 10.7 % | 42 % | 72 % | -16 pp |
| coffee-push-v3 | RTX 4090 (99 ms), gut | 105 ms | 0.34 | 0.25 s | 0.0 % | 65 % | 98 % | -1 pp |
| coffee-push-v3 | RTX 4090 (99 ms), Büro-WLAN | 125 ms | 0.40 | 0.27 s | 0.0 % | 67 % | 102 % | +1 pp |
| coffee-push-v3 | RTX 3090 (174 ms), gut | 180 ms | 0.58 | 0.31 s | 0.0 % | 65 % | 98 % | -1 pp |
| coffee-push-v3 | RTX 4090 (99 ms), schlecht | 185 ms | 0.59 | 0.32 s | 1.4 % | 67 % | 102 % | +1 pp |
| coffee-push-v3 | RTX 3090 (174 ms), Büro-WLAN | 200 ms | 0.64 | 0.33 s | 0.0 % | 69 % | 105 % | +3 pp |
| coffee-push-v3 | RTX 4090 (99 ms), sehr schlecht | 256 ms | 0.82 | 0.38 s | 10.5 % | 60 % | 91 % | -6 pp |
| coffee-push-v3 | RTX 3090 (174 ms), schlecht | 260 ms | 0.83 | 0.39 s | 3.1 % | 61 % | 92 % | -5 pp |
| coffee-push-v3 | Apple M4 (311 ms), gut | 319 ms | 1.02 | 0.43 s | 6.9 % | 59 % | 89 % | -7 pp |
| coffee-push-v3 | RTX 3090 (174 ms), sehr schlecht | 332 ms | 1.06 | 0.42 s | 22.4 % | 59 % | 89 % | -7 pp |
| coffee-push-v3 | Apple M4 (311 ms), Büro-WLAN | 338 ms | 1.08 | 0.43 s | 15.2 % | 58 % | 88 % | -8 pp |
| coffee-push-v3 | Apple M4 (311 ms), schlecht | 398 ms | 1.27 | 0.44 s | 38.2 % | 59 % | 89 % | -7 pp |
| coffee-push-v3 | Apple M4 (311 ms), sehr schlecht | 469 ms | 1.50 | 0.43 s | 58.1 % | 58 % | 88 % | -8 pp |
| coffee-push-v3 | Apple M1 (722 ms), gut | 731 ms | 2.34 | 0.31 s | 87.1 % | 26 % | 39 % | -40 pp |
| drawer-close-v3 | RTX 4090 (99 ms), gut | 105 ms | 0.34 | 0.23 s | 0.0 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | RTX 4090 (99 ms), Büro-WLAN | 125 ms | 0.40 | 0.24 s | 0.0 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | RTX 3090 (174 ms), gut | 181 ms | 0.58 | 0.28 s | 0.0 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | RTX 4090 (99 ms), schlecht | 187 ms | 0.60 | 0.29 s | 0.8 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | RTX 3090 (174 ms), Büro-WLAN | 201 ms | 0.64 | 0.29 s | 0.0 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | RTX 4090 (99 ms), sehr schlecht | 256 ms | 0.82 | 0.33 s | 9.2 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | RTX 3090 (174 ms), schlecht | 261 ms | 0.84 | 0.33 s | 2.0 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | Apple M4 (311 ms), gut | 317 ms | 1.01 | 0.36 s | 4.0 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | RTX 3090 (174 ms), sehr schlecht | 332 ms | 1.06 | 0.36 s | 17.3 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | Apple M4 (311 ms), Büro-WLAN | 338 ms | 1.08 | 0.36 s | 9.3 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | Apple M4 (311 ms), schlecht | 399 ms | 1.28 | 0.37 s | 25.2 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | Apple M4 (311 ms), sehr schlecht | 467 ms | 1.49 | 0.38 s | 48.0 % | 100 % | 100 % | +0 pp |
| drawer-close-v3 | Apple M1 (722 ms), gut | 730 ms | 2.34 | 0.31 s | 90.0 % | 0 % | 0 % | -100 pp |
| push-v3 | RTX 4090 (99 ms), gut | 105 ms | 0.34 | 0.25 s | 0.0 % | 63 % | 91 % | -6 pp |
| push-v3 | RTX 4090 (99 ms), Büro-WLAN | 125 ms | 0.40 | 0.27 s | 0.0 % | 62 % | 90 % | -7 pp |
| push-v3 | RTX 3090 (174 ms), gut | 180 ms | 0.58 | 0.32 s | 0.0 % | 61 % | 88 % | -8 pp |
| push-v3 | RTX 4090 (99 ms), schlecht | 186 ms | 0.59 | 0.33 s | 1.6 % | 59 % | 86 % | -10 pp |
| push-v3 | RTX 3090 (174 ms), Büro-WLAN | 200 ms | 0.64 | 0.34 s | 0.0 % | 61 % | 88 % | -8 pp |
| push-v3 | RTX 4090 (99 ms), sehr schlecht | 258 ms | 0.83 | 0.39 s | 10.4 % | 54 % | 78 % | -15 pp |
| push-v3 | RTX 3090 (174 ms), schlecht | 261 ms | 0.84 | 0.40 s | 3.1 % | 50 % | 72 % | -19 pp |
| push-v3 | Apple M4 (311 ms), gut | 317 ms | 1.01 | 0.44 s | 6.6 % | 46 % | 67 % | -23 pp |
| push-v3 | RTX 3090 (174 ms), sehr schlecht | 333 ms | 1.06 | 0.44 s | 23.3 % | 43 % | 62 % | -26 pp |
| push-v3 | Apple M4 (311 ms), Büro-WLAN | 337 ms | 1.08 | 0.45 s | 15.7 % | 41 % | 59 % | -28 pp |
| push-v3 | Apple M4 (311 ms), schlecht | 398 ms | 1.27 | 0.46 s | 40.9 % | 37 % | 54 % | -32 pp |
| push-v3 | Apple M4 (311 ms), sehr schlecht | 469 ms | 1.50 | 0.46 s | 63.4 % | 26 % | 38 % | -43 pp |
| push-v3 | Apple M1 (722 ms), gut | 730 ms | 2.33 | 0.31 s | 89.6 % | 7 % | 10 % | -62 pp |
