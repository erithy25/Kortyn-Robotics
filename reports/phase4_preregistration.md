# Kortyn Phase 4 – Präregistrierung: Real-Time Chunking (RTC) für SmolVLA

_Committet am 26.10.2026 vor dem ersten neuen Lauf dieser Phase (Rollout oder Flotten-Simulation). Danach
bleibt diese Datei unverändert. Jede Abweichung steht mit Zeitpunkt und Begründung im Report
(`reports/phase4.md`, Abschnitt „Abweichungen“)._

**Was vor diesem Commit schon gelaufen ist:**

- Recherche zu LeRobots RTC und eine informelle CPU-Zeitmessung (nur Rechenzeit, kein Erfolg).
- Auswertungen vorhandener Phase-3-Daten:
  - Kurvenanpassungen
  - Korrelationen zwischen Bedingungen
  - Takt-Analyse der Anfragen
  - Diagnose des Validierungs-Bias
- Die Power-Simulation in Abschnitt 6 (`results/phase4/power.json`).
- Unit-Tests der noch nicht committeten Codec- und RTC-Integration:
  - synthetische Bilder
  - ein Meta-World-Bild mit dem Phase-3a-Checkpoint
  - keine Erfolgsmessung, keine LIBERO-Episode

Kein neuer Rollout. Über den Erfolg mit RTC ist nichts bekannt.

## 0. Frage und Antwortlogik

**Kann RTC SmolVLA so verzögerungstolerant machen, dass ein Brain viele Roboter ohne nennenswerten
Qualitätsverlust bedient?** Die Antwort setzt sich aus vier vorab festgelegten Bausteinen zusammen:

| Baustein | Messgröße | Entscheidung |
|---|---|---|
| 1. Wirkt RTC? | mittlerer Erfolgsgewinn im Budget (H1) | Abschnitt 7.2 |
| 2. Wie weit trägt die Latenztoleranz? | L* ohne und mit RTC, Punkt und konservativ (H2) | Abschnitt 7.1 |
| 3. Wie viele Roboter je Brain? | Kapazität mit 3-pp- und 1-%-Kriterium, je Strategie | Abschnitt 7.3 |
| 4. Lohnt es sich? | Kosten je Roboter bei gleichem Erfolg gegen Jetson-Rechner im Roboter | Abschnitt 8 |

„Nennenswert“ heißt hier, wie vorgegeben: mehr als 3 pp vorhergesagter Erfolgsverlust des schlechtesten
Roboters gegenüber einem einzelnen Roboter.

## 1. Arbeitspunkt (unverändert aus Phase 3)

- **Policy:** `lerobot/smolvla_libero` (Revision `a24f1f3…`), 10 Flow-Schritte, Chunk 50.
- **Aufgaben:** LIBERO-Long, Aufgaben 8 und 0 × Init-States 0–49, also 100 Episoden (Seeds 1000–1099).
- **Takt:** 20 Hz. Alle 25 Takte (1,25 s) eine Anfrage, Latenzbudget 1 250 ms.
- **Start und Halten:** Kaltstart. Ohne Plan steht der Arm still (Strategie „halten“).
- **Kameras:** zwei Kameras mit 360 × 360 Pixeln.
- **Rechnen:** CPU, Simulationszeit.

Alle bisherigen Modi bleiben per Konfiguration reproduzierbar; RTC und die neuen Codecs kommen als neue
Optionen hinzu.

## 2. Befunde aus den Phase-3-Daten, die das Design bestimmen

**2.1 Takt-Regel.** Ein Plan, dessen Beobachtung im Takt k entstand, wird im ersten Takt übernommen, der
nach seiner Ankunft beginnt. Eine Latenz L verzögert ihn also um d(L) = max(1, ⌈L / 50 ms⌉) Takte.
Geprüft an allen Phase-3-Anfragen: `n_stale` ist exakt d (0 ms → 1, 500 ms → 10, 1 000 ms → 20, 1 250 ms →
25). Erfolg hängt deshalb nur über die **wirksame Latenz** x(L) = 50 ms · d(L) von L ab: 0 und 50 ms sind
dieselbe Bedingung, 460 ms wirkt wie 500 ms. Phase 3 hat linear in L interpoliert und war damit zwischen den
Stufen um bis zu einen Takt zu optimistisch. Auf die drei informativen Validierungsfälle angewandt erklärt
das aber nur rund 1 pp des Bias (−7,8 → −6,9 pp, −7,8 → −6,8 pp, −6,7 → −6,0 pp).

**2.2 Kurvenform zwischen 0 und 500 ms ist unbestimmt.** Phase 3 hat dort keine Stufe. Laut
Profil-Likelihood (log-logistische Kurve, Abschnitt 5.2) verträgt Aufgabe 0 nur einen späten, steilen
Abfall (Steilheit h ≥ 2,5; h ≤ 1,2 abgelehnt, p ≤ 0,012). Aufgabe 8 verträgt h von 0,8 bis 5. Damit
liegt das wahre L* ohne RTC irgendwo zwischen 50 und 300 ms. Erst die dichten Stufen klären das.

**2.3 Abhängigkeit der Seeds.** Dieselbe Episode ist über Bedingungen hinweg korreliert:

| Paar | tetrachorische Korrelation |
|---|---|
| Bedingungen nahe 0 ms (sync25, JPEG, fest_0ms) | 0,60–0,75 |
| 0 gegen 500 ms | 0,59 |
| 500 gegen 1 000 ms | 0,42 |
| 0 gegen 1 000 ms | 0,26 |

Die Power-Simulation bildet das nach (Abschnitt 6).

**2.4 Phase-3-Bias, gepoolt.** Phase-3-Regel, drei informative Fälle (a mit 2, e mit 2 und e mit 4
Robotern), 180 Episoden, Vorhersage aus den eigenen gemessenen Latenzen: gemessen − vorhergesagt =
**−7,4 pp, 95-%-KI [−17,3; +2,1]** (Cluster-Bootstrap über Seeds, 10 000 Ziehungen). Mit den Latenzen
aus Stufe 1 sind es −6,5 pp [−15,9; +3,0]. Das KI enthält die Null schon heute. „Bias verschwunden“ wird
deshalb über die Punktschätzung entschieden (Abschnitt 5.4).

## 3. Schritt 1: verlustfreier Bildtransport

**Kandidaten**, alle auf die zwei Kamerabilder je Anfrage (übereinander gestapelt wie bisher):

- PNG (OpenCV), Kompressionsstufe 1 und 6
- WebP verlustfrei (Pillow), `method` 0 und 4
- zstd (zstandard) auf den Rohbytes, Stufe 3 und 9

Referenz ist roh.

**Messmaterial:** Kamerabilder echter LIBERO-Zustände aller 100 Episoden (Aufgaben 8 und 0, Init-States
0–49). Je Episode Reset, dann 200 Takte mit einer festen, aus dem Seed gezogenen Zufallsfolge kleiner
Endeffektor-Bewegungen. Aufgenommen werden die Bilder in den Takten 0, 50, 100, 150 und 200. Das ergibt
500 Anfragen mit je zwei Kameras. Der Arm bewegt sich, die Szene ist dieselbe wie in den Rollouts. Für
die Kompression einzelner Bilder ist das repräsentativ, denn keiner der Codecs nutzt zeitliche
Redundanz.

**Bitgleichheit (Test, statt Erfolgsexperiment):**

1. decode(encode(x)) == x bitweise für alle 1 000 Messbilder, dazu Rauschbilder, einfarbige Bilder und ein
   Kanal.
2. Die Modelleingabe des Brains nach der Vorverarbeitung ist für jeden Kandidaten tensorgleich zur
   Eingabe aus Rohbildern (`torch.equal`).

**Metriken je Kandidat:**

- Bytes je Anfrage (Mittel, p95)
- Kodierzeit (Roboter) und Dekodierzeit (Brain), jeweils Median über alle Anfragen auf einem CPU-Thread
  dieser Maschine
- Bandbreite je Roboter = Bytes × 8 / 1,25 s

**Auswahlregel:** Standard wird der Kandidat mit den kleinsten mittleren Bytes je Anfrage unter denen, deren
Median aus Kodier- plus Dekodierzeit je Anfrage ≤ 25 ms ist (2 % des Budgets). Liegen zwei Kandidaten
weniger als 2 % auseinander, gewinnt der schnellere. Der Standard gilt ab der Umstellung für alle neuen
Läufe. Läufe davor mit roh sind wegen der Bitgleichheit identisch (gleiche Modelleingabe, gleicher
Zufallsstrom).

**Einschränkung:** Ein Roboter-Rechner (etwa ein Raspberry Pi 5) kodiert langsamer als diese Maschine. Die
Zeiten werden als relative Größe berichtet.

## 4. Schritt 2: RTC in Brain und Agent

**Implementierung:** LeRobots RTC (`lerobot.policies.rtc`, Version 0.6.1) bleibt unverändert, kein Fork.
Eingeschaltet wird es über die Konfiguration der geladenen Policy (`rtc_config`,
`init_rtc_processor()`). Die Einstellungen:

| Parameter | Wert | Begründung |
|---|---|---|
| `prefix_attention_schedule` | EXP | LeRobots Empfehlung |
| `max_guidance_weight` | 10,0 | LeRobots Standard und Empfehlung für 10 Flow-Schritte |
| `execution_horizon` | Länge der Überlappung, bei uns 25 | aktuelle Aktion plus 24 noch nicht ausgeführte Aktionen des laufenden Plans. LeRobots typische 8–12 würden ab d ≥ 10 Aktionen ungeführt lassen, die vor Ankunft des neuen Plans schon ausgeführt werden |
| `inference_delay` d | erwartete Verzögerung in Takten | aus dem Latenzschätzer des Agenten (Median der bisherigen Latenzen der Episode, Takt-Regel); bei fester Latenz ab dem zweiten Plan exakt |

Der erste Plan einer Episode hat keinen Vorgänger (Kaltstart) und läuft ohne Führung. Die Agent-Strategie
`rtc` ist die Phase-3-Strategie „halten“ (Anfrage bei ≤ 24 Restaktionen, ohne Plan steht der Arm still)
plus Glättung `{mode: rtc, max_overlap_steps: 24}`. Der Latenzschätzer behält seine Phase-2-Standardwerte.

**Datenfluss:**

- **Agent:** Er schickt mit jeder Anfrage die aktuelle Aktion, die 24 gequeuten Aktionen (in
  Umgebungseinheiten) und d. Neuer Glättungsmodus `rtc`, Wire-Format wie die Inpainting-Führung aus
  Phase 2.
- **Brain:** Er normalisiert diese Aktionen mit dem Normalisierer der Policy und ruft
  `predict_action_chunk(batch, noise, inference_delay=d, prev_chunk_left_over=…, execution_horizon=25)`
  unter `torch.no_grad()` auf. Die Gewichte der Policy verlangen keine Gradienten. Das Startrauschen ist
  dasselbe wie ohne RTC (Seed je Episode und Beobachtungsschritt). Mit und ohne RTC sind die Läufe dadurch
  auch im Zufall gepaart.

**Bekannte Eigenheit von LeRobot 0.6.1:** Die Geschwindigkeit v_t wird berechnet, bevor x_t Gradienten
verlangt. Die Korrektur ist damit der gewichtete Fehler selbst. Es gibt keine Vektor-Jacobi-Produkte
durch das Netz wie im RTC-Paper. Gemessen wird diese Implementierung, nicht das Paper-Verfahren.

**Prüfungen vor den RTC-Läufen (Unit-Tests):**

1. RTC aus → bitgleich zum Phase-3-Pfad.
2. RTC an, aber ohne Vorgänger → bitgleich zur Abtastung ohne RTC.
3. Mit Vorgänger und d Takten liegen die ersten d Aktionen des neuen Plans näher am Vorgänger als ohne
   Führung.

**R-RTC: „RTC bei 0 ms entspricht der synchronen Referenz“.** RTC bei 0 ms (100 Episoden) gegen `sync25`
aus Phase 3 (53 %), gepaart über die Episoden. Die Prüfung gilt als erfüllt, wenn

1. McNemar exakt p ≥ 0,05 **und**
2. die untere Grenze des 95-%-KI der Differenz (RTC@0 − sync25, Bootstrap, 10 000 Ziehungen) ≥ −15 pp.

Teststärke (Simulation, Abschnitt 6):

| Wahre Erfolgsquote RTC@0 | Bestehenswahrscheinlichkeit |
|---|---|
| 53 % (wie sync25) | 87 % |
| 46 % (Niveau ohne RTC bei 0 ms) | 39 % |
| 38 % (15 pp schlechter, Fehl-Bestehen) | 4 % |

Ist RTC signifikant **besser** als `sync25`, ist das kein Integrationsfehler. Es wird aber als „nicht
entsprechend (besser)“ ausgewiesen. Fällt die Prüfung durch, berichte ich das; die RTC-Läufe laufen
weiter, die Schlussfolgerungen werden markiert.

**Mehraufwand:** Gemessen auf der LIBERO-Eingabe, ein CPU-Thread:

- 5 Aufwärm-Inferenzen, dann 30 abwechselnde Paare.
- Ohne RTC: Phase-3-Pfad mit `inference_mode`.
- Mit RTC: `no_grad`, voller Vorgänger, d = 5.

Ergebnis ist das Median-Verhältnis r mit Bootstrap-KI. Die Servicezeit mit RTC ist auf jeder GPU
s₁ · (1 + r). Annahme: Der relative Mehraufwand überträgt sich von der CPU auf die GPU. Ein negatives r
wird als 0 gesetzt.

## 5. Schritt 3: dichte Dosis-Wirkung und Kurvenmodell

**5.1 Stufen** (je 100 Episoden, Seeds 1000–1099, Einstellungen wie Phase 3):

| | gemessene Latenzen |
|---|---|
| ohne RTC | 0, 500, 1 000, 1 250, 1 500, 2 000 ms aus Phase 3 (wiederverwendet), neu: **100, 200, 300, 750 ms** |
| mit RTC | **0, 100, 200, 300, 500, 750, 1 000, 1 250 ms** (alle neu) |

**Wiederverwendung:** Alle neuen Rollouts laufen mit einem festen Code-Stand: dem ersten Commit nach dieser
Präregistrierung, der Codec- und RTC-Integration enthält, in einem eigenen Git-Worktree, damit die
Weiterentwicklung die laufenden Experimente nicht berührt. Ohne RTC und mit Rohbildern ist das der
Phase-3-Pfad. Die Phase-3-Stufen werden nur gepoolt, wenn ein Re-Lauf von `fest_500ms` (Seeds
1000–1003) mit genau diesem Code-Stand je Episode die protokollierten Felder exakt reproduziert:

- Erfolg, Schritte, Anfragen, verhungerte Takte
- Takt-Reihen
- je Anfrage Beobachtungsschritt, Bytes, Queue-Stände, verworfene und genutzte Aktionen, Latenz

Andernfalls werden alle Phase-3-Stufen neu gemessen (+≈ 3 h, dann greifen die Kürzungen).

**5.2 Kurvenmodell** (umgesetzt in `kortyn/dose_curve.py`, in diesem Commit enthalten):

- **Takt-Regel:** Gekurvt wird über die wirksame Latenz x(L) = 50 ms · max(1, ⌈L / 50 ms⌉).
- **Anpassung je Aufgabe:** g(x) = p / (1 + (x / x₅₀)^h), streng fallend. Binomiale
  Maximum-Likelihood mit analytischem Gradienten. Schranken: logit p ∈ [−10, 10], x₅₀ ∈ [10, 10⁵] ms,
  h ∈ [0,2; 20]. Startpunkte (x₅₀, h) ∈ {(150; 0,8), (400; 2), (800; 5), (1 500; 1,2)}, im Bootstrap
  zusätzlich der Punktschätzer.
- **Anpassungsbereich:** Stufen mit x ≤ 1 250 ms (Budget). Jenseits davon verhungert der Roboter in jedem
  Zyklus, ein anderes Regime.
- **Schwanzregel:** Ab dem Budget fällt die Kurve linear von g(1 250) auf 0 bei 1 500 ms und bleibt 0
  (Phase 3: 0/100 bei 1 500 und 2 000 ms). Die Phase-3-Stufen 1 500 und 2 000 ms gehen nicht in die
  glatte Anpassung ein.
- **Anpassungstest:** Devianz gegen eine Rate je Stufe, χ² mit (Stufen − 3) Freiheitsgraden. Bei
  p < 0,01 wird die Aufgabe über eine isotone, C¹-glatte PCHIP-Kurve durch die Stufenraten gekurvt
  (Pool-Adjacent-Violators). Die Familie wird einmal auf allen Daten entschieden; die Bootstrap-Ziehungen
  übernehmen sie.
- **Gepoolte Kurve:** Mittel der Aufgabenkurven (gleiche Gewichte wie in Phase 3).
- **Unsicherheit:** Gepaarter Bootstrap über die Seeds, je Aufgabe geschichtet. B = 2 000 Ziehungen,
  Seed 4242. Beide Bedingungen (ohne/mit RTC) nutzen dieselben Multiplizitäten, so dass alle Differenzen
  gepaart sind. Konfidenzband: punktweise 2,5- und 97,5-%-Quantile.

**5.3 Vorhersage für einen Roboter:** Mittel der gepoolten Kurve über die Latenzen seiner Anfragen im
Fenster, mit Takt- und Schwanzregel. Bei der Validierung (Episode einer bekannten Aufgabe) zählt die Kurve
dieser Aufgabe.

**5.4 Re-Test der Phase-3-Validierung.**

- **Vorhersage:** Die neue Kurve ohne RTC sagt alle 240 Validierungsepisoden aus Phase 3 neu vorher. Je
  Episode: Kurve ihrer Aufgabe, gemittelt über die gemessenen Latenzen ihrer Anfragen. Zusätzlich die
  Variante mit den Latenzen aus Stufe 1.
- **Bias:** Gepoolter Bias (gemessen − vorhergesagt) über die drei informativen Fälle (180 Episoden) mit
  95-%-KI (Cluster-Bootstrap über Seeds, 10 000 Ziehungen). Der Überlastfall a mit 4 Robotern wird getrennt
  berichtet.
- **Vergleich:** Phase-3-Regel auf denselben Episoden, −7,4 pp.

**Entscheidung:**

| Urteil | Bedingung |
|---|---|
| Bias verschwunden | \|neuer Bias\| ≤ 2,5 pp |
| Bias verkleinert | \|neuer Bias\| ≤ ½ · 7,4 pp |
| nicht erklärt | sonst |

Die Präzision (KI etwa ±10 pp) erlaubt keine signifikante Entscheidung; das Urteil beschreibt die
Punktschätzung.

## 6. Power-Analyse: Sind die Schwellen mit der verfügbaren Präzision nachweisbar?

**Methode** (`kortyn/power_phase4.py`, Ergebnis `results/phase4/power.json`, Seed 20261026):

- Ich simuliere das geplante Design aus Abschnitt 5.1: 2 Aufgaben × 50 Seeds, 8 Stufen je Bedingung, ohne und
  mit RTC auf denselben Seeds.
- Je Szenario sind es 100 Datensätze mit je 150 Bootstrap-Ziehungen. Ausgewertet wird jeder Datensatz mit
  genau dem Schätzer aus Abschnitt 5.2.
- **Wahre Kurven:**
  - drei log-logistische Formen, deren Steilheit h die Spanne abdeckt, die die Phase-3-Daten zulassen
    (Abschnitt 2.2; p und x₅₀ je Aufgabe per Profil-Likelihood an die Phase-3-Stufen angepasst)
  - eine bewusst fehlspezifizierte Wahrheit, stückweise linear durch die Phase-3-Stufenraten
- **RTC-Wirkung:** x₅₀ beziehungsweise die Latenzachse wird mit k gestreckt: k = 1 (keine Wirkung), 1,5 oder 2.
- **Abhängigkeit:** Gauß-Copula. Die latente Korrelation einer Episode beträgt 0,7 · exp(−|Δx| / 1 000 ms)
  zwischen Stufen und zusätzlich 0,85 zwischen den Bedingungen, kalibriert an Abschnitt 2.3.

| Wahre Form | h (Aufg. 0 / 8) | k | wahres L* ohne / mit (ms) | Median Punkt-L* | Median konservatives L* | Abdeckung 95-%-Intervall | SD des Verlusts am wahren L* (pp) | wahrer Gewinn H1 (pp) | Power H1 | Power H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| früher Abfall | 1,6 / 1,2 | 1 | 50 / 50 | 50 / 50 | 50 / 50 | 100 / 100 % | 0,0 / 0,0 | 0,0 | 0 % | 0 % |
| früher Abfall | 1,6 / 1,2 | 1,5 | 50 / 50 | 50 / 50 | 50 / 50 | 100 / 99 % | 0,0 / 0,0 | 4,9 | 75 % | 0 % |
| früher Abfall | 1,6 / 1,2 | 2 | 50 / 50 | 50 / 50 | 50 / 50 | 100 / 100 % | 0,0 / 0,0 | 8,8 | 99 % | 0 % |
| nahe Profil-MLE | 5 / 2,5 | 1 | 200 / 200 | 200 / 200 | 100 / 100 | 98 / 99 % | 1,8 / 2,2 | 0,0 | 1 % | 0 % |
| nahe Profil-MLE | 5 / 2,5 | 1,5 | 200 / 300 | 200 / 300 | 100 / 100 | 98 / 93 % | 2,0 / 2,3 | 7,8 | 98 % | 10 % |
| nahe Profil-MLE | 5 / 2,5 | 2 | 200 / 400 | 200 / 400 | 100 / 100 | 99 / 98 % | 1,9 / 2,8 | 13,9 | 100 % | 9 % |
| später Abfall | 8 / 3,5 | 1 | 300 / 300 | 300 / 300 | 150 / 175 | 98 / 99 % | 2,6 / 2,2 | 0,0 | 2 % | 0 % |
| später Abfall | 8 / 3,5 | 1,5 | 300 / 450 | 300 / 450 | 150 / 200 | 100 / 96 % | 2,8 / 2,6 | 8,5 | 100 % | 12 % |
| später Abfall | 8 / 3,5 | 2 | 300 / 600 | 300 / 600 | 150 / 150 | 99 / 98 % | 2,7 / 3,0 | 15,0 | 100 % | 18 % |
| linear (fehlspezifiziert) | – | 1 | 100 / 100 | 150 / 150 | 50 / 50 | 96 / 97 % | 1,1 / 1,0 | 0,0 | 1 % | 1 % |
| linear (fehlspezifiziert) | – | 1,5 | 100 / 100 | 150 / 150 | 50 / 50 | 93 / 98 % | 1,2 / 1,4 | 6,5 | 88 % | 0 % |
| linear (fehlspezifiziert) | – | 2 | 100 / 150 | 150 / 200 | 50 / 50 | 93 / 98 % | 1,1 / 1,9 | 10,5 | 100 % | 0 % |

**Folgerungen für die Schwellen:**

1. **3-pp-Schwelle für L* (Vorgabe):** über das Kurvenmodell nachweisbar, aber nur grob.
   - Für alle log-logistischen Wahrheiten trifft der Punktschätzer das wahre L* im Median genau (mittlere
     absolute Abweichung ein Takt, 50 ms).
   - Bei der fehlspezifizierten Wahrheit liegt er einen Takt zu hoch.
   - Die Bootstrap-Streuung des Verlusts am wahren L* beträgt 1–3 pp. Das konservative L* liegt deshalb
     typisch 2–3 Takte (100–150 ms) unter dem Punkt, die Intervalle decken die Wahrheit zu 93–100 %.
   - Ohne Kurvenmodell wäre die Schwelle nicht nachweisbar: Der Unterschied zweier Stufen ist mit 100
     gepaarten Episoden nur auf etwa ±10 pp genau.
   - Die Schwelle bleibt wie vorgegeben. Punkt und konservativer Wert werden immer gemeinsam berichtet.
2. **RTC-Wirkung:** H1, der mittlere Gewinn im Budget, hat Power. Bei k = 1,5 sind es 75–100 %, bei
   k = 2 99–100 %, der Fehlalarm bei k = 1 liegt bei ≤ 2 %. H1 ist deshalb das primäre Kriterium.
   H2 (ΔL* > 0) erreicht nur 0–18 % und wird nur beschreibend verwendet: Ein Nichtnachweis ist kein Beleg
   gegen eine Verschiebung.
3. **Früher Abfall als Grenzfall:** Ist die wahre Kurve schon im ersten Takt steil, bleibt L* auch mit RTC
   bei 50 ms, obwohl RTC den Erfolg spürbar hebt (+5 bis +9 pp). Deshalb berichte ich neben L* immer die
   ganzen Kurven und H1.
4. **3-pp-Kapazitätskriterium:** Ein typischer Kontrast (Kurve bei 250 gegen 400 ms) streut im Bootstrap
   mit 1,5–3,7 pp (Median 3,1 pp). Die Punkt-Kapazität ist so genau wie L*, die konservative Kapazität
   (Abschnitt 7.3) wird deutlich kleiner ausfallen.
5. **Ausweichkurve:** Die isotone Ausweichkurve griff in ≤ 0,5 % der simulierten Anpassungen.

## 7. Schritt 4: L*, Kapazität, WLAN, Validierung

**7.1 L*** (Definition der Vorgabe) ist die größte Latenz, deren vorhergesagter Verlust gegenüber 0 ms
derselben Bedingung höchstens 3 pp beträgt (gepoolte Kurve, Takt-Regel). Berichtet wird es je Bedingung
(ohne/mit RTC):

- als Punktschätzung,
- konservativ als untere Grenze des 95-%-Bootstrap-Intervalls (2,5-%-Quantil),
- mit oberer Grenze.

L* ist ein Vielfaches von 50 ms. Bleibt der Verlust bis zum Budget unter 3 pp, wird L* als „≥ 1 250 ms“
(zensiert) ausgewiesen. Zusätzlich gibt es die stetige wirksame Latenz x*, an der die glatte Kurve 3 pp
verloren hat (für die Abbildungen), und den absoluten Erfolg bei L*. Weil L* relativ zur eigenen 0-ms-Stufe
definiert ist, stelle ich beide Kurven zusätzlich absolut und gegen `sync25` dar.

**7.2 Hypothesen zur RTC-Wirkung:**

- **H1 (primär):** RTC erhöht den mittleren vorhergesagten Erfolg im Budget, also das Mittel der
  gepoolten Kurve mit minus ohne RTC über die 25 Takte 50, 100, …, 1 250 ms, gleich gewichtet.
  - Bestätigt: untere Grenze des gepaarten 95-%-Bootstrap-KI > 0.
  - „RTC schadet“: obere Grenze < 0.
  - Sonst: „nicht nachweisbar“.
- **H2:** RTC erhöht L*. Nachgewiesen nur, wenn die untere Grenze des gepaarten 95-%-Intervalls von
  ΔL* = L*(RTC) − L*(ohne) > 0 ist. Die Teststärke ist gering (Abschnitt 6). Ein Nichtnachweis heißt
  „nicht nachgewiesen“, nicht „kein Effekt“.

**7.3 Kapazität** (Definition der Vorgabe). Die Kapazität ist die größte Flottengröße n, für die bei n
**und jeder kleineren Rastergröße** zwei Kriterien gelten:

1. Höchstens 1 % aller Anfragen über alle Wiederholungen reißen ihr Budget.
2. Der schlechteste Roboter liegt in der Vorhersage höchstens 3 pp unter dem Einzelroboter.
   - Schlechtester Roboter: je Wiederholung der Roboter mit der niedrigsten Vorhersage, gemittelt über
     die Wiederholungen.
   - Einzelroboter: Vorhersage für n = 1 in derselben Konfiguration.
   - Ein von der Zugangskontrolle abgewiesener Roboter zählt als 0.

Die Vorhersage nutzt die Kurve der jeweiligen Bedingung, also die RTC-Kurve für Konfigurationen mit RTC.
Primär zählt die Punktschätzung. **Konservative Kapazität:** Kriterium 2 mit dem 97,5-%-Bootstrap-Quantil
von (Einzel − schlechtester) ≤ 3 pp.

**Konfigurationen** (Stufe 1, Warteschlangen-Simulation in Simulationszeit, Einstellungen wie Phase 3):

- **GPU:** RTX 3060 Ti (`rtx3060ti_schaetzung`, s₁ = 220 ms). Eine Messdatei nach
  `docs/benchmark_rtx3060ti.md` liegt nicht im Repository, deshalb gelten die Batching-Profile. Mit RTC
  s₁ · (1 + r).
- **Batching:** realistisch, optimistisch, pessimistisch, kein Gewinn (β = 0,15 / 0,05 / 0,45 / 1,0).
- **Strategien:**
  - a FIFO
  - b FIFO + Batching
  - d EDF + Batching
  - e d + Zugangskontrolle

  Definitionen wie in Phase 3. a hängt nicht vom Batching-Profil ab.
- **Netz:** Büro-WLAN (Latenzmodell aus Phase 3, Bandbreite unbegrenzt).
- **Raster:** jede Flottengröße ab 1 aufwärts, bis drei aufeinanderfolgende Größen Kriterium 1 verfehlen
  oder die Zugangskontrolle einen Roboter abweist (dann kann die Kapazität nicht weiter wachsen),
  höchstens 128.
- **Wiederholungen:** 5 je Größe. Horizont 630 s, Fenster ab 30 s, Startstreuung 10 s, Servicezeit-CV 0,05.
- **Gepaart:** gleiche Zufallsströme ohne und mit RTC.

**7.4 WLAN-Modell (geteilte Bandbreite).**

- **Modell:** Ein gemeinsames Funkmedium mit Nutzrate C (Mbit/s) überträgt alle Nachrichten aller Roboter
  nacheinander (FIFO, beide Richtungen, Halbduplex wie WLAN). Belegungszeit je Nachricht = Bytes · 8 / C.
  Das Phase-3-Latenzmodell des Büro-WLANs kommt obendrauf.
- **Nutzlast je Anfrage:** gemessene mittlere Bytes des Standard-Codecs (zum Vergleich: roh, 777 600
  Bytes). Antwort: der Aktions-Chunk.
- **Raster:** C ∈ {25, 50, 100, 200, 400} Mbit/s, als Annahmen markiert (ausgelastetes 2,4-GHz-WLAN bis
  gutes Wi-Fi 6).
- **Konfiguration:** Strategie d, Batching realistisch, ohne und mit RTC. Größenraster wie 7.3.
- **„Das Netz wird zum Engpass“:** Die Kapazität mit WLAN-Modell liegt unter der Kapazität mit
  unbegrenztem Netz. Das Netz limitiert dann ab Flottengröße (Kapazität mit WLAN + 1). Je C werden beide
  Kapazitäten und die Auslastungen von GPU und Funkmedium an der Kapazitätsgrenze berichtet.

**7.5 Validierung mit echten Rollouts.**

- **Konfiguration:** beste Strategie mit RTC = größte Kapazität (Punkt) auf der RTX 3060 Ti, Batching
  realistisch, Büro-WLAN. Bei Gleichstand in der Reihenfolge d, e, b, a.
- **Flottengrößen:** n₁ = max(2, min(K, 8)) und n₂ = min(2 · n₁, 16), K = vorhergesagte Kapazität. Passt
  n₂ nicht in den Speicher, nehme ich die größte passende Größe (als Abweichung ausgewiesen).
- **Umfang:** je Fall 100 gezählte Roboter-Episoden (Seed-Zuteilung wie Phase 3), 4 Inferenz-Prozesse.
- **Vorhersage je Episode:** Kurve der Aufgabe der Episode, gemittelt über die Latenzen des Roboters aus
  Stufe 1 (Wiederholung 0), RTC-Kurve.
- **Kriterium:** Je Fall enthält das 97,5-%-KI (Bonferroni über 2 Fälle, Bootstrap über Episoden,
  10 000 Ziehungen) der Differenz gemessen − vorhergesagt die Null.
- **Diagnose:** Vorhersage aus den eigenen gemessenen Latenzen; Latenzquantile Stufe 1 gegen Rollouts.
- **Teststärke:** KI-Halbbreite ≈ ±10 pp. Ein Bias von 7 pp wird mit 24 % Wahrscheinlichkeit erkannt,
  einer von 15 pp mit 85 %. Die Validierung schließt grobe Fehler aus, keine 5-pp-Abweichungen.

## 8. Schritt 5: Wirtschaftlichkeit

- **Kostenmodell:** wie Phase 3 (`kortyn/experiments/phase3/economics.yaml`).
  - Preise mit Quelle und Datum 25.10.2026, alle als Annahme markiert, netto.
  - 36 Monate Abschreibung, Strom 0,2366 €/kWh, Betrieb rund um die Uhr.
  - Host-PC ohne GPU, je Roboter ein Raspberry Pi 5 als Client.
- **Brain:** RTX 3060 Ti ohne und mit RTC.
- **Rechner im Roboter:**

  | Gerät | Latenz (SmolVLA, Quellen in der Datei) |
  |---|---|
  | Jetson Orin Nano Super | 567 ms |
  | AGX Orin | 262 ms |
  | Thor | 50 ms |

  Ihr Erfolg kommt aus derselben Kurve (Latenz = Inferenzzeit, kein Netz), mit RTC aus der RTC-Kurve
  (Latenz · (1 + r)).
- **Gleicher Erfolg:** Für jedes Gerät und jede RTC-Einstellung nehme ich die größte Flottengröße, bei der
  Kriterium 1 gilt und der schlechteste Roboter des Brains mindestens den vorhergesagten Erfolg des Geräts
  erreicht. Daraus folgen die Brain-Kosten je Roboter einschließlich Client. Das Brain gewinnt, wo diese
  Kosten unter den Kosten des Geräts liegen. Erreicht das Brain den Erfolg des Geräts in keiner Größe,
  gewinnt das Gerät.
- **Zusätzlich:** Kosten je Roboter an der Kapazität aus 7.3.

**Entscheidungsregel für die Geschäftsfrage** (Batching realistisch, mit RTC, beste Strategie):

| Urteil | Bedingung |
|---|---|
| **trägt** | Das Brain ist bei gleichem Erfolg günstiger je Roboter als das günstigste Gerät, das diesen Erfolg erreicht, und die Kapazität (7.3) ist ≥ 4 Roboter |
| **trägt nur unter Annahmen** | Das gilt nur im optimistischen Batching-Profil |
| **trägt nicht** | sonst |

## 9. Übersicht der Entscheidungen

| Nr. | Frage | Kriterium | Abschnitt |
|---|---|---|---|
| C1 | Codec verlustfrei | Bitgleichheit Bild und Modelleingabe (Test) | 3 |
| C2 | Standard-Codec | kleinste Bytes bei ≤ 25 ms Kodieren + Dekodieren | 3 |
| R0 | Phase-3-Daten wiederverwendbar | exakte Reproduktion von 4 Episoden | 5.1 |
| R-RTC | RTC bei 0 ms wie sync25 | McNemar p ≥ 0,05 und untere KI-Grenze ≥ −15 pp | 4 |
| B | Phase-3-Bias | \|Bias\| ≤ 2,5 pp verschwunden, ≤ 3,7 pp verkleinert | 5.4 |
| H1 | RTC wirkt | untere KI-Grenze des mittleren Gewinns im Budget > 0 | 7.2 |
| H2 | RTC erhöht L* | untere Grenze des Intervalls von ΔL* > 0 | 7.2 |
| K | Kapazität | ≤ 1 % verpasst und schlechtester ≤ 3 pp unter Einzelroboter | 7.3 |
| V | Validierung | 97,5-%-KI von gemessen − vorhergesagt enthält 0, je Fall | 7.5 |
| W | Wirtschaftlichkeit | Regel in Abschnitt 8 | 8 |

## 10. Laufzeitschätzung und Kürzungen

_Stand 26.10.2026, 18:40 UTC, vor dem ersten Lauf. Phase 4 läuft seit ≈ 17:10 UTC (PR, Recherche,
Power-Analyse, Integration)._

**Gemessene Kosten auf dieser Maschine** (4 vCPU, 13,4 GB Speichergrenze):

- **Inferenz:** Eine SmolVLA-Inferenz auf der LIBERO-Eingabe kostet 3,7 s auf einem CPU-Kern.
- **Episode:** Eine Episode kostet ≈ 74 s je Worker allein und ≈ 95 s neben Warteschlangen-Simulationen.
  Planwert: 80 s bei 4 Workern.
- **RTC:** Der Mehraufwand ist nach der informellen Messung klein und geht hier nicht ein.
- **Flotten-Rollouts:** In Phase 3 kostete eine Roboter-Episode 39 s bei 2–4 Robotern. Ab etwa 8 Robotern
  sind mehr Anfragen gleichzeitig offen; Planwert 26 s je Roboter-Episode bei 4 Inferenz-Prozessen.
- **Warteschlangen-Simulation:** 0,04 s je Roboter-Lauf.

| Block | Umfang | Wandzeit |
|---|---|---|
| Vorbereitung, Power-Analyse, Präregistrierung | – | ≈ 1,5 h (17:10–18:40) |
| Tests, Commit der Integration, Worktree, R0 (4 Episoden), Codec-Messung | – | ≈ 0,4 h |
| A: ohne RTC, neue Stufen 100/200/300/750 ms | 400 Episoden | ≈ 2,2 h |
| RTC-Mehraufwand (nach A, ruhige CPU) | 30 Paare | ≈ 0,25 h |
| B: mit RTC, 8 Stufen | 800 Episoden | ≈ 4,4 h |
| Stufe 1 ohne/mit RTC und WLAN (neben A/B) | ≈ 100 000 Roboter-Läufe | ≈ 1,1 CPU-h, ≈ +0,3 h |
| C: Validierung | 2 × 100 Roboter-Episoden | ≈ 1,4 h |
| Auswertung, Report, README, Abnahme | – | ≈ 1,3 h |
| **Summe** | | **≈ 11,8 h, Ende ≈ 05:00 UTC** |

Das liegt an der 12-Stunden-Grenze, und Phase 3 lief 30 % langsamer als geplant. Deshalb lege ich drei
Kürzungen vorab fest. Ich setze sie in dieser Reihenfolge ein, sobald der Plan an einer Blockgrenze mehr als
eine Stunde zurückliegt. Jede wird vor den betroffenen Läufen als Abweichung mit Zeitpunkt ausgewiesen.

| | Kürzung | Ersparnis | Was verloren geht |
|---|---|---|---|
| 1 | RTC-Stufe 1 250 ms entfällt | ≈ 0,55 h | Die RTC-Kurve wird zwischen 1 000 und 1 250 ms nur geglättet fortgesetzt; das betrifft nur Roboter nahe Überlast |
| 2 | Validierung mit 60 statt 100 Roboter-Episoden je Fall | ≈ 0,6 h | KI ±13 statt ±10 pp |
| 3 | RTC-Stufe 750 ms entfällt | ≈ 0,55 h | Stützstelle der RTC-Kurve zwischen 500 und 1 000 ms |

## 11. Abnahme (Vorgabe)

1. pytest, ruff und mypy grün (beide Umgebungen).
2. Diese Präregistrierung ist vor dem ersten Lauf committet.
3. Der verlustfreie Codec ist bitgleich nachgewiesen.
4. RTC bei 0 ms entspricht der synchronen Referenz (R-RTC).
5. Kurven, L*, Kapazitäten und Validierung liegen vor.
6. `reports/phase4.md` und README sind aktualisiert.
