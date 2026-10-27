# Phase 4: Daten (automatisch erzeugt)

_Erzeugt am 27.10.2026 05:04 UTC mit `python -m kortyn.report_phase4`._

## Schritt 1: verlustfreier Bildtransport

500 Anfragen aus 100 Episoden (Takte 0, 50, 100, 150, 200), Bilder observation.images.camera1 1×360×360×3, observation.images.camera2 1×360×360×3. Zeiten: Median je Anfrage über beide Kameras, 3 Wiederholungen. Bandbreite je Roboter = mittlere Bytes · 8 / 1,25 s Anfrageabstand.

| Kandidat | bitgleich (Bild · Modelleingabe) | Bytes je Anfrage | p95 | Anteil an roh | Mbit/s je Roboter | Kodieren ms | Dekodieren ms | Summe ms |
|---|---|---|---|---|---|---|---|---|
| raw | 500/500 · 500/500 | 778 kB | 778 kB | 100 % | 5,0 | 0,2 | 0,1 | 0,3 |
| png_1 | 500/500 · 500/500 | 260 kB | 319 kB | 33 % | 1,7 | 18,3 | 9,4 | 27,8 |
| png_6 | 500/500 · 500/500 | 228 kB | 289 kB | 29 % | 1,5 | 74,8 | 8,5 | 83,8 |
| webp_0 | 500/500 · 500/500 | 191 kB | 251 kB | 25 % | 1,2 | 41,3 | 6,3 | 47,3 |
| webp_4 | 500/500 · 500/500 | 155 kB | 206 kB | 20 % | 1,0 | 291,2 | 6,8 | 297,4 |
| zstd_3 | 500/500 · 500/500 | 312 kB | 403 kB | 40 % | 2,0 | 4,9 | 1,1 | 6,0 |
| **zstd_9** | 500/500 · 500/500 | 292 kB | 381 kB | 38 % | 1,9 | 15,9 | 1,1 | 17,1 |

**Standard (Regel aus Abschnitt 3):** `zstd_9` – kleinste mittlere Bytes unter den verlustfreien Kandidaten mit Kodieren + Dekodieren ≤ 25 ms; innerhalb von 2 % der schnellere.

## Schritt 2: RTC

| Messung | ohne RTC (Median) | mit RTC (Median) | r [95-%-KI] | angewandt | s₁ mit RTC |
|---|---|---|---|---|---|
| 30 abwechselnde Paare, d = 5, H = 25 | 2 439 ms | 2 467 ms | +1,1 % [-0,4, +6,6] % | 1,1 % | 222 ms |

**R-RTC** (RTC bei 0 ms gegen die synchrone Referenz `sync25`, gleiche Seeds):

| Paare | sync25 | RTC 0 ms | Differenz [95-%-KI] | nur sync / nur RTC | McNemar p | Urteil |
|---|---|---|---|---|---|---|
| 100 | 53 % | 42 % | -11 pp [-23, +1] pp | 27 / 16 | 0,126 | nicht erfüllt |

## Schritt 3: Dosis-Wirkung

100 Seeds je Stufe (Aufgaben 0, 8), gepaarter Seed-Bootstrap mit 2000 Ziehungen (Seed 4242), Toleranz 3 %.

**ohne RTC** – gemessene Stufen und glatte Kurve (95-%-Bootstrap-Band):

| Stufe | wirksam (Takt-Regel) | Erfolg (Aufgaben gleich gewichtet) | Aufgabe 0 | Aufgabe 8 | Kurve [95-%-Band] |
|---|---|---|---|---|---|
| 0 ms | 50 ms | 47 % | 42 % | 52 % | 45 % [37, 53] % |
| 100 ms | 100 ms | 48 % | 48 % | 48 % | 45 % [37, 53] % |
| 200 ms | 200 ms | 47 % | 46 % | 48 % | 44 % [37, 51] % |
| 300 ms | 300 ms | 30 % | 26 % | 34 % | 39 % [32, 46] % |
| 500 ms | 500 ms | 29 % | 14 % | 44 % | 27 % [21, 35] % |
| 750 ms | 750 ms | 17 % | 2 % | 32 % | 17 % [10, 23] % |
| 1 000 ms | 1 000 ms | 5 % | 0 % | 10 % | 5 % [2, 9] % |
| 1 250 ms | 1 250 ms | 1 % | 0 % | 2 % | 1 % [0, 3] % |
| 1 500 ms | 1 500 ms | 0 % | 0 % | 0 % | 0 % [0, 0] % |
| 2 000 ms | 2 000 ms | 0 % | 0 % | 0 % | – |

| Aufgabe | Familie | p | x₅₀ | h | Devianz / FG | GOF p |
|---|---|---|---|---|---|---|
| Aufgabe 0 | log-logistisch | 45 % | 370 ms | 4,17 | 3,34 / 5 | 0,648 |
| Aufgabe 8 | log-logistisch | 45 % | 843 ms | 7,59 | 3,84 / 5 | 0,573 |

**mit RTC** – gemessene Stufen und glatte Kurve (95-%-Bootstrap-Band):

| Stufe | wirksam (Takt-Regel) | Erfolg (Aufgaben gleich gewichtet) | Aufgabe 0 | Aufgabe 8 | Kurve [95-%-Band] |
|---|---|---|---|---|---|
| 0 ms | 50 ms | 42 % | 28 % | 56 % | 38 % [32, 48] % |
| 100 ms | 100 ms | 38 % | 24 % | 52 % | 38 % [32, 46] % |
| 200 ms | 200 ms | 38 % | 30 % | 46 % | 38 % [31, 45] % |
| 300 ms | 300 ms | 33 % | 28 % | 38 % | 38 % [29, 44] % |
| 500 ms | 500 ms | 30 % | 14 % | 46 % | 29 % [18, 36] % |
| 1 000 ms | 1 000 ms | 5 % | 0 % | 10 % | 5 % [1, 11] % |

| Aufgabe | Familie | p | x₅₀ | h | Devianz / FG | GOF p |
|---|---|---|---|---|---|---|
| Aufgabe 0 | log-logistisch | 28 % | 501 ms | 20,00 | 0,48 / 3 | 0,923 |
| Aufgabe 8 | log-logistisch | 48 % | 788 ms | 5,59 | 3,66 / 3 | 0,301 |

**L\*** (größte Latenz mit höchstens 3 pp vorhergesagtem Verlust gegenüber 0 ms derselben Bedingung):

| Bedingung | L* (Punkt) | konservativ (2,5 %) | obere Grenze (97,5 %) | x* (stetig) | Erfolg bei 0 ms | Erfolg bei L* |
|---|---|---|---|---|---|---|
| ohne RTC | 200 ms | 150 ms | 450 ms | 236 ms | 45 % | 44 % |
| mit RTC | 450 ms | 50 ms | 450 ms | 457 ms | 38 % | 35 % |

| Hypothese | Schätzung | 95-%-KI (gepaart) | Urteil |
|---|---|---|---|
| H1: mittlerer Gewinn über die 25 Takte im Budget | -1,6 pp | [-6,1, +3,4] pp | nicht nachweisbar |
| H2: ΔL* = L*(mit) − L*(ohne) | +250 ms | [-200, +300] ms | nicht nachgewiesen |

![Dosis-Wirkung](phase4_kurven.png)

### Re-Test des Phase-3-Validierungsbias

| Fall | Episoden | gemessen | dicht · eigene | dicht · Stufe 1 | Phase 3 · eigene | Phase 3 · Stufe 1 |
|---|---|---|---|---|---|---|
| a_n2 | 60 | 17 % | 36 % | 36 % | 24 % | 22 % |
| a_n4 | 60 | 0 % | 0 % | 0 % | 0 % | 0 % |
| e_n2 | 60 | 15 % | 36 % | 36 % | 23 % | 22 % |
| e_n4 | 60 | 10 % | 28 % | 26 % | 17 % | 17 % |

Bias gemessen − vorhergesagt über die aussagekräftigen Fälle a_n2, e_n2, e_n4 (180 Episoden, Cluster-Bootstrap über Seeds):

| Vorhersage | Bias | 95-%-KI |
|---|---|---|
| dichte Kurve, eigene Latenzen | -19,6 pp | [-26,8, -11,7] pp |
| dichte Kurve, Latenzen aus Stufe 1 | -18,8 pp | [-26,0, -11,1] pp |
| Phase-3-Regel, eigene Latenzen | -7,4 pp | [-17,1, +2,1] pp |
| Phase-3-Regel, Latenzen aus Stufe 1 | -6,5 pp | [-16,1, +3,1] pp |

**Urteil (eigene Latenzen):** nicht erklärt · **Urteil (Stufe-1-Latenzen):** nicht erklärt

### Reihenfolge der Läufe: frische Prozesse

Ergänzung (nicht präregistriert): Seeds 1000–1049 bei 500 ms, jede Episode in einem frischen Prozess (50 Episoden, Erfolg 46 %).

| Vergleich | Paare | Referenz | frischer Prozess | Differenz [95-%-KI] | nur Referenz / nur frisch | McNemar p |
|---|---|---|---|---|---|---|
| Block A (alle Stufen eines Seeds nacheinander) | 50 | 44 % | 46 % | +2 pp [-14, +18] pp | 8 / 9 | 1,000 |
| Phase 3 | 50 | 26 % | 46 % | +20 pp [+4, +36] pp | 4 / 14 | 0,031 |

### Sensitivität: Phase-3-Stufen gepoolt

Ergänzung (nicht präregistriert): Die Phase-3-Episoden der wieder gemessenen Stufen (0 ms, 500 ms, 1 000 ms, 1 250 ms) gepoolt in die Kurve ohne RTC; Punktschätzung mit derselben Anpassungsregel.

| Kurve ohne RTC | Familie je Aufgabe | L* | Erfolg bei 0 ms | 250 / 500 / 750 ms | Re-Test-Bias [95-%-KI] | Urteil |
|---|---|---|---|---|---|---|
| präregistriert (nur Phase 4) | 0: loglogistic, 8: loglogistic | 200 ms | 45 % | 42 % / 27 % / 17 % | -19,6 pp [-26,8, -11,7] pp | nicht erklärt |
| mit Phase-3-Stufen gepoolt | 0: loglogistic, 8: loglogistic | 250 ms | 44 % | 42 % / 27 % / 15 % | -17,0 pp [-24,1, -9,2] pp | nicht erklärt |

## Schritt 4: Kapazität

Netz Büro-WLAN (Bandbreite unbegrenzt), Toleranz 3 %, höchstens 1 % verpasste Anfragen; Kapazität = größte Größe, bei der sie und jede kleinere beide Kriterien erfüllen. Strategie a batcht nicht und gilt für jedes Profil.

| Batching | Strategie | ohne RTC: Kapazität (konservativ) | ohne RTC: schlechtester (Verlust) bei K | ohne RTC: GPU belegt / verpasst bei K | mit RTC: Kapazität (konservativ) | mit RTC: schlechtester (Verlust) bei K | mit RTC: GPU belegt / verpasst bei K |
|---|---|---|---|---|---|---|---|
| optimistisch | a FIFO | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **3** (2) | 36 % (1,6 pp) | 54 % / 0,00 % |
| optimistisch | b FIFO + Batching | **3** (2) | 36 % (2,6 pp) | 50 % / 0,00 % | **9** (3) | 35 % (2,9 pp) | 92 % / 0,00 % |
| optimistisch | d EDF + Batching | **3** (2) | 38 % (2,9 pp) | 52 % / 0,00 % | **9** (2) | 35 % (2,8 pp) | 94 % / 0,00 % |
| optimistisch | e d + Zugangskontrolle | **3** (2) | 38 % (2,9 pp) | 52 % / 0,00 % | **9** (2) | 35 % (2,8 pp) | 94 % / 0,00 % |
| realistisch | a FIFO | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **3** (2) | 36 % (1,6 pp) | 54 % / 0,00 % |
| realistisch | b FIFO + Batching | **3** (2) | 36 % (2,9 pp) | 51 % / 0,00 % | **5** (2) | 35 % (2,4 pp) | 75 % / 0,00 % |
| realistisch | d EDF + Batching | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **5** (2) | 36 % (2,1 pp) | 77 % / 0,00 % |
| realistisch | e d + Zugangskontrolle | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **5** (2) | 36 % (2,1 pp) | 77 % / 0,00 % |
| pessimistisch | a FIFO | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **3** (2) | 36 % (1,6 pp) | 54 % / 0,00 % |
| pessimistisch | b FIFO + Batching | **2** (2) | 38 % (1,2 pp) | 35 % / 0,00 % | **3** (2) | 36 % (1,3 pp) | 52 % / 0,00 % |
| pessimistisch | d EDF + Batching | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **4** (2) | 35 % (2,7 pp) | 69 % / 0,00 % |
| pessimistisch | e d + Zugangskontrolle | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **4** (2) | 35 % (2,7 pp) | 69 % / 0,00 % |
| kein Gewinn | a FIFO | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **3** (2) | 36 % (1,6 pp) | 54 % / 0,00 % |
| kein Gewinn | b FIFO + Batching | **2** (2) | 37 % (1,6 pp) | 36 % / 0,00 % | **2** (2) | 37 % (0,8 pp) | 36 % / 0,00 % |
| kein Gewinn | d EDF + Batching | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **3** (2) | 35 % (2,5 pp) | 54 % / 0,00 % |
| kein Gewinn | e d + Zugangskontrolle | **2** (2) | 39 % (1,4 pp) | 36 % / 0,00 % | **3** (2) | 35 % (2,5 pp) | 54 % / 0,00 % |

![Kapazität](phase4_kapazitaet.png)

### WLAN mit geteilter Bandbreite

Strategie d, Batching realistisch; ein gemeinsames Funkmedium (FIFO, beide Richtungen) zusätzlich zum Latenzmodell des Büro-WLANs. In Klammern jeweils dieselbe Konfiguration mit unbegrenztem Netz. Ergänzung (nicht präregistriert): die Kapazität nach dem Budget allein (≤ 1 % verpasste Anfragen, jeder Roboter zugelassen), also ohne die 3-pp-Grenze; sie zeigt, wo das Funkmedium sättigt.

| Nutzrate C | Nutzlast je Anfrage | Bedingung | Kapazität (3 pp, 1 %) | Engpass | GPU / Funk belegt | nur Budget | Engpass (Budget) | GPU / Funk belegt (Budget) |
|---|---|---|---|---|---|---|---|---|
| 25 Mbit/s | 292 kB | mit RTC | **2** (5) | Netz ab n = 3 | 36 % / 15 % | 12 (22) | Netz ab n = 13 | 100 % / 91 % |
| 50 Mbit/s | 292 kB | mit RTC | **3** (5) | Netz ab n = 4 | 53 % / 11 % | 21 (22) | Netz ab n = 22 | 100 % / 79 % |
| 100 Mbit/s | 292 kB | mit RTC | **4** (5) | Netz ab n = 5 | 67 % / 8 % | 22 (22) | GPU | 100 % / 42 % |
| 200 Mbit/s | 292 kB | mit RTC | **4** (5) | Netz ab n = 5 | 67 % / 4 % | 23 (22) | GPU | 100 % / 22 % |
| 400 Mbit/s | 292 kB | mit RTC | **4** (5) | Netz ab n = 5 | 66 % / 2 % | 23 (22) | GPU | 100 % / 11 % |
| 25 Mbit/s | 292 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 15 % | 12 (23) | Netz ab n = 13 | 100 % / 91 % |
| 50 Mbit/s | 292 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 8 % | 21 (23) | Netz ab n = 22 | 100 % / 79 % |
| 100 Mbit/s | 292 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 4 % | 23 (23) | GPU | 100 % / 44 % |
| 200 Mbit/s | 292 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 2 % | 23 (23) | GPU | 100 % / 22 % |
| 400 Mbit/s | 292 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 1 % | 23 (23) | GPU | 100 % / 11 % |
| 25 Mbit/s | 778 kB | mit RTC | **1** (5) | Netz ab n = 2 | 18 % / 20 % | 4 (22) | Netz ab n = 5 | 72 % / 80 % |
| 50 Mbit/s | 778 kB | mit RTC | **2** (5) | Netz ab n = 3 | 36 % / 20 % | 9 (22) | Netz ab n = 10 | 100 % / 90 % |
| 100 Mbit/s | 778 kB | mit RTC | **3** (5) | Netz ab n = 4 | 53 % / 15 % | 18 (22) | Netz ab n = 19 | 100 % / 90 % |
| 200 Mbit/s | 778 kB | mit RTC | **4** (5) | Netz ab n = 5 | 66 % / 10 % | 21 (22) | Netz ab n = 22 | 100 % / 53 % |
| 400 Mbit/s | 778 kB | mit RTC | **5** (5) | nicht das Netz | 78 % / 6 % | 22 (22) | GPU | 100 % / 28 % |
| 25 Mbit/s | 778 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 40 % | 4 (23) | Netz ab n = 5 | 71 % / 80 % |
| 50 Mbit/s | 778 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 20 % | 9 (23) | Netz ab n = 10 | 100 % / 90 % |
| 100 Mbit/s | 778 kB | ohne RTC | **3** (2) | nicht das Netz | 52 % / 15 % | 19 (23) | Netz ab n = 20 | 100 % / 95 % |
| 200 Mbit/s | 778 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 5 % | 23 (23) | GPU | 100 % / 58 % |
| 400 Mbit/s | 778 kB | ohne RTC | **2** (2) | nicht das Netz | 36 % / 3 % | 23 (23) | GPU | 100 % / 29 % |

![WLAN](phase4_wlan.png)

### Validierung mit echten Rollouts

| Fall | Episoden | gemessen | vorhergesagt (Stufe 1) | Differenz [97,5-%-KI] | Kriterium V | Diagnose: Vorhersage aus eigenen Latenzen |
|---|---|---|---|---|---|---|
| d_n10 | 50 | 38 % | 45 % | -7,4 pp [-21,4, +8,6] pp | erfüllt | 45 %: -7,4 pp [-21,6, +8,5] pp |
| d_n5 | 60 | 32 % | 43 % | -11,7 pp [-24,6, +1,8] pp | erfüllt | 43 %: -11,7 pp [-24,7, +1,9] pp |

**Validierung insgesamt:** erfüllt (Bedingung mit RTC).

## Schritt 5: Wirtschaftlichkeit

**Annahmen** (alle Preise netto, Abschreibung 36 Monate, Strom 0,2366 €/kWh [Destatis, Pressemitteilung Nr. 345 vom 30.09.2026: Gewerbe 20-500 MWh/Jahr, 1. Halbjahr 2026 (vorläufig), ohne MwSt.; https://www.destatis.de/DE/Presse/Pressemitteilungen/2026/09/PD26_345_61243.html, Stand 2026-09-30], 730 h Betrieb je Monat):

| Komponente | Preis (Annahme) | Leistung | Quelle |
|---|---|---|---|
| RTX 3060 Ti 8 GB | 255 € | 200 W | resaleprices.com, eBay-US-Angebote, Durchschnitt 275 USD; umgerechnet mit 0,93 EUR/USD (Annahme); https://resaleprices.com/gpu/nvidia-rtx-3060-ti, Stand 2026-10-25 (Neu nicht mehr erhältlich; kein verlässlicher deutscher Preis gefunden) |
| PC ohne GPU (Ryzen 7 7700, 32 GB, 1 TB, 750 W) | 950 € | 70 W | PC Games Hardware, Spiele-PC-Konfigurationen (17.07.2026), Seiten 3-4, GPU herausgerechnet; Speicherpreise Oktober (PCGH, 18.10.2026); https://www.pcgameshardware.de/Komplett-PC-Hardware-217554/Specials/Gaming-Build-zusammenstellen-Setup-guenstig-kaufen-1408073/, Stand 2026-07-17 (Straßenpreise inkl. MwSt. 1.130 EUR (inkl. 100 EUR Aufschlag für die Speicherpreise im Oktober 2026, Annahme), netto 950 EUR) |
| Raspberry Pi 5 4 GB (Kameras, WLAN, Motorbefehle) | 114,62 € | 6 W | reichelt elektronik, Raspberry Pi 5 B 4 GB (RASP PI 5 B 4GB), inkl. MwSt., lieferbar ab 24.11.2026; https://www.reichelt.com/de/de/shop/search/raspberry%20pi%205%204gb, Stand 2026-10-25 (Derzeit nicht ab Lager; Preis durch die Speicherknappheit 2026 erhöht) |
| Jetson Orin Nano Super 8 GB (Dev Kit) · SmolVLA 567 ms (vla.cpp, C++/GGUF, nicht PyTorch) | 498,40 € | 25 W | reichelt elektronik, inkl. MwSt.; https://www.reichelt.com/de/en/shop/search/jetson%20orin, Stand 2026-10-25; Latenz: vla.cpp, arXiv 2606.08094 v2: 141,8 ms je ausgeführter Aktion bei 4 Aktionen je Aufruf (≈ 567 ms je Aufruf), LIBERO, Stand 2026-09-21 |
| Jetson AGX Orin 64 GB (Dev Kit) · SmolVLA 262 ms (vla.cpp, C++/GGUF, nicht PyTorch) | 3 967,06 € | 50 W | reichelt elektronik, inkl. MwSt.; https://www.reichelt.com/de/en/shop/search/jetson%20agx, Stand 2026-10-25; Latenz: vla.cpp, arXiv 2606.08094 v2: 65,4 ms je ausgeführter Aktion bei 4 Aktionen je Aufruf (≈ 262 ms je Aufruf), Leistungsmodus nicht angegeben, Stand 2026-09-21 |
| Jetson AGX Thor (Dev Kit) · SmolVLA 50 ms (Laufzeitumgebung nicht angegeben) | 5 713,61 € | 70 W | reichelt elektronik, inkl. MwSt.; MyBotShop 6.699,95 EUR; https://www.mybotshop.de/NVIDIA-Jetson-AGX-Thor-Developer-Kit_1, Stand 2026-10-25; Latenz: Robion, arXiv 2609.12075: SmolVLA 50 ms auf Thor bei 70 W, Stand 2026-09-10 |

**Rechner im Roboter** (Erfolg aus der gepoolten Kurve der Bedingung bei seiner Inferenzzeit, mit RTC · (1 + r), r = 1,1 %; Kosten je Roboter und Monat):

| Gerät | Bedingung | Latenz | Erfolg | € je Roboter und Monat |
|---|---|---|---|---|
| Jetson Orin Nano Super 8 GB (Dev Kit) | ohne RTC | 567 ms | 24 % | 18,16 € |
| Jetson Orin Nano Super 8 GB (Dev Kit) | mit RTC | 573 ms | 20 % | 18,16 € |
| Jetson AGX Orin 64 GB (Dev Kit) | ohne RTC | 262 ms | 39 % | 118,83 € |
| Jetson AGX Orin 64 GB (Dev Kit) | mit RTC | 265 ms | 38 % | 118,83 € |
| Jetson AGX Thor (Dev Kit) | ohne RTC | 50 ms | 45 % | 170,80 € |
| Jetson AGX Thor (Dev Kit) | mit RTC | 51 ms | 38 % | 170,80 € |

**Brain an der Kapazität (7.3)** (je Roboter und Monat inklusive Client 4,22 €):

| Bedingung | Batching | Strategie | K | schlechtester | € je Roboter und Monat |
|---|---|---|---|---|---|
| ohne RTC | kein Gewinn | b FIFO + Batching | 2 | 37 % | 33,99 € |
| ohne RTC | kein Gewinn | d EDF + Batching | 2 | 39 % | 33,99 € |
| ohne RTC | kein Gewinn | e d + Zugangskontrolle | 2 | 39 % | 33,99 € |
| ohne RTC | optimistisch | b FIFO + Batching | 3 | 36 % | 25,61 € |
| ohne RTC | optimistisch | d EDF + Batching | 3 | 38 % | 25,77 € |
| ohne RTC | optimistisch | e d + Zugangskontrolle | 3 | 38 % | 25,77 € |
| ohne RTC | pessimistisch | b FIFO + Batching | 2 | 38 % | 33,92 € |
| ohne RTC | pessimistisch | d EDF + Batching | 2 | 39 % | 33,99 € |
| ohne RTC | pessimistisch | e d + Zugangskontrolle | 2 | 39 % | 33,99 € |
| ohne RTC | realistisch | a FIFO | 2 | 39 % | 33,99 € |
| ohne RTC | realistisch | b FIFO + Batching | 3 | 36 % | 25,66 € |
| ohne RTC | realistisch | d EDF + Batching | 2 | 39 % | 33,99 € |
| ohne RTC | realistisch | e d + Zugangskontrolle | 2 | 39 % | 33,99 € |
| mit RTC | kein Gewinn | b FIFO + Batching | 2 | 37 % | 34,05 € |
| mit RTC | kein Gewinn | d EDF + Batching | 3 | 35 % | 26,02 € |
| mit RTC | kein Gewinn | e d + Zugangskontrolle | 3 | 35 % | 26,02 € |
| mit RTC | optimistisch | b FIFO + Batching | 9 | 35 % | 12,82 € |
| mit RTC | optimistisch | d EDF + Batching | 9 | 35 % | 12,89 € |
| mit RTC | optimistisch | e d + Zugangskontrolle | 9 | 35 % | 12,89 € |
| mit RTC | pessimistisch | b FIFO + Batching | 3 | 36 % | 25,78 € |
| mit RTC | pessimistisch | d EDF + Batching | 4 | 35 % | 21,73 € |
| mit RTC | pessimistisch | e d + Zugangskontrolle | 4 | 35 % | 21,73 € |
| mit RTC | realistisch | a FIFO | 3 | 36 % | 26,02 € |
| mit RTC | realistisch | b FIFO + Batching | 5 | 35 % | 18,62 € |
| mit RTC | realistisch | d EDF + Batching | 5 | 36 % | 18,80 € |
| mit RTC | realistisch | e d + Zugangskontrolle | 5 | 36 % | 18,80 € |

**Gleicher Erfolg** (beste Strategie je Profil; größte Größe mit Kriterium 1, deren schlechtester Roboter den Erfolg des Geräts erreicht; in Klammern: bis zu der jede kleinere Größe ihn erreicht; alle Strategien in `results/phase4/economics.json`):

| Gerät | Bedingung | Batching | Strategie | Erfolg Gerät | € Gerät | Brain-Größe | € Brain je Roboter | günstiger |
|---|---|---|---|---|---|---|---|---|
| Jetson Orin Nano Super 8 GB (Dev Kit) | ohne RTC | realistisch | b | 24 % | 18,16 € | 16 (16) | 9,22 € | **Brain** |
| Jetson AGX Orin 64 GB (Dev Kit) | ohne RTC | realistisch | b | 39 % | 118,83 € | 1 (1) | 58,07 € | **Brain** |
| Jetson AGX Thor (Dev Kit) | ohne RTC | realistisch | b | 45 % | 170,80 € | 0 (0) | – | **Gerät** |
| Jetson Orin Nano Super 8 GB (Dev Kit) | mit RTC | realistisch | d | 20 % | 18,16 € | 17 (17) | 8,93 € | **Brain** |
| Jetson AGX Orin 64 GB (Dev Kit) | mit RTC | realistisch | d | 38 % | 118,83 € | 1 (1) | 58,14 € | **Brain** |
| Jetson AGX Thor (Dev Kit) | mit RTC | realistisch | d | 38 % | 170,80 € | 0 (0) | – | **Gerät** |
| Jetson Orin Nano Super 8 GB (Dev Kit) | ohne RTC | optimistisch | d | 24 % | 18,16 € | 43 (43) | 6,08 € | **Brain** |
| Jetson AGX Orin 64 GB (Dev Kit) | ohne RTC | optimistisch | d | 39 % | 118,83 € | 2 (2) | 33,99 € | **Brain** |
| Jetson AGX Thor (Dev Kit) | ohne RTC | optimistisch | d | 45 % | 170,80 € | 0 (0) | – | **Gerät** |
| Jetson Orin Nano Super 8 GB (Dev Kit) | mit RTC | optimistisch | d | 20 % | 18,16 € | 46 (46) | 5,96 € | **Brain** |
| Jetson AGX Orin 64 GB (Dev Kit) | mit RTC | optimistisch | d | 38 % | 118,83 € | 1 (1) | 58,14 € | **Brain** |
| Jetson AGX Thor (Dev Kit) | mit RTC | optimistisch | d | 38 % | 170,80 € | 0 (0) | – | **Gerät** |

**Entscheidungsregel** (beste Strategie = größte Kapazität, Gleichstand d, e, b, a):

| Batching | Bedingung | Brain | Referenzgerät | Brain günstiger | K ≥ 4 |
|---|---|---|---|---|---|
| realistisch | ohne RTC | b, K = 3, schlechtester 36 %, 25,66 € | Jetson AGX Orin 64 GB (Dev Kit) (39 %, 118,83 €) | ja | nein |
| realistisch | mit RTC | d, K = 5, schlechtester 36 %, 18,80 € | Jetson AGX Orin 64 GB (Dev Kit) (38 %, 118,83 €) | ja | ja |
| optimistisch | ohne RTC | d, K = 3, schlechtester 38 %, 25,77 € | Jetson AGX Orin 64 GB (Dev Kit) (39 %, 118,83 €) | ja | nein |
| optimistisch | mit RTC | d, K = 9, schlechtester 35 %, 12,89 € | Jetson AGX Orin 64 GB (Dev Kit) (38 %, 118,83 €) | ja | ja |

**Urteil (Abschnitt 8): trägt**

![Kosten](phase4_kosten.png)
