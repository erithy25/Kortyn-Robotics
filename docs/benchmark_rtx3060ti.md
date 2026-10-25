# Anleitung: SmolVLA-Benchmark auf deiner RTX 3060 Ti (Windows + WSL2)

Diese Anleitung misst, wie lange deine Grafikkarte für eine SmolVLA-Inferenz braucht, wenn sie 1, 2, 3 … 32
Roboter-Anfragen gleichzeitig (als „Batch“) bearbeitet. Das Ergebnis ist eine kleine JSON-Datei. Kortyn liest
sie direkt ein und ersetzt damit die geschätzten Batching-Profile durch echte Messwerte deiner Karte.

**Du brauchst keine Programmierkenntnisse.** Du kopierst Befehle in ein Fenster und drückst Enter.

| | |
|---|---|
| Zeitbedarf | ca. 30 Minuten Einrichtung, 15–40 Minuten Messung (der PC rechnet allein) |
| Speicherplatz | ca. 15 GB frei auf Laufwerk C: |
| Kosten | keine. Alles ist Open Source; ein Konto bei Hugging Face ist nicht nötig |
| Daten | Das Skript lädt nur das Modell herunter (anonym, ohne Telemetrie) und sendet nichts |

## Schritt 1: Windows prüfen

1. Drücke `Windows-Taste + R`, tippe `winver` und drücke Enter.
2. Du brauchst **Windows 11** oder **Windows 10 Version 21H2 oder neuer**. Ist deine Version älter, führe
   zuerst Windows Update aus.

## Schritt 2: NVIDIA-Treiber aktualisieren (unter Windows)

1. Öffne <https://www.nvidia.com/de-de/drivers/> und wähle: GeForce → GeForce RTX 30 Series →
   GeForce RTX 3060 Ti → Windows 11 (bzw. 10).
2. Lade den aktuellen **Game Ready**- oder **Studio**-Treiber herunter und installiere ihn („Express“).
3. Starte den PC neu.

**Wichtig:** Installiere später in Ubuntu **keinen** NVIDIA-Treiber. WSL2 nutzt automatisch den
Windows-Treiber.

## Schritt 3: WSL2 mit Ubuntu installieren

1. Klicke mit der rechten Maustaste auf das Startmenü und wähle **„Terminal (Administrator)“** bzw.
   **„Windows PowerShell (Administrator)“**.
2. Kopiere diesen Befehl hinein und drücke Enter:

   ```
   wsl --install -d Ubuntu-24.04
   ```

3. Starte den PC neu, wenn Windows dich dazu auffordert.
4. Nach dem Neustart öffnet sich ein Fenster „Ubuntu“ (sonst: Startmenü → „Ubuntu“ eintippen → öffnen).
   Es fragt nach einem **Benutzernamen** (z. B. `kortyn`, nur Kleinbuchstaben) und einem **Passwort**. Beim
   Tippen des Passworts erscheinen keine Zeichen, das ist normal. Merke dir das Passwort.

Meldet Windows einen Fehler mit „Virtualisierung“ oder „0x80370102“: Die Virtualisierung ist im BIOS
ausgeschaltet. Schalte im BIOS „Intel VT-x“ bzw. „AMD SVM“ ein; das Handbuch deines Mainboards zeigt, wo.

## Schritt 4: Prüfen, ob Ubuntu die Grafikkarte sieht

Tippe im Ubuntu-Fenster:

```
nvidia-smi
```

Du solltest eine Tabelle sehen, in der **„NVIDIA GeForce RTX 3060 Ti“** steht. Erscheint stattdessen
„command not found“ oder „No devices were found“, ist der Treiber aus Schritt 2 nicht aktuell: Installiere
ihn neu, starte den PC neu und versuche es noch einmal.

## Schritt 5: Werkzeug `uv` installieren

`uv` lädt automatisch die richtige Python-Version und alle Pakete. Kopiere diese zwei Zeilen nacheinander in
das Ubuntu-Fenster:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Prüfe mit `uv --version`: Es erscheint eine Versionsnummer.

## Schritt 6: Das Benchmark-Skript holen

1. Öffne im Browser unter Windows die Datei im Repository:
   <https://github.com/erithy25/Kortyn-Robotics/blob/phase-3-fleet-brain/kortyn/bench_batch.py>
   (Das Repository ist privat: Melde dich bei GitHub mit dem Konto an, das Zugriff hat.)
2. Klicke rechts oben über dem Code auf das Symbol **„Download raw file“** (Pfeil nach unten). Die Datei
   `bench_batch.py` landet in deinem Windows-Ordner „Downloads“.
3. Kopiere sie nach Ubuntu. Ersetze `DEINNAME` durch deinen **Windows**-Benutzernamen (den Ordnernamen unter
   `C:\Users\`):

   ```
   cp /mnt/c/Users/DEINNAME/Downloads/bench_batch.py ~/
   cd ~
   ls bench_batch.py
   ```

   Die letzte Zeile muss `bench_batch.py` ausgeben.

## Schritt 7: Messung starten

1. Schließe vorher Spiele, Videoschnitt und andere Programme, die die Grafikkarte nutzen. Der PC darf in der
   Zeit nicht in den Ruhezustand gehen (Windows-Einstellungen → System → Energie → „Bildschirm und
   Ruhezustand“ → beim Netzbetrieb „Nie“).
2. Starte im Ubuntu-Fenster:

   ```
   uv run bench_batch.py
   ```

3. Beim ersten Mal lädt `uv` Python, PyTorch und LeRobot (ca. 3–4 GB) und danach das Modell (ca. 2 GB). Das
   dauert je nach Internet 10–20 Minuten. Danach misst das Skript jede Batchgröße von 1 bis 32 und schreibt
   eine Zeile je Batchgröße, zum Beispiel:

   ```
     batch  1: median    120.0 ms   p90    125.0 ms   je Anfrage   120.0 ms
     batch  2: median    150.0 ms   p90    157.0 ms   je Anfrage    75.0 ms
   ```

   Die Zahlen hier sind nur ein Beispiel. Erscheint bei großen Batches „kein GPU-Speicher mehr“, ist das
   normal (8 GB Grafikspeicher): Das Skript vermerkt das und überspringt die größeren Batches.
4. Am Ende steht: `Fertig. Ergebnis gespeichert in: /home/…/smolvla_batch_rtx3060ti_JJJJMMTT.json`.

Nur zum Ausprobieren (dauert etwa zwei Minuten): `uv run bench_batch.py --sizes 1-4 --repeats 5`.

## Schritt 8: Ergebnis zurückgeben

Kopiere die Ergebnisdatei in deinen Windows-Ordner „Downloads“:

```
cp ~/smolvla_batch_*.json /mnt/c/Users/DEINNAME/Downloads/
```

Lege die Datei im Repository unter `kortyn/experiments/phase3/measurements/` ab (GitHub-Weboberfläche: Ordner
öffnen → „Add file“ → „Upload files“) oder schick sie mir. Das Flottenmodell liest sie so ein (Eintrag in
`kortyn/experiments/phase3/fleet.yaml` unter `gpus`):

```yaml
gpus:
  rtx3060ti_gemessen: {table: kortyn/experiments/phase3/measurements/smolvla_batch_rtx3060ti_JJJJMMTT.json}
```

Danach rechnen diese zwei Befehle die Kapazität mit deiner gemessenen Karte neu:

```
uv run python -m kortyn.fleet --config kortyn/experiments/phase3/fleet.yaml
uv run python -m kortyn.fleet_predict --config kortyn/experiments/phase3/fleet.yaml
```

## Was die Datei enthält

| Feld | Bedeutung |
|---|---|
| `format` | `kortyn-batch-service-v1` (daran erkennt Kortyn die Tabelle) |
| `device` | Grafikkarte, Grafikspeicher, Treiber- und Softwareversionen |
| `model` | Modell `lerobot/smolvla_libero` mit fester Revision, 10 Flow-Schritte, zwei Kameras 360 × 360 |
| `results` | je Batchgröße: Median und 90. Perzentil der Servicezeit in ms, Zeit je Anfrage, Status (`ok`, `oom`, `skipped`) |

Gemessen wird pro Batch das, was das Brain pro Batch tut: Vorverarbeitung (Normierung, Skalierung der
Bilder, Tokenisierung), die Policy selbst und die Rückgabe der Aktionen an die CPU. Das Netz zum Roboter
gehört nicht dazu, das modelliert Kortyn getrennt.

## Häufige Probleme

| Meldung | Lösung |
|---|---|
| `nvidia-smi: command not found` | Windows-Treiber aktualisieren (Schritt 2), PC neu starten |
| `FEHLER: PyTorch sieht keine NVIDIA-GPU` | wie oben; außerdem in PowerShell `wsl --update` ausführen und Ubuntu neu öffnen |
| `No space left on device` | Auf Laufwerk C: Platz schaffen (mindestens 15 GB frei) |
| `cp: cannot stat '/mnt/c/Users/DEINNAME/…'` | `DEINNAME` durch deinen Windows-Benutzernamen ersetzen; mit `ls /mnt/c/Users/` siehst du die Namen |
| Download bricht ab | Befehl einfach noch einmal starten; Fertiges wird nicht erneut geladen |
| Das Fenster wurde geschlossen | Ubuntu neu öffnen, `cd ~` und `uv run bench_batch.py` erneut starten |
