# Wochenbericht – State Interpreter

## 1. Ziel der Arbeit

Ziel dieser Woche war es, einen ersten State Interpreter für niedrigdimensionale Schwingungsrepräsentationen aufzubauen und seine Übertragbarkeit zwischen zwei Betriebsbedingungen zu untersuchen.

Der State Interpreter soll die internen Embeddings eines Encoders nicht direkt als Zustand verwenden, sondern daraus wenige verständliche und zeitlich konsistente Zustandsgrößen ableiten. Langfristig soll dieser Ansatz auf die Embeddings von VibFM übertragen werden. Für die erste Untersuchung wurde bewusst ein kleiner AutoEncoder verwendet, damit der latente Raum einfacher erzeugt, analysiert und kontrolliert werden kann.

Der aktuelle Ablauf ist:

```text
Schwingungssignal
→ STFT-Zeit-Frequenz-Darstellung
→ kleiner Convolutional AutoEncoder
→ acht-dimensionales Embedding z
→ State Interpreter
→ [Level, Trend, Movement]
```

## 2. Projektstruktur

Für die Entwicklung wurde ein eigenständiges `state-interpreter`-Repository aufgebaut. Dadurch bleiben die Repräsentationsanalyse und die Zustandsinterpretation klar vom bestehenden Gearbox-/RL-Projekt getrennt.

Die aktuelle Modulstruktur ist:

```text
Dataset Adapter
→ STFT Preprocessor
→ AutoEncoder
→ Embedding z
→ State Interpreter
→ Experiment und Evaluation
```

Die Schnittstellen wurden bewusst allgemein gehalten:

- AutoEncoder-Eingabe: `[batch, 2, 32, 32]`,
- AutoEncoder-Ausgabe: `[batch, 8]`,
- State-Interpreter-Eingabe: zeitlich geordnete acht-dimensionale Embeddings,
- State-Interpreter-Ausgabe: `[Level, Trend, Movement]`.

Der State Interpreter ist damit nicht von internen Details des AutoEncoders abhängig. Später kann der Encoder durch VibFM ersetzt werden, ohne die gesamte Interpretationslogik neu zu implementieren.

## 3. Datenbasis und Signalvorverarbeitung

### 3.1 Datenbasis

Als Datengrundlage wurde der XJTU-SY-Lagerdatensatz verwendet. Er enthält zeitlich geordnete Schwingungsmessungen mehrerer Lager von einem frühen Betriebszustand bis zum Ausfall. Dadurch kann untersucht werden, wie sich ein Punkt im latenten Raum während der Degradation bewegt.

Die zweikanaligen Rohsignale wurden mit einer Short-Time Fourier Transform in Zeit-Frequenz-Darstellungen umgewandelt. Die aktuell verwendete Vorverarbeitung besteht aus:

- Hann-Fenster,
- `n_fft = 1024`,
- `win_length = 1024`,
- `hop_length = 512`,
- logarithmischer Magnitudenkompression,
- Ausgabeformat `[2, 32, 32]`.

Die Normalisierungsstatistiken werden ausschließlich auf den jeweiligen Trainingslagern bestimmt, um eine Informationsleckage aus Validierungs- oder Testdaten zu vermeiden.

### 3.2 Datenprüfung und Adapter

Vor dem Modelltraining wurden die reale Verzeichnis- und CSV-Struktur geprüft. Dabei wurden insbesondere folgende Punkte kontrolliert:

- vorhandene Betriebsbedingungen und Lager,
- zweikanalige Messstruktur,
- Vollständigkeit der Messdateien,
- numerische zeitliche Reihenfolge der CSV-Dateien,
- Zuordnung von Betriebsbedingung, Lager, Episode und Messschritt.

Der implementierte Dataset Adapter liest die Dateien nach ihrer numerischen Messnummer und nicht nach lexikographischer Dateireihenfolge. Dadurch bleibt die tatsächliche zeitliche Reihenfolge der Degradation erhalten.

## 4. AutoEncoder und latenter Raum

### 4.1 Modellaufbau

Ein kleiner Convolutional AutoEncoder wurde auf dem XJTU-SY-Datensatz trainiert. Der Encoder komprimiert jede Zeit-Frequenz-Darstellung in einen acht-dimensionalen latenten Vektor:

```text
Encoder: [2, 32, 32] → z ∈ R⁸
Decoder: z → rekonstruierte Zeit-Frequenz-Darstellung
```

Der Decoder wird für das Rekonstruktionstraining benötigt. Für den State Interpreter wird anschließend nur das Embedding `z` verwendet. Eine gute Rekonstruktion allein beweist jedoch noch nicht, dass `z` einen Gesundheitszustand repräsentiert. Deshalb wird die zeitliche Struktur der Embeddings separat interpretiert.

### 4.2 Erste Betriebsbedingung

Die erste technische Baseline wurde unter `35 Hz / 12 kN` aufgebaut. Die Lager wurden getrennt verwendet:

```text
Training:     Bearing1_1, Bearing1_2, Bearing1_3
Validierung:  Bearing1_4
Evaluation nach Parametersperrung: Bearing1_5
```

Nach dem Training wurden die Embeddings in zeitlicher Reihenfolge extrahiert. Zusätzlich wurden Rekonstruktionsfehler, Trajektorien im latenten Raum und mehrere mögliche Gesundheitsindikatoren untersucht. Dabei zeigte sich, dass die Embeddings eine zeitliche Struktur enthalten. Dies motivierte die Entwicklung eines eigenen Interpreters, beweist aber noch keine physikalische Gesundheitssemantik.

## 5. Implementierung des State Interpreters

### 5.1 Zustandsdefinition

Aus den zeitlich geordneten Embeddings werden drei interpretierbare Zustandsgrößen abgeleitet:

- **Level:** relative Entfernung des aktuellen Embeddings vom anfänglichen Referenzzustand des jeweiligen Lagers;
- **Trend:** zeitliche Entwicklung des Levels innerhalb eines kausalen Fensters;
- **Movement:** kurzfristige Bewegung zwischen aufeinanderfolgenden Punkten im latenten Raum.

Vor der Zustandsausgabe besitzt der Interpreter eine Kalibrierungsphase. Die ersten 15 Messungen eines Lagers werden verwendet, um einen individuellen Referenzzustand zu bestimmen. Danach wird mit einem zeitlichen Fenster von 10 Messpunkten gearbeitet:

```text
calibration_steps = 15
temporal_window = 10
state = [Level, Trend, Movement]
```

Diese Parameter wurden vor der aktuellen Vergleichsauswertung festgelegt und anschließend nicht mehr an die Ergebnisse der zweiten Betriebsbedingung angepasst.

### 5.2 Parameterauswahl und erste Evaluation

Für die Auswahl von Kalibrierungslänge und Zeitfenster wurden mehrere Kriterien betrachtet:

- Spearman-Korrelation zwischen Level und normalisierter Lebensdauer,
- gerichtete Monotonie,
- Glattheit des Levels,
- Rauschen des Trends,
- Anzahl der READY-Zustände,
- Höhe und Position von Movement-Spitzen.

Nach der Parametersperrung erreichte `Bearing1_5` eine Level-Lebensdauer-Korrelation von `ρ = 0,950`. Dieses Lager war jedoch bereits in früheren explorativen Analysen sichtbar. Das Ergebnis ist daher eine Evaluation nach der Parameterauswahl, aber kein während des gesamten Projekts vollständig ungesehener Blind-Test.

## 6. Versuchsdesign für die zweite Betriebsbedingung

### 6.1 Vergleich zweier Arme

Für die Zielbetriebsbedingung `37.5 Hz / 11 kN` wurden zwei Varianten verglichen:

- **Arm A – direkte Generalisierung:** Der auf `35 Hz / 12 kN` trainierte AutoEncoder und seine ursprüngliche Normalisierung werden ohne weiteres Training auf die Zielbedingung übertragen.
- **Arm B – Encoder-Anpassung:** Dieselbe AutoEncoder-Architektur wird auf der Zielbedingung neu trainiert. Der State Interpreter und seine Parameter bleiben unverändert.

Die Daten der Zielbedingung wurden vorab lagerweise aufgeteilt:

```text
Training:     Bearing2_1, Bearing2_2, Bearing2_3
Validierung:  Bearing2_4
Blind-Test:   Bearing2_5
```

Diese zwei Varianten trennen zwei unterschiedliche Fragestellungen:

1. Kann das vollständige ursprüngliche System direkt auf eine neue Betriebsbedingung generalisieren?
2. Kann derselbe State Interpreter weiterverwendet werden, wenn nur der Encoder an die neue Betriebsbedingung angepasst wird?

### 6.2 Schutz vor Datenleckage

Die Datenaufteilung, Interpreter-Parameter und Hauptmetrik wurden vor der abschließenden Blind-Evaluation in einer schreibgeschützten Konfiguration festgelegt. Beim Start eines Experiments werden unter anderem folgende Bedingungen geprüft:

- Trainings-, Validierungs- und Blind-Lager sind disjunkt;
- `Bearing2_5` darf nicht in Entwicklungs- oder Trainingsläufen verwendet werden;
- die Normalisierung von Arm B wird nur auf den Trainingslagern angepasst;
- Arm A verwendet weiterhin den eingefrorenen ursprünglichen Checkpoint;
- Konfigurations- und Checkpoint-Hashes stimmen mit den registrierten Werten überein;
- die Interpreter-Parameter bleiben bei 15 und 10;
- nach der Blind-Evaluation ist keine erneute Anpassung anhand des Blind-Ergebnisses vorgesehen.

Konfiguration, Ausführungsstatus und Ergebnisse werden getrennt gespeichert:

```text
configs/  → festgelegte Versuchskonfigurationen
records/  → Status abgeschlossener Versuchsschritte
runs/     → maschinelle Ergebnisartefakte
```

### 6.3 Training des angepassten Encoders

Der AutoEncoder von Arm B wurde fünf Epochen auf `Bearing2_1` bis `Bearing2_3` trainiert. `Bearing2_4` wurde nur zur Validierung und Auswahl des Checkpoints verwendet.

- Trainingsdaten: 1.185 Messungen,
- Validierungsdaten: 42 Messungen,
- Trainings-MSE: von `0,791` auf `0,069`,
- bester Validierungs-MSE: `0,285` in Epoche 4.

Da der Validierungsfehler in Epoche 5 leicht auf `0,289` anstieg, wurde der Checkpoint aus Epoche 4 gespeichert. Die Trainingsdauer oder Hyperparameter wurden danach nicht anhand der Ergebnisse verändert.

## 7. Ergebnisse von Arm A und Arm B

### 7.1 Hauptmetrik

Als Hauptmetrik wurde die Spearman-Rangkorrelation zwischen dem berechneten `Level` und der normalisierten Lebensdauer verwendet. Eine hohe positive Korrelation bedeutet, dass das Level im Allgemeinen mit fortschreitender Lebensdauer beziehungsweise Degradation ansteigt.

### 7.2 Vergleichsergebnisse

| Bearing | Arm A: Spearman-ρ | Arm B: Spearman-ρ | Differenz |
|---|---:|---:|---:|
| Bearing2_1 | 0,887 | 0,914 | +0,026 |
| Bearing2_2 | 0,991 | 0,994 | +0,003 |
| Bearing2_3 | 0,974 | 0,979 | +0,005 |
| Bearing2_4 | 0,681 | 0,993 | +0,312 |

Der angepasste Encoder aus Arm B erzielte auf allen Entwicklungs-Bearings mindestens gleich gute Ergebnisse. Die durchschnittliche Veränderung gegenüber Arm A betrug `Δρ = +0,087`. Die deutlichste Verbesserung zeigte sich bei `Bearing2_4`, dessen Korrelation von `0,681` auf `0,993` anstieg.

Die Ergebnisse der ersten drei Lager unterscheiden sich nur gering zwischen beiden Armen. Dies deutet darauf hin, dass der ursprüngliche Encoder dort bereits eine brauchbare zeitliche Struktur erzeugt. Bei `Bearing2_4` scheint die Anpassung des Encoders an die Zielbetriebsbedingung dagegen besonders relevant zu sein.

## 8. Gemeinsamer Evaluationsablauf

### 8.1 Technischer Aufbau

Für die spätere Blind-Evaluation wurde ein gemeinsames Evaluationsskript implementiert. Beide Arme erhalten dieselbe einmalig gelesene und vorverarbeitete Messsequenz. Anschließend verwendet jeder Arm seinen eigenen eingefrorenen Encoder und seine eigene Normalisierung, während der State Interpreter identisch bleibt.

Der Ablauf wurde zunächst mit dem bereits verwendeten `Bearing2_4` erprobt. Dadurch konnte die technische Funktion geprüft werden, ohne den Blind-Holdout zu öffnen.

### 8.2 Nicht blinde Probedurchführung

Die gemeinsame Evaluation auf `Bearing2_4` verwendete 42 Rohmessungen, die nur einmal materialisiert wurden. Daraus entstanden 42 Zustandszeilen pro Arm beziehungsweise 84 gemeinsame Zustandszeilen. Die zuvor getrennt berechneten Korrelationen wurden exakt reproduziert:

```text
Arm A: ρ = 0,681319
Arm B: ρ = 0,993284
Δρ = +0,311966
```

Damit ist bestätigt, dass die Integration beider Arme den Berechnungsablauf nicht verändert hat.

### 8.3 Sicherheitsmechanismus

Das gemeinsame Skript besitzt zwei getrennte Modi:

```text
rehearsal → verwendet ausschließlich Bearing2_4
blind     → benötigt ein explizites einmaliges Bestätigungstoken
```

Ohne diese Bestätigung beendet sich der Blind-Modus vor dem Lesen der Daten. Aktuell wurde `Bearing2_5` weder für Signalstatistiken noch für Embeddings oder Zustandsausgaben geöffnet.

## 9. Interpretation und Einschränkungen

Die bisherigen Ergebnisse unterstützen vorläufig die Annahme, dass der relative zeitliche State Interpreter auch auf einem neu gelernten latenten Raum verwendet werden kann. Sie zeigen jedoch noch nicht, dass `Level` einem physikalisch messbaren Schadensgrad entspricht.

Außerdem sind die bisherigen Resultate Entwicklungsergebnisse:

- `Bearing2_1` bis `Bearing2_3` wurden zum Training von Arm B verwendet.
- `Bearing2_4` wurde zur Auswahl des besten Checkpoints verwendet.
- Die Rekonstruktionsfehler von Arm A und Arm B sind wegen unterschiedlicher Normalisierungen nicht direkt als Qualitätsvergleich interpretierbar.
- Mit nur einem verbleibenden Blind-Lager ist die spätere Aussage weiterhin eine Fallstudie und keine allgemeine statistische Validierung.

Die Ergebnisse dürfen deshalb aktuell nur als technische und methodische Evidenz für den Ansatz interpretiert werden.

## 10. Aktueller Stand und nächster Schritt

- Der AutoEncoder, der State Interpreter und beide experimentellen Varianten sind implementiert.
- Die Datenaufteilung, Interpreter-Parameter und Auswertungsmetrik wurden vor der Blind-Evaluation festgelegt.
- Alle aktuellen Softwaretests wurden erfolgreich ausgeführt (`66 passed`).
- Der gemeinsame Evaluationsablauf wurde erfolgreich getestet.
- `Bearing2_5` wurde noch nicht ausgewertet und bleibt als Blind-Holdout erhalten.
- Der nächste Schritt ist nach erneuter Bestätigung die einmalige gemeinsame Blind-Evaluation von Arm A und Arm B auf `Bearing2_5`.
- Nach dieser Evaluation dürfen die Parameter nicht anhand des Blind-Ergebnisses erneut angepasst werden.
