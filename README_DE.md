# State Interpreter

[English](README_EN.md) | **Deutsch** | [Chinesische Dokumentation](docs/README.md)

Eigenständiger Forschungscode zum Lernen kompakter Schwingungsrepräsentationen und zu deren Abbildung auf interpretierbare Zustände für Prognostics and Health Management (PHM) und Reinforcement Learning.

## Aktueller Forschungspfad

```text
XJTU-SY-Schwingungsdaten
→ kleiner Convolutional AutoEncoder
→ z (8 oder 16 Dimensionen)
→ State Interpreter im latenten Raum
→ spätere Übertragung auf VibFM
```

Der AutoEncoder ist ein austauschbares vorgelagertes Modul. Der Interpreter verarbeitet einen standardisierten Embedding-Tensor und darf nicht von internen Details des AutoEncoders abhängen. Dadurch bleibt der spätere Austausch `AutoEncoder → VibFM` auf den Encoder-Adapter begrenzt.

## Aktuelle Baseline

```text
zeitlich geordnetes z
→ episodespezifische anfängliche Kalibrierung
→ relative Distanz + kausale zeitliche Merkmale
→ [level, trend, movement]
```

`RelativeTemporalStateInterpreter` ist die aktuelle unüberwachte Online-Baseline. Er besitzt einen expliziten Lebenszyklus `CALIBRATING → READY`, gibt während der Kalibrierung keinen formalen Zustand aus und benötigt vor jeder neuen Episode einen Aufruf von `reset()`. Der vorhandene `MLPStateInterpreter` bleibt eine formgeprüfte Option für spätere überwachte oder Multi-Task-Experimente, ist aber nicht die ausgewählte erste Baseline.

Der erste technische Lauf mit XJTU-SY umfasst inzwischen einen trainierten `z=8`-AutoEncoder, die zeitlich geordnete Extraktion latenter Vektoren, den Vergleich von Zustandsindikatoren und die Online-Wiedergabe von fünf Lagern. Diese Ergebnisse belegen nur die technische Machbarkeit und zeitliche Konsistenz; sie beweisen nicht, dass die Ausgabe einem physikalischen Gesundheitszustand entspricht.

```python
from state_interpreter import RelativeTemporalStateInterpreter

interpreter = RelativeTemporalStateInterpreter(
    embedding_dim=8,
    calibration_steps=15,
    temporal_window=10,
)

for z_t in ordered_embeddings:
    output = interpreter.update(z_t)
    if output is not None:
        state_t = output.state  # [level, trend, movement]
```

Die erste Sensitivitätsanalyse ausschließlich auf dem Validierungsdatensatz legte diese Parameter vor der vorgesehenen Holdout-Auswertung fest. Mit unveränderten Parametern erreichte `Bearing1_5` über 37 READY-Zustände eine Spearman-Korrelation von `0.950` für Level. Dabei handelt es sich um einen Holdout nach der Parameterauswahl, jedoch nicht um einen während des gesamten Projekts strikt ungesehenen Blind-Holdout, da er bereits in früheren explorativen Analysen vorkam. Der maschinenlesbare Entscheidungsdatensatz liegt unter `configs/xjtu_z8_interpreter_v1.json`; die Auswertungsartefakte befinden sich in `runs/locked_holdout_bearing1_5_20260802/`.

Das nächste Protokoll wurde vor der Blind-Auswertung ergänzt und ist in `configs/xjtu_cross_condition_v2_1.json` präregistriert. Es vergleicht zwei eingefrorene Zweige unter `37.5Hz11kN`:

- Arm A: direkte Generalisierung mit dem ursprünglichen AutoEncoder und der ursprünglichen Normalisierung;
- Arm B: derselbe Interpreter, nachdem dieselbe `z=8`-AutoEncoder-Architektur auf `Bearing2_1`–`Bearing2_3` neu trainiert wurde.

`Bearing2_4` dient der Entwicklung und AutoEncoder-Validierung. `Bearing2_5` bleibt für eine einzige gemeinsame Blind-Auswertung beider Zweige reserviert. Die schreibgeschützte Konfiguration und die Schutzmechanismen gegen Datenleckage sind in `state_interpreter.experiment_config` implementiert. Abgeschlossene Ausführungsschritte werden getrennt unter `records/` protokolliert, ohne die gesperrte Konfiguration umzuschreiben.

Die Entwicklungsinferenz von Arm A ist in `scripts/run_direct_generalization_v2.py` implementiert. Auf `Bearing2_1`–`Bearing2_4` erzielte das eingefrorene Ursprungssystem positive Spearman-Korrelationen zwischen Level und Lebensdauer von `0.887`, `0.991`, `0.974` und `0.681`. Dies sind ausschließlich Entwicklungsergebnisse; `Bearing2_5` wurde für die abschließende gemeinsame Blind-Auswertung noch nicht gelesen.

Das Training von Arm B für die Zielbetriebsbedingung ist in `scripts/train_adapted_autoencoder_v2.py` implementiert. Die Normalisierung wurde auf `Bearing2_1`–`Bearing2_3` angepasst und dieselbe `z=8`-Architektur darauf trainiert. Epoche 4 wurde anhand des Validierungs-MSE von `0.285` auf `Bearing2_4` ausgewählt. `Bearing2_5` wurde nicht gelesen. Der angepasste Checkpoint liegt unter `runs/arm_b_target_autoencoder_20260802/`.

Die Entwicklungsinferenz von Arm B ist in `scripts/run_adapted_encoder_development_v2.py` implementiert. Mit dem angepassten Encoder und dem weiterhin eingefrorenen Interpreter ergaben sich für `Bearing2_1`–`Bearing2_4` Spearman-Korrelationen von `0.914`, `0.994`, `0.979` und `0.993`. Gegenüber Arm A betragen die Änderungen je Lager `+0.026`, `+0.003`, `+0.005` und `+0.312`. Auch dies sind Entwicklungsergebnisse; `Bearing2_5` bleibt bis zur einmaligen gemeinsamen Blind-Auswertung ungelesen.

Der gemeinsame Ablauf ist in `scripts/run_joint_evaluation_v2.py` implementiert und wurde mit dem bereits verwendeten `Bearing2_4` erprobt. Beide Zweige nutzten dieselbe einmalige Daten-/STFT-Materialisierung und reproduzierten ihre vorherigen Korrelationen exakt (`0.681319` und `0.993284`). Der Rehearsal-Modus kann den Holdout nicht auswählen; der Blind-Modus verlangt ein explizites einmaliges Bestätigungstoken. `Bearing2_5` wurde weiterhin nicht gelesen.

## Entwicklungsumgebung

```powershell
python -m pip install -e .[dev]
python -m pytest -q
```

## Smoke-Test der XJTU-SY-Rohdaten

Der Datensatz-Adapter liest rohe CSV-Dateien in numerischer Messreihenfolge und gibt zweikanalige Tensoren aus, ohne STFT oder Normalisierung anzuwenden:

```powershell
python scripts/smoke_test_xjtu_sy.py `
  --root "D:\path\to\XJTU-SY_Bearing_Datasets" `
  --condition 35Hz12kN `
  --bearing Bearing1_1 `
  --limit 2
```

## Kontrolle der Log-STFT

Die erste Vorverarbeitungs-Baseline verwendet ein Hann-Fenster, `n_fft=1024`, `win_length=1024`, `hop_length=512`, eine `log1p`-Kompression der Magnitude und adaptives Average Pooling auf `[2, 32, 32]`. In diesem Schritt wird keine datensatzweite Normalisierung angewendet.

```powershell
python scripts/inspect_xjtu_stft.py `
  --root "D:\path\to\XJTU-SY_Bearing_Datasets" `
  --condition 35Hz12kN `
  --bearing Bearing1_1 `
  --output outputs/xjtu_stft_bearing1_1.png
```

## Erste AutoEncoder-Baseline

Die erste technische Baseline verwendet eine Betriebsbedingung und eine lagerweise Aufteilung: Bearings 1_1–1_3 für das Training, 1_4 für die Validierung und 1_5 als unberührten Holdout. Die kanalweise Normalisierung wird ausschließlich auf den Trainingslagern angepasst.

```powershell
python scripts/train_autoencoder_baseline.py `
  --root "D:\path\to\XJTU-SY_Bearing_Datasets" `
  --output-dir "D:\path\to\run-output" `
  --condition 35Hz12kN `
  --latent-dim 8 `
  --epochs 5
```

## Integrationsgrenze

- `SmallConvAutoEncoder` akzeptiert `[batch, 2, 32, 32]` und stellt `encode(x)` bereit.
- Der State Interpreter akzeptiert Embeddings der Form `[batch, embedding_dim]`.
- Datensatz-, VibFM- und Gearbox-spezifischer Code soll als Adapter implementiert werden, statt in das Kernmodell importiert zu werden.
