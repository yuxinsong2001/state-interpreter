# Kurzbericht: Vom VibFM-Embedding zum RL-tauglichen Zustand

Datum: 27.07.2026

## Folie 1 – Titel

### Sichtbarer Inhalt

**Vom VibFM-Embedding zum RL-tauglichen Zustand**  
Literaturbasierte Interpreter-Ideen und eine schrittweise Versuchsstrategie

### Sprechtext

Guten Morgen. Diese Woche habe ich mich vor allem mit der Frage beschäftigt, wie aus dem Gesundheits-Embedding von VibFM ein interpretierbarer und für Reinforcement Learning nutzbarer Zustand entstehen kann. Der Schwerpunkt meines Berichts liegt auf den in der Literatur gefundenen Interpreter-Ideen und auf der Frage, wie wir zwischen diesen Ideen systematisch auswählen können.

## Folie 2 – Forschungsfrage

### Sichtbarer Inhalt

**z_health ist ein gutes Embedding. Aber ist es schon ein guter RL-Zustand?**

```text
Vibration → VibFM → z_health → State Interpreter → Wartungsentscheidung
```

### Sprechtext

Der Encoder ist in diesem Projekt festgelegt: Wir verwenden VibFM. Deshalb ist die zentrale Frage nicht mehr, welcher Encoder am besten ist. Die Frage lautet, ob `z_health` bereits alle Informationen enthält, die ein Wartungsagent benötigt. Ein Embedding kann gesunde und geschädigte Beispiele trennen und trotzdem wichtige Informationen über Trend, Unsicherheit, Last oder zukünftige Zustandsübergänge verlieren.

## Folie 3 – Vier Interpreter-Ideen

### Sichtbarer Inhalt

1. Gesundheitsabstand: kontinuierlicher Health Indicator  
2. Zustandscluster: Degradationsphasen  
3. Zeitmodell: Trend und Hidden State  
4. Entscheidungszustand: Reward- und Transition-Relevanz

### Sprechtext

Da der Begriff State Interpreter in der PHM-Literatur nicht einheitlich verwendet wird, habe ich unter verwandten Begriffen gesucht: Health Indicator, Degradation Stage, Hidden Health State und State Representation Learning for Control. Daraus lassen sich vier Hauptideen ableiten.

Erstens kann der Abstand zur gesunden Region als kontinuierlicher Health Indicator dienen. Zweitens können Cluster im Embedding-Raum als Degradationsphasen interpretiert werden. Drittens können Zeitmodelle Verlauf und Übergänge berücksichtigen. Viertens kann der Zustand explizit auf Reward- und Transitionsinformation ausgerichtet werden.

## Folie 4 – Einordnung

### Sichtbarer Inhalt

**Die vier Ansätze sind keine vier gleichartigen Wettbewerber.**

```text
Gesundheitssemantik:
- kontinuierlicher HI
- diskrete Degradationsphase

Zustandssuffizienz:
- reicht ein Zeitfenster?
- bleiben Reward und Dynamik erhalten?
```

### Sprechtext

Das ist für die Versuchsplanung besonders wichtig: Die ersten beiden Ansätze definieren hauptsächlich, was der Gesundheitszustand bedeutet. Ein HI liefert eine kontinuierliche Semantik, Cluster liefern diskrete Phasen. Zeitmodelle und entscheidungsorientierte Modelle prüfen dagegen, ob dieser Zustand ausreichend ist.

Deshalb wäre es methodisch falsch, alle vier Varianten einfach zu implementieren und nur anhand der Klassifikationsgenauigkeit zu vergleichen.

## Folie 5 – Progressive Strategie

### Sichtbarer Inhalt

```text
Basis
→ Zeitprüfung
→ Zeitmodell
→ RL-Prüfung
```

Ziel: das einfachste Modell, das für die Wartungsentscheidung ausreicht.

### Sprechtext

Daraus ergibt sich eine progressive Strategie. Zuerst vergleiche ich einfache Baselines: das unveränderte `z_health`, einen One-Class Health Indicator, Cluster-Phasen und ein kleines MLP mit HI und Phasenwahrscheinlichkeiten.

Danach prüfe ich die zeitliche Stabilität und den zusätzlichen Nutzen der Historie. Erst wenn die Historie einen messbaren Vorteil bringt, werden Change-Point Detection oder ein Left-right-HMM und anschließend gegebenenfalls GRU oder TCN getestet. Die letzte Stufe ist die RL-Prüfung mit Reward, Transition und Wartungskosten.

## Folie 6 – Entscheidungskriterien

### Sichtbarer Inhalt

Einfache Lösung beibehalten, wenn:

- HI und Phasen zeitlich stabil sind;
- Historie keinen Zusatznutzen liefert;
- der Zustand Reward und Dynamik ausreichend erklärt.

Komplexität erhöhen, wenn:

- Phasen häufig rückwärts springen;
- Historie Next State oder Reward verbessert;
- ähnliche Zustände unterschiedliche Aktionen erfordern.

### Sprechtext

Die nächste Modellstufe wird nicht nach Modellnamen ausgewählt, sondern nach Diagnoseergebnissen. Wenn HI und Phasen stabil sind und die Historie keinen Zusatznutzen liefert, bleibt das einfache Modell bestehen.

Wenn Zustände häufig rückwärts springen, kann zunächst ein Change-Point-Verfahren oder ein Left-right-HMM helfen. Wenn die Historie die Vorhersage von Reward oder nächstem Zustand deutlich verbessert, ist ein GRU oder TCN begründet. Attention wird nur dann behalten, wenn es darüber hinaus einen messbaren Mehrwert liefert.

## Folie 7 – Diese Woche

### Sichtbarer Inhalt

- eigenständiges State-Interpreter-Repository;
- erstes MLP-Interface;
- einheitliche Datenverträge;
- Adapter-Schnittstellen für VibFM, Datensatz und Gearbox;
- sechs Softwaretests bestanden.

### Sprechtext

Als Vorbereitung habe ich ein eigenständiges State-Interpreter-Repository initialisiert. Darin gibt es ein erstes MLP-Interface, gemeinsame Datenverträge für Messung, Embedding und Transition sowie Adapter-Schnittstellen für VibFM, reale Datensätze und die Gearbox-Umgebung.

Sechs Softwaretests sind erfolgreich. Wichtig ist die Abgrenzung: Das sind Schnittstellen- und Formtests mit künstlichen Embeddings. Ich habe noch keine realen VibFM-Embeddings extrahiert und noch kein Modell trainiert.

## Folie 8 – Nächste Schritte

### Sichtbarer Inhalt

1. VibFM-Checkpoint und Vorverarbeitung klären;  
2. ersten Run-to-Failure-Datensatz auswählen;  
3. zeitlich geordnete reale `z_health` extrahieren;  
4. Identity, HI, Cluster und MLP vergleichen.

### Sprechtext

Der nächste Schritt ist der Übergang von der Schnittstelle zum ersten echten Baseline-Experiment. Dafür brauche ich den pretrained VibFM-Checkpoint, die dazugehörige Codeversion und die Vorverarbeitungsparameter.

Danach wähle ich einen Run-to-Failure-Datensatz, extrahiere zeitlich geordnete reale `z_health`-Embeddings und vergleiche die vier einfachen Baselines. Von Ihnen würde ich gern bestätigen lassen, ob die erste State-Version sowohl einen kontinuierlichen HI als auch Phasenwahrscheinlichkeiten ausgeben soll und welcher Datensatz zuerst verwendet werden soll.
