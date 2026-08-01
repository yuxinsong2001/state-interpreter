# Abschlussfassung des Gruppenberichts vom 27.07.2026

## Titel

**Von VibFM-Embeddings zu RL-tauglichen Zuständen: Literaturübersicht und eine progressive Versuchsstrategie**

## Kernaussage des Berichts

Die unterschiedlichen State-Interpreter-Methoden aus der Literatur sollten nicht als vier konkurrierende Modelle verstanden werden, die sofort vollständig implementiert werden müssen. Stattdessen bilden sie eine Versuchsroute, in der die Fähigkeiten des Zustands schrittweise erweitert werden:

```text
zuerst den einfachsten Zustand aufbauen
        ↓
untersuchen, welche Informationen ihm fehlen
        ↓
Zeit-, Betriebs- oder Entscheidungsbedingungen
nur bei entsprechender Evidenz ergänzen
        ↓
das einfachste Modell auswählen,
das die Aufgabenanforderungen erfüllt
```

---

# Teil 1: Forschungsfrage

## 1. Die zentrale Frage von der Tafel

Das aktuelle System lässt sich folgendermaßen darstellen:

```text
Vibration
    ↓
STFT
    ↓
eingefrorenes VibFM
    ↓
z_health-Embedding
    ↓
State Interpreter
    ↓
RL-Zustand
    ↓
Wartungsentscheidung
```

Meine Forschungsfrage besteht nicht darin, erneut einen Encoder auszuwählen, sondern lautet:

> Wie sollte `z_health` bei einem fest vorgegebenen VibFM-Encoder in einen interpretierbaren und entscheidungsrelevanten Zustand transformiert werden?

Dabei müssen zwei Fragen überprüft werden:

1. Enthält `z_health` bereits Strukturen, die mit Gesundheit und Degradation zusammenhängen?
2. Falls Gesundheitsinformationen enthalten sind: Reichen diese bereits für RL-Entscheidungen aus?

Daher gilt:

```text
z_health ≠ automatisch ein entscheidungsreifer Zustand
```

## 2. Bewertungskriterien aus den vom Betreuer bereitgestellten Materialien

Die beiden bereitgestellten Dokumente helfen vor allem dabei, zu definieren, welche Anforderungen der endgültige Zustand erfüllen sollte:

- kompakt;
- interpretierbar;
- zeitlich konsistent;
- robust;
- annähernd Markov;
- unsicherheitsbewusst;
- für Reward und Aktionen relevant;
- übertragbar.

Die Google-Scholar-Literaturrecherche beantwortet anschließend die Frage:

> Mit welchen Methoden werden diese Fähigkeiten in bisherigen Arbeiten umgesetzt?

---

# Teil 2: Vier Umsetzungsideen aus der Literatur

## 3. Methode 1: Abstand zum gesunden Zustandsraum

Repräsentative Forschungsrichtungen:

- Konstruktion von Health Indicators;
- SVDD;
- One-Class SVM;
- Abstand zu einer gesunden Referenz.

Repräsentative Veröffentlichung:

Zhou et al. (2016), *Bearing Performance Degradation Assessment Using Lifting Wavelet Packet Symbolic Entropy and SVDD*.

Kernidee:

> Anhand von Daten aus der gesunden Betriebsphase wird ein Normalbereich definiert. Der Abstand eines neuen Beispiels von diesem gesunden Bereich beschreibt den kontinuierlichen Degradationsgrad.

Nach der Übertragung auf VibFM:

```text
gesundes z_health
→ gesunder Embedding-Bereich
→ Abstand vom gesunden Bereich
→ kontinuierlicher HI
```

Die Methode beantwortet:

> Wie weit ist der aktuelle Zustand vom gesunden Zustand entfernt?

Wesentliche Fähigkeiten:

- kontinuierliche Beschreibung des Gesundheitszustands;
- Erkennung anomaler Abweichungen;
- geringer Bedarf an Labels.

Wesentliche Lücken:

- Der HI ist nicht zwangsläufig zeitlich glatt oder monoton;
- die Methode liefert nicht unmittelbar Degradationsphasen;
- entscheidungsrelevante Informationen bleiben nicht zwangsläufig erhalten.

## 4. Methode 2: Clustering und Degradationsphasen

Repräsentative Veröffentlichung:

Juodelyte et al. (2022), *Predicting Bearings' Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry*.

Kernidee:

> Im latenten Raum werden verschiedene Datenregionen entdeckt. Anschließend erhalten die Cluster anhand ihrer zeitlichen Position im Lebenszyklus eine Degradationsbedeutung.

Nach der Übertragung auf VibFM:

```text
zeitlich geordnetes z_health
→ Clustering
→ Cluster nach ihrer Lebenszyklusposition ordnen
→ Pseudo-Labels für Degradationsphasen
→ Phasenwahrscheinlichkeiten
```

Die Methode beantwortet:

> In welcher Degradationsphase befindet sich das aktuelle Beispiel?

Wesentliche Fähigkeiten:

- diskrete und interpretierbare Phasen;
- keine manuelle Kennzeichnung jedes einzelnen Zeitfensters erforderlich;
- Analyse der Struktur des Embedding-Raums.

Wesentliche Lücken:

- Cluster sind Pseudo-Labels und keine physikalische Ground Truth;
- es wird eine kontinuierliche Run-to-Failure-Sequenz benötigt;
- unplausible rückwärtsgerichtete Phasenwechsel sind möglich;
- die Annahme „später bedeutet stärker degradiert“ ist ein zusätzlicher zeitlicher Prior.

## 5. Methode 3: Zeitliche Zustandsmodelle

Repräsentative Forschungsrichtungen:

- Change-Point Detection;
- Left-right-HMM;
- GRU/TCN;
- Attention.

Repräsentative Veröffentlichungen:

- Cartella und Sahli: Continuous Hidden Markov Model;
- Wang et al.: topologische Repräsentation und HMM;
- Singleton et al.: Entdeckung versteckter Gesundheitszustände.

Kernidee:

> Der Gesundheitszustand einer Maschine ist ein versteckter Prozess, der sich mit der Zeit entwickelt. Ein einzelnes aktuelles Zeitfenster reicht möglicherweise nicht aus, um den Trend zu beschreiben.

Nach der Übertragung auf VibFM:

```text
z_health,t-k:t
→ zeitliches Zustandsmodell
→ Gesundheitszustand + Trend
```

Die Methode beantwortet:

> Wie hat sich der aktuelle Zustand bis zu diesem Zeitpunkt entwickelt?

Wesentliche Fähigkeiten:

- Trendinformation;
- Change Points;
- zeitliche Konsistenz;
- Übergänge zwischen versteckten Zuständen.

Wesentliche Lücken:

- es werden kontinuierliche Sequenzen benötigt;
- das Modell ist komplexer;
- bei einer kleinen Zahl von Anlagen kann Overfitting entstehen;
- Attention allein garantiert weder Interpretierbarkeit noch Entscheidungsrelevanz.

## 6. Methode 4: Entscheidungsrelevanter Zustand

Repräsentative Literatur:

- Lesort et al.: State Representation Learning for Control;
- DeepMDP;
- Deep Bisimulation for Control;
- Belief- und selbstprädiktive Repräsentationen.

Kernidee:

> Healthy und Damaged unterscheiden zu können bedeutet noch nicht, dass ein Zustand für RL geeignet ist. Der Zustand sollte zusätzlich Reward-Informationen und aktionsbedingte Zustandsübergänge erhalten.

Bewertungsbeziehungen:

```text
s_t, a_t → reward_t
s_t, a_t → s_(t+1)
```

Mögliche Umsetzung:

```text
z_health / Historie
→ State Interpreter
├── HI
├── Phasenwahrscheinlichkeiten
├── Reward-Vorhersage
└── Next-State-Vorhersage
        ↓
kompakter Entscheidungszustand
```

Die Methode beantwortet:

> Was bedeutet der aktuelle Zustand für die zukünftige Degradation und die Wartungsaktion?

Wesentliche Fähigkeiten:

- Reward-Relevanz;
- aktionsbedingte Dynamik;
- Erkennung von State Aliasing;
- direkte Ausrichtung auf RL.

Wesentliche Lücken:

- Aktionen, Rewards und Transitionen werden benötigt;
- öffentliche Run-to-Failure-Datensätze enthalten in der Regel keine Wartungsaktionen;
- eine RL-Umgebung wird benötigt;
- Trainings- und Bewertungsaufwand sind am höchsten.

---

# Teil 3: Müssen alle vier Methoden implementiert werden?

## 7. Es handelt sich nicht um vier konkurrierende Modelle auf derselben Ebene

Die ersten beiden Methoden definieren hauptsächlich die Gesundheitssemantik des Zustands:

```text
kontinuierliche Semantik: HI
diskrete Semantik: Degradationsphase
```

Die letzten beiden Methoden prüfen hauptsächlich, ob der Zustand ausreichend ist:

```text
zeitliche Suffizienz: Wird Historie benötigt?
Entscheidungssuffizienz:
Bleiben Reward, Aktionen und Transitionen erhalten?
```

Daher wäre das folgende Vorgehen nicht angemessen:

```text
alle vier Modelle vollständig implementieren
→ Klassifikationsgenauigkeit vergleichen
→ das Modell mit der höchsten Genauigkeit auswählen
```

Der Grund ist, dass diese Methoden unterschiedliche Fragen beantworten.

## 8. Empfohlenes Prinzip: Fähigkeiten schrittweise ergänzen

Empfohlene Versuchslogik:

```text
einfacher Zustand
→ Diagnose auf Repräsentationsebene
→ Diagnose der zeitlichen Suffizienz
→ Diagnose der Entscheidungssuffizienz
→ Komplexität nur bei Bedarf erhöhen
```

Das endgültige Prinzip lautet:

> Es wird der einfachste State Interpreter ausgewählt, der für die nachgelagerte Wartungsentscheidung ausreichend ist.

Mit anderen Worten:

> Es wird das einfachste Modell gewählt, das die Anforderungen an die Zustandssuffizienz erfüllt.

## 9. Phase 1: Notwendige und kostengünstige Baselines

### Baseline 0: Identity

```text
Zustand = z_health
```

Ziel:

> Prüfen, ob ein State Interpreter im Vergleich zur direkten Verwendung des VibFM-Embeddings tatsächlich einen Mehrwert bietet.

### Baseline 1: Kontinuierlicher HI

```text
z_health
→ Abstand zum gesunden Zustandsraum
→ HI
```

Ziel:

> Prüfen, ob `z_health` eine kontinuierliche Degradationsrichtung enthält.

In der ersten Runde reicht eine One-Class-Methode aus. One-Class SVM, SVDD und alle möglichen Distanzverfahren müssen nicht gleichzeitig implementiert werden.

### Baseline 2: Degradationsphasen

```text
zeitlich geordnetes z_health
→ Clustering
→ zeitlich geordnete Phasen
```

Ziel:

> Prüfen, ob der Embedding-Raum interpretierbare Phasen bildet.

### Baseline 3: MLP für ein einzelnes Zeitfenster

```text
z_health
→ MLP
→ HI + Phasenwahrscheinlichkeiten
```

Ziel:

> Prüfen, ob ein einzelnes Embedding gleichzeitig die kontinuierliche und diskrete Gesundheitssemantik ausdrücken kann.

Für die erste Phase werden diese vier kostengünstigen Baselines empfohlen, da sie notwendige Vergleichspunkte für alle späteren komplexeren Modelle bilden.

## 10. Phase 2: Entscheidung über HMM, GRU oder Attention

Zunächst werden die Zustände aus Phase 1 untersucht:

- Schwankt der HI im Zeitverlauf stark?
- Treten häufig rückwärtsgerichtete Phasenwechsel auf?
- Kann der Degradationsbeginn nicht stabil erkannt werden?
- Verwechselt ein einzelnes Embedding „stabil gesund“ mit „aktuell schnell degradierend“?

Anschließend wird ein History-Gain-Test durchgeführt:

```text
Modell A:
s_t, a_t → reward_t / nächster Zustand

Modell B:
s_t + Historie, a_t → reward_t / nächster Zustand
```

Wenn die Historie keinen signifikanten Zusatznutzen bringt:

```text
einfaches Einzelzeitfenster-Modell beibehalten
```

Wenn die Historie einen signifikanten Zusatznutzen bringt:

```text
einfache Glättung / CPD
→ Left-right-HMM
→ GRU oder TCN
→ Attention nur bei zusätzlichem Mehrwert
```

GRU und Attention von der Tafel sollten daher als mögliche Richtungen verstanden werden und nicht als Strukturen, die bereits zu Projektbeginn zwingend eingesetzt werden müssen.

## 11. Phase 3: Entscheidung über einen entscheidungsrelevanten Interpreter

Öffentliche Run-to-Failure-Daten ermöglichen die Bewertung von:

- HI;
- Degradationsphasen;
- Trends;
- Change Points;
- zeitlicher Konsistenz.

In der Regel erlauben sie jedoch keine vollständige Bewertung von:

- Wartungsaktionen;
- Reward;
- aktionsbedingten Transitionen;
- Policy-Performance.

Sobald Gearbox oder Maintenance Arena RL-Transitionen bereitstellt, werden folgende Zustände verglichen:

```text
PHM-Zustand:
HI + Phase

prädiktiver Zustand:
HI + Phase
+ Reward-Vorhersage
+ Next-State-Vorhersage
```

Diese zusätzlichen Module werden nur beibehalten, wenn der prädiktive Zustand:

- Reward besser vorhersagt;
- Zustandsänderungen nach einer Aktion besser vorhersagt;
- die Sample Efficiency des RL-Agenten verbessert;
- die Wartungskosten reduziert;
- Ausfälle und Downtime verringert.

---

# Teil 4: Entscheidungsbaum für die gesamte Versuchsroute

## 12. Versuchsroute

```text
reales z_health erhalten
        ↓
Identity + HI + Phasen + Einzelzeitfenster-MLP
        ↓
besitzt die Repräsentation eine Gesundheitssemantik?
        │
  nein ─┴─ ja
   │       ↓
VibFM, Labels und       zeitliche Konsistenz prüfen
Datendefinition prüfen          ↓
                     ist der Zustand stabil?
                            │
                      ┌─────┴─────┐
                      │           │
                     ja          nein
                      │           ↓
                      │      CPD / Left-right-HMM
                      │           ↓
                      │      bringt Historie weiterhin
                      │      einen Zusatznutzen?
                      │        ┌──┴──┐
                      │       nein   ja
                      │        │      ↓
                      │        │   GRU / TCN
                      │        │      ↓
                      │        │  bringt Attention
                      │        │  zusätzlichen Nutzen?
                      └────────┴─────────────→ RL-Umgebung
                                                  ↓
                                      Reward-/Transition-Suffizienz
                                                  ↓
                                      verbessert sich die
                                      Wartungsentscheidung?
```

## 13. Abbruchkriterien für die einzelnen Schritte

Die Komplexität wird nicht unbegrenzt erhöht.

- Wenn ein One-Class-HI bereits stabil eine gute Entscheidungsgrundlage bietet, ist kein komplexes Zeitmodell erforderlich;
- wenn ein HMM die zeitliche Konsistenz bereits ausreichend verbessert, wird nicht zwingend ein GRU benötigt;
- wenn GRU und TCN keine signifikanten Unterschiede zeigen, wird das einfachere und stabilere Modell ausgewählt;
- wenn Attention keinen zusätzlichen Nutzen liefert, wird Attention nicht beibehalten;
- wenn ein zusätzlicher Reward-/Transition-Loss die RL-Leistung nicht verbessert, werden die entsprechenden Heads nicht beibehalten;
- wenn eine Betriebsbedingung Reward oder Dynamik nicht beeinflusst, sollte sie als Nuisance behandelt und nicht in den Zustand aufgenommen werden.

---

# Teil 5: Bewertung und Auswahl des endgültigen Modells

## 14. Bewertung auf Repräsentationsebene

Bewertungskriterien:

- Zusammenhang des HI mit Damage, RUL oder Lebenszyklusposition;
- Monotonie;
- Trendability;
- Robustheit;
- Macro F1 der Degradationsphasen;
- Anzahl rückwärtsgerichteter Zustandswechsel;
- EDP- beziehungsweise Change-Point-Erkennung;
- Generalisierung über Lager und Betriebsbedingungen;
- Kalibrierung der Unsicherheit;
- Zustandsdimension.

Diese Kennzahlen beantworten:

> Besitzt der Zustand eine sinnvolle Gesundheitssemantik?

## 15. Bewertung auf Entscheidungsebene

Bewertungskriterien:

- Reward-Vorhersage;
- Transition-Vorhersage;
- History Gain;
- State Aliasing;
- RL-Konvergenz;
- Sample Efficiency;
- kumulativer Return;
- Wartungskosten;
- unnötige Wartung;
- Ausfälle und Downtime;
- Stabilität der Policy.

Diese Kennzahlen beantworten:

> Unterstützt der Zustand tatsächlich die Wartungsentscheidung?

Der endgültige State Interpreter darf nicht ausschließlich anhand der Klassifikationsgenauigkeit ausgewählt werden.

---

# Teil 6: In dieser Woche abgeschlossene Versuchsvorbereitung

## 16. Initialisierung eines eigenständigen Repositorys

Um klare Modulgrenzen zu erhalten, wird der State Interpreter in einem eigenständigen Repository entwickelt:

```text
VibFM
  ↓ z_health
State-Interpreter-Repository
  ↓ kompakter Zustand
Gearbox / RL-Umgebung
```

Das aktuelle Repository enthält:

- das erste MLP-Interface für ein einzelnes Zeitfenster;
- `RawMeasurement`;
- `EmbeddingSample`;
- `StateTransition`;
- `VibFMAdapter`;
- `DatasetAdapter`;
- `GearboxAdapter`;
- Tests für Modell und Datenverträge.

Aktueller Stand:

- sechs Softwaretests wurden bestanden;
- künstliches `z_health [8,256]` erzeugt einen kompakten Zustand `[8,5]`;
- falsche Shapes, Transitionen zwischen unterschiedlichen Episoden und falsche Zeitreihenfolgen werden erkannt.

Grenze der Aussage:

> Dabei handelt es sich um Software- und Schnittstellentests und nicht um reale State-Interpreter-Experimente.

Noch nicht durchgeführt:

- realen Checkpoint erhalten;
- reales `z_health` extrahieren;
- HI- und Phasenlabels formal definieren;
- Modell trainieren;
- RL anbinden;
- Leistungswerte erzeugen.

---

# Teil 7: Nächste Schritte

## 17. Daten und VibFM

1. Pretrained VibFM-Checkpoint erhalten;
2. Codeversion, STFT und Normalisierung bestätigen;
3. tatsächlichen Ausgang und Dimension von `z_health` bestätigen;
4. ersten Run-to-Failure-Datensatz auswählen;
5. Daten nach Lager, Run und Zeit organisieren;
6. reale, zeitlich geordnete Embeddings extrahieren.

## 18. Erste Versuchsrunde

Folgende Bedingungen werden für alle Vergleiche festgehalten:

- derselbe VibFM-Checkpoint;
- dieselben Embeddings;
- dieselbe Datenaufteilung;
- derselbe Split auf Lagerebene;
- dasselbe Bewertungsprotokoll.

Verglichen werden:

1. Identity;
2. One-Class-HI;
3. K-means-Phasen;
4. MLP für ein einzelnes Zeitfenster.

Anschließend wird anhand der zeitlichen Konsistenz und des History Gains entschieden, ob HMM, GRU/TCN oder Attention untersucht werden.

## 19. Fragen 

1. Sollte für die ersten Versuche ein öffentlicher Run-to-Failure-Datensatz oder das aktuelle Gearbox-Projekt verwendet werden?
2. Können der VibFM-Checkpoint, die entsprechende Codeversion und die Vorverarbeitungskonfiguration bereitgestellt werden?
3. Wann kann Maintenance Arena für die Bewertung von Aktionen, Reward und Transitionen verwendet werden?

---


