# Sprechtext zum Arbeitsbericht

## Folie 1

Hallo zusammen. Heute zeige ich kurz meine Arbeit vom 12. September. Zuerst kommt das Ergebnis des Blind-Tests. Danach zeige ich die Stabilitätsstudie mit mehreren Seeds und Bearings. Zum Schluss erkläre ich die Analyse von Bearing1_4 und den nächsten Schritt.

## Folie 2

Das war mein Ausgangspunkt. Das System war schon vollständig aufgebaut. Der AutoEncoder erzeugt acht Werte. Der Interpreter macht daraus Level, Trend und Movement. Jetzt wollte ich nicht mehr nur zeigen, dass der Code läuft. Ich wollte prüfen, ob das Ergebnis auch bei unbekannten Daten und bei anderen Trainingsläufen stabil bleibt. Die ersten 15 Messungen bilden die Referenz. Level ist der geglättete Abstand, Trend die Steigung. Movement misst den Abstand zwischen zwei z-Punkten.

## Folie 3

Hier ist der echte Blind-Test mit Bearing2_5. Die Rohdaten wurden einmal geladen und dann durch beide Arme geführt. Arm A erreicht eine Korrelation von 0,946. Arm B erreicht 0,964. Beide Werte sind klar positiv und höher als die vorher festgelegte Grenze von 0,7. Arm B ist etwas besser, aber der Unterschied von 0,018 ist klein. Deshalb sage ich nicht, dass Arm B klar überlegen ist. Das Ergebnis gilt für dieses eine zuvor unbekannte Bearing. Spätere Wiederverwendung in der Stabilitätsstudie ist keine neue Blind-Evaluation.

## Folie 4

Ein einzelner Blind-Test reicht noch nicht für eine Stabilitätsaussage. Deshalb habe ich eine größere Studie gemacht. Ich benutze zwei Betriebsbedingungen. In jeder Bedingung rotiert jedes der fünf Bearings einmal in den Test. Dazu kommen drei verschiedene Seeds. Insgesamt entstehen 30 Trainingsläufe. Wichtig ist: Die Normalisierung benutzt immer nur die drei Trainings-Bearings. Training und Test liegen hier jeweils in derselben Betriebsbedingung. Das ist eine Wiederholungsstudie in zwei Bedingungen. Die Daten waren im Projekt schon bekannt. Es sind keine 30 unabhängigen Bearings.

## Folie 5

Das wichtigste Ergebnis ist Level. Über alle 30 Läufe liegt die mittlere Spearman-Korrelation bei 0,959. Auch getrennt nach den beiden Betriebsbedingungen bleiben die Werte hoch. Die Streuung ist mit 0,048 relativ klein. Das zeigt: Das gute Blind-Ergebnis kommt nicht nur von einem einzelnen Seed oder einem einzelnen Bearing. Bearing1_4 bleibt aber der schwierigste Fall. Die Läufe teilen sich Daten. Deshalb zeige ich Mittelwert und Streuung. Diese Studie allein prüft keine direkte Übertragung von einer Bedingung in die andere. Zeit ist hier nur ein Vergleichswert für die Reihenfolge, kein gemessener Schaden.

## Folie 6

Hier benutze ich unterschiedliche Kennzahlen. Level hat eine hohe Korrelation mit der Zeit. Trend stimmt in rund 82 Prozent der Fälle mit der Richtung der Level-Änderung überein. Movement korreliert mit dem Betrag dieser Änderung mit 0,379. Trend entsteht selbst aus Level. Deshalb ist das nur ein interner Check. Daraus kann ich noch keine Fehlererkennung oder Vorhersage ableiten. Dafür brauche ich unabhängige Labels.

## Folie 7

Bei Bearing1_4 liegt der mittlere Level-Wert für die Korrelation bei 0,860. Die Kurven der drei Seeds sind sehr ähnlich. Der Rekonstruktionsfehler ist 1,77-mal so hoch wie der Mittelwert der anderen Bearings. Diese Vergleichsgruppe enthält aber Trainingsdaten. Deshalb ist das nur ein Hinweis und noch keine Erklärung der Ursache. Außerdem habe ich 16 Kombinationen von Kalibrierung und Fenster geprüft. Dabei ändern sich auch die ausgewerteten Zeitpunkte. Ich wähle daraus keine neuen Parameter für v0.1.

## Folie 8

Mein Fazit ist: Level zeigt in den bisherigen Daten eine hohe Zeitkorrelation. Die Ergebnisse und Version v0.1 sind im Repository dokumentiert. Als Nächstes möchte ich die Kalibrierung und die Distanz vergleichen. Zum Beispiel ein robusteres Zentrum oder eine skalierte Distanz. Dafür brauche ich vorher einen festen Vergleichsplan mit gleichen Zeitpunkten. Die bisherigen Testdaten gelten dann als Entwicklungsdaten. Physischer Schaden und der Nutzen für RL bleiben offen. Passt dieser nächste Schritt für die weitere Arbeit?