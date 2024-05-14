# datascienceproject
Data Science Project

Anmerkungen zum Preprocessing:
Zeile 23: Was ist mit blank, refused und don't knows. Das sind zwar weniger werte aber die müssten wir nach der logik auch skippen.
Vielleicht wäre es hier gut diese Kategorien in eine Zahl zu überführen für die analyse. Ausserdem was passiert mit den restlichen? Hier bräuchte es noch eine Strategie siehe Vorlesung 2 handling missing data. 
Genauso Zeile 27: Habt ihr euch angeshcaut wie viel wir droppen? Hier könnte man aber denken wir argumentieren dass das Sinn macht.

Zeile 32,33,35,36: wieso ändert ihr nur hier nur die 91 von HALLUCEVR und INHALEVER es gibt ja auch in z.B. SednMlif? Hier gibt es auch noch eine Kategorie die ja bedeutet.
Zeile 40... geht sehr viel verloren wegen dem
Zeile 43: mental health hat values von 0-3, hier werden alle excluded ausser 1,2 dementsprechend macht zeile 49 und 50 keinen sinn.

Zeile 64: Haben wir auf Histplot geändert um multiple einzuführen damit man die Daten visuell vergleichen kann
+ die x achsen labels rotiert

Im df_cleaned gibt es auch noch generelle Jugenfragen die man noch rausnehmen müsste.

Wir haben jetzt schonmal ein grobes Gerüst für die feature selection geschrieben und es wäre super wenn ihr am Ende des Preprocessings einen df_clean haben könntet der ein Pandas dataframe ist mit dem Columns als eventuelle features. 






