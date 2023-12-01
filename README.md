# UnifiedLog
Transformer alapú robusztus módszerek eseménynaplók alapján történő hibakeresésre.


## Leírás
A program eseménynaplókban történő anomáliadetektálást csinál a Loghub (https://github.com/logpai/loghub) projekten belül található adathalmazokon.
Az általános naplórsorreprezentáció gyártást egy BERT-hez hasonló architektúrájú transformerrel vézni. A naplósorok reprezentációinak a sorozatában egy másik transformer modellel prediktál anomáliákat.
<br>
<br>
A projekt a szakdolgozatnak és tdk dolgozatnak készült.

## Használat 
A környezet reprodukálásáát az Anaconda csomagkezelő szoftverrel lehet megtenni:<br>
<code>conda env create -f environment.yml</code>
<br>
<br>
Az adatok letöltéséhez a <i>loghub_downloader.py</i> script felel.
<br>
<code>python3 loghub _downloa.py -s \<mentesi mappa\></code>
<br>
<br>
Az adatok előfeldolgozásáért a <i>data_preprocess.py</i> script felel.
<br>
<code>python3 data_preproceess.py -d \<az a mappa ahova a data_downloader.py mentett\> -s \<mentesi mappa\> -t \<tokenek szama (alapertelmezett=1002)\></code>
<br>
<br>
A gépi tanulási modelleket a <i>run.py</i> scriptel lehet tanítani és kiértékelni.
<br>
<code>python3 run.py -c \<konfiguracios yaml fajl vagy konfiguracios fajlokat tartalmaz mappa\> -t \<cpu szalak szama\></code>

## Naplózás
A <i>run.py</i> fájl a neptune.ai felhő alapú naplózási szolgáltatást használja a mérések követéséhez.
