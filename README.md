# 🛒 Atlântico Wholesale - Kundensegmentierung (Unsupervised ML)

Dieses Projekt nutzt Unsupervised Machine Learning, um die Kundenbasis des portugiesischen Lebensmittelgroßhändlers **Atlântico Wholesale** anhand ihres jährlichen Kaufverhaltens in natürliche Segmente zu unterteilen.

Das Ziel ist es, dem Marketing-Team ein datengestütztes Dashboard zur Verfügung zu stellen, mit dem Segmente analysiert und Neukunden automatisch klassifiziert werden können.

## 📁 Projektstruktur
atlantico-ml/
├── data/
│   └── data_atlantico.csv      # Ursprünglicher Kundendatensatz
├── eda.ipynb                   # Explorative Datenanalyse (Verteilungen, Schiefe, Korrelationen)
├── modeling.ipynb              # Outlier-Handling (Isolation Forest), Scaling, PCA & KMeans Clustering
├── app2.py                     # Interaktive Streamlit-App (Plotly)
├── requirements.txt            # Python-Abhängigkeiten
└── .gitignore                  # Von Git ignorierte Dateien
## 🛠️ Einrichtung & Installation

1. **Repository klonen:**
   ```bash
   git clone [https://github.com/thombow-source/atlantico-ml.git](https://github.com/thombow-source/atlantico-ml.git)
   cd atlantico-ml
* Virtuelle Umgebung erstellen und aktivieren:
```bash
python -m venv .atlantic_ML
.atlantic_ML\Scripts\activate
```
2. **Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```
## 🚀 Ausführung & Reproduktion
Um das Modell von Grund auf neu zu trainieren und die App zu starten, folge diesen Schritten:
1. Explorative Analyse ansehen (optional):Öffne eda.ipynb, um die Verteilung und Schiefe der Ausgabekategorien zu untersuchen.Modellierung & 
2. Pipeline-Generierung:Führe das Notebook modeling.ipynb von oben nach unten aus.
* Bereinigt Ausreißer mit einem IsolationForest (5% Contamination).
* Standardisiert die Ausgaben mit dem StandardScaler.
* Reduziert die Dimensionalität mittels PCA auf 2 Hauptkomponenten.
* Clustert die Kunden mit KMeans (K=3).
* Generiert die Artefakte segmentation_pipeline.pkl und data_clustered.csv.
3. **Streamlit-App starten:**
```bash
streamlit run app.py
```