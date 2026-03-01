# F1-Strategy-Analyses

Analiza strategije guma u Formuli 1 korišćenjem klasifikacije i klasterovanja nad FastF1 podacima.

## Struktura

- `src/data_collection.py` - prikupljanje sirovih podataka iz FastF1
- `src/preprocessing.py` - transformacija i filtriranje krugova
- `src/modeling.py` - modeli klasifikacije i klasterovanja + evaluacija
- `src/run_pipeline.py` - pokretanje kompletnog pipeline-a
- `notebooks/f1_strategija_analiza.ipynb` - notebook sa objašnjenjima na srpskom i vizualizacijama podataka/evaluacije (uključujući confusion matrix poslednjeg run-a)

## Instalacija

```bash
pip install -r requirements.txt
```

## Pokretanje pipeline-a

```bash
python -m src.run_pipeline --start-year 2019 --end-year 2025
```

Evaluacioni rezultati se čuvaju u direktorijumu `results/`.

Za robustnije prikupljanje bez prethodnog cache-a (jedan run umesto više pokušaja), možeš povećati retry/passes:

```bash
python -m src.run_pipeline --start-year 2019 --end-year 2025 --max-schedule-retries 4 --max-session-retries 4 --max-collection-passes 4 --retry-sleep-seconds 2.0
```

## Istorija run-ova

Svaki run dobija:

- `run_id` i `run_timestamp`
- istorijske metrikе sa parametrima modela:
  - `results/classification/classification_metrics_history.csv`
  - `results/clustering/clustering_metrics_history.csv`
- run-specifične artefakte u:
  - `results/classification/runs/`
  - `results/clustering/runs/`
