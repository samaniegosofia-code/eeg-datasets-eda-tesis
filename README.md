# EDA — 5 Datasets Públicos EEG

**Autora:** Sofía Samaniego López  
**Director:** Dr. Everardo Inzunza González  
**Institución:** FIAD — Universidad Autónoma de Baja California  
**Semestre:** 2026-1

## Descripción

Análisis exploratorio (EDA) de cinco datasets públicos de EEG para la tesis:
*Clasificación de estrés, ansiedad y depresión mediante imágenes topográficas
EEG con CNN y Vision Transformers.*

## Datasets

| # | Dataset | Sujetos | Condición |
|---|---------|---------|-----------|
| 1 | MODMA | 53 | Depresión / Control |
| 2 | Kaggle BRMH | 945 | 7 trastornos psiquiátricos |
| 3 | DASPS | 23 | Ansiedad inducida |
| 4 | DEAP | 32 | Valencia / Arousal emocional |
| 5 | EEG-ADHD | 121 | ADHD / Control |

## Dependencias

```bash
pip install numpy scipy matplotlib seaborn pandas
```

## Uso

```bash
python scripts/01_MODMA_EDA.py --mat_file datos.mat --out_dir fig/
python scripts/02_BRMH_EDA.py  --csv_file datos.csv --out_dir fig/
python scripts/03_DASPS_EDA.py --edf_file S01.edf   --out_dir fig/
python scripts/04_DEAP_EDA.py  --dat_file s02.dat   --out_dir fig/
python scripts/05_ADHD_EDA.py  --csv_file datos.csv --out_dir fig/
```

## Nota sobre los datos

Los archivos de datos originales no se incluyen por restricciones de licencia.
Descárgalos desde las fuentes oficiales indicadas en el documento técnico.
