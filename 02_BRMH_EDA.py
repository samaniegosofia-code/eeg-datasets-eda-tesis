"""
02_BRMH_EDA.py
==============
Análisis Exploratorio (EDA) — Dataset Kaggle EEG Psychiatric Disorders (BRMH)
Genera las 5 figuras requeridas por la tarea de datasets públicos.

Autora : Sofía Samaniego López
Tesis  : Clasificación de estrés, ansiedad y depresión mediante
         imágenes topográficas EEG (CNN + ViT) — FIAD UABC 2026
Dataset: Park, S.M. et al. (2021). Frontiers in Psychiatry.
         https://doi.org/10.3389/fpsyt.2021.707581
         Kaggle: https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset

USO:
    python 02_BRMH_EDA.py --csv_file EEG_machinelearing_data_BRMH.csv --out_dir fig/

SALIDA (carpeta fig/):
    kaggle_barras_demograficas.png  → Fig 1: distribución demográfica por sexo y diagnóstico
    kaggle_hist_clases.png          → Fig 2: desbalance de clases (horizontal)
    kaggle_heatmap_alpha_beta.png   → Fig 3: heatmap Pearson Alpha/Beta canales frontales
    kaggle_proyeccion_epocx.png     → Fig 4: proyección EPOC X sobre topología 10-20
    kaggle_dist_edad.png            → Fig 5: distribución de edad vs. rango 18-25 años

DEPENDENCIAS:
    pip install pandas numpy scipy matplotlib seaborn

NOTA SOBRE ESTADÍSTICAS REALES (calculadas del CSV):
    Las distribuciones PSD Alpha son no normales (KS p<0.01) → usar Mann-Whitney
    en lugar de t-test para comparaciones entre grupos. Se aplica normalización
    Min-Max o log1p antes del entrenamiento de modelos.
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.stats import kstest

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────

# Mapeo EMOTIV EPOC X (14ch) → canal 10-20 más cercano
EPOCX_MAP = {
    'AF3': 'FP1', 'F7': 'F7',  'F3': 'F3',
    'T7':  'T3',  'P7': 'T5',  'O1': 'O1',
    'O2':  'O2',  'P8': 'T6',  'T8': 'T4',
    'F4':  'F4',  'F8': 'F8',  'AF4': 'FP2',
    # FC5 y FC6 no están en el dataset → se interpolan con MNE
}
# Letra de columna para cada canal 10-20 en el CSV
CH_LETTER = {
    'FP1': 'a', 'FP2': 'b', 'F7': 'c',  'F3': 'd',  'Fz': 'e',
    'F4':  'f', 'F8':  'g', 'T3': 'h',  'C3': 'i',  'Cz': 'j',
    'C4':  'k', 'T4':  'l', 'T5': 'm',  'P3': 'n',  'Pz': 'o',
    'P4':  'p', 'T6':  'q', 'O1': 'r',  'O2': 's',
}
# Prefijos de banda en el CSV
BAND_PREFIX = {
    'delta': 'AB.A.delta', 'theta': 'AB.B.theta', 'alpha': 'AB.C.alpha',
    'beta':  'AB.D.beta',  'gamma': 'AB.F.gamma',
}

# Etiquetas legibles
LABEL_MAP = {
    'Healthy control':                    'Control (HC)',
    'Mood disorder':                      'Depresión (MDD)',
    'Anxiety disorder':                   'Ansiedad',
    'Trauma and stress related disorder': 'Trauma/Estrés',
    'Schizophrenia':                      'Esquizofrenia',
    'Addictive disorder':                 'T. Adictivo',
    'Obsessive compulsive disorder':      'TOC',
}
THESIS_CLASSES = ['Control (HC)', 'Depresión (MDD)', 'Ansiedad', 'Trauma/Estrés']

PALETTE = {
    'Control (HC)':   '#2ECC71', 'Depresión (MDD)': '#E74C3C',
    'Ansiedad':       '#3498DB', 'Trauma/Estrés':   '#9B59B6',
    'Esquizofrenia':  '#E67E22', 'T. Adictivo':     '#F39C12',
    'TOC':            '#1ABC9C',
}

# Posiciones 2D estándar 10-20 (proyección esférica)
POS_1020 = {
    'FP1': (-0.18,  0.92), 'FP2': ( 0.18,  0.92),
    'F7':  (-0.71,  0.55), 'F8':  ( 0.71,  0.55),
    'F3':  (-0.40,  0.60), 'F4':  ( 0.40,  0.60),
    'Fz':  ( 0.00,  0.65),
    'T3':  (-0.87,  0.00), 'T4':  ( 0.87,  0.00),
    'C3':  (-0.45,  0.00), 'Cz':  ( 0.00,  0.00), 'C4': ( 0.45, 0.00),
    'T5':  (-0.71, -0.55), 'T6':  ( 0.71, -0.55),
    'P3':  (-0.40, -0.60), 'Pz':  ( 0.00, -0.65), 'P4': ( 0.40, -0.60),
    'O1':  (-0.18, -0.92), 'O2':  ( 0.18, -0.92),
}
# Posición aproximada para canales faltantes (FC5/FC6)
POS_FC = {'FC5': (-0.60, 0.30), 'FC6': (0.60, 0.30)}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ──────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────

def band_cols(band_prefix_key, channels_1020):
    """Construye lista de nombres de columna para los canales pedidos."""
    prefix = BAND_PREFIX[band_prefix_key]
    return [f'{prefix}.{CH_LETTER[ch]}.{ch}' for ch in channels_1020]


def load_brmh(csv_path: str) -> pd.DataFrame:
    """Carga el CSV y agrega columna 'label' con nombres legibles."""
    df = pd.read_csv(csv_path)
    df['label'] = df['main.disorder'].map(LABEL_MAP)
    epocx_1020 = list(EPOCX_MAP.values())
    alpha_cols = band_cols('alpha', epocx_1020)
    df['alpha_mean'] = df[alpha_cols].mean(axis=1)
    return df


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


# ──────────────────────────────────────────────────────────────
# FIGURAS EDA
# ──────────────────────────────────────────────────────────────

def fig1_demographic_bars(df, out_dir):
    """Figura 1 — Distribución demográfica: barras agrupadas por diagnóstico y sexo."""
    counts = df.groupby(['label', 'sex']).size().unstack(fill_value=0)
    order  = df['label'].value_counts().index.tolist()
    counts = counts.reindex(order)
    total  = counts.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    x, w   = np.arange(len(counts)), 0.35
    male   = counts.get('M', pd.Series(0, index=counts.index))
    female = counts.get('F', pd.Series(0, index=counts.index))

    b1 = ax.bar(x - w/2, male,   w, label='Masculino', color='#5DADE2', edgecolor='white')
    b2 = ax.bar(x + w/2, female, w, label='Femenino',  color='#F1948A', edgecolor='white')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 2, str(int(h)),
                    ha='center', va='bottom', fontsize=8)

    for i, (lbl, tot) in enumerate(total.items()):
        ax.text(i, tot + 12, f'n={int(tot)}', ha='center',
                fontsize=8.5, fontweight='bold', color='#2C3E50')
        if lbl in THESIS_CLASSES[1:]:
            ax.text(i, -22, '*', ha='center', fontsize=14,
                    color='red', fontweight='bold', clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Número de sujetos', fontsize=11)
    ax.set_ylim(0, total.max() + 40)
    ax.set_title('1. Distribución Demográfica — Dataset BRMH (N = 945)\n'
                 '(*) = condiciones objetivo de la tesis',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'kaggle_barras_demograficas.png'))


def fig2_class_imbalance(df, out_dir):
    """Figura 2 — Desbalance de clases (barras horizontales)."""
    class_counts = df['label'].value_counts()
    ideal        = len(df) / len(class_counts)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [PALETTE.get(lbl, '#BDC3C7') for lbl in class_counts.index]
    bars    = ax.barh(class_counts.index, class_counts.values,
                      color=colors, edgecolor='white', linewidth=0.8)

    ax.axvline(ideal, color='#E74C3C', ls='--', lw=1.8,
               label=f'Distribución ideal (n≈{ideal:.0f})')

    for bar, n in zip(bars, class_counts.values):
        ax.text(n + 3, bar.get_y() + bar.get_height()/2,
                f'{n}  ({n/len(df)*100:.1f}%)', va='center', fontsize=9)

    for bar, lbl in zip(bars, class_counts.index):
        if lbl in THESIS_CLASSES[1:]:
            bar.set_edgecolor('#C0392B')
            bar.set_linewidth(2.2)

    ax.set_xlabel('Número de sujetos', fontsize=11)
    ax.set_xlim(0, 320)
    ax.set_title('2. Desbalance de Clases — Dataset BRMH (N = 945)\n'
                 'Borde rojo = condiciones objetivo de la tesis',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'kaggle_hist_clases.png'))


def fig3_alpha_beta_heatmap(df, out_dir):
    """Figura 3 — Heatmap Pearson Alpha/Beta en canales frontales EPOC X."""
    frontal_epocx = ['AF3', 'F7', 'F3', 'F4', 'F8', 'AF4']
    frontal_1020  = [EPOCX_MAP[ch] for ch in frontal_epocx]
    front_alpha   = band_cols('alpha', frontal_1020)
    front_beta    = band_cols('beta',  frontal_1020)

    sub = df[front_alpha + front_beta].copy()
    sub.columns = ([f'{ch}\nα' for ch in frontal_epocx] +
                   [f'{ch}\nβ' for ch in frontal_epocx])
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    ax.set_title('3. Heatmap Correlación Pearson Alpha–Beta\n'
                 'Canales Frontales EPOC X — Dataset BRMH (N = 945)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'kaggle_heatmap_alpha_beta.png'))


def fig4_epocx_projection(out_dir):
    """Figura 4 — Proyección 14 canales EPOC X sobre topología 10-20."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Cabeza
    head = Circle((0, 0), 1.0, fill=False, edgecolor='#2C3E50',
                  linewidth=2.5, zorder=3)
    ax.add_patch(head)
    ax.plot([-0.07, 0, 0.07], [0.98, 1.12, 0.98], 'k-', lw=2.5, zorder=4)
    for s in [-1, 1]:
        ax.plot([s*1.0, s*1.08, s*1.08, s*1.0],
                [0.12, 0.08, -0.08, -0.12], 'k-', lw=2.5, zorder=4)

    # Canales 10-20 del CSV (fondo)
    for ch, (x, y) in POS_1020.items():
        ax.scatter(x, y, s=140, color='#D5DBDB', edgecolors='#7F8C8D',
                   lw=1, zorder=5)
        ax.text(x, y + 0.09, ch, ha='center', fontsize=7, color='#7F8C8D')

    # Canales EPOC X disponibles (azul)
    for epoc, ch in EPOCX_MAP.items():
        x, y = POS_1020[ch]
        ax.scatter(x, y, s=220, color='#2E86C1', edgecolors='white',
                   lw=1.5, zorder=6)
        ax.text(x, y + 0.09, epoc, ha='center', fontsize=7.5,
                color='#1A5276', fontweight='bold')

    # FC5 / FC6 faltantes (rojo X)
    for ch, (x, y) in POS_FC.items():
        ax.scatter(x, y, s=220, color='#E74C3C', edgecolors='white',
                   lw=1.5, marker='X', zorder=7)
        ax.text(x, y + 0.09, ch, ha='center', fontsize=7.5,
                color='#922B21', fontweight='bold')

    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#D5DBDB', markeredgecolor='#7F8C8D',
                      markersize=10, label='Canales 10-20 del CSV (19 ch)'),
        mlines.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#2E86C1', markeredgecolor='white',
                      markersize=10, label='EPOC X disponible (12 ch)'),
        mlines.Line2D([0], [0], marker='X', color='w',
                      markerfacecolor='#E74C3C', markeredgecolor='white',
                      markersize=10, label='FC5 / FC6 — a interpolar con MNE'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=8,
              bbox_to_anchor=(0.5, -0.08))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    ax.set_title('4. Proyección 14 Canales EPOC X sobre Topología 10-20\n'
                 'Dataset BRMH — Canales disponibles vs. a interpolar',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'kaggle_proyeccion_epocx.png'))


def fig5_age_distribution(df, out_dir):
    """Figura 5 — Distribución de edad vs. rango objetivo 18-25 años."""
    target = (df['age'] >= 18) & (df['age'] <= 25)
    n_target = target.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df['age'], bins=35, color='#AED6F1', edgecolor='white',
            linewidth=0.6, alpha=0.85, label='Todos los sujetos (N=945)')
    ax.hist(df[target]['age'], bins=35, color='#2E86C1', edgecolor='white',
            linewidth=0.6, alpha=0.90,
            label=f'Rango 18–25 años (n={n_target}, {n_target/len(df)*100:.1f}%)')
    ax.axvline(18, color='#E74C3C', ls='--', lw=1.8, alpha=0.9,
               label='Límites 18–25 años')
    ax.axvline(25, color='#E74C3C', ls='--', lw=1.8, alpha=0.9)
    ax.axvline(df['age'].mean(), color='#117A65', ls='-', lw=1.5,
               label=f'Media general = {df["age"].mean():.1f} años')

    ax.set_xlabel('Edad (años)', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.set_title(f'5. Distribución de Edad — Dataset BRMH (N = 945)\n'
                 f'Rango objetivo universitario 18–25 años: '
                 f'n={n_target} ({n_target/len(df)*100:.1f}%)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'kaggle_dist_edad.png'))


# ──────────────────────────────────────────────────────────────
# ESTADÍSTICAS DESCRIPTIVAS
# ──────────────────────────────────────────────────────────────

def descriptive_stats(df, out_dir):
    """Calcula y exporta estadísticas descriptivas PSD Alpha por clase objetivo."""
    import csv

    print("\n── Estadísticas PSD Alpha media (12 canales EPOC X) por clase objetivo ──")
    header = ['Grupo', 'n', 'Media', 'Mediana', 'SD', 'Min', 'Max', 'IC_lo', 'IC_hi',
              'KS_stat', 'KS_p', 'Normal']
    rows = []
    for lbl in ['Control (HC)', 'Depresión (MDD)', 'Ansiedad', 'Trauma/Estrés']:
        sub = df[df['label'] == lbl]['alpha_mean'].dropna()
        n   = len(sub)
        m, med, sd = sub.mean(), sub.median(), sub.std()
        lo  = m - 1.96 * sd / np.sqrt(n)
        hi  = m + 1.96 * sd / np.sqrt(n)
        stat, p = kstest(sub, 'norm', args=(m, sd))
        normal  = 'Sí' if p > 0.05 else 'No'
        rows.append([lbl, n, round(m,2), round(med,2), round(sd,2),
                     round(sub.min(),2), round(sub.max(),2),
                     round(lo,2), round(hi,2), round(stat,3), round(p,4), normal])
        print(f"  {lbl:<20} n={n:3d}  media={m:.2f}  SD={sd:.2f}  IC=[{lo:.2f},{hi:.2f}]  "
              f"KS p={p:.4f} ({normal})")

    print(f"\n  NaN en alpha_mean: {df['alpha_mean'].isna().sum()}")
    z_outliers = (np.abs((df['alpha_mean']-df['alpha_mean'].mean())/df['alpha_mean'].std()) > 3).sum()
    print(f"  Outliers Z>3: {z_outliers}")
    print(f"  Desbalance ratio máx: {df['label'].value_counts().max()} : "
          f"{df['label'].value_counts().min()} → SMOTE recomendado")

    csv_path = os.path.join(out_dir, 'brmh_alpha_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\n  CSV exportado: {csv_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='EDA — Dataset BRMH Kaggle EEG Psychiatric Disorders (5 figuras)')
    parser.add_argument('--csv_file', type=str,
                        default='EEG_machinelearing_data_BRMH.csv',
                        help='Ruta al CSV del dataset BRMH')
    parser.add_argument('--out_dir', type=str, default='fig',
                        help='Carpeta de salida para las figuras')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nCargando: {args.csv_file}")

    df = load_brmh(args.csv_file)
    print(f"  → {len(df)} sujetos, {len(df.columns)} columnas")
    print(f"  → Clases: {df['label'].value_counts().to_dict()}")

    print("\nGenerando figuras EDA:")
    fig1_demographic_bars(df, args.out_dir)
    fig2_class_imbalance(df, args.out_dir)
    fig3_alpha_beta_heatmap(df, args.out_dir)
    fig4_epocx_projection(args.out_dir)
    fig5_age_distribution(df, args.out_dir)

    descriptive_stats(df, args.out_dir)
    print("\n✅ EDA completado. Figuras guardadas en:", args.out_dir)


if __name__ == '__main__':
    main()
