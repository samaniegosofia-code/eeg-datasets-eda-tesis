"""
05_ADHD_EDA.py
==============
Análisis Exploratorio (EDA) — Dataset EEG-ADHD
(EEG data for ADHD and Control Children)
Genera las 5 figuras requeridas por la tarea de datasets públicos.

Autora : Sofía Samaniego López
Tesis  : Clasificación de estrés, ansiedad y depresión mediante
         imágenes topográficas EEG (CNN + ViT) — FIAD UABC 2026
Dataset: Nasrabadi, A.M. et al. (2020). IEEE DataPort.
         https://doi.org/10.21227/bhbf-4438
Ref DL : Sanchis, J. et al. (2024). Heliyon, 10(4), e26028.
         https://doi.org/10.1016/j.heliyon.2024.e26028

USO:
    python 05_ADHD_EDA.py --csv_file adhdata.csv --out_dir fig/

SALIDA (carpeta fig/):
    adhd_pie_clases.png        → Fig 1: gráfico circular + barras de distribución
    adhd_raw_signal.png        → Fig 2: serie cruda canal F3, ADHD vs Control
    adhd_psd.png               → Fig 3: PSD Welch Theta sombreada, media entre sujetos
    adhd_heatmap_pearson.png   → Fig 4: heatmap Pearson 19×19 (ADHD | Control | Diferencia)
    adhd_topomap_theta.png     → Fig 5: topomapa 2D Theta ADHD vs Control

ESTRUCTURA DEL CSV (adhdata.csv):
    - 2 166 383 filas × 21 columnas
    - 19 columnas EEG (10-20, 128 Hz, en µV): Fp1, Fp2, F3, F4, C3, C4,
      P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz
    - 'Class': 'ADHD' | 'Control'
    - 'ID'   : identificador de sujeto (121 sujetos: 61 ADHD + 60 Control)
    - fs = 128 Hz | Duración media por sujeto ≈ 2.3 min

NOTA SOBRE PREPROCESAMIENTO:
    Descartar las primeras 5 000 muestras (≈39 s) por artefactos de inicio.
    Este script aplica un skip adicional de 2 s para las figuras de serie temporal.

NOTA ESTADÍSTICA:
    PSD Theta NO normalmente distribuida (Shapiro-Wilk p<0.001) en ambas clases.
    Usar Mann-Whitney U para comparaciones; normalización log1p antes del modelo.
    La mayor variabilidad en Control refleja la heterogeneidad clínica del grupo.

DEPENDENCIAS:
    pip install pandas numpy scipy matplotlib seaborn
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt, welch
from scipy.stats import shapiro

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────

FS       = 128    # Hz
SKIP_INI = 5000   # muestras a descartar al inicio por sujeto

EEG_COLS = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']

# Canales EPOC X disponibles (10 de 14 — AF3, FC5, FC6, AF4 no están en 10-20 de 19ch)
EPOCX_AVAIL = ['F7','F3','T7','P7','O1','O2','P8','T8','F4','F8']
EPOCX_MISS  = ['AF3','FC5','FC6','AF4']

# Posiciones 2D (proyección esférica estándar) para los 19 canales
POS_19 = {
    'Fp1':(-0.18, 0.92), 'Fp2':( 0.18, 0.92),
    'F7': (-0.72, 0.45), 'F3': (-0.38, 0.50), 'Fz': ( 0.00, 0.65),
    'F4': ( 0.38, 0.50), 'F8': ( 0.72, 0.45),
    'T7': (-0.87, 0.00), 'C3': (-0.45, 0.00), 'Cz': ( 0.00, 0.00),
    'C4': ( 0.45, 0.00), 'T8': ( 0.87, 0.00),
    'P7': (-0.72,-0.45), 'P3': (-0.38,-0.50), 'Pz': ( 0.00,-0.65),
    'P4': ( 0.38,-0.50), 'P8': ( 0.72,-0.45),
    'O1': (-0.18,-0.92), 'O2': ( 0.18,-0.92),
}

COL_ADHD = '#E74C3C'
COL_CTRL = '#2ECC71'
BAND_COLORS = {
    'Delta':'#AED6F1','Theta':'#F9E79F',
    'Alpha':'#A9DFBF','Beta':'#FADBD8',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ──────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────

def load_adhd(csv_path: str) -> pd.DataFrame:
    """Carga el CSV y devuelve el DataFrame completo."""
    return pd.read_csv(csv_path)


def clean_subject(sub_df: pd.DataFrame, skip: int = SKIP_INI) -> pd.DataFrame:
    """Descarta las primeras `skip` muestras de un sujeto."""
    return sub_df.iloc[skip:].reset_index(drop=True) if len(sub_df) > skip else sub_df


def bandpass(sig: np.ndarray, fs: int = FS,
             lo: float = 1.0, hi: float = 40.0, order: int = 4) -> np.ndarray:
    """Filtro Butterworth pasa-banda + remoción de DC."""
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, sig - sig.mean())


def mean_psd_class(df, cls, channel='F3', n_sub=30):
    """PSD Welch media (± SD) de hasta n_sub sujetos de la clase dada."""
    ids  = df[df['Class'] == cls]['ID'].unique()[:n_sub]
    psds = []
    for sid in ids:
        sub = clean_subject(df[df['ID'] == sid])
        sig = bandpass(sub[channel].values)
        fw, Pxx = welch(sig, fs=FS, nperseg=256)
        psds.append(Pxx)
    return fw, np.mean(psds, axis=0), np.std(psds, axis=0)


def mean_corr_class(df, cls, n_sub=15):
    """Correlación Pearson 19×19 media de hasta n_sub sujetos."""
    ids   = df[df['Class'] == cls]['ID'].unique()[:n_sub]
    corrs = []
    for sid in ids:
        sub  = clean_subject(df[df['ID'] == sid])
        sigs = np.array([bandpass(sub[ch].values) for ch in EEG_COLS])
        corrs.append(np.corrcoef(sigs))
    return np.mean(corrs, axis=0)


def theta_topomap_class(df, cls, n_sub=25):
    """Potencia Theta (4-8 Hz) media por canal, promediada entre sujetos."""
    ids   = df[df['Class'] == cls]['ID'].unique()[:n_sub]
    theta = np.zeros(len(EEG_COLS))
    for sid in ids:
        sub = clean_subject(df[df['ID'] == sid])
        for ci, ch in enumerate(EEG_COLS):
            sig = bandpass(sub[ch].values)
            fw, Pxx = welch(sig, fs=FS, nperseg=256)
            theta[ci] += Pxx[(fw >= 4) & (fw <= 8)].mean()
    return theta / len(ids)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


# ──────────────────────────────────────────────────────────────
# FIGURAS EDA
# ──────────────────────────────────────────────────────────────

def fig1_class_distribution(df, out_dir):
    """Figura 1 — Pie (sujetos) + barras (muestras totales)."""
    n_adhd_s = df[df['Class']=='ADHD']['ID'].nunique()
    n_ctrl_s = df[df['Class']=='Control']['ID'].nunique()
    n_adhd_m = (df['Class']=='ADHD').sum()
    n_ctrl_m = (df['Class']=='Control').sum()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # — Pie de sujetos —
    wedges, texts, autotexts = axes[0].pie(
        [n_adhd_s, n_ctrl_s],
        labels=[f'ADHD\n(n={n_adhd_s})', f'Control\n(n={n_ctrl_s})'],
        colors=[COL_ADHD, COL_CTRL],
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops=dict(fontsize=11, fontweight='bold'))
    for at in autotexts: at.set_fontsize(10)
    axes[0].set_title('Distribución de Sujetos\n(N = 121)',
                      fontsize=10, fontweight='bold')

    # — Barras de muestras —
    bars = axes[1].bar(['ADHD','Control'], [n_adhd_m, n_ctrl_m],
                       color=[COL_ADHD, COL_CTRL],
                       edgecolor='white', width=0.45, linewidth=1.2)
    for bar, nm, ns in zip(bars, [n_adhd_m,n_ctrl_m], [n_adhd_s,n_ctrl_s]):
        axes[1].text(
            bar.get_x()+bar.get_width()/2, bar.get_height()+8000,
            f'N={nm:,}\n≈{nm/FS/3600:.1f} h total\n{ns} sujetos',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_ylabel('Número de muestras', fontsize=10)
    axes[1].set_title('Muestras Totales por Clase\n(@ 128 Hz)',
                      fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, max(n_adhd_m, n_ctrl_m) * 1.28)

    fig.suptitle(
        '1. Distribución de Clases — Dataset EEG-ADHD (N = 121 sujetos)\n'
        '61 ADHD (DSM-5) + 60 Control | 2 166 383 muestras totales',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'adhd_pie_clases.png'))


def fig2_raw_signal(df, out_dir, skip_extra_s=2):
    """Figura 2 — Serie de tiempo canal F3, ADHD vs Control (10 s)."""
    adhd_id = df[df['Class']=='ADHD'].groupby('ID').size().idxmax()
    ctrl_id = df[df['Class']=='Control'].groupby('ID').size().idxmax()
    skip    = int(skip_extra_s * FS)
    win     = 10 * FS

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t10 = np.linspace(0, 10, win)

    for ax, sid, cls, color in zip(
            axes,
            [adhd_id, ctrl_id],
            ['ADHD', 'Control'],
            [COL_ADHD, COL_CTRL]):
        sub = clean_subject(df[df['ID'] == sid])
        sig = bandpass(sub['F3'].values)
        seg = sig[skip: skip + win]
        ax.plot(t10, seg, color=color, lw=0.85, alpha=0.92)
        ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
        ax.set_ylabel('Amplitud (µV)', fontsize=10)
        ax.set_title(f'Canal F3 — {cls} (ID: {sid})  σ={seg.std():.1f} µV',
                     fontsize=10, fontweight='bold', color=color)

    axes[-1].set_xlabel('Tiempo (s)', fontsize=10)
    axes[-1].set_xlim(0, 10)
    fig.suptitle(
        '2. Serie de Tiempo EEG — Canal F3 (filtrado 1–40 Hz)\n'
        'Dataset EEG-ADHD | Desde t=2.0 s (descartadas primeras 5 000 muestras)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'adhd_raw_signal.png'))


def fig3_psd(df, out_dir, n_sub=30):
    """Figura 3 — PSD Welch ADHD vs Control, banda Theta sombreada."""
    print(f"    Calculando PSD ADHD ({n_sub} sujetos)...")
    fw, psd_adhd, std_adhd = mean_psd_class(df, 'ADHD',    n_sub=n_sub)
    print(f"    Calculando PSD Control ({n_sub} sujetos)...")
    fw, psd_ctrl, std_ctrl = mean_psd_class(df, 'Control', n_sub=n_sub)

    fig, ax = plt.subplots(figsize=(10, 5))
    mask = (fw >= 1) & (fw <= 40)

    for psd, std, cls, color in [
            (psd_adhd, std_adhd, 'ADHD',    COL_ADHD),
            (psd_ctrl, std_ctrl, 'Control',  COL_CTRL)]:
        ax.semilogy(fw[mask], psd[mask], color=color, lw=2,
                    label=f'{cls} (n={n_sub} sujetos)', alpha=0.9)
        ax.fill_between(fw[mask],
                        np.maximum(psd[mask] - std[mask]/2, 1e-6),
                        psd[mask] + std[mask]/2,
                        color=color, alpha=0.15)

    for name, lo, hi, col in [
            ('Delta',1,4,BAND_COLORS['Delta']),
            ('Theta',4,8,BAND_COLORS['Theta']),
            ('Alpha',8,13,BAND_COLORS['Alpha']),
            ('Beta',13,30,BAND_COLORS['Beta'])]:
        ax.axvspan(lo, hi, color=col, alpha=0.22, label=name)

    # Anotación Theta
    theta_adhd = psd_adhd[(fw >= 4) & (fw <= 8)].mean()
    theta_ctrl = psd_ctrl[(fw >= 4) & (fw <= 8)].mean()
    pct = (theta_adhd/theta_ctrl - 1) * 100
    ax.annotate(
        f'Theta ADHD\n({"+" if pct>0 else ""}{pct:.0f}% vs Control)',
        xy=(6, max(theta_adhd, theta_ctrl)*0.7),
        xytext=(9, max(theta_adhd, theta_ctrl)*2.5),
        fontsize=8.5, color=COL_ADHD, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COL_ADHD, lw=1.5))

    ax.set_xlabel('Frecuencia (Hz)', fontsize=11)
    ax.set_ylabel('PSD media (µV²/Hz)', fontsize=11)
    ax.set_xlim(1, 40)
    ax.set_title(
        '3. PSD Welch — ADHD vs. Control (Canal F3)\n'
        'Banda Theta sombreada (4–8 Hz) | Media ± ½SD entre sujetos',
        fontsize=11, fontweight='bold')
    ax.legend(ncol=3, fontsize=8.5, loc='upper right')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'adhd_psd.png'))


def fig4_heatmap(df, out_dir, n_sub=15):
    """Figura 4 — Heatmap Pearson 19×19: ADHD | Control | Diferencia."""
    print(f"    Calculando heatmap ADHD ({n_sub} sujetos)...")
    corr_adhd = mean_corr_class(df, 'ADHD',    n_sub=n_sub)
    print(f"    Calculando heatmap Control ({n_sub} sujetos)...")
    corr_ctrl = mean_corr_class(df, 'Control', n_sub=n_sub)
    corr_diff = corr_adhd - corr_ctrl

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    configs = [
        (corr_adhd, f'ADHD (media {n_sub} suj.)',    'RdYlGn', -1, 1),
        (corr_ctrl, f'Control (media {n_sub} suj.)', 'RdYlGn', -1, 1),
        (corr_diff, 'Diferencia ADHD − Control',     'RdBu_r', -0.3, 0.3),
    ]
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, configs):
        sns.heatmap(mat, annot=False, cmap=cmap, vmin=vmin, vmax=vmax,
                    xticklabels=EEG_COLS, yticklabels=EEG_COLS,
                    linewidths=0.2, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)

    fig.suptitle(
        '4. Heatmap Correlación Pearson 19×19 — Dataset EEG-ADHD\n'
        'ADHD vs. Control vs. Diferencia (media entre sujetos)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'adhd_heatmap_pearson.png'))


def fig5_topomap(df, out_dir, n_sub=25):
    """Figura 5 — Topomapa 2D potencia Theta, ADHD vs Control."""
    print(f"    Calculando topomapa Theta ADHD ({n_sub} sujetos)...")
    theta_adhd = theta_topomap_class(df, 'ADHD',    n_sub=n_sub)
    print(f"    Calculando topomapa Theta Control ({n_sub} sujetos)...")
    theta_ctrl = theta_topomap_class(df, 'Control', n_sub=n_sub)

    vmin = min(theta_adhd.min(), theta_ctrl.min())
    vmax = max(theta_adhd.max(), theta_ctrl.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_t = cm.RdYlBu_r

    def draw_topo(ax, vals, title):
        xs = np.array([POS_19[ch][0] for ch in EEG_COLS])
        ys = np.array([POS_19[ch][1] for ch in EEG_COLS])
        xi = np.linspace(-1.1, 1.1, 300)
        yi = np.linspace(-1.1, 1.1, 300)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((xs, ys), vals, (Xi, Yi), method='cubic')
        Zi[Xi**2 + Yi**2 > 1.0] = np.nan
        im = ax.contourf(Xi, Yi, Zi, levels=64, cmap=cmap_t,
                         norm=norm, zorder=1)
        head = Circle((0,0), 1.0, fill=False, edgecolor='k',
                      linewidth=2.5, zorder=4)
        ax.add_patch(head)
        ax.plot([-0.07,0,0.07],[0.98,1.12,0.98],'k-',lw=2.5,zorder=5)
        for s in [-1,1]:
            ax.plot([s*1.0,s*1.08,s*1.08,s*1.0],
                    [0.12,0.08,-0.08,-0.12],'k-',lw=2.5,zorder=5)
        ax.scatter(xs, ys, c=vals, cmap=cmap_t, norm=norm,
                   s=100, edgecolors='k', linewidths=1.1, zorder=6)
        for ch in EEG_COLS:
            ax.text(POS_19[ch][0], POS_19[ch][1]+0.09, ch,
                    ha='center', fontsize=6.5, zorder=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.09',
                              fc='white', alpha=0.72, lw=0))
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')
        return im

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes: ax.set_aspect('equal')
    draw_topo(axes[0], theta_adhd, f'ADHD (media {n_sub} sujetos)')
    im = draw_topo(axes[1], theta_ctrl, f'Control (media {n_sub} sujetos)')

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
    plt.colorbar(im, cax=cbar_ax, label='Potencia Theta media (µV²/Hz)')
    fig.suptitle(
        '5. Topomapa 2D — Potencia Banda Theta (4–8 Hz)\n'
        'Dataset EEG-ADHD | ADHD vs. Control | Media 25 sujetos por clase',
        fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(out_dir,'adhd_topomap_theta.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {os.path.join(out_dir,'adhd_topomap_theta.png')}")


# ──────────────────────────────────────────────────────────────
# ESTADÍSTICAS DESCRIPTIVAS
# ──────────────────────────────────────────────────────────────

def descriptive_stats(df):
    """Estadísticas PSD Theta canal F3 por clase (todos los sujetos)."""
    import csv as csv_mod
    print("\n── Estadísticas PSD Theta (4-8 Hz) canal F3 — todos los sujetos ──")
    print(f"  {'Clase':<10} {'n':>4} {'Media':>9} {'Med.':>9} {'SD':>9} "
          f"{'IC_lo':>9} {'IC_hi':>9} {'SW_p':>8} {'Normal':>7}")
    print("  " + "-"*78)
    rows = []
    for cls, color in [('ADHD', COL_ADHD), ('Control', COL_CTRL)]:
        ids  = df[df['Class'] == cls]['ID'].unique()
        vals = []
        for sid in ids:
            sub = clean_subject(df[df['ID'] == sid])
            sig = bandpass(sub['F3'].values)
            fw, Pxx = welch(sig, fs=FS, nperseg=256)
            vals.append(Pxx[(fw >= 4) & (fw <= 8)].mean())
        v  = np.array(vals); n = len(v)
        m  = v.mean(); sd = v.std(); med = np.median(v)
        ic = 1.96 * sd / np.sqrt(n)
        W, p = shapiro(v[:min(n, 5000)])
        normal = 'Sí' if p > 0.05 else 'No'
        print(f"  {cls:<10} {n:>4} {m:>9.2f} {med:>9.2f} {sd:>9.2f} "
              f"{m-ic:>9.2f} {m+ic:>9.2f} {p:>8.4f} {normal:>7}")
        rows.append([cls, n, round(m,3), round(med,3), round(sd,3),
                     round(m-ic,3), round(m+ic,3), round(W,4), round(p,4), normal])
    return rows


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='EDA — Dataset EEG-ADHD (5 figuras + estadísticas)')
    parser.add_argument('--csv_file', type=str, default='adhdata.csv',
                        help='Ruta al CSV del dataset EEG-ADHD')
    parser.add_argument('--out_dir',  type=str, default='fig',
                        help='Carpeta de salida para las figuras')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nCargando: {args.csv_file} ...")
    df = load_adhd(args.csv_file)
    n_subj = df['ID'].nunique()
    n_adhd = df[df['Class']=='ADHD']['ID'].nunique()
    n_ctrl = df[df['Class']=='Control']['ID'].nunique()
    print(f"  → {len(df):,} filas | {n_subj} sujetos ({n_adhd} ADHD, {n_ctrl} Control)")
    print(f"  → {len(EEG_COLS)} canales EEG | {len(EPOCX_AVAIL)} coinciden con EPOC X")

    print("\nGenerando figuras EDA:")
    fig1_class_distribution(df, args.out_dir)
    fig2_raw_signal(df, args.out_dir)
    fig3_psd(df, args.out_dir)
    fig4_heatmap(df, args.out_dir)
    fig5_topomap(df, args.out_dir)

    descriptive_stats(df)
    print(f"\n✅ EDA completado. Figuras guardadas en: {args.out_dir}")


if __name__ == '__main__':
    main()
