"""
01_MODMA_EDA.py
================
Análisis Exploratorio (EDA) — Dataset MODMA
Genera las 5 figuras requeridas por la tarea de datasets públicos.

Autora : Sofía Samaniego López
Tesis  : Clasificación de estrés, ansiedad y depresión mediante
         imágenes topográficas EEG (CNN + ViT) — FIAD UABC 2026
Dataset: Cai, H. et al. (2020). arXiv:2002.09283

USO:
    python 01_MODMA_EDA.py --mat_file ruta/al/archivo.mat --out_dir fig/

SALIDA (carpeta fig/):
    modma_clases.png          → Fig 1: distribución de clases (n=53)
    modma_raw_signal.png      → Fig 2: serie de tiempo cruda 5 s
    modma_psd.png             → Fig 3: PSD Welch 1-40 Hz
    modma_heatmap.png         → Fig 4: heatmap Pearson 14×14
    modma_topomap_alpha.png   → Fig 5: topomapa 2D banda Alpha

DEPENDENCIAS:
    pip install scipy matplotlib seaborn numpy
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────

# Mapeo EMOTIV EPOC X (14 canales) → índices 0-based en la malla EGI 128 ch
# Fuente: Cai et al. (2020) + tabla estándar EGI 128 ↔ sistema 10-20
EPOCX_EGI_MAP = {
    'AF3': 10, 'F7': 23,  'F3': 21,  'FC5': 32,
    'T7':  44, 'P7': 57,  'O1': 69,  'O2':  82,
    'P8':  95, 'T8': 107, 'FC6': 117, 'F4': 123,
    'F8': 121, 'AF4': 3,
}
CH_NAMES = list(EPOCX_EGI_MAP.keys())
CH_IDX   = list(EPOCX_EGI_MAP.values())

# Posiciones 2D normalizadas para el topomapa (proyección esférica estándar)
POS_2D = {
    'AF3': (-0.30,  0.72), 'AF4': ( 0.30,  0.72),
    'F7':  (-0.72,  0.45), 'F8':  ( 0.72,  0.45),
    'F3':  (-0.38,  0.50), 'F4':  ( 0.38,  0.50),
    'FC5': (-0.60,  0.22), 'FC6': ( 0.60,  0.22),
    'T7':  (-0.87,  0.00), 'T8':  ( 0.87,  0.00),
    'P7':  (-0.72, -0.45), 'P8':  ( 0.72, -0.45),
    'O1':  (-0.30, -0.78), 'O2':  ( 0.30, -0.78),
}

# Estilo global
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ──────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────

def load_modma(mat_path: str):
    """
    Carga un archivo .mat de MODMA y devuelve la señal EEG de 14 canales
    re-referenciada y filtrada (1-40 Hz), junto con la frecuencia de muestreo.

    Parameters
    ----------
    mat_path : str
        Ruta al archivo .mat (e.g. '02010002rest_20150416_1017_.mat')

    Returns
    -------
    eeg_f : ndarray (14, n_samples)   Señal filtrada en µV
    fs    : float                     Frecuencia de muestreo (Hz)
    """
    mat  = loadmat(mat_path)
    # La clave principal varía por archivo; tomamos la primera que no sea metadato
    key  = [k for k in mat.keys() if not k.startswith('_')][0]
    raw  = mat[key]                           # (129, n_samples)
    fs   = float(mat['samplingRate'][0, 0])   # 250.0 Hz

    # Extrae los 14 canales EPOC X
    eeg14 = raw[CH_IDX, :]

    # Re-referencia promedio común (CAR)
    eeg14 = eeg14 - eeg14.mean(axis=0, keepdims=True)

    # Filtro pasa-banda 1-40 Hz (Butterworth orden 4)
    b, a  = butter(4, [1.0 / (fs / 2), 40.0 / (fs / 2)], btype='band')
    eeg_f = filtfilt(b, a, eeg14, axis=1)

    return eeg_f, fs


def compute_alpha_power(eeg_f, fs):
    """PSD Welch por canal; devuelve potencia media en banda Alpha (8-13 Hz)."""
    alpha = []
    for sig in eeg_f:
        f, Pxx = welch(sig, fs=fs, nperseg=512)
        alpha.append(Pxx[(f >= 8) & (f <= 13)].mean())
    return np.array(alpha)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


# ──────────────────────────────────────────────────────────────
# FIGURAS EDA
# ──────────────────────────────────────────────────────────────

def fig1_class_distribution(out_dir):
    """Figura 1 — Distribución de clases MODMA (n=53, estadísticas del paper)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    classes = ['Control\nSaludable (HC)', 'Depresión Mayor\n(MDD)']
    counts  = [29, 24]
    colors  = ['#2ECC71', '#E67E22']

    bars = ax.bar(classes, counts, color=colors, edgecolor='white',
                  linewidth=1.2, width=0.5)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f'n={n}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Cantidad de Sujetos', fontsize=11)
    ax.set_ylim(0, 35)
    ax.axhline(26.5, color='gray', ls='--', lw=1, alpha=0.6,
               label='Distribución ideal (50/50)')
    ax.set_title('1. Distribución de Clases — Dataset MODMA (n = 53)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'modma_clases.png'))


def fig2_raw_signal(eeg_f, fs, out_dir):
    """Figura 2 — Serie de tiempo cruda (5 s), 14 canales."""
    t_start, t_end = 30, 35        # evitar artefactos de inicio
    s0, s1 = int(t_start * fs), int(t_end * fs)
    seg = eeg_f[:, s0:s1]
    t   = np.linspace(0, 5, s1 - s0)

    fig, ax = plt.subplots(figsize=(10, 6))
    spacing = 60
    cmap    = plt.cm.tab20(np.linspace(0, 1, 14))
    for i, (ch, sig) in enumerate(zip(CH_NAMES, seg)):
        offset = i * spacing
        ax.plot(t, sig + offset, color=cmap[i], lw=0.8)
        ax.text(-0.08, offset, ch, ha='right', va='center',
                fontsize=7.5, color=cmap[i], fontweight='bold')

    ax.set_xlabel('Tiempo (s)', fontsize=11)
    ax.set_title(
        '2. Serie de Tiempo EEG (5 s) — 14 Canales EPOC X\n'
        'Dataset MODMA, sujeto 02010002 (HC)',
        fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 5)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'modma_raw_signal.png'))


def fig3_psd(eeg_f, fs, out_dir):
    """Figura 3 — PSD Welch 1-40 Hz con bandas sombreadas."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap    = plt.cm.tab20(np.linspace(0, 1, 14))

    for i, (ch, sig) in enumerate(zip(CH_NAMES, eeg_f)):
        f, Pxx = welch(sig, fs=fs, nperseg=512)
        mask   = (f >= 1) & (f <= 40)
        ax.semilogy(f[mask], Pxx[mask], color=cmap[i], lw=1.1,
                    label=ch, alpha=0.85)

    bands = [
        ('Delta',  1,  4, '#AED6F1', 0.25),
        ('Theta',  4,  8, '#A9DFBF', 0.25),
        ('Alpha',  8, 13, '#F9E79F', 0.30),
        ('Beta',  13, 30, '#FADBD8', 0.20),
    ]
    for name, lo, hi, col, alpha in bands:
        ax.axvspan(lo, hi, color=col, alpha=alpha, label=name)

    ax.set_xlabel('Frecuencia (Hz)', fontsize=11)
    ax.set_ylabel('PSD (µV²/Hz)', fontsize=11)
    ax.set_xlim(1, 40)
    ax.set_title(
        '3. PSD Welch (1–40 Hz) — 14 Canales EPOC X\n'
        'Dataset MODMA, sujeto 02010002 (HC)',
        fontsize=11, fontweight='bold')
    ax.legend(ncol=3, fontsize=7.5, loc='upper right')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'modma_psd.png'))


def fig4_heatmap(eeg_f, out_dir):
    """Figura 4 — Heatmap de correlación Pearson 14×14."""
    corr = np.corrcoef(eeg_f)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=CH_NAMES, yticklabels=CH_NAMES,
        vmin=-1, vmax=1, linewidths=0.4, ax=ax,
        annot_kws={'size': 7}, cbar_kws={'shrink': 0.8},
    )
    ax.set_title(
        '4. Correlación Pearson 14×14 — EPOC X Channels\n'
        'Dataset MODMA, sujeto 02010002 (HC)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'modma_heatmap.png'))


def fig5_topomap(eeg_f, fs, out_dir):
    """Figura 5 — Topomapa 2D potencia banda Alpha (8-13 Hz)."""
    alpha_power = compute_alpha_power(eeg_f, fs)

    xs = np.array([POS_2D[ch][0] for ch in CH_NAMES])
    ys = np.array([POS_2D[ch][1] for ch in CH_NAMES])

    # Interpolación cúbica sobre cuadrícula 300×300
    xi  = np.linspace(-1.1, 1.1, 300)
    yi  = np.linspace(-1.1, 1.1, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi  = griddata((xs, ys), alpha_power, (Xi, Yi), method='cubic')
    Zi[Xi ** 2 + Yi ** 2 > 1.0 ** 2] = np.nan   # máscara circular

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    norm = Normalize(vmin=np.nanmin(Zi), vmax=np.nanmax(Zi))
    cmap_topo = cm.RdYlBu_r
    im = ax.contourf(Xi, Yi, Zi, levels=64, cmap=cmap_topo, norm=norm, zorder=1)

    # Contorno de cabeza
    head = Circle((0, 0), 1.0, fill=False, edgecolor='k', linewidth=2.5, zorder=4)
    ax.add_patch(head)
    # Nariz
    ax.plot([-0.07, 0, 0.07], [0.98, 1.12, 0.98], 'k-', lw=2.5, zorder=5)
    # Orejas
    for s in [-1, 1]:
        ax.plot([s * 1.0, s * 1.08, s * 1.08, s * 1.0],
                [0.12, 0.08, -0.08, -0.12], 'k-', lw=2.5, zorder=5)

    # Electrodos
    ax.scatter(xs, ys, c=alpha_power, cmap=cmap_topo, norm=norm,
               s=120, edgecolors='k', linewidths=1.2, zorder=6)
    for ch in CH_NAMES:
        ax.annotate(ch, xy=POS_2D[ch],
                    xytext=(POS_2D[ch][0] + 0.01, POS_2D[ch][1] + 0.08),
                    fontsize=7.5, ha='center', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              alpha=0.7, lw=0))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Potencia Alpha media (µV²/Hz)', fontsize=9)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    ax.set_title(
        '5. Topomapa 2D — Potencia Banda Alpha (8–13 Hz)\n'
        '14 Canales EPOC X | Dataset MODMA, sujeto 02010002 (HC)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'modma_topomap_alpha.png'))


# ──────────────────────────────────────────────────────────────
# ESTADÍSTICAS DESCRIPTIVAS (imprime en consola / guarda CSV)
# ──────────────────────────────────────────────────────────────

def descriptive_stats(eeg_f, fs, out_dir):
    """Calcula estadísticas descriptivas de PSD Alpha por canal y las exporta."""
    import csv
    from scipy.stats import shapiro

    alpha_power = compute_alpha_power(eeg_f, fs)

    print("\n── Estadísticas PSD Alpha (8-13 Hz) — 14 canales ──")
    print(f"  Media  : {alpha_power.mean():.4f} µV²/Hz")
    print(f"  Mediana: {np.median(alpha_power):.4f}")
    print(f"  SD     : {alpha_power.std():.4f}")
    print(f"  Mín    : {alpha_power.min():.4f}  Máx: {alpha_power.max():.4f}")
    n = len(alpha_power)
    ic_lo = alpha_power.mean() - 1.96 * alpha_power.std() / np.sqrt(n)
    ic_hi = alpha_power.mean() + 1.96 * alpha_power.std() / np.sqrt(n)
    print(f"  IC 95%: [{ic_lo:.4f}, {ic_hi:.4f}]")
    W, p = shapiro(alpha_power)
    print(f"  Shapiro-Wilk: W={W:.4f}, p={p:.4f} "
          f"({'normal' if p > 0.05 else 'no normal'})")

    csv_path = os.path.join(out_dir, 'modma_alpha_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['canal', 'alpha_power_uV2Hz'])
        for ch, val in zip(CH_NAMES, alpha_power):
            writer.writerow([ch, f'{val:.6f}'])
    print(f"\n  CSV exportado: {csv_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='EDA — Dataset MODMA (5 figuras + estadísticas)')
    parser.add_argument(
        '--mat_file', type=str,
        default='02010002rest_20150416_1017_.mat',
        help='Ruta al archivo .mat de MODMA')
    parser.add_argument(
        '--out_dir', type=str, default='fig',
        help='Carpeta de salida para las figuras')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nCargando: {args.mat_file}")

    eeg_f, fs = load_modma(args.mat_file)
    print(f"  → {len(CH_NAMES)} canales EPOC X, {eeg_f.shape[1]} muestras, "
          f"fs={fs} Hz, duración={eeg_f.shape[1]/fs/60:.2f} min")

    print("\nGenerando figuras EDA:")
    fig1_class_distribution(args.out_dir)
    fig2_raw_signal(eeg_f, fs, args.out_dir)
    fig3_psd(eeg_f, fs, args.out_dir)
    fig4_heatmap(eeg_f, args.out_dir)
    fig5_topomap(eeg_f, fs, args.out_dir)

    descriptive_stats(eeg_f, fs, args.out_dir)
    print("\n✅ EDA completado. Figuras guardadas en:", args.out_dir)


if __name__ == '__main__':
    main()
