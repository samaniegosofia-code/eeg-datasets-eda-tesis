"""
03_DASPS_EDA.py
===============
Análisis Exploratorio (EDA) — Dataset DASPS
(Database for Anxious States based on Psychological Stimulation)
Genera las 5 figuras requeridas por la tarea de datasets públicos.

Autora : Sofía Samaniego López
Tesis  : Clasificación de estrés, ansiedad y depresión mediante
         imágenes topográficas EEG (CNN + ViT) — FIAD UABC 2026
Dataset: Aytar, A. & Kolyak, A. (2020). IEEE DataPort.
         https://ieee-dataport.org/open-access/dasps-database

USO:
    python 03_DASPS_EDA.py --edf_file S01.edf --out_dir fig/

    Para procesar múltiples sujetos:
    for f in S*.edf; do python 03_DASPS_EDA.py --edf_file $f; done

SALIDA (carpeta fig/):
    dasps_dist_clases.png      → Fig 1: distribución binaria y multiclase (N=138)
    dasps_raw_af3_af4.png      → Fig 2: serie filtrada AF3/AF4 Normal vs Ansioso
    dasps_psd_alpha_beta.png   → Fig 3: PSD Alpha y Beta Normal vs Ansioso
    dasps_heatmap_pearson.png  → Fig 4: heatmap Pearson 14×14
    dasps_topomap_alpha.png    → Fig 5: topomapa 2D banda Alpha

ESTRUCTURA DEL DATASET:
    - 23 sujetos (S01–S23), un .edf por sujeto
    - Grabado con EMOTIV EPOC X: 14 canales EEG + COUNTER + INTERPOLATED
      + 14 canales CQ + GYROX + GYROY + MARKER = 36 señales totales
    - fs = 128 Hz | ~329 s por sujeto
    - Protocolo: 6 escenarios × 2 fases (escuchar 15 s + recordar 15 s)
      → 12 segmentos de 15 s por sujeto
    - Etiquetas: escenarios 1-3 → Normal, escenarios 4-6 → Ansioso
      (Multiclase: Normal / Leve / Moderada / Severa según puntuación SAM)

NOTA SOBRE CALIDAD DE CONTACTO:
    El EMOTIV EPOC X es sensible al contacto de los electrodos. Algunos
    canales pueden tener señal ruidosa (std > 500 µV) en ciertos sujetos.
    El script los detecta automáticamente y los excluye del topomapa e
    identifica con asterisco (*) en el heatmap.

DEPENDENCIAS:
    pip install numpy scipy matplotlib seaborn
"""

import argparse
import os
import struct
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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

EEG_CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
N_SIGNALS   = 36
SPR         = 128     # samples per record
FS          = 128     # Hz
HEADER_BYTES= 9472    # calculated: 256 + 36*256
NOISE_THR   = 500     # µV std threshold for noisy channel detection

# Posiciones 2D normalizadas para topomapa
POS_2D = {
    'AF3':(-0.30,  0.72), 'AF4':( 0.30,  0.72),
    'F7': (-0.72,  0.45), 'F8': ( 0.72,  0.45),
    'F3': (-0.38,  0.50), 'F4': ( 0.38,  0.50),
    'FC5':(-0.60,  0.22), 'FC6':( 0.60,  0.22),
    'T7': (-0.87,  0.00), 'T8': ( 0.87,  0.00),
    'P7': (-0.72, -0.45), 'P8': ( 0.72, -0.45),
    'O1': (-0.30, -0.78), 'O2': ( 0.30, -0.78),
}

COL_NORM = '#2ECC71'
COL_ANS  = '#E74C3C'
BAND_COLS = {
    'Delta': '#AED6F1', 'Theta': '#A9DFBF',
    'Alpha': '#F9E79F', 'Beta':  '#FADBD8',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ──────────────────────────────────────────────────────────────
# CARGA DE DATOS EDF
# ──────────────────────────────────────────────────────────────

def load_edf(filepath: str):
    """
    Lee un archivo .edf de DASPS (EMOTIV EPOC X, 36 señales) y devuelve:
      - eeg_f  : ndarray (14, n_samples) señal filtrada 1-40 Hz
      - eeg_dc : ndarray (14, n_samples) señal con sólo DC removido
      - good_mask : ndarray bool (14,) True = canal limpio
      - n_records : int
    """
    # ── Leer cabecera para obtener n_records ──────────────────
    with open(filepath, 'rb') as f:
        hdr = f.read(256).decode('latin-1')
    n_records = int(hdr[236:244].strip())

    # ── Leer datos ────────────────────────────────────────────
    with open(filepath, 'rb') as f:
        f.seek(HEADER_BYTES)
        data = np.zeros((N_SIGNALS, n_records * SPR), dtype=np.float32)
        for rec in range(n_records):
            for sig in range(N_SIGNALS):
                raw = f.read(SPR * 2)
                data[sig, rec*SPR:(rec+1)*SPR] = (
                    np.frombuffer(raw, dtype='<i2').astype(np.float32)
                )

    # ── Extraer 14 canales EEG (índices 2-15) ─────────────────
    eeg_raw  = data[2:16, :]
    eeg_dc   = eeg_raw - eeg_raw.mean(axis=1, keepdims=True)

    # ── Filtro pasa-banda 1-40 Hz (Butterworth orden 4) ───────
    b, a   = butter(4, [1.0/(FS/2), 40.0/(FS/2)], btype='band')
    eeg_f  = filtfilt(b, a, eeg_dc, axis=1)

    # ── Detección de canales ruidosos ─────────────────────────
    stds      = eeg_dc.std(axis=1)
    good_mask = stds < NOISE_THR

    return eeg_f, eeg_dc, good_mask, n_records


def get_trial_onsets(total_samples: int, n_trials: int = 12,
                     seg_len: int = None, baseline: int = None):
    """Estima los onsets de cada segmento de 15 s (protocolo DASPS)."""
    if seg_len  is None: seg_len  = int(15  * FS)
    if baseline is None: baseline = int(5   * FS)
    inter = max(0, (total_samples - baseline - n_trials * seg_len) // n_trials)
    return [baseline + i * (seg_len + inter) for i in range(n_trials)]


# ──────────────────────────────────────────────────────────────
# FIGURAS EDA
# ──────────────────────────────────────────────────────────────

def fig1_class_distribution(out_dir):
    """Figura 1 — Distribución de clases del dataset DASPS completo (N=138)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # — Binaria —
    ax = axes[0]
    classes    = ['Normal\n(n=65)', 'Ansioso\n(n=73)']
    counts_bin = [65, 73]
    bars = ax.bar(classes, counts_bin, color=[COL_NORM, COL_ANS],
                  edgecolor='white', linewidth=1.2, width=0.45)
    for bar, n in zip(bars, counts_bin):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'n={n}\n({n/138*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(69, color='gray', ls='--', lw=1, alpha=0.6, label='Ideal (50/50)')
    ax.set_ylim(0, 90)
    ax.set_ylabel('Número de ensayos', fontsize=10)
    ax.set_title('Clasificación Binaria', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    # — Multiclase (4 niveles) —
    ax = axes[1]
    sev_labels = ['Normal\n(n=65)', 'Leve\n(n=43)', 'Moderada\n(n=15)', 'Severa\n(n=15)']
    sev_counts = [65, 43, 15, 15]
    sev_colors = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C']
    bars2 = ax.bar(sev_labels, sev_counts, color=sev_colors,
                   edgecolor='white', linewidth=1.2, width=0.5)
    for bar, n in zip(bars2, sev_counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{n}\n({n/138*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 82)
    ax.set_ylabel('Número de ensayos', fontsize=10)
    ax.set_title('Clasificación Multiclase (4 niveles)', fontsize=10, fontweight='bold')

    fig.suptitle(
        '1. Distribución de Clases — Dataset DASPS (N = 138 ensayos, 23 sujetos)\n'
        'Etiquetas SAM: Normal / Ansiedad leve / Moderada / Severa',
        fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'dasps_dist_clases.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


def fig2_raw_signal(eeg_f, good_mask, trial_onsets, out_dir):
    """Figura 2 — Serie de tiempo filtrada AF3 y AF4, Normal vs. Ansioso."""
    seg_len = int(15 * FS)
    af3_idx = EEG_CH.index('AF3')
    af4_idx = EEG_CH.index('AF4')
    t15     = np.linspace(0, 15, seg_len)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey='row')

    for row, (ch_idx, ch_name) in enumerate([(af3_idx,'AF3'), (af4_idx,'AF4')]):
        quality = ("✓ buena calidad" if good_mask[ch_idx]
                   else "⚠ contacto deficiente (artefacto)")
        col_ch  = '#1A5276' if good_mask[ch_idx] else '#C0392B'

        for col, (tr_idx, label, color) in enumerate(
                [(0, 'Normal', COL_NORM), (6, 'Ansioso', COL_ANS)]):
            onset = trial_onsets[tr_idx]
            end   = onset + seg_len
            seg   = eeg_f[ch_idx, onset:end] if end <= eeg_f.shape[1] else eeg_f[ch_idx, -seg_len:]
            axes[row, col].plot(t15[:len(seg)], seg, color=color, lw=0.9)
            axes[row, col].set_title(
                f'{ch_name} — {label}  ({quality})',
                fontsize=9, color=col_ch, fontweight='bold')
            axes[row, col].set_xlabel('Tiempo (s)', fontsize=9)
            axes[row, col].set_xlim(0, 15)
            if col == 0:
                axes[row, col].set_ylabel('Amplitud (µV)', fontsize=9)

    fig.suptitle(
        '2. Serie de Tiempo Filtrada (1–40 Hz) — Canales AF3 y AF4\n'
        'Dataset DASPS, Sujeto S01 | Normal vs. Ansioso (15 s/segmento)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'dasps_raw_af3_af4.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


def fig3_psd_alpha_beta(eeg_f, good_mask, trial_onsets, out_dir):
    """Figura 3 — PSD Welch bandas Alpha y Beta, Normal vs. Ansioso."""
    seg_len  = int(15 * FS)
    good_idx = [i for i, g in enumerate(good_mask) if g]
    good_chs = [EEG_CH[i] for i in good_idx]

    def concat_segs(trial_range):
        segs = []
        for i in trial_range:
            o, e = trial_onsets[i], trial_onsets[i] + seg_len
            if e <= eeg_f.shape[1]:
                segs.append(eeg_f[np.array(good_idx), o:e])
        return np.concatenate(segs, axis=1) if segs else eeg_f[np.array(good_idx), :seg_len]

    norm_cat = concat_segs(range(6))
    ans_cat  = concat_segs(range(6, 12))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(good_chs)))

    for ax, (cat, label, color) in zip(
            axes, [(norm_cat,'Normal',COL_NORM), (ans_cat,'Ansioso',COL_ANS)]):
        for i, (ch, sig) in enumerate(zip(good_chs, cat)):
            fw, Pxx = welch(sig, fs=FS, nperseg=256)
            mask    = (fw >= 1) & (fw <= 40)
            ax.semilogy(fw[mask], Pxx[mask], color=cmap[i],
                        lw=1.2, label=ch, alpha=0.85)
        for band, (lo, hi) in [('Alpha',(8,13)), ('Beta',(13,30))]:
            ax.axvspan(lo, hi, color=BAND_COLS[band], alpha=0.25, label=band)
        ax.set_xlabel('Frecuencia (Hz)', fontsize=10)
        ax.set_ylabel('PSD (µV²/Hz)', fontsize=10)
        ax.set_xlim(1, 40)
        ax.set_title(f'Estado: {label}', fontsize=10,
                     fontweight='bold', color=color)
        ax.legend(ncol=2, fontsize=7.5, loc='upper right')

    fig.suptitle(
        '3. PSD Welch — Bandas Alpha (8–13 Hz) y Beta (13–30 Hz)\n'
        'Dataset DASPS, Sujeto S01 | Canales con buena calidad de contacto',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'dasps_psd_alpha_beta.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


def fig4_heatmap(eeg_f, good_mask, out_dir):
    """Figura 4 — Heatmap correlación Pearson 14×14."""
    corr14    = np.corrcoef(eeg_f)
    ch_labels = [f'{ch}{"*" if not good_mask[i] else ""}' for i, ch in enumerate(EEG_CH)]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(corr14, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=ch_labels, yticklabels=ch_labels,
                vmin=-1, vmax=1, linewidths=0.4, ax=ax,
                annot_kws={'size': 6.5}, cbar_kws={'shrink': 0.8})
    ax.set_title(
        '4. Correlación Pearson 14×14 Canales — Dataset DASPS\n'
        'Sujeto S01 | * = canal con contacto deficiente (etiqueta en rojo)',
        fontsize=10, fontweight='bold')
    for tick, is_good in zip(ax.get_xticklabels(), good_mask):
        tick.set_color('#C0392B' if not is_good else '#1A5276')
    for tick, is_good in zip(ax.get_yticklabels(), good_mask):
        tick.set_color('#C0392B' if not is_good else '#1A5276')
    plt.tight_layout()
    path = os.path.join(out_dir, 'dasps_heatmap_pearson.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


def fig5_topomap(eeg_f, good_mask, out_dir):
    """Figura 5 — Topomapa 2D potencia banda Alpha (8-13 Hz)."""
    alpha_pow = []
    for sig in eeg_f:
        fw, Pxx = welch(sig, fs=FS, nperseg=256)
        alpha_pow.append(Pxx[(fw >= 8) & (fw <= 13)].mean())
    alpha_pow = np.array(alpha_pow)

    good_idx  = [i for i, g in enumerate(good_mask) if g]
    xs_good   = np.array([POS_2D[EEG_CH[i]][0] for i in good_idx])
    ys_good   = np.array([POS_2D[EEG_CH[i]][1] for i in good_idx])
    vals_good = alpha_pow[np.array(good_idx)]

    xi  = np.linspace(-1.1, 1.1, 300)
    yi  = np.linspace(-1.1, 1.1, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi  = griddata((xs_good, ys_good), vals_good, (Xi, Yi), method='cubic')
    Zi[Xi**2 + Yi**2 > 1.0**2] = np.nan

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect('equal')
    norm_c = Normalize(vmin=np.nanmin(Zi), vmax=np.nanmax(Zi))
    cmap_t = cm.RdYlBu_r
    im = ax.contourf(Xi, Yi, Zi, levels=64, cmap=cmap_t, norm=norm_c, zorder=1)

    head = Circle((0,0), 1.0, fill=False, edgecolor='k', linewidth=2.5, zorder=4)
    ax.add_patch(head)
    ax.plot([-0.07, 0, 0.07], [0.98, 1.12, 0.98], 'k-', lw=2.5, zorder=5)
    for s in [-1, 1]:
        ax.plot([s*1.0, s*1.08, s*1.08, s*1.0],
                [0.12, 0.08, -0.08, -0.12], 'k-', lw=2.5, zorder=5)

    for i, ch in enumerate(EEG_CH):
        x, y = POS_2D[ch]
        if good_mask[i]:
            ax.scatter(x, y, c=[alpha_pow[i]], cmap=cmap_t, norm=norm_c,
                       s=130, edgecolors='k', linewidths=1.2, zorder=6)
            ax.text(x, y+0.09, ch, ha='center', fontsize=7.5,
                    fontweight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', alpha=0.75, lw=0))
        else:
            ax.scatter(x, y, s=100, c='#BDC3C7', edgecolors='#7F8C8D',
                       marker='X', linewidths=1, zorder=6)
            ax.text(x, y+0.09, f'{ch}*', ha='center', fontsize=7,
                    color='#7F8C8D', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.10', fc='white', alpha=0.6, lw=0))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Potencia Alpha media (µV²/Hz)', fontsize=9)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis('off')
    ax.set_title(
        '5. Topomapa 2D — Potencia Banda Alpha (8–13 Hz)\n'
        'Dataset DASPS, Sujeto S01 | * = contacto deficiente (excluido de interpolación)',
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'dasps_topomap_alpha.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


# ──────────────────────────────────────────────────────────────
# ESTADÍSTICAS DESCRIPTIVAS
# ──────────────────────────────────────────────────────────────

def descriptive_stats(eeg_f, good_mask, trial_onsets):
    """Calcula estadísticas PSD Alpha por estado (Normal / Ansioso)."""
    seg_len  = int(15 * FS)
    good_idx = [i for i, g in enumerate(good_mask) if g]

    results = {}
    for label, t_range in [('Normal', range(6)), ('Ansioso', range(6,12))]:
        vals = []
        for i in t_range:
            o, e = trial_onsets[i], trial_onsets[i] + seg_len
            if e <= eeg_f.shape[1]:
                for gi in good_idx:
                    fw, Pxx = welch(eeg_f[gi, o:e], fs=FS, nperseg=128)
                    vals.append(Pxx[(fw>=8)&(fw<=13)].mean())
        vals   = np.array(vals)
        n      = len(vals)
        m, med, sd = vals.mean(), np.median(vals), vals.std()
        ic_lo  = m - 1.96*sd/np.sqrt(n)
        ic_hi  = m + 1.96*sd/np.sqrt(n)
        W, p   = shapiro(vals) if n <= 5000 else (np.nan, np.nan)
        results[label] = dict(n=n, mean=m, median=med, sd=sd,
                              min=vals.min(), max=vals.max(),
                              ic_lo=ic_lo, ic_hi=ic_hi, sw_W=W, sw_p=p)

    print("\n── Estadísticas PSD Alpha (8-13 Hz) por estado — Sujeto S01 ──")
    for lbl, r in results.items():
        print(f"  {lbl:<10} n={r['n']:3d}  media={r['mean']:.3f}  SD={r['sd']:.3f}  "
              f"IC=[{r['ic_lo']:.3f},{r['ic_hi']:.3f}]  "
              f"SW p={r['sw_p']:.3f}")
    return results


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='EDA — Dataset DASPS (5 figuras + estadísticas)')
    parser.add_argument('--edf_file', type=str, default='S01.edf',
                        help='Ruta al archivo .edf de DASPS (e.g. S01.edf)')
    parser.add_argument('--out_dir',  type=str, default='fig',
                        help='Carpeta de salida para las figuras')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nCargando: {args.edf_file}")

    eeg_f, eeg_dc, good_mask, n_records = load_edf(args.edf_file)
    total_s = n_records * SPR / FS
    print(f"  → 14 canales EEG, {eeg_f.shape[1]} muestras, "
          f"fs={FS} Hz, duración={total_s:.1f} s = {total_s/60:.2f} min")
    good_chs = [EEG_CH[i] for i, g in enumerate(good_mask) if g]
    print(f"  → Canales limpios ({good_mask.sum()}/14): {good_chs}")

    trial_onsets = get_trial_onsets(eeg_f.shape[1])
    print(f"  → Onsets estimados (s): {[f'{o/FS:.1f}' for o in trial_onsets]}")

    print("\nGenerando figuras EDA:")
    fig1_class_distribution(args.out_dir)
    fig2_raw_signal(eeg_f, good_mask, trial_onsets, args.out_dir)
    fig3_psd_alpha_beta(eeg_f, good_mask, trial_onsets, args.out_dir)
    fig4_heatmap(eeg_f, good_mask, args.out_dir)
    fig5_topomap(eeg_f, good_mask, args.out_dir)

    descriptive_stats(eeg_f, good_mask, trial_onsets)
    print(f"\n✅ EDA completado. Figuras guardadas en: {args.out_dir}")


if __name__ == '__main__':
    main()
