"""
04_DEAP_EDA.py
==============
Análisis Exploratorio (EDA) — Dataset DEAP
(Database for Emotion Analysis Using Physiological Signals)
Genera las 5 figuras requeridas por la tarea de datasets públicos.

Autora : Sofía Samaniego López
Tesis  : Clasificación de estrés, ansiedad y depresión mediante
         imágenes topográficas EEG (CNN + ViT) — FIAD UABC 2026
Dataset: Koelstra, S. et al. (2012). IEEE Trans. Affective Computing, 3(1), 18-31.
         https://doi.org/10.1109/T-AFFC.2011.15
         Acceso: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ (EULA académica)

USO:
    python 04_DEAP_EDA.py --dat_file s02.dat --out_dir fig/

SALIDA (carpeta fig/):
    deap_scatter_valence_arousal.png  → Fig 1: dispersión Valencia vs. Arousal + cuadrantes
    deap_raw_t7_t8.png                → Fig 2: serie de tiempo T7/T8, HVHA vs LVHA
    deap_psd_alpha_beta.png           → Fig 3: PSD Alpha y Beta por cuadrante
    deap_14ch_epocx.png               → Fig 4: extracción 14ch EPOC X + heatmap Pearson
    deap_topomap_alpha.png            → Fig 5: topomapa 2D Alpha HVHA vs LVHA

ESTRUCTURA DEL ARCHIVO .dat:
    - Python pickle (protocol 2), cargar con encoding='latin1'
    - 'data'  : ndarray (40, 40, 8064)
               40 ensayos × 40 canales × 8064 muestras
               Canales 0-31: EEG 10-20 | Canales 32-39: señales periféricas
               8064 muestras @ 128 Hz = 63 s (3 s baseline + 60 s estímulo)
    - 'labels': ndarray (40, 4)
               Columnas: [valence, arousal, dominance, liking] en escala SAM 1-9

MAPEO 14 CANALES EPOC X → DEAP:
    AF3→1, F7→3, F3→2, FC5→4, T7→7, P7→11,
    O1→13, O2→30, P8→28, T8→24, FC6→22, F4→19, F8→20, AF4→17

NOTA ESTADÍSTICA:
    Las distribuciones PSD Alpha son NO normales (Shapiro-Wilk p<0.0001)
    → usar Mann-Whitney U en comparaciones entre cuadrantes.

DEPENDENCIAS:
    pip install numpy scipy matplotlib seaborn
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from scipy.signal import welch
from scipy.stats import shapiro

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────

FS = 128   # Hz (preprocesado)
BASELINE_SAMPLES = 384   # 3 s × 128 Hz → remover baseline

# Canales DEAP (32 EEG en orden del paper)
CH_32 = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
         'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
         'Fp2','AF4','Fz','F4','F8','FC2','FC6','C4',
         'T8','CP2','CP6','P4','P8','PO4','O2','F0']

# Mapeo EPOC X 14ch → índice 0-based en DEAP
EPOCX_IDX = {
    'AF3':1,  'F7':3,  'F3':2,  'FC5':4,
    'T7':7,   'P7':11, 'O1':13, 'O2':30,
    'P8':28,  'T8':24, 'FC6':22,'F4':19,
    'F8':20,  'AF4':17,
}
CH_EPOCX = list(EPOCX_IDX.keys())
IDX14    = list(EPOCX_IDX.values())

# Posiciones 2D para topomapa
POS_2D = {
    'AF3':(-0.30,  0.72), 'AF4':( 0.30,  0.72),
    'F7': (-0.72,  0.45), 'F8': ( 0.72,  0.45),
    'F3': (-0.38,  0.50), 'F4': ( 0.38,  0.50),
    'FC5':(-0.60,  0.22), 'FC6':( 0.60,  0.22),
    'T7': (-0.87,  0.00), 'T8': ( 0.87,  0.00),
    'P7': (-0.72, -0.45), 'P8': ( 0.72, -0.45),
    'O1': (-0.30, -0.78), 'O2': ( 0.30, -0.78),
}

# Posiciones de los 32 canales DEAP para el diagrama de cabeza
POS_32 = {
    'Fp1':(-0.18,0.92),'Fp2':(0.18,0.92),
    'AF3':(-0.30,0.72),'AF4':(0.30,0.72),
    'F7': (-0.72,0.45),'F3': (-0.38,0.50),'Fz': (0.00,0.65),
    'F4': (0.38,0.50), 'F8': (0.72,0.45),
    'FC5':(-0.60,0.22),'FC1':(-0.22,0.25),'FC2':(0.22,0.25),'FC6':(0.60,0.22),
    'C3': (-0.45,0.00),'Cz': (0.00,0.00), 'C4': (0.45,0.00),
    'T7': (-0.87,0.00),'T8': (0.87,0.00),
    'CP5':(-0.60,-0.22),'CP1':(-0.22,-0.25),'CP2':(0.22,-0.25),'CP6':(0.60,-0.22),
    'P7': (-0.72,-0.45),'P3':(-0.38,-0.50),'Pz': (0.00,-0.65),
    'P4': (0.38,-0.50), 'P8':(0.72,-0.45),
    'PO3':(-0.30,-0.68),'PO4':(0.30,-0.68),
    'O1': (-0.18,-0.92),'Oz':(0.00,-0.95),'O2':(0.18,-0.92),
    'F0': (0.00,0.50),
}

QCOL = {'HVHA':'#2ECC71','HVLA':'#3498DB','LVHA':'#E74C3C','LVLA':'#9B59B6'}
QLAB = {'HVHA':'HV-HA\n(Feliz)',    'HVLA':'HV-LA\n(Relajado)',
        'LVHA':'LV-HA\n(Estresado)','LVLA':'LV-LA\n(Triste)'}
BAND_COLS = {'Alpha':'#F9E79F','Beta':'#FADBD8'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ──────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────

def load_deap(dat_path: str):
    """
    Carga un archivo .dat de DEAP y devuelve:
      eeg14   : ndarray (40, 14, 7680) — 14 canales EPOC X, 60 s (sin baseline)
      val     : ndarray (40,)  — valence SAM 1-9
      aro     : ndarray (40,)  — arousal SAM 1-9
      quadrant: ndarray (40,) str — 'HVHA'|'HVLA'|'LVHA'|'LVLA'
    """
    with open(dat_path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    data   = d['data']    # (40, 40, 8064)
    labels = d['labels']  # (40, 4)

    # Extraer 14 canales EPOC X, quitar 3 s de baseline
    eeg14 = data[:, IDX14, BASELINE_SAMPLES:]  # (40, 14, 7680)

    val = labels[:, 0]
    aro = labels[:, 1]
    val_bin = (val >= 5).astype(int)
    aro_bin = (aro >= 5).astype(int)
    quadrant = np.array([
        'HVHA' if v==1 and a==1 else
        'HVLA' if v==1 and a==0 else
        'LVHA' if v==0 and a==1 else 'LVLA'
        for v, a in zip(val_bin, aro_bin)
    ])
    return eeg14, val, aro, quadrant


def alpha_power_per_channel(eeg14, trial_indices):
    """PSD Alpha media (8-13 Hz) por canal, promediada entre ensayos."""
    alpha = np.zeros(14)
    for ti in trial_indices:
        for ci in range(14):
            fw, Pxx = welch(eeg14[ti, ci, :], fs=FS, nperseg=256)
            alpha[ci] += Pxx[(fw >= 8) & (fw <= 13)].mean()
    return alpha / len(trial_indices)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {path}")


# ──────────────────────────────────────────────────────────────
# FIGURAS EDA
# ──────────────────────────────────────────────────────────────

def fig1_scatter_valence_arousal(val, aro, quadrant, out_dir):
    """Figura 1 — Dispersión Valencia vs. Arousal con cuadrantes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — Scatter —
    ax = axes[0]
    for q in ['HVHA','HVLA','LVHA','LVLA']:
        mask = quadrant == q
        ax.scatter(val[mask], aro[mask], c=QCOL[q], s=90,
                   edgecolors='white', linewidths=0.8,
                   label=f'{QLAB[q]} (n={mask.sum()})', zorder=3)
    ax.axvline(5, color='gray', ls='--', lw=1.2, alpha=0.7)
    ax.axhline(5, color='gray', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('Valencía (SAM 1–9)', fontsize=10)
    ax.set_ylabel('Arousal (SAM 1–9)',  fontsize=10)
    ax.set_xlim(0.5, 9.5); ax.set_ylim(0.5, 9.5)
    ax.set_xticks(range(1,10)); ax.set_yticks(range(1,10))
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Dispersión Valencia vs. Arousal\n(40 ensayos, sujeto s02)',
                 fontsize=10, fontweight='bold')
    for q, (tx, ty) in [('HVHA',(7.5,7.5)),('HVLA',(7.5,2.5)),
                         ('LVHA',(2.5,7.5)),('LVLA',(2.5,2.5))]:
        ax.text(tx, ty, q, ha='center', va='center', fontsize=11,
                fontweight='bold', color=QCOL[q], alpha=0.35)

    # — Pie de cuadrantes —
    ax2 = axes[1]
    q_counts = {q: (quadrant==q).sum() for q in ['HVHA','HVLA','LVHA','LVLA']}
    wedges, texts, autotexts = ax2.pie(
        q_counts.values(),
        labels=[f'{QLAB[q]}\nn={n}' for q,n in q_counts.items()],
        colors=[QCOL[q] for q in q_counts],
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops=dict(fontsize=8.5))
    for at in autotexts: at.set_fontsize(8)
    ax2.set_title('Distribución por Cuadrante\n(umbral SAM = 5)',
                  fontsize=10, fontweight='bold')

    fig.suptitle('1. Dispersión Valencia vs. Arousal — Dataset DEAP, Sujeto s02\n'
                 'Líneas punteadas = umbral de binarización (SAM = 5)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'deap_scatter_valence_arousal.png'))


def fig2_raw_signal(eeg14, val, aro, quadrant, out_dir):
    """Figura 2 — Series de tiempo T7 y T8, HVHA vs LVHA (10 s)."""
    t7_i = CH_EPOCX.index('T7')
    t8_i = CH_EPOCX.index('T8')
    hvha_tr = np.where(quadrant == 'HVHA')[0][0]
    lvha_tr = np.where(quadrant == 'LVHA')[0][0]
    t10 = np.linspace(0, 10, 10*FS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey='row')
    for row, (ch_i, ch_name) in enumerate([(t7_i,'T7'), (t8_i,'T8')]):
        for col, (tr_idx, q_label, color) in enumerate([
                (hvha_tr,
                 f'HVHA — Feliz (V={val[hvha_tr]:.1f}, A={aro[hvha_tr]:.1f})',
                 '#2ECC71'),
                (lvha_tr,
                 f'LVHA — Estresado (V={val[lvha_tr]:.1f}, A={aro[lvha_tr]:.1f})',
                 '#E74C3C')]):
            seg = eeg14[tr_idx, ch_i, :10*FS]
            axes[row,col].plot(t10, seg, color=color, lw=0.85)
            axes[row,col].set_title(f'{ch_name} — {q_label}',
                                    fontsize=9, fontweight='bold', color=color)
            axes[row,col].set_xlabel('Tiempo (s)', fontsize=9)
            axes[row,col].set_xlim(0, 10)
            if col == 0:
                axes[row,col].set_ylabel('Amplitud (µV)', fontsize=9)

    fig.suptitle('2. Serie de Tiempo EEG (10 s) — Canales T7 y T8\n'
                 'Dataset DEAP, Sujeto s02 | HVHA (Feliz) vs. LVHA (Estresado)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'deap_raw_t7_t8.png'))


def fig3_psd_alpha_beta(eeg14, quadrant, out_dir):
    """Figura 3 — PSD media Alpha y Beta por cuadrante."""
    def mean_psd_q(q):
        tr_idx = np.where(quadrant == q)[0]
        psds   = []
        for ti in tr_idx:
            for ci in range(14):
                fw, Pxx = welch(eeg14[ti, ci, :], fs=FS, nperseg=256)
                psds.append(Pxx)
        return fw, np.mean(psds, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (q1, q2, title) in zip(axes, [
            ('HVHA','LVHA','Alta Activación (HA): Feliz vs. Estresado'),
            ('HVLA','LVLA','Baja Activación  (LA): Relajado vs. Triste')]):
        fw1, p1 = mean_psd_q(q1)
        fw2, p2 = mean_psd_q(q2)
        mask    = (fw1 >= 1) & (fw1 <= 40)
        ax.semilogy(fw1[mask], p1[mask], color=QCOL[q1], lw=2,
                    label=f'{QLAB[q1]} (n={(quadrant==q1).sum()})', alpha=0.9)
        ax.semilogy(fw2[mask], p2[mask], color=QCOL[q2], lw=2,
                    label=f'{QLAB[q2]} (n={(quadrant==q2).sum()})', alpha=0.9)
        for band, (lo, hi) in [('Alpha',(8,13)), ('Beta',(13,30))]:
            ax.axvspan(lo, hi, color=BAND_COLS[band], alpha=0.25, label=band)
        ax.set_xlabel('Frecuencia (Hz)', fontsize=10)
        ax.set_ylabel('PSD media (µV²/Hz)', fontsize=10)
        ax.set_xlim(1, 40)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.legend(ncol=2, fontsize=7.5, loc='upper right')

    fig.suptitle('3. PSD Welch — Bandas Alpha (8–13 Hz) y Beta (13–30 Hz)\n'
                 '14 Canales EPOC X — Dataset DEAP, Sujeto s02',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'deap_psd_alpha_beta.png'))


def fig4_extraction_heatmap(eeg14, out_dir):
    """Figura 4 — Diagrama extracción 14ch EPOC X + heatmap Pearson 14×14."""
    corr_all = np.mean([np.corrcoef(eeg14[ti]) for ti in range(40)], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    epocx_set = set(CH_EPOCX)

    # — Diagrama de cabeza —
    ax = axes[0]
    ax.set_aspect('equal')
    head = Circle((0,0),1.0,fill=False,edgecolor='#2C3E50',linewidth=2.5,zorder=3)
    ax.add_patch(head)
    ax.plot([-0.07,0,0.07],[0.98,1.12,0.98],'k-',lw=2.5,zorder=4)
    for s in [-1,1]:
        ax.plot([s*1.0,s*1.08,s*1.08,s*1.0],
                [0.12,0.08,-0.08,-0.12],'k-',lw=2.5,zorder=4)
    for ch, (x,y) in POS_32.items():
        if ch in epocx_set:
            ax.scatter(x,y,s=200,color='#E74C3C',edgecolors='white',lw=1.5,zorder=6)
            ax.text(x,y+0.09,ch,ha='center',fontsize=7,fontweight='bold',
                    color='#922B21',zorder=7,
                    bbox=dict(boxstyle='round,pad=0.1',fc='white',alpha=0.75,lw=0))
        else:
            ax.scatter(x,y,s=80,color='#D5DBDB',edgecolors='#95A5A6',lw=0.8,zorder=5)
            ax.text(x,y+0.08,ch,ha='center',fontsize=6,color='#7F8C8D',zorder=6)
    legend_el = [
        mlines.Line2D([0],[0],marker='o',color='w',markerfacecolor='#E74C3C',
                      markersize=10,label='EPOC X (14 ch seleccionados)'),
        mlines.Line2D([0],[0],marker='o',color='w',markerfacecolor='#D5DBDB',
                      markeredgecolor='#95A5A6',markersize=8,
                      label='DEAP restantes (18 ch)'),
    ]
    ax.legend(handles=legend_el,loc='lower center',fontsize=8,
              bbox_to_anchor=(0.5,-0.10))
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.axis('off')
    ax.set_title('Extracción 14 Ch EPOC X\nde los 32 canales EEG de DEAP',
                 fontsize=9.5, fontweight='bold')

    # — Heatmap —
    sns.heatmap(corr_all, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=CH_EPOCX, yticklabels=CH_EPOCX,
                vmin=-1, vmax=1, linewidths=0.4, ax=axes[1],
                annot_kws={'size':6.5}, cbar_kws={'shrink':0.8})
    axes[1].set_title('Correlación Pearson 14×14\n(media 40 ensayos, sujeto s02)',
                      fontsize=9.5, fontweight='bold')

    fig.suptitle('4. Extracción de 14 Canales EPOC X de los 32 de DEAP\n'
                 'Dataset DEAP, Sujeto s02',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    save(fig, os.path.join(out_dir, 'deap_14ch_epocx.png'))


def fig5_topomap(eeg14, quadrant, out_dir):
    """Figura 5 — Topomapa 2D Alpha comparando HVHA vs LVHA."""
    hvha_idx = np.where(quadrant == 'HVHA')[0]
    lvha_idx = np.where(quadrant == 'LVHA')[0]
    alpha_hvha = alpha_power_per_channel(eeg14, hvha_idx)
    alpha_lvha = alpha_power_per_channel(eeg14, lvha_idx)

    vmin = min(alpha_hvha.min(), alpha_lvha.min())
    vmax = max(alpha_hvha.max(), alpha_lvha.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_t = cm.RdYlBu_r

    def topo_panel(ax, alpha_vals, title):
        xs = np.array([POS_2D[ch][0] for ch in CH_EPOCX])
        ys = np.array([POS_2D[ch][1] for ch in CH_EPOCX])
        xi = np.linspace(-1.1, 1.1, 300)
        yi = np.linspace(-1.1, 1.1, 300)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((xs, ys), alpha_vals, (Xi, Yi), method='cubic')
        Zi[Xi**2 + Yi**2 > 1.0] = np.nan
        im = ax.contourf(Xi, Yi, Zi, levels=64, cmap=cmap_t, norm=norm, zorder=1)
        head = Circle((0,0),1.0,fill=False,edgecolor='k',linewidth=2.5,zorder=4)
        ax.add_patch(head)
        ax.plot([-0.07,0,0.07],[0.98,1.12,0.98],'k-',lw=2.5,zorder=5)
        for s in [-1,1]:
            ax.plot([s*1.0,s*1.08,s*1.08,s*1.0],
                    [0.12,0.08,-0.08,-0.12],'k-',lw=2.5,zorder=5)
        ax.scatter(xs, ys, c=alpha_vals, cmap=cmap_t, norm=norm,
                   s=110, edgecolors='k', linewidths=1.1, zorder=6)
        for ch in CH_EPOCX:
            ax.annotate(ch, xy=POS_2D[ch],
                        xytext=(POS_2D[ch][0]+0.01, POS_2D[ch][1]+0.08),
                        fontsize=7, ha='center', zorder=7,
                        bbox=dict(boxstyle='round,pad=0.1',fc='white',alpha=0.7,lw=0))
        ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')
        return im

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes: ax.set_aspect('equal')
    topo_panel(axes[0], alpha_hvha,
               f'HVHA — Feliz/Excitado (n={len(hvha_idx)} trials)')
    im = topo_panel(axes[1], alpha_lvha,
               f'LVHA — Estresado/Ansioso (n={len(lvha_idx)} trials)')
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
    plt.colorbar(im, cax=cbar_ax, label='Potencia Alpha media (µV²/Hz)')

    fig.suptitle('5. Topomapa 2D — Potencia Banda Alpha (8–13 Hz)\n'
                 '14 Canales EPOC X | Dataset DEAP, Sujeto s02 | HVHA vs. LVHA',
                 fontsize=11, fontweight='bold')
    fig.savefig(os.path.join(out_dir,'deap_topomap_alpha.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Guardada: {os.path.join(out_dir,'deap_topomap_alpha.png')}")


# ──────────────────────────────────────────────────────────────
# ESTADÍSTICAS DESCRIPTIVAS
# ──────────────────────────────────────────────────────────────

def descriptive_stats(eeg14, quadrant):
    """Estadísticas PSD Alpha por cuadrante."""
    print("\n── Estadísticas PSD Alpha (8-13 Hz) por cuadrante — Sujeto s02 ──")
    print(f"  {'Cuadrante':<8} {'n':>5} {'Media':>8} {'Med.':>8} {'SD':>8} "
          f"{'IC_lo':>8} {'IC_hi':>8} {'SW p':>8} {'Normal':>7}")
    print("  " + "-"*74)
    for q in ['HVHA','HVLA','LVHA','LVLA']:
        tr_idx = np.where(quadrant == q)[0]
        vals   = []
        for ti in tr_idx:
            for ci in range(14):
                fw, Pxx = welch(eeg14[ti, ci, :], fs=FS, nperseg=256)
                vals.append(Pxx[(fw >= 8) & (fw <= 13)].mean())
        v  = np.array(vals)
        n  = len(v); m = v.mean(); sd = v.std()
        ic = 1.96 * sd / np.sqrt(n)
        W, p = shapiro(v[:min(n, 5000)])
        normal = 'Sí' if p > 0.05 else 'No'
        print(f"  {q:<8} {n:>5} {m:>8.3f} {np.median(v):>8.3f} {sd:>8.3f} "
              f"{m-ic:>8.3f} {m+ic:>8.3f} {p:>8.4f} {normal:>7}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='EDA — Dataset DEAP (5 figuras + estadísticas)')
    parser.add_argument('--dat_file', type=str, default='s02.dat',
                        help='Ruta al archivo .dat de DEAP (e.g. s02.dat)')
    parser.add_argument('--out_dir',  type=str, default='fig',
                        help='Carpeta de salida para las figuras')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nCargando: {args.dat_file}")

    eeg14, val, aro, quadrant = load_deap(args.dat_file)
    print(f"  → eeg14 shape: {eeg14.shape}  "
          f"({eeg14.shape[0]} trials × {eeg14.shape[1]} ch × {eeg14.shape[2]} samples)")
    print(f"  → Cuadrantes: { {q:(quadrant==q).sum() for q in ['HVHA','HVLA','LVHA','LVLA']} }")

    print("\nGenerando figuras EDA:")
    fig1_scatter_valence_arousal(val, aro, quadrant, args.out_dir)
    fig2_raw_signal(eeg14, val, aro, quadrant, args.out_dir)
    fig3_psd_alpha_beta(eeg14, quadrant, args.out_dir)
    fig4_extraction_heatmap(eeg14, args.out_dir)
    fig5_topomap(eeg14, quadrant, args.out_dir)

    descriptive_stats(eeg14, quadrant)
    print(f"\n✅ EDA completado. Figuras guardadas en: {args.out_dir}")


if __name__ == '__main__':
    main()
