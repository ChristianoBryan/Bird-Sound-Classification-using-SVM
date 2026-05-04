import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve, roc_auc_score,
    classification_report
)
import seaborn as sns
import librosa
import warnings
warnings.filterwarnings('ignore')

# KONFIGURASI
DATA_DIR         = 'data_burung'
SEGMENT_DURATION = 5.0
SR_TARGET        = 22050
N_MFCC           = 13
N_CLASSES        = 5
N_FOLDS          = 5
OUTPUT_DIR       = 'output_plots'

COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
LABELS = [f'Kelas {i}' for i in range(N_CLASSES)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

#1. AUDIT DATA
print("=" * 60)
print("LANGKAH 1: AUDIT DATA")
print("=" * 60)

all_files  = []
all_labels = []

for cls in range(N_CLASSES):
    pattern = os.path.join(DATA_DIR, f'class_{cls}', '**', '*.ogg')
    files   = glob.glob(pattern, recursive=True)
    all_files.extend(files)
    all_labels.extend([cls] * len(files))
    print(f"  class_{cls}: {len(files)} file .ogg ditemukan")

print(f"\n  TOTAL: {len(all_files)} file")
assert len(all_files) > 0, "TIDAK ADA FILE DITEMUKAN! Cek struktur folder DATA_DIR."

# LANGKAH 2 & 3 — EKSTRAKSI FITUR + NORMALISASI
print("\n" + "=" * 60)
print("LANGKAH 2 & 3: EKSTRAKSI FITUR (MFCC + SC + ZCR)")
print("=" * 60)
print("  Fitur 1: MFCC (13 koefisien) — timbre/warna suara")
print("  Fitur 2: Spectral Centroid   — frekuensi dominan")
print("  Fitur 3: Zero Crossing Rate  — kecepatan getaran")
print()

X_raw    = []
y_raw    = []
skipped  = 0

for fpath, cls in zip(all_files, all_labels):
    try:
        # Load dan resample audio
        y_audio, sr = librosa.load(fpath, sr=SR_TARGET, mono=True)

        # Segmentasi: ambil 5 detik dari tengah file
        seg_samples   = int(SEGMENT_DURATION * SR_TARGET)
        total_samples = len(y_audio)

        if total_samples >= seg_samples:
            start   = (total_samples - seg_samples) // 2
            y_seg   = y_audio[start:start + seg_samples]
        else:
            # Pad dengan nol jika file terlalu pendek
            y_seg = np.pad(y_audio, (0, seg_samples - total_samples))

        # MFCC
        mfcc      = librosa.feature.mfcc(y=y_seg, sr=SR_TARGET, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)          # shape (13,)

        # Spectral Centroid
        sc        = librosa.feature.spectral_centroid(y=y_seg, sr=SR_TARGET)
        sc_mean   = float(np.mean(sc))             # scalar

        # Zero Crossing Rate
        zcr       = librosa.feature.zero_crossing_rate(y_seg)
        zcr_mean  = float(np.mean(zcr))            # scalar

        # Gabungkan
        feature_vec = np.concatenate([mfcc_mean, [sc_mean, zcr_mean]])
        X_raw.append(feature_vec)
        y_raw.append(cls)

        print(f"  [OK] class_{cls} | {os.path.basename(fpath):<30} "
              f"SC={sc_mean:>7.1f}Hz  ZCR={zcr_mean:.4f}")

    except Exception as e:
        print(f"  [SKIP] {fpath}: {e}")
        skipped += 1

X = np.array(X_raw)
y = np.array(y_raw)
print(f"\n  Ekstraksi selesai: {len(X)} sampel berhasil, {skipped} dilewati")
print(f"  Shape X: {X.shape}  (sampel × dimensi fitur)")

# Normalisasi
print("\n  Normalisasi StandardScaler (z-score)...")
print(f"  Sebelum — SC range: {X[:,-2].min():.1f} – {X[:,-2].max():.1f} Hz")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Sesudah — SC mean: {X_scaled[:,-2].mean():.4f}, std: {X_scaled[:,-2].std():.4f}")

print("\n  Transformasi PCA (15D -> 3D)...")
pca   = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
evr   = pca.explained_variance_ratio_
print(f"  PC1={evr[0]*100:.1f}%  PC2={evr[1]*100:.1f}%  PC3={evr[2]*100:.1f}%"
      f"  Total={evr.sum()*100:.1f}%")

# 5. TRAINING MODEL SVM RBF
print("\n" + "=" * 60)
print("LANGKAH 5: TRAINING MODEL — SVM RBF (5-Fold Stratified CV)")
print("=" * 60)

cv      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
model   = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)

print(f"  Model   : SVM — Kernel RBF")
print(f"  C       : 10  (regularisasi)")
print(f"  Gamma   : scale")
print(f"  CV      : {N_FOLDS}-Fold Stratified")
print(f"\n  Menjalankan cross-validation...")

y_pred  = cross_val_predict(model, X_scaled, y, cv=cv)
y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')
y_bin   = label_binarize(y, classes=range(N_CLASSES))

print("  Cross-validation selesai!")

# 7. EVALUASI
print("\n" + "=" * 60)
print("LANGKAH 7: EVALUASI PERFORMA")
print("=" * 60)

acc  = accuracy_score(y, y_pred)
f1s  = f1_score(y, y_pred, average=None, labels=range(N_CLASSES), zero_division=0)
prec = precision_score(y, y_pred, average=None, labels=range(N_CLASSES), zero_division=0)
rec  = recall_score(y, y_pred, average=None, labels=range(N_CLASSES), zero_division=0)
aucs = [roc_auc_score(y_bin[:,c], y_proba[:,c]) for c in range(N_CLASSES)]

print(f"\n  {'Kelas':<10} {'N':>4} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("  " + "-" * 52)
for cls in range(N_CLASSES):
    n = int(np.sum(y == cls))
    print(f"  Kelas {cls:<4} {n:>4} {prec[cls]:>10.3f} {rec[cls]:>8.3f} "
          f"{f1s[cls]:>8.3f} {aucs[cls]:>8.3f}")
print("  " + "-" * 52)
print(f"  {'Macro':<10} {len(y):>4} {prec.mean():>10.3f} {rec.mean():>8.3f} "
      f"{f1s.mean():>8.3f} {np.mean(aucs):>8.3f}")
print(f"\n  Overall Accuracy : {acc:.4f} ({acc:.1%})")
print(f"  Macro F1-Score   : {f1s.mean():.4f}")
print(f"  Macro ROC-AUC    : {np.mean(aucs):.4f}")

print("\n" + "=" * 60)
print("MENYIMPAN GRAFIK...")
print("=" * 60)

# 1. 3D PCA
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
for cls in range(N_CLASSES):
    mask = y == cls
    ax.scatter(X_pca[mask,0], X_pca[mask,1], X_pca[mask,2],
               c=COLORS[cls], label=LABELS[cls], s=60,
               alpha=0.9, edgecolors='gray', linewidths=0.4)
ax.set_title('3D Feature Space — Proyeksi PCA', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', fontsize=10)
ax.set_zlabel(f'PC3 ({evr[2]*100:.1f}%)', fontsize=10)
ax.legend(fontsize=10)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'plot1_3d_pca.png')
plt.savefig(out1, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  -> {out1}")

# 2. Scatter PC1 vs PC2
fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
for cls in range(N_CLASSES):
    mask = y == cls
    ax.scatter(X_pca[mask,0], X_pca[mask,1],
               c=COLORS[cls], label=f'{LABELS[cls]} (n={mask.sum()})',
               s=80, alpha=0.9, edgecolors='gray', linewidths=0.5)
for cls in range(N_CLASSES):
    mask = y == cls
    cx, cy = X_pca[mask,0].mean(), X_pca[mask,1].mean()
    ax.annotate(f'K{cls}', (cx, cy), fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[cls], alpha=0.6))
ax.set_title('Proyeksi PC1 vs PC2 — Feature Space', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 — {evr[0]*100:.1f}% Variance', fontsize=11)
ax.set_ylabel(f'PC2 — {evr[1]*100:.1f}% Variance', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, color='#ddd', linewidth=0.5)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'plot2_scatter_pc1_pc2.png')
plt.savefig(out2, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  -> {out2}")

# 3. Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=LABELS, yticklabels=LABELS,
            linewidths=1.0, linecolor='#cccccc',
            annot_kws={'color':'#111111','fontsize':14,'fontweight':'bold'})
ax.set_title('Confusion Matrix — SVM RBF (5-Fold CV)', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Label', fontsize=12)
fig.text(0.5, 0.01, f'Overall Accuracy: {acc:.1%}', ha='center', fontsize=10, color='#333')
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, 'plot3_confusion_matrix.png')
plt.savefig(out3, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  -> {out3}")

# 4. ROC Curve
fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
for cls in range(N_CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:,cls], y_proba[:,cls])
    ax.plot(fpr, tpr, color=COLORS[cls], linewidth=2.5,
            label=f'{LABELS[cls]}  AUC={aucs[cls]:.2f}')
ax.plot([0,1],[0,1],'--', color='gray', linewidth=1.5, label='Random (AUC=0.50)')
ax.set_title('ROC Curve — One-vs-Rest per Kelas', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, color='#ddd', linewidth=0.5)
fig.text(0.5, 0.01, f'Macro AUC = {np.mean(aucs):.3f}', ha='center', fontsize=10, color='#333')
plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, 'plot4_roc_curve.png')
plt.savefig(out4, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  -> {out4}")

# 5. F1-Score
fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
bars = ax.bar(LABELS, f1s, color=COLORS, edgecolor='gray', linewidth=0.8, width=0.55)
ax.axhline(f1s.mean(), color='#e67e22', linestyle='--', linewidth=2.0,
           label=f'Macro F1 = {f1s.mean():.3f}')
ax.axhline(0.5, color='#aaa', linestyle=':', linewidth=1.2, label='Baseline = 0.50')
for bar, f in zip(bars, f1s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f'{f:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color='#111')
ax.set_ylim(0, 1.15)
ax.set_title('F1-Score per Kelas — SVM RBF (5-Fold CV)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, axis='y', color='#ddd', linewidth=0.5)
plt.tight_layout()
out5 = os.path.join(OUTPUT_DIR, 'plot5_f1_score.png')
plt.savefig(out5, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  -> {out5}")

print("\n" + "=" * 60)
print("PIPELINE SELESAI!")
print(f"Semua grafik tersimpan di folder: ./{OUTPUT_DIR}/")
print("=" * 60)