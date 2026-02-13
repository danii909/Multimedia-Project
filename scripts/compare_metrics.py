"""
Script per confrontare le metriche di due run di stabilizzazione.
Genera grafici e tabelle comparative tra block matching e optical flow.

Usage:
    python compare_metrics.py data/output/video_stabilizzato_block_matching_metrics.json data/output/video_stabilizzato_optical_flow_metrics.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _to_numpy_trajectory(traj):
    arr = np.array(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Traiettoria non valida: atteso array (N,3), ottenuto {arr.shape}")
    return arr


def _is_incremental_raw(raw: np.ndarray, smooth: np.ndarray) -> bool:
    """Heuristica per riconoscere vecchi JSON dove raw_trajectory è moto incrementale (dx,dy,dangle)
    e smoothed_trajectory è traiettoria cumulativa."""
    raw_max = float(np.max(np.abs(raw))) if raw.size else 0.0
    smooth_max = float(np.max(np.abs(smooth))) if smooth.size else 0.0
    if smooth_max <= 0:
        return False
    return raw_max < 0.2 * smooth_max


def _normalize_metrics(metrics: dict) -> dict:
    """Normalizza raw/smoothed in traiettorie assolute e ricalcola le metriche aggregate."""
    raw = _to_numpy_trajectory(metrics.get('raw_trajectory', []))
    smooth = _to_numpy_trajectory(metrics.get('smoothed_trajectory', []))

    min_len = min(len(raw), len(smooth))
    raw = raw[:min_len]
    smooth = smooth[:min_len]

    converted = False
    if _is_incremental_raw(raw, smooth):
        raw = np.cumsum(raw, axis=0)
        converted = True

    if len(raw) >= 2:
        raw_step = np.diff(raw, axis=0)
        smooth_step = np.diff(smooth, axis=0)
        rms_dx = float(np.sqrt(np.mean(raw_step[:, 0] ** 2)))
        rms_dy = float(np.sqrt(np.mean(raw_step[:, 1] ** 2)))
        rms_angle = float(np.sqrt(np.mean(raw_step[:, 2] ** 2)))
    else:
        raw_step = np.zeros((0, 3), dtype=float)
        smooth_step = np.zeros((0, 3), dtype=float)
        rms_dx = 0.0
        rms_dy = 0.0
        rms_angle = 0.0

    def jitter_reduction(step_raw: np.ndarray, step_smooth: np.ndarray) -> np.ndarray:
        if len(step_raw) < 2 or len(step_smooth) < 2:
            return np.zeros(3, dtype=float)
        raw_var = np.var(step_raw, axis=0)
        smooth_var = np.var(step_smooth, axis=0)
        out = np.zeros(3, dtype=float)
        for i in range(3):
            if raw_var[i] > 0:
                out[i] = (1.0 - smooth_var[i] / raw_var[i]) * 100.0
        return out

    jr = jitter_reduction(raw_step, smooth_step)

    offsets = smooth - raw
    max_offset_x = float(np.max(np.abs(offsets[:, 0]))) if len(offsets) else 0.0
    max_offset_y = float(np.max(np.abs(offsets[:, 1]))) if len(offsets) else 0.0

    metrics['raw_trajectory'] = raw.tolist()
    metrics['smoothed_trajectory'] = smooth.tolist()
    metrics['rms_dx'] = rms_dx
    metrics['rms_dy'] = rms_dy
    metrics['rms_angle'] = rms_angle
    metrics['jitter_reduction_x'] = float(jr[0])
    metrics['jitter_reduction_y'] = float(jr[1])
    metrics['jitter_reduction_angle'] = float(jr[2])
    metrics['max_offset_x'] = max_offset_x
    metrics['max_offset_y'] = max_offset_y
    metrics['num_frames'] = int(min_len)

    if converted:
        metrics['_note'] = 'raw_trajectory convertita da incrementale a cumulativa per confronto corretto'

    return metrics


def load_metrics(filepath):
    """Carica metriche da file JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return _normalize_metrics(metrics)


def plot_trajectories(metrics1, metrics2, output_dir):
    """
    Plotta le traiettorie raw vs smoothed per entrambi i metodi.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Confronto Traiettorie: Raw vs Smoothed', fontsize=16, fontweight='bold')
    
    # Converti traiettorie in array numpy
    raw1 = np.array(metrics1['raw_trajectory'])
    smooth1 = np.array(metrics1['smoothed_trajectory'])
    raw2 = np.array(metrics2['raw_trajectory'])
    smooth2 = np.array(metrics2['smoothed_trajectory'])
    
    frames1 = np.arange(len(raw1))
    frames2 = np.arange(len(raw2))
    
    method1 = metrics1['method'].replace('_', ' ').title()
    method2 = metrics2['method'].replace('_', ' ').title()
    
    # Plot X displacement
    axes[0, 0].plot(frames1, raw1[:, 0], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[0, 0].plot(frames1, smooth1[:, 0], 'r-', linewidth=1.5, label='Smoothed')
    axes[0, 0].set_title(f'{method1} - Displacement X')
    axes[0, 0].set_ylabel('Pixels')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(frames2, raw2[:, 0], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[0, 1].plot(frames2, smooth2[:, 0], 'r-', linewidth=1.5, label='Smoothed')
    axes[0, 1].set_title(f'{method2} - Displacement X')
    axes[0, 1].set_ylabel('Pixels')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Y displacement
    axes[1, 0].plot(frames1, raw1[:, 1], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[1, 0].plot(frames1, smooth1[:, 1], 'r-', linewidth=1.5, label='Smoothed')
    axes[1, 0].set_title(f'{method1} - Displacement Y')
    axes[1, 0].set_ylabel('Pixels')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(frames2, raw2[:, 1], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[1, 1].plot(frames2, smooth2[:, 1], 'r-', linewidth=1.5, label='Smoothed')
    axes[1, 1].set_title(f'{method2} - Displacement Y')
    axes[1, 1].set_ylabel('Pixels')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot Rotation
    axes[2, 0].plot(frames1, raw1[:, 2], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[2, 0].plot(frames1, smooth1[:, 2], 'r-', linewidth=1.5, label='Smoothed')
    axes[2, 0].set_title(f'{method1} - Rotation')
    axes[2, 0].set_xlabel('Frame')
    axes[2, 0].set_ylabel('Degrees')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(frames2, raw2[:, 2], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[2, 1].plot(frames2, smooth2[:, 2], 'r-', linewidth=1.5, label='Smoothed')
    axes[2, 1].set_title(f'{method2} - Rotation')
    axes[2, 1].set_xlabel('Frame')
    axes[2, 1].set_ylabel('Degrees')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'trajectory_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Grafico traiettorie salvato: {output_path}")
    plt.close()


def plot_metrics_comparison(metrics1, metrics2, output_dir):
    """
    Crea grafici a barre per confrontare le metriche aggregate.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Confronto Metriche Aggregate', fontsize=16, fontweight='bold')
    
    method1 = metrics1['method'].replace('_', ' ').title()
    method2 = metrics2['method'].replace('_', ' ').title()
    methods = [method1, method2]
    
    # RMS Displacement
    rms_dx_values = [metrics1['rms_dx'], metrics2['rms_dx']]
    axes[0, 0].bar(methods, rms_dx_values, color=['#3498db', '#e74c3c'])
    axes[0, 0].set_title('RMS Displacement X')
    axes[0, 0].set_ylabel('Pixels')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rms_dx_values):
        axes[0, 0].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    rms_dy_values = [metrics1['rms_dy'], metrics2['rms_dy']]
    axes[0, 1].bar(methods, rms_dy_values, color=['#3498db', '#e74c3c'])
    axes[0, 1].set_title('RMS Displacement Y')
    axes[0, 1].set_ylabel('Pixels')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rms_dy_values):
        axes[0, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    rms_angle_values = [metrics1['rms_angle'], metrics2['rms_angle']]
    axes[0, 2].bar(methods, rms_angle_values, color=['#3498db', '#e74c3c'])
    axes[0, 2].set_title('RMS Rotation')
    axes[0, 2].set_ylabel('Degrees')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rms_angle_values):
        axes[0, 2].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Jitter Reduction
    jitter_x_values = [metrics1['jitter_reduction_x'], metrics2['jitter_reduction_x']]
    axes[1, 0].bar(methods, jitter_x_values, color=['#2ecc71', '#f39c12'])
    axes[1, 0].set_title('Jitter Reduction X')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(jitter_x_values):
        axes[1, 0].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    jitter_y_values = [metrics1['jitter_reduction_y'], metrics2['jitter_reduction_y']]
    axes[1, 1].bar(methods, jitter_y_values, color=['#2ecc71', '#f39c12'])
    axes[1, 1].set_title('Jitter Reduction Y')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(jitter_y_values):
        axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Processing Time
    time_values = [metrics1['total_processing_time'], metrics2['total_processing_time']]
    axes[1, 2].bar(methods, time_values, color=['#9b59b6', '#e67e22'])
    axes[1, 2].set_title('Total Processing Time')
    axes[1, 2].set_ylabel('Seconds')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(time_values):
        axes[1, 2].text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Grafico metriche salvato: {output_path}")
    plt.close()


def plot_displacement_magnitude(metrics1, metrics2, output_dir):
    """
    Plotta la magnitude del displacement frame per frame.
    """
    # raw_trajectory qui è una traiettoria assoluta; per-frame motion = diff
    raw1 = np.array(metrics1['raw_trajectory'], dtype=float)
    raw2 = np.array(metrics2['raw_trajectory'], dtype=float)

    step1 = np.diff(raw1, axis=0) if len(raw1) >= 2 else np.zeros((0, 3), dtype=float)
    step2 = np.diff(raw2, axis=0) if len(raw2) >= 2 else np.zeros((0, 3), dtype=float)
    
    # Calcola magnitude (distanza euclidea)
    mag1 = np.sqrt(step1[:, 0]**2 + step1[:, 1]**2)
    mag2 = np.sqrt(step2[:, 0]**2 + step2[:, 1]**2)
    
    frames1 = np.arange(len(mag1))
    frames2 = np.arange(len(mag2))
    
    method1 = metrics1['method'].replace('_', ' ').title()
    method2 = metrics2['method'].replace('_', ' ').title()
    
    plt.figure(figsize=(14, 6))
    plt.plot(frames1, mag1, label=method1, alpha=0.7, linewidth=1.5)
    plt.plot(frames2, mag2, label=method2, alpha=0.7, linewidth=1.5)
    plt.xlabel('Frame')
    plt.ylabel('Displacement Magnitude (pixels)')
    plt.title('Confronto Magnitude Displacement per Frame (da diff della traiettoria)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / 'displacement_magnitude.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Grafico magnitude salvato: {output_path}")
    plt.close()


def print_comparison_table(metrics1, metrics2):
    """
    Stampa una tabella di confronto delle metriche.
    """
    method1 = metrics1['method'].replace('_', ' ').title()
    method2 = metrics2['method'].replace('_', ' ').title()
    
    print("\n" + "=" * 80)
    print("TABELLA COMPARATIVA METRICHE")
    print("=" * 80)
    print(f"{'Metrica':<30} | {method1:>20} | {method2:>20}")
    print("-" * 80)
    print(f"{'Numero Frames':<30} | {metrics1['num_frames']:>20d} | {metrics2['num_frames']:>20d}")
    print(f"{'RMS Displacement X (px)':<30} | {metrics1['rms_dx']:>20.2f} | {metrics2['rms_dx']:>20.2f}")
    print(f"{'RMS Displacement Y (px)':<30} | {metrics1['rms_dy']:>20.2f} | {metrics2['rms_dy']:>20.2f}")
    print(f"{'RMS Rotation (deg)':<30} | {metrics1['rms_angle']:>20.3f} | {metrics2['rms_angle']:>20.3f}")
    print(f"{'Jitter Reduction X (%)':<30} | {metrics1['jitter_reduction_x']:>20.1f} | {metrics2['jitter_reduction_x']:>20.1f}")
    print(f"{'Jitter Reduction Y (%)':<30} | {metrics1['jitter_reduction_y']:>20.1f} | {metrics2['jitter_reduction_y']:>20.1f}")
    print(f"{'Jitter Reduction Angle (%)':<30} | {metrics1['jitter_reduction_angle']:>20.1f} | {metrics2['jitter_reduction_angle']:>20.1f}")
    print(f"{'Max Offset X (px)':<30} | {metrics1['max_offset_x']:>20.2f} | {metrics2['max_offset_x']:>20.2f}")
    print(f"{'Max Offset Y (px)':<30} | {metrics1['max_offset_y']:>20.2f} | {metrics2['max_offset_y']:>20.2f}")
    print(f"{'Avg Frame Time (s)':<30} | {metrics1['avg_frame_time']:>20.4f} | {metrics2['avg_frame_time']:>20.4f}")
    print(f"{'Total Processing Time (s)':<30} | {metrics1['total_processing_time']:>20.2f} | {metrics2['total_processing_time']:>20.2f}")
    print("=" * 80)
    
    # Calcola "winner" per alcune metriche chiave
    print("\n🏆 SOMMARIO CONFRONTO:")
    
    if metrics1['jitter_reduction_x'] > metrics2['jitter_reduction_x']:
        print(f"   ✓ {method1} riduce meglio il jitter X ({metrics1['jitter_reduction_x']:.1f}% vs {metrics2['jitter_reduction_x']:.1f}%)")
    else:
        print(f"   ✓ {method2} riduce meglio il jitter X ({metrics2['jitter_reduction_x']:.1f}% vs {metrics1['jitter_reduction_x']:.1f}%)")
    
    if metrics1['jitter_reduction_y'] > metrics2['jitter_reduction_y']:
        print(f"   ✓ {method1} riduce meglio il jitter Y ({metrics1['jitter_reduction_y']:.1f}% vs {metrics2['jitter_reduction_y']:.1f}%)")
    else:
        print(f"   ✓ {method2} riduce meglio il jitter Y ({metrics2['jitter_reduction_y']:.1f}% vs {metrics1['jitter_reduction_y']:.1f}%)")
    
    if metrics1['avg_frame_time'] < metrics2['avg_frame_time']:
        print(f"   ✓ {method1} è più veloce ({metrics1['avg_frame_time']*1000:.1f}ms vs {metrics2['avg_frame_time']*1000:.1f}ms per frame)")
    else:
        print(f"   ✓ {method2} è più veloce ({metrics2['avg_frame_time']*1000:.1f}ms vs {metrics1['avg_frame_time']*1000:.1f}ms per frame)")
    
    print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_metrics.py <metrics_file_1.json> <metrics_file_2.json>")
        print("\nEsempio:")
        print("  python compare_metrics.py data/output/video_stabilizzato_block_matching_metrics.json data/output/video_stabilizzato_optical_flow_metrics.json")
        sys.exit(1)
    
    file1 = Path(sys.argv[1])
    file2 = Path(sys.argv[2])
    
    # Verifica esistenza file
    if not file1.exists():
        print(f"❌ File non trovato: {file1}")
        sys.exit(1)
    if not file2.exists():
        print(f"❌ File non trovato: {file2}")
        sys.exit(1)
    
    print("=" * 80)
    print("CONFRONTO METRICHE DI STABILIZZAZIONE")
    print("=" * 80)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print()
    
    # Carica metriche
    print("📂 Caricamento metriche...")
    metrics1 = load_metrics(file1)
    metrics2 = load_metrics(file2)
    
    # Crea directory per output
    output_dir = Path('data/output/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stampa tabella di confronto
    print_comparison_table(metrics1, metrics2)
    
    # Genera grafici
    print("\n📊 Generazione grafici...")
    plot_trajectories(metrics1, metrics2, output_dir)
    plot_metrics_comparison(metrics1, metrics2, output_dir)
    plot_displacement_magnitude(metrics1, metrics2, output_dir)
    
    print("\n✅ Confronto completato!")
    print(f"📁 Grafici salvati in: {output_dir}")
    print()


if __name__ == '__main__':
    main()
