"""Funzioni di plotting per traiettorie e metriche"""
import numpy as np
import matplotlib.pyplot as plt


def create_trajectory_plot(metrics):
    """Crea grafico della traiettoria raw vs smoothed"""
    if not metrics or 'raw_trajectory' not in metrics:
        return None

    raw = np.array(metrics['raw_trajectory'])
    smooth = np.array(metrics['smoothed_trajectory'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(raw[:, 0], label='Raw X', alpha=0.7, linewidth=1)
    axes[0].plot(smooth[:, 0], label='Smoothed X', linewidth=2)
    axes[0].set_ylabel('X Displacement (px)')
    axes[0].set_title('Traiettoria X')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(raw[:, 1], label='Raw Y', alpha=0.7, linewidth=1)
    axes[1].plot(smooth[:, 1], label='Smoothed Y', linewidth=2)
    axes[1].set_ylabel('Y Displacement (px)')
    axes[1].set_xlabel('Frame')
    axes[1].set_title('Traiettoria Y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_metrics_comparison(metrics):
    """Crea grafico a barre delle metriche principali"""
    if not metrics:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    metric_names = ['RMS X', 'RMS Y', 'RMS Angle', 'Jitter X', 'Jitter Y', 'Jitter Angle']
    values = [
        metrics.get('rms_dx', 0),
        metrics.get('rms_dy', 0),
        metrics.get('rms_angle', 0) * 10,
        metrics.get('jitter_reduction_x', 0),
        metrics.get('jitter_reduction_y', 0),
        metrics.get('jitter_reduction_angle', 0),
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(metric_names, values, color=colors, alpha=0.7)

    ax.set_ylabel('Valore')
    ax.set_title('Metriche di Stabilizzazione')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{value:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def create_rms_comparison_chart(successful_results):
    """Crea grafici a barre RMS per confronto tra metodi"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = [m.replace('_', ' ').title() for m in successful_results]
    rms_x = [successful_results[m]['metrics'].get('rms_dx', 0) for m in successful_results]
    rms_y = [successful_results[m]['metrics'].get('rms_dy', 0) for m in successful_results]
    rms_angle = [successful_results[m]['metrics'].get('rms_angle', 0) for m in successful_results]

    for ax, values, label, title, color in [
        (axes[0], rms_x,    'RMS X (px)',    'RMS Displacement X', '#1f77b4'),
        (axes[1], rms_y,    'RMS Y (px)',    'RMS Displacement Y', '#ff7f0e'),
        (axes[2], rms_angle,'RMS Angle (°)', 'RMS Rotation',       '#2ca02c'),
    ]:
        bars = ax.bar(methods, values, color=color, alpha=0.7)
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        fmt = '.4f' if 'Angle' in label else '.2f'
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_jitter_comparison_chart(successful_results):
    """Crea grafico a barre per jitter reduction tra metodi"""
    methods = [m.replace('_', ' ').title() for m in successful_results]
    jitter_x = [successful_results[m]['metrics'].get('jitter_reduction_x', 0) for m in successful_results]
    jitter_y = [successful_results[m]['metrics'].get('jitter_reduction_y', 0) for m in successful_results]
    jitter_angle = [successful_results[m]['metrics'].get('jitter_reduction_angle', 0) for m in successful_results]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x = np.arange(len(methods))
    width = 0.25

    groups = [
        (x - width, jitter_x,    'Jitter Red. X',     '#1f77b4'),
        (x,         jitter_y,    'Jitter Red. Y',     '#ff7f0e'),
        (x + width, jitter_angle,'Jitter Red. Angle', '#2ca02c'),
    ]

    for positions, values, label, color in groups:
        bars = ax.bar(positions, values, width, label=label, color=color, alpha=0.7)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Riduzione (%)')
    ax.set_title('Jitter Reduction Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def create_displacement_histogram(metrics):
    """Istogrammi degli spostamenti frame-by-frame: raw vs stabilizzato"""
    if not metrics or 'raw_trajectory' not in metrics or 'smoothed_trajectory' not in metrics:
        return None

    raw    = np.array(metrics['raw_trajectory'])
    smooth = np.array(metrics['smoothed_trajectory'])

    # Spostamenti frame-by-frame (differenze prime)
    raw_dx    = np.diff(raw[:, 0])
    raw_dy    = np.diff(raw[:, 1])
    smooth_dx = np.diff(smooth[:, 0])
    smooth_dy = np.diff(smooth[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bins = 40

    for ax, raw_d, smooth_d, axis_label in [
        (axes[0], raw_dx, smooth_dx, 'X'),
        (axes[1], raw_dy, smooth_dy, 'Y'),
    ]:
        ax.hist(raw_d,    bins=bins, alpha=0.6, color='#d62728', label='Raw',          density=True)
        ax.hist(smooth_d, bins=bins, alpha=0.6, color='#1f77b4', label='Stabilizzato', density=True)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f'Spostamento {axis_label} (px)')
        ax.set_ylabel('Densità')
        ax.set_title(f'Distribuzione Spostamenti {axis_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
