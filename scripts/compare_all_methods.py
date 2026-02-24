"""
Script per confrontare tutti i metodi di stabilizzazione:
- Block Matching
- Optical Flow (Shi-Tomasi + LK)
- ORB + RANSAC (Partial)
- ORB + RANSAC (Affine)
- ORB + RANSAC (Homography)

Usage:
    python scripts/compare_all_methods.py --input data/input/video_instabile.mp4
"""

import yaml
import json
import sys
import argparse
from pathlib import Path
import time

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_stabilizer import VideoStabilizer


def test_method(config_path: str, input_video: str, output_video: str, method_name: str):
    """Testa un metodo di stabilizzazione."""
    print(f"\n{'='*70}")
    print(f"🎬 Testing: {method_name}")
    print(f"{'='*70}\n")
    
    # Carica configurazione
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Mostra configurazione rilevante
    gm = config.get('global_motion', {})
    of = gm.get('optical_flow', {})
    
    print(f"📋 Configurazione:")
    print(f"   Estimation method: {gm.get('estimation_method', 'block_matching')}")
    
    if gm.get('estimation_method') == 'optical_flow':
        print(f"   Feature type: {of.get('feature_type', 'shi_tomasi')}")
        print(f"   Transform type: {of.get('transform_type', 'partial')}")
        print(f"   RANSAC threshold: {of.get('ransac_reproj_threshold', 3.0)}")
        
        if of.get('feature_type') == 'orb':
            print(f"   ORB features: {of.get('orb_n_features', 500)}")
            print(f"   ORB ratio threshold: {of.get('orb_ratio_threshold', 0.75)}")
    
    print()
    
    # Crea stabilizzatore
    stabilizer = VideoStabilizer(config)
    
    # Esegui stabilizzazione
    start_time = time.time()
    success = stabilizer.stabilize_video(input_video, output_video, show_progress=True)
    elapsed_time = time.time() - start_time
    
    if success:
        metrics = stabilizer.get_metrics()
        
        print(f"\n✅ Stabilizzazione completata in {elapsed_time:.2f}s")
        print(f"\n📊 Metriche {method_name}:")
        print(f"   Frames processati: {metrics['num_frames']}")
        print(f"   Tempo totale: {metrics['total_processing_time']:.2f}s")
        print(f"   Tempo/frame: {metrics['avg_frame_time']*1000:.2f}ms")
        print(f"   RMS displacement: X={metrics['rms_dx']:.2f}px, Y={metrics['rms_dy']:.2f}px")
        print(f"   RMS rotation: {metrics['rms_angle']:.3f}°")
        print(f"   Jitter reduction: X={metrics['jitter_reduction_x']:.1f}%, Y={metrics['jitter_reduction_y']:.1f}%, Rot={metrics['jitter_reduction_angle']:.1f}%")
        print(f"   Max offset: X={metrics['max_offset_x']:.2f}px, Y={metrics['max_offset_y']:.2f}px")
        
        # Salva metriche in JSON
        metrics_file = output_video.replace('.mp4', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Metriche salvate: {metrics_file}")
        
        return metrics
    else:
        print(f"\n❌ Errore durante la stabilizzazione di {method_name}")
        return None


def compare_metrics(all_metrics: dict):
    """Confronta le metriche di tutti i metodi."""
    print(f"\n{'='*70}")
    print("📈 CONFRONTO FINALE")
    print(f"{'='*70}\n")
    
    # Tabella comparativa
    header = f"{'Metodo':<30} {'RMS X':<10} {'RMS Y':<10} {'Jitter X%':<12} {'Jitter Y%':<12} {'FPS':<10}"
    print(header)
    print("-" * len(header))
    
    for method_name, metrics in all_metrics.items():
        if metrics:
            fps = metrics['num_frames'] / metrics['total_processing_time']
            print(
                f"{method_name:<30} "
                f"{metrics['rms_dx']:<10.2f} "
                f"{metrics['rms_dy']:<10.2f} "
                f"{metrics['jitter_reduction_x']:<12.1f} "
                f"{metrics['jitter_reduction_y']:<12.1f} "
                f"{fps:<10.2f}"
            )
    
    print()
    
    # Trova il migliore per ogni metrica
    print("🏆 Migliori per categoria:")
    
    best_jitter_x = max(all_metrics.items(), key=lambda x: x[1]['jitter_reduction_x'] if x[1] else -1)
    best_jitter_y = max(all_metrics.items(), key=lambda x: x[1]['jitter_reduction_y'] if x[1] else -1)
    best_speed = max(all_metrics.items(), key=lambda x: x[1]['num_frames']/x[1]['total_processing_time'] if x[1] else -1)
    
    print(f"   Jitter Reduction X: {best_jitter_x[0]} ({best_jitter_x[1]['jitter_reduction_x']:.1f}%)")
    print(f"   Jitter Reduction Y: {best_jitter_y[0]} ({best_jitter_y[1]['jitter_reduction_y']:.1f}%)")
    print(f"   Velocità: {best_speed[0]} ({best_speed[1]['num_frames']/best_speed[1]['total_processing_time']:.2f} fps)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Confronta tutti i metodi di stabilizzazione')
    parser.add_argument('--input', type=str, default='data/input/video_instabile.mp4',
                       help='Path al video di input')
    parser.add_argument('--output-dir', type=str, default='data/output',
                       help='Directory per i video di output')
    args = parser.parse_args()
    
    input_video = args.input
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verifica che il video di input esista
    if not Path(input_video).exists():
        print(f"❌ Errore: File di input non trovato: {input_video}")
        return
    
    print(f"\n{'='*70}")
    print("🎬 CONFRONTO METODI DI STABILIZZAZIONE VIDEO")
    print(f"{'='*70}")
    print(f"Input: {input_video}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Definisci i metodi da testare
    methods = [
        #("config/config_block_matching.yaml", "stabilized_block_matching.mp4", "Block Matching"),
        ("config/config_optical_flow.yaml", "stabilized_optical_flow.mp4", "Optical Flow (Shi-Tomasi+LK)"),
        ("config/config_orb.yaml", "stabilized_orb_partial.mp4", "ORB + RANSAC Partial (4 param)"),
        ("config/config_orb_affine.yaml", "stabilized_orb_affine.mp4", "ORB + RANSAC Affine (6 param)"),
        ("config/config_orb_homography.yaml", "stabilized_orb_homography.mp4", "ORB + RANSAC Homography (8 param)"),
    ]
    
    all_metrics = {}
    
    # Esegui tutti i metodi
    for config_path, output_filename, method_name in methods:
        config_full_path = Path(config_path)
        
        if not config_full_path.exists():
            print(f"⚠️  Warning: Config non trovato: {config_path}, skip...")
            continue
        
        output_path = output_dir / output_filename
        
        try:
            metrics = test_method(str(config_full_path), input_video, str(output_path), method_name)
            all_metrics[method_name] = metrics
        except Exception as e:
            print(f"\n❌ Errore durante il test di {method_name}: {e}")
            all_metrics[method_name] = None
    
    # Confronto finale
    compare_metrics(all_metrics)
    
    print(f"\n{'='*70}")
    print("✅ CONFRONTO COMPLETATO")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
