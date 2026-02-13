"""
Script principale per eseguire la stabilizzazione video
Esempio di utilizzo del sistema completo
"""

import sys
import yaml
import json
import argparse
from pathlib import Path
from src.video_stabilizer import VideoStabilizer


def main():
    """
    Funzione principale per eseguire la stabilizzazione video.
    
    Flusso:
    1. Carica la configurazione da config/config.yaml
    2. Verifica l'esistenza del video di input
    3. Crea il VideoStabilizer
    4. Esegue la stabilizzazione
    """
    
    # Parse argomenti linea di comando
    parser = argparse.ArgumentParser(description='Video Stabilization System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path al file di configurazione YAML (default: config/config.yaml)')
    parser.add_argument('--input', type=str, default='data/input/video_instabile.mp4',
                       help='Path al video di input (default: data/input/video_instabile.mp4)')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=(
            'Path al video di output. Se omesso, viene scelto automaticamente in base al metodo: '
            'data/output/video_stabilizzato_BM.mp4 (block_matching) oppure '
            'data/output/video_stabilizzato_OF.mp4 (optical_flow)'
        )
    )
    args = parser.parse_args()
    
    # Carica configurazione dal file YAML
    print("=" * 60)
    print("VIDEO STABILIZATION SYSTEM")
    print("Sistema di stabilizzazione video tramite stima e compensazione")
    print("del movimento globale - Progetto Multimedia")
    print("=" * 60)
    print()
    
    # Path del file di configurazione
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"❌ File di configurazione non trovato: {config_path}")
        print("Assicurati che il file config/config.yaml esista.")
        return
    
    # Carica configurazione
    print(f"📄 Caricamento configurazione da: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("✅ Configurazione caricata con successo")
    estimation_method = (config.get('global_motion', {}) or {}).get('estimation_method', 'block_matching')
    estimation_method = (estimation_method or 'block_matching').lower()
    print(f"   - Estimation Method: {estimation_method}")
    print(f"   - Block Size: {(config.get('motion_estimation', {}) or {}).get('block_size')}")
    print(f"   - Search Range: {(config.get('motion_estimation', {}) or {}).get('search_range')}")
    print(f"   - Smoothing Window: {(config.get('trajectory_smoothing', {}) or {}).get('smoothing_window')}")
    print(f"   - Filter Type: {(config.get('trajectory_smoothing', {}) or {}).get('filter_type')}")
    print(f"   - Border Mode: {(config.get('motion_compensation', {}) or {}).get('border_mode')}")
    print()
    
    # Percorsi dei video
    input_video = Path(args.input)
    if args.output:
        output_video = Path(args.output)
    else:
        suffix = 'BM' if estimation_method == 'block_matching' else 'OF'
        output_video = Path('data/output') / f"video_stabilizzato_{suffix}.mp4"
    
    # Verifica esistenza video di input
    if not input_video.exists():
        print(f"❌ Video di input non trovato: {input_video}")
        print()
        print("ISTRUZIONI:")
        print("1. Inserisci un video instabile nella cartella: data/input/")
        print("2. Rinomina il file come: video_instabile.mp4")
        print("   (oppure usa --input per specificare un altro video)")
        print()
        print("Puoi usare un video registrato con smartphone a mano libera")
        print("per testare il sistema di stabilizzazione.")
        return
    
    # Mostra informazioni sul video
    print(f"📹 Video di input trovato: {input_video}")
    print(f"📁 Output verrà salvato in: {output_video}")
    print()
    
    # Crea il Video Stabilizer con la configurazione caricata
    print("🔧 Inizializzazione Video Stabilizer...")
    stabilizer = VideoStabilizer(config)
    print("✅ Video Stabilizer pronto")
    print()
    
    # Esegui la stabilizzazione
    print("=" * 60)
    print("🎬 AVVIO STABILIZZAZIONE VIDEO")
    print("=" * 60)
    print()
    print("Questo processo si articola in due fasi:")
    print("  Fase 1: Stima del movimento e costruzione traiettoria")
    print("  Fase 2: Applicazione compensazione e salvataggio")
    print()
    
    try:
        # Chiama il metodo di stabilizzazione
        success = stabilizer.stabilize_video(
            input_path=str(input_video),
            output_path=str(output_video),
            show_progress=True  # Mostra il progresso durante l'elaborazione
        )
        
        # Verifica il risultato
        if success:
            print()
            print("=" * 60)
            print("✅ STABILIZZAZIONE COMPLETATA CON SUCCESSO!")
            print("=" * 60)
            print(f"📁 Video stabilizzato salvato in: {output_video}")
            
            # Salva metriche se richiesto
            if config.get('save_metrics', False):
                print()
                print("📊 Raccolta e salvataggio metriche...")
                metrics = stabilizer.get_metrics()
                
                # Genera nome file metriche basato sul metodo
                method = metrics.get('method', 'unknown')
                metrics_filename = f"video_stabilizzato_{method}_metrics.json"
                metrics_path = Path('data/output') / metrics_filename
                
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Metriche salvate in: {metrics_path}")
                print(f"   - Metodo: {metrics['method']}")
                print(f"   - Frames: {metrics['num_frames']}")
                print(f"   - RMS DX: {metrics['rms_dx']:.2f} px")
                print(f"   - RMS DY: {metrics['rms_dy']:.2f} px")
                print(f"   - Jitter Reduction X: {metrics['jitter_reduction_x']:.1f}%")
                print(f"   - Jitter Reduction Y: {metrics['jitter_reduction_y']:.1f}%")
                print(f"   - Tempo processing: {metrics['total_processing_time']:.2f} s")
            
            print()
            print("Puoi ora confrontare il video originale con quello stabilizzato")
            print("per valutare la riduzione del jitter e dei movimenti involontari.")
        else:
            print()
            print("=" * 60)
            print("❌ ERRORE DURANTE LA STABILIZZAZIONE")
            print("=" * 60)
            print("Controlla i messaggi di errore precedenti per dettagli.")
    
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERRORE IMPREVISTO")
        print("=" * 60)
        print(f"Dettagli: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Esegue la funzione principale
    main()
