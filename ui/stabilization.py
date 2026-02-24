"""Logica di stabilizzazione: costruzione config e loop di esecuzione"""
import tempfile
import time
from src.video_stabilizer import VideoStabilizer
from ui.video_utils import convert_to_web_compatible


METHOD_NAMES = {
    "block_matching":                "📦 Block Matching",
    "optical_shi_tomasi_partial":    "🌊 Optical Flow (LK) - Partial",
    "optical_shi_tomasi_affine":     "🌊 Optical Flow (LK) - Affine",
    "optical_shi_tomasi_homography": "🌊 Optical Flow (LK) - Homography",
    "optical_orb_partial":           "🎯 ORB Matching - Partial",
    "optical_orb_affine":            "🎯 ORB Matching - Affine",
    "optical_orb_homography":        "🎯 ORB Matching - Homography",
}


def build_method_config(method, params):
    """
    Costruisce il dizionario di configurazione per un singolo metodo.

    Args:
        method: identificatore del metodo (es. 'block_matching')
        params: dict con tutti i parametri (letto da session_state)

    Returns:
        dict di configurazione da passare a VideoStabilizer
    """
    config = {
        'trajectory_smoothing': {
            'smoothing_window': params['smoothing_window'],
            'filter_type':      params['filter_type'],
            'deadband_px':      0.0,
        },
        'motion_compensation': {
            'crop_ratio':  params['crop_ratio'],
            'border_mode': params['border_mode'],
        },
    }

    if method == "block_matching":
        config['global_motion'] = {
            'estimation_method': 'block_matching',
            'motion_model':      'affine' if params['estimate_rotation_bm'] else 'translation',
            'aggregation_method': params['aggregation'],
            'outlier_threshold': 2.0,
            'estimate_rotation': params['estimate_rotation_bm'],
        }
        config['motion_estimation'] = {
            'block_size':   params['block_size'],
            'search_range': params['search_range'],
            'metric':       params['metric'],
        }

    elif method.startswith("optical_shi_tomasi"):
        transform_type = method.replace("optical_shi_tomasi_", "")
        config['global_motion'] = {
            'estimation_method': 'optical_flow',
            'motion_model':      'affine',
            'aggregation_method':'median',
            'outlier_threshold': 2.0,
            'estimate_rotation': True,
            'optical_flow': {
                'feature_type':          'shi_tomasi',
                'transform_type':        transform_type,
                'max_corners':           params['max_corners'],
                'quality_level':         params['quality_level'],
                'min_distance':          params['min_distance'],
                'win_size':              params['win_size'],
                'ransac_reproj_threshold': params['ransac_threshold_st'],
                'ransac_confidence':     0.995,
            },
        }

    elif method.startswith("optical_orb"):
        transform_type = method.replace("optical_orb_", "")
        config['global_motion'] = {
            'estimation_method': 'optical_flow',
            'motion_model':      'affine',
            'aggregation_method':'median',
            'outlier_threshold': 2.0,
            'estimate_rotation': True,
            'optical_flow': {
                'feature_type':            'orb',
                'transform_type':          transform_type,
                'orb_n_features':          params['orb_n_features'],
                'orb_scale_factor':        params['orb_scale_factor'],
                'orb_n_levels':            params['orb_n_levels'],
                'orb_edge_threshold':      31,
                'orb_ratio_threshold':     params['orb_ratio_threshold'],
                'ransac_reproj_threshold': params['ransac_threshold_orb'],
                'ransac_confidence':       0.995,
            },
        }

    return config


def read_params_from_session(st_session, defaults):
    """
    Legge i parametri di elaborazione da st.session_state con fallback ai default YAML.

    Args:
        st_session: st.session_state
        defaults: dict dei valori di default da config_loader

    Returns:
        dict dei parametri
    """
    def _get(key, default_key=None):
        return st_session.get(f'comp_{key}', defaults[default_key or key])

    return {
        'smoothing_window':     _get('smoothing_window'),
        'filter_type':          _get('filter_type'),
        'crop_ratio':           _get('crop_ratio'),
        'border_mode':          _get('border_mode'),
        'block_size':           _get('block_size'),
        'search_range':         _get('search_range'),
        'metric':               _get('metric'),
        'aggregation':          _get('aggregation'),
        'estimate_rotation_bm': _get('estimate_rotation_bm', 'estimate_rotation'),
        'max_corners':          _get('max_corners'),
        'quality_level':        _get('quality_level'),
        'min_distance':         _get('min_distance'),
        'win_size':             _get('win_size'),
        'ransac_threshold_st':  _get('ransac_threshold_st'),
        'orb_n_features':       _get('orb_n_features'),
        'orb_scale_factor':     _get('orb_scale_factor'),
        'orb_n_levels':         _get('orb_n_levels'),
        'orb_ratio_threshold':  _get('orb_ratio_threshold'),
        'ransac_threshold_orb': _get('ransac_threshold_orb'),
    }


def run_all_methods(selected_methods, input_path, params, progress_callbacks):
    """
    Esegue la stabilizzazione per tutti i metodi selezionati.

    Args:
        selected_methods: lista di metodi da elaborare
        input_path: path del video originale
        params: dict dei parametri (da read_params_from_session)
        progress_callbacks: dict con:
            'overall':       callable(fraction)   – barra di progresso globale
            'algo_slot':     callable(idx) → callable(fraction, text)
            'title':         callable(text)       – testo titolo

    Returns:
        dict {method: {'output_path', 'display_path', 'metrics', 'processing_time', 'success'}}
    """
    results = {}
    n = len(selected_methods)

    for idx, method in enumerate(selected_methods):
        display_name = METHOD_NAMES.get(method, method)

        progress_callbacks['title'](f"🔄 {display_name} ({idx + 1}/{n})...")
        progress_callbacks['algo_slot'](idx, 0.0, f"⏳ {display_name}: 0%")
        progress_callbacks['overall'](idx / n)

        def _make_cb(slot_idx, name):
            def _cb(frac):
                pct = int(frac * 100)
                progress_callbacks['algo_slot'](slot_idx, min(frac, 1.0), f"⏳ {name}: {pct}%")
            return _cb

        algo_cb = _make_cb(idx, display_name)
        config = build_method_config(method, params)

        try:
            start_time = time.time()
            stabilizer = VideoStabilizer(config)

            with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{method}.mp4') as tmp:
                output_path = tmp.name

            success = stabilizer.stabilize_video(input_path, output_path, progress_callback=algo_cb)
            elapsed = time.time() - start_time

            if success:
                web_output = output_path.replace('.mp4', '_web.mp4')
                ok = convert_to_web_compatible(output_path, web_output)
                results[method] = {
                    'output_path':    output_path,
                    'display_path':   web_output if ok else output_path,
                    'metrics':        stabilizer.get_metrics(),
                    'processing_time': elapsed,
                    'success':        True,
                }
            else:
                results[method] = {'success': False, 'error': 'Stabilization failed'}

        except Exception as e:
            results[method] = {'success': False, 'error': str(e)}

        progress_callbacks['algo_slot'](idx, 1.0, f"✅ {display_name}: completato")
        progress_callbacks['overall']((idx + 1) / n)

    progress_callbacks['title']("✅ Confronto completato!")
    return results
