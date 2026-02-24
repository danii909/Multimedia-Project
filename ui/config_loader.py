"""Caricamento configurazioni YAML e valori di default"""
from pathlib import Path
import yaml


def load_configs(config_dir=None):
    """
    Carica i file YAML di configurazione e restituisce un dizionario
    con tutti i valori di default e le opzioni per i widget.

    Args:
        config_dir: Path alla cartella config. Se None usa 'config/' relativa a questo file.

    Returns:
        dict con chiavi:
            raw_block, raw_optical  – dizionari YAML grezzi
            defaults                – valori di default per tutti i parametri
            options                 – liste per i selectbox/dropdowns
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / 'config'
    else:
        config_dir = Path(config_dir)

    with open(config_dir / 'config_block_matching.yaml', 'r') as f:
        raw_block = yaml.safe_load(f)

    with open(config_dir / 'config_optical_flow.yaml', 'r') as f:
        raw_optical = yaml.safe_load(f)

    of = raw_optical['global_motion']['optical_flow']

    defaults = {
        # Block Matching
        'block_size':           raw_block['motion_estimation']['block_size'],
        'search_range':         raw_block['motion_estimation']['search_range'],
        'metric':               raw_block['motion_estimation']['metric'],
        'aggregation':          raw_block['global_motion']['aggregation_method'],
        'estimate_rotation':    raw_block['global_motion']['estimate_rotation'],
        # Optical Flow – Shi-Tomasi
        'max_corners':          of['max_corners'],
        'quality_level':        of['quality_level'],
        'min_distance':         of['min_distance'],
        'win_size':             of['win_size'],
        'ransac_threshold_st':  of['ransac_reproj_threshold'],
        # Optical Flow – ORB
        'orb_n_features':       of.get('orb_n_features', 500),
        'orb_scale_factor':     of.get('orb_scale_factor', 1.2),
        'orb_n_levels':         of.get('orb_n_levels', 8),
        'orb_ratio_threshold':  of.get('orb_ratio_threshold', 0.75),
        'ransac_threshold_orb': of['ransac_reproj_threshold'],
        # Comuni (smoothing & compensation)
        'smoothing_window':     raw_block['trajectory_smoothing']['smoothing_window'],
        'filter_type':          raw_block['trajectory_smoothing']['filter_type'],
        'crop_ratio':           raw_block['motion_compensation']['crop_ratio'],
        'border_mode':          raw_block['motion_compensation']['border_mode'],
    }

    options = {
        'metric':      ['sad', 'mad', 'ssd', 'ncc'],
        'aggregation': ['median', 'mean', 'weighted'],
        'filter':      ['moving_average', 'gaussian', 'exponential'],
        'border':      ['constant', 'replicate', 'reflect'],
    }

    return {
        'raw_block':   raw_block,
        'raw_optical': raw_optical,
        'defaults':    defaults,
        'options':     options,
    }
