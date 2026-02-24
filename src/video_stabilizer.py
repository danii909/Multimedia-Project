"""
Video Stabilizer - Pipeline Principale
Integra tutti i moduli per la stabilizzazione video completa
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

from .motion_estimation import MotionEstimator
from .global_motion import GlobalMotionEstimator
from .trajectory_smoothing import TrajectoryFilter
from .motion_compensation import MotionCompensator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoStabilizer:
    """
    Pipeline completa per la stabilizzazione video.
    
    Fasi:
    1. Stima del movimento locale (Block Matching)
    2. Stima del movimento globale (GMV)
    3. Filtraggio temporale della traiettoria
    4. Compensazione del movimento
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Inizializza il Video Stabilizer con la configurazione fornita.
        
        Args:
            config: Dizionario di configurazione (opzionale)
        """
        config = config or {}
        
        # Estrai le sottosezioni della configurazione
        me_config = config.get('motion_estimation', {})
        gm_config = config.get('global_motion', {})
        ts_config = config.get('trajectory_smoothing', {})
        mc_config = config.get('motion_compensation', {})

        self.gmv_estimation_method = (gm_config.get('estimation_method', 'block_matching') or 'block_matching').lower()
        self.optical_flow_config = gm_config.get('optical_flow', {}) or {}
        
        # Inizializza i moduli della pipeline con i parametri corretti
        self.motion_estimator = MotionEstimator(
            block_size=me_config.get('block_size', 16),
            search_range=me_config.get('search_range', 16),
            metric=me_config.get('metric', 'sad')
        )
        
        self.global_motion_estimator = GlobalMotionEstimator(
            motion_model=gm_config.get('motion_model', 'translation'),
            outlier_threshold=gm_config.get('outlier_threshold', 2.0),
            aggregation_method=gm_config.get('aggregation_method', 'median'),
            estimate_rotation=gm_config.get('estimate_rotation', True)
        )
        
        self.trajectory_filter = TrajectoryFilter(
            smoothing_window=ts_config.get('smoothing_window', 30),
            filter_type=ts_config.get('filter_type', 'moving_average'),
            gaussian_sigma=ts_config.get('gaussian_sigma', None),
            exponential_alpha=ts_config.get('exponential_alpha', None),
            deadband_px=ts_config.get('deadband_px', 0.0)
        )
        
        self.motion_compensator = MotionCompensator(
            crop_ratio=mc_config.get('crop_ratio', 0.9),
            border_mode=mc_config.get('border_mode', 'constant')  # Default: constant (was replicate)
        )
        
        # Inizializza struttura per raccolta metriche
        self.metrics = {
            'method': self.gmv_estimation_method,
            # raw_motion: moto incrementale stimato tra frame consecutivi (dx, dy, dangle)
            # raw_trajectory: traiettoria assoluta (cumulativa) (x, y, angle)
            'raw_motion': [],          # [(dx, dy, dangle), ...]
            'raw_trajectory': [],      # [(x, y, angle), ...]
            'smoothed_trajectory': [], # [(x, y, angle), ...]
            'frame_times': [],         # Tempo di processing per frame
            'config': config           # Configurazione usata
        }
        
        logger.info(f"📊 Configurazione caricata:")
        logger.info(f"   Motion Estimation: block_size={me_config.get('block_size', 16)}, search_range={me_config.get('search_range', 16)}, metric={me_config.get('metric', 'sad')}")
        
        if self.gmv_estimation_method == 'optical_flow':
            feature_type = self.optical_flow_config.get('feature_type', 'shi_tomasi')
            transform_type = self.optical_flow_config.get('transform_type', 'partial')
            transform_params = {'partial': '4 param', 'affine': '6 param', 'homography': '8 param'}
            transform_desc = transform_params.get(transform_type, '4 param')
            logger.info(
                f"   Global Motion: model={gm_config.get('motion_model', 'translation')}, "
                f"method={self.gmv_estimation_method}, "
                f"features={feature_type}, "
                f"transform={transform_type} ({transform_desc}), "
                f"ransac_threshold={self.optical_flow_config.get('ransac_reproj_threshold', 3.0)}"
            )
        else:
            logger.info(
                f"   Global Motion: model={gm_config.get('motion_model', 'translation')}, "
                f"method={self.gmv_estimation_method}, "
                f"outlier_threshold={gm_config.get('outlier_threshold', 2.0)}, "
                f"aggregation={gm_config.get('aggregation_method', 'median')}, "
                f"estimate_rotation={gm_config.get('estimate_rotation', True)}"
            )
        
        logger.info(f"   Trajectory Smoothing: window={ts_config.get('smoothing_window', 30)}, filter={ts_config.get('filter_type', 'moving_average')}, deadband_px={ts_config.get('deadband_px', 0.0)}")
        logger.info(f"   Motion Compensation: crop_ratio={mc_config.get('crop_ratio', 0.9)}, border_mode={mc_config.get('border_mode', 'constant')}")

    def _estimate_gmv(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float, float]:
        """Stima GMV (dx, dy, dangle) con il metodo configurato."""
        if self.gmv_estimation_method == 'optical_flow':
            return self._estimate_gmv_optical_flow(prev_gray, curr_gray)

        # Default/compatibilità: block matching + aggregazione (comportamento precedente)
        motion_vectors, confidence = self.motion_estimator.estimate_motion(curr_gray, prev_gray)
        return self.global_motion_estimator.estimate_global_motion(motion_vectors, confidence)

    def _estimate_gmv_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float, float]:
        """Stima GMV sub-pixel usando features (Shi-Tomasi/ORB) + RANSAC."""
        cfg = self.optical_flow_config

        # Scegli il tipo di feature detector
        feature_type = cfg.get('feature_type', 'shi_tomasi').lower()  # 'shi_tomasi' o 'orb'
        
        if feature_type == 'orb':
            p0_good, p1_good = self._extract_orb_matches(prev_gray, curr_gray, cfg)
        else:
            # Default: Shi-Tomasi corners + Lucas-Kanade optical flow
            p0_good, p1_good = self._extract_lk_matches(prev_gray, curr_gray, cfg)
        
        if p0_good is None or len(p0_good) < 10:
            return 0.0, 0.0, 0.0

        # Scegli il metodo RANSAC in base al transform type
        transform_type = cfg.get('transform_type', 'partial').lower()  # 'partial', 'affine', 'homography'
        ransac_thr = float(cfg.get('ransac_reproj_threshold', 3.0))
        
        if transform_type == 'homography':
            # 8 parametri - per movimenti prospettici (camera 3D)
            H, _inliers = cv2.findHomography(
                p0_good,
                p1_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thr,
            )
            
            if H is None:
                d = p1_good - p0_good
                return float(np.median(d[:, 0])), float(np.median(d[:, 1])), 0.0
            
            # Estrai parametri da homography (approssimazione)
            tx = float(H[0, 2])
            ty = float(H[1, 2])
            angle = self.global_motion_estimator.extract_rotation_from_affine(H[:2, :])
            
        elif transform_type == 'affine':
            # 6 parametri - affine completa (tx, ty, rot, scale_x, scale_y, shear)
            M, _inliers = cv2.estimateAffine2D(
                p0_good,
                p1_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thr,
            )
            
            if M is None:
                d = p1_good - p0_good
                return float(np.median(d[:, 0])), float(np.median(d[:, 1])), 0.0
            
            tx = float(M[0, 2])
            ty = float(M[1, 2])
            angle = self.global_motion_estimator.extract_rotation_from_affine(M)
            
        else:  # 'partial' (default)
            # 4 parametri - similarità rigida (tx, ty, rot, scale uniforme)
            M, _inliers = cv2.estimateAffinePartial2D(
                p0_good,
                p1_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thr,
            )
            
            if M is None:
                d = p1_good - p0_good
                return float(np.median(d[:, 0])), float(np.median(d[:, 1])), 0.0
            
            tx = float(M[0, 2])
            ty = float(M[1, 2])
            angle = self.global_motion_estimator.extract_rotation_from_affine(M)

        return tx, ty, angle

    def _extract_lk_matches(self, prev_gray: np.ndarray, curr_gray: np.ndarray, cfg: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estrae feature matches usando Shi-Tomasi corners + Lucas-Kanade optical flow."""
        max_corners = int(cfg.get('max_corners', 400))
        quality_level = float(cfg.get('quality_level', 0.01))
        min_distance = float(cfg.get('min_distance', 8))
        block_size = int(cfg.get('block_size', 7))

        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

        if p0 is None or len(p0) < 10:
            return None, None

        win_size = int(cfg.get('win_size', 21))
        max_level = int(cfg.get('max_level', 3))
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

        p1, st, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=criteria,
        )

        if p1 is None or st is None:
            return None, None

        st = st.reshape(-1) == 1
        p0_good = p0.reshape(-1, 2)[st]
        p1_good = p1.reshape(-1, 2)[st]

        return p0_good, p1_good

    def _extract_orb_matches(self, prev_gray: np.ndarray, curr_gray: np.ndarray, cfg: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estrae feature matches usando ORB descriptor + Brute Force matcher."""
        # Parametri ORB
        n_features = int(cfg.get('orb_n_features', 500))
        scale_factor = float(cfg.get('orb_scale_factor', 1.2))
        n_levels = int(cfg.get('orb_n_levels', 8))
        edge_threshold = int(cfg.get('orb_edge_threshold', 31))
        
        # Crea detector ORB
        orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold
        )
        
        # Rileva keypoints e descrittori
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None
        
        # Matching con BFMatcher (Brute Force con Hamming distance per ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Applica Lowe's ratio test per filtrare match ambigui
        ratio_threshold = float(cfg.get('orb_ratio_threshold', 0.75))
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, None
        
        # Estrai coordinate dei punti matchati
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        return pts1, pts2
        
    def stabilize_video(self, 
                       input_path: str, 
                       output_path: str,
                       show_progress: bool = True,
                       progress_callback=None) -> bool:
        """
        Stabilizza un video completo.
        
        Args:
            input_path: Path del video di input
            output_path: Path del video stabilizzato di output
            show_progress: Se True, mostra il progresso
            progress_callback: callable(float 0→1) chiamato ad ogni step di avanzamento
            
        Returns:
            success: True se la stabilizzazione è completata con successo
        """
        # Reset della traiettoria per nuova stabilizzazione
        self.trajectory_filter.reset()

        # Reset metriche (supporta più run nella stessa istanza)
        self.metrics['method'] = self.gmv_estimation_method
        self.metrics['raw_motion'] = []
        self.metrics['raw_trajectory'] = []
        self.metrics['smoothed_trajectory'] = []
        self.metrics['frame_times'] = []
        
        # Apri video di input
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Impossibile aprire il video: {input_path}")
            return False
        
        # Ottieni proprietà del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # Fallback a 30 fps se il valore letto è invalido
        if fps <= 0 or fps > 240:
            logger.warning(f"FPS invalido ({fps}), uso 30 fps di default")
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adatta la smoothing_window al video: clamp automatico se window > frame disponibili
        self.trajectory_filter.configure_for_video(total_frames, fps)

        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Crea directory di output se non esiste
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepara video di output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Verifica che il VideoWriter sia stato aperto correttamente
        if not out.isOpened():
            logger.error(f"Impossibile creare il video di output: {output_path}")
            cap.release()
            return False
        
        # Prima passata: stima movimento e costruisci traiettoria
        logger.info("Fase 1: Stima del movimento...")
        self._first_pass(cap, total_frames, show_progress, progress_callback)
        
        # Resetta video reader
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Seconda passata: applica compensazione
        logger.info("Fase 2: Compensazione del movimento...")
        self._second_pass(cap, out, total_frames, show_progress, progress_callback)
        
        # Cleanup
        cap.release()
        out.release()
        
        logger.info(f"Stabilizzazione completata: {output_path}")
        return True
    
    def _first_pass(self, cap, total_frames: int, show_progress: bool, progress_callback=None):
        """Prima passata: stima movimento e costruisci traiettoria."""
        import time
        
        ret, prev_frame = cap.read()
        if not ret:
            return
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_idx = 1
        
        total_motion_x = 0
        total_motion_y = 0
        total_rotation = 0
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Stima movimento globale (configurabile: block matching o optical flow)
            start_time = time.time()
            gmv_x, gmv_y, gmv_angle = self._estimate_gmv(prev_gray, curr_gray)
            frame_time = time.time() - start_time
            
            total_motion_x += abs(gmv_x)
            total_motion_y += abs(gmv_y)
            total_rotation += abs(gmv_angle)
            
            # Raccogli metriche
            self.metrics['raw_motion'].append((float(gmv_x), float(gmv_y), float(gmv_angle)))
            self.metrics['frame_times'].append(float(frame_time))
            
            # Aggiungi alla traiettoria
            self.trajectory_filter.add_motion(gmv_x, gmv_y, gmv_angle)
            
            prev_gray = curr_gray
            frame_idx += 1
            
            if frame_idx % 10 == 0:
                if progress_callback is not None:
                    progress_callback(frame_idx / max(total_frames, 1) * 0.5)
                if show_progress and frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    avg_mx = total_motion_x / frame_idx
                    avg_my = total_motion_y / frame_idx
                    avg_rot = total_rotation / frame_idx
                    logger.info(
                        f"Progresso: {progress:.1f}% - "
                        f"GMV medio: ({avg_mx:.2f}, {avg_my:.2f}) px, {avg_rot:.3f}°"
                    )
        
        logger.info(f"✅ Traiettoria costruita: {len(self.trajectory_filter.trajectory_x)} frames")
        logger.info(
            f"   Movimento totale rilevato: X={total_motion_x:.1f}px, "
            f"Y={total_motion_y:.1f}px, Rot={total_rotation:.2f}°"
        )

        # Salva traiettoria originale (assoluta/cumulativa) nelle metriche
        self.metrics['raw_trajectory'] = [
            (float(x), float(y), float(a))
            for x, y, a in zip(
                list(self.trajectory_filter.trajectory_x),
                list(self.trajectory_filter.trajectory_y),
                list(self.trajectory_filter.trajectory_angle),
            )
        ]
    
    def _second_pass(self, cap, out, total_frames: int, show_progress: bool, progress_callback=None):
        """Seconda passata: applica compensazione del movimento."""
        frame_idx = 0
        
        total_offset_x = 0
        total_offset_y = 0
        total_offset_angle = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ottieni trasformazione di stabilizzazione
            offset_x, offset_y, offset_angle = self.trajectory_filter.get_stabilization_transform(frame_idx)
            
            total_offset_x += abs(offset_x)
            total_offset_y += abs(offset_y)
            total_offset_angle += abs(offset_angle)
            
            # Applica compensazione
            stabilized = self.motion_compensator.compensate_frame(
                frame, offset_x, offset_y, offset_angle
            )
            
            # Scrivi frame stabilizzato
            out.write(stabilized)
            
            frame_idx += 1
            
            if frame_idx % 10 == 0:
                if progress_callback is not None:
                    progress_callback(0.5 + frame_idx / max(total_frames, 1) * 0.5)
                if show_progress and frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    avg_ox = total_offset_x / frame_idx
                    avg_oy = total_offset_y / frame_idx
                    avg_oa = total_offset_angle / frame_idx
                    logger.info(
                        f"Progresso: {progress:.1f}% - "
                        f"Offset medio: ({avg_ox:.2f}, {avg_oy:.2f}) px, {avg_oa:.3f}°"
                    )
        
        logger.info(f"✅ Compensazione completata: {frame_idx} frames processati")
        logger.info(
            f"   Offset totale applicato: X={total_offset_x:.1f}px, "
            f"Y={total_offset_y:.1f}px, Rot={total_offset_angle:.2f}°"
        )
        
        # Salva traiettoria smussata nelle metriche
        smoothed = self.trajectory_filter.get_smoothed_trajectory()
        self.metrics['smoothed_trajectory'] = [
            (float(smoothed[i, 0]), float(smoothed[i, 1]), float(smoothed[i, 2]))
            for i in range(len(smoothed))
        ]
    
    def get_metrics(self) -> dict:
        """
        Restituisce le metriche raccolte durante la stabilizzazione.
        
        Returns:
            dict: Dizionario contenente:
                - method: Metodo di stima GMV usato
                - raw_motion: Lista di tuple (dx, dy, dangle) moto incrementale stimato per frame
                - raw_trajectory: Lista di tuple (x, y, angle) traiettoria assoluta (cumulativa)
                - smoothed_trajectory: Lista di tuple (x, y, angle) movimento smoothed
                - rms_dx: Root Mean Square displacement X
                - rms_dy: Root Mean Square displacement Y  
                - rms_angle: Root Mean Square rotazione
                - jitter_reduction_x: Riduzione jitter asse X (%)
                - jitter_reduction_y: Riduzione jitter asse Y (%)
                - jitter_reduction_angle: Riduzione jitter rotazione (%)
                - avg_frame_time: Tempo medio elaborazione per frame (s)
                - total_processing_time: Tempo totale elaborazione (s)
                - num_frames: Numero di frames processati
                - config: Configurazione usata
        """
        if not self.metrics['raw_trajectory']:
            return {'error': 'Nessuna metrica raccolta. Esegui prima stabilize_video()'}
        
        raw_traj = np.array(self.metrics['raw_trajectory'])
        smooth_traj = np.array(self.metrics['smoothed_trajectory'])
        
        # Assicurati che abbiano la stessa lunghezza (potrebbero differire di 1 frame)
        min_len = min(len(raw_traj), len(smooth_traj))
        raw_traj = raw_traj[:min_len]
        smooth_traj = smooth_traj[:min_len]
        
        # RMS del moto incrementale (per-frame), non della traiettoria cumulativa
        if len(raw_traj) >= 2:
            raw_step = np.diff(raw_traj, axis=0)
            rms_dx = float(np.sqrt(np.mean(raw_step[:, 0]**2)))
            rms_dy = float(np.sqrt(np.mean(raw_step[:, 1]**2)))
            rms_angle = float(np.sqrt(np.mean(raw_step[:, 2]**2)))
        else:
            rms_dx = 0.0
            rms_dy = 0.0
            rms_angle = 0.0
        
        # Calcola riduzione jitter (varianza sul moto incrementale)
        def calculate_jitter_reduction(raw_abs, smooth_abs):
            if len(raw_abs) < 3 or len(smooth_abs) < 3:
                return np.zeros(3)
            raw_step = np.diff(raw_abs, axis=0)
            smooth_step = np.diff(smooth_abs, axis=0)
            raw_var = np.var(raw_step, axis=0)
            smooth_var = np.var(smooth_step, axis=0)
            reduction = np.zeros(3)
            for i in range(3):
                if raw_var[i] > 0:
                    reduction[i] = (1.0 - smooth_var[i] / raw_var[i]) * 100.0
            return reduction
        
        jitter_reduction = calculate_jitter_reduction(raw_traj, smooth_traj)
        
        # Calcola crop loss stimato (bounds della traiettoria corretta)
        offsets_x = smooth_traj[:, 0] - raw_traj[:, 0]
        offsets_y = smooth_traj[:, 1] - raw_traj[:, 1]
        max_offset_x = float(np.max(np.abs(offsets_x)))
        max_offset_y = float(np.max(np.abs(offsets_y)))
        
        return {
            'method': self.metrics['method'],
            'raw_motion': self.metrics.get('raw_motion', [])[:max(0, min_len - 1)],
            'raw_trajectory': self.metrics['raw_trajectory'][:min_len],
            'smoothed_trajectory': self.metrics['smoothed_trajectory'][:min_len],
            'rms_dx': rms_dx,
            'rms_dy': rms_dy,
            'rms_angle': rms_angle,
            'jitter_reduction_x': float(jitter_reduction[0]),
            'jitter_reduction_y': float(jitter_reduction[1]),
            'jitter_reduction_angle': float(jitter_reduction[2]),
            'max_offset_x': max_offset_x,
            'max_offset_y': max_offset_y,
            'avg_frame_time': float(np.mean(self.metrics['frame_times'])),
            'total_processing_time': float(np.sum(self.metrics['frame_times'])),
            'num_frames': min_len,
            'smoothing_window_used':      self.trajectory_filter.smoothing_window,
            'smoothing_window_requested': self.trajectory_filter._requested_smoothing_window,
            'smoothing_window_clamped':   self.trajectory_filter.smoothing_window_clamped,
            'config': self.metrics['config']
        }
