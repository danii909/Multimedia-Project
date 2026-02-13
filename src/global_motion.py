"""
Global Motion Estimation Module
Ricostruzione del movimento globale della camera
"""

import numpy as np
from typing import Tuple, Optional


class GlobalMotionEstimator:
    """
    Classe per la stima del movimento globale della telecamera
    dai vettori di movimento locali.
    """
    
    def __init__(self, 
                 motion_model: str = 'translation',
                 outlier_threshold: float = 2.0,
                 aggregation_method: str = 'median',
                 estimate_rotation: bool = True):
        """
        Inizializza il Global Motion Estimator.
        
        Args:
            motion_model: Modello di movimento ('translation', 'affine', 'homography')
            outlier_threshold: Soglia per rimozione outlier (deviazioni standard)
            aggregation_method: Metodo aggregazione ('median', 'mean', 'weighted_mean')
            estimate_rotation: Se True, stima anche l'angolo di rotazione (roll)
        """
        self.motion_model = motion_model
        self.outlier_threshold = outlier_threshold
        self.aggregation_method = aggregation_method
        self.estimate_rotation = estimate_rotation
        
    def estimate_global_motion(self,
                               motion_vectors: np.ndarray,
                               confidence: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Stima il Global Motion Vector (GMV) aggregando i vettori locali.
        
        Args:
            motion_vectors: Array dei vettori di movimento (N, M, 2)
            confidence: Array opzionale di confidenza per ogni vettore
            
        Returns:
            gmv_x: Componente orizzontale del movimento globale
            gmv_y: Componente verticale del movimento globale
            rotation_angle: Angolo di rotazione in gradi (roll)
        """
        # Fase 1: RIMOZIONE OUTLIER
        # Gli outlier sono vettori di movimento che si discostano significativamente
        # dalla tendenza generale e tipicamente corrispondono a oggetti in movimento
        # nella scena piuttosto che al movimento della camera
        filtered_vectors, inliers_mask = self._remove_outliers(motion_vectors, confidence)
        
        # Verifica che ci siano ancora vettori validi dopo il filtraggio
        if filtered_vectors.size == 0:
            # Se tutti i vettori sono stati rimossi, usa l'originale
            filtered_vectors = motion_vectors.reshape(-1, 2)
            filtered_confidence = confidence.flatten() if confidence is not None else None
        else:
            # Filtra anche la confidence usando la stessa maschera
            if confidence is not None:
                conf_flat = confidence.flatten()
                filtered_confidence = conf_flat[inliers_mask]
            else:
                filtered_confidence = None
        
        # Fase 2: AGGREGAZIONE DEI VETTORI
        # Aggreghiamo i vettori filtrati per ottenere un unico vettore globale
        # che rappresenta il movimento della camera
        
        method = (self.aggregation_method or 'median').lower()
        if method == 'weighted_mean':
            if filtered_confidence is not None:
                gmv_x, gmv_y = self._compute_weighted_mean(filtered_vectors, filtered_confidence)
            else:
                gmv_x, gmv_y = self._compute_mean_motion(filtered_vectors)
        elif method == 'mean':
            gmv_x, gmv_y = self._compute_mean_motion(filtered_vectors)
        else:  # 'median' (default)
            gmv_x, gmv_y = self._compute_median_motion(filtered_vectors)
        
        # Fase 3: STIMA ROTAZIONE (se abilitata)
        rotation_angle = 0.0
        if self.estimate_rotation:
            rotation_angle = self._estimate_rotation_angle(filtered_vectors)
        
        # Fase 4: RESTITUISCI IL GLOBAL MOTION VECTOR E ROTAZIONE
        # GMV rappresenta lo spostamento della camera tra due frame consecutivi
        return gmv_x, gmv_y, rotation_angle

    def _compute_mean_motion(self, motion_vectors: np.ndarray) -> Tuple[float, float]:
        """Calcola la media dei vettori di movimento."""
        mv_flat = motion_vectors.reshape(-1, 2)
        gmv_x = float(np.mean(mv_flat[:, 0]))
        gmv_y = float(np.mean(mv_flat[:, 1]))
        return gmv_x, gmv_y
    
    def _estimate_rotation_angle(self, motion_vectors: np.ndarray) -> float:
        """
        Stima l'angolo di rotazione dalla distribuzione dei vettori di movimento.
        Utilizza l'analisi del campo vettoriale per inferire rotazioni.
        
        Args:
            motion_vectors: Array filtrato dei vettori (N, 2)
            
        Returns:
            angle: Angolo di rotazione stimato in gradi
        """
        # Per una stima robusta della rotazione dal block matching,
        # analizziamo la componente rotazionale del campo vettoriale.
        # In una rotazione pura attorno al centro dell'immagine,
        # i vettori sono tangenti a cerchi concentrici.
        # Stimiamo l'angolo medio pesando la componente perpendicolare.
        
        if len(motion_vectors) < 3:
            return 0.0
        
        # Metodo semplificato: assume rotazione piccola e uniforme
        # Per rotazioni maggiori, servirebbero coordinate dei blocchi
        # o meglio usare optical flow con estimateAffinePartial2D
        
        # Calcola la varianza direzionale dei vettori
        # Una rotazione crea un pattern di vettori con angoli correlati
        angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
        
        # Rimuovi l'angolo medio (componente traslazionale)
        mean_angle = np.mean(angles)
        
        # Per block matching puro, la stima di rotazione è limitata
        # Restituiamo 0 e deleghiamo a optical flow se necessario
        # Questo verrà sovrascritto quando si usa optical flow
        return 0.0
    
    def _remove_outliers(self, 
                        motion_vectors: np.ndarray,
                        confidence: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rimuove i motion vector che rappresentano outlier.
        
        Returns:
            filtered_vectors: Motion vector filtrati
            inliers_mask: Maschera booleana degli inlier (per filtrare anche confidence)
        """
        # Appiattisci l'array 3D (N, M, 2) in 2D (N*M, 2)
        # Ogni riga rappresenta un vettore di movimento (dx, dy)
        mv_flat = motion_vectors.reshape(-1, 2)
        
        # METODO: Z-SCORE (Deviazione Standard)
        # Un vettore è considerato outlier se si discosta dalla media
        # più di outlier_threshold deviazioni standard
        
        # Calcola la media dei vettori per ogni componente
        mean_x = np.mean(mv_flat[:, 0])  # Media spostamento orizzontale
        mean_y = np.mean(mv_flat[:, 1])  # Media spostamento verticale
        
        # Calcola la deviazione standard per ogni componente
        std_x = np.std(mv_flat[:, 0])  # Deviazione std orizzontale
        std_y = np.std(mv_flat[:, 1])  # Deviazione std verticale
        
        # Evita divisione per zero se tutti i vettori sono identici
        if std_x == 0:
            std_x = 1.0
        if std_y == 0:
            std_y = 1.0
        
        # Calcola lo z-score (distanza dalla media in unità di deviazione standard)
        # per ogni vettore
        z_scores_x = np.abs((mv_flat[:, 0] - mean_x) / std_x)
        z_scores_y = np.abs((mv_flat[:, 1] - mean_y) / std_y)
        
        # Un vettore è considerato inlier se ENTRAMBE le componenti
        # hanno z-score inferiore alla soglia
        inliers_mask = (z_scores_x < self.outlier_threshold) & \
                       (z_scores_y < self.outlier_threshold)
        
        # Filtra i vettori mantenendo solo gli inlier
        filtered_vectors = mv_flat[inliers_mask]
        
        # Se abbiamo rimosso troppi vettori (>90%), potrebbe essere un problema
        # In questo caso, restituisci tutti i vettori per sicurezza
        if len(filtered_vectors) < len(mv_flat) * 0.1:
            # Crea una maschera con tutti True (nessun vettore rimosso)
            return mv_flat, np.ones(len(mv_flat), dtype=bool)
        
        return filtered_vectors, inliers_mask
    
    def _compute_median_motion(self, motion_vectors: np.ndarray) -> Tuple[float, float]:
        """Calcola la mediana dei vettori di movimento (più robusta agli outlier)."""
        mv_flat = motion_vectors.reshape(-1, 2)
        gmv_x = np.median(mv_flat[:, 0])
        gmv_y = np.median(mv_flat[:, 1])
        return gmv_x, gmv_y
    
    def _compute_weighted_mean(self,
                              motion_vectors: np.ndarray,
                              confidence: np.ndarray) -> Tuple[float, float]:
        """Calcola la media pesata dei vettori di movimento."""
        # MEDIA PESATA: i vettori con confidenza alta contribuiscono di più
        # Formula: weighted_mean = sum(value * weight) / sum(weight)
        
        # Gli array possono essere già appiattiti dopo il filtraggio outlier
        # Verifica le dimensioni e appiattisci solo se necessario
        if motion_vectors.ndim == 3:
            mv_flat = motion_vectors.reshape(-1, 2)
        else:
            mv_flat = motion_vectors
        
        if confidence.ndim > 1:
            conf_flat = confidence.flatten()
        else:
            conf_flat = confidence
        
        # Verifica che le dimensioni corrispondano
        if len(mv_flat) != len(conf_flat):
            raise ValueError(f"Mismatch dimensioni: motion_vectors={len(mv_flat)}, confidence={len(conf_flat)}")
        
        # Filtra vettori con confidence troppo bassa (< 0.01)
        valid_mask = conf_flat > 0.01
        
        # Se troppi vettori hanno confidence bassa, usa la mediana
        if np.sum(valid_mask) < len(conf_flat) * 0.1:
            return self._compute_median_motion(mv_flat)
        
        # Usa solo i vettori con confidence accettabile
        mv_valid = mv_flat[valid_mask]
        conf_valid = conf_flat[valid_mask]
        
        # Normalizza i pesi per evitare overflow
        # I pesi devono essere positivi e sommare a 1
        sum_conf = np.sum(conf_valid)
        if sum_conf < 1e-6:  # Se la somma è troppo bassa, usa mediana
            return self._compute_median_motion(mv_flat)
        
        weights = conf_valid / sum_conf
        
        # Calcola la media pesata per ogni componente
        # Ogni vettore contribuisce proporzionalmente alla sua confidenza
        gmv_x = np.sum(mv_valid[:, 0] * weights)  # Media pesata componente x
        gmv_y = np.sum(mv_valid[:, 1] * weights)  # Media pesata componente y
        
        return float(gmv_x), float(gmv_y)
    @staticmethod
    def extract_rotation_from_affine(M: np.ndarray) -> float:
        """
        Estrae l'angolo di rotazione da una matrice affine 2x3.
        
        Args:
            M: Matrice affine 2x3 (come restituita da estimateAffinePartial2D)
            
        Returns:
            angle: Angolo di rotazione in gradi
        """
        if M is None or M.shape != (2, 3):
            return 0.0
        
        # La matrice affine parziale ha la forma:
        # [cos(θ)*s  -sin(θ)*s  tx]
        # [sin(θ)*s   cos(θ)*s  ty]
        # dove θ è la rotazione e s è la scala
        
        # Estrai l'angolo di rotazione usando atan2
        # Usa la prima colonna per evitare divisioni per zero
        angle_rad = np.arctan2(M[1, 0], M[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)