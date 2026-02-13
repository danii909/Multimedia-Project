"""
Trajectory Smoothing Module
Filtraggio temporale del movimento per separare movimento intenzionale da jitter
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque


class TrajectoryFilter:
    """
    Classe per il filtraggio temporale della traiettoria della camera.
    Separa il movimento intenzionale dal jitter.
    """
    
    def __init__(self, 
                 smoothing_window: int = 30,
                 filter_type: str = 'moving_average',
                 gaussian_sigma: Optional[float] = None,
                 exponential_alpha: Optional[float] = None,
                 deadband_px: float = 0.0):
        """
        Inizializza il Trajectory Filter.
        
        Args:
            smoothing_window: Finestra temporale per lo smoothing (frame)
            filter_type: Tipo di filtro ('moving_average', 'gaussian', 'exponential')
            gaussian_sigma: Sigma per filtro gaussiano (opzionale)
            exponential_alpha: Alpha per filtro esponenziale (opzionale, 0<alpha<1)
            deadband_px: Soglia (px) sotto la quale l'offset viene azzerato
        """
        self.smoothing_window = smoothing_window
        self.filter_type = filter_type
        self.gaussian_sigma = gaussian_sigma
        self.exponential_alpha = exponential_alpha
        self.deadband_px = float(deadband_px) if deadband_px is not None else 0.0

        self._smoothed_cache: Optional[np.ndarray] = None
        # Rimuovo maxlen per evitare perdita di dati su video lunghi
        self.trajectory_x = deque()
        self.trajectory_y = deque()
        self.trajectory_angle = deque()  # Angolo di rotazione (gradi)
        # Inizializza con (0,0,0) per il frame 0 (riferimento assoluto)
        self.trajectory_x.append(0.0)
        self.trajectory_y.append(0.0)
        self.trajectory_angle.append(0.0)
    
    def reset(self):
        """Reset della traiettoria per nuova stabilizzazione."""
        self.trajectory_x.clear()
        self.trajectory_y.clear()
        self.trajectory_angle.clear()
        self._smoothed_cache = None
        # Reinizializza con (0,0,0) per frame 0
        self.trajectory_x.append(0.0)
        self.trajectory_y.append(0.0)
        self.trajectory_angle.append(0.0)
        
    def add_motion(self, dx: float, dy: float, dangle: float = 0.0):
        """
        Aggiunge un nuovo vettore di movimento alla traiettoria.
        
        Args:
            dx: Spostamento orizzontale
            dy: Spostamento verticale
            dangle: Rotazione incrementale in gradi
        """
        # Accumula sempre dalla posizione precedente (già inizializzata con (0,0,0))
        self.trajectory_x.append(self.trajectory_x[-1] + dx)
        self.trajectory_y.append(self.trajectory_y[-1] + dy)
        self.trajectory_angle.append(self.trajectory_angle[-1] + dangle)
        self._smoothed_cache = None
    
    def get_smoothed_trajectory(self) -> np.ndarray:
        """
        Restituisce la traiettoria smussata.
        
        Returns:
            smoothed_trajectory: Array (N, 3) con traiettoria stabilizzata (x, y, angle)
        """
        if self._smoothed_cache is not None:
            return self._smoothed_cache

        if len(self.trajectory_x) < self.smoothing_window:
            # Se non ci sono abbastanza frame, restituisce la traiettoria originale
            self._smoothed_cache = np.column_stack([
                np.array(self.trajectory_x),
                np.array(self.trajectory_y),
                np.array(self.trajectory_angle)
            ])
            return self._smoothed_cache
        
        if self.filter_type == 'moving_average':
            self._smoothed_cache = self._moving_average_filter()
        elif self.filter_type == 'gaussian':
            self._smoothed_cache = self._gaussian_filter()
        elif self.filter_type == 'exponential':
            self._smoothed_cache = self._exponential_filter()
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        return self._smoothed_cache
    
    def _moving_average_filter(self) -> np.ndarray:
        """Aplica un filtro a media mobile."""
        # Converti le deque in array numpy per elaborazione efficiente
        trajectory = np.column_stack([
            np.array(self.trajectory_x),
            np.array(self.trajectory_y),
            np.array(self.trajectory_angle)
        ])
        
        # FILTRO A MEDIA MOBILE (Moving Average)
        # Calcola la media dei valori in una finestra scorrevole
        # Questo rimuove le oscillazioni ad alta frequenza (jitter)
        # mantenendo i movimenti lenti (panoramiche intenzionali)
        
        n_frames = len(trajectory)
        smoothed = np.zeros_like(trajectory)
        
        # Per ogni frame, calcola la media di una finestra centrata
        for i in range(n_frames):
            # Definisci i limiti della finestra centrata sul frame corrente
            # La finestra si estende per smoothing_window/2 frame prima e dopo
            half_window = self.smoothing_window // 2
            start_idx = max(0, i - half_window)
            end_idx = min(n_frames, i + half_window + 1)
            
            # Estrai i valori nella finestra
            window_data = trajectory[start_idx:end_idx]
            
            # Calcola la media per ogni componente (x, y, angle)
            # Questa è la posizione "smussata" al frame i
            smoothed[i, 0] = np.mean(window_data[:, 0])  # Componente x
            smoothed[i, 1] = np.mean(window_data[:, 1])  # Componente y
            smoothed[i, 2] = np.mean(window_data[:, 2])  # Angolo
        
        # ALTERNATIVA EFFICIENTE CON CONVOLUZIONE
        # Usa np.convolve per calcolare la media mobile in modo più efficiente
        # kernel = np.ones(self.smoothing_window) / self.smoothing_window
        # smoothed[:, 0] = np.convolve(trajectory[:, 0], kernel, mode='same')
        # smoothed[:, 1] = np.convolve(trajectory[:, 1], kernel, mode='same')
        
        return smoothed
    
    def _gaussian_filter(self) -> np.ndarray:
        """Aplica un filtro gaussiano."""
        # Converti traiettoria in array numpy
        trajectory = np.column_stack([
            np.array(self.trajectory_x),
            np.array(self.trajectory_y),
            np.array(self.trajectory_angle)
        ])
        
        # FILTRO GAUSSIANO
        # Usa una finestra pesata gaussiana invece di una media uniforme
        # I frame più vicini hanno peso maggiore, i più lontani peso minore
        # Questo produce uno smoothing più naturale rispetto alla media mobile
        
        # Crea il kernel gaussiano
        # sigma determina la "larghezza" della gaussiana
        sigma = float(self.gaussian_sigma) if self.gaussian_sigma else (self.smoothing_window / 6.0)
        if sigma <= 0:
            sigma = self.smoothing_window / 6.0
        
        # Genera la finestra gaussiana
        # La gaussiana è centrata e simmetrica
        x = np.arange(-self.smoothing_window // 2, self.smoothing_window // 2 + 1)
        gaussian_kernel = np.exp(-(x**2) / (2 * sigma**2))
        # Normalizza il kernel per sommare a 1 (conservazione dell'energia)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        
        n_frames = len(trajectory)
        smoothed = np.zeros_like(trajectory)
        
        # Applica il filtro gaussiano ad ogni frame
        for i in range(n_frames):
            # Definisci la finestra centrata sul frame corrente
            half_window = self.smoothing_window // 2
            start_idx = max(0, i - half_window)
            end_idx = min(n_frames, i + half_window + 1)
            
            # Estrai i valori nella finestra
            window_data = trajectory[start_idx:end_idx]
            
            # Adatta il kernel alla dimensione effettiva della finestra
            # (ai bordi la finestra può essere più piccola)
            kernel_start = half_window - (i - start_idx)
            kernel_end = kernel_start + len(window_data)
            window_kernel = gaussian_kernel[kernel_start:kernel_end]
            
            # Rinormalizza il kernel per la finestra corrente
            window_kernel = window_kernel / np.sum(window_kernel)
            
            # Calcola la media pesata gaussiana
            smoothed[i, 0] = np.sum(window_data[:, 0] * window_kernel)
            smoothed[i, 1] = np.sum(window_data[:, 1] * window_kernel)
            smoothed[i, 2] = np.sum(window_data[:, 2] * window_kernel)
        
        return smoothed
    
    def _exponential_filter(self) -> np.ndarray:
        """Aplica un filtro esponenziale (IIR)."""
        # Converti traiettoria in array numpy
        trajectory = np.column_stack([
            np.array(self.trajectory_x),
            np.array(self.trajectory_y),
            np.array(self.trajectory_angle)
        ])
        
        # FILTRO ESPONENZIALE (Exponential Moving Average - EMA)
        # Tipo di filtro IIR (Infinite Impulse Response)
        # Formula: smoothed[i] = alpha * trajectory[i] + (1-alpha) * smoothed[i-1]
        # 
        # Caratteristiche:
        # - Risponde rapidamente ai cambiamenti (low latency)
        # - Pesa maggiormente i frame recenti
        # - Computazionalmente molto efficiente (elaborazione single-pass)
        # - Alpha vicino a 1 = poco smoothing (segue più fedelmente l'originale)
        # - Alpha vicino a 0 = molto smoothing (più lento a reagire)
        
        if self.exponential_alpha is None:
            # Calcola alpha dal smoothing_window
            # Relazione approssimativa: alpha = 2 / (window + 1)
            alpha = 2.0 / (self.smoothing_window + 1)
        else:
            alpha = float(self.exponential_alpha)
        # Clamp di sicurezza
        if not (0.0 < alpha < 1.0):
            alpha = 2.0 / (self.smoothing_window + 1)
        
        n_frames = len(trajectory)
        smoothed = np.zeros_like(trajectory)
        
        # Inizializza il primo frame con il valore originale
        smoothed[0] = trajectory[0]
        
        # Applica il filtro esponenziale frame per frame
        # Ogni frame dipende solo dal frame precedente (IIR)
        for i in range(1, n_frames):
            # Formula EMA per componente x
            smoothed[i, 0] = alpha * trajectory[i, 0] + (1 - alpha) * smoothed[i-1, 0]
            # Formula EMA per componente y
            smoothed[i, 1] = alpha * trajectory[i, 1] + (1 - alpha) * smoothed[i-1, 1]
            # Formula EMA per angolo
            smoothed[i, 2] = alpha * trajectory[i, 2] + (1 - alpha) * smoothed[i-1, 2]
        
        # VANTAGGI:
        # - Elaborazione in tempo reale (single-pass, causale)
        # - Memoria O(1) invece di O(window)
        # - Ottimo per streaming video
        
        return smoothed
    
    def get_stabilization_transform(self, frame_idx: int) -> Tuple[float, float, float]:
        """
        Calcola la trasformazione necessaria per stabilizzare il frame corrente.
        
        Args:
            frame_idx: Indice del frame corrente
            
        Returns:
            offset_x, offset_y, offset_angle: Offset da applicare per compensare il movimento
        """
        smoothed = self.get_smoothed_trajectory()
        
        if frame_idx >= len(smoothed):
            return 0.0, 0.0, 0.0
        
        # La compensazione è la differenza tra traiettoria originale e smussata
        original_x = self.trajectory_x[frame_idx]
        original_y = self.trajectory_y[frame_idx]
        original_angle = self.trajectory_angle[frame_idx]
        smoothed_x = smoothed[frame_idx, 0]
        smoothed_y = smoothed[frame_idx, 1]
        smoothed_angle = smoothed[frame_idx, 2]
        
        offset_x = smoothed_x - original_x
        offset_y = smoothed_y - original_y
        offset_angle = smoothed_angle - original_angle

        # Deadband: evita di applicare micro-offset rumorosi
        if self.deadband_px > 0.0:
            if abs(offset_x) < self.deadband_px:
                offset_x = 0.0
            if abs(offset_y) < self.deadband_px:
                offset_y = 0.0
        
        return offset_x, offset_y, offset_angle
