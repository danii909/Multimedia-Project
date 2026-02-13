"""
Motion Estimation Module
Stima del movimento locale tramite Block Matching
"""

import numpy as np
from typing import Tuple, Optional


class MotionEstimator:
    """
    Classe per la stima del movimento locale tra frame consecutivi.
    Utilizza la tecnica del Block Matching con metrica SAD/MAD.
    """
    
    def __init__(self, 
                 block_size: int = 16,
                 search_range: int = 16,
                 metric: str = 'sad'):
        """
        Inizializza il Motion Estimator.
        
        Args:
            block_size: Dimensione dei blocchi (block_size x block_size)
            search_range: Raggio della finestra di ricerca
            metric: Metrica di similarità ('sad' o 'mad')
        """
        self.block_size = block_size
        self.search_range = search_range
        self.metric = metric
        
    def estimate_motion(self, 
                       frame_current: np.ndarray,
                       frame_previous: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stima il campo di movimento tra due frame consecutivi.
        
        Args:
            frame_current: Frame corrente (t)
            frame_previous: Frame precedente (t-1)
            
        Returns:
            motion_vectors: Array dei vettori di movimento (N, M, 2)
            confidence: Array di confidenza per ogni vettore
        """
        # Ottieni dimensioni dei frame
        # I frame devono essere in scala di grigi per il matching
        h, w = frame_current.shape[:2]
        
        # Calcola il numero di blocchi che copriranno il frame
        # Dividiamo l'immagine in una griglia regolare di blocchi
        n_blocks_y = h // self.block_size  # Numero di blocchi verticali
        n_blocks_x = w // self.block_size  # Numero di blocchi orizzontali
        
        # Inizializza array per memorizzare i risultati
        # motion_vectors[i, j] contiene il vettore (dx, dy) per il blocco in posizione (i, j)
        motion_vectors = np.zeros((n_blocks_y, n_blocks_x, 2), dtype=np.float32)
        # confidence[i, j] contiene l'errore di matching (più basso = migliore match)
        confidence = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)
        
        # Itera su tutti i blocchi dell'immagine
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                # Calcola le coordinate del blocco corrente nel frame
                y_start = i * self.block_size
                x_start = j * self.block_size
                y_end = y_start + self.block_size
                x_end = x_start + self.block_size
                
                # Estrai il blocco dal frame corrente
                # Questo è il blocco per cui cerchiamo il match
                block_current = frame_current[y_start:y_end, x_start:x_end]
                
                # Definisci l'area di ricerca nel frame precedente
                # Cerchiamo attorno alla stessa posizione, entro search_range pixel
                # IMPORTANTE: Usiamo coordinate top-left valide per evitare slice vuoti
                search_y_start = max(0, y_start - self.search_range)
                search_x_start = max(0, x_start - self.search_range)
                search_y_end = min(h - self.block_size, y_start + self.search_range) + self.block_size
                search_x_end = min(w - self.block_size, x_start + self.search_range) + self.block_size
                
                # Estrai l'area di ricerca dal frame precedente
                search_area = frame_previous[search_y_start:search_y_end, 
                                            search_x_start:search_x_end]
                
                # Cerca il miglior match nell'area di ricerca
                # Restituisce lo spostamento (dx, dy) e l'errore di matching
                dx, dy, error = self._search_block(
                    block_current, 
                    search_area,
                    (x_start, y_start),
                    (search_x_start, search_y_start)
                )
                
                # Memorizza il vettore di movimento per questo blocco
                motion_vectors[i, j, 0] = dx  # Spostamento orizzontale
                motion_vectors[i, j, 1] = dy  # Spostamento verticale
                
                # Memorizza la confidenza (inverso dell'errore normalizzato)
                # Più l'errore è basso, più la confidenza è alta
                # Normalizziamo per le dimensioni del blocco
                # IMPORTANTE: max_error dipende dalla metrica usata
                if self.metric == 'mad':
                    max_error = 255.0  # MAD ha range [0, 255]
                else:  # sad
                    max_error = 255.0 * self.block_size * self.block_size  # SAD ha range [0, 255*N^2]
                # Limita la confidence tra 0 e 1
                confidence[i, j] = max(0.0, min(1.0, 1.0 - (error / max_error)))
        
        return motion_vectors, confidence
    
    def _compute_sad(self, block1: np.ndarray, block2: np.ndarray) -> float:
        """Calcola Sum of Absolute Differences."""
        return np.sum(np.abs(block1.astype(float) - block2.astype(float)))
    
    def _compute_mad(self, block1: np.ndarray, block2: np.ndarray) -> float:
        """Calcola Mean Absolute Difference."""
        return np.mean(np.abs(block1.astype(float) - block2.astype(float)))
    
    def _search_block(self,
                     block: np.ndarray,
                     search_area: np.ndarray,
                     block_pos: Tuple[int, int],
                     search_pos: Tuple[int, int]) -> Tuple[int, int, float]:
        """
        Cerca il miglior match per un blocco nell'area di ricerca.
        
        Args:
            block: Blocco da cercare
            search_area: Area in cui cercare
            block_pos: Posizione originale del blocco (x, y)
            search_pos: Posizione dell'angolo superiore sinistro dell'area di ricerca
        
        Returns:
            dx, dy: Spostamento trovato
            error: Errore minimo di matching
        """
        # Dimensioni del blocco da cercare
        block_h, block_w = block.shape[:2]
        # Dimensioni dell'area di ricerca
        search_h, search_w = search_area.shape[:2]
        
        # Inizializza con valori molto alti (peggiore caso)
        min_error = float('inf')  # Errore minimo trovato finora
        best_dx = 0  # Miglior spostamento orizzontale
        best_dy = 0  # Miglior spostamento verticale
        
        # Esegui ricerca esaustiva nell'area di ricerca
        # Prova ogni possibile posizione per il blocco
        for y in range(search_h - block_h + 1):
            for x in range(search_w - block_w + 1):
                # Estrai il blocco candidato dall'area di ricerca
                candidate = search_area[y:y + block_h, x:x + block_w]
                
                # Verifica che il blocco candidato abbia le stesse dimensioni
                if candidate.shape != block.shape:
                    continue
                
                # Calcola l'errore di matching usando la metrica scelta
                if self.metric == 'sad':
                    # Sum of Absolute Differences: somma delle differenze assolute
                    error = self._compute_sad(block, candidate)
                elif self.metric == 'mad':
                    # Mean Absolute Difference: media delle differenze assolute
                    error = self._compute_mad(block, candidate)
                else:
                    # Default: usa SAD
                    error = self._compute_sad(block, candidate)
                
                # Se questo match è migliore del precedente, memorizzalo
                if error < min_error:
                    min_error = error
                    # Calcola lo spostamento rispetto alla posizione originale
                    # Considera l'offset dell'area di ricerca
                    best_dx = block_pos[0] - (search_pos[0] + x)
                    best_dy = block_pos[1] - (search_pos[1] + y)
        
        return best_dx, best_dy, min_error
