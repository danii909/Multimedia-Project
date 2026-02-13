"""
Motion Compensation Module
Compensazione del movimento tramite trasformazioni geometriche inverse
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class MotionCompensator:
    """
    Classe per la compensazione del movimento e stabilizzazione dei frame.
    """
    
    def __init__(self, 
                 crop_ratio: float = 0.9,
                 border_mode: str = 'constant'):
        """
        Inizializza il Motion Compensator.
        
        Args:
            crop_ratio: Rapporto di crop per evitare bordi neri (0-1)
            border_mode: Modalità di gestione bordi ('replicate', 'reflect', 'constant')
        """
        self.crop_ratio = crop_ratio
        self.border_mode = border_mode
        self._border_mode_map = {
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT,
            'constant': cv2.BORDER_CONSTANT
        }
        
    def compensate_frame(self,
                        frame: np.ndarray,
                        offset_x: float,
                        offset_y: float,
                        offset_angle: float = 0.0) -> np.ndarray:
        """
        Applica la compensazione del movimento a un frame.
        
        Args:
            frame: Frame da stabilizzare
            offset_x: Offset orizzontale per la compensazione
            offset_y: Offset verticale per la compensazione
            offset_angle: Offset angolare per la compensazione (gradi)
            
        Returns:
            stabilized_frame: Frame stabilizzato
        """
        # COMPENSAZIONE DEL MOVIMENTO
        # Principio: Se la camera si è spostata di (dx, dy) e ruotata di dangle,
        # applichiamo una trasformazione inversa per "annullare" il movimento
        # e mantenere il frame "fermo" rispetto a un riferimento fisso
        
        # Fase 1: CREA MATRICE DI TRASFORMAZIONE AFFINE
        # La matrice trasla e ruota l'immagine nella direzione opposta al movimento
        h, w = frame.shape[:2]
        M = self._create_affine_matrix(offset_x, offset_y, offset_angle, w, h)
        
        # Fase 2: APPLICA TRASFORMAZIONE GEOMETRICA
        # warpAffine applica la trasformazione affine a tutto il frame
        # Ricampiona i pixel nella nuova posizione
        
        # Ottieni il border mode da OpenCV
        border_mode = self._get_border_mode()
        
        # Adatta borderValue al numero di canali del frame
        if len(frame.shape) == 2:  # Grayscale
            border_value = 0
        else:  # BGR/RGB
            border_value = (0, 0, 0)
        
        # Applica la trasformazione affine
        # - M: matrice di trasformazione 2x3
        # - (w, h): dimensioni output (manteniamo le dimensioni originali)
        # - borderMode: come gestire i pixel fuori dai bordi
        # - borderValue: valore per i pixel fuori bordo (se border_mode=CONSTANT)
        stabilized = cv2.warpAffine(
            frame,                    # Frame di input
            M,                        # Matrice di trasformazione
            (w, h),                   # Dimensioni output
            flags=cv2.INTER_LINEAR,   # Interpolazione bilineare (buon compromesso qualità/velocità)
            borderMode=border_mode,   # Modalità di gestione bordi
            borderValue=border_value  # Nero per i bordi se CONSTANT
        )
        
        # Fase 3: GESTIONE BORDI
        # NOTA CRITICA: NON applichiamo crop+resize qui because annulla la stabilizzazione!
        # apply_crop() con resize ridimensiona il frame alle dim originali, cancellando l'effetto.
        # Restituiamo il frame così com'è per preservare la compensazione.
        
        return stabilized
    
    def _create_translation_matrix(self, tx: float, ty: float) -> np.ndarray:
        """
        Crea una matrice di trasformazione affine per traslazione.
        
        Args:
            tx: Traslazione orizzontale
            ty: Traslazione verticale
            
        Returns:
            M: Matrice di trasformazione 2x3
        """
        M = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
        return M
    
    def _create_affine_matrix(self, tx: float, ty: float, angle: float, 
                             width: int, height: int) -> np.ndarray:
        """
        Crea una matrice di trasformazione affine per traslazione + rotazione.
        
        Args:
            tx: Traslazione orizzontale
            ty: Traslazione verticale
            angle: Angolo di rotazione in gradi
            width: Larghezza del frame
            height: Altezza del frame
            
        Returns:
            M: Matrice di trasformazione 2x3
        """
        if abs(angle) < 0.01:  # Se la rotazione è trascurabile, usa solo traslazione
            return self._create_translation_matrix(tx, ty)
        
        # Centro del frame (punto fisso di rotazione)
        cx = width / 2.0
        cy = height / 2.0
        
        # Crea matrice di rotazione attorno al centro
        # OpenCV usa angoli in gradi e centro di rotazione specificato
        # NOTA: neghiamo l'angolo perché offset_angle è la CORREZIONE
        # Se il video ha ruotato di +5°, applichiamo -5° per stabilizzare
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(cx, cy),
            angle=-angle,  # Negazione: compensazione inversa del movimento
            scale=1.0     # Nessuna scalatura
        )
        
        # Aggiungi la traslazione alla matrice di rotazione
        # La traslazione deve essere applicata DOPO la rotazione
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty
        
        return rotation_matrix

    def apply_crop(self, frame: np.ndarray) -> np.ndarray:
        """
        Applica crop centrale per rimuovere bordi vuoti.
        
        Args:
            frame: Frame da croppare
            
        Returns:
            cropped_frame: Frame croppato
        """
        h, w = frame.shape[:2]
        new_h = int(h * self.crop_ratio)
        new_w = int(w * self.crop_ratio)
        
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
        
        # Ridimensiona al size originale
        return cv2.resize(cropped, (w, h))
    
    def _get_border_mode(self) -> int:
        """Restituisce il border mode di OpenCV."""
        return self._border_mode_map.get(self.border_mode, cv2.BORDER_REPLICATE)
