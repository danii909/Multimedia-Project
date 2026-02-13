"""
Video Stabilization System
Sistema di stabilizzazione video tramite stima e compensazione del movimento globale
"""

__version__ = "1.0.0"
__author__ = "Multimedia Project"

from .motion_estimation import MotionEstimator
from .global_motion import GlobalMotionEstimator
from .trajectory_smoothing import TrajectoryFilter
from .motion_compensation import MotionCompensator
from .video_stabilizer import VideoStabilizer

__all__ = [
    'MotionEstimator',
    'GlobalMotionEstimator',
    'TrajectoryFilter',
    'MotionCompensator',
    'VideoStabilizer'
]
