"""
EnergyFingerprint
=================
Lightweight 1D-CNN for missense variant classification using
mRNA thermodynamic stacking profiles and ESM-1v evolutionary scores.

Modules:
  - energy: Nearest-neighbor stacking profiles (SantaLucia 1998 / Turner 2004)
  - genetics: Codon table, CDS translation, variant parsing
  - model: EnergySignalCNN, GradCAM1D, ChannelAttribution, training utilities
"""

__version__ = '1.0.0'
__author__ = 'Jose Antonio Vilar Sanchez'
__license__ = 'MIT'
