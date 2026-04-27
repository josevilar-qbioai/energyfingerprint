# EnergyFingerprint

[🇪🇸 Leer en español](README_ES.md)

[![DOI Paper](https://img.shields.io/badge/Paper-10.5281/zenodo.19831154-blue)](https://doi.org/10.5281/zenodo.19831154)
[![DOI Code](https://img.shields.io/badge/Code-10.5281/zenodo.19784943-green)](https://doi.org/10.5281/zenodo.19784943)

**mRNA thermodynamic profiling + protein language models for zero-shot missense variant classification.**

EnergyFingerprint is a lightweight 1D-CNN (~228K parameters) that classifies missense variants as pathogenic or benign by combining two orthogonal information axes:

- **Biophysical axis:** mRNA stacking free energy profiles (ΔG nearest-neighbor, SantaLucia 1998 + Turner 2004)
- **Evolutionary axis:** ESM-1v protein language model (masked marginal scoring, log-likelihood ratio)

Each variant is encoded as an **11-channel × 128-nucleotide tensor** and classified by a CNN with multi-head self-attention.

> **Patent:** EnergoRNA P202630522 (OEPM, April 2026). The method is patent-protected; this code is released under the MIT license for research purposes.

---

## Key Results

### Intra-gene classification (K-fold CV)

| Gene | Disease | Family | AUC |
|------|---------|--------|:---:|
| BRCA1 | Breast/ovarian cancer | DNA repair | 0.943 |
| TP53 | Multi-cancer | Tumor suppressor | 0.907 |
| PTEN | Multi-cancer | Phosphatase | 0.994 |
| PALB2 | Breast cancer | DNA repair | 0.950 |
| CFTR | Cystic fibrosis | ABC transporter | 0.979 |
| HBB | Sickle cell disease | Globin | 0.721 |
| SCN1A | Epilepsy/Dravet | Voltage-gated Na⁺ channel | 0.868 |
| MECP2 | Rett syndrome | Transcription regulator (IDR) | 0.911 |

### Zero-shot cross-gene transfer (train BRCA1 → predict others, no retraining)

| Target gene | Zero-shot AUC | Transfer type |
|------------|:------------:|---------------|
| TP53 | 0.854 | Cancer → Cancer |
| PTEN | 0.975 | Cancer → Cancer |
| PALB2 | 0.954 | Cancer → Cancer |
| CFTR | 0.777 | Cancer → ABC transporter |
| HBB | 0.707 | Cancer → Globin |
| SCN1A | 0.637 | Cancer → TM channel |
| MECP2 | 0.430 | Cancer → IDR (**fails**) |

**Transferability gradient:** Cancer (~0.95) > ABC transporter (~0.78) > Globin (~0.71) > TM channel (~0.64) > IDR (~0.43)

---

## Architecture

```
Input: 11-channel × 128-nucleotide tensor
  │
  ├─ Channels 1-4: ΔG stacking profiles (wt/mut × SantaLucia/Turner)
  ├─ Channels 5-6: ΔΔG difference profiles
  ├─ Channel 7:    GC content (sliding window)
  ├─ Channel 8:    Purine fraction (sliding window)
  ├─ Channel 9:    Codon position indicator
  ├─ Channel 10:   Reserved (mRNA secondary structure)
  └─ Channel 11:   ESM-1v LLR × Gaussian kernel
  │
  ▼
  Conv1D(11→64, k=7) → BN → ReLU → Dropout(0.3)
  Conv1D(64→128, k=5) → BN → ReLU → Dropout(0.3)
  Multi-head Self-Attention (4 heads)
  Conv1D(128→64, k=3) → BN → ReLU
  Global Average Pooling
  FC(64→2) → Softmax
  │
  ▼
  Pathogenic / Benign
```

~228K trainable parameters. Interpretable via GradCAM-1D and channel ablation.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/josevilar-qbioai/energyfingerprint.git
cd energyfingerprint

# Create conda environment
conda env create -f environment.yml
conda activate energyfp
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- fair-esm ≥ 2.0 (for ESM-1v scoring)
- BioPython, NumPy, Pandas, scikit-learn, matplotlib, seaborn

See `environment.yml` for the complete dependency list.

---

## Usage

### Quick start: classify variants for a new gene

**Step 1.** Download ClinVar variants and CDS sequence (run locally — NCBI access required):

```bash
# Adapt from examples in examples/ for your gene of interest
python examples/download_gene.py --gene YOUR_GENE --refseq NM_XXXXXX
```

**Step 2.** Compute ESM-1v scores (run locally — requires GPU recommended):

```bash
python examples/run_esm.py --gene YOUR_GENE
```

**Step 3.** Generate tensors and classify (notebooks):

```
notebooks/
├── 01_generate_tensors.ipynb    # CDS + ClinVar + ESM → 11-channel tensors
├── 02_train_and_evaluate.ipynb  # Train CNN with K-fold CV (intra-gene)
└── 03_crossgene_validation.ipynb # Zero-shot transfer from BRCA1
```

### Using the library directly

```python
from energyfingerprint.energy import stacking_profile, STACKING_SANTALUCIA, STACKING_TURNER
from energyfingerprint.genetics import load_fasta, translate_cds, parse_protein_change
from energyfingerprint.model import EnergySignalCNN, GradCAM1D

# Compute thermodynamic profiles
wt_profile = stacking_profile(wt_sequence, STACKING_SANTALUCIA)
mut_profile = stacking_profile(mut_sequence, STACKING_SANTALUCIA)

# Build and run model
model = EnergySignalCNN(n_channels=11, n_classes=2)
logits = model(tensor)  # tensor shape: (batch, 128, 11)

# Interpretability
gradcam = GradCAM1D(model)
heatmap, pred_class, confidence = gradcam(tensor)
```

---

## Repository Structure

```
energyfingerprint/
├── energyfingerprint/           # Core library
│   ├── __init__.py
│   ├── energy.py                # ΔG stacking profiles (SantaLucia + Turner)
│   ├── genetics.py              # Codon table, CDS translation, variant parsing
│   └── model.py                 # EnergySignalCNN, GradCAM1D, training utilities
├── notebooks/                   # Reproducible pipeline
│   ├── 01_generate_tensors.ipynb
│   ├── 02_train_and_evaluate.ipynb
│   └── 03_crossgene_validation.ipynb
├── examples/                    # Example scripts for new genes
│   └── ...
├── environment.yml              # Conda environment
├── CITATION.cff                 # Citation metadata
├── LICENSE                      # MIT License
└── README.md
```

---

## Citation

If you use EnergyFingerprint in your research, please cite:

```bibtex
@article{vilar2026energyfingerprint,
  title={EnergyFingerprint: mRNA Thermodynamic Profiling Combined with Protein
         Language Models Enables Zero-Shot Cross-Gene Missense Variant Classification},
  author={Vilar Sanchez, Jose Antonio},
  year={2026},
  doi={10.5281/zenodo.19784943},
  note={Preprint: \url{https://doi.org/10.5281/zenodo.19831154}}
}
```

---

## License

This code is released under the [MIT License](LICENSE).

The underlying method is protected by patent EnergoRNA P202630522 (OEPM, April 2026). Use of this code for research purposes is permitted under the MIT license. Commercial use of the patented method may require a separate license — contact qmetrika@proton.me.

---

## Author

**Jose Antonio Vilar Sanchez**
Independent Researcher, Madrid, Spain
qmetrika@proton.me
