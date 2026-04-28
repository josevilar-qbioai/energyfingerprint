# EnergyFingerprint

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19831154.svg)](https://doi.org/10.5281/zenodo.19831154)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19784943-blue)](https://doi.org/10.5281/zenodo.19784943)

**Perfiles termodinámicos del mRNA + modelos de lenguaje de proteínas para clasificación zero-shot de variantes missense.**

EnergyFingerprint es una CNN 1D ligera (~228K parámetros) que clasifica variantes missense como patogénicas o benignas combinando dos ejes de información ortogonales:

- **Eje biofísico:** Perfiles de energía libre de apilamiento del mRNA (ΔG nearest-neighbor, SantaLucia 1998 + Turner 2004)
- **Eje evolutivo:** Modelo de lenguaje de proteínas ESM-1v (puntuación marginal enmascarada, ratio log-likelihood)

Cada variante se codifica como un **tensor de 11 canales × 128 nucleótidos** y se clasifica mediante una CNN con autoatención multi-cabeza.

> **Patente:** EnergoRNA P202630522 (OEPM, abril 2026). El método está protegido por patente; este código se publica bajo licencia MIT para uso en investigación.

---

## Resultados principales

### Clasificación intra-gen (validación cruzada K-fold)

| Gen | Enfermedad | Familia | AUC |
|------|---------|--------|:---:|
| BRCA1 | Cáncer mama/ovario | Reparación DNA | 0,943 |
| TP53 | Multi-cáncer | Supresor tumoral | 0,907 |
| PTEN | Multi-cáncer | Fosfatasa | 0,994 |
| PALB2 | Cáncer de mama | Reparación DNA | 0,950 |
| CFTR | Fibrosis quística | Transportador ABC | 0,979 |
| HBB | Anemia falciforme | Globina | 0,721 |
| SCN1A | Epilepsia/Dravet | Canal Na⁺ dependiente de voltaje | 0,868 |
| MECP2 | Síndrome de Rett | Regulador transcripcional (IDR) | 0,911 |

### Transferencia zero-shot entre genes (entrenar BRCA1 → predecir otros, sin reentrenamiento)

| Gen objetivo | AUC zero-shot | Tipo de transferencia |
|------------|:------------:|---------------|
| TP53 | 0,854 | Cáncer → Cáncer |
| PTEN | 0,975 | Cáncer → Cáncer |
| PALB2 | 0,954 | Cáncer → Cáncer |
| CFTR | 0,777 | Cáncer → Transportador ABC |
| HBB | 0,707 | Cáncer → Globina |
| SCN1A | 0,637 | Cáncer → Canal TM |
| MECP2 | 0,430 | Cáncer → IDR (**falla**) |

**Gradiente de transferibilidad:** Cáncer (~0,95) > Transportador ABC (~0,78) > Globina (~0,71) > Canal TM (~0,64) > IDR (~0,43)

---

## Arquitectura

```
Entrada: tensor de 11 canales × 128 nucleótidos
  │
  ├─ Canales 1-4: Perfiles de apilamiento ΔG (wt/mut × SantaLucia/Turner)
  ├─ Canales 5-6: Perfiles de diferencia ΔΔG
  ├─ Canal 7:     Contenido GC (ventana deslizante)
  ├─ Canal 8:     Fracción de purinas (ventana deslizante)
  ├─ Canal 9:     Indicador de posición del codón
  ├─ Canal 10:    Reservado (estructura secundaria del mRNA)
  └─ Canal 11:    LLR de ESM-1v × kernel gaussiano
  │
  ▼
  Conv1D(11→64, k=7) → BN → ReLU → Dropout(0,3)
  Conv1D(64→128, k=5) → BN → ReLU → Dropout(0,3)
  Autoatención multi-cabeza (4 cabezas)
  Conv1D(128→64, k=3) → BN → ReLU
  Global Average Pooling
  FC(64→2) → Softmax
  │
  ▼
  Patogénica / Benigna
```

~228K parámetros entrenables. Interpretable mediante GradCAM-1D y ablación de canales.

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/josevilar-qbioai/energyfingerprint.git
cd energyfingerprint

# Crear entorno conda
conda env create -f environment.yml
conda activate energyfp
```

### Requisitos

- Python ≥ 3.10
- PyTorch ≥ 2.1
- fair-esm ≥ 2.0 (para puntuación ESM-1v)
- BioPython, NumPy, Pandas, scikit-learn, matplotlib, seaborn

Consulta `environment.yml` para la lista completa de dependencias.

---

## Uso

### Inicio rápido: clasificar variantes de un nuevo gen

**Paso 1.** Descargar variantes de ClinVar y secuencia CDS (ejecutar en local — requiere acceso a NCBI):

```bash
# Adaptar desde los ejemplos en examples/ para tu gen de interés
python examples/download_gene.py --gene TU_GEN --refseq NM_XXXXXX
```

**Paso 2.** Calcular puntuaciones ESM-1v (ejecutar en local — se recomienda GPU):

```bash
python examples/run_esm.py --gene TU_GEN
```

**Paso 3.** Generar tensores y clasificar (notebooks):

```
notebooks/
├── 01_generate_tensors.ipynb    # CDS + ClinVar + ESM → tensores de 11 canales
├── 02_train_and_evaluate.ipynb  # Entrenar CNN con K-fold CV (intra-gen)
└── 03_crossgene_validation.ipynb # Transferencia zero-shot desde BRCA1
```

### Uso directo de la librería

```python
from energyfingerprint.energy import stacking_profile, STACKING_SANTALUCIA, STACKING_TURNER
from energyfingerprint.genetics import load_fasta, translate_cds, parse_protein_change
from energyfingerprint.model import EnergySignalCNN, GradCAM1D

# Calcular perfiles termodinámicos
perfil_wt = stacking_profile(secuencia_wt, STACKING_SANTALUCIA)
perfil_mut = stacking_profile(secuencia_mut, STACKING_SANTALUCIA)

# Construir y ejecutar el modelo
modelo = EnergySignalCNN(n_channels=11, n_classes=2)
logits = modelo(tensor)  # tensor shape: (batch, 128, 11)

# Interpretabilidad
gradcam = GradCAM1D(modelo)
mapa, clase_pred, confianza = gradcam(tensor)
```

---

## Estructura del repositorio

```
energyfingerprint/
├── energyfingerprint/           # Librería core
│   ├── __init__.py
│   ├── energy.py                # Perfiles de apilamiento ΔG (SantaLucia + Turner)
│   ├── genetics.py              # Tabla de codones, traducción CDS, parsing de variantes
│   └── model.py                 # EnergySignalCNN, GradCAM1D, utilidades de entrenamiento
├── notebooks/                   # Pipeline reproducible
│   ├── 01_generate_tensors.ipynb
│   ├── 02_train_and_evaluate.ipynb
│   └── 03_crossgene_validation.ipynb
├── examples/                    # Scripts de ejemplo para nuevos genes
│   └── ...
├── environment.yml              # Entorno conda
├── CITATION.cff                 # Metadatos de citación
├── LICENSE                      # Licencia MIT
├── README.md                    # Documentación en inglés
└── README_ES.md                 # Esta documentación en español
```

---

## Citación

Si utilizas EnergyFingerprint en tu investigación, por favor cita:

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

## Licencia

Este código se publica bajo la [Licencia MIT](LICENSE).

El método subyacente está protegido por la patente EnergoRNA P202630522 (OEPM, abril 2026). El uso de este código con fines de investigación está permitido bajo la licencia MIT. El uso comercial del método patentado puede requerir una licencia separada — contactar a qmetrika@proton.me.

---

## Autor

**Jose Antonio Vilar Sanchez**
Investigador Independiente, Madrid, España
qmetrika@proton.me
