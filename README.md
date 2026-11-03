# NuSTEC School 2026 — neutrino interactions in the quasielastic region

Interactive notebooks (demos) for the lectures of **Joanna E. Sobczyk** (Chalmers University of Technology) at the NuSTEC School 2026, Fermilab.

---

## ▶ Open the demos in Google Colab — nothing to install

| Lecture | Demo | Click to open |
|:---:|---|:---:|
| **1** | The quasielastic peak: scaling and the Fermi gas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L1_FermiGas/L1_QE_FermiGas_MeanField.ipynb) |
| **2** | From the local Fermi gas to the shell model: (e,e′p) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L2_Orbitals/L2_eep_Orbitals_SpectralFunction.ipynb) |
| **3** | Your own Hartree–Fock solver | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L3_HartreeFock/L3_SkyrmeHF.ipynb) |

Links as plain text (if the buttons above do not show, for example in a downloaded copy):

- Lecture 1: https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L1_FermiGas/L1_QE_FermiGas_MeanField.ipynb
- Lecture 2: https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L2_Orbitals/L2_eep_Orbitals_SpectralFunction.ipynb
- Lecture 3: https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L3_HartreeFock/L3_SkyrmeHF.ipynb

You need a Google account. Colab runs everything in your browser.

---

## How to use Colab

1. Click **Open in Colab** for your lecture.
2. Colab warns that the notebook was not written by Google: click **Run anyway**.
3. Run the first cell (▶). It downloads this repository into your Colab session (a few seconds).
4. Then **Runtime → Run all**, or run the cells one by one, and move the sliders.
5. To keep your changes: **File → Save a copy in Drive**. Otherwise they are lost when you close the tab.

If the session disconnects (for example after a long break), run the first cell again.

## What is in each demo

**Lecture 1 — The quasielastic peak: scaling and the Fermi gas** ([open in Colab](https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L1_FermiGas/L1_QE_FermiGas_MeanField.ipynb), folder `L1_FermiGas/`)  
Scaling and superscaling of inclusive (e,e′) data, fitting the relativistic Fermi gas, the five neutrino responses.  
Panels: **A** Scaling · **B** Superscaling · **C** Fit the Fermi gas · **D** Neutrinos

**Lecture 2 — From the local Fermi gas to the shell model: (e,e′p)** ([open in Colab](https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L2_Orbitals/L2_eep_Orbitals_SpectralFunction.ipynb), folder `L2_Orbitals/`)  
Design an (e,e′p) measurement on argon, compare Woods–Saxon orbitals of carbon with (e,e′p) data, count the protons in the shells, and go back to the inclusive and separated responses.  
Panels: **A** Design a measurement (argon) · **B** The well and the data · **C** Count the protons · **D** Close the loop · **E** Rosenbluth-separated responses

**Lecture 3 — Your own Hartree–Fock solver** ([open in Colab](https://colab.research.google.com/github/jesobczyk/NustecSchool2026/blob/main/L3_HartreeFock/L3_SkyrmeHF.ipynb), folder `L3_HartreeFock/`)  
A Skyrme Hartree–Fock calculation for closed-shell nuclei: from a guessed well to a self-consistent mean field.

Each notebook is short: text, sliders and live figures, questions to try, and the answers at the end. All the physics code is in the `.py` files next to each notebook, so you can read how every figure is made.

## Running on your own computer

```bash
git clone https://github.com/jesobczyk/NustecSchool2026.git
cd NustecSchool2026
```

With **conda** (recommended if you use Anaconda; keep the demos in their own environment):

```bash
conda create -n nustec -c conda-forge python=3.12 numpy scipy matplotlib ipywidgets jupyterlab
conda activate nustec
jupyter lab
```

With **pip** (in a virtual environment):

```bash
python -m venv nustec && source nustec/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab
```

Do not mix pip and conda in the same environment. If the sliders do not appear, run `demo.USE_WIDGETS = False` (Lectures 1–2) or `L.USE_WIDGETS = False` (Lecture 3) after the import cell: every panel is then shown as static figures.

## Data

- Inclusive (e,e′) cross sections of ⁴He, ¹²C, ¹⁶O, ²⁷Al, ⁴⁰Ca, ⁵⁶Fe, ¹⁹⁷Au, ²⁰⁸Pb: the Quasielastic Electron Nucleus Scattering Archive (O. Benhar, D. Day, I. Sick), https://discovery.phys.virginia.edu/research/groups/qes-archive/
- ¹²C(e,e′p): J. Mougey *et al.*, Nucl. Phys. A 262 (1976) 461; D. Dutta *et al.*, Phys. Rev. C 68 (2003) 064603 (digitised from the published figures).
- Argon spectral function (Lecture 2, Panel A): shell energies, widths and strengths from W. Jiang *et al.* (JLab E12-14-012), Phys. Rev. D 107 (2023) 012005.
- Rosenbluth-separated responses of ¹²C: J. Jourdan, Nucl. Phys. A 603 (1996) 117.

Please cite the original papers if you use these data.

## License

MIT (see `LICENSE`).
