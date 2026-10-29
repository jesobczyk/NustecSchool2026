# NuSTEC School 2026 — neutrino interactions in the quasielastic region

Interactive notebooks for the lectures of **Joanna E. Sobczyk** (Chalmers University of Technology) at the NuSTEC School 2026, Fermilab.

Each notebook is short: text, a few sliders and live figures. All the physics code lives in the `.py` files next to it, so you can read it whenever you want to see how a figure is made.

| Lecture | Notebook | Open |
|---|---|---|
| 1 | The quasielastic peak, the Fermi gas and the mean field | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GHUSER/NustecSchool2026/blob/main/L1_FermiGas/L1_QE_FermiGas_MeanField.ipynb) |
| 2 | From the local Fermi gas to orbitals: what (e,e′p) sees | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GHUSER/NustecSchool2026/blob/main/L2_Orbitals/L2_eep_Orbitals_SpectralFunction.ipynb) |
| 3 | Your own Hartree–Fock solver | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GHUSER/NustecSchool2026/blob/main/L3_HartreeFock/L3_SkyrmeHF.ipynb) |

## Running on Google Colab (recommended)

1. Click the **Open in Colab** badge. You need a Google account.
2. Colab warns that the notebook was not written by Google: click **Run anyway**.
3. Run the first cell (▶). It downloads this repository into your Colab session, which takes a few seconds.
4. Then use **Runtime → Run all**, or run the cells one by one.
5. To keep your changes, use **File → Save a copy in Drive**. Otherwise they are lost when you close the tab.

If the session disconnects (for example after a long break), just run the first cell again.

## Running on your own computer

```bash
git clone https://github.com/GHUSER/NustecSchool2026.git
cd NustecSchool2026
pip install -r requirements.txt jupyterlab
jupyter lab
```

If the sliders do not appear, run `demo.USE_WIDGETS = False` (Lectures 1–2) or `L.USE_WIDGETS = False` (Lecture 3) after the import cell: every panel is then shown as a set of static figures.

## Data

- Inclusive ¹²C(e,e′) cross sections: the Quasielastic Electron Nucleus Scattering Archive (O. Benhar, D. Day, I. Sick), https://discovery.phys.virginia.edu/research/groups/qes-archive/
- ¹²C(e,e′p): J. Mougey *et al.*, Nucl. Phys. A 262 (1976) 461; D. Dutta *et al.*, Phys. Rev. C 68 (2003) 064603 (digitised from the published figures).
- Rosenbluth-separated responses of ¹²C and ⁴⁰Ca: Jourdan, Nucl. Phys. A 603 (1996) 117; Williamson *et al.*, Phys. Rev. C 56 (1997) 3152; Meziani *et al.*, Phys. Rev. Lett. 52 (1984) 2130.

Please cite the original papers if you use these data.
