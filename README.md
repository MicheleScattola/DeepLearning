Lecture exercises and final project for the Deep Learning exam.
# Project :
Implementation of a Riemann-Theta Boltzmann Machine in the study of tau lepton polarization.

- Inspired by:
Daniel Krefl, Stefano Carrazza, Babak Haghighat, Jens Kahlen:
 *Riemann-Theta Boltzmann Machine* **[arXiv:1712.07581]**.
 
  Credits to https://github.com/RiemannAI/theta.git for the library.

 - Simple re-weighting approach was developed in my Bachelor thesis on official FCC data:
 [github.com/MicheleScattola/FCCAnalyses.git](https://github.com/MicheleScattola/FCCAnalyses.git)
 - Datasets for the project were generated via:
  *MadGraph5_aMC@NLO* **[arXiv:1405.0301]**
  *Pythia8* **[arXiv:2203.11601]**
  *Delphes* **[arXiv:1307.6346]**.

  The detector implemented is the Delphes rendition of the IDEA detector proposed to work at FCC-ee. The card used is the configuration from the 2023 winter simulation campaign (same as thesis data), available at: [github.com/HEP-FCC/FCC-config/tree/winter2023](https://github.com/HEP-FCC/FCC-config/tree/winter2023)

## Difficulties in classical approach :
A classical approach suffers from evident distorsions in kinematic distributions from real-world detector interactions. This requires some technical work to be able to perform polarization fits, namely the construction of histogram templates for the PDFs of the spin-analyzer kinematic variables.

This technique is highly sensitive to strong background contaminations, and requires kinematics cuts to achieve decent results.
A Machine-Learning approach could enable the use of a wider parameter space for the event identification and background discrimination.

## Objectives :
The main objective is to estimate the signal to noise ratio in the reconstruction for simple pi decays $\tau \to \pi \nu$ which suffer from $\tau \to \rho \nu \to \pi \pi^0 \nu$ contamination with lost $\pi^0 \to \gamma\gamma$.

In my thesis I obtained a $\sim 70\%$ purity of the pion channel in reconstruction. The objective is to quantify the performance increase of a ML-based discriminator with respect to background rejection. This can be implemented in two methods which I wish to compare:

- Autoencoder trained on $\pi$ decays in order to implement an anomaly detection.
- RTBM classifier which should be able to learn the PDFs of the pion phase space and flag based on the log-likelihood.



## Data generated
The folder `/data_generation` contains the *madgraph* scripts for the data generation.

$ee\to Z \to\tau\tau\to \pi^- \nu_\tau \pi^+ \nu_\tau$
- 2 samples of $1\times10^4$ decays for Left-Handed and Right-Handed production of pi decays.

$ee\to Z \to \tau\tau \to \rho \nu_\tau \rho\nu_\tau\to\pi^-\pi^0\nu_\tau\pi^+\pi^0\nu_\tau$
- 2 samples of $5\times10^4$ decays for Left-Handed and Right-Handed production of rho decays.


## Data scraping
The folder `/data_scraping` contains the *upROOT* (analogous to pyROOT) scripts for analyzing the simulated data. In standard analysis the pion from tau decays is analyzed by construcing a cone and checking its isolation with respect to other particles or photons.  A similar approach was the one implemented in my thesis, based on simple particle counts. The same analysis is applied to both type of datasets, in order to identify the characteristics of $\rho$ decays with miss-identified photons.

The objective is to feed to the networks more information than raw counts, such as the energy deposited in the calorimeters, which could help map different features in the traces of missing photon events.

The 4D parameter space for the analyzed datasets contains the following variables:
- $x_{vis} = E_{track}/E_\tau$ : (where in $e^+e^-$ colliders the energy is fixed by the beam. While operating at the Z pole $E_\tau \simeq 45.5 \text{GeV}$) .
- $E_{cone}$ : the total calorimeter energy inside the reconstruction cone $\Delta R <0.4$ . Rho decays with missing photons should show a total energy in the cone different from the charged track's own energy.
- $f_{had}$ : the energy fraction deposited in the hadronic calorimeter. It is supposed that rho decays with lost photons will have spillage in the electromagnetic calorimeter.
- $\eta_{track}$ : eta of the charged particle track. This is because geometrical detector limitations also impact the possibility of reconstructing the photons.