Lecture exercises and final project for the exam of Deep Learning course.
# Project :
Implementation of a Riemann-Theta Boltzmann Machine in the study of tau lepton polarization.

- Inspired by:
Daniel Krefl, Stefano Carrazza, Babak Haghighat, Jens Kahlen:
 *Riemann-Theta Boltzmann Machine* **[arXiv:1712.07581]**.

 - Classic kinematics approach was developed in my Bachelor thesis:
 [github.com/MicheleScattola/FCCAnalyses.git](https://github.com/MicheleScattola/FCCAnalyses.git)
 - Datasets were generated via:
*MadGraph5_aMC@NLO* **[arXiv:1405.0301]**
*Pythia8* **[arXiv:2203.11601]**
*Delphes* **[arXiv:1307.6346]**
The detector implemented is the Delphes rendition of the IDEA detector proposed to work at the Future-Circular-Collider (FCC-ee). The card used is the configuration from the 2023 winter simulation campaign, available at: [github.com/HEP-FCC/FCC-config/tree/winter2023](https://github.com/HEP-FCC/FCC-config/tree/winter2023)

## Difficulties in classical approach :
A classical approach suffers from evident distorsions in kinematic distributions from real-world detector interactions. This requires some technical work to be able to perform polarization fits, namely:
- Computationally heavy Montecarlo campaign with huge datasets of fully polarized data.
- Apply the reconstruction analysis (includes background).
- Construct histogram templates of the PDFs for the spin-analyzer kinematic variables.
- Fit data with a linear combination of polarized templates.

This technique is highly sensitive to strong background contaminations, and requires strict kinematics cuts.
A Machine-Learning approach could enable the use of a wider parameter space for the event identification and background discrimination.

## Data generated
The folder `/data_generation` contains the *madgraph* scripts for the data generation.

$ee\to Z \to\tau\tau\to \pi^- \nu_\tau \pi^+ \nu_\tau$
- 2 samples of $1\times10^4$ decays for Left-Handed and Right-Handed production of pi decays.

$ee\to Z \to \tau\tau \to \rho \nu_\tau \rho\nu_\tau\to\pi^-\pi^0\nu_\tau\pi^+\pi^0\nu_\tau$
- 2 samples of $5\times10^4$ decays for Left-Handed and Right-Handed production of rho decays.

This is because the performance of the RTBM will be tested at different levels of background (manually introduced), such as rho decays with missing photons that get reconstructed as single pi decays.

In my thesis I obtained a $\sim 70\%$ purity of the pi channel, mainly diluted by rho decays. This can certainly be improved by more kinematics cuts, but I was already imposing no reconstructed photons. This means that a more accurate work on calorimeter towers and hits is needed. For this approach I propose to analyze a 4D paramater space in order to discriminate the background events.

## Data scraping
The folder `/data_scraping` contains the *upROOT* (analogous to pyROOT) scripts for analyzing the simulated data. In standard analysis the pion from tau decays is analyzed by construcing a cone and checking its isolation with respect to other particles or photons. The same analysis is applied to both type of datasets, in order to identify the characteristics of $\rho$ decays with miss-identified photons.

The analysis saves information from the calorimeter towers, in order to explore a 4D parameter space which includes:
- $x_{vis} = E_{track}/E_\tau$ : (where in $e^+e^-$ colliders the energy is fixed by the beam. While operating at the Z pole $E_\tau \simeq 45.5 \text{GeV}$) .
- $E_{cone}$ : the total calorimeter energy inside the reconstruction cone $\Delta R <0.4$ . Rho decays with missing photons should show a total energy in the cone different from the charged track's own energy.
- $f_{had}$ : the energy fraction deposited in the hadronic calorimeter. It is supposed that rho decays with lost photons will have spillage in the electromagnetic calorimeter.
- $\eta_{track}$ : eta of the charged particle track. This is because geometrical detector limitations also impact the possibility of reconstructing the photons.