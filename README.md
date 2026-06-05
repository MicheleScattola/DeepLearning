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

## Issues in classical approach :
A classical approach suffers from evident distorsions in kinematic distributions from real-world detector interactions. This requires some technical work to be able to perform polarization fits, namely:
- Computationally heavy Montecarlo campaign with fully polarized data
- construction of polarized histogram templates
- fit data with a linear combination of polarized templates

This technique is highly sensitive to strong background contaminations, and requires strict kinematics cuts.
A Machine-Learning approach could help approach the problems with more ease, and enable the use of a wider parameter space for the event identification.