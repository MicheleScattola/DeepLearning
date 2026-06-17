# Terminal output
```
[INFO] Opening file: /mnt/data/physics_data/pi_LH.root with 100000 entries
[INFO] Extracted 197599 events with isolated pion, 99% of the total pions.
[INFO] Saved dataset as ../datasets/pi_LH.npy
[INFO] Process took 6.27 seconds

[INFO] Opening file: /mnt/data/physics_data/pi_RH.root with 100000 entries
[INFO] Extracted 197742 events with isolated pion, 99% of the total pions.
[INFO] Saved dataset as ../datasets/pi_RH.npy
[INFO] Process took 6.14 seconds

[INFO] Opening file: /mnt/data/physics_data/rho_LH.root with 500000 entries
[INFO] Extracted 109864 events with isolated pion, 11% of the total pions.
[INFO] Saved dataset as ../datasets/rho_LH.npy
[INFO] Process took 29.81 seconds

[INFO] Opening file: /mnt/data/physics_data/rho_RH.root with 500000 entries
[INFO] Extracted 150390 events with isolated pion, 15% of the total pions.
[INFO] Saved dataset as ../datasets/rho_RH.npy
[INFO] Process took 31.85 seconds
```

## Phase Space Selection and Unsupervised Feature Engineering

To successfully train unsupervised machine learning models (Autoencoder and RTBM) to distinguish $\tau$ polarization states and reject anomalous background decays ($\tau \to \rho\nu \to \pi^\pm \pi^0 \nu$), the raw detector data must be mapped into a well-bounded, physically meaningful phase space. 

We define a 4-dimensional feature vector `[x_vis, I_gamma, f_had, eta]` for each event. The variables and their extraction methodologies were chosen specifically to bypass high-level reconstruction biases and leverage raw Particle-Flow sub-detector data.

### 1. The Physics Variables

* **Visible Energy Fraction ($x_{vis}$):** Calculated as $E_{track} / E_{\tau}$. This variable maps the spin-dependent energy sharing between the visible pion and the invisible neutrino, serving as the primary signature for Left-Handed vs. Right-Handed polarization discrimination.
* **Neutral Track Isolation ($I_\gamma$):** Calculated as $\sum E_{cell} / E_{track}$ for neutral electromagnetic cells within a cone of $\Delta R < 0.4$. This ratio explicitly identifies the presence of a decaying neutral pion ($\pi^0 \to \gamma\gamma$), serving as the primary anomaly trigger for $\rho$ background events.
* **Hadronic Core Fraction ($f_{had}$):** Calculated as $E_{HCAL} / (E_{HCAL} + E_{ECAL})$ strictly within the core track footprint ($\Delta R < 0.2$). This acts as a localized Particle ID (PID) to confirm the core track is a purely hadronic pion, decoupled from the broader $0.4$ isolation cone.
* **Pseudorapidity ($\eta$):** The geometric trajectory of the track. It allows the models to learn the spatial acceptance and resolution boundaries of the IDEA detector barrel and endcaps. 

### 2. Data Extraction Methodology and Detector Realities

Extracting these features natively from the Delphes ROOT trees required navigating several distinct detector reconstruction effects to ensure stable machine learning convergence:

**Mitigating Tracking Artifacts via Particle-Flow**
Raw tracking branches occasionally produce severe momentum mismeasurements ($p_T \to \infty$) due to minimal spatial resolution errors simulating highly straight tracks. To prevent these $1000+$ GeV anomalies from destroying the bounded phase space, the primary track kinematics are extracted exclusively from the `EFlowTrack` branch. This ensures the tracker data has been successfully cross-referenced and momentum-conserved against the calorimeters by the Particle-Flow algorithm.

**Bypassing Software Isolation via Raw Calorimetry**
High-level reconstructed `Photon` branches actively reject electromagnetic deposits that land too close to a charged track, artificially blinding the dataset to boosted $\rho$ decays where the $\gamma\gamma$ pair overlaps the $\pi^\pm$ track. To recover this crucial background signature, the isolation variable $I_\gamma$ is calculated using the raw `EFlowPhoton` tower elements. By measuring the raw cell energy rather than software-flagged objects, the hidden $\rho$ decay topologies are exposed to the unsupervised models.

**Applying Realistic Hardware Thresholds**
Because the `EFlowPhoton` branch represents raw calorimeter cells, it contains baseline electronic noise and charged-track ionization leakage. To prevent this noise from smearing the signal peak at $I_\gamma = 0$, a realistic detector sensitivity threshold of **$0.5$ GeV** is applied during extraction. Cells below this limit are discarded, resulting in a pristine dataset where true pions peak exactly at zero, while soft background photons from asymmetric $\pi^0$ decays are fully captured.