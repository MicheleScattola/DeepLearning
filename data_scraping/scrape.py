import uproot
import numpy as np
import numba as nb
import os
import time

FOLDER = '/mnt/data/physics_data'
OUT_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'datasets')
os.makedirs(OUT_FOLDER, exist_ok=True)

#@nb.njit(parallel=True)
def process_data(track_pt, track_eta, track_phi, track_mass,
                 photon_pt, photon_eta, photon_phi, 
                 tower_eem, tower_ehad,tower_eta, tower_phi):
    
    '''
    Process data from events.
    Store up to 2 tracks per event and 4 kinematic informations:
    x = E_vis / E_tau        [visible energy fraction, in (0,1)]
    E_cone (DeltaR = 0.4)    [energy inside a given reconstruction cone]
    f_had                    [fraction of energy deposited in hadronic calorimeters, in (0,1)]
    '''
    
    num_events = len(track_pt)
    
    output_matrix = np.zeros((num_events * 2, 4), dtype=np.float32)
    idx_counter = 0
    
    for i in range(num_events):
        ev_track_pt = track_pt[i]
        num_tracks = len(ev_track_pt)
        
        # photon arrays
        ev_photon_pt = photon_pt[i]
        ev_photon_eta = photon_eta[i]
        ev_photon_phi = photon_phi[i]
        
        for t in range(num_tracks):
            t_pt = ev_track_pt[t]
            t_eta = track_eta[i][t]
            t_phi = track_phi[i][t]
            t_mass = track_mass[i][t]
            
            if abs(t_eta) > 2.5:
                continue
                
            # check for isolated photons in a cone near track
            local_photon_count = 0
            for p in range(len(ev_photon_pt)):
                dp_eta = ev_photon_eta[p] - t_eta
                dp_phi = ev_photon_phi[p] - t_phi
                
                if dp_phi > np.pi: dp_phi -= 2 * np.pi
                if dp_phi < -np.pi: dp_phi += 2 * np.pi
                dr_photon = np.sqrt(dp_eta**2 + dp_phi**2)
                
                if dr_photon < 0.4:
                    local_photon_count += 1
            if local_photon_count != 0:
                continue
            
            # 4D parameter space
            t_p = t_pt * np.cosh(t_eta)
            x_vis = t_p / 45.6
            
            # Construct track P4 vector and calculate track energy
            e_track = np.sqrt((t_p)**2 + t_mass**2)
            
            total_ecal = 0.0
            total_hcal = 0.0
            
            ev_tw_eem = tower_eem[i]
            ev_tw_ehad = tower_ehad[i]
            ev_tw_eta = tower_eta[i]
            ev_tw_phi = tower_phi[i]
            
            # sum in calorimeter towers inside Delta R < 0.4
            for j in range(len(ev_tw_eem)):
                deta = ev_tw_eta[j] - t_eta
                dphi = ev_tw_phi[j] - t_phi
                if dphi > np.pi: dphi -= 2 * np.pi
                if dphi < -np.pi: dphi += 2 * np.pi
                dr_tower = np.sqrt(deta**2 + dphi**2)
                
                if dr_tower < 0.4:
                    total_ecal += ev_tw_eem[j]
                    total_hcal += ev_tw_ehad[j]
                     
            e_cone = total_ecal + total_hcal
            had_fraction = (total_hcal / e_cone) if e_cone > 0.0 else 0.0
            e_frac = (e_track / e_cone) if e_cone > 0.0 else 0.0
            
            # Save into the matrix
            slot = idx_counter
            output_matrix[slot, 0] = x_vis
            output_matrix[slot, 1] = e_frac 
            output_matrix[slot, 2] = had_fraction
            output_matrix[slot, 3] = t_eta
            idx_counter += 1

    return output_matrix[:idx_counter]


def extract_dataset(root_path, save_name):
    '''
    return dataset information into numpy arrays
    '''
    start = time.time()

    file = uproot.open(root_path)
    tree = file["Delphes;1"]
    nentries = tree.num_entries
    print(f"[INFO] Opening file: {root_path} with {nentries} entries")

    
    #print("Loading branches into numpy arrays...")
    track_pt  = tree["Track.PT"].array(library="np")
    track_eta = tree["Track.Eta"].array(library="np")
    track_phi = tree["Track.Phi"].array(library="np")
    track_mass = tree["Track.Mass"].array(library="np")
    
    tower_eem  = tree["Tower.Eem"].array(library="np")
    tower_ehad = tree["Tower.Ehad"].array(library="np")
    tower_eta  = tree["Tower.Eta"].array(library="np")
    tower_phi  = tree["Tower.Phi"].array(library="np")
    
    photon_pt  = tree["Photon.PT"].array(library="np")
    photon_eta = tree["Photon.Eta"].array(library="np")
    photon_phi = tree["Photon.Phi"].array(library="np")
    
    raw_matrix = process_data(track_pt, track_eta, track_phi, track_mass,
                              photon_pt, photon_eta, photon_phi,
                              tower_eem, tower_ehad, tower_eta, tower_phi)
    
    # filter empty rows
    # mask = (raw_matrix[:, 0] > 0)
    # final_matrix = raw_matrix[mask]
    percentage = len(raw_matrix)/(2*nentries) * 100
    print(f"[INFO] Extracted {len(raw_matrix)} events with isolated pion, {percentage:.0f}% of the total pions.")
    np.save(save_name, raw_matrix)
    print(f"[INFO] Saved dataset as {save_name}")

    end = time.time()

    print(f'[INFO] Process took {end-start:.2f} seconds\n')


# MAIN
if __name__ == "__main__":
    
    extract_dataset(os.path.join(FOLDER,'pi_LH.root'), os.path.join(OUT_FOLDER,'pi_LH.npy'))
    extract_dataset(os.path.join(FOLDER,'pi_RH.root'), os.path.join(OUT_FOLDER,'pi_RH.npy'))
    extract_dataset(os.path.join(FOLDER,'rho_LH.root'), os.path.join(OUT_FOLDER,'rho_LH.npy'))
    extract_dataset(os.path.join(FOLDER,'rho_RH.root'), os.path.join(OUT_FOLDER,'rho_RH.npy'))