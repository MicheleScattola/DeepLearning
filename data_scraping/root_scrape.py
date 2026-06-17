import uproot
import numpy as np
import numba as nb
import os
import time
import ROOT
from ROOT.Math import PtEtaPhiMVector

FOLDER = '/mnt/data/physics_data'
OUT_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'datasets')
os.makedirs(OUT_FOLDER, exist_ok=True)
TREE_NAME = 'Delphes'
CONE = 0.4
E_TAU = 45.6

def deltaR(deta,dphi):
    if dphi > np.pi: dphi -= 2 * np.pi
    if dphi < -np.pi: dphi += 2 * np.pi
    return np.sqrt(deta**2 + dphi**2)


def extract_dataset(INFILE, OUTFILE):
    '''
    return dataset information into numpy arrays
    '''
    start = time.time()

    print(f'[INFO] analyzing file {INFILE}')
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    # load ROOT and Delphes
    ROOT.gROOT.SetBatch(True)
    #ROOT.EnableImplicitMT()
    ROOT.gSystem.Load("libDelphes.so")

    ROOT.gInterpreter.Declare(r'''
    #include "classes/SortableObject.h"
    #include "classes/DelphesClasses.h"
    #include "ExRootAnalysis/ExRootTreeReader.h"
    ''')

    # open file
    f = ROOT.TFile.Open(INFILE)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open input file: {INFILE}")
    tree = f.Get(TREE_NAME)
    if not tree:
        raise RuntimeError(f"Could not find tree '{TREE_NAME}'")

    reader = ROOT.ExRootTreeReader(tree)
    eflow_photons = reader.UseBranch("EFlowPhoton")
    photons = reader.UseBranch("Photon")
    eflow_tracks = reader.UseBranch("EFlowTrack")
    neut_had = reader.UseBranch("EFlowNeutralHadron")
    towers = reader.UseBranch("Tower")

    n_entries = reader.GetEntries()

    x_vis = []
    track_eta = []
    f_had = []
    track_iso = []

    for i in range(n_entries):
        reader.ReadEntry(i)

        for j in range(eflow_tracks.GetEntries()):
            # isolation check
            tr = eflow_tracks.At(j)
            tr_p4 = PtEtaPhiMVector(tr.PT,tr.Eta,tr.Phi,tr.Mass)
            E_ph = 0.0
            ph_counts = 0
            # check reconstructed photons
            for k in range(photons.GetEntries()):
                ph = photons.At(k)
                ph_p4 = PtEtaPhiMVector(ph.PT,ph.Eta,ph.Phi,0.0)
                dR = tr_p4.DeltaR(ph_p4)

                if dR<CONE:
                    ph_counts += 1

            if ph_counts != 0:
                continue

            # EFlowPhotons as calorimeter deposits
            for l in range(eflow_photons.GetEntries()):
                ph = eflow_photons.At(l)
                deta = ph.Eta - tr.Eta
                dphi = ph.Phi - tr.Phi
                dR = deltaR(deta,dphi)
                if dR<CONE and ph.E > 0.5 :
                    E_ph += ph.E


            E_track = tr_p4.E()
            x_vis.append(E_track/E_TAU)
            track_eta.append(tr.Eta)
            track_iso.append(E_ph / E_track)

            # hcal fraction
            total_ecal = 0.0
            total_hcal = 0.0
            for m in range(towers.GetEntries()):
                twr = towers.At(m)
                deta = twr.Eta - tr.Eta
                dphi = twr.Phi - tr.Phi
                dr_tower = deltaR(deta,dphi)
                
                if dr_tower < 0.2:
                    total_ecal += twr.Eem
                    total_hcal += twr.Ehad
            
            e_cone = total_ecal + total_hcal
            f_had_val = (total_hcal / e_cone) if e_cone > 0.0 else 0.0
            f_had.append(f_had_val)

    # convert to numpy
    x_vis_arr = np.array(x_vis, dtype=np.float32)
    track_eta_arr = np.array(track_eta, dtype=np.float32)
    f_had_arr = np.array(f_had, dtype=np.float32)
    track_iso_arr = np.array(track_iso, dtype=np.float32)
    
    raw_matrix = np.column_stack((x_vis_arr, track_iso_arr, f_had_arr, track_eta_arr))
    
    print(f"[INFO] Extracted {len(raw_matrix)} events")
    np.save(OUTFILE, raw_matrix)
    print(f"[INFO] Saved dataset as {OUTFILE}")

    end = time.time()

    print(f'[INFO] Process took {end-start:.2f} seconds\n')


# MAIN
if __name__ == "__main__":
    
    extract_dataset(os.path.join(FOLDER,'pi_LH.root'), os.path.join(OUT_FOLDER,'pi_LH.npy'))
    extract_dataset(os.path.join(FOLDER,'pi_RH.root'), os.path.join(OUT_FOLDER,'pi_RH.npy'))
    extract_dataset(os.path.join(FOLDER,'rho_LH.root'), os.path.join(OUT_FOLDER,'rho_LH.npy'))
    extract_dataset(os.path.join(FOLDER,'rho_RH.root'), os.path.join(OUT_FOLDER,'rho_RH.npy'))