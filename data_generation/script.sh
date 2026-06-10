#!/bin/bash

# Stop on error
set -e

echo ">>> Copying analysis files"
cp /home/mike/Physics/physics_data/pi_LH/Events/LH/tag_1_delphes_events.root /home/mike/Physics/physics_data/pi_LH.root
cp /home/mike/Physics/physics_data/pi_RH/Events/RH/tag_1_delphes_events.root /home/mike/Physics/physics_data/pi_RH.root
cp /home/mike/Physics/physics_data/rho_LH/Events/LH/tag_1_delphes_events.root /home/mike/Physics/physics_data/rho_LH.root
cp /home/mike/Physics/physics_data/rho_RH/Events/RH/tag_1_delphes_events.root /home/mike/Physics/physics_data/rho_RH.root