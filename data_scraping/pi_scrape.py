import ROOT
import os
import sys

FOLDER = '/home/mike/Physics/physics_data'
helicity = ['LH','RH']
FILES = [os.path.join('pi_{h}.root',FOLDER) for h in helicity]
TREE_NAME = 'Delphes'

try:
  ROOT.gROOT.SetBatch(True)
  ROOT.TH1.SetDefaultSumw2(True)
  ROOT.gROOT.SetBatch(True)
  ROOT.EnableImplicitMT()
  ROOT.gSystem.Load("libDelphes.so")

  ROOT.gInterpreter.Declare(r'''
  #include "classes/SortableObject.h"
  #include "classes/DelphesClasses.h"
  #include "ExRootAnalysis/ExRootTreeReader.h"
  ''')
except:
   print('[INFO] ROOT and Delphes import failed.')
   sys.exit(1)
else:
   print('[INFO] ROOT and Delphes loaded correctly.')


def main():

  for FILE in FILES:

    f = ROOT.TFile.Open(FILE)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open input file: {FILE}")
    tree = f.Get(TREE_NAME)
    if not tree:
        raise RuntimeError(f"Could not find tree '{TREE_NAME}'")
    
    # read tree
    reader = ROOT.ExRootTreeReader(tree)
    branch_electron = reader.UseBranch("Electron")
    branch_particle = reader.UseBranch("Particle")
    branch_track = reader.UseBranch("Track")

# MAIN
if __name__ == "__main__":
  main()