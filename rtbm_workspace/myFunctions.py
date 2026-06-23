import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

class NegativeLogLikelihood:
    '''
    The RTM runs in Mode.LogProbability() , therefore the minimization 
    has to act on the negative log-likelihood.
    A Manager list is used to retrieve the history.
    '''
    def __init__(self, shared_history):
        self.shared_history = shared_history

    def cost(self, predictions, y_data=None):
        loss_val = -np.mean(predictions)
        self.shared_history.append(loss_val)
        return loss_val
    

def progress_monitor(shared_list, total_evals, length=40):
  """
  Background thread to monitor and print a real-time progress bar.
  It reads the length of the shared_history list updated by the parallel workers.
  """
  while True:
      current = len(shared_list)
      current_clamped = min(current, total_evals) # Prevent over-100% display
      
      percent = ("{0:.1f}").format(100 * (current_clamped / float(total_evals)))
      filled = int(length * current_clamped // total_evals)
      bar = '█' * filled + '-' * (length - filled)
      
      # \r forces the terminal to overwrite the current line
      sys.stdout.write(f'\r[TRAINING] |{bar}| {percent}% ({current_clamped}/{total_evals} evals)')
      sys.stdout.flush()
      
      if current >= total_evals:
          sys.stdout.write('\n')
          break
          
      time.sleep(0.5)