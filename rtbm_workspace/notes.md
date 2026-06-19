# Perform training

## Stand-alone 'train_rtbm'

```python
def train_rtbm():
  # declare RTMB model and initialize with param bounds
  from theta.rtbm import RTBM
  model = RTBM(N_VISIBLE, n_hidden,
               init_max_param_bound=pb, random_bound=1)

  # define initial estimate
  from cma import CMAEvolutionStrategy
  # https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html

  # def: CMAEvolutionStrategy(x0, sigma0, opts)
  cma_opts = {
        'bounds':   model.get_bounds(),
        'tolfun':   tolfun,
        'maxiter':  maxiter,
        'verb_log': 0,
    }
    evolution = CMAEvolutionStrategy(initsol, sigma, cma_opts)

    # consider only multithreading case
    # initialize worker
    from theta.minimizer import worker_initialize, worker_compute
    # where the initialization reserves a container for the load:
    # resource = Resource(cost, model, x_data, y_data)

    # the training run untils a call "evolution.stop()"
    


```