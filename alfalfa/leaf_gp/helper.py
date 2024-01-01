def set_rnd_states(args):
    # init random seed and method
    import numpy as np
    import torch
    np.random.seed(args.rnd_seed)
    torch.manual_seed((args.rnd_seed))