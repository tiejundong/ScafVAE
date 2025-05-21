import geatpy as ea
from multiprocessing.pool import ThreadPool
from ScafVAE.model.config import *


class OptProblem(ea.Problem):  # Inherited from Problem class.
    def __init__(
        self,
        name,  # Problem's name.
        base_model,  # base model
        sur_models,  # surrogate model
        maxormins,  # 1 for minimization / -1 for maximization
        Dim=ScafVAE_config.noise_hidden,  # Set the dimension of decision variables.
        bound=3.0,
        multi_cpu=True,
        n_pool=32,
        max_chunk_size=8,
        device='cpu',
    ):
        varTypes = [0] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [-bound] * Dim  # The lower bound of each decision variable.
        ub = [bound] * Dim  # The upper bound of each decision variable.
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [1] * Dim  # Whether the upper boundary is included.

        M = len(sur_models)  # number of tasks

        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        self.base_model = base_model.to(device)
        self.base_model.train(False)
        self.sur_models = sur_models
        self.device = device

        self.multi_cpu = multi_cpu
        if multi_cpu:
            self.pool = ThreadPool(n_pool)
            self.max_chunk_size = max_chunk_size

        self._n_task_ = M

    def aimFunc(self, pop):  # Write the aim function here, pop is an object of Population class.
        Phen = pop.Phen  # Get the decision variables

        repr = to_np(self.base_model.noise2repr(torch.from_numpy(Phen).float().to(self.device)))
        if self.multi_cpu:
            for b in range(1, self.max_chunk_size+1)[::-1]:
                if repr.shape[0] % b == 0:
                    break
            pred = np.concatenate(self.pool.map(self.run_single, repr.reshape(repr.shape[0] // b, b, repr.shape[1])), axis=0)
        else:
            pred = self.run_single(repr)

        pop.ObjV = pred

    def run_single(self, x):
        # x: [1 or b, hidden]
        all_pred = []
        for model in self.sur_models:
            all_pred.append(model.predict(x))
        all_pred = np.stack(all_pred, axis=-1)

        return all_pred