from models.criterions.CROPStructured import CROPStructured


class CROPStructuredDuring(CROPStructured):
    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNAP-it (before training)
    SNAP-it (before training) provides computational benefits from the start of training
    """

    def __init__(self, *args, limit=0.0, start=0.5, steps=5, **kwargs):
        super(CROPStructuredDuring, self).__init__(*args, **kwargs)
        if limit < 0.5:
            self.steps = [limit - (limit - start) * (0.25 ** i) for i in range(steps + 1)] + [limit]
        else:
            self.steps = [limit - (limit - start) * (0.5 ** i) for i in range(steps + 1)] + [limit]
        self.left = 1.0
        self.pruned = 0.0

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) == 0:
            print("finished all pruning events already")
            return

        # get k_i
        percentage = self.steps.pop(0)
        prune_now = (percentage - self.pruned) / (self.left + 1e-8)

        # prune
        kwargs["percentage"] = prune_now
        StructuredEFG.prune(self, **kwargs)

        # adjust
        self.pruned = self.model.structural_sparsity  # percentage
        self.left = 1.0 - self.pruned
