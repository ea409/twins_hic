import torch
from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes

class SequentialSubsetSampler(Sampler):
    """Samples elements sequentially given a list of indices,

    Arguments:
        indices (sequence): a sequence of indices
        
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)

class WeightedSubsetSampler(Sampler):
    """Samples elements randomly given a weighting, from a defined subset.
    Arguments:
        weights (sequence): a sequence of weights, not necessary summing up to one
        indices (sequence): a sequence of indices
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """
    def __init__(self, weights, indices, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.indices=indices
        self.replacement = replacement

    def __iter__(self):
        return iter(self.indices[torch.multinomial(self.weights[self.indices], self.num_samples, self.replacement)])

    def __len__(self):
        return self.num_samples