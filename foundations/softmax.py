import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_num = np.max(z)
        n = z - max_num
        softmax = np.exp(n)/np.sum(np.exp(n))
        return np.round(softmax,4)