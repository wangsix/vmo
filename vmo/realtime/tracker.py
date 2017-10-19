import numpy as np

from ..analysis import hmm
from ..VMO.oracle import find_threshold
from ..VMO.oracle import build_oracle


class VMOTracker(object):
    def __init__(self, obs, dim=1):
        r = (0, 0.1, 0.0001)
        ideal_x = find_threshold(obs, r=r, dim=dim)
        self.threshold = ideal_x[0][1]
        print(self.threshold)
        self.oracle = build_oracle(obs, threshold=self.threshold, dim=dim)

        self.hmm_tensor = hmm.extract_hmm_tensor(self.oracle)
        cluster_means = np.array([np.median(self.oracle.f_array.data[np.array(c), :].T, axis=1)
                                  for c in self.oracle.latent])

        cluster_means += np.finfo('float').eps
        self.cluster_means = (cluster_means.T / np.sum(cluster_means, axis=1)).T

        a = self.hmm_tensor[-1]
        a += np.finfo('float').eps
        a += 1.0
        divider = np.sum(a, axis=1)
        a = np.divide(a.T, divider).T
        self.log_a = np.log(a)
        self.v = np.zeros((len(self.oracle.latent)))
        self.prev = self.v
        self.current_frame = -1

    def track(self, query):
        if self.current_frame == -1:
            self.v = np.log(np.dot(self.cluster_means, query)) + np.log(1.0 / len(self.oracle.latent))
        else:
            self.prev = self.v
            s = self.prev + self.log_a.T
            self.v = np.max(s, axis=1) + np.log(np.dot(self.cluster_means, query))

        current_latent = np.argmax(s, axis=1)[np.argmax(self.v)]
        if self.current_frame == -1:
            next_frame = self.oracle.latent[current_latent][0]
        else:
            if self.current_frame + 1 in self.oracle.latent[current_latent]:
                next_frame = self.current_frame + 1
            else:
                next_frame = self.oracle.latent[current_latent][
                    np.argmin(np.abs(np.array(self.oracle.latent[current_latent] - self.current_frame)))]
        self.current_frame = next_frame
        return next_frame - 1
