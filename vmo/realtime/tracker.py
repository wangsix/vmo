import numpy as np
import scipy.spatial.distance as distance

from ..analysis import hmm
from ..VMO.oracle import find_threshold
from ..VMO.oracle import build_oracle


class VMOTracker(object):
    def __init__(self, obs, dim=1, thresh_r=(0, 1.0, 0.001), decay=1.0):
        r = thresh_r
        ideal_x = find_threshold(obs, r=r, dim=dim)
        self.threshold = ideal_x[0][1]
        self.decay = decay
        print(self.threshold)
        self.oracle = build_oracle(obs, threshold=self.threshold, dim=dim)

        self.hmm_tensor = hmm.extract_hmm_tensor(self.oracle, max_lrs=1)
        cluster_means = np.array([np.median(self.oracle.f_array.data[np.array(c), :].T, axis=1)
                                  for c in self.oracle.latent])
        self.cluster_means = cluster_means

        # cluster_means += np.finfo('float').eps
        # self.cluster_means = (cluster_means.T / np.sum(cluster_means, axis=1)).T

        a = self.hmm_tensor[-1]
        a += np.finfo('float').eps
        a += 1.0
        divider = np.sum(a, axis=1)
        a = np.divide(a.T, divider).T
        self.log_a = np.log(a)
        self.v = np.zeros((1, len(self.oracle.latent)))
        self.prev = self.v
        self.current_frame = -1
        self.time_idx = 0

    def track(self, query):
        dist = (2.0 - distance.cdist(self.cluster_means, query.reshape((1, -1)),
                                    metric='cosine')+
                np.finfo('float').eps).squeeze()/2
        if self.time_idx == 0:
            self.v[0] = np.log(dist) + np.log(1.0 / len(self.oracle.latent))
            s = self.v
            current_latent = np.argmax(np.max(s, axis=1))
        else:
            self.prev = self.decay*self.v
            s = self.prev + self.log_a.T
            self.v[0] = np.max(s, axis=1) + np.log(dist)
            # current_latent = np.argmax(s, axis=1)[np.argmax(self.v)]
            current_latent = np.argmax(np.max(s, axis=1))
        if self.time_idx == 0:
            next_frame = self.oracle.latent[current_latent][0]
        else:
            if self.current_frame + 1 in self.oracle.latent[current_latent]:
                next_frame = self.current_frame + 1
            else:
                next_frame = self.oracle.latent[current_latent][
                    np.argmin(np.abs(np.array(self.oracle.latent[current_latent]) - self.current_frame))]
        self.current_frame = next_frame
        self.time_idx += 1
        return next_frame - 1

    def reset(self):

        a = self.hmm_tensor[-1]
        a += np.finfo('float').eps
        a += 1.0
        divider = np.sum(a, axis=1)
        a = np.divide(a.T, divider).T

        self.log_a = np.log(a)
        self.v = np.zeros((1, len(self.oracle.latent)))
        self.prev = self.v
        self.current_frame = -1
