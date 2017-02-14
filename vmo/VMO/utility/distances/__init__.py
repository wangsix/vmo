"""vmo/distances/__init__.py
Initializer for the distances module
Overload scipy's cdist function with self-defined distances. 

Copyright (C) 7.28.2014 Cheng-i Wang

This file is part of vmo.

vmo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

vmo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with vmo.  If not, see <http://www.gnu.org/licenses/>.
"""

# import numpy as np
#
# import scipy.spatial.distance as scidist
# import sklearn.preprocessing as preproc
# import vmo.distances.tonnetz
#
# def cdist(XA, XB, dfunc='euclidean', normalize=True, **kwargs):
#     """Compute distances between each pair of the two collections of inputs.
#
#     Overloads the same-named function from scipy with self-defined metrics.
#     The arguments are normalized by default.
#
#     Added keyword arguments:
#         normalize: bool, optional
#            Whether the input arrays should be column-wise normalized
#            (default True).
#
#     Keyword arguments: (taken from the SciPy doc)
#         XA : numpy.ndarray
#             An mA by n array of mA original observations in an
#             n-dimensional space. Inputs are converted to float type.
#         XB : numpy.ndarray
#             An mB by n array of mB original observations in an
#             n-dimensional space. Inputs are converted to float type.
#         dfunc : str or callable, optional
#             (Called `metric` within scipy.)
#             The distance metric to use.
#             See list of scipy supported metrics in the associated doc.
#             If callable, must provide a function of arity 2, e.g.:
#             <lambda u, v: np.sqrt(((u-v)**2).sum()>, for euclidean distance.
#
#     See specification for scipy.spatial.distance.cdist for further details.
#     """
#     if normalize:
#         XA = preproc.normalize(np.array(XA).astype(float))
#         XB = preproc.normalize(np.array(XB).astype(float))
#
#     if dfunc == 'tonnetz':
#         return scidist.cdist(XA, XB, metric=tonnetz.distance, **kwargs)
#     else:
#         return scidist.cdist(XA, XB, metric=dfunc, **kwargs)
