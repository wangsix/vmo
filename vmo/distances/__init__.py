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

import scipy.spatial.distance as scidist
import vmo.distances.tonnetz

def cdist(XA, XB, dfunc='euclidean', **kwargs):
    """Compute distance between each pair of the two collections of inputs.

    Overloads the same-named function from scipy with self-defined metrics.
    
    Keyword arguments: (taken from the SciPy doc)
        XA: numpy.ndarray
            An mA by n array of mA original observations in an
            n-dimensional space. Inputs are converted to float type.
        XB : ndarray
            An mB by n array of mB original observations in an
            n-dimensional space. Inputs are converted to float type.
        metric : str or callable, optional
            The distance metric to use.
            See list of scipy supported metrics in the associated doc.
            If callable, must provide a function of arity 2, e.g.:
            <lambda u, v: np.sqrt(((u-v)**2).sum()>, for euclidean distance
    
    See specification for scipy.spatial.distance.cdist for further details.
    """
    if dfunc == 'tonnetz':
        return scidist.cdist(XA, XB, metric=tonnetz.distance, **kwargs)
    else:
        return scidist.cdist(XA, XB, metric=dfunc, **kwargs)
