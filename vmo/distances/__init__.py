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
import tonnetz

def cdist(XA, XB, metric='euclidean', **kwargs):
    """Compute distance between each pair of the two collections of inputs.

    Overloads the same-named function from scipy with self-defined metrics.
    See specification for scipy.spatial.distance.cdist for further details.
    """
    if metric is 'tonnetz':
        return scidist.cdist(XA, XB, tonnetz.distance, **kwargs)
    else:
        return scidist.cdist(XA, XB, metric, **kwargs)
