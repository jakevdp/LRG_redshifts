import numpy as np

def interpolate_with_error(x, y, dy, x_new, fill_value=np.nan):
    """Linear Interpolation, keeping track of error propagation

    Parameters
    ----------
    x, y, dy : arrays
        locations, values, and errors of input points
    x_new : array
        locations of points to be interpolated
    fill_value : array
        value to use outside the domain of x

    Returns
    -------
    y_new, dy_new : arrays
        values and errors at the interpolated locations
    """
    out_of_bounds = (x_new < x[0]) | (x_new > x[-1])
    i_x_new = np.searchsorted(x, x_new)
    i_x_new = i_x_new.clip(1, len(x) - 1).astype(int)
    lo = i_x_new - 1
    hi = i_x_new

    x_lo = x[lo]
    x_hi = x[hi]
    y_lo = y[lo]
    y_hi = y[hi]
    dy_lo = dy[lo]
    dy_hi = dy[hi]

    fact = (x_new - x_lo) / (x_hi - x_lo)

    y_new = y_lo + (y_hi - y_lo) * fact    
    dy_new = np.sqrt(dy_lo ** 2 + (dy_hi ** 2 + dy_lo ** 2) * abs(fact))

    y_new[out_of_bounds] = 0
    dy_new[out_of_bounds] = np.inf

    return y_new, dy_new
