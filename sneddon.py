from math import sqrt, pi


def cod_exact(data, pts):
    '''
    Compute the crack opening displacements (COD) at a given set of
    points for Sneddon's problem based on data.

    Parameters
    ----------
    data : dict{'l0', 'G_c', 'E', 'nu_s', 'p'}
        Physical parameters determining Sneddon's test.
    pts : list
        List of points where to compute the COD.
    Returns
    -------
    total crack volume : float
    '''
    cod = []

    E = data['E']
    nu_s = data['nu_s']
    p = data['p']
    l0 = data['l0']

    for x in pts:
        if sqrt(x**2) > l0:
            cod.append((x, 0))
        else:
            _c = 4 * (1 - nu_s**2) * l0 * p / E * sqrt(1 - x**2 / l0**2)
            cod.append((x, _c))

    return cod


def tcv_exact(data):
    '''
    Compute the volume (TCV) for Sneddon's problem based on data.

    Parameters
    ----------
    data : dict{'l0', 'G_c', 'E', 'nu_s', 'p'}
        Physical parameters determining Sneddon's test.

    Returns
    -------
    total crack volume : float
    '''
    vol_ex = (1 - data['nu_s']**2) * data['l0']**2 * data['p'] / data['E']
    vol_ex *= 2 * pi
    return vol_ex
