# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Utilies for interacting with the LIGO Algorithm Library.

This module provides conversions between

- LAL type codes/names and `numpy` types
- `lal.Unit` and `astropy.units.Unit`
- GPS representations to `lal.LIGOTimeGPS` (one-way only)
"""

from __future__ import absolute_import

from six import string_types

import numpy

from astropy import units

import lal

from ..time import to_gps
from ..detector import units as gunits  # pylint: disable=unused-import

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# -- map LAL types ------------------------------------------------------------

# LAL type enum
LAL_TYPE_STR = {
    lal.I2_TYPE_CODE: 'INT2',
    lal.I4_TYPE_CODE: 'INT4',
    lal.I8_TYPE_CODE: 'INT8',
    lal.U2_TYPE_CODE: 'UINT2',
    lal.U4_TYPE_CODE: 'UINT4',
    lal.U8_TYPE_CODE: 'UINT8',
    lal.S_TYPE_CODE: 'REAL4',
    lal.D_TYPE_CODE: 'REAL8',
    lal.C_TYPE_CODE: 'COMPLEX8',
    lal.Z_TYPE_CODE: 'COMPLEX16',
}
LAL_TYPE_FROM_STR = dict((v, k) for k, v in LAL_TYPE_STR.items())

LAL_TYPE_FROM_NUMPY = {
    numpy.int16: lal.I2_TYPE_CODE,
    numpy.int32: lal.I4_TYPE_CODE,
    numpy.int64: lal.I8_TYPE_CODE,
    numpy.uint16: lal.U2_TYPE_CODE,
    numpy.uint32: lal.U4_TYPE_CODE,
    numpy.uint64: lal.U8_TYPE_CODE,
    numpy.float32: lal.S_TYPE_CODE,
    numpy.float64: lal.D_TYPE_CODE,
    numpy.complex64: lal.C_TYPE_CODE,
    numpy.complex128: lal.Z_TYPE_CODE,
}
LAL_TYPE_STR_FROM_NUMPY = dict(
    (key, LAL_TYPE_STR[value]) for (key, value) in LAL_TYPE_FROM_NUMPY.items())


def to_lal_type(dtype):
    """Return the LAL type code for the given python type

    Parameters
    ----------
    dtype : `str`, `type`, `numpy.dtype`
        the input python type

    Returns
    -------
    laltypecode : `int`
        the LAL type code for the input type

    Raises
    ------
    TypeError
        if the input python type cannot be converted

    Examples
    --------
    >>> to_lal_type('float')
    11
    >>> to_lal_type('c8')
    12
    """
    dtype = numpy.dtype(dtype).type
    try:
        return LAL_TYPE_FROM_NUMPY[dtype]
    except KeyError:
        raise TypeError("Cannot map %s to LAL type code" % dtype)


def to_lal_typestr(dtype):
    """Return the LAL type name for the given python type

    Parameters
    ----------
    dtype : `str`, `type`, `numpy.dtype`
        the input python type

    Returns
    -------
    laltypestr : `str`
        the LAL type name for the input type

    Raises
    ------
    TypeError
        if the input python type cannot be converted

    Examples
    --------
    >>> to_lal_type('float')
    'REAL8'
    >>> to_lal_type('c8')
    'COMPLEX8'
    """
    return LAL_TYPE_STR[to_lal_type(dtype)]


# -- units --------------------------------------------------------------------

LAL_UNITS = [
    lal.MeterUnit,
    lal.KiloGramUnit,
    lal.SecondUnit,
    lal.AmpereUnit,
    lal.KelvinUnit,
    lal.StrainUnit,
    lal.ADCCountUnit,
]
LAL_UNIT_FROM_ASTROPY = dict((units.Unit(str(u)), u) for u in LAL_UNITS)


def to_lal_unit(aunit):
    """Convert the input unit into a `LALUnit`

    Parameters
    ----------
    aunit : `~astropy.units.Unit`, `str`
        the input unit

    Returns
    -------
    unit : `lal.Unit`
        the LAL representation of the input

    Raises
    ------
    ValueError
        if LAL doesn't understand the base units for the input

    Examples
    --------
    >>> to_lal_unit('m**2 / kg ** 4')
    m^2 kg^-4
    """
    if isinstance(aunit, string_types):
        aunit = units.Unit(aunit)
    aunit = aunit.decompose()
    lunit = lal.Unit()
    for base, power in zip(aunit.bases, aunit.powers):
        # try this base
        try:
            lalbase = LAL_UNIT_FROM_ASTROPY[base]
        except KeyError:
            lalbase = None
            # otherwise loop through the equivalent bases
            for eqbase in base.find_equivalent_units():
                try:
                    lalbase = LAL_UNIT_FROM_ASTROPY[eqbase]
                except KeyError:
                    continue
        # if we didn't find anything, raise an exception
        if lalbase is None:
            raise ValueError("LAL has no unit corresponding to %r" % base)
        lunit *= lalbase ** power
    return lunit


def from_lal_unit(lunit):
    """Convert a `lal.Unit` into a `~astropy.units.Unit`

    Parameters
    ----------
    lunit : `lal.Unit`
        the input unit

    Returns
    -------
    unit : `~astropy.units.Unit`
        the Astropy representation of the input

    Raises
    ------
    ValueError
        if Astropy doesn't understand the base units for the input

    Examples
    --------
    >>> u = to_lal_unit('m**2 / kg ** 4')
    >>> from_lal_unit(u)
    Unit("m2 / kg4")
    """
    try:
        lunit = lal.Unit(lunit)
    except RuntimeError:
        raise ValueError("Cannot convert %r to lal.Unit" % lunit)
    aunit = units.Unit("")
    for power, lalbase in zip(lunit.unitNumerator, LAL_UNITS):
        # if not used, continue
        if not power:
            continue
        # convert to astropy unit
        try:
            newu = units.Unit(str(lalbase))
        except ValueError:
            raise ValueError("Astropy has no unit corresponding to %r"
                             % lalbase)
        aunit *= newu ** power
    return aunit


# -- LIGOTimeGPS --------------------------------------------------------------

def to_lal_ligotimegps(gps):
    """Convert the given GPS time to a `lal.LIGOTimeGPS` object

    Parameters
    ----------
    gps : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        input GPS time, can be anything parsable by :meth:`~gwpy.time.to_gps`

    Returns
    -------
    ligotimegps : `lal.LIGOTimeGPS`
        a SWIG-LAL `~lal.LIGOTimeGPS` representation of the given GPS time

    Examples
    --------
    >>> to_lal_ligotimegps(1)
    1.000000000
    """
    if isinstance(gps, lal.LIGOTimeGPS):
        return gps
    gps = to_gps(gps)
    return lal.LIGOTimeGPS(gps.gpsSeconds, gps.gpsNanoSeconds)
