# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Implementation of second-order-section filtering
"""

import numpy

from scipy import signal

from astropy.units import (Unit, Hertz, radian, second)

from .. import version

__author__ = 'Max Isi <max.isi@ligo.org>'
__credits__ = 'Matt Evans'
__version__ = version.version

MINUSTWOPI = -2 * numpy.pi


class SOSFilter(signal.lti):
    """Filter with application by second-order section.

    Parameters
    ----------
    *args :
        ``args`` should be given as:

        - (numerator, denominator)
        - (zeros, poles, gain)
        - (A, B, C, D) : state-space.
    unit : `str`, `~astropy.units.Unit`, optional
        unit for critical filter components, default: 'Hertz'.
        Must be one of 'Hertz' or 'radian/second'.

    Examples
    --------
    For example, for a ZPK-format with two zeros at 100 Hz, and two
    poles at 1 Hz:

        >>> SOSFilter([100, 100], [1, 1], 1)

    Raises
    ------
    ValueError
        if ``unit`` is given something not comparable to Hz or rad/s
    """
    def __init__(self, *args, **kwargs):
        """Create a new `SOSFilter` from component definitions.
        """
        unit = Unit(kwargs.pop('unit', 'Hz'))
        if kwargs:
            raise TypeError("Invalid keyword argument for SOSFilter: %r"
                            % kwargs.keys()[0])
        # if Hertz, simply initialise
        if unit == Hertz:
            super(SOSFilter, self).__init__(*args)
        # if rad/s convert zpk into Hertz on init
        elif unit == radian/second:
            # assuming zeros, poles and gain given in rad/s
            lti_ = signal.lti(*args)
            z = lti_.zeros
            p = lti_.poles
            k = lti_.gain
            super(SOSFilter, self).__init__(
                z / MINUSTWOPI,
                p / MINUSTWOPI,
                k / (2 * numpy.pi)**(len(p) - len(z)))
        # otherwise fail
        else:
            raise ValueError("Invalid unit '%s', please give one of 'Hertz' "
                             "or 'radian/second'" % unit)
        self.zeros_s = MINUSTWOPI * self.zeros
        self.poles_s = MINUSTWOPI * self.poles
        self.gain_s = (
            (2 * numpy.pi)**(len(self.poles) - len(self.zeros)) * self.gain)

    def sresp(self, f):
        """Returns the S-domain frequency response of LTI at f (in Hz)

        Parameters
        ----------
        f : `float`, `numpy.ndarray`, `list`
            array of frequencies or single frequency in Hz

        Returns
        -------
        h : `numpy.ndarray` of `complex`
            S-domain frequency response of LTI at f (in Hz)

        Notes
        -----
        Matt Evans's Matlab code: sresp.m
        """
        # compute frequency response with flipped signs
        _, h = signal.freqresp((-self.zeros, -self.poles, self.gain),
                               w=numpy.atleast_1d(f))

        # check for infinities
        if len(self.poles) > len(self.zeros):
            h[numpy.where(f == numpy.Inf)] = 0
        elif len(self.poles) < len(self.zeros):
            h[numpy.where(f == numpy.Inf)] = numpy.Inf
        else:
            h[numpy.where(f == numpy.Inf)] = self.gain

        return h

    def get_zpk_z(self, fs, fp=None):
        """Get Z-domain zeros and poles

        Parameters
        ----------
        fs : `float`, `numpy.ndarray`
            sample frequency in Hz
        fp : `float`
            pre-warping match frequency (Hz)

        Returns
        -------
        zz : `numpy.ndarray`
            zeros in Z-domain.
        pz : `numpy.ndarray`
            poles in Z-domain.
        kz : `numpy.ndarray`
            gain in Z-domain.

        Notes
        -----
        Matt Evans' Matlab code: getZPKz.m
        """
        zs = self.zeros_s
        ps = self.poles_s
        ks = self.gain_s

        if fp is None:
            # transform zeros and poles
            ws = MINUSTWOPI * fs
            zz = ztransform(zs / ws)
            pz = ztransform(ps / ws)

            # add extra zeros at -1
            zz = numpy.append(zz, -numpy.ones(len(pz) - len(zz)))

            # compute gain at a few frequencies
            wsg = 2 * numpy.pi * fs * numpy.logspace(-9, -1, 10)
            wzg = numpy.exp(wsg / fs)

            ks0 = []
            kz0 = []
            for n in range(len(wsg)):
                ks0.append(abs((wsg[n] - zs).prod() / (wsg[n] - ps).prod()))
                kz0.append(abs((wzg[n] - zz).prod() / (wzg[n] - pz).prod()))

            kz = numpy.median(ks * numpy.array(ks0) / numpy.array(kz0))
        else:
            # Pre-warp using given match frequency (Matlab's bilinear):
            fp *= 2 * numpy.pi
            fs = fp / numpy.tan(fp/(fs*2))

            # Strip infinities from zeros
            zs = zs[numpy.isfinite(zs)]
            # Do bilinear transformation
            pz = (1 + ps/fs) / (1 - ps/fs)
            zz = (1 + zs/fs) / (1 - zs/fs)
            # [real kz based on Matt's code]
            # (comment in original Matlab function: "real kz or just kz?")
            kz = (ks * (fs-zs).prod() / (fs-ps).prod()).real
            # Add extra zeros at -1
            zz = numpy.contatenate((zz, -numpy.ones(len(pz) - len(zz))))

        return zz, pz, kz

    def getSOS(self, fs, *args):
        """
        Returns SOSFilter in second-order-sections for use in real-time filtering.

        Parameters
        ----------
        fs : `float`, `numpy.ndarray`
            sample frequency in Hz.

        Returns
        -------
        sos : `numpy.ndarray`
            second-order sections matrix.

        Notes
        -----
        Matt Evans' Matlab code: getSOS.m
        """
        z, p, k = self.get_zpk_z(fs, *args)

        # Matlab's zp2sos
        # ---------------

        # (0) Set default values
        #direction_flag = 'up'
        #scale = 'none'
        #krzflag = False

        # (1) Order the poles and zeros in complex conj. pairs
        # and return the number of poles and zeros
        z = cplxpair(z)
        p = cplxpair(p)

        lz = len(z)
        lp = len(p)
        l = int(numpy.ceil(lp/2))

        if lz > lp: raise Exception('Too many zeros.')

        # (2) Break up conjugate pairs and real poles and zeros
        p_conj = p[numpy.where(p.imag!=0)] # poles that are conjugate pairs
        p_real = p[numpy.where(p.imag==0)] # poles that are real

        z_conj = z[numpy.where(z.imag!=0)] # zeros that are conjugate pairs
        z_real = z[numpy.where(z.imag==0)] # zeros that are real

        # (3) Order poles according to proximity to unit circle
        dist_c = abs(p_conj - p_conj/abs(p_conj))
        p_conj = numpy.array([pc for (d, pc) in sorted(zip(dist_c, p_conj))])

        dist_r = abs(p_real - numpy.sign(p_real))
        p_real = numpy.array([pr for (d, pr) in sorted(zip(dist_r, p_real))])

        new_p = numpy.append(p_conj, p_real)

        # (4) Order zeros according to proximity to pole pairs
        new_z = []
        zc_pool = list(z_conj)
        zr_pool = list(z_real)
        pc_pool = list(p_conj)
        pr_pool = list(p_real)
        def dist(p): return lambda z: abs(z - p)

        # order complex zero pairs
        for i in range(len(z_conj)/2):
            if pc_pool:
                # there are conjugate pole pairs left
                zc_pool = sorted(zc_pool, key = dist(pc_pool.pop(0)))
                new_z += [zc_pool.pop(0)]
                zc_pool = sorted(zc_pool, key = dist(pc_pool.pop(1)))
                new_z += [zc_pool.pop(0)]
            elif pr_pool:
                # there are real poles left
                zc_pool = sorted(zc_pool, key = dist(pr_pool.pop(0)))
                new_z += [zc_pool.pop(0), zc_pool.pop(0)]
                if pr_pool: del pr_pool[0]
            else:
                new_z += zc_pool
                break

        # order remaining real zeros
        for i in range(len(z_real)):
            if pc_pool:
                # there are conjugate pole pairs left
                zr_pool = sorted(zr_pool, key = dist(pc_pool.pop(0)))
                new_z += [zr_pool.pop(0)]
            elif pr_pool:
                # there are real poles left
                zr_pool = sorted(zr_pool, key = dist(pr_pool.pop(0)))
                new_z += [zr_pool.pop(0)]
            else:
                new_z += zr_pool
                break

        new_z = numpy.array(new_z)

        # (5) Form SOS matrix
        sos = []
        if lz == 0:
            #print "no zeros"
            if lp == 0:
                ##print "no poles"
                sos = [1, 0, 0, 1, 0, 0]
            elif lp % 2 == 0:
                #print "even number of poles"
                new_p2 = numpy.reshape(new_p, (lp/2, 2))
                for np_pair in new_p2:
                    lti = signal.lti([], np_pair, 1)
                    sos = [numpy.concatenate((lti.num, lti.den))] + sos
            else:
                #print "odd number of poles"
                new_p2 = numpy.reshape(new_p[:-1], ((lp-1)/2, 2))
                for np_pair in new_p2:
                    lti = signal.lti([], np_pair, 1)
                    sos = [numpy.concatenate((lti.num, lti.den))] + sos
                # handle last pole separately
                lti = signal.lti([], new_p[-1], 1)
                sos = [numpy.concatenate((lti.num, [0], lti.den, [0]))] + sos
        else:
            if lz % 2 == 0:
                #print "even number of zeros" , lz
                # handle first lz poles
                new_z2 = numpy.reshape(new_z, (lz/2, 2))
                new_p2 = numpy.reshape(new_p[:lz], (lz/2, 2))
                for (zpair, ppair) in zip(new_z2, new_p2):
                    lti = signal.lti(zpair, ppair, 1)
                    sos = [numpy.concatenate((lti.num, lti.den))] + sos
                # continue for remaining poles if any
                if not numpy.mod(lp, 2):
                    #print "even number of poles"
                    new_p2 = numpy.reshape(new_p[lz:], ((lp-lz)/2, 2))
                    for np_pair in new_p2:
                        lti = signal.lti([], np_pair, 1)
                        sos = [numpy.concatenate((lti.num, lti.den))] + sos
                else:
                    #print "odd number of poles"
                    new_p2 = numpy.reshape(new_p[lz:-1], ((lp-lz-1)/2, 2))
                    for np_pair in new_p2:
                        lti = signal.lti([], np_pair, 1)
                        sos = [numpy.concatenate((lti.num, lti.den))] + sos
                    # handle last pole separately
                    lti = signal.lti([], new_p[-1], 1)
                    sos = [numpy.concatenate((lti.num, [0], lti.den, [0]))] + sos
            else:
                #print "odd number of zeros"
                # handle first lz-1 poles
                new_z2 = numpy.reshape(new_z[:-1], ((lz-1)/2, 2))
                new_p2 = numpy.reshape(new_p[:(lz-1)], ((lz-1)/2, 2))
                for (zpair, ppair) in zip(new_z2, new_p2):
                    lti = signal.lti(zpair, ppair, 1)
                    sos = [numpy.concatenate((lti.num, lti.den))] + sos
                # handle last zero separately
                if lz == lp:
                    #print "same number of poles and zeros"
                    lti = signal.lti(new_z[-1], new_p[-1], 1)
                    sos = [numpy.concatenate((lti.num, [0], lti.den, [0]))] + sos
                else:
                    #print "more poles than zeros"
                    lti = signal.lti(new_z[-1], new_p[lz:lz+2], 1)
                    sos = [numpy.concatenate((lti.num, lti.den))] + sos
                    # continue for remaining poles if any
                    if not numpy.mod(lp, 2):
                        # even number of poles
                        new_p2 = numpy.reshape(new_p[lz+1:], ((lp-lz-1)/2, 2))
                        for np_pair in new_p2:
                            lti = signal.lti([], np_pair, 1)
                            sos = [numpy.concatenate((lti.num, lti.den))] + sos
                    else:
                        #print "odd number of poles"
                        new_p2 = numpy.reshape(new_p[lz+1:-1], ((lp-lz-2)/2, 2))
                        for np_pair in new_p2:
                            lti = signal.lti([], np_pair, 1)
                            sos = [numpy.concatenate((lti.num, lti.den))] + sos
                        # handle last pole separately
                        lti = signal.lti([], new_p[-1], 1)
                        sos = [numpy.concatenate((lti.num, [0], lti.den, [0]))] + sos

        sos = numpy.array(sos)

        # (6) Change direction if requested
        # ['flag' IS 'up' WHEN MATT CALLS THIS, SO I'M SKIPPING THIS STEP]

        # At this point no scaling has been performed
        # The leading coefficients of both num and den are one.

        # (7) Perform appropriate scaling
        # ['scale' IS 'none' WHEN MATT CALLS THIS, SO I'M SKIPPING THIS STEP]

        # (8) Embed the gain if only one output argument was specified
        # [THIS IS ALWAYS TRUE IN OUR CASE]
        sos[0][0:3] = k * sos[0][0:3]

        return sos

    def zfilt(self, x, fs, *args):
        """
        Filters x (with sample frequency fs) to produce y

        Parameters
        ----------
        x : `numpy.ndarray`
            Time series to be filtered.
        fs : `float`, `numpy.ndarray`
            sample frequency in Hz.

        Returns
        -------
        y : `numpy.ndarray`
            filtered time series

        Notes
        -----
        Matt Evans' Matlab code: zfilt.m
        """
        sos = self.getSOS(fs, *args)
        y = x
        for row in sos:
            y = signal.lfilter(row[:3], row[3:], y, axis=0)
        return y


def ztransform(rs):
    """Z-transforms zeros and poles. [used in getZPKz]

    Parameters
    ----------
    rs : `numpy.ndarray`
        sample frequency in Hz

    Returns
    -------
    zz : `numpy.ndarray`
        zeros in Z-domain.
    pz : `numpy.ndarray`
        poles in Z-domain.
    kz : `numpy.ndarray`
        gain in Z-domain.

    Notes
    -----
    Matt Evans' Matlab code: getZPKz.m
    """
    FREQ_MAX_WARP = 0.4
    FREQ_MIN_WARP = 1e-9

    #### frequency prewarping ####
    fWarp = numpy.abs(rs)

    # limit warp range
    fWarp = numpy.minimum(fWarp, FREQ_MAX_WARP)
    fWarp = numpy.maximum(fWarp, FREQ_MIN_WARP)

    # warped radial frequency
    rw = rs * numpy.tan(-numpy.pi * fWarp) / fWarp

    #### transform ####
    rz = (1 + rw) / (1 - rw)

    return rz


def cplxpair(x, tol=None):
    """
    Sort numbers into complex conjugate pairs. [used in getSOS]

    rearranges the elements of X so that complex numbers are collected
    into matched pairs of complex conjugates. The pairs are ordered by
    increasing real part. Any purely real elements are placed after all
    the complex pairs.

    Parameters
    ----------
    x : `numpy.ndarray`
        complex conjugate pairs and/or real numbers.

    Returns
    -------
    xsorted : `numpy.ndarray`
        rearranged array.

    Notes
    -----
    Matlab code: cplxpair.m (not copied! recreated)
    """
    x = numpy.atleast_1d(x)

    if x.size == 0:
        # if x is empty, don't do anything
        xsorted = x
    else:
        # get tolerance
        tol = tol or 100 * numpy.spacing(1)
        # I'm not sure this is the right way to replicate:
        # tol = 100*eps(class(x))

        x = x.flatten()

        # pick & sort real-only elements
        xr = numpy.sort(x[numpy.where(abs(x.imag) <= tol)])
        # pick & sort complex elements, so that conjugates end up together
        xc = list(set(x) - set(xr))
        xc_posim = [xi.real + 1j*abs(xi.imag) for xi in xc]
        xc = [xci for (xi, xci) in sorted(zip(xc_posim, xc))]
        xc = numpy.array(xc)


        if xc.size == 0:
            # no complex numbers, just return reals
            xsorted = xr
        elif numpy.mod(len(xc), 2) == 1:
            # there's an odd number of complex elements
            raise Exception('Complex numbers cannot be paired: odd number.')
        else:
            # check pairs of conjugates
            xcpairs = numpy.reshape(xc, (len(xc)/2, 2))
            for pair in xcpairs:
                realdif = pair[0].real - pair[1].real
                if realdif > tol*abs(pair[0].real):
                    raise Exception('Complex numbers cannot be paired: '
                                    'not in conjugate pairs (Re).')
                imagsum = pair[0].imag + pair[1].imag
                if imagsum > tol*abs(pair[0].real):
                    raise Exception('Complex numbers cannot be paired: '
                                    'not in conjugate pairs (Im).')

            xsorted = numpy.append(xc, xr)

        return xsorted
