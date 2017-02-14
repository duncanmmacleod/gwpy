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

"""Unit test for utils module
"""

import os
import subprocess

import pytest

from compat import unittest

from astropy import units

from gwpy.utils import shell

# conditionally import lal
try:
    import lal
except ImportError:
    HAS_LAL = False
else:
    HAS_LAL = True
    from gwpy.utils import lal as lalutils

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class ShellUtilsTestCase(unittest.TestCase):
    """`TestCase` for the utils.shell module
    """
    def test_shell_call(self):
        out, err = shell.call(["echo", "This works"])
        self.assertEqual(out, "This works\n")
        self.assertEqual(err, '')
        out2, err2 = shell.call("echo 'This works'")
        self.assertEqual(out, out2)
        self.assertEqual(err, err2)
        self.assertRaises(OSError, shell.call, ['this-command-doesnt-exist'])
        self.assertRaises(subprocess.CalledProcessError, shell.call,
                          'this-command-doesnt-exist')
        self.assertRaises(subprocess.CalledProcessError, shell.call, 'false')
        with pytest.warns(UserWarning):
            shell.call('false', on_error='warn')

    def test_which(self):
        try:
            result, _ = shell.call('which true')
        except Exception as e:
            self.skipTest(str(e))
        else:
            result = result.rstrip('\n')
        self.assertEqual(shell.which('true'), result)


@unittest.skipIf(not HAS_LAL, 'lal-python not installed')
class LALUtilsTestCase(unittest.TestCase):
    def test_to_lal_type(self):
        self.assertEqual(lalutils.to_lal_type(float), lal.D_TYPE_CODE)
        self.assertEqual(lalutils.to_lal_type(int), lal.I8_TYPE_CODE)
        self.assertEqual(lalutils.to_lal_type('c8'), lal.C_TYPE_CODE)
        self.assertRaises(TypeError, lalutils.to_lal_type, 'blah')

    def test_to_lal_typestr(self):
        self.assertEqual(lalutils.to_lal_typestr(float), 'REAL8')
        self.assertEqual(lalutils.to_lal_typestr(int), 'INT8')
        self.assertEqual(lalutils.to_lal_typestr('c8'), 'COMPLEX8')
        self.assertRaises(TypeError, lalutils.to_lal_typestr, 'blah')

    def test_to_lal_unit(self):
        self.assertEqual(lalutils.to_lal_unit('m**2 / kg ** 4'),
                         lal.MeterUnit ** 2 / lal.KiloGramUnit ** 4)
        with pytest.raises(ValueError) as exc:
            lalutils.to_lal_unit('rad')
        self.assertIn('LAL has no unit corresponding to', str(exc))

    def test_from_lal_unit(self):
        u = lal.MeterUnit ** 2 / lal.KiloGramUnit ** 4
        self.assertEqual(lalutils.from_lal_unit(u), units.Unit('m^2/kg^4'))
        self.assertEqual(lalutils.from_lal_unit('m^2'), units.Unit('m^2'))
        with pytest.raises(ValueError) as exc:
            lalutils.from_lal_unit('test')
        self.assertIn('Cannot convert', str(exc))
        self.assertEqual(
            units.Newton,
            lalutils.from_lal_unit(lalutils.to_lal_unit(units.Newton)))

    def test_to_lal_ligotimegps(self):
        # check simple conversion
        ltg = lalutils.to_lal_ligotimegps(1)
        self.assertIsInstance(ltg, lal.LIGOTimeGPS)
        self.assertEqual(ltg, 1)
        # check input lal.LIGOTimeGPS returns same object (noop)
        self.assertIs(lalutils.to_lal_ligotimegps(ltg), ltg)
