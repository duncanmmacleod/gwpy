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

"""Input/Output utilities for LAL Cache files.
"""

from __future__ import (division, print_function)

import os.path
import tempfile
import warnings
from functools import wraps
from gzip import GzipFile

from six import string_types
from six.moves import StringIO

from glue.lal import Cache

try:
    from lal.utils import CacheEntry
except ImportError:  # no lal
    HAS_CACHEENTRY = False
else:
    HAS_CACHEENTRY = True
    Cache.entry_class = CacheEntry

from ..time import LIGOTimeGPS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# build list of file-like types
try:  # python2.x
    FILE_LIKE = [file, GzipFile]
except NameError:  # python3.x
    from io import IOBase
    FILE_LIKE = [IOBase, GzipFile]
try:  # protect against private member being removed
    # pylint: disable=protected-access
    FILE_LIKE.append(tempfile._TemporaryFileWrapper)
except AttributeError:
    pass
FILE_LIKE = tuple(FILE_LIKE)


def _preformat_line(func):
    """Pre-format a line of text as a `str` without ending breaks
    """
    @wraps(func)
    def wrapped(line, *args, **kwargs):  # pylint: disable=missing-docstring
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if isinstance(line, string_types):
            line = line.rstrip('\n')
        return func(line, *args, **kwargs)
    return wrapped


# -- cache I/O ----------------------------------------------------------------

def read_cache(lcf, coltype=LIGOTimeGPS):
    """Read a LAL-format cache file into memory as a `Cache`

    Parameters
    ----------
    lcf : `str`, `file`
        input file or file path to read

    coltype : `LIGOTimeGPS`, `int`, optional
        `type` for GPS times

    Returns
    -------
    cache : :class:`glue.lal.Cache`
        a cache object, representing each line in the file as a
        :class:`~lal.utils.CacheEntry`
    """
    # open file
    if not isinstance(lcf, FILE_LIKE):
        with open(lcf, 'r') as fobj:
            return read_cache(fobj, coltype=coltype)

    # read file
    out = Cache()
    for line in lcf:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        out.append(read_cache_entry(line, coltype=coltype))
    return out


@_preformat_line
def read_cache_entry(line, coltype=LIGOTimeGPS):
    """Read a `lal.CacheEntry` from a line of text

    Parameters
    ----------
    line : `str`, `bytes`
        Line of text to parse

    Returns
    -------
    entry : `lal.CacheEntry`
       The parsed entry

    Raises
    ------
    ValueError
        if the line cannot be parsed successfully as a `lal.CacheEntry`
    """
    try:
        return lal_cache_entry(line, coltype=coltype)
    except ValueError as exc:  # maybe its an FFL line?
        try:
            return ffl_cache_entry(line, coltype=coltype)
        except (ValueError, TypeError):
            raise exc


@_preformat_line
def lal_cache_entry(line, coltype=LIGOTimeGPS):
    """Parse a `lal.CacheEntry` from a line in a LAL cache file
    """
    entry = Cache.entry_class(line, coltype=coltype)

    # Virgo FFL entry will return a 'path' of '0.000000', so check here
    try:
        float(entry.path)
    except ValueError:  # parsed as LAL format entry
        return entry
    raise ValueError("could not convert {0} to CacheEntry".format(
        line))


@_preformat_line
def ffl_cache_entry(line, coltype=LIGOTimeGPS):
    """Parse a `lal.CacheEntry` from a line in an FFL file

    Parameters
    ----------
    line : `str`, `bytes`
        Line of text to parse

    Returns
    -------
    entry : `lal.CacheEntry`
       The parsed entry
    """
    filename, gps, duration, _, _ = line.split()
    gps = coltype(str(gps))
    duration = float(duration)
    entry = Cache.entry_class.from_T050017(filename)
    entry.segment = type(entry.segment)(gps, gps+duration)
    return entry


def open_cache(*args, **kwargs):  # pylint: disable=missing-docstring
    warnings.warn("gwpy.io.cache.open_cache was renamed read_cache",
                  DeprecationWarning)
    return read_cache(*args, **kwargs)


def write_cache(cache, fobj):
    """Write a :cache:`~glue.lal.Cache` to a file

    Parameters
    ----------
    cache : :class:`glue.lal.Cache`
        the cache to write

    fobj : `file`, `str`
        the open file object, or file path to write to
    """
    # open file
    if isinstance(fobj, string_types):
        with open(fobj, 'w') as fobj2:
            return write_cache(cache, fobj2)

    ffl = getattr(fobj, 'name', '').endswith('.ffl')

    # write file
    for entry in cache:
        if ffl:
            line = '{0.path} {0.segment[0]} {1} 0.0 0.0'.format(
                entry, abs(entry.segment))
        else:
            line = str(entry)
        try:
            print(line, file=fobj)
        except TypeError:
            print(line.encode('utf-8'), file=fobj)


def is_cache(cache):
    """Returns `True` if ``cache`` is a readable cache file or object

    Parameters
    ----------
    cache : `str`, `file`, :class:`~glue.lal.Cache`
        object to detect as cache

    Returns
    -------
    iscache : `bool`
        `True` if the input object is a cache, or a file in LAL cache format,
        otherwise `False`
    """
    if isinstance(cache, string_types + FILE_LIKE):
        try:
            cache_ = read_cache(cache)
        except (TypeError, ValueError, UnicodeDecodeError, ImportError):
            # failed to parse cache
            return False
        else:
            if not cache_:  # return empty file as False
                return False
            return True
    elif isinstance(cache, Cache):
        return True
    return False


# -- cache manipulation -------------------------------------------------------

def file_list(flist):
    """Parse a number of possible input types into a list of filepaths.

    Parameters
    ----------
    flist : `file-like` or `list-like` iterable
        the input data container, normally just a single file path, or a list
        of paths, but can generally be any of the following

        - `str` representing a single file path (or comma-separated collection)
        - open `file` or `~gzip.GzipFile` object
        - `~lal.utils.CacheEntry`
        - :class:`~glue.lal.Cache` object or `str` with `.cache` or
          `.lcf` extension
        - simple `list` or `tuple` of `str` paths

    Returns
    -------
    files : `list`
        `list` of `str` file paths

    Raises
    ------
    ValueError
        if the input `flist` cannot be interpreted as any of the above inputs
    """
    # open a cache file and return list of paths
    if isinstance(flist, string_types) and flist.endswith(('.cache', '.lcf')):
        return read_cache(flist).pfnlist()

    # separate comma-separate list of names
    if isinstance(flist, string_types):
        return flist.split(',')

    # parse list of entries (of some format)
    if isinstance(flist, (list, tuple)):
        return list(map(file_name, flist))

    # otherwise parse a single entry
    try:
        return [file_name(flist)]
    except ValueError as exc:
        exc.args = ("Could not parse input %r as one or more "
                    "file-like objects" % flist,)
        raise


def file_name(fobj):
    """Returns the name (path) of the file object
    """
    if isinstance(fobj, string_types):
        return fobj
    if isinstance(fobj, FILE_LIKE) and not isinstance(fobj, StringIO):
        return fobj.name
    if HAS_CACHEENTRY and isinstance(fobj, CacheEntry):
        return fobj.path
    raise ValueError("Cannot parse file name for %r" % fobj)


def file_segment(filename):
    """Return the data segment for a filename following T050017

    Parameters
    ---------
    filename : `str`, `~lal.utils.CacheEntry`
        the path name of a file

    Returns
    -------
    segment : `~gwpy.segments.Segment`
        the ``[start, stop)`` GPS segment covered by the given file

    Notes
    -----
    `LIGO-T050017 <https://dcc.ligo.org/LIGO-T050017/public>`_ declares
    a filenaming convention that includes documenting the GPS start integer
    and integer duration of a file, see that document for more details.
    """
    try:  # filename object provides its own segment information
        return filename.segment
    except AttributeError:  # otherwise parse from T050017 spec
        from ..segments import Segment
        base = os.path.basename(filename)
        try:
            _, _, start, end = base.split('-')
        except ValueError as exc:
            exc.args = ('Failed to parse %r as LIGO-T050017-compatible '
                        'filename' % filename,)
            raise
        start = float(start)
        end = int(end.split('.')[0])
        return Segment(start, start+end)


def cache_segments(*caches):
    """Build a `SegmentList` of data availability for these `Caches`

    Parameters
    ----------
    *cache : :class:`~glue.lal.Cache`, `list`
        one of more `Cache` objects (or simple `list`)

    Returns
    -------
    segments : `~gwpy.segments.SegmentList`
        a list of segments for when data should be available
    """
    from ..segments import SegmentList
    out = SegmentList()
    for cache in caches:
        out.extend(file_segment(e) for e in cache)
    return out.coalesce()


def flatten(*caches):
    """Flatten a list of :class:`Caches <glue.lal.Cache>` into a single cache

    Parameters
    ----------
    *caches
        one or more :class:`~glue.lal.Cache` objects

    Returns
    -------
    flat : :class:`~glue.lal.Cache`
        a single cache containing the unique set of entries across
        each input
    """
    cache_type = type(caches[0])
    return cache_type([e for c in caches for e in c]).unique()


def find_contiguous(*caches):
    """Separate one or more caches into sets of contiguous caches

    Parameters
    ----------
    *caches
        one or more :class:`~glue.lal.Cache` objects

    Returns
    -------
    caches : `iter` of :class:`~glue.lal.Cache`
        an interable yielding each contiguous cache
    """
    try:
        flat = flatten(*caches)
    except IndexError:
        flat = Cache()
    for segment in cache_segments(flat):
        yield flat.sieve(segment=segment)
