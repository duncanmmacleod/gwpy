# Copyright (c) 2014-2017 Louisiana State University
#               2017-2026 Cardiff University
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

"""GPS axis locators, formatters, and scales."""

from __future__ import annotations

import contextlib
from decimal import Decimal
from math import (
    ceil,
    floor,
)
from numbers import Number
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)

import numpy
from astropy import units
from matplotlib import ticker
from matplotlib.scale import (
    LinearScale,
    get_scale_names,
    register_scale,
)
from matplotlib.transforms import Transform

try:
    from matplotlib import _docstring
    from matplotlib.scale import _get_scale_docs
except ImportError:  # maybe matplotlib >= 3.12?
    HAVE_DOCSTRING = False
else:
    HAVE_DOCSTRING = True

from ..time import (
    from_gps,
    to_gps,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import (
        Literal,
        SupportsFloat,
        TypeVar,
    )

    from astropy.units import NamedUnit
    from matplotlib.axis import Axis
    from numpy.typing import ArrayLike

    from ..time import SupportsToGps
    from ..typing import UnitLike

    NumericType = TypeVar("NumericType", bound=SupportsFloat)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

#: Maximum number of week ticks to display
WEEK_SCALE_MAJOR_TICKS = 6

#: Supported time scales
TIME_UNITS = (
    units.nanosecond,
    units.microsecond,
    units.millisecond,
    units.second,
    units.minute,
    units.hour,
    units.day,
    units.week,
    units.year,
    units.kiloyear,
    units.megayear,
    units.gigayear,
)

GPS_SCALES = {}


def _truncate(f: SupportsFloat, n: int) -> str:
    """Truncates/pads a float `f` to `n` decimal places without rounding.

    From https://stackoverflow.com/a/783927/1307974 (CC-BY-SA)
    """
    s = str(f)
    if "e" in s or "E" in s:
        return f"{f:.{n}f}"
    i, _, d = s.partition(".")
    return ".".join([i, (d+"0"*n)[:n]])


# -- base mixin for all GPS manipulations

class GPSMixin:
    """Mixin adding GPS-related attributes to any class."""

    def __init__(
        self,
        *args,
        unit: NamedUnit | str | None = None,
        epoch: SupportsToGps | None = None,
        **kwargs,
    ) -> None:
        """Initialise a new GPS-scaled object."""
        self.set_unit(unit)
        self.set_epoch(epoch)

        # call super for __init__ if this is part of a larger MRO
        with contextlib.suppress(TypeError):
            super().__init__(*args, **kwargs)

    def get_epoch(self) -> float | None:
        """Return the GPS epoch."""
        return self._epoch

    def set_epoch(self, epoch: SupportsToGps | None) -> None:
        """Set the GPS epoch."""
        if epoch is None:
            self._epoch = None
        else:
            self._epoch = float(to_gps(epoch))

    epoch = property(
        fget=get_epoch,
        fset=set_epoch,
        doc=get_epoch.__doc__,
    )

    def get_unit(self) -> NamedUnit | None:
        """GPS step scale."""
        return self._unit

    def set_unit(self, unit: UnitLike | Number | None) -> None:
        """Set the GPS step scale."""
        # accept all core time units
        if (
            unit is None
            or (isinstance(unit, units.NamedUnit) and unit.physical_type == "time")
        ):
            self._unit = unit
            return

        second: NamedUnit = units.second

        # convert float to custom unit in seconds
        if isinstance(unit, Number):
            unit = units.Unit(unit * second)

        # otherwise, should be able to convert to a time unit
        try:
            unit = units.Unit(unit)
        except ValueError:
            # catch annoying plurals
            unit = units.Unit(str(unit).rstrip("s"))

        # decompose and check that it's actually a time unit
        dec = unit.decompose()
        if dec.bases != [second]:
            msg = f"cannot set GPS unit to '{unit}'"
            raise ValueError(msg)

        # check equivalent units
        for other in TIME_UNITS:
            if other.decompose().scale == dec.scale:
                self._unit = other
                return

        msg = f"unrecognised unit '{unit}'"
        raise ValueError(msg)

    unit = property(
        fget=get_unit,
        fset=set_unit,
        doc=get_unit.__doc__,
    )

    def get_unit_name(self) -> str | None:
        """Return the name of the unit for this GPS scale.

        Note that this returns a simply-pluralised version of the name.
        """
        unit: NamedUnit | None = self.get_unit()
        if not unit:
            return None
        try:
            name = unit.long_names[0]
        except IndexError:
            name = unit.name
        if len(name) > 1:
            return name + "s"  # pluralise for humans
        return name

    def get_scale(self) -> float:
        """Return the scale (in seconds) of the current GPS unit."""
        if self.unit is None:
            return 1
        return cast("float", self.unit.decompose().scale)

    scale = property(fget=get_scale, doc=get_scale.__doc__)


# -- GPS transforms ----------------

class _GPSTransformBase(GPSMixin, Transform):
    """Transform GPS time into N * scale from epoch.

    This class uses the `decimal.Decimal` object to protect against precision
    errors when converting to and from GPS times that may have 19 significant
    digits, which is more than `float` can handle.

    There is some logic to _only_ use the slow decimal transforms when
    absolutely necessary, normally when transforming tick positions.
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    is_affine = True
    has_inverse = True

    def transform(self, values: ArrayLike) -> numpy.ndarray:
        """Transform an array of GPS times."""
        # format ticks using decimal for precision display
        if isinstance(values, float | Decimal):
            return numpy.asanyarray(
                self._transform_decimal(values, self.epoch or 0, self.scale),
            )
        return super().transform(values)

    def transform_non_affine(self, values: ArrayLike) -> ArrayLike:
        """Transform an array of GPS times.

        This method is designed to filter out transformations that will
        generate text elements that require exact precision, and use
        `Decimal` objects to do the transformation, and simple `float`
        otherwise.
        """
        scale = self.scale or 1
        epoch = self.epoch or 0

        values = numpy.asarray(values)

        # handle simple or data transformations with floats
        if self._parents or (  # part of composite transform (from draw())
            # no large additions
            epoch == 0
            # no multiplications
            and scale == 1
        ):
            return self._transform(values, float(epoch), float(scale))

        # otherwise do things carefully (and slowly) with Decimals
        # -- ideally this only gets called for transforming tick positions
        flat = values.flatten()

        def _trans(x: float) -> float:
            return self._transform_decimal(x, epoch, scale)

        return numpy.fromiter(
            map(_trans, flat),
            dtype=float,
            count=flat.size,
        ).reshape(values.shape)

    @overload
    @staticmethod
    def _transform(value: ArrayLike, epoch: float, scale: float) -> ArrayLike: ...
    @overload
    @staticmethod
    def _transform(value: Decimal, epoch: Decimal, scale: Decimal) -> Decimal: ...

    @staticmethod
    def _transform(
        value: ArrayLike | Decimal,
        epoch: float | Decimal,
        scale: float | Decimal,
    ) -> ArrayLike | Decimal:
        """Transform the GPS ``value`` into a scaled time relative to an epoch."""
        # convert GPS into scaled time from epoch
        return (value - epoch) / scale

    @classmethod
    def _transform_decimal(
        cls,
        value: NumericType,
        epoch: SupportsFloat,
        scale: SupportsFloat,
    ) -> NumericType:
        """Transform to/from GPS using `decimal.Decimal` for precision."""
        vdec = Decimal(_truncate(value, 12))
        edec = Decimal(_truncate(epoch, 12))
        sdec = Decimal(_truncate(scale, 12))
        return type(value)(cls._transform(vdec, edec, sdec))


class GPSTransform(_GPSTransformBase):
    """Transform GPS into time (scaled units) from epoch."""

    def inverted(self) -> InvertedGPSTransform:
        """Return the inverse of this `GPSTransform`."""
        return InvertedGPSTransform(unit=self.unit, epoch=self.epoch)


class InvertedGPSTransform(_GPSTransformBase):
    """Transform time (scaled units) from epoch into GPS time."""

    @overload
    @staticmethod
    def _transform(value: ArrayLike, epoch: float, scale: float) -> ArrayLike: ...
    @overload
    @staticmethod
    def _transform(value: Decimal, epoch: Decimal, scale: Decimal) -> Decimal: ...

    @staticmethod
    def _transform(
        value: ArrayLike | Decimal,
        epoch: float | Decimal,
        scale: float | Decimal,
    ) -> ArrayLike | Decimal:
        """Transform the scaled time back into GPS."""
        return value * scale + epoch

    def inverted(self) -> GPSTransform:
        """Return the inverse of this `InvertedGPSTransform`."""
        return GPSTransform(unit=self.unit, epoch=self.epoch)


# -- locators and formatters ---------

class GPSAutoLocator(ticker.MaxNLocator):
    """Find the best position for ticks on a given axis from the data.

    This auto-locator gives a simple extension to the matplotlib
    `~matplotlib.ticker.AutoLocator` allowing for variations in scale
    and zero-time epoch.
    """

    def __init__(
        self,
        nbins: int | Literal["auto"] | None = 12,
        **kwargs,
    ) -> None:
        """Initialise a new `GPSAutoLocator`.

        Each of the `epoch` and `scale` keyword arguments should match those
        passed to the `GPSFormatter`
        """
        super().__init__(
            nbins=nbins,
            **kwargs,
        )

    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        """Generate the list of tick values for the given interval."""
        self.axis: Axis
        transform = self.axis.get_transform()
        if not isinstance(transform, GPSTransform):
            msg = "GPSAutoLocator can only be used with a GPSTransform"
            raise TypeError(msg)
        unit = transform.get_unit()
        steps = self._steps

        vmin, vmax = transform.transform((vmin, vmax))

        # if less than 6 weeks, major tick every week
        if (
            steps is None
            and unit == units.week
            and vmax - vmin <= WEEK_SCALE_MAJOR_TICKS
        ):
            self.set_params(steps=[1, 10])
        else:
            self.set_params(steps=None)

        try:
            ticks = super().tick_values(vmin, vmax)
        finally:
            self._steps = steps
        return transform.inverted().transform(ticks)


class GPSAutoMinorLocator(ticker.AutoMinorLocator):
    """Find the best position for minor ticks on a given GPS-scaled axis."""

    def __call__(self) -> Sequence[float]:
        """Return the locations of the ticks."""
        self.axis: Axis
        majorlocs: numpy.ndarray = self.axis.get_majorticklocs()
        trans = self.axis.get_transform()
        try:
            majorstep = majorlocs[1] - majorlocs[0]
        except IndexError:
            # Need at least two major ticks to find minor tick locations
            # TODO (@duncanmmacleod): Figure out a way to still be able to
            # display minor ticks without two major ticks visible.
            # For now, just display no ticks at all.
            majorstep = 0

        if self.ndivs is None:
            if majorstep == 0:
                # TODO (@duncanmmacleod): Need a better way to figure out ndivs
                ndivs = 1
            else:
                scale_ = trans.get_scale()
                gpsstep = majorstep / scale_
                x = round(10 ** (numpy.log10(gpsstep) % 1))
                if trans.unit == units.week and gpsstep == 1:
                    ndivs = 7
                elif trans.unit == units.year and gpsstep <= 1:
                    ndivs = 6
                elif trans.unit != units.day and x in [1, 5, 10]:
                    ndivs = 5
                else:
                    ndivs = 4
        else:
            ndivs = self.ndivs

        minorstep = majorstep / ndivs

        vmin, vmax = cast("tuple[float, float]", self.axis.get_view_interval())
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if numpy.size(majorlocs):
            epoch = majorlocs[0]
            tmin = floor((vmin - epoch) / minorstep) * minorstep
            tmax = ceil((vmax - epoch) / minorstep) * minorstep
            locs = numpy.arange(tmin, tmax, minorstep) + epoch
            cond = numpy.abs((locs - epoch) % majorstep) > minorstep / 10.0
            locs = locs.compress(cond)
        else:
            locs = []

        return self.raise_if_exceeds(locs)


class GPSFormatter(ticker.Formatter):
    """Format ticks on a `GPSScale` axis."""

    def __call__(
        self,
        x: float,
        pos: int | None = None,  # noqa: ARG002
    ) -> str:
        """Format a tick on this scale."""
        trans = self.axis.get_transform()
        flt = float(trans.transform(float(x)))
        if flt.is_integer():
            return str(int(flt))
        return str(flt)


# -- scales --------------------------

class GPSScale(GPSMixin, LinearScale):
    """A GPS scale, displaying time (scaled units) from an epoch.

    Parameters
    ----------
    unit : `astropy.units.Unit`, optional
        The unit to use for ticks on the axis.

    epoch : `float`, `gwpy.time.LIGOTimeGPS`, optional
        The GPS epoch (origin) for axis ticks.

    Notes
    -----
    Unlike most matplotlib scales, `GPSScale` is deliberately not
    axis-agnostic: when ``unit``/``epoch`` are not set explicitly,
    `get_transform` re-derives them on every call from the axis bound
    via `set_axis`/`set_default_locators_and_formatters`, using its
    current data/view limits -- both so the displayed unit adapts as
    data changes or the view is zoomed, and so GPS times (up to 19
    significant digits) are rescaled near a nearby epoch before being
    handed to matplotlib's float-based rendering pipeline. This
    mirrors the ``axis``/``set_axis()`` convention used by
    `matplotlib.ticker.TickHelper` (the base of
    `~matplotlib.ticker.Locator`/`~matplotlib.ticker.Formatter`,
    including `GPSAutoLocator`/`GPSFormatter` below) -- the standard
    matplotlib idiom for helper objects needing a live per-axis
    reference established after construction, rather than the
    constructor-time ``axis`` parameter matplotlib >= 3.11 pending-
    deprecates. Unlike `TickHelper.create_dummy_axis`, `GPSScale` does
    not silently fall back to a dummy axis when unbound: its
    `get_transform` output feeds `Axes.transData` directly, so a
    missing axis fails loudly rather than silently producing a
    plausible-looking but numerically wrong transform.
    """

    name = "auto-gps"
    Transform = GPSTransform
    InvertedTransform = InvertedGPSTransform

    #: Default for `axis` before `set_axis` is first called. Assigning
    #: ``self.axis = ...`` (in `set_axis`, below) always creates a
    #: per-instance attribute that shadows this class-level default; it
    #: is never mutated in place, so instances never share state through
    #: it (mirrors `matplotlib.ticker.TickHelper.axis`).
    axis: Axis | None = None

    def __init__(
        self,
        _axis: Axis | None = None,
        *,
        unit: NamedUnit | str | Number | None = None,
        epoch: SupportsToGps | None = None,
    ) -> None:
        """Initialise this `GPSScale`.

        The leading positional argument is accepted only for
        compatibility with matplotlib < 3.11, which always passes the
        `~matplotlib.axis.Axis` to scale constructors positionally;
        matplotlib >= 3.11 does not, and does not need to, since the
        axis is bound separately via `set_axis`.
        """
        super().__init__(unit=unit, epoch=epoch)
        self.set_axis(_axis)

    def set_axis(self, axis: Axis | None) -> None:
        """Bind this scale to ``axis``.

        Mirrors `matplotlib.ticker.TickHelper.set_axis`, the
        convention matplotlib uses for helper objects that need a
        live per-axis reference established after construction.
        """
        self.axis = axis

    def set_default_locators_and_formatters(self, axis: Axis) -> None:
        """Set the defualt locators and formatters for ``axis``."""
        self.set_axis(axis)
        # set tight scaling on parent axes
        getattr(axis.axes, f"set_{axis._get_axis_name()}margin")(0)  # ty: ignore[unresolved-attribute]
        axis.set_major_locator(GPSAutoLocator())
        axis.set_major_formatter(GPSFormatter())
        axis.set_minor_locator(GPSAutoMinorLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    @staticmethod
    def _lim(axis: Axis) -> tuple[float, float]:
        """Find the current view limits of this ``axis``."""
        # if autoscaling and datalim is set, use it
        dlim = axis.get_data_interval()
        if (
            getattr(axis.axes, f"get_autoscale{axis._get_axis_name()}_on")()  # ty: ignore[unresolved-attribute]
            and not numpy.isinf(dlim).any()
        ):
            return dlim

        # otherwise use the view lim
        return axis.get_view_interval()

    def _auto_epoch(self, axis: Axis) -> int:
        """Find the best GPS epoch (origin) for this ``axis``."""
        # use the lower data/view limit as the epoch
        epoch = round(self._lim(axis)[0])

        # round epoch in successive units for large scales
        unit = self.get_unit()
        date = from_gps(epoch)
        fields = ("second", "minute", "hour", "day")
        for i, u in enumerate(fields[1:]):
            if unit < units.Unit(u):
                break
            if u in ("day",):
                date = date.replace(**{fields[i]: 1})  # ty: ignore[invalid-argument-type]
            else:
                date = date.replace(**{fields[i]: 0})  # ty: ignore[invalid-argument-type]
        return int(to_gps(date))

    def _auto_unit(self, axis: Axis) -> NamedUnit:
        """Find the best scaled unit for this ``axis``."""
        # get width of axis
        vmin, vmax = self._lim(axis)
        duration = vmax - vmin

        second: NamedUnit = units.second

        # find time unit that fits the duration well;
        # the magic scaling of 4 or 0.01 is entirely arbitrary,
        # but in practice results in figures that scale nicely
        for scale in TIME_UNITS[::-1]:
            base = cast("float", scale.decompose().scale)
            # for large durations, prefer smaller units
            if scale > second:
                base *= 4
            # for smaller durations, prefer larger units
            else:
                base *= 0.01
            if duration >= base:
                return scale

        # if nothing else worked, just use seconds
        return second

    def _require_axis(self) -> Axis:
        """Return `axis`, raising if this scale is not yet attached.

        Used to derive automatic unit/epoch values, which need a live
        `~matplotlib.axis.Axis` to inspect -- see the class `Notes` for
        why this fails loudly rather than degrading gracefully like
        `matplotlib.ticker.TickHelper.create_dummy_axis`.
        """
        if self.axis is None:
            msg = (
                f"{type(self).__name__} is not attached to a matplotlib "
                "Axis; call Axes.set_xscale()/set_yscale() (which binds "
                "it via set_axis()) before relying on automatic "
                "unit/epoch selection"
            )
            raise RuntimeError(msg)
        return self.axis

    def get_transform(self) -> GPSTransform:
        """Return the `GPSTransform` associated with this scale."""
        # get current settings
        epoch = self.get_epoch()
        unit = self.get_unit()

        # dynamically set epoch and/or unit if None
        if unit is None:
            self.set_unit(self._auto_unit(self._require_axis()))
        if epoch is None:
            self.set_epoch(self._auto_epoch(self._require_axis()))

        # build transform on-the-fly
        try:
            return self.Transform(
                unit=self.get_unit(),
                epoch=self.get_epoch(),
            )
        finally:  # reset to current settings
            self.set_epoch(epoch)
            self.set_unit(unit)


# -- registrations -------------------

def register_gps_scale(scale_class: type[GPSScale]) -> None:
    """Register a new GPS scale.

    ``scale_class`` must be a subclass of `GPSScale`.
    """
    register_scale(scale_class)
    GPS_SCALES[scale_class.name] = scale_class


def _gps_scale_factory(unit: NamedUnit) -> type[GPSScale]:
    """Construct a GPSScale for this unit."""

    class FixedGPSScale(GPSScale):
        """`GPSScale` for a specific GPS time unit."""

        name = (unit.long_names or unit.names)[0] + "s"

        def __init__(
            self,
            _axis: Axis | None = None,
            *,
            epoch: SupportsToGps | None = None,
        ) -> None:
            super().__init__(_axis, epoch=epoch, unit=unit)

    return FixedGPSScale


register_gps_scale(GPSScale)  # auto-gps

for _unit in TIME_UNITS:
    # don't go past 'year' for GPSScale
    if _unit is units.kiloyear:
        break
    register_gps_scale(_gps_scale_factory(_unit))

# update the docstring for matplotlib scale methods
if HAVE_DOCSTRING:
    _docstring_interp_params = {
        "scale": " | ".join([repr(x) for x in get_scale_names()]),
        "scale_docs": _get_scale_docs().rstrip(),
    }
    try:
        _docstring.interpd.register(**_docstring_interp_params)
    except AttributeError:  # matplotlib < 3.10
        _docstring.interpd.update(_docstring_interp_params)  # ty: ignore[unresolved-attribute]
