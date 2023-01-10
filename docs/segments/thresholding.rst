#############################################
Generating data-quality flags by thresholding
#############################################

.. currentmodule:: gwpy.segments

The first- and second-generation ground-based laser interferometer
gravitational-wave detectors are subject to a large variety of linear noise
sources, in which noise in control signals can couple directly into the
gravitational-wave readout.
If the coupling between an auxiliary signal and the gravitational-wave signal
can be detected, noise in the auxiliary signal can be flagged by recording
times when the time-series signal exceeded a nominal range.

These times can then be recorded as ``[start, stop)`` semi-open intervals
('segments'), and applied to any analysis of gravitational-wave data as a veto.

In GWpy, a :class:`DataQualityFlag` can be generated from any
:class:`~gwpy.timeseries.TimeSeries` by applying a simple mathematical operator:

.. plot::
    :context: reset
    :caption: Generating a :class:`DataQualityFlag` by thresholding on band-passed data from a low-freuency microphone.

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.get(
    ...     "L1:PEM-CS_LOWFMIC_LVEA_VERTEX_DQ",
    ...     1186736512,
    ...     1186747264,
    ...     host="losc-nds.ligo.org",
    ... )
    >>> bandpassed = data.bandpass(0.1, 1)
    >>> loudbang = bandpassed.abs() > 30
    >>> flag = loudbang.to_dqflag(
    ...     name='L1:loug_bang',
    ...     label="Loud bang",
    ...     round=True,
    ...     description="Large peaks in 0.1-1Hz region",
    ... )
    >>> print(flag)
    <DataQualityFlag('L1:loud-bang',
                     known=[[1186736512.0 ... 1186747264.0)]
                     active=[[1186743246.0 ... 1186743249.0)
                             [1186743373.0 ... 1186743379.0)
                             [1186744628.0 ... 1186744633.0)
                             [1186745535.0 ... 1186745537.0)
                             [1186745563.0 ... 1186745564.0)
                             [1186745875.0 ... 1186745881.0)
                             [1186745937.0 ... 1186745938.0)]
                     description='Large peaks in 0.1-1Hz region')>

We can plot the original data alongside the segments to visualise the flag we
have generated:

.. plot::
    :context:
    :caption: Plotting a :class:`DataQualityFlag` alongside data

    >>> plot = data.plot(
    ...     title="LIGO-Livingston LVEA low-frequency sound",
    ...     ylabel="Microphone reading [Pa]",
    ... )
    >>> plot.add_segments_bar(flag)
    >>> plot.show()

In this worked example, times where the absolute value
(:meth:`~gwpy.timeseries.TimeSeries.abs`) of the data in the
0.1-1 Hz frequency band exceeded 30 Pa, as recorded by a low-frequency
microphone, are recorded as a `DataQualityFlag`.
The keyword arguments given to the
:meth:`~gwpy.timeseries.StateTimeSeries.to_dqflag` method give the flag a
sensible name (using the standard naming convention), a label (for
visualisation), a description, and make sure the segments are rounded outwards
to integer GPS start and stop times.

The flag is fairly crudely generated, but captures all of the large excursions
seen in the raw data.
Further study would be required to understand the impact of these excursions
('loud bangs') on the scientific measurement (detection of
gravitational waves).
