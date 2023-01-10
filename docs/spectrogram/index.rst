.. currentmodule:: gwpy.spectrogram

.. _gwpy-spectrogram:

#################
The `Spectrogram`
#################

While the `~gwpy.timeseries.TimeSeries` allows us to study how the amplitude of a signal changes over time, and the `~gwpy.frequencyseries.FrequencySeries` allows us to study how that amplitude changes over frequency, the time-frequency `~gwpy.spectrogram.Spectrogram` allows us to track the evolution of the `~gwpy.frequencyseries.FrequencySeries` over time.

This object is a 2-dimensional array, essentially a stacked set of spectra, one per unit time.

As always, a `Spectrogram` can be generated from any arbitrary data sequence, but here the required metadata are a combination of those required for the `~gwpy.timeserises.TimeSeries` and `~gwpy.frequencyseries.FrequencySeries`:

.. code-block:: python
    :caption: Creating a :class:`Spectrogram` from random data

    >>> import numpy
    >>> specgram = Spectrogram(numpy.random.random((100, 1000)), epoch=1000000000, sample_rate=1, f0=0, df=1)
    >>> print(specgram)
    Spectrogram([[ 0.58030742  0.94586261  0.79559404 ...,  0.25253688  0.61626489
                   0.22785403]
                 [ 0.95930736  0.93154594  0.13234058 ...,  0.13920997  0.94432426
                   0.29442085]
                 [ 0.66572174  0.77702177  0.8900096  ...,  0.18828231  0.81440898
                   0.97455031]
                 ...,
                 [ 0.46696636  0.72475187  0.17941277 ...,  0.19095158  0.83843501
                   0.92154324]
                 [ 0.81492468  0.01945053  0.77665596 ...,  0.73642962  0.78723728
                   0.20995951]
                 [ 0.35161785  0.79137264  0.50710421 ...,  0.39068193  0.61551753
                   0.74846848]],
                name: None,
                unit: None,
                epoch: 2011-09-14 01:46:59.000,
                dt: 1 s,
                f0: 0 Hz,
                df: 1 Hz,
                logf: False)

The full set of metadata that can be provided is as follows:

.. autosummary::

   ~Spectrogram.name
   ~Spectrogram.unit
   ~Spectrogram.epoch
   ~Spectrogram.dt
   ~Spectrogram.f0
   ~Spectrogram.df

=======================================================================
Calculating a `Spectrogram` from a :class:`~gwpy.timeseries.TimeSeries`
=======================================================================

The time-frequency :class:`Spectrogram` of a
:class:`~gwpy.timeseries.TimeSeries` can be calculated using the
:meth:`~gwpy.timeseries.TimeSeries.spectrogram` method.
We can extend previous examples of plotting a
:class:`~gwpy.timeseries.TimeSeries` with calculation of a `Spectrogram`
with a 2-second stride:

.. plot::
    :context: reset
    :include-source:
    :nofigs:
    :caption: Generating a :class:`Spectrogram` from a :class:`~gwpy.timeseries.TimeSeries`

    from gwpy.timeseries import TimeSeries
    gwdata = TimeSeries.fetch_open_data(
        "H1",
        "Sep 14 2015 09:45",
        "Sep 14 2015 09:55",
    )
    specgram = gwdata.spectrogram(2, fftlength=1) ** (1/2.)

.. _gwpy-spectrogram-plot:

===============================
Plotting a :class:`Spectrogram`
===============================

Like the :class:`~gwpy.timeseries.TimeSeries` and
:class:`~gwpy.frequencyseries.FrequencySeries`, the :class:`Spectrogram` has a
convenient :meth:`~Spectrogram.plot` method, allowing us to create views of
the daa.
We can extend the previous snippet to include a plot:

.. plot::
    :context:
    :include-source:
    :caption: Plotting a :class:`Spectrogram`

    plot = specgram.plot(norm='log', vmin=5e-24, vmax=1e-19)
    ax = plot.gca()
    ax.set_yscale('log')
    ax.set_ylim(10, 2000)
    ax.colorbar(
        label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')
    plot.show()

=================================
:class:`Spectrogram` applications
=================================

.. toctree::
   :titlesonly:

   ../spectrum/filtering

=============
Reference/API
=============

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   Spectrogram
