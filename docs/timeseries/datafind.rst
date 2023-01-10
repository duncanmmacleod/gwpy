.. currentmodule:: gwpy.timeseries

.. _gwpy-timeseries-datafind:

########################
Automatic data-discovery
########################

.. _gwpy-timeseries-datafind-intro:

The :meth:`TimeSeries.fetch_open_data` method is only able to download
GW strain data from those datasets exposed through the |GWOSC|_ API;
this notably does not include the |GWOSC_AUX_RELEASE|_, or strain data
for any events or observing runs not yet published.

In addition, the GW strain data make up only a tiny fraction of the 'raw'
output of a gravitational-wave detector, which includes in excess of
100,000 different 'channels' (data streams) from sensors and digital control
systems that are used to operate the interferometer and diagnose measure
performance.

The full data set for each detector is archived at the relevant observatory
and is made freely available to all registered collaboration members.
A discovery service called |gwdatafind|_ is provided at each location to
simplify discovering the file(s) that contain the data of interest for any
research.

These data are also made available remotely using |nds2|_, which enables
sending data directly over a network to any location.
This is used for both the full proprietary data (which requires an
authorisation credential to access) and also the |GWOSC_AUX_RELEASE|_
(which is freely available).

.. _gwpy-timeseries-get:

======================
:meth:`TimeSeries.get`
======================

**Additional dependencies:** |LDAStools.frameCPP|_ or |nds2|_

GWpy provides the :meth:`TimeSeries.get` method as a one-stop interface
to all automatically-discoverable data hosted locally at an IGWN
computing centre, or available remotely.

------------
How it works
------------

Without any customisation, :meth:`TimeSeries.get` will attempt to locate
data 'by any means necessary'; in practice that is

- if the ``GWDATAFIND_SERVER`` environment variable (or legacy
  ``LIGO_DATAFIND_SERVER`` variable) points to a server URL,
  use |gwdatafind|_ to identify which data set includes
  the channel(s) requested, then locate the files for that data set,
  and then read them,
- if that doesn't work (for any reason), loop through the NDS2 servers
  identified in the ``NDSSERVER`` environment variable to see if they
  have the data -- if the ``NDSSERVER`` variable isn't set, guess
  which NDS2 server(s) to use based on the interferometer whose data was
  requested.

.. admonition:: Regarding channel names

   To use :meth:`TimeSeries.get`, you need to know the full name of the
   data channel you want, which is often not obvious.
   The |GWOSC_AUX_RELEASE|_ includes a link to a full listing of all
   included channels.
   For the full proprietary data set, the IGWN Detector
   Characterisation working group maintains a record of the most relevant
   channels for studying a given interferometer subsystem.

.. _gwpy-timeseries-get-example:

-------
Example
-------

For example, to channel that records the power incident on the
input mode cleaner (IMC) for the 'H1' instrument at LIGO-Hanford is called:

.. code-block:: text

   H1:IMC-PWR_IN_OUT_DQ

We can use :meth:`TimeSeries.get` to 'get' the data for that
channel by specifying the special GWOSC NDS2 server url using the
``host`` keyword:

.. plot::
    :context: reset
    :caption: Get H1 power data from GWOSC using NDS2

    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries
    gps = event_gps("GW170814")
    start = int(gps) - 100
    end = int(gps) + 100
    data = TimeSeries.get("H1:IMC-PWR_IN_OUT_DQ", start, end, host="losc-nds.ligo.org")
    plot = data.plot(ylabel="Power [W]")
    plot.show()

.. _gwpy-timeseries-datafind-datasets:

==================
Available datasets
==================

All data archived at an IGWN computing centre are identified by a data
set 'tag', which identifies which data are contained in a given ``gwf``
file(s).
By default, as described, :meth:`TimeSeries.get` will search through all
available data to find the correct files to read, so this may take a
while if the server has knowledge of a large number of different datasets.
If you know the dataset name -- the tag associated with files containing your
data -- you can pass that via the ``frametype`` keyword argument to
significantly speed up the search.

Different computing centres archive different datasets, so what data
are available is highly-dependent on the location.
To discover which datasets are available using the default GWDataFind server:

.. code-block:: python
    :caption: Listing datasets with GWDataFind

    from gwdatafind import find_types
    print(find_types())

.. note::

    GWDataFind does not provide any documentation for data type; it can
    be hard to determine the purpose or contents of a dataset just from
    its name, see below for some special cases.

-----------------------
Publicly available data
-----------------------

Data published through |GWOSCl|_ are distributed using
`CVMFS <https://cernvm.cern.ch/>`__, and can be discovered using
:mod:`gwdatafind` via the ``https://datafind.gw-openscience.org`` server URL:

.. code-block:: python
    :caption: Listing GWOSC datasets with GWDataFind

    from gwdatafind import find_types
    >>> print(find_types("L", host="datafind.gw-openscience.org"))
    ['L1_GWOSC_O3a_16KHZ_R1',
     'L1_GWOSC_O2_16KHZ_R1',
     'L1_GWOSC_O3a_4KHZ_R1',
     'L1_GWOSC_O2_4KHZ_R1',
     'L1_LOSC_16_V1',
     'L1_LOSC_4_V1',
     'L1_GWOSC_O3b_16KHZ_R1',
     'L1_GWOSC_O3b_4KHZ_R1',
    ]

File URLs for specific times can be discovered using the
:external+gwdatafind:func:`gwdatafind.find_urls` function:

.. code-block:: python
    :caption: Listing GWOSC datasets with GWDataFind

    from gwdatafind import find_urls
    >>> print(find_urls(
    ...     "L",
    ...     "L1_GWOSC_O3a_16KHZ_R1",
    ...     1238163456,
    ...     1238163466,
    ...     host="datafind.gw-openscience.org",
    ... ))
    ['file://localhost/cvmfs/gwosc.osgstorage.org/gwdata/O3a/strain.16k/frame.v1/L1/1237319680/L-L1_GWOSC_O3a_16KHZ_R1-1238163456-4096.gwf']

If CVMFS is properly configured this file can be read directly using
:meth:`TimeSeries.read` similarly to any other data.

For more details on configuring CVMFS to read GWOSC data, see

https://computing.docs.ligo.org/guide/cvmfs/#gwosc.osgstorage.org

----------------
Proprietary data
----------------

Proprietary data for the current generation of gravitational wave observatories
are distributed between various computing centres and a restricted CVMFS
distribution, and can all be discovered using GWDataFind.

For more details on access to proprietary data, see

https://computing.docs.ligo.org/guide/data/

---------------
LIGO trend data
---------------

The LIGO observatories produce second- and minute- trends of all channels
automatically, and store them in the ``{H,L}1_T`` (second) and ``{H,L}1_M``
(minute) datasets.
However, **the channels in each trend type have the same names**, so
:meth:`TimeSeries.get` doesn't know how to distinguish between the two
different trends when given only the channel name.

To get around this you can directly specify (e.g.) ``frametype="H1_T"``
(for the LIGO-Hanford second trends) in your :meth:`TimeSeries.get`
method call, or you can use a suffix in the channel name:

.. table:: Channel name suffices for LIGO trends
   :name: gwpy-timeseries-datafind-trend-types

   ==========  ============  ===========
   Trend type  Dataset       Suffix
   ==========  ============  ===========
   second      ``{H,L}1_T``  ``,s-trend``
   minute      ``{H,L}1_M``  ``,m-trend``
   ==========  ============  ===========

e.g.

.. code-block:: python
    :caption: Accessing minute-trend data using :meth:`TimeSeries.get`

    TimeSeries.get("L1:IMC-PWR_IN_OUT_DQ.mean,s-trend", 1186741850, 1186741870)

will specifically access the second trends of power incident on the
LIGO-Livingston IMC.

==========================
:meth:`TimeSeriesDict.get`
==========================

:meth:`TimeSeries.get` can only retrieve data for a single channel at a time.
Looping over a list of names to get data for many channels can be very slow,
as each individual call will have to discover and read/download the data for
each channel individually.
The :meth:`TimeSeriesDict.get` method enables retrieval of multiple channels
in a single call, for a single ``(start, stop)`` time interval, greatly
reducing the I/O overhead.

To access data for multiple channels in this way, just pass a `list` of names
rather than a single name.
In this example we download the second trend (average) of ground motion in
the 0.03Hz-0.1Hz range at two locations of the LIGO-Hanford observatory:

.. warning::

   This example uses proprietary data that are only available to members
   of the LIGO Scientific Collaboration and its partners.

.. plot::
   :context: reset

   >>> from gwpy.timeseries import TimeSeriesDict
   >>> data = TimeSeriesDict.get(
   ...     ["H1:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M.rms,s-trend",
   ...      "H1:ISI-GND_STS_ETMY_Z_BLRMS_30M_100M.rms,s-trend"],
   ...     "July 22 2021 12:00",
   ...     "July 22 2021 14:00",
   ... )
   >>> plot = data.plot(ylabel="Ground motion [nm/s]")
   >>> plot.show()
