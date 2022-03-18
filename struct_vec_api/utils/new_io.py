import numpy as np
import time
from . import c3dpy3 as c3d
import xml.etree.ElementTree as ET


def frame_dtype():
    point = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
    return [('frame', 'f8'), ('point', point)]


def loadDataNew(filename):
    """Loads motion capture data from a c3d file.

    Parameters
    ----------
    filename : str
        Path of the c3d file to be loaded.

    Returns
    -------
    data : array
        `data` is a list of dict. Each dict represents one frame in
        the trial.

    Examples
    --------
    RoboResults.c3d in SampleData are used to
    test the output.

    >>> csvFile = 'SampleData/ROM/Sample_Static.csv'
    >>> c3dFile = 'SampleData/Sample_2/RoboStatic.c3d'
    >>> csvData = loadData(csvFile)
    SampleData/ROM/Sample_Static.csv
    >>> c3dData = loadData(c3dFile)
    SampleData/Sample_2/RoboStatic.c3d

    Testing for some values from the loaded csv file.

    >>> csvData[0]['RHNO'] #doctest: +NORMALIZE_WHITESPACE
    array([ 811.9591064,  677.3413696, 1055.390991 ])
    >>> csvData[0]['C7'] #doctest: +NORMALIZE_WHITESPACE
    array([ 250.765976,  165.616333, 1528.094116])

    Testing for some values from the loaded c3d file.

    >>> c3dData[0]['RHNO'] #doctest: +NORMALIZE_WHITESPACE
    array([-259.45016479, -844.99560547, 1464.26330566])
    >>> c3dData[0]['C7'] #doctest: +NORMALIZE_WHITESPACE
    array([-2.20681717e+02, -1.07236075e+00, 1.45551550e+03])
    """
    # print(filename)
    start = time.time()

    reader = c3d.Reader(open(filename, 'rb'))
    labels = reader.get('POINT:LABELS').string_array
    marker_names = [str(label.rstrip()) for label in labels]

    frames_list = np.array(list(reader.read_frames(True, True, yield_frame_no=False)), dtype=object)

    num_markers = len(frames_list[0][0])
    num_frames = len(frames_list)
    frame_numbers = np.arange(1, num_frames + 1)

    marker_positions = np.empty((num_markers, num_frames), dtype=(("4f8")))
    marker_xyz = [(key, (frame_dtype(), (num_frames,))) for key in marker_names]

    float_arr = np.vstack(frames_list[:, 0]).astype(np.float).reshape(num_markers, num_frames, 3)

    marker_positions = np.insert(float_arr, 0, frame_numbers, axis=2)
    marker_positions.dtype = frame_dtype()

    dynamic_struct = np.empty((1), dtype=marker_xyz)

    for i, name in enumerate(dynamic_struct.dtype.names):
        dynamic_struct[name][0][:, np.newaxis] = marker_positions[i]


    end = time.time()
    print(f'Time to read/structure {filename}: {end - start}')
    return dynamic_struct


def loadVSK(filename, dict=True):
    """Open and load a vsk file.

    Parameters
    ----------
    filename : str
        Path to the vsk file to be loaded
    dict : bool, optional
        Returns loaded vsk file values as a dictionary if False.
        Otherwise, return as an array.

    Returns
    -------
    [vsk_keys, vsk_data] : array
        `vsk_keys` is a list of labels. `vsk_data` is a list of values
        corresponding to the labels in `vsk_keys`.

    Examples
    --------
    RoboSM.vsk in SampleData is used to test the output.

    >>> filename = 'SampleData/Sample_2/RoboSM.vsk'
    >>> result = loadVSK(filename)
    >>> vsk_keys = result[0]
    >>> vsk_data = result[1]
    >>> vsk_keys
    ['Bodymass', 'Height', 'InterAsisDistance',...]
    >>> vsk_data
    [72.0, 1730.0, 281.118011474609,...]

    Return as a dictionary.

    >>> result = loadVSK(filename, False)
    >>> type(result)
    <...'dict'>

    Testing for some dictionary values.

    >>> result['Bodymass']
    72.0
    >>> result['RightStaticPlantFlex']
    0.17637075483799
    """
    # Check if the filename is valid
    # if not, return None
    if filename == '':
        return None

    # Create an XML tree from file
    tree = ET.parse(filename)

    # Get the root of the file
    # <KinematicModel>
    root = tree.getroot()

    # Store the values of each parameter in a dictionary
    # the format is (NAME,VALUE)
    vsk_keys = [r.get('NAME') for r in root[0]]
    vsk_data = []
    for R in root[0]:
        val = (R.get('VALUE'))
        if val == None:
            val = 0
        vsk_data.append(float(val))

    # print vsk_keys
    if dict == False:
        vsk = {}
        for key, data in zip(vsk_keys, vsk_data):
            vsk[key] = data
        return vsk

    return [vsk_keys, vsk_data]

