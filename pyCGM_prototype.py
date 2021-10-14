import numpy as np
from utils import pycgmIO
from numpy import array, random
import os
os.chdir('/Users/nilesh12/PyCGM_Prototypes/')
print(os.getcwd())
data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')

class pycgm():
    vsk = {}
    def __init__(self, measurements, markers):
        # marker data is flattened [xyzxyz...] per frame
        self.vsk = dict(zip(measurements[0], measurements[1]))
        # list of marker names
        self.marker_keys = markers[0].keys()
        if 'LFHD' in markers:
            ok = 'ok'
        markers_0= markers[0]
        self.marker_mapping = {marker_key: slice(index * 3, index * 3 + 3, 1) for index, marker_key in
                               enumerate(self.marker_keys)}

        # structured marker data
        self.marker_struct = self.structure_marker_data(markers)

        # some struct helper attributes
        self.num_frames = len(self.marker_struct)
        self.num_axes = len(self.axis_result_keys)
        self.num_floats_per_frame = self.num_axes * 16
        self.axis_results_shape = (self.num_frames, self.num_axes, 4, 4)

    def structure_marker_data(self, marker_frames):
        # takes flat marker data xyzxyzxyzxyz
        # returns a structured array that allows marker [x, y, z]
        # arrays to be retrieved by marker_struct[optional frame index or slice][marker name]

        marker_row_dtype = np.dtype([(name, '3f8') for name in self.marker_keys])
        marker_data_dtype = np.dtype((marker_row_dtype))

        return np.array([tuple(list(frame.values())) for frame in marker_frames], dtype=marker_row_dtype)

    def calc_axis_pelvis(self,rasi, lasi, rpsi, lpsi, sacr=None):
        if sacr is None:
            sacr = (rpsi + lpsi) / 2.0

        # REQUIRED LANDMARKS:
        # sacrum

        # Origin is Midpoint between RASI and LASI
        o = (rasi + lasi) / 2.0

        b1 = o - sacr
        b2 = lasi - rasi

        # y is normalized b2
        y = b2 / np.linalg.norm(b2)

        b3 = b1 - (np.dot(b1, y) * y)
        x = b3 / np.linalg.norm(b3)

        # Z-axis is cross product of x and y vectors.
        z = np.cross(x, y)

        pelvis = np.zeros((4, 4))
        pelvis[3, 3] = 1.0
        pelvis[0, :3] = x
        pelvis[1, :3] = y
        pelvis[2, :3] = z
        pelvis[:3, 3] = o

        return pelvis


    def JointAngleCalc(self, frame):
        # print(frame['sgfh'])
        rasi = frame.keys()
        pelvis_axis = self.calc_axis_pelvis(frame['RASI'] if 'RASI' in frame else None,
                                       frame['LASI'] if 'LASI' in frame else None,
                                       frame['RPSI'] if 'RPSI' in frame else None,
                                       frame['LPSI'] if 'LPSI' in frame else None,
                                       frame['SACR'] if 'SACR' in frame else None)
        RightKneeWidth = self.vsk['RightKneeWidth']
        # print(RightKneeWidth)
        return pelvis_axis

    def run(self):
        # First Calculate Pelvis
        for frame in data:
            pelvis_axis = self.JointAngleCalc(frame)

    @property
    def angle_result_keys(self):

        # list of default angle result names

        return ['Pelvis', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Ankle',
                'L Ankle', 'R Foot', 'L Foot',
                'Head', 'Thorax', 'Neck', 'Spine', 'R Shoulder', 'L Shoulder',
                'R Elbow', 'L Elbow', 'R Wrist', 'L Wrist']

    @property
    def axis_result_keys(self):
        # list of default axis result names

        return ['Pelvis', 'Hip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head',
                'Thorax', 'RClav', 'LClav', 'RHum', 'LHum', 'RRad', 'LRad', 'RHand', 'LHand']

nil = pycgm(measurements,data)
mesure = nil.vsk
nil.run()
