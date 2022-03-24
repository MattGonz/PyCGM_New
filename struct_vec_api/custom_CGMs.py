from model import Model
import numpy as np

class Model_CustomPelvis(Model):
    def __init__(self, static_trial, dynamic_trials, measurements):
        super().__init__(static_trial, dynamic_trials, measurements)

        # Override the calc_axis_pelvis function in Model
        self.modify_function('calc_axis_pelvis', measurements=["Bodymass", "ImaginaryMeasurement"],
                                                      markers=["RASI", "LASI", "RPSI", "LPSI", "SACR"],
                                                 returns_axes=['Pelvis'])


    def calc_axis_pelvis(self, bodymass, imaginary_measurement, rasi, lasi, rpsi, lpsi, sacr):
        """
        Make the Pelvis Axis.
        """

        num_frames = rasi.shape[0]
        x = np.zeros((num_frames, 3))
        y = np.zeros((num_frames, 3))
        z = np.zeros((num_frames, 3))
        o = np.zeros((num_frames, 3))

        pel_axis_stack = np.column_stack([x,y,z,o])
        pel_axis_matrix = pel_axis_stack.reshape(num_frames,4,3).transpose(0,2,1)
        # [ xx xy xz xo ] = pel_axis_matrix[0]
        # [ yx yy yz yo ] = pel_axis_matrix[1]
        # [ zx zy zz zo ] = pel_axis_matrix[2]

        return pel_axis_matrix


class Model_NewFunction(Model):
    def __init__(self, static_trial, dynamic_trials, measurements):
        super().__init__(static_trial, dynamic_trials, measurements)

        # Add a custom function to the Model
        self.add_function('calc_axis_eye', measurements=["Bodymass", "HeadOffset"],
                                                markers=["RFHD", "LFHD", "RBHD", "LBHD"],
                                                   axes=["Head"],
                                           returns_axes=['REye', 'LEye'],
                                                  order=['calc_axis_head', 1]) 

    def calc_axis_eye(self, bodymass, head_offset, rfhd, lfhd, rbhd, lbhd, head_axis):
        """
        Make the Eye Axis.
        """

        num_frames = rfhd.shape[0]
        x = np.zeros((num_frames, 3))
        y = np.zeros((num_frames, 3))
        z = np.zeros((num_frames, 3))
        o = np.zeros((num_frames, 3))

        r_eye_axis_stack = np.column_stack([x,y,z,o])
        l_eye_axis_stack = np.column_stack([x,y,z,o])

        r_eye_axis_matrix = r_eye_axis_stack.reshape(num_frames,4,3).transpose(0,2,1)
        l_eye_axis_matrix = l_eye_axis_stack.reshape(num_frames,4,3).transpose(0,2,1)
        # [ xx xy xz xo ] = [r/l]_eye_axis_matrix[0]
        # [ yx yy yz yo ] = [r/l]_eye_axis_matrix[1]
        # [ zx zy zz zo ] = [r/l]_eye_axis_matrix[2]

        return np.array([r_eye_axis_matrix, l_eye_axis_matrix])

