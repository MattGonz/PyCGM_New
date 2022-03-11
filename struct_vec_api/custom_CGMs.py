from pyCGM import Subject
import numpy as np

class Subject_CustomPelvis(Subject):
    def __init__(self, static_trial, dynamic_trials, measurements):
        super().__init__(static_trial, dynamic_trials, measurements)
        self.modify_function('calc_axis_pelvis', measurements=["Bodymass", "ImaginaryMeasurement"],
                                                 markers=["RASI", "LASI", "RPSI", "LPSI", "SACR"],
                                                 returns_axes=['Pelvis'])

    def calc_axis_pelvis(self, bodymass, imaginary_measurement, rasi, lasi, rpsi, lpsi, sacr):
        """
        Make the Pelvis Axis.
        """

        # Verify that the input data is the correct shape
        # print(f"{rasi.shape=}")
        # print(f"{lasi.shape=}")
        # print(f"{rpsi.shape=}")
        # print(f"{lpsi.shape=}")

        # Print to verify overridden function is being called
        # print("\n\tCustomPelvis called instead") 
        # print(f"\t\t{bodymass=},\n\t\t{imaginary_measurement=},\n\t\t{rasi.shape=},\n\t\t{lasi.shape=},\n\t\t{rpsi.shape=},\n\t\t{lpsi.shape=},\n\t\t{sacr=}\n")

        # Get the Pelvis Joint Centre
        if sacr is None:
            sacr = (rpsi + lpsi) / 2.0

        # Origin is Midpoint between RASI and LASI
        o = (rasi+lasi)/2.0

        b1 = o - sacr
        b2 = lasi - rasi

        # y is normalized b2
        y = b2 / np.linalg.norm(b2,axis=1)[:, np.newaxis]

        b3 = b1 - ( y * np.sum(b1*y,axis=1)[:, np.newaxis] )
        x = b3/np.linalg.norm(b3,axis=1)[:, np.newaxis]

        # Z-axis is cross product of x and y vectors.
        z = np.cross(x, y)

        new_stack_col = np.column_stack([x,y,z,o])

        return new_stack_col

