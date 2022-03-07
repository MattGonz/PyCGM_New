import numpy as np
import math
from math import pi, sin, cos, radians


class CalcAxes():

    def __init__(self):
        self.funcs = [self.pelvis_axis, self.hip_axis, self.knee_axis, self.ankle_axis, self.foot_axis,
                      self.head_axis, self.thorax_axis, self.clav_axis, self.hum_axis, self.rad_axis, self.hand_axis]

    def pelvis_axis(self, rasi, lasi, rpsi, lpsi, sacr=None):
        # get the refactored 4x4 pelvis axis

        if rpsi is not None and lpsi is not None:
            sacrum = (rpsi + lpsi)/2.0
        else:
            sacrum = sacr

        origin = (rasi+lasi)/2.0
        beta1 = origin - sacrum
        beta2 = lasi - rasi
        y_axis = beta2 / np.linalg.norm(beta2)
        beta3_cal = np.dot(beta1, y_axis)
        beta3_cal2 = beta3_cal * y_axis
        beta3 = beta1-beta3_cal2
        x_axis = beta3/np.linalg.norm(beta3)
        z_axis = np.cross(x_axis, y_axis)

        # I added these back to check for correctness
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin

        pelvis = np.zeros((4, 4))
        pelvis[3, 3] = 1.0
        pelvis[0, :3] = x_axis
        pelvis[1, :3] = y_axis
        pelvis[2, :3] = z_axis
        pelvis[:3, 3] = origin

        return pelvis

    def hip_axis(self, r_hip_jc, l_hip_jc, pelvis_axis):
        # Get shared hip axis, it is inbetween the two hip joint centers
        hipaxis_center = (np.asarray(r_hip_jc) + np.asarray(l_hip_jc)) / 2.0

        # convert pelvis_axis to x,y,z axis to use more easy
        pelvis_x_axis = pelvis_axis[0, :3]
        pelvis_y_axis = pelvis_axis[1, :3]
        pelvis_z_axis = pelvis_axis[2, :3]

        axis = np.zeros((4, 4))
        axis[3, 3] = 1.0
        axis[0, :3] = pelvis_x_axis
        axis[1, :3] = pelvis_y_axis
        axis[2, :3] = pelvis_z_axis
        axis[:3, 3] = hipaxis_center

        return axis

    def knee_axis(self, rthi, lthi, rkne, lkne, hip_jc, rkne_width, lkne_width):
        # Get Global Values
        mm = 7.0
        R_delta = (rkne_width/2.0) + mm
        L_delta = (lkne_width/2.0) + mm

        r_hip_jc = hip_jc[1][:3, 3]
        l_hip_jc = hip_jc[0][:3, 3]

        # Determine the position of kneeJointCenter using findJointC function
        R = CalcUtils.find_joint_center(rthi, r_hip_jc, rkne, R_delta)
        L = CalcUtils.find_joint_center(lthi, l_hip_jc, lkne, L_delta)

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = r_hip_jc-R

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector.
        # the axis is then normalized.
        # axis_x = cross(axis_z,thi_kne_R)
        axis_x = np.cross(axis_z, rkne-r_hip_jc)

        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        Raxis = np.asarray([axis_x, axis_y, axis_z])

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = l_hip_jc-L

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector.
        # the axis is then normalized.
        # axis_x = cross(thi_kne_L,axis_z)
        # using hipjc instead of thigh marker
        axis_x = np.cross(lkne-l_hip_jc, axis_z)

        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        Laxis = np.asarray([axis_x, axis_y, axis_z])

        # Clear the name of axis and then nomalize it.
        R_knee_x_axis = Raxis[0]
        R_knee_x_axis = R_knee_x_axis/np.linalg.norm(R_knee_x_axis)
        R_knee_y_axis = Raxis[1]
        R_knee_y_axis = R_knee_y_axis/np.linalg.norm(R_knee_y_axis)
        R_knee_z_axis = Raxis[2]
        R_knee_z_axis = R_knee_z_axis/np.linalg.norm(R_knee_z_axis)
        L_knee_x_axis = Laxis[0]
        L_knee_x_axis = L_knee_x_axis/np.linalg.norm(L_knee_x_axis)
        L_knee_y_axis = Laxis[1]
        L_knee_y_axis = L_knee_y_axis/np.linalg.norm(L_knee_y_axis)
        L_knee_z_axis = Laxis[2]
        L_knee_z_axis = L_knee_z_axis/np.linalg.norm(L_knee_z_axis)

        r_axis = np.zeros((4, 4))
        r_axis[3, 3] = 1.0
        r_axis[0, :3] = R_knee_x_axis
        r_axis[1, :3] = R_knee_y_axis
        r_axis[2, :3] = R_knee_z_axis
        r_axis[:3, 3] = R

        # Add the origin back to the vector
        y_axis = L_knee_y_axis
        z_axis = L_knee_z_axis
        x_axis = L_knee_x_axis
        Laxis = np.asarray([x_axis, y_axis, z_axis])

        l_axis = np.zeros((4, 4))
        l_axis[3, 3] = 1.0
        l_axis[0, :3] = L_knee_x_axis
        l_axis[1, :3] = L_knee_y_axis
        l_axis[2, :3] = L_knee_z_axis
        l_axis[:3, 3] = L

        axis = np.asarray([r_axis, l_axis])

        return axis

    def ankle_axis(self, rtib, ltib, rank, lank, knee_axis, rank_width, lank_width, rtib_torsion, ltib_torsion):
        # Get Global Values
        mm = 7.0
        R_delta = (rank_width/2.0)+mm
        L_delta = (lank_width/2.0)+mm

        # REQUIRED MARKERS:
        # tib_R
        # tib_L
        # ank_R
        # ank_L
        # knee_JC

        r_knee_axis, l_knee_axis = knee_axis
        r_knee_origin = r_knee_axis[:3, 3]
        l_knee_origin = l_knee_axis[:3, 3]

        # This is Torsioned Tibia and this describe the ankle angles
        # Tibial frontal plane being defined by ANK,TIB and KJC

        # Determine the position of ankleJointCenter using findJointC function
        R = CalcUtils.find_joint_center(rtib, r_knee_origin, rank, R_delta)
        L = CalcUtils.find_joint_center(ltib, l_knee_origin, lank, L_delta)

        # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        # Right axis calculation

        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = r_knee_origin-R

        # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
        # and calculated by each point's vector cross vector.
        # tib_ank_R vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_R = rtib-rank
        axis_x = np.cross(axis_z, tib_ank_R)

        # Y axis is determined by cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        Raxis = [axis_x, axis_y, axis_z]

        # Left axis calculation

        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = l_knee_origin-L

        # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
        # and calculated by each point's vector cross vector.
        # tib_ank_L vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_L = ltib-lank
        axis_x = np.cross(tib_ank_L, axis_z)

        # Y axis is determined by cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        Laxis = [axis_x, axis_y, axis_z]

        # Clear the name of axis and then normalize it.
        R_ankle_x_axis = Raxis[0]
        R_ankle_x_axis_div = np.linalg.norm(R_ankle_x_axis)
        R_ankle_x_axis = [R_ankle_x_axis[0]/R_ankle_x_axis_div, R_ankle_x_axis[1] /
                          R_ankle_x_axis_div, R_ankle_x_axis[2]/R_ankle_x_axis_div]

        R_ankle_y_axis = Raxis[1]
        R_ankle_y_axis_div = np.linalg.norm(R_ankle_y_axis)
        R_ankle_y_axis = [R_ankle_y_axis[0]/R_ankle_y_axis_div, R_ankle_y_axis[1] /
                          R_ankle_y_axis_div, R_ankle_y_axis[2]/R_ankle_y_axis_div]

        R_ankle_z_axis = Raxis[2]
        R_ankle_z_axis_div = np.linalg.norm(R_ankle_z_axis)
        R_ankle_z_axis = [R_ankle_z_axis[0]/R_ankle_z_axis_div, R_ankle_z_axis[1] /
                          R_ankle_z_axis_div, R_ankle_z_axis[2]/R_ankle_z_axis_div]

        L_ankle_x_axis = Laxis[0]
        L_ankle_x_axis_div = np.linalg.norm(L_ankle_x_axis)
        L_ankle_x_axis = [L_ankle_x_axis[0]/L_ankle_x_axis_div, L_ankle_x_axis[1] /
                          L_ankle_x_axis_div, L_ankle_x_axis[2]/L_ankle_x_axis_div]

        L_ankle_y_axis = Laxis[1]
        L_ankle_y_axis_div = np.linalg.norm(L_ankle_y_axis)
        L_ankle_y_axis = [L_ankle_y_axis[0]/L_ankle_y_axis_div, L_ankle_y_axis[1] /
                          L_ankle_y_axis_div, L_ankle_y_axis[2]/L_ankle_y_axis_div]

        L_ankle_z_axis = Laxis[2]
        L_ankle_z_axis_div = np.linalg.norm(L_ankle_z_axis)
        L_ankle_z_axis = [L_ankle_z_axis[0]/L_ankle_z_axis_div, L_ankle_z_axis[1] /
                          L_ankle_z_axis_div, L_ankle_z_axis[2]/L_ankle_z_axis_div]

        # Put both axis in array
        Raxis = [R_ankle_x_axis, R_ankle_y_axis, R_ankle_z_axis]
        Laxis = [L_ankle_x_axis, L_ankle_y_axis, L_ankle_z_axis]

        # Rotate the axes about the tibia torsion.
        rtib_torsion = np.radians(rtib_torsion)
        ltib_torsion = np.radians(ltib_torsion)

        Raxis = [[math.cos(rtib_torsion)*Raxis[0][0]-math.sin(rtib_torsion)*Raxis[1][0],
                  math.cos(rtib_torsion)*Raxis[0][1] -
                  math.sin(rtib_torsion)*Raxis[1][1],
                  math.cos(rtib_torsion)*Raxis[0][2]-math.sin(rtib_torsion)*Raxis[1][2]],
                 [math.sin(rtib_torsion)*Raxis[0][0]+math.cos(rtib_torsion)*Raxis[1][0],
                 math.sin(rtib_torsion)*Raxis[0][1] +
                  math.cos(rtib_torsion)*Raxis[1][1],
                 math.sin(rtib_torsion)*Raxis[0][2]+math.cos(rtib_torsion)*Raxis[1][2]],
                 [Raxis[2][0], Raxis[2][1], Raxis[2][2]]]

        Laxis = [[math.cos(ltib_torsion)*Laxis[0][0]-math.sin(ltib_torsion)*Laxis[1][0],
                  math.cos(ltib_torsion)*Laxis[0][1] -
                  math.sin(ltib_torsion)*Laxis[1][1],
                  math.cos(ltib_torsion)*Laxis[0][2]-math.sin(ltib_torsion)*Laxis[1][2]],
                 [math.sin(ltib_torsion)*Laxis[0][0]+math.cos(ltib_torsion)*Laxis[1][0],
                 math.sin(ltib_torsion)*Laxis[0][1] +
                  math.cos(ltib_torsion)*Laxis[1][1],
                 math.sin(ltib_torsion)*Laxis[0][2]+math.cos(ltib_torsion)*Laxis[1][2]],
                 [Laxis[2][0], Laxis[2][1], Laxis[2][2]]]

        r_axis = np.zeros((4, 4))
        r_axis[3, 3] = 1.0
        r_axis[0, :3] = Raxis[0]
        r_axis[1, :3] = Raxis[1]
        r_axis[2, :3] = Raxis[2]
        r_axis[:3, 3] = R

        l_axis = np.zeros((4, 4))
        l_axis[3, 3] = 1.0
        l_axis[0, :3] = Laxis[0]
        l_axis[1, :3] = Laxis[1]
        l_axis[2, :3] = Laxis[2]
        l_axis[:3, 3] = L

        # Both of axis in array.
        axis = [r_axis, l_axis]

        return axis

    def foot_axis(self):
        return np.zeros((2, 4, 4))

    def head_axis(self):
        return np.zeros((4, 4))

    def thorax_axis(self, clav, c7, strn, t10):
        clav, c7, strn, t10 = map(np.asarray, [clav, c7, strn, t10])

        # Set or get a marker size as mm
        marker_size = (14.0) / 2.0

        # Get the midpoints of the upper and lower sections, as well as the front and back sections
        upper = (clav + c7)/2.0
        lower = (strn + t10)/2.0
        front = (clav + strn)/2.0
        back = (t10 + c7)/2.0

        # Get the direction of the primary axis Z (facing down)
        z_direc = lower - upper
        z = z_direc/np.linalg.norm(z_direc)

        # The secondary axis X is from back to front
        x_direc = front - back
        x = x_direc/np.linalg.norm(x_direc)

        # make sure all the axes are orthogonal to each other by cross-product
        y_direc = np.cross(z, x)
        y = y_direc/np.linalg.norm(y_direc)
        x_direc = np.cross(y, z)
        x = x_direc/np.linalg.norm(x_direc)
        z_direc = np.cross(x, y)
        z = z_direc/np.linalg.norm(z_direc)

        # move the axes about offset along the x axis.
        offset = x * marker_size

        # Add the CLAV back to the vector to get it in the right position before translating it
        o = clav - offset

        thorax = np.zeros((4, 4))
        thorax[3, 3] = 1.0
        thorax[0, :3] = x
        thorax[1, :3] = y
        thorax[2, :3] = z
        thorax[:3, 3] = o

        return thorax

    def clav_axis(self):
        return np.zeros((2, 4, 4))

    def hum_axis(self, relb, lelb, rwra, rwrb, lwra, lwrb, shoulder_jc, r_elbow_width, l_elbow_width, r_wrist_width, l_wrist_width, mm):
        """Calculate the Elbow joint axis (Humerus) function.

        Takes in markers that correspond to (x, y, z) positions of the current
        frame, the shoulder joint center, elbow widths, wrist widths, and the
        marker mm size.

        Calculates each elbow joint axis.

        Markers used: relb, lelb, rwra, rwrb, lwra, lwrb.

        Subject Measurement values used: r_elbow_width, l_elbow_width, r_wrist_width,
        l_wrist_width.

        Parameters
        ----------
        relb : array
            1x3 RELB marker
        lelb : array
            1x3 LELB marker
        rwra : array
            1x3 RWRA marker
        rwrb : array
            1x3 RWRB marker
        lwra : array
            1x3 LWRA marker
        lwrb : array
            1x3 LWRB marker
        shoulder_jc : ndarray
            A 4x4 identity matrix that holds the shoulder joint_center
        r_elbow_width : float
            The width of the right elbow
        l_elbow_width : float
            The width of the left elbow
        r_wrist_width : float
            The width of the right wrist
        l_wrist_width : float
            The width of the left wrist
        mm : float
            The thickness of the marker in millimeters.

        Returns
        -------
        [r_axis, l_axis, np.array([r_wri_origin, l_wri_origin])] : array
        An array with three items consisting of a 4x4 affine matrix representing the
        right elbow axis, a 4x4 affine matrix representing the left elbow axis, and
        a list of the right wrist origin and the left wrist origin.

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAxes
        >>> np.set_printoptions(suppress=True)
        >>> shoulder_jc = [np.array([[1., 0., 0., 429.66],
        ...   [0., 1., 0.,  275.06],
        ...   [0., 0., 1., 1453.95],
        ...   [0., 0., 0.,    1.  ]]),
        ...   np.array([[1., 0., 0., 64.51],
        ...   [0., 1., 0., 274.93],
        ...   [0., 0., 1.,1463.63],
        ...   [0., 0., 0.,   1.  ]])]
        >>> [np.around(arr, 2) for arr in CalcAxes().hum_axis(
        ... np.array([658.90, 326.07, 1285.28]), # RELB marker
        ... np.array([-156.32, 335.25, 1287.39]), # LELB marker
        ... np.array([776.51,495.68, 1108.38]), # RWRA marker
        ... np.array([830.90, 436.75, 1119.11]), # RWRB marker
        ... np.array([-249.28, 525.32, 1117.09]), # LWRA marker
        ... np.array([-311.77, 477.22, 1125.16]), # LWRB marker
        ... shoulder_jc,
        ... 74.0, 74.0, 55.0, 55.0, 7.0)] #doctest: +NORMALIZE_WHITESPACE
        [array([[   0.14,   -0.99,   -0.  ,  633.66],
                [   0.69,    0.1 ,    0.72,  304.95],
                [  -0.71,   -0.1 ,    0.69, 1256.07],
                [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.15,   -0.99,   -0.06, -129.16],
                [   0.72,   -0.07,   -0.69,  316.86],
                [   0.68,   -0.15,    0.72, 1258.06],
                [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[[   1.  ,    0.  ,    0.  ,  793.32],
                [   0.  ,    1.  ,    0.  ,  451.29],
                [   0.  ,    0.  ,    1.  , 1084.43],
                [   0.  ,    0.  ,    0.  ,    1.  ]],
                [[   1.  ,    0.  ,    0.  , -272.46],
                [   0.  ,    1.  ,    0.  ,  485.79],
                [   0.  ,    0.  ,    1.  , 1091.37],
                [   0.  ,    0.  ,    0.  ,    1.  ]]])]
        """

        r_elbow_width *= -1
        r_delta = (r_elbow_width/2.0)-mm
        l_delta = (l_elbow_width/2.0)+mm

        rwri = [(rwra[0]+rwrb[0])/2.0, (rwra[1]+rwrb[1]) /
                2.0, (rwra[2]+rwrb[2])/2.0]
        lwri = [(lwra[0]+lwrb[0])/2.0, (lwra[1]+lwrb[1]) /
                2.0, (lwra[2]+lwrb[2])/2.0]

        rsjc = shoulder_jc[0][:3, 3]
        lsjc = shoulder_jc[1][:3, 3]

        # make the construction vector for finding the elbow joint center
        r_con_1 = np.subtract(rsjc, relb)
        r_con_1_div = np.linalg.norm(r_con_1)
        r_con_1 = [r_con_1[0]/r_con_1_div, r_con_1[1] /
                   r_con_1_div, r_con_1[2]/r_con_1_div]

        r_con_2 = np.subtract(rwri, relb)
        r_con_2_div = np.linalg.norm(r_con_2)
        r_con_2 = [r_con_2[0]/r_con_2_div, r_con_2[1] /
                   r_con_2_div, r_con_2[2]/r_con_2_div]

        r_cons_vec = np.cross(r_con_1, r_con_2)
        r_cons_vec_div = np.linalg.norm(r_cons_vec)
        r_cons_vec = [r_cons_vec[0]/r_cons_vec_div, r_cons_vec[1] /
                      r_cons_vec_div, r_cons_vec[2]/r_cons_vec_div]

        r_cons_vec = [r_cons_vec[0]*500+relb[0], r_cons_vec[1]
                      * 500+relb[1], r_cons_vec[2]*500+relb[2]]

        l_con_1 = np.subtract(lsjc, lelb)
        l_con_1_div = np.linalg.norm(l_con_1)
        l_con_1 = [l_con_1[0]/l_con_1_div, l_con_1[1] /
                   l_con_1_div, l_con_1[2]/l_con_1_div]

        l_con_2 = np.subtract(lwri, lelb)
        l_con_2_div = np.linalg.norm(l_con_2)
        l_con_2 = [l_con_2[0]/l_con_2_div, l_con_2[1] /
                   l_con_2_div, l_con_2[2]/l_con_2_div]

        l_cons_vec = np.cross(l_con_1, l_con_2)
        l_cons_vec_div = np.linalg.norm(l_cons_vec)

        l_cons_vec = [l_cons_vec[0]/l_cons_vec_div, l_cons_vec[1] /
                      l_cons_vec_div, l_cons_vec[2]/l_cons_vec_div]

        l_cons_vec = [l_cons_vec[0]*500+lelb[0], l_cons_vec[1]
                      * 500+lelb[1], l_cons_vec[2]*500+lelb[2]]

        rejc = CalcUtils.find_joint_center(r_cons_vec, rsjc, relb, r_delta)
        lejc = CalcUtils.find_joint_center(l_cons_vec, lsjc, lelb, l_delta)

        # this is radius axis for humerus
        # right
        x_axis = np.subtract(rwra, rwrb)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        z_axis = np.subtract(rejc, rwri)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        y_axis = np.cross(z_axis, x_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        r_radius = [x_axis, y_axis, z_axis]

        # left
        x_axis = np.subtract(lwra, lwrb)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        z_axis = np.subtract(lejc, lwri)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        y_axis = np.cross(z_axis, x_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        l_radius = [x_axis, y_axis, z_axis]

        # calculate wrist joint center for humerus
        r_wrist_width = (r_wrist_width/2.0 + mm)
        l_wrist_width = (l_wrist_width/2.0 + mm)

        rwjc = [rwri[0]+r_wrist_width*r_radius[1][0], rwri[1] +
                r_wrist_width*r_radius[1][1], rwri[2]+r_wrist_width*r_radius[1][2]]
        lwjc = [lwri[0]-l_wrist_width*l_radius[1][0], lwri[1] -
                l_wrist_width*l_radius[1][1], lwri[2]-l_wrist_width*l_radius[1][2]]

        # recombine the humerus axis
        # right
        z_axis = np.subtract(rsjc, rejc)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        x_axis = np.subtract(rwjc, rejc)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        y_axis = np.cross(x_axis, z_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        r_axis = np.zeros((4, 4))
        r_axis[3, 3] = 1.0
        r_axis[0, :3] = x_axis
        r_axis[1, :3] = y_axis
        r_axis[2, :3] = z_axis
        r_axis[:3, 3] = rejc

        # left
        z_axis = np.subtract(lsjc, lejc)
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        x_axis = np.subtract(lwjc, lejc)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        y_axis = np.cross(x_axis, z_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        l_axis = np.zeros((4, 4))
        l_axis[3, 3] = 1.0
        l_axis[0, :3] = x_axis
        l_axis[1, :3] = y_axis
        l_axis[2, :3] = z_axis
        l_axis[:3, 3] = lejc

        r_wri_origin = np.identity(4)
        r_wri_origin[:3, 3] = rwjc

        l_wri_origin = np.identity(4)
        l_wri_origin[:3, 3] = lwjc

        return [r_axis, l_axis, np.array([r_wri_origin, l_wri_origin])]

    def rad_axis(self, elbow_jc):
        r"""Calculate the wrist joint axis (Radius) function.
        Takes in the elbow axis to calculate each wrist joint axis and returns it.

        Parameters
        ----------
        elbow_jc : array
            A list of three elements containing a 4x4 affine matrix representing the
            right elbow, a 4x4 affine matrix representing the left elbow, and a list
            of two 4x4 matrices representing the left and right wrist joint centers.

        Returns
        --------
        [r_axis, l_axis] : array
            A list of two 4x4 affine matrices representing the right hand axis as
            well as the left hand axis.

        Notes
        -----
        .. math::
            \begin{matrix}
                o_{L} = \textbf{m}_{LWJC} & o_{R} = \textbf{m}_{RWJC} \\
                \hat{y}_{L} = Elbow\_Flex_{L} & \hat{y}_{R} =  Elbow\_Flex_{R} \\
                \hat{z}_{L} = \textbf{m}_{LEJC} - \textbf{m}_{LWJC} & \hat{z}_{R} = \textbf{m}_{REJC} - \textbf{m}_{RWJC} \\
                \hat{x}_{L} = \hat{y}_{L} \times \hat{z}_{L} & \hat{x}_{R} = \hat{y}_{R} \times \hat{z}_{R} \\
                \hat{z}_{L} = \hat{x}_{L} \times \hat{y}_{L} & \hat{z}_{R} = \hat{x}_{R} \times \hat{y}_{R} \\
            \end{matrix}

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAxes
        >>> np.set_printoptions(suppress=True)
        >>> elbow_jc = [ np.array([[   0.15,   -0.99,    0.  ,  633.66],
        ...        [ 0.69,  0.1,  0.72,  304.95],
        ...        [-0.71, -0.1,  0.7 , 1256.07],
        ...        [ 0.  ,  0. ,  0.  ,    1.  ]]),
        ...        np.array([[  -0.16,   -0.98,   -0.06, -129.16],
        ...        [ 0.71, -0.07, -0.69,  316.86],
        ...        [ 0.67, -0.14,  0.72, 1258.06],
        ...        [ 0.  ,  0.  ,  0.  ,    1.  ]]),
        ...        np.array([[[1, 0, 0,  793.32],
        ...             [0, 1, 0,  451.29],
        ...             [0, 0, 1, 1084.43],
        ...             [0, 0, 0,    1.  ]],
        ...            [[1, 0, 0, -272.45],
        ...             [0, 1, 0,  485.8 ],
        ...             [0, 0, 1, 1091.36],
        ...             [0, 0, 0,    1.  ]]])
        ...    ]
        >>> [np.around(arr, 2) for arr in CalcAxes().rad_axis(elbow_jc)] #doctest: +NORMALIZE_WHITESPACE
        [array([[   0.44,   -0.84,   -0.31,  793.32],
            [   0.69,    0.1 ,    0.72,  451.29],
            [  -0.57,   -0.53,    0.62, 1084.43],
            [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.47,   -0.79,   -0.4 , -272.45],
            [   0.72,   -0.07,   -0.7 ,  485.8 ],
            [   0.52,   -0.61,    0.6 , 1091.36],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
        """
        # Bring Elbow joint center, axes and Wrist Joint Center for calculating Radius Axes

        rejc = elbow_jc[0][:3, 3]
        lejc = elbow_jc[1][:3, 3]

        r_elbow_flex = elbow_jc[0][1, :3]
        l_elbow_flex = elbow_jc[1][1, :3]

        rwjc = elbow_jc[2][0][:3, 3]
        lwjc = elbow_jc[2][1][:3, 3]

        # this is the axis of radius
        # right
        y_axis = r_elbow_flex
        y_axis = y_axis/np.linalg.norm(y_axis)

        z_axis = np.subtract(rejc, rwjc)
        z_axis = z_axis/np.linalg.norm(z_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)

        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)

        r_axis = np.zeros((4, 4))
        r_axis[3, 3] = 1.0
        r_axis[0, :3] = x_axis
        r_axis[1, :3] = y_axis
        r_axis[2, :3] = z_axis
        r_axis[:3, 3] = rwjc

        # left
        y_axis = l_elbow_flex
        y_axis = y_axis/np.linalg.norm(y_axis)

        z_axis = np.subtract(lejc, lwjc)
        z_axis = z_axis/np.linalg.norm(z_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)

        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)

        l_axis = np.zeros((4, 4))
        l_axis[3, 3] = 1.0
        l_axis[0, :3] = x_axis
        l_axis[1, :3] = y_axis
        l_axis[2, :3] = z_axis
        l_axis[:3, 3] = lwjc

        return [r_axis, l_axis]

    def hand_axis(self, rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_jc, r_hand_thickness, l_hand_thickness):
        r"""Calculate the Hand joint axis.

        Takes in markers that correspond to (x, y, z) positions of the current
        frame as well as the wrist joint center.

        Calculates each hand joint axis and returns it.

        Markers used: RWRA, RWRB, LWRA, LWRB, RFIN, LFIN

        Subject Measurement values used: RightHandThickness, LeftHandThickness

        Parameters
        ----------
        rwra : array
            1x3 RWRA marker
        rwrb : array
            1x3 RWRB marker
        lwra : array
            1x3 LWRA marker
        lwrb : array
            1x3 LWRB marker
        rfin : array
            1x3 RFIN marker
        lfin : array
            1x3 LFIN marker
        wrist_jc : array
            The x,y,z position of the wrist joint center.
        r_hand_thickness : float
            The thickness of the right hand.
        l_hand_thickness : float
            The thickness of the left hand.

        Returns
        -------
        [r_axis, l_axis] : array
            A list of two 4x4 affine matrices representing the right hand axis as well as the
            left hand axis.

        Notes
        -----
        :math:`r_{delta} = (\frac{r\_hand\_thickness}{2.0} + 7.0) \hspace{1cm} l_{delta} = (\frac{l\_hand\_thickness}{2.0} + 7.0)`

        :math:`\textbf{m}_{RHND} = JC(\textbf{m}_{RWRI}, \textbf{m}_{RWJC}, \textbf{m}_{RFIN}, r_{delta})`

        :math:`\textbf{m}_{LHND} = JC(\textbf{m}_{LWRI}, \textbf{m}_{LWJC}, \textbf{m}_{LFIN}, r_{delta})`

        .. math::

            \begin{matrix}
                o_{L} = \frac{\textbf{m}_{LWRA} + \textbf{m}_{LWRB}}{2} & o_{R} = \frac{\textbf{m}_{RWRA} + \textbf{m}_{RWRB}}{2} \\
                \hat{z}_{L} = \textbf{m}_{LWJC} - \textbf{m}_{LHND} & \hat{z}_{R} = \textbf{m}_{RWJC} - \textbf{m}_{RHND} \\
                \hat{y}_{L} = \textbf{m}_{LWRI} - \textbf{m}_{LWRA} & \hat{y}_{R} = \textbf{m}_{RWRA} - \textbf{m}_{RWRI} \\
                \hat{x}_{L} = \hat{y}_{L} \times \hat{z}_{L} & \hat{x}_{R} = \hat{y}_{R} \times \hat{z}_{R} \\
                \hat{y}_{L} = \hat{z}_{L} \times \hat{x}_{L} & \hat{y}_{R} = \hat{z}_{R} \times \hat{x}_{R}
            \end{matrix}

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAxes
        >>> np.set_printoptions(suppress=True)
        >>> rwra = np.array([776.51, 495.68, 1108.38])
        >>> rwrb = np.array([830.90, 436.75, 1119.11])
        >>> lwra = np.array([-249.28, 525.32, 1117.09])
        >>> lwrb = np.array([-311.77, 477.22, 1125.16])
        >>> rfin = np.array([863.71, 524.44, 1074.54])
        >>> lfin = np.array([-326.65, 558.34, 1091.04])
        >>> wrist_jc = [[
        ... [793.77, 450.44, 1084.12, 793.32],
        ... [794.01, 451.38, 1085.15, 451.29],
        ... [792.75, 450.76, 1085.05, 1084.43],
        ... [0, 0, 0, 1]
        ... ], [
        ... [-272.92, 485.01, 1090.96, -272.45],
        ... [-271.74, 485.72, 1090.67, 485.8],
        ... [-271.94, 485.19, 1091.96, 1091.36],
        ... [0, 0, 0, 1]
        ... ]]
        >>> r_hand_thickness = 34.0
        >>> l_hand_thickness = 34.0
        >>> [np.around(arr, 2) for arr in CalcAxes().hand_axis(
        ...     rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_jc, r_hand_thickness, l_hand_thickness)] #doctest: +NORMALIZE_WHITESPACE
        [array([[   0.15,    0.31,    0.94,  859.8 ],
            [  -0.73,    0.68,   -0.11,  517.27],
            [  -0.67,   -0.67,    0.33, 1051.97],
            [   0.  ,    0.  ,    0.  ,    1.  ]]), array([[  -0.09,    0.27,    0.96, -324.52],
            [  -0.8 ,   -0.59,    0.1 ,  551.89],
            [   0.6 ,   -0.76,    0.27, 1068.02],
            [   0.  ,    0.  ,    0.  ,    1.  ]])]
        """

        rwri = [(rwra[0]+rwrb[0])/2.0, (rwra[1]+rwrb[1]) /
                2.0, (rwra[2]+rwrb[2])/2.0]
        lwri = [(lwra[0]+lwrb[0])/2.0, (lwra[1]+lwrb[1]) /
                2.0, (lwra[2]+lwrb[2])/2.0]

        rwjc = [wrist_jc[0][0][-1], wrist_jc[0][1][-1], wrist_jc[0][2][-1]]
        lwjc = [wrist_jc[1][0][-1], wrist_jc[1][1][-1], wrist_jc[1][2][-1]]

        mm = 7.0

        r_delta = (r_hand_thickness/2.0 + mm)
        l_delta = (l_hand_thickness/2.0 + mm)

        lhnd = CalcUtils.find_joint_center(lwri, lwjc, lfin, l_delta)
        rhnd = CalcUtils.find_joint_center(rwri, rwjc, rfin, r_delta)

        # Left
        z_axis = [lwjc[0]-lhnd[0], lwjc[1]-lhnd[1], lwjc[2]-lhnd[2]]
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        y_axis = [lwri[0]-lwra[0], lwri[1]-lwra[1], lwri[2]-lwra[2]]
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        y_axis = np.cross(z_axis, x_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        l_axis = np.zeros((4, 4))
        l_axis[3, 3] = 1.0
        l_axis[0, :3] = x_axis
        l_axis[1, :3] = y_axis
        l_axis[2, :3] = z_axis
        l_axis[:3, 3] = lhnd

        # Right
        z_axis = [rwjc[0]-rhnd[0], rwjc[1]-rhnd[1], rwjc[2]-rhnd[2]]
        z_axis_div = np.linalg.norm(z_axis)
        z_axis = [z_axis[0]/z_axis_div, z_axis[1] /
                  z_axis_div, z_axis[2]/z_axis_div]

        y_axis = [rwra[0]-rwri[0], rwra[1]-rwri[1], rwra[2]-rwri[2]]
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        x_axis = np.cross(y_axis, z_axis)
        x_axis_div = np.linalg.norm(x_axis)
        x_axis = [x_axis[0]/x_axis_div, x_axis[1] /
                  x_axis_div, x_axis[2]/x_axis_div]

        y_axis = np.cross(z_axis, x_axis)
        y_axis_div = np.linalg.norm(y_axis)
        y_axis = [y_axis[0]/y_axis_div, y_axis[1] /
                  y_axis_div, y_axis[2]/y_axis_div]

        r_axis = np.zeros((4, 4))
        r_axis[3, 3] = 1.0
        r_axis[0, :3] = x_axis
        r_axis[1, :3] = y_axis
        r_axis[2, :3] = z_axis
        r_axis[:3, 3] = rhnd

        return [r_axis, l_axis]


class CalcAngles():

    def __init__(self):
        self.funcs = [self.pelvis_angle, self.hip_angle, self.knee_angle, self.ankle_angle, self.foot_angle, self.head_angle,
                      self.thorax_angle, self.neck_angle, self.spine_angle, self.shoulder_angle, self.elbow_angle, self.wrist_angle]

    def pelvis_angle(self, axis_p, axis_d):
        r"""Pelvis angle calculation.

        This function takes in two axes and returns three angles and uses the
        inverse Euler rotation matrix in YXZ order.

        Returns the angles in degrees.

        Parameters
        ----------
        axis_p : list
            Shows the unit vector of axis_p, the position of the proximal axis.
        axis_d : list
            Shows the unit vector of axis_d, the position of the distal axis.

        Returns
        -------
        angle : list
            Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

        Notes
        -----
        :math:`\beta = \arctan2{((axis\_d_{z} \cdot axis\_p_{y}), \sqrt{(axis\_d_{z} \cdot axis\_p_{x})^2 + (axis\_d_{z} \cdot axis\_p_{z})^2}})`

        :math:`\alpha = \arctan2{((axis\_d_{z} \cdot axis\_p_{x}), axis\_d_{z} \cdot axis\_p_{z})}`

        :math:`\gamma = \arctan2{((axis\_d_{x} \cdot axis\_p_{y}), axis\_d_{y} \cdot axis\_p_{y})}`

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAngles
        >>> axis_p = [[ 0.04, 0.99, 0.06, 452.12],
        ...           [ 0.99, -0.04, -0.05, 987.36],
        ...           [-0.05,  0.07, -0.99, 125.68],
        ...           [0, 0, 0, 1]]
        >>> axis_d = [[-0.18, -0.98, -0.02, 418.56],
        ...           [ 0.71, -0.11, -0.69, 857.41],
        ...           [ 0.67, -0.14, 0.72, 418.56],
        ...           [0, 0, 0, 1]]
        >>> np.around(CalcAngles().pelvis_angle(axis_p,axis_d), 2)
        array([-174.82,   39.82,  -10.54])
        """

        beta = np.arctan2((
            (axis_d[2][0] * axis_p[1][0])
            + (axis_d[2][1] * axis_p[1][1])
            + (axis_d[2][2] * axis_p[1][2])
        ),
            np.sqrt((
                axis_d[2][0] * axis_p[0][0]
                + axis_d[2][1] * axis_p[0][1]
                + axis_d[2][2] * axis_p[0][2]
            ) ** 2 + (
                axis_d[2][0] * axis_p[2][0]
                + axis_d[2][1] * axis_p[2][1]
                + axis_d[2][2] * axis_p[2][2]
            ) ** 2
        )
        )

        alpha = np.arctan2((
            (axis_d[2][0] * axis_p[0][0])
            + (axis_d[2][1] * axis_p[0][1])
            + (axis_d[2][2] * axis_p[0][2])
        ), (
            (axis_d[2][0] * axis_p[2][0])
            + (axis_d[2][1] * axis_p[2][1])
            + (axis_d[2][2] * axis_p[2][2])
        )
        )

        gamma = np.arctan2((
            (axis_d[0][0] * axis_p[1][0])
            + (axis_d[0][1] * axis_p[1][1])
            + (axis_d[0][2] * axis_p[1][2])
        ), (
            (axis_d[1][0] * axis_p[1][0])
            + (axis_d[1][1] * axis_p[1][1])
            + (axis_d[1][2] * axis_p[1][2])
        )
        )

        alpha = 180.0 * alpha / pi
        beta = 180.0 * beta / pi
        gamma = 180.0 * gamma / pi

        angle = [alpha, beta, gamma]
        return np.asarray(angle)

    def hip_angle(self, r_axis_p, r_axis_d, l_axis_p, l_axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """
        
        right_angles = self.get_angle(r_axis_p, r_axis_d)
        left_angles = self.get_angle(l_axis_p, l_axis_d)

        return np.array([right_angles, left_angles])

    def knee_angle(self, r_axis_p, r_axis_d, l_axis_p, l_axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """
        
        right_angles = self.get_angle(r_axis_p, r_axis_d)
        left_angles = self.get_angle(l_axis_p, l_axis_d)

        return np.array([right_angles, left_angles])

    def ankle_angle(self, r_axis_p, r_axis_d, l_axis_p, l_axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """
        
        right_angles = self.get_angle(r_axis_p, r_axis_d)
        left_angles = self.get_angle(l_axis_p, l_axis_d)

        return np.array([right_angles, left_angles])

    def foot_angle(self, r_axis_p, r_axis_d, l_axis_p, l_axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """
        
        right_angles = self.get_angle(r_axis_p, r_axis_d)
        left_angles = self.get_angle(l_axis_p, l_axis_d)

        return np.array([right_angles, left_angles])

    def head_angle(self, axis_p, axis_d):
        r"""Head angle calculation function.

        This function takes in two axes and returns three angles and uses the
        inverse Euler rotation matrix in YXZ order.

        Returns the angles in degrees.

        Parameters
        ----------
        axis_p : list
            Shows the unit vector of axis_p, the position of the proximal axis.
        axis_d : list
            Shows the unit vector of axis_d, the position of the distal axis.

        Returns
        -------
        angle : list
            Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

        Notes
        -----
        :math:`\beta = \arctan2{((axisD_{z} \cdot axisP_{y}), \sqrt{(axisD_{x} \cdot axisP_{y})^2 + (axisD_{y} \cdot axisP_{y})^2}})`

        :math:`\alpha = \arctan2{(-(axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})}`

        :math:`\gamma = \arctan2{(-(axisD_{x} \cdot axisP_{y}), axisD_{y} \cdot axisP_{y})}`

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAngles
        >>> axis_p = [[0.04, 0.99, 0.06, 512.34],
        ...           [0.99, -0.04, -0.05, 471.15],
        ...           [-0.05,  0.07, -0.99, 124.14],
        ...           [0, 0, 0, 1]]
        >>> axis_d = [[-0.18, -0.98, -0.02, 842.14],
        ...           [ 0.71, -0.11, -0.69, 985.38],
        ...           [ 0.67, -0.14, 0.72, 412.87],
        ...           [0, 0, 0, 1]]
        >>> np.around(CalcAngles().head_angle(axis_p,axis_d), 2)
        array([ 185.18,  -39.99, -190.54])
        """
        # this is the angle calculation which order is Y-X-Z
        # alpha is abdcution angle.

        ang = (
            (-1 * axis_d[2][0] * axis_p[1][0])
            + (-1 * axis_d[2][1] * axis_p[1][1])
            + (-1 * axis_d[2][2] * axis_p[1][2])
        )
        alpha = np.nan
        if -1 <= ang <= 1:
            alpha = np.arcsin(ang)

        # check the abduction angle is in the area between -pi/2 and pi/2
        # beta is flextion angle
        # gamma is rotation angle

        beta = np.arctan2(
            (axis_d[2][0] * axis_p[1][0])
            + (axis_d[2][1] * axis_p[1][1])
            + (axis_d[2][2] * axis_p[1][2]),
            np.sqrt(
                (
                    axis_d[0][0] * axis_p[1][0]
                    + axis_d[0][1] * axis_p[1][1]
                    + axis_d[0][2] * axis_p[1][2]
                ) ** 2
                + (
                    axis_d[1][0] * axis_p[1][0]
                    + axis_d[1][1] * axis_p[1][1]
                    + axis_d[1][2] * axis_p[1][2]
                ) ** 2
            ),
        )

        alpha = np.arctan2(
            -1 * (
                (axis_d[2][0] * axis_p[0][0])
                + (axis_d[2][1] * axis_p[0][1])
                + (axis_d[2][2] * axis_p[0][2])
            ), (
                (axis_d[2][0] * axis_p[2][0])
                + (axis_d[2][1] * axis_p[2][1])
                + (axis_d[2][2] * axis_p[2][2])
            )
        )

        gamma = np.arctan2(
            -1 * (
                (axis_d[0][0] * axis_p[1][0])
                + (axis_d[0][1] * axis_p[1][1])
                + (axis_d[0][2] * axis_p[1][2])
            ), (
                (axis_d[1][0] * axis_p[1][0])
                + (axis_d[1][1] * axis_p[1][1])
                + (axis_d[1][2] * axis_p[1][2])
            ),
        )

        alpha = 180.0 * alpha / pi
        beta = 180.0 * beta / pi
        gamma = 180.0 * gamma / pi

        beta *= -1

        if alpha < 0:
            alpha *= -1
        else:
            if 0 < alpha < 180:
                alpha = 180 + (180 - alpha)

        if gamma > 90.0:
            if gamma > 120:
                gamma = (gamma - 180) * -1
            else:
                gamma = (gamma + 180) * -1
        else:
            if gamma < 0:
                gamma = (gamma + 180) * -1
            else:
                gamma = (gamma * -1) - 180.0

        angle = [alpha, beta, gamma]

        return angle

    def thorax_angle(self, axis_p, axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """

        return np.array(self.get_angle(axis_p, axis_d))

    def neck_angle(self, axis_p, axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """

        return np.array(self.get_angle(axis_p, axis_d))

    def spine_angle(self, axis_p, axis_d):
        r"""Spine angle calculation.

        This function takes in two axes and returns three angles and uses the
        inverse Euler rotation matrix in YXZ order.
        Returns the angles in degrees.

        Parameters
        ----------
        axis_p : list
            Shows the unit vector of axis_p, the position of the proximal axis.
        axis_d : list
            Shows the unit vector of axis_d, the position of the distal axis.

        Returns
        -------
        angle : list
            Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

        Notes
        -----
        :math:`\alpha = \arcsin{(axis\_d_{y} \cdot axis\_p_{z})}`

        :math:`\gamma = \arcsin{(-(axis\_d_{y} \cdot axis\_p_{x}) / \cos{\alpha})}`

        :math:`\beta = \arcsin{(-(axis\_d_{x} \cdot axis\_p_{z}) / \cos{\alpha})}`

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAngles
        >>> axis_p = [[ 0.04,   0.99,  0.06, 749.24],
        ...        [ 0.99, -0.04, -0.05, 321.12],
        ...        [-0.05,  0.07, -0.99, 145.12],
        ...        [0, 0, 0, 1]]
        >>> axis_d = [[-0.18, -0.98,-0.02, 541.68],
        ...        [ 0.71, -0.11,  -0.69, 112.48],
        ...        [ 0.67, -0.14,   0.72, 155.77],
        ...        [0, 0, 0, 1]]
        >>> np.around(CalcAngles().spine_angle(axis_p,axis_d), 2)
        array([ 2.97,  9.13, 39.78])
        """
        # this angle calculation is for spine angle.

        alpha = np.arcsin(
            (axis_d[1][0] * axis_p[2][0])
            + (axis_d[1][1] * axis_p[2][1])
            + (axis_d[1][2] * axis_p[2][2])
        )

        gamma = np.arcsin((
            (-1 * axis_d[1][0] * axis_p[0][0])
            + (-1 * axis_d[1][1] * axis_p[0][1])
            + (-1 * axis_d[1][2] * axis_p[0][2])) / np.cos(alpha)
        )

        beta = np.arcsin((
            (-1 * axis_d[0][0] * axis_p[2][0])
            + (-1 * axis_d[0][1] * axis_p[2][1])
            + (-1 * axis_d[0][2] * axis_p[2][2])) / np.cos(alpha)
        )

        angle = [180.0 * beta / pi, 180.0 * gamma / pi, 180.0 * alpha / pi]

        return angle

    def shoulder_angle(self, r_axis_p, r_axis_d, l_axis_p, l_axis_d):
        r"""Shoulder angle calculation.

        This function takes in two axes and returns three angles and uses the
        inverse Euler rotation matrix in YXZ order.

        Returns the angles in degrees.

        Parameters
        ----------
        axis_p : list
            Shows the unit vector of axis_p, the position of the proximal axis.
        axis_d : list
            Shows the unit vector of axis_d, the position of the distal axis.

        Returns
        -------
        angle : list
            Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

        Notes
        -----
        :math:`\alpha = \arcsin{(axis\_d_{z} \cdot axis\_p_{x})}`

        :math:`\beta = \arctan2{(-(axis\_d_{z} \cdot axis\_p_{y}), axis\_d_{z} \cdot axis\_p_{z})}`

        :math:`\gamma = \arctan2{(-(axis\_d_{y} \cdot axis\_p_{x}), axis\_d_{x} \cdot axis\_p_{x})}`

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAngles
        >>> axis_p = [[ 0.04, 0.99, 0.06, 214.14],
        ...        [ 0.99, -0.04, -0.05, 32.14],
        ...       [-0.05,  0.07, -0.99, 452.89],
        ...       [0, 0, 0, 1]]
        >>> axis_d = [[-0.18, -0.98, -0.02, 874.12],
        ...        [ 0.71, -0.11, -0.69, 128.16],
        ...        [ 0.67, -0.14, 0.72, 541.98],
        ...        [0, 0, 0, 1]]
        >>> np.around(CalcAngles().shoulder_angle(axis_p,axis_d,axis_p,axis_d), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[  -3.93, -140.07,  172.9 ],
        [  -3.93, -140.07,  172.9 ]])
        """

        # beta is flexion / extension
        # gamma is adduction / abduction
        # alpha is internal / external rotation

        # this is the right shoulder angle calculation
        alpha = np.arcsin(
            (r_axis_d[2][0] * r_axis_p[0][0])
            + (r_axis_d[2][1] * r_axis_p[0][1])
            + (r_axis_d[2][2] * r_axis_p[0][2])
        )

        beta = np.arctan2(
            -1 * (
                (r_axis_d[2][0] * r_axis_p[1][0])
                + (r_axis_d[2][1] * r_axis_p[1][1])
                + (r_axis_d[2][2] * r_axis_p[1][2])
            ), (
                (r_axis_d[2][0] * r_axis_p[2][0])
                + (r_axis_d[2][1] * r_axis_p[2][1])
                + (r_axis_d[2][2] * r_axis_p[2][2])
            )
        )

        gamma = np.arctan2(
            -1 * (
                (r_axis_d[1][0] * r_axis_p[0][0])
                + (r_axis_d[1][1] * r_axis_p[0][1])
                + (r_axis_d[1][2] * r_axis_p[0][2])
            ), (
                (r_axis_d[0][0] * r_axis_p[0][0])
                + (r_axis_d[0][1] * r_axis_p[0][1])
                + (r_axis_d[0][2] * r_axis_p[0][2])
            ),
        )

        right_angle = [180.0 * alpha / pi,
                       180.0 * beta / pi, 180.0 * gamma / pi]

        # this is the left shoulder angle calculation
        alpha = np.arcsin(
            (l_axis_d[2][0] * l_axis_p[0][0])
            + (l_axis_d[2][1] * l_axis_p[0][1])
            + (l_axis_d[2][2] * l_axis_p[0][2])
        )

        beta = np.arctan2(
            -1 * (
                (l_axis_d[2][0] * l_axis_p[1][0])
                + (l_axis_d[2][1] * l_axis_p[1][1])
                + (l_axis_d[2][2] * l_axis_p[1][2])
            ), (
                (l_axis_d[2][0] * l_axis_p[2][0])
                + (l_axis_d[2][1] * l_axis_p[2][1])
                + (l_axis_d[2][2] * l_axis_p[2][2])
            )
        )

        gamma = np.arctan2(
            -1 * (
                (l_axis_d[1][0] * l_axis_p[0][0])
                + (l_axis_d[1][1] * l_axis_p[0][1])
                + (l_axis_d[1][2] * l_axis_p[0][2])
            ), (
                (l_axis_d[0][0] * l_axis_p[0][0])
                + (l_axis_d[0][1] * l_axis_p[0][1])
                + (l_axis_d[0][2] * l_axis_p[0][2])
            ),
        )

        left_angle = [180.0 * alpha / pi,
                      180.0 * beta / pi, 180.0 * gamma / pi]

        return np.array([right_angle, left_angle])

    def elbow_angle(self, r_axis_p, r_axis_d, l_axis_p, l_axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """
        
        right_angles = self.get_angle(r_axis_p, r_axis_d)
        left_angles = self.get_angle(l_axis_p, l_axis_d)

        return np.array([right_angles, left_angles])

    def wrist_angle(self, r_axis_p, r_axis_d, l_axis_p, l_axis_d):
        r"""Normal angle calculation.

            Please refer to the static get_angle function for documentation.
        """
        
        right_angles = self.get_angle(r_axis_p, r_axis_d)
        left_angles = self.get_angle(l_axis_p, l_axis_d)

        return np.array([right_angles, left_angles])
    
    @staticmethod
    def get_angle(axis_p, axis_d):
        r"""Normal angle calculation.

        This function takes in two axes and returns three angles and uses the
        inverse Euler rotation matrix in YXZ order.

        Returns the angles in degrees.

        Parameters
        ----------
        axis_p : list
            Shows the unit vector of axis_p, the position of the proximal axis.
        axis_d : list
            Shows the unit vector of axis_d, the position of the distal axis.

        Returns
        -------
        angle : list
            Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list.

        Notes
        -----
        As we use arcsin we have to care about if the angle is in area between -pi/2 to pi/2
        
        :math:`\alpha = \arcsin{(-axis\_d_{z} \cdot axis\_p_{y})}`

        If alpha is between -pi/2 and pi/2

        :math:`\beta = \arctan2{((axis\_d_{z} \cdot axis\_p_{x}), axis\_d_{z} \cdot axis\_p_{z})}`
        
        :math:`\gamma = \arctan2{((axis\_d_{y} \cdot axis\_p_{y}), axis\_d_{x} \cdot axis\_p_{y})}`

        Otherwise

        :math:`\beta = \arctan2{(-(axis\_d_{z} \cdot axis\_p_{x}), axis\_d_{z} \cdot axis\_p_{z})}`

        :math:`\gamma = \arctan2{(-(axis\_d_{y} \cdot axis\_p_{y}), axis\_d_{x} \cdot axis\_p_{y})}`

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcAngles
        >>> axis_p = [[ 0.04,   0.99,  0.06, 429.67],
        ...         [ 0.99, -0.04, -0.05, 275.15],
        ...         [-0.05,  0.07, -0.99, 1452.95],
        ...         [0, 0, 0, 1]]
        >>> axis_d = [[-0.18, -0.98, -0.02, 64.09],
        ...         [ 0.71, -0.11,  -0.69, 275.83],
        ...         [ 0.67, -0.14,   0.72, 1463.78],
        ...         [0, 0, 0, 1]]
        >>> np.around(CalcAngles.get_angle(axis_p, axis_d), 2)
        array([-174.82,  -39.26,  100.54])
        """
        # this is the angle calculation which order is Y-X-Z, alpha is the abdcution angle.

        ang = (
            (-1 * axis_d[2][0] * axis_p[1][0])
            + (-1 * axis_d[2][1] * axis_p[1][1])
            + (-1 * axis_d[2][2] * axis_p[1][2])
        )

        alpha = np.nan
        if -1 <= ang <= 1:
            alpha = np.arcsin(ang)

        # check the abduction angle is in the area between -pi/2 and pi/2
        # beta is flextion angle, gamma is rotation angle

        if -1.57079633 < alpha < 1.57079633:
            beta = np.arctan2(
                (axis_d[2][0] * axis_p[0][0])
                + (axis_d[2][1] * axis_p[0][1])
                + (axis_d[2][2] * axis_p[0][2]),

                (axis_d[2][0] * axis_p[2][0])
                + (axis_d[2][1] * axis_p[2][1])
                + (axis_d[2][2] * axis_p[2][2])
            )

            gamma = np.arctan2(
                (axis_d[1][0] * axis_p[1][0])
                + (axis_d[1][1] * axis_p[1][1])
                + (axis_d[1][2] * axis_p[1][2]),

                (axis_d[0][0] * axis_p[1][0])
                + (axis_d[0][1] * axis_p[1][1])
                + (axis_d[0][2] * axis_p[1][2])
            )
        else:
            beta = np.arctan2(
                -1 * (
                    (axis_d[2][0] * axis_p[0][0])
                    + (axis_d[2][1] * axis_p[0][1])
                    + (axis_d[2][2] * axis_p[0][2])
                ),
                    (axis_d[2][0] * axis_p[2][0])
                    + (axis_d[2][1] * axis_p[2][1])
                    + (axis_d[2][2] * axis_p[2][2])
            )
            gamma = np.arctan2(
                -1 * (
                    (axis_d[1][0] * axis_p[1][0])
                    + (axis_d[1][1] * axis_p[1][1])
                    + (axis_d[1][2] * axis_p[1][2])
                ),
                    (axis_d[0][0] * axis_p[1][0])
                    + (axis_d[0][1] * axis_p[1][1])
                    + (axis_d[0][2] * axis_p[1][2])
            )

        angle = [180.0*beta/pi, 180.0*alpha / pi, 180.0*gamma/pi]

        return angle


class CalcUtils:
    @staticmethod
    def rotmat(x=0, y=0, z=0):
        r"""Rotation Matrix.

        This function creates and returns a rotation matrix.

        Parameters
        ----------
        x, y, z : float, optional
            Angle, which will be converted to radians, in
            each respective axis to describe the rotations.
            The default is 0 for each unspecified angle.

        Returns
        -------
        r_xyz : list
            The product of the matrix multiplication.

        Notes
        -----
        :math:`r_x = [ [1,0,0], [0, \cos(x), -sin(x)], [0, sin(x), cos(x)] ]`
        :math:`r_y = [ [cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)] ]`
        :math:`r_z = [ [cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1] ]`
        :math:`r_{xy} = r_x * r_y`
        :math:`r_{xyz} = r_{xy} * r_z`

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcUtils
        >>> x = 0.5
        >>> y = 0.3
        >>> z = 0.8
        >>> np.around(CalcUtils.rotmat(x, y, z), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[ 1.  , -0.01,  0.01],
        [ 0.01,  1.  , -0.01],
        [-0.01,  0.01,  1.  ]])
        >>> x = 0.5
        >>> np.around(CalcUtils.rotmat(x), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[ 1.  ,  0.  ,  0.  ],
        [ 0.  ,  1.  , -0.01],
        [ 0.  ,  0.01,  1.  ]])
        >>> x = 1
        >>> y = 1
        >>> np.around(CalcUtils.rotmat(x,y), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[ 1.  ,  0.  ,  0.02],
        [ 0.  ,  1.  , -0.02],
        [-0.02,  0.02,  1.  ]])
        """
        x, y, z = math.radians(x), math.radians(y), math.radians(z)
        r_x = [[1, 0, 0], [0, math.cos(x), math.sin(
            x)*-1], [0, math.sin(x), math.cos(x)]]
        r_y = [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [
            math.sin(y)*-1, 0, math.cos(y)]]
        r_z = [[math.cos(z), math.sin(z)*-1, 0],
               [math.sin(z), math.cos(z), 0], [0, 0, 1]]
        r_xy = np.matmul(r_x, r_y)
        r_xyz = np.matmul(r_xy, r_z)

        return r_xyz

    @staticmethod
    def find_joint_center(p_a, p_b, p_c, delta):
        r"""Calculate the Joint Center.

        This function is based on the physical markers p_a, p_b, p_c
        and the resulting joint center are all on the same plane.

        Parameters
        ----------
        p_a, p_b, p_c : list
            Three markers x, y, z position of a, b, c.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.

        Returns
        -------
        joint_center : array
            Returns the joint center's x, y, z positions in a 1x3 array.

        Notes
        -----
        :math:`vec_{1} = p\_a-p\_c, \ vec_{2} = (p\_b-p\_c), \ vec_{3} = vec_{1} \times vec_{2}`

        :math:`mid = \frac{(p\_b+p\_c)}{2.0}`

        :math:`length = (p\_b - mid)`

        :math:`\theta = \arccos(\frac{delta}{vec_{2}})`

        :math:`\alpha = \cos(\theta*2), \ \beta = \sin(\theta*2)`

        :math:`u_x, u_y, u_z = vec_{3}`

        .. math::

            rot =
            \begin{bmatrix}
                \alpha+u_x^2*(1-\alpha) & u_x*u_y*(1.0-\alpha)-u_z*\beta & u_x*u_z*(1.0-\alpha)+u_y*\beta \\
                u_y*u_x*(1.0-\alpha+u_z*\beta & \alpha+u_y^2.0*(1.0-\alpha) & u_y*u_z*(1.0-\alpha)-u_x*\beta \\
                u_z*u_x*(1.0-\alpha)-u_y*\beta & u_z*u_y*(1.0-\alpha)+u_x*\beta & \alpha+u_z**2.0*(1.0-\alpha) \\
            \end{bmatrix}

        :math:`r\_vec = rot * vec_2`

        :math:`r\_vec = r\_vec * \frac{length}{norm(r\_vec)}`

        :math:`joint\_center = r\_vec + mid`

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm_calc import CalcUtils
        >>> p_a = [468.14, 325.09, 673.12]
        >>> p_b = [355.90, 365.38, 940.69]
        >>> p_c = [452.35, 329.06, 524.77]
        >>> delta = 59.5
        >>> CalcUtils.find_joint_center(p_a, p_b, p_c, delta).round(2)
        array([396.25, 347.92, 518.63])
        """

        # make the two vector using 3 markers, which is on the same plane.
        vec_1 = (p_a[0]-p_c[0], p_a[1]-p_c[1], p_a[2]-p_c[2])
        vec_2 = (p_b[0]-p_c[0], p_b[1]-p_c[1], p_b[2]-p_c[2])

        # vec_3 is cross vector of vec_1, vec_2, and then it normalized.
        vec_3 = np.cross(vec_1, vec_2)
        vec_3_div = np.linalg.norm(vec_3)
        vec_3 = [vec_3[0]/vec_3_div, vec_3[1]/vec_3_div, vec_3[2]/vec_3_div]

        mid = [(p_b[0]+p_c[0])/2.0, (p_b[1]+p_c[1])/2.0, (p_b[2]+p_c[2])/2.0]
        length = np.subtract(p_b, mid)
        length = np.linalg.norm(length)

        theta = math.acos(delta/np.linalg.norm(vec_2))

        cs_th = math.cos(theta*2)
        sn_th = math.sin(theta*2)

        u_x, u_y, u_z = vec_3

        # This rotation matrix is called Rodriques' rotation formula.
        # In order to make a plane, at least 3 number of markers is required which
        # means three physical markers on the segment can make a plane.
        # then the orthogonal vector of the plane will be rotating axis.
        # joint center is determined by rotating the one vector of plane around rotating axis.

        rot = np.matrix([
            [cs_th+u_x**2.0*(1.0-cs_th), u_x*u_y*(1.0-cs_th) -
             u_z*sn_th, u_x*u_z*(1.0-cs_th)+u_y*sn_th],
            [u_y*u_x*(1.0-cs_th)+u_z*sn_th, cs_th+u_y**2.0 *
             (1.0-cs_th), u_y*u_z*(1.0-cs_th)-u_x*sn_th],
            [u_z*u_x*(1.0-cs_th)-u_y*sn_th, u_z*u_y*(1.0-cs_th) +
             u_x*sn_th, cs_th+u_z**2.0*(1.0-cs_th)]
        ])

        r_vec = rot * (np.matrix(vec_2).transpose())
        r_vec = r_vec * length/np.linalg.norm(r_vec)

        r_vec = [r_vec[0, 0], r_vec[1, 0], r_vec[2, 0]]
        joint_center = np.array(
            [r_vec[0]+mid[0], r_vec[1]+mid[1], r_vec[2]+mid[2]])

        return joint_center
