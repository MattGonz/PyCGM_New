from pyCGM import pyCGM
import numpy as np

class harrington_hip_CGM(pyCGM):

    # example hip_axis function using harrington hip joint center

    def __init__(self, measurements, static_trial, dynamic_trial):
        super().__init__(measurements, static_trial, dynamic_trial)
        self.modify_function('hip_axis', measurements=['InterAsisDistance', 'GCS'], axes=['Pelvis'])

    def hip_axis(self, pelvis_width, pelvis_depth, pelvis_axis):
        # get the refactored 4x4 hip axis, using harrington hip joint center

        # unfortunately we don't have data with pelvis_depth
        return np.zeros((4,4))

        # what it would look like:
        l_hip_jc_x = -0.24 * pelvis_depth - 9.9
        l_hip_jc_y =  0.30 * pelvis_width + 10.9
        l_hip_jc_z = -0.33 * pelvis_width - 7.3
        l_hip_jc = [l_hip_jc_x, l_hip_jc_y, l_hip_jc_z]

        r_hip_jc_x = -0.24 * pelvis_depth - 9.9
        r_hip_jc_y = (0.30 * pelvis_width + 10.9) * -1
        r_hip_jc_z = -0.33 * pelvis_width - 7.3
        r_hip_jc = [r_hip_jc_x, r_hip_jc_y, r_hip_jc_z]

        hipaxis_center = (np.asarray(r_hip_jc) + np.asarray(l_hip_jc)) / 2

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

class oxfordCGM(pyCGM):

    # an example of an overridden function that returns custom angles

    def __init__(self, measurements, static_trial, dynamic_trial):
        super().__init__(measurements, static_trial, dynamic_trial)
        self.modify_function('foot_angle', markers=['RANK', 'RCPG', 'RSTL', 'RHLX', 'RLCA', 'RHEE', 'RP1M', 'RP5M', 'RD5M', 'RTOE'],
                                         returns_angles=['LForefoot', 'RForefoot', 'LHindfoot', 'RHindfoot'])
        

    def foot_angle(self, rank, rcpg, rstl, rhlx, rlca, rhee, rp1m, rp5m, rd5m, rtoe):
        '''
        returns:
        LFF/RFF: L/R forefoot with respect to tibia
        LHF/RHF: L/R hindfoot with respect to tibia
        '''
        # show all parameters passed
        # print('Oxford called with:')
        # print('rank', rank, 'rcpg', rcpg,
        #       'rstl', rstl, 'rhlx', rhlx,
        #       'rlca', rlca, 'rhee', rhee,
        #       'rp1m', rp1m, 'rp5m', rp5m,
        #       'rd5m', rd5m, 'rtoe', rtoe)

        # calculations would take this shape
        lff = np.zeros((3))
        rff = np.zeros((3))
        lhf = np.ones((3))
        rhf = np.ones((3))

        return np.array([lff,rff,lhf,rhf])
        
        
class eyeballCGM(pyCGM):
    # an example of an additional custom function that returns a custom axis

    def __init__(self, measurements, static_trial, dynamic_trial):
        super().__init__(measurements, static_trial, dynamic_trial)
        self.add_function('eyeball_axis', markers=['LFHD', 'RFHD', 'LBHD', 'RBHD'],
                                          returns_axes=['LEyeball', 'REyeball'])

    def eyeball_axis(self, lfhd, rfhd, lbhd, rbhd):

        eyeball_l = np.zeros((4,4))
        eyeball_l[3, 3] = 1.0
        eyeball_l[0, :3] = lfhd
        eyeball_l[1, :3] = lbhd
        eyeball_l[2, :3] = 1.0
        eyeball_l[:3, 3] = 1.0

        eyeball_r = np.zeros((4,4))
        eyeball_r[3, 3] = 1.0
        eyeball_r[0, :3] = rfhd
        eyeball_r[1, :3] = rbhd
        eyeball_r[2, :3] = 1.0
        eyeball_r[:3, 3] = 1.0

        return np.array([eyeball_l, eyeball_r])
