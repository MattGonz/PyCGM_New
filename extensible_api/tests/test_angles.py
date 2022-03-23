import unittest
import pycgm_calc as angles # Need to fix import not to require file locally.
import numpy as np
import pytest

rounding_precision = 5

class TestPycgmAngle():
    """
    This class tests the functions used for getting angles in axis.py:
    get_angle
    pelvis_angle
    head_angle
    shoulder_angle
    get_angle_spi
    """

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, 90]),
        # X rotations
        (90, 0, 0, [0, 90, 90]), (30, 0, 0, [0, 30, 90]), (-30, 0, 0, [0, -30, 90]), (120, 0, 0, [180, 60, -90]), (-120, 0, 0, [180, -60, -90]), (180, 0, 0, [180, 0, -90]),
        # Y rotations
        (0, 90, 0, [90, 0, 90]), (0, 30, 0, [30, 0, 90]), (0, -30, 0, [-30, 0, 90]), (0, 120, 0, [120, 0, 90]), (0, -120, 0, [-120, 0, 90]), (0, 180, 0, [180, 0, 90]),
        # Z rotations
        (0, 0, 90, [0, 0, 0]), (0, 0, 30, [0, 0, 60]), (0, 0, -30, [0, 0, 120]), (0, 0, 120, [0, 0, -30]), (0, 0, -120, [0, 0, -150]), (0, 0, 180, [0, 0, -90]),
        # Multiple Rotations
        (150, 30, 0, [146.30993247, 25.65890627, -73.89788625]), (45, 0, 60, [0, 45, 30]), (0, 90, 120, [90, 0, -30]), (135, 45, 90, [125.26438968, 30, -144.73561032])
    ])
    def test_get_angle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the get_angle function in axis.py,
        defined as get_angle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis

        get_angle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [beta, alpha, gamma]. get_angle uses the YXZ
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. Since arcsin
        is being used, the function checks wether the angle alpha is between -pi/2 and pi/2.
        The angles are calculated as follows:

        .. math::
            \[ \alpha = \arcsin{(-axisD_{z} \cdot axisP_{y})} \]

        If alpha is between -pi/2 and pi/2

        .. math::
            \[ \beta = \arctan2{((axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]

            \[ \gamma = \arctan2{((axisD_{y} \cdot axisP_{y}), axisD_{x} \cdot axisP_{y})} \]

        Otherwise

        .. math::
            \[ \beta = \arctan2{(-(axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]

            \[ \gamma = \arctan2{(-(axisD_{y} \cdot axisP_{y}), axisD_{x} \cdot axisP_{y})} \]

        This test calls angles.CalcUtils.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls axis.get_angle() with axisP and axisD, which was created with no rotation in the x, y or z
        direction. This result is then compared to the expected result. The results from this test will be in the
        YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same angle in
        the z direction. There is also an additional 90 degree angle in the z direction if there was no z rotation.
        If there was a z rotation than there will be a different angle in the z direction. A z rotation of 90, 30, -30,
        120, -120, 180 degrees results in a 0, 60, 120, -30, -150, -90 degree angle in the z direction respectively.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = angles.CalcUtils.rotmat(xRot, yRot, zRot)
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        result = angles.CalcAngles.get_angle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_get_angle_datatypes(self):
        """
        This test provides coverage of the get_angle function in axis.py, defined as get_angle(axisP,axisD).
        It checks that the resulting output from calling get_angle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        axisP_floats = angles.CalcUtils.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [0, 90, 0]

        # Check that calling get_angle on a list of ints yields the expected results
        result_int_list = angles.CalcAngles.get_angle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling get_angle on a numpy array of ints yields the expected results
        result_int_nparray = angles.CalcAngles.get_angle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling get_angle on a list of floats yields the expected results
        result_float_list = angles.CalcAngles.get_angle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling get_angle on a numpy array of floats yields the expected results
        result_float_nparray = angles.CalcAngles.get_angle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)
    
    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, 0]),
        # X rotations
        (90, 0, 0, [0, -90, 0]), (30, 0, 0, [0, -30, 0]), (-30, 0, 0, [0, 30, 0]), (120, 0, 0, [180, -60, 180]), (-120, 0, 0, [180, 60, 180]), (180, 0, 0, [180, 0, 180]),
        # Y rotations
        (0, 90, 0, [90, 0, 0]), (0, 30, 0, [30, 0, 0]), (0, -30, 0, [-30, 0, 0]), (0, 120, 0, [120, 0, 0]), (0, -120, 0, [-120, 0, -0]), (0, 180, 0, [180, 0, 0]),
        # Z rotations
        (0, 0, 90, [0, 0, 90]), (0, 0, 30, [0, 0, 30]), (0, 0, -30, [0, 0, -30]), (0, 0, 120, [0, 0, 120]), (0, 0, -120, [0, 0, -120]), (0, 0, 180, [0, 0, 180]),
        # Multiple Rotations
        (150, 30, 0, [146.30993247, -25.65890627, 163.89788625]), (45, 0, 60, [0, -45, 60]), (0, 90, 120, [90, 0, 120]), (135, 45, 90, [125.26438968, -30, -125.26438968])
    ])
    def test_pelvis_angle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the pelvis_angle function in axis.py,
        defined as pelvis_angle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis

        pelvis_angle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [alpha, beta, gamma]. pelvis_angle uses the YXZ
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:

        .. math::
            \[ \beta = \arctan2{((axisD_{z} \cdot axisP_{y}), \sqrt{(axisD_{z} \cdot axisP_{x})^2 + (axisD_{z} \cdot axisP_{z})^2}}) \]

            \[ \alpha = \arctan2{((axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]

            \[ \gamma = \arctan2{((axisD_{x} \cdot axisP_{y}), axisD_{y} \cdot axisP_{y})} \]

        This test calls angles.CalcUtils.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls axis.getHeadangle() with axisP and axisD, which was created with no rotation in the x, y or z
        direction. This result is then compared to the expected result. The results from this test will be in the
        YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same angle in
        the z direction. The exception to this is x rotations. An x rotation of 90, 30, -30, 120, -120, 180
        degrees results in a -90, -30, 30, -6, 60, 0 degree angle in the y direction respectively. A x rotation or
        120, -120, or 180 also results in a 180 degree rotation in the x and z angles.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = angles.CalcUtils.rotmat(xRot, yRot, zRot)
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        result = angles.CalcAngles().pelvis_angle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_pelvis_angle_datatypes(self):
        """
        This test provides coverage of the pelvis_angle function in axis.py, defined as pelvis_angle(axisP,axisD).
        It checks that the resulting output from calling pelvis_angle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        axisP_floats = angles.CalcUtils.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [90, 0, 180]

        # Check that calling pelvis_angle on a list of ints yields the expected results
        result_int_list = angles.CalcAngles().pelvis_angle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling get_pelvis_angle on a numpy array of ints yields the expected results
        result_int_nparray = angles.CalcAngles().pelvis_angle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling get_pelvis_angle on a list of floats yields the expected results
        result_float_list = angles.CalcAngles().pelvis_angle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling get_pelvis_angle on a numpy array of floats yields the expected results
        result_float_nparray = angles.CalcAngles().pelvis_angle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, -180]),
        # X rotations
        (90, 0, 0, [0, 90, -180]), (30, 0, 0, [0, 30, -180]), (-30, 0, 0, [0, -30, -180]), (120, 0, 0, [180, 60, 0]), (-120, 0, 0, [180, -60, 0]), (180, 0, 0, [180, 0, 0]),
        # Y rotations
        (0, 90, 0, [90, 0, -180]), (0, 30, 0, [30, 0, -180]), (0, -30, 0, [330, 0, -180]), (0, 120, 0, [120, 0, -180]), (0, -120, 0, [240, 0, -180]), (0, 180, 0, [180, 0, -180]),
        # Z rotations
        (0, 0, 90, [0, 0, -90]), (0, 0, 30, [0, 0, -150]), (0, 0, -30, [0, 0, -210]), (0, 0, 120, [0, 0, -60]), (0, 0, -120, [0, 0, -300]), (0, 0, 180, [0, 0, 0]),
        # Multiple Rotations
        (150, 30, 0, [146.30993247, 25.65890627, -16.10211375]), (45, 0, 60, [0, 45, -120]), (0, 90, 120, [90, 0, -60]), (135, 45, 90, [125.26438968, 30, 54.73561032])
    ])
    def test_head_angle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the get_head_angle function in axis.py,
        defined as get_head_angle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis

        get_head_angle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [alpha, beta, gamma]. get_head_angle uses the YXZ
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:

        .. math::
            \[ \beta = \arctan2{((axisD_{z} \cdot axisP_{y}), \sqrt{(axisD_{x} \cdot axisP_{y})^2 + (axisD_{y} \cdot axisP_{y})^2}}) \]

            \[ \alpha = \arctan2{(-(axisD_{z} \cdot axisP_{x}), axisD_{z} \cdot axisP_{z})} \]

            \[ \gamma = \arctan2{(-(axisD_{x} \cdot axisP_{y}), axisD_{y} \cdot axisP_{y})} \]

        This test calls axis.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls axis.get_head_angle() with axisP and axisD, which was created with no rotation in the x, y or z
        direction. This result is then compared to the expected result. The results from this test will be in the
        YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same angle in
        the z direction. There is also an additional -180 degree angle in the z direction if there was no z rotation.
        If there was a z rotation than there will be a different angle in the z direction. A z rotation of 90, 30, -30,
        120, -120, 180 degrees results in a -90, -150, -210, -60, -300, 0 degree angle in the z direction respectively.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = angles.CalcUtils.rotmat(xRot, yRot, zRot)
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        result = angles.CalcAngles().head_angle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_head_angle_datatypes(self):
        """
        This test provides coverage of the get_head_angle function in axis.py, defined as get_head_angle(axisP,axisD).
        It checks that the resulting output from calling get_head_angle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        axisP_floats = angles.CalcUtils.rotmat(90, 90, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [90, 0, 0]

        # Check that calling get_head_angle on a list of ints yields the expected results
        result_int_list = angles.CalcAngles().head_angle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling get_head_angle on a numpy array of ints yields the expected results
        result_int_nparray = angles.CalcAngles().head_angle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling get_head_angle on a list of floats yields the expected results
        result_float_list = angles.CalcAngles().head_angle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling get_head_angle on a numpy array of floats yields the expected results
        result_float_nparray = angles.CalcAngles().head_angle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)
    
    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [[0, 0, 0], [0, 0, 0]]),
        # X rotations
        (90, 0, 0, [[0, 90, 0], [0, 90, 0]]), (30, 0, 0, [[0, 30, 0], [0, 30, 0]]), (-30, 0, 0, [[0, -30, 0], [0, -30, 0]]), 
        (120, 0, 0, [[0, 120, 0], [0, 120, 0]]), (-120, 0, 0, [[0, -120, 0], [0, -120, 0]]), (180, 0, 0, [[0, 180, 0], [0, 180, 0]]),
        # Y rotations
        (0, 90, 0, [[90, 0, 0], [90, 0, 0]]), (0, 30, 0, [[30, 0, 0], [30, 0, 0]]), (0, -30, 0, [[-30, 0, 0], [-30, 0, 0]]), 
        (0, 120, 0, [[60, -180, -180], [60, -180, -180]]), (0, -120, 0, [[-60, -180, -180], [-60, -180, -180]]), 
        (0, 180, 0, [[0, -180, -180], [0, -180, -180]]),
        # Z rotations
        (0, 0, 90, [[0, 0, 90], [0, 0, 90]]), (0, 0, 30, [[0, 0, 30], [0, 0, 30]]), (0, 0, -30, [[0, 0, -30], [0, 0, -30]]), 
        (0, 0, 120, [[0, 0, 120], [0, 0, 120]]), (0, 0, -120, [[0, 0, -120], [0, 0, -120]]), (0, 0, 180, [[0, 0, 180], [0, 0, 180]]),
        # Multiple Rotations
        (150, 30, 0, [[30, 150, 0], [30, 150, 0]]), (45, 0, 60, [[0, 45, 60], [0, 45, 60]]), (0, 90, 120, [[90, 0, 120], [90, 0, 120]]), 
        (135, 45, 90, [[45, 135, 90], [45, 135, 90]])
    ])
    def test_get_shoulder_angle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the get_shoulder_angle function in axis.py,
        defined as get_shoulder_angle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis.

        get_shoulder_angle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [alpha, beta, gamma]. get_shoulder_angle uses the XYZ
        order Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:

        .. math::
            \[ \alpha = \arcsin{(axisD_{z} \cdot axisP_{x})} \]

            \[ \beta = \arctan2{(-(axisD_{z} \cdot axisP_{y}), axisD_{z} \cdot axisP_{z})} \]

            \[ \gamma = \arctan2{(-(axisD_{y} \cdot axisP_{x}), axisD_{x} \cdot axisP_{x})} \]

        This test calls axis.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls axis.get_shoulder_angle() with axisP and axisD, which was created with no rotation in the
        x, y or z direction. This result is then compared to the expected result. The results from this test will
        be in the YXZ order, meaning that a parameter with an inputed x rotation will have a result with the same
        angle in the y direction. The only exception to this is a 120, -120, or 180 degree Y rotation. These will end
        up with a 60, -60, and 0 degree angle in the X direction respectively, and with a -180 degree
        angle in the y and z direction.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = angles.CalcUtils.rotmat(xRot, yRot, zRot)
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        result = angles.CalcAngles().shoulder_angle(axisP, axisD, axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_get_shoulder_angle_datatypes(self):
        """
        This test provides coverage of the get_shoulder_angle function in axis.py, defined as get_shoulder_angle(axisP,axisD).
        It checks that the resulting output from calling get_shoulder_angle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        axisP_floats = angles.CalcUtils.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [[0, 90, 90], [0, 90, 90]]

        # Check that calling get_shoulder_angle on a list of ints yields the expected results
        result_int_list = angles.CalcAngles().shoulder_angle(axisP_ints, axisD, axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling get_shoulder_angle on a numpy array of ints yields the expected results
        result_int_nparray = angles.CalcAngles().shoulder_angle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'), np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling get_shoulder_angle on a list of floats yields the expected results
        result_float_list = angles.CalcAngles().shoulder_angle(axisP_floats, axisD, axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling get_shoulder_angle on a numpy array of floats yields the expected results
        result_float_nparray = angles.CalcAngles().shoulder_angle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'), np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)

    @pytest.mark.parametrize(["xRot", "yRot", "zRot", "expected"], [
        (0, 0, 0, [0, 0, 0]),
        # X rotations
        (90, 0, 0, [0, 0, 90]), (30, 0, 0, [0, 0, 30]), (-30, 0, 0, [0, 0, -30]), (120, 0, 0, [0, 0, 60]), (-120, 0, 0, [0, 0, -60]), (180, 0, 0, [0, 0, 0]),
        # Y rotations
        (0, 90, 0, [90, 0, 0]), (0, 30, 0, [30, 0, 0]), (0, -30, 0, [-30, 0, 0]), (0, 120, 0, [60, 0, 0]), (0, -120, 0, [-60, 0, 0]), (0, 180, 0, [0, 0, 0]),
        # Z rotations
        (0, 0, 90, [0, 90, 0]), (0, 0, 30, [0, 30, 0]), (0, 0, -30, [0, -30, 0]), (0, 0, 120, [0, 60, 0]), (0, 0, -120, [0, -60, 0]), (0, 0, 180, [0, 0, 0]),
        # Multiple Rotations
        (150, 30, 0, [-30, 0, 30]), (45, 0, 60, [-40.89339465, 67.7923457, 20.70481105]), (0, 90, 120, [-90, 0, 60]), (135, 45, 90, [-54.73561032, 54.73561032, -30])
    ])
    def test_get_spine_angle(self, xRot, yRot, zRot, expected):
        """
        This test provides coverage of the get_spine_angle function in axis.py,
        defined as get_spine_angle(axisP,axisD) where axisP is the proximal axis and axisD is the distal axis
        get_spine_angle takes in as input two axes, axisP and axisD, and returns in degrees, the Euler angle
        rotations required to rotate axisP to axisD as a list [beta, gamma, alpha]. get_spine_angle uses the XZX
        order of Euler rotations to calculate the angles. The rotation matrix is obtained by directly comparing
        the vectors in axisP to those in axisD through dot products between different components
        of each axis. axisP and axisD each have 3 components to their axis, x, y, and z. 
        The angles are calculated as follows:
        .. math::
            \[ alpha = \arcsin{(axisD_{y} \cdot axisP_{z})} \]
            \[ gamma = \arcsin{(-(axisD_{y} \cdot axisP_{x}) / \cos{\alpha})} \]
            \[ beta = \arcsin{(-(axisD_{x} \cdot axisP_{z}) / \cos{\alpha})} \]
        This test calls axis.rotmat() to create axisP with an x, y, and z rotation defined in the parameters.
        It then calls axis.get_spine_angle() with axisP and axisD, which was created with no rotation in the
        x, y or z direction. This result is then compared to the expected result. The results from this test will
        be in the YZX order, meaning that a parameter with an inputed x rotation will have a result with the same
        angle in the z direction. The only exception to this is a 120, -120, or 180 degree Y rotation. The exception
        to this is that 120, -120, and 180 degree rotations end up with 60, -60, and 0 degree angles respectively.
        """
        # Create axisP as a rotatinal matrix using the x, y, and z rotations given in testcase
        axisP = angles.CalcUtils.rotmat(xRot, yRot, zRot)
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        result = angles.CalcAngles().spine_angle(axisP, axisD)
        np.testing.assert_almost_equal(result, expected, rounding_precision)

    def test_get_spine_angle_datatypes(self):
        """
        This test provides coverage of the get_spine_angle function in axis.py, defined as get_spine_angle(axisP,axisD).
        It checks that the resulting output from calling get_spine_angle is correct for a list of ints, a numpy array of
        ints, a list of floats, and a numpy array of floats.
        """
        axisD = angles.CalcUtils.rotmat(0, 0, 0)
        axisP_floats = angles.CalcUtils.rotmat(90, 0, 90)
        axisP_ints = [[int(y) for y in x] for x in axisP_floats]
        expected = [-90, 90, 0]

        # Check that calling get_spine_angle on a list of ints yields the expected results
        result_int_list = angles.CalcAngles().spine_angle(axisP_ints, axisD)
        np.testing.assert_almost_equal(result_int_list, expected, rounding_precision)

        # Check that calling get_spine_angle on a numpy array of ints yields the expected results
        result_int_nparray = angles.CalcAngles().spine_angle(np.array(axisP_ints, dtype='int'), np.array(axisD, dtype='int'))
        np.testing.assert_almost_equal(result_int_nparray, expected, rounding_precision)

        # Check that calling get_spine_angle on a list of floats yields the expected results
        result_float_list = angles.CalcAngles().spine_angle(axisP_floats, axisD)
        np.testing.assert_almost_equal(result_float_list, expected, rounding_precision)

        # Check that calling get_spine_angle on a numpy array of floats yields the expected results
        result_float_nparray = angles.CalcAngles().spine_angle(np.array(axisP_floats, dtype='float'), np.array(axisD, dtype='float'))
        np.testing.assert_almost_equal(result_float_nparray, expected, rounding_precision)