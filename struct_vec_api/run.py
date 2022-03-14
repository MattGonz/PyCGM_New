from pyCGM import PyCGM, Model
from custom_CGMs import Model_CustomPelvis, Model_NewFunction
import cProfile, pstats, io
from pstats import SortKey

# Includes 59993 frame trial
# matt = Model('SampleData/Sample_2/RoboStatic.c3d', \
#             ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d', 'SampleData/59993_Frame/59993_Frame_Dynamic.c3d'], \
#              'SampleData/Sample_2/RoboSM.vsk')

# Standard Model
matt = Model('SampleData/Sample_2/RoboStatic.c3d', \
            ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
             'SampleData/Sample_2/RoboSM.vsk')


# Model with an overridden calc_axis_pelvis
matt_modified = Model_CustomPelvis('SampleData/Sample_2/RoboStatic.c3d', \
                                  ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
                                   'SampleData/Sample_2/RoboSM.vsk')

# Model with a custom function calc_axis_eyeball
matt_custom = Model_NewFunction('SampleData/Sample_2/RoboStatic.c3d', \
                               ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
                                'SampleData/Sample_2/RoboSM.vsk')


cgm = PyCGM([matt, matt_modified, matt_custom])
cgm.run_all()

