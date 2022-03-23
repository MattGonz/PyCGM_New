from pyCGM import PyCGM
from model import Model
from custom_CGMs import Model_CustomPelvis, Model_NewFunction

# Includes 59993 frame trial
# matt = Model('SampleData/Sample_2/RoboStatic.c3d', \
#             ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d', 'SampleData/59993_Frame/59993_Frame_Dynamic.c3d'], \
#              'SampleData/Sample_2/RoboSM.vsk')

# Standard Model, 1 dynamic trial
# matt = Model('SampleData/Sample_2/RoboStatic.c3d', \
#             ['SampleData/Sample_2/RoboWalk.c3d'], \
#              'SampleData/Sample_2/RoboSM.vsk')

# Standard Model, 2 dynamic trials
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

# Access output axes from the original model (matt)
print(f"{matt.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
print(f"{matt.data.dynamic.RoboWalk.angles.RKnee.shape=}")

# Access output axes from the model at index 2 (matt_custom)
print(f"{cgm[2].data.dynamic.RoboWalk.axes.REyeball.shape=}")

