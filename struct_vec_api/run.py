from custom_CGMs import Model_CustomPelvis, Model_NewFunction
from pyCGM import PyCGM
from model import Model
from csv_diff import diff_pycgm_csv

# 3 Dynamic Trials (Includes 59993 frame trial)
# model = Model('SampleData/Sample_2/RoboStatic.c3d', \
#              ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d', 'SampleData/59993_Frame/59993_Frame_Dynamic.c3d'], \
#               'SampleData/Sample_2/RoboSM.vsk')

# Standard Model, 2 dynamic trials
model = Model('SampleData/Sample_2/RoboStatic.c3d', \
             ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
              'SampleData/Sample_2/RoboSM.vsk')

# Model with an overridden function
model_modified = Model_CustomPelvis('SampleData/Sample_2/RoboStatic.c3d', \
                                   ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
                                    'SampleData/Sample_2/RoboSM.vsk')

# Model with additional function
model_extended = Model_NewFunction('SampleData/Sample_2/RoboStatic.c3d', \
                                  ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
                                   'SampleData/Sample_2/RoboSM.vsk')

cgm = PyCGM([model, model_modified, model_extended])
cgm.run_all()

# Access model output
print(f"{model.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
print(f"{model.data.dynamic.Sample_Dynamic.angles.RHip.shape=}")

# Indexing cgm
print(f"{cgm[2].data.dynamic.RoboWalk.axes.REye.shape=}")

# Compare RoboWalk output to known CSV
diff_pycgm_csv(model, 'RoboWalk', 'SampleData/Sample_2/pycgm_results.csv')

# TODO: Export

