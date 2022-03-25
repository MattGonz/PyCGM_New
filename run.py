from pycgm.CGMs.additional_function import Model_NewFunction
from pycgm.CGMs.modified_function import Model_CustomPelvis
from pycgm.model.model import Model
from pycgm.pyCGM import PyCGM
from pycgm.utils.csv_diff import diff_pycgm_csv

# 3 Dynamic Trials (Includes 59993 frame trial)
# model = Model('pycgm/SampleData/Sample_2/RoboStatic.c3d', \
#              ['pycgm/SampleData/Sample_2/RoboWalk.c3d', 'pycgm/SampleData/ROM/Sample_Dynamic.c3d', 'pycgm/SampleData/59993_Frame/59993_Frame_Dynamic.c3d'], \
#               'pycgm/SampleData/Sample_2/RoboSM.vsk')

# Standard Model, 2 dynamic trials
model = Model('pycgm/SampleData/Sample_2/RoboStatic.c3d', \
             ['pycgm/SampleData/Sample_2/RoboWalk.c3d', 'pycgm/SampleData/ROM/Sample_Dynamic.c3d'], \
              'pycgm/SampleData/Sample_2/RoboSM.vsk')

# Model with an overridden function
model_modified = Model_CustomPelvis('pycgm/SampleData/Sample_2/RoboStatic.c3d', \
                                   ['pycgm/SampleData/Sample_2/RoboWalk.c3d', 'pycgm/SampleData/ROM/Sample_Dynamic.c3d'], \
                                    'pycgm/SampleData/Sample_2/RoboSM.vsk')

# Model with additional function
model_extended = Model_NewFunction('pycgm/SampleData/Sample_2/RoboStatic.c3d', \
                                  ['pycgm/SampleData/Sample_2/RoboWalk.c3d', 'pycgm/SampleData/ROM/Sample_Dynamic.c3d'], \
                                   'pycgm/SampleData/Sample_2/RoboSM.vsk')

cgm = PyCGM([model, model_modified, model_extended])
cgm.run_all()

# Access model output
print(f"{model.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
print(f"{model.data.dynamic.Sample_Dynamic.angles.RHip.shape=}")

# Indexing cgm
print(f"{cgm[2].data.dynamic.RoboWalk.axes.REye.shape=}")

# Compare RoboWalk output to known CSV
diff_pycgm_csv(model, 'RoboWalk', 'pycgm/SampleData/Sample_2/pycgm_results.csv')

# TODO: Export

