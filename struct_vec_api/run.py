from pyCGM import PyCGM, Subject
from custom_CGMs import Subject_CustomPelvis

matt = Subject('SampleData/Sample_2/RoboStatic.c3d', \
              ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
               'SampleData/Sample_2/RoboSM.vsk')

matt_custom = Subject_CustomPelvis('SampleData/Sample_2/RoboStatic.c3d', \
                                  ['SampleData/Sample_2/RoboWalk.c3d', 'SampleData/ROM/Sample_Dynamic.c3d'], \
                                   'SampleData/Sample_2/RoboSM.vsk')

cgm = PyCGM([matt, matt_custom])
cgm.run_all()

