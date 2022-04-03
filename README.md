## Extensible, vectorized pyCGM API
* ### Usage: [run.py](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/run.py)
* ### Custom models:
  * [Modifying existing functions](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/pycgm/CGMs/modified_function.py)
  * [Adding additional functions](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/pycgm/CGMs/additional_function.py)
* ### Speed comparison (RoboWalk, 6000 frames)
     | API                                    | Time | cProfile
     | :---:                                    |      :---:      | :---: | 
     | Original ([cadop/master](https://github.com/cadop/pyCGM/))               |    `9.146s`   |[cprofile_RoboWalk_original.txt](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/speed_tests/cprofile_RoboWalk_original.txt) |
     | Extensible (dev branch/extensible_api) |   `34.884s`   |[cprofile_RoboWalk_extensible.txt](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/speed_tests/cprofile_RoboWalk_extensible.txt) |
     | Vectorized / Extensible                |    `0.142s`   |[cprofile_RoboWalk_vectorized.txt](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/speed_tests/cprofile_RoboWalk_vectorized.txt) |
