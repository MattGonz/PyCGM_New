## Extensible, vectorized pyCGM API
* ### Usage: [run.py](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/run.py)
* ### Custom models:
  * [Modifying existing functions](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/pycgm/CGMs/modified_function.py)
  * [Adding additional functions](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/pycgm/CGMs/additional_function.py)
* ### Speed comparison:
  * Non-vectorized: `34.884s` [cprofile_RoboWalk_original.txt](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/speed_tests/cprofile_RoboWalk_original.txt)
  * Vectorized: `0.164s` [cprofile_RoboWalk_vectorized.txt](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/speed_tests/cprofile_RoboWalk_vectorized.txt)
