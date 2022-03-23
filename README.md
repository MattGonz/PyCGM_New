## Extensible, vectorized pyCGM API
* ### Usage: [run.py](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/struct_vec_api/run.py)
* ### Custom models: [custom_CGMs.py](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/struct_vec_api/custom_CGMs.py)
* ### Speed comparison:
  * Non-vectorized: `34.884s` [cprofile_RoboWalk_original.txt](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/cprofile_RoboWalk_original.txt)
  * Vectorized: `0.164s` [cprofile_RoboWalk_vectorized.txt](https://github.com/MattGonz/PyCGM_Prototypes/blob/main/cprofile_RoboWalk_vectorized.txt)
