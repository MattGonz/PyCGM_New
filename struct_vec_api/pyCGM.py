import time
from itertools import chain

import numpy as np
import numpy.lib.recfunctions as rfn
from utils import subject_utils
from defaults.parameters import Measurement, Marker, Axis, Angle, AxisFunctions, AngleFunctions
from defaults import return_keys
from pycgm_calc import CalcAngles, CalcAxes


class Subject():
    def __init__(self, static_filename, dynamic_filenames, measurement_filename):
        self.data = subject_utils.structure_subject(static_filename, dynamic_filenames, measurement_filename)
        self.trial_names = self.data.dynamic.dtype.names

        self.measurement_keys          = list(self.data.static.measurements.dtype.names)
        self.measurement_name_to_index = {measurement_key: index for index, measurement_key in enumerate(self.measurement_keys)}

        # add non-overridden default pycgm_calc funcs to funcs list
        self.axis_functions  = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAxes().funcs]
        self.angle_functions = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAngles().funcs]

        # map function names to indices: 'pelvis_axis': 0 ...
        self.axis_function_to_index  = { function.__name__: index for index, function in enumerate(self.axis_functions) }
        self.angle_function_to_index = { function.__name__: index for index, function in enumerate(self.angle_functions) }

        # map function names to the axes they return
        self.axis_function_to_return = return_keys.axes()

        # map function names to the angles they return
        self.angle_function_to_return = return_keys.angles()

        # map returned axis and angle indices so functions can use results of the current frame
        self.axis_keys           = list(chain(*self.axis_function_to_return.values())) # flat list of all returned axes
        self.angle_keys          = list(chain(*self.angle_function_to_return.values())) # flat list of all returned angles
        self.axis_name_to_index  = { axis: index for index, axis in enumerate(self.axis_keys) }
        self.angle_name_to_index = { angle: index for index, angle in enumerate(self.angle_keys) }

        # get default parameter keys
        self.axis_func_parameter_names  = AxisFunctions().parameters()
        self.angle_func_parameter_names = AngleFunctions().parameters()

        self.axis_func_parameters = dict(zip(self.trial_names, []*len(self.trial_names)))
        self.angle_func_parameters = dict(zip(self.trial_names, []*len(self.trial_names)))

        self.update_trial_parameters()


    def update_trial_parameters(self):
        start = time.time()
        for trial in self.trial_names:
            # set parameters of this trial
            self.axis_func_parameters[trial]  = self.names_to_values(self.axis_func_parameter_names, trial)
            self.angle_func_parameters[trial] = self.names_to_values(self.angle_func_parameter_names, trial)
        end = time.time()
        print(f'Time to set all trial parameters: {end-start}')


    def run(self):
        self.axis_functions = [self.calc_axis_pelvis]
        ## TODO change this once all parameters types (axes, angles) 
        ## are in struct and can be passed 

        all_trial_results = []
        for trial in self.trial_names:

            axis_results = []
            angle_results = []

            # TODO consider how functions are going to take in already-calculated axes
            # maybe use [dataset, index/slice] like before
            # data = [self.measurement_values, frame, axis_results, angle_results]

            for index, func in enumerate(self.axis_functions):
                # running using pre-fetched parameters
                start = time.time()

                parameters = self.axis_func_parameters[trial][index]
                ret_axes = np.asarray(func(*parameters))

                if ret_axes.ndim == 3:  # multiple axes returned by one function
                    for axis in ret_axes:
                        axis_results.append(axis)
                else:
                    axis_results.append(ret_axes)

                end = time.time()

                print(f"\t{trial}\t pelvis axis done in {end-start}s")

            all_trial_results.append([np.asarray(axis_results), np.asarray(angle_results)])

        return all_trial_results
    

    def get_markers(self, arr, names, points_only=True, debug=False):
        start = time.time()

        if isinstance(names, str):
            names = [names]
        num_frames = arr[0][0].shape[0]

        if any(name not in arr[0].dtype.names for name in names):
            return None

        rec = rfn.repack_fields(arr[names]).view(subject_utils.frame_dtype()).reshape(len(names), int(num_frames))


        if points_only:
            rec = rec['point'][['x', 'y', 'z']]

        rec = rfn.structured_to_unstructured(rec)

        end = time.time()
        if debug:
            print(f'Time to get {len(names)} markers: {end-start}')

        return rec


    def names_to_values(self, function_list, trial_name):
        """
        takes in a trial's function parameter objects e.g:
        [
            [
                # knee_axis parameters
                Marker('RTHI'),
                Marker('LTHI'),
                Marker('RKNE'),
                Marker('LKNE'),
                Axis('RHipJC'),
                Axis('LHipJC'),
                Measurement('RightKneeWidth'),
                Measurement('LeftKneeWidth')
            ],
            ...other functions
        ]

        as well as the trial name

        returns a list of the function parameter values for that trial
        """

        updated_parameters_list = [[] for _ in range(len(function_list))]

        for function_index, function_parameters in enumerate(function_list):
            for parameter in function_parameters:

                if isinstance(parameter, Marker):
                    # use marker name to retrieve from struct
                    new_parameter = self.get_markers(self.data.dynamic[trial_name].markers, parameter.name, True)
                    if new_parameter is not None:
                        new_parameter = new_parameter[0]
                    updated_parameters_list[function_index].append(new_parameter)

                elif isinstance(parameter, Measurement):
                    # use measurement name to retrieve from struct
                    try:
                        parameter = self.data.static.measurements[parameter.name][0]
                    except ValueError:
                        parameter = None
                        # print(f"{trial_name}\t does not have a measurement named {parameter.name}")

                    updated_parameters_list[function_index].append(parameter)

                # TODO axes and angles in struct
                # elif isinstance(parameter, Axis):
                #     # use axis name to find index
                #     parameter_index = self.axis_name_to_index[parameter.name] if parameter.name in self.axis_name_to_index.keys() else None

                #     # add axis index
                #     updated_parameters_list[function_index].append([parameter.dataset_index, parameter_index])

                # elif isinstance(parameter, Angle):
                #     # use angle name to find index
                #     parameter_index = self.angle_name_to_index[parameter.name] if parameter.name in self.angle_name_to_index.keys() else None

                #     # add angle index
                #     updated_parameters_list[function_index].append([parameter.dataset_index, parameter_index])

                else:
                    # parameter is a constant
                    updated_parameters_list[function_index].append(parameter)

        return updated_parameters_list


    def calc_axis_pelvis(self, rasi, lasi, rpsi, lpsi, sacr=None):
        """
        Make the Pelvis Axis.
        """

        # Verify that the input data is the correct shape
        # print(f"{rasi.shape=}")
        # print(f"{lasi.shape=}")
        # print(f"{rpsi.shape=}")
        # print(f"{lpsi.shape=}")

        # Get the Pelvis Joint Centre
        if sacr is None:
            sacr = (rpsi + lpsi) / 2.0

        # Origin is Midpoint between RASI and LASI
        o = (rasi+lasi)/2.0

        b1 = o - sacr
        b2 = lasi - rasi

        # y is normalized b2
        y = b2 / np.linalg.norm(b2,axis=1)[:, np.newaxis]

        b3 = b1 - ( y * np.sum(b1*y,axis=1)[:, np.newaxis] )
        x = b3/np.linalg.norm(b3,axis=1)[:, np.newaxis]

        # Z-axis is cross product of x and y vectors.
        z = np.cross(x, y)

        new_stack_col = np.column_stack([x,y,z,o])

        return new_stack_col


    def modify_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        """
        modify an existing function's parameters and returned values
        used for overriding a function's parameters or returned results
        """

        if returns_axes is not None and returns_angles is not None:
            raise Exception('{} must return either an axis or an angle, not both'.format(function))

        # get the value or location of parameters
        params = []

        for measurement_name in [measurement_name for measurement_name in (measurements or [])]:
            # add all measurement values
            params.append(Measurement(measurement_name))

        for marker_name in [marker_name for marker_name in (markers or [])]:
            # add all marker slices
            params.append(Marker(marker_name))

        for axis_name in [axis_name for axis_name in (axes or [])]:
            # all all axis indices
            params.append(Axis(axis_name))

        for angle_name in [angle_name for angle_name in (angles or [])]:
            # all all angle indices
            params.append(Angle(angle_name))


        if isinstance(function, str):  # make sure a function name is passed
            if function in self.axis_function_to_index:
                self.axis_func_parameter_names[self.axis_function_to_index[function]] = params

            elif function in self.angle_function_to_index:
                self.angle_func_parameter_names[self.angle_function_to_index[function]] = params

            else:
                raise Exception(('Function {} not found'.format(function)))
        else:
            raise Exception('Pass the name of the function as a string like so: \'{}\''.format(function.__name__))

        if returns_axes is not None:
        # add returned axes, update related attributes

            self.axis_function_to_return[function] = returns_axes

            self.num_axes                          = len(list(chain(*self.axis_function_to_return.values()))) # len(all of the returned axes)
            self.num_axis_floats_per_frame         = self.num_axes * 16
            # self.axis_results_shape                = (self.num_frames, self.num_axes, 4, 4)

            self.axis_name_to_index                = {axis_name: index for index, axis_name in enumerate(self.axis_keys)}

        if returns_angles is not None:
        # add returned angles, update related attributes

            self.angle_function_to_return[function] = returns_angles

            self.num_angles                         = len(list(chain(*self.angle_function_to_return.values()))) # len(all of the returned angles)
            self.num_angle_floats_per_frame         = self.num_angles * 3
            # self.angle_results_shape                = (self.num_frames, self.num_angles, 3)

            self.angle_name_to_index                = {angle_name: index for index, angle_name in enumerate(self.angle_keys)}

        self.update_trial_parameters()


    def add_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        """
        old function, may need to use some variation of it later

        add a custom function to pycgm
        get func object and name
        """
        if isinstance(function, str):
            func_name = function
            func      = getattr(self, func_name)
        elif callable(function):
            func_name = function.__name__
            func      = function

        if returns_axes is not None and returns_angles is not None:
            raise Exception('{} must return either an axis or an angle, not both'.format(func_name))
        if returns_axes is None and returns_angles is None:
            raise Exception('{} must return a custom axis or angle. if the axis or angle already exists by default, just use self.modify_function()'.format(func_name))

        # get the value or location of parameters
        params = []
        for measurement_name in [measurement_name for measurement_name in (measurements or [])]:
            # add all measurement values
            params.append(Measurement(measurement_name))

        for marker_name in [marker_name for marker_name in (markers or [])]:
            # add all marker slices
            params.append(Marker(marker_name))

        for axis_name in [axis_name for axis_name in (axes or [])]:
            # all all axis indices
            params.append(Axis(axis_name))

        for angle_name in [angle_name for angle_name in (angles or [])]:
            # all all angle indices
            params.append(Angle(angle_name))

        if returns_axes is not None:
            # add returned axes, update related attributes

            self.axis_functions.append(func)
            self.axis_func_parameters.append([])
            self.axis_keys.extend(returns_axes)

            self.axis_name_to_index                = { axis_name: index for index, axis_name in enumerate(self.axis_keys) }
            self.axis_function_to_index            = { function.__name__: index for index, function in enumerate(self.axis_functions)}
            self.axis_function_to_return[function] = returns_axes

            self.num_axes                          = len(list(chain(*self.axis_function_to_return.values())))
            self.num_axis_floats_per_frame         = self.num_axes * 16
            # self.axis_results_shape                = (self.num_frames, self.num_axes, 4, 4)

            # set parameters of new function
            self.axis_func_parameters[self.axis_function_to_index[func_name]] = params

        if returns_angles is not None:  # extend angles and update
            # add returned angles, update related attributes

            self.angle_functions.append(func)
            self.angle_func_parameters.append([])
            self.angle_keys.extend(returns_angles)

            self.angle_function_to_index            = { function.__name__: index for index, function in enumerate(self.angle_functions)}
            self.angle_function_to_return[function] = returns_angles
            self.angle_name_to_index                = {angle_name: index for index, angle_name in enumerate(self.angle_keys)}

            self.num_angles                         = len(list(chain(*self.angle_function_to_return.values())))
            self.num_angle_floats_per_frame         = self.num_angles * 3
            # self.angle_results_shape                = (self.num_frames, self.num_angles, 3)

            # set parameters of new function
            self.angle_func_parameters[self.angle_function_to_index[func_name]] = params


class PyCGM():
    def __init__(self, subjects):
        if isinstance(subjects, Subject):
            subjects = [subjects]

        self.subjects = subjects

    def run_all(self):
        for i, subject in enumerate(self.subjects):
            print(f"\nRunning subject {i+1} of {len(self.subjects)}")
            subject.run()


    def structure_trial_axes(self, axis_results):
        """
        old function, may need to use some variation of it later

        takes a flat array of floats that represent the 4x4 axes at each frame
        returns a structured array, indexed by axes[optional frame slice or index][axis name]
        """

        axis_result_keys = list(chain(*self.axis_function_to_return.values()))
        axis_row_dtype = np.dtype([(key, 'f8', (4, 4)) for key in axis_result_keys])

        return np.array([tuple(frame) for frame in axis_results.reshape(self.axis_results_shape)], dtype=axis_row_dtype)

    def structure_trial_angles(self, angle_results):
        """
        old function, may need to use some variation of it later

        takes a flat array of floats that represent the 3x1 angles at each frame
        returns a structured array, indexed by angles[optional frame slice or index][angle name]
        """

        angle_result_keys = list(chain(*self.angle_function_to_return.values()))
        angle_row_dtype   = np.dtype([(key, 'f8', (3,)) for key in angle_result_keys])

        return np.array([tuple(frame) for frame in angle_results.reshape(self.angle_results_shape)], dtype=angle_row_dtype)

