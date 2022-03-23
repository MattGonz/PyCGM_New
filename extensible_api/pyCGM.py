import os
import multiprocessing as mp
from itertools import chain
import numpy as np
from static import getStatic
from pycgm_calc import CalcAxes, CalcAngles
from utils import pycgmIO
# from pipelines import rigid_fill, filtering, prep, clearMarker
from defaults.parameters import Measurement, Marker, Axis, Angle, AxisFunctions, AngleFunctions

class pyCGM():
    def __init__(self, measurements, static_trial, dynamic_trial):

        # get calibrated subject measurements
        measurements_as_dict = dict(zip(measurements[0], measurements[1]))
        self.measurements    = getStatic(static_trial, measurements_as_dict)

        # map the list of measurement names to indices
        self.measurement_keys          = list(self.measurements.keys())
        self.measurement_values        = list(self.measurements.values())
        self.measurement_name_to_index = {measurement_key: index for index, measurement_key in enumerate(self.measurement_keys)}

        # fill, filter, and prep marker data
        # static_trial_dict  = pycgmIO.data_as_dict(static_trial, npArray=True)
        # dynamic_trial_dict = pycgmIO.data_as_dict(dynamic_trial, npArray=True)
        # dynamic_filled     = rigid_fill(dynamic_trial_dict, static_trial_dict)
        # dynamic_filtered   = filtering(dynamic_filled)
        # dynamic_prepped    = prep(dynamic_filtered)

        # convert list of marker data dicts to array of flat marker data arrays
        self.marker_data = pycgmIO.dicts_to_flat_arrays(dynamic_trial)

        # map the list of marker names to slices
        self.marker_keys          = dynamic_trial[0].keys()
        self.marker_name_to_slice = {marker_key: slice(index*3, index*3+3, 1) for index, marker_key in enumerate(self.marker_keys)}

        # add non-overridden default pycgm_calc funcs to funcs list
        self.axis_functions  = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAxes().funcs]
        self.angle_functions = [func if not hasattr(self, func.__name__) else getattr(self, func.__name__) for func in CalcAngles().funcs]

        # map function names to indices: 'pelvis_axis': 0 ...
        self.axis_function_to_index  = { function.__name__: index for index, function in enumerate(self.axis_functions) }
        self.angle_function_to_index = { function.__name__: index for index, function in enumerate(self.angle_functions) }

        # map function names to the axes they return
        self.axis_function_to_return = {'pelvis_axis': ['Pelvis'],
                                        'hip_joint_center': ['RHipJC', 'LHipJC'],
                                        'hip_axis': ['Hip'],
                                        'knee_axis': ['RKnee', 'LKnee'],
                                        'ankle_axis': ['RAnkle', 'LAnkle'],
                                        'foot_axis': ['RFoot', 'LFoot'],
                                        'head_axis': ['Head'],
                                        'thorax_axis': ['Thorax'],
                                        'wand_marker': ['RWand', 'LWand'],
                                        'clav_joint_center': ['RClavJC', 'LClavJC'],
                                        'clav_axis': ['RClav', 'LClav'],
                                        'hum_axis': ['RHum', 'LHum', 'RWristJC', 'LWristJC'],
                                        'rad_axis': ['RRad', 'LRad'],
                                        'hand_axis': ['RHand', 'LHand']}

        # map function names to the angles they return
        self.angle_function_to_return = {'pelvis_angle': ['Pelvis'],
                                         'hip_angle': ['RHip', 'LHip'],
                                         'knee_angle': ['RKnee', 'LKnee'],
                                         'ankle_angle': ['RAnkle', 'LAnkle'],
                                         'foot_angle': ['RFoot', 'LFoot'],
                                         'head_angle': ['Head'],
                                         'thorax_angle': ['Thorax'],
                                         'neck_angle': ['Neck'],
                                         'spine_angle': ['Spine'],
                                         'shoulder_angle': ['RShoulder', 'LShoulder'],
                                         'elbow_angle': ['RElbow', 'LElbow'],
                                         'wrist_angle': ['RWrist', 'LWrist']}

        # map returned axis and angle indices so functions can use results of the current frame
        self.axis_keys           = list(chain(*self.axis_function_to_return.values())) # flat list of all returned axes
        self.angle_keys          = list(chain(*self.angle_function_to_return.values())) # flat list of all returned angles
        self.axis_name_to_index  = { axis: index for index, axis in enumerate(self.axis_keys) }
        self.angle_name_to_index = { angle: index for index, angle in enumerate(self.angle_keys) }

        # structured array helper attributes
        self.num_frames                 = len(self.marker_data)
        self.num_axes                   = len(self.axis_keys)
        self.num_angles                 = len(self.angle_keys)
        self.num_axis_floats_per_frame  = self.num_axes * 16
        self.num_angle_floats_per_frame = self.num_angles * 3
        self.axis_results_shape         = (self.num_frames, self.num_axes, 4, 4)
        self.angle_results_shape        = (self.num_frames, self.num_angles, 3)

        # get default parameter keys
        axis_func_parameter_names  = AxisFunctions().parameters()
        angle_func_parameter_names = AngleFunctions().parameters()

        # set parameters of this trial
        self.axis_func_parameters  = self.names_to_values(axis_func_parameter_names)
        self.angle_func_parameters = self.names_to_values(angle_func_parameter_names)

    def names_to_values(self, function_list):
        """
        convert list of function parameter names:
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
                ...
            ]

        to a list of function parameters with their proper indices:
            [
                [
                    # knee_axis parameters

                    [1, slice(0, 3)],
                    [1, slice(3, 6)],
                    [1, slice(6, 9)], 
                    [1, slice(9, 12)],
                    [2, 0],
                    [2, 1],
                    [0, 20],
                    [0, 21]
                ]
                ...
            ]


        the new parameters are indexed as:
            [dataset, index]

        and the datasets being indexed are:
            [measurements, markers, axes, angles]

        e.g. 
            Marker('RTHI')       = [1, slice(0, 3)]
            data[1][slice(0, 3)] = the first 3 values in marker data

        """
        updated_parameters_list = [[] for i in range(len(function_list))]

        for function_index, function_parameters in enumerate(function_list):
            for parameter in function_parameters:

                if isinstance(parameter, Marker):
                    # use marker name to find slice
                    parameter_index = self.marker_name_to_slice[parameter.name] if parameter.name in self.marker_name_to_slice.keys() else None

                    # add marker slice
                    updated_parameters_list[function_index].append([parameter.dataset_index, parameter_index])

                elif isinstance(parameter, Measurement):
                    # use measurement name to find index
                    parameter_index = self.measurement_name_to_index[parameter.name] if parameter.name in self.measurements.keys() else None

                    # add measurement index
                    updated_parameters_list[function_index].append([parameter.dataset_index, parameter_index])

                elif isinstance(parameter, Axis):
                    # use axis name to find index
                    parameter_index = self.axis_name_to_index[parameter.name] if parameter.name in self.axis_name_to_index.keys() else None

                    # add axis index
                    updated_parameters_list[function_index].append([parameter.dataset_index, parameter_index])

                elif isinstance(parameter, Angle):
                    # use angle name to find index
                    parameter_index = self.angle_name_to_index[parameter.name] if parameter.name in self.angle_name_to_index.keys() else None

                    # add angle index
                    updated_parameters_list[function_index].append([parameter.dataset_index, parameter_index])
                else:
                    # parameter is a constant
                    updated_parameters_list[function_index].append(parameter)

        return updated_parameters_list


    def modify_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        # modify an existing function's parameters and returned values
        # used for overriding an existing function's parameters or returned results

        if returns_axes is not None and returns_angles is not None:
            raise Exception('{} must return either an axis or an angle, not both'.format(function))

        # get the value or location of parameters
        params = []
        for measurement_name in [measurement_name for measurement_name in (measurements or [])]:
            # add all measurement values
            params.append([0, self.measurement_name_to_index[measurement_name]])

        for marker_name in [marker_name for marker_name in (markers or [])]:
            # add all marker slices
            params.append([1, self.marker_name_to_slice[marker_name]])

        for axis_name in [axis_name for axis_name in (axes or [])]:
            # all all axis indices
            params.append([2, self.axis_name_to_index[axis_name]])

        for angle_name in [angle_name for angle_name in (angles or [])]:
            # all all angle indices
            params.append([3, self.angle_name_to_index[angle_name]])

        if isinstance(function, str):  # make sure a function name is passed
            if function in self.axis_function_to_index:
                # set parameters of modified axis function

                self.axis_functions[self.axis_function_to_index[function]] = getattr(self, function)
                self.axis_func_parameters[self.axis_function_to_index[function]] = params

            elif function in self.angle_function_to_index:
                # set parameters of modified angle function

                self.angle_functions[self.angle_function_to_index[function]] = getattr(self, function)
                self.angle_func_parameters[self.angle_function_to_index[function]] = params

            else:
                raise Exception(('Function {} not found'.format(function)))
        else:
            raise Exception('Pass the name of the function as a string like so: \'{}\''.format(function.__name__))

        if returns_axes is not None:
        # add returned axes, update related attributes

            self.axis_function_to_return[function] = returns_axes

            self.num_axes                          = len(list(chain(*self.axis_function_to_return.values()))) # len(all of the returned axes)
            self.num_axis_floats_per_frame         = self.num_axes * 16
            self.axis_results_shape                = (self.num_frames, self.num_axes, 4, 4)

            self.axis_name_to_index                = {axis_name: index for index, axis_name in enumerate(self.axis_keys)}

        if returns_angles is not None:
        # add returned angles, update related attributes

            self.angle_function_to_return[function] = returns_angles

            self.num_angles                         = len(list(chain(*self.angle_function_to_return.values()))) # len(all of the returned angles)
            self.num_angle_floats_per_frame         = self.num_angles * 3
            self.angle_results_shape                = (self.num_frames, self.num_angles, 3)

            self.angle_name_to_index                = {angle_name: index for index, angle_name in enumerate(self.angle_keys)}

    def add_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        # add a custom function to pycgm

        # get func object and name
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
            params.append([0, self.measurement_name_to_index[measurement_name]])

        for marker_name in [marker_name for marker_name in (markers or [])]:
            # add all marker slices
            params.append([1, self.marker_name_to_slice[marker_name]])

        for axis_name in [axis_name for axis_name in (axes or [])]:
            # all all axis indices
            params.append([2, self.axis_name_to_index[axis_name]])

        for angle_name in [angle_name for angle_name in (angles or [])]:
            # all all angle indices
            params.append([3, self.angle_name_to_index[angle_name]])

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
            self.axis_results_shape                = (self.num_frames, self.num_axes, 4, 4)

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
            self.angle_results_shape                = (self.num_frames, self.num_angles, 3)

            # set parameters of new function
            self.angle_func_parameters[self.angle_function_to_index[func_name]] = params

    def structure_trial_axes(self, axis_results):
        # takes a flat array of floats that represent the 4x4 axes at each frame
        # returns a structured array, indexed by axes[optional frame slice or index][axis name]

        axis_result_keys = list(chain(*self.axis_function_to_return.values()))
        axis_row_dtype = np.dtype([(key, 'f8', (4, 4)) for key in axis_result_keys])

        return np.array([tuple(frame) for frame in axis_results.reshape(self.axis_results_shape)], dtype=axis_row_dtype)

    def structure_trial_angles(self, angle_results):
        # takes a flat array of floats that represent the 3x1 angles at each frame
        # returns a structured array, indexed by angles[optional frame slice or index][angle name]

        angle_result_keys = list(chain(*self.angle_function_to_return.values()))
        angle_row_dtype   = np.dtype([(key, 'f8', (3,)) for key in angle_result_keys])

        return np.array([tuple(frame) for frame in angle_results.reshape(self.angle_results_shape)], dtype=angle_row_dtype)

    def multi_run(self, cores=None):
        # parallelize on blocks of frames
        nprocs = cores if cores is not None else os.cpu_count() - 1

        # create a shared array to store axis and angle results
        shared_axes        = mp.RawArray('d', self.num_frames * self.num_axes * 16)
        shared_angles      = mp.RawArray('d', self.num_frames * self.num_angles * 3)
        flat_rows          = self.marker_data
        marker_data_blocks = np.array_split(flat_rows, nprocs)

        processes = []
        frame_index_offset = 0
        for i in range(nprocs):
            frame_count = len(marker_data_blocks[i])

            processes.append(mp.Process(target=self.run,
                                        args=(marker_data_blocks[i],
                                              frame_index_offset,
                                              frame_index_offset + frame_count,
                                              self.num_axis_floats_per_frame,
                                              self.num_angle_floats_per_frame,
                                              shared_axes,
                                              shared_angles),
                                        daemon=True))

            processes[i].start()
            frame_index_offset += frame_count

        for process in processes:
            process.join()

        # structure flat axis and angle results
        self.axes   = self.structure_trial_axes(  np.frombuffer(shared_axes,   dtype=np.float64))
        self.angles = self.structure_trial_angles(np.frombuffer(shared_angles, dtype=np.float64))

    def run(self, frames=None, index_offset=None, index_end=None, axis_results_size=None, angle_results_size=None, shared_axes=None, shared_angles=None):

        flat_axis_results  = np.array([], dtype=np.float64)
        flat_angle_results = np.array([], dtype=np.float64)

        if shared_angles is not None:  # multiprocessing, write to shared memory
            shared_axes   = np.frombuffer(shared_axes,   dtype=np.float64)
            shared_angles = np.frombuffer(shared_angles, dtype=np.float64)

            for frame in frames:
                flat_axis_results  = np.append(flat_axis_results,  self.calc(frame)[0].flatten())
                flat_angle_results = np.append(flat_angle_results, self.calc(frame)[1].flatten())

            shared_axes[  index_offset * axis_results_size:  index_end * axis_results_size]  = flat_axis_results
            shared_angles[index_offset * angle_results_size: index_end * angle_results_size] = flat_angle_results

        else:  # single core, just calculate and structure
            for frame in self.marker_data:
                axes, angles = self.calc(frame)
                flat_axis_results  = np.append(flat_axis_results,  axes.flatten())
                flat_angle_results = np.append(flat_angle_results, angles.flatten())

            # structure flat axis and angle results
            self.axes   = self.structure_trial_axes(flat_axis_results)
            self.angles = self.structure_trial_angles(flat_angle_results)

    def calc(self, frame):
        axis_results = []
        angle_results = []

        data = [self.measurement_values, frame, axis_results, angle_results]

        # run axis functions in sequence
        for index, func in enumerate(self.axis_functions):
            axis_params = []

            for param in self.axis_func_parameters[index]:
                # param[0]: which dataset the parameter belongs to
                    # measurement = 0, marker = 1, axis = 2, angle = 3
                # param[1]: the index or slice of the parameter in its respective dataset

                try:
                    axis_params.append(data[param[0]][param[1]])
                except TypeError: # param is a constant e.g. 7.0
                    axis_params.append(param)


            ret_axes = np.asarray(func(*axis_params))

            if ret_axes.ndim == 3:  # multiple axes returned by one function
                for axis in ret_axes:
                    axis_results.append(axis)
            else:
                axis_results.append(ret_axes)

        # run angle functions in sequence
        for index, func in enumerate(self.angle_functions):
            angle_params = []

            for param in self.angle_func_parameters[index]:
                # param[0]: which dataset the parameter belongs to
                    # measurement = 0, marker = 1, axis = 2, angle = 3
                # param[1]: the index or slice of the parameter in its respective dataset

                try:
                    angle_params.append(data[param[0]][param[1]])
                except TypeError: # param is a constant e.g. 7.0
                    angle_params.append(param)

            ret_angles = np.asarray(func(*angle_params))

            if ret_angles.ndim == 2:  # multiple angles returned by one function
                for angle in ret_angles:
                    angle_results.append(angle)
            else:
                angle_results.append(ret_angles)

        return [np.asarray(axis_results), np.asarray(angle_results)]
