import time
from itertools import chain

import numpy as np
import numpy.lib.recfunctions as rfn
from utils import subject_utils
from defaults.parameters import Measurement, Marker, Axis, Angle, AxisFunctions, AngleFunctions
from defaults import return_keys
from pycgm_calc import CalcAngles, CalcAxes


class ModelCreator():
    def __init__(self, static_filename, dynamic_filenames, measurement_filename):
        self.data = subject_utils.structure_subject(static_filename, dynamic_filenames, measurement_filename)
        self.trial_names = self.data.dynamic.dtype.names

        self.measurement_keys          = list(self.data.static.measurements.dtype.names)
        self.measurement_name_to_index = {measurement_key: index for index, measurement_key in enumerate(self.measurement_keys)}

        # add non-overridden default pycgm_calc funcs to funcs list
        self.set_axis_functions()
        self.set_angle_functions()

        # HACK for testing since only a few functions are currently vectorized
        self.axis_functions = self.axis_functions[:3]

        # map function names to indices: 'calc_pelvis_axis': 0 ...
        self.map_function_names_to_index()

        # map function names to the names of the data they return: 'calc_pelvis_axis': ['Pelvis'] ...
        self.map_function_names_to_returns()

        # map returned axis and angle indices so functions can use results of the current frame
        # flat list of all returned axis names
        # e.g. ['Pelvis','RHipJC', 'LHipJC','Hip','RKnee', 'LKnee', ...]
        self.axis_keys           = list(chain.from_iterable(self.axis_function_to_return.values()))
        self.angle_keys          = list(chain.from_iterable(self.angle_function_to_return.values()))
        self.axis_name_to_index  = { axis: index for index, axis in enumerate(self.axis_keys) }
        self.angle_name_to_index = { angle: index for index, angle in enumerate(self.angle_keys) }

        self.set_axis_struct()
        self.set_angle_struct()

        # get default parameter keys
        self.axis_func_parameter_names  = AxisFunctions().parameters()
        self.angle_func_parameter_names = AngleFunctions().parameters()

        self.axis_func_parameters = dict(zip(self.trial_names, []*len(self.trial_names)))
        self.angle_func_parameters = dict(zip(self.trial_names, []*len(self.trial_names)))

        self.update_trial_parameters()


    def set_axis_functions(self):
        """
        Initialize axis functions from pycgm_calc.CalcAxes if they have not
        already been defined in a custom model.
        """
        self.axis_functions = []
        for func in CalcAxes().funcs:
            if hasattr(self, func.__name__):
                self.axis_functions.append(getattr(self, func.__name__))
            else:
                self.axis_functions.append(func)


    def set_angle_functions(self):
        """
        Initialize angle functions from pycgm_calc.CalcAngles if they have not
        already been defined in a custom model.
        """
        self.angle_functions = []
        for func in CalcAngles().funcs:
            if hasattr(self, func.__name__):
                self.angle_functions.append(getattr(self, func.__name__))
            else:
                self.angle_functions.append(func)


    def map_function_names_to_index(self):
        """
        Map function names to indices.

        e.g: Axis functions
        self.axis_function_to_index  = { 'calc_pelvis_axis': 0,
                                         'calc_joint_center_hip: 1,
                                         ...
                                       }

        e.g: Angle functions
        self.angle_function_to_index = { 'calc_angle_pelvis': 0,
                                         'calc_angle_hip': 1,
                                         ...
                                       }

        }
        """
        self.axis_function_to_index  = {}
        self.angle_function_to_index = {}

        for index, function in enumerate(self.axis_functions):
            self.axis_function_to_index[function.__name__] = index

        for index, function in enumerate(self.angle_functions):
            self.angle_function_to_index[function.__name__] = index


    def map_function_names_to_returns(self):
        """
        Map function names to the names of the data they return.

        e.g: Axis functions
        self.axis_function_to_return  = { 'calc_pelvis_axis': ['Pelvis'],
                                          'calc_joint_center_hip: ['RHipJC', 'LHipJC'],
                                          ...
                                        }

        e.g: Angle functions
        self.angle_function_to_return = { 'calc_angle_pelvis': ['Pelvis'],
                                          'calc_angle_hip': ['RHip', 'LHip'],
                                          ...
                                        }

        }
        """
        self.axis_function_to_return = return_keys.axes()
        self.angle_function_to_return = return_keys.angles()


    def set_axis_struct(self):
        """
        Create a dictionary where each key is a trial name and each value
        is a structured array of that trial's calculated axes.

        Example of dtypes:
        {'6000FrameTrial': [
                            ('Pelvis', '<f8', (6000, 3, 4)),
                            ('RHipJC', '<f8', (6000, 3, 4)), 
                            ('LHipJC', '<f8', (6000, 3, 4)),
                            ...
                        ],
         '59993FrameTrial': [
                            ('Pelvis', '<f8', (59993, 3, 4)),
                            ('RHipJC', '<f8', (59993, 3, 4)), 
                            ('LHipJC', '<f8', (59993, 3, 4)),
                            ...
                        ],
        }
        """

        self.axis_results = {}
        for trial_name in self.trial_names:
            num_frames = self.data.dynamic[trial_name].markers[0][0].shape[0]
            axis_dtype = np.dtype([(key, 'f8', ((num_frames, 3, 4))) for key in self.axis_keys])
            self.axis_results[trial_name] = np.zeros([], dtype=axis_dtype)


    def set_angle_struct(self):
        """
        Create a dictionary where each key is a trial name and each value
        is a structured array of that trial's calculated angles.

        Example of dtypes:
        {'6000FrameTrial': [
                            ('Pelvis', '<f8', (6000, 3)),
                            ('RHip',   '<f8', (6000, 3)), 
                            ('LHip',   '<f8', (6000, 3)),
                            ...
                        ],
         '59993FrameTrial': [
                            ('Pelvis', '<f8', (59993, 3)),
                            ('RHip',   '<f8', (59993, 3)), 
                            ('LHip',   '<f8', (59993, 3)),
                            ...
                        ],
        }
        """

        self.angle_results = {}
        for trial_name in self.trial_names:
            num_frames = self.data.dynamic[trial_name].markers[0][0].shape[0]
            angle_dtype = np.dtype([(key, 'f8', ((num_frames, 3))) for key in self.angle_keys])
            self.angle_results[trial_name] = np.zeros([], dtype=angle_dtype)


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
                    # Use marker name to retrieve from marker struct
                    new_parameter = self.get_markers(self.data.dynamic[trial_name].markers, parameter.name, True)
                    if new_parameter is not None:
                        new_parameter = new_parameter[0]
                    updated_parameters_list[function_index].append(new_parameter)

                elif isinstance(parameter, Measurement):
                    # Use measurement name to retrieve from measurements struct
                    try:
                        new_parameter = self.data.static.measurements[parameter.name][0]
                    except ValueError:
                        new_parameter = None
                        # print(f"{trial_name}\t does not have a measurement named {parameter.name}")

                    updated_parameters_list[function_index].append(new_parameter)

                elif isinstance(parameter, Axis):
                    # Add parameter from axis_results struct
                    updated_parameters_list[function_index].append(self.axis_results[trial_name][parameter.name])

                elif isinstance(parameter, Angle):
                    updated_parameters_list[function_index].append(self.angle_results[trial_name][parameter.name])

                else:
                    # parameter is a constant
                    updated_parameters_list[function_index].append(parameter)

        return updated_parameters_list


    def update_trial_parameters(self):
        for trial in self.trial_names:
            self.axis_func_parameters[trial]  = self.names_to_values(self.axis_func_parameter_names, trial)
            self.angle_func_parameters[trial] = self.names_to_values(self.angle_func_parameter_names, trial)


class Model(ModelCreator):
    def __init__(self, static_filename, dynamic_filenames, measurement_filename):
        super().__init__(static_filename, dynamic_filenames, measurement_filename)


    def run(self):
        """
        Run each trial in the model and insert output values into 
        their respective axis_results and angle_results structs.
        """

        for trial_name in self.trial_names:
            for index, func in enumerate(self.axis_functions):

                # Retrieve the names of the axes returned by this function
                # e.g. 'calc_axis_pelvis' -> 'Pelvis'
                returned_axis_names = self.axis_function_to_return[func.__name__]

                start = time.time()

                # Get the parameters for this function, run it
                parameters = self.axis_func_parameters[trial_name][index]
                ret_axes = np.array(func(*parameters))

                # Insert returned axes into the self.axis_results structured array
                if ret_axes.ndim == 4:
                    # Multiple axes returned by one function
                    for ret_axes_index, axis in enumerate(ret_axes):
                        # Insert each axis into axis_results
                        self.axis_results[trial_name][returned_axis_names[ret_axes_index]] = axis

                else:
                    # Insert returned axis into axis_results
                    self.axis_results[trial_name][returned_axis_names[0]] = ret_axes

                end = time.time()

                print(f"\t{trial_name[:10]}...\t{func.__name__}\t{end-start:.5f}s")

        start = time.time()
        self.structure_model_output()
        end = time.time()
        print(f'\tTime to structure model output:\t\t{end-start:.5f}s\n')


    def structure_model_output(self):
        """
        Recreates the original model structure, but with the 
        model outputs inserted.

        Notes
        -----
        Accessing measurement data:
            subject.static.measurements.{measurement name}
            e.g. subject.static.measurements.LeftLegLength

        Accessing static trial data:
            subject.static.markers.{marker name}.point.{x, y, z}
            e.g. subject.static.markers.LASI.point.x

        Accessing dynamic trial data:
            Input markers:
                subject.dynamic.{filename}.markers.{marker name}.point.{x, y, z}
                e.g. subject.RoboWalk.markers.LASI.point.x
            Output axes:
                subject.dynamic.{filename}.axes.{axis name}
                e.g. subject.RoboWalk.axes.Pelvis
            Output angles:
                subject.dynamic.{filename}.angles.{angle name}
                e.g. subject.RoboWalk.angles.RHip
        """

        dynamic_dtype = []

        for trial_name in self.trial_names:
            axis_output_dtype   = self.axis_results[trial_name].dtype
            angle_output_dtype  = self.angle_results[trial_name].dtype

            marker_input_dtype = self.data.dynamic[trial_name].markers.dtype
            
            trial_dtype = [('markers', marker_input_dtype), \
                           ('axes', axis_output_dtype),     \
                           ('angles', angle_output_dtype)]

            dynamic_dtype.append((trial_name, trial_dtype))


        subject_dtype = [('static', [('markers', self.data.static.markers.dtype), \
                                     ('measurements', self.data.static.measurements.dtype)]), \
                         ('dynamic', dynamic_dtype)]

        subject = np.zeros((1), dtype=subject_dtype)
        subject['static']['markers'] = self.data.static.markers
        subject['static']['measurements'] = self.data.static.measurements

        for trial_name in self.trial_names:
            subject['dynamic'][trial_name]['markers'] = self.data.dynamic[trial_name].markers
            subject['dynamic'][trial_name]['axes']    = self.axis_results[trial_name]
            subject['dynamic'][trial_name]['angles']  = self.angle_results[trial_name]

        subject = subject.view(np.recarray)

        self.data = subject
            

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


    def modify_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        """
        Modify an existing function's parameters and returned values
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
            self.axis_name_to_index                = {axis_name: index for index, axis_name in enumerate(self.axis_keys)}

        if returns_angles is not None:
        # add returned angles, update related attributes

            self.angle_function_to_return[function] = returns_angles
            self.angle_name_to_index                = {angle_name: index for index, angle_name in enumerate(self.angle_keys)}

        self.update_trial_parameters()


    def add_function(self, function, markers=None, measurements=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        """
        Add a custom function to the model.
        """

        # Get func object and name
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
            self.axis_keys.extend(returns_axes)
            self.map_function_names_to_index()

            self.axis_name_to_index                = { axis_name: index for index, axis_name in enumerate(self.axis_keys) }
            self.axis_function_to_return[function] = returns_axes
            self.axis_func_parameter_names[self.axis_function_to_index[function]] = params
            self.set_axis_struct()

            for trial_name in self.trial_names:
                self.axis_func_parameters[trial_name].append([])

        if returns_angles is not None:  # extend angles and update
            # add returned angles, update related attributes

            self.angle_functions.append(func)
            self.angle_keys.extend(returns_angles)
            self.map_function_names_to_index()

            self.angle_function_to_return[function] = returns_angles
            self.angle_name_to_index                = {angle_name: index for index, angle_name in enumerate(self.angle_keys)}
            self.angle_func_parameter_names[self.angle_function_to_index[function]] = params
            self.set_angle_struct()

            for trial_name in self.trial_names:
                self.angle_func_parameters[trial_name].append([])

        self.update_trial_parameters()


class PyCGM():
    def __init__(self, subjects):
        if isinstance(subjects, Model):
            subjects = [subjects]

        self.subjects = subjects


    def run_all(self):
        for i, subject in enumerate(self.subjects):
            print(f"Running subject {i+1} of {len(self.subjects)}")
            subject.run()


    def __getitem__(self, index):
        return self.subjects[index]

