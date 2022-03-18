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

        # Add non-overridden default pycgm_calc funcs to funcs list
        self.axis_functions  = self.get_axis_functions()
        self.angle_functions = self.get_angle_functions()

        # HACK for testing since only a few functions are currently vectorized
        self.axis_functions = self.axis_functions[:4]

        # Map function names to indices: 'calc_pelvis_axis': 0 ...
        self.axis_execution_order, self.angle_execution_order = self.map_function_names_to_index()

        # Map function names to the names of the data they return: 'calc_pelvis_axis': ['Pelvis'] ...
        self.axis_function_to_return, self.angle_function_to_return = self.map_function_names_to_returns()

        # Update returned axis and angle list to be used in their respective structed
        # array datatypes
        #   self.axis_keys:  ['Pelvis','RHipJC', 'LHipJC', 'Hip',   'RKnee', 'LKnee', ...]
        #   self.angle_keys: ['Pelvis','RHip',   'LHip',   'RKnee', 'LKnee',  ...]
        self.axis_keys, self.angle_keys = self.update_return_keys()           

        # Make empty structured arrays of returned axes and angles
        self.axis_results  = self.make_axis_struct()
        self.angle_results = self.make_angle_struct()

        # Get default parameter objects
        self.axis_func_parameter_names  = AxisFunctions().parameters()
        self.angle_func_parameter_names = AngleFunctions().parameters()

        # Expand required parameter names to their values in each trial's dataset
        self.axis_func_parameters, self.angle_func_parameters = self.update_trial_parameters()


    def get_axis_functions(self):
        """
        Initialize axis functions from pycgm_calc.CalcAxes if they have not
        already been defined in a custom model.
        """
        axis_functions = []
        for func in CalcAxes().funcs:
            if hasattr(self, func.__name__):
                axis_functions.append(getattr(self, func.__name__))
            else:
                axis_functions.append(func)

        return axis_functions


    def get_angle_functions(self):
        """
        Initialize angle functions from pycgm_calc.CalcAngles if they have not
        already been defined in a custom model.
        """
        angle_functions = []
        for func in CalcAngles().funcs:
            if hasattr(self, func.__name__):
                angle_functions.append(getattr(self, func.__name__))
            else:
                angle_functions.append(func)

        return angle_functions


    def map_function_names_to_index(self):
        """Map function names to indices.

        e.g: Axis functions
        self.axis_execution_order  = { 'calc_pelvis_axis': 0,
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
        axis_execution_order  = {}
        angle_execution_order = {}

        for index, function in enumerate(self.axis_functions):
            axis_execution_order[function.__name__] = index

        for index, function in enumerate(self.angle_functions):
            angle_execution_order[function.__name__] = index

        return axis_execution_order, angle_execution_order


    def map_function_names_to_returns(self):
        """Map function names to the names of the data they return.

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
        axis_function_to_return  = return_keys.axes()
        angle_function_to_return = return_keys.angles()
        return axis_function_to_return, angle_function_to_return

    def update_return_keys(self):
        axis_keys = list(chain.from_iterable(self.axis_function_to_return.values()))
        angle_keys = list(chain.from_iterable(self.angle_function_to_return.values()))

        return axis_keys, angle_keys

    def make_axis_struct(self):
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

        axis_results = {}
        for trial_name in self.trial_names:
            num_frames = self.data.dynamic[trial_name].markers[0][0].shape[0]
            axis_dtype = np.dtype([(key, 'f8', ((num_frames, 3, 4))) for key in self.axis_keys])
            axis_results[trial_name] = np.zeros([], dtype=axis_dtype)

        return axis_results


    def make_angle_struct(self):
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

        angle_results = {}
        for trial_name in self.trial_names:
            num_frames  = self.data.dynamic[trial_name].markers[0][0].shape[0]
            angle_dtype = np.dtype([(key, 'f8', ((num_frames, 3))) for key in self.angle_keys])
            angle_results[trial_name] = np.zeros([], dtype=angle_dtype)

        return angle_results


    def names_to_values(self, function_parameters, trial_name):
        """Convert a list of function parameter objects to their values in each trial's dataset

        Parameters
        ----------
        function_parameters : list of lists of parameter objects
            Required parameter objects for all functions
        trial_name : str
            The name of the trial
        
        Returns
        -------
        updated_parameters_list : list of list of ndarray
            The values of all specified parameters of all functions of the given trial

        Notes
        -----
        function_parameters is a list of lists of parameter objects like so:
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
        """

        updated_parameters_list = [[] for _ in range(len(function_parameters))]

        for function_index, function_parameters in enumerate(function_parameters):
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

                    updated_parameters_list[function_index].append(new_parameter)

                elif isinstance(parameter, Axis):
                    # Add parameter from axis_results struct
                    updated_parameters_list[function_index].append(self.axis_results[trial_name][parameter.name])

                elif isinstance(parameter, Angle):
                    # Add parameter from angle_results struct
                    updated_parameters_list[function_index].append(self.angle_results[trial_name][parameter.name])

                else:
                    # Parameter is a constant, append as is
                    updated_parameters_list[function_index].append(parameter)

        return updated_parameters_list


    def update_trial_parameters(self):
        axis_func_parameters  = {}
        angle_func_parameters = {}
        for trial in self.trial_names:
            axis_func_parameters[trial]  = self.names_to_values(self.axis_func_parameter_names,  trial)
            angle_func_parameters[trial] = self.names_to_values(self.angle_func_parameter_names, trial)

        return axis_func_parameters, angle_func_parameters


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
                parameters = self.axis_func_parameters[trial_name][self.axis_execution_order[func.__name__]]
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

                print(f"\t{trial_name:<20}\t{func.__name__:<25}\t{end-start:.5f}s")

        start = time.time()
        self.structure_model_output()
        end = time.time()
        print(f'\tTime to structure output:\t\t\t\t{end-start:.5f}s\n')


    def structure_model_output(self):
        """Recreates the original model structure, but with the model outputs inserted.

        Notes
        -----
        Using model = Model():

        Accessing measurement data:
            model.data.static.measurements.{measurement name}
            e.g. model.data.static.measurements.LeftLegLength

        Accessing static trial data:
            model.data.static.markers.{marker name}.point.{x, y, z}
            e.g. model.data.static.markers.LASI.point.x

        Accessing dynamic trial data:
            Input markers:
                model.data.dynamic.{filename}.markers.{marker name}.point.{x, y, z}
                e.g. model.data.RoboWalk.markers.LASI.point.x
            Output axes:
                model.data.dynamic.{filename}.axes.{axis name}
                e.g. model.data.RoboWalk.axes.Pelvis
            Output angles:
                model.data.dynamic.{filename}.angles.{angle name}
                e.g. model.data.RoboWalk.angles.RHip
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
        subject['static']['markers']      = self.data.static.markers
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
        """Modify an existing function's parameters and returned values

        Parameters
        ----------
        function : str
            Name of the function that is to be modified.
        measurements : list of str, optional
            Name(s) of required measurement parameters.
        markers : list of str, optional
            Name(s) of required marker parameters.
        axes : list of str, optional
            Name(s) of required axis parameters.
        angles : list of str, optional
            Name(s) of required angle parameters.
        returns_axes : list of str, optional
            Name(s) of returned axes.
        returns_angles : list of str, optional
            Name(s) of returned angles.

        Raises
        ------
        Exception
            If the function returns both axes and angles
            If function is not of type str
        """

        if returns_axes is not None and returns_angles is not None:
            raise Exception(f'{function} must return either an axis or an angle, not both')

        # Create list of parameter objects from parameter names
        params = []
        for measurement_name in [measurement_name for measurement_name in (measurements or [])]:
            params.append(Measurement(measurement_name))

        for marker_name in [marker_name for marker_name in (markers or [])]:
            params.append(Marker(marker_name))

        for axis_name in [axis_name for axis_name in (axes or [])]:
            params.append(Axis(axis_name))

        for angle_name in [angle_name for angle_name in (angles or [])]:
            params.append(Angle(angle_name))


        if isinstance(function, str):  # make sure a function name is passed
            if function in self.axis_execution_order:
                self.axis_func_parameter_names[self.axis_execution_order[function]] = params

            elif function in self.angle_execution_order:
                self.angle_func_parameter_names[self.angle_execution_order[function]] = params

            else:
                raise KeyError(f"Unable to find function {function} in execution order")
        else:
            raise Exception(f'Pass the name of the function as a string like so: \'{function.__name__}\'')

        if returns_axes is not None:
            # Add returned axes, update related attributes
            self.axis_function_to_return[function] = returns_axes

        if returns_angles is not None:
            # Add returned angles, update related attributes
            self.angle_function_to_return[function] = returns_angles

        # Expand required parameter names to their values in each trial's dataset
        self.axis_func_parameters, self.angle_func_parameters = self.update_trial_parameters()


    def add_function(self, function, order=None, measurements=None, markers=None, axes=None, angles=None, returns_axes=None, returns_angles=None):
        """Add a custom function to the model.

        Parameters
        ----------
        function : str or function
            Name or function object that is to be added.
        order : list or tuple of [str, int], optional
            Index in the execution order the function is to be run, represented as [function_name, offset].
        measurements : list of str, optional
            Name(s) of required measurement parameters.
        markers : list of str, optional
            Name(s) of required marker parameters.
        axes : list of str, optional
            Name(s) of required axis parameters.
        angles : list of str, optional
            Name(s) of required angle parameters.
        returns_axes : list of str, optional
            Name(s) of returned axes.
        returns_angles : list of str, optional
            Name(s) of returned angles.

        Raises
        ------
        Exception
            If the function returns both axes and angles
            If the function does not return a custom axis or angle
        KeyError
            If function is not found in the function execution order.
        
        Notes
        -----
        order is represented by [function_name, offset]
            - function_name is the name of the function that the new function will be run relative to.
            - offset is offset from the target function_name that the new function will be run.
            - an order of ['calc_axis_knee', -1] will run the custom function immediately before calc_axis_knee.
        """

        def insert_axis_function(target_function_name, offset, func):
            """Insert a custom axis function at the desired offset from a target function name.

            Parameters
            ----------
            target_function_name : str
                The name of the function that the axis function will be run relative to.
            offset : int
                The offset from the target function that the new function will be run.

            Raises
            ------
            KeyError
                If target_function_name is not found in the axis function execution order.

            Notes
            -----
                A target name of 'calc_axis_knee' and an offset of -1 will run the custom function
                immediately before calc_axis_knee.
            """

            # Get the index in the execution order where the new function is to be run
            try:
                target_index = self.axis_execution_order[target_function_name] + offset
            except KeyError:
                raise KeyError(f"Unable to find function {target_function_name} in axis execution order")

            # If the target index is out of bounds, append the function to the end or beginning
            if target_index > len(self.axis_execution_order):
                target_index = len(self.axis_execution_order)
            elif target_index < 0:
                target_index = 0

            # Insert at specified index and update execution order
            self.axis_functions.insert(target_index, func)
            self.axis_execution_order, self.angle_execution_order = self.map_function_names_to_index()

            # Extend the returned axes of the function BEFORE the new function
            # e.g. calc_joint_center_hip returns [RHipJC, LHipJC]
            #      We don't want to position the returned axes between RHipJC and LHipJC,
            #      so calc_joint_center_hip's returned axes must be extended
            function_to_extend = self.axis_functions[target_index - 1].__name__
            self.axis_function_to_return[function_to_extend].extend(returns_axes)

            # Update return keys 
            self.axis_keys, self.angle_keys = self.update_return_keys()           

            # Insert the function's parameters into the target index
            self.axis_func_parameter_names.insert(self.axis_execution_order[func_name], params)

            # Add empty space for custom function's parameters in each trial
            for trial_name in self.trial_names:
                self.axis_func_parameters[trial_name].insert(target_index+1, [])


        def insert_angle_function(target_function_name, offset, func):
            """Insert a custom angle function at the desired offset from a target function name.

            Parameters
            ----------
            target_function_name : str
                The name of the function that the angle function will be run relative to.
            offset : int
                The offset from the target function that the new function will be run.

            Raises
            ------
            KeyError
                If target_function_name is not found in the angle function execution order.

            Notes
            -----
                A target name of 'calc_angle_knee' and an offset of -1 will run the custom function
                immediately before calc_angle_knee.
            """

            # Get the index in the execution order where the new function is to be run
            try:
                target_index = self.angle_execution_order[target_function_name] + offset
            except KeyError:
                raise KeyError(f"Unable to find function {target_function_name} in angle execution order")

            # If the target index is out of bounds, append the function to the end
            if target_index > len(self.angle_execution_order):
                target_index = len(self.angle_execution_order)
            elif target_index < 0:
                target_index = 0

            # Insert at specified index and update execution order
            self.angle_functions.insert(target_index, func)
            self.angle_execution_order, self.angle_execution_order = self.map_function_names_to_index()

            # Extend the returned angles of the function BEFORE the new function
            # e.g. calc_joint_center_hip returns [RHipJC, LHipJC]
            #      We don't want to position the returned angles between RHipJC and LHipJC,
            #      so calc_joint_center_hip's returned angles must be extended
            function_to_extend = self.angle_functions[target_index - 1].__name__
            self.angle_function_to_return[function_to_extend].extend(returns_angles)

            # Update return keys 
            self.angle_keys, self.angle_keys = self.update_return_keys()           

            # Insert the function's parameters into the target index
            self.angle_func_parameter_names.insert(self.angle_execution_order[func_name], params)

            # Add empty space for custom function's parameters in each trial
            for trial_name in self.trial_names:
                self.angle_func_parameters[trial_name].insert(target_index+1, [])


        # Get func object and name
        if isinstance(function, str):
            func_name = function
            func      = getattr(self, func_name)
        elif callable(function):
            func_name = function.__name__
            func      = function

        if returns_axes is not None and returns_angles is not None:
            raise Exception(f'{func_name} must return either an axis or an angle, not both')
        if returns_axes is None and returns_angles is None:
            raise Exception(f'{func_name} must return a custom axis or angle. If the axis or angle already exists in the model, use self.modify_function()')

        # Create list of parameter objects from parameter names
        params = []
        for measurement_name in [measurement_name for measurement_name in (measurements or [])]:
            params.append(Measurement(measurement_name))

        for marker_name in [marker_name for marker_name in (markers or [])]:
            params.append(Marker(marker_name))

        for axis_name in [axis_name for axis_name in (axes or [])]:
            params.append(Axis(axis_name))

        for angle_name in [angle_name for angle_name in (angles or [])]:
            params.append(Angle(angle_name))


        if returns_axes is not None:
            # Add returned axes, update related attributes

            if order is not None:
                # Find the position of the function name in the execution order
                # Then apply the offset to get the desired execution index
                # e.g. ['calc_axis_knee', -1] -> run the custom function right before calc_axis_knee

                target_function_name = order[0]
                offset = order[1] + 1

                if offset > 0:
                    offset -= 1

                insert_axis_function(target_function_name, offset, func)

            else:
                # Append to the end
                self.axis_functions.append(func)
                self.axis_keys.extend(returns_axes)

                for trial_name in self.trial_names:
                    self.axis_func_parameters[trial_name].append([])

            # Update parameters and returns
            self.axis_func_parameter_names[self.axis_execution_order[function]] = params
            self.axis_function_to_return[function] = returns_axes

            # Update structured axis array dtype
            self.axis_results = self.make_axis_struct()

        if returns_angles is not None:
            # Add returned angles, update related attributes

            if order is not None:
                # Find the position of the function name in the execution order
                # Then apply the offset to get the desired execution index
                # e.g. ['calc_angle_knee', -1] -> run the custom function right before calc_angle_knee

                target_function_name = order[0]
                offset = order[1] + 1

                if offset > 0:
                    offset -= 1

                insert_angle_function(target_function_name, offset, func)

            else:
                # Append to the end
                self.angle_functions.append(func)
                self.angle_keys.extend(returns_angles)

                for trial_name in self.trial_names:
                    self.angle_func_parameters[trial_name].append([])

            # Update parameters and returns
            self.angle_func_parameter_names[self.angle_execution_order[function]] = params
            self.angle_function_to_return[function] = returns_angles

            # Update structured angle array dtype
            self.angle_results = self.make_angle_struct()

        # Expand required parameter names to their values in each trial's dataset
        self.axis_func_parameters, self.angle_func_parameters = self.update_trial_parameters()


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

