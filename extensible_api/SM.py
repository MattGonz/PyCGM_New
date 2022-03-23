import multiprocessing as mp
import numpy as np
from os import cpu_count

class SubjectManager():

        def __init__(self, list_of_subjects):
            self.subject_list = list_of_subjects

        def multi_run(self):
            # parallelize on trials

            # get the total number of floats in all trials for the flat shared buffer
            num_axis_floats =  0
            num_angle_floats = 0
            for subject in self.subject_list:
                num_axis_floats += subject.num_frames * subject.num_axes * subject.num_axis_floats_per_frame
                num_angle_floats += subject.num_frames * subject.num_angles * subject.num_angle_floats_per_frame

            # create a shared array of all subjects' results
            shared_axes = mp.RawArray('f', num_axis_floats)
            shared_angles = mp.RawArray('f', num_angle_floats)

            nprocs = cpu_count() - 1
            split_subjects_by_core = np.array_split(np.asarray(self.subject_list), nprocs)

            processes = []
            axis_start_offset = 0
            angle_start_offset = 0
            for i in range(nprocs):
                core_axis_float_count = np.sum([np.prod(subject.axis_results_shape) for subject in split_subjects_by_core[i]])
                core_angle_float_count = np.sum([np.prod(subject.angle_results_shape) for subject in split_subjects_by_core[i]])

                if core_axis_float_count > 0 and core_angle_float_count > 0:
                    processes.append(mp.Process(
                        target=self.run_core_subjects,
                        args=(split_subjects_by_core,
                            i,
                            shared_axes,
                            shared_angles,
                            axis_start_offset,
                            axis_start_offset + core_axis_float_count,
                            angle_start_offset,
                            angle_start_offset + core_angle_float_count)))

                    axis_start_offset += core_axis_float_count
                    angle_start_offset += core_angle_float_count

            for process in processes:
                process.start()
            for process in processes:
                process.join()

            # reconstruct per-subject results and set values of objects in main process
            axis_floats_offset = 0
            angle_floats_offset = 0
            for subject in self.subject_list:
                axis_floats_count = np.prod((subject.axis_results_shape))
                angle_floats_count = np.prod((subject.angle_results_shape))

                subject.axes = subject.structure_trial_axes(np.asarray(shared_axes[axis_floats_offset:axis_floats_offset + axis_floats_count]))
                subject.angles = subject.structure_trial_angles(np.asarray(shared_angles[angle_floats_offset:angle_floats_offset + angle_floats_count]))

                axis_floats_offset += axis_floats_count
                angle_floats_offset += angle_floats_count

        def run_core_subjects(self, split_subjects_by_core, core_index, shared_axes, shared_angles, axis_start_index, axis_end_index, angle_start_index, angle_end_index):
            # if more subjects than cores, each core may end up running more than one subject
            # cores will run their respective subjects and write their flattened results to the results buffer

            shared_axes = np.frombuffer(shared_axes, dtype=np.float32)
            shared_angles = np.frombuffer(shared_angles, dtype=np.float32)

            axis_floats = []
            angle_floats = []
            for subject in split_subjects_by_core[core_index]:
               subject.run()
               axis_floats.append(np.array([[frame[i].flatten()] for i in range(subject.num_axes) for frame in subject.axes]).flatten())
               angle_floats.append(np.array([[frame[i].flatten()] for i in range(subject.num_angles) for frame in subject.angles]).flatten())

            shared_axes[axis_start_index:axis_end_index] = np.asarray(axis_floats).flatten()
            shared_angles[angle_start_index:angle_end_index] = np.asarray(angle_floats).flatten()

        def joint_axis(self, key, start=None, end=None):
            # get specific joint axis of all subjects at optional frame index or slice

            if start is not None and end is None: # user is asking for just an index, not a slice
                return [subject.axes[start][key] for subject in self.subject_list]
            else:
                return [subject.axes[start:end][key] for subject in self.subject_list]

        def joint_angle(self, key, start=None, end=None):
            # get specific joint axis of all subjects at optional frame index or slice

            if start is not None and end is None: # user is asking for just an index, not a slice
                return [subject.angles[start][key] for subject in self.subject_list]
            else:
                return [subject.angles[start:end][key] for subject in self.subject_list]

