import numpy as np

def check_robo_results_accuracy(axis_results, angle_results):
    # test structured axes and angles against existing csv output file

    axis_array_fields = ['Pelvis', 'Hip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head',
                         'Thorax', 'RClav', 'LClav', 'RHum', 'LHum', 'RRad', 'LRad', 'RHand', 'LHand']

    angle_array_fields = ['Pelvis', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot', 'Head', 'Thorax',
                          'Neck', 'Spine', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']

    # 'Pelvis' in the structured axis array corresponds to 'PELO', 'PELX', 'PELY', 'PELZ' in the csv (12 values)
    axis_slice_map = { key: slice( index*12, index*12+12, 1) for index, key in enumerate(axis_array_fields) }


    # 'Pelvis' in the structured angle array corresponds to 'X', 'Y', 'Z' in the csv (3 values)
    angle_slice_map = { key: slice( index*3, index*3+3, 1) for index, key in enumerate(angle_array_fields) }

    accurate = True

    actual_results = np.genfromtxt('SampleData/Sample_2/pycgm_results.csv', delimiter=',')
    for frame_idx, frame in enumerate(actual_results):
            frame = frame[58:]

            for key, slc in axis_slice_map.items():
                original_o = frame[slc.start    :slc.start + 3]
                original_x = frame[slc.start + 3:slc.start + 6] - original_o
                original_y = frame[slc.start + 6:slc.start + 9] - original_o
                original_z = frame[slc.start + 9:slc.stop]      - original_o
                refactored_o = axis_results[frame_idx][key][:3, 3]
                refactored_x = axis_results[frame_idx][key][0, :3]
                refactored_y = axis_results[frame_idx][key][1, :3]
                refactored_z = axis_results[frame_idx][key][2, :3]
                if not np.allclose(original_o, refactored_o):
                    accurate = False
                    error = abs((original_o - refactored_o) / original_o) * 100
                    print(f'{frame_idx}: {key}, origin ({error}%')
                if not np.allclose(original_x, refactored_x):
                    accurate = False
                    error = abs((original_x - refactored_x) / original_x) * 100
                    print(f'{frame_idx}: {key}, x-axis ({error}%')
                if not np.allclose(original_y, refactored_y):
                    accurate = False
                    error = abs((original_y - refactored_y) / original_y) * 100
                    print(f'{frame_idx}: {key}, y-axis ({error}%')
                if not np.allclose(original_z, refactored_z):
                    accurate = False
                    error = abs((original_z - refactored_z) / original_z) * 100
                    print(f'{frame_idx}: {key}, z-axis ({error}%')
    print('Axes accurate:', accurate)

    for frame_idx, frame in enumerate(actual_results):
            frame = frame[1:59]

            for key, slc in angle_slice_map.items():
                original_x = frame[slc][0]
                original_y = frame[slc][1]
                original_z = frame[slc][2]
                refactored_x = angle_results[frame_idx][key][0]
                refactored_y = angle_results[frame_idx][key][1]
                refactored_z = angle_results[frame_idx][key][2]
                if not np.allclose(original_x, refactored_x):
                    accurate = False
                    error = abs((original_x - refactored_x) / original_x) * 100
                    print(f'{frame_idx}: {key} angle, x ({error})%')
                if not np.allclose(original_y, refactored_y):
                    accurate = False
                    error = abs((original_y - refactored_y) / original_y) * 100
                    print(f'{frame_idx}: {key} angle, y ({error})%')
                if not np.allclose(original_z, refactored_z):
                    accurate = False
                    error = abs((original_z - refactored_z) / original_z) * 100
                    print(f'{frame_idx}: {key} angle, z ({error})%')
    print('Angles accurate:', accurate)
