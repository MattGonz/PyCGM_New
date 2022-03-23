from utils import pycgmIO
from custom_CGMs import harrington_hip_CGM, oxfordCGM, eyeballCGM
from pyCGM import pyCGM
from SM import SubjectManager
from test_robowalk import check_robo_results_accuracy

measurements = pycgmIO.loadVSK('SampleData/Sample_2/RoboSM.vsk')
marker_data = pycgmIO.loadData('SampleData/Sample_2/RoboWalk.c3d')
static_trial = pycgmIO.loadData('SampleData/Sample_2/RoboStatic.c3d')
oxford_marker_data = pycgmIO.loadData('SampleData/Oxford/OFM_Walk1.c3d')

def default_cgm_example():
    # running one subject, giving each core a block of frames to calculate

    subject = pyCGM(measurements, static_trial, marker_data)

    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

    subject.run()

    pr.disable()
    ss = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=ss).sort_stats(sortby)
    ps.print_stats()
    print(ss.getvalue())


    # check_robo_results_accuracy(subject.axes, subject.angles)
    subject_pelvis                 = subject.axes['Pelvis']        # subject's pelvis axis at each frame
    subject_pelvis_axis_frame_400  = subject.axes[400]['Pelvis']   # subject's pelvis axis at frame 400
    subject_pelvis_angle_frame_400 = subject.angles[400]['Pelvis'] # subject's pelvis angle at frame 400

def harrington_cgm_example():
    # running one subject with an overridden hip axis calculation

    harrington_subject = harrington_hip_CGM(measurements, static_trial, marker_data)
    harrington_subject.multi_run()
    harrington_hip_frame_400 = harrington_subject.axes[400]['Hip']

def oxford_foot_example():
    # running one subject with an overridden function that returns additional angles

    oxford_subject = oxfordCGM(measurements, static_trial, oxford_marker_data)
    oxford_subject.multi_run()
    subject_forefoot_left  = oxford_subject.angles['RForefoot']
    subject_hindfoot_right = oxford_subject.angles['RHindfoot']


def additional_function_example():
    # running one subject with an additional function that returns custom axes

    eyeball_subject = eyeballCGM(measurements, static_trial, marker_data)
    eyeball_subject.multi_run()
    subject_eyeball_left  = eyeball_subject.axes['LEyeball']
    subject_eyeball_right = eyeball_subject.axes['REyeball']


def multiple_similar_subjects_example():
    # run 9 default trials in parallel

    # create subjects
    subject_1 = pyCGM(measurements, static_trial, marker_data)
    subject_2 = pyCGM(measurements, static_trial, marker_data)
    subject_3 = pyCGM(measurements, static_trial, marker_data)

    # pass to subject manager and run
    subjects = SubjectManager([subject_1, subject_2, subject_3, 
                               subject_1, subject_2, subject_3, 
                               subject_1, subject_2, subject_3])
    subjects.multi_run()

    # retrieving results:
    all_pelvis             = subjects.joint_axis('Pelvis')         # pelvis axes of all subjects, all frames
    pelvis_frames_5_15     = subjects.joint_axis('Pelvis', 5, 15)  # pelvis axes of all subjects, frames 5-15
    left_ankle_frames_5_15 = subjects.joint_angle('LAnkle', 5, 15) # left ankle angle of all subjects, frames 5-15


def multiple_custom_subjects_example():
    # run multiple custom trials in parallel, retrieve specific results

    # create subjects
    default_subject    = pyCGM(measurements, static_trial, marker_data)
    harrington_subject = harrington_hip_CGM(measurements, static_trial, marker_data)
    oxford_subject     = oxfordCGM(measurements, static_trial, oxford_marker_data)
    eyeball_subject    = eyeballCGM(measurements, static_trial, marker_data)

    # pass to subject manager and run
    various_subjects = SubjectManager([default_subject, harrington_subject, oxford_subject, eyeball_subject])
    various_subjects.multi_run()

    # retrieve results from all subjects
    all_pelvis             = various_subjects.joint_axis('Pelvis')         # pelvis axes of all subjects, all frames
    pelvis_frames_5_15     = various_subjects.joint_axis('Pelvis', 5, 15)  # pelvis axes of all subjects, frames 5-15
    left_ankle_frames_5_15 = various_subjects.joint_angle('LAnkle', 5, 15) # left ankle angle of all subjects, frames 5-15

    # retrieve specific results from custom subjects
    oxford_forefoot_left = oxford_subject.angles['LForefoot'] # forefoot angle, all frames
    eyeball_left         = eyeball_subject.axes['LEyeball']   # left eyeball axis, all frames

default_cgm_example()
# harrington_cgm_example()
# oxford_foot_example()
# additional_function_example()
# multiple_similar_subjects_example()
# multiple_custom_subjects_example()
