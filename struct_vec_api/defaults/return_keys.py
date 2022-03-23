def axes():
    """
    map function names to the axes they return
    """
    return {'calc_axis_pelvis': ['Pelvis'],
            'calc_joint_center_hip': ['RHipJC', 'LHipJC'],
            'calc_axis_hip': ['Hip'],
            'calc_axis_knee': ['RKnee', 'LKnee'],
            'calc_axis_ankle': ['RAnkle', 'LAnkle'],
            'calc_axis_foot': ['RFoot', 'LFoot'],
            'calc_axis_head': ['Head'],
            'calc_axis_thorax': ['Thorax'],
            'calc_marker_wand': ['RWand', 'LWand'],
            'calc_joint_center_shoulder': ['RClavJC', 'LClavJC'],
            'calc_axis_shoulder': ['RClav', 'LClav'],
            'calc_axis_elbow': ['RHum', 'LHum', 'RWristJC', 'LWristJC'],
            'calc_axis_wrist': ['RRad', 'LRad'],
            'calc_axis_hand': ['RHand', 'LHand']}

def angles():
    """
    map function names to the angles they return
    """
    return {'pelvis_angle': ['Pelvis'],
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
