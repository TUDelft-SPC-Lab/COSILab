# left_eye, right_eye, left_ear, right_ear -> missing in conflab
# left_foot, right_foot, head, neck -> missing in coco

# TODO: NEED TO UPDATE THE WEIGHTS AND SIGMAS, THEY WERE COPIED FROM COCO
dataset_info = dict(
    dataset_name='conflab',
    paper_info=dict(
        author='',
        title='Conflab',
        container='European conference on computer vision',
        year='2017',
        homepage='conflab',
    ),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(name='nose', id=1, color=[51, 153, 255], type='upper', swap=''),
        2:
        dict(name='neck', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
        dict(
            name='right_shoulder',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        4:
        dict(
            name='right_elbow',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        5:
        dict(
            name='right_wrist',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        6:
        dict(
            name='left_shoulder',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='left_wrist',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        9:
        dict(
            name='right_hip',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        10:
        dict(
            name='right_knee',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        12:
        dict(
            name='left_hip',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        
        14:
        dict(
            name='left_ankle',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        15:
        dict(
            name='right_foot',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='left_foot'),
        16:
        dict(
            name='left_foot',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='right_foot'),
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_foot', 'left_ankle'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('right_foot', 'right_ankle'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('nose', 'head'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('neck', 'head'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('left_shoulder', 'neck'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('right_shoulder', 'neck'), id=17, color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
