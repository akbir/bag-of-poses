#feature extraction
import numpy as np

def get_joints(limbs, joints, limbSeq):
    angles = []
    for joint in joints:
        start = tuple(limbSeq[joint[0]])
        end = tuple(limbSeq[joint[1]])
        if start in limbs.keys() and end in limbs.keys():
            angle= angle_between(limbs[start],limbs[end])
            angles.append(angle)
        else:
            angles.append(0) #if limb is obstructed by other object
    neck_nose = limbSeq[12]
    angles.append(angle_between(neck_nose,[0,1]))
    #final angle in vector tells us frame relative to the pict
    return(angles)

def unit_vector(vector):
    if np.linalg.norm(vector) ==0:
        answer =  vector
    else:
        answer = vector / np.linalg.norm(vector)
    return(answer)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    dot = np.dot(v1_u, v2_u)
    det = v1_u[0]*v2_u[1]-v1_u[1]*v2_u[0]
    angle = np.arctan2(det,dot)
    if np.isnan(angle):
        angle = 0
    return angle
