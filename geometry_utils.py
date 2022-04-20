class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def do_intersect(p1, q1, p2, q2):
    '''
    The main function that returns true if the line segment 'p1q1' and 'p2q2' intersect.

    :param point_list: must be two line segments represented by [p1, q1, p2, q2] where the points are objects with .x and .y params
    :return: True if the segments intersect    
    '''

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
 
    # Special Cases
 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
 
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
 
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
 
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
 
    # If none of the cases
    return False
 
def too_close(p1, q1, p2, q2, d_min): 
    '''
    Tests if two line segments are closer than d_min in the x or y-directions
    '''
    #x-direction 
    if  abs(p1.x - p2.x) >= d_min or \
        abs(p1.x - q2.x) >= d_min or \
        abs(q1.x - p2.x) >= d_min or \
        abs(q1.x - q2.x) >= d_min:
        return False
    
    #y-direction
    if  abs(p1.y - p2.y) >= d_min or \
        abs(p1.y - q2.y) >= d_min or \
        abs(q1.y - p2.y) >= d_min or \
        abs(q1.y - q2.y) >= d_min:
        return False
    
    #z1-direction
    if  abs(p1.z1 - p2.z1) >= d_min or \
        abs(p1.z1 - q2.z1) >= d_min or \
        abs(q1.z1 - p2.z1) >= d_min or \
        abs(q1.z1 - q2.z1) >= d_min:
        return False

    #z2-direction
    if  abs(p1.z2 - p2.z2) >= d_min or \
        abs(p1.z2 - q2.z2) >= d_min or \
        abs(p2.z2 - q1.z2) >= d_min or \
        abs(p2.z2 - q2.z2) >= d_min:
        return  False
    
    return True

# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False
 
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
     
    # for details of below formula.
     
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
         
        # Clockwise orientation
        return 1
    elif (val < 0):
         
        # Counterclockwise orientation
        return 2
    else:
         
        # Collinear orientation
        return 0