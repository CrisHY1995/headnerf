import numpy as np
from scipy.spatial.transform import Rotation as R
from tool_funcs import Rmat2EulurAng, eulurangle2Rmat

class ArcBall:
    def __init__(self, NewWidth: float, NewHeight: float):
        self.StVec = np.zeros(3, 'f4')  # Saved click vector
        self.EnVec = np.zeros(3, 'f4')  # Saved drag vector
        self.AdjustWidth = 0.           # Mouse bounds width
        self.AdjustHeight = 0.          # Mouse bounds height
        self.setBounds(NewWidth, NewHeight)
        self.Epsilon = 1.0e-5


    def setBounds(self, NewWidth: float, NewHeight: float):
        assert((NewWidth > 1.0) and (NewHeight > 1.0))

        # Set adjustment factor for width/height
        self.AdjustWidth = 1.0 / ((NewWidth - 1.0) * 0.5)
        self.AdjustHeight = 1.0 / ((NewHeight - 1.0) * 0.5)
        

    def click(self, NewPt):  # Mouse down
        # Map the point to the sphere
        self._mapToSphere(NewPt, self.StVec)


    def drag(self, NewPt):  # Mouse drag, calculate rotation
        NewRot = np.zeros((4,), 'f4')

        # Map the point to the sphere
        self._mapToSphere(NewPt, self.EnVec)

        # Return the quaternion equivalent to the rotation
        # Compute the vector perpendicular to the begin and end vectors
        Perp = np.cross(self.StVec, self.EnVec)

        # Compute the length of the perpendicular vector
        if np.linalg.norm(Perp) > self.Epsilon:  # if its non-zero
            # We're ok, so return the perpendicular vector as the transform
            # after all
            NewRot[:3] = Perp[:3]
            # In the quaternion values, w is cosine (theta / 2), where theta
            # is rotation angle
            NewRot[3] = np.dot(self.StVec, self.EnVec)
        else:  # if its zero
            # The begin and end vectors coincide, so return an identity
            # transform
            pass
        return NewRot


    def _mapToSphere(self, NewPt, NewVec):
        # Copy paramter into temp point
        TempPt = NewPt.copy()

        # Adjust point coords and scale down to range of [-1 ... 1]
        TempPt[0] = (TempPt[0] * self.AdjustWidth) - 1.0
        TempPt[1] = 1.0 - (TempPt[1] * self.AdjustHeight)

        # Compute the square of the length of the vector to the point from the
        # center
        length2 = np.dot(TempPt, TempPt)

        # If the point is mapped outside of the sphere...
        # (length^2 > radius squared)
        if length2 > 1.0:
            # Compute a normalizing factor (radius / sqrt(length))
            norm = 1.0 / np.sqrt(length2)

            # Return the "normalized" vector, a point on the sphere
            NewVec[0] = TempPt[0] * norm
            NewVec[1] = TempPt[1] * norm
            NewVec[2] = 0.0
        else:    # Else it's on the inside
            # Return a vector to a point mapped inside the sphere
            # sqrt(radius squared - length^2)
            NewVec[0] = TempPt[0]
            NewVec[1] = TempPt[1]
            NewVec[2] = np.sqrt(1.0 - length2)
            
            
class ArcBallUtil(ArcBall):
    def __init__(self, NewWidth: float, NewHeight: float,  min_ang = -0.999, max_ang = 0.999):
        # self.Transform = np.identity(4, 'f4')
        self.LastRot = np.identity(3, 'f4')
        self.ThisRot = np.identity(3, 'f4')
        self.eulur_angle = np.zeros((3,), "f4")
        
        self.isDragging = False
        
        self.min_ang = min_ang
        self.max_ang = max_ang
        
        super().__init__(NewWidth, NewHeight)


    def resetRotation(self):
        self.isDragging = False
        self.LastRot = np.identity(3, 'f4')
        self.ThisRot = np.identity(3, 'f4')
        self.eulur_angle = np.zeros((3,), "f4")
        # self.Transform = self.Matrix4fSetRotationFromMatrix3f(self.Transform, self.ThisRot)

 
    def onClickLeftDown(self, cursor_x: float, cursor_y: float):
        # Set Last Static Rotation To Last Dynamic One
        self.LastRot = self.ThisRot.copy()
        # Prepare For Dragging
        self.isDragging = True
        mouse_pt = np.array([cursor_x, cursor_y], 'f4')
        # Update Start Vector And Prepare For Dragging
        self.click(mouse_pt)
        return


    def onDrag(self, cursor_x, cursor_y):
        """ Mouse cursor is moving
        """
        if self.isDragging:
            mouse_pt = np.array([cursor_x, cursor_y], 'f4')
            # Update End Vector And Get Rotation As Quaternion
            self.ThisQuat = self.drag(mouse_pt)
            # Convert Quaternion Into Matrix3fT
            self.ThisRot = self.Matrix3fSetRotationFromQuat4f(self.ThisQuat)
            
            temp_rot = np.matmul(self.LastRot, self.ThisRot)
            temp_eulur_angle = Rmat2EulurAng(temp_rot)
            
            if np.sum(temp_eulur_angle > self.max_ang) > 0 or np.sum([temp_eulur_angle < self.min_ang]) > 0:
            #     # self.click(mouse_pt)
                return
            
            # temp_eulur_angle[temp_eulur_angle > self.max_ang] = self.max_ang
            # temp_eulur_angle[temp_eulur_angle < self.min_ang] = self.min_ang
            
            self.ThisRot = eulurangle2Rmat(temp_eulur_angle)
            
            # self.ThisRot = np.matmul(self.LastRot, self.ThisRot)
            
            self.eulur_angle = temp_eulur_angle
            # Set Our Final Transform's Rotation From This One
            # self.Transform = self.Matrix4fSetRotationFromMatrix3f(self.Transform, self.ThisRot)
            # print(self.Transform)  # for debugging
        return
    
    
    def onClickLeftUp(self):
        self.isDragging = False
        # Set Last Static Rotation To Last Dynamic One
        self.LastRot = self.ThisRot.copy()
        

    def Matrix3fSetRotationFromQuat4f(self, q1):
        if np.sum(np.dot(q1, q1)) < self.Epsilon:
            return np.identity(3, 'f4')
        r = R.from_quat(q1)

        # transpose to make it identical to the C++ version
        return r.as_matrix()
    