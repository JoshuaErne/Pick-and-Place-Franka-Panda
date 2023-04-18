import sys
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time 

import rospy
import roslib
import threading

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# The library you implemented over the course of this semester!
from lib.calculateFK import FK
from lib.calcJacobian import FK
from lib.solveIK import IK

            
def rotate_about(axis: str, theta: float):
    """
    Get homogenous transformation matrix that represents an
    euler rotation about an axis (x/y/z) by theta degrees.
    Input:
        axis:   axis of rotation as a str
        theta:  angle of rotation about the axis
    Output:
        H:      4x4 homogenous transformation matrix with 0 translation
    """
    try:
        H = R.from_euler(axis, theta, degrees=True).as_matrix()
    except:
        H = R.from_euler(axis, theta, degrees=True).as_dcm()
    H = np.append(H, np.zeros(shape=(1,3)), axis=0)
    H = np.append(H, np.array([[0],[0],[0],[1]]), axis=1)
    return H


Ry_180 = rotate_about('y', 180)
Ry_90  = rotate_about('y', 90)
Ry_45  = rotate_about('y', 45)
Rz_90  = rotate_about('z', 90)
Rz_180  = rotate_about('z', 180)


class Challenger:
    def __init__(self, arm, detector, team):
        self.team = team
        self.arm = arm
        self.detector = detector
        self.ik = IK(max_steps=1000)
        self.n_stacked = 0
        self.safe_move_thread = None
        self.queued_gripper_action = None


    def get_detections(self):
        """
        Get list of tag detections from vision system. 
        Seperate out tag 0 for reference transformation, 
        and save tag 5s for last. 
        Output:
            detections:   list of (tagID, pose) for each detected non-zero tag
            pose_tag0:    pose of tag 0 relative to camera frame
        """
        detections = [
            (int(x[0][3:]), x[1]) for x in self.detector.get_detections() if int(x[0][3:]) < 7
        ]

        detections_others = []
        detections_5 = []
        for tag, pose in detections:
            if tag == 0:
                pose_tag0 = pose
            elif tag == 5:
                detections_5.append((tag, pose)) 
            else:
                detections_others.append((tag, pose))
        print("Tags Detected:", [x[0] for x in detections])
        return detections_others + detections_5, pose_tag0


    def safe_move(self, q, gripper_action=None):
        # check if existing thread is running
        if (self.safe_move_thread is not None):
            # wait for thread to terminate
            self.safe_move_thread.join()

            # check if robot arm has stopped moving
            while not np.all(np.abs(self.arm.get_velocities()) < 0.1):
                continue

            # perform gripper action when arm stops moving
            if self.queued_gripper_action == 'close':
                self.arm.exec_gripper_cmd(0.03, 50)
            elif self.queued_gripper_action == 'open':
                self.arm.exec_gripper_cmd(0.2, 50)
        
        # start thread for safe_move_to_position
        self.safe_move_thread = threading.Thread(target=self.arm.safe_move_to_position, args=(q,))
        self.queued_gripper_action = gripper_action
        self.safe_move_thread.start()


    def flip_block(self, H_tag_to_world):
        print("================= flip_block =====================")
        # make sure gripper is open
        self.arm.exec_gripper_cmd(0.2, 50)

        # hard-coded corner positions
        if self.team == 'blue':
            H_corner = np.array([[-0.012, -1.0,    0.0, 0.47], 
                                [-1.0,    0.012, -0.0, 0.27], 
                                [0.0,    -0.0,   -1.0, 0.25], 
                                [0.0,     0.0,    0.0, 1.0]])
        elif self.team == 'red':
            H_corner = np.array([[-1.0,  0.009, -0.0,  0.469],
                                 [0.009, 1.0,   -0.0, -0.253],
                                 [0.0,  -0.0,   -1.0,  0.25],
                                 [0.0,   0.0,    0.0,  1.0]])

        # move above target block
        _, q_above_original = self.move_above(H_tag_to_world)

        # secure block with tag 5 facing up (i.e white side down)
        q, success, roll_out = self.ik.inverse(target=H_tag_to_world, seed=q_above_original)
        print("IK Solver [success: %r] [iterations: %i] [q: %s]" % (success, len(roll_out), np.round(q, 4).tolist()))
        self.safe_move(q, 'close')

        # move end-effector with the block straight up
        self.safe_move(q_above_original, 'idle')

        # move block to corner position and release
        # H_corner[2][-1] += 0.005
        q, success, roll_out = self.ik.inverse(target=H_corner, seed=q)
        print("IK Solver [success: %r] [iterations: %i] [q: %s]" % (success, len(roll_out), np.round(q, 4).tolist()))
        self.safe_move(q, 'open')

        # move gripper above block position
        _, q_above_corner = self.move_above(H_corner, q_seed=q)

        # determine orientation to acquire the block from the side
        if self.team == 'blue':
            H_5 = H_corner @ rotate_about('y', 90) 
        elif self.team == 'red':
            H_5 = H_corner @ rotate_about('x', -90) @ rotate_about('z', -90)
        
        # move above the target pose
        _, q_above_corner_side = self.move_above(H_5, q_seed=q_above_corner)

        # move to acquire block form the side
        H_5[2][-1] -= 0.025
        q, success, roll_out = self.ik.inverse(target=H_5, seed=q_above_corner_side)
        print("IK Solver [success: %r] [iterations: %i] [q: %s]" % (success, len(roll_out), np.round(q, 4).tolist()))
        self.safe_move(q, 'close')

        # move above the target pose
        self.safe_move(q_above_corner_side, 'idle')

        # re-orient for goal delivery
        self.move_above(H_corner @ rotate_about('z', 180), q_seed=q_above_corner)


    def move_to_box(self, H_tag_to_world, q_seed):
        """
        Open gripper, move end-effector to the world pose of the tag,
        and then close gripper to secure the target block.
        Input:
            H_tag_to_world:  World pose of the detected tag
        """
        print("================= move_to_box =====================")
        # make sure gripper is open 
        self.arm.exec_gripper_cmd(0.2, 50)

        # move to target pose using IK solver
        q, success, roll_out = self.ik.inverse(target=H_tag_to_world, seed=q_seed)
        print("IK Solver [success: %r] [iterations: %i] [q: %s]" % (success, len(roll_out), np.round(q, 4).tolist()))
        self.safe_move(q, 'close')


    def move_above(self, H_tag_to_world, q_seed=None, distance=5*0.05):
        """
        Move end-effector straight up in the pure z-direction.
        Used after gripper has secured a block and we do not want
        the arm to hit other score-able blocks in the vicinity.
        Input:
            H_tag_to_world:  Pose of the block the gripper just secured
            q_seed:          Seed for IK solver (default is current arm pose)
            distance:        Distance above the block pose to move the end-effector
                             (default is height of 3 blocks + a small offset)
        Output:
            H_up_to_world:   Pose directly above the target block
            q_above:         Joint angles needed to achieve this pose
        """
        print("================= move_above =====================")
        # compute pose that is directly above given block pose
        H_up_to_world = np.copy(H_tag_to_world)
        H_up_to_world[2][-1] += distance

        # set seed to current arm pose if none is given
        if q_seed is None:
            q_seed = self.arm.neutral_position()

        # move to target pose using IK solver
        q_above, success, roll_out = self.ik.inverse(target=H_up_to_world, seed=q_seed)
        print("IK Solver [success: %r] [iterations: %i] [q_above: %s]" % (success, len(roll_out), np.round(q_above, 4).tolist()))
        self.safe_move(q_above, 'idle')
        return H_up_to_world, q_above


    def move_to_goal_solver(self, tag, n_stacked, offset=False):
        """
        Move arm with block secured to the goal platform for stacking.
        The center point of the stack is pre-determined/hard-coded to
        be around the center of the goal platform. The height that we
        drop the block on the stack will be dependent on the height of
        the existing stack, computed from n_stacked.
        Input:
            tag:        Integer ID of the detected tag
            n_stacked:  Number of blocks in current stack
        """
        print("================= move_to_goal_solver =====================")
        # hard-coded pose for target goal platform
        H_goal = np.array([[1, 0, 0, 0.55],
                           [0, 1, 0, 0.17],
                           [0, 0, 1, 0.25],
                           [0, 0, 0, 1.00]])

        if offset:
            H_goal[1][-1] -= 0.01

        # flip sign for y if we are on blue side
        if self.team == 'blue':
            H_goal[1][-1] = -1.0 * H_goal[1][-1]

        # determine end-effector orientation for goal placement
        if tag == 6:
            H_goal = H_goal @ Ry_180
        elif tag in [1,2,3,4,5]:
            if self.team == 'red':
                if n_stacked == 0:
                    H_goal = H_goal @ rotate_about('z', 90) @ Ry_90 @ Ry_180
                elif n_stacked >=3: 
                    H_goal = H_goal @ rotate_about('z', -90) @ Ry_90 @ rotate_about('x', 180) 
                else:
                    H_goal = H_goal @ rotate_about('z', -90) @ Ry_90 @ Ry_180
            else:
                if tag == 5:
                    H_goal = H_goal @ rotate_about('y', 180) @ rotate_about('z', -90) @ Ry_90 @ Ry_180
                else:
                    H_goal = H_goal @ rotate_about('z', -90) @ Ry_90 @ Ry_180

        # determine seed for IK solver
        if self.team == 'blue':
            if tag == 6:
                q_up_seed = np.array([-2.91623208e-01, 1.47476583e-01, -9.39437771e-03, -1.22381023e+00, 1.41328374e-03, 1.37127638e+00, -2.65683213e+00])
            else:
                q_up_seed = np.array([1.5641, -1.2495, -1.5337, -1.5578, 0.3212, 1.5635, -0.7461])
        else:
            if tag == 6:
                q_up_seed = np.array([0.1184, 0.0507, 0.1872, -1.6151, -0.0090, 1.6650, -2.0498])
            else:
                if n_stacked == 0:
                    q_up_seed = np.array([0.7111, 0.46, -0.0405, -1.4365, -1.3045, 0.9429, 2.7622])
                else:
                    q_up_seed = np.array([-0.0759, 0.042, -0.0089, -2.0086, 1.532, 1.4955, -1.2667])

        if n_stacked >= 4:
            q_up_seed = self.arm.neutral_position()

        # move right above target
        if n_stacked > 3:
            distance_above = (5 + n_stacked - 3) * 0.05
        else:
            distance_above = 5 * 0.05

        _, q_above = self.move_above(H_goal, q_seed=q_up_seed, distance=distance_above)

        # determine goal offsets
        if (n_stacked == 0) and (tag in [1,2,3,4]):
            H_goal[2][-1] += 0.005
        elif n_stacked > 0:
            H_goal[2][-1] += n_stacked * 0.05 - 0.005
            if tag in [1,2,3,4]:
                H_goal[0][-1] -= 0.00

        q, success, roll_out = self.ik.inverse(target=H_goal, seed=q_above)
        print("IK Solver [success: %r] [iterations: %i] [q_goal: %s]" % (success, len(roll_out), np.round(q, 4).tolist()))

        # clipping to acount for joint limit tolerance on physical robot
        if self.n_stacked == 0 and self.team == 'red':
            q[-1] = np.clip(q[-1], -2.77, 2.77)

        # move to target pose
        self.safe_move(q, 'open')

        # move back above the stack
        self.safe_move(q_above, 'idle')

        return q_above


    def solve_static(self, flip=False):
        print("================= solve_static =====================")
        # Get block detections from camera
        detections, pose_tag0 = self.get_detections()
        
        # Get cam to world transformation using tag 0
        H_cam_to_world = np.linalg.inv(pose_tag0)
        H_cam_to_world[0][-1] -= 0.5

        # Solver for all four static blocks
        for tag, pose in detections:
            
            print("Working on block with tag %s facing up!" % (tag))

            # Get pose of tag in world coordinates
            H_tag_to_world = (H_cam_to_world @ pose) @ Ry_180
            print(tag, np.round(H_tag_to_world, 3).tolist())

            # Add z-offset so gripper grips box center
            H_tag_to_world[2][-1] -= 0.020

            # Treat tag 5 as 6 if we do not perform flipping
            if not flip and tag == 5:
                tag = 6

            # Perform block flipping to expose tag 6 (white) 
            # on the side if tag 5 is facing up
            if tag == 5:
                # flip block to free up white side
                self.flip_block(H_tag_to_world)

                # Stack block on goal platform
                self.move_to_goal_solver(tag, self.n_stacked)

                # Update number of blocks we moved thus far
                self.n_stacked += 1

                continue

            # if tag 6 is on the side, rotate end-effector about z-axis by 90 degrees
            # so the gripper will NOT hold the block by touching the white side
            if tag in [1, 2, 3, 4]:
                H_tag_to_world = H_tag_to_world @ Rz_90

            # pre-determined seed to block acquisition 
            if self.team == 'blue':
                up_guess = np.array([0.27196885,  0.38545116, -0.09641445, -1.81657193,  0.04478432,  2.19992482,  1.26397763])
            elif self.team == 'red':
                up_guess = np.array([0.10205506, -0.20467078, -0.27345035, -2.1013167, -0.05810551,  1.90353383, -2.31504299])
            
            # Move directly above the block
            _, q_above = self.move_above(H_tag_to_world, q_seed=up_guess)

            # Move straight down to block and grab it
            self.move_to_box(H_tag_to_world, q_above)

            # Move straight back up with the block
            self.safe_move(q_above)

            # Stack block on goal platform
            self.move_to_goal_solver(tag, self.n_stacked)

            # Update number of blocks we moved thus far
            self.n_stacked += 1

            if self.team == 'red' and self.n_stacked == 3:
                return


    def solve_dynamic(self):
        print("================= solve_dynamic =====================")
        # Reset arm to neutral position
        # self.arm.safe_move_to_position(self.arm.neutral_position())

        # Move right next to fishing position
        if self.team == 'red':
            q = np.array([-0.9950232, -0.68437996, 2.05260015, -2.09794916, 1.44217399, 1.0948742, 1.26544277])
        elif self.team == 'blue':
            q = np.array([ 0.09944714, -1.2001083,  -1.84091989, -2.03610745,  0.24376155,  2.06714711, -1.14179131])
        self.safe_move(q, 'open')

        # Move to fishing position
        if self.team == 'red':
            q = np.array([-1.37040617, -1.04756311, 2.3305908, -1.28996346, 1.75509485, 1.11841457, 1.67379912])
        elif self.team == 'blue':
            q = np.array([0.51568679, -1.50415501, -1.7691142,  -1.36260843, -0.01920058,  1.88511792, -0.91914018])
        self.safe_move(q)

        # Wait 6 seconds for block to come
        time.sleep(6)

        # Repeatly close and open gripper until it grabs a block
        n_tries = 0
        while True:
            # close gripper
            self.arm.exec_gripper_cmd(0.03, 50)

            # check gripper position
            gripper_pos = np.array(self.arm.get_gripper_state()["position"])
            print("Gripper positions are:", gripper_pos)

            # terminate loop if block secured
            if np.all(gripper_pos > 0.01):
                print('Block secured!')
                break
             # open gripper back up and wait 2s if gripper closed all the way
            else:
                self.arm.exec_gripper_cmd(0.5, 50)
                time.sleep(5)

            # increment try count
            n_tries += 1
            
            # stop dynamic if gripper failed twice
            if n_tries == 3 and self.team == 'blue':
                return False

        # Deliver dynamic block to goal
        if self.team == 'blue':
            q_above = self.move_to_goal_solver(tag=3, n_stacked=self.n_stacked, offset=True)
        else:
            q_above = self.move_to_goal_solver(tag=3, n_stacked=self.n_stacked)

        # Move to the right/left instead of back to neutral
        if self.team == 'blue':
            q_above[0] -= 0.5
        else:
            q_above = self.arm.neutral_position()
            
        self.safe_move(q_above)
            
        # Update stacked block count
        self.n_stacked += 1
        return True


    def solve_blue(self):
        # Keep trying for dynamic until we fail to grab a block in 2 tries
        blocks_dynamic = 0
        success_dynamic = True
        while (success_dynamic) and (blocks_dynamic < 3):
            blocks_dynamic += 1
            success_dynamic = self.solve_dynamic()

        # Go for static blocks
        self.solve_static()


    def solve_red(self):
        # Go for static blocks first
        self.solve_static()

        # Use remaining time for dynamic
        success_dynamic = True
        while success_dynamic:
            success_dynamic = self.solve_dynamic()


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    arm.safe_move_to_position(arm.neutral_position()) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM **")
    else:
        print("** RED TEAM **")
    print("****************")

    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    tstart = time_in_seconds()

    if team == 'blue':
        Challenger(arm, detector, team).solve_blue()
    elif team == 'red':
        Challenger(arm, detector, team).solve_red()

    tfinal = time_in_seconds()

    print(f"Took {tfinal-tstart} seconds to complete!")
