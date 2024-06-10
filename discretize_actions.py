import numpy as np
import re


def verify_action(action_string):
    # Check if the action string follows the expected pattern
    pattern = r'^<.*?><.*?><.*?><.*?><.*?><.*?><.*?><.*?>$'

    if not re.match(pattern, action_string):
        return False

    # Check if numbers are within limits (0 and 255)
    numbers = re.findall(r'<(.*?)>', action_string)

    if len(numbers) != 8:
        return False

    for num in numbers:
        if not num.isdigit() or not (0 <= int(num) <= 255):
            return False

    return True


def discretize(action):
    '''
        Discretizes the action space of the robot.

        Arguments:
            action (np.array): 6 commanded Cartesian Positions of the robot, 1 commanded Gripper Position and 1 Is Terminal state (to indicate if it is the last step of the episode) at this time step

            First 3 Cartesian Positions are coordinates (in meters):            [0, .855] [-.855, 855] [-.365, 1.188]
            Last 3 Cartesian Positions are angles (in radiants):                [-pi, pi] [-pi/2, pi/2] [-pi, pi]
            Gripper Position is the oppening of the gripper (normalized):       [0, 1]
            Terminate state is a boolean (binary):                              [0 or 1]

        Returns:
            string: Discretized actions between 0 and 255 in the form <y0><y1><y2><y3><y4><y5><y6><y7>
    '''

    discretized_actions = np.zeros(8, dtype=int)

    # Discretize Cartesian Positions    
    discretized_actions[0] = int((max(0.0000, min(action[0], 0.855)) + 0.000) * 127.5 /  (0.855 / 2))
    discretized_actions[1] = int((max(-0.855, min(action[1], 0.855)) + 0.855) * 127.5 /   0.855)
    discretized_actions[2] = int((max(-0.365, min(action[2], 1.188)) + 0.365) * 127.5 / ((1.188 + 0.365) / 2))

    # Discretize Angles
    discretized_actions[3] = int((max(-np.pi,   min(action[3], np.pi))   + np.pi)   * 127.5 /  np.pi)
    discretized_actions[4] = int((max(-np.pi/2, min(action[4], np.pi/2)) + np.pi/2) * 127.5 / (np.pi / 2))
    discretized_actions[5] = int((max(-np.pi,   min(action[5], np.pi))   + np.pi)   * 127.5 /  np.pi)

    # Discretize Gripper Position
    discretized_actions[6] = int(max(0, min(action[6], 1)) * 255)

    # Discretize Is Terminal
    discretized_actions[7] = int(max(0, min(action[7], 1)))

    action_string = f"<{np.array2string(discretized_actions, separator='><', prefix='<', suffix='>').replace('[', '').replace(']', '').replace(' ', '')}>"

    if not verify_action(action_string):
        raise Exception("Invalid Action String")


    return action_string
