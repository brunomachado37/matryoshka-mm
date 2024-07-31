import numpy as np
import re

DATASET_STATS = {
    "droid": {
        "q01": np.array([-0.7776297926902771, -0.5803514122962952, -0.5795090794563293, -0.6464047729969025, -0.7041108310222626, -0.8895104378461838, 0.0]),
        "q99": np.array([0.7597932070493698, 0.5726242214441299, 0.7351000607013702, 0.6705610305070877, 0.6464948207139969, 0.8897542208433151, 1.0])
    },
    "fractal20220817": {
        "q01": np.array([-0.22453527510166169, -0.14820013284683228, -0.231589707583189, -0.3517994859814644, -0.4193011274933815, -0.43643461108207704, 0.0]), 
        "q99": np.array([0.17824687153100965, 0.14938379630446405, 0.21842354819178575, 0.5892666035890578, 0.35272657424211445, 0.44796681255102094, 1.0])
    }
}


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


def discretize_velocities(action, dataset_name):
    '''
        Discretizes the action space of the robot.

        Arguments:
            action (np.array): 6 commanded Cartesian Velocities of the robot, 1 commanded Gripper Position and 1 Is Terminal state (to indicate if it is the last step of the episode) at this time step

            First 3 Cartesian Velocities are coordinates :                      [-1, 1] [-1, 1] [-1, 1]
            Last 3 Cartesian Positions are angles (in radiants):                [-1, 1] [-1, 1] [-1, 1]
            Gripper Position is the oppening of the gripper (normalized):       [0, 1]
            Terminate state is a boolean (binary):                              [0 or 1]

        Returns:
            string: Discretized actions between 0 and 255 in the form <y0><y1><y2><y3><y4><y5><y6><y7>
    '''

    low =  DATASET_STATS[dataset_name]["q01"]
    high = DATASET_STATS[dataset_name]["q99"]

    x = action[:7]
    act = 2 * (x - low) / (high - low + 1e-8) - 1
    act = np.clip(act, a_min=-1.0, a_max=1.0)                                                           # Clipping will affect np.digitize, since no values will be outside the range

    bins = np.linspace(-1, 1, 257)                                                                      # This will results in actual 256 bins, since the last one will only be used when the value is exactly 1
    discretized_action = np.digitize(act, bins) - 1                                                     # Digitize will return values from [1, #bins] due to clipping, so we subtract 1 to get [0, #bins-1]
    discretized_action = np.clip(discretized_action, a_min=0, a_max=255)                                # Clip the last value to the last bin (only happens when it is exactly 1)

    discretized_action = np.append(discretized_action, action[7:], axis=0).astype(int)
    action_string = f"<{np.array2string(discretized_action, separator='><', prefix='<', suffix='>').replace('[', '').replace(']', '').replace(' ', '')}>"

    if not verify_action(action_string):
        raise Exception("Invalid Action String")

    return action_string


def undiscretize_velocities(action_string):
    '''
        Undiscretizes the action space of the robot.

        Arguments:
            action_string (string): Discretized actions between 0 and 255 in the form <y0><y1><y2><y3><y4><y5><y6><y7>
            
        Returns:
            action (np.array): 6 commanded Cartesian Velocities of the robot, 1 commanded Gripper Position and 1 Is Terminal state (to indicate if it is the last step of the episode) at this time step
    '''

    if not verify_action(action_string):
        raise Exception("Invalid Action String")

    low =  np.array([-0.7776297926902771, -0.5803514122962952, -0.5795090794563293, -0.6464047729969025, -0.7041108310222626, -0.8895104378461838, 0.0])               # DROID Q01
    high = np.array([0.7597932070493698, 0.5726242214441299, 0.7351000607013702, 0.6705610305070877, 0.6464948207139969, 0.8897542208433151, 1.0])                     # DROID Q99

    discretized_action = np.fromstring(action_string[1:-1], count=8, dtype=int, sep='><')

    bins = np.linspace(-1, 1, 257)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    normalized_action = bin_centers[discretized_action[:7]]
    unnormalized_action = 0.5 * (normalized_action + 1) * (high - low) + low

    return np.append(unnormalized_action, discretized_action[7:], axis=0)
