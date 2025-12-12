import math
import numpy as np

# Rotation Matrices ---------
def rotate_x(point, angle):  # w
    angle = math.radians(angle)
    x, y, z = point
    y_rot = y * math.cos(angle) - z * math.sin(angle)
    z_rot = y * math.sin(angle) + z * math.cos(angle)
    return [x, y_rot, z_rot]

def rotate_y(point, angle):  # p
    angle = math.radians(angle)
    x, y, z = point
    x_rot = x * math.cos(angle) + z * math.sin(angle)
    z_rot = -x * math.sin(angle) + z * math.cos(angle)
    return [x_rot, y, z_rot]

def rotate_z(point, angle):  # r
    angle = math.radians(angle)
    x, y, z = point
    x_rot = x * math.cos(angle) - y * math.sin(angle)
    y_rot = x * math.sin(angle) + y * math.cos(angle)
    return [x_rot, y_rot, z]

# Calculate estimated reference point
def get_robot_frame_coord(tcp_pos, laser_output, sensor_frame):

    laser_output = [0, -laser_output[1], -laser_output[2]]

    #testing laser coordinate switches
    #laser_output = [laser_output[2], -laser_output[1], 0]


    # First rotation stage - rotate predicted laser offset (x,y,z of guessed sensor frame)
    laser_dynamic_offset = rotate_x(sensor_frame[:3], tcp_pos[3])
    laser_dynamic_offset = rotate_y(laser_dynamic_offset, tcp_pos[4])
    laser_dynamic_offset = rotate_z(laser_dynamic_offset, tcp_pos[5])
    total_laser_offset = laser_dynamic_offset

    #print(total_laser_offset)

    #Calculate total angle of sensor by adding predicted laser angles to robot angles
    total_angles = [tcp_pos[i] + sensor_frame[i] for i in range(3,6)]
    P_ee = rotate_x(laser_output, total_angles[0])
    P_ee = rotate_y(P_ee, total_angles[1])
    P_ee = rotate_z(P_ee, total_angles[2])
    P_ee = [P_ee[i] + total_laser_offset[i] for i in range(3)]
    P_base = [P_ee[i] + tcp_pos[i] for i in range(3)]
    return P_base

# Calculate Error between predicted reference point and actual reference point
def euclidean_distance(point1, point2):
    # Distance between 3D point vectors
    return math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(3)))

# Error calculation
def optimisation_target(params, data):
    #params to be optimised (minimise error) are the sensor frame: (x,y,z,w,p,r)
    sensor_frame = params #x,y,z,w,p,r
    total_error = 0
    #call each row of data matrix
    for observed_data, tcp_pos, laser_output in data:
        #calculate coordinates
        computed_coords = get_robot_frame_coord(tcp_pos, laser_output, sensor_frame)
        #update total error
        total_error += euclidean_distance(computed_coords, observed_data)

    #get average error per scan
    avg_error = total_error / len(data)
    #print(total_error)
    return total_error, avg_error

# Gradient descent algorithm for minimisation (finding sensor frame) - set learning rate and iterations as required
def optimise_params_gd(initial_guess, data, learning_rate=0.01, iterations = 30000):
    # Initialise start point (guess) - base it roughly on sensor position on robot arm
    params = initial_guess
    for _ in range(iterations):
        gradients = [0] * len(params)
        # repeat algorithm for each sensor frame value to be found
        for i in range(len(params)):
            delta = 1e-5
            params[i] += delta
            error_plus, _ = optimisation_target(params, data)
            params[i] -= 2 * delta
            error_minus, _ = optimisation_target(params, data)
            params[i] += delta
            gradients[i] = (error_plus - error_minus) / (2 * delta)
        #update new best estimate for sensor frame values
        params = [params[i] - learning_rate * gradients[i] for i in range(len(params))]
    return params


'''
# brute force optimisation algorithm 
def optimise_params_bf(data):
    x_list = np.arange(30, 70, step=0.1)
    y_list = np.arange(-10, 10, step=0.1)
    z_list = np.arange(-60, -20, step=0.1)
    w_list = np.arange(-5, 5, step=0.1)
    p_list = np.arange(10, 30, step=0.1)
    r_list = np.arange(-5, 5, step=0.1)
    best_loss = np.inf
    best_params = []
    for x in x_list:
        for y in y_list:
            for z in z_list:
                for w in w_list:
                    for p in p_list:
                        for r in r_list:
                            params = [x, y, z, w, p, r]
                            loss = optimisation_target(params, data)
                            if loss < best_loss:
                                best_loss = loss
                                best_params = params
                                print(f"New Best Loss: {best_loss}")
    print(f"Best Sensor Frame: {best_params}; Best Loss: {best_loss}")

    return best_params
'''

#Sample Data:
ref1 = [1065.01, 371.82, -143.10]
data = [
    (ref1, [1057.73, 291.74, -150.99, -1.84, 7.66, 89.41], [0, 16.44, -0.87]),
    (ref1, [1084.52, 311.86, -127.61, -1.84, 7.66, 89.41], [0, -10.86, 28.72]),
    (ref1, [1085.52, 292.18, -150.99, -1.84, 7.66, 89.41], [0, -9.84, -1.76]),
    (ref1, [1056.58, 311.48, -127.65, -1.83, 7.66, 89.41], [0, 16.64, 29.46]),
]


#Initialise guess
initial_guess = [0, 0, 0, 0, 0, 0]
# Run gradient descent algorithm
optimised_params = optimise_params_gd(initial_guess, data)
# Get the total error for the calculated sensor frame values
total_error, avg_error = optimisation_target(optimised_params, data)
print(f"Sensor Frame: x,y,z - {optimised_params[:3]} ; w,p,r - {optimised_params[3:]}")
print(f"Total Error (mm): {total_error}")
print(f"Average Error (mm): {avg_error}")
