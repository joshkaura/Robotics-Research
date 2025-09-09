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
    #laser_output = [0, -laser_output[1], -laser_output[2]]

    #testing laser coordinate switches
    laser_output = [laser_output[2], -laser_output[1], 0]


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
    #print(total_error)
    return total_error

# Gradient descent algorithm for minimisation - set learning rate and iterations as required
def optimise_params_gd(initial_guess, data, learning_rate=0.01, iterations = 30000):
    # Initialise start point (guess) - base it roughly on sensor position on robot arm
    params = initial_guess
    for _ in range(iterations):
        gradients = [0] * len(params)
        # repeat algorithm for each sensor frame value to be found
        for i in range(len(params)):
            delta = 1e-5
            params[i] += delta
            error_plus = optimisation_target(params, data)
            params[i] -= 2 * delta
            error_minus = optimisation_target(params, data)
            params[i] += delta
            gradients[i] = (error_plus - error_minus) / (2 * delta)
        #update new best estimate for sensor frame values
        params = [params[i] - learning_rate * gradients[i] for i in range(len(params))]
    return params


'''
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


'''
#Actual reference point
ref1 = [1319.715, -36.597, 117.90]


#Sample data
data = [
    (ref1, [1264.111, -38.164, 162.019], [0, 0, 0], [0, -5.47, 5.41]),
    (ref1, [1266.098, -50.641, 161.951], [0, 0, 0], [0, 5.78, 6.89]),
    (ref1, [1271.982, -38.164, 177.187], [0, 0, 0], [0, -4.27, -9.99]),
    (ref1, [1273.998, -52.084, 177.187], [0, 0, 0], [0, 8.44, -9.03]),
    (ref1, [1268.548, -43.712, 168.077], [0, 0, 0], [0, -0.25, 0.03]),
    (ref1, [1269.730, -53.504, 168.126], [0, 0, 0], [0, 9.21, 0.50]),
]
'''

'''
#Sample CW data:
ref1 = [1065.01, 371.82, -143.10]
data = [
    (ref1, [1057.73, 291.74, -150.99, -1.84, 7.66, 89.41], [0, 16.44, -0.87]),
    (ref1, [1084.52, 311.86, -127.61, -1.84, 7.66, 89.41], [0, -10.86, 28.72]),
    (ref1, [1085.52, 292.18, -150.99, -1.84, 7.66, 89.41], [0, -9.84, -1.76]),
    (ref1, [1056.58, 311.48, -127.65, -1.83, 7.66, 89.41], [0, 16.64, 29.46]),
]
'''

'''
#Sample OSL Data
ref1 = [1519.65, -415.29, 203.17]
data = [
    (ref1, [1465.48, -403.29, 238.42, 0, 0, 0], [0, 7.64, -7.42]),
    (ref1, [1464.33, -420.57, 238.42, 0, 0, 0], [0, -9.17, -7.72]),
    (ref1, [1471.93, -420.57, 254.74, 0, 0, 0], [0, -9.29, 8.76]),
    (ref1, [1473.04, -401.55, 254.65, 0, 0, 0], [0, 9.50, 8.85]),
]
'''


#Sample RR F Data
ref1 = [777.38, 396.39, 950.95]
data = [
    (ref1, [785.82, 386.15, 953.79, -87.62, 0, 0], [0, 0.31, 2.86]),
    (ref1, [785.84, 385.81, 960.89, -87.62, 0, 0], [0, -7.34, 2.59]),
    (ref1, [765.51, 396.34, 960.87, -87.62, 0, 0], [0, -6.98, 25.49]),
    (ref1, [765.53, 396.39, 953.79, -87.62, 0, 0], [0, 0.03, 25.63]),
]


#Initialise guess
initial_guess = [-50, 5, 0, 0, -25, 0]
# Run gradient descent algorithm
optimised_params = optimise_params_gd(initial_guess, data)
# Get the total error for the calculated sensor frame values
total_error = optimisation_target(optimised_params, data)
print(f"Sensor Frame: x,y,z - {optimised_params[:3]} ; w,p,r - {optimised_params[3:]}")
print(f"Total Error: {total_error}")

#brute force
#best_params = optimise_params_bf(data)
