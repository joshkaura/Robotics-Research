import math

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
    # First rotation stage - rotate predicted laser offset (x,y,z of guessed sensor frame)
    laser_dynamic_offset = rotate_x(sensor_frame[:3], tcp_pos[3])
    laser_dynamic_offset = rotate_y(laser_dynamic_offset, tcp_pos[4])
    laser_dynamic_offset = rotate_z(laser_dynamic_offset, tcp_pos[5])
    total_laser_offset = laser_dynamic_offset

    #Calculate total angle of sensor by adding predicted laser angles to robot angles
    total_angles = [tcp_pos[i] + sensor_frame[i] for i in range(3,6)]
    P_ee = rotate_x(laser_output, total_angles[0])
    P_ee = rotate_y(P_ee, total_angles[1])
    P_ee = rotate_z(P_ee, total_angles[2])
    P_ee = [P_ee[i] + total_laser_offset[i] for i in range(3)]
    P_base = [P_ee[i] + tcp_pos[i] for i in range(3)]
    return P_base

def main():
    sensor_frame = [51.084, -4.132, -43.053, -1.191, 24.789, -3.557]
    tcp_pos = [1473.04, -401.55, 254.65, 0, 0, 0]
    sensor_output = [0, 9.53, 8.84]

    robot_coord = get_robot_frame_coord(tcp_pos, sensor_output, sensor_frame)

    print(robot_coord)

    return robot_coord


if __name__ == '__main__':
    main()