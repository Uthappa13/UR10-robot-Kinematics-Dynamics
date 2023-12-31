from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
init_printing()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Generalized transformation matrix for the robot

# Defining the constants
theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1, theta2, theta3, theta4, theta5, theta6')

# Transformation matrix between frames
T_1 = Matrix([[cos(theta1), 0, -sin(theta1), 0], [sin(theta1), 0, cos(theta1), 0], [0, -1, 0, 128], [0, 0, 0, 1]])
T_2 = Matrix([[-sin(theta2), cos(theta2), 0, 612.7 * sin(theta2)], [cos(theta2), sin(theta2), 0, -612.7 * cos(theta2)], [0, 0, -1, 0], [0, 0, 0, 1]])
T_3 = Matrix([[cos(theta3), sin(theta3), 0, -571.6 * cos(theta3)], [sin(theta3), -cos(theta3), 0, -571.6 * sin(theta3)], [0, 0, -1, 0], [0, 0, 0, 1]])
T_4 = Matrix([[sin(theta4), 0, -cos(theta4), 0], [-cos(theta4), 0, -sin(theta4), 0], [0, 1, 0, 163.9], [0, 0, 0, 1]])
T_5 = Matrix([[cos(theta5), 0, -sin(theta5), 0], [sin(theta5), 0, cos(theta5), 0], [0, -1, 0, 115.7], [0, 0, 0, 1]])
T_6 = Matrix([[cos(theta6), -sin(theta6), 0, 0], [sin(theta6), cos(theta6), 0, 0], [0, 0, 1, 192.2], [0, 0, 0, 1]])

# Transformation matrix with respect to the base frame
T_01 = T_1 
T_02 = T_1 * T_2
T_03 = T_1 * T_2 * T_3
T_04 = T_1 * T_2 * T_3 * T_4
T_05 = T_1 * T_2 * T_3 * T_4 * T_5
T_06 = T_1 * T_2 * T_3 * T_4 * T_5 * T_6


# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define a matrix for the Jacobian - J_w

J_w = Matrix([[0,T_01[0,2],T_02[0,2],T_03[0,2],T_04[0,2],T_05[0,2]],[0,T_01[1,2],T_02[1,2],T_03[1,2],T_04[1,2],T_05[1,2]],[1,T_01[2,2],T_02[2,2],T_03[2,2],T_04[2,2],T_05[2,2]]])
# pprint(J_w)

# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Define a matrix for Jacobian - J_v

P_x = T_06[0,3]
P_y = T_06[1,3]
P_z = T_06[2,3]

J_v = Matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
J_v[0,0] = diff(P_x, theta1)
J_v[1,0] = diff(P_y, theta1)
J_v[2,0] = diff(P_z, theta1)
J_v[0,1] = diff(P_x, theta2)
J_v[1,1] = diff(P_y, theta2)
J_v[2,1] = diff(P_z, theta2)
J_v[0,2] = diff(P_x, theta3)
J_v[1,2] = diff(P_y, theta3)
J_v[2,2] = diff(P_z, theta3)
J_v[0,3] = diff(P_x, theta4)
J_v[1,3] = diff(P_y, theta4)
J_v[2,3] = diff(P_z, theta4)
J_v[0,4] = diff(P_x, theta5)
J_v[1,4] = diff(P_y, theta5)
J_v[2,4] = diff(P_z, theta5)
J_v[0,5] = diff(P_x, theta6)
J_v[1,5] = diff(P_y, theta6)
J_v[2,5] = diff(P_z, theta6)
# pprint(J_v)

# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Define the complete jacobian matrix

J = Matrix.vstack(J_v, J_w)

# # Print the jacobian on the terminal
# pprint(J)

# # Substitute the values of theta for the home position
theta_values = {theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0, theta6: 0}
J_sp = J.subs(theta_values)
# pprint(J_sp)
J_value = np.matrix(J_sp).astype(np.float64)
J_inv = np.linalg.pinv(J_value)


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Calculations for the end effector trajectory

# Define the values for variables
r = 100                         # radius of the circle
omega = (2 * np.pi)/20          # angular velocity
Time = 20                       # total time to draw the circle
t = 0
dt = 0.03
q = np.zeros((6,1))
X_values = []
Y_values = []
Z_values = []


while (t < Time):
# Derivative of parametric equation of a circle to give the velocity components
    x_dot = r * np.cos(omega*t) * omega
    z_dot = -(r * np.sin(omega*t) * omega)

# Define the matrix for the end effector velocity
    X_dot = np.matrix([[x_dot], [0], [z_dot], [0], [0], [0]])

# Calculate the joint velocities and perform numerical integration to find the joint angles
    q_dot = J_inv*X_dot
    q = q + q_dot*dt

# Substitute the angles in the jacobian matrix
    [angle1, angle2, angle3, angle4, angle5, angle6] = [q[i].item() for i in range(6)]
    J_sp = J.subs({theta1: angle1, theta2: angle2, theta3: angle3, theta4: angle4, theta5: angle5, theta6: angle6})
    J_value = np.matrix(J_sp).astype(np.float64)
    J_inv = np.linalg.pinv(J_value)

# Substitute the angles in the final transformation matrix to obtain the end effector coordinates
    T_values = T_06.subs({theta1: angle1, theta2: angle2, theta3: angle3, theta4: angle4, theta5: angle5, theta6: angle6})
    X_values.append(T_values[0,3])
    Y_values.append(T_values[1,3])
    Z_values.append(T_values[2,3])

    t = t + dt


# # 2D plot of the trajectory
# plt.title("Trajectory of the robot")
# plt.xlabel("X : X coordinates")
# plt.ylabel("Z : Z coordinates")
# plt.grid(True)
# plt.plot(X_values, Z_values)
# plt.show()

# 3D plot of the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_values, Y_values, Z_values)
ax.set_xlabel('X coordinates')
ax.set_ylabel('Y coordinates')
ax.set_zlabel('Z coordinates')
ax.set_title('Trajectory of the robot')

ax.set_xlabel([-125,125])
ax.set_ylabel([200,400])
ax.set_zlabel([1450,1150])
plt.show()




