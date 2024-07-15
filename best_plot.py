import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

from typing import List, Tuple

import tudatpy
from tudatpy.util import result2array
from tudatpy.numerical_simulation import environment_setup
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy import constants
from tudatpy.astro.frame_conversion import inertial_to_body_fixed_rotation_matrix
from tudatpy.interface.spice import get_body_cartesian_state_at_epoch
from tudatpy.interface import spice_interface

bodies_to_create = ["Sun", "Earth", "Jupiter","Neptune"]
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"

# Create the bodies
spice_interface.load_standard_kernels()
spice_interface.load_kernel("c:/Users/TeamRed/.tudat/resource/spice_kernels/de440.bsp")
bodies_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)
bodies = environment_setup.create_system_of_bodies(bodies_settings)
                                                   
### Plot the transfer
"""
Finally, the position history throughout the transfer can be retrieved from the transfer trajectory object and plotted.
"""

def convert_trajectory_parameters (transfer_trajectory_object: tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory,
                                   trajectory_parameters: List[float]
                                   ) -> Tuple[ List[float], List[List[float]], List[List[float]] ]:

    # Declare lists of transfer parameters
    node_times = list()
    leg_free_parameters = list()
    node_free_parameters = list()

    # Extract from trajectory parameters the lists with each type of parameters
    departure_time = trajectory_parameters[0]
    times_of_flight_per_leg = trajectory_parameters[1:4]
    dsm_leg_fraction = trajectory_parameters[4]
    v_inf = trajectory_parameters[5]
    v_inf_in_plane_angle = trajectory_parameters[6]
    v_inf_out_of_plane_angle = trajectory_parameters[7]

    # Get node times
    # Node time for the intial node: departure time
    node_times.append(departure_time)
    # None times for other nodes: node time of the previous node plus time of flight
    accumulated_time = departure_time
    for i in range(0, transfer_trajectory_object.number_of_nodes - 1):
        accumulated_time += times_of_flight_per_leg[i]
        node_times.append(accumulated_time)

    # Get leg free parameters and node free parameters: one empty list per leg
    leg_free_parameters.append(np.array([dsm_leg_fraction]))
    for i in range(transfer_trajectory_object.number_of_legs-1):
        leg_free_parameters.append( [ ] )
    # One empty array for each node
    node_free_parameters.append(np.array([v_inf, v_inf_in_plane_angle, v_inf_out_of_plane_angle]))
    for i in range(transfer_trajectory_object.number_of_nodes-1):
        node_free_parameters.append( [ ] )

    return node_times, leg_free_parameters, node_free_parameters


def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = state_history[:i, 0]
    y = np.rad2deg(declination[:i])
    line.set_data(x, y)
    return line,

# Define the central body
central_body = "Sun"

# Define order of bodies (nodes)
transfer_body_order = ["Earth", 'Earth', 'Jupiter', 'Neptune']

# Define departure orbit
departure_semi_major_axis = np.inf
departure_eccentricity = 0

# Define insertion orbit
arrival_altitude_of_periapsis = 1e6 # 1000 km
arrival_eccentricity = 0.95
body_radius = 24764e3
arrival_radius_of_periapsis = body_radius + arrival_altitude_of_periapsis
arrival_semi_major_axis = arrival_radius_of_periapsis/arrival_eccentricity

# Manually create the legs settings

# First create an empty list and then append to that the settings of each transfer leg
transfer_leg_settings = [transfer_trajectory.dsm_velocity_based_leg()]
for i in range(len(transfer_body_order) - 2):
    transfer_leg_settings.append( transfer_trajectory.unpowered_leg() )

# Manually create the nodes settings

# First create an empty list and then append to that the settings of each transfer node
transfer_node_settings = []

# Initial node: departure_node
transfer_node_settings.append( transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity) )

# Intermediate nodes: swingby_node
transfer_node_settings.append( transfer_trajectory.swingby_node(6678000.0) )
transfer_node_settings.append( transfer_trajectory.swingby_node(600e6) )

# Final node: capture_node
transfer_node_settings.append( transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity) )

# Create the transfer calculation object
transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body)


minimuim_dv, best_individual = pickle.load(open("best.pkl", "rb"))

# Reevaluate the transfer trajectory using the champion design variables
node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(transfer_trajectory_object, best_individual)
transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

# Extract the state history
state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

print(transfer_trajectory_object.delta_v_per_node)
print(transfer_trajectory_object.delta_v_per_leg)

# Plot the state history
fig = plt.figure(num = 1, figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au)
ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, color='blue', label='Earth departure')
ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, color='green', label='Earth fly-by')
ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, color='green', label = "Jupiter fly-by")
ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, color='brown', label='Neptune arrival')
ax.scatter([0], [0], color='orange', label='Sun')
ax.set_xlabel('x wrt Sun [AU]')
ax.set_ylabel('y wrt Sun [AU]')
ax.set_aspect('equal')
ax.grid()
ax.legend(bbox_to_anchor=[1, 1])
plt.show()

# Plot the z-coordinate
fig = plt.figure(num = 2, figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(state_history[:,0], state_history[:, 3] / au)
ax.scatter(node_times[0], fly_by_states[0, 2] / au, color='blue', label='Earth departure')
ax.scatter(node_times[1], fly_by_states[1, 2] / au, color='green', label='Earth fly-by')
ax.scatter(node_times[2], fly_by_states[2, 2] / au, color='green', label='Jupiter fly-by')
ax.scatter(node_times[3], fly_by_states[3, 2] / au, color='brown', label='Neptune arrival')
ax.set_xlabel('Time [days]')
ax.set_ylabel('z wrt Sun [AU]')
plt.show()

#Plot velocities
fig = plt.figure(num = 3, figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(state_history[:,0], np.linalg.norm(state_history[:, 4:],axis = 1), color='blue', label='$V_x$[m/s]')
ax.scatter(node_times[0], fly_by_states[0, 2] / au, color='blue', label='Earth departure')
ax.scatter(node_times[1], fly_by_states[1, 2] / au, color='green', label='Earth fly-by')
ax.scatter(node_times[2], fly_by_states[2, 2] / au, color='green', label='Jupiter fly-by')
ax.scatter(node_times[3], fly_by_states[3, 2] / au, color='brown', label='Neptune arrival')
ax.set_xlabel('Time [days]')
ax.set_ylabel('Velocity [m/s]')
ax.legend()
plt.show()

pole_right_ascension = (299.3)
pole_declination = (42.950)
prime_meridian = (0) # irrelevant for declination
rotation_matrix = inertial_to_body_fixed_rotation_matrix(pole_declination, pole_right_ascension, prime_meridian)

# Plot the state history in the body-fixed frame
declination = np.zeros((len(state_history[:,0])))
for i, state in enumerate(state_history):
    state_hat = state[1:4]/np.linalg.norm(state[1:4])
    body_fixed_state_history= np.dot(rotation_matrix, state_hat)   
    declination[i] = np.arcsin(body_fixed_state_history[2]) 

fig = plt.figure(num = 4, figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(state_history[:,0], np.rad2deg(declination))
ax.set_xlabel('Time [days]')
ax.set_ylabel('Declination [deg]')
plt.show()

fig, ax = plt.subplots(num = 5, figsize=(8, 5))
fig.patch.set_facecolor('black')  # Set the figure background to black
ax.set_facecolor('black')  # Set the axes background to black

line, = ax.plot([], [], lw=2, color='white')  # Initialize an empty line for the trajectory with a contrasting color

# Setting up the plot limits, labels, and changing label colors to white
ax.set_xlim([-8,8])
ax.set_ylim([-8,8])
ax.set_xlabel('x wrt Sun [AU]', color='white')
ax.set_ylabel('y wrt Sun [AU]', color='white')
ax.tick_params(axis='x', colors='white')  # Change x-axis tick colors to white
ax.tick_params(axis='y', colors='white')  # Change y-axis tick colors to white
ax.scatter([0], [0], color='yellow', label='Sun')  # Sun position with a bright color
ax.set_aspect('equal')

planet_names = ['Earth', 'Jupiter_Barycenter', 'Neptune_Barycenter']
planet_colors = ['blue', 'orange', 'cyan']  # Example colors for Earth, Jupiter, and Neptune

# Initialize scatter plots for planets
planet_scatters = [ax.scatter([], [], color=color, label=name) for name, color in zip(planet_names, planet_colors)]

def update_planet_positions(ephemeris_time):
    positions = []
    for name in planet_names:
        position = get_body_cartesian_state_at_epoch(name, 'Sun', 'ECLIPJ2000', "None", ephemeris_time)
        positions.append(position[:2])  # Assuming the function returns [x, y, z] and we only need [x, y]
    return positions

def init():
    line.set_data([], [])
    for scatter in planet_scatters:
        scatter.set_offsets(np.empty((0, 2)))  # Initialize with empty data
    return [line, *planet_scatters]

def animate(i):
    line.set_data(state_history[:i, 1] / au, state_history[:i, 2] / au)

    if i == 0:
        # Set a default zoom level for the first frame
        current_max = 8
    else:
        current_max_x = np.max(np.abs(state_history[:i, 1] / au))
        current_max_y = np.max(np.abs(state_history[:i, 2] / au))
        current_max = max(current_max_x, current_max_y) + 5

    if current_max < 12:
        ax.set_xlim([-8,8])
        ax.set_ylim([-8,8])
    else:
        ax.set_xlim([-current_max, current_max])
        ax.set_ylim([-current_max, current_max])

    # Update planet positions based on the current ephemeris time
    current_ephemeris_time = state_history[i, 0]
    planet_positions = update_planet_positions(current_ephemeris_time)
    for scatter, position in zip(planet_scatters, planet_positions):
        scatter.set_offsets([position])  # Update each planet's position

    # Zoom and limits logic remains unchanged
    # ...

    return [line, *planet_scatters]

# Creating the animation
ani = FuncAnimation(fig, animate, frames=len(state_history[:, 0]), init_func=init, blit=True, interval=20)

#ax.legend(bbox_to_anchor=[1, 1])
plt.show()