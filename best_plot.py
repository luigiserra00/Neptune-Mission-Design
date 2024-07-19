import numpy as np
import matplotlib.pyplot as plt
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
from tudatpy.astro.time_conversion import DateTime, julian_day_to_calendar_date
from matplotlib.animation import FuncAnimation, FFMpegWriter

bodies = environment_setup.create_simplified_system_of_bodies()

plt.rcParams.update({'font.size': 14})
                                                   
### Plot the transfer
"""
Finally, the position history throughout the transfer can be retrieved from the transfer trajectory object and plotted.
"""

def find_closest_index(arr, target_value):
    # Calculate the absolute difference between each array element and the target value
    differences = np.abs(arr - target_value)
    # Find the index of the smallest difference
    closest_index = np.argmin(differences)
    return closest_index

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


minimum_dv, best_individual = pickle.load(open("best.pkl", "rb"))

# Reevaluate the transfer trajectory using the champion design variables
node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(transfer_trajectory_object, best_individual)
transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

# Extract the state history
state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

dsm_fraction = best_individual[4]
dsm_time = node_times[0] + dsm_fraction * (node_times[1] - node_times[0])
dsm_idx = find_closest_index(state_history[:,0], dsm_time)

# Plot the state history
fig = plt.figure(num = 1, figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(state_history[:, 2] / au, state_history[:, 1] / au)
ax.scatter(fly_by_states[0, 1] / au, fly_by_states[0, 0] / au, color='blue', label='Earth departure')
ax.scatter(fly_by_states[1, 1] / au, fly_by_states[1, 0] / au, color='green', label='Earth fly-by')
ax.scatter(fly_by_states[2, 1] / au, fly_by_states[2, 0] / au, color='green', label = "Jupiter fly-by")
ax.scatter(fly_by_states[3, 1] / au, fly_by_states[3, 0] / au, color='brown', label='Neptune arrival')
ax.scatter(state_history[dsm_idx, 2] / au, state_history[dsm_idx,1] / au, color='red', label='DSM', marker="x")
ax.scatter([0], [0], color='orange', label='Sun')
ax.set_ylabel('x wrt Sun [AU]')
ax.set_xlabel('y wrt Sun [AU]')
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

time = state_history[:,0]
orbit_Earth = np.zeros((len(time), 6))
orbit_Jupiter = np.zeros((len(time), 6))
orbit_Neptune = np.zeros((len(time), 6))
orbit_Saturn = np.zeros((len(time), 6))

for i, t in enumerate(time):
    orbit_Earth[i,:] = bodies.get_body("Earth").state_in_base_frame_from_ephemeris(t)/au
    orbit_Jupiter[i,:] = bodies.get_body("Jupiter").state_in_base_frame_from_ephemeris(t)/au
    orbit_Neptune[i,:] = bodies.get_body("Neptune").state_in_base_frame_from_ephemeris(t)/au
    orbit_Saturn[i,:] = bodies.get_body("Saturn").state_in_base_frame_from_ephemeris(t)/au

fig, ax = plt.subplots(num = 5, figsize=(8, 5))
fig.patch.set_facecolor('black')  # Set the figure background to black
ax.set_facecolor('black')  # Set the axes background to black

line, = ax.plot([], [], lw=1, color='magenta')  # Initialize an empty line for the trajectory with a contrasting color
line_Earth, = ax.plot([], [], lw=1, color='cyan')  # Initialize an empty line for the trajectory with a contrasting color
line_Jupiter, = ax.plot([], [], lw=1, color='lime')  # Initialize an empty line for the trajectory with a contrasting color
line_Neptune, = ax.plot([], [], lw=1, color='coral')  # Initialize an empty line for the trajectory with a contrasting color
line_Saturn, = ax.plot([], [], lw=1, color='red')  # Initialize an empty line for the trajectory with a contrasting color
spacecraft_scatter = ax.scatter([], [], s=7, color="magenta", label = "Nostromo")
Earth_scatter = ax.scatter([], [], s=7, color="cyan", label = "Earth")
Jupiter_scatter = ax.scatter([], [], s=7, color="lime", label = "Jupiter")
Neptune_scatter = ax.scatter([], [], s=7, color="coral", label = "Neptune")
Saturn_scatter = ax.scatter([], [], s=7, color="red", label = "Saturn")
date_text = ax.text(0.55, 0.01, '', transform=ax.transAxes, color="white", fontsize=10)  # Posiziona il testo nell'angolo in alto a sinistra
vel_text = ax.text(0.15, 0.01, '', transform=ax.transAxes, color="white", fontsize=10)  # Posiziona il testo nell'angolo in alto a sinistra

# Setting up the plot limits, labels, and changing label colors to white
ax.set_xlim([-8,8])
ax.set_ylim([-8,8])
#ax.set_xlabel('x wrt Sun [AU]', color='white')
#ax.set_ylabel('y wrt Sun [AU]', color='white')
#ax.tick_params(axis='x', colors='white')  # Change x-axis tick colors to white
#ax.tick_params(axis='y', colors='white')  # Change y-axis tick colors to white
ax.scatter([0], [0], color='yellow', label='Sun')  # Sun position with a bright color
ax.set_aspect('equal')
ax.text(0.4, 1.1, "Nostromo", transform=ax.transAxes, color = "white", fontsize=10)

planet_names = ['Earth', 'Jupiter', "Saturn" 'Neptune']
planet_colors = ['cyan', 'lime', "red", 'coral']  # Example colors for Earth, Jupiter, and Neptune

def init():
    line.set_data([], [])
    line_Earth.set_data([], [])
    line_Jupiter.set_data([], [])
    line_Neptune.set_data([], [])
    spacecraft_scatter.set_offsets(np.empty((0, 2)))  # Initialize with empty data
    Earth_scatter.set_offsets(np.empty((0, 2)))  # Initialize with empty data
    Jupiter_scatter.set_offsets(np.empty((0, 2)))  # Initialize with empty data
    Neptune_scatter.set_offsets(np.empty((0, 2)))  # Initialize with empty data
    Saturn_scatter.set_offsets(np.empty((0, 2)))  # Initialize with empty data
    date_text.set_text('')
    vel_text.set_text('')
    return [line, line_Earth, line_Jupiter, line_Neptune, spacecraft_scatter, Earth_scatter, Jupiter_scatter, Neptune_scatter, Saturn_scatter, date_text, vel_text]

def animate(i):
    line.set_data(state_history[:i, 1] / au, state_history[:i, 2] / au)
    line_Earth.set_data(orbit_Earth[:i, 0], orbit_Earth[:i, 1])
    line_Jupiter.set_data(orbit_Jupiter[:i, 0], orbit_Jupiter[:i, 1])
    line_Neptune.set_data(orbit_Neptune[:i, 0], orbit_Neptune[:i, 1])
    line_Saturn.set_data(orbit_Saturn[:i, 0], orbit_Saturn[:i, 1])

    date = julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000 + state_history[i, 0]/constants.JULIAN_DAY)
    date_string = str(date.year) + '-' + str(date.month) + '-' + str(date.day) 
    date_text.set_text(date_string) 
    vel_text.set_text(f'Velocity: {np.linalg.norm(state_history[i, 4:])/1000:.2f} km/s')
    if i == 0:
        # Set a default zoom level for the first frame
        current_max = 8
    else:
        current_max_x = np.max(np.abs(state_history[:i, 1] / au))
        current_max_y = np.max(np.abs(state_history[:i, 2] / au))
        current_max = max(current_max_x, current_max_y) + 5

    ax.set_xlim([-current_max, current_max])
    ax.set_ylim([-current_max, current_max])

    # Update planet positions based on the current ephemeris time
    current_ephemeris_time = state_history[i, 0]
    spacecraft_scatter.set_offsets([state_history[i, 1] / au, state_history[i, 2] / au])
    Earth_scatter.set_offsets([bodies.get_body("Earth").state_in_base_frame_from_ephemeris(current_ephemeris_time)[:2]/au])
    Jupiter_scatter.set_offsets([bodies.get_body("Jupiter").state_in_base_frame_from_ephemeris(current_ephemeris_time)[:2]/au])
    Neptune_scatter.set_offsets([bodies.get_body("Neptune").state_in_base_frame_from_ephemeris(current_ephemeris_time)[:2]/au])
    Saturn_scatter.set_offsets([bodies.get_body("Saturn").state_in_base_frame_from_ephemeris(current_ephemeris_time)[:2]/au])

    return [line, line_Earth, line_Jupiter, line_Neptune, spacecraft_scatter, Earth_scatter, Jupiter_scatter, Neptune_scatter, Saturn_scatter, date_text, vel_text]
# Creating the animation
ani = FuncAnimation(fig, animate, frames=len(state_history[:, 0]), init_func=init, blit=True, interval=8)
writer = FFMpegWriter(fps=24)
legend = ax.legend(facecolor='black', edgecolor='none',bbox_to_anchor=(1.5,0.01),ncol = 5)
plt.setp(legend.get_texts(), color='white')
#ani.save('nostromo.mp4', writer=writer)
plt.show()