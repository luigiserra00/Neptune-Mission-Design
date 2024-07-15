import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle
import tudatpy
from tudatpy import constants
from tudatpy.util import result2array
import tudatpy.util as util
from tudatpy.numerical_simulation import environment_setup
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy.astro.time_conversion import DateTime, julian_day_to_calendar_date

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

objective_list, individuals_list = pickle.load(open("E(DSM)EJN_montecarlo_results.pkl", "rb"))

departure_dates_list = list()
tof_list = list()
dv_list = list()
c3_list = list()

for i, individual in enumerate(individuals_list):
    
    individual[:4] = individual[:4] / constants.JULIAN_DAY
    departure_dates_list.append(julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000+individual[0]))
    tof_list.append(sum(individual[1:4])/365)
    dv_list.append(objective_list[i] - individual[5])
    c3_list.append(individual[5]**2)

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.grid()
#ax.scatter(dv_list, tof_list, s =5, color = "blue")
ax.set_xlabel( "Delta-V [km/s]")
ax.set_ylabel( "Time of flight [years]")
ax.set_title("Pareto front for Earth-Neptune transfer exploiting DSMs")
cs = ax.scatter(tof_list,
                dv_list,
                s=100,
                c="blue",
                marker='.',
                alpha=0.65)

# Add the Pareto from itself in green
tof_array = np.array(tof_list)
dv_array = np.array(dv_list)
optimum_mask = util.pareto_optimums(np.array([tof_array,dv_array]).T)
ax.step(
    sorted(tof_array[optimum_mask], reverse=True),
    sorted(dv_array[optimum_mask], reverse=False),
    color="blue",
    linewidth=2,
    alpha=0.75)
plt.show()

arr = np.array(dv_list)  # Assuming dv_list is the array you're interested in
target_value = 2319.33
closest_index = find_closest_index(arr, target_value)
vec = np.linspace(3000*constants.JULIAN_DAY, 6000*constants.JULIAN_DAY, 300)
print(vec[closest_index])
print('Departure time w.r.t J2000 [years]: ', individuals_list[closest_index][0]/365)
print("Departure date:")
print(julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000+individuals_list[closest_index][0]))
print('Earth-Earth time of flight [years]: ', individuals_list[closest_index][1]/365)
print("Fly-by date at Earth:")
print(julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000+sum(individuals_list[closest_index][0:2])))
print('Earth-Jupiter time of flight [years]: ', individuals_list[closest_index][2]/365)
print("Fly-by date at Jupiter:")
print(julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000+sum(individuals_list[closest_index][0:3])))
print('Earth-Earth time of flight [years]: ', individuals_list[closest_index][3]/365)
print("\nTotal time of flight [years]: ", sum(individuals_list[closest_index][1:4])/365)
print("DSM leg fraction: ", individuals_list[closest_index][4])
print("V_inf [m/s]: ", individuals_list[closest_index][5])
print("V_inf in-plane angle [rad]: ", individuals_list[closest_index][6])
print("V_inf out-of-plane angle [rad]: ", individuals_list[closest_index][7])
print("DeltaV [km/s]: ", dv_list[closest_index])

# Create simplified system of bodies
bodies = environment_setup.create_simplified_system_of_bodies()
transfer_body_order = ["Earth", 'Earth', 'Jupiter', 'Neptune']
central_body = "Sun"

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

print(individuals_list[closest_index])
print(tof_list[closest_index])
print(objective_list[closest_index])
node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(transfer_trajectory_object, individuals_list[closest_index])
transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

print(transfer_trajectory_object.delta_v_per_node)
print(transfer_trajectory_object.delta_v_per_leg)