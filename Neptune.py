# Load standard modules
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

plt.rcParams.update({'font.size': 18})

# Load spice kernels
spice.load_standard_kernels()
spice.load_kernel("/Users/luigiserra/.tudat/resource/spice_kernels/nep097.bsp")
spice.load_kernel("/Users/luigiserra/.tudat/resource/spice_kernels/pck00011.tpc")
spice.load_kernel("/Users/luigiserra/.tudat/resource/spice_kernels/gm_de440.tpc.txt")

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2069, 1, 1).epoch()
simulation_end_epoch   = DateTime(2073, 1, 1).epoch()

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun","Neptune","Triton"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Neptune"
global_frame_orientation = "ECLIPJ2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

r_Neptune = 24622e3
r_p = 1000e3
r_a = 370605e3

bodies_to_propagate = bodies_to_create

# Create bodies in simulation.
body_settings = environment_setup.get_default_body_settings(bodies_to_create)
body_system = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle objects.
bodies.create_empty_body("Delfi-C3")
bodies.get("Delfi-C3").mass = 2000

# Define bodies that are propagated
bodies_to_propagate = ["Delfi-C3"]

# Define central bodies of propagation
central_bodies = ["Neptune"]

# Define accelerations acting on Delfi-C3 by Sun and Earth.
accelerations_settings_delfi_c3 = dict(
    Sun=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Neptune=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Triton =[
        propagation_setup.acceleration.point_mass_gravity()
    
    ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {"Delfi-C3": accelerations_settings_delfi_c3}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

a = (r_p + r_a) / 2
e = (r_a - r_p) / (r_a + r_p)
inc = np.deg2rad(157)
raan = np.deg2rad(0)
arg_per = np.deg2rad(0)
true_anomaly = np.deg2rad(0)

mu_Neptune = bodies.get_body("Neptune").gravitational_parameter

initial_state = element_conversion.keplerian_to_cartesian([a, e, inc, raan, arg_per, true_anomaly], mu_Neptune)

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                10,
                propagation_setup.integrator.CoefficientSets.rkdp_87,
                np.finfo(float).eps,
                np.inf,
                1e-10,
                1e-10)

# Define list of dependent variables to save
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Delfi-C3"),
    propagation_setup.dependent_variable.keplerian_state("Delfi-C3", "Neptune"),
    propagation_setup.dependent_variable.latitude("Delfi-C3", "Neptune"),
    propagation_setup.dependent_variable.longitude("Delfi-C3", "Neptune"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Neptune"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3","Triton"
    )
]

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)

# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state and depedent variable history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.dependent_variable_history
dep_vars_array = result2array(dep_vars)

#Make 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3])
plt.show()

time = states_array[:,0]
time_hours = time/3600

# Plot Kepler elements as a function of time
kepler_elements = dep_vars_array[:,4:10]
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Semi-major Axis
semi_major_axis = kepler_elements[:,0] / 1e3
ax1.plot(time_hours, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:,1]
ax2.plot(time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:,2])
ax3.plot(time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_elements[:,3])
ax4.plot(time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:,4])
ax5.plot(time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:,5])
ax6.scatter(time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_sun = dep_vars_array[:,12]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')

# Point Mass Gravity Acceleration Moon
acceleration_norm_pm_moon = dep_vars_array[:,13]
plt.plot(time_hours, acceleration_norm_pm_moon, label='PM Neptune')

# Point Mass Gravity Acceleration Triton
acceleration_norm_pm_triton = dep_vars_array[:,14]
plt.plot(time_hours, acceleration_norm_pm_triton, label='PM Triton')

plt.xlim([min(time_hours), max(time_hours)])
plt.xlabel('Time [hr]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend(bbox_to_anchor=(1.005, 1))
plt.suptitle("Accelerations norms on Delfi-C3, distinguished by type and origin, over the course of propagation.")
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()

r_ring = 42000e3

#plot with radius of periapsis and apoapsis
plt.figure(figsize=(9, 5))
plt.plot(time_hours, kepler_elements[:,0] * (1 - kepler_elements[:,1]), label='Periapsis')
plt.plot(time_hours, kepler_elements[:,0] * (1 + kepler_elements[:,1]), label='Apoapsis')
plt.plot([time_hours[0],time_hours[-1]], [r_ring, r_ring], label='Innermost ring', linestyle='--', color='black')
plt.xlabel('Time [hr]')
plt.ylabel('Distance from Triton [m]')
plt.legend()
plt.show()

# Plot ground track for a period of 3 hours
latitude = dep_vars_array[:,10]
longitude = dep_vars_array[:,11]
hours = 3
subset = int(len(time_hours) / 24 * hours)
latitude = np.rad2deg(latitude[0: subset])
longitude = np.rad2deg(longitude[0: subset])
plt.figure(figsize=(9, 5))
plt.title("4 years ground track of Nostromo over Neptune")
plt.scatter(longitude, latitude, s=1)
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xlim([min(longitude), max(longitude)])
plt.yticks(np.arange(-90, 91, step=45))
plt.grid()
plt.tight_layout()
plt.show()