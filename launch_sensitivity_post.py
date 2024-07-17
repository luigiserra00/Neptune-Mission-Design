import numpy as np
import matplotlib.pyplot as plt
import pickle
from tudatpy.astro.time_conversion import DateTime, julian_day_to_calendar_date
from tudatpy import constants

plt.rcParams.update({'font.size': 18})

objective_list, individuals_list = pickle.load(open("launch_sensitivity.pkl", "rb"))

departure_dates_list = list()
tof_list = list()
dv_list = list()
c3_list = list()
dsm_frac = list()

departure_date_lb = DateTime(2053,  2,  25).epoch()
departure_date_ub = DateTime(2058,  1,  1).epoch()

dates = np.linspace(departure_date_lb, departure_date_ub, 300)

for i, individual in enumerate(individuals_list):
    
    individual[:3] = individual[:3] / constants.JULIAN_DAY
    departure_dates_list.append(julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000+dates[i]/constants.JULIAN_DAY))
    dsm_frac.append(individual[3])
    tof_list.append(sum(individual[:3])/365)
    dv_list.append(objective_list[i] - individual[4])

fig = plt.figure(num = 1, figsize=(4,3))
ax = fig.add_subplot(111)
ax.grid()
#ax.scatter(dv_list, tof_list, s =5, color = "blue")
ax.set_xlabel( "Delta-V [km/s]")
ax.set_ylabel( "Time of flight [years]")
ax.set_title("Pareto front for Earth-Neptune transfer exploiting DSMs")
cs = ax.scatter(tof_list,
                dv_list,
                s=45,
                c="blue",
                marker='.',
                alpha=1)


fig = plt.figure(num = 2, figsize=(4,3))
ax = fig.add_subplot(111)
ax.grid(visible=True)
ax.scatter(departure_dates_list, dv_list, color = "blue", s = 45)
ax.set_xlabel("Launch date")
ax.set_ylabel("$\Delta$V [m/s]")

fig = plt.figure(num = 3, figsize=(4,3))
ax = fig.add_subplot(111)
ax.scatter(departure_dates_list, dsm_frac)
plt.show()