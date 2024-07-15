import numpy as np
import matplotlib.pyplot as plt
import pickle
from tudatpy import constants
from tudatpy.astro.time_conversion import DateTime, julian_day_to_calendar_date

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
ax.scatter(dv_list, tof_list, s=100, c="blue", marker='.', alpha=0.65)
plt.show()