%% Final Project
format long g

mu_sun = 132.712e9;

mu_venus = 0.32486e6;
r_min_venus = 200;
d_venus = 108.2e6;
vp_earth = sqrt(mu_sun / d_venus);
[rp_venus, ra_venus, e_venus, a_venus, T_venus, energy_venus, delta_venus] = Tisserand_func(vp_earth, d_venus, mu_venus, mu_sun, 3, 7, r_min_venus);

mu_earth = 0.39860e6;
r_min_earth = 200;
d_earth = 149.6e6;
vp_earth = sqrt(mu_sun / d_earth);
[rp_earth, ra_earth, e_earth, a_earth, T_earth, energy_earth, delta_earth] = Tisserand_func(vp_earth, d_earth, mu_earth, mu_sun, 3, 10, r_min_earth);

mu_mars = 0.042828e6;
r_min_mars = 200;
d_mars = 228.0e6;
vp_mars = sqrt(mu_sun / d_mars);
[rp_mars, ra_mars, e_mars, a_mars, T_mars, energy_mars, delta_mars] = Tisserand_func(vp_mars, d_mars, mu_mars, mu_sun, 1, 7, r_min_mars);

mu_jupiter = 126.687e6;
r_min_jupiter = 3 * 66854;
d_jupiter = 778.5e6;
vp_jupiter = sqrt(mu_sun / d_jupiter);
[rp_jupiter, ra_jupiter, e_jupiter, a_jupiter, T_jupiter, energy_jupiter, delta_jupiter] = Tisserand_func(vp_jupiter, d_jupiter, mu_jupiter, mu_sun, 3, 7, r_min_jupiter);

mu_saturn = 37.931e6;
r_min_saturn = 3 * 60268;
d_saturn = 1432.041e6;
vp_saturn = sqrt(mu_sun / d_saturn);
[rp_saturn, ra_saturn, e_saturn, a_saturn, T_saturn, energy_saturn, delta_saturn] = Tisserand_func(vp_saturn, d_saturn, mu_saturn, mu_sun, 1, 5, r_min_saturn);

mu_neptune = 6.8351e6;
r_min_neptune = 3 * 24764;
d_neptune = 4514.953e6;
vp_neptune = sqrt(mu_sun / d_saturn);
[rp_neptune, ra_neptune, e_neptune, a_neptune, T_neptune, energy_neptune, delta_neptune] = Tisserand_func(vp_neptune, d_neptune, mu_neptune, mu_sun, 1, 5, r_min_neptune);

figure(1)
%p1(1:5) = loglog(ra_venus, rp_venus, "color", "#77AC30", 'DisplayName', "Venus");
p1(6:13) = loglog(ra_earth, rp_earth, "color", "#0072BD","LineWidth",1);
hold on
p1(12) = loglog(ra_earth(:,7), rp_earth(:,7), "-k","LineWidth",2);
p1(14:18) = loglog(ra_jupiter, rp_jupiter, "color", "#D95319","LineWidth",1);
p1(17) = loglog(ra_jupiter(:,4), rp_jupiter(:,4), "-k" ,"LineWidth",2);
% loglog(ra_saturn, rp_saturn, "color", "#7E2F8E");
p1(18:22) = loglog(ra_neptune, rp_neptune, "color", "#7E2F8E","LineWidth",1);
p1(22) = loglog(ra_neptune(:,5), rp_neptune(:,5), "-k","LineWidth",2);
title('Tisserand Graph - V_i_n_f Curves')
xlabel('Apoapse Distance [km]')
ylabel('Periapse Distance [km]')
legend(p1([6,14,21]), { "Earth", "Jupiter", "Neptune"},"Fontsize",14)
set(gca,"Fontsize",14)
grid on

figure(2)
loglog(rp_venus, T_venus, "color", "#77AC30", 'DisplayName', "Venus");
hold on
loglog(rp_earth, T_earth, "color", "#0072BD");
loglog(rp_jupiter, T_jupiter, "color", "#D95319");
% loglog(ra_saturn, rp_saturn, "color", "#7E2F8E");
loglog(rp_neptune, T_neptune, "color", "#7E2F8E");
title('Tisserand Graph - V_i_n_f Curves')
xlabel('Periapse Distance [km]')
ylabel('Period year [km]')
% legend(p1([1,5,11,15]), {"Venus", "Earth", "Mars", "Jupiter"})
grid on

figure(3)
semilogx(rp_venus, energy_venus, "color", "#77AC30", 'DisplayName', "Venus")
semilogx(rp_earth, energy_earth, "color", "#0072BD")
semilogx(rp_jupiter, energy_jupiter, "color", "#0072BD")
semilogx(rp_saturn, energy_saturn, "color", "#0072BD")
title('Tisserand Graph - V_i_n_f Curves')
xlabel('Periapse Distance [km]')
ylabel('Energy [km^2/s^2]')
grid on


ra_earth_mars = 272846000;
rp_earth_mars = 146227000;
e_earth_mars = (1-rp_earth_mars/ra_earth_mars)/(1+rp_earth_mars/ra_earth_mars);
a_earth_mars = (rp_earth_mars+ra_earth_mars)/2;
v0_earth_mars = sqrt(mu_sun*(2/rp_earth_mars-1/a_earth_mars));
P_earth_mars = 2*pi*sqrt(a_earth_mars^3/mu_sun);
Y0 = [rp_earth_mars; 0; 0; 0; v0_earth_mars; 0];
[t,Y_earth_mars] = ode45(@(t,Y) x_propagator(t, Y, mu_sun),[0, P_earth_mars],Y0);

ra_mars_jupiter = 803180000;
rp_mars_jupiter = 219392000;
e_mars_jupiter = (1-rp_mars_jupiter/ra_mars_jupiter)/(1+rp_mars_jupiter/ra_mars_jupiter);
a_mars_jupiter = (rp_mars_jupiter+ra_mars_jupiter)/2;
v0_mars_jupiter = sqrt(mu_sun*(2/rp_mars_jupiter-1/a_mars_jupiter));
P_mars_jupiter = 2*pi*sqrt(a_mars_jupiter^3/mu_sun);
Y0 = [rp_mars_jupiter; 0; 0; 0; v0_mars_jupiter; 0];
[t,Y_mars_jupiter] = ode45(@(t,Y) x_propagator(t, Y, mu_sun),[0, P_mars_jupiter],Y0);


% figure(3)
% plot(Y_earth_mars(:,1), Y_earth_mars(:,2), "color", "#4DBEEE");
% hold on;
% plot(Y_mars_jupiter(:,1), Y_mars_jupiter(:,2), "color", "#7E2F8E");
% plot(d_earth*cos(0:0.01:2*pi), d_earth*sin(0:0.01:2*pi), "color", "#0072BD")
% plot(d_mars*cos(0:0.01:2*pi), d_mars*sin(0:0.01:2*pi), "color","#A2142F")
% plot(d_jupiter*cos(0:0.01:2*pi), d_jupiter*sin(0:0.01:2*pi), "color","#D95319")
% legend("Transfer Earth-Mars", "Transfer Mars-Jupiter", "Earth", "Mars", "Jupiter")



function dY = x_propagator(t, Y, mu)
    dY = zeros(6,1);
    dY(1:3) = Y(4:6);
    dY(4:6) = -mu/norm(Y(1:3))^3*Y(1:3);
end

function [rp, ra, e, a, T, energy, delta] = Tisserand_func(vp, d, mu_planet, mu_sun, v_inf_min, v_inf_max, r_min)
    v_inf_array = v_inf_min:1:v_inf_max;
    alpha_array = linspace(0, 180, 100);

    delta = 2 * asin(mu_planet./(mu_planet+r_min.*v_inf_array));

    [v_inf, alpha] = meshgrid(v_inf_array, alpha_array);
    v_inf = v_inf ./ vp;

    a = 1 ./ abs(1 - v_inf.^2 - 2 .* v_inf .* cosd(alpha));
    e = sqrt(1 - 1 ./ a .* ((3 - 1 ./ a - v_inf.^2) ./ (2)).^2);

    rp = a .* d .* (1 - e);
    ra = a .* d .* (1 + e);
    T = 2 .* pi .* sqrt((a .* d).^3 ./ mu_sun);
    energy = -mu_sun ./ (2 .* a .* d);
end
