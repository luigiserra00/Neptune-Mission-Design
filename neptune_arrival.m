mu_Neptune = 6.836525e6;
r_Neptune  = 24622;

r_Triton = 1353.4;
v_Triton = 4.39; 
r_p_t = r_Triton + 200;
r_a_t = 6000 + r_Triton;
e_t = (r_a_t-r_p_t)/(r_a_t+r_p_t);
a_t = r_p_t/(1-e_t);
mu_Triton = 1428.3;
r_SOI_T = 11968;

r_a = 355000;

rp_vect = r_Neptune + linspace(1000,200e3,1000);
e_vect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

for j = 1:numel(e_vect)
    for i = 1:numel(rp_vect)
        
        r_p = rp_vect(i);
        ec = e_vect(j);
        a = (r_a+r_p)/2;
    
        v_sc = sqrt(mu_Neptune/(a*(1-ec^2)))*(1-ec);
        v_inf = v_sc - v_Triton;
        
        a = (2/r_SOI_T - v_inf^2/mu_Triton)^(-1);
        DV(j,i) = sqrt(mu_Triton*(2/(200+r_Triton) - 1/a)) - sqrt(mu_Triton/(a_t*(1-e_t^2)))*(1+e_t);
        
    end
    figure(1)
    plot(rp_vect-r_Neptune, DV(j,:),"LineWidth",2)
    hold on
end



xlabel("Periapsis altitude at Neptune [km]")
ylabel("DV for injection orbit at Triton [km/s]")
legend("e=0.1","e=0.2","e=0.3","e=0.4","e=0.5","e=0.6")
grid on
set(gca,"Fontsize", 14)

%%

r_p_n = 1e3 + r_Neptune;
r_a_n = 2525872 + r_Neptune;
a_1 = (r_p_n + r_a_n)/2;
e_1 = (r_a_n - r_p_n)/(r_a_n + r_p_n);

r_a_n_2 = 345841 + r_Neptune;
a_2 = (r_p_n + r_a_n_2)/2;
e_2 = (r_a_n_2 - r_p_n)/(r_a_n_2 + r_p_n);

DV = sqrt(mu_Neptune/(a_1*(1-e_1^2)))*(1+e_1) - sqrt(mu_Neptune/(a_2*(1-e_2^2)))*(1+e_2)