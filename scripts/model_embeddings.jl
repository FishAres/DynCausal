using DrWatson
@quickactivate "DynCausal"
using DifferentialEquations, Plots, DynamicalSystems

## ====

function izh!(du, u, p, t)
    a, b, c, d, I = p
    du[1] = 0.04 * u[1]^2 + 5 * u[1] + 140 - u[2] + I
    du[2] = a * (b * u[1] - u[2])
end

function thr(u, t, integrator)
    integrator.u[1] >= 30
end

function reset!(integrator)
    integrator.u[1] = integrator.p[3]
    integrator.u[2] += integrator.p[4]
end

threshold = DiscreteCallback(thr, reset!)
current_step = PresetTimeCallback(50, integrator -> integrator.p[5] += 10)
cb = CallbackSet(current_step, threshold)

p = [0.02, 0.2, -50, 2, 0]
u0 = [-65, p[2] * -65]
tspan = (0.0, 300)

prob = ODEProblem(izh!, u0, tspan, p, callback=cb)

sol = solve(prob);
plot(sol, vars=1)
us = reduce(hcat, sol.u)
plot(sol.t, us[1, :])
D, τ, e = optimal_separated_de(us[1, :])

D[:, 1]
plot3d(D[:, 1], D[:, 2], D[:, 3])

De = Matrix(transpose(reduce(hcat, (D[:, 1], D[:, 2], D[:, 3]))))

using MultivariateStats
M = fit(PCA, De)

Yte = predict(M, De)
plot(Yte[1, :], Yte[2, :])




## ==== hodgkin-huxley

# Potassium ion-channel rate functions
alpha_n(v) = (0.02 * (v - 25.0)) / (1.0 - exp((-1.0 * (v - 25.0)) / 9.0))
beta_n(v) = (-0.002 * (v - 25.0)) / (1.0 - exp((v - 25.0) / 9.0))

# Sodium ion-channel rate functions
alpha_m(v) = (0.182 * (v + 35.0)) / (1.0 - exp((-1.0 * (v + 35.0)) / 9.0))
beta_m(v) = (-0.124 * (v + 35.0)) / (1.0 - exp((v + 35.0) / 9.0))

alpha_h(v) = 0.25 * exp((-1.0 * (v + 90.0)) / 12.0)
beta_h(v) = (0.25 * exp((v + 62.0) / 6.0)) / exp((v + 90.0) / 12.0)

function HH!(du, u, p, t)
    gK, gNa, gL, EK, ENa, EL, C, I = p
    v, n, m, h = u

    du[1] = (-(gK * (n^4.0) * (v - EK)) - (gNa * (m^3.0) * h * (v - ENa)) - (gL * (v - EL)) + I) / C
    du[2] = (alpha_n(v) * (1.0 - n)) - (beta_n(v) * n)
    du[3] = (alpha_m(v) * (1.0 - m)) - (beta_m(v) * m)
    du[4] = (alpha_h(v) * (1.0 - h)) - (beta_h(v) * h)
end

current_step = PresetTimeCallback(100, integrator -> integrator.p[8] += 1)

# n, m & h steady-states
n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v))
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v))
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v))

p = [35.0, 40.0, 0.3, -77.0, 55.0, -65.0, 1, 0]
u0 = [-60, n_inf(-60), m_inf(-60), h_inf(-60)]
tspan = (0.0, 1000)

prob = ODEProblem(HH!, u0, tspan, p, callback=current_step)
## ====
sol = solve(prob);
plot(sol, vars=1)

us = reduce(hcat, sol.u)
plot(sol.t, us[1, :])
D, τ, e = optimal_separated_de(us[1, :])

plot3d(D[:, 1], D[:, 2], D[:, 3])

De = Matrix(transpose(reduce(hcat, (D[:, 1], D[:, 2], D[:, 3]))))

using MultivariateStats
M = fit(PCA, De)

Yte = predict(M, De)
plot(Yte[1, :], Yte[2, :])



