using DrWatson
@quickactivate "DynCausal"

using DynamicalSystems
using CairoMakie

function lorenz96_rule!(du, u, p, t)
    F = p[1]
    N = length(u)
    # 3 edge cases
    du[1] = (u[2] - u[N-1]) * u[N] - u[1] + F
    du[2] = (u[3] - u[N]) * u[1] - u[2] + F
    du[N] = (u[1] - u[N-2]) * u[N-1] - u[N] + F
    # then the general case
    for n in 3:(N-1)
        du[n] = (u[n+1] - u[n-2]) * u[n-1] - u[n] + F
    end
    return nothing # always `return nothing` for in-place form!
end

N = 6
u0 = range(0.1, 1; length=N)
p0 = [8.0]
lorenz96 = CoupledODEs(lorenz96_rule!, u0, p0)

total_time = 12.5
sampling_time = 0.02
Y, t = trajectory(lorenz96, total_time; Ttr=2.2, Δt=sampling_time)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="time", ylabel="variable")
for var in columns(Y)
    lines!(ax, t, var)
end
fig

using OrdinaryDiffEq
diffeq = (alg=Vern9(), abstol=1e-9, reltol=1e-9)

lorenz96_vern = ContinuousDynamicalSystem(lorenz96_rule!, u0, p0; diffeq)

Y, t = trajectory(lorenz96_vern, total_time; Ttr=2.2, Δt=sampling_time)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="time", ylabel="variable")
    for var in columns(Y)
        lines!(ax, t, var)
    end
    fig
end

## Lyapunov spectrum
steps = 10_000
lyapunovspectrum(lorenz96, steps)

#??
tmp = lyapunov_from_data(Y, 1:100)
plot(tmp)

## attractors

xg = yg = range(-2, 2; length=400)

mapper = AttractorsViaRecurrences(henon, (xg, yg); sparse=false)

basins, attractors = basins_of_attraction(mapper)

begin
    fig, ax = heatmap(xg, yg, basins)
    x, y = columns(X)
    y_ = reshape(y, 1, :)
    scatter!(ax, transpose(y_)[:], x; color="black")
    fig
end

## ComplexityMeasures
prob_est = ValueHistogram(50)
entropy(prob_est, X)

grassberger_proccacia_dim(X)

R = RecurrenceMatrix(Y, 8.0)
Rg = grayscale(R)
rr = recurrencerate(R)
heatmap(Rg; colormap=:grays)

## delay embeddings
w = Y[:, 1]
plot(w)
D, τ, e = optimal_separated_de(w)

begin
    fig = Figure()
    axs = [Axis3(fig[1, i]) for i in 1:2]
    for (S, ax) in zip((Y, D), axs)
        lines!(S[:, 1], S[:, 2], S[:, 3])
    end
    fig
end

plot(Y[:, 1], Y[:, 2], Y[:, 3])
plot(D[:, 1], D[:, 2], D[:, 3])

ld = lyapunov_from_data(D, 1:5)
ly = lyapunov_from_data(Y, 1:5)

begin
    fig = Figure()
    lines(ld)
    lines!(ly)
    fig
end



