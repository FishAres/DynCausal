using Statistics, Distances, Plots

function construct_shadow_manifold(series::Vector{Float64}, E::Int, tau::Int)
    N = length(series)
    manifold = [series[i:tau:i+(E-1)*tau] for i in 1:N-(E-1)*tau]
    return manifold
end

function find_nearest_neighbors(point::Vector{Float64}, manifold::Vector{Vector{Float64}}, E::Int)
    distances = [euclidean(point, m) for m in manifold]
    sorted_indices = sortperm(distances)[1:E+1]
    return sorted_indices, distances[sorted_indices]
end

function compute_weights(distances::Vector{Float64})
    u = maximum(distances)
    weights = @. exp(-distances / u)
    weights ./= sum(weights)
    return weights
end

function cross_map(manifold::Vector{Vector{Float64}}, target::Vector{Float64}, E::Int, tau::Int)
    estimates = Float64[]
    for i in 1:length(manifold)
        neighbors, distances = find_nearest_neighbors(manifold[i], manifold, E)
        weights = compute_weights(distances)
        target_indices = [n + (E - 1) * tau for n in neighbors]
        estimate = sum(target[target_indices] .* weights)
        push!(estimates, estimate)
    end
    return estimates
end

function ccm(X::Vector{Float64}, Y::Vector{Float64}, E::Int, tau::Int, L_range::Vector{Int})
    results = []
    for L in L_range
        X_short = X[1:L]
        Y_short = Y[1:L]

        M_X = construct_shadow_manifold(X_short, E, tau)
        M_Y = construct_shadow_manifold(Y_short, E, tau)

        X_est = cross_map(M_Y, X_short, E, tau)
        Y_est = cross_map(M_X, Y_short, E, tau)

        rho_X = cor(X_short[(E-1)*tau+1:end], X_est)
        rho_Y = cor(Y_short[(E-1)*tau+1:end], Y_est)

        push!(results, (L, rho_X, rho_Y))
    end
    return results
end

# Helper function to generate example data
function generate_coupled_logistic_maps(N::Int, r::Float64, b_xy::Float64, b_yx::Float64)
    X = zeros(N)
    Y = zeros(N)
    X[1], Y[1] = rand(2)
    for i in 2:N
        X[i] = X[i-1] * (r - r * X[i-1] - b_xy * Y[i-1])
        Y[i] = Y[i-1] * (r - r * Y[i-1] - b_yx * X[i-1])
    end
    return X, Y
end
## ====
# Main execution

# Generate example data
N = 1000
X, Y = generate_coupled_logistic_maps(N, 3.8, 0.02, 0.1)

# Set CCM parameters
E = 3  # embedding dimension
tau = 1  # time delay
L_range = collect(100:50:900)  # range of library sizes

M_X = construct_shadow_manifold(X, E, tau, L_range[1])
M_Y = construct_shadow_manifold(Y, E, tau, L_range[1])

plot(X)
M_X


function do_cmm(M_X, M_Y, X, Y, E, tau; L_range=collect(100:50:900))
    results = ccm(X, Y, E, tau, L_range)

    # Extract results for plotting
    L_values = [r[1] for r in results]
    rho_X = [r[2] for r in results]
    rho_Y = [r[3] for r in results]
    return L_values, rho_X, rho_Y
end
# Plot results
p = plot(L_values, [rho_X rho_Y], label=["X causes Y" "Y causes X"],
    xlabel="Library Size", ylabel="Correlation ρ",
    title="Convergent Cross Mapping",
    legend=:bottomright)
