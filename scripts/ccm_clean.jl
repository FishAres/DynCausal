using DrWatson
@quickactivate "DynCausal"
using LinearAlgebra, Statistics
using DynamicalSystems, NearestNeighbors
using Plots
using JLD2
using MAT
## =====

data_path = "C:/Users/aresf/Desktop/OMM_archive_submission_final/OMM_archive_submission_final/data"

proj_meta = matread(data_path * "/OMM_1_meta.mat")["proj_meta"]["rd"][5]["act"][:, 6]

## ====

# animal, condition = 5, 4
# cond = proj_meta["rd"][animal]["Condition"]
# tp = findall(x -> x == 8.0, proj_meta["rd"][animal]["timepoint"])

# act = vcat(proj_meta["rd"][animal]["act"][:, 6]...)
act = vcat(proj_meta...)
inds = sortperm(mean(act, dims=2)[:])[end-10:end]

## ====
function norm_std(x)
    μ, σ = mean(x), std(x)
    return (x .- μ) ./ σ
end

function fast_stack(x)
    X = zeros(size(x))'
    for j in 1:size(X, 2)
        X[:, j] = x.data[j]
    end
    return collect(X)
end

function statespace_to_vec(m, E_)
    q = fast_stack(m)[1:E_, :]
    [q[:, i] for i in 1:size(q, 2)]
    # [m[i, :] for i in 1:size(m, 1)]
end

function get_embeddings(x)
    Ds, Ts, Es = [], [], []
    for cell_act in eachrow(x)
        x_norm = norm_std(cell_act)
        D, τ, e = optimal_separated_de(x_norm)
        push!(Ds, D)
        push!(Ts, τ)
        push!(Es, e)
    end
    return Ds, Ts, Es
end

function find_nearest_neighbors(point, manifold::StateSpaceSet, k)
    kdtree = BallTree(manifold.data)
    idxs, dists = knn(kdtree, point, k, true)
end

function find_nearest_neighbors(point, kdtree::BallTree, k)
    idxs, dists = knn(kdtree, point, k, true)
end

function compute_weights(distances::Vector{Float64})
    u = maximum(distances)
    weights = @. exp(-distances / u)
    weights ./= sum(weights)
    return weights
end

function cross_map(manifold::StateSpaceSet, target::Vector{Float64}, E::Int, tau_target::Int, d::Int)
    estimates = Float64[]
    manifold_kdtree = BallTree(manifold.data)
    target_length = length(target)
    for i in 1:d:length(manifold)
        neighbors, distances = find_nearest_neighbors(manifold[i], manifold_kdtree, E + 1)
        weights = compute_weights(distances)
        target_indices = [n + (E - 1) * tau_target for n in neighbors]
        target_indices = filter(idx -> idx <= target_length, target_indices)

        if length(target_indices) < E + 1
            continue  # Skip this point if we don't have enough valid indices
        end

        estimate = sum(target[target_indices] .* weights[1:length(target_indices)])
        estimate = sum(target[target_indices] .* weights)
        push!(estimates, estimate)
    end
    return estimates
end

## ====

@time Ds, Ts, Es = get_embeddings(act)


function do_cmm(Ds, act, x_ind, y_ind, Ls::StepRange; d=20)
    M_X, M_Y = Ds[x_ind], Ds[y_ind]
    l, E = minimum(size.((M_X, M_Y)))
    results = []
    for L in Ls
        X_short = act[x_ind, 1:L]
        Y_short = act[y_ind, 1:L]

        X_est = cross_map(M_Y[1:L], X_short, E, Ts[x_ind], d)
        Y_est = cross_map(M_X[1:L], Y_short, E, Ts[y_ind], d)

        start_idx = (E - 1) * max(tau_X, tau_Y) + 1

        xt = X_short[start_idx:(length(X_est)+start_idx-1)]
        yt = Y_short[start_idx:(length(Y_est)+start_idx-1)]

        rho_X = isempty(xt) ? NaN : cor(xt, X_est)
        rho_Y = isempty(yt) ? NaN : cor(yt, Y_est)

        # try
        # rho_X = cor(X_short[start_idx:(length(X_est)+start_idx-1)], X_est)
        # catch
        # rho_X = NaN
        # end
        # try
        # rho_Y = cor(Y_short[start_idx:(length(Y_est)+start_idx-1)], Y_est)
        # catch
        # rho_Y = NaN
        # end

        push!(results, (L, rho_X, rho_Y))

    end
    Ls = [r[1] for r in results]
    rho_Xs = [r[2] for r in results]
    rho_Ys = [r[3] for r in results]
    return (Ls, rho_Xs, rho_Ys)
end


x_ind, y_ind = 101, 36
L = 500
inds

begin
    ids = []
    for i in 1:126
        for j in 1:126
            if i != j && (j, i) ∉ ids
                push!(ids, (i, j))
            end
        end
    end
end

@time begin
    Ls = 200:100:7000
    rho_Xs, rho_Ys = [], []
    for (ind, (i, j)) in enumerate(ids)
        L, rho_X, rho_Y = do_cmm(Ds, act, i, j, Ls)
        push!(rho_Xs, rho_X)
        push!(rho_Ys, rho_Y)
    end
end

rho_end = reduce(hcat, [[r[end], j[end]] for (r, j) in zip(rho_Xs, rho_Ys)])
plot(rho_end[1, :])
plot!(rho_end[2, :])

p = plot(Ls, [rho_Xs[1] rho_Ys[1]], label=["X causes Y" "Y causes X"],
    xlabel="Library Size", ylabel="Correlation ρ",
    title="Convergent Cross Mapping",
    legend=:bottomright)

