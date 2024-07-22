using LinearAlgebra, Statistics
using DynamicalSystems, NearestNeighbors

function norm_std(x)
    μ, σ = mean(x), std(x)
    return (x .- μ) ./ σ
end

function linear_detrend(y)
    x = 1:length(y)
    b = cov(x, y) / var(x)
    a = mean(y) - b * mean(x)
    return y .- (a .+ b .* x)
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
end

function get_embeddings(x; clean_func=linear_detrend, nn_func="ifnn")
    Ds, Ts, Es = [], [], []
    for cell_act in eachrow(x)
        x_norm = clean_func(cell_act)
        D, τ, e = optimal_separated_de(x_norm, nn_func)
        push!(Ds, D)
        push!(Ts, τ)
        push!(Es, e)
    end
    return Ds, Ts, Es
end

function get_embeddings_pecuzal(x; clean_func=linear_detrend)
    Ds, Ts, Es = [], [], []
    for (i, cell_act) in enumerate(eachrow(x))
        Threads.@spawn begin
            x_norm = clean_func(cell_act)
            Y, τ_vals, ts_vals, ΔLs, ϵ = pecuzal_embedding(x_norm)
            push!(Ds, Y)
            push!(Ts, τ_vals)
            push!(Es, ϵ)
        end
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


function do_cmm(Ds, act, x_ind, y_ind, Ls::StepRange; d=20)
    M_X, M_Y = Ds[x_ind], Ds[y_ind]
    l, E = minimum(size.((M_X, M_Y)))
    results = []
    for L in Ls
        X_short = act[x_ind, 1:L]
        Y_short = act[y_ind, 1:L]

        X_est = cross_map(M_Y[1:L], X_short, E, Ts[x_ind], d)
        Y_est = cross_map(M_X[1:L], Y_short, E, Ts[y_ind], d)

        start_idx = (E - 1) * max(Ts[x_ind], Ts[y_ind]) + 1

        xt = X_short[start_idx:(length(X_est)+start_idx-1)]
        yt = Y_short[start_idx:(length(Y_est)+start_idx-1)]

        rho_X = isempty(xt) ? NaN : cor(xt, X_est)
        rho_Y = isempty(yt) ? NaN : cor(yt, Y_est)

        push!(results, (L, rho_X, rho_Y))

    end
    Ls = [r[1] for r in results]
    rho_Xs = [r[2] for r in results]
    rho_Ys = [r[3] for r in results]
    return (Ls, rho_Xs, rho_Ys)
end

function get_active_manifolds(data; act_thresh=1.005, thresh_func=mean)
    act = data[1:end-1, :]
    velP = data[end, :]
    mns = thresh_func(act, dims=2)[:]
    active_inds = findall(x -> x > act_thresh, mns)
    # inactive_inds = setdiff(1:size(act, 1), active_inds)
    active = act[active_inds, :]
    # inactive = act[inactive_inds, :]
    ccm_data = [active; reshape(velP, 1, :)]
    shadow_manifolds = Ds[[active_inds; 163]]

    return ccm_data, shadow_manifolds, active_inds
end

function get_active_pairs(active_inds, velP_ind)
    ids = []
    for i in [active_inds; velP_ind]
        for j in [active_inds; velP_ind]
            if i != j && (j, i) ∉ ids
                push!(ids, (i, j))
            end
        end
    end
    return ids
end


function collect_ccm(shadow_manifolds, ccm_data, ids, Ls)
    rho_Xs = zeros(length(ids), length(Ls))
    rho_Ys = zeros(length(ids), length(Ls))
    for (ind, (i, j)) in enumerate(ids)
        Threads.@spawn begin
            L, rho_X, rho_Y = do_cmm(shadow_manifolds, ccm_data, i, j, Ls)
            rho_Xs[ind, :] = rho_X
            rho_Ys[ind, :] = rho_Y
        end
    end
    return rho_Xs, rho_Ys
end