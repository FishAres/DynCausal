using DrWatson
@quickactivate "DynCausal"
using Plots
using NPZ
plotlyjs()
include(srcdir("ccm_utils.jl"))
## =====

# data_path = "C:/Users/aresf/Desktop/OMM_archive_submission_final/OMM_archive_submission_final/data"

# proj_meta = matread(data_path * "/OMM_1_meta.mat")["proj_meta"]["rd"][5]

# act = reduce(vcat, proj_meta["act"][:, 6])
# velP = -proj_meta["velP_smoothed"][1, 6]

# data = [act; velP]

data = npzread(datadir("exp_pro", "v1_anim5_tp6_actwvelp.npy"))

using ToeplitzMatrices

H = Hankel(data[101, :])
U, Σ, V = svd(H)
begin
    plot(U[:, 1], label="col. 1")
    plot!(U[:, 2], label="col. 2")
    plot!(U[:, 3], label="col. 3")
end



using Peaks

using SignalDecomposition

m = 5
k = 30
Q = [2, 2, 2, 3, 3, 3, 3]
s = data[101, :]
x, r = decompose(s, ManifoldProjection(m, Q, k))

H = Hankel(x)
U, Σ, V = svd(H)

plot(cumsum(Σ) ./ sum(Σ))

plot(x)
plot(r)

## ======
D, τ, e = pecuzal_embedding(linear_detrend(data[101, :]))
Y, τ_vals, ts_vals, ΔLs, ϵ = pecuzal_embedding(linear_detrend(data[101, :]))

τ_vals
Y[:, 1]
plot(Y[:, 1], Y[:, 2])


begin
    p1 = plot(data[101, :])
    xlabel!("Time [frame]")
    ylabel!("Fluorescence")
    p2 = plot(D[:, 1], D[:, 2], D[:, 3])
    plot(p1, p2, layout=(1, 2))
end


## ====

@time Ds, Ts, Es = get_embeddings_pecuzal(data, clean_func=norm_std)



ccm_data, shadow_manifolds, active_inds = get_active_manifolds(data; act_thresh=1.005)
ids = get_active_pairs(active_inds, size(data, 1))

ms = filter(x -> length(size(x)) > 1, Ds)
Ls = 200:500:7000
rho_Xs, rho_Ys = collect_ccm(shadow_manifolds, ccm_data, ids, Ls)




begin
    px = plot(Ls, rho_Xs', legend=false)
    xlabel!("Library length")
    ylabel!("ρ")

    py = plot(Ls, rho_Ys', legend=false)
    xlabel!("Library length")
    ylabel!("ρ")

    plot(px, py, layout=(2, 1))
end

ids[302]

function plot_ccm_pair(id_ind)
    xind, yind = ids[id_ind]
    pc = plot()
    plot!(pc, rho_Ys[id_ind, :], label="X->Y")
    plot!(pc, rho_Xs[id_ind, :], label="Y->X")

    pt = plot(legend=false)
    if yind == size(data, 1)
        plot!(pt, norm_std(ccm_data[yind, :]) ./ 2 .+ 1, label="velP")
    else
        plot!(pt, norm_std(ccm_data[yind, :]), label="Neuron $yind")
    end
    plot!(pt, norm_std(ccm_data[xind, :]), label="Neuron $xind")

    plot(pc, pt, layout=(2, 1), legend=:best, foreground_color_legend=nothing, background_color_legend=nothing)
end

inds = sortperm(rho_Ys[:, end])[end-10:end]

ids[310]

plot_ccm_pair(inds[4])



# Usage
detrended_data = linear_detrend(data[21, :])

plot(detrended_data)
plot!(data[21, :] .- 1.0)

