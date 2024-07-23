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


using Wavelets

data_denoised = let
    x_denoised = denoise.(eachrow(data))
    reduce(vcat, x_denoised')
end

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

@time Ds, Ts, Es = get_embeddings_pecuzal(data_denoised, clean_func=linear_detrend)

ccm_data, shadow_manifolds, active_inds = get_active_manifolds(data_denoised; act_thresh=1.005)
ids = get_active_pairs(active_inds, size(data, 1))

ms = filter(x -> length(size(x)) > 1, Ds)
Ls = 200:500:7000
rho_Xs, rho_Ys = collect_ccm(shadow_manifolds, ccm_data, ids, Ls)

maximum(Ts[101])
vp_est = cross_map(Ds[101], data_denoised[163, :], 3, 50, 20)
plot(vp_est)
start_idx = 2 * 50

xt = data_denoised[163, start_idx:length(vp_est)+start_idx-1]

plot(vp_est)
plot!(xt)

filter(x -> length(x) > 1, Ts)

minimum(size.((Ds[101], Ds[163])))

minimum(intersect(Ts[101][2:end], Ts[163][2:end]))

histogram(Ts[101], bins=0:1:60)
histogram!(Ts[163], bins=0:1:60)

Ts[101]

median(Ts[101])
median(Ts[163])

plot(Ts[101])


do_cmm(Ds, data_denoised, 101, 163, Ls)


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

