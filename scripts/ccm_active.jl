using DrWatson
@quickactivate "DynCausal"
using Plots
using NPZ
using FFTW
plotlyjs()
include(srcdir("ccm_utils.jl"))
## =====

data = npzread(datadir("exp_pro", "v1_anim5_tp6_actwvelp.npy"))

## ======
function denoise_fourier(x; win=400)
    fts = fftshift(fft(x))
    mp = Int(ceil(median(1:length(x))))
    sx = zeros(ComplexF64, length(x))
    inds_a = mp-win:mp-1
    inds_b = mp+1:mp+win
    sx[inds_a] .= fts[inds_a]
    sx[inds_b] .= fts[inds_b]
    return ifft(ifftshift(sx)) |> real
end

function filter_active_data(data, threshold; thresh_fn=mean)
    mns = thresh_fn(data[1:end-1, :], dims=2)[:]
    active_inds = findall(x -> x > threshold, mns)
    return [data[active_inds, :]; data[end:end, :]]
end

## =====

function do_cmm(Ds, Ts, act, x_ind, y_ind, Ls::StepRange; d=20)
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

        push!(results, (L, rho_X, rho_Y, xt, yt))

    end
    Ls = [r[1] for r in results]
    rho_Xs = [r[2] for r in results]
    rho_Ys = [r[3] for r in results]
    xts = [r[4] for r in results]
    yts = [r[5] for r in results]
    return (Ls, rho_Xs, rho_Ys, xts, yts)
end

function collect_ccm(shadow_manifolds, τs, ccm_data, ids, Ls)
    rho_Xs = zeros(length(ids), length(Ls))
    rho_Ys = zeros(length(ids), length(Ls))
    for (ind, (i, j)) in enumerate(ids)
        Threads.@spawn begin
            L, rho_X, rho_Y, xt, yt = do_cmm(shadow_manifolds, τs, ccm_data, i, j, Ls)
            rho_Xs[ind, :] = rho_X
            rho_Ys[ind, :] = rho_Y
        end
    end
    return rho_Xs, rho_Ys
end

## ====

data_active = filter_active_data(data_active, 1.005)

data_denoised = let
    # df = denoise_fourier.(eachrow(data_active[1:end-1, :]), win=60)
    df = kalman_filter.(eachrow(data_active[1:end-1, :]), 0.1, 6.0)
    [reduce(vcat, df'); data_active[end:end, :]]
end

@time Ds, Ts, Es = get_embeddings_pecuzal(data_denoised, clean_func=linear_detrend)

manifold_inds = findall(x -> length(x) > 1, Ts)

shadow_manifolds = Ds[manifold_inds]
E = Es[manifold_inds]
T = Ts[manifold_inds]

ccm_data = data_denoised[manifold_inds, :]
ids = get_active_pairs(1:length(manifold_inds)-1, length(manifold_inds))

n0f(f, x) = f(x[x.!=0])

T_ = trunc.(Int, n0f.(minimum, T))
Ls = 200:500:7000
@time rho_Xs, rho_Ys = collect_ccm(shadow_manifolds, T_, ccm_data, ids, Ls)

begin
    px = plot(Ls, rho_Xs', legend=false)
    xlabel!("Library length")
    ylabel!("ρ")

    py = plot(Ls, rho_Ys', legend=false)
    xlabel!("Library length")
    ylabel!("ρ")

    plot(px, py, layout=(2, 1))
end

function plot_ccm_pair(id_ind)
    xind, yind = ids[id_ind]
    pc = plot()
    plot!(pc, rho_Ys[id_ind, :], label="X->Y")
    plot!(pc, rho_Xs[id_ind, :], label="Y->X")

    pt = plot(legend=false)
    if yind == length(manifold_inds)
        plot!(pt, norm_std(ccm_data[yind, :]) ./ 2 .+ 1, label="velP")
    else
        plot!(pt, norm_std(ccm_data[yind, :]), label="Neuron $yind")
    end
    plot!(pt, norm_std(ccm_data[xind, :]), label="Neuron $xind")

    plot(pc, pt, layout=(2, 1), legend=:best, foreground_color_legend=nothing, background_color_legend=nothing)
end

inds = sortperm(rho_Ys[:, end])[end-10:end]

ind = 0
begin
    ind += 1
    plot_ccm_pair(inds[ind])
end

## ==== Find neuron n-plicates

cc = cor(ccm_data')
triu(cc, 1) |> heatmap
ct = triu(cc, 1)
duplicate_inds = findall(x -> x >= 0.9, ct)

plot(ccm_data[3, :])
plot!(ccm_data[18, :])

## ====

function plot_freq_spectrum(x)
    N = length(x)
    fft_x = fft(x)
    freqs = fftfreq(N, 15)

    plot(freqs[1:N÷2], log.(abs.(fft_x[1:N÷2])),
        xlabel="Frequency", ylabel="Magnitude",
        title="Frequency Spectrum")
end

plot_freq_spectrum(data[101, :])
plot_freq_spectrum(denoise_fourier(data[101, :], win=200))

using DSP

function plot_psd(x, fs)
    psd = periodogram(x, fs=fs)
    plot(psd.freq, psd.power,
        xlabel="Frequency (Hz)", ylabel="Power/Frequency",
        title="Power Spectral Density", xscale=:log10, yscale=:log10)
end

function plot_psd!(x, fs)
    psd = periodogram(x, fs=fs)
    plot!(psd.freq, psd.power,
        xlabel="Frequency (Hz)", ylabel="Power/Frequency",
        title="Power Spectral Density", xscale=:log10, yscale=:log10)
end

plot_psd(data[101, :], 15)
plot_psd!(denoise_fourier(data[101, :], win=200), 15)

sp = spectrogram(data[101, :], fs=15)
heatmap(sp.time, sp.freq, log.(sp.power))

function kalman_filter(y, Q, R)
    n = length(y)
    x = zeros(n)
    P = zeros(n)

    x[1] = y[1]
    P[1] = 1

    for k in 2:n
        # Prediction
        x_pred = x[k-1]
        P_pred = P[k-1] + Q

        # Update
        K = P_pred / (P_pred + R)
        x[k] = x_pred + K * (y[k] - x_pred)
        P[k] = (1 - K) * P_pred
    end

    return x
end

kx = kalman_filter(data[101, :], 0.1, 6.0)

plot(data[101, :])
plot!(kx)