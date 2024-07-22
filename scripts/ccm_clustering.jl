using DrWatson
@quickactivate "DynCausal"
using Plots
using NPZ

plotlyjs()

include(srcdir("ccm_utils.jl"))

## =====

data = npzread(datadir("exp_pro", "v1_anim5_tp6_actwvelp.npy"))

## ====

using Clustering

act, velP = data[1:end-1, :], data[end, :]

R = kmeans(act', 6)

inds = assignments(R)

histogram(inds)

plot(act[inds.==6, :]')


using Distances
using MultivariateStats
R.centers

m = fit(PCA, act)
p = predict(m, act)

p_ = p[:, 1:4:end]
scatter(p_[1, :], p_[2, :], p_[3, :], markersize=1, c=1:1875)

plot(p[1, :], p[2, :], c=1:7500)

plot(m.prinvars)
plot!(cumsum(m.prinvars) ./ sum(m.prinvars))


using TSne

at = collect(hcat(linear_detrend.(eachrow(act))...)')
mn = mean(act, dims=2)[:]
id = sortperm(mn)[end-20:end]

Y = tsne(act[:, :], 2, 0, 1000, 10)

scatter(Y[:, 1], Y[:, 2])

using UMAP

model = UMAP_(act', 3)

kn = model.knns

kn[1, :]

A = collect(model.graph)

heatmap(A)

plot(A[87, :])
plot(act[45, :])

heatmap(tmp)

plot(A[101, :])
tmp = act[A[101, :].>0.0, :]
plot(tmp', legend=false)

plot(act[50, :])
plot!(act[101, :])

sortperm(mean(act, dims=2)[:])

model.embedding