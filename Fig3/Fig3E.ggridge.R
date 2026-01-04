setwd("E:/0_0_0/AAAAA.IBV/20250407_NEW/Spike_Nglycosylation_NTD")
library(ggplot2)
library(ggridges)
library(viridis)

bc <- read.csv("02.ggridge.mor.csv", header=T)

pdf("02.ggridge.mor.pdf", height=5, width=8)
ggplot(bc, aes(x = rate, y = label)) +
	geom_density_ridges_gradient(aes(fill=stat(x)), scale = 1.2, rel_min_height = 0.0000000000001) +
	scale_x_continuous(breaks = c(0,0.05,0.10,0.15,0.2,0.25)) +
	coord_cartesian(xlim = c(0, 0.25)) +
	theme_ridges(grid = TRUE) +
	scale_fill_gradient2(name = "rate", low="white", midpoint = 0.125, mid= "#F5D668", high="#BD1F32")
dev.off()

bc <- read.csv("02.ggridge.inc.csv", header=T)

pdf("02.ggridge.inc.pdf", height=5, width=8)
ggplot(bc, aes(x = rate, y = label)) +
	geom_density_ridges_gradient(aes(fill=stat(x)), scale = 1.2, rel_min_height = 0.0000000000001) +
	scale_x_continuous(breaks = c(0,0.2,0.4,0.6,0.8,1)) +
	coord_cartesian(xlim = c(0, 1)) +
	theme_ridges(grid = TRUE) +
	scale_fill_gradient2(name = "rate", low="white", midpoint = 0.5, mid= "#F5D668", high="#BD1F32")
dev.off()