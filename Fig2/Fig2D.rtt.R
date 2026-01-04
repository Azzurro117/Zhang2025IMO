setwd("E:/0_0_0/AAAAA.IBV/20250407_NEW/Regions/Timetree")
library(ggplot2)
library(ggpmisc)
library(ggtree)

rtt_all = read.csv("rtt_all.csv", header=T)
df_all2 = data.frame(rtt_all$date, rtt_all$distance)
p_all2 = ggplot(data = df_all2, aes(x = rtt_all.date, y = rtt_all.distance)) +
	geom_smooth(method = "lm", se=TRUE, formula = y ~ x, color="#696969", fill="lightgray", alpha=0.7, size=0.8,) +
	geom_point(size=3, alpha=0.8, color="#808080", shape=21, fill="#343434") +
	stat_poly_eq(formula = y~x, aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), parse = TRUE) +
	theme_bw() +
	scale_x_continuous(breaks = c(1940,1960,1980,2000,2020,2025)) +
	coord_cartesian(ylim = c(0,0.4), xlim = c(1940,2025)) +
	theme(panel.grid = element_blank()) +
	xlab("Time in years") +
	ylab("Root-to-tip distance")
p_all2
ggsave("rtt_all.pdf", width = 4, height = 3.2)
