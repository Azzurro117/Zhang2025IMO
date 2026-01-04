# test
library(tidyverse)
library(ggplot2)
library(ggnewscale)
library(openxlsx)

bshape <- tibble(x0 = c(0, 1, 1, 0),
                 y0 = c(1, 1, 1/3, 1/3),
                 x1 = c(0, 0, 0.5, 0.5),
                 y1 = c(0, 1/3, 1/3, 0),
                 x2 = c(0.5, 1, 1, 0.5),
                 y2 = c(1/3, 1/3, 0, 0),
                 id = 1)

ggplot(bshape)+
  geom_polygon(aes(x=x0, y=y0,fill = "black"),
               color = 'grey', 
               linewidth=0.15)+
  geom_polygon(aes(x=x1, y=y1,fill = "yellow"),
               color = 'grey', 
               linewidth=0.15)+
  geom_polygon(aes(x=x2, y=y2,fill = "orange"),
               color = 'grey', 
               linewidth=0.15)



# try
### input data
fname <- 'sim.08.5b.xlsx'
ml_ave <- data.frame((read.xlsx(fname, sheet =  1, rowNames=TRUE)))
ml_min <- data.frame((read.xlsx(fname, sheet =  2, rowNames=TRUE)))
ml_max <- data.frame((read.xlsx(fname, sheet =  3, rowNames=TRUE)))

### merge into one column
ave <- c(ml_ave[,1])
for (i in 2:ncol(ml_ave)) {
  ave <- c(ave, ml_ave[,i])
}
min <- c(ml_min[,1])
for (i in 2:ncol(ml_min)) {
  min <- c(min, ml_min[,i])
}
max <- c(ml_max[,1])
for (i in 2:ncol(ml_max)) {
  max <- c(max, ml_max[,i])
}

### merge 
ml <- expand_grid(x=0:6, y=0:6) %>% 
  mutate(id = 1,
         group = 1:n(),
         position = rep(c(1:7),7),
         AVE = ave,
         MIN = min,
         MAX = max)
### merge shape points
ml %>% 
  inner_join(bshape, by="id", 
             relationship ="many-to-many") -> ml_plot

### draw 
f <- ggplot(data = ml_plot)+
  geom_polygon(aes(x = x+x0,y = y0+5-y, group = group, fill = AVE), color = 'white', linewidth = 0.8) +
  geom_polygon(aes(x = x+x1,y = y1+5-y, group = group, fill = MIN), color = 'white', linewidth = 0.8) +
  geom_polygon(aes(x = x+x2,y = y2+5-y, group = group, fill = MAX), color = 'white', linewidth = 0.8) +
  geom_text(data = ml, mapping = aes(label = sprintf('%0.3f', AVE), x = x+0.5,y = 5-y+2/3), size = 4, color = 'white')+
  geom_text(data = ml, mapping = aes(label = sprintf('%0.2f', MIN), x = x+0.25,y = 5-y+0.17), size = 2, color = 'white')+
  geom_text(data = ml, mapping = aes(label = sprintf('%0.2f', MAX), x = x+0.75,y = 5-y+0.17), size = 2, color = 'white')+
  scale_fill_gradientn(limits = c(0.5, 1), colors = RColorBrewer::brewer.pal(n=11, name = "BuGn")) +
  theme_void() +
  coord_equal() +
  theme(legend.key.height = unit(1, "null"), plot.background = element_rect(fill = "white", color = "white")) +
  labs(fill = 'Similarity')

ggsave("sim.08.5b.pdf", f, width = 5, height = 5)

