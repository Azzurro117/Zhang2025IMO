library(RColorBrewer)
library(ggplot2)
library(ggforce)

ynn <- c(22,20,18,10,9,5,4,3,3,3,3,2,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
serotype <- c("QX","Mass","4/91","TW","Arkansas","Conn","DMV/1639","Cal","Holte","LDT3-A","TC07-2","K","ArkDPI","CAV","DE","FL","Gray","HN08","JMK","Q1","tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10","tag11","tag12","tag13","tag14","tag15","tag16","tag17","tag18","tag19","tag20")

colors = c(
        '#8b1A1A', '#B22222', '#FF3030', '#EE2C2C', '#CD2626', '#EE3B3B',
        '#CD6090', '#EE6AA7', '#FF6EB4', '#FF69B4', '#F08080', '#FFAEB9',           
        '#8B5A2B', '#CD853F', '#EE9A49', '#FFA54F', '#F5DEB3', '#FFE7BA',
        '#EEB422', '#FFC125') 
#        '#EEC900', '#FFD700', '#EEE685', '#FFF68F',
#        '#698B22', '#6B8E23', '#9ACD32', '#B3EE3A', '#C0FF3E', '#98FB98')

A <- data.frame(serotype, ynn)        
pdf("serotype.pdf")
ggplot(data = A,aes(x=reorder(serotype, -ynn),y=ynn))+ 
    geom_bar(width = 0.8,stat = "identity")+
    scale_fill_manual(values=colors)+
    geom_hline(yintercept =c(1,2,5,10,20),linetype=2,linewidth=.1)+
    coord_polar(theta="x",start=-1.57)+
    ylim(-3, 25)+
    theme_minimal()+xlab(" ")+ylab(" ")+
    theme_void()
    
dev.off()
