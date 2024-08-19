library(ggplot2)
library(pROC)
library(grpnet)
library(glmnet)
library(xlsx)
options(warn=-1)



v2m = function(v){
  x=length(v)
  n=(1+sqrt(1+8*x))/2
  mat <- matrix(0, nrow = n, ncol =  n)
  mat[upper.tri(mat, diag = FALSE)] <- v
  mat = mat + t(mat)
  return(mat)
}


m2v = function(m){
  x=dim(m)[1]
  n=(x-1)*x/2
  v <- vector("double", n)
  v = m[upper.tri(m, diag = FALSE)] 
  return(v)
}


library(R.matlab)
read_vec = readMat("vector.mat")

read_vec[1]


## edges 
edges = as.numeric(read_vec[[1]])
connectivitvals = v2m(edges)

library('igraph');
connectivitvalsones=connectivitvals
connectivitvalsones[lower.tri(connectivitvalsones,diag = F)] =0
t=which(connectivitvalsones!=0, arr.ind=TRUE)
t <- cbind(t, connectivitvals[which(connectivitvals!=0,arr.ind=TRUE)]) 
t.graph=graph.data.frame(t,directed=F)
E(t.graph)$color <- ifelse(E(t.graph)$V3 > 0,'red','blue') 
#t.names <- colnames(cor.matrix)[as.numeric(V(t.graph)$name)]
minC <- rep(-Inf, vcount(t.graph))
maxC <- rep(Inf, vcount(t.graph))
minC[1] <- maxC[1] <- 0
l <- layout_with_fr(t.graph, minx=minC, maxx=maxC,
                    miny=minC, maxy=maxC)      

pathnames='anatomyInfo_whiston_new.csv'
datanmes=read.csv(pathnames, header = TRUE, sep = ",", quote = "")
datanmes$ROI

#noreadcsf=c(148,152,161,314,318,327) # dont read csf already in matlab

#datanmes=datanmes[-noreadcsf]

datanmess=datanmes$ROI # remove csf
#datanmess=datanmes$ROI



par(mfrow=c(1,1))

#set.vertex.attribute(t.graph, "name", value=datanmes$ROI   )


png("SBL_abprop.png", units="cm", width=20, height=20, res=300)  

plot(t.graph, layout=l, 
     rescale=T,
     asp=0,
     edge.arrow.size=0.1, 
     vertex.label.cex=0.8, 
     vertex.label.family="Helvetica",
     vertex.label.font=4,
     #vertex.label=t.names,
     vertex.shape="circle", 
     vertex.size=7, 
     vertex.color="white",
     vertex.label.color="black", 
     #edge.color=E(t.graph)$color, ##do not need this since E(t.graph)$color is already defined.
     edge.width=as.integer(cut(abs(E(t.graph)$V3), breaks = 5)))
dev.off()
 connectivitvals=connectivitvals+t(connectivitvals) #symetric

#nonzeroposition=which(connectivityexample=="nonzero", arr.ind=TRUE)
getwd()
filename=paste(getwd(), "/", "valandpos.mat", sep = "")
#writeMat(filename, nonzeroposition = nonzeroposition, connectivitvals = connectivitvals , oddzeroposition=indexofzeros)




subnets=groups(components(t.graph))

subnetsresults=vector(mode = "list", length = length(subnets))
colsumabs=colSums(abs(connectivitvals))
colsum=colSums(connectivitvals)

leftright=datanmes$Bigpart




####################3
for (i in 1:length(subnets)) {
  temp=subnets[[i]]
  temp=as.numeric(temp)
  net=matrix(NA,9,length(temp) )
  net[1,]=as.numeric(temp)
  tt=as.numeric(net[1,])
  #tt=c(1,200)
  #indofleftright=tt>=164
  #net[5,][indofleftright]="Right"
  #net[5,][!indofleftright]="Left"
  
  
  net[2,]=datanmess[temp]
  net[5,]=leftright[temp]
  net[1,]=paste(net[1,],net[5,])
  net[3,]= as.numeric( colsum[temp]   )
  net[4,]= as.numeric( colsumabs[temp]   )
  net[6,]=sum(as.numeric(net[4,]))
  net[7,]=sum(as.numeric(net[3,]))
  for (j in 1:length( net[8,])) {
    tempindex=which(datanmes$ROI %in% net[2,j]  )
    if (net[5,j]=="Right" ) {net[8,j]= max(tempindex) } else { net[8,j]=min(tempindex) }
  }
  net[9,]=sum(abs(as.numeric(net[3,])))/length(net[3,])
  
  subnetsresults[[i]]=net 
}

#install.packages("xlsx")
# library(xlsx)
# 
# 
# for (i in 1:length(subnetsresults)){
#   net=t(subnetsresults[[i]])
#   write.xlsx2(net, "nets.xlsx", sheetName =  paste0(i), append=TRUE )
# }




net_new=matrix(NA, length(subnetsresults),5)


for (j in 1:dim(net_new)[1]) {
  temps=subnetsresults[[j]]
  net_new[j,1]=j
  net_new[j,2]= paste(temps[8,], collapse = ", ")
  net_new[j,3] = paste(paste(temps[5,],temps[2,]), collapse = ", ")
  net_new[j,4] = paste(temps[7,1])
  net_new[j,5] = paste(temps[6,1])
}
colnames(net_new)=c("Sub-Network", "Region Number", "Region Name", "Sub-Network Weight", "Sub_Network Average of reduced sum")


write.xlsx2(net_new, "SBL_abprop.xlsx" )



