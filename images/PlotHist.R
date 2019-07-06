PlotHist <- function(image) {

  hist(image[,,1], breaks=seq(0, 1, length.out=255), col="#FF000099", border="white", xlab="", ylim=c(0, 80000), main="")
  par(new=TRUE)
  hist(image[,,2], breaks=seq(0, 1, length.out=255), col="#00FF0099", border="white", xlab="", ylim=c(0, 80000), main="")
  par(new=TRUE)
  hist(image[,,3], breaks=seq(0, 1, length.out=255), col="#0000FF99", border="white", xlab="", ylim=c(0, 80000), main="")

}
