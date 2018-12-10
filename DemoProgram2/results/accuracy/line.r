library(ggplot2)

suppressPackageStartupMessages(library("optparse"))


option_list = list(
  make_option(c("-i", "--input"), action="store", default="10D1G.csv", type='character', help="Input data"),
  make_option(c("-r", "--ref"), action="store", default="1G1D.csv", type='character', help="Reference data"),
  make_option(c("-r2", "--ref2"), action="store", default="100sample.csv", type='character', help="Reference data"),
  make_option(c("-o", "--output"), action="store", default="1.png", type='character', help="Output file")
)

opts = parse_args(OptionParser(option_list=option_list))

c <- seq(0,19999, 500)

df1 <- read.csv(opts$input, header = FALSE)
dfr <- read.csv(opts$ref, header = FALSE)
#dfr2 <- read.csv(opts$ref2, header = FALSE)

colnames(df1) <- c("percentage")
colnames(dfr) <- c("percentage")
#colnames(dfr2) <- c("percentage")

df1$iteration <- c
dfr$iteration <- c
#dfr2$iteration <- c

head(df1)
head(dfr)

#g <- ggplot(df1, aes(x=iteration, y=percentage)) + geom_line(color="red") + geom_line(data = dfr, color="blue") + geom_line(data = dfr2, color="orange")
g <- ggplot(df1, aes(x=iteration, y=percentage)) + geom_line(color="red") + geom_line(data = dfr, color="blue")
#g2 <- ggplot(dfr, aes(x=V1, y=V2)) + geom_point()

ggsave(opts$output, plot=g)
