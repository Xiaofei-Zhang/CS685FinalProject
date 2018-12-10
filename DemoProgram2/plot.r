library(ggplot2)

suppressPackageStartupMessages(library("optparse"))


option_list = list(
  make_option(c("-i", "--input"), action="store", default="0.txt", type='character', help="Input data"),
  make_option(c("-r", "--ref"), action="store", default="training_data.txt", type='character', help="Reference data"),
  make_option(c("-o", "--output"), action="store", default="0.png", type='character', help="Output file")
)

opts = parse_args(OptionParser(option_list=option_list))



df1 <- read.csv(opts$input, header = FALSE)
dfr <- read.csv(opts$ref, header = FALSE)
head(df1)
head(dfr)

g <- ggplot(df1, aes(x=V1, y=V2)) + geom_point(color="red") + geom_point(data = dfr, color="blue")
#g2 <- ggplot(dfr, aes(x=V1, y=V2)) + geom_point()

ggsave(opts$output, plot=g)
