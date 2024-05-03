list.of.packages <- c("data.table", "ggplot2")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/cdp-paf/"
setwd(wd)

crs = fread("large_data/crs_2022_predictions.csv")
# Set blanks to false and 0
blanks = c("", "-")
blank_indices = which(crs$project_title %in% blanks & crs$short_description %in% blanks & crs$long_description %in% blanks)
crs$`Crisis finance confidence`[blank_indices] = 0
crs$`Crisis finance predicted`[blank_indices] = F
crs$`PAF confidence`[blank_indices] = 0
crs$`PAF predicted`[blank_indices] = F
crs$`AA confidence`[blank_indices] = 0
crs$`AA predicted`[blank_indices] = F

# Change thresholds
# crs$`AA predicted`[which(crs$`AA confidence`<0.75)] = F
# crs$`PAF predicted`[which(crs$`PAF confidence`<0.75)] = F
# crs$`Crisis finance predicted`[which(crs$`Crisis finance confidence`<0.75)] = F

# Set PAF confidence equal to AA confidence if AA predicted and PAF not
crs$`PAF confidence`[which(crs$`AA predicted` & !crs$`PAF predicted`)] = 
  crs$`AA confidence`[which(crs$`AA predicted` & !crs$`PAF predicted`)]
crs$`PAF predicted`[which(crs$`AA predicted`)] = T

# Set CF confidence equal to PAF confidence if PAF predicted and CF not
crs$`Crisis finance confidence`[which(crs$`PAF predicted` & !crs$`Crisis finance predicted`)] = 
  crs$`PAF confidence`[which(crs$`PAF predicted` & !crs$`Crisis finance predicted`)]
crs$`Crisis finance predicted`[which(crs$`PAF predicted`)] = T

# From source data, 23.6% of CRS is Crisis financing
# 1.2% of crisis financing is PAF
# 22.5% of PAF is AA
mean(crs$`Crisis finance predicted`) # 30.9%
cf = subset(crs, `Crisis finance predicted`)
cf = cf[order(-cf$`Crisis finance confidence`)]
notcf = subset(crs, !`Crisis finance predicted`)
mean(cf$`PAF predicted`) # 2.3%
paf = subset(cf, `PAF predicted`)
paf = paf[order(-paf$`PAF confidence`)]
notpaf = subset(cf, !`PAF predicted`)
mean(paf$`AA predicted`) # 8.4%
aa = subset(paf, `AA predicted`)
aa = aa[order(-aa$`AA confidence`)]
notaa = subset(paf, !`AA predicted`)

ggplot(crs, aes(x=`Crisis finance confidence`)) + geom_density()
ggplot(cf, aes(x=`Crisis finance confidence`)) + geom_density()
ggplot(notcf, aes(x=`Crisis finance confidence`)) + geom_density()

ggplot(cf, aes(x=`PAF confidence`)) + geom_density()
ggplot(paf, aes(x=`PAF confidence`)) + geom_density()
ggplot(notpaf, aes(x=`PAF confidence`)) + geom_density()

ggplot(paf, aes(x=`AA confidence`)) + geom_density()
ggplot(aa, aes(x=`AA confidence`)) + geom_density()
ggplot(notaa, aes(x=`AA confidence`)) + geom_density()

crs = crs[order(
  -crs$`AA predicted`,
  -crs$`PAF predicted`,
  -crs$`Crisis finance predicted`,
  -crs$`AA confidence`,
  -crs$`PAF confidence`,
  -crs$`Crisis finance confidence`
),]
fwrite(crs,
       "large_data/crs_2022_predictions_ordered.csv")
skinny_cols = c(
  "project_title",
  "short_description",
  "long_description",
  "Crisis finance predicted",
  "Crisis finance confidence",
  "PAF predicted",
  "PAF confidence",
  "AA predicted",
  "AA confidence"
)
fwrite(cf[,skinny_cols,with=F], "large_data/crisis_finance_2022_predictions_ordered_skinny.csv")
fwrite(paf[,skinny_cols,with=F], "large_data/paf_2022_predictions_ordered_skinny.csv")
fwrite(aa[,skinny_cols,with=F], "large_data/aa_2022_predictions_ordered_skinny.csv")

