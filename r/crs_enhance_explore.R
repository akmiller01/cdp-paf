library(data.table)
library(Hmisc)

setwd("~/git/cdp-paf/")

dat = fread("./large_data/iati-text-enhanced-crs.csv")
describe(dat$iati_match_type)
duplicated_ids = dat$matched_iati_identifier[which(duplicated(dat$matched_iati_identifier))]
View(table(duplicated_ids))

sub = subset(dat, matched_iati_identifier=="CH-4-1981000167")
