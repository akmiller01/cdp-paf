library(data.table)
library(Hmisc)

setwd("~/git/cdp-paf/")

dat = fread("./large_data/iati-text-enhanced-crs.csv")
describe(dat$iati_match_type)
duplicated_ids = dat$matched_iati_identifier[which(duplicated(dat$matched_iati_identifier))]
View(table(duplicated_ids))

matches = subset(dat, iati_match_type!="")
long = subset(matches,
    iati_match_type=="identifier" |
      (iati_match_type=="project title" & nchar(ProjectTitle) > 20) |
      (iati_match_type=="short description" & nchar(ShortDescription) > 20) |
      (iati_match_type=="long description" & nchar(LongDescription) > 20)
)
duplicated_ids = long$matched_iati_identifier[which(duplicated(long$matched_iati_identifier))]
View(table(duplicated_ids))

sub = subset(dat, matched_iati_identifier=="CH-4-2021-2018011021")
