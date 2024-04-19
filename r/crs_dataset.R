list.of.packages <- c("data.table", "tidyverse")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)
rm(list.of.packages,new.packages)

setwd("~/git/cdp-paf/")

load("large_data/crs.RData")

names(crs) = gsub(" ", "", names(crs))

textual_cols_for_classification = c(
  "Year",
  "ProjectNumber",
  "RecipientName",
  "RecipientCode",
  "DonorName",
  "DonorCode",
  "ProjectTitle",
  "SectorName",
  "PurposeName",
  "FlowName",
  "ChannelName",
  "ShortDescription",
  "LongDescription"
)

crs = crs[,textual_cols_for_classification, with=F]
crs = unique(crs)

crs = crs %>%
  unite(text, c("ProjectTitle", "ShortDescription", "LongDescription"), sep=" ", na.rm=T, remove=F)

text_unique = which(!duplicated(crs$text))
crs = crs[text_unique,]

col_order = c("text", textual_cols_for_classification) 
crs = crs[,col_order]
fwrite(crs, "large_data/crs_for_dataset.csv")
