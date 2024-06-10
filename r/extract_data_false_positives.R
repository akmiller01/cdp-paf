list.of.packages <- c("data.table", "openxlsx", "tidyverse", "Hmisc", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)
rm(list.of.packages,new.packages)

setwd("~/git/cdp-paf/")

### Load ####
paf = read.xlsx("large_data/Full PAF dataset.xlsx", sheet="PAF", na.strings = c("", "#N/A"))
paf = paf[which(!duplicated(paf)),]
crisis = read.xlsx("large_data/Full PAF dataset.xlsx", sheet="Total crisis financing", na.strings = c("", "#N/A"))
crisis = crisis[which(!duplicated(crisis)),]
dat = fread("./large_data/iati-text-enhanced-crs.csv")
dat$long = dat$iati_match_type=="identifier" |
  (dat$iati_match_type=="project title" & nchar(dat$ProjectTitle) > 20) |
  (dat$iati_match_type=="short description" & nchar(dat$ShortDescription) > 20) |
  (dat$iati_match_type=="long description" & nchar(dat$LongDescription) > 20)
sum(dat$iati_text=="")
dat$iati_text[which(!dat$long)] = ""
sum(dat$iati_text=="")
dat$long = NULL
dat = dat[which(!duplicated(dat)),]
dat = dat[which(dat$Year %in% c(2017:2021))]

### Join metadata from CDP files ####
common_names = names(crisis)
common_names = names(dat)[which(names(dat) %in% common_names)]
crs = dat[,c(common_names, "iati_text"), with=F]

### Join/process PAF ####
paf$label = "PAF"

paf = paf[,c(common_names, "label")]
crs = merge(crs, paf, by=common_names, all=T)

### Concatenate textual columns ####
textual_cols_for_classification = c(
  "ProjectTitle",
  "ShortDescription",
  "LongDescription",
  "iati_text"
)
textual_cols_for_classification %in% names(crs)

crs = crs %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T)

crs$label[which(is.na(crs$label))] = "Not PAF"
crs = crs[,c("text","label")]
crs = unique(crs)

quotemeta <- function(string) {
  str_replace_all(string, "(\\W)", "\\\\\\1")
}

remove_punct = function(string){
  str_replace_all(string, "[[:punct:]]", " ")
}

collapse_whitespace = function(string){
  str_replace_all(string, "\\s+", " ")
}

crs$clean_text = collapse_whitespace(remove_punct(tolower(trimws(crs$text))))

keywords = fread("data/keywords.csv")
keywords$keyword = quotemeta(collapse_whitespace(remove_punct(tolower(trimws(keywords$keyword)))))
# Exclude relief for now because extra logic for bad relief and debt relief
keywords = subset(keywords, keyword != "relief")

paf_keywords = subset(keywords, category!="CF")$keyword
paf_regex = paste0(
  "\\b",
  paste(paf_keywords, collapse="\\b|\\b"),
  "\\b"
)

crs$`PAF keyword match` = grepl(paf_regex, crs$text, perl=T, ignore.case = T)

crs = subset(crs, `PAF keyword match`)
describe(crs$label)

fwrite(crs, "./large_data/data_for_false_positive_distinction_training.csv")
