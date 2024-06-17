list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

YEAR = 2022

wd = "~/git/cdp-paf/"
setwd(wd)

wb = createWorkbook("large_data/Full PAF dataset 2024.xlsx")

addWorksheet(wb, "Total crisis financing")
addWorksheet(wb, "PAF")
addWorksheet(wb, "Humanitarian funding")
addWorksheet(wb, "AA")

conv_ascii = function(string){
  iconv(string, "latin1", "ASCII", sub="")
}

collapse_whitespace = function(string){
  str_replace_all(string, "\\s+", " ")
}

remove_punct = function(string){
  str_replace_all(string, "[[:punct:]]", " ")
}

remove_lead_zero_if_number = function(strings){
  try_numeric = as.numeric(strings)
  str_try_numeric = as.character(try_numeric)
  str_try_numeric[which(is.na(str_try_numeric))] = strings[which(is.na(str_try_numeric))]
  return(str_try_numeric)
}

clean_text = function(text){
  collapse_whitespace(trimws(remove_punct(tolower(conv_ascii(remove_lead_zero_if_number(trimws(text)))))))
}

auto = fread(paste0("large_data/crs_",YEAR,"_cdp_automated.csv"))

textual_cols_for_classification = c(
  "project_title",
  "short_description",
  "long_description"
)

auto = auto %>%
  unite(join_text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

auto$join_text = clean_text(auto$join_text)

cf_auto = subset(auto, `Crisis finance determination` %in% c("Yes"))
paf_auto = subset(auto, `PAF determination` %in% c("Yes", "Review"))
hum_auto = subset(auto, humanitarian==T)
aa_auto = subset(auto, `AA determination` %in% c("Yes", "Review"))

drop_names = c(
  "join_text",
  "humanitarian",
  "Crisis finance identified",
  "Crisis finance eligible",
  "Crisis finance determination",
  "Crisis finance keyword match",
  "Crisis finance predicted ML",
  "Crisis finance confidence ML",
  "PAF determination",
  "PAF keyword match",
  "PAF predicted ML",
  "PAF confidence ML",
  "AA determination",
  "AA keyword match",
  "AA predicted ML",
  "AA confidence ML",
  "Direct predicted ML",
  "Direct confidence ML",
  "Indirect predicted ML",
  "Indirect confidence ML",
  "Part predicted ML",
  "Part confidence ML"
)
hum_auto[,drop_names] = NULL
writeData(wb, sheet="Humanitarian funding", hum_auto)

# CF

drop_names = c(
  "join_text",
  "humanitarian",
  "Crisis finance identified",
  "Crisis finance eligible",
  "Crisis finance determination",
  "Crisis finance keyword match",
  "Crisis finance predicted ML",
  "PAF determination",
  "PAF keyword match",
  "PAF predicted ML",
  "PAF confidence ML",
  "AA determination",
  "AA keyword match",
  "AA predicted ML",
  "AA confidence ML",
  "Direct predicted ML",
  "Direct confidence ML",
  "Indirect predicted ML",
  "Indirect confidence ML",
  "Part predicted ML",
  "Part confidence ML"
)
cf_auto[,drop_names] = NULL
writeData(wb, sheet="Total crisis financing", cf_auto)

# Join PAF to review
paf_review = read.xlsx("large_data/Reviewed_13June/2022_paf.xlsx")

paf_review = paf_review %>%
  unite(join_text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

paf_review$join_text = clean_text(paf_review$join_text)

paf = subset(paf_review, Review == "PAF")
not_paf = subset(paf_review, Review == 'Not PAF')

stopifnot({length(setdiff(paf$join_text, paf_auto$join_text))==0})
paf = paf[,c("join_text", "Review")]
paf = unique(paf)
stopifnot({length(unique(paf$join_text))==nrow(paf)})
paf_auto = merge(paf_auto, paf, by="join_text", all.y=T)
paf_auto = subset(paf_auto, Review == "PAF")

drop_names = c(
  "join_text",
  "humanitarian",
  "Crisis finance identified",
  "Crisis finance eligible",
  "Crisis finance determination",
  "Crisis finance keyword match",
  "Crisis finance predicted ML",
  "Crisis finance confidence ML",
  "PAF determination",
  "PAF keyword match",
  "PAF predicted ML",
  "AA determination",
  "AA keyword match",
  "AA predicted ML",
  "AA confidence ML",
  "Review"
)
paf_auto[,drop_names] = NULL
writeData(wb, sheet="PAF", paf_auto)

# AA
aa_auto = subset(aa_auto, !(join_text %in% not_paf$join_text))
table(aa_auto$`AA determination`)

drop_names = c(
  "join_text",
  "humanitarian",
  "Crisis finance identified",
  "Crisis finance eligible",
  "Crisis finance determination",
  "Crisis finance keyword match",
  "Crisis finance predicted ML",
  "Crisis finance confidence ML",
  "PAF determination",
  "PAF keyword match",
  "PAF predicted ML",
  "PAF confidence ML",
  "AA keyword match",
  "AA predicted ML"
)
aa_auto[,drop_names] = NULL
writeData(wb, sheet="AA", aa_auto)

saveWorkbook(wb, paste0("large_data/Full PAF dataset ",YEAR,".xlsx"), overwrite=T)

