list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

YEAR = 2021

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

cf_auto = subset(auto, `Crisis finance determination` %in% c("Yes", "Review"))
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

# Mark all old CF as Yes, keep all other Yes CF
cf_review = fread(paste0("data/new_", YEAR, "_cf.csv"))

cf_review = cf_review %>%
  unite(join_text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

cf_review$join_text = clean_text(cf_review$join_text)

new_cf = subset(cf_auto, join_text %in% cf_review$join_text)
old_cf = subset(cf_auto, !(join_text %in% cf_review$join_text))

old_cf$`Crisis finance determination` = "Yes"
new_cf = subset(new_cf, `Crisis finance determination` == "Yes")

cf = rbind(new_cf, old_cf)
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
cf[,drop_names] = NULL
writeData(wb, sheet="Total crisis financing", cf)

# Join PAF to review
paf_review = fread(paste0("large_data/Reviewed_13June/new_", YEAR, "_paf.csv"))
missed_paf = fread(paste0("data/missed_", YEAR, "_paf.csv"))

paf_review = paf_review %>%
  unite(join_text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

paf_review$join_text = clean_text(paf_review$join_text)

if(YEAR == 2019){
  mismatch_auto = subset(paf_auto, crs_id == "2017000688")
  paf_review[which(paf_review$crs_id == "2017000688"),]$join_text = mismatch_auto$join_text
}

stopifnot({
  length(setdiff(unique(paf_review$join_text), unique(paf_auto$join_text))) == 0
})

missed_paf = missed_paf %>%
  unite(join_text, all_of(c("ProjectTitle", "ShortDescription", "LongDescription")), sep=" ", na.rm=T, remove=F)

missed_paf$join_text = clean_text(missed_paf$join_text)
missed_paf = subset(missed_paf, join_text != "")
stopifnot({nrow(subset(missed_paf, join_text == "")) == 0})

if(YEAR == 2020){
  mismatch_missed_auto = subset(auto, project_number=="2020 IVT 001")
  missed_paf[which(missed_paf$ProjectNumber=="2020 ivt 001"),]$join_text = mismatch_missed_auto$join_text
}

stopifnot({
  length(setdiff(unique(missed_paf$join_text), unique(auto$join_text))) == 0
})

if(nrow(missed_paf) > 0){
  stopifnot({
    length(setdiff(unique(missed_paf$join_text), unique(paf_auto$join_text))) != 0
  })
}


missed_paf_auto = subset(
  auto, (join_text %in% missed_paf$join_text) |
    (YEAR == 2021 & project_number=="21-RR-FAO-007")
)
if(nrow(missed_paf) > 0){
  missed_paf_auto$`PAF determination` = "Yes"
}

new_paf = subset(paf_auto, join_text %in% unique(paf_review$join_text))
old_paf = subset(paf_auto, !(join_text %in% unique(paf_review$join_text)))
table(new_paf$`PAF determination`)
table(old_paf$`PAF determination`)

old_paf$`PAF determination` = "Yes"

paf_review = subset(paf_review, Review == "PAF")
new_paf = subset(new_paf, join_text %in% unique(paf_review$join_text))

new_paf$`PAF determination` = "Yes"

paf = rbindlist(list(new_paf, old_paf, missed_paf_auto))
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
  "AA confidence ML"
)
paf[,drop_names] = NULL
writeData(wb, sheet="PAF", paf)

# Mark all old AA as Yes, drop AA if reviewed as not PAF
aa_review = fread(paste0("large_data/Reviewed_13June/new_", YEAR, "_paf.csv"))
missed_aa = fread(paste0("data/missed_", YEAR, "_aa.csv"))

aa_review = aa_review %>%
  unite(join_text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

aa_review$join_text = clean_text(aa_review$join_text)

if(YEAR == 2019){
  mismatch_auto = subset(paf_auto, crs_id == "2017000688")
  aa_review[which(aa_review$crs_id == "2017000688"),]$join_text = mismatch_auto$join_text
}

missed_aa = missed_aa %>%
  unite(join_text, all_of(c("ProjectTitle", "ShortDescription", "LongDescription")), sep=" ", na.rm=T, remove=F)

missed_aa$join_text = clean_text(missed_aa$join_text)
missed_aa = subset(missed_aa, join_text != "")

if(YEAR == 2020){
  mismatch_missed_auto = subset(auto, project_number=="2020 IVT 001")
  missed_aa[which(missed_aa$ProjectNumber=="2020 ivt 001"),]$join_text = mismatch_missed_auto$join_text
  mismatch_missed_auto = subset(auto, project_number=="20-RR-FAO-034")
  missed_aa[which(missed_aa$ProjectNumber=="20 rr fao 034"),]$join_text = mismatch_missed_auto$join_text
}

stopifnot({
  length(setdiff(unique(missed_aa$join_text), unique(auto$join_text))) == 0
})

if(nrow(missed_aa) > 0){
  stopifnot({
    length(setdiff(unique(missed_aa$join_text), unique(aa_auto$join_text))) != 0
  })
}


missed_aa_auto = subset(
  auto, 
  (join_text %in% missed_aa$join_text) |
    (YEAR == 2021 & project_number=="21-RR-FAO-007") # Blank text
)
if(nrow(missed_aa) > 0){
  missed_aa_auto$`AA determination` = "Yes"
}

new_aa = subset(aa_auto, join_text %in% unique(aa_review$join_text))
old_aa = subset(aa_auto, !(join_text %in% unique(aa_review$join_text)))
table(new_aa$`AA determination`)
table(old_aa$`AA determination`)

old_aa$`AA determination` = "Yes"

aa_review = subset(aa_review, Review == "PAF")
new_aa = subset(new_aa, join_text %in% unique(aa_review$join_text))


aa = rbindlist(list(new_aa, old_aa, missed_aa_auto))
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
aa[,drop_names] = NULL
writeData(wb, sheet="AA", aa)

saveWorkbook(wb, paste0("large_data/Full PAF dataset ",YEAR,".xlsx"), overwrite=T)

