list.of.packages <- c("data.table", "openxlsx", "Hmisc", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

YEAR = 2017

wd = "~/git/cdp-paf/"
setwd(wd)

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

fix_pn = function(project_numbers){
  collapse_whitespace(trimws(remove_punct(tolower(conv_ascii(remove_lead_zero_if_number(trimws(project_numbers)))))))
}

auto = fread(paste0("large_data/crs_",YEAR,"_cdp_automated.csv"))
auto$project_number = fix_pn(auto$project_number)
all_pn = unique(auto$project_number)

paf_auto = subset(auto, `PAF determination` %in% c("Yes", "Review"))
aa_auto = subset(auto, `AA determination` %in% c("Yes", "Review"))
cf_auto = subset(auto, `Crisis finance determination` %in% c("Yes", "Review"))

paf = read.xlsx("large_data/Full PAF dataset.xlsx", sheet="PAF", na.strings = c("", "#N/A"))
paf = subset(paf, Year==YEAR)
paf$ProjectNumber = fix_pn(paf$ProjectNumber)
cf = read.xlsx("large_data/Full PAF dataset.xlsx", sheet="Total crisis financing", na.strings = c("", "#N/A"))
cf = subset(cf, Year==YEAR)
cf$ProjectNumber = fix_pn(cf$ProjectNumber)
aa = read.xlsx("large_data/Full PAF dataset.xlsx", sheet="Funding for AA", na.strings = c("", "#N/A"))
aa = subset(aa, Year==YEAR)
aa$ProjectNumber = fix_pn(aa$ProjectNumber)

# AA
aa_pn = unique(aa$ProjectNumber)
aa_pn = subset(aa_pn, aa_pn %in% all_pn)
aa_auto_pn = unique(aa_auto$project_number)
missed_aa_pn = setdiff(aa_pn, aa_auto_pn)
new_aa_pn = setdiff(aa_auto_pn, aa_pn)

length(missed_aa_pn) / length(aa_pn) 
missed_aa = subset(aa, ProjectNumber %in% missed_aa_pn)
new_aa = subset(aa_auto, project_number %in% new_aa_pn)
new_aa = new_aa[order(-new_aa$`AA confidence ML`),c("crs_id","project_title","short_description","long_description","purpose_name","AA determination","AA confidence ML")]
describe(new_aa$`AA determination`)
fwrite(missed_aa, paste0("data/missed_",YEAR,"_aa.csv"))
fwrite(new_aa, paste0("data/new_",YEAR,"_aa.csv"))

# PAF
paf_pn = unique(paf$ProjectNumber)
paf_pn = subset(paf_pn, paf_pn %in% all_pn)
paf_auto_pn = unique(paf_auto$project_number)
missed_paf_pn = setdiff(paf_pn, paf_auto_pn)
missed_paf_pn = subset(missed_paf_pn, !is.na(missed_paf_pn))
new_paf_pn = setdiff(paf_auto_pn, paf_pn)
new_paf_pn = subset(new_paf_pn, !is.na(new_paf_pn))

length(missed_paf_pn) / length(paf_pn) 
missed_paf = subset(paf, ProjectNumber %in% missed_paf_pn)
new_paf = subset(paf_auto, project_number %in% new_paf_pn)
new_paf = new_paf[order(-new_paf$`PAF confidence ML`),c("crs_id","project_title","short_description","long_description","purpose_name","PAF determination","PAF confidence ML")]
describe(new_paf$`PAF determination`)
fwrite(missed_paf, paste0("data/missed_",YEAR,"_paf.csv"))
fwrite(new_paf, paste0("data/new_",YEAR,"_paf.csv"))

# CF
cf_pn = unique(cf$ProjectNumber)
cf_pn = subset(cf_pn, cf_pn %in% all_pn)
cf_auto_pn = unique(cf_auto$project_number)
missed_cf_pn = setdiff(cf_pn, cf_auto_pn)
missed_cf_pn = subset(missed_cf_pn, !is.na(missed_cf_pn))
new_cf_pn = setdiff(cf_auto_pn, cf_pn)
new_cf_pn = subset(new_cf_pn, !is.na(new_cf_pn))

length(missed_cf_pn) / length(cf_pn) 
missed_cf = subset(cf, ProjectNumber %in% missed_cf_pn)
new_cf = subset(cf_auto, project_number %in% new_cf_pn)
new_cf = new_cf[order(-new_cf$`Crisis finance confidence ML`),c("crs_id","project_title","short_description","long_description","purpose_name","Crisis finance determination","Crisis finance confidence ML")]
describe(new_cf$`Crisis finance determination`)
fwrite(missed_cf, paste0("data/missed_",YEAR,"_cf.csv"))
fwrite(new_cf, paste0("data/new_",YEAR,"_cf.csv"))
