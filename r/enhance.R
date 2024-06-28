list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/cdp-paf/"
setwd(wd)

# Additional data ####

wb_lending_groups = read.xlsx("additional_data/World Bank Country and Lending Groups.xlsx")
wb_lending_groups = wb_lending_groups[,c("Code", "Region", "Income.group")]
names(wb_lending_groups) = c("recipient_iso3_code","WB.Region","WB.Income.Group")

pop = fread("additional_data/Total population_World_Development_Indicators.csv", header=T, na.strings=c("",".."))
pop = melt(pop, id.vars="recipient_iso3_code", variable.name = "year", value.name = "population")

oecd_fcas = fread("additional_data/DAC-CRS-CODES_recipient.csv")

wb_fcas = read.xlsx("additional_data/FY24 List of Fragile and Conflict-affected Situations.xlsx")
wb_fcas = wb_fcas[,c("Code", "FCAS")]
names(wb_fcas) = c("recipient_iso3_code", "wb_fcas")

# Enhancement functions ####

quotemeta <- function(string) {
  str_replace_all(string, "(\\W)", "\\\\\\1")
}

remove_punct = function(string){
  str_replace_all(string, "[[:punct:]]", " ")
}

collapse_whitespace = function(string){
  str_replace_all(string, "\\s+", " ")
}

remove_zero_transactions = function(df){
  columns = c("usd_commitment","usd_disbursement")
  non_zero_indices = apply(cf[,columns], 1, sum, na.rm=T) > 0
  return(df[non_zero_indices,])
}

generate_donor_type = function(df){
  dt_bi_multi_dict = c(
    "1"="Government and EU institutions",
    "3"="Government and EU institutions",
    "4"="Multilateral (including UN) agencies and funds",
    "6"="Private donors",
    "7"="Government and EU institutions",
    "8"="Government and EU institutions"
  )
  donor_type = dt_bi_multi_dict[as.character(df$bilateral_multilateral)]
  mdbs = c(
    "African Development Fund",
    "African Development Bank",
    "Asian Development Bank", 
    "Caribbean Development Bank", 
    "Development Bank of Latin America", 
    "Inter-American Development Bank", 
    "International Bank for Reconstruction and Development",
    "International Development Association"
  )
  donor_type[which(
    df$donor_name %in% mdbs
  )] = "Multilateral development banks"
  return(donor_type)
}

enhance = function(df){
  df = remove_zero_transactions(df)

  df$donor_type = generate_donor_type(df)

  df = merge(df, wb_lending_groups, by="recipient_iso3_code", all.x = T)

  df = merge(df, pop, by=c("recipient_iso3_code", "year"), all.x=T)
  df$usd_commitment_per_capita = (df$usd_commitment * 1000000) / df$population
  df$usd_disbursement_per_capita = (df$usd_disbursement * 1000000) / df$population
  df$usd_commitment_deflated_per_capita = (df$usd_commitment_deflated * 1000000) / df$population
  df$usd_disbursement_deflated_per_capita = (df$usd_disbursement_deflated * 1000000) / df$population
  
  df = merge(df, oecd_fcas, by="recipient_iso3_code", all.x=T)
  df$oecd_fcas[which(is.na(df$oecd_fcas))] = 0
  
  df = merge(df, wb_fcas, by="recipient_iso3_code", all.x=T)
  df$wb_fcas[which(is.na(df$wb_fcas))] = 0
  return(df)
}

add_text_column = function(df){
  textual_cols_for_classification = c(
    "project_title",
    "short_description",
    "long_description"
  )
  
  df = df %>%
    unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)
  df$text = trimws(collapse_whitespace(remove_punct(tolower(df$text))))
  return(df)
}

generate_contingent_financing = function(df){
  contingent_financing = rep(NA, nrow(df))
  ibrd = df$donor_name == "International Bank for Reconstruction and Development"
  ida = df$donor_name == "International Development Association"
  adb = df$donor_name == "Asian Development Bank"
  iadb = df$donor_name == "Inter-American Development Bank"
  relevant_donor = ibrd | ida | adb | iadb
  
  ddo_keywords = c("ddo","deferred drawdown")
  dri_keywords = c("disaster resilience program", "disaster resilience improvement program")
  ccf_keywords = c("contingent loan")
  contingent_keywords = c(
    ddo_keywords, dri_keywords, ccf_keywords
  )
  contingent_keywords = quotemeta(contingent_keywords)
  contingent_regex = paste0(
    "\\b",
    paste(contingent_keywords, collapse="\\b|\\b"),
    "\\b"
  )
  contingent_match = grepl(contingent_regex, df$text, perl=T, ignore.case = T)
  contingent_financing[which(
    contingent_match & relevant_donor
  )] = 1

  return(contingent_financing)
}

generate_paf_imputed_share = function(df){
  paf_imputed_share = rep(1, nrow(df))
  sfera = grepl("\\bsfera\\b", df$text, perl=T, ignore.case=T)
  dref = grepl("\\bdref\\b", df$text, perl=T, ignore.case=T)
  
  paf_imputed_share[which(
    sfera & df$year == 2017
  )] = 0.104
  paf_imputed_share[which(
    sfera & df$year == 2018
  )] = 0.088
  paf_imputed_share[which(
    sfera & df$year == 2019
  )] = 0.313
  paf_imputed_share[which(
    sfera & df$year == 2020
  )] = 0.044
  paf_imputed_share[which(
    sfera & df$year == 2021
  )] = 0.288
  paf_imputed_share[which(
    sfera & df$year == 2022
  )] = 0.26
  
  paf_imputed_share[which(
    dref & df$year == 2017
  )] = 0
  paf_imputed_share[which(
    dref & df$year == 2018
  )] = 0
  paf_imputed_share[which(
    dref & df$year == 2019
  )] = 0.024
  paf_imputed_share[which(
    dref & df$year == 2020
  )] = 0.072
  paf_imputed_share[which(
    dref & df$year == 2021
  )] = 0.033
  paf_imputed_share[which(
    dref & df$year == 2022
  )] = 0.117

  return(paf_imputed_share)
}

paf_enhance = function(df){
  df = add_text_column(df)
  df$contingent_financing = generate_contingent_financing(df)
  df$paf_imputed_share = generate_paf_imputed_share(df)
  df$paf_usd_commitment_imputed = (df$usd_commitment * 1000000) * df$paf_imputed_share
  df$paf_usd_disbursement_imputed = (df$usd_disbursement * 1000000) * df$paf_imputed_share
  df$paf_usd_commitment_deflated_imputed = (df$usd_commitment_deflated * 1000000) * df$paf_imputed_share
  df$paf_usd_disbursement_deflated_imputed = (df$usd_disbursement_deflated * 1000000) * df$paf_imputed_share
  
  df$paf_usd_commitment_imputed_per_capita = df$usd_commitment_per_capita * df$paf_imputed_share
  df$paf_usd_disbursement_imputed_per_capita = df$usd_disbursement_per_capita * df$paf_imputed_share
  df$paf_usd_commitment_deflated_imputed_per_capita = df$usd_commitment_deflated_per_capita * df$paf_imputed_share
  df$paf_usd_disbursement_deflated_imputed_per_capita = df$usd_disbursement_deflated_per_capita * df$paf_imputed_share
  df$text = NULL
  return(df)
}

wb = createWorkbook("large_data/Full PAF dataset enhanced 2024.xlsx")

# CF ####
addWorksheet(wb, "Total crisis financing")
cf = read.xlsx(
  "large_data/Full PAF dataset 2024_27June.xlsx",
  sheet="Total crisis financing"
)
cf = enhance(cf)
writeData(wb, sheet="Total crisis financing", cf)

# PAF ####
addWorksheet(wb, "PAF")
paf = read.xlsx(
  "large_data/Full PAF dataset 2024_27June.xlsx",
  sheet="PAF"
)
not_paf = subset(paf, Classification=="Not PAF")
paf = subset(paf, Classification!="Not PAF")
paf = enhance(paf)
paf = paf_enhance(paf)
writeData(wb, sheet="PAF", paf)

# Humanitarian ####
addWorksheet(wb, "Humanitarian funding")
hum = read.xlsx(
  "large_data/Full PAF dataset 2024_27June.xlsx",
  sheet="Humanitarian funding"
)
hum = enhance(hum)
writeData(wb, sheet="Humanitarian funding", hum)

# AA ####
addWorksheet(wb, "AA")
aa = read.xlsx(
  "large_data/Full PAF dataset 2024_27June.xlsx",
  sheet="AA"
)
aa = subset(aa, !(crs_id %in% unique(not_paf$crs_id)))
aa = enhance(aa)
aa = paf_enhance(aa)
writeData(wb, sheet="AA", aa)

saveWorkbook(wb, "large_data/Full PAF dataset enhanced 2024.xlsx", overwrite=T)

