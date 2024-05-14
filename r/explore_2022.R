list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/cdp-paf/"
setwd(wd)

crs = fread("large_data/crs_2022_predictions.csv")

skinny_cols = c(
  "donor_name",
  "sector_name",
  "purpose_name",
  "flow_name",
  "channel_name",
  "project_title",
  "short_description",
  "long_description",
  "Crisis finance predicted",
  "Crisis finance confidence",
  "PAF predicted",
  "PAF confidence",
  "AA predicted",
  "AA confidence",
  "Direct predicted",
  "Direct confidence",
  "Indirect predicted",
  "Indirect confidence",
  "Part predicted",
  "Part confidence"
)

crs = crs[,skinny_cols,with=F]

# Set blanks to false and 0
blanks = c("", "-")
blank_indices = which(crs$project_title %in% blanks & crs$short_description %in% blanks & crs$long_description %in% blanks)
crs$`Crisis finance confidence`[blank_indices] = 0
crs$`Crisis finance predicted`[blank_indices] = F
crs$`PAF confidence`[blank_indices] = 0
crs$`PAF predicted`[blank_indices] = F
crs$`AA confidence`[blank_indices] = 0
crs$`AA predicted`[blank_indices] = F
crs$`Direct confidence`[blank_indices] = 0
crs$`Direct predicted`[blank_indices] = F
crs$`Indirect confidence`[blank_indices] = 0
crs$`Indirect predicted`[blank_indices] = F
crs$`Part confidence`[blank_indices] = 0
crs$`Part predicted`[blank_indices] = F


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
mean(cf$`PAF predicted`) # 2.1%
paf = subset(cf, `PAF predicted`)
paf = paf[order(-paf$`PAF confidence`)]
notpaf = subset(cf, !`PAF predicted`)
mean(paf$`AA predicted`) # 9.1%
aa = subset(paf, `AA predicted`)
aa = aa[order(-aa$`AA confidence`)]
notaa = subset(paf, !`AA predicted`)

ggplot(crs) +
  geom_density(aes(x=`Crisis finance confidence`), color="black")
ggplot(cf) +
  geom_density(aes(x=`Crisis finance confidence`), color="black")
ggplot(notcf) +
  geom_density(aes(x=`Crisis finance confidence`), color="black")

ggplot(cf) +
  geom_density(aes(x=`PAF confidence`), color="black")
ggplot(paf) +
  geom_density(aes(x=`PAF confidence`), color="black")
ggplot(notpaf) +
  geom_density(aes(x=`PAF confidence`), color="black")

ggplot(paf) +
  geom_density(aes(x=`AA confidence`), color="black")
ggplot(aa) +
  geom_density(aes(x=`AA confidence`), color="black")
ggplot(notaa) +
  geom_density(aes(x=`AA confidence`), color="black")

keep= c("donor_name",
  "sector_name",
  "purpose_name",
  "flow_name",
  "channel_name",
  "project_title",
  "short_description",
  "long_description",
  "Crisis finance predicted",
  "Crisis finance confidence",
  "PAF predicted",
  "PAF confidence",
  "AA predicted",
  "AA confidence",
  "Direct predicted",
  "Direct confidence",
  "Indirect predicted",
  "Indirect confidence",
  "Part predicted",
  "Part confidence"
)

crs = crs[order(
  -crs$`AA predicted`,
  -crs$`PAF predicted`,
  -crs$`Crisis finance predicted`,
  -crs$`AA confidence`,
  -crs$`PAF confidence`,
  -crs$`Crisis finance confidence`
),keep, with=F]
fwrite(crs,
       "large_data/crs_2022_predictions_ordered.csv")

textual_cols_for_classification = c(
  "project_title",
  "short_description",
  "long_description"
)

crs = crs %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

cf_keywords = c(
  'preparedness', 'disaster', 'crisis', 'refugee', 'hazard', 'humanitaire', 'displaced', 'emergencies', 'recovery', 'urgence', 'migrants', 'hygiene', 'immediate', 'drought', 'crises', 'flood', 'crise', 'mine', 'réfugiés', 'internally', 'idps', 'shelter', 'saving', 'malnutrition', 'explosive', 'vulnérables', 'war', 'réponse', 'besoins', 'armed', 'disasters', 'rapid', 'camp', 'victims', 'risks', 'humanitaires', 'removal', 'displacement', 'mines', 'camps', 'acute', 'situations', 'earthquake', 'cyclone', 'rapidly', 'déplacées', 'communautés', 'warning', 'conflict', 'appeal'
)
cf_regex = paste0(
  "\\b",
  paste(cf_keywords, collapse="\\b|\\b"),
  "\\b"
)
paf_keywords = c(
  'insurance', 'ddo', 'cat', 'catastrophe', 'dref', 'deferred', 'contingent', 'drawdown', 'option', 'index','financed', 'quickly', 'weather', 'naturels', 'assurance','insuresilience', 'climatiques', 'instruments','landslide', 'innovations', 'pooling', 'parametric'
)
paf_regex = paste0(
  "\\b",
  paste(paf_keywords, collapse="\\b|\\b"),
  "\\b"
)
aa_keywords = c(
  'anticipatory', 'forecast', 'forecast-based', 'forecasts', 'désastre', 'forpac', 'probabilistic', 'prévisions', 'forecased', 'anticipation', 'rapidement', 'flexible', 'shock', 'predict'
)
aa_regex = paste0(
  "\\b",
  paste(aa_keywords, collapse="\\b|\\b"),
  "\\b"
)

crs$cf_keyword_search = grepl(cf_regex, crs$text, perl=T, ignore.case = T)
crs$paf_keyword_search = grepl(paf_regex, crs$text, perl=T, ignore.case = T)
crs$aa_keyword_search = grepl(aa_regex, crs$text, perl=T, ignore.case = T)
crs$text = NULL

predicted_cols = names(crs)[which(endsWith(names(crs), "predicted"))]
for(predicted_col in predicted_cols){
  crs[which(crs[,predicted_col]==F),predicted_col] = ""
}

search_cols = names(crs)[which(endsWith(names(crs), "search"))]
for(search_col in search_cols){
  crs[which(crs[,search_col]==F),search_col] = ""
}

fwrite(crs,
       "large_data/crs_2022_predictions_ordered_keyword_search.csv")
