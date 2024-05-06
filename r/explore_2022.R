list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/cdp-paf/"
setwd(wd)

most_confident <- function(vec1, vec2) {
  # Check if input vectors are of the same length
  if(length(vec1) != length(vec2)) {
    stop("Input vectors must be of the same length")
  }
  
  # Calculate distances to nearest integers for each element in both vectors
  dist_vec1 <- abs(vec1 - round(vec1))
  dist_vec2 <- abs(vec2 - round(vec2))
  
  # Select the most confident number from each pair of elements
  output <- ifelse(dist_vec1 < dist_vec2, vec1, vec2)
  
  return(output)
}

crs = fread("large_data/crs_2022_predictions_xgb.csv")

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
  "Crisis finance predicted XGB",
  "Crisis finance confidence XGB",
  "PAF predicted XGB",
  "PAF confidence XGB",
  "AA predicted XGB",
  "AA confidence XGB"
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
crs$`Crisis finance confidence XGB`[blank_indices] = 0
crs$`Crisis finance predicted XGB`[blank_indices] = F
crs$`PAF confidence XGB`[blank_indices] = 0
crs$`PAF predicted XGB`[blank_indices] = F
crs$`AA confidence XGB`[blank_indices] = 0
crs$`AA predicted XGB`[blank_indices] = F

# Use more confident to create joint indicator
crs$`Crisis finance confidence joint` = most_confident(
  crs$`Crisis finance confidence`,
  crs$`Crisis finance confidence XGB`
)
crs$`Crisis finance predicted joint` = crs$`Crisis finance confidence joint` >= 0.5

crs$`PAF confidence joint` = most_confident(
  crs$`PAF confidence`,
  crs$`PAF confidence XGB`
)
crs$`PAF predicted joint` = crs$`PAF confidence joint` >= 0.5

crs$`AA confidence joint` = most_confident(
  crs$`AA confidence`,
  crs$`AA confidence XGB`
)
crs$`AA predicted joint` = crs$`AA confidence joint` >= 0.5

# examine xgb diffs
cf_diff = subset(crs, `Crisis finance predicted`!=`Crisis finance predicted XGB`)
paf_diff = subset(crs, `PAF predicted`!=`PAF predicted XGB`)
aa_diff = subset(crs, `AA predicted`!=`AA predicted XGB`)

# Set PAF confidence equal to AA confidence if AA predicted and PAF not
crs$`PAF confidence joint`[which(crs$`AA predicted joint` & !crs$`PAF predicted joint`)] =
  crs$`AA confidence joint`[which(crs$`AA predicted joint` & !crs$`PAF predicted joint`)]
crs$`PAF predicted joint`[which(crs$`AA predicted joint`)] = T

# Set CF confidence equal to PAF confidence if PAF predicted and CF not
crs$`Crisis finance confidence joint`[which(crs$`PAF predicted joint` & !crs$`Crisis finance predicted joint`)] =
  crs$`PAF confidence joint`[which(crs$`PAF predicted joint` & !crs$`Crisis finance predicted joint`)]
crs$`Crisis finance predicted joint`[which(crs$`PAF predicted joint`)] = T


# Change thresholds
# crs$`AA predicted`[which(crs$`AA confidence`<0.75)] = F
# crs$`PAF predicted`[which(crs$`PAF confidence`<0.75)] = F
# crs$`Crisis finance predicted`[which(crs$`Crisis finance confidence`<0.75)] = F

# Allow a majority vote by PAF and AA to overrule CF
# crs$`Crisis finance confidence`[which(crs$`PAF predicted` & crs$`AA predicted`)] =
#   crs$`PAF confidence`[which(crs$`PAF predicted` & crs$`AA predicted`)]
# crs$`Crisis finance predicted`[which(crs$`PAF predicted` & crs$`AA predicted`)] = T

# Set PAF confidence equal to CF confidence if PAF predicted and CF not
# crs$`PAF confidence`[which(crs$`PAF predicted` & !crs$`Crisis finance predicted`)] =
#   crs$`Crisis finance confidence`[which(crs$`PAF predicted` & !crs$`Crisis finance predicted`)]
# crs$`PAF predicted`[which(!crs$`Crisis finance predicted`)] = F

# Set AA confidence equal to PAF confidence if AA predicted and PAF not
# crs$`AA confidence`[which(crs$`AA predicted` & !crs$`PAF predicted`)] =
#   crs$`PAF confidence`[which(crs$`AA predicted` & !crs$`PAF predicted`)]
# crs$`AA predicted`[which(!crs$`PAF predicted`)] = F

# From source data, 23.6% of CRS is Crisis financing
# 1.2% of crisis financing is PAF
# 22.5% of PAF is AA
mean(crs$`Crisis finance predicted joint`) # 26.5%
cf = subset(crs, `Crisis finance predicted joint`)
cf = cf[order(-cf$`Crisis finance confidence joint`)]
notcf = subset(crs, !`Crisis finance predicted joint`)
mean(cf$`PAF predicted joint`) # 1.37%
paf = subset(cf, `PAF predicted joint`)
paf = paf[order(-paf$`PAF confidence joint`)]
notpaf = subset(cf, !`PAF predicted joint`)
mean(paf$`AA predicted joint`) # 13.9%
aa = subset(paf, `AA predicted joint`)
aa = aa[order(-aa$`AA confidence joint`)]
notaa = subset(paf, !`AA predicted joint`)

ggplot(crs) +
  geom_density(aes(x=`Crisis finance confidence`), color="black") +
  geom_density(aes(x=`Crisis finance confidence XGB`), color="blue") +
  geom_density(aes(x=`Crisis finance confidence joint`), color="red")
ggplot(cf) +
  geom_density(aes(x=`Crisis finance confidence`), color="black") +
  geom_density(aes(x=`Crisis finance confidence XGB`), color="blue") +
  geom_density(aes(x=`Crisis finance confidence joint`), color="red")
ggplot(notcf) +
  geom_density(aes(x=`Crisis finance confidence`), color="black") +
  geom_density(aes(x=`Crisis finance confidence XGB`), color="blue") +
  geom_density(aes(x=`Crisis finance confidence joint`), color="red")

ggplot(cf) +
  geom_density(aes(x=`PAF confidence`), color="black") +
  geom_density(aes(x=`PAF confidence XGB`), color="blue") +
  geom_density(aes(x=`PAF confidence joint`), color="red")
ggplot(paf) +
  geom_density(aes(x=`PAF confidence`), color="black") +
  geom_density(aes(x=`PAF confidence XGB`), color="blue") +
  geom_density(aes(x=`PAF confidence joint`), color="red")
ggplot(notpaf) +
  geom_density(aes(x=`PAF confidence`), color="black") +
  geom_density(aes(x=`PAF confidence XGB`), color="blue") +
  geom_density(aes(x=`PAF confidence joint`), color="red")

ggplot(paf) +
  geom_density(aes(x=`AA confidence`), color="black") +
  geom_density(aes(x=`AA confidence XGB`), color="blue") +
  geom_density(aes(x=`AA confidence joint`), color="red")
ggplot(aa) +
  geom_density(aes(x=`AA confidence`), color="black") +
  geom_density(aes(x=`AA confidence XGB`), color="blue") +
  geom_density(aes(x=`AA confidence joint`), color="red")
ggplot(notaa) +
  geom_density(aes(x=`AA confidence`), color="black") +
  geom_density(aes(x=`AA confidence XGB`), color="blue") +
  geom_density(aes(x=`AA confidence joint`), color="red")

keep= c("donor_name",
  "sector_name",
  "purpose_name",
  "flow_name",
  "channel_name",
  "project_title",
  "short_description",
  "long_description",
  "Crisis finance predicted joint",
  "Crisis finance confidence joint",
  "PAF predicted joint",
  "PAF confidence joint",
  "AA predicted joint",
  "AA confidence joint"
)

crs = crs[order(
  -crs$`AA predicted joint`,
  -crs$`PAF predicted joint`,
  -crs$`Crisis finance predicted joint`,
  -crs$`AA confidence joint`,
  -crs$`PAF confidence joint`,
  -crs$`Crisis finance confidence joint`
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
  'anticipatory', 'forecasts', 'désastre', 'forpac', 'probabilistic', 'prévisions', 'forecased', 'anticipation', 'rapidement', 'flexible', 'shock', 'predict'
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
fwrite(crs,
       "large_data/crs_2022_predictions_ordered_keyword_search.csv")
