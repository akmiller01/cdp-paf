list.of.packages <- c("data.table", "openxlsx", "tidyverse")
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
aa = read.xlsx("large_data/Full PAF dataset.xlsx", sheet="Funding for AA", na.strings = c("", "#N/A"))
aa = aa[which(!duplicated(aa)),]
hum = read.xlsx("large_data/Full PAF dataset.xlsx", sheet="Humanitarian funding", na.strings = c("", "#N/A"))
hum = hum[which(!duplicated(hum)),]

### Exploration ####
common_names = names(crisis)
paf_limited = paf[,common_names]
aa_limited = aa[,common_names]
# Overlap between PAF and crisis = 921, out of 1428 PAF
sum(duplicated(rbind(paf_limited, crisis)))
nrow(paf)
nrow(crisis)
# Overlap between PAF and hum = 406 out of 1428 PAF
sum(duplicated(rbind(paf_limited, hum)))
# Overlap between PAF and AA = 247, out of 248 AA
sum(duplicated(rbind(paf_limited, aa_limited)))
nrow(aa)

# Overlap between AA and crisis = 190, out of 248 AA
sum(duplicated(rbind(aa_limited, crisis)))
# Overlap between AA and hum = 177, out of 248 AA
sum(duplicated(rbind(aa_limited, hum)))

### Build unrelated, unique CRS activities ####
related = unique(rbind(paf_limited, aa_limited))
unrelated = unique(rbind(crisis, hum))
related$related = T
unrelated$related = F
sum(duplicated(rbindlist(list(related, unrelated)), by=common_names)) # 921
joint = rbindlist(list(related, unrelated))
joint$duplicated = duplicated(joint, by=common_names)
sense_check = subset(joint, related & duplicated)
stopifnot({nrow(sense_check) == 0})
unique_unrelated = subset(joint, !related & !duplicated)
unique_unrelated$duplicated = NULL
stopifnot(
  {
    sum(duplicated(rbindlist(list(related, unique_unrelated)), by=common_names)) == 0
  }
)
unique_unrelated$related = NULL

### Build related ####
paf_limited$paf = T
aa_limited$paf = F
paf_and_or_aa = rbindlist(list(paf_limited, aa_limited))
paf_and_or_aa$duplicated = duplicated(paf_and_or_aa, by=common_names)
aa_not_paf = subset(paf_and_or_aa, !paf & !duplicated)
aa_and_or_paf = rbindlist(list(aa_limited, paf_limited))
aa_and_or_paf$duplicated = duplicated(aa_and_or_paf, by=common_names)
paf_not_aa = subset(aa_and_or_paf, paf & !duplicated)
paf_and_aa = subset(paf_and_or_aa, duplicated)
stopifnot({
  nrow(aa_not_paf) + nrow(paf_not_aa) + 2 * nrow(paf_and_aa) == nrow(paf_and_or_aa)
})
aa_not_paf$label = "AA"
paf_not_aa$label = "PAF"
paf_and_aa$label = "AA,PAF"
labeled_related = rbindlist(
  list(aa_not_paf, paf_not_aa, paf_and_aa)
)

### Concatenate textual columns ####
textual_cols_for_classification = c(
  "ProjectTitle",
  "SectorName",
  "PurposeName",
  "FlowName",
  "ShortDescription",
  "LongDescription",
  "DonorName",
  "AgencyName",
  "RecipientName",
  "ChannelName",
  "ChannelReportedName"
)

unique_unrelated = unique_unrelated %>%
  unite(text, textual_cols_for_classification, sep=" ", na.rm=T)
labeled_related = labeled_related %>%
  unite(text, textual_cols_for_classification, sep=" ", na.rm=T)

### Make labels ####
unique_unrelated$label = "Unrelated"
unique_unrelated = unique_unrelated[,c("text", "label")]
labeled_related = labeled_related[,c("text", "label")]

# TODO: Some additional word searches on unique_unrelated to filter out
# those maybe missed by CDP

### Combine and export ####
meta_model_data = rbindlist(list(
  labeled_related, unique_unrelated
))
# Post-rectify to binary
meta_model_data$label[which(meta_model_data$label %in% c("AA", "PAF", "AA,PAF"))] = "PAF/AA"
fwrite(meta_model_data, "./large_data/meta_model_data.csv")

### Examine lengths ####
context_window = 512
pos = subset(meta_model_data, label=="PAF/AA")
text_split = strsplit(pos$text, split=" ")
length(which(sapply(text_split, length)>context_window))/nrow(pos)
hist(sapply(text_split, length))
