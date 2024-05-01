list.of.packages <- c("data.table", "openxlsx", "tidyverse", "Hmisc")
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

### Join/process Crisis ####
crisis = crisis[,common_names]
crisis$crisis_label = "Crisis financing"
crs = merge(crs, crisis, by=common_names, all=T)

### Join/process PAF ####
paf$crisis_label = "Crisis financing"

paf$paf_label = "PAF"

paf$direct_label = NA
paf$direct_label[which(paf$Type.of.PAF=="Direct")] = "Direct"
paf$direct_label[which(paf$Type.of.PAF=="Indirect")] = "Indirect"
paf$direct_label[which(paf$Type.of.PAF=="Both")] = "Direct,Indirect"
paf$direct_label[which(paf$Type.of.PAF=="Part PAF")] = "Part"

paf$ddo_label = NA
paf$ddo_label[which(paf$WB.Cat.DDO=="Yes")] = "WB CAT DDO"

paf$cf_label = NA
paf$cf_label[which(paf$Contingent.financing=="Yes")] = "Contingent financing"

paf = paf[,c(common_names, "crisis_label", "paf_label", "direct_label", "ddo_label", "cf_label")]
crs = merge(crs, paf, by=common_names, all=T)
diff = subset(crs, crisis_label.x!=crisis_label.y)
stopifnot({nrow(diff)==0})
crs$crisis_label.x[which(is.na(crs$crisis_label.x))] = crs$crisis_label.y[which(is.na(crs$crisis_label.x))]
crs$crisis_label.y = NULL
setnames(crs, "crisis_label.x", "crisis_label")

### Join/process AA ####
aa$crisis_label = "Crisis financing"

aa$paf_label = "PAF"

aa$aa_label = "AA"

aa$direct_label = NA
aa$direct_label[which(aa$Type.oF.AF=="Direct")] = "Direct"
aa$direct_label[which(aa$Type.oF.AF=="Indirect")] = "Indirect"
aa$direct_label[which(aa$Type.oF.AF=="Both")] = "Direct,Indirect"
aa$direct_label[which(aa$Type.oF.AF=="Part PAF")] = "Part"

aa = aa[,c(common_names, "crisis_label", "paf_label", "aa_label", "direct_label")]
crs = merge(crs, aa, by=common_names, all=T)
diff = subset(crs, crisis_label.x!=crisis_label.y)
stopifnot({nrow(diff)==0})
diff = subset(crs, paf_label.x!=paf_label.y)
stopifnot({nrow(diff)==0})
diff = subset(crs, direct_label.x!=direct_label.y)
stopifnot({nrow(diff)==0})
crs$crisis_label.x[which(is.na(crs$crisis_label.x))] = crs$crisis_label.y[which(is.na(crs$crisis_label.x))]
crs$crisis_label.y = NULL
crs$paf_label.x[which(is.na(crs$paf_label.x))] = crs$paf_label.y[which(is.na(crs$paf_label.x))]
crs$paf_label.y = NULL
crs$direct_label.x[which(is.na(crs$direct_label.x))] = crs$direct_label.y[which(is.na(crs$direct_label.x))]
crs$direct_label.y = NULL
setnames(crs, "crisis_label.x", "crisis_label")
setnames(crs, "paf_label.x", "paf_label")
setnames(crs, "direct_label.x", "direct_label")

### Concatenate textual columns ####
textual_cols_for_classification = c(
  # "DonorName",
  "ProjectTitle",
  # "SectorName",
  # "PurposeName",
  # "FlowName",
  # "ChannelName",
  "ShortDescription",
  "LongDescription",
  "iati_text"
)
textual_cols_for_classification %in% names(crs)

crs = crs %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T)

### Make labels ####
label_cols = c(
  "crisis_label",
  "paf_label",
  "direct_label",
  "ddo_label",
  "cf_label",
  "aa_label"
)

crs = crs %>%
  unite(labels, all_of(label_cols), sep=",", na.rm=T)

crs$labels[which(crs$labels=="")] = "Unrelated"
crs = crs[,c("text","labels")]
crs = unique(crs)

fwrite(crs, "./large_data/meta_model_data_limited.csv")

### Examine prevalences ####
pos = subset(crs, labels!="Unrelated")
neg = subset(crs, labels=="Unrelated")
examine_set = rbind(
  pos,
  neg[c(1:nrow(pos)),]
)
unique_labels = unique(unlist(str_split(unique(examine_set$labels), pattern=",")))
for(unique_label in unique_labels){
  message(
    paste0(
      unique_label,
      ": ",
      sum(grepl(unique_label, examine_set$labels)),
      ", ",
      round(
        (sum(grepl(unique_label, examine_set$labels)) / nrow(examine_set)) * 100,
      digits=4),
      "%"
    )
  )
}

pos = subset(pos, labels!="Crisis financing")
View(pos)

### Examine lengths ####
context_window = 512
text_split = strsplit(crs$text, split=" ")
length(which(sapply(text_split, length)>context_window))/nrow(crs)
hist(sapply(text_split, length))
