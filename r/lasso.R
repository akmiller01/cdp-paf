list.of.packages <- c("data.table", "openxlsx", "tidyverse", "Hmisc", "glmnet", "fastDummies")
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
limited_textual_cols_for_classification = c(
  "ProjectTitle",
  "ShortDescription",
  "LongDescription",
  "iati_text"
)
textual_cols_for_classification = c(
  "DonorName",
  "ProjectTitle",
  "SectorName",
  "PurposeName",
  "FlowName",
  "ShortDescription",
  "LongDescription",
  "iati_text"
)
textual_cols_for_classification %in% names(crs)

crs = crs %>%
  unite(limited_text, all_of(limited_textual_cols_for_classification), sep=" ", na.rm=T, remove=F) %>%
  unite(full_text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)
keep_for_xgboost = c(
  "full_text",
  "limited_text",
  "DonorName",
  "SectorName",
  "PurposeName",
  "FlowName",
  "ChannelName"
)
crs_xgboost = crs[,keep_for_xgboost]
fwrite(crs_xgboost, "large_data/meta_model_for_xgboost.csv")

clean_up_env = ls()
for(env_var in clean_up_env){
  if(!env_var %in% c("crs", "clean_up_env")){
    rm(list=env_var)
  }
}
rm(clean_up_env,  env_var)

predicted = fread("large_data/predicted_meta_model_data_combo.csv")
dat = merge(predicted, crs)
rm(crs, predicted)
gc()

# Just NLP
dat = subset(dat, `AA confidence` > 0)
fit = glm(`AA actual`~`AA confidence`, data=dat)
summary(fit)
1 - fit$deviance/fit$null.deviance

# NLP plus all
dat$DonorName[which(dat$DonorName=="")] = NA
dat$SectorName[which(dat$SectorName=="")] = NA
dat$PurposeName[which(dat$PurposeName=="")] = NA
dat$FlowName[which(dat$FlowName=="")] = NA
dat$ChannelName[which(dat$ChannelName=="")] = NA

dat$DonorName = factor(dat$DonorName)
dat$SectorName = factor(dat$SectorName)
dat$PurposeName = factor(dat$PurposeName)
dat$FlowName = factor(dat$FlowName)
dat$ChannelName = factor(dat$ChannelName)
fit = glm(
  `PAF actual`~`PAF confidence`+DonorName+SectorName+PurposeName+FlowName+ChannelName
  , data=dat)
summary(fit)
1 - fit$deviance/fit$null.deviance

dummies = dummy_cols(
  dat,
  select_columns = c(
    "DonorName"
    ,"SectorName"
    ,"PurposeName"
    ,"FlowName"
    ,"ChannelName"
  )
)
dummy_names = setdiff(names(dummies), names(dat))
dummies = subset(dummies, select=c(
  # 'AA confidence',
  dummy_names
  ))

y = dat$`AA actual` * 1
x = data.matrix(
  dummies
)
x[is.na(x)] = 0

#perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

#produce plot of test MSE by lambda value
plot(cv_model) 

#find coefficients of best model
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
best_model_coef = coef(best_model)
best_model_coef_df = data.frame(variable=dimnames(best_model_coef)[[1]], beta=0)
x_i = 1
for(i in attributes(best_model_coef)$i){
  best_model_coef_df$beta[i+1] = attributes(best_model_coef)$x[x_i]
  x_i = x_i + 1
}
best_model_coef_df = subset(best_model_coef_df, 
                            variable!="(Intercept)" &
                              variable!="AA confidence" &
                              beta!=0
                            )
best_model_coef_df$variable = gsub("_NA", "_nan", best_model_coef_df$variable, fixed=T)
View(best_model_coef_df)
fwrite(best_model_coef_df, "data/aa_coefficients.csv")
