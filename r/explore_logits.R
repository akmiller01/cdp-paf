list.of.packages <- c("data.table")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/cdp-paf/"
setwd(wd)

quick_summ_stat = function(vector){
  p5 = quantile(vector, probs=0.05, na.rm=T)
  p10 = quantile(vector, probs=0.1, na.rm=T)
  p90 = quantile(vector, probs=0.9, na.rm=T)
  p95 = quantile(vector, probs=0.95, na.rm=T)
  s_mean = mean(vector, na.rm=T)
  s_median = median(vector, na.rm=T)
  s_min = min(vector, na.rm=T)
  s_max = max(vector, na.rm=T)
  message(
    paste0(
      "Min: ",
      s_min,
      "\nP5: ",
      p5,
      "\nP10: ",
      p10,
      "\nMean: ",
      s_mean,
      "\nMedian: ",
      s_median,
      "\nP90: ",
      p90,
      "\nP95: ",
      p95,
      "\nMax: ",
      s_max
    )
  )
}

dat = fread("large_data/predicted_meta_model_data_combo.csv")

quick_summ_stat(dat$`Crisis finance confidence`)
quick_summ_stat(dat$`PAF confidence`)
quick_summ_stat(dat$`AA confidence`)

# Cut-off of ~0.6 is free
cf_true = subset(dat, `Crisis finance actual`)
quick_summ_stat(cf_true$`Crisis finance confidence`)
# Min 0.63
cf_false_positive = subset(dat, !`Crisis finance actual` & `Crisis finance predicted`)
quick_summ_stat(cf_false_positive$`Crisis finance confidence`)
# Min 0.52
cf_false_negative = subset(dat, `Crisis finance actual` & !`Crisis finance predicted`)
quick_summ_stat(cf_false_negative$`Crisis finance confidence`)

# Cut-off of ~0.6 is free, 0.85 should lose 5% of true in exchange for mean of false
paf_true = subset(dat, `PAF actual`)
quick_summ_stat(paf_true$`PAF confidence`)
# Min 0.63
paf_false_positive = subset(dat, !`PAF actual` & `PAF predicted`)
quick_summ_stat(paf_false_positive$`PAF confidence`)
# Min 0.50
paf_false_negative = subset(dat, `PAF actual` & !`PAF predicted`)
quick_summ_stat(paf_false_negative$`PAF confidence`)

# 0.60 is free.
aa_true = subset(dat, `AA actual`)
quick_summ_stat(aa_true$`AA confidence`)
# Min 0.67
aa_false_positive = subset(dat, !`AA actual` & `AA predicted`)
quick_summ_stat(aa_false_positive$`AA confidence`)
# Min 0.50
aa_false_negative = subset(dat, `AA actual` & !`AA predicted`)
quick_summ_stat(aa_false_negative$`AA confidence`)
