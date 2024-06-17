list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr", "openxlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/cdp-paf/"
setwd(wd)

wb = createWorkbook("large_data/Full PAF dataset 2024.xlsx")

years = c(2017:2022)
sheets = c("Total crisis financing", "PAF", "Humanitarian funding", "AA")
max = length(years) * length(sheets)
pb = txtProgressBar(max=max, style=3)
progress = 0
for(sheet in sheets){
  addWorksheet(wb, sheet)
  d_l = list()
  d_i = 1
  for(year in years){
    progress = progress + 1
    setTxtProgressBar(pb, progress)
    tmp = read.xlsx(
      paste0("large_data/Full PAF dataset ",year,".xlsx"),
      sheet=sheet
    )
    d_l[[d_i]] = tmp
    d_i = d_i + 1
  }
  dat = rbindlist(d_l)
  writeData(wb, sheet=sheet, dat)
}
close(pb)

saveWorkbook(wb, "large_data/Full PAF dataset 2024.xlsx", overwrite=T)