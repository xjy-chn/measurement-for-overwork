*
use "2012-2016panelCLDS.dta",clear

gen day=day(date)
replace month=month-1 if day<=15
replace year=year-1 if month==0
replace month=12 if month==0

merge m:1 citycode year month using  "monthly overwork.dta"
keep if _merge==3
xtset ID year
drop if overwork_dummy==2
winsor2 overwork_duration overwork_paid,cut(0 99) replace
gen lnduration=log( overwork_duration +1)
gen lnpaid=log( overwork_paid +1)
reg  overwork_dummy monthly_overwork,r 
est store reg1
reghdfe overwork_dummy monthly_overwork ,absorb( year  ) keepsin vce(cluster ID)
est store reg2
reghdfe overwork_duration monthly_overwork ,absorb(ID month  year date )    keepsin vce(cluster ID)
reg2docx  reg1 reg2  using CLDS-crossValidation.docx,replace ///
scalars(N r2_a(%11.3f)) b(%11.3f) t(%11.3f) ///
title("CLDS")
