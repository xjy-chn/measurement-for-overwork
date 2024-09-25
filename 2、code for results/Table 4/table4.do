*
use "2012-2016panelCLDS.dta",clear
rename CITY 市
gen day=day(date)
replace month=month-1 if day<=15
replace year=year-1 if month==0
replace month=12 if month==0
replace 市="北京市" if  市=="110100"
replace 市="天津市" if  市=="120100"|市=="120200"
replace 市="上海市" if  市=="310100"
replace 市="重庆市" if  市=="500100"|市=="500200"

merge m:1 市 year month using  "超时加班月度加总_城市层面_企业数量加权.dta"
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
title("超时加班CLDS")
// reghdfe overwork_paid monthly_overwork ,absorb(市 year month )  keepsin
reghdfe overwork_duration monthly_overwork ,absorb(市 year)  keepsin
est store reg3
reghdfe lnduration monthly_overwork ,absorb(市 year  ) keepsin vce(cluster ID)
est store reg4
reghdfe lnduration monthly_overwork ,absorb(市 year )  keepsin vce(cluster ID)


reg overwork_dummy monthly_overwork,r 


*不加权
*
use "2012-2016panelCLDS.dta",clear
rename CITY 市
gen day=day(date)
replace month=month-1 if day<=15
replace year=year-1 if month==0
replace month=12 if month==0
replace 市="北京市" if  市=="110100"
replace 市="天津市" if  市=="120100"|市=="120200"
replace 市="上海市" if  市=="310100"
replace 市="重庆市" if  市=="500100"|市=="500200"

merge m:1 市 year month using  "uw超时加班月度加总_城市层面_企业数量加权.dta"
keep if _merge==3
xtset ID year
drop if overwork_dummy==2
winsor2 overwork_duration overwork_paid,cut(0 99) replace
gen lnduration=log( overwork_duration +1)
gen lnpaid=log( overwork_paid +1)
reg  overwork_dummy monthly_overwork,r 
est store reg1
reghdfe overwork_dummy monthly_overwork ,absorb(市 year  ) keepsin vce(cluster ID)
est store reg2
reghdfe overwork_duration monthly_overwork ,absorb(ID month  year  )    keepsin vce(cluster ID)