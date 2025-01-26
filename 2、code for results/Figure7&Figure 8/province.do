*use province daily zonal data
forvalues i=2012(1)2020{
	import excel `i'statistics.xlsx,firstrow clear
	save `i'.dta,replace
}
use 2012.dta,clear
forvalues i=2013(1)2020{
	append using  `i'.dta
}
save 2012-2020city_zonal.dta,replace
forvalues i=2012(1)2020{
	erase `i'.dta
}

import excel dayofweek.xlsx,clear firstrow 
save dayofweek.dta,replace

import excel citycode-cityname.xlsx,clear firstrow 
save citycode-cityname.dta,replace

preserve
use 2012-2020city_zonal.dta,clear
destring dayOfYear,replace
merge m:1 year dayOfYear using   dayofweek.dta,nogen keep(1 3)
merge m:1 citycode using citycode-cityname.dta,nogen keep(1 3)
keep if  type==1
gen date2=date(date,"YMD")
gen date3=date(date,"YMD")
format date2 %td
xtset citycode date2


keep if year==2020
keep if prov=="Shanghai"|cityname=="XiAn"|cityname=="Ningbo"|cityname=="Beihai"
gen area=prov
bysort area date2:egen mean2=mean(MEAN)
duplicates drop area date2,force
replace area=city if cityname=="XiAn"|cityname=="Ningbo"|cityname=="Beihai"
replace area="(a) Shanghai" if area=="Shanghai"
replace area="(b) Neimenggu" if area=="Neimenggu"
replace area="(c) Beihai" if area=="Beihai"
replace area="(d) Jilin" if area=="Jiling"
replace area="(e) Sichuan" if area=="Sichuan"
replace area="(f) Ningbo" if area=="Ningbo"
replace area="(g) Shandong" if area=="Shandong"
replace area="(h) Guangdong" if area=="Guangdong"
replace area="(i) Jiangxi" if area=="Jiangxi"
replace area="(j) Hubei" if area=="Hubei"
replace area="(k) Xian" if area=="XiAn"

save "Figure7_part2",replace
restore
