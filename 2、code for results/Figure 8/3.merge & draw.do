use w劳动争议元化解试点城市数据.dta,clear
append using w劳动争议元化解试点省份数据.dta
gen weight=1
append using 劳动争议元化解试点省份数据.dta
append using 劳动争议元化解试点城市数据.dta
replace weight=0 if weight==.
replace MEAN=MEAN*100 if weight==0
duplicates drop 区域 date2 weight,force
fabplot line  MEAN date2 if weight==1, by(区域)  front(connect) frontopts(mc(red) lc(red) xlabel(none) xtitle(date) ytitle("overwork(%)") ) 
addplot : , tline(20Feb2020, lp(dash)) norescaling 
graph save 加权劳动争议多元试点.gph,replace
graph export 加权劳动争议多元试点.svg,replace

fabplot line  MEAN date2 if weight==0, by(区域)  front(connect) frontopts(mc(red) lc(red) xlabel(none) xtitle(date) ytitle("overwork(%)") ) 
addplot : , tline(20Feb2020, lp(dash)) norescaling 
graph save 劳动争议多元试点.gph,replace
graph export 劳动争议多元试点.svg,replace