use Figure7_part1.dta,clear
append using Figure7_part2.dta.dta
gen weight=1
append using  Figure8_part1.dta
append using  Figure8_part2.dta
replace weight=0 if weight==.
replace MEAN=MEAN*100 if weight==0
duplicates drop area date2 weight,force
fabplot line  MEAN date2 if weight==1, by(area)  front(connect) frontopts(mc(red) lc(red) xlabel(none) xtitle(date) ytitle("overwork(%)") ) 
addplot : , tline(20Feb2020, lp(dash)) norescaling 
graph save Figure7.gph,replace
graph export Figure7.svg,replace

fabplot line  MEAN date2 if weight==0, by(area)  front(connect) frontopts(mc(red) lc(red) xlabel(none) xtitle(date) ytitle("overwork(%)") ) 
addplot : , tline(20Feb2020, lp(dash)) norescaling 
graph save Figure8.gph,replace
graph export Figure8.svg,replace