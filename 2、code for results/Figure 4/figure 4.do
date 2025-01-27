global figure output\figure
global table output\table

import excel "citycode-cityname.xlsx",clear firstrow
save citycode-cityname.dta,replace


use "ratio/dummy12_20.dta",clear


*merge firmnum in cities
merge m:1 citycode year using "firmnum/firms12_20.dta"
drop if citycode==710000 // enterprises data in Taiwan, a province of China is not available for us
drop _merge

bysort year isweight  :egen mean_intensity=mean(MEAN)

twoway (scatter  MEAN firm   if isweight==0) ///
(lfit MEAN firm  if isweight==0,color("red") lpattern(solid)) ///
(qfit MEAN firm   if isweight==0,color("blue") lpattern(dash)),title(raw) xtitle(number of firms) ytitle("overwork(%)") legend(label(1 ratio of overwork days) label(2 linear fit) label(3 quardratic fit)) note("(a)",pos(6) size(middle)) xsize(50) ysize(40) xlabel(,labsize(small)) ylabel(,labsize(small))
graph save $figure\Fig4_panel_a.gph,replace
graph export $figure\Fig4_panel_a.eps,replace


preserve
winsor2 MEAN firm if isweight==0,cut(1 99) replace
twoway (scatter  MEAN firm   if isweight==0) ///
(lfit MEAN firm   if isweight==0,color("red") lpattern(solid)) ///
(qfit MEAN firm   if isweight==0,color("blue") lpattern(dash)),title(winsorized) xtitle(number of firms) ytitle("overwork(%)") legend(label(1 ratio of overwork days) label(2 linear fit) label(3 quardratic fit)) note("(b)",pos(6) size(middle))  xsize(50) ysize(40) xlabel(,labsize(small)) ylabel(,labsize(small))
graph save $figure\Fig4_panel_b.gph,replace
graph export $figure\Fig4_panel_b.eps,replace
restore
 grc1leg  "$figure\Fig4_panel_a.gph"  "$figure\Fig4_panel_b.gph" ,xsize(50)  ysize(20) position(6) iscale(0.8 0.8)
graph save "$figure\Fig4.gph",replace
graph export "$figure\Fig4.eps",replace

