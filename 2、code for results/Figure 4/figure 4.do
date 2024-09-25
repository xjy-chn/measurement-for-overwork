global figure 结果\figure
global table 结果\table

import excel "城市名-行政编码.xlsx",clear firstrow
save 城市名称编码对照表.dta,replace

*合并强度数据
use "不去掉居民区+加班天数占比/dummy12_20.dta",clear
append using "去掉居民区+加班天数占比/DR_dummy12_20.dta"

rename SUM 超时加班强度总和
*合并企业数量
merge m:1 市代码 year using "城市企业数量/firms12_20.dta"
drop if 市代码==710000 //缺少台湾的数据
drop _merge
rename SUM 城市企业存量
// winsor2 MEAN 城市企业存量,cut(1 99) replace
bysort year 是否使用企业数量加权 是否去掉居民区 :egen mean_intensity=mean(MEAN)

twoway (scatter  MEAN 城市企业存量   if 是否使用企业数量加权==0&是否去掉居民区==0) ///
(lfit MEAN 城市企业存量   if 是否使用企业数量加权==0&是否去掉居民区==0,color("red") lpattern(solid)) ///
(qfit MEAN 城市企业存量   if 是否使用企业数量加权==0&是否去掉居民区==0,color("blue") lpattern(dash)),title(raw) xtitle(number of firms) ytitle("overwork(%)") legend(label(1 ratio of overwork days) label(2 linear fit) label(3 quardratic fit)) note("(a)",pos(6) size(middle)) xsize(50) ysize(40) xlabel(,labsize(small)) ylabel(,labsize(small))
graph save $figure\城市层面加班天数占比与企业数量.gph,replace
graph export $figure\城市层面加班天数占比与企业数量.svg,replace
graph export $figure\城市层面加班天数占比与企业数量.jpg,replace

*缩尾
preserve
winsor2 MEAN 城市企业存量,cut(1 99) replace
twoway (scatter  MEAN 城市企业存量   if 是否使用企业数量加权==0&是否去掉居民区==0) ///
(lfit MEAN 城市企业存量   if 是否使用企业数量加权==0&是否去掉居民区==0,color("red") lpattern(solid)) ///
(qfit MEAN 城市企业存量   if 是否使用企业数量加权==0&是否去掉居民区==0,color("blue") lpattern(dash)),title(winsorized) xtitle(number of firms) ytitle("overwork(%)") legend(label(1 ratio of overwork days) label(2 linear fit) label(3 quardratic fit)) note("(b)",pos(6) size(middle))  xsize(50) ysize(40) xlabel(,labsize(small)) ylabel(,labsize(small))
graph save $figure\城市层面加班天数占比与企业数量_缩尾.gph,replace
graph export $figure\城市层面加班天数占比与企业数量_缩尾.svg,replace
graph export $figure\城市层面加班天数占比与企业数量_缩尾.jpg,replace
restore
 grc1leg  "$figure\城市层面加班天数占比与企业数量.gph"  "$figure\城市层面加班天数占比与企业数量_缩尾.gph" ,xsize(50)  ysize(20) position(6) iscale(0.8 0.8)
graph save "$figure\城市加班与企业数量.gph",replace
graph export "$figure\城市加班与企业数量.svg",replace
