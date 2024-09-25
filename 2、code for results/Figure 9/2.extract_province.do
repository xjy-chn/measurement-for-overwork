forvalues i=2012(1)2020{
	import excel `i'年度汇总.xlsx,firstrow clear
	save `i'.dta,replace
}
use 2012.dta,clear
forvalues i=2013(1)2020{
	append using  `i'.dta
}
save 2012-2020日度省份层面加班情况.dta,replace
forvalues i=2012(1)2020{
	erase `i'.dta
}

import excel 天数日期对照表.xlsx,clear firstrow 
save 天数日期对照表.dta,replace

import excel 城市名-行政编码.xlsx,clear firstrow 
keep 省代码 省
duplicates drop 省代码,force
duplicates report
save 省份名称编码对照表.dta,replace

preserve
use 2012-2020日度省份层面加班情况.dta,clear
destring dayOfYear,replace
merge m:1 year dayOfYear using  天数日期对照表.dta,nogen keep(1 3)
merge m:1 省代码 using 省份名称编码对照表.dta,nogen keep(1 3)
gen date2=date(date,"YMD")
gen date3=date(date,"YMD")
format date2 %td
xtset 省代码 date2


keep if year==2020
keep if 省=="内蒙古自治区"|省=="吉林省"|省=="江西省"|省=="山东省"|省=="湖北省"|省=="广东省"|省=="四川省"
gen 区域=省
bysort 区域 date2:egen mean2=mean(MEAN)
duplicates drop 区域 date2,force

replace 区域="(a) Shanghai" if 区域=="上海市"
replace 区域="(b) Neimenggu" if 区域=="内蒙古自治区"
replace 区域="(c) Beihai" if 区域=="北海市"
replace 区域="(d) Jilin" if 区域=="吉林省"
replace 区域="(e) Sichuan" if 区域=="四川省"
replace 区域="(f) Ningbo" if 区域=="宁波市"
replace 区域="(g) Shandong" if 区域=="山东省"
replace 区域="(h) Guangdong" if 区域=="广东省"
replace 区域="(i) Jiangxi" if 区域=="江西省"
replace 区域="(j) Hubei" if 区域=="湖北省"
replace 区域="(k) Xian" if 区域=="西安市"
save "F:\日度夜间灯光\原始数据\result\日度分区统计\deep\劳动争议元化解试点省份数据.dta",replace
fabplot line  MEAN date2, by(区域)  front(connect) frontopts(mc(red) lc(red) xlabel(none)) 
addplot : , tline(20Feb2020, lp(dash)) norescaling 


graph save 劳动争议多元化解试点.gph,replace 
graph export 劳动争议多元化解试点.svg,replace 
save "F:\日度夜间灯光\原始数据\result\日度分区统计\deep\劳动争议元化解试点城市数据.dta",replace
restore

*收敛性
use 2012-2020日度城市层面加班情况.dta,clear
destring dayOfYear,replace
merge m:1 year dayOfYear using  天数日期对照表.dta,nogen keep(1 3)
merge m:1 市代码 using 城市名称编码对照表.dta,nogen keep(1 3)
gen date2=date(date,"YMD")
gen date3=date(date,"YMD")
format date2 %td
xtset 市代码 date2

bysort  date2:egen mean_overwork=mean(MEAN)
gen dif=(MEAN-mean_overwork)^2

bysort  date2:egen mean_dif=mean(dif)
gen sigma=mean_dif^0.5

sum sigma
return list
gen s=sigma-r(mean)
ttest s==0
duplicates drop date2,force
tsset date2
gen sigma2=sigma-l.sigma
tsline  sigma,xsize(20) ysize(8) xlabel(#19,angle(45)) xtitle(date)
addplot:,yline(3.49,lc(red) lp(dash) lwidth(1)) norescaling
graph save 超时加班sigma日度收敛性.gph,replace 
graph export 超时加班sigma日度收敛性.svg,replace
// tsline MEAN