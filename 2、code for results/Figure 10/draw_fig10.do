*描述性统计
use "不去掉居民区+加班天数占比/dummy12_20.dta",clear
append using "去掉居民区+加班天数占比/DR_dummy12_20.dta"

rename SUM 超时加班强度总和
*合并企业数量
merge m:1 市代码 year using "城市企业数量/firms12_20.dta",nogen keep(1 3)
merge m:1 市代码 using 城市名称编码对照表.dta,nogen keep(1 3)
drop if 市代码==710000 //缺少台湾的数据
// bysort  year 是否使用企业数量加权 是否去掉居民区:egen mean_overwork=mean(MEAN)
// gen dif=(MEAN-mean_overwork)^2
//
// bysort  year 是否使用企业数量加权 是否去掉居民区:egen mean_dif=mean(dif)
// gen sigma=mean_dif^0.5
// xtset
// tsline  sigma,xsize(20) ysize(8) xlabel(#9,angle(45)) xtitle(date)
// addplot:,yline(0.083,lc(red) lp(dash) lwidth(1)) norescaling

drop if 类型=="不统计"|类型=="省直辖县"|类型=="地区"|类型=="特别行政区"
drop if 省=="西藏自治区"|省=="青海省"|省=="海南省"|省=="新疆维吾尔自治区"
rename SUM 城市企业存量
sum MEAN if  是否使用企业数量加权==1&是否去掉居民区==0
sum MEAN if  是否使用企业数量加权==0&是否去掉居民区==0
replace MEAN=-MEAN
sort year 是否使用企业数量加权 是否去掉居民区 MEAN
replace MEAN=MEAN
by year 是否使用企业数量加权 是否去掉居民区 :gen rank=_n
by year 是否使用企业数量加权 是否去掉居民区:gen num=_N
bysort 市代码 是否使用企业数量加权 是否去掉居民区 :egen variation=sd(rank)
order rank num year  是否使用企业数量加权 是否去掉居民区 是否使用企业数量加权 是否去掉居民区 MEAN 市,first
*画图
preserve 
// keep if  是否使用企业数量加权==0&是否去掉居民区==0
tostring 是否使用企业数量加权,replace
replace 是否使用企业数量加权="加权" if  是否使用企业数量加权=="1"
replace 是否使用企业数量加权="不加权" if  是否使用企业数量加权=="0"
keep if 市=="北京市"|市=="上海市"|市=="广州市"|市=="深圳市"

replace 市="Beijing" if 市=="北京市"
replace 市="Shanghai" if 市=="上海市"
replace 市="Guangzhou" if 市=="广州市"
replace 市="Shenzhen" if 市=="深圳市"
replace 是否使用企业数量加权="(a)" if 是否使用企业数量加权=="不加权"&市=="Shanghai"
replace 是否使用企业数量加权="(b)" if 是否使用企业数量加权=="加权"&市=="Shanghai"
replace 是否使用企业数量加权="(c)" if 是否使用企业数量加权=="不加权"&市=="Beijing"
replace 是否使用企业数量加权="(d)" if 是否使用企业数量加权=="加权"&市=="Beijing"
replace 是否使用企业数量加权="(e)" if 是否使用企业数量加权=="不加权"&市=="Guangzhou"
replace 是否使用企业数量加权="(f)" if 是否使用企业数量加权=="加权"&市=="Guangzhou"
replace 是否使用企业数量加权="(g)" if 是否使用企业数量加权=="不加权"&市=="Shenzhen"
replace 是否使用企业数量加权="(h)" if 是否使用企业数量加权=="加权"&市=="Shenzhen"

fabplot line  rank year, by(是否使用企业数量加权 市 )  front(connect) frontopts(mc(red) lc(red))

restore 

preserve 
// keep if  是否使用企业数量加权==0&是否去掉居民区==0
tostring 是否使用企业数量加权,replace
replace 是否使用企业数量加权="加权" if  是否使用企业数量加权=="1"
replace 是否使用企业数量加权="不加权" if  是否使用企业数量加权=="0"
keep if 市=="北京市"|市=="上海市"|市=="广州市"|市=="深圳市"
fabplot line  rank year, by(市 是否使用企业数量加权)  front(connect) frontopts(mc(red) lc(red)) 
graph save 北上广深超时加班排名变化.gph,replace
graph export 北上广深超时加班排名变化.svg,replace
restore 

preserve 
keep if  是否使用企业数量加权==0&是否去掉居民区==0
duplicates drop variation,force
gsort -variation
gen r=_n
keep if r<=10
keep variation 市
save var_top10_uw.dta,replace
keep variation
save 变异最大的10个城市-不加权.dta,replace
restore

preserve 
keep if  是否使用企业数量加权==1&是否去掉居民区==0
duplicates drop variation,force
gsort -variation
gen r=_n
keep if r<=10
keep variation 市
save var_top10_w.dta,replace
keep variation
save 变异最大的10个城市-加权.dta,replace
restore

preserve
merge m:1 variation using  变异最大的10个城市-加权.dta
rename _merge wv
merge m:1 variation using  变异最大的10个城市-不加权.dta
rename _merge uwv
fabplot line  rank year if wv==3, by(市 )  front(connect) frontopts(mc(red) lc(red)) 
graph save 排名变化最大的10个市-加权.gph,replace
graph export 排名变化最大的10个市-加权.svg,replace
gsort -variation
fabplot line  rank year if uwv==3, by(市 )  front(connect) frontopts(mc(red) lc(red)) 
graph save 排名变化最大的10个市-不加权.gph,replace
graph export 排名变化最大的10个市-不加权.svg,replace
restore 

graph combine "$figure\weighted variation.gph" "$figure\unweighted variation.gph"  ,col(2)
