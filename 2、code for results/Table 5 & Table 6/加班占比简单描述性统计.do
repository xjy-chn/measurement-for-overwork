
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

keep if rank<=10|rank>=num-9
replace rank=31-(num-rank+11) if rank>=num-9
keep 市 year rank 是否使用企业数量加权 是否去掉居民区
tostring rank 是否使用企业数量加权 是否去掉居民区,replace
gen id=rank+是否使用企业数量加权+是否去掉居民区
destring rank 是否使用企业数量加权 是否去掉居民区 id ,replace
duplicates report id
reshape wide 市,i(id) j(year)
export excel using "加班天数占比城市排名.xlsx",replace  firstrow(variables) 