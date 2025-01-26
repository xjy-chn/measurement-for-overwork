global figure output\figure
global table output\table

import excel "citycode-cityname.xlsx",clear firstrow
save citycode-cityname.dta,replace


use "ratio/dummy12_20.dta",clear


drop if citycode==710000 // Taiwan Province


drop if 类型=="不统计"|类型=="省直辖县"|类型=="地区"|类型=="特别行政区"
drop if 省=="西藏自治区"|省=="青海省"|省=="海南省"|省=="新疆维吾尔自治区"
rename SUM 城市企业存量
sum MEAN if  isweight==1
sum MEAN if  isweight==0
replace MEAN=-MEAN
sort year isweight  MEAN
by year isweight  :gen rank=_n
by year isweight :gen num=_N
order rank num year  isweight  isweight  MEAN ,first

*draw
preserve 
tostring isweight,replace

keep if cityname=="Beijing"|cityname=="Shanghai"|cityname=="Guangzhou"|cityname=="Shenzhen"


replace isweight="(a)" if isweight=="1"&cityname=="Shanghai"
replace isweight="(b)" if isweight=="0"&cityname=="Shanghai"
replace isweight="(c)" if isweight=="1"&cityname=="Beijing"
replace isweight="(d)" if isweight=="0"&cityname=="Beijing"
replace isweight="(e)" if isweight=="1"&cityname=="Guangzhou"
replace isweight="(f)" if isweight=="0"&cityname=="Guangzhou"
replace isweight="(g)" if isweight=="1"&cityname=="Shenzhen"
replace isweight="(h)" if isweight=="0"&cityname=="Shenzhen"

fabplot line  rank year, by(isweight cityname )  front(connect) frontopts(mc(red) lc(red)  ) 
restore 
