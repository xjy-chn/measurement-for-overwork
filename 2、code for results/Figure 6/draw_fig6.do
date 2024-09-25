forvalues i=2012(1)2020{
	import excel `i'年度汇总.xlsx,firstrow clear
	save `i'.dta,replace
}
use 2012.dta,clear
forvalues i=2013(1)2020{
	append using  `i'.dta
}
save 2012-2020日度城市层面加班情况.dta,replace
forvalues i=2012(1)2020{
	erase `i'.dta
}

import excel 天数日期对照表.xlsx,clear firstrow 
save 天数日期对照表.dta,replace

import excel 城市名-行政编码.xlsx,clear firstrow
save 城市名称编码对照表.dta,replace



use 2012-2020日度城市层面加班情况.dta,clear
destring dayOfYear,replace
merge m:1 year dayOfYear using  天数日期对照表.dta,nogen keep(1 3)
// merge m:1 市代码 using  天数日期对照表.dta,nogen keep(1 3)
gen date2=date(date,"YMD")
gen date3=date(date,"YMD")
format date2 %td

tsset date2                                  //设定为时间序列变量
gsort date2


*计算每年最大值最小值
sort year MEAN
by year:gen s=_n
by year:egen mins=min(s)
by year:egen maxs=max(s)
gen tops=(s==maxs)
gen bottoms=(s==mins)



gen toprline=0.01945
gen bottomrline=0.0196



*为图例修改变量名
gen 年度最高点=MEAN
gen 年度最低点=MEAN
replace MEAN=MEAN/100
tostring(MEAN),gen(labmean) force
gen labmean2="0"+substr(labmean,1,4)
drop if MEAN==2
// winsor2 MEAN,cut(1 99) replace
twoway (tsline MEAN, lcolor(%80) lwidth(vthin) tlabel(#19,angle(forty_five))  ttick(08Jan2012 26Jan2013 16Jan2014 04Feb2015 24Jan2016 13Jan2017 01Feb2018 21Jan2019 10Jan2020 16Feb2012 06Mar2013 24Feb2014 15Mar2015 03Mar2016 21Feb2017 12Mar2018 01Mar2019 18Feb2020, tpos(in) )) ///
 (scatter  MEAN date2 if tops==1,  ms(o) mcolor(red) mlabangle(75) mlabsize(quarter_tiny) mlabposition(12) mlabsize(small)  mlabel(labmean2) ) /// 
 (scatter  MEAN date2 if bottoms==1,  ms(d) mlabangle(-30)  mcolor(black) mlabel(labmean2))  /// 
 (tsrline toprline bottomrline  if date3==19078,  lcolor(red) lpattern(dash) ) /// 
 (tsrline toprline bottomrline  if date3==19078,  lcolor(black) lpattern(dash) ) , ///
 xsize(80) ysize(30) legend(label(1 超时加班情况) label(2 年度最高点) label(3 年度最低点) )  title("daily average overwork") xtitle("date") ytitle("overwork")

*劳动合同法修订
// addplot : , tline(28dec2012,lc(cyan))  norescaling 
*上下90%分位数
// addplot : , yline(0.320, lp(dash) lc(orange) lwidth(1))  legend(label(1 超时加班情况) label(2 年度最高点) label(3 年度最低点) )   norescaling 
// addplot : ,  yline(0.213, lp(dash) lc(orange)  lwidth(1) )  legend(order()) norescaling
*春运
**2012
addplot : , tline(08Jan2012 26Jan2013 16Jan2014 04Feb2015 24Jan2016 13Jan2017 01Feb2018 21Jan2019 10Jan2020, lp(dash) lc(red))  legend('a') norescaling 
addplot : , tline(16Feb2012 06Mar2013 24Feb2014 15Mar2015 03Mar2016 21Feb2017 12Mar2018 01Mar2019 18Feb2020, lp(dash) lc(black) ) legend()   norescaling

*标注事件
addplot : pcarrowi  0.255 19410 0.22 19355 "Revision of the Labor Contract Law" ///
 0.22 21247 0.1 21547  "Revision of the Labor Contract Law" ///
0.26 21600 0.16 21921 "Initial extraction of COVID-19" ///
 ,legend(label("超时加班情况") ) 
 
graph save 全国全年加班情况-春运.gph,replace
 
 
 
// addplot : pcarrowi  0.5 21347 0.27 21547 "劳动法修订",legend(order()) 	      //中间数字代表箭头首尾坐标
// addplot : pcarrowi  0.45 21450 0.33 21650 "996福报",legend(order()) 	      //中间数字代表箭头首尾坐标
// addplot : pcarrowi  0.4 21600 0.33 21921 "提取新冠病毒",legend(order(label(1 超时加班情况) label(2 年度最高点) label(3 年度最低点))) 	      //中间数字代表箭头首尾坐标