forvalues i=2012(1)2020{
	import excel `i'statistic.xlsx,firstrow clear
	save `i'.dta,replace
}
use 2012.dta,clear
forvalues i=2013(1)2020{
	append using  `i'.dta
}
save 2012-2020national_overwork.dta,replace
forvalues i=2012(1)2020{
	erase `i'.dta
}

import excel dayofweek.xlsx,clear firstrow 
save dayofweek.dta,replace

import excel citycode-cityname.xlsx,clear firstrow
save citycode-cityname.dta,replace



use 2012-2020national_overwork.dta,clear
destring dayOfYear,replace
merge m:1 year dayOfYear using  dayofweek.dta,nogen keep(1 3)

gen date2=date(date,"YMD")
gen date3=date(date,"YMD")
format date2 %td

tsset date2                                  
sort date3
gen id2=_n
tab id2 if date=="2013-01-26"|date=="2013-03-06"|date=="2014-01-16"|date=="2014-02-24"|date=="2015-02-04"|date=="2015-03-15"|date=="2016-01-24"|date=="2016-03-03"|date=="2017-01-13"|date=="2017-02-21"|date=="2018-02-01"|date=="2018-03-12"|date=="2019-01-20"|date=="2019-03-01"|date=="2020-01-10"|date=="2020-02-18"
preserve
gen duration1=(id2>=367&id2<=406)
gen duration2=(id2>=722&id2<=761)
gen duration3=(id2>=1106&id2<=1145)
gen duration4=(id2>=1460&id2<=1499)
gen duration5=(id2>=1815&id2<=1854)
gen duration6=(id2>=2198&id2<=2237)
gen duration7=(id2>=2550&id2<=2589)
gen duration8=(id2>=2904&id2<=2943)
keep if duration1==1|duration2==1|duration3==1|duration4==1|duration5==1|duration6==1|duration7==1|duration8==1
save spring_festival.dta,replace
restore
*zero denotes the workdays
keep  if type==0
sort date3
gen id=_n
*annual max min
sort year MEAN
by year:gen s=_n
by year:egen mins=min(s)
by year:egen maxs=max(s)
gen tops=(s==maxs)
gen bottoms=(s==mins)



gen toprline=0.1945
gen bottomrline=0.196


sort date2

gen top=MEAN
gen bottom=MEAN
// replace MEAN=MEAN/100
tostring(MEAN),gen(labmean) force
gen labmean2="0"+substr(labmean,1,4)
gen add1=real(substr(labmean,5,1))


destring labmean2,gen(labmean3)
replace labmean3=labmean3+0.001 if add1>=5
tostring labmean3,replace force
drop labmean2
gen labmean2="0"+substr(labmean3,1,4)
replace labmean2="0.130" if labmean2=="0.13"
replace labmean2="0.150" if labmean2=="0.15"
// winsor2 MEAN,cut(1 99) replace

twoway (tsline MEAN , lcolor(%80) lwidth(vthin) tlabel(#19,angle(forty_five)) ) ///
 (scatter  MEAN date2 if tops==1,  ms(o) mcolor(red) mlabangle(75) mlabsize(quarter_tiny) mlabposition(12) mlabsize(small)  mlabel(labmean2) ) /// 
 (scatter  MEAN date2 if bottoms==1,  ms(d) mlabangle(-30)  mcolor(black) mlabel(labmean2))  /// 
 (tsrline toprline bottomrline  if date3==19078,  lcolor(red) lpattern(dash) ) /// 
 (tsrline toprline bottomrline  if date3==19078,  lcolor(black) lpattern(dash) ) , ///
 xsize(80) ysize(30) legend(off)  title("(a) daily average overwork") xtitle("date") ytitle("overwork")





*标注事件
addplot : pcarrowi  0.255 19410 0.13 19355 "Revision of the Labor Contract Law" ///
 0.18 21000 0.115 21547  "Revision of the Labor  Law" ///
0.20 21200 0.14 21921 "Initial extraction of COVID-19" 
 
graph save overwork.gph,replace

*panel B
tab id if date=="2012-12-28"
global short 10
global long 60

twoway (line MEAN id  if  id>=235-$short&id<=235+$short , lcolor(%100) lwidth( medium  ) tlabel(none)) ///
(lfit MEAN id if id>=235-$short&id<=235 ) ///
(lfit MEAN id if id>=235&id<=235+$short ),xline(235,lp(dash) lc(red)) xtitle("") ylabel(none) legend(off) title("b1) Revision of the Labor Contract Law",size(thin)) 
graph save b1.gph,replace

 tab id if date=="2020-01-07"
 twoway (line MEAN id  if  id>=1980-$short&id<=1980+$short , lcolor(%100) lwidth( medium  ) tlabel(none)) ///
 (lfit MEAN id if id>=1980-$short&id<=1980 ) ///
(lfit MEAN id if id>=1980&id<=1980+$short ),xline(1980,lp(dash) lc(red)) xtitle("") ylabel(none) legend(off) title("b2) Initial extraction of COVID-19" ,size(thin) )
graph save b2.gph,replace


 tab id if date=="2018-12-29"
  twoway (line MEAN id  if  id>=1727-$short&id<=1727+$short , lcolor(%100) lwidth( medium  ) tlabel(none)) ///
 (lfit MEAN id if id>=1727-$short&id<=1727 ) ///
(lfit MEAN id if id>=1727&id<=1727+$short ),xline(1727,lp(dash) lc(red)) xtitle("") ylabel(none) legend(off) title("b3) Revision of the Labor Contract Law",size(thin))
graph save b3.gph,replace


twoway (line MEAN id  if  id>=235-$long&id<=235+$long , lcolor(%100) lwidth( medium  ) tlabel(none)) ///
(lfit MEAN id if id>=235-$long&id<=235 ) ///
(lfit MEAN id if id>=235&id<=235+$long ),xline(235,lp(dash) lc(red)) xtitle("") ylabel(none) legend(off) title("b4) Revision of the Labor Contract Law",size(thin)) 
graph save b4.gph,replace


 tab id if date=="2020-01-07"
 twoway (line MEAN id  if  id>=1980-$long&id<=1980+$long , lcolor(%100) lwidth( medium  ) tlabel(none)) ///
 (lfit MEAN id if id>=1980-$long&id<=1980 ) ///
(lfit MEAN id if id>=1980&id<=1980+$long ),xline(1980,lp(dash) lc(red)) xtitle("") ylabel(none) legend(off) title("b5) Initial extraction of COVID-19",size(thin))
graph save b5.gph,replace


 tab id if date=="2018-12-29"
  twoway (line MEAN id  if  id>=1727-$long&id<=1727+$long , lcolor(%100) lwidth( medium  ) tlabel(none)) ///
 (lfit MEAN id if id>=1727-$long&id<=1727 ) ///
(lfit MEAN id if id>=1726&id<=1726+$long ),xline(1726,lp(dash) lc(red)) xtitle("") ylabel(none) legend(off) title("b6) Revision of the Labor Contract Law",size(thin))
graph save b6.gph,replace

graph combine b1.gph b2.gph b3.gph,title("10 workdays",position(6)) rows(1) xsize(90) ysize(30)
graph save b1-b3.gph,replace
graph combine b4.gph b5.gph b6.gph,title("60 workdays",position(6)) rows(1) xsize(90) ysize(30)
graph save b4-b6.gph,replace
graph combine b1-b3.gph b4-b6.gph,rows(2)
graph save b1-b6.gph,replace

*panel c
use spring_festival,clear
bysort year typedes:egen minid=min(id2) if typedes=="春节"
bysort year:egen springfes=min(minid)
keep if  type==0
replace MEAN=MEAN/100
twoway (lfit MEAN   id2 if year==2013&id2<springfes) ///
(scatter MEAN   id2 if year==2013&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("") title("c1) 2013",size(small))
graph save 2013.gph,replace
twoway (lfit MEAN   id2 if year==2014&id2<springfes) ///
(scatter MEAN   id2 if year==2014&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("")  title("c2) 2014",size(small))
graph save 2014.gph,replace
twoway (lfit MEAN   id2 if year==2015&id2<springfes)  ///
(scatter MEAN   id2 if year==2015&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("") title("c3) 2015",size(small))
graph save 2015.gph,replace
twoway (lfit MEAN   id2 if year==2016&id2<springfes)  ///
(scatter MEAN   id2 if year==2016&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("") title("c4) 2016",size(small))
graph save 2016.gph,replace
twoway (lfit MEAN   id2 if year==2017&id2<springfes)  ///
(scatter MEAN   id2 if year==2017&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("") title("c5) 2017",size(small))
graph save 2017.gph,replace
twoway (lfit MEAN   id2 if year==2018&id2<springfes)  ///
(scatter MEAN   id2 if year==2018&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("") title("c6) 2018",size(small))
graph save 2018.gph,replace
twoway (lfit MEAN   id2 if year==2019&id2<springfes)  ///
(scatter MEAN   id2 if year==2019&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("") title("c7) 2019",size(small))
graph save 2019.gph,replace
twoway (lfit MEAN   id2 if year==2020&id2<springfes)   ///
(scatter MEAN   id2 if year==2020&id2<springfes) ,xlabel(none) ylabel(none) legend(off) xtitle("") title("c8) 2020",size(small))
graph save 2020.gph,replace
graph combine 2013.gph 2014.gph 2015.gph 2016.gph 2017.gph 2018.gph 2019.gph 2020.gph,rows(2)  cols(4) iscale(1) xsize(80) ysize(40) title("(c) Trends before the Spring Festival",size(middle))
graph save festival.gph,replace
graph combine overwork.gph b1-b6.gph festival.gph ,rows(3) xsize(65) ysize(100)
graph save fig6.gph,replace
graph export fig6.svg,replace

