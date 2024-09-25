forvalues i=2012(1)2020{
	insheet using `i'dots.csv,clear
	drop if firms==32767
	save `i'dots.dta,replace
}
forvalues i=2012(1)2020{
	use `i'dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
	binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("分仓散点`i'.gph" ) savedata(分仓散点`i') replace title("`i'年" ) xsize(10) ysize(8) x
}

graph combine "分仓散点2012.gph" "分仓散点2013.gph"  "分仓散点2014.gph" "分仓散点2015.gph" ///
 "分仓散点2016.gph" "分仓散点2017.gph"  "分仓散点2018.gph" "分仓散点2019.gph" "分仓散点2020.gph" ,xsize(100) ysize(75) plotregion() 

 graph save 分仓散点2012-2020.dta,replace
 graph export 分仓散点2012-2020.svg,replace