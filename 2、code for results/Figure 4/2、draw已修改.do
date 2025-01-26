*In this part we only consider the register place ,so delete the sample of the work place of listed firms
forvalues i=2012(1)2020{
	insheet using `i'dots.csv,clear
	drop if firms==0
	save `i'dots.dta,replace
}

use 2012dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2012.gph" ) savedata(bin scatter`i') replace title("(a)2012" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))

use 2013dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2013.gph" ) savedata(bin scatter`i') replace title("(b)2013" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))

use 2014dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2014.gph" ) savedata(bin scatter`i') replace title("(c)2014" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))



use 2015dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2015.gph" ) savedata(bin scatter`i') replace title("(d)2015" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))



use 2016dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2016.gph" ) savedata(bin scatter`i') replace title("(e)2016" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))


use 2017dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2017.gph" ) savedata(bin scatter`i') replace title("(f)2017" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))




use 2018dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2018.gph" ) savedata(bin scatter`i') replace title("(g)2018" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))



use 2019dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2019.gph" ) savedata(bin scatter`i') replace title("(h)2019" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))


use 2020dots.dta,clear
// winsor2 overwork firms if firms>=1,cut(1 99) replace
binscatter overwork firms if firms>=1  ,linetype(lfit) savegraph("bin scatter2020.gph" ) savedata(bin scatter`i') replace title("(i)2020" ) xsize(10) ysize(8) ylabel(,labsize(small)) xlabel(,labsize(small))



graph combine "bin scatter2012.gph" "bin scatter2013.gph"  "bin scatter2014.gph" "bin scatter2015.gph" ///
 "bin scatter2016.gph" "bin scatter2017.gph"  "bin scatter2018.gph" "bin scatter2019.gph" "bin scatter2020.gph" ,xsize(100) ysize(75) plotregion() 

//  graph save bin scatter2012-2020.dta,replace
 graph export "bin scatter2012-2020.svg",replace
  graph export "bin scatter2012-2020.eps",replace