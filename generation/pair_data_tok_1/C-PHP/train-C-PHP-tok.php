< ? php function search ( $ pat , $ txt ) { $ M = strlen ( $ pat ) ; $ N = strlen ( $ txt ) ; $ i = 0 ; while ( $ i <= $ N - $ M ) { $ j ;
for ( $ j = 0 ; $ j < $ M ; $ j ++ ) if ( $ txt [ $ i + $ j ] != $ pat [ $ j ] ) break ;
if ( $ j == $ M ) { echo ( " Pattern ▁ found ▁ at ▁ index ▁ $ i " . " STRNEWLINE " ) ; $ i = $ i + $ M ; } else if ( $ j == 0 ) $ i = $ i + 1 ; else
$ i = $ i + $ j ; } }
$ txt = " ABCEABCDABCEABCD " ; $ pat = " ABCD " ; search ( $ pat , $ txt ) ; ? >
< ? php function isPalRec ( $ str , $ s , $ e ) {
if ( $ s == $ e ) return true ;
if ( $ str [ $ s ] != $ str [ $ e ] ) return false ;
if ( $ s < $ e + 1 ) return isPalRec ( $ str , $ s + 1 , $ e - 1 ) ; return true ; } function isPalindrome ( $ str ) { $ n = strlen ( $ str ) ;
if ( $ n == 0 ) return true ; return isPalRec ( $ str , 0 , $ n - 1 ) ; }
{ $ str = " geeg " ; if ( isPalindrome ( $ str ) ) echo ( " Yes " ) ; else echo ( " No " ) ; return 0 ; } ? >
< ? php function CalPeri ( $ s ) { $ Perimeter = 10 * $ s ; echo " The ▁ Perimeter ▁ of ▁ Decagon ▁ is ▁ : ▁ $ Perimeter " ; }
$ s = 5 ; CalPeri ( $ s ) ; ? >
< ? php function distance ( $ a1 , $ b1 , $ c1 , $ a2 , $ b2 , $ c2 ) { $ d = ( $ a1 * $ a2 + $ b1 * $ b2 + $ c1 * $ c2 ) ; $ e1 = sqrt ( $ a1 * $ a1 + $ b1 * $ b1 + $ c1 * $ c1 ) ; $ e2 = sqrt ( $ a2 * $ a2 + $ b2 * $ b2 + $ c2 * $ c2 ) ; $ d = $ d / ( $ e1 * $ e2 ) ; $ pi = 3.14159 ; $ A = ( 180 / $ pi ) * ( acos ( $ d ) ) ; echo sprintf ( " Angle ▁ is ▁ % .2f ▁ degree " , $ A ) ; }
$ a1 = 1 ; $ b1 = 1 ; $ c1 = 2 ; $ d1 = 1 ; $ a2 = 2 ; $ b2 = -1 ; $ c2 = 1 ; $ d2 = -4 ; distance ( $ a1 , $ b1 , $ c1 , $ a2 , $ b2 , $ c2 ) ; ? >
< ? php function mirror_point ( $ a , $ b , $ c , $ d , $ x1 , $ y1 , $ z1 ) { $ k = ( - $ a * $ x1 - $ b * $ y1 - $ c * $ z1 - $ d ) / ( $ a * $ a + $ b * $ b + $ c * $ c ) ; $ x2 = $ a * $ k + $ x1 ; $ y2 = $ b * $ k + $ y1 ; $ z2 = $ c * $ k + $ z1 ; $ x3 = 2 * $ x2 - $ x1 ; $ y3 = 2 * $ y2 - $ y1 ; $ z3 = 2 * $ z2 - $ z1 ; echo sprintf ( " x3 ▁ = ▁ % .1f ▁ " , $ x3 ) ; echo sprintf ( " y3 ▁ = ▁ % .1f ▁ " , $ y3 ) ; echo sprintf ( " z3 ▁ = ▁ % .1f ▁ " , $ z3 ) ; }
$ a = 1 ; $ b = -2 ; $ c = 0 ; $ d = 0 ; $ x1 = -1 ; $ y1 = 3 ; $ z1 = 4 ;
mirror_point ( $ a , $ b , $ c , $ d , $ x1 , $ y1 , $ z1 ) ; ? >
< ? php function calculateSpan ( $ price , $ n , $ S ) {
$ S [ 0 ] = 1 ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) {
$ S [ $ i ] = 1 ;
for ( $ j = $ i - 1 ; ( $ j >= 0 ) && ( $ price [ $ i ] >= $ price [ $ j ] ) ; $ j -- ) $ S [ $ i ] ++ ; }
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ S [ $ i ] . " ▁ " ; ; }
$ price = array ( 10 , 4 , 5 , 90 , 120 , 80 ) ; $ n = count ( $ price ) ; $ S = array ( $ n ) ;
calculateSpan ( $ price , $ n , $ S ) ; ? >
< ? php function printNGE ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ next = -1 ; for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) { if ( $ arr [ $ i ] < $ arr [ $ j ] ) { $ next = $ arr [ $ j ] ; break ; } } echo $ arr [ $ i ] . " -- " . ▁ $ next . " " } }
$ arr = array ( 11 , 13 , 21 , 3 ) ; $ n = count ( $ arr ) ; printNGE ( $ arr , $ n ) ; ? >
< ? php function gcd ( $ a , $ b ) {
if ( $ a == 0 && $ b == 0 ) return 0 ; if ( $ a == 0 ) return $ b ; if ( $ b == 0 ) return $ a ;
if ( $ a == $ b ) return $ a ;
if ( $ a > $ b ) return gcd ( $ a - $ b , $ b ) ; return gcd ( $ a , $ b - $ a ) ; }
$ a = 98 ; $ b = 56 ; echo " GCD ▁ of ▁ $ a ▁ and ▁ $ b ▁ is ▁ " , gcd ( $ a , $ b ) ; ? >
< ? php function msbPos ( $ n ) { $ pos = 0 ; while ( $ n != 0 ) { $ pos ++ ;
$ n = $ n >> 1 ; } return $ pos ; }
function josephify ( $ n ) {
$ position = msbPos ( $ n ) ;
$ j = 1 << ( $ position - 1 ) ;
$ n = $ n ^ $ j ;
$ n = $ n << 1 ;
$ n = $ n | 1 ; return $ n ; }
$ n = 41 ; print ( josephify ( $ n ) ) ; ? >
< ? php function pairAndSum ( $ arr , $ n ) {
for ( $ i = 0 ; $ i < 32 ; $ i ++ ) {
$ k = 0 ; for ( $ j = 0 ; $ j < $ n ; $ j ++ ) if ( ( $ arr [ $ j ] & ( 1 << $ i ) ) ) $ k ++ ;
$ ans += ( 1 << $ i ) * ( $ k * ( $ k - 1 ) / 2 ) ; } return $ ans ; }
$ arr = array ( 5 , 10 , 15 ) ; $ n = sizeof ( $ arr ) ; echo pairAndSum ( $ arr , $ n ) ; ? >
< ? php function countSquares ( $ n ) {
return ( $ n * ( $ n + 1 ) / 2 ) * ( 2 * $ n + 1 ) / 3 ; }
$ n = 4 ; echo " Count ▁ of ▁ squares ▁ is ▁ " , countSquares ( $ n ) ; ? >
< ? php function gcd ( $ a , $ b ) {
if ( $ a == 0 ) return $ b ; if ( $ b == 0 ) return $ a ;
if ( $ a == $ b ) return $ a ;
if ( $ a > $ b ) return gcd ( $ a - $ b , $ b ) ; return gcd ( $ a , $ b - $ a ) ; }
$ a = 98 ; $ b = 56 ; echo " GCD ▁ of ▁ $ a ▁ and ▁ $ b ▁ is ▁ " , gcd ( $ a , $ b ) ; ? >
< ? php function largest ( $ arr , $ n ) { $ i ;
$ max = $ arr [ 0 ] ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] > $ max ) $ max = $ arr [ $ i ] ; return $ max ; }
$ arr = array ( 10 , 324 , 45 , 90 , 9808 ) ; $ n = sizeof ( $ arr ) ; echo " Largest ▁ in ▁ given ▁ array ▁ is ▁ " , largest ( $ arr , $ n ) ; ? >
< ? php function print2largest ( $ arr , $ arr_size ) {
if ( $ arr_size < 2 ) { echo ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } $ first = $ second = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ arr_size ; $ i ++ ) {
if ( $ arr [ $ i ] > $ first ) { $ second = $ first ; $ first = $ arr [ $ i ] ; }
else if ( $ arr [ $ i ] > $ second && $ arr [ $ i ] != $ first ) $ second = $ arr [ $ i ] ; } if ( $ second == PHP_INT_MIN ) echo ( " There ▁ is ▁ no ▁ second ▁ largest ▁ element STRNEWLINE " ) ; else echo ( " The ▁ second ▁ largest ▁ element ▁ is ▁ " . $ second . " STRNEWLINE " ) ; }
$ arr = array ( 12 , 35 , 1 , 10 , 34 , 1 ) ; $ n = sizeof ( $ arr ) ; print2largest ( $ arr , $ n ) ; ? >
< ? php function minJumps ( $ arr , $ l , $ h ) {
if ( $ h == $ l ) return 0 ;
if ( $ arr [ $ l ] == 0 ) return INT_MAX ;
$ min = 999999 ; for ( $ i = $ l + 1 ; $ i <= $ h && $ i <= $ l + $ arr [ $ l ] ; $ i ++ ) { $ jumps = minJumps ( $ arr , $ i , $ h ) ; if ( $ jumps != 999999 && $ jumps + 1 < $ min ) $ min = $ jumps + 1 ; } return $ min ; }
$ arr = array ( 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 ) ; $ n = count ( $ arr ) ; echo " Minimum ▁ number ▁ of ▁ jumps ▁ to ▁ reach ▁ " . " end ▁ is ▁ " . minJumps ( $ arr , 0 , $ n - 1 ) ; ? >
< ? php function smallestSubWithSum ( $ arr , $ n , $ x ) {
$ min_len = $ n + 1 ;
for ( $ start = 0 ; $ start < $ n ; $ start ++ ) {
$ curr_sum = $ arr [ $ start ] ;
if ( $ curr_sum > $ x ) return 1 ;
for ( $ end = $ start + 1 ; $ end < $ n ; $ end ++ ) {
$ curr_sum += $ arr [ $ end ] ;
if ( $ curr_sum > $ x && ( $ end - $ start + 1 ) < $ min_len ) $ min_len = ( $ end - $ start + 1 ) ; } } return $ min_len ; }
$ arr1 = array ( 1 , 4 , 45 , 6 , 10 , 19 ) ; $ x = 51 ; $ n1 = sizeof ( $ arr1 ) ; $ res1 = smallestSubWithSum ( $ arr1 , $ n1 , $ x ) ; if ( ( $ res1 == $ n1 + 1 ) == true ) echo " Not ▁ possible STRNEWLINE " ; else echo $ res1 , " STRNEWLINE " ; $ arr2 = array ( 1 , 10 , 5 , 2 , 7 ) ; $ n2 = sizeof ( $ arr2 ) ; $ x = 9 ; $ res2 = smallestSubWithSum ( $ arr2 , $ n2 , $ x ) ; if ( ( $ res2 == $ n2 + 1 ) == true ) echo " Not ▁ possible STRNEWLINE " ; else echo $ res2 , " STRNEWLINE " ; $ arr3 = array ( 1 , 11 , 100 , 1 , 0 , 200 , 3 , 2 , 1 , 250 ) ; $ n3 = sizeof ( $ arr3 ) ; $ x = 280 ; $ res3 = smallestSubWithSum ( $ arr3 , $ n3 , $ x ) ; if ( ( $ res3 == $ n3 + 1 ) == true ) echo " Not ▁ possible STRNEWLINE " ; else echo $ res3 , " STRNEWLINE " ; ? >
< ? php $ NA = -1 ;
function moveToEnd ( & $ mPlusN , $ size ) { global $ NA ; $ j = $ size - 1 ; for ( $ i = $ size - 1 ; $ i >= 0 ; $ i -- ) if ( $ mPlusN [ $ i ] != $ NA ) { $ mPlusN [ $ j ] = $ mPlusN [ $ i ] ; $ j -- ; } }
function merge ( & $ mPlusN , & $ N , $ m , $ n ) { $ i = $ n ;
$ j = 0 ;
$ k = 0 ;
while ( $ k < ( $ m + $ n ) ) {
if ( ( $ j == $ n ) || ( $ i < ( $ m + $ n ) && $ mPlusN [ $ i ] <= $ N [ $ j ] ) ) { $ mPlusN [ $ k ] = $ mPlusN [ $ i ] ; $ k ++ ; $ i ++ ; }
else { $ mPlusN [ $ k ] = $ N [ $ j ] ; $ k ++ ; $ j ++ ; } } }
function printArray ( & $ arr , $ size ) { for ( $ i = 0 ; $ i < $ size ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; echo " STRNEWLINE " ; }
$ mPlusN = array ( 2 , 8 , $ NA , $ NA , $ NA , 13 , $ NA , 15 , 20 ) ; $ N = array ( 5 , 7 , 9 , 25 ) ; $ n = sizeof ( $ N ) ; $ m = sizeof ( $ mPlusN ) - $ n ;
moveToEnd ( $ mPlusN , $ m + $ n ) ;
merge ( $ mPlusN , $ N , $ m , $ n ) ;
printArray ( $ mPlusN , $ m + $ n ) ; ? >
< ? php function getInvCount ( & $ arr , $ n ) { $ inv_count = 0 ; for ( $ i = 0 ; $ i < $ n - 1 ; $ i ++ ) for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) if ( $ arr [ $ i ] > $ arr [ $ j ] ) $ inv_count ++ ; return $ inv_count ; }
$ arr = array ( 1 , 20 , 6 , 4 , 5 ) ; $ n = sizeof ( $ arr ) ; echo " Number ▁ of ▁ inversions ▁ are ▁ " , getInvCount ( $ arr , $ n ) ; ? >
< ? php function minAbsSumPair ( $ arr , $ arr_size ) { $ inv_count = 0 ;
if ( $ arr_size < 2 ) { echo " Invalid ▁ Input " ; return ; }
$ min_l = 0 ; $ min_r = 1 ; $ min_sum = $ arr [ 0 ] + $ arr [ 1 ] ; for ( $ l = 0 ; $ l < $ arr_size - 1 ; $ l ++ ) { for ( $ r = $ l + 1 ; $ r < $ arr_size ; $ r ++ ) { $ sum = $ arr [ $ l ] + $ arr [ $ r ] ; if ( abs ( $ min_sum ) > abs ( $ sum ) ) { $ min_sum = $ sum ; $ min_l = $ l ; $ min_r = $ r ; } } } echo " The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are ▁ " . $ arr [ $ min_l ] . " ▁ and ▁ " . $ arr [ $ min_r ] ; }
$ arr = array ( 1 , 60 , -10 , 70 , -80 , 85 ) ; minAbsSumPair ( $ arr , 6 ) ; ? >
< ? php function printUnion ( $ arr1 , $ arr2 , $ m , $ n ) { $ i = 0 ; $ j = 0 ; while ( $ i < $ m && $ j < $ n ) { if ( $ arr1 [ $ i ] < $ arr2 [ $ j ] ) echo ( $ arr1 [ $ i ++ ] . " ▁ " ) ; else if ( $ arr2 [ $ j ] < $ arr1 [ $ i ] ) echo ( $ arr2 [ $ j ++ ] . " ▁ " ) ; else { echo ( $ arr2 [ $ j ++ ] . " " ) ; $ i ++ ; } }
while ( $ i < $ m ) echo ( $ arr1 [ $ i ++ ] . " ▁ " ) ; while ( $ j < $ n ) echo ( $ arr2 [ $ j ++ ] . " ▁ " ) ; }
$ arr1 = array ( 1 , 2 , 4 , 5 , 6 ) ; $ arr2 = array ( 2 , 3 , 5 , 7 ) ; $ m = sizeof ( $ arr1 ) ; $ n = sizeof ( $ arr2 ) ; printUnion ( $ arr1 , $ arr2 , $ m , $ n ) ; ? >
< ? php function printIntersection ( $ arr1 , $ arr2 , $ m , $ n ) { $ i = 0 ; $ j = 0 ; while ( $ i < $ m && $ j < $ n ) { if ( $ arr1 [ $ i ] < $ arr2 [ $ j ] ) $ i ++ ; else if ( $ arr2 [ $ j ] < $ arr1 [ $ i ] ) $ j ++ ; else { echo $ arr2 [ $ j ] , " " ; $ i ++ ; $ j ++ ; } } }
$ arr1 = array ( 1 , 2 , 4 , 5 , 6 ) ; $ arr2 = array ( 2 , 3 , 5 , 7 ) ; $ m = count ( $ arr1 ) ; $ n = count ( $ arr2 ) ;
printIntersection ( $ arr1 , $ arr2 , $ m , $ n ) ; ? >
< ? php function swap ( & $ a , & $ b ) { $ temp = $ a ; $ a = $ b ; $ b = $ temp ; }
function sort012 ( & $ a , $ arr_size ) { $ lo = 0 ; $ hi = $ arr_size - 1 ; $ mid = 0 ; while ( $ mid <= $ hi ) { switch ( $ a [ $ mid ] ) { case 0 : swap ( $ a [ $ lo ++ ] , $ a [ $ mid ++ ] ) ; break ; case 1 : $ mid ++ ; break ; case 2 : swap ( $ a [ $ mid ] , $ a [ $ hi -- ] ) ; break ; } } }
function printArray ( & $ arr , $ arr_size ) { for ( $ i = 0 ; $ i < $ arr_size ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; echo " STRNEWLINE " ; }
$ arr = array ( 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 ) ; $ arr_size = sizeof ( $ arr ) ; sort012 ( $ arr , $ arr_size ) ; echo " array ▁ after ▁ segregation ▁ " ; printArray ( $ arr , $ arr_size ) ; ? >
< ? php function printUnsorted ( & $ arr , $ n ) { $ s = 0 ; $ e = $ n - 1 ;
for ( $ s = 0 ; $ s < $ n - 1 ; $ s ++ ) { if ( $ arr [ $ s ] > $ arr [ $ s + 1 ] ) break ; } if ( $ s == $ n - 1 ) { echo " The ▁ complete ▁ array ▁ is ▁ sorted " ; return ; }
for ( $ e = $ n - 1 ; $ e > 0 ; $ e -- ) { if ( $ arr [ $ e ] < $ arr [ $ e - 1 ] ) break ; }
$ max = $ arr [ $ s ] ; $ min = $ arr [ $ s ] ; for ( $ i = $ s + 1 ; $ i <= $ e ; $ i ++ ) { if ( $ arr [ $ i ] > $ max ) $ max = $ arr [ $ i ] ; if ( $ arr [ $ i ] < $ min ) $ min = $ arr [ $ i ] ; }
for ( $ i = 0 ; $ i < $ s ; $ i ++ ) { if ( $ arr [ $ i ] > $ min ) { $ s = $ i ; break ; } }
for ( $ i = $ n - 1 ; $ i >= $ e + 1 ; $ i -- ) { if ( $ arr [ $ i ] < $ max ) { $ e = $ i ; break ; } }
echo " ▁ The ▁ unsorted ▁ subarray ▁ which ▁ makes ▁ " . " the ▁ given ▁ array ▁ " . " STRNEWLINE " . " ▁ sorted ▁ lies ▁ between ▁ the ▁ indees ▁ " . $ s . " ▁ and ▁ " . $ e ; return ; } $ arr = array ( 10 , 12 , 20 , 30 , 25 , 40 , 32 , 31 , 35 , 50 , 60 ) ; $ arr_size = sizeof ( $ arr ) ; printUnsorted ( $ arr , $ arr_size ) ; ? >
< ? php function findNumberOfTriangles ( $ arr ) { $ n = count ( $ arr ) ;
sort ( $ arr ) ;
$ count = 0 ;
for ( $ i = 0 ; $ i < $ n - 2 ; ++ $ i ) {
$ k = $ i + 2 ;
for ( $ j = $ i + 1 ; $ j < $ n ; ++ $ j ) {
while ( $ k < $ n && $ arr [ $ i ] + $ arr [ $ j ] > $ arr [ $ k ] ) ++ $ k ;
if ( $ k > $ j ) $ count += $ k - $ j - 1 ; } } return $ count ; }
$ arr = array ( 10 , 21 , 22 , 100 , 101 , 200 , 300 ) ; echo " Total ▁ number ▁ of ▁ triangles ▁ is ▁ " , findNumberOfTriangles ( $ arr ) ; ? >
< ? php function findElement ( $ arr , $ n , $ key ) { $ i ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] == $ key ) return $ i ; return -1 ; }
$ arr = array ( 12 , 34 , 10 , 6 , 40 ) ; $ n = sizeof ( $ arr ) ;
$ key = 40 ; $ position = findElement ( $ arr , $ n , $ key ) ; if ( $ position == - 1 ) echo ( " Element ▁ not ▁ found " ) ; else echo ( " Element ▁ Found ▁ at ▁ Position : ▁ " . ( $ position + 1 ) ) ; ? >
< ? php function insertSorted ( & $ arr , $ n , $ key , $ capacity ) {
if ( $ n >= $ capacity ) return $ n ; array_push ( $ arr , $ key ) ; return ( $ n + 1 ) ; }
$ arr = array ( 12 , 16 , 20 , 40 , 50 , 70 ) ; $ capacity = 20 ; $ n = 6 ; $ key = 26 ; echo " Before ▁ Insertion : ▁ " ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ;
$ n = insertSorted ( $ arr , $ n , $ key , $ capacity ) ; echo " After Insertion : " for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; ? >
< ? php function findElement ( & $ arr , $ n , $ key ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] == $ key ) return $ i ; return -1 ; }
function deleteElement ( & $ arr , $ n , $ key ) {
$ pos = findElement ( $ arr , $ n , $ key ) ; if ( $ pos == -1 ) { echo " Element ▁ not ▁ found " ; return $ n ; }
for ( $ i = $ pos ; $ i < $ n - 1 ; $ i ++ ) $ arr [ $ i ] = $ arr [ $ i + 1 ] ; return $ n - 1 ; }
$ arr = array ( 10 , 50 , 30 , 40 , 20 ) ; $ n = count ( $ arr ) ; $ key = 30 ; echo " Array ▁ before ▁ deletion STRNEWLINE " ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; $ n = deleteElement ( $ arr , $ n , $ key ) ; echo " Array after deletion " ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; ? >
< ? php function binarySearch ( $ arr , $ low , $ high , $ key ) { if ( $ high < $ low ) return -1 ;
$ mid = ( $ low + $ high ) / 2 ; if ( $ key == $ arr [ ( int ) $ mid ] ) return $ mid ; if ( $ key > $ arr [ ( int ) $ mid ] ) return binarySearch ( $ arr , ( $ mid + 1 ) , $ high , $ key ) ; return ( binarySearch ( $ arr , $ low , ( $ mid -1 ) , $ key ) ) ; }
$ arr = array ( 5 , 6 , 7 , 8 , 9 , 10 ) ; $ n = count ( $ arr ) ; $ key = 10 ; echo " Index : ▁ " , ( int ) binarySearch ( $ arr , 0 , $ n -1 , $ key ) ; ? >
< ? php function equilibrium ( $ arr , $ n ) { $ i ; $ j ; $ leftsum ; $ rightsum ;
for ( $ i = 0 ; $ i < $ n ; ++ $ i ) { $ leftsum = 0 ; $ rightsum = 0 ;
for ( $ j = 0 ; $ j < $ i ; $ j ++ ) $ leftsum += $ arr [ $ j ] ;
for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) $ rightsum += $ arr [ $ j ] ;
if ( $ leftsum == $ rightsum ) return $ i ; }
return -1 ; }
$ arr = array ( -7 , 1 , 5 , 2 , -4 , 3 , 0 ) ; $ arr_size = sizeof ( $ arr ) ; echo equilibrium ( $ arr , $ arr_size ) ; ? >
< ? php function equilibrium ( $ arr , $ n ) {
$ sum = 0 ;
$ leftsum = 0 ;
for ( $ i = 0 ; $ i < $ n ; ++ $ i ) $ sum += $ arr [ $ i ] ; for ( $ i = 0 ; $ i < $ n ; ++ $ i ) {
$ sum -= $ arr [ $ i ] ; if ( $ leftsum == $ sum ) return $ i ; $ leftsum += $ arr [ $ i ] ; }
return -1 ; }
$ arr = array ( -7 , 1 , 5 , 2 , -4 , 3 , 0 ) ; $ arr_size = sizeof ( $ arr ) ; echo " First ▁ equilibrium ▁ index ▁ is ▁ " , equilibrium ( $ arr , $ arr_size ) ; ? >
< ? php function ceilSearch ( $ arr , $ low , $ high , $ x ) {
if ( $ x <= $ arr [ $ low ] ) return $ low ;
for ( $ i = $ low ; $ i < $ high ; $ i ++ ) { if ( $ arr [ $ i ] == $ x ) return $ i ;
if ( $ arr [ $ i ] < $ x && $ arr [ $ i + 1 ] >= $ x ) return $ i + 1 ; }
return -1 ; }
$ arr = array ( 1 , 2 , 8 , 10 , 10 , 12 , 19 ) ; $ n = sizeof ( $ arr ) ; $ x = 3 ; $ index = ceilSearch ( $ arr , 0 , $ n - 1 , $ x ) ; if ( $ index == -1 ) echo ( " Ceiling ▁ of ▁ " . $ x . " ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " ) ; else echo ( " ceiling ▁ of ▁ " . $ x . " ▁ is ▁ " . $ arr [ $ index ] ) ; ? >
< ? php function ceilSearch ( $ arr , $ low , $ high , $ x ) { $ mid ;
if ( $ x <= $ arr [ $ low ] ) return $ low ;
if ( $ x > $ arr [ $ high ] ) return -1 ;
$ mid = ( $ low + $ high ) / 2 ;
if ( $ arr [ $ mid ] == $ x ) return $ mid ;
else if ( $ arr [ $ mid ] < $ x ) { if ( $ mid + 1 <= $ high && $ x <= $ arr [ $ mid + 1 ] ) return $ mid + 1 ; else return ceilSearch ( $ arr , $ mid + 1 , $ high , $ x ) ; }
else { if ( $ mid - 1 >= $ low && $ x > $ arr [ $ mid - 1 ] ) return $ mid ; else return ceilSearch ( $ arr , $ low , $ mid - 1 , $ x ) ; } }
$ arr = array ( 1 , 2 , 8 , 10 , 10 , 12 , 19 ) ; $ n = sizeof ( $ arr ) ; $ x = 20 ; $ index = ceilSearch ( $ arr , 0 , $ n - 1 , $ x ) ; if ( $ index == -1 ) echo ( " Ceiling ▁ of ▁ $ x ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " ) ; else echo ( " ceiling ▁ of ▁ $ x ▁ is " ) ; echo ( isset ( $ arr [ $ index ] ) ) ; ? >
< ? php $ NUM_LINE = 2 ; $ NUM_STATION = 4 ;
function carAssembly ( $ a , $ t , $ e , $ x ) { global $ NUM_LINE , $ NUM_STATION ; $ T1 = array ( ) ; $ T2 = array ( ) ; $ i ;
$ T1 [ 0 ] = $ e [ 0 ] + $ a [ 0 ] [ 0 ] ;
$ T2 [ 0 ] = $ e [ 1 ] + $ a [ 1 ] [ 0 ] ;
for ( $ i = 1 ; $ i < $ NUM_STATION ; ++ $ i ) { $ T1 [ $ i ] = min ( $ T1 [ $ i - 1 ] + $ a [ 0 ] [ $ i ] , $ T2 [ $ i - 1 ] + $ t [ 1 ] [ $ i ] + $ a [ 0 ] [ $ i ] ) ; $ T2 [ $ i ] = min ( $ T2 [ $ i - 1 ] + $ a [ 1 ] [ $ i ] , $ T1 [ $ i - 1 ] + $ t [ 0 ] [ $ i ] + $ a [ 1 ] [ $ i ] ) ; }
return min ( $ T1 [ $ NUM_STATION - 1 ] + $ x [ 0 ] , $ T2 [ $ NUM_STATION - 1 ] + $ x [ 1 ] ) ; }
$ a = array ( array ( 4 , 5 , 3 , 2 ) , array ( 2 , 10 , 1 , 4 ) ) ; $ t = array ( array ( 0 , 7 , 4 , 5 ) , array ( 0 , 9 , 2 , 8 ) ) ; $ e = array ( 10 , 12 ) ; $ x = array ( 18 , 7 ) ; echo carAssembly ( $ a , $ t , $ e , $ x ) ; ? >
< ? php function minPalPartion ( $ str ) {
$ n = strlen ( $ str ) ;
$ C = array_fill ( 0 , $ n , array_fill ( 0 , $ n , NULL ) ) ; $ P = array_fill ( false , $ n , array_fill ( false , $ n , NULL ) ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ P [ $ i ] [ $ i ] = true ; $ C [ $ i ] [ $ i ] = 0 ; }
for ( $ L = 2 ; $ L <= $ n ; $ L ++ ) {
for ( $ i = 0 ; $ i < $ n - $ L + 1 ; $ i ++ ) {
$ j = $ i + $ L - 1 ;
if ( $ L == 2 ) $ P [ $ i ] [ $ j ] = ( $ str [ $ i ] == $ str [ $ j ] ) ; else $ P [ $ i ] [ $ j ] = ( $ str [ $ i ] == $ str [ $ j ] ) && $ P [ $ i + 1 ] [ $ j - 1 ] ;
if ( $ P [ $ i ] [ $ j ] == true ) $ C [ $ i ] [ $ j ] = 0 ; else {
$ C [ $ i ] [ $ j ] = PHP_INT_MAX ; for ( $ k = $ i ; $ k <= $ j - 1 ; $ k ++ ) $ C [ $ i ] [ $ j ] = min ( $ C [ $ i ] [ $ j ] , $ C [ $ i ] [ $ k ] + $ C [ $ k + 1 ] [ $ j ] + 1 ) ; } } }
return $ C [ 0 ] [ $ n - 1 ] ; }
$ str = " ababbbabbababa " ; echo " Min ▁ cuts ▁ needed ▁ for ▁ Palindrome ▁ Partitioning ▁ is ▁ " . minPalPartion ( $ str ) ; return 0 ; ? >
< ? php function sum ( $ n ) { $ i ; $ s = 0.0 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ s = $ s + 1 / $ i ; return $ s ; }
$ n = 5 ; echo ( " Sum ▁ is ▁ " ) ; echo ( sum ( $ n ) ) ; ? >
< ? php function nthTermOfTheSeries ( $ n ) {
if ( $ n % 2 == 0 ) $ nthTerm = pow ( $ n - 1 , 2 ) + $ n ;
else $ nthTerm = pow ( $ n + 1 , 2 ) + $ n ;
return $ nthTerm ; }
$ n = 8 ; echo nthTermOfTheSeries ( $ n ) . " STRNEWLINE " ; $ n = 12 ; echo nthTermOfTheSeries ( $ n ) . " STRNEWLINE " ; $ n = 102 ; echo nthTermOfTheSeries ( $ n ) . " STRNEWLINE " ; $ n = 999 ; echo nthTermOfTheSeries ( $ n ) . " STRNEWLINE " ; $ n = 9999 ; echo nthTermOfTheSeries ( $ n ) . " STRNEWLINE " ; ? >
< ? php function Log2n ( $ n ) { return ( $ n > 1 ) ? 1 + Log2n ( $ n / 2 ) : 0 ; }
$ n = 32 ; echo Log2n ( $ n ) ; ? >
< ? php function findAmount ( $ X , $ W , $ Y ) { return ( $ X * ( $ Y - $ W ) ) / ( 100 - $ Y ) ; }
$ X = 100 ; $ W = 50 ; $ Y = 60 ; echo " Water ▁ to ▁ be ▁ added ▁ = ▁ " . findAmount ( $ X , $ W , $ Y ) ; ? >
< ? php function AvgofSquareN ( $ n ) { return ( ( $ n + 1 ) * ( 2 * $ n + 1 ) ) / 6 ; }
$ n = 2 ; echo ( AvgofSquareN ( $ n ) ) ; ? >
< ? php function triangular_series ( $ n ) { for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) echo ( " ▁ " . $ i * ( $ i + 1 ) / 2 . " ▁ " ) ; }
$ n = 5 ; triangular_series ( $ n ) ; ? >
< ? php function divisorSum ( $ n ) { $ sum = 0 ; for ( $ i = 1 ; $ i <= $ n ; ++ $ i ) $ sum += floor ( $ n / $ i ) * $ i ; return $ sum ; }
$ n = 4 ; echo divisorSum ( $ n ) , " STRNEWLINE " ; $ n = 5 ; echo divisorSum ( $ n ) , " STRNEWLINE " ; ? >
< ? php function sum ( $ x , $ n ) { $ i ; $ total = 1.0 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ total = $ total + ( pow ( $ x , $ i ) / $ i ) ; return $ total ; }
$ x = 2 ; $ n = 5 ; echo ( sum ( $ x , $ n ) ) ; ? >
< ? php function check ( $ n ) {
return 1162261467 % $ n == 0 ; }
$ n = 9 ; if ( check ( $ n ) ) echo ( " Yes " ) ; else echo ( " No " ) ; ? >
< ? php function per ( $ n ) { $ a = 3 ; $ b = 0 ; $ c = 2 ; $ i ; $ m ; if ( $ n == 0 ) return $ a ; if ( $ n == 1 ) return $ b ; if ( $ n == 2 ) return $ c ; while ( $ n > 2 ) { $ m = $ a + $ b ; $ a = $ b ; $ b = $ c ; $ c = $ m ; $ n -- ; } return $ m ; }
$ n = 9 ; echo per ( $ n ) ; ? >
< ? php function countDivisors ( $ n ) {
$ count = 0 ;
for ( $ i = 1 ; $ i <= sqrt ( $ n ) + 1 ; $ i ++ ) { if ( $ n % $ i == 0 )
$ count += ( $ n / $ i == $ i ) ? 1 : 2 ; } if ( $ count % 2 == 0 ) echo " Even STRNEWLINE " ; else echo " Odd STRNEWLINE " ; }
echo " The ▁ count ▁ of ▁ divisor : ▁ " ; countDivisors ( 10 ) ; ? >
< ? php function countSquares ( $ m , $ n ) {
if ( $ n < $ m ) list ( $ m , $ n ) = array ( $ n , $ m ) ;
return $ m * ( $ m + 1 ) * ( 2 * $ m + 1 ) / 6 + ( $ n - $ m ) * $ m * ( $ m + 1 ) / 2 ; }
$ m = 4 ; $ n = 3 ; echo ( " Count ▁ of ▁ squares ▁ is ▁ " . countSquares ( $ m , $ n ) ) ; ? >
< ? php function sum ( $ n ) { $ i ; $ s = 0.0 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ s = $ s + 1 / $ i ; return $ s ; }
$ n = 5 ; echo ( " Sum ▁ is ▁ " ) ; echo ( sum ( $ n ) ) ; ? >
< ? php function gcd ( $ a , $ b ) { if ( $ b == 0 ) return $ a ; return gcd ( $ b , $ a % $ b ) ; }
$ a = 98 ; $ b = 56 ; echo " GCD ▁ of ▁ $ a ▁ and ▁ $ b ▁ is ▁ " , gcd ( $ a , $ b ) ; ? >
< ? php function printArray ( $ arr , $ size ) { for ( $ i = 0 ; $ i < $ size ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; echo " STRNEWLINE " ; return ; }
function printSequencesRecur ( $ arr , $ n , $ k , $ index ) { if ( $ k == 0 ) { printArray ( $ arr , $ index ) ; } if ( $ k > 0 ) { for ( $ i = 1 ; $ i <= $ n ; ++ $ i ) { $ arr [ $ index ] = $ i ; printSequencesRecur ( $ arr , $ n , $ k - 1 , $ index + 1 ) ; } } }
function printSequences ( $ n , $ k ) { $ arr = array ( ) ; printSequencesRecur ( $ arr , $ n , $ k , 0 ) ; return ; }
$ n = 3 ; $ k = 2 ; printSequences ( $ n , $ k ) ; ? >
< ? php function isMultipleof5 ( $ n ) { while ( $ n > 0 ) $ n = $ n - 5 ; if ( $ n == 0 ) return true ; return false ; }
$ n = 19 ; if ( isMultipleof5 ( $ n ) == true ) echo ( " $ n ▁ is ▁ multiple ▁ of ▁ 5" ) ; else echo ( " $ n ▁ is ▁ not ▁ a ▁ multiple ▁ of ▁ 5" ) ; ? >
< ? php function countBits ( $ n ) { $ count = 0 ; while ( $ n ) { $ count ++ ; $ n >>= 1 ; } return $ count ; }
$ i = 65 ; echo ( countBits ( $ i ) ) ; ? >
< ? php function isKthBitSet ( $ x , $ k ) { return ( $ x & ( 1 << ( $ k - 1 ) ) ) ? 1 : 0 ; }
function leftmostSetBit ( $ x ) { $ count = 0 ; while ( $ x ) { $ count ++ ; $ x = $ x >> 1 ; } return $ count ; }
function isBinPalindrome ( $ x ) { $ l = leftmostSetBit ( $ x ) ; $ r = 1 ;
while ( $ l > $ r ) {
if ( isKthBitSet ( $ x , $ l ) != isKthBitSet ( $ x , $ r ) ) return 0 ; $ l -- ; $ r ++ ; } return 1 ; } function findNthPalindrome ( $ n ) { $ pal_count = 0 ;
$ i = 0 ; for ( $ i = 1 ; $ i <= PHP_INT_MAX ; $ i ++ ) { if ( isBinPalindrome ( $ i ) ) { $ pal_count ++ ; }
if ( $ pal_count == $ n ) break ; } return $ i ; }
$ n = 9 ;
echo ( findNthPalindrome ( $ n ) ) ; ? >
< ? php function lps ( $ seq , $ i , $ j ) {
if ( $ i == $ j ) return 1 ;
if ( $ seq [ $ i ] == $ seq [ $ j ] && $ i + 1 == $ j ) return 2 ;
if ( $ seq [ $ i ] == $ seq [ $ j ] ) return lps ( $ seq , $ i + 1 , $ j - 1 ) + 2 ;
return max ( lps ( $ seq , $ i , $ j - 1 ) , lps ( $ seq , $ i + 1 , $ j ) ) ; }
$ seq = " GEEKSFORGEEKS " ; $ n = strlen ( $ seq ) ; echo " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ " . lps ( $ seq , 0 , $ n - 1 ) ; ? >
< ? php function exponentMod ( $ A , $ B , $ C ) {
if ( $ A == 0 ) return 0 ; if ( $ B == 0 ) return 1 ;
if ( $ B % 2 == 0 ) { $ y = exponentMod ( $ A , $ B / 2 , $ C ) ; $ y = ( $ y * $ y ) % $ C ; }
else { $ y = $ A % $ C ; $ y = ( $ y * exponentMod ( $ A , $ B - 1 , $ C ) % $ C ) % $ C ; } return ( ( $ y + $ C ) % $ C ) ; }
$ A = 2 ; $ B = 5 ; $ C = 13 ; echo " Power ▁ is ▁ " . exponentMod ( $ A , $ B , $ C ) ; ? >
< ? php function printknapSack ( $ W , & $ wt , & $ val , $ n ) { $ K = array_fill ( 0 , $ n + 1 , array_fill ( 0 , $ W + 1 , NULL ) ) ;
for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) { for ( $ w = 0 ; $ w <= $ W ; $ w ++ ) { if ( $ i == 0 $ w == 0 ) $ K [ $ i ] [ $ w ] = 0 ; else if ( $ wt [ $ i - 1 ] <= $ w ) $ K [ $ i ] [ $ w ] = max ( $ val [ $ i - 1 ] + $ K [ $ i - 1 ] [ $ w - $ wt [ $ i - 1 ] ] , $ K [ $ i - 1 ] [ $ w ] ) ; else $ K [ $ i ] [ $ w ] = $ K [ $ i - 1 ] [ $ w ] ; } }
$ res = $ K [ $ n ] [ $ W ] ; echo $ res . " STRNEWLINE " ; $ w = $ W ; for ( $ i = $ n ; $ i > 0 && $ res > 0 ; $ i -- ) {
if ( $ res == $ K [ $ i - 1 ] [ $ w ] ) continue ; else {
echo $ wt [ $ i - 1 ] . " " ;
$ res = $ res - $ val [ $ i - 1 ] ; $ w = $ w - $ wt [ $ i - 1 ] ; } } }
$ val = array ( 60 , 100 , 120 ) ; $ wt = array ( 10 , 20 , 30 ) ; $ W = 50 ; $ n = sizeof ( $ val ) ; printknapSack ( $ W , $ wt , $ val , $ n ) ; ? >
< ? php function eggDrop ( $ n , $ k ) {
$ eggFloor = array ( array ( ) ) ; ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { $ eggFloor [ $ i ] [ 1 ] = 1 ; $ eggFloor [ $ i ] [ 0 ] = 0 ; }
for ( $ j = 1 ; $ j <= $ k ; $ j ++ ) $ eggFloor [ 1 ] [ $ j ] = $ j ;
for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) { for ( $ j = 2 ; $ j <= $ k ; $ j ++ ) { $ eggFloor [ $ i ] [ $ j ] = 999999 ; for ( $ x = 1 ; $ x <= $ j ; $ x ++ ) { $ res = 1 + max ( $ eggFloor [ $ i - 1 ] [ $ x - 1 ] , $ eggFloor [ $ i ] [ $ j - $ x ] ) ; if ( $ res < $ eggFloor [ $ i ] [ $ j ] ) $ eggFloor [ $ i ] [ $ j ] = $ res ; } } }
return $ eggFloor [ $ n ] [ $ k ] ; }
$ n = 2 ; $ k = 36 ; echo " Minimum ▁ number ▁ of ▁ trials ▁ in ▁ worst ▁ case ▁ with ▁ " . $ n . " ▁ eggs ▁ and ▁ " . $ k . " ▁ floors ▁ is ▁ " . eggDrop ( $ n , $ k ) ; ? >
< ? php function knapSack ( $ W , $ wt , $ val , $ n ) {
if ( $ n == 0 $ W == 0 ) return 0 ;
if ( $ wt [ $ n - 1 ] > $ W ) return knapSack ( $ W , $ wt , $ val , $ n - 1 ) ;
else return max ( $ val [ $ n - 1 ] + knapSack ( $ W - $ wt [ $ n - 1 ] , $ wt , $ val , $ n - 1 ) , knapSack ( $ W , $ wt , $ val , $ n -1 ) ) ; }
$ val = array ( 60 , 100 , 120 ) ; $ wt = array ( 10 , 20 , 30 ) ; $ W = 50 ; $ n = count ( $ val ) ; echo knapSack ( $ W , $ wt , $ val , $ n ) ; ? >
< ? php $ d = 256 ;
function search ( $ pat , $ txt , $ q ) { $ M = strlen ( $ pat ) ; $ N = strlen ( $ txt ) ; $ i ; $ j ;
$ h = 1 ; $ d = 1 ;
for ( $ i = 0 ; $ i < $ M - 1 ; $ i ++ ) $ h = ( $ h * $ d ) % $ q ;
for ( $ i = 0 ; $ i < $ M ; $ i ++ ) { $ p = ( $ d * $ p + $ pat [ $ i ] ) % $ q ; $ t = ( $ d * $ t + $ txt [ $ i ] ) % $ q ; }
for ( $ i = 0 ; $ i <= $ N - $ M ; $ i ++ ) {
if ( $ p == $ t ) {
for ( $ j = 0 ; $ j < $ M ; $ j ++ ) { if ( $ txt [ $ i + $ j ] != $ pat [ $ j ] ) break ; }
if ( $ j == $ M ) echo " Pattern ▁ found ▁ at ▁ index ▁ " , $ i , " STRNEWLINE " ; }
if ( $ i < $ N - $ M ) { $ t = ( $ d * ( $ t - $ txt [ $ i ] * $ h ) + $ txt [ $ i + $ M ] ) % $ q ;
if ( $ t < 0 ) $ t = ( $ t + $ q ) ; } } }
$ txt = " GEEKS ▁ FOR ▁ GEEKS " ; $ pat = " GEEK " ;
$ q = 101 ;
search ( $ pat , $ txt , $ q ) ; ? >
< ? php function gcd ( $ a , $ b ) { if ( $ b == 0 ) return $ a ; return gcd ( $ b , $ a % $ b ) ; }
$ a = 98 ; $ b = 56 ; echo " GCD ▁ of ▁ $ a ▁ and ▁ $ b ▁ is ▁ " , gcd ( $ a , $ b ) ; ? >
< ? php function checkSemiprime ( $ num ) { $ cnt = 0 ; for ( $ i = 2 ; $ cnt < 2 && $ i * $ i <= $ num ; ++ $ i ) while ( $ num % $ i == 0 ) $ num /= $ i ; ++ $ cnt ;
if ( $ num > 1 ) ++ $ cnt ;
return $ cnt == 2 ; }
function semiprime ( $ n ) { if ( checkSemiprime ( $ n ) ) echo " True STRNEWLINE " ; else echo " False STRNEWLINE " ; }
$ n = 6 ; semiprime ( $ n ) ; $ n = 8 ; semiprime ( $ n ) ; ? >
< ? php function indexedSequentialSearch ( $ arr , $ n , $ k ) { $ elements = array ( ) ; $ indices = array ( ) ; $ temp = array ( ) ; $ j = 0 ; $ ind = 0 ; $ start = 0 ; $ end = 0 ; $ set = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i += 3 ) {
$ elements [ $ ind ] = $ arr [ $ i ] ;
$ indices [ $ ind ] = $ i ; $ ind ++ ; } if ( $ k < $ elements [ 0 ] ) { echo " Not ▁ found " ; } else { for ( $ i = 1 ; $ i <= $ ind ; $ i ++ ) if ( $ k < $ elements [ $ i ] ) { $ start = $ indices [ $ i - 1 ] ; $ set = 1 ; $ end = $ indices [ $ i ] ; break ; } } if ( $ set == 1 ) { $ start = $ indices [ $ i - 1 ] ; $ end = $ n ; } for ( $ i = $ start ; $ i <= $ end ; $ i ++ ) { if ( $ k == $ arr [ $ i ] ) { $ j = 1 ; break ; } } if ( $ j == 1 ) echo " Found ▁ at ▁ index ▁ " , $ i ; else echo " Not ▁ found " ; }
$ arr = array ( 6 , 7 , 8 , 9 , 10 ) ; $ n = count ( $ arr ) ;
$ k = 8 ;
indexedSequentialSearch ( $ arr , $ n , $ k ) ; ? >
< ? php function printNSE ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ next = -1 ; for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) { if ( $ arr [ $ i ] > $ arr [ $ j ] ) { $ next = $ arr [ $ j ] ; break ; } } echo $ arr [ $ i ] . " -- " . ▁ $ next . " " } }
$ arr = array ( 11 , 13 , 21 , 3 ) ; $ n = count ( $ arr ) ; printNSE ( $ arr , $ n ) ; ? >
< ? php $ NO_OF_CHARS = 256 ; $ count = array_fill ( 0 , 200 , 0 ) ;
function getCharCountArray ( $ str ) { global $ count ; for ( $ i = 0 ; $ i < strlen ( $ str ) ; $ i ++ ) $ count [ ord ( $ str [ $ i ] ) ] ++ ; }
function firstNonRepeating ( $ str ) { global $ count ; getCharCountArray ( $ str ) ; $ index = -1 ; for ( $ i = 0 ; $ i < strlen ( $ str ) ; $ i ++ ) { if ( $ count [ ord ( $ str [ $ i ] ) ] == 1 ) { $ index = $ i ; break ; } } return $ index ; }
$ str = " geeksforgeeks " ; $ index = firstNonRepeating ( $ str ) ; if ( $ index == -1 ) echo " Either ▁ all ▁ characters ▁ are " . " ▁ repeating ▁ or ▁ string ▁ is ▁ empty " ; else echo " First ▁ non - repeating ▁ " . " character ▁ is ▁ " . $ str [ $ index ] ; ? >
< ? php function divideString ( $ str , $ n ) { $ str_size = strlen ( $ str ) ; $ i ; $ part_size ;
if ( $ str_size % $ n != 0 ) { echo " Invalid ▁ Input : ▁ String ▁ size " ; echo " ▁ is ▁ not ▁ divisible ▁ by ▁ n " ; return ; }
$ part_size = $ str_size / $ n ; for ( $ i = 0 ; $ i < $ str_size ; $ i ++ ) { if ( $ i % $ part_size == 0 ) echo " STRNEWLINE " ; echo $ str [ $ i ] ; } }
$ str = " a _ simple _ divide _ string _ quest " ;
divideString ( $ str , 4 ) ; ? >
< ? php function collinear ( $ x1 , $ y1 , $ x2 , $ y2 , $ x3 , $ y3 ) { if ( ( $ y3 - $ y2 ) * ( $ x2 - $ x1 ) == ( $ y2 - $ y1 ) * ( $ x3 - $ x2 ) ) echo ( " Yes " ) ; else echo ( " No " ) ; }
$ x1 = 1 ; $ x2 = 1 ; $ x3 = 0 ; $ y1 = 1 ; $ y2 = 6 ; $ y3 = 9 ; collinear ( $ x1 , $ y1 , $ x2 , $ y2 , $ x3 , $ y3 ) ; ? >
< ? php function bestApproximate ( $ x , $ y , $ n ) { $ i ; $ j ; $ m ; $ c ; $ sum_x = 0 ; $ sum_y = 0 ; $ sum_xy = 0 ; $ sum_x2 = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ sum_x += $ x [ $ i ] ; $ sum_y += $ y [ $ i ] ; $ sum_xy += $ x [ $ i ] * $ y [ $ i ] ; $ sum_x2 += ( $ x [ $ i ] * $ x [ $ i ] ) ; } $ m = ( $ n * $ sum_xy - $ sum_x * $ sum_y ) / ( $ n * $ sum_x2 - ( $ sum_x * $ sum_x ) ) ; $ c = ( $ sum_y - $ m * $ sum_x ) / $ n ; echo " m = " , ▁ $ m ; STRNEWLINE echo ▁ " c = " }
$ x = array ( 1 , 2 , 3 , 4 , 5 ) ; $ y = array ( 14 , 27 , 40 , 55 , 68 ) ; $ n = sizeof ( $ x ) ; bestApproximate ( $ x , $ y , $ n ) ; ? >
< ? php function printSorted ( $ arr , $ start , $ end ) { if ( $ start > $ end ) return ;
printSorted ( $ arr , $ start * 2 + 1 , $ end ) ;
echo ( $ arr [ $ start ] . " " ) ;
printSorted ( $ arr , $ start * 2 + 2 , $ end ) ; }
$ arr = array ( 4 , 2 , 5 , 1 , 3 ) ; printSorted ( $ arr , 0 , sizeof ( $ arr ) - 1 ) ;
< ? php function Identity ( $ num ) { $ row ; $ col ; for ( $ row = 0 ; $ row < $ num ; $ row ++ ) { for ( $ col = 0 ; $ col < $ num ; $ col ++ ) {
if ( $ row == $ col ) echo 1 , " ▁ " ; else echo 0 , " ▁ " ; } echo " " } return 0 ; }
$ size = 5 ; identity ( $ size ) ; ? >
< ? php function search ( & $ mat , $ n , $ x ) { $ i = 0 ;
$ j = $ n - 1 ; while ( $ i < $ n && $ j >= 0 ) { if ( $ mat [ $ i ] [ $ j ] == $ x ) { echo " n ▁ found ▁ at ▁ " . $ i . " , ▁ " . $ j ; return 1 ; } if ( $ mat [ $ i ] [ $ j ] > $ x ) $ j -- ;
else $ i ++ ; } echo " n ▁ Element ▁ not ▁ found " ;
return 0 ; }
$ mat = array ( array ( 10 , 20 , 30 , 40 ) , array ( 15 , 25 , 35 , 45 ) , array ( 27 , 29 , 37 , 48 ) , array ( 32 , 33 , 39 , 50 ) ) ; search ( $ mat , 4 , 29 ) ; ? >
< ? php function fill0X ( $ m , $ n ) {
$ k = 0 ; $ l = 0 ;
$ r = $ m ; $ c = $ n ;
$ x = ' X ' ;
while ( $ k < $ m && $ l < $ n ) {
for ( $ i = $ l ; $ i < $ n ; ++ $ i ) $ a [ $ k ] [ $ i ] = $ x ; $ k ++ ;
for ( $ i = $ k ; $ i < $ m ; ++ $ i ) $ a [ $ i ] [ $ n - 1 ] = $ x ; $ n -- ;
if ( $ k < $ m ) { for ( $ i = $ n - 1 ; $ i >= $ l ; -- $ i ) $ a [ $ m - 1 ] [ $ i ] = $ x ; $ m -- ; }
if ( $ l < $ n ) { for ( $ i = $ m - 1 ; $ i >= $ k ; -- $ i ) $ a [ $ i ] [ $ l ] = $ x ; $ l ++ ; }
$ x = ( $ x == '0' ) ? ' X ' : '0' ; }
for ( $ i = 0 ; $ i < $ r ; $ i ++ ) { for ( $ j = 0 ; $ j < $ c ; $ j ++ ) echo ( $ a [ $ i ] [ $ j ] . " ▁ " ) ; echo " STRNEWLINE " ; } }
echo " Output ▁ for ▁ m ▁ = ▁ 5 , ▁ n ▁ = ▁ 6 STRNEWLINE " ; fill0X ( 5 , 6 ) ; echo " Output for m = 4 , n = 4 " ; fill0X ( 4 , 4 ) ; echo " Output for m = 3 , n = 4 " ; fill0X ( 3 , 4 ) ; ? >
< ? php function findPeakUtil ( $ arr , $ low , $ high , $ n ) {
$ mid = $ low + ( $ high - $ low ) / 2 ;
if ( ( $ mid == 0 $ arr [ $ mid - 1 ] <= $ arr [ $ mid ] ) && ( $ mid == $ n - 1 $ arr [ $ mid + 1 ] <= $ arr [ $ mid ] ) ) return $ mid ;
else if ( $ mid > 0 && $ arr [ $ mid - 1 ] > $ arr [ $ mid ] ) return findPeakUtil ( $ arr , $ low , ( $ mid - 1 ) , $ n ) ;
else return ( findPeakUtil ( $ arr , ( $ mid + 1 ) , $ high , $ n ) ) ; }
function findPeak ( $ arr , $ n ) { return floor ( findPeakUtil ( $ arr , 0 , $ n - 1 , $ n ) ) ; }
$ arr = array ( 1 , 3 , 20 , 4 , 1 , 0 ) ; $ n = sizeof ( $ arr ) ; echo " Index ▁ of ▁ a ▁ peak ▁ point ▁ is ▁ " , findPeak ( $ arr , $ n ) ; ? >
< ? php function printRepeating ( $ arr , $ size ) { $ i ; $ j ; echo " ▁ Repeating ▁ elements ▁ are ▁ " ; for ( $ i = 0 ; $ i < $ size ; $ i ++ ) for ( $ j = $ i + 1 ; $ j < $ size ; $ j ++ ) if ( $ arr [ $ i ] == $ arr [ $ j ] ) echo $ arr [ $ i ] , " ▁ " ; }
$ arr = array ( 4 , 2 , 4 , 5 , 2 , 3 , 1 ) ; $ arr_size = sizeof ( $ arr , 0 ) ; printRepeating ( $ arr , $ arr_size ) ; ? >
< ? php function printRepeating ( $ arr , $ size ) { $ count = array_fill ( 0 , $ size , 0 ) ; echo " Repeated ▁ elements ▁ are ▁ " ; for ( $ i = 0 ; $ i < $ size ; $ i ++ ) { if ( $ count [ $ arr [ $ i ] ] == 1 ) echo $ arr [ $ i ] . " ▁ " ; else $ count [ $ arr [ $ i ] ] ++ ; } }
$ arr = array ( 4 , 2 , 4 , 5 , 2 , 3 , 1 ) ; $ arr_size = count ( $ arr ) ; printRepeating ( $ arr , $ arr_size ) ; ? >
< ? php function printRepeating ( $ arr , $ size ) {
$ S = 0 ;
$ P = 1 ;
$ x ; $ y ;
$ D ; $ n = $ size - 2 ;
for ( $ i = 0 ; $ i < $ size ; $ i ++ ) { $ S = $ S + $ arr [ $ i ] ; $ P = $ P * $ arr [ $ i ] ; }
$ S = $ S - $ n * ( $ n + 1 ) / 2 ;
$ P = $ P / fact ( $ n ) ;
$ D = sqrt ( $ S * $ S - 4 * $ P ) ; $ x = ( $ D + $ S ) / 2 ; $ y = ( $ S - $ D ) / 2 ; echo " The ▁ two ▁ Repeating ▁ elements ▁ are ▁ " . $ x . " & " }
function fact ( $ n ) { return ( $ n == 0 ) ? 1 : $ n * fact ( $ n - 1 ) ; }
$ arr = array ( 4 , 2 , 4 , 5 , 2 , 3 , 1 ) ; $ arr_size = count ( $ arr ) ; printRepeating ( $ arr , $ arr_size ) ; ? >
< ? php function printRepeating ( $ arr , $ size ) {
$ xor = $ arr [ 0 ] ;
$ set_bit_no ; $ i ; $ n = $ size - 2 ; $ x = 0 ; $ y = 0 ;
for ( $ i = 1 ; $ i < $ size ; $ i ++ ) $ xor ^= $ arr [ $ i ] ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ xor ^= $ i ;
$ set_bit_no = $ xor & ~ ( $ xor - 1 ) ;
for ( $ i = 0 ; $ i < $ size ; $ i ++ ) { if ( $ arr [ $ i ] & $ set_bit_no ) $ x = $ x ^ $ arr [ $ i ] ;
else $ y = $ y ^ $ arr [ $ i ] ;
} for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { if ( $ i & $ set_bit_no ) $ x = $ x ^ $ i ;
else $ y = $ y ^ $ i ; }
echo " n ▁ The ▁ two ▁ repeating ▁ elements ▁ are ▁ " ; echo $ y . " ▁ " . $ x ; } ? >
$ arr = array ( 4 , 2 , 4 , 5 , 2 , 3 , 1 ) ; $ arr_size = count ( $ arr ) ; printRepeating ( $ arr , $ arr_size ) ;
< ? php function printRepeating ( $ arr , $ size ) { $ i ; echo " The ▁ repeating ▁ elements ▁ are " , " ▁ " ; for ( $ i = 0 ; $ i < $ size ; $ i ++ ) { if ( $ arr [ abs ( $ arr [ $ i ] ) ] > 0 ) $ arr [ abs ( $ arr [ $ i ] ) ] = - $ arr [ abs ( $ arr [ $ i ] ) ] ; else echo abs ( $ arr [ $ i ] ) , " ▁ " ; } }
$ arr = array ( 4 , 2 , 4 , 5 , 2 , 3 , 1 ) ; $ arr_size = sizeof ( $ arr ) ; printRepeating ( $ arr , $ arr_size ) ; #This  code is contributed by aj_36 NEW_LINE ? >
< ? php function subArraySum ( $ arr , $ n , $ sum ) { $ curr_sum ; $ i ; $ j ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ curr_sum = $ arr [ $ i ] ;
for ( $ j = $ i + 1 ; $ j <= $ n ; $ j ++ ) { if ( $ curr_sum == $ sum ) { echo " Sum ▁ found ▁ between ▁ indexes ▁ " , $ i , " ▁ and ▁ " , $ j - 1 ; return 1 ; } if ( $ curr_sum > $ sum $ j == $ n ) break ; $ curr_sum = $ curr_sum + $ arr [ $ j ] ; } } echo " No ▁ subarray ▁ found " ; return 0 ; }
$ arr = array ( 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 ) ; $ n = sizeof ( $ arr ) ; $ sum = 23 ; subArraySum ( $ arr , $ n , $ sum ) ; return 0 ; ? >
< ? php function subArraySum ( $ arr , $ n , $ sum ) {
$ curr_sum = $ arr [ 0 ] ; $ start = 0 ; $ i ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) {
while ( $ curr_sum > $ sum and $ start < $ i - 1 ) { $ curr_sum = $ curr_sum - $ arr [ $ start ] ; $ start ++ ; }
if ( $ curr_sum == $ sum ) { echo " Sum ▁ found ▁ between ▁ indexes " , " ▁ " , $ start , " ▁ " , " and ▁ " , " ▁ " , $ i - 1 ; return 1 ; }
if ( $ i < $ n ) $ curr_sum = $ curr_sum + $ arr [ $ i ] ; }
echo " No ▁ subarray ▁ found " ; return 0 ; }
$ arr = array ( 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 ) ; $ n = count ( $ arr ) ; $ sum = 23 ; subArraySum ( $ arr , $ n , $ sum ) ;
< ? php function find3Numbers ( $ A , $ arr_size , $ sum ) { $ l ; $ r ;
for ( $ i = 0 ; $ i < $ arr_size - 2 ; $ i ++ ) {
for ( $ j = $ i + 1 ; $ j < $ arr_size - 1 ; $ j ++ ) {
for ( $ k = $ j + 1 ; $ k < $ arr_size ; $ k ++ ) { if ( $ A [ $ i ] + $ A [ $ j ] + $ A [ $ k ] == $ sum ) { echo " Triplet ▁ is " , " ▁ " , $ A [ $ i ] , " , ▁ " , $ A [ $ j ] , " , ▁ " , $ A [ $ k ] ; return true ; } } } }
return false ; }
$ A = array ( 1 , 4 , 45 , 6 , 10 , 8 ) ; $ sum = 22 ; $ arr_size = sizeof ( $ A ) ; find3Numbers ( $ A , $ arr_size , $ sum ) ; ? >
< ? php function search ( $ arr , $ n , $ x ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ arr [ $ i ] == $ x ) return $ i ; } return -1 ; }
$ arr = array ( 1 , 10 , 30 , 15 ) ; $ x = 30 ; $ n = sizeof ( $ arr ) ; echo $ x . " ▁ is ▁ present ▁ at ▁ index ▁ " . search ( $ arr , $ n , $ x ) ;
< ? php function binarySearch ( $ arr , $ l , $ r , $ x ) { if ( $ r >= $ l ) { $ mid = ceil ( $ l + ( $ r - $ l ) / 2 ) ;
if ( $ arr [ $ mid ] == $ x ) return floor ( $ mid ) ;
if ( $ arr [ $ mid ] > $ x ) return binarySearch ( $ arr , $ l , $ mid - 1 , $ x ) ;
return binarySearch ( $ arr , $ mid + 1 , $ r , $ x ) ; }
return -1 ; }
$ arr = array ( 2 , 3 , 4 , 10 , 40 ) ; $ n = count ( $ arr ) ; $ x = 10 ; $ result = binarySearch ( $ arr , 0 , $ n - 1 , $ x ) ; if ( ( $ result == -1 ) ) echo " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ; else echo " Element ▁ is ▁ present ▁ at ▁ index ▁ " , $ result ; ? >
< ? php function binarySearch ( $ arr , $ l , $ r , $ x ) { while ( $ l <= $ r ) { $ m = $ l + ( $ r - $ l ) / 2 ;
if ( $ arr [ $ m ] == $ x ) return floor ( $ m ) ;
if ( $ arr [ $ m ] < $ x ) $ l = $ m + 1 ;
else $ r = $ m - 1 ; }
return -1 ; }
$ arr = array ( 2 , 3 , 4 , 10 , 40 ) ; $ n = count ( $ arr ) ; $ x = 10 ; $ result = binarySearch ( $ arr , 0 , $ n - 1 , $ x ) ; if ( ( $ result == -1 ) ) echo " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ; else echo " Element ▁ is ▁ present ▁ at ▁ index ▁ " , $ result ; ? >
< ? php function swap ( & $ a , & $ b ) { $ t = $ a ; $ a = $ b ; $ b = $ t ; }
function partition ( & $ arr , $ l , $ h ) { $ x = $ arr [ $ h ] ; $ i = ( $ l - 1 ) ; for ( $ j = $ l ; $ j <= $ h - 1 ; $ j ++ ) { if ( $ arr [ $ j ] <= $ x ) { $ i ++ ; swap ( $ arr [ $ i ] , $ arr [ $ j ] ) ; } } swap ( $ arr [ $ i + 1 ] , $ arr [ $ h ] ) ; return ( $ i + 1 ) ; }
function quickSortIterative ( & $ arr , $ l , $ h ) {
$ stack = array_fill ( 0 , $ h - $ l + 1 , 0 ) ;
$ top = -1 ;
$ stack [ ++ $ top ] = $ l ; $ stack [ ++ $ top ] = $ h ;
while ( $ top >= 0 ) {
$ h = $ stack [ $ top -- ] ; $ l = $ stack [ $ top -- ] ;
$ p = partition ( $ arr , $ l , $ h ) ;
if ( $ p - 1 > $ l ) { $ stack [ ++ $ top ] = $ l ; $ stack [ ++ $ top ] = $ p - 1 ; }
if ( $ p + 1 < $ h ) { $ stack [ ++ $ top ] = $ p + 1 ; $ stack [ ++ $ top ] = $ h ; } } }
function printArr ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; ++ $ i ) echo $ arr [ $ i ] . " ▁ " ; }
$ arr = array ( 4 , 3 , 5 , 2 , 1 , 3 , 2 , 3 ) ; $ n = count ( $ arr ) ;
quickSortIterative ( $ arr , 0 , $ n - 1 ) ; printArr ( $ arr , $ n ) ; ? >
< ? php function printMaxActivities ( $ s , $ f , $ n ) { echo " Following ▁ activities ▁ are ▁ selected ▁ " . " STRNEWLINE " ;
$ i = 0 ; echo $ i . " " ;
for ( $ j = 1 ; $ j < $ n ; $ j ++ ) {
if ( $ s [ $ j ] >= $ f [ $ i ] ) { echo $ j . " " ; $ i = $ j ; } } }
$ s = array ( 1 , 3 , 0 , 5 , 8 , 5 ) ; $ f = array ( 2 , 4 , 6 , 7 , 9 , 9 ) ; $ n = sizeof ( $ s ) ; printMaxActivities ( $ s , $ f , $ n ) ; ? >
< ? php function lcs ( $ X , $ Y , $ m , $ n ) { if ( $ m == 0 $ n == 0 ) return 0 ; else if ( $ X [ $ m - 1 ] == $ Y [ $ n - 1 ] ) return 1 + lcs ( $ X , $ Y , $ m - 1 , $ n - 1 ) ; else return max ( lcs ( $ X , $ Y , $ m , $ n - 1 ) , lcs ( $ X , $ Y , $ m - 1 , $ n ) ) ; }
$ X = " AGGTAB " ; $ Y = " GXTXAYB " ; echo " Length ▁ of ▁ LCS ▁ is ▁ " ; echo lcs ( $ X , $ Y , strlen ( $ X ) , strlen ( $ Y ) ) ; ? >
< ? php function lcs ( $ X , $ Y ) {
$ m = strlen ( $ X ) ; $ n = strlen ( $ Y ) ;
for ( $ i = 0 ; $ i <= $ m ; $ i ++ ) { for ( $ j = 0 ; $ j <= $ n ; $ j ++ ) { if ( $ i == 0 $ j == 0 ) $ L [ $ i ] [ $ j ] = 0 ; else if ( $ X [ $ i - 1 ] == $ Y [ $ j - 1 ] ) $ L [ $ i ] [ $ j ] = $ L [ $ i - 1 ] [ $ j - 1 ] + 1 ; else $ L [ $ i ] [ $ j ] = max ( $ L [ $ i - 1 ] [ $ j ] , $ L [ $ i ] [ $ j - 1 ] ) ; } }
return $ L [ $ m ] [ $ n ] ; }
$ X = " AGGTAB " ; $ Y = " GXTXAYB " ; echo " Length ▁ of ▁ LCS ▁ is ▁ " ; echo lcs ( $ X , $ Y ) ; ? >
< ? php $ R = 3 ; $ C = 3 ;
function min1 ( $ x , $ y , $ z ) { if ( $ x < $ y ) return ( $ x < $ z ) ? $ x : $ z ; else return ( $ y < $ z ) ? $ y : $ z ; }
function minCost ( $ cost , $ m , $ n ) { global $ R ; global $ C ; if ( $ n < 0 $ m < 0 ) return PHP_INT_MAX ; else if ( $ m == 0 && $ n == 0 ) return $ cost [ $ m ] [ $ n ] ; else return $ cost [ $ m ] [ $ n ] + min1 ( minCost ( $ cost , $ m - 1 , $ n - 1 ) , minCost ( $ cost , $ m - 1 , $ n ) , minCost ( $ cost , $ m , $ n - 1 ) ) ; }
$ cost = array ( array ( 1 , 2 , 3 ) , array ( 4 , 8 , 2 ) , array ( 1 , 5 , 3 ) ) ; echo minCost ( $ cost , 2 , 2 ) ; ? >
< ? php $ R = 3 ; $ C = 3 ; function minCost ( $ cost , $ m , $ n ) { global $ R ; global $ C ;
$ tc ; for ( $ i = 0 ; $ i <= $ R ; $ i ++ ) for ( $ j = 0 ; $ j <= $ C ; $ j ++ ) $ tc [ $ i ] [ $ j ] = 0 ; $ tc [ 0 ] [ 0 ] = $ cost [ 0 ] [ 0 ] ;
for ( $ i = 1 ; $ i <= $ m ; $ i ++ ) $ tc [ $ i ] [ 0 ] = $ tc [ $ i - 1 ] [ 0 ] + $ cost [ $ i ] [ 0 ] ;
for ( $ j = 1 ; $ j <= $ n ; $ j ++ ) $ tc [ 0 ] [ $ j ] = $ tc [ 0 ] [ $ j - 1 ] + $ cost [ 0 ] [ $ j ] ;
for ( $ i = 1 ; $ i <= $ m ; $ i ++ ) for ( $ j = 1 ; $ j <= $ n ; $ j ++ )
$ tc [ $ i ] [ $ j ] = min ( $ tc [ $ i - 1 ] [ $ j - 1 ] , $ tc [ $ i - 1 ] [ $ j ] , $ tc [ $ i ] [ $ j - 1 ] ) + $ cost [ $ i ] [ $ j ] ; return $ tc [ $ m ] [ $ n ] ; }
$ cost = array ( array ( 1 , 2 , 3 ) , array ( 4 , 8 , 2 ) , array ( 1 , 5 , 3 ) ) ; echo minCost ( $ cost , 2 , 2 ) ; ? >
< ? php function knapSack ( $ W , $ wt , $ val , $ n ) {
if ( $ n == 0 $ W == 0 ) return 0 ;
if ( $ wt [ $ n - 1 ] > $ W ) return knapSack ( $ W , $ wt , $ val , $ n - 1 ) ;
else return max ( $ val [ $ n - 1 ] + knapSack ( $ W - $ wt [ $ n - 1 ] , $ wt , $ val , $ n - 1 ) , knapSack ( $ W , $ wt , $ val , $ n -1 ) ) ; }
$ val = array ( 60 , 100 , 120 ) ; $ wt = array ( 10 , 20 , 30 ) ; $ W = 50 ; $ n = count ( $ val ) ; echo knapSack ( $ W , $ wt , $ val , $ n ) ; ? >
< ? php function knapSack ( $ W , $ wt , $ val , $ n ) { $ K = array ( array ( ) ) ;
for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) { for ( $ w = 0 ; $ w <= $ W ; $ w ++ ) { if ( $ i == 0 $ w == 0 ) $ K [ $ i ] [ $ w ] = 0 ; else if ( $ wt [ $ i - 1 ] <= $ w ) $ K [ $ i ] [ $ w ] = max ( $ val [ $ i - 1 ] + $ K [ $ i - 1 ] [ $ w - $ wt [ $ i - 1 ] ] , $ K [ $ i - 1 ] [ $ w ] ) ; else $ K [ $ i ] [ $ w ] = $ K [ $ i - 1 ] [ $ w ] ; } } return $ K [ $ n ] [ $ W ] ; }
$ val = array ( 60 , 100 , 120 ) ; $ wt = array ( 10 , 20 , 30 ) ; $ W = 50 ; $ n = count ( $ val ) ; echo knapSack ( $ W , $ wt , $ val , $ n ) ; ? >
< ? php function lps ( $ seq , $ i , $ j ) {
if ( $ i == $ j ) return 1 ;
if ( $ seq [ $ i ] == $ seq [ $ j ] && $ i + 1 == $ j ) return 2 ;
if ( $ seq [ $ i ] == $ seq [ $ j ] ) return lps ( $ seq , $ i + 1 , $ j - 1 ) + 2 ;
return max ( lps ( $ seq , $ i , $ j - 1 ) , lps ( $ seq , $ i + 1 , $ j ) ) ; }
$ seq = " GEEKSFORGEEKS " ; $ n = strlen ( $ seq ) ; echo " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ " . lps ( $ seq , 0 , $ n - 1 ) ; ? >
< ? php function counts ( $ n ) {
for ( $ j = 0 ; $ j < $ n + 1 ; $ j ++ ) $ table [ $ j ] = 0 ;
$ table [ 0 ] = 1 ;
for ( $ i = 3 ; $ i <= $ n ; $ i ++ ) $ table [ $ i ] += $ table [ $ i - 3 ] ; for ( $ i = 5 ; $ i <= $ n ; $ i ++ ) $ table [ $ i ] += $ table [ $ i - 5 ] ; for ( $ i = 10 ; $ i <= $ n ; $ i ++ ) $ table [ $ i ] += $ table [ $ i - 10 ] ; return $ table [ $ n ] ; }
$ n = 20 ; echo " Count ▁ for ▁ " ; echo ( $ n ) ; echo ( " ▁ is ▁ " ) ; echo counts ( $ n ) ; $ n = 13 ; echo ( " STRNEWLINE " ) ; echo " Count ▁ for ▁ " ; echo ( $ n ) ; echo ( " ▁ is ▁ " ) ; echo counts ( $ n ) ; ? >
< ? php function search ( $ pat , $ txt ) { $ M = strlen ( $ pat ) ; $ N = strlen ( $ txt ) ;
for ( $ i = 0 ; $ i <= $ N - $ M ; $ i ++ ) {
for ( $ j = 0 ; $ j < $ M ; $ j ++ ) if ( $ txt [ $ i + $ j ] != $ pat [ $ j ] ) break ;
if ( $ j == $ M ) echo " Pattern ▁ found ▁ at ▁ index ▁ " , $ i . " STRNEWLINE " ; } }
$ txt = " AABAACAADAABAAABAA " ; $ pat = " AABA " ; search ( $ pat , $ txt ) ; ? >
< ? php $ d = 256 ;
function search ( $ pat , $ txt , $ q ) { $ M = strlen ( $ pat ) ; $ N = strlen ( $ txt ) ; $ i ; $ j ;
$ p = 0 ;
$ t = 0 ; $ h = 1 ; $ d = 1 ;
for ( $ i = 0 ; $ i < $ M - 1 ; $ i ++ ) $ h = ( $ h * $ d ) % $ q ;
for ( $ i = 0 ; $ i < $ M ; $ i ++ ) { $ p = ( $ d * $ p + $ pat [ $ i ] ) % $ q ; $ t = ( $ d * $ t + $ txt [ $ i ] ) % $ q ; }
for ( $ i = 0 ; $ i <= $ N - $ M ; $ i ++ ) {
if ( $ p == $ t ) {
for ( $ j = 0 ; $ j < $ M ; $ j ++ ) { if ( $ txt [ $ i + $ j ] != $ pat [ $ j ] ) break ; }
if ( $ j == $ M ) echo " Pattern ▁ found ▁ at ▁ index ▁ " , $ i , " STRNEWLINE " ; }
if ( $ i < $ N - $ M ) { $ t = ( $ d * ( $ t - $ txt [ $ i ] * $ h ) + $ txt [ $ i + $ M ] ) % $ q ;
if ( $ t < 0 ) $ t = ( $ t + $ q ) ; } } }
$ txt = " GEEKS ▁ FOR ▁ GEEKS " ; $ pat = " GEEK " ;
$ q = 101 ;
search ( $ pat , $ txt , $ q ) ; ? >
< ? php function search ( $ pat , $ txt ) { $ M = strlen ( $ pat ) ; $ N = strlen ( $ txt ) ; $ i = 0 ; while ( $ i <= $ N - $ M ) { $ j ;
for ( $ j = 0 ; $ j < $ M ; $ j ++ ) if ( $ txt [ $ i + $ j ] != $ pat [ $ j ] ) break ;
if ( $ j == $ M ) { echo ( " Pattern ▁ found ▁ at ▁ index ▁ $ i " . " STRNEWLINE " ) ; $ i = $ i + $ M ; } else if ( $ j == 0 ) $ i = $ i + 1 ; else
$ i = $ i + $ j ; } }
$ txt = " ABCEABCDABCEABCD " ; $ pat = " ABCD " ; search ( $ pat , $ txt ) ; ? >
< ? php function getMedian ( $ ar1 , $ ar2 , $ n ) { $ i = 0 ; $ j = 0 ; $ count ; $ m1 = -1 ; $ m2 = -1 ;
for ( $ count = 0 ; $ count <= $ n ; $ count ++ ) {
if ( $ i == $ n ) { $ m1 = $ m2 ; $ m2 = $ ar2 [ 0 ] ; break ; }
else if ( $ j == $ n ) { $ m1 = $ m2 ; $ m2 = $ ar1 [ 0 ] ; break ; }
if ( $ ar1 [ $ i ] <= $ ar2 [ $ j ] ) {
$ m1 = $ m2 ; $ m2 = $ ar1 [ $ i ] ; $ i ++ ; } else {
$ m1 = $ m2 ; $ m2 = $ ar2 [ $ j ] ; $ j ++ ; } } return ( $ m1 + $ m2 ) / 2 ; }
$ ar1 = array ( 1 , 12 , 15 , 26 , 38 ) ; $ ar2 = array ( 2 , 13 , 17 , 30 , 45 ) ; $ n1 = sizeof ( $ ar1 ) ; $ n2 = sizeof ( $ ar2 ) ; if ( $ n1 == $ n2 ) echo ( " Median ▁ is ▁ " . getMedian ( $ ar1 , $ ar2 , $ n1 ) ) ; else echo ( " Doesn ' t ▁ work ▁ for ▁ arrays " . " of ▁ unequal ▁ size " ) ; ? >
< ? php function isLucky ( $ n ) { $ counter = 2 ;
$ next_position = $ n ; if ( $ counter > $ n ) return 1 ; if ( $ n % $ counter == 0 ) return 0 ;
$ next_position -= $ next_position / $ counter ; $ counter ++ ; return isLucky ( $ next_position ) ; }
$ x = 5 ; if ( isLucky ( $ x ) ) echo $ x , " ▁ is ▁ a ▁ lucky ▁ no . " ; else echo $ x , " ▁ is ▁ not ▁ a ▁ lucky ▁ no . " ; ? >
< ? php function poww ( $ a , $ b ) { if ( $ b == 0 ) return 1 ; $ answer = $ a ; $ increment = $ a ; $ i ; $ j ; for ( $ i = 1 ; $ i < $ b ; $ i ++ ) { for ( $ j = 1 ; $ j < $ a ; $ j ++ ) { $ answer += $ increment ; } $ increment = $ answer ; } return $ answer ; }
echo ( poww ( 5 , 3 ) ) ; ? >
< ? php function multiply ( $ x , $ y ) { if ( $ y ) return ( $ x + multiply ( $ x , $ y - 1 ) ) ; else return 0 ; }
function p_ow ( $ a , $ b ) { if ( $ b ) return multiply ( $ a , p_ow ( $ a , $ b - 1 ) ) ; else return 1 ; }
echo pow ( 5 , 3 ) ; ? >
< ? php function count1 ( $ n ) {
if ( $ n < 3 ) return $ n ; if ( $ n >= 3 && $ n < 10 ) return $ n - 1 ;
$ po = 1 ; for ( $ x = intval ( $ n / $ po ) ; $ x > 9 ; $ x = intval ( $ n / $ po ) ) $ po = $ po * 10 ;
$ msd = intval ( $ n / $ po ) ; if ( $ msd != 3 )
return count1 ( $ msd ) * count1 ( $ po - 1 ) + count1 ( $ msd ) + count1 ( $ n % $ po ) ; else
return count1 ( $ msd * $ po - 1 ) ; }
echo count1 ( 578 ) ; ? >
< ? php function fact ( $ n ) { return ( $ n <= 1 ) ? 1 : $ n * fact ( $ n - 1 ) ; }
function findSmallerInRight ( $ str , $ low , $ high ) { $ countRight = 0 ; for ( $ i = $ low + 1 ; $ i <= $ high ; ++ $ i ) if ( $ str [ $ i ] < $ str [ $ low ] ) ++ $ countRight ; return $ countRight ; }
function findRank ( $ str ) { $ len = strlen ( $ str ) ; $ mul = fact ( $ len ) ; $ rank = 1 ; for ( $ i = 0 ; $ i < $ len ; ++ $ i ) { $ mul /= $ len - $ i ;
$ countRight = findSmallerInRight ( $ str , $ i , $ len - 1 ) ; $ rank += $ countRight * $ mul ; } return $ rank ; }
$ str = " string " ; echo findRank ( $ str ) ; ? >
< ? php $ MAX_CHAR = 256 ;
$ count = array_fill ( 0 , $ MAX_CHAR + 1 , 0 ) ;
function fact ( $ n ) { return ( $ n <= 1 ) ? 1 : $ n * fact ( $ n - 1 ) ; }
function populateAndIncreaseCount ( & $ count , $ str ) { global $ MAX_CHAR ; for ( $ i = 0 ; $ i < strlen ( $ str ) ; ++ $ i ) ++ $ count [ ord ( $ str [ $ i ] ) ] ; for ( $ i = 1 ; $ i < $ MAX_CHAR ; ++ $ i ) $ count [ $ i ] += $ count [ $ i - 1 ] ; }
function updatecount ( & $ count , $ ch ) { global $ MAX_CHAR ; for ( $ i = ord ( $ ch ) ; $ i < $ MAX_CHAR ; ++ $ i ) -- $ count [ $ i ] ; }
function findRank ( $ str ) { global $ MAX_CHAR ; $ len = strlen ( $ str ) ; $ mul = fact ( $ len ) ; $ rank = 1 ;
populateAndIncreaseCount ( $ count , $ str ) ; for ( $ i = 0 ; $ i < $ len ; ++ $ i ) { $ mul = ( int ) ( $ mul / ( $ len - $ i ) ) ;
$ rank += $ count [ ord ( $ str [ $ i ] ) - 1 ] * $ mul ;
updatecount ( $ count , $ str [ $ i ] ) ; } return $ rank ; }
$ str = " string " ; echo findRank ( $ str ) ; ? >
< ? php function exponential ( $ n , $ x ) {
$ sum = 1.0 ; for ( $ i = $ n - 1 ; $ i > 0 ; -- $ i ) $ sum = 1 + $ x * $ sum / $ i ; return $ sum ; }
$ n = 10 ; $ x = 1.0 ; echo ( " e ^ x ▁ = ▁ " . exponential ( $ n , $ x ) ) ; ? >
< ? php function mintwo ( $ x , $ y ) { return ( $ x < $ y ) ? $ x : $ y ; }
function calcAngle ( $ h , $ m ) {
if ( $ h < 0 $ m < 0 $ h > 12 $ m > 60 ) echo " Wrong ▁ input " ; if ( $ h == 12 ) $ h = 0 ; if ( $ m == 60 ) { $ m = 0 ; $ h += 1 ; if ( $ h > 12 ) $ h = $ h - 12 ; }
$ hour_angle = 0.5 * ( $ h * 60 + $ m ) ; $ minute_angle = 6 * $ m ;
$ angle = abs ( $ hour_angle - $ minute_angle ) ;
$ angle = min ( 360 - $ angle , $ angle ) ; return $ angle ; }
echo calcAngle ( 9 , 60 ) , " STRNEWLINE " ; echo calcAngle ( 3 , 30 ) , " STRNEWLINE " ; ? >
< ? php function getSingle ( $ arr , $ n ) { $ ones = 0 ; $ twos = 0 ; $ common_bit_mask ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
$ twos = $ twos | ( $ ones & $ arr [ $ i ] ) ;
$ ones = $ ones ^ $ arr [ $ i ] ;
$ common_bit_mask = ~ ( $ ones & $ twos ) ;
$ ones &= $ common_bit_mask ;
$ twos &= $ common_bit_mask ; } return $ ones ; }
$ arr = array ( 3 , 3 , 2 , 3 ) ; $ n = sizeof ( $ arr ) ; echo " The ▁ element ▁ with ▁ single ▁ " . " occurrence ▁ is ▁ " , getSingle ( $ arr , $ n ) ; ? >
< ? php $ INT_SIZE = 32 ; function getSingle ( $ arr , $ n ) { global $ INT_SIZE ;
$ result = 0 ; $ x ; $ sum ;
for ( $ i = 0 ; $ i < $ INT_SIZE ; $ i ++ ) {
$ sum = 0 ; $ x = ( 1 << $ i ) ; for ( $ j = 0 ; $ j < $ n ; $ j ++ ) { if ( $ arr [ $ j ] & $ x ) $ sum ++ ; }
if ( ( $ sum % 3 ) != 0 ) $ result |= $ x ; } return $ result ; }
$ arr = array ( 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 ) ; $ n = sizeof ( $ arr ) ; echo " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ " , getSingle ( $ arr , $ n ) ; ? >
< ? php function swapBits ( $ x , $ p1 , $ p2 , $ n ) {
$ set1 = ( $ x >> $ p1 ) & ( ( 1 << $ n ) - 1 ) ;
$ set2 = ( $ x >> $ p2 ) & ( ( 1 << $ n ) - 1 ) ;
$ xor = ( $ set1 ^ $ set2 ) ;
$ xor = ( $ xor << $ p1 ) | ( $ xor << $ p2 ) ;
$ result = $ x ^ $ xor ; return $ result ; }
$ res = swapBits ( 28 , 0 , 3 , 2 ) ; echo " Result = " ? >
< ? php function smallest ( $ x , $ y , $ z ) { $ c = 0 ; while ( $ x && $ y && $ z ) { $ x -- ; $ y -- ; $ z -- ; $ c ++ ; } return $ c ; }
$ x = 12 ; $ y = 15 ; $ z = 5 ; echo " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ " . smallest ( $ x , $ y , $ z ) ; ? >
< ? php function addOne ( $ x ) { $ m = 1 ;
while ( $ x & $ m ) { $ x = $ x ^ $ m ; $ m <<= 1 ; }
$ x = $ x ^ $ m ; return $ x ; }
echo addOne ( 13 ) ; ? >
< ? php function addOne ( $ x ) { return ( - ( ~ $ x ) ) ; }
echo addOne ( 13 ) ; ? >
< ? php function fun ( $ n ) { return $ n & ( $ n - 1 ) ; }
$ n = 7 ; echo " The ▁ number ▁ after ▁ unsetting ▁ the " . " ▁ rightmost ▁ set ▁ bit ▁ " , fun ( $ n ) ; ? >
< ? php function isPowerOfFour ( $ n ) { if ( $ n == 0 ) return 0 ; while ( $ n != 1 ) { if ( $ n % 4 != 0 ) return 0 ; $ n = $ n / 4 ; } return 1 ; }
$ test_no = 64 ; if ( isPowerOfFour ( $ test_no ) ) echo $ test_no , " ▁ is ▁ a ▁ power ▁ of ▁ 4" ; else echo $ test_no , " ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" ; ? >
< ? php function isPowerOfFour ( $ n ) { $ count = 0 ;
if ( $ n && ! ( $ n & ( $ n - 1 ) ) ) {
while ( $ n > 1 ) { $ n >>= 1 ; $ count += 1 ; }
return ( $ count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
$ test_no = 64 ; if ( isPowerOfFour ( $ test_no ) ) echo $ test_no , " ▁ is ▁ a ▁ power ▁ of ▁ 4" ; else echo $ test_no , " ▁ not ▁ is ▁ a ▁ power ▁ of ▁ 4" ; ? >
< ? php function m_in ( $ x , $ y ) { return $ y ^ ( ( $ x ^ $ y ) & - ( $ x < $ y ) ) ; }
function m_ax ( $ x , $ y ) { return $ x ^ ( ( $ x ^ $ y ) & - ( $ x < $ y ) ) ; }
$ x = 15 ; $ y = 6 ; echo " Minimum ▁ of " , " ▁ " , $ x , " ▁ " , " and " , " ▁ " , $ y , " ▁ " , " ▁ is ▁ " , " ▁ " ; echo m_in ( $ x , $ y ) ; echo " Maximum of " , " " , $ x , " " , STRNEWLINE " and " , " " , $ y , " " , ▁ " is " echo m_ax ( $ x , $ y ) ; ? >
< ? php function getFirstSetBitPos ( $ n ) { return ceil ( log ( ( $ n & - $ n ) + 1 , 2 ) ) ; }
$ n = 12 ; echo getFirstSetBitPos ( $ n ) ; ? >
< ? php function swapBits ( $ x ) {
$ even_bits = $ x & 0xAAAAAAAA ;
$ odd_bits = $ x & 0x55555555 ;
$ even_bits >>= 1 ;
$ odd_bits <<= 1 ;
return ( $ even_bits $ odd_bits ) ; }
$ x = 23 ;
echo swapBits ( $ x ) ; ? >
< ? php function isPowerOfTwo ( $ n ) { return $ n && ( ! ( $ n & ( $ n - 1 ) ) ) ; }
function findPosition ( $ n ) { if ( ! isPowerOfTwo ( $ n ) ) return -1 ; $ i = 1 ; $ pos = 1 ;
while ( ! ( $ i & $ n ) ) {
$ i = $ i << 1 ;
++ $ pos ; } return $ pos ; }
$ n = 16 ; $ pos = findPosition ( $ n ) ; if ( ( $ pos == -1 ) == true ) echo " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Invalid number " , ▁ " " ; STRNEWLINE else STRNEWLINE echo ▁ " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Position " , ▁ $ pos , ▁ " " $ n = 12 ; $ pos = findPosition ( $ n ) ; if ( ( $ pos == -1 ) == true ) echo " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Invalid number " , ▁ " " ; STRNEWLINE else STRNEWLINE echo ▁ " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Position " , ▁ $ pos , ▁ " " $ n = 128 ; $ pos = findPosition ( $ n ) ; if ( ( $ pos == -1 ) == true ) echo " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Invalid number " , ▁ " " ; STRNEWLINE else STRNEWLINE echo ▁ " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Position " , ▁ $ pos , ▁ " " ? >
< ? php function isPowerOfTwo ( $ n ) { return $ n && ( ! ( $ n & ( $ n - 1 ) ) ) ; }
function findPosition ( $ n ) { if ( ! isPowerOfTwo ( $ n ) ) return -1 ; $ count = 0 ;
while ( $ n ) { $ n = $ n >> 1 ;
++ $ count ; } return $ count ; }
$ n = 0 ; $ pos = findPosition ( $ n ) ; if ( ( $ pos == -1 ) == true ) echo " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Invalid number " , ▁ " " ; STRNEWLINE else STRNEWLINE echo ▁ " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Position " , ▁ $ pos , ▁ " " $ n = 12 ; $ pos = findPosition ( $ n ) ; if ( ( $ pos == -1 ) == true ) echo " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Invalid number " , ▁ " " ; STRNEWLINE else STRNEWLINE echo ▁ " n = " , ▁ $ n , STRNEWLINE " Position " , ▁ $ pos , ▁ " " $ n = 128 ; $ pos = findPosition ( $ n ) ; if ( ( $ pos == -1 ) == true ) echo " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Invalid number " , ▁ " " ; STRNEWLINE else STRNEWLINE echo ▁ " n = " , ▁ $ n , ▁ " , " , STRNEWLINE " Position " , ▁ $ pos , ▁ " " ? >
< ? php $ x = 10 ; $ y = 5 ;
$ x = $ x * $ y ;
$ y = $ x / $ y ;
$ x = $ x / $ y ; echo " After ▁ Swapping : ▁ x ▁ = ▁ " , $ x , " ▁ " , " y ▁ = ▁ " , $ y ; ? >
< ? php $ x = 10 ; $ y = 5 ;
$ x = $ x ^ $ y ;
$ y = $ x ^ $ y ;
$ x = $ x ^ $ y ; echo " After ▁ Swapping : ▁ x ▁ = ▁ " , $ x , " , ▁ " , " y ▁ = ▁ " , $ y ; ? >
< ? php function swap ( & $ xp , & $ yp ) { $ xp = $ xp ^ $ yp ; $ yp = $ xp ^ $ yp ; $ xp = $ xp ^ $ yp ; }
$ x = 10 ; swap ( $ x , $ x ) ; print ( " After ▁ swap ( & x , ▁ & x ) : ▁ x ▁ = ▁ " . $ x ) ; ? >
< ? php function nextGreatest ( & $ arr , $ size ) {
$ max_from_right = $ arr [ $ size - 1 ] ;
$ arr [ $ size - 1 ] = -1 ;
for ( $ i = $ size - 2 ; $ i >= 0 ; $ i -- ) {
$ temp = $ arr [ $ i ] ;
$ arr [ $ i ] = $ max_from_right ;
if ( $ max_from_right < $ temp ) $ max_from_right = $ temp ; } }
function printArray ( $ arr , $ size ) { for ( $ i = 0 ; $ i < $ size ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; echo " STRNEWLINE " ; }
$ arr = array ( 16 , 17 , 4 , 3 , 5 , 2 ) ; $ size = count ( $ arr ) ; nextGreatest ( $ arr , $ size ) ; echo " The ▁ modified ▁ array ▁ is : ▁ STRNEWLINE " ; printArray ( $ arr , $ size ) ; ? >
< ? php function maxDiff ( $ arr , $ arr_size ) { $ max_diff = $ arr [ 1 ] - $ arr [ 0 ] ; for ( $ i = 0 ; $ i < $ arr_size ; $ i ++ ) { for ( $ j = $ i + 1 ; $ j < $ arr_size ; $ j ++ ) { if ( $ arr [ $ j ] - $ arr [ $ i ] > $ max_diff ) $ max_diff = $ arr [ $ j ] - $ arr [ $ i ] ; } } return $ max_diff ; }
$ arr = array ( 1 , 2 , 90 , 10 , 110 ) ; $ n = sizeof ( $ arr ) ;
echo " Maximum ▁ difference ▁ is ▁ " . maxDiff ( $ arr , $ n ) ;
< ? php function findMaximum ( $ arr , $ low , $ high ) { $ max = $ arr [ $ low ] ; $ i ; for ( $ i = $ low ; $ i <= $ high ; $ i ++ ) { if ( $ arr [ $ i ] > $ max ) $ max = $ arr [ $ i ] ; } return $ max ; }
$ arr = array ( 1 , 30 , 40 , 50 , 60 , 70 , 23 , 20 ) ; $ n = count ( $ arr ) ; echo " The ▁ maximum ▁ element ▁ is ▁ " , findMaximum ( $ arr , 0 , $ n - 1 ) ; ? >
< ? php function findMaximum ( $ arr , $ low , $ high ) {
if ( $ low == $ high ) return $ arr [ $ low ] ;
if ( ( $ high == $ low + 1 ) && $ arr [ $ low ] >= $ arr [ $ high ] ) return $ arr [ $ low ] ;
if ( ( $ high == $ low + 1 ) && $ arr [ $ low ] < $ arr [ $ high ] ) return $ arr [ $ high ] ; $ mid = ( $ low + $ high ) / 2 ;
if ( $ arr [ $ mid ] > $ arr [ $ mid + 1 ] && $ arr [ $ mid ] > $ arr [ $ mid - 1 ] ) return $ arr [ $ mid ] ;
if ( $ arr [ $ mid ] > $ arr [ $ mid + 1 ] && $ arr [ $ mid ] < $ arr [ $ mid - 1 ] ) return findMaximum ( $ arr , $ low , $ mid - 1 ) ;
else return findMaximum ( $ arr , $ mid + 1 , $ high ) ; }
$ arr = array ( 1 , 3 , 50 , 10 , 9 , 7 , 6 ) ; $ n = sizeof ( $ arr ) ; echo ( " The ▁ maximum ▁ element ▁ is ▁ " ) ; echo ( findMaximum ( $ arr , 0 , $ n -1 ) ) ; ? >
< ? php function getMissingNo ( $ a , $ n ) { $ total = ( $ n + 1 ) * ( $ n + 2 ) / 2 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ total -= $ a [ $ i ] ; return $ total ; }
$ a = array ( 1 , 2 , 4 , 5 , 6 ) ; $ miss = getMissingNo ( $ a , 5 ) ; echo ( $ miss ) ; ? >
< ? php function printTwoElements ( $ arr , $ size ) { $ i ; echo " The ▁ repeating ▁ element ▁ is " , " ▁ " ; for ( $ i = 0 ; $ i < $ size ; $ i ++ ) { if ( $ arr [ abs ( $ arr [ $ i ] ) - 1 ] > 0 ) $ arr [ abs ( $ arr [ $ i ] ) - 1 ] = - $ arr [ abs ( $ arr [ $ i ] ) - 1 ] ; else echo ( abs ( $ arr [ $ i ] ) ) ; } echo " and the missing element is " ; for ( $ i = 0 ; $ i < $ size ; $ i ++ ) { if ( $ arr [ $ i ] > 0 ) echo ( $ i + 1 ) ; } }
$ arr = array ( 7 , 3 , 4 , 5 , 5 , 6 , 2 ) ; $ n = count ( $ arr ) ; printTwoElements ( $ arr , $ n ) ; ? >
< ? php function getTwoElements ( & $ arr , $ n ) {
$ xor1 ;
$ set_bit_no ; $ i ; $ x = 0 ; $ y = 0 ; $ xor1 = $ arr [ 0 ] ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ xor1 = $ xor1 ^ $ arr [ $ i ] ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ xor1 = $ xor1 ^ $ i ;
$ set_bit_no = $ xor1 & ~ ( $ xor1 - 1 ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( ( $ arr [ $ i ] & $ set_bit_no ) != 0 )
$ x = $ x ^ $ arr [ $ i ] ; else
$ y = $ y ^ $ arr [ $ i ] ; } for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { if ( ( $ i & $ set_bit_no ) != 0 )
$ x = $ x ^ $ i ; else
$ y = $ y ^ $ i ; }
}
$ arr = array ( 1 , 3 , 4 , 5 , 1 , 6 , 2 ) ; $ n = sizeof ( $ arr ) ; getTwoElements ( $ arr , $ n ) ;
< ? php function findFourElements ( $ A , $ n , $ X ) {
for ( $ i = 0 ; $ i < $ n - 3 ; $ i ++ ) {
for ( $ j = $ i + 1 ; $ j < $ n - 2 ; $ j ++ ) {
for ( $ k = $ j + 1 ; $ k < $ n - 1 ; $ k ++ ) {
for ( $ l = $ k + 1 ; $ l < $ n ; $ l ++ ) if ( $ A [ $ i ] + $ A [ $ j ] + $ A [ $ k ] + $ A [ $ l ] == $ X ) echo $ A [ $ i ] , " , ▁ " , $ A [ $ j ] , " , ▁ " , $ A [ $ k ] , " , ▁ " , $ A [ $ l ] ; } } } }
$ A = array ( 10 , 20 , 30 , 40 , 1 , 2 ) ; $ n = sizeof ( $ A ) ; $ X = 91 ; findFourElements ( $ A , $ n , $ X ) ; ? >
< ? php function minDistance ( $ arr , $ n ) { $ maximum_element = $ arr [ 0 ] ; $ min_dis = $ n ; $ index = 0 ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) {
if ( $ maximum_element == $ arr [ $ i ] ) { $ min_dis = min ( $ min_dis , ( $ i - $ index ) ) ; $ index = $ i ; }
else if ( $ maximum_element < $ arr [ $ i ] ) { $ maximum_element = $ arr [ $ i ] ; $ min_dis = $ n ; $ index = $ i ; }
else continue ; } return $ min_dis ; }
$ arr = array ( 6 , 3 , 1 , 3 , 6 , 4 , 6 ) ; $ n = count ( $ arr ) ; echo " Minimum ▁ distance ▁ = ▁ " . minDistance ( $ arr , $ n ) ; ? >
< ? php function maxSumIS ( $ arr , $ n ) { $ max = 0 ; $ msis = array ( $ n ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ msis [ $ i ] = $ arr [ $ i ] ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) for ( $ j = 0 ; $ j < $ i ; $ j ++ ) if ( $ arr [ $ i ] > $ arr [ $ j ] && $ msis [ $ i ] < $ msis [ $ j ] + $ arr [ $ i ] ) $ msis [ $ i ] = $ msis [ $ j ] + $ arr [ $ i ] ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ max < $ msis [ $ i ] ) $ max = $ msis [ $ i ] ; return $ max ; }
$ arr = array ( 1 , 101 , 2 , 3 , 100 , 4 , 5 ) ; $ n = count ( $ arr ) ; echo " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ subsequence ▁ is ▁ " . maxSumIS ( $ arr , $ n ) ; ? >
< ? php function per ( $ a , $ b ) { return ( $ a + $ b ) ; }
function area ( $ s ) { return ( $ s / 2 ) ; }
$ a = 7 ; $ b = 8 ; $ s = 10 ; echo ( per ( $ a , $ b ) " " ) ; echo " STRNEWLINE " ; echo ( area ( $ s ) ) ; ? >
< ? php $ PI = 3.14159265 ;
function area_leaf ( $ a ) { global $ PI ; return ( $ a * $ a * ( $ PI / 2 - 1 ) ) ; }
$ a = 7 ; echo ( area_leaf ( $ a ) ) ; ? >
< ? php $ PI = 3.14159265 ;
function length_rope ( $ r ) { global $ PI ; return ( ( 2 * $ PI * $ r ) + 6 * $ r ) ; }
$ r = 7 ; echo ( length_rope ( $ r ) ) ; ? >
< ? php $ PI = 3.14159265 ;
function area_inscribed ( $ P , $ B , $ H ) { global $ PI ; return ( ( $ P + $ B - $ H ) * ( $ P + $ B - $ H ) * ( $ PI / 4 ) ) ; }
$ P = 3 ; $ B = 4 ; $ H = 5 ; echo ( area_inscribed ( $ P , $ B , $ H ) ) ; ? >
< ? php $ PI = 3.14159265 ;
function area_circumscribed ( $ c ) { global $ PI ; return ( $ c * $ c * ( $ PI / 4 ) ) ; }
$ c = 8 ; echo ( area_circumscribed ( $ c ) ) ; ? >
< ? php $ PI = 3.14159265 ;
function area_inscribed ( $ a ) { global $ PI ; return ( $ a * $ a * ( $ PI / 12 ) ) ; }
function perm_inscribed ( $ a ) { global $ PI ; return ( $ PI * ( $ a / sqrt ( 3 ) ) ) ; }
$ a = 6 ; echo ( " Area ▁ of ▁ inscribed ▁ circle ▁ is ▁ : " ) ; echo ( area_inscribed ( $ a ) ) ; echo ( " Perimeter ▁ of ▁ inscribed ▁ circle ▁ is ▁ : " ) ; echo ( perm_inscribed ( $ a ) ) ; ? >
< ? php function area ( $ r ) {
return ( 0.5 ) * ( 3.14 ) * ( $ r * $ r ) ; }
function perimeter ( $ r ) {
return ( 3.14 ) * ( $ r ) ; }
$ r = 10 ;
echo " The ▁ Area ▁ of ▁ Semicircle : ▁ " , area ( $ r ) , " STRNEWLINE " ;
echo " The ▁ Perimeter ▁ of ▁ Semicircle : ▁ " , perimeter ( $ r ) , " STRNEWLINE " ; ? >
< ? php function equation_plane ( $ x1 , $ y1 , $ z1 , $ x2 , $ y2 , $ z2 , $ x3 , $ y3 , $ z3 ) { $ a1 = $ x2 - $ x1 ; $ b1 = $ y2 - $ y1 ; $ c1 = $ z2 - $ z1 ; $ a2 = $ x3 - $ x1 ; $ b2 = $ y3 - $ y1 ; $ c2 = $ z3 - $ z1 ; $ a = $ b1 * $ c2 - $ b2 * $ c1 ; $ b = $ a2 * $ c1 - $ a1 * $ c2 ; $ c = $ a1 * $ b2 - $ b1 * $ a2 ; $ d = ( - $ a * $ x1 - $ b * $ y1 - $ c * $ z1 ) ; echo sprintf ( " equation ▁ of ▁ the ▁ plane ▁ is ▁ % .2fx " . " ▁ + ▁ % .2fy ▁ + ▁ % .2fz ▁ + ▁ % .2f ▁ = ▁ 0" , $ a , $ b , $ c , $ d ) ; }
$ x1 = -1 ; $ y1 = 2 ; $ z1 = 1 ; $ x2 = 0 ; $ y2 = -3 ; $ z2 = 2 ; $ x3 = 1 ; $ y3 = 1 ; $ z3 = -4 ; equation_plane ( $ x1 , $ y1 , $ z1 , $ x2 , $ y2 , $ z2 , $ x3 , $ y3 , $ z3 ) ; ? >
< ? php function shortest_distance ( $ x1 , $ y1 , $ a , $ b , $ c ) { $ d = abs ( ( $ a * $ x1 + $ b * $ y1 + $ c ) ) / ( sqrt ( $ a * $ a + $ b * $ b ) ) ; echo " Perpendicular ▁ distance ▁ is ▁ " , $ d ; }
$ x1 = 5 ; $ y1 = 6 ; $ a = -2 ; $ b = 3 ; $ c = 4 ; shortest_distance ( $ x1 , $ y1 , $ a , $ b , $ c ) ; ? >
< ? php function octant ( $ x , $ y , $ z ) { if ( $ x >= 0 && $ y >= 0 && $ z >= 0 ) echo " Point ▁ lies ▁ in ▁ 1st ▁ octant STRNEWLINE " ; else if ( $ x < 0 && $ y >= 0 && $ z >= 0 ) echo " Point ▁ lies ▁ in ▁ 2nd ▁ octant STRNEWLINE " ; else if ( $ x < 0 && $ y < 0 && $ z >= 0 ) echo " Point ▁ lies ▁ in ▁ 3rd ▁ octant STRNEWLINE " ; else if ( $ x >= 0 && $ y < 0 && $ z >= 0 ) echo " Point ▁ lies ▁ in ▁ 4th ▁ octant STRNEWLINE " ; else if ( $ x >= 0 && $ y >= 0 && $ z < 0 ) echo " Point ▁ lies ▁ in ▁ 5th ▁ octant STRNEWLINE " ; else if ( $ x < 0 && $ y >= 0 && $ z < 0 ) echo " Point ▁ lies ▁ in ▁ 6th ▁ octant STRNEWLINE " ; else if ( $ x < 0 && $ y < 0 && $ z < 0 ) echo " Point ▁ lies ▁ in ▁ 7th ▁ octant STRNEWLINE " ; else if ( $ x >= 0 && $ y < 0 && $ z < 0 ) echo " Point ▁ lies ▁ in ▁ 8th ▁ octant STRNEWLINE " ; }
$ x = 2 ; $ y = 3 ; $ z = 4 ; octant ( $ x , $ y , $ z ) ; $ x = -4 ; $ y = 2 ; $ z = -8 ; octant ( $ x , $ y , $ z ) ; $ x = -6 ; $ y = -2 ; $ z = 8 ; octant ( $ x , $ y , $ z ) ; ? >
< ? php function maxArea ( $ a , $ b , $ c , $ d ) {
$ semiperimeter = ( $ a + $ b + $ c + $ d ) / 2 ;
return sqrt ( ( $ semiperimeter - $ a ) * ( $ semiperimeter - $ b ) * ( $ semiperimeter - $ c ) * ( $ semiperimeter - $ d ) ) ; }
$ a = 1 ; $ b = 2 ; $ c = 1 ; $ d = 2 ; echo ( maxArea ( $ a , $ b , $ c , $ d ) ) ; ? >
< ? php function midptellipse ( $ rx , $ ry , $ xc , $ yc ) { $ x = 0 ; $ y = $ ry ;
$ d1 = ( $ ry * $ ry ) - ( $ rx * $ rx * $ ry ) + ( 0.25 * $ rx * $ rx ) ; $ dx = 2 * $ ry * $ ry * $ x ; $ dy = 2 * $ rx * $ rx * $ y ;
while ( $ dx < $ dy ) {
echo " ( ▁ " , $ x + $ xc , " , ▁ " , $ y + $ yc , " ▁ ) STRNEWLINE " ; echo " ( ▁ " , - $ x + $ xc , " , ▁ " , $ y + $ yc , " ▁ ) STRNEWLINE " ; echo " ( ▁ " , $ x + $ xc , " , ▁ " , - $ y + $ yc , " ▁ ) STRNEWLINE " ; echo " ( ▁ " , - $ x + $ xc , " , ▁ " , - $ y + $ yc , " ▁ ) STRNEWLINE " ;
if ( $ d1 < 0 ) { $ x ++ ; $ dx = $ dx + ( 2 * $ ry * $ ry ) ; $ d1 = $ d1 + $ dx + ( $ ry * $ ry ) ; } else { $ x ++ ; $ y -- ; $ dx = $ dx + ( 2 * $ ry * $ ry ) ; $ dy = $ dy - ( 2 * $ rx * $ rx ) ; $ d1 = $ d1 + $ dx - $ dy + ( $ ry * $ ry ) ; } }
$ d2 = ( ( $ ry * $ ry ) * ( ( $ x + 0.5 ) * ( $ x + 0.5 ) ) ) + ( ( $ rx * $ rx ) * ( ( $ y - 1 ) * ( $ y - 1 ) ) ) - ( $ rx * $ rx * $ ry * $ ry ) ;
while ( $ y >= 0 ) {
echo " ( ▁ " , $ x + $ xc , " , ▁ " , $ y + $ yc , " ▁ ) STRNEWLINE " ; echo " ( ▁ " , - $ x + $ xc , " , ▁ " , $ y + $ yc , " ▁ ) STRNEWLINE " ; echo " ( ▁ " , $ x + $ xc , " , ▁ " , - $ y + $ yc , " ▁ ) STRNEWLINE " ; echo " ( ▁ " , - $ x + $ xc , " , ▁ " , - $ y + $ yc , " ▁ ) STRNEWLINE " ;
if ( $ d2 > 0 ) { $ y -- ; $ dy = $ dy - ( 2 * $ rx * $ rx ) ; $ d2 = $ d2 + ( $ rx * $ rx ) - $ dy ; } else { $ y -- ; $ x ++ ; $ dx = $ dx + ( 2 * $ ry * $ ry ) ; $ dy = $ dy - ( 2 * $ rx * $ rx ) ; $ d2 = $ d2 + $ dx - $ dy + ( $ rx * $ rx ) ; } } }
midptellipse ( 10 , 15 , 50 , 50 ) ; ? >
< ? php function HexToBin ( $ hexdec ) { $ i = 0 ; while ( $ hexdec [ $ i ] ) { switch ( $ hexdec [ $ i ] ) { case '0' : echo "0000" ; break ; case '1' : echo "0001" ; break ; case '2' : echo "0010" ; break ; case '3' : echo "0011" ; break ; case '4' : echo "0100" ; break ; case '5' : echo "0101" ; break ; case '6' : echo "0110" ; break ; case '7' : echo "0111" ; break ; case '8' : echo "1000" ; break ; case '9' : echo "1001" ; break ; case ' A ' : case ' a ' : echo "1010" ; break ; case ' B ' : case ' b ' : echo "1011" ; break ; case ' C ' : case ' c ' : echo "1100" ; break ; case ' D ' : case ' d ' : echo "1101" ; break ; case ' E ' : case ' e ' : echo "1110" ; break ; case ' F ' : case ' f ' : echo "1111" ; break ; default : echo " Invalid hexadecimal digit " $ hexdec [ $ i ] ; } $ i ++ ; } }
$ hexdec = "1AC5" ;
echo " Equivalent Binary value is : " HexToBin ( $ hexdec ) ;
< ? php function distance ( $ x1 , $ y1 , $ z1 , $ x2 , $ y2 , $ z2 ) { $ d = sqrt ( pow ( $ x2 - $ x1 , 2 ) + pow ( $ y2 - $ y1 , 2 ) + pow ( $ z2 - $ z1 , 2 ) * 1.0 ) ; echo " Distance ▁ is ▁ " . $ d ; }
$ x1 = 2 ; $ y1 = -5 ; $ z1 = 7 ; $ x2 = 3 ; $ y2 = 4 ; $ z2 = 5 ;
distance ( $ x1 , $ y1 , $ z1 , $ x2 , $ y2 , $ z2 ) ; ? >
< ? php function No_Of_Pairs ( $ N ) { $ i = 1 ;
while ( ( $ i * $ i * $ i ) + ( 2 * $ i * $ i ) + $ i <= $ N ) $ i ++ ; return ( $ i - 1 ) ; }
function print_pairs ( $ pairs ) { $ i = 1 ; $ mul ; for ( $ i = 1 ; $ i <= $ pairs ; $ i ++ ) { $ mul = $ i * ( $ i + 1 ) ; echo " Pair ▁ no . " , $ i , " ▁ - - > ▁ ( " , ( $ mul * $ i ) , " , ▁ " , $ mul * ( $ i + 1 ) , " ) ▁ STRNEWLINE " ; } }
$ N = 500 ; $ pairs ; $ mul ; $ i = 1 ; $ pairs = No_Of_Pairs ( $ N ) ; echo " No . ▁ of ▁ pairs ▁ = ▁ " , $ pairs , " ▁ STRNEWLINE " ; print_pairs ( $ pairs ) ; ? >
< ? php function findArea ( $ d ) { return ( $ d * $ d ) / 2 ; }
$ d = 10 ; echo ( findArea ( $ d ) ) ; ? >
< ? php function AvgofSquareN ( $ n ) { $ sum = 0 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ sum += ( $ i * $ i ) ; return $ sum / $ n ; }
$ n = 2 ; echo ( AvgofSquareN ( $ n ) ) ; ? >
< ? php function Series ( $ x , $ n ) { $ sum = 1 ; $ term = 1 ; $ fct ; $ j ; $ y = 2 ; $ m ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { $ fct = 1 ; for ( $ j = 1 ; $ j <= $ y ; $ j ++ ) { $ fct = $ fct * $ j ; } $ term = $ term * ( -1 ) ; $ m = $ term * pow ( $ x , $ y ) / $ fct ; $ sum = $ sum + $ m ; $ y += 2 ; } return $ sum ; }
$ x = 9 ; $ n = 10 ; $ precision = 4 ; echo substr ( number_format ( Series ( $ x , $ n ) , $ precision + 1 , ' . ' , ' ' ) , 0 , -1 ) ; ? >
< ? php function maxPrimeFactors ( $ n ) {
$ maxPrime = -1 ;
while ( $ n % 2 == 0 ) { $ maxPrime = 2 ;
$ n >>= 1 ; }
while ( $ n % 3 == 0 ) { $ maxPrime = 3 ; $ n = $ n / 3 ; }
for ( $ i = 3 ; $ i <= sqrt ( $ n ) ; $ i += 2 ) { while ( $ n % $ i == 0 ) { $ maxPrime = $ i ; $ n = $ n / $ i ; } while ( $ n % ( $ i + 2 ) == 0 ) { $ maxPrime = $ i + 2 ; $ n = $ n / ( $ i + 2 ) ; } }
if ( $ n > 4 ) $ maxPrime = $ n ; return $ maxPrime ; }
$ n = 15 ; echo maxPrimeFactors ( $ n ) , " STRNEWLINE " ; $ n = 25698751364526 ; echo maxPrimeFactors ( $ n ) , " STRNEWLINE " ; ? >
< ? php function sum ( $ x , $ n ) { $ i ; $ total = 1.0 ; $ multi = $ x ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { $ total = $ total + $ multi / $ i ; $ multi = $ multi * $ x ; } return $ total ; }
$ x = 2 ; $ n = 5 ; echo ( sum ( $ x , $ n ) ) ; ? >
< ? php function triangular_series ( $ n ) { $ i ; $ j = 1 ; $ k = 1 ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { echo ( " ▁ " . $ k . " ▁ " ) ;
$ j = $ j + 1 ;
$ k = $ k + $ j ; } }
$ n = 5 ; triangular_series ( $ n ) ; ? >
< ? php function countDigit ( $ n ) { if ( $ n / 10 == 0 ) return 1 ; return 1 + countDigit ( ( int ) ( $ n / 10 ) ) ; }
$ n = 345289467 ; print ( " Number ▁ of ▁ digits ▁ : ▁ " . ( countDigit ( $ n ) ) ) ; ? >
< ? php function findMaxValue ( ) { $ res = 2 ; $ fact = 2 ; $ pos = -1 ; while ( true ) {
$ mystring = $ fact ; $ pos = strpos ( $ mystring , ' E ' ) ; if ( $ pos > 0 ) break ; $ res ++ ; $ fact = $ fact * $ res ; } return $ res - 1 ; }
echo " Maximum ▁ value ▁ of " . " ▁ integer ▁ " . findMaxValue ( ) ; ? >
< ? php function firstkdigits ( $ n , $ k ) {
$ product = $ n * log10 ( $ n ) ;
$ decimal_part = $ product - floor ( $ product ) ;
$ decimal_part = pow ( 10 , $ decimal_part ) ;
$ digits = pow ( 10 , $ k - 1 ) ; $ i = 0 ; return floor ( $ decimal_part * $ digits ) ; }
$ n = 1450 ; $ k = 6 ; echo firstkdigits ( $ n , $ k ) ; ? >
< ? php function moduloMultiplication ( $ a , $ b , $ mod ) {
$ a %= $ mod ; while ( $ b ) {
if ( $ b & 1 ) $ res = ( $ res + $ a ) % $ mod ;
$ a = ( 2 * $ a ) % $ mod ;
} return $ res ; }
$ a = 10123465234878998 ; $ b = 65746311545646431 ; $ m = 10005412336548794 ; echo moduloMultiplication ( $ a , $ b , $ m ) ; ? >
< ? php function findRoots ( $ a , $ b , $ c ) {
if ( $ a == 0 ) { echo " Invalid " ; return ; } $ d = $ b * $ b - 4 * $ a * $ c ; $ sqrt_val = sqrt ( abs ( $ d ) ) ; if ( $ d > 0 ) { echo " Roots ▁ are ▁ real ▁ and ▁ " . " different ▁ STRNEWLINE " ; echo ( - $ b + $ sqrt_val ) / ( 2 * $ a ) , " STRNEWLINE " , ( - $ b - $ sqrt_val ) / ( 2 * $ a ) ; } else if ( $ d == 0 ) { echo " Roots ▁ are ▁ real ▁ and ▁ same ▁ STRNEWLINE " ; echo - $ b / ( 2 * $ a ) ; }
else { echo " Roots ▁ are ▁ complex ▁ STRNEWLINE " ; echo - $ b / ( 2 * $ a ) , " ▁ + ▁ i " , $ sqrt_val , " STRNEWLINE " , - $ b / ( 2 * $ a ) , " ▁ - ▁ i " , $ sqrt_val ; } }
$ a = 1 ; $ b = -7 ; $ c = 12 ;
findRoots ( $ a , $ b , $ c ) ; ? >
< ? php function val ( $ c ) { if ( $ c >= '0' && $ c <= '9' ) return ord ( $ c ) - ord ( '0' ) ; else return ord ( $ c ) - ord ( ' A ' ) + 10 ; }
function toDeci ( $ str , $ base ) { $ len = strlen ( $ str ) ;
$ power = 1 ;
$ num = 0 ;
for ( $ i = $ len - 1 ; $ i >= 0 ; $ i -- ) {
if ( val ( $ str [ $ i ] ) >= $ base ) { print ( " Invalid ▁ Number " ) ; return -1 ; } $ num += val ( $ str [ $ i ] ) * $ power ; $ power = $ power * $ base ; } return $ num ; }
$ str = "11A " ; $ base = 16 ; print ( " Decimal ▁ equivalent ▁ of ▁ $ str ▁ " . " in ▁ base ▁ $ base ▁ is ▁ " . toDeci ( $ str , $ base ) ) ; ? >
< ? php function seriesSum ( $ calculated , $ current , $ N ) { $ i ; $ cur = 1 ;
if ( $ current == $ N + 1 ) return 0 ;
for ( $ i = $ calculated ; $ i < $ calculated + $ current ; $ i ++ ) $ cur *= $ i ;
return $ cur + seriesSum ( $ i , $ current + 1 , $ N ) ; }
$ N = 5 ;
echo ( seriesSum ( 1 , 1 , $ N ) ) ; ? >
< ? php function modInverse ( $ a , $ m ) { $ m0 = $ m ; $ y = 0 ; $ x = 1 ; if ( $ m == 1 ) return 0 ; while ( $ a > 1 ) {
$ q = ( int ) ( $ a / $ m ) ; $ t = $ m ;
$ m = $ a % $ m ; $ a = $ t ; $ t = $ y ;
$ y = $ x - $ q * $ y ; $ x = $ t ; }
if ( $ x < 0 ) $ x += $ m0 ; return $ x ; }
$ a = 3 ; $ m = 11 ;
echo " Modular ▁ multiplicative ▁ inverse ▁ is STRNEWLINE " , modInverse ( $ a , $ m ) ; a . . . >
< ? php function gcd ( $ a , $ b ) { if ( $ a == 0 ) return $ b ; return gcd ( $ b % $ a , $ a ) ; }
function phi ( $ n ) { $ result = 1 ; for ( $ i = 2 ; $ i < $ n ; $ i ++ ) if ( gcd ( $ i , $ n ) == 1 ) $ result ++ ; return $ result ; }
for ( $ n = 1 ; $ n <= 10 ; $ n ++ ) echo " phi ( " ▁ . $ n . ▁ " ) = " ▁ . ▁ phi ( $ n ) . " " I >
< ? php function phi ( $ n ) {
for ( $ p = 2 ; $ p * $ p <= $ n ; ++ $ p ) {
if ( $ n % $ p == 0 ) {
while ( $ n % $ p == 0 ) $ n /= $ p ; $ result *= ( 1.0 - ( 1.0 / $ p ) ) ; } }
if ( $ n > 1 ) $ result *= ( 1.0 - ( 1.0 / $ n ) ) ; return intval ( $ result ) ; }
for ( $ n = 1 ; $ n <= 10 ; $ n ++ ) echo " phi ( " ▁ . $ n . ▁ " ) = " ▁ . ▁ phi ( $ n ) . " " I >
< ? php function printFibonacciNumbers ( $ n ) { $ f1 = 0 ; $ f2 = 1 ; $ i ; if ( $ n < 1 ) return ; echo ( $ f1 ) ; echo ( " ▁ " ) ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { echo ( $ f2 ) ; echo ( " ▁ " ) ; $ next = $ f1 + $ f2 ; $ f1 = $ f2 ; $ f2 = $ next ; } }
printFibonacciNumbers ( 7 ) ; ? >
< ? php function gcd ( $ a , $ b ) { if ( $ a == 0 ) return $ b ; return gcd ( $ b % $ a , $ a ) ; }
function lcm ( $ a , $ b ) { return ( $ a / gcd ( $ a , $ b ) ) * $ b ; }
$ a = 15 ; $ b = 20 ; echo " LCM ▁ of ▁ " , $ a , " ▁ and ▁ " , $ b , " ▁ is ▁ " , lcm ( $ a , $ b ) ; ? >
< ? php function convert_to_words ( $ num ) {
if ( $ len == 0 ) { echo " empty ▁ string STRNEWLINE " ; return ; } if ( $ len > 4 ) { echo " Length ▁ more ▁ than ▁ 4 ▁ " . " is ▁ not ▁ supported STRNEWLINE " ; return ; }
$ single_digits = array ( " zero " , " one " , " two " , " three " , " four " , " five " , " six " , " seven " , " eight " , " nine " ) ;
$ two_digits = array ( " " , " ten " , " eleven " , " twelve " , " thirteen " , " fourteen " , " fifteen " , " sixteen " , " seventeen " , " eighteen " , " nineteen " ) ;
$ tens_multiple = array ( " " , " " , " twenty " , " thirty " , " forty " , " fifty " , " sixty " , " seventy " , " eighty " , " ninety " ) ; $ tens_power = array ( " hundred " , " thousand " ) ;
echo $ num . " : ▁ " ;
if ( $ len == 1 ) { echo $ single_digits [ $ num [ 0 ] - '0' ] . " ▁ STRNEWLINE " ; return ; }
$ x = 0 ; while ( $ x < strlen ( $ num ) ) {
if ( $ len >= 3 ) { if ( $ num [ $ x ] - '0' != 0 ) { echo $ single_digits [ $ num [ $ x ] - '0' ] . " " ; echo $ tens_power [ $ len - 3 ] . " " ;
} -- $ len ; }
else {
if ( $ num [ $ x ] - '0' == 1 ) { $ sum = $ num [ $ x ] - '0' + $ num [ $ x ] - '0' ; echo $ two_digits [ $ sum ] . " ▁ STRNEWLINE " ; return ; }
else if ( $ num [ $ x ] - '0' == 2 && $ num [ $ x + 1 ] - '0' == 0 ) { echo " twenty STRNEWLINE " ; return ; }
else { $ i = $ num [ $ x ] - '0' ; if ( $ i > 0 ) echo $ tens_multiple [ $ i ] . " ▁ " ; else echo " " ; ++ $ x ; if ( $ num [ $ x ] - '0' != 0 ) echo $ single_digits [ $ num [ $ x ] - '0' ] . " ▁ STRNEWLINE " ; } } ++ $ x ; } }
convert_to_words ( "9923" ) ; convert_to_words ( "523" ) ; convert_to_words ( "89" ) ; convert_to_words ( "8" ) ; ? >
< ? php $ MAX = 11 ; function isMultipleof5 ( $ n ) { global $ MAX ; $ str = ( string ) $ n ; $ len = strlen ( $ str ) ;
if ( $ str [ $ len - 1 ] == '5' $ str [ $ len - 1 ] == '0' ) return true ; return false ; }
$ n = 19 ; if ( isMultipleof5 ( $ n ) == true ) echo " $ n ▁ is ▁ multiple ▁ of ▁ 5" ; else echo " $ n ▁ is ▁ not ▁ a ▁ multiple ▁ of ▁ 5" ; ? >
< ? php function add ( $ x , $ y ) { $ keep = ( $ x & $ y ) << 1 ; $ res = $ x ^ $ y ;
if ( $ keep == 0 ) { echo $ res ; exit ( 0 ) ; } add ( $ keep , $ res ) ; }
$ k = add ( 15 , 38 ) ; ? >
< ? php function countBits ( $ number ) {
return ( int ) ( log ( $ number ) / log ( 2 ) ) + 1 ; }
$ num = 65 ; echo ( countBits ( $ num ) ) ; ? >
< ? php $ INT_SIZE = 32 ;
function constructNthNumber ( $ group_no , $ aux_num , $ op ) { global $ INT_SIZE ; $ a = array_fill ( 0 , $ INT_SIZE , 0 ) ; $ num = 0 ; $ i = 0 ; $ len_f = 0 ;
if ( $ op == 2 ) {
$ len_f = 2 * $ group_no ;
$ a [ $ len_f - 1 ] = $ a [ 0 ] = 1 ;
while ( $ aux_num ) {
$ a [ $ group_no + i ] = $ a [ $ group_no - 1 - $ i ] = $ aux_num & 1 ; $ aux_num = $ aux_num >> 1 ; $ i ++ ; } }
else if ( $ op == 0 ) {
$ len_f = 2 * $ group_no + 1 ;
$ a [ $ len_f - 1 ] = $ a [ 0 ] = 1 ; $ a [ $ group_no ] = 0 ;
while ( $ aux_num ) {
$ a [ $ group_no + 1 + $ i ] = $ a [ $ group_no - 1 - $ i ] = $ aux_num & 1 ; $ aux_num = $ aux_num >> 1 ; $ i ++ ; } }
{
$ len_f = 2 * $ group_no + 1 ;
$ a [ $ len_f - 1 ] = $ a [ 0 ] = 1 ; $ a [ $ group_no ] = 1 ;
while ( $ aux_num ) {
$ a [ $ group_no + 1 + $ i ] = $ a [ $ group_no - 1 - $ i ] = $ aux_num & 1 ; $ aux_num = $ aux_num >> 1 ; $ i ++ ; } }
for ( $ i = 0 ; $ i < $ len_f ; $ i ++ ) $ num += ( 1 << $ i ) * $ a [ $ i ] ; return $ num ; }
function getNthNumber ( $ n ) { $ group_no = 0 ; $ count_upto_group = 0 ; $ count_temp = 1 ; $ op = $ aux_num = 0 ;
while ( $ count_temp < $ n ) { $ group_no ++ ;
$ count_upto_group = $ count_temp ; $ count_temp += 3 * ( 1 << ( $ group_no - 1 ) ) ; }
$ group_offset = $ n - $ count_upto_group - 1 ;
if ( ( $ group_offset + 1 ) <= ( 1 << ( $ group_no - 1 ) ) ) {
$ aux_num = $ group_offset ; } else { if ( ( ( $ group_offset + 1 ) - ( 1 << ( $ group_no - 1 ) ) ) % 2 )
else
$ aux_num = ( int ) ( ( ( $ group_offset ) - ( 1 << ( $ group_no - 1 ) ) ) / 2 ) ; } return constructNthNumber ( $ group_no , $ aux_num , $ op ) ; }
$ n = 9 ;
print ( getNthNumber ( $ n ) ) ; ? >
< ? php function flip ( & $ arr , $ i ) { $ start = 0 ; while ( $ start < $ i ) { $ temp = $ arr [ $ start ] ; $ arr [ $ start ] = $ arr [ $ i ] ; $ arr [ $ i ] = $ temp ; $ start ++ ; $ i -- ; } }
function findMax ( $ arr , $ n ) { $ mi = 0 ; for ( $ i = 0 ; $ i < $ n ; ++ $ i ) if ( $ arr [ $ i ] > $ arr [ $ mi ] ) $ mi = $ i ; return $ mi ; }
function pancakeSort ( & $ arr , $ n ) {
for ( $ curr_size = $ n ; $ curr_size > 1 ; -- $ curr_size ) {
$ mi = findMax ( $ arr , $ curr_size ) ;
if ( $ mi != $ curr_size - 1 ) {
flip ( $ arr , $ mi ) ;
flip ( $ arr , $ curr_size -1 ) ; } } }
function printArray ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; ++ $ i ) print ( $ arr [ $ i ] . " ▁ " ) ; }
$ arr = array ( 23 , 10 , 20 , 11 , 12 , 6 , 7 ) ; $ n = count ( $ arr ) ; pancakeSort ( $ arr , $ n ) ; echo ( " Sorted ▁ Array ▁ STRNEWLINE " ) ; printArray ( $ arr , $ n ) ; return 0 ; ? >
