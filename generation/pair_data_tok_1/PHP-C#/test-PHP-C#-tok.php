< ? php function Loss ( $ SP , $ P ) { $ loss = 0 ; $ loss = ( ( 2 * $ P * $ P * $ SP ) / ( 100 * 100 - $ P * $ P ) ) ; print ( " Loss ▁ = ▁ " . round ( $ loss , 3 ) ) ; }
$ SP = 2400 ; $ P = 30 ;
Loss ( $ SP , $ P ) ; ? >
< ? php $ MAXN = 10001 ;
$ spf = array_fill ( 0 , $ MAXN , 0 ) ;
$ hash1 = array_fill ( 0 , $ MAXN , 0 ) ;
function sieve ( ) { global $ spf , $ MAXN , $ hash1 ; $ spf [ 1 ] = 1 ; for ( $ i = 2 ; $ i < $ MAXN ; $ i ++ )
$ spf [ $ i ] = $ i ;
for ( $ i = 4 ; $ i < $ MAXN ; $ i += 2 ) $ spf [ $ i ] = 2 ;
for ( $ i = 3 ; $ i * $ i < $ MAXN ; $ i ++ ) {
if ( $ spf [ $ i ] == $ i ) { for ( $ j = $ i * $ i ; $ j < $ MAXN ; $ j += $ i )
if ( $ spf [ $ j ] == $ j ) $ spf [ $ j ] = $ i ; } } }
function getFactorization ( $ x ) { global $ spf , $ MAXN , $ hash1 ; while ( $ x != 1 ) { $ temp = $ spf [ $ x ] ; if ( $ x % $ temp == 0 ) {
$ hash1 [ $ spf [ $ x ] ] ++ ; $ x = ( int ) ( $ x / $ spf [ $ x ] ) ; } while ( $ x % $ temp == 0 ) $ x = ( int ) ( $ x / $ temp ) ; } }
function check ( $ x ) { global $ spf , $ MAXN , $ hash1 ; while ( $ x != 1 ) { $ temp = $ spf [ $ x ] ;
if ( $ x % $ temp == 0 && $ hash1 [ $ temp ] > 1 ) return false ; while ( $ x % $ temp == 0 ) $ x = ( int ) ( $ x / $ temp ) ; } return true ; }
function hasValidNum ( $ arr , $ n ) { global $ spf , $ MAXN , $ hash1 ;
sieve ( ) ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) getFactorization ( $ arr [ $ i ] ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( check ( $ arr [ $ i ] ) ) return true ; return false ; }
$ arr = array ( 2 , 8 , 4 , 10 , 6 , 7 ) ; $ n = count ( $ arr ) ; if ( hasValidNum ( $ arr , $ n ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function countWays ( $ N ) {
$ E = ( $ N * ( $ N - 1 ) ) / 2 ; if ( $ N == 1 ) return 0 ; return ( int ) pow ( 2 , $ E - 1 ) ; }
$ N = 4 ; echo ( countWays ( $ N ) ) ; ? >
< ? php function minAbsDiff ( $ n ) { $ mod = $ n % 4 ; if ( $ mod == 0 $ mod == 3 ) return 0 ; return 1 ; }
$ n = 5 ; echo minAbsDiff ( $ n ) ; ? >
< ? php function check ( $ s ) {
$ freq = array_fill ( 0 , 10 , 0 ) ; while ( $ s != 0 ) {
$ r = $ s % 10 ;
$ s = ( int ) ( $ s / 10 ) ;
$ freq [ $ r ] += 1 ; } $ xor = 0 ;
for ( $ i = 0 ; $ i < 10 ; $ i ++ ) $ xor = $ xor ^ $ freq [ $ i ] ; if ( $ xor == 0 ) return true ; else return false ; }
$ s = 122233 ; if ( check ( $ s ) ) print ( " Yes " ) ; else print ( " No " ) ; ? >
< ? php function printLines ( $ n , $ k ) {
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { echo ( $ k * ( 6 * $ i + 1 ) ) ; echo ( " ▁ " ) ; echo ( $ k * ( 6 * $ i + 2 ) ) ; echo ( " ▁ " ) ; echo ( $ k * ( 6 * $ i + 3 ) ) ; echo ( " ▁ " ) ; echo ( $ k * ( 6 * $ i + 5 ) ) ; echo ( " STRNEWLINE " ) ; } }
$ n = 2 ; $ k = 2 ; printLines ( $ n , $ k ) ; ? >
< ? php function calculateSum ( $ n ) {
return ( pow ( 2 , $ n + 1 ) + $ n - 2 ) ; }
$ n = 4 ;
echo " Sum = " ? >
< ? php function partitions ( $ n ) { $ p = array_fill ( 0 , $ n + 1 , 0 ) ;
$ p [ 0 ] = 1 ; for ( $ i = 1 ; $ i < $ n + 1 ; $ i ++ ) { $ k = 1 ; while ( ( $ k * ( 3 * $ k - 1 ) ) / 2 <= $ i ) { $ p [ $ i ] += ( ( $ k % 2 ? 1 : -1 ) * $ p [ $ i - ( $ k * ( 3 * $ k - 1 ) ) / 2 ] ) ; if ( $ k > 0 ) $ k *= -1 ; else $ k = 1 - $ k ; } } return $ p [ $ n ] ; }
$ N = 20 ; print ( partitions ( $ N ) ) ; ? >
< ? php function countPaths ( $ n , $ m ) {
if ( $ n == 0 $ m == 0 ) return 1 ;
return ( countPaths ( $ n - 1 , $ m ) + countPaths ( $ n , $ m - 1 ) ) ; }
$ n = 3 ; $ m = 2 ; echo " ▁ Number ▁ of ▁ Paths ▁ " , countPaths ( $ n , $ m ) ; ? >
< ? php function getMaxGold ( $ gold , $ m , $ n ) { $ MAX = 100 ;
$ goldTable = array ( array ( ) ) ; for ( $ i = 0 ; $ i < $ m ; $ i ++ ) for ( $ j = 0 ; $ j < $ n ; $ j ++ ) $ goldTable [ $ i ] [ $ j ] = 0 ; for ( $ col = $ n - 1 ; $ col >= 0 ; $ col -- ) { for ( $ row = 0 ; $ row < $ m ; $ row ++ ) {
if ( $ col == $ n - 1 ) $ right = 0 ; else $ right = $ goldTable [ $ row ] [ $ col + 1 ] ;
if ( $ row == 0 or $ col == $ n - 1 ) $ right_up = 0 ; else $ right_up = $ goldTable [ $ row - 1 ] [ $ col + 1 ] ;
if ( $ row == $ m - 1 or $ col == $ n - 1 ) $ right_down = 0 ; else $ right_down = $ goldTable [ $ row + 1 ] [ $ col + 1 ] ;
$ goldTable [ $ row ] [ $ col ] = $ gold [ $ row ] [ $ col ] + max ( $ right , $ right_up , $ right_down ) ; } }
$ res = $ goldTable [ 0 ] [ 0 ] ; for ( $ i = 0 ; $ i < $ m ; $ i ++ ) $ res = max ( $ res , $ goldTable [ $ i ] [ 0 ] ) ; return $ res ; }
$ gold = array ( array ( 1 , 3 , 1 , 5 ) , array ( 2 , 2 , 4 , 1 ) , array ( 5 , 0 , 2 , 3 ) , array ( 0 , 6 , 1 , 2 ) ) ; $ m = 4 ; $ n = 4 ; echo getMaxGold ( $ gold , $ m , $ n ) ; ? >
< ? php $ M = 100 ;
function minAdjustmentCost ( $ A , $ n , $ target ) {
global $ M ; $ dp = array ( array ( ) ) ;
for ( $ j = 0 ; $ j <= $ M ; $ j ++ ) $ dp [ 0 ] [ $ j ] = abs ( $ j - $ A [ 0 ] ) ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) {
for ( $ j = 0 ; $ j <= $ M ; $ j ++ ) {
$ dp [ $ i ] [ $ j ] = PHP_INT_MAX ;
for ( $ k = max ( $ j - $ target , 0 ) ; $ k <= min ( $ M , $ j + $ target ) ; $ k ++ ) $ dp [ $ i ] [ $ j ] = min ( $ dp [ $ i ] [ $ j ] , $ dp [ $ i - 1 ] [ $ k ] + abs ( $ A [ $ i ] - $ j ) ) ; } }
$ res = PHP_INT_MAX ; for ( $ j = 0 ; $ j <= $ M ; $ j ++ ) $ res = min ( $ res , $ dp [ $ n - 1 ] [ $ j ] ) ; return $ res ; }
$ arr = array ( 55 , 77 , 52 , 61 , 39 , 6 , 25 , 60 , 49 , 47 ) ; $ n = count ( $ arr ) ; $ target = 10 ; echo " Minimum ▁ adjustment ▁ cost ▁ is ▁ " , minAdjustmentCost ( $ arr , $ n , $ target ) ; ? >
< ? php function countChar ( $ str , $ x ) { $ count = 0 ; $ n = 10 ; for ( $ i = 0 ; $ i < strlen ( $ str ) ; $ i ++ ) if ( $ str [ $ i ] == $ x ) $ count ++ ;
$ repetitions = ( int ) ( $ n / strlen ( $ str ) ) ; $ count = $ count * $ repetitions ;
for ( $ i = 0 ; $ i < $ n % strlen ( $ str ) ; $ i ++ ) { if ( $ str [ $ i ] == $ x ) $ count ++ ; } return $ count ; }
$ str = " abcac " ; echo countChar ( $ str , ' a ' ) ; ? >
< ? php function check ( $ s , $ m ) {
$ l = count ( $ s ) ;
$ c1 = 0 ;
$ c2 = 0 ; for ( $ i = 0 ; $ i <= $ l ; $ i ++ ) { if ( $ s [ $ i ] == '0' ) { $ c2 = 0 ;
$ c1 ++ ; } else { $ c1 = 0 ;
$ c2 ++ ; } if ( $ c1 == $ m or $ c2 == $ m ) return true ; } return false ; }
$ s = "001001" ; $ m = 2 ;
if ( check ( $ s , $ m ) ) echo " YES " ; else echo " NO " ; ? >
< ? php function productAtKthLevel ( $ tree , $ k ) { $ level = -1 ;
$ n = strlen ( $ tree ) ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ tree [ $ i ] == ' ( ' ) $ level ++ ;
else if ( $ tree [ $ i ] == ' ) ' $ level -- ; else {
if ( $ level == $ k ) $ product *= ( ord ( $ tree [ $ i ] ) - ord ( '0' ) ) ; } }
return $ product ; }
$ tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; $ k = 2 ; echo productAtKthLevel ( $ tree , $ k ) ; ? >
< ? php function isValidISBN ( $ isbn ) {
$ n = strlen ( $ isbn ) ; if ( $ n != 10 ) return -1 ;
$ sum = 0 ; for ( $ i = 0 ; $ i < 9 ; $ i ++ ) { $ digit = $ isbn [ $ i ] - '0' ; if ( 0 > $ digit 9 < $ digit ) return -1 ; $ sum += ( $ digit * ( 10 - $ i ) ) ; }
$ last = $ isbn [ 9 ] ; if ( $ last != ' X ' && ( $ last < '0' $ last > '9' ) ) return -1 ;
$ sum += ( ( $ last == ' X ' ) ? 10 : ( $ last - '0' ) ) ;
return ( $ sum % 11 == 0 ) ; }
$ isbn = "007462542X " ; if ( isValidISBN ( $ isbn ) ) echo " Valid " ; else echo " Invalid " ; ? >
< ? php $ d = 10 ;
$ a = ( 360 - ( 6 * $ d ) ) / 4 ;
echo $ a , " , ▁ " , $ a + $ d , " , ▁ " , $ a + ( 2 * $ d ) , " , ▁ " , $ a + ( 3 * $ d ) ; ? >
< ? php function distance ( $ a1 , $ b1 , $ c1 , $ d1 , $ a2 , $ b2 , $ c2 , $ d2 ) { if ( $ a1 / $ a2 == $ b1 / $ b2 && $ b1 / $ b2 == $ c1 / $ c2 ) { $ x1 = $ y1 = 0 ; $ z1 = - $ d1 / $ c1 ; $ d = abs ( ( $ c2 * $ z1 + $ d2 ) ) / ( sqrt ( $ a2 * $ a2 + $ b2 * $ b2 + $ c2 * $ c2 ) ) ; echo " Perpendicular ▁ distance ▁ is ▁ " , $ d ; } else echo " Planes ▁ are ▁ not ▁ parallel " ; }
$ a1 = 1 ; $ b1 = 2 ; $ c1 = -1 ; $ d1 = 1 ; $ a2 = 3 ; $ b2 = 6 ; $ c2 = -3 ; $ d2 = -4 ; distance ( $ a1 , $ b1 , $ c1 , $ d1 , $ a2 , $ b2 , $ c2 , $ d2 ) ; ? >
< ? php function PrintMinNumberForPattern ( $ arr ) {
$ curr_max = 0 ;
$ last_entry = 0 ; $ j ;
for ( $ i = 0 ; $ i < strlen ( $ arr ) ; $ i ++ ) {
$ noOfNextD = 0 ; switch ( $ arr [ $ i ] ) { case ' I ' :
$ j = $ i + 1 ; while ( $ arr [ $ j ] == ' D ' && $ j < strlen ( $ arr ) ) { $ noOfNextD ++ ; $ j ++ ; } if ( $ i == 0 ) { $ curr_max = $ noOfNextD + 2 ;
echo " ▁ " , ++ $ last_entry ; echo " ▁ " , $ curr_max ;
$ last_entry = $ curr_max ; } else {
$ curr_max = $ curr_max + $ noOfNextD + 1 ;
$ last_entry = $ curr_max ; echo " ▁ " , $ last_entry ; }
for ( $ k = 0 ; $ k < $ noOfNextD ; $ k ++ ) { echo " ▁ " , -- $ last_entry ; $ i ++ ; } break ;
case ' D ' : if ( $ i == 0 ) {
$ j = $ i + 1 ; while ( ( $ arr [ $ j ] == ' D ' ) && ( $ j < strlen ( $ arr ) ) ) { $ noOfNextD ++ ; $ j ++ ; }
$ curr_max = $ noOfNextD + 2 ;
echo " " ▁ , ▁ $ curr _ max ▁ , STRNEWLINE " "
$ last_entry = $ curr_max - 1 ; } else {
echo " ▁ " , $ last_entry - 1 ; $ last_entry -- ; } break ; } } echo " " }
PrintMinNumberForPattern ( " IDID " ) ; PrintMinNumberForPattern ( " I " ) ; PrintMinNumberForPattern ( " DD " ) ; PrintMinNumberForPattern ( " II " ) ; PrintMinNumberForPattern ( " DIDI " ) ; PrintMinNumberForPattern ( " IIDDD " ) ; PrintMinNumberForPattern ( " DDIDDIID " ) ; ? >
< ? php function isPrime ( $ n ) { $ c = 0 ; for ( $ i = 1 ; $ i < $ n / 2 ; $ i ++ ) { if ( $ n % $ i == 0 ) $ c ++ ; } if ( $ c == 1 ) return 1 ; else return 0 ; }
function findMinNum ( $ arr , $ n ) {
$ first = 0 ; $ last = 0 ; $ num ; $ rev ; $ i ; $ hash = array_fill ( 0 , 20 , 0 ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ hash [ $ arr [ $ i ] ] ++ ; }
echo " Minimum ▁ number : ▁ " ; for ( $ i = 0 ; $ i <= 9 ; $ i ++ ) {
for ( $ j = 0 ; $ j < $ hash [ $ i ] ; $ j ++ ) echo $ i ; }
for ( $ i = 0 ; $ i <= 9 ; $ i ++ ) { if ( $ hash [ $ i ] != 0 ) { $ first = $ i ; break ; } }
for ( $ i = 9 ; $ i >= 0 ; $ i -- ) { if ( $ hash [ $ i ] != 0 ) { $ last = $ i ; break ; } } $ num = $ first * 10 + $ last ; $ rev = $ last * 10 + $ first ;
echo " Prime combinations : " if ( isPrime ( $ num ) && isPrime ( $ rev ) ) echo $ num . " ▁ " . $ rev ; else if ( isPrime ( $ num ) ) echo $ num ; else if ( isPrime ( $ rev ) ) echo $ rev ; else echo " No ▁ combinations ▁ exist " ; }
$ arr = array ( 1 , 2 , 4 , 7 , 8 ) ; findMinNum ( $ arr , 5 ) ; ? >
< ? php function gcd ( $ a , $ b ) { if ( $ a == 0 ) return $ b ; return gcd ( $ b % $ a , $ a ) ; }
function coprime ( $ a , $ b ) {
return ( gcd ( $ a , $ b ) == 1 ) ; }
function possibleTripletInRange ( $ L , $ R ) { $ flag = false ; $ possibleA ; $ possibleB ; $ possibleC ;
for ( $ a = $ L ; $ a <= $ R ; $ a ++ ) { for ( $ b = $ a + 1 ; $ b <= $ R ; $ b ++ ) { for ( $ c = $ b + 1 ; $ c <= $ R ; $ c ++ ) {
if ( coprime ( $ a , $ b ) && coprime ( $ b , $ c ) && ! coprime ( $ a , $ c ) ) { $ flag = true ; $ possibleA = $ a ; $ possibleB = $ b ; $ possibleC = $ c ; break ; } } } }
if ( $ flag == true ) { echo " ( " , $ possibleA , " , ▁ " , $ possibleB , " , ▁ " , $ possibleC , " ) " , " ▁ is ▁ one ▁ such ▁ possible ▁ triplet ▁ between ▁ " , $ L , " ▁ and ▁ " , $ R , " STRNEWLINE " ; } else { echo " No ▁ Such ▁ Triplet ▁ exists ▁ between ▁ " , $ L , " ▁ and ▁ " , $ R , " STRNEWLINE " ; } }
$ L ; $ R ;
$ L = 2 ; $ R = 10 ; possibleTripletInRange ( $ L , $ R ) ;
$ L = 23 ; $ R = 46 ; possibleTripletInRange ( $ L , $ R ) ; ? >
< ? php function possibleToReach ( $ a , $ b ) {
$ c = ( $ a * $ b ) ;
$ re1 = $ a / $ c ; $ re2 = $ b / $ c ;
if ( ( $ re1 * $ re1 * $ re2 == $ a ) && ( $ re2 * $ re2 * $ re1 == $ b ) ) return 1 ; else return -1 ; }
$ A = 60 ; $ B = 450 ; if ( possibleToReach ( $ A , $ B ) ) echo " yes " ; else echo " no " ; ? >
< ? php function isUndulating ( $ n ) {
if ( strlen ( $ n ) <= 2 ) return false ;
for ( $ i = 2 ; $ i < strlen ( $ n ) ; $ i ++ ) if ( $ n [ $ i - 2 ] != $ n [ $ i ] ) false ; return true ; }
$ n = "1212121" ; if ( isUndulating ( $ n ) ) echo ( " Yes " ) ; else echo ( " No " ) ; ? >
< ? php function Series ( $ n ) { $ i ; $ sums = 0 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ sums += ( $ i * $ i ) ; return $ sums ; }
$ n = 3 ; $ res = Series ( $ n ) ; echo ( $ res ) ; ? >
< ? php function sum ( $ L , $ R ) {
$ p = intval ( $ R / 6 ) ;
$ q = intval ( ( $ L - 1 ) / 6 ) ;
$ sumR = intval ( 3 * ( $ p * ( $ p + 1 ) ) ) ;
$ sumL = intval ( ( $ q * ( $ q + 1 ) ) * 3 ) ;
return $ sumR - $ sumL ; }
$ L = 1 ; $ R = 20 ; echo sum ( $ L , $ R ) ; ? >
< ? php function prevNum ( $ str ) { $ len = strlen ( $ str ) ; $ index = -1 ;
for ( $ i = $ len - 2 ; $ i >= 0 ; $ i -- ) { if ( $ str [ $ i ] > $ str [ $ i + 1 ] ) { $ index = $ i ; break ; } }
$ smallGreatDgt = -1 ; for ( $ i = $ len - 1 ; $ i > $ index ; $ i -- ) { if ( $ str [ $ i ] < $ str [ $ index ] ) { if ( $ smallGreatDgt == -1 ) $ smallGreatDgt = $ i ; else if ( $ str [ $ i ] >= $ str [ $ smallGreatDgt ] ) $ smallGreatDgt = $ i ; } }
if ( $ index == -1 ) return " - 1" ;
if ( $ smallGreatDgt != -1 ) { list ( $ str [ $ index ] , $ str [ $ smallGreatDgt ] ) = array ( $ str [ $ smallGreatDgt ] , $ str [ $ index ] ) ;
return $ str ; } return " - 1" ; }
$ str = "34125" ; echo prevNum ( $ str ) ; ? >
< ? php function horner ( $ poly , $ n , $ x ) {
$ result = $ poly [ 0 ] ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ result = $ result * $ x + $ poly [ $ i ] ; return $ result ; }
function findSign ( $ poly , $ n , $ x ) { $ result = horner ( $ poly , $ n , $ x ) ; if ( $ result > 0 ) return 1 ; else if ( $ result < 0 ) return -1 ; return 0 ; }
$ poly = array ( 2 , -6 , 2 , -1 ) ; $ x = 3 ; $ n = count ( $ poly ) ; echo " Sign ▁ of ▁ polynomial ▁ is ▁ " , findSign ( $ poly , $ n , $ x ) ; ? >
< ? php $ MAX = 100005 ;
function sieveOfEratostheneses ( ) { $ isPrime = array_fill ( true , $ MAX , NULL ) ; $ isPrime [ 1 ] = false ; for ( $ i = 2 ; $ i * $ i < $ MAX ; $ i ++ ) { if ( $ isPrime [ $ i ] ) { for ( $ j = 2 * $ i ; $ j < $ MAX ; $ j += $ i ) $ isPrime [ $ j ] = false ; } } }
function findPrime ( $ n ) { $ num = $ n + 1 ;
while ( $ num ) {
if ( $ isPrime [ $ num ] ) return $ num ;
$ num = $ num + 1 ; } return 0 ; }
function minNumber ( & $ arr , $ n ) {
sieveOfEratostheneses ( ) ; $ sum = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ sum += $ arr [ $ i ] ; if ( $ isPrime [ $ sum ] ) return 0 ;
$ num = findPrime ( $ sum ) ;
return $ num - $ sum ; }
$ arr = array ( 2 , 4 , 6 , 8 , 12 ) ; $ n = sizeof ( $ arr ) / sizeof ( $ arr [ 0 ] ) ; echo minNumber ( $ arr , $ n ) ; return 0 ; ? >
< ? php function SubArraySum ( $ arr , $ n ) { $ result = 0 ; $ temp = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
$ temp = 0 ; for ( $ j = $ i ; $ j < $ n ; $ j ++ ) {
$ temp += $ arr [ $ j ] $ result += $ temp ; } } return $ result ; }
$ arr = array ( 1 , 2 , 3 ) ; $ n = sizeof ( $ arr ) ; echo " Sum ▁ of ▁ SubArray ▁ : ▁ " , SubArraySum ( $ arr , $ n ) , " STRNEWLINE " ; ? >
< ? php function highestPowerof2 ( $ n ) { $ p = ( int ) log ( $ n , 2 ) ; return ( int ) pow ( 2 , $ p ) ; }
$ n = 10 ; echo highestPowerof2 ( $ n ) ; ? >
< ? php function aModM ( $ s , $ mod ) { $ number = 0 ; for ( $ i = 0 ; $ i < strlen ( $ s ) ; $ i ++ ) {
$ number = ( $ number * 10 + ( $ s [ $ i ] - '0' ) ) ; $ number %= $ mod ; } return $ number ; }
function ApowBmodM ( $ a , $ b , $ m ) {
$ ans = aModM ( $ a , $ m ) ; $ mul = $ ans ;
for ( $ i = 1 ; $ i < $ b ; $ i ++ ) $ ans = ( $ ans * $ mul ) % $ m ; return $ ans ; }
$ a = "987584345091051645734583954832576" ; $ b = 3 ; $ m = 11 ; echo ApowBmodM ( $ a , $ b , $ m ) ; return 0 ; ? >
< ? php function SieveOfSundaram ( $ n ) {
$ nNew = ( $ n - 1 ) / 2 ;
$ marked = array_fill ( 0 , ( $ nNew + 1 ) , false ) ;
for ( $ i = 1 ; $ i <= $ nNew ; $ i ++ ) for ( $ j = $ i ; ( $ i + $ j + 2 * $ i * $ j ) <= $ nNew ; $ j ++ ) $ marked [ $ i + $ j + 2 * $ i * $ j ] = true ;
if ( $ n > 2 ) echo "2 ▁ " ;
for ( $ i = 1 ; $ i <= $ nNew ; $ i ++ ) if ( $ marked [ $ i ] == false ) echo ( 2 * $ i + 1 ) . " ▁ " ; }
$ n = 20 ; SieveOfSundaram ( $ n ) ; ? >
< ? php function hammingDistance ( $ n1 , $ n2 ) { $ x = $ n1 ^ $ n2 ; $ setBits = 0 ; while ( $ x > 0 ) { $ setBits += $ x & 1 ; $ x >>= 1 ; } return $ setBits ; }
$ n1 = 9 ; $ n2 = 14 ; echo ( hammingDistance ( 9 , 14 ) ) ; ? >
< ? php function printSubsets ( $ n ) { for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) if ( ( $ n & $ i ) == $ i ) echo $ i . " " ; }
$ n = 9 ; printSubsets ( $ n ) ; ? >
< ? php function setBitNumber ( $ n ) {
$ k = ( int ) ( log ( $ n , 2 ) ) ;
return 1 << $ k ; }
$ n = 273 ; echo setBitNumber ( $ n ) ; ? >
< ? php function subset ( $ ar , $ n ) {
$ res = 0 ;
sort ( $ ar ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ count = 1 ;
for ( ; $ i < $ n - 1 ; $ i ++ ) { if ( $ ar [ $ i ] == $ ar [ $ i + 1 ] ) $ count ++ ; else break ; }
$ res = max ( $ res , $ count ) ; } return $ res ; }
$ arr = array ( 5 , 6 , 9 , 3 , 4 , 3 , 4 ) ; $ n = sizeof ( $ arr ) ; echo subset ( $ arr , $ n ) ; ? >
< ? php function areElementsContiguous ( $ arr , $ n ) {
sort ( $ arr ) ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] - $ arr [ $ i - 1 ] > 1 ) return false ; return true ; }
$ arr = array ( 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ) ; $ n = sizeof ( $ arr ) ; if ( areElementsContiguous ( $ arr , $ n ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function findLargestd ( $ S , $ n ) { $ found = false ;
sort ( $ S ) ;
for ( $ i = $ n - 1 ; $ i >= 0 ; $ i -- ) { for ( $ j = 0 ; $ j < $ n ; $ j ++ ) {
if ( $ i == $ j ) continue ; for ( $ k = $ j + 1 ; $ k < $ n ; $ k ++ ) { if ( $ i == $ k ) continue ; for ( $ l = $ k + 1 ; $ l < $ n ; $ l ++ ) { if ( $ i == $ l ) continue ;
if ( $ S [ $ i ] == $ S [ $ j ] + $ S [ $ k ] + $ S [ $ l ] ) { $ found = true ; return $ S [ $ i ] ; } } } } } if ( $ found == false ) return PHP_INT_MIN ; }
$ S = array ( 2 , 3 , 5 , 7 , 12 ) ; $ n = count ( $ S ) ; $ ans = findLargestd ( $ S , $ n ) ; if ( $ ans == PHP_INT_MIN ) echo " No ▁ Solution " ; else echo " Largest ▁ d ▁ such ▁ that ▁ a ▁ + ▁ b ▁ + ▁ " , " c ▁ = ▁ d ▁ is ▁ " , $ ans ; ? >
< ? php function leftRotatebyOne ( & $ arr , $ n ) { $ temp = $ arr [ 0 ] ; for ( $ i = 0 ; $ i < $ n - 1 ; $ i ++ ) $ arr [ $ i ] = $ arr [ $ i + 1 ] ; $ arr [ $ n - 1 ] = $ temp ; }
function leftRotate ( & $ arr , $ d , $ n ) { for ( $ i = 0 ; $ i < $ d ; $ i ++ ) leftRotatebyOne ( $ arr , $ n ) ; }
function printArray ( & $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; }
$ arr = array ( 1 , 2 , 3 , 4 , 5 , 6 , 7 ) ; $ n = sizeof ( $ arr ) ; leftRotate ( $ arr , 2 , $ n ) ; printArray ( $ arr , $ n ) ; ? >
< ? php < ? php # PHP  program to sort the NEW_LINE # array  in a given index range
function partSort ( $ arr , $ N , $ a , $ b ) {
$ l = min ( $ a , $ b ) ; $ r = max ( $ a , $ b ) ;
$ temp = array ( ) ; $ j = 0 ; for ( $ i = $ l ; $ i <= $ r ; $ i ++ ) { $ temp [ $ j ] = $ arr [ $ i ] ; $ j ++ ; }
sort ( $ temp ) ;
$ j = 0 ; for ( $ i = $ l ; $ i <= $ r ; $ i ++ ) { $ arr [ $ i ] = $ temp [ $ j ] ; $ j ++ ; }
for ( $ i = 0 ; $ i < $ N ; $ i ++ ) { echo $ arr [ $ i ] . " " ; } }
$ arr = array ( 7 , 8 , 4 , 5 , 2 ) ; $ a = 1 ; $ b = 4 ;
$ N = count ( $ arr ) ; partSort ( $ arr , $ N , $ a , $ b ) ; ? >
< ? php function pushZerosToEnd ( & $ arr , $ n ) {
$ count = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] != 0 )
$ arr [ $ count ++ ] = $ arr [ $ i ] ;
while ( $ count < $ n ) $ arr [ $ count ++ ] = 0 ; }
$ arr = array ( 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 ) ; $ n = sizeof ( $ arr ) ; pushZerosToEnd ( $ arr , $ n ) ; echo " Array ▁ after ▁ pushing ▁ all ▁ " . " zeros ▁ to ▁ end ▁ of ▁ array ▁ : STRNEWLINE " ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; ? >
< ? php function printArray ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo ( $ arr [ $ i ] . " ▁ " ) ; }
function RearrangePosNeg ( & $ arr , $ n ) { $ key ; $ j ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { $ key = $ arr [ $ i ] ;
if ( $ key > 0 ) continue ;
$ j = $ i - 1 ; while ( $ j >= 0 && $ arr [ $ j ] > 0 ) { $ arr [ $ j + 1 ] = $ arr [ $ j ] ; $ j = $ j - 1 ; }
$ arr [ $ j + 1 ] = $ key ; } }
{ $ arr = array ( -12 , 11 , -13 , -5 , 6 , -7 , 5 , -3 , -6 ) ; $ n = sizeof ( $ arr ) ; RearrangePosNeg ( $ arr , $ n ) ; printArray ( $ arr , $ n ) ; }
< ? php function findElements ( $ arr , $ n ) {
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ count = 0 ; for ( $ j = 0 ; $ j < $ n ; $ j ++ ) if ( $ arr [ $ j ] > $ arr [ $ i ] ) $ count ++ ; if ( $ count >= 2 ) echo $ arr [ $ i ] . " ▁ " ; } }
$ arr = array ( 2 , -6 , 3 , 5 , 1 ) ; $ n = sizeof ( $ arr ) ; findElements ( $ arr , $ n ) ; ? >
< ? php function findElements ( $ arr , $ n ) { sort ( $ arr ) ; for ( $ i = 0 ; $ i < $ n - 2 ; $ i ++ ) echo $ arr [ $ i ] , " ▁ " ; }
$ arr = array ( 2 , -6 , 3 , 5 , 1 ) ; $ n = count ( $ arr ) ; findElements ( $ arr , $ n ) ; ? > ;
< ? php function findElements ( $ arr , $ n ) { $ first = PHP_INT_MIN ; $ second = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ arr [ $ i ] > $ first ) { $ second = $ first ; $ first = $ arr [ $ i ] ; }
else if ( $ arr [ $ i ] > $ second ) $ second = $ arr [ $ i ] ; } for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] < $ second ) echo $ arr [ $ i ] , " ▁ " ; }
$ arr = array ( 2 , -6 , 3 , 5 , 1 ) ; $ n = count ( $ arr ) ; findElements ( $ arr , $ n ) ; ? >
< ? php function findFirstMissing ( $ array , $ start , $ end ) { if ( $ start > $ end ) return $ end + 1 ; if ( $ start != $ array [ $ start ] ) return $ start ; $ mid = ( $ start + $ end ) / 2 ;
if ( $ array [ $ mid ] == $ mid ) return findFirstMissing ( $ array , $ mid + 1 , $ end ) ; return findFirstMissing ( $ array , $ start , $ mid ) ; }
$ arr = array ( 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ) ; $ n = count ( $ arr ) ; echo " Smallest ▁ missing ▁ element ▁ is ▁ " , findFirstMissing ( $ arr , 2 , $ n - 1 ) ; ? >
< ? php function FindMaxSum ( $ arr , $ n ) { $ incl = $ arr [ 0 ] ; $ excl = 0 ; $ excl_new ; $ i ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) {
$ excl_new = ( $ incl > $ excl ) ? $ incl : $ excl ;
$ incl = $ excl + $ arr [ $ i ] ; $ excl = $ excl_new ; }
return ( ( $ incl > $ excl ) ? $ incl : $ excl ) ; }
$ arr = array ( 5 , 5 , 10 , 100 , 10 , 5 ) ; $ n = sizeof ( $ arr ) ; echo FindMaxSum ( $ arr , $ n ) ; ? >
< ? php function findMaxAverage ( $ arr , $ n , $ k ) {
if ( $ k > $ n ) return -1 ;
$ csum = array ( ) ; $ csum [ 0 ] = $ arr [ 0 ] ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ csum [ $ i ] = $ csum [ $ i - 1 ] + $ arr [ $ i ] ;
$ max_sum = $ csum [ $ k - 1 ] ; $ max_end = $ k - 1 ;
for ( $ i = $ k ; $ i < $ n ; $ i ++ ) { $ curr_sum = $ csum [ $ i ] - $ csum [ $ i - $ k ] ; if ( $ curr_sum > $ max_sum ) { $ max_sum = $ curr_sum ; $ max_end = $ i ; } }
return $ max_end - $ k + 1 ; }
$ arr = array ( 1 , 12 , -5 , -6 , 50 , 3 ) ; $ k = 4 ; $ n = count ( $ arr ) ; echo " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " , " length ▁ " , $ k , " ▁ begins ▁ at ▁ index ▁ " , findMaxAverage ( $ arr , $ n , $ k ) ; ? >
< ? php function findMaxAverage ( $ arr , $ n , $ k ) {
if ( $ k > $ n ) return -1 ;
$ sum = $ arr [ 0 ] ; for ( $ i = 1 ; $ i < $ k ; $ i ++ ) $ sum += $ arr [ $ i ] ; $ max_sum = $ sum ; $ max_end = $ k - 1 ;
for ( $ i = $ k ; $ i < $ n ; $ i ++ ) { $ sum = $ sum + $ arr [ $ i ] - $ arr [ $ i - $ k ] ; if ( $ sum > $ max_sum ) { $ max_sum = $ sum ; $ max_end = $ i ; } }
return $ max_end - $ k + 1 ; }
$ arr = array ( 1 , 12 , -5 , -6 , 50 , 3 ) ; $ k = 4 ; $ n = count ( $ arr ) ; echo " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " , " length ▁ " , $ k , " ▁ begins ▁ at ▁ index ▁ " , findMaxAverage ( $ arr , $ n , $ k ) ; ? >
< ? php function isMajority ( $ arr , $ n , $ x ) { $ i ;
$ last_index = $ n % 2 ? ( $ n / 2 + 1 ) : ( $ n / 2 ) ;
for ( $ i = 0 ; $ i < $ last_index ; $ i ++ ) {
if ( $ arr [ $ i ] == $ x && $ arr [ $ i + $ n / 2 ] == $ x ) return 1 ; } return 0 ; }
$ arr = array ( 1 , 2 , 3 , 4 , 4 , 4 , 4 ) ; $ n = sizeof ( $ arr ) ; $ x = 4 ; if ( isMajority ( $ arr , $ n , $ x ) ) echo $ x , " ▁ appears ▁ more ▁ than ▁ " , floor ( $ n / 2 ) , " ▁ times ▁ in ▁ arr [ ] " ; else echo $ x , " does ▁ not ▁ appear ▁ more ▁ than ▁ " , floor ( $ n / 2 ) , " times ▁ in ▁ arr [ ] " ; ? >
< ? php function cutRod ( $ price , $ n ) { $ val = array ( ) ; $ val [ 0 ] = 0 ; $ i ; $ j ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { $ max_val = PHP_INT_MIN ; for ( $ j = 0 ; $ j < $ i ; $ j ++ ) $ max_val = max ( $ max_val , $ price [ $ j ] + $ val [ $ i - $ j - 1 ] ) ; $ val [ $ i ] = $ max_val ; } return $ val [ $ n ] ; }
$ arr = array ( 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ) ; $ size = count ( $ arr ) ; echo " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " , cutRod ( $ arr , $ size ) ; ? >
< ? php function primeCount ( $ arr , $ n ) {
$ max_val = max ( $ arr ) ;
$ prime = array_fill ( 0 , $ max_val + 1 , true ) ;
$ prime [ 0 ] = false ; $ prime [ 1 ] = false ; for ( $ p = 2 ; $ p * $ p <= $ max_val ; $ p ++ ) {
if ( $ prime [ $ p ] == true ) {
for ( $ i = $ p * 2 ; $ i <= $ max_val ; $ i += $ p ) $ prime [ $ i ] = false ; } }
$ count = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ prime [ $ arr [ $ i ] ] ) $ count ++ ; return $ count ; }
function getPrefixArray ( $ arr , $ n , $ pre ) {
$ pre [ 0 ] = $ arr [ 0 ] ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { $ pre [ $ i ] = $ pre [ $ i - 1 ] + $ arr [ $ i ] ; } return $ pre ; }
$ arr = array ( 1 , 4 , 8 , 4 ) ; $ n = count ( $ arr ) ;
$ pre = array ( ) ; $ pre = getPrefixArray ( $ arr , $ n , $ pre ) ;
echo primeCount ( $ pre , $ n ) ; ? >
< ? php function minValue ( $ n , $ x , $ y ) {
$ val = ( $ y * $ n ) / 100 ;
if ( $ x >= $ val ) return 0 ; else return ( ceil ( $ val ) - $ x ) ; }
{ $ n = 10 ; $ x = 2 ; $ y = 40 ; echo ( minValue ( $ n , $ x , $ y ) ) ; }
< ? php function isPrime ( $ n ) {
if ( $ n <= 1 ) return false ; if ( $ n <= 3 ) return true ;
if ( $ n % 2 == 0 $ n % 3 == 0 ) return false ; for ( $ i = 5 ; $ i * $ i <= $ n ; $ i = $ i + 6 ) if ( $ n % $ i == 0 || $ n % ( $ i + 2 ) == 0 ) return false ; return true ; }
function isFactorialPrime ( $ n ) {
if ( ! isPrime ( $ n ) ) return false ; $ fact = 1 ; $ i = 1 ; while ( $ fact <= $ n + 1 ) {
$ fact = $ fact * $ i ;
if ( $ n + 1 == $ fact $ n - 1 == $ fact ) return true ; $ i ++ ; }
return false ; }
$ n = 23 ; if ( isFactorialPrime ( $ n ) ) echo " Yes " ; else echo " No " ; ? >
< ? php $ n = 5 ;
$ fac1 = 1 ; for ( $ i = 2 ; $ i <= $ n - 1 ; $ i ++ ) $ fac1 = $ fac1 * $ i ;
$ fac2 = $ fac1 * $ n ;
$ totalWays = $ fac1 * $ fac2 ;
echo $ totalWays . " STRNEWLINE " ;
< ? php function nextPerfectCube ( $ N ) { $ nextN = ( int ) ( floor ( pow ( $ N , ( 1 / 3 ) ) ) + 1 ) ; return $ nextN * $ nextN * $ nextN ; }
$ n = 35 ; print ( nextPerfectCube ( $ n ) ) ; ? >
< ? php function findpos ( $ n ) { $ pos = 0 ; for ( $ i = 0 ; isset ( $ n [ $ i ] ) != NULL ; $ i ++ ) { switch ( $ n [ $ i ] ) {
case '2' : $ pos = $ pos * 4 + 1 ; break ;
case '3' : $ pos = $ pos * 4 + 2 ; break ;
case '5' : $ pos = $ pos * 4 + 3 ; break ;
case '7' : $ pos = $ pos * 4 + 4 ; break ; } } return $ pos ; }
$ n = "777" ; echo findpos ( $ n ) ; ? >
< ? php $ mod = 1000000007 ;
function digitNumber ( $ n ) { global $ mod ;
if ( $ n == 0 ) return 1 ;
if ( $ n == 1 ) return 9 ;
if ( $ n % 2 != 0 ) {
$ temp = digitNumber ( ( $ n - 1 ) / 2 ) % $ mod ; return ( 9 * ( $ temp * $ temp ) % $ mod ) % $ mod ; } else {
$ temp = digitNumber ( $ n / 2 ) % $ mod ; return ( $ temp * $ temp ) % $ mod ; } } function countExcluding ( $ n , $ d ) { global $ mod ;
if ( $ d == 0 ) return ( 9 * digitNumber ( $ n - 1 ) ) % $ mod ; else return ( 8 * digitNumber ( $ n - 1 ) ) % $ mod ; }
$ d = 9 ; $ n = 3 ; print ( countExcluding ( $ n , $ d ) ) ; ? >
< ? php function isPrime ( $ n ) {
if ( $ n <= 1 ) return -1 ;
for ( $ i = 2 ; $ i < $ n ; $ i ++ ) if ( $ n % $ i == 0 ) return -1 ; return 1 ; }
function isEmirp ( $ n ) {
if ( isPrime ( $ n ) == -1 ) return -1 ;
$ rev = 0 ; while ( $ n != 0 ) { $ d = $ n % 10 ; $ rev = $ rev * 10 + $ d ; $ n /= 10 ; }
return isPrime ( $ rev ) ; }
$ n = 13 ; if ( isEmirp ( $ n ) == -1 ) echo " Yes " ; else echo " No " ; ? >
< ? php function Convert ( $ radian ) { $ pi = 3.14159 ; return ( $ radian * ( 180 / $ pi ) ) ; }
$ radian = 5.0 ; $ degree = Convert ( $ radian ) ; echo ( $ degree ) ; ? >
< ? php function sn ( $ n , $ an ) { return ( $ n * ( 1 + $ an ) ) / 2 ; }
function trace ( $ n , $ m ) {
$ an = 1 + ( $ n - 1 ) * ( $ m + 1 ) ;
$ rowmajorSum = sn ( $ n , $ an ) ;
$ an = 1 + ( $ n - 1 ) * ( $ n + 1 ) ;
$ colmajorSum = sn ( $ n , $ an ) ; return $ rowmajorSum + $ colmajorSum ; }
$ N = 3 ; $ M = 3 ; echo trace ( $ N , $ M ) , " STRNEWLINE " ; ? >
< ? php function max_area ( $ n , $ m , $ k ) { if ( $ k > ( $ n + $ m - 2 ) ) echo " Not ▁ possible " , " STRNEWLINE " ; else { $ result ;
if ( $ k < max ( $ m , $ n ) - 1 ) { $ result = max ( $ m * ( $ n / ( $ k + 1 ) ) , $ n * ( $ m / ( $ k + 1 ) ) ) ; }
else { $ result = max ( $ m / ( $ k - $ n + 2 ) , $ n / ( $ k - $ m + 2 ) ) ; }
echo $ result , " STRNEWLINE " ; } }
$ n = 3 ; $ m = 4 ; $ k = 1 ; max_area ( $ n , $ m , $ k ) ; ? >
< ? php function area_fun ( $ side ) { $ area = $ side * $ side ; return $ area ; }
$ side = 4 ; $ area = area_fun ( $ side ) ; echo ( $ area ) ; ? >
< ? php function countConsecutive ( $ N ) {
$ count = 0 ; for ( $ L = 1 ; $ L * ( $ L + 1 ) < 2 * $ N ; $ L ++ ) { $ a = ( int ) ( 1.0 * $ N - ( $ L * ( int ) ( $ L + 1 ) ) / 2 ) / ( $ L + 1 ) ; if ( $ a - ( int ) $ a == 0.0 ) $ count ++ ; } return $ count ; }
$ N = 15 ; echo countConsecutive ( $ N ) , " STRNEWLINE " ; $ N = 10 ; echo countConsecutive ( $ N ) , " STRNEWLINE " ; ? >
< ? php function isAutomorphic ( $ N ) {
$ sq = $ N * $ N ;
while ( $ N > 0 ) {
if ( $ N % 10 != $ sq % 10 ) return -1 ;
$ N /= 10 ; $ sq /= 10 ; } return 1 ; }
$ N = 5 ; $ geeks = isAutomorphic ( $ N ) ? " Automorphic " : " Not ▁ Automorphic " ; echo $ geeks ; ? >
< ? php function maxPrimefactorNum ( $ N ) {
$ arr = array_fill ( 0 , $ N + 5 , true ) ;
for ( $ i = 3 ; $ i * $ i <= $ N ; $ i += 2 ) { if ( $ arr [ $ i ] ) for ( $ j = $ i * $ i ; $ j <= $ N ; $ j += $ i ) $ arr [ $ j ] = false ; }
$ prime = array ( ) ; array_push ( $ prime , 2 ) ; for ( $ i = 3 ; $ i <= $ N ; $ i += 2 ) if ( $ arr [ $ i ] ) array_push ( $ prime , $ i ) ;
$ i = 0 ; $ ans = 1 ; while ( $ ans * $ prime [ $ i ] <= $ N && $ i < count ( $ prime ) ) { $ ans *= $ prime [ $ i ] ; $ i ++ ; } return $ ans ; }
$ N = 40 ; print ( maxPrimefactorNum ( $ N ) ) ; ? >
< ? php function divSum ( $ num ) {
$ result = 0 ;
for ( $ i = 2 ; $ i <= sqrt ( $ num ) ; $ i ++ ) {
if ( $ num % $ i == 0 ) {
if ( $ i == ( $ num / $ i ) ) $ result += $ i ; else $ result += ( $ i + $ num / $ i ) ; } }
return ( $ result + 1 ) ; }
$ num = 36 ; echo ( divSum ( $ num ) ) ; ? >
< ? php function power ( $ x , $ y , $ p ) {
$ res = 1 ;
$ x = $ x % $ p ; while ( $ y > 0 ) {
if ( $ y & 1 ) $ res = ( $ res * $ x ) % $ p ;
$ y = $ y >> 1 ; $ x = ( $ x * $ x ) % $ p ; } return $ res ; }
function squareRoot ( $ n , $ p ) { if ( $ p % 4 != 3 ) { echo " Invalid ▁ Input " ; return ; }
$ n = $ n % $ p ; $ x = power ( $ n , ( $ p + 1 ) / 4 , $ p ) ; if ( ( $ x * $ x ) % $ p == $ n ) { echo " Square ▁ root ▁ is ▁ " , $ x ; return ; }
$ x = $ p - $ x ; if ( ( $ x * $ x ) % $ p == $ n ) { echo " Square ▁ root ▁ is ▁ " , $ x ; return ; }
echo " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ; }
< ? php function power ( $ x , $ y , $ p ) {
$ res = 1 ;
$ x = $ x % $ p ; while ( $ y > 0 ) {
if ( $ y & 1 ) $ res = ( $ res * $ x ) % $ p ;
$ x = ( $ x * $ x ) % $ p ; } return $ res ; }
function miillerTest ( $ d , $ n ) {
$ a = 2 + rand ( ) % ( $ n - 4 ) ;
$ x = power ( $ a , $ d , $ n ) ; if ( $ x == 1 $ x == $ n - 1 ) return true ;
while ( $ d != $ n - 1 ) { $ x = ( $ x * $ x ) % $ n ; $ d *= 2 ; if ( $ x == 1 ) return false ; if ( $ x == $ n - 1 ) return true ; }
return false ; }
function isPrime ( $ n , $ k ) {
if ( $ n <= 1 $ n == 4 ) return false ; if ( $ n <= 3 ) return true ;
$ d = $ n - 1 ; while ( $ d % 2 == 0 ) $ d /= 2 ;
for ( $ i = 0 ; $ i < $ k ; $ i ++ ) if ( ! miillerTest ( $ d , $ n ) ) return false ; return true ; }
$ k = 4 ; echo " All ▁ primes ▁ smaller ▁ than ▁ 100 : ▁ STRNEWLINE " ; for ( $ n = 1 ; $ n < 100 ; $ n ++ ) if ( isPrime ( $ n , $ k ) ) echo $ n , " ▁ " ; ? >
< ? php function maxConsecutiveOnes ( $ x ) {
$ count = 0 ;
while ( $ x != 0 ) {
$ x = ( $ x & ( $ x << 1 ) ) ; $ count ++ ; } return $ count ; }
echo maxConsecutiveOnes ( 14 ) , " STRNEWLINE " ; echo maxConsecutiveOnes ( 222 ) , " STRNEWLINE " ; ? >
< ? php function subtract ( $ x , $ y ) {
while ( $ y != 0 ) {
$ borrow = ( ~ $ x ) & $ y ;
$ x = $ x ^ $ y ;
$ y = $ borrow << 1 ; } return $ x ; }
$ x = 29 ; $ y = 13 ; echo " x ▁ - ▁ y ▁ is ▁ " , subtract ( $ x , $ y ) ; ? >
< ? php function subtract ( $ x , $ y ) { if ( $ y == 0 ) return $ x ; return subtract ( $ x ^ $ y , ( ~ $ x & $ y ) << 1 ) ; }
$ x = 29 ; $ y = 13 ; echo " x ▁ - ▁ y ▁ is ▁ " , subtract ( $ x , $ y ) ; # This  code is contributed by ajit NEW_LINE ? >
< ? php $ N = 6 ; $ Even = $ N / 2 ; $ Odd = $ N - $ Even ; echo $ Even * $ Odd ; ? >
< ? php function steps ( $ str , $ n ) {
$ x = 0 ;
for ( $ i = 0 ; $ i < strlen ( $ str ) ; $ i ++ ) {
if ( $ x == 0 ) $ flag = true ;
if ( $ x == $ n - 1 ) $ flag = false ;
for ( $ j = 0 ; $ j < $ x ; $ j ++ ) echo " * " ; echo $ str [ $ i ] , " STRNEWLINE " ;
if ( $ flag == true ) $ x ++ ; else $ x -- ; } }
$ n = 4 ; $ str = " GeeksForGeeks " ; echo " String : ▁ " , $ str , " STRNEWLINE " ; echo " Max ▁ Length ▁ of ▁ Steps : ▁ " , $ n , " STRNEWLINE " ;
steps ( $ str , $ n ) ; ? >
< ? php function isDivisible ( $ str , $ k ) { $ n = strlen ( $ str ) ; $ c = 0 ;
for ( $ i = 0 ; $ i < $ k ; $ i ++ ) if ( $ str [ $ n - $ i - 1 ] == '0' ) $ c ++ ;
return ( $ c == $ k ) ; }
$ str1 = "10101100" ; $ k = 2 ; if ( isDivisible ( $ str1 , $ k ) ) echo " Yes " , " STRNEWLINE " ; else echo " No " , " STRNEWLINE " ;
$ str2 = "111010100" ; $ k = 2 ; if ( isDivisible ( $ str2 , $ k ) ) echo " Yes " , " STRNEWLINE " ; else echo " No " , " STRNEWLINE " ; ? >
< ? php function isNumber ( $ s ) { for ( $ i = 0 ; $ i < strlen ( $ s ) ; $ i ++ ) if ( is_numeric ( $ s [ $ i ] ) == false ) return false ; return true ; }
$ str = "6790" ;
if ( isNumber ( $ str ) ) echo " Integer " ;
else echo " String " ; ? >
< ? php function reverse ( $ str ) { if ( ( $ str == null ) || ( strlen ( $ str ) <= 1 ) ) echo ( $ str ) ; else { echo ( $ str [ strlen ( $ str ) - 1 ] ) ; reverse ( substr ( $ str , 0 , ( strlen ( $ str ) - 1 ) ) ) ; } }
$ str = " Geeks ▁ for ▁ Geeks " ; reverse ( $ str ) ; ? >
< ? php function polyarea ( $ n , $ r ) {
if ( $ r < 0 && $ n < 0 ) return -1 ;
$ A = ( ( $ r * $ r * $ n ) * sin ( ( 360 / $ n ) * 3.14159 / 180 ) ) / 2 ; return $ A ; }
$ r = 9 ; $ n = 6 ; echo polyarea ( $ n , $ r ) . " STRNEWLINE " ; ? >
< ? php function findPCSlope ( $ m ) { return -1.0 / $ m ; }
$ m = 2.0 ; echo findPCSlope ( $ m ) ; ? >
< ? php function area_of_segment ( $ radius , $ angle ) { $ pi = 3.14159 ;
$ area_of_sector = $ pi * ( $ radius * $ radius ) * ( $ angle / 360 ) ;
$ area_of_triangle = 1 / 2 * ( $ radius * $ radius ) * sin ( ( $ angle * $ pi ) / 180 ) ; return $ area_of_sector - $ area_of_triangle ; }
$ radius = 10.0 ; $ angle = 90.0 ; echo ( " Area ▁ of ▁ minor ▁ segment ▁ = ▁ " ) ; echo ( area_of_segment ( $ radius , $ angle ) ) ; echo ( " STRNEWLINE " ) ; echo ( " Area ▁ of ▁ major ▁ segment ▁ = ▁ " ) ; echo ( area_of_segment ( $ radius , ( 360 - $ angle ) ) ) ; ? >
< ? php function SectorArea ( $ radius , $ angle ) { if ( $ angle >= 360 ) echo ( " Angle ▁ not ▁ possible " ) ;
else { $ sector = ( ( 22 * $ radius * $ radius ) / 7 ) * ( $ angle / 360 ) ; echo ( $ sector ) ; } }
$ radius = 9 ; $ angle = 60 ; SectorArea ( $ radius , $ angle ) ; ? >
< ? php function insertionSortRecursive ( & $ arr , $ n ) {
if ( $ n <= 1 ) return ;
insertionSortRecursive ( $ arr , $ n - 1 ) ;
$ last = $ arr [ $ n - 1 ] ; $ j = $ n - 2 ;
while ( $ j >= 0 && $ arr [ $ j ] > $ last ) { $ arr [ $ j + 1 ] = $ arr [ $ j ] ; $ j -- ; } $ arr [ $ j + 1 ] = $ last ; }
$ arr = array ( 12 , 11 , 13 , 5 , 6 ) ; $ n = sizeof ( $ arr ) ; insertionSortRecursive ( $ arr , $ n ) ; printArray ( $ arr , $ n ) ; ? >
< ? php function isWaveArray ( $ arr , $ n ) { $ result = true ;
if ( $ arr [ 1 ] > $ arr [ 0 ] && $ arr [ 1 ] > $ arr [ 2 ] ) { for ( $ i = 1 ; $ i < ( $ n - 1 ) ; $ i += 2 ) { if ( $ arr [ $ i ] > $ arr [ $ i - 1 ] && $ arr [ $ i ] > $ arr [ $ i + 1 ] ) { $ result = true ; } else { $ result = false ; break ; } }
if ( $ result == true && $ n % 2 == 0 ) { if ( $ arr [ $ n - 1 ] <= $ arr [ $ n - 2 ] ) { $ result = false ; } } } else if ( $ arr [ 1 ] < $ arr [ 0 ] && $ arr [ 1 ] < $ arr [ 2 ] ) { for ( $ i = 1 ; $ i < $ n - 1 ; $ i += 2 ) { if ( $ arr [ $ i ] < $ arr [ $ i - 1 ] && $ arr [ $ i ] < $ arr [ $ i + 1 ] ) { $ result = true ; } else { $ result = false ; break ; } }
if ( $ result == true && $ n % 2 == 0 ) { if ( $ arr [ $ n - 1 ] >= $ arr [ $ n - 2 ] ) { $ result = false ; } } } return $ result ; }
$ arr = array ( 1 , 3 , 2 , 4 ) ; $ n = sizeof ( $ arr ) ; if ( isWaveArray ( $ arr , $ n ) ) { echo " YES " ; } else { echo " NO " ; } ? >
< ? php $ mod = 1000000007 ;
function sumOddFibonacci ( $ n ) { global $ mod ; $ Sum [ $ n + 1 ] = array ( ) ;
$ Sum [ 0 ] = 0 ; $ Sum [ 1 ] = 1 ; $ Sum [ 2 ] = 2 ; $ Sum [ 3 ] = 5 ; $ Sum [ 4 ] = 10 ; $ Sum [ 5 ] = 23 ; for ( $ i = 6 ; $ i <= $ n ; $ i ++ ) { $ Sum [ $ i ] = ( ( $ Sum [ $ i - 1 ] + ( 4 * $ Sum [ $ i - 2 ] ) % $ mod - ( 4 * $ Sum [ $ i - 3 ] ) % $ mod + $ mod ) % $ mod + ( $ Sum [ $ i - 4 ] - $ Sum [ $ i - 5 ] + $ mod ) % $ mod ) % $ mod ; } return $ Sum [ $ n ] ; }
$ n = 6 ; echo sumOddFibonacci ( $ n ) ; ? >
< ? php function solve ( $ N , $ K ) {
$ combo [ $ N + 1 ] = array ( ) ;
$ combo [ 0 ] = 1 ;
for ( $ i = 1 ; $ i <= $ K ; $ i ++ ) {
for ( $ j = 0 ; $ j <= $ N ; $ j ++ ) {
if ( $ j >= $ i ) {
$ combo [ $ j ] += $ combo [ $ j - $ i ] ; } } }
return $ combo [ $ N ] ; }
$ N = 29 ; $ K = 5 ; echo solve ( $ N , $ K ) ; solve ( $ N , $ K ) ; ? >
< ? php function computeLIS ( $ circBuff , $ start , $ end , $ n ) { $ LIS = Array ( ) ;
for ( $ i = $ start ; $ i < $ end ; $ i ++ ) $ LIS [ $ i ] = 1 ;
for ( $ i = $ start + 1 ; $ i < $ end ; $ i ++ )
for ( $ j = $ start ; $ j < $ i ; $ j ++ ) if ( $ circBuff [ $ i ] > $ circBuff [ $ j ] && $ LIS [ $ i ] < $ LIS [ $ j ] + 1 ) $ LIS [ $ i ] = $ LIS [ $ j ] + 1 ;
$ res = PHP_INT_MIN ; for ( $ i = $ start ; $ i < $ end ; $ i ++ ) $ res = max ( $ res , $ LIS [ $ i ] ) ; return $ res ; }
function LICS ( $ arr , $ n ) {
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ circBuff [ $ i ] = $ arr [ $ i ] ; for ( $ i = $ n ; $ i < 2 * $ n ; $ i ++ ) $ circBuff [ $ i ] = $ arr [ $ i - $ n ] ;
$ res = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ res = max ( computeLIS ( $ circBuff , $ i , $ i + $ n , $ n ) , $ res ) ; return $ res ; }
$ arr = array ( 1 , 4 , 6 , 2 , 3 ) ; $ n = sizeof ( $ arr ) ; echo " Length ▁ of ▁ LICS ▁ is ▁ " , LICS ( $ arr , $ n ) ; ? >
< ? php function LCIS ( $ arr1 , $ n , $ arr2 , $ m ) {
$ table = Array ( ) ; for ( $ j = 0 ; $ j < $ m ; $ j ++ ) $ table [ $ j ] = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
$ current = 0 ;
for ( $ j = 0 ; $ j < $ m ; $ j ++ ) {
if ( $ arr1 [ $ i ] == $ arr2 [ $ j ] ) if ( $ current + 1 > $ table [ $ j ] ) $ table [ $ j ] = $ current + 1 ;
if ( $ arr1 [ $ i ] > $ arr2 [ $ j ] ) if ( $ table [ $ j ] > $ current ) $ current = $ table [ $ j ] ; } }
$ result = 0 ; for ( $ i = 0 ; $ i < $ m ; $ i ++ ) if ( $ table [ $ i ] > $ result ) $ result = $ table [ $ i ] ; return $ result ; }
$ arr1 = array ( 3 , 4 , 9 , 1 ) ; $ arr2 = array ( 5 , 3 , 8 , 9 , 10 , 2 , 1 ) ; $ n = sizeof ( $ arr1 ) ; $ m = sizeof ( $ arr2 ) ; echo " Length ▁ of ▁ LCIS ▁ is ▁ " , LCIS ( $ arr1 , $ n , $ arr2 , $ m ) ; ? >
< ? php function maxValue ( $ a , $ b ) {
sort ( $ b ) ; $ n = sizeof ( $ a ) ; $ m = sizeof ( $ b ) ;
$ j = $ m - 1 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ j < 0 ) break ; if ( $ b [ $ j ] > $ a [ $ i ] ) { $ a [ $ i ] = $ b [ $ j ] ;
$ j -- ; } }
return $ a ; }
# convert  string into array NEW_LINE $ a = str_split ( "1234" ) ; $ b = str_split ( "4321" ) ; echo maxValue ( $ a , $ b ) ; ? >
< ? php function checkIfUnequal ( $ n , $ q ) {
$ s1 = strval ( $ n ) ; $ a = array_fill ( 0 , 26 , NULL ) ;
for ( $ i = 0 ; $ i < strlen ( $ s1 ) ; $ i ++ ) $ a [ ord ( $ s1 [ $ i ] ) - ord ( '0' ) ] ++ ;
$ prod = $ n * $ q ;
$ s2 = strval ( $ prod ) ;
for ( $ i = 0 ; $ i < strlen ( $ s2 ) ; $ i ++ ) {
if ( $ a [ ord ( $ s2 [ $ i ] ) - ord ( '0' ) ] ) return false ; }
return true ; }
function countInRange ( $ l , $ r , $ q ) { $ count = 0 ; for ( $ i = $ l ; $ i <= $ r ; $ i ++ ) {
if ( checkIfUnequal ( $ i , $ q ) ) $ count ++ ; } return $ count ; }
$ l = 10 ; $ r = 12 ; $ q = 2 ;
echo countInRange ( $ l , $ r , $ q ) ; ? >
< ? php function is_possible ( $ s ) {
$ l = strlen ( $ s ) ; $ one = 0 ; $ zero = 0 ; for ( $ i = 0 ; $ i < $ l ; $ i ++ ) {
if ( $ s [ $ i ] == '0' ) $ zero ++ ;
else $ one ++ ; }
if ( $ l % 2 == 0 ) return ( $ one == $ zero ) ;
else return ( abs ( $ one - $ zero ) == 1 ) ; }
$ s = "100110" ; if ( is_possible ( $ s ) ) echo ( " Yes " ) ; else echo ( " No " ) ; ? >
< ? php function convert ( $ s ) { $ n = strlen ( $ s ) ; $ s [ 0 ] = strtolower ( $ s [ 0 ] ) ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) {
if ( $ s [ $ i ] == ' ▁ ' && $ i < $ n ) {
$ s [ $ i + 1 ] = strtolower ( $ s [ $ i + 1 ] ) ; $ i ++ ; }
else $ s [ $ i ] = strtoupper ( $ s [ $ i ] ) ; }
return $ s ; }
$ str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " ; echo ( convert ( $ str ) ) ; ? >
< ? php function change_case ( $ a ) { $ l = strlen ( $ a ) ; for ( $ i = 0 ; $ i < $ l ; $ i ++ ) {
if ( $ a [ $ i ] >= ' a ' && $ a [ $ i ] <= ' z ' ) $ a [ $ i ] = chr ( 65 + ( ord ( $ a [ $ i ] ) - ord ( ' a ' ) ) ) ;
else if ( $ a [ $ i ] >= ' A ' && $ a [ $ i ] <= ' Z ' ) $ a [ $ i ] = chr ( 97 + ( ord ( $ a [ $ i ] ) - ord ( ' a ' ) ) ) ; } return $ a ; }
function delete_vowels ( $ a ) { $ temp = " " ; $ l = strlen ( $ a ) ; for ( $ i = 0 ; $ i < $ l ; $ i ++ ) {
if ( $ a [ $ i ] != ' a ' && $ a [ $ i ] != ' e ' && $ a [ $ i ] != ' i ' && $ a [ $ i ] != ' o ' && $ a [ $ i ] != ' u ' && $ a [ $ i ] != ' A ' && $ a [ $ i ] != ' E ' && $ a [ $ i ] != ' O ' && $ a [ $ i ] != ' U ' && $ a [ $ i ] != ' I ' ) $ temp = $ temp . $ a [ $ i ] ; } return $ temp ; }
function insert_hash ( $ a ) { $ temp = " " ; $ l = strlen ( $ a ) ; for ( $ i = 0 ; $ i < $ l ; $ i ++ ) {
if ( ( $ a [ $ i ] >= ' a ' && $ a [ $ i ] <= ' z ' ) || ( $ a [ $ i ] >= ' A ' && $ a [ $ i ] <= ' Z ' ) ) $ temp = $ temp . ' # ' . $ a [ $ i ] ; else $ temp = $ temp . $ a [ $ i ] ; } return $ temp ; }
function transformSting ( $ a ) { $ b = delete_vowels ( $ a ) ; $ c = change_case ( $ b ) ; $ d = insert_hash ( $ c ) ; echo ( $ d ) ; }
$ a = " SunshinE ! ! " ;
transformSting ( $ a ) ; ? >
< ? php function findNthNo ( $ n ) { $ res = " " ; while ( $ n >= 1 ) {
if ( $ n & 1 ) { $ res = $ res + "3" ; $ n = ( $ n - 1 ) / 2 ; }
else { $ res = $ res . "5" ; $ n = ( $ n - 2 ) / 2 ; } }
$ res = strrev ( $ res ) ; return $ res ; }
$ n = 5 ; echo findNthNo ( $ n ) ; ? >
< ? php function findNthNonSquare ( $ n ) {
$ x = $ n ;
$ ans = $ x + floor ( 0.5 + sqrt ( $ x ) ) ; return ( int ) $ ans ; }
$ n = 16 ;
echo " The ▁ " . $ n . " th ▁ Non - Square ▁ number ▁ is ▁ " ; echo findNthNonSquare ( $ n ) ;
< ? php function seiresSum ( $ n , $ a ) { return $ n * ( $ a [ 0 ] * $ a [ 0 ] - $ a [ 2 * $ n - 1 ] * $ a [ 2 * $ n - 1 ] ) / ( 2 * $ n - 1 ) ; }
$ n = 2 ; $ a = array ( 1 , 2 , 3 , 4 ) ; echo seiresSum ( $ n , $ a ) ; ? >
< ? php function checkdigit ( $ n , $ k ) { while ( $ n ) {
$ rem = $ n % 10 ;
if ( $ rem == $ k ) return 1 ; $ n = $ n / 10 ; } return 0 ; }
function findNthNumber ( $ n , $ k ) {
for ( $ i = $ k + 1 , $ count = 1 ; $ count < $ n ; $ i ++ ) {
if ( checkdigit ( $ i , $ k ) || ( $ i % $ k == 0 ) ) $ count ++ ; if ( $ count == $ n ) return $ i ; } return -1 ; }
$ n = 10 ; $ k = 2 ; echo findNthNumber ( $ n , $ k ) ; ? >
< ? php function middleOfThree ( $ a , $ b , $ c ) {
function middleOfThree ( $ a , $ b , $ c ) {
if ( ( $ a < $ b && $ b < $ c ) or ( $ c < $ b && $ b < $ a ) ) return $ b ;
else if ( ( $ b < $ a and $ a < $ c ) or ( $ c < $ a and $ a < $ b ) ) return $ a ; else return $ c ; }
$ a = 20 ; $ b = 30 ; $ c = 40 ; echo middleOfThree ( $ a , $ b , $ c ) ; ? >
< ? php $ INF = PHP_INT_MAX ; $ N = 4 ;
function minCost ( $ cost ) { global $ INF ; global $ N ;
$ dist [ $ N ] = array ( ) ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) $ dist [ $ i ] = $ INF ; $ dist [ 0 ] = 0 ;
for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = $ i + 1 ; $ j < $ N ; $ j ++ ) if ( $ dist [ $ j ] > $ dist [ $ i ] + $ cost [ $ i ] [ $ j ] ) $ dist [ $ j ] = $ dist [ $ i ] + $ cost [ $ i ] [ $ j ] ; return $ dist [ $ N - 1 ] ; }
$ cost = array ( array ( 0 , 15 , 80 , 90 ) , array ( INF , 0 , 40 , 50 ) , array ( INF , INF , 0 , 70 ) , array ( INF , INF , INF , 0 ) ) ; echo " The ▁ Minimum ▁ cost ▁ to ▁ reach ▁ station ▁ " , $ N , " ▁ is ▁ " , minCost ( $ cost ) ; ? >
< ? php function numOfways ( $ n , $ k ) { $ p = 1 ; if ( $ k % 2 ) $ p = -1 ; return ( pow ( $ n - 1 , $ k ) + $ p * ( $ n - 1 ) ) / $ n ; }
$ n = 4 ; $ k = 2 ; echo numOfways ( $ n , $ k ) ; ? >
< ? php function length_of_chord ( $ r , $ x ) { echo " The ▁ length ▁ of ▁ the ▁ chord " , " ▁ of ▁ the ▁ circle ▁ is ▁ " , 2 * $ r * sin ( $ x * ( 3.14 / 180 ) ) ; }
$ r = 4 ; $ x = 63 ; length_of_chord ( $ r , $ x ) ; ? >
< ? php function area ( $ a ) {
if ( $ a < 0 ) return -1 ;
$ area = sqrt ( $ a ) / 6 ; return $ area ; }
$ a = 10 ; echo area ( $ a ) ; ? >
< ? php function longestRodInCuboid ( $ length , $ breadth , $ height ) { $ result ; $ temp ;
$ temp = $ length * $ length + $ breadth * $ breadth + $ height * $ height ;
$ result = sqrt ( $ temp ) ; return $ result ; }
$ length = 12 ; $ breadth = 9 ; $ height = 8 ;
echo longestRodInCuboid ( $ length , $ breadth , $ height ) ; ? >
< ? php function LiesInsieRectangle ( $ a , $ b , $ x , $ y ) { if ( $ x - $ y - $ b <= 0 && $ x - $ y + $ b >= 0 && $ x + $ y - 2 * $ a + $ b <= 0 && $ x + $ y - $ b >= 0 ) return true ; return false ; }
$ a = 7 ; $ b = 2 ; $ x = 4 ; $ y = 5 ; if ( LiesInsieRectangle ( $ a , $ b , $ x , $ y ) ) echo " Given ▁ point ▁ lies ▁ " . " inside ▁ the ▁ rectangle " ; else echo " Given ▁ point ▁ does ▁ not " . " ▁ lie ▁ on ▁ the ▁ rectangle " ; ? >
< ? php function maxvolume ( $ s ) { $ maxvalue = 0 ;
for ( $ i = 1 ; $ i <= $ s - 2 ; $ i ++ ) {
for ( $ j = 1 ; $ j <= $ s - 1 ; $ j ++ ) {
$ k = $ s - $ i - $ j ;
$ maxvalue = max ( $ maxvalue , $ i * $ j * $ k ) ; } } return $ maxvalue ; }
$ s = 8 ; echo ( maxvolume ( $ s ) ) ; ? >
< ? php function maxvolume ( $ s ) {
$ length = ( int ) ( $ s / 3 ) ; $ s -= $ length ;
$ breadth = ( int ) ( $ s / 2 ) ;
$ height = $ s - $ breadth ; return $ length * $ breadth * $ height ; }
$ s = 8 ; echo ( maxvolume ( $ s ) ) ; ? >
< ? php function hexagonArea ( $ s ) { return ( ( 3 * sqrt ( 3 ) * ( $ s * $ s ) ) / 2 ) ; }
$ s = 4 ; echo ( " Area ▁ : ▁ " ) ; echo ( hexagonArea ( $ s ) ) ; ? >
< ? php function maxSquare ( $ b , $ m ) {
return ( $ b / $ m - 1 ) * ( $ b / $ m ) / 2 ; }
$ b = 10 ; $ m = 2 ; echo maxSquare ( $ b , $ m ) ;
< ? php function findRightAngle ( $ A , $ H ) {
$ D = pow ( $ H , 4 ) - 16 * $ A * $ A ; if ( $ D >= 0 ) {
$ root1 = ( $ H * $ H + sqrt ( $ D ) ) / 2 ; $ root2 = ( $ H * $ H - sqrt ( $ D ) ) / 2 ; $ a = sqrt ( $ root1 ) ; $ b = sqrt ( $ root2 ) ; if ( $ b >= $ a ) echo $ a , " ▁ " , $ b , " ▁ " , $ H ; else echo $ b , " ▁ " , $ a , " ▁ " , $ H ; } else echo " - 1" ; }
findRightAngle ( 6 , 5 ) ;
< ? php function numberOfSquares ( $ base ) {
$ base = ( $ base - 2 ) ;
$ base = intdiv ( $ base , 2 ) ; return $ base * ( $ base + 1 ) / 2 ; }
$ base = 8 ; echo numberOfSquares ( $ base ) ; ? >
< ? php function fib ( $ n ) { if ( $ n <= 1 ) return $ n ; return fib ( $ n - 1 ) + fib ( $ n - 2 ) ; }
function findVertices ( $ n ) {
return fib ( $ n + 2 ) ; }
$ n = 3 ; echo findVertices ( $ n ) ; ? >
< ? php $ MAX_SIZE = 10 ;
function sortByRow ( & $ mat , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ )
sort ( $ mat [ $ i ] ) ; }
function transpose ( & $ mat , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) {
$ t = $ mat [ $ i ] [ $ j ] ; $ mat [ $ i ] [ $ j ] = $ mat [ $ j ] [ $ i ] ; $ mat [ $ j ] [ $ i ] = $ t ; } } }
function sortMatRowAndColWise ( & $ mat , $ n ) {
sortByRow ( $ mat , $ n ) ;
transpose ( $ mat , $ n ) ;
sortByRow ( $ mat , $ n ) ;
transpose ( $ mat , $ n ) ; }
function printMat ( & $ mat , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = 0 ; $ j < $ n ; $ j ++ ) echo $ mat [ $ i ] [ $ j ] . " ▁ " ; echo " STRNEWLINE " ; } }
$ mat = array ( array ( 4 , 1 , 3 ) , array ( 9 , 6 , 8 ) , array ( 5 , 2 , 7 ) ) ; $ n = 3 ; echo " Original ▁ Matrix : STRNEWLINE " ; printMat ( $ mat , $ n ) ; sortMatRowAndColWise ( $ mat , $ n ) ; echo " Matrix After Sorting : " ; printMat ( $ mat , $ n ) ; ? >
< ? php function doublyEven ( $ n ) { $ arr = array_fill ( 0 , $ n , array_fill ( 0 , $ n , 0 ) ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) for ( $ j = 0 ; $ j < $ n ; $ j ++ ) $ arr [ $ i ] [ $ j ] = ( $ n * $ i ) + $ j + 1 ;
for ( $ i = 0 ; $ i < $ n / 4 ; $ i ++ ) for ( $ j = 0 ; $ j < $ n / 4 ; $ j ++ ) $ arr [ $ i ] [ $ j ] = ( $ n * $ n + 1 ) - $ arr [ $ i ] [ $ j ] ;
for ( $ i = 0 ; $ i < $ n / 4 ; $ i ++ ) for ( $ j = 3 * ( $ n / 4 ) ; $ j < $ n ; $ j ++ ) $ arr [ $ i ] [ $ j ] = ( $ n * $ n + 1 ) - $ arr [ $ i ] [ $ j ] ;
for ( $ i = 3 * $ n / 4 ; $ i < $ n ; $ i ++ ) for ( $ j = 0 ; $ j < $ n / 4 ; $ j ++ ) $ arr [ $ i ] [ $ j ] = ( $ n * $ n + 1 ) - $ arr [ $ i ] [ $ j ] ;
for ( $ i = 3 * $ n / 4 ; $ i < $ n ; $ i ++ ) for ( $ j = 3 * $ n / 4 ; $ j < $ n ; $ j ++ ) $ arr [ $ i ] [ $ j ] = ( $ n * $ n + 1 ) - $ arr [ $ i ] [ $ j ] ;
for ( $ i = $ n / 4 ; $ i < 3 * $ n / 4 ; $ i ++ ) for ( $ j = $ n / 4 ; $ j < 3 * $ n / 4 ; $ j ++ ) $ arr [ $ i ] [ $ j ] = ( $ n * $ n + 1 ) - $ arr [ $ i ] [ $ j ] ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = 0 ; $ j < $ n ; $ j ++ ) echo $ arr [ $ i ] [ $ j ] . " ▁ " ; echo " STRNEWLINE " ; } }
$ n = 8 ;
doublyEven ( $ n ) ; ? >
< ? php $ cola = 2 ; $ rowa = 3 ; $ colb = 3 ; $ rowb = 2 ;
function Kroneckerproduct ( $ A , $ B ) { global $ cola ; global $ rowa ; global $ colb ; global $ rowb ; $ C ;
for ( $ i = 0 ; $ i < $ rowa ; $ i ++ ) {
for ( $ k = 0 ; $ k < $ rowb ; $ k ++ ) {
for ( $ j = 0 ; $ j < $ cola ; $ j ++ ) {
for ( $ l = 0 ; $ l < $ colb ; $ l ++ ) {
$ C [ $ i + $ l + 1 ] [ $ j + $ k + 1 ] = $ A [ $ i ] [ $ j ] * $ B [ $ k ] [ $ l ] ; echo ( $ C [ $ i + $ l + 1 ] [ $ j + $ k + 1 ] ) , " TABSYMBOL " ; } } echo " " } } }
$ A = array ( array ( 1 , 2 ) , array ( 3 , 4 ) , array ( 1 , 0 ) ) ; $ B = array ( array ( 0 , 5 , 2 ) , array ( 6 , 7 , 3 ) ) ; Kroneckerproduct ( $ A , $ B ) ; ? >
< ? php $ N = 4 ;
function isLowerTriangularMatrix ( $ mat ) { global $ N ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = $ i + 1 ; $ j < $ N ; $ j ++ ) if ( $ mat [ $ i ] [ $ j ] != 0 ) return false ; return true ; }
$ mat = array ( array ( 1 , 0 , 0 , 0 ) , array ( 1 , 4 , 0 , 0 ) , array ( 4 , 6 , 2 , 0 ) , array ( 0 , 4 , 7 , 6 ) ) ;
if ( isLowerTriangularMatrix ( $ mat ) ) echo ( " Yes " ) ; else echo ( " No " ) ; ? >
< ? php $ N = 4 ;
function isUpperTriangularMatrix ( $ mat ) { global $ N ; for ( $ i = 1 ; $ i < $ N ; $ i ++ ) for ( $ j = 0 ; $ j < $ i ; $ j ++ ) if ( $ mat [ $ i ] [ $ j ] != 0 ) return false ; return true ; }
$ mat = array ( array ( 1 , 3 , 5 , 3 ) , array ( 0 , 4 , 6 , 2 ) , array ( 0 , 0 , 2 , 5 ) , array ( 0 , 0 , 0 , 6 ) ) ; if ( isUpperTriangularMatrix ( $ mat ) ) echo " Yes " ; else echo " No " ; ? >
< ? php $ m = 3 ;
$ n = 2 ;
function countSets ( $ a ) { global $ m , $ n ;
$ res = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ u = 0 ; $ v = 0 ; for ( $ j = 0 ; $ j < $ m ; $ j ++ ) $ a [ $ i ] [ $ j ] ? $ u ++ : $ v ++ ; $ res += pow ( 2 , $ u ) - 1 + pow ( 2 , $ v ) - 1 ; }
for ( $ i = 0 ; $ i < $ m ; $ i ++ ) { $ u = 0 ; $ v = 0 ; for ( $ j = 0 ; $ j < $ n ; $ j ++ ) $ a [ $ j ] [ $ i ] ? $ u ++ : $ v ++ ; $ res += pow ( 2 , $ u ) - 1 + pow ( 2 , $ v ) - 1 ; }
return $ res - ( $ n * $ m ) ; }
$ a = array ( array ( 1 , 0 , 1 ) , array ( 0 , 1 , 0 ) ) ; echo countSets ( $ a ) ; ? >
< ? php for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = 0 ; $ j < $ N ; $ j ++ ) if ( $ mat [ $ i ] [ $ j ] != $ tr [ $ i ] [ $ j ] ) return false ; return true ; }
function isSymmetric ( $ mat , $ N ) { $ tr = array ( array ( ) ) ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = 0 ; $ j < $ N ; $ j ++ ) $ tr [ $ i ] [ $ j ] = $ mat [ $ j ] [ $ i ] ;
$ mat = array ( array ( 1 , 3 , 5 ) , array ( 3 , 2 , 4 ) , array ( 5 , 4 , 1 ) ) ; if ( isSymmetric ( $ mat , 3 ) ) echo " Yes " ; else echo " No " ; ? >
< ? php $ MAX = 100 ;
function isSymmetric ( $ mat , $ N ) { for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = 0 ; $ j < $ N ; $ j ++ ) if ( $ mat [ $ i ] [ $ j ] != $ mat [ $ j ] [ $ i ] ) return false ; return true ; }
$ mat = array ( array ( 1 , 3 , 5 ) , array ( 3 , 2 , 4 ) , array ( 5 , 4 , 1 ) ) ; if ( isSymmetric ( $ mat , 3 ) ) echo ( " Yes " ) ; else echo ( " No " ) ; ? >
function findNormal ( $ mat , $ n ) { $ sum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) for ( $ j = 0 ; $ j < $ n ; $ j ++ ) $ sum += $ mat [ $ i ] [ $ j ] * $ mat [ $ i ] [ $ j ] ; return floor ( sqrt ( $ sum ) ) ; }
function findTrace ( $ mat , $ n ) { $ sum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ sum += $ mat [ $ i ] [ $ i ] ; return $ sum ; }
$ mat = array ( array ( 1 , 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 , 3 ) , array ( 4 , 4 , 4 , 4 , 4 ) , array ( 5 , 5 , 5 , 5 , 5 ) ) ; echo " Trace ▁ of ▁ Matrix ▁ = ▁ " , findTrace ( $ mat , 5 ) , " STRNEWLINE " ; echo " Normal ▁ of ▁ Matrix ▁ = ▁ " , findNormal ( $ mat , 5 ) ; ? >
< ? php function maxDet ( $ n ) { return ( 2 * $ n * $ n * $ n ) ; }
function resMatrix ( $ n ) { for ( $ i = 0 ; $ i < 3 ; $ i ++ ) { for ( $ j = 0 ; $ j < 3 ; $ j ++ ) {
if ( $ i == 0 && $ j == 2 ) echo "0 ▁ " ; else if ( $ i == 1 && $ j == 0 ) echo "0 ▁ " ; else if ( $ i == 2 && $ j == 1 ) echo "0 ▁ " ;
else echo $ n , " " ; } echo " " } }
$ n = 15 ; echo " Maximum ▁ Determinant ▁ = ▁ " , maxDet ( $ n ) ; echo " Resultant Matrix : " resMatrix ( $ n ) ; ? >
< ? php function countNegative ( $ M , $ n , $ m ) { $ count = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = 0 ; $ j < $ m ; $ j ++ ) { if ( $ M [ $ i ] [ $ j ] < 0 ) $ count += 1 ;
else break ; } } return $ count ; }
$ M = array ( array ( -3 , -2 , -1 , 1 ) , array ( -2 , 2 , 3 , 4 ) , array ( 4 , 5 , 7 , 8 ) ) ; echo countNegative ( $ M , 3 , 4 ) ; ? >
< ? php function countNegative ( $ M , $ n , $ m ) {
$ count = 0 ;
$ i = 0 ; $ j = $ m - 1 ;
while ( $ j >= 0 and $ i < $ n ) { if ( $ M [ $ i ] [ $ j ] < 0 ) {
$ count += $ j + 1 ;
$ i += 1 ; }
else $ j -= 1 ; } return $ count ; }
$ M = array ( array ( -3 , -2 , -1 , 1 ) , array ( -2 , 2 , 3 , 4 ) , array ( 4 , 5 , 7 , 8 ) ) ; echo countNegative ( $ M , 3 , 4 ) ; return 0 ; ? >
< ? php $ N = 5 ;
function findMaxValue ( & $ mat ) { global $ N ;
$ maxValue = PHP_INT_MIN ;
for ( $ a = 0 ; $ a < $ N - 1 ; $ a ++ ) for ( $ b = 0 ; $ b < $ N - 1 ; $ b ++ ) for ( $ d = $ a + 1 ; $ d < $ N ; $ d ++ ) for ( $ e = $ b + 1 ; $ e < $ N ; $ e ++ ) if ( $ maxValue < ( $ mat [ $ d ] [ $ e ] - $ mat [ $ a ] [ $ b ] ) ) $ maxValue = $ mat [ $ d ] [ $ e ] - $ mat [ $ a ] [ $ b ] ; return $ maxValue ; }
$ mat = array ( array ( 1 , 2 , -1 , -4 , -20 ) , array ( -8 , -3 , 4 , 2 , 1 ) , array ( 3 , 8 , 6 , 1 , 3 ) , array ( -4 , -1 , 1 , 7 , -6 ) , array ( 0 , -4 , 10 , -5 , 1 ) ) ; echo " Maximum ▁ Value ▁ is ▁ " . findMaxValue ( $ mat ) ; ? >
< ? php $ N = 5 ;
function findMaxValue ( $ mat ) { global $ N ;
$ maxValue = PHP_INT_MIN ;
$ maxArr [ $ N ] [ $ N ] = array ( ) ;
$ maxArr [ $ N - 1 ] [ $ N - 1 ] = $ mat [ $ N - 1 ] [ $ N - 1 ] ;
$ maxv = $ mat [ $ N - 1 ] [ $ N - 1 ] ; for ( $ j = $ N - 2 ; $ j >= 0 ; $ j -- ) { if ( $ mat [ $ N - 1 ] [ $ j ] > $ maxv ) $ maxv = $ mat [ $ N - 1 ] [ $ j ] ; $ maxArr [ $ N - 1 ] [ $ j ] = $ maxv ; }
$ maxv = $ mat [ $ N - 1 ] [ $ N - 1 ] ; for ( $ i = $ N - 2 ; $ i >= 0 ; $ i -- ) { if ( $ mat [ $ i ] [ $ N - 1 ] > $ maxv ) $ maxv = $ mat [ $ i ] [ $ N - 1 ] ; $ maxArr [ $ i ] [ $ N - 1 ] = $ maxv ; }
for ( $ i = $ N - 2 ; $ i >= 0 ; $ i -- ) { for ( $ j = $ N - 2 ; $ j >= 0 ; $ j -- ) {
if ( $ maxArr [ $ i + 1 ] [ $ j + 1 ] - $ mat [ $ i ] [ $ j ] > $ maxValue ) $ maxValue = $ maxArr [ $ i + 1 ] [ $ j + 1 ] - $ mat [ $ i ] [ $ j ] ;
$ maxArr [ $ i ] [ $ j ] = max ( $ mat [ $ i ] [ $ j ] , max ( $ maxArr [ $ i ] [ $ j + 1 ] , $ maxArr [ $ i + 1 ] [ $ j ] ) ) ; } } return $ maxValue ; }
$ mat = array ( array ( 1 , 2 , -1 , -4 , -20 ) , array ( -8 , -3 , 4 , 2 , 1 ) , array ( 3 , 8 , 6 , 1 , 3 ) , array ( -4 , -1 , 1 , 7 , -6 ) , array ( 0 , -4 , 10 , -5 , 1 ) ) ; echo " Maximum ▁ Value ▁ is ▁ " . findMaxValue ( $ mat ) ; ? >
< ? php $ n = 5 ;
function printSumSimple ( $ mat , $ k ) { global $ n ;
if ( $ k > $ n ) return ;
for ( $ i = 0 ; $ i < $ n - $ k + 1 ; $ i ++ ) {
for ( $ j = 0 ; $ j < $ n - $ k + 1 ; $ j ++ ) {
$ sum = 0 ; for ( $ p = $ i ; $ p < $ k + $ i ; $ p ++ ) for ( $ q = $ j ; $ q < $ k + $ j ; $ q ++ ) $ sum += $ mat [ $ p ] [ $ q ] ; echo $ sum , " " ; }
echo " STRNEWLINE " ; } }
$ mat = array ( array ( 1 , 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 , 2 , ) , array ( 3 , 3 , 3 , 3 , 3 , ) , array ( 4 , 4 , 4 , 4 , 4 , ) , array ( 5 , 5 , 5 , 5 , 5 ) ) ; $ k = 3 ; printSumSimple ( $ mat , $ k ) ; ? >
< ? php $ n = 5 ;
function printSumTricky ( $ mat , $ k ) { global $ n ;
if ( $ k > $ n ) return ;
$ stripSum = array ( array ( ) ) ;
for ( $ j = 0 ; $ j < $ n ; $ j ++ ) {
$ sum = 0 ; for ( $ i = 0 ; $ i < $ k ; $ i ++ ) $ sum += $ mat [ $ i ] [ $ j ] ; $ stripSum [ 0 ] [ $ j ] = $ sum ;
for ( $ i = 1 ; $ i < $ n - $ k + 1 ; $ i ++ ) { $ sum += ( $ mat [ $ i + $ k - 1 ] [ $ j ] - $ mat [ $ i - 1 ] [ $ j ] ) ; $ stripSum [ $ i ] [ $ j ] = $ sum ; } }
for ( $ i = 0 ; $ i < $ n - $ k + 1 ; $ i ++ ) {
$ sum = 0 ; for ( $ j = 0 ; $ j < $ k ; $ j ++ ) $ sum += $ stripSum [ $ i ] [ $ j ] ; echo $ sum , " " ;
for ( $ j = 1 ; $ j < $ n - $ k + 1 ; $ j ++ ) { $ sum += ( $ stripSum [ $ i ] [ $ j + $ k - 1 ] - $ stripSum [ $ i ] [ $ j - 1 ] ) ; echo $ sum , " " ; } echo " STRNEWLINE " ; } }
$ mat = array ( array ( 1 , 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 , 3 ) , array ( 4 , 4 , 4 , 4 , 4 ) , array ( 5 , 5 , 5 , 5 , 5 ) ) ; $ k = 3 ; printSumTricky ( $ mat , $ k ) ; ? >
< ? php $ N = 4 ; $ M = 3 ;
function transpose ( & $ A , & $ B ) { for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = 0 ; $ j < $ M ; $ j ++ ) $ B [ $ i ] [ $ j ] = $ A [ $ j ] [ $ i ] ; }
$ A = array ( array ( 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 ) ) ; $ N = 4 ; $ M = 3 ; transpose ( $ A , $ B ) ; echo " Result ▁ matrix ▁ is ▁ STRNEWLINE " ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) { for ( $ j = 0 ; $ j < $ M ; $ j ++ ) { echo $ B [ $ i ] [ $ j ] ; echo " ▁ " ; } echo " STRNEWLINE " ; } ? >
< ? php $ N = 4 ;
function transpose ( & $ A ) { for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = $ i + 1 ; $ j < $ N ; $ j ++ ) { $ temp = $ A [ $ i ] [ $ j ] ; $ A [ $ i ] [ $ j ] = $ A [ $ j ] [ $ i ] ; $ A [ $ j ] [ $ i ] = $ temp ; } }
$ N = 4 ; $ A = array ( array ( 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 ) , array ( 4 , 4 , 4 , 4 ) ) ; transpose ( $ A ) ; echo " Modified ▁ matrix ▁ is ▁ " . " STRNEWLINE " ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) { for ( $ j = 0 ; $ j < $ N ; $ j ++ ) echo $ A [ $ i ] [ $ j ] . " ▁ " ; echo " STRNEWLINE " ; } ? >
< ? php $ R = 3 ; $ C = 3 ;
function pathCountRec ( $ mat , $ m , $ n , $ k ) {
if ( $ m < 0 or $ n < 0 ) return 0 ; if ( $ m == 0 and $ n == 0 ) return ( $ k == $ mat [ $ m ] [ $ n ] ) ;
return pathCountRec ( $ mat , $ m - 1 , $ n , $ k - $ mat [ $ m ] [ $ n ] ) + pathCountRec ( $ mat , $ m , $ n - 1 , $ k - $ mat [ $ m ] [ $ n ] ) ; }
function pathCount ( $ mat , $ k ) { global $ R , $ C ; return pathCountRec ( $ mat , $ R - 1 , $ C - 1 , $ k ) ; }
$ k = 12 ; $ mat = array ( array ( 1 , 2 , 3 ) , array ( 4 , 6 , 5 ) , array ( 3 , 2 , 1 ) ) ; echo pathCount ( $ mat , $ k ) ; ? >
< ? php function selection_sort ( & $ arr , $ n ) {
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
$ low = $ i ; for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) { if ( $ arr [ $ j ] < $ arr [ $ low ] ) { $ low = $ j ; } }
if ( $ arr [ $ i ] > $ arr [ $ low ] ) { $ tmp = $ arr [ $ i ] ; $ arr [ $ i ] = $ arr [ $ low ] ; $ arr [ $ low ] = $ tmp ; } } }
$ arr = array ( 64 , 25 , 12 , 22 , 11 ) ; $ len = count ( $ arr ) ; selection_sort ( $ arr , $ len ) ; echo " Sorted ▁ array ▁ : ▁ STRNEWLINE " ; for ( $ i = 0 ; $ i < $ len ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; ? >
< ? php function bubbleSort ( & $ arr ) { $ n = sizeof ( $ arr ) ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ swapped = False ; for ( $ j = 0 ; $ j < $ n - $ i - 1 ; $ j ++ ) { if ( $ arr [ $ j ] > $ arr [ $ j + 1 ] ) {
$ t = $ arr [ $ j ] ; $ arr [ $ j ] = $ arr [ $ j + 1 ] ; $ arr [ $ j + 1 ] = $ t ; $ swapped = True ; } }
if ( $ swapped == False ) break ; } }
$ arr = array ( 64 , 34 , 25 , 12 , 22 , 11 , 90 ) ; $ len = sizeof ( $ arr ) ; bubbleSort ( $ arr ) ; echo " Sorted ▁ array ▁ : ▁ STRNEWLINE " ; for ( $ i = 0 ; $ i < $ len ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; ? >
< ? php function findCrossOver ( $ arr , $ low , $ high , $ x ) {
if ( $ arr [ $ high ] <= $ x ) return $ high ;
if ( $ arr [ $ low ] > $ x ) return $ low ;
$ mid = ( $ low + $ high ) / 2 ;
if ( $ arr [ $ mid ] <= $ x and $ arr [ $ mid + 1 ] > $ x ) return $ mid ;
if ( $ arr [ $ mid ] < $ x ) return findCrossOver ( $ arr , $ mid + 1 , $ high , $ x ) ; return findCrossOver ( $ arr , $ low , $ mid - 1 , $ x ) ; }
function printKclosest ( $ arr , $ x , $ k , $ n ) {
$ l = findCrossOver ( $ arr , 0 , $ n - 1 , $ x ) ;
$ r = $ l + 1 ;
$ count = 0 ;
if ( $ arr [ $ l ] == $ x ) $ l -- ;
while ( $ l >= 0 and $ r < $ n and $ count < $ k ) { if ( $ x - $ arr [ $ l ] < $ arr [ $ r ] - $ x ) echo $ arr [ $ l -- ] , " ▁ " ; else echo $ arr [ $ r ++ ] , " ▁ " ; $ count ++ ; }
while ( $ count < $ k and $ l >= 0 ) echo $ arr [ $ l -- ] , " ▁ " ; $ count ++ ;
while ( $ count < $ k and $ r < $ n ) echo $ arr [ $ r ++ ] ; $ count ++ ; }
$ arr = array ( 12 , 16 , 22 , 30 , 35 , 39 , 42 , 45 , 48 , 50 , 53 , 55 , 56 ) ; $ n = count ( $ arr ) ; $ x = 35 ; $ k = 4 ; printKclosest ( $ arr , $ x , 4 , $ n ) ; ? >
< ? php function coun ( $ S , $ m , $ n ) {
if ( $ n == 0 ) return 1 ;
if ( $ n < 0 ) return 0 ;
if ( $ m <= 0 && $ n >= 1 ) return 0 ;
return coun ( $ S , $ m - 1 , $ n ) + coun ( $ S , $ m , $ n - $ S [ $ m - 1 ] ) ; }
$ arr = array ( 1 , 2 , 3 ) ; $ m = count ( $ arr ) ; echo coun ( $ arr , $ m , 4 ) ; ? >
< ? php function count_1 ( & $ S , $ m , $ n ) {
$ table = array_fill ( 0 , $ n + 1 , NULl ) ;
$ table [ 0 ] = 1 ;
for ( $ i = 0 ; $ i < $ m ; $ i ++ ) for ( $ j = $ S [ $ i ] ; $ j <= $ n ; $ j ++ ) $ table [ $ j ] += $ table [ $ j - $ S [ $ i ] ] ; return $ table [ $ n ] ; }
$ arr = array ( 1 , 2 , 3 ) ; $ m = sizeof ( $ arr ) ; $ n = 4 ; $ x = count_1 ( $ arr , $ m , $ n ) ; echo $ x ; ? >
< ? php function MatrixChainOrder ( $ p , $ n ) {
$ m [ ] [ ] = array ( $ n , $ n ) ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ m [ $ i ] [ $ i ] = 0 ;
for ( $ L = 2 ; $ L < $ n ; $ L ++ ) { for ( $ i = 1 ; $ i < $ n - $ L + 1 ; $ i ++ ) { $ j = $ i + $ L - 1 ; if ( $ j == $ n ) continue ; $ m [ $ i ] [ $ j ] = PHP_INT_MAX ; for ( $ k = $ i ; $ k <= $ j - 1 ; $ k ++ ) {
$ q = $ m [ $ i ] [ $ k ] + $ m [ $ k + 1 ] [ $ j ] + $ p [ $ i - 1 ] * $ p [ $ k ] * $ p [ $ j ] ; if ( $ q < $ m [ $ i ] [ $ j ] ) $ m [ $ i ] [ $ j ] = $ q ; } } } return $ m [ 1 ] [ $ n - 1 ] ; }
$ arr = array ( 1 , 2 , 3 , 4 ) ; $ size = sizeof ( $ arr ) ; echo " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " . MatrixChainOrder ( $ arr , $ size ) ; ? >
< ? php function cutRod ( $ price , $ n ) { if ( $ n <= 0 ) return 0 ; $ max_val = PHP_INT_MIN ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ max_val = max ( $ max_val , $ price [ $ i ] + cutRod ( $ price , $ n - $ i - 1 ) ) ; return $ max_val ; }
$ arr = array ( 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ) ; $ size = count ( $ arr ) ; echo " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " , cutRod ( $ arr , $ size ) ; ? >
< ? php function cutRod ( $ price , $ n ) { $ val = array ( ) ; $ val [ 0 ] = 0 ; $ i ; $ j ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { $ max_val = PHP_INT_MIN ; for ( $ j = 0 ; $ j < $ i ; $ j ++ ) $ max_val = max ( $ max_val , $ price [ $ j ] + $ val [ $ i - $ j - 1 ] ) ; $ val [ $ i ] = $ max_val ; } return $ val [ $ n ] ; }
$ arr = array ( 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ) ; $ size = count ( $ arr ) ; echo " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " , cutRod ( $ arr , $ size ) ; ? >
< ? php function multiply ( $ x , $ y ) {
if ( $ y == 0 ) return 0 ;
if ( $ y > 0 ) return ( $ x + multiply ( $ x , $ y - 1 ) ) ;
if ( $ y < 0 ) return - multiply ( $ x , - $ y ) ; }
echo multiply ( 5 , -11 ) ; ? >
< ? php function SieveOfEratosthenes ( $ n ) {
$ prime = array_fill ( 0 , $ n + 1 , true ) ; for ( $ p = 2 ; $ p * $ p <= $ n ; $ p ++ ) {
if ( $ prime [ $ p ] == true ) {
for ( $ i = $ p * $ p ; $ i <= $ n ; $ i += $ p ) $ prime [ $ i ] = false ; } }
for ( $ p = 2 ; $ p <= $ n ; $ p ++ ) if ( $ prime [ $ p ] ) echo $ p . " " ; }
$ n = 30 ; echo " Following ▁ are ▁ the ▁ prime ▁ numbers ▁ " . " smaller ▁ than ▁ or ▁ equal ▁ to ▁ " . $ n . " STRNEWLINE " ; SieveOfEratosthenes ( $ n ) ; ? >
< ? php function binomialCoeff ( $ n , $ k ) { $ res = 1 ; if ( $ k > $ n - $ k ) $ k = $ n - $ k ; for ( $ i = 0 ; $ i < $ k ; ++ $ i ) { $ res *= ( $ n - $ i ) ; $ res /= ( $ i + 1 ) ; } return $ res ; }
function printPascal ( $ n ) {
for ( $ line = 0 ; $ line < $ n ; $ line ++ ) {
for ( $ i = 0 ; $ i <= $ line ; $ i ++ ) echo " " . binomialCoeff ( $ line , ▁ $ i ) . " " echo " STRNEWLINE " ; } }
$ n = 7 ; printPascal ( $ n ) ; ? >
< ? php function printPascal ( $ n ) {
$ arr = array ( array ( ) ) ;
for ( $ line = 0 ; $ line < $ n ; $ line ++ ) {
for ( $ i = 0 ; $ i <= $ line ; $ i ++ ) {
if ( $ line == $ i $ i == 0 ) $ arr [ $ line ] [ $ i ] = 1 ;
else $ arr [ $ line ] [ $ i ] = $ arr [ $ line - 1 ] [ $ i - 1 ] + $ arr [ $ line - 1 ] [ $ i ] ; echo $ arr [ $ line ] [ $ i ] . " " ; } echo " " } }
$ n = 5 ; printPascal ( $ n ) ; ? >
< ? php function printPascal ( $ n ) { for ( $ line = 1 ; $ line <= $ n ; $ line ++ ) {
$ C = 1 ; for ( $ i = 1 ; $ i <= $ line ; $ i ++ ) {
print ( $ C . " " ) ; $ C = $ C * ( $ line - $ i ) / $ i ; } print ( " STRNEWLINE " ) ; } }
$ n = 5 ; printPascal ( $ n ) ; ? >
< ? php function Add ( $ x , $ y ) {
while ( $ y != 0 ) {
$ carry = $ x & $ y ;
$ x = $ x ^ $ y ;
$ y = $ carry << 1 ; } return $ x ; }
echo Add ( 15 , 32 ) ; ? >
< ? php function getModulo ( $ n , $ d ) { return ( $ n & ( $ d - 1 ) ) ; }
$ n = 6 ;
$ d = 4 ; echo $ n , " ▁ moduo " , " ▁ " , $ d , " ▁ is ▁ " , " ▁ " , getModulo ( $ n , $ d ) ; ? >
< ? php function countSetBits ( $ n ) { $ count = 0 ; while ( $ n ) { $ count += $ n & 1 ; $ n >>= 1 ; } return $ count ; }
$ i = 9 ; echo countSetBits ( $ i ) ; ? >
< ? php function countSetBits ( $ n ) {
if ( $ n == 0 ) return 0 ; else return 1 + countSetBits ( $ n & ( $ n - 1 ) ) ; }
$ n = 9 ;
echo countSetBits ( $ n ) ; ? >
< ? php $ t = log10 ( 4 ) ; $ x = log ( 15 , 2 ) ; $ tt = ceil ( $ t ) ; $ xx = ceil ( $ x ) ; echo ( $ tt ) , " STRNEWLINE " ; echo ( $ xx ) , " STRNEWLINE " ; ? >
< ? php $ num_to_bits = array ( 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 ) ;
function countSetBitsRec ( $ num ) { global $ num_to_bits ; $ nibble = 0 ; if ( 0 == $ num ) return $ num_to_bits [ 0 ] ;
$ nibble = $ num & 0xf ;
return $ num_to_bits [ $ nibble ] + countSetBitsRec ( $ num >> 4 ) ; }
$ num = 31 ; echo ( countSetBitsRec ( $ num ) ) ; ? >
< ? php function getParity ( $ n ) { $ parity = 0 ; while ( $ n ) { $ parity = ! $ parity ; $ n = $ n & ( $ n - 1 ) ; } return $ parity ; }
$ n = 7 ; echo " Parity ▁ of ▁ no ▁ " , $ n , " ▁ = ▁ " , getParity ( $ n ) ? " odd " : " even " ; ? >
< ? php function Log2 ( $ x ) { return ( log10 ( $ x ) / log10 ( 2 ) ) ; }
function isPowerOfTwo ( $ n ) { return ( ceil ( Log2 ( $ n ) ) == floor ( Log2 ( $ n ) ) ) ; }
if ( isPowerOfTwo ( 31 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; if ( isPowerOfTwo ( 64 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; ? >
< ? php function isPowerOfTwo ( $ n ) { if ( $ n == 0 ) return 0 ; while ( $ n != 1 ) { if ( $ n % 2 != 0 ) return 0 ; $ n = $ n / 2 ; } return 1 ; }
if ( isPowerOfTwo ( 31 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; if ( isPowerOfTwo ( 64 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; ? >
< ? php function isPowerOfTwo ( $ x ) {
return $ x && ( ! ( $ x & ( $ x - 1 ) ) ) ; }
if ( isPowerOfTwo ( 31 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; if ( isPowerOfTwo ( 64 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; ? >
< ? php function maxRepeating ( $ arr , $ n , $ k ) {
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ arr [ $ arr [ $ i ] % $ k ] += $ k ;
$ max = $ arr [ 0 ] ; $ result = 0 ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { if ( $ arr [ $ i ] > $ max ) { $ max = $ arr [ $ i ] ; $ result = $ i ; } }
return $ result ; }
$ arr = array ( 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 ) ; $ n = sizeof ( $ arr ) ; $ k = 8 ; echo " The ▁ maximum ▁ repeating ▁ number ▁ is ▁ " , maxRepeating ( $ arr , $ n , $ k ) ; ? >
< ? php function fun ( $ x ) { $ y = ( ( int ) ( $ x / 4 ) * 4 ) ;
$ ans = 0 ; for ( $ i = $ y ; $ i <= $ x ; $ i ++ ) $ ans ^= $ i ; return $ ans ; }
function query ( $ x ) {
if ( $ x == 0 ) return 0 ; $ k = ( int ) ( ( $ x + 1 ) / 2 ) ;
return ( $ x %= 2 ) ? 2 * fun ( $ k ) : ( ( fun ( $ k - 1 ) * 2 ) ^ ( $ k & 1 ) ) ; } function allQueries ( $ q , $ l , $ r ) { for ( $ i = 0 ; $ i < $ q ; $ i ++ ) echo ( query ( $ r [ $ i ] ) ^ query ( $ l [ $ i ] - 1 ) ) , " STRNEWLINE " ; }
$ q = 3 ; $ l = array ( 2 , 2 , 5 ) ; $ r = array ( 4 , 8 , 9 ) ; allQueries ( $ q , $ l , $ r ) ; ? >
< ? php function findMinSwaps ( $ arr , $ n ) {
$ noOfZeroes [ $ n ] = array ( ) ; $ noOfZeroes = array_fill ( 0 , $ n , true ) ; $ count = 0 ;
$ noOfZeroes [ $ n - 1 ] = 1 - $ arr [ $ n - 1 ] ; for ( $ i = $ n - 2 ; $ i >= 0 ; $ i -- ) { $ noOfZeroes [ $ i ] = $ noOfZeroes [ $ i + 1 ] ; if ( $ arr [ $ i ] == 0 ) $ noOfZeroes [ $ i ] ++ ; }
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ arr [ $ i ] == 1 ) $ count += $ noOfZeroes [ $ i ] ; } return $ count ; }
$ arr = array ( 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ) ; $ n = sizeof ( $ arr ) ; echo findMinSwaps ( $ arr , $ n ) ; ? >
< ? php function printTwoOdd ( $ arr , $ size ) {
$ xor2 = $ arr [ 0 ] ;
$ set_bit_no ; $ i ; $ n = $ size - 2 ; $ x = 0 ; $ y = 0 ;
for ( $ i = 1 ; $ i < $ size ; $ i ++ ) $ xor2 = $ xor2 ^ $ arr [ $ i ] ;
$ set_bit_no = $ xor2 & ~ ( $ xor2 - 1 ) ;
for ( $ i = 0 ; $ i < $ size ; $ i ++ ) {
if ( $ arr [ $ i ] & $ set_bit_no ) $ x = $ x ^ $ arr [ $ i ] ;
else $ y = $ y ^ $ arr [ $ i ] ; } echo " The ▁ two ▁ ODD ▁ elements ▁ are ▁ " , $ x , " ▁ & ▁ " , $ y ; }
$ arr = array ( 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 ) ; $ arr_size = sizeof ( $ arr ) ; printTwoOdd ( $ arr , $ arr_size ) ; ? >
< ? php function findPair ( & $ arr , $ size , $ n ) {
$ i = 0 ; $ j = 1 ;
while ( $ i < $ size && $ j < $ size ) { if ( $ i != $ j && $ arr [ $ j ] - $ arr [ $ i ] == $ n ) { echo " Pair ▁ Found : ▁ " . " ( " . $ arr [ $ i ] . " , ▁ " . $ arr [ $ j ] . " ) " ; return true ; } else if ( $ arr [ $ j ] - $ arr [ $ i ] < $ n ) $ j ++ ; else $ i ++ ; } echo " No ▁ such ▁ pair " ; return false ; }
$ arr = array ( 1 , 8 , 30 , 40 , 100 ) ; $ size = sizeof ( $ arr ) ; $ n = 60 ; findPair ( $ arr , $ size , $ n ) ; ? >
< ? php function checkIsAP ( $ arr , $ n ) { if ( $ n == 1 ) return true ;
sort ( $ arr ) ;
$ d = $ arr [ 1 ] - $ arr [ 0 ] ; for ( $ i = 2 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] - $ arr [ $ i - 1 ] != $ d ) return false ; return true ; }
$ arr = array ( 20 , 15 , 5 , 0 , 10 ) ; $ n = count ( $ arr ) ; if ( checkIsAP ( $ arr , $ n ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function countPairs ( $ a , $ n ) {
$ mn = PHP_INT_MAX ; $ mx = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ mn = min ( $ mn , $ a [ $ i ] ) ; $ mx = max ( $ mx , $ a [ $ i ] ) ; }
$ c1 = 0 ;
$ c2 = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ a [ $ i ] == $ mn ) $ c1 ++ ; if ( $ a [ $ i ] == $ mx ) $ c2 ++ ; }
if ( $ mn == $ mx ) return $ n * ( $ n - 1 ) / 2 ; else return $ c1 * $ c2 ; }
$ a = array ( 3 , 2 , 1 , 1 , 3 ) ; $ n = count ( $ a ) ; echo countPairs ( $ a , $ n ) ; ? >
< ? php function findNumbers ( $ arr , $ n ) {
$ sumN = ( $ n * ( $ n + 1 ) ) / 2 ;
$ sumSqN = ( $ n * ( $ n + 1 ) * ( 2 * $ n + 1 ) ) / 6 ;
$ sum = 0 ; $ sumSq = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ sum += $ arr [ $ i ] ; $ sumSq += pow ( $ arr [ $ i ] , 2 ) ; } $ B = ( ( ( $ sumSq - $ sumSqN ) / ( $ sum - $ sumN ) ) + $ sumN - $ sum ) / 2 ; $ A = $ sum - $ sumN + $ B ; echo " A = " , ▁ $ A , ▁ " B = " }
$ arr = array ( 1 , 2 , 2 , 3 , 4 ) ; $ n = sizeof ( $ arr ) ; findNumbers ( $ arr , $ n ) ; ? >
< ? php function countLessThan ( & $ arr , $ n , $ key ) { $ l = 0 ; $ r = $ n - 1 ; $ index = -1 ;
while ( $ l <= $ r ) { $ m = intval ( ( $ l + $ r ) / 2 ) ; if ( $ arr [ $ m ] < $ key ) { $ l = $ m + 1 ; $ index = $ m ; } else { $ r = $ m - 1 ; } } return ( $ index + 1 ) ; }
function countGreaterThan ( & $ arr , $ n , $ key ) { $ l = 0 ; $ r = $ n - 1 ; $ index = -1 ;
while ( $ l <= $ r ) { $ m = intval ( ( $ l + $ r ) / 2 ) ; if ( $ arr [ $ m ] <= $ key ) { $ l = $ m + 1 ; } else { $ r = $ m - 1 ; $ index = $ m ; } } if ( $ index == -1 ) return 0 ; return ( $ n - $ index ) ; }
function countTriplets ( $ n , & $ a , & $ b , & $ c ) {
sort ( $ a ) ; sort ( $ b ) ; sort ( $ c ) ; $ count = 0 ;
for ( $ i = 0 ; $ i < $ n ; ++ $ i ) { $ current = $ b [ $ i ] ; $ a_index = -1 ; $ c_index = -1 ;
$ low = countLessThan ( $ a , $ n , $ current ) ;
$ high = countGreaterThan ( $ c , $ n , $ current ) ;
$ count += ( $ low * $ high ) ; } return $ count ; }
$ a = array ( 1 , 5 ) ; $ b = array ( 2 , 4 ) ; $ c = array ( 3 , 6 ) ; $ size = sizeof ( $ a ) ; echo countTriplets ( $ size , $ a , $ b , $ c ) ; ? >
< ? php function middleOfThree ( $ a , $ b , $ c ) {
$ x = $ a - $ b ;
$ y = $ b - $ c ;
$ z = $ a - $ c ;
if ( $ x * $ y > 0 ) return $ b ;
else if ( $ x * $ z > 0 ) return $ c ; else return $ a ; }
$ a = 20 ; $ b = 30 ; $ c = 40 ; echo middleOfThree ( $ a , $ b , $ c ) ; ? >
< ? php function missing4 ( $ arr , $ n ) {
$ helper = array ( 0 , 0 , 0 , 0 ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ temp = abs ( $ arr [ $ i ] ) ;
if ( $ temp <= $ n ) $ arr [ $ temp - 1 ] = $ arr [ $ temp - 1 ] * ( -1 ) ;
else if ( $ temp > $ n ) { if ( $ temp % $ n != 0 ) $ helper [ $ temp % $ n - 1 ] = -1 ; else $ helper [ ( $ temp % $ n ) + $ n - 1 ] = -1 ; } }
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] > 0 ) { $ a = $ i + 1 ; echo " $ a " , " ▁ " ; } for ( $ i = 0 ; $ i < 4 ; $ i ++ ) if ( $ helper [ $ i ] >= 0 ) { $ b = $ n + $ i + 1 ; echo " $ b " , " ▁ " ; } echo " STRNEWLINE " ; return ; }
$ arr = array ( 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 ) ; $ n = sizeof ( $ arr ) ; missing4 ( $ arr , $ n ) ; ? >
< ? php function minMovesToSort ( $ arr , $ n ) { $ moves = 0 ; $ mn = $ arr [ $ n - 1 ] ; for ( $ i = $ n - 2 ; $ i >= 0 ; $ i -- ) {
if ( $ arr [ $ i ] > $ mn ) $ moves += $ arr [ $ i ] - $ mn ;
} return $ moves ; }
$ arr = array ( 3 , 5 , 2 , 8 , 4 ) ; $ n = sizeof ( $ arr ) ; echo minMovesToSort ( $ arr , $ n ) ; ? >
< ? php function findOptimalPairs ( $ arr , $ N ) { sort ( $ arr ) ;
for ( $ i = 0 , $ j = $ N - 1 ; $ i <= $ j ; $ i ++ , $ j -- ) echo " ( " , $ arr [ $ i ] , " , ▁ " , $ arr [ $ j ] , " ) " , " ▁ " ; }
$ arr = array ( 9 , 6 , 5 , 1 ) ; $ N = sizeof ( $ arr ) ; findOptimalPairs ( $ arr , $ N ) ; ? >
< ? php function minOperations ( $ arr , $ n ) { $ result = 0 ; $ freq = array ( ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ freq [ $ arr [ $ i ] ] = 0 ; } for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ x = $ arr [ $ i ] ; $ freq [ $ x ] ++ ; }
$ maxi = max ( $ arr ) ; for ( $ i = 1 ; $ i <= $ maxi ; $ i ++ ) { if ( $ freq [ $ i ] != 0 ) {
for ( $ j = $ i * 2 ; $ j <= $ maxi ; $ j = $ j + $ i ) {
$ freq [ $ j ] = 0 ; }
$ result ++ ; } } return $ result ; }
$ arr = array ( 2 , 4 , 2 , 4 , 4 , 4 ) ; $ n = count ( $ arr ) ; echo minOperations ( $ arr , $ n ) ; ? >
< ? php function __gcd ( $ a , $ b ) { if ( $ a == 0 ) return $ b ; return __gcd ( $ b % $ a , $ a ) ; } function minGCD ( $ arr , $ n ) { $ minGCD = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ minGCD = __gcd ( $ minGCD , $ arr [ $ i ] ) ; return $ minGCD ; }
function minLCM ( $ arr , $ n ) { $ minLCM = $ arr [ 0 ] ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ minLCM = min ( $ minLCM , $ arr [ $ i ] ) ; return $ minLCM ; }
$ arr = array ( 2 , 66 , 14 , 521 ) ; $ n = sizeof ( $ arr ) ; echo " LCM ▁ = ▁ " . minLCM ( $ arr , $ n ) . " , ▁ " ; echo " GCD ▁ = ▁ " . minGCD ( $ arr , $ n ) ; ? >
< ? php function formStringMinOperations ( $ s ) {
$ count = array_fill ( 0 , 3 , 0 ) ; for ( $ i = 0 ; $ i < strlen ( $ s ) ; $ i ++ ) $ count [ $ s [ $ i ] - '0' ] ++ ;
$ processed = array_fill ( 0 , 3 , 0 ) ;
$ reqd = floor ( strlen ( $ s ) / 3 ) ; for ( $ i = 0 ; $ i < strlen ( $ s ) ; $ i ++ ) {
if ( $ count [ $ s [ $ i ] - '0' ] == $ reqd ) continue ;
if ( $ s [ $ i ] == '0' && $ count [ 0 ] > $ reqd && $ processed [ 0 ] >= $ reqd ) {
if ( $ count [ 1 ] < $ reqd ) { $ s [ $ i ] = '1' ; $ count [ 1 ] ++ ; $ count [ 0 ] -- ; }
else if ( $ count [ 2 ] < $ reqd ) { $ s [ $ i ] = '2' ; $ count [ 2 ] ++ ; $ count [ 0 ] -- ; } }
if ( $ s [ $ i ] == '1' && $ count [ 1 ] > $ reqd ) { if ( $ count [ 0 ] < $ reqd ) { $ s [ $ i ] = '0' ; $ count [ 0 ] ++ ; $ count [ 1 ] -- ; } else if ( count [ 2 ] < $ reqd && $ processed [ 1 ] >= $ reqd ) { $ s [ $ i ] = '2' ; $ count [ 2 ] ++ ; $ count [ 1 ] -- ; } }
if ( $ s [ $ i ] == '2' && $ count [ 2 ] > $ reqd ) { if ( $ count [ 0 ] < $ reqd ) { $ s [ $ i ] = '0' ; $ count [ 0 ] ++ ; $ count [ 2 ] -- ; } else if ( $ count [ 1 ] < $ reqd ) { $ s [ $ i ] = '1' ; $ count [ 1 ] ++ ; $ count [ 2 ] -- ; } }
$ processed [ $ s [ $ i ] - '0' ] ++ ; } return $ s ; }
$ s = "011200" ; echo formStringMinOperations ( $ s ) ; ? >
< ? php $ N = 3 ;
function FindMaximumSum ( $ ind , $ kon , $ a , $ b , $ c , $ n , $ dp ) { global $ N ;
if ( $ ind == $ n ) return 0 ;
if ( $ dp [ $ ind ] [ $ kon ] != -1 ) return $ dp [ $ ind ] [ $ kon ] ; $ ans = -1e9 + 5 ;
if ( $ kon == 0 ) { $ ans = max ( $ ans , $ b [ $ ind ] + FindMaximumSum ( $ ind + 1 , 1 , $ a , $ b , $ c , $ n , $ dp ) ) ; $ ans = max ( $ ans , $ c [ $ ind ] + FindMaximumSum ( $ ind + 1 , 2 , $ a , $ b , $ c , $ n , $ dp ) ) ; }
else if ( $ kon == 1 ) { $ ans = max ( $ ans , $ a [ $ ind ] + FindMaximumSum ( $ ind + 1 , 0 , $ a , $ b , $ c , $ n , $ dp ) ) ; $ ans = max ( $ ans , $ c [ $ ind ] + FindMaximumSum ( $ ind + 1 , 2 , $ a , $ b , $ c , $ n , $ dp ) ) ; }
else if ( $ kon == 2 ) { $ ans = max ( $ ans , $ a [ $ ind ] + FindMaximumSum ( $ ind + 1 , 1 , $ a , $ b , $ c , $ n , $ dp ) ) ; $ ans = max ( $ ans , $ b [ $ ind ] + FindMaximumSum ( $ ind + 1 , 0 , $ a , $ b , $ c , $ n , $ dp ) ) ; } return $ dp [ $ ind ] [ $ kon ] = $ ans ; }
$ a = array ( 6 , 8 , 2 , 7 , 4 , 2 , 7 ) ; $ b = array ( 7 , 8 , 5 , 8 , 6 , 3 , 5 ) ; $ c = array ( 8 , 3 , 2 , 6 , 8 , 4 , 1 ) ; $ n = count ( $ a ) ; $ dp = array_fill ( 0 , $ n , array_fill ( 0 , $ N , -1 ) ) ;
$ x = FindMaximumSum ( 0 , 0 , $ a , $ b , $ c , $ n , $ dp ) ;
$ y = FindMaximumSum ( 0 , 1 , $ a , $ b , $ c , $ n , $ dp ) ;
$ z = FindMaximumSum ( 0 , 2 , $ a , $ b , $ c , $ n , $ dp ) ;
print ( max ( $ x , max ( $ y , $ z ) ) ) ; ? >
< ? php $ mod = 1000000007 ;
function noOfBinaryStrings ( $ N , $ k ) { global $ mod ; $ dp = array ( 0 , 100002 , NULL ) ; for ( $ i = 1 ; $ i <= $ k - 1 ; $ i ++ ) { $ dp [ $ i ] = 1 ; } $ dp [ $ k ] = 2 ; for ( $ i = $ k + 1 ; $ i <= $ N ; $ i ++ ) { $ dp [ $ i ] = ( $ dp [ $ i - 1 ] + $ dp [ $ i - $ k ] ) % $ mod ; } return $ dp [ $ N ] ; }
$ N = 4 ; $ K = 2 ; echo noOfBinaryStrings ( $ N , $ K ) ; ? >
< ? php function findWaysToPair ( $ p ) {
$ dp = array ( ) ; $ dp [ 1 ] = 1 ; $ dp [ 2 ] = 2 ;
for ( $ i = 3 ; $ i <= $ p ; $ i ++ ) { $ dp [ $ i ] = $ dp [ $ i - 1 ] + ( $ i - 1 ) * $ dp [ $ i - 2 ] ; } return $ dp [ $ p ] ; }
$ p = 3 ; echo findWaysToPair ( $ p ) ; ? >
< ? php function CountWays ( $ n ) {
if ( $ n == 0 ) { return 1 ; } if ( $ n == 1 ) { return 1 ; } if ( $ n == 2 ) { return 1 + 1 ; }
return CountWays ( $ n - 1 ) + CountWays ( $ n - 3 ) ; }
$ n = 5 ; echo CountWays ( $ n ) ; ? >
< ? php function maxSubArraySumRepeated ( $ a , $ n , $ k ) { $ INT_MIN = 0 ; $ max_so_far = $ INT_MIN ; $ max_ending_here = 0 ; for ( $ i = 0 ; $ i < $ n * $ k ; $ i ++ ) {
$ max_ending_here = $ max_ending_here + $ a [ $ i % $ n ] ; if ( $ max_so_far < $ max_ending_here ) $ max_so_far = $ max_ending_here ; if ( $ max_ending_here < 0 ) $ max_ending_here = 0 ; } return $ max_so_far ; }
$ a = array ( 10 , 20 , -30 , -1 ) ; $ n = sizeof ( $ a ) ; $ k = 3 ; echo " Maximum ▁ contiguous ▁ sum ▁ is ▁ " , maxSubArraySumRepeated ( $ a , $ n , $ k ) ; ? >
< ? php function longOddEvenIncSeq ( & $ arr , $ n ) {
$ lioes = array_fill ( 0 , $ n , NULL ) ;
$ maxLen = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ lioes [ $ i ] = 1 ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) for ( $ j = 0 ; $ j < $ i ; $ j ++ ) if ( $ arr [ $ i ] > $ arr [ $ j ] && ( $ arr [ $ i ] + $ arr [ $ j ] ) % 2 != 0 && $ lioes [ $ i ] < $ lioes [ $ j ] + 1 ) $ lioes [ $ i ] = $ lioes [ $ j ] + 1 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ maxLen < $ lioes [ $ i ] ) $ maxLen = $ lioes [ $ i ] ;
return $ maxLen ; }
$ arr = array ( 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 ) ; $ n = sizeof ( $ arr ) ; echo " Longest ▁ Increasing ▁ Odd ▁ Even ▁ " . " Subsequence : ▁ " . longOddEvenIncSeq ( $ arr , $ n ) ; ? >
< ? php function MatrixChainOrder ( & $ p , $ i , $ j ) { if ( $ i == $ j ) return 0 ; $ min = PHP_INT_MAX ;
for ( $ k = $ i ; $ k < $ j ; $ k ++ ) { $ count = MatrixChainOrder ( $ p , $ i , $ k ) + MatrixChainOrder ( $ p , $ k + 1 , $ j ) + $ p [ $ i - 1 ] * $ p [ $ k ] * $ p [ $ j ] ; if ( $ count < $ min ) $ min = $ count ; }
return $ min ; }
$ arr = array ( 1 , 2 , 3 , 4 , 3 ) ; $ n = sizeof ( $ arr ) ; echo " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " . MatrixChainOrder ( $ arr , 1 , $ n - 1 ) ; ? >
< ? php function getCount ( $ a , $ b ) {
if ( strlen ( $ b ) % strlen ( $ a ) != 0 ) return -1 ; $ count = floor ( strlen ( $ b ) / strlen ( $ a ) ) ;
$ str = " " ; for ( $ i = 0 ; $ i < $ count ; $ i ++ ) { $ str = $ str . $ a ; } if ( strcmp ( $ a , $ b ) ) return $ count ; return -1 ; }
$ a = ' eeks ' $ b = ' eeksgeeks ' echo getCount ( $ a , $ b ) ; ? >
< ? php function countPattern ( $ str ) { $ len = strlen ( $ str ) ; $ oneSeen = 0 ;
for ( $ i = 0 ; $ i < $ len ; $ i ++ ) {
if ( $ str [ $ i ] == '1' && $ oneSeen == 1 ) if ( $ str [ $ i - 1 ] == '0' ) $ count ++ ;
if ( $ str [ $ i ] == '1' && $ oneSeen == 0 ) $ oneSeen = 1 ;
if ( $ str [ $ i ] != '0' && $ str [ $ i ] != '1' ) $ oneSeen = 0 ; } return $ count ; }
$ str = "100001abc101" ; echo countPattern ( $ str ) ; ? >
< ? php function minOperations ( $ s , $ t , $ n ) { $ ct0 = 0 ; $ ct1 = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ s [ $ i ] == $ t [ $ i ] ) continue ;
if ( $ s [ $ i ] == '0' ) $ ct0 ++ ;
else $ ct1 ++ ; } return max ( $ ct0 , $ ct1 ) ; }
$ s = "010" ; $ t = "101" ; $ n = strlen ( $ s ) ; echo minOperations ( $ s , $ t , $ n ) ; ? >
< ? php function decryptString ( $ str , $ n ) {
$ i = 0 ; $ jump = 1 ; $ decryptedStr = " " ; while ( $ i < $ n ) { $ decryptedStr . = $ str [ $ i ] ; $ i += $ jump ;
$ jump ++ ; } return $ decryptedStr ; }
$ str = " geeeeekkkksssss " ; $ n = strlen ( $ str ) ; echo decryptString ( $ str , $ n ) ; ? >
< ? php function bitToBeFlipped ( $ s ) {
$ last = $ s [ strlen ( $ s ) - 1 ] ; $ first = $ s [ 0 ] ;
if ( $ last == $ first ) { if ( $ last == '0' ) { return '1' ; } else { return '0' ; } }
else if ( $ last != $ first ) { return $ last ; } }
$ s = "1101011000" ; echo bitToBeFlipped ( $ s ) ; ? >
< ? php function findSubSequence ( $ s , $ num ) {
$ res = 0 ;
$ i = 0 ; while ( $ num ) {
if ( $ num & 1 ) $ res += $ s [ $ i ] - '0' ; $ i ++ ;
$ num = $ num >> 1 ; } return $ res ; }
function combinedSum ( string $ s ) {
$ n = strlen ( $ s ) ;
$ c_sum = 0 ;
$ range = ( 1 << $ n ) - 1 ;
for ( $ i = 0 ; $ i <= $ range ; $ i ++ ) $ c_sum += findSubSequence ( $ s , $ i ) ;
return $ c_sum ; }
$ s = "123" ; echo combinedSum ( $ s ) ; ? >
< ? php function findSubsequence ( $ str , $ k ) {
$ a = array ( 1024 ) ; for ( $ i = 0 ; $ i < 26 ; $ i ++ ) $ a [ $ i ] = 0 ;
for ( $ i = 0 ; $ i < strlen ( $ str ) ; $ i ++ ) { $ temp = ord ( $ str [ $ i ] ) - ord ( ' a ' ) ; $ a [ $ temp ] += 1 ; }
for ( $ i = 0 ; $ i < strlen ( $ str ) ; $ i ++ ) if ( $ a [ ord ( $ str [ $ i ] ) - ord ( ' a ' ) ] >= $ k ) echo $ str [ $ i ] ; }
$ k = 2 ; findSubsequence ( " geeksforgeeks " , $ k ) ; ? >
< ? php function convert ( $ str ) {
$ w = " " ; $ z = " " ;
$ str = strtoupper ( $ str ) . " ▁ " ; for ( $ i = 0 ; $ i < strlen ( $ str ) ; $ i ++ ) {
$ ch = $ str [ $ i ] ; if ( $ ch != ' ▁ ' ) $ w = $ w . $ ch ; else {
$ z = $ z . strtolower ( $ w [ 0 ] ) . substr ( $ w , 1 ) . " ▁ " ; $ w = " " ; } } return $ z ; }
$ str = " I ▁ got ▁ intern ▁ at ▁ geeksforgeeks " ; echo ( convert ( $ str ) ) ; ? >
< ? php function countOccurrences ( $ str , $ word ) {
$ a = explode ( " ▁ " , $ str ) ;
$ count = 0 ; for ( $ i = 0 ; $ i < sizeof ( $ a ) ; $ i ++ ) {
if ( $ word == $ a [ $ i ] ) $ count ++ ; } return $ count ; }
$ str = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " ; $ word = " portal " ; echo ( countOccurrences ( $ str , $ word ) ) ; ? >
< ? php function permute ( $ input ) { $ n = strlen ( $ input ) ;
$ max = 1 << $ n ;
$ input = strtolower ( $ input ) ;
for ( $ i = 0 ; $ i < $ max ; $ i ++ ) { $ combination = $ input ;
for ( $ j = 0 ; $ j < $ n ; $ j ++ ) { if ( ( ( $ i >> $ j ) & 1 ) == 1 ) $ combination [ $ j ] = chr ( ord ( $ combination [ $ j ] ) - 32 ) ; }
echo $ combination . " " ; } }
permute ( " ABC " ) ; ? >
< ? php function isPalindrome ( $ str ) {
$ l = 0 ; $ h = strlen ( $ str ) - 1 ;
while ( $ h > $ l ) if ( $ str [ $ l ++ ] != $ str [ $ h -- ] ) return false ; return true ; }
function minRemovals ( $ str ) {
if ( $ str [ 0 ] == ' ' ) return 0 ;
if ( isPalindrome ( $ str ) ) return 1 ;
return 2 ; }
echo minRemovals ( " 010010 " ) , ▁ " " echo minRemovals ( "0100101" ) , " STRNEWLINE " ; ? >
< ? php function power ( $ x , $ y , $ p ) {
$ res = 1 ;
$ x = $ x % $ p ; while ( $ y > 0 ) {
if ( $ y & 1 ) $ res = ( $ res * $ x ) % $ p ;
$ y = $ y >> 1 ; $ x = ( $ x * $ x ) % $ p ; } return $ res ; }
function findModuloByM ( $ X , $ N , $ M ) {
if ( $ N < 6 ) {
$ temp = chr ( 48 + $ X ) * $ N ;
$ res = intval ( $ temp ) % $ M ; return $ res ; }
if ( $ N % 2 == 0 ) {
$ half = findModuloByM ( $ X , ( int ) ( $ N / 2 ) , $ M ) % $ M ;
$ res = ( $ half * power ( 10 , ( int ) ( $ N / 2 ) , $ M ) + $ half ) % $ M ; return $ res ; } else {
$ half = findModuloByM ( $ X , ( int ) ( $ N / 2 ) , $ M ) % $ M ;
$ res = ( $ half * power ( 10 , ( int ) ( $ N / 2 ) + 1 , $ M ) + $ half * 10 + $ X ) % $ M ; return $ res ; } }
$ X = 6 ; $ N = 14 ; $ M = 9 ;
print ( findModuloByM ( $ X , $ N , $ M ) ) ; ? >
< ? php function lengtang ( $ r1 , $ r2 , $ d ) { echo " The ▁ length ▁ of ▁ the ▁ direct ▁ common ▁ tangent ▁ is ▁ " , sqrt ( pow ( $ d , 2 ) - pow ( ( $ r1 - $ r2 ) , 2 ) ) ; }
$ r1 = 4 ; $ r2 = 6 ; $ d = 3 ; lengtang ( $ r1 , $ r2 , $ d ) ; ? >
< ? php function rad ( $ d , $ h ) { echo " The ▁ radius ▁ of ▁ the ▁ circle ▁ is ▁ " , ( ( $ d * $ d ) / ( 8 * $ h ) + $ h / 2 ) , " STRNEWLINE " ; }
$ d = 4 ; $ h = 1 ; rad ( $ d , $ h ) ; ? >
< ? php function shortdis ( $ r , $ d ) { echo " The ▁ shortest ▁ distance ▁ " ; echo " from ▁ the ▁ chord ▁ to ▁ centre ▁ " ; echo sqrt ( ( $ r * $ r ) - ( ( $ d * $ d ) / 4 ) ) ; }
$ r = 4 ; $ d = 3 ; shortdis ( $ r , $ d ) ; ? >
< ? php function lengtang ( $ r1 , $ r2 , $ d ) { echo " The ▁ length ▁ of ▁ the ▁ direct " , " ▁ common ▁ tangent ▁ is ▁ " , sqrt ( pow ( $ d , 2 ) - pow ( ( $ r1 - $ r2 ) , 2 ) ) , " STRNEWLINE " ; }
$ r1 = 4 ; $ r2 = 6 ; $ d = 12 ; lengtang ( $ r1 , $ r2 , $ d ) ; ? >
< ? php function square ( $ a ) {
if ( $ a < 0 ) return -1 ;
$ x = 0.464 * $ a ; return $ x ; }
$ a = 5 ; echo square ( $ a ) ; ? >
< ? php function polyapothem ( $ n , $ a ) {
if ( $ a < 0 && $ n < 0 ) return -1 ;
return $ a / ( 2 * tan ( ( 180 / $ n ) * 3.14159 / 180 ) ) ; }
$ a = 9 ; $ n = 6 ; echo polyapothem ( $ n , $ a ) . " STRNEWLINE " ; ? >
< ? php function polyarea ( $ n , $ a ) {
if ( $ a < 0 && $ n < 0 ) return -1 ;
$ A = ( $ a * $ a * $ n ) / ( 4 * tan ( ( 180 / $ n ) * 3.14159 / 180 ) ) ; return $ A ; }
$ a = 9 ; $ n = 6 ; echo round ( polyarea ( $ n , $ a ) , 3 ) ; ? >
< ? php function calculateSide ( $ n , $ r ) { $ theta ; $ theta_in_radians ; $ theta = 360 / $ n ; $ theta_in_radians = $ theta * 3.14 / 180 ; return 2 * $ r * sin ( $ theta_in_radians / 2 ) ; }
$ n = 3 ;
$ r = 5 ; echo calculateSide ( $ n , $ r ) ; ? >
< ? php function cyl ( $ r , $ R , $ h ) {
if ( $ h < 0 && $ r < 0 && $ R < 0 ) return -1 ;
$ r1 = $ r ;
$ h1 = $ h ;
$ V = ( 3.14 * pow ( $ r1 , 2 ) * $ h1 ) ; return $ V ; }
$ r = 7 ; $ R = 11 ; $ h = 6 ; echo cyl ( $ r , $ R , $ h ) ;
< ? php function Perimeter ( $ s , $ n ) { $ perimeter = 1 ;
$ perimeter = $ n * $ s ; return $ perimeter ; }
$ n = 5 ;
$ s = 2.5 ;
$ peri = Perimeter ( $ s , $ n ) ; echo " Perimeter ▁ of ▁ Regular ▁ Polygon " , " ▁ with ▁ " , $ n , " ▁ sides ▁ of ▁ length ▁ " , $ s , " ▁ = ▁ " , $ peri ; ? >
< ? php function rhombusarea ( $ l , $ b ) {
if ( $ l < 0 $ b < 0 ) return -1 ;
return ( $ l * $ b ) / 2 ; }
$ l = 16 ; $ b = 6 ; echo rhombusarea ( $ l , $ b ) . " STRNEWLINE " ;
< ? php function FindPoint ( $ x1 , $ y1 , $ x2 , $ y2 , $ x , $ y ) { if ( $ x > $ x1 and $ x < $ x2 and $ y > $ y1 and $ y < $ y2 ) return true ; return false ; }
$ x1 = 0 ; $ y1 = 0 ; $ x2 = 10 ; $ y2 = 8 ;
$ x = 1 ; $ y = 5 ;
if ( FindPoint ( $ x1 , $ y1 , $ x2 , $ y2 , $ x , $ y ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function shortest_distance ( $ x1 , $ y1 , $ z1 , $ a , $ b , $ c , $ d ) { $ d = abs ( ( $ a * $ x1 + $ b * $ y1 + $ c * $ z1 + $ d ) ) ; $ e = sqrt ( $ a * $ a + $ b * $ b + $ c * $ c ) ; echo " Perpendicular ▁ distance ▁ is ▁ " . $ d / $ e ; }
$ x1 = 4 ; $ y1 = -4 ; $ z1 = 3 ; $ a = 2 ; $ b = -2 ; $ c = 5 ; $ d = 8 ;
shortest_distance ( $ x1 , $ y1 , $ z1 , $ a , $ b , $ c , $ d ) ; ? >
< ? php function findVolume ( $ l , $ b , $ h ) {
$ volume = ( $ l * $ b * $ h ) / 2 ; return $ volume ; }
$ l = 18 ; $ b = 12 ; $ h = 9 ;
echo " Volume ▁ of ▁ triangular ▁ prism : ▁ " . findVolume ( $ l , $ b , $ h ) ; ? >
< ? php function midpoint ( $ x1 , $ x2 , $ y1 , $ y2 ) { echo ( ( float ) ( $ x1 + $ x2 ) / 2 . " ▁ , ▁ " . ( float ) ( $ y1 + $ y2 ) / 2 ) ; }
$ x1 = -1 ; $ y1 = 2 ; $ x2 = 3 ; $ y2 = -6 ; midpoint ( $ x1 , $ x2 , $ y1 , $ y2 ) ; ? >
< ? php function arcLength ( $ diameter , $ angle ) { $ pi = 22.0 / 7.0 ; $ arc ; if ( $ angle >= 360 ) { echo " Angle ▁ cannot " , " ▁ be ▁ formed " ; return 0 ; } else { $ arc = ( $ pi * $ diameter ) * ( $ angle / 360.0 ) ; return $ arc ; } }
$ diameter = 25.0 ; $ angle = 45.0 ; $ arc_len = arcLength ( $ diameter , $ angle ) ; echo ( $ arc_len ) ; ? >
< ? php function checkCollision ( $ a , $ b , $ c , $ x , $ y , $ radius ) {
$ dist = ( abs ( $ a * $ x + $ b * $ y + $ c ) ) / sqrt ( $ a * $ a + $ b * $ b ) ;
if ( $ radius == $ dist ) echo " Touch " ; else if ( $ radius > $ dist ) echo " Intersect " ; else echo " Outside " ; }
$ radius = 5 ; $ x = 0 ; $ y = 0 ; $ a = 3 ; $ b = 4 ; $ c = 25 ; checkCollision ( $ a , $ b , $ c , $ x , $ y , $ radius ) ; ? >
< ? php function polygonArea ( $ X , $ Y , $ n ) {
$ area = 0.0 ;
$ j = $ n - 1 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ area += ( $ X [ $ j ] + $ X [ $ i ] ) * ( $ Y [ $ j ] - $ Y [ $ i ] ) ;
$ j = $ i ; }
return abs ( $ area / 2.0 ) ; }
$ X = array ( 0 , 2 , 4 ) ; $ Y = array ( 1 , 3 , 7 ) ; $ n = count ( $ X ) ; echo polygonArea ( $ X , $ Y , $ n ) ; ? >
< ? php function getAverage ( $ x , $ y ) {
$ avg = ( $ x & $ y ) + ( ( $ x ^ $ y ) >> 1 ) ; return $ avg ; }
$ x = 10 ; $ y = 9 ; echo getAverage ( $ x , $ y ) ; ? >
< ? php function smallestIndex ( $ a , $ n ) {
$ right1 = 0 ; $ right0 = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ a [ $ i ] == 1 ) $ right1 = $ i ;
else $ right0 = $ i ; }
return min ( $ right1 , $ right0 ) ; }
$ a = array ( 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 ) ; $ n = sizeof ( $ a ) ; echo smallestIndex ( $ a , $ n ) ; ? >
< ? php function countSquares ( $ r , $ c , $ m ) {
$ squares = 0 ;
for ( $ i = 1 ; $ i <= 8 ; $ i ++ ) { for ( $ j = 1 ; $ j <= 8 ; $ j ++ ) {
if ( max ( abs ( $ i - $ r ) , abs ( $ j - $ c ) ) <= $ m ) $ squares ++ ; } }
return $ squares ; }
$ r = 4 ; $ c = 4 ; $ m = 1 ; echo countSquares ( $ r , $ c , $ m ) ; ? >
< ? php function countNumbers ( $ L , $ R , $ K ) { if ( $ K == 9 ) $ K = 0 ;
$ totalnumbers = $ R - $ L + 1 ;
$ factor9 = intval ( $ totalnumbers / 9 ) ;
$ rem = $ totalnumbers % 9 ;
$ ans = $ factor9 ;
for ( $ i = $ R ; $ i > $ R - $ rem ; $ i -- ) { $ rem1 = $ i % 9 ; if ( $ rem1 == $ K ) $ ans ++ ; } return $ ans ; }
$ L = 10 ; $ R = 22 ; $ K = 3 ; echo countNumbers ( $ L , $ R , $ K ) ; ? >
< ? php function BalanceArray ( $ A , & $ Q ) { $ ANS = array ( ) ; $ sum = 0 ; for ( $ i = 0 ; $ i < count ( $ A ) ; $ i ++ )
if ( $ A [ $ i ] % 2 == 0 ) $ sum = $ sum + $ A [ $ i ] ; for ( $ i = 0 ; $ i < count ( $ Q ) ; $ i ++ ) { $ index = $ Q [ $ i ] [ 0 ] ; $ value = $ Q [ $ i ] [ 1 ] ;
if ( $ A [ $ index ] % 2 == 0 ) $ sum = $ sum - $ A [ $ index ] ; $ A [ $ index ] = $ A [ $ index ] + $ value ;
if ( $ A [ $ index ] % 2 == 0 ) $ sum = $ sum + $ A [ $ index ] ;
array_push ( $ ANS , $ sum ) ; }
for ( $ i = 0 ; $ i < count ( $ ANS ) ; $ i ++ ) echo $ ANS [ $ i ] . " ▁ " ; }
$ A = array ( 1 , 2 , 3 , 4 ) ; $ Q = array ( array ( 0 , 1 ) , array ( 1 , -3 ) , array ( 0 , -4 ) , array ( 3 , 2 ) ) ; BalanceArray ( $ A , $ Q ) ; ? >
< ? php function Cycles ( $ N ) { $ fact = 1 ; $ result = 0 ; $ result = $ N - 1 ;
$ i = $ result ; while ( $ i > 0 ) { $ fact = $ fact * $ i ; $ i -- ; } return floor ( $ fact / 2 ) ; }
$ N = 5 ; $ Number = Cycles ( $ N ) ; echo " Hamiltonian ▁ cycles ▁ = ▁ " , $ Number ; ? >
< ? php function digitWell ( $ n , $ m , $ k ) { $ cnt = 0 ; while ( $ n > 0 ) { if ( $ n % 10 == $ m ) ++ $ cnt ; $ n = floor ( $ n / 10 ) ; } return $ cnt == $ k ; }
function findInt ( $ n , $ m , $ k ) { $ i = $ n + 1 ; while ( true ) { if ( digitWell ( $ i , $ m , $ k ) ) return $ i ; $ i ++ ; } }
$ n = 111 ; $ m = 2 ; $ k = 2 ; echo findInt ( $ n , $ m , $ k ) ; ? >
< ? php function countOdd ( $ arr , $ n ) {
$ odd = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ arr [ $ i ] % 2 == 1 ) $ odd ++ ; } return $ odd ; }
function countValidPairs ( $ arr , $ n ) { $ odd = countOdd ( $ arr , $ n ) ; return ( $ odd * ( $ odd - 1 ) ) / 2 ; }
$ arr = array ( 1 , 2 , 3 , 4 , 5 ) ; $ n = sizeof ( $ arr ) ; echo countValidPairs ( $ arr , $ n ) ; ? >
< ? php function gcd ( $ a , $ b ) { if ( $ b == 0 ) return $ a ; else return gcd ( $ b , $ a % $ b ) ; }
function lcmOfArray ( & $ arr , $ n ) { if ( $ n < 1 ) return 0 ; $ lcm = $ arr [ 0 ] ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ lcm = ( $ lcm * $ arr [ $ i ] ) / gcd ( $ lcm , $ arr [ $ i ] ) ;
return $ lcm ; }
function minPerfectCube ( & $ arr , $ n ) {
$ lcm = lcmOfArray ( $ arr , $ n ) ; $ minPerfectCube = $ lcm ; $ cnt = 0 ; while ( $ lcm > 1 && $ lcm % 2 == 0 ) { $ cnt ++ ; $ lcm /= 2 ; }
if ( $ cnt % 3 == 2 ) $ minPerfectCube *= 2 ; else if ( $ cnt % 3 == 1 ) $ minPerfectCube *= 4 ; $ i = 3 ;
while ( $ lcm > 1 ) { $ cnt = 0 ; while ( $ lcm % $ i == 0 ) { $ cnt ++ ; $ lcm /= $ i ; } if ( $ cnt % 3 == 1 ) $ minPerfectCube *= $ i * $ i ; else if ( $ cnt % 3 == 2 ) $ minPerfectCube *= $ i ; $ i += 2 ; }
return $ minPerfectCube ; }
$ arr = array ( 10 , 125 , 14 , 42 , 100 ) ; $ n = sizeof ( $ arr ) ; echo ( minPerfectCube ( $ arr , $ n ) ) ; ? >
< ? php function isPrime ( $ n ) {
if ( $ n <= 1 ) return false ; if ( $ n <= 3 ) return true ;
if ( $ n % 2 == 0 $ n % 3 == 0 ) return false ; for ( $ i = 5 ; $ i * $ i <= $ n ; $ i = $ i + 6 ) if ( $ n % $ i == 0 || $ n % ( $ i + 2 ) == 0 ) return false ; return true ; }
function isStrongPrime ( $ n ) {
if ( ! isPrime ( $ n ) $ n == 2 ) return false ;
$ previous_prime = $ n - 1 ; $ next_prime = $ n + 1 ;
while ( ! isPrime ( $ next_prime ) ) $ next_prime ++ ;
while ( ! isPrime ( $ previous_prime ) ) $ previous_prime -- ;
$ mean = ( $ previous_prime + $ next_prime ) / 2 ;
if ( $ n > $ mean ) return true ; else return false ; }
$ n = 11 ; if ( isStrongPrime ( $ n ) ) echo ( " Yes " ) ; else echo ( " No " ) ; ? >
< ? php function countDigitsToBeRemoved ( $ N , $ K ) {
$ s = strval ( $ N ) ;
$ res = 0 ;
$ f_zero = 0 ; for ( $ i = strlen ( $ s ) - 1 ; $ i >= 0 ; $ i -- ) { if ( $ K == 0 ) return $ res ; if ( $ s [ $ i ] == '0' ) {
$ f_zero = 1 ; $ K -- ; } else $ res ++ ; }
if ( ! $ K ) return $ res ; else if ( $ f_zero ) return strlen ( $ s ) - 1 ; return -1 ; }
$ N = 10904025 ; $ K = 2 ; echo countDigitsToBeRemoved ( $ N , $ K ) . " " ; $ N = 1000 ; $ K = 5 ; echo countDigitsToBeRemoved ( $ N , $ K ) . " " ; $ N = 23985 ; $ K = 2 ; echo countDigitsToBeRemoved ( $ N , $ K ) ; ? >
< ? php function getSum ( $ a , $ n ) {
$ sum = 0 ; for ( $ i = 1 ; $ i <= $ n ; ++ $ i ) {
$ sum += ( $ i / pow ( $ a , $ i ) ) ; } return $ sum ; }
$ a = 3 ; $ n = 3 ;
echo ( getSum ( $ a , $ n ) ) ; ? >
< ? php function largestPrimeFactor ( $ n ) {
$ max = -1 ;
while ( $ n % 2 == 0 ) { $ max = 2 ;
}
for ( $ i = 3 ; $ i <= sqrt ( $ n ) ; $ i += 2 ) { while ( $ n % $ i == 0 ) { $ max = $ i ; $ n = $ n / $ i ; } }
if ( $ n > 2 ) $ max = $ n ; return $ max ; }
function checkUnusual ( $ n ) {
$ factor = largestPrimeFactor ( $ n ) ;
if ( $ factor > sqrt ( $ n ) ) { return true ; } else { return false ; } }
$ n = 14 ; if ( checkUnusual ( $ n ) ) { echo " YES " . " STRNEWLINE " ; } else { echo " NO " . " STRNEWLINE " ; } ? >
< ? php function isHalfReducible ( $ arr , $ n , $ m ) { $ frequencyHash = array_fill ( 0 , $ m + 1 , 0 ) ; $ i = 0 ; for ( ; $ i < $ n ; $ i ++ ) { $ frequencyHash [ ( $ arr [ $ i ] % ( $ m + 1 ) ) ] ++ ; } for ( $ i = 0 ; $ i <= $ m ; $ i ++ ) { if ( $ frequencyHash [ $ i ] >= ( $ n / 2 ) ) break ; } if ( $ i <= $ m ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; }
$ arr = array ( 8 , 16 , 32 , 3 , 12 ) ; $ n = sizeof ( $ arr ) ; $ m = 7 ; isHalfReducible ( $ arr , $ n , $ m ) ; ? >
< ? php function isPrime ( $ n ) {
if ( $ n <= 1 ) return false ; if ( $ n <= 3 ) return true ;
if ( $ n % 2 == 0 or $ n % 3 == 0 ) return false ; for ( $ i = 5 ; $ i * $ i <= $ n ; $ i = $ i + 6 ) { if ( $ n % $ i == 0 or $ n % ( $ i + 2 ) == 0 ) { return false ; } } return true ; }
function isPowerOfTwo ( $ n ) { return ( $ n && ! ( $ n & ( $ n - 1 ) ) ) ; }
$ n = 43 ;
if ( isPrime ( $ n ) && ( isPowerOfTwo ( $ n * 3 - 1 ) ) ) { echo " YES " ; } else { echo " NO " ; } ? >
< ? php function area ( $ a ) {
if ( $ a < 0 ) return -1 ;
$ area = pow ( ( $ a * sqrt ( 3 ) ) / ( sqrt ( 2 ) ) , 2 ) ; return $ area ; }
$ a = 5 ; echo area ( $ a ) . " STRNEWLINE " ; ? >
< ? php function nthTerm ( $ n ) { return 3 * pow ( $ n , 2 ) - 4 * $ n + 2 ; }
$ N = 4 ; echo nthTerm ( $ N ) . " STRNEWLINE " ; ? >
< ? php function calculateSum ( $ n ) { return $ n * ( $ n + 1 ) / 2 + pow ( ( $ n * ( $ n + 1 ) / 2 ) , 2 ) ; }
$ n = 3 ;
echo " Sum = " ? >
< ? php function arePermutations ( $ a , $ b , $ n , $ m ) { $ sum1 = 0 ; $ sum2 = 0 ; $ mul1 = 1 ; $ mul2 = 1 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ sum1 += $ a [ $ i ] ; $ mul1 *= $ a [ $ i ] ; }
for ( $ i = 0 ; $ i < $ m ; $ i ++ ) { $ sum2 += $ b [ $ i ] ; $ mul2 *= $ b [ $ i ] ; }
return ( ( $ sum1 == $ sum2 ) && ( $ mul1 == $ mul2 ) ) ; }
$ a = array ( 1 , 3 , 2 ) ; $ b = array ( 3 , 1 , 2 ) ; $ n = sizeof ( $ a ) ; $ m = sizeof ( $ b ) ; if ( arePermutations ( $ a , $ b , $ n , $ m ) ) echo " Yes " . " STRNEWLINE " ; else echo " No " . " STRNEWLINE " ;
< ? php function Race ( $ B , $ C ) { $ result = 0 ;
$ result = ( ( $ C * 100 ) / $ B ) ; return 100 - $ result ; }
$ B = 10 ; $ C = 28 ;
$ B = 100 - $ B ; $ C = 100 - $ C ; echo Race ( $ B , $ C ) . " ▁ meters " ; ? >
< ? php function T_ime ( $ arr , $ n , $ Emptypipe ) { $ fill = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ fill += 1 / $ arr [ $ i ] ; $ fill = $ fill - ( 1 / $ Emptypipe ) ; return 1 / $ fill ; }
$ arr = array ( 12 , 14 ) ; $ Emptypipe = 30 ; $ n = count ( $ arr ) ; echo ( int ) T_ime ( $ arr , $ n , $ Emptypipe ) . " ▁ Hours " ; ? >
< ? php function check ( $ n ) { $ sum = 0 ;
while ( $ n != 0 ) { $ sum += $ n % 10 ; $ n = ( int ) ( $ n / 10 ) ; }
if ( $ sum % 7 == 0 ) return 1 ; else return 0 ; }
$ n = 25 ; ( check ( $ n ) == 1 ) ? print ( " YES STRNEWLINE " ) : print ( " NO STRNEWLINE " ) ; ? >
< ? php $ N = 1000005 ;
function isPrime ( $ n ) { global $ N ;
if ( $ n <= 1 ) return false ; if ( $ n <= 3 ) return true ;
if ( $ n % 2 == 0 $ n % 3 == 0 ) return false ; for ( $ i = 5 ; $ i * $ i <= $ n ; $ i = $ i + 6 ) if ( $ n % $ i == 0 || $ n % ( $ i + 2 ) == 0 ) return false ; return true ; }
function SumOfPrimeDivisors ( $ n ) { $ sum = 0 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { if ( $ n % $ i == 0 ) { if ( isPrime ( $ i ) ) $ sum += $ i ; } } return $ sum ; }
$ n = 60 ; echo " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " . SumOfPrimeDivisors ( $ n ) ; ? >
< ? php function Sum ( $ N ) { for ( $ i = 0 ; $ i <= $ N ; $ i ++ ) $ SumOfPrimeDivisors [ $ i ] = 0 ; for ( $ i = 2 ; $ i <= $ N ; ++ $ i ) {
if ( ! $ SumOfPrimeDivisors [ $ i ] ) {
for ( $ j = $ i ; $ j <= $ N ; $ j += $ i ) { $ SumOfPrimeDivisors [ $ j ] += $ i ; } } } return $ SumOfPrimeDivisors [ $ N ] ; }
$ N = 60 ; echo " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " . Sum ( $ N ) ; ? >
< ? php function power ( $ x , $ y , $ p ) {
$ x = $ x % $ p ; while ( $ y > 0 ) {
if ( $ y & 1 ) $ res = ( $ res * $ x ) % $ p ;
$ x = ( $ x * $ x ) % $ p ; } return $ res ; }
$ a = 3 ;
$ b = "100000000000000000000000000" ; $ remainderB = 0 ; $ MOD = 1000000007 ;
for ( $ i = 0 ; $ i < strlen ( $ b ) ; $ i ++ ) $ remainderB = ( $ remainderB * 10 + $ b [ $ i ] - '0' ) % ( $ MOD - 1 ) ; echo power ( $ a , $ remainderB , $ MOD ) ; ? >
< ? php function find_Square_369 ( $ num ) {
if ( $ num [ 0 ] == '3' ) { $ a = '1' ; $ b = '0' ; $ c = '8' ; $ d = '9' ; }
else if ( $ num [ 0 ] == ' 6 ' ) { $ a = '4' ; $ b = '3' ; $ c = '5' ; $ d = '6' ; }
else { $ a = '9' ; $ b = '8' ; $ c = '0' ; $ d = '1' ; }
$ result = " " ;
$ size = strlen ( $ num ) ;
for ( $ i = 1 ; $ i < $ size ; $ i ++ ) $ result = $ result . $ a ;
$ result = $ result . $ b ;
for ( $ i = 1 ; $ i < $ size ; $ i ++ ) $ result = $ result . $ c ;
$ result = $ result . $ d ;
return $ result ; }
$ num_3 = "3333" ; $ num_6 = "6666" ; $ num_9 = "9999" ; $ result = " " ;
$ result = find_Square_369 ( $ num_3 ) ; echo " Square ▁ of ▁ " . $ num_3 . " ▁ is ▁ : ▁ " . $ result . " STRNEWLINE " ;
$ result = find_Square_369 ( $ num_6 ) ; echo " Square ▁ of ▁ " . $ num_6 . " ▁ is ▁ : ▁ " . $ result . " STRNEWLINE " ;
$ result = find_Square_369 ( $ num_9 ) ; echo " Square ▁ of ▁ " . $ num_9 . " ▁ is ▁ : ▁ " . $ result . " STRNEWLINE " ; return 0 ; ? >
< ? php < ? php $ ans = 1 ; $ mod = 1000000007 * 120 ; for ( $ i = 0 ; $ i < 5 ; $ i ++ ) $ ans = ( $ ans * ( 55555 - $ i ) ) % $ mod ; $ ans = $ ans / 120 ; echo " Answer ▁ using ▁ shortcut : ▁ " , $ ans ; ? >
< ? php function fact ( $ n ) { if ( $ n == 0 $ n == 1 ) return 1 ; $ ans = 1 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ ans = $ ans * $ i ; return $ ans ; }
function nCr ( $ n , $ r ) { $ Nr = $ n ; $ Dr = 1 ; $ ans = 1 ; for ( $ i = 1 ; $ i <= $ r ; $ i ++ ) { $ ans = ( $ ans * $ Nr ) / ( $ Dr ) ; $ Nr -- ; $ Dr ++ ; } return $ ans ; }
function solve ( $ n ) { $ N = 2 * $ n - 2 ; $ R = $ n - 1 ; return nCr ( $ N , $ R ) * fact ( $ n - 1 ) ; }
$ n = 6 ; echo solve ( $ n ) ; ? >
< ? php function pythagoreanTriplet ( $ n ) {
for ( $ i = 1 ; $ i <= $ n / 3 ; $ i ++ ) {
for ( $ j = $ i + 1 ; $ j <= $ n / 2 ; $ j ++ ) { $ k = $ n - $ i - $ j ; if ( $ i * $ i + $ j * $ j == $ k * $ k ) { echo $ i , " , ▁ " , $ j , " , ▁ " , $ k ; return ; } } } echo " No ▁ Triplet " ; }
$ n = 12 ; pythagoreanTriplet ( $ n ) ; ? >
< ? php function factorial ( $ n ) { $ f = 1 ; for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) $ f *= $ i ; return $ f ; }
function series ( $ A , $ X , $ n ) {
$ nFact = factorial ( $ n ) ;
for ( $ i = 0 ; $ i < $ n + 1 ; $ i ++ ) {
$ niFact = factorial ( $ n - $ i ) ; $ iFact = factorial ( $ i ) ;
$ aPow = pow ( $ A , $ n - $ i ) ; $ xPow = pow ( $ X , $ i ) ;
echo ( $ nFact * $ aPow * $ xPow ) / ( $ niFact * $ iFact ) , " " ; } }
$ A = 3 ; $ X = 4 ; $ n = 5 ; series ( $ A , $ X , $ n ) ; ? >
< ? php function seiresSum ( $ n , $ a ) { $ res = 0 ; for ( $ i = 0 ; $ i < 2 * $ n ; $ i ++ ) { if ( $ i % 2 == 0 ) $ res += $ a [ $ i ] * $ a [ $ i ] ; else $ res -= $ a [ $ i ] * $ a [ $ i ] ; } return $ res ; }
$ n = 2 ; $ a = array ( 1 , 2 , 3 , 4 ) ; echo seiresSum ( $ n , $ a ) ; ? >
< ? php function power ( $ n , $ r ) {
$ count = 0 ; for ( $ i = $ r ; ( $ n / $ i ) >= 1 ; $ i = $ i * $ r ) $ count += $ n / $ i ; return $ count ; }
$ n = 6 ; $ r = 3 ; echo power ( $ n , $ r ) ; ? >
< ? php function avg_of_odd_num ( $ n ) {
$ sum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ sum += ( 2 * $ i + 1 ) ;
return $ sum / $ n ; }
$ n = 20 ; echo ( avg_of_odd_num ( $ n ) ) ; ? >
< ? php function avg_of_odd_num ( $ n ) { return $ n ; }
$ n = 8 ; echo ( avg_of_odd_num ( $ n ) ) ; ? >
< ? php function fib ( & $ f , $ N ) {
$ f [ 1 ] = 1 ; $ f [ 2 ] = 1 ; for ( $ i = 3 ; $ i <= $ N ; $ i ++ )
$ f [ $ i ] = $ f [ $ i - 1 ] + $ f [ $ i - 2 ] ; } function fiboTriangle ( $ n ) {
$ N = $ n * ( $ n + 1 ) / 2 ; $ f = array ( ) ; fib ( $ f , $ N ) ;
$ fiboNum = 1 ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) {
for ( $ j = 1 ; $ j <= $ i ; $ j ++ ) echo ( $ f [ $ fiboNum ++ ] . " ▁ " ) ; echo ( " STRNEWLINE " ) ; } }
$ n = 5 ; fiboTriangle ( $ n ) ; ? >
< ? php function averageOdd ( $ n ) { if ( $ n % 2 == 0 ) { echo ( " Invalid ▁ Input " ) ; return -1 ; } $ sum = 0 ; $ count = 0 ; while ( $ n >= 1 ) {
$ count ++ ;
$ sum += $ n ; $ n = $ n - 2 ; } return $ sum / $ count ; }
$ n = 15 ; echo ( averageOdd ( $ n ) ) ; ? >
< ? php function averageOdd ( $ n ) { if ( $ n % 2 == 0 ) { echo ( " Invalid ▁ Input " ) ; return -1 ; } return ( $ n + 1 ) / 2 ; }
$ n = 15 ; echo ( averageOdd ( $ n ) ) ; ? >
< ? php function TrinomialValue ( $ n , $ k ) {
if ( $ n == 0 && $ k == 0 ) return 1 ;
if ( $ k < - $ n $ k > $ n ) return 0 ;
return TrinomialValue ( $ n - 1 , $ k - 1 ) + TrinomialValue ( $ n - 1 , $ k ) + TrinomialValue ( $ n - 1 , $ k + 1 ) ; }
function printTrinomial ( $ n ) {
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
for ( $ j = - $ i ; $ j <= 0 ; $ j ++ ) echo TrinomialValue ( $ i , $ j ) , " ▁ " ;
for ( $ j = 1 ; $ j <= $ i ; $ j ++ ) echo TrinomialValue ( $ i , $ j ) , " ▁ " ; echo " STRNEWLINE " ; } }
$ n = 4 ; printTrinomial ( $ n ) ; ? >
< ? php $ MAX = 10 ;
function TrinomialValue ( $ dp , $ n , $ k ) {
if ( $ k < 0 ) $ k = - $ k ;
if ( $ dp [ $ n ] [ $ k ] != 0 ) return $ dp [ $ n ] [ $ k ] ;
if ( $ n == 0 && $ k == 0 ) return 1 ;
if ( $ k < - $ n $ k > $ n ) return 0 ;
return ( $ dp [ $ n ] [ $ k ] = TrinomialValue ( $ dp , $ n - 1 , $ k - 1 ) + TrinomialValue ( $ dp , $ n - 1 , $ k ) + TrinomialValue ( $ dp , $ n - 1 , $ k + 1 ) ) ; }
function printTrinomial ( $ n ) { global $ MAX ; $ dp ; for ( $ i = 0 ; $ i < $ MAX ; $ i ++ ) for ( $ j = 0 ; $ j < $ MAX ; $ j ++ ) $ dp [ $ i ] [ $ j ] = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
for ( $ j = - $ i ; $ j <= 0 ; $ j ++ ) echo TrinomialValue ( $ dp , $ i , $ j ) . " ▁ " ;
for ( $ j = 1 ; $ j <= $ i ; $ j ++ ) echo TrinomialValue ( $ dp , $ i , $ j ) . " ▁ " ; echo " STRNEWLINE " ; } }
$ n = 4 ; printTrinomial ( $ n ) ; ? >
< ? php function sumOfLargePrimeFactor ( $ n ) {
$ prime = array_fill ( 0 , $ n + 1 , 0 ) ; $ sum = 0 ; $ max = ( int ) ( $ n / 2 ) ; for ( $ p = 2 ; $ p <= $ max ; $ p ++ ) {
if ( $ prime [ $ p ] == 0 ) {
for ( $ i = $ p * 2 ; $ i <= $ n ; $ i += $ p ) $ prime [ $ i ] = $ p ; } }
for ( $ p = 2 ; $ p <= $ n ; $ p ++ ) {
if ( $ prime [ $ p ] ) $ sum += $ prime [ $ p ] ;
else $ sum += $ p ; }
return $ sum ; }
$ n = 12 ; echo " Sum ▁ = ▁ " . sumOfLargePrimeFactor ( $ n ) ; ? >
< ? php function calculate_sum ( $ a , $ N ) {
$ m = $ N / $ a ;
$ sum = $ m * ( $ m + 1 ) / 2 ;
$ ans = $ a * $ sum ; return $ ans ; }
$ a = 7 ; $ N = 49 ; echo " Sum ▁ of ▁ multiples ▁ of ▁ " . $ a , " ▁ up ▁ to ▁ " . $ N . " ▁ = ▁ " . calculate_sum ( $ a , $ N ) ; ? >
< ? php function isPowerOf2 ( $ str ) { $ len_str = strlen ( $ str ) ;
$ num = 0 ;
if ( $ len_str == 1 && $ str [ $ len_str - 1 ] == '1' ) return 0 ;
while ( $ len_str != 1 $ str [ $ len_str - 1 ] != '1' ) {
if ( ord ( $ str [ $ len_str - 1 ] - '0' ) % 2 == 1 ) return 0 ;
$ j = 0 ; for ( $ i = 0 ; $ i < $ len_str ; $ i ++ ) { $ num = $ num * 10 + ( ord ( $ str [ $ i ] ) - ord ( '0' ) ) ;
if ( $ num < 2 ) {
if ( $ i != 0 ) $ str [ $ j ++ ] = '0' ;
continue ; } $ str [ $ j ++ ] = chr ( ( int ) ( $ num / 2 ) + ord ( '0' ) ) ; $ num = ( $ num ) - ( int ) ( $ num / 2 ) * 2 ; }
$ len_str = $ j ; }
return 1 ; }
$ str1 = "124684622466842024680246842024662202000002" ; $ str2 = "1" ; $ str3 = "128" ; print ( isPowerOf2 ( $ str1 ) . " " . isPowerOf2 ( $ str2 ) . " " ? > ? >
< ? php function ispowerof2 ( $ num ) { if ( ( $ num & ( $ num - 1 ) ) == 0 ) return 1 ; return 0 ; }
$ num = 549755813888 ; echo ispowerof2 ( $ num ) ; ? >
< ? php function counDivisors ( $ X ) {
$ count = 0 ;
for ( $ i = 1 ; $ i <= $ X ; ++ $ i ) { if ( $ X % $ i == 0 ) { $ count ++ ; } }
return $ count ; }
function countDivisorsMult ( $ arr , $ n ) {
$ mul = 1 ; for ( $ i = 0 ; $ i < $ n ; ++ $ i ) $ mul *= $ arr [ $ i ] ;
return counDivisors ( $ mul ) ; }
$ arr = array ( 2 , 4 , 6 ) ; $ n = sizeof ( $ arr ) ; echo countDivisorsMult ( $ arr , $ n ) ; ? >
< ? php function freqPairs ( $ arr , $ n ) {
$ max = max ( $ arr ) ;
$ freq = array_fill ( 0 , $ max + 1 , 0 ) ;
$ count = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ freq [ $ arr [ $ i ] ] ++ ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = 2 * $ arr [ $ i ] ; $ j <= $ max ; $ j += $ arr [ $ i ] ) {
if ( $ freq [ $ j ] >= 1 ) $ count += $ freq [ $ j ] ; }
if ( $ freq [ $ arr [ $ i ] ] > 1 ) { $ count += $ freq [ $ arr [ $ i ] ] - 1 ; $ freq [ $ arr [ $ i ] ] -- ; } } return $ count ; }
$ arr = array ( 3 , 2 , 4 , 2 , 6 ) ; $ n = count ( $ arr ) ; echo freqPairs ( $ arr , $ n ) ; ? >
< ? php function Nth_Term ( $ n ) { return ( 2 * pow ( $ n , 3 ) - 3 * pow ( $ n , 2 ) + $ n + 6 ) / 6 ; }
$ N = 8 ; echo Nth_Term ( $ N ) ; ? >
< ? php function printNthElement ( $ n ) {
$ arr = array_fill ( 0 , ( $ n + 1 ) , NULL ) ; $ arr [ 1 ] = 3 ; $ arr [ 2 ] = 5 ; for ( $ i = 3 ; $ i <= $ n ; $ i ++ ) {
if ( $ i % 2 != 0 ) $ arr [ $ i ] = $ arr [ $ i / 2 ] * 10 + 3 ; else $ arr [ $ i ] = $ arr [ ( $ i / 2 ) - 1 ] * 10 + 5 ; } return $ arr [ $ n ] ; }
$ n = 6 ; echo printNthElement ( $ n ) ; ? >
< ? php function nthTerm ( $ N ) {
return ( $ N * ( ( int ) ( $ N / 2 ) + ( ( $ N % 2 ) * 2 ) + $ N ) ) ; }
$ N = 5 ;
echo " Nth ▁ term ▁ for ▁ N ▁ = ▁ " , $ N , " ▁ : ▁ " , nthTerm ( $ N ) ; ? >
< ? php function series ( $ A , $ X , $ n ) {
$ term = pow ( $ A , $ n ) ; echo $ term , " " ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) {
$ term = $ term * $ X * ( $ n - $ i + 1 ) / ( $ i * $ A ) ; echo $ term , " " ; } }
$ A = 3 ; $ X = 4 ; $ n = 5 ; series ( $ A , $ X , $ n ) ; ? >
< ? php function Div_by_8 ( $ n ) { return ( ( ( $ n >> 3 ) << 3 ) == $ n ) ; }
$ n = 16 ; if ( Div_by_8 ( $ n ) ) echo " YES " ; else echo " NO " ; ? >
< ? php function averageEven ( $ n ) { if ( $ n % 2 != 0 ) { echo ( " Invalid ▁ Input " ) ; return -1 ; } $ sum = 0 ; $ count = 0 ; while ( $ n >= 2 ) {
$ count ++ ;
$ sum += $ n ; $ n = $ n - 2 ; } return $ sum / $ count ; }
$ n = 16 ; echo ( averageEven ( $ n ) ) ; ? >
< ? php function averageEven ( $ n ) { if ( $ n % 2 != 0 ) { echo ( " Invalid ▁ Input " ) ; return -1 ; } return ( $ n + 2 ) / 2 ; }
$ n = 16 ; echo ( averageEven ( $ n ) ) ; return 0 ; ? >
< ? php function gcd ( $ a , $ b ) {
if ( $ a == 0 $ b == 0 ) return 0 ;
if ( $ a == $ b ) return $ a ;
if ( $ a > $ b ) return gcd ( $ a - $ b , $ b ) ; return gcd ( $ a , $ b - $ a ) ; }
function cpFact ( $ x , $ y ) { while ( gcd ( $ x , $ y ) != 1 ) { $ x = $ x / gcd ( $ x , $ y ) ; } return $ x ; }
$ x = 15 ; $ y = 3 ; echo cpFact ( $ x , $ y ) , " STRNEWLINE " ; $ x = 14 ; $ y = 28 ; echo cpFact ( $ x , $ y ) , " STRNEWLINE " ; $ x = 7 ; $ y = 3 ; echo cpFact ( $ x , $ y ) ; ? >
< ? php function counLastDigitK ( $ low , $ high , $ k ) { $ count = 0 ; for ( $ i = $ low ; $ i <= $ high ; $ i ++ ) if ( $ i % 10 == $ k ) $ count ++ ; return $ count ; }
$ low = 3 ; $ high = 35 ; $ k = 3 ; echo counLastDigitK ( $ low , $ high , $ k ) ; ? >
< ? php function printTaxicab2 ( $ N ) {
$ i = 1 ; $ count = 0 ; while ( $ count < $ N ) { $ int_count = 0 ;
for ( $ j = 1 ; $ j <= pow ( $ i , 1.0 / 3 ) ; $ j ++ ) for ( $ k = $ j + 1 ; $ k <= pow ( $ i , 1.0 / 3 ) ; $ k ++ ) if ( $ j * $ j * $ j + $ k * $ k * $ k == $ i ) $ int_count ++ ;
if ( $ int_count == 2 ) { $ count ++ ; echo $ count , " " , ▁ $ i , ▁ " " } $ i ++ ; } }
$ N = 5 ; printTaxicab2 ( $ N ) ; ? >
< ? php function isComposite ( $ n ) {
if ( $ n <= 1 ) return false ; if ( $ n <= 3 ) return false ;
if ( $ n % 2 == 0 $ n % 3 == 0 ) return true ; for ( $ i = 5 ; $ i * $ i <= $ n ; $ i = $ i + 6 ) if ( $ n % $ i == 0 || $ n % ( $ i + 2 ) == 0 ) return true ; return false ; }
if ( isComposite ( 11 ) ) echo " true " ; else echo " false " ; echo " STRNEWLINE " ; if ( isComposite ( 15 ) ) echo " true " ; else echo " false " ; echo " STRNEWLINE " ; ? >
< ? php function isPrime ( $ n ) {
if ( $ n <= 1 ) return false ;
for ( $ i = 2 ; $ i < $ n ; $ i ++ ) if ( $ n % $ i == 0 ) return false ; return true ; }
function findPrime ( $ n ) { $ num = $ n + 1 ;
while ( $ num ) {
if ( isPrime ( $ num ) ) return $ num ;
$ num = $ num + 1 ; } return 0 ; }
function minNumber ( $ arr , $ n ) { $ sum = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ sum += $ arr [ $ i ] ;
if ( isPrime ( $ sum ) ) return 0 ;
$ num = findPrime ( $ sum ) ;
return $ num - $ sum ; }
$ arr = array ( 2 , 4 , 6 , 8 , 12 ) ; $ n = sizeof ( $ arr ) ; echo minNumber ( $ arr , $ n ) ; ? >
< ? php function fact ( $ n ) { if ( $ n == 0 ) return 1 ; return $ n * fact ( $ n - 1 ) ; }
function div ( $ x ) { $ ans = 0 ; for ( $ i = 1 ; $ i <= $ x ; $ i ++ ) if ( $ x % $ i == 0 ) $ ans += $ i ; return $ ans ; }
function sumFactDiv ( $ n ) { return div ( fact ( $ n ) ) ; }
$ n = 4 ; echo sumFactDiv ( $ n ) ; ? >
< ? php $ allPrimes = array ( ) ;
function sieve ( $ n ) { global $ allPrimes ;
$ prime = array_fill ( 0 , $ n + 1 , true ) ;
for ( $ p = 2 ; $ p * $ p <= $ n ; $ p ++ ) {
if ( $ prime [ $ p ] == true ) {
for ( $ i = $ p * 2 ; $ i <= $ n ; $ i += $ p ) $ prime [ $ i ] = false ; } }
for ( $ p = 2 ; $ p <= $ n ; $ p ++ ) if ( $ prime [ $ p ] ) array_push ( $ allPrimes , $ p ) ; }
function factorialDivisors ( $ n ) { global $ allPrimes ;
$ result = 1 ;
for ( $ i = 0 ; $ i < count ( $ allPrimes ) ; $ i ++ ) {
$ p = $ allPrimes [ $ i ] ;
$ exp = 0 ; while ( $ p <= $ n ) { $ exp = $ exp + ( int ) ( $ n / $ p ) ; $ p = $ p * $ allPrimes [ $ i ] ; }
$ result = $ result * ( pow ( $ allPrimes [ $ i ] , $ exp +1 ) - 1 ) / ( $ allPrimes [ $ i ] - 1 ) ; }
return $ result ; }
print ( factorialDivisors ( 4 ) ) ; ? >
< ? php function checkPandigital ( $ b , $ n ) {
if ( strlen ( $ n ) < $ b ) return 0 ; $ hash = array ( ) ; for ( $ i = 0 ; $ i < $ b ; $ i ++ ) $ hash [ $ i ] = 0 ;
for ( $ i = 0 ; $ i < strlen ( $ n ) ; $ i ++ ) {
if ( $ n [ $ i ] >= '0' && $ n [ $ i ] <= '9' ) $ hash [ $ n [ $ i ] - '0' ] = 1 ;
else if ( ord ( $ n [ $ i ] ) - ord ( ' A ' ) <= $ b - 11 ) $ hash [ ord ( $ n [ $ i ] ) - ord ( ' A ' ) + 10 ] = 1 ; }
for ( $ i = 0 ; $ i < $ b ; $ i ++ ) if ( $ hash [ $ i ] == 0 ) return 0 ; return 1 ; }
$ b = 13 ; $ n = "1298450376ABC " ; if ( checkPandigital ( $ b , $ n ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function convert ( $ m , $ n ) { if ( $ m == $ n ) return 0 ;
if ( $ m > $ n ) return $ m - $ n ;
if ( $ m <= 0 && $ n > 0 ) return -1 ;
if ( $ n % 2 == 1 )
return 1 + convert ( $ m , $ n + 1 ) ;
else
return 1 + convert ( $ m , $ n / 2 ) ; }
{ $ m = 3 ; $ n = 11 ; echo " Minimum ▁ number ▁ of ▁ " . " operations ▁ : ▁ " , convert ( $ m , $ n ) ; return 0 ; } ? >
< ? php $ MAX = 10000 ; $ prodDig = array_fill ( 0 , $ MAX , 0 ) ;
function getDigitProduct ( $ x ) { global $ prodDig ;
if ( $ x < 10 ) return $ x ;
if ( $ prodDig [ $ x ] != 0 ) return $ prodDig [ $ x ] ;
$ prod = ( int ) ( $ x % 10 ) * getDigitProduct ( ( int ) ( $ x / 10 ) ) ; $ prodDig [ $ x ] = $ prod ; return $ prod ; }
function findSeed ( $ n ) {
$ res = array ( ) ; for ( $ i = 1 ; $ i <= ( int ) ( $ n / 2 + 1 ) ; $ i ++ ) if ( $ i * getDigitProduct ( $ i ) == $ n ) array_push ( $ res , $ i ) ;
if ( count ( $ res ) == 0 ) { echo " NO ▁ seed ▁ exists STRNEWLINE " ; return ; }
for ( $ i = 0 ; $ i < count ( $ res ) ; $ i ++ ) echo $ res [ $ i ] . " ▁ " ; }
$ n = 138 ; findSeed ( $ n ) ; ? >
< ? php function maxPrimefactorNum ( $ N ) { $ arr [ $ N + 5 ] = array ( ) ; $ arr = array_fill ( 0 , $ N + 1 , NULL ) ;
for ( $ i = 2 ; ( $ i * $ i ) <= $ N ; $ i ++ ) { if ( ! $ arr [ $ i ] ) for ( $ j = 2 * $ i ; $ j <= $ N ; $ j += $ i ) $ arr [ $ j ] ++ ; $ arr [ $ i ] = 1 ; } $ maxval = 0 ; $ maxint = 1 ;
for ( $ i = 1 ; $ i <= $ N ; $ i ++ ) { if ( $ arr [ $ i ] > $ maxval ) { $ maxval = $ arr [ $ i ] ; $ maxint = $ i ; } } return $ maxint ; }
$ N = 40 ; echo maxPrimefactorNum ( $ N ) , " STRNEWLINE " ; ? >
< ? php function SubArraySum ( $ arr , $ n ) { $ result = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ result += ( $ arr [ $ i ] * ( $ i + 1 ) * ( $ n - $ i ) ) ;
return $ result ; }
$ arr = array ( 1 , 2 , 3 ) ; $ n = sizeof ( $ arr ) ; echo " Sum ▁ of ▁ SubArray ▁ : ▁ " , SubArraySum ( $ arr , $ n ) , " STRNEWLINE " ; #This  code is contributed by aj_36 NEW_LINE ? >
< ? php function highestPowerof2 ( $ n ) { $ res = 0 ; for ( $ i = $ n ; $ i >= 1 ; $ i -- ) {
if ( ( $ i & ( $ i - 1 ) ) == 0 ) { $ res = $ i ; break ; } } return $ res ; }
$ n = 10 ; echo highestPowerof2 ( $ n ) ; ? >
< ? php function findPairs ( $ n ) {
$ cubeRoot = pow ( $ n , 1.0 / 3.0 ) ;
$ cube = array ( ) ;
for ( $ i = 1 ; $ i <= $ cubeRoot ; $ i ++ ) $ cube [ $ i ] = $ i * $ i * $ i ;
$ l = 1 ; $ r = $ cubeRoot ; while ( $ l < $ r ) { if ( $ cube [ $ l ] + $ cube [ $ r ] < $ n ) $ l ++ ; else if ( $ cube [ $ l ] + $ cube [ $ r ] > $ n ) $ r -- ; else { echo " ( " , $ l , " , ▁ " , floor ( $ r ) , " ) " ; echo " STRNEWLINE " ; $ l ++ ; $ r -- ; } } }
$ n = 20683 ; findPairs ( $ n ) ; ? >
< ? php function gcd ( $ a , $ b ) { while ( $ b != 0 ) { $ t = $ b ; $ b = $ a % $ b ; $ a = $ t ; } return $ a ; }
function findMinDiff ( $ a , $ b , $ x , $ y ) {
$ g = gcd ( $ a , $ b ) ;
$ diff = abs ( $ x - $ y ) % $ g ; return min ( $ diff , $ g - $ diff ) ; }
$ a = 20 ; $ b = 52 ; $ x = 5 ; $ y = 7 ; echo findMinDiff ( $ a , $ b , $ x , $ y ) , " STRNEWLINE " ; ? >
< ? php function printDivisors ( $ n ) {
$ v ; $ t = 0 ; for ( $ i = 1 ; $ i <= ( int ) sqrt ( $ n ) ; $ i ++ ) { if ( $ n % $ i == 0 ) {
if ( ( int ) $ n / $ i == $ i ) echo $ i . " " ; else { echo $ i . " " ;
$ v [ $ t ++ ] = ( int ) $ n / $ i ; } } }
for ( $ i = count ( $ v ) - 1 ; $ i >= 0 ; $ i -- ) echo $ v [ $ i ] . " ▁ " ; }
echo " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ; printDivisors ( 100 ) ; ? >
< ? php function printDivisors ( $ n ) { for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) if ( $ n % $ i == 0 ) echo $ i , " ▁ " ; }
echo " The ▁ divisors ▁ of ▁ 100 ▁ are : STRNEWLINE " ; printDivisors ( 100 ) ; ? >
< ? php function printDivisors ( $ n ) {
for ( $ i = 1 ; $ i <= sqrt ( $ n ) ; $ i ++ ) { if ( $ n % $ i == 0 ) {
if ( $ n / $ i == $ i ) echo $ i , " ▁ " ;
else echo $ i , " " , ▁ $ n / $ i , " " } } }
echo " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ; printDivisors ( 100 ) ; ? >
< ? php function SieveOfAtkin ( $ limit ) {
if ( $ limit > 2 ) echo 2 , " ▁ " ; if ( $ limit > 3 ) echo 3 , " ▁ " ;
$ sieve [ $ limit ] = 0 ; for ( $ i = 0 ; $ i < $ limit ; $ i ++ ) $ sieve [ $ i ] = false ;
for ( $ x = 1 ; $ x * $ x < $ limit ; $ x ++ ) { for ( $ y = 1 ; $ y * $ y < $ limit ; $ y ++ ) {
$ n = ( 4 * $ x * $ x ) + ( $ y * $ y ) ; if ( $ n <= $ limit && ( $ n % 12 == 1 $ n % 12 == 5 ) ) $ sieve [ $ n ] ^= true ; $ n = ( 3 * $ x * $ x ) + ( $ y * $ y ) ; if ( $ n <= $ limit && $ n % 12 == 7 ) $ sieve [ $ n ] = true ; $ n = ( 3 * $ x * $ x ) - ( $ y * $ y ) ; if ( $ x > $ y && $ n <= $ limit && $ n % 12 == 11 ) $ sieve [ $ n ] ^= true ; } }
for ( $ r = 5 ; $ r * $ r < $ limit ; $ r ++ ) { if ( $ sieve [ $ r ] ) { for ( $ i = $ r * $ r ; $ i < $ limit ; $ i += $ r * $ r ) $ sieve [ $ i ] = false ; } }
for ( $ a = 5 ; $ a < $ limit ; $ a ++ ) if ( $ sieve [ $ a ] ) echo $ a , " ▁ " ; }
$ limit = 20 ; SieveOfAtkin ( $ limit ) ; ? >
< ? php function isInside ( $ circle_x , $ circle_y , $ rad , $ x , $ y ) {
if ( ( $ x - $ circle_x ) * ( $ x - $ circle_x ) + ( $ y - $ circle_y ) * ( $ y - $ circle_y ) <= $ rad * $ rad ) return true ; else return false ; }
$ x = 1 ; $ y = 1 ; $ circle_x = 0 ; $ circle_y = 1 ; $ rad = 2 ; if ( isInside ( $ circle_x , $ circle_y , $ rad , $ x , $ y ) ) echo " Inside " ; else echo " Outside " ; ? >
< ? php function eval1 ( $ a , $ op , $ b ) { if ( $ op == ' + ' ) return $ a + $ b ; if ( $ op == ' - ' ) return $ a - $ b ; if ( $ op == ' * ' ) return $ a * $ b ; }
function eval1uateAll ( $ expr , $ low , $ high ) {
$ res = array ( ) ;
if ( $ low == $ high ) { array_push ( $ res , ord ( $ expr [ $ low ] ) - ord ( ' 0 ' ) ) ; return $ res ; }
if ( $ low == ( $ high - 2 ) ) { $ num = eval1 ( ord ( $ expr [ $ low ] ) - ord ( '0' ) , $ expr [ $ low + 1 ] , ord ( $ expr [ $ low + 2 ] ) - ord ( '0' ) ) ; array_push ( $ res , $ num ) ; return $ res ; }
for ( $ i = $ low + 1 ; $ i <= $ high ; $ i += 2 ) {
$ l = eval1uateAll ( $ expr , $ low , $ i - 1 ) ;
$ r = eval1uateAll ( $ expr , $ i + 1 , $ high ) ;
for ( $ s1 = 0 ; $ s1 < count ( $ l ) ; $ s1 ++ ) {
for ( $ s2 = 0 ; $ s2 < count ( $ r ) ; $ s2 ++ ) {
$ val = eval1 ( $ l [ $ s1 ] , $ expr [ $ i ] , $ r [ $ s2 ] ) ; array_push ( $ res , $ val ) ; } } } return $ res ; }
$ expr = "1*2 + 3*4" ; $ len = strlen ( $ expr ) ; $ ans = eval1uateAll ( $ expr , 0 , $ len - 1 ) ; for ( $ i = 0 ; $ i < count ( $ ans ) ; $ i ++ ) echo $ ans [ $ i ] . " STRNEWLINE " ; ? >
< ? php function isLucky ( $ n ) {
$ arr = array ( ) ; for ( $ i = 0 ; $ i < 10 ; $ i ++ ) $ arr [ $ i ] = false ;
while ( $ n > 0 ) {
$ digit = $ n % 10 ;
if ( $ arr [ $ digit ] ) return false ;
$ arr [ $ digit ] = true ;
$ n = ( int ) ( $ n / 10 ) ; } return true ; }
$ arr = array ( 1291 , 897 , 4566 , 1232 , 80 , 700 ) ; $ n = sizeof ( $ arr ) ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( isLucky ( $ arr [ $ i ] ) ) echo $ arr [ $ i ] , " ▁ is ▁ Lucky ▁ STRNEWLINE " ; else echo $ arr [ $ i ] , " ▁ is ▁ not ▁ Lucky ▁ STRNEWLINE " ; ? >
< ? php function printSquares ( $ n ) {
$ square = 0 ; $ odd = 1 ;
for ( $ x = 0 ; $ x < $ n ; $ x ++ ) {
echo $ square , " " ;
$ square = $ square + $ odd ; $ odd = $ odd + 2 ; } }
$ n = 5 ; printSquares ( $ n ) ; ? >
function reversDigits ( $ num ) { global $ rev_num ; global $ base_pos ; if ( $ num > 0 ) { reversDigits ( ( int ) ( $ num / 10 ) ) ; $ rev_num += ( $ num % 10 ) * $ base_pos ; $ base_pos *= 10 ; } return $ rev_num ; }
$ num = 4562 ; echo " Reverse ▁ of ▁ no . ▁ is ▁ " , reversDigits ( $ num ) ; ? >
< ? php function printSubsets ( $ n ) { for ( $ i = $ n ; $ i > 0 ; $ i = ( $ i - 1 ) & $ n ) echo $ i . " " ; echo "0" ; }
$ n = 9 ; printSubsets ( $ n ) ; ? >
< ? php function isDivisibleby17 ( $ n ) {
if ( $ n == 0 $ n == 17 ) return true ;
if ( $ n < 17 ) return false ;
return isDivisibleby17 ( ( int ) ( $ n >> 4 ) - ( int ) ( $ n & 15 ) ) ; }
$ n = 35 ; if ( isDivisibleby17 ( $ n ) ) echo $ n . " ▁ is ▁ divisible ▁ by ▁ 17" ; else echo $ n . " ▁ is ▁ not ▁ divisible ▁ by ▁ 17" ; ? >
< ? php function answer ( $ n ) {
$ m = 2 ;
$ ans = 1 ; $ r = 1 ;
while ( $ r < $ n ) {
$ r = ( pow ( 2 , $ m ) - 1 ) * ( pow ( 2 , $ m - 1 ) ) ;
if ( $ r < $ n ) $ ans = $ r ;
$ m ++ ; } return $ ans ; }
$ n = 7 ; echo answer ( $ n ) ; ? >
< ? php function setBitNumber ( $ n ) { if ( $ n == 0 ) return 0 ; $ msb = 0 ; $ n = $ n / 2 ; while ( $ n != 0 ) { $ n = $ n / 2 ; $ msb ++ ; } return ( 1 << $ msb ) ; }
$ n = 0 ; echo setBitNumber ( $ n ) ; ? >
< ? php function setBitNumber ( $ n ) {
$ n |= $ n >> 1 ;
$ n |= $ n >> 2 ; $ n |= $ n >> 4 ; $ n |= $ n >> 8 ; $ n |= $ n >> 16 ;
$ n = $ n + 1 ;
return ( $ n >> 1 ) ; }
$ n = 273 ; echo setBitNumber ( $ n ) ; ? >
< ? php function countTrailingZero ( $ x ) { $ count = 0 ; while ( ( $ x & 1 ) == 0 ) { $ x = $ x >> 1 ; $ count ++ ; } return $ count ; }
echo countTrailingZero ( 11 ) , " STRNEWLINE " ; ? >
< ? php function countTrailingZero ( $ x ) {
$ lookup = array ( 32 , 0 , 1 , 26 , 2 , 23 , 27 , 0 , 3 , 16 , 24 , 30 , 28 , 11 , 0 , 13 , 4 , 7 , 17 , 0 , 25 , 22 , 31 , 15 , 29 , 10 , 12 , 6 , 0 , 21 , 14 , 9 , 5 , 20 , 8 , 19 , 18 ) ;
return $ lookup [ ( - $ x & $ x ) % 37 ] ; }
echo countTrailingZero ( 48 ) , " STRNEWLINE " ; ? >
< ? php function multiplyBySevenByEight ( $ n ) {
return ( $ n - ( $ n >> 3 ) ) ; }
$ n = 9 ; echo multiplyBySevenByEight ( $ n ) ; ? >
< ? php function multiplyBySevenByEight ( $ n ) {
return ( ( $ n << 3 ) - $ n ) >> 3 ; }
$ n = 15 ; echo multiplyBySevenByEight ( $ n ) ; ? >
< ? php function getMaxMedian ( $ arr , $ n , $ k ) { $ size = $ n + $ k ;
sort ( $ arr , $ n ) ;
if ( $ size % 2 == 0 ) { $ median = ( float ) ( $ arr [ ( $ size / 2 ) - 1 ] + $ arr [ $ size / 2 ] ) / 2 ; return $ median ; }
$ median = $ arr [ $ size / 2 ] ; return $ median ; }
$ arr = array ( 3 , 2 , 3 , 4 , 2 ) ; $ n = sizeof ( $ arr ) ; $ k = 2 ; echo ( getMaxMedian ( $ arr , $ n , $ k ) ) ;
< ? php function printSorted ( $ a , $ b , $ c ) {
$ get_max = max ( $ a , max ( $ b , $ c ) ) ;
$ get_min = - max ( - $ a , max ( - $ b , - $ c ) ) ; $ get_mid = ( $ a + $ b + $ c ) - ( $ get_max + $ get_min ) ; echo $ get_min , " " ▁ , ▁ $ get _ mid , ▁ " " }
$ a = 4 ; $ b = 1 ; $ c = 9 ; printSorted ( $ a , $ b , $ c ) ; ? >
< ? php function insertionSort ( & $ arr , $ n ) { for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { $ key = $ arr [ $ i ] ; $ j = $ i - 1 ;
while ( $ j >= 0 && $ arr [ $ j ] > $ key ) { $ arr [ $ j + 1 ] = $ arr [ $ j ] ; $ j = $ j - 1 ; } $ arr [ $ j + 1 ] = $ key ; } }
function printArray ( & $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; echo " STRNEWLINE " ; }
$ arr = array ( 12 , 11 , 13 , 5 , 6 ) ; $ n = sizeof ( $ arr ) ; insertionSort ( $ arr , $ n ) ; printArray ( $ arr , $ n ) ; ? >
< ? php function countPaths ( $ n , $ m ) {
for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) $ dp [ $ i ] [ 0 ] = 1 ; for ( $ i = 0 ; $ i <= $ m ; $ i ++ ) $ dp [ 0 ] [ $ i ] = 1 ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) for ( $ j = 1 ; $ j <= $ m ; $ j ++ ) $ dp [ $ i ] [ $ j ] = $ dp [ $ i - 1 ] [ $ j ] + $ dp [ $ i ] [ $ j - 1 ] ; return $ dp [ $ n ] [ $ m ] ; }
$ n = 3 ; $ m = 2 ; echo " ▁ Number ▁ of ▁ Paths ▁ " , countPaths ( $ n , $ m ) ; ? >
< ? php function coun ( $ S , $ m , $ n ) {
if ( $ n == 0 ) return 1 ;
if ( $ n < 0 ) return 0 ;
if ( $ m <= 0 && $ n >= 1 ) return 0 ;
return coun ( $ S , $ m - 1 , $ n ) + coun ( $ S , $ m , $ n - $ S [ $ m - 1 ] ) ; }
$ arr = array ( 1 , 2 , 3 ) ; $ m = count ( $ arr ) ; echo coun ( $ arr , $ m , 4 ) ; ? >
< ? php function isVowel ( $ c ) { return ( $ c == ' a ' $ c == ' e ' $ c == ' i ' $ c == ' o ' $ c == ' u ' ) ; }
function encryptString ( $ s , $ n , $ k ) { $ countVowels = 0 ; $ countConsonants = 0 ; $ ans = " " ;
for ( $ l = 0 ; $ l <= $ n - $ k ; $ l ++ ) { $ countVowels = 0 ; $ countConsonants = 0 ;
for ( $ r = $ l ; $ r <= $ l + $ k - 1 ; $ r ++ ) {
if ( isVowel ( $ s [ $ r ] ) == true ) $ countVowels ++ ; else $ countConsonants ++ ; }
$ ans = $ ans . ( string ) ( $ countVowels * $ countConsonants ) ; } return $ ans ; }
$ s = " hello " ; $ n = strlen ( $ s ) ; $ k = 2 ; echo encryptString ( $ s , $ n , $ k ) . " STRNEWLINE " ;
< ? php function findVolume ( $ a ) {
if ( $ a < 0 ) return -1 ;
$ r = $ a / 2 ;
$ h = $ a ;
$ V = 3.14 * pow ( $ r , 2 ) * $ h ; return $ V ; }
$ a = 5 ; echo findVolume ( $ a ) . " STRNEWLINE " ;
< ? php function volumeTriangular ( $ a , $ b , $ h ) { $ vol = ( 0.1666 ) * $ a * $ b * $ h ; return $ vol ; }
function volumeSquare ( $ b , $ h ) { $ vol = ( 0.33 ) * $ b * $ b * $ h ; return $ vol ; }
function volumePentagonal ( $ a , $ b , $ h ) { $ vol = ( 0.83 ) * $ a * $ b * $ h ; return $ vol ; }
function volumeHexagonal ( $ a , $ b , $ h ) { $ vol = $ a * $ b * $ h ; return $ vol ; }
$ b = 4 ; $ h = 9 ; $ a = 4 ; echo ( " Volume ▁ of ▁ triangular ▁ base ▁ pyramid ▁ is ▁ " ) ; echo ( volumeTriangular ( $ a , $ b , $ h ) ) ; echo ( " STRNEWLINE " ) ; echo ( " Volume ▁ of ▁ square ▁ base ▁ pyramid ▁ is ▁ " ) ; echo ( volumeSquare ( $ b , $ h ) ) ; echo ( " STRNEWLINE " ) ; echo ( " Volume ▁ of ▁ pentagonal ▁ base ▁ pyramid ▁ is ▁ " ) ; echo ( volumePentagonal ( $ a , $ b , $ h ) ) ; echo ( " STRNEWLINE " ) ; echo ( " Volume ▁ of ▁ Hexagonal ▁ base ▁ pyramid ▁ is ▁ " ) ; echo ( volumeHexagonal ( $ a , $ b , $ h ) ) ; ? >
< ? php function Area ( $ b1 , $ b2 , $ h ) { return ( ( $ b1 + $ b2 ) / 2 ) * $ h ; }
$ base1 = 8 ; $ base2 = 10 ; $ height = 6 ; $ area = Area ( $ base1 , $ base2 , $ height ) ; echo ( " Area ▁ is : ▁ " ) ; echo ( $ area ) ; ? >
< ? php function numberOfDiagonals ( $ n ) { return $ n * ( $ n - 3 ) / 2 ; }
$ n = 5 ; echo $ n , " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ; echo numberOfDiagonals ( $ n ) , " ▁ diagonals " ;
< ? php function Printksubstring ( $ str , $ n , $ k ) {
$ total = floor ( ( $ n * ( $ n + 1 ) ) / 2 ) ;
if ( $ k > $ total ) { printf ( " - 1 STRNEWLINE " ) ; return ; }
$ substring = array ( ) ; $ substring [ 0 ] = 0 ;
$ temp = $ n ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) {
$ substring [ $ i ] = $ substring [ $ i - 1 ] + $ temp ; $ temp -- ; }
$ l = 1 ; $ h = $ n ; $ start = 0 ; while ( $ l <= $ h ) { $ m = floor ( ( $ l + $ h ) / 2 ) ; if ( $ substring [ $ m ] > $ k ) { $ start = $ m ; $ h = $ m - 1 ; } else if ( $ substring [ $ m ] < $ k ) $ l = $ m + 1 ; else { $ start = $ m ; break ; } }
$ end = $ n - ( $ substring [ $ start ] - $ k ) ;
for ( $ i = $ start - 1 ; $ i < $ end ; $ i ++ ) print ( $ str [ $ i ] ) ; }
$ str = " abc " ; $ k = 4 ; $ n = strlen ( $ str ) ; Printksubstring ( $ str , $ n , $ k ) ; ? >
< ? php function LowerInsertionPoint ( $ arr , $ n , $ X ) {
if ( $ X < $ arr [ 0 ] ) return 0 ; else if ( $ X > $ arr [ $ n - 1 ] ) return $ n ; $ lowerPnt = 0 ; $ i = 1 ; while ( $ i < $ n && $ arr [ $ i ] < $ X ) { $ lowerPnt = $ i ; $ i = $ i * 2 ; }
while ( $ lowerPnt < $ n && $ arr [ $ lowerPnt ] < $ X ) $ lowerPnt ++ ; return $ lowerPnt ; }
$ arr = array ( 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 ) ; $ n = sizeof ( $ arr ) ; $ X = 4 ; echo LowerInsertionPoint ( $ arr , $ n , $ X ) ; ? >
< ? php function getCount ( $ M , $ N ) { $ count = 0 ;
if ( $ M == 1 ) return $ N ;
if ( $ N == 1 ) return $ M ; if ( $ N > $ M ) {
for ( $ i = 1 ; $ i <= $ M ; $ i ++ ) { $ numerator = $ N * $ i - $ N + $ M - $ i ; $ denominator = $ M - 1 ;
if ( $ numerator % $ denominator == 0 ) { $ j = $ numerator / $ denominator ;
if ( $ j >= 1 and $ j <= $ N ) $ count ++ ; } } } else {
for ( $ j = 1 ; $ j <= $ N ; $ j ++ ) { $ numerator = $ M * $ j - $ M + $ N - $ j ; $ denominator = $ N - 1 ;
if ( $ numerator % $ denominator == 0 ) { $ i = $ numerator / $ denominator ;
if ( $ i >= 1 and $ i <= $ M ) $ count ++ ; } } } return $ count ; }
$ M = 3 ; $ N = 5 ; echo getCount ( $ M , $ N ) ; ? >
< ? php function middleOfThree ( $ a , $ b , $ c ) {
if ( $ a > $ b ) { if ( $ b > $ c ) return $ b ; else if ( $ a > $ c ) return $ c ; else return $ a ; } else {
if ( $ a > $ c ) return $ a ; else if ( $ b > $ c ) return $ c ; else return $ b ; } }
$ a = 20 ; $ b = 30 ; $ c = 40 ; echo middleOfThree ( $ a , $ b , $ c ) ; ? >
< ? php function printArr ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] ; }
function compare ( $ num1 , $ num2 ) {
$ A = ( string ) $ num1 ;
$ B = ( string ) $ num2 ;
if ( ( int ) ( $ A . $ B ) <= ( int ) ( $ B . $ A ) ) { return true ; } else return false ; }
function printSmallest ( $ N , $ arr ) {
$ arr = sort_arr ( $ arr ) ;
printArr ( $ arr , $ N ) ; }
$ arr = array ( 5 , 6 , 2 , 9 , 21 , 1 ) ; $ N = count ( $ arr ) ; printSmallest ( $ N , $ arr ) ; ? >
< ? php function isPossible ( $ a , $ b , $ n , $ k ) {
sort ( $ a ) ;
rsort ( $ b ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ a [ $ i ] + $ b [ $ i ] < $ k ) return false ; return true ; }
$ a = array ( 2 , 1 , 3 ) ; $ b = array ( 7 , 8 , 9 ) ; $ k = 10 ; $ n = count ( $ a ) ; if ( isPossible ( $ a , $ b , $ n , $ k ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function encryptString ( $ str , $ n ) { $ i = 0 ; $ cnt = 0 ; $ encryptedStr = " " ; while ( $ i < $ n ) {
$ cnt = $ i + 1 ;
while ( $ cnt -- ) $ encryptedStr . = $ str [ $ i ] ; $ i ++ ; } return $ encryptedStr ; }
$ str = " geeks " ; $ n = strlen ( $ str ) ; echo encryptString ( $ str , $ n ) ; ? >
< ? php function minDiff ( $ n , $ x , $ A ) { $ mn = $ A [ 0 ] ; $ mx = $ A [ 0 ] ;
for ( $ i = 0 ; $ i < $ n ; ++ $ i ) { $ mn = min ( $ mn , $ A [ $ i ] ) ; $ mx = max ( $ mx , $ A [ $ i ] ) ; }
return max ( 0 , $ mx - $ mn - 2 * $ x ) ; }
$ n = 3 ; $ x = 3 ; $ A = array ( 1 , 3 , 6 ) ;
echo minDiff ( $ n , $ x , $ A ) ; ? >
