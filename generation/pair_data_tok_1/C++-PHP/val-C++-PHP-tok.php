< ? php function Conversion ( $ centi ) { $ pixels = ( 96 * $ centi ) / 2.54 ; echo ( $ pixels . " " ) ; }
$ centi = 15 ; Conversion ( $ centi ) ; ? >
< ? php function maxOfMin ( $ a , $ n , $ S ) {
$ mi = PHP_INT_MAX ;
$ s1 = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ s1 += $ a [ $ i ] ; $ mi = min ( $ a [ $ i ] , $ mi ) ; }
if ( $ s1 < $ S ) return -1 ;
if ( $ s1 == $ S ) return 0 ;
$ low = 0 ;
$ high = $ mi ;
$ ans ;
while ( $ low <= $ high ) { $ mid = ( $ low + $ high ) / 2 ;
if ( $ s1 - ( $ mid * $ n ) >= $ S ) { $ ans = $ mid ; $ low = $ mid + 1 ; }
else $ high = $ mid - 1 ; }
return $ ans ; }
$ a = array ( 10 , 10 , 10 , 10 , 10 ) ; $ S = 10 ; $ n = sizeof ( $ a ) ; echo maxOfMin ( $ a , $ n , $ S ) ; ? >
< ? php function Alphabet_N_Pattern ( $ N ) { $ index ; $ side_index ; $ size ;
$ Right = 1 ; $ Left = 1 ; $ Diagonal = 2 ;
for ( $ index = 0 ; $ index < $ N ; $ index ++ ) {
echo $ Left ++ ;
for ( $ side_index = 0 ; $ side_index < 2 * ( $ index ) ; $ side_index ++ ) echo " ▁ " ;
if ( $ index != 0 && $ index != $ N - 1 ) echo $ Diagonal ++ ; else echo " ▁ " ;
for ( $ side_index = 0 ; $ side_index < 2 * ( $ N - $ index - 1 ) ; $ side_index ++ ) echo " ▁ " ;
echo $ Right ++ ; echo " STRNEWLINE " ; } }
$ Size = 6 ;
Alphabet_N_Pattern ( $ Size ) ; ? >
< ? php function isSumDivides ( $ N ) { $ temp = $ N ; $ sum = 0 ;
while ( $ temp ) { $ sum += $ temp % 10 ; $ temp = ( int ) $ temp / 10 ; } if ( $ N % $ sum == 0 ) return 1 ; else return 0 ; }
$ N = 12 ; if ( isSumDivides ( $ N ) ) echo " YES " ; else echo " NO " ; ? >
< ? php function sum ( $ N ) { $ S1 ; $ S2 ; $ S3 ; $ S1 = ( ( $ N / 3 ) ) * ( 2 * 3 + ( $ N / 3 - 1 ) * 3 ) / 2 ; $ S2 = ( ( $ N / 4 ) ) * ( 2 * 4 + ( $ N / 4 - 1 ) * 4 ) / 2 ; $ S3 = ( ( $ N / 12 ) ) * ( 2 * 12 + ( $ N / 12 - 1 ) * 12 ) / 2 ; return $ S1 + $ S2 - $ S3 ; }
$ N = 20 ; echo sum ( 12 ) ; ? >
< ? php function nextGreater ( $ N ) { $ power_of_2 = 1 ; $ shift_count = 0 ;
while ( true ) {
if ( ( ( $ N >> $ shift_count ) & 1 ) % 2 == 0 ) break ;
$ shift_count ++ ;
$ power_of_2 = $ power_of_2 * 2 ; }
return ( $ N + $ power_of_2 ) ; }
$ N = 11 ;
echo " The ▁ next ▁ number ▁ is ▁ = ▁ " , nextGreater ( $ N ) ; ? >
< ? php function printTetra ( $ n ) { $ dp = array_fill ( 0 , $ n + 5 , 0 ) ;
$ dp [ 0 ] = 0 ; $ dp [ 1 ] = $ dp [ 2 ] = 1 ; $ dp [ 3 ] = 2 ; for ( $ i = 4 ; $ i <= $ n ; $ i ++ ) $ dp [ $ i ] = $ dp [ $ i - 1 ] + $ dp [ $ i - 2 ] + $ dp [ $ i - 3 ] + $ dp [ $ i - 4 ] ; echo $ dp [ $ n ] ; }
$ n = 10 ; printTetra ( $ n ) ; ? >
< ? php function maxSum1 ( $ arr , $ n ) { $ dp [ $ n ] = array ( ) ; $ maxi = 0 ; for ( $ i = 0 ; $ i < $ n - 1 ; $ i ++ ) {
$ dp [ $ i ] = $ arr [ $ i ] ;
if ( $ maxi < $ arr [ $ i ] ) $ maxi = $ arr [ $ i ] ; }
for ( $ i = 2 ; $ i < $ n - 1 ; $ i ++ ) {
for ( $ j = 0 ; $ j < $ i - 1 ; $ j ++ ) {
if ( $ dp [ $ i ] < $ dp [ $ j ] + $ arr [ $ i ] ) { $ dp [ $ i ] = $ dp [ $ j ] + $ arr [ $ i ] ;
if ( $ maxi < $ dp [ $ i ] ) $ maxi = $ dp [ $ i ] ; } } }
return $ maxi ; }
function maxSum2 ( $ arr , $ n ) { $ dp [ $ n ] = array ( ) ; $ maxi = 0 ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { $ dp [ $ i ] = $ arr [ $ i ] ; if ( $ maxi < $ arr [ $ i ] ) $ maxi = $ arr [ $ i ] ; }
for ( $ i = 3 ; $ i < $ n ; $ i ++ ) {
for ( $ j = 1 ; $ j < $ i - 1 ; $ j ++ ) {
if ( $ dp [ $ i ] < $ arr [ $ i ] + $ dp [ $ j ] ) { $ dp [ $ i ] = $ arr [ $ i ] + $ dp [ $ j ] ;
if ( $ maxi < $ dp [ $ i ] ) $ maxi = $ dp [ $ i ] ; } } }
return $ maxi ; } function findMaxSum ( $ arr , $ n ) { return max ( maxSum1 ( $ arr , $ n ) , maxSum2 ( $ arr , $ n ) ) ; }
$ arr = array ( 1 , 2 , 3 , 1 ) ; $ n = sizeof ( $ arr ) ; echo findMaxSum ( $ arr , $ n ) ; ? >
< ? php function permutationCoeff ( $ n , $ k ) { $ fact = array ( ) ;
$ fact [ 0 ] = 1 ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ fact [ $ i ] = $ i * $ fact [ $ i - 1 ] ;
return $ fact [ $ n ] / $ fact [ $ n - $ k ] ; }
$ n = 10 ; $ k = 2 ; echo " Value ▁ of ▁ P ( " , $ n , " ▁ " , $ k , " ) ▁ is ▁ " , permutationCoeff ( $ n , $ k ) ; ? >
< ? php function isSubsetSum ( $ set , $ n , $ sum ) {
if ( $ sum == 0 ) return true ; if ( $ n == 0 ) return false ;
if ( $ set [ $ n - 1 ] > $ sum ) return isSubsetSum ( $ set , $ n - 1 , $ sum ) ;
return isSubsetSum ( $ set , $ n - 1 , $ sum ) || isSubsetSum ( $ set , $ n - 1 , $ sum - $ set [ $ n - 1 ] ) ; }
$ set = array ( 3 , 34 , 4 , 12 , 5 , 2 ) ; $ sum = 9 ; $ n = 6 ; if ( isSubsetSum ( $ set , $ n , $ sum ) == true ) echo " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else echo " No ▁ subset ▁ with ▁ given ▁ sum " ; ? >
< ? php function no_of_ways ( $ s ) { $ n = strlen ( $ s ) ;
$ count_left = 0 ; $ count_right = 0 ;
for ( $ i = 0 ; $ i < $ n ; ++ $ i ) { if ( $ s [ $ i ] == $ s [ 0 ] ) { ++ $ count_left ; } else break ; }
for ( $ i = $ n - 1 ; $ i >= 0 ; -- $ i ) { if ( $ s [ $ i ] == $ s [ $ n - 1 ] ) { ++ $ count_right ; } else break ; }
if ( $ s [ 0 ] == $ s [ $ n - 1 ] ) return ( ( $ count_left + 1 ) * ( $ count_right + 1 ) ) ;
else return ( $ count_left + $ count_right + 1 ) ; }
$ s = " geeksforgeeks " ; echo no_of_ways ( $ s ) ; ? >
< ? php function preCompute ( $ n , $ s , & $ pref ) { $ pref [ 0 ] = 0 ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { $ pref [ $ i ] = $ pref [ $ i - 1 ] ; if ( $ s [ $ i - 1 ] == $ s [ $ i ] ) $ pref [ $ i ] ++ ; } }
function query ( & $ pref , $ l , $ r ) { return $ pref [ $ r ] - $ pref [ $ l ] ; }
$ s = " ggggggg " ; $ n = strlen ( $ s ) ; $ pref = array_fill ( 0 , $ n , NULL ) ; preCompute ( $ n , $ s , $ pref ) ;
$ l = 1 ; $ r = 2 ; echo query ( $ pref , $ l , $ r ) . " STRNEWLINE " ;
$ l = 1 ; $ r = 5 ; echo query ( $ pref , $ l , $ r ) . " STRNEWLINE " ; ? >
< ? php function findDirection ( $ s ) { $ count = 0 ; $ d = " " ; for ( $ i = 0 ; $ i < strlen ( $ s ) ; $ i ++ ) { if ( $ s [ 0 ] == ' ' ) return null ; if ( $ s [ $ i ] == ' L ' ) $ count -= 1 ; else { if ( $ s [ $ i ] == ' R ' ) $ count += 1 ; } }
if ( $ count > 0 ) { if ( $ count % 4 == 0 ) $ d = " N " ; else if ( $ count % 4 == 1 ) $ d = " E " ; else if ( $ count % 4 == 2 ) $ d = " S " ; else if ( $ count % 4 == 3 ) $ d = " W " ; }
if ( $ count < 0 ) { if ( $ count % 4 == 0 ) $ d = " N " ; else if ( $ count % 4 == -1 ) $ d = " W " ; else if ( $ count % 4 == -2 ) $ d = " S " ; else if ( $ count % 4 == -3 ) $ d = " E " ; } return $ d ; }
$ s = " LLRLRRL " ; echo findDirection ( $ s ) . " STRNEWLINE " ; $ s = " LL " ; echo findDirection ( $ s ) . " STRNEWLINE " ; ? >
< ? php function encode ( $ s , $ k ) {
$ newS = " " ;
for ( $ i = 0 ; $ i < strlen ( $ s ) ; ++ $ i ) {
$ val = ord ( $ s [ $ i ] ) ;
$ dup = $ k ;
if ( $ val + $ k > 122 ) { $ k -= ( 122 - $ val ) ; $ k = $ k % 26 ; $ newS = $ newS . chr ( 96 + $ k ) ; } else $ newS = $ newS . chr ( $ val + $ k ) ; $ k = $ dup ; }
echo $ newS ; }
$ str = " abc " ; $ k = 28 ;
encode ( $ str , $ k ) ; ? >
< ? php function isVowel ( $ x ) { if ( $ x == ' a ' $ x == ' e ' $ x == ' i ' $ x == ' o ' $ x == ' u ' ) return true ; else return false ; }
function updateSandwichedVowels ( $ a ) { $ n = strlen ( $ a ) ;
$ updatedString = " " ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( ! $ i $ i == $ n - 1 ) { $ updatedString . = $ a [ $ i ] ; continue ; }
if ( isVowel ( $ a [ $ i ] ) && ! isVowel ( $ a [ $ i - 1 ] ) && ! isVowel ( $ a [ $ i + 1 ] ) ) { continue ; }
$ updatedString . = $ a [ $ i ] ; } return $ updatedString ; }
$ str = " geeksforgeeks " ;
$ updatedString = updateSandwichedVowels ( $ str ) ; echo $ updatedString ; ? >
< ? php function findNumbers ( $ n , $ w ) { $ x = 0 ; $ sum = 0 ;
if ( $ w >= 0 && $ w <= 8 ) {
$ x = 9 - $ w ; }
else if ( $ w >= -9 && $ w <= -1 ) {
$ x = 10 + $ w ; } $ sum = pow ( 10 , $ n - 2 ) ; $ sum = ( $ x * $ sum ) ; return $ sum ; }
$ n = 3 ; $ w = 4 ;
echo findNumbers ( $ n , $ w ) ;
< ? php function MaximumHeight ( $ a , $ n ) { $ result = 1 ; for ( $ i = 1 ; $ i <= $ n ; ++ $ i ) {
$ y = ( $ i * ( $ i + 1 ) ) / 2 ;
if ( $ y < $ n ) $ result = $ i ;
else break ; } return $ result ; }
$ arr = array ( 40 , 100 , 20 , 30 ) ; $ n = count ( $ arr ) ; echo MaximumHeight ( $ arr , $ n ) ; ? >
< ? php function findK ( $ n , $ k ) { $ a ; $ index = 0 ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) if ( $ i % 2 == 1 ) $ a [ $ index ++ ] = $ i ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) if ( $ i % 2 == 0 ) $ a [ $ index ++ ] = $ i ; return ( $ a [ $ k - 1 ] ) ; }
$ n = 10 ; $ k = 3 ; echo findK ( $ n , $ k ) ; ? >
< ? php function factorial ( $ n ) {
return ( $ n == 1 $ n == 0 ) ? 1 : $ n * factorial ( $ n - 1 ) ; }
$ num = 5 ; echo " Factorial ▁ of ▁ " , $ num , " ▁ is ▁ " , factorial ( $ num ) ; ? >
< ? php function pell ( $ n ) { if ( $ n <= 2 ) return $ n ; $ a = 1 ; $ b = 2 ; $ c ; $ i ; for ( $ i = 3 ; $ i <= $ n ; $ i ++ ) { $ c = 2 * $ b + $ a ; $ a = $ b ; $ b = $ c ; } return $ b ; }
$ n = 4 ; echo ( pell ( $ n ) ) ; ? >
< ? php function isMultipleOf10 ( $ n ) { return ( $ n % 15 == 0 ) ; }
$ n = 30 ; if ( isMultipleOf10 ( $ n ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; ? >
< ? php function countOddPrimeFactors ( $ n ) { $ result = 1 ;
while ( $ n % 2 == 0 ) $ n /= 2 ;
for ( $ i = 3 ; $ i * $ i <= $ n ; $ i += 2 ) { $ divCount = 0 ;
while ( $ n % $ i == 0 ) { $ n /= $ i ; ++ $ divCount ; } $ result *= $ divCount + 1 ; }
if ( $ n > 2 ) $ result *= 2 ; return $ result ; } function politness ( $ n ) { return countOddPrimeFactors ( $ n ) - 1 ; }
$ n = 90 ; echo " Politness ▁ of ▁ " , $ n , " ▁ = ▁ " , politness ( $ n ) , " STRNEWLINE " ; $ n = 15 ; echo " Politness ▁ of ▁ " , $ n , " ▁ = ▁ " , politness ( $ n ) , " STRNEWLINE " ; ? >
< ? php $ MAX = 10000 ;
$ primes = array ( ) ;
function Sieve ( ) { global $ MAX , $ primes ; $ n = $ MAX ;
$ nNew = ( int ) ( sqrt ( $ n ) ) ;
$ marked = array_fill ( 0 , ( int ) ( $ n / 2 + 500 ) , 0 ) ;
for ( $ i = 1 ; $ i <= ( $ nNew - 1 ) / 2 ; $ i ++ ) for ( $ j = ( $ i * ( $ i + 1 ) ) << 1 ; $ j <= $ n / 2 ; $ j = $ j + 2 * $ i + 1 ) $ marked [ $ j ] = 1 ;
array_push ( $ primes , 2 ) ;
for ( $ i = 1 ; $ i <= $ n / 2 ; $ i ++ ) if ( $ marked [ $ i ] == 0 ) array_push ( $ primes , 2 * $ i + 1 ) ; }
function binarySearch ( $ left , $ right , $ n ) { global $ primes ; if ( $ left <= $ right ) { $ mid = ( int ) ( ( $ left + $ right ) / 2 ) ;
if ( $ mid == 0 || $ mid == count ( $ primes ) - 1 ) return $ primes [ $ mid ] ;
if ( $ primes [ $ mid ] == $ n ) return $ primes [ $ mid - 1 ] ;
if ( $ primes [ $ mid ] < $ n && $ primes [ $ mid + 1 ] > $ n ) return $ primes [ $ mid ] ; if ( $ n < $ primes [ $ mid ] ) return binarySearch ( $ left , $ mid - 1 , $ n ) ; else return binarySearch ( $ mid + 1 , $ right , $ n ) ; } return 0 ; }
Sieve ( ) ; $ n = 17 ; echo binarySearch ( 0 , count ( $ primes ) - 1 , $ n ) ; ? >
< ? php function factorial ( $ n ) { if ( $ n == 0 ) return 1 ; return $ n * factorial ( $ n - 1 ) ; }
$ num = 5 ; echo " Factorial ▁ of ▁ " , $ num , " ▁ is ▁ " , factorial ( $ num ) ; ? >
< ? php function printKDistinct ( $ arr , $ n , $ k ) { $ dist_count = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
$ j ; for ( $ j = 0 ; $ j < $ n ; $ j ++ ) if ( $ i != $ j && $ arr [ $ j ] == $ arr [ $ i ] ) break ;
if ( $ j == $ n ) $ dist_count ++ ; if ( $ dist_count == $ k ) return $ arr [ $ i ] ; } return -1 ; }
$ ar = array ( 1 , 2 , 1 , 3 , 4 , 2 ) ; $ n = sizeof ( $ ar ) / sizeof ( $ ar [ 0 ] ) ; $ k = 2 ; echo printKDistinct ( $ ar , $ n , $ k ) ; ? >
< ? php function calculate ( $ a , $ n ) {
sort ( $ a ) ; $ count = 1 ; $ answer = 0 ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { if ( $ a [ $ i ] == $ a [ $ i - 1 ] ) {
$ count += 1 ; } else {
$ answer = $ answer + ( $ count * ( $ count - 1 ) ) / 2 ; $ count = 1 ; } } $ answer = $ answer + ( $ count * ( $ count - 1 ) ) / 2 ; return $ answer ; }
$ a = array ( 1 , 2 , 1 , 2 , 4 ) ; $ n = count ( $ a ) ;
echo calculate ( $ a , $ n ) ; ? >
< ? php function calculate ( $ a , $ n ) {
$ maximum = max ( $ a ) ;
$ frequency = array_fill ( 0 , $ maximum + 1 , 0 ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
$ frequency [ $ a [ $ i ] ] += 1 ; } $ answer = 0 ;
for ( $ i = 0 ; $ i < ( $ maximum ) + 1 ; $ i ++ ) {
$ answer = $ answer + $ frequency [ $ i ] * ( $ frequency [ $ i ] - 1 ) ; } return $ answer / 2 ; }
$ a = array ( 1 , 2 , 1 , 2 , 4 ) ; $ n = count ( $ a ) ;
echo ( calculate ( $ a , $ n ) ) ; ? >
< ? php function findSubArray ( & $ arr , $ n ) { $ sum = 0 ; $ maxsize = -1 ;
for ( $ i = 0 ; $ i < $ n - 1 ; $ i ++ ) { $ sum = ( $ arr [ $ i ] == 0 ) ? -1 : 1 ;
for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) { ( $ arr [ $ j ] == 0 ) ? ( $ sum += -1 ) : ( $ sum += 1 ) ;
if ( $ sum == 0 && $ maxsize < $ j - $ i + 1 ) { $ maxsize = $ j - $ i + 1 ; $ startindex = $ i ; } } } if ( $ maxsize == -1 ) echo " No ▁ such ▁ subarray " ; else echo $ startindex . " ▁ to ▁ " . ( $ startindex + $ maxsize - 1 ) ; return $ maxsize ; }
$ arr = array ( 1 , 0 , 0 , 1 , 0 , 1 , 1 ) ; $ size = sizeof ( $ arr ) ; findSubArray ( $ arr , $ size ) ; ? >
< ? php function findMax ( $ arr , $ low , $ high ) {
if ( $ high <= $ low ) return $ arr [ $ low ] ;
$ mid = $ low + ( $ high - $ low ) / 2 ;
if ( $ mid == 0 && $ arr [ $ mid ] > $ arr [ $ mid - 1 ] ) return $ arr [ 0 ] ;
if ( $ mid < $ high && $ arr [ $ mid + 1 ] < $ arr [ $ mid ] && $ mid > 0 && $ arr [ $ mid ] > $ arr [ $ mid - 1 ] ) { return $ arr [ $ mid ] ; }
if ( $ arr [ $ low ] > $ arr [ $ mid ] ) { return findMax ( $ arr , $ low , $ mid - 1 ) ; } else { return findMax ( $ arr , $ mid + 1 , $ high ) ; } }
$ arr = array ( 5 , 6 , 1 , 2 , 3 , 4 ) ; $ n = sizeof ( $ arr ) ; echo findMax ( $ arr , 0 , $ n - 1 ) ;
< ? php function search ( $ arr , $ l , $ h , $ key ) { if ( $ l > $ h ) return -1 ; $ mid = ( $ l + $ h ) / 2 ; if ( $ arr [ $ mid ] == $ key ) return $ mid ;
if ( $ arr [ $ l ] <= $ arr [ $ mid ] ) {
if ( $ key >= $ arr [ $ l ] && $ key <= $ arr [ $ mid ] ) return search ( $ arr , $ l , $ mid - 1 , $ key ) ;
return search ( $ arr , $ mid + 1 , $ h , $ key ) ; }
if ( $ key >= $ arr [ $ mid ] && $ key <= $ arr [ $ h ] ) return search ( $ arr , $ mid + 1 , $ h , $ key ) ; return search ( $ arr , $ l , $ mid - 1 , $ key ) ; }
$ arr = array ( 4 , 5 , 6 , 7 , 8 , 9 , 1 , 2 , 3 ) ; $ n = sizeof ( $ arr ) ; $ key = 6 ; $ i = search ( $ arr , 0 , $ n - 1 , $ key ) ; if ( $ i != -1 ) echo " Index : ▁ " , floor ( $ i ) , " ▁ STRNEWLINE " ; else echo " Key ▁ not ▁ found " ; ? >
< ? php function findMin ( $ arr , $ low , $ high ) {
if ( $ high < $ low ) return $ arr [ 0 ] ;
if ( $ high == $ low ) return $ arr [ $ low ] ;
$ mid = $ low + ( $ high - $ low ) / 2 ;
if ( $ mid < $ high && $ arr [ $ mid + 1 ] < $ arr [ $ mid ] ) return $ arr [ $ mid + 1 ] ;
if ( $ mid > $ low && $ arr [ $ mid ] < $ arr [ $ mid - 1 ] ) return $ arr [ $ mid ] ;
if ( $ arr [ $ high ] > $ arr [ $ mid ] ) return findMin ( $ arr , $ low , $ mid - 1 ) ; return findMin ( $ arr , $ mid + 1 , $ high ) ; }
$ arr1 = array ( 5 , 6 , 1 , 2 , 3 , 4 ) ; $ n1 = sizeof ( $ arr1 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr1 , 0 , $ n1 - 1 ) . " STRNEWLINE " ; $ arr2 = array ( 1 , 2 , 3 , 4 ) ; $ n2 = sizeof ( $ arr2 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr2 , 0 , $ n2 - 1 ) . " STRNEWLINE " ; $ arr3 = array ( 1 ) ; $ n3 = sizeof ( $ arr3 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr3 , 0 , $ n3 - 1 ) . " STRNEWLINE " ; $ arr4 = array ( 1 , 2 ) ; $ n4 = sizeof ( $ arr4 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr4 , 0 , $ n4 - 1 ) . " STRNEWLINE " ; $ arr5 = array ( 2 , 1 ) ; $ n5 = sizeof ( $ arr5 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr5 , 0 , $ n5 - 1 ) . " STRNEWLINE " ; $ arr6 = array ( 5 , 6 , 7 , 1 , 2 , 3 , 4 ) ; $ n6 = sizeof ( $ arr6 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr6 , 0 , $ n6 - 1 ) . " STRNEWLINE " ; $ arr7 = array ( 1 , 2 , 3 , 4 , 5 , 6 , 7 ) ; $ n7 = sizeof ( $ arr7 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr7 , 0 , $ n7 - 1 ) . " STRNEWLINE " ; $ arr8 = array ( 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 ) ; $ n8 = sizeof ( $ arr8 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr8 , 0 , $ n8 - 1 ) . " STRNEWLINE " ; $ arr9 = array ( 3 , 4 , 5 , 1 , 2 ) ; $ n9 = sizeof ( $ arr9 ) ; echo " The ▁ minimum ▁ element ▁ is ▁ " . findMin ( $ arr9 , 0 , $ n9 - 1 ) . " STRNEWLINE " ; ? >
< ? php function print2Smallest ( $ arr , $ arr_size ) { $ INT_MAX = 2147483647 ;
if ( $ arr_size < 2 ) { echo ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } $ first = $ second = $ INT_MAX ; for ( $ i = 0 ; $ i < $ arr_size ; $ i ++ ) {
if ( $ arr [ $ i ] < $ first ) { $ second = $ first ; $ first = $ arr [ $ i ] ; }
else if ( $ arr [ $ i ] < $ second && $ arr [ $ i ] != $ first ) $ second = $ arr [ $ i ] ; } if ( $ second == $ INT_MAX ) echo ( " There ▁ is ▁ no ▁ second ▁ smallest ▁ element STRNEWLINE " ) ; else echo " The ▁ smallest ▁ element ▁ is ▁ " , $ first , " ▁ and ▁ second ▁ Smallest ▁ element ▁ is ▁ " , $ second ; }
$ arr = array ( 12 , 13 , 1 , 10 , 34 , 1 ) ; $ n = count ( $ arr ) ; print2Smallest ( $ arr , $ n ) ? >
< ? php function isSubsetSum ( $ arr , $ n , $ sum ) {
$ subset [ 2 ] [ $ sum + 1 ] = array ( ) ; for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) { for ( $ j = 0 ; $ j <= $ sum ; $ j ++ ) {
if ( $ j == 0 ) $ subset [ $ i % 2 ] [ $ j ] = true ;
else if ( $ i == 0 ) $ subset [ $ i % 2 ] [ $ j ] = false ; else if ( $ arr [ $ i - 1 ] <= $ j ) $ subset [ $ i % 2 ] [ $ j ] = $ subset [ ( $ i + 1 ) % 2 ] [ $ j - $ arr [ $ i - 1 ] ] || $ subset [ ( $ i + 1 ) % 2 ] [ $ j ] ; else $ subset [ $ i % 2 ] [ $ j ] = $ subset [ ( $ i + 1 ) % 2 ] [ $ j ] ; } } return $ subset [ $ n % 2 ] [ $ sum ] ; }
$ arr = array ( 6 , 2 , 5 ) ; $ sum = 7 ; $ n = sizeof ( $ arr ) ; if ( isSubsetSum ( $ arr , $ n , $ sum ) == true ) echo ( " There ▁ exists ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else echo ( " No ▁ subset ▁ exists ▁ with ▁ given ▁ sum " ) ; ? >
< ? php function findMaxSum ( $ arr , $ n ) { $ res = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ prefix_sum = $ arr [ $ i ] ; for ( $ j = 0 ; $ j < $ i ; $ j ++ ) $ prefix_sum += $ arr [ $ j ] ; $ suffix_sum = $ arr [ $ i ] ; for ( $ j = $ n - 1 ; $ j > $ i ; $ j -- ) $ suffix_sum += $ arr [ $ j ] ; if ( $ prefix_sum == $ suffix_sum ) $ res = max ( $ res , $ prefix_sum ) ; } return $ res ; }
$ arr = array ( -2 , 5 , 3 , 1 , 2 , 6 , -4 , 2 ) ; $ n = count ( $ arr ) ; echo findMaxSum ( $ arr , $ n ) ; ? >
< ? php function findMaxSum ( $ arr , $ n ) {
$ preSum [ $ n ] = array ( ) ;
$ suffSum [ $ n ] = array ( ) ;
$ ans = PHP_INT_MIN ;
$ preSum [ 0 ] = $ arr [ 0 ] ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ preSum [ $ i ] = $ preSum [ $ i - 1 ] + $ arr [ $ i ] ;
$ suffSum [ $ n - 1 ] = $ arr [ $ n - 1 ] ; if ( $ preSum [ $ n - 1 ] == $ suffSum [ $ n - 1 ] ) $ ans = max ( $ ans , $ preSum [ $ n - 1 ] ) ; for ( $ i = $ n - 2 ; $ i >= 0 ; $ i -- ) { $ suffSum [ $ i ] = $ suffSum [ $ i + 1 ] + $ arr [ $ i ] ; if ( $ suffSum [ $ i ] == $ preSum [ $ i ] ) $ ans = max ( $ ans , $ preSum [ $ i ] ) ; } return $ ans ; }
$ arr = array ( -2 , 5 , 3 , 1 , 2 , 6 , -4 , 2 ) ; $ n = sizeof ( $ arr ) ; echo findMaxSum ( $ arr , $ n ) ;
< ? php function findMajority ( $ arr , $ n ) { $ maxCount = 0 ;
$ index = -1 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ count = 0 ; for ( $ j = 0 ; $ j < $ n ; $ j ++ ) { if ( $ arr [ $ i ] == $ arr [ $ j ] ) $ count ++ ; }
if ( $ count > $ maxCount ) { $ maxCount = $ count ; $ index = $ i ; } }
if ( $ maxCount > $ n / 2 ) echo $ arr [ $ index ] . " STRNEWLINE " ; else echo " No ▁ Majority ▁ Element " . " STRNEWLINE " ; }
$ arr = array ( 1 , 1 , 2 , 1 , 3 , 5 , 1 ) ; $ n = sizeof ( $ arr ) ;
findMajority ( $ arr , $ n ) ;
< ? php function findCandidate ( $ a , $ size ) { $ maj_index = 0 ; $ count = 1 ; for ( $ i = 1 ; $ i < $ size ; $ i ++ ) { if ( $ a [ $ maj_index ] == $ a [ $ i ] ) $ count ++ ; else $ count -- ; if ( $ count == 0 ) { $ maj_index = $ i ; $ count = 1 ; } } return $ a [ $ maj_index ] ; }
function isMajority ( $ a , $ size , $ cand ) { $ count = 0 ; for ( $ i = 0 ; $ i < $ size ; $ i ++ ) if ( $ a [ $ i ] == $ cand ) $ count ++ ; if ( $ count > $ size / 2 ) return 1 ; else return 0 ; }
function printMajority ( $ a , $ size ) {
$ cand = findCandidate ( $ a , $ size ) ;
if ( isMajority ( $ a , $ size , $ cand ) ) echo " " , ▁ $ cand , ▁ " " else echo " No ▁ Majority ▁ Element " ; }
$ a = array ( 1 , 3 , 3 , 1 , 2 ) ; $ size = sizeof ( $ a ) ;
printMajority ( $ a , $ size ) ; ? >
< ? php function isSubsetSum ( $ set , $ n , $ sum ) {
$ subset = array ( array ( ) ) ;
for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) $ subset [ $ i ] [ 0 ] = true ;
for ( $ i = 1 ; $ i <= $ sum ; $ i ++ ) $ subset [ 0 ] [ $ i ] = false ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { for ( $ j = 1 ; $ j <= $ sum ; $ j ++ ) { if ( $ j < $ set [ $ i - 1 ] ) $ subset [ $ i ] [ $ j ] = $ subset [ $ i - 1 ] [ $ j ] ; if ( $ j >= $ set [ $ i - 1 ] ) $ subset [ $ i ] [ $ j ] = $ subset [ $ i - 1 ] [ $ j ] || $ subset [ $ i - 1 ] [ $ j - $ set [ $ i - 1 ] ] ; } }
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) printf ( " % 4d " , subset [ i ] [ j ] ) ; printf ( " n " ) ; } return $ subset [ $ n ] [ $ sum ] ; }
$ set = array ( 3 , 34 , 4 , 12 , 5 , 2 ) ; $ sum = 9 ; $ n = count ( $ set ) ; if ( isSubsetSum ( $ set , $ n , $ sum ) == true ) echo " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else echo " No ▁ subset ▁ with ▁ given ▁ sum " ; ? >
function print_gcd_online ( $ n , $ m , $ query , $ arr ) {
$ max_gcd = 0 ; $ i = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ max_gcd = gcd ( $ max_gcd , $ arr [ $ i ] ) ;
for ( $ i = 0 ; $ i < $ m ; $ i ++ ) {
$ query [ $ i ] [ 0 ] -- ;
$ arr [ $ query [ $ i ] [ 0 ] ] /= $ query [ $ i ] [ 1 ] ;
$ max_gcd = gcd ( $ arr [ $ query [ $ i ] [ 0 ] ] , $ max_gcd ) ;
echo ( $ max_gcd ) , " STRNEWLINE " ; } }
$ n = 3 ; $ m = 3 ; $ query ; $ arr = array ( 36 , 24 , 72 ) ; $ query [ 0 ] [ 0 ] = 1 ; $ query [ 0 ] [ 1 ] = 3 ; $ query [ 1 ] [ 0 ] = 3 ; $ query [ 1 ] [ 1 ] = 12 ; $ query [ 2 ] [ 0 ] = 2 ; $ query [ 2 ] [ 1 ] = 4 ; print_gcd_online ( $ n , $ m , $ query , $ arr ) ; ? >
< ? php $ MAX = 100000 ;
$ prime = array_fill ( 0 , $ MAX + 1 , true ) ;
$ sum = array_fill ( 0 , $ MAX + 1 , 0 ) ;
function SieveOfEratosthenes ( ) { global $ MAX , $ sum , $ prime
; $ prime [ 1 ] = false ; for ( $ p = 2 ; $ p * $ p <= $ MAX ; $ p ++ ) {
if ( $ prime [ $ p ] ) {
for ( $ i = $ p * 2 ; $ i <= $ MAX ; $ i += $ p ) $ prime [ $ i ] = false ; } }
for ( $ i = 1 ; $ i <= $ MAX ; $ i ++ ) { if ( $ prime [ $ i ] == true ) $ sum [ $ i ] = 1 ; $ sum [ $ i ] += $ sum [ $ i - 1 ] ; } }
SieveOfEratosthenes ( ) ;
$ l = 3 ; $ r = 9 ;
$ c = ( $ sum [ $ r ] - $ sum [ $ l - 1 ] ) ;
echo " Count : " ▁ . ▁ $ c ▁ . ▁ " " ? >
< ? php function area ( $ r ) {
if ( $ r < 0 ) return -1 ;
$ area = 3.14 * pow ( $ r / ( 2 * sqrt ( 2 ) ) , 2 ) ; return $ area ; }
$ a = 5 ; echo area ( $ a ) ;
< ? php $ N = 100005 ;
$ prime = array_fill ( 0 , $ N , true ) ; function SieveOfEratosthenes ( ) { global $ N , $ prime ; $ prime [ 1 ] = false ; for ( $ p = 2 ; $ p < ( int ) ( sqrt ( $ N ) ) ; $ p ++ ) {
if ( $ prime [ $ p ] == true )
for ( $ i = 2 * $ p ; $ i < $ N ; $ i += $ p ) $ prime [ $ i ] = false ; } }
function almostPrimes ( $ n ) { global $ prime ;
$ ans = 0 ;
for ( $ i = 6 ; $ i < $ n + 1 ; $ i ++ ) {
$ c = 0 ; for ( $ j = 2 ; $ i >= $ j * $ j ; $ j ++ ) {
if ( $ i % $ j == 0 ) { if ( $ j * $ j == $ i ) { if ( $ prime [ $ j ] ) $ c += 1 ; } else { if ( $ prime [ $ j ] ) $ c += 1 ; if ( $ prime [ ( $ i / $ j ) ] ) $ c += 1 ; } } }
if ( $ c == 2 ) $ ans += 1 ; } return $ ans ; }
SieveOfEratosthenes ( ) ; $ n = 21 ; print ( almostPrimes ( $ n ) ) ; ? >
< ? php function sumOfDigitsSingle ( $ x ) { $ ans = 0 ; while ( $ x ) { $ ans += $ x % 10 ; $ x /= 10 ; } return $ ans ; }
function closest ( $ x ) { $ ans = 0 ; while ( $ ans * 10 + 9 <= $ x ) $ ans = $ ans * 10 + 9 ; return $ ans ; } function sumOfDigitsTwoParts ( $ N ) { $ A = closest ( $ N ) ; return sumOfDigitsSingle ( $ A ) + sumOfDigitsSingle ( $ N - $ A ) ; }
$ N = 35 ; echo sumOfDigitsTwoParts ( $ N ) ; ? >
< ? php function isPrime ( $ p ) {
$ checkNumber = pow ( 2 , $ p ) - 1 ;
$ nextval = 4 % $ checkNumber ;
for ( $ i = 1 ; $ i < $ p - 1 ; $ i ++ ) $ nextval = ( $ nextval * $ nextval - 2 ) % $ checkNumber ;
return ( $ nextval == 0 ) ; }
$ p = 7 ; $ checkNumber = pow ( 2 , $ p ) - 1 ; if ( isPrime ( $ p ) ) echo $ checkNumber , " ▁ is ▁ Prime . " ; else echo $ checkNumber , " ▁ is ▁ not ▁ Prime . " ; ? >
< ? php function sieve ( $ n , & $ prime ) { for ( $ p = 2 ; $ p * $ p <= $ n ; $ p ++ ) {
if ( $ prime [ $ p ] == true ) {
for ( $ i = $ p * 2 ; $ i <= $ n ; $ i += $ p ) $ prime [ $ i ] = false ; } } } function printSophieGermanNumber ( $ n ) {
$ prime = array ( ) ; for ( $ i = 0 ; $ i < ( 2 * $ n + 1 ) ; $ i ++ ) $ prime [ $ i ] = true ; sieve ( 2 * $ n + 1 , $ prime ) ; for ( $ i = 2 ; $ i <= $ n ; ++ $ i ) {
if ( $ prime [ $ i ] && $ prime [ 2 * $ i + 1 ] ) echo ( $ i . " ▁ " ) ; } }
$ n = 25 ; printSophieGermanNumber ( $ n ) ; ? >
< ? php function ucal ( $ u , $ n ) { if ( $ n == 0 ) return 1 ; $ temp = $ u ; for ( $ i = 1 ; $ i <= ( int ) ( $ n / 2 ) ; $ i ++ ) $ temp = $ temp * ( $ u - $ i ) ; for ( $ i = 1 ; $ i < ( int ) ( $ n / 2 ) ; $ i ++ ) $ temp = $ temp * ( $ u + $ i ) ; return $ temp ; }
function fact ( $ n ) { $ f = 1 ; for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) $ f *= $ i ; return $ f ; }
$ n = 6 ; $ x = array ( 25 , 26 , 27 , 28 , 29 , 30 ) ;
$ y ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) for ( $ j = 0 ; $ j < $ n ; $ j ++ ) $ y [ $ i ] [ $ j ] = 0.0 ; $ y [ 0 ] [ 0 ] = 4.000 ; $ y [ 1 ] [ 0 ] = 3.846 ; $ y [ 2 ] [ 0 ] = 3.704 ; $ y [ 3 ] [ 0 ] = 3.571 ; $ y [ 4 ] [ 0 ] = 3.448 ; $ y [ 5 ] [ 0 ] = 3.333 ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) for ( $ j = 0 ; $ j < $ n - $ i ; $ j ++ ) $ y [ $ j ] [ $ i ] = $ y [ $ j + 1 ] [ $ i - 1 ] - $ y [ $ j ] [ $ i - 1 ] ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = 0 ; $ j < $ n - $ i ; $ j ++ ) echo str_pad ( $ y [ $ i ] [ $ j ] , 4 ) . " TABSYMBOL " ; echo " STRNEWLINE " ; }
$ value = 27.4 ;
$ sum = ( $ y [ 2 ] [ 0 ] + $ y [ 3 ] [ 0 ] ) / 2 ;
$ k ;
$ k = $ n / 2 ; else
$ u = ( $ value - $ x [ $ k ] ) / ( $ x [ 1 ] - $ x [ 0 ] ) ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { if ( $ i % 2 ) $ sum = $ sum + ( ( $ u - 0.5 ) * ucal ( $ u , $ i - 1 ) * $ y [ $ k ] [ $ i ] ) / fact ( $ i ) ; else $ sum = $ sum + ( ucal ( $ u , $ i ) * ( $ y [ $ k ] [ $ i ] + $ y [ -- $ k ] [ $ i ] ) / ( fact ( $ i ) * 2 ) ) ; } echo " Value ▁ at ▁ " . $ value . " ▁ is ▁ " . $ sum . " STRNEWLINE " ; ? >
< ? php function fibonacci ( $ n ) { $ a = 0 ; $ b = 1 ; $ c ; if ( $ n <= 1 ) return $ n ; for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) { $ c = $ a + $ b ; $ a = $ b ; $ b = $ c ; } return $ c ; }
function isMultipleOf10 ( $ n ) { $ f = fibonacci ( 30 ) ; return ( $ f % 10 == 0 ) ; }
$ n = 30 ; if ( isMultipleOf10 ( $ n ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; ? >
< ? php function isPowerOfTwo ( $ x ) {
return $ x && ( ! ( $ x & ( $ x - 1 ) ) ) ; }
if ( isPowerOfTwo ( 31 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; if ( isPowerOfTwo ( 64 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; ? >
< ? php function nextPowerOf2 ( $ n ) {
$ p = 1 ;
if ( $ n && ! ( $ n & ( $ n - 1 ) ) ) return $ n ;
while ( $ p < $ n ) $ p <<= 1 ; return $ p ; }
function memoryUsed ( & $ arr , $ n ) {
$ sum = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ sum += $ arr [ $ i ] ;
$ nearest = nextPowerOf2 ( $ sum ) ; return $ nearest ; }
$ arr = array ( 1 , 2 , 3 , 2 ) ; $ n = sizeof ( $ arr ) ; echo ( memoryUsed ( $ arr , $ n ) ) ; ? >
< ? php function toggleKthBit ( $ n , $ k ) { return ( $ n ^ ( 1 << ( $ k - 1 ) ) ) ; }
$ n = 5 ; $ k = 1 ; echo toggleKthBit ( $ n , $ k ) ; ? >
< ? php function nextPowerOf2 ( $ n ) { $ count = 0 ;
if ( $ n && ! ( $ n & ( $ n - 1 ) ) ) return $ n ; while ( $ n != 0 ) { $ n >>= 1 ; $ count += 1 ; } return 1 << $ count ; }
$ n = 0 ; echo ( nextPowerOf2 ( $ n ) ) ; ? >
< ? php function printTetra ( $ n ) { if ( $ n < 0 ) return ;
$ first = 0 ; $ second = 1 ; $ third = 1 ; $ fourth = 2 ;
$ curr ; if ( $ n == 0 ) echo $ first ; else if ( $ n == 1 $ n == 2 ) echo $ second ; else if ( $ n == 3 ) echo $ fourth ; else {
for ( $ i = 4 ; $ i <= $ n ; $ i ++ ) { $ curr = $ first + $ second + $ third + $ fourth ; $ first = $ second ; $ second = $ third ; $ third = $ fourth ; $ fourth = $ curr ; } echo $ curr ; } }
$ n = 10 ; printTetra ( $ n ) ; ? >
< ? php function countWays ( $ n ) { $ res [ 0 ] = 1 ; $ res [ 1 ] = 1 ; $ res [ 2 ] = 2 ; for ( $ i = 3 ; $ i <= $ n ; $ i ++ ) $ res [ $ i ] = $ res [ $ i - 1 ] + $ res [ $ i - 2 ] + $ res [ $ i - 3 ] ; return $ res [ $ n ] ; }
$ n = 4 ; echo countWays ( $ n ) ; ? >
< ? php function maxTasks ( $ high , $ low , $ n ) {
if ( $ n <= 0 ) return 0 ;
return max ( $ high [ $ n - 1 ] + maxTasks ( $ high , $ low , ( $ n - 2 ) ) , $ low [ $ n - 1 ] + maxTasks ( $ high , $ low , ( $ n - 1 ) ) ) ; }
$ n = 5 ; $ high = array ( 3 , 6 , 8 , 7 , 6 ) ; $ low = array ( 1 , 5 , 4 , 5 , 3 ) ; print ( maxTasks ( $ high , $ low , $ n ) ) ; ? >
< ? php function countSubstr ( $ str , $ n , $ x , $ y ) {
$ tot_count = 0 ;
$ count_x = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ str [ $ i ] == $ x ) $ count_x ++ ;
if ( $ str [ $ i ] == $ y ) $ tot_count += $ count_x ; }
return $ tot_count ; }
$ str = " abbcaceghcak " ; $ n = strlen ( $ str ) ; $ x = ' a ' ; $ y = ' c ' ; echo " Count = "
< ? php $ OUT = 0 ; $ IN = 1 ;
function countWords ( $ str ) { global $ OUT , $ IN ; $ state = $ OUT ;
$ wc = 0 ; $ i = 0 ;
while ( $ i < strlen ( $ str ) ) {
if ( $ str [ $ i ] == " ▁ " $ str [ $ i ] == " STRNEWLINE " $ str [ $ i ] == " TABSYMBOL " ) $ state = $ OUT ;
else if ( $ state == $ OUT ) { $ state = $ IN ; ++ $ wc ; }
++ $ i ; } return $ wc ; }
$ str = " One ▁ twothree STRNEWLINE ▁ four TABSYMBOL five ▁ " ; echo " No ▁ of ▁ words ▁ : ▁ " . countWords ( $ str ) ; ? >
< ? php function nthEnneadecagonal ( $ n ) {
return ( 17 * $ n * $ n - 15 * $ n ) / 2 ; }
$ n = 6 ; echo $ n , " th ▁ Enneadecagonal ▁ number ▁ : " , nthEnneadecagonal ( $ n ) ; ? >
< ? php $ PI = 3.14159265 ;
function areacircumscribed ( $ a ) { global $ PI ; return ( $ a * $ a * ( $ PI / 2 ) ) ; }
$ a = 6 ; echo " ▁ Area ▁ of ▁ an ▁ circumscribed ▁ circle ▁ is ▁ : ▁ " , areacircumscribed ( $ a ) ; ? >
< ? php function printTetraRec ( $ n ) {
if ( $ n == 0 ) return 0 ;
if ( $ n == 1 $ n == 2 ) return 1 ;
if ( $ n == 3 ) return 2 ; else return printTetraRec ( $ n - 1 ) + printTetraRec ( $ n - 2 ) + printTetraRec ( $ n - 3 ) + printTetraRec ( $ n - 4 ) ; }
function printTetra ( $ n ) { echo printTetraRec ( $ n ) . " " ; }
$ n = 10 ; printTetra ( $ n ) ; ? >
< ? php function max1 ( $ x , $ y ) { return ( $ x > $ y ? $ x : $ y ) ; }
return ( $ x > $ y ? $ x : $ y ) ; }
function maxTasks ( $ high , $ low , $ n ) {
$ task_dp = array ( $ n + 1 ) ;
$ task_dp [ 0 ] = 0 ;
$ task_dp [ 1 ] = $ high [ 0 ] ;
for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) $ task_dp [ $ i ] = max ( $ high [ $ i - 1 ] + $ task_dp [ $ i - 2 ] , $ low [ $ i - 1 ] + $ task_dp [ $ i - 1 ] ) ; return $ task_dp [ $ n ] ; }
{ $ n = 5 ; $ high = array ( 3 , 6 , 8 , 7 , 6 ) ; $ low = array ( 1 , 5 , 4 , 5 , 3 ) ; echo ( maxTasks ( $ high , $ low , $ n ) ) ; }
< ? php function PermutationCoeff ( $ n , $ k ) { $ Fn = 1 ; $ Fk ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { $ Fn *= $ i ; if ( $ i == $ n - $ k ) $ Fk = $ Fn ; } $ coeff = $ Fn / $ Fk ; return $ coeff ; }
$ n = 10 ; $ k = 2 ; echo " Value ▁ of ▁ P ( " , $ n , " , ▁ " , $ k , " ) STRNEWLINE is ▁ " , PermutationCoeff ( $ n , $ k ) ; ? >
< ? php $ dfa = 0 ;
function start ( $ c ) { global $ dfa ;
if ( $ c == ' t ' $ c == ' T ' ) $ dfa = 1 ; }
function state1 ( $ c ) { global $ dfa ;
if ( $ c == ' t ' $ c == ' T ' ) $ dfa = 1 ;
else if ( $ c == ' h ' $ c == ' H ' ) $ dfa = 2 ;
else $ dfa = 0 ; }
function state2 ( $ c ) { global $ dfa ;
if ( $ c == ' e ' $ c == ' E ' ) $ dfa = 3 ; else $ dfa = 0 ; }
function state3 ( $ c ) { global $ dfa ;
if ( $ c == ' t ' $ c == ' T ' ) $ dfa = 1 ; else $ dfa = 0 ; } function isAccepted ( $ str ) { global $ dfa ;
$ len = strlen ( $ str ) ; for ( $ i = 0 ; $ i < $ len ; $ i ++ ) { if ( $ dfa == 0 ) start ( $ str [ $ i ] ) ; else if ( $ dfa == 1 ) state1 ( $ str [ $ i ] ) ; else if ( $ dfa == 2 ) state2 ( $ str [ $ i ] ) ; else state3 ( $ str [ $ i ] ) ; } return ( $ dfa != 3 ) ; }
$ str = " forTHEgeeks " ; if ( isAccepted ( $ str ) == true ) echo " ACCEPTED STRNEWLINE " ; else echo " NOT ▁ ACCEPTED STRNEWLINE " ; ? >
< ? php function startsWith ( $ str , $ pre ) { $ strLen = strlen ( $ str ) ; $ preLen = strlen ( $ pre ) ; $ i = 0 ; $ j = 0 ;
while ( $ i < $ strLen && $ j < $ preLen ) {
if ( $ str [ $ i ] != $ pre [ $ j ] ) return false ; $ i ++ ; $ j ++ ; }
return true ; }
function endsWith ( $ str , $ suff ) { $ i = strlen ( $ str ) - 0 ; $ j = strlen ( $ suff ) - 0 ;
while ( $ i >= 0 && $ j >= 0 ) {
if ( $ str [ $ i ] != $ suff [ $ j ] ) return false ; $ i -- ; $ j -- ; }
return true ; }
function checkString ( $ str , $ a , $ b ) {
if ( strlen ( $ str ) != strlen ( $ a ) + strlen ( $ b ) ) return false ;
if ( startsWith ( $ str , $ a ) ) {
if ( endsWith ( $ str , $ b ) ) return true ; }
if ( startsWith ( $ str , $ b ) ) {
if ( endsWith ( $ str , $ a ) ) return true ; } return false ; }
$ str = " GeeksforGeeks " ; $ a = " Geeksfo " ; $ b = " rGeeks " ; if ( checkString ( $ str , $ a , $ b ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function minOperations ( $ str , $ n ) {
$ i ; $ lastUpper = -1 ; $ firstLower = -1 ;
for ( $ i = $ n - 1 ; $ i >= 0 ; $ i -- ) { if ( ctype_upper ( $ str [ $ i ] ) ) { $ lastUpper = $ i ; break ; } }
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( ctype_lower ( $ str [ $ i ] ) ) { $ firstLower = $ i ; break ; } }
if ( $ lastUpper == -1 $ firstLower == -1 ) return 0 ;
$ countUpper = 0 ; for ( $ i = $ firstLower ; $ i < $ n ; $ i ++ ) { if ( ctype_upper ( $ str [ $ i ] ) ) { $ countUpper ++ ; } }
$ countLower = 0 ; for ( $ i = 0 ; $ i < $ lastUpper ; $ i ++ ) { if ( ctype_lower ( $ str [ $ i ] ) ) { $ countLower ++ ; } }
return min ( $ countLower , $ countUpper ) ; }
{ $ str = " geEksFOrGEekS " ; $ n = strlen ( $ str ) ; echo ( minOperations ( $ str , $ n ) ) ; } ? >
< ? php function rainDayProbability ( $ a , $ n ) { $ count = 0 ; $ m ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ a [ $ i ] == 1 ) $ count ++ ; }
$ m = $ count / $ n ; return $ m ; }
$ a = array ( 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 ) ; $ n = count ( $ a ) ; echo rainDayProbability ( $ a , $ n ) ; ? >
< ? php function Series ( $ n ) { $ i ; $ sums = 0.0 ; $ ser ; for ( $ i = 1 ; $ i <= $ n ; ++ $ i ) { $ ser = 1 / pow ( $ i , $ i ) ; $ sums += $ ser ; } return $ sums ; }
$ n = 3 ; $ res = Series ( $ n ) ; echo $ res ; ? >
< ? php function ternarySearch ( $ l , $ r , $ key , $ ar ) { if ( $ r >= $ l ) {
$ mid1 = ( int ) ( $ l + ( $ r - $ l ) / 3 ) ; $ mid2 = ( int ) ( $ r - ( $ r - $ l ) / 3 ) ;
if ( $ ar [ $ mid1 ] == $ key ) { return $ mid1 ; } if ( $ ar [ $ mid2 ] == $ key ) { return $ mid2 ; }
if ( $ key < $ ar [ $ mid1 ] ) {
return ternarySearch ( $ l , $ mid1 - 1 , $ key , $ ar ) ; } else if ( $ key > $ ar [ $ mid2 ] ) {
return ternarySearch ( $ mid2 + 1 , $ r , $ key , $ ar ) ; } else {
return ternarySearch ( $ mid1 + 1 , $ mid2 - 1 , $ key , $ ar ) ; } }
return -1 ; }
$ ar = array ( 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ) ;
$ l = 0 ;
$ r = 9 ;
$ key = 5 ;
$ p = ternarySearch ( $ l , $ r , $ key , $ ar ) ;
echo " Index ▁ of ▁ " , $ key , " ▁ is ▁ " , ( int ) $ p , " STRNEWLINE " ;
$ key = 50 ;
$ p = ternarySearch ( $ l , $ r , $ key , $ ar ) ;
echo " Index ▁ of ▁ " , $ key , " ▁ is ▁ " , ( int ) $ p , " STRNEWLINE " ; ? >
< ? php $ SIZE = 26 ;
function printCharWithFreq ( $ str ) { global $ SIZE ;
$ n = strlen ( $ str ) ;
$ freq = array_fill ( 0 , $ SIZE , NULL ) ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ freq [ ord ( $ str [ $ i ] ) - ord ( ' a ' ) ] ++ ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ freq [ ord ( $ str [ $ i ] ) - ord ( ' a ' ) ] != 0 ) {
echo $ str [ $ i ] . $ freq [ ord ( $ str [ $ i ] ) - ord ( ' a ' ) ] . " " ;
$ freq [ ord ( $ str [ $ i ] ) - ord ( ' a ' ) ] = 0 ; } } }
$ str = " geeksforgeeks " ; printCharWithFreq ( $ str ) ; ? >
< ? php function checkHV ( $ arr , $ N , $ M ) {
$ horizontal = true ; $ vertical = true ;
for ( $ i = 0 , $ k = $ N - 1 ; $ i < $ N / 2 ; $ i ++ , $ k -- ) {
for ( $ j = 0 ; $ j < $ M ; $ j ++ ) {
if ( $ arr [ $ i ] [ $ j ] != $ arr [ $ k ] [ $ j ] ) { $ horizontal = false ; break ; } } }
for ( $ i = 0 , $ k = $ M - 1 ; $ i < $ M / 2 ; $ i ++ , $ k -- ) {
for ( $ j = 0 ; $ j < $ N ; $ j ++ ) {
if ( $ arr [ $ i ] [ $ j ] != $ arr [ $ k ] [ $ j ] ) { $ horizontal = false ; break ; } } } if ( ! $ horizontal && ! $ vertical ) echo " NO STRNEWLINE " ; else if ( $ horizontal && ! $ vertical ) cout << " HORIZONTAL STRNEWLINE " ; else if ( $ vertical && ! $ horizontal ) echo " VERTICAL STRNEWLINE " ; else echo " BOTH STRNEWLINE " ; }
$ mat = array ( array ( 1 , 0 , 1 ) , array ( 0 , 0 , 0 ) , array ( 1 , 0 , 1 ) ) ; checkHV ( $ mat , 3 , 3 ) ; ? >
< ? php $ N = 4 ;
function add ( & $ A , & $ B , & $ C ) { for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = 0 ; $ j < $ N ; $ j ++ ) $ C [ $ i ] [ $ j ] = $ A [ $ i ] [ $ j ] + $ B [ $ i ] [ $ j ] ; }
$ A = array ( array ( 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 ) , array ( 4 , 4 , 4 , 4 ) ) ; $ B = array ( array ( 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 ) , array ( 4 , 4 , 4 , 4 ) ) ; $ N = 4 ; add ( $ A , $ B , $ C ) ; echo " Result ▁ matrix ▁ is ▁ STRNEWLINE " ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) { for ( $ j = 0 ; $ j < $ N ; $ j ++ ) { echo $ C [ $ i ] [ $ j ] ; echo " ▁ " ; } echo " STRNEWLINE " ; } ? >
< ? php function subtract ( & $ A , & $ B , & $ C ) { $ N = 4 ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) for ( $ j = 0 ; $ j < $ N ; $ j ++ ) $ C [ $ i ] [ $ j ] = $ A [ $ i ] [ $ j ] - $ B [ $ i ] [ $ j ] ; }
$ N = 4 ; $ A = array ( array ( 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 ) , array ( 4 , 4 , 4 , 4 ) ) ; $ B = array ( array ( 1 , 1 , 1 , 1 ) , array ( 2 , 2 , 2 , 2 ) , array ( 3 , 3 , 3 , 3 ) , array ( 4 , 4 , 4 , 4 ) ) ; subtract ( $ A , $ B , $ C ) ; echo " Result ▁ matrix ▁ is ▁ STRNEWLINE " ; for ( $ i = 0 ; $ i < $ N ; $ i ++ ) { for ( $ j = 0 ; $ j < $ N ; $ j ++ ) { echo $ C [ $ i ] [ $ j ] ; echo " ▁ " ; } echo " STRNEWLINE " ; } ? >
< ? php function linearSearch ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ arr [ $ i ] == $ i ) return $ i ; }
return -1 ; }
$ arr = array ( -10 , -1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ) ; $ n = count ( $ arr ) ; echo " Fixed ▁ Point ▁ is ▁ " . linearSearch ( $ arr , $ n ) ; ? >
< ? php function binarySearch ( $ arr , $ low , $ high ) { if ( $ high >= $ low ) {
$ mid = ( int ) ( ( $ low + $ high ) / 2 ) ; if ( $ mid == $ arr [ $ mid ] ) return $ mid ; if ( $ mid > $ arr [ $ mid ] ) return binarySearch ( $ arr , ( $ mid + 1 ) , $ high ) ; else return binarySearch ( $ arr , $ low , ( $ mid - 1 ) ) ; }
return -1 ; }
$ arr = array ( -10 , -1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ) ; $ n = count ( $ arr ) ; echo " Fixed ▁ Point ▁ is : ▁ " . binarySearch ( $ arr , 0 , $ n - 1 ) ; ? >
< ? php function maxTripletSum ( $ arr , $ n ) {
$ sum = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) for ( $ k = $ j + 1 ; $ k < $ n ; $ k ++ ) if ( $ sum < $ arr [ $ i ] + $ arr [ $ j ] + $ arr [ $ k ] ) $ sum = $ arr [ $ i ] + $ arr [ $ j ] + $ arr [ $ k ] ; return $ sum ; }
$ arr = array ( 1 , 0 , 8 , 6 , 4 , 2 ) ; $ n = count ( $ arr ) ; echo maxTripletSum ( $ arr , $ n ) ; ? >
< ? php function maxTripletSum ( $ arr , $ n ) {
sort ( $ arr ) ;
return $ arr [ $ n - 1 ] + $ arr [ $ n - 2 ] + $ arr [ $ n - 3 ] ; }
$ arr = array ( 1 , 0 , 8 , 6 , 4 , 2 ) ; $ n = count ( $ arr ) ; echo maxTripletSum ( $ arr , $ n ) ; ? >
< ? php function maxTripletSum ( $ arr , $ n ) {
$ maxA = PHP_INT_MIN ; $ maxB = PHP_INT_MIN ; $ maxC = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ arr [ $ i ] > $ maxA ) { $ maxC = $ maxB ; $ maxB = $ maxA ; $ maxA = $ arr [ $ i ] ; }
else if ( $ arr [ $ i ] > $ maxB ) { $ maxC = $ maxB ; $ maxB = $ arr [ $ i ] ; }
else if ( $ arr [ $ i ] > $ maxC ) $ maxC = $ arr [ $ i ] ; } return ( $ maxA + $ maxB + $ maxC ) ; }
$ arr = array ( 1 , 0 , 8 , 6 , 4 , 2 ) ; $ n = count ( $ arr ) ; echo maxTripletSum ( $ arr , $ n ) ; ? >
< ? php function search ( $ arr , $ x ) { $ n = sizeof ( $ arr ) ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ arr [ $ i ] == $ x ) return $ i ; } return -1 ; }
$ arr = array ( 2 , 3 , 4 , 10 , 40 ) ; $ x = 10 ;
$ result = search ( $ arr , $ x ) ; if ( $ result == -1 ) echo " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ; else echo " Element ▁ is ▁ present ▁ at ▁ index ▁ " , $ result ; ? >
< ? php $ RANGE = 255 ;
function countSort ( $ arr ) { global $ RANGE ;
$ output = array ( strlen ( $ arr ) ) ; $ len = strlen ( $ arr ) ;
$ count = array_fill ( 0 , $ RANGE + 1 , 0 ) ;
for ( $ i = 0 ; $ i < $ len ; ++ $ i ) ++ $ count [ ord ( $ arr [ $ i ] ) ] ;
for ( $ i = 1 ; $ i <= $ RANGE ; ++ $ i ) $ count [ $ i ] += $ count [ $ i - 1 ] ;
for ( $ i = $ len - 1 ; $ i >= 0 ; $ i -- ) { $ output [ $ count [ ord ( $ arr [ $ i ] ) ] - 1 ] = $ arr [ $ i ] ; -- $ count [ ord ( $ arr [ $ i ] ) ] ; }
for ( $ i = 0 ; $ i < $ len ; ++ $ i ) $ arr [ $ i ] = $ output [ $ i ] ; return $ arr ; }
$ arr = " geeksforgeeks " ; $ arr = countSort ( $ arr ) ; echo " Sorted ▁ character ▁ array ▁ is ▁ " . $ arr ; ? >
< ? php function binomialCoeff ( $ n , $ k ) {
if ( $ k > $ n ) return 0 ; if ( $ k == 0 $ k == $ n ) return 1 ;
return binomialCoeff ( $ n - 1 , $ k - 1 ) + binomialCoeff ( $ n - 1 , $ k ) ; }
$ n = 5 ; $ k = 2 ; echo " Value ▁ of ▁ C " , " ( " , $ n , $ k , " ) ▁ is ▁ " , binomialCoeff ( $ n , $ k ) ; ? >
< ? php function binomialCoeff ( $ n , $ k ) { $ C = array_fill ( 0 , $ k + 1 , 0 ) ;
$ C [ 0 ] = 1 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) {
for ( $ j = min ( $ i , $ k ) ; $ j > 0 ; $ j -- ) $ C [ $ j ] = $ C [ $ j ] + $ C [ $ j - 1 ] ; } return $ C [ $ k ] ; }
$ n = 5 ; $ k = 2 ; echo " Value ▁ of ▁ C [ $ n , ▁ $ k ] ▁ is ▁ " . binomialCoeff ( $ n , $ k ) ; ? >
< ? php function isSubsetSum ( $ set , $ n , $ sum ) {
if ( $ sum == 0 ) return true ; if ( $ n == 0 ) return false ;
if ( $ set [ $ n - 1 ] > $ sum ) return isSubsetSum ( $ set , $ n - 1 , $ sum ) ;
return isSubsetSum ( $ set , $ n - 1 , $ sum ) || isSubsetSum ( $ set , $ n - 1 , $ sum - $ set [ $ n - 1 ] ) ; }
$ set = array ( 3 , 34 , 4 , 12 , 5 , 2 ) ; $ sum = 9 ; $ n = 6 ; if ( isSubsetSum ( $ set , $ n , $ sum ) == true ) echo " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else echo " No ▁ subset ▁ with ▁ given ▁ sum " ; ? >
< ? php function isSubsetSum ( $ set , $ n , $ sum ) {
$ subset = array ( array ( ) ) ;
for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) $ subset [ $ i ] [ 0 ] = true ;
for ( $ i = 1 ; $ i <= $ sum ; $ i ++ ) $ subset [ 0 ] [ $ i ] = false ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { for ( $ j = 1 ; $ j <= $ sum ; $ j ++ ) { if ( $ j < $ set [ $ i - 1 ] ) $ subset [ $ i ] [ $ j ] = $ subset [ $ i - 1 ] [ $ j ] ; if ( $ j >= $ set [ $ i - 1 ] ) $ subset [ $ i ] [ $ j ] = $ subset [ $ i - 1 ] [ $ j ] || $ subset [ $ i - 1 ] [ $ j - $ set [ $ i - 1 ] ] ; } }
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) printf ( " % 4d " , subset [ i ] [ j ] ) ; printf ( " n " ) ; } return $ subset [ $ n ] [ $ sum ] ; }
$ set = array ( 3 , 34 , 4 , 12 , 5 , 2 ) ; $ sum = 9 ; $ n = count ( $ set ) ; if ( isSubsetSum ( $ set , $ n , $ sum ) == true ) echo " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else echo " No ▁ subset ▁ with ▁ given ▁ sum " ; ? >
< ? php function findoptimal ( $ N ) {
if ( $ N <= 6 ) return $ N ;
$ max = 0 ;
$ b ; for ( $ b = $ N - 3 ; $ b >= 1 ; $ b -= 1 ) {
$ curr = ( $ N - $ b - 1 ) * findoptimal ( $ b ) ; if ( $ curr > $ max ) $ max = $ curr ; } return $ max ; }
$ N ;
for ( $ N = 1 ; $ N <= 20 ; $ N += 1 ) echo ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with " . $ N . " keystrokes ▁ is ▁ " . findoptimal ( $ N ) . " STRNEWLINE " ) ; ? >
< ? php function power ( $ x , $ y ) { if ( $ y == 0 ) return 1 ; else if ( $ y % 2 == 0 ) return power ( $ x , ( int ) $ y / 2 ) * power ( $ x , ( int ) $ y / 2 ) ; else return $ x * power ( $ x , ( int ) $ y / 2 ) * power ( $ x , ( int ) $ y / 2 ) ; }
$ x = 2 ; $ y = 3 ; echo power ( $ x , $ y ) ; ? >
< ? php function power ( $ x , $ y ) { $ temp ; if ( $ y == 0 ) return 1 ; $ temp = power ( $ x , $ y / 2 ) ; if ( $ y % 2 == 0 ) return $ temp * $ temp ; else { if ( $ y > 0 ) return $ x * $ temp * $ temp ; else return ( $ temp * $ temp ) / $ x ; } }
$ x = 2 ; $ y = -3 ; echo power ( $ x , $ y ) ; ? >
< ? php function squareRoot ( $ n ) {
$ x = $ n ; $ y = 1 ;
$ e = 0.000001 ; while ( $ x - $ y > $ e ) { $ x = ( $ x + $ y ) / 2 ; $ y = $ n / $ x ; } return $ x ; }
{ $ n = 50 ; echo " Square ▁ root ▁ of ▁ $ n ▁ is ▁ " , squareRoot ( $ n ) ; } ? >
< ? php function getAvg ( $ prev_avg , $ x , $ n ) { return ( $ prev_avg * $ n + $ x ) / ( $ n + 1 ) ; }
function streamAvg ( $ arr , $ n ) { $ avg = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ avg = getAvg ( $ avg , $ arr [ $ i ] , $ i ) ; echo " Average ▁ of ▁ " , $ i + 1 , " numbers ▁ is ▁ " , $ avg , " STRNEWLINE " ; } return ; }
$ arr = array ( 10 , 20 , 30 , 40 , 50 , 60 ) ; $ n = sizeof ( $ arr ) ; streamAvg ( $ arr , $ n ) ; ? >
< ? php function getAvg ( $ x ) { static $ sum ; static $ n ; $ sum += $ x ; return ( ( ( float ) $ sum ) / ++ $ n ) ; }
function streamAvg ( $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ avg = getAvg ( $ arr [ $ i ] ) ; echo " Average ▁ of ▁ " . ( $ i + 1 ) . " ▁ numbers ▁ is ▁ " . $ avg . " ▁ STRNEWLINE " ; } return ; }
$ arr = array ( 10 , 20 , 30 , 40 , 50 , 60 ) ; $ n = sizeof ( $ arr ) / sizeof ( $ arr [ 0 ] ) ; streamAvg ( $ arr , $ n ) ; ? >
< ? php function binomialCoeff ( $ n , $ k ) { $ res = 1 ;
if ( $ k > $ n - $ k ) $ k = $ n - $ k ;
for ( $ i = 0 ; $ i < $ k ; ++ $ i ) { $ res *= ( $ n - $ i ) ; $ res /= ( $ i + 1 ) ; } return $ res ; }
$ n = 8 ; $ k = 2 ; echo " ▁ Value ▁ of ▁ C ▁ ( $ n , ▁ $ k ) ▁ is ▁ " , binomialCoeff ( $ n , $ k ) ; ? >
< ? php function primeFactors ( $ n ) {
while ( $ n % 2 == 0 ) { echo 2 , " ▁ " ; $ n = $ n / 2 ; }
for ( $ i = 3 ; $ i <= sqrt ( $ n ) ; $ i = $ i + 2 ) {
while ( $ n % $ i == 0 ) { echo $ i , " " ; $ n = $ n / $ i ; } }
if ( $ n > 2 ) echo $ n , " ▁ " ; }
$ n = 315 ; primeFactors ( $ n ) ; ? >
< ? php function printCombination ( $ arr , $ n , $ r ) {
$ data = array ( ) ;
combinationUtil ( $ arr , $ data , 0 , $ n - 1 , 0 , $ r ) ; }
function combinationUtil ( $ arr , $ data , $ start , $ end , $ index , $ r ) {
if ( $ index == $ r ) { for ( $ j = 0 ; $ j < $ r ; $ j ++ ) echo $ data [ $ j ] ; echo " STRNEWLINE " ; return ; }
for ( $ i = $ start ; $ i <= $ end && $ end - $ i + 1 >= $ r - $ index ; $ i ++ ) { $ data [ $ index ] = $ arr [ $ i ] ; combinationUtil ( $ arr , $ data , $ i + 1 , $ end , $ index + 1 , $ r ) ; } }
$ arr = array ( 1 , 2 , 3 , 4 , 5 ) ; $ r = 3 ; $ n = sizeof ( $ arr ) ; printCombination ( $ arr , $ n , $ r ) ; ? >
< ? php function printCombination ( $ arr , $ n , $ r ) {
$ data = Array ( ) ;
combinationUtil ( $ arr , $ n , $ r , 0 , $ data , 0 ) ; }
function combinationUtil ( $ arr , $ n , $ r , $ index , $ data , $ i ) {
if ( $ index == $ r ) { for ( $ j = 0 ; $ j < $ r ; $ j ++ ) echo $ data [ $ j ] , " ▁ " ; echo " STRNEWLINE " ; return ; }
if ( $ i >= $ n ) return ;
$ data [ $ index ] = $ arr [ $ i ] ; combinationUtil ( $ arr , $ n , $ r , $ index + 1 , $ data , $ i + 1 ) ;
combinationUtil ( $ arr , $ n , $ r , $ index , $ data , $ i + 1 ) ; }
$ arr = array ( 1 , 2 , 3 , 4 , 5 ) ; $ r = 3 ; $ n = sizeof ( $ arr ) ; printCombination ( $ arr , $ n , $ r ) ; ? >
< ? php function findgroups ( $ arr , $ n ) {
$ c = array ( 0 , 0 , 0 ) ;
$ res = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ c [ $ arr [ $ i ] % 3 ] += 1 ;
$ res += ( ( $ c [ 0 ] * ( $ c [ 0 ] - 1 ) ) >> 1 ) ;
$ res += $ c [ 1 ] * $ c [ 2 ] ;
$ res += ( $ c [ 0 ] * ( $ c [ 0 ] - 1 ) * ( $ c [ 0 ] - 2 ) ) / 6 ;
$ res += ( $ c [ 1 ] * ( $ c [ 1 ] - 1 ) * ( $ c [ 1 ] - 2 ) ) / 6 ;
$ res += ( ( $ c [ 2 ] * ( $ c [ 2 ] - 1 ) * ( $ c [ 2 ] - 2 ) ) / 6 ) ;
$ res += $ c [ 0 ] * $ c [ 1 ] * $ c [ 2 ] ;
return $ res ; }
$ arr = array ( 3 , 6 , 7 , 2 , 9 ) ; $ n = count ( $ arr ) ; echo " Required ▁ number ▁ of ▁ groups ▁ are ▁ " . ( int ) ( findgroups ( $ arr , $ n ) ) ; ? >
< ? php function nextPowerOf2 ( $ n ) { $ count = 0 ;
if ( $ n && ! ( $ n & ( $ n - 1 ) ) ) return $ n ; while ( $ n != 0 ) { $ n >>= 1 ; $ count += 1 ; } return 1 << $ count ; }
$ n = 0 ; echo ( nextPowerOf2 ( $ n ) ) ; ? >
< ? php function nextPowerOf2 ( $ n ) { $ count = 0 ; if ( $ n && ! ( $ n & ( $ n - 1 ) ) ) return $ n ; while ( $ n != 0 ) { $ n >>= 1 ; $ count += 1 ; } return 1 << $ count ; }
$ n = 5 ; echo ( nextPowerOf2 ( $ n ) ) ; ? >
< ? php function nextPowerOf2 ( $ n ) { $ n -- ; $ n |= $ n >> 1 ; $ n |= $ n >> 2 ; $ n |= $ n >> 4 ; $ n |= $ n >> 8 ; $ n |= $ n >> 16 ; $ n ++ ; return $ n ; }
$ n = 5 ; echo nextPowerOf2 ( $ n ) ; ? >
< ? php function segregate0and1 ( & $ arr , $ n ) {
$ count = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ arr [ $ i ] == 0 ) $ count ++ ; }
for ( $ i = 0 ; $ i < $ count ; $ i ++ ) $ arr [ $ i ] = 0 ;
for ( $ i = $ count ; $ i < $ n ; $ i ++ ) $ arr [ $ i ] = 1 ; }
function toprint ( & $ arr , $ n ) { echo ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo ( $ arr [ $ i ] . " ▁ " ) ; }
$ arr = array ( 0 , 1 , 0 , 1 , 1 , 1 ) ; $ n = sizeof ( $ arr ) ; segregate0and1 ( $ arr , $ n ) ; toprint ( $ arr , $ n ) ; ? >
< ? php function segregate0and1 ( & $ arr , $ size ) {
$ left = 0 ; $ right = $ size - 1 ; while ( $ left < $ right ) {
while ( $ arr [ $ left ] == 0 && $ left < $ right ) $ left ++ ;
while ( $ arr [ $ right ] == 1 && $ left < $ right ) $ right -- ;
if ( $ left < $ right ) { $ arr [ $ left ] = 0 ; $ arr [ $ right ] = 1 ; $ left ++ ; $ right -- ; } } }
$ arr = array ( 0 , 1 , 0 , 1 , 1 , 1 ) ; $ arr_size = sizeof ( $ arr ) ; segregate0and1 ( $ arr , $ arr_size ) ; printf ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( $ i = 0 ; $ i < 6 ; $ i ++ ) echo ( $ arr [ $ i ] . " ▁ " ) ; ? >
< ? php function segregate0and1 ( & $ arr , $ size ) { $ type0 = 0 ; $ type1 = $ size - 1 ; while ( $ type0 < $ type1 ) { if ( $ arr [ $ type0 ] == 1 ) { $ temp = $ arr [ $ type0 ] ; $ arr [ $ type0 ] = $ arr [ $ type1 ] ; $ arr [ $ type1 ] = $ temp ; $ type1 -- ; } else $ type0 ++ ; } }
$ arr = array ( 0 , 1 , 0 , 1 , 1 , 1 ) ; $ arr_size = sizeof ( $ arr ) ; segregate0and1 ( $ arr , $ arr_size ) ; echo ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( $ i = 0 ; $ i < $ arr_size ; $ i ++ ) echo ( $ arr [ $ i ] . " ▁ " ) ; ? >
< ? php function maxIndexDiff ( $ arr , $ n ) { $ maxDiff = -1 ; for ( $ i = 0 ; $ i < $ n ; ++ $ i ) { for ( $ j = $ n - 1 ; $ j > $ i ; -- $ j ) { if ( $ arr [ $ j ] > $ arr [ $ i ] && $ maxDiff < ( $ j - $ i ) ) $ maxDiff = $ j - $ i ; } } return $ maxDiff ; }
$ arr = array ( 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ) ; $ n = count ( $ arr ) ; $ maxDiff = maxIndexDiff ( $ arr , $ n ) ; echo $ maxDiff ; ? >
< ? php function missingK ( & $ a , $ k , $ n ) { $ difference = 0 ; $ ans = 0 ; $ count = $ k ; $ flag = 0 ;
for ( $ i = 0 ; $ i < $ n - 1 ; $ i ++ ) { $ difference = 0 ;
if ( ( $ a [ $ i ] + 1 ) != $ a [ $ i + 1 ] ) {
$ difference += ( $ a [ $ i + 1 ] - $ a [ $ i ] ) - 1 ;
if ( $ difference >= $ count ) { $ ans = $ a [ $ i ] + $ count ; $ flag = 1 ; break ; } else $ count -= $ difference ; } }
if ( $ flag ) return $ ans ; else return -1 ; }
$ a = array ( 1 , 5 , 11 , 19 ) ;
$ k = 11 ; $ n = count ( $ a ) ;
$ missing = missingK ( $ a , $ k , $ n ) ; echo $ missing ; ? >
< ? php function findRotations ( $ str ) {
$ tmp = ( $ str + $ str ) ; $ n = strlen ( $ str ) ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) {
$ substring = $ tmp . substr ( $ i , strlen ( $ str ) ) ;
if ( $ str == $ substring ) return $ i ; } return $ n ; }
$ str = " abc " ; echo findRotations ( $ str ) , " STRNEWLINE " ; ? >
< ? php function findKth ( $ arr , $ n , $ k ) { $ missing = array ( ) ; $ count = 0 ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) array_push ( $ missing , $ arr [ $ i ] ) ; $ missing = array_unique ( $ missing ) ;
$ maxm = max ( $ arr ) ; $ minm = min ( $ arr ) ;
for ( $ i = $ minm + 1 ; $ i < $ maxm ; $ i ++ ) {
if ( ! in_array ( $ i , $ missing , false ) ) $ count += 1 ;
if ( $ count == $ k ) return $ i ; }
return -1 ; }
$ arr = array ( 2 , 10 , 9 , 4 ) ; $ n = sizeof ( $ arr ) ; $ k = 5 ; echo findKth ( $ arr , $ n , $ k ) ; ? >
< ? php function waysToKAdjacentSetBits ( $ n , $ k , $ currentIndex , $ adjacentSetBits , $ lastBit ) {
if ( $ currentIndex == $ n ) {
if ( $ adjacentSetBits == $ k ) return 1 ; return 0 ; } $ noOfWays = 0 ;
if ( $ lastBit == 1 ) {
$ noOfWays += waysToKAdjacentSetBits ( $ n , $ k , $ currentIndex + 1 , $ adjacentSetBits + 1 , 1 ) ;
$ noOfWays += waysToKAdjacentSetBits ( $ n , $ k , $ currentIndex + 1 , $ adjacentSetBits , 0 ) ; } else if ( ! $ lastBit ) { $ noOfWays += waysToKAdjacentSetBits ( $ n , $ k , $ currentIndex + 1 , $ adjacentSetBits , 1 ) ; $ noOfWays += waysToKAdjacentSetBits ( $ n , $ k , $ currentIndex + 1 , $ adjacentSetBits , 0 ) ; } return $ noOfWays ; }
$ n = 5 ; $ k = 2 ;
$ totalWays = waysToKAdjacentSetBits ( $ n , $ k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( $ n , $ k , 1 , 0 , 0 ) ; echo " Number ▁ of ▁ ways ▁ = ▁ " , $ totalWays , " STRNEWLINE " ; ? >
< ? php function findStep ( $ n ) { if ( $ n == 1 $ n == 0 ) return 1 ; else if ( $ n == 2 ) return 2 ; else return findStep ( $ n - 3 ) + findStep ( $ n - 2 ) + findStep ( $ n - 1 ) ; }
$ n = 4 ; echo findStep ( $ n ) ; ? >
< ? php function isSubsetSum ( $ arr , $ n , $ sum ) {
if ( $ sum == 0 ) return true ; if ( $ n == 0 && $ sum != 0 ) return false ;
if ( $ arr [ $ n - 1 ] > $ sum ) return isSubsetSum ( $ arr , $ n - 1 , $ sum ) ;
return isSubsetSum ( $ arr , $ n - 1 , $ sum ) || isSubsetSum ( $ arr , $ n - 1 , $ sum - $ arr [ $ n - 1 ] ) ; }
function findPartiion ( $ arr , $ n ) {
$ sum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ sum += $ arr [ $ i ] ;
if ( $ sum % 2 != 0 ) return false ;
return isSubsetSum ( $ arr , $ n , $ sum / 2 ) ; }
$ arr = array ( 3 , 1 , 5 , 9 , 12 ) ; $ n = count ( $ arr ) ;
if ( findPartiion ( $ arr , $ n ) == true ) echo " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ; else echo " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ; ? >
< ? php function findRepeatFirstN2 ( $ s ) {
$ p = -1 ; for ( $ i = 0 ; $ i < strlen ( $ s ) ; $ i ++ ) { for ( $ j = ( $ i + 1 ) ; $ j < strlen ( $ s ) ; $ j ++ ) { if ( $ s [ $ i ] == $ s [ $ j ] ) { $ p = $ i ; break ; } } if ( $ p != -1 ) break ; } return $ p ; }
$ str = " geeksforgeeks " ; $ pos = findRepeatFirstN2 ( $ str ) ; if ( $ pos == -1 ) echo ( " Not ▁ found " ) ; else echo ( $ str [ $ pos ] ) ; ? >
< ? php function possibleStrings ( $ n , $ r , $ b , $ g ) {
$ fact [ 0 ] = 1 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ fact [ $ i ] = $ fact [ $ i - 1 ] * $ i ;
$ left = $ n - ( $ r + $ g + $ b ) ; $ sum = 0 ;
for ( $ i = 0 ; $ i <= $ left ; $ i ++ ) { for ( $ j = 0 ; $ j <= $ left - $ i ; $ j ++ ) { $ k = $ left - ( $ i + $ j ) ;
$ sum = $ sum + $ fact [ $ n ] / ( $ fact [ $ i + $ r ] * $ fact [ $ j + $ b ] * $ fact [ $ k + $ g ] ) ; } }
return $ sum ; }
$ n = 4 ; $ r = 2 ; $ b = 0 ; $ g = 1 ; echo possibleStrings ( $ n , $ r , $ b , $ g ) ; ? >
< ? php function remAnagram ( $ str1 , $ str2 ) {
$ count1 = array ( 26 ) ; $ count2 = array ( 26 ) ;
for ( $ i = 0 ; $ i < strlen ( $ str1 ) ; $ i ++ ) $ count1 [ $ str1 [ $ i ] - ' a ' ] ++ ;
for ( $ i = 0 ; $ i < strlen ( $ str2 ) ; $ i ++ ) $ count2 [ $ str2 [ $ i ] - ' a ' ] ++ ;
$ result = 0 ; for ( $ i = 0 ; $ i < 26 ; $ i ++ ) $ result += abs ( $ count1 [ $ i ] - $ count2 [ $ i ] ) ; return $ result ; }
{ $ str1 = " bcadeh " ; $ str2 = " hea " ; echo ( remAnagram ( $ str1 , $ str2 ) ) ; }
< ? php function printPath ( $ res , $ nThNode , $ kThNode ) {
if ( $ kThNode > $ nThNode ) return ;
array_push ( $ res , $ kThNode ) ;
for ( $ i = 0 ; $ i < count ( $ res ) ; $ i ++ ) echo $ res [ $ i ] . " ▁ " ; echo " STRNEWLINE " ;
printPath ( $ res , $ nThNode , $ kThNode * 2 ) ;
printPath ( $ res , $ nThNode , $ kThNode * 2 + 1 ) ; }
function printPathToCoverAllNodeUtil ( $ nThNode ) {
$ res = array ( ) ;
printPath ( $ res , $ nThNode , 1 ) ; }
$ nThNode = 7 ;
printPathToCoverAllNodeUtil ( $ nThNode ) ; ? >
< ? php function shortestLength ( $ n , & $ x , & $ y ) { $ answer = 0 ;
$ i = 0 ; while ( $ n -- ) {
if ( $ x [ $ i ] + $ y [ $ i ] > $ answer ) $ answer = $ x [ $ i ] + $ y [ $ i ] ; $ i ++ ; }
echo " Length ▁ - > ▁ " . $ answer . " STRNEWLINE " ; echo " Path ▁ - > ▁ " . " ( ▁ 1 , ▁ " . $ answer . " ▁ ) " . " and ▁ ( ▁ " . $ answer . " , ▁ 1 ▁ ) " ; }
$ n = 4 ;
$ x = array ( 1 , 4 , 2 , 1 ) ; $ y = array ( 4 , 1 , 1 , 2 ) ; shortestLength ( $ n , $ x , $ y ) ; ? >
< ? php function FindPoints ( $ x1 , $ y1 , $ x2 , $ y2 , $ x3 , $ y3 , $ x4 , $ y4 ) {
$ x5 = max ( $ x1 , $ x3 ) ; $ y5 = max ( $ y1 , $ y3 ) ;
$ x6 = min ( $ x2 , $ x4 ) ; $ y6 = min ( $ y2 , $ y4 ) ;
if ( $ x5 > $ x6 $ y5 > $ y6 ) { echo " No ▁ intersection " ; return ; } echo " ( " . $ x5 . " , " ▁ . ▁ $ y5 ▁ . ▁ " ) " ; echo " ( " . $ x6 . " , " ▁ . ▁ $ y6 ▁ . ▁ " ) " ;
$ x7 = $ x5 ; $ y7 = $ y6 ; echo " ( " . $ x7 . " , ▁ " . $ y7 . " ) ▁ " ;
$ x8 = $ x6 ; $ y8 = $ y5 ; echo " ( " . $ x8 . " , ▁ " . $ y8 . " ) ▁ " ; }
$ x1 = 0 ; $ y1 = 0 ; $ x2 = 10 ; $ y2 = 8 ;
$ x3 = 2 ; $ y3 = 3 ; $ x4 = 7 ; $ y4 = 9 ;
FindPoints ( $ x1 , $ y1 , $ x2 , $ y2 , $ x3 , $ y3 , $ x4 , $ y4 ) ; ? >
< ? php function area ( $ a , $ b , $ c ) { $ d = abs ( ( $ c * $ c ) / ( 2 * $ a * $ b ) ) ; return $ d ; }
$ a = -2 ; $ b = 4 ; $ c = 3 ; echo area ( $ a , $ b , $ c ) ; ? >
< ? php function addToArrayForm ( $ A , $ K ) {
$ v = array ( ) ; $ ans = array ( ) ;
$ rem = 0 ; $ i = 0 ;
for ( $ i = count ( $ A ) - 1 ; $ i >= 0 ; $ i -- ) {
$ my = $ A [ $ i ] + $ K % 10 + $ rem ; if ( $ my > 9 ) {
$ rem = 1 ;
array_push ( $ v , $ my % 10 ) ; } else { array_push ( $ v , $ my ) ; $ rem = 0 ; } $ K = floor ( $ K / 10 ) ; }
while ( $ K > 0 ) {
$ my = $ K % 10 + $ rem ; array_push ( $ v , $ my % 10 ) ;
if ( $ my / 10 > 0 ) $ rem = 1 ; else $ rem = 0 ; $ K = floor ( $ K / 10 ) ; } if ( $ rem > 0 ) array_push ( $ v , $ rem ) ;
for ( $ i = count ( $ v ) - 1 ; $ i >= 0 ; $ i -- ) array_push ( $ ans , $ v [ $ i ] ) ; return $ ans ; }
$ A = array ( 2 , 7 , 4 ) ; $ K = 181 ; $ ans = addToArrayForm ( $ A , $ K ) ;
for ( $ i = 0 ; $ i < count ( $ ans ) ; $ i ++ ) echo $ ans [ $ i ] ; ? >
< ? php function findThirdDigit ( $ n ) {
if ( $ n < 3 ) return 0 ;
return $ n & 1 ? 1 : 6 ; }
$ n = 7 ; echo findThirdDigit ( $ n ) ; ? >
< ? php function getProbability ( $ a , $ b , $ c , $ d ) {
$ p = $ a / $ b ; $ q = $ c / $ d ;
$ ans = $ p * ( 1 / ( 1 - ( 1 - $ q ) * ( 1 - $ p ) ) ) ; return round ( $ ans , 6 ) ; }
$ a = 1 ; $ b = 2 ; $ c = 10 ; $ d = 11 ; echo getProbability ( $ a , $ b , $ c , $ d ) ; ? >
< ? php function isPalindrome ( $ n ) {
$ divisor = 1 ; while ( ( int ) ( $ n / $ divisor ) >= 10 ) $ divisor *= 10 ; while ( $ n != 0 ) { $ leading = ( int ) ( $ n / $ divisor ) ; $ trailing = $ n % 10 ;
if ( $ leading != $ trailing ) return false ;
$ n = ( $ n % $ divisor ) / 10 ;
$ divisor = $ divisor / 100 ; } return true ; }
function largestPalindrome ( $ A , $ n ) { $ currentMax = -1 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( $ A [ $ i ] > $ currentMax && isPalindrome ( $ A [ $ i ] ) ) $ currentMax = $ A [ $ i ] ; }
return $ currentMax ; }
$ A = array ( 1 , 232 , 54545 , 999991 ) ; $ n = sizeof ( $ A ) ;
echo ( largestPalindrome ( $ A , $ n ) ) ; ? >
< ? php function getFinalElement ( $ n ) { $ finalNum = 0 ; for ( $ finalNum = 2 ; ( $ finalNum * 2 ) <= $ n ; $ finalNum *= 2 ) ; return $ finalNum ; }
$ N = 12 ; echo getFinalElement ( $ N ) ; ? >
< ? php function isPalindrome ( $ num ) { $ reverse_num = 0 ; $ remainder ; $ temp ;
$ temp = $ num ; while ( $ temp != 0 ) { $ remainder = $ temp % 10 ; $ reverse_num = $ reverse_num * 10 + $ remainder ; $ temp = ( int ) ( $ temp / 10 ) ; }
if ( $ reverse_num == $ num ) { return true ; } return false ; }
function isOddLength ( $ num ) { $ count = 0 ; while ( $ num > 0 ) { $ num = ( int ) ( $ num / 10 ) ; $ count ++ ; } if ( $ count % 2 != 0 ) { return true ; } return false ; }
function sumOfAllPalindrome ( $ L , $ R ) { $ sum = 0 ; if ( $ L <= $ R ) for ( $ i = $ L ; $ i <= $ R ; $ i ++ ) {
if ( isPalindrome ( $ i ) && isOddLength ( $ i ) ) { $ sum += $ i ; } } return $ sum ; }
$ L = 110 ; $ R = 1130 ; echo sumOfAllPalindrome ( $ L , $ R ) ; ? >
< ? php function calculateAlternateSum ( $ n ) { if ( $ n <= 0 ) return 0 ; $ fibo = array ( ) ; $ fibo [ 0 ] = 0 ; $ fibo [ 1 ] = 1 ;
$ sum = pow ( $ fibo [ 0 ] , 2 ) + pow ( $ fibo [ 1 ] , 2 ) ;
for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) { $ fibo [ $ i ] = $ fibo [ $ i - 1 ] + $ fibo [ $ i - 2 ] ;
if ( $ i % 2 == 0 ) $ sum -= $ fibo [ $ i ] ;
else $ sum += $ fibo [ $ i ] ; }
return $ sum ; }
$ n = 8 ;
echo ( " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " ) ; echo $ n ; echo " ▁ terms : ▁ " ; echo ( calculateAlternateSum ( $ n ) ) ; ? >
< ? php function getValue ( $ n ) { $ i = 0 ; $ k = 1 ; while ( $ i < $ n ) { $ i = $ i + $ k ; $ k = $ k * 2 ; } return ( int ) $ k / 2 ; }
$ n = 9 ;
echo getValue ( $ n ) , " STRNEWLINE " ;
$ n = 1025 ;
echo getValue ( $ n ) , " STRNEWLINE " ; ? >
< ? php function countDigits ( $ val , & $ arr ) { while ( $ val > 0 ) { $ digit = $ val % 10 ; $ arr [ ( int ) ( $ digit ) ] += 1 ; $ val = ( int ) ( $ val / 10 ) ; } return ; } function countFrequency ( $ x , $ n ) {
$ freq_count = array_fill ( 0 , 10 , 0 ) ;
for ( $ i = 1 ; $ i < $ n + 1 ; $ i ++ ) {
$ val = pow ( $ x , $ i ) ;
countDigits ( $ val , $ freq_count ) ; }
for ( $ i = 0 ; $ i < 10 ; $ i ++ ) { echo $ freq_count [ $ i ] . " " ; } }
$ x = 15 ; $ n = 3 ; countFrequency ( $ x , $ n ) ? >
< ? php function countSolutions ( $ a ) { $ count = 0 ;
for ( $ i = 0 ; $ i <= $ a ; $ i ++ ) { if ( $ a == ( $ i + ( $ a ^ $ i ) ) ) $ count ++ ; } return $ count ; }
$ a = 3 ; echo countSolutions ( $ a ) ; ? >
< ? php function countSolutions ( $ a ) { $ count = bitCount ( $ a ) ; $ count = ( int ) pow ( 2 , $ count ) ; return $ count ; } function bitCount ( $ n ) { $ count = 0 ; while ( $ n != 0 ) { $ count ++ ; $ n &= ( $ n - 1 ) ; } return $ count ; }
$ a = 3 ; echo ( countSolutions ( $ a ) ) ; ? >
< ? php function calculateAreaSum ( $ l , $ b ) { $ size = 1 ;
$ maxSize = min ( $ l , $ b ) ; $ totalArea = 0 ; for ( $ i = 1 ; $ i <= $ maxSize ; $ i ++ ) {
$ totalSquares = ( $ l - $ size + 1 ) * ( $ b - $ size + 1 ) ;
$ area = $ totalSquares * $ size * $ size ;
$ totalArea += $ area ;
$ size ++ ; } return $ totalArea ; }
$ l = 4 ; $ b = 3 ; echo calculateAreaSum ( $ l , $ b ) ; ? >
< ? php function boost_hyperfactorial ( $ num ) {
$ val = 1 ; for ( $ i = 1 ; $ i <= $ num ; $ i ++ ) { $ val = $ val * pow ( $ i , $ i ) ; }
return $ val ; }
$ num = 5 ; echo boost_hyperfactorial ( $ num ) ; ? >
< ? php function boost_hyperfactorial ( $ num ) {
$ val = 1 ; for ( $ i = 1 ; $ i <= $ num ; $ i ++ ) { for ( $ j = 1 ; $ j <= $ i ; $ j ++ ) {
$ val *= $ i ; } }
return $ val ; }
$ num = 5 ; echo boost_hyperfactorial ( $ num ) ; ? >
< ? php function subtractOne ( $ x ) { $ m = 1 ;
while ( ! ( $ x & $ m ) ) { $ x = $ x ^ $ m ; $ m <<= 1 ; }
$ x = $ x ^ $ m ; return $ x ; }
echo subtractOne ( 13 ) ; ? >
< ? php $ rows = 3 ; $ cols = 3 ;
function meanVector ( $ mat ) { global $ rows , $ cols ; echo " [ "
for ( $ i = 0 ; $ i < $ rows ; $ i ++ ) {
$ mean = 0.00 ;
$ sum = 0 ; for ( $ j = 0 ; $ j < $ cols ; $ j ++ ) $ sum += $ mat [ $ j ] [ $ i ] ; $ mean = $ sum / $ rows ; echo $ mean , " " ; ▁ } ▁ echo ▁ " ] " }
$ mat = array ( array ( 1 , 2 , 3 ) , array ( 4 , 5 , 6 ) , array ( 7 , 8 , 9 ) ) ; meanVector ( $ mat ) ; ? >
< ? php function primeFactors ( $ n ) { $ res = array ( ) ; if ( $ n % 2 == 0 ) { while ( $ n % 2 == 0 ) $ n = ( int ) $ n / 2 ; array_push ( $ res , 2 ) ; }
for ( $ i = 3 ; $ i <= sqrt ( $ n ) ; $ i = $ i + 2 ) {
if ( $ n % $ i == 0 ) { while ( $ n % $ i == 0 ) $ n = ( int ) $ n / $ i ; array_push ( $ res , $ i ) ; } }
if ( $ n > 2 ) array_push ( $ res , $ n ) ; return $ res ; }
function isHoax ( $ n ) {
$ pf = primeFactors ( $ n ) ;
if ( $ pf [ 0 ] == $ n ) return false ;
$ all_pf_sum = 0 ; for ( $ i = 0 ; $ i < count ( $ pf ) ; $ i ++ ) {
$ pf_sum ; for ( $ pf_sum = 0 ; $ pf [ $ i ] > 0 ; $ pf_sum += $ pf [ $ i ] % 10 , $ pf [ $ i ] /= 10 ) ; $ all_pf_sum += $ pf_sum ; }
for ( $ sum_n = 0 ; $ n > 0 ; $ sum_n += $ n % 10 , $ n /= 10 ) ;
return $ sum_n == $ all_pf_sum ; }
$ n = 84 ; if ( isHoax ( $ n ) ) echo ( " A ▁ Hoax ▁ Number STRNEWLINE " ) ; else echo ( " Not ▁ a ▁ Hoax ▁ Number STRNEWLINE " ) ; ? >
< ? php function modInverse ( int $ a , int $ prime ) { $ a = $ a % $ prime ; for ( $ x = 1 ; $ x < $ prime ; $ x ++ ) if ( ( $ a * $ x ) % $ prime == 1 ) return $ x ; return -1 ; } function printModIverses ( $ n , $ prime ) { for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) echo modInverse ( $ i , $ prime ) , " ▁ " ; }
$ n = 10 ; $ prime = 17 ; printModIverses ( $ n , $ prime ) ; ? >
< ? php function minOp ( $ num ) {
$ count = 0 ;
while ( $ num ) { $ rem = intval ( $ num % 10 ) ; if ( ! ( $ rem == 3 $ rem == 8 ) ) $ count ++ ; $ num = intval ( $ num / 10 ) ; } return $ count ; }
$ num = 234198 ; echo " Minimum ▁ Operations ▁ = ▁ " . minOp ( $ num ) ; ? >
< ? php function sumOfDigits ( $ a ) { $ sum = 0 ; while ( $ a ) { $ sum += $ a % 10 ; $ a = ( int ) $ a / 10 ; } return $ sum ; }
function findMax ( $ x ) {
$ b = 1 ; $ ans = $ x ;
while ( $ x ) {
$ cur = ( $ x - 1 ) * $ b + ( $ b - 1 ) ;
if ( sumOfDigits ( $ cur ) > sumOfDigits ( $ ans ) || ( sumOfDigits ( $ cur ) == sumOfDigits ( $ ans ) && $ cur > $ ans ) ) $ ans = $ cur ;
$ x = ( int ) $ x / 10 ; $ b *= 10 ; } return $ ans ; }
$ n = 521 ; echo findMax ( $ n ) ; ? >
< ? php function median ( $ a , $ l , $ r ) { $ n = $ r - $ l + 1 ; $ n = ( int ) ( ( $ n + 1 ) / 2 ) - 1 ; return $ n + $ l ; }
function IQR ( $ a , $ n ) { sort ( $ a ) ;
$ mid_index = median ( $ a , 0 , $ n ) ;
$ Q1 = $ a [ median ( $ a , 0 , $ mid_index ) ] ;
$ Q3 = $ a [ $ mid_index + median ( $ a , $ mid_index + 1 , $ n ) ] ;
return ( $ Q3 - $ Q1 ) ; }
$ a = array ( 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 ) ; $ n = count ( $ a ) ; echo IQR ( $ a , $ n ) ; ? >
< ? php function isPalindrome ( $ n ) {
$ divisor = 1 ; while ( ( int ) ( $ n / $ divisor ) >= 10 ) $ divisor *= 10 ; while ( $ n != 0 ) { $ leading = ( int ) ( $ n / $ divisor ) ; $ trailing = $ n % 10 ;
if ( $ leading != $ trailing ) return false ;
$ n = ( int ) ( ( $ n % $ divisor ) / 10 ) ;
$ divisor = ( int ) ( $ divisor / 100 ) ; } return true ; }
function largestPalindrome ( $ A , $ n ) {
sort ( $ A ) ; for ( $ i = $ n - 1 ; $ i >= 0 ; -- $ i ) {
if ( isPalindrome ( $ A [ $ i ] ) ) return $ A [ $ i ] ; }
return -1 ; }
$ A = array ( 1 , 232 , 54545 , 999991 ) ; $ n = sizeof ( $ A ) ;
echo largestPalindrome ( $ A , $ n ) ; ? >
< ? php function findSum ( $ n , $ a , $ b ) { $ sum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ )
if ( $ i % $ a == 0 $ i % $ b == 0 ) $ sum += $ i ; return $ sum ; }
$ n = 10 ; $ a = 3 ; $ b = 5 ; echo findSum ( $ n , $ a , $ b ) ; ? >
< ? php function subtractOne ( $ x ) { return ( ( $ x << 1 ) + ( ~ $ x ) ) ; } print ( subtractOne ( 13 ) ) ; ? >
< ? php function pell ( $ n ) { if ( $ n <= 2 ) return $ n ; return 2 * pell ( $ n - 1 ) + pell ( $ n - 2 ) ; }
$ n = 4 ; echo ( pell ( $ n ) ) ; ? >
< ? php function LCM ( $ arr , $ n ) {
$ max_num = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) if ( $ max_num < $ arr [ $ i ] ) $ max_num = $ arr [ $ i ] ;
$ res = 1 ;
while ( $ x <= $ max_num ) {
$ indexes = array ( ) ; for ( $ j = 0 ; $ j < $ n ; $ j ++ ) if ( $ arr [ $ j ] % $ x == 0 ) array_push ( $ indexes , $ j ) ;
if ( count ( $ indexes ) >= 2 ) {
for ( $ j = 0 ; $ j < count ( $ indexes ) ; $ j ++ ) $ arr [ $ indexes [ $ j ] ] = ( int ) ( $ arr [ $ indexes [ $ j ] ] / $ x ) ; $ res = $ res * $ x ; } else $ x ++ ; }
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ res = $ res * $ arr [ $ i ] ; return $ res ; }
$ arr = array ( 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 ) ; $ n = count ( $ arr ) ; echo LCM ( $ arr , $ n ) . " STRNEWLINE " ; ? >
< ? php $ MAX = 10000 ;
$ primes = array ( ) ;
function sieveSundaram ( ) { global $ MAX , $ primes ;
$ marked = array_fill ( 0 , ( int ) ( $ MAX / 2 ) + 100 , false ) ;
for ( $ i = 1 ; $ i <= ( sqrt ( $ MAX ) - 1 ) / 2 ; $ i ++ ) for ( $ j = ( $ i * ( $ i + 1 ) ) << 1 ; $ j <= $ MAX / 2 ; $ j = $ j + 2 * $ i + 1 ) $ marked [ $ j ] = true ;
array_push ( $ primes , 2 ) ;
for ( $ i = 1 ; $ i <= $ MAX / 2 ; $ i ++ ) if ( $ marked [ $ i ] == false ) array_push ( $ primes , 2 * $ i + 1 ) ; }
function findPrimes ( $ n ) { global $ MAX , $ primes ;
if ( $ n <= 2 $ n % 2 != 0 ) { print ( " Invalid ▁ Input ▁ STRNEWLINE " ) ; return ; }
for ( $ i = 0 ; $ primes [ $ i ] <= $ n / 2 ; $ i ++ ) {
$ diff = $ n - $ primes [ $ i ] ;
if ( in_array ( $ diff , $ primes ) ) {
print ( $ primes [ $ i ] . " + " ▁ . ▁ $ diff ▁ . ▁ " = " ▁ . ▁ $ n ▁ . ▁ " " return ; } } }
sieveSundaram ( ) ;
findPrimes ( 4 ) ; findPrimes ( 38 ) ; findPrimes ( 100 ) ; ? >
< ? php function kPrimeFactor ( $ n , $ k ) {
while ( $ n % 2 == 0 ) { $ k -- ; $ n = $ n / 2 ; if ( $ k == 0 ) return 2 ; }
for ( $ i = 3 ; $ i <= sqrt ( $ n ) ; $ i = $ i + 2 ) {
while ( $ n % $ i == 0 ) { if ( $ k == 1 ) return $ i ; $ k -- ; $ n = $ n / $ i ; } }
if ( $ n > 2 && $ k == 1 ) return $ n ; return -1 ; }
{ $ n = 12 ; $ k = 3 ; echo kPrimeFactor ( $ n , $ k ) , " STRNEWLINE " ; $ n = 14 ; $ k = 3 ; echo kPrimeFactor ( $ n , $ k ) ; return 0 ; } ? >
< ? php $ MAX = 10001 ;
function sieveOfEratosthenes ( & $ s ) { global $ MAX ;
$ prime = array_fill ( 0 , $ MAX + 1 , false ) ;
for ( $ i = 2 ; $ i <= $ MAX ; $ i += 2 ) $ s [ $ i ] = 2 ;
for ( $ i = 3 ; $ i <= $ MAX ; $ i += 2 ) { if ( $ prime [ $ i ] == false ) {
$ s [ $ i ] = $ i ;
for ( $ j = $ i ; $ j * $ i <= $ MAX ; $ j += 2 ) { if ( $ prime [ $ i * $ j ] == false ) { $ prime [ $ i * $ j ] = true ;
$ s [ $ i * $ j ] = $ i ; } } } } }
function kPrimeFactor ( $ n , $ k , $ s ) {
while ( $ n > 1 ) { if ( $ k == 1 ) return $ s [ $ n ] ;
$ k -- ;
$ n = ( int ) ( $ n / $ s [ $ n ] ) ; } return -1 ; }
$ s = array_fill ( 0 , $ MAX + 1 , -1 ) ; sieveOfEratosthenes ( $ s ) ; $ n = 12 ; $ k = 3 ; print ( kPrimeFactor ( $ n , $ k , $ s ) . " " ) ; $ n = 14 ; $ k = 3 ; print ( kPrimeFactor ( $ n , $ k , $ s ) ) ; ? >
< ? php function squareRootExists ( $ n , $ p ) { $ n = $ n % $ p ;
for ( $ x = 2 ; $ x < $ p ; $ x ++ ) if ( ( $ x * $ x ) % $ p == $ n ) return true ; return false ; }
$ p = 7 ; $ n = 2 ; if ( squareRootExists ( $ n , $ p ) == true ) echo " Yes " ; else echo " No " ; ? >
< ? php function largestPower ( $ n , $ p ) {
$ x = 0 ;
while ( $ n ) { $ n = ( int ) $ n / $ p ; $ x += $ n ; } return floor ( $ x ) ; }
$ n = 10 ; $ p = 3 ; echo " The ▁ largest ▁ power ▁ of ▁ " , $ p ; echo " ▁ that ▁ divides ▁ " , $ n , " ! ▁ is ▁ " ; echo largestPower ( $ n , $ p ) ; ? >
< ? php function factorial ( $ n ) {
return ( $ n == 1 $ n == 0 ) ? 1 : $ n * factorial ( $ n - 1 ) ; }
$ num = 5 ; echo " Factorial ▁ of ▁ " , $ num , " ▁ is ▁ " , factorial ( $ num ) ; ? >
< ? php function reverseBits ( $ n ) { $ rev = 0 ;
while ( $ n > 0 ) {
$ rev <<= 1 ;
if ( $ n & 1 == 1 ) $ rev ^= 1 ;
$ n >>= 1 ; }
return $ rev ; }
$ n = 11 ; echo reverseBits ( $ n ) ; ? >
< ? php function countgroup ( $ a , $ n ) { $ xs = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ xs = $ xs ^ $ a [ $ i ] ;
if ( $ xs == 0 ) return ( 1 << ( $ n - 1 ) ) - 1 ; return 0 ; }
$ a = array ( 1 , 2 , 3 ) ; $ n = count ( $ a ) ; echo countgroup ( $ a , $ n ) ; ? >
< ? php function bitExtracted ( $ number , $ k , $ p ) { return ( ( ( 1 << $ k ) - 1 ) & ( $ number >> ( $ p - 1 ) ) ) ; }
$ number = 171 ; $ k = 5 ; $ p = 2 ; echo " The ▁ extracted ▁ number ▁ is ▁ " , bitExtracted ( $ number , $ k , $ p ) ; ? >
< ? php function isAMultipleOf4 ( $ n ) {
if ( ( $ n & 3 ) == 0 ) return " Yes " ;
return " No " ; }
$ n = 16 ; echo isAMultipleOf4 ( $ n ) ; ? >
< ? php function square ( $ n ) {
if ( $ n < 0 ) $ n = - $ n ;
$ res = $ n ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ res += $ n ; return $ res ; }
for ( $ n = 1 ; $ n <= 5 ; $ n ++ ) echo " n = " , ▁ $ n , ▁ " , " , ▁ " n ^ 2 = " , square ( $ n ) , " STRNEWLINE ▁ " ; ? >
< ? php function PointInKSquares ( $ n , $ a , $ k ) { sort ( $ a ) ; return $ a [ $ n - $ k ] ; }
$ k = 2 ; $ a = array ( 1 , 2 , 3 , 4 ) ; $ n = sizeof ( $ a ) ; $ x = PointInKSquares ( $ n , $ a , $ k ) ; echo " ( " . $ x . " , " ▁ . ▁ $ x ▁ . ▁ " ) " ; ? >
< ? php function answer ( $ n ) {
$ dp = array_fill ( 0 , 10 , 0 ) ;
$ prev = array_fill ( 0 , 10 , 0 ) ; ;
if ( $ n == 1 ) return 10 ;
for ( $ j = 0 ; $ j <= 9 ; $ j ++ ) $ dp [ $ j ] = 1 ;
for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) { for ( $ j = 0 ; $ j <= 9 ; $ j ++ ) { $ prev [ $ j ] = $ dp [ $ j ] ; } for ( $ j = 0 ; $ j <= 9 ; $ j ++ ) {
if ( $ j == 0 ) $ dp [ $ j ] = $ prev [ $ j + 1 ] ;
else if ( $ j == 9 ) $ dp [ $ j ] = $ prev [ $ j - 1 ] ;
else $ dp [ $ j ] = $ prev [ $ j - 1 ] + $ prev [ $ j + 1 ] ; } }
$ sum = 0 ; for ( $ j = 1 ; $ j <= 9 ; $ j ++ ) $ sum += $ dp [ $ j ] ; return $ sum ; }
$ n = 2 ; echo answer ( $ n ) ; ? >
< ? php $ MAX = 1000 ;
$ catalan = array_fill ( 0 , $ MAX , 0 ) ;
function catalanDP ( $ n ) { global $ catalan ;
$ catalan [ 0 ] = $ catalan [ 1 ] = 1 ;
for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) { $ catalan [ $ i ] = 0 ; for ( $ j = 0 ; $ j < $ i ; $ j ++ ) { $ catalan [ $ i ] += $ catalan [ $ j ] * $ catalan [ $ i - $ j - 1 ] ; } } }
function CatalanSequence ( $ arr , $ n ) { global $ catalan ;
catalanDP ( $ n ) ; $ s = array ( ) ;
$ a = $ b = 1 ;
array_push ( $ s , $ a ) ; if ( $ n >= 2 ) { array_push ( $ s , $ b ) ; } for ( $ i = 2 ; $ i < $ n ; $ i ++ ) { array_push ( $ s , $ catalan [ $ i ] ) ; } $ s = array_unique ( $ s ) ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
if ( in_array ( $ arr [ $ i ] , $ s ) ) { unset ( $ s [ array_search ( $ arr [ $ i ] , $ s ) ] ) ; } }
return count ( $ s ) ; }
$ arr = array ( 1 , 1 , 2 , 5 , 41 ) ; $ n = count ( $ arr ) ; print ( CatalanSequence ( $ arr , $ n ) ) ; ? >
< ? php function solve ( $ a , $ n ) { $ max1 = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = 0 ; $ j < $ n ; $ j ++ ) { if ( abs ( $ a [ $ i ] - $ a [ $ j ] ) > $ max1 ) { $ max1 = abs ( $ a [ $ i ] - $ a [ $ j ] ) ; } } } return $ max1 ; }
$ arr = array ( -1 , 2 , 3 , -4 , -10 , 22 ) ; $ size = count ( $ arr ) ; echo " Largest ▁ gap ▁ is ▁ : ▁ " , solve ( $ arr , $ size ) ; ? >
< ? php function solve ( $ a , $ n ) { $ min1 = $ a [ 0 ] ; $ max1 = $ a [ 0 ] ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ a [ $ i ] > $ max1 ) $ max1 = $ a [ $ i ] ; if ( $ a [ $ i ] < $ min1 ) $ min1 = $ a [ $ i ] ; } return abs ( $ min1 - $ max1 ) ; }
$ arr = array ( -1 , 2 , 3 , 4 , -10 ) ; $ size = count ( $ arr ) ; echo " Largest ▁ gap ▁ is ▁ : ▁ " , solve ( $ arr , $ size ) ; ? >
< ? php function minElements ( $ arr , $ n ) {
$ halfSum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) $ halfSum = $ halfSum + $ arr [ $ i ] ; $ halfSum = $ halfSum / 2 ;
rsort ( $ arr ) ; $ res = 0 ; $ curr_sum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { $ curr_sum += $ arr [ $ i ] ; $ res ++ ;
if ( $ curr_sum > $ halfSum ) return $ res ; } return $ res ; }
$ arr = array ( 3 , 1 , 7 , 1 ) ; $ n = sizeof ( $ arr ) ; echo minElements ( $ arr , $ n ) ; ? >
< ? php function minCost ( $ N , $ P , $ Q ) {
$ cost = 0 ;
while ( $ N > 0 ) { if ( $ N & 1 ) { $ cost += $ P ; $ N -- ; } else { $ temp = $ N / 2 ;
if ( $ temp * $ P > $ Q ) $ cost += $ Q ;
else $ cost += $ P * $ temp ; $ N /= 2 ; } }
return $ cost ; }
$ N = 9 ; $ P = 5 ; $ Q = 1 ; echo minCost ( $ N , $ P , $ Q ) ; ? >
