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
< ? php function permutationCoeff ( $ n , $ k ) { $ P = array ( array ( ) ) ;
for ( $ i = 0 ; $ i <= $ n ; $ i ++ ) { for ( $ j = 0 ; $ j <= min ( $ i , $ k ) ; $ j ++ ) {
if ( $ j == 0 ) $ P [ $ i ] [ $ j ] = 1 ;
else $ P [ $ i ] [ $ j ] = $ P [ $ i - 1 ] [ $ j ] + ( $ j * $ P [ $ i - 1 ] [ $ j - 1 ] ) ;
$ P [ $ i ] [ $ j + 1 ] = 0 ; } } return $ P [ $ n ] [ $ k ] ; }
$ n = 10 ; $ k = 2 ; echo " Value ▁ of ▁ P ( " , $ n , " ▁ , " , $ k , " ) ▁ is ▁ " , permutationCoeff ( $ n , $ k ) ; ? >
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
< ? php function pell ( $ n ) { if ( $ n <= 2 ) return $ n ; $ a = 1 ; $ b = 2 ; $ c ; $ i ; for ( $ i = 3 ; $ i <= $ n ; $ i ++ ) { $ c = 2 * $ b + $ a ; $ a = $ b ; $ b = $ c ; } return $ b ; }
$ n = 4 ; echo ( pell ( $ n ) ) ; ? >
< ? php function factorial ( $ n ) { if ( $ n == 0 ) return 1 ; return $ n * factorial ( $ n - 1 ) ; }
$ num = 5 ; echo " Factorial ▁ of ▁ " , $ num , " ▁ is ▁ " , factorial ( $ num ) ; ? >
< ? php function findSubArray ( & $ arr , $ n ) { $ sum = 0 ; $ maxsize = -1 ;
for ( $ i = 0 ; $ i < $ n - 1 ; $ i ++ ) { $ sum = ( $ arr [ $ i ] == 0 ) ? -1 : 1 ;
for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) { ( $ arr [ $ j ] == 0 ) ? ( $ sum += -1 ) : ( $ sum += 1 ) ;
if ( $ sum == 0 && $ maxsize < $ j - $ i + 1 ) { $ maxsize = $ j - $ i + 1 ; $ startindex = $ i ; } } } if ( $ maxsize == -1 ) echo " No ▁ such ▁ subarray " ; else echo $ startindex . " ▁ to ▁ " . ( $ startindex + $ maxsize - 1 ) ; return $ maxsize ; }
$ arr = array ( 1 , 0 , 0 , 1 , 0 , 1 , 1 ) ; $ size = sizeof ( $ arr ) ; findSubArray ( $ arr , $ size ) ; ? >
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
< ? php function isPowerOfTwo ( $ x ) {
return $ x && ( ! ( $ x & ( $ x - 1 ) ) ) ; }
if ( isPowerOfTwo ( 31 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; if ( isPowerOfTwo ( 64 ) ) echo " Yes STRNEWLINE " ; else echo " No STRNEWLINE " ; ? >
< ? php function nextPowerOf2 ( $ n ) { $ count = 0 ;
if ( $ n && ! ( $ n & ( $ n - 1 ) ) ) return $ n ; while ( $ n != 0 ) { $ n >>= 1 ; $ count += 1 ; } return 1 << $ count ; }
$ n = 0 ; echo ( nextPowerOf2 ( $ n ) ) ; ? >
< ? php function countWays ( $ n ) { $ res [ 0 ] = 1 ; $ res [ 1 ] = 1 ; $ res [ 2 ] = 2 ; for ( $ i = 3 ; $ i <= $ n ; $ i ++ ) $ res [ $ i ] = $ res [ $ i - 1 ] + $ res [ $ i - 2 ] + $ res [ $ i - 3 ] ; return $ res [ $ n ] ; }
$ n = 4 ; echo countWays ( $ n ) ; ? >
< ? php function maxTasks ( $ high , $ low , $ n ) {
if ( $ n <= 0 ) return 0 ;
return max ( $ high [ $ n - 1 ] + maxTasks ( $ high , $ low , ( $ n - 2 ) ) , $ low [ $ n - 1 ] + maxTasks ( $ high , $ low , ( $ n - 1 ) ) ) ; }
$ n = 5 ; $ high = array ( 3 , 6 , 8 , 7 , 6 ) ; $ low = array ( 1 , 5 , 4 , 5 , 3 ) ; print ( maxTasks ( $ high , $ low , $ n ) ) ; ? >
< ? php $ OUT = 0 ; $ IN = 1 ;
function countWords ( $ str ) { global $ OUT , $ IN ; $ state = $ OUT ;
$ wc = 0 ; $ i = 0 ;
while ( $ i < strlen ( $ str ) ) {
if ( $ str [ $ i ] == " ▁ " $ str [ $ i ] == " STRNEWLINE " $ str [ $ i ] == " TABSYMBOL " ) $ state = $ OUT ;
else if ( $ state == $ OUT ) { $ state = $ IN ; ++ $ wc ; }
++ $ i ; } return $ wc ; }
$ str = " One ▁ twothree STRNEWLINE ▁ four TABSYMBOL five ▁ " ; echo " No ▁ of ▁ words ▁ : ▁ " . countWords ( $ str ) ; ? >
< ? php function max1 ( $ x , $ y ) { return ( $ x > $ y ? $ x : $ y ) ; }
return ( $ x > $ y ? $ x : $ y ) ; }
function maxTasks ( $ high , $ low , $ n ) {
$ task_dp = array ( $ n + 1 ) ;
$ task_dp [ 0 ] = 0 ;
$ task_dp [ 1 ] = $ high [ 0 ] ;
for ( $ i = 2 ; $ i <= $ n ; $ i ++ ) $ task_dp [ $ i ] = max ( $ high [ $ i - 1 ] + $ task_dp [ $ i - 2 ] , $ low [ $ i - 1 ] + $ task_dp [ $ i - 1 ] ) ; return $ task_dp [ $ n ] ; }
{ $ n = 5 ; $ high = array ( 3 , 6 , 8 , 7 , 6 ) ; $ low = array ( 1 , 5 , 4 , 5 , 3 ) ; echo ( maxTasks ( $ high , $ low , $ n ) ) ; }
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
< ? php function segregate0and1 ( & $ arr , $ size ) {
$ left = 0 ; $ right = $ size - 1 ; while ( $ left < $ right ) {
while ( $ arr [ $ left ] == 0 && $ left < $ right ) $ left ++ ;
while ( $ arr [ $ right ] == 1 && $ left < $ right ) $ right -- ;
if ( $ left < $ right ) { $ arr [ $ left ] = 0 ; $ arr [ $ right ] = 1 ; $ left ++ ; $ right -- ; } } }
$ arr = array ( 0 , 1 , 0 , 1 , 1 , 1 ) ; $ arr_size = sizeof ( $ arr ) ; segregate0and1 ( $ arr , $ arr_size ) ; printf ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( $ i = 0 ; $ i < 6 ; $ i ++ ) echo ( $ arr [ $ i ] . " ▁ " ) ; ? >
< ? php function maxIndexDiff ( $ arr , $ n ) { $ maxDiff = -1 ; for ( $ i = 0 ; $ i < $ n ; ++ $ i ) { for ( $ j = $ n - 1 ; $ j > $ i ; -- $ j ) { if ( $ arr [ $ j ] > $ arr [ $ i ] && $ maxDiff < ( $ j - $ i ) ) $ maxDiff = $ j - $ i ; } } return $ maxDiff ; }
$ arr = array ( 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ) ; $ n = count ( $ arr ) ; $ maxDiff = maxIndexDiff ( $ arr , $ n ) ; echo $ maxDiff ; ? >
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
< ? php function isPalindrome ( $ num ) { $ reverse_num = 0 ; $ remainder ; $ temp ;
$ temp = $ num ; while ( $ temp != 0 ) { $ remainder = $ temp % 10 ; $ reverse_num = $ reverse_num * 10 + $ remainder ; $ temp = ( int ) ( $ temp / 10 ) ; }
if ( $ reverse_num == $ num ) { return true ; } return false ; }
function isOddLength ( $ num ) { $ count = 0 ; while ( $ num > 0 ) { $ num = ( int ) ( $ num / 10 ) ; $ count ++ ; } if ( $ count % 2 != 0 ) { return true ; } return false ; }
function sumOfAllPalindrome ( $ L , $ R ) { $ sum = 0 ; if ( $ L <= $ R ) for ( $ i = $ L ; $ i <= $ R ; $ i ++ ) {
if ( isPalindrome ( $ i ) && isOddLength ( $ i ) ) { $ sum += $ i ; } } return $ sum ; }
$ L = 110 ; $ R = 1130 ; echo sumOfAllPalindrome ( $ L , $ R ) ; ? >
< ? php function subtractOne ( $ x ) { $ m = 1 ;
while ( ! ( $ x & $ m ) ) { $ x = $ x ^ $ m ; $ m <<= 1 ; }
$ x = $ x ^ $ m ; return $ x ; }
echo subtractOne ( 13 ) ; ? >
< ? php function findSum ( $ n , $ a , $ b ) { $ sum = 0 ; for ( $ i = 0 ; $ i < $ n ; $ i ++ )
if ( $ i % $ a == 0 $ i % $ b == 0 ) $ sum += $ i ; return $ sum ; }
$ n = 10 ; $ a = 3 ; $ b = 5 ; echo findSum ( $ n , $ a , $ b ) ; ? >
< ? php function pell ( $ n ) { if ( $ n <= 2 ) return $ n ; return 2 * pell ( $ n - 1 ) + pell ( $ n - 2 ) ; }
$ n = 4 ; echo ( pell ( $ n ) ) ; ? >
< ? php function largestPower ( $ n , $ p ) {
$ x = 0 ;
while ( $ n ) { $ n = ( int ) $ n / $ p ; $ x += $ n ; } return floor ( $ x ) ; }
$ n = 10 ; $ p = 3 ; echo " The ▁ largest ▁ power ▁ of ▁ " , $ p ; echo " ▁ that ▁ divides ▁ " , $ n , " ! ▁ is ▁ " ; echo largestPower ( $ n , $ p ) ; ? >
< ? php function factorial ( $ n ) {
return ( $ n == 1 $ n == 0 ) ? 1 : $ n * factorial ( $ n - 1 ) ; }
$ num = 5 ; echo " Factorial ▁ of ▁ " , $ num , " ▁ is ▁ " , factorial ( $ num ) ; ? >
< ? php function bitExtracted ( $ number , $ k , $ p ) { return ( ( ( 1 << $ k ) - 1 ) & ( $ number >> ( $ p - 1 ) ) ) ; }
$ number = 171 ; $ k = 5 ; $ p = 2 ; echo " The ▁ extracted ▁ number ▁ is ▁ " , bitExtracted ( $ number , $ k , $ p ) ; ? >
< ? php function solve ( $ a , $ n ) { $ max1 = PHP_INT_MIN ; for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { for ( $ j = 0 ; $ j < $ n ; $ j ++ ) { if ( abs ( $ a [ $ i ] - $ a [ $ j ] ) > $ max1 ) { $ max1 = abs ( $ a [ $ i ] - $ a [ $ j ] ) ; } } } return $ max1 ; }
$ arr = array ( -1 , 2 , 3 , -4 , -10 , 22 ) ; $ size = count ( $ arr ) ; echo " Largest ▁ gap ▁ is ▁ : ▁ " , solve ( $ arr , $ size ) ; ? >
< ? php function solve ( $ a , $ n ) { $ min1 = $ a [ 0 ] ; $ max1 = $ a [ 0 ] ;
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) { if ( $ a [ $ i ] > $ max1 ) $ max1 = $ a [ $ i ] ; if ( $ a [ $ i ] < $ min1 ) $ min1 = $ a [ $ i ] ; } return abs ( $ min1 - $ max1 ) ; }
$ arr = array ( -1 , 2 , 3 , 4 , -10 ) ; $ size = count ( $ arr ) ; echo " Largest ▁ gap ▁ is ▁ : ▁ " , solve ( $ arr , $ size ) ; ? >
