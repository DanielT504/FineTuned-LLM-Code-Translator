< ? php function distance ( $ a1 , $ b1 , $ c1 , $ d1 , $ a2 , $ b2 , $ c2 , $ d2 ) { if ( $ a1 / $ a2 == $ b1 / $ b2 && $ b1 / $ b2 == $ c1 / $ c2 ) { $ x1 = $ y1 = 0 ; $ z1 = - $ d1 / $ c1 ; $ d = abs ( ( $ c2 * $ z1 + $ d2 ) ) / ( sqrt ( $ a2 * $ a2 + $ b2 * $ b2 + $ c2 * $ c2 ) ) ; echo " Perpendicular ▁ distance ▁ is ▁ " , $ d ; } else echo " Planes ▁ are ▁ not ▁ parallel " ; }
$ a1 = 1 ; $ b1 = 2 ; $ c1 = -1 ; $ d1 = 1 ; $ a2 = 3 ; $ b2 = 6 ; $ c2 = -3 ; $ d2 = -4 ; distance ( $ a1 , $ b1 , $ c1 , $ d1 , $ a2 , $ b2 , $ c2 , $ d2 ) ; ? >
< ? php function Series ( $ n ) { $ i ; $ sums = 0 ; for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) $ sums += ( $ i * $ i ) ; return $ sums ; }
$ n = 3 ; $ res = Series ( $ n ) ; echo ( $ res ) ; ? >
< ? php function areElementsContiguous ( $ arr , $ n ) {
sort ( $ arr ) ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) if ( $ arr [ $ i ] - $ arr [ $ i - 1 ] > 1 ) return false ; return true ; }
$ arr = array ( 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ) ; $ n = sizeof ( $ arr ) ; if ( areElementsContiguous ( $ arr , $ n ) ) echo " Yes " ; else echo " No " ; ? >
< ? php function leftRotatebyOne ( & $ arr , $ n ) { $ temp = $ arr [ 0 ] ; for ( $ i = 0 ; $ i < $ n - 1 ; $ i ++ ) $ arr [ $ i ] = $ arr [ $ i + 1 ] ; $ arr [ $ n - 1 ] = $ temp ; }
function leftRotate ( & $ arr , $ d , $ n ) { for ( $ i = 0 ; $ i < $ d ; $ i ++ ) leftRotatebyOne ( $ arr , $ n ) ; }
function printArray ( & $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; }
$ arr = array ( 1 , 2 , 3 , 4 , 5 , 6 , 7 ) ; $ n = sizeof ( $ arr ) ; leftRotate ( $ arr , 2 , $ n ) ; printArray ( $ arr , $ n ) ; ? >
< ? php function findFirstMissing ( $ array , $ start , $ end ) { if ( $ start > $ end ) return $ end + 1 ; if ( $ start != $ array [ $ start ] ) return $ start ; $ mid = ( $ start + $ end ) / 2 ;
if ( $ array [ $ mid ] == $ mid ) return findFirstMissing ( $ array , $ mid + 1 , $ end ) ; return findFirstMissing ( $ array , $ start , $ mid ) ; }
$ arr = array ( 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ) ; $ n = count ( $ arr ) ; echo " Smallest ▁ missing ▁ element ▁ is ▁ " , findFirstMissing ( $ arr , 2 , $ n - 1 ) ; ? >
< ? php function FindMaxSum ( $ arr , $ n ) { $ incl = $ arr [ 0 ] ; $ excl = 0 ; $ excl_new ; $ i ; for ( $ i = 1 ; $ i < $ n ; $ i ++ ) {
$ excl_new = ( $ incl > $ excl ) ? $ incl : $ excl ;
$ incl = $ excl + $ arr [ $ i ] ; $ excl = $ excl_new ; }
return ( ( $ incl > $ excl ) ? $ incl : $ excl ) ; }
$ arr = array ( 5 , 5 , 10 , 100 , 10 , 5 ) ; $ n = sizeof ( $ arr ) ; echo FindMaxSum ( $ arr , $ n ) ; ? >
< ? php function isMajority ( $ arr , $ n , $ x ) { $ i ;
$ last_index = $ n % 2 ? ( $ n / 2 + 1 ) : ( $ n / 2 ) ;
for ( $ i = 0 ; $ i < $ last_index ; $ i ++ ) {
if ( $ arr [ $ i ] == $ x && $ arr [ $ i + $ n / 2 ] == $ x ) return 1 ; } return 0 ; }
$ arr = array ( 1 , 2 , 3 , 4 , 4 , 4 , 4 ) ; $ n = sizeof ( $ arr ) ; $ x = 4 ; if ( isMajority ( $ arr , $ n , $ x ) ) echo $ x , " ▁ appears ▁ more ▁ than ▁ " , floor ( $ n / 2 ) , " ▁ times ▁ in ▁ arr [ ] " ; else echo $ x , " does ▁ not ▁ appear ▁ more ▁ than ▁ " , floor ( $ n / 2 ) , " times ▁ in ▁ arr [ ] " ; ? >
< ? php function cutRod ( $ price , $ n ) { $ val = array ( ) ; $ val [ 0 ] = 0 ; $ i ; $ j ;
for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) { $ max_val = PHP_INT_MIN ; for ( $ j = 0 ; $ j < $ i ; $ j ++ ) $ max_val = max ( $ max_val , $ price [ $ j ] + $ val [ $ i - $ j - 1 ] ) ; $ val [ $ i ] = $ max_val ; } return $ val [ $ n ] ; }
$ arr = array ( 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ) ; $ size = count ( $ arr ) ; echo " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " , cutRod ( $ arr , $ size ) ; ? >
< ? php function Convert ( $ radian ) { $ pi = 3.14159 ; return ( $ radian * ( 180 / $ pi ) ) ; }
$ radian = 5.0 ; $ degree = Convert ( $ radian ) ; echo ( $ degree ) ; ? >
< ? php function subtract ( $ x , $ y ) {
while ( $ y != 0 ) {
$ borrow = ( ~ $ x ) & $ y ;
$ x = $ x ^ $ y ;
$ y = $ borrow << 1 ; } return $ x ; }
$ x = 29 ; $ y = 13 ; echo " x ▁ - ▁ y ▁ is ▁ " , subtract ( $ x , $ y ) ; ? >
< ? php function subtract ( $ x , $ y ) { if ( $ y == 0 ) return $ x ; return subtract ( $ x ^ $ y , ( ~ $ x & $ y ) << 1 ) ; }
$ x = 29 ; $ y = 13 ; echo " x ▁ - ▁ y ▁ is ▁ " , subtract ( $ x , $ y ) ; # This  code is contributed by ajit NEW_LINE ? >
< ? php function reverse ( $ str ) { if ( ( $ str == null ) || ( strlen ( $ str ) <= 1 ) ) echo ( $ str ) ; else { echo ( $ str [ strlen ( $ str ) - 1 ] ) ; reverse ( substr ( $ str , 0 , ( strlen ( $ str ) - 1 ) ) ) ; } }
$ str = " Geeks ▁ for ▁ Geeks " ; reverse ( $ str ) ; ? >
< ? php $ cola = 2 ; $ rowa = 3 ; $ colb = 3 ; $ rowb = 2 ;
function Kroneckerproduct ( $ A , $ B ) { global $ cola ; global $ rowa ; global $ colb ; global $ rowb ; $ C ;
for ( $ i = 0 ; $ i < $ rowa ; $ i ++ ) {
for ( $ k = 0 ; $ k < $ rowb ; $ k ++ ) {
for ( $ j = 0 ; $ j < $ cola ; $ j ++ ) {
for ( $ l = 0 ; $ l < $ colb ; $ l ++ ) {
$ C [ $ i + $ l + 1 ] [ $ j + $ k + 1 ] = $ A [ $ i ] [ $ j ] * $ B [ $ k ] [ $ l ] ; echo ( $ C [ $ i + $ l + 1 ] [ $ j + $ k + 1 ] ) , " TABSYMBOL " ; } } echo " " } } }
$ A = array ( array ( 1 , 2 ) , array ( 3 , 4 ) , array ( 1 , 0 ) ) ; $ B = array ( array ( 0 , 5 , 2 ) , array ( 6 , 7 , 3 ) ) ; Kroneckerproduct ( $ A , $ B ) ; ? >
< ? php function selection_sort ( & $ arr , $ n ) {
for ( $ i = 0 ; $ i < $ n ; $ i ++ ) {
$ low = $ i ; for ( $ j = $ i + 1 ; $ j < $ n ; $ j ++ ) { if ( $ arr [ $ j ] < $ arr [ $ low ] ) { $ low = $ j ; } }
if ( $ arr [ $ i ] > $ arr [ $ low ] ) { $ tmp = $ arr [ $ i ] ; $ arr [ $ i ] = $ arr [ $ low ] ; $ arr [ $ low ] = $ tmp ; } } }
$ arr = array ( 64 , 25 , 12 , 22 , 11 ) ; $ len = count ( $ arr ) ; selection_sort ( $ arr , $ len ) ; echo " Sorted ▁ array ▁ : ▁ STRNEWLINE " ; for ( $ i = 0 ; $ i < $ len ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; ? >
< ? php function MatrixChainOrder ( $ p , $ n ) {
$ m [ ] [ ] = array ( $ n , $ n ) ;
for ( $ i = 1 ; $ i < $ n ; $ i ++ ) $ m [ $ i ] [ $ i ] = 0 ;
for ( $ L = 2 ; $ L < $ n ; $ L ++ ) { for ( $ i = 1 ; $ i < $ n - $ L + 1 ; $ i ++ ) { $ j = $ i + $ L - 1 ; if ( $ j == $ n ) continue ; $ m [ $ i ] [ $ j ] = PHP_INT_MAX ; for ( $ k = $ i ; $ k <= $ j - 1 ; $ k ++ ) {
$ q = $ m [ $ i ] [ $ k ] + $ m [ $ k + 1 ] [ $ j ] + $ p [ $ i - 1 ] * $ p [ $ k ] * $ p [ $ j ] ; if ( $ q < $ m [ $ i ] [ $ j ] ) $ m [ $ i ] [ $ j ] = $ q ; } } } return $ m [ 1 ] [ $ n - 1 ] ; }
$ arr = array ( 1 , 2 , 3 , 4 ) ; $ size = sizeof ( $ arr ) ; echo " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " . MatrixChainOrder ( $ arr , $ size ) ; ? >
< ? php function multiply ( $ x , $ y ) {
if ( $ y == 0 ) return 0 ;
if ( $ y > 0 ) return ( $ x + multiply ( $ x , $ y - 1 ) ) ;
if ( $ y < 0 ) return - multiply ( $ x , - $ y ) ; }
echo multiply ( 5 , -11 ) ; ? >
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
< ? php function countSetBits ( $ n ) { $ count = 0 ; while ( $ n ) { $ count += $ n & 1 ; $ n >>= 1 ; } return $ count ; }
$ i = 9 ; echo countSetBits ( $ i ) ; ? >
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
< ? php function MatrixChainOrder ( & $ p , $ i , $ j ) { if ( $ i == $ j ) return 0 ; $ min = PHP_INT_MAX ;
for ( $ k = $ i ; $ k < $ j ; $ k ++ ) { $ count = MatrixChainOrder ( $ p , $ i , $ k ) + MatrixChainOrder ( $ p , $ k + 1 , $ j ) + $ p [ $ i - 1 ] * $ p [ $ k ] * $ p [ $ j ] ; if ( $ count < $ min ) $ min = $ count ; }
return $ min ; }
$ arr = array ( 1 , 2 , 3 , 4 , 3 ) ; $ n = sizeof ( $ arr ) ; echo " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " . MatrixChainOrder ( $ arr , 1 , $ n - 1 ) ; ? >
< ? php function Perimeter ( $ s , $ n ) { $ perimeter = 1 ;
$ perimeter = $ n * $ s ; return $ perimeter ; }
$ n = 5 ;
$ s = 2.5 ;
$ peri = Perimeter ( $ s , $ n ) ; echo " Perimeter ▁ of ▁ Regular ▁ Polygon " , " ▁ with ▁ " , $ n , " ▁ sides ▁ of ▁ length ▁ " , $ s , " ▁ = ▁ " , $ peri ; ? >
< ? php function shortest_distance ( $ x1 , $ y1 , $ z1 , $ a , $ b , $ c , $ d ) { $ d = abs ( ( $ a * $ x1 + $ b * $ y1 + $ c * $ z1 + $ d ) ) ; $ e = sqrt ( $ a * $ a + $ b * $ b + $ c * $ c ) ; echo " Perpendicular ▁ distance ▁ is ▁ " . $ d / $ e ; }
$ x1 = 4 ; $ y1 = -4 ; $ z1 = 3 ; $ a = 2 ; $ b = -2 ; $ c = 5 ; $ d = 8 ;
shortest_distance ( $ x1 , $ y1 , $ z1 , $ a , $ b , $ c , $ d ) ; ? >
< ? php function averageOdd ( $ n ) { if ( $ n % 2 == 0 ) { echo ( " Invalid ▁ Input " ) ; return -1 ; } return ( $ n + 1 ) / 2 ; }
$ n = 15 ; echo ( averageOdd ( $ n ) ) ; ? >
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
< ? php function averageEven ( $ n ) { if ( $ n % 2 != 0 ) { echo ( " Invalid ▁ Input " ) ; return -1 ; } return ( $ n + 2 ) / 2 ; }
$ n = 16 ; echo ( averageEven ( $ n ) ) ; return 0 ; ? >
< ? php function fact ( $ n ) { if ( $ n == 0 ) return 1 ; return $ n * fact ( $ n - 1 ) ; }
function div ( $ x ) { $ ans = 0 ; for ( $ i = 1 ; $ i <= $ x ; $ i ++ ) if ( $ x % $ i == 0 ) $ ans += $ i ; return $ ans ; }
function sumFactDiv ( $ n ) { return div ( fact ( $ n ) ) ; }
$ n = 4 ; echo sumFactDiv ( $ n ) ; ? >
< ? php function printDivisors ( $ n ) { for ( $ i = 1 ; $ i <= $ n ; $ i ++ ) if ( $ n % $ i == 0 ) echo $ i , " ▁ " ; }
echo " The ▁ divisors ▁ of ▁ 100 ▁ are : STRNEWLINE " ; printDivisors ( 100 ) ; ? >
< ? php function printDivisors ( $ n ) {
for ( $ i = 1 ; $ i <= sqrt ( $ n ) ; $ i ++ ) { if ( $ n % $ i == 0 ) {
if ( $ n / $ i == $ i ) echo $ i , " ▁ " ;
else echo $ i , " " , ▁ $ n / $ i , " " } } }
echo " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ; printDivisors ( 100 ) ; ? >
< ? php $ rev_num = 0 ; $ base_pos = 1 ;
function reversDigits ( $ num ) { global $ rev_num ; global $ base_pos ; if ( $ num > 0 ) { reversDigits ( ( int ) ( $ num / 10 ) ) ; $ rev_num += ( $ num % 10 ) * $ base_pos ; $ base_pos *= 10 ; } return $ rev_num ; }
$ num = 4562 ; echo " Reverse ▁ of ▁ no . ▁ is ▁ " , reversDigits ( $ num ) ; ? >
< ? php function multiplyBySevenByEight ( $ n ) {
return ( $ n - ( $ n >> 3 ) ) ; }
$ n = 9 ; echo multiplyBySevenByEight ( $ n ) ; ? >
< ? php function multiplyBySevenByEight ( $ n ) {
return ( ( $ n << 3 ) - $ n ) >> 3 ; }
$ n = 15 ; echo multiplyBySevenByEight ( $ n ) ; ? >
< ? php function insertionSort ( & $ arr , $ n ) { for ( $ i = 1 ; $ i < $ n ; $ i ++ ) { $ key = $ arr [ $ i ] ; $ j = $ i - 1 ;
while ( $ j >= 0 && $ arr [ $ j ] > $ key ) { $ arr [ $ j + 1 ] = $ arr [ $ j ] ; $ j = $ j - 1 ; } $ arr [ $ j + 1 ] = $ key ; } }
function printArray ( & $ arr , $ n ) { for ( $ i = 0 ; $ i < $ n ; $ i ++ ) echo $ arr [ $ i ] . " ▁ " ; echo " STRNEWLINE " ; }
$ arr = array ( 12 , 11 , 13 , 5 , 6 ) ; $ n = sizeof ( $ arr ) ; insertionSort ( $ arr , $ n ) ; printArray ( $ arr , $ n ) ; ? >
< ? php function coun ( $ S , $ m , $ n ) {
if ( $ n == 0 ) return 1 ;
if ( $ n < 0 ) return 0 ;
if ( $ m <= 0 && $ n >= 1 ) return 0 ;
return coun ( $ S , $ m - 1 , $ n ) + coun ( $ S , $ m , $ n - $ S [ $ m - 1 ] ) ; }
$ arr = array ( 1 , 2 , 3 ) ; $ m = count ( $ arr ) ; echo coun ( $ arr , $ m , 4 ) ; ? >
< ? php function Area ( $ b1 , $ b2 , $ h ) { return ( ( $ b1 + $ b2 ) / 2 ) * $ h ; }
$ base1 = 8 ; $ base2 = 10 ; $ height = 6 ; $ area = Area ( $ base1 , $ base2 , $ height ) ; echo ( " Area ▁ is : ▁ " ) ; echo ( $ area ) ; ? >
