function distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) { let x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = Math . abs ( ( c2 * z1 + d2 ) ) / ( Math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; document . write ( " " + d ) ; } else document . write ( " " ) ; }
let a1 = 1 ; let b1 = 2 ; let c1 = - 1 ; let d1 = 1 ; let a2 = 3 ; let b2 = 6 ; let c2 = - 3 ; let d2 = - 4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ;
function Series ( n ) { let i ; let sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
let n = 3 ; let res = Series ( n ) ; document . write ( res ) ;
function areElementsContiguous ( arr , n ) {
arr . sort ( function ( a , b ) { return a - b } ) ;
for ( let i = 1 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] > 1 ) return false ; return true ; }
let arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] ; let n = arr . length ; if ( areElementsContiguous ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function leftRotatebyOne ( arr , n ) { var i , temp ; temp = arr [ 0 ] ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
function leftRotate ( arr , d , n ) { for ( i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
function printArray ( arr , n ) { for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; }
var arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ;
function findFirstMissing ( array , start , end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; let mid = parseInt ( ( start + end ) / 2 , 10 ) ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
let arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ] ; let n = arr . length ; document . write ( " " + findFirstMissing ( arr , 0 , n - 1 ) ) ;
function FindMaxSum ( arr , n ) { let incl = arr [ 0 ] ; let excl = 0 ; let excl_new ; let i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
let arr = [ 5 , 5 , 10 , 100 , 10 , 5 ] ; document . write ( FindMaxSum ( arr , arr . length ) ) ;
function isMajority ( arr , n , x ) { let i , last_index = 0 ;
last_index = ( n % 2 == 0 ) ? parseInt ( n / 2 , 10 ) : parseInt ( n / 2 , 10 ) + 1 ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + parseInt ( n / 2 , 10 ) ] == x ) return true ; } return false ; }
let arr = [ 1 , 2 , 3 , 4 , 4 , 4 , 4 ] ; let n = arr . length ; let x = 4 ; if ( isMajority ( arr , n , x ) == true ) document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ; else document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ;
function _binarySearch ( arr , low , high , x ) { if ( high >= low ) { let mid = parseInt ( ( low + high ) / 2 , 10 ) ;
if ( ( mid == 0 x > arr [ mid - 1 ] ) && ( arr [ mid ] == x ) ) return mid ; else if ( x > arr [ mid ] ) return _binarySearch ( arr , ( mid + 1 ) , high , x ) ; else return _binarySearch ( arr , low , ( mid - 1 ) , x ) ; } return - 1 ; }
function isMajority ( arr , n , x ) {
let i = _binarySearch ( arr , 0 , n - 1 , x ) ;
if ( i == - 1 ) return false ;
if ( ( ( i + parseInt ( n / 2 , 10 ) ) <= ( n - 1 ) ) && arr [ i + parseInt ( n / 2 , 10 ) ] == x ) return true ; else return false ; }
let arr = [ 1 , 2 , 3 , 3 , 3 , 3 , 10 ] ; let n = arr . length ; let x = 3 ; if ( isMajority ( arr , n , x ) == true ) document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ; else document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ;
function isMajorityElement ( arr , n , key ) { if ( arr [ parseInt ( n / 2 , 10 ) ] == key ) return true ; else return false ; }
let arr = [ 1 , 2 , 3 , 3 , 3 , 3 , 10 ] ; let n = arr . length ; let x = 3 ; if ( isMajorityElement ( arr , n , x ) ) document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ; else document . write ( x + " " + " " + parseInt ( n / 2 , 10 ) + " " ) ;
function cutRod ( price , n ) { let val = new Array ( n + 1 ) ; val [ 0 ] = 0 ;
for ( let i = 1 ; i <= n ; i ++ ) { let max_val = Number . MIN_VALUE ; for ( let j = 0 ; j < i ; j ++ ) max_val = Math . max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
let arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] ; let size = arr . length ; document . write ( " " + cutRod ( arr , size ) + " " ) ;
let t = new Array ( 9 ) ; for ( var i = 0 ; i < t . length ; i ++ ) { t [ i ] = new Array ( 2 ) ; }
function un_kp ( price , length , Max_len , n ) {
if ( n == 0 Max_len == 0 ) { return 0 ; }
if ( length [ n - 1 ] <= Max_len ) { t [ n ] [ Max_len ] = Math . max ( price [ n - 1 ] + un_kp ( price , length , Max_len - length [ n - 1 ] , n ) , un_kp ( price , length , Max_len , n - 1 ) ) ; }
else { t [ n ] [ Max_len ] = un_kp ( price , length , Max_len , n - 1 ) ; }
return t [ n ] [ Max_len ] ; }
let price = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] ; let n = price . length ; let length = Array ( n ) . fill ( 0 ) ; for ( let i = 0 ; i < n ; i ++ ) { length [ i ] = i + 1 ; } let Max_len = n ;
document . write ( " " + un_kp ( price , length , n , Max_len ) ) ;
function Convert ( radian ) { let pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
let radian = 5.0 ; let degree = Convert ( radian ) ; document . write ( degree ) ;
function subtract ( x , y ) {
while ( y != 0 ) {
let borrow = ( ~ x ) & y ;
x = x ^ y ;
y = borrow << 1 ; } return x ; }
let x = 29 , y = 13 ; document . write ( " " + subtract ( x , y ) ) ;
function subtract ( x , y ) { if ( y == 0 ) return x ; return subtract ( x ^ y , ( ~ x & y ) << 1 ) ; }
var x = 29 , y = 13 ; document . write ( " " + subtract ( x , y ) ) ;
function reverse ( str , len ) { if ( len == str . length ) { return ; } reverse ( str , len + 1 ) ; document . write ( str [ len ] ) ; }
let a = " " ; reverse ( a , 0 ) ;
let cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
function Kroneckerproduct ( A , B ) { let C = new Array ( rowa * rowb ) for ( let i = 0 ; i < ( rowa * rowb ) ; i ++ ) { C [ i ] = new Array ( cola * colb ) ; for ( let j = 0 ; j < ( cola * colb ) ; j ++ ) { C [ i ] [ j ] = 0 ; } }
for ( let i = 0 ; i < rowa ; i ++ ) {
for ( let k = 0 ; k < rowb ; k ++ ) {
for ( let j = 0 ; j < cola ; j ++ ) {
for ( let l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; document . write ( C [ i + l + 1 ] [ j + k + 1 ] + " " ) ; } } document . write ( " " ) ; } } }
let A = [ [ 1 , 2 ] , [ 3 , 4 ] , [ 1 , 0 ] ] ; let B = [ [ 0 , 5 , 2 ] , [ 6 , 7 , 3 ] ] ; Kroneckerproduct ( A , B ) ;
function MatrixChainOrder ( p , n ) {
var m = Array ( n ) . fill ( 0 ) . map ( x => Array ( n ) . fill ( 0 ) ) ; var i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i ] [ i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; if ( j == n ) continue ; m [ i ] [ j ] = Number . MAX_VALUE ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i ] [ j ] ) m [ i ] [ j ] = q ; } } } return m [ 1 ] [ n - 1 ] ; }
var arr = [ 1 , 2 , 3 , 4 ] ; var size = arr . length ; document . write ( " " + MatrixChainOrder ( arr , size ) ) ;
let t = new Array ( 9 ) ; for ( var i = 0 ; i < t . length ; i ++ ) { t [ i ] = new Array ( 2 ) ; }
function un_kp ( price , length , Max_len , n ) {
if ( n == 0 Max_len == 0 ) { return 0 ; }
if ( length [ n - 1 ] <= Max_len ) { t [ n ] [ Max_len ] = Math . max ( price [ n - 1 ] + un_kp ( price , length , Max_len - length [ n - 1 ] , n ) , un_kp ( price , length , Max_len , n - 1 ) ) ; }
else { t [ n ] [ Max_len ] = un_kp ( price , length , Max_len , n - 1 ) ; }
return t [ n ] [ Max_len ] ; }
let price = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] ; let n = price . length ; let length = Array ( n ) . fill ( 0 ) ; for ( let i = 0 ; i < n ; i ++ ) { length [ i ] = i + 1 ; } let Max_len = n ;
document . write ( " " + un_kp ( price , length , n , Max_len ) ) ;
function multiply ( x , y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; }
document . write ( multiply ( 5 , - 11 ) ) ;
function printPascal ( n ) {
arr = a = Array ( n ) . fill ( 0 ) . map ( x => Array ( n ) . fill ( 0 ) ) ;
for ( line = 0 ; line < n ; line ++ ) {
for ( i = 0 ; i <= line ; i ++ ) {
if ( line == i i == 0 ) arr [ line ] [ i ] = 1 ; else
arr [ line ] [ i ] = arr [ line - 1 ] [ i - 1 ] + arr [ line - 1 ] [ i ] ; document . write ( arr [ line ] [ i ] ) ; } document . write ( " " ) ; } }
var n = 5 ; printPascal ( n ) ;
function printPascal ( n ) { for ( line = 1 ; line <= n ; line ++ ) {
var C = 1 ; for ( i = 1 ; i <= line ; i ++ ) {
document . write ( C + " " ) ; C = C * ( line - i ) / i ; } document . write ( " " ) ; } }
var n = 5 ; printPascal ( n ) ;
function Add ( x , y ) {
while ( y != 0 ) {
let carry = x & y ;
x = x ^ y ;
y = carry << 1 ; } return x ; }
document . write ( Add ( 15 , 32 ) ) ;
function Add ( x , y ) { if ( y == 0 ) return x ; else return Add ( x ^ y , ( x & y ) << 1 ) ; }
function countSetBits ( n ) { var count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
var i = 9 ; document . write ( countSetBits ( i ) ) ;
var num_to_bits = [ 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 ] ;
function countSetBitsRec ( num ) { var nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
var num = 31 ; document . write ( countSetBitsRec ( num ) ) ;
function countSetBits ( N ) { var count = 0 ;
for ( i = 0 ; i < 4 * 8 ; i ++ ) { if ( ( N & ( 1 << i ) ) != 0 ) count ++ ; } return count ; }
var N = 15 ; document . write ( countSetBits ( N ) ) ;
function getParity ( n ) { var parity = false ; while ( n != 0 ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
var n = 7 ; document . write ( " " + n + " " + ( getParity ( n ) ? " " : " " ) ) ;
function isPowerOfTwo ( n ) { if ( n == 0 ) return false ; return parseInt ( ( Math . ceil ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) ) == parseInt ( ( Math . floor ( ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) ) ) ; }
if ( isPowerOfTwo ( 31 ) ) document . write ( " " ) ; else document . write ( " " ) ; if ( isPowerOfTwo ( 64 ) ) document . write ( " " ) ; else document . write ( " " ) ;
function isPowerOfTwo ( n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 2 != 0 ) return 0 ; n = n / 2 ; } return 1 ; }
isPowerOfTwo ( 31 ) ? document . write ( " " + " " ) : document . write ( " " + " " ) ; isPowerOfTwo ( 64 ) ? document . write ( " " ) : document . write ( " " ) ;
function isPowerOfTwo ( x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
document . write ( isPowerOfTwo ( 31 ) ? " " : " " ) ; document . write ( " " + ( isPowerOfTwo ( 64 ) ? " " : " " ) ) ;
function printTwoOdd ( arr , size ) {
let xor2 = arr [ 0 ] ;
let set_bit_no ; let i ; int n = size - 2 ; let x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( ( arr [ i ] & set_bit_no ) > 0 ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } document . write ( " " + x + " " + y + " " ) ; }
let arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 ] ; let arr_size = arr . length ; printTwoOdd ( arr , arr_size ) ;
function findPair ( arr , size , n ) {
let i = 0 ; let j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { document . write ( " " + arr [ i ] + " " + arr [ j ] + " " ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } document . write ( " " ) ; return false ; }
let arr = [ 1 , 8 , 30 , 40 , 100 ] ; let size = arr . length ; let n = 60 ; findPair ( arr , size , n ) ;
function MatrixChainOrder ( p , i , j ) { if ( i == j ) return 0 ; var min = Number . MAX_VALUE ;
var k = 0 ; for ( k = i ; k < j ; k ++ ) { var count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
var arr = [ 1 , 2 , 3 , 4 , 3 ] ; var n = arr . length ; document . write ( " " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ;
function Perimeter ( s , n ) { var perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
var n = 5 ;
var s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; document . write ( " " + " " + n + " " + s . toFixed ( 6 ) + " " + peri . toFixed ( 6 ) ) ;
function shortest_distance ( x1 , y1 , z1 , a , b , c , d ) { d = Math . abs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; let e = Math . sqrt ( a * a + b * b + c * c ) ; document . write ( " " + ( d / e ) ) ; return ; }
let x1 = 4 ; let y1 = - 4 ; let z1 = 3 ; let a = 2 ; let b = - 2 ; let c = 5 ; let d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ;
function averageOdd ( n ) { if ( n % 2 == 0 ) { document . write ( " " ) ; return - 1 ; } return ( n + 1 ) / 2 ; }
let n = 15 ; document . write ( averageOdd ( n ) ) ;
var MAX = 10
function TrinomialValue ( dp , n , k ) {
if ( k < 0 ) k = - k ;
if ( dp [ n ] [ k ] != 0 ) return dp [ n ] [ k ] ;
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return ( dp [ n ] [ k ] = TrinomialValue ( dp , n - 1 , k - 1 ) + TrinomialValue ( dp , n - 1 , k ) + TrinomialValue ( dp , n - 1 , k + 1 ) ) ; }
function printTrinomial ( n ) { var dp = Array . from ( Array ( MAX ) , ( ) => Array ( MAX ) . fill ( 0 ) ) ;
for ( var i = 0 ; i < n ; i ++ ) {
for ( var j = - i ; j <= 0 ; j ++ ) document . write ( TrinomialValue ( dp , i , j ) + " " ) ;
for ( var j = 1 ; j <= i ; j ++ ) document . write ( TrinomialValue ( dp , i , j ) + " " ) ; document . write ( " " ) ; } }
var n = 4 ; printTrinomial ( n ) ;
function averageEven ( n ) { if ( n % 2 != 0 ) { document . write ( " " ) ; return - 1 ; } return ( n + 2 ) / 2 ; }
let n = 16 ; document . write ( averageEven ( n ) ) ;
function fact ( n ) { if ( n == 0 ) return 1 ; return n * fact ( n - 1 ) ; }
function div ( x ) { let ans = 0 ; for ( let i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
function sumFactDiv ( n ) { return div ( fact ( n ) ) ; }
let n = 4 ; document . write ( sumFactDiv ( n ) ) ;
function printDivisors ( n ) { for ( var i = 1 ; i * i < n ; i ++ ) { if ( n % i == 0 ) document . write ( i + " " ) ; } for ( var i = Math . sqrt ( n ) ; i >= 1 ; i -- ) { if ( n % i == 0 ) document . write ( " " + n / i ) ; } }
document . write ( " " ) ; printDivisors ( 100 ) ;
function printDivisors ( n ) { for ( i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) document . write ( i + " " ) ; }
document . write ( " " + " " ) ; printDivisors ( 100 ) ;
function printDivisors ( n ) {
for ( let i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( parseInt ( n / i , 10 ) == i ) document . write ( i ) ;
else document . write ( i + " " + parseInt ( n / i , 10 ) + " " ) ; } } }
document . write ( " " ) ; printDivisors ( 100 ) ;
var rev_num = 0 ; var base_pos = 1 ; function reversDigits ( num ) { if ( num > 0 ) { reversDigits ( Math . floor ( num / 10 ) ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
let num = 4562 ; document . write ( " " + reversDigits ( num ) ) ;
function multiplyBySevenByEight ( n ) {
return ( n - ( n >> 3 ) ) ; }
let n = 9 ; document . write ( multiplyBySevenByEight ( n ) ) ;
function multiplyBySevenByEight ( n ) {
return ( ( n << 3 ) - n ) >> 3 ; }
var n = 15 ; document . write ( multiplyBySevenByEight ( n ) ) ;
function binarySearch ( a , item , low , high ) { while ( low <= high ) { var mid = low + ( high - low ) / 2 ; if ( item == a [ mid ] ) return mid + 1 ; else if ( item > a [ mid ] ) low = mid + 1 ; else high = mid - 1 ; } return low ; }
function insertionSort ( a , n ) { var i , loc , j , k , selected ; for ( i = 1 ; i < n ; ++ i ) { j = i - 1 ; selected = a [ i ] ;
loc = binarySearch ( a , selected , 0 , j ) ;
while ( j >= loc ) { a [ j + 1 ] = a [ j ] ; j -- ; } a [ j + 1 ] = selected ; } }
var a = [ 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 ] ; var n = a . length , i ; insertionSort ( a , n ) ; document . write ( " " + " " ) ; for ( i = 0 ; i < n ; i ++ ) document . write ( a [ i ] + " " ) ;
function insertionSort ( arr , n ) { let i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
function printArray ( arr , n ) { let i ; for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; }
let arr = [ 12 , 11 , 13 , 5 , 6 ] ; let n = arr . length ; insertionSort ( arr , n ) ; printArray ( arr , n ) ;
function count ( S , m , n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
var arr = [ 1 , 2 , 3 ] ; var m = arr . length ; document . write ( count ( arr , m , 4 ) ) ;
function Area ( b1 , b2 , h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
let base1 = 8 , base2 = 10 , height = 6 ; let area = Area ( base1 , base2 , height ) ; document . write ( " " + area ) ;
