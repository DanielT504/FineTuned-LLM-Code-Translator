function count_setbit ( N ) {
let result = 0 ;
for ( let i = 0 ; i < 32 ; i ++ ) {
if ( ( ( 1 << i ) & N ) > 0 ) {
result ++ ; } } document . write ( result ) ; }
let N = 43 ; count_setbit ( N ) ;
function isPowerOfTwo ( n ) { return ( Math . ceil ( Math . log ( n ) / Math . log ( 2 ) ) == Math . floor ( Math . log ( n ) / Math . log ( 2 ) ) ) ; }
let N = 8 ; if ( isPowerOfTwo ( N ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
class Cantor { constructor ( ) { this . start = 0 ; this . end = 0 ; this . next = null ; } } ; var cantor = null ;
function startList ( head , start_num , end_num ) { if ( head == null ) { head = new Cantor ( ) ; head . start = start_num ; head . end = end_num ; head . next = null ; } return head ; }
function propagate ( head ) { var temp = head ; if ( temp != null ) { var newNode = new Cantor ( ) ; var diff = ( ( ( temp . end ) - ( temp . start ) ) / 3 ) ;
newNode . end = temp . end ; temp . end = ( ( temp . start ) + diff ) ; newNode . start = ( newNode . end ) - diff ;
newNode . next = temp . next ; temp . next = newNode ;
propagate ( temp . next . next ) ; } return head ; }
function print ( temp ) { while ( temp != null ) { document . write ( " " + temp . start . toFixed ( 6 ) + " " + temp . end . toFixed ( 6 ) + " " ) ; temp = temp . next ; } document . write ( " " ) ; }
function buildCantorSet ( A , B , L ) { var head = null ; head = startList ( head , A , B ) ; for ( var i = 0 ; i < L ; i ++ ) { document . write ( " " + i + " " ) ; print ( head ) ; propagate ( head ) ; } document . write ( " " + L + " " ) ; print ( head ) ; }
var A = 0 ; var B = 9 ; var L = 2 ; buildCantorSet ( A , B , L ) ;
function search ( pat , txt ) { let M = pat . length ; let N = txt . length ; let i = 0 ; while ( i <= N - M ) { let j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) { document . write ( " " + i + " " ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
let txt = " " ; let pat = " " ; search ( pat , txt ) ;
function encrypt ( input ) {
let evenPos = ' ' , oddPos = ' ' ; let repeat , ascii ; for ( let i = 0 ; i < input . length ; i ++ ) {
ascii = input [ i ] . charCodeAt ( ) ; repeat = ascii >= 97 ? ascii - 96 : ascii - 64 ; for ( let j = 0 ; j < repeat ; j ++ ) {
if ( i % 2 == 0 ) document . write ( oddPos ) ; else document . write ( evenPos ) ; } } }
let input = [ ' ' , ' ' , ' ' , ' ' ] ;
encrypt ( input ) ;
function isPalRec ( str , s , e ) {
if ( s == e ) return true ;
if ( ( str . charAt ( s ) ) != ( str . charAt ( e ) ) ) return false ;
if ( s < e + 1 ) return isPalRec ( str , s + 1 , e - 1 ) ; return true ; } function isPalindrome ( str ) { var n = str . length ;
if ( n == 0 ) return true ; return isPalRec ( str , 0 , n - 1 ) ; }
var str = " " ; if ( isPalindrome ( str ) ) document . write ( " " ) ; else document . write ( " " ) ;
function myAtoi ( str ) { var sign = 1 , base = 0 , i = 0 ;
while ( str [ i ] == ' ' ) { i ++ ; }
if ( str [ i ] == ' ' str [ i ] == ' ' ) { sign = 1 - 2 * ( str [ i ++ ] == ' ' ) ; }
while ( str [ i ] >= ' ' && str [ i ] <= ' ' ) {
if ( base > Number . MAX_VALUE / 10 || ( base == Number . MAX_VALUE / 10 && str [ i ] - ' ' > 7 ) ) { if ( sign == 1 ) return Number . MAX_VALUE ; else return Number . MAX_VALUE ; } base = 10 * base + ( str [ i ++ ] - ' ' ) ; } return base * sign ; }
var str = " " ;
var val = myAtoi ( str ) ; document . write ( " " , val ) ;
function fillUtil ( res , curr , n ) {
if ( curr == 0 ) return true ;
let i ; for ( i = 0 ; i < 2 * n - curr - 1 ; i ++ ) {
if ( res [ i ] == 0 && res [ i + curr + 1 ] == 0 ) {
res [ i ] = res [ i + curr + 1 ] = curr ;
if ( fillUtil ( res , curr - 1 , n ) ) return true ;
res [ i ] = res [ i + curr + 1 ] = 0 ; } } return false ; }
function fill ( n ) {
let res = new Array ( 2 * n ) ; let i ; for ( i = 0 ; i < ( 2 * n ) ; i ++ ) res [ i ] = 0 ;
if ( fillUtil ( res , n , n ) ) { for ( i = 0 ; i < 2 * n ; i ++ ) document . write ( res [ i ] + " " ) ; } else document . write ( " " ) ; }
fill ( 7 ) ;
function findNumberOfDigits ( n , base ) {
var dig = ( Math . floor ( Math . log ( n ) / Math . log ( base ) ) + 1 ) ;
return ( dig ) ; }
function isAllKs ( n , b , k ) { var len = findNumberOfDigits ( n , b ) ;
var sum = k * ( 1 - Math . pow ( b , len ) ) / ( 1 - b ) ; if ( sum == n ) { return ( sum ) ; } }
var N = 13 ;
var B = 3 ;
var K = 1 ;
if ( isAllKs ( N , B , K ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function CalPeri ( ) { var S = 5 , Perimeter ; Perimeter = 10 * S ; document . write ( " " + Perimeter ) ; }
CalPeri ( ) ;
function distance ( a1 , b1 , c1 , a2 , b2 , c2 ) { var d = a1 * a2 + b1 * b2 + c1 * c2 ; var e1 = Math . sqrt ( a1 * a1 + b1 * b1 + c1 * c1 ) ; var e2 = Math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ; d = parseFloat ( d / ( e1 * e2 ) ) ; var pi = 3.14159 ; var A = ( 180 / pi ) * Math . acos ( d ) ; document . write ( " " + A . toFixed ( 1 ) + " " ) ; }
var a1 = 1 ; var b1 = 1 ; var c1 = 2 ; var d1 = 1 ; var a2 = 2 ; var b2 = - 1 ; var c2 = 1 ; var d2 = - 4 ; distance ( a1 , b1 , c1 , a2 , b2 , c2 ) ;
function mirror_point ( a , b , c , d , x1 , y1 , z1 ) { var k = parseFloat ( ( - a * x1 - b * y1 - c * z1 - d ) / parseFloat ( a * a + b * b + c * c ) ) ; var x2 = parseFloat ( a * k + x1 ) ; var y2 = parseFloat ( b * k + y1 ) ; var z2 = parseFloat ( c * k + z1 ) ; var x3 = parseFloat ( 2 * x2 - x1 ) . toFixed ( 1 ) ; var y3 = parseFloat ( 2 * y2 - y1 ) . toFixed ( 1 ) ; var z3 = parseFloat ( 2 * z2 - z1 ) . toFixed ( 1 ) ; document . write ( " " + x3 ) ; document . write ( " " + y3 ) ; document . write ( " " + z3 ) ; }
var a = 1 ; var b = - 2 ; var c = 0 ; var d = 0 ; var x1 = - 1 ; var y1 = 3 ; var z1 = 4 ;
mirror_point ( a , b , c , d , x1 , y1 , z1 ) ;
function calculateSpan ( price , n , S ) {
S [ 0 ] = 1 ;
for ( let i = 1 ; i < n ; i ++ ) {
S [ i ] = 1 ;
for ( let j = i - 1 ; ( j >= 0 ) && ( price [ i ] >= price [ j ] ) ; j -- ) S [ i ] ++ ; } }
function printArray ( arr ) { let result = arr . join ( " " ) ; document . write ( result ) ; }
let price = [ 10 , 4 , 5 , 90 , 120 , 80 ] ; let n = price . length ; let S = new Array ( n ) ; S . fill ( 0 ) ;
calculateSpan ( price , n , S ) ;
printArray ( S ) ;
function printNGE ( arr , n ) { var next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = - 1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] < arr [ j ] ) { next = arr [ j ] ; break ; } } document . write ( arr [ i ] + " " + next ) ; document . write ( " " ) ; } }
var arr = [ 11 , 13 , 21 , 3 ] ; var n = arr . length ; printNGE ( arr , n ) ;
function gcd ( a , b ) {
if ( a == 0 && b == 0 ) return 0 ; if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
var a = 98 , b = 56 ; document . write ( " " + a + " " + b + " " + gcd ( a , b ) ) ;
function msbPos ( n ) { var pos = 0 ; while ( n != 0 ) { pos ++ ;
n = n >> 1 ; } return pos ; }
function josephify ( n ) {
var position = msbPos ( n ) ;
var j = 1 << ( position - 1 ) ;
n = n ^ j ;
n = n << 1 ;
n = n | 1 ; return n ; }
var n = 41 ; document . write ( josephify ( n ) ) ;
function pairAndSum ( arr , n ) {
for ( let i = 0 ; i < 32 ; i ++ ) {
let k = 0 ; for ( let j = 0 ; j < n ; j ++ ) { if ( ( arr [ j ] & ( 1 << i ) ) != 0 ) k ++ ; }
ans += ( 1 << i ) * ( k * ( k - 1 ) / 2 ) ; } return ans ; }
let arr = [ 5 , 10 , 15 ] ; let n = arr . length ; document . write ( pairAndSum ( arr , n ) ) ;
function countSquares ( n ) {
return ( n * ( n + 1 ) / 2 ) * ( 2 * n + 1 ) / 3 ; }
let n = 4 ; document . write ( " " + countSquares ( n ) ) ;
function gcd ( a , b ) {
if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
let a = 98 , b = 56 ; document . write ( " " + a + " " + b + " " + gcd ( a , b ) ) ;
var maxsize = 100005 ;
var xor_tree = Array ( maxsize ) ;
function construct_Xor_Tree_Util ( current , start , end , x ) {
if ( start == end ) { xor_tree [ x ] = current [ start ] ;
return ; }
var left = x * 2 + 1 ;
var right = x * 2 + 2 ;
var mid = start + parseInt ( ( end - start ) / 2 ) ;
construct_Xor_Tree_Util ( current , start , mid , left ) ; construct_Xor_Tree_Util ( current , mid + 1 , end , right ) ;
xor_tree [ x ] = ( xor_tree [ left ] ^ xor_tree [ right ] ) ; }
function construct_Xor_Tree ( arr , n ) { construct_Xor_Tree_Util ( arr , 0 , n - 1 , 0 ) ; }
var leaf_nodes = [ 40 , 32 , 12 , 1 , 4 , 3 , 2 , 7 ] ; var n = leaf_nodes . length ;
construct_Xor_Tree ( leaf_nodes , n ) ;
var x = ( Math . ceil ( Math . log2 ( n ) ) ) ;
var max_size = 2 * Math . pow ( 2 , x ) - 1 ; document . write ( " " ) ; for ( var i = 0 ; i < max_size ; i ++ ) { document . write ( xor_tree [ i ] + " " ) ; }
var root = 0 ;
document . write ( " " + xor_tree [ root ] ) ;
function swapBits ( n , p1 , p2 ) {
n ^= 1 << p1 ; n ^= 1 << p2 ; return n ; }
document . write ( " " + swapBits ( 28 , 0 , 3 ) ) ;
class Node { constructor ( item ) { this . data = item ; this . left = this . right = null ; } } var root ;
function isFullTree ( node ) {
if ( node == null ) return true ;
if ( node . left == null && node . right == null ) return true ;
if ( ( node . left != null ) && ( node . right != null ) ) return ( isFullTree ( node . left ) && isFullTree ( node . right ) ) ;
return false ; }
root = new Node ( 10 ) ; root . left = new Node ( 20 ) ; root . right = new Node ( 30 ) ; root . left . right = new Node ( 40 ) ; root . left . left = new Node ( 50 ) ; root . right . left = new Node ( 60 ) ; root . left . left . left = new Node ( 80 ) ; root . right . right = new Node ( 70 ) ; root . left . left . right = new Node ( 90 ) ; root . left . right . left = new Node ( 80 ) ; root . left . right . right = new Node ( 90 ) ; root . right . left . left = new Node ( 80 ) ; root . right . left . right = new Node ( 90 ) ; root . right . right . left = new Node ( 80 ) ; root . right . right . right = new Node ( 90 ) ; if ( isFullTree ( root ) ) document . write ( " " ) ; else document . write ( " " ) ;
function printAlter ( arr , N ) {
for ( var currIndex = 0 ; currIndex < N ; currIndex += 2 ) {
document . write ( arr [ currIndex ] + " " ) ; } }
var arr = [ 1 , 2 , 3 , 4 , 5 ] ; var N = 5 ; printAlter ( arr , N ) ;
let leftRotate = ( arr , d , n ) => {
if ( d == 0 d == n ) return ;
if ( n - d == d ) { arr = swap ( arr , 0 , n - d , d ) ; return ; }
if ( d < n - d ) { arr = swap ( arr , 0 , n - d , d ) ; leftRotate ( arr , d , n - d ) ; }
else { arr = swap ( arr , 0 , d , n - d ) ; leftRotate ( arr + n - d , 2 * d - n , d ) ; } }
let printArray = ( arr , size ) => { ans = ' ' for ( let i = 0 ; i < size ; i ++ ) ans += arr [ i ] + " " ; document . write ( ans ) }
let swap = ( arr , fi , si , d ) => { for ( let i = 0 ; i < d ; i ++ ) { let temp = arr [ fi + i ] ; arr [ fi + i ] = arr [ si + i ] ; arr [ si + i ] = temp ; } return arr }
arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ;
function leftRotate ( arr , d , n ) { if ( d == 0 d == n ) return ; let i = d ; let j = n - d ; while ( i != j ) {
if ( i < j ) { arr = swap ( arr , d - i , d + j - i , i ) ; j -= i ; }
else { arr = swap ( arr , d - i , d , j ) ; i -= j ; } }
arr = swap ( arr , d - i , d , i ) ; }
function selectionSort ( arr , n ) {
for ( let i = 0 ; i < n - 1 ; i ++ ) {
let min_index = i ; let minStr = arr [ i ] ; for ( let j = i + 1 ; j < n ; j ++ ) {
if ( ( arr [ j ] ) . localeCompare ( minStr ) === - 1 ) {
minStr = arr [ j ] ; min_index = j ; } }
if ( min_index != i ) { let temp = arr [ min_index ] ; arr [ min_index ] = arr [ i ] ; arr [ i ] = temp ; } } }
let arr = [ " " , " " , " " ] ; let n = arr . length ; document . write ( " " + " " ) ;
for ( let i = 0 ; i < n ; i ++ ) { document . write ( i + " " + arr [ i ] + " " ) ; } document . write ( " " ) ; selectionSort ( arr , n ) ; document . write ( " " + " " ) ;
for ( let i = 0 ; i < n ; i ++ ) { document . write ( i + " " + arr [ i ] + " " ) ; }
function rearrangeNaive ( arr , n ) {
let temp = new Array ( n ) , i ;
for ( i = 0 ; i < n ; i ++ ) temp [ arr [ i ] ] = i ;
for ( i = 0 ; i < n ; i ++ ) arr [ i ] = temp [ i ] ; }
function printArray ( arr , n ) { let i ; for ( i = 0 ; i < n ; i ++ ) document . write ( " " + arr [ i ] ) ; document . write ( " " ) ; }
let arr = [ 1 , 3 , 0 , 2 ] ; let n = arr . length ; document . write ( " " ) ; printArray ( arr , n ) ; rearrangeNaive ( arr , n ) ; document . write ( " " ) ; printArray ( arr , n ) ;
function largest ( arr ) { let i ;
let max = arr [ 0 ] ;
for ( i = 1 ; i < arr . length ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; } return max ; }
let arr = [ 10 , 324 , 45 , 90 , 9808 ] ; document . write ( " " + largest ( arr ) ) ;
function print2largest ( arr , arr_size ) { let i ; let largest = second = - 2454635434 ;
if ( arr_size < 2 ) { document . write ( " " ) ; return ; } for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] > largest ) { second = largest ; largest = arr [ i ] }
else if ( arr [ i ] != largest && arr [ i ] > second ) { second = arr [ i ] ; } } if ( second == - 2454635434 ) { document . write ( " " ) ; } else { document . write ( " " + second ) ; return ; } }
let arr = [ 12 , 35 , 1 , 10 , 34 , 1 ] ; let n = arr . length ; print2largest ( arr , n ) ;
function minJumps ( arr , n ) {
if ( n == 1 ) return 0 ;
let res = Number . MAX_VALUE ; for ( let i = n - 2 ; i >= 0 ; i -- ) { if ( i + arr [ i ] >= n - 1 ) { let sub_res = minJumps ( arr , i + 1 ) ; if ( sub_res != Number . MAX_VALUE ) res = Math . min ( res , sub_res + 1 ) ; } } return res ; }
let arr = [ 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 ] ; let n = arr . length ; document . write ( " " ) ; document . write ( " " + minJumps ( arr , n ) ) ;
function smallestSubWithSum ( arr , n , x ) {
let min_len = n + 1 ;
for ( let start = 0 ; start < n ; start ++ ) {
let curr_sum = arr [ start ] ;
if ( curr_sum > x ) return 1 ;
for ( let end = start + 1 ; end < n ; end ++ ) {
curr_sum += arr [ end ] ;
if ( curr_sum > x && ( end - start + 1 ) < min_len ) min_len = ( end - start + 1 ) ; } } return min_len ; }
let arr1 = [ 1 , 4 , 45 , 6 , 10 , 19 ] ; let x = 51 ; let n1 = arr1 . length ; let res1 = smallestSubWithSum ( arr1 , n1 , x ) ; ( res1 == n1 + 1 ) ? document . write ( " " ) : document . write ( res1 + " " ) ; let arr2 = [ 1 , 10 , 5 , 2 , 7 ] ; let n2 = arr2 . length ; x = 9 ; let res2 = smallestSubWithSum ( arr2 , n2 , x ) ; ( res2 == n2 + 1 ) ? document . write ( " " ) : document . write ( res2 + " " ) ; let arr3 = [ 1 , 11 , 100 , 1 , 0 , 200 , 3 , 2 , 1 , 250 ] ; let n3 = arr3 . length ; x = 280 ; let res3 = smallestSubWithSum ( arr3 , n3 , x ) ; ( res3 == n3 + 1 ) ? document . write ( " " ) : document . write ( res3 + " " ) ;
class Node { constructor ( val ) { this . key = val ; this . left = null ; this . right = null ; } }
function printPostorder ( node ) { if ( node == null ) return ;
printPostorder ( node . left ) ;
printPostorder ( node . right ) ;
document . write ( node . key + " " ) ; }
function printInorder ( node ) { if ( node == null ) return ;
printInorder ( node . left ) ;
document . write ( node . key + " " ) ;
printInorder ( node . right ) ; }
function printPreorder ( node ) { if ( node == null ) return ;
document . write ( node . key + " " ) ;
printPreorder ( node . left ) ;
printPreorder ( node . right ) ; }
root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . left . left = new Node ( 4 ) ; root . left . right = new Node ( 5 ) ; document . write ( " " ) ; printPreorder ( root ) ; document . write ( " " ) ; printInorder ( root ) ; document . write ( " " ) ; printPostorder ( root ) ;
function moveToEnd ( mPlusN , size ) { let i = 0 ; let j = size - 1 ; for ( i = size - 1 ; i >= 0 ; i -- ) { if ( mPlusN [ i ] != - 1 ) { mPlusN [ j ] = mPlusN [ i ] ; j -- ; } } }
function merge ( mPlusN , N , m , n ) { let i = n ;
let j = 0 ;
let k = 0 ;
while ( k < ( m + n ) ) {
if ( ( i < ( m + n ) && mPlusN [ i ] <= N [ j ] ) || ( j == n ) ) { mPlusN [ k ] = mPlusN [ i ] ; k ++ ; i ++ ; }
else { mPlusN [ k ] = N [ j ] ; k ++ ; j ++ ; } } }
function printArray ( arr , size ) { let i = 0 ; for ( i = 0 ; i < size ; i ++ ) { document . write ( arr [ i ] + " " ) ; } document . write ( " " ) ; }
let mPlusN = [ 2 , 8 , - 1 , - 1 , - 1 , 13 , - 1 , 15 , 20 ] ; let N = [ 5 , 7 , 9 , 25 ] let n = N . length ; let m = mPlusN . length - n ;
moveToEnd ( mPlusN , m + n ) ;
merge ( mPlusN , N , m , n ) ;
printArray ( mPlusN , m + n ) ;
function max ( num1 , num2 ) { return ( num1 > num2 ) ? num1 : num2 ; }
function min ( num1 , num2 ) { return ( num1 > num2 ) ? num2 : num1 ; }
function getCount ( n , k ) {
if ( n == 1 ) return 10 ;
var dp = Array ( 11 ) . fill ( 0 ) ;
var next = Array ( 11 ) . fill ( 0 ) ;
for ( var i = 1 ; i <= 9 ; i ++ ) dp [ i ] = 1 ;
for ( var i = 2 ; i <= n ; i ++ ) { for ( var j = 0 ; j <= 9 ; j ++ ) {
var l = Math . max ( 0 , ( j - k ) ) ; var r = Math . min ( 9 , ( j + k ) ) ;
next [ l ] += dp [ j ] ; next [ r + 1 ] -= dp [ j ] ; }
for ( var j = 1 ; j <= 9 ; j ++ ) next [ j ] += next [ j - 1 ] ;
for ( var j = 0 ; j < 10 ; j ++ ) { dp [ j ] = next [ j ] ; next [ j ] = 0 ; } }
var count = 0 ; for ( var i = 0 ; i <= 9 ; i ++ ) count += dp [ i ] ;
return count ; }
var n = 2 , k = 1 ; document . write ( getCount ( n , k ) ) ;
function getInvCount ( arr ) { let inv_count = 0 ; for ( let i = 0 ; i < arr . length - 1 ; i ++ ) { for ( let j = i + 1 ; j < arr . length ; j ++ ) { if ( arr [ i ] > arr [ j ] ) inv_count ++ ; } } return inv_count ; }
arr = [ 1 , 20 , 6 , 4 , 5 ] ; document . write ( " " + getInvCount ( arr ) ) ;
function minAbsSumPair ( arr , arr_size ) { var inv_count = 0 ; var l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { document . write ( " " ) ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( Math . abs ( min_sum ) > Math . abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } document . write ( " " + arr [ min_l ] + " " + arr [ min_r ] ) ; }
arr = new Array ( 1 , 60 , - 10 , 70 , - 80 , 85 ) ; minAbsSumPair ( arr , 6 ) ;
function sort012 ( a , arr_size ) { let lo = 0 ; let hi = arr_size - 1 ; let mid = 0 ; let temp = 0 ; while ( mid <= hi ) { if ( a [ mid ] == 0 ) { temp = a [ lo ] ; a [ lo ] = a [ mid ] ; a [ mid ] = temp ; lo ++ ; mid ++ ; } else if ( a [ mid ] == 1 ) { mid ++ ; } else { temp = a [ mid ] ; a [ mid ] = a [ hi ] ; a [ hi ] = temp ; hi -- ; } } }
function printArray ( arr , arr_size ) { let i ; for ( i = 0 ; i < arr_size ; i ++ ) { document . write ( arr [ i ] + " " ) ; } document . write ( " " ) ; }
let arr = [ 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 ] ; let arr_size = arr . length ; sort012 ( arr , arr_size ) ; document . write ( " " ) printArray ( arr , arr_size ) ;
function printUnsorted ( arr , n ) { let s = 0 , e = n - 1 , i , max , min ;
for ( s = 0 ; s < n - 1 ; s ++ ) { if ( arr [ s ] > arr [ s + 1 ] ) break ; } if ( s == n - 1 ) { document . write ( " " ) ; return ; }
for ( e = n - 1 ; e > 0 ; e -- ) { if ( arr [ e ] < arr [ e - 1 ] ) break ; }
max = arr [ s ] ; min = arr [ s ] ; for ( i = s + 1 ; i <= e ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; if ( arr [ i ] < min ) min = arr [ i ] ; }
for ( i = 0 ; i < s ; i ++ ) { if ( arr [ i ] > min ) { s = i ; break ; } }
for ( i = n - 1 ; i >= e + 1 ; i -- ) { if ( arr [ i ] < max ) { e = i ; break ; } }
document . write ( " " + " " + " " + s + " " + e ) ; return ; } let arr = [ 10 , 12 , 20 , 30 , 25 , 40 , 32 , 31 , 35 , 50 , 60 ] ; let arr_size = arr . length ; printUnsorted ( arr , arr_size ) ;
function findNumberOfTriangles ( arr ) { let n = arr . length ;
arr . sort ( ( a , b ) => a - b ) ;
let count = 0 ;
for ( let i = 0 ; i < n - 2 ; ++ i ) {
let k = i + 2 ;
for ( let j = i + 1 ; j < n ; ++ j ) {
while ( k < n && arr [ i ] + arr [ j ] > arr [ k ] ) ++ k ;
if ( k > j ) count += k - j - 1 ; } } return count ; }
let arr = [ 10 , 21 , 22 , 100 , 101 , 200 , 300 ] ; let size = arr . length ; document . write ( " " + findNumberOfTriangles ( arr , size ) ) ;
function findElement ( arr , n , key ) { let i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return - 1 ; }
let arr = [ 12 , 34 , 10 , 6 , 40 ] ; let n = arr . length ;
let key = 40 ; let position = findElement ( arr , n , key ) ; if ( position == - 1 ) document . write ( " " ) ; else document . write ( " " + ( position + 1 ) ) ;
function insertSorted ( arr , n , key , capacity ) {
if ( n >= capacity ) return n ; arr [ n ] = key ; return ( n + 1 ) ; }
let arr = new Array ( 20 ) ; arr [ 0 ] = 12 ; arr [ 1 ] = 16 ; arr [ 2 ] = 20 ; arr [ 3 ] = 40 ; arr [ 4 ] = 50 ; arr [ 5 ] = 70 ; let capacity = 20 ; let n = 6 ; let i , key = 26 ; document . write ( " " ) ; for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ;
n = insertSorted ( arr , n , key , capacity ) ; document . write ( " " ) ; for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ;
function findElement ( arr , n , key ) { let i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return - 1 ; }
function deleteElement ( arr , n , key ) {
let pos = findElement ( arr , n , key ) ; if ( pos == - 1 ) { document . write ( " " ) ; return n ; }
let i ; for ( i = pos ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; return n - 1 ; }
let i ; let arr = [ 10 , 50 , 30 , 40 , 20 ] ; let n = arr . length ; let key = 30 ; document . write ( " " ) ; for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; n = deleteElement ( arr , n , key ) ; document . write ( " " ) ; for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ;
function binarySearch ( arr , low , high , key ) { if ( high < low ) return - 1 ;
let mid = Math . trunc ( ( low + high ) / 2 ) ; if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
let arr = [ 5 , 6 , 7 , 8 , 9 , 10 ] ; let n , key ; n = arr . length ; key = 10 ; document . write ( " " + binarySearch ( arr , 0 , n - 1 , key ) + " " ) ;
function equilibrium ( arr , n ) { var i , j ; var leftsum , rightsum ;
for ( i = 0 ; i < n ; ++ i ) { leftsum = 0 ; rightsum = 0 ;
for ( let j = 0 ; j < i ; j ++ ) leftsum += arr [ j ] ;
for ( let j = i + 1 ; j < n ; j ++ ) rightsum += arr [ j ] ;
if ( leftsum == rightsum ) return i ; }
return - 1 ; }
var arr = new Array ( - 7 , 1 , 5 , 2 , - 4 , 3 , 0 ) ; n = arr . length ; document . write ( equilibrium ( arr , n ) ) ;
function equilibrium ( arr , n ) {
sum = 0 ;
leftsum = 0 ;
for ( let i = 0 ; i < n ; ++ i ) sum += arr [ i ] ; for ( let i = 0 ; i < n ; ++ i ) {
sum -= arr [ i ] ; if ( leftsum == sum ) return i ; leftsum += arr [ i ] ; }
return - 1 ; }
arr = new Array ( - 7 , 1 , 5 , 2 , - 4 , 3 , 0 ) ; n = arr . length ; document . write ( " " + equilibrium ( arr , n ) ) ;
function ceilSearch ( arr , low , high , x ) { let i ;
if ( x <= arr [ low ] ) return low ;
for ( i = low ; i < high ; i ++ ) { if ( arr [ i ] == x ) return i ;
if ( arr [ i ] < x && arr [ i + 1 ] >= x ) return i + 1 ; }
return - 1 ; }
let arr = [ 1 , 2 , 8 , 10 , 10 , 12 , 19 ] ; let n = arr . length ; let x = 3 ; let index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == - 1 ) document . write ( " " + x + " " ) ; else document . write ( " " + x + " " + arr [ index ] ) ;
function ceilSearch ( arr , low , high , x ) { let mid ;
if ( x <= arr [ low ] ) return low ;
if ( x > arr [ high ] ) return - 1 ;
mid = ( low + high ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
else if ( arr [ mid ] < x ) { if ( mid + 1 <= high && x <= arr [ mid + 1 ] ) return mid + 1 ; else return ceilSearch ( arr , mid + 1 , high , x ) ; }
else { if ( mid - 1 >= low && x > arr [ mid - 1 ] ) return mid ; else return ceilSearch ( arr , low , mid - 1 , x ) ; } }
let arr = [ 1 , 2 , 8 , 10 , 10 , 12 , 19 ] ; let n = arr . length ; let x = 20 ; let index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == - 1 ) { document . write ( ` ${ x } ` ) ; } else { document . write ( ` ${ x } ${ arr [ index ] } ` ) ; }
function isPairSum ( A , N , X ) {
var i = 0 ;
var j = N - 1 ; while ( i < j ) {
if ( A [ i ] + A [ j ] == X ) return true ;
else if ( A [ i ] + A [ j ] < X ) i ++ ;
else j -- ; } return false ; }
var arr = [ 3 , 5 , 9 , 2 , 8 , 10 , 11 ] ;
var val = 17 ;
var arrSize = 7 ;
document . write ( isPairSum ( arr , arrSize , val ) ) ;
const NUM_LINE = 2 ; const NUM_STATION = 4 ;
function min ( a , b ) { return a < b ? a : b ; } function carAssembly ( a , t , e , x ) { let T1 = new Array ( NUM_STATION ) ; let T2 = new Array ( NUM_STATION ) ; let i ;
T1 [ 0 ] = e [ 0 ] + a [ 0 ] [ 0 ] ;
T2 [ 0 ] = e [ 1 ] + a [ 1 ] [ 0 ] ;
for ( i = 1 ; i < NUM_STATION ; ++ i ) { T1 [ i ] = min ( T1 [ i - 1 ] + a [ 0 ] [ i ] , T2 [ i - 1 ] + t [ 1 ] [ i ] + a [ 0 ] [ i ] ) ; T2 [ i ] = min ( T2 [ i - 1 ] + a [ 1 ] [ i ] , T1 [ i - 1 ] + t [ 0 ] [ i ] + a [ 1 ] [ i ] ) ; }
return min ( T1 [ NUM_STATION - 1 ] + x [ 0 ] , T2 [ NUM_STATION - 1 ] + x [ 1 ] ) ; }
let a = [ [ 4 , 5 , 3 , 2 ] , [ 2 , 10 , 1 , 4 ] ] ; let t = [ [ 0 , 7 , 4 , 5 ] , [ 0 , 9 , 2 , 8 ] ] ; let e = [ 10 , 12 ] , x = [ 18 , 7 ] ; document . write ( carAssembly ( a , t , e , x ) ) ;
function findMinInsertionsDP ( str , n ) {
let table = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { table [ i ] = new Array ( n ) ; } for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) { table [ i ] [ j ] = 0 ; } } let l = 0 , h = 0 , gap = 0 ;
for ( gap = 1 ; gap < n ; gap ++ ) { for ( l = 0 , h = gap ; h < n ; l ++ , h ++ ) { table [ l ] [ h ] = ( str [ l ] == str [ h ] ) ? table [ l + 1 ] [ h - 1 ] : ( Math . min ( table [ l ] [ h - 1 ] , table [ l + 1 ] [ h ] ) + 1 ) ; } }
return table [ 0 ] [ n - 1 ] ; }
let str = " " ; document . write ( findMinInsertionsDP ( str , str . length ) ) ;
function max ( x , y ) { return ( x > y ) ? x : y ; }
class Node { constructor ( data ) { this . data = data ; this . left = this . right = null ; } }
function LISS ( root ) { if ( root == null ) return 0 ;
let size_excl = LISS ( root . left ) + LISS ( root . right ) ;
let size_incl = 1 ; if ( root . left != null ) size_incl += LISS ( root . left . left ) + LISS ( root . left . right ) ; if ( root . right != null ) size_incl += LISS ( root . right . left ) + LISS ( root . right . right ) ;
return max ( size_incl , size_excl ) ; }
let root = new Node ( 20 ) ; root . left = new Node ( 8 ) ; root . left . left = new Node ( 4 ) ; root . left . right = new Node ( 12 ) ; root . left . right . left = new Node ( 10 ) ; root . left . right . right = new Node ( 14 ) ; root . right = new Node ( 22 ) ; root . right . right = new Node ( 25 ) ; document . write ( " " + " " + LISS ( root ) ) ;
class Pair { constructor ( a , b ) { this . a = a ; this . b = b ; } }
function maxChainLength ( arr , n ) { let i , j , max = 0 ; let mcl = new Array ( n ) ;
for ( i = 0 ; i < n ; i ++ ) mcl [ i ] = 1 ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] . a > arr [ j ] . b && mcl [ i ] < mcl [ j ] + 1 ) mcl [ i ] = mcl [ j ] + 1 ;
for ( i = 0 ; i < n ; i ++ ) if ( max < mcl [ i ] ) max = mcl [ i ] ; return max ; }
let arr = [ new Pair ( 5 , 24 ) , new Pair ( 15 , 25 ) , new Pair ( 27 , 40 ) , new Pair ( 50 , 60 ) ] ; document . write ( " " + maxChainLength ( arr , arr . length ) ) ;
let NO_OF_CHARS = 256 ;
function max ( a , b ) { return ( a > b ) ? a : b ; }
function badCharHeuristic ( str , size , badchar ) {
for ( let i = 0 ; i < NO_OF_CHARS ; i ++ ) badchar [ i ] = - 1 ;
for ( i = 0 ; i < size ; i ++ ) badchar [ str [ i ] . charCodeAt ( 0 ) ] = i ; }
function search ( txt , pat ) { let m = pat . length ; let n = txt . length ; let badchar = new Array ( NO_OF_CHARS ) ;
badCharHeuristic ( pat , m , badchar ) ;
let s = 0 ; while ( s <= ( n - m ) ) { let j = m - 1 ;
while ( j >= 0 && pat [ j ] == txt [ s + j ] ) j -- ;
if ( j < 0 ) { document . write ( " " + s ) ;
s += ( s + m < n ) ? m - badchar [ txt [ s + m ] . charCodeAt ( 0 ) ] : 1 ; } else
s += max ( 1 , j - badchar [ txt [ s + j ] . charCodeAt ( 0 ) ] ) ; } }
let txt = " " . split ( " " ) ; let pat = " " . split ( " " ) ; search ( txt , pat ) ;
function minChocolates ( a , n ) { let i = 0 , j = 0 ; let res = 0 , val = 1 ; while ( j < n - 1 ) { if ( a [ j ] > a [ j + 1 ] ) {
j += 1 ; continue ; } if ( i == j )
res += val ; else {
res += get_sum ( val , i , j ) ;
} if ( a [ j ] < a [ j + 1 ] )
val += 1 ; else
val = 1 ; j += 1 ; i = j ; }
if ( i == j ) res += val ; else res += get_sum ( val , i , j ) ; return res ; }
function get_sum ( peak , start , end ) {
let count = end - start + 1 ;
peak = ( peak > count ) ? peak : count ;
let s = peak + ( ( ( count - 1 ) * count ) >> 1 ) ; return s ; }
let a = [ 5 , 5 , 4 , 3 , 2 , 1 ] ; let n = a . length ; document . write ( " " + minChocolates ( a , n ) ) ;
function sum ( n ) { let i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
let n = 5 ; document . write ( " " + sum ( n ) ) ;
function nthTermOfTheSeries ( n ) {
let nthTerm ;
if ( n % 2 == 0 ) nthTerm = Math . pow ( n - 1 , 2 ) + n ;
else nthTerm = Math . pow ( n + 1 , 2 ) + n ;
return nthTerm ; }
let n ; n = 8 ; document . write ( nthTermOfTheSeries ( n ) + " " ) ; n = 12 ; document . write ( nthTermOfTheSeries ( n ) + " " ) ; n = 102 ; document . write ( nthTermOfTheSeries ( n ) + " " ) ; n = 999 ; document . write ( nthTermOfTheSeries ( n ) + " " ) ; n = 9999 ; document . write ( nthTermOfTheSeries ( n ) + " " ) ;
function findAmount ( X , W , Y ) { return ( X * ( Y - W ) ) / ( 100 - Y ) ; }
let X = 100 , W = 50 , Y = 60 ; document . write ( " " + findAmount ( X , W , Y ) . toFixed ( 2 ) ) ;
function AvgofSquareN ( n ) { return ( ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ; }
var n = 2 ; document . write ( AvgofSquareN ( n ) ) ;
function triangular_series ( n ) { for ( let i = 1 ; i <= n ; i ++ ) document . write ( " " + i * ( i + 1 ) / 2 ) ; }
let n = 5 ; triangular_series ( n ) ;
function divisorSum ( n ) { let sum = 0 ; for ( let i = 1 ; i <= n ; ++ i ) sum += Math . floor ( n / i ) * i ; return sum ; }
let n = 4 ; document . write ( divisorSum ( n ) + " " ) ; n = 5 ; document . write ( divisorSum ( n ) + " " ) ;
function sum ( x , n ) { let i , total = 1.0 ; for ( i = 1 ; i <= n ; i ++ ) total = total + ( Math . pow ( x , i ) / i ) ; return total ; }
let g ; let x = 2 ; let n = 5 ; document . write ( sum ( x , n ) . toFixed ( 2 ) ) ;
function check ( n ) {
return 1162261467 % n == 0 ; }
let n = 9 ; if ( check ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function per ( n ) { let a = 3 ; let b = 0 ; let c = 2 ; let i ; let m ; if ( n == 0 ) return a ; if ( n == 1 ) return b ; if ( n == 2 ) return c ; while ( n > 2 ) { m = a + b ; a = b ; b = c ; c = m ; n -- ; } return m ; }
n = 9 ; document . write ( per ( n ) ) ;
function countDivisors ( n ) {
let count = 0 ;
for ( let i = 1 ; i <= Math . sqrt ( n ) + 1 ; i ++ ) { if ( n % i == 0 )
count += ( Math . floor ( n / i ) == i ) ? 1 : 2 ; } if ( count % 2 == 0 ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ; }
document . write ( " " ) ; countDivisors ( 10 ) ;
' ' ' ' ' ' function multiply ( a , b , mod ) { return ( ( a % mod ) * ( b % mod ) ) % mod ; }
function countSquares ( m , n ) {
if ( n < m ) [ m , n ] = [ n , m ] ;
return m * ( m + 1 ) * ( 2 * m + 1 ) / 6 + ( n - m ) * m * ( m + 1 ) / 2 ; }
let m = 4 ; let n = 3 ; document . write ( " " + countSquares ( n , m ) ) ;
function sum ( n ) { var i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
var n = 5 ; document . write ( sum ( n ) . toFixed ( 5 ) ) ;
function gcd ( a , b ) { if ( b == 0 ) { return a ; } return gcd ( b , a % b ) ; }
let a = 98 ; let b = 56 ; document . write ( ` ${ a } ${ b } ${ gcd ( a , b ) } ` ) ;
function printArray ( arr , size ) { for ( i = 0 ; i < size ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; return ; }
function printSequencesRecur ( arr , n , k , index ) { var i ; if ( k == 0 ) { printArray ( arr , index ) ; } if ( k > 0 ) { for ( i = 1 ; i <= n ; ++ i ) { arr [ index ] = i ; printSequencesRecur ( arr , n , k - 1 , index + 1 ) ; } } }
function printSequences ( n , k ) { var arr = Array . from ( { length : k } , ( _ , i ) => 0 ) ; printSequencesRecur ( arr , n , k , 0 ) ; return ; }
var n = 3 ; var k = 2 ; printSequences ( n , k ) ;
function isMultipleof5 ( n ) { while ( n > 0 ) n = n - 5 ; if ( n == 0 ) return true ; return false ; }
let n = 19 ; if ( isMultipleof5 ( n ) == true ) document . write ( n + " " ) ; else document . write ( n + " " ) ;
function countBits ( n ) { var count = 0 ; while ( n != 0 ) { count ++ ; n >>= 1 ; } return count ; }
var i = 65 ; document . write ( countBits ( i ) ) ;
let INT_MAX = 2147483647 ;
function isKthBitSet ( x , k ) { return ( ( x & ( 1 << ( k - 1 ) ) ) > 0 ) ? 1 : 0 ; }
function leftmostSetBit ( x ) { let count = 0 ; while ( x > 0 ) { count ++ ; x = x >> 1 ; } return count ; }
function isBinPalindrome ( x ) { let l = leftmostSetBit ( x ) ; let r = 1 ;
while ( l > r ) {
if ( isKthBitSet ( x , l ) != isKthBitSet ( x , r ) ) return 0 ; l -- ; r ++ ; } return 1 ; } function findNthPalindrome ( n ) { let pal_count = 0 ;
let i = 0 ; for ( i = 1 ; i <= INT_MAX ; i ++ ) { if ( isBinPalindrome ( i ) > 0 ) { pal_count ++ ; }
if ( pal_count == n ) break ; } return i ; }
let n = 9 ;
document . write ( findNthPalindrome ( n ) ) ;
function temp_convert ( F1 , B1 , F2 , B2 , T ) { var t2 ;
t2 = F2 + ( B2 - F2 ) / ( B1 - F1 ) * ( T - F1 ) ; return t2 ; }
var F1 = 0 , B1 = 100 ; var F2 = 32 , B2 = 212 ; var T = 37 ; var t2 ; document . write ( temp_convert ( F1 , B1 , F2 , B2 , T ) . toFixed ( 2 ) ) ;
function findpath ( N , a ) {
if ( a [ 0 ] ) {
document . write ( N + 1 ) ; for ( let i = 1 ; i <= N ; i ++ ) document . write ( i ) ; return ; }
for ( let i = 0 ; i < N - 1 ; i ++ ) { if ( ! a [ i ] && a [ i + 1 ] ) {
for ( let j = 1 ; j <= i ; j ++ ) document . write ( j + " " ) ; document . write ( N + 1 + " " ) ; for ( let j = i + 1 ; j <= N ; j ++ ) document . write ( j + " " ) ; return ; } }
for ( let i = 1 ; i <= N ; i ++ ) document . write ( i + " " ) ; document . write ( N + 1 + " " ) ; }
let N = 3 , arr = [ 0 , 1 , 0 ] ;
findpath ( N , arr ) ;
function printArr ( arr , n ) {
arr . sort ( ) ;
if ( arr [ 0 ] == arr [ n - 1 ] ) { document . write ( " " ) ; }
else { document . write ( " " ) ; for ( i = 0 ; i < n ; i ++ ) { document . write ( arr [ i ] + " " ) ; } } }
var arr = [ 1 , 2 , 2 , 1 , 3 , 1 ] ; var N = arr . length ;
printArr ( arr , N ) ;
let deno = [ 1 , 2 , 5 , 10 , 20 , 50 , 100 , 500 , 1000 ] ; let n = deno . length ; function findMin ( V ) {
let ans = [ ] ;
for ( let i = n - 1 ; i >= 0 ; i -- ) {
while ( V >= deno [ i ] ) { V -= deno [ i ] ; ans . push ( deno [ i ] ) ; } }
for ( let i = 0 ; i < ans . length ; i ++ ) { document . write ( " " + ans [ i ] ) ; } }
n = 93 ; document . write ( " " + " " + n + " " ) ; findMin ( n ) ;
function findMinInsertions ( str , l , h ) {
if ( l > h ) return Number . MAX_VALUE ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( Math . min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) }
let str = " " ; document . write ( findMinInsertions ( str , 0 , str . length - 1 ) ) ;
function max ( x , y ) { return ( x > y ) ? x : y ; }
function lps ( seq , i , j ) {
if ( i == j ) { return 1 ; }
if ( seq [ i ] == seq [ j ] && i + 1 == j ) { return 2 ; }
if ( seq [ i ] == seq [ j ] ) { return lps ( seq , i + 1 , j - 1 ) + 2 ; }
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
let seq = " " ; let n = seq . length ; document . write ( " " , lps ( seq . split ( " " ) , 0 , n - 1 ) ) ;
let NO_OF_CHARS = 256 ; function getNextState ( pat , M , state , x ) {
if ( state < M && x == pat [ state ] . charCodeAt ( 0 ) ) return state + 1 ;
let ns , i ;
for ( ns = state ; ns > 0 ; ns -- ) { if ( pat [ ns - 1 ] . charCodeAt ( 0 ) == x ) { for ( i = 0 ; i < ns - 1 ; i ++ ) if ( pat [ i ] != pat [ state - ns + 1 + i ] ) break ; if ( i == ns - 1 ) return ns ; } } return 0 ; }
function computeTF ( pat , M , TF ) { let state , x ; for ( state = 0 ; state <= M ; ++ state ) for ( x = 0 ; x < NO_OF_CHARS ; ++ x ) TF [ state ] [ x ] = getNextState ( pat , M , state , x ) ; }
function search ( pat , txt ) { let M = pat . length ; let N = txt . length ; let TF = new Array ( M + 1 ) ; for ( let i = 0 ; i < M + 1 ; i ++ ) { TF [ i ] = new Array ( NO_OF_CHARS ) ; for ( let j = 0 ; j < NO_OF_CHARS ; j ++ ) TF [ i ] [ j ] = 0 ; } computeTF ( pat , M , TF ) ;
let i , state = 0 ; for ( i = 0 ; i < N ; i ++ ) { state = TF [ state ] [ txt [ i ] . charCodeAt ( 0 ) ] ; if ( state == M ) document . write ( " " + " " + ( i - M + 1 ) + " " ) ; } }
let pat = " " . split ( " " ) ; let txt = " " . split ( " " ) ; search ( txt , pat ) ;
class Node { constructor ( item ) { this . data = item ; this . left = this . right = null ; } } var root ;
function printInorder ( node ) { if ( node != null ) { printInorder ( node . left ) ; document . write ( node . data + " " ) ; printInorder ( node . right ) ; } }
function RemoveHalfNodes ( node ) { if ( node == null ) return null ; node . left = RemoveHalfNodes ( node . left ) ; node . right = RemoveHalfNodes ( node . right ) ; if ( node . left == null && node . right == null ) return node ;
if ( node . left == null ) { new_root = node . right ; return new_root ; }
if ( node . right == null ) { new_root = node . left ; return new_root ; } return node ; }
NewRoot = null ; root = new Node ( 2 ) ; root . left = new Node ( 7 ) ; root . right = new Node ( 5 ) ; root . left . right = new Node ( 6 ) ; root . left . right . left = new Node ( 1 ) ; root . left . right . right = new Node ( 11 ) ; root . right . right = new Node ( 9 ) ; root . right . right . left = new Node ( 4 ) ; document . write ( " " ) ; printInorder ( root ) ; NewRoot = RemoveHalfNodes ( root ) ; document . write ( " " ) ; printInorder ( NewRoot ) ; script
function printSubstrings ( str ) {
for ( var i = 0 ; i < n ; i ++ ) {
for ( var j = i ; j < n ; j ++ ) {
for ( var k = i ; k <= j ; k ++ ) { document . write ( str . charAt ( k ) ) ; }
document . write ( " " ) ; } } }
var str = " " ;
printSubstrings ( str ) ;
let N = 9 ;
function print ( grid ) { for ( let i = 0 ; i < N ; i ++ ) { for ( let j = 0 ; j < N ; j ++ ) document . write ( grid [ i ] [ j ] + " " ) ; document . write ( " " ) ; } }
function isSafe ( grid , row , col , num ) {
for ( let x = 0 ; x <= 8 ; x ++ ) if ( grid [ row ] [ x ] == num ) return false ;
for ( let x = 0 ; x <= 8 ; x ++ ) if ( grid [ x ] [ col ] == num ) return false ;
let startRow = row - row % 3 , startCol = col - col % 3 ; for ( let i = 0 ; i < 3 ; i ++ ) for ( let j = 0 ; j < 3 ; j ++ ) if ( grid [ i + startRow ] [ j + startCol ] == num ) return false ; return true ; }
function solveSuduko ( grid , row , col ) {
if ( row == N - 1 && col == N ) return true ;
if ( col == N ) { row ++ ; col = 0 ; }
if ( grid [ row ] [ col ] != 0 ) return solveSuduko ( grid , row , col + 1 ) ; for ( let num = 1 ; num < 10 ; num ++ ) {
if ( isSafe ( grid , row , col , num ) ) {
grid [ row ] [ col ] = num ;
if ( solveSuduko ( grid , row , col + 1 ) ) return true ; }
grid [ row ] [ col ] = 0 ; } return false ; }
let grid = [ [ 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 ] , [ 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ] , [ 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 ] , [ 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 ] , [ 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 ] , [ 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 ] , [ 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 ] , [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 ] , [ 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 ] ] if ( solveSuduko ( grid , 0 , 0 ) ) print ( grid ) else document . write ( " " )
function printpairs ( arr , sum ) { let s = new Set ( ) ; for ( let i = 0 ; i < arr . length ; ++ i ) { let temp = sum - arr [ i ] ;
if ( s . has ( temp ) ) { document . write ( " " + sum + " " + arr [ i ] + " " + temp + " " ) ; } s . add ( arr [ i ] ) ; } }
let A = [ 1 , 4 , 45 , 6 , 10 , 8 ] ; let n = 16 ; printpairs ( A , n ) ;
function exponentMod ( A , B , C ) {
if ( A == 0 ) return 0 ; if ( B == 0 ) return 1 ;
var y ; if ( B % 2 == 0 ) { y = exponentMod ( A , B / 2 , C ) ; y = ( y * y ) % C ; }
else { y = A % C ; y = ( y * exponentMod ( A , B - 1 , C ) % C ) % C ; } return parseInt ( ( ( y + C ) % C ) ) ; }
var A = 2 , B = 5 , C = 13 ; document . write ( " " + exponentMod ( A , B , C ) ) ;
function power ( x , y ) {
let res = 1 ; while ( y > 0 ) {
if ( y & 1 ) res = res * x ;
y = y >> 1 ;
x = x * x ; } return res ; }
function eggDrop ( n , k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; let min = Number . MAX_VALUE ; let x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = Math . max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
let n = 2 , k = 10 ; document . write ( " " + " " + n + " " + k + " " + eggDrop ( n , k ) ) ;
class Node { constructor ( val ) { this . data = val ; this . left = null ; this . right = null ; } }
function extractLeafList ( root ) { if ( root == null ) return null ; if ( root . left == null && root . right == null ) { if ( head == null ) { head = root ; prev = root ; } else { prev . right = root ; root . left = prev ; prev = root ; } return null ; } root . left = extractLeafList ( root . left ) ; root . right = extractLeafList ( root . right ) ; return root ; }
function inorder ( node ) { if ( node == null ) return ; inorder ( node . left ) ; document . write ( node . data + " " ) ; inorder ( node . right ) ; }
function printDLL ( head ) { var last = null ; while ( head != null ) { document . write ( head . data + " " ) ; last = head ; head = head . right ; } }
root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . left . left = new Node ( 4 ) ; root . left . right = new Node ( 5 ) ; root . right . right = new Node ( 6 ) ; root . left . left . left = new Node ( 7 ) ; root . left . left . right = new Node ( 8 ) ; root . right . right . left = new Node ( 9 ) ; root . right . right . right = new Node ( 10 ) ; document . write ( " " ) ; inorder ( root ) ; extractLeafList ( root ) ; document . write ( " " ) ; document . write ( " " ) ; printDLL ( head ) ; document . write ( " " ) ; document . write ( " " ) ; inorder ( root ) ;
function countNumberOfStrings ( s ) {
let n = s . length - 1 ;
let count = ( Math . pow ( 2 , n ) ) ; return count ; }
let S = " " ; document . write ( countNumberOfStrings ( S ) ) ;
function makeArraySumEqual ( a , N ) {
let count_0 = 0 , count_1 = 0 ;
let odd_sum = 0 , even_sum = 0 ; for ( let i = 0 ; i < N ; i ++ ) {
if ( a [ i ] == 0 ) count_0 ++ ;
else count_1 ++ ;
if ( ( i + 1 ) % 2 == 0 ) even_sum += a [ i ] ; else if ( ( i + 1 ) % 2 > 0 ) odd_sum += a [ i ] ; }
if ( odd_sum == even_sum ) {
for ( let i = 0 ; i < N ; i ++ ) document . write ( a [ i ] + " " ) ; }
else { if ( count_0 >= N / 2 ) {
for ( let i = 0 ; i < count_0 ; i ++ ) document . write ( " " ) ; } else {
let is_Odd = count_1 % 2 ;
count_1 -= is_Odd ;
for ( let i = 0 ; i < count_1 ; i ++ ) document . write ( " " ) ; } } }
let arr = [ 1 , 1 , 1 , 0 ] ; let N = arr . length ;
makeArraySumEqual ( arr , N ) ;
function countDigitSum ( N , K ) {
var l = parseInt ( Math . pow ( 10 , N - 1 ) ) , var r = parseInt ( Math . pow ( 10 , N ) - 1 ) ; var count = 0 ; for ( i = l ; i <= r ; i ++ ) { var num = i ;
var digits = Array ( N ) . fill ( 0 ) ; for ( j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num = parseInt ( num / 10 ) ; } var sum = 0 , flag = 0 ;
for ( j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( j = K ; j < N ; j ++ ) { if ( sum - digits [ j - K ] + digits [ j ] != sum ) { flag = 1 ; break ; } } if ( flag == 0 ) { count ++ ; } } return count ; }
var N = 2 , K = 1 ; document . write ( countDigitSum ( N , K ) ) ;
function findpath ( N , a ) {
if ( a [ 0 ] ) {
document . write ( N + 1 ) ; for ( let i = 1 ; i <= N ; i ++ ) document . write ( i ) ; return ; }
for ( let i = 0 ; i < N - 1 ; i ++ ) { if ( ! a [ i ] && a [ i + 1 ] ) {
for ( let j = 1 ; j <= i ; j ++ ) document . write ( j + " " ) ; document . write ( N + 1 + " " ) ; for ( let j = i + 1 ; j <= N ; j ++ ) document . write ( j + " " ) ; return ; } }
for ( let i = 1 ; i <= N ; i ++ ) document . write ( i + " " ) ; document . write ( N + 1 + " " ) ; }
let N = 3 , arr = [ 0 , 1 , 0 ] ;
findpath ( N , arr ) ;
function max ( a , b ) { return ( a > b ) ? a : b ; }
function printknapSack ( W , wt , val , n ) { let i , w ; let K = new Array ( n + 1 ) ; for ( i = 0 ; i < K . length ; i ++ ) { K [ i ] = new Array ( W + 1 ) ; for ( let j = 0 ; j < W + 1 ; j ++ ) { K [ i ] [ j ] = 0 ; } }
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = Math . max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } }
let res = K [ n ] [ W ] ; document . write ( res + " " ) ; w = W ; for ( i = n ; i > 0 && res > 0 ; i -- ) {
if ( res == K [ i - 1 ] [ w ] ) continue ; else {
document . write ( wt [ i - 1 ] + " " ) ;
res = res - val [ i - 1 ] ; w = w - wt [ i - 1 ] ; } } } let val = [ 60 , 100 , 120 ] ; let wt = [ 10 , 20 , 30 ] ; let W = 50 ; let n = val . length ; printknapSack ( W , wt , val , n ) ;
function optCost ( freq , i , j ) {
return 0 ;
if ( j == i ) return freq [ i ] ;
var fsum = sum ( freq , i , j ) ;
var min = Number . MAX_SAFE_INTEGER ;
for ( var r = i ; r <= j ; ++ r ) { var cost = optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ; if ( cost < min ) min = cost ; }
return min + fsum ; }
function optimalSearchTree ( keys , freq , n ) {
return optCost ( freq , 0 , n - 1 ) ; }
function sum ( freq , i , j ) { var s = 0 ; for ( var k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
var keys = [ 10 , 12 , 20 ] ; var freq = [ 34 , 8 , 50 ] ; var n = keys . length ; document . write ( " " + optimalSearchTree ( keys , freq , n ) ) ;
let MAX = Number . MAX_VALUE ;
function printSolution ( p , n ) { let k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; document . write ( " " + " " + k + " " + " " + " " + p [ n ] + " " + " " + " " + n + " " ) ; return k ; }
function solveWordWrap ( l , n , M ) {
let extras = new Array ( n + 1 ) ;
let lc = new Array ( n + 1 ) ; for ( let i = 0 ; i < n + 1 ; i ++ ) { extras [ i ] = new Array ( n + 1 ) ; lc [ i ] = new Array ( n + 1 ) ; for ( let j = 0 ; j < n + 1 ; j ++ ) { extras [ i ] [ j ] = 0 ; lc [ i ] [ j ] = 0 ; } }
let c = new Array ( n + 1 ) ;
let p = new Array ( n + 1 ) ;
for ( let i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( let j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( let i = 1 ; i <= n ; i ++ ) { for ( let j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = MAX ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( let j = 1 ; j <= n ; j ++ ) { c [ j ] = MAX ; for ( let i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != MAX && lc [ i ] [ j ] != MAX && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; }
let l = [ 3 , 2 , 2 , 5 ] ; let n = l . length ; let M = 6 ; solveWordWrap ( l , n , M ) ;
function max ( a , b ) { return ( a > b ) ? a : b ; }
function eggDrop ( n , k ) {
let eggFloor = new Array ( n + 1 ) ; for ( let i = 0 ; i < ( n + 1 ) ; i ++ ) { eggFloor [ i ] = new Array ( k + 1 ) ; } let res ; let i , j , x ;
for ( i = 1 ; i <= n ; i ++ ) { eggFloor [ i ] [ 1 ] = 1 ; eggFloor [ i ] [ 0 ] = 0 ; }
for ( j = 1 ; j <= k ; j ++ ) eggFloor [ 1 ] [ j ] = j ;
for ( i = 2 ; i <= n ; i ++ ) { for ( j = 2 ; j <= k ; j ++ ) { eggFloor [ i ] [ j ] = Number . MAX_VALUE ; for ( x = 1 ; x <= j ; x ++ ) { res = 1 + max ( eggFloor [ i - 1 ] [ x - 1 ] , eggFloor [ i ] [ j - x ] ) ; if ( res < eggFloor [ i ] [ j ] ) eggFloor [ i ] [ j ] = res ; } } }
return eggFloor [ n ] [ k ] ; }
let n = 2 , k = 36 ; document . write ( " " + " " + n + " " + k + " " + eggDrop ( n , k ) ) ;
function max ( a , b ) { return ( a > b ) ? a : b ; }
function knapSack ( W , wt , val , n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
let val = [ 60 , 100 , 120 ] ; let wt = [ 10 , 20 , 30 ] ; let W = 50 ; let n = val . length ; document . write ( knapSack ( W , wt , val , n ) ) ;
let max_ref ;
function _lis ( arr , n ) {
if ( n == 1 ) return 1 ;
let res , max_ending_here = 1 ;
for ( let i = 1 ; i < n ; i ++ ) { res = _lis ( arr , i ) ; if ( arr [ i - 1 ] < arr [ n - 1 ] && res + 1 > max_ending_here ) max_ending_here = res + 1 ; }
if ( max_ref < max_ending_here ) max_ref = max_ending_here ;
return max_ending_here ; }
function lis ( arr , n ) {
max_ref = 1 ;
_lis ( arr , n ) ;
return max_ref ; }
let arr = [ 10 , 22 , 9 , 33 , 21 , 50 , 41 , 60 ] let n = arr . length ; document . write ( " " + lis ( arr , n ) + " " ) ;
let d = 256 ;
function search ( pat , txt , q ) { let M = pat . length ; let N = txt . length ; let i , j ;
let p = 0 ; let t = 0 ; let h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat [ i ] . charCodeAt ( ) ) % q ; t = ( d * t + txt [ i ] . charCodeAt ( ) ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) {
for ( j = 0 ; j < M ; j ++ ) { if ( txt [ i + j ] != pat [ j ] ) break ; }
if ( j == M ) document . write ( " " + i + " " ) ; }
if ( i < N - M ) { t = ( d * ( t - txt [ i ] . charCodeAt ( ) * h ) + txt [ i + M ] . charCodeAt ( ) ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
let txt = " " ; let pat = " " ;
let q = 101 ;
search ( pat , txt , q ) ;
let N = 8 ;
function isSafe ( x , y , sol ) { return ( x >= 0 && x < N && y >= 0 && y < N && sol [ x ] [ y ] == - 1 ) ; }
function printSolution ( sol ) { for ( let x = 0 ; x < N ; x ++ ) { for ( let y = 0 ; y < N ; y ++ ) document . write ( sol [ x ] [ y ] + " " ) ; document . write ( " " ) ; } }
function solveKT ( ) { let sol = new Array ( 8 ) ; for ( var i = 0 ; i < sol . length ; i ++ ) { sol [ i ] = new Array ( 2 ) ; }
for ( let x = 0 ; x < N ; x ++ ) for ( let y = 0 ; y < N ; y ++ ) sol [ x ] [ y ] = - 1 ;
let xMove = [ 2 , 1 , - 1 , - 2 , - 2 , - 1 , 1 , 2 ] ; let yMove = [ 1 , 2 , 2 , 1 , - 1 , - 2 , - 2 , - 1 ] ;
sol [ 0 ] [ 0 ] = 0 ;
if ( ! solveKTUtil ( 0 , 0 , 1 , sol , xMove , yMove ) ) { document . write ( " " ) ; return false ; } else printSolution ( sol ) ; return true ; }
function solveKTUtil ( x , y , movei , sol , xMove , yMove ) { let k , next_x , next_y ; if ( movei == N * N ) return true ;
for ( k = 0 ; k < 8 ; k ++ ) { next_x = x + xMove [ k ] ; next_y = y + yMove [ k ] ; if ( isSafe ( next_x , next_y , sol ) ) { sol [ next_x ] [ next_y ] = movei ; if ( solveKTUtil ( next_x , next_y , movei + 1 , sol , xMove , yMove ) ) return true ; else
sol [ next_x ] [ next_y ] = - 1 ; } } return false ; }
solveKT ( ) ;
let V = 4 ;
function printSolution ( color ) { document . write ( " " + " " ) ; for ( let i = 0 ; i < V ; i ++ ) document . write ( " " + color [ i ] ) ; document . write ( " " ) ; }
function isSafe ( graph , color ) {
for ( let i = 0 ; i < V ; i ++ ) for ( let j = i + 1 ; j < V ; j ++ ) if ( graph [ i ] [ j ] && color [ j ] == color [ i ] ) return false ; return true ; }
function graphColoring ( graph , m , i , color ) {
if ( i == V ) {
if ( isSafe ( graph , color ) ) {
printSolution ( color ) ; return true ; } return false ; }
for ( let j = 1 ; j <= m ; j ++ ) { color [ i ] = j ;
if ( graphColoring ( graph , m , i + 1 , color ) ) return true ; color [ i ] = 0 ; } return false ; }
let graph = [ [ false , true , true , true ] , [ true , false , true , false ] , [ true , true , false , true ] , [ true , false , true , false ] ] ;
let m = 3 ;
let color = new Array ( V ) ; for ( let i = 0 ; i < V ; i ++ ) color [ i ] = 0 ; if ( ! graphColoring ( graph , m , 0 , color ) ) document . write ( " " ) ;
function prevPowerofK ( n , k ) { let p = parseInt ( Math . log ( n ) / Math . log ( k ) , 10 ) ; return Math . pow ( k , p ) ; }
function nextPowerOfK ( n , k ) { return prevPowerofK ( n , k ) * k ; }
let N = 7 ; let K = 2 ; document . write ( prevPowerofK ( N , K ) + " " ) ; document . write ( nextPowerOfK ( N , K ) ) ;
function gcd ( a , b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
var a = 98 , b = 56 ; document . write ( " " + a + " " + b + " " + gcd ( a , b ) ) ;
function checkSemiprime ( num ) { let cnt = 0 ; for ( let i = 2 ; cnt < 2 && i * i <= num ; ++ i ) while ( num % i == 0 ) { num /= i ; ++ cnt ; }
if ( num > 1 ) ++ cnt ;
return cnt == 2 ? 1 : 0 ; }
function semiprime ( n ) { if ( checkSemiprime ( n ) != 0 ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ; }
let n = 6 ; semiprime ( n ) ; n = 8 ; semiprime ( n ) ;
function printNSE ( arr , n ) { var next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = - 1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] > arr [ j ] ) { next = arr [ j ] ; break ; } } document . write ( arr [ i ] + " " + next + " " ) ; } }
var arr = [ 11 , 13 , 21 , 3 ] ; var n = arr . length ; printNSE ( arr , n ) ;
let SIZE = 100 ;
function base64Decoder ( encoded , len_str ) { let decoded_String ; decoded_String = new Array ( SIZE ) ; let i , j , k = 0 ;
let num = 0 ;
let count_bits = 0 ;
for ( i = 0 ; i < len_str ; i += 4 ) { num = 0 ; count_bits = 0 ; for ( j = 0 ; j < 4 ; j ++ ) {
if ( encoded [ i + j ] != ' ' ) { num = num << 6 ; count_bits += 6 ; }
if ( encoded [ i + j ] . charCodeAt ( 0 ) >= ' ' . charCodeAt ( 0 ) && encoded [ i + j ] . charCodeAt ( 0 ) <= ' ' . charCodeAt ( 0 ) ) num = num | ( encoded [ i + j ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ;
else if ( encoded [ i + j ] . charCodeAt ( 0 ) >= ' ' . charCodeAt ( 0 ) && encoded [ i + j ] . charCodeAt ( 0 ) <= ' ' . charCodeAt ( 0 ) ) num = num | ( encoded [ i + j ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) + 26 ) ;
else if ( encoded [ i + j ] . charCodeAt ( 0 ) >= ' ' . charCodeAt ( 0 ) && encoded [ i + j ] . charCodeAt ( 0 ) <= ' ' . charCodeAt ( 0 ) ) num = num | ( encoded [ i + j ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) + 52 ) ;
else if ( encoded [ i + j ] == ' ' ) num = num | 62 ;
else if ( encoded [ i + j ] == ' ' ) num = num | 63 ;
else { num = num >> 2 ; count_bits -= 2 ; } } while ( count_bits != 0 ) { count_bits -= 8 ;
decoded_String [ k ++ ] = String . fromCharCode ( ( num >> count_bits ) & 255 ) ; } } return ( decoded_String ) ; }
let encoded_String = " " . split ( " " ) ; let len_str = encoded_String . length ;
len_str -= 1 ; document . write ( " " + ( encoded_String ) . join ( " " ) + " " ) ; document . write ( " " + base64Decoder ( encoded_String , len_str ) . join ( " " ) + " " ) ;
function divideString ( str , n ) { let str_size = str . length ; let part_size ;
if ( str_size % n != 0 ) { document . write ( " " + " " ) ; return ; }
part_size = parseInt ( str_size / n , 10 ) ; for ( let i = 0 ; i < str_size ; i ++ ) { if ( i % part_size == 0 ) document . write ( " " ) ; document . write ( str [ i ] ) ; } }
let str = " " ;
divideString ( str , 4 ) ;
function cool_line ( x1 , y1 , x2 , y2 , x3 , y3 ) { if ( ( y3 - y2 ) * ( x2 - x1 ) == ( y2 - y1 ) * ( x3 - x2 ) ) document . write ( " " ) ; else document . write ( " " ) ; }
var a1 = 1 , a2 = 1 , a3 = 0 , b1 = 1 , b2 = 6 , b3 = 9 ; cool_line ( a1 , b1 , a2 , b2 , a3 , b3 ) ;
function bestApproximate ( x , y , n ) { let m , c , sum_x = 0 , sum_y = 0 , sum_xy = 0 , sum_x2 = 0 ; for ( let i = 0 ; i < n ; i ++ ) { sum_x += x [ i ] ; sum_y += y [ i ] ; sum_xy += x [ i ] * y [ i ] ; sum_x2 += Math . pow ( x [ i ] , 2 ) ; } m = ( n * sum_xy - sum_x * sum_y ) / ( n * sum_x2 - Math . pow ( sum_x , 2 ) ) ; c = ( sum_y - m * sum_x ) / n ; document . write ( " " + m ) ; document . write ( " " + c ) ; }
let x = [ 1 , 2 , 3 , 4 , 5 ] ; let y = [ 14 , 27 , 40 , 55 , 68 ] ; let n = x . length ; bestApproximate ( x , y , n ) ;
function findMinInsertions ( str , l , h ) {
if ( l > h ) return Number . MAX_VALUE ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( Math . min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) }
let str = " " ; document . write ( findMinInsertions ( str , 0 , str . length - 1 ) ) ;
function push ( new_data ) {
var new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
function insertAfter ( prev_node , new_data ) {
if ( prev_node == null ) { document . write ( " " ) ; return ; }
var new_node = new Node ( new_data ) ;
new_node . next = prev_node . next ;
prev_node . next = new_node ; }
class Node { constructor ( val ) { this . data = val ; this . next = null ; } }
function isPalindrome ( head ) { slow_ptr = head ; fast_ptr = head ; var prev_of_slow_ptr = head ;
var midnode = null ;
var res = true ; if ( head != null && head . next != null ) {
while ( fast_ptr != null && fast_ptr . next != null ) { fast_ptr = fast_ptr . next . next ;
prev_of_slow_ptr = slow_ptr ; slow_ptr = slow_ptr . next ; }
if ( fast_ptr != null ) { midnode = slow_ptr ; slow_ptr = slow_ptr . next ; }
second_half = slow_ptr ;
prev_of_slow_ptr . next = null ;
reverse ( ) ;
res = compareLists ( head , second_half ) ;
reverse ( ) ; if ( midnode != null ) {
prev_of_slow_ptr . next = midnode ; midnode . next = second_half ; } else prev_of_slow_ptr . next = second_half ; } return res ; }
function reverse ( ) { var prev = null ; var current = second_half ; var next ; while ( current != null ) { next = current . next ; current . next = prev ; prev = current ; current = next ; } second_half = prev ; }
function compareLists ( head1 , head2 ) { var temp1 = head1 ; var temp2 = head2 ; while ( temp1 != null && temp2 != null ) { if ( temp1 . data == temp2 . data ) { temp1 = temp1 . next ; temp2 = temp2 . next ; } else return false ; }
if ( temp1 == null && temp2 == null ) return true ;
return false ; }
function push ( new_data ) {
var new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
function printList ( ptr ) { while ( ptr != null ) { document . write ( ptr . data + " " ) ; ptr = ptr . next ; } document . write ( " " ) ; }
var str = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ] ; var string = str . toString ( ) ; for ( i = 0 ; i < 7 ; i ++ ) { push ( str [ i ] ) ; printList ( head ) ; if ( isPalindrome ( head ) != false ) { document . write ( " " ) ; document . write ( " " ) ; } else { document . write ( " " ) ; document . write ( " " ) ; } }
class Node { constructor ( val ) { this . data = val ; this . next = null ; } }
function swapNodes ( x , y ) {
if ( x == y ) return ;
var prevX = null , currX = head ; while ( currX != null && currX . data != x ) { prevX = currX ; currX = currX . next ; }
var prevY = null , currY = head ; while ( currY != null && currY . data != y ) { prevY = currY ; currY = currY . next ; }
if ( currX == null currY == null ) return ;
if ( prevX != null ) prevX . next = currY ;
else head = currY ;
if ( prevY != null ) prevY . next = currX ;
else head = currX ;
var temp = currX . next ; currX . next = currY . next ; currY . next = temp ; }
function push ( new_data ) {
var new_Node = new Node ( new_data ) ;
new_Node . next = head ;
head = new_Node ; }
function printList ( ) { var tNode = head ; while ( tNode != null ) { document . write ( tNode . data + " " ) ; tNode = tNode . next ; } }
push ( 7 ) ; push ( 6 ) ; push ( 5 ) ; push ( 4 ) ; push ( 3 ) ; push ( 2 ) ; push ( 1 ) ; document . write ( " " ) ; printList ( ) ; swapNodes ( 4 , 3 ) ; document . write ( " " ) ; printList ( ) ;
class Node { constructor ( ) { this . info = 0 ; this . prev = null ; this . next = null ; } } var head , tail ;
function nodeInsetail ( key ) { p = new Node ( ) ; p . info = key ; p . next = null ;
if ( head == null ) { head = p ; tail = p ; head . prev = null ; return ; }
if ( p . info < head . info ) { p . prev = null ; head . prev = p ; p . next = head ; head = p ; return ; }
if ( p . info > tail . info ) { p . prev = tail ; tail . next = p ; tail = p ; return ; }
temp = head . next ; while ( temp . info < p . info ) temp = temp . next ;
( temp . prev ) . next = p ; p . prev = temp . prev ; temp . prev = p ; p . next = temp ; }
function printList ( temp ) { while ( temp != null ) { document . write ( temp . info + " " ) ; temp = temp . next ; } }
head = tail = null ; nodeInsetail ( 30 ) ; nodeInsetail ( 50 ) ; nodeInsetail ( 90 ) ; nodeInsetail ( 10 ) ; nodeInsetail ( 40 ) ; nodeInsetail ( 110 ) ; nodeInsetail ( 60 ) ; nodeInsetail ( 95 ) ; nodeInsetail ( 23 ) ; document . write ( " " ) ; printList ( head ) ;
class Node { constructor ( item ) { this . data = item ; this . next = null ; } }
class Node { constructor ( val ) { this . data = val ; this . left = null ; this . right = null ; this . parent = null ; } } var head ;
function insert ( node , data ) {
if ( node == null ) { return ( new Node ( data ) ) ; } else { var temp = null ;
if ( data <= node . data ) { temp = insert ( node . left , data ) ; node . left = temp ; temp . parent = node ; } else { temp = insert ( node . right , data ) ; node . right = temp ; temp . parent = node ; }
return node ; } } function inOrderSuccessor ( root , n ) {
if ( n . right != null ) { return minValue ( n . right ) ; }
var p = n . parent ; while ( p != null && n == p . right ) { n = p ; p = p . parent ; } return p ; }
function minValue ( node ) { var current = node ;
while ( current . left != null ) { current = current . left ; } return current ; }
var root = null , temp = null , suc = null , min = null ; root = insert ( root , 20 ) ; root = insert ( root , 8 ) ; root = insert ( root , 22 ) ; root = insert ( root , 4 ) ; root = insert ( root , 12 ) ; root = insert ( root , 10 ) ; root = insert ( root , 14 ) ; temp = root . left . right . right ; suc = inOrderSuccessor ( root , temp ) ; if ( suc != null ) { document . write ( " " + temp . data + " " + suc . data ) ; } else { document . write ( " " ) ; }
class node { constructor ( val ) { this . key = val ; this . left = null ; this . right = null ; } } var head ; var tail ;
function convertBSTtoDLL ( root ) {
if ( root == null ) return ;
if ( root . left != null ) convertBSTtoDLL ( root . left ) ;
root . left = tail ;
if ( tail != null ) ( tail ) . right = root ; else head = root ;
tail = root ;
if ( root . right != null ) convertBSTtoDLL ( root . right ) ; }
function isPresentInDLL ( head , tail , sum ) { while ( head != tail ) { var curr = head . key + tail . key ; if ( curr == sum ) return true ; else if ( curr > sum ) tail = tail . left ; else head = head . right ; } return false ; }
function isTripletPresent ( root ) {
if ( root == null ) return false ;
head = null ; tail = null ; convertBSTtoDLL ( root ) ;
while ( ( head . right != tail ) && ( head . key < 0 ) ) {
if ( isPresentInDLL ( head . right , tail , - 1 * head . key ) ) return true ; else head = head . right ; }
return false ; }
function newNode ( num ) { var temp = new node ( ) ; temp . key = num ; temp . left = temp . right = null ; return temp ; }
function insert ( root , key ) { if ( root == null ) return newNode ( key ) ; if ( root . key > key ) root . left = insert ( root . left , key ) ; else root . right = insert ( root . right , key ) ; return root ; }
var root = null ; root = insert ( root , 6 ) ; root = insert ( root , - 13 ) ; root = insert ( root , 14 ) ; root = insert ( root , - 8 ) ; root = insert ( root , 15 ) ; root = insert ( root , 13 ) ; root = insert ( root , 7 ) ; if ( isTripletPresent ( root ) ) document . write ( " " ) ; else document . write ( " " ) ;
function printSorted ( arr , start , end ) { if ( start > end ) return ;
printSorted ( arr , start * 2 + 1 , end ) ;
document . write ( arr [ start ] + " " ) ;
printSorted ( arr , start * 2 + 2 , end ) ; }
var arr = [ 4 , 2 , 5 , 1 , 3 ] ; printSorted ( arr , 0 , arr . length - 1 ) ;
class Node { constructor ( x ) { this . data = x ; this . left = null ; this . right = null ; } } let root ;
function Ceil ( node , input ) {
if ( node == null ) { return - 1 ; }
if ( node . data == input ) { return node . data ; }
if ( node . data < input ) { return Ceil ( node . right , input ) ; }
let ceil = Ceil ( node . left , input ) ; return ( ceil >= input ) ? ceil : node . data ; }
root = new Node ( 8 ) root . left = new Node ( 4 ) root . right = new Node ( 12 ) root . left . left = new Node ( 2 ) root . left . right = new Node ( 6 ) root . right . left = new Node ( 10 ) root . right . right = new Node ( 14 ) for ( let i = 0 ; i < 16 ; i ++ ) { document . write ( i + " " + Ceil ( root , i ) + " " ) ; }
class node { constructor ( ) { this . key = 0 ; this . count = 0 ; this . left = null ; this . right = null ; } }
function newNode ( item ) { var temp = new node ( ) ; temp . key = item ; temp . left = temp . right = null ; temp . count = 1 ; return temp ; }
function inorder ( root ) { if ( root != null ) { inorder ( root . left ) ; document . write ( root . key + " " + root . count + " " ) ; inorder ( root . right ) ; } }
function insert ( node , key ) {
if ( node == null ) return newNode ( key ) ;
if ( key == node . key ) { ( node . count ) ++ ; return node ; }
if ( key < node . key ) node . left = insert ( node . left , key ) ; else node . right = insert ( node . right , key ) ;
return node ; }
function minValueNode ( node ) { var current = node ;
while ( current . left != null ) current = current . left ; return current ; }
function deleteNode ( root , key ) {
if ( root == null ) return root ;
if ( key < root . key ) root . left = deleteNode ( root . left , key ) ;
else if ( key > root . key ) root . right = deleteNode ( root . right , key ) ;
else {
if ( root . count > 1 ) { ( root . count ) -- ; return root ; }
if ( root . left == null ) { var temp = root . right ; root = null ; return temp ; } else if ( root . right == null ) { var temp = root . left ; root = null ; return temp ; }
var temp = minValueNode ( root . right ) ;
root . key = temp . key ;
root . right = deleteNode ( root . right , temp . key ) ; } return root ; }
var root = null ; root = insert ( root , 12 ) ; root = insert ( root , 10 ) ; root = insert ( root , 20 ) ; root = insert ( root , 9 ) ; root = insert ( root , 11 ) ; root = insert ( root , 10 ) ; root = insert ( root , 12 ) ; root = insert ( root , 12 ) ; document . write ( " " + " " + " " ) ; inorder ( root ) ; document . write ( " " ) ; root = deleteNode ( root , 20 ) ; document . write ( " " + " " ) ; inorder ( root ) ; document . write ( " " ) ; root = deleteNode ( root , 12 ) ; document . write ( " " + " " ) ; inorder ( root ) ; document . write ( " " ) ; root = deleteNode ( root , 9 ) ; document . write ( " " + " " ) ; inorder ( root ) ;
class node { constructor ( ) { this . key = 0 ; this . left = null ; this . right = null ; } } var root = null ;
function newNode ( item ) { var temp = new node ( ) ; temp . key = item ; temp . left = null ; temp . right = null ; return temp ; }
function inorder ( root ) { if ( root != null ) { inorder ( root . left ) ; document . write ( root . key + " " ) ; inorder ( root . right ) ; } }
function insert ( node , key ) {
if ( node == null ) return newNode ( key ) ;
if ( key < node . key ) node . left = insert ( node . left , key ) ; else node . right = insert ( node . right , key ) ;
return node ; }
function minValueNode ( Node ) { var current = Node ;
while ( current . left != null ) current = current . left ; return current ; }
function deleteNode ( root , key ) {
if ( root == null ) return root ;
if ( key < root . key ) root . left = deleteNode ( root . left , key ) ;
else if ( key > root . key ) root . right = deleteNode ( root . right , key ) ;
else {
if ( root . left == null ) { temp = root . right ; return temp ; } else if ( root . right == null ) { temp = root . left ; return temp ; }
var temp = minValueNode ( root . right ) ;
root . key = temp . key ;
root . right = deleteNode ( root . right , temp . key ) ; } return root ; }
function changeKey ( root , oldVal , newVal ) {
root = deleteNode ( root , oldVal ) ;
root = insert ( root , newVal ) ;
return root ; }
root = insert ( root , 50 ) ; root = insert ( root , 30 ) ; root = insert ( root , 20 ) ; root = insert ( root , 40 ) ; root = insert ( root , 70 ) ; root = insert ( root , 60 ) ; root = insert ( root , 80 ) ; document . write ( " " ) ; inorder ( root ) ; root = changeKey ( root , 40 , 10 ) ;
document . write ( " " ) ; inorder ( root ) ;
class Node { constructor ( d ) { this . info = d ; this . left = this . right = null ; } } let head ; let count = 0 ;
function insert ( node , info ) {
if ( node == null ) { return ( new Node ( info ) ) ; } else {
if ( info <= node . info ) { node . left = insert ( node . left , info ) ; } else { node . right = insert ( node . right , info ) ; }
function check ( num ) { let sum = 0 , i = num , sum_of_digits , prod_of_digits ;
if ( num < 10 num > 99 ) return 0 ; else { sum_of_digits = ( i % 10 ) + Math . floor ( i / 10 ) ; prod_of_digits = ( i % 10 ) * Math . floor ( i / 10 ) ; sum = sum_of_digits + prod_of_digits ; } if ( sum == num ) return 1 ; else return 0 ; }
function countSpecialDigit ( rt ) { let x ; if ( rt == null ) return ; else { x = check ( rt . info ) ; if ( x == 1 ) count = count + 1 ; countSpecialDigit ( rt . left ) ; countSpecialDigit ( rt . right ) ; } }
let root = null ; root = insert ( root , 50 ) ; root = insert ( root , 29 ) ; root = insert ( root , 59 ) ; root = insert ( root , 19 ) ; root = insert ( root , 53 ) ; root = insert ( root , 556 ) ; root = insert ( root , 56 ) ; root = insert ( root , 94 ) ; root = insert ( root , 13 ) ;
countSpecialDigit ( root ) ; document . write ( count ) ;
function Identity ( num ) { var row ; var col ; for ( row = 0 ; row < num ; row ++ ) { for ( col = 0 ; col < num ; col ++ ) {
if ( row == col ) document . write ( 1 + " " ) ; else document . write ( 0 + " " ) ; } document . write ( " " + " " ) ; } return 0 ; }
size = 5 ; Identity ( size ) ;
function search ( mat , n , x ) {
let i = 0 , j = n - 1 ; while ( i < n && j >= 0 ) { if ( mat [ i ] [ j ] == x ) { document . write ( " " + i + " " + j ) ; return ; } if ( mat [ i ] [ j ] > x ) j -- ;
else i ++ ; } document . write ( " " ) ;
return ; }
let mat = [ [ 10 , 20 , 30 , 40 ] , [ 15 , 25 , 35 , 45 ] , [ 27 , 29 , 37 , 48 ] , [ 32 , 33 , 39 , 50 ] ] ; search ( mat , 4 , 29 ) ;
function fill0X ( m , n ) {
let i , k = 0 , l = 0 ;
let r = m , c = n ;
let a = new Array ( m ) ; for ( let i = 0 ; i < m ; i ++ ) { a [ i ] = new Array ( n ) ; }
let x = ' ' ;
while ( k < m && l < n ) {
for ( i = l ; i < n ; ++ i ) { a [ k ] [ i ] = x ; } k ++ ;
for ( i = k ; i < m ; ++ i ) { a [ i ] [ n - 1 ] = x ; } n -- ;
if ( k < m ) { for ( i = n - 1 ; i >= l ; -- i ) a [ m - 1 ] [ i ] = x ; m -- ; }
if ( l < n ) { for ( i = m - 1 ; i >= k ; -- i ) { a [ i ] [ l ] = x ; } l ++ ; }
x = ( x == ' ' ) ? ' ' : ' ' ; }
for ( i = 0 ; i < r ; i ++ ) { for ( let j = 0 ; j < c ; j ++ ) { document . write ( a [ i ] [ j ] + " " ) ; } document . write ( " " ) ; } }
document . write ( " " ) ; fill0X ( 5 , 6 ) ; document . write ( " " ) ; fill0X ( 4 , 4 ) ; document . write ( " " ) ; fill0X ( 3 , 4 ) ;
let N = 3 ;
function interchangeDiagonals ( array ) {
for ( let i = 0 ; i < N ; ++ i ) if ( i != parseInt ( N / 2 ) ) { let temp = array [ i ] [ i ] ; array [ i ] [ i ] = array [ i ] [ N - i - 1 ] ; array [ i ] [ N - i - 1 ] = temp ; } for ( let i = 0 ; i < N ; ++ i ) { for ( let j = 0 ; j < N ; ++ j ) document . write ( " " + array [ i ] [ j ] ) ; document . write ( " " ) ; } }
let array = [ [ 4 , 5 , 6 ] , [ 1 , 2 , 3 ] , [ 7 , 8 , 9 ] ] ; interchangeDiagonals ( array ) ;
class Node { constructor ( val ) { this . data = val ; this . left = null ; this . right = null ; } } var root ;
function bintree2listUtil ( node ) {
if ( node == null ) return node ;
if ( node . left != null ) {
var left = bintree2listUtil ( node . left ) ;
for ( ; left . right != null ; left = left . right )
left . right = node ;
node . left = left ; }
if ( node . right != null ) {
var right = bintree2listUtil ( node . right ) ;
for ( ; right . left != null ; right = right . left )
right . left = node ;
node . right = right ; } return node ; }
function bintree2list ( node ) {
if ( node == null ) return node ;
node = bintree2listUtil ( node ) ;
while ( node . left != null ) node = node . left ; return node ; }
function printList ( node ) { while ( node != null ) { document . write ( node . data + " " ) ; node = node . right ; } }
root = new Node ( 10 ) ; root . left = new Node ( 12 ) ; root . right = new Node ( 15 ) ; root . left . left = new Node ( 25 ) ; root . left . right = new Node ( 30 ) ; root . right . left = new Node ( 36 ) ;
var head = bintree2list ( root ) ;
printList ( head ) ;
let M = 4 ; let N = 5 ;
function findCommon ( mat ) {
let column = new Array ( M ) ;
let min_row ;
let i ; for ( i = 0 ; i < M ; i ++ ) column [ i ] = N - 1 ;
min_row = 0 ;
while ( column [ min_row ] >= 0 ) {
for ( i = 0 ; i < M ; i ++ ) { if ( mat [ i ] [ column [ i ] ] < mat [ min_row ] [ column [ min_row ] ] ) min_row = i ; }
let eq_count = 0 ;
for ( i = 0 ; i < M ; i ++ ) {
if ( mat [ i ] [ column [ i ] ] > mat [ min_row ] [ column [ min_row ] ] ) { if ( column [ i ] == 0 ) return - 1 ;
column [ i ] -= 1 ; } else eq_count ++ ; }
if ( eq_count == M ) return mat [ min_row ] [ column [ min_row ] ] ; } return - 1 ; }
let mat = [ [ 1 , 2 , 3 , 4 , 5 ] , [ 2 , 4 , 5 , 8 , 10 ] , [ 3 , 5 , 7 , 9 , 11 ] , [ 1 , 3 , 5 , 7 , 9 ] ] ; let result = findCommon ( mat ) if ( result == - 1 ) { document . write ( " " ) ; } else { document . write ( " " , result ) ; }
class node { constructor ( val ) { this . data = val ; this . left = null ; this . right = null ; } } var prev ;
function inorder ( root ) { if ( root == null ) return ; inorder ( root . left ) ; document . write ( root . data + " " ) ; inorder ( root . right ) ; }
function fixPrevptr ( root ) { if ( root == null ) return ; fixPrevptr ( root . left ) ; root . left = prev ; prev = root ; fixPrevptr ( root . right ) ; }
function fixNextptr ( root ) {
while ( root . right != null ) root = root . right ;
while ( root != null && root . left != null ) { var left = root . left ; left . right = root ; root = root . left ; }
return root ; }
function BTTtoDLL ( root ) { prev = null ;
fixPrevptr ( root ) ;
return fixNextptr ( root ) ; }
function printlist ( root ) { while ( root != null ) { document . write ( root . data + " " ) ; root = root . right ; } }
var root = new node ( 10 ) ; root . left = new node ( 12 ) ; root . right = new node ( 15 ) ; root . left . left = new node ( 25 ) ; root . left . right = new node ( 30 ) ; root . right . left = new node ( 36 ) ; document . write ( " " ) ; inorder ( root ) ; var head = BTTtoDLL ( root ) ; document . write ( " " ) ; printlist ( head ) ;
function findPeakUtil ( arr , low , high , n ) {
var mid = low + parseInt ( ( high - low ) / 2 ) ;
if ( ( mid == 0 arr [ mid - 1 ] <= arr [ mid ] ) && ( mid == n - 1 arr [ mid + 1 ] <= arr [ mid ] ) ) return mid ;
else if ( mid > 0 && arr [ mid - 1 ] > arr [ mid ] ) return findPeakUtil ( arr , low , ( mid - 1 ) , n ) ;
else return findPeakUtil ( arr , ( mid + 1 ) , high , n ) ; }
function findPeak ( arr , n ) { return findPeakUtil ( arr , 0 , n - 1 , n ) ; }
var arr = [ 1 , 3 , 20 , 4 , 1 , 0 ] ; var n = arr . length ; document . write ( " " + findPeak ( arr , n ) ) ;
function printRepeating ( arr , size ) { var i , j ; document . write ( " " ) ; for ( i = 0 ; i < size ; i ++ ) { for ( j = i + 1 ; j < size ; j ++ ) { if ( arr [ i ] == arr [ j ] ) document . write ( arr [ i ] + " " ) ; } } }
var arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] ; var arr_size = arr . length ; printRepeating ( arr , arr_size ) ;
function printRepeating ( arr , size ) { let count = new Array ( size ) ; count . fill ( 0 ) ; let i ; document . write ( " " ) ; for ( i = 0 ; i < size ; i ++ ) { if ( count [ arr [ i ] ] == 1 ) document . write ( arr [ i ] + " " ) ; else count [ arr [ i ] ] ++ ; } }
let arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] ; let arr_size = arr . length ; printRepeating ( arr , arr_size ) ;
function printRepeating ( arr , size ) {
var S = 0 ;
var P = 1 ;
var x , y ;
var D ; var n = size - 2 , i ;
for ( i = 0 ; i < size ; i ++ ) { S = S + arr [ i ] ; P = P * arr [ i ] ; }
S = S - n * parseInt ( ( n + 1 ) / 2 ) ;
P = parseInt ( P / fact ( n ) ) ;
D = parseInt ( Math . sqrt ( S * S - 4 * P ) ) ; x = parseInt ( ( D + S ) / 2 ) ; y = parseInt ( ( S - D ) / 2 ) ; document . write ( " " ) ; document . write ( x + " " + y ) ; }
function fact ( n ) { var ans = false ; if ( n == 0 ) return 1 ; else return ( n * fact ( n - 1 ) ) ; }
var arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] ; var arr_size = arr . length ; printRepeating ( arr , arr_size ) ;
function printRepeating ( arr , size ) {
let Xor = arr [ 0 ] ;
let set_bit_no ; let i ; let n = size - 2 ; let x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) Xor ^= arr [ i ] ; for ( i = 1 ; i <= n ; i ++ ) Xor ^= i ;
set_bit_no = Xor & ~ ( Xor - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ;
} for ( i = 1 ; i <= n ; i ++ ) { if ( i & set_bit_no ) x = x ^ i ;
else y = y ^ i ;
} document . write ( " " + y + " " + x ) ; }
let arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] ; let arr_size = arr . length ; printRepeating ( arr , arr_size ) ;
function printRepeating ( arr , size ) { var i ; document . write ( " " ) ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ Math . abs ( arr [ i ] ) ] > 0 ) arr [ Math . abs ( arr [ i ] ) ] = - arr [ Math . abs ( arr [ i ] ) ] ; else document . write ( Math . abs ( arr [ i ] ) + " " ) ; } }
var arr = [ 4 , 2 , 4 , 5 , 2 , 3 , 1 ] ; var arr_size = arr . length ; printRepeating ( arr , arr_size ) ;
function subArraySum ( arr , n , sum ) { let curr_sum = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { curr_sum = arr [ i ] ;
for ( let j = i + 1 ; j <= n ; j ++ ) { if ( curr_sum == sum ) { document . write ( " " + i + " " + ( j - 1 ) ) ; return ; } if ( curr_sum > sum j == n ) break ; curr_sum = curr_sum + arr [ j ] ; } } document . write ( " " ) ; return ; }
let arr = [ 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 ] ; let n = arr . length ; let sum = 23 ; subArraySum ( arr , n , sum ) ;
function subArraySum ( arr , n , sum ) {
let curr_sum = arr [ 0 ] , start = 0 , i ;
for ( i = 1 ; i <= n ; i ++ ) {
while ( curr_sum > sum && start < i - 1 ) { curr_sum = curr_sum - arr [ start ] ; start ++ ; }
if ( curr_sum == sum ) { let p = i - 1 ; document . write ( " " + start + " " + p + " " ) ; return 1 ; }
if ( i < n ) curr_sum = curr_sum + arr [ i ] ; }
document . write ( " " ) ; return 0 ; }
let arr = [ 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 ] ; let n = arr . length ; let sum = 23 ; subArraySum ( arr , n , sum ) ;
function find3Numbers ( A , arr_size , sum ) { let l , r ;
for ( let i = 0 ; i < arr_size - 2 ; i ++ ) {
for ( let j = i + 1 ; j < arr_size - 1 ; j ++ ) {
for ( let k = j + 1 ; k < arr_size ; k ++ ) { if ( A [ i ] + A [ j ] + A [ k ] == sum ) { document . write ( " " + A [ i ] + " " + A [ j ] + " " + A [ k ] ) ; return true ; } } } }
return false ; }
let A = [ 1 , 4 , 45 , 6 , 10 , 8 ] ; let sum = 22 ; let arr_size = A . length ; find3Numbers ( A , arr_size , sum ) ;
function binarySearch ( arr , l , r , x ) { if ( r >= l ) { let mid = l + Math . floor ( ( r - l ) / 2 ) ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return - 1 ; }
let arr = [ 2 , 3 , 4 , 10 , 40 ] ; let x = 10 ; let n = arr . length let result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == - 1 ) ? document . write ( " " ) : document . write ( " " + result ) ;
function interpolationSearch ( arr , lo , hi , x ) { let pos ;
if ( lo <= hi && x >= arr [ lo ] && x <= arr [ hi ] ) {
pos = lo + Math . floor ( ( ( hi - lo ) / ( arr [ hi ] - arr [ lo ] ) ) * ( x - arr [ lo ] ) ) ; ;
if ( arr [ pos ] == x ) { return pos ; }
if ( arr [ pos ] < x ) { return interpolationSearch ( arr , pos + 1 , hi , x ) ; }
if ( arr [ pos ] > x ) { return interpolationSearch ( arr , lo , pos - 1 , x ) ; } } return - 1 ; }
let arr = [ 10 , 12 , 13 , 16 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 33 , 35 , 42 , 47 ] ; let n = arr . length ;
let x = 18 let index = interpolationSearch ( arr , 0 , n - 1 , x ) ;
if ( index != - 1 ) { document . write ( ` ${ index } ` ) } else { document . write ( " " ) ; }
function merge ( arr , l , m , r ) {
var n1 = m - l + 1 ; var n2 = r - m ;
var L = new Array ( n1 ) ; var R = new Array ( n2 ) ;
for ( var i = 0 ; i < n1 ; i ++ ) L [ i ] = arr [ l + i ] ; for ( var j = 0 ; j < n2 ; j ++ ) R [ j ] = arr [ m + 1 + j ] ;
var i = 0 ; var j = 0 ;
var k = l ; while ( i < n1 && j < n2 ) { if ( L [ i ] <= R [ j ] ) { arr [ k ] = L [ i ] ; i ++ ; } else { arr [ k ] = R [ j ] ; j ++ ; } k ++ ; }
while ( i < n1 ) { arr [ k ] = L [ i ] ; i ++ ; k ++ ; }
while ( j < n2 ) { arr [ k ] = R [ j ] ; j ++ ; k ++ ; } }
function mergeSort ( arr , l , r ) { if ( l >= r ) { return ; }
var m = l + parseInt ( ( r - l ) / 2 ) ;
mergeSort ( arr , l , m ) ; mergeSort ( arr , m + 1 , r ) ;
merge ( arr , l , m , r ) ; }
function printArray ( A , size ) { for ( var i = 0 ; i < size ; i ++ ) document . write ( A [ i ] + " " ) ; }
var arr = [ 12 , 11 , 13 , 5 , 6 , 7 ] ; var arr_size = arr . length ; document . write ( " " ) ; printArray ( arr , arr_size ) ; mergeSort ( arr , 0 , arr_size - 1 ) ; document . write ( " " ) ; printArray ( arr , arr_size ) ;
function printMaxActivities ( s , f , n ) { let i , j ; document . write ( " " ) ;
i = 0 ; document . write ( i + " " ) ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { document . write ( j + " " ) ; i = j ; } } }
let s = [ 1 , 3 , 0 , 5 , 8 , 5 ] let f = [ 2 , 4 , 6 , 7 , 9 , 9 ] let n = s . length ; printMaxActivities ( s , f , n ) ;
function min ( x , y , z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
function minCost ( cost , m , n ) { if ( n < 0 m < 0 ) return Number . MAX_VALUE ; else if ( m == 0 && n == 0 ) return cost [ m ] [ n ] ; else return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) ; }
var cost = [ [ 1 , 2 , 3 ] , [ 4 , 8 , 2 ] , [ 1 , 5 , 3 ] ] ; document . write ( minCost ( cost , 2 , 2 ) ) ;
function minCost ( cost , m , n ) { let i , j ;
let tc = new Array ( m + 1 ) ; for ( let k = 0 ; k < m + 1 ; k ++ ) { tc [ k ] = new Array ( n + 1 ) ; } tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] ;
for ( i = 1 ; i <= m ; i ++ ) tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] ;
for ( j = 1 ; j <= n ; j ++ ) tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] ;
for ( i = 1 ; i <= m ; i ++ ) for ( j = 1 ; j <= n ; j ++ ) tc [ i ] [ j ] = Math . min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] ; return tc [ m ] [ n ] ; }
function min ( x , y , z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
let cost = [ [ 1 , 2 , 3 ] , [ 4 , 8 , 2 ] , [ 1 , 5 , 3 ] ] ; document . write ( minCost ( cost , 2 , 2 ) ) ;
function max ( a , b ) { return ( a > b ) ? a : b ; }
function knapSack ( W , wt , val , n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
let val = [ 60 , 100 , 120 ] ; let wt = [ 10 , 20 , 30 ] ; let W = 50 ; let n = val . length ; document . write ( knapSack ( W , wt , val , n ) ) ;
function max ( a , b ) { return ( a > b ) ? a : b ; }
function knapSack ( W , wt , val , n ) { let i , w ; let K = new Array ( n + 1 ) ;
for ( i = 0 ; i <= n ; i ++ ) { K [ i ] = new Array ( W + 1 ) ; for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
let val = [ 60 , 100 , 120 ] ; let wt = [ 10 , 20 , 30 ] ; let W = 50 ; let n = val . length ; document . write ( knapSack ( W , wt , val , n ) ) ;
function eggDrop ( n , k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; let min = Number . MAX_VALUE ; let x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = Math . max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
let n = 2 , k = 10 ; document . write ( " " + " " + n + " " + k + " " + eggDrop ( n , k ) ) ;
function max ( x , y ) { return ( x > y ) ? x : y ; }
function lps ( seq , i , j ) {
if ( i == j ) { return 1 ; }
if ( seq [ i ] == seq [ j ] && i + 1 == j ) { return 2 ; }
if ( seq [ i ] == seq [ j ] ) { return lps ( seq , i + 1 , j - 1 ) + 2 ; }
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
let seq = " " ; let n = seq . length ; document . write ( " " , lps ( seq . split ( " " ) , 0 , n - 1 ) ) ;
let MAX = Number . MAX_VALUE ;
function printSolution ( p , n ) { let k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; document . write ( " " + " " + k + " " + " " + " " + p [ n ] + " " + " " + " " + n + " " ) ; return k ; }
function solveWordWrap ( l , n , M ) {
let extras = new Array ( n + 1 ) ;
let lc = new Array ( n + 1 ) ; for ( let i = 0 ; i < n + 1 ; i ++ ) { extras [ i ] = new Array ( n + 1 ) ; lc [ i ] = new Array ( n + 1 ) ; for ( let j = 0 ; j < n + 1 ; j ++ ) { extras [ i ] [ j ] = 0 ; lc [ i ] [ j ] = 0 ; } }
let c = new Array ( n + 1 ) ;
let p = new Array ( n + 1 ) ;
for ( let i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( let j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( let i = 1 ; i <= n ; i ++ ) { for ( let j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = MAX ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( let j = 1 ; j <= n ; j ++ ) { c [ j ] = MAX ; for ( let i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != MAX && lc [ i ] [ j ] != MAX && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; }
let l = [ 3 , 2 , 2 , 5 ] ; let n = l . length ; let M = 6 ; solveWordWrap ( l , n , M ) ;
function sum ( freq , i , j ) { var s = 0 ; for ( var k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
function optCost ( freq , i , j ) {
if ( j < i ) return 0 ;
if ( j == i ) return freq [ i ] ;
var fsum = sum ( freq , i , j ) ;
var min = Number . MAX_SAFE_INTEGER ;
for ( var r = i ; r <= j ; ++ r ) { var cost = optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ; if ( cost < min ) min = cost ; }
return min + fsum ; }
function optimalSearchTree ( keys , freq , n ) {
return optCost ( freq , 0 , n - 1 ) ; }
var keys = [ 10 , 12 , 20 ] ; var freq = [ 34 , 8 , 50 ] ; var n = keys . length ; document . write ( " " + optimalSearchTree ( keys , freq , n ) ) ;
function sum ( freq , i , j ) { var s = 0 ; for ( var k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
function optimalSearchTree ( keys , freq , n ) {
var cost = new Array ( n ) ; for ( var i = 0 ; i < n ; i ++ ) cost [ i ] = new Array ( n ) ;
for ( var i = 0 ; i < n ; i ++ ) cost [ i ] [ i ] = freq [ i ] ;
for ( var L = 2 ; L <= n ; L ++ ) {
for ( var i = 0 ; i <= n - L + 1 ; i ++ ) {
var j = i + L - 1 ; if ( i >= n j >= n ) break cost [ i ] [ j ] = Number . MAX_SAFE_INTEGER ;
for ( var r = i ; r <= j ; r ++ ) {
var c = 0 ; if ( r > i ) c += cost [ i ] [ r - 1 ] if ( r < j ) c += cost [ r + 1 ] [ j ] c += sum ( freq , i , j ) ; if ( c < cost [ i ] [ j ] ) cost [ i ] [ j ] = c ; } } } return cost [ 0 ] [ n - 1 ] ; }
var keys = [ 10 , 12 , 20 ] ; var freq = [ 34 , 8 , 50 ] ; var n = keys . length ; document . write ( " " + optimalSearchTree ( keys , freq , n ) ) ;
function getCount ( keypad , n ) { if ( keypad == null n <= 0 ) return 0 ; if ( n == 1 ) return 10 ;
var odd = Array . from ( { length : 10 } , ( _ , i ) => 0 ) ; var even = Array . from ( { length : 10 } , ( _ , i ) => 0 ) ; var i = 0 , j = 0 , useOdd = 0 , totalCount = 0 ; for ( i = 0 ; i <= 9 ; i ++ )
odd [ i ] = 1 ;
for ( j = 2 ; j <= n ; j ++ ) { useOdd = 1 - useOdd ;
if ( useOdd == 1 ) { even [ 0 ] = odd [ 0 ] + odd [ 8 ] ; even [ 1 ] = odd [ 1 ] + odd [ 2 ] + odd [ 4 ] ; even [ 2 ] = odd [ 2 ] + odd [ 1 ] + odd [ 3 ] + odd [ 5 ] ; even [ 3 ] = odd [ 3 ] + odd [ 2 ] + odd [ 6 ] ; even [ 4 ] = odd [ 4 ] + odd [ 1 ] + odd [ 5 ] + odd [ 7 ] ; even [ 5 ] = odd [ 5 ] + odd [ 2 ] + odd [ 4 ] + odd [ 8 ] + odd [ 6 ] ; even [ 6 ] = odd [ 6 ] + odd [ 3 ] + odd [ 5 ] + odd [ 9 ] ; even [ 7 ] = odd [ 7 ] + odd [ 4 ] + odd [ 8 ] ; even [ 8 ] = odd [ 8 ] + odd [ 0 ] + odd [ 5 ] + odd [ 7 ] + odd [ 9 ] ; even [ 9 ] = odd [ 9 ] + odd [ 6 ] + odd [ 8 ] ; } else { odd [ 0 ] = even [ 0 ] + even [ 8 ] ; odd [ 1 ] = even [ 1 ] + even [ 2 ] + even [ 4 ] ; odd [ 2 ] = even [ 2 ] + even [ 1 ] + even [ 3 ] + even [ 5 ] ; odd [ 3 ] = even [ 3 ] + even [ 2 ] + even [ 6 ] ; odd [ 4 ] = even [ 4 ] + even [ 1 ] + even [ 5 ] + even [ 7 ] ; odd [ 5 ] = even [ 5 ] + even [ 2 ] + even [ 4 ] + even [ 8 ] + even [ 6 ] ; odd [ 6 ] = even [ 6 ] + even [ 3 ] + even [ 5 ] + even [ 9 ] ; odd [ 7 ] = even [ 7 ] + even [ 4 ] + even [ 8 ] ; odd [ 8 ] = even [ 8 ] + even [ 0 ] + even [ 5 ] + even [ 7 ] + even [ 9 ] ; odd [ 9 ] = even [ 9 ] + even [ 6 ] + even [ 8 ] ; } }
totalCount = 0 ; if ( useOdd == 1 ) { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += even [ i ] ; } else { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += odd [ i ] ; } return totalCount ; }
var keypad = [ [ ' ' , ' ' , ' ' ] , [ ' ' , ' ' , ' ' ] , [ ' ' , ' ' , ' ' ] , [ ' ' , ' ' , ' ' ] ] ; document . write ( " " + 1 + " " + getCount ( keypad , 1 ) ) ; document . write ( " " + 2 + " " + getCount ( keypad , 2 ) ) ; document . write ( " " + 3 + " " + getCount ( keypad , 3 ) ) ; document . write ( " " + 4 + " " + getCount ( keypad , 4 ) ) ; document . write ( " " + 5 + " " + getCount ( keypad , 5 ) ) ;
function count ( n ) {
let table = new Array ( n + 1 ) , i ;
for ( let j = 0 ; j < n + 1 ; j ++ ) table [ j ] = 0 ;
table [ 0 ] = 1 ;
for ( i = 3 ; i <= n ; i ++ ) table [ i ] += table [ i - 3 ] ; for ( i = 5 ; i <= n ; i ++ ) table [ i ] += table [ i - 5 ] ; for ( i = 10 ; i <= n ; i ++ ) table [ i ] += table [ i - 10 ] ; return table [ n ] ; }
let n = 20 ; document . write ( " " + n + " " + count ( n ) + " " ) ; n = 13 ; document . write ( " " + n + " " + count ( n ) + " " ) ;
function search ( txt , pat ) { let M = pat . length ; let N = txt . length ;
for ( let i = 0 ; i <= N - M ; i ++ ) { let j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) document . write ( " " + i + " " ) ; } }
let txt = " " ; let pat = " " ; search ( txt , pat ) ;
let d = 256 ;
function search ( pat , txt , q ) { let M = pat . length ; let N = txt . length ; let i , j ;
let p = 0 ;
let t = 0 ; let h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat [ i ] . charCodeAt ( ) ) % q ; t = ( d * t + txt [ i ] . charCodeAt ( ) ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) {
for ( j = 0 ; j < M ; j ++ ) { if ( txt [ i + j ] != pat [ j ] ) break ; }
if ( j == M ) document . write ( " " + i + " " ) ; }
if ( i < N - M ) { t = ( d * ( t - txt [ i ] . charCodeAt ( ) * h ) + txt [ i + M ] . charCodeAt ( ) ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
let txt = " " ; let pat = " " ;
let q = 101 ;
search ( pat , txt , q ) ;
function search ( pat , txt ) { let M = pat . length ; let N = txt . length ; let i = 0 ; while ( i <= N - M ) { let j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) { document . write ( " " + i + " " ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
let txt = " " ; let pat = " " ; search ( pat , txt ) ;
function getMedian ( ar1 , ar2 , n ) { var i = 0 ; var j = 0 ; var count ; var m1 = - 1 , m2 = - 1 ;
for ( count = 0 ; count <= n ; count ++ ) {
if ( i == n ) { m1 = m2 ; m2 = ar2 [ 0 ] ; break ; }
else if ( j == n ) { m1 = m2 ; m2 = ar1 [ 0 ] ; break ; }
if ( ar1 [ i ] <= ar2 [ j ] ) { m1 = m2 ;
m2 = ar1 [ i ] ; i ++ ; } else { m1 = m2 ;
m2 = ar2 [ j ] ; j ++ ; } } return ( m1 + m2 ) / 2 ; }
var ar1 = [ 1 , 12 , 15 , 26 , 38 ] ; var ar2 = [ 2 , 13 , 17 , 30 , 45 ] ; var n1 = ar1 . length ; var n2 = ar2 . length ; if ( n1 == n2 ) document . write ( " " + getMedian ( ar1 , ar2 , n1 ) ) ; else document . write ( " " ) ;
function isLucky ( n ) { let counter = 2 ;
let next_position = n ; if ( counter > n ) return 1 ; if ( n % counter == 0 ) return 0 ;
next_position -= Math . floor ( next_position / counter ) ; counter ++ ; return isLucky ( next_position ) ; }
let x = 5 ; if ( isLucky ( x ) ) document . write ( x + " " ) ; else document . write ( x + " " ) ;
function pow ( a , b ) { if ( b == 0 ) return 1 ; var answer = a ; var increment = a ; var i , j ; for ( i = 1 ; i < b ; i ++ ) { for ( j = 1 ; j < a ; j ++ ) { answer += increment ; } increment = answer ; } return answer ; }
document . write ( pow ( 5 , 3 ) ) ;
function multiply ( x , y ) { if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ; else return 0 ; }
function pow ( a , b ) { if ( b > 0 ) return multiply ( a , pow ( a , b - 1 ) ) ; else return 1 ; }
document . write ( pow ( 5 , 3 ) ) ;
function count ( n ) {
if ( n < 3 ) return n ; if ( n >= 3 && n < 10 ) return n - 1 ;
var po = 1 ; while ( parseInt ( n / po ) > 9 ) po = po * 10 ;
var msd = parseInt ( n / po ) ; if ( msd != 3 )
return count ( msd ) * count ( po - 1 ) + count ( msd ) + count ( n % po ) ; else
return count ( msd * po - 1 ) ; }
var n = 578 ; document . write ( count ( n ) ) ;
function fact ( n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
function findSmallerInRight ( str , low , high ) { let countRight = 0 ; let i ; for ( i = low + 1 ; i <= high ; ++ i ) if ( str [ i ] < str [ low ] ) ++ countRight ; return countRight ; }
function findRank ( str ) { let len = ( str ) . length ; let mul = fact ( len ) ; let rank = 1 ; let countRight ; let i ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
countRight = findSmallerInRight ( str , i , len - 1 ) ; rank += countRight * mul ; } return rank ; }
let str = " " ; document . write ( findRank ( str ) ) ;
function exponential ( n , x ) {
var sum = 1 ; for ( i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
var n = 10 ; var x = 1 ; document . write ( " " + exponential ( n , x ) . toFixed ( 6 ) ) ;
function findCeil ( arr , r , l , h ) { let mid ; while ( l < h ) {
mid = l + ( ( h - l ) >> 1 ) ; ( r > arr [ mid ] ) ? ( l = mid + 1 ) : ( h = mid ) ; } return ( arr [ l ] >= r ) ? l : - 1 ; }
function myRand ( arr , freq , n ) {
let prefix = [ ] ; let i ; prefix [ 0 ] = freq [ 0 ] ; for ( i = 1 ; i < n ; ++ i ) prefix [ i ] = prefix [ i - 1 ] + freq [ i ] ;
let r = Math . floor ( ( Math . random ( ) * prefix [ n - 1 ] ) ) + 1 ;
let indexc = findCeil ( prefix , r , 0 , n - 1 ) ; return arr [ indexc ] ; }
let arr = [ 1 , 2 , 3 , 4 ] ; let freq = [ 10 , 5 , 20 , 100 ] ; let i ; let n = arr . length ;
for ( i = 0 ; i < 5 ; i ++ ) document . write ( myRand ( arr , freq , n ) ) ;
function min ( x , y ) { return ( x < y ) ? x : y ; }
function calcAngle ( h , m ) {
if ( h < 0 m < 0 h > 12 m > 60 ) document . write ( " " ) ; if ( h == 12 ) h = 0 ; if ( m == 60 ) { m = 0 ; h += 1 ; if ( h > 12 ) h = h - 12 ; }
let hour_angle = 0.5 * ( h * 60 + m ) ; let minute_angle = 6 * m ;
let angle = Math . abs ( hour_angle - minute_angle ) ;
angle = min ( 360 - angle , angle ) ; return angle ; }
document . write ( calcAngle ( 9 , 60 ) + " " ) ; document . write ( calcAngle ( 3 , 30 ) + " " ) ;
function getSingle ( arr , n ) { let ones = 0 , twos = 0 ; let common_bit_mask ; for ( let i = 0 ; i < n ; i ++ ) {
twos = twos | ( ones & arr [ i ] ) ;
ones = ones ^ arr [ i ] ;
common_bit_mask = ~ ( ones & twos ) ;
ones &= common_bit_mask ;
twos &= common_bit_mask ; } return ones ; }
let arr = [ 3 , 3 , 2 , 3 ] ; let n = arr . length ; document . write ( " " + getSingle ( arr , n ) ) ;
let INT_SIZE = 32 ; function getSingle ( arr , n ) {
let result = 0 ; let x , sum ;
for ( let i = 0 ; i < INT_SIZE ; i ++ ) {
sum = 0 ; x = ( 1 << i ) ; for ( let j = 0 ; j < n ; j ++ ) { if ( arr [ j ] & x ) sum ++ ; }
if ( ( sum % 3 ) != 0 ) result |= x ; } return result ; }
let arr = [ 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 ] ; let n = arr . length ; document . write ( " " + getSingle ( arr , n ) ) ;
function swapBits ( x , p1 , p2 , n ) {
let set1 = ( x >> p1 ) & ( ( 1 << n ) - 1 ) ;
let set2 = ( x >> p2 ) & ( ( 1 << n ) - 1 ) ;
let xor = ( set1 ^ set2 ) ;
xor = ( xor << p1 ) | ( xor << p2 ) ;
let result = x ^ xor ; return result ; }
let res = swapBits ( 28 , 0 , 3 , 2 ) ; document . write ( " " + res ) ;
function smallest ( x , y , z ) { let c = 0 ; while ( x && y && z ) { x -- ; y -- ; z -- ; c ++ ; } return c ; }
let x = 12 , y = 15 , z = 5 ; document . write ( " " + smallest ( x , y , z ) ) ;
let CHAR_BIT = 8 ;
function min ( x , y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( 32 * CHAR_BIT - 1 ) ) ) }
function smallest ( x , y , z ) { return Math . min ( x , Math . min ( y , z ) ) ; }
let x = 12 , y = 15 , z = 5 ; document . write ( " " + smallest ( x , y , z ) ) ;
function smallest ( x , y , z ) {
if ( ! ( y / x ) ) return ( ! ( y / z ) ) ? y : z ; return ( ! ( x / z ) ) ? x : z ; }
let x = 78 , y = 88 , z = 68 ; document . write ( " " + smallest ( x , y , z ) ) ;
function changeToZero ( a ) { a [ a [ 1 ] ] = a [ 1 - a [ 1 ] ] ; }
let arr ; arr = [ ] ; arr [ 0 ] = 1 ; arr [ 1 ] = 0 ; changeToZero ( arr ) ; document . write ( " " + arr [ 0 ] + " " ) ; document . write ( " " + arr [ 1 ] ) ;
function addOne ( x ) { let m = 1 ;
while ( x & m ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
document . write ( addOne ( 13 ) ) ;
function addOne ( x ) { return ( - ( ~ x ) ) ; }
document . write ( addOne ( 13 ) ) ;
function fun ( n ) { return n & ( n - 1 ) ; }
let n = 7 ; document . write ( " " + " " + fun ( n ) ) ;
function isPowerOfFour ( n ) { if ( n == 0 ) return false ; while ( n != 1 ) { if ( n % 4 != 0 ) return false ; n = n / 4 ; } return true ; }
let test_no = 64 ; if ( isPowerOfFour ( test_no ) ) document . write ( test_no + " " ) ; else document . write ( test_no + " " ) ;
function isPowerOfFour ( n ) { let count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) {
while ( n > 1 ) { n >>= 1 ; count += 1 ; }
return ( count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
let test_no = 64 ; if ( isPowerOfFour ( test_no ) ) document . write ( test_no + " " ) ; else document . write ( test_no + " " ) ;
function isPowerOfFour ( n ) { return n != 0 && ( ( n & ( n - 1 ) ) == 0 ) && ! ( n & 0xAAAAAAAA ) ; }
test_no = 64 ; if ( isPowerOfFour ( test_no ) ) document . write ( test_no + " " ) ; else document . write ( test_no + " " ) ;
function min ( x , y ) { return y ^ ( ( x ^ y ) & - ( x << y ) ) ; }
function max ( x , y ) { return x ^ ( ( x ^ y ) & - ( x << y ) ) ; }
let x = 15 let y = 6 document . write ( " " + x + " " + y + " " ) ; document . write ( min ( x , y ) + " " ) ; document . write ( " " + x + " " + y + " " ) ; document . write ( max ( x , y ) + " " ) ;
var CHAR_BIT = 4 ; var INT_BIT = 8 ;
function min ( x , y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( INT_BIT * CHAR_BIT - 1 ) ) ) ; }
function max ( x , y ) { return x - ( ( x - y ) & ( ( x - y ) >> ( INT_BIT * CHAR_BIT - 1 ) ) ) ; }
var x = 15 ; var y = 6 ; document . write ( " " + x + " " + y + " " + min ( x , y ) + " " ) ; document . write ( " " + x + " " + y + " " + max ( x , y ) ) ;
function getFirstSetBitPos ( n ) { return Math . log2 ( n & - n ) + 1 ; }
let g ; let n = 12 ; document . write ( getFirstSetBitPos ( n ) ) ;
function bin ( n ) { let i ; document . write ( " " ) ; for ( i = 1 << 30 ; i > 0 ; i = Math . floor ( i / 2 ) ) { if ( ( n & i ) != 0 ) { document . write ( " " ) ; } else { document . write ( " " ) ; } } }
bin ( 7 ) ; document . write ( " " ) ; bin ( 4 ) ;
function swapBits ( x ) {
even_bits = x & 0xAAAAAAAA ;
odd_bits = x & 0x55555555 ;
even_bits >>= 1 ;
odd_bits <<= 1 ;
return ( even_bits odd_bits ) ; }
let x = 23 ;
document . write ( swapBits ( x ) ) ;
function isPowerOfTwo ( n ) { return ( n > 0 && ( ( n & ( n - 1 ) ) == 0 ) ) ? true : false ; }
function findPosition ( n ) { if ( isPowerOfTwo ( n ) == false ) return - 1 ; var i = 1 ; var pos = 1 ;
while ( ( i & n ) == 0 ) {
i = i << 1 ;
pos += 1 ; } return pos ; }
var n = 16 ; var pos = findPosition ( n ) ; if ( pos == - 1 ) document . write ( " " + n + " " ) ; else document . write ( " " + n + " " + pos ) ; document . write ( " " ) ; n = 12 ; pos = findPosition ( n ) ; if ( pos == - 1 ) document . write ( " " + n + " " ) ; else document . write ( " " + n + " " , pos ) ; document . write ( " " ) ; n = 128 ; pos = findPosition ( n ) ; if ( pos == - 1 ) document . write ( " " + n + " " ) ; else document . write ( " " + n + " " + pos ) ;
function isPowerOfTwo ( n ) { return ( n && ( ! ( n & ( n - 1 ) ) ) ) }
function findPosition ( n ) { if ( ! isPowerOfTwo ( n ) ) return - 1 var count = 0
while ( n ) { n = n >> 1
count += 1 } return count }
var n = 0 var pos = findPosition ( n ) if ( pos == - 1 ) document . write ( " " , n , " " ) else document . write ( " " , n , " " , pos ) document . write ( " " ) n = 12 pos = findPosition ( n ) if ( pos == - 1 ) document . write ( " " , n , " " ) else document . write ( " " , n , " " , pos ) document . write ( " " ) n = 128 pos = findPosition ( n ) if ( pos == - 1 ) document . write ( " " , n , " " ) else document . write ( " " , n , " " , pos ) document . write ( " " )
var x = 10 ; var y = 5 ;
x = x * y ;
y = x / y ;
x = x / y ; document . write ( " " + " " + x + " " + y ) ;
let x = 10 , y = 5 ;
x = x ^ y ;
y = x ^ y ;
x = x ^ y ; document . write ( " " + x + " " + y ) ;
function swap ( xp , yp ) { xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] ; yp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] ; xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ] ; }
let x = [ 10 ] ; swap ( x , x ) ; document . write ( " " + x [ 0 ] ) ;
function nextGreatest ( arr , size ) {
max_from_right = arr [ size - 1 ] ;
arr [ size - 1 ] = - 1 ;
for ( let i = size - 2 ; i >= 0 ; i -- ) {
temp = arr [ i ] ;
arr [ i ] = max_from_right ;
if ( max_from_right < temp ) max_from_right = temp ; } }
function printArray ( arr , size ) { var i ; for ( let i = 0 ; i < size ; i ++ ) document . write ( arr [ i ] + " " ) ; }
arr = new Array ( 16 , 17 , 4 , 3 , 5 , 2 ) ; size = arr . length ; nextGreatest ( arr , size ) ; document . write ( " " + " " + " " ) ; printArray ( arr , size ) ;
function maxDiff ( arr , arr_size ) { let max_diff = arr [ 1 ] - arr [ 0 ] ; for ( let i = 0 ; i < arr_size ; i ++ ) { for ( let j = i + 1 ; j < arr_size ; j ++ ) { if ( arr [ j ] - arr [ i ] > max_diff ) max_diff = arr [ j ] - arr [ i ] ; } } return max_diff ; }
let arr = [ 1 , 2 , 90 , 10 , 110 ] ; let n = arr . length ;
document . write ( " " + maxDiff ( arr , n ) ) ;
function findMaximum ( arr , low , high ) { var max = arr [ low ] ; var i ; for ( i = low + 1 ; i <= high ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; else break ; } return max ; }
var arr = [ 1 , 30 , 40 , 50 , 60 , 70 , 23 , 20 ] ; var n = arr . length ; document . write ( " " + findMaximum ( arr , 0 , n - 1 ) ) ;
function findMaximum ( arr , low , high ) {
if ( low == high ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] >= arr [ high ] ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] < arr [ high ] ) return arr [ high ] ; mid = ( low + high ) / 2 ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] > arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] < arr [ mid - 1 ] ) return findMaximum ( arr , low , mid - 1 ) ;
return findMaximum ( arr , mid + 1 , high ) ; }
arr = new Array ( 1 , 3 , 50 , 10 , 9 , 7 , 6 ) ; n = arr . length ; document . write ( " " + " " + findMaximum ( arr , 0 , n - 1 ) ) ;
function constructLowerArray ( arr , countSmaller , n ) { let i , j ;
for ( i = 0 ; i < n ; i ++ ) countSmaller [ i ] = 0 ; for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] < arr [ i ] ) countSmaller [ i ] ++ ; } } }
function printArray ( arr , size ) { let i ; for ( i = 0 ; i < size ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; }
let arr = [ 12 , 10 , 5 , 4 , 2 , 20 , 6 , 1 , 0 , 2 ] ; let n = arr . length ; let low = new Array ( n ) ; constructLowerArray ( arr , low , n ) ; printArray ( low , n ) ;
function segregate ( arr , size ) { let j = 0 , i ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] <= 0 ) { let temp ; temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ;
j ++ ; } } return j ; }
function findMissingPositive ( arr , size ) { let i ;
for ( i = 0 ; i < size ; i ++ ) { let x = Math . abs ( arr [ i ] ) ; if ( x - 1 < size && arr [ x - 1 ] > 0 ) arr [ x - 1 ] = - arr [ x - 1 ] ; }
for ( i = 0 ; i < size ; i ++ ) if ( arr [ i ] > 0 )
return i + 1 ; return size + 1 ; }
function findMissing ( arr , size ) {
let shift = segregate ( arr , size ) ; let arr2 = new Array ( size - shift ) ; let j = 0 ; for ( let i = shift ; i < size ; i ++ ) { arr2 [ j ] = arr [ i ] ; j ++ ; }
return findMissingPositive ( arr2 , j ) ; }
let arr = [ 0 , 10 , 2 , - 10 , - 20 ] ; let arr_size = arr . length ; let missing = findMissing ( arr , arr_size ) ; document . write ( " " + missing ) ;
function getMissingNo ( a , n ) { let total = Math . floor ( ( n + 1 ) * ( n + 2 ) / 2 ) ; for ( let i = 0 ; i < n ; i ++ ) total -= a [ i ] ; return total ; }
let arr = [ 1 , 2 , 4 , 5 , 6 ] ; let n = arr . length ; let miss = getMissingNo ( arr , n ) ; document . write ( miss ) ;
function printTwoElements ( arr , size ) { var i ; document . write ( " " ) ; for ( i = 0 ; i < size ; i ++ ) { var abs_value = Math . abs ( arr [ i ] ) ; if ( arr [ abs_value - 1 ] > 0 ) arr [ abs_value - 1 ] = - arr [ abs_value - 1 ] ; else document . write ( abs_value ) ; } document . write ( " " ) ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) document . write ( i + 1 ) ; } }
arr = new Array ( 7 , 3 , 4 , 5 , 5 , 6 , 2 ) ; n = arr . length ; printTwoElements ( arr , n ) ;
function findFourElements ( A , n , X ) {
for ( let i = 0 ; i < n - 3 ; i ++ ) {
for ( let j = i + 1 ; j < n - 2 ; j ++ ) {
for ( let k = j + 1 ; k < n - 1 ; k ++ ) {
for ( let l = k + 1 ; l < n ; l ++ ) if ( A [ i ] + A [ j ] + A [ k ] + A [ l ] == X ) document . write ( A [ i ] + " " + A [ j ] + " " + A [ k ] + " " + A [ l ] ) ; } } } }
let A = [ 10 , 20 , 30 , 40 , 1 , 2 ] ; let n = A . length ; let X = 91 ; findFourElements ( A , n , X ) ;
function minDistance ( arr , n ) { let maximum_element = arr [ 0 ] ; let min_dis = n ; let index = 0 ; for ( let i = 1 ; i < n ; i ++ ) {
if ( maximum_element == arr [ i ] ) { min_dis = Math . min ( min_dis , ( i - index ) ) ; index = i ; }
else if ( maximum_element < arr [ i ] ) { maximum_element = arr [ i ] ; min_dis = n ; index = i ; }
else continue ; } return min_dis ; }
let arr = [ 6 , 3 , 1 , 3 , 6 , 4 , 6 ] ; let n = arr . length ; document . write ( " " + minDistance ( arr , n ) ) ;
class Node { constructor ( val ) { this . data = val ; this . next = null ; } }
function push ( new_data ) {
var new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
function search ( head , x ) {
if ( head == null ) return false ;
if ( head . data == x ) return true ;
return search ( head . next , x ) ; }
push ( 10 ) ; push ( 30 ) ; push ( 11 ) ; push ( 21 ) ; push ( 14 ) ; if ( search ( head , 21 ) ) document . write ( " " ) ; else document . write ( " " ) ;
function deleteAlt ( head ) { if ( head == null ) return ; var node = head . next ; if ( node == null ) return ;
head . next = node . next ;
head . next = deleteAlt ( head . next ) ; }
function AlternatingSplit ( source , aRef , bRef ) { var aDummy = new Node ( ) ; var aTail = aDummy ;
var bDummy = new Node ( ) ; var bTail = bDummy ;
var current = source ; aDummy . next = null ; bDummy . next = null ; while ( current != null ) { MoveNode ( ( aTail . next ) , current ) ;
aTail = aTail . next ;
if ( current != null ) { MoveNode ( ( bTail . next ) , current ) ; bTail = bTail . next ; } } aRef = aDummy . next ; bRef = bDummy . next ; }
function areIdenticalRecur ( a , b ) {
if ( a == null && b == null ) return true ;
if ( a != null && b != null ) return ( a . data == b . data ) && areIdenticalRecur ( a . next , b . next ) ;
return false ; }
class Node { constructor ( val ) { this . data = val ; this . next = null ; } } function sortList ( ) {
var count = [ 0 , 0 , 0 ] ; var ptr = head ;
while ( ptr != null ) { count [ ptr . data ] ++ ; ptr = ptr . next ; } var i = 0 ; ptr = head ;
while ( ptr != null ) { if ( count [ i ] == 0 ) i ++ ; else { ptr . data = i ; -- count [ i ] ; ptr = ptr . next ; } } }
function push ( new_data ) {
var new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
function printList ( ) { var temp = head ; while ( temp != null ) { document . write ( temp . data + " " ) ; temp = temp . next ; } document . write ( " " ) ; }
push ( 0 ) ; push ( 1 ) ; push ( 0 ) ; push ( 2 ) ; push ( 1 ) ; push ( 1 ) ; push ( 2 ) ; push ( 1 ) ; push ( 2 ) ; document . write ( " " ) ; printList ( ) ; sortList ( ) ; document . write ( " " ) ; printList ( ) ;
class List { constructor ( ) { this . data = 0 ; this . next = null ; this . child = null ; } }
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } }
function newNode ( key ) { var temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
function rearrangeEvenOdd ( head ) {
if ( head == null ) return null ;
var odd = head ; var even = head . next ;
var evenFirst = even ; while ( 1 == 1 ) {
if ( odd == null || even == null || ( even . next ) == null ) { odd . next = evenFirst ; break ; }
odd . next = even . next ; odd = even . next ;
if ( odd . next == null ) { even . next = null ; odd . next = evenFirst ; break ; }
even . next = odd . next ; even = odd . next ; } return head ; }
function printlist ( node ) { while ( node != null ) { document . write ( node . data + " " ) ; node = node . next ; } document . write ( " " ) ; }
var head = newNode ( 1 ) ; head . next = newNode ( 2 ) ; head . next . next = newNode ( 3 ) ; head . next . next . next = newNode ( 4 ) ; head . next . next . next . next = newNode ( 5 ) ; document . write ( " " ) ; printlist ( head ) ; head = rearrangeEvenOdd ( head ) ; document . write ( " " ) ; printlist ( head ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } }
function deleteLast ( head , x ) { var temp = head , ptr = null ; while ( temp != null ) {
if ( temp . data == x ) ptr = temp ; temp = temp . next ; }
if ( ptr != null && ptr . next == null ) { temp = head ; while ( temp . next != ptr ) temp = temp . next ; temp . next = null ; }
if ( ptr != null && ptr . next != null ) { ptr . data = ptr . next . data ; temp = ptr . next ; ptr . next = ptr . next . next ; } }
function newNode ( x ) { var node = new Node ( ) ; node . data = x ; node . next = null ; return node ; }
function display ( head ) { var temp = head ; if ( head == null ) { document . write ( " " ) ; return ; } while ( temp != null ) { document . write ( temp . data + " " ) ; temp = temp . next ; } document . write ( " " ) ; }
var head = newNode ( 1 ) ; head . next = newNode ( 2 ) ; head . next . next = newNode ( 3 ) ; head . next . next . next = newNode ( 4 ) ; head . next . next . next . next = newNode ( 5 ) ; head . next . next . next . next . next = newNode ( 4 ) ; head . next . next . next . next . next . next = newNode ( 4 ) ; document . write ( " " ) ; display ( head ) ; deleteLast ( head , 4 ) ; document . write ( " " ) ; display ( head ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } }
function LinkedListLength ( head ) { while ( head != null && head . next != null ) { head = head . next . next ; } if ( head == null ) return 0 ; return 1 ; }
function push ( head , info ) {
node = new Node ( ) ;
node . data = info ;
node . next = ( head ) ;
( head ) = node ; }
head = null ;
push ( head , 4 ) ; push ( head , 5 ) ; push ( head , 7 ) ; push ( head , 2 ) ; push ( head , 9 ) ; push ( head , 6 ) ; push ( head , 1 ) ; push ( head , 2 ) ; push ( head , 0 ) ; push ( head , 5 ) ; push ( head , 5 ) ; var check = LinkedListLength ( head ) ;
if ( check == 0 ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function SortedMerge ( a , b ) { let result = null ;
let lastPtrRef = result ; while ( 1 ) { if ( a == null ) { lastPtrRef = b ; break ; } else if ( b == null ) { lastPtrRef = a ; break ; } if ( a . data <= b . data ) { MoveNode ( lastPtrRef , a ) ; } else { MoveNode ( lastPtrRef , b ) ; }
lastPtrRef = ( ( lastPtrRef ) . next ) ; } return ( result ) ; }
class Node { constructor ( val ) { this . data = val ; this . next = null ; } } var head ;
function setMiddleHead ( ) { if ( head == null ) return ;
one_node = head ;
two_node = head ;
prev = null ; while ( two_node != null && two_node . next != null ) {
prev = one_node ;
two_node = two_node . next . next ;
one_node = one_node . next ; }
prev . next = prev . next . next ; one_node . next = head ; head = one_node ; }
function push ( new_data ) {
new_node = new Node ( new_data ) ;
new_node . next = head ;
head = new_node ; }
function printList ( ptr ) { while ( ptr != null ) { document . write ( ptr . data + " " ) ; ptr = ptr . next ; } document . write ( " " ) ; }
head = null ; var i ; for ( i = 5 ; i > 0 ; i -- ) push ( i ) ; document . write ( " " ) ; printList ( head ) ; setMiddleHead ( ) ; document . write ( " " ) ; printList ( head ) ;
function InsertAfter ( prev_Node , new_data ) {
if ( prev_Node == null ) { document . write ( " " ) ; return ; }
let new_node = new Node ( new_data ) ;
new_node . next = prev_Node . next ;
prev_Node . next = new_node ;
new_node . prev = prev_Node ;
if ( new_node . next != null ) new_node . next . prev = new_node ; }
class Node { constructor ( item ) { this . data = item ; this . left = null ; this . right = null ; } } var root = null ; function printKDistant ( node , k ) { if ( node == null k < 0 ) { return ; } if ( k == 0 ) { document . write ( node . data + " " ) ; return ; } printKDistant ( node . left , k - 1 ) ; printKDistant ( node . right , k - 1 ) ; }
root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . left . left = new Node ( 4 ) ; root . left . right = new Node ( 5 ) ; root . right . left = new Node ( 8 ) ; printKDistant ( root , 2 ) ;
let COUNT = 10 ;
class Node {
constructor ( data ) { this . data = data ; this . left = null ; this . right = null ; } }
function print2DUtil ( root , space ) {
if ( root == null ) return ;
space += COUNT ;
print2DUtil ( root . right , space ) ;
document . write ( " " ) ; for ( let i = COUNT ; i < space ; i ++ ) document . write ( " " ) ; document . write ( root . data + " " ) ;
print2DUtil ( root . left , space ) ; }
function print2D ( root ) {
print2DUtil ( root , 0 ) ; }
let root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . left . left = new Node ( 4 ) ; root . left . right = new Node ( 5 ) ; root . right . left = new Node ( 6 ) ; root . right . right = new Node ( 7 ) ; root . left . left . left = new Node ( 8 ) ; root . left . left . right = new Node ( 9 ) ; root . left . right . left = new Node ( 10 ) ; root . left . right . right = new Node ( 11 ) ; root . right . left . left = new Node ( 12 ) ; root . right . left . right = new Node ( 13 ) ; root . right . right . left = new Node ( 14 ) ; root . right . right . right = new Node ( 15 ) ; print2D ( root ) ;
class Node { constructor ( item ) { this . data = item ; this . left = null ; this . right = null ; } }
function leftViewUtil ( node , level ) {
if ( node == null ) { return ; }
if ( max_level < level ) { document . write ( " " + node . data ) ; max_level = level ; }
leftViewUtil ( node . left , level + 1 ) ; leftViewUtil ( node . right , level + 1 ) ; }
function leftView ( ) { leftViewUtil ( root , 1 ) ; }
root = new Node ( 12 ) ; root . left = new Node ( 10 ) ; root . right = new Node ( 30 ) ; root . right . left = new Node ( 25 ) ; root . right . right = new Node ( 40 ) ; leftView ( ) ;
function cntRotations ( s , n ) { let lh = 0 , rh = 0 , i , ans = 0 ;
for ( i = 0 ; i < parseInt ( n / 2 , 10 ) ; ++ i ) if ( s [ i ] == ' ' s [ i ] == ' ' s [ i ] == ' ' s [ i ] == ' ' s [ i ] == ' ' ) { lh ++ ; }
for ( i = parseInt ( n / 2 , 10 ) ; i < n ; ++ i ) if ( s [ i ] == ' ' s [ i ] == ' ' s [ i ] == ' ' s [ i ] == ' ' s [ i ] == ' ' ) { rh ++ ; }
if ( lh > rh ) ans ++ ;
for ( i = 1 ; i < n ; ++ i ) { if ( s [ i - 1 ] == ' ' s [ i - 1 ] == ' ' s [ i - 1 ] == ' ' s [ i - 1 ] == ' ' s [ i - 1 ] == ' ' ) { rh ++ ; lh -- ; } if ( s [ ( i - 1 + n / 2 ) % n ] == ' ' || s [ ( i - 1 + n / 2 ) % n ] == ' ' || s [ ( i - 1 + n / 2 ) % n ] == ' ' || s [ ( i - 1 + n / 2 ) % n ] == ' ' || s [ ( i - 1 + n / 2 ) % n ] == ' ' ) { rh -- ; lh ++ ; } if ( lh > rh ) ans ++ ; }
return ans ; }
let s = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ] ; let n = s . length ;
document . write ( cntRotations ( s , n ) ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } } var tail ;
function rotateHelper ( blockHead , blockTail , d , k ) { if ( d == 0 ) return blockHead ;
if ( d > 0 ) { var temp = blockHead ; for ( i = 1 ; temp . next . next != null && i < k - 1 ; i ++ ) temp = temp . next ; blockTail . next = blockHead ; tail = temp ; return rotateHelper ( blockTail , temp , d - 1 , k ) ; }
if ( d < 0 ) { blockTail . next = blockHead ; tail = blockHead ; return rotateHelper ( blockHead . next , blockHead , d + 1 , k ) ; } return blockHead ; }
function rotateByBlocks ( head , k , d ) {
if ( head == null head . next == null ) return head ;
if ( d == 0 ) return head ; var temp = head ; tail = null ;
var i ; for ( i = 1 ; temp . next != null && i < k ; i ++ ) temp = temp . next ;
var nextBlock = temp . next ;
if ( i < k ) head = rotateHelper ( head , temp , d % k , i ) ; else head = rotateHelper ( head , temp , d % k , k ) ;
tail . next = rotateByBlocks ( nextBlock , k , d % k ) ;
return head ; }
function push ( head_ref , new_data ) { var new_node = new Node ( ) ; new_node . data = new_data ; new_node . next = head_ref ; head_ref = new_node ; return head_ref ; }
function printList ( node ) { while ( node != null ) { document . write ( node . data + " " ) ; node = node . next ; } }
var head = null ;
for ( i = 9 ; i > 0 ; i -= 1 ) head = push ( head , i ) ; document . write ( " " ) ; printList ( head ) ;
var k = 3 , d = 2 ; head = rotateByBlocks ( head , k , d ) ; document . write ( " " ) ; printList ( head ) ;
function DeleteFirst ( head ) { previous = head , firstNode = head ;
if ( head == null ) { document . write ( " " ) ; return ; }
if ( previous . next == previous ) { head = null ; return ; }
while ( previous . next != head ) { previous = previous . next ; }
previous . next = firstNode . next ;
head = previous . next ; return ; }
function DeleteLast ( head ) { let current = head , temp = head , previous = null ;
if ( head == null ) { document . write ( " " ) ; return null ; }
if ( current . next == current ) { head = null ; return null ; }
while ( current . next != head ) { previous = current ; current = current . next ; } previous . next = current . next ; head = previous . next ; return head ; }
function countSubarrays ( arr , n ) {
var count = 0 ; var i , j ;
for ( i = 0 ; i < n ; i ++ ) { var sum = 0 ; for ( j = i ; j < n ; j ++ ) {
if ( ( j - i ) % 2 == 0 ) sum += arr [ j ] ;
else sum -= arr [ j ] ;
if ( sum == 0 ) count ++ ; } }
document . write ( count ) ; }
var arr = [ 2 , 4 , 6 , 4 , 2 ] ;
var n = arr . length ;
countSubarrays ( arr , n ) ;
function printAlter ( arr , N ) {
for ( var currIndex = 0 ; currIndex < N ; currIndex ++ ) {
if ( currIndex % 2 == 0 ) { document . write ( arr [ currIndex ] + " " ) ; } } }
var arr = [ 1 , 2 , 3 , 4 , 5 ] var N = arr . length ; printAlter ( arr , N ) ;
function reverse ( arr , start , end ) {
let mid = ( end - start + 1 ) / 2 ;
for ( let i = 0 ; i < mid ; i ++ ) {
let temp = arr [ start + i ] ;
arr [ start + i ] = arr [ end - i ] ;
arr [ end - i ] = temp ; } return ; }
function shuffleArrayUtil ( arr , start , end ) { let i ;
let l = end - start + 1 ;
if ( l == 2 ) return ;
let mid = start + l / 2 ;
if ( l % 4 > 0 ) {
mid -= 1 ; }
let mid1 = start + ( mid - start ) / 2 ; let mid2 = mid + ( end + 1 - mid ) / 2 ;
reverse ( arr , mid1 , mid2 - 1 ) ;
reverse ( arr , mid1 , mid - 1 ) ;
reverse ( arr , mid , mid2 - 1 ) ;
shuffleArrayUtil ( arr , start , mid - 1 ) ; shuffleArrayUtil ( arr , mid , end ) ; }
function shuffleArray ( arr , N , start , end ) {
shuffleArrayUtil ( arr , start , end ) ;
for ( let i = 0 ; i < N ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 1 , 3 , 5 , 2 , 4 , 6 ] ;
let N = arr . length ;
shuffleArray ( arr , N , 0 , N - 1 ) ;
function canMadeEqual ( A , B , n ) {
A . sort ( ) ; B . sort ( ) ;
for ( var i = 0 ; i < n ; i ++ ) if ( A [ i ] != B [ i ] ) return false ; return true ; }
var A = [ 1 , 2 , 3 ] ; var B = [ 1 , 3 , 2 ] ; var n = A . length ; if ( canMadeEqual ( A , B , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function merge ( arr , start , mid , end ) { let start2 = mid + 1 ;
if ( arr [ mid ] <= arr [ start2 ] ) { return ; }
while ( start <= mid && start2 <= end ) {
if ( arr [ start ] <= arr [ start2 ] ) { start ++ ; } else { let value = arr [ start2 ] ; let index = start2 ;
while ( index != start ) { arr [ index ] = arr [ index - 1 ] ; index -- ; } arr [ start ] = value ;
start ++ ; mid ++ ; start2 ++ ; } } }
function mergeSort ( arr , l , r ) { if ( l < r ) {
let m = l + Math . floor ( ( r - l ) / 2 ) ;
mergeSort ( arr , l , m ) ; mergeSort ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } }
function printArray ( A , size ) { let i ; for ( i = 0 ; i < size ; i ++ ) document . write ( A [ i ] + " " ) ; document . write ( " " ) ; }
let arr = [ 12 , 11 , 13 , 5 , 6 , 7 ] ; let arr_size = arr . length ; mergeSort ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ;
class pair { constructor ( first , second ) { this [ 0 ] = first ; this [ 1 ] = second ; } }
function constGraphWithCon ( N , K ) {
var Max = ( ( N - 1 ) * ( N - 2 ) ) / 2 ;
if ( K > Max ) { document . write ( - 1 + " " ) ; return ; }
var ans = [ ] ;
for ( var i = 1 ; i < N ; i ++ ) { for ( var j = i + 1 ; j <= N ; j ++ ) { ans . push ( [ i , j ] ) ; } }
for ( var i = 0 ; i < ( N - 1 ) + Max - K ; i ++ ) { document . write ( ans [ i ] [ 0 ] + " " + ans [ i ] [ 1 ] + " " ) ; } }
var N = 5 , K = 3 ; constGraphWithCon ( N , K ) ;
function findArray ( N , K ) {
if ( N == 1 ) { document . write ( K + " " ) ; return ; } if ( N == 2 ) { document . write ( 0 + " " + K ) ; return ; }
let P = N - 2 ; let Q = N - 1 ;
let VAL = 0 ;
for ( let i = 1 ; i <= ( N - 3 ) ; i ++ ) { document . write ( i + " " ) ;
VAL ^= i ; } if ( VAL == K ) { document . write ( P + " " + Q + " " + ( P ^ Q ) ) ; } else { document . write ( 0 + " " + P + " " + ( P ^ K ^ VAL ) ) ; } }
let N = 4 , X = 6 ;
findArray ( N , X ) ;
function countDigitSum ( N , K ) {
let l = parseInt ( Math . pow ( 10 , N - 1 ) ) , r = parseInt ( Math . pow ( 10 , N ) - 1 ) ; let count = 0 ; for ( let i = l ; i <= r ; i ++ ) { let num = i ;
let digits = new Array ( N ) ; for ( let j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num = parseInt ( num / 10 ) ; } let sum = 0 , flag = 0 ;
for ( let j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( let j = 1 ; j < N - K + 1 ; j ++ ) { let curr_sum = 0 ; for ( let m = j ; m < j + K ; m ++ ) curr_sum += digits [ m ] ;
if ( sum != curr_sum ) { flag = 1 ; break ; } }
if ( flag == 0 ) { count ++ ; } } return count ; }
let N = 2 , K = 1 ;
document . write ( countDigitSum ( N , K ) ) ;
function arithmeticThree ( set , n ) {
for ( let j = 1 ; j < n - 1 ; j ++ ) {
let i = j - 1 , k = j + 1 ;
while ( i >= 0 && k <= n - 1 ) { if ( set [ i ] + set [ k ] == 2 * set [ j ] ) return true ; ( set [ i ] + set [ k ] < 2 * set [ j ] ) ? k ++ : i -- ; } } return false ; }
function maxSumIS ( arr , n ) { let i , j , max = 0 ; let msis = new Array ( n ) ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
let arr = [ 1 , 101 , 2 , 3 , 100 , 4 , 5 ] ; let n = arr . length ; document . write ( " " + " " + maxSumIS ( arr , n ) ) ;
function reverse ( str , start , end ) {
let temp ; while ( start <= end ) {
temp = str [ start ] ; str [ start ] = str [ end ] ; str [ end ] = temp ; start ++ ; end -- ; } }
function reverseletter ( str , start , end ) { let wstart , wend ; for ( wstart = wend = start ; wend < end ; wend ++ ) { if ( str [ wend ] == ' ' ) { continue ; }
while ( wend <= end && str [ wend ] != ' ' ) { wend ++ ; } wend -- ;
reverse ( str , wstart , wend ) ; } }
let str = " " . split ( " " ) ; reverseletter ( str , 0 , str . length - 1 ) ; document . write ( ( str ) . join ( " " ) ) ;
function have_same_frequency ( freq , k ) { for ( let i = 0 ; i < 26 ; i ++ ) { if ( freq [ i ] != 0 && freq [ i ] != k ) { return false ; } } return true ; } function count_substrings ( s , k ) { let count = 0 ; let distinct = 0 ; let have = new Array ( 26 ) ; for ( let i = 0 ; i < 26 ; i ++ ) { have [ i ] = false ; } for ( let i = 0 ; i < s . length ; i ++ ) { have [ ( ( s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ) ] = true ; } for ( let i = 0 ; i < 26 ; i ++ ) { if ( have [ i ] ) { distinct ++ ; } } for ( let length = 1 ; length <= distinct ; length ++ ) { let window_length = length * k ; let freq = new Array ( 26 ) ; for ( let i = 0 ; i < 26 ; i ++ ) freq [ i ] = 0 ; let window_start = 0 ; let window_end = window_start + window_length - 1 ; for ( let i = window_start ; i <= Math . min ( window_end , s . length - 1 ) ; i ++ ) { freq [ ( ( s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ) ] ++ ; } while ( window_end < s . length ) { if ( have_same_frequency ( freq , k ) ) { count ++ ; } freq [ ( ( s [ window_start ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ) ] -- ; window_start ++ ; window_end ++ ; if ( window_end < s . length ) { freq [ ( s [ window_end ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ] ++ ; } } } return count ; } let s = " " ; let k = 2 ; document . write ( count_substrings ( s , k ) + " " ) ; s = " " ; k = 2 ; document . write ( count_substrings ( s , k ) + " " ) ;
x = 32 ;
function toggleCase ( a ) { for ( i = 0 ; i < a . length ; i ++ ) {
a [ i ] = String . fromCharCode ( a [ i ] . charCodeAt ( 0 ) ^ 32 ) ; } return a . join ( " " ) ; ; }
var str = " " ; document . write ( " " ) ; str = toggleCase ( str . split ( ' ' ) ) ; document . write ( str ) ; document . write ( " " ) ; str = toggleCase ( str . split ( ' ' ) ) ; document . write ( str ) ;
let NO_OF_CHARS = 256 ;
function areAnagram ( str1 , str2 ) {
let count1 = new Array ( NO_OF_CHARS ) ; let count2 = new Array ( NO_OF_CHARS ) ; for ( let i = 0 ; i < NO_OF_CHARS ; i ++ ) { count1 [ i ] = 0 ; count2 [ i ] = 0 ; } let i ;
for ( i = 0 ; i < str1 . length && i < str2 . length ; i ++ ) { count1 [ str1 [ i ] . charCodeAt ( 0 ) ] ++ ; count2 [ str1 [ i ] . charCodeAt ( 0 ) ] ++ ; }
if ( str1 . length != str2 . length ) return false ;
for ( i = 0 ; i < NO_OF_CHARS ; i ++ ) if ( count1 [ i ] != count2 [ i ] ) return false ; return true ; }
let str1 = ( " " ) . split ( " " ) ; let str2 = ( " " ) . split ( " " ) ;
if ( areAnagram ( str1 , str2 ) ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function heptacontagonNum ( n ) { return ( 68 * n * n - 66 * n ) / 2 ; }
var N = 3 ; document . write ( " " + heptacontagonNum ( N ) ) ;
function isEqualFactors ( N ) { if ( ( N % 2 == 0 ) && ( N % 4 != 0 ) ) document . write ( " " ) ; else document . write ( " " ) ; }
var N = 10 ; isEqualFactors ( N ) ; N = 125 ; isEqualFactors ( N ) ;
function checkDivisibility ( n , digit ) {
return ( digit != 0 && n % digit == 0 ) ; }
function isAllDigitsDivide ( n ) { let temp = n ; while ( temp > 0 ) {
let digit = temp % 10 ; if ( ! ( checkDivisibility ( n , digit ) ) ) return false ; temp = parseInt ( temp / 10 ) ; } return true ; }
function isAllDigitsDistinct ( n ) {
let arr = Array ( 10 ) . fill ( 0 ) ;
while ( n > 0 ) {
let digit = n % 10 ;
if ( arr [ digit ] ) return false ;
arr [ digit ] = true ;
n = parseInt ( n / 10 ) ; } return true ; }
function isLynchBell ( n ) { return isAllDigitsDivide ( n ) && isAllDigitsDistinct ( n ) ; }
let N = 12 ;
if ( isLynchBell ( N ) ) document . write ( " " ) ; else document . write ( " " ) ;
function maximumAND ( L , R ) { return R ; }
let l = 3 ; let r = 7 ; document . write ( maximumAND ( l , r ) ) ;
function findAverageOfCube ( n ) {
let sum = 0 ;
let i ; for ( i = 1 ; i <= n ; i ++ ) { sum += i * i * i ; }
return sum / n ; }
let n = 3 ;
document . write ( findAverageOfCube ( n ) . toFixed ( 6 ) ) ;
function isPower ( N , K ) {
var res1 = Math . floor ( Math . log ( N ) / Math . log ( K ) ) ; var res2 = Math . log ( N ) / Math . log ( K ) ;
return ( res1 == res2 ) ; }
var N = 8 ; var K = 2 ; if ( isPower ( N , K ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function y ( x ) { return ( 1 / ( 1 + x ) ) ; }
function BooleRule ( a , b ) {
let n = 4 ; let h ;
h = ( ( b - a ) / n ) ; let sum = 0 ;
let bl = ( 7 * y ( a ) + 32 * y ( a + h ) + 12 * y ( a + 2 * h ) + 32 * y ( a + 3 * h ) + 7 * y ( a + 4 * h ) ) * 2 * h / 45 ; sum = sum + bl ; return sum ; }
document . write ( " " + BooleRule ( 0 , 4 ) . toFixed ( 4 ) ) ;
function y ( x ) { let num = 1 ; let denom = 1.0 + x * x ; return num / denom ; }
function WeedleRule ( a , b ) {
let h = ( b - a ) / 6 ;
let sum = 0 ;
sum = sum + ( ( ( 3 * h ) / 10 ) * ( y ( a ) + y ( a + 2 * h ) + 5 * y ( a + h ) + 6 * y ( a + 3 * h ) + y ( a + 4 * h ) + 5 * y ( a + 5 * h ) + y ( a + 6 * h ) ) ) ;
return sum . toFixed ( 6 ) ; }
let a = 0 , b = 6 ;
let num = WeedleRule ( a , b ) ; document . write ( " " + num ) ;
function dydx ( x , y ) { return ( x + y - 2 ) ; }
function rungeKutta ( x0 , y0 , x , h ) {
let n = ( ( x - x0 ) / h ) ; let k1 , k2 ;
let y = y0 ; for ( let i = 1 ; i <= n ; i ++ ) {
k1 = h * dydx ( x0 , y ) ; k2 = h * dydx ( x0 + 0.5 * h , y + 0.5 * k1 ) ;
y = y + ( 1.0 / 6.0 ) * ( k1 + 2 * k2 ) ;
x0 = x0 + h ; } return y ; }
let x0 = 0 , y = 1 , x = 2 , h = 0.2 ; document . write ( rungeKutta ( x0 , y , x , h ) . toFixed ( 6 ) ) ;
function per ( a , b ) { return ( a + b ) ; }
function area ( s ) { return ( s / 2 ) ; }
var a = 7 , b = 8 , s = 10 ; document . write ( per ( a , b ) ) ; document . write ( area ( s ) ) ;
const PI = 3.14159265 ;
function area_leaf ( a ) { return ( a * a * ( PI / 2 - 1 ) ) ; }
let a = 7 ; document . write ( Math . round ( area_leaf ( a ) ) ) ;
const PI = 3.14159265 ;
function length_rope ( r ) { return ( ( 2 * PI * r ) + 6 * r ) ; }
let r = 7 ; document . write ( Math . ceil ( length_rope ( r ) ) ) ;
let PI = 3.14159265 ;
function area_inscribed ( P , B , H ) { return ( ( P + B - H ) * ( P + B - H ) * ( PI / 4 ) ) ; }
var P = 3 , B = 4 , H = 5 ; document . write ( area_inscribed ( P , B , H ) . toFixed ( 6 ) ) ;
let PI = 3.14159265 ;
function area_cicumscribed ( c ) { return ( c * c * ( PI / 4 ) ) ; }
var c = 8.0 ; document . write ( area_cicumscribed ( c ) . toFixed ( 6 ) ) ;
function area ( r ) {
return ( ( 0.5 ) * ( 3.14 ) * ( r * r ) ) ; }
function perimeter ( r ) {
return ( ( 3.14 ) * ( r ) ) ; }
var r = 10 ;
document . write ( " " + area ( r ) . toFixed ( 6 ) + " " ) ;
document . write ( " " + perimeter ( r ) . toFixed ( 6 ) + " " ) ;
function equation_plane ( x1 , y1 , z1 , x2 , y2 , z2 , x3 , y3 , z3 ) { var a1 = x2 - x1 ; var b1 = y2 - y1 ; var c1 = z2 - z1 ; var a2 = x3 - x1 ; var b2 = y3 - y1 ; var c2 = z3 - z1 ; var a = b1 * c2 - b2 * c1 ; var b = a2 * c1 - a1 * c2 ; var c = a1 * b2 - b1 * a2 ; var d = ( - a * x1 - b * y1 - c * z1 ) ; document . write ( " " + a + " " + b + " " + c + " " + d + " " ) ; }
var x1 = - 1 ; var y1 = 2 ; var z1 = 1 ; var x2 = 0 ; var y2 = - 3 ; var z2 = 2 ; var x3 = 1 ; var y3 = 1 ; var z3 = - 4 ; equation_plane ( x1 , y1 , z1 , x2 , y2 , z2 , x3 , y3 , z3 ) ;
function shortest_distance ( x1 , y1 , a , b , c ) { var d = Math . abs ( ( ( a * x1 + b * y1 + c ) ) / ( Math . sqrt ( a * a + b * b ) ) ) ; document . write ( " " + " " + d . toFixed ( 11 ) ) ; return ; }
var x1 = 5 ; var y1 = 6 ; var a = - 2 ; var b = 3 ; var c = 4 ; shortest_distance ( x1 , y1 , a , b , c ) ;
function octant ( x , y , z ) { if ( x >= 0 && y >= 0 && z >= 0 ) document . write ( " " + " " ) ; else if ( x < 0 && y >= 0 && z >= 0 ) document . write ( " " + " " ) ; else if ( x < 0 && y < 0 && z >= 0 ) document . write ( " " + " " ) ; else if ( x >= 0 && y < 0 && z >= 0 ) document . write ( " " + " " ) ; else if ( x >= 0 && y >= 0 && z < 0 ) document . write ( " " + " " ) ; else if ( x < 0 && y >= 0 && z < 0 ) document . write ( " " + " " ) ; else if ( x < 0 && y < 0 && z < 0 ) document . write ( " " + " " ) ; else if ( x >= 0 && y < 0 && z < 0 ) document . write ( " " + " " ) ; }
let x = 2 , y = 3 , z = 4 ; octant ( x , y , z ) ; x = - 4 , y = 2 , z = - 8 ; octant ( x , y , z ) ; x = - 6 , y = - 2 , z = 8 ; octant ( x , y , z ) ;
function maxArea ( a , b , c , d ) {
let semiperimeter = ( a + b + c + d ) / 2 ;
return Math . sqrt ( ( semiperimeter - a ) * ( semiperimeter - b ) * ( semiperimeter - c ) * ( semiperimeter - d ) ) ; }
let a = 1 , b = 2 , c = 1 , d = 2 ; document . write ( maxArea ( a , b , c , d ) ) ;
function addAP ( A , Q , operations ) {
for ( let Q of operations ) { let L = Q [ 0 ] , R = Q [ 1 ] , a = Q [ 2 ] , d = Q [ 3 ] curr = a
for ( let i = L - 1 ; i < R ; i ++ ) {
A [ i ] += curr
curr += d } }
for ( let i of A ) { document . write ( i + " " ) } }
let A = [ 5 , 4 , 2 , 8 ] let Q = 2 let Query = [ [ 1 , 2 , 1 , 3 ] , [ 1 , 4 , 4 , 1 ] ]
addAP ( A , Q , Query )
function log_a_to_base_b ( a , b ) { return parseInt ( Math . log ( a ) / Math . log ( b ) ) ; }
var a = 3 ; var b = 2 ; document . write ( log_a_to_base_b ( a , b ) + " " ) ; a = 256 ; b = 4 ; document . write ( log_a_to_base_b ( a , b ) ) ;
function log_a_to_base_b ( a , b ) { var rslt = ( a > b - 1 ) ? 1 + log_a_to_base_b ( parseInt ( a / b ) , b ) : 0 ; return rslt ; }
var a = 3 ; var b = 2 ; document . write ( log_a_to_base_b ( a , b ) + " " ) ; a = 256 ; b = 4 ; document . write ( log_a_to_base_b ( a , b ) ) ;
function maximum ( x , y ) { return ( ( x + y + Math . abs ( x - y ) ) / 2 ) ; }
function minimum ( x , y ) { return ( ( x + y - Math . abs ( x - y ) ) / 2 ) ; }
let x = 99 , y = 18 ;
document . write ( " " + maximum ( x , y ) + " " ) ;
document . write ( " " + minimum ( x , y ) ) ;
p = 1 , f = 1 ; function e ( x , n ) { var r ;
if ( n == 0 ) return 1 ;
r = e ( x , n - 1 ) ;
p = p * x ;
f = f * n ; return ( r + p / f ) ; }
var x = 4 , n = 15 ; var res = e ( x , n ) ; document . write ( res . toFixed ( 6 ) ) ;
function midptellipse ( rx , ry , xc , yc ) { var dx , dy , d1 , d2 , x , y ; x = 0 ; y = ry ;
d1 = ( ry * ry ) - ( rx * rx * ry ) + ( 0.25 * rx * rx ) ; dx = 2 * ry * ry * x ; dy = 2 * rx * rx * y ;
while ( dx < dy ) {
document . write ( " " + ( x + xc ) . toFixed ( 5 ) + " " + ( y + yc ) . toFixed ( 5 ) + " " + " " ) ; document . write ( " " + ( - x + xc ) . toFixed ( 5 ) + " " + ( y + yc ) . toFixed ( 5 ) + " " + " " ) ; document . write ( " " + ( x + xc ) . toFixed ( 5 ) + " " + ( - y + yc ) . toFixed ( 5 ) + " " + " " ) ; document . write ( " " + ( - x + xc ) . toFixed ( 5 ) + " " + ( - y + yc ) . toFixed ( 5 ) + " " + " " ) ;
if ( d1 < 0 ) { x ++ ; dx = dx + ( 2 * ry * ry ) ; d1 = d1 + dx + ( ry * ry ) ; } else { x ++ ; y -- ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d1 = d1 + dx - dy + ( ry * ry ) ; } }
d2 = ( ( ry * ry ) * ( ( x + 0.5 ) * ( x + 0.5 ) ) ) + ( ( rx * rx ) * ( ( y - 1 ) * ( y - 1 ) ) ) - ( rx * rx * ry * ry ) ;
while ( y >= 0 ) {
document . write ( " " + ( x + xc ) . toFixed ( 5 ) + " " + ( y + yc ) . toFixed ( 5 ) + " " + " " ) ; document . write ( " " + ( - x + xc ) . toFixed ( 5 ) + " " + ( y + yc ) . toFixed ( 5 ) + " " + " " ) ; document . write ( " " + ( x + xc ) . toFixed ( 5 ) + " " + ( - y + yc ) . toFixed ( 5 ) + " " + " " ) ; document . write ( " " + ( - x + xc ) . toFixed ( 5 ) + " " + ( - y + yc ) . toFixed ( 5 ) + " " + " " ) ;
if ( d2 > 0 ) { y -- ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + ( rx * rx ) - dy ; } else { y -- ; x ++ ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + dx - dy + ( rx * rx ) ; } } }
midptellipse ( 10 , 15 , 50 , 50 ) ;
let matrix = new Array ( 5 ) ; for ( let i = 0 ; i < 5 ; i ++ ) { matrix [ i ] = new Array ( 5 ) ; for ( let j = 0 ; j < 5 ; j ++ ) { matrix [ i ] [ j ] = 0 ; } } let row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
document . write ( " " ) ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { document . write ( matrix [ row_index ] [ column_index ] + " " ) ; } document . write ( " " ) ; }
document . write ( " " ) ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) == size - 1 ) document . write ( matrix [ row_index ] [ column_index ] + " " ) ; } }
let matrix = new Array ( 5 ) ; for ( let i = 0 ; i < 5 ; i ++ ) { matrix [ i ] = new Array ( 5 ) ; } let row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
document . write ( " " ) ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { document . write ( " " , matrix [ row_index ] [ column_index ] ) ; } document . write ( " " ) ; }
document . write ( " " + " " ) ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) < size - 1 ) document . write ( matrix [ row_index ] [ column_index ] + " " ) ; } }
function distance ( x1 , y1 , z1 , x2 , y2 , z2 ) { var d = Math . pow ( ( Math . pow ( x2 - x1 , 2 ) + Math . pow ( y2 - y1 , 2 ) + Math . pow ( z2 - z1 , 2 ) * 1.0 ) , 0.5 ) ; document . write ( " " + d . toFixed ( 10 ) ) ; return ; }
var x1 = 2 ; var y1 = - 5 ; var z1 = 7 ; var x2 = 3 ; var y2 = 4 ; var z2 = 5 ;
distance ( x1 , y1 , z1 , x2 , y2 , z2 ) ;
function No_Of_Pairs ( N ) { let i = 1 ;
while ( ( i * i * i ) + ( 2 * i * i ) + i <= N ) i ++ ; return ( i - 1 ) ; }
function print_pairs ( pairs ) { let i = 1 , mul ; for ( i = 1 ; i <= pairs ; i ++ ) { mul = i * ( i + 1 ) ; document . write ( " " + i + " " + ( mul * i ) + " " + mul * ( i + 1 ) + " " ) ; } }
let N = 500 , pairs , mul , i = 1 ; pairs = No_Of_Pairs ( N ) ; document . write ( " " + pairs + " " ) ; print_pairs ( pairs ) ;
function findArea ( d ) { return ( d * d ) / 2 ; }
let d = 10 ; document . write ( findArea ( d ) ) ;
function AvgofSquareN ( n ) { let sum = 0 ; for ( let i = 1 ; i <= n ; i ++ ) sum += ( i * i ) ; return sum / n ; }
let n = 2 ; document . write ( AvgofSquareN ( n ) . toFixed ( 6 ) ) ;
function Series ( x , n ) { let sum = 1 , term = 1 , fct , j , y = 2 , m ;
let i ; for ( i = 1 ; i < n ; i ++ ) { fct = 1 ; for ( j = 1 ; j <= y ; j ++ ) { fct = fct * j ; } term = term * ( - 1 ) ; m = term * Math . pow ( x , y ) / fct ; sum = sum + m ; y += 2 ; } return sum ; }
let x = 9 ; let n = 10 ; document . write ( Series ( x , n ) . toFixed ( 4 ) ) ;
function sum ( x , n ) { let total = 1.0 ; let multi = x ; for ( let i = 1 ; i <= n ; i ++ ) { total = total + multi / i ; multi = multi * x ; } return total ; }
let x = 2 ; let n = 5 ; document . write ( sum ( x , n ) . toFixed ( 2 ) ) ;
function chiliagonNum ( n ) { return ( 998 * n * n - 996 * n ) / 2 ; }
let n = 3 ; document . write ( " " + chiliagonNum ( n ) ) ;
function pentacontagonNum ( n ) { return ( 48 * n * n - 46 * n ) / 2 ; }
let n = 3 ; document . write ( " " + pentacontagonNum ( n ) ) ;
function lastElement ( arr ) {
let pq = [ ] ; for ( let i = 0 ; i < arr . length ; i ++ ) { pq . push ( arr [ i ] ) ; }
let m1 , m2 ;
while ( pq . length ) {
if ( pq . length == 1 )
return pq [ pq . length - 1 ] ; pq . sort ( ( a , b ) => a - b ) m1 = pq [ pq . length - 1 ] ; pq . pop ( ) ; m2 = pq [ pq . length - 1 ] ; pq . pop ( ) ;
if ( m1 != m2 ) pq . push ( m1 - m2 ) ; }
return 0 ; }
let arr = [ 2 , 7 , 4 , 1 , 8 , 1 , 1 ] ; document . write ( lastElement ( arr ) + " " ) ;
function countDigit ( n ) { return Math . floor ( Math . log10 ( n ) + 1 ) ; }
var n = 80 ; document . write ( countDigit ( n ) ) ;
function sum ( x , n ) { let i , total = 1.0 , multi = x ;
document . write ( total + " " ) ;
for ( i = 1 ; i < n ; i ++ ) { total = total + multi ; document . write ( multi + " " ) ; multi = multi * x ; } document . write ( " " ) ; return total ; }
let x = 2 ; let n = 5 ; document . write ( sum ( x , n ) . toFixed ( 2 ) ) ;
function findRemainder ( n ) {
let x = n & 3 ;
return x ; }
let N = 43 ; let ans = findRemainder ( N ) ; document . write ( ans ) ;
function triangular_series ( n ) { let i , j = 1 , k = 1 ;
for ( i = 1 ; i <= n ; i ++ ) { document . write ( k + " " ) ;
j = j + 1 ;
k = k + j ; } }
let n = 5 ; triangular_series ( n ) ;
function countDigit ( n ) { if ( n / 10 == 0 ) return 1 ; return 1 + countDigit ( parseInt ( n / 10 ) ) ; }
var n = 345289467 ; document . write ( " " + countDigit ( n ) ) ;
var x = 1234 ;
if ( x % 9 == 1 ) document . write ( " " ) ; else document . write ( " " ) ;
var MAX = 100 ;
var arr = new Array ( MAX ) ; arr [ 0 ] = 0 ; arr [ 1 ] = 1 ; for ( var i = 2 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] + arr [ i - 2 ] ; document . write ( " " ) ; for ( var i = 1 ; i < MAX ; i ++ ) if ( arr [ i ] % i == 0 ) document . write ( i + " " ) ;
function findMaxValue ( ) { let res = 2 ; let fact = 2 ; while ( true ) {
if ( fact < 0 fact > 9223372036854775807 ) break ; res ++ ; fact = fact * res ; } return res - 1 ; }
document . write ( " " + " " + findMaxValue ( ) ) ;
function firstkdigits ( n , k ) {
let product = n * Math . log10 ( n ) ;
let decimal_part = product - Math . floor ( product ) ;
decimal_part = Math . pow ( 10 , decimal_part ) ;
let digits = Math . pow ( 10 , k - 1 ) , i = 0 ; return ( Math . floor ( decimal_part * digits ) ) ; }
let n = 1450 ; let k = 6 ; document . write ( firstkdigits ( n , k ) ) ;
function moduloMultiplication ( a , b , mod ) {
a = ( a % mod ) ; while ( b > 0 ) {
if ( ( b & 1 ) > 0 ) { res = ( res + a ) % mod ; }
a = ( 2 * a ) % mod ;
} return res ; }
let a = 426 ; let b = 964 ; let m = 235 ; document . write ( moduloMultiplication ( a , b , m ) ) ;
function findRoots ( a , b , c ) {
if ( a == 0 ) { document . write ( " " ) ; return ; } let d = b * b - 4 * a * c ; let sqrt_val = Math . sqrt ( Math . abs ( d ) ) ; if ( d > 0 ) { document . write ( " " + " " ) ; document . write ( ( - b + sqrt_val ) / ( 2 * a ) + " " + ( - b - sqrt_val ) / ( 2 * a ) ) ; } else if ( d == 0 ) { document . write ( " " + " " ) ; document . write ( - b / ( 2 * a ) + " " + - b / ( 2 * a ) ) ; }
{ document . write ( " " ) ; document . write ( - b / ( 2 * a ) + " " + sqrt_val + " " + - b / ( 2 * a ) + " " + sqrt_val ) ; } }
let a = 1 , b = - 7 , c = 12 ;
findRoots ( a , b , c ) ;
function val ( c ) { if ( c >= ' ' . charCodeAt ( ) && c <= ' ' . charCodeAt ( ) ) return ( c - ' ' . charCodeAt ( ) ) ; else return ( c - ' ' . charCodeAt ( ) + 10 ) ; }
function toDeci ( str , b_ase ) { let len = str . length ;
let power = 1 ;
let num = 0 ; let i ;
for ( i = len - 1 ; i >= 0 ; i -- ) {
if ( val ( str [ i ] . charCodeAt ( ) ) >= b_ase ) { document . write ( " " ) ; return - 1 ; } num += val ( str [ i ] . charCodeAt ( ) ) * power ; power = power * b_ase ; } return num ; }
let str = " " ; let b_ase = 16 ; document . write ( " " + str + " " + b_ase + " " + toDeci ( str , b_ase ) ) ;
function seriesSum ( calculated , current , N ) { let i , cur = 1 ;
if ( current == N + 1 ) return 0 ;
for ( i = calculated ; i < calculated + current ; i ++ ) cur *= i ;
return cur + seriesSum ( i , current + 1 , N ) ; }
let N = 5 ;
document . write ( seriesSum ( 1 , 1 , N ) ) ;
let N = 30 ;
let fib = new Array ( N ) ;
function largestFiboLessOrEqual ( n ) {
fib [ 0 ] = 1 ;
fib [ 1 ] = 2 ;
let i ; for ( i = 2 ; fib [ i - 1 ] <= n ; i ++ ) { fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ; }
return ( i - 2 ) ; }
function fibonacciEncoding ( n ) { let index = largestFiboLessOrEqual ( n ) ;
let codeword = new Array ( index + 3 ) ;
let i = index ; while ( n > 0 ) {
codeword [ i ] = ' ' ;
n = n - fib [ i ] ;
i = i - 1 ;
while ( i >= 0 && fib [ i ] > n ) { codeword [ i ] = ' ' ; i = i - 1 ; } }
codeword [ index + 1 ] = ' ' ; codeword [ index + 2 ] = ' \0 ' ; let string = ( codeword ) . join ( " " ) ;
return string ; }
let n = 143 ; document . write ( " " + n + " " + fibonacciEncoding ( n ) ) ;
function countSquares ( m , n ) {
if ( n < m ) { var temp = m ; m = n ; n = temp ; }
return n * ( n + 1 ) * ( 3 * m - n + 1 ) / 6 ; }
var m = 4 ; var n = 3 ; document . write ( " " + countSquares ( m , n ) ) ;
function simpleSieve ( limit ) {
var mark = Array ( limit ) . fill ( true ) ;
for ( p = 2 ; p * p < limit ; p ++ ) {
if ( mark [ p ] == true ) {
for ( i = p * p ; i < limit ; i += p ) mark [ i ] = false ; } }
for ( p = 2 ; p < limit ; p ++ ) if ( mark [ p ] == true ) document . write ( p + " " ) ; }
function modInverse ( a , m ) { let m0 = m ; let y = 0 ; let x = 1 ; if ( m == 1 ) return 0 ; while ( a > 1 ) {
let q = parseInt ( a / m ) ; let t = m ;
m = a % m ; a = t ; t = y ;
y = x - q * y ; x = t ; }
if ( x < 0 ) x += m0 ; return x ; }
let a = 3 ; let m = 11 ;
document . write ( ` ${ modInverse ( a , m ) } ` ) ;
function gcd ( a , b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
function phi ( n ) { let result = 1 ; for ( let i = 2 ; i < n ; i ++ ) if ( gcd ( i , n ) == 1 ) result ++ ; return result ; }
for ( let n = 1 ; n <= 10 ; n ++ ) document . write ( ` ${ n } ${ phi ( n ) } ` ) ;
function phi ( n ) {
for ( let p = 2 ; p * p <= n ; ++ p ) {
if ( n % p == 0 ) {
while ( n % p == 0 ) n /= p ; result *= ( 1.0 - ( 1.0 / p ) ) ; } }
if ( n > 1 ) result *= ( 1.0 - ( 1.0 / n ) ) ; return parseInt ( result ) ; }
for ( let n = 1 ; n <= 10 ; n ++ ) document . write ( ` ${ n } ${ phi ( n ) } ` ) ;
function printFibonacciNumbers ( n ) { let f1 = 0 , f2 = 1 , i ; if ( n < 1 ) return ; document . write ( f1 + " " ) ; for ( i = 1 ; i < n ; i ++ ) { document . write ( f2 + " " ) ; let next = f1 + f2 ; f1 = f2 ; f2 = next ; } }
printFibonacciNumbers ( 7 ) ;
function gcd ( a , b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
function lcm ( a , b ) { return ( a / gcd ( a , b ) ) * b ; }
let a = 15 , b = 20 ; document . write ( " " + a + " " + b + " " + lcm ( a , b ) ) ;
MAX = 11 ; function isMultipleof5 ( n ) { str = Array ( n ) . fill ( ' ' ) ; var len = str . length ;
if ( str [ len - 1 ] == ' ' str [ len - 1 ] == ' ' ) return true ; return false ; }
var n = 19 ; if ( isMultipleof5 ( n ) == true ) document . write ( n + " " + " " ) ; else document . write ( n + " " + " " ) ;
function toggleBit ( n , k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
var n = 5 , k = 2 ; document . write ( toggleBit ( n , k ) ) ;
function clearBit ( n , k ) { return ( n & ( ~ ( 1 << ( k - 1 ) ) ) ) ; }
var n = 5 , k = 1 ; document . write ( clearBit ( n , k ) ) ;
function add ( x , y ) { let keep = ( x & y ) << 1 ; let res = x ^ y ;
if ( keep == 0 ) return res ; return add ( keep , res ) ; }
document . write ( add ( 15 , 38 ) ) ;
function countBits ( number ) {
return Math . floor ( Math . log2 ( number ) + 1 ) ; }
let num = 65 ; document . write ( countBits ( num ) ) ;
var INT_SIZE = 32 ;
function constructNthNumber ( group_no , aux_num , op ) { var a = Array . from ( { length : INT_SIZE } , ( _ , i ) => 0 ) ; var num = 0 , len_f = 0 ; var i = 0 ;
if ( op == 2 ) {
len_f = 2 * group_no ;
a [ len_f - 1 ] = a [ 0 ] = 1 ;
while ( aux_num > 0 ) {
a [ group_no + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
else if ( op == 0 ) {
len_f = 2 * group_no + 1 ;
a [ len_f - 1 ] = a [ 0 ] = 1 ; a [ group_no ] = 0 ;
while ( aux_num > 0 ) {
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
else {
len_f = 2 * group_no + 1 ;
a [ len_f - 1 ] = a [ 0 ] = 1 ; a [ group_no ] = 1 ;
while ( aux_num > 0 ) {
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
for ( i = 0 ; i < len_f ; i ++ ) num += ( 1 << i ) * a [ i ] ; return num ; }
function getNthNumber ( n ) { var group_no = 0 , group_offset ; var count_upto_group = 0 , count_temp = 1 ; var op , aux_num ;
while ( count_temp < n ) { group_no ++ ;
count_upto_group = count_temp ; count_temp += 3 * ( 1 << ( group_no - 1 ) ) ; }
group_offset = n - count_upto_group - 1 ;
if ( ( group_offset + 1 ) <= ( 1 << ( group_no - 1 ) ) ) {
aux_num = group_offset ; } else { if ( ( ( group_offset + 1 ) - ( 1 << ( group_no - 1 ) ) ) % 2 == 1 )
op = 0 ; else
op = 1 ; aux_num = ( ( group_offset ) - ( 1 << ( group_no - 1 ) ) ) / 2 ; } return constructNthNumber ( group_no , aux_num , op ) ; }
var n = 9 ;
document . write ( getNthNumber ( n ) ) ;
function toggleAllExceptK ( n , k ) {
return ~ ( n ^ ( 1 << k ) ) ; }
let n = 4294967295 ; let k = 0 ; document . write ( toggleAllExceptK ( n , k ) ) ;
function swapBits ( n , p1 , p2 ) {
var bit1 = ( n >> p1 ) & 1 ;
var bit2 = ( n >> p2 ) & 1 ;
var x = ( bit1 ^ bit2 ) ;
x = ( x << p1 ) | ( x << p2 ) ;
var result = n ^ x ; return result ; }
var res = swapBits ( 28 , 0 , 3 ) ; document . write ( " " + res ) ;
function firstNonRepeating ( str ) { let NO_OF_CHARS = 256 ;
let arr = new Array ( NO_OF_CHARS ) ; for ( let i = 0 ; i < NO_OF_CHARS ; i ++ ) arr [ i ] = - 1 ;
for ( let i = 0 ; i < str . length ; i ++ ) { if ( arr [ str [ i ] . charCodeAt ( 0 ) ] == - 1 ) arr [ str [ i ] . charCodeAt ( 0 ) ] = i ; else arr [ str [ i ] . charCodeAt ( 0 ) ] = - 2 ; } let res = Number . MAX_VALUE ; for ( let i = 0 ; i < NO_OF_CHARS ; i ++ )
if ( arr [ i ] >= 0 ) res = Math . min ( res , arr [ i ] ) ; return res ; }
let str = " " ; let index = firstNonRepeating ( str ) ; if ( index == Number . MAX_VALUE ) document . write ( " " + " " ) ; else document . write ( " " + " " + str [ index ] ) ;
function triacontagonalNum ( n ) { return ( 28 * n * n - 26 * n ) / 2 ; }
var n = 3 ; document . write ( " " + triacontagonalNum ( n ) ) ;
function hexacontagonNum ( n ) { return ( 58 * n * n - 56 * n ) / 2 ; }
var n = 3 ; document . write ( " " + hexacontagonNum ( n ) ) ;
function enneacontagonNum ( n ) { return ( 88 * n * n - 86 * n ) / 2 ; }
var n = 3 ; document . write ( " " + enneacontagonNum ( n ) ) ;
function triacontakaidigonNum ( n ) { return ( 30 * n * n - 28 * n ) / 2 ; }
let n = 3 ; document . write ( " " + triacontakaidigonNum ( n ) ) ;
function IcosihexagonalNum ( n ) { return ( 24 * n * n - 22 * n ) / 2 ; }
let n = 3 ; document . write ( " " + IcosihexagonalNum ( n ) ) ;
function icosikaioctagonalNum ( n ) { return ( 26 * n * n - 24 * n ) / 2 ; }
var n = 3 ; document . write ( " " + icosikaioctagonalNum ( n ) ) ;
function octacontagonNum ( n ) { return ( 78 * n * n - 76 * n ) / 2 ; }
var n = 3 ; document . write ( " " + octacontagonNum ( n ) ) ;
function hectagonNum ( n ) { return ( 98 * n * n - 96 * n ) / 2 ; }
var n = 3 ; document . write ( " " + hectagonNum ( n ) ) ;
function tetracontagonNum ( n ) { return ( 38 * n * n - 36 * n ) / 2 ; }
let n = 3 ; document . write ( " " + tetracontagonNum ( n ) ) ;
function binarySearch ( arr , N , X ) {
let start = 0 ;
let end = N ; while ( start <= end ) {
let mid = Math . floor ( start + ( end - start ) / 2 ) ;
if ( X == arr [ mid ] ) {
return mid ; }
else if ( X < arr [ mid ] ) {
start = mid + 1 ; } else {
end = mid - 1 ; } }
return - 1 ; }
let arr = [ 5 , 4 , 3 , 2 , 1 ] ; let N = arr . length ; let X = 5 ; document . write ( binarySearch ( arr , N , X ) ) ;
function flip ( arr , i ) { let temp , start = 0 ; while ( start < i ) { temp = arr [ start ] ; arr [ start ] = arr [ i ] ; arr [ i ] = temp ; start ++ ; i -- ; } }
function findMax ( arr , n ) { let mi , i ; for ( mi = 0 , i = 0 ; i < n ; ++ i ) if ( arr [ i ] > arr [ mi ] ) mi = i ; return mi ; }
function pancakeSort ( arr , n ) {
for ( let curr_size = n ; curr_size > 1 ; -- curr_size ) {
let mi = findMax ( arr , curr_size ) ;
if ( mi != curr_size - 1 ) {
flip ( arr , mi ) ;
flip ( arr , curr_size - 1 ) ; } } return 0 ; }
function printArray ( arr , arr_size ) { for ( let i = 0 ; i < arr_size ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; }
let arr = [ 23 , 10 , 20 , 11 , 12 , 6 , 7 ] ; let n = arr . length ; pancakeSort ( arr , n ) ; document . write ( " " + " " ) ; printArray ( arr , n ) ;
