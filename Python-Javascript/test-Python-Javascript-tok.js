function minSum ( A , N ) {
let mp = new Map ( ) ; let sum = 0 ;
for ( let i = 0 ; i < N ; i ++ ) {
sum += A [ i ] ;
mp [ A [ i ] ] ++ ; if ( mp . has ( A [ i ] ) ) { mp . set ( A [ i ] , mp . get ( A [ i ] ) + 1 ) } else { mp . set ( A [ i ] , 1 ) } }
let minSum = Number . MAX_SAFE_INTEGER ;
for ( let it of mp ) {
minSum = Math . min ( minSum , sum - ( it [ 0 ] * it [ 1 ] ) ) ; }
return minSum ; }
let arr = [ 4 , 5 , 6 , 6 ] ;
let N = arr . length document . write ( minSum ( arr , N ) + " " ) ;
function maxAdjacent ( arr , N ) { var res = [ ] ;
for ( var i = 1 ; i < N - 1 ; i ++ ) { var prev = arr [ 0 ] ;
var maxi = Number . MIN_VALUE ;
for ( var j = 1 ; j < N ; j ++ ) {
if ( i == j ) continue ;
maxi = Math . max ( maxi , Math . abs ( arr [ j ] - prev ) ) ;
prev = arr [ j ] ; }
res . push ( maxi ) ; }
for ( var j = 0 ; j < res . length ; j ++ ) document . write ( res [ j ] + " " ) ; document . write ( " " ) ; }
var arr = [ 1 , 3 , 4 , 7 , 8 ] ; var N = arr . length ; maxAdjacent ( arr , N ) ;
function findSize ( N ) {
if ( N == 0 ) return 1 ; if ( N == 1 ) return 1 ; let Size = 2 * findSize ( parseInt ( N / 2 , 10 ) ) + 1 ;
return Size ; }
function CountOnes ( N , L , R ) { if ( L > R ) { return 0 ; }
if ( N <= 1 ) { return N ; } let ret = 0 ; let M = parseInt ( N / 2 , 10 ) ; let Siz_M = findSize ( M ) ;
if ( L <= Siz_M ) {
ret += CountOnes ( parseInt ( N / 2 , 10 ) , L , Math . min ( Siz_M , R ) ) ; }
if ( L <= Siz_M + 1 && Siz_M + 1 <= R ) { ret += N % 2 ; }
if ( Siz_M + 1 < R ) { ret += CountOnes ( parseInt ( N / 2 , 10 ) , Math . max ( 1 , L - Siz_M - 1 ) , R - Siz_M - 1 ) ; } return ret ; }
let N = 7 , L = 2 , R = 5 ;
document . write ( CountOnes ( N , L , R ) ) ;
function prime ( n ) {
if ( n == 1 ) return false ;
for ( i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; }
return true ; }
function minDivisior ( n ) {
if ( prime ( n ) ) { document . write ( 1 + " " + ( n - 1 ) ) ; }
else { for ( i = 2 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
document . write ( n / i + " " + ( n / i * ( i - 1 ) ) ) ; break ; } } } }
var N = 4 ;
minDivisior ( N ) ;
var Landau = - 1000000000 ;
function gcd ( a , b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
function lcm ( a , b ) { return ( a * b ) / gcd ( a , b ) ; }
function findLCM ( arr ) { var nth_lcm = arr [ 0 ] ; for ( var i = 1 ; i < arr . length ; i ++ ) nth_lcm = lcm ( nth_lcm , arr [ i ] ) ;
Landau = Math . max ( Landau , nth_lcm ) ; }
function findWays ( arr , i , n ) {
if ( n == 0 ) findLCM ( arr ) ;
for ( var j = i ; j <= n ; j ++ ) {
arr . push ( j ) ;
findWays ( arr , j , n - j ) ;
arr . pop ( ) ; } }
function Landau_function ( n ) { arr = [ ] ;
findWays ( arr , 1 , n ) ;
document . write ( Landau ) ; }
var N = 4 ;
Landau_function ( N ) ;
function isPrime ( n ) {
if ( n == 1 ) return true ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ;
for ( let i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
function checkExpression ( n ) { if ( isPrime ( n ) ) document . write ( " " ) ; else document . write ( " " ) ; }
let N = 3 ; checkExpression ( N ) ;
function checkArray ( n , k , arr ) {
var cnt = 0 ; for ( i = 0 ; i < n ; i ++ ) {
if ( ( arr [ i ] & 1 ) != 0 ) cnt += 1 ; }
if ( cnt >= k && cnt % 2 == k % 2 ) return true ; else return false ; }
var arr = [ 1 , 3 , 4 , 7 , 5 , 3 , 1 ] ; var n = arr . length ; var k = 4 ; if ( checkArray ( n , k , arr ) ) document . write ( " " ) ; else document . write ( " " ) ;
function func ( arr , n ) { let ans = 0 ; let maxx = 0 ; let freq = Array . from ( { length : 100005 } , ( _ , i ) => 0 ) ; let temp ;
for ( let i = 0 ; i < n ; i ++ ) { temp = arr [ i ] ; freq [ temp ] ++ ; maxx = Math . max ( maxx , temp ) ; }
for ( let i = 1 ; i <= maxx ; i ++ ) { freq [ i ] += freq [ i - 1 ] ; } for ( let i = 1 ; i <= maxx ; i ++ ) { if ( freq [ i ] != 0 ) { let j ;
let cur = Math . ceil ( 0.5 * i ) - 1.0 ; for ( j = 1.5 ; ; j ++ ) { let val = Math . min ( maxx , ( Math . ceil ( i * j ) - 1.0 ) ) ; let times = ( freq [ i ] - freq [ i - 1 ] ) , con = ( j - 0.5 ) ;
ans += times * con * ( freq [ val ] - freq [ cur ] ) ; cur = val ; if ( val == maxx ) break ; } } }
return ans ; }
let arr = [ 1 , 2 , 3 ] ; let n = arr . length ; document . write ( func ( arr , n ) ) ;
function insert_element ( a , n ) {
let Xor = 0 ;
let Sum = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { Xor ^= a [ i ] ; Sum += a [ i ] ; }
if ( Sum == 2 * Xor ) {
document . write ( " " + " " ) ; return ; }
if ( Xor == 0 ) { document . write ( " " + " " ) ; document . write ( Sum + " " ) ; return ; }
let num1 = Sum + Xor ; let num2 = Xor ;
document . write ( " " + " " ) ;
document . write ( num1 + " " + num2 + " " ) ; }
let a = [ 1 , 2 , 3 ] ; let n = a . length ; insert_element ( a , n ) ;
function checkSolution ( a , b , c ) { if ( a == c ) document . write ( " " ) ; else document . write ( " " ) ; }
let a = 2 , b = 0 , c = 2 ; checkSolution ( a , b , c ) ;
function isPerfectSquare ( x ) {
var sr = Math . sqrt ( x ) ;
return ( ( sr - Math . floor ( sr ) ) == 0 ) ; }
function checkSunnyNumber ( N ) {
if ( isPerfectSquare ( N + 1 ) ) { document . write ( " " ) ; }
else { document . write ( " " ) ; } }
var N = 8 ;
checkSunnyNumber ( N ) ;
function countValues ( n ) { let answer = 0 ;
for ( let i = 2 ; i <= n ; i ++ ) { let k = n ;
while ( k >= i ) { if ( k % i == 0 ) k /= i ; else k -= i ; }
if ( k == 1 ) answer ++ ; } return answer ; }
let N = 6 ; document . write ( countValues ( N ) ) ;
function printKNumbers ( N , K ) {
for ( let i = 0 ; i < K - 1 ; i ++ ) document . write ( 1 + " " ) ;
document . write ( N - K + 1 ) ; }
let N = 10 , K = 3 ; printKNumbers ( N , K ) ;
function NthSmallest ( K ) {
var Q = [ ] ; var x ;
for ( var i = 1 ; i < 10 ; i ++ ) Q . push ( i ) ;
for ( var i = 1 ; i <= K ; i ++ ) {
x = Q [ 0 ] ;
Q . shift ( ) ;
if ( x % 10 != 0 ) {
Q . push ( x * 10 + x % 10 - 1 ) ; }
Q . push ( x * 10 + x % 10 ) ;
if ( x % 10 != 9 ) {
Q . push ( x * 10 + x % 10 + 1 ) ; } }
return x ; }
var N = 16 ; document . write ( NthSmallest ( N ) ) ;
function nearest ( n ) {
var prevSquare = parseInt ( Math . sqrt ( n ) ) ; var nextSquare = prevSquare + 1 ; prevSquare = prevSquare * prevSquare ; nextSquare = nextSquare * nextSquare ;
if ( ( n - prevSquare ) < ( nextSquare - n ) ) { ans = parseInt ( ( prevSquare - n ) ) ; } else ans = parseInt ( ( nextSquare - n ) ) ;
return ans ; }
var n = 14 ; document . write ( nearest ( n ) + " " ) ; n = 16 ; document . write ( nearest ( n ) + " " ) ; n = 18 ; document . write ( nearest ( n ) + " " ) ;
function printValueOfPi ( N ) {
let pi = 2 * Math . acos ( 0.0 ) ;
document . write ( pi . toFixed ( 4 ) ) ; }
let N = 4 ;
printValueOfPi ( N ) ;
function decBinary ( arr , n ) { let k = parseInt ( Math . log2 ( n ) , 10 ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n = parseInt ( n / 2 , 10 ) ; } }
function binaryDec ( arr , n ) { let ans = 0 ; for ( let i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
function getNum ( n , k ) {
let l = parseInt ( Math . log2 ( n ) , 10 ) + 1 ;
let a = new Array ( l ) ; a . fill ( 0 ) ; decBinary ( a , n ) ;
if ( k > l ) return n ;
a [ k - 1 ] = ( a [ k - 1 ] == 0 ) ? 1 : 0 ;
return binaryDec ( a , l ) ; }
let n = 56 , k = 2 ; document . write ( getNum ( n , k ) ) ;
let MAX = 1000000 ; let MOD = 10000007 ;
let result = new Array ( MAX + 1 ) ; result . fill ( 0 ) ; let fact = new Array ( MAX + 1 ) ; fact . fill ( 0 ) ;
function preCompute ( ) {
fact [ 0 ] = 1 ; result [ 0 ] = 1 ;
for ( let i = 1 ; i <= MAX ; i ++ ) {
fact [ i ] = ( ( fact [ i - 1 ] % MOD ) * i ) % MOD ;
result [ i ] = ( ( result [ i - 1 ] % MOD ) * ( fact [ i ] % MOD ) ) % MOD ; } }
function performQueries ( q , n ) {
preCompute ( ) ;
for ( let i = 0 ; i < n ; i ++ ) document . write ( result [ q [ i ] ] + " " ) ; }
let q = [ 4 , 5 ] ; let n = q . length ; performQueries ( q , n ) ;
function gcd ( a , b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
function divTermCount ( a , b , c , num ) {
return parseInt ( ( ( num / a ) + ( num / b ) + ( num / c ) - ( num / ( ( a * b ) / gcd ( a , b ) ) ) - ( num / ( ( c * b ) / gcd ( c , b ) ) ) - ( num / ( ( a * c ) / gcd ( a , c ) ) ) + ( num / ( ( ( ( a * b ) / gcd ( a , b ) ) * c ) / gcd ( ( ( a * b ) / gcd ( a , b ) ) , c ) ) ) ) ) ; }
function findNthTerm ( a , b , c , n ) {
var low = 1 , high = Number . MAX_SAFE_INTEGER , mid ; while ( low < high ) { mid = low + ( high - low ) / 2 ;
if ( divTermCount ( a , b , c , mid ) < n ) low = mid + 1 ;
else high = mid ; } return low ; }
var a = 2 , b = 3 , c = 5 , n = 100 ; document . write ( parseInt ( findNthTerm ( a , b , c , n ) ) ) ;
function calculate_angle ( n , i , j , k ) {
var x , y ;
if ( i < j ) x = j - i ; else x = j + n - i ; if ( j < k ) y = k - j ; else y = k + n - j ;
var ang1 = ( 180 * x ) / n ; var ang2 = ( 180 * y ) / n ;
var ans = 180 - ang1 - ang2 ; return ans ; }
var n = 5 ; var a1 = 1 ; var a2 = 2 ; var a3 = 5 ; document . write ( parseInt ( calculate_angle ( n , a1 , a2 , a3 ) ) ) ;
function Loss ( SP , P ) { var loss = 0 ; loss = ( 2 * P * P * SP ) / ( 100 * 100 - P * P ) ; document . write ( " " + loss . toFixed ( 3 ) ) ; }
var SP = 2400 , P = 30 ;
Loss ( SP , P ) ;
let MAXN = 1000001 ;
let spf = new Array ( MAXN ) ;
let hash1 = new Array ( MAXN ) ;
function sieve ( ) { spf [ 1 ] = 1 ; for ( let i = 2 ; i < MAXN ; i ++ )
for ( let i = 4 ; i < MAXN ; i += 2 ) spf [ i ] = 2 ;
for ( let i = 3 ; i * i < MAXN ; i ++ ) {
if ( spf [ i ] == i ) { for ( let j = i * i ; j < MAXN ; j += i )
if ( spf [ j ] == j ) spf [ j ] = i ; } } }
function getFactorization ( x ) { let temp ; while ( x != 1 ) { temp = spf [ x ] ; if ( x % temp == 0 ) {
hash1 [ spf [ x ] ] ++ ; x = x / spf [ x ] ; } while ( x % temp == 0 ) x = x / temp ; } }
function check ( x ) { let temp ; while ( x != 1 ) { temp = spf [ x ] ;
if ( x % temp == 0 && hash1 [ temp ] > 1 ) return false ; while ( x % temp == 0 ) x = x / temp ; } return true ; }
function hasValidNum ( arr , n ) {
sieve ( ) ; for ( let i = 0 ; i < n ; i ++ ) getFactorization ( arr [ i ] ) ;
for ( let i = 0 ; i < n ; i ++ ) if ( check ( arr [ i ] ) ) return true ; return false ; }
let arr = [ 2 , 8 , 4 , 10 , 6 , 7 ] ; let n = arr . length ; if ( hasValidNum ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function countWays ( N ) {
let E = parseInt ( ( N * ( N - 1 ) ) / 2 , 10 ) ; if ( N == 1 ) return 0 ; return Math . pow ( 2 , E - 1 ) ; }
let N = 4 ; document . write ( countWays ( N ) ) ;
let l = new Array ( 1001 ) . fill ( 0 ) . map ( ( ) => new Array ( 1001 ) . fill ( 0 ) ) ; function initialize ( ) {
l [ 0 ] [ 0 ] = 1 ; for ( let i = 1 ; i < 1001 ; i ++ ) {
l [ i ] [ 0 ] = 1 ; for ( let j = 1 ; j < i + 1 ; j ++ ) {
l [ i ] [ j ] = ( l [ i - 1 ] [ j - 1 ] + l [ i - 1 ] [ j ] ) ; } } }
function nCr ( n , r ) {
return l [ n ] [ r ] ; }
initialize ( ) ; let n = 8 ; let r = 3 ; document . write ( nCr ( n , r ) ) ;
function minAbsDiff ( n ) { let mod = n % 4 ; if ( mod == 0 mod == 3 ) { return 0 ; } return 1 ; }
let n = 5 ; document . write ( minAbsDiff ( n ) ) ;
function check ( s ) {
let freq = new Array ( 10 ) . fill ( 0 ) , r ; while ( s != 0 ) {
r = s % 10 ;
s = parseInt ( s / 10 ) ;
freq [ r ] += 1 ; } let xor__ = 0 ;
for ( let i = 0 ; i < 10 ; i ++ ) { xor__ = xor__ ^ freq [ i ] ; if ( xor__ == 0 ) return true ; else return false ; } }
let s = 122233 ; if ( check ( s ) ) document . write ( " " ) ; else document . write ( " " ) ;
function printLines ( n , k ) {
for ( i = 0 ; i < n ; i ++ ) { document . write ( k * ( 6 * i + 1 ) + " " + k * ( 6 * i + 2 ) + " " + k * ( 6 * i + 3 ) + " " + k * ( 6 * i + 5 ) + " " ) ; } }
var n = 2 , k = 2 ; printLines ( n , k ) ;
function calculateSum ( n ) {
return ( Math . pow ( 2 , n + 1 ) + n - 2 ) ; }
let n = 4 ;
document . write ( " " + calculateSum ( n ) ) ;
var mod = 1000000007 ;
function count_special ( n ) {
var fib = [ ... Array ( n + 1 ) ] ;
fib [ 0 ] = 1 ;
fib [ 1 ] = 2 ; for ( var i = 2 ; i <= n ; i ++ ) {
fib [ i ] = ( ( fib [ i - 1 ] % mod ) + ( fib [ i - 2 ] % mod ) ) % mod ; }
return fib [ n ] ; }
var n = 3 ; document . write ( count_special ( n ) + " " ) ;
let mod = 1000000000 ;
function ways ( i , arr , n ) {
if ( i == n - 1 ) return 1 ; let sum = 0 ;
for ( let j = 1 ; j + i < n && j <= arr [ i ] ; j ++ ) { sum += ( ways ( i + j , arr , n ) ) % mod ; sum %= mod ; } return sum % mod ; }
let arr = [ 5 , 3 , 1 , 4 , 3 ] ; let n = arr . length ; document . write ( ways ( 0 , arr , n ) ) ;
let mod = ( 1e9 + 7 ) ;
function ways ( arr , n ) {
let dp = new Array ( n + 1 ) ; dp . fill ( 0 ) ;
dp [ n - 1 ] = 1 ;
for ( let i = n - 2 ; i >= 0 ; i -- ) { dp [ i ] = 0 ;
for ( let j = 1 ; ( ( j + i ) < n && j <= arr [ i ] ) ; j ++ ) { dp [ i ] += dp [ i + j ] ; dp [ i ] %= mod ; } }
return dp [ 0 ] % mod ; }
let arr = [ 5 , 3 , 1 , 4 , 3 ] ; let n = arr . length ; document . write ( ways ( arr , n ) % mod ) ;
function countSum ( arr , n ) { var result = 0 ;
count_odd = 0 ; count_even = 0 ;
for ( var i = 1 ; i <= n ; i ++ ) {
if ( arr [ i - 1 ] % 2 == 0 ) { count_even = count_even + count_even + 1 ; count_odd = count_odd + count_odd ; }
else { var temp = count_even ; count_even = count_even + count_odd ; count_odd = count_odd + temp + 1 ; } } return new pair ( count_even , count_odd ) ; }
var arr = [ 1 , 2 , 2 , 3 ] ; var n = arr . length ;
var ans = countSum ( arr , n ) ; document . write ( " " + ans . first ) ; document . write ( " " + ans . second ) ;
let MAX = 10 ;
function numToVec ( N ) { let digit = [ ] ;
while ( N != 0 ) { digit . push ( N % 10 ) ; N = Math . floor ( N / 10 ) ; }
if ( digit . length == 0 ) digit . push ( 0 ) ;
digit . reverse ( ) ;
return digit ; }
function solve ( A , B , C ) { let digit = [ ] ; let d , d2 ;
digit = numToVec ( C ) ; d = A . length ;
if ( B > digit . length d == 0 ) return 0 ;
else if ( B < digit . length ) {
if ( A [ 0 ] == 0 && B != 1 ) return Math . floor ( ( d - 1 ) * Math . pow ( d , B - 1 ) ) ; else return Math . floor ( Math . pow ( d , B ) ) ; }
else { let dp = new Array ( B + 1 ) ; let lower = new Array ( MAX + 1 ) ; for ( let i = 0 ; i < dp . length ; i ++ ) { dp [ i ] = 0 ; } for ( let i = 0 ; i < lower . length ; i ++ ) { lower [ i ] = 0 ; }
for ( let i = 0 ; i < d ; i ++ ) lower [ A [ i ] + 1 ] = 1 ; for ( let i = 1 ; i <= MAX ; i ++ ) lower [ i ] = lower [ i - 1 ] + lower [ i ] ; let flag = true ; dp [ 0 ] = 0 ; for ( let i = 1 ; i <= B ; i ++ ) { d2 = lower [ digit [ i - 1 ] ] ; dp [ i ] = dp [ i - 1 ] * d ;
if ( i == 1 && A [ 0 ] == 0 && B != 1 ) d2 = d2 - 1 ;
if ( flag ) dp [ i ] += d2 ;
flag = ( flag & ( lower [ digit [ i - 1 ] + 1 ] == lower [ digit [ i - 1 ] ] + 1 ) ) ; } return dp [ B ] ; } }
let arr = [ 0 , 1 , 2 , 5 ] ; let N = 2 ; let k = 21 ; document . write ( solve ( arr , N , k ) ) ;
function solve ( dp , wt , K , M , used ) {
if ( wt < 0 ) { return 0 ; } if ( wt == 0 ) {
if ( used == 1 ) { return 1 ; } return 0 ; } if ( dp [ wt ] [ used ] != - 1 ) { return dp [ wt ] [ used ] ; } let ans = 0 ; for ( let i = 1 ; i <= K ; i ++ ) {
if ( i >= M ) { ans += solve ( dp , wt - i , K , M , used 1 ) ; } else { ans += solve ( dp , wt - i , K , M , used ) ; } } return dp [ wt ] [ used ] = ans ; }
let W = 3 , K = 3 , M = 2 ; let dp = new Array ( W + 1 ) ; for ( let i = 0 ; i < W + 1 ; i ++ ) { dp [ i ] = new Array ( 2 ) ; for ( let j = 0 ; j < 2 ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } document . write ( solve ( dp , W , K , M , 0 ) + " " ) ;
function partitions ( n ) { var p = Array ( n + 1 ) . fill ( 0 ) ;
p [ 0 ] = 1 ; for ( i = 1 ; i <= n ; ++ i ) { var k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 != 0 ? 1 : - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) { k *= - 1 ; } else { k = 1 - k ; } } } return p [ n ] ; }
var N = 20 ; document . write ( partitions ( N ) ) ;
function LIP ( dp , mat , n , m , x , y ) {
if ( dp [ x ] [ y ] < 0 ) { let result = 0 ;
if ( x == n - 1 && y == m - 1 ) return dp [ x ] [ y ] = 1 ;
if ( x == n - 1 y == m - 1 ) result = 1 ;
if ( x + 1 < n && mat [ x ] [ y ] < mat [ x + 1 ] [ y ] ) result = 1 + LIP ( dp , mat , n , m , x + 1 , y ) ;
if ( y + 1 < m && mat [ x ] [ y ] < mat [ x ] [ y + 1 ] ) result = Math . max ( result , 1 + LIP ( dp , mat , n , m , x , y + 1 ) ) ; dp [ x ] [ y ] = result ; } return dp [ x ] [ y ] ; }
function wrapper ( mat , n , m ) { let dp = new Array ( 10 ) ; for ( let i = 0 ; i < 10 ; i ++ ) { dp [ i ] = new Array ( 10 ) ; for ( let j = 0 ; j < 10 ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } return LIP ( dp , mat , n , m , 0 , 0 ) ; }
let mat = [ [ 1 , 2 , 3 , 4 ] , [ 2 , 2 , 3 , 4 ] , [ 3 , 2 , 3 , 4 ] , [ 4 , 5 , 6 , 7 ] , ] ; let n = 4 , m = 4 ; document . write ( wrapper ( mat , n , m ) ) ;
function countPaths ( n , m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
let n = 3 , m = 2 ; document . write ( " " + countPaths ( n , m ) ) ;
let MAX = 100 ;
function getMaxGold ( gold , m , n ) {
let goldTable = new Array ( m ) ; for ( let i = 0 ; i < m ; i ++ ) { goldTable [ i ] = new Array ( n ) ; for ( let j = 0 ; j < n ; j ++ ) { goldTable [ i ] [ j ] = 0 ; } } for ( let col = n - 1 ; col >= 0 ; col -- ) { for ( let row = 0 ; row < m ; row ++ ) {
let right = ( col == n - 1 ) ? 0 : goldTable [ row ] [ col + 1 ] ;
let right_up = ( row == 0 col == n - 1 ) ? 0 : goldTable [ row - 1 ] [ col + 1 ] ;
let right_down = ( row == m - 1 col == n - 1 ) ? 0 : goldTable [ row + 1 ] [ col + 1 ] ;
goldTable [ row ] [ col ] = gold [ row ] [ col ] + Math . max ( right , Math . max ( right_up , right_down ) ) ; } }
let res = goldTable [ 0 ] [ 0 ] ; for ( let i = 1 ; i < m ; i ++ ) res = Math . max ( res , goldTable [ i ] [ 0 ] ) ; return res ; }
let gold = [ [ 1 , 3 , 1 , 5 ] , [ 2 , 2 , 4 , 1 ] , [ 5 , 0 , 2 , 3 ] , [ 0 , 6 , 1 , 2 ] ] ; let m = 4 , n = 4 ; document . write ( getMaxGold ( gold , m , n ) ) ;
let M = 100 ;
function minAdjustmentCost ( A , n , target ) {
let dp = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { dp [ i ] = new Array ( n ) ; for ( let j = 0 ; j <= M ; j ++ ) { dp [ i ] [ j ] = 0 ; } }
for ( let j = 0 ; j <= M ; j ++ ) dp [ 0 ] [ j ] = Math . abs ( j - A [ 0 ] ) ;
for ( let i = 1 ; i < n ; i ++ ) {
for ( let j = 0 ; j <= M ; j ++ ) {
dp [ i ] [ j ] = Number . MAX_VALUE ;
let k = Math . max ( j - target , 0 ) ; for ( ; k <= Math . min ( M , j + target ) ; k ++ ) dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , dp [ i - 1 ] [ k ] + Math . abs ( A [ i ] - j ) ) ; } }
let res = Number . MAX_VALUE ; for ( let j = 0 ; j <= M ; j ++ ) res = Math . min ( res , dp [ n - 1 ] [ j ] ) ; return res ; }
let arr = [ 55 , 77 , 52 , 61 , 39 , 6 , 25 , 60 , 49 , 47 ] ; let n = arr . length ; let target = 10 ; document . write ( " " + minAdjustmentCost ( arr , n , target ) ) ;
function totalCombination ( L , R ) {
let count = 0 ;
let K = R - L ;
if ( K < L ) return 0 ;
let ans = K - L ;
count = ( ( ans + 1 ) * ( ans + 2 ) ) / 2 ;
return count ; }
let L = 2 , R = 6 ; document . write ( totalCombination ( L , R ) ) ;
function printArrays ( n ) {
let A = [ ] ; let B = [ ] ;
for ( let i = 1 ; i <= 2 * n ; i ++ ) {
if ( i % 2 == 0 ) A . push ( i ) ; else B . push ( i ) ; }
document . write ( " " ) ; for ( let i = 0 ; i < n ; i ++ ) { document . write ( A [ i ] ) ; if ( i != n - 1 ) document . write ( " " ) ; } document . write ( " " + " " ) ;
document . write ( " " ) ; for ( let i = 0 ; i < n ; i ++ ) { document . write ( B [ i ] ) ; if ( i != n - 1 ) document . write ( " " ) ; } document . write ( " " ) ; }
let N = 5 ;
printArrays ( N ) ;
function flipBitsOfAandB ( A , B ) {
for ( i = 0 ; i < 32 ; i ++ ) {
if ( ( ( A & ( 1 << i ) ) & ( B & ( 1 << i ) ) ) != 0 ) {
A = A ^ ( 1 << i ) ;
B = B ^ ( 1 << i ) ; } }
document . write ( A + " " + B ) ; }
var A = 7 , B = 4 ; flipBitsOfAandB ( A , B ) ;
function findDistinctSums ( N ) { return ( 2 * N - 1 ) ; }
let N = 3 ; document . write ( findDistinctSums ( N ) ) ;
function countSubstrings ( str ) {
let freq = new Array ( 3 ) . fill ( 0 )
let count = 0 ; let i = 0 ;
for ( let j = 0 ; j < str . length ; j ++ ) {
freq [ str . charCodeAt ( j ) - ' ' . charCodeAt ( 0 ) ] ++ ;
while ( freq [ 0 ] > 0 && freq [ 1 ] > 0 && freq [ 2 ] > 0 ) { freq [ str . charCodeAt ( i ++ ) - ' ' . charCodeAt ( 0 ) ] -- ; }
count += i ; }
return count ; }
let str = " " ; let count = countSubstrings ( str ) ; document . write ( count ) ;
function minFlips ( str ) {
let count = 0 ;
if ( str . length <= 2 ) { return 0 ; }
for ( let i = 0 ; i < str . length - 2 ; ) {
if ( str [ i ] == str [ i + 1 ] && str [ i + 2 ] == str [ i + 1 ] ) { i = i + 3 ; count ++ ; } else { i ++ ; } }
return count ; }
let S = " " ; document . write ( minFlips ( S ) ) ;
function convertToHex ( num ) { let temp = " " ; while ( num != 0 ) { let rem = num % 16 ; let c = 0 ; if ( rem < 10 ) { c = rem + 48 ; } else { c = rem + 87 ; } temp += String . fromCharCode ( c ) ; num = Math . floor ( num / 16 ) ; } return temp ; }
function encryptString ( S , N ) { let ans = " " ;
for ( let i = 0 ; i < N ; i ++ ) { let ch = S [ i ] ; let count = 0 ; let hex ;
while ( i < N && S [ i ] == ch ) {
count ++ ; i ++ ; }
i -- ;
hex = convertToHex ( count ) ;
ans += ch ;
ans += hex ; }
ans = ans . split ( ' ' ) . reverse ( ) . join ( " " ) ;
return ans ; }
let S = " " ; let N = S . length ;
document . write ( encryptString ( S , N ) ) ;
function binomialCoeff ( n , k ) { let res = 1 ;
if ( k > n - k ) k = n - k ;
for ( let i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
function countOfString ( N ) {
let Stotal = Math . pow ( 2 , N ) ;
let Sequal = 0 ;
if ( N % 2 == 0 ) Sequal = binomialCoeff ( N , N / 2 ) ; let S1 = ( Stotal - Sequal ) / 2 ; return S1 ; }
let N = 3 ; document . write ( countOfString ( N ) ) ;
function removeCharRecursive ( str , X ) {
if ( str . length == 0 ) { return " " ; }
if ( str . charAt ( 0 ) == X ) {
return removeCharRecursive ( str . substring ( 1 ) , X ) ; }
return str . charAt ( 0 ) + removeCharRecursive ( str . substring ( 1 ) , X ) ; }
var str = " " ;
var X = ' ' ;
str = removeCharRecursive ( str , X ) ; document . write ( str ) ;
function isValid ( a1 , a2 , str , flag ) { let v1 , v2 ;
if ( flag == 0 ) { v1 = str [ 4 ] ; v2 = str [ 3 ] ; } else {
v1 = str [ 1 ] ; v2 = str [ 0 ] ; }
if ( v1 != a1 && v1 != ' ' ) return false ; if ( v2 != a2 && v2 != ' ' ) return false ; return true ; }
function inRange ( hh , mm , L , R ) { let a = Math . abs ( hh - mm ) ;
if ( a < L a > R ) return false ; return true ; }
function displayTime ( hh , mm ) { if ( hh > 10 ) document . write ( hh + " " ) ; else if ( hh < 10 ) document . write ( " " + hh + " " ) ; if ( mm > 10 ) document . write ( mm + " " ) ; else if ( mm < 10 ) document . write ( " " + mm + " " ) ; }
function maximumTimeWithDifferenceInRange ( str , L , R ) { let i = 0 , j = 0 ; let h1 , h2 , m1 , m2 ;
for ( i = 23 ; i >= 0 ; i -- ) { h1 = i % 10 ; h2 = Math . floor ( i / 10 ) ;
if ( ! isValid ( String . fromCharCode ( h1 ) , String . fromCharCode ( h2 ) , str , 1 ) ) { continue ; }
for ( j = 59 ; j >= 0 ; j -- ) { m1 = j % 10 ; m2 = Math . floor ( j / 10 ) ;
if ( ! isValid ( String . fromCharCode ( m1 ) , String . fromCharCode ( m2 ) , str , 0 ) ) { continue ; } if ( inRange ( i , j , L , R ) ) { displayTime ( i , j ) ; return ; } } } if ( inRange ( i , j , L , R ) ) displayTime ( i , j ) ; else document . write ( " " ) ; }
let timeValue = " " ;
let L = 20 , R = 39 ; maximumTimeWithDifferenceInRange ( timeValue , L , R ) ;
function check ( s , n ) {
var st = [ ] ;
for ( var i = 0 ; i < n ; i ++ ) {
if ( st . length != 0 && st [ st . length - 1 ] == s [ i ] ) st . pop ( ) ;
else st . push ( s [ i ] ) ; }
if ( st . length == 0 ) { return true ; }
else { return false ; } }
var str = " " ; var n = str . length ;
if ( check ( str , n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function findNumOfValidWords ( w , p ) {
var m = new Map ( ) ;
var res = [ ] ;
w . forEach ( s => { var val = 0 ;
s . split ( ' ' ) . forEach ( c => { val = val | ( 1 << ( c . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ) ; } ) ;
if ( m . has ( val ) ) m . set ( val , m . get ( val ) + 1 ) else m . set ( val , 1 ) } ) ;
p . forEach ( s => { var val = 0 ;
s . split ( ' ' ) . forEach ( c => { val = val | ( 1 << ( c . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ) ; } ) ; var temp = val ; var first = s [ 0 ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ; var count = 0 ; while ( temp != 0 ) {
if ( ( ( temp >> first ) & 1 ) == 1 ) { if ( m . has ( temp ) ) { count += m . get ( temp ) ; } }
temp = ( temp - 1 ) & val ; }
res . push ( count ) ; } ) ;
res . forEach ( it => { document . write ( it + " " ) ; } ) ; }
var arr1 = [ " " , " " , " " , " " , " " , " " , " " ] ; var arr2 = [ " " , " " , " " , " " , " " , " " ] ;
findNumOfValidWords ( arr1 , arr2 ) ;
function flip ( s ) { for ( let i = 0 ; i < s . length ; i ++ ) {
if ( s [ i ] == ' ' ) {
while ( s [ i ] == ' ' ) {
s [ i ] = ' ' ; i ++ ; }
break ; } } return s . join ( " " ) ; }
let s = " " ; document . write ( flip ( s . split ( ' ' ) ) ) ;
function getOrgString ( s ) {
document . write ( s [ 0 ] ) ;
var i = 1 ; while ( i < s . length ) {
if ( s [ i ] . charCodeAt ( 0 ) >= " " . charCodeAt ( 0 ) && s [ i ] . charCodeAt ( 0 ) <= " " . charCodeAt ( 0 ) ) document . write ( " " + s [ i ] . toLowerCase ( ) ) ;
else document . write ( s [ i ] ) ; i ++ ; } }
var s = " " ; getOrgString ( s ) ;
function countChar ( str , x ) { let count = 0 ; let n = 10 ; for ( let i = 0 ; i < str . length ; i ++ ) if ( str [ i ] == x ) count ++ ;
let repetitions = n / str . length ; count = count * repetitions ;
for ( let i = 0 ; i < n % str . length ; i ++ ) { if ( str [ i ] == x ) count ++ ; } return count ; }
let str = " " ; document . write ( countChar ( str , ' ' ) ) ;
function countFreq ( arr , n , limit ) {
let count = new Array ( limit + 1 ) ; count . fill ( 0 ) ;
for ( let i = 0 ; i < n ; i ++ ) count [ arr [ i ] ] ++ ; for ( let i = 0 ; i <= limit ; i ++ ) if ( count [ i ] > 0 ) document . write ( i + " " + count [ i ] + " " ) ; }
let arr = [ 5 , 5 , 6 , 6 , 5 , 6 , 1 , 2 , 3 , 10 , 10 ] ; let n = arr . length ; let limit = 10 ; countFreq ( arr , n , limit ) ;
function check ( s , m ) {
let l = s . length ;
let c1 = 0 ;
let c2 = 0 ; for ( let i = 0 ; i < l ; i ++ ) { if ( s [ i ] == ' ' ) { c2 = 0 ;
c1 ++ ; } else { c1 = 0 ;
c2 ++ ; } if ( c1 == m c2 == m ) return true ; } return false ; }
let s = " " ; let m = 2 ;
if ( check ( s , m ) ) document . write ( " " ) ; else document . write ( " " ) ;
function productAtKthLevel ( tree , k ) { let level = - 1 ;
let product = 1 ; let n = tree . length ; for ( let i = 0 ; i < n ; i ++ ) {
if ( tree [ i ] == ' ' ) level ++ ;
else if ( tree [ i ] == ' ' ) level -- ; else {
if ( level == k ) product *= ( tree [ i ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ) ; } }
return product ; }
let tree = " " ; let k = 2 ; document . write ( productAtKthLevel ( tree , k ) ) ;
function findDuplciates ( a , n , m ) {
var isPresent = Array ( n ) . fill ( ) . map ( ( ) => Array ( m ) . fill ( 0 ) ) ; for ( var i = 0 ; i < n ; i ++ ) { for ( var j = 0 ; j < m ; j ++ ) { isPresent [ i ] [ j ] = false ; } } for ( var i = 0 ; i < n ; i ++ ) { for ( var j = 0 ; j < m ; j ++ ) {
for ( var k = 0 ; k < n ; k ++ ) { if ( a [ i ] . charAt ( j ) == a [ k ] . charAt ( j ) && i != k ) { isPresent [ i ] [ j ] = true ; isPresent [ k ] [ j ] = true ; } }
for ( k = 0 ; k < m ; k ++ ) { if ( a [ i ] . charAt ( j ) == a [ i ] . charAt ( k ) && j != k ) { isPresent [ i ] [ j ] = true ; isPresent [ i ] [ k ] = true ; } } } } for ( var i = 0 ; i < n ; i ++ ) for ( var j = 0 ; j < m ; j ++ )
if ( isPresent [ i ] [ j ] == false ) document . write ( a [ i ] . charAt ( j ) ) ; }
var n = 2 , m = 2 ;
var a = [ " " , " " ] ;
findDuplciates ( a , n , m ) ;
function isValidISBN ( isbn ) {
let n = isbn . length ; if ( n != 10 ) return false ;
let sum = 0 ; for ( let i = 0 ; i < 9 ; i ++ ) { let digit = isbn [ i ] - ' ' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
let last = isbn [ 9 ] ; if ( last != ' ' && ( last < ' ' last > ' ' ) ) return false ;
sum += ( ( last == ' ' ) ? 10 : ( last - ' ' ) ) ;
return ( sum % 11 == 0 ) ; }
let isbn = " " ; if ( isValidISBN ( isbn ) ) document . write ( " " ) ; else document . write ( " " ) ;
function isVowel ( c ) { return ( c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' ) ; }
function reverseVowel ( str1 ) { let j = 0 ;
let str = str1 . split ( ' ' ) ; let vowel = " " ; for ( let i = 0 ; i < str . length ; i ++ ) { if ( isVowel ( str [ i ] ) ) { j ++ ; vowel += str [ i ] ; } }
for ( let i = 0 ; i < str . length ; i ++ ) { if ( isVowel ( str [ i ] ) ) { str [ i ] = vowel [ -- j ] ; } } return str . join ( " " ) ; }
let str = " " ; document . write ( reverseVowel ( str ) ) ;
function firstLetterWord ( str ) { let result = " " ;
let v = true ; for ( let i = 0 ; i < str . length ; i ++ ) {
if ( str [ i ] == ' ' ) { v = true ; }
else if ( str [ i ] != ' ' && v == true ) { result += ( str [ i ] ) ; v = false ; } } return result ; }
let str = " " ; document . write ( firstLetterWord ( str ) ) ;
function dfs ( i , j , grid , vis , z , z_count ) { let n = grid . length , m = grid [ 0 ] . length ;
vis [ i ] [ j ] = true ; if ( grid [ i ] [ j ] == 0 )
z ++ ;
if ( grid [ i ] [ j ] == 2 ) {
if ( z == z_count ) ans ++ ; vis [ i ] [ j ] = false ; return ; }
if ( i >= 1 && ! vis [ i - 1 ] [ j ] && grid [ i - 1 ] [ j ] != - 1 ) dfs ( i - 1 , j , grid , vis , z , z_count ) ;
if ( i < n - 1 && ! vis [ i + 1 ] [ j ] && grid [ i + 1 ] [ j ] != - 1 ) dfs ( i + 1 , j , grid , vis , z , z_count ) ;
if ( j >= 1 && ! vis [ i ] [ j - 1 ] && grid [ i ] [ j - 1 ] != - 1 ) dfs ( i , j - 1 , grid , vis , z , z_count ) ;
if ( j < m - 1 && ! vis [ i ] [ j + 1 ] && grid [ i ] [ j + 1 ] != - 1 ) dfs ( i , j + 1 , grid , vis , z , z_count ) ;
vis [ i ] [ j ] = false ; }
function uniquePaths ( grid ) {
let n = grid . length , m = grid [ 0 ] . length ; let vis = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { vis [ i ] = new Array ( m ) ; for ( let j = 0 ; j < m ; j ++ ) { vis [ i ] [ j ] = false ; } } let x = 0 , y = 0 ; for ( let i = 0 ; i < n ; ++ i ) { for ( let j = 0 ; j < m ; ++ j ) {
if ( grid [ i ] [ j ] == 0 ) z_count ++ ; else if ( grid [ i ] [ j ] == 1 ) {
x = i ; y = j ; } } } dfs ( x , y , grid , vis , 0 , z_count ) ; return ans ; }
let grid = [ [ 1 , 0 , 0 , 0 ] , [ 0 , 0 , 0 , 0 ] , [ 0 , 0 , 2 , - 1 ] ] ; document . write ( uniquePaths ( grid ) ) ;
function numPairs ( a , n ) { let ans , i , index ;
ans = 0 ;
for ( i = 0 ; i < n ; i ++ ) a [ i ] = Math . abs ( a [ i ] ) ;
a . sort ( ) ;
for ( i = 0 ; i < n ; i ++ ) { index = 2 ; ans += index - i - 1 ; }
return ans ; }
let a = [ 3 , 6 ] ; let n = a . length ; document . write ( numPairs ( a , n ) ) ;
function areaOfSquare ( S ) {
let area = S * S ; return area ; }
let S = 5 ;
document . write ( areaOfSquare ( S ) ) ;
function maxPointOfIntersection ( x , y ) { let k = y * ( y - 1 ) / 2 ; k = k + x * ( 2 * y + x - 1 ) ; return k ; }
let x = 3 ;
let y = 4 ;
document . write ( maxPointOfIntersection ( x , y ) ) ;
function Icosihenagonal_num ( n ) {
return ( 19 * n * n - 17 * n ) / 2 ; }
let n = 3 ; document . write ( Icosihenagonal_num ( n ) + " " ) ; n = 10 ; document . write ( Icosihenagonal_num ( n ) ) ;
function find_Centroid ( v ) { let ans = new Array ( 2 ) ; ans . fill ( 0 ) ; let n = v . length ; let signedArea = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { let x0 = v [ i ] [ 0 ] , y0 = v [ i ] [ 1 ] ; let x1 = v [ ( i + 1 ) % n ] [ 0 ] , y1 = v [ ( i + 1 ) % n ] [ 1 ] ;
let A = ( x0 * y1 ) - ( x1 * y0 ) ; signedArea += A ;
ans [ 0 ] += ( x0 + x1 ) * A ; ans [ 1 ] += ( y0 + y1 ) * A ; } signedArea *= 0.5 ; ans [ 0 ] = ( ans [ 0 ] ) / ( 6 * signedArea ) ; ans [ 1 ] = ( ans [ 1 ] ) / ( 6 * signedArea ) ; return ans ; }
let vp = [ [ 1 , 2 ] , [ 3 , - 4 ] , [ 6 , - 7 ] ] ; let ans = find_Centroid ( vp ) ; document . write ( ans [ 0 ] . toFixed ( 11 ) + " " + ans [ 1 ] ) ;
var d = 10 ; var a ;
a = parseInt ( ( 360 - ( 6 * d ) ) / 4 ) ;
document . write ( a + " " + ( a + d ) + " " + ( a + ( 2 * d ) ) + " " + ( a + ( 3 * d ) ) ) ;
function distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) { let x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = Math . abs ( ( c2 * z1 + d2 ) ) / ( Math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; document . write ( " " + d ) ; } else document . write ( " " ) ; }
let a1 = 1 ; let b1 = 2 ; let c1 = - 1 ; let d1 = 1 ; let a2 = 3 ; let b2 = 6 ; let c2 = - 3 ; let d2 = - 4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ;
function factorial ( n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
function numOfNecklace ( N ) {
var ans = factorial ( N ) / ( factorial ( N / 2 ) * factorial ( N / 2 ) ) ;
ans = ans * factorial ( N / 2 - 1 ) ; ans = ans * factorial ( N / 2 - 1 ) ;
ans /= 2 ;
return ans ; }
var N = 4 ;
document . write ( numOfNecklace ( N ) ) ;
function isDivisibleByDivisor ( S , D ) {
S %= D ;
var hashMap = [ ] ; hashMap . push ( S ) ; for ( var i = 0 ; i <= D ; i ++ ) {
S += S % D ; S %= D ;
if ( hashMap . includes ( S ) ) {
if ( S == 0 ) { return " " ; } return " " ; }
else hashMap . push ( S ) ; } return " " ; }
var S = 3 , D = 6 ; document . write ( isDivisibleByDivisor ( S , D ) ) ;
function minimumSteps ( x , y ) {
var cnt = 0 ;
while ( x != 0 && y != 0 ) {
if ( x > y ) {
cnt += x / y ; x %= y ; }
else {
cnt += y / x ; y %= x ; } } cnt -- ;
if ( x > 1 y > 1 ) cnt = - 1 ;
document . write ( cnt ) ; }
var x = 3 , y = 1 ; minimumSteps ( x , y ) ;
function printLeast ( arr ) {
let min_avail = 1 , pos_of_I = 0 ;
let al = [ ] ;
if ( arr [ 0 ] == ' ' ) { al . push ( 1 ) ; al . push ( 2 ) ; min_avail = 3 ; pos_of_I = 1 ; } else { al . push ( 2 ) ; al . push ( 1 ) ; min_avail = 3 ; pos_of_I = 0 ; }
for ( let i = 1 ; i < arr . length ; i ++ ) { if ( arr [ i ] == ' ' ) { al . push ( min_avail ) ; min_avail ++ ; pos_of_I = i + 1 ; } else { al . push ( al [ i ] ) ; for ( let j = pos_of_I ; j <= i ; j ++ ) al [ j ] = al [ j ] + 1 ; min_avail ++ ; } }
for ( let i = 0 ; i < al . length ; i ++ ) document . write ( al [ i ] + " " ) ; document . write ( " " ) ; }
printLeast ( " " ) ; printLeast ( " " ) ; printLeast ( " " ) ; printLeast ( " " ) ; printLeast ( " " ) ; printLeast ( " " ) ; printLeast ( " " ) ;
function PrintMinNumberForPattern ( seq ) {
let result = " " ;
let stk = [ ] ;
for ( let i = 0 ; i <= seq . length ; i ++ ) {
stk . push ( i + 1 ) ;
if ( i == seq . length seq [ i ] == ' ' ) {
while ( stk . length != 0 ) {
result += ( stk [ stk . length - 1 ] ) . toString ( ) ; result += " " ; stk . pop ( ) ; } } } document . write ( result + " " ) ; }
PrintMinNumberForPattern ( " " ) ; PrintMinNumberForPattern ( " " ) ; PrintMinNumberForPattern ( " " ) ; PrintMinNumberForPattern ( " " ) ; PrintMinNumberForPattern ( " " ) ; PrintMinNumberForPattern ( " " ) ; PrintMinNumberForPattern ( " " ) ;
function getMinNumberForPattern ( seq ) { let n = seq . length ; if ( n >= 9 ) return " " ; let result = new Array ( n + 1 ) ; let count = 1 ;
for ( let i = 0 ; i <= n ; i ++ ) { if ( i == n seq [ i ] == ' ' ) { for ( let j = i - 1 ; j >= - 1 ; j -- ) { result [ j + 1 ] = String . fromCharCode ( ' ' . charCodeAt ( ) + count ++ ) ; if ( j >= 0 && seq [ j ] == ' ' ) break ; } } } return result . join ( " " ) ; }
let inputs = [ " " , " " , " " , " " , " " , " " , " " ] ; for ( let input = 0 ; input < inputs . length ; input ++ ) { document . write ( getMinNumberForPattern ( inputs [ input ] ) + " " ) ; }
function isPrime ( n ) { var i , c = 0 ; for ( i = 1 ; i < n / 2 ; i ++ ) { if ( n % i == 0 ) c ++ ; } if ( c == 1 ) return 1 ; else return 0 ; }
function findMinNum ( arr , n ) {
var first = 0 , last = 0 , num , rev , i ; var hash = new Array ( 10 ) . fill ( 0 ) ;
for ( var i = 0 ; i < n ; i ++ ) { hash [ arr [ i ] ] ++ ; }
document . write ( " " ) ; for ( var i = 0 ; i <= 9 ; i ++ ) {
for ( var j = 0 ; j < hash [ i ] ; j ++ ) document . write ( i ) ; } document . write ( " " ) ;
for ( i = 0 ; i <= 9 ; i ++ ) { if ( hash [ i ] != 0 ) { first = i ; break ; } }
for ( i = 9 ; i >= 0 ; i -- ) { if ( hash [ i ] != 0 ) { last = i ; break ; } } num = first * 10 + last ; rev = last * 10 + first ;
document . write ( " " ) ; if ( isPrime ( num ) && isPrime ( rev ) ) document . write ( num + " " + rev ) ; else if ( isPrime ( num ) ) document . write ( num ) ; else if ( isPrime ( rev ) ) document . write ( rev ) ; else document . write ( " " ) ; }
var arr = [ 1 , 2 , 4 , 7 , 8 ] ; findMinNum ( arr , 5 ) ;
function gcd ( a , b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
function coprime ( a , b ) {
return ( gcd ( a , b ) == 1 ) ; }
function possibleTripletInRange ( L , R ) { let flag = false ; let possibleA = 0 , possibleB = 0 , possibleC = 0 ;
for ( let a = L ; a <= R ; a ++ ) { for ( let b = a + 1 ; b <= R ; b ++ ) { for ( let c = b + 1 ; c <= R ; c ++ ) {
if ( coprime ( a , b ) && coprime ( b , c ) && ! coprime ( a , c ) ) { flag = true ; possibleA = a ; possibleB = b ; possibleC = c ; break ; } } } }
if ( flag == true ) { document . write ( " " + possibleA + " " + possibleB + " " + possibleC + " " + " " + " " + L + " " + R + " " ) ; } else { document . write ( " " + " " + L + " " + R + " " ) ; } }
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ;
function possibleToReach ( a , b ) {
let c = Math . cbrt ( a * b ) ;
let re1 = a / c ; let re2 = b / c ;
if ( ( re1 * re1 * re2 == a ) && ( re2 * re2 * re1 == b ) ) return true ; else return false ; }
let A = 60 , B = 450 ; if ( possibleToReach ( A , B ) ) document . write ( " " ) ; else document . write ( " " ) ;
function isUndulating ( n ) {
if ( n . length <= 2 ) return false ;
for ( let i = 2 ; i < n . length ; i ++ ) if ( n [ i - 2 ] != n [ i ] ) return false ; return true ; }
let n = " " ; if ( isUndulating ( n ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function Series ( n ) { let i ; let sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
let n = 3 ; let res = Series ( n ) ; document . write ( res ) ;
function counLastDigitK ( low , high , k ) { let mlow = 10 * ( Math . ceil ( low / 10.0 ) ) ; let mhigh = 10 * ( Math . floor ( high / 10.0 ) ) ; let count = ( mhigh - mlow ) / 10 ; if ( high % 10 >= k ) count ++ ; if ( low % 10 <= k && ( low % 10 ) > 0 ) count ++ ; return count ; }
let low = 3 , high = 35 , k = 3 ; document . write ( counLastDigitK ( low , high , k ) ) ;
function sum ( L , R ) {
let p = Math . floor ( R / 6 ) ;
let q = Math . floor ( ( L - 1 ) / 6 ) ;
let sumR = Math . floor ( 3 * ( p * ( p + 1 ) ) ) ;
let sumL = Math . floor ( ( q * ( q + 1 ) ) * 3 ) ;
return sumR - sumL ; }
let L = 1 , R = 20 ; document . write ( sum ( L , R ) ) ;
function prevNum ( str ) { let len = str . length ; let index = - 1 ;
for ( let i = len - 2 ; i >= 0 ; i -- ) { if ( str [ i ] > str [ i + 1 ] ) { index = i ; break ; } }
let smallGreatDgt = - 1 ; for ( let i = len - 1 ; i > index ; i -- ) { if ( str [ i ] < str [ index ] ) { if ( smallGreatDgt == - 1 ) { smallGreatDgt = i ; } else if ( str [ i ] >= str [ smallGreatDgt ] ) { smallGreatDgt = i ; } } }
if ( index == - 1 ) { return " " ; }
if ( smallGreatDgt != - 1 ) { str = swap ( str , index , smallGreatDgt ) ; return str ; } return " " ; } function swap ( str , i , j ) { let ch = str . split ( ' ' ) ; let temp = ch [ i ] ; ch [ i ] = ch [ j ] ; ch [ j ] = temp ; return ch . join ( " " ) ; }
let str = " " ; document . write ( prevNum ( str ) ) ;
function horner ( poly , n , x ) {
var result = poly [ 0 ] ;
for ( var i = 1 ; i < n ; i ++ ) result = result * x + poly [ i ] ; return result ; }
function findSign ( poly , n , x ) { var result = horner ( poly , n , x ) ; if ( result > 0 ) return 1 ; else if ( result < 0 ) return - 1 ; return 0 ; }
var poly = [ 2 , - 6 , 2 , - 1 ] ; var x = 3 ; var n = poly . length ; document . write ( " " + findSign ( poly , n , x ) ) ;
let MAX = 100005 ;
let isPrime = new Array ( MAX ) . fill ( 0 ) ;
function sieveOfEratostheneses ( ) { isPrime [ 1 ] = true ; for ( let i = 2 ; i * i < MAX ; i ++ ) { if ( ! isPrime [ i ] ) { for ( let j = 2 * i ; j < MAX ; j += i ) isPrime [ j ] = true ; } } }
function findPrime ( n ) { let num = n + 1 ;
while ( num > 0 ) {
if ( ! isPrime [ num ] ) return num ;
num = num + 1 ; } return 0 ; }
function minNumber ( arr , n ) {
sieveOfEratostheneses ( ) ; let sum = 0 ;
for ( let i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( ! isPrime [ sum ] ) return 0 ;
let num = findPrime ( sum ) ;
return num - sum ; }
let arr = [ 2 , 4 , 6 , 8 , 12 ] ; let n = arr . length ; document . write ( minNumber ( arr , n ) ) ;
function SubArraySum ( arr , n ) { let result = 0 , temp = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
temp = 0 ; for ( let j = i ; j < n ; j ++ ) {
temp += arr [ j ] ; result += temp ; } } return result ; }
let arr = [ 1 , 2 , 3 ] ; let n = arr . length ; document . write ( " " + SubArraySum ( arr , n ) + " " ) ;
function highestPowerof2 ( n ) { let p = parseInt ( Math . log ( n ) / Math . log ( 2 ) , 10 ) ; return Math . pow ( 2 , p ) ; }
let n = 10 ; document . write ( highestPowerof2 ( n ) ) ;
function aModM ( s , mod ) { let number = 0 ; for ( let i = 0 ; i < s . length ; i ++ ) {
number = ( number * 10 ) ; let x = ( s [ i ] - ' ' ) ; number = number + x ; number %= mod ; } return number ; }
function ApowBmodM ( a , b , m ) {
let ans = aModM ( a , m ) ; let mul = ans ;
for ( let i = 1 ; i < b ; i ++ ) ans = ( ans * mul ) % m ; return ans ; }
let a = " " ; let b = 3 , m = 11 ; document . write ( ApowBmodM ( a , b , m ) ) ;
class Data { constructor ( x , y ) { this . x = x ; this . y = y ; } }
function interpolate ( f , xi , n ) {
for ( let i = 0 ; i < n ; i ++ ) {
let term = f [ i ] . y ; for ( let j = 0 ; j < n ; j ++ ) { if ( j != i ) term = term * ( xi - f [ j ] . x ) / ( f [ i ] . x - f [ j ] . x ) ; }
result += term ; } return result ; }
let f = [ new Data ( 0 , 2 ) , new Data ( 1 , 3 ) , new Data ( 2 , 12 ) , new Data ( 5 , 147 ) ] ;
document . write ( " " + interpolate ( f , 3 , 4 ) ) ;
function SieveOfSundaram ( n ) {
let nNew = ( n - 1 ) / 2 ;
let marked = [ ] ;
for ( let i = 1 ; i <= nNew ; i ++ ) for ( let j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) document . write ( 2 + " " ) ;
for ( let i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) document . write ( 2 * i + 1 + " " ) ; return - 1 ; }
let n = 20 ; SieveOfSundaram ( n ) ;
function constructArray ( A , N , K ) {
let B = new Array ( N ) ;
let totalXOR = A [ 0 ] ^ K ;
for ( let i = 0 ; i < N ; i ++ ) B [ i ] = totalXOR ^ A [ i ] ;
for ( let i = 0 ; i < N ; i ++ ) { document . write ( B [ i ] + " " ) ; } }
let A = [ 13 , 14 , 10 , 6 ] , K = 2 ; let N = A . length ;
constructArray ( A , N , K ) ;
function extraElement ( A , B , n ) {
let ans = 0 ;
for ( let i = 0 ; i < n ; i ++ ) ans ^= A [ i ] ; for ( let i = 0 ; i < n + 1 ; i ++ ) ans ^= B [ i ] ; return ans ; }
let A = [ 10 , 15 , 5 ] ; let B = [ 10 , 100 , 15 , 5 ] ; let n = A . length ; document . write ( extraElement ( A , B , n ) ) ;
function hammingDistance ( n1 , n2 ) { let x = n1 ^ n2 ; let setBits = 0 ; while ( x > 0 ) { setBits += x & 1 ; x >>= 1 ; } return setBits ; }
let n1 = 9 , n2 = 14 ; document . write ( hammingDistance ( 9 , 14 ) ) ;
function printSubsets ( n ) { for ( let i = n ; i > 0 ; i = ( i - 1 ) & n ) document . write ( i + " " ) ; document . write ( " " ) ; }
let n = 9 ; printSubsets ( n ) ;
function setBitNumber ( n ) {
let k = parseInt ( Math . log ( n ) / Math . log ( 2 ) , 10 ) ;
return 1 << k ; }
let n = 273 ; document . write ( setBitNumber ( n ) ) ;
function subset ( ar , n ) {
let res = 0 ;
ar . sort ( ) ;
for ( let i = 0 ; i < n ; i ++ ) { let count = 1 ;
for ( ; i < n - 1 ; i ++ ) { if ( ar [ i ] == ar [ i + 1 ] ) count ++ ; else break ; }
res = Math . max ( res , count ) ; } return res ; }
let arr = [ 5 , 6 , 9 , 3 , 4 , 3 , 4 ] ; let n = 7 ; document . write ( subset ( arr , n ) ) ;
var psquare = [ ]
function calcPsquare ( N ) { var i ; for ( i = 1 ; i * i <= N ; i ++ ) psquare . push ( i * i ) ; }
function countWays ( index , target ) {
if ( target == 0 ) return 1 ; if ( index < 0 target < 0 ) return 0 ;
var inc = countWays ( index , target - psquare [ index ] ) ;
var exc = countWays ( index - 1 , target ) ;
return inc + exc ; }
var N = 9 ;
calcPsquare ( N ) ;
document . write ( countWays ( psquare . length - 1 , N ) ) ;
class Node {
constructor ( data ) { this . data = data ; this . size = 0 ; this . left = this . right = null ; } }
function sumofsubtree ( root ) {
let p = new pair ( 1 , 0 ) ;
if ( root . left != null ) { let ptemp = sumofsubtree ( root . left ) ; p . second += ptemp . first + ptemp . second ; p . first += ptemp . first ; }
if ( root . right != null ) { let ptemp = sumofsubtree ( root . right ) ; p . second += ptemp . first + ptemp . second ; p . first += ptemp . first ; }
root . size = p . first ; return p ; }
function distance ( root , target , distancesum , n ) {
if ( root . data == target ) { sum = distancesum ; }
if ( root . left != null ) {
let tempsum = distancesum - root . left . size + ( n - root . left . size ) ;
distance ( root . left , target , tempsum , n ) ; }
if ( root . right != null ) {
let tempsum = distancesum - root . right . size + ( n - root . right . size ) ;
distance ( root . right , target , tempsum , n ) ; } }
let root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . left . left = new Node ( 4 ) ; root . left . right = new Node ( 5 ) ; root . right . left = new Node ( 6 ) ; root . right . right = new Node ( 7 ) ; root . left . left . left = new Node ( 8 ) ; root . left . left . right = new Node ( 9 ) ; let target = 3 ; let p = sumofsubtree ( root ) ;
let totalnodes = p . first ; distance ( root , target , p . second , totalnodes ) ;
document . write ( sum + " " ) ;
function rearrangeArray ( A , B , N , K ) {
B . sort ( ) ; B = reverse ( B ) ; let flag = true ; for ( let i = 0 ; i < N ; i ++ ) {
if ( A [ i ] + B [ i ] > K ) { flag = false ; break ; } } if ( ! flag ) { document . write ( " " + " " ) ; } else {
for ( let i = 0 ; i < N ; i ++ ) { document . write ( B [ i ] + " " ) ; } } }
let A = [ 1 , 2 , 3 , 4 , 2 ] ; let B = [ 1 , 2 , 3 , 1 , 1 ] ; let N = A . length ; let K = 5 ; rearrangeArray ( A , B , N , K ) ;
function countRows ( mat ) {
var count = 0 ;
var totalSum = 0 ;
for ( var i = 0 ; i < N ; i ++ ) { for ( var j = 0 ; j < M ; j ++ ) { totalSum += mat [ i ] [ j ] ; } }
for ( var i = 0 ; i < N ; i ++ ) {
var currSum = 0 ;
for ( var j = 0 ; j < M ; j ++ ) { currSum += mat [ i ] [ j ] ; }
if ( currSum > totalSum - currSum )
count ++ ; }
document . write ( count ) ; }
var mat = [ [ 2 , - 1 , 5 ] , [ - 3 , 0 , - 2 ] , [ 5 , 1 , 2 ] ] ;
countRows ( mat ) ;
function areElementsContiguous ( arr , n ) {
arr . sort ( function ( a , b ) { return a - b } ) ;
for ( let i = 1 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] > 1 ) return false ; return true ; }
let arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] ; let n = arr . length ; if ( areElementsContiguous ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function areElementsContiguous ( arr , n ) {
let max = Number . MIN_VALUE ; let min = Number . MAX_VALUE ; for ( let i = 0 ; i < n ; i ++ ) { max = Math . max ( max , arr [ i ] ) ; min = Math . min ( min , arr [ i ] ) ; } let m = max - min + 1 ;
if ( m > n ) return false ;
let visited = new Array ( n ) ; visited . fill ( false ) ;
for ( let i = 0 ; i < n ; i ++ ) visited [ arr [ i ] - min ] = true ;
for ( let i = 0 ; i < m ; i ++ ) if ( visited [ i ] == false ) return false ; return true ; }
let arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] ; let n = arr . length ; if ( areElementsContiguous ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function areElementsContiguous ( arr , n ) {
var us = new Set ( ) ; for ( var i = 0 ; i < n ; i ++ ) us . add ( arr [ i ] ) ;
var count = 1 ;
var curr_ele = arr [ 0 ] - 1 ;
while ( us . has ( curr_ele ) ) {
count ++ ;
curr_ele -- ; }
curr_ele = arr [ 0 ] + 1 ;
while ( us . has ( curr_ele ) ) {
count ++ ;
curr_ele ++ ; }
return ( count == ( us . size ) ) ; }
var arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] ; var n = arr . length ; if ( areElementsContiguous ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function longest ( a , n , k ) { var freq = Array ( 7 ) . fill ( 0 ) ; var start = 0 , end = 0 , now = 0 , l = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
freq [ a [ i ] ] ++ ;
if ( freq [ a [ i ] ] == 1 ) now ++ ;
while ( now > k ) {
freq [ a [ l ] ] -- ;
if ( freq [ a [ l ] ] == 0 ) now -- ;
l ++ ; }
if ( i - l + 1 >= end - start + 1 ) { end = i ; start = l ; } }
for ( var i = start ; i <= end ; i ++ ) document . write ( a [ i ] + " " ) ; }
var a = [ 6 , 5 , 1 , 2 , 3 , 2 , 1 , 4 , 5 ] ; var n = a . length ; var k = 3 ; longest ( a , n , k ) ;
function kOverlap ( pairs , k ) {
var vec = [ ] ; for ( var i = 0 ; i < pairs . length ; i ++ ) {
vec . push ( [ pairs [ i ] [ 0 ] , - 1 ] ) ; vec . push ( [ pairs [ i ] [ 1 ] , + 1 ] ) ; }
vec . sort ( ( a , b ) => { if ( a [ 0 ] != b [ 0 ] ) return a [ 0 ] - b [ 0 ] return a [ 1 ] - b [ 1 ] } ) ;
var st = [ ] ; for ( var i = 0 ; i < vec . length ; i ++ ) {
var cur = vec [ i ] ;
if ( cur [ 1 ] == - 1 ) {
st . push ( cur ) ; }
else {
st . pop ( ) ; }
if ( st . length >= k ) { return true ; } } return false ; }
var pairs = [ ] ; pairs . push ( [ 1 , 3 ] ) ; pairs . push ( [ 2 , 4 ] ) ; pairs . push ( [ 3 , 5 ] ) ; pairs . push ( [ 7 , 10 ] ) ; var n = pairs . length , k = 3 ; if ( kOverlap ( pairs , k ) ) document . write ( " " ) ; else document . write ( " " ) ;
let N = 5 ;
let ptr = new Array ( 501 ) ;
function findSmallestRange ( arr , n , k ) { let i , minval , maxval , minrange , minel = 0 , maxel = 0 , flag , minind ;
for ( i = 0 ; i <= k ; i ++ ) { ptr [ i ] = 0 ; } minrange = Number . MAX_VALUE ; while ( true ) {
minind = - 1 ; minval = Number . MAX_VALUE ; maxval = Number . MIN_VALUE ; flag = 0 ;
for ( i = 0 ; i < k ; i ++ ) {
if ( ptr [ i ] == n ) { flag = 1 ; break ; }
if ( ptr [ i ] < n && arr [ i ] [ ptr [ i ] ] < minval ) {
minind = i ; minval = arr [ i ] [ ptr [ i ] ] ; }
if ( ptr [ i ] < n && arr [ i ] [ ptr [ i ] ] > maxval ) { maxval = arr [ i ] [ ptr [ i ] ] ; } }
if ( flag == 1 ) { break ; } ptr [ minind ] ++ ;
if ( ( maxval - minval ) < minrange ) { minel = minval ; maxel = maxval ; minrange = maxel - minel ; } } document . write ( " " + minel + " " + maxel + " " ) ; }
let arr = [ [ 4 , 7 , 9 , 12 , 15 ] , [ 0 , 8 , 10 , 14 , 20 ] , [ 6 , 12 , 16 , 30 , 50 ] ] let k = arr . length ; findSmallestRange ( arr , N , k ) ;
function findLargestd ( S , n ) { let found = false ;
S . sort ( ) ;
for ( let i = n - 1 ; i >= 0 ; i -- ) { for ( let j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( let k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( let l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return Number . MAX_VALUE ; return - 1 ; }
let S = [ 2 , 3 , 5 , 7 , 12 ] ; let n = S . length ; let ans = findLargestd ( S , n ) ; if ( ans == Number . MAX_VALUE ) document . write ( " " ) ; else document . write ( " " + " " + ans ) ;
function findFourElements ( arr , n ) { let map = new Map ( ) ;
for ( let i = 0 ; i < n - 1 ; i ++ ) { for ( let j = i + 1 ; j < n ; j ++ ) { map . set ( arr [ i ] + arr [ j ] , new Indexes ( i , j ) ) ; } } let d = Number . MIN_VALUE ;
for ( let i = 0 ; i < n - 1 ; i ++ ) { for ( let j = i + 1 ; j < n ; j ++ ) { let abs_diff = Math . abs ( arr [ i ] - arr [ j ] ) ;
if ( map . has ( abs_diff ) ) { let indexes = map . get ( abs_diff ) ;
if ( indexes . getI ( ) != i && indexes . getI ( ) != j && indexes . getJ ( ) != i && indexes . getJ ( ) != j ) { d = Math . max ( d , Math . max ( arr [ i ] , arr [ j ] ) ) ; } } } } return d ; }
let arr = [ 2 , 3 , 5 , 7 , 12 ] ; let n = arr . length ; let res = findFourElements ( arr , n ) ; if ( res == Number . MIN_VALUE ) document . write ( " " ) ; else document . write ( res ) ;
function CountMaximum ( arr , n , k ) {
arr . sort ( function ( a , b ) { return a - b } ) ; let sum = 0 , count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
let arr = [ 30 , 30 , 10 , 10 ] ; let n = arr . length ; let k = 50 ;
document . write ( CountMaximum ( arr , n , k ) ) ;
function leftRotatebyOne ( arr , n ) { var i , temp ; temp = arr [ 0 ] ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
function leftRotate ( arr , d , n ) { for ( i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
function printArray ( arr , n ) { for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; }
var arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ;
function partSort ( arr , N , a , b ) {
let l = Math . min ( a , b ) ; let r = Math . max ( a , b ) ;
let temp = new Array ( r - l + 1 ) ; temp . fill ( 0 ) ; let j = 0 ; for ( let i = l ; i <= r ; i ++ ) { temp [ j ] = arr [ i ] ; j ++ ; }
temp . sort ( function ( a , b ) { return a - b } ) ;
j = 0 ; for ( let i = l ; i <= r ; i ++ ) { arr [ i ] = temp [ j ] ; j ++ ; }
for ( let i = 0 ; i < N ; i ++ ) { document . write ( arr [ i ] + " " ) ; } }
let arr = [ 7 , 8 , 4 , 5 , 2 ] ; let a = 1 , b = 4 ;
let N = arr . length ; partSort ( arr , N , a , b ) ;
let MAX_SIZE = 10 ;
function sortByRow ( mat , n , descending ) { let temp = 0 ; for ( let i = 0 ; i < n ; i ++ ) { if ( descending == true ) { let t = i ; for ( let p = 0 ; p < n ; p ++ ) { for ( let j = p + 1 ; j < n ; j ++ ) { if ( mat [ t ] [ p ] < mat [ t ] [ j ] ) { temp = mat [ t ] [ p ] ; mat [ t ] [ p ] = mat [ t ] [ j ] ; mat [ t ] [ j ] = temp ; } } } } else mat [ i ] . sort ( function ( a , b ) { return a - b ; } ) ; } }
function transpose ( mat , n ) { let temp = 0 ; for ( let i = 0 ; i < n ; i ++ ) { for ( let j = i + 1 ; j < n ; j ++ ) {
temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ j ] [ i ] ; mat [ j ] [ i ] = temp ; } } }
function sortMatRowAndColWise ( mat , n ) {
sortByRow ( mat , n , true ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n , false ) ;
transpose ( mat , n ) ; }
function printMat ( mat , n ) { for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) document . write ( mat [ i ] [ j ] + " " ) ; document . write ( " " ) ; } }
let n = 3 ; let mat = [ [ 3 , 2 , 1 ] , [ 9 , 8 , 7 ] , [ 6 , 5 , 4 ] ] ; document . write ( " " ) ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; document . write ( " " + " " ) ; printMat ( mat , n ) ;
function pushZerosToEnd ( arr , n ) {
let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
let arr = [ 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 ] ; let n = arr . length ; pushZerosToEnd ( arr , n ) ; document . write ( " " ) ; for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ;
function moveZerosToEnd ( arr , n ) {
let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 ) { temp = arr [ count ] ; arr [ count ] = arr [ i ] ; arr [ i ] = temp ; count = count + 1 ; } }
function printArray ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 0 , 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 ] ; let n = arr . length ; document . write ( " " ) ; printArray ( arr , n ) ; moveZerosToEnd ( arr , n ) ; document . write ( " " + " " ) ; printArray ( arr , n ) ;
function pushZerosToEnd ( arr , n ) {
var count = 0 ;
for ( var i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
function modifyAndRearrangeArr ( arr , n ) {
if ( n == 1 ) return ;
for ( var i = 0 ; i < n - 1 ; i ++ ) {
if ( arr [ i ] != 0 && arr [ i ] == arr [ i + 1 ] ) {
arr [ i ] = 2 * arr [ i ] ;
arr [ i + 1 ] = 0 ;
i ++ ; } }
pushZerosToEnd ( arr , n ) ; }
function printArray ( arr , n ) { for ( var i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; }
var arr = [ 0 , 2 , 2 , 2 , 0 , 6 , 6 , 0 , 0 , 8 ] ; var n = arr . length ; document . write ( " " ) ; printArray ( arr , n ) ; modifyAndRearrangeArr ( arr , n ) ; document . write ( " " ) ; document . write ( " " ) ; printArray ( arr , n ) ;
function shiftAllZeroToLeft ( array , n ) {
let lastSeenNonZero = 0 ; for ( let index = 0 ; index < n ; index ++ ) {
if ( array [ index ] != 0 ) {
swap ( array , array [ index ] , array [ lastSeenNonZero ] ) ;
lastSeenNonZero ++ ; } } } }
function printArray ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; }
function RearrangePosNeg ( arr , n ) { let key , j ; for ( let i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
let arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] ; let n = arr . length ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ;
function printArray ( A , size ) { for ( let i = 0 ; i < size ; i ++ ) document . write ( A [ i ] + " " ) ; document . write ( " " ) ; }
function reverse ( arr , l , r ) { if ( l < r ) { arr = swap ( arr , l , r ) ; reverse ( arr , ++ l , -- r ) ; } }
function merge ( arr , l , m , r ) {
let i = l ;
let j = m + 1 ; while ( i <= m && arr [ i ] < 0 ) i ++ ;
while ( j <= r && arr [ j ] < 0 ) j ++ ;
reverse ( arr , i , m ) ;
reverse ( arr , m + 1 , j - 1 ) ;
reverse ( arr , i , j - 1 ) ; }
function RearrangePosNeg ( arr , l , r ) { if ( l < r ) {
let m = l + Math . floor ( ( r - l ) / 2 ) ;
RearrangePosNeg ( arr , l , m ) ; RearrangePosNeg ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } } function swap ( arr , i , j ) { let temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; return arr ; }
let arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] ; let arr_size = arr . length ; RearrangePosNeg ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ;
function RearrangePosNeg ( arr ) { var i = 0 ; var j = arr . length - 1 ; while ( true ) {
while ( arr [ i ] < 0 && i < arr . length ) i ++ ;
while ( arr [ j ] > 0 && j >= 0 ) j -- ;
if ( i < j ) { var temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } else break ; } }
var arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] ; RearrangePosNeg ( arr ) ; for ( i = 0 ; i < arr . length ; i ++ ) document . write ( arr [ i ] + " " ) ;
function winner ( arr , N ) {
if ( N % 2 === 1 ) { document . write ( " " ) ; }
else { document . write ( " " ) ; } }
var arr = [ 24 , 45 , 45 , 24 ] ;
var N = arr . length ; winner ( arr , N ) ;
function findElements ( arr , n ) {
for ( let i = 0 ; i < n ; i ++ ) { let count = 0 ; for ( let j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) document . write ( arr [ i ] + " " ) ; } }
let arr = [ 2 , - 6 , 3 , 5 , 1 ] ; let n = arr . length ; findElements ( arr , n ) ;
function findElements ( arr , n ) { arr . sort ( ) ; for ( let i = 0 ; i < n - 2 ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 2 , - 6 , 3 , 5 , 1 ] ; let n = arr . length ; findElements ( arr , n ) ;
function findElements ( arr , n ) { let first = Number . MIN_VALUE ; let second = Number . MAX_VALUE ; for ( let i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( let i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 2 , - 6 , 3 , 5 , 1 ] ; let n = arr . length ; findElements ( arr , n ) ;
function getMinOps ( arr ) {
var res = 0 ; for ( i = 0 ; i < arr . length - 1 ; i ++ ) {
res += Math . max ( arr [ i + 1 ] - arr [ i ] , 0 ) ; }
return res ; }
var arr = [ 1 , 3 , 4 , 1 , 2 ] ; document . write ( getMinOps ( arr ) ) ;
function findFirstMissing ( array , start , end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; let mid = parseInt ( ( start + end ) / 2 , 10 ) ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
let arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ] ; let n = arr . length ; document . write ( " " + findFirstMissing ( arr , 0 , n - 1 ) ) ;
function findFirstMissing ( arr , start , end , first ) { if ( start < end ) { let mid = ( start + end ) / 2 ;
if ( arr [ mid ] != mid + first ) return findFirstMissing ( arr , start , mid , first ) ; else return findFirstMissing ( arr , mid + 1 , end , first ) ; } return start + first ; }
function findSmallestMissinginSortedArray ( arr ) {
if ( arr [ 0 ] != 0 ) return 0 ;
if ( arr [ arr . length - 1 ] == arr . length - 1 ) return arr . length ; let first = arr [ 0 ] ; return findFirstMissing ( arr , 0 , arr . length - 1 , first ) ; }
let arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 7 ] ; let n = arr . length ;
document . write ( " " + findSmallestMissinginSortedArray ( arr ) ) ;
function FindMaxSum ( arr , n ) { let incl = arr [ 0 ] ; let excl = 0 ; let excl_new ; let i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
let arr = [ 5 , 5 , 10 , 100 , 10 , 5 ] ; document . write ( FindMaxSum ( arr , arr . length ) ) ;
function countChanges ( matrix , n , m ) {
var dist = n + m - 1 ;
var freq = Array . from ( Array ( dist ) , ( ) => Array ( 10 ) ) ;
for ( var i = 0 ; i < n ; i ++ ) { for ( var j = 0 ; j < m ; j ++ ) {
freq [ i + j ] [ matrix [ i ] [ j ] ] ++ ; } } var min_changes_sum = 0 ; for ( var i = 0 ; i < parseInt ( dist / 2 ) ; i ++ ) { var maximum = 0 ; var total_values = 0 ;
for ( var j = 0 ; j < 10 ; j ++ ) { maximum = Math . max ( maximum , freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) ; total_values += ( freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) ; }
min_changes_sum += ( total_values - maximum ) ; }
return min_changes_sum ; }
var mat = [ [ 1 , 2 ] , [ 3 , 5 ] ] ;
document . write ( countChanges ( mat , 2 , 2 ) ) ;
var MAX = 500 ;
function buildSparseTable ( arr , n ) {
for ( var i = 0 ; i < n ; i ++ ) lookup [ i ] [ 0 ] = arr [ i ] ;
for ( var j = 1 ; ( 1 << j ) <= n ; j ++ ) {
for ( var i = 0 ; ( i + ( 1 << j ) - 1 ) < n ; i ++ ) {
if ( lookup [ i ] [ j - 1 ] < lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ) lookup [ i ] [ j ] = lookup [ i ] [ j - 1 ] ; else lookup [ i ] [ j ] = lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ; } } }
function query ( L , R ) {
var j = parseInt ( Math . log2 ( R - L + 1 ) ) ;
if ( lookup [ L ] [ j ] <= lookup [ R - ( 1 << j ) + 1 ] [ j ] ) return lookup [ L ] [ j ] ; else return lookup [ R - ( 1 << j ) + 1 ] [ j ] ; }
var a = [ 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 ] ; var n = a . length ; buildSparseTable ( a , n ) ; document . write ( query ( 0 , 4 ) + " " ) ; document . write ( query ( 4 , 7 ) + " " ) ; document . write ( query ( 7 , 8 ) ) ;
function minimizeWithKSwaps ( arr , n , k ) { for ( let i = 0 ; i < n - 1 && k > 0 ; ++ i ) {
let pos = i ; for ( let j = i + 1 ; j < n ; ++ j ) {
if ( j - i > k ) break ;
if ( arr [ j ] < arr [ pos ] ) pos = j ; }
let temp ; for ( let j = pos ; j > i ; -- j ) { temp = arr [ j ] ; arr [ j ] = arr [ j - 1 ] ; arr [ j - 1 ] = temp ; }
k -= pos - i ; } }
let arr = [ 7 , 6 , 9 , 2 , 1 ] ; let n = arr . length ; let k = 3 ;
minimizeWithKSwaps ( arr , n , k ) ;
document . write ( " " ) ; for ( let i = 0 ; i < n ; ++ i ) document . write ( arr [ i ] + " " ) ;
function findMaxAverage ( arr , n , k ) {
if ( k > n ) return - 1 ;
let csum = new Array ( n ) ; csum [ 0 ] = arr [ 0 ] ; for ( let i = 1 ; i < n ; i ++ ) csum [ i ] = csum [ i - 1 ] + arr [ i ] ;
let max_sum = csum [ k - 1 ] , max_end = k - 1 ;
for ( let i = k ; i < n ; i ++ ) { let curr_sum = csum [ i ] - csum [ i - k ] ; if ( curr_sum > max_sum ) { max_sum = curr_sum ; max_end = i ; } }
return max_end - k + 1 ; }
let arr = [ 1 , 12 , - 5 , - 6 , 50 , 3 ] ; let k = 4 ; let n = arr . length ; document . write ( " " + " " + k + " " + findMaxAverage ( arr , n , k ) ) ;
function findMaxAverage ( arr , n , k ) {
if ( k > n ) return - 1 ;
let sum = arr [ 0 ] ; for ( let i = 1 ; i < k ; i ++ ) sum += arr [ i ] ; let max_sum = sum ; let max_end = k - 1 ;
for ( let i = k ; i < n ; i ++ ) { sum = sum + arr [ i ] - arr [ i - k ] ; if ( sum > max_sum ) { max_sum = sum ; max_end = i ; } }
return max_end - k + 1 ; }
let arr = [ 1 , 12 , - 5 , - 6 , 50 , 3 ] ; let k = 4 ; let n = arr . length ; document . write ( " " + " " + k + " " + findMaxAverage ( arr , n , k ) ) ;
let m = new Map ( ) ;
function findMinimum ( arr , N , pos , turn ) {
let x = [ pos , turn ] ; if ( m . has ( x ) ) { return m [ x ] ; }
if ( pos >= N - 1 ) { return 0 ; }
if ( turn == 0 ) {
let ans = Math . min ( findMinimum ( arr , N , pos + 1 , 1 ) + arr [ pos ] , findMinimum ( arr , N , pos + 2 , 1 ) + arr [ pos ] + arr [ pos + 1 ] ) ;
let v = [ pos , turn ] ; m [ v ] = ans ;
return ans ; }
if ( turn != 0 ) {
let ans = Math . min ( findMinimum ( arr , N , pos + 1 , 0 ) , findMinimum ( arr , N , pos + 2 , 0 ) ) ;
let v = [ pos , turn ] ; m [ v ] = ans ;
return ans ; } return 0 ; }
function countPenality ( arr , N ) {
let pos = 0 ;
let turn = 0 ;
return findMinimum ( arr , N , pos , turn ) + 1 ; }
function printAnswer ( arr , N ) {
let a = countPenality ( arr , N ) ;
let sum = 0 ; for ( let i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
document . write ( a ) ; }
let arr = [ 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 ] ; let N = 8 ; printAnswer ( arr , N ) ;
let MAX = 1000001 ; let prime = new Array ( MAX ) ;
function SieveOfEratosthenes ( ) {
prime . fill ( 1 ) ; for ( let p = 2 ; p * p <= MAX ; p ++ ) {
if ( prime [ p ] == 1 ) {
for ( let i = p * p ; i <= MAX - 1 ; i += p ) prime [ i ] = 0 ; } } }
function getMid ( s , e ) { return s + parseInt ( ( e - s ) / 2 , 10 ) ; }
function getSumUtil ( st , ss , se , qs , qe , si ) {
if ( qs <= ss && qe >= se ) return st [ si ] ;
if ( se < qs ss > qe ) return 0 ;
let mid = getMid ( ss , se ) ; return getSumUtil ( st , ss , mid , qs , qe , 2 * si + 1 ) + getSumUtil ( st , mid + 1 , se , qs , qe , 2 * si + 2 ) ; }
function updateValueUtil ( st , ss , se , i , diff , si ) {
if ( i < ss i > se ) return ;
st [ si ] = st [ si ] + diff ; if ( se != ss ) { let mid = getMid ( ss , se ) ; updateValueUtil ( st , ss , mid , i , diff , 2 * si + 1 ) ; updateValueUtil ( st , mid + 1 , se , i , diff , 2 * si + 2 ) ; } }
function updateValue ( arr , st , n , i , new_val ) {
if ( i < 0 i > n - 1 ) { document . write ( " " ) ; return ; }
let diff = new_val - arr [ i ] ; let prev_val = arr [ i ] ;
arr [ i ] = new_val ;
if ( ( prime [ new_val ] prime [ prev_val ] ) != 0 ) {
if ( prime [ prev_val ] == 0 ) updateValueUtil ( st , 0 , n - 1 , i , new_val , 0 ) ;
else if ( prime [ new_val ] == 0 ) updateValueUtil ( st , 0 , n - 1 , i , - prev_val , 0 ) ;
else updateValueUtil ( st , 0 , n - 1 , i , diff , 0 ) ; } }
function getSum ( st , n , qs , qe ) {
if ( qs < 0 qe > n - 1 qs > qe ) { document . write ( " " ) ; return - 1 ; } return getSumUtil ( st , 0 , n - 1 , qs , qe , 0 ) ; }
function constructSTUtil ( arr , ss , se , st , si ) {
if ( ss == se ) {
if ( prime [ arr [ ss ] ] != 0 ) st [ si ] = arr [ ss ] ; else st [ si ] = 0 ; return st [ si ] ; }
let mid = getMid ( ss , se ) ; st [ si ] = constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) + constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ; return st [ si ] ; }
function constructST ( arr , n ) {
let x = parseInt ( ( Math . ceil ( Math . log ( n ) / Math . log ( 2 ) ) ) , 10 ) ;
let max_size = 2 * Math . pow ( 2 , x ) - 1 ;
let st = new Array ( max_size ) ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
let arr = [ 1 , 3 , 5 , 7 , 9 , 11 ] ; let n = arr . length ; let Q = [ [ 1 , 1 , 3 ] , [ 2 , 1 , 10 ] , [ 1 , 1 , 3 ] ] ;
SieveOfEratosthenes ( ) ;
let st = constructST ( arr , n ) ;
document . write ( getSum ( st , n , 1 , 3 ) + " " ) ;
updateValue ( arr , st , n , 1 , 10 ) ;
document . write ( getSum ( st , n , 1 , 3 ) + " " ) ;
let mod = 1000000007 ; let dp = new Array ( 1000 ) ; for ( let i = 0 ; i < 1000 ; i ++ ) { dp [ i ] = new Array ( 1000 ) ; } function calculate ( pos , prev , s , index ) {
if ( pos == s . length ) return 1 ;
if ( dp [ pos ] [ prev ] != - 1 ) return dp [ pos ] [ prev ] ;
let answer = 5 ; for ( let i = 0 ; i < index . length ; i ++ ) { if ( ( String . fromCharCode ( index [ i ] ) ) . localeCompare ( prev ) > 1 ) { answer = ( answer % mod + calculate ( pos + 1 , index [ i ] , s , index ) % mod ) % mod ; } }
dp [ pos ] [ prev ] = answer ; return dp [ pos ] [ prev ] ; } function countWays ( a , s ) { let n = a . length ;
let index = [ ] ; for ( let i = 0 ; i < 26 ; i ++ ) index . push ( [ ] ) ; for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < a [ i ] . length ; j ++ ) {
index [ a [ i ] [ j ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] . push ( j + 1 ) ; } }
for ( let i = 0 ; i < 1000 ; i ++ ) { for ( let j = 0 ; j < 1000 ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } return calculate ( 0 , 0 , s , index [ 0 ] ) ; }
let A = [ ] ; A . push ( " " ) ; A . push ( " " ) ; A . push ( " " ) ; let S = " " ; document . write ( countWays ( A , S ) ) ;
var MAX = 100005 ; var MOD = 1000000007 ;
function countNum ( idx , sum , tight , num , len , k ) { if ( len == idx ) { if ( sum == 0 ) return 1 ; else return 0 ; } if ( dp [ idx ] [ sum ] [ tight ] != - 1 ) return dp [ idx ] [ sum ] [ tight ] ; var res = 0 , limit ;
if ( tight == 0 ) { limit = num [ idx ] ; }
else { limit = 9 ; } for ( var i = 0 ; i <= limit ; i ++ ) {
var new_tight = tight ; if ( tight == 0 && i < limit ) new_tight = 1 ; res += countNum ( idx + 1 , ( sum + i ) % k , new_tight , num , len , k ) ; res %= MOD ; }
if ( res < 0 ) res += MOD ; return dp [ idx ] [ sum ] [ tight ] = res ; }
function process ( s ) { var num = [ ] ; for ( var i = 0 ; i < s . length ; i ++ ) { num . push ( s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ; } return num ; }
var n = " " ;
var len = n . length ; var k = 58 ;
var num = process ( n ) ; document . write ( countNum ( 0 , 0 , 0 , num , len , k ) ) ;
function maxWeight ( arr , n , w1_r , w2_r , i ) {
if ( i == n ) return 0 ; if ( dp [ i ] [ w1_r ] [ w2_r ] != - 1 ) return dp [ i ] [ w1_r ] [ w2_r ] ;
var fill_w1 = 0 , fill_w2 = 0 , fill_none = 0 ; if ( w1_r >= arr [ i ] ) fill_w1 = arr [ i ] + maxWeight ( arr , n , w1_r - arr [ i ] , w2_r , i + 1 ) ; if ( w2_r >= arr [ i ] ) fill_w2 = arr [ i ] + maxWeight ( arr , n , w1_r , w2_r - arr [ i ] , i + 1 ) ; fill_none = maxWeight ( arr , n , w1_r , w2_r , i + 1 ) ;
dp [ i ] [ w1_r ] [ w2_r ] = Math . max ( fill_none , Math . max ( fill_w1 , fill_w2 ) ) ; return dp [ i ] [ w1_r ] [ w2_r ] ; }
var arr = [ 8 , 2 , 3 ] ;
var n = arr . length ;
var w1 = 10 , w2 = 3 ;
document . write ( maxWeight ( arr , n , w1 , w2 , 0 ) ) ;
function CountWays ( n ) {
var noOfWays = Array ( 3 ) . fill ( 0 ) ; noOfWays [ 0 ] = 1 ; noOfWays [ 1 ] = 1 ; noOfWays [ 2 ] = 1 + 1 ;
for ( var i = 3 ; i < n + 1 ; i ++ ) { noOfWays [ i ] =
noOfWays [ 3 - 1 ]
+ noOfWays [ 3 - 3 ] ;
noOfWays [ 0 ] = noOfWays [ 1 ] ; noOfWays [ 1 ] = noOfWays [ 2 ] ; noOfWays [ 2 ] = noOfWays [ i ] ; } return noOfWays [ n ] ; }
var n = 5 ; document . write ( CountWays ( n ) ) ;
let MAX = 105 , q = 0 ; let prime = new Array ( MAX ) ; function sieve ( ) { for ( let i = 2 ; i * i < MAX ; i ++ ) { if ( prime [ i ] == 0 ) { for ( let j = i * i ; j < MAX ; j += i ) prime [ j ] = 1 ; } } }
function dfs ( i , j , k , n , m , mappedMatrix , mark , ans ) {
if ( ( mappedMatrix [ i ] [ j ] == 0 ? true : false ) || ( i > n ? true : false ) || ( j > m ? true : false ) || ( mark [ i ] [ j ] != 0 ? true : false ) || ( q != 0 ? true : false ) ) return ;
mark [ i ] [ j ] = 1 ;
ans [ k ] [ 0 ] = i ; ans [ k ] [ 1 ] = j ;
if ( i == n && j == m ) {
q = k ; return ; }
dfs ( i + 1 , j + 1 , k + 1 , n , m , mappedMatrix , mark , ans ) ;
dfs ( i + 1 , j , k + 1 , n , m , mappedMatrix , mark , ans ) ;
dfs ( i , j + 1 , k + 1 , n , m , mappedMatrix , mark , ans ) ; }
function lexicographicalPath ( n , m , mappedMatrix ) {
let ans = new Array ( MAX ) ;
let mark = new Array ( MAX ) ; for ( let i = 0 ; i < MAX ; i ++ ) { mark [ i ] = new Array ( MAX ) ; ans [ i ] = new Array ( 2 ) ; }
dfs ( 1 , 1 , 1 , n , m , mappedMatrix , mark , ans ) ; let anss = [ [ 1 , 1 ] , [ 2 , 1 ] , [ 3 , 2 ] , [ 3 , 3 ] ] ;
for ( let i = 0 ; i < 4 ; i ++ ) { document . write ( anss [ i ] [ 0 ] + " " + anss [ i ] [ 1 ] + " " ) ; } }
function countPrimePath ( mappedMatrix , n , m ) { let dp = new Array ( MAX ) ; for ( let i = 0 ; i < MAX ; i ++ ) { dp [ i ] = new Array ( MAX ) ; for ( let j = 0 ; j < MAX ; j ++ ) { dp [ i ] [ j ] = 0 ; } } dp [ 1 ] [ 1 ] = 1 ;
for ( let i = 1 ; i <= n ; i ++ ) { for ( let j = 1 ; j <= m ; j ++ ) {
if ( i == 1 && j == 1 ) continue ; dp [ i ] [ j ] = ( dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] + dp [ i - 1 ] [ j - 1 ] ) ;
if ( mappedMatrix [ i ] [ j ] == 0 ) dp [ i ] [ j ] = 0 ; } } dp [ n ] [ m ] = 4 ; document . write ( dp [ n ] [ m ] + " " ) ; }
function preprocessMatrix ( mappedMatrix , a , n , m ) {
sieve ( ) ; for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < m ; j ++ ) {
if ( prime [ a [ i ] [ j ] ] == 0 ) mappedMatrix [ i + 1 ] [ j + 1 ] = 1 ;
else mappedMatrix [ i + 1 ] [ j + 1 ] = 0 ; } } }
let n = 3 ; let m = 3 ; let a = [ [ 2 , 3 , 7 ] , [ 5 , 4 , 2 ] , [ 3 , 7 , 11 ] ] ; let mappedMatrix = new Array ( MAX ) ; for ( let i = 0 ; i < MAX ; i ++ ) { mappedMatrix [ i ] = new Array ( MAX ) ; for ( let j = 0 ; j < MAX ; j ++ ) { mappedMatrix [ i ] [ j ] = 0 ; } } preprocessMatrix ( mappedMatrix , a , n , m ) ; countPrimePath ( mappedMatrix , n , m ) ; lexicographicalPath ( n , m , mappedMatrix ) ;
function isSubsetSum ( set , n , sum ) {
let subset = new Array ( sum + 1 ) ; for ( var i = 0 ; i < subset . length ; i ++ ) { subset [ i ] = new Array ( 2 ) ; } let count = new Array ( sum + 1 ) ; for ( var i = 0 ; i < count . length ; i ++ ) { count [ i ] = new Array ( 2 ) ; }
for ( let i = 0 ; i <= n ; i ++ ) { subset [ 0 ] [ i ] = true ; count [ 0 ] [ i ] = 0 ; }
for ( let i = 1 ; i <= sum ; i ++ ) { subset [ i ] [ 0 ] = false ; count [ i ] [ 0 ] = - 1 ; }
for ( let i = 1 ; i <= sum ; i ++ ) { for ( let j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; count [ i ] [ j ] = count [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) { subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; if ( subset [ i ] [ j ] ) count [ i ] [ j ] = Math . max ( count [ i ] [ j - 1 ] , count [ i - set [ j - 1 ] ] [ j - 1 ] + 1 ) ; } } } return count [ sum ] [ n ] ; }
let set = [ 2 , 3 , 5 , 10 ] ; let sum = 20 ; let n = set . length ; document . write ( isSubsetSum ( set , n , sum ) ) ;
let MAX = 100 ;
let dp = new Array ( MAX ) ;
function lcs ( str1 , str2 , len1 , len2 , i , j ) { let ret = dp [ i ] [ j ] ;
if ( i == len1 j == len2 ) return ret = 0 ;
if ( ret != - 1 ) return ret ; ret = 0 ;
if ( str1 [ i ] == str2 [ j ] ) ret = 1 + lcs ( str1 , str2 , len1 , len2 , i + 1 , j + 1 ) ; else ret = Math . max ( lcs ( str1 , str2 , len1 , len2 , i + 1 , j ) , lcs ( str1 , str2 , len1 , len2 , i , j + 1 ) ) ; return ret ; }
function printAll ( str1 , str2 , len1 , len2 , data , indx1 , indx2 , currlcs ) {
if ( currlcs == lcslen ) { data [ currlcs ] = null ; document . write ( data . join ( " " ) + " " ) ; return ; }
if ( indx1 == len1 indx2 == len2 ) return ;
for ( let ch = ' ' . charCodeAt ( 0 ) ; ch <= ' ' . charCodeAt ( 0 ) ; ch ++ ) {
let done = false ; for ( let i = indx1 ; i < len1 ; i ++ ) {
if ( ch == str1 [ i ] . charCodeAt ( 0 ) ) { for ( let j = indx2 ; j < len2 ; j ++ ) {
if ( ch == str2 [ j ] . charCodeAt ( 0 ) && lcs ( str1 , str2 , len1 , len2 , i , j ) == lcslen - currlcs ) { data [ currlcs ] = String . fromCharCode ( ch ) ; printAll ( str1 , str2 , len1 , len2 , data , i + 1 , j + 1 , currlcs + 1 ) ; done = true ; break ; } } }
if ( done ) break ; } } }
function prinlAllLCSSorted ( str1 , str2 ) {
let len1 = str1 . length , len2 = str2 . length ;
for ( let i = 0 ; i < MAX ; i ++ ) { dp [ i ] = new Array ( MAX ) ; for ( let j = 0 ; j < MAX ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } lcslen = lcs ( str1 , str2 , len1 , len2 , 0 , 0 ) ;
let data = new Array ( MAX ) ; printAll ( str1 , str2 , len1 , len2 , data , 0 , 0 , 0 ) ; }
let str1 = " " , str2 = " " ; prinlAllLCSSorted ( str1 , str2 ) ;
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
function isMajorityElement ( arr , n , key ) { if ( arr [ parseInt ( n / 2 , 10 ) ] == key ) return true ; else return false ; }
let arr = [ 1 , 2 , 3 , 3 , 3 , 3 , 10 ] ; let n = arr . length ; let x = 3 ; if ( isMajorityElement ( arr , n , x ) ) document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ; else document . write ( x + " " + " " + parseInt ( n / 2 , 10 ) + " " ) ;
function cutRod ( price , n ) { let val = new Array ( n + 1 ) ; val [ 0 ] = 0 ;
for ( let i = 1 ; i <= n ; i ++ ) { let max_val = Number . MIN_VALUE ; for ( let j = 0 ; j < i ; j ++ ) max_val = Math . max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
let arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] ; let size = arr . length ; document . write ( " " + cutRod ( arr , size ) + " " ) ;
function isPossible ( target ) {
var max = 0 ;
var index = 0 ;
for ( i = 0 ; i < target . length ; i ++ ) {
if ( max < target [ i ] ) { max = target [ i ] ; index = i ; } }
if ( max == 1 ) return true ;
for ( i = 0 ; i < target . length ; i ++ ) {
if ( i != index ) {
max -= target [ i ] ;
if ( max <= 0 ) return false ; } }
target [ index ] = max ;
return isPossible ( target ) ; }
var target = [ 9 , 3 , 5 ] ; res = isPossible ( target ) ; if ( res ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function nCr ( n , r ) {
let res = 1 ;
if ( r > n - r ) r = n - r ;
for ( let i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
let n = 3 , m = 2 , k = 2 ; document . write ( nCr ( n + m , k ) ) ;
function Is_possible ( N ) { let C = 0 ; let D = 0 ;
while ( N % 10 == 0 ) { N = N / 10 ; C += 1 ; }
if ( Math . pow ( 2 , ( Math . log ( N ) / ( Math . log ( 2 ) ) ) ) == N ) { D = ( Math . log ( N ) / ( Math . log ( 2 ) ) ) ;
if ( C >= D ) document . write ( " " ) ; else document . write ( " " ) ; } else document . write ( " " ) ; }
let N = 2000000000000 ; Is_possible ( N ) ;
function findNthTerm ( n ) { document . write ( n * n - n + 1 ) ; }
N = 4 ; findNthTerm ( N ) ;
function rev ( num ) { var rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = Math . floor ( num / 10 ) ; }
return rev_num ; }
function divSum ( num ) {
var result = 0 ;
for ( var i = 2 ; i <= Math . floor ( Math . sqrt ( num ) ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += rev ( i ) ; else result += ( rev ( i ) + rev ( num / i ) ) ; } }
result += 1 ; return result ; }
function isAntiPerfect ( n ) { return divSum ( n ) == n ; }
var N = 244 ;
if ( isAntiPerfect ( N ) ) document . write ( " " ) ; else document . write ( " " ) ;
function printSeries ( n , a , b , c ) { let d ;
if ( n == 1 ) { document . write ( a + " " ) ; return ; } if ( n == 2 ) { document . write ( a + " " + b + " " ) ; return ; } document . write ( a + " " + b + " " + c + " " ) ; for ( let i = 4 ; i <= n ; i ++ ) { d = a + b + c ; document . write ( d + " " ) ; a = b ; b = c ; c = d ; } }
let N = 7 , a = 1 , b = 3 ; let c = 4 ;
printSeries ( N , a , b , c ) ;
function diameter ( n ) {
var L , H , templen ; L = 1 ;
H = 0 ;
if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 2 ; } if ( n == 3 ) { return 3 ; }
while ( L * 2 <= n ) { L *= 2 ; H ++ ; }
if ( n >= L * 2 - 1 ) return 2 * H + 1 ; else if ( n >= L + ( L / 2 ) - 1 ) return 2 * H ; return 2 * H - 1 ; }
var n = 15 ; document . write ( diameter ( n ) ) ;
function compareValues ( a , b , c , d ) {
let log1 = Math . log ( a ) / Math . log ( 10 ) ; let num1 = log1 * b ;
let log2 = Math . log ( c ) / Math . log ( 10 ) ; let num2 = log2 * d ;
if ( num1 > num2 ) document . write ( a + " " + b ) ; else document . write ( c + " " + d ) ; }
let a = 8 , b = 29 , c = 60 , d = 59 ; compareValues ( a , b , c , d ) ;
const MAX = 100005 ;
function addPrimes ( ) { let n = MAX ; let prime = new Array ( n + 1 ) . fill ( true ) ; for ( let p = 2 ; p * p <= n ; p ++ ) { if ( prime [ p ] == true ) { for ( let i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } let ans = [ ] ;
for ( let p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) ans . push ( p ) ; return ans ; }
function is_prime ( n ) { return ( n == 3 n == 5 n == 7 ) ; }
function find_Sum ( n ) {
let sum = 0 ;
let v = addPrimes ( ) ;
for ( let i = 0 ; i < v . length && n > 0 ; i ++ ) {
let flag = 1 ; let a = v [ i ] ;
while ( a != 0 ) { let d = a % 10 ; a = parseInt ( a / 10 ) ; if ( is_prime ( d ) ) { flag = 0 ; break ; } }
if ( flag == 1 ) { n -- ; sum = sum + v [ i ] ; } }
return sum ; }
let n = 7 ;
document . write ( find_Sum ( n ) ) ;
function primeCount ( arr , n ) {
let max_val = Math . max ( ... arr ) ;
let prime = new Array ( max_val + 1 ) . fill ( true ) ;
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( let p = 2 ; p * p <= max_val ; p ++ ) {
if ( prime [ p ] == true ) {
for ( let i = p * 2 ; i <= max_val ; i += p ) prime [ i ] = false ; } }
let count = 0 ; for ( let i = 0 ; i < n ; i ++ ) if ( prime [ arr [ i ] ] ) count ++ ; return count ; }
function getPrefixArray ( arr , n , pre ) {
pre [ 0 ] = arr [ 0 ] ; for ( let i = 1 ; i < n ; i ++ ) { pre [ i ] = pre [ i - 1 ] + arr [ i ] ; } }
let arr = [ 1 , 4 , 8 , 4 ] ; let n = arr . length ;
let pre = new Array ( n ) ; getPrefixArray ( arr , n , pre ) ;
document . write ( primeCount ( pre , n ) ) ;
function minValue ( n , x , y ) {
let val = ( y * n ) / 100 ;
if ( x >= val ) return 0 ; else return ( Math . ceil ( val ) - x ) ; }
let n = 10 , x = 2 , y = 40 ; document . write ( minValue ( n , x , y ) ) ;
function isPrime ( n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( let i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
function isFactorialPrime ( n ) {
if ( ! isPrime ( n ) ) return false ; let fact = 1 ; let i = 1 ; while ( fact <= n + 1 ) {
fact = fact * i ;
if ( n + 1 == fact n - 1 == fact ) return true ; i ++ ; }
return false ; }
let n = 23 ; if ( isFactorialPrime ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
let n = 5 ;
let fac1 = 1 ; for ( let i = 2 ; i <= n - 1 ; i ++ ) fac1 = fac1 * i ;
fac2 = fac1 * n ;
totalWays = fac1 * fac2 ;
document . write ( totalWays + " " ) ;
var MAX = 10000 ; var arr = [ ] ;
function SieveOfEratosthenes ( ) {
var prime = Array ( MAX ) . fill ( true ) ; ; for ( var p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( var i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
for ( var p = 2 ; p < MAX ; p ++ ) if ( prime [ p ] ) arr . push ( p ) ; }
function isEuclid ( n ) { var product = 1 ; var i = 0 ; while ( product < n ) {
product = product * arr [ i ] ; if ( product + 1 == n ) return true ; i ++ ; } return false ; }
SieveOfEratosthenes ( ) ;
var n = 31 ;
if ( isEuclid ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
n = 42 ;
if ( isEuclid ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function nextPerfectCube ( N ) { let nextN = Math . floor ( Math . cbrt ( N ) ) + 1 ; return nextN * nextN * nextN ; }
let n = 35 ; document . write ( nextPerfectCube ( n ) ) ;
function isPrime ( n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( let i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
function SumOfPrimeDivisors ( n ) { let sum = 0 ;
let root_n = parseInt ( Math . sqrt ( n ) , 10 ) ; for ( let i = 1 ; i <= root_n ; i ++ ) { if ( n % i == 0 ) {
if ( i == parseInt ( n / i , 10 ) && isPrime ( i ) ) { sum += i ; } else {
if ( isPrime ( i ) ) { sum += i ; } if ( isPrime ( parseInt ( n / i , 10 ) ) ) { sum += ( parseInt ( n / i , 10 ) ) ; } } } } return sum ; }
let n = 60 ; document . write ( " " + SumOfPrimeDivisors ( n ) + " " ) ;
function findpos ( n ) { var pos = 0 ; for ( i = 0 ; i < n . length ; i ++ ) { switch ( n . charAt ( i ) ) {
case ' ' : pos = pos * 4 + 1 ; break ;
case ' ' : pos = pos * 4 + 2 ; break ;
case ' ' : pos = pos * 4 + 3 ; break ;
case ' ' : pos = pos * 4 + 4 ; break ; } } return pos ; }
var n = " " ; document . write ( findpos ( n ) ) ;
function possibleTripletInRange ( L , R ) { let flag = false ; let possibleA , possibleB , possibleC ; let numbersInRange = ( R - L + 1 ) ;
if ( numbersInRange < 3 ) { flag = false ; }
else if ( numbersInRange > 3 ) { flag = true ;
if ( L % 2 ) { L ++ ; } possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
if ( ! ( L % 2 ) ) { flag = true ; possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
flag = false ; } }
if ( flag == true ) { document . write ( " " + possibleA + " " + possibleB + " " + possibleC + " " + " " + L + " " + R + " " ) ; } else { document . write ( " " + L + " " + R + " " ) ; } }
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ;
const mod = 1000000007 ;
function digitNumber ( n ) {
if ( n == 0 ) return 1 ;
if ( n == 1 ) return 9 ;
if ( n % 2 ) {
let temp = digitNumber ( ( n - 1 ) / 2 ) % mod ; return ( 9 * ( temp * temp ) % mod ) % mod ; } else {
let temp = digitNumber ( n / 2 ) % mod ; return ( temp * temp ) % mod ; } } function countExcluding ( n , d ) {
if ( d == 0 ) return ( 9 * digitNumber ( n - 1 ) ) % mod ; else return ( 8 * digitNumber ( n - 1 ) ) % mod ; }
let d = 9 ; let n = 3 ; document . write ( countExcluding ( n , d ) + " " ) ;
function isPrime ( n ) {
if ( n <= 1 ) return false ;
for ( i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
function isEmirp ( n ) {
if ( isPrime ( n ) == false ) return false ;
var rev = 0 ; while ( n != 0 ) { var d = n % 10 ; rev = rev * 10 + d ; n = parseInt ( n / 10 ) ; }
return isPrime ( rev ) ; }
var n = 13 ; if ( isEmirp ( n ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function Convert ( radian ) { let pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
let radian = 5.0 ; let degree = Convert ( radian ) ; document . write ( degree ) ;
function sn ( n , an ) { return ( n * ( 1 + an ) ) / 2 ; }
function trace ( n , m ) {
let an = 1 + ( n - 1 ) * ( m + 1 ) ;
let rowmajorSum = sn ( n , an ) ;
an = 1 + ( n - 1 ) * ( n + 1 ) ;
let colmajorSum = sn ( n , an ) ; return rowmajorSum + colmajorSum ; }
let N = 3 , M = 3 ; document . write ( trace ( N , M ) ) ;
function max_area ( n , m , k ) { if ( k > ( n + m - 2 ) ) document . write ( " " ) ; else { let result ;
if ( k < Math . max ( m , n ) - 1 ) { result = Math . max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; }
else { result = Math . max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; }
document . write ( result ) ; } }
let n = 3 , m = 4 , k = 1 ; max_area ( n , m , k ) ;
function area_fun ( side ) { let area = side * side ; return area ; }
let side = 4 ; let area = area_fun ( side ) ; document . write ( area ) ;
function countConsecutive ( N ) {
let count = 0 ; for ( let L = 1 ; L * ( L + 1 ) < 2 * N ; L ++ ) { let a = ( ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) ) ; if ( a - parseInt ( a , 10 ) == 0.0 ) count ++ ; } return count ; }
let N = 15 ; document . write ( countConsecutive ( N ) + " " ) ; N = 10 ; document . write ( countConsecutive ( N ) ) ;
function isAutomorphic ( N ) {
let sq = N * N ;
while ( N > 0 ) {
if ( N % 10 != sq % 10 ) return - 1 ;
N /= 10 ; sq /= 10 ; } return 1 ; }
let N = 5 ; let geeks = isAutomorphic ( N ) ? " " : " " ; document . write ( geeks ) ;
function maxPrimefactorNum ( N ) {
let arr = new Array ( N + 5 ) ; arr . fill ( false ) ; let i ;
for ( i = 3 ; i * i <= N ; i += 2 ) { if ( ! arr [ i ] ) { for ( let j = i * i ; j <= N ; j += i ) { arr [ j ] = true ; } } }
let prime = [ ] ; prime . push ( 2 ) ; for ( i = 3 ; i <= N ; i += 2 ) { if ( ! arr [ i ] ) { prime . push ( i ) ; } }
let ans = 1 ; i = 0 ; while ( ans * prime [ i ] <= N && i < prime . length ) { ans *= prime [ i ] ; i ++ ; } return ans ; }
let N = 40 ; document . write ( maxPrimefactorNum ( N ) ) ;
function divSum ( num ) {
let result = 0 ;
for ( let i = 2 ; i <= Math . sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; }
let num = 36 ; document . write ( divSum ( num ) ) ;
function power ( x , y , p ) {
while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
function squareRoot ( n , p ) { if ( p % 4 != 3 ) { document . write ( " " ) ; return ; }
n = n % p ; let x = power ( n , Math . floor ( ( p + 1 ) / 4 ) , p ) ; if ( ( x * x ) % p == n ) { document . write ( " " + x ) ; return ; }
x = p - x ; if ( ( x * x ) % p == n ) { document . write ( " " + x ) ; return ; }
document . write ( " " ) ; }
let p = 7 ; let n = 2 ; squareRoot ( n , p ) ;
function power ( x , y , p ) {
let res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
function miillerTest ( d , n ) {
let a = 2 + Math . floor ( Math . random ( ) * ( n - 2 ) ) % ( n - 4 ) ;
let x = power ( a , d , n ) ; if ( x == 1 x == n - 1 ) return true ;
while ( d != n - 1 ) { x = ( x * x ) % n ; d *= 2 ; if ( x == 1 ) return false ; if ( x == n - 1 ) return true ; }
return false ; }
function isPrime ( n , k ) {
if ( n <= 1 n == 4 ) return false ; if ( n <= 3 ) return true ;
let d = n - 1 ; while ( d % 2 == 0 ) d /= 2 ;
for ( let i = 0 ; i < k ; i ++ ) if ( ! miillerTest ( d , n ) ) return false ; return true ; }
let k = 4 ; document . write ( " " ) ; for ( let n = 1 ; n < 100 ; n ++ ) if ( isPrime ( n , k ) ) document . write ( n , " " ) ;
function maxConsecutiveOnes ( x ) {
let count = 0 ;
while ( x != 0 ) {
x = ( x & ( x << 1 ) ) ; count ++ ; } return count ; }
document . write ( maxConsecutiveOnes ( 14 ) + " " ) ; document . write ( maxConsecutiveOnes ( 222 ) ) ;
function subtract ( x , y ) {
while ( y != 0 ) {
let borrow = ( ~ x ) & y ;
x = x ^ y ;
y = borrow << 1 ; } return x ; }
let x = 29 , y = 13 ; document . write ( " " + subtract ( x , y ) ) ;
function subtract ( x , y ) { if ( y == 0 ) return x ; return subtract ( x ^ y , ( ~ x & y ) << 1 ) ; }
var x = 29 , y = 13 ; document . write ( " " + subtract ( x , y ) ) ;
function addEdge ( v , x , y ) { v [ x ] . push ( y ) ; v [ y ] . push ( x ) ; }
function dfs ( tree , temp , ancestor , u , parent , k ) {
temp . push ( u ) ;
for ( let i = 0 ; i < tree [ u ] . length ; i ++ ) { if ( tree [ u ] [ i ] == parent ) continue ; dfs ( tree , temp , ancestor , tree [ u ] [ i ] , u , k ) ; } temp . pop ( ) ;
if ( temp . length < k ) { ancestor [ u ] = - 1 ; } else {
ancestor [ u ] = temp [ temp . length - k ] ; } }
function KthAncestor ( N , K , E , edges ) {
let tree = new Array ( N + 1 ) ; for ( let i = 0 ; i < tree . length ; i ++ ) tree [ i ] = [ ] ; for ( let i = 0 ; i < E ; i ++ ) { addEdge ( tree , edges [ i ] [ 0 ] , edges [ i ] [ 1 ] ) ; }
let temp = [ ] ;
let ancestor = new Array ( N + 1 ) ; dfs ( tree , temp , ancestor , 1 , 0 , K ) ;
for ( let i = 1 ; i <= N ; i ++ ) { document . write ( ancestor [ i ] + " " ) ; } }
let N = 9 ; let K = 2 ;
let E = 8 ; let edges = [ [ 1 , 2 ] , [ 1 , 3 ] , [ 2 , 4 ] , [ 2 , 5 ] , [ 2 , 6 ] , [ 3 , 7 ] , [ 3 , 8 ] , [ 3 , 9 ] ] ;
KthAncestor ( N , K , E , edges ) ;
function build ( sum , a , l , r , rt ) {
if ( l == r ) { sum [ rt ] = a [ l - 1 ] ; return ; }
let m = ( l + r ) >> 1 ;
build ( sum , a , l , m , rt << 1 ) ; build ( sum , a , m + 1 , r , rt << 1 1 ) ; }
function pushDown ( sum , add , rt , ln , rn ) { if ( add [ rt ] != 0 ) { add [ rt << 1 ] = add [ rt ] ; add [ rt << 1 1 ] = add [ rt ] ; sum [ rt << 1 ] = sum [ rt << 1 ] + add [ rt ] * ln ; sum [ rt << 1 1 ] = sum [ rt << 1 1 ] + add [ rt ] * rn ; add [ rt ] = 0 ; } }
function update ( sum , add , L , R , C , l , r , rt ) {
if ( L <= l && r <= R ) { sum [ rt ] = sum [ rt ] + C * ( r - l + 1 ) ; add [ rt ] = add [ rt ] + C ; return ; }
let m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ;
if ( L <= m ) { update ( sum , add , L , R , C , l , m , rt << 1 ) ; } if ( R > m ) { update ( sum , add , L , R , C , m + 1 , r , rt << 1 1 ) ; } }
function query ( sum , add , L , R , l , r , rt ) {
if ( L <= l && r <= R ) { return sum [ rt ] ; }
let m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ; let ans = 0 ;
if ( L <= m ) { ans += query ( sum , add , L , R , l , m , rt << 1 ) ; } if ( R > m ) { ans += query ( sum , add , L , R , m + 1 , r , rt << 1 1 ) ; }
return ans ; }
function sequenceMaintenance ( n , q , a , b , m ) {
a . sort ( function ( a , b ) { return a - b ; } ) ;
let sum = [ ] ; let ad = [ ] ; let ans = [ ] ; for ( let i = 0 ; i < ( n << 2 ) ; i ++ ) { sum . push ( 0 ) ; ad . push ( 0 ) ; }
build ( sum , a , 1 , n , 1 ) ;
for ( let i = 0 ; i < q ; i ++ ) { let l = 1 , r = n , pos = - 1 ; while ( l <= r ) { m = ( l + r ) >> 1 ; if ( query ( sum , ad , m , m , 1 , n , 1 ) >= b [ i ] ) { r = m - 1 ; pos = m ; } else { l = m + 1 ; } } if ( pos == - 1 ) { ans . push ( 0 ) ; } else {
ans . push ( n - pos + 1 ) ;
update ( sum , ad , pos , n , - m , 1 , n , 1 ) ; } }
for ( let i = 0 ; i < ans . length ; i ++ ) { document . write ( ans [ i ] + " " ) ; } }
let N = 4 ; let Q = 3 ; let M = 1 ; let arr = [ 1 , 2 , 3 , 4 ] ; let Query = [ 4 , 3 , 1 ] ;
sequenceMaintenance ( N , Q , arr , Query , M ) ;
function hasCoprimePair ( arr , n ) {
for ( i = 0 ; i < n - 1 ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) {
if ( ( __gcd ( arr [ i ] , arr [ j ] ) ) == 1 ) { return true ; } } }
return false ; }
var n = 3 ; var arr = [ 6 , 9 , 15 ] ;
if ( hasCoprimePair ( arr , n ) ) { document . write ( 1 + " " ) ; }
else { document . write ( n + " " ) ; }
function Numberofways ( n ) { var count = 0 ; for ( var a = 1 ; a < n ; a ++ ) { for ( var b = 1 ; b < n ; b ++ ) { var c = n - ( a + b ) ;
if ( a + b > c && a + c > b && b + c > a ) { count ++ ; } } }
return count ; }
var n = 15 ; document . write ( Numberofways ( n ) ) ;
function countPairs ( N , arr ) { let count = 0 ;
for ( let i = 0 ; i < N ; i ++ ) { if ( i == arr [ arr [ i ] - 1 ] - 1 ) {
count ++ ; } }
document . write ( count / 2 ) ; } let arr = [ 2 , 1 , 4 , 3 ] ; let N = arr . length ; countPairs ( N , arr ) ;
let arr = [ 2 , 1 , 4 , 3 ] ; let N = arr . length ; countPairs ( N , arr ) ;
function LongestFibSubseq ( A , n ) {
var S = new Set ( A ) ; var maxLen = 0 , x , y ; for ( var i = 0 ; i < n ; ++ i ) { for ( var j = i + 1 ; j < n ; ++ j ) { x = A [ j ] ; y = A [ i ] + A [ j ] ; var length = 2 ;
while ( S . has ( y ) ) {
var z = x + y ; x = y ; y = z ; maxLen = Math . max ( maxLen , ++ length ) ; } } } return maxLen >= 3 ? maxLen : 0 ; }
var A = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ] ; var n = A . length ; document . write ( LongestFibSubseq ( A , n ) ) ;
function CountMaximum ( arr , n , k ) {
arr . sort ( ) ; let sum = 0 , count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
let arr = [ 30 , 30 , 10 , 10 ] ; let n = 4 ; let k = 50 ;
document . write ( CountMaximum ( arr , n , k ) ) ;
function num_candyTypes ( candies ) {
let s = new Set ( ) ;
for ( let i = 0 ; i < candies . length ; i ++ ) { s . add ( candies [ i ] ) ; }
return s . size ; }
function distribute_candies ( candies ) {
let allowed = candies . length / 2 ;
let types = num_candyTypes ( candies ) ;
if ( types < allowed ) document . write ( types ) ; else document . write ( allowed ) ; }
let candies = [ 4 , 4 , 5 , 5 , 3 , 3 ] ;
distribute_candies ( candies ) ;
function Length_Diagonals ( a , theta ) { let p = a * Math . sqrt ( 2 + ( 2 * Math . cos ( theta * ( Math . PI / 180 ) ) ) ) ; let q = a * Math . sqrt ( 2 - ( 2 * Math . cos ( theta * ( Math . PI / 180 ) ) ) ) ; return [ p , q ] ; }
let A = 6 ; let theta = 45 ; let ans = Length_Diagonals ( A , theta ) ; document . write ( ans [ 0 ] . toFixed ( 2 ) + " " + ans [ 1 ] . toFixed ( 2 ) ) ;
function countEvenOdd ( arr , n , K ) { let even = 0 , odd = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
let x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } let y ;
y = __builtin_popcount ( K ) ;
if ( ( y & 1 ) != 0 ) { document . write ( " " + odd + " " + even ) ; }
else { document . write ( " " + even + " " + odd ) ; } }
let arr = [ 4 , 2 , 15 , 9 , 8 , 8 ] ; let K = 3 ; let n = arr . length ;
countEvenOdd ( arr , n , K ) ;
let N = 6 ; let Even = Math . floor ( N / 2 ) ; let Odd = N - Even ; document . write ( Even * Odd ) ;
function countTriplets ( A ) {
var cnt = 0 ;
var tuples = new Map ( ) ;
A . forEach ( a => {
A . forEach ( b => { if ( tuples . has ( a & b ) ) tuples . set ( a & b , tuples . get ( a & b ) + 1 ) else tuples . set ( a & b , 1 ) } ) ; } ) ;
A . forEach ( a => {
tuples . forEach ( ( value , key ) => {
if ( ( key & a ) == 0 ) cnt += value ; } ) ; } ) ;
return cnt ; }
var A = [ 2 , 1 , 3 ] ;
document . write ( countTriplets ( A ) ) ;
function CountWays ( n ) {
var noOfWays = Array ( n + 3 ) . fill ( 0 ) ; noOfWays [ 0 ] = 1 ; noOfWays [ 1 ] = 1 ; noOfWays [ 2 ] = 1 + 1 ;
for ( var i = 3 ; i < n + 1 ; i ++ ) {
noOfWays [ i ] = noOfWays [ i - 1 ] + noOfWays [ i - 3 ] ; } return noOfWays [ n ] ; }
var n = 5 ; document . write ( CountWays ( n ) ) ;
function findWinner ( a , n ) {
let v = [ ] ;
let c = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == ' ' ) { c ++ ; }
else { if ( c != 0 ) v . push ( c ) ; c = 0 ; } } if ( c != 0 ) v . push ( c ) ;
if ( v . length == 0 ) { document . write ( " " ) ; return ; }
if ( v . length == 1 ) { if ( ( v [ 0 ] & 1 ) != 0 ) document . write ( " " ) ;
else document . write ( " " ) ; return ; }
let first = Number . MIN_VALUE ; let second = Number . MIN_VALUE ;
for ( let i = 0 ; i < v . length ; i ++ ) {
if ( a [ i ] > first ) { second = first ; first = a [ i ] ; }
else if ( a [ i ] > second && a [ i ] != first ) second = a [ i ] ; }
if ( ( first & 1 ) != 0 && parseInt ( ( first + 1 ) / 2 , 10 ) > second ) document . write ( " " ) ; else document . write ( " " ) ; }
let S = " " ; let N = S . length ; findWinner ( S , N ) ;
function can_Construct ( S , K ) {
var m = new Map ( ) ; var i = 0 , j = 0 , p = 0 ;
if ( S . length == K ) { return true ; }
for ( i = 0 ; i < S . length ; i ++ ) { if ( m . has ( S [ i ] ) ) m . set ( S [ i ] , m . get ( S [ i ] ) + 1 ) else m . set ( S [ i ] , 1 ) }
if ( K > S . length ) { return false ; } else {
m . forEach ( ( value , key ) => { if ( value % 2 != 0 ) { p = p + 1 ; } } ) ; }
if ( K < p ) { return false ; } return true ; }
var S = " " ; var K = 4 ; if ( can_Construct ( S , K ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function equalIgnoreCase ( str1 , str2 ) { let i = 0 ;
str1 = str1 . toLowerCase ( ) ; str2 = str2 . toLowerCase ( ) ;
let x = ( str1 == ( str2 ) ) ;
return x == true ; }
function equalIgnoreCaseUtil ( str1 , str2 ) { let res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) document . write ( " " ) ; else document . write ( " " ) ; }
let str1 , str2 ; str1 = " " ; str2 = " " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " " ; str2 = " " ; equalIgnoreCaseUtil ( str1 , str2 ) ;
function steps ( str , n ) {
var flag ; var x = 0 ;
for ( var i = 0 ; i < str . length ; i ++ ) {
if ( x == 0 ) flag = true ;
if ( x == n - 1 ) flag = false ;
for ( var j = 0 ; j < x ; j ++ ) document . write ( " " ) ; document . write ( str [ i ] + " " ) ;
if ( flag == true ) x ++ ; else x -- ; } }
var n = 4 ; var str = " " ; document . write ( " " + str + " " ) ; document . write ( " " + n + " " ) ;
steps ( str , n ) ;
function countFreq ( arr , n ) {
let visited = new Array ( n ) ; visited . fill ( false ) ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( visited [ i ] == true ) continue ;
let count = 1 ; for ( let j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) { visited [ j ] = true ; count ++ ; } } document . write ( arr [ i ] + " " + count + " " ) ; } }
let arr = [ 10 , 20 , 20 , 10 , 10 , 20 , 5 , 20 ] ; let n = arr . length ; countFreq ( arr , n ) ;
function isDivisible ( str , k ) { let n = str . length ; let c = 0 ;
for ( let i = 0 ; i < k ; i ++ ) if ( str [ n - i - 1 ] == ' ' ) c ++ ;
return ( c == k ) ; }
let str1 = " " ; let k = 2 ; if ( isDivisible ( str1 , k ) == true ) document . write ( " " + " " ) ; else document . write ( " " ) ;
let str2 = " " ; k = 2 ; if ( isDivisible ( str2 , k ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
let NO_OF_CHARS = 256 ;
function canFormPalindrome ( str ) {
let count = new Array ( NO_OF_CHARS ) ; count . fill ( 0 ) ;
for ( let i = 0 ; i < str . length ; i ++ ) count [ str [ i ] . charCodeAt ( ) ] ++ ;
let odd = 0 ; for ( let i = 0 ; i < NO_OF_CHARS ; i ++ ) { if ( ( count [ i ] & 1 ) != 0 ) odd ++ ; if ( odd > 1 ) return false ; }
return true ; }
document . write ( canFormPalindrome ( " " ) ? " " + " " : " " + " " ) ; document . write ( canFormPalindrome ( " " ) ? " " : " " ) ;
function isNumber ( s ) { for ( let i = 0 ; i < s . length ; i ++ ) if ( s [ i ] < ' ' s [ i ] > ' ' ) return false ; return true ; }
let str = " " ;
if ( isNumber ( str ) ) document . write ( " " ) ;
else document . write ( " " ) ;
function reverse ( str , len ) { if ( len == str . length ) { return ; } reverse ( str , len + 1 ) ; document . write ( str [ len ] ) ; }
let a = " " ; reverse ( a , 0 ) ;
var box1 = 0 ;
var box2 = 0 ; var fact = Array ( 11 ) ;
function getProbability ( balls , M ) {
factorial ( 10 ) ;
box2 = M ;
var K = 0 ;
for ( var i = 0 ; i < M ; i ++ ) K += balls [ i ] ;
if ( K % 2 == 1 ) return 0 ;
var all = comb ( K , K / 2 ) ;
var validPermutation = validPermutations ( K / 2 , balls , 0 , 0 , M ) ;
return validPermutation / all ; }
function validPermutations ( n , balls , usedBalls , i , M ) {
if ( usedBalls == n ) {
return box1 == box2 ? 1 : 0 ; }
if ( i >= M ) return 0 ;
var res = validPermutations ( n , balls , usedBalls , i + 1 , M ) ;
box1 ++ ;
for ( var j = 1 ; j <= balls [ i ] ; j ++ ) {
if ( j == balls [ i ] ) box2 -- ;
var combinations = comb ( balls [ i ] , j ) ;
res += combinations * validPermutations ( n , balls , usedBalls + j , i + 1 , M ) ; }
box1 -- ;
box2 ++ ; return res ; }
function factorial ( N ) {
fact [ 0 ] = 1 ;
for ( var i = 1 ; i <= N ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ; }
function comb ( n , r ) { var res = fact [ n ] / fact [ r ] ; res /= fact [ n - r ] ; return res ; }
var arr = [ 2 , 1 , 1 ] ; var N = 4 ; var M = arr . length ;
document . write ( getProbability ( arr , M ) ) ;
function polyarea ( n , r ) {
if ( r < 0 && n < 0 ) return - 1 ;
var A = ( ( r * r * n ) * Math . sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
var r = 9 , n = 6 ; document . write ( polyarea ( n , r ) . toFixed ( 5 ) ) ;
function is_partition_possible ( n , x , y , w ) { let weight_at_x = new Map ( ) ; let max_x = - 2e3 , min_x = 2e3 ;
for ( let i = 0 ; i < n ; i ++ ) { let new_x = x [ i ] - y [ i ] ; max_x = Math . max ( max_x , new_x ) ; min_x = Math . min ( min_x , new_x ) ;
if ( weight_at_x . has ( new_x ) ) { weight_at_x . set ( new_x , weight_at_x . get ( new_x ) + w [ i ] ) ; } else { weight_at_x . set ( new_x , w [ i ] ) ; } } let sum_till = [ ] ; sum_till . push ( 0 ) ;
for ( let s = min_x ; s <= max_x ; s ++ ) { if ( weight_at_x . get ( s ) == null ) sum_till . push ( sum_till [ sum_till . length - 1 ] ) ; else sum_till . push ( sum_till [ sum_till . length - 1 ] + weight_at_x . get ( s ) ) ; } let total_sum = sum_till [ sum_till . length - 1 ] ; let partition_possible = 0 ; for ( let i = 1 ; i < sum_till . length ; i ++ ) { if ( sum_till [ i ] == total_sum - sum_till [ i ] ) partition_possible = 1 ;
if ( sum_till [ i - 1 ] == total_sum - sum_till [ i ] ) partition_possible = 1 ; } document . write ( partition_possible == 1 ? " " : " " ) ; }
let n = 3 ; let x = [ - 1 , - 2 , 1 ] ; let y = [ 1 , 1 , - 1 ] ; let w = [ 3 , 1 , 4 ] ; is_partition_possible ( n , x , y , w ) ;
function findPCSlope ( m ) { return - 1.0 / m ; }
let m = 2.0 ; document . write ( findPCSlope ( m ) ) ;
let pi = 3.14159 ; function
area_of_segment ( radius , angle ) {
let area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
let area_of_triangle = 1 / 2 * ( radius * radius ) * Math . sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
let radius = 10.0 , angle = 90.0 ; document . write ( " " + area_of_segment ( radius , angle ) + " " ) ; document . write ( " " + area_of_segment ( radius , ( 360 - angle ) ) ) ;
function SectorArea ( radius , angle ) { if ( angle >= 360 ) document . write ( " " ) ;
else { let sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; document . write ( sector ) ; } }
let radius = 9 ; let angle = 60 ; SectorArea ( radius , angle ) ;
function gcd ( a , b ) {
function PrimeFactor ( N ) { let primef = new Map ( ) ;
while ( N % 2 == 0 ) { if ( primef . has ( 2 ) ) { primef . set ( 2 , primef . get ( 2 ) + 1 ) ; } else { primef . set ( 2 , 1 ) ; }
N = parseInt ( N / 2 , 10 ) ; }
for ( let i = 3 ; i <= Math . sqrt ( N ) ; i ++ ) {
while ( N % i == 0 ) { if ( primef . has ( i ) ) { primef . set ( i , primef . get ( i ) + 1 ) ; } else { primef . set ( i , 1 ) ; }
N = parseInt ( N / 2 , 10 ) ; } } if ( N > 2 ) { primef [ N ] = 1 ; } return primef ; }
function CountToMakeEqual ( X , Y ) {
let gcdofXY = gcd ( X , Y ) ;
let newX = parseInt ( Y / gcdofXY , 10 ) ; let newY = parseInt ( X / gcdofXY , 10 ) ;
let primeX = PrimeFactor ( newX ) ; let primeY = PrimeFactor ( newY ) ;
let ans = 0 ;
primeX . forEach ( ( values , keys ) => { if ( X % keys != 0 ) { return - 1 ; } ans += primeX . get ( keys ) ; } ) ans += 1 ;
primeY . forEach ( ( values , keys ) => { if ( Y % keys != 0 ) { return - 1 ; } ans += primeY . get ( keys ) ; } )
return ans ; }
let X = 36 ; let Y = 48 ;
let ans = CountToMakeEqual ( X , Y ) ; document . write ( ans ) ;
function check ( Adj , Src , N , visited ) { let color = new Array ( N ) ;
visited [ Src ] = true ; let q = [ ] ;
q . push ( Src ) ; while ( q . length != 0 ) {
let u = q . shift ( ) ;
let Col = color [ u ] ;
for ( let x = 0 ; x < Adj [ u ] . length ; x ++ ) {
if ( visited [ Adj [ u ] [ x ] ] == true && color [ Adj [ u ] [ x ] ] == Col ) { return false ; } else if ( visited [ Adj [ u ] [ x ] ] == false ) {
visited [ Adj [ u ] [ x ] ] = true ;
q . push ( Adj [ u ] [ x ] ) ;
color [ Adj [ u ] [ x ] ] = 1 - Col ; } } }
return true ; }
function addEdge ( Adj , u , v ) { Adj [ u ] . push ( v ) ; Adj [ v ] . push ( u ) ; }
function isPossible ( Arr , N ) {
let Adj = new Array ( N ) ;
for ( let i = 0 ; i < N - 1 ; i ++ ) { for ( let j = i + 1 ; j < N ; j ++ ) {
if ( Arr [ i ] . R < Arr [ j ] . L Arr [ i ] . L > Arr [ j ] . R ) { continue ; }
else { if ( Arr [ i ] . V == Arr [ j ] . V ) {
addEdge ( Adj , i , j ) ; } } } }
let visited = new Array ( N ) ; for ( let i = 0 ; i < N ; i ++ ) visited [ i ] = false ;
for ( let i = 0 ; i < N ; i ++ ) { if ( visited [ i ] == false && Adj [ i ] . length > 0 ) {
if ( check ( Adj , i , N , visited ) == false ) { document . write ( " " ) ; return ; } } }
document . write ( " " ) ; }
let arr = [ new Node ( 5 , 7 , 2 ) , new Node ( 4 , 6 , 1 ) , new Node ( 1 , 5 , 2 ) , new Node ( 6 , 5 , 1 ) ] ; let N = arr . length ; isPossible ( arr , N ) ;
function lexNumbers ( n ) { var sol = [ ] ; dfs ( 1 , n , sol ) ; document . write ( " " + sol [ 0 ] ) ; for ( var i = 1 ; i < sol . length ; i ++ ) document . write ( " " + sol [ i ] ) ; document . write ( " " ) ; } function dfs ( temp , n , sol ) { if ( temp > n ) return ; sol . push ( temp ) ; dfs ( temp * 10 , n , sol ) ; if ( temp % 10 != 9 ) dfs ( temp + 1 , n , sol ) ; }
var n = 15 ; lexNumbers ( n ) ;
function minimumSwaps ( arr ) {
let count = 0 ; let i = 0 ; while ( i < arr . length ) {
if ( arr [ i ] != i + 1 ) { while ( arr [ i ] != i + 1 ) { let temp = 0 ;
temp = arr [ arr [ i ] - 1 ] ; arr [ arr [ i ] - 1 ] = arr [ i ] ; arr [ i ] = temp ; count ++ ; } }
i ++ ; } return count ; }
let arr = [ 2 , 3 , 4 , 1 , 5 ] ;
document . write ( minimumSwaps ( arr ) ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; this . prev = null ; } }
function append ( head_ref , new_data ) {
new_node = new Node ( ) ; last = head_ref ;
new_node . data = new_data ;
new_node . next = null ;
if ( head_ref == null ) { new_node . prev = null ; head_ref = new_node ; return head_ref ; }
while ( last . next != null ) last = last . next ;
last . next = new_node ;
new_node . prev = last ; return head_ref ; }
function printList ( node ) { last ;
while ( node != null ) { document . write ( node . data + " " ) ; last = node ; node = node . next ; } }
function mergeList ( p , q ) { s = null ;
if ( p == null q == null ) { return ( p == null ? q : p ) ; }
if ( p . data < q . data ) { p . prev = s ; s = p ; p = p . next ; } else { q . prev = s ; s = q ; q = q . next ; }
head = s ; while ( p != null && q != null ) { if ( p . data < q . data ) {
s . next = p ; p . prev = s ; s = s . next ; p = p . next ; } else {
s . next = q ; q . prev = s ; s = s . next ; q = q . next ; } }
if ( p == null ) { s . next = q ; q . prev = s ; } if ( q == null ) { s . next = p ; p . prev = s ; }
return head ; }
function mergeAllList ( head , k ) { finalList = null ; for ( i = 0 ; i < k ; i ++ ) {
finalList = mergeList ( finalList , head [ i ] ) ; }
return finalList ; }
var k = 3 ; head = Array ( k ) . fill ( null ) ;
for ( i = 0 ; i < k ; i ++ ) { head [ i ] = null ; }
head [ 0 ] = append ( head [ 0 ] , 1 ) ; head [ 0 ] = append ( head [ 0 ] , 5 ) ; head [ 0 ] = append ( head [ 0 ] , 9 ) ;
head [ 1 ] = append ( head [ 1 ] , 2 ) ; head [ 1 ] = append ( head [ 1 ] , 3 ) ; head [ 1 ] = append ( head [ 1 ] , 7 ) ; head [ 1 ] = append ( head [ 1 ] , 12 ) ;
head [ 2 ] = append ( head [ 2 ] , 8 ) ; head [ 2 ] = append ( head [ 2 ] , 11 ) ; head [ 2 ] = append ( head [ 2 ] , 13 ) ; head [ 2 ] = append ( head [ 2 ] , 18 ) ;
finalList = mergeAllList ( head , k ) ;
printList ( finalList ) ;
function insertionSortRecursive ( arr , n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
let last = arr [ n - 1 ] ; let j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
let arr = [ 12 , 11 , 13 , 5 , 6 ] ; insertionSortRecursive ( arr , arr . length ) ; for ( let i = 0 ; i < arr . length ; i ++ ) { document . write ( arr [ i ] + " " ) ; }
function bubbleSort ( arr , n ) {
if ( n == 1 ) return ;
for ( var i = 0 ; i < n - 1 ; i ++ ) if ( arr [ i ] > arr [ i + 1 ] ) {
var temp = arr [ i ] ; arr [ i ] = arr [ i + 1 ] ; arr [ i + 1 ] = temp ; }
bubbleSort ( arr , n - 1 ) ; }
function maxSumAfterPartition ( arr , n ) {
let pos = [ ] ;
let neg = [ ] ;
let zero = 0 ;
let pos_sum = 0 ;
let neg_sum = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > 0 ) { pos . push ( arr [ i ] ) ; pos_sum += arr [ i ] ; } else if ( arr [ i ] < 0 ) { neg . push ( arr [ i ] ) ; neg_sum += arr [ i ] ; } else { zero ++ ; } }
let ans = 0 ;
pos . sort ( function ( a , b ) { return a - b } )
neg . sort ( function ( a , b ) { return b - a } )
if ( pos . length > 0 && neg . length > 0 ) { ans = ( pos_sum - neg_sum ) ; } else if ( pos . length > 0 ) { if ( zero > 0 ) {
ans = ( pos_sum ) ; } else {
ans = ( pos_sum - 2 * pos [ 0 ] ) ; } } else { if ( zero > 0 ) {
ans = ( - 1 * neg_sum ) ; } else {
ans = ( neg [ 0 ] - ( neg_sum - neg [ 0 ] ) ) ; } } return ans ; }
let arr = [ 1 , 2 , 3 , - 5 , - 7 ] ; let n = arr . length ; document . write ( maxSumAfterPartition ( arr , n ) ) ;
function MaxXOR ( arr , N ) {
var res = 0 ;
for ( var i = 0 ; i < N ; i ++ ) { res |= arr [ i ] ; }
return res ; }
var arr = [ 1 , 5 , 7 ] ; var N = arr . length ; document . write ( MaxXOR ( arr , N ) ) ;
function countEqual ( A , B , N ) {
let first = 0 ; let second = N - 1 ;
let count = 0 ; while ( first < N && second >= 0 ) {
if ( A [ first ] < B [ second ] ) {
first ++ ; }
else if ( B [ second ] < A [ first ] ) {
second -- ; }
else {
count ++ ;
first ++ ;
second -- ; } }
return count ; }
let A = [ 2 , 4 , 5 , 8 , 12 , 13 , 17 , 18 , 20 , 22 , 309 , 999 ] ; let B = [ 109 , 99 , 68 , 54 , 22 , 19 , 17 , 13 , 11 , 5 , 3 , 1 ] ; let N = A . length ; document . write ( countEqual ( A , B , N ) ) ;
let arr = [ ] ; for ( let m = 0 ; m < 100005 ; m ++ ) { arr [ m ] = 0 ; }
function isPalindrome ( N ) {
int temp = N ;
let res = 0 ;
while ( temp != 0 ) { let rem = temp % 10 ; res = res * 10 + rem ; temp = Math . floor ( temp / 10 ) ; }
if ( res == N ) { return true ; } else { return false ; } }
function sumOfDigits ( N ) {
let sum = 0 ; while ( N != 0 ) {
sum += N % 10 ;
N = Math . floor ( N / 10 ) ; }
return sum ; }
function isPrime ( n ) {
if ( n <= 1 ) { return false ; }
for ( let i = 2 ; i <= Math . floor ( n / 2 ) ; ++ i ) {
if ( n % i == 0 ) return false ; } return true ; }
function precompute ( ) {
for ( let i = 1 ; i <= 100000 ; i ++ ) {
if ( isPalindrome ( i ) ) {
let sum = sumOfDigits ( i ) ;
if ( isPrime ( sum ) ) arr [ i ] = 1 ; else arr [ i ] = 0 ; } else arr [ i ] = 0 ; }
for ( let i = 1 ; i <= 100000 ; i ++ ) { arr [ i ] = arr [ i ] + arr [ i - 1 ] ; } }
function countNumbers ( Q , N ) {
precompute ( ) ;
for ( let i = 0 ; i < N ; i ++ ) {
document . write ( ( arr [ Q [ i ] [ 1 ] ] - arr [ Q [ i ] [ 0 ] - 1 ] ) + " " ) ; } }
let Q = [ [ 5 , 9 ] , [ 1 , 101 ] ] ; let N = Q . length ;
countNumbers ( Q , N ) ;
function sum ( n ) { var res = 0 ; while ( n > 0 ) { res += n % 10 ; n /= 10 ; } return res ; }
function smallestNumber ( n , s ) {
if ( sum ( n ) <= s ) { return n ; }
var ans = n , k = 1 ; for ( i = 0 ; i < 9 ; ++ i ) {
var digit = ( ans / k ) % 10 ;
var add = k * ( ( 10 - digit ) % 10 ) ; ans += add ;
if ( sum ( ans ) <= s ) { break ; }
k *= 10 ; } return ans ; }
var N = 3 , S = 2 ;
document . write ( smallestNumber ( N , S ) ) ;
function maxSubsequences ( arr , n ) {
let map = new Map ( ) ;
let maxCount = 0 ;
let count ; for ( let i = 0 ; i < n ; i ++ ) {
if ( map . has ( arr [ i ] ) ) {
count = map [ arr [ i ] ] ;
if ( count > 1 ) {
map . add ( arr [ i ] , count - 1 ) ; }
else map . delete ( arr [ i ] ) ;
if ( arr [ i ] - 1 > 0 ) if ( map . has ( arr [ i ] - 1 ) ) map [ arr [ i ] - 1 ] ++ ; else map . set ( arr [ i ] - 1 , 1 ) ; } else {
maxCount ++ ;
if ( arr [ i ] - 1 > 0 ) if ( map . has ( arr [ i ] - 1 ) ) map [ arr [ i ] - 1 ] ++ ; else map . set ( arr [ i ] - 1 , 1 ) ; } }
return maxCount ; }
let n = 5 ; let arr = [ 4 , 5 , 2 , 1 , 4 ] ; document . write ( maxSubsequences ( arr , n ) ) ;
Function to remove first and last occurrence of a given character from the given String * / function removeOcc ( s , ch ) {
for ( var i = 0 ; i < s . length ; i ++ ) {
if ( s [ i ] === ch ) { s = s . substring ( 0 , i ) + s . substring ( i + 1 ) ; break ; } }
for ( var i = s . length - 1 ; i > - 1 ; i -- ) {
if ( s [ i ] === ch ) { s = s . substring ( 0 , i ) + s . substring ( i + 1 ) ; break ; } } return s ; }
var s = " " ; var ch = " " ; document . write ( removeOcc ( s , ch ) ) ;
function minSteps ( N , increasing , decreasing , m1 , m2 ) {
var mini = 2147483647 ; var i ;
for ( i = 0 ; i < m1 ; i ++ ) { if ( mini > increasing [ i ] ) mini = increasing [ i ] ; }
var maxi = - 2147483648 ;
for ( i = 0 ; i < m2 ; i ++ ) { if ( maxi < decreasing [ i ] ) maxi = decreasing [ i ] ; }
var minSteps = Math . max ( maxi , N - mini ) ;
document . write ( minSteps ) ; }
var N = 7 ;
var increasing = [ 3 , 5 ] ; var decreasing = [ 6 ] ;
minSteps ( N , increasing , decreasing , m1 , m2 ) ;
function solve ( P , n ) {
let arr = Array . from ( { length : n + 1 } , ( _ , i ) => 0 ) ; arr [ 0 ] = 0 ; for ( let i = 0 ; i < n ; i ++ ) arr [ i + 1 ] = P [ i ] ;
let cnt = 0 ; for ( let i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] == i ) { let t = arr [ i + 1 ] ; arr [ i + 1 ] = arr [ i ] ; arr [ i ] = t ; cnt ++ ; } }
if ( arr [ n ] == n ) {
let t = arr [ n - 1 ] ; arr [ n - 1 ] = arr [ n ] ; arr [ n ] = t ; cnt ++ ; }
document . write ( cnt ) ; }
let N = 9 ;
let P = [ 1 , 2 , 4 , 9 , 5 , 8 , 7 , 3 , 6 ] ;
solve ( P , N ) ;
function isWaveArray ( arr , n ) { let result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( let i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( let i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
let arr = [ 1 , 3 , 2 , 4 ] ; let n = arr . length ; if ( isWaveArray ( arr , n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function countPossiblities ( arr , n ) {
let lastOccur = new Array ( 100000 ) ; for ( let i = 0 ; i < n ; i ++ ) { lastOccur [ i ] = - 1 ; }
dp = new Array ( n + 1 ) ;
dp [ 0 ] = 1 ; for ( let i = 1 ; i <= n ; i ++ ) { let curEle = arr [ i - 1 ] ;
dp [ i ] = dp [ i - 1 ] ;
if ( lastOccur [ curEle ] != - 1 & lastOccur [ curEle ] < i - 1 ) { dp [ i ] += dp [ lastOccur [ curEle ] ] ; }
lastOccur [ curEle ] = i ; }
document . write ( dp [ n ] + " " ) ; }
let arr = [ 1 , 2 , 1 , 2 , 2 ] ; let N = arr . length ; countPossiblities ( arr , N ) ;
function maxSum ( arr , n , m ) {
let dp = new Array ( n ) ; for ( var i = 0 ; i < dp . length ; i ++ ) { dp [ i ] = new Array ( 2 ) ; }
dp [ 0 ] [ m - 1 ] = arr [ 0 ] [ m - 1 ] ; dp [ 1 ] [ m - 1 ] = arr [ 1 ] [ m - 1 ] ;
for ( let j = m - 2 ; j >= 0 ; j -- ) {
for ( let i = 0 ; i < 2 ; i ++ ) { if ( i == 1 ) { dp [ i ] [ j ] = Math . max ( arr [ i ] [ j ] + dp [ 0 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 0 ] [ j + 2 ] ) ; } else { dp [ i ] [ j ] = Math . max ( arr [ i ] [ j ] + dp [ 1 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 1 ] [ j + 2 ] ) ; } } }
document . write ( Math . max ( dp [ 0 ] [ 0 ] , dp [ 1 ] [ 0 ] ) ) ; }
let arr = [ [ 1 , 50 , 21 , 5 ] , [ 2 , 10 , 10 , 5 ] ] ;
let N = arr [ 0 ] . length ;
maxSum ( arr , 2 , N ) ;
function maxSum ( arr , n ) {
var r1 = 0 , r2 = 0 ;
for ( i = 0 ; i < n ; i ++ ) { var temp = r1 ; r1 = Math . max ( r1 , r2 + arr [ 0 ] [ i ] ) ; r2 = Math . max ( r2 , temp + arr [ 1 ] [ i ] ) ; }
document . write ( Math . max ( r1 , r2 ) ) ; }
var arr = [ [ 1 , 50 , 21 , 5 ] , [ 2 , 10 , 10 , 5 ] ] ;
var n = arr [ 0 ] . length ; maxSum ( arr , n ) ;
var mod = parseInt ( 1e9 + 7 ) ; var mx = 1000000 ; var fact = new Array ( mx + 1 ) . fill ( 0 ) ;
function Calculate_factorial ( ) { fact [ 0 ] = 1 ;
for ( var i = 1 ; i <= mx ; i ++ ) { fact [ i ] = i * fact [ i - 1 ] ; fact [ i ] %= mod ; } }
function UniModal_per ( a , b ) { var res = 1 ;
while ( b > 0 ) {
if ( b % 2 !== 0 ) res = res * a ; res %= mod ; a = a * a ; a %= mod ;
b = parseInt ( b / 2 ) ; }
return res ; }
function countPermutations ( n ) {
Calculate_factorial ( ) ;
var uni_modal = UniModal_per ( 2 , n - 1 ) ;
var nonuni_modal = fact [ n ] - uni_modal ; document . write ( uni_modal + " " + nonuni_modal ) ; return ; }
var N = 4 ;
countPermutations ( N ) ;
function longestSubseq ( s , len ) {
var ones = new Array ( len + 1 ) . fill ( 0 ) ; var zeroes = new Array ( len + 1 ) . fill ( 0 ) ;
for ( var i = 0 ; i < len ; i ++ ) {
if ( s [ i ] === " " ) { ones [ i + 1 ] = ones [ i ] + 1 ; zeroes [ i + 1 ] = zeroes [ i ] ; }
x += ones [ i ] ;
x += zeroes [ j ] - zeroes [ i ] ;
x += ones [ len ] - ones [ j ] ;
answer = Math . max ( answer , x ) ; x = 0 ; } }
document . write ( answer ) ; }
var s = " " ; var len = s . length ; longestSubseq ( s , len ) ;
var MAX = 100 ;
function largestSquare ( matrix , R , C , q_i , q_j , K , Q ) {
for ( var q = 0 ; q < Q ; q ++ ) { var i = q_i [ q ] ; var j = q_j [ q ] ; var min_dist = Math . min ( Math . min ( i , j ) , Math . min ( R - i - 1 , C - j - 1 ) ) ; var ans = - 1 ; for ( var k = 0 ; k <= min_dist ; k ++ ) { var count = 0 ;
for ( var row = i - k ; row <= i + k ; row ++ ) for ( var col = j - k ; col <= j + k ; col ++ ) count += matrix [ row ] [ col ] ;
if ( count > K ) break ; ans = 2 * k + 1 ; } document . write ( ans + " " ) ; } }
var matrix = [ [ 1 , 0 , 1 , 0 , 0 ] , [ 1 , 0 , 1 , 1 , 1 ] , [ 1 , 1 , 1 , 1 , 1 ] , [ 1 , 0 , 0 , 1 , 0 ] ] ; var K = 9 , Q = 1 ; var q_i = [ 1 ] ; var q_j = [ 2 ] ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ;
function largestSquare ( matrix , R , C , q_i , q_j , K , Q ) { let countDP = new Array ( R ) ; for ( let i = 0 ; i < R ; i ++ ) { countDP [ i ] = new Array ( C ) ; for ( let j = 0 ; j < C ; j ++ ) countDP [ i ] [ j ] = 0 ; }
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] ; for ( let i = 1 ; i < R ; i ++ ) countDP [ i ] [ 0 ] = countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ; for ( let j = 1 ; j < C ; j ++ ) countDP [ 0 ] [ j ] = countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ; for ( let i = 1 ; i < R ; i ++ ) for ( let j = 1 ; j < C ; j ++ ) countDP [ i ] [ j ] = matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ;
for ( let q = 0 ; q < Q ; q ++ ) { let i = q_i [ q ] ; let j = q_j [ q ] ;
let min_dist = Math . min ( Math . min ( i , j ) , Math . min ( R - i - 1 , C - j - 1 ) ) ; let ans = - 1 ; for ( let k = 0 ; k <= min_dist ; k ++ ) { let x1 = i - k , x2 = i + k ; let y1 = j - k , y2 = j + k ;
let count = countDP [ x2 ] [ y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 ] [ y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 ] [ y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 ] [ y1 - 1 ] ; if ( count > K ) break ; ans = 2 * k + 1 ; } document . write ( ans + " " ) ; } }
let matrix = [ [ 1 , 0 , 1 , 0 , 0 ] , [ 1 , 0 , 1 , 1 , 1 ] , [ 1 , 1 , 1 , 1 , 1 ] , [ 1 , 0 , 0 , 1 , 0 ] ] ; let K = 9 , Q = 1 ; let q_i = [ 1 ] ; let q_j = [ 2 ] ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ;
function MinCost ( arr , n ) {
let dp = new Array ( n + 5 ) ; let sum = new Array ( n + 5 ) ; for ( let i = 0 ; i < n + 5 ; i ++ ) { dp [ i ] = [ ] ; sum [ i ] = [ ] ; for ( let j = 0 ; j < n + 5 ; j ++ ) { dp [ i ] . push ( 0 ) sum [ i ] . push ( 0 ) } } console . log ( dp )
for ( let i = 0 ; i < n ; i ++ ) { let k = arr [ i ] ; for ( let j = i ; j < n ; j ++ ) { if ( i == j ) sum [ i ] [ j ] = k ; else { k += arr [ j ] ; sum [ i ] [ j ] = k ; } } }
for ( let i = n - 1 ; i >= 0 ; i -- ) {
for ( let j = i ; j < n ; j ++ ) { dp [ i ] [ j ] = Number . MAX_SAFE_INTEGER ;
if ( i == j ) dp [ i ] [ j ] = 0 ; else { for ( let k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , dp [ i ] [ k ] + dp [ k + 1 ] [ j ] + sum [ i ] [ j ] ) ; } } } } return dp [ 0 ] [ n - 1 ] ; }
let arr = [ 7 , 6 , 8 , 6 , 1 , 1 ] ; let n = arr . length ; document . write ( MinCost ( arr , n ) ) ;
function f ( i , state , A , dp , N ) { if ( i >= N ) return 0 ;
else if ( dp [ i ] [ state ] != - 1 ) { return dp [ i ] [ state ] ; }
else { if ( i == N - 1 ) dp [ i ] [ state ] = 1 ; else if ( state == 1 && A [ i ] > A [ i + 1 ] ) dp [ i ] [ state ] = 1 ; else if ( state == 2 && A [ i ] < A [ i + 1 ] ) dp [ i ] [ state ] = 1 ; else if ( state == 1 && A [ i ] <= A [ i + 1 ] ) dp [ i ] [ state ] = 1 + f ( i + 1 , 2 , A , dp , N ) ; else if ( state == 2 && A [ i ] >= A [ i + 1 ] ) dp [ i ] [ state ] = 1 + f ( i + 1 , 1 , A , dp , N ) ; return dp [ i ] [ state ] ; } }
function maxLenSeq ( A , N ) { let i , j , tmp , y , ans ;
let dp = new Array ( 1000 ) ;
for ( i = 0 ; i < N ; i ++ ) { tmp = f ( i , 1 , A , dp , N ) ; tmp = f ( i , 2 , A , dp , N ) ; }
ans = - 1 ; for ( i = 0 ; i < N ; i ++ ) {
y = dp [ i ] [ 1 ] ; if ( i + y >= N ) ans = Math . max ( ans , dp [ i ] [ 1 ] + 1 ) ;
else if ( y % 2 == 0 ) { ans = Math . max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 2 ] ) ; }
else if ( y % 2 == 1 ) { ans = Math . max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 1 ] ) ; } } return ans ; }
let A = [ 1 , 10 , 3 , 20 , 25 , 24 ] ; let n = A . length ; document . write ( maxLenSeq ( A , n ) ) ;
function MaxGCD ( a , n ) {
let Prefix = new Array ( n + 2 ) ; let Suffix = new Array ( n + 2 ) ;
Prefix [ 1 ] = a [ 0 ] ; for ( let i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = gcd ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( let i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = gcd ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
let ans = Math . max ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( let i = 2 ; i < n ; i += 1 ) { ans = Math . max ( ans , gcd ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; }
let a = [ 14 , 17 , 28 , 70 ] ; let n = a . length ; document . write ( MaxGCD ( a , n ) ) ;
let right = 2 ; let left = 4 ; let dp = new Array ( left ) ;
function findSubarraySum ( ind , flips , n , a , k ) {
if ( flips > k ) return ( - 1e9 ) ;
if ( ind == n ) return 0 ;
if ( dp [ ind ] [ flips ] != - 1 ) return dp [ ind ] [ flips ] ;
let ans = 0 ;
ans = Math . max ( 0 , a [ ind ] + findSubarraySum ( ind + 1 , flips , n , a , k ) ) ; ans = Math . max ( ans , - a [ ind ] + findSubarraySum ( ind + 1 , flips + 1 , n , a , k ) ) ;
return dp [ ind ] [ flips ] = ans ; }
function findMaxSubarraySum ( a , n , k ) {
for ( let i = 0 ; i < n ; i ++ ) { dp [ i ] = new Array ( k ) ; for ( let j = 0 ; j < k + 1 ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } let ans = ( - 1e9 ) ;
for ( let i = 0 ; i < n ; i ++ ) ans = Math . max ( ans , findSubarraySum ( i , 0 , n , a , k ) ) ;
if ( ans == 0 && k == 0 ) { let max = Number . MIN_VALUE ; for ( let i = 0 ; i < a . length ; i ++ ) { max = Math . max ( max , a [ i ] ) ; } return max ; } return ans ; }
let a = [ - 1 , - 2 , - 100 , - 10 ] ; let n = a . length ; let k = 1 ; document . write ( findMaxSubarraySum ( a , n , k ) ) ;
var mod = 1000000007 ;
function sumOddFibonacci ( n ) { var Sum = Array ( n + 1 ) . fill ( 0 ) ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
var n = 6 ; document . write ( sumOddFibonacci ( n ) ) ;
function fun ( marks , n ) {
let dp = new Array ( n ) ; let temp ; for ( let i = 0 ; i < n ; i ++ ) dp [ i ] = 1 ; for ( let i = 0 ; i < n - 1 ; i ++ ) {
if ( marks [ i ] > marks [ i + 1 ] ) { temp = i ; while ( true ) { if ( ( marks [ temp ] > marks [ temp + 1 ] ) && temp >= 0 ) { if ( dp [ temp ] > dp [ temp + 1 ] ) { temp -= 1 ; continue ; } else { dp [ temp ] = dp [ temp + 1 ] + 1 ; temp -= 1 ; } } else break ; } }
else if ( marks [ i ] < marks [ i + 1 ] ) dp [ i + 1 ] = dp [ i ] + 1 ; } let sum = 0 ; for ( let i = 0 ; i < n ; i ++ ) sum += dp [ i ] ; return sum ; }
let n = 6 ;
let marks = [ 1 , 4 , 5 , 2 , 2 , 1 ] ;
document . write ( fun ( marks , n ) ) ;
function solve ( N , K ) {
let combo = new Array ( 50 ) ; combo . fill ( 0 ) ;
combo [ 0 ] = 1 ;
for ( let i = 1 ; i <= K ; i ++ ) {
for ( let j = 0 ; j <= N ; j ++ ) {
if ( j >= i ) {
combo [ j ] += combo [ j - i ] ; } } }
return combo [ N ] ; }
let N = 29 ; let K = 5 ; document . write ( solve ( N , K ) ) ; solve ( N , K ) ;
function computeLIS ( circBuff , start , end , n ) { let LIS = new Array ( n + end - start ) ;
for ( let i = start ; i < end ; i ++ ) LIS [ i ] = 1 ;
for ( let i = start + 1 ; i < end ; i ++ )
for ( let j = start ; j < i ; j ++ ) if ( circBuff [ i ] > circBuff [ j ] && LIS [ i ] < LIS [ j ] + 1 ) LIS [ i ] = LIS [ j ] + 1 ;
let res = Number . MIN_VALUE ; for ( let i = start ; i < end ; i ++ ) res = Math . max ( res , LIS [ i ] ) ; return res ; }
function LICS ( arr , n ) {
let circBuff = new Array ( 2 * n ) ; for ( let i = 0 ; i < n ; i ++ ) circBuff [ i ] = arr [ i ] ; for ( let i = n ; i < 2 * n ; i ++ ) circBuff [ i ] = arr [ i - n ] ;
let res = Number . MIN_VALUE ; for ( let i = 0 ; i < n ; i ++ ) res = Math . max ( computeLIS ( circBuff , i , i + n , n ) , res ) ; return res ; }
let arr = [ 1 , 4 , 6 , 2 , 3 ] ; document . write ( " " + LICS ( arr , arr . length ) ) ;
function binomialCoeff ( n , k ) { var C = Array ( k + 1 ) . fill ( 0 ) ; C [ 0 ] = 1 ;
for ( i = 1 ; i <= n ; i ++ ) { for ( j = min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
var n = 3 , m = 2 ; document . write ( " " + binomialCoeff ( n + m , n ) ) ;
function LCIS ( arr1 , n , arr2 , m ) {
let table = [ ] ; for ( let j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
let current = 0 ;
for ( let j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
let result = 0 ; for ( let i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
let arr1 = [ 3 , 4 , 9 , 1 ] ; let arr2 = [ 5 , 3 , 8 , 9 , 10 , 2 , 1 ] ; let n = arr1 . length ; let m = arr2 . length ; document . write ( " " + LCIS ( arr1 , n , arr2 , m ) ) ;
function longComPre ( arr , N ) {
let freq = new Array ( N ) ; for ( let i = 0 ; i < N ; i ++ ) { freq [ i ] = new Array ( 256 ) ; for ( let j = 0 ; j < 256 ; j ++ ) { freq [ i ] [ j ] = 0 ; } }
for ( let i = 0 ; i < N ; i ++ ) {
let M = arr [ i ] . length ;
for ( let j = 0 ; j < M ; j ++ ) {
freq [ i ] [ arr [ i ] [ j ] . charCodeAt ( 0 ) ] ++ ; } }
let maxLen = 0 ;
for ( let j = 0 ; j < 256 ; j ++ ) {
let minRowVal = Number . MAX_VALUE ;
for ( let i = 0 ; i < N ; i ++ ) {
minRowVal = Math . min ( minRowVal , freq [ i ] [ j ] ) ; }
maxLen += minRowVal ; } return maxLen ; }
let arr = [ " " , " " , " " ] ; let N = 3 ; document . write ( longComPre ( arr , N ) ) ;
let MAX_CHAR = 26 ;
function removeChars ( arr , k ) {
let hash = Array . from ( { length : MAX_CHAR } , ( _ , i ) => 0 ) ;
let n = arr . length ; for ( let i = 0 ; i < n ; ++ i ) hash [ arr [ i ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] ++ ;
let ans = " " ;
for ( let i = 0 ; i < n ; ++ i ) {
if ( hash [ arr [ i ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] != k ) { ans += arr [ i ] ; } } return ans ; }
let str = " " . split ( ' ' ) ; let k = 2 ;
document . write ( removeChars ( str , k ) ) ;
function sub_segments ( str , n ) { let l = str . length ; for ( let x = 0 ; x < l ; x += n ) { let newlist = str . substr ( x , n ) ;
let arr = [ ] ; for ( let y of newlist ) {
if ( ! arr . includes ( y ) ) arr . push ( y ) ; } for ( let y of arr ) document . write ( y ) ; document . write ( " " ) ; } }
let str = " " ; let n = 4 ; sub_segments ( str , n ) ;
function equalIgnoreCase ( str1 , str2 ) { let i = 0 ;
let len1 = str1 . length ;
let len2 = str2 . length ;
if ( len1 != len2 ) return false ;
while ( i < len1 ) {
if ( str1 [ i ] == str2 [ i ] ) { i ++ ; }
else if ( ! ( ( str1 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str1 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) || ( str1 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str1 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) ) ) { return false ; }
else if ( ! ( ( str2 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str2 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) || ( str2 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str2 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) ) ) { return false ; }
else {
if ( str1 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str1 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) { if ( str1 [ i ] . charCodeAt ( ) - 32 != str2 [ i ] . charCodeAt ( ) ) return false ; } else if ( str1 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str1 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) { if ( str1 [ i ] . charCodeAt ( ) + 32 != str2 [ i ] . charCodeAt ( ) ) return false ; }
i ++ ;
return true ;
function equalIgnoreCaseUtil ( str1 , str2 ) { let res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ; }
let str1 , str2 ; str1 = " " ; str2 = " " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " " ; str2 = " " ; equalIgnoreCaseUtil ( str1 , str2 ) ;
function maxValue ( a , b ) {
b . sort ( function ( x , y ) { return x - y ; } ) ; let n = a . length ; let m = b . length ;
let j = m - 1 ; for ( let i = 0 ; i < n ; i ++ ) {
if ( j < 0 ) break ; if ( b [ j ] > a [ i ] ) { a [ i ] = b [ j ] ;
j -- ; } }
return ( a ) . join ( " " ) ; }
let a = " " ; let b = " " ; document . write ( maxValue ( a . split ( " " ) , b . split ( " " ) ) ) ;
function checkIfUnequal ( n , q ) {
let s1 = n . toString ( ) ; let a = new Array ( 26 ) ; for ( let i = 0 ; i < a . length ; i ++ ) { a [ i ] = 0 ; }
for ( let i = 0 ; i < s1 . length ; i ++ ) a [ s1 [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
let prod = n * q ;
let s2 = prod . toString ( ) ;
for ( let i = 0 ; i < s2 . length ; i ++ ) {
if ( a [ s2 [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] > 0 ) return false ; }
return true ; }
function countInRange ( l , r , q ) { let count = 0 ; for ( let i = l ; i <= r ; i ++ ) {
if ( checkIfUnequal ( i , q ) ) count ++ ; } return count ; }
let l = 10 , r = 12 , q = 2 ;
document . write ( countInRange ( l , r , q ) ) ;
function is_possible ( s ) {
let l = s . length ; let one = 0 , zero = 0 ; for ( let i = 0 ; i < l ; i ++ ) {
if ( s [ i ] == ' ' ) zero ++ ;
else one ++ ; }
if ( l % 2 == 0 ) return ( one == zero ) ;
else return ( Math . abs ( one - zero ) == 1 ) ; } let s = " " ; if ( is_possible ( s ) ) document . write ( " " ) ; else document . write ( " " ) ;
let limit = 255 ; function countFreq ( str ) {
let count = new Array ( limit + 1 ) ; for ( let i = 0 ; i < count . length ; i ++ ) { count [ i ] = 0 ; }
for ( let i = 0 ; i < str . length ; i ++ ) count [ str [ i ] . charCodeAt ( 0 ) ] ++ ; for ( let i = 0 ; i <= limit ; i ++ ) { if ( count [ i ] > 0 ) document . write ( String . fromCharCode ( i ) + " " + count [ i ] + " " ) ; } }
let str = " " ; countFreq ( str ) ;
function countEvenOdd ( arr , n , K ) { let even = 0 , odd = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
let x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } let y ;
y = __builtin_popcount ( K ) ;
if ( ( y & 1 ) != 0 ) { document . write ( " " + odd + " " + even ) ; }
else { document . write ( " " + even + " " + odd ) ; } }
let arr = [ 4 , 2 , 15 , 9 , 8 , 8 ] ; let K = 3 ; let n = arr . length ;
countEvenOdd ( arr , n , K ) ;
function convert ( s ) { var n = s . length ; var s1 = " " ; s1 = s1 + s . charAt ( 0 ) . toLowerCase ( ) ; for ( i = 1 ; i < n ; i ++ ) {
if ( s . charAt ( i ) == ' ' && i < n ) {
s1 = s1 + " " + s . charAt ( i + 1 ) . toLowerCase ( ) ; i ++ ; }
else s1 = s1 + s . charAt ( i ) . toUpperCase ( ) ; }
return s1 ; }
var str = " " ; document . write ( convert ( str ) ) ;
function reverse ( num ) { let rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = parseInt ( num / 10 ) ; } return rev_num ; }
function properDivSum ( num ) {
let result = 0 ;
for ( i = 2 ; i <= Math . sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; } function isTcefrep ( n ) { return properDivSum ( n ) == reverse ( n ) ; }
let N = 6 ;
if ( isTcefrep ( N ) ) document . write ( " " ) ; else document . write ( " " ) ;
function reverseString ( str ) { return str . split ( " " ) . reverse ( ) . join ( " " ) ; } function findNthNo ( n ) { let res = " " ; while ( n >= 1 ) {
if ( ( n & 1 ) == 1 ) { res = res + " " ; n = ( n - 1 ) / 2 ; }
else { res = res + " " ; n = ( n - 2 ) / 2 ; } }
sb = ( res ) ; sb = reverseString ( sb ) ; return ( sb ) ; }
let n = 5 ; document . write ( findNthNo ( n ) ) ;
function findNthNonSquare ( n ) {
var x = n ;
var ans = x + Math . floor ( 0.5 + Math . sqrt ( x ) ) ; return parseInt ( ans ) ; }
var n = 16 ;
document . write ( " " + n + " " ) ; document . write ( findNthNonSquare ( n ) ) ;
function seiresSum ( n , a ) { return n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ; }
let n = 2 ; a = [ 1 , 2 , 3 , 4 ] ; document . write ( seiresSum ( n , a ) ) ;
function checkdigit ( n , k ) { while ( n != 0 ) {
let rem = n % 10 ;
if ( rem == k ) return true ; n = n / 10 ; } return false ; }
function findNthNumber ( n , k ) {
for ( let i = k + 1 , count = 1 ; count < n ; i ++ ) {
if ( checkdigit ( i , k ) || ( i % k == 0 ) ) count ++ ; if ( count == n ) return i ; } return - 1 ; }
let n = 10 , k = 2 ; document . write ( findNthNumber ( n , k ) ) ;
function find_permutations ( arr ) { var cnt = 0 ; var max_ind = - 1 , min_ind = 10000000 ; var n = arr . length ; var index_of = new Map ( ) ;
for ( var i = 0 ; i < n ; i ++ ) { index_of . set ( arr [ i ] , i + 1 ) ; } for ( var i = 1 ; i <= n ; i ++ ) {
max_ind = Math . max ( max_ind , index_of . get ( i ) ) ; min_ind = Math . min ( min_ind , index_of . get ( i ) ) ; if ( max_ind - min_ind + 1 == i ) cnt ++ ; } return cnt ; }
var nums = [ ] ; nums . push ( 2 ) ; nums . push ( 3 ) ; nums . push ( 1 ) ; nums . push ( 5 ) ; nums . push ( 4 ) ; document . write ( find_permutations ( nums ) ) ;
function getCount ( a , n ) {
let gcd = 0 ; for ( let i = 0 ; i < n ; i ++ ) gcd = calgcd ( gcd , a [ i ] ) ;
let cnt = 0 ; for ( let i = 1 ; i * i <= gcd ; i ++ ) { if ( gcd % i == 0 ) {
if ( i * i == gcd ) cnt ++ ;
else cnt += 2 ; } } return cnt ; }
let a = [ 4 , 16 , 1024 , 48 ] ; let n = a . length ; document . write ( getCount ( a , n ) ) ;
function delCost ( s , cost ) {
var visited = Array ( s . length ) . fill ( false ) ;
var ans = 0 ;
for ( i = 0 ; i < s . length ; i ++ ) {
if ( visited [ i ] ) { continue ; }
var maxDel = 0 ;
var totalCost = 0 ;
visited [ i ] = true ;
for ( j = i ; j < s . length ; j ++ ) {
if ( s . charAt ( i ) == s . charAt ( j ) ) {
maxDel = Math . max ( maxDel , cost [ j ] ) ; totalCost += cost [ j ] ;
visited [ j ] = true ; } }
ans += totalCost - maxDel ; }
return ans ; }
var s = " " ;
var cost = [ 1 , 2 , 3 , 4 , 5 , 6 ] ;
document . write ( delCost ( s , cost ) ) ;
function checkXOR ( arr , N ) {
if ( N % 2 == 0 ) {
let xro = 0 ;
for ( let i = 0 ; i < N ; i ++ ) {
xro ^= arr [ i ] ; }
if ( xro != 0 ) { document . write ( - 1 + " " ) ; return ; }
for ( let i = 0 ; i < N - 3 ; i += 2 ) { document . write ( i + " " + ( i + 1 ) + " " + ( i + 2 ) + " " ) ; }
for ( let i = 0 ; i < N - 3 ; i += 2 ) { document . write ( i + " " + ( i + 1 ) + " " + ( N - 1 ) + " " ) ; } } else {
for ( let i = 0 ; i < N - 2 ; i += 2 ) { document . write ( i + " " + ( i + 1 ) + " " + ( i + 2 ) + " " ) ; }
for ( let i = 0 ; i < N - 2 ; i += 2 ) { document . write ( i + " " + ( i + 1 ) + " " + ( N - 1 ) + " " ) ; } } }
let arr = [ 4 , 2 , 1 , 7 , 2 ] ;
let N = arr . length ;
checkXOR ( arr , N ) ;
function make_array_element_even ( arr , N ) {
let res = 0 ;
let odd_cont_seg = 0 ;
for ( let i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 == 1 ) {
odd_cont_seg ++ ; } else { if ( odd_cont_seg > 0 ) {
if ( odd_cont_seg % 2 == 0 ) {
res += odd_cont_seg / 2 ; } else {
res += ( odd_cont_seg / 2 ) + 2 ; }
odd_cont_seg = 0 ; } } }
if ( odd_cont_seg > 0 ) {
if ( odd_cont_seg % 2 == 0 ) {
res += odd_cont_seg / 2 ; } else {
res += odd_cont_seg / 2 + 2 ; } }
return res ; }
let arr = [ 2 , 4 , 5 , 11 , 6 ] ; let N = arr . length ; document . write ( make_array_element_even ( arr , N ) ) ;
function zvalue ( nums ) {
var m = max_element ( nums ) ; var cnt = 0 ;
for ( i = 0 ; i <= m ; i ++ ) { cnt = 0 ;
for ( j = 0 ; j < nums . length ; j ++ ) {
if ( nums [ j ] >= i ) cnt ++ ; }
if ( cnt == i ) return i ; }
return - 1 ; }
nums = [ 7 , 8 , 9 , 0 , 0 , 1 ] ; document . write ( zvalue ( nums ) ) ;
function lexico_smallest ( s1 , s2 ) {
let M = new Map ( ) ; let S = new Set ( ) ; let pr ;
for ( let i = 0 ; i <= s1 . length - 1 ; ++ i ) {
if ( M . has ( s1 [ i ] ) ) { M [ s1 [ i ] ] ++ ; } else { M [ s1 [ i ] ] = 1 ; }
S . add ( s1 [ i ] ) ; }
for ( let i = 0 ; i <= s2 . length - 1 ; ++ i ) { if ( M . has ( s2 [ i ] ) ) { M [ s2 [ i ] ] -- ; } else { M [ s2 [ i ] ] = - 1 ; } } let c = s2 [ 0 ] ; let index = 0 ; let res = " " ;
S . forEach ( function ( x ) {
if ( x != c ) { for ( let i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } } else {
let j = 0 ; index = res . length ;
while ( s2 [ j ] == x ) { j ++ ; }
if ( s2 [ j ] < c ) { res += s2 ; for ( let i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } } else { for ( let i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } index += M [ x ] ; res += s2 ; } } } ) res = " " ; pr = [ res , index ] ;
return pr ; }
function lexico_largest ( s1 , s2 ) {
let pr = lexico_smallest ( s1 , s2 ) ;
let d1 = " " ; for ( let i = pr [ 1 ] - 1 ; i >= 0 ; i -- ) { d1 += pr [ 0 ] [ i ] ; }
let d2 = " " ; for ( let i = pr [ 0 ] . length - 1 ; i >= pr [ 1 ] + s2 . length ; -- i ) { d2 += pr [ 0 ] [ i ] ; } let res = d2 + s2 + d1 ;
return res ; }
let s1 = " " ; let s2 = " " ;
document . write ( lexico_smallest ( s1 , s2 ) [ 0 ] + " " ) ; document . write ( lexico_largest ( s1 , s2 ) ) ;
var sz = 100005 ;
var tree = Array . from ( Array ( sz ) , ( ) => Array ( ) )
var n ;
var vis = Array ( sz ) ;
var subtreeSize = Array ( sz ) ;
function addEdge ( a , b ) {
tree [ a ] . push ( b ) ;
tree [ b ] . push ( a ) ; }
function dfs ( x ) {
vis [ x ] = true ;
subtreeSize [ x ] = 1 ;
tree [ x ] . forEach ( i => { if ( ! vis [ i ] ) { dfs ( i ) ; subtreeSize [ x ] += subtreeSize [ i ] ; } } ) ; }
function countPairs ( a , b ) { var sub = Math . min ( subtreeSize [ a ] , subtreeSize [ b ] ) ; document . write ( sub * ( n - sub ) + " " ) ; }
n = 6 ; addEdge ( 0 , 1 ) ; addEdge ( 0 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 3 , 4 ) ; addEdge ( 3 , 5 ) ;
dfs ( 0 ) ;
countPairs ( 1 , 3 ) ; countPairs ( 0 , 2 ) ;
function findPermutation ( arr , N ) { var pos = arr . size + 1 ;
if ( pos > N ) return 1 ; var res = 0 ; for ( var i = 1 ; i <= N ; i ++ ) {
if ( ! arr . has ( i ) ) {
if ( i % pos == 0 pos % i == 0 ) {
arr . add ( i ) ;
res += findPermutation ( arr , N ) ;
arr . delete ( i ) ; } } }
return res ; }
var N = 5 ; var arr = new Set ( ) ; document . write ( findPermutation ( arr , N ) ) ;
function solve ( arr , n , X , Y ) {
var diff = Y - X ;
for ( var i = 0 ; i < n ; i ++ ) { if ( arr [ i ] != 1 ) { diff = diff % ( arr [ i ] - 1 ) ; } }
if ( diff == 0 ) document . write ( " " ) ; else document . write ( " " ) ; }
var arr = [ 1 , 2 , 7 , 9 , 10 ] ; var n = arr . length ; var X = 11 , Y = 13 ; solve ( arr , n , X , Y ) ;
let maxN = 100001 ;
let adj = new Array ( maxN ) ; adj . fill ( 0 ) ;
let height = new Array ( maxN ) ; height . fill ( 0 ) ;
let dist = new Array ( maxN ) ; dist . fill ( 0 ) ;
function addEdge ( u , v ) {
adj [ u ] . push ( v ) ;
adj [ v ] . push ( u ) ; }
function dfs1 ( cur , par ) {
for ( let u = 0 ; u < adj [ cur ] . length ; u ++ ) { if ( adj [ cur ] [ u ] != par ) {
dfs1 ( adj [ cur ] [ u ] , cur ) ;
height [ cur ] = Math . max ( height [ cur ] , height [ adj [ cur ] [ u ] ] ) ; } }
height [ cur ] += 1 ; }
function dfs2 ( cur , par ) { let max1 = 0 ; let max2 = 0 ;
for ( let u = 0 ; u < adj [ cur ] . length ; u ++ ) { if ( adj [ cur ] [ u ] != par ) {
if ( height [ adj [ cur ] [ u ] ] >= max1 ) { max2 = max1 ; max1 = height [ adj [ cur ] [ u ] ] ; } else if ( height [ adj [ cur ] [ u ] ] > max2 ) { max2 = height [ adj [ cur ] [ u ] ] ; } } } let sum = 0 ; for ( let u = 0 ; u < adj [ cur ] . length ; u ++ ) { if ( adj [ cur ] [ u ] != par ) {
sum = ( ( max1 == height [ adj [ cur ] [ u ] ] ) ? max2 : max1 ) ; if ( max1 == height [ adj [ cur ] [ u ] ] ) dist [ adj [ cur ] [ u ] ] = 1 + Math . max ( 1 + max2 , dist [ cur ] ) ; else dist [ adj [ cur ] [ u ] ] = 1 + Math . max ( 1 + max1 , dist [ cur ] ) ;
dfs2 ( adj [ cur ] [ u ] , cur ) ; } } }
let n = 6 ; for ( let i = 0 ; i < adj . length ; i ++ ) adj [ i ] = [ ] ; addEdge ( 1 , 2 ) ; addEdge ( 2 , 3 ) ; addEdge ( 2 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 5 , 6 ) ;
dfs1 ( 1 , 0 ) ;
dfs2 ( 1 , 0 ) ;
for ( let i = 1 ; i <= n ; i ++ ) document . write ( ( Math . max ( dist [ i ] , height [ i ] ) - 1 ) + " " ) ;
function middleOfThree ( a , b , c ) {
function middleOfThree ( $a , $b , $c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && a < c ) || ( c < a && a < b ) ) return a ; else return c ; }
let a = 20 , b = 30 , c = 40 ; document . write ( middleOfThree ( a , b , c ) ) ;
function selectionSort ( arr , n ) { let i , j , min_idx ;
for ( i = 0 ; i < n - 1 ; i ++ ) {
min_idx = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
let temp = arr [ min_idx ] ; arr [ min_idx ] = arr [ i ] ; arr [ i ] = temp ; } }
function printArray ( arr , size ) { let i ; for ( i = 0 ; i < size ; i ++ ) { document . write ( arr [ i ] + " " ) ; } document . write ( " " ) ; }
let arr = [ 64 , 25 , 12 , 22 , 11 ] ; let n = arr . length ;
selectionSort ( arr , n ) ; document . write ( " " ) ;
printArray ( arr , n ) ;
function checkStr1CanConStr2 ( str1 , str2 ) {
var N = str1 . length ;
var M = str2 . length ;
var st1 = new Set ( ) ;
var st2 = new Set ( ) ;
var hash1 = Array ( 256 ) . fill ( 0 ) ;
for ( var i = 0 ; i < N ; i ++ ) {
hash1 [ str1 [ i ] . charCodeAt ( 0 ) ] ++ ; }
for ( var i = 0 ; i < N ; i ++ ) {
st1 . add ( str1 [ i ] ) ; }
for ( var i = 0 ; i < M ; i ++ ) {
st2 . add ( str2 [ i ] ) ; }
if ( st1 . size != st2 . size ) { return false ; }
var hash2 = Array ( 256 ) . fill ( 0 ) ;
for ( var i = 0 ; i < M ; i ++ ) {
hash2 [ str2 [ i ] . charCodeAt ( 0 ) ] ++ ; }
hash1 . sort ( ( a , b ) => a - b ) ;
hash2 . sort ( ( a , b ) => a - b ) ;
for ( var i = 0 ; i < 256 ; i ++ ) {
if ( hash1 [ i ] != hash2 [ i ] ) { return false ; } } return true ; }
var str1 = " " ; var str2 = " " ; if ( checkStr1CanConStr2 ( str1 , str2 ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function partSort ( arr , N , a , b ) {
var l = Math . min ( a , b ) ; var r = Math . max ( a , b ) ;
for ( i = 0 ; i < N ; i ++ ) document . write ( arr [ i ] + " " ) ; }
var arr = [ 7 , 8 , 4 , 5 , 2 ] ; var a = 1 , b = 4 ; var N = arr . length ; partSort ( arr , N , a , b ) ;
let INF = Number . MAX_VALUE , N = 4 ;
function minCost ( cost ) {
let dist = new Array ( N ) ; dist . fill ( 0 ) ; for ( let i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( let i = 0 ; i < N ; i ++ ) for ( let j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) dist [ j ] = dist [ i ] + cost [ i ] [ j ] ; return dist [ N - 1 ] ; }
let cost = [ [ 0 , 15 , 80 , 90 ] , [ INF , 0 , 40 , 50 ] , [ INF , INF , 0 , 70 ] , [ INF , INF , INF , 0 ] ] ; document . write ( " " + " " + N + " " + minCost ( cost ) ) ;
function numOfways ( n , k ) { let p = 1 ; if ( k % 2 != 0 ) p = - 1 ; return ( Math . pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
let n = 4 , k = 2 ; document . write ( numOfways ( n , k ) ) ;
function largest_alphabet ( a , n ) {
let max = ' ' ;
for ( let i = 0 ; i < n ; i ++ ) if ( a [ i ] . charCodeAt ( ) > max . charCodeAt ( ) ) max = a [ i ] ;
return max ; }
function smallest_alphabet ( a , n ) {
let min = ' ' ;
for ( let i = 0 ; i < n - 1 ; i ++ ) if ( a [ i ] . charCodeAt ( ) < min . charCodeAt ( ) ) min = a [ i ] ;
return min ; }
let a = " " ;
let size = a . length ;
document . write ( " " ) ; document . write ( largest_alphabet ( a , size ) + " " ) ; document . write ( smallest_alphabet ( a , size ) ) ;
function maximumPalinUsingKChanges ( str , k ) { let palin = str . split ( " " ) ; let ans = " " ;
let l = 0 ; let r = str . length - 1 ;
while ( l < r ) {
if ( str [ l ] != str [ r ] ) { palin [ l ] = palin [ r ] = String . fromCharCode ( Math . max ( str . charAt ( l ) , str . charAt ( r ) ) ) ; k -- ; } l ++ ; r -- ; }
if ( k < 0 ) { return " " ; } l = 0 ; r = str . length - 1 ; while ( l <= r ) {
if ( l == r ) { if ( k > 0 ) { palin [ l ] = ' ' ; } }
if ( palin [ l ] < ' ' ) {
if ( k >= 2 && palin [ l ] == str [ l ] && palin [ r ] == str [ r ] ) { k -= 2 ; palin [ l ] = palin [ r ] = ' ' ; }
else if ( k >= 1 && ( palin [ l ] != str [ l ] palin [ r ] != str [ r ] ) ) { k -- ; palin [ l ] = palin [ r ] = ' ' ; } } l ++ ; r -- ; } for ( let i = 0 ; i < palin . length ; i ++ ) ans += palin [ i ] ; return ans ; }
let str = " " ; let k = 3 ; document . write ( maximumPalinUsingKChanges ( str , k ) ) ;
function countTriplets ( A ) {
var cnt = 0 ;
var tuples = new Map ( ) ;
A . forEach ( a => {
A . forEach ( b => { if ( tuples . has ( a & b ) ) tuples . set ( a & b , tuples . get ( a & b ) + 1 ) else tuples . set ( a & b , 1 ) } ) ; } ) ;
A . forEach ( a => {
tuples . forEach ( ( value , key ) => {
if ( ( key & a ) == 0 ) cnt += value ; } ) ; } ) ;
return cnt ; }
var A = [ 2 , 1 , 3 ] ;
document . write ( countTriplets ( A ) ) ;
var min = 10000 ;
function parity ( even , odd , v , i ) {
if ( i == v . length even . length == 0 && odd . length == 0 ) { var count = 0 ; for ( var j = 0 ; j < v . length - 1 ; j ++ ) { if ( v [ j ] % 2 != v [ j + 1 ] % 2 ) count ++ ; } if ( count < min ) min = count ; return min ; }
if ( v [ i ] != - 1 ) min = parity ( even , odd , v , i + 1 ) ;
else { if ( even . length != 0 ) { var x = even . back ( ) ; even . pop ( ) ; v [ i ] = x ; min = parity ( even , odd , v , i + 1 ) ;
even . push ( x ) ; } if ( odd . length != 0 ) { var x = odd [ odd . length - 1 ] ; odd . pop ( ) ; v [ i ] = x ; min = parity ( even , odd , v , i + 1 ) ;
odd . push ( x ) ; } } return min ; }
function minDiffParity ( v , n ) {
var even = [ ] ;
var odd = [ ] ; var m = new Map ( ) ; for ( var i = 1 ; i <= n ; i ++ ) m . set ( i , 1 ) ; for ( var i = 0 ; i < v . length ; i ++ ) {
if ( v [ i ] != - 1 ) m . delete ( v [ i ] ) ; }
m . forEach ( ( value , key ) => { if ( i . first % 2 == 0 ) even . push ( key ) ; else odd . push ( key ) ; } ) ; min = parity ( even , odd , v , 0 ) ; document . write ( min ) ; }
var n = 8 ; var v = [ 2 , 1 , 4 , - 1 , - 1 , 6 , - 1 , 8 ] ; minDiffParity ( v , n ) ;
let MAX = 100005 ; let adjacent = [ ] ; let visited = new Array ( MAX ) ;
let startnode , endnode , thirdnode ; let maxi = - 1 , N ;
let parent = new Array ( MAX ) ;
let vis = new Array ( MAX ) ;
function dfs ( u , count ) { visited [ u ] = true ; let temp = 0 ; for ( let i = 0 ; i < adjacent [ u ] . length ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] ) { temp ++ ; dfs ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; startnode = u ; } } }
function dfs1 ( u , count ) { visited [ u ] = true ; let temp = 0 ; for ( let i = 0 ; i < adjacent [ u ] . length ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] ) { temp ++ ; parent [ adjacent [ u ] [ i ] ] = u ; dfs1 ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; endnode = u ; } } }
function dfs2 ( u , count ) { visited [ u ] = true ; let temp = 0 ; for ( let i = 0 ; i < adjacent [ u ] . length ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] && ! vis [ adjacent [ u ] [ i ] ] ) { temp ++ ; dfs2 ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; thirdnode = u ; } } }
function findNodes ( ) {
dfs ( 1 , 0 ) ; for ( let i = 0 ; i <= N ; i ++ ) visited [ i ] = false ; maxi = - 1 ;
dfs1 ( startnode , 0 ) ; for ( let i = 0 ; i <= N ; i ++ ) visited [ i ] = false ;
let x = endnode ; vis [ startnode ] = true ;
while ( x != startnode ) { vis [ x ] = true ; x = parent [ x ] ; } maxi = - 1 ;
for ( let i = 1 ; i <= N ; i ++ ) { if ( vis [ i ] ) dfs2 ( i , 0 ) ; } }
for ( let i = 0 ; i < MAX ; i ++ ) adjacent . push ( [ ] ) ; N = 4 ; adjacent [ 1 ] . push ( 2 ) ; adjacent [ 2 ] . push ( 1 ) ; adjacent [ 1 ] . push ( 3 ) ; adjacent [ 3 ] . push ( 1 ) ; adjacent [ 1 ] . push ( 4 ) ; adjacent [ 4 ] . push ( 1 ) ; findNodes ( ) ; document . write ( " " + startnode + " " + endnode + " " + thirdnode + " " ) ;
function newvol ( x ) { document . write ( " " + " " + ( Math . pow ( x , 3 ) / 10000 + 3 * x + ( 3 * Math . pow ( x , 2 ) ) / 100 ) + " " ) ; }
var x = 10 ; newvol ( x ) ;
function length_of_chord ( r , x ) { document . write ( " " + " " + 2 * r * Math . sin ( x * ( 3.14 / 180 ) ) + " " ) ; }
let r = 4 , x = 63 ; length_of_chord ( r , x ) ;
function area ( a ) {
if ( a < 0 ) return - 1 ;
var area = Math . sqrt ( a ) / 6 ; return area ; }
var a = 10 ; document . write ( area ( a ) . toFixed ( 6 ) ) ;
function longestRodInCuboid ( length , breadth , height ) { let result ; let temp ;
temp = length * length + breadth * breadth + height * height ;
result = Math . sqrt ( temp ) ; return result ; }
let length = 12 , breadth = 9 , height = 8 ;
document . write ( longestRodInCuboid ( length , breadth , height ) ) ;
function LiesInsieRectangle ( a , b , x , y ) { if ( x - y - b <= 0 && x - y + b >= 0 && x + y - 2 * a + b <= 0 && x + y - b >= 0 ) return true ; return false ; }
let a = 7 , b = 2 , x = 4 , y = 5 ; if ( LiesInsieRectangle ( a , b , x , y ) ) document . write ( " " ) ; else document . write ( " " ) ;
function maxvolume ( s ) { let maxvalue = 0 ;
for ( let i = 1 ; i <= s - 2 ; i ++ ) {
for ( let j = 1 ; j <= s - 1 ; j ++ ) {
let k = s - i - j ;
maxvalue = Math . max ( maxvalue , i * j * k ) ; } } return maxvalue ; }
let s = 8 ; document . write ( maxvolume ( s ) ) ;
function maxvolume ( s ) {
let length = parseInt ( s / 3 ) ; s -= length ;
let breadth = parseInt ( s / 2 ) ;
let height = s - breadth ; return length * breadth * height ; }
let s = 8 ; document . write ( maxvolume ( s ) ) ;
function hexagonArea ( s ) { return ( ( 3 * Math . sqrt ( 3 ) * ( s * s ) ) / 2 ) ; }
let s = 4 ; document . write ( " " + hexagonArea ( s ) ) ;
function maxSquare ( b , m ) {
return ( b / m - 1 ) * ( b / m ) / 2 ; a }
let b = 10 , m = 2 ; document . write ( maxSquare ( b , m ) ) ;
function findRightAngle ( A , H ) {
let D = Math . pow ( H , 4 ) - 16 * A * A ; if ( D >= 0 ) {
let root1 = ( H * H + Math . sqrt ( D ) ) / 2 ; let root2 = ( H * H - Math . sqrt ( D ) ) / 2 ; let a = Math . sqrt ( root1 ) ; let b = Math . sqrt ( root2 ) ; if ( b >= a ) document . write ( a + " " + b + " " + H + " " ) ; else document . write ( b + " " + a + " " + H + " " ) ; } else document . write ( " " ) ; }
findRightAngle ( 6 , 5 ) ;
function numberOfSquares ( base ) {
base = ( base - 2 ) ;
base = Math . floor ( base / 2 ) ; return base * ( base + 1 ) / 2 ; }
let base = 8 ; document . write ( numberOfSquares ( base ) ) ;
function performQuery ( arr , Q ) {
for ( let i = 0 ; i < Q . length ; i ++ ) {
let or = 0 ;
let x = Q [ i ] [ 0 ] ; arr [ x - 1 ] = Q [ i ] [ 1 ] ;
for ( let j = 0 ; j < arr . length ; j ++ ) { or = or | arr [ j ] ; }
document . write ( or + " " ) ; } }
let arr = [ 1 , 2 , 3 ] ; let Q = [ [ 1 , 4 ] , [ 3 , 0 ] ] ; performQuery ( arr , Q ) ;
function smallest ( k , d ) { let cnt = 1 ; let m = d % k ;
let v = new Array ( k ) . fill ( 0 ) ; v [ m ] = 1 ;
while ( 1 ) { if ( m == 0 ) return cnt ; m = ( ( ( m * ( 10 % k ) ) % k ) + ( d % k ) ) % k ;
if ( v [ m ] == 1 ) return - 1 ; v [ m ] = 1 ; cnt ++ ; } return - 1 ; }
let d = 1 ; let k = 41 ; document . write ( smallest ( k , d ) ) ;
function fib ( n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
function findVertices ( n ) {
return fib ( n + 2 ) ; }
var n = 3 ; document . write ( findVertices ( n ) ) ;
function checkCommonDivisor ( arr , N , X ) {
var G = 0 ;
for ( i = 0 ; i < N ; i ++ ) { G = gcd ( G , arr [ i ] ) ; } var copy_G = G ; for ( divisor = 2 ; divisor <= X ; divisor ++ ) {
while ( G % divisor == 0 ) {
G = G / divisor ; } }
if ( G <= X ) { document . write ( " " ) ;
for ( i = 0 ; i < N ; i ++ ) document . write ( ( arr [ i ] / copy_G ) + " " ) ; document . write ( ) ; }
else document . write ( " " ) ; }
var arr = [ 6 , 15 , 6 ] ; var X = 6 ;
var N = arr . length ; checkCommonDivisor ( arr , N , X ) ;
class Node { constructor ( ) { this . data = 0 ; this . prev = null ; this . next = null ; } }
function reverse ( head_ref ) { var temp = null ; var current = head_ref ;
while ( current != null ) { temp = current . prev ; current . prev = current . next ; current . next = temp ; current = current . prev ; }
if ( temp != null ) head_ref = temp . prev ; return head_ref ; }
function merge ( first , second ) {
if ( first == null ) return second ;
if ( second == null ) return first ;
if ( first . data < second . data ) { first . next = merge ( first . next , second ) ; first . next . prev = first ; first . prev = null ; return first ; } else { second . next = merge ( first , second . next ) ; second . next . prev = second ; second . prev = null ; return second ; } }
function sort ( head ) {
if ( head == null head . next == null ) return head ; var current = head . next ; while ( current != null ) {
if ( current . data < current . prev . data ) break ;
current = current . next ; }
if ( current == null ) return head ;
current . prev . next = null ; current . prev = null ;
current = reverse ( current ) ;
return merge ( head , current ) ; }
function push ( head_ref , new_data ) {
var new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . prev = null ;
new_node . next = ( head_ref ) ;
if ( ( head_ref ) != null ) ( head_ref ) . prev = new_node ;
( head_ref ) = new_node ; return head_ref ; }
function printList ( head ) {
if ( head == null ) document . write ( " " ) ; while ( head != null ) { document . write ( head . data + " " ) ; head = head . next ; } }
var head = null ;
head = push ( head , 1 ) ; head = push ( head , 4 ) ; head = push ( head , 6 ) ; head = push ( head , 10 ) ; head = push ( head , 12 ) ; head = push ( head , 7 ) ; head = push ( head , 5 ) ; head = push ( head , 2 ) ; document . write ( " " ) ; printList ( head ) ;
head = sort ( head ) ; document . write ( " " ) ; printList ( head ) ;
class Node {
function printlist ( head ) { if ( head == null ) { document . write ( " " ) ; return ; } while ( head != null ) { document . write ( head . data + " " ) ; if ( head . next != null ) document . write ( " " ) ; head = head . next ; } document . write ( " " ) ; }
function isVowel ( x ) { return ( x == ' ' x == ' ' x == ' ' x == ' ' x == ' ' ) ; }
function arrange ( head ) { let newHead = head ;
let latestVowel ; let curr = head ;
if ( head == null ) return null ;
if ( isVowel ( head . data ) == true )
latestVowel = head ; else {
while ( curr . next != null && ! isVowel ( curr . next . data ) ) curr = curr . next ;
if ( curr . next == null ) return head ;
latestVowel = newHead = curr . next ; curr . next = curr . next . next ; latestVowel . next = head ; }
while ( curr != null && curr . next != null ) { if ( isVowel ( curr . next . data ) == true ) {
if ( curr == latestVowel ) {
latestVowel = curr = curr . next ; } else {
let temp = latestVowel . next ;
latestVowel . next = curr . next ;
latestVowel = latestVowel . next ;
curr . next = curr . next . next ;
latestVowel . next = temp ; } } else {
curr = curr . next ; } } return newHead ; }
let head = new Node ( ' ' ) ; head . next = new Node ( ' ' ) ; head . next . next = new Node ( ' ' ) ; head . next . next . next = new Node ( ' ' ) ; head . next . next . next . next = new Node ( ' ' ) ; head . next . next . next . next . next = new Node ( ' ' ) ; head . next . next . next . next . next . next = new Node ( ' ' ) ; head . next . next . next . next . next . next . next = new Node ( ' ' ) ; document . write ( " " ) ; printlist ( head ) ; head = arrange ( head ) ; document . write ( " " ) ; printlist ( head ) ;
function newNode ( data ) { var temp = new Node ( ) ; temp . data = data ; temp . right = null ; temp . left = null ; return temp ; } function KthLargestUsingMorrisTraversal ( root , k ) { var curr = root ; var Klargest = null ;
var count = 0 ; while ( curr != null ) {
if ( curr . right == null ) {
if ( ++ count == k ) Klargest = curr ;
curr = curr . left ; } else {
var succ = curr . right ; while ( succ . left != null && succ . left != curr ) succ = succ . left ; if ( succ . left == null ) {
succ . left = curr ;
curr = curr . right ; }
else { succ . left = null ; if ( ++ count == k ) Klargest = curr ;
curr = curr . left ; } } } return Klargest ; }
root = newNode ( 4 ) ; root . left = newNode ( 2 ) ; root . right = newNode ( 7 ) ; root . left . left = newNode ( 1 ) ; root . left . right = newNode ( 3 ) ; root . right . left = newNode ( 6 ) ; root . right . right = newNode ( 10 ) ; document . write ( " " + KthLargestUsingMorrisTraversal ( root , 2 ) . data ) ;
let MAX_SIZE = 10 ;
function sortByRow ( mat , n , ascending ) { for ( let i = 0 ; i < n ; i ++ ) { if ( ascending ) mat [ i ] . sort ( function ( a , b ) { return a - b ; } ) ; else mat [ i ] . sort ( function ( a , b ) { return b - a ; } ) ; } }
function transpose ( mat , n ) { for ( let i = 0 ; i < n ; i ++ ) for ( let j = i + 1 ; j < n ; j ++ ) {
let temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ j ] [ i ] ; mat [ j ] [ i ] = temp ; } }
function sortMatRowAndColWise ( mat , n ) {
sortByRow ( mat , n , true ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n , false ) ;
transpose ( mat , n ) ; }
function printMat ( mat , n ) { for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) document . write ( mat [ i ] [ j ] + " " ) ; document . write ( " " ) ; } }
let n = 3 ; let mat = [ [ 3 , 2 , 1 ] , [ 9 , 8 , 7 ] , [ 6 , 5 , 4 ] ] ; document . write ( " " ) ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; document . write ( " " ) ; printMat ( mat , n ) ;
let MAX_SIZE = 10 ;
function sortByRow ( mat , n ) { for ( let i = 0 ; i < n ; i ++ )
mat [ i ] . sort ( function ( a , b ) { return a - b ; } ) ; }
function transpose ( mat , n ) { for ( let i = 0 ; i < n ; i ++ ) for ( let j = i + 1 ; j < n ; j ++ ) {
let temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ j ] [ i ] ; mat [ j ] [ i ] = temp ; } }
function sortMatRowAndColWise ( mat , n ) {
sortByRow ( mat , n ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n ) ;
transpose ( mat , n ) ; }
function printMat ( mat , n ) { for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) document . write ( mat [ i ] [ j ] + " " ) ; document . write ( " " ) ; } }
let mat = [ [ 4 , 1 , 3 ] , [ 9 , 6 , 8 ] , [ 5 , 2 , 7 ] ] ; let n = 3 ; document . write ( " " ) ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; document . write ( " " ) ; printMat ( mat , n ) ;
function doublyEven ( n ) { var arr = Array ( n ) . fill ( 0 ) . map ( x => Array ( n ) . fill ( 0 ) ) ; var i , j ;
for ( i = 0 ; i < n ; i ++ ) for ( j = 0 ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * i ) + j + 1 ;
for ( i = 0 ; i < parseInt ( n / 4 ) ; i ++ ) for ( j = 0 ; j < parseInt ( n / 4 ) ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 0 ; i < parseInt ( n / 4 ) ; i ++ ) for ( j = 3 * ( parseInt ( n / 4 ) ) ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 3 * parseInt ( n / 4 ) ; i < n ; i ++ ) for ( j = 0 ; j < parseInt ( n / 4 ) ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 3 * parseInt ( n / 4 ) ; i < n ; i ++ ) for ( j = 3 * parseInt ( n / 4 ) ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = parseInt ( n / 4 ) ; i < 3 * parseInt ( n / 4 ) ; i ++ ) for ( j = parseInt ( n / 4 ) ; j < 3 * parseInt ( n / 4 ) ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = 0 ; j < n ; j ++ ) document . write ( arr [ i ] [ j ] + " " ) ; document . write ( ' ' ) ; } }
var n = 8 ;
doublyEven ( n ) ;
let cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
function Kroneckerproduct ( A , B ) { let C = new Array ( rowa * rowb ) for ( let i = 0 ; i < ( rowa * rowb ) ; i ++ ) { C [ i ] = new Array ( cola * colb ) ; for ( let j = 0 ; j < ( cola * colb ) ; j ++ ) { C [ i ] [ j ] = 0 ; } }
for ( let i = 0 ; i < rowa ; i ++ ) {
for ( let k = 0 ; k < rowb ; k ++ ) {
for ( let j = 0 ; j < cola ; j ++ ) {
for ( let l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; document . write ( C [ i + l + 1 ] [ j + k + 1 ] + " " ) ; } } document . write ( " " ) ; } } }
let A = [ [ 1 , 2 ] , [ 3 , 4 ] , [ 1 , 0 ] ] ; let B = [ [ 0 , 5 , 2 ] , [ 6 , 7 , 3 ] ] ; Kroneckerproduct ( A , B ) ;
function isLowerTriangularMatrix ( mat ) { for ( let i = 0 ; i < N ; i ++ ) for ( let j = i + 1 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != 0 ) return false ; return true ; }
let mat = [ [ 1 , 0 , 0 , 0 ] , [ 1 , 4 , 0 , 0 ] , [ 4 , 6 , 2 , 0 ] , [ 0 , 4 , 7 , 6 ] ] ;
if ( isLowerTriangularMatrix ( mat ) ) document . write ( " " ) ; else document . write ( " " ) ;
function isUpperTriangularMatrix ( mat ) { for ( let i = 1 ; i < N ; i ++ ) for ( let j = 0 ; j < i ; j ++ ) if ( mat [ i ] [ j ] != 0 ) return false ; return true ; }
let mat = [ [ 1 , 3 , 5 , 3 ] , [ 0 , 4 , 6 , 2 ] , [ 0 , 0 , 2 , 5 ] , [ 0 , 0 , 0 , 6 ] ] ; if ( isUpperTriangularMatrix ( mat ) ) document . write ( " " ) ; else document . write ( " " ) ;
var m = 3 ;
var n = 2 ;
function countSets ( a ) {
var res = 0 ;
for ( i = 0 ; i < n ; i ++ ) { var u = 0 , v = 0 ; for ( j = 0 ; j < m ; j ++ ) { if ( a [ i ] [ j ] == 1 ) u ++ ; else v ++ ; } res += Math . pow ( 2 , u ) - 1 + Math . pow ( 2 , v ) - 1 ; }
for ( i = 0 ; i < m ; i ++ ) { var u = 0 , v = 0 ; for ( j = 0 ; j < n ; j ++ ) { if ( a [ j ] [ i ] == 1 ) u ++ ; else v ++ ; } res += Math . pow ( 2 , u ) - 1 + Math . pow ( 2 , v ) - 1 ; }
return res - ( n * m ) ; }
var a = [ [ 1 , 0 , 1 ] , [ 0 , 1 , 0 ] ] ; document . write ( countSets ( a ) ) ;
function transpose ( mat , tr , N ) { for ( let i = 0 ; i < N ; i ++ ) for ( let j = 0 ; j < N ; j ++ ) tr [ i ] [ j ] = mat [ j ] [ i ] ; }
function isSymmetric ( mat , N ) { let tr = new Array ( N ) ; for ( let i = 0 ; i < N ; i ++ ) { tr [ i ] = new Array ( MAX ) ; } transpose ( mat , tr , N ) ; for ( let i = 0 ; i < N ; i ++ ) for ( let j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != tr [ i ] [ j ] ) return false ; return true ; }
let mat = [ [ 1 , 3 , 5 ] , [ 3 , 2 , 4 ] , [ 5 , 4 , 1 ] ] ; if ( isSymmetric ( mat , 3 ) ) document . write ( " " ) ; else document . write ( " " ) ;
function isSymmetric ( mat , N ) { for ( let i = 0 ; i < N ; i ++ ) for ( let j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != mat [ j ] [ i ] ) return false ; return true ; }
let mat = [ [ 1 , 3 , 5 ] , [ 3 , 2 , 4 ] , [ 5 , 4 , 1 ] ] ; if ( isSymmetric ( mat , 3 ) ) document . write ( " " ) ; else document . write ( " " ) ;
var MAX = 100 ;
function findNormal ( mat , n ) { var sum = 0 ; for ( var i = 0 ; i < n ; i ++ ) for ( var j = 0 ; j < n ; j ++ ) sum += mat [ i ] [ j ] * mat [ i ] [ j ] ; return parseInt ( Math . sqrt ( sum ) ) ; }
function findTrace ( mat , n ) { var sum = 0 ; for ( var i = 0 ; i < n ; i ++ ) sum += mat [ i ] [ i ] ; return sum ; }
var mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] ; document . write ( " " + findTrace ( mat , 5 ) + " " ) ; document . write ( " " + findNormal ( mat , 5 ) ) ;
function maxDet ( n ) { return ( 2 * n * n * n ) ; }
function resMatrix ( n ) { for ( let i = 0 ; i < 3 ; i ++ ) { for ( let j = 0 ; j < 3 ; j ++ ) {
if ( i == 0 && j == 2 ) document . write ( " " ) ; else if ( i == 1 && j == 0 ) document . write ( " " ) ; else if ( i == 2 && j == 1 ) document . write ( " " ) ;
else document . write ( n + " " ) ; } document . write ( " " ) ; } }
let n = 15 ; document . write ( " " + maxDet ( n ) + " " ) ; document . write ( " " ) ; resMatrix ( n ) ;
function countNegative ( M , n , m ) { let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < m ; j ++ ) { if ( M [ i ] [ j ] < 0 ) count += 1 ;
else break ; } } return count ; }
let M = [ [ - 3 , - 2 , - 1 , 1 ] , [ - 2 , 2 , 3 , 4 ] , [ 4 , 5 , 7 , 8 ] ] ; document . write ( countNegative ( M , 3 , 4 ) ) ;
function countNegative ( M , n , m ) {
let count = 0 ;
let i = 0 ; let j = m - 1 ;
while ( j >= 0 && i < n ) { if ( M [ i ] [ j ] < 0 ) {
count += j + 1 ;
i += 1 ; }
else j -= 1 ; } return count ; } `
let M = [ [ - 3 , - 2 , - 1 , 1 ] , [ - 2 , 2 , 3 , 4 ] , [ 4 , 5 , 7 , 8 ] ] ; document . write ( countNegative ( M , 3 , 4 ) ) ;
function findMaxValue ( N , mat ) {
let maxValue = Number . MIN_VALUE ;
for ( let a = 0 ; a < N - 1 ; a ++ ) for ( let b = 0 ; b < N - 1 ; b ++ ) for ( let d = a + 1 ; d < N ; d ++ ) for ( let e = b + 1 ; e < N ; e ++ ) if ( maxValue < ( mat [ d ] [ e ] - mat [ a ] [ b ] ) ) maxValue = mat [ d ] [ e ] - mat [ a ] [ b ] ; return maxValue ; }
let N = 5 ; let mat = [ [ 1 , 2 , - 1 , - 4 , - 20 ] , [ - 8 , - 3 , 4 , 2 , 1 ] , [ 3 , 8 , 6 , 1 , 3 ] , [ - 4 , - 1 , 1 , 7 , - 6 ] , [ 0 , - 4 , 10 , - 5 , 1 ] ] ; document . write ( " " + findMaxValue ( N , mat ) ) ;
function findMaxValue ( N , mat ) {
let maxValue = Number . MIN_VALUE ;
let maxArr = new Array ( N ) ; for ( let i = 0 ; i < N ; i ++ ) { maxArr [ i ] = new Array ( N ) ; }
maxArr [ N - 1 ] [ N - 1 ] = mat [ N - 1 ] [ N - 1 ] ;
let maxv = mat [ N - 1 ] [ N - 1 ] ; for ( let j = N - 2 ; j >= 0 ; j -- ) { if ( mat [ N - 1 ] [ j ] > maxv ) maxv = mat [ N - 1 ] [ j ] ; maxArr [ N - 1 ] [ j ] = maxv ; }
maxv = mat [ N - 1 ] [ N - 1 ] ; for ( let i = N - 2 ; i >= 0 ; i -- ) { if ( mat [ i ] [ N - 1 ] > maxv ) maxv = mat [ i ] [ N - 1 ] ; maxArr [ i ] [ N - 1 ] = maxv ; }
for ( let i = N - 2 ; i >= 0 ; i -- ) { for ( let j = N - 2 ; j >= 0 ; j -- ) {
if ( maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] > maxValue ) maxValue = maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] ;
maxArr [ i ] [ j ] = Math . max ( mat [ i ] [ j ] , Math . max ( maxArr [ i ] [ j + 1 ] , maxArr [ i + 1 ] [ j ] ) ) ; } } return maxValue ; }
let N = 5 ; let mat = [ [ 1 , 2 , - 1 , - 4 , - 20 ] , [ - 8 , - 3 , 4 , 2 , 1 ] , [ 3 , 8 , 6 , 1 , 3 ] , [ - 4 , - 1 , 1 , 7 , - 6 ] , [ 0 , - 4 , 10 , - 5 , 1 ] ] ; document . write ( " " + findMaxValue ( N , mat ) ) ;
let INF = Number . MAX_VALUE ; let N = 4 ;
function youngify ( mat , i , j ) {
let downVal = ( i + 1 < N ) ? mat [ i + 1 ] [ j ] : INF ; let rightVal = ( j + 1 < N ) ? mat [ i ] [ j + 1 ] : INF ;
if ( downVal == INF && rightVal == INF ) { return ; }
if ( downVal < rightVal ) { mat [ i ] [ j ] = downVal ; mat [ i + 1 ] [ j ] = INF ; youngify ( mat , i + 1 , j ) ; } else { mat [ i ] [ j ] = rightVal ; mat [ i ] [ j + 1 ] = INF ; youngify ( mat , i , j + 1 ) ; } }
function extractMin ( mat ) { let ret = mat [ 0 ] [ 0 ] ; mat [ 0 ] [ 0 ] = INF ; youngify ( mat , 0 , 0 ) ; return ret ; }
function printSorted ( mat ) { document . write ( " " ) ; for ( let i = 0 ; i < N * N ; i ++ ) { document . write ( extractMin ( mat ) + " " ) ; } } let mat = [ [ 10 , 20 , 30 , 40 ] , [ 15 , 25 , 35 , 45 ] , [ 27 , 29 , 37 , 48 ] , [ 32 , 33 , 39 , 50 ] ] ; printSorted ( mat ) ;
let n = 5 ;
function printSumSimple ( mat , k ) {
if ( k > n ) return ;
for ( let i = 0 ; i < n - k + 1 ; i ++ ) {
for ( let j = 0 ; j < n - k + 1 ; j ++ ) {
let sum = 0 ; for ( let p = i ; p < k + i ; p ++ ) for ( let q = j ; q < k + j ; q ++ ) sum += mat [ p ] [ q ] ; document . write ( sum + " " ) ; }
document . write ( " " ) ; } }
let mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] let k = 3 ; printSumSimple ( mat , k ) ;
let n = 5 ;
function printSumTricky ( mat , k ) {
if ( k > n ) return ;
let stripSum = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { stripSum [ i ] = new Array ( n ) ; } for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) { stripSum [ i ] [ j ] = 0 ; } }
for ( let j = 0 ; j < n ; j ++ ) {
let sum = 0 ; for ( let i = 0 ; i < k ; i ++ ) sum += mat [ i ] [ j ] ; stripSum [ 0 ] [ j ] = sum ;
for ( let i = 1 ; i < n - k + 1 ; i ++ ) { sum += ( mat [ i + k - 1 ] [ j ] - mat [ i - 1 ] [ j ] ) ; stripSum [ i ] [ j ] = sum ; } }
for ( let i = 0 ; i < n - k + 1 ; i ++ ) {
let sum = 0 ; for ( let j = 0 ; j < k ; j ++ ) sum += stripSum [ i ] [ j ] ; document . write ( sum + " " ) ;
for ( let j = 1 ; j < n - k + 1 ; j ++ ) { sum += ( stripSum [ i ] [ j + k - 1 ] - stripSum [ i ] [ j - 1 ] ) ; document . write ( sum + " " ) ; } document . write ( " " ) ; } }
let mat = [ [ 1 , 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 , 4 ] , [ 5 , 5 , 5 , 5 , 5 ] ] ; let k = 3 ; printSumTricky ( mat , k ) ;
var M = 3 ; var N = 4 ;
function transpose ( A , B ) { var i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < M ; j ++ ) B [ i ] [ j ] = A [ j ] [ i ] ; }
var A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] ] ; var B = Array ( N ) ; for ( i = 0 ; i < N ; i ++ ) B [ i ] = Array ( M ) . fill ( 0 ) ; transpose ( A , B ) ; document . write ( " " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < M ; j ++ ) document . write ( B [ i ] [ j ] + " " ) ; document . write ( " " ) ; }
var N = 4 ;
function transpose ( A ) { for ( i = 0 ; i < N ; i ++ ) for ( j = i + 1 ; j < N ; j ++ ) { var temp = A [ i ] [ j ] ; A [ i ] [ j ] = A [ j ] [ i ] ; A [ j ] [ i ] = temp ; } }
var A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] ; transpose ( A ) ; document . write ( " " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) document . write ( A [ i ] [ j ] + " " ) ; document . write ( " \< " ) ; }
let R = 3 ; let C = 3 ;
function pathCountRec ( mat , m , n , k ) {
if ( m < 0 n < 0 ) return 0 ; if ( m == 0 && n == 0 ) return ( k == mat [ m ] [ n ] ) ;
return pathCountRec ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountRec ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ; }
function pathCount ( mat , k ) { return pathCountRec ( mat , R - 1 , C - 1 , k ) ; }
let k = 12 ; let mat = [ [ 1 , 2 , 3 ] , [ 4 , 6 , 5 ] , [ 3 , 2 , 1 ] ] ; document . write ( pathCount ( mat , k ) ) ;
var R = 3 ; var C = 3 ; var MAX_K = 100 ; var dp = Array ( R ) . fill ( ) . map ( ( ) => Array ( C ) . fill ( ) . map ( ( ) => Array ( MAX_K ) . fill ( 0 ) ) ) ; function pathCountDPRecDP ( mat , m , n , k ) {
if ( m < 0 n < 0 ) return 0 ; if ( m == 0 && n == 0 ) return ( k == mat [ m ] [ n ] ? 1 : 0 ) ;
if ( dp [ m ] [ n ] [ k ] != - 1 ) return dp [ m ] [ n ] [ k ] ;
dp [ m ] [ n ] [ k ] = pathCountDPRecDP ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountDPRecDP ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ; return dp [ m ] [ n ] [ k ] ; }
function pathCountDP ( mat , k ) { for ( i = 0 ; i < R ; i ++ ) for ( j = 0 ; j < C ; j ++ ) for ( l = 0 ; l < MAX_K ; l ++ ) dp [ i ] [ j ] [ l ] = - 1 ; return pathCountDPRecDP ( mat , R - 1 , C - 1 , k ) ; }
var k = 12 ; var mat = [ [ 1 , 2 , 3 ] , [ 4 , 6 , 5 ] , [ 3 , 2 , 1 ] ] ; document . write ( pathCountDP ( mat , k ) ) ;
let SIZE = 10
function sortMat ( mat , n ) {
let temp = new Array ( n * n ) ; let k = 0 ;
for ( let i = 0 ; i < n ; i ++ ) for ( let j = 0 ; j < n ; j ++ ) temp [ k ++ ] = mat [ i ] [ j ] ;
temp . sort ( ) ;
k = 0 ; for ( let i = 0 ; i < n ; i ++ ) for ( let j = 0 ; j < n ; j ++ ) mat [ i ] [ j ] = temp [ k ++ ] ; }
function printMat ( mat , n ) { for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) document . write ( mat [ i ] [ j ] + " " ) ; document . write ( " " ) ; } }
let mat = [ [ 5 , 4 , 7 ] , [ 1 , 3 , 8 ] , [ 2 , 9 , 6 ] ] ; let n = 3 ; document . write ( " " + " " ) ; printMat ( mat , n ) ; sortMat ( mat , n ) ; document . write ( " " ) ; document . write ( " " + " " ) ; printMat ( mat , n ) ;
function findCrossOver ( arr , low , high , x ) {
if ( arr [ high ] <= x ) return high
if ( arr [ low ] > x ) return low
var mid = ( low + high ) / 2
if ( arr [ mid ] <= x && arr [ mid + 1 ] > x ) return mid
if ( arr [ mid ] < x ) return findCrossOver ( arr , mid + 1 , high , x ) return findCrossOver ( arr , low , mid - 1 , x ) }
function printKclosest ( arr , x , k , n ) {
var l = findCrossOver ( arr , 0 , n - 1 , x )
var r = l + 1
var count = 0
if ( arr [ l ] == x ) l -= 1
while ( l >= 0 && r < n && count < k ) { if ( x - arr [ l ] < arr [ r ] - x ) { document . write ( arr [ l ] + " " ) l -= 1 } else { document . write ( arr [ r ] + " " ) r += 1 } count += 1 }
while ( count < k && l >= 0 ) { print ( arr [ l ] ) l -= 1 count += 1 }
while ( count < k && r < n ) { print ( arr [ r ] ) r += 1 count += 1 } }
var arr = [ 12 , 16 , 22 , 30 , 35 , 39 , 42 , 45 , 48 , 50 , 53 , 55 , 56 ] var n = arr . length var x = 35 var k = 4 printKclosest ( arr , x , 4 , n )
var head = null ; var sorted = null ; class node { constructor ( val ) { this . val = val ; this . next = null ; } }
function push ( val ) {
var newnode = new node ( val ) ;
newnode . next = head ;
head = newnode ; }
function insertionSort ( headref ) {
var sorted = null ; var current = headref ;
while ( current != null ) {
var next = current . next ;
sortedInsert ( current ) ;
current = next ; }
head = sorted ; }
function sortedInsert ( newnode ) {
if ( sorted == null sorted . val >= newnode . val ) { newnode . next = sorted ; sorted = newnode ; } else { var current = sorted ;
while ( current . next != null && current . next . val < newnode . val ) { current = current . next ; } newnode . next = current . next ; current . next = newnode ; } }
function printlist ( head ) { while ( head != null ) { document . write ( head . val + " " ) ; head = head . next ; } }
push ( 5 ) ; push ( 20 ) ; push ( 4 ) ; push ( 3 ) ; push ( 30 ) ; document . write ( " " ) ; printlist ( head ) ; insertionSort ( head ) ; document . write ( " " ) ; printlist ( sorted ) ;
function count ( S , m , n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
var arr = [ 1 , 2 , 3 ] ; var m = arr . length ; document . write ( count ( arr , m , 4 ) ) ;
function count ( S , m , n ) {
let table = new Array ( n + 1 ) ; table . fill ( 0 ) ;
table [ 0 ] = 1 ;
for ( let i = 0 ; i < m ; i ++ ) for ( let j = S [ i ] ; j <= n ; j ++ ) table [ j ] += table [ j - S [ i ] ] ; return table [ n ] ; }
let arr = [ 1 , 2 , 3 ] ; let m = arr . length ; let n = 4 ; document . write ( count ( arr , m , n ) ) ;
let dp = new Array ( 100 ) ; for ( var i = 0 ; i < dp . length ; i ++ ) { dp [ i ] = new Array ( 2 ) ; }
function matrixChainMemoised ( p , i , j ) { if ( i == j ) { return 0 ; } if ( dp [ i ] [ j ] != - 1 ) { return dp [ i ] [ j ] ; } dp [ i ] [ j ] = Number . MAX_VALUE ; for ( let k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i ] [ j ] ; } function MatrixChainOrder ( p , n ) { let i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
let arr = [ 1 , 2 , 3 , 4 ] ; let n = arr . length ; for ( var i = 0 ; i < dp . length ; i ++ ) { for ( var j = 0 ; j < dp . length ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } document . write ( " " + MatrixChainOrder ( arr , n ) ) ;
function MatrixChainOrder ( p , n ) {
var m = Array ( n ) . fill ( 0 ) . map ( x => Array ( n ) . fill ( 0 ) ) ; var i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i ] [ i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; if ( j == n ) continue ; m [ i ] [ j ] = Number . MAX_VALUE ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i ] [ j ] ) m [ i ] [ j ] = q ; } } } return m [ 1 ] [ n - 1 ] ; }
var arr = [ 1 , 2 , 3 , 4 ] ; var size = arr . length ; document . write ( " " + MatrixChainOrder ( arr , size ) ) ;
function cutRod ( price , n ) { if ( n <= 0 ) return 0 ; let max_val = Number . MIN_VALUE ;
for ( let i = 0 ; i < n ; i ++ ) max_val = Math . max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) ; return max_val ; }
let arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] ; let size = arr . length ; document . write ( " " + cutRod ( arr , size ) ) ;
function cutRod ( price , n ) { let val = new Array ( n + 1 ) ; val [ 0 ] = 0 ;
for ( let i = 1 ; i <= n ; i ++ ) { let max_val = Number . MIN_VALUE ; for ( let j = 0 ; j < i ; j ++ ) max_val = Math . max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
let arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] ; let size = arr . length ; document . write ( " " + cutRod ( arr , size ) + " " ) ;
function multiply ( x , y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; }
document . write ( multiply ( 5 , - 11 ) ) ;
function sieveOfEratosthenes ( n ) {
prime = Array . from ( { length : n + 1 } , ( _ , i ) => true ) ; for ( p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( i = 2 ; i <= n ; i ++ ) { if ( prime [ i ] == true ) document . write ( i + " " ) ; } }
var n = 30 ; document . write ( " " ) ; document . write ( " " + n + " " ) ; sieveOfEratosthenes ( n ) ;
function binomialCoeff ( n , k ) { let res = 1 ; if ( k > n - k ) k = n - k ; for ( let i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
function printPascal ( n ) {
for ( let line = 0 ; line < n ; line ++ ) {
for ( let i = 0 ; i <= line ; i ++ ) document . write ( binomialCoeff ( line , i ) + " " ) ; document . write ( " " ) ; } }
let n = 7 ; printPascal ( n ) ;
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
function getModulo ( n , d ) { return ( n & ( d - 1 ) ) ; }
n = 6 ;
d = 4 ; document . write ( n + " " + d + " " + getModulo ( n , d ) ) ;
function countSetBits ( n ) { var count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
var i = 9 ; document . write ( countSetBits ( i ) ) ;
function countSetBits ( n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
var n = 9 ;
document . write ( countSetBits ( n ) ) ;
var BitsSetTable256 = Array . from ( { length : 256 } , ( _ , i ) => 0 ) ;
function initialize ( ) {
BitsSetTable256 [ 0 ] = 0 ; for ( var i = 0 ; i < 256 ; i ++ ) { BitsSetTable256 [ i ] = ( i & 1 ) + BitsSetTable256 [ parseInt ( i / 2 ) ] ; } }
function countSetBits ( n ) { return ( BitsSetTable256 [ n & 0xff ] + BitsSetTable256 [ ( n >> 8 ) & 0xff ] + BitsSetTable256 [ ( n >> 16 ) & 0xff ] + BitsSetTable256 [ n >> 24 ] ) ; }
initialize ( ) ; var n = 9 ; document . write ( countSetBits ( n ) ) ;
document . write ( ( 4 ) . toString ( 2 ) . split ( ' ' ) . filter ( x => x == ' ' ) . length + " " ) ; document . write ( ( 15 ) . toString ( 2 ) . split ( ' ' ) . filter ( x => x == ' ' ) . length ) ;
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
function powerOf2 ( n ) {
if ( n == 1 ) return true ;
else if ( n % 2 != 0 n == 0 ) return false ;
return powerOf2 ( n / 2 ) ; }
var n = 64 ;
var m = 12 ; if ( powerOf2 ( n ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ; if ( powerOf2 ( m ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function isPowerOfTwo ( x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
document . write ( isPowerOfTwo ( 31 ) ? " " : " " ) ; document . write ( " " + ( isPowerOfTwo ( 64 ) ? " " : " " ) ) ;
function maxRepeating ( arr , n , k ) {
for ( let i = 0 ; i < n ; i ++ ) arr [ arr [ i ] % k ] += k ;
let max = arr [ 0 ] , result = 0 ; for ( let i = 1 ; i < n ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; result = i ; } }
return result ; }
let arr = [ 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 ] ; let n = arr . length ; let k = 8 ; document . write ( " " + maxRepeating ( arr , n , k ) + " " ) ;
function fun ( x ) { let y = parseInt ( x / 4 ) * 4 ;
let ans = 0 ; for ( let i = y ; i <= x ; i ++ ) ans ^= i ; return ans ; }
function query ( x ) {
if ( x == 0 ) return 0 ; let k = parseInt ( ( x + 1 ) / 2 ) ;
return ( x %= 2 ) ? 2 * fun ( k ) : ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) ; } function allQueries ( q , l , r ) { for ( let i = 0 ; i < q ; i ++ ) document . write ( ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) + " " ) ; }
let q = 3 ; let l = [ 2 , 2 , 5 ] ; let r = [ 4 , 8 , 9 ] ; allQueries ( q , l , r ) ;
function prefixXOR ( arr , preXOR , n ) {
for ( let i = 0 ; i < n ; i ++ ) { while ( arr [ i ] % 2 != 1 ) arr [ i ] = parseInt ( arr [ i ] / 2 ) ; preXOR [ i ] = arr [ i ] ; }
for ( let i = 1 ; i < n ; i ++ ) preXOR [ i ] = preXOR [ i - 1 ] ^ preXOR [ i ] ; }
function query ( preXOR , l , r ) { if ( l == 0 ) return preXOR [ r ] ; else return preXOR [ r ] ^ preXOR [ l - 1 ] ; }
let arr = [ 3 , 4 , 5 ] ; let n = arr . length ; let preXOR = new Array ( n ) ; prefixXOR ( arr , preXOR , n ) ; document . write ( query ( preXOR , 0 , 2 ) + " " ) ; document . write ( query ( preXOR , 1 , 2 ) + " " ) ;
function findMinSwaps ( arr , n ) {
let noOfZeroes = [ ] ; let i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
let ar = [ 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ] ; document . write ( findMinSwaps ( ar , ar . length ) ) ;
function minswaps ( arr , n ) { var count = 0 ; var num_unplaced_zeros = 0 ; for ( var index = n - 1 ; index >= 0 ; index -- ) { if ( arr [ index ] == 0 ) num_unplaced_zeros += 1 ; else count += num_unplaced_zeros ; } return count ; }
var arr = [ 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ] ; document . write ( minswaps ( arr , 9 ) ) ;
function arraySortedOrNot ( arr , n ) {
if ( n == 0 n == 1 ) return true ; for ( let i = 1 ; i < n ; i ++ )
if ( arr [ i - 1 ] > arr [ i ] ) return false ;
return true ; }
let arr = [ 20 , 23 , 23 , 45 , 78 , 88 ] ; let n = arr . length ; if ( arraySortedOrNot ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
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
function printMax ( arr , k , n ) {
var brr = arr . slice ( ) ;
brr . sort ( ( a , b ) => b - a ) ;
for ( var i = 0 ; i < n ; ++ i ) if ( brr . indexOf ( arr [ i ] ) < k ) document . write ( arr [ i ] + " " ) ; }
var arr = [ 50 , 8 , 45 , 12 , 25 , 40 , 84 ] ; var n = arr . length ; var k = 3 ; printMax ( arr , k , n ) ;
function printSmall ( arr , asize , n ) {
let copy_arr = [ ... arr ] ;
copy_arr . sort ( ( a , b ) => a - b ) ;
for ( let i = 0 ; i < asize ; ++ i ) { if ( arr [ i ] < copy_arr [ n ] ) document . write ( arr [ i ] + " " ) ; } }
let arr = [ 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 ] ; let asize = arr . length ; let n = 5 ; printSmall ( arr , asize , n ) ;
function checkIsAP ( arr , n ) { if ( n == 1 ) return true ;
arr . sort ( ( a , b ) => a - b ) ;
let d = arr [ 1 ] - arr [ 0 ] ; for ( let i = 2 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] != d ) return false ; return true ; }
let arr = [ 20 , 15 , 5 , 0 , 10 ] ; let n = arr . length ; ( checkIsAP ( arr , n ) ) ? ( document . write ( " " + " " ) ) : ( document . write ( " " + " " ) ) ;
function checkIsAP ( arr , n ) { var hm = new Map ( ) ; var smallest = 1000000000 , second_smallest = 1000000000 ; for ( var i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] < smallest ) { second_smallest = smallest ; smallest = arr [ i ] ; }
else if ( arr [ i ] != smallest && arr [ i ] < second_smallest ) second_smallest = arr [ i ] ;
if ( ! hm . has ( arr [ i ] ) ) { hm . set ( arr [ i ] , 1 ) ; }
else return false ; }
var diff = second_smallest - smallest ;
for ( var i = 0 ; i < n - 1 ; i ++ ) { if ( ! hm . has ( second_smallest ) ) return false ; second_smallest += diff ; } return true ; }
var arr = [ 20 , 15 , 5 , 0 , 10 ] ; var n = arr . length ; ( checkIsAP ( arr , n ) ) ? ( document . write ( " " ) ) : ( document . write ( " " ) ) ;
function countPairs ( a , n ) {
let mn = Number . MAX_VALUE ; let mx = Number . MIN_VALUE ; for ( let i = 0 ; i < n ; i ++ ) { mn = Math . min ( mn , a [ i ] ) ; mx = Math . max ( mx , a [ i ] ) ; }
let c1 = 0 ;
let c2 = 0 ; for ( let i = 0 ; i < n ; i ++ ) { if ( a [ i ] == mn ) c1 ++ ; if ( a [ i ] == mx ) c2 ++ ; }
if ( mn == mx ) return n * ( n - 1 ) / 2 ; else return c1 * c2 ; }
let a = [ 3 , 2 , 1 , 1 , 3 ] ; let n = a . length ; document . write ( countPairs ( a , n ) ) ;
class Node { constructor ( ) { this . data ; this . next = null ; } }
function rearrange ( head ) {
if ( head == null ) return null ;
let prev = head , curr = head . next ; while ( curr != null ) {
if ( prev . data > curr . data ) { let t = prev . data ; prev . data = curr . data ; curr . data = t ; }
if ( curr . next != null && curr . next . data > curr . data ) { let t = curr . next . data ; curr . next . data = curr . data ; curr . data = t ; } prev = curr . next ; if ( curr . next == null ) break ; curr = curr . next . next ; } return head ; }
function push ( head , k ) { let tem = new Node ( ) ; tem . data = k ; tem . next = head ; head = tem ; return head ; }
function display ( head ) { let curr = head ; while ( curr != null ) { document . write ( curr . data + " " ) ; curr = curr . next ; } }
let head = null ; head = push ( head , 7 ) ; head = push ( head , 3 ) ; head = push ( head , 8 ) ; head = push ( head , 6 ) ; head = push ( head , 9 ) ; head = rearrange ( head ) ; display ( head ) ;
class Node { constructor ( val ) { this . data = val ; this . next = null ; } }
var left = null ;
function printlist ( head ) { while ( head != null ) { document . write ( head . data + " " ) ; if ( head . next != null ) { document . write ( " " ) ; } head = head . next ; } document . write ( " " ) ; }
function rearrange ( head ) { if ( head != null ) { left = head ; reorderListUtil ( left ) ; } } function reorderListUtil ( right ) { if ( right == null ) { return ; } reorderListUtil ( right . next ) ;
if ( left == null ) { return ; }
if ( left != right && left . next != right ) { var temp = left . next ; left . next = right ; right . next = temp ; left = temp ;
} else { if ( left . next == right ) {
left . next . next = null ; left = null ; } else {
left . next = null ; left = null ; } } }
var head = new Node ( 1 ) ; head . next = new Node ( 2 ) ; head . next . next = new Node ( 3 ) ; head . next . next . next = new Node ( 4 ) ; head . next . next . next . next = new Node ( 5 ) ;
printlist ( head ) ;
rearrange ( head ) ;
printlist ( head ) ;
class Node { constructor ( d ) { this . data = d ; this . next = null ; } }
function getLength ( node ) { var size = 0 ; while ( node != null ) { node = node . next ; size ++ ; } return size ; }
function paddZeros ( sNode , diff ) { if ( sNode == null ) return null ; var zHead = new Node ( 0 ) ; diff -- ; var temp = zHead ; while ( ( diff -- ) != 0 ) { temp . next = new Node ( 0 ) ; temp = temp . next ; } temp . next = sNode ; return zHead ; }
function subtractLinkedListHelper ( l1 , l2 ) { if ( l1 == null && l2 == null && borrow == false ) return null ; var previous = subtractLinkedListHelper ( ( l1 != null ) ? l1 . next : null , ( l2 != null ) ? l2 . next : null ) ; var d1 = l1 . data ; var d2 = l2 . data ; var sub = 0 ;
if ( borrow ) { d1 -- ; borrow = false ; }
if ( d1 < d2 ) { borrow = true ; d1 = d1 + 10 ; }
sub = d1 - d2 ;
var current = new Node ( sub ) ;
current . next = previous ; return current ; }
function subtractLinkedList ( l1 , l2 ) {
if ( l1 == null && l2 == null ) return null ;
var len1 = getLength ( l1 ) ; var len2 = getLength ( l2 ) ; var lNode = null , sNode = null ; var temp1 = l1 ; var temp2 = l2 ;
if ( len1 != len2 ) { lNode = len1 > len2 ? l1 : l2 ; sNode = len1 > len2 ? l2 : l1 ; sNode = paddZeros ( sNode , Math . abs ( len1 - len2 ) ) ; } else {
while ( l1 != null && l2 != null ) { if ( l1 . data != l2 . data ) { lNode = l1 . data > l2 . data ? temp1 : temp2 ; sNode = l1 . data > l2 . data ? temp2 : temp1 ; break ; } l1 = l1 . next ; l2 = l2 . next ; } }
borrow = false ; return subtractLinkedListHelper ( lNode , sNode ) ; }
function printList ( head ) { var temp = head ; while ( temp != null ) { document . write ( temp . data + " " ) ; temp = temp . next ; } }
var head = new Node ( 1 ) ; head . next = new Node ( 0 ) ; head . next . next = new Node ( 0 ) ; var head2 = new Node ( 1 ) ; var result = subtractLinkedList ( head , head2 ) ; printList ( result ) ;
class Node {
constructor ( d ) { this . data = d ; this . next = null ; } }
function insertAtMid ( x ) {
if ( head == null ) head = new Node ( x ) ; else {
var newNode = new Node ( x ) ; var ptr = head ; var len = 0 ;
while ( ptr != null ) { len ++ ; ptr = ptr . next ; }
var count = ( ( len % 2 ) == 0 ) ? ( len / 2 ) : ( len + 1 ) / 2 ; ptr = head ;
while ( count -- > 1 ) ptr = ptr . next ;
newNode . next = ptr . next ; ptr . next = newNode ; } }
function display ( ) { var temp = head ; while ( temp != null ) { document . write ( temp . data + " " ) ; temp = temp . next ; } }
head = new Node ( 1 ) ; head . next = new Node ( 2 ) ; head . next . next = new Node ( 4 ) ; head . next . next . next = new Node ( 5 ) ; document . write ( " " + " " ) ; display ( ) ; var x = 3 ; insertAtMid ( x ) ; document . write ( " " + " " ) ; display ( ) ;
class Node { constructor ( val ) { this . data = val ; this . prev = null ; this . next = null ; } }
function getNode ( data ) {
var newNode = new Node ( ) ;
newNode . data = data ; newNode . prev = newNode . next = null ; return newNode ; }
function sortedInsert ( head_ref , newNode ) { var current ;
if ( head_ref == null ) head_ref = newNode ;
else if ( ( head_ref ) . data >= newNode . data ) { newNode . next = head_ref ; newNode . next . prev = newNode ; head_ref = newNode ; } else { current = head_ref ;
while ( current . next != null && current . next . data < newNode . data ) current = current . next ;
newNode . next = current . next ;
if ( current . next != null ) newNode . next . prev = newNode ; current . next = newNode ; newNode . prev = current ; } return head_ref ; }
function insertionSort ( head_ref ) {
var sorted = null ;
var current = head_ref ; while ( current != null ) {
var next = current . next ;
current . prev = current . next = null ;
sorted = sortedInsert ( sorted , current ) ;
current = next ; }
head_ref = sorted ; return head_ref ; }
function printList ( head ) { while ( head != null ) { document . write ( head . data + " " ) ; head = head . next ; } }
function push ( head_ref , new_data ) {
var new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = ( head_ref ) ; new_node . prev = null ;
if ( ( head_ref ) != null ) ( head_ref ) . prev = new_node ;
( head_ref ) = new_node ; return head_ref ; }
var head = null ;
head = push ( head , 9 ) ; head = push ( head , 3 ) ; head = push ( head , 5 ) ; head = push ( head , 10 ) ; head = push ( head , 12 ) ; head = push ( head , 8 ) ; document . write ( " " ) ; printList ( head ) ; head = insertionSort ( head ) ; document . write ( " " ) ; printList ( head ) ;
function reverse ( arr , s , e ) { while ( s < e ) { var tem = arr [ s ] ; arr [ s ] = arr [ e ] ; arr [ e ] = tem ; s = s + 1 ; e = e - 1 ; } }
function fun ( arr , k ) { var n = 4 - 1 ; var v = n - k ; if ( v >= 0 ) { reverse ( arr , 0 , v ) ; reverse ( arr , v + 1 , n ) ; reverse ( arr , 0 , n ) ; } }
arr [ 0 ] = 1 ; arr [ 1 ] = 2 ; arr [ 2 ] = 3 ; arr [ 3 ] = 4 ; for ( i = 0 ; i < 4 ; i ++ ) { fun ( arr , i ) ; document . write ( " " ) ; for ( j = 0 ; j < 4 ; j ++ ) { document . write ( arr [ j ] + " " ) ; } document . write ( " " ) ; }
const MAX = 100005 ;
var seg = Array ( 4 * MAX ) . fill ( 0 ) ;
function build ( node , l , r , a ) { if ( l == r ) seg [ node ] = a [ l ] ; else { var mid = parseInt ( ( l + r ) / 2 ) ; build ( 2 * node , l , mid , a ) ; build ( 2 * node + 1 , mid + 1 , r , a ) ; seg [ node ] = ( seg [ 2 * node ] seg [ 2 * node + 1 ] ) ; } }
function query ( node , l , r , start , end , a ) {
if ( l > end r < start ) return 0 ; if ( start <= l && r <= end ) return seg [ node ] ;
var mid = parseInt ( ( l + r ) / 2 ) ;
return ( ( query ( 2 * node , l , mid , start , end , a ) ) | ( query ( 2 * node + 1 , mid + 1 , r , start , end , a ) ) ) ; }
function orsum ( a , n , q , k ) {
build ( 1 , 0 , n - 1 , a ) ;
for ( j = 0 ; j < q ; j ++ ) {
var i = k [ j ] % ( n / 2 ) ;
var sec = query ( 1 , 0 , n - 1 , n / 2 - i , n - i - 1 , a ) ;
var first = ( query ( 1 , 0 , n - 1 , 0 , n / 2 - 1 - i , a ) | query ( 1 , 0 , n - 1 , n - i , n - 1 , a ) ) ; var temp = sec + first ;
document . write ( temp + " " ) ; } }
var a = [ 7 , 44 , 19 , 86 , 65 , 39 , 75 , 101 ] ; var n = a . length ; var q = 2 ; var k = [ 4 , 2 ] ; orsum ( a , n , q , k ) ;
function maximumEqual ( a , b , n ) {
let store = Array . from ( { length : 1e5 } , ( _ , i ) => 0 ) ;
for ( let i = 0 ; i < n ; i ++ ) { store [ b [ i ] ] = i + 1 ; }
let ans = Array . from ( { length : 1e5 } , ( _ , i ) => 0 ) ;
for ( let i = 0 ; i < n ; i ++ ) {
let d = Math . abs ( store [ a [ i ] ] - ( i + 1 ) ) ;
if ( store [ a [ i ] ] < i + 1 ) { d = n - d ; }
ans [ d ] ++ ; } let finalans = 0 ;
for ( let i = 0 ; i < 1e5 ; i ++ ) finalans = Math . max ( finalans , ans [ i ] ) ;
document . write ( finalans + " " ) ; }
let A = [ 6 , 7 , 3 , 9 , 5 ] ; let B = [ 7 , 3 , 9 , 5 , 6 ] ; let size = A . length ;
maximumEqual ( A , B , size ) ;
function RightRotate ( a , n , k ) {
k = k % n ; for ( let i = 0 ; i < n ; i ++ ) { if ( i < k ) {
document . write ( a [ n + i - k ] + " " ) ; } else {
document . write ( ( a [ i - k ] ) + " " ) ; } } document . write ( " " ) ; }
let Array = [ 1 , 2 , 3 , 4 , 5 ] ; let N = Array . length ; let K = 2 ; RightRotate ( Array , N , K ) ;
function restoreSortedArray ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > arr [ i + 1 ] ) {
reverse ( arr , 0 , i ) ; reverse ( arr , i + 1 , n ) ; reverse ( arr , 0 , n ) ; } } } function reverse ( arr , i , j ) { let temp ; while ( i < j ) { temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; i ++ ; j -- ; } }
function printArray ( arr , size ) { for ( let i = 0 ; i < size ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 3 , 4 , 5 , 1 , 2 ] ; let n = arr . length ; restoreSortedArray ( arr , n - 1 ) ; printArray ( arr , n )
function findStartIndexOfArray ( arr , low , high ) { if ( low > high ) { return - 1 ; } if ( low == high ) { return low ; } let mid = low + parseInt ( ( high - low ) / 2 , 10 ) ; if ( arr [ mid ] > arr [ mid + 1 ] ) { return mid + 1 ; } if ( arr [ mid - 1 ] > arr [ mid ] ) { return mid ; } if ( arr [ low ] > arr [ mid ] ) { return findStartIndexOfArray ( arr , low , mid - 1 ) ; } else { return findStartIndexOfArray ( arr , mid + 1 , high ) ; } }
function restoreSortedArray ( arr , n ) {
if ( arr [ 0 ] < arr [ n - 1 ] ) { return ; } let start = findStartIndexOfArray ( arr , 0 , n - 1 ) ;
arr . sort ( ) ; }
function printArray ( arr , size ) { for ( let i = 0 ; i < size ; i ++ ) { document . write ( arr [ i ] + " " ) ; } }
let arr = [ 1 , 2 , 3 , 4 , 5 ] ; let n = arr . length ; restoreSortedArray ( arr , n ) ; printArray ( arr , n ) ;
function leftrotate ( str , d ) { var ans = str . substring ( d , str . length ) + str . substring ( 0 , d ) ; return ans ; }
function rightrotate ( str , d ) { return leftrotate ( str , str . length - d ) ; }
var str1 = " " ; document . write ( leftrotate ( str1 , 2 ) + " " ) ; var str2 = " " ; document . write ( rightrotate ( str2 , 2 ) + " " ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = this . prev = null ; } }
function insertNode ( start , value ) {
if ( start == null ) { let new_node = new Node ( ) ; new_node . data = value ; new_node . next = new_node . prev = new_node ; start = new_node ; return new_node ; }
let last = ( start ) . prev ;
let new_node = new Node ( ) ; new_node . data = value ;
new_node . next = start ;
( start ) . prev = new_node ;
new_node . prev = last ;
last . next = new_node ; return start ; }
function displayList ( start ) { let temp = start ; while ( temp . next != start ) { document . write ( temp . data + " " ) ; temp = temp . next ; } document . write ( temp . data + " " ) ; }
function searchList ( start , search ) {
let temp = start ;
let count = 0 , flag = 0 , value ;
if ( temp == null ) return - 1 ; else {
while ( temp . next != start ) {
count ++ ;
if ( temp . data == search ) { flag = 1 ; count -- ; break ; }
temp = temp . next ; }
if ( temp . data == search ) { count ++ ; flag = 1 ; }
if ( flag == 1 ) document . write ( " " + search + " " + count ) ; else document . write ( " " + search + " " ) ; } return - 1 ; }
let start = null ;
start = insertNode ( start , 4 ) ;
start = insertNode ( start , 5 ) ;
start = insertNode ( start , 7 ) ;
start = insertNode ( start , 8 ) ;
start = insertNode ( start , 6 ) ; document . write ( " " ) ; displayList ( start ) ; searchList ( start , 5 ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; this . prev = null ; } }
function getNode ( data ) { var newNode = new Node ( ) ; newNode . data = data ; return newNode ; }
function insertEnd ( head , new_node ) {
if ( head == null ) { new_node . next = new_node . prev = new_node ; head = new_node ; return head ; }
var last = head . prev ;
new_node . next = head ;
head . prev = new_node ;
new_node . prev = last ;
last . next = new_node ; return head ; }
function reverse ( head ) { if ( head == null ) return null ;
var new_head = null ;
var last = head . prev ;
var curr = last , prev ;
while ( curr . prev != last ) { prev = curr . prev ;
new_head = insertEnd ( new_head , curr ) ; curr = prev ; } new_head = insertEnd ( new_head , curr ) ;
return new_head ; }
function display ( head ) { if ( head == null ) return ; var temp = head ; document . write ( " " ) ; while ( temp . next != head ) { document . write ( temp . data + " " ) ; temp = temp . next ; } document . write ( temp . data + " " ) ; var last = head . prev ; temp = last ; document . write ( " " ) ; while ( temp . prev != last ) { document . write ( temp . data + " " ) ; temp = temp . prev ; } document . write ( temp . data + " " ) ; }
var head = null ; head = insertEnd ( head , getNode ( 1 ) ) ; head = insertEnd ( head , getNode ( 2 ) ) ; head = insertEnd ( head , getNode ( 3 ) ) ; head = insertEnd ( head , getNode ( 4 ) ) ; head = insertEnd ( head , getNode ( 5 ) ) ; document . write ( " " ) ; display ( head ) ; head = reverse ( head ) ; document . write ( " " ) ; display ( head ) ;
var MAXN = 1001 ;
var depth = Array ( MAXN ) ;
var parent = Array ( MAXN ) ; var adj = Array . from ( Array ( MAXN ) , ( ) => Array ( ) ) ; function addEdge ( u , v ) { adj [ u ] . push ( v ) ; adj [ v ] . push ( u ) ; } function dfs ( cur , prev ) {
parent [ cur ] = prev ;
depth [ cur ] = depth [ prev ] + 1 ;
for ( var i = 0 ; i < adj [ cur ] . length ; i ++ ) if ( adj [ cur ] [ i ] != prev ) dfs ( adj [ cur ] [ i ] , cur ) ; } function preprocess ( ) {
depth [ 0 ] = - 1 ;
dfs ( 1 , 0 ) ; }
function LCANaive ( u , v ) { if ( u == v ) return u ; if ( depth [ u ] > depth [ v ] ) { var temp = u ; u = v ; v = temp ; } v = parent [ v ] ; return LCANaive ( u , v ) ; }
for ( var i = 0 ; i < MAXN ; i ++ ) adj [ i ] = [ ] ;
addEdge ( 1 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 1 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 2 , 6 ) ; addEdge ( 3 , 7 ) ; addEdge ( 4 , 8 ) ; addEdge ( 4 , 9 ) ; addEdge ( 9 , 10 ) ; addEdge ( 9 , 11 ) ; addEdge ( 7 , 12 ) ; addEdge ( 7 , 13 ) ; preprocess ( ) ; document . write ( " " + LCANaive ( 11 , 8 ) + " " ) ; document . write ( " " + LCANaive ( 3 , 13 ) ) ;
let N = 3 ;
document . write ( Math . pow ( 2 , N + 1 ) - 2 ) ;
function countOfNum ( n , a , b ) { let cnt_of_a , cnt_of_b , cnt_of_ab , sum ;
cnt_of_a = Math . floor ( n / a ) ;
cnt_of_b = Math . floor ( n / b ) ;
sum = cnt_of_b + cnt_of_a ;
cnt_of_ab = Math . floor ( n / ( a * b ) ) ;
sum = sum - cnt_of_ab ; return sum ; }
function sumOfNum ( n , a , b ) { let i ; let sum = 0 ;
let ans = new Set ( ) ;
for ( i = a ; i <= n ; i = i + a ) { ans . add ( i ) ; }
for ( i = b ; i <= n ; i = i + b ) { ans . add ( i ) ; }
for ( let it of ans . values ( ) ) { sum = sum + it ; } return sum ; }
let N = 88 ; let A = 11 ; let B = 8 ; let count = countOfNum ( N , A , B ) ; let sumofnum = sumOfNum ( N , A , B ) ; document . write ( sumofnum % count ) ;
function get ( L , R ) {
let x = 1.0 / L ;
let y = 1.0 / ( R + 1.0 ) ; return ( x - y ) ; }
let L = 6 , R = 12 ;
let ans = get ( L , R ) ; document . write ( Math . round ( ans * 100 ) / 100 ) ;
const MAX = 100000 ;
let v = [ ] ; function upper_bound ( ar , k ) { let s = 0 ; let e = ar . length ; while ( s != e ) { let mid = s + e >> 1 ; if ( ar [ mid ] <= k ) { s = mid + 1 ; } else { e = mid ; } } if ( s == ar . length ) { return - 1 ; } return s ; }
function consecutiveOnes ( x ) {
let p = 0 ; while ( x > 0 ) {
if ( x % 2 == 1 && p == 1 ) return true ;
p = x % 2 ;
x = parseInt ( x / 2 ) ; } return false ; }
function preCompute ( ) {
for ( let i = 0 ; i <= MAX ; i ++ ) { if ( ! consecutiveOnes ( i ) ) v . push ( i ) ; } }
function nextValid ( n ) {
let it = upper_bound ( v , n ) ; let val = v [ it ] ; return val ; }
function performQueries ( queries , q ) { for ( let i = 0 ; i < q ; i ++ ) document . write ( nextValid ( queries [ i ] ) + " " ) ; }
let queries = [ 4 , 6 ] ; let q = queries . length ;
preCompute ( ) ;
performQueries ( queries , q ) ;
function changeToOnes ( str ) {
var i , l , ctr = 0 ; l = str . length ;
for ( i = l - 1 ; i >= 0 ; i -- ) {
if ( str [ i ] == ' ' ) ctr ++ ;
else break ; }
return l - ctr ; }
function removeZeroesFromFront ( str ) { var s ; var i = 0 ;
while ( i < str . length && str [ i ] == ' ' ) i ++ ;
if ( i == str . length ) s = " " ;
else s = str . substring ( i , str . length - i ) ; return s ; }
var str = " " ;
str = removeZeroesFromFront ( str ) ; document . write ( changeToOnes ( str ) ) ;
function MinDeletion ( a , n ) {
let map = new Map ( ) ;
for ( let i = 0 ; i < n ; i ++ ) { if ( map [ a [ i ] ] ) map [ a [ i ] ] ++ ; else map [ a [ i ] ] = 1 }
let ans = 0 ; for ( var m in map ) {
let x = m ;
let frequency = map [ m ] ;
if ( x <= frequency ) {
ans += ( frequency - x ) ; }
else ans += frequency ; } ; return ans ; }
let a = [ 2 , 3 , 2 , 3 , 4 , 4 , 4 , 4 , 5 ] ; let n = a . length ; document . write ( MinDeletion ( a , n ) ) ;
function maxCountAB ( s , n ) {
var A = 0 , B = 0 , BA = 0 , ans = 0 ; for ( var i = 0 ; i < n ; i ++ ) { var S = s [ i ] ; var L = S . length ; for ( var j = 0 ; j < L - 1 ; j ++ ) {
if ( S [ j ] == ' ' && S [ j + 1 ] == ' ' ) { ans ++ ; } }
if ( S [ 0 ] == ' ' && S [ L - 1 ] == ' ' ) BA ++ ;
else if ( S [ 0 ] == ' ' ) B ++ ;
else if ( S [ L - 1 ] == ' ' ) A ++ ; }
if ( BA == 0 ) ans += Math . min ( B , A ) ; else if ( A + B == 0 ) ans += BA - 1 ; else ans += BA + Math . min ( B , A ) ; return ans ; }
var s = [ " " , " " , " " ] ; var n = s . length ; document . write ( maxCountAB ( s , n ) ) ;
function MinOperations ( n , x , arr ) {
let total = 0 ; for ( let i = 0 ; i < n ; ++ i ) {
if ( arr [ i ] > x ) { let difference = arr [ i ] - x ; total = total + difference ; arr [ i ] = x ; } }
for ( let i = 1 ; i < n ; ++ i ) { let LeftNeigbouringSum = arr [ i ] + arr [ i - 1 ] ;
if ( LeftNeigbouringSum > x ) { let current_diff = LeftNeigbouringSum - x ; arr [ i ] = Math . max ( 0 , arr [ i ] - current_diff ) ; total = total + current_diff ; } } return total ; }
let X = 1 ; let arr = [ 1 , 6 , 1 , 2 , 0 , 4 ] ; let N = arr . length ; document . write ( MinOperations ( N , X , arr ) + " " ) ;
function findNumbers ( arr , n ) {
sumN = ( n * ( n + 1 ) ) / 2 ;
sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
let sum = 0 ; let sumSq = 0 ; for ( let i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq += Math . pow ( arr [ i ] , 2 ) ; } B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; A = sum - sumN + B ; document . write ( " " + A , " " , B ) ; }
let arr = [ 1 , 2 , 2 , 3 , 4 ] ; n = arr . length ; findNumbers ( arr , n ) ;
function is_prefix ( temp , str ) {
if ( temp . length < str . length ) return 0 ; else {
for ( let i = 0 ; i < str . length ; i ++ ) { if ( str [ i ] != temp [ i ] ) return 0 ; } return 1 ; } }
function lexicographicallyString ( input , n , str ) {
input = Array . from ( input ) . sort ( ) ; for ( let i = 0 ; i < n ; i ++ ) { let temp = input [ i ] ;
if ( is_prefix ( temp , str ) ) { return temp ; } }
return " " ; }
let arr = [ " " , " " , " " , " " , " " ] ; let S = " " ; let N = 5 ; document . write ( lexicographicallyString ( arr , N , S ) ) ;
function Rearrange ( arr , K , N ) {
let ans = new Array ( N + 1 ) ;
let f = - 1 ; for ( let i = 0 ; i < N ; i ++ ) { ans [ i ] = - 1 ; }
for ( let i = 0 ; i < arr . length ; i ++ ) { if ( arr [ i ] == K ) { K = i ; break ; } }
let smaller = [ ] ; let greater = [ ] ;
for ( let i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] < arr [ K ] ) smaller . push ( arr [ i ] ) ;
else if ( arr [ i ] > arr [ K ] ) greater . push ( arr [ i ] ) ; } let low = 0 , high = N - 1 ;
while ( low <= high ) {
let mid = Math . floor ( ( low + high ) / 2 ) ;
if ( mid == K ) { ans [ mid ] = arr [ K ] ; f = 1 ; break ; }
else if ( mid < K ) { if ( smaller . length == 0 ) { break ; } ans [ mid ] = smaller [ smaller . length - 1 ] ; smaller . pop ( ) ; low = mid + 1 ; }
else { if ( greater . length == 0 ) { break ; } ans [ mid ] = greater [ greater . length - 1 ] ; greater . pop ( ) ; high = mid - 1 ; } }
if ( f == - 1 ) { document . write ( - 1 ) ; return ; }
for ( let i = 0 ; i < N ; i ++ ) {
if ( ans [ i ] == - 1 ) { if ( smaller . length != 0 ) { ans [ i ] = smaller [ smaller . length - 1 ] ; smaller . pop ( ) ; } else if ( greater . length != 0 ) { ans [ i ] = greater [ greater . length - 1 ] ; greater . pop ; } } }
for ( let i = 0 ; i < N ; i ++ ) document . write ( ans [ i ] + " " ) ; document . write ( " " ) }
let arr = [ 10 , 7 , 2 , 5 , 3 , 8 ] ; let K = 7 ; let N = arr . length ;
Rearrange ( arr , K , N ) ;
function minimumK ( arr , M , N ) {
let good = Math . floor ( ( N * 1.0 ) / ( ( M + 1 ) * 1.0 ) ) + 1 ;
for ( let i = 1 ; i <= N ; i ++ ) { let K = i ;
let candies = N ;
let taken = 0 ; while ( candies > 0 ) {
taken += Math . min ( K , candies ) ; candies -= Math . min ( K , candies ) ;
for ( let j = 0 ; j < M ; j ++ ) {
let consume = ( arr [ j ] * candies ) / 100 ;
candies -= consume ; } }
if ( taken >= good ) { document . write ( i ) ; return ; } } }
let N = 13 , M = 1 ; let arr = new Array ( ) ; arr . push ( 50 ) ; minimumK ( arr , M , N ) ;
function calcTotalTime ( path ) {
var time = 0 ;
var x = 0 , y = 0 ;
var s = new Set ( ) ; for ( var i = 0 ; i < path . length ; i ++ ) { var p = x ; var q = y ; if ( path [ i ] == ' ' ) y ++ ; else if ( path [ i ] == ' ' ) y -- ; else if ( path [ i ] == ' ' ) x ++ ; else if ( path [ i ] == ' ' ) x -- ;
if ( ! s . has ( [ p + x , q + y ] . toString ( ) ) ) {
time += 2 ;
s . add ( [ p + x , q + y ] . toString ( ) ) ; } else time += 1 ; }
document . write ( time ) }
var path = " " ; calcTotalTime ( path ) ;
function findCost ( A , N ) {
var totalCost = 0 ; var i ;
for ( i = 0 ; i < N ; i ++ ) {
if ( A [ i ] == 0 ) {
A [ i ] = 1 ;
totalCost += i ; } }
return totalCost ; }
var arr = [ 1 , 0 , 1 , 0 , 1 , 0 ] var N = arr . length document . write ( findCost ( arr , N ) ) ;
function peakIndex ( arr ) { var N = arr . length ;
if ( arr . length < 3 ) return - 1 ; var i = 0 ;
while ( i + 1 < N ) {
if ( arr [ i + 1 ] < arr [ i ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; } if ( i == 0 i == N - 1 ) return - 1 ;
var ans = i ;
while ( i < N - 1 ) {
if ( arr [ i ] < arr [ i + 1 ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; }
if ( i == N - 1 ) return ans ;
return - 1 ; }
var arr = [ 0 , 1 , 0 ] ; document . write ( peakIndex ( arr ) ) ;
function hasArrayTwoPairs ( nums , n , target ) {
nums . sort ( ) ; var i ;
for ( i = 0 ; i < n ; i ++ ) {
var x = target - nums [ i ] ;
var low = 0 , high = n - 1 ; while ( low <= high ) {
var mid = low + ( Math . floor ( ( high - low ) / 2 ) ) ;
if ( nums [ mid ] > x ) { high = mid - 1 ; }
else if ( nums [ mid ] < x ) { low = mid + 1 ; }
else {
if ( mid == i ) { if ( ( mid - 1 >= 0 ) && nums [ mid - 1 ] == x ) { document . write ( nums [ i ] + " " ) ; document . write ( nums [ mid - 1 ] ) ; return ; } if ( ( mid + 1 < n ) && nums [ mid + 1 ] == x ) { document . write ( nums [ i ] + " " ) ; document . write ( nums [ mid + 1 ] ) ; return ; } break ; }
else { document . write ( nums [ i ] + " " ) ; document . write ( nums [ mid ] ) ; return ; } } } }
document . write ( - 1 ) ; }
var A = [ 0 , - 1 , 2 , - 3 , 1 ] ; var X = - 2 ; var N = A . length ;
hasArrayTwoPairs ( A , N , X ) ;
function findClosest ( N , target ) { let closest = - 1 ; let diff = Number . MAX_VALUE ;
for ( let i = 1 ; i <= Math . sqrt ( N ) ; i ++ ) { if ( N % i == 0 ) {
if ( N / i == i ) {
if ( Math . abs ( target - i ) < diff ) { diff = Math . abs ( target - i ) ; closest = i ; } } else {
if ( Math . abs ( target - i ) < diff ) { diff = Math . abs ( target - i ) ; closest = i ; }
if ( Math . abs ( target - N / i ) < diff ) { diff = Math . abs ( target - N / i ) ; closest = N / i ; } } } }
document . write ( closest ) ; }
let N = 16 , X = 5 ;
findClosest ( N , X ) ;
function power ( A , N ) {
let count = 0 ; if ( A == 1 ) return 0 ; while ( N > 0 ) {
count ++ ;
N /= A ; } return count ; }
function Pairs ( N , A , B ) { let powerA , powerB ;
powerA = power ( A , N ) ;
powerB = power ( B , N ) ;
let letialB = B , letialA = A ;
A = 1 ; for ( let i = 0 ; i <= powerA ; i ++ ) { B = 1 ; for ( let j = 0 ; j <= powerB ; j ++ ) {
if ( B == N - A ) { document . write ( i + " " + j ) ; return ; }
B *= letialB ; }
A *= letialA ; }
document . write ( " " ) ; return ; }
let N = 106 , A = 3 , B = 5 ;
Pairs ( N , A , B ) ;
function findNonMultiples ( arr , n , k ) {
let multiples = new Set ( ) ;
for ( let i = 0 ; i < n ; ++ i ) {
if ( ! multiples . has ( arr [ i ] ) ) {
for ( let j = 1 ; j <= k / arr [ i ] ; j ++ ) { multiples . add ( arr [ i ] * j ) ; } } }
return k - multiples . size ; }
function countValues ( arr , N , L , R ) {
return findNonMultiples ( arr , N , R ) - findNonMultiples ( arr , N , L - 1 ) ; }
let arr = [ 2 , 3 , 4 , 5 , 6 ] ; let N = arr . length ; let L = 1 , R = 20 ;
document . write ( countValues ( arr , N , L , R ) ) ;
function minCollectingSpeed ( piles , H ) {
var ans = - 1 ; var low = 1 , high ;
high = piles . reduce ( ( a , b ) => Math . max ( a , b ) ) ;
while ( low <= high ) {
var K = low + parseInt ( ( high - low ) / 2 ) ; var time = 0 ;
piles . forEach ( ai => { time += parseInt ( ( ai + K - 1 ) / K ) ; } ) ;
if ( time <= H ) { ans = K ; high = K - 1 ; }
else { low = K + 1 ; } }
document . write ( ans ) ; }
var arr = [ 3 , 6 , 7 , 11 ] ; var H = 8 ;
minCollectingSpeed ( arr , H ) ;
function cntDisPairs ( arr , N , K ) {
var cntPairs = 0 ;
arr . sort ( ) ;
var i = 0 ;
var j = N - 1 ;
while ( i < j ) {
if ( arr [ i ] + arr [ j ] == K ) {
while ( i < j && arr [ i ] == arr [ i + 1 ] ) {
i ++ ; }
while ( i < j && arr [ j ] == arr [ j - 1 ] ) {
j -- ; }
cntPairs += 1 ;
i ++ ;
j -- ; }
else if ( arr [ i ] + arr [ j ] < K ) {
i ++ ; } else {
j -- ; } } return cntPairs ; }
var arr = [ 5 , 6 , 5 , 7 , 7 , 8 ] ; var N = arr . length ; var K = 13 ; document . write ( cntDisPairs ( arr , N , K ) ) ;
function longestSubsequence ( N , Q , arr , Queries ) { for ( let i = 0 ; i < Q ; i ++ ) {
let x = Queries [ i ] [ 0 ] ; let y = Queries [ i ] [ 1 ] ;
arr [ x - 1 ] = y ;
let count = 1 ; for ( let j = 1 ; j < N ; j ++ ) {
if ( arr [ j ] != arr [ j - 1 ] ) { count += 1 ; } }
document . write ( count + " " ) ; } }
let arr = [ 1 , 1 , 2 , 5 , 2 ] ; let N = arr . length ; let Q = 2 ; let Queries = [ [ 1 , 3 ] , [ 4 , 2 ] ] ;
longestSubsequence ( N , Q , arr , Queries ) ;
function longestSubsequence ( N , Q , arr , Queries ) { var count = 1 ;
for ( var i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] != arr [ i - 1 ] ) { count += 1 ; } }
for ( var i = 0 ; i < Q ; i ++ ) {
var x = Queries [ i ] [ 0 ] ; var y = Queries [ i ] [ 1 ] ;
if ( x > 1 ) {
if ( arr [ x - 1 ] != arr [ x - 2 ] ) { count -= 1 ; }
if ( arr [ x - 2 ] != y ) { count += 1 ; } }
if ( x < N ) {
if ( arr [ x ] != arr [ x - 1 ] ) { count -= 1 ; }
if ( y != arr [ x ] ) { count += 1 ; } } document . write ( count + " " ) ;
arr [ x - 1 ] = y ; } }
var arr = [ 1 , 1 , 2 , 5 , 2 ] ; var N = arr . length ; var Q = 2 ; var Queries = [ [ 1 , 3 ] , [ 4 , 2 ] ] ;
longestSubsequence ( N , Q , arr , Queries ) ;
function sum ( arr , n ) {
var mp = new Map ( ) ;
for ( var i = 0 ; i < n ; i ++ ) { if ( mp . has ( arr [ i ] ) ) { var tmp = mp . get ( arr [ i ] ) ; tmp . push ( i ) ; mp . set ( arr [ i ] , tmp ) ; } else { mp . set ( arr [ i ] , [ i ] ) ; } }
var ans = Array ( n ) ;
for ( var i = 0 ; i < n ; i ++ ) {
var sum = 0 ;
mp . get ( arr [ i ] ) . forEach ( it => {
sum += Math . abs ( it - i ) ; } ) ;
ans [ i ] = sum ; }
for ( var i = 0 ; i < n ; i ++ ) { document . write ( ans [ i ] + " " ) ; } return ; }
var arr = [ 1 , 3 , 1 , 1 , 2 ] ;
var n = arr . length ;
sum ( arr , n ) ;
function conVowUpp ( str ) {
var N = str . length ; for ( var i = 0 ; i < N ; i ++ ) { if ( str [ i ] === " " str [ i ] === " " str [ i ] === " " str [ i ] === " " str [ i ] === " " ) { document . write ( str [ i ] . toUpperCase ( ) ) ; } else { document . write ( str [ i ] ) ; } } }
var str = " " ; conVowUpp ( str ) ;
var mp = new Map ( ) ; var N , P ;
function helper ( mid ) { var cnt = 0 ; mp . forEach ( ( value , ) => { var temp = value ; while ( temp >= mid ) { temp -= mid ; cnt ++ ; } } ) ;
return cnt >= N ; }
function findMaximumDays ( arr ) {
for ( var i = 0 ; i < P ; i ++ ) { if ( mp . has ( arr [ i ] ) ) mp . set ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) else mp . set ( arr [ i ] , 1 ) ; }
var start = 0 , end = P , ans = 0 ; while ( start <= end ) {
var mid = start + parseInt ( ( end - start ) / 2 ) ;
if ( mid != 0 && helper ( mid ) ) { ans = mid ;
start = mid + 1 ; } else if ( mid == 0 ) { start = mid + 1 ; } else { end = mid - 1 ; } } return ans ; }
N = 3 , P = 10 ; var arr = [ 1 , 2 , 2 , 1 , 1 , 3 , 3 , 3 , 2 , 4 ] ;
document . write ( findMaximumDays ( arr ) ) ;
function countSubarrays ( a , n , k ) {
var ans = 0 ;
var pref = [ ] ; pref . push ( 0 ) ;
for ( var i = 0 ; i < n ; i ++ ) pref . push ( ( a [ i ] + pref [ i ] ) % k ) ;
for ( var i = 1 ; i <= n ; i ++ ) { for ( var j = i ; j <= n ; j ++ ) {
if ( ( pref [ j ] - pref [ i - 1 ] + k ) % k == j - i + 1 ) { ans ++ ; } } }
document . write ( ans + ' ' ) ; }
var arr = [ 2 , 3 , 5 , 3 , 1 , 5 ] ;
var N = arr . length ;
var K = 4 ;
countSubarrays ( arr , N , K ) ;
function check ( s , k ) { let n = s . length ;
for ( let i = 0 ; i < k ; i ++ ) { for ( let j = i ; j < n ; j += k ) {
if ( s [ i ] != s [ j ] ) return false ; } } let c = 0 ;
for ( let i = 0 ; i < k ; i ++ ) {
if ( s [ i ] == ' ' )
c ++ ;
else
c -- ; }
if ( c == 0 ) return true ; else return false ; }
let s = " " ; let k = 2 ; if ( check ( s , k ) ) document . write ( " " + " " ) ; else document . write ( " " ) ;
function isSame ( str , n ) {
var mp = { } ; for ( var i = 0 ; i < str . length ; i ++ ) { if ( mp . hasOwnProperty ( str [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ) ) { mp [ str [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] = mp [ str [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] + 1 ; } else { mp [ str [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] = 1 ; } } for ( const [ key , value ] of Object . entries ( mp ) ) {
if ( value >= n ) { return true ; } }
return false ; }
var str = " " ; var n = 4 ;
if ( isSame ( str , n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
let eps = 1e-6 ;
function func ( a , b , c , x ) { return a * x * x + b * x + c ; }
function findRoot ( a , b , c , low , high ) { let x = - 1 ;
while ( Math . abs ( high - low ) > eps ) {
x = ( low + high ) / 2 ;
if ( func ( a , b , c , low ) * func ( a , b , c , x ) <= 0 ) { high = x ; }
else { low = x ; } }
return x ; }
function solve ( a , b , c , A , B ) {
if ( func ( a , b , c , A ) * func ( a , b , c , B ) > 0 ) { document . write ( " " ) ; }
else { document . write ( findRoot ( a , b , c , A , B ) ) ; } }
let a = 2 , b = - 3 , c = - 2 , A = 0 , B = 3 ;
solve ( a , b , c , A , B ) ;
function possible ( mid , a ) {
let n = a . length ;
let total = parseInt ( ( n * ( n - 1 ) ) / 2 ) ;
let need = parseInt ( ( total + 1 ) / 2 ) ; let count = 0 ; let start = 0 , end = 1 ;
while ( end < n ) { if ( a [ end ] - a [ start ] <= mid ) { end ++ ; } else { count += ( end - start - 1 ) ; start ++ ; } }
if ( end == n && start < end && a [ end - 1 ] - a [ start ] <= mid ) { let t = end - start - 1 ; count += parseInt ( t * ( t + 1 ) / 2 ) ; }
if ( count >= need ) return true ; else return false ; }
function findMedian ( a ) {
let n = a . length ;
let low = 0 , high = a [ n - 1 ] - a [ 0 ] ;
while ( low <= high ) {
let mid = ( low + high ) / 2 ;
if ( possible ( mid , a ) ) high = mid - 1 ; else low = mid + 1 ; }
return high + 1 ; }
let a = [ 1 , 7 , 5 , 2 ] ; a . sort ( ) ; document . write ( findMedian ( a ) ) ;
function UniversalSubset ( A , B ) {
var n1 = A . length ; var n2 = B . length ;
var res = [ ] ;
var A_fre = Array . from ( Array ( n1 ) , ( ) => Array ( 26 ) ) ; for ( var i = 0 ; i < n1 ; i ++ ) { for ( var j = 0 ; j < 26 ; j ++ ) A_fre [ i ] [ j ] = 0 ; }
for ( var i = 0 ; i < n1 ; i ++ ) { for ( var j = 0 ; j < A [ i ] . length ; j ++ ) { A_fre [ i ] [ A [ i ] . charCodeAt ( j ) - ' ' . charCodeAt ( 0 ) ] ++ ; } }
var B_fre = Array ( 26 ) . fill ( 0 ) ; for ( var i = 0 ; i < n2 ; i ++ ) { var arr = Array ( 26 ) . fill ( 0 ) ; for ( var j = 0 ; j < B [ i ] . length ; j ++ ) { arr [ B [ i ] . charCodeAt ( j ) - ' ' . charCodeAt ( 0 ) ] ++ ; B_fre [ B [ i ] . charCodeAt ( j ) - ' ' . charCodeAt ( 0 ) ] = Math . max ( B_fre [ B [ i ] . charCodeAt ( j ) - ' ' . charCodeAt ( 0 ) ] , arr [ B [ i ] . charCodeAt ( j ) - ' ' . charCodeAt ( 0 ) ] ) ; } } for ( var i = 0 ; i < n1 ; i ++ ) { var flag = 0 ; for ( var j = 0 ; j < 26 ; j ++ ) {
if ( A_fre [ i ] [ j ] < B_fre [ j ] ) {
flag = 1 ; break ; } }
if ( flag == 0 )
res . push ( A [ i ] ) ; }
if ( res . length > 0 ) {
for ( var i = 0 ; i < res . length ; i ++ ) { for ( var j = 0 ; j < res [ i ] . length ; j ++ ) document . write ( res [ i ] [ j ] ) ; } document . write ( " " ) ; }
else document . write ( " " ) ; }
var A = [ " " , " " , " " ] ; var B = [ " " , " " ] ; UniversalSubset ( A , B ) ;
function findPair ( a , n ) {
let min_dist = Number . MAX_VALUE ; let index_a = - 1 , index_b = - 1 ;
for ( let i = 0 ; i < n ; i ++ ) {
for ( let j = i + 1 ; j < n ; j ++ ) {
if ( j - i < min_dist ) {
if ( a [ i ] % a [ j ] == 0 a [ j ] % a [ i ] == 0 ) {
min_dist = j - i ;
index_a = i ; index_b = j ; } } } }
if ( index_a == - 1 ) { document . write ( " " ) ; }
else { document . write ( " " + a [ index_a ] + " " + a [ index_b ] + " " ) ; } }
let a = [ 2 , 3 , 4 , 5 , 6 ] ; let n = a . length ;
findPair ( a , n ) ;
function printNum ( L , R ) {
for ( let i = L ; i <= R ; i ++ ) { let temp = i ; let c = 10 ; let flag = 0 ;
while ( temp > 0 ) {
if ( temp % 10 >= c ) { flag = 1 ; break ; } c = temp % 10 ; temp /= 10 ; }
if ( flag == 0 ) document . write ( i + " " ) ; } }
let L = 10 , R = 15 ;
printNum ( L , R ) ;
function findMissing ( arr , left , right , diff ) {
if ( right <= left ) return 0 ;
let mid = left + parseInt ( ( right - left ) / 2 , 10 ) ;
if ( arr [ mid + 1 ] - arr [ mid ] != diff ) return ( arr [ mid ] + diff ) ;
if ( mid > 0 && arr [ mid ] - arr [ mid - 1 ] != diff ) return ( arr [ mid - 1 ] + diff ) ;
if ( arr [ mid ] == arr [ 0 ] + mid * diff ) return findMissing ( arr , mid + 1 , right , diff ) ;
return findMissing ( arr , left , mid - 1 , diff ) ; }
function missingElement ( arr , n ) {
arr . sort ( function ( a , b ) { return a - b } ) ;
let diff = parseInt ( ( arr [ n - 1 ] - arr [ 0 ] ) / n , 10 ) ;
return findMissing ( arr , 0 , n - 1 , diff ) ; }
let arr = [ 2 , 8 , 6 , 10 ] ; let n = arr . length ;
document . write ( missingElement ( arr , n ) ) ;
function power ( x , y ) { let temp ; if ( y == 0 ) return 1 ; temp = power ( x , Math . floor ( y / 2 ) ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
function nthRootSearch ( low , high , N , K ) {
if ( low <= high ) {
let mid = Math . floor ( ( low + high ) / 2 ) ;
if ( ( power ( mid , K ) <= N ) && ( power ( mid + 1 , K ) > N ) ) { return mid ; }
else if ( power ( mid , K ) < N ) { return nthRootSearch ( mid + 1 , high , N , K ) ; } else { return nthRootSearch ( low , mid - 1 , N , K ) ; } } return low ; }
let N = 16 , K = 4 ;
document . write ( nthRootSearch ( 0 , N , N , K ) ) ;
function get_subset_count ( arr , K , N ) {
( arr ) . sort ( function ( a , b ) { return a - b ; } ) ; let left , right ; left = 0 ; right = N - 1 ;
let ans = 0 ; while ( left <= right ) { if ( arr [ left ] + arr [ right ] < K ) {
ans += 1 << ( right - left ) ; left ++ ; } else {
right -- ; } } return ans ; }
let arr = [ 2 , 4 , 5 , 7 ] ; let K = 8 ; let N = arr . length ; document . write ( get_subset_count ( arr , K , N ) ) ;
function minMaxDiff ( arr , n , k ) { var max_adj_dif = - 1000000000 ;
for ( var i = 0 ; i < n - 1 ; i ++ ) max_adj_dif = Math . max ( max_adj_dif , Math . abs ( arr [ i ] - arr [ i + 1 ] ) ) ;
if ( max_adj_dif == 0 ) return 0 ;
var best = 1 ; var worst = max_adj_dif ; var mid , required ; while ( best < worst ) { mid = ( best + worst ) / 2 ;
required = 0 ; for ( var i = 0 ; i < n - 1 ; i ++ ) { required += parseInt ( ( Math . abs ( arr [ i ] - arr [ i + 1 ] ) - 1 ) / mid ) ; }
if ( required > k ) best = mid + 1 ;
else worst = mid ; } return worst ; }
var arr = [ 3 , 12 , 25 , 50 ] ; var n = arr . length ; var k = 7 ; document . write ( minMaxDiff ( arr , n , k ) ) ;
function checkMin ( arr , len ) {
var smallest = Number . INFINITY , secondSmallest = Number . INFINITY ; for ( var i = 0 ; i < len ; i ++ ) {
if ( arr [ i ] < smallest ) { secondSmallest = smallest ; smallest = arr [ i ] ; }
else if ( arr [ i ] < secondSmallest ) { secondSmallest = arr [ i ] ; } } if ( 2 * smallest <= secondSmallest ) document . write ( " " ) ; else document . write ( " " ) ; }
var arr = [ 2 , 3 , 4 , 5 ] ; var len = 4 ; checkMin ( arr , len ) ;
function createHash ( hash , maxElement ) {
let prev = 0 , curr = 1 ; hash . add ( prev ) ; hash . add ( curr ) ; while ( curr <= maxElement ) {
let temp = curr + prev ; hash . add ( temp ) ;
prev = curr ; curr = temp ; } }
function fibonacci ( arr , n ) {
let max_val = Math . max ( ... arr ) ;
let hash = new Set ( ) ; createHash ( hash , max_val ) ;
let minimum = Number . MAX_VALUE ; let maximum = Number . MIN_VALUE ; for ( let i = 0 ; i < n ; i ++ ) {
if ( hash . has ( arr [ i ] ) ) {
minimum = Math . min ( minimum , arr [ i ] ) ; maximum = Math . max ( maximum , arr [ i ] ) ; } } document . write ( minimum + " " + maximum + " " ) ; }
let arr = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; let n = arr . length ; fibonacci ( arr , n ) ;
function isValidLen ( s , len , k ) {
var n = s . length ;
var mp = new Map ( ) ; var right = 0 ;
while ( right < len ) { if ( mp . has ( s [ right ] ) ) mp . set ( s [ right ] , mp . get ( s [ right ] ) + 1 ) else mp . set ( s [ right ] , 1 ) right ++ ; } if ( mp . size <= k ) return true ;
while ( right < n ) {
if ( mp . has ( s [ right ] ) ) mp . set ( s [ right ] , mp . get ( s [ right ] ) + 1 ) else mp . set ( s [ right ] , 1 )
if ( mp . has ( s [ right - len ] ) ) mp . set ( s [ right - len ] , mp . get ( s [ right - len ] ) - 1 )
if ( mp . has ( s [ right - len ] ) && mp . get ( s [ right - len ] ) == 0 ) mp . delete ( s [ right - len ] ) ; if ( mp . size <= k ) return true ; right ++ ; } return mp . size <= k ; }
function maxLenSubStr ( s , k ) {
var uni = new Set ( ) ; s . split ( ' ' ) . forEach ( x => { uni . add ( x ) ; } ) ; if ( uni . size < k ) return - 1 ;
var n = s . length ;
var lo = - 1 , hi = n + 1 ; while ( hi - lo > 1 ) { var mid = lo + hi >> 1 ; if ( isValidLen ( s , mid , k ) ) lo = mid ; else hi = mid ; } return lo ; }
var s = " " ; var k = 3 ; document . write ( maxLenSubStr ( s , k ) ) ;
function isSquarePossible ( arr , n , l ) {
let cnt = 0 ; for ( let i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] >= l ) cnt ++ ;
if ( cnt >= l ) return true ; } return false ; }
function maxArea ( arr , n ) { let l = 0 , r = n ; let len = 0 ; while ( l <= r ) { let m = l + Math . floor ( ( r - l ) / 2 ) ;
if ( isSquarePossible ( arr , n , m ) ) { len = m ; l = m + 1 ; }
else r = m - 1 ; }
return ( len * len ) ; }
let arr = [ 1 , 3 , 4 , 5 , 5 ] ; let n = arr . length ; document . write ( maxArea ( arr , n ) ) ;
function insertNames ( arr , n ) {
let set = new Set ( ) ; for ( let i = 0 ; i < n ; i ++ ) {
if ( ! set . has ( arr [ i ] ) ) { document . write ( " " + " " ) ; set . add ( arr [ i ] ) ; } else { document . write ( " " + " " ) ; } } }
let arr = [ " " , " " , " " ] ; let n = arr . length ; insertNames ( arr , n ) ;
function countLessThan ( arr , n , key ) { let l = 0 , r = n - 1 ; let index = - 1 ;
while ( l <= r ) { let m = Math . floor ( ( l + r ) / 2 ) ; if ( arr [ m ] < key ) { l = m + 1 ; index = m ; } else { r = m - 1 ; } } return ( index + 1 ) ; }
function countGreaterThan ( arr , n , key ) { let l = 0 , r = n - 1 ; let index = - 1 ;
while ( l <= r ) { let m = Math . floor ( ( l + r ) / 2 ) ; if ( arr [ m ] <= key ) { l = m + 1 ; } else { r = m - 1 ; index = m ; } } if ( index == - 1 ) return 0 ; return ( n - index ) ; }
function countTriplets ( n , a , b , c ) {
a . sort ( function ( e , f ) { return e - f ; } ) ; b . sort ( function ( e , f ) { return e - f ; } ) ; c . sort ( function ( e , f ) { return e - f ; } ) ; let count = 0 ;
for ( let i = 0 ; i < n ; ++ i ) { let current = b [ i ] ;
let low = countLessThan ( a , n , current ) ;
let high = countGreaterThan ( c , n , current ) ;
count += ( low * high ) ; } return count ; }
let a = [ 1 , 5 ] ; let b = [ 2 , 4 ] ; let c = [ 3 , 6 ] ; let size = a . length ; document . write ( countTriplets ( size , a , b , c ) ) ;
function costToBalance ( s ) { if ( s . length == 0 ) document . write ( 0 ) ;
var ans = 0 ;
var o = 0 , c = 0 ; for ( var i = 0 ; i < s . length ; i ++ ) { if ( s [ i ] == ' ' ) o ++ ; if ( s [ i ] == ' ' ) c ++ ; } if ( o != c ) return - 1 ; var a = new Array ( s . Length ) ; if ( s [ 0 ] == ' ' ) a [ 0 ] = 1 ; else a [ 0 ] = - 1 ; if ( a [ 0 ] < 0 ) ans += Math . abs ( a [ 0 ] ) ; for ( var i = 1 ; i < s . length ; i ++ ) { if ( s [ i ] == ' ' ) a [ i ] = a [ i - 1 ] + 1 ; else a [ i ] = a [ i - 1 ] - 1 ; if ( a [ i ] < 0 ) ans += Math . abs ( a [ i ] ) ; } return ans ; }
var s ; s = " " ; document . write ( costToBalance ( s ) + " " ) ; s = " " ; document . write ( costToBalance ( s ) + " " ) ;
function middleOfThree ( a , b , c ) {
let x = a - b ;
let y = b - c ;
let z = a - c ;
if ( x * y > 0 ) return b ;
else if ( x * z > 0 ) return c ; else return a ; }
let a = 20 , b = 30 , c = 40 ; document . write ( middleOfThree ( a , b , c ) ) ;
function missing4 ( arr ) {
let helper = [ ] ; for ( let i = 0 ; i < 4 ; i ++ ) { helper [ i ] = 0 ; }
for ( let i = 0 ; i < arr . length ; i ++ ) { let temp = Math . abs ( arr [ i ] ) ;
if ( temp <= arr . length ) arr [ temp - 1 ] = Math . floor ( arr [ temp - 1 ] * ( - 1 ) ) ;
else if ( temp > arr . length ) { if ( temp % arr . length != 0 ) helper [ temp % arr . length - 1 ] = - 1 ; else helper [ ( temp % arr . length ) + arr . length - 1 ] = - 1 ; } }
for ( let i = 0 ; i < arr . length ; i ++ ) if ( arr [ i ] > 0 ) document . write ( i + 1 + " " ) ; for ( let i = 0 ; i < helper . length ; i ++ ) if ( helper [ i ] >= 0 ) document . write ( arr . length + i + 1 + " " ) ; return ; }
let arr = [ 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 ] ; missing4 ( arr ) ;
function lexiMiddleSmallest ( K , N ) {
if ( K % 2 == 0 ) {
document . write ( K / 2 + " " ) ;
for ( let i = 0 ; i < N - 1 ; ++ i ) { document . write ( K + " " ) ; } document . write ( " " ) ; return ; }
let a = [ ] ;
for ( let i = 0 ; i < N / 2 ; ++ i ) {
if ( a [ a . length - 1 ] == 1 ) {
a . pop ( a . length - 1 ) ; }
else {
a [ a . length - 1 ] -= 1 ;
while ( a . length < N ) { a . push ( K ) ; } } }
for ( let i in a ) { document . write ( i + " " ) ; } document . write ( " " ) ; }
let K = 2 , N = 4 ; lexiMiddleSmallest ( K , N ) ;
function findLastElement ( arr , N ) {
arr . sort ( ) ; let i = 0 ;
for ( i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] - arr [ i - 1 ] != 0 && arr [ i ] - arr [ i - 1 ] != 2 ) { document . write ( " " + " " ) ; return ; } }
document . write ( arr [ N - 1 ] + " " ) ; }
let arr = [ 2 , 4 , 6 , 8 , 0 , 8 ] ; let N = arr . length ; findLastElement ( arr , N ) ;
function maxDivisions ( arr , N , X ) {
arr . sort ( ) ;
let maxSub = 0 ;
let size = 0 ;
for ( let i = 0 ; i < N ; i ++ ) {
size ++ ;
if ( arr [ i ] * size >= X ) {
maxSub ++ ;
size = 0 ; } } document . write ( maxSub + " " ) ; }
let arr = [ 1 , 3 , 3 , 7 ] ;
let N = arr . length ;
let X = 3 ; maxDivisions ( arr , N , X ) ;
function maxPossibleSum ( arr , N ) {
arr . sort ( ) ; let sum = 0 ; let j = N - 3 ; while ( j >= 0 ) {
sum += arr [ j ] ; j -= 3 ; }
document . write ( sum ) ; }
let arr = [ 7 , 4 , 5 , 2 , 3 , 1 , 5 , 9 ] ;
let N = arr . length ; maxPossibleSum ( arr , N ) ;
function insertionSort ( arr , n ) { let i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
function printArray ( arr , n ) { let i ;
for ( i = 0 ; i < n ; i ++ ) { document . write ( arr [ i ] + " " ) ; } document . write ( " " ) ; }
let arr = [ 12 , 11 , 13 , 5 , 6 ] ; let N = arr . length ;
insertionSort ( arr , N ) ; printArray ( arr , N ) ;
function getPairs ( arr , N , K ) {
let count = 0 ;
for ( let i = 0 ; i < N ; i ++ ) { for ( let j = i + 1 ; j < N ; j ++ ) {
if ( arr [ i ] > K * arr [ i + 1 ] ) count ++ ; } } document . write ( count ) ; }
let arr = [ 5 , 6 , 2 , 1 ] ; let N = arr . length ; let K = 2 ;
getPairs ( arr , N , K ) ;
function merge ( arr , temp , l , m , r , K ) {
let i = l ;
let j = m + 1 ;
let cnt = 0 ; for ( i = l ; i <= m ; i ++ ) { let found = false ;
while ( j <= r ) {
if ( arr [ i ] >= K * arr [ j ] ) { found = true ; } else break ; j ++ ; }
if ( found == true ) { cnt += j - ( m + 1 ) ; j -- ; } }
let k = l ; i = l ; j = m + 1 ; while ( i <= m && j <= r ) { if ( arr [ i ] <= arr [ j ] ) temp [ k ++ ] = arr [ i ++ ] ; else temp [ k ++ ] = arr [ j ++ ] ; }
while ( i <= m ) temp [ k ++ ] = arr [ i ++ ] ;
while ( j <= r ) temp [ k ++ ] = arr [ j ++ ] ; for ( i = l ; i <= r ; i ++ ) arr [ i ] = temp [ i ] ;
return cnt ; }
function mergeSortUtil ( arr , temp , l , r , K ) { let cnt = 0 ; if ( l < r ) {
let m = parseInt ( ( l + r ) / 2 , 10 ) ;
cnt += mergeSortUtil ( arr , temp , l , m , K ) ; cnt += mergeSortUtil ( arr , temp , m + 1 , r , K ) ;
cnt += merge ( arr , temp , l , m , r , K ) ; } return cnt ; }
function mergeSort ( arr , N , K ) { let temp = new Array ( N ) ; document . write ( mergeSortUtil ( arr , temp , 0 , N - 1 , K ) ) ; }
let arr = [ 5 , 6 , 2 , 5 ] ; let N = arr . length ; let K = 2 ;
mergeSort ( arr , N , K ) ;
function minRemovals ( A , N ) {
A . sort ( ) ;
let mx = A [ N - 1 ] ;
let sum = 1 ;
for ( let i = 0 ; i < N ; i ++ ) { sum += A [ i ] ; } if ( sum - mx >= mx ) { document . write ( 0 ) ; } else { document . write ( 2 * mx - sum ) ; } }
let A = [ 3 , 3 , 2 ] ; let N = A . length ;
minRemovals ( A , N ) ;
function rearrangeArray ( a , n ) {
a . sort ( ) ;
for ( let i = 0 ; i < n - 1 ; i ++ ) {
if ( a [ i ] == i + 1 ) {
let temp = a [ i ] ; a [ i ] = a [ i + 1 ] ; a [ i + 1 ] = temp ; } }
if ( a [ n - 1 ] == n ) {
let temp = a [ n - 1 ] ; a [ n - 1 ] = a [ n - 2 ] ; a [ n - 2 ] = temp ; }
for ( let i = 0 ; i < n ; i ++ ) { document . write ( a [ i ] + " " ) ; } }
let arr = [ 1 , 5 , 3 , 2 , 4 ] ; let N = arr . length ;
rearrangeArray ( arr , N ) ;
function minOperations ( arr1 , arr2 , i , j , n ) {
let f = 0 ; for ( let i = 0 ; i < n ; i ++ ) { if ( arr1 [ i ] != arr2 [ i ] ) f = 1 ; break ; } if ( f == 0 ) return 0 ; if ( i >= n j >= n ) return 0 ;
if ( arr1 [ i ] < arr2 [ j ] )
return 1 + minOperations ( arr1 , arr2 , i + 1 , j + 1 , n ) ;
return Math . max ( minOperations ( arr1 , arr2 , i , j + 1 , n ) , minOperations ( arr1 , arr2 , i + 1 , j , n ) ) ; }
function minOperationsUtil ( arr , n ) { let brr = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) brr [ i ] = arr [ i ] ; brr . sort ( ) ; let f = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { if ( arr [ i ] != brr [ i ] )
f = 1 ; break ; }
if ( f == 1 )
document . write ( minOperations ( arr , brr , 0 , 0 , n ) ) ; else cout << " " ; }
let arr = [ 4 , 7 , 2 , 3 , 9 ] ; let n = arr . length ; minOperationsUtil ( arr , n ) ;
function canTransform ( s , t ) { var n = s . length ;
var occur = Array . from ( Array ( 26 ) , ( ) => new Array ( ) ) ; for ( var x = 0 ; x < n ; x ++ ) { var ch = s [ x ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ; occur [ ch ] . push ( x ) ; }
var idx = Array ( 26 ) . fill ( 0 ) ; var poss = true ; for ( var x = 0 ; x < n ; x ++ ) { var ch = t [ x ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ;
if ( idx [ ch ] >= occur [ ch ] . length ) {
poss = false ; break ; } for ( var small = 0 ; small < ch ; small ++ ) {
if ( idx [ small ] < occur [ small ] . length && occur [ small ] [ idx [ small ] ] < occur [ ch ] [ idx [ ch ] ] ) {
poss = false ; break ; } } idx [ ch ] ++ ; }
if ( poss ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
var s , t ; s = " " ; t = " " ; canTransform ( s , t ) ;
function inversionCount ( s ) {
var freq = Array ( 26 ) . fill ( 0 ) ; var inv = 0 ; for ( var i = 0 ; i < s . length ; i ++ ) { var temp = 0 ;
for ( var j = 0 ; j < String . fromCharCode ( s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ; j ++ )
temp += freq [ j ] ; inv += ( i - temp ) ;
freq [ s [ i ] - ' ' ] ++ ; } return inv ; }
function haveRepeated ( S1 , S2 ) { var freq = Array ( 26 ) . fill ( 0 ) ; S1 . forEach ( i => { if ( freq [ i - ' ' ] > 0 ) return true ; freq [ i - ' ' ] ++ ; } ) ; for ( var i = 0 ; i < 26 ; i ++ ) freq [ i ] = 0 ; S2 . split ( ' ' ) . forEach ( i => { if ( freq [ i - ' ' ] > 0 ) return true ; freq [ i - ' ' ] ++ ; } ) ; return false ; }
function checkToMakeEqual ( S1 , S2 ) {
var freq = Array ( 26 ) . fill ( 0 ) ; for ( var i = 0 ; i < S1 . length ; i ++ ) {
freq [ S1 [ i ] - ' ' ] ++ ; } var flag = 0 ; for ( var i = 0 ; i < S2 . length ; i ++ ) { if ( freq [ S2 [ i ] - ' ' ] == 0 ) {
flag = true ; break ; }
freq [ S2 [ i ] - ' ' ] -- ; } if ( flag == true ) {
document . write ( " " ) ; return ; }
var invCount1 = inversionCount ( S1 ) ; var invCount2 = inversionCount ( S2 ) ; if ( invCount1 == invCount2 || ( invCount1 & 1 ) == ( invCount2 & 1 ) || haveRepeated ( S1 , S2 ) ) {
document . write ( " " ) ; } else document . write ( " " ) ; }
var S1 = " " , S2 = " " ; checkToMakeEqual ( S1 , S2 ) ;
function sortArr ( a , n ) { let i , k ;
k = parseInt ( Math . log ( n ) / Math . log ( 2 ) ) ; k = parseInt ( Math . pow ( 2 , k ) ) ;
while ( k > 0 ) { for ( i = 0 ; i + k < n ; i ++ ) if ( a [ i ] > a [ i + k ] ) { let tmp = a [ i ] ; a [ i ] = a [ i + k ] ; a [ i + k ] = tmp ; }
k = k / 2 ; }
for ( i = 0 ; i < n ; i ++ ) { document . write ( a [ i ] + " " ) ; } }
let arr = [ 5 , 20 , 30 , 40 , 36 , 33 , 25 , 15 , 10 ] ; let n = arr . length ;
sortArr ( arr , n ) ;
function maximumSum ( arr , n , k ) {
let elt = ( n / k ) ; let sum = 0 ;
arr . sort ( ( a , b ) => a - b ) ; let count = 0 ; let i = n - 1 ;
while ( count < k ) { sum += arr [ i ] ; i -- ; count ++ ; } count = 0 ; i = 0 ;
while ( count < k ) { sum += arr [ i ] ; i += elt - 1 ; count ++ ; }
document . write ( sum ) ; }
let Arr = [ 1 , 13 , 7 , 17 , 6 , 5 ] ; let K = 2 ; let size = Arr . length ; maximumSum ( Arr , size , K ) ;
function findMinSum ( arr , K , L , size ) { if ( K * L > size ) return - 1 ; let minsum = 0 ;
arr . sort ( ( a , b ) => a - b ) ;
for ( let i = 0 ; i < K ; i ++ ) minsum += arr [ i ] ;
return minsum ; }
let arr = [ 2 , 15 , 5 , 1 , 35 , 16 , 67 , 10 ] ; let K = 3 ; let L = 2 ; let length = arr . length ; document . write ( findMinSum ( arr , K , L , length ) ) ;
function findKthSmallest ( arr , n , k ) {
let max = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; }
let counter = Array . from ( { length : max + 1 } , ( _ , i ) => 0 ) ;
let smallest = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { counter [ arr [ i ] ] ++ ; }
for ( let num = 1 ; num <= max ; num ++ ) {
if ( counter [ num ] > 0 ) {
smallest += counter [ num ] ; }
if ( smallest >= k ) {
return num ; } } return - 1 ; }
let arr = [ 7 , 1 , 4 , 4 , 20 , 15 , 8 ] ; let N = arr . length ; let K = 5 ;
document . write ( findKthSmallest ( arr , N , K ) ) ;
function lexNumbers ( n ) { let s = [ ] ; for ( let i = 1 ; i <= n ; i ++ ) { s . push ( i . toString ( ) ) ; } s . sort ( ) ; let ans = [ ] ; for ( let i = 0 ; i < n ; i ++ ) ans . push ( parseInt ( s [ i ] ) ) ; for ( let i = 0 ; i < n ; i ++ ) document . write ( ans [ i ] + " " ) ; }
let n = 15 ; lexNumbers ( n ) ;
let N = 4 ; function func ( a ) {
for ( let i = 0 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) { for ( let j = 0 ; j < N ; j ++ ) { for ( let k = j + 1 ; k < N ; ++ k ) {
if ( a [ i ] [ j ] > a [ i ] [ k ] ) {
let temp = a [ i ] [ j ] ; a [ i ] [ j ] = a [ i ] [ k ] ; a [ i ] [ k ] = temp ; } } } }
else { for ( let j = 0 ; j < N ; j ++ ) { for ( let k = j + 1 ; k < N ; ++ k ) {
if ( a [ i ] [ j ] < a [ i ] [ k ] ) {
let temp = a [ i ] [ j ] ; a [ i ] [ j ] = a [ i ] [ k ] ; a [ i ] [ k ] = temp ; } } } } }
for ( let i = 0 ; i < N ; i ++ ) { for ( let j = 0 ; j < N ; j ++ ) { document . write ( " " + a [ i ] [ j ] ) ; } document . write ( " " ) ; } }
let a = [ [ 5 , 7 , 3 , 4 ] , [ 9 , 5 , 8 , 2 ] , [ 6 , 3 , 8 , 1 ] , [ 5 , 8 , 9 , 3 ] ] ; func ( a ) ;
let g = new Array ( 200005 ) ; for ( let i = 0 ; i < 200005 ; i ++ ) g [ i ] = new Map ( ) ; let s = new Set ( ) ; let ns = new Set ( ) ;
function dfs ( x ) { let v = [ ] ;
for ( let it of s . values ( ) ) {
if ( g [ x ] . get ( it ) != null ) { v . push ( it ) ; } else { ns . add ( it ) ; } } s = ns ; for ( let i of v . values ( ) ) { dfs ( i ) ; } }
function weightOfMST ( N ) {
let cnt = 0 ;
for ( let i = 1 ; i <= N ; ++ i ) { s . add ( i ) ; } let qt = [ ] for ( let t of s . values ( ) ) qt . push ( t ) ;
while ( qt . length != 0 ) {
++ cnt ; let t = qt [ 0 ] ; qt . shift ( ) ;
dfs ( t ) ; } document . write ( cnt - 4 ) ; }
let N = 6 , M = 11 ; let edges = [ [ 1 , 3 ] , [ 1 , 4 ] , [ 1 , 5 ] , [ 1 , 6 ] , [ 2 , 3 ] , [ 2 , 4 ] , [ 2 , 5 ] , [ 2 , 6 ] , [ 3 , 4 ] , [ 3 , 5 ] , [ 3 , 6 ] ] ;
for ( let i = 0 ; i < M ; ++ i ) { let u = edges [ i ] [ 0 ] ; let v = edges [ i ] [ 1 ] ; g [ u ] . set ( v , 1 ) ; g [ v ] . set ( u , 1 ) ; }
weightOfMST ( N ) ;
function countPairs ( A , B ) { let n = A . length ; let ans = 0 ; A . sort ( ) ; B . sort ( ) ; for ( let i = 0 ; i < n ; i ++ ) { if ( A [ i ] > B [ ans ] ) { ans ++ ; } } return ans ; }
let A = [ 30 , 28 , 45 , 22 ] ; let B = [ 35 , 25 , 22 , 48 ] ; document . write ( countPairs ( A , B ) ) ;
function maxMod ( arr , n ) { let maxVal = arr . sort ( ( a , b ) => b - a ) [ 0 ] let secondMax = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { if ( arr [ i ] < maxVal && arr [ i ] > secondMax ) { secondMax = arr [ i ] ; } } return secondMax ; }
let arr = [ 2 , 4 , 1 , 5 , 3 , 6 ] ; let n = arr . length ; document . write ( maxMod ( arr , n ) ) ;
function isPossible ( A , B , n , m , x , y ) {
if ( x > n y > m ) return false ;
A . sort ( ) ; B . sort ( ) ;
if ( A [ x - 1 ] < B [ m - y ] ) return true ; else return false ; }
var A = [ 1 , 1 , 1 , 1 , 1 ] ; var B = [ 2 , 2 ] ; var n = A . length ; var m = B . length ; ; var x = 3 , y = 1 ; if ( isPossible ( A , B , n , m , x , y ) ) document . write ( " " ) ; else document . write ( " " ) ;
var MAX = 100005 ;
function Min_Replace ( arr , n , k ) { arr . sort ( ( a , b ) => a - b )
var freq = Array ( MAX ) . fill ( 0 ) ; var p = 0 ; freq [ p ] = 1 ;
for ( var i = 1 ; i < n ; i ++ ) { if ( arr [ i ] == arr [ i - 1 ] ) ++ freq [ p ] ; else ++ freq [ ++ p ] ; }
freq . sort ( ( a , b ) => b - a ) ;
var ans = 0 ; for ( var i = k ; i <= p ; i ++ ) ans += freq [ i ] ;
return ans ; }
var arr = [ 1 , 2 , 7 , 8 , 2 , 3 , 2 , 3 ] ; var n = arr . length ; var k = 2 ; document . write ( Min_Replace ( arr , n , k ) ) ;
function Segment ( x , l , n ) {
if ( n == 1 ) return 1 ;
let ans = 2 ; for ( let i = 1 ; i < n - 1 ; i ++ ) {
if ( x [ i ] - l [ i ] > x [ i - 1 ] ) ans ++ ;
else if ( x [ i ] + l [ i ] < x [ i + 1 ] ) {
x [ i ] = x [ i ] + l [ i ] ; ans ++ ; } }
return ans ; }
let x = [ 1 , 3 , 4 , 5 , 8 ] , l = [ 10 , 1 , 2 , 2 , 5 ] ; let n = x . length ;
document . write ( Segment ( x , l , n ) ) ;
function MinimizeleftOverSum ( a , n ) { var v1 = [ ] , v2 = [ ] ; for ( i = 0 ; i < n ; i ++ ) { if ( a [ i ] % 2 == 1 ) v1 . push ( a [ i ] ) ; else v2 . push ( a [ i ] ) ; }
if ( v1 . length > v2 . length ) {
v1 . sort ( ) ; v2 . sort ( ) ;
var x = v1 . length - v2 . length - 1 ; var sum = 0 ; var i = 0 ;
while ( i < x ) { sum += v1 [ i ++ ] ; }
return sum ; }
else if ( v2 . length > v1 . length ) {
v1 . sort ( ) ; v2 . sort ( ) ;
var x = v2 . length - v1 . length - 1 ; var sum = 0 ; var i = 0 ;
while ( i < x ) { sum += v2 [ i ++ ] ; }
return sum ; }
else return 0 ; }
var a = [ 2 , 2 , 2 , 2 ] ; var n = a . length ; document . write ( MinimizeleftOverSum ( a , n ) ) ;
function minOperation ( S , N , K ) {
if ( N % K ) { document . write ( " " ) ; return ; }
var count = Array ( 26 ) . fill ( 0 ) ; for ( var i = 0 ; i < N ; i ++ ) { count [ S [ i ] . charCodeAt ( 0 ) - 97 ] ++ ; } var E = N / K ; var greaterE = [ ] ; var lessE = [ ] ; for ( var i = 0 ; i < 26 ; i ++ ) {
if ( count [ i ] < E ) lessE . push ( E - count [ i ] ) ; else greaterE . push ( count [ i ] - E ) ; } greaterE . sort ( ) ; lessE . sort ( ) ; var mi = 1000000000 ; for ( var i = 0 ; i <= K ; i ++ ) {
var set1 = i ; var set2 = K - i ; if ( greaterE . length >= set1 && lessE . length >= set2 ) { var step1 = 0 ; var step2 = 0 ; for ( var j = 0 ; j < set1 ; j ++ ) step1 += greaterE [ j ] ; for ( var j = 0 ; j < set2 ; j ++ ) step2 += lessE [ j ] ; mi = Math . min ( mi , Math . max ( step1 , step2 ) ) ; } } document . write ( mi ) ; }
var S = " " ; var N = S . length ; var K = 2 ; minOperation ( S , N , K ) ;
function minMovesToSort ( arr , n ) { var moves = 0 ; var i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
mn = arr [ i ] ; } return moves ; }
var arr = [ 3 , 5 , 2 , 8 , 4 ] ; var n = arr . length ; document . write ( minMovesToSort ( arr , n ) ) ;
var prime = Array ( 100005 ) . fill ( true ) ; function SieveOfEratosthenes ( n ) {
prime [ 1 ] = false ; for ( var p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] ) {
for ( var i = p * 2 ; i <= n ; i += p ) prime [ i ] = false ; } } }
function sortPrimes ( arr , n ) { SieveOfEratosthenes ( 100005 ) ;
var v = [ ] ; for ( var i = 0 ; i < n ; i ++ ) {
if ( prime [ arr [ i ] ] ) v . push ( arr [ i ] ) ; } v . sort ( ( a , b ) => b - a ) var j = 0 ;
for ( var i = 0 ; i < n ; i ++ ) { if ( prime [ arr [ i ] ] ) arr [ i ] = v [ j ++ ] ; } }
var arr = [ 4 , 3 , 2 , 6 , 100 , 17 ] ; var n = arr . length ; sortPrimes ( arr , n ) ;
for ( var i = 0 ; i < n ; i ++ ) { document . write ( arr [ i ] + " " ) ; }
function findOptimalPairs ( arr , N ) { arr . sort ( function ( a , b ) { return a - b ; } ) ;
for ( var i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) document . write ( " " + arr [ i ] + " " + arr [ j ] + " " + " " ) ; }
var arr = [ 9 , 6 , 5 , 1 ] ; var N = arr . length ; findOptimalPairs ( arr , N ) ;
function countBits ( a ) { let count = 0 ; while ( a > 0 ) { if ( ( a & 1 ) > 0 ) count += 1 ; a = a >> 1 ; } return count ; }
function insertionSort ( arr , aux , n ) { for ( let i = 1 ; i < n ; i ++ ) {
let key1 = aux [ i ] ; let key2 = arr [ i ] ; let j = i - 1 ;
while ( j >= 0 && aux [ j ] < key1 ) { aux [ j + 1 ] = aux [ j ] ; arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } aux [ j + 1 ] = key1 ; arr [ j + 1 ] = key2 ; } }
function sortBySetBitCount ( arr , n ) {
let aux = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) aux [ i ] = countBits ( arr [ i ] ) ;
insertionSort ( arr , aux , n ) ; }
function printArr ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 1 , 2 , 3 , 4 , 5 , 6 ] ; let n = arr . length ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ;
function countBits ( a ) { let count = 0 ; while ( a > 0 ) { if ( ( a & 1 ) > 0 ) count += 1 ; a = a >> 1 ; } return count ; }
function sortBySetBitCount ( arr , n ) { let count = new Array ( 32 ) ; for ( let i = 0 ; i < count . length ; i ++ ) count [ i ] = [ ] ; let setbitcount = 0 ; for ( let i = 0 ; i < n ; i ++ ) { setbitcount = countBits ( arr [ i ] ) ; count [ setbitcount ] . push ( arr [ i ] ) ; }
for ( let i = 31 ; i >= 0 ; i -- ) { let v1 = count [ i ] ; for ( let p = 0 ; p < v1 . length ; p ++ ) arr [ j ++ ] = v1 [ p ] ; } }
function printArr ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 1 , 2 , 3 , 4 , 5 , 6 ] ; let n = arr . length ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ;
function generateString ( k1 , k2 , s ) {
let C1s = 0 , C0s = 0 ; let flag = 0 ; let pos = [ ] ;
for ( let i = 0 ; i < s . length ; i ++ ) { if ( s [ i ] == ' ' ) { C0s ++ ;
if ( ( i + 1 ) % k1 != 0 && ( i + 1 ) % k2 != 0 ) { pos . push ( i ) ; } } else { C1s ++ ; } if ( C0s >= C1s ) {
if ( pos . length == 0 ) { cout << - 1 ; flag = 1 ; break ; }
else { let k = pos [ pos . length - 1 ] ; var ns = s . replace ( s [ k ] , ' ' ) ; C0s -- ; C1s ++ ; pos . pop ( ) ; } } }
if ( flag == 0 ) { document . write ( ns ) ; } }
let K1 = 2 , K2 = 4 ; let S = " " ; generateString ( K1 , K2 , S ) ;
function maximizeProduct ( N ) {
let MSB = Math . log2 ( N ) ;
let X = 1 << MSB ;
let Y = N - ( 1 << MSB ) ;
for ( let i = 0 ; i < MSB ; i ++ ) {
if ( ! ( N & ( 1 << i ) ) ) {
X += 1 << i ;
Y += 1 << i ; } }
document . write ( X + " " + Y ) ; }
let N = 45 ; maximizeProduct ( N ) ;
function check ( num ) {
let sm = 0 ;
let num2 = num * num ; while ( num ) { sm += num % 10 ; num = Math . floor ( num / 10 ) ; }
let sm2 = 0 ; while ( num2 ) { sm2 += num2 % 10 ; num2 = Math . floor ( num2 / 10 ) ; } return sm * sm == sm2 ; }
function convert ( s ) { let val = 0 ; s = s . split ( " " ) . reverse ( ) . join ( " " ) ; let cur = 1 ; for ( let i = 0 ; i < s . length ; i ++ ) { val += ( s [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ) * cur ; cur *= 10 ; } return val ; }
function generate ( s , len , uniq ) {
if ( s . length == len ) {
if ( check ( convert ( s ) ) ) { uniq . add ( convert ( s ) ) ; } return ; }
for ( let i = 0 ; i <= 3 ; i ++ ) { generate ( s + String . fromCharCode ( i + " " . charCodeAt ( 0 ) ) , len , uniq ) ; } }
function totalNumbers ( L , R ) {
let ans = 0 ;
let max_len = Math . log10 ( R ) + 1 ;
let uniq = new Set ( ) ; for ( let i = 1 ; i <= max_len ; i ++ ) {
generate ( " " , i , uniq ) ; }
for ( let x of uniq ) { if ( x >= L && x <= R ) { ans ++ ; } } return ans ; }
let L = 22 , R = 22 ; document . write ( totalNumbers ( L , R ) ) ;
function convertXintoY ( X , Y ) {
while ( Y > X ) {
if ( Y % 2 == 0 ) Y = parseInt ( Y / 2 ) ;
else if ( Y % 10 == 1 ) Y = parseInt ( Y /= 10 ) ;
else break ; }
if ( X == Y ) document . write ( " " ) ; else document . write ( " " ) ; }
let X = 100 , Y = 40021 ; convertXintoY ( X , Y ) ;
function generateString ( K ) {
var s = " " ;
for ( var i = 97 ; i < 97 + K ; i ++ ) { s = s + String . fromCharCode ( i ) ;
for ( var j = i + 1 ; j < 97 + K ; j ++ ) { s += String . fromCharCode ( i ) ; s += String . fromCharCode ( j ) ; } }
s += String . fromCharCode ( 97 ) ;
document . write ( s ) ; }
var K = 4 ; generateString ( K ) ;
function findEquation ( S , M ) {
document . write ( " " + ( ( - 1 ) * S ) + " " + M ) ; }
var S = 5 , M = 6 ; findEquation ( S , M ) ;
function minSteps ( a , n ) {
var prefix_sum = Array ( n ) . fill ( 0 ) ; prefix_sum [ 0 ] = a [ 0 ] ;
for ( var i = 1 ; i < n ; i ++ ) prefix_sum [ i ] += prefix_sum [ i - 1 ] + a [ i ] ;
var mx = - 1 ;
for ( var subgroupsum = 0 ; subgroupsum < prefix_sum . length ; subgroupsum ++ ) { var sum = 0 ; var i = 0 ; var grp_count = 0 ;
while ( i < n ) { sum += a [ i ] ;
if ( sum == prefix_sum [ subgroupsum ] ) {
grp_count += 1 ; sum = 0 ; }
else if ( sum > prefix_sum [ subgroupsum ] ) { grp_count = - 1 ; break ; } i += 1 ; }
if ( grp_count > mx ) mx = grp_count ; }
return n - mx ; }
var A = [ 1 , 2 , 3 , 2 , 1 , 3 ] ; var N = A . length ;
document . write ( minSteps ( A , N ) ) ;
function maxOccuringCharacter ( s ) {
var count0 = 0 , count1 = 0 ;
for ( var i = 0 ; i < s . length ; i ++ ) {
if ( s . charAt ( i ) == ' ' ) { count1 ++ ; }
else if ( s . charAt ( i ) == ' ' ) { count0 ++ ; } }
var prev = - 1 ; for ( var i = 0 ; i < s . length ; i ++ ) { if ( s . charAt ( i ) == ' ' ) { prev = i ; break ; } }
for ( var i = prev + 1 ; i < s . length ; i ++ ) {
if ( s . charAt ( i ) != ' ' ) {
if ( s . charAt ( i ) == ' ' ) { count1 += i - prev - 1 ; prev = i ; }
else {
flag = true ; for ( var j = i + 1 ; j < s . length ; j ++ ) { if ( s . charAt ( j ) == ' ' ) { flag = false ; prev = j ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . length ; } } } }
prev = - 1 ; for ( var i = 0 ; i < s . length ; i ++ ) { if ( s . charAt ( i ) == ' ' ) { prev = i ; break ; } }
for ( var i = prev + 1 ; i < s . length ; i ++ ) {
if ( s . charAt ( i ) != ' ' ) {
if ( s . charAt ( i ) == ' ' ) {
count0 += i - prev - 1 ;
prev = i ; }
else {
flag = true ; for ( var j = i + 1 ; j < s . length ; j ++ ) { if ( s . charAt ( j ) == ' ' ) { prev = j ; flag = false ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . length ; } } } }
if ( s . charAt ( 0 ) == ' ' ) {
var count = 0 ; var i = 0 ; while ( s . charAt ( i ) == ' ' ) { count ++ ; i ++ ; }
if ( s . charAt ( i ) == ' ' ) { count1 += count ; } }
if ( s . charAt ( s . length - 1 ) == ' ' ) {
var count = 0 ; var i = s . length - 1 ; while ( s . charAt ( i ) == ' ' ) { count ++ ; i -- ; }
if ( s . charAt ( i ) == ' ' ) { count0 += count ; } }
if ( count0 == count1 ) { document . write ( " " ) ; }
else if ( count0 > count1 ) { document . write ( 0 ) ; }
else document . write ( 1 ) ; }
var S = " " ; maxOccuringCharacter ( S ) ;
function maxSheets ( A , B ) { let area = A * B ;
let count = 1 ;
while ( area % 2 == 0 ) {
area /= 2 ;
count *= 2 ; } return count ; }
let A = 5 , B = 10 ; document . write ( maxSheets ( A , B ) ) ;
function findMinMoves ( a , b ) {
let ans = 0 ;
if ( a == b || Math . abs ( a - b ) == 1 ) { ans = a + b ; } else {
let k = Math . min ( a , b ) ;
let j = Math . max ( a , b ) ; ans = 2 * k + 2 * ( j - k ) - 1 ; }
document . write ( ans ) ; }
let a = 3 , b = 5 ;
findMinMoves ( a , b ) ;
function cntEvenSumPairs ( X , Y ) {
var cntXEvenNums = parseInt ( X / 2 ) ;
var cntXOddNums = parseInt ( ( X + 1 ) / 2 ) ;
var cntYEvenNums = parseInt ( Y / 2 ) ;
var cntYOddNums = parseInt ( ( Y + 1 ) / 2 ) ;
var cntPairs = ( cntXEvenNums * cntYEvenNums ) + ( cntXOddNums * cntYOddNums ) ;
return cntPairs ; }
var X = 2 ; var Y = 3 ; document . write ( cntEvenSumPairs ( X , Y ) ) ;
function minMoves ( arr ) { let N = arr . length ;
if ( N <= 2 ) return 0 ;
let ans = Number . MAX_VALUE ;
for ( let i = - 1 ; i <= 1 ; i ++ ) { for ( let j = - 1 ; j <= 1 ; j ++ ) {
let num1 = arr [ 0 ] + i ;
let num2 = arr [ 1 ] + j ; let flag = 1 ; let moves = Math . abs ( i ) + Math . abs ( j ) ;
for ( let idx = 2 ; idx < N ; idx ++ ) {
let num = num1 + num2 ;
if ( Math . abs ( arr [ idx ] - num ) > 1 ) flag = 0 ;
else moves += Math . abs ( arr [ idx ] - num ) ; num1 = num2 ; num2 = num ; }
if ( flag > 0 ) ans = Math . min ( ans , moves ) ; } }
if ( ans == Number . MAX_VALUE ) return - 1 ; return ans ; }
let arr = [ 4 , 8 , 9 , 17 , 27 ] ; document . write ( minMoves ( arr ) ) ;
function querySum ( arr , N , Q , M ) {
for ( let i = 0 ; i < M ; i ++ ) { let x = Q [ i ] [ 0 ] ; let y = Q [ i ] [ 1 ] ;
let sum = 0 ;
while ( x < N ) {
sum += arr [ x ] ;
x += y ; } document . write ( sum + " " ) ; } }
let arr = [ 1 , 2 , 7 , 5 , 4 ] ; let Q = [ [ 2 , 1 ] , [ 3 , 2 ] ] ; let N = arr . length ; let M = Q . length ; querySum ( arr , N , Q , M ) ;
function findBitwiseORGivenXORAND ( X , Y ) { return X + Y ; }
let X = 5 , Y = 2 ; document . write ( findBitwiseORGivenXORAND ( X , Y ) ) ;
function GCD ( a , b ) {
if ( b == 0 ) return a ;
return GCD ( b , a % b ) ; }
function canReach ( N , A , B , K ) {
var gcd = GCD ( N , K ) ;
if ( Math . abs ( A - B ) % gcd == 0 ) { document . write ( " " ) ; }
else { document . write ( " " ) ; } }
var N = 5 , A = 2 , B = 1 , K = 2 ;
canReach ( N , A , B , K ) ;
function countOfSubarray ( arr , N ) {
var mp = new Map ( ) ;
var answer = 0 ;
var sum = 0 ;
if ( ! mp . has ( 1 ) ) mp . set ( 1 , 1 ) else mp . set ( 1 , mp . get ( 1 ) + 1 )
for ( var i = 0 ; i < N ; i ++ ) {
sum += arr [ i ] ; answer += mp . has ( sum - i ) ? mp . get ( sum - i ) : 0 ;
if ( mp . has ( sum - i ) ) mp . set ( sum - i , mp . get ( sum - i ) + 1 ) else mp . set ( sum - i , 1 ) }
document . write ( answer ) ; }
var arr = [ 1 , 0 , 2 , 1 , 2 , - 2 , 2 , 4 ] ;
var N = arr . length ;
countOfSubarray ( arr , N ) ;
function minAbsDiff ( N ) {
var sumSet1 = 0 ;
var sumSet2 = 0 ;
for ( i = N ; i > 0 ; i -- ) {
if ( sumSet1 <= sumSet2 ) { sumSet1 += i ; } else { sumSet2 += i ; } } return Math . abs ( sumSet1 - sumSet2 ) ; }
var N = 6 ; document . write ( minAbsDiff ( N ) ) ;
function checkDigits ( n ) {
do { var r = n % 10 ;
if ( r == 3 r == 4 r == 6 r == 7 r == 9 ) return false ; n = parseInt ( n / 10 ) ; } while ( n != 0 ) ; return true ; }
function isPrime ( n ) { if ( n <= 1 ) return false ;
for ( var i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; } return true ; }
function isAllPrime ( n ) { return isPrime ( n ) && checkDigits ( n ) ; }
var N = 101 ; if ( isAllPrime ( N ) ) document . write ( " " ) ; else document . write ( " " ) ;
function minCost ( str , a , b ) {
let openUnbalanced = 0 ;
let closedUnbalanced = 0 ;
let openCount = 0 ;
let closedCount = 0 ; for ( let i = 0 ; i < str . length ; i ++ ) {
if ( str [ i ] == ' ' ) { openUnbalanced ++ ; openCount ++ ; }
else {
if ( openUnbalanced == 0 )
closedUnbalanced ++ ;
else
openUnbalanced -- ;
closedCount ++ ; } }
let result = a * ( Math . abs ( openCount - closedCount ) ) ;
if ( closedCount > openCount ) closedUnbalanced -= ( closedCount - openCount ) ; if ( openCount > closedCount ) openUnbalanced -= ( openCount - closedCount ) ;
result += Math . min ( a * ( openUnbalanced + closedUnbalanced ) , b * closedUnbalanced ) ;
document . write ( result + " " ) ; }
let str = " " ; let A = 1 , B = 3 ; minCost ( str , A , B ) ;
function countEvenSum ( low , high , k ) {
let even_count = high / 2 - ( low - 1 ) / 2 ; let odd_count = ( high + 1 ) / 2 - low / 2 ; let even_sum = 1 ; let odd_sum = 0 ;
for ( let i = 0 ; i < k ; i ++ ) {
let prev_even = even_sum ; let prev_odd = odd_sum ;
even_sum = ( prev_even * even_count ) + ( prev_odd * odd_count ) ;
odd_sum = ( prev_even * odd_count ) + ( prev_odd * even_count ) ; }
document . write ( even_sum ) ; }
let low = 4 ; let high = 5 ;
let K = 3 ;
countEvenSum ( low , high , K ) ;
function count ( n , k ) { let count = Math . pow ( 10 , k ) - Math . pow ( 10 , k - 1 ) ;
document . write ( count ) ; }
let n = 2 , k = 1 ; count ( n , k ) ;
function func ( N , P ) {
let sumUptoN = ( N * ( N + 1 ) / 2 ) ; let sumOfMultiplesOfP ;
if ( N < P ) { return sumUptoN ; }
else if ( ( N / P ) == 1 ) { return sumUptoN - P + 1 ; }
sumOfMultiplesOfP = ( ( N / P ) * ( 2 * P + ( N / P - 1 ) * P ) ) / 2 ;
return ( sumUptoN + func ( N / P , P ) - sumOfMultiplesOfP ) ; }
let N = 10 , P = 5 ;
document . write ( func ( N , P ) ) ;
function findShifts ( A , N ) {
let shift = Array . from ( { length : N } , ( _ , i ) => 0 ) ; for ( let i = 0 ; i < N ; i ++ ) {
if ( i == A [ i ] - 1 ) shift [ i ] = 0 ;
else
shift [ i ] = ( A [ i ] - 1 - i + N ) % N ; }
for ( let i = 0 ; i < N ; i ++ ) document . write ( shift [ i ] + " " ) ; }
let arr = [ 1 , 4 , 3 , 2 , 5 ] ; let N = arr . length ; findShifts ( arr , N ) ;
function constructmatrix ( N ) { let check = true ; for ( let i = 0 ; i < N ; i ++ ) { for ( let j = 0 ; j < N ; j ++ ) {
if ( i == j ) { document . write ( " " ) ; } else if ( check ) {
document . write ( " " ) ; check = false ; } else {
document . write ( " " ) ; check = true ; } } document . write ( " " ) ; } }
let N = 5 ; constructmatrix ( 5 ) ;
function check ( unit_digit , X ) { let times , digit ;
for ( times = 1 ; times <= 10 ; times ++ ) { digit = ( X * times ) % 10 ; if ( digit == unit_digit ) return times ; }
return - 1 ; }
function getNum ( N , X ) { let unit_digit ;
unit_digit = N % 10 ;
let times = check ( unit_digit , X ) ;
if ( times == - 1 ) return times ;
else {
if ( N >= ( times * X ) )
return times ;
else return - 1 ; } }
let N = 58 , X = 7 ; document . write ( getNum ( N , X ) ) ;
function minPolets ( n , m ) { let ans = 0 ;
if ( ( n % 2 != 0 ) && ( m % 2 != 0 ) ) { ans = Math . floor ( ( n * m ) / 2 ) + 1 ; } else { ans = Math . floor ( ( n * m ) / 2 ) ; }
return ans ; }
let N = 5 , M = 7 ;
document . write ( minPolets ( N , M ) ) ;
function getLargestString ( s , k ) {
let frequency_array = new Array ( 26 ) ; for ( let i = 0 ; i < 26 ; i ++ ) { frequency_array [ i ] = 0 ; }
for ( let i = 0 ; i < s . length ; i ++ ) { frequency_array [ s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; }
let ans = " " ;
for ( let i = 25 ; i >= 0 ; ) {
if ( frequency_array [ i ] > k ) {
let temp = k ; let st = String . fromCharCode ( i + ' ' . charCodeAt ( 0 ) ) ; while ( temp > 0 ) {
ans += st ; temp -- ; } frequency_array [ i ] -= k ;
let j = i - 1 ; while ( frequency_array [ j ] <= 0 && j >= 0 ) { j -- ; }
if ( frequency_array [ j ] > 0 && j >= 0 ) { let str = String . fromCharCode ( j + ' ' . charCodeAt ( 0 ) ) ; ans += str ; frequency_array [ j ] -= 1 ; } else {
break ; } }
else if ( frequency_array [ i ] > 0 ) {
let temp = frequency_array [ i ] ; frequency_array [ i ] -= temp ; let st = String . fromCharCode ( i + ' ' . charCodeAt ( 0 ) ) ; while ( temp > 0 ) { ans += st ; temp -- ; } }
else { i -- ; } } return ans ; }
let S = " " ; let k = 3 ; document . write ( getLargestString ( S , k ) ) ;
function minOperations ( a , b , n ) {
var minA = Math . max . apply ( Math , a ) ; ;
for ( x = minA ; x >= 0 ; x -- ) {
var check = true ;
var operations = 0 ;
for ( i = 0 ; i < n ; i ++ ) { if ( x % b [ i ] == a [ i ] % b [ i ] ) { operations += ( a [ i ] - x ) / b [ i ] ; }
else { check = false ; break ; } } if ( check ) return operations ; } return - 1 ; }
var N = 5 ; var A = [ 5 , 7 , 10 , 5 , 15 ] ; var B = [ 2 , 2 , 1 , 3 , 5 ] ; document . write ( minOperations ( A , B , N ) ) ;
function getLargestSum ( N ) {
var max_sum = 0 ;
for ( i = 1 ; i <= N ; i ++ ) { for ( j = i + 1 ; j <= N ; j ++ ) {
if ( i * j % ( i + j ) == 0 )
max_sum = Math . max ( max_sum , i + j ) ; } }
return max_sum ; }
var N = 25 ; var max_sum = getLargestSum ( N ) ; document . write ( max_sum ) ;
function maxSubArraySum ( a , size ) { var max_so_far = Number . MIN_VALUE , max_ending_here = 0 ;
for ( i = 0 ; i < size ; i ++ ) { max_ending_here = max_ending_here + a [ i ] ; if ( max_ending_here < 0 ) max_ending_here = 0 ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; } return max_so_far ; }
function maxSum ( a , n ) {
var S = 0 ; var i ;
for ( i = 0 ; i < n ; i ++ ) S += a [ i ] ; var X = maxSubArraySum ( a , n ) ;
return 2 * X - S ; }
var a = [ - 1 , - 2 , - 3 ] ; var n = a . length ; var max_sum = maxSum ( a , n ) ; document . write ( max_sum ) ;
function isPrime ( n ) { let flag = 1 ;
for ( let i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) { flag = 0 ; break ; } } return ( flag == 1 ? true : false ) ; }
function isPerfectSquare ( x ) {
let sr = Math . sqrt ( x ) ;
return ( ( sr - Math . floor ( sr ) ) == 0 ) ; }
function countInterestingPrimes ( n ) { let answer = 0 ; for ( let i = 2 ; i <= n ; i ++ ) {
if ( isPrime ( i ) ) {
for ( let j = 1 ; j * j * j * j <= i ; j ++ ) {
if ( isPerfectSquare ( i - j * j * j * j ) ) { answer ++ ; break ; } } } }
return answer ; }
let N = 10 ; document . write ( countInterestingPrimes ( N ) ) ;
function decBinary ( arr , n ) { let k = Math . log2 ( n ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n = Math . floor ( n / 2 ) ; } }
function binaryDec ( arr , n ) { let ans = 0 ; for ( let i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
function maxNum ( n , k ) {
let l = Math . log2 ( n ) + 1 ;
let a = new Array ( l ) . fill ( 0 ) ; decBinary ( a , n ) ;
let cn = 0 ; for ( let i = 0 ; i < l ; i ++ ) { if ( a [ i ] == 0 && cn < k ) { a [ i ] = 1 ; cn ++ ; } }
return binaryDec ( a , l ) ; }
let n = 4 , k = 1 ; document . write ( maxNum ( n , k ) ) ;
function findSubSeq ( arr , n , sum ) { for ( let i = n - 1 ; i >= 0 ; i -- ) {
if ( sum < arr [ i ] ) arr [ i ] = - 1 ;
else sum -= arr [ i ] ; }
for ( let i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != - 1 ) document . write ( arr [ i ] + " " ) ; } }
let arr = [ 17 , 25 , 46 , 94 , 201 , 400 ] ; let n = arr . length ; let sum = 272 ; findSubSeq ( arr , n , sum ) ;
const MAX = 26 ;
function maxAlpha ( str , len ) {
for ( var i = 0 ; i < MAX ; i ++ ) { first [ i ] = - 1 ; last [ i ] = - 1 ; }
for ( var i = 0 ; i < len ; i ++ ) { var index = str [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ;
if ( first [ index ] === - 1 ) first [ index ] = i ; last [ index ] = i ; }
var ans = - 1 , maxVal = - 1 ;
for ( var i = 0 ; i < MAX ; i ++ ) {
if ( first [ i ] === - 1 ) continue ;
if ( last [ i ] - first [ i ] > maxVal ) { maxVal = last [ i ] - first [ i ] ; ans = i ; } } return String . fromCharCode ( ans + " " . charCodeAt ( 0 ) ) ; }
var str = " " ; var len = str . length ; document . write ( maxAlpha ( str , len ) ) ;
function find_distinct ( a , n , q , queries ) { let MAX = 100001 ; let check = new Array ( MAX ) . fill ( 0 ) ; let idx = new Array ( MAX ) . fill ( 0 ) ; let cnt = 1 ; let i = n - 1 ; while ( i >= 0 ) {
if ( check [ a [ i ] ] == 0 ) {
idx [ i ] = cnt ; check [ a [ i ] ] = 1 ; cnt ++ ; } else {
idx [ i ] = cnt - 1 ; } i -- ; }
for ( let i = 0 ; i < q ; i ++ ) { let m = queries [ i ] ; document . write ( idx [ m ] + " " ) ; } }
let a = [ 1 , 2 , 3 , 1 , 2 , 3 , 4 , 5 ] ; let n = a . length ; let queries = [ 0 , 3 , 5 , 7 ] ; let q = queries . length ; find_distinct ( a , n , q , queries ) ;
const MAX = 24 ;
function countOp ( x ) {
let arr = new Array ( MAX ) ; arr [ 0 ] = 1 ; for ( let i = 1 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] * 2 ;
let temp = x ; let flag = true ;
let ans ;
let operations = 0 ; let flag2 = false ; for ( let i = 0 ; i < MAX ; i ++ ) { if ( arr [ i ] - 1 == x ) flag2 = true ;
if ( arr [ i ] > x ) { ans = i ; break ; } }
if ( flag2 ) return 0 ; while ( flag ) {
if ( arr [ ans ] < x ) ans ++ ; operations ++ ;
for ( let i = 0 ; i < MAX ; i ++ ) { let take = x ^ ( arr [ i ] - 1 ) ; if ( take <= arr [ ans ] - 1 ) {
if ( take > temp ) temp = take ; } }
if ( temp == arr [ ans ] - 1 ) { flag = false ; break ; } temp ++ ; operations ++ ; x = temp ; if ( x == arr [ ans ] - 1 ) flag = false ; }
return operations ; }
let x = 39 ; document . write ( countOp ( x ) ) ;
function minOperations ( arr , n ) { let maxi , result = 0 ;
let freq = new Array ( 1000001 ) . fill ( 0 ) ; for ( let i = 0 ; i < n ; i ++ ) { let x = arr [ i ] ; freq [ x ] ++ ; }
maxi = Math . max ( ... arr ) ; for ( let i = 1 ; i <= maxi ; i ++ ) { if ( freq [ i ] != 0 ) {
for ( let j = i * 2 ; j <= maxi ; j = j + i ) {
freq [ j ] = 0 ; }
result ++ ; } } return result ; }
let arr = [ 2 , 4 , 2 , 4 , 4 , 4 ] ; let n = arr . length ; document . write ( minOperations ( arr , n ) ) ;
function __gcd ( a , b ) { if ( a == 0 ) return b ; return __gcd ( b % a , a ) ; } function minGCD ( arr , n ) { var minGCD = 0 ;
for ( i = 0 ; i < n ; i ++ ) minGCD = __gcd ( minGCD , arr [ i ] ) ; return minGCD ; }
function minLCM ( arr , n ) { var minLCM = arr [ 0 ] ;
for ( i = 1 ; i < n ; i ++ ) minLCM = Math . min ( minLCM , arr [ i ] ) ; return minLCM ; }
var arr = [ 2 , 66 , 14 , 521 ] ; var n = arr . length ; document . write ( " " + minLCM ( arr , n ) + " " + minGCD ( arr , n ) ) ;
function formStringMinOperations ( s ) {
var count = new Array ( 3 ) . fill ( 0 ) ; for ( const c of s ) { count += 1 ; }
var processed = new Array ( 3 ) . fill ( 0 ) ;
var reqd = parseInt ( s . length / 3 ) ; for ( var i = 0 ; i < s . length ; i ++ ) {
if ( count [ s [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] === reqd ) { continue ; }
if ( s [ i ] === " " && count [ 0 ] > reqd && processed [ 0 ] >= reqd ) {
if ( count [ 1 ] < reqd ) { s [ i ] = " " ; count [ 1 ] ++ ; count [ 0 ] -- ; }
else if ( count [ 2 ] < reqd ) { s [ i ] = " " ; count [ 2 ] ++ ; count [ 0 ] -- ; } }
if ( s [ i ] === " " && count [ 1 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = " " ; count [ 0 ] ++ ; count [ 1 ] -- ; } else if ( count [ 2 ] < reqd && processed [ 1 ] >= reqd ) { s [ i ] = " " ; count [ 2 ] ++ ; count [ 1 ] -- ; } }
if ( s [ i ] === " " && count [ 2 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = " " ; count [ 0 ] ++ ; count [ 2 ] -- ; } else if ( count [ 1 ] < reqd ) { s [ i ] = " " ; count [ 1 ] ++ ; count [ 2 ] -- ; } }
processed [ s [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] ++ ; } return s . join ( " " ) ; }
var s = " " ; document . write ( formStringMinOperations ( s . split ( " " ) ) ) ;
function findMinimumAdjacentSwaps ( arr , N ) {
let visited = Array ( N + 1 ) . fill ( false ) ; let minimumSwaps = 0 ; for ( let i = 0 ; i < 2 * N ; i ++ ) {
if ( visited [ arr [ i ] ] == false ) { visited [ arr [ i ] ] = true ;
let count = 0 ; for ( let j = i + 1 ; j < 2 * N ; j ++ ) {
if ( visited [ arr [ j ] ] == false ) count ++ ;
else if ( arr [ i ] == arr [ j ] ) minimumSwaps += count ; } } } return minimumSwaps ; }
let arr = [ 1 , 2 , 3 , 3 , 1 , 2 ] ; let N = arr . length ; N = Math . floor ( N / 2 ) ; document . write ( findMinimumAdjacentSwaps ( arr , N ) ) ;
function possibility ( m , length , s ) {
var countodd = 0 ; for ( var i = 0 ; i < length ; i ++ ) {
if ( m . get ( s . charCodeAt ( i ) - 48 ) & 1 ) countodd ++ ;
if ( countodd > 1 ) return false ; } return true ; }
function largestPalindrome ( s ) {
var l = s . length ;
var m = new Map ( ) ; for ( var i = 0 ; i < l ; i ++ ) { if ( m . has ( s . charCodeAt ( i ) - 48 ) ) m . set ( s . charCodeAt ( i ) - 48 , m . get ( s . charCodeAt ( i ) - 48 ) + 1 ) ; else m . set ( s . charCodeAt ( i ) - 48 , 1 ) ; }
if ( possibility ( m , l , s ) == false ) { document . write ( " " ) ; return ; }
var largest = new Array ( l ) ;
var front = 0 ;
for ( var i = 9 ; i >= 0 ; i -- ) {
if ( m . has ( i ) & 1 ) {
largest [ Math . floor ( l / 2 ) ] = String . fromCharCode ( i + 48 ) ;
m . set ( i , m . get ( i ) - 1 ) ;
while ( m . get ( i ) > 0 ) { largest [ front ] = String . fromCharCode ( i + 48 ) ; largest [ l - front - 1 ] = String . fromCharCode ( i + 48 ) ; m . set ( i , m . get ( i ) - 2 ) ; front ++ ; } } else {
while ( m . get ( i ) > 0 ) {
largest [ front ] = String . fromCharCode ( i + 48 ) ; largest [ l - front - 1 ] = String . fromCharCode ( i + 48 ) ;
m . set ( i , m . get ( i ) - 2 ) ;
front ++ ; } } }
for ( var i = 0 ; i < l ; i ++ ) document . write ( largest [ i ] ) ; }
var s = " " ; largestPalindrome ( s ) ;
function swapCount ( s ) {
let pos = [ ] ; for ( let i = 0 ; i < s . length ; ++ i ) if ( s [ i ] == ' ' ) pos . push ( i ) ;
let count = 0 ;
let p = 0 ;
let sum = 0 ; let S = s . split ( ' ' ) ; for ( let i = 0 ; i < s . length ; ++ i ) {
if ( S [ i ] == ' ' ) { ++ count ; ++ p ; } else if ( S [ i ] == ' ' ) -- count ;
if ( count < 0 ) {
sum += pos [ p ] - i ; let temp = S [ i ] ; S [ i ] = S [ pos [ p ] ] ; S [ pos [ p ] ] = temp ; ++ p ;
count = 1 ; } } return sum ; }
let s = " " ; document . write ( swapCount ( s ) + " " ) ; s = " " ; document . write ( swapCount ( s ) ) ;
function minimumCostOfBreaking ( X , Y , m , n ) { let res = 0 ;
X . sort ( ) ; X . reverse ( ) ;
Y . sort ( ) ; Y . reverse ( ) ;
let hzntl = 1 , vert = 1 ;
let i = 0 , j = 0 ; while ( i < m && j < n ) { if ( X [ i ] > Y [ j ] ) { res += X [ i ] * vert ;
hzntl ++ ; i ++ ; } else { res += Y [ j ] * hzntl ;
vert ++ ; j ++ ; } }
let total = 0 ; while ( i < m ) total += X [ i ++ ] ; res += total * vert ;
total = 0 ; while ( j < n ) total += Y [ j ++ ] ; res += total * hzntl ; return res ; }
let m = 6 , n = 4 ; let X = [ 2 , 1 , 3 , 1 , 4 ] ; let Y = [ 4 , 1 , 2 ] ; document . write ( minimumCostOfBreaking ( X , Y , m - 1 , n - 1 ) ) ;
function getMin ( x , y , z ) { return Math . min ( Math . min ( x , y ) , z ) ; }
function editDistance ( str1 , str2 , m , n ) {
let dp = new Array ( m + 1 ) . fill ( new Array ( n + 1 ) ) ;
for ( let i = 0 ; i <= m ; i ++ ) { for ( let j = 0 ; j <= n ; j ++ ) {
if ( i == 0 )
dp [ i ] [ j ] = j ;
else if ( j == 0 )
dp [ i ] [ j ] = i ;
else if ( str1 [ i - 1 ] == str2 [ j - 1 ] ) dp [ i ] [ j ] = dp [ i - 1 ] [ j - 1 ] ;
else {
dp [ i ] [ j ] = 1 + getMin ( dp [ i ] [ j - 1 ] , dp [ i - 1 ] [ j ] , dp [ i - 1 ] [ j - 1 ] ) ; } } }
return dp [ m ] [ n ] ; }
function minimumSteps ( S , N ) {
let ans = Number . MAX_VALUE ;
for ( let i = 1 ; i < N ; i ++ ) { let S1 = S . substring ( 0 , i ) ; let S2 = S . substring ( i ) ;
let count = editDistance ( S1 , S2 , S1 . length , S2 . length ) ;
ans = Math . min ( ans , count ) ; }
document . write ( ans - 1 ) ; }
let S = " " ; let N = S . length ; minimumSteps ( S , N ) ;
function minimumOperations ( N ) {
let dp = new Array ( N + 1 ) ; let i ;
for ( i = 0 ; i <= N ; i ++ ) { dp [ i ] = 1e9 ; }
dp [ 2 ] = 0 ;
for ( i = 2 ; i <= N ; i ++ ) {
if ( dp [ i ] == 1e9 ) continue ;
if ( i * 5 <= N ) { dp [ i * 5 ] = Math . min ( dp [ i * 5 ] , dp [ i ] + 1 ) ; }
if ( i + 3 <= N ) { dp [ i + 3 ] = Math . min ( dp [ i + 3 ] , dp [ i ] + 1 ) ; } }
if ( dp [ N ] == 1e9 ) return - 1 ;
return dp [ N ] ; }
let N = 25 ; document . write ( minimumOperations ( N ) ) ;
function MaxProfit ( arr , n , transactionFee ) { let buy = - arr [ 0 ] ; let sell = 0 ;
for ( let i = 1 ; i < n ; i ++ ) { let temp = buy ;
buy = Math . max ( buy , sell - arr [ i ] ) ; sell = Math . max ( sell , temp + arr [ i ] - transactionFee ) ; }
return Math . max ( sell , buy ) ; }
let arr = [ 6 , 1 , 7 , 2 , 8 , 4 ] ; let n = arr . length ; let transactionFee = 2 ;
document . write ( MaxProfit ( arr , n , transactionFee ) ) ;
var start = Array . from ( Array ( 3 ) , ( ) => Array ( 3 ) ) ;
var ending = Array . from ( Array ( 3 ) , ( ) => Array ( 3 ) ) ;
function calculateStart ( n , m ) {
for ( var i = 1 ; i < m ; ++ i ) { start [ 0 ] [ i ] += start [ 0 ] [ i - 1 ] ; }
for ( var i = 1 ; i < n ; ++ i ) { start [ i ] [ 0 ] += start [ i - 1 ] [ 0 ] ; }
for ( var i = 1 ; i < n ; ++ i ) { for ( var j = 1 ; j < m ; ++ j ) {
start [ i ] [ j ] += Math . max ( start [ i - 1 ] [ j ] , start [ i ] [ j - 1 ] ) ; } } }
function calculateEnd ( n , m ) {
for ( var i = n - 2 ; i >= 0 ; -- i ) { ending [ i ] [ m - 1 ] += ending [ i + 1 ] [ m - 1 ] ; }
for ( var i = m - 2 ; i >= 0 ; -- i ) { ending [ n - 1 ] [ i ] += ending [ n - 1 ] [ i + 1 ] ; }
for ( var i = n - 2 ; i >= 0 ; -- i ) { for ( var j = m - 2 ; j >= 0 ; -- j ) {
ending [ i ] [ j ] += Math . max ( ending [ i + 1 ] [ j ] , ending [ i ] [ j + 1 ] ) ; } } }
function maximumPathSum ( mat , n , m , q , coordinates ) {
for ( var i = 0 ; i < n ; ++ i ) { for ( var j = 0 ; j < m ; ++ j ) { start [ i ] [ j ] = mat [ i ] [ j ] ; ending [ i ] [ j ] = mat [ i ] [ j ] ; } }
calculateStart ( n , m ) ;
calculateEnd ( n , m ) ;
var ans = 0 ;
for ( var i = 0 ; i < q ; ++ i ) { var X = coordinates [ i ] [ 0 ] - 1 ; var Y = coordinates [ i ] [ 1 ] - 1 ;
ans = Math . max ( ans , start [ X ] [ Y ] + ending [ X ] [ Y ] - mat [ X ] [ Y ] ) ; }
document . write ( ans ) ; }
var mat = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] , [ 7 , 8 , 9 ] ] ; var N = 3 ; var M = 3 ; var Q = 2 ; var coordinates = [ [ 1 , 2 ] , [ 2 , 2 ] ] ; maximumPathSum ( mat , N , M , Q , coordinates ) ;
function MaxSubsetlength ( arr , A , B ) {
var dp = Array . from ( Array ( A + 1 ) , ( ) => Array ( B + 1 ) . fill ( 0 ) ) ;
arr . forEach ( str => {
var zeros = [ ... str ] . filter ( x => x == ' ' ) . length ; var ones = [ ... str ] . filter ( x => x == ' ' ) . length ;
for ( var i = A ; i >= zeros ; i -- )
for ( var j = B ; j >= ones ; j -- )
dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - zeros ] [ j - ones ] + 1 ) ; } ) ;
return dp [ A ] [ B ] ; }
var arr = [ " " , " " , " " , " " , " " ] ; var A = 5 , B = 3 ; document . write ( MaxSubsetlength ( arr , A , B ) ) ;
function numOfWays ( a , n , i , blue ) {
if ( i == n ) return 1 ;
let count = 0 ;
for ( let j = 0 ; j < n ; j ++ ) {
if ( a [ i ] [ j ] == 1 && ! blue . has ( j ) ) { blue . add ( j ) ; count += numOfWays ( a , n , i + 1 , blue ) ; blue . delete ( j ) ; } } return count ; }
let n = 3 ; let mat = [ [ 0 , 1 , 1 ] , [ 1 , 0 , 1 ] , [ 1 , 1 , 1 ] ] ; let mpp = new Set ( ) ; document . write ( numOfWays ( mat , n , 0 , mpp ) ) ;
function minCost ( arr , n ) {
if ( n < 3 ) { document . write ( arr [ 0 ] ) ; return ; }
let dp = [ ] ;
dp [ 0 ] = arr [ 0 ] ; dp [ 1 ] = dp [ 0 ] + arr [ 1 ] + arr [ 2 ] ;
for ( let i = 2 ; i < n - 1 ; i ++ ) dp [ i ] = Math . min ( dp [ i - 2 ] + arr [ i ] , dp [ i - 1 ] + arr [ i ] + arr [ i + 1 ] ) ;
dp [ n - 1 ] = Math . min ( dp [ n - 2 ] , dp [ n - 3 ] + arr [ n - 1 ] ) ;
document . write ( dp [ n - 1 ] ) ; }
let arr = [ 9 , 4 , 6 , 8 , 5 ] ; let N = arr . length ; minCost ( arr , N ) ;
M = 1000000007 ;
function power ( X , Y ) {
var res = 1 ;
X = X % M ;
if ( X == 0 ) return 0 ;
while ( Y > 0 ) {
if ( ( Y & 1 ) != 0 ) {
res = ( res * X ) % M ; }
Y = Y >> 1 ;
X = ( X * X ) % M ; } return res ; }
function findValue ( n ) {
var X = 0 ;
var pow_10 = 1 ;
while ( n != 0 ) {
if ( ( n & 1 ) != 0 ) {
X += pow_10 ; }
pow_10 *= 10 ;
n /= 2 ; }
X = ( X * 2 ) % M ;
var res = power ( 2 , X ) ; return res ; }
var n = 2 ; document . write ( findValue ( n ) ) ;
function findWays ( N ) {
if ( N == 0 ) { return 1 ; }
var cnt = 0 ;
for ( var i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i ) ; } }
return cnt ; }
var N = 4 ;
document . write ( findWays ( N ) ) ;
function checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 , j ) {
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; } else {
let l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
let m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
let r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
return Math . max ( Math . max ( l , m ) , r ) ; } }
function checkEqualSum ( arr , N ) {
let sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
let arr = [ 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 ] ; let N = arr . length ;
checkEqualSum ( arr , N ) ;
var dp = new Map ( ) ;
function checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 , j ) { var s = ( sm1 . toString ( ) ) + " " + ( sm2 . toString ( ) ) + ( j . toString ( ) ) ;
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; }
if ( dp . has ( s ) ) return dp [ s ] ; else {
var l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
var m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
var r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
return dp [ s ] = Math . max ( Math . max ( l , m ) , r ) ; } }
function checkEqualSum ( arr , N ) {
var sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
var arr = [ 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 ] ; var N = arr . length ;
checkEqualSum ( arr , N ) ;
function precompute ( nextpos , arr , N ) {
nextpos [ N - 1 ] = N ; for ( var i = N - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] == arr [ i + 1 ] ) nextpos [ i ] = nextpos [ i + 1 ] ; else nextpos [ i ] = i + 1 ; } }
function findIndex ( query , arr , N , Q ) {
var nextpos = Array ( N ) ; precompute ( nextpos , arr , N ) ; for ( var i = 0 ; i < Q ; i ++ ) { var l , r , x ; l = query [ i ] [ 0 ] ; r = query [ i ] [ 1 ] ; x = query [ i ] [ 2 ] ; var ans = - 1 ;
if ( arr [ l ] != x ) ans = l ;
else {
var d = nextpos [ l ] ;
if ( d <= r ) ans = d ; } document . write ( ans + " " ) ; } }
var N , Q ; N = 6 ; Q = 3 ; var arr = [ 1 , 2 , 1 , 1 , 3 , 5 ] ; var query = [ [ 0 , 3 , 1 ] , [ 1 , 5 , 2 ] , [ 2 , 3 , 1 ] ] ; findIndex ( query , arr , N , Q ) ;
let mod = 10000000007 ;
function countWays ( s , t , k ) {
let n = s . length ;
let a = 0 , b = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { let p = s . substr ( i , n - i ) + s . substr ( 0 , i ) ;
if ( p == t ) a ++ ; else b ++ ; }
let dp1 = Array . from ( { length : k + 1 } , ( _ , i ) => 0 ) ; let dp2 = Array . from ( { length : k + 1 } , ( _ , i ) => 0 ) ; if ( s == t ) { dp1 [ 0 ] = 1 ; dp2 [ 0 ] = 0 ; } else { dp1 [ 0 ] = 0 ; dp2 [ 0 ] = 1 ; }
for ( let i = 1 ; i <= k ; i ++ ) { dp1 [ i ] = ( ( dp1 [ i - 1 ] * ( a - 1 ) ) % mod + ( dp2 [ i - 1 ] * a ) % mod ) % mod ; dp2 [ i ] = ( ( dp1 [ i - 1 ] * ( b ) ) % mod + ( dp2 [ i - 1 ] * ( b - 1 ) ) % mod ) % mod ; }
return dp1 [ k ] ; }
let S = " " , T = " " ;
let K = 2 ;
document . write ( countWays ( S , T , K ) ) ;
function minOperation ( k ) {
let dp = Array . from ( { length : k + 1 } , ( _ , i ) => 0 ) ; for ( let i = 1 ; i <= k ; i ++ ) { dp [ i ] = dp [ i - 1 ] + 1 ;
if ( i % 2 == 0 ) { dp [ i ] = Math . min ( dp [ i ] , dp [ i / 2 ] + 1 ) ; } } return dp [ k ] ; }
let K = 12 ; document . write ( minOperation ( K ) ) ;
function maxSum ( p0 , p1 , a , pos , n ) { if ( pos == n ) { if ( p0 == p1 ) return p0 ; else return 0 ; }
var ans = maxSum ( p0 , p1 , a , pos + 1 , n ) ;
ans = Math . max ( ans , maxSum ( p0 + a [ pos ] , p1 , a , pos + 1 , n ) ) ;
ans = Math . max ( ans , maxSum ( p0 , p1 + a [ pos ] , a , pos + 1 , n ) ) ; return ans ; }
var n = 4 ; var a = [ 1 , 2 , 3 , 6 ] ; document . write ( maxSum ( 0 , 0 , a , 0 , n ) ) ;
function maxSum ( a , n ) {
var sum = 0 ; for ( var i = 0 ; i < n ; i ++ ) sum += a [ i ] ; var limit = 2 * sum + 1 ;
var dp = Array . from ( Array ( n + 1 ) , ( ) => Array ( limit ) ) ;
for ( var i = 0 ; i < n + 1 ; i ++ ) { for ( var j = 0 ; j < limit ; j ++ ) dp [ i ] [ j ] = - 1000000000 ; }
dp [ 0 ] [ sum ] = 0 ; for ( var i = 1 ; i <= n ; i ++ ) { for ( var j = 0 ; j < limit ; j ++ ) {
if ( ( j - a [ i - 1 ] ) >= 0 && dp [ i - 1 ] [ j - a [ i - 1 ] ] != - 1000000000 ) dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j - a [ i - 1 ] ] + a [ i - 1 ] ) ;
if ( ( j + a [ i - 1 ] ) < limit && dp [ i - 1 ] [ j + a [ i - 1 ] ] != - 1000000000 ) dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j + a [ i - 1 ] ] ) ;
if ( dp [ i - 1 ] [ j ] != - 1000000000 ) dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j ] ) ; } } return dp [ n ] [ sum ] ; }
var n = 4 ; var a = [ 1 , 2 , 3 , 6 ] ; document . write ( maxSum ( a , n ) ) ;
fib = Array ( 100005 ) . fill ( 0 ) ;
function computeFibonacci ( ) { fib [ 0 ] = 1 ; fib [ 1 ] = 1 ; for ( i = 2 ; i < 100005 ; i ++ ) { fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ; } }
function countString ( str ) {
var ans = 1 ; var cnt = 1 ; for ( i = 1 ; i < str . length ; i ++ ) {
if ( str . charAt ( i ) == str . charAt ( i - 1 ) ) { cnt ++ ; }
else { ans = ans * fib [ cnt ] ; cnt = 1 ; } }
ans = ans * fib [ cnt ] ;
return ans ; }
var str = " " ;
computeFibonacci ( ) ;
document . write ( countString ( str ) ) ;
var MAX = 100001 ;
function printGolombSequence ( N ) {
var arr = Array ( MAX ) ;
var cnt = 0 ;
arr [ 0 ] = 0 ; arr [ 1 ] = 1 ;
var M = new Map ( ) ;
M . set ( 2 , 2 ) ;
for ( var i = 2 ; i <= N ; i ++ ) {
if ( cnt == 0 ) { arr [ i ] = 1 + arr [ i - 1 ] ; cnt = M . get ( arr [ i ] ) ; cnt -- ; }
else { arr [ i ] = arr [ i - 1 ] ; cnt -- ; }
M . set ( i , arr [ i ] ) ; }
for ( var i = 1 ; i <= N ; i ++ ) { document . write ( arr [ i ] + ' ' ) ; } }
var N = 11 ; printGolombSequence ( N ) ;
function number_of_ways ( n ) {
let includes_3 = new Uint8Array ( n + 1 ) ;
let not_includes_3 = new Uint8Array ( n + 1 ) ;
includes_3 [ 3 ] = 1 ; not_includes_3 [ 1 ] = 1 ; not_includes_3 [ 2 ] = 2 ; not_includes_3 [ 3 ] = 3 ;
for ( let i = 4 ; i <= n ; i ++ ) { includes_3 [ i ] = includes_3 [ i - 1 ] + includes_3 [ i - 2 ] + not_includes_3 [ i - 3 ] ; not_includes_3 [ i ] = not_includes_3 [ i - 1 ] + not_includes_3 [ i - 2 ] ; } return includes_3 [ n ] ; }
let n = 7 ; document . write ( number_of_ways ( n ) ) ;
const MAX = 100000 ;
var divisors = new Array ( MAX ) . fill ( 0 ) ;
function generateDivisors ( n ) { for ( var i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) { if ( n / i == i ) { divisors [ i ] ++ ; } else { divisors [ i ] ++ ; divisors [ n / i ] ++ ; } } } }
function findMaxMultiples ( arr , n ) {
var ans = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
ans = Math . max ( divisors [ arr [ i ] ] , ans ) ;
generateDivisors ( arr [ i ] ) ; } return ans ; }
var arr = [ 8 , 1 , 28 , 4 , 2 , 6 , 7 ] ; var n = arr . length ; document . write ( findMaxMultiples ( arr , n ) ) ;
var n = 3 ; var maxV = 20 ;
var dp = new Array ( n ) ; for ( var i = 0 ; i < n ; i ++ ) { dp [ i ] = new Array ( n ) ; for ( var j = 0 ; j < n ; j ++ ) { dp [ i ] [ j ] = new Array ( maxV ) ; } } var v = new Array ( n ) ;
for ( var i = 0 ; i < n ; i ++ ) { v [ i ] = new Array ( n ) ; for ( var j = 0 ; j < n ; j ++ ) { v [ i ] [ j ] = new Array ( maxV ) ; } }
function countWays ( i , j , x , arr ) {
if ( i == n j == n ) return 0 ; x = ( x & arr [ i ] [ j ] ) ; if ( x == 0 ) return 0 ; if ( i == n - 1 && j == n - 1 ) return 1 ;
if ( v [ i ] [ j ] [ x ] ) return dp [ i ] [ j ] [ x ] ; v [ i ] [ j ] [ x ] = 1 ;
dp [ i ] [ j ] [ x ] = countWays ( i + 1 , j , x , arr ) + countWays ( i , j + 1 , x , arr ) ; return dp [ i ] [ j ] [ x ] ; }
var arr = [ [ 1 , 2 , 1 ] , [ 1 , 1 , 0 ] , [ 2 , 1 , 1 ] ] ; document . write ( countWays ( 0 , 0 , arr [ 0 ] [ 0 ] , arr ) ) ;
function FindMaximumSum ( ind , kon , a , b , c , n , dp ) {
if ( ind == n ) return 0 ;
if ( dp [ ind ] [ kon ] != - 1 ) return dp [ ind ] [ kon ] ; var ans = - 1000000005 ;
if ( kon == 0 ) { ans = Math . max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = Math . max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon == 1 ) { ans = Math . max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; ans = Math . max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon == 2 ) { ans = Math . max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = Math . max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; } return dp [ ind ] [ kon ] = ans ; }
var a = [ 6 , 8 , 2 , 7 , 4 , 2 , 7 ] ; var b = [ 7 , 8 , 5 , 8 , 6 , 3 , 5 ] ; var c = [ 8 , 3 , 2 , 6 , 8 , 4 , 1 ] ; var n = a . length ; var dp = Array . from ( Array ( n ) , ( ) => Array ( n ) . fill ( - 1 ) ) ;
var x = FindMaximumSum ( 0 , 0 , a , b , c , n , dp ) ;
var y = FindMaximumSum ( 0 , 1 , a , b , c , n , dp ) ;
var z = FindMaximumSum ( 0 , 2 , a , b , c , n , dp ) ;
document . write ( Math . max ( x , Math . max ( y , z ) ) ) ;
let mod = 1000000007 ;
function noOfBinaryStrings ( N , k ) { let dp = new Array ( 100002 ) ; for ( let i = 1 ; i <= k - 1 ; i ++ ) { dp [ i ] = 1 ; } dp [ k ] = 2 ; for ( let i = k + 1 ; i <= N ; i ++ ) { dp [ i ] = ( dp [ i - 1 ] + dp [ i - k ] ) % mod ; } return dp [ N ] ; }
let N = 4 ; let K = 2 ; document . write ( noOfBinaryStrings ( N , K ) ) ;
function findWaysToPair ( p ) {
var dp = Array ( p + 1 ) ; dp [ 1 ] = 1 ; dp [ 2 ] = 2 ;
for ( var i = 3 ; i <= p ; i ++ ) { dp [ i ] = dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ; } return dp [ p ] ; }
var p = 3 ; document . write ( findWaysToPair ( p ) ) ;
function CountWays ( n , flag ) {
if ( n == 0 ) { return 1 ; } if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 1 + 1 ; }
return CountWays ( n - 1 ) + CountWays ( n - 3 ) ; }
let n = 5 ; document . write ( CountWays ( n , false ) ) ;
function factors ( n ) {
var v = [ ] ; v . push ( 1 ) ;
for ( var i = 2 ; i <= Math . sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) { v . push ( i ) ;
if ( n / i != i ) { v . push ( n / i ) ; } } }
return v ; }
function checkAbundant ( n ) { var v = [ ] ; var sum = 0 ;
v = factors ( n ) ;
for ( var i = 0 ; i < v . length ; i ++ ) { sum += v [ i ] ; }
if ( sum > n ) return true ; else return false ; }
function checkSemiPerfect ( n ) { var v = [ ] ;
v = factors ( n ) ;
v . sort ( ) var r = v . length ;
var subset = Array . from ( Array ( r + 1 ) , ( ) => Array ( n + 1 ) ) ;
for ( var i = 0 ; i <= r ; i ++ ) subset [ i ] [ 0 ] = true ;
for ( var i = 1 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = false ;
for ( var i = 1 ; i <= r ; i ++ ) { for ( var j = 1 ; j <= n ; j ++ ) {
if ( j < v [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] ; else { subset [ i ] [ j ] = subset [ i - 1 ] [ j ] || subset [ i - 1 ] [ j - v [ i - 1 ] ] ; } } }
if ( ( subset [ r ] [ n ] ) == 0 ) return false ; else return true ; }
function checkweird ( n ) { if ( checkAbundant ( n ) == true && checkSemiPerfect ( n ) == false ) return true ; else return false ; }
var n = 70 ; if ( checkweird ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function maxSubArraySumRepeated ( a , n , k ) { let max_so_far = 0 ; let INT_MIN , max_ending_here = 0 ; for ( let i = 0 ; i < n * k ; i ++ ) {
max_ending_here = max_ending_here + a [ i % n ] ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; if ( max_ending_here < 0 ) max_ending_here = 0 ; } return max_so_far ; }
let a = [ 10 , 20 , - 30 , - 1 ] ; let n = a . length ; let k = 3 ; document . write ( " " + maxSubArraySumRepeated ( a , n , k ) ) ;
function longOddEvenIncSeq ( arr , n ) {
let lioes = [ ] ;
let maxLen = 0 ;
for ( let i = 0 ; i < n ; i ++ ) lioes [ i ] = 1 ;
for ( let i = 1 ; i < n ; i ++ ) for ( let j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && ( arr [ i ] + arr [ j ] ) % 2 != 0 && lioes [ i ] < lioes [ j ] + 1 ) lioes [ i ] = lioes [ j ] + 1 ;
for ( let i = 0 ; i < n ; i ++ ) if ( maxLen < lioes [ i ] ) maxLen = lioes [ i ] ;
return maxLen ; }
let arr = [ 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 ] ; let n = 10 ; document . write ( " " + " " + longOddEvenIncSeq ( arr , n ) ) ;
function isOperator ( op ) { return ( op == ' ' op == ' ' ) ; }
function printMinAndMaxValueOfExp ( exp ) { let num = [ ] ; let opr = [ ] ; let tmp = " " ;
for ( let i = 0 ; i < exp . length ; i ++ ) { if ( isOperator ( exp [ i ] ) ) { opr . push ( exp [ i ] ) ; num . push ( parseInt ( tmp ) ) ; tmp = " " ; } else { tmp += exp [ i ] ; } }
num . push ( parseInt ( tmp ) ) ; let len = num . length ; let minVal = new Array ( len ) ; let maxVal = new Array ( len ) ;
for ( let i = 0 ; i < len ; i ++ ) { minVal [ i ] = new Array ( len ) ; maxVal [ i ] = new Array ( len ) ; for ( let j = 0 ; j < len ; j ++ ) { minVal [ i ] [ j ] = Number . MAX_VALUE ; maxVal [ i ] [ j ] = 0 ;
if ( i == j ) minVal [ i ] [ j ] = maxVal [ i ] [ j ] = num [ i ] ; } }
for ( let L = 2 ; L <= len ; L ++ ) { for ( let i = 0 ; i < len - L + 1 ; i ++ ) { let j = i + L - 1 ; for ( let k = i ; k < j ; k ++ ) { let minTmp = 0 , maxTmp = 0 ;
if ( opr [ k ] == ' ' ) { minTmp = minVal [ i ] [ k ] + minVal [ k + 1 ] [ j ] ; maxTmp = maxVal [ i ] [ k ] + maxVal [ k + 1 ] [ j ] ; }
else if ( opr [ k ] == ' ' ) { minTmp = minVal [ i ] [ k ] * minVal [ k + 1 ] [ j ] ; maxTmp = maxVal [ i ] [ k ] * maxVal [ k + 1 ] [ j ] ; }
if ( minTmp < minVal [ i ] [ j ] ) minVal [ i ] [ j ] = minTmp ; if ( maxTmp > maxVal [ i ] [ j ] ) maxVal [ i ] [ j ] = maxTmp ; } } }
document . write ( " " + minVal [ 0 ] [ len - 1 ] + " " + maxVal [ 0 ] [ len - 1 ] ) ; }
let expression = " " ; printMinAndMaxValueOfExp ( expression ) ;
function MatrixChainOrder ( p , i , j ) { if ( i == j ) return 0 ; var min = Number . MAX_VALUE ;
var k = 0 ; for ( k = i ; k < j ; k ++ ) { var count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
var arr = [ 1 , 2 , 3 , 4 , 3 ] ; var n = arr . length ; document . write ( " " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ;
let dp = new Array ( 100 ) ; for ( var i = 0 ; i < dp . length ; i ++ ) { dp [ i ] = new Array ( 2 ) ; }
function matrixChainMemoised ( p , i , j ) { if ( i == j ) { return 0 ; } if ( dp [ i ] [ j ] != - 1 ) { return dp [ i ] [ j ] ; } dp [ i ] [ j ] = Number . MAX_VALUE ; for ( let k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i ] [ j ] ; } function MatrixChainOrder ( p , n ) { let i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
let arr = [ 1 , 2 , 3 , 4 ] ; let n = arr . length ; for ( var i = 0 ; i < dp . length ; i ++ ) { for ( var j = 0 ; j < dp . length ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } document . write ( " " + MatrixChainOrder ( arr , n ) ) ;
function flipBitsOfAandB ( A , B ) {
A = A ^ ( A & B ) ;
B = B ^ ( A & B ) ;
document . write ( A + " " + B ) ; }
var A = 10 , B = 20 ; flipBitsOfAandB ( A , B ) ;
function TotalHammingDistance ( n ) { let i = 1 , sum = 0 ; while ( Math . floor ( n / i ) > 0 ) { sum = sum + Math . floor ( n / i ) ; i = i * 2 ; } return sum ; }
let N = 9 ; document . write ( TotalHammingDistance ( N ) ) ;
let m = 1000000007 ;
function solve ( n ) {
let s = 0 ; for ( let l = 1 ; l <= n ; ) {
let r = ( n / Math . floor ( n / l ) ) ; let x = Math . floor ( ( ( r % m ) * ( ( r + 1 ) % m ) ) / 2 ) % m ; let y = Math . floor ( ( ( l % m ) * ( ( l - 1 ) % m ) ) / 2 ) % m ; let p = ( Math . floor ( n / l ) % m ) ;
s = ( s + ( ( ( x - y ) % m ) * p ) % m + m ) % m ; s %= m ; l = r + 1 ; }
document . write ( ( s + m ) % m ) ; }
let n = 12 ; solve ( n ) ;
function min_time_to_cut ( N ) { if ( N == 0 ) return 0 ;
return Math . ceil ( Math . log ( N ) / Math . log ( 2 ) ) ; }
let N = 100 ; document . write ( min_time_to_cut ( N ) ) ;
function findDistinctSums ( n ) {
s = new Set ( ) ; for ( var i = 1 ; i <= n ; i ++ ) { for ( var j = i ; j <= n ; j ++ ) {
s . add ( i + j ) ; } }
return s . size ; }
var N = 3 ; document . write ( findDistinctSums ( N ) ) ;
function printPattern ( i , j , n ) {
if ( j >= n ) { return 0 ; } if ( i >= n ) { return 1 ; }
if ( j == i j == n - 1 - i ) {
if ( i == n - 1 - j ) { document . write ( " " ) ; }
else { document . write ( " \\ " ) ; } }
else { document . write ( " " ) ; }
if ( printPattern ( i , j + 1 , n ) == 1 ) { return 1 ; } document . write ( " " ) ;
return printPattern ( i + 1 , 0 , n ) ; }
let N = 9 ;
printPattern ( 0 , 0 , N ) ;
function zArray ( arr ) { let n = arr . length ; let z = new Array ( n ) ; let r = 0 , l = 0 ;
for ( let k = 1 ; k < n ; k ++ ) {
if ( k > r ) { r = l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; }
else { let k1 = k - l ; if ( z [ k1 ] < r - k + 1 ) z [ k ] = z [ k1 ] ; else { l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; } } } return z ; }
function mergeArray ( A , B ) { let n = A . length ; let m = B . length ; let z = new Array ( ) ;
let c = new Array ( n + m + 1 ) ;
for ( let i = 0 ; i < m ; i ++ ) c [ i ] = B [ i ] ;
c [ m ] = Number . MAX_SAFE_INTEGER ;
for ( let i = 0 ; i < n ; i ++ ) c [ m + i + 1 ] = A [ i ] ;
z = zArray ( c ) ; return z ; }
function findZArray ( A , B , n ) { let flag = 0 ; let z = [ ] ; z = mergeArray ( A , B ) ;
for ( let i = 0 ; i < z . length ; i ++ ) { if ( z [ i ] == n ) { document . write ( ( i - n - 1 ) + " " ) ; flag = 1 ; } } if ( flag == 0 ) { document . write ( " " ) ; } }
let A = [ 1 , 2 , 3 , 2 , 3 , 2 ] ; let B = [ 2 , 3 ] ; let n = B . length ; findZArray ( A , B , n ) ;
function getCount ( a , b ) {
if ( b . length % a . length != 0 ) return - 1 ; var count = parseInt ( b . length / a . length ) ;
var str = " " ; for ( i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str == ( b ) ) return count ; return - 1 ; }
var a = " " ; var b = " " ; document . write ( getCount ( a , b ) ) ;
function check ( S1 , S2 ) {
var n1 = S1 . length ; var n2 = S2 . length ;
var mp = { } ;
for ( var i = 0 ; i < n1 ; i ++ ) { if ( mp . hasOwnProperty ( S1 [ i ] ) ) { mp [ S1 [ i ] ] = mp [ S1 [ i ] ] + 1 ; } else { mp [ S1 [ i ] ] = 1 ; } }
for ( var i = 0 ; i < n2 ; i ++ ) {
if ( mp . hasOwnProperty ( S2 [ i ] ) ) { mp [ S2 [ i ] ] = mp [ S2 [ i ] ] - 1 ; }
else if ( mp . hasOwnProperty ( String . fromCharCode ( S2 [ i ] . charCodeAt ( 0 ) - 1 ) ) && mp . hasOwnProperty ( String . fromCharCode ( S2 [ i ] . charCodeAt ( 0 ) - 2 ) ) ) { mp [ String . fromCharCode ( S2 [ i ] . charCodeAt ( 0 ) - 1 ) ] = mp [ String . fromCharCode ( S2 [ i ] . charCodeAt ( 0 ) - 1 ) ] - 1 ; mp [ String . fromCharCode ( S2 [ i ] . charCodeAt ( 0 ) - 2 ) ] = mp [ String . fromCharCode ( S2 [ i ] . charCodeAt ( 0 ) - 2 ) ] - 1 ; } else { return false ; } } return true ; }
var S1 = " " ; var S2 = " " ;
if ( check ( S1 , S2 ) ) document . write ( " " ) ; else document . write ( " " ) ;
function countPattern ( str ) { let len = str . length ; let oneSeen = false ;
for ( let i = 0 ; i < len ; i ++ ) { let getChar = str [ i ] ;
if ( getChar == ' ' && oneSeen == true ) { if ( str [ i - 1 ] == ' ' ) count ++ ; }
if ( getChar == ' ' && oneSeen == false ) oneSeen = true ;
if ( getChar != ' ' && str [ i ] != ' ' ) oneSeen = false ; } return count ; }
let str = " " ; document . write ( countPattern ( str ) ) ;
function checkIfPossible ( N , arr , T ) {
let freqS = new Array ( 256 ) . fill ( 0 ) ;
let freqT = new Array ( 256 ) . fill ( 0 ) ;
for ( let ch of T ) { freqT [ ch . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; }
for ( let i = 0 ; i < N ; i ++ ) {
for ( let ch of arr [ i ] ) { freqS [ ch . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; } } for ( let i = 0 ; i < 256 ; i ++ ) {
if ( freqT [ i ] == 0 && freqS [ i ] != 0 ) { return " " ; }
else if ( freqS [ i ] == 0 && freqT [ i ] != 0 ) { return " " ; }
else if ( freqT [ i ] != 0 && freqS [ i ] != ( freqT [ i ] * N ) ) { return " " ; } }
return " " ; }
let arr = [ " " , " " , " " ] ; let T = " " ; let N = arr . length ; document . write ( checkIfPossible ( N , arr , T ) ) ;
function groupsOfOnes ( S , N ) {
let count = 0 ;
var st = [ ] ;
for ( let i = 0 ; i < N ; i ++ ) {
if ( S [ i ] == ' ' ) st . push ( 1 ) ;
else {
if ( st . length != 0 ) { count ++ ; while ( st . length != 0 ) { st . pop ( ) ; } } } }
if ( st . length != 0 ) count ++ ;
return count ; }
var S = " " ; let N = S . length ;
document . write ( groupsOfOnes ( S , N ) ) ;
function generatePalindrome ( S ) {
let Hash = new Map ( ) ;
for ( let ch = 0 ; ch < S . length ; ch ++ ) { if ( ! Hash . has ( S [ ch ] ) ) Hash . set ( S [ ch ] , 1 ) ; else { Hash . set ( S [ ch ] , Hash . get ( S [ ch ] ) + 1 ) } }
let st = new Set ( ) ;
for ( let i = ' ' . charCodeAt ( 0 ) ; i <= ' ' . charCodeAt ( 0 ) ; i ++ ) {
if ( Hash . get ( String . fromCharCode ( i ) ) == 2 ) {
for ( let j = ' ' . charCodeAt ( 0 ) ; j <= ' ' . charCodeAt ( 0 ) ; j ++ ) {
let s = " " ; if ( Hash . get ( String . fromCharCode ( j ) ) && i != j ) { s += String . fromCharCode ( i ) ; s += String . fromCharCode ( j ) ; s += String . fromCharCode ( i ) ;
st . add ( s ) ; } } }
if ( Hash . get ( String . fromCharCode ( i ) ) >= 3 ) {
for ( let j = ' ' . charCodeAt ( 0 ) ; j <= ' ' . charCodeAt ( 0 ) ; j ++ ) {
let s = " " ;
if ( Hash . get ( String . fromCharCode ( j ) ) ) { s += String . fromCharCode ( i ) ; s += String . fromCharCode ( j ) ; s += String . fromCharCode ( i ) ;
st . add ( s ) ; } } } }
for ( let item of st . values ( ) ) { document . write ( item + " " ) } }
let S = " " ; generatePalindrome ( S ) ;
function countOccurrences ( S , X , Y ) {
let count = 0 ;
let N = S . length , A = X . length ; let B = Y . length ;
for ( let i = 0 ; i < N ; i ++ ) {
if ( S . substr ( i , B ) == Y ) count ++ ;
if ( S . substr ( i , A ) == X ) document . write ( count , " " ) ; } }
let S = " " , X = " " , Y = " " ; countOccurrences ( S , X , Y ) ;
function DFA ( str , N ) {
if ( N <= 1 ) { document . write ( " " ) ; return ; }
let count = 0 ;
if ( str [ 0 ] == ' ' ) { count ++ ;
for ( let i = 1 ; i < N ; i ++ ) {
if ( str [ i ] == ' ' str [ i ] == ' ' ) count ++ ; else break ; } } else {
document . write ( " " ) ; return ; }
if ( count == N ) document . write ( " " ) ; else document . write ( " " ) ; }
let str = " " ; let N = str . length ; DFA ( str , N ) ;
function minMaxDigits ( str , N ) {
let arr = [ ] ; for ( let i = 0 ; i < N ; i ++ ) arr [ i ] = ( str [ i ] - ' ' ) % 3 ;
let zero = 0 , one = 0 , two = 0 ;
for ( let i = 0 ; i < N ; i ++ ) { if ( arr [ i ] == 0 ) zero ++ ; if ( arr [ i ] == 1 ) one ++ ; if ( arr [ i ] == 2 ) two ++ ; }
let sum = 0 ; for ( let i = 0 ; i < N ; i ++ ) { sum = ( sum + arr [ i ] ) % 3 ; }
if ( sum == 0 ) { document . write ( 0 + " " ) ; } if ( sum == 1 ) { if ( ( one != 0 ) && ( N > 1 ) ) document . write ( 1 + " " ) ; else if ( two > 1 && N > 2 ) document . write ( 2 + " " ) ; else document . write ( - 1 + " " ) ; } if ( sum == 2 ) { if ( two != 0 && N > 1 ) document . write ( 1 + " " ) ; else if ( one > 1 && N > 2 ) document . write ( 2 + " " ) ; else document . write ( - 1 + " " ) ; }
if ( zero > 0 ) document . write ( N - 1 + " " ) ; else if ( one > 0 && two > 0 ) document . write ( N - 2 + " " ) ; else if ( one > 2 two > 2 ) document . write ( N - 3 + " " ) ; else document . write ( - 1 + " " ) ; }
let str = " " ; let N = str . length ;
minMaxDigits ( str , N ) ;
function findMinimumChanges ( N , K , S ) {
var ans = 0 ;
for ( var i = 0 ; i < parseInt ( ( K + 1 ) / 2 ) ; i ++ ) {
var mp = new Map ( ) ;
for ( var j = i ; j < N ; j += K ) {
if ( mp . has ( S [ j ] ) ) { mp . set ( S [ j ] , mp . get ( S [ j ] ) + 1 ) ; } else { mp . set ( S [ j ] , 1 ) ; } }
for ( var j = N - i - 1 ; j >= 0 ; j -= K ) {
if ( ( K & 1 ) && i == parseInt ( K / 2 ) ) break ;
if ( mp . has ( S [ j ] ) ) { mp . set ( S [ j ] , mp . get ( S [ j ] ) + 1 ) ; } else { mp . set ( S [ j ] , 1 ) ; } }
var curr_max = - 1000000000 ; mp . forEach ( ( value , key ) => { curr_max = Math . max ( curr_max , value ) ; } ) ;
if ( K & 1 && i == parseInt ( K / 2 ) ) ans += ( parseInt ( N / K ) - curr_max ) ;
else ans += ( parseInt ( N / K ) * 2 - curr_max ) ; }
return ans ; }
var S = " " ; var N = S . length ; var K = 3 ;
document . write ( findMinimumChanges ( N , K , S ) ) ;
function checkString ( s , K ) { var n = s . length ;
var mp = new Map ( ) ; for ( var i = 0 ; i < n ; i ++ ) { if ( mp . has ( s [ i ] ) ) { mp . set ( s [ i ] , mp . get ( s [ i ] ) + 1 ) ; } else mp . set ( s [ i ] , 1 ) ; } var cnt = 0 , f = 0 ;
var st = new Set ( ) ; for ( var i = 0 ; i < n ; i ++ ) {
st . add ( s [ i ] ) ;
if ( st . size > K ) { f = 1 ; break ; }
if ( mp . get ( s [ i ] ) == i ) st . delete ( s [ i ] ) ; } return ( f == 1 ? " " : " " ) ; }
var s = " " ; var k = 2 ; document . write ( checkString ( s , k ) ) ;
function distinct ( S , M , n ) { let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
let set1 = new Set ( ) ; for ( let j = 0 ; j < S [ i ] . length ; j ++ ) { if ( ! set1 . has ( S [ i ] [ j ] ) ) set1 . add ( S [ i ] [ j ] ) ; } let c = set1 . size ;
if ( c <= M ) count += 1 ; } document . write ( count ) ; }
let S = [ " " , " " , " " ] ; let M = 7 ; let n = S . length ; distinct ( S , M , n ) ;
function removeOddFrequencyCharacters ( s ) {
let m = new Map ( ) ; for ( let i = 0 ; i < s . length ; i ++ ) { let p = s [ i ] ; let count = m . get ( p ) ; if ( count == null ) { count = 0 ; m . set ( p , 1 ) ; } else m . set ( p , count + 1 ) ; }
let new_string = " " ;
for ( let i = 0 ; i < s . length ; i ++ ) {
if ( ( m . get ( s [ i ] ) & 1 ) == 1 ) continue ;
new_string += s [ i ] ; }
return new_string ; }
let str = " " ;
str = removeOddFrequencyCharacters ( str ) ; document . write ( str ) ;
function productAtKthLevel ( tree , k , level ) { if ( tree [ i ++ ] == ' ' ) {
if ( tree [ i ] == ' ' ) return 1 ; var product = 1 ;
if ( level == k ) product = tree [ i ] - ' ' ;
++ i ; var leftproduct = productAtKthLevel ( tree , k , level + 1 ) ;
++ i ; var rightproduct = productAtKthLevel ( tree , k , level + 1 ) ;
++ i ; return product * leftproduct * rightproduct ; } return int . MinValue ; }
var tree = " " ; var k = 2 ; i = 0 ; document . write ( productAtKthLevel ( tree , k , 0 ) ) ;
function findMostOccurringChar ( str ) {
var hash = Array ( 26 ) . fill ( 0 ) ;
for ( var i = 0 ; i < str . length ; i ++ ) {
for ( var j = 0 ; j < str [ i ] . length ; j ++ ) {
hash [ str [ i ] [ j ] ] ++ ; } }
var max = 0 ; for ( var i = 0 ; i < 26 ; i ++ ) { max = hash [ i ] > hash [ max ] ? i : max ; } document . write ( String . fromCharCode ( max + 97 ) ) ; }
var str = [ ] ; str . push ( " " ) ; str . push ( " " ) ; str . push ( " " ) ; str . push ( " " ) ; findMostOccurringChar ( str ) ;
function isPalindrome ( num ) {
var s = num . toString ( ) ;
var low = 0 ; var high = s . length - 1 ; while ( low < high ) {
if ( s [ low ] != s [ high ] ) return false ;
low ++ ; high -- ; } return true ; }
var n = 123.321 ; if ( isPalindrome ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
const MAX = 26 ;
function maxSubStr ( str1 , len1 , str2 , len2 ) {
if ( len1 > len2 ) return 0 ;
let freq1 = new Array ( MAX ) . fill ( 0 ) ; for ( let i = 0 ; i < len1 ; i ++ ) freq1 [ str1 . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ] ++ ;
let freq2 = new Array ( MAX ) . fill ( 0 ) ; for ( let i = 0 ; i < len2 ; i ++ ) freq2 [ str2 . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ] ++ ;
let minPoss = Number . MAX_SAFE_INTEGER ; for ( let i = 0 ; i < MAX ; i ++ ) {
if ( freq1 [ i ] == 0 ) continue ;
if ( freq1 [ i ] > freq2 [ i ] ) return 0 ;
minPoss = Math . min ( minPoss , Math . floor ( freq2 [ i ] / freq1 [ i ] ) ) ; } return minPoss ; }
let str1 = " " , str2 = " " ; let len1 = str1 . length ; let len2 = str2 . length ; document . write ( maxSubStr ( str1 , len1 , str2 , len2 ) ) ;
function cntWays ( str , n ) { var x = n + 1 ; var ways = x * x * ( x * x - 1 ) / 12 ; return ways ; }
var str = " " ; var n = str . length ; document . write ( cntWays ( str , n ) ) ;
var uSet = new Set ( ) ;
var minCnt = 1000000000 ;
function findSubStr ( str , cnt , start ) {
if ( start == str . length ) {
minCnt = Math . min ( cnt , minCnt ) ; }
for ( var len = 1 ; len <= ( str . length - start ) ; len ++ ) {
var subStr = str . substring ( start , start + len ) ;
if ( uSet . has ( subStr ) ) {
findSubStr ( str , cnt + 1 , start + len ) ; } } }
function findMinSubStr ( arr , n , str ) {
for ( var i = 0 ; i < n ; i ++ ) uSet . add ( arr [ i ] ) ;
findSubStr ( str , 0 , 0 ) ; }
var str = " " ; var arr = [ " " , " " , " " , " " , " " , " " ] ; var n = arr . length ; findMinSubStr ( arr , n , str ) ; document . write ( minCnt ) ;
function countSubStr ( s , n ) { var c1 = 0 , c2 = 0 ;
for ( var i = 0 ; i < n ; i ++ ) {
if ( s . substring ( i , i + 5 ) == " " ) c1 ++ ;
if ( s . substring ( i , i + 3 ) == " " ) c2 = c2 + c1 ; } return c2 ; }
var s = " " ; var n = s . length ; document . write ( countSubStr ( s , n ) ) ;
let string = " " ;
let lst1 = [ ' ' , ' ' , ' ' ] ;
let lst2 = [ ' ' , ' ' , ' ' ] ;
let lst = [ ] ;
let Dict = { ' ' : ' ' , ' ' : ' ' , ' ' : ' ' } let a = 0 , b = 0 , c = 0 ;
if ( string [ 0 ] in lst2 ) { document . write ( 1 + " " ) ; } else {
for ( let i = 0 ; i < string . length ; i ++ ) { if ( string [ i ] in lst1 ) { lst . push ( string [ i ] ) ; k = i + 2 ; } else {
if ( lst . lengt == 0 && ( string [ i ] in lst2 ) ) { document . write ( ( i + 1 ) + " " ) ; c = 1 ; break ; } else {
if ( Dict [ string [ i ] ] == lst [ lst . length - 1 ] ) { lst . pop ( ) ; } else {
break ; document . write ( ( i + 1 ) + " " ) ; a = 1 ; } } } }
if ( lst . length == 0 && c == 0 ) { document . write ( 0 + " " ) ; b = 1 ; } if ( a == 0 && b == 0 && c == 0 ) { document . write ( k + " " ) ; } }
var MAX = 26 ;
function encryptStr ( str , n , x ) {
x = x % MAX ;
var freq = Array ( MAX ) . fill ( 0 ) ; for ( var i = 0 ; i < n ; i ++ ) { freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; } for ( var i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] % 2 == 0 ) { var pos = ( str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) + x ) % MAX ; str [ i ] = String . fromCharCode ( pos + ' ' . charCodeAt ( 0 ) ) ; }
else { var pos = ( str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) - x ) ; if ( pos < 0 ) { pos += MAX ; } str [ i ] = String . fromCharCode ( pos + ' ' . charCodeAt ( 0 ) ) ; } }
return str . join ( ' ' ) ; }
var s = " " ; var n = s . length ; var x = 3 ; document . write ( encryptStr ( s . split ( ' ' ) , n , x ) ) ;
function isPossible ( str ) {
let freq = new Map ( ) ;
let max_freq = 0 ; for ( let j = 0 ; j < ( str . length ) ; j ++ ) { if ( freq . has ( str [ j ] ) ) { freq . set ( str [ j ] , freq . get ( str [ j ] ) + 1 ) ; if ( freq . get ( str [ j ] ) > max_freq ) max_freq = freq . get ( str [ j ] ) ; } else { freq . set ( str [ j ] , 1 ) ; if ( freq . get ( str [ j ] ) > max_freq ) max_freq = freq . get ( str [ j ] ) ; } }
if ( max_freq <= ( str . length - max_freq + 1 ) ) return true ; return false ; }
let str = " " ; if ( isPossible ( str . split ( ' ' ) ) ) document . write ( " " ) ; else document . write ( " " ) ;
function printUncommon ( str1 , str2 ) { var a1 = 0 , a2 = 0 ; for ( var i = 0 ; i < str1 . length ; i ++ ) {
var ch = ( str1 [ i ] . charCodeAt ( 0 ) ) - ' ' . charCodeAt ( 0 ) ;
a1 = a1 | ( 1 << ch ) ; } for ( var i = 0 ; i < str2 . length ; i ++ ) {
var ch = ( str2 [ i ] . charCodeAt ( 0 ) ) - ' ' . charCodeAt ( 0 ) ;
a2 = a2 | ( 1 << ch ) ; }
var ans = a1 ^ a2 ; var i = 0 ; while ( i < 26 ) { if ( ans % 2 == 1 ) { document . write ( String . fromCharCode ( ' ' . charCodeAt ( 0 ) + i ) ) ; } ans = parseInt ( ans / 2 ) ; i ++ ; } }
var str1 = " " ; var str2 = " " ; printUncommon ( str1 , str2 ) ;
function countMinReversals ( expr ) { var len = expr . length ;
if ( len % 2 ) return - 1 ;
var ans = 0 ; var i ;
var open = 0 ;
var close = 0 ; for ( i = 0 ; i < len ; i ++ ) {
if ( expr [ i ] == ' ' ) open ++ ;
else { if ( ! open ) close ++ ; else open -- ; } } ans = ( close / 2 ) + ( open / 2 ) ;
close %= 2 ; open %= 2 ; if ( close ) ans += 2 ; return ans ; }
var expr = " " ; document . write ( countMinReversals ( expr ) ) ;
function totalPairs ( s1 , s2 ) { var a1 = 0 , b1 = 0 ;
for ( var i = 0 ; i < s1 . length ; i ++ ) { if ( ( s1 [ i ] . charCodeAt ( 0 ) ) % 2 != 0 ) a1 ++ ; else b1 ++ ; } var a2 = 0 , b2 = 0 ;
for ( var i = 0 ; i < s2 . length ; i ++ ) { if ( ( s2 [ i ] . charCodeAt ( 0 ) ) % 2 != 0 ) a2 ++ ; else b2 ++ ; }
return ( ( a1 * a2 ) + ( b1 * b2 ) ) ; }
var s1 = " " , s2 = " " ; document . write ( totalPairs ( s1 , s2 ) ) ;
function prefixOccurrences ( str ) { var c = str . charAt ( 0 ) ; var countc = 0 ;
for ( var i = 0 ; i < str . length ; i ++ ) { if ( str . charAt ( i ) == c ) countc ++ ; } return countc ; }
var str = " " ; document . write ( prefixOccurrences ( str ) ) ;
function minOperations ( s , t , n ) { var ct0 = 0 , ct1 = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
if ( s [ i ] === t [ i ] ) continue ;
if ( s [ i ] === " " ) ct0 ++ ;
else ct1 ++ ; } return Math . max ( ct0 , ct1 ) ; }
var s = " " , t = " " ; var n = s . length ; document . write ( minOperations ( s , t , n ) ) ;
function decryptString ( str , n ) {
let i = 0 , jump = 1 ; let decryptedStr = " " ; while ( i < n ) { decryptedStr += str [ i ] ; i += jump ;
jump ++ ; } return decryptedStr ; }
let str = " " ; let n = str . length ; document . write ( decryptString ( str , n ) ) ;
function bitToBeFlipped ( s ) {
let last = s [ s . length - 1 ] ; let first = s [ 0 ] ;
if ( last == first ) { if ( last == ' ' ) { return ' ' ; } else { return ' ' ; } }
else if ( last != first ) { return last ; } }
let s = " " ; document . write ( bitToBeFlipped ( s ) , ' ' ) ;
function SieveOfEratosthenes ( prime , p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( let p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( let i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
function sumProdOfPrimeFreq ( s ) { let prime = new Array ( s . length + 1 ) ; prime . fill ( true ) ; SieveOfEratosthenes ( prime , s . length + 1 ) ; let i , j ;
let m = new Map ( ) ; for ( i = 0 ; i < s . length ; i ++ ) m . set ( s [ i ] , m . get ( s [ i ] ) == null ? 1 : m . get ( s [ i ] ) + 1 ) ; let sum = 0 , product = 1 ;
for ( let it of m ) { console . log ( m )
if ( prime [ it [ 1 ] ] ) { sum += it [ 1 ] ; product *= it [ 1 ] ; } } document . write ( " " + sum ) ; document . write ( " " + product ) ; }
let s = " " ; sumProdOfPrimeFreq ( s ) ;
function multipleOrFactor ( s1 , s2 ) {
let m1 = new Map ( ) ; let m2 = new Map ( ) ; for ( let i = 0 ; i < s1 . length ; i ++ ) { if ( m1 [ s1 [ i ] ] ) m1 [ s1 [ i ] ] ++ ; else m1 [ s1 [ i ] ] = 1 } for ( let i = 0 ; i < s2 . length ; i ++ ) { if ( m2 [ s2 [ i ] ] ) m2 [ s2 [ i ] ] ++ ; else m2 [ s2 [ i ] ] = 1 } for ( var it in m1 ) {
if ( ! ( m2 [ it ] ) ) continue ;
if ( m2 [ it ] % m1 [ it ] == 0 m1 [ it ] % m2 [ it ] == 0 ) continue ;
else return false ; } return true ; }
let s1 = " " ; let s2 = " " ; multipleOrFactor ( s1 , s2 ) ? document . write ( " " ) : document . write ( " " ) ;
function solve ( s ) {
let m = new Map ( ) ; for ( let i = 0 ; i < s . length ; i ++ ) { if ( m . has ( s [ i ] ) ) m . set ( s [ i ] , m . get ( s [ i ] ) + 1 ) ; else m . set ( s [ i ] , 1 ) ; }
let new_string = " " ;
for ( let i = 0 ; i < s . length ; i ++ ) {
if ( m . get ( s [ i ] ) % 2 == 0 ) continue ;
new_string = new_string + s [ i ] ; }
document . write ( new_string ) ; }
let s = " " ;
solve ( s ) ;
function isPalindrome ( str ) { var i = 0 , j = str . length - 1 ;
while ( i < j )
if ( str [ i ++ ] != str [ j -- ] ) return false ;
return true ; }
function removePalinWords ( str ) {
var final_str = " " , word = " " ;
str = str + " " ; var n = str . length ;
for ( var i = 0 ; i < n ; i ++ ) {
if ( str [ i ] != ' ' ) word = word + str [ i ] ; else {
if ( ! ( isPalindrome ( word ) ) ) final_str += word + " " ;
word = " " ; } }
return final_str ; }
var str = " " ; document . write ( removePalinWords ( str ) ) ;
function findSubSequence ( s , num ) {
let res = 0 ;
let i = 0 ; while ( num > 0 ) {
if ( ( num & 1 ) == 1 ) res += s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ; i ++ ;
num = num >> 1 ; } return res ; }
function combinedSum ( s ) {
let n = s . length ;
let c_sum = 0 ;
let range = ( 1 << n ) - 1 ;
for ( let i = 0 ; i <= range ; i ++ ) c_sum += findSubSequence ( s , i ) ;
return c_sum ; }
let s = " " ; document . write ( combinedSum ( s ) ) ;
var MAX_CHAR = 26 ;
function findSubsequence ( str , k ) {
var a = Array ( MAX_CHAR ) . fill ( 0 ) ;
for ( var i = 0 ; i < str . length ; i ++ ) a [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
for ( var i = 0 ; i < str . length ; i ++ ) if ( a [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] >= k ) document . write ( str [ i ] ) ; }
var k = 2 ; findSubsequence ( " " , k ) ;
function convert ( str ) {
var w = " " , z = " " ;
str = str . toUpperCase ( ) + " " ; for ( i = 0 ; i < str . length ; i ++ ) {
var ch = str [ i ] ; if ( ch != ' ' ) w = w + ch ; else {
z = z + ( w [ 0 ] . toLowerCase ( ) ) + w . substring ( 1 ) + " " ; w = " " ; } } return z ; }
var str = " " ; document . write ( convert ( str ) ) ;
function isVowel ( c ) { return c === " " || c === " " || c === " " || c === " " || c === " " ; }
function encryptString ( s , n , k ) {
var cv = new Array ( n ) . fill ( 0 ) ; var cc = new Array ( n ) . fill ( 0 ) ; if ( isVowel ( s [ 0 ] ) ) cv [ 0 ] = 1 ; else cc [ 0 ] = 1 ;
for ( var i = 1 ; i < n ; i ++ ) { cv [ i ] = cv [ i - 1 ] + ( isVowel ( s [ i ] ) === true ? 1 : 0 ) ; cc [ i ] = cc [ i - 1 ] + ( isVowel ( s [ i ] ) === true ? 0 : 1 ) ; } var ans = " " ; var prod = 0 ; prod = cc [ k - 1 ] * cv [ k - 1 ] ; ans += prod ;
for ( var i = k ; i < s . length ; i ++ ) { prod = ( cc [ i ] - cc [ i - k ] ) * ( cv [ i ] - cv [ i - k ] ) ; ans += prod ; } return ans ; }
var s = " " ; var n = s . length ; var k = 2 ; document . write ( encryptString ( s . split ( " " ) , n , k ) + " " ) ;
function countOccurrences ( str , word ) {
let a = str . split ( " " ) ;
let count = 0 ; for ( let i = 0 ; i < a . length ; i ++ ) {
if ( word == ( a [ i ] ) ) count ++ ; } return count ; }
let str = " " ; let word = " " ; document . write ( countOccurrences ( str , word ) ) ;
function permute ( input ) { var n = input . length ;
var max = 1 << n ;
input = input . toLowerCase ( ) ;
for ( var i = 0 ; i < max ; i ++ ) { var combination = input . split ( ' ' ) ;
for ( var j = 0 ; j < n ; j ++ ) { if ( ( ( i >> j ) & 1 ) == 1 ) combination [ j ] = String . fromCharCode ( combination [ j ] . charCodeAt ( 0 ) - 32 ) ; }
document . write ( combination . join ( ' ' ) ) ; document . write ( " " ) ; } }
permute ( " " ) ;
function printString ( str , ch , count ) { var occ = 0 , i ;
if ( count == 0 ) { document . write ( str ) ; return ; }
for ( i = 0 ; i < str . length ; i ++ ) {
if ( str . charAt ( i ) == ch ) occ ++ ;
if ( occ == count ) break ; }
if ( i < str . length - 1 ) document . write ( str . substring ( i + 1 ) ) ;
else document . write ( " " ) ; }
var str = " " ; printString ( str , ' ' , 2 ) ;
function isVowel ( c ) { return ( c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' ) ; }
function reverseVowel ( str ) {
let i = 0 ; let j = str . length - 1 ; let str1 = str . split ( " " ) ; while ( i < j ) { if ( ! isVowel ( str1 [ i ] ) ) { i ++ ; continue ; } if ( ! isVowel ( str1 [ j ] ) ) { j -- ; continue ; }
let t = str1 [ i ] ; str1 [ i ] = str1 [ j ] ; str1 [ j ] = t ; i ++ ; j -- ; } let str2 = ( str1 ) . join ( " " ) ; return str2 ; }
let str = " " ; document . write ( reverseVowel ( str ) ) ;
function isPalindrome ( str ) {
let l = 0 ; let h = str . length - 1 ;
while ( h > l ) if ( str [ l ++ ] != str [ h -- ] ) return false ; return true ; }
function minRemovals ( str ) {
if ( str [ 0 ] == ' ' ) return 0 ;
if ( isPalindrome ( str ) ) return 1 ;
return 2 ; }
document . write ( minRemovals ( " " ) + " " ) ; document . write ( minRemovals ( " " ) ) ;
function power ( x , y , p ) {
var res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
function findModuloByM ( X , N , M ) {
if ( N < 6 ) { var temp = " " ;
for ( var i = 1 ; i < N ; i ++ ) { temp += String . fromCharCode ( 48 + X ) ; }
var res = parseInt ( temp ) % M ; return res ; }
if ( N % 2 == 0 ) {
var half = findModuloByM ( X , N / 2 , M ) % M ;
var res = ( half * power ( 10 , N / 2 , M ) + half ) % M ; return res ; } else {
var half = findModuloByM ( X , N / 2 , M ) % M ;
var res = ( half * power ( 10 , N / 2 + 1 , M ) + half * 10 + X ) % M ; return res ; } }
var X = 6 , N = 14 , M = 9 ;
document . write ( findModuloByM ( X , N , M ) ) ;
class circle { constructor ( x , y , r ) { this . x = x ; this . y = y ; this . r = r ; } }
function check ( C ) {
let C1C2 = Math . sqrt ( ( C [ 1 ] . x - C [ 0 ] . x ) * ( C [ 1 ] . x - C [ 0 ] . x ) + ( C [ 1 ] . y - C [ 0 ] . y ) * ( C [ 1 ] . y - C [ 0 ] . y ) ) ;
let flag = false ;
if ( C1C2 < ( C [ 0 ] . r + C [ 1 ] . r ) ) {
if ( ( C [ 0 ] . x + C [ 1 ] . x ) == 2 * C [ 2 ] . x && ( C [ 0 ] . y + C [ 1 ] . y ) == 2 * C [ 2 ] . y ) {
flag = true ; } }
return flag ; }
function IsFairTriplet ( c ) { let f = false ;
f |= check ( c ) ; for ( let i = 0 ; i < 2 ; i ++ ) { swap ( c [ 0 ] , c [ 2 ] ) ;
f |= check ( c ) ; } return f ; } function swap ( circle1 , circle2 ) { let temp = circle1 ; circle1 = circle2 ; circle2 = temp ; }
let C = new Array ( 3 ) ; C [ 0 ] = new circle ( 0 , 0 , 8 ) ; C [ 1 ] = new circle ( 0 , 10 , 6 ) ; C [ 2 ] = new circle ( 0 , 5 , 5 ) ; if ( IsFairTriplet ( C ) ) document . write ( " " ) ; else document . write ( " " ) ;
function eccHyperbola ( A , B ) {
let r = B * B / A * A ;
r += 1 ;
return Math . sqrt ( r ) ; }
let A = 3.0 ; let B = 2.0 ; document . write ( eccHyperbola ( A , B ) ) ;
function calculateArea ( A , B , C , D ) {
let S = ( A + B + C + D ) / 2
let area = Math . sqrt ( ( S - A ) * ( S - B ) * ( S - C ) * ( S - D ) )
return area ; }
let A = 10 ; let B = 15 ; let C = 20 ; let D = 25 ; document . write ( calculateArea ( A , B , C , D ) . toFixed ( 3 ) )
function triangleArea ( a , b ) {
ratio = b / a
document . write ( ratio ) }
var a = 1 var b = 2 triangleArea ( a , b )
function distance ( m , n , p , q ) { return Math . sqrt ( Math . pow ( n - m , 2 ) + Math . pow ( q - p , 2 ) * 1.0 ) ; }
function Excenters ( x1 , y1 , x2 , y2 , x3 , y3 ) {
var a = distance ( x2 , x3 , y2 , y3 ) ; var b = distance ( x3 , x1 , y3 , y1 ) ; var c = distance ( x1 , x2 , y1 , y2 ) ;
var excenter = new Array ( 4 ) ; for ( var i = 0 ; i < 4 ; i ++ ) excenter [ i ] = new Array ( 2 ) ;
excenter [ 1 ] [ 0 ] = ( - ( a * x1 ) + ( b * x2 ) + ( c * x3 ) ) / ( - a + b + c ) ; excenter [ 1 ] [ 1 ] = ( - ( a * y1 ) + ( b * y2 ) + ( c * y3 ) ) / ( - a + b + c ) ;
excenter [ 2 ] [ 0 ] = ( ( a * x1 ) - ( b * x2 ) + ( c * x3 ) ) / ( a - b + c ) ; excenter [ 2 ] [ 1 ] = ( ( a * y1 ) - ( b * y2 ) + ( c * y3 ) ) / ( a - b + c ) ;
excenter [ 3 ] [ 0 ] = ( ( a * x1 ) + ( b * x2 ) - ( c * x3 ) ) / ( a + b - c ) ; excenter [ 3 ] [ 1 ] = ( ( a * y1 ) + ( b * y2 ) - ( c * y3 ) ) / ( a + b - c ) ;
for ( var i = 1 ; i <= 3 ; i ++ ) { document . write ( excenter [ i ] [ 0 ] + " " + excenter [ i ] [ 1 ] + " " ) ; } }
var x1 , x2 , x3 , y1 , y2 , y3 ; x1 = 0 ; x2 = 3 ; x3 = 0 ; y1 = 0 ; y2 = 0 ; y3 = 4 ; Excenters ( x1 , y1 , x2 , y2 , x3 , y3 ) ;
function Icositetragonal_num ( n ) {
return ( 22 * n * n - 20 * n ) / 2 ; }
let n = 3 ; document . write ( Icositetragonal_num ( n ) + " " ) ; n = 10 ; document . write ( Icositetragonal_num ( n ) ) ;
function area_of_circle ( m , n ) {
var square_of_radius = ( m * n ) / 4 ; var area = ( 3.141 * square_of_radius ) ; return area ; }
var n = 10 ; var m = 30 ; document . write ( area_of_circle ( m , n ) ) ;
function area ( R ) {
var base = 1.732 * R ; var height = ( 1.5 ) * R ;
var area = 0.5 * base * height ; return area ; }
var R = 7 ; document . write ( area ( R ) ) ;
function circlearea ( R ) {
if ( R < 0 ) return - 1 ;
var a = 3.14 * R * R / 4 ; return a ; }
var R = 2 ; document . write ( circlearea ( R ) ) ;
function countPairs ( P , Q , N , M ) {
var A = [ 0 , 0 ] , B = [ 0 , 0 ] ;
for ( var i = 0 ; i < N ; i ++ ) A [ P [ i ] % 2 ] ++ ;
for ( var i = 0 ; i < M ; i ++ ) B [ Q [ i ] % 2 ] ++ ;
return ( A [ 0 ] * B [ 0 ] + A [ 1 ] * B [ 1 ] ) ; }
var P = [ 1 , 3 , 2 ] , Q = [ 3 , 0 ] ; var N = P . length ; var M = Q . length ; document . write ( countPairs ( P , Q , N , M ) ) ;
function countIntersections ( n ) { return n * ( n - 1 ) / 2 ; }
var n = 3 ; document . write ( countIntersections ( n ) ) ;
var PI = 3.14159
function areaOfTriangle ( d ) {
var c = 1.618 * d ; var s = ( d + c + c ) / 2 ;
var area = Math . sqrt ( s * ( s - c ) * ( s - c ) * ( s - d ) ) ;
return 5 * area ; }
function areaOfRegPentagon ( d ) {
var cal = 4 * Math . tan ( PI / 5 ) ; var area = ( 5 * d * d ) / cal ;
return area ; }
function areaOfPentagram ( d ) {
return areaOfRegPentagon ( d ) + areaOfTriangle ( d ) ; }
var d = 5 ; document . write ( areaOfPentagram ( d ) . toFixed ( 3 ) ) ;
function anglequichord ( z ) { document . write ( " " + z + " " ) ; }
var z = 48 ; anglequichord ( z ) ;
function convertToASCII ( N ) { let num = N . toString ( ) ; for ( let ch = 0 ; ch < num . length ; ch ++ ) { document . write ( num [ ch ] + " " + num [ ch ] . charCodeAt ( 0 ) + " " ) ; } }
let N = 36 ; convertToASCII ( N ) ;
function productExceptSelf ( arr , N ) {
let product = 1 ;
let z = 0 ;
for ( let i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] != 0 ) product *= arr [ i ] ;
if ( arr [ i ] == 0 ) z += 1 ; }
let a = Math . abs ( product ) ; for ( let i = 0 ; i < N ; i ++ ) {
if ( z == 1 ) {
if ( arr [ i ] != 0 ) arr [ i ] = 0 ;
else arr [ i ] = product ; continue ; }
else if ( z > 1 ) {
arr [ i ] = 0 ; continue ; }
let b = Math . abs ( arr [ i ] ) ;
let curr = Math . round ( Math . exp ( Math . log ( a ) - Math . log ( b ) ) ) ;
if ( arr [ i ] < 0 && product < 0 ) arr [ i ] = curr ;
else if ( arr [ i ] > 0 && product > 0 ) arr [ i ] = curr ;
else arr [ i ] = - 1 * curr ; }
for ( let i = 0 ; i < N ; i ++ ) { document . write ( arr [ i ] + " " ) ; } }
let arr = [ 10 , 3 , 5 , 6 , 2 ] ; let N = arr . length ;
productExceptSelf ( arr , N ) ;
function singleDigitSubarrayCount ( arr , N ) {
let res = 0 ;
let count = 0 ;
for ( let i = 0 ; i < N ; i ++ ) { if ( arr [ i ] <= 9 ) {
count ++ ;
res += count ; } else {
count = 0 ; } } document . write ( res ) ; }
let arr = [ 0 , 1 , 14 , 2 , 5 ] ;
let N = arr . length ; singleDigitSubarrayCount ( arr , N ) ;
function isPossible ( N ) { return ( ( ( N & ( N - 1 ) ) & N ) ) ; }
function countElements ( N ) {
var count = 0 ; for ( i = 1 ; i <= N ; i ++ ) { if ( isPossible ( i ) != 0 ) count ++ ; } document . write ( count ) ; }
var N = 15 ; countElements ( N ) ;
function countElements ( N ) { var Cur_Ele = 1 ; var Count = 0 ;
while ( Cur_Ele <= N ) {
Count ++ ;
Cur_Ele = Cur_Ele * 2 ; } document . write ( N - Count ) ; }
var N = 15 ; countElements ( N ) ;
function maxAdjacent ( arr , N ) { let res = [ ] ; let arr_max = Number . MIN_VALUE ;
for ( let i = 1 ; i < N ; i ++ ) { arr_max = Math . max ( arr_max , Math . abs ( arr [ i - 1 ] - arr [ i ] ) ) ; } for ( let i = 1 ; i < N - 1 ; i ++ ) { let curr_max = Math . abs ( arr [ i - 1 ] - arr [ i + 1 ] ) ;
let ans = Math . max ( curr_max , arr_max ) ;
res . push ( ans ) ; }
document . write ( res . join ( " " ) ) ; }
let arr = [ 1 , 3 , 4 , 7 , 8 ] ; let N = arr . length ; maxAdjacent ( arr , N ) ;
function minimumIncrement ( arr , N ) {
if ( N % 2 != 0 ) { document . write ( " " ) ; System . exit ( 0 ) ; }
var cntEven = 0 ;
var cntOdd = 0 ;
for ( i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 == 0 ) {
cntEven += 1 ; } }
cntOdd = N - cntEven ;
return Math . abs ( cntEven - cntOdd ) / 2 ; }
var arr = [ 1 , 3 , 4 , 9 ] ; var N = arr . length ;
document . write ( minimumIncrement ( arr , N ) ) ;
function cntWaysConsArray ( A , N ) {
var total = 1 ;
var oddArray = 1 ;
for ( i = 0 ; i < N ; i ++ ) {
total = total * 3 ;
if ( A [ i ] % 2 == 0 ) {
oddArray *= 2 ; } }
document . write ( total - oddArray ) ; }
var A = [ 2 , 4 ] ; var N = A . length ; cntWaysConsArray ( A , N ) ;
function countNumberHavingKthBitSet ( N , K ) {
let numbers_rightmost_setbit_K = 0 ; for ( let i = 1 ; i <= K ; i ++ ) {
let numbers_rightmost_bit_i = ( N + 1 ) / 2 ;
N -= numbers_rightmost_bit_i ;
if ( i == K ) { numbers_rightmost_setbit_K = numbers_rightmost_bit_i ; } } document . write ( numbers_rightmost_setbit_K ) ; }
let N = 15 ; let K = 2 ; countNumberHavingKthBitSet ( N , K ) ;
function countSetBits ( N ) { let count = 0 ;
while ( N != 0 ) { N = N & ( N - 1 ) ; count ++ ; }
return count ; }
let N = 4 ; let bits = countSetBits ( N ) ;
document . write ( " " + " " + ( Math . pow ( 2 , bits ) ) + " " ) ;
document . write ( " " + " " + ( N + 1 - ( Math . pow ( 2 , bits ) ) ) ) ;
function minMoves ( arr , N ) {
var odd_element_cnt = 0 ; var i ;
for ( i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 != 0 ) { odd_element_cnt ++ ; } }
var moves = Math . floor ( ( odd_element_cnt ) / 2 ) ;
if ( odd_element_cnt % 2 != 0 ) moves += 2 ;
document . write ( moves ) ; }
var arr = [ 5 , 6 , 3 , 7 , 20 ] ; N = arr . length ;
minMoves ( arr , N ) ;
function minimumSubsetDifference ( N ) {
let blockOfSize8 = N / 8 ;
let str = " " ;
let subsetDifference = 0 ;
let partition = " " ; while ( blockOfSize8 -- > 0 ) { partition += str ; }
let A = [ ] ; let B = [ ] ; let x = 0 , y = 0 ; for ( let i = 0 ; i < N ; i ++ ) {
if ( partition [ i ] == ' ' ) { A [ x ++ ] = ( ( i + 1 ) * ( i + 1 ) ) ; }
else { B [ y ++ ] = ( ( i + 1 ) * ( i + 1 ) ) ; } }
document . write ( subsetDifference + " " ) ;
for ( let i = 0 ; i < x ; i ++ ) document . write ( A [ i ] + " " ) ; document . write ( " " ) ;
for ( let i = 0 ; i < y ; i ++ ) document . write ( B [ i ] + " " ) ; }
let N = 8 ;
minimumSubsetDifference ( N ) ;
function findTheGreatestX ( P , Q ) {
var divisiors = new Map ( ) ; for ( var i = 2 ; i * i <= Q ; i ++ ) { while ( Q % i == 0 && Q > 1 ) { Q = parseInt ( Q / i ) ;
if ( divisiors . has ( i ) ) divisiors . set ( i , divisiors . get ( i ) + 1 ) else divisiors . set ( i , 1 ) } }
if ( Q > 1 ) if ( divisiors . has ( Q ) ) divisiors . set ( Q , divisiors . get ( Q ) + 1 ) else divisiors . set ( Q , 1 )
var ans = 0 ;
divisiors . forEach ( ( value , key ) => { var frequency = value ; var temp = P ;
var cur = 0 ; while ( temp % key == 0 ) { temp = parseInt ( temp / key ) ;
cur ++ ; }
if ( cur < frequency ) { ans = P ; } temp = P ;
for ( var j = cur ; j >= frequency ; j -- ) { temp = parseInt ( temp / key ) ; }
ans = Math . max ( temp , ans ) ; } ) ;
document . write ( ans ) ; }
var P = 10 , Q = 4 ;
findTheGreatestX ( P , Q ) ;
function checkRearrangements ( mat , N , M ) {
for ( let i = 0 ; i < N ; i ++ ) { for ( let j = 1 ; j < M ; j ++ ) { if ( mat [ i ] [ 0 ] != mat [ i ] [ j ] ) { return " " ; } } } return " " ; }
function nonZeroXor ( mat , N , M ) { let res = 0 ;
for ( let i = 0 ; i < N ; i ++ ) { res = res ^ mat [ i ] [ 0 ] ; }
if ( res != 0 ) return " " ;
else return checkRearrangements ( mat , N , M ) ; }
let mat = [ [ 1 , 1 , 2 ] , [ 2 , 2 , 2 ] , [ 3 , 3 , 3 ] ] ; let N = mat . length ; let M = mat [ 0 ] . length ;
document . write ( nonZeroXor ( mat , N , M ) ) ;
function functionMax ( arr , n ) {
var setBit = Array . from ( Array ( 32 ) , ( ) => new Array ( ) ) ; for ( var i = 0 ; i < n ; i ++ ) { for ( var j = 0 ; j < size_int ; j ++ ) {
if ( arr [ i ] & ( 1 << j ) )
setBit [ j ] . push ( i ) ; } }
for ( var i = size_int - 1 ; i >= 0 ; i -- ) { if ( setBit [ i ] . length == 1 ) {
[ arr [ 0 ] , arr [ setBit [ i ] [ 0 ] ] ] = [ arr [ setBit [ i ] [ 0 ] ] , arr [ 0 ] ] ; break ; } }
var maxAnd = arr [ 0 ] ; for ( var i = 1 ; i < n ; i ++ ) { maxAnd = maxAnd & ( ~ arr [ i ] ) ; }
return maxAnd ; }
var arr = [ 1 , 2 , 4 , 8 , 16 ] ; var n = arr . length ;
document . write ( functionMax ( arr , n ) ) ;
function nCr ( n , r ) {
let res = 1 ;
if ( r > n - r ) r = n - r ;
for ( let i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
function solve ( n , m , k ) {
let sum = 0 ;
for ( let i = 0 ; i <= k ; i ++ ) sum += nCr ( n , i ) * nCr ( m , k - i ) ; return sum ; }
let n = 3 , m = 2 , k = 2 ; document . write ( solve ( n , m , k ) ) ;
function powerOptimised ( a , n ) {
let ans = 1 ; while ( n > 0 ) { let last_bit = ( n & 1 ) ;
if ( last_bit > 0 ) { ans = ans * a ; } a = a * a ;
n = n >> 1 ; } return ans ; }
let a = 3 , n = 5 ; document . write ( powerOptimised ( a , n ) ) ;
function findMaximumGcd ( n ) {
let max_gcd = 1 ;
for ( let i = 1 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
if ( i > max_gcd ) max_gcd = i ; if ( ( n / i != i ) && ( n / i != n ) && ( ( n / i ) > max_gcd ) ) max_gcd = n / i ; } }
return max_gcd ; }
let N = 10 ;
document . write ( findMaximumGcd ( N ) ) ;
let x = 2000021
let v = new Array ( x ) ;
function sieve ( ) { v [ 1 ] = 1 ;
for ( let i = 2 ; i < x ; i ++ ) v [ i ] = i ;
for ( let i = 4 ; i < x ; i += 2 ) v [ i ] = 2 ; for ( let i = 3 ; i * i < x ; i ++ ) {
if ( v [ i ] == i ) {
for ( let j = i * i ; j < x ; j += i ) {
if ( v [ j ] == j ) { v [ j ] = i ; } } } } }
function prime_factors ( n ) { let s = new Set ( ) ; while ( n != 1 ) { s . add ( v [ n ] ) ; n = n / v [ n ] ; } return s . size ; }
function distinctPrimes ( m , k ) {
let result = new Array ( ) ; for ( let i = 14 ; i < m + k ; i ++ ) {
let count = prime_factors ( i ) ;
if ( count == k ) { result . push ( i ) ; } } let p = result . length ; for ( let index = 0 ; index < p - 1 ; index ++ ) { let element = result [ index ] ; let count = 1 , z = index ;
while ( z < p - 1 && count <= k && result [ z ] + 1 == result [ z + 1 ] ) {
count ++ ; z ++ ; }
if ( count >= k ) document . write ( element + ' ' ) ; } }
sieve ( ) ;
let N = 1000 , K = 3 ;
distinctPrimes ( N , K ) ;
function print_product ( a , b , c , d ) {
let prod1 = a * c ; let prod2 = b * d ; let prod3 = ( a + b ) * ( c + d ) ;
let real = prod1 - prod2 ;
let imag = prod3 - ( prod1 + prod2 ) ;
document . write ( real + " " + imag + " " ) ; }
a = 2 ; b = 3 ; c = 4 ; d = 5 ;
print_product ( a , b , c , d ) ;
function isInsolite ( n ) { let N = n ;
let sum = 0 ;
let product = 1 ; while ( n != 0 ) {
let r = n % 10 ; sum = sum + r * r ; product = product * r * r ; n = parseInt ( n / 10 ) ; } return ( N % sum == 0 ) && ( N % product == 0 ) ; }
let N = 111 ;
if ( isInsolite ( N ) ) document . write ( " " ) ; else document . write ( " " ) ;
function sigma ( n ) { if ( n == 1 ) return 1 ;
var result = 0 ;
for ( var i = 2 ; i <= Math . sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) {
if ( i == ( n / i ) ) result += i ; else result += ( i + n / i ) ; } }
return ( result + n + 1 ) ; }
function isSuperabundant ( N ) {
for ( var i = 1 ; i < N ; i ++ ) { var x = sigma ( i ) / i ; var y = sigma ( N ) / ( N * 1.0 ) ; if ( x > y ) return false ; } return true ; }
var N = 4 ; isSuperabundant ( N ) ? document . write ( " " ) : document . write ( " " ) ;
function isDNum ( n ) {
if ( n < 4 ) return false ; let numerator = 0 , hcf = 0 ;
for ( k = 2 ; k <= n ; k ++ ) { numerator = parseInt ( ( Math . pow ( k , n - 2 ) - k ) ) ; hcf = __gcd ( n , k ) ; }
if ( hcf == 1 && ( numerator % n ) != 0 ) return false ; return true ; } function __gcd ( a , b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
let n = 15 ; let a = isDNum ( n ) ; if ( a ) document . write ( " " ) ; else document . write ( " " ) ;
function Sum ( N ) { let SumOfPrimeDivisors = Array ( N + 1 ) . fill ( 0 ) ; for ( let i = 2 ; i <= N ; ++ i ) {
if ( SumOfPrimeDivisors [ i ] == 1 ) {
for ( let j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
function RuthAaronNumber ( n ) { if ( Sum ( n ) == Sum ( n + 1 ) ) return true ; else return false ; }
let N = 714 ; if ( RuthAaronNumber ( N ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function maxAdjacentDifference ( N , K ) {
if ( N == 1 ) { return 0 ; }
if ( N == 2 ) { return K ; }
return 2 * K ; }
let N = 6 ; let K = 11 ; document . write ( maxAdjacentDifference ( N , K ) ) ;
let mod = 1000000007 ;
function linearSum ( n ) { return ( n * ( n + 1 ) / 2 ) % mod ; }
function rangeSum ( b , a ) { return ( linearSum ( b ) - linearSum ( a ) ) % mod ; }
function totalSum ( n ) {
let result = 0 ; let i = 1 ;
while ( true ) {
result += rangeSum ( Math . floor ( n / i ) , Math . floor ( n / ( i + 1 ) ) ) * ( i % mod ) % mod ; result %= mod ; if ( i == n ) break ; i = Math . floor ( n / ( n / ( i + 1 ) ) ) ; } return result ; }
let N = 4 ; document . write ( totalSum ( N ) + " " ) ; N = 12 ; document . write ( totalSum ( N ) ) ;
function isDouble ( num ) { let s = num . toString ( ) ; let l = s . length ;
if ( s [ 0 ] == s . charAt [ 1 ] ) return false ;
if ( l % 2 == 1 ) { s = s + s [ 1 ] ; l ++ ; }
let s1 = s . substr ( 0 , l / 2 ) ;
let s2 = s . substr ( l / 2 ) ;
return ( s1 == s2 ) ; }
function isNontrivialUndulant ( N ) { return N > 100 && isDouble ( N ) ; }
let n = 121 ; if ( isNontrivialUndulant ( n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function MegagonNum ( n ) { return ( 999998 * n * n - 999996 * n ) / 2 ; }
var n = 3 ; document . write ( MegagonNum ( n ) ) ;
mod = 1000000007
function productPairs ( arr , n ) {
let product = 1 ;
for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) {
product *= ( arr [ i ] % mod * arr [ j ] % mod ) % mod ; product = product % mod ; } }
return product % mod ; }
let arr = [ 1 , 2 , 3 ] ; let n = arr . length ; document . write ( productPairs ( arr , n ) ) ;
let mod = 1000000007 ;
function power ( x , y ) { let p = 1000000007 ;
let res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ; y = y >> 1 ; x = ( x * x ) % p ; }
return res ; }
function productPairs ( arr , n ) {
let product = 1 ;
for ( let i = 0 ; i < n ; i ++ ) {
product = ( product % mod * power ( arr [ i ] , ( 2 * n ) ) % mod ) % mod ; } return product % mod ; }
let arr = [ 1 , 2 , 3 ] ; let n = arr . length ; document . write ( productPairs ( arr , n ) ) ;
function constructArray ( N ) { let arr = new Array ( N ) ;
for ( let i = 1 ; i <= N ; i ++ ) { arr [ i - 1 ] = i ; }
for ( let i = 0 ; i < N ; i ++ ) { document . write ( arr [ i ] + " " ) ; } }
let N = 6 ; constructArray ( N ) ;
function isPrime ( n ) { if ( n <= 1 ) return false ; for ( var i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
function countSubsequences ( arr , n ) {
var totalSubsequence = Math . pow ( 2 , n ) - 1 ; var countPrime = 0 , countOnes = 0 ;
for ( var i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) countOnes ++ ; else if ( isPrime ( arr [ i ] ) ) countPrime ++ ; } var compositeSubsequence ;
var onesSequence = Math . pow ( 2 , countOnes ) - 1 ;
compositeSubsequence = totalSubsequence - countPrime - onesSequence - onesSequence * countPrime ; return compositeSubsequence ; }
var arr = [ 2 , 1 , 2 ] ; var n = arr . length ; document . write ( countSubsequences ( arr , n ) ) ;
function checksum ( n , k ) {
var first_term = ( ( ( 2 * n ) / k + ( 1 - k ) ) / 2.0 ) ;
if ( first_term - parseInt ( ( first_term ) ) == 0 ) {
for ( i = parseInt ( first_term ) ; i <= first_term + k - 1 ; i ++ ) { document . write ( i + " " ) ; } } else document . write ( " " ) ; }
var n = 33 , k = 6 ; checksum ( n , k ) ;
function sumEvenNumbers ( N , K ) { let check = N - 2 * ( K - 1 ) ;
if ( check > 0 && check % 2 == 0 ) { for ( let i = 0 ; i < K - 1 ; i ++ ) { document . write ( " " ) ; } document . write ( check ) ; } else { document . write ( " " ) ; } }
let N = 8 ; let K = 2 ; sumEvenNumbers ( N , K ) ;
function calculateWays ( n ) { let x = 0 ;
let v = Array . from ( { length : n } , ( _ , i ) => 0 ) ; for ( let i = 0 ; i < n ; i ++ ) v [ i ] = 0 ;
for ( let i = 0 ; i < n / 2 ; i ++ ) {
if ( n % 2 == 0 && i == n / 2 ) break ;
x = n * ( i + 1 ) - ( i + 1 ) * i ;
v [ i ] = x ; v [ n - i - 1 ] = x ; } return v ; }
function prletArray ( v ) { for ( let i = 0 ; i < v . length ; i ++ ) document . write ( v [ i ] + " " ) ; }
let v ; v = calculateWays ( 4 ) ; prletArray ( v ) ;
var MAXN = 10000000 ;
function sumOfDigits ( n ) {
var sum = 0 ; while ( n > 0 ) {
sum += n % 10 ;
n = parseInt ( n / 10 ) ; } return sum ; }
function smallestNum ( X , Y ) {
var res = - 1 ;
for ( i = X ; i < MAXN ; i ++ ) {
var sum_of_digit = sumOfDigits ( i ) ;
if ( sum_of_digit % Y == 0 ) { res = i ; break ; } } return res ; }
var X = 5923 , Y = 13 ; document . write ( smallestNum ( X , Y ) ) ;
function countValues ( N ) { var div = [ ] ;
for ( var i = 2 ; i * i <= N ; i ++ ) {
if ( N % i == 0 ) { div . push ( i ) ;
if ( N != i * i ) { div . push ( N / i ) ; } } } var answer = 0 ;
for ( var i = 1 ; i * i <= N - 1 ; i ++ ) {
if ( ( N - 1 ) % i == 0 ) { if ( i * i == N - 1 ) answer ++ ; else answer += 2 ; } }
div . forEach ( d => { var K = N ; while ( K % d == 0 ) K /= d ; if ( ( K - 1 ) % d == 0 ) answer ++ ; } ) ; return answer ; }
var N = 6 ; document . write ( countValues ( N ) ) ;
function findMaxPrimeDivisor ( n ) { let max_possible_prime = 0 ;
while ( n % 2 == 0 ) { max_possible_prime ++ ; n = Math . floor ( n / 2 ) ; }
for ( let i = 3 ; i * i <= n ; i = i + 2 ) { while ( n % i == 0 ) { max_possible_prime ++ ; n = Math . floor ( n / i ) ; } }
if ( n > 2 ) { max_possible_prime ++ ; } document . write ( max_possible_prime + " " ) ; }
let n = 4 ;
findMaxPrimeDivisor ( n ) ;
function CountWays ( n ) { let ans = Math . floor ( ( n - 1 ) / 2 ) ; return ans ; }
let N = 8 ; document . write ( CountWays ( N ) ) ;
function Solve ( arr , size , n ) { let v = Array . from ( { length : n + 1 } , ( _ , i ) => 0 ) ;
for ( let i = 0 ; i < size ; i ++ ) v [ arr [ i ] ] ++ ;
let max1 = - 1 , mx = - 1 ; for ( let i = 0 ; i < v . length ; i ++ ) { if ( v [ i ] > mx ) { mx = v [ i ] ; max1 = i ; } }
let cnt = 0 ; for ( let i in v ) { if ( i == 0 ) ++ cnt ; } let diff1 = n + 1 - cnt ;
let max_size = Math . max ( Math . min ( v [ max1 ] - 1 , diff1 ) , Math . min ( v [ max1 ] , diff1 - 1 ) ) ; document . write ( " " + max_size + " " ) ;
document . write ( " " + " " ) ; for ( let i = 0 ; i < max_size ; i ++ ) { document . write ( max1 + " " ) ; v [ max1 ] -= 1 ; } document . write ( " " ) ;
document . write ( " " + " " ) ; for ( let i = 0 ; i < ( n + 1 ) ; i ++ ) { if ( v [ i ] > 0 ) { document . write ( i + " " ) ; max_size -- ; } if ( max_size < 1 ) break ; } document . write ( " " ) ; }
let n = 7 ;
let arr = [ 1 , 2 , 1 , 5 , 1 , 6 , 7 , 2 ] ;
let size = arr . length ; Solve ( arr , size , n ) ;
function power ( x , y , p ) {
let res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
function modInverse ( n , p ) { return power ( n , p - 2 , p ) ; }
function nCrModPFermat ( n , r , p ) {
if ( r == 0 ) return 1 ; if ( n < r ) return 0 ;
let fac = Array . from ( { length : n + 1 } , ( _ , i ) => 0 ) ; fac [ 0 ] = 1 ; for ( let i = 1 ; i <= n ; i ++ ) fac [ i ] = fac [ i - 1 ] * i % p ; return ( fac [ n ] * modInverse ( fac [ r ] , p ) % p * modInverse ( fac [ n - r ] , p ) % p ) % p ; }
function SumOfXor ( a , n ) { let mod = 10037 ; let answer = 0 ;
for ( let k = 0 ; k < 32 ; k ++ ) {
let x = 0 , y = 0 ; for ( let i = 0 ; i < n ; i ++ ) {
if ( ( a [ i ] & ( 1 << k ) ) != 0 ) x ++ ; else y ++ ; }
answer += ( ( 1 << k ) % mod * ( nCrModPFermat ( x , 3 , mod ) + x * nCrModPFermat ( y , 2 , mod ) ) % mod ) % mod ; } return answer ; }
let n = 5 ; let A = [ 3 , 5 , 2 , 18 , 7 ] ; document . write ( SumOfXor ( A , n ) ) ;
function round ( vr , digit ) { var value = parseInt ( ( vr * Math . pow ( 10 , digit ) + .5 ) ) ; return value / Math . pow ( 10 , digit ) ; }
function probability ( N ) {
var a = 2 ; var b = 3 ;
if ( N == 1 ) { return a ; } else if ( N == 2 ) { return b ; } else {
for ( i = 3 ; i <= N ; i ++ ) { var c = a + b ; a = b ; b = c ; } return b ; } }
function operations ( N ) {
var x = probability ( N ) ;
var y = parseInt ( Math . pow ( 2 , N ) ) ; return round ( x / y , 2 ) ; }
var N = 10 ; document . write ( ( operations ( N ) ) ) ;
function isPerfectCube ( x ) { var cr = Math . round ( Math . cbrt ( x ) ) ; return ( cr * cr * cr == x ) ; }
function checkCube ( a , b ) {
s1 = a . toString ( ) ; s2 = b . toString ( ) ;
var c = parseInt ( s1 + s2 ) ;
if ( isPerfectCube ( c ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
var a = 6 ; var b = 4 ; checkCube ( a , b ) ;
function largest_sum ( arr , n ) {
let maximum = - 1 ;
let m = new Map ( ) ;
for ( let i = 0 ; i < n ; i ++ ) { if ( m . has ( arr [ i ] ) ) { m . set ( arr [ i ] , m . get ( arr [ i ] ) + 1 ) ; } else { m . set ( arr [ i ] , 1 ) ; } }
for ( let i = 0 ; i < n ; i ++ ) {
if ( m . get ( arr [ i ] ) > 1 ) { if ( m . has ( 2 * arr [ i ] ) ) {
m . set ( 2 * arr [ i ] , m . get ( 2 * arr [ i ] ) + m . get ( arr [ i ] ) / 2 ) ; } else { m . set ( 2 * arr [ i ] , m . get ( arr [ i ] ) / 2 ) ; }
if ( 2 * arr [ i ] > maximum ) maximum = 2 * arr [ i ] ; } }
return maximum ; }
let arr = [ 1 , 1 , 2 , 4 , 7 , 8 ] ; let n = arr . length ;
document . write ( largest_sum ( arr , n ) ) ;
function canBeReduced ( x , y ) { var maxi = Math . max ( x , y ) ; var mini = Math . min ( x , y ) ;
if ( ( ( x + y ) % 3 ) == 0 && maxi <= 2 * mini ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ; }
var x = 6 , y = 9 ;
canBeReduced ( x , y ) ;
function isPrime ( N ) { let isPrime = true ;
let arr = [ 7 , 11 , 13 , 17 , 19 , 23 , 29 , 31 ] ;
if ( N < 2 ) { isPrime = false ; }
if ( N % 2 == 0 N % 3 == 0 N % 5 == 0 ) { isPrime = false ; }
for ( let i = 0 ; i < Math . sqrt ( N ) ; i += 30 ) {
for ( let c of arr ) {
if ( c > Math . sqrt ( N ) ) { break ; }
else { if ( N % ( c + i ) == 0 ) { isPrime = false ; break ; } }
if ( ! isPrime ) break ; } } if ( isPrime ) document . write ( " " ) ; else document . write ( " " ) ; }
let N = 121 ;
isPrime ( N ) ;
function printPairs ( arr , n ) {
for ( var i = 0 ; i < n ; i ++ ) { for ( var j = 0 ; j < n ; j ++ ) { document . write ( " " + arr [ i ] + " " + arr [ j ] + " " + " " ) ; } } }
var arr = [ 1 , 2 ] ; var n = arr . length ; printPairs ( arr , n ) ;
function circle ( x1 , y1 , x2 , y2 , r1 , r2 ) { var distSq = parseInt ( Math . sqrt ( ( ( x1 - x2 ) * ( x1 - x2 ) ) + ( ( y1 - y2 ) * ( y1 - y2 ) ) ) ) ; if ( distSq + r2 == r1 ) { document . write ( " " + " " + " " + " " ) ; } else if ( distSq + r2 < r1 ) { document . write ( " " + " " + " " + " " ) ; } else { document . write ( " " + " " ) ; } }
var x1 = 10 , y1 = 8 ; var x2 = 1 , y2 = 2 ; var r1 = 30 , r2 = 10 ; circle ( x1 , y1 , x2 , y2 , r1 , r2 ) ;
function lengtang ( r1 , r2 , d ) { document . write ( " " + " " + ( Math . sqrt ( Math . pow ( d , 2 ) - Math . pow ( ( r1 - r2 ) , 2 ) ) ) . toFixed ( 5 ) ) ; }
var r1 = 4 , r2 = 6 , d = 3 ; lengtang ( r1 , r2 , d ) ;
function rad ( d , h ) { document . write ( " " + ( ( d * d ) / ( 8 * h ) + h / 2 ) ) ; }
var d = 4 , h = 1 ; rad ( d , h ) ;
function shortdis ( r , d ) { document . write ( " " + " " + Math . sqrt ( ( r * r ) - ( ( d * d ) / 4 ) ) + " " ) ; }
let r = 4 , d = 3 ; shortdis ( r , d ) ;
function lengtang ( r1 , r2 , d ) { document . write ( " " + Math . sqrt ( Math . pow ( d , 2 ) - Math . pow ( ( r1 - r2 ) , 2 ) ) ) ; }
var r1 = 4 , r2 = 6 , d = 12 ; lengtang ( r1 , r2 , d ) ;
function square ( a ) {
if ( a < 0 ) return - 1 ;
var x = 0.464 * a ; return x ; }
var a = 5 ; document . write ( square ( a ) . toFixed ( 2 ) ) ;
function polyapothem ( n , a ) {
if ( a < 0 && n < 0 ) return - 1 ;
return ( a / ( 2 * Math . tan ( ( 180 / n ) * 3.14159 / 180 ) ) ) ; }
var a = 9 , n = 6 ; document . write ( polyapothem ( n , a ) . toFixed ( 5 ) ) ;
function polyarea ( n , a ) {
if ( a < 0 && n < 0 ) return - 1 ;
var A = ( a * a * n ) / ( 4 * Math . tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; return A ; }
var a = 9 , n = 6 ; document . write ( polyarea ( n , a ) . toFixed ( 5 ) ) ;
function calculateSide ( n , r ) { var theta , theta_in_radians ; theta = 360 / n ; theta_in_radians = theta * 3.14 / 180 ; return 2 * r * Math . sin ( theta_in_radians / 2 ) ; }
var n = 3 ;
var r = 5 ; document . write ( calculateSide ( n , r ) . toFixed ( 5 ) ) ;
function cyl ( r , R , h ) {
if ( h < 0 && r < 0 && R < 0 ) return - 1 ;
var r1 = r ;
var h1 = h ;
var V = ( 3.14 * Math . pow ( r1 , 2 ) * h1 ) ; return V ; }
var r = 7 , R = 11 , h = 6 ; document . write ( cyl ( r , R , h ) . toFixed ( 5 ) ) ;
function Perimeter ( s , n ) { var perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
var n = 5 ;
var s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; document . write ( " " + " " + n + " " + s . toFixed ( 6 ) + " " + peri . toFixed ( 6 ) ) ;
function rhombusarea ( l , b ) {
if ( l < 0 b < 0 ) return - 1 ;
return ( l * b ) / 2 ; }
var l = 16 , b = 6 ; document . write ( rhombusarea ( l , b ) ) ;
function FindPoint ( x1 , y1 , x2 , y2 , x , y ) { if ( x > x1 && x < x2 && y > y1 && y < y2 ) return true ; return false ; }
let x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
let x = 1 , y = 5 ;
if ( FindPoint ( x1 , y1 , x2 , y2 , x , y ) ) document . write ( " " ) ; else document . write ( " " ) ;
function shortest_distance ( x1 , y1 , z1 , a , b , c , d ) { d = Math . abs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; let e = Math . sqrt ( a * a + b * b + c * c ) ; document . write ( " " + ( d / e ) ) ; return ; }
let x1 = 4 ; let y1 = - 4 ; let z1 = 3 ; let a = 2 ; let b = - 2 ; let c = 5 ; let d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ;
function findVolume ( l , b , h ) {
let volume = ( l * b * h ) / 2 ; return volume ; }
let l = 18 , b = 12 , h = 9 ;
document . write ( " " + findVolume ( l , b , h ) ) ;
function isRectangle ( a , b , c , d ) {
if ( a == b && a == c && a == d && c == d && b == c && b == d ) return true ; else if ( a == b && c == d ) return true ; else if ( a == d && c == b ) return true ; else if ( a == c && d == b ) return true ; else return false ; }
let a = 1 , b = 2 , c = 3 , d = 4 ; if ( isRectangle ( a , b , c , d ) ) document . write ( " " ) ; else document . write ( " " ) ;
function midpoint ( x1 , x2 , y1 , y2 ) { document . write ( ( x1 + x2 ) / 2 + " " + ( y1 + y2 ) / 2 ) ; }
let x1 = - 1 , y1 = 2 ; let x2 = 3 , y2 = - 6 ; midpoint ( x1 , x2 , y1 , y2 ) ;
function arcLength ( diameter , angle ) { let pi = 22.0 / 7.0 ; let arc ; if ( angle >= 360 ) { document . write ( " " + " " ) ; return 0 ; } else { arc = ( pi * diameter ) * ( angle / 360.0 ) ; return arc ; } }
let diameter = 25.0 ; let angle = 45.0 ; let arc_len = arcLength ( diameter , angle ) ; document . write ( arc_len ) ;
function checkCollision ( a , b , c , x , y , radius ) {
let dist = ( Math . abs ( a * x + b * y + c ) ) / Math . sqrt ( a * a + b * b ) ;
if ( radius == dist ) document . write ( " " ) ; else if ( radius > dist ) document . write ( " " ) ; else document . write ( " " ) ; }
let radius = 5 ; let x = 0 , y = 0 ; let a = 3 , b = 4 , c = 25 ; checkCollision ( a , b , c , x , y , radius ) ;
function polygonArea ( X , Y , n ) {
let area = 0.0 ;
let j = n - 1 ; for ( let i = 0 ; i < n ; i ++ ) { area += ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) ;
return Math . abs ( area / 2.0 ) ; }
let X = [ 0 , 2 , 4 ] ; let Y = [ 1 , 3 , 7 ] ; let n = X . length ; document . write ( polygonArea ( X , Y , n ) ) ;
const chk = ( n ) => {
let v = [ ] ; while ( n != 0 ) { v . push ( n % 2 ) ; n = parseInt ( n / 2 ) ; } for ( let i = 0 ; i < v . length ; i ++ ) { if ( v [ i ] == 1 ) { return Math . pow ( 2 , i ) ; } } return 0 ; }
const sumOfLSB = ( arr , N ) => {
let lsb_arr = [ ] ; for ( let i = 0 ; i < N ; i ++ ) {
lsb_arr . push ( chk ( arr [ i ] ) ) ; }
lsb_arr . sort ( ( a , b ) => a - b ) let ans = 0 ; for ( let i = 0 ; i < N - 1 ; i += 2 ) {
ans += ( lsb_arr [ i + 1 ] ) ; }
document . write ( ans ) ; }
let N = 5 ; let arr = [ 1 , 2 , 3 , 4 , 5 ] ;
sumOfLSB ( arr , N ) ;
function countSubsequences ( arr ) {
let odd = 0 ;
for ( let x = 0 ; x < arr . length ; x ++ ) {
if ( arr [ x ] & 1 ) odd ++ ; }
return ( 1 << odd ) - 1 ; }
let arr = [ 1 , 3 , 3 ] ;
document . write ( countSubsequences ( arr ) ) ;
function getPairsCount ( arr , n ) {
let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
for ( let j = arr [ i ] - ( i % arr [ i ] ) ; j < n ; j += arr [ i ] ) {
if ( i < j && Math . abs ( arr [ i ] - arr [ j ] ) >= Math . min ( arr [ i ] , arr [ j ] ) ) { count ++ ; } } }
return count ; }
let arr = [ 1 , 2 , 2 , 3 ] ; let N = arr . length ; document . write ( getPairsCount ( arr , N ) ) ;
function check ( N ) { var twos = 0 , fives = 0 ;
while ( N % 2 == 0 ) { N /= 2 ; twos ++ ; }
while ( N % 5 == 0 ) { N /= 5 ; fives ++ ; } if ( N == 1 && twos <= fives ) { document . write ( 2 * fives - twos ) ; } else { document . write ( - 1 ) ; } }
var N = 50 ; check ( N ) ;
function rangeSum ( arr , N , L , R ) {
let sum = 0 ;
for ( let i = L - 1 ; i < R ; i ++ ) { sum += arr [ i % N ] ; }
document . write ( sum ) ; }
let arr = [ 5 , 2 , 6 , 9 ] ; let L = 10 , R = 13 ; let N = arr . length rangeSum ( arr , N , L , R ) ;
function rangeSum ( arr , N , L , R ) {
let prefix = new Array ( N + 1 ) ; prefix [ 0 ] = 0 ;
for ( let i = 1 ; i <= N ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] + arr [ i - 1 ] ; }
let leftsum = ( ( L - 1 ) / N ) * prefix [ N ] + prefix [ ( L - 1 ) % N ] ;
let rightsum = ( R / N ) * prefix [ N ] + prefix [ R % N ] ;
document . write ( rightsum - leftsum ) ; }
let arr = [ 5 , 2 , 6 , 9 ] ; let L = 10 , R = 13 ; let N = arr . length ; rangeSum ( arr , N , L , R ) ;
function ExpoFactorial ( N ) {
let res = 1 ; let mod = 1000000007 ;
for ( let i = 2 ; i < N + 1 ; i ++ )
res = Math . pow ( i , res ) % mod ;
return res ; }
let N = 4 ;
document . write ( ( ExpoFactorial ( N ) ) ) ;
function maxSubArraySumRepeated ( arr , N , K ) {
let sum = 0 ;
for ( let i = 0 ; i < N ; i ++ ) sum += arr [ i ] ; let curr = arr [ 0 ] ;
let ans = arr [ 0 ] ;
if ( K == 1 ) {
for ( let i = 1 ; i < N ; i ++ ) { curr = Math . max ( arr [ i ] , curr + arr [ i ] ) ; ans = Math . max ( ans , curr ) ; }
return ans ; }
let V = [ ] ;
for ( let i = 0 ; i < 2 * N ; i ++ ) { V . push ( arr [ i % N ] ) ; }
let maxSuf = V [ 0 ] ;
let maxPref = V [ 2 * N - 1 ] ; curr = V [ 0 ] ; for ( let i = 1 ; i < 2 * N ; i ++ ) { curr += V [ i ] ; maxPref = Math . max ( maxPref , curr ) ; } curr = V [ 2 * N - 1 ] ; for ( let i = 2 * N - 2 ; i >= 0 ; i -- ) { curr += V [ i ] ; maxSuf = Math . max ( maxSuf , curr ) ; } curr = V [ 0 ] ;
for ( let i = 1 ; i < 2 * N ; i ++ ) { curr = Math . max ( V [ i ] , curr + V [ i ] ) ; ans = Math . max ( ans , curr ) ; }
if ( sum > 0 ) { let temp = sum * ( K - 2 ) ; ans = Math . max ( ans , Math . max ( temp + maxPref , temp + maxSuf ) ) ; }
return ans ; }
let arr = [ 10 , 20 , - 30 , - 1 , 40 ] ; let N = arr . length ; let K = 10 ;
document . write ( maxSubArraySumRepeated ( arr , N , K ) ) ;
function countSubarray ( arr , n ) {
let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { for ( let j = i ; j < n ; j ++ ) {
let mxSubarray = 0 ;
let mxOther = 0 ;
for ( let k = i ; k <= j ; k ++ ) { mxSubarray = Math . max ( mxSubarray , arr [ k ] ) ; }
for ( let k = 0 ; k < i ; k ++ ) { mxOther = Math . max ( mxOther , arr [ k ] ) ; } for ( let k = j + 1 ; k < n ; k ++ ) { mxOther = Math . max ( mxOther , arr [ k ] ) ; }
if ( mxSubarray > ( 2 * mxOther ) ) count ++ ; } }
document . write ( count ) ; }
let arr = [ 1 , 6 , 10 , 9 , 7 , 3 ] ; let N = arr . length ; countSubarray ( arr , N ) ;
function countSubarray ( arr , n ) { var count = 0 , L = 0 , R = 0 ;
var mx = Math . max . apply ( null , arr ) ; var i ;
for ( i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] * 2 > mx ) {
L = i ; break ; } } for ( i = n - 1 ; i >= 0 ; i -- ) {
if ( arr [ i ] * 2 > mx ) {
R = i ; break ; } }
document . write ( ( L + 1 ) * ( n - R ) ) ; }
var arr = [ 1 , 6 , 10 , 9 , 7 , 3 ] var N = arr . length ; countSubarray ( arr , N ) ;
function isPrime ( X ) { for ( let i = 2 ; i * i <= X ; i ++ )
return false ; return true ; }
function printPrimes ( A , N ) {
for ( let i = 0 ; i < N ; i ++ ) {
for ( let j = A [ i ] - 1 ; ; j -- ) {
if ( isPrime ( j ) ) { document . write ( j + " " ) ; break ; } }
for ( let j = A [ i ] + 1 ; ; j ++ ) {
if ( isPrime ( j ) ) { document . write ( j + " " ) ; break ; } } document . write ( " " ) ; } }
let A = [ 17 , 28 ] ; let N = A . length ;
printPrimes ( A , N ) ;
function KthSmallest ( A , B , N , K ) { let M = 0 ;
for ( let i = 0 ; i < N ; i ++ ) { M = Math . max ( A [ i ] , M ) ; }
let freq = Array . from ( { length : M + 1 } , ( _ , i ) => 0 ) ;
for ( let i = 0 ; i < N ; i ++ ) { freq [ A [ i ] ] += B [ i ] ; }
let sum = 0 ;
for ( let i = 0 ; i <= M ; i ++ ) {
sum += freq [ i ] ;
if ( sum >= K ) {
return i ; } }
return - 1 ; }
let A = [ 3 , 4 , 5 ] ; let B = [ 2 , 1 , 3 ] ; let N = A . length ; let K = 4 ;
document . write ( KthSmallest ( A , B , N , K ) ) ;
function findbitwiseOR ( a , n ) {
let res = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
let curr_sub_array = a [ i ] ;
res = res | curr_sub_array ; for ( let j = i ; j < n ; j ++ ) {
curr_sub_array = curr_sub_array & a [ j ] ; res = res | curr_sub_array ; } }
document . write ( res ) ; }
let A = [ 1 , 2 , 3 ] ; let N = A . length ; findbitwiseOR ( A , N ) ;
function findbitwiseOR ( a , n ) {
var res = 0 ; var i ;
for ( i = 0 ; i < n ; i ++ ) res = res | a [ i ] ;
document . write ( res ) ; }
var A = [ 1 , 2 , 3 ] ; var N = A . length ; findbitwiseOR ( A , N ) ;
function check ( n ) {
let sumOfDigit = 0 ; let prodOfDigit = 1 ; while ( n > 0 ) {
let rem ; rem = n % 10 ;
sumOfDigit += rem ;
prodOfDigit *= rem ;
n = Math . floor ( n / 10 ) ; }
if ( sumOfDigit > prodOfDigit ) document . write ( " " ) ; else document . write ( " " ) ; }
let N = 1234 ; check ( N ) ;
function evenOddBitwiseXOR ( N ) { document . write ( " " + 0 + " " ) ;
for ( let i = 4 ; i <= N ; i = i + 4 ) { document . write ( i + " " ) ; } document . write ( " " ) ; document . write ( " " + 1 + " " ) ;
for ( let i = 4 ; i <= N ; i = i + 4 ) { document . write ( i - 1 + " " ) ; } if ( N % 4 == 2 ) document . write ( N + 1 ) ; else if ( N % 4 == 3 ) document . write ( N ) ; }
let N = 6 ; evenOddBitwiseXOR ( N ) ;
function findPermutation ( arr ) { let N = arr . length ; let i = N - 2 ;
while ( i >= 0 && arr [ i ] <= arr [ i + 1 ] ) i -- ;
if ( i == - 1 ) { document . write ( " " ) ; return ; } let j = N - 1 ;
while ( j > i && arr [ j ] >= arr [ i ] ) j -- ;
while ( j > i && arr [ j ] == arr [ j - 1 ] ) {
j -- ; }
let temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ;
for ( let it in arr ) { document . write ( arr [ it ] + " " ) ; } }
let arr = [ 1 , 2 , 5 , 3 , 4 , 6 ] ; findPermutation ( arr ) ;
function sieveOfEratosthenes ( N , s ) {
let prime = Array . from ( { length : N + 1 } , ( _ , i ) => 0 ) ;
for ( let i = 2 ; i <= N ; i += 2 ) s [ i ] = 2 ;
for ( let i = 3 ; i <= N ; i += 2 ) {
if ( prime [ i ] == false ) { s [ i ] = i ;
for ( let j = i ; j * i <= N ; j += 2 ) {
if ( ! prime [ i * j ] ) { prime [ i * j ] = true ; s [ i * j ] = i ; } } } } }
function findDifference ( N ) {
let s = Array . from ( { length : N + 1 } , ( _ , i ) => 0 ) ;
sieveOfEratosthenes ( N , s ) ;
let total = 1 , odd = 1 , even = 0 ;
let curr = s [ N ] ;
let cnt = 1 ;
while ( N > 1 ) { N /= s [ N ] ;
if ( curr == s [ N ] ) { cnt ++ ; continue ; }
if ( curr == 2 ) { total = total * ( cnt + 1 ) ; }
else { total = total * ( cnt + 1 ) ; odd = odd * ( cnt + 1 ) ; }
curr = s [ N ] ; cnt = 1 ; }
even = total - odd ;
document . write ( Math . abs ( even - odd ) ) ; }
let N = 12 ; findDifference ( N ) ;
function findMedian ( Mean , Mode ) {
var Median = ( 2 * Mean + Mode ) / 3.0 ;
document . write ( Median ) ; }
var mode = 6 , mean = 3 ; findMedian ( mean , mode ) ;
function vectorMagnitude ( x , y , z ) {
var sum = x * x + y * y + z * z ;
return Math . sqrt ( sum ) ; }
var x = 1 ; var y = 2 ; var z = 3 ; document . write ( vectorMagnitude ( x , y , z ) ) ;
function multiplyByMersenne ( N , M ) {
let x = ( Math . log ( M + 1 ) / Math . log ( 2 ) ) ;
return ( ( N << x ) - N ) ; }
let N = 4 ; let M = 15 ; document . write ( multiplyByMersenne ( N , M ) ) ;
function perfectSquare ( num ) {
let sr = Math . floor ( Math . sqrt ( num ) ) ;
let a = sr * sr ; let b = ( sr + 1 ) * ( sr + 1 ) ;
if ( ( num - a ) < ( b - num ) ) { return a ; } else { return b ; } }
function powerOfTwo ( num ) {
let lg = Math . floor ( Math . log2 ( num ) ) ;
let p = Math . pow ( 2 , lg ) ; return p ; }
function uniqueElement ( arr , N ) { let ans = true ; arr . reverse ( ) ;
let freq = new Map ( ) ;
for ( let i = 0 ; i < N ; i ++ ) { freq [ arr [ i ] ] ++ ; if ( freq . has ( arr [ i ] ) ) { freq . set ( arr [ i ] , freq . get ( arr [ i ] ) + 1 ) } else [ freq . set ( arr [ i ] , 1 ) ] }
for ( let el of freq ) {
if ( el [ 1 ] == 1 ) { ans = false ;
let ps = perfectSquare ( el [ 0 ] ) ;
document . write ( powerOfTwo ( ps ) + ' ' ) ; } }
if ( ans ) document . write ( " " ) ; }
let arr = [ 4 , 11 , 4 , 3 , 4 ] ; let N = arr . length ; uniqueElement ( arr , N ) ;
function partitionArray ( a , n ) {
var min = Array ( n ) . fill ( 0 ) ;
var mini = Number . MAX_VALUE ;
for ( i = n - 1 ; i >= 0 ; i -- ) {
mini = Math . min ( mini , a [ i ] ) ;
min [ i ] = mini ; }
var maxi = Number . MIN_VALUE ;
var ind = - 1 ; for ( i = 0 ; i < n - 1 ; i ++ ) {
maxi = Math . max ( maxi , a [ i ] ) ;
if ( maxi < min [ i + 1 ] ) {
ind = i ;
break ; } }
if ( ind != - 1 ) {
for ( i = 0 ; i <= ind ; i ++ ) document . write ( a [ i ] + " " ) ; document . write ( " " ) ;
for ( i = ind + 1 ; i < n ; i ++ ) document . write ( a [ i ] + " " ) ; }
else document . write ( " " ) ; }
var arr = [ 5 , 3 , 2 , 7 , 9 ] ; var N = arr . length ; partitionArray ( arr , N ) ;
function countPrimeFactors ( n ) { var count = 0 ;
while ( n % 2 == 0 ) { n = parseInt ( n / 2 ) ; count ++ ; }
for ( i = 3 ; i <= parseInt ( Math . sqrt ( n ) ) ; i = i + 2 ) {
while ( n % i == 0 ) { n = parseInt ( n / i ) ; count ++ ; } }
if ( n > 2 ) count ++ ; return ( count ) ; }
function findSum ( n ) {
var sum = 0 ; for ( i = 1 , num = 2 ; i <= n ; num ++ ) {
if ( countPrimeFactors ( num ) == 2 ) { sum += num ;
i ++ ; } } return sum ; }
function check ( n , k ) {
var s = findSum ( k - 1 ) ;
if ( s >= n ) document . write ( " " ) ;
else document . write ( " " ) ; }
var n = 100 , k = 6 ; check ( n , k ) ;
function gcd ( a , b ) {
while ( b > 0 ) { let rem = a % b ; a = b ; b = rem ; }
return a ; }
function countNumberOfWays ( n ) {
if ( n == 1 ) return - 1 ;
let g = 0 ; let power = 0 ;
while ( n % 2 == 0 ) { power ++ ; n /= 2 ; } g = gcd ( g , power ) ;
for ( let i = 3 ; i <= Math . sqrt ( n ) ; i += 2 ) { power = 0 ;
while ( n % i == 0 ) { power ++ ; n /= i ; } g = gcd ( g , power ) ; }
if ( n > 2 ) g = gcd ( g , 1 ) ;
let ways = 1 ;
power = 0 ; while ( g % 2 == 0 ) { g /= 2 ; power ++ ; }
ways *= ( power + 1 ) ;
for ( let i = 3 ; i <= Math . sqrt ( g ) ; i += 2 ) { power = 0 ;
while ( g % i == 0 ) { power ++ ; g /= i ; }
ways *= ( power + 1 ) ; }
if ( g > 2 ) ways *= 2 ;
return ways ; }
let N = 64 ; document . write ( countNumberOfWays ( N ) ) ;
function powOfPositive ( n ) {
let pos = Math . floor ( Math . log2 ( n ) ) ; return Math . pow ( 2 , pos ) ; }
function powOfNegative ( n ) {
let pos = Math . ceil ( Math . log2 ( n ) ) ; return ( - 1 * Math . pow ( 2 , pos ) ) ; }
function highestPowerOf2 ( n ) {
if ( n > 0 ) { document . write ( powOfPositive ( n ) ) ; } else {
n = - n ; document . write ( powOfNegative ( n ) ) ; } }
let n = - 24 ; highestPowerOf2 ( n ) ;
function noOfCards ( n ) { return parseInt ( n * ( 3 * n + 1 ) / 2 ) ; }
var n = 3 ; document . write ( noOfCards ( n ) ) ;
function smallestPoss ( s , n ) {
var ans = " " ;
var arr = Array ( 10 ) . fill ( 0 ) ;
for ( var i = 0 ; i < n ; i ++ ) { arr [ s [ i ] . charCodeAt ( 0 ) - 48 ] ++ ; }
for ( var i = 0 ; i < 10 ; i ++ ) { for ( var j = 0 ; j < arr [ i ] ; j ++ ) ans = ans + i . toString ( ) ; }
return ans ; }
var N = 15 ; var K = " " ; document . write ( smallestPoss ( K , N ) ) ;
function Count_subarray ( arr , n ) { var subarray_sum , remaining_sum , count = 0 ; var i , j , k , l ;
for ( i = 0 ; i < n ; i ++ ) {
for ( j = i ; j < n ; j ++ ) {
subarray_sum = 0 ; remaining_sum = 0 ;
for ( k = i ; k <= j ; k ++ ) { subarray_sum += arr [ k ] ; }
for ( l = 0 ; l < i ; l ++ ) { remaining_sum += arr [ l ] ; } for ( l = j + 1 ; l < n ; l ++ ) { remaining_sum += arr [ l ] ; }
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
var arr = [ 10 , 9 , 12 , 6 ] ; var n = arr . length ; document . write ( Count_subarray ( arr , n ) ) ;
function Count_subarray ( arr , n ) { var total_sum = 0 , subarray_sum , remaining_sum , count = 0 ;
for ( i = 0 ; i < n ; i ++ ) { total_sum += arr [ i ] ; }
for ( i = 0 ; i < n ; i ++ ) {
subarray_sum = 0 ;
for ( j = i ; j < n ; j ++ ) {
subarray_sum += arr [ j ] ; remaining_sum = total_sum - subarray_sum ;
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
var arr = [ 10 , 9 , 12 , 6 ] ; var n = arr . length ; document . write ( Count_subarray ( arr , n ) ) ;
function maxXOR ( arr , n ) {
let xorArr = 0 ; for ( let i = 0 ; i < n ; i ++ ) xorArr ^= arr [ i ] ;
let ans = 0 ;
for ( let i = 0 ; i < n ; i ++ ) ans = Math . max ( ans , ( xorArr ^ arr [ i ] ) ) ;
return ans ; }
let arr = [ 1 , 1 , 3 ] ; let n = arr . length ; document . write ( maxXOR ( arr , n ) ) ;
function digitDividesK ( num , k ) { while ( num ) {
let d = num % 10 ;
if ( d != 0 && k % d == 0 ) return true ;
num = parseInt ( num / 10 ) ; }
return false ; }
function findCount ( l , r , k ) {
let count = 0 ;
for ( let i = l ; i <= r ; i ++ ) {
if ( digitDividesK ( i , k ) ) count ++ ; } return count ; }
let l = 20 , r = 35 ; let k = 45 ; document . write ( findCount ( l , r , k ) ) ;
function isFactorial ( n ) { for ( var i = 1 ; ; i ++ ) { if ( n % i == 0 ) { n = parseInt ( n / i ) ; } else { break ; } } if ( n == 1 ) { return true ; } else { return false ; } }
var n = 24 ; var ans = isFactorial ( n ) ; if ( ans == 1 ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function lcm ( a , b ) { let GCD = __gcd ( a , b ) ; return Math . floor ( ( a * b ) / GCD ) ; } function __gcd ( a , b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
function MinLCM ( a , n ) {
let Prefix = new Array ( n + 2 ) ; let Suffix = new Array ( n + 2 ) ;
Prefix [ 1 ] = a [ 0 ] ; for ( let i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = lcm ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( let i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = lcm ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
let ans = Math . min ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( let i = 2 ; i < n ; i += 1 ) { ans = Math . min ( ans , lcm ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; }
let a = [ 5 , 15 , 9 , 36 ] ; let n = a . length ; document . write ( MinLCM ( a , n ) ) ;
function count ( n ) { return parseInt ( n * ( 3 * n - 1 ) / 2 ) ; }
var n = 3 ; document . write ( count ( n ) ) ;
function findMinValue ( arr , n ) {
let sum = 0 ; for ( let i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
return ( parseInt ( sum / n ) + 1 ) ; }
let arr = [ 4 , 2 , 1 , 10 , 6 ] ; let n = arr . length ; document . write ( findMinValue ( arr , n ) ) ;
const MOD = 1000000007 ;
function modFact ( n , m ) { let result = 1 ; for ( let i = 1 ; i <= m ; i ++ ) result = ( result * i ) % MOD ; return result ; }
let n = 3 , m = 2 ; document . write ( modFact ( n , m ) ) ;
const mod = 1000000000 + 7 ;
function power ( p ) { let res = 1 ; for ( let i = 1 ; i <= p ; ++ i ) { res *= 2 ; res %= mod ; } return res % mod ; }
function subset_square_sum ( A ) { let n = A . length ; let ans = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { ans += ( A [ i ] * A [ i ] ) % mod ; ans %= mod ; } return ( ans * power ( n - 1 ) ) % mod ; }
let A = [ 3 , 7 ] ; document . write ( subset_square_sum ( A ) ) ;
var N = 100050 ; var lpf = Array ( N ) . fill ( 0 ) ; var mobius = Array ( N ) . fill ( 0 ) ;
function least_prime_factor ( ) { for ( i = 2 ; i < N ; i ++ )
if ( lpf [ i ] == 0 ) for ( j = i ; j < N ; j += i )
if ( lpf [ j ] == 0 ) lpf [ j ] = i ; }
function Mobius ( ) { for ( i = 1 ; i < N ; i ++ ) {
if ( i == 1 ) mobius [ i ] = 1 ; else {
if ( lpf [ i / lpf [ i ] ] == lpf [ i ] ) mobius [ i ] = 0 ;
else mobius [ i ] = - 1 * mobius [ i / lpf [ i ] ] ; } } }
function gcd_pairs ( a , n ) {
var maxi = 0 ;
var fre = Array ( n + 1 ) . fill ( 0 ) ;
for ( i = 0 ; i < n ; i ++ ) { fre [ a [ i ] ] ++ ; maxi = Math . max ( a [ i ] , maxi ) ; } least_prime_factor ( ) ; Mobius ( ) ;
var ans = 0 ;
for ( i = 1 ; i <= maxi ; i ++ ) { if ( mobius [ i ] == 0 ) continue ; var temp = 0 ; for ( j = i ; j <= maxi ; j += i ) temp = parseInt ( temp + fre [ j ] ) ; ans += parseInt ( temp * ( temp - 1 ) / 2 * mobius [ i ] ) ; }
return ans ; }
var a = [ 1 , 2 , 3 , 4 , 5 , 6 ] ; var n = a . length ;
document . write ( gcd_pairs ( a , n ) ) ;
function compareVal ( x , y ) {
let a = y * Math . log ( x ) ; let b = x * Math . log ( y ) ;
if ( a > b ) document . write ( x + " " + y + " " + y + " " + x ) ; else if ( a < b ) document . write ( x + " " + y + " " + y + " " + x ) ; else if ( a == b ) document . write ( x + " " + y + " " + y + " " + x ) ; }
let x = 4 , y = 5 ; compareVal ( x , y ) ;
function ZigZag ( n ) {
var fact = Array ( n + 1 ) . fill ( 0 ) ; var zig = Array ( n + 1 ) . fill ( 0 ) ;
fact [ 0 ] = 1 ; for ( var i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
zig [ 0 ] = 1 ; zig [ 1 ] = 1 ; document . write ( " " ) ;
document . write ( zig [ 0 ] + " " + zig [ 1 ] + " " ) ;
for ( var i = 2 ; i < n ; i ++ ) { var sum = 0 ; for ( var k = 0 ; k <= i - 1 ; k ++ ) {
sum += parseInt ( fact [ i - 1 ] / ( fact [ i - 1 - k ] * fact [ k ] ) ) * zig [ k ] * zig [ i - 1 - k ] ; }
zig [ i ] = parseInt ( sum / 2 ) ;
document . write ( parseInt ( sum / 2 ) + " " ) ; } }
var n = 10 ;
ZigZag ( n ) ;
function find_count ( ele ) {
let count = 0 ; for ( let i = 0 ; i < ele . length ; i ++ ) {
let p = [ ] ;
let c = 0 ;
for ( let j = ele . length - 1 ; j >= ( ele . length - 1 - i ) && j >= 0 ; j -- ) p . push ( ele [ j ] ) ; let j = ele . length - 1 , k = 0 ;
while ( j >= 0 ) {
if ( ele [ j ] != p [ k ] ) break ; j -- ; k ++ ;
if ( k == p . length ) { c ++ ; k = 0 ; } } count = Math . max ( count , c ) ; }
return count ; }
function solve ( n ) {
let count = 1 ;
let ele = [ ] ;
for ( let i = 0 ; i < n ; i ++ ) { document . write ( count + " " ) ;
ele . push ( count ) ;
count = find_count ( ele ) ; } }
let n = 10 ; solve ( n ) ;
var store = new Map ( ) ;
function Wedderburn ( n ) {
if ( n <= 2 ) return store [ n ] ;
else if ( n % 2 == 0 ) {
var x = parseInt ( n / 2 ) , ans = 0 ;
for ( var i = 1 ; i < x ; i ++ ) { ans += store [ i ] * store [ n - i ] ; }
ans += ( store [ x ] * ( store [ x ] + 1 ) ) / 2 ;
store [ n ] = ans ;
return ans ; } else {
var x = ( n + 1 ) / 2 , ans = 0 ;
for ( var i = 1 ; i < x ; i ++ ) { ans += store [ i ] * store [ n - i ] ; }
store [ n ] = ans ;
return ans ; } }
function Wedderburn_Etherington ( n ) {
store [ 0 ] = 0 ; store [ 1 ] = 1 ; store [ 2 ] = 1 ;
for ( var i = 0 ; i < n ; i ++ ) { document . write ( Wedderburn ( i ) ) ; if ( i != n - 1 ) document . write ( " " ) ; } }
var n = 10 ;
Wedderburn_Etherington ( n ) ;
function Max_sum ( a , n ) {
let pos = 0 , neg = 0 ; for ( let i = 0 ; i < n ; i ++ ) {
if ( a [ i ] > 0 ) pos = 1 ;
else if ( a [ i ] < 0 ) neg = 1 ;
if ( pos == 1 && neg == 1 ) break ; }
let sum = 0 ; if ( pos == 1 && neg == 1 ) { for ( let i = 0 ; i < n ; i ++ ) sum += Math . abs ( a [ i ] ) ; } else if ( pos == 1 ) {
let mini = a [ 0 ] ; sum = a [ 0 ] ; for ( let i = 1 ; i < n ; i ++ ) { mini = Math . min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; } else if ( neg == 1 ) {
for ( let i = 0 ; i < n ; i ++ ) a [ i ] = Math . abs ( a [ i ] ) ;
let mini = a [ 0 ] ; sum = a [ 0 ] ; for ( let i = 1 ; i < n ; i ++ ) { mini = Math . min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; }
return sum ; }
let a = [ 1 , 3 , 5 , - 2 , - 6 ] ; let n = a . length ;
document . write ( Max_sum ( a , n ) ) ;
function decimalToBinary ( n ) {
if ( n == 0 ) { document . write ( " " ) ; return ; }
decimalToBinary ( parseInt ( n / 2 ) ) ; document . write ( n % 2 ) ; }
var n = 13 ; decimalToBinary ( n ) ;
function MinimumValue ( x , y ) {
if ( x > y ) { var temp = x ; x = y ; y = temp ; }
var a = 1 ; var b = x - 1 ; var c = y - b ; document . write ( a + " " + b + " " + c ) ; }
var x = 123 , y = 13 ;
MinimumValue ( x , y ) ;
function canConvert ( a , b ) { while ( b > a ) {
if ( b % 10 == 1 ) { b = parseInt ( b / 10 ) ; continue ; }
if ( b % 2 == 0 ) { b = parseInt ( b / 2 ) ; continue ; }
return false ; }
if ( b == a ) return true ; return false ; }
let A = 2 , B = 82 ; if ( canConvert ( A , B ) ) document . write ( " " ) ; else document . write ( " " ) ;
function count ( N ) { var a = 0 ; a = ( N * ( N + 1 ) ) / 2 ; return a ; }
var n = 4 ; document . write ( count ( n ) ) ;
function numberOfDays ( a , b , n ) { var Days = b * ( n + a ) / ( a + b ) ; return Days ; }
var a = 10 , b = 20 , n = 5 ; document . write ( numberOfDays ( a , b , n ) ) ;
function getAverage ( x , y ) {
var avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; return avg ; }
var x = 10 , y = 9 ; document . write ( getAverage ( x , y ) ) ;
function smallestIndex ( a , n ) {
let right1 = 0 , right0 = 0 ;
let i ; for ( i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == 1 ) right1 = i ;
else right0 = i ; }
return Math . min ( right1 , right0 ) ; }
var a = [ 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 ] ; let n = a . length ; document . write ( smallestIndex ( a , n ) ) ;
function countSquares ( r , c , m ) {
let squares = 0 ;
for ( let i = 1 ; i <= 8 ; i ++ ) { for ( let j = 1 ; j <= 8 ; j ++ ) {
if ( Math . max ( Math . abs ( i - r ) , Math . abs ( j - c ) ) <= m ) squares ++ ; } }
return squares ; }
let r = 4 , c = 4 , m = 1 ; document . write ( countSquares ( r , c , m ) ) ;
function countQuadruples ( a , n ) {
let mp = new Map ( ) ;
for ( let i = 0 ; i < n ; i ++ ) if ( mp . has ( a [ i ] ) ) { mp . set ( a [ i ] , mp . get ( a [ i ] ) + 1 ) ; } else { mp . set ( a [ i ] , 1 ) ; } let count = 0 ;
for ( let j = 0 ; j < n ; j ++ ) { for ( let k = 0 ; k < n ; k ++ ) {
if ( j == k ) continue ;
mp . set ( a [ j ] , mp . get ( a [ j ] ) - 1 ) ; mp . set ( a [ k ] , mp . get ( a [ k ] ) - 1 ) ;
let first = a [ j ] - ( a [ k ] - a [ j ] ) ;
let fourth = ( a [ k ] * a [ k ] ) / a [ j ] ;
if ( ( a [ k ] * a [ k ] ) % a [ j ] == 0 ) {
if ( a [ j ] != a [ k ] ) { if ( mp . has ( first ) && mp . has ( fourth ) ) count += mp . get ( first ) * mp . get ( fourth ) ; }
else if ( mp . has ( first ) && mp . has ( fourth ) ) count += mp . get ( first ) * ( mp . get ( fourth ) - 1 ) ; }
if ( mp . has ( a [ j ] ) ) { mp . set ( a [ j ] , mp . get ( a [ j ] ) + 1 ) ; } else { mp . set ( a [ j ] , 1 ) ; } if ( mp . has ( a [ k ] ) ) { mp . set ( a [ k ] , mp . get ( a [ k ] ) + 1 ) ; } else { mp . set ( a [ k ] , 1 ) ; } } } return count ; }
let a = [ 2 , 6 , 4 , 9 , 2 ] ; let n = a . length ; document . write ( countQuadruples ( a , n ) ) ;
function countNumbers ( L , R , K ) { if ( K == 9 ) { K = 0 ; }
var totalnumbers = R - L + 1 ;
var factor9 = totalnumbers / 9 ;
var rem = totalnumbers % 9 ;
var ans = factor9 ;
for ( var i = R ; i > R - rem ; i -- ) { var rem1 = i % 9 ; if ( rem1 == K ) { ans ++ ; } } return ans ; }
var L = 10 ; var R = 22 ; var K = 3 ; document . write ( Math . round ( countNumbers ( L , R , K ) ) ) ;
function EvenSum ( A , index , value ) {
A [ index ] = A [ index ] + value ;
var sum = 0 ; for ( var i = 0 ; i < A . length ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; return sum ; }
function BalanceArray ( A , Q ) {
var ANS = [ ] ; var i , sum ; for ( i = 0 ; i < Q . length ; i ++ ) { var index = Q [ i ] [ 0 ] ; var value = Q [ i ] [ 1 ] ;
sum = EvenSum ( A , index , value ) ;
ANS . push ( sum ) ; }
for ( i = 0 ; i < ANS . length ; i ++ ) document . write ( ANS [ i ] + " " ) ; }
var A = [ 1 , 2 , 3 , 4 ] ; var Q = [ [ 0 , 1 ] , [ 1 , - 3 ] , [ 0 , - 4 ] , [ 3 , 2 ] ] ; BalanceArray ( A , Q ) ;
function BalanceArray ( A , Q ) { var ANS = [ ] ; var i , sum = 0 ; for ( i = 0 ; i < A . length ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; for ( i = 0 ; i < Q . length ; i ++ ) { var index = Q [ i ] [ 0 ] ; var value = Q [ i ] [ 1 ] ;
if ( A [ index ] % 2 == 0 ) sum = sum - A [ index ] ; A [ index ] = A [ index ] + value ;
if ( A [ index ] % 2 == 0 ) sum = sum + A [ index ] ;
ANS . push ( sum ) ; }
for ( i = 0 ; i < ANS . length ; i ++ ) document . write ( ANS [ i ] + " " ) ; }
var A = [ 1 , 2 , 3 , 4 ] ; var Q = [ [ 0 , 1 ] , [ 1 , - 3 ] , [ 0 , - 4 ] , [ 3 , 2 ] ] ; BalanceArray ( A , Q ) ;
function Cycles ( N ) { var fact = 1 , result = 0 ; result = N - 1 ;
var i = result ; while ( i > 0 ) { fact = fact * i ; i -- ; } return fact / 2 ; }
var N = 5 ; var Number = Cycles ( N ) ; document . write ( " " + Number ) ;
function digitWell ( n , m , k ) { var cnt = 0 ; while ( n > 0 ) { if ( n % 10 == m ) ++ cnt ; n = Math . floor ( n / 10 ) ; } if ( cnt == k ) return true ; else return false ; }
function findInt ( n , m , k ) { var i = n + 1 ; while ( true ) { if ( digitWell ( i , m , k ) ) return i ; i ++ ; } }
var n = 111 , m = 2 , k = 2 ; document . write ( findInt ( n , m , k ) ) ;
function countOdd ( arr , n ) {
var odd = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] % 2 == 1 ) odd ++ ; } return odd ; }
function countValidPairs ( arr , n ) { var odd = countOdd ( arr , n ) ; return ( odd * ( odd - 1 ) ) / 2 ; }
var arr = [ 1 , 2 , 3 , 4 , 5 ] ; var n = arr . length ; document . write ( countValidPairs ( arr , n ) ) ;
function gcd ( a , b ) { if ( b == 0 ) return a ; else return gcd ( b , a % b ) ; }
function lcmOfArray ( arr , n ) { if ( n < 1 ) return 0 ; let lcm = arr [ 0 ] ;
for ( let i = 1 ; i < n ; i ++ ) lcm = parseInt ( ( lcm * arr [ i ] ) / gcd ( lcm , arr [ i ] ) ) ;
return lcm ; }
function minPerfectCube ( arr , n ) { let minPerfectCube ;
let lcm = lcmOfArray ( arr , n ) ; minPerfectCube = lcm ; let cnt = 0 ; while ( lcm > 1 && lcm % 2 == 0 ) { cnt ++ ; lcm = parseInt ( lcm / 2 ) ; }
if ( cnt % 3 == 2 ) minPerfectCube *= 2 ; else if ( cnt % 3 == 1 ) minPerfectCube *= 4 ; let i = 3 ;
while ( lcm > 1 ) { cnt = 0 ; while ( lcm % i == 0 ) { cnt ++ ; lcm = parseInt ( lcm / i ) ; } if ( cnt % 3 == 1 ) minPerfectCube *= i * i ; else if ( cnt % 3 == 2 ) minPerfectCube *= i ; i += 2 ; }
return minPerfectCube ; }
let arr = [ 10 , 125 , 14 , 42 , 100 ] ; let n = arr . length ; document . write ( minPerfectCube ( arr , n ) ) ;
function isPrime ( n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( let i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
function isStrongPrime ( n ) {
if ( ! isPrime ( n ) n == 2 ) return false ;
let previous_prime = n - 1 ; let next_prime = n + 1 ;
while ( ! isPrime ( next_prime ) ) next_prime ++ ;
while ( ! isPrime ( previous_prime ) ) previous_prime -- ;
let mean = parseInt ( ( previous_prime + next_prime ) / 2 ) ;
if ( n > mean ) return true ; else return false ; }
let n = 11 ; if ( isStrongPrime ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function countDigitsToBeRemoved ( N , K ) {
var s = N . toString ( ) ;
var res = 0 ;
var f_zero = 0 ; for ( var i = s . length - 1 ; i >= 0 ; i -- ) { if ( K === 0 ) return res ; if ( s [ i ] === " " ) {
f_zero = 1 ; K -- ; } else res ++ ; }
if ( K === 0 ) return res ; else if ( f_zero === 1 ) return s . length - 1 ; return - 1 ; }
var N = 10904025 ; var K = 2 ; document . write ( countDigitsToBeRemoved ( N , K ) + " " ) ; N = 1000 ; K = 5 ; document . write ( countDigitsToBeRemoved ( N , K ) + " " ) ; N = 23985 ; K = 2 ; document . write ( countDigitsToBeRemoved ( N , K ) + " " ) ;
function getSum ( a , n ) {
let sum = 0 ; for ( let i = 1 ; i <= n ; ++ i ) {
sum += ( i / Math . pow ( a , i ) ) ; } return sum ; }
let a = 3 , n = 3 ;
document . write ( getSum ( a , n ) . toFixed ( 7 ) ) ;
function largestPrimeFactor ( n ) {
var max = - 1 ;
while ( n % 2 == 0 ) { max = 2 ;
for ( var i = 3 ; i <= Math . sqrt ( n ) ; i += 2 ) { while ( n % i == 0 ) { max = i ; n = n / i ; } }
if ( n > 2 ) max = n ; return max ; }
function checkUnusual ( n ) {
var factor = largestPrimeFactor ( n ) ;
if ( factor > Math . sqrt ( n ) ) { return true ; } else { return false ; } }
var n = 14 ; if ( checkUnusual ( n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function isHalfReducible ( arr , n , m ) { var frequencyHash = Array ( m + 1 ) . fill ( 0 ) ; var i ; for ( i = 0 ; i < n ; i ++ ) { frequencyHash [ arr [ i ] % ( m + 1 ) ] ++ ; } for ( i = 0 ; i <= m ; i ++ ) { if ( frequencyHash [ i ] >= n / 2 ) break ; } if ( i <= m ) document . write ( " " ) ; else document . write ( " " ) ; }
var arr = [ 8 , 16 , 32 , 3 , 12 ] ; var n = arr . length ; var m = 7 ; isHalfReducible ( arr , n , m ) ;
var arr = [ ] ;
function generateDivisors ( n ) {
for ( var i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) arr . push ( i ) ;
{ arr . push ( i ) ; arr . push ( n / i ) ; } } } }
function harmonicMean ( n ) { generateDivisors ( n ) ;
var sum = 0.0 ; var len = arr . length ;
for ( var i = 0 ; i < len ; i ++ ) sum = sum + ( n / arr [ i ] ) ; sum = ( sum / n ) ;
return ( arr . length / sum ) ; }
function isOreNumber ( n ) {
var mean = harmonicMean ( n ) ;
if ( mean - parseInt ( mean ) == 0 ) return true ; else return false ; }
var n = 28 ; if ( isOreNumber ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
let MAX = 10000 ; let s = new Set ( ) ;
function SieveOfEratosthenes ( ) {
let prime = new Array ( MAX ) ; for ( let i = 0 ; i < prime . length ; i ++ ) { prime [ i ] = true ; } prime [ 0 ] = false ; prime [ 1 ] = false ; for ( let p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( let i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
let product = 1 ; for ( let p = 2 ; p < MAX ; p ++ ) { if ( prime [ p ] ) {
product = product * p ;
s . add ( product + 1 ) ; } } }
function isEuclid ( n ) {
if ( s . has ( n ) ) return true ; else return false ; }
SieveOfEratosthenes ( ) ;
let n = 31 ;
if ( isEuclid ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
n = 42 ;
if ( isEuclid ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function isPrime ( n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( var i = 5 ; i * i <= n ; i = i + 6 ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) { return false ; } } return true ; }
function isPowerOfTwo ( n ) { return ( n != 0 ) && ( ( n & ( n - 1 ) ) == 0 ) ; }
var n = 43 ;
if ( isPrime ( n ) && ( isPowerOfTwo ( n * 3 - 1 ) ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function area ( a ) {
if ( a < 0 ) return - 1 ;
var area = Math . pow ( ( a * Math . sqrt ( 3 ) ) / ( Math . sqrt ( 2 ) ) , 2 ) ; return area ; }
var a = 5 ; document . write ( area ( a ) . toFixed ( 5 ) ) ;
function nthTerm ( n ) { return 3 * Math . pow ( n , 2 ) - 4 * n + 2 ; }
let N = 4 ; document . write ( nthTerm ( N ) ) ;
function calculateSum ( n ) { return n * ( n + 1 ) / 2 + Math . pow ( ( n * ( n + 1 ) / 2 ) , 2 ) ; }
let n = 3 ;
document . write ( " " + calculateSum ( n ) ) ;
function arePermutations ( a , b , n , m ) { let sum1 = 0 , sum2 = 0 , mul1 = 1 , mul2 = 1 ;
for ( let i = 0 ; i < n ; i ++ ) { sum1 += a [ i ] ; mul1 *= a [ i ] ; }
for ( let i = 0 ; i < m ; i ++ ) { sum2 += b [ i ] ; mul2 *= b [ i ] ; }
return ( ( sum1 == sum2 ) && ( mul1 == mul2 ) ) ; }
let a = [ 1 , 3 , 2 ] ; let b = [ 3 , 1 , 2 ] ; let n = a . length ; let m = b . length ; if ( arePermutations ( a , b , n , m ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function Race ( B , C ) { var result = 0 ;
result = ( ( C * 100 ) / B ) ; return 100 - result ; }
var B = 10 , C = 28 ;
B = 100 - B ; C = 100 - C ; document . write ( Race ( B , C ) + " " ) ;
function Time ( arr , n , Emptypipe ) { var fill = 0 ; for ( var i = 0 ; i < n ; i ++ ) fill += 1 / arr [ i ] ; fill = fill - ( 1 / Emptypipe ) ; return 1 / fill ; }
var arr = [ 12 , 14 ] ; var Emptypipe = 30 ; var n = arr . length ; document . write ( Math . floor ( Time ( arr , n , Emptypipe ) ) + " " ) ;
function check ( n ) { let sum = 0 ;
while ( n != 0 ) { sum += n % 10 ; n = Math . floor ( n / 10 ) ; }
if ( sum % 7 == 0 ) return 1 ; else return 0 ; }
let n = 25 ; ( check ( n ) == 1 ) ? document . write ( " " ) : document . write ( " " ) ;
let N = 1000005 ;
function isPrime ( n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( let i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
function SumOfPrimeDivisors ( n ) { let sum = 0 ; for ( let i = 1 ; i <= n ; i ++ ) { if ( n % i == 0 ) { if ( isPrime ( i ) ) sum += i ; } } return sum ; }
let n = 60 ; document . write ( " " + SumOfPrimeDivisors ( n ) ) ;
function Sum ( N ) { let SumOfPrimeDivisors = new Array ( N + 1 ) ; for ( let i = 0 ; i < SumOfPrimeDivisors . length ; i ++ ) { SumOfPrimeDivisors [ i ] = 0 ; } for ( let i = 2 ; i <= N ; ++ i ) {
if ( SumOfPrimeDivisors [ i ] == 0 ) {
for ( let j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
let N = 60 ; document . write ( " " + " " + Sum ( N ) + " " ) ;
function find_Square_369 ( num ) { let a , b , c , d ;
if ( num [ 0 ] == ' ' ) { a = ' ' ; b = ' ' ; c = ' ' ; d = ' ' ; }
else if ( num [ 0 ] == ' ' ) { a = ' ' ; b = ' ' ; c = ' ' ; d = ' ' ; }
else { a = ' ' ; b = ' ' ; c = ' ' ; d = ' ' ; }
let result = " " ;
let size = num . length ;
for ( let i = 1 ; i < size ; i ++ ) result += a ;
result += b ;
for ( let i = 1 ; i < size ; i ++ ) result += c ;
result += d ;
return result ; }
let num_3 , num_6 , num_9 ; num_3 = " " ; num_6 = " " ; num_9 = " " ; let result = " " ;
result = find_Square_369 ( num_3 ) ; document . write ( " " + num_3 + " " + result + " " ) ;
result = find_Square_369 ( num_6 ) ; document . write ( " " + num_9 + " " + result + " " ) ;
result = find_Square_369 ( num_9 ) ; document . write ( " " + num_9 + " " + result + " " ) ;
var ans = 1 ; var mod = 1000000007 * 120 ; for ( var i = 0 ; i < 5 ; i ++ ) ans = ( ans * ( 55555 - i ) ) % mod ; ans = ans / 120 ; document . write ( " " + " " + ans ) ;
function fact ( n ) { if ( n == 0 n == 1 ) return 1 ; let ans = 1 ; for ( let i = 1 ; i <= n ; i ++ ) ans = ans * i ; return ans ; }
function nCr ( n , r ) { let Nr = n , Dr = 1 , ans = 1 ; for ( let i = 1 ; i <= r ; i ++ ) { ans = parseInt ( ( ans * Nr ) / ( Dr ) , 10 ) ; Nr -- ; Dr ++ ; } return ans ; }
function solve ( n ) { let N = 2 * n - 2 ; let R = n - 1 ; return nCr ( N , R ) * fact ( n - 1 ) ; }
let n = 6 ; document . write ( solve ( n ) ) ;
function pythagoreanTriplet ( n ) {
for ( let i = 1 ; i <= n / 3 ; i ++ ) {
for ( let j = i + 1 ; j <= n / 2 ; j ++ ) { let k = n - i - j ; if ( i * i + j * j == k * k ) { document . write ( i + " " + j + " " + k ) ; return ; } } } document . write ( " " ) ; }
let n = 12 ; pythagoreanTriplet ( n ) ;
function factorial ( n ) { let f = 1 ; for ( let i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
function series ( A , X , n ) {
let nFact = factorial ( n ) ;
for ( let i = 0 ; i < n + 1 ; i ++ ) {
let niFact = factorial ( n - i ) ; let iFact = factorial ( i ) ;
let aPow = Math . pow ( A , n - i ) ; let xPow = Math . pow ( X , i ) ;
document . write ( ( nFact * aPow * xPow ) / ( niFact * iFact ) + " " ) ; } }
let A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ;
function seiresSum ( n , a ) { let res = 0 , i ; for ( i = 0 ; i < 2 * n ; i ++ ) { if ( i % 2 == 0 ) res += a [ i ] * a [ i ] ; else res -= a [ i ] * a [ i ] ; } return res ; }
let n = 2 ; let a = [ 1 , 2 , 3 , 4 ] ; document . write ( seiresSum ( n , a ) ) ;
function power ( n , r ) {
let count = 0 ; for ( let i = r ; ( n / i ) >= 1 ; i = i * r ) count += n / i ; return count ; }
let n = 6 , r = 3 ; document . write ( power ( n , r ) ) ;
function avg_of_odd_num ( n ) {
let sum = 0 ; for ( let i = 0 ; i < n ; i ++ ) sum += ( 2 * i + 1 ) ;
return sum / n ; }
let n = 20 ; document . write ( avg_of_odd_num ( n ) ) ;
function avg_of_odd_num ( n ) { return n ; }
var n = 8 ; document . write ( avg_of_odd_num ( n ) ) ;
function fib ( f , N ) {
f [ 1 ] = 1 ; f [ 2 ] = 1 ; for ( var i = 3 ; i <= N ; i ++ )
f [ i ] = f [ i - 1 ] + f [ i - 2 ] ; } function fiboTriangle ( n ) {
var N = ( n * ( n + 1 ) ) / 2 ; var f = [ ... Array ( N + 1 ) ] ; fib ( f , N ) ;
var fiboNum = 1 ;
for ( var i = 1 ; i <= n ; i ++ ) {
for ( var j = 1 ; j <= i ; j ++ ) document . write ( f [ fiboNum ++ ] + " " ) ; document . write ( " " ) ; } }
var n = 5 ; fiboTriangle ( n ) ;
function averageOdd ( n ) { if ( n % 2 == 0 ) { document . write ( " " ) ; return - 1 ; } let sum = 0 , count = 0 ; while ( n >= 1 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
let n = 15 ; document . write ( averageOdd ( n ) ) ;
function averageOdd ( n ) { if ( n % 2 == 0 ) { document . write ( " " ) ; return - 1 ; } return ( n + 1 ) / 2 ; }
let n = 15 ; document . write ( averageOdd ( n ) ) ;
class Rational { constructor ( nume , deno ) { this . nume = nume ; this . deno = deno ; } }
function lcm ( a , b ) { return ( a * b ) / ( __gcd ( a , b ) ) ; }
function maxRational ( first , sec ) {
let k = lcm ( first . deno , sec . deno ) ;
let nume1 = first . nume ; let nume2 = sec . nume ; nume1 *= k / ( first . deno ) ; nume2 *= k / ( sec . deno ) ; return ( nume2 < nume1 ) ? first : sec ; } function __gcd ( a , b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
let first = new Rational ( 3 , 2 ) ; let sec = new Rational ( 3 , 4 ) ; let res = maxRational ( first , sec ) ; document . write ( res . nume + " " + res . deno ) ;
function TrinomialValue ( n , k ) {
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ; }
function printTrinomial ( n ) {
for ( let i = 0 ; i < n ; i ++ ) {
for ( let j = - i ; j <= 0 ; j ++ ) document . write ( TrinomialValue ( i , j ) + " " ) ;
for ( let j = 1 ; j <= i ; j ++ ) document . write ( TrinomialValue ( i , j ) + " " ) ; document . write ( " " ) ; } }
let n = 4 ; printTrinomial ( n ) ;
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
function sumOfLargePrimeFactor ( n ) {
let prime = new Array ( n + 1 ) ; let sum = 0 ; let max = n / 2 ; for ( let i = 0 ; i < n + 1 ; i ++ ) prime [ i ] = 0 ; for ( let p = 2 ; p <= max ; p ++ ) {
if ( prime [ p ] == 0 ) {
for ( let i = p * 2 ; i <= n ; i += p ) prime [ i ] = p ; } }
for ( let p = 2 ; p <= n ; p ++ ) {
if ( prime [ p ] ) sum += prime [ p ] ;
else sum += p ; }
return sum ; }
let n = 12 ; document . write ( " " + sumOfLargePrimeFactor ( n ) ) ;
function calculate_sum ( a , N ) {
m = N / a ;
sum = m * ( m + 1 ) / 2 ;
ans = a * sum ; return ans ; }
let a = 7 ; let N = 49 ; document . write ( " " + a + " " + N + " " + calculate_sum ( a , N ) ) ;
function ispowerof2 ( num ) { if ( ( num & ( num - 1 ) ) == 0 ) return 1 ; return 0 ; }
var num = 549755813888 ; document . write ( ispowerof2 ( num ) ) ;
function counDivisors ( X ) {
let count = 0 ;
for ( let i = 1 ; i <= X ; ++ i ) { if ( X % i == 0 ) { count ++ ; } }
return count ; }
function countDivisorsMult ( arr , n ) {
let mul = 1 ; for ( let i = 0 ; i < n ; ++ i ) mul *= arr [ i ] ;
return counDivisors ( mul ) ; }
let arr = [ 2 , 4 , 6 ] ; let n = arr . length ; document . write ( countDivisorsMult ( arr , n ) ) ;
function SieveOfEratosthenes ( largest , prime ) {
var isPrime = Array ( largest + 1 ) . fill ( true ) ; var p , i ; for ( p = 2 ; p * p <= largest ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( i = p * 2 ; i <= largest ; i += p ) isPrime [ i ] = false ; } }
for ( p = 2 ; p <= largest ; p ++ ) if ( isPrime [ p ] ) prime . push ( p ) ; }
function countDivisorsMult ( arr , n ) {
var largest = Math . max . apply ( null , arr ) ; var prime = [ ] ; SieveOfEratosthenes ( largest , prime ) ;
var j ; var mp = new Map ( ) ; for ( i = 0 ; i < n ; i ++ ) { for ( j = 0 ; j < prime . length ; j ++ ) { while ( arr [ i ] > 1 && arr [ i ] % prime [ j ] == 0 ) { arr [ i ] /= prime [ j ] ; if ( mp . has ( prime [ j ] ) ) mp . set ( prime [ j ] , mp . get ( prime [ j ] ) + 1 ) ; else mp . set ( prime [ j ] , 1 ) ; } } if ( arr [ i ] != 1 ) { if ( mp . has ( arr [ i ] ) ) mp . set ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; else mp . set ( arr [ i ] , 1 ) ; } }
var res = 1 ; for ( const [ key , value ] of mp . entries ( ) ) { res *= ( value + 1 ) ; } return res ; }
var arr = [ 2 , 4 , 6 ] ; var n = arr . length ; document . write ( countDivisorsMult ( arr , n ) ) ;
function findPrimeNos ( L , R , M ) {
for ( var i = L ; i <= R ; i ++ ) { if ( M . has ( i ) ) M . set ( i , M . get ( i ) + 1 ) else M . set ( i , 1 ) }
if ( M . has ( 1 ) ) { M . delete ( 1 ) ; }
for ( var i = 2 ; i <= parseInt ( Math . sqrt ( R ) ) ; i ++ ) { var multiple = 2 ; while ( ( i * multiple ) <= R ) {
if ( M . has ( i * multiple ) ) {
M . delete ( i * multiple ) ; }
multiple ++ ; } } return M ; }
function getPrimePairs ( L , R , K ) { var M = new Map ( ) ;
M = findPrimeNos ( L , R , M ) ;
M . forEach ( ( value , key ) => {
if ( M . has ( key + K ) ) { document . write ( " " + key + " " + ( key + K ) + " " ) ; } } ) ; }
var L = 1 , R = 19 ;
var K = 6 ;
getPrimePairs ( L , R , K ) ;
function EnneacontahexagonNum ( n ) { return ( 94 * n * n - 92 * n ) / 2 ; }
let n = 3 ; document . write ( EnneacontahexagonNum ( n ) ) ;
function find_composite_nos ( n ) { document . write ( 9 * n + " " + 8 * n ) ; }
var n = 4 ; find_composite_nos ( n ) ;
function freqPairs ( arr , n ) {
let max = Math . max ( ... arr ) ;
let freq = new Array ( max + 1 ) . fill ( 0 ) ;
let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) freq [ arr [ i ] ] ++ ;
for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 2 * arr [ i ] ; j <= max ; j += arr [ i ] ) {
if ( freq [ j ] >= 1 ) count += freq [ j ] ; }
if ( freq [ arr [ i ] ] > 1 ) { count += freq [ arr [ i ] ] - 1 ; freq [ arr [ i ] ] -- ; } } return count ; }
let arr = [ 3 , 2 , 4 , 2 , 6 ] ; let n = arr . length ; document . write ( freqPairs ( arr , n ) ) ;
function Nth_Term ( n ) { return ( 2 * Math . pow ( n , 3 ) - 3 * Math . pow ( n , 2 ) + n + 6 ) / 6 ; }
let N = 8 ; document . write ( Nth_Term ( N ) ) ;
function prletNthElement ( n ) {
let arr = Array ( n + 1 ) . fill ( 0 ) ; arr [ 1 ] = 3 ; arr [ 2 ] = 5 ; for ( i = 3 ; i <= n ; i ++ ) {
if ( i % 2 != 0 ) arr [ i ] = arr [ i / 2 ] * 10 + 3 ; else arr [ i ] = arr [ ( i / 2 ) - 1 ] * 10 + 5 ; } return arr [ n ] ; }
let n = 6 ; document . write ( prletNthElement ( n ) ) ;
function nthTerm ( N ) {
return parseInt ( N * ( parseInt ( N / 2 ) + ( ( N % 2 ) * 2 ) + N ) ) ; }
let N = 5 ;
document . write ( " " + N + " " + nthTerm ( N ) ) ;
function series ( A , X , n ) {
let term = Math . pow ( A , n ) ; document . write ( term + " " ) ;
for ( let i = 1 ; i <= n ; i ++ ) {
term = term * X * ( n - i + 1 ) / ( i * A ) ; document . write ( term + " " ) ; } }
let A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ;
function Div_by_8 ( n ) { return ( ( ( n >> 3 ) << 3 ) == n ) ; }
var n = 16 ; if ( Div_by_8 ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function averageEven ( n ) { if ( n % 2 != 0 ) { document . write ( " " ) ; return - 1 ; } let sum = 0 , count = 0 ; while ( n >= 2 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
let n = 16 ; document . write ( averageEven ( n ) ) ;
function averageEven ( n ) { if ( n % 2 != 0 ) { document . write ( " " ) ; return - 1 ; } return ( n + 2 ) / 2 ; }
let n = 16 ; document . write ( averageEven ( n ) ) ;
function gcd ( a , b ) {
if ( a == 0 b == 0 ) return 0 ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
function cpFact ( x , y ) { while ( gcd ( x , y ) != 1 ) { x = x / gcd ( x , y ) ; } return x ; }
let x = 15 ; let y = 3 ; document . write ( cpFact ( x , y ) + " " ) ; x = 14 ; y = 28 ; document . write ( cpFact ( x , y ) , " " ) ; x = 7 ; y = 3 ; document . write ( cpFact ( x , y ) ) ;
function counLastDigitK ( low , high , k ) { let count = 0 ; for ( let i = low ; i <= high ; i ++ ) if ( i % 10 == k ) count ++ ; return count ; }
let low = 3 ; let high = 35 ; let k = 3 ; document . write ( counLastDigitK ( low , high , k ) ) ;
function printTaxicab2 ( N ) {
let i = 1 ; count = 0 ; while ( count < N ) { let int_count = 0 ;
for ( let j = 1 ; j <= Math . pow ( i , 1.0 / 3 ) ; j ++ ) for ( let k = j + 1 ; k <= Math . pow ( i , 1.0 / 3 ) ; k ++ ) if ( j * j * j + k * k * k == i ) int_count ++ ;
if ( int_count == 2 ) { count ++ ; document . write ( count + " " + i + " " ) ; } i ++ ; } }
let N = 5 ; printTaxicab2 ( N ) ;
function isComposite ( n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return false ;
if ( n % 2 == 0 n % 3 == 0 ) return true ; for ( let i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return true ; return false ; }
isComposite ( 11 ) ? document . write ( " " + " " ) : document . write ( " " + " " ) ; isComposite ( 15 ) ? document . write ( " " + " " ) : document . write ( " " + " " ) ;
function isPrime ( n ) {
if ( n <= 1 ) return false ;
for ( let i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
function findPrime ( n ) { let num = n + 1 ;
while ( num > 0 ) {
if ( isPrime ( num ) ) return num ;
num = num + 1 ; } return 0 ; }
function minNumber ( arr , n ) { let sum = 0 ;
for ( let i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( isPrime ( sum ) ) return 0 ;
let num = findPrime ( sum ) ;
return num - sum ; }
let arr = [ 2 , 4 , 6 , 8 , 12 ] ; let n = arr . length ; document . write ( minNumber ( arr , n ) ) ;
function fact ( n ) { if ( n == 0 ) return 1 ; return n * fact ( n - 1 ) ; }
function div ( x ) { let ans = 0 ; for ( let i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
function sumFactDiv ( n ) { return div ( fact ( n ) ) ; }
let n = 4 ; document . write ( sumFactDiv ( n ) ) ;
let allPrimes = [ ] ;
function sieve ( n ) {
let prime = new Array ( n + 1 ) ; for ( let i = 0 ; i < n + 1 ; i ++ ) prime [ i ] = true ;
for ( let p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( let i = p * 2 ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( let p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) allPrimes . push ( p ) ; }
function factorialDivisors ( n ) {
let result = 1 ;
for ( let i = 0 ; i < allPrimes . length ; i ++ ) {
let p = allPrimes [ i ] ;
let exp = 0 ; while ( p <= n ) { exp = exp + Math . floor ( n / p ) ; p = p * allPrimes [ i ] ; }
result = Math . floor ( result * ( Math . pow ( allPrimes [ i ] , exp + 1 ) - 1 ) / ( allPrimes [ i ] - 1 ) ) ; }
return result ; }
document . write ( factorialDivisors ( 4 ) ) ;
function checkPandigital ( b , n ) {
if ( n . length < b ) return 0 ; let hash = [ ] ; for ( let i = 0 ; i < b ; i ++ ) hash [ i ] = 0 ;
for ( let i = 0 ; i < n . length ; i ++ ) {
if ( n [ i ] >= ' ' && n [ i ] <= ' ' ) hash [ n [ i ] - ' ' ] = 1 ;
else if ( n . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) <= b - 11 ) hash [ n . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) + 10 ] = 1 ; }
for ( let i = 0 ; i < b ; i ++ ) if ( hash [ i ] == 0 ) return 0 ; return 1 ; }
let b = 13 ; let n = " " ; if ( checkPandigital ( b , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function convert ( m , n ) { if ( m == n ) return 0 ;
if ( m > n ) return m - n ;
if ( m <= 0 && n > 0 ) return - 1 ;
if ( n % 2 == 1 )
return 1 + convert ( m , n + 1 ) ;
else
return 1 + convert ( m , n / 2 ) ; }
var m = 3 , n = 11 ; document . write ( " " + " " + convert ( m , n ) ) ;
var MAX = 10000 ; var prodDig = Array . from ( { length : MAX } , ( _ , i ) => 0 ) ;
function getDigitProduct ( x ) {
if ( x < 10 ) return x ;
if ( prodDig [ x ] != 0 ) return prodDig [ x ] ;
var prod = ( x % 10 ) * getDigitProduct ( parseInt ( x / 10 ) ) ; return ( prodDig [ x ] = prod ) ; }
function findSeed ( n ) {
var res = [ ] ; for ( var i = 1 ; i <= parseInt ( n / 2 ) ; i ++ ) if ( i * getDigitProduct ( i ) == n ) res . push ( i ) ;
if ( res . length == 0 ) { document . write ( " " ) ; return ; }
for ( i = 0 ; i < res . length ; i ++ ) document . write ( res [ i ] + " " ) ; }
var n = 138 ; findSeed ( n ) ;
function maxPrimefactorNum ( N ) { var arr = Array . from ( { length : N + 5 } , ( _ , i ) => 0 ) ;
for ( i = 2 ; i * i <= N ; i ++ ) { if ( arr [ i ] == 0 ) { for ( j = 2 * i ; j <= N ; j += i ) { arr [ j ] ++ ; } } arr [ i ] = 1 ; } var maxval = 0 , maxvar = 1 ;
for ( i = 1 ; i <= N ; i ++ ) { if ( arr [ i ] > maxval ) { maxval = arr [ i ] ; maxvar = i ; } } return maxvar ; }
var N = 40 ; document . write ( maxPrimefactorNum ( N ) ) ;
function SubArraySum ( arr , n ) { let result = 0 ;
for ( let i = 0 ; i < n ; i ++ ) result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) ;
return result ; }
let arr = [ 1 , 2 , 3 ] ; let n = arr . length ; document . write ( " " + SubArraySum ( arr , n ) ) ;
function highestPowerof2 ( n ) { let res = 0 ; for ( let i = n ; i >= 1 ; i -- ) {
if ( ( i & ( i - 1 ) ) == 0 ) { res = i ; break ; } } return res ; }
let n = 10 ; document . write ( highestPowerof2 ( n ) ) ;
function findPairs ( n ) {
var cubeRoot = parseInt ( Math . pow ( n , 1.0 / 3.0 ) ) ;
var cube = Array . from ( { length : cubeRoot + 1 } , ( _ , i ) => 0 ) ;
for ( i = 1 ; i <= cubeRoot ; i ++ ) cube [ i ] = i * i * i ;
var l = 1 ; var r = cubeRoot ; while ( l < r ) { if ( cube [ l ] + cube [ r ] < n ) l ++ ; else if ( cube [ l ] + cube [ r ] > n ) r -- ; else { document . write ( " " + l + " " + r + " " ) ; l ++ ; r -- ; } } }
var n = 20683 ; findPairs ( n ) ;
function gcd ( a , b ) { while ( b != 0 ) { let t = b ; b = a % b ; a = t ; } return a ; }
function findMinDiff ( a , b , x , y ) {
let g = gcd ( a , b ) ;
let diff = Math . abs ( x - y ) % g ; return Math . min ( diff , g - diff ) ; }
let a = 20 , b = 52 , x = 5 , y = 7 ; document . write ( findMinDiff ( a , b , x , y ) ) ;
function printDivisors ( n ) {
let v = [ ] ; let t = 0 ; for ( let i = 1 ; i <= parseInt ( Math . sqrt ( n ) ) ; i ++ ) { if ( n % i == 0 ) {
if ( parseInt ( n / i ) == i ) document . write ( i + " " ) ; else { document . write ( i + " " ) ;
v [ t ++ ] = parseInt ( n / i ) ; } } }
for ( let i = v . length - 1 ; i >= 0 ; i -- ) { document . write ( v [ i ] + " " ) ; } }
document . write ( " " ) ; printDivisors ( 100 ) ;
function printDivisors ( n ) { for ( var i = 1 ; i * i < n ; i ++ ) { if ( n % i == 0 ) document . write ( i + " " ) ; } for ( var i = Math . sqrt ( n ) ; i >= 1 ; i -- ) { if ( n % i == 0 ) document . write ( " " + n / i ) ; } }
document . write ( " " ) ; printDivisors ( 100 ) ;
function printDivisors ( n ) { for ( i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) document . write ( i + " " ) ; }
document . write ( " " + " " ) ; printDivisors ( 100 ) ;
function printDivisors ( n ) {
for ( let i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( parseInt ( n / i , 10 ) == i ) document . write ( i ) ;
else document . write ( i + " " + parseInt ( n / i , 10 ) + " " ) ; } } }
document . write ( " " ) ; printDivisors ( 100 ) ;
function SieveOfAtkin ( limit ) {
if ( limit > 2 ) document . write ( 2 + " " ) ; if ( limit > 3 ) document . write ( 3 + " " ) ;
let sieve = new Array ( ) sieve [ limit ] = 0 ; for ( let i = 0 ; i < limit ; i ++ ) sieve [ i ] = false ;
for ( let x = 1 ; x * x < limit ; x ++ ) { for ( let y = 1 ; y * y < limit ; y ++ ) {
let n = ( 4 * x * x ) + ( y * y ) ; if ( n <= limit && ( n % 12 == 1 n % 12 == 5 ) ) sieve [ n ] ^= true ; n = ( 3 * x * x ) + ( y * y ) ; if ( n <= limit && n % 12 == 7 ) sieve [ n ] = true ; n = ( 3 * x * x ) - ( y * y ) ; if ( x > y && n <= limit && n % 12 == 11 ) sieve [ n ] ^= true ; } }
for ( let r = 5 ; r * r < limit ; r ++ ) { if ( sieve [ r ] ) { for ( i = r * r ; i < limit ; i += r * r ) sieve [ i ] = false ; } }
for ( let a = 5 ; a < limit ; a ++ ) if ( sieve [ a ] ) document . write ( a , " " ) ; }
let limit = 20 ; SieveOfAtkin ( limit ) ;
function isInside ( circle_x , circle_y , rad , x , y ) {
if ( ( x - circle_x ) * ( x - circle_x ) + ( y - circle_y ) * ( y - circle_y ) <= rad * rad ) return true ; else return false ; }
var x = 1 ; var y = 1 ; var circle_x = 0 ; var circle_y = 1 ; var rad = 2 ; if ( isInside ( circle_x , circle_y , rad , x , y ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function eval ( a , op , b ) { if ( op == ' ' ) { return a + b ; } if ( op == ' ' ) { return a - b ; } if ( op == ' ' ) { return a * b ; } return Number . MAX_VALUE ; }
function evaluateAll ( expr , low , high ) {
let res = [ ] ;
if ( low == high ) { res . push ( expr [ low ] - ' ' ) ; return res ; }
if ( low == ( high - 2 ) ) { let num = eval ( expr [ low ] - ' ' , expr [ low + 1 ] , expr [ low + 2 ] - ' ' ) ; res . push ( num ) ; return res ; }
for ( let i = low + 1 ; i <= high ; i += 2 ) {
let l = evaluateAll ( expr , low , i - 1 ) ;
let r = evaluateAll ( expr , i + 1 , high ) ;
for ( let s1 = 0 ; s1 < l . length ; s1 ++ ) {
for ( let s2 = 0 ; s2 < r . length ; s2 ++ ) {
let val = eval ( l [ s1 ] , expr [ i ] , r [ s2 ] ) ; res . push ( val ) ; } } } return res ; }
let expr = " " ; let len = expr . length ; let ans = evaluateAll ( expr , 0 , len - 1 ) ; for ( let i = 0 ; i < ans . length ; i ++ ) { document . write ( ans [ i ] + " " ) ; }
function isLucky ( n ) {
var arr = Array ( 10 ) . fill ( 0 ) ; for ( var i = 0 ; i < 10 ; i ++ ) arr [ i ] = false ;
while ( n > 0 ) {
var digit = n % 10 ;
if ( arr [ digit ] ) return false ;
arr [ digit ] = true ;
n = parseInt ( n / 10 ) ; } return true ; }
var arr = [ 1291 , 897 , 4566 , 1232 , 80 , 700 ] var n = arr . length ; for ( var i = 0 ; i < n ; i ++ ) isLucky ( arr [ i ] ) ? document . write ( arr [ i ] + " " ) : document . write ( arr [ i ] + " " ) ;
function printSquares ( n ) {
let square = 0 , odd = 1 ;
for ( let x = 0 ; x < n ; x ++ ) {
document . write ( square + " " ) ;
square = square + odd ; odd = odd + 2 ; } }
let n = 5 ; printSquares ( n ) ;
var rev_num = 0 ; var base_pos = 1 ; function reversDigits ( num ) { if ( num > 0 ) { reversDigits ( Math . floor ( num / 10 ) ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
let num = 4562 ; document . write ( " " + reversDigits ( num ) ) ;
function RecursiveFunction ( ref , bit ) {
if ( ref . length == 0 bit < 0 ) return 0 ; let curr_on = [ ] , curr_off = [ ] ; for ( let i = 0 ; i < ref . length ; i ++ ) {
if ( ( ( ref [ i ] >> bit ) & 1 ) == 0 ) curr_off . push ( ref [ i ] ) ;
else curr_on . push ( ref [ i ] ) ; }
if ( curr_off . length == 0 ) return RecursiveFunction ( curr_on , bit - 1 ) ;
if ( curr_on . length == 0 ) return RecursiveFunction ( curr_off , bit - 1 ) ;
return Math . min ( RecursiveFunction ( curr_off , bit - 1 ) , RecursiveFunction ( curr_on , bit - 1 ) ) + ( 1 << bit ) ; }
function PrintMinimum ( a , n ) { let v = [ ] ;
for ( let i = 0 ; i < n ; i ++ ) v . push ( a [ i ] ) ;
document . write ( RecursiveFunction ( v , 30 ) + " " ) ; }
let arr = [ 3 , 2 , 1 ] ; let size = arr . length ; PrintMinimum ( arr , size ) ;
function cntElements ( arr , n ) {
let cnt = 0 ;
for ( let i = 0 ; i < n - 2 ; i ++ ) {
if ( arr [ i ] == ( arr [ i + 1 ] ^ arr [ i + 2 ] ) ) { cnt ++ ; } } return cnt ; }
let arr = [ 4 , 2 , 1 , 3 , 7 , 8 ] ; let n = arr . length ; document . write ( cntElements ( arr , n ) ) ;
function xor_triplet ( arr , n ) {
let ans = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
for ( let j = i + 1 ; j < n ; j ++ ) {
for ( let k = j ; k < n ; k ++ ) { let xor1 = 0 , xor2 = 0 ;
for ( let x = i ; x < j ; x ++ ) { xor1 ^= arr [ x ] ; }
for ( let x = j ; x <= k ; x ++ ) { xor2 ^= arr [ x ] ; }
if ( xor1 == xor2 ) { ans ++ ; } } } } return ans ; }
let arr = [ 1 , 2 , 3 , 4 , 5 ] ; let n = arr . length ;
document . write ( xor_triplet ( arr , n ) )
let N = 100005 ; let n , k ;
let al = new Array ( N ) . fill ( 0 ) . map ( ( t ) => [ ] ) ; let Ideal_pair ; let bit = new Array ( N ) ; let root_node = new Array ( N ) ;
function bit_q ( i , j ) { let sum = 0 ; while ( j > 0 ) { sum += bit [ j ] ; j -= ( j & ( j * - 1 ) ) ; } i -- ; while ( i > 0 ) { sum -= bit [ i ] ; i -= ( i & ( i * - 1 ) ) ; } return sum ; }
function bit_up ( i , diff ) { while ( i <= n ) { bit [ i ] += diff ; i += i & - i ; } }
function dfs ( node ) { Ideal_pair += bit_q ( Math . max ( 1 , node - k ) , Math . min ( n , node + k ) ) ; bit_up ( node , 1 ) ; for ( let i = 0 ; i < al [ node ] . length ; i ++ ) dfs ( al [ node ] [ i ] ) ; bit_up ( node , - 1 ) ; }
function initialise ( ) { Ideal_pair = 0 ; for ( let i = 0 ; i <= n ; i ++ ) { root_node [ i ] = true ; bit [ i ] = 0 ; } }
function Add_Edge ( x , y ) { al [ x ] . push ( y ) ; root_node [ y ] = false ; }
function Idealpairs ( ) {
let r = - 1 ; for ( let i = 1 ; i <= n ; i ++ ) if ( root_node [ i ] ) { r = i ; break ; } dfs ( r ) ; return Ideal_pair ; }
n = 6 , k = 3 ; initialise ( ) ;
Add_Edge ( 1 , 2 ) ; Add_Edge ( 1 , 3 ) ; Add_Edge ( 3 , 4 ) ; Add_Edge ( 3 , 5 ) ; Add_Edge ( 3 , 6 ) ;
document . write ( Idealpairs ( ) ) ;
function printSubsets ( n ) { for ( let i = n ; i > 0 ; i = ( i - 1 ) & n ) document . write ( i + " " ) ; document . write ( " " + " " ) ; }
let n = 9 ; printSubsets ( n ) ;
function isDivisibleby17 ( n ) {
if ( n == 0 n == 17 ) return true ;
if ( n < 17 ) return false ;
return isDivisibleby17 ( Math . floor ( n >> 4 ) - Math . floor ( n & 15 ) ) ; }
let n = 35 ; if ( isDivisibleby17 ( n ) ) document . write ( n + " " ) ; else document . write ( n + " " ) ;
function answer ( n ) {
let m = 2 ;
let ans = 1 ; let r = 1 ;
while ( r < n ) {
r = ( Math . pow ( 2 , m ) - 1 ) * ( Math . pow ( 2 , m - 1 ) ) ;
if ( r < n ) ans = r ;
m ++ ; } return ans ; }
let n = 7 ; document . write ( answer ( n ) ) ;
function setBitNumber ( n ) { if ( n == 0 ) return 0 ; let msb = 0 ; n = n / 2 ; while ( n != 0 ) { n = $n / 2 ; msb ++ ; } return ( 1 << msb ) ; }
let n = 0 ; document . write ( setBitNumber ( n ) ) ;
function setBitNumber ( n ) {
n |= n >> 1 ;
n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ;
n = n + 1 ;
return ( n >> 1 ) ; }
let n = 273 ; document . write ( setBitNumber ( n ) ) ;
function countTrailingZero ( x ) { let count = 0 ; while ( ( x & 1 ) == 0 ) { x = x >> 1 ; count ++ ; } return count ; }
document . write ( countTrailingZero ( 11 ) ) ;
function countTrailingZero ( x ) {
let lookup = [ 32 , 0 , 1 , 26 , 2 , 23 , 27 , 0 , 3 , 16 , 24 , 30 , 28 , 11 , 0 , 13 , 4 , 7 , 17 , 0 , 25 , 22 , 31 , 15 , 29 , 10 , 12 , 6 , 0 , 21 , 14 , 9 , 5 , 20 , 8 , 19 , 18 ] ;
return lookup [ ( - x & x ) % 37 ] ; }
document . write ( countTrailingZero ( 48 ) ) ;
function multiplyBySevenByEight ( n ) {
return ( n - ( n >> 3 ) ) ; }
let n = 9 ; document . write ( multiplyBySevenByEight ( n ) ) ;
function multiplyBySevenByEight ( n ) {
return ( ( n << 3 ) - n ) >> 3 ; }
var n = 15 ; document . write ( multiplyBySevenByEight ( n ) ) ;
function search ( list , num ) { var low = 0 , high = list . length - 1 ;
var ans = - 1 ; while ( low <= high ) {
var mid = low + ( high - low ) / 2 ;
if ( list [ mid ] <= num ) {
ans = mid ;
low = mid + 1 ; } else
high = mid - 1 ; }
return ans ; }
function isPalindrome ( n ) { var rev = 0 ; var temp = n ;
while ( n > 0 ) { rev = rev * 10 + n % 10 ; n = parseInt ( n / 10 ) ; }
return rev == temp ; }
function countNumbers ( L , R , K ) {
var list = [ ] ;
for ( var i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) ) {
list . push ( i ) ; } }
var count = 0 ;
for ( var i = 0 ; i < list . length ; i ++ ) {
var right_index = search ( list , list [ i ] + K - 1 ) ;
if ( right_index != - 1 ) count = Math . max ( count , right_index - i + 1 ) ; }
return count ; }
var L = 98 , R = 112 ; var K = 13 ; document . write ( countNumbers ( L , R , K ) ) ;
function findMaximumSum ( a , n ) {
var prev_smaller = findPrevious ( a , n ) ;
var next_smaller = findNext ( a , n ) ; var max_value = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
max_value = Math . max ( max_value , a [ i ] * ( next_smaller [ i ] - prev_smaller [ i ] - 1 ) ) ; }
return max_value ; }
function findPrevious ( a , n ) { var ps = Array ( n ) . fill ( 0 ) ;
ps [ 0 ] = - 1 ;
let stack = Array ( ) ;
stack . push ( 0 ) ; for ( var i = 1 ; i < a . length ; i ++ ) {
while ( stack . length > 0 && a [ stack [ stack . length - 1 ] ] >= a [ i ] ) stack . pop ( ) ;
ps [ i ] = stack . length > 0 ? stack [ stack . length - 1 ] : - 1 ;
stack . push ( i ) ; }
return ps ; }
function findNext ( a , n ) { var ns = Array ( n ) . fill ( 0 ) ; ns [ n - 1 ] = n ;
var stack = Array ( ) ; stack . push ( n - 1 ) ;
for ( var i = n - 2 ; i >= 0 ; i -- ) {
while ( stack . length > 0 && a [ stack [ stack . length - 1 ] ] >= a [ i ] ) stack . pop ( ) ;
ns [ i ] = stack . length > 0 ? stack [ stack . length - 1 ] : a . length ;
stack . push ( i ) ; }
return ns ; }
var n = 3 ; var a = [ 80 , 48 , 82 ] ; document . write ( findMaximumSum ( a , n ) ) ;
function compare ( arr1 , arr2 ) { for ( let i = 0 ; i < 256 ; i ++ ) if ( arr1 [ i ] != arr2 [ i ] ) return false ; return true ; }
function search ( pat , txt ) { let M = pat . length ; let N = txt . length ;
let countP = new Array ( 256 ) ; let countTW = new Array ( 256 ) ; for ( let i = 0 ; i < 256 ; i ++ ) { countP [ i ] = 0 ; countTW [ i ] = 0 ; } for ( let i = 0 ; i < 256 ; i ++ ) { countP [ i ] = 0 ; countTW [ i ] = 0 ; } for ( let i = 0 ; i < M ; i ++ ) { ( countP [ pat [ i ] . charCodeAt ( 0 ) ] ) ++ ; ( countTW [ txt [ i ] . charCodeAt ( 0 ) ] ) ++ ; }
for ( let i = M ; i < N ; i ++ ) {
if ( compare ( countP , countTW ) ) return true ;
( countTW [ txt [ i ] . charCodeAt ( 0 ) ] ) ++ ;
countTW [ txt [ i - M ] . charCodeAt ( 0 ) ] -- ; }
if ( compare ( countP , countTW ) ) return true ; return false ; }
let txt = " " ; let pat = " " ; if ( search ( pat , txt ) ) document . write ( " " ) ; else document . write ( " " ) ;
function getMaxMedian ( arr , n , k ) { let size = n + k ;
arr . sort ( ( a , b ) => a - b ) ;
if ( size % 2 == 0 ) { let median = ( arr [ Math . floor ( size / 2 ) - 1 ] + arr [ Math . floor ( size / 2 ) ] ) / 2 ; return median ; }
let median = arr [ Math . floor ( size / 2 ) ] ; return median ; }
let arr = [ 3 , 2 , 3 , 4 , 2 ] ; let n = arr . length ; let k = 2 ; document . write ( getMaxMedian ( arr , n , k ) ) ;
function printSorted ( a , b , c ) {
let get_max = Math . max ( a , Math . max ( b , c ) ) ;
let get_min = - Math . max ( - a , Math . max ( - b , - c ) ) ; let get_mid = ( a + b + c ) - ( get_max + get_min ) ; document . write ( get_min + " " + get_mid + " " + get_max ) ; }
let a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ;
function binarySearch ( a , item , low , high ) { while ( low <= high ) { var mid = low + ( high - low ) / 2 ; if ( item == a [ mid ] ) return mid + 1 ; else if ( item > a [ mid ] ) low = mid + 1 ; else high = mid - 1 ; } return low ; }
function insertionSort ( a , n ) { var i , loc , j , k , selected ; for ( i = 1 ; i < n ; ++ i ) { j = i - 1 ; selected = a [ i ] ;
loc = binarySearch ( a , selected , 0 , j ) ;
while ( j >= loc ) { a [ j + 1 ] = a [ j ] ; j -- ; } a [ j + 1 ] = selected ; } }
var a = [ 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 ] ; var n = a . length , i ; insertionSort ( a , n ) ; document . write ( " " + " " ) ; for ( i = 0 ; i < n ; i ++ ) document . write ( a [ i ] + " " ) ;
function insertionSort ( arr , n ) { let i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
let arr = [ 12 , 11 , 13 , 5 , 6 ] ; let n = arr . length ; insertionSort ( arr , n ) ; printArray ( arr , n ) ;
function validPermutations ( str ) { let m = new Map ( ) ;
let count = str . length , ans = 0 ;
for ( let i = 0 ; i < str . length ; i ++ ) { if ( m . has ( str [ i ] ) ) { m . set ( str [ i ] , m . get ( str [ i ] ) + 1 ) ; } else { m . set ( str [ i ] , 1 ) ; } } for ( let i = 0 ; i < str . length ; i ++ ) {
ans += count - m . get ( str [ i ] ) ;
m . set ( str [ i ] , m . get ( str [ i ] ) - 1 ) ; count -- ; }
return ans + 1 ; }
let str = " " ; document . write ( validPermutations ( str ) ) ;
function countPaths ( n , m ) { var dp = Array ( n + 1 ) . fill ( 0 ) . map ( x => Array ( m + 1 ) . fill ( 0 ) ) ;
for ( i = 0 ; i <= n ; i ++ ) dp [ i ] [ 0 ] = 1 ; for ( i = 0 ; i <= m ; i ++ ) dp [ 0 ] [ i ] = 1 ;
for ( i = 1 ; i <= n ; i ++ ) for ( j = 1 ; j <= m ; j ++ ) dp [ i ] [ j ] = dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] ; return dp [ n ] [ m ] ; }
var n = 3 , m = 2 ; document . write ( " " + countPaths ( n , m ) ) ;
function count ( S , m , n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
var arr = [ 1 , 2 , 3 ] ; var m = arr . length ; document . write ( count ( arr , m , 4 ) ) ;
str1 = str1 . toUpperCase ( ) ; str2 = str2 . toUpperCase ( ) ;
let x = str1 == ( str2 ) ;
if ( ! x ) { return false ; } else { return true ; } }
function equalIgnoreCaseUtil ( str1 , str2 ) { let res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
let str1 , str2 ; str1 = " " ; str2 = " " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " " ; str2 = " " ; equalIgnoreCaseUtil ( str1 , str2 ) ;
function replaceConsonants ( str ) {
var res = " " ; var i = 0 , count = 0 ;
while ( i < str . length ) {
if ( str [ i ] !== " " && str [ i ] !== " " && str [ i ] !== " " && str [ i ] !== " " && str [ i ] !== " " ) { i ++ ; count ++ ; } else {
if ( count > 0 ) res += count . toString ( ) ;
res += str [ i ] ; i ++ ; count = 0 ; } }
if ( count > 0 ) res += count . toString ( ) ;
return res ; }
var str = " " ; document . write ( replaceConsonants ( str ) ) ;
function isVowel ( c ) { return ( c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' ) ; }
function encryptString ( s , n , k ) { var countVowels = 0 ; var countConsonants = 0 ; var ans = " " ;
for ( var l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( var r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s [ r ] ) == true ) countVowels ++ ; else countConsonants ++ ; }
ans += ( countVowels * countConsonants ) . toString ( ) ; } return ans ; }
var s = " " ; var n = s . length ; var k = 2 ; document . write ( encryptString ( s , n , k ) ) ;
var charBuffer = " " ; function processWords ( input ) {
var s = input . split ( ' ' ) ; s . forEach ( element => {
charBuffer += element [ 0 ] ; } ) ; return charBuffer ; }
var input = " " ; document . write ( processWords ( input ) ) ;
function generateAllStringsUtil ( K , str , n ) {
if ( n == K ) {
str [ n ] = ' \0 ' ; document . write ( str . join ( " " ) + " " ) ; return ; }
if ( str [ n - 1 ] == ' ' ) { str [ n ] = ' ' ; generateAllStringsUtil ( K , str , n + 1 ) ; }
if ( str [ n - 1 ] == ' ' ) { str [ n ] = ' ' ; generateAllStringsUtil ( K , str , n + 1 ) ; str [ n ] = ' ' ; generateAllStringsUtil ( K , str , n + 1 ) ; } }
function generateAllStrings ( K ) {
if ( K <= 0 ) return ;
var str = new Array ( K ) ;
str [ 0 ] = ' ' ; generateAllStringsUtil ( K , str , 1 ) ;
str [ 0 ] = ' ' ; generateAllStringsUtil ( K , str , 1 ) ; }
var K = 3 ; generateAllStrings ( K ) ;
function findVolume ( a ) {
if ( a < 0 ) return - 1 ;
var r = a / 2 ;
var h = a ;
var V = ( 3.14 * Math . pow ( r , 2 ) * h ) ; return V ; }
var a = 5 ; document . write ( findVolume ( a ) ) ;
function volumeTriangular ( a , b , h ) { let vol = ( 0.1666 ) * a * b * h ; return vol ; }
function volumeSquare ( b , h ) { let vol = ( 0.33 ) * b * b * h ; return vol ; }
function volumePentagonal ( a , b , h ) { let vol = ( 0.83 ) * a * b * h ; return vol ; }
function volumeHexagonal ( a , b , h ) { let vol = a * b * h ; return vol ; }
let b = 4 , h = 9 , a = 4 ; document . write ( " " + " " + volumeTriangular ( a , b , h ) + " " ) ; document . write ( " " + " " + volumeSquare ( b , h ) + " " ) ; document . write ( " " + " " + volumePentagonal ( a , b , h ) + " " ) ; document . write ( " " + " " + volumeHexagonal ( a , b , h ) ) ;
function Area ( b1 , b2 , h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
let base1 = 8 , base2 = 10 , height = 6 ; let area = Area ( base1 , base2 , height ) ; document . write ( " " + area ) ;
function numberOfDiagonals ( n ) { return n * ( n - 3 ) / 2 ; }
var n = 5 ; document . write ( n + " " ) ; document . write ( numberOfDiagonals ( n ) + " " ) ;
function maximumArea ( l , b , x , y ) {
var left = x * b ; var right = ( l - x - 1 ) * b ; var above = l * y ; var below = ( b - y - 1 ) * l ;
document . write ( Math . max ( Math . max ( left , right ) , Math . max ( above , below ) ) ) ; }
var L = 8 , B = 8 ; var X = 0 , Y = 0 ;
maximumArea ( L , B , X , Y ) ;
function delCost ( s , cost ) {
var ans = 0 ;
var forMax = new Map ( ) ;
var forTot = new Map ( ) ;
for ( var i = 0 ; i < s . length ; i ++ ) {
if ( ! forMax . has ( s [ i ] ) ) { forMax . set ( s [ i ] , cost [ i ] ) ; } else {
forMax . set ( s [ i ] , Math . max ( forMax . get ( s [ i ] ) , cost [ i ] ) ) }
if ( ! forTot . has ( s [ i ] ) ) { forTot . set ( s [ i ] , cost [ i ] ) ; } else {
forTot . set ( s [ i ] , forTot . get ( s [ i ] ) + cost [ i ] ) } }
forMax . forEach ( ( value , key ) => {
ans += forTot . get ( key ) - value ; } ) ;
return ans ; }
var s = " " ;
var cost = [ 1 , 2 , 3 , 4 , 5 , 6 ] ;
document . write ( delCost ( s , cost ) ) ;
function computeDivisors ( ) { for ( let i = 1 ; i <= MAX ; i ++ ) { for ( let j = i ; j <= MAX ; j += i ) {
divisors [ j ] . push ( i ) ; } } }
function getClosest ( val1 , val2 , target ) { if ( target - val1 >= val2 - target ) return val2 ; else return val1 ; }
function findClosest ( arr , n , target ) {
if ( target <= arr [ 0 ] ) return arr [ 0 ] ; if ( target >= arr [ n - 1 ] ) return arr [ n - 1 ] ;
let i = 0 , j = n , mid = 0 ; while ( i < j ) { mid = Math . floor ( ( i + j ) / 2 ) ; if ( arr [ mid ] == target ) return arr [ mid ] ;
if ( target < arr [ mid ] ) {
if ( mid > 0 && target > arr [ mid - 1 ] ) return getClosest ( arr [ mid - 1 ] , arr [ mid ] , target ) ;
j = mid ; }
else { if ( mid < n - 1 && target < arr [ mid + 1 ] ) return getClosest ( arr [ mid ] , arr [ mid + 1 ] , target ) ;
i = mid + 1 ; } }
return arr [ mid ] ; }
function printClosest ( N , X ) {
computeDivisors ( ) ;
let ans = findClosest ( divisors [ N ] , divisors [ N ] . length , X ) ;
document . write ( ans ) ; }
let N = 16 , X = 5 ; for ( let i = 0 ; i < divisors . length ; i ++ ) divisors [ i ] = [ ] ;
printClosest ( N , X ) ;
function maxMatch ( A , B ) {
var Aindex = { } ;
var diff = { } ;
for ( var i = 0 ; i < A . length ; i ++ ) { Aindex [ A [ i ] ] = i ; }
for ( var i = 0 ; i < B . length ; i ++ ) {
if ( i - Aindex [ B [ i ] ] < 0 ) { if ( ! diff . hasOwnProperty ( A . length + i - Aindex [ B [ i ] ] ) ) { diff [ A . length + i - Aindex [ B [ i ] ] ] = 1 ; } else { diff [ A . length + i - Aindex [ B [ i ] ] ] += 1 ; } }
else { if ( ! diff . hasOwnProperty ( i - Aindex [ B [ i ] ] ) ) { diff [ i - Aindex [ B [ i ] ] ] = 1 ; } else { diff [ i - Aindex [ B [ i ] ] ] += 1 ; } } }
var max = 0 ; for ( const [ key , value ] of Object . entries ( diff ) ) { if ( value > max ) { max = value ; } } return max ; }
var A = [ 5 , 3 , 7 , 9 , 8 ] ; var B = [ 8 , 7 , 3 , 5 , 9 ] ;
document . write ( maxMatch ( A , B ) ) ;
function isinRange ( board ) {
for ( var i = 0 ; i < N ; i ++ ) { for ( var j = 0 ; j < N ; j ++ ) {
function isValidSudoku ( board ) {
if ( isinRange ( board ) == false ) { return false ; }
var unique = Array ( N + 1 ) . fill ( false ) ;
for ( var i = 0 ; i < N ; i ++ ) { unique = Array ( N + 1 ) . fill ( false ) ;
unique = Array ( N + 1 ) . fill ( false ) ;
for ( var j = 0 ; j < N ; j ++ ) {
var Z = board [ i ] [ j ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( var i = 0 ; i < N ; i ++ ) {
for ( var j = 0 ; j < N ; j ++ ) {
var Z = board [ j ] [ i ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( var i = 0 ; i < N - 2 ; i += 3 ) {
for ( var j = 0 ; j < N - 2 ; j += 3 ) {
unique = Array ( N + 1 ) . fill ( false ) ;
for ( var k = 0 ; k < 3 ; k ++ ) { for ( var l = 0 ; l < 3 ; l ++ ) {
var X = i + k ;
var Y = j + l ;
var Z = board [ X ] [ Y ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } } } }
return true ; }
var board = [ [ 7 , 9 , 2 , 1 , 5 , 4 , 3 , 8 , 6 ] , [ 6 , 4 , 3 , 8 , 2 , 7 , 1 , 5 , 9 ] , [ 8 , 5 , 1 , 3 , 9 , 6 , 7 , 2 , 4 ] , [ 2 , 6 , 5 , 9 , 7 , 3 , 8 , 4 , 1 ] , [ 4 , 8 , 9 , 5 , 6 , 1 , 2 , 7 , 3 ] , [ 3 , 1 , 7 , 4 , 8 , 2 , 9 , 6 , 5 ] , [ 1 , 3 , 6 , 7 , 4 , 8 , 5 , 9 , 2 ] , [ 9 , 7 , 4 , 2 , 1 , 5 , 6 , 3 , 8 ] , [ 5 , 2 , 8 , 6 , 3 , 9 , 4 , 1 , 7 ] ] ; if ( isValidSudoku ( board ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function palindrome ( a , i , j ) { while ( i < j ) {
if ( a [ i ] != a [ j ] ) return false ;
i ++ ; j -- ; }
return true ; }
function findSubArray ( arr , k ) { let n = arr . length ;
for ( let i = 0 ; i <= n - k ; i ++ ) { if ( palindrome ( arr , i , i + k - 1 ) ) return i ; }
return - 1 ; }
let arr = [ 2 , 3 , 5 , 1 , 3 ] ; let k = 4 ; let ans = findSubArray ( arr , k ) ; if ( ans == - 1 ) document . write ( - 1 + " " ) ; else { for ( let i = ans ; i < ans + k ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; }
function isCrossed ( path ) { if ( path . length == 0 ) return ;
let ans = false ;
let mySet = new Set ( ) ;
let x = 0 , y = 0 ; mySet . add ( [ x , y ] ) ;
for ( let i = 0 ; i < path . length ; i ++ ) {
if ( path [ i ] == ' ' ) mySet . add ( [ x , y ++ ] ) ; if ( path [ i ] == ' ' ) mySet . add ( [ x , y -- ] ) ; if ( path [ i ] == ' ' ) mySet . add ( [ x ++ , y ] ) ; if ( path [ i ] == ' ' ) mySet . add ( [ x -- , y ] ) ;
if ( ! mySet . has ( [ x , y ] ) ) { ans = true ; break ; } }
if ( ans ) document . write ( " " ) ; else document . write ( " " ) ; }
let path = " " ;
isCrossed ( path ) ;
function maxWidth ( N , M , cost , s ) {
let adj = [ ] ; for ( let i = 0 ; i < N ; i ++ ) { adj . push ( [ ] ) ; } for ( let i = 0 ; i < M ; i ++ ) { adj [ s [ i ] [ 0 ] ] . push ( s [ i ] [ 1 ] ) ; }
let result = 0 ;
let q = [ ] ;
q . push ( 0 ) ;
while ( q . length != 0 ) {
let count = q . length ;
result = Math . max ( count , result ) ;
while ( count -- > 0 ) {
let temp = q . shift ( ) ;
for ( let i = 0 ; i < adj [ temp ] . length ; i ++ ) { q . push ( adj [ temp ] [ i ] ) ; } } }
return result ; }
let N = 11 , M = 10 ; let edges = [ ] ; edges . push ( [ 0 , 1 ] ) ; edges . push ( [ 0 , 2 ] ) ; edges . push ( [ 0 , 3 ] ) ; edges . push ( [ 1 , 4 ] ) ; edges . push ( [ 1 , 5 ] ) ; edges . push ( [ 3 , 6 ] ) ; edges . push ( [ 4 , 7 ] ) ; edges . push ( [ 6 , 10 ] ) ; edges . push ( [ 6 , 8 ] ) ; edges . push ( [ 6 , 9 ] ) ; let cost = [ 1 , 2 , - 1 , 3 , 4 , 5 , 8 , 2 , 6 , 12 , 7 ] ;
document . write ( maxWidth ( N , M , cost , edges ) ) ;
let MAX = 10000000 ;
let isPrime = new Array ( MAX ) ;
let primes = new Array ( ) ;
function SieveOfEratosthenes ( ) { isPrime . fill ( true ) ; for ( let p = 2 ; p * p <= MAX ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( let i = p * p ; i <= MAX ; i += p ) isPrime [ i ] = false ; } }
for ( let p = 2 ; p <= MAX ; p ++ ) if ( isPrime [ p ] ) primes . push ( p ) ; }
function prime_search ( primes , diff ) {
let low = 0 ; let high = primes . length - 1 ; let res = 0 ; while ( low <= high ) { let mid = Math . floor ( ( low + high ) / 2 ) ;
if ( primes [ mid ] == diff ) {
return primes [ mid ] ; }
else if ( primes [ mid ] < diff ) {
low = mid + 1 ; }
else { res = primes [ mid ] ;
high = mid - 1 ; } }
return res ; }
function minCost ( arr , n ) {
SieveOfEratosthenes ( ) ;
let res = 0 ;
for ( let i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] < arr [ i - 1 ] ) { let diff = arr [ i - 1 ] - arr [ i ] ;
let closest_prime = prime_search ( primes , diff ) ;
res += closest_prime ;
arr [ i ] += closest_prime ; } }
return res ; }
let arr = [ 2 , 1 , 5 , 4 , 3 ] ; let n = 5 ;
document . write ( minCost ( arr , n ) )
function count ( s ) {
var cnt = 0 ;
s . split ( ' ' ) . forEach ( c => { cnt += ( c == ' ' ) ? 1 : 0 ; } ) ;
if ( cnt % 3 != 0 ) return 0 ; var res = 0 , k = parseInt ( cnt / 3 ) , sum = 0 ;
var mp = new Map ( ) ;
for ( var i = 0 ; i < s . length ; i ++ ) {
sum += ( s [ i ] == ' ' ) ? 1 : 0 ;
if ( sum == 2 * k && mp . has ( k ) && i < s . length - 1 && i > 0 ) { res += mp . get ( k ) ; }
if ( mp . has ( sum ) ) mp . set ( sum , mp . get ( sum ) + 1 ) else mp . set ( sum , 1 ) ; }
return res ; }
var str = " " ;
document . write ( count ( str ) ) ;
function splitstring ( s ) { let n = s . length ;
let zeros = 0 ; for ( let i = 0 ; i < n ; i ++ ) if ( s [ i ] == ' ' ) zeros ++ ;
if ( zeros % 3 != 0 ) return 0 ;
if ( zeros == 0 ) return parseInt ( ( ( n - 1 ) * ( n - 2 ) ) / 2 , 10 ) ;
let zerosInEachSubstring = parseInt ( zeros / 3 , 10 ) ;
let waysOfFirstCut = 0 ; let waysOfSecondCut = 0 ;
let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == ' ' ) count ++ ;
if ( count == zerosInEachSubstring ) waysOfFirstCut ++ ;
else if ( count == 2 * zerosInEachSubstring ) waysOfSecondCut ++ ; }
return waysOfFirstCut * waysOfSecondCut ; }
let s = " " ;
document . write ( " " + " " + splitstring ( s ) ) ;
function canTransform ( str1 , str2 ) { var s1 = " " ; var s2 = " " ;
for ( const c of str1 ) { if ( c !== " " ) { s1 += c ; } } for ( const c of str2 ) { if ( c !== " " ) { s2 += c ; } }
if ( s1 !== s2 ) return false ; var i = 0 ; var j = 0 ; var n = str1 . length ;
while ( i < n && j < n ) { if ( str1 [ i ] === " " ) { i ++ ; } else if ( str2 [ j ] === " " ) { j ++ ; }
else { if ( ( str1 [ i ] === " " && i < j ) || ( str1 [ i ] === " " && i > j ) ) { return false ; } i ++ ; j ++ ; } } return true ; }
var str1 = " " ; var str2 = " " ;
if ( canTransform ( str1 . split ( " " ) , str2 . split ( " " ) ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function maxsubStringLength ( S , N ) { let arr = Array . from ( { length : N } , ( _ , i ) => 0 ) ;
for ( let i = 0 ; i < N ; i ++ ) if ( S [ i ] == ' ' S [ i ] == ' ' S [ i ] == ' ' S [ i ] == ' ' S [ i ] == ' ' ) arr [ i ] = 1 ; else arr [ i ] = - 1 ;
let maxLen = 0 ;
let curr_sum = 0 ;
let hash = new Map ( ) ;
for ( let i = 0 ; i < N ; i ++ ) { curr_sum += arr [ i ] ;
if ( curr_sum == 0 )
maxLen = Math . max ( maxLen , i + 1 ) ;
if ( hash . has ( curr_sum ) ) maxLen = Math . max ( maxLen , i - hash . get ( curr_sum ) ) ;
else hash . set ( curr_sum , i ) ; }
return maxLen ; }
let S = " " ; let n = S . length ; document . write ( maxsubStringLength ( S . split ( ' ' ) , n ) ) ;
let mat = new Array ( 1001 ) ; for ( let i = 0 ; i < 1001 ; i ++ ) { mat [ i ] = new Array ( 1001 ) ; for ( let j = 0 ; j < 1001 ; j ++ ) { mat [ i ] [ j ] = 0 ; } } let r , c , x , y ;
let dx = [ 0 , - 1 , - 1 , - 1 , 0 , 1 , 1 , 1 ] ; let dy = [ 1 , 1 , 0 , - 1 , - 1 , - 1 , 0 , 1 ] ;
function FindMinimumDistance ( ) {
let q = [ ] ;
q . push ( [ x , y ] ) ; mat [ x ] [ y ] = 0 ;
while ( q . length > 0 ) {
x = q [ 0 ] [ 0 ] ; y = q [ 0 ] [ 1 ] ;
q . shift ( ) ; for ( let i = 0 ; i < 8 ; i ++ ) { let a = x + dx [ i ] ; let b = y + dy [ i ] ;
if ( a < 0 a >= r b >= c b < 0 ) continue ;
if ( mat [ a ] [ b ] == 0 ) {
mat [ a ] [ b ] = mat [ x ] [ y ] + 1 ;
q . push ( [ a , b ] ) ; } } } }
r = 5 , c = 5 , x = 1 , y = 1 ; let t = x ; let l = y ; mat [ x ] [ y ] = 0 ; FindMinimumDistance ( ) ; mat [ t ] [ l ] = 0 ;
for ( let i = 0 ; i < r ; i ++ ) { for ( let j = 0 ; j < c ; j ++ ) { document . write ( mat [ i ] [ j ] + " " ) ; } document . write ( " " ) ; }
function minOperations ( S , K ) {
var ans = 0 ;
for ( var i = 0 ; i < K ; i ++ ) {
var zero = 0 , one = 0 ;
for ( var j = i ; j < S . length ; j += K ) {
if ( S [ j ] === " " ) zero ++ ;
else one ++ ; }
ans += Math . min ( zero , one ) ; }
return ans ; }
var S = " " ; var K = 3 ; document . write ( minOperations ( S , K ) ) ;
function missingElement ( arr , n ) {
let max_ele = arr [ 0 ] ;
let min_ele = arr [ 0 ] ;
let x = 0 ;
let d ;
for ( let i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max_ele ) max_ele = arr [ i ] ; if ( arr [ i ] < min_ele ) min_ele = arr [ i ] ; }
d = parseInt ( ( max_ele - min_ele ) / n , 10 ) ;
for ( let i = 0 ; i < n ; i ++ ) { x = x ^ arr [ i ] ; }
for ( let i = 0 ; i <= n ; i ++ ) { x = x ^ ( min_ele + ( i * d ) ) ; }
return x ; }
let arr = [ 12 , 3 , 6 , 15 , 18 ] ; let n = arr . length ;
let element = missingElement ( arr , n ) ;
document . write ( element ) ;
function Printksubstring ( str , n , k ) {
let total = parseInt ( ( n * ( n + 1 ) ) / 2 , 10 ) ;
if ( k > total ) { document . write ( " " + " " ) ; return ; }
let substring = new Array ( n + 1 ) ; substring [ 0 ] = 0 ;
let temp = n ; for ( let i = 1 ; i <= n ; i ++ ) {
substring [ i ] = substring [ i - 1 ] + temp ; temp -- ; }
let l = 1 ; let h = n ; let start = 0 ; while ( l <= h ) { let m = parseInt ( ( l + h ) / 2 , 10 ) ; if ( substring [ m ] > k ) { start = m ; h = m - 1 ; } else if ( substring [ m ] < k ) { l = m + 1 ; } else { start = m ; break ; } }
let end = n - ( substring [ start ] - k ) ;
for ( let i = start - 1 ; i < end ; i ++ ) { document . write ( str [ i ] ) ; } }
let str = " " ; let k = 4 ; let n = str . length ; Printksubstring ( str , n , k ) ;
function LowerInsertionPoint ( arr , n , X ) {
if ( X < arr [ 0 ] ) return 0 ; else if ( X > arr [ n - 1 ] ) return n ; let lowerPnt = 0 ; let i = 1 ; while ( i < n && arr [ i ] < X ) { lowerPnt = i ; i = i * 2 ; }
while ( lowerPnt < n && arr [ lowerPnt ] < X ) lowerPnt ++ ; return lowerPnt ; }
let arr = [ 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 ] ; let n = arr . length ; let X = 4 ; document . write ( LowerInsertionPoint ( arr , n , X ) ) ;
function getCount ( M , N ) { let count = 0 ;
if ( M == 1 ) return N ;
if ( N == 1 ) return M ; if ( N > M ) {
for ( let i = 1 ; i <= M ; i ++ ) { let numerator = N * i - N + M - i ; let denominator = M - 1 ;
if ( numerator % denominator == 0 ) { let j = parseInt ( numerator / denominator , 10 ) ;
if ( j >= 1 && j <= N ) count ++ ; } } } else {
for ( let j = 1 ; j <= N ; j ++ ) { let numerator = M * j - M + N - j ; let denominator = N - 1 ;
if ( numerator % denominator == 0 ) { let i = parseInt ( numerator / denominator , 10 ) ;
if ( i >= 1 && i <= M ) count ++ ; } } } return count ; }
let M = 3 , N = 5 ; document . write ( getCount ( M , N ) ) ;
function swapElement ( arr1 , arr2 , n ) {
let wrongIdx = 0 ; for ( let i = 1 ; i < n ; i ++ ) { if ( arr1 [ i ] < arr1 [ i - 1 ] ) { wrongIdx = i ; } } let maximum = Number . MIN_VALUE ; let maxIdx = - 1 ; let res = false ;
for ( let i = 0 ; i < n ; i ++ ) { if ( arr2 [ i ] > maximum && arr2 [ i ] >= arr1 [ wrongIdx - 1 ] ) { if ( wrongIdx + 1 <= n - 1 && arr2 [ i ] <= arr1 [ wrongIdx + 1 ] ) { maximum = arr2 [ i ] ; maxIdx = i ; res = true ; } } }
if ( res ) { swap ( arr1 , wrongIdx , arr2 , maxIdx ) ; } return res ; } function swap ( a , wrongIdx , b , maxIdx ) { let c = a [ wrongIdx ] ; a [ wrongIdx ] = b [ maxIdx ] ; b [ maxIdx ] = c ; }
function getSortedArray ( arr1 , arr2 , n ) { if ( swapElement ( arr1 , arr2 , n ) ) { for ( let i = 0 ; i < n ; i ++ ) { document . write ( arr1 [ i ] + " " ) ; } } else { document . write ( " " ) ; } }
let arr1 = [ 1 , 3 , 7 , 4 , 10 ] ; let arr2 = [ 2 , 1 , 6 , 8 , 9 ] ; let n = arr1 . length ; getSortedArray ( arr1 , arr2 , n ) ;
function middleOfThree ( a , b , c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
let a = 20 , b = 30 , c = 40 ; document . write ( middleOfThree ( a , b , c ) ) ;
function transpose ( mat , row , col ) {
let tr = new Array ( col ) ; for ( let i = 0 ; i < col ; i ++ ) { tr [ i ] = new Array ( row ) ; }
for ( let i = 0 ; i < row ; i ++ ) {
for ( let j = 0 ; j < col ; j ++ ) {
tr [ j ] [ i ] = mat [ i ] [ j ] ; } } return tr ; }
function RowWiseSort ( B ) {
for ( let i = 0 ; i < B . length ; i ++ ) {
( B [ i ] ) . sort ( function ( a , b ) { return a - b ; } ) ; } }
function sortCol ( mat , n , M ) {
let B = transpose ( mat , N , M ) ;
RowWiseSort ( B ) ;
mat = transpose ( B , M , N ) ;
for ( let i = 0 ; i < N ; i ++ ) { for ( let j = 0 ; j < M ; j ++ ) { document . write ( mat [ i ] [ j ] + " " ) ; } document . write ( " " ) ; } }
let mat = [ [ 1 , 6 , 10 ] , [ 8 , 5 , 9 ] , [ 9 , 4 , 15 ] , [ 7 , 3 , 60 ] ] ; let N = mat . length ; let M = mat [ 0 ] . length ;
sortCol ( mat , N , M ) ;
function largestArea ( N , M , H , V , h , v ) {
var s1 = new Set ( ) ; var s2 = new Set ( ) ;
for ( var i = 1 ; i <= N + 1 ; i ++ ) s1 . add ( i ) ;
for ( var i = 1 ; i <= M + 1 ; i ++ ) s2 . add ( i ) ;
for ( var i = 0 ; i < h ; i ++ ) { s1 . delete ( H [ i ] ) ; }
for ( var i = 0 ; i < v ; i ++ ) { s2 . delete ( V [ i ] ) ; }
var list1 = Array ( s1 . size ) ; var list2 = Array ( s2 . size ) ; var i = 0 ; s1 . forEach ( element => { list1 [ i ++ ] = element ; } ) ; i = 0 ; s2 . forEach ( element => { list2 [ i ++ ] = element ; } ) ;
list1 . sort ( ( a , b ) => a - b ) list2 . sort ( ( a , b ) => a - b ) var maxH = 0 , p1 = 0 , maxV = 0 , p2 = 0 ;
for ( var j = 0 ; j < s1 . size ; j ++ ) { maxH = Math . max ( maxH , list1 [ j ] - p1 ) ; p1 = list1 [ j ] ; }
for ( var j = 0 ; j < s2 . size ; j ++ ) { maxV = Math . max ( maxV , list2 [ j ] - p2 ) ; p2 = list2 [ j ] ; }
document . write ( maxV * maxH ) ; }
var N = 3 , M = 3 ;
var H = [ 2 ] ; var V = [ 2 ] ; var h = H . length ; var v = V . length ;
largestArea ( N , M , H , V , h , v ) ;
function checkifSorted ( A , B , N ) {
var flag = false ;
for ( i = 0 ; i < N - 1 ; i ++ ) {
if ( A [ i ] > A [ i + 1 ] ) {
flag = true ; break ; } }
if ( ! flag ) { return true ; }
var count = 0 ;
for ( i = 0 ; i < N ; i ++ ) {
if ( B [ i ] == 0 ) {
count ++ ; break ; } }
for ( i = 0 ; i < N ; i ++ ) {
if ( B [ i ] == 1 ) { count ++ ; break ; } }
if ( count == 2 ) { return true ; } return false ; }
var A = [ 3 , 1 , 2 ] ;
var B = [ 0 , 1 , 1 ] ; var N = A . length ;
var check = checkifSorted ( A , B , N ) ;
if ( check ) { document . write ( " " ) ; }
else { document . write ( " " ) ; }
function minSteps ( A , B , M , N ) { if ( A [ 0 ] > B [ 0 ] ) return 0 ; if ( B [ 0 ] > A [ 0 ] ) { return 1 ; }
if ( M <= N && A [ 0 ] == B [ 0 ] && count ( A , A [ 0 ] ) == M && count ( B , B [ 0 ] ) == N ) return - 1 ;
for ( var i = 1 ; i < N ; i ++ ) { if ( B [ i ] > B [ 0 ] ) return 1 ; }
for ( var i = 1 ; i < M ; i ++ ) { if ( A [ i ] < A [ 0 ] ) return 1 ; }
for ( var i = 1 ; i < M ; i ++ ) { if ( A [ i ] > A [ 0 ] ) { swap ( A , i , B , 0 ) ; swap ( A , 0 , B , 0 ) ; return 2 ; } }
for ( var i = 1 ; i < N ; i ++ ) { if ( B [ i ] < B [ 0 ] ) { swap ( A , 0 , B , i ) ; swap ( A , 0 , B , 0 ) ; return 2 ; } }
return 0 ; } function count ( a , c ) { count = 0 ; for ( var i = 0 ; i < a . length ; i ++ ) if ( a [ i ] == c ) count ++ ; return count ; } function swap ( s1 , index1 , s2 , index2 ) { var c = s1 [ index1 ] ; s1 [ index1 ] = s2 [ index2 ] ; s2 [ index2 ] = c ; }
var A = " " ; var B = " " ; var M = A . length ; var N = B . length ; document . write ( minSteps ( A , B , M , N ) ) ;
var maxN = 201 ;
var n1 , n2 , n3 ;
var dp = Array . from ( Array ( maxN ) , ( ) => Array ( maxN ) ) ; for ( var i = 0 ; i < maxN ; i ++ ) for ( var j = 0 ; j < maxN ; j ++ ) dp [ i ] [ j ] = new Array ( maxN ) . fill ( - 1 ) ;
function getMaxSum ( i , j , k , arr1 , arr2 , arr3 ) {
var cnt = 0 ; if ( i >= n1 ) cnt ++ ; if ( j >= n2 ) cnt ++ ; if ( k >= n3 ) cnt ++ ;
if ( cnt >= 2 ) return 0 ;
if ( dp [ i ] [ j ] [ k ] != - 1 ) return dp [ i ] [ j ] [ k ] ; var ans = 0 ;
if ( i < n1 && j < n2 )
ans = Math . max ( ans , getMaxSum ( i + 1 , j + 1 , k , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr2 [ j ] ) ; if ( i < n1 && k < n3 ) ans = Math . max ( ans , getMaxSum ( i + 1 , j , k + 1 , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr3 [ k ] ) ; if ( j < n2 && k < n3 ) ans = Math . max ( ans , getMaxSum ( i , j + 1 , k + 1 , arr1 , arr2 , arr3 ) + arr2 [ j ] * arr3 [ k ] ) ;
dp [ i ] [ j ] [ k ] = ans ;
return dp [ i ] [ j ] [ k ] ; }
function maxProductSum ( arr1 , arr2 , arr3 ) {
arr1 . sort ( ) ; arr1 . reverse ( ) ; arr2 . sort ( ) ; arr2 . reverse ( ) ; arr3 . sort ( ) ; arr3 . reverse ( ) ; return getMaxSum ( 0 , 0 , 0 , arr1 , arr2 , arr3 ) ; }
n1 = 2 ; var arr1 = [ 3 , 5 ] ; n2 = 2 ; var arr2 = [ 2 , 1 ] ; n3 = 3 ; var arr3 = [ 4 , 3 , 5 ] ; document . write ( maxProductSum ( arr1 , arr2 , arr3 ) ) ;
function findTriplet ( arr , N ) {
arr . sort ( ( a , b ) => a - b ) ; var flag = 0 , i ;
for ( i = N - 1 ; i - 2 >= 0 ; i -- ) {
if ( arr [ i - 2 ] + arr [ i - 1 ] > arr [ i ] ) { flag = 1 ; break ; } }
if ( flag ) {
document . write ( arr [ i - 2 ] + " " + arr [ i - 1 ] + " " + arr [ i ] + " " ) ; }
else { document . write ( - 1 + " " ) ; } }
var arr = [ 4 , 2 , 10 , 3 , 5 ] ; var N = arr . length ; findTriplet ( arr , N ) ;
function numberofpairs ( arr , N ) {
let answer = 0 ;
arr . sort ( ) ;
let minDiff = Number . MAX_VALUE ; for ( let i = 0 ; i < N - 1 ; i ++ )
minDiff = Math . min ( minDiff , arr [ i + 1 ] - arr [ i ] ) ; for ( let i = 0 ; i < N - 1 ; i ++ ) { if ( arr [ i + 1 ] - arr [ i ] == minDiff )
answer ++ ; }
return answer ; }
let arr = [ 4 , 2 , 1 , 3 ] ; let N = arr . length ;
document . write ( numberofpairs ( arr , N ) ) ;
let max_length = 0 ;
let store = [ ] ;
let ans = [ ] ;
function find_max_length ( arr , index , sum , k ) { sum = sum + arr [ index ] ; store . push ( arr [ index ] ) ; if ( sum == k ) { if ( max_length < store . length ) {
max_length = store . length ;
ans = store ; } } for ( let i = index + 1 ; i < arr . length ; i ++ ) { if ( sum + arr [ i ] <= k ) {
find_max_length ( arr , i , sum , k ) ;
store . pop ( ) ; }
else return ; } return ; } function longestSubsequence ( arr , n , k ) {
arr . sort ( function ( a , b ) { return a - b ; } ) ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( max_length >= n - i ) break ; store = [ ] ; find_max_length ( arr , i , 0 , k ) ; } return max_length ; }
let arr = [ - 3 , 0 , 1 , 1 , 2 ] ; let n = arr . length ; let k = 1 ; document . write ( longestSubsequence ( arr , n , k ) ) ;
function sortArray ( A , N ) {
if ( N % 4 == 0 N % 4 == 1 ) {
for ( let i = 0 ; i < N / 2 ; i ++ ) { x = i ; if ( i % 2 == 0 ) { y = N - i - 2 ; z = N - i - 1 ; }
A [ z ] = A [ y ] ; A [ y ] = A [ x ] ; A [ x ] = x + 1 ; }
document . write ( " " ) ; for ( let i = 0 ; i < N ; i ++ ) document . write ( A [ i ] + " " ) ; }
else { document . write ( " " ) ; } }
let A = [ 5 , 4 , 3 , 2 , 1 ] ; let N = A . length ; sortArray ( A , N ) ;
function findK ( arr , size , N ) {
arr . sort ( function ( a , b ) { return a - b } ) ; let temp_sum = 0 ;
for ( let i = 0 ; i < size ; i ++ ) { temp_sum += arr [ i ] ;
if ( N - temp_sum == arr [ i ] * ( size - i - 1 ) ) { return arr [ i ] ; } } return - 1 ; }
let arr = [ 3 , 1 , 10 , 4 , 8 ] ; let size = arr . length ; let N = 16 ; document . write ( findK ( arr , size , N ) ) ;
function existsTriplet ( a , b , c , x , l1 , l2 , l3 ) {
if ( l2 <= l1 && l2 <= l3 ) { temp = l1 ; l1 = l2 ; l2 = temp ; temp = a ; a = b ; b = temp ; } else if ( l3 <= l1 && l3 <= l2 ) { temp = l1 ; l1 = l3 ; l3 = temp ; temp = a ; a = c ; c = temp ; }
for ( var i = 0 ; i < l1 ; i ++ ) {
var j = 0 , k = l3 - 1 ; while ( j < l2 && k >= 0 ) {
if ( a [ i ] + b [ j ] + c [ k ] == x ) return true ; if ( a [ i ] + b [ j ] + c [ k ] < x ) j ++ ; else k -- ; } } return false ; }
var a = [ 2 , 7 , 8 , 10 , 15 ] ; var b = [ 1 , 6 , 7 , 8 ] ; var c = [ 4 , 5 , 5 ] ; var l1 = a . length ; var l2 = b . length ; var l3 = c . length ; var x = 14 ; if ( existsTriplet ( a , b , c , x , l1 , l2 , l3 ) ) document . write ( " " ) ; else document . write ( " " ) ;
function printArr ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] ) ; }
function compare ( num1 , num2 ) {
let A = num1 . toString ( ) ;
let B = num2 . toString ( ) ;
return ( A + B ) . localeCompare ( B + A ) ; }
function printSmallest ( N , arr ) {
for ( let i = 0 ; i < N ; i ++ ) { for ( let j = i + 1 ; j < N ; j ++ ) { if ( compare ( arr [ i ] , arr [ j ] ) > 0 ) { let temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } } }
printArr ( arr , N ) ; }
let arr = [ 5 , 6 , 2 , 9 , 21 , 1 ] ; let N = arr . length ; printSmallest ( N , arr ) ;
function stableSelectionSort ( a , n ) {
for ( let i = 0 ; i < n - 1 ; i ++ ) {
let min = i ; for ( let j = i + 1 ; j < n ; j ++ ) if ( a [ min ] > a [ j ] ) min = j ;
let key = a [ min ] ; while ( min > i ) { a [ min ] = a [ min - 1 ] ; min -- ; } a [ i ] = key ; } } function prletArray ( a , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( a [ i ] + " " ) ; document . write ( " " ) ; }
let a = [ 4 , 5 , 3 , 2 , 4 , 1 ] ; let n = a . length ; stableSelectionSort ( a , n ) ; prletArray ( a , n ) ;
function isPossible ( a , b , n , k ) {
a . sort ( function ( a , b ) { return a - b } ) ;
b . reverse ( ) ;
for ( let i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
let a = [ 2 , 1 , 3 ] ; let b = [ 7 , 8 , 9 ] ; let k = 10 ; let n = a . length ; if ( isPossible ( a , b , n , k ) ) document . write ( " " ) ; else document . write ( " " ) ;
const canReach = ( s , L , R ) => {
let dp = new Array ( s . length ) . fill ( 1 ) ;
let pre = 0 ;
for ( let i = 1 ; i < s . length ; i ++ ) {
if ( i >= L ) { pre += dp [ i - L ] ; }
if ( i > R ) { pre -= dp [ i - R - 1 ] ; } dp [ i ] = ( pre > 0 ) && ( s [ i ] == ' ' ) ; }
return dp [ s . length - 1 ] ; }
let S = " " ; let L = 2 , R = 3 ; if ( canReach ( S , L , R ) ) document . write ( " " ) ; else document . write ( " " ) ;
function maxXORUtil ( arr , N , xrr , orr ) {
if ( N == 0 ) return xrr ^ orr ;
let x = maxXORUtil ( arr , N - 1 , xrr ^ orr , arr [ N - 1 ] ) ;
let y = maxXORUtil ( arr , N - 1 , xrr , orr arr [ N - 1 ] ) ;
return Math . max ( x , y ) ; }
function maximumXOR ( arr , N ) {
return maxXORUtil ( arr , N , 0 , 0 ) ; }
let arr = [ 1 , 5 , 7 ] ; let N = arr . length ; document . write ( maximumXOR ( arr , N ) ) ;
let N = 100000 + 5 ;
let visited = new Array ( N ) ; visited . fill ( 0 ) ;
function construct_tree ( weights , n ) { let minimum = Number . MAX_VALUE ; let maximum = Number . MIN_VALUE ; for ( let i = 0 ; i < weights . length ; i ++ ) { minimum = Math . min ( minimum , weights [ i ] ) ; maximum = Math . max ( maximum , weights [ i ] ) ; }
if ( minimum == maximum ) {
document . write ( " " ) ; return ; }
else {
document . write ( " " + " " ) ; }
let root = weights [ 0 ] ;
visited [ 1 ] = 1 ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] != root && visited [ i + 1 ] == 0 ) { document . write ( 1 + " " + ( i + 1 ) + " " ) ;
visited [ i + 1 ] = 1 ; } }
let notroot = 0 ; for ( let i = 0 ; i < n ; i ++ ) { if ( weights [ i ] != root ) { notroot = i + 1 ; break ; } }
for ( let i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] == root && visited [ i + 1 ] == 0 ) { document . write ( notroot + " " + ( i + 1 ) + " " ) ; visited [ i + 1 ] = 1 ; } } }
let weights = [ 1 , 2 , 1 , 2 , 5 ] ; let n = weights . length ;
construct_tree ( weights , n ) ;
function minCost ( s , k ) {
var n = s . length ;
var ans = 0 ;
for ( var i = 0 ; i < k ; i ++ ) {
var a = new Array ( 26 ) . fill ( 0 ) ; for ( var j = i ; j < n ; j += k ) { a [ s [ j ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; }
var min_cost = 1000000000 ;
for ( var ch = 0 ; ch < 26 ; ch ++ ) { var cost = 0 ;
for ( var tr = 0 ; tr < 26 ; tr ++ ) cost += Math . abs ( ch - tr ) * a [ tr ] ;
min_cost = Math . min ( min_cost , cost ) ; }
ans += min_cost ; }
document . write ( ans ) ; }
var S = " " ; var K = 3 ;
minCost ( S , K ) ;
function minAbsDiff ( N ) { if ( N % 4 == 0 N % 4 == 3 ) { return 0 ; } return 1 ; }
var N = 6 ; document . write ( minAbsDiff ( N ) ) ;
let N = 10000 ;
let adj = new Array ( N ) ; let used = new Array ( N ) ; used . fill ( 0 ) ; let max_matching = 0 ;
function AddEdge ( u , v ) {
adj [ u ] . push ( v ) ;
adj [ v ] . push ( u ) ; }
function Matching_dfs ( u , p ) { for ( let i = 0 ; i < adj [ u ] . length ; i ++ ) {
if ( adj [ u ] [ i ] != p ) { Matching_dfs ( adj [ u ] [ i ] , u ) ; } }
if ( used [ u ] == 0 && used [ p ] == 0 && p != 0 ) {
max_matching ++ ; used [ u ] = used [ p ] = 1 ; } }
function maxMatching ( ) {
Matching_dfs ( 1 , 0 ) ;
document . write ( max_matching + " " ) ; }
for ( let i = 0 ; i < adj . length ; i ++ ) adj [ i ] = [ ] ;
AddEdge ( 1 , 2 ) ; AddEdge ( 1 , 3 ) ; AddEdge ( 3 , 4 ) ; AddEdge ( 3 , 5 ) ;
maxMatching ( ) ;
function getMinCost ( A , B , N ) { let mini = Number . MAX_VALUE ; for ( let i = 0 ; i < N ; i ++ ) { mini = Math . min ( mini , Math . min ( A [ i ] , B [ i ] ) ) ; }
return mini * ( 2 * N - 1 ) ; }
let N = 3 ; let A = [ 1 , 4 , 2 ] ; let B = [ 10 , 6 , 12 ] ; document . write ( getMinCost ( A , B , N ) ) ;
function printVector ( arr ) { if ( arr . length != 1 ) {
for ( var i = 0 ; i < arr . length ; i ++ ) { document . write ( arr [ i ] + " " ) ; } document . write ( " " ) ; } }
function findWays ( arr , i , n ) {
if ( n == 0 ) printVector ( arr ) ;
for ( var j = i ; j <= n ; j ++ ) {
arr . push ( j ) ;
findWays ( arr , j , n - j ) ;
arr . pop ( ) ; } }
var n = 4 ;
var arr = [ ] ;
findWays ( arr , 1 , n ) ;
function Maximum_subsequence ( A , N ) {
var frequency = new Map ( ) ;
var max_freq = 0 ; for ( var i = 0 ; i < N ; i ++ ) {
if ( value > max_freq ) { max_freq = value ; } } ) ;
document . write ( max_freq ) ; }
var arr = [ 5 , 2 , 6 , 5 , 2 , 4 , 5 , 2 ] ; var N = arr . length ; Maximum_subsequence ( arr , N ) ;
function DivideString ( s , n , k ) { var i , c = 0 , no = 1 ; var c1 = 0 , c2 = 0 ;
var fr = new Array ( 26 ) . fill ( 0 ) ; var ans = [ ] ; for ( i = 0 ; i < n ; i ++ ) { fr [ s [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] ++ ; } var ch = " " , ch1 = " " ; for ( i = 0 ; i < 26 ; i ++ ) {
if ( fr [ i ] === k ) { c ++ ; }
if ( fr [ i ] > k && fr [ i ] !== 2 * k ) { c1 ++ ; ch = String . fromCharCode ( i + " " . charCodeAt ( 0 ) ) ; } if ( fr [ i ] === 2 * k ) { c2 ++ ; ch1 = String . fromCharCode ( i + " " . charCodeAt ( 0 ) ) ; } } for ( i = 0 ; i < n ; i ++ ) ans . push ( " " ) ; var mp = { } ; if ( c % 2 === 0 c1 > 0 c2 > 0 ) { for ( i = 0 ; i < n ; i ++ ) {
if ( fr [ s [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] === k ) { if ( mp . hasOwnProperty ( s [ i ] ) ) { ans [ i ] = " " ; } else { if ( no <= parseInt ( c / 2 ) ) { ans [ i ] = " " ; no ++ ; mp [ s [ i ] ] = 1 ; } } } }
if ( c % 2 === 1 && c1 > 0 ) { no = 1 ; for ( i = 0 ; i < n ; i ++ ) { if ( s [ i ] === ch && no <= k ) { ans [ i ] = " " ; no ++ ; } } }
if ( c % 2 === 1 && c1 === 0 ) { no = 1 ; var flag = 0 ; for ( i = 0 ; i < n ; i ++ ) { if ( s [ i ] === ch1 && no <= k ) { ans [ i ] = " " ; no ++ ; } if ( fr [ s [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] === k && flag === 0 && ans [ i ] === " " ) { ans [ i ] = " " ; flag = 1 ; } } } document . write ( ans . join ( " " ) ) ; } else {
document . write ( " " ) ; } }
var S = " " ; var N = S . length ; var K = 1 ; DivideString ( S , N , K ) ;
function check ( S , prices , type , n ) {
for ( let j = 0 ; j < n ; j ++ ) { for ( let k = j + 1 ; k < n ; k ++ ) {
if ( ( type [ j ] == 0 && type [ k ] == 1 ) || ( type [ j ] == 1 && type [ k ] == 0 ) ) { if ( prices [ j ] + prices [ k ] <= S ) { return " " ; } } } } return " " ; }
let prices = [ 3 , 8 , 6 , 5 ] ; let type = [ 0 , 1 , 1 , 0 ] ; let S = 10 ; let n = 4 ;
document . write ( check ( S , prices , type , n ) ) ;
function getLargestSum ( N ) {
for ( let i = 1 ; i * i <= N ; i ++ ) { for ( let j = i + 1 ; j * j <= N ; j ++ ) {
let k = parseInt ( N / j , 10 ) ; let a = k * i ; let b = k * j ;
if ( a <= N && b <= N && a * b % ( a + b ) == 0 )
max_sum = Math . max ( max_sum , a + b ) ; } }
return max_sum ; }
let N = 25 ; let max_sum = getLargestSum ( N ) ; document . write ( max_sum + " " ) ;
function encryptString ( str , n ) { let i = 0 , cnt = 0 ; let encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- > 0 ) encryptedStr += str [ i ] ; i ++ ; } return encryptedStr ; }
let str = " " ; let n = str . length ; document . write ( encryptString ( str , n ) ) ;
function minDiff ( n , x , A ) { var mn = A [ 0 ] , mx = A [ 0 ] ;
for ( var i = 0 ; i < n ; ++ i ) { mn = Math . min ( mn , A [ i ] ) ; mx = Math . max ( mx , A [ i ] ) ; }
return Math . max ( 0 , mx - mn - 2 * x ) ; }
var n = 3 , x = 3 ; var A = [ 1 , 3 , 6 ] ;
document . write ( minDiff ( n , x , A ) ) ;
function swapCount ( s ) { let chars = s . split ( ' ' ) ;
let countLeft = 0 , countRight = 0 ;
let swap = 0 , imbalance = 0 ; for ( let i = 0 ; i < chars . length ; i ++ ) { if ( chars [ i ] == ' ' ) {
countLeft ++ ; if ( imbalance > 0 ) {
swap += imbalance ;
imbalance -- ; } } else if ( chars [ i ] == ' ' ) {
countRight ++ ;
imbalance = ( countRight - countLeft ) ; } } return swap ; }
let s = " " ; document . write ( swapCount ( s ) + " " ) ; s = " " ; document . write ( swapCount ( s ) ) ;
function longestSubSequence ( A , N ) {
let dp = new Array ( N ) ; for ( let i = 0 ; i < N ; i ++ ) {
dp [ i ] = 1 ; for ( let j = 0 ; j < i ; j ++ ) {
if ( A [ j ] [ 0 ] < A [ i ] [ 0 ] && A [ j ] [ 1 ] > A [ i ] [ 1 ] ) { dp [ i ] = Math . max ( dp [ i ] , dp [ j ] + 1 ) ; } } }
document . write ( dp [ N - 1 ] + " " ) ; }
let A = [ [ 1 , 2 ] , [ 2 , 2 ] , [ 3 , 1 ] ] ; let N = A . length ;
longestSubSequence ( A , N ) ;
function findWays ( N , dp ) {
if ( N == 0 ) { return 1 ; }
if ( dp [ N ] != - 1 ) { return dp [ N ] ; } let cnt = 0 ;
for ( let i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i , dp ) ; } }
return dp [ N ] = cnt ; }
let N = 4 ;
let dp = new Array ( N + 1 ) ; for ( let i = 0 ; i < dp . length ; i ++ ) dp [ i ] = - 1 ;
document . write ( findWays ( N , dp ) ) ;
function findWays ( N ) {
let dp = new Array ( N + 1 ) ; dp [ 0 ] = 1 ;
for ( let i = 1 ; i <= N ; i ++ ) { dp [ i ] = 0 ;
for ( let j = 1 ; j <= 6 ; j ++ ) { if ( i - j >= 0 ) { dp [ i ] = dp [ i ] + dp [ i - j ] ; } } }
document . write ( dp [ N ] ) ; }
let N = 4 ;
findWays ( N ) ;
let INF = ( 1e9 + 9 ) ;
class Node { constructor ( ) { } } function TrieNode ( ) { let temp = new Node ( ) ; temp . child = new Node ( 26 ) ; for ( let i = 0 ; i < 26 ; i ++ ) { temp . child [ i ] = null ; } return temp ; }
function insert ( idx , s , root ) { let temp = root ; for ( let i = idx ; i < s . length ; i ++ ) {
if ( temp . child [ s [ i ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] == null )
temp . child [ s [ i ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] = new TrieNode ( ) ; temp = temp . child [ s [ i ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] ; } }
function minCuts ( S1 , S2 ) { let n1 = S1 . length ; let n2 = S2 . length ;
let root = new TrieNode ( ) ; for ( let i = 0 ; i < n2 ; i ++ ) {
insert ( i , S2 , root ) ; }
let dp = new Array ( n1 + 1 ) ; dp . fill ( INF ) ;
dp [ 0 ] = 0 ; for ( let i = 0 ; i < n1 ; i ++ ) {
let temp = root ; for ( let j = i + 1 ; j <= n1 ; j ++ ) { if ( temp . child [ S1 [ j - 1 ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] == null )
break ;
dp [ j ] = Math . min ( dp [ j ] , dp [ i ] + 1 ) ;
temp = temp . child [ S1 [ j - 1 ] . charCodeAt ( ) - ' ' . charCodeAt ( ) ] ; } }
if ( dp [ n1 ] >= INF ) return - 1 ; else return dp [ n1 ] ; }
let S1 = " " ; let S2 = " " ; document . write ( minCuts ( S1 , S2 ) ) ;
function largestSquare ( matrix , R , C , q_i , q_j , K , Q ) { let countDP = new Array ( R ) ; for ( let i = 0 ; i < R ; i ++ ) { countDP [ i ] = new Array ( C ) ; for ( let j = 0 ; j < C ; j ++ ) countDP [ i ] [ j ] = 0 ; }
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] ; for ( let i = 1 ; i < R ; i ++ ) countDP [ i ] [ 0 ] = countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ; for ( let j = 1 ; j < C ; j ++ ) countDP [ 0 ] [ j ] = countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ; for ( let i = 1 ; i < R ; i ++ ) for ( let j = 1 ; j < C ; j ++ ) countDP [ i ] [ j ] = matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ;
for ( let q = 0 ; q < Q ; q ++ ) { let i = q_i [ q ] ; let j = q_j [ q ] ; let min_dist = Math . min ( Math . min ( i , j ) , Math . min ( R - i - 1 , C - j - 1 ) ) ; let ans = - 1 , l = 0 , u = min_dist ;
while ( l <= u ) { let mid = Math . floor ( ( l + u ) / 2 ) ; let x1 = i - mid , x2 = i + mid ; let y1 = j - mid , y2 = j + mid ;
let count = countDP [ x2 ] [ y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 ] [ y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 ] [ y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 ] [ y1 - 1 ] ;
if ( count <= K ) { ans = 2 * mid + 1 ; l = mid + 1 ; } else u = mid - 1 ; } document . write ( ans + " " ) ; } }
let matrix = [ [ 1 , 0 , 1 , 0 , 0 ] , [ 1 , 0 , 1 , 1 , 1 ] , [ 1 , 1 , 1 , 1 , 1 ] , [ 1 , 0 , 0 , 1 , 0 ] ] ; let K = 9 , Q = 1 ; let q_i = [ 1 ] ; let q_j = [ 2 ] ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ;
