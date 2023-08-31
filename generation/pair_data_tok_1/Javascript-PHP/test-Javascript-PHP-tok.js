function Loss ( SP , P ) { var loss = 0 ; loss = ( 2 * P * P * SP ) / ( 100 * 100 - P * P ) ; document . write ( " " + loss . toFixed ( 3 ) ) ; }
var SP = 2400 , P = 30 ;
Loss ( SP , P ) ;
let MAXN = 1000001 ;
let spf = new Array ( MAXN ) ;
let hash1 = new Array ( MAXN ) ;
function sieve ( ) { spf [ 1 ] = 1 ; for ( let i = 2 ; i < MAXN ; i ++ )
spf [ i ] = i ;
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
function partitions ( n ) { var p = Array ( n + 1 ) . fill ( 0 ) ;
p [ 0 ] = 1 ; for ( i = 1 ; i <= n ; ++ i ) { var k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 != 0 ? 1 : - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) { k *= - 1 ; } else { k = 1 - k ; } } } return p [ n ] ; }
var N = 20 ; document . write ( partitions ( N ) ) ;
function countPaths ( n , m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
let n = 3 , m = 2 ; document . write ( " " + countPaths ( n , m ) ) ;
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
function countChar ( str , x ) { let count = 0 ; let n = 10 ; for ( let i = 0 ; i < str . length ; i ++ ) if ( str [ i ] == x ) count ++ ;
let repetitions = n / str . length ; count = count * repetitions ;
for ( let i = 0 ; i < n % str . length ; i ++ ) { if ( str [ i ] == x ) count ++ ; } return count ; }
let str = " " ; document . write ( countChar ( str , ' ' ) ) ;
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
function isValidISBN ( isbn ) {
let n = isbn . length ; if ( n != 10 ) return false ;
let sum = 0 ; for ( let i = 0 ; i < 9 ; i ++ ) { let digit = isbn [ i ] - ' ' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
let last = isbn [ 9 ] ; if ( last != ' ' && ( last < ' ' last > ' ' ) ) return false ;
sum += ( ( last == ' ' ) ? 10 : ( last - ' ' ) ) ;
return ( sum % 11 == 0 ) ; }
let isbn = " " ; if ( isValidISBN ( isbn ) ) document . write ( " " ) ; else document . write ( " " ) ;
var d = 10 ; var a ;
a = parseInt ( ( 360 - ( 6 * d ) ) / 4 ) ;
document . write ( a + " " + ( a + d ) + " " + ( a + ( 2 * d ) ) + " " + ( a + ( 3 * d ) ) ) ;
function distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) { let x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = Math . abs ( ( c2 * z1 + d2 ) ) / ( Math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; document . write ( " " + d ) ; } else document . write ( " " ) ; }
let a1 = 1 ; let b1 = 2 ; let c1 = - 1 ; let d1 = 1 ; let a2 = 3 ; let b2 = 6 ; let c2 = - 3 ; let d2 = - 4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ;
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
let L , R ;
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
function SieveOfSundaram ( n ) {
let nNew = ( n - 1 ) / 2 ;
for ( let i = 0 ; i < nNew + 1 ; i ++ ) marked [ i ] = false ;
for ( let i = 1 ; i <= nNew ; i ++ ) for ( let j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) document . write ( 2 + " " ) ;
for ( let i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) document . write ( 2 * i + 1 + " " ) ; return - 1 ; }
let n = 20 ; SieveOfSundaram ( n ) ;
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
function areElementsContiguous ( arr , n ) {
arr . sort ( function ( a , b ) { return a - b } ) ;
for ( let i = 1 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] > 1 ) return false ; return true ; }
let arr = [ 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 ] ; let n = arr . length ; if ( areElementsContiguous ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function findLargestd ( S , n ) { let found = false ;
S . sort ( ) ;
for ( let i = n - 1 ; i >= 0 ; i -- ) { for ( let j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( let k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( let l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return Number . MAX_VALUE ; return - 1 ; }
let S = [ 2 , 3 , 5 , 7 , 12 ] ; let n = S . length ; let ans = findLargestd ( S , n ) ; if ( ans == Number . MAX_VALUE ) document . write ( " " ) ; else document . write ( " " + " " + ans ) ;
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
function pushZerosToEnd ( arr , n ) {
let count = 0 ;
for ( let i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
let arr = [ 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 ] ; let n = arr . length ; pushZerosToEnd ( arr , n ) ; document . write ( " " ) ; for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ;
function printArray ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; }
function RearrangePosNeg ( arr , n ) { let key , j ; for ( let i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
let arr = [ - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 ] ; let n = arr . length ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ;
function findElements ( arr , n ) {
for ( let i = 0 ; i < n ; i ++ ) { let count = 0 ; for ( let j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) document . write ( arr [ i ] + " " ) ; } }
let arr = [ 2 , - 6 , 3 , 5 , 1 ] ; let n = arr . length ; findElements ( arr , n ) ;
function findElements ( arr , n ) { arr . sort ( ) ; for ( let i = 0 ; i < n - 2 ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 2 , - 6 , 3 , 5 , 1 ] ; let n = arr . length ; findElements ( arr , n ) ;
function findElements ( arr , n ) { let first = Number . MIN_VALUE ; let second = Number . MAX_VALUE ; for ( let i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( let i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 2 , - 6 , 3 , 5 , 1 ] ; let n = arr . length ; findElements ( arr , n ) ;
function findFirstMissing ( array , start , end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; let mid = parseInt ( ( start + end ) / 2 , 10 ) ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
let arr = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 ] ; let n = arr . length ; document . write ( " " + findFirstMissing ( arr , 0 , n - 1 ) ) ;
function FindMaxSum ( arr , n ) { let incl = arr [ 0 ] ; let excl = 0 ; let excl_new ; let i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
let arr = [ 5 , 5 , 10 , 100 , 10 , 5 ] ; document . write ( FindMaxSum ( arr , arr . length ) ) ;
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
function isMajority ( arr , n , x ) { let i , last_index = 0 ;
last_index = ( n % 2 == 0 ) ? parseInt ( n / 2 , 10 ) : parseInt ( n / 2 , 10 ) + 1 ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + parseInt ( n / 2 , 10 ) ] == x ) return true ; } return false ; }
let arr = [ 1 , 2 , 3 , 4 , 4 , 4 , 4 ] ; let n = arr . length ; let x = 4 ; if ( isMajority ( arr , n , x ) == true ) document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ; else document . write ( x + " " + parseInt ( n / 2 , 10 ) + " " ) ;
function cutRod ( price , n ) { let val = new Array ( n + 1 ) ; val [ 0 ] = 0 ;
for ( let i = 1 ; i <= n ; i ++ ) { let max_val = Number . MIN_VALUE ; for ( let j = 0 ; j < i ; j ++ ) max_val = Math . max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
let arr = [ 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 ] ; let size = arr . length ; document . write ( " " + cutRod ( arr , size ) + " " ) ;
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
function nextPerfectCube ( N ) { let nextN = Math . floor ( Math . cbrt ( N ) ) + 1 ; return nextN * nextN * nextN ; }
let n = 35 ; document . write ( nextPerfectCube ( n ) ) ;
function findpos ( n ) { var pos = 0 ; for ( i = 0 ; i < n . length ; i ++ ) { switch ( n . charAt ( i ) ) {
case ' ' : pos = pos * 4 + 1 ; break ;
case ' ' : pos = pos * 4 + 2 ; break ;
case ' ' : pos = pos * 4 + 3 ; break ;
case ' ' : pos = pos * 4 + 4 ; break ; } } return pos ; }
var n = " " ; document . write ( findpos ( n ) ) ;
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
let N = 6 ; let Even = Math . floor ( N / 2 ) ; let Odd = N - Even ; document . write ( Even * Odd ) ;
function steps ( str , n ) {
var flag ; var x = 0 ;
for ( var i = 0 ; i < str . length ; i ++ ) {
if ( x == 0 ) flag = true ;
if ( x == n - 1 ) flag = false ;
for ( var j = 0 ; j < x ; j ++ ) document . write ( " " ) ; document . write ( str [ i ] + " " ) ;
if ( flag == true ) x ++ ; else x -- ; } }
var n = 4 ; var str = " " ; document . write ( " " + str + " " ) ; document . write ( " " + n + " " ) ;
steps ( str , n ) ;
function isDivisible ( str , k ) { let n = str . length ; let c = 0 ;
for ( let i = 0 ; i < k ; i ++ ) if ( str [ n - i - 1 ] == ' ' ) c ++ ;
return ( c == k ) ; }
let str1 = " " ; let k = 2 ; if ( isDivisible ( str1 , k ) == true ) document . write ( " " + " " ) ; else document . write ( " " ) ;
let str2 = " " ; k = 2 ; if ( isDivisible ( str2 , k ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function isNumber ( s ) { for ( let i = 0 ; i < s . length ; i ++ ) if ( s [ i ] < ' ' s [ i ] > ' ' ) return false ; return true ; }
let str = " " ;
if ( isNumber ( str ) ) document . write ( " " ) ;
else document . write ( " " ) ;
function reverse ( str , len ) { if ( len == str . length ) { return ; } reverse ( str , len + 1 ) ; document . write ( str [ len ] ) ; }
let a = " " ; reverse ( a , 0 ) ;
function polyarea ( n , r ) {
if ( r < 0 && n < 0 ) return - 1 ;
var A = ( ( r * r * n ) * Math . sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
var r = 9 , n = 6 ; document . write ( polyarea ( n , r ) . toFixed ( 5 ) ) ;
function findPCSlope ( m ) { return - 1.0 / m ; }
let m = 2.0 ; document . write ( findPCSlope ( m ) ) ;
area_of_segment ( radius , angle ) {
let area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
let area_of_triangle = 1 / 2 * ( radius * radius ) * Math . sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
let radius = 10.0 , angle = 90.0 ; document . write ( " " + area_of_segment ( radius , angle ) + " " ) ; document . write ( " " + area_of_segment ( radius , ( 360 - angle ) ) ) ;
function SectorArea ( radius , angle ) { if ( angle >= 360 ) document . write ( " " ) ;
else { let sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; document . write ( sector ) ; } }
let radius = 9 ; let angle = 60 ; SectorArea ( radius , angle ) ;
function insertionSortRecursive ( arr , n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
let last = arr [ n - 1 ] ; let j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
let arr = [ 12 , 11 , 13 , 5 , 6 ] ; insertionSortRecursive ( arr , arr . length ) ; for ( let i = 0 ; i < arr . length ; i ++ ) { document . write ( arr [ i ] + " " ) ; }
function isWaveArray ( arr , n ) { let result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( let i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( let i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
let arr = [ 1 , 3 , 2 , 4 ] ; let n = arr . length ; if ( isWaveArray ( arr , n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
var mod = 1000000007 ;
function sumOddFibonacci ( n ) { var Sum = Array ( n + 1 ) . fill ( 0 ) ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
var n = 6 ; document . write ( sumOddFibonacci ( n ) ) ;
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
function LCIS ( arr1 , n , arr2 , m ) {
let table = [ ] ; for ( let j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
let current = 0 ;
for ( let j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
let result = 0 ; for ( let i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
let arr1 = [ 3 , 4 , 9 , 1 ] ; let arr2 = [ 5 , 3 , 8 , 9 , 10 , 2 , 1 ] ; let n = arr1 . length ; let m = arr2 . length ; document . write ( " " + LCIS ( arr1 , n , arr2 , m ) ) ;
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
function convert ( s ) { var n = s . length ; var s1 = " " ; s1 = s1 + s . charAt ( 0 ) . toLowerCase ( ) ; for ( i = 1 ; i < n ; i ++ ) {
if ( s . charAt ( i ) == ' ' && i < n ) {
s1 = s1 + " " + s . charAt ( i + 1 ) . toLowerCase ( ) ; i ++ ; }
else s1 = s1 + s . charAt ( i ) . toUpperCase ( ) ; }
return s1 ; }
var str = " " ; document . write ( convert ( str ) ) ;
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
function middleOfThree ( a , b , c ) {
function middleOfThree ( $a , $b , $c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && a < c ) || ( c < a && a < b ) ) return a ; else return c ; }
let a = 20 , b = 30 , c = 40 ; document . write ( middleOfThree ( a , b , c ) ) ;
let INF = Number . MAX_VALUE , N = 4 ;
function minCost ( cost ) {
let dist = new Array ( N ) ; dist . fill ( 0 ) ; for ( let i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( let i = 0 ; i < N ; i ++ ) for ( let j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) dist [ j ] = dist [ i ] + cost [ i ] [ j ] ; return dist [ N - 1 ] ; }
let cost = [ [ 0 , 15 , 80 , 90 ] , [ INF , 0 , 40 , 50 ] , [ INF , INF , 0 , 70 ] , [ INF , INF , INF , 0 ] ] ; document . write ( " " + " " + N + " " + minCost ( cost ) ) ;
function numOfways ( n , k ) { let p = 1 ; if ( k % 2 != 0 ) p = - 1 ; return ( Math . pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
let n = 4 , k = 2 ; document . write ( numOfways ( n , k ) ) ;
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
function fib ( n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
function findVertices ( n ) {
return fib ( n + 2 ) ; }
var n = 3 ; document . write ( findVertices ( n ) ) ;
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
let N = 4 ;
function isLowerTriangularMatrix ( mat ) { for ( let i = 0 ; i < N ; i ++ ) for ( let j = i + 1 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != 0 ) return false ; return true ; }
let mat = [ [ 1 , 0 , 0 , 0 ] , [ 1 , 4 , 0 , 0 ] , [ 4 , 6 , 2 , 0 ] , [ 0 , 4 , 7 , 6 ] ] ;
if ( isLowerTriangularMatrix ( mat ) ) document . write ( " " ) ; else document . write ( " " ) ;
let N = 4 ;
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
let MAX = 100 ;
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
function getModulo ( n , d ) { return ( n & ( d - 1 ) ) ; }
n = 6 ;
d = 4 ; document . write ( n + " " + d + " " + getModulo ( n , d ) ) ;
function countSetBits ( n ) { var count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
var i = 9 ; document . write ( countSetBits ( i ) ) ;
function countSetBits ( n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
var n = 9 ;
document . write ( countSetBits ( n ) ) ;
document . write ( ( 4 ) . toString ( 2 ) . split ( ' ' ) . filter ( x => x == ' ' ) . length + " " ) ; document . write ( ( 15 ) . toString ( 2 ) . split ( ' ' ) . filter ( x => x == ' ' ) . length ) ;
var num_to_bits = [ 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 ] ;
function countSetBitsRec ( num ) { var nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
var num = 31 ; document . write ( countSetBitsRec ( num ) ) ;
function getParity ( n ) { var parity = false ; while ( n != 0 ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
var n = 7 ; document . write ( " " + n + " " + ( getParity ( n ) ? " " : " " ) ) ;
function isPowerOfTwo ( n ) { if ( n == 0 ) return false ; return parseInt ( ( Math . ceil ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) ) == parseInt ( ( Math . floor ( ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) ) ) ; }
if ( isPowerOfTwo ( 31 ) ) document . write ( " " ) ; else document . write ( " " ) ; if ( isPowerOfTwo ( 64 ) ) document . write ( " " ) ; else document . write ( " " ) ;
function isPowerOfTwo ( n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 2 != 0 ) return 0 ; n = n / 2 ; } return 1 ; }
isPowerOfTwo ( 31 ) ? document . write ( " " + " " ) : document . write ( " " + " " ) ; isPowerOfTwo ( 64 ) ? document . write ( " " ) : document . write ( " " ) ;
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
function findMinSwaps ( arr , n ) {
let noOfZeroes = [ ] ; let i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
let ar = [ 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ] ; document . write ( findMinSwaps ( ar , ar . length ) ) ;
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
function checkIsAP ( arr , n ) { if ( n == 1 ) return true ;
arr . sort ( ( a , b ) => a - b ) ;
let d = arr [ 1 ] - arr [ 0 ] ; for ( let i = 2 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] != d ) return false ; return true ; }
let arr = [ 20 , 15 , 5 , 0 , 10 ] ; let n = arr . length ; ( checkIsAP ( arr , n ) ) ? ( document . write ( " " + " " ) ) : ( document . write ( " " + " " ) ) ;
function countPairs ( a , n ) {
let mn = Number . MAX_VALUE ; let mx = Number . MIN_VALUE ; for ( let i = 0 ; i < n ; i ++ ) { mn = Math . min ( mn , a [ i ] ) ; mx = Math . max ( mx , a [ i ] ) ; }
let c1 = 0 ;
let c2 = 0 ; for ( let i = 0 ; i < n ; i ++ ) { if ( a [ i ] == mn ) c1 ++ ; if ( a [ i ] == mx ) c2 ++ ; }
if ( mn == mx ) return n * ( n - 1 ) / 2 ; else return c1 * c2 ; }
let a = [ 3 , 2 , 1 , 1 , 3 ] ; let n = a . length ; document . write ( countPairs ( a , n ) ) ;
function findNumbers ( arr , n ) {
sumN = ( n * ( n + 1 ) ) / 2 ;
sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
let sum = 0 ; let sumSq = 0 ; for ( let i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq += Math . pow ( arr [ i ] , 2 ) ; } B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; A = sum - sumN + B ; document . write ( " " + A , " " , B ) ; }
let arr = [ 1 , 2 , 2 , 3 , 4 ] ; n = arr . length ; findNumbers ( arr , n ) ;
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
function minMovesToSort ( arr , n ) { var moves = 0 ; var i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
mn = arr [ i ] ; } return moves ; }
var arr = [ 3 , 5 , 2 , 8 , 4 ] ; var n = arr . length ; document . write ( minMovesToSort ( arr , n ) ) ;
function findOptimalPairs ( arr , N ) { arr . sort ( function ( a , b ) { return a - b ; } ) ;
for ( var i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) document . write ( " " + arr [ i ] + " " + arr [ j ] + " " + " " ) ; }
var arr = [ 9 , 6 , 5 , 1 ] ; var N = arr . length ; findOptimalPairs ( arr , N ) ;
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
var N = 3 ;
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
function MatrixChainOrder ( p , i , j ) { if ( i == j ) return 0 ; var min = Number . MAX_VALUE ;
var k = 0 ; for ( k = i ; k < j ; k ++ ) { var count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
var arr = [ 1 , 2 , 3 , 4 , 3 ] ; var n = arr . length ; document . write ( " " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ;
function getCount ( a , b ) {
if ( b . length % a . length != 0 ) return - 1 ; var count = parseInt ( b . length / a . length ) ;
var str = " " ; for ( i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str == ( b ) ) return count ; return - 1 ; }
var a = " " ; var b = " " ; document . write ( getCount ( a , b ) ) ;
function countPattern ( str ) { let len = str . length ; let oneSeen = false ;
for ( let i = 0 ; i < len ; i ++ ) { let getChar = str [ i ] ;
if ( getChar == ' ' && oneSeen == true ) { if ( str [ i - 1 ] == ' ' ) count ++ ; }
if ( getChar == ' ' && oneSeen == false ) oneSeen = true ;
if ( getChar != ' ' && str [ i ] != ' ' ) oneSeen = false ; } return count ; }
let str = " " ; document . write ( countPattern ( str ) ) ;
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
}
return Math . abs ( area / 2.0 ) ; }
let X = [ 0 , 2 , 4 ] ; let Y = [ 1 , 3 , 7 ] ; let n = X . length ; document . write ( polygonArea ( X , Y , n ) ) ;
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
function countNumbers ( L , R , K ) { if ( K == 9 ) { K = 0 ; }
var totalnumbers = R - L + 1 ;
var factor9 = totalnumbers / 9 ;
var rem = totalnumbers % 9 ;
var ans = factor9 ;
for ( var i = R ; i > R - rem ; i -- ) { var rem1 = i % 9 ; if ( rem1 == K ) { ans ++ ; } } return ans ; }
var L = 10 ; var R = 22 ; var K = 3 ; document . write ( Math . round ( countNumbers ( L , R , K ) ) ) ;
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
}
for ( var i = 3 ; i <= Math . sqrt ( n ) ; i += 2 ) { while ( n % i == 0 ) { max = i ; n = n / i ; } }
if ( n > 2 ) max = n ; return max ; }
function checkUnusual ( n ) {
var factor = largestPrimeFactor ( n ) ;
if ( factor > Math . sqrt ( n ) ) { return true ; } else { return false ; } }
var n = 14 ; if ( checkUnusual ( n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function isHalfReducible ( arr , n , m ) { var frequencyHash = Array ( m + 1 ) . fill ( 0 ) ; var i ; for ( i = 0 ; i < n ; i ++ ) { frequencyHash [ arr [ i ] % ( m + 1 ) ] ++ ; } for ( i = 0 ; i <= m ; i ++ ) { if ( frequencyHash [ i ] >= n / 2 ) break ; } if ( i <= m ) document . write ( " " ) ; else document . write ( " " ) ; }
var arr = [ 8 , 16 , 32 , 3 , 12 ] ; var n = arr . length ; var m = 7 ; isHalfReducible ( arr , n , m ) ;
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
function TrinomialValue ( n , k ) {
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ; }
function printTrinomial ( n ) {
for ( let i = 0 ; i < n ; i ++ ) {
for ( let j = - i ; j <= 0 ; j ++ ) document . write ( TrinomialValue ( i , j ) + " " ) ;
for ( let j = 1 ; j <= i ; j ++ ) document . write ( TrinomialValue ( i , j ) + " " ) ; document . write ( " " ) ; } }
let n = 4 ; printTrinomial ( n ) ;
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
function getMaxMedian ( arr , n , k ) { let size = n + k ;
arr . sort ( ( a , b ) => a - b ) ;
if ( size % 2 == 0 ) { let median = ( arr [ Math . floor ( size / 2 ) - 1 ] + arr [ Math . floor ( size / 2 ) ] ) / 2 ; return median ; }
let median = arr [ Math . floor ( size / 2 ) ] ; return median ; }
function printSorted ( a , b , c ) {
let get_max = Math . max ( a , Math . max ( b , c ) ) ;
let get_min = - Math . max ( - a , Math . max ( - b , - c ) ) ; let get_mid = ( a + b + c ) - ( get_max + get_min ) ; document . write ( get_min + " " + get_mid + " " + get_max ) ; }
let a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ;
function insertionSort ( arr , n ) { let i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
function printArray ( arr , n ) { let i ; for ( i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; document . write ( " " ) ; }
let arr = [ 12 , 11 , 13 , 5 , 6 ] ; let n = arr . length ; insertionSort ( arr , n ) ; printArray ( arr , n ) ;
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
function isVowel ( c ) { return ( c == ' ' c == ' ' c == ' ' c == ' ' c == ' ' ) ; }
function encryptString ( s , n , k ) { var countVowels = 0 ; var countConsonants = 0 ; var ans = " " ;
for ( var l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( var r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s [ r ] ) == true ) countVowels ++ ; else countConsonants ++ ; }
ans += ( countVowels * countConsonants ) . toString ( ) ; } return ans ; }
var s = " " ; var n = s . length ; var k = 2 ; document . write ( encryptString ( s , n , k ) ) ;
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
function middleOfThree ( a , b , c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
let a = 20 , b = 30 , c = 40 ; document . write ( middleOfThree ( a , b , c ) ) ;
function printArr ( arr , n ) { for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] ) ; }
function compare ( num1 , num2 ) {
let A = num1 . toString ( ) ;
let B = num2 . toString ( ) ;
return ( A + B ) . localeCompare ( B + A ) ; }
function printSmallest ( N , arr ) {
for ( let i = 0 ; i < N ; i ++ ) { for ( let j = i + 1 ; j < N ; j ++ ) { if ( compare ( arr [ i ] , arr [ j ] ) > 0 ) { let temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } } }
printArr ( arr , N ) ; }
let arr = [ 5 , 6 , 2 , 9 , 21 , 1 ] ; let N = arr . length ; printSmallest ( N , arr ) ;
function isPossible ( a , b , n , k ) {
a . sort ( function ( a , b ) { return a - b } ) ;
b . reverse ( ) ;
for ( let i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
let a = [ 2 , 1 , 3 ] ; let b = [ 7 , 8 , 9 ] ; let k = 10 ; let n = a . length ; if ( isPossible ( a , b , n , k ) ) document . write ( " " ) ; else document . write ( " " ) ;
function encryptString ( str , n ) { let i = 0 , cnt = 0 ; let encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- > 0 ) encryptedStr += str [ i ] ; i ++ ; } return encryptedStr ; }
let str = " " ; let n = str . length ; document . write ( encryptString ( str , n ) ) ;
function minDiff ( n , x , A ) { var mn = A [ 0 ] , mx = A [ 0 ] ;
for ( var i = 0 ; i < n ; ++ i ) { mn = Math . min ( mn , A [ i ] ) ; mx = Math . max ( mx , A [ i ] ) ; }
return Math . max ( 0 , mx - mn - 2 * x ) ; }
var n = 3 , x = 3 ; var A = [ 1 , 3 , 6 ] ;
document . write ( minDiff ( n , x , A ) ) ;
