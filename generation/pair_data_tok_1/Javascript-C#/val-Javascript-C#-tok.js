function Conversion ( centi ) { let pixels = ( 96 * centi ) / 2.54 ; document . write ( pixels ) ; return 0 ; }
let centi = 15 ; Conversion ( centi )
function xor_operations ( N , arr , M , K ) {
if ( M < 0 M >= N ) return - 1 ;
if ( K < 0 K >= N - M ) return - 1 ;
for ( let p = 0 ; p < M ; p ++ ) {
let temp = [ ] ;
for ( let i = 0 ; i < N ; i ++ ) {
let value = arr [ i ] ^ arr [ i + 1 ] ;
temp . push ( value ) ;
arr [ i ] = temp [ i ] ; } }
let ans = arr [ K ] ; return ans ; }
let N = 5 ;
let arr = [ 1 , 4 , 5 , 6 , 7 ] ; let M = 1 , K = 2 ;
document . write ( xor_operations ( N , arr , M , K ) ) ;
function canBreakN ( n ) {
for ( let i = 2 ; ; i ++ ) {
let m = parseInt ( i * ( i + 1 ) / 2 , 10 ) ;
if ( m > n ) break ; let k = n - m ;
if ( k % i != 0 ) continue ;
document . write ( i ) ; return ; }
document . write ( " " ) ; }
let N = 12 ;
canBreakN ( N ) ;
function findCoprimePair ( N ) {
for ( let x = 2 ; x <= Math . sqrt ( N ) ; x ++ ) { if ( N % x == 0 ) {
while ( N % x == 0 ) { N = Math . floor ( N / x ) ; } if ( N > 1 ) {
document . write ( x + " " + N + " " ) ; return ; } } }
document . write ( - 1 + " " ) ; }
let N = 45 ; findCoprimePair ( N ) ;
N = 25 ; findCoprimePair ( N ) ;
let MAX = 10000 ;
let primes = [ ] ;
function sieveSundaram ( ) {
let marked = Array . from ( { length : MAX / 2 + 1 } , ( _ , i ) => 0 ) ;
for ( let i = 1 ; i <= Math . floor ( ( Math . sqrt ( MAX ) - 1 ) / 2 ) ; i ++ ) { for ( let j = ( i * ( i + 1 ) ) << 1 ; j <= Math . floor ( MAX / 2 ) ; j = j + 2 * i + 1 ) { marked [ j ] = true ; } }
primes . push ( 2 ) ;
for ( let i = 1 ; i <= Math . floor ( MAX / 2 ) ; i ++ ) if ( marked [ i ] == false ) primes . push ( 2 * i + 1 ) ; }
function isWasteful ( n ) { if ( n == 1 ) return false ;
let original_no = n ; let sumDigits = 0 ; while ( original_no > 0 ) { sumDigits ++ ; original_no = Math . floor ( original_no / 10 ) ; } let pDigit = 0 , count_exp = 0 , p = 0 ;
for ( let i = 0 ; primes [ i ] <= Math . floor ( n / 2 ) ; i ++ ) {
while ( n % primes [ i ] == 0 ) {
p = primes [ i ] ; n = Math . floor ( n / p ) ;
count_exp ++ ; }
while ( p > 0 ) { pDigit ++ ; p = Math . floor ( p / 10 ) ; }
while ( count_exp > 1 ) { pDigit ++ ; count_exp = Math . floor ( count_exp / 10 ) ; } }
if ( n != 1 ) { while ( n > 0 ) { pDigit ++ ; n = Math . floor ( n / 10 ) ; } }
return ( pDigit > sumDigits ) ; }
function Solve ( N ) {
for ( let i = 1 ; i < N ; i ++ ) { if ( isWasteful ( i ) ) { document . write ( i + " " ) ; } } }
sieveSundaram ( ) ; let N = 10 ;
Solve ( N ) ;
function printhexaRec ( n ) { if ( n == 0 n == 1 n == 2 n == 3 n == 4 n == 5 ) return 0 ; else if ( n == 6 ) return 1 ; else return ( printhexaRec ( n - 1 ) + printhexaRec ( n - 2 ) + printhexaRec ( n - 3 ) + printhexaRec ( n - 4 ) + printhexaRec ( n - 5 ) + printhexaRec ( n - 6 ) ) ; } function printhexa ( n ) { document . write ( printhexaRec ( n ) + " " ) ; }
let n = 11 ; printhexa ( n ) ;
function printhexa ( n ) { if ( n < 0 ) return ;
let first = 0 ; let second = 0 ; let third = 0 ; let fourth = 0 ; let fifth = 0 ; let sixth = 1 ;
let curr = 0 ; if ( n < 6 ) document . write ( first ) ; else if ( n == 6 ) document . write ( sixth ) ; else {
for ( let i = 6 ; i < n ; i ++ ) { curr = first + second + third + fourth + fifth + sixth ; first = second ; second = third ; third = fourth ; fourth = fifth ; fifth = sixth ; sixth = curr ; } } document . write ( curr ) ; }
let n = 11 ; printhexa ( n ) ;
function smallestNumber ( N ) { document . write ( ( N % 9 + 1 ) * Math . pow ( 10 , parseInt ( N / 9 , 10 ) ) - 1 ) ; }
let N = 10 ; smallestNumber ( N ) ;
let compo = [ ] ;
function isComposite ( n ) {
if ( n <= 3 ) return false ;
if ( n % 2 == 0 n % 3 == 0 ) return true ; let i = 5 ; while ( i * i <= n ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) return true ; i = i + 6 ; } return false ; }
function Compositorial_list ( n ) { let l = 0 ; for ( let i = 4 ; i < 1000000 ; i ++ ) { if ( l < n ) { if ( isComposite ( i ) ) { compo . push ( i ) ; l += 1 ; } } } }
function calculateCompositorial ( n ) {
let result = 1 ; for ( let i = 0 ; i < n ; i ++ ) result = result * compo [ i ] ; return result ; }
let n = 5 ;
Compositorial_list ( n ) ; document . write ( calculateCompositorial ( n ) ) ;
let b = new Array ( 50 ) ; b . fill ( 0 ) ;
function PowerArray ( n , k ) {
let count = 0 ;
while ( k > 0 ) { if ( k % n == 0 ) { k = parseInt ( k / n , 10 ) ; count ++ ; }
else if ( k % n == 1 ) { k -= 1 ; b [ count ] ++ ;
if ( b [ count ] > 1 ) { document . write ( - 1 ) ; return 0 ; } }
else { document . write ( - 1 ) ; return 0 ; } }
for ( let i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] != 0 ) { document . write ( i + " " ) ; } } return Number . MIN_VALUE ; }
let N = 3 ; let K = 40 ; PowerArray ( N , K ) ;
function findSum ( N , k ) {
let sum = 0 ; for ( let i = 1 ; i <= N ; i ++ ) {
sum += Math . pow ( i , k ) ; }
return sum ; }
let N = 8 , k = 4 ;
document . write ( findSum ( N , k ) ) ;
function countIndices ( arr , n ) {
var cnt = 0 ;
var max = 0 ; for ( i = 0 ; i < n ; i ++ ) {
if ( max < arr [ i ] ) {
max = arr [ i ] ;
cnt ++ ; } } return cnt ; }
var arr = [ 1 , 2 , 3 , 4 ] ; var n = arr . length ; document . write ( countIndices ( arr , n ) ) ;
bin = [ " " , " " , " " , " " , " " , " " , " " , " " ] ;
function maxFreq ( s ) {
var binary = " " ;
for ( var i = 0 ; i < s . length ; i ++ ) { binary += bin [ s . charAt ( i ) - ' ' ] ; }
binary = binary . substr ( 0 , binary . length - 1 ) ; var count = 1 , prev = - 1 , i , j = 0 ; for ( i = binary . length - 1 ; i >= 0 ; i -- , j ++ )
if ( binary . charAt ( i ) == ' ' ) {
count = Math . max ( count , j - prev ) ; prev = j ; } return count ; }
var octal = " " ; document . write ( maxFreq ( octal ) ) ;
let sz = 100000 ; let isPrime = new Array ( sz + 1 ) ; isPrime . fill ( false ) ;
function sieve ( ) { for ( let i = 0 ; i <= sz ; i ++ ) isPrime [ i ] = true ; isPrime [ 0 ] = isPrime [ 1 ] = false ; for ( let i = 2 ; i * i <= sz ; i ++ ) { if ( isPrime [ i ] ) { for ( let j = i * i ; j < sz ; j += i ) { isPrime [ j ] = false ; } } } }
function findPrimesD ( d ) {
let left = Math . pow ( 10 , d - 1 ) ; let right = Math . pow ( 10 , d ) - 1 ;
for ( let i = left ; i <= right ; i ++ ) {
if ( isPrime [ i ] ) { document . write ( i + " " ) ; } } }
sieve ( ) ; let d = 1 ; findPrimesD ( d ) ;
function Cells ( n , x ) { if ( n <= 0 x <= 0 x > n * n ) return 0 ; var i = 0 , count = 0 ; while ( ++ i * i < x ) if ( x % i == 0 && x <= n * i ) count += 2 ; return i * i == x ? count + 1 : count ; }
var n = 6 , x = 12 ;
document . write ( Cells ( n , x ) ) ;
function maxOfMin ( a , n , S ) {
let mi = Number . MAX_VALUE ;
let s1 = 0 ; for ( let i = 0 ; i < n ; i ++ ) { s1 += a [ i ] ; mi = Math . min ( a [ i ] , mi ) ; }
if ( s1 < S ) return - 1 ;
if ( s1 == S ) return 0 ;
let low = 0 ;
let high = mi ;
let ans = 0 ;
while ( low <= high ) { let mid = parseInt ( ( low + high ) / 2 , 10 ) ;
if ( s1 - ( mid * n ) >= S ) { ans = mid ; low = mid + 1 ; }
else high = mid - 1 ; }
return ans ; }
let a = [ 10 , 10 , 10 , 10 , 10 ] ; let S = 10 ; let n = a . length ; document . write ( maxOfMin ( a , n , S ) ) ;
function Alphabet_N_Pattern ( N ) { var index , side_index , size ;
var Right = 1 , Left = 1 , Diagonal = 2 ;
for ( index = 0 ; index < N ; index ++ ) {
document . write ( Left ++ ) ;
for ( side_index = 0 ; side_index < 2 * index ; side_index ++ ) document . write ( " " ) ;
if ( index != 0 && index != N - 1 ) document . write ( Diagonal ++ ) ; else document . write ( " " ) ;
for ( side_index = 0 ; side_index < 2 * ( N - index - 1 ) ; side_index ++ ) document . write ( " " ) ;
document . write ( Right ++ ) ; document . write ( " " ) ; } }
var Size = 6 ;
Alphabet_N_Pattern ( Size ) ;
function isSumDivides ( N ) { var temp = N ; var sum = 0 ;
while ( temp > 0 ) { sum += temp % 10 ; temp = parseInt ( temp / 10 ) ; } if ( N % sum == 0 ) return 1 ; else return 0 ; }
var N = 12 ; if ( isSumDivides ( N ) == 1 ) document . write ( " " ) ; else document . write ( " " ) ;
function sum ( N ) { var S1 , S2 , S3 ; S1 = ( ( N / 3 ) ) * ( 2 * 3 + ( N / 3 - 1 ) * 3 ) / 2 ; S2 = ( ( N / 4 ) ) * ( 2 * 4 + ( N / 4 - 1 ) * 4 ) / 2 ; S3 = ( ( N / 12 ) ) * ( 2 * 12 + ( N / 12 - 1 ) * 12 ) / 2 ; return S1 + S2 - S3 ; }
var N = 20 ; document . write ( sum ( 12 ) ) ;
function nextGreater ( N ) { var power_of_2 = 1 , shift_count = 0 ;
while ( true ) {
if ( ( ( N >> shift_count ) & 1 ) % 2 == 0 ) break ;
shift_count ++ ;
power_of_2 = power_of_2 * 2 ; }
return ( N + power_of_2 ) ; }
var N = 11 ;
document . write ( " " + nextGreater ( N ) ) ;
function countWays ( n ) {
if ( n == 0 ) return 1 ; if ( n <= 2 ) return n ;
let f0 = 1 , f1 = 1 , f2 = 2 ; let ans = 0 ;
for ( let i = 3 ; i <= n ; i ++ ) { ans = f0 + f1 + f2 ; f0 = f1 ; f1 = f2 ; f2 = ans ; }
return ans ; }
let n = 4 ; document . write ( countWays ( n ) ) ;
const n = 6 , m = 6 ;
function maxSum ( arr ) {
const dp = new Array ( n + 1 ) . fill ( 0 ) . map ( ( ) => new Array ( 3 ) . fill ( 0 ) ) ;
for ( var i = 0 ; i < n ; i ++ ) {
var m1 = 0 , m2 = 0 , m3 = 0 ; for ( var j = 0 ; j < m ; j ++ ) {
if ( parseInt ( j / ( m / 3 ) ) == 0 ) { m1 = Math . max ( m1 , arr [ i ] [ j ] ) ; }
else if ( parseInt ( j / ( m / 3 ) ) == 1 ) { m2 = Math . max ( m2 , arr [ i ] [ j ] ) ; }
else if ( parseInt ( j / ( m / 3 ) ) == 2 ) { m3 = Math . max ( m3 , arr [ i ] [ j ] ) ; } }
dp [ i + 1 ] [ 0 ] = Math . max ( dp [ i ] [ 1 ] , dp [ i ] [ 2 ] ) + m1 ; dp [ i + 1 ] [ 1 ] = Math . max ( dp [ i ] [ 0 ] , dp [ i ] [ 2 ] ) + m2 ; dp [ i + 1 ] [ 2 ] = Math . max ( dp [ i ] [ 1 ] , dp [ i ] [ 0 ] ) + m3 ; }
document . write ( parseInt ( Math . max ( Math . max ( dp [ n ] [ 0 ] , dp [ n ] [ 1 ] ) , dp [ n ] [ 2 ] ) ) + " " ) ; }
arr = [ [ 1 , 3 , 5 , 2 , 4 , 6 ] , [ 6 , 4 , 5 , 1 , 3 , 2 ] , [ 1 , 3 , 5 , 2 , 4 , 6 ] , [ 6 , 4 , 5 , 1 , 3 , 2 ] , [ 6 , 4 , 5 , 1 , 3 , 2 ] , [ 1 , 3 , 5 , 2 , 4 , 6 ] ] ; maxSum ( arr ) ;
function solve ( s ) { n = s . length ;
let dp = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { dp [ i ] = new Array ( n ) ; for ( let j = 0 ; j < n ; j ++ ) dp [ i ] [ j ] = 0 ; }
for ( let len = n - 1 ; len >= 0 ; -- len ) {
for ( let i = 0 ; i + len < n ; ++ i ) {
let j = i + len ;
if ( i == 0 && j == n - 1 ) { if ( s [ i ] == s [ j ] ) dp [ i ] [ j ] = 2 ; else if ( s [ i ] != s [ j ] ) dp [ i ] [ j ] = 1 ; } else { if ( s [ i ] == s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i ] [ j ] += dp [ i - 1 ] [ j ] ; } if ( j + 1 <= n - 1 ) { dp [ i ] [ j ] += dp [ i ] [ j + 1 ] ; } if ( i - 1 < 0 j + 1 >= n ) {
dp [ i ] [ j ] += 1 ; } } else if ( s [ i ] != s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i ] [ j ] += dp [ i - 1 ] [ j ] ; } if ( j + 1 <= n - 1 ) { dp [ i ] [ j ] += dp [ i ] [ j + 1 ] ; } if ( i - 1 >= 0 && j + 1 <= n - 1 ) {
dp [ i ] [ j ] -= dp [ i - 1 ] [ j + 1 ] ; } } } } } let ways = [ ] ; for ( let i = 0 ; i < n ; ++ i ) { if ( i == 0 i == n - 1 ) {
ways . push ( 1 ) ; } else {
let total = dp [ i - 1 ] [ i + 1 ] ; ways . push ( total ) ; } } for ( let i = 0 ; i < ways . length ; ++ i ) { document . write ( ways [ i ] + " " ) ; } }
let s = " " . split ( " " ) ; solve ( s ) ;
function getChicks ( n ) {
let size = Math . max ( n , 7 ) ; let dp = new Array ( size ) ; dp . fill ( 0 ) ; dp [ 0 ] = 0 ; dp [ 1 ] = 1 ;
for ( let i = 2 ; i < 6 ; i ++ ) { dp [ i ] = dp [ i - 1 ] * 3 ; }
dp [ 6 ] = 726 ;
for ( let i = 8 ; i <= n ; i ++ ) {
dp [ i ] = ( dp [ i - 1 ] - ( 2 * parseInt ( dp [ i - 6 ] / 3 , 10 ) ) ) * 3 ; } return dp [ n ] ; }
let n = 3 ; document . write ( getChicks ( n ) ) ;
function getChicks ( n ) { let chicks = Math . pow ( 3 , n - 1 ) ; return chicks ; }
let n = 3 ; document . write ( getChicks ( n ) ) ;
let n = 3 ;
let dp = new Array ( n ) ;
let v = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { dp [ i ] = new Array ( n ) ; v [ i ] = new Array ( n ) ; for ( let j = 0 ; j < n ; j ++ ) { dp [ i ] [ j ] = 0 ; v [ i ] [ j ] = 0 ; } }
function minSteps ( i , j , arr ) {
if ( i == n - 1 && j == n - 1 ) { return 0 ; } if ( i > n - 1 j > n - 1 ) { return 9999999 ; }
if ( v [ i ] [ j ] == 1 ) { return dp [ i ] [ j ] ; } v [ i ] [ j ] = 1 ; dp [ i ] [ j ] = 9999999 ;
for ( let k = Math . max ( 0 , arr [ i ] [ j ] + j - n + 1 ) ; k <= Math . min ( n - i - 1 , arr [ i ] [ j ] ) ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , minSteps ( i + k , j + arr [ i ] [ j ] - k , arr ) ) ; } dp [ i ] [ j ] ++ ; return dp [ i ] [ j ] ; }
let arr = [ [ 4 , 1 , 2 ] , [ 1 , 1 , 1 ] , [ 2 , 1 , 1 ] ] ; let ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) { document . write ( - 1 ) ; } else { document . write ( ans ) ; }
let n = 3 ;
let dp = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { dp [ i ] = new Array ( n ) ; }
let v = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { v [ i ] = new Array ( n ) ; }
function minSteps ( i , j , arr ) {
if ( i == n - 1 && j == n - 1 ) { return 0 ; } if ( i > n - 1 j > n - 1 ) { return 9999999 ; }
if ( v [ i ] [ j ] == 1 ) { return dp [ i ] [ j ] ; } v [ i ] [ j ] = 1 ;
dp [ i ] [ j ] = 1 + Math . min ( minSteps ( i + arr [ i ] [ j ] , j , arr ) , minSteps ( i , j + arr [ i ] [ j ] , arr ) ) ; return dp [ i ] [ j ] ; }
let arr = [ [ 2 , 1 , 2 ] , [ 1 , 1 , 1 ] , [ 1 , 1 , 1 ] ] ; let ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) { document . write ( - 1 ) ; } else { document . write ( ans ) ; }
let MAX = 1001 ; let dp = new Array ( MAX ) ; for ( let i = 0 ; i < MAX ; i ++ ) { dp [ i ] = new Array ( MAX ) ; for ( let j = 0 ; j < MAX ; j ++ ) dp [ i ] [ j ] = - 1 ; }
function MaxProfit ( treasure , color , n , k , col , A , B ) {
return dp [ k ] [ col ] = 0 ; if ( dp [ k ] [ col ] != - 1 ) return dp [ k ] [ col ] ; let sum = 0 ;
if ( col == color [ k ] ) sum += Math . max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += Math . max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return dp [ k ] [ col ] = sum ; }
let A = - 5 , B = 7 ; let treasure = [ 4 , 8 , 2 , 9 ] ; let color = [ 2 , 2 , 6 , 2 ] ; let n = color . length ; document . write ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) ;
function printTetra ( n ) { let dp = new Array ( n + 5 ) ;
dp [ 0 ] = 0 ; dp [ 1 ] = dp [ 2 ] = 1 ; dp [ 3 ] = 2 ; for ( let i = 4 ; i <= n ; i ++ ) dp [ i ] = dp [ i - 1 ] + dp [ i - 2 ] + dp [ i - 3 ] + dp [ i - 4 ] ; document . write ( dp [ n ] ) ; }
let n = 10 ; printTetra ( n ) ;
function maxSum1 ( arr , n ) { let dp = new Array ( n ) ; let maxi = 0 ; for ( i = 0 ; i < n - 1 ; i ++ ) {
dp [ i ] = arr [ i ] ;
if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( i = 2 ; i < n - 1 ; i ++ ) {
for ( j = 0 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < dp [ j ] + arr [ i ] ) { dp [ i ] = dp [ j ] + arr [ i ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; }
function maxSum2 ( arr , n ) { let dp = new Array ( n ) ; let maxi = 0 ; for ( i = 1 ; i < n ; i ++ ) { dp [ i ] = arr [ i ] ; if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( i = 3 ; i < n ; i ++ ) {
for ( j = 1 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < arr [ i ] + dp [ j ] ) { dp [ i ] = arr [ i ] + dp [ j ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; } function findMaxSum ( arr , n ) { let t = Math . max ( maxSum1 ( arr , n ) , maxSum2 ( arr , n ) ) ; return t ; }
let arr = [ 1 , 2 , 3 , 1 ] ; let n = arr . length ; document . write ( findMaxSum ( arr , n ) ) ;
function permutationCoeff ( n , k ) { let P = new Array ( n + 2 ) ; for ( let i = 0 ; i < n + 2 ; i ++ ) { P [ i ] = new Array ( k + 2 ) ; }
for ( let i = 0 ; i <= n ; i ++ ) { for ( let j = 0 ; j <= Math . min ( i , k ) ; j ++ ) {
if ( j == 0 ) P [ i ] [ j ] = 1 ;
else P [ i ] [ j ] = P [ i - 1 ] [ j ] + ( j * P [ i - 1 ] [ j - 1 ] ) ;
P [ i ] [ j + 1 ] = 0 ; } } return P [ n ] [ k ] ; }
let n = 10 , k = 2 ; document . write ( " " + n + " " + k + " " + " " + permutationCoeff ( n , k ) ) ;
function permutationCoeff ( n , k ) { let fact = new Array ( n + 1 ) ;
fact [ 0 ] = 1 ;
for ( let i = 1 ; i <= n ; i ++ ) fact [ i ] = i * fact [ i - 1 ] ;
return parseInt ( fact [ n ] / fact [ n - k ] , 10 ) ; }
let n = 10 , k = 2 ; document . write ( " " + " " + n + " " + k + " " + permutationCoeff ( n , k ) ) ;
function isSubsetSum ( set , n , sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
let set = [ 3 , 34 , 4 , 12 , 5 , 2 ] ; let sum = 9 ; let n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function compute_z ( s , z ) { var l = 0 , r = 0 ; var n = s . length ; for ( var i = 1 ; i <= n - 1 ; i ++ ) { if ( i > r ) { l = i ; r = i ; while ( r < n && s [ r - l ] === s [ r ] ) { r ++ ; } z [ i ] = r - l ; r -- ; } else { var k = i - l ; if ( z [ k ] < r - i + 1 ) { z [ i ] = z [ k ] ; } else { l = i ; while ( r < n && s [ r - l ] === s [ r ] ) { r ++ ; } z [ i ] = r - l ; r -- ; } } } }
function countPermutation ( a , b ) {
b = b + b ;
b = b . substring ( 0 , b . length - 1 ) ;
var ans = 0 ; var s = a + " " + b ; var n = s . length ;
var z = new Array ( n ) . fill ( 0 ) ; compute_z ( s , z ) ; for ( var i = 1 ; i <= n - 1 ; i ++ ) {
if ( z [ i ] === a . length ) { ans ++ ; } } return ans ; }
var a = " " ; var b = " " ; document . write ( countPermutation ( a , b ) ) ;
function reverse ( input ) { a = input ; var l , r = a . length - 1 ; for ( l = 0 ; l < r ; l ++ , r -- ) { var temp = a [ l ] ; a [ l ] = a [ r ] ; a [ r ] = temp ; } return a ; }
function smallestSubsequence ( S , K ) {
var N = S . length ;
answer = [ ] ;
for ( var i = 0 ; i < N ; ++ i ) {
if ( answer . length == 0 ) { answer . push ( S [ i ] ) ; } else {
while ( ( answer . length != 0 ) && ( S [ i ] < answer [ answer . length - 1 ] )
&& ( answer . length - 1 + N - i >= K ) ) { answer . pop ( ) ; }
if ( answer . length == 0 answer . length < K ) {
answer . push ( S [ i ] ) ; } } }
var ret = [ ] ;
while ( answer . length != 0 ) { ret += answer [ answer . length - 1 ] ; answer . pop ( ) ; }
reverse ( ret ) ;
document . write ( ret ) ; } var S = " " ; var K = 3 ; smallestSubsequence ( S , K ) ;
function is_rtol ( s ) { let tmp = ( Math . sqrt ( s . length ) ) - 1 ; let first = s [ tmp ] ;
for ( let pos = tmp ; pos < s . length - 1 ; pos += tmp ) {
if ( s [ pos ] != first ) { return false ; } } return true ; }
let str = " " ;
if ( is_rtol ( str ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function check ( str , K ) {
if ( str . length % K === 0 ) { var sum = 0 , i ;
for ( i = 0 ; i < K ; i ++ ) { sum += str [ i ] . charCodeAt ( 0 ) ; }
for ( var j = i ; j < str . length ; j += K ) { var s_comp = 0 ; for ( var p = j ; p < j + K ; p ++ ) s_comp += str [ p ] . charCodeAt ( 0 ) ;
if ( s_comp !== sum )
return false ; }
return true ; }
return false ; }
var K = 3 ; var str = " " ; if ( check ( str , K ) ) document . write ( " " ) ; else document . write ( " " ) ;
function maxSum ( str ) { var maximumSum = 0 ;
var totalOnes = 0 ;
str . split ( ' ' ) . forEach ( c => { if ( c == ' ' ) totalOnes ++ ; } ) ;
var zero = 0 , ones = 0 ;
for ( var i = 0 ; str [ i ] ; i ++ ) { if ( str [ i ] == ' ' ) { zero ++ ; } else { ones ++ ; }
maximumSum = Math . max ( maximumSum , zero + ( totalOnes - ones ) ) ; } return maximumSum ; }
var str = " " ;
document . write ( maxSum ( str ) ) ;
function maxLenSubStr ( s ) {
if ( s . length < 3 ) return s . length ;
let temp = 2 ; let ans = 2 ;
for ( let i = 2 ; i < s . length ; i ++ ) {
if ( s [ i ] != s [ i - 1 ] s [ i ] != s [ i - 2 ] ) temp ++ ;
else { ans = Math . max ( temp , ans ) ; temp = 2 ; } } ans = Math . max ( temp , ans ) ; return ans ; }
let s = " " ; document . write ( maxLenSubStr ( s ) ) ;
function no_of_ways ( s ) { let n = s . length ;
let count_left = 0 , count_right = 0 ;
for ( let i = 0 ; i < n ; ++ i ) { if ( s [ i ] == s [ 0 ] ) { ++ count_left ; } else break ; }
for ( let i = n - 1 ; i >= 0 ; -- i ) { if ( s [ i ] == s [ n - 1 ] ) { ++ count_right ; } else break ; }
if ( s [ 0 ] == s [ n - 1 ] ) return ( ( count_left + 1 ) * ( count_right + 1 ) ) ;
else return ( count_left + count_right + 1 ) ; }
let s = " " ; document . write ( no_of_ways ( s ) ) ;
function preCompute ( n , s , pref ) { pref [ 0 ] = 0 ; for ( let i = 1 ; i < n ; i ++ ) { pref [ i ] = pref [ i - 1 ] ; if ( s [ i - 1 ] == s [ i ] ) pref [ i ] ++ ; } }
function query ( pref , l , r ) { return pref [ r ] - pref [ l ] ; }
let s = " " ; let n = s . length ; let pref = new Array ( n ) ; preCompute ( n , s , pref ) ;
let l = 1 ; let r = 2 ; document . write ( query ( pref , l , r ) + " " ) ;
l = 1 ; r = 5 ; document . write ( query ( pref , l , r ) + " " ) ;
function findDirection ( s ) { let count = 0 ; let d = " " ; for ( let i = 0 ; i < s . length ; i ++ ) { if ( s [ 0 ] == ' ' ) return null ; if ( s [ i ] == ' ' ) count -- ; else { if ( s [ i ] == ' ' ) count ++ ; } }
if ( count > 0 ) { if ( count % 4 == 0 ) d = " " ; else if ( count % 4 == 1 ) d = " " ; else if ( count % 4 == 2 ) d = " " ; else if ( count % 4 == 3 ) d = " " ; }
if ( count < 0 ) { if ( count % 4 == 0 ) d = " " ; else if ( count % 4 == - 1 ) d = " " ; else if ( count % 4 == - 2 ) d = " " ; else if ( count % 4 == - 3 ) d = " " ; } return d ; }
let s = " " ; document . write ( findDirection ( s ) + " " ) ; s = " " ; document . write ( findDirection ( s ) ) ;
function isCheck ( str ) { var len = str . length ; var lowerStr = " " , upperStr = " " ;
for ( var i = 0 ; i < len ; i ++ ) {
if ( str [ i ] >= ' ' && str [ i ] < ' ' ) upperStr = upperStr + str [ i ] ; else lowerStr = lowerStr + str [ i ] ; }
lowerStr = lowerStr . toUpperCase ( ) ; console . log ( lowerStr ) ; return lowerStr === upperStr ; }
var str = " " ; isCheck ( str ) ? document . write ( " " ) : document . write ( " " ) ;
function encode ( s , k ) {
let newS = " " ;
for ( let i = 0 ; i < s . length ; ++ i ) {
let val = s [ i ] . charCodeAt ( 0 ) ;
let dup = k ;
if ( val + k > 122 ) { k -= ( 122 - val ) ; k = k % 26 ; newS += String . fromCharCode ( 96 + k ) ; } else { newS += String . fromCharCode ( val + k ) ; } k = dup ; }
document . write ( newS ) ; }
let str = " " ; let k = 28 ;
encode ( str , k ) ;
function isVowel ( x ) { if ( x === " " x === " " x === " " x === " " x === " " ) return true ; else return false ; }
function updateSandwichedVowels ( a ) { var n = a . length ;
var updatedString = " " ;
for ( var i = 0 ; i < n ; i ++ ) {
if ( i === 0 i === n - 1 ) { updatedString += a [ i ] ; continue ; }
if ( isVowel ( a [ i ] ) === true && isVowel ( a [ i - 1 ] ) === false && isVowel ( a [ i + 1 ] ) === false ) { continue ; }
updatedString += a [ i ] ; } return updatedString ; }
var str = " " ;
var updatedString = updateSandwichedVowels ( str ) ; document . write ( updatedString ) ;
class Node { constructor ( ) { this . data = 0 ; this . left = null ; this . right = null ; } } ; var ans = 0 ;
function newNode ( data ) { var newNode = new Node ( ) ; newNode . data = data ; newNode . left = newNode . right = null ; return ( newNode ) ; }
function findPathUtil ( root , k , path , flag ) { if ( root == null ) return ;
if ( root . data >= k ) flag = 1 ;
if ( root . left == null && root . right == null ) { if ( flag == 1 ) { ans = 1 ; document . write ( " " ) ; for ( var i = 0 ; i < path . length ; i ++ ) { document . write ( path [ i ] + " " ) ; } document . write ( root . data + " " ) ; } return ; }
path . push ( root . data ) ;
findPathUtil ( root . left , k , path , flag ) ; findPathUtil ( root . right , k , path , flag ) ;
path . pop ( ) ; }
function findPath ( root , k ) {
var flag = 0 ;
ans = 0 ; var v = [ ] ;
findPathUtil ( root , k , v , flag ) ;
if ( ans == 0 ) document . write ( " " ) ; }
var K = 25 ;
var root = newNode ( 10 ) ; root . left = newNode ( 5 ) ; root . right = newNode ( 8 ) ; root . left . left = newNode ( 29 ) ; root . left . right = newNode ( 2 ) ; root . right . right = newNode ( 98 ) ; root . right . left = newNode ( 1 ) ; root . right . right . right = newNode ( 50 ) ; root . left . left . left = newNode ( 20 ) ; findPath ( root , K ) ;
function Tridecagonal_num ( n ) {
return ( 11 * n * n - 9 * n ) / 2 ; }
let n = 3 ; document . write ( Tridecagonal_num ( n ) + " " ) ; n = 10 ; document . write ( Tridecagonal_num ( n ) ) ;
function findNumbers ( n , w ) { let x = 0 , sum = 0 ;
if ( w >= 0 && w <= 8 ) {
x = 9 - w ; }
else if ( w >= - 9 && w <= - 1 ) {
x = 10 + w ; } sum = Math . pow ( 10 , n - 2 ) ; sum = ( x * sum ) ; return sum ; }
let n , w ;
n = 3 ; w = 4 ;
document . write ( findNumbers ( n , w ) ) ;
function MaximumHeight ( a , n ) { let result = 1 ; for ( i = 1 ; i <= n ; ++ i ) {
let y = ( i * ( i + 1 ) ) / 2 ;
if ( y < n ) result = i ;
else break ; } return result ; }
let arr = [ 40 , 100 , 20 , 30 ] ; let n = arr . length ; document . write ( MaximumHeight ( arr , n ) ) ;
function findK ( n , k ) { let a = [ ] ;
for ( let i = 1 ; i < n ; i ++ ) if ( i % 2 == 1 ) a . push ( i ) ;
for ( let i = 1 ; i < n ; i ++ ) if ( i % 2 == 0 ) a . push ( i ) ; return ( a [ k - 1 ] ) ; }
let n = 10 , k = 3 ; document . write ( findK ( n , k ) ) ;
function factorial ( n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
let num = 5 ; document . write ( " " , num , " " , factorial ( num ) ) ;
function pell ( n ) { if ( n <= 2 ) return n ; let a = 1 ; let b = 2 ; let c ; for ( let i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
let n = 4 ; document . write ( pell ( n ) ) ;
function isMultipleOf10 ( n ) { return ( n % 15 == 0 ) ; }
let n = 30 ; if ( isMultipleOf10 ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function countOddPrimeFactors ( n ) { let result = 1 ;
while ( n % 2 == 0 ) n /= 2 ;
for ( let i = 3 ; i * i <= n ; i += 2 ) { let divCount = 0 ;
while ( n % i == 0 ) { n /= i ; ++ divCount ; } result *= divCount + 1 ; }
if ( n > 2 ) result *= 2 ; return result ; } function politness ( n ) { return countOddPrimeFactors ( n ) - 1 ; }
let n = 90 ; document . write ( " " + n + " " + politness ( n ) + " " ) ; n = 15 ; document . write ( " " + n + " " + politness ( n ) ) ;
var primes = [ ] ;
var MAX = 1000000 ; function Sieve ( ) { let n = MAX ;
let nNew = parseInt ( Math . sqrt ( n ) ) ;
var marked = new Array ( n / 2 + 500 ) . fill ( 0 ) ;
for ( let i = 1 ; i <= parseInt ( ( nNew - 1 ) / 2 ) ; i ++ ) for ( let j = ( i * ( i + 1 ) ) << 1 ; j <= parseInt ( n / 2 ) ; j = j + 2 * i + 1 ) marked [ j ] = 1 ;
primes . push ( 2 ) ;
for ( let i = 1 ; i <= parseInt ( n / 2 ) ; i ++ ) if ( marked [ i ] == 0 ) primes . push ( 2 * i + 1 ) ; }
function binarySearch ( left , right , n ) { if ( left <= right ) { let mid = parseInt ( ( left + right ) / 2 ) ;
if ( mid == 0 mid == primes . length - 1 ) return primes [ mid ] ;
if ( primes [ mid ] == n ) return primes [ mid - 1 ] ;
if ( primes [ mid ] < n && primes [ mid + 1 ] > n ) return primes [ mid ] ; if ( n < primes [ mid ] ) return binarySearch ( left , mid - 1 , n ) ; else return binarySearch ( mid + 1 , right , n ) ; } return 0 ; }
Sieve ( ) ; let n = 17 ; document . write ( binarySearch ( 0 , primes . length - 1 , n ) ) ;
function factorial ( n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
let num = 5 ; document . write ( " " + num + " " + factorial ( num ) ) ;
function FlipBits ( n ) { return n -= ( n & ( - n ) ) ; }
let N = 12 ; document . write ( " " ) ; document . write ( " " + FlipBits ( N ) ) ;
function Maximum_xor_Triplet ( n , a ) {
let s = new Set ( ) ; for ( let i = 0 ; i < n ; i ++ ) { for ( let j = i ; j < n ; j ++ ) {
s . add ( a [ i ] ^ a [ j ] ) ; } } let ans = 0 ; for ( let i of s . values ( ) ) { for ( let j = 0 ; j < n ; j ++ ) {
ans = Math . max ( ans , i ^ a [ j ] ) ; } } document . write ( ans , " " ) ; }
let a = [ 1 , 3 , 8 , 15 ] ; let n = a . length ; Maximum_xor_Triplet ( n , a ) ;
function prletMissing ( arr , low , high ) {
let polets_of_range = Array ( high - low + 1 ) . fill ( 0 ) ; for ( let i = 0 ; i < arr . length ; i ++ ) {
if ( low <= arr [ i ] && arr [ i ] <= high ) polets_of_range [ arr [ i ] - low ] = true ; }
for ( let x = 0 ; x <= high - low ; x ++ ) { if ( polets_of_range [ x ] == false ) document . write ( ( low + x ) + " " ) ; } }
let arr = [ 1 , 3 , 5 , 4 ] ; let low = 1 , high = 10 ; prletMissing ( arr , low , high ) ;
function find ( a , b , k , n1 , n2 ) {
var s = new Set ( ) ; for ( var i = 0 ; i < n2 ; i ++ ) s . add ( b [ i ] ) ;
var missing = 0 ; for ( var i = 0 ; i < n1 ; i ++ ) { if ( ! s . has ( a [ i ] ) ) missing ++ ; if ( missing == k ) return a [ i ] ; } return - 1 ; }
var a = [ 0 , 2 , 4 , 6 , 8 , 10 , 12 , 14 , 15 ] ; var b = [ 4 , 10 , 6 , 8 , 12 ] ; var n1 = a . length ; var n2 = b . length ; var k = 3 ; document . write ( find ( a , b , k , n1 , n2 ) ) ;
function findString ( S , N ) {
let amounts = new Array ( 26 ) ;
for ( let i = 0 ; i < 26 ; i ++ ) { amounts [ i ] = 0 ; }
for ( let i = 0 ; i < S . length ; i ++ ) { amounts [ S [ i ] . charCodeAt ( 0 ) - 97 ] ++ ; } let count = 0 ;
for ( let i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) count ++ ; }
if ( count > N ) { document . write ( " " ) ; }
else { let ans = " " ; let high = 100001 ; let low = 0 ; let mid , total ;
while ( high - low > 1 ) { total = 0 ;
mid = Math . floor ( ( high + low ) / 2 ) ;
for ( let i = 0 ; i < 26 ; i ++ ) {
if ( amounts [ i ] > 0 ) { total += Math . floor ( ( amounts [ i ] - 1 ) / mid + 1 ) ; } }
if ( total <= N ) { high = mid ; } else { low = mid ; } } document . write ( high + " " ) ; total = 0 ;
for ( let i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) { total += Math . floor ( ( amounts [ i ] - 1 ) / high + 1 ) ; for ( let j = 0 ; j < Math . floor ( ( amounts [ i ] - 1 ) / high + 1 ) ; j ++ ) {
ans += String . fromCharCode ( i + 97 ) ; } } }
for ( let i = total ; i < N ; i ++ ) { ans += " " ; } ans = ans . split ( " " ) . reverse ( ) . join ( " " ) ;
document . write ( ans ) ; } }
let S = " " ; let K = 4 ; findString ( S , K ) ;
function printFirstRepeating ( arr ) {
let min = - 1 ;
let set = new Set ( ) ;
for ( let i = arr . length - 1 ; i >= 0 ; i -- ) {
if ( set . has ( arr [ i ] ) ) min = i ;
else set . add ( arr [ i ] ) ; }
if ( min != - 1 ) document . write ( " " + arr [ min ] ) ; else document . write ( " " ) ; }
let arr = [ 10 , 5 , 3 , 4 , 3 , 5 , 6 ] ; printFirstRepeating ( arr ) ;
function printFirstRepeating ( arr , n ) {
var k = 0 ;
var max = n ; for ( i = 0 ; i < n ; i ++ ) if ( max < arr [ i ] ) max = arr [ i ] ;
var a = Array ( max + 1 ) . fill ( 0 ) ;
var b = Array ( max + 1 ) . fill ( 0 ) ; for ( var i = 0 ; i < n ; i ++ ) {
if ( a [ arr [ i ] ] != 0 ) { b [ arr [ i ] ] = 1 ; k = 1 ; continue ; } else
a [ arr [ i ] ] = i ; } if ( k == 0 ) document . write ( " " ) ; else { var min = max + 1 ;
for ( i = 0 ; i < max + 1 ; i ++ ) if ( a [ i ] != 0 && min > a [ i ] && b [ i ] != 0 ) min = a [ i ] ; document . write ( arr [ min ] ) ; } document . write ( " " ) ; }
var arr = [ 10 , 5 , 3 , 4 , 3 , 5 , 6 ] ; var n = arr . length ; printFirstRepeating ( arr , n ) ;
function printKDistinct ( arr , n , k ) { var dist_count = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
var j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return - 1 ; }
var ar = [ 1 , 2 , 1 , 3 , 4 , 2 ] ; var n = ar . length ; var k = 2 ; document . write ( printKDistinct ( ar , n , k ) ) ;
function countSubarrays ( A , N ) {
var res = 0 ;
var curr = A [ 0 ] ; var cnt = [ ] ; cnt . fill ( 1 ) for ( var c = 1 ; c < N ; c ++ ) {
if ( A == curr )
cnt [ cnt . length - 1 ] ++ ; else
curr = A ; cnt . push ( 1 ) ; }
for ( var i = 1 ; i < cnt . length ; i ++ ) {
res += Math . min ( cnt [ i - 1 ] , cnt [ i ] ) ; } document . write ( res ) ; }
var A = [ 1 , 1 , 0 , 0 , 1 , 0 ] ; var N = A . length ;
countSubarrays ( A , N ) ;
class Node { constructor ( data ) { this . left = null ; this . right = null ; this . val = data ; } }
function newNode ( data ) { let temp = new Node ( data ) ; return temp ; }
function isEvenOddBinaryTree ( root ) { if ( root == null ) return true ;
let q = [ ] ; q . push ( root ) ;
let level = 0 ;
while ( q . length > 0 ) {
let size = q . length ; for ( let i = 0 ; i < size ; i ++ ) { let node = q [ 0 ] ; q . shift ( ) ;
if ( level % 2 == 0 ) { if ( node . val % 2 == 1 ) return false ; } else if ( level % 2 == 1 ) { if ( node . val % 2 == 0 ) return false ; }
if ( node . left != null ) { q . push ( node . left ) ; } if ( node . right != null ) { q . push ( node . right ) ; } }
level ++ ; } return true ; }
let root = null ; root = newNode ( 2 ) ; root . left = newNode ( 3 ) ; root . right = newNode ( 9 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 10 ) ; root . right . right = newNode ( 6 ) ;
if ( isEvenOddBinaryTree ( root ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function findMaxLen ( a ) {
var freq = Array ( n + 1 ) . fill ( 0 ) ; var i ; for ( i = 0 ; i < n ; ++ i ) { freq [ a [ i ] ] ++ ; } var maxFreqElement = - 2147483648 ; var maxFreqCount = 1 ; for ( i = 1 ; i <= n ; ++ i ) {
if ( freq [ i ] > maxFreqElement ) { maxFreqElement = freq [ i ] ; maxFreqCount = 1 ; }
else if ( freq [ i ] == maxFreqElement ) maxFreqCount ++ ; } var ans ;
if ( maxFreqElement == 1 ) ans = 0 ; else {
ans = ( ( n - maxFreqCount ) / ( maxFreqElement - 1 ) ) ; }
return ans ; }
var a = [ 1 , 2 , 1 , 2 ] ; document . write ( findMaxLen ( a ) ) ;
function getMid ( s , e ) { return ( s + Math . floor ( ( e - s ) / 2 ) ) ; }
function MaxUtil ( st , ss , se , l , r , node ) {
if ( l <= ss && r >= se )
return st [ node ] ;
if ( se < l ss > r ) return - 1 ;
let mid = getMid ( ss , se ) ; return Math . max ( MaxUtil ( st , ss , mid , l , r , 2 * node + 1 ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 2 ) ) ; }
function getMax ( st , n , l , r ) {
if ( l < 0 r > n - 1 l > r ) { document . write ( " " ) ; return - 1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
function constructSTUtil ( arr , ss , se , st , si ) {
if ( ss == se ) { st [ si ] = arr [ ss ] ; return arr [ ss ] ; }
let mid = getMid ( ss , se ) ;
st [ si ] = Math . max ( constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) ,
constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ) ; return st [ si ] ; }
function constructST ( arr , n ) {
let x = ( Math . ceil ( Math . log ( n ) ) ) ;
let max_size = 2 * Math . pow ( 2 , x ) - 1 ;
let st = Array . from ( { length : max_size } , ( _ , i ) => 0 ) ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
let arr = [ 5 , 2 , 3 , 0 ] ; let n = arr . length ;
let st = constructST ( arr , n ) ; let Q = [ [ 1 , 3 ] , [ 0 , 2 ] ] ; for ( let i = 0 ; i < Q . length ; i ++ ) { let max = getMax ( st , n , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) ; let ok = 0 ; for ( let j = 30 ; j >= 0 ; j -- ) { if ( ( max & ( 1 << j ) ) != 0 ) ok = 1 ; if ( ok <= 0 ) continue ; max |= ( 1 << j ) ; } document . write ( max + " " ) ; }
function calculate ( a , n ) {
a . sort ( ) ; let count = 1 ; let answer = 0 ;
for ( let i = 1 ; i < n ; i ++ ) { if ( a [ i ] == a [ i - 1 ] ) {
count += 1 ; } else {
answer = answer + Math . floor ( ( count * ( count - 1 ) ) / 2 ) ; count = 1 ; } } answer = answer + Math . floor ( ( count * ( count - 1 ) ) / 2 ) ; return answer ; }
let a = [ 1 , 2 , 1 , 2 , 4 ] ; let n = a . length ;
document . write ( calculate ( a , n ) ) ;
function calculate ( a , n ) {
let maximum = Math . max ( ... a ) ;
let frequency = new Array ( maximum + 1 ) . fill ( 0 ) ;
for ( let i = 0 ; i < n ; i ++ ) {
frequency [ a [ i ] ] += 1 ; } let answer = 0 ;
for ( let i = 0 ; i < maximum + 1 ; i ++ ) {
answer = answer + frequency [ i ] * ( frequency [ i ] - 1 ) ; } return parseInt ( answer / 2 ) ; }
let a = [ 1 , 2 , 1 , 2 , 4 ] ; let n = a . length ;
document . write ( calculate ( a , n ) ) ;
function findSubArray ( arr , n ) { let sum = 0 ; let maxsize = - 1 , startindex = 0 ; let endindex = 0 ;
for ( let i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? - 1 : 1 ;
for ( let j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] == 0 ) sum += - 1 ; else sum += 1 ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } endindex = startindex + maxsize - 1 ; if ( maxsize == - 1 ) document . write ( " " ) ; else document . write ( startindex + " " + endindex ) ; return maxsize ; }
let arr = [ 1 , 0 , 0 , 1 , 0 , 1 , 1 ] ; let size = arr . length ; findSubArray ( arr , size ) ;
function findMax ( arr , low , high ) {
if ( high == low ) return arr [ low ] ;
let mid = low + ( high - low ) / 2 ;
if ( mid == 0 && arr [ mid ] > arr [ mid + 1 ] ) { return arr [ mid ] ; }
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] && mid > 0 && arr [ mid ] > arr [ mid - 1 ] ) { return arr [ mid ] ; }
if ( arr [ low ] > arr [ mid ] ) { return findMax ( arr , low , mid - 1 ) ; } else { return findMax ( arr , mid + 1 , high ) ; } }
let arr = [ 5 , 6 , 1 , 2 , 3 , 4 ] ; let n = arr . length ; document . write ( findMax ( arr , 0 , n - 1 ) ) ;
function ternarySearch ( l , r , key , ar ) { while ( r >= l ) {
let mid1 = l + parseInt ( ( r - l ) / 3 , 10 ) ; let mid2 = r - parseInt ( ( r - l ) / 3 , 10 ) ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
r = mid1 - 1 ; } else if ( key > ar [ mid2 ] ) {
l = mid2 + 1 ; } else {
l = mid1 + 1 ; r = mid2 - 1 ; } }
return - 1 ; }
let l , r , p , key ;
let ar = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
document . write ( " " + key + " " + p + " " ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
document . write ( " " + key + " " + p ) ;
function majorityNumber ( arr , n ) { let ans = - 1 ; let freq = new Map ( ) ; for ( let i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; if ( freq . has ( arr [ i ] ) ) { freq . set ( arr [ i ] , freq . get ( arr [ i ] ) + 1 ) } else { freq . set ( arr [ i ] , 1 ) } if ( freq . get ( arr [ i ] ) > n / 2 ) ans = arr [ i ] ; } return ans ; }
let a = [ 2 , 2 , 1 , 1 , 1 , 2 , 2 ] ; let n = a . length ; document . write ( majorityNumber ( a , n ) ) ;
function search ( arr , l , h , key ) { if ( l > h ) return - 1 ; let mid = Math . floor ( ( l + h ) / 2 ) ; if ( arr [ mid ] == key ) return mid ;
if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
return search ( arr , mid + 1 , h , key ) ; }
if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
let arr = [ 4 , 5 , 6 , 7 , 8 , 9 , 1 , 2 , 3 ] ; let n = arr . length ; let key = 6 ; let i = search ( arr , 0 , n - 1 , key ) ; if ( i != - 1 ) document . write ( " " + i + " " ) ; else document . write ( " " ) ;
function findMin ( arr , low , high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
let mid = low + Math . floor ( ( high - low ) / 2 ) ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
let arr1 = [ 5 , 6 , 1 , 2 , 3 , 4 ] ; let n1 = arr1 . length ; document . write ( " " + findMin ( arr1 , 0 , n1 - 1 ) + " " ) ; let arr2 = [ 1 , 2 , 3 , 4 ] ; let n2 = arr2 . length ; document . write ( " " + findMin ( arr2 , 0 , n2 - 1 ) + " " ) ; let arr3 = [ 1 ] ; let n3 = arr3 . length ; document . write ( " " + findMin ( arr3 , 0 , n3 - 1 ) + " " ) ; let arr4 = [ 1 , 2 ] ; let n4 = arr4 . length ; document . write ( " " + findMin ( arr4 , 0 , n4 - 1 ) + " " ) ; let arr5 = [ 2 , 1 ] ; let n5 = arr5 . length ; document . write ( " " + findMin ( arr5 , 0 , n5 - 1 ) + " " ) ; let arr6 = [ 5 , 6 , 7 , 1 , 2 , 3 , 4 ] ; let n6 = arr6 . length ; document . write ( " " + findMin ( arr6 , 0 , n6 - 1 ) + " " ) ; let arr7 = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; let n7 = arr7 . length ; document . write ( " " + findMin ( arr7 , 0 , n7 - 1 ) + " " ) ; let arr8 = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 ] ; let n8 = arr8 . length ; document . write ( " " + findMin ( arr8 , 0 , n8 - 1 ) + " " ) ; let arr9 = [ 3 , 4 , 5 , 1 , 2 ] ; let n9 = arr9 . length ; document . write ( " " + findMin ( arr9 , 0 , n9 - 1 ) + " " ) ;
function findMin ( arr , low , high ) { while ( low < high ) { let mid = Math . floor ( low + ( high - low ) / 2 ) ; if ( arr [ mid ] == arr [ high ] ) high -- ; else if ( arr [ mid ] > arr [ high ] ) low = mid + 1 ; else high = mid ; } return arr [ high ] ; }
var arr1 = [ 5 , 6 , 1 , 2 , 3 , 4 ] ; var n1 = arr1 . length ; document . write ( " " + findMin ( arr1 , 0 , n1 - 1 ) + " " ) ; var arr2 = [ 1 , 2 , 3 , 4 ] ; var n2 = arr2 . length ; document . write ( " " + findMin ( arr2 , 0 , n2 - 1 ) + " " ) ; var arr3 = [ 1 ] ; var n3 = arr3 . length ; document . write ( " " + findMin ( arr3 , 0 , n3 - 1 ) + " " ) ; var arr4 = [ 1 , 2 ] ; var n4 = arr4 . length ; document . write ( " " + findMin ( arr4 , 0 , n4 - 1 ) + " " ) ; var arr5 = [ 2 , 1 ] ; var n5 = arr5 . length ; document . write ( " " + findMin ( arr5 , 0 , n5 - 1 ) + " " ) ; var arr6 = [ 5 , 6 , 7 , 1 , 2 , 3 , 4 ] ; var n6 = arr6 . length ; document . write ( " " + findMin ( arr6 , 0 , n6 - 1 ) + " " ) ; var arr7 = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ] ; var n7 = arr7 . length ; document . write ( " " + findMin ( arr7 , 0 , n7 - 1 ) + " " ) ; var arr8 = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 ] ; var n8 = arr8 . length ; document . write ( " " + findMin ( arr8 , 0 , n8 - 1 ) + " " ) ; var arr9 = [ 3 , 4 , 5 , 1 , 2 ] ; var n9 = arr9 . length ; document . write ( " " + findMin ( arr9 , 0 , n9 - 1 ) + " " ) ;
function print2Smallest ( arr , arr_size ) { let i , first , second ;
if ( arr_size < 2 ) { document . write ( " " ) ; return ; } first = Number . MAX_VALUE ; second = Number . MAX_VALUE ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == Number . MAX_VALUE ) document . write ( " " ) ; else document . write ( " " + first + " " + " " + second + ' ' ) ; }
let arr = [ 12 , 13 , 1 , 10 , 34 , 1 ] ; let n = arr . length ; print2Smallest ( arr , n ) ;
const MAX = 1000
var tree = new Array ( 4 * MAX ) ;
var arr = new Array ( MAX ) ;
function gcd ( a , b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
function lcm ( a , b ) { return Math . floor ( a * b / gcd ( a , b ) ) ; }
function build ( node , start , end ) {
if ( start == end ) { tree [ node ] = arr [ start ] ; return ; } let mid = Math . floor ( ( start + end ) / 2 ) ;
build ( 2 * node , start , mid ) ; build ( 2 * node + 1 , mid + 1 , end ) ;
let left_lcm = tree [ 2 * node ] ; let right_lcm = tree [ 2 * node + 1 ] ; tree [ node ] = lcm ( left_lcm , right_lcm ) ; }
function query ( node , start , end , l , r ) {
if ( end < l start > r ) return 1 ;
if ( l <= start && r >= end ) return tree [ node ] ;
let mid = Math . floor ( ( start + end ) / 2 ) ; let left_lcm = query ( 2 * node , start , mid , l , r ) ; let right_lcm = query ( 2 * node + 1 , mid + 1 , end , l , r ) ; return lcm ( left_lcm , right_lcm ) ; }
arr [ 0 ] = 5 ; arr [ 1 ] = 7 ; arr [ 2 ] = 5 ; arr [ 3 ] = 2 ; arr [ 4 ] = 10 ; arr [ 5 ] = 12 ; arr [ 6 ] = 11 ; arr [ 7 ] = 17 ; arr [ 8 ] = 14 ; arr [ 9 ] = 1 ; arr [ 10 ] = 44 ;
build ( 1 , 0 , 10 ) ;
document . write ( query ( 1 , 0 , 10 , 2 , 5 ) + " " ) ;
document . write ( query ( 1 , 0 , 10 , 5 , 10 ) + " " ) ;
document . write ( query ( 1 , 0 , 10 , 0 , 10 ) + " " ) ;
let M = 1000000007 ; function waysOfDecoding ( s ) { let dp = new Array ( s . length + 1 ) ; for ( let i = 0 ; i < s . length + 1 ; i ++ ) dp [ i ] = 0 ; dp [ 0 ] = 1 ;
dp [ 1 ] = s [ 0 ] == ' ' ? 9 : s [ 0 ] == ' ' ? 0 : 1 ;
for ( let i = 1 ; i < s . length ; i ++ ) {
if ( s [ i ] == ' ' ) { dp [ i + 1 ] = 9 * dp [ i ] ;
if ( s [ i - 1 ] == ' ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 9 * dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 6 * dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 15 * dp [ i - 1 ] ) % M ; } else {
dp [ i + 1 ] = s [ i ] != ' ' ? dp [ i ] : 0 ;
if ( s [ i - 1 ] == ' ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' ' && s [ i ] <= ' ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + ( s [ i ] <= ' ' ? 2 : 1 ) * dp [ i - 1 ] ) % M ; } } return dp [ s . length ] ; }
let s = " " ; document . write ( waysOfDecoding ( s ) ) ;
function countSubset ( arr , n , diff ) {
var sum = 0 ; for ( var i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; } sum += diff ; sum = sum / 2 ;
var t = new Array ( n + 1 ) ; for ( var i = 0 ; i < t . length ; i ++ ) { t [ i ] = new Array ( sum + 1 ) ; } for ( var i = 0 ; i < t . length ; i ++ ) { for ( var j = 0 ; j < t [ i ] . length ; j ++ ) { t [ i ] [ j ] = 0 ; } }
for ( var j = 0 ; j <= sum ; j ++ ) t [ 0 ] [ j ] = 0 ;
for ( var i = 0 ; i <= n ; i ++ ) t [ i ] [ 0 ] = 1 ;
for ( var i = 1 ; i <= n ; i ++ ) { for ( var j = 1 ; j <= sum ; j ++ ) {
if ( arr [ i - 1 ] > j ) t [ i ] [ j ] = t [ i - 1 ] [ j ] ; else { t [ i ] [ j ] = t [ i - 1 ] [ j ] + t [ i - 1 ] [ j - arr [ i - 1 ] ] ; } } }
return t [ n ] [ sum ] ; }
var diff = 1 ; var n = 4 ; var arr = [ 1 , 1 , 2 , 3 ] ;
document . write ( countSubset ( arr , n , diff ) ) ;
let dp = new Array ( 105 ) ; for ( var i = 0 ; i < dp . length ; i ++ ) { dp [ i ] = new Array ( 2 ) ; } for ( var i = 0 ; i < dp . length ; i ++ ) { for ( var j = 0 ; j < dp . length ; j ++ ) { dp [ i ] [ j ] = 0 ; } }
function find ( N , a , b ) { let probability = 0.0 ;
for ( let i = 1 ; i <= 6 ; i ++ ) dp [ 1 ] [ i ] = ( 1.0 / 6 ) ; for ( let i = 2 ; i <= N ; i ++ ) { for ( let j = i ; j <= 6 * i ; j ++ ) { for ( let k = 1 ; k <= 6 && k <= j ; k ++ ) { dp [ i ] [ j ] = dp [ i ] [ j ] + dp [ i - 1 ] [ j - k ] / 6 ; } } }
for ( let sum = a ; sum <= b ; sum ++ ) probability = probability + dp [ N ] [ sum ] ; return probability ; }
let N = 4 , a = 13 , b = 17 ; let probability = find ( N , a , b ) ;
document . write ( probability ) ;
class Node { constructor ( data ) { this . data = data ; this . left = this . right = null ; } }
function getSumAlternate ( root ) { if ( root == null ) return 0 ; let sum = root . data ; if ( root . left != null ) { sum += getSum ( root . left . left ) ; sum += getSum ( root . left . right ) ; } if ( root . right != null ) { sum += getSum ( root . right . left ) ; sum += getSum ( root . right . right ) ; } return sum ; }
function getSum ( root ) { if ( root == null ) return 0 ;
return Math . max ( getSumAlternate ( root ) , ( getSumAlternate ( root . left ) + getSumAlternate ( root . right ) ) ) ; }
let root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . right . left = new Node ( 4 ) ; root . right . left . right = new Node ( 5 ) ; root . right . left . right . left = new Node ( 6 ) ; document . write ( getSum ( root ) ) ;
function isSubsetSum ( arr , n , sum ) {
let subset = new Array ( 2 ) ; for ( var i = 0 ; i < subset . length ; i ++ ) { subset [ i ] = new Array ( 2 ) ; } for ( let i = 0 ; i <= n ; i ++ ) { for ( let j = 0 ; j <= sum ; j ++ ) {
if ( j == 0 ) subset [ i % 2 ] [ j ] = true ;
else if ( i == 0 ) subset [ i % 2 ] [ j ] = false ; else if ( arr [ i - 1 ] <= j ) subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] || subset [ ( i + 1 ) % 2 ] [ j ] ; else subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] ; } } return subset [ n % 2 ] [ sum ] ; }
let arr = [ 1 , 2 , 5 ] ; let sum = 7 ; let n = arr . length ; if ( isSubsetSum ( arr , n , sum ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function findMaxSum ( arr , n ) { var res = - 1000000000 ; for ( var i = 0 ; i < n ; i ++ ) { var prefix_sum = arr [ i ] ; for ( var j = 0 ; j < i ; j ++ ) prefix_sum += arr [ j ] ; var suffix_sum = arr [ i ] ; for ( var j = n - 1 ; j > i ; j -- ) suffix_sum += arr [ j ] ; if ( prefix_sum == suffix_sum ) res = Math . max ( res , prefix_sum ) ; } return res ; }
var arr = [ - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 ] ; var n = arr . length ; document . write ( findMaxSum ( arr , n ) ) ;
function findMaxSum ( arr , n ) {
let preSum = new Array ( n ) ; preSum . fill ( 0 ) ;
let suffSum = new Array ( n ) ; suffSum . fill ( 0 ) ;
let ans = Number . MIN_VALUE ;
preSum [ 0 ] = arr [ 0 ] ; for ( let i = 1 ; i < n ; i ++ ) preSum [ i ] = preSum [ i - 1 ] + arr [ i ] ;
suffSum [ n - 1 ] = arr [ n - 1 ] ; if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) ans = Math . max ( ans , preSum [ n - 1 ] ) ; for ( let i = n - 2 ; i >= 0 ; i -- ) { suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] ; if ( suffSum [ i ] == preSum [ i ] ) ans = Math . max ( ans , preSum [ i ] ) ; } return ans ; }
let arr = [ - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 ] ; let n = arr . length ; document . write ( findMaxSum ( arr , n ) ) ;
function findMajority ( arr , n ) { let maxCount = 0 ;
let index = - 1 ; for ( let i = 0 ; i < n ; i ++ ) { let count = 0 ; for ( let j = 0 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) count ++ ; }
if ( count > maxCount ) { maxCount = count ; index = i ; } }
if ( maxCount > n / 2 ) document . write ( arr [ index ] ) ; else document . write ( " " ) ; }
let arr = [ 1 , 1 , 2 , 1 , 3 , 5 , 1 ] ; let n = arr . length ;
findMajority ( arr , n ) ;
function findCandidate ( a , size ) { let maj_index = 0 , count = 1 ; let i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
function isMajority ( a , size , cand ) { let i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) { if ( a [ i ] == cand ) count ++ ; } if ( count > parseInt ( size / 2 , 10 ) ) return true ; else return false ; }
function printMajority ( a , size ) {
let cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) document . write ( " " + cand + " " ) ; else document . write ( " " ) ; }
let a = [ 1 , 3 , 3 , 1 , 2 ] ; let size = a . length ;
printMajority ( a , size ) ;
function majorityElement ( arr , n ) {
arr . sort ( function ( a , b ) { return a - b } ) ; let count = 1 , max_ele = - 1 , temp = arr [ 0 ] , ele = 0 , f = 0 ; for ( let i = 1 ; i < n ; i ++ ) {
if ( temp == arr [ i ] ) { count ++ ; } else { count = 1 ; temp = arr [ i ] ; }
if ( max_ele < count ) { max_ele = count ; ele = arr [ i ] ; if ( max_ele > parseInt ( n / 2 , 10 ) ) { f = 1 ; break ; } } }
return ( f == 1 ? ele : - 1 ) ; }
let arr = [ 1 , 1 , 2 , 1 , 3 , 5 , 1 ] ; let n = 7 ;
document . write ( majorityElement ( arr , n ) ) ;
function isSubsetSum ( set , n , sum ) {
let subset = new Array ( sum + 1 ) ; for ( let i = 0 ; i < sum + 1 ; i ++ ) { subset [ i ] = new Array ( sum + 1 ) ; for ( let j = 0 ; j < n + 1 ; j ++ ) { subset [ i ] [ j ] = 0 ; } }
for ( let i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( let i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( let i = 1 ; i <= sum ; i ++ ) { for ( let j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
let set = [ 3 , 34 , 4 , 12 , 5 , 2 ] ; let sum = 9 ; let n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function subsetSum ( a , n , sum ) {
if ( sum == 0 ) return 1 ; if ( n <= 0 ) return 0 ;
if ( tab [ n - 1 ] [ sum ] != - 1 ) return tab [ n - 1 ] [ sum ] ;
if ( a [ n - 1 ] > sum ) return tab [ n - 1 ] [ sum ] = subsetSum ( a , n - 1 , sum ) ; else {
return tab [ n - 1 ] [ sum ] = subsetSum ( a , n - 1 , sum ) || subsetSum ( a , n - 1 , sum - a [ n - 1 ] ) ; } }
let tab = Array ( 2000 ) . fill ( ) . map ( ( ) => Array ( 2000 ) . fill ( - 1 ) ) ; let n = 5 ; let a = [ 1 , 5 , 3 , 7 , 4 ] ; let sum = 12 ; if ( subsetSum ( a , n , sum ) ) { document . write ( " " + " " ) ; } else { document . write ( " " + " " ) ; }
function binpow ( a , b ) { let res = 1 ; while ( b ) { if ( b & 1 ) res = res * a ; a = a * a ; b = Math . floor ( b / 2 ) ; } return res ; }
function find ( x ) { if ( x == 0 ) return 0 ; let p = Math . log2 ( x ) ; return binpow ( 2 , p + 1 ) - 1 ; }
function getBinary ( n ) {
let ans = " " ;
while ( n ) { let dig = n % 2 ; ans += String ( dig ) ; n = Math . floor ( n / 2 ) ; }
return ans ; }
function totalCountDifference ( n ) {
let ans = getBinary ( n ) ;
let req = 0 ;
for ( let i = 0 ; i < ans . length ; i ++ ) {
if ( ans [ i ] == ' ' ) { req += find ( binpow ( 2 , i ) ) ; } } return req ; }
let N = 5 ;
document . write ( totalCountDifference ( N ) ) ;
function Maximum_Length ( a ) {
let counts = new Array ( 11 ) ; counts . fill ( 0 ) ;
let ans = 0 ; for ( let index = 0 ; index < a . length ; index ++ ) {
counts [ a [ index ] ] += 1 ;
let k = [ ] ; for ( let i = 0 ; i < counts . length ; i ++ ) { if ( counts [ i ] != 0 ) { k . push ( i ) ; } } k . sort ( function ( a , b ) { return a - b } ) ;
if ( k . length == 1 || ( k [ 0 ] == k [ k . length - 2 ] && k [ k . length - 1 ] - k [ k . length - 2 ] == 1 ) || ( k [ 0 ] == 1 && k [ 1 ] == k [ k . length - 1 ] ) ) ans = index ; }
return ( ans ) ; }
let a = [ 1 , 1 , 1 , 2 , 2 , 2 ] ; document . write ( Maximum_Length ( a ) ) ;
function gcd ( a , b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
function print_gcd_online ( n , m , query , arr ) {
let max_gcd = 0 ; let i = 0 ;
for ( i = 0 ; i < n ; i ++ ) max_gcd = gcd ( max_gcd , arr [ i ] ) ;
for ( i = 0 ; i < m ; i ++ ) {
query [ i ] [ 0 ] -- ;
arr [ query [ i ] [ 0 ] ] /= query [ i ] [ 1 ] ;
max_gcd = gcd ( arr [ query [ i ] [ 0 ] ] , max_gcd ) ;
document . write ( max_gcd + " " ) ; } }
let n = 3 ; let m = 3 ; let query = new Array ( m ) ; for ( let i = 0 ; i < m ; i ++ ) { query [ i ] = new Array ( 2 ) ; for ( let j = 0 ; j < 2 ; j ++ ) { query [ i ] [ j ] = 0 ; } } let arr = [ 36 , 24 , 72 ] ; query [ 0 ] [ 0 ] = 1 ; query [ 0 ] [ 1 ] = 3 ; query [ 1 ] [ 0 ] = 3 ; query [ 1 ] [ 1 ] = 12 ; query [ 2 ] [ 0 ] = 2 ; query [ 2 ] [ 1 ] = 4 ; print_gcd_online ( n , m , query , arr ) ;
var MAX = 1000000 ;
var prime = Array ( MAX + 1 ) . fill ( true ) ;
var sum = Array ( MAX + 1 ) . fill ( 0 ) ;
function SieveOfEratosthenes ( ) {
prime [ 1 ] = false ; for ( var p = 2 ; p * p <= MAX ; p ++ ) {
if ( prime [ p ] ) {
for ( var i = p * 2 ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( var i = 1 ; i <= MAX ; i ++ ) { if ( prime [ i ] == true ) sum [ i ] = 1 ; sum [ i ] += sum [ i - 1 ] ; } }
SieveOfEratosthenes ( ) ;
var l = 3 , r = 9 ;
var c = ( sum [ r ] - sum [ l - 1 ] ) ;
document . write ( " " + c ) ;
function area ( r ) {
if ( r < 0 ) return - 1 ;
var area = ( 3.14 * Math . pow ( r / ( 2 * Math . sqrt ( 2 ) ) , 2 ) ) ; return area ; }
var a = 5 ; document . write ( area ( a ) . toFixed ( 6 ) ) ;
let N = 100005 ;
let prime = new Array ( N ) . fill ( true ) ; function SieveOfEratosthenes ( ) { prime [ 1 ] = false ; for ( let p = 2 ; p < Math . floor ( Math . sqrt ( N ) ) ; p ++ ) {
if ( prime [ p ] == true )
for ( let i = 2 * p ; i < N ; i += p ) prime [ i ] = false ; } }
function almostPrimes ( n ) {
let ans = 0 ;
for ( let i = 6 ; i < n + 1 ; i ++ ) {
let c = 0 ; for ( let j = 2 ; i >= j * j ; j ++ ) {
if ( i % j == 0 ) { if ( j * j == i ) { if ( prime [ j ] ) c += 1 ; } else { if ( prime [ j ] ) c += 1 ; if ( prime [ ( i / j ) ] ) c += 1 ; } } }
if ( c == 2 ) ans += 1 ; } return ans ; }
SieveOfEratosthenes ( ) ; let n = 21 ; document . write ( almostPrimes ( n ) ) ;
function sumOfDigitsSingle ( x ) { let ans = 0 ; while ( x ) { ans += x % 10 ; x = Math . floor ( x / 10 ) ; } return ans ; }
function closest ( x ) { let ans = 0 ; while ( ans * 10 + 9 <= x ) ans = ans * 10 + 9 ; return ans ; } function sumOfDigitsTwoParts ( N ) { let A = closest ( N ) ; return sumOfDigitsSingle ( A ) + sumOfDigitsSingle ( N - A ) ; }
let N = 35 ; document . write ( sumOfDigitsTwoParts ( N ) ) ;
function isPrime ( p ) {
let checkNumber = Math . pow ( 2 , p ) - 1 ;
let nextval = 4 % checkNumber ;
for ( let i = 1 ; i < p - 1 ; i ++ ) nextval = ( nextval * nextval - 2 ) % checkNumber ;
return ( nextval == 0 ) ; }
let p = 7 ; let checkNumber = Math . pow ( 2 , p ) - 1 ; if ( isPrime ( p ) ) document . write ( checkNumber + " " ) ; else document . write ( checkNumber + " " ) ;
function sieve ( n , prime ) { for ( let p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( let i = p * 2 ; i <= n ; i += p ) prime [ i ] = false ; } } } function printSophieGermanNumber ( n ) {
let prime = new Array ( ) ; for ( let i = 0 ; i < ( 2 * n + 1 ) ; i ++ ) prime [ i ] = true ; sieve ( 2 * n + 1 , prime ) ; for ( let i = 2 ; i <= n ; ++ i ) {
if ( prime [ i ] && prime [ 2 * i + 1 ] ) document . write ( i + " " ) ; } }
let n = 25 ; printSophieGermanNumber ( n ) ;
function ucal ( u , n ) { if ( n == 0 ) return 1 ; var temp = u ; for ( var i = 1 ; i <= n / 2 ; i ++ ) temp = temp * ( u - i ) ; for ( var i = 1 ; i < n / 2 ; i ++ ) temp = temp * ( u + i ) ; return temp ; }
function fact ( n ) { var f = 1 ; for ( var i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
var n = 6 ; var x = [ 25 , 26 , 27 , 28 , 29 , 30 ] ;
var y = Array ( n ) . fill ( 0.0 ) . map ( x => Array ( n ) . fill ( 0.0 ) ) ; ; y [ 0 ] [ 0 ] = 4.000 ; y [ 1 ] [ 0 ] = 3.846 ; y [ 2 ] [ 0 ] = 3.704 ; y [ 3 ] [ 0 ] = 3.571 ; y [ 4 ] [ 0 ] = 3.448 ; y [ 5 ] [ 0 ] = 3.333 ;
for ( var i = 1 ; i < n ; i ++ ) for ( var j = 0 ; j < n - i ; j ++ ) y [ j ] [ i ] = y [ j + 1 ] [ i - 1 ] - y [ j ] [ i - 1 ] ;
for ( var i = 0 ; i < n ; i ++ ) { for ( var j = 0 ; j < n - i ; j ++ ) document . write ( y [ i ] [ j ] . toFixed ( 6 ) + " " ) ; document . write ( ' ' ) ; }
var value = 27.4 ;
var sum = ( y [ 2 ] [ 0 ] + y [ 3 ] [ 0 ] ) / 2 ;
var k ;
if ( ( n % 2 ) > 0 ) k = n / 2 ; else
k = n / 2 - 1 ; var u = ( value - x [ k ] ) / ( x [ 1 ] - x [ 0 ] ) ;
for ( var i = 1 ; i < n ; i ++ ) { if ( ( i % 2 ) > 0 ) sum = sum + ( ( u - 0.5 ) * ucal ( u , i - 1 ) * y [ k ] [ i ] ) / fact ( i ) ; else sum = sum + ( ucal ( u , i ) * ( y [ k ] [ i ] + y [ -- k ] [ i ] ) / ( fact ( i ) * 2 ) ) ; } document . write ( " " + value . toFixed ( 6 ) + " " + sum . toFixed ( 6 ) ) ;
function fibonacci ( n ) { let a = 0 ; let b = 1 ; let c ; if ( n <= 1 ) return n ; for ( let i = 2 ; i <= n ; i ++ ) { c = a + b ; a = b ; b = c ; } return c ; }
function isMultipleOf10 ( n ) { let f = fibonacci ( 30 ) ; return ( f % 10 == 0 ) ; }
let n = 30 ; if ( isMultipleOf10 ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function powerOf2 ( n ) {
if ( n == 1 ) return true ;
else if ( n % 2 != 0 n == 0 ) return false ;
return powerOf2 ( n / 2 ) ; }
var n = 64 ;
var m = 12 ; if ( powerOf2 ( n ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ; if ( powerOf2 ( m ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function isPowerOfTwo ( x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
document . write ( isPowerOfTwo ( 31 ) ? " " : " " ) ; document . write ( " " + ( isPowerOfTwo ( 64 ) ? " " : " " ) ) ;
function isPowerofTwo ( n ) { if ( n == 0 ) return false ; if ( ( n & ( ~ ( n - 1 ) ) ) == n ) return true ; return false ; }
if ( isPowerofTwo ( 30 ) == true ) document . write ( " " ) ; else document . write ( " " ) ; if ( isPowerofTwo ( 128 ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function nextPowerOf2 ( n ) {
let p = 1 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ;
while ( p < n ) p <<= 1 ; return p ; }
function memoryUsed ( arr , n ) {
let sum = 0 ;
for ( let i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
let nearest = nextPowerOf2 ( sum ) ; return nearest ; }
let arr = [ 1 , 2 , 3 , 2 ] ; let n = arr . length ; document . write ( memoryUsed ( arr , n ) ) ;
function toggleKthBit ( n , k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
let n = 5 , k = 1 ; document . write ( toggleKthBit ( n , k ) ) ;
function nextPowerOf2 ( n ) { var count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
var n = 0 ; document . write ( nextPowerOf2 ( n ) ) ;
function gcd ( A , B ) { if ( B === 0 ) return A ; return gcd ( B , A % B ) ; }
function lcm ( A , B ) { return ( A * B ) / gcd ( A , B ) ; }
function checkA ( A , B , C , K ) {
var start = 1 ; var end = K ;
var ans = - 1 ; while ( start <= end ) { var mid = parseInt ( ( start + end ) / 2 ) ; var value = A * mid ; var divA = mid - 1 ; var divB = parseInt ( value % B === 0 ? value / B - 1 : value / B ) ; var divC = parseInt ( value % C === 0 ? value / C - 1 : value / C ) ; var divAB = parseInt ( value % lcm ( A , B ) === 0 ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ) ; var divBC = parseInt ( value % lcm ( C , B ) === 0 ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ) ; var divAC = parseInt ( value % lcm ( A , C ) === 0 ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ) ; var divABC = parseInt ( value % lcm ( A , lcm ( B , C ) ) === 0 ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ) ;
var elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem === K - 1 ) { ans = value ; break ; }
else if ( elem > K - 1 ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
function checkB ( A , B , C , K ) {
var start = 1 ; var end = K ;
var ans = - 1 ; while ( start <= end ) { var mid = parseInt ( ( start + end ) / 2 ) ; var value = B * mid ; var divB = mid - 1 ; var divA = parseInt ( value % A === 0 ? value / A - 1 : value / A ) ; var divC = parseInt ( value % C === 0 ? value / C - 1 : value / C ) ; var divAB = parseInt ( value % lcm ( A , B ) === 0 ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ) ; var divBC = parseInt ( value % lcm ( C , B ) === 0 ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ) ; var divAC = parseInt ( value % lcm ( A , C ) === 0 ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ) ; var divABC = parseInt ( value % lcm ( A , lcm ( B , C ) ) === 0 ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ) ;
var elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem === K - 1 ) { ans = value ; break ; }
else if ( elem > K - 1 ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
function checkC ( A , B , C , K ) {
var start = 1 ; var end = K ;
var ans = - 1 ; while ( start <= end ) { var mid = parseInt ( ( start + end ) / 2 ) ; var value = C * mid ; var divC = mid - 1 ; var divB = parseInt ( value % B === 0 ? value / B - 1 : value / B ) ; var divA = parseInt ( value % A === 0 ? value / A - 1 : value / A ) ; var divAB = parseInt ( value % lcm ( A , B ) === 0 ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ) ; var divBC = parseInt ( value % lcm ( C , B ) === 0 ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ) ; var divAC = parseInt ( value % lcm ( A , C ) === 0 ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ) ; var divABC = parseInt ( value % lcm ( A , lcm ( B , C ) ) === 0 ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ) ;
var elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem === K - 1 ) { ans = value ; break ; }
else if ( elem > K - 1 ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
function findKthMultiple ( A , B , C , K ) {
var res = checkA ( A , B , C , K ) ; console . log ( res ) ;
if ( res === - 1 ) res = checkB ( A , B , C , K ) ;
if ( res === - 1 ) res = checkC ( A , B , C , K ) ; return res ; }
var A = 2 , B = 4 , C = 5 , K = 5 ; document . write ( findKthMultiple ( A , B , C , K ) ) ;
function variationStalinsort ( arr ) { let j = 0 ; while ( true ) { let moved = 0 ; for ( let i = 0 ; i < ( arr . length - 1 - j ) ; i ++ ) { if ( arr [ i ] > arr [ i + 1 ] ) {
let index ; let temp ; index = arr [ i ] ; temp = arr [ i + 1 ] ; arr . splice ( i , 1 ) ; arr . splice ( i , 0 , temp ) ; arr [ i ] = temp ; arr . splice ( i + 1 , 1 ) ; arr . splice ( i + 1 , 0 , index ) arr [ i + 1 ] = index ; moved ++ ; } } j ++ ; if ( moved == 0 ) { break ; } } document . write ( " " + arr + " " ) ; }
let arr = [ 2 , 1 , 4 , 3 , 6 , 5 , 8 , 7 , 10 , 9 ] ; let arr1 = [ ] ; for ( let i = 0 ; i < arr . length ; i ++ ) arr1 . push ( arr [ i ] ) ;
variationStalinsort ( arr1 ) ;
function printArray ( arr , N ) {
for ( var i = 0 ; i < N ; i ++ ) { document . write ( arr [ i ] + ' ' ) ; } }
function sortArray ( arr , N ) {
for ( var i = 0 ; i < N ; ) {
if ( arr [ i ] == i + 1 ) { i ++ ; }
else {
var temp1 = arr [ i ] ; var temp2 = arr [ arr [ i ] - 1 ] ; arr [ i ] = temp2 ; arr [ temp1 - 1 ] = temp1 ; } } }
var arr = [ 2 , 1 , 5 , 3 , 4 ] ; var N = arr . length ;
sortArray ( arr , N ) ;
printArray ( arr , N ) ;
function maximum ( value , weight , weight1 , flag , K , index ) {
if ( index >= value . length ) { return 0 ; }
if ( flag == K ) {
var skip = maximum ( value , weight , weight1 , flag , K , index + 1 ) ; var full = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 ) ; }
return Math . max ( full , skip ) ; }
else {
var skip = maximum ( value , weight , weight1 , flag , K , index + 1 ) ; var full = 0 ; var half = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 ) ; }
if ( weight [ index ] / 2 <= weight1 ) { half = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] / 2 , flag , K , index + 1 ) ; }
return Math . max ( full , Math . max ( skip , half ) ) ; } }
var value = [ 17 , 20 , 10 , 15 ] ; var weight = [ 4 , 2 , 7 , 5 ] ; var K = 1 ; var W = 4 ; document . write ( maximum ( value , weight , W , 0 , K , 0 ) ) ;
let N = 1005 ;
class Node { constructor ( data ) { this . left = null ; this . right = null ; this . data = data ; } }
function newNode ( data ) { let node = new Node ( data ) ; return node ; }
let dp = new Array ( N ) ;
function minDominatingSet ( root , covered , compulsory ) {
if ( root == null ) return 0 ;
if ( root . left != null && root . right != null && covered > 0 ) compulsory = 1 ;
if ( dp [ root . data ] [ covered ] [ compulsory ] != - 1 ) return dp [ root . data ] [ covered ] [ compulsory ] ;
if ( compulsory > 0 ) {
return dp [ root . data ] [ covered ] [ compulsory ] = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; }
if ( covered > 0 ) { return dp [ root . data ] [ covered ] [ compulsory ] = Math . min ( 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; }
let ans = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; if ( root . left != null ) { ans = Math . min ( ans , minDominatingSet ( root . left , 0 , 1 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; } if ( root . right != null ) { ans = Math . min ( ans , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 1 ) ) ; }
dp [ root . data ] [ covered ] [ compulsory ] = ans ; return dp [ root . data ] [ covered ] [ compulsory ] ; }
for ( let i = 0 ; i < N ; i ++ ) { dp [ i ] = new Array ( 5 ) ; for ( let j = 0 ; j < 5 ; j ++ ) { dp [ i ] [ j ] = new Array ( 5 ) ; for ( let l = 0 ; l < 5 ; l ++ ) dp [ i ] [ j ] [ l ] = - 1 ; } }
let root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . left . left = newNode ( 3 ) ; root . left . right = newNode ( 4 ) ; root . left . left . left = newNode ( 5 ) ; root . left . left . left . left = newNode ( 6 ) ; root . left . left . left . right = newNode ( 7 ) ; root . left . left . left . right . right = newNode ( 10 ) ; root . left . left . left . left . left = newNode ( 8 ) ; root . left . left . left . left . right = newNode ( 9 ) ; document . write ( minDominatingSet ( root , 0 , 0 ) ) ;
var maxSum = 100 var arrSize = 51
var dp = Array . from ( Array ( arrSize ) , ( ) => Array ( maxSum ) ) ; var visit = Array . from ( Array ( arrSize ) , ( ) => Array ( maxSum ) ) ;
function SubsetCnt ( i , s , arr , n ) {
if ( i == n ) { if ( s == 0 ) return 1 ; else return 0 ; }
if ( visit [ i ] [ s + maxSum ] ) return dp [ i ] [ s + maxSum ] ;
visit [ i ] [ s + maxSum ] = 1 ;
dp [ i ] [ s + maxSum ] = SubsetCnt ( i + 1 , s + arr [ i ] , arr , n ) + SubsetCnt ( i + 1 , s , arr , n ) ;
return dp [ i ] [ s + maxSum ] ; }
var arr = [ 2 , 2 , 2 , - 4 , - 4 ] ; var n = arr . length ; document . write ( SubsetCnt ( 0 , 0 , arr , n ) ) ;
var MAX = 1000 ;
function waysToKAdjacentSetBits ( dp , n , k , currentIndex , adjacentSetBits , lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } if ( dp [ currentIndex ] [ adjacentSetBits ] [ lastBit ] != - 1 ) { return dp [ currentIndex ] [ adjacentSetBits ] [ lastBit ] ; } var noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( ! lastBit ) { noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } dp [ currentIndex ] [ adjacentSetBits ] [ lastBit ] = noOfWays ; return noOfWays ; }
var n = 5 , k = 2 ;
var dp = Array . from ( Array ( MAX ) , ( ) => Array ( MAX ) ) ;
for ( var i = 0 ; i < MAX ; i ++ ) for ( var j = 0 ; j < MAX ; j ++ ) dp [ i ] [ j ] = new Array ( 2 ) . fill ( - 1 ) ;
var totalWays = waysToKAdjacentSetBits ( dp , n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( dp , n , k , 1 , 0 , 0 ) ; document . write ( " " + totalWays + " " ) ;
function printTetra ( n ) { if ( n < 0 ) return ;
var first = 0 , second = 1 ; var third = 1 , fourth = 2 ;
var curr ; if ( n == 0 ) cout << first ; else if ( n == 1 n == 2 ) cout << second ; else if ( n == 3 ) cout << fourth ; else {
for ( var i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } document . write ( curr ) ; } }
var n = 10 ; printTetra ( n ) ;
function countWays ( n ) { let res = new Array ( n + 2 ) ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( let i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
let n = 4 ; document . write ( countWays ( n ) ) ;
function countWays ( n ) {
var a = 1 , b = 2 , c = 4 ;
var d = 0 ; if ( n == 0 n == 1 n == 2 ) return n ; if ( n == 3 ) return c ;
for ( var i = 4 ; i <= n ; i ++ ) { d = c + b + a ; a = b ; b = c ; c = d ; } return d ; }
var n = 4 ; document . write ( countWays ( n ) ) ;
function isPossible ( elements , sum ) { var dp = [ sum + 1 ] ;
dp [ 0 ] = 1 ;
for ( var i = 0 ; i < elements . length ; i ++ ) {
for ( var j = sum ; j >= elements [ i ] ; j -- ) { if ( dp [ j - elements [ i ] ] == 1 ) dp [ j ] = 1 ; } }
if ( dp [ sum ] == 1 ) return true ; return false ; }
var elements = [ 6 , 2 , 5 ] ; var sum = 7 ; if ( isPossible ( elements , sum ) ) document . write ( " " ) ; else document . write ( " " ) ;
function maxTasks ( high , low , n ) {
if ( n <= 0 ) return 0 ;
return Math . max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
let n = 5 ; let high = [ 3 , 6 , 8 , 7 , 6 ] ; let low = [ 1 , 5 , 4 , 5 , 3 ] ; document . write ( maxTasks ( high , low , n ) ) ; ;
function gcd ( a , b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; } function nCr ( n , r ) {
if ( r > n ) return 0 ;
if ( r > n - r ) r = n - r ; mod = 1000000007 ;
var arr = new Array ( r ) ; for ( var i = n - r + 1 ; i <= n ; i ++ ) { arr [ i + r - n - 1 ] = i ; } var ans = 1 ;
for ( var k = 1 ; k < r + 1 ; k ++ ) { var j = 0 , i = k ; while ( j < arr . length ) { var x = gcd ( i , arr [ j ] ) ; if ( x > 1 ) {
arr [ j ] /= x ; i /= x ; }
if ( i == 1 ) break ; j += 1 ; } }
arr . forEach ( function ( i ) { ans = ( ans * i ) % mod ; } ) ; return ans ; }
var n = 5 , r = 2 ; document . write ( " " + n + " " + r + " " + nCr ( n , r ) + " " ) ;
function FindKthChar ( str , K , X ) {
var ans = " " ; var sum = 0 ;
for ( i = 0 ; i < str . length ; i ++ ) {
var digit = parseInt ( str [ i ] ) ;
var range = parseInt ( Math . pow ( digit , X ) ) ; sum += range ;
if ( K <= sum ) { ans = str [ i ] ; break ; } }
return ans ; }
var str = " " ; var K = 9 ; var X = 3 ;
var ans = FindKthChar ( str , K , X ) ; document . write ( ans ) ;
function countSetBits ( n ) { var count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
function totalPairs ( s1 , s2 ) { var count = 0 ; var arr1 = new Array ( 7 ) . fill ( 0 ) ; var arr2 = new Array ( 7 ) . fill ( 0 ) ;
for ( let i = 0 ; i < s1 . length ; i ++ ) { set_bits = countSetBits ( s1 [ i ] . charCodeAt ( 0 ) ) ; arr1 [ set_bits ] += 1 ; }
for ( let i = 0 ; i < s2 . length ; i ++ ) { set_bits = countSetBits ( s2 [ i ] . charCodeAt ( 0 ) ) ; arr2 [ set_bits ] += 1 ; }
for ( let i = 1 ; i < 7 ; i ++ ) { count += arr1 [ i ] * arr2 [ i ] ; }
return count ; }
var s1 = " " ; var s2 = " " ; document . write ( totalPairs ( s1 , s2 ) ) ;
function countSubstr ( str , n , x , y ) {
var tot_count = 0 ;
var count_x = 0 ;
for ( var i = 0 ; i < n ; i ++ ) {
if ( str [ i ] === x ) count_x ++ ;
if ( str [ i ] === y ) tot_count += count_x ; }
return tot_count ; }
var str = " " ; var n = str . length ; var x = " " , y = " " ; document . write ( " " + countSubstr ( str , n , x , y ) ) ;
var OUT = 0 ; var IN = 1 ;
function countWords ( str ) { var state = OUT ;
var wc = 0 ; var i = 0 ;
while ( i < str . length ) {
if ( str [ i ] == ' ' str [ i ] == ' ' str [ i ] == ' ' ) state = OUT ;
else if ( state == OUT ) { state = IN ; ++ wc ; }
++ i ; } return wc ; }
var str = " " ; document . write ( " " + countWords ( str ) ) ;
function nthEnneadecagonal ( n ) {
return ( 17 * n * n - 15 * n ) / 2 ; }
let n = 6 ; document . write ( n + " " ) ; document . write ( nthEnneadecagonal ( n ) ) ;
function areacircumscribed ( a ) { return ( a * a * ( 3.1415 / 2 ) ) ; }
let a = 6 ; document . write ( " " , areacircumscribed ( a ) ) ;
function itemType ( n ) {
let count = 0 ; let day = 1 ;
while ( count + day * ( day + 1 ) / 2 < n ) {
count += day * ( day + 1 ) / 2 ; day ++ ; } for ( let type = day ; type > 0 ; type -- ) {
count += type ;
if ( count >= n ) { return type ; } } }
let N = 10 ; document . write ( itemType ( N ) ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } }
function isSortedDesc ( head ) {
if ( head == null head . next == null ) return true ;
return head . data > head . next . data && isSortedDesc ( head . next ) ; } function newNode ( data ) { var temp = new Node ( ) ; temp . next = null ; temp . data = data ; return temp ; }
var head = newNode ( 7 ) ; head . next = newNode ( 5 ) ; head . next . next = newNode ( 4 ) ; head . next . next . next = newNode ( 3 ) ; if ( isSortedDesc ( head ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
const maxLength = ( str , n , c , k ) => {
let ans = - 1 ;
let cnt = 0 ;
let left = 0 ; for ( let right = 0 ; right < n ; right ++ ) { if ( str [ right ] == c ) { cnt ++ ; }
while ( cnt > k ) { if ( str [ left ] == c ) { cnt -- ; }
left ++ ; }
ans = Math . max ( ans , right - left + 1 ) ; } return ans ; }
const maxConsecutiveSegment = ( S , K ) => { let N = S . length ;
return Math . max ( maxLength ( S , N , ' ' , K ) , maxLength ( S , N , ' ' , K ) ) ; }
let S = " " ; let K = 1 ; document . write ( maxConsecutiveSegment ( S , K ) ) ;
function find ( N ) { var T , F , O ;
F = parseInt ( ( N - 4 ) / 5 ) ;
if ( ( ( N - 5 * F ) % 2 ) == 0 ) { O = 2 ; } else { O = 1 ; }
T = Math . floor ( ( N - 5 * F - O ) / 2 ) ; document . write ( " " + F + " " ) ; document . write ( " " + T + " " ) ; document . write ( " " + O + " " ) ; }
var N = 8 ; find ( N ) ;
function findMaxOccurence ( str , N ) {
for ( var i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' ' ) {
str [ i ] = ' ' ; } } document . write ( str . join ( ' ' ) ) ; }
var str = " " . split ( ' ' ) ; var N = str . length ; findMaxOccurence ( str , N ) ;
function checkInfinite ( s ) {
var flag = 1 ; var N = s . length ;
for ( var i = 0 ; i < N - 1 ; i ++ ) {
if ( s [ i ] == String . fromCharCode ( ( s [ i + 1 ] . charCodeAt ( 0 ) ) + 1 ) ) { continue ; }
else if ( s [ i ] == ' ' && s [ i + 1 ] == ' ' ) { continue ; }
else { flag = 0 ; break ; } }
if ( flag == 0 ) document . write ( " " ) ; else document . write ( " " ) ; }
var s = " " ;
checkInfinite ( s ) ;
function minChangeInLane ( barrier , n ) { let dp = [ 1 , 0 , 1 ] ; for ( let j = 0 ; j < n ; j ++ ) {
let val = barrier [ j ] ; if ( val > 0 ) { dp [ val - 1 ] = 1e6 ; } for ( let i = 0 ; i < 3 ; i ++ ) {
if ( val != i + 1 ) { dp [ i ] = Math . min ( dp [ i ] , Math . min ( dp [ ( i + 1 ) % 3 ] , dp [ ( i + 2 ) % 3 ] ) + 1 ) ; } } }
return Math . min ( dp [ 0 ] , Math . min ( dp [ 1 ] , dp [ 2 ] ) ) ; }
let barrier = [ 0 , 1 , 2 , 3 , 0 ] ; let N = barrier . length ; document . write ( minChangeInLane ( barrier , N ) ) ;
function numWays ( ratings , queries ) {
var dp = Array . from ( Array ( n ) , ( ) => Array ( 10002 ) . fill ( 0 ) ) ;
for ( var i = 0 ; i < k ; i ++ ) dp [ 0 ] [ ratings [ 0 ] [ i ] ] += 1 ;
for ( var i = 1 ; i < n ; i ++ ) {
for ( var sum = 0 ; sum <= 10000 ; sum ++ ) {
for ( var j = 0 ; j < k ; j ++ ) {
if ( sum >= ratings [ i ] [ j ] ) dp [ i ] [ sum ] += dp [ i - 1 ] [ sum - ratings [ i ] [ j ] ] ; } } }
for ( var sum = 1 ; sum <= 10000 ; sum ++ ) { dp [ n - 1 ] [ sum ] += dp [ n - 1 ] [ sum - 1 ] ; }
for ( var q = 0 ; q < 2 ; q ++ ) { var a = queries [ q ] [ 0 ] ; var b = queries [ q ] [ 1 ] ;
document . write ( dp [ n - 1 ] [ b ] - dp [ n - 1 ] [ a - 1 ] + " " ) ; } }
var n = 2 ; var k = 3 ;
var ratings = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] ] ;
var queries = [ [ 6 , 6 ] , [ 1 , 6 ] ] ;
numWays ( ratings , queries ) ;
function numberOfPermWithKInversion ( N , K ) {
let dp = new Array ( 2 ) ; for ( var i = 0 ; i < dp . length ; i ++ ) { dp [ i ] = new Array ( 2 ) ; } let mod = 1000000007 ; for ( let i = 1 ; i <= N ; i ++ ) { for ( let j = 0 ; j <= K ; j ++ ) {
if ( i == 1 ) { dp [ i % 2 ] [ j ] = ( j == 0 ) ? 1 : 0 ; }
else if ( j == 0 ) dp [ i % 2 ] [ j ] = 1 ;
else dp [ i % 2 ] [ j ] = ( dp [ i % 2 ] [ j - 1 ] % mod + ( dp [ 1 - i % 2 ] [ j ] - ( ( Math . max ( j - ( i - 1 ) , 0 ) == 0 ) ? 0 : dp [ 1 - i % 2 ] [ Math . max ( j - ( i - 1 ) , 0 ) - 1 ] ) + mod ) % mod ) % mod ; } }
document . write ( dp [ N % 2 ] [ K ] ) ; }
let N = 3 , K = 2 ;
numberOfPermWithKInversion ( N , K ) ;
function MaxProfit ( treasure , color , n , k , col , A , B ) { let sum = 0 ;
if ( k == n ) return 0 ;
if ( col == color [ k ] ) sum += Math . max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += Math . max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return sum ; }
let A = - 5 , B = 7 ; let treasure = [ 4 , 8 , 2 , 9 ] ; let color = [ 2 , 2 , 6 , 2 ] ; let n = color . length ;
document . write ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) ;
function printTetraRec ( n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
function printTetra ( n ) { document . write ( printTetraRec ( n ) + " " + " " ) ; }
let n = 10 ; printTetra ( n ) ;
let sum = 0 ; function Combination ( a , combi , n , r , depth , index ) {
if ( index == r ) {
let product = 1 ; for ( let i = 0 ; i < r ; i ++ ) product = product * combi [ i ] ;
sum += product ; return ; }
for ( let i = depth ; i < n ; i ++ ) { combi [ index ] = a [ i ] ; Combination ( a , combi , n , r , i + 1 , index + 1 ) ; } }
function allCombination ( a , n ) { for ( let i = 1 ; i <= n ; i ++ ) {
let combi = [ ] ;
Combination ( a , combi , n , i , 0 , 0 ) ;
document . write ( " " + i + " " + sum + " " ) ; sum = 0 ; } }
let n = 5 ; let a = [ ] ;
for ( let i = 0 ; i < n ; i ++ ) a [ i ] = i + 1 ;
allCombination ( a , n ) ;
function max ( x , y ) { return ( x > y ? x : y ) ; }
function maxTasks ( high , low , n ) {
var task_dp = Array . from ( { length : n + 1 } , ( _ , i ) => 0 ) ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( i = 2 ; i <= n ; i ++ ) task_dp [ i ] = Math . max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
var n = 5 ; var high = [ 3 , 6 , 8 , 7 , 6 ] ; var low = [ 1 , 5 , 4 , 5 , 3 ] ; document . write ( maxTasks ( high , low , n ) ) ;
function PermutationCoeff ( n , k ) { let P = 1 ;
for ( let i = 0 ; i < k ; i ++ ) P *= ( n - i ) ; return P ; }
let n = 10 , k = 2 ; document . write ( " " + n + " " + k + " " + PermutationCoeff ( n , k ) ) ;
function findPartition ( arr , n ) { var sum = 0 ; var i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; var part = Array ( parseInt ( sum / 2 ) + 1 ) . fill ( ) . map ( ( ) => Array ( n + 1 ) . fill ( 0 ) ) ;
for ( i = 0 ; i <= n ; i ++ ) part [ 0 ] [ i ] = true ;
for ( i = 1 ; i <= parseInt ( sum / 2 ) ; i ++ ) part [ i ] [ 0 ] = false ;
for ( i = 1 ; i <= parseInt ( sum / 2 ) ; i ++ ) { for ( j = 1 ; j <= n ; j ++ ) { part [ i ] [ j ] = part [ i ] [ j - 1 ] ; if ( i >= arr [ j - 1 ] ) part [ i ] [ j ] = part [ i ] [ j ] || part [ i - arr [ j - 1 ] ] [ j - 1 ] ; } }
return part [ parseInt ( sum / 2 ) ] [ n ] ; }
var arr = [ 3 , 1 , 1 , 2 , 2 , 1 ] ; var n = arr . length ;
if ( findPartition ( arr , n ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function minimumOperations ( orig_str , m , n ) {
let orig = orig_str ;
let turn = 1 ; let j = 1 ;
for ( let i = 0 ; i < orig_str . length ; i ++ ) {
let m_cut = orig_str . substring ( orig_str . length - m ) ; orig_str = orig_str . substring ( 0 , orig_str . length - m ) ;
orig_str = m_cut + orig_str ;
j = j + 1 ;
if ( orig != orig_str ) { turn = turn + 1 ;
let n_cut = orig_str . substring ( orig_str . length - n ) ; orig_str = orig_str . substring ( 0 , orig_str . length - n ) ;
orig_str = n_cut + orig_str ;
j = j + 1 ; }
if ( orig == orig_str ) { break ; }
turn = turn + 1 ; } document . write ( turn ) ; }
let S = " " ; let X = 5 , Y = 3 ;
minimumOperations ( S , X , Y ) ;
function KMPSearch ( pat , txt ) { let M = pat . length ; let N = txt . length ;
let lps = new Array ( M ) ; lps . fill ( 0 ) ;
computeLPSArray ( pat , M , lps ) ;
let i = 0 ; let j = 0 ; while ( i < N ) { if ( pat [ j ] == txt [ i ] ) { j ++ ; i ++ ; } if ( j == M ) { return i - j ; }
else if ( i < N && pat [ j ] != txt [ i ] ) {
if ( j != 0 ) j = lps [ j - 1 ] ; else i = i + 1 ; } } return 0 ; }
function computeLPSArray ( pat , M , lps ) {
let len = 0 ;
lps [ 0 ] = 0 ;
let i = 1 ; while ( i < M ) { if ( pat [ i ] == pat [ len ] ) { len ++ ; lps [ i ] = len ; i ++ ; }
else {
if ( len != 0 ) { len = lps [ len - 1 ] ; } else { lps [ i ] = 0 ; i ++ ; } } } }
function countRotations ( s ) {
let s1 = s . substring ( 1 , s . length ) + s ;
let pat = s . split ( ' ' ) ; let text = s1 . split ( ' ' ) ;
return 1 + KMPSearch ( pat , text ) ; }
let s1 = " " ; document . write ( countRotations ( s1 ) ) ;
let dfa = 0 ;
function start ( c ) {
if ( c == ' ' c == ' ' ) dfa = 1 ; }
function state1 ( c ) {
if ( c == ' ' c == ' ' ) dfa = 1 ;
else if ( c == ' ' c == ' ' ) dfa = 2 ;
else dfa = 0 ; }
function state2 ( c ) {
if ( c == ' ' c == ' ' ) dfa = 3 ; else dfa = 0 ; }
function state3 ( c ) {
if ( c == ' ' c == ' ' ) dfa = 1 ; else dfa = 0 ; } function isAccepted ( str ) {
let len = str . length ; for ( let i = 0 ; i < len ; i ++ ) { if ( dfa == 0 ) start ( str [ i ] ) ; else if ( dfa == 1 ) state1 ( str [ i ] ) ; else if ( dfa == 2 ) state2 ( str [ i ] ) ; else state3 ( str [ i ] ) ; } return ( dfa != 3 ) ; }
let str = " " . split ( ' ' ) ; if ( isAccepted ( str ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
var parent = new Array ( 26 ) . fill ( 0 ) ;
function find ( x ) { if ( x !== parent [ x ] ) return ( parent [ x ] = find ( parent [ x ] ) ) ; return x ; }
function join ( x , y ) { var px = find ( x ) ; var pz = find ( y ) ; if ( px !== pz ) { parent [ pz ] = px ; } }
function convertible ( s1 , s2 ) {
var mp = { } ; for ( var i = 0 ; i < s1 . length ; i ++ ) { if ( ! mp . hasOwnProperty ( s1 [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ) ) { mp [ s1 [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] = s2 [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ; } else { if ( mp [ s1 [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] !== s2 [ i ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ) return false ; } }
for ( const [ key , value ] of Object . entries ( mp ) ) { if ( key === value ) continue ; else { if ( find ( key ) == find ( value ) ) return false ; else join ( key , value ) ; } } return true ; }
function initialize ( ) { for ( var i = 0 ; i < 26 ; i ++ ) { parent [ i ] = i ; } }
var s1 , s2 ; s1 = " " ; s2 = " " ; initialize ( ) ; if ( convertible ( s1 , s2 ) ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
let SIZE = 26 ;
function SieveOfEratosthenes ( prime , p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( let p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( let i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } return prime ; }
function printChar ( str , n ) { let prime = [ ] ; for ( let i = 0 ; i < n + 1 ; i ++ ) { prime . push ( true ) ; }
prime = SieveOfEratosthenes ( prime , str . length + 1 ) ;
let freq = [ ] ;
for ( let i = 0 ; i < 26 ; i ++ ) { freq . push ( 0 ) ; }
for ( let i = 0 ; i < n ; i ++ ) freq [ str . charCodeAt ( i ) - 97 ] ++ ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( prime [ freq [ str . charCodeAt ( i ) - 97 ] ] ) { document . write ( str [ i ] ) ; } } }
let str = " " ; let n = str . length ; printChar ( str , n ) ;
function prime ( n ) { if ( n <= 1 ) return false ; let max_div = Math . floor ( Math . sqrt ( n ) ) ; for ( let i = 2 ; i < 1 + max_div ; i ++ ) { if ( n % i == 0 ) return false ; } return true ; } function checkString ( s ) {
let freq = new Map ( ) ; for ( let i = 0 ; i < s . length ; i ++ ) { if ( ! freq . has ( s [ i ] ) ) freq . set ( s [ i ] , 0 ) ; freq . set ( s [ i ] , freq . get ( s [ i ] ) + 1 ) ; }
for ( let i = 0 ; i < s . length ; i ++ ) { if ( prime ( freq . get ( s [ i ] ) ) ) document . write ( s [ i ] ) ; } }
let s = " " ;
checkString ( s ) ;
let SIZE = 26 ;
function printChar ( str , n ) {
let freq = new Array ( SIZE ) ;
for ( let i = 0 ; i < n ; i ++ ) { freq [ str . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ] ++ ; }
for ( let i = 0 ; i < n ; i ++ ) {
if ( freq [ str . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ] % 2 == 0 ) { document . write ( str [ i ] ) ; } } }
let str = " " ; let n = str . length ; printChar ( str , n ) ;
function CompareAlphanumeric ( str1 , str2 ) {
let i , j ; i = 0 ; j = 0 ;
let len1 = str1 . length ;
let len2 = str2 . length ;
while ( i <= len1 && j <= len2 ) {
while ( i < len1 && ( ! ( ( str1 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str1 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) || ( str1 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str1 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) || ( str1 [ i ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str1 [ i ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) ) ) ) { i ++ ; }
while ( j < len2 && ( ! ( ( str2 [ j ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str2 [ j ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) || ( str2 [ j ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str2 [ j ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) || ( str2 [ j ] . charCodeAt ( ) >= ' ' . charCodeAt ( ) && str2 [ j ] . charCodeAt ( ) <= ' ' . charCodeAt ( ) ) ) ) ) { j ++ ; }
if ( i == len1 && j == len2 ) { return true ; }
else if ( str1 [ i ] != str2 [ j ] ) { return false ; }
else { i ++ ; j ++ ; } }
return false ; }
function CompareAlphanumericUtil ( str1 , str2 ) { let res ;
res = CompareAlphanumeric ( str1 . split ( ' ' ) , str2 . split ( ' ' ) ) ;
if ( res == true ) { document . write ( " " + " " ) ; }
else { document . write ( " " ) ; } }
let str1 , str2 ; str1 = " " ; str2 = " " ; CompareAlphanumericUtil ( str1 , str2 ) ; str1 = " " ; str2 = " " ; CompareAlphanumericUtil ( str1 , str2 ) ;
function solveQueries ( str , query ) {
let len = str . length ;
let Q = query . length ;
let pre = new Array ( len ) ; for ( let i = 0 ; i < len ; i ++ ) { pre [ i ] = new Array ( 26 ) ; for ( let j = 0 ; j < 26 ; j ++ ) { pre [ i ] [ j ] = 0 ; } }
for ( let i = 0 ; i < len ; i ++ ) {
pre [ i ] [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
if ( i > 0 ) {
for ( let j = 0 ; j < 26 ; j ++ ) pre [ i ] [ j ] += pre [ i - 1 ] [ j ] ; } }
for ( let i = 0 ; i < Q ; i ++ ) {
let l = query [ i ] [ 0 ] ; let r = query [ i ] [ 1 ] ; let maxi = 0 ; let c = ' ' ;
for ( let j = 0 ; j < 26 ; j ++ ) {
let times = pre [ r ] [ j ] ;
if ( l > 0 ) times -= pre [ l - 1 ] [ j ] ;
if ( times > maxi ) { maxi = times ; c = String . fromCharCode ( ' ' . charCodeAt ( 0 ) + j ) ; } }
document . write ( " " + ( i + 1 ) + " " + c + " " ) ; } }
let str = " " ; let query = [ [ 0 , 1 ] , [ 1 , 6 ] , [ 5 , 6 ] ] ; solveQueries ( str , query ) ;
function startsWith ( str , pre ) { let strLen = str . length ; let preLen = pre . length ; let i = 0 , j = 0 ;
while ( i < strLen && j < preLen ) {
if ( str [ i ] != pre [ j ] ) return false ; i ++ ; j ++ ; }
return true ; }
function endsWith ( str , suff ) { let i = str . length - 1 ; let j = suff . length - 1 ;
while ( i >= 0 && j >= 0 ) {
if ( str [ i ] != suff [ j ] ) return false ; i -- ; j -- ; }
return true ; }
function checkString ( str , a , b ) {
if ( str . length != a . length + b . length ) return false ;
if ( startsWith ( str , a ) ) {
if ( endsWith ( str , b ) ) return true ; }
if ( startsWith ( str , b ) ) {
if ( endsWith ( str , a ) ) return true ; } return false ; }
let str = " " ; let a = " " ; let b = " " ; if ( checkString ( str , a , b ) ) document . write ( " " ) ; else document . write ( " " ) ;
let SIZE = 26 ;
function printChar ( str , n ) {
let freq = [ ] ;
for ( let i = 0 ; i < n ; i ++ ) freq [ str . charCodeAt ( i ) - 97 ] ++ ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( freq [ str . charCodeAt ( i ) - 97 ] % 2 == 1 ) { document . write ( str [ i ] ) ; } } }
let str = " " ; let n = str . length ; printChar ( str , n ) ;
function isupper ( str ) { return str === str . toUpperCase ( ) ; }
function minOperations ( str , n ) {
var i , lastUpper = - 1 , firstLower = - 1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( isupper ( str [ i ] ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( islower ( str [ i ] ) ) { firstLower = i ; break ; } }
if ( lastUpper === - 1 firstLower === - 1 ) return 0 ;
var countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( isupper ( str [ i ] ) ) { countUpper ++ ; } }
var countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( islower ( str [ i ] ) ) { countLower ++ ; } }
return Math . min ( countLower , countUpper ) ; }
var str = " " ; var n = str . length ; document . write ( minOperations ( str , n ) + " " ) ;
function Betrothed_Sum ( n ) {
let Set = [ ] ; for ( let number_1 = 1 ; number_1 < n ; number_1 ++ ) {
let sum_divisor_1 = 1 ;
let i = 2 ; while ( i * i <= number_1 ) { if ( number_1 % i == 0 ) { sum_divisor_1 = sum_divisor_1 + i ; if ( i * i != number_1 ) sum_divisor_1 += parseInt ( number_1 / i ) ; } i ++ ; } if ( sum_divisor_1 > number_1 ) { let number_2 = sum_divisor_1 - 1 ; let sum_divisor_2 = 1 ; let j = 2 ; while ( j * j <= number_2 ) { if ( number_2 % j == 0 ) { sum_divisor_2 += j ; if ( j * j != number_2 ) sum_divisor_2 += parseInt ( number_2 / j ) ; } j = j + 1 ; } if ( ( sum_divisor_2 == number_1 + 1 ) && number_1 <= n && number_2 <= n ) { Set . push ( number_1 ) ; Set . push ( number_2 ) ; } } }
let Summ = 0 ; for ( let i = 0 ; i < Set . length ; i ++ ) { if ( Set [ i ] <= n ) Summ += Set [ i ] ; } return Summ ; }
let n = 78 ; document . write ( Betrothed_Sum ( n ) ) ;
function rainDayProbability ( a , n ) { let count = 0 , m ;
for ( let i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
let a = [ 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 ] ; let n = a . length ; document . write ( rainDayProbability ( a , n ) ) ;
function Series ( n ) { let sums = 0.0 ; for ( let i = 1 ; i < n + 1 ; i ++ ) { ser = 1 / Math . pow ( i , i ) ; sums += ser ; } return sums ; }
let n = 3 ; let res = Math . round ( Series ( n ) * 100000 ) / 100000 ; document . write ( res ) ;
function lexicographicallyMaximum ( S , N ) {
let M = new Map ( ) ;
for ( let i = 0 ; i < N ; ++ i ) { if ( M . has ( S [ i ] ) ) M . set ( S [ i ] , M . get ( S [ i ] ) + 1 ) ; else M . set ( S [ i ] , 1 ) ; }
let V = [ ] ; for ( let i = ' ' . charCodeAt ( ) ; i < ( ' ' . charCodeAt ( ) + Math . min ( N , 25 ) ) ; ++ i ) { if ( M . has ( String . fromCharCode ( i ) ) == false ) { V . push ( String . fromCharCode ( i ) ) ; } }
let j = V . length - 1 ;
for ( let i = 0 ; i < N ; ++ i ) {
if ( S [ i ] . charCodeAt ( ) >= ( ' ' . charCodeAt ( ) + Math . min ( N , 25 ) ) || ( M . has ( S [ i ] ) && M . get ( S [ i ] ) > 1 ) ) { if ( V [ j ] . charCodeAt ( ) < S [ i ] . charCodeAt ( ) ) continue ;
M . set ( S [ i ] , M . get ( S [ i ] ) - 1 ) ;
S = S . substr ( 0 , i ) + V [ j ] + S . substr ( i + 1 ) ;
j -- ; } if ( j < 0 ) break ; } let l = 0 ;
for ( let i = N - 1 ; i >= 0 ; i -- ) { if ( l > j ) break ; if ( S [ i ] . charCodeAt ( ) >= ( ' ' . charCodeAt ( ) + Math . min ( N , 25 ) ) || M . has ( S [ i ] ) && M . get ( S [ i ] ) > 1 ) {
M . set ( S [ i ] , M . get ( S [ i ] ) - 1 ) ;
S = S . substr ( 0 , i ) + V [ l ] + S . substr ( i + 1 ) ;
l ++ ; } }
return S ; }
let S = " " ; let N = S . length ;
document . write ( lexicographicallyMaximum ( S , N ) ) ;
function isConsistingSubarrayUtil ( arr , n ) {
var mp = new Map ( ) ;
for ( var i = 0 ; i < n ; ++ i ) {
if ( mp . has ( arr [ i ] ) ) { mp . set ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . set ( arr [ i ] , 1 ) ; } } var ans = false ;
mp . forEach ( ( value , key ) => {
if ( value > 1 ) { ans = true ; } } ) ; if ( ans ) return true ;
return false ; }
function isConsistingSubarray ( arr , N ) { if ( isConsistingSubarrayUtil ( arr , N ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
var arr = [ 1 , 2 , 3 , 4 , 5 , 1 ] ;
var N = arr . length ;
isConsistingSubarray ( arr , N ) ;
function createhashmap ( Max ) {
var hashmap = new Set ( ) ;
var curr = 1 ;
var prev = 0 ;
hashmap . add ( prev ) ;
while ( curr <= Max ) {
hashmap . add ( curr ) ;
var temp = curr ;
curr = curr + prev ;
prev = temp ; } return hashmap ; }
function SieveOfEratosthenes ( Max ) {
var isPrime = Array ( Max + 1 ) . fill ( true ) ; isPrime [ 0 ] = false ; isPrime [ 1 ] = false ;
for ( var p = 2 ; p * p <= Max ; p ++ ) {
if ( isPrime [ p ] ) {
for ( var i = p * p ; i <= Max ; i += p ) {
isPrime [ i ] = false ; } } } return isPrime ; }
function cntFibonacciPrime ( arr , N ) {
var Max = arr [ 0 ] ;
for ( var i = 1 ; i < N ; i ++ ) {
Max = Math . max ( Max , arr [ i ] ) ; }
var isPrime = SieveOfEratosthenes ( Max ) ;
var hashmap = createhashmap ( Max ) ;
for ( var i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 1 ) continue ;
if ( hashmap . has ( arr [ i ] ) && ! isPrime [ arr [ i ] ] ) {
document . write ( arr [ i ] + " " ) ; } } }
var arr = [ 13 , 55 , 7 , 3 , 5 , 21 , 233 , 144 , 89 ] ; var N = arr . length ; cntFibonacciPrime ( arr , N ) ;
function key ( N ) {
let num = " " + N . toString ( ) ; let ans = 0 ; let j = 0 ;
for ( j = 0 ; j < num . length ; j ++ ) {
if ( ( num [ j ] . charCodeAt ( ) - 48 ) % 2 == 0 ) { let add = 0 ; let i ;
for ( i = j ; j < num . length ; j ++ ) { add += num [ j ] . charCodeAt ( ) - 48 ;
if ( add % 2 == 1 ) break ; } if ( add == 0 ) { ans *= 10 ; } else { let digit = Math . floor ( Math . log10 ( add ) + 1 ) ; ans *= parseInt ( Math . pow ( 10 , digit ) , 10 ) ;
ans += add ; }
i = j ; } else {
let add = 0 ; let i ;
for ( i = j ; j < num . length ; j ++ ) { add += num [ j ] . charCodeAt ( ) - 48 ;
if ( add % 2 == 0 ) { break ; } } if ( add == 0 ) { ans *= 10 ; } else { let digit = Math . floor ( Math . log10 ( add ) + 1 ) ; ans *= parseInt ( Math . pow ( 10 , digit ) , 10 ) ;
ans += add ; }
i = j ; } }
if ( j + 1 >= num . length ) { return ans ; } else { return ans += num [ num . length - 1 ] . charCodeAt ( ) - 48 ; } }
let N = 1667848271 ; document . write ( key ( N ) ) ;
function sentinelSearch ( arr , n , key ) {
var last = arr [ n - 1 ] ;
arr [ n - 1 ] = key ; var i = 0 ; while ( arr [ i ] != key ) i ++ ;
arr [ n - 1 ] = last ; if ( ( i < n - 1 ) || ( arr [ n - 1 ] == key ) ) document . write ( key + " " + i ) ; else document . write ( " " ) ; }
var arr = [ 10 , 20 , 180 , 30 , 60 , 50 , 110 , 100 , 70 ] ; var n = arr . length ; var key = 180 ; sentinelSearch ( arr , n , key ) ;
function maximum_middle_value ( n , k , arr ) {
let ans = - 1 ;
let low = Math . floor ( ( n + 1 - k ) / 2 ) ; let high = Math . floor ( ( ( n + 1 - k ) / 2 ) + k ) ;
for ( let i = low ; i <= high ; i ++ ) {
ans = Math . max ( ans , arr [ i - 1 ] ) ; }
return ans ; }
let n = 5 , k = 2 ; let arr = [ 9 , 5 , 3 , 7 , 10 ] ; document . write ( maximum_middle_value ( n , k , arr ) + " " ) ; n = 9 ; k = 3 ; let arr1 = [ 2 , 4 , 3 , 9 , 5 , 8 , 7 , 6 , 10 ] ; document . write ( maximum_middle_value ( n , k , arr1 ) + " " ) ;
function ternarySearch ( l , r , key , ar ) { if ( r >= l ) {
let mid1 = l + parseInt ( ( r - l ) / 3 , 10 ) ; let mid2 = r - parseInt ( ( r - l ) / 3 , 10 ) ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
return ternarySearch ( l , mid1 - 1 , key , ar ) ; } else if ( key > ar [ mid2 ] ) {
return ternarySearch ( mid2 + 1 , r , key , ar ) ; } else {
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) ; } }
return - 1 ; } let l , r , p , key ;
let ar = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ] ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
document . write ( " " + key + " " + p + " " ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
document . write ( " " + key + " " + p ) ;
function findmin ( p , n ) { let a = 0 , b = 0 , c = 0 , d = 0 ; for ( let i = 0 ; i < n ; i ++ ) {
if ( p [ i ] [ 0 ] <= 0 ) a ++ ;
else if ( p [ i ] [ 0 ] >= 0 ) b ++ ;
if ( p [ i ] [ 1 ] >= 0 ) c ++ ;
else if ( p [ i ] [ 1 ] <= 0 ) d ++ ; } return Math . min ( Math . min ( a , b ) , Math . min ( c , d ) ) ; }
let p = [ [ 1 , 1 ] , [ 2 , 2 ] , [ - 1 , - 1 ] , [ - 2 , 2 ] ] let n = p . length ; document . write ( findmin ( p , n ) ) ;
function maxOps ( a , b , c ) {
let arr = [ a , b , c ] ;
let count = 0 ; while ( 1 ) {
arr . sort ( ) ;
if ( ! arr [ 0 ] && ! arr [ 1 ] ) break ;
arr [ 1 ] -= 1 ; arr [ 2 ] -= 1 ;
count += 1 ; }
document . write ( count ) ; }
let a = 4 , b = 3 , c = 2 ; maxOps ( a , b , c ) ;
var MAX = 26 ;
function getSortedString ( s , n ) {
var lower = Array ( MAX ) . fill ( 0 ) ; var upper = Array ( MAX ) . fill ( 0 ) ; for ( var i = 0 ; i < n ; i ++ ) {
if ( ( s [ i ] ) == s [ i ] . toLowerCase ( ) ) lower [ s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
else if ( s [ i ] = s [ i ] . toUpperCase ( ) ) upper [ s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; }
var i = 0 , j = 0 ; while ( i < MAX && lower [ i ] == 0 ) i ++ ; while ( j < MAX && upper [ j ] == 0 ) j ++ ;
for ( var k = 0 ; k < n ; k ++ ) {
if ( s [ k ] == s [ k ] . toLowerCase ( ) ) { while ( lower [ i ] == 0 ) i ++ ; s [ k ] = String . fromCharCode ( i + ' ' . charCodeAt ( 0 ) ) ;
lower [ i ] -- ; }
else if ( s [ k ] == s [ k ] . toUpperCase ( ) ) { while ( upper [ j ] == 0 ) j ++ ; s [ k ] = String . fromCharCode ( j + ' ' . charCodeAt ( 0 ) ) ;
upper [ j ] -- ; } }
return s . join ( ' ' ) ; }
var s = " " ; var n = s . length ; document . write ( getSortedString ( s . split ( ' ' ) , n ) ) ;
let SIZE = 26 ;
function printCharWithFreq ( str ) {
let n = str . length ;
let freq = new Array ( SIZE ) ; for ( let i = 0 ; i < freq . length ; i ++ ) { freq [ i ] = 0 ; }
for ( let i = 0 ; i < n ; i ++ ) freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] != 0 ) {
document . write ( str [ i ] ) ; document . write ( freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] + " " ) ;
freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] = 0 ; } } }
let str = " " ; printCharWithFreq ( str ) ;
var s = [ " " , " " , " " , " " , " " , " " ] ; var ans = " " ; for ( var i = 5 ; i >= 0 ; i -- ) { ans += s [ i ] + " " ; } document . write ( " " + " " ) ; document . write ( ans . slice ( 0 , ans . length - 1 ) ) ;
function SieveOfEratosthenes ( prime , n ) { for ( let p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( let i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } }
function segregatePrimeNonPrime ( prime , arr , N ) {
SieveOfEratosthenes ( prime , 10000000 ) ;
let left = 0 , right = N - 1 ;
while ( left < right ) {
while ( prime [ arr [ left ] ] ) left ++ ;
while ( ! prime [ arr [ right ] ] ) right -- ;
if ( left < right ) {
let temp = arr [ left ] ; arr [ left ] = arr [ right ] ; arr [ right ] = temp ; left ++ ; right -- ; } }
for ( let i = 0 ; i < N ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let prime = Array . from ( { length : 10000001 } , ( _ , i ) => true ) ; let arr = [ 2 , 3 , 4 , 6 , 7 , 8 , 9 , 10 ] ; let N = arr . length ;
segregatePrimeNonPrime ( prime , arr , N ) ;
function findDepthRec ( tree , n , index ) { if ( index >= n tree [ index ] == ' ' ) return 0 ;
index ++ ; let left = findDepthRec ( tree , n , index ) ;
index ++ ; let right = findDepthRec ( tree , n , index ) ; return Math . max ( left , right ) + 1 ; }
function findDepth ( tree , n ) { let index = 0 ; return ( findDepthRec ( tree , n , index ) ) ; }
let tree = " " . split ( ' ' ) ; let n = tree . length ; document . write ( findDepth ( tree , n ) ) ;
class Node { constructor ( ) { this . key = 0 ; this . left = null , this . right = null ; } }
function newNode ( item ) { var temp = new Node ( ) ; temp . key = item ; temp . left = null ; temp . right = null ; return temp ; }
function insert ( node , key ) {
if ( node == null ) return newNode ( key ) ;
if ( key < node . key ) node . left = insert ( node . left , key ) ; else if ( key > node . key ) node . right = insert ( node . right , key ) ;
return node ; }
function findMaxforN ( root , N ) {
if ( root == null ) return - 1 ; if ( root . key == N ) return N ;
else if ( root . key < N ) { var k = findMaxforN ( root . right , N ) ; if ( k == - 1 ) return root . key ; else return k ; }
else if ( root . key > N ) return findMaxforN ( root . left , N ) ; return - 1 ; }
var N = 4 ;
var root = null ; root = insert ( root , 25 ) ; insert ( root , 2 ) ; insert ( root , 1 ) ; insert ( root , 3 ) ; insert ( root , 12 ) ; insert ( root , 9 ) ; insert ( root , 21 ) ; insert ( root , 19 ) ; insert ( root , 25 ) ; document . write ( findMaxforN ( root , N ) ) ;
class Node { constructor ( val ) { this . data = val ; this . left = null ; this . right = null ; } }
function createNode ( x ) { var p = new Node ( ) ; p . data = x ; p . left = p . right = null ; return p ; }
function insertNode ( root , x ) { var p = root , q = null ; while ( p != null ) { q = p ; if ( p . data < x ) p = p . right ; else p = p . left ; } if ( q == null ) p = createNode ( x ) ; else { if ( q . data < x ) q . right = createNode ( x ) ; else q . left = createNode ( x ) ; } }
function maxelpath ( q , x ) { var p = q ; var mx = - 1 ;
while ( p . data != x ) { if ( p . data > x ) { mx = Math . max ( mx , p . data ) ; p = p . left ; } else { mx = Math . max ( mx , p . data ) ; p = p . right ; } } return Math . max ( mx , x ) ; }
function maximumElement ( root , x , y ) { var p = root ;
while ( ( x < p . data && y < p . data ) || ( x > p . data && y > p . data ) ) {
if ( x < p . data && y < p . data ) p = p . left ;
else if ( x > p . data && y > p . data ) p = p . right ; }
return Math . max ( maxelpath ( p , x ) , maxelpath ( p , y ) ) ; }
var arr = [ 18 , 36 , 9 , 6 , 12 , 10 , 1 , 8 ] ; var a = 1 , b = 10 ; var n = arr . length ;
var root = createNode ( arr [ 0 ] ) ;
for ( i = 1 ; i < n ; i ++ ) insertNode ( root , arr [ i ] ) ; document . write ( maximumElement ( root , a , b ) ) ;
class Node { constructor ( ) { this . left = null , this . right = null ; this . info = 0 ;
this . lthread = false ;
this . rthread = false ; } }
function insert ( root , ikey ) {
var ptr = root ;
var par = null ; while ( ptr != null ) {
if ( ikey == ( ptr . info ) ) { document . write ( " " ) ; return root ; }
par = ptr ;
if ( ikey < ptr . info ) { if ( ptr . lthread == false ) ptr = ptr . left ; else break ; }
else { if ( ptr . rthread == false ) ptr = ptr . right ; else break ; } }
var tmp = new Node ( ) ; tmp . info = ikey ; tmp . lthread = true ; tmp . rthread = true ; if ( par == null ) { root = tmp ; tmp . left = null ; tmp . right = null ; } else if ( ikey < ( par . info ) ) { tmp . left = par . left ; tmp . right = par ; par . lthread = false ; par . left = tmp ; } else { tmp . left = par ; tmp . right = par . right ; par . rthread = false ; par . right = tmp ; } return root ; }
function inorderSuccessor ( ptr ) {
if ( ptr . rthread == true ) return ptr . right ;
ptr = ptr . right ; while ( ptr . lthread == false ) ptr = ptr . left ; return ptr ; }
function inorder ( root ) { if ( root == null ) document . write ( " " ) ;
var ptr = root ; while ( ptr . lthread == false ) ptr = ptr . left ;
while ( ptr != null ) { document . write ( ptr . info + " " ) ; ptr = inorderSuccessor ( ptr ) ; } }
var root = null ; root = insert ( root , 20 ) ; root = insert ( root , 10 ) ; root = insert ( root , 30 ) ; root = insert ( root , 5 ) ; root = insert ( root , 16 ) ; root = insert ( root , 14 ) ; root = insert ( root , 17 ) ; root = insert ( root , 13 ) ; inorder ( root ) ;
class Node { constructor ( ) { this . left = null , this . right = null ; this . info = 0 ;
this . lthread = false ;
this . rthread = false ; } }
function checkHV ( arr , N , M ) {
let horizontal = true ; let vertical = true ;
for ( let i = 0 , k = N - 1 ; i < parseInt ( N / 2 , 10 ) ; i ++ , k -- ) {
for ( let j = 0 ; j < M ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } }
for ( let i = 0 , k = M - 1 ; i < parseInt ( M / 2 , 10 ) ; i ++ , k -- ) {
for ( let j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } } if ( ! horizontal && ! vertical ) document . write ( " " ) ; else if ( horizontal && ! vertical ) document . write ( " " ) ; else if ( vertical && ! horizontal ) document . write ( " " ) ; else document . write ( " " ) ; }
let mat = [ [ 1 , 0 , 1 ] , [ 0 , 0 , 0 ] , [ 1 , 0 , 1 ] ] ; checkHV ( mat , 3 , 3 ) ;
let R = 3 ; let C = 4 ;
function gcd ( a , b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
function replacematrix ( mat , n , m ) { let rgcd = new Array ( R ) ; rgcd . fill ( 0 ) ; let cgcd = new Array ( C ) ; cgcd . fill ( 0 ) ;
for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < m ; j ++ ) { rgcd [ i ] = gcd ( rgcd [ i ] , mat [ i ] [ j ] ) ; cgcd [ j ] = gcd ( cgcd [ j ] , mat [ i ] [ j ] ) ; } }
for ( let i = 0 ; i < n ; i ++ ) for ( let j = 0 ; j < m ; j ++ ) mat [ i ] [ j ] = Math . max ( rgcd [ i ] , cgcd [ j ] ) ; }
let m = [ [ 1 , 2 , 3 , 3 ] , [ 4 , 5 , 6 , 6 ] , [ 7 , 8 , 9 , 9 ] ] ; replacematrix ( m , R , C ) ; for ( let i = 0 ; i < R ; i ++ ) { for ( let j = 0 ; j < C ; j ++ ) document . write ( m [ i ] [ j ] + " " ) ; document . write ( " " ) ; }
let N = 4 ;
function add ( A , B , C ) { let i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] ; }
let A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] ; let B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] ; let C = new Array ( N ) ; for ( let k = 0 ; k < N ; k ++ ) C [ k ] = new Array ( N ) ; let i , j ; add ( A , B , C ) ; document . write ( " " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) document . write ( C [ i ] [ j ] + " " ) ; document . write ( " " ) ; }
var N = 4 ;
function subtract ( A , B , C ) { var i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] ; }
var A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] ; var B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] ; var C = Array . from ( Array ( N ) , ( ) => Array ( N ) ) ; var i , j ; subtract ( A , B , C ) ; document . write ( " " + " " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) document . write ( C [ i ] [ j ] + " " ) ; document . write ( " " ) ; }
function linearSearch ( arr , n ) { let i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == i ) return i ; }
return - 1 ; }
let arr = [ - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ] ; let n = arr . length ; document . write ( " " + linearSearch ( arr , n ) ) ;
function binarySearch ( arr , low , high ) { if ( high >= low ) { let mid = Math . floor ( ( low + high ) / 2 ) ;
if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return - 1 ; }
let arr = [ - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 ] ; let n = arr . length ; document . write ( " " + binarySearch ( arr , 0 , n - 1 ) ) ;
function maxTripletSum ( arr , n ) {
let sum = - 1000000 ; for ( let i = 0 ; i < n ; i ++ ) for ( let j = i + 1 ; j < n ; j ++ ) for ( let k = j + 1 ; k < n ; k ++ ) if ( sum < arr [ i ] + arr [ j ] + arr [ k ] ) sum = arr [ i ] + arr [ j ] + arr [ k ] ; return sum ; }
let arr = [ 1 , 0 , 8 , 6 , 4 , 2 ] ; let n = arr . length ; document . write ( maxTripletSum ( arr , n ) ) ;
function maxTripletSum ( arr , n ) {
arr . sort ( ) ;
return arr [ n - 1 ] + arr [ n - 2 ] + arr [ n - 3 ] ; }
let arr = [ 1 , 0 , 8 , 6 , 4 , 2 ] ; let n = arr . length ; document . write ( maxTripletSum ( arr , n ) ) ;
function maxTripletSum ( arr , n ) {
let maxA = Number . MIN_SAFE_INTEGER ; let maxB = Number . MIN_SAFE_INTEGER ; let maxC = Number . MIN_SAFE_INTEGER ; for ( let i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > maxA ) { maxC = maxB ; maxB = maxA ; maxA = arr [ i ] ; }
else if ( arr [ i ] > maxB ) { maxC = maxB ; maxB = arr [ i ] ; }
else if ( arr [ i ] > maxC ) maxC = arr [ i ] ; } return ( maxA + maxB + maxC ) ; }
let arr = [ 1 , 0 , 8 , 6 , 4 , 2 ] ; let n = arr . length ; document . write ( maxTripletSum ( arr , n ) ) ;
function search ( arr , n , x ) { let i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == x ) return i ; return - 1 ; }
let arr = [ 2 , 3 , 4 , 10 , 40 ] ; let x = 10 ; let n = arr . length ;
let result = search ( arr , n , x ) ; ( result == - 1 ) ? document . write ( " " ) : document . write ( " " + result ) ;
function search ( arr , search_Element ) { let left = 0 ; let length = arr . length ; let right = length - 1 ; let position = - 1 ;
for ( left = 0 ; left <= right ; ) {
if ( arr [ left ] == search_Element ) { position = left ; document . write ( " " + ( position + 1 ) + " " + ( left + 1 ) + " " ) ; break ; }
if ( arr [ right ] == search_Element ) { position = right ; document . write ( " " + ( position + 1 ) + " " + ( length - right ) + " " ) ; break ; } left ++ ; right -- ; }
if ( position == - 1 ) document . write ( " " + left + " " ) ; }
let arr = [ 1 , 2 , 3 , 4 , 5 ] ; let search_element = 5 ;
search ( arr , search_element ) ;
function sort ( arr ) { var n = arr . length ;
var output = Array . from ( { length : n } , ( _ , i ) => 0 ) ;
var count = Array . from ( { length : 256 } , ( _ , i ) => 0 ) ;
for ( var i = 0 ; i < n ; ++ i ) ++ count [ arr [ i ] . charCodeAt ( 0 ) ] ;
for ( var i = 1 ; i <= 255 ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( var i = n - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] . charCodeAt ( 0 ) ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] . charCodeAt ( 0 ) ] ; }
for ( var i = 0 ; i < n ; ++ i ) arr [ i ] = output [ i ] ; return arr ; }
var arr = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ] ; arr = sort ( arr ) ; document . write ( " " ) ; for ( var i = 0 ; i < arr . length ; ++ i ) document . write ( arr [ i ] ) ; cript
function countSort ( arr ) { var max = Math . max . apply ( Math , arr ) ; var min = Math . min . apply ( Math , arr ) ; var range = max - min + 1 ; var count = Array . from ( { length : range } , ( _ , i ) => 0 ) ; var output = Array . from ( { length : arr . length } , ( _ , i ) => 0 ) ; for ( i = 0 ; i < arr . length ; i ++ ) { count [ arr [ i ] - min ] ++ ; } for ( i = 1 ; i < count . length ; i ++ ) { count [ i ] += count [ i - 1 ] ; } for ( i = arr . length - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] - min ] - 1 ] = arr [ i ] ; count [ arr [ i ] - min ] -- ; } for ( i = 0 ; i < arr . length ; i ++ ) { arr [ i ] = output [ i ] ; } }
function printArray ( arr ) { for ( i = 0 ; i < arr . length ; i ++ ) { document . write ( arr [ i ] + " " ) ; } document . write ( ' ' ) ; }
var arr = [ - 5 , - 10 , 0 , - 3 , 8 , 5 , - 1 , 10 ] ; countSort ( arr ) ; printArray ( arr ) ;
function binomialCoeff ( n , k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
var n = 5 , k = 2 ; document . write ( " " + n + " " + k + " " + binomialCoeff ( n , k ) ) ;
function binomialCoeff ( n , k ) { let C = new Array ( k + 1 ) ; C . fill ( 0 ) ;
C [ 0 ] = 1 ; for ( let i = 1 ; i <= n ; i ++ ) {
for ( let j = Math . min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
let n = 5 , k = 2 ; document . write ( " " + n + " " + k + " " + binomialCoeff ( n , k ) ) ;
function findPartiion ( arr , n ) { let sum = 0 ; let i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; let part = new Array ( parseInt ( sum / 2 + 1 , 10 ) ) ;
for ( i = 0 ; i <= parseInt ( sum / 2 , 10 ) ; i ++ ) { part [ i ] = false ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = parseInt ( sum / 2 , 10 ) ; j >= arr [ i ] ; j -- ) {
if ( part [ j - arr [ i ] ] == true j == arr [ i ] ) part [ j ] = true ; } } return part [ parseInt ( sum / 2 , 10 ) ] ; }
let arr = [ 1 , 3 , 3 , 2 , 3 , 2 ] ; let n = arr . length ;
if ( findPartiion ( arr , n ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function isSubsetSum ( set , n , sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
let set = [ 3 , 34 , 4 , 12 , 5 , 2 ] ; let sum = 9 ; let n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function isSubsetSum ( set , n , sum ) {
let subset = new Array ( sum + 1 ) ; for ( let i = 0 ; i < sum + 1 ; i ++ ) { subset [ i ] = new Array ( sum + 1 ) ; for ( let j = 0 ; j < n + 1 ; j ++ ) { subset [ i ] [ j ] = 0 ; } }
for ( let i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( let i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( let i = 1 ; i <= sum ; i ++ ) { for ( let j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
let set = [ 3 , 34 , 4 , 12 , 5 , 2 ] ; let sum = 9 ; let n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function findoptimal ( N ) {
if ( N <= 6 ) return N ;
let max = 0 ;
let b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
let curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
let N ;
for ( N = 1 ; N <= 20 ; N ++ ) document . write ( " " + N + " " + findoptimal ( N ) + " " ) ;
function findoptimal ( N ) {
if ( N <= 6 ) return N ;
let screen = new Array ( N ) ; for ( let i = 0 ; i < N ; i ++ ) { screen [ i ] = 0 ; }
let b ;
let n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = 0 ;
for ( b = n - 3 ; b >= 1 ; b -- ) {
let curr = ( n - b - 1 ) * screen [ b - 1 ] ; if ( curr > screen [ n - 1 ] ) screen [ n - 1 ] = curr ; } } return screen [ N - 1 ] ; }
let N ;
for ( N = 1 ; N <= 20 ; N ++ ) document . write ( " " + N + " " + findoptimal ( N ) + " " ) ;
function findoptimal ( N ) {
if ( N <= 6 ) return N ;
let screen = [ ] ;
let n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = Math . max ( 2 * screen [ n - 4 ] , Math . max ( 3 * screen [ n - 5 ] , 4 * screen [ n - 6 ] ) ) ; } return screen [ N - 1 ] ; }
let N ;
for ( N = 1 ; N <= 20 ; N ++ ) document . write ( " " + N + " " + findoptimal ( N ) + " " ) ;
function power ( x , y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , parseInt ( y / 2 , 10 ) ) * power ( x , parseInt ( y / 2 , 10 ) ) ; else return x * power ( x , parseInt ( y / 2 , 10 ) ) * power ( x , parseInt ( y / 2 , 10 ) ) ; }
let x = 2 ; let y = 3 ; document . write ( power ( x , y ) ) ;
function power ( x , y ) { var temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
function power ( x , y ) { var temp ; if ( y == 0 ) return 1 ; temp = power ( x , parseInt ( y / 2 ) ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
var x = 2 ; var y = - 3 ; document . write ( power ( x , y ) . toFixed ( 6 ) ) ;
function power ( x , y ) {
if ( y == 0 ) return 1 ;
if ( x == 0 ) return 0 ;
return x * power ( x , y - 1 ) ; }
var x = 2 ; var y = 3 ; document . write ( power ( x , y ) ) ;
function power ( x , y ) {
return parseInt ( Math . pow ( x , y ) ) ; }
let x = 2 ; let y = 3 ; document . write ( power ( x , y ) ) ;
function squareRoot ( n ) {
let x = n ; let y = 1 ; let e = 0.000001 ;
while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
let n = 50 ; document . write ( " " + n + " " + squareRoot ( n ) . toFixed ( 6 ) ) ;
function getAvg ( prev_avg , x , n ) { return ( prev_avg * n + x ) / ( n + 1 ) ; }
function streamAvg ( arr , n ) { let avg = 0 ; for ( let i = 0 ; i < n ; i ++ ) { avg = getAvg ( avg , arr [ i ] , i ) ; document . write ( " " + ( i + 1 ) + " " + avg . toFixed ( 6 ) + " " ) ; } return ; }
let arr = [ 10 , 20 , 30 , 40 , 50 , 60 ] ; let n = arr . length ; streamAvg ( arr , n ) ;
var sum = 0 , n = 0 ;
function getAvg ( x ) { sum += x ; n ++ ; return ( sum / n ) ; }
function streamAvg ( arr , m ) { var avg = 0 ; for ( i = 0 ; i < m ; i ++ ) { avg = getAvg ( parseInt ( arr [ i ] ) ) ; document . write ( " " + ( i + 1 ) + " " + avg . toFixed ( 1 ) + " " ) ; } return ; }
var arr = [ 10 , 20 , 30 , 40 , 50 , 60 ] ; var m = arr . length ; streamAvg ( arr , m ) ;
function binomialCoeff ( n , k ) { let res = 1 ;
if ( k > n - k ) k = n - k ;
for ( let i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
let n = 8 ; let k = 2 ; document . write ( " " + n + " " + k + " " + " " + " " + binomialCoeff ( n , k ) ) ;
function primeFactors ( n ) {
while ( n % 2 == 0 ) { document . write ( 2 + " " ) ; n = Math . floor ( n / 2 ) ; }
for ( let i = 3 ; i <= Math . floor ( Math . sqrt ( n ) ) ; i = i + 2 ) {
while ( n % i == 0 ) { document . write ( i + " " ) ; n = Math . floor ( n / i ) ; } }
if ( n > 2 ) document . write ( n + " " ) ; }
let n = 315 ; primeFactors ( n ) ;
function printCombination ( arr , n , r ) {
let data = new Array ( r ) ;
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; }
function combinationUtil ( arr , data , start , end , index , r ) {
if ( index == r ) { for ( let j = 0 ; j < r ; j ++ ) { document . write ( data [ j ] + " " ) ; } document . write ( " " ) }
for ( let i = start ; i <= end && end - i + 1 >= r - index ; i ++ ) { data [ index ] = arr [ i ] ; combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; } }
let arr = [ 1 , 2 , 3 , 4 , 5 ] ; let r = 3 ; let n = arr . length ; printCombination ( arr , n , r ) ;
function printCombination ( arr , n , r ) {
let data = new Array ( r ) ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
function combinationUtil ( arr , n , r , index , data , i ) {
if ( index == r ) { for ( let j = 0 ; j < r ; j ++ ) { document . write ( data [ j ] + " " ) ; } document . write ( " " ) ; return ; }
if ( i >= n ) { return ; }
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
let arr = [ 1 , 2 , 3 , 4 , 5 ] ; let r = 3 ; let n = arr . length ; printCombination ( arr , n , r ) ;
function findgroups ( arr , n ) {
let c = [ 0 , 0 , 0 ] ; let i ;
let res = 0 ;
for ( i = 0 ; i < n ; i ++ ) c [ arr [ i ] % 3 ] ++ ;
res += ( ( c [ 0 ] * ( c [ 0 ] - 1 ) ) >> 1 ) ;
res += c [ 1 ] * c [ 2 ] ;
res += ( c [ 0 ] * ( c [ 0 ] - 1 ) * Math . floor ( ( c [ 0 ] - 2 ) ) / 6 ) ;
res += ( c [ 1 ] * ( c [ 1 ] - 1 ) * Math . floor ( ( c [ 1 ] - 2 ) ) / 6 ) ;
res += ( Math . floor ( c [ 2 ] * ( c [ 2 ] - 1 ) * ( c [ 2 ] - 2 ) ) / 6 ) ;
res += c [ 0 ] * c [ 1 ] * c [ 2 ] ;
return res ; }
let arr = [ 3 , 6 , 7 , 2 , 9 ] ; let n = arr . length ; document . write ( " " + findgroups ( arr , n ) ) ;
function nextPowerOf2 ( n ) { var count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
var n = 0 ; document . write ( nextPowerOf2 ( n ) ) ;
function nextPowerOf2 ( n ) { p = 1 ; if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( p < n ) p <<= 1 ; return p ; }
n = 5 ; document . write ( nextPowerOf2 ( n ) ) ;
function nextPowerOf2 ( n ) { n -= 1 n |= n >> 1 n |= n >> 2 n |= n >> 4 n |= n >> 8 n |= n >> 16 n += 1 return n }
n = 5 ; document . write ( nextPowerOf2 ( n ) ) ;
function segregate0and1 ( arr , n ) {
let count = 0 ; for ( let i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 0 ) count ++ ; }
for ( let i = 0 ; i < count ; i ++ ) arr [ i ] = 0 ;
for ( let i = count ; i < n ; i ++ ) arr [ i ] = 1 ; }
function print ( arr , n ) { document . write ( " " ) ; for ( let i = 0 ; i < n ; i ++ ) document . write ( arr [ i ] + " " ) ; }
let arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] ; let n = arr . length ; segregate0and1 ( arr , n ) ; print ( arr , n ) ;
function segregate0and1 ( arr , size ) {
let left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
let arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] ; let i , arr_size = arr . length ; segregate0and1 ( arr , arr_size ) ; document . write ( " " ) ; for ( i = 0 ; i < 6 ; i ++ ) document . write ( arr [ i ] + " " ) ;
function segregate0and1 ( arr , size ) { let type0 = 0 ; let type1 = size - 1 ; while ( type0 < type1 ) { if ( arr [ type0 ] == 1 ) { arr [ type1 ] = arr [ type1 ] + arr [ type0 ] ; arr [ type0 ] = arr [ type1 ] - arr [ type0 ] ; arr [ type1 ] = arr [ type1 ] - arr [ type0 ] ; type1 -- ; } else type0 ++ ; } }
let arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] ; let i , arr_size = arr . length ; segregate0and1 ( arr , arr_size ) ; document . write ( " " ) ; for ( i = 0 ; i < arr_size ; i ++ ) document . write ( arr [ i ] + " " ) ;
function distinctAdjacentElement ( a , n ) {
let m = new Map ( ) ;
for ( let i = 0 ; i < n ; ++ i ) { m [ a [ i ] ] ++ ; if ( m . has ( a [ i ] ) ) { m . set ( a [ i ] , m . get ( a [ i ] ) + 1 ) } else { m . set ( a [ i ] , 1 ) } }
let mx = 0 ;
for ( let i = 0 ; i < n ; ++ i ) if ( mx < m . get ( a [ i ] ) ) mx = m . get ( a [ i ] ) ;
if ( mx > Math . floor ( ( n + 1 ) / 2 ) ) document . write ( " " + " " ) ; else document . write ( " " ) ; }
let a = [ 7 , 7 , 7 , 7 ] ; let n = a . length ; distinctAdjacentElement ( a , n ) ;
function maxIndexDiff ( arr , n ) { let maxDiff = - 1 ; let i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
let arr = [ 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ] ; let n = arr . length ; let maxDiff = maxIndexDiff ( arr , n ) ; document . write ( maxDiff ) ;
let v = [ 34 , 8 , 10 , 3 , 2 , 80 , 30 , 33 , 1 ] ; let n = v . length ; let maxFromEnd = new Array ( n + 1 ) ; for ( let i = 0 ; i < maxFromEnd . length ; i ++ ) maxFromEnd [ i ] = Number . MIN_VALUE ;
for ( let i = v . length - 1 ; i >= 0 ; i -- ) { maxFromEnd [ i ] = Math . max ( maxFromEnd [ i + 1 ] , v [ i ] ) ; } let result = 0 ; for ( let i = 0 ; i < v . length ; i ++ ) { let low = i + 1 , high = v . length - 1 , ans = i ; while ( low <= high ) { let mid = parseInt ( ( low + high ) / 2 , 10 ) ; if ( v [ i ] <= maxFromEnd [ mid ] ) {
ans = Math . max ( ans , mid ) ; low = mid + 1 ; } else { high = mid - 1 ; } }
result = Math . max ( result , ans - i ) ; } document . write ( result ) ;
function printRepeating ( arr , size ) {
var s = new Set ( arr ) ;
[ ... s ] . sort ( ( a , b ) => a - b ) . forEach ( x => { document . write ( x + " " ) } ) ; }
var arr = [ 1 , 3 , 2 , 2 , 1 ] ; var n = arr . length ; printRepeating ( arr , n ) ;
function minSwapsToSort ( arr , n ) {
let arrPos = [ ] ; for ( let i = 0 ; i < n ; i ++ ) { arrPos . push ( [ arr [ i ] , i ] ) ; }
arrPos . sort ( function ( a , b ) { return a [ 0 ] - b [ 0 ] ; } ) ;
let vis = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { vis [ i ] = false ; }
let ans = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( vis [ i ] arrPos [ i ] [ 1 ] == i ) continue ;
let cycle_size = 0 ; let j = i ; while ( ! vis [ j ] ) { vis [ j ] = true ;
j = arrPos [ j ] [ 1 ] ; cycle_size ++ ; }
ans += ( cycle_size - 1 ) ; }
return ans ; }
function minSwapToMakeArraySame ( a , b , n ) {
let mp = new Map ( ) ; for ( let i = 0 ; i < n ; i ++ ) { mp . set ( b [ i ] , i ) ; }
for ( let i = 0 ; i < n ; i ++ ) b [ i ] = mp . get ( a [ i ] ) ;
return minSwapsToSort ( b , n ) ; }
let a = [ 3 , 6 , 4 , 8 ] ; let b = [ 4 , 6 , 8 , 3 ] ; let n = a . length ; document . write ( minSwapToMakeArraySame ( a , b , n ) ) ;
function missingK ( a , k , n ) { let difference = 0 , ans = 0 , count = k ; let flag = false ;
for ( let i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = true ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return - 1 ; }
let a = [ 1 , 5 , 11 , 19 ] ;
let k = 11 ; let n = a . length ;
let missing = missingK ( a , k , n ) ; document . write ( missing ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } }
function printList ( node ) { while ( node != null ) { document . write ( node . data + " " ) ; node = node . next ; } document . write ( ) ; }
function newNode ( key ) { var temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
function insertBeg ( head , val ) { var temp = newNode ( val ) ; temp . next = head ; head = temp ; return head ; }
function rearrangeOddEven ( head ) { var odd = [ ] ; var even = [ ] ; var i = 1 ; while ( head != null ) { if ( head . data % 2 != 0 && i % 2 == 0 ) {
odd . push ( head ) ; } else if ( head . data % 2 == 0 && i % 2 != 0 ) {
even . push ( head ) ; } head = head . next ; i ++ ; } while ( odd . length > 0 && even . length > 0 ) {
var k = odd [ odd . length - 1 ] . data ; odd [ odd . length - 1 ] . data = even [ even . length - 1 ] . data ; even [ even . length - 1 ] . data = k ; odd . pop ( ) ; even . pop ( ) ; } }
var head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 1 ) ; document . write ( " " ) ; printList ( head ) ; rearrangeOddEven ( head ) ; document . write ( " " + " " ) ; printList ( head ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } }
function printList ( node ) { while ( node != null ) { document . write ( node . data + " " ) ; node = node . next ; } document . write ( " " ) ; }
function newNode ( key ) { var temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
function insertBeg ( head , val ) { var temp = newNode ( val ) ; temp . next = head ; head = temp ; return head ; }
function rearrange ( head ) {
var even ; var temp , prev_temp ; var i , j , k , l , ptr = null ;
temp = ( head ) . next ; prev_temp = head ; while ( temp != null ) {
var x = temp . next ;
if ( temp . data % 2 != 0 ) { prev_temp . next = x ; temp . next = ( head ) ; ( head ) = temp ; } else { prev_temp = temp ; }
temp = x ; }
temp = ( head ) . next ; prev_temp = ( head ) ; while ( temp != null && temp . data % 2 != 0 ) { prev_temp = temp ; temp = temp . next ; } even = temp ;
prev_temp . next = null ;
i = head ; j = even ; while ( j != null && i != null ) {
k = i . next ; l = j . next ; i . next = j ; j . next = k ;
ptr = j ;
i = k ; j = l ; } if ( i == null ) {
ptr . next = j ; }
return head ; }
var head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 1 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 10 ) ; document . write ( " " ) ; printList ( head ) ; document . write ( " " ) ; head = rearrange ( head ) ; printList ( head ) ;
function print ( mat ) {
for ( let i = 0 ; i < mat . length ; i ++ ) {
for ( let j = 0 ; j < mat [ 0 ] . length ; j ++ )
document . write ( mat [ i ] [ j ] + " " ) ; document . write ( " " ) ; } }
function performSwap ( mat , i , j ) { let N = mat . length ;
let ei = N - 1 - i ;
let ej = N - 1 - j ;
let temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ ej ] [ i ] ; mat [ ej ] [ i ] = mat [ ei ] [ ej ] ; mat [ ei ] [ ej ] = mat [ j ] [ ei ] ; mat [ j ] [ ei ] = temp ; }
function rotate ( mat , N , K ) {
K = K % 4 ;
while ( K -- > 0 ) {
for ( let i = 0 ; i < N / 2 ; i ++ ) {
for ( let j = i ; j < N - i - 1 ; j ++ ) {
if ( i != j && ( i + j ) != N - 1 ) {
performSwap ( mat , i , j ) ; } } } }
print ( mat ) ; }
let K = 5 ; let mat = [ [ 1 , 2 , 3 , 4 ] , [ 6 , 7 , 8 , 9 ] , [ 11 , 12 , 13 , 14 ] , [ 16 , 17 , 18 , 19 ] , ] ; let N = mat . length ; rotate ( mat , N , K ) ;
let MAX = 10000 ;
let prefix = Array . from ( { length : MAX + 1 } , ( _ , i ) => 0 ) ; function isPowerOfTwo ( x ) { if ( x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ) return true ; return false ; }
function computePrefix ( n , a ) {
if ( isPowerOfTwo ( a [ 0 ] ) ) prefix [ 0 ] = 1 ; for ( let i = 1 ; i < n ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] ; if ( isPowerOfTwo ( a [ i ] ) ) prefix [ i ] ++ ; } }
function query ( L , R ) { if ( L == 0 ) return prefix [ R ] ; return prefix [ R ] - prefix [ L - 1 ] ; }
let A = [ 3 , 8 , 5 , 2 , 5 , 10 ] ; let N = A . length ; computePrefix ( N , A ) ; document . write ( query ( 0 , 4 ) + " " ) ; document . write ( query ( 3 , 5 ) ) ;
function countIntgralPoints ( x1 , y1 , x2 , y2 ) { document . write ( ( y2 - y1 - 1 ) * ( x2 - x1 - 1 ) ) ; }
var x1 = 1 , y1 = 1 ; var x2 = 4 , y2 = 4 ; countIntgralPoints ( x1 , y1 , x2 , y2 ) ;
function findNextNumber ( n ) { let h = Array . from ( { length : 10 } , ( _ , i ) => 0 ) ; let i = 0 , msb = n , rem = 0 ; let next_num = - 1 , count = 0 ;
while ( msb > 9 ) { rem = msb % 10 ; h [ rem ] = 1 ; msb = Math . floor ( msb / 10 ) ; count ++ ; } h [ msb ] = 1 ; count ++ ;
for ( i = msb + 1 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; break ; } }
if ( next_num == - 1 ) { for ( i = 1 ; i < msb ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; count ++ ; break ; } } }
if ( next_num > 0 ) {
for ( i = 0 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { msb = i ; break ; } }
for ( i = 1 ; i < count ; i ++ ) { next_num = ( ( next_num * 10 ) + msb ) ; }
if ( next_num > n ) document . write ( next_num + " " ) ; else document . write ( " " ) ; } else { document . write ( " " ) ; } }
let n = 2019 ; findNextNumber ( n ) ;
function CalculateValues ( N ) { var A = 0 , B = 0 , C = 0 ;
for ( C = 0 ; C < N / 7 ; C ++ ) {
for ( B = 0 ; B < N / 5 ; B ++ ) {
var A = N - 7 * C - 5 * B ;
if ( A >= 0 && A % 3 == 0 ) { document . write ( " " + A / 3 + " " + B + " " + C ) ; return ; } } }
document . write ( - 1 ) ; }
var N = 19 ; CalculateValues ( 19 ) ;
function minimumTime ( arr , n ) {
var sum = 0 ;
var T = Math . max ( ... arr ) ;
for ( i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
document . write ( Math . max ( 2 * T , sum ) ) ; }
var arr = [ 2 , 8 , 3 ] ; var N = arr . length ;
minimumTime ( arr , N ) ;
function lexicographicallyMax ( s ) {
var n = s . length ;
for ( var i = 0 ; i < n ; i ++ ) {
var count = 0 ;
var beg = i ;
var end = i ;
if ( s [ i ] == ' ' ) count ++ ;
for ( var j = i + 1 ; j < n ; j ++ ) { if ( s [ j ] == ' ' ) count ++ ; if ( count % 2 == 0 && count != 0 ) { end = j ; break ; } }
for ( var i = beg ; i < parseInt ( ( end + 1 ) / 2 ) ; i ++ ) { let temp = s [ i ] ; s [ i ] = s [ end - i + 1 ] ; s [ end - i + 1 ] = temp ; } }
document . write ( s . join ( " " ) + " " ) ; }
var S = " " . split ( ' ' ) ; lexicographicallyMax ( S ) ;
function maxPairs ( nums , k ) {
nums . sort ( ) ;
let result = 0 ;
let start = 0 , end = nums . length - 1 ;
while ( start < end ) { if ( nums [ start ] + nums [ end ] > k )
end -- ; else if ( nums [ start ] + nums [ end ] < k )
start ++ ;
else { start ++ ; end -- ; result ++ ; } }
document . write ( result ) ; }
let arr = [ 1 , 2 , 3 , 4 ] ; let K = 5 ;
maxPairs ( arr , K ) ;
function maxPairs ( nums , k ) {
var m = new Map ( ) ;
var result = 0 ;
nums . forEach ( i => {
if ( m . has ( i ) && m . get ( i ) > 0 ) { m . set ( i , m . get ( i ) - 1 ) ; result ++ ; }
else { if ( m . has ( k - i ) ) m . set ( k - i , m . get ( k - i ) + 1 ) else m . set ( k - i , 1 ) } } ) ;
document . write ( result ) ; }
var arr = [ 1 , 2 , 3 , 4 ] ; var K = 5 ;
maxPairs ( arr , K ) ;
function removeIndicesToMakeSumEqual ( arr ) {
var N = arr . length ;
var odd = Array ( N ) . fill ( 0 ) ;
var even = Array ( N ) . fill ( 0 ) ;
even [ 0 ] = arr [ 0 ] ;
for ( i = 1 ; i < N ; i ++ ) {
odd [ i ] = odd [ i - 1 ] ;
even [ i ] = even [ i - 1 ] ;
if ( i % 2 == 0 ) {
even [ i ] += arr [ i ] ; }
else {
odd [ i ] += arr [ i ] ; } }
var find = false ;
var p = odd [ N - 1 ] ;
var q = even [ N - 1 ] - arr [ 0 ] ;
if ( p == q ) { document . write ( " " ) ; find = true ; }
for ( i = 1 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) {
p = even [ N - 1 ] - even [ i - 1 ] - arr [ i ] + odd [ i - 1 ] ;
q = odd [ N - 1 ] - odd [ i - 1 ] + even [ i - 1 ] ; } else {
q = odd [ N - 1 ] - odd [ i - 1 ] - arr [ i ] + even [ i - 1 ] ;
p = even [ N - 1 ] - even [ i - 1 ] + odd [ i - 1 ] ; }
if ( p == q ) {
find = true ;
document . write ( i + " " ) ; } }
if ( ! find ) {
document . write ( - 1 ) ; } }
var arr = [ 4 , 1 , 6 , 2 ] ; removeIndicesToMakeSumEqual ( arr ) ;
function min_element_removal ( arr , N ) {
var left = Array ( N ) . fill ( 1 ) ;
var right = Array ( N ) . fill ( 1 ) ;
for ( var i = 1 ; i < N ; i ++ ) {
for ( var j = 0 ; j < i ; j ++ ) {
if ( arr [ j ] < arr [ i ] ) {
left [ i ] = Math . max ( left [ i ] , left [ j ] + 1 ) ; } } }
for ( var i = N - 2 ; i >= 0 ; i -- ) {
for ( var j = N - 1 ; j > i ; j -- ) {
if ( arr [ i ] > arr [ j ] ) {
right [ i ] = Math . max ( right [ i ] , right [ j ] + 1 ) ; } } }
var maxLen = 0 ;
for ( var i = 1 ; i < N - 1 ; i ++ ) {
maxLen = Math . max ( maxLen , left [ i ] + right [ i ] - 1 ) ; } document . write ( ( N - maxLen ) + " " ) ; }
function makeBitonic ( arr , N ) { if ( N == 1 ) { document . write ( " " + " " ) ; return ; } if ( N == 2 ) { if ( arr [ 0 ] != arr [ 1 ] ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ; return ; } min_element_removal ( arr , N ) ; }
var arr = [ 2 , 1 , 1 , 5 , 6 , 2 , 3 , 1 ] ; var N = arr . length ; makeBitonic ( arr , N ) ;
function countSubarrays ( A , N ) {
let ans = 0 ; for ( let i = 0 ; i < N - 1 ; i ++ ) {
if ( A [ i ] != A [ i + 1 ] ) {
ans ++ ;
for ( let j = i - 1 , k = i + 2 ; j >= 0 && k < N && A [ j ] == A [ i ] && A [ k ] == A [ i + 1 ] ; j -- , k ++ ) {
ans ++ ; } } }
document . write ( ans + " " ) ; }
let A = [ 1 , 1 , 0 , 0 , 1 , 0 ] ; let N = A . length ;
countSubarrays ( A , N ) ;
let maxN = 2002 ;
let lcount = new Array ( maxN ) ;
let rcount = new Array ( maxN ) ; for ( let i = 0 ; i < maxN ; i ++ ) { lcount [ i ] = new Array ( maxN ) ; rcount [ i ] = new Array ( maxN ) ; for ( let j = 0 ; j < maxN ; j ++ ) { lcount [ i ] [ j ] = 0 ; rcount [ i ] [ j ] = 0 ; } }
function fill_counts ( a , n ) { let i , j ;
let maxA = a [ 0 ] ; for ( i = 0 ; i < n ; i ++ ) { if ( a [ i ] > maxA ) { maxA = a [ i ] ; } } for ( i = 0 ; i < n ; i ++ ) { lcount [ a [ i ] ] [ i ] = 1 ; rcount [ a [ i ] ] [ i ] = 1 ; } for ( i = 0 ; i <= maxA ; i ++ ) {
for ( j = 1 ; j < n ; j ++ ) { lcount [ i ] [ j ] = lcount [ i ] [ j - 1 ] + lcount [ i ] [ j ] ; }
for ( j = n - 2 ; j >= 0 ; j -- ) { rcount [ i ] [ j ] = rcount [ i ] [ j + 1 ] + rcount [ i ] [ j ] ; } } }
function countSubsequence ( a , n ) { let i , j ; fill_counts ( a , n ) ; let answer = 0 ; for ( i = 1 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n - 1 ; j ++ ) { answer += lcount [ a [ j ] ] [ i - 1 ] * rcount [ a [ i ] ] [ j + 1 ] ; } } return answer ; }
let a = [ 1 , 2 , 3 , 2 , 1 , 3 , 2 ] ; document . write ( countSubsequence ( a , a . length ) ) ;
function removeOuterParentheses ( S ) {
let res = " " ;
let count = 0 ;
for ( let c = 0 ; c < S . length ; c ++ ) {
if ( S . charAt ( c ) == ' ' && count ++ > 0 )
res += S . charAt ( c ) ;
if ( S . charAt ( c ) == ' ' && count -- > 1 )
res += S . charAt ( c ) ; }
return res ; }
let S = " " ; document . write ( removeOuterParentheses ( S ) ) ;
function maxiConsecutiveSubarray ( arr , N ) {
let maxi = 0 ; for ( let i = 0 ; i < N - 1 ; i ++ ) {
let cnt = 1 , j ; for ( j = i ; j < N - 1 ; j ++ ) {
if ( arr [ j + 1 ] == arr [ j ] + 1 ) { cnt ++ ; }
else { break ; } }
maxi = Math . max ( maxi , cnt ) ; i = j ; }
return maxi ; }
let N = 11 ; let arr = [ 1 , 3 , 4 , 2 , 3 , 4 , 2 , 3 , 5 , 6 , 7 ] ; document . write ( maxiConsecutiveSubarray ( arr , N ) ) ;
let N = 100005 ;
function SieveOfEratosthenes ( prime , p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( let p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( let i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
function digitSum ( number ) {
let sum = 0 ; while ( number > 0 ) {
sum += ( number % 10 ) ; number = Math . floor ( number / 10 ) ; }
return sum ; }
function longestCompositeDigitSumSubsequence ( arr , n ) { let count = 0 ; let prime = new Array ( N + 1 ) ; for ( let i = 0 ; i <= N ; i ++ ) prime [ i ] = true ; SieveOfEratosthenes ( prime , N ) ; for ( let i = 0 ; i < n ; i ++ ) {
let res = digitSum ( arr [ i ] ) ;
if ( res == 1 ) { continue ; }
if ( prime [ res ] == false ) { count ++ ; } } document . write ( count ) ; }
let arr = [ 13 , 55 , 7 , 3 , 5 , 1 , 10 , 21 , 233 , 144 , 89 ] ; let n = arr . length ;
longestCompositeDigitSumSubsequence ( arr , n ) ;
let sum ;
class Node { constructor ( data ) { this . left = null ; this . right = null ; this . data = data ; } }
function newnode ( data ) { let temp = new Node ( data ) ;
return temp ; }
function insert ( s , i , N , root , temp ) { if ( i == N ) return temp ;
if ( s [ i ] == ' ' ) root . left = insert ( s , i + 1 , N , root . left , temp ) ;
else root . right = insert ( s , i + 1 , N , root . right , temp ) ;
return root ; }
function SBTUtil ( root ) {
if ( root == null ) return 0 ; if ( root . left == null && root . right == null ) return root . data ;
let left = SBTUtil ( root . left ) ;
let right = SBTUtil ( root . right ) ;
if ( root . left != null && root . right != null ) {
if ( ( left % 2 == 0 && right % 2 != 0 ) || ( left % 2 != 0 && right % 2 == 0 ) ) { sum += root . data ; } }
return left + right + root . data ; }
function build_tree ( R , N , str , values ) {
let root = newnode ( R ) ; let i ;
for ( i = 0 ; i < N - 1 ; i ++ ) { let s = str [ i ] ; let x = values [ i ] ;
let temp = newnode ( x ) ;
root = insert ( s , 0 , s . length , root , temp ) ; }
return root ; }
function speciallyBalancedNodes ( R , N , str , values ) {
let root = build_tree ( R , N , str , values ) ;
sum = 0 ;
SBTUtil ( root ) ;
document . write ( sum + " " ) ; }
let N = 7 ;
let R = 12 ;
let str = [ " " , " " , " " , " " , " " , " " ] ;
let values = [ 17 , 16 , 4 , 9 , 2 , 3 ] ;
speciallyBalancedNodes ( R , N , str , values ) ;
function position ( arr , N ) {
let pos = - 1 ;
let count ;
for ( let i = 0 ; i < N ; i ++ ) {
count = 0 ; for ( let j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ 0 ] <= arr [ j ] [ 0 ] && arr [ i ] [ 1 ] >= arr [ j ] [ 1 ] ) { count ++ ; } }
if ( count == N ) { pos = i ; } }
if ( pos == - 1 ) { document . write ( pos ) ; }
else { document . write ( pos + 1 ) ; } }
let arr = [ [ 3 , 3 ] , [ 1 , 3 ] , [ 2 , 2 ] , [ 2 , 3 ] , [ 1 , 2 ] ] ; let N = arr . length ;
position ( arr , N ) ;
function ctMinEdits ( str1 , str2 ) { let N1 = str1 . length ; let N2 = str2 . length ;
let freq1 = new Array ( 256 ) . fill ( 0 ) ; for ( let i = 0 ; i < N1 ; i ++ ) { freq1 [ str1 [ i ] . charCodeAt ( ) ] ++ ; }
let freq2 = new Array ( 256 ) . fill ( 0 ) ; for ( let i = 0 ; i < N2 ; i ++ ) { freq2 [ str2 [ i ] . charCodeAt ( ) ] ++ ; }
for ( let i = 0 ; i < 256 ; i ++ ) {
if ( freq1 [ i ] > freq2 [ i ] ) { freq1 [ i ] = freq1 [ i ] - freq2 [ i ] ; freq2 [ i ] = 0 ; }
else { freq2 [ i ] = freq2 [ i ] - freq1 [ i ] ; freq1 [ i ] = 0 ; } }
let sum1 = 0 ;
let sum2 = 0 ; for ( let i = 0 ; i < 256 ; i ++ ) { sum1 += freq1 [ i ] ; sum2 += freq2 [ i ] ; } return Math . max ( sum1 , sum2 ) ; }
let str1 = " " ; let str2 = " " ; document . write ( ctMinEdits ( str1 . split ( ' ' ) , str2 . split ( ' ' ) ) ) ;
function CountPairs ( a , b , n ) {
var C = Array ( n ) ;
for ( var i = 0 ; i < n ; i ++ ) { C [ i ] = a [ i ] + b [ i ] ; }
var freqCount = new Map ( ) ; for ( var i = 0 ; i < n ; i ++ ) { if ( freqCount . has ( C [ i ] ) ) freqCount . set ( C [ i ] , freqCount . get ( C [ i ] ) + 1 ) else freqCount . set ( C [ i ] , 1 ) }
var NoOfPairs = 0 ; freqCount . forEach ( ( value , key ) => { var y = value ;
NoOfPairs = NoOfPairs + y * ( y - 1 ) / 2 ; } ) ;
document . write ( NoOfPairs ) ; }
var arr = [ 1 , 4 , 20 , 3 , 10 , 5 ] ; var brr = [ 9 , 6 , 1 , 7 , 11 , 6 ] ;
var N = arr . length ;
CountPairs ( arr , brr , N ) ;
function medianChange ( arr1 , arr2 ) { let N = arr1 . length ;
let median = [ ] ;
if ( ( N & 1 ) ) median . push ( ( arr1 [ Math . floor ( N / 2 ) ] * 1 ) ) ;
else median . push ( Math . floor ( ( arr1 [ Math . floor ( N / 2 ) ] + arr1 [ Math . floor ( ( N - 1 ) / 2 ) ] ) / 2 ) ) ; for ( let x = 0 ; x < arr2 . length ; x ++ ) {
let it = arr1 . indexOf ( arr2 [ x ] ) ;
arr1 . splice ( it , 1 ) ;
N -- ;
if ( ( N & 1 ) ) { median . push ( arr1 [ Math . floor ( N / 2 ) ] * 1 ) ; }
else { median . push ( Math . floor ( ( arr1 [ Math . floor ( N / 2 ) ] + arr1 [ Math . floor ( ( N - 1 ) / 2 ) ] ) / 2 ) ) ; } }
for ( let i = 0 ; i < median . length - 1 ; i ++ ) { document . write ( ( median [ i + 1 ] - median [ i ] ) + " " ) ; } }
let arr1 = [ 2 , 4 , 6 , 8 , 10 ] ; let arr2 = [ 4 , 6 ] ;
medianChange ( arr1 , arr2 )
let nfa = 1 ;
let flag = 0 ;
function state1 ( c ) {
if ( c == ' ' ) nfa = 2 ; else if ( c == ' ' c == ' ' ) nfa = 1 ; else flag = 1 ; }
function state2 ( c ) {
if ( c == ' ' ) nfa = 3 ; else if ( c == ' ' c == ' ' ) nfa = 2 ; else flag = 1 ; }
function state3 ( c ) {
if ( c == ' ' ) nfa = 1 ; else if ( c == ' ' c == ' ' ) nfa = 3 ; else flag = 1 ; }
function state4 ( c ) {
if ( c == ' ' ) nfa = 5 ; else if ( c == ' ' c == ' ' ) nfa = 4 ; else flag = 1 ; }
function state5 ( c ) {
if ( c == ' ' ) nfa = 6 ; else if ( c == ' ' c == ' ' ) nfa = 5 ; else flag = 1 ; }
function state6 ( c ) {
if ( c == ' ' ) nfa = 4 ; else if ( c == ' ' c == ' ' ) nfa = 6 ; else flag = 1 ; }
function state7 ( c ) {
if ( c == ' ' ) nfa = 8 ; else if ( c == ' ' c == ' ' ) nfa = 7 ; else flag = 1 ; }
function state8 ( c ) {
if ( c == ' ' ) nfa = 9 ; else if ( c == ' ' c == ' ' ) nfa = 8 ; else flag = 1 ; }
function state9 ( c ) {
if ( c == ' ' ) nfa = 7 ; else if ( c == ' ' c == ' ' ) nfa = 9 ; else flag = 1 ; }
function checkA ( s , x ) { for ( let i = 0 ; i < x ; i ++ ) { if ( nfa == 1 ) state1 ( s [ i ] ) ; else if ( nfa == 2 ) state2 ( s [ i ] ) ; else if ( nfa == 3 ) state3 ( s [ i ] ) ; } if ( nfa == 1 ) { return true ; } else { nfa = 4 ; } }
function checkB ( s , x ) { for ( let i = 0 ; i < x ; i ++ ) { if ( nfa == 4 ) state4 ( s [ i ] ) ; else if ( nfa == 5 ) state5 ( s [ i ] ) ; else if ( nfa == 6 ) state6 ( s [ i ] ) ; } if ( nfa == 4 ) { return true ; } else { nfa = 7 ; } }
function checkC ( s , x ) { for ( let i = 0 ; i < x ; i ++ ) { if ( nfa == 7 ) state7 ( s [ i ] ) ; else if ( nfa == 8 ) state8 ( s [ i ] ) ; else if ( nfa == 9 ) state9 ( s [ i ] ) ; } if ( nfa == 7 ) { return true ; } }
let s = " " ; let x = 5 ;
if ( checkA ( s , x ) || checkB ( s , x ) || checkC ( s , x ) ) { document . write ( " " ) ; } else { if ( flag == 0 ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
function getPositionCount ( a , n ) {
var count = 1 ;
var min = a [ 0 ] ;
for ( var i = 1 ; i < n ; i ++ ) {
if ( a [ i ] <= min ) {
min = a [ i ] ;
count ++ ; } } return count ; }
var a = [ 5 , 4 , 6 , 1 , 3 , 1 ] ; var n = a . length ; document . write ( getPositionCount ( a , n ) ) ;
function maxSum ( arr , n , k ) {
if ( n < k ) { return - 1 ; }
var res = 0 ; for ( i = 0 ; i < k ; i ++ ) res += arr [ i ] ;
var curr_sum = res ; for ( i = k ; i < n ; i ++ ) { curr_sum += arr [ i ] - arr [ i - k ] ; res = Math . max ( res , curr_sum ) ; } return res ; }
function solve ( arr , n , k ) { var max_len = 0 , l = 0 , r = n , m ;
while ( l <= r ) { m = parseInt ( ( l + r ) / 2 ) ;
if ( maxSum ( arr , n , m ) > k ) r = m - 1 ; else { l = m + 1 ;
max_len = m ; } } return max_len ; }
var arr = [ 1 , 2 , 3 , 4 , 5 ] ; var n = arr . length ; var k = 10 ; document . write ( solve ( arr , n , k ) ) ;
var MAX = 100001 var ROW = 10 var COl = 3 var indices = Array . from ( Array ( MAX ) , ( ) => new Array ( ) ) ;
var test = [ [ 2 , 3 , 6 ] , [ 2 , 4 , 4 ] , [ 2 , 6 , 3 ] , [ 3 , 2 , 6 ] , [ 3 , 3 , 3 ] , [ 3 , 6 , 2 ] , [ 4 , 2 , 4 ] , [ 4 , 4 , 2 ] , [ 6 , 2 , 3 ] , [ 6 , 3 , 2 ] ] ;
function find_triplet ( array , n ) { var answer = 0 ;
for ( var i = 0 ; i < n ; i ++ ) { indices [ array [ i ] ] . push ( i ) ; } for ( var i = 0 ; i < n ; i ++ ) { var y = array [ i ] ; for ( var j = 0 ; j < ROW ; j ++ ) { var s = test [ j ] [ 1 ] * y ;
if ( s % test [ j ] [ 0 ] != 0 ) continue ; if ( s % test [ j ] [ 2 ] != 0 ) continue ; var x = s / test [ j ] [ 0 ] ; var z = s / test [ j ] [ 2 ] ; if ( x > MAX z > MAX ) continue ; var l = 0 ; var r = indices [ x ] . length - 1 ; var first = - 1 ;
while ( l <= r ) { var m = ( l + r ) / 2 ; if ( indices [ x ] [ m ] < i ) { first = m ; l = m + 1 ; } else { r = m - 1 ; } } l = 0 ; r = indices [ z ] . length - 1 ; var third = - 1 ;
while ( l <= r ) { var m = ( l + r ) / 2 ; if ( indices [ z ] [ m ] > i ) { third = m ; r = m - 1 ; } else { l = m + 1 ; } } if ( first != - 1 && third != - 1 ) {
answer += ( first + 1 ) * ( indices [ z ] . length - third ) ; } } } return answer ; }
var array = [ 2 , 4 , 5 , 6 , 7 ] ; var n = array . length ; document . write ( find_triplet ( array , n ) ) ;
function distinct ( arr , n ) { let count = 0 ;
if ( n == 1 ) return 1 ; for ( let i = 0 ; i < n - 1 ; i ++ ) {
if ( i == 0 ) { if ( arr [ i ] != arr [ i + 1 ] ) count += 1 ; }
else { if ( arr [ i ] != arr [ i + 1 ] arr [ i ] != arr [ i - 1 ] ) count += 1 ; } }
if ( arr [ n - 1 ] != arr [ n - 2 ] ) count += 1 ; return count ; }
let arr = [ 0 , 0 , 0 , 0 , 0 , 1 , 0 ] ; let n = arr . length ; document . write ( distinct ( arr , n ) ) ;
function isSorted ( arr , N ) {
for ( let i = 1 ; i < N ; i ++ ) { if ( arr [ i ] [ 0 ] > arr [ i - 1 ] [ 0 ] ) { return false ; } }
return true ; }
function isPossibleToSort ( arr , N ) {
let group = arr [ 0 ] [ 1 ] ;
for ( let i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] [ 1 ] != group ) { return " " ; } }
if ( isSorted ( arr , N ) ) { return " " ; } else { return " " ; } }
let arr = [ [ 340000 , 2 ] , [ 15000 , 2 ] , [ 34000 , 2 ] , [ 10000 , 2 ] ] ; let N = arr . length ; document . write ( isPossibleToSort ( arr , N ) ) ;
class Node { constructor ( data ) { this . data = data ; this . left = null ; this . right = null ; } } var root = null ; function AlphaScore ( ) { root = null ; } var sum = 0 , total_sum = 0 ; var mod = 1000000007 ;
function getAlphaScore ( node ) {
if ( node . left != null ) getAlphaScore ( node . left ) ;
sum = ( sum + node . data ) % mod ;
total_sum = ( total_sum + sum ) % mod ;
if ( node . right != null ) getAlphaScore ( node . right ) ;
return total_sum ; }
function constructBST ( arr , start , end , root ) { if ( start > end ) return null ; var mid = parseInt ( ( start + end ) / 2 ) ;
if ( root == null ) root = new Node ( arr [ mid ] ) ;
root . left = constructBST ( arr , start , mid - 1 , root . left ) ;
root . right = constructBST ( arr , mid + 1 , end , root . right ) ;
return root ; }
var arr = [ 10 , 11 , 12 ] ; var length = arr . length ;
arr . sort ( ) ; var root = null ;
root = constructBST ( arr , 0 , length - 1 , root ) ; document . write ( getAlphaScore ( root ) ) ;
function sortByFreq ( arr , n ) {
var maxE = - 1 ;
for ( var i = 0 ; i < n ; i ++ ) { maxE = Math . max ( maxE , arr [ i ] ) ; }
var freq = new Array ( maxE + 1 ) . fill ( 0 ) ;
for ( var i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; }
var cnt = 0 ;
for ( var i = 0 ; i <= maxE ; i ++ ) {
if ( freq [ i ] > 0 ) { var value = 100000 - i ; arr [ cnt ] = 100000 * freq [ i ] + value ; cnt ++ ; } }
return cnt ; }
function printSortedArray ( arr , cnt ) {
for ( var i = 0 ; i < cnt ; i ++ ) {
var frequency = parseInt ( arr [ i ] / 100000 ) ;
var value = 100000 - ( arr [ i ] % 100000 ) ;
for ( var j = 0 ; j < frequency ; j ++ ) { document . write ( value + " " ) ; } } }
var arr = [ 4 , 4 , 5 , 6 , 4 , 2 , 2 , 8 , 5 ] ;
var n = arr . length ;
var cnt = sortByFreq ( arr , n ) ;
arr . sort ( ( a , b ) => b - a ) ;
printSortedArray ( arr , cnt ) ;
function checkRectangles ( arr , n ) { let ans = true ;
arr . sort ( ) ;
var area = arr [ 0 ] * arr [ 4 * n - 1 ] ;
for ( let i = 0 ; i < 2 * n ; i = i + 2 ) { if ( arr [ i ] != arr [ i + 1 ] arr [ 4 * n - i - 1 ] != arr [ 4 * n - i - 2 ] arr [ i ] * arr [ 4 * n - i - 1 ] != area ) {
ans = false ; break ; } }
if ( ans ) return true ; return false ; }
var arr = [ 1 , 8 , 2 , 1 , 2 , 4 , 4 , 8 ] ; var n = 2 ; if ( checkRectangles ( arr , n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function cntElements ( arr , n ) {
let copy_arr = new Array ( n ) ;
for ( let i = 0 ; i < n ; i ++ ) copy_arr [ i ] = arr [ i ] ;
let count = 0 ;
arr . sort ( ( a , b ) => a - b ) ; for ( let i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != copy_arr [ i ] ) { count ++ ; } } return count ; }
let arr = [ 1 , 2 , 6 , 2 , 4 , 5 ] ; let n = arr . length ; document . write ( cntElements ( arr , n ) ) ;
function findPairs ( arr , n , k , d ) {
if ( n < 2 * k ) { document . write ( - 1 ) ; return ; }
let pairs = [ ] ;
arr . sort ( ( a , b ) => a - b ) ;
for ( let i = 0 ; i < k ; i ++ ) {
if ( arr [ n - k + i ] - arr [ i ] >= d ) {
let p = [ arr [ i ] , arr [ n - k + i ] ] ; pairs . push ( p ) ; } }
if ( pairs . length < k ) { document . write ( - 1 ) ; return ; }
for ( let v of pairs ) { document . write ( " " + v [ 0 ] + " " + v [ 1 ] + " " + " " ) ; } }
let arr = [ 4 , 6 , 10 , 23 , 14 , 7 , 2 , 20 , 9 ] ; let n = arr . length ; let k = 4 , d = 3 ; findPairs ( arr , n , k , d ) ;
function pairs_count ( arr , n , sum ) {
let ans = 0 ;
arr . sort ( ) ;
let i = 0 , j = n - 1 ; while ( i < j ) {
if ( arr [ i ] + arr [ j ] < sum ) i ++ ;
else if ( arr [ i ] + arr [ j ] > sum ) j -- ;
else {
let x = arr [ i ] , xx = i ; while ( i < j && arr [ i ] == x ) i ++ ;
let y = arr [ j ] , yy = j ; while ( j >= i && arr [ j ] == y ) j -- ;
if ( x == y ) { let temp = i - xx + yy - j - 1 ; ans += ( temp * ( temp + 1 ) ) / 2 ; } else ans += ( i - xx ) * ( yy - j ) ; } }
return ans ; }
let arr = [ 1 , 5 , 7 , 5 , - 1 ] ; let n = arr . length ; let sum = 6 ; document . write ( pairs_count ( arr , n , sum ) ) ;
function check ( str ) { var min = Number . MAX_VALUE ; var max = Number . MIN_VALUE ; var sum = 0 ;
for ( i = 0 ; i < str . length ; i ++ ) {
var ascii = parseInt ( str . charCodeAt ( i ) ) ;
if ( ascii < 96 ascii > 122 ) return false ;
sum += ascii ;
if ( min > ascii ) min = ascii ;
if ( max < ascii ) max = ascii ; }
min -= 1 ;
var eSum = parseInt ( ( max * ( max + 1 ) ) / 2 ) - ( ( min * ( min + 1 ) ) / 2 ) ;
return sum == eSum ; }
var str = " " ; if ( check ( str ) ) document . write ( " " ) ; else document . write ( " " ) ;
var str1 = " " ; if ( check ( str1 ) ) document . write ( " " ) ; else document . write ( " " ) ;
function findKth ( arr , n , k ) { var missing = new Set ( ) ; var count = 0 ;
for ( var i = 0 ; i < n ; i ++ ) missing . add ( arr [ i ] ) ;
var maxm = arr . reduce ( ( a , b ) => Math . max ( a , b ) ) ; var minm = arr . reduce ( ( a , b ) => Math . min ( a , b ) ) ;
for ( var i = minm + 1 ; i < maxm ; i ++ ) {
if ( ! missing . has ( i ) ) count ++ ;
if ( count == k ) return i ; }
return - 1 ; }
var arr = [ 2 , 10 , 9 , 4 ] ; var n = arr . length ; var k = 5 ; document . write ( findKth ( arr , n , k ) ) ;
class Node { constructor ( ) { this . data = 0 ; this . next = null ; } } var start = null ;
function sortList ( head ) { var startVal = 1 ; while ( head != null ) { head . data = startVal ; startVal ++ ; head = head . next ; } }
function push ( head_ref , new_data ) {
var new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = head_ref ;
head_ref = new_node ; start = head_ref ; }
function printList ( node ) { while ( node != null ) { document . write ( node . data + " " ) ; node = node . next ; } }
start = null ;
push ( start , 2 ) ; push ( start , 1 ) ; push ( start , 6 ) ; push ( start , 4 ) ; push ( start , 5 ) ; push ( start , 3 ) ; sortList ( start ) ; printList ( start ) ;
function minSum ( arr , n ) {
let evenArr = [ ] ; let oddArr = [ ] ;
arr . sort ( function ( a , b ) { return a - b ; } ) ;
for ( let i = 0 ; i < n ; i ++ ) { if ( i < Math . floor ( n / 2 ) ) { oddArr . push ( arr [ i ] ) ; } else { evenArr . push ( arr [ i ] ) ; } }
evenArr . sort ( function ( a , b ) { return b - a ; } ) ;
let i = 0 , sum = 0 ; for ( let j = 0 ; j < evenArr . length ; j ++ ) { arr [ i ++ ] = evenArr [ j ] ; arr [ i ++ ] = oddArr [ j ] ; sum += evenArr [ j ] * oddArr [ j ] ; } return sum ; }
let arr = [ 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 ] ; let n = arr . length ; document . write ( " " + minSum ( arr , n ) + " " ) ; document . write ( " " ) ; for ( let i = 0 ; i < n ; i ++ ) { document . write ( arr [ i ] + " " ) ; }
funs minTime ( string word ) { int ans = 0 ;
let curr = 0 ; for ( let i = 0 ; i < word . Length ; i ++ ) {
int k = word [ i ] . charAt ( 0 ) - ' ' . charAt ( 0 ) ;
let a = Math . abs ( curr - k ) ;
let b = 26 - Math . abs ( curr - k ) ;
ans += Math . min ( a , b ) ;
ans ++ ; curr = word [ i ] . charAt ( 0 ) - ' ' . charAt ( 0 ) ; }
document . write ( ans ) ; }
let str = " " ;
minTime ( str ) ;
function reduceToOne ( N ) {
let cnt = 0 ; while ( N != 1 ) {
if ( N == 2 || ( N % 2 == 1 ) ) {
N = N - 1 ;
cnt ++ ; }
else if ( N % 2 == 0 ) {
N = Math . floor ( N / Math . floor ( N / 2 ) ) ;
cnt ++ ; } }
return cnt ; }
let N = 35 ; document . write ( reduceToOne ( N ) ) ;
function maxDiamonds ( A , N , K ) {
let pq = [ ] ;
for ( let i = 0 ; i < N ; i ++ ) { pq . push ( A [ i ] ) ; }
let ans = 0 ;
pq . sort ( ( a , b ) => a - b ) while ( pq . length && K -- ) { pq . sort ( ( a , b ) => a - b )
let top = pq [ pq . length - 1 ] ;
pq . pop ( ) ;
ans += top ;
top = Math . floor ( top / 2 ) ; pq . push ( top ) ; }
document . write ( ans ) ; }
let A = [ 2 , 1 , 7 , 4 , 2 ] ; let K = 3 ; let N = A . length ; maxDiamonds ( A , N , K ) ;
function MinimumCost ( A , B , N ) {
var totalCost = 0 ;
for ( i = 0 ; i < N ; i ++ ) {
var mod_A = B [ i ] % A [ i ] ; var totalCost_A = Math . min ( mod_A , A [ i ] - mod_A ) ;
var mod_B = A [ i ] % B [ i ] ; var totalCost_B = Math . min ( mod_B , B [ i ] - mod_B ) ;
totalCost += Math . min ( totalCost_A , totalCost_B ) ; }
return totalCost ; }
var A = [ 3 , 6 , 3 ] ; var B = [ 4 , 8 , 13 ] ; var N = A . length ; document . write ( MinimumCost ( A , B , N ) ) ;
function printLargestDivisible ( arr , N ) { var i , count0 = 0 , count7 = 0 ; for ( i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 0 ) count0 ++ ; else count7 ++ ; }
if ( count7 % 50 == 0 ) { while ( count7 -- ) document . write ( 7 ) ; while ( count0 -- ) document . write ( 0 ) ; }
else if ( count7 < 5 ) { if ( count0 == 0 ) document . write ( " " ) ; else document . write ( " " ) ; }
else {
count7 = count7 - count7 % 5 ; while ( count7 -- ) document . write ( 7 ) ; while ( count0 -- ) document . write ( 0 ) ; } }
var arr = [ 0 , 7 , 0 , 7 , 7 , 7 , 7 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 7 , 7 ] ;
var N = arr . length ; printLargestDivisible ( arr , N ) ;
function findMaxValByRearrArr ( arr , N ) {
arr . sort ( ) ;
let res = 0 ;
do {
let sum = 0 ;
for ( let i = 0 ; i < N ; i ++ ) {
sum += __gcd ( i + 1 , arr [ i ] ) ; }
res = Math . max ( res , sum ) ; } while ( next_permutation ( arr ) ) ; return res ; } function __gcd ( a , b ) { return b == 0 ? a : __gcd ( b , a % b ) ; } function next_permutation ( p ) { for ( let a = p . length - 2 ; a >= 0 ; -- a ) if ( p [ a ] < p [ a + 1 ] ) for ( let b = p . length - 1 ; ; -- b ) if ( p [ b ] > p [ a ] ) { let t = p [ a ] ; p [ a ] = p [ b ] ; p [ b ] = t ; for ( ++ a , b = p . length - 1 ; a < b ; ++ a , -- b ) { t = p [ a ] ; p [ a ] = p [ b ] ; p [ b ] = t ; } return true ; } return false ; }
let arr = [ 3 , 2 , 1 ] ; let N = arr . length ; document . write ( findMaxValByRearrArr ( arr , N ) ) ;
function min_elements ( arr , N ) {
var mp = new Map ( ) ;
for ( var i = 0 ; i < N ; i ++ ) {
if ( mp . has ( arr [ i ] ) ) { mp . set ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . set ( arr [ i ] , 1 ) ; } }
var cntMinRem = 0 ;
mp . forEach ( ( value , key ) => {
var i = key ;
if ( mp . get ( i ) < i ) {
cntMinRem += mp . get ( i ) ; }
else if ( mp . get ( i ) > i ) {
cntMinRem += ( mp . get ( i ) - i ) ; } } ) ; return cntMinRem ; }
var arr = [ 2 , 4 , 1 , 4 , 2 ] ; var N = arr . length ; document . write ( min_elements ( arr , N ) ) ;
function CheckAllarrayEqual ( arr , N ) {
if ( N == 1 ) { return true ; }
let totalSum = arr [ 0 ] ;
let secMax = Number . MIN_VALUE ;
let Max = arr [ 0 ] ;
for ( let i = 1 ; i < N ; i ++ ) { if ( arr [ i ] >= Max ) {
secMax = Max ;
Max = arr [ i ] ; } else if ( arr [ i ] > secMax ) {
secMax = arr [ i ] ; }
totalSum += arr [ i ] ; }
if ( ( secMax * ( N - 1 ) ) > totalSum ) { return false ; }
if ( totalSum % ( N - 1 ) != 0 ) { return false ; } return true ; }
let arr = [ 6 , 2 , 2 , 2 ] ; let N = arr . length ; if ( CheckAllarrayEqual ( arr , N ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function Remove_one_element ( arr , n ) {
let post_odd = 0 , post_even = 0 ;
let curr_odd = 0 , curr_even = 0 ;
let res = 0 ;
for ( let i = n - 1 ; i >= 0 ; i -- ) {
if ( i % 2 != 0 ) post_odd ^= arr [ i ] ;
else post_even ^= arr [ i ] ; }
for ( let i = 0 ; i < n ; i ++ ) {
if ( i % 2 != 0 ) post_odd ^= arr [ i ] ;
else post_even ^= arr [ i ] ;
let X = curr_odd ^ post_even ;
let Y = curr_even ^ post_odd ;
if ( X == Y ) res ++ ;
if ( i % 2 != 0 ) curr_odd ^= arr [ i ] ;
else curr_even ^= arr [ i ] ; }
document . write ( res ) ; }
let arr = [ 1 , 0 , 1 , 0 , 1 ] ;
let N = arr . length ;
Remove_one_element ( arr , N ) ;
function cntIndexesToMakeBalance ( arr , n ) {
if ( n == 1 ) { return 1 ; }
if ( n == 2 ) return 0 ;
let sumEven = 0 ;
let sumOdd = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( i % 2 == 0 ) {
sumEven += arr [ i ] ; }
else {
sumOdd += arr [ i ] ; } }
let currOdd = 0 ;
let currEven = arr [ 0 ] ;
let res = 0 ;
let newEvenSum = 0 ;
let newOddSum = 0 ;
for ( let i = 1 ; i < n - 1 ; i ++ ) {
if ( i % 2 ) {
currOdd += arr [ i ] ;
newEvenSum = currEven + sumOdd - currOdd ;
newOddSum = currOdd + sumEven - currEven - arr [ i ] ; }
else {
currEven += arr [ i ] ;
newOddSum = currOdd + sumEven - currEven ;
newEvenSum = currEven + sumOdd - currOdd - arr [ i ] ; }
if ( newEvenSum == newOddSum ) {
res ++ ; } }
if ( sumOdd == sumEven - arr [ 0 ] ) {
res ++ ; }
if ( n % 2 == 1 ) {
if ( sumOdd == sumEven - arr [ n - 1 ] ) {
res ++ ; } }
else {
if ( sumEven == sumOdd - arr [ n - 1 ] ) {
res ++ ; } } return res ; }
let arr = [ 1 , 1 , 1 ] ; let n = arr . length ; document . write ( cntIndexesToMakeBalance ( arr , n ) ) ;
function findNums ( X , Y ) {
let A , B ;
if ( X < Y ) { A = - 1 ; B = - 1 ; }
else if ( Math . abs ( X - Y ) & 1 ) { A = - 1 ; B = - 1 ; }
else if ( X == Y ) { A = 0 ; B = Y ; }
else {
A = ( X - Y ) / 2 ;
if ( ( A & Y ) == 0 ) {
B = ( A + Y ) ; }
else { A = - 1 ; B = - 1 ; } }
document . write ( A + " " + B ) ; }
let X = 17 , Y = 13 ;
findNums ( X , Y ) ;
function checkCount ( A , Q , q ) {
for ( let i = 0 ; i < q ; i ++ ) { let L = Q [ i ] [ 0 ] ; let R = Q [ i ] [ 1 ] ;
L -- ; R -- ;
if ( ( A [ L ] < A [ L + 1 ] ) != ( A [ R - 1 ] < A [ R ] ) ) { document . write ( " " + " " ) ; } else { document . write ( " " + " " ) ; } } }
let arr = [ 11 , 13 , 12 , 14 ] ; let Q = [ [ 1 , 4 ] , [ 2 , 4 ] ] ; let q = Q . length ; checkCount ( arr , Q , q ) ;
function pairProductMean ( arr , N ) {
var pairArray = [ ] ;
for ( i = 0 ; i < N ; i ++ ) { for ( j = i + 1 ; j < N ; j ++ ) { var pairProduct = arr [ i ] * arr [ j ] ;
pairArray . push ( pairProduct ) ; } }
var length = pairArray . length ;
var sum = 0 ; for ( i = 0 ; i < length ; i ++ ) sum += pairArray [ i ] ;
var mean ;
if ( length != 0 ) mean = sum / length ; else mean = 0 ;
return mean ; }
var arr = [ 1 , 2 , 4 , 8 ] ; var N = arr . length ;
document . write ( pairProductMean ( arr , N ) . toFixed ( 2 ) ) ;
function findPlayer ( str , n ) {
let move_first = 0 ;
let move_sec = 0 ;
for ( let i = 0 ; i < n - 1 ; i ++ ) {
if ( str [ i ] [ 0 ] == str [ i ] [ str [ i ] . length - 1 ] ) {
if ( str [ i ] [ 0 ] == 48 ) move_first ++ ; else move_sec ++ ; } }
if ( move_first <= move_sec ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
let str = [ " " , " " ] ; let N = str [ 0 ] . length ;
findPlayer ( str , N ) ;
function find_next ( n , k ) {
let M = n + 1 ; while ( true ) {
if ( ( M & ( 1 << k ) ) > 0 ) break ;
M ++ ; }
return M ; }
let N = 15 , K = 2 ;
document . write ( find_next ( N , K ) ) ;
function largestString ( num , k ) {
var ans = " " ; var str = num . split ( " " ) ; for ( const i of str ) {
while ( ans . length > 0 && ans [ ans . length - 1 ] < i && k > 0 ) {
ans = ans . substring ( 0 , ans . length - 1 ) ;
k -- ; }
ans += i ; }
while ( ans . length > 0 && k -- > 0 ) { ans = ans . substring ( 0 , ans . length - 1 ) ; }
return ans ; }
var str = " " ; var k = 1 ; document . write ( largestString ( str , k ) + " " ) ;
function maxLengthSubArray ( A , N ) {
let forward = Array . from ( { length : N } , ( _ , i ) => 0 ) ; let backward = Array . from ( { length : N } , ( _ , i ) => 0 ) ;
for ( let i = 0 ; i < N ; i ++ ) { if ( i == 0 A [ i ] != A [ i - 1 ] ) { forward [ i ] = 1 ; } else forward [ i ] = forward [ i - 1 ] + 1 ; }
for ( let i = N - 1 ; i >= 0 ; i -- ) { if ( i == N - 1 A [ i ] != A [ i + 1 ] ) { backward [ i ] = 1 ; } else backward [ i ] = backward [ i + 1 ] + 1 ; }
let ans = 0 ;
for ( let i = 0 ; i < N - 1 ; i ++ ) { if ( A [ i ] != A [ i + 1 ] ) ans = Math . max ( ans , Math . min ( forward [ i ] , backward [ i + 1 ] ) * 2 ) ; }
document . write ( ans ) ; }
let arr = [ 1 , 2 , 3 , 4 , 4 , 4 , 6 , 6 , 6 , 9 ] ;
let N = arr . length ;
maxLengthSubArray ( arr , N ) ;
function minNum ( n ) { if ( n < 3 ) document . write ( - 1 ) ; else document . write ( ( 210 * ( parseInt ( Math . pow ( 10 , n - 1 ) / 210 ) + 1 ) ) ) ; }
var n = 5 ; minNum ( n ) ;
function helper ( d , s ) {
let ans = [ ] ; for ( let i = 0 ; i < d ; i ++ ) { ans . push ( " " ) ; } for ( let i = d - 1 ; i >= 0 ; i -- ) {
if ( s >= 9 ) { ans [ i ] = ' ' ; s -= 9 ; }
else { let c = String . fromCharCode ( s + ' ' . charCodeAt ( 0 ) ) ; ans [ i ] = c ; s = 0 ; } } return ans . join ( " " ) ; }
function findMin ( x , Y ) {
let y = Y . toString ( ) ; let n = y . length ; let p = [ ] ; for ( let i = 0 ; i < n ; i ++ ) { p . push ( 0 ) ; }
for ( let i = 0 ; i < n ; i ++ ) { p [ i ] = y [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ; if ( i > 0 ) { p [ i ] = p [ i ] + p [ i - 1 ] ; } }
for ( let i = n - 1 , k = 0 ; ; i -- , k ++ ) {
let d = 0 ; if ( i >= 0 ) { d = y [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ; }
for ( let j = d + 1 ; j <= 9 ; j ++ ) { let r = j ;
if ( i > 0 ) { r += p [ i - 1 ] ; }
if ( x - r >= 0 && x - r <= 9 * k ) {
let suf = helper ( k , x - r ) ; let pre = " " ; if ( i > 0 ) pre = y . substring ( 0 , i ) ;
let cur = String . fromCharCode ( j + ' ' . charCodeAt ( 0 ) ) ; pre += cur ;
return pre + suf ; } } } }
let x = 18 ; let y = 99 ;
document . write ( findMin ( x , y ) ) ;
function largestNumber ( n , X , Y ) { let maxm = Math . max ( X , Y ) ;
Y = X + Y - maxm ;
X = maxm ;
let Xs = 0 ; let Ys = 0 ; while ( n > 0 ) {
if ( n % Y == 0 ) {
Xs += n ;
n = 0 ; } else {
n -= X ;
Ys += X ; } }
if ( n == 0 ) { while ( Xs -- > 0 ) document . write ( X ) ; while ( Ys -- > 0 ) document . write ( Y ) ; }
else document . write ( " " ) ; }
let n = 19 , X = 7 , Y = 5 ; largestNumber ( n , X , Y ) ;
function minChanges ( str , N ) { var res ; var count0 = 0 , count1 = 0 ;
str . split ( ' ' ) . forEach ( x => { count0 += ( x == ' ' ) ; } ) ; res = count0 ;
str . split ( ' ' ) . forEach ( x => { count0 -= ( x == ' ' ) ; count1 += ( x == ' ' ) ; res = Math . min ( res , count1 + count0 ) ; } ) ; return res ; }
var N = 9 ; var str = " " ; document . write ( minChanges ( str , N ) ) ;
function missingnumber ( n , arr ) { let mn = 10000 ; let mx = - 10000 ;
for ( let i = 0 ; i < n ; i ++ ) { if ( i > 0 && arr [ i ] == - 1 && arr [ i - 1 ] != - 1 ) { mn = Math . min ( mn , arr [ i - 1 ] ) ; mx = Math . max ( mx , arr [ i - 1 ] ) ; } if ( i < ( n - 1 ) && arr [ i ] == - 1 && arr [ i + 1 ] != - 1 ) { mn = Math . min ( mn , arr [ i + 1 ] ) ; mx = Math . max ( mx , arr [ i + 1 ] ) ; } } let res = ( mx + mn ) / 2 ; return res ; }
let n = 5 ; let arr = [ - 1 , 10 , - 1 , 12 , - 1 ] ;
let res = missingnumber ( n , arr ) ; document . write ( res ) ;
function LCSubStr ( A , B , m , n ) {
let LCSuff = Array ( m + 1 ) . fill ( Array ( n + 1 ) ) ; let result = 0 ;
for ( let i = 0 ; i <= m ; i ++ ) { for ( let j = 0 ; j <= n ; j ++ ) {
if ( i == 0 j == 0 ) LCSuff [ i ] [ j ] = 0 ;
else if ( A . charAt ( i - 1 ) == B . charAt ( j - 1 ) ) { LCSuff [ i ] [ j ] = LCSuff [ i - 1 ] [ j - 1 ] + 1 ; if ( LCSuff [ i ] [ j ] > result ) { result = LCSuff [ i ] [ j ] ; } }
else LCSuff [ i ] [ j ] = 0 ; } } result ++ ;
return result ; }
let A = " " ; let B = " " ; let M = A . length ; let N = B . length ;
document . write ( LCSubStr ( A , B , M , N ) ) ;
var maxN = 20 ; var maxSum = 50 ; var minSum = 50 ; var base = 50 ;
var dp = Array . from ( Array ( maxN ) , ( ) => Array ( maxSum + minSum ) ) ; var v = Array . from ( Array ( maxN ) , ( ) => Array ( maxSum + minSum ) ) ;
function findCnt ( arr , i , required_sum , n ) {
if ( i == n ) { if ( required_sum == 0 ) return 1 ; else return 0 ; }
if ( v [ i ] [ required_sum + base ] ) return dp [ i ] [ required_sum + base ] ;
v [ i ] [ required_sum + base ] = 1 ;
dp [ i ] [ required_sum + base ] = findCnt ( arr , i + 1 , required_sum , n ) + findCnt ( arr , i + 1 , required_sum - arr [ i ] , n ) ; return dp [ i ] [ required_sum + base ] ; }
function countSubsets ( arr , K , n ) {
var sum = 0 ;
for ( var i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
var S1 = ( sum + K ) / 2 ;
document . write ( findCnt ( arr , 0 , S1 , n ) ) ; }
var arr = [ 1 , 1 , 2 , 3 ] ; var N = arr . length ; var K = 1 ;
countSubsets ( arr , K , N ) ;
var dp = Array ( 105 ) . fill ( ) . map ( ( ) => Array ( 605 ) . fill ( 0.0 ) ) ;
function find ( N , sum ) { if ( N < 0 sum < 0 ) return 0 ; if ( dp [ N ] [ sum ] > 0 ) return dp [ N ] [ sum ] ;
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return ( 1.0 / 6 ) ; else return 0 ; } for ( var i = 1 ; i <= 6 ; i ++ ) dp [ N ] [ sum ] = dp [ N ] [ sum ] + find ( N - 1 , sum - i ) / 6 ; return dp [ N ] [ sum ] ; }
var N = 4 , a = 13 , b = 17 ; var probability = 0.0 ;
for ( sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
document . write ( probability . toFixed ( 6 ) ) ;
function count ( n ) {
var dp = new Map ( ) ;
dp . set ( 0 , 0 ) ; dp . set ( 1 , 1 ) ;
if ( ! dp . has ( n ) ) dp . set ( n , 1 + Math . min ( n % 2 + count ( parseInt ( n / 2 ) ) , n % 3 + count ( parseInt ( n / 3 ) ) ) ) ;
return dp . get ( n ) ; }
var N = 6 ;
document . write ( count ( N ) ) ;
function find_minimum_operations ( n , b , k ) {
let d = new Array ( n + 1 ) ; d . fill ( 0 ) ;
let i , operations = 0 , need ; for ( i = 0 ; i < n ; i ++ ) {
if ( i > 0 ) { d [ i ] += d [ i - 1 ] ; }
if ( b [ i ] > d [ i ] ) {
operations += b [ i ] - d [ i ] ; need = b [ i ] - d [ i ] ;
d [ i ] += need ;
if ( i + k <= n ) { d [ i + k ] -= need ; } } } document . write ( operations ) ; }
let n = 5 ; let b = [ 1 , 2 , 3 , 4 , 5 ] ; let k = 2 ;
find_minimum_operations ( n , b , k ) ;
function ways ( arr , K ) { let R = arr . length ; let C = arr [ 0 ] . length ; let preSum = new Array ( R ) ; for ( let i = 0 ; i < R ; i ++ ) { preSum [ i ] = new Array ( C ) ; for ( let j = 0 ; j < C ; j ++ ) { preSum [ i ] [ j ] = 0 ; } }
for ( let r = R - 1 ; r >= 0 ; r -- ) { for ( let c = C - 1 ; c >= 0 ; c -- ) { preSum [ r ] = arr [ r ] ; if ( r + 1 < R ) preSum [ r ] += preSum [ r + 1 ] ; if ( c + 1 < C ) preSum [ r ] += preSum [ r ] ; if ( r + 1 < R && c + 1 < C ) preSum [ r ] -= preSum [ r + 1 ] ; } }
let dp = new Array ( K + 1 ) ; for ( let i = 0 ; i < dp . length ; i ++ ) { dp [ i ] = new Array ( R ) ; for ( let j = 0 ; j < R ; j ++ ) { dp [ i ] [ j ] = new Array ( C ) ; for ( let k = 0 ; k < C ; k ++ ) { dp [ i ] [ j ] [ k ] = 0 ; } } }
for ( let k = 1 ; k <= K ; k ++ ) { for ( let r = R - 1 ; r >= 0 ; r -- ) { for ( let c = C - 1 ; c >= 0 ; c -- ) { if ( k == 1 ) { dp [ k ] [ r ] = ( preSum [ r ] > 0 ) ? 1 : 0 ; } else { dp [ k ] [ r ] = 0 ; for ( let r1 = r + 1 ; r1 < R ; r1 ++ ) {
if ( preSum [ r ] - preSum [ r1 ] > 0 ) dp [ k ] [ r ] += dp [ k - 1 ] [ r1 ] ; } for ( let c1 = c + 1 ; c1 < C ; c1 ++ ) {
if ( preSum [ r ] - preSum [ r ] [ c1 ] > 0 ) dp [ k ] [ r ] += dp [ k - 1 ] [ r ] [ c1 ] ; } } } } } return dp [ K ] [ 0 ] [ 0 ] ; }
let arr = [ [ 1 , 0 , 0 ] , [ 1 , 1 , 1 ] , [ 0 , 0 , 0 ] ] ; let k = 3 ;
document . write ( ways ( arr , k ) ) ;
let p = 1000000007 ;
function power ( x , y , p ) { let res = 1 ; x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
function nCr ( n , p , f , m ) { for ( let i = 0 ; i <= n ; i ++ ) { for ( let j = 0 ; j <= m ; j ++ ) {
if ( j > i ) { f [ i ] [ j ] = 0 ; }
else if ( j == 0 j == i ) { f [ i ] [ j ] = 1 ; } else { f [ i ] [ j ] = ( f [ i - 1 ] [ j ] + f [ i - 1 ] [ j - 1 ] ) % p ; } } } }
function ProductOfSubsets ( arr , n , m ) { let f = new Array ( n + 1 ) ; for ( var i = 0 ; i < f . length ; i ++ ) { f [ i ] = new Array ( 2 ) ; } nCr ( n , p - 1 , f , m ) ; arr . sort ( ) ;
let ans = 1 ; for ( let i = 0 ; i < n ; i ++ ) {
let x = 0 ; for ( let j = 1 ; j <= m ; j ++ ) {
if ( m % j == 0 ) {
x = ( x + ( f [ n - i - 1 ] [ m - j ] * f [ i ] [ j - 1 ] ) % ( p - 1 ) ) % ( p - 1 ) ; } } ans = ( ( ans * power ( arr [ i ] , x , p ) ) % p ) ; } document . write ( ans + " " ) ; }
let arr = [ 4 , 5 , 7 , 9 , 3 ] ; let K = 4 ; let N = arr . length ; ProductOfSubsets ( arr , N , K ) ;
function countWays ( n , m ) {
var dp = Array . from ( Array ( m + 1 ) , ( ) => Array ( n + 1 ) ) ;
for ( var i = 0 ; i <= n ; i ++ ) { dp [ 1 ] [ i ] = 1 ; }
var sum ; for ( var i = 2 ; i <= m ; i ++ ) { for ( var j = 0 ; j <= n ; j ++ ) { sum = 0 ;
for ( var k = 0 ; k <= j ; k ++ ) { sum += dp [ i - 1 ] [ k ] ; }
dp [ i ] [ j ] = sum ; } }
return dp [ m ] [ n ] ; }
var N = 2 , K = 3 ;
document . write ( countWays ( N , K ) ) ;
function countWays ( n , m ) {
let dp = new Array ( m + 1 ) ; for ( let i = 0 ; i < m + 1 ; i ++ ) { dp [ i ] = new Array ( n + 1 ) ; for ( let j = 0 ; j < n + 1 ; j ++ ) dp [ i ] [ j ] = 0 ; }
for ( let i = 0 ; i <= n ; i ++ ) { dp [ 1 ] [ i ] = 1 ; if ( i != 0 ) { dp [ 1 ] [ i ] += dp [ 1 ] [ i - 1 ] ; } }
for ( let i = 2 ; i <= m ; i ++ ) { for ( let j = 0 ; j <= n ; j ++ ) {
if ( j == 0 ) { dp [ i ] [ j ] = dp [ i - 1 ] [ j ] ; }
else { dp [ i ] [ j ] = dp [ i - 1 ] [ j ] ;
if ( i == m && j == n ) { return dp [ i ] [ j ] ; }
dp [ i ] [ j ] += dp [ i ] [ j - 1 ] ; } } } return Number . MIN_VALUE ; }
let N = 2 , K = 3 ;
document . write ( countWays ( N , K ) ) ;
function SieveOfEratosthenes ( MAX , primes ) { let prime = new Array ( MAX + 1 ) . fill ( true ) ;
for ( let p = 2 ; p * p <= MAX ; p ++ ) { if ( prime [ p ] == true ) {
for ( let i = p * p ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( let i = 2 ; i <= MAX ; i ++ ) { if ( prime [ i ] ) primes . push ( i ) ; } }
function findLongest ( A , n ) {
let mpp = new Map ( ) ; let primes = new Array ( ) ;
SieveOfEratosthenes ( A [ n - 1 ] , primes ) ; let dp = new Array ( n ) ; dp . fill ( 0 )
dp [ n - 1 ] = 1 ; mpp . set ( A [ n - 1 ] , n - 1 ) ;
for ( let i = n - 2 ; i >= 0 ; i -- ) {
let num = A [ i ] ;
dp [ i ] = 1 ; let maxi = 0 ;
for ( let it of primes ) {
let xx = num * it ;
if ( xx > A [ n - 1 ] ) break ;
else if ( mpp . get ( xx ) ) {
dp [ i ] = Math . max ( dp [ i ] , 1 + dp [ mpp . get ( xx ) ] ) ; } }
mpp . set ( A [ i ] , i ) ; } let ans = 1 ;
for ( let i = 0 ; i < n ; i ++ ) { ans = Math . max ( ans , dp [ i ] ) ; } return ans ; }
let a = [ 1 , 2 , 5 , 6 , 12 , 35 , 60 , 385 ] ; let n = a . length ; document . write ( findLongest ( a , n ) ) ;
function waysToKAdjacentSetBits ( n , k , currentIndex , adjacentSetBits , lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } let noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( ! lastBit ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
let n = 5 , k = 2 ;
let totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; document . write ( " " + totalWays ) ;
function postfix ( a , n ) { for ( let i = n - 1 ; i > 0 ; i -- ) { a [ i - 1 ] = a [ i - 1 ] + a [ i ] ; } }
function modify ( a , n ) { for ( let i = 1 ; i < n ; i ++ ) { a [ i - 1 ] = i * a [ i ] ; } }
function allCombination ( a , n ) { let sum = 0 ;
for ( let i = 1 ; i <= n ; i ++ ) { sum += i ; } document . write ( " " + sum + " " ) ;
for ( let i = 1 ; i < n ; i ++ ) {
postfix ( a , n - i + 1 ) ;
sum = 0 ; for ( let j = 1 ; j <= n - i ; j ++ ) { sum += ( j * a [ j ] ) ; } document . write ( " " + ( i + 1 ) + " " + sum + " " ) ;
modify ( a , n ) ; } }
let n = 5 ; let a = new Array ( n ) ;
for ( let i = 0 ; i < n ; i ++ ) { a [ i ] = i + 1 ; }
allCombination ( a , n ) ;
function findStep ( n ) { if ( n == 1 n == 0 ) return 1 ; else if ( n == 2 ) return 2 ; else return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) ; }
let n = 4 ; document . write ( findStep ( n ) ) ;
function isSubsetSum ( arr , n , sum ) {
if ( sum == 0 ) return true ; if ( n == 0 && sum != 0 ) return false ;
if ( arr [ n - 1 ] > sum ) return isSubsetSum ( arr , n - 1 , sum ) ;
return isSubsetSum ( arr , n - 1 , sum ) || isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) ; }
function findPartition ( arr , n ) {
let sum = 0 ; for ( let i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( sum % 2 != 0 ) return false ;
return isSubsetSum ( arr , n , Math . floor ( sum / 2 ) ) ; }
let arr = [ 3 , 1 , 5 , 9 , 12 ] ; let n = arr . length ;
if ( findPartition ( arr , n ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function findPartiion ( arr , n ) { let sum = 0 ; let i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; let part = new Array ( parseInt ( sum / 2 + 1 , 10 ) ) ;
for ( i = 0 ; i <= parseInt ( sum / 2 , 10 ) ; i ++ ) { part [ i ] = false ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = parseInt ( sum / 2 , 10 ) ; j >= arr [ i ] ; j -- ) {
if ( part [ j - arr [ i ] ] == true j == arr [ i ] ) part [ j ] = true ; } } return part [ parseInt ( sum / 2 , 10 ) ] ; }
let arr = [ 1 , 3 , 3 , 2 , 3 , 2 ] ; let n = arr . length ;
if ( findPartiion ( arr , n ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function binomialCoeff ( n , r ) { if ( r > n ) return 0 ; let m = 1000000007 ; let inv = new Array ( r + 1 ) . fill ( 0 ) ; inv [ 0 ] = 1 ; if ( r + 1 >= 2 ) inv [ 1 ] = 1 ;
for ( let i = 2 ; i <= r ; i ++ ) { inv [ i ] = m - Math . floor ( m / i ) * inv [ m % i ] % m ; } let ans = 1 ;
for ( let i = 2 ; i <= r ; i ++ ) { ans = ( ( ans % m ) * ( inv [ i ] % m ) ) % m ; }
for ( let i = n ; i >= ( n - r + 1 ) ; i -- ) { ans = ( ( ans % m ) * ( i % m ) ) % m ; } return ans ; }
let n = 5 , r = 2 ; document . write ( " " + n + " " + r + " " + binomialCoeff ( n , r ) + " " ) ;
function gcd ( a , b ) {
if ( a < b ) { let t = a ; a = b ; b = t ; } if ( a % b == 0 ) return b ;
return gcd ( b , a % b ) ; }
function printAnswer ( x , y ) {
let val = gcd ( x , y ) ;
if ( ( val & ( val - 1 ) ) == 0 ) document . write ( " " ) ; else document . write ( " " ) ; }
let x = 4 ; let y = 7 ;
printAnswer ( x , y ) ;
function getElement ( N , r , c ) {
if ( r > c ) return 0 ;
if ( r == 1 ) { return c ; }
let a = ( r + 1 ) * parseInt ( Math . pow ( 2 , ( r - 2 ) ) ) ;
let d = parseInt ( Math . pow ( 2 , ( r - 1 ) ) ) ;
c = c - r ; let element = a + d * c ; return element ; }
let N = 4 , R = 3 , C = 4 ; document . write ( getElement ( N , R , C ) ) ;
function MinValue ( N , X ) {
let len = N . length ;
let position = len + 1 ;
if ( N [ 0 ] == ' ' ) {
for ( let i = len - 1 ; i >= 1 ; i -- ) { if ( ( N [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) < X ) { position = i ; } } } else {
for ( let i = len - 1 ; i >= 0 ; i -- ) { if ( ( N [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) > X ) { position = i ; } } }
const c = String . fromCharCode ( X + ' ' . charCodeAt ( 0 ) ) ; let str = N . slice ( 0 , position ) + c + N . slice ( position ) ;
return str ; }
let N = " " ; let X = 1 ;
document . write ( MinValue ( N , X ) ) ;
function divisibleByk ( s , n , k ) {
let poweroftwo = new Array ( n ) ;
poweroftwo [ 0 ] = 1 % k ; for ( let i = 1 ; i < n ; i ++ ) {
poweroftwo [ i ] = ( poweroftwo [ i - 1 ] * ( 2 % k ) ) % k ; }
let rem = 0 ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( s [ n - i - 1 ] == ' ' ) {
rem += ( poweroftwo [ i ] ) ; rem %= k ; } }
if ( rem == 0 ) { return " " ; }
else return " " ; }
let s = " " ; let k = 9 ;
let n = s . length ;
document . write ( divisibleByk ( s , n , k ) ) ;
function maxSumbySplittingstring ( str , N ) {
var cntOne = 0 ;
for ( var i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' ' ) {
cntOne ++ ; } }
var zero = 0 ;
var one = 0 ;
var res = 0 ;
for ( var i = 0 ; i < N - 1 ; i ++ ) {
if ( str [ i ] == ' ' ) {
zero ++ ; }
else {
one ++ ; }
res = Math . max ( res , zero + cntOne - one ) ; } return res ; }
var str = " " ; var N = str . length ; document . write ( maxSumbySplittingstring ( str , N ) ) ;
function cntBalancedParenthesis ( s , N ) {
var cntPairs = 0 ;
var cntCurly = 0 ;
var cntSml = 0 ;
var cntSqr = 0 ;
for ( i = 0 ; i < N ; i ++ ) { if ( s . charAt ( i ) == ' ' ) {
cntCurly ++ ; } else if ( s . charAt ( i ) == ' ' ) {
cntSml ++ ; } else if ( s . charAt ( i ) == ' ' ) {
cntSqr ++ ; } else if ( s . charAt ( i ) == ' ' && cntCurly > 0 ) {
cntCurly -- ;
cntPairs ++ ; } else if ( s . charAt ( i ) == ' ' && cntSml > 0 ) {
cntSml -- ;
cntPairs ++ ; } else if ( s . charAt ( i ) == ' ' && cntSqr > 0 ) {
cntSqr -- ;
cntPairs ++ ; } } document . write ( cntPairs ) ; }
var s = " " ; var N = s . length ;
cntBalancedParenthesis ( s , N ) ;
function arcIntersection ( S , len ) { var stk = [ ] ;
for ( var i = 0 ; i < len ; i ++ ) {
stk . push ( S [ i ] ) ; if ( stk . length >= 2 ) {
var temp = stk [ stk . length - 1 ] ;
stk . pop ( ) ;
if ( stk [ stk . length - 1 ] == temp ) { stk . pop ( ) ; }
else { stk . push ( temp ) ; } } }
if ( stk . length == 0 ) return 1 ; return 0 ; }
function countString ( arr , N ) {
var count = 0 ;
for ( var i = 0 ; i < N ; i ++ ) {
var len = arr [ i ] . length ;
count += arcIntersection ( arr [ i ] , len ) ; }
document . write ( count + " " ) ; }
var arr = [ " " , " " , " " ] ; var N = arr . length ;
countString ( arr , N ) ;
function ConvertequivalentBase8 ( S ) {
let mp = new Map ( ) ;
mp . set ( " " , ' ' ) ; mp . set ( " " , ' ' ) ; mp . set ( " " , ' ' ) ; mp . set ( " " , ' ' ) ; mp . set ( " " , ' ' ) ; mp . set ( " " , ' ' ) ; mp . set ( " " , ' ' ) ; mp . set ( " " , ' ' ) ;
let N = S . length ; if ( N % 3 == 2 ) {
S = " " + S ; } else if ( N % 3 == 1 ) {
S = " " + S ; }
N = S . length ;
let oct = " " ;
for ( let i = 0 ; i < N ; i += 3 ) {
let temp = S . substring ( i , i + 3 ) ;
oct += mp . get ( temp ) ; } return oct ; }
function binString_div_9 ( S , N ) {
let oct = " " ; oct = ConvertequivalentBase8 ( S ) ;
let oddSum = 0 ;
let evenSum = 0 ;
let M = oct . length ;
for ( let i = 0 ; i < M ; i += 2 )
oddSum += ( oct [ i ] - ' ' ) ;
for ( let i = 1 ; i < M ; i += 2 ) {
evenSum += ( oct [ i ] - ' ' ) ; }
let Oct_9 = 11 ;
if ( Math . abs ( oddSum - evenSum ) % Oct_9 == 0 ) { return " " ; } return " " ; }
let S = " " ; let N = S . length ; document . write ( binString_div_9 ( S , N ) ) ;
function min_cost ( S ) {
let cost = 0 ;
let F = 0 ;
let B = 0 ; let count = 0 ; for ( let i in S ) if ( S [ i ] == ' ' ) count ++ ;
let n = S . length - count ;
if ( n == 1 ) return cost ;
for ( let i in S ) {
if ( S [ i ] != ' ' ) {
if ( B != 0 ) {
cost += Math . min ( n - F , F ) * B ; B = 0 ; }
F += 1 ; }
else {
B += 1 ; } }
return cost ; }
let S = " " ; document . write ( min_cost ( S . split ( ' ' ) ) ) ;
function isVowel ( ch ) { if ( ch == ' ' ch == ' ' ch == ' ' ch == ' ' ch == ' ' ) return true ; else return false ; }
function minCost ( S ) {
var cA = 0 ; var cE = 0 ; var cI = 0 ; var cO = 0 ; var cU = 0 ;
for ( var i = 0 ; i < S . length ; i ++ ) {
if ( isVowel ( S [ i ] ) ) {
cA += Math . abs ( S . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ) ; cE += Math . abs ( S . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ) ; cI += Math . abs ( S . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ) ; cO += Math . abs ( S . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ) ; cU += Math . abs ( S . charCodeAt ( i ) - ' ' . charCodeAt ( 0 ) ) ; } }
return Math . min ( Math . min ( Math . min ( Math . min ( cA , cE ) , cI ) , cO ) , cU ) ; }
var S = " " ; document . write ( minCost ( S ) ) ;
function decode_String ( str , K ) { let ans = " " ;
for ( let i = 0 ; i < str . length ; i += K )
ans += str [ i ] ;
for ( let i = str . length - ( K - 1 ) ; i < str . length ; i ++ ) ans += str [ i ] ; document . write ( ans ) ; }
let K = 3 ; let str = " " ; decode_String ( str , K ) ;
function maxVowelSubString ( str , K ) {
var N = str . length ;
var pref = Array ( N ) ;
for ( var i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' ' str [ i ] == ' ' str [ i ] == ' ' str [ i ] == ' ' str [ i ] == ' ' ) pref [ i ] = 1 ;
else pref [ i ] = 0 ;
if ( i ) pref [ i ] += pref [ i - 1 ] ; }
var maxCount = pref [ K - 1 ] ;
var res = str . substring ( 0 , K ) ;
for ( var i = K ; i < N ; i ++ ) {
var currCount = pref [ i ] - pref [ i - K ] ;
if ( currCount > maxCount ) { maxCount = currCount ; res = str . substring ( i - K + 1 , i - 1 ) ; }
else if ( currCount == maxCount ) { var temp = str . substring ( i - K + 1 , i + 1 ) ; if ( temp < res ) res = temp ; } }
return res ; }
var str = " " ; var K = 3 ; document . write ( maxVowelSubString ( str , K ) ) ;
function decodeStr ( str , len ) {
var c = Array ( len ) . fill ( " " ) ; var med , pos = 1 , k ;
if ( len % 2 == 1 ) med = parseInt ( len / 2 ) ; else med = parseInt ( len / 2 ) - 1 ;
c [ med ] = str [ 0 ] ;
if ( len % 2 == 0 ) c [ med + 1 ] = str [ 1 ] ;
if ( len & 1 ) k = 1 ; else k = 2 ; for ( var i = k ; i < len ; i += 2 ) { c [ med - pos ] = str [ i ] ;
if ( len % 2 == 1 ) c [ med + pos ] = str [ i + 1 ] ;
else c [ med + pos + 1 ] = str [ i + 1 ] ; pos ++ ; }
for ( var i = 0 ; i < len ; i ++ ) { document . write ( c [ i ] ) ; } }
var str = " " ; var len = str . length ; decodeStr ( str , len ) ;
function findCount ( s , L , R ) {
var distinct = 0 ;
var frequency = Array ( 26 ) . fill ( 0 ) ;
for ( var i = L ; i <= R ; i ++ ) {
frequency [ s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; } for ( var i = 0 ; i < 26 ; i ++ ) {
if ( frequency [ i ] > 0 ) distinct ++ ; } document . write ( distinct + " " ) ; }
var s = " " ; var queries = 3 ; var Q = [ [ 0 , 10 ] , [ 15 , 18 ] , [ 12 , 20 ] ] ; for ( var i = 0 ; i < queries ; i ++ ) findCount ( s , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) ;
function ReverseComplement ( s , n , k ) {
var rev = parseInt ( ( k + 1 ) / 2 ) ;
var complement = k - rev ;
if ( rev % 2 ) { s = s . split ( ' ' ) . reverse ( ) . join ( ' ' ) ; }
if ( complement % 2 ) { for ( var i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == ' ' ) s [ i ] = ' ' ; else s [ i ] = ' ' ; } }
return s ; }
var str = " " ; var k = 5 ; var n = str . length ;
document . write ( ReverseComplement ( str , n , k ) ) ;
function repeatingString ( s , n , k ) {
if ( n % k != 0 ) { return false ; }
var frequency = new Array ( 123 ) ;
for ( let i = 0 ; i < 123 ; i ++ ) { frequency [ i ] = 0 ; }
for ( let i = 0 ; i < n ; i ++ ) { frequency [ s [ i ] ] ++ ; } var repeat = n / k ;
for ( let i = 0 ; i < 123 ; i ++ ) { if ( frequency [ i ] % repeat != 0 ) { return false ; } } return true ; }
var s = " " ; var n = s . length ; var k = 3 ; if ( repeatingString ( s , n , k ) ) { console . log ( " " ) ; } else { console . log ( " " ) ; }
function findPhoneNumber ( n ) { let temp = n ; let sum = 0 ;
while ( temp != 0 ) { sum += temp % 10 ; temp = Math . floor ( temp / 10 ) ; }
if ( sum < 10 ) document . write ( n + " " + sum ) ;
else document . write ( n + " " + sum ) ; }
let n = 98765432 ; findPhoneNumber ( n ) ;
var maxN = 20 ; var maxM = 64 ;
function cntSplits ( s ) {
if ( s [ s . length - 1 ] == ' ' ) return 0 ;
var c_zero = 0 ;
for ( var i = 0 ; i < s . length ; i ++ ) c_zero += ( s [ i ] == ' ' ) ;
return Math . pow ( 2 , c_zero - 1 ) ; }
var s = " " ; document . write ( cntSplits ( s ) ) ;
function findNumbers ( s ) {
var n = s . length ;
var count = 1 ; var result = 0 ;
var left = 0 ; var right = 1 ; while ( right < n ) {
if ( s [ left ] == s [ right ] ) { count ++ ; }
else {
result += parseInt ( count * ( count + 1 ) / 2 ) ;
left = right ; count = 1 ; } right ++ ; }
result += parseInt ( count * ( count + 1 ) / 2 ) ; document . write ( result ) ; }
var s = " " ; findNumbers ( s ) ;
function isVowel ( ch ) { ch = ch . toUpperCase ( ) ; return ( ch == ' ' ch == ' ' ch == ' ' ch == ' ' ch == ' ' ) ; }
function duplicateVowels ( str ) { let t = str . length ;
let res = " " ;
for ( let i = 0 ; i < t ; i ++ ) { if ( isVowel ( str [ i ] ) ) res += str [ i ] ; res += str [ i ] ; } return res ; }
let str = " " ;
document . write ( " " + str + " " ) ; let res = duplicateVowels ( str ) ;
document . write ( " " + res + " " ) ;
function stringToInt ( str ) {
if ( str . length == 1 ) return ( str [ 0 ] - ' ' ) ;
var y = stringToInt ( str . substring ( 1 ) ) ;
var x = str [ 0 ] - ' ' ;
x = x * Math . pow ( 10 , str . Length - 1 ) + y ; return ( x ) ; }
var str = " " . split ( ) document . write ( stringToInt ( str ) ) ;
var MAX = 26 ;
function largestSubSeq ( arr , n ) {
var count = Array ( MAX ) . fill ( 0 ) ;
for ( var i = 0 ; i < n ; i ++ ) { var str = arr [ i ] ;
var hash = Array ( MAX ) . fill ( 0 ) ; for ( var j = 0 ; j < str . length ; j ++ ) { hash [ str [ j ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] = true ; } for ( var j = 0 ; j < MAX ; j ++ ) {
if ( hash [ j ] ) count [ j ] ++ ; } } return count . reduce ( ( a , b ) => Math . max ( a , b ) ) ; }
var arr = [ " " , " " , " " ] ; var n = arr . length ; document . write ( largestSubSeq ( arr , n ) ) ;
function isPalindrome ( str ) { let len = str . length ; for ( let i = 0 ; i < len / 2 ; i ++ ) { if ( str [ i ] != str [ len - 1 - i ] ) return false ; } return true ; }
function createStringAndCheckPalindrome ( N ) {
let sub = " " + N , res_str = " " ; let sum = 0 ;
while ( N > 0 ) { let digit = N % 10 ; sum += digit ; N = N / 10 ; }
while ( res_str . length < sum ) res_str += sub ;
if ( res_str . length > sum ) res_str = res_str . substring ( 0 , sum ) ;
if ( isPalindrome ( res_str ) ) return true ; return false ; }
let N = 10101 ; if ( createStringAndCheckPalindrome ( N ) ) document . write ( " " ) ; else document . write ( " " ) ;
function minimumLength ( s ) { var maxOcc = 0 , n = s . length ; var arr = Array ( 26 ) . fill ( 0 ) ;
for ( var i = 0 ; i < n ; i ++ ) arr [ s [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
for ( var i = 0 ; i < 26 ; i ++ ) if ( arr [ i ] > maxOcc ) maxOcc = arr [ i ] ;
return ( n - maxOcc ) ; }
var str = " " ; document . write ( minimumLength ( str ) ) ;
function removeSpecialCharacter ( s ) { for ( let i = 0 ; i < s . length ; i ++ ) {
if ( s [ i ] < ' ' s [ i ] > ' ' && s [ i ] < ' ' s [ i ] > ' ' ) {
s = s . substring ( 0 , i ) + s . substring ( i + 1 ) ; i -- ; } } document . write ( s ) ; }
let s = " " ; removeSpecialCharacter ( s ) ;
function removeSpecialCharacter ( str ) { let s = str . split ( " " ) ; let j = 0 ; for ( let i = 0 ; i < s . length ; i ++ ) {
if ( ( s [ i ] >= ' ' && s [ i ] <= ' ' ) || ( s [ i ] >= ' ' && s [ i ] <= ' ' ) ) { s [ j ] = s [ i ] ; j ++ ; } } document . write ( ( s ) . join ( " " ) . substring ( 0 , j ) ) ; }
let s = " " ; removeSpecialCharacter ( s ) ;
function findRepeatFirstN2 ( s ) {
let p = - 1 , i , j ; for ( i = 0 ; i < s . length ; i ++ ) { for ( j = i + 1 ; j < s . length ; j ++ ) { if ( s [ i ] == s [ j ] ) { p = i ; break ; } } if ( p != - 1 ) break ; } return p ; }
let str = " " ; let pos = findRepeatFirstN2 ( str ) ; if ( pos == - 1 ) document . write ( " " ) ; else document . write ( str [ pos ] ) ;
function prCharWithFreq ( s ) {
var d = new Map ( ) ; s . split ( ' ' ) . forEach ( element => { if ( d . has ( element ) ) { d . set ( element , d . get ( element ) + 1 ) ; } else d . set ( element , 1 ) ; } ) ;
s . split ( ' ' ) . forEach ( element => {
if ( d . has ( element ) && d . get ( element ) != 0 ) { document . write ( element + d . get ( element ) + " " ) ; d . set ( element , 0 ) ; } } ) ; }
var s = " " ; prCharWithFreq ( s ) ;
function possibleStrings ( n , r , b , g ) {
let fact = new Array ( n + 1 ) ; fact [ 0 ] = 1 ; for ( let i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
let left = n - ( r + g + b ) ; let sum = 0 ;
for ( let i = 0 ; i <= left ; i ++ ) { for ( let j = 0 ; j <= left - i ; j ++ ) { let k = left - ( i + j ) ;
sum = sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ; } }
return sum ; }
let n = 4 , r = 2 ; let b = 0 , g = 1 ; document . write ( possibleStrings ( n , r , b , g ) ) ;
function remAnagram ( str1 , str2 ) {
var count1 = Array . from ( { length : 26 } , ( _ , i ) => 0 ) ; var count2 = Array . from ( { length : 26 } , ( _ , i ) => 0 ) ;
for ( i = 0 ; i < str1 . length ; i ++ ) count1 [ str1 . charAt ( i ) . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
for ( i = 0 ; i < str2 . length ; i ++ ) count2 [ str2 . charAt ( i ) . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
var result = 0 ; for ( i = 0 ; i < 26 ; i ++ ) result += Math . abs ( count1 [ i ] - count2 [ i ] ) ; return result ; }
var str1 = " " , str2 = " " ; document . write ( remAnagram ( str1 , str2 ) ) ;
let CHARS = 26 ;
function isValidString ( str ) { let freq = new Array ( CHARS ) ; for ( let i = 0 ; i < CHARS ; i ++ ) { freq [ i ] = 0 ; }
for ( let i = 0 ; i < str . length ; i ++ ) { freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; }
let i , freq1 = 0 , count_freq1 = 0 ; for ( i = 0 ; i < CHARS ; i ++ ) { if ( freq [ i ] != 0 ) { freq1 = freq [ i ] ; count_freq1 = 1 ; break ; } }
let j , freq2 = 0 , count_freq2 = 0 ; for ( j = i + 1 ; j < CHARS ; j ++ ) { if ( freq [ j ] != 0 ) { if ( freq [ j ] == freq1 ) { count_freq1 ++ ; } else { count_freq2 = 1 ; freq2 = freq [ j ] ; break ; } } }
for ( let k = j + 1 ; k < CHARS ; k ++ ) { if ( freq [ k ] != 0 ) { if ( freq [ k ] == freq1 ) { count_freq1 ++ ; } if ( freq [ k ] == freq2 ) { count_freq2 ++ ;
{ return false ; } }
if ( count_freq1 > 1 && count_freq2 > 1 ) { return false ; } }
return true ; }
let str = " " ; if ( isValidString ( str ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function checkForVariation ( str ) { if ( str == null str . length == 0 ) { return true ; } let map = new Map ( ) ;
for ( let i = 0 ; i < str . length ; i ++ ) { if ( ! map . has ( str [ i ] ) ) map . set ( str [ i ] , 0 ) ; map . set ( str [ i ] , map . get ( str [ i ] ) + 1 ) ; }
let first = true , second = true ; let val1 = 0 , val2 = 0 ; let countOfVal1 = 0 , countOfVal2 = 0 ; for ( let [ key , value ] of map . entries ( ) ) { let i = value ;
if ( first ) { val1 = i ; first = false ; countOfVal1 ++ ; continue ; } if ( i == val1 ) { countOfVal1 ++ ; continue ; }
if ( second ) { val2 = i ; countOfVal2 ++ ; second = false ; continue ; } if ( i == val2 ) { countOfVal2 ++ ; continue ; } return false ; } if ( countOfVal1 > 1 && countOfVal2 > 1 ) { return false ; } else { return true ; } }
document . write ( checkForVariation ( " " ) ) ;
function countCompletePairs ( set1 , set2 , n , m ) { let result = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < m ; j ++ ) {
let concat = set1 [ i ] + set2 [ j ] ;
let frequency = new Array ( 26 ) ; for ( let i = 0 ; i < 26 ; i ++ ) { frequency [ i ] = 0 ; } for ( let k = 0 ; k < concat . length ; k ++ ) { frequency [ concat [ k ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ; }
let k ; for ( k = 0 ; k < 26 ; k ++ ) { if ( frequency [ k ] < 1 ) { break ; } } if ( k == 26 ) { result ++ ; } } } return result ; }
let set1 = [ " " , " " , " " , " " ] ; let set2 = [ " " , " " , " " ] let n = set1 . length ; let m = set2 . length ; document . write ( countCompletePairs ( set1 , set2 , n , m ) ) ;
function countCompletePairs ( set1 , set2 , n , m ) { let result = 0 ;
let con_s1 = new Array ( n ) ; let con_s2 = new Array ( m ) ;
for ( let i = 0 ; i < n ; i ++ ) {
con_s1 [ i ] = 0 ; for ( let j = 0 ; j < set1 [ i ] . length ; j ++ ) {
con_s1 [ i ] = con_s1 [ i ] | ( 1 << ( set1 [ i ] [ j ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ) ; } }
for ( let i = 0 ; i < m ; i ++ ) {
con_s2 [ i ] = 0 ; for ( let j = 0 ; j < set2 [ i ] . length ; j ++ ) {
con_s2 [ i ] = con_s2 [ i ] | ( 1 << ( set2 [ i ] [ j ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ) ) ; } }
let complete = ( 1 << 26 ) - 1 ;
for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < m ; j ++ ) {
if ( ( con_s1 [ i ] con_s2 [ j ] ) == complete ) { result ++ ; } } } return result ; }
let set1 = [ " " , " " , " " , " " ] ; let set2 = [ " " , " " , " " ] let n = set1 . length ; let m = set2 . length ; document . write ( countCompletePairs ( set1 , set2 , n , m ) ) ;
function encodeString ( str ) { let map = new Map ( ) ; let res = " " ; let i = 0 ;
let ch ; for ( let j = 0 ; j < str . length ; j ++ ) { ch = str [ j ] ;
if ( ! map . has ( ch ) ) map . set ( ch , i ++ ) ;
res += map . get ( ch ) ; } return res ; }
function findMatchedWords ( dict , pattern ) {
let len = pattern . length ;
let hash = encodeString ( pattern ) ;
for ( let word = 0 ; word < dict . length ; word ++ ) {
if ( dict [ word ] . length == len && encodeString ( dict [ word ] ) == ( hash ) ) document . write ( dict [ word ] + " " ) ; } }
let dict = [ " " , " " , " " , " " ] ; let pattern = " " ; findMatchedWords ( dict , pattern ) ;
function check ( pattern , word ) { if ( pattern . length != word . length ) return false ; let ch = new Array ( 128 ) ; for ( let i = 0 ; i < 128 ; i ++ ) { ch [ i ] = 0 ; } let Len = word . length ; for ( let i = 0 ; i < Len ; i ++ ) { if ( ch [ pattern [ i ] . charCodeAt ( 0 ) ] == 0 ) { ch [ pattern [ i ] . charCodeAt ( 0 ) ] = word [ i ] ; } else if ( ch [ pattern [ i ] . charCodeAt ( 0 ) ] != word [ i ] ) { return false ; } } return true ; }
function findMatchedWords ( dict , pattern ) {
let Len = pattern . length ;
let result = " " ; for ( let word of dict . values ( ) ) { if ( check ( pattern , word ) ) { result = word + " " + result ; } } document . write ( result ) ; }
let dict = new Set ( ) ; dict . add ( " " ) ; dict . add ( " " ) ; dict . add ( " " ) ; dict . add ( " " ) ; let pattern = " " ; findMatchedWords ( dict , pattern ) ;
function countWords ( str ) {
if ( str == null str . length == 0 ) return 0 ; let wordCount = 0 ; let isWord = false ; let endOfLine = str . length - 1 ;
let ch = str . split ( " " ) ; for ( let i = 0 ; i < ch . length ; i ++ ) {
if ( isLetter ( ch [ i ] ) && i != endOfLine ) isWord = true ;
else if ( ! isLetter ( ch [ i ] ) && isWord ) { wordCount ++ ; isWord = false ; }
else if ( isLetter ( ch [ i ] ) && i == endOfLine ) wordCount ++ ; }
return wordCount ; } function isLetter ( c ) { return c . toLowerCase ( ) != c . toUpperCase ( ) ; }
let str = " " ;
document . write ( " " + countWords ( str ) ) ;
function RevString ( s , l ) {
if ( l % 2 == 0 ) {
let j = parseInt ( l / 2 , 10 ) ;
while ( j <= l - 1 ) { let temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } }
else {
let j = parseInt ( ( l / 2 ) , 10 ) + 1 ;
while ( j <= l - 1 ) { let temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } } let S = s [ 0 ] ;
for ( let i = 1 ; i < 9 ; i ++ ) { S = S + " " + s [ i ] ; } return S ; }
let s = " " " " ; let words = [ " " , " " , " " , " " , " " , " " , " " , " " , " " ] ; document . write ( RevString ( words , 9 ) ) ;
function printPath ( res , nThNode , kThNode ) {
if ( kThNode > nThNode ) return ;
res . push ( kThNode ) ;
for ( var i = 0 ; i < res . length ; i ++ ) document . write ( res [ i ] + " " ) ; document . write ( " " ) ;
printPath ( res , nThNode , kThNode * 2 ) ;
printPath ( res , nThNode , kThNode * 2 + 1 ) ; res . pop ( ) }
function printPathToCoverAllNodeUtil ( nThNode ) {
var res = [ ] ;
printPath ( res , nThNode , 1 ) ; }
var nThNode = 7 ;
printPathToCoverAllNodeUtil ( nThNode ) ;
function getMid ( s , e ) { return s + Math . floor ( ( e - s ) / 2 ) ; }
function isArmstrong ( x ) { let n = ( x ) . toString ( ) . length ; let sum1 = 0 ; let temp = x ; while ( temp > 0 ) { let digit = temp % 10 ; sum1 += Math . pow ( digit , n ) ; temp = Math . floor ( temp / 10 ) ; } if ( sum1 == x ) return true ; return false ; }
function MaxUtil ( st , ss , se , l , r , node ) {
if ( l <= ss && r >= se ) return st [ node ] ;
if ( se < l ss > r ) return - 1 ;
let mid = getMid ( ss , se ) ; return Math . max ( MaxUtil ( st , ss , mid , l , r , 2 * node ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 1 ) ) ; }
function updateValue ( arr , st , ss , se , index , value , node ) { if ( index < ss index > se ) { document . write ( " " + " " ) ; return ; } if ( ss == se ) {
arr [ index ] = value ; if ( isArmstrong ( value ) ) st [ node ] = value ; else st [ node ] = - 1 ; } else { let mid = getMid ( ss , se ) ; if ( index >= ss && index <= mid ) updateValue ( arr , st , ss , mid , index , value , 2 * node ) ; else updateValue ( arr , st , mid + 1 , se , index , value , 2 * node + 1 ) ; st [ node ] = Math . max ( st [ 2 * node + 1 ] , st [ 2 * node + 2 ] ) ; } return ; }
function getMax ( st , n , l , r ) {
if ( l < 0 r > n - 1 l > r ) { document . write ( " " ) ; return - 1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
function constructSTUtil ( arr , ss , se , st , si ) {
if ( ss == se ) { if ( isArmstrong ( arr [ ss ] ) ) st [ si ] = arr [ ss ] ; else st [ si ] = - 1 ; return st [ si ] ; }
let mid = getMid ( ss , se ) ; st [ si ] = Math . max ( constructSTUtil ( arr , ss , mid , st , si * 2 ) , constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 1 ) ) ; return st [ si ] ; }
function constructST ( arr , n ) {
let x = ( Math . ceil ( Math . log ( n ) ) ) ;
let max_size = 2 * Math . pow ( 2 , x ) - 1 ;
let st = new Array ( max_size ) ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
let arr = [ 192 , 113 , 535 , 7 , 19 , 111 ] ; let n = arr . length ;
let st = constructST ( arr , n ) ;
document . write ( " " + " " + getMax ( st , n , 1 , 3 ) + " " ) ;
updateValue ( arr , st , 0 , n - 1 , 1 , 153 , 0 ) ;
document . write ( " " + " " + getMax ( st , n , 1 , 3 ) + " " ) ;
function maxRegions ( n ) { let num ; num = parseInt ( n * ( n + 1 ) / 2 ) + 1 ;
document . write ( num ) ; }
let n = 10 ; maxRegions ( n ) ;
function checkSolveable ( n , m ) {
if ( n == 1 m == 1 ) document . write ( " " ) ;
else if ( m == 2 && n == 2 ) document . write ( " " ) ; else document . write ( " " ) ; }
let n = 1 , m = 3 ; checkSolveable ( n , m ) ;
function GCD ( a , b ) {
if ( b == 0 ) return a ;
else return GCD ( b , a % b ) ; }
function check ( x , y ) {
if ( GCD ( x , y ) == 1 ) { document . write ( " " ) ; } else { document . write ( " " ) ; } }
let X = 2 , Y = 7 ;
check ( X , Y ) ;
let size = 1000001 ; let prime = new Array ( size , 0 ) ;
function seiveOfEratosthenes ( ) { prime [ 0 ] = 1 ; prime [ 1 ] = 0 ; for ( let i = 2 ; i * i < 1000001 ; i ++ ) {
if ( prime [ i ] == 0 ) { for ( let j = i * i ; j < 1000001 ; j += i ) {
prime [ j ] = 1 ; } } } }
function probabiltyEuler ( L , R , M ) { let arr = new Array ( size , 0 ) ; let eulerTotient = new Array ( size , 0 ) ; let count = 0 ;
for ( let i = L ; i <= R ; i ++ ) {
eulerTotient [ i - L ] = i ; arr [ i - L ] = i ; } for ( let i = 2 ; i < 1000001 ; i ++ ) {
if ( prime [ i ] == 0 ) {
for ( let j = ( L / i ) * i ; j <= R ; j += i ) { if ( j - L >= 0 ) {
eulerTotient [ j - L ] = eulerTotient [ j - L ] / i * ( i - 1 ) ; while ( arr [ j - L ] % i == 0 ) { arr [ j - L ] /= i ; } } } } }
for ( let i = L ; i <= R ; i ++ ) { if ( arr [ i - L ] > 1 ) { eulerTotient [ i - L ] = ( eulerTotient [ i - L ] / arr [ i - L ] ) * ( arr [ i - L ] - 1 ) ; } } for ( let i = L ; i <= R ; i ++ ) {
if ( ( eulerTotient [ i - L ] % M ) == 0 ) { count ++ ; } } count /= 2 ;
return1 .0 * count / ( R + 1 - L ) ; }
seiveOfEratosthenes ( ) ; let L = 1 ; let R = 7 ; let M = 3 ; document . write ( probabiltyEuler ( L , R , M ) . toFixed ( 7 ) ) ;
function findWinner ( n , k ) { let cnt = 0 ;
if ( n == 1 ) document . write ( " " ) ;
else if ( ( n & 1 ) != 0 n == 2 ) document . write ( " " ) ; else { let tmp = n ; let val = 1 ;
while ( tmp > k && tmp % 2 == 0 ) { tmp /= 2 ; val *= 2 ; }
for ( let i = 3 ; i <= Math . sqrt ( tmp ) ; i ++ ) { while ( tmp % i == 0 ) { cnt ++ ; tmp /= i ; } } if ( tmp > 1 ) cnt ++ ;
if ( val == n ) document . write ( " " ) ; else if ( n / tmp == 2 && cnt == 1 ) document . write ( " " ) ;
else document . write ( " " ) ; } }
let n = 1 , k = 1 ; findWinner ( n , k ) ;
function pen_hex ( n ) { var pn = 1 ; for ( i = 1 ; i < n ; i ++ ) {
pn = parseInt ( i * ( 3 * i - 1 ) / 2 ) ; if ( pn > n ) break ;
var seqNum = ( 1 + Math . sqrt ( 8 * pn + 1 ) ) / 4 ; if ( seqNum == parseInt ( seqNum ) ) document . write ( pn + " " ) ; } }
var N = 1000000 ; pen_hex ( N ) ;
function isPal ( a , n , m ) {
for ( let i = 0 ; i < n / 2 ; i ++ ) { for ( let j = 0 ; j < m - 1 ; j ++ ) { if ( a [ i ] [ j ] != a [ n - 1 - i ] [ m - 1 - j ] ) return false ; } } return true ; }
let n = 3 , m = 3 ; let a = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 4 ] , [ 3 , 2 , 1 ] ] ; if ( isPal ( a , n , m ) ) { document . write ( " " + " " ) ; } else { document . write ( " " + " " ) ; }
function getSum ( n ) { let sum = 0 ; while ( n != 0 ) { sum = sum + n % 10 ; n = Math . floor ( n / 10 ) ; } return sum ; }
function smallestNumber ( N ) { let i = 1 ; while ( 1 ) {
if ( getSum ( i ) == N ) { document . write ( i ) ; break ; } i ++ ; } }
let N = 10 ; smallestNumber ( N ) ;
function reversDigits ( num ) { let rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = parseInt ( num / 10 ) ; } return rev_num ; }
function isPerfectSquare ( x ) {
let sr = Math . sqrt ( x ) ;
return ( ( sr - Math . floor ( sr ) ) == 0 ) ; }
function isRare ( N ) {
let reverseN = reversDigits ( N ) ;
if ( reverseN == N ) return false ; return isPerfectSquare ( N + reverseN ) && isPerfectSquare ( N - reverseN ) ; }
let n = 65 ; if ( isRare ( n ) ) { document . write ( " " ) ; } else { document . write ( " " ) ; }
function calc_ans ( l , r ) { var power2 = [ ] , power3 = [ ] ;
var mul2 = 1 ; while ( mul2 <= r ) { power2 . push ( mul2 ) ; mul2 *= 2 ; }
var mul3 = 1 ; while ( mul3 <= r ) { power3 . push ( mul3 ) ; mul3 *= 3 ; }
var power23 = [ ] ; for ( var x = 0 ; x < power2 . length ; x ++ ) { for ( var y = 0 ; y < power3 . length ; y ++ ) { var mul = power2 [ x ] * power3 [ y ] ; if ( mul == 1 ) continue ;
if ( mul <= r ) power23 . push ( mul ) ; } }
var ans = 0 ; power23 . forEach ( x => { if ( x >= l && x <= r ) ans ++ ; } ) ;
document . write ( ans ) ; }
var l = 1 , r = 10 ; calc_ans ( l , r ) ;
function nCr ( n , r ) { if ( r > n ) return 0 ; return fact ( n ) / ( fact ( r ) * fact ( n - r ) ) ; }
function fact ( n ) { var res = 1 ; for ( var i = 2 ; i <= n ; i ++ ) res = res * i ; return res ; }
function countSubsequences ( arr , n , k ) { var countOdd = 0 ;
for ( var i = 0 ; i < n ; i ++ ) { if ( arr [ i ] & 1 ) countOdd ++ ; } var ans = nCr ( n , k ) - nCr ( countOdd , k ) ; return ans ; }
var arr = [ 2 , 4 ] ; var K = 1 ; var N = arr . length ; document . write ( countSubsequences ( arr , N , K ) ) ;
function first_digit ( x , y ) {
var length = parseInt ( Math . log ( x ) / Math . log ( y ) ) + 1 ;
var first_digit = parseInt ( x / Math . pow ( y , length - 1 ) ) ; document . write ( first_digit ) ; }
var X = 55 , Y = 3 ; first_digit ( X , Y ) ;
function checkIfCurzonNumber ( N ) { var powerTerm , productTerm ;
powerTerm = Math . pow ( 2 , N ) + 1 ;
productTerm = 2 * N + 1 ;
if ( powerTerm % productTerm == 0 ) { document . write ( " " + " " ) ; } else { document . write ( " " ) ; } }
var N = 5 ; checkIfCurzonNumber ( N ) ; N = 10 ; checkIfCurzonNumber ( N ) ;
function minCount ( n ) {
let hasharr = [ 10 , 3 , 6 , 9 , 2 , 5 , 8 , 1 , 4 , 7 ] ;
if ( n > 69 ) return hasharr [ n % 10 ] ; else {
if ( n >= hasharr [ n % 10 ] * 7 ) return ( hasharr [ n % 10 ] ) ; else return - 1 ; } }
let n = 38 ; document . write ( minCount ( n ) ) ;
function modifiedBinaryPattern ( n ) {
for ( let i = 1 ; i <= n ; i ++ ) {
for ( let j = 1 ; j <= i ; j ++ ) {
if ( j == 1 j == i ) document . write ( 1 ) ;
else document . write ( 0 ) ; }
document . write ( " " ) ; } }
let n = 7 ;
modifiedBinaryPattern ( n ) ;
function findRealAndImag ( s ) {
let l = s . length - 1 ;
let i ;
if ( s . indexOf ( ' ' ) != - 1 ) { i = s . indexOf ( ' ' ) ; }
else { i = s . indexOf ( ' ' ) ; }
let real = s . substr ( 0 , i ) ;
let imaginary = s . substr ( i + 1 , l - 2 ) ; document . write ( " " + real + " " ) ; document . write ( " " + imaginary ) ; }
let s = " " ; findRealAndImag ( s ) ;
function highestPower ( n , k ) { let i = 0 ; let a = Math . pow ( n , i ) ;
while ( a <= k ) { i += 1 ; a = Math . pow ( n , i ) ; } return i - 1 ; }
let b = Array . from ( { length : 50 } , ( _ , i ) => 0 ) ;
function PowerArray ( n , k ) { while ( k > 0 ) {
let t = highestPower ( n , k ) ;
if ( b [ t ] > 0 ) {
document . write ( - 1 ) ; return 0 ; } else
b [ t ] = 1 ;
k -= Math . pow ( n , t ) ; }
for ( let i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] > 0 ) { document . write ( i + " " ) ; } } return 0 ; }
let N = 3 ; let K = 40 ; PowerArray ( N , K ) ;
let N = 100005
function SieveOfEratosthenes ( composite ) { for ( let i = 0 ; i < N ; i ++ ) composite [ i ] = false ; for ( let p = 2 ; p * p < N ; p ++ ) {
if ( ! composite [ p ] ) {
for ( let i = p * 2 ; i < N ; i += p ) composite [ i ] = true ; } } }
function sumOfElements ( arr , n ) { let composite = new Array ( N ) ; SieveOfEratosthenes ( composite ) ;
let m = new Map ( ) ; for ( let i = 0 ; i < n ; i ++ ) if ( m . has ( arr [ i ] ) ) { m [ arr [ i ] ] = m [ arr [ i ] ] + 1 ; } else { m . set ( arr [ i ] , 1 ) ; }
let sum = 0 ;
m . forEach ( ( value , key ) => {
if ( composite [ key ] ) { sum += value ; } } ) return sum ; }
let arr = [ 1 , 2 , 1 , 1 , 1 , 3 , 3 , 2 , 4 ] ; let n = arr . length ;
document . write ( sumOfElements ( arr , n ) ) ;
function remove ( arr , n ) {
let mp = new Map ( ) ; for ( let i = 0 ; i < n ; i ++ ) { if ( mp . has ( arr [ i ] ) ) { mp . set ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . set ( arr [ i ] , 1 ) ; } }
for ( let i = 0 ; i < n ; i ++ ) {
if ( ( mp . has ( arr [ i ] ) && mp . get ( arr [ i ] ) % 2 == 1 ) ) continue ; document . write ( arr [ i ] + " " ) ; } }
let arr = [ 3 , 3 , 3 , 2 , 2 , 4 , 7 , 7 ] ; let n = arr . length ;
remove ( arr , n ) ;
function getmax ( arr , n , x ) {
let s = 0 ;
for ( let i = 0 ; i < n ; i ++ ) { s = s + arr [ i ] ; }
document . write ( Math . min ( s , x ) ) ; }
let arr = [ 1 , 2 , 3 , 4 ] ; let x = 5 ; let arr_size = arr . length ; getmax ( arr , arr_size , x ) ;
function shortestLength ( n , x , y ) { let answer = 0 ;
let i = 0 ; while ( n != 0 && i < x . length ) {
if ( x [ i ] + y [ i ] > answer ) answer = x [ i ] + y [ i ] ; i ++ ; }
document . write ( " " + answer + " " ) ; document . write ( " " + " " + answer + " " + " " + answer + " " ) ; }
let n = 4 ;
let x = [ 1 , 4 , 2 , 1 ] ; let y = [ 4 , 1 , 1 , 2 ] ; shortestLength ( n , x , y ) ;
function FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) {
var x5 = Math . max ( x1 , x3 ) ; var y5 = Math . max ( y1 , y3 ) ;
var x6 = Math . min ( x2 , x4 ) ; var y6 = Math . min ( y2 , y4 ) ;
if ( x5 > x6 y5 > y6 ) { document . write ( " " ) ; return ; } document . write ( " " + x5 + " " + y5 + " " ) ; document . write ( " " + x6 + " " + y6 + " " ) ;
var x7 = x5 ; var y7 = y6 ; document . write ( " " + x7 + " " + y7 + " " ) ;
var x8 = x6 ; var y8 = y5 ; document . write ( " " + x8 + " " + y8 + " " ) ; }
var x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
var x3 = 2 , y3 = 3 , x4 = 7 , y4 = 9 ;
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) ;
class Point { constructor ( a , b ) { this . x = a ; this . y = b ; } }
function printCorners ( p , q , l ) { let a = new Point ( ) , b = new Point ( ) , c = new Point ( ) , d = new Point ( ) ;
if ( p . x == q . x ) { a . x = ( p . x - ( l / 2.0 ) ) ; a . y = p . y ; d . x = ( p . x + ( l / 2.0 ) ) ; d . y = p . y ; b . x = ( q . x - ( l / 2.0 ) ) ; b . y = q . y ; c . x = ( q . x + ( l / 2.0 ) ) ; c . y = q . y ; }
else if ( p . y == q . y ) { a . y = ( p . y - ( l / 2.0 ) ) ; a . x = p . x ; d . y = ( p . y + ( l / 2.0 ) ) ; d . x = p . x ; b . y = ( q . y - ( l / 2.0 ) ) ; b . x = q . x ; c . y = ( q . y + ( l / 2.0 ) ) ; c . x = q . x ; }
else {
let m = ( p . x - q . x ) / ( q . y - p . y ) ;
let dx = ( ( l / Math . sqrt ( 1 + ( m * m ) ) ) * 0.5 ) ; let dy = m * dx ; a . x = p . x - dx ; a . y = p . y - dy ; d . x = p . x + dx ; d . y = p . y + dy ; b . x = q . x - dx ; b . y = q . y - dy ; c . x = q . x + dx ; c . y = q . y + dy ; } document . write ( a . x + " " + a . y + " " + b . x + " " + b . y + " " + c . x + " " + c . y + " " + d . x + " " + d . y + " " ) ; }
let p1 = new Point ( 1 , 0 ) , q1 = new Point ( 1 , 2 ) ; printCorners ( p1 , q1 , 2 ) ; let p = new Point ( 1 , 1 ) , q = new Point ( - 1 , - 1 ) ; printCorners ( p , q , ( 2 * Math . sqrt ( 2 ) ) ) ;
function minimumCost ( arr , N , X , Y ) {
let even_count = 0 , odd_count = 0 ; for ( let i = 0 ; i < N ; i ++ ) {
if ( ( arr [ i ] & 1 ) && ( i % 2 == 0 ) ) { odd_count ++ ; }
if ( ( arr [ i ] % 2 ) == 0 && ( i & 1 ) ) { even_count ++ ; } }
let cost1 = X * Math . min ( odd_count , even_count ) ;
let cost2 = Y * ( Math . max ( odd_count , even_count ) - Math . min ( odd_count , even_count ) ) ;
let cost3 = ( odd_count + even_count ) * Y ;
return Math . min ( cost1 + cost2 , cost3 ) ; }
let arr = [ 5 , 3 , 7 , 2 , 1 ] , X = 10 , Y = 2 ; let N = arr . length ; document . write ( minimumCost ( arr , N , X , Y ) ) ;
function findMinMax ( a ) {
let min_val = 1000000000 ;
for ( let i = 1 ; i < a . length ; ++ i ) {
min_val = Math . min ( min_val , a [ i ] * a [ i - 1 ] ) ; }
return min_val ; }
let arr = [ 6 , 4 , 5 , 6 , 2 , 4 , 1 ] ; document . write ( findMinMax ( arr ) )
let sum = 0 ; class TreeNode {
constructor ( data = " " , left = null , right = null ) { this . data = data ; this . left = left ; this . right = right ; } }
function kDistanceDownSum ( root , k ) {
if ( root == null k < 0 ) { return }
if ( k == 0 ) { sum += root . data ; return ; }
kDistanceDownSum ( root . left , k - 1 ) ; kDistanceDownSum ( root . right , k - 1 ) ; }
function kDistanceSum ( root , target , k ) {
if ( root == null ) return - 1 ;
if ( root . data == target ) { kDistanceDownSum ( root . left , k - 1 ) ; return 0 ; }
let dl = - 1 ;
if ( target < root . data ) { dl = kDistanceSum ( root . left , target , k ) ; }
if ( dl != - 1 ) {
if ( dl + 1 == k ) sum += root . data ;
return - 1 ; }
let dr = - 1 ; if ( target > root . data ) { dr = kDistanceSum ( root . right , target , k ) ; } if ( dr != - 1 ) {
if ( dr + 1 == k ) sum += root . data ;
else kDistanceDownSum ( root . left , k - dr - 2 ) ; return 1 + dr ; }
return - 1 ; }
function insertNode ( data , root ) {
if ( root == null ) { let node = new TreeNode ( data ) ; return node ; }
else if ( data > root . data ) { root . right = insertNode ( data , root . right ) ; }
else if ( data <= root . data ) { root . left = insertNode ( data , root . left ) ; }
return root ; }
function findSum ( root , target , K ) {
kDistanceSum ( root , target , K , sum ) ;
document . write ( sum ) ; }
let root = null ; let N = 11 ; let tree = [ 3 , 1 , 7 , 0 , 2 , 5 , 10 , 4 , 6 , 9 , 8 ] ;
for ( let i = 0 ; i < N ; i ++ ) { root = insertNode ( tree [ i ] , root ) ; } let target = 7 ; let K = 2 ; findSum ( root , target , K ) ;
function itemType ( n ) {
let count = 0 ;
for ( let day = 1 ; ; day ++ ) {
for ( let type = day ; type > 0 ; type -- ) { count += type ;
if ( count >= n ) return type ; } } }
let N = 10 ; document . write ( itemType ( N ) ) ;
function FindSum ( arr , N ) {
let res = 0 ;
for ( let i = 0 ; i < N ; i ++ ) {
let power = Math . floor ( Math . log2 ( arr [ i ] ) ) ;
let LesserValue = Math . pow ( 2 , power ) ;
let LargerValue = Math . pow ( 2 , power + 1 ) ;
if ( ( arr [ i ] - LesserValue ) == ( LargerValue - arr [ i ] ) ) {
res += arr [ i ] ; } }
return res ; }
let arr = [ 10 , 24 , 17 , 3 , 8 ] ; let N = arr . length ; document . write ( FindSum ( arr , N ) ) ;
function findLast ( mat ) { let m = mat . length ; let n = mat [ 0 ] . length ;
let rows = new Set ( ) ; let cols = new Set ( ) ; for ( let i = 0 ; i < m ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) { if ( mat [ i ] [ j ] ) { rows . add ( i ) ; cols . add ( j ) ; } } }
let avRows = m - rows . size ; let avCols = n - cols . size ;
let choices = Math . min ( avRows , avCols ) ;
if ( choices & 1 )
document . write ( " " )
else document . write ( " " ) }
let mat = [ [ 1 , 0 , 0 ] , [ 0 , 0 , 0 ] , [ 0 , 0 , 1 ] ] findLast ( mat ) ;
let MOD = 1000000007 ;
function sumOfBinaryNumbers ( n ) {
let ans = 0 ; let one = 1 ;
while ( true ) {
if ( n <= 1 ) { ans = ( ans + n ) % MOD ; break ; }
let x = Math . floor ( Math . log ( n ) / Math . log ( 2 ) ) ; let cur = 0 ; let add = Math . floor ( Math . pow ( 2 , ( x - 1 ) ) ) ;
for ( let i = 1 ; i <= x ; i ++ ) {
cur = ( cur + add ) % MOD ; add = ( add * 10 % MOD ) ; }
ans = ( ans + cur ) % MOD ;
let rem = n - Math . floor ( Math . pow ( 2 , x ) ) + 1 ;
let p = Math . floor ( Math . pow ( 10 , x ) ) ; p = ( p * ( rem % MOD ) ) % MOD ; ans = ( ans + p ) % MOD ;
n = rem - 1 ; }
document . write ( ans ) ; }
let N = 3 ; sumOfBinaryNumbers ( N ) ;
function nearestFibonacci ( num ) {
if ( num == 0 ) { document . write ( 0 ) ; return ; }
let first = 0 , second = 1 ;
let third = first + second ;
while ( third <= num ) {
first = second ;
second = third ;
third = first + second ; }
let ans = ( Math . abs ( third - num ) >= Math . abs ( second - num ) ) ? second : third ;
document . write ( ans ) ; }
let N = 17 ; nearestFibonacci ( N ) ;
function checkPermutation ( ans , a , n ) {
let Max = Number . MIN_VALUE ;
for ( let i = 0 ; i < n ; i ++ ) {
Max = Math . max ( Max , ans [ i ] ) ;
if ( Max != a [ i ] ) return false ; }
return true ; }
function findPermutation ( a , n ) {
let ans = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { ans [ i ] = 0 ; }
let um = new Map ( ) ;
for ( let i = 0 ; i < n ; i ++ ) {
if ( ! um . has ( a [ i ] ) ) {
ans [ i ] = a [ i ] ; um . set ( a [ i ] , i ) ; } }
let v = [ ] ; let j = 0 ;
for ( let i = 1 ; i <= n ; i ++ ) {
if ( ! um . has ( i ) ) { v . push ( i ) ; } }
for ( let i = 0 ; i < n ; i ++ ) {
if ( ans [ i ] == 0 ) { ans [ i ] = v [ j ] ; j ++ ; } }
if ( checkPermutation ( ans , a , n ) ) {
for ( let i = 0 ; i < n ; i ++ ) { document . write ( ans [ i ] + " " ) ; } }
else document . write ( " " ) ; }
let arr = [ 1 , 3 , 4 , 5 , 5 ] ; let N = arr . length ;
findPermutation ( arr , N ) ;
function countEqualElementPairs ( arr , N ) {
var mp = new Map ( ) ;
for ( var i = 0 ; i < N ; i ++ ) { if ( mp . has ( arr [ i ] ) ) { mp . set ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . set ( arr [ i ] , 1 ) ; } }
var total = 0 ;
mp . forEach ( ( value , key ) => {
total += ( value * ( value - 1 ) ) / 2 ; } ) ;
for ( var i = 0 ; i < N ; i ++ ) {
document . write ( total - ( mp . get ( arr [ i ] ) - 1 ) + " " ) ; } }
var arr = [ 1 , 1 , 2 , 1 , 2 ] ;
var N = arr . length ; countEqualElementPairs ( arr , N ) ;
function count ( N ) { var sum = 0 ;
for ( var i = 1 ; i <= N ; i ++ ) { sum += 7 * Math . pow ( 8 , i - 1 ) ; } return sum ; }
var N = 4 ; document . write ( count ( N ) ) ;
function isPalindrome ( n ) {
var str = ( n . toString ( ) ) ;
var s = 0 , e = str . length - 1 ; while ( s < e ) {
if ( str [ s ] != str [ e ] ) { return false ; } s ++ ; e -- ; } return true ; }
function palindromicDivisors ( n ) {
var PalindromDivisors = [ ] ; for ( var i = 1 ; i <= parseInt ( Math . sqrt ( n ) ) ; i ++ ) {
if ( n % i == 0 ) {
if ( n / i == i ) {
if ( isPalindrome ( i ) ) { PalindromDivisors . push ( i ) ; } } else {
if ( isPalindrome ( i ) ) { PalindromDivisors . push ( i ) ; }
if ( isPalindrome ( n / i ) ) { PalindromDivisors . push ( n / i ) ; } } } }
PalindromDivisors . sort ( ( a , b ) => a - b ) for ( var i = 0 ; i < PalindromDivisors . length ; i ++ ) { document . write ( PalindromDivisors [ i ] + " " ) ; } }
var n = 66 ;
palindromicDivisors ( n ) ;
function findMinDel ( arr , n ) {
var min_num = 1000000000 ;
for ( var i = 0 ; i < n ; i ++ ) min_num = Math . min ( arr [ i ] , min_num ) ;
var cnt = 0 ;
for ( var i = 0 ; i < n ; i ++ ) if ( arr [ i ] == min_num ) cnt ++ ;
return n - cnt ; }
var arr = [ 3 , 3 , 2 ] ; var n = arr . length ; document . write ( findMinDel ( arr , n ) ) ;
function __gcd ( a , b ) { if ( b == 0 ) return a ; return __gcd ( b , a % b ) ; }
function cntSubArr ( arr , n ) {
var ans = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
var curr_gcd = 0 ;
for ( var j = i ; j < n ; j ++ ) { curr_gcd = __gcd ( curr_gcd , arr [ j ] ) ;
ans += ( curr_gcd == 1 ) ; } }
return ans ; }
var arr = [ 1 , 1 , 1 ] ; var n = arr . length ; document . write ( cntSubArr ( arr , n ) ) ;
function print_primes_till_N ( N ) {
let i , j , flag ;
document . write ( " " + N + " " ) ;
for ( i = 1 ; i <= N ; i ++ ) {
if ( i == 1 i == 0 ) continue ;
flag = 1 ; for ( j = 2 ; j <= i / 2 ; ++ j ) { if ( i % j == 0 ) { flag = 0 ; break ; } }
if ( flag == 1 ) document . write ( i + " " ) ; } }
let N = 100 ; print_primes_till_N ( N ) ;
function findX ( A , B ) { var X = 0 ; var MAX = 32 ;
for ( var bit = 0 ; bit < MAX ; bit ++ ) {
var tempBit = 1 << bit ;
var bitOfX = A & B & tempBit ;
X += bitOfX ; } return X ; }
var A = 11 , B = 13 ; document . write ( findX ( A , B ) ) ;
function cntSubSets ( arr , n ) {
var maxVal = arr . reduce ( function ( a , b ) { return Math . max ( a , b ) ; } ) ;
var cnt = 0 ; for ( var i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == maxVal ) cnt ++ ; }
return ( Math . pow ( 2 , cnt ) - 1 ) ; }
var arr = [ 1 , 2 , 1 , 2 ] var n = arr . length ; document . write ( cntSubSets ( arr , n ) ) ;
function findProb ( arr , n ) {
var maxSum = - 100000000 , maxCount = 0 , totalPairs = 0 ;
for ( var i = 0 ; i < n - 1 ; i ++ ) { for ( var j = i + 1 ; j < n ; j ++ ) {
var sum = arr [ i ] + arr [ j ] ;
if ( sum == maxSum ) {
maxCount ++ ; }
else if ( sum > maxSum ) {
maxSum = sum ; maxCount = 1 ; } totalPairs ++ ; } }
var prob = maxCount / totalPairs ; return prob ; }
var arr = [ 1 , 1 , 1 , 2 , 2 , 2 ] var n = arr . length ; document . write ( findProb ( arr , n ) ) ;
function GCD ( a , b ) { if ( b == 0 ) return a ; return GCD ( b , a % b ) ; }
function maxCommonFactors ( a , b ) {
let gcd = GCD ( a , b ) ;
let ans = 1 ;
for ( let i = 2 ; i * i <= gcd ; i ++ ) { if ( gcd % i == 0 ) { ans ++ ; while ( gcd % i == 0 ) gcd = parseInt ( gcd / i ) ; } }
if ( gcd != 1 ) ans ++ ;
return ans ; }
let a = 12 , b = 18 ; document . write ( maxCommonFactors ( a , b ) ) ;
var days = [ 31 , 28 , 31 , 30 , 31 , 30 , 31 , 31 , 30 , 31 , 30 , 31 ] ;
function dayOfYear ( date ) {
var year = parseInt ( date . substring ( 0 , 4 ) ) ; var month = parseInt ( date . substring ( 5 , 6 ) ) ; var day = parseInt ( date . substring ( 8 ) ) ;
if ( month > 2 && year % 4 == 0 && ( year % 100 != 0 year % 400 == 0 ) ) { ++ day ; }
while ( month -- > 0 ) { day = day + days [ month - 1 ] ; } return day ; }
var date = " " ; document . write ( dayOfYear ( date ) ) ;
function Cells ( n , x ) { let ans = 0 ; for ( let i = 1 ; i <= n ; i ++ ) if ( x % i == 0 && parseInt ( x / i ) <= n ) ans ++ ; return ans ; }
let n = 6 , x = 12 ;
document . write ( Cells ( n , x ) ) ;
function nextPowerOfFour ( n ) { let x = Math . floor ( Math . sqrt ( Math . sqrt ( n ) ) ) ;
if ( Math . pow ( x , 4 ) == n ) return n ; else { x = x + 1 ; return Math . pow ( x , 4 ) ; } }
let n = 122 ; document . write ( nextPowerOfFour ( n ) ) ;
function minOperations ( x , y , p , q ) {
if ( y % x != 0 ) return - 1 var d = Math . floor ( y / x )
var a = 0
while ( d % p == 0 ) { d = Math . floor ( d / p ) a += 1 }
var b = 0
while ( d % q == 0 ) { d = Math . floor ( d / q ) b += 1 }
if ( d != 1 ) return - 1
return ( a + b ) }
var x = 12 var y = 2592 var p = 2 var q = 3 document . write ( minOperations ( x , y , p , q ) )
function nCr ( n ) {
if ( n < 4 ) return 0 ; let answer = n * ( n - 1 ) * ( n - 2 ) * ( n - 3 ) ; answer = parseInt ( answer / 24 ) ; return answer ; }
function countQuadruples ( N , K ) {
let M = parseInt ( N / K ) ; let answer = nCr ( M ) ;
for ( let i = 2 ; i < M ; i ++ ) { let j = i ;
let temp2 = parseInt ( M / i ) ;
let count = 0 ;
let check = 0 ; let temp = j ; while ( j % 2 == 0 ) { count ++ ; j = parseInt ( j / 2 ) ; if ( count >= 2 ) break ; } if ( count >= 2 ) { check = 1 ; } for ( let k = 3 ; k <= Math . sqrt ( temp ) ; k += 2 ) { let cnt = 0 ; while ( j % k == 0 ) { cnt ++ ; j = parseInt ( j / k ) ; if ( cnt >= 2 ) break ; } if ( cnt >= 2 ) { check = 1 ; break ; } else if ( cnt == 1 ) count ++ ; } if ( j > 2 ) { count ++ ; }
if ( check ) continue ; else {
if ( count % 2 == 1 ) { answer -= nCr ( temp2 ) ; } else { answer += nCr ( temp2 ) ; } } } return answer ; }
let N = 10 , K = 2 ; document . write ( countQuadruples ( N , K ) ) ;
function getX ( a , b , c , d ) { var X = ( b * c - a * d ) / ( d - c ) ; return X ; }
var a = 2 , b = 3 , c = 4 , d = 5 ; document . write ( getX ( a , b , c , d ) ) ;
function isVowel ( ch ) { if ( ch == ' ' ch == ' ' ch == ' ' ch == ' ' ch == ' ' ) return true ; else return false ; }
function fact ( n ) { if ( n < 2 ) { return 1 ; } return n * fact ( n - 1 ) ; }
function only_vowels ( freq ) { let denom = 1 ; let cnt_vwl = 0 ;
for ( let [ key , value ] of freq . entries ( ) ) { if ( isVowel ( key ) ) { denom *= fact ( value ) ; cnt_vwl += value ; } } return Math . floor ( fact ( cnt_vwl ) / denom ) ; }
function all_vowels_together ( freq ) {
let vow = only_vowels ( freq ) ;
let denom = 1 ;
let cnt_cnst = 0 ; for ( let [ key , value ] of freq . entries ( ) ) { if ( ! isVowel ( key ) ) { denom *= fact ( value ) ; cnt_cnst += value ; } }
let ans = Math . floor ( fact ( cnt_cnst + 1 ) / denom ) ; return ( ans * vow ) ; }
function total_permutations ( freq ) {
let cnt = 0 ;
let denom = 1 ; for ( let [ key , value ] of freq . entries ( ) ) { denom *= fact ( value ) ; cnt += value ; }
return Math . floor ( fact ( cnt ) / denom ) ; }
function no_vowels_together ( word ) {
let freq = new Map ( ) ;
for ( let i = 0 ; i < word . length ; i ++ ) { let ch = word [ i ] . toLowerCase ( ) ; if ( freq . has ( ch ) ) { freq . set ( ch , freq . get ( ch ) + 1 ) ; } else { freq . set ( ch , 1 ) ; } }
let total = total_permutations ( freq ) ;
let vwl_tgthr = all_vowels_together ( freq ) ;
let res = total - vwl_tgthr ;
return res ; }
let word = " " ; let ans = no_vowels_together ( word ) ; document . write ( ans + " " ) ; word = " " ; ans = no_vowels_together ( word ) ; document . write ( ans + " " ) ; word = " " ; ans = no_vowels_together ( word ) ; document . write ( ans + " " ) ;
function numberOfMen ( D , m , d ) { var Men = ( m * ( D - d ) ) / d ; return Men ; }
var D = 5 , m = 4 , d = 4 ; document . write ( numberOfMen ( D , m , d ) ) ;
function area ( a , b , c ) { var d = Math . abs ( ( c * c ) / ( 2 * a * b ) ) ; return d ; }
var a = - 2 , b = 4 , c = 3 ; document . write ( area ( a , b , c ) ) ;
function addToArrayForm ( A , K ) {
let v = [ ] ; let ans = [ ] ;
let rem = 0 ; let i = 0 ;
for ( i = A . length - 1 ; i >= 0 ; i -- ) {
let my = A [ i ] + K % 10 + rem ; if ( my > 9 ) {
rem = 1 ;
v . push ( my % 10 ) ; } else { v . push ( my ) ; rem = 0 ; } K = parseInt ( K / 10 , 10 ) ; }
while ( K > 0 ) {
let my = K % 10 + rem ; v . push ( my % 10 ) ;
if ( parseInt ( my / 10 , 10 ) > 0 ) rem = 1 ; else rem = 0 ; K = parseInt ( K / 10 , 10 ) ; } if ( rem > 0 ) v . push ( rem ) ;
for ( let j = v . length - 1 ; j >= 0 ; j -- ) ans . push ( v [ j ] ) ; return ans ; }
let A = [ ] ; A . push ( 2 ) ; A . push ( 7 ) ; A . push ( 4 ) ; let K = 181 ; let ans = addToArrayForm ( A , K ) ;
for ( let i = 0 ; i < ans . length ; i ++ ) document . write ( ans [ i ] ) ;
const MAX = 100005 ;
function kadaneAlgorithm ( ar , n ) { let sum = 0 , maxSum = 0 ; for ( let i = 0 ; i < n ; i ++ ) { sum += ar [ i ] ; if ( sum < 0 ) sum = 0 ; maxSum = Math . max ( maxSum , sum ) ; } return maxSum ; }
function maxFunction ( arr , n ) { let b = new Array ( MAX ) , c = new Array ( MAX ) ;
for ( let i = 0 ; i < n - 1 ; i ++ ) { if ( i & 1 ) { b [ i ] = Math . abs ( arr [ i + 1 ] - arr [ i ] ) ; c [ i ] = - b [ i ] ; } else { c [ i ] = Math . abs ( arr [ i + 1 ] - arr [ i ] ) ; b [ i ] = - c [ i ] ; } }
let ans = kadaneAlgorithm ( b , n - 1 ) ; ans = Math . max ( ans , kadaneAlgorithm ( c , n - 1 ) ) ; return ans ; }
let arr = [ 1 , 5 , 4 , 7 ] ; let n = arr . length ; document . write ( maxFunction ( arr , n ) ) ;
function findThirdDigit ( n ) {
if ( n < 3 ) return 0 ;
return n & 1 ? 1 : 6 ; }
var n = 7 ; document . write ( findThirdDigit ( n ) ) ;
function getProbability ( a , b , c , d ) {
var p = a / b ; var q = c / d ;
var ans = p * ( 1 / ( 1 - ( 1 - q ) * ( 1 - p ) ) ) ; return ans ; }
var a = 1 , b = 2 , c = 10 , d = 11 ; document . write ( getProbability ( a , b , c , d ) . toFixed ( 5 ) ) ;
function isPalindrome ( n ) {
var divisor = 1 ; while ( parseInt ( n / divisor ) >= 10 ) divisor *= 10 ; while ( n != 0 ) { var leading = parseInt ( n / divisor ) ; var trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = parseInt ( ( n % divisor ) / 10 ) ;
divisor = parseInt ( divisor / 100 ) ; } return true ; }
function largestPalindrome ( A , n ) { var currentMax = - 1 ; for ( var i = 0 ; i < n ; i ++ ) {
if ( A [ i ] > currentMax && isPalindrome ( A [ i ] ) ) currentMax = A [ i ] ; }
return currentMax ; }
var A = [ 1 , 232 , 54545 , 999991 ] ; var n = A . length ;
document . write ( largestPalindrome ( A , n ) ) ;
function getFinalElement ( n ) { let finalNum ; for ( finalNum = 2 ; finalNum * 2 <= n ; finalNum *= 2 ) ; return finalNum ; }
let N = 12 ; document . write ( getFinalElement ( N ) ) ;
function SieveOfEratosthenes ( prime , p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( let p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( let i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
function sumOfElements ( arr , n ) { let prime = new Array ( n + 1 ) ; prime . fill ( true ) SieveOfEratosthenes ( prime , n + 1 ) ; let i , j ;
let m = new Map ( ) ; for ( i = 0 ; i < n ; i ++ ) { if ( m . has ( arr [ i ] ) ) m . set ( arr [ i ] , m . get ( arr [ i ] ) + 1 ) ; else m . set ( arr [ i ] , 1 ) ; } let sum = 0 ;
for ( let it of m ) {
if ( prime [ it [ 1 ] ] ) { sum += ( it [ 0 ] ) ; } } return sum ; }
let arr = [ 5 , 4 , 6 , 5 , 4 , 6 ] ; let n = arr . length ; document . write ( sumOfElements ( arr , n ) ) ;
function isPalindrome ( num ) { let reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp = Math . floor ( temp / 10 ) ; }
if ( reverse_num == num ) { return true ; } return false ; }
function isOddLength ( num ) { let count = 0 ; while ( num > 0 ) { num = Math . floor ( num / 10 ) ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
function sumOfAllPalindrome ( L , R ) { let sum = 0 ; if ( L <= R ) for ( let i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
let L = 110 , R = 1130 ; document . write ( sumOfAllPalindrome ( L , R ) ) ;
function fact ( n ) { let f = 1 ; for ( let i = 2 ; i <= n ; i ++ ) f = f * i ; return f ; }
function waysOfConsonants ( size1 , freq ) { let ans = fact ( size1 ) ; for ( let i = 0 ; i < 26 ; i ++ ) {
if ( i == 0 i == 4 i == 8 i == 14 i == 20 ) continue ; else ans = Math . floor ( ans / fact ( freq [ i ] ) ) ; } return ans ; }
function waysOfVowels ( size2 , freq ) { return Math . floor ( fact ( size2 ) / ( fact ( freq [ 0 ] ) * fact ( freq [ 4 ] ) * fact ( freq [ 8 ] ) * fact ( freq [ 14 ] ) * fact ( freq [ 20 ] ) ) ) ; }
function countWays ( str ) { let freq = new Array ( 200 ) ; for ( let i = 0 ; i < 200 ; i ++ ) freq [ i ] = 0 ; for ( let i = 0 ; i < str . length ; i ++ ) freq [ str [ i ] . charCodeAt ( 0 ) - ' ' . charCodeAt ( 0 ) ] ++ ;
let vowel = 0 , consonant = 0 ; for ( let i = 0 ; i < str . length ; i ++ ) { if ( str [ i ] != ' ' && str [ i ] != ' ' && str [ i ] != ' ' && str [ i ] != ' ' && str [ i ] != ' ' ) consonant ++ ; else vowel ++ ; }
return waysOfConsonants ( consonant + 1 , freq ) * waysOfVowels ( vowel , freq ) ; }
let str = " " ; document . write ( countWays ( str ) ) ;
function calculateAlternateSum ( n ) { if ( n <= 0 ) return 0 ; var fibo = Array ( n + 1 ) . fill ( 0 ) ; fibo [ 0 ] = 0 ; fibo [ 1 ] = 1 ;
var sum = Math . pow ( fibo [ 0 ] , 2 ) + Math . pow ( fibo [ 1 ] , 2 ) ;
for ( i = 2 ; i <= n ; i ++ ) { fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] ;
if ( i % 2 == 0 ) sum -= fibo [ i ] ;
else sum += fibo [ i ] ; }
return sum ; }
var n = 8 ;
document . write ( " " + n + " " + calculateAlternateSum ( n ) ) ;
function getValue ( n ) { let i = 0 , k = 1 ; while ( i < n ) { i = i + k ; k = k * 2 ; } return parseInt ( k / 2 ) ; }
let n = 9 ;
document . write ( getValue ( n ) + " " ) ;
n = 1025 ;
document . write ( getValue ( n ) + " " ) ;
function countDigits ( val , arr ) { while ( val > 0 ) { let digit = val % 10 ; arr [ digit ] ++ ; val = Math . floor ( val / 10 ) ; } return ; } function countFrequency ( x , n ) {
let freq_count = new Array ( 10 ) ; for ( let i = 0 ; i < 10 ; i ++ ) { freq_count [ i ] = 0 ; }
for ( let i = 1 ; i <= n ; i ++ ) {
let val = Math . pow ( x , i ) ;
countDigits ( val , freq_count ) ; }
for ( let i = 0 ; i <= 9 ; i ++ ) { document . write ( freq_count [ i ] + " " ) ; } }
let x = 15 , n = 3 ; countFrequency ( x , n ) ;
function countSolutions ( a ) { let count = 0 ;
for ( let i = 0 ; i <= a ; i ++ ) { if ( a == ( i + ( a ^ i ) ) ) count ++ ; } return count ; }
let a = 3 ; document . write ( countSolutions ( a ) ) ;
function bitCount ( n ) { let count = 0 ; while ( n != 0 ) { count ++ ; n &= ( n - 1 ) ; } return count ; }
function countSolutions ( a ) { let count = bitCount ( a ) ; count = Math . pow ( 2 , count ) ; return count ; }
let a = 3 ; document . write ( countSolutions ( a ) ) ;
function calculateAreaSum ( l , b ) { var size = 1 ;
var maxSize = Math . min ( l , b ) ; var totalArea = 0 ; for ( var i = 1 ; i <= maxSize ; i ++ ) {
var totalSquares = ( l - size + 1 ) * ( b - size + 1 ) ;
var area = totalSquares * size * size ;
totalArea += area ;
size ++ ; } return totalArea ; }
var l = 4 , b = 3 ; document . write ( calculateAreaSum ( l , b ) ) ;
function boost_hyperfactorial ( num ) {
let val = 1 ; for ( let i = 1 ; i <= num ; i ++ ) { val = val * Math . pow ( i , i ) ; }
return val ; }
let num = 5 ; document . write ( boost_hyperfactorial ( num ) ) ;
function boost_hyperfactorial ( num ) {
var val = 1 ; for ( var i = 1 ; i <= num ; i ++ ) { for ( var j = 1 ; j <= i ; j ++ ) {
val *= i ; } }
return val ; }
var num = 5 ; document . write ( boost_hyperfactorial ( num ) ) ;
function subtractOne ( x ) { let m = 1 ;
while ( ! ( x & m ) ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
document . write ( subtractOne ( 13 ) ) ;
var rows = 3 ; var cols = 3 ;
function meanVector ( mat ) { document . write ( " " ) ;
for ( var i = 0 ; i < rows ; i ++ ) {
var mean = 0.00 ;
var sum = 0 ; for ( var j = 0 ; j < cols ; j ++ ) sum += mat [ j ] [ i ] ; mean = sum / rows ; document . write ( mean + " " ) ; } document . write ( " " ) ; }
var mat = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] , [ 7 , 8 , 9 ] ] ; meanVector ( mat ) ;
function primeFactors ( n ) { var res = [ ] ; if ( n % 2 == 0 ) { while ( n % 2 == 0 ) n = parseInt ( n / 2 ) ; res . push ( 2 ) ; }
for ( var i = 3 ; i <= Math . sqrt ( n ) ; i = i + 2 ) {
if ( n % i == 0 ) { while ( n % i == 0 ) n = parseInt ( n / i ) ; res . push ( i ) ; } }
if ( n > 2 ) res . push ( n ) ; return res ; }
function isHoax ( n ) {
var pf = primeFactors ( n ) ;
if ( pf [ 0 ] == n ) return false ;
var all_pf_sum = 0 ; for ( var i = 0 ; i < pf . length ; i ++ ) {
var pf_sum ; for ( pf_sum = 0 ; pf [ i ] > 0 ; pf_sum += pf [ i ] % 10 , pf [ i ] = parseInt ( pf [ i ] / 10 ) ) ; all_pf_sum += pf_sum ; }
var sum_n ; for ( sum_n = 0 ; n > 0 ; sum_n += n % 10 , n = parseInt ( n / 10 ) ) ;
return sum_n == all_pf_sum ; }
var n = 84 ; if ( isHoax ( n ) ) document . write ( " " ) ; else document . write ( " " ) ;
function LucasLehmer ( n ) {
let current_val = 4 ;
let series = [ ] ;
series . push ( current_val ) ; for ( let i = 0 ; i < n ; i ++ ) { current_val = ( current_val * current_val ) - 2 ; series . push ( current_val ) ; }
for ( let i = 0 ; i <= n ; i ++ ) { document . write ( " " + i + " " + series [ i ] + " " ) ; } }
let n = 5 ; LucasLehmer ( n ) ;
function modInverse ( a , prime ) { a = a % prime ; for ( let x = 1 ; x < prime ; x ++ ) if ( ( a * x ) % prime == 1 ) return x ; return - 1 ; } function printModIverses ( n , prime ) { for ( let i = 1 ; i <= n ; i ++ ) document . write ( modInverse ( i , prime ) + " " ) ; }
let n = 10 ; let prime = 17 ; printModIverses ( n , prime ) ;
function minOp ( num ) {
var rem ; var count = 0 ;
while ( num ) { rem = num % 10 ; if ( ! ( rem == 3 rem == 8 ) ) count ++ ; num = parseInt ( num / 10 ) ; } return count ; }
var num = 234198 ; document . write ( " " + minOp ( num ) ) ;
function sumOfDigits ( a ) { var sum = 0 ; while ( a != 0 ) { sum += a % 10 ; a = parseInt ( a / 10 ) ; } return sum ; }
function findMax ( x ) {
var b = 1 , ans = x ;
while ( x != 0 ) {
var cur = ( x - 1 ) * b + ( b - 1 ) ;
if ( sumOfDigits ( cur ) > sumOfDigits ( ans ) || ( sumOfDigits ( cur ) == sumOfDigits ( ans ) && cur > ans ) ) ans = cur ;
x = parseInt ( x / 10 ) ; b *= 10 ; } return ans ; }
var n = 521 ; document . write ( findMax ( n ) ) ;
function median ( a , l , r ) { var n = r - l + 1 ; n = parseInt ( ( n + 1 ) / 2 ) - 1 ; return parseInt ( n + l ) ; }
function IQR ( a , n ) { a . sort ( ( a , b ) => a - b ) ;
var mid_index = median ( a , 0 , n ) ;
var Q1 = a [ median ( a , 0 , mid_index ) ] ;
var Q3 = a [ mid_index + median ( a , mid_index + 1 , n ) ] ;
return ( Q3 - Q1 ) ; }
var a = [ 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 ] ; var n = a . length ; document . write ( IQR ( a , n ) ) ;
function ssort ( a , n ) { var i , j , min , temp ; for ( i = 0 ; i < n - 1 ; i ++ ) { min = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( a [ j ] < a [ min ] ) min = j ; temp = a [ i ] ; a [ i ] = a [ min ] ; a [ min ] = temp ; } }
function isPalindrome ( n ) {
var divisor = 1 ; while ( parseInt ( n / divisor ) >= 10 ) divisor *= 10 ; while ( n != 0 ) { var leading = parseInt ( n / divisor ) ; var trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = parseInt ( ( n % divisor ) / 10 ) ;
divisor = parseInt ( divisor / 100 ) ; } return true ; }
function largestPalindrome ( A , n ) {
ssort ( A , A . length ) ; for ( var i = n - 1 ; i >= 0 ; -- i ) {
if ( isPalindrome ( A [ i ] ) ) return A [ i ] ; }
return - 1 ; } var A = [ 1 , 232 , 54545 , 999991 ] ; var n = A . length ;
document . write ( largestPalindrome ( A , n ) ) ;
function findSum ( n , a , b ) { let sum = 0 ; for ( let i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
let n = 10 ; let a = 3 ; let b = 5 ; document . write ( findSum ( n , a , b ) ) ;
function subtractOne ( x ) { return ( ( x << 1 ) + ( ~ x ) ) ; } document . write ( ( subtractOne ( 13 ) ) ) ;
function pell ( n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
let n = 4 ; document . write ( pell ( n ) ) ;
function LCM ( arr , n ) {
var max_num = 0 ; for ( var i = 0 ; i < n ; i ++ ) if ( max_num < arr [ i ] ) max_num = arr [ i ] ;
var res = 1 ;
while ( x <= max_num ) {
var indexes = [ ] ; for ( var j = 0 ; j < n ; j ++ ) if ( arr [ j ] % x == 0 ) indexes . push ( j ) ;
if ( indexes . length >= 2 ) {
for ( var j = 0 ; j < indexes . length ; j ++ ) arr [ indexes [ j ] ] = arr [ indexes [ j ] ] / x ; res = res * x ; } else x ++ ; }
for ( var i = 0 ; i < n ; i ++ ) res = res * arr [ i ] ; return res ; }
var arr = [ 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 ] ; var n = arr . length ; document . write ( LCM ( arr , n ) + " " ) ;
function politness ( n ) { let count = 0 ;
for ( let i = 2 ; i <= Math . sqrt ( 2 * n ) ; i ++ ) { let a ; if ( ( 2 * n ) % i != 0 ) continue ; a = 2 * n ; a = Math . floor ( a / i ) ; a -= ( i - 1 ) ; if ( a % 2 != 0 ) continue ; a = Math . floor ( a / 2 ) ; if ( a > 0 ) { count ++ ; } } return count ; }
let n = 90 ; document . write ( " " + n + " " + politness ( n ) + " " ) ; n = 15 ; document . write ( " " + n + " " + politness ( n ) ) ;
let MAX = 10000 ;
let primes = new Array ( ) ;
function sieveSundaram ( ) {
let marked = new Array ( parseInt ( MAX / 2 ) + 100 ) . fill ( false ) ;
for ( let i = 1 ; i <= ( Math . sqrt ( MAX ) - 1 ) / 2 ; i ++ ) for ( let j = ( i * ( i + 1 ) ) << 1 ; j <= MAX / 2 ; j = j + 2 * i + 1 ) marked [ j ] = true ;
primes . push ( 2 ) ;
for ( let i = 1 ; i <= MAX / 2 ; i ++ ) if ( marked [ i ] == false ) primes . push ( 2 * i + 1 ) ; }
function findPrimes ( n ) {
if ( n <= 2 n % 2 != 0 ) { document . write ( " " ) ; return ; }
for ( let i = 0 ; primes [ i ] <= n / 2 ; i ++ ) {
let diff = n - primes [ i ] ;
if ( primes . includes ( diff ) ) {
document . write ( primes [ i ] + " " + diff + " " + n + " " ) ; return ; } } }
sieveSundaram ( ) ;
findPrimes ( 4 ) ; findPrimes ( 38 ) ; findPrimes ( 100 ) ;
function kPrimeFactor ( n , k ) {
while ( n % 2 == 0 ) { k -- ; n = n / 2 ; if ( k == 0 ) return 2 ; }
for ( let i = 3 ; i <= Math . sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { if ( k == 1 ) return i ; k -- ; n = n / i ; } }
if ( n > 2 && k == 1 ) return n ; return - 1 ; }
let n = 12 , k = 3 ; document . write ( kPrimeFactor ( n , k ) + " " ) ; n = 14 ; k = 3 ; document . write ( kPrimeFactor ( n , k ) ) ;
var MAX = 10001 ;
function sieveOfEratosthenes ( s ) {
prime = Array . from ( { length : MAX + 1 } , ( _ , i ) => false ) ;
for ( i = 2 ; i <= MAX ; i += 2 ) s [ i ] = 2 ;
for ( i = 3 ; i <= MAX ; i += 2 ) { if ( prime [ i ] == false ) {
s [ i ] = i ;
for ( j = i ; j * i <= MAX ; j += 2 ) { if ( prime [ i * j ] == false ) { prime [ i * j ] = true ;
s [ i * j ] = i ; } } } } }
function kPrimeFactor ( n , k , s ) {
while ( n > 1 ) { if ( k == 1 ) return s [ n ] ;
k -- ;
n /= s [ n ] ; } return - 1 ; }
var s = Array . from ( { length : MAX + 1 } , ( _ , i ) => 0 ) ; sieveOfEratosthenes ( s ) ; var n = 12 , k = 3 ; document . write ( kPrimeFactor ( n , k , s ) + " " ) ; n = 14 ; k = 3 ; document . write ( kPrimeFactor ( n , k , s ) ) ;
function sumDivisorsOfDivisors ( n ) {
let mp = new Map ( ) ; for ( let j = 2 ; j <= Math . sqrt ( n ) ; j ++ ) { let count = 0 ; while ( n % j == 0 ) { n = Math . floor ( n / j ) ; count ++ ; } if ( count != 0 ) mp . set ( j , count ) ; }
if ( n != 1 ) mp . set ( n , 1 ) ;
let ans = 1 ; for ( let [ key , value ] of mp . entries ( ) ) { let pw = 1 ; let sum = 0 ; for ( let i = value + 1 ; i >= 1 ; i -- ) { sum += ( i * pw ) ; pw = key ; } ans *= sum ; } return ans ; }
let n = 10 ; document . write ( sumDivisorsOfDivisors ( n ) ) ;
function prime ( n ) {
if ( n & 1 ) n -= 2 ; else n -- ; let i , j ; for ( i = n ; i >= 2 ; i -= 2 ) { if ( i % 2 == 0 ) continue ; for ( j = 3 ; j <= Math . sqrt ( i ) ; j += 2 ) { if ( i % j == 0 ) break ; } if ( j > Math . sqrt ( i ) ) return i ; }
return 2 ; }
let n = 17 ; document . write ( prime ( n ) ) ;
function fractionToDecimal ( numr , denr ) {
let res = " " ;
let mp = new Map ( ) ; mp . clear ( ) ;
let rem = numr % denr ;
while ( ( rem != 0 ) && ( ! mp . has ( rem ) ) ) {
mp . set ( rem , res . length ) ;
rem = rem * 10 ;
let res_part = Math . floor ( rem / denr ) ; res += res_part . toString ( ) ;
rem = rem % denr ; } if ( rem == 0 ) return " " ; else if ( mp . has ( rem ) ) return res . substr ( mp . get ( rem ) ) ; return " " ; }
let numr = 50 , denr = 22 ; let res = fractionToDecimal ( numr , denr ) ; if ( res == " " ) document . write ( " " ) ; else document . write ( " " + res ) ;
function has0 ( x ) {
while ( x ) {
if ( x % 10 == 0 ) return 1 ; x = Math . floor ( x / 10 ) ; } return 0 ; }
function getCount ( n ) {
let count = 0 ;
for ( let i = 1 ; i <= n ; i ++ ) count += has0 ( i ) ; return count ; }
let n = 107 ; document . write ( " " + " " + n + " " + getCount ( n ) ) ;
function squareRootExists ( n , p ) { n = n % p ;
for ( let x = 2 ; x < p ; x ++ ) if ( ( x * x ) % p == n ) return true ; return false ; }
let p = 7 ; let n = 2 ; if ( squareRootExists ( n , p ) === true ) document . write ( " " ) ; else document . write ( " " ) ;
function largestPower ( n , p ) {
let x = 0 ;
while ( n ) { n = parseInt ( n / p ) ; x += n ; } return Math . floor ( x ) ; }
let n = 10 ; let p = 3 ; document . write ( " " + p ) ; document . write ( " " + n + " " ) ; document . write ( largestPower ( n , p ) ) ;
function factorial ( n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
var num = 5 ; document . write ( " " + num + " " + factorial ( num ) ) ;
function getBit ( num , i ) {
return ( ( num & ( 1 << i ) ) != 0 ) ; }
function clearBit ( num , i ) {
let mask = ~ ( 1 << i ) ;
return num & mask ; }
function Bitwise_AND_sum_i ( arr1 , arr2 , M , N ) {
let frequency = new Array ( 32 ) . fill ( 0 ) ;
for ( let i = 0 ; i < N ; i ++ ) {
let bit_position = 0 ; let num = arr1 [ i ] ;
while ( num ) {
if ( num & 1 ) {
frequency [ bit_position ] += 1 ; }
bit_position += 1 ;
num >>= 1 ; } }
for ( let i = 0 ; i < M ; i ++ ) { let num = arr2 [ i ] ;
let value_at_that_bit = 1 ;
let bitwise_AND_sum = 0 ;
for ( let bit_position = 0 ; bit_position < 32 ; bit_position ++ ) {
if ( num & 1 ) {
bitwise_AND_sum += frequency [ bit_position ] * value_at_that_bit ; }
num >>= 1 ;
value_at_that_bit <<= 1 ; }
document . write ( bitwise_AND_sum + ' ' ) ; } return ; }
let arr1 = [ 1 , 2 , 3 ] ;
let arr2 = [ 1 , 2 , 3 ] ;
let N = arr1 . length ;
let M = arr2 . length
Bitwise_AND_sum_i ( arr1 , arr2 , M , N ) ;
function FlipBits ( n ) { for ( let bit = 0 ; bit < 32 ; bit ++ ) {
if ( ( ( n >> bit ) & 1 ) > 0 ) {
n = n ^ ( 1 << bit ) ; break ; } } document . write ( " " ) ; document . write ( " " + n ) ; }
let N = 12 ; FlipBits ( N ) ;
function bitwiseAndOdd ( n ) {
var result = 1 ;
for ( var i = 3 ; i <= n ; i = i + 2 ) { result = ( result & i ) ; } return result ; }
var n = 10 ; document . write ( bitwiseAndOdd ( n ) ) ;
function bitwiseAndOdd ( n ) { return 1 ; }
var n = 10 ; document . write ( bitwiseAndOdd ( n ) ) ;
function reverseBits ( n ) { let rev = 0 ;
while ( n > 0 ) {
rev <<= 1 ;
if ( ( n & 1 ) == 1 ) rev ^= 1 ;
n >>= 1 ; }
return rev ; }
let n = 11 ; document . write ( reverseBits ( n ) ) ;
function countgroup ( a , n ) { var xs = 0 ; for ( var i = 0 ; i < n ; i ++ ) xs = xs ^ a [ i ] ;
if ( xs == 0 ) return ( 1 << ( n - 1 ) ) - 1 ; }
var a = [ 1 , 2 , 3 ] ; var n = a . length ; document . write ( countgroup ( a , n ) + " " ) ;
function bitExtracted ( number , k , p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
let number = 171 , k = 5 , p = 2 ; document . write ( " " , bitExtracted ( number , k , p ) ) ;
function findMax ( num ) { let num_copy = num ;
let j = 4 * 8 - 1 ; let i = 0 ; while ( i < j ) {
let m = ( num_copy >> i ) & 1 ; let n = ( num_copy >> j ) & 1 ;
if ( m > n ) { let x = ( 1 << i 1 << j ) ; num = num ^ x ; } i ++ ; j -- ; } return num ; }
let num = 4 ; document . write ( findMax ( num ) ) ;
function isAMultipleOf4 ( n ) {
if ( ( n & 3 ) == 0 ) return true ;
return false ; }
let n = 16 ; document . write ( isAMultipleOf4 ( n ) ? " " : " " ) ;
function square ( n ) {
if ( n < 0 ) n = - n ;
let res = n ;
for ( let i = 1 ; i < n ; i ++ ) res += n ; return res ; }
for ( let n = 1 ; n <= 5 ; n ++ ) document . write ( " " + n + " " + square ( n ) + " " ) ;
function PointInKSquares ( n , a , k ) { a . sort ( ) ; return a [ n - k ] ; }
let k = 2 ; let a = [ 1 , 2 , 3 , 4 ] ; let n = a . length ; let x = PointInKSquares ( n , a , k ) ; document . write ( " " + x + " " + x + " " ) ;
function answer ( n ) {
var dp = Array ( 10 ) ;
var prev = Array ( 10 ) ;
if ( n == 1 ) return 10 ;
for ( var j = 0 ; j <= 9 ; j ++ ) dp [ j ] = 1 ;
for ( var i = 2 ; i <= n ; i ++ ) { for ( var j = 0 ; j <= 9 ; j ++ ) { prev [ j ] = dp [ j ] ; } for ( var j = 0 ; j <= 9 ; j ++ ) {
if ( j == 0 ) dp [ j ] = prev [ j + 1 ] ;
else if ( j == 9 ) dp [ j ] = prev [ j - 1 ] ;
else dp [ j ] = prev [ j - 1 ] + prev [ j + 1 ] ; } }
var sum = 0 ; for ( var j = 1 ; j <= 9 ; j ++ ) sum += dp [ j ] ; return sum ; }
var n = 2 ; document . write ( answer ( n ) ) ;
var MAX = 100000
var catalan = Array ( MAX ) ;
function catalanDP ( n ) {
catalan [ 0 ] = catalan [ 1 ] = 1 ;
for ( var i = 2 ; i <= n ; i ++ ) { catalan [ i ] = 0 ; for ( var j = 0 ; j < i ; j ++ ) catalan [ i ] += catalan [ j ] * catalan [ i - j - 1 ] ; } }
function CatalanSequence ( arr , n ) {
catalanDP ( n ) ; var s = [ ] ;
var a = 1 , b = 1 ; var c ;
s . push ( a ) ; if ( n >= 2 ) s . push ( b ) ; for ( var i = 2 ; i < n ; i ++ ) { s . push ( catalan [ i ] ) ; } s . sort ( ( a , b ) => b - a ) ; for ( var i = 0 ; i < n ; i ++ ) {
if ( s . includes ( arr [ i ] ) ) { s . pop ( arr [ i ] ) ; } }
return s . length ; }
var arr = [ 1 , 1 , 2 , 5 , 41 ] ; var n = arr . length ; document . write ( CatalanSequence ( arr , n ) ) ;
function composite ( n ) { let flag = 0 ; let c = 0 ;
for ( let j = 1 ; j <= n ; j ++ ) { if ( n % j == 0 ) { c += 1 ; } }
if ( c >= 3 ) flag = 1 ; return flag ; }
function odd_indices ( arr , n ) { let sum = 0 ;
for ( let k = 0 ; k < n ; k += 2 ) { let check = composite ( arr [ k ] ) ;
if ( check == 1 ) sum += arr [ k ] ; }
document . write ( sum + " " ) ; }
let arr = [ 13 , 5 , 8 , 16 , 25 ] ; let n = arr . length odd_indices ( arr , n ) ;
function preprocess ( p , x , y , n ) { for ( let i = 0 ; i < n ; i ++ ) p [ i ] = x [ i ] * x [ i ] + y [ i ] * y [ i ] ; p . sort ( function ( a , b ) { return a - b ; } ) ; }
function query ( p , n , rad ) { let start = 0 , end = n - 1 ; while ( ( end - start ) > 1 ) { let mid = Math . floor ( ( start + end ) / 2 ) ; let tp = Math . sqrt ( p [ mid ] ) ; if ( tp > ( rad * 1.0 ) ) end = mid - 1 ; else start = mid ; } let tp1 = Math . sqrt ( p [ start ] ) ; let tp2 = Math . sqrt ( p [ end ] ) ; if ( tp1 > ( rad * 1.0 ) ) return 0 ; else if ( tp2 <= ( rad * 1.0 ) ) return end + 1 ; else return start + 1 ; }
let x = [ 1 , 2 , 3 , - 1 , 4 ] ; let y = [ 1 , 2 , 3 , - 1 , 4 ] ; let n = x . length ;
let p = new Array ( n ) ; for ( let i = 0 ; i < n ; i ++ ) { p [ i ] = 0 ; } preprocess ( p , x , y , n ) ;
document . write ( query ( p , n , 3 ) + " " ) ;
document . write ( query ( p , n , 32 ) + " " ) ;
function find_Numb_ways ( n ) {
var odd_indices = n / 2 ;
var even_indices = ( n / 2 ) + ( n % 2 ) ;
var arr_odd = Math . pow ( 4 , odd_indices ) ;
var arr_even = Math . pow ( 5 , even_indices ) ;
return arr_odd * arr_even ; }
var n = 4 ; document . write ( find_Numb_ways ( n ) ) ;
function isSpiralSorted ( arr , n ) {
let start = 0 ;
let end = n - 1 ; while ( start < end ) {
if ( arr [ start ] > arr [ end ] ) { return false ; }
start ++ ;
if ( arr [ end ] > arr [ start ] ) { return false ; }
end -- ; } return true ; }
let arr = [ 1 , 10 , 14 , 20 , 18 , 12 , 5 ] ; let N = arr . length ;
if ( isSpiralSorted ( arr , N ) ) document . write ( " " ) ; else document . write ( " " ) ;
function findWordsSameRow ( arr ) {
var mp = { } ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 1 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 2 ; mp [ " " ] = 3 ; mp [ " " ] = 3 ; mp [ " " ] = 3 ; mp [ " " ] = 3 ; mp [ " " ] = 3 ; mp [ " " ] = 3 ; mp [ " " ] = 3 ;
for ( const word of arr ) {
if ( word . length !== 0 ) {
var flag = true ;
var rowNum = mp [ word [ 0 ] . toLowerCase ( ) ] ;
var M = word . length ;
for ( var i = 1 ; i < M ; i ++ ) {
if ( mp [ word [ i ] . toLowerCase ( ) ] !== rowNum ) {
flag = false ; break ; } }
if ( flag ) {
document . write ( word + " " ) ; } } } }
var words = [ " " , " " , " " , " " ] ; findWordsSameRow ( words ) ;
function countSubsequece ( a , n ) { let i , j , k , l ;
let answer = 0 ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { for ( k = j + 1 ; k < n ; k ++ ) { for ( l = k + 1 ; l < n ; l ++ ) {
if ( a [ j ] == a [ l ] &&
a [ i ] == a [ k ] ) { answer ++ ; } } } } } return answer ; }
let a = [ 1 , 2 , 3 , 2 , 1 , 3 , 2 ] ; document . write ( countSubsequece ( a , 7 ) ) ;
function minDistChar ( s ) { let n = s . length ;
let first = new Array ( 26 ) ; let last = new Array ( 26 ) ;
for ( let i = 0 ; i < 26 ; i ++ ) { first [ i ] = - 1 ; last [ i ] = - 1 ; }
for ( let i = 0 ; i < n ; i ++ ) {
if ( first [ s [ i ] - ' ' ] == - 1 ) { first [ s [ i ] - ' ' ] = i ; }
last [ s [ i ] - ' ' ] = i ; }
let min = 100000 ; var ans = ' ' ;
for ( let i = 0 ; i < 26 ; i ++ ) {
if ( last [ i ] == first [ i ] ) continue ;
if ( min > last [ i ] - first [ i ] ) { min = last [ i ] - first [ i ] ; ans = String . fromCharCode ( i + 97 ) ; } }
return ans ; }
str = " " ;
document . write ( minDistChar ( str ) ) ;
var n = 3 ;
function minSteps ( arr ) {
var v = Array . from ( Array ( n ) , ( ) => Array ( n ) . fill ( 0 ) ) ;
var q = [ ] ;
q . push ( [ 0 , 0 ] ) ;
var depth = 0 ;
while ( q . length != 0 ) {
var x = q . length ; while ( x -- ) {
var y = q [ 0 ] ;
var i = y [ 0 ] , j = y [ 1 ] ; q . shift ( ) ;
if ( v [ i ] [ j ] ) continue ;
if ( i == n - 1 && j == n - 1 ) return depth ;
v [ i ] [ j ] = 1 ;
if ( i + arr [ i ] [ j ] < n ) q . push ( [ i + arr [ i ] [ j ] , j ] ) ; if ( j + arr [ i ] [ j ] < n ) q . push ( [ i , j + arr [ i ] [ j ] ] ) ; } depth ++ ; } return - 1 ; }
var arr = [ [ 1 , 1 , 1 ] , [ 1 , 1 , 1 ] , [ 1 , 1 , 1 ] ] ; document . write ( minSteps ( arr ) ) ;
function solve ( a , n ) { let max1 = Number . MIN_VALUE ; for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) { if ( Math . abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
let arr = [ - 1 , 2 , 3 , - 4 , - 10 , 22 ] ; let size = arr . length ; document . write ( " " + solve ( arr , size ) ) ;
function solve ( a , n ) { let min1 = a [ 0 ] ; let max1 = a [ 0 ] ;
for ( let i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . abs ( min1 - max1 ) ; }
let arr = [ - 1 , 2 , 3 , 4 , - 10 ] ; let size = arr . length ; document . write ( " " + solve ( arr , size ) ) ;
function replaceOriginal ( s , n ) {
var r = new Array ( n ) ;
for ( var i = 0 ; i < n ; i ++ ) {
r [ i ] = s . charAt ( n - 1 - i ) ;
if ( s . charAt ( i ) != ' ' && s . charAt ( i ) != ' ' && s . charAt ( i ) != ' ' && s . charAt ( i ) != ' ' && s . charAt ( i ) != ' ' ) { document . write ( r [ i ] ) ; } } document . write ( " " ) ; }
var s = " " ; var n = s . length ; replaceOriginal ( s , n ) ;
function sameStrings ( str1 , str2 ) { var N = str1 . length ; var M = str2 . length ;
if ( N !== M ) { return false ; }
var a = new Array ( 256 ) . fill ( 0 ) ; var b = new Array ( 256 ) . fill ( 0 ) ;
for ( var j = 0 ; j < N ; j ++ ) { a [ str1 [ j ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] ++ ; b [ str2 [ j ] . charCodeAt ( 0 ) - " " . charCodeAt ( 0 ) ] ++ ; }
var i = 0 ; while ( i < 256 ) { if ( ( a [ i ] === 0 && b [ i ] === 0 ) || ( a [ i ] !== 0 && b [ i ] !== 0 ) ) { i ++ ; }
else { return false ; } }
a . sort ( ( x , y ) => x - y ) ; b . sort ( ( x , y ) => x - y ) ;
for ( var j = 0 ; j < 256 ; j ++ ) {
if ( a [ j ] !== b [ j ] ) return false ; }
return true ; }
var S1 = " " , S2 = " " ; if ( sameStrings ( S1 , S2 ) ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function solution ( A , B , C ) { let arr = Array . from ( { length : 3 } , ( _ , i ) => 0 ) ;
arr [ 0 ] = A ; arr [ 1 ] = B ; arr [ 2 ] = C ;
arr . sort ( ) ;
if ( arr [ 2 ] < arr [ 0 ] + arr [ 1 ] ) return ( ( arr [ 0 ] + arr [ 1 ] + arr [ 2 ] ) / 2 ) ;
else return ( arr [ 0 ] + arr [ 1 ] ) ; }
let A = 8 , B = 1 , C = 5 ;
document . write ( solution ( A , B , C ) ) ;
function search ( arr , l , h , key ) { if ( l > h ) return - 1 ; let mid = parseInt ( ( l + h ) / 2 , 10 ) ; if ( arr [ mid ] == key ) return mid ;
if ( ( arr [ l ] == arr [ mid ] ) && ( arr [ h ] == arr [ mid ] ) ) { ++ l ; -- h ; return search ( arr , l , h , key ) }
if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
return search ( arr , mid + 1 , h , key ) ; }
if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
let arr = [ 3 , 3 , 1 , 2 , 3 , 3 ] ; let n = arr . length ; let key = 3 ; document . write ( search ( arr , 0 , n - 1 , key ) ) ;
function getSortedString ( s , n ) {
var v1 = [ ] ; var v2 = [ ] ; var i = 0 ; for ( i = 0 ; i < n ; i ++ ) { if ( s [ i ] . charCodeAt ( 0 ) > " " . charCodeAt ( 0 ) && s [ i ] . charCodeAt ( 0 ) <= " " . charCodeAt ( 0 ) ) v1 . push ( s [ i ] ) ; if ( s [ i ] . charCodeAt ( 0 ) > " " . charCodeAt ( 0 ) && s [ i ] . charCodeAt ( 0 ) <= " " . charCodeAt ( 0 ) ) v2 . push ( s [ i ] ) ; }
console . log ( v1 ) ; v1 . sort ( ) ; v2 . sort ( ) ; var j = 0 ; i = 0 ; for ( var k = 0 ; k < n ; k ++ ) {
if ( s [ k ] . charCodeAt ( 0 ) > " " . charCodeAt ( 0 ) && s [ k ] . charCodeAt ( 0 ) <= " " . charCodeAt ( 0 ) ) { s [ k ] = v1 [ i ] ; ++ i ; }
else if ( s [ k ] . charCodeAt ( 0 ) > " " . charCodeAt ( 0 ) && s [ k ] . charCodeAt ( 0 ) <= " " . charCodeAt ( 0 ) ) { s [ k ] = v2 [ j ] ; ++ j ; } }
return s . join ( " " ) ; }
var s = " " ; var n = s . length ; document . write ( getSortedString ( s . split ( " " ) , n ) ) ;
function check ( s ) {
let l = s . length ;
s . sort ( ) ;
for ( let i = 1 ; i < l ; i ++ ) {
if ( ( s [ i ] . charCodeAt ( ) - s [ i - 1 ] . charCodeAt ( ) ) != 1 ) return false ; } return true ; }
let str = " " ; if ( check ( str . split ( ' ' ) ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
let str1 = " " ; if ( check ( str1 . split ( ' ' ) ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
function minElements ( arr , n ) {
let halfSum = 0 ; for ( let i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = parseInt ( halfSum / 2 , 10 ) ;
arr . sort ( function ( a , b ) { return a - b } ) ; arr . reverse ( ) ; let res = 0 , curr_sum = 0 ; for ( let i = 0 ; i < n ; i ++ ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
let arr = [ 3 , 1 , 7 , 1 ] ; let n = arr . length ; document . write ( minElements ( arr , n ) ) ;
function arrayElementEqual ( arr , N ) {
var sum = 0 ;
for ( i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
if ( sum % N == 0 ) { document . write ( " " ) ; }
else { document . write ( " " + " " ) ; } }
var arr = [ 1 , 5 , 6 , 4 ] ;
var N = arr . length ; arrayElementEqual ( arr , N ) ;
function findMaxValByRearrArr ( arr , N ) {
let res = 0 ;
res = parseInt ( ( N * ( N + 1 ) ) / 2 , 10 ) ; return res ; }
let arr = [ 3 , 2 , 1 ] ; let N = arr . length ; document . write ( findMaxValByRearrArr ( arr , N ) ) ;
function MaximumSides ( n ) {
if ( n < 4 ) return - 1 ;
return n % 2 == 0 ? n / 2 : - 1 ; }
let N = 8 ;
document . write ( MaximumSides ( N ) ) ;
function pairProductMean ( arr , N ) {
var suffixSumArray = Array ( N ) ; suffixSumArray [ N - 1 ] = arr [ N - 1 ] ;
for ( var i = N - 2 ; i >= 0 ; i -- ) { suffixSumArray [ i ] = suffixSumArray [ i + 1 ] + arr [ i ] ; }
var length = ( N * ( N - 1 ) ) / 2 ;
var res = 0 ; for ( var i = 0 ; i < N - 1 ; i ++ ) { res += arr [ i ] * suffixSumArray [ i + 1 ] ; }
var mean ;
if ( length != 0 ) mean = res / length ; else mean = 0 ;
return mean ; }
var arr = [ 1 , 2 , 4 , 8 ] ; var N = arr . length ;
document . write ( pairProductMean ( arr , N ) . toFixed ( 2 ) ) ;
function ncr ( n , k ) { var res = 1 ;
if ( k > n - k ) k = n - k ;
for ( var i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
function countPath ( N , M , K ) { var answer ; if ( K >= 2 ) answer = 0 ; else if ( K == 0 ) answer = ncr ( N + M - 2 , N - 1 ) ; else {
answer = ncr ( N + M - 2 , N - 1 ) ;
var X = ( N - 1 ) / 2 + ( M - 1 ) / 2 ; var Y = ( N - 1 ) / 2 ; var midCount = ncr ( X , Y ) ;
X = ( ( N - 1 ) - ( N - 1 ) / 2 ) + ( ( M - 1 ) - ( M - 1 ) / 2 ) ; Y = ( ( N - 1 ) - ( N - 1 ) / 2 ) ; midCount *= ncr ( X , Y ) ; answer -= midCount ; } return answer ; }
var N = 3 ; var M = 3 ; var K = 1 ; document . write ( countPath ( N , M , K ) ) ;
function find_max ( v , n ) {
let count = 0 ; if ( n >= 2 ) count = 2 ; else count = 1 ;
for ( let i = 1 ; i < n - 1 ; i ++ ) {
if ( v [ i - 1 ] [ 0 ] < ( v [ i ] [ 0 ] - v [ i ] [ 1 ] ) ) count ++ ;
else if ( v [ i + 1 ] [ 0 ] > ( v [ i ] [ 0 ] + v [ i ] [ 1 ] ) ) { count ++ ; v [ i ] [ 0 ] = v [ i ] [ 0 ] + v [ i ] [ 1 ] ; }
else continue ; }
return count ; }
let n = 3 ; let v = [ [ 10 , 20 ] , [ 15 , 10 ] , [ 20 , 16 ] ] ; document . write ( find_max ( v , n ) ) ;
function numberofsubstrings ( str , k , charArray ) { var N = str . length ;
var available = [ 26 ] ;
for ( var i = 0 ; i < k ; i ++ ) { available [ charArray [ i ] - ' ' ] = 1 ; }
var lastPos = - 1 ;
var ans = ( N * ( N + 1 ) ) / 2 ;
for ( var i = 0 ; i < N ; i ++ ) {
if ( available [ str . charAt ( i ) - ' ' ] == 0 ) {
ans -= ( ( i - lastPos ) * ( N - i ) ) ;
lastPos = i ; } }
document . write ( ans ) ; }
var str = " " ; var k = 2 ;
var charArray = [ ' ' , ' ' ] ;
numberofsubstrings ( str , k , charArray ) ;
function minCost ( N , P , Q ) {
var cost = 0 ;
while ( N > 0 ) { if ( N & 1 ) { cost += P ; N -- ; } else { var temp = parseInt ( N / 2 ) ;
if ( temp * P > Q ) cost += Q ;
else cost += P * temp ; N = parseInt ( N / 2 ) ; } }
return cost ; }
var N = 9 , P = 5 , Q = 1 ; document . write ( minCost ( N , P , Q ) ) ;
function numberOfWays ( n , k ) {
let dp = Array ( 1000 ) ;
for ( let i = 0 ; i < n ; i ++ ) { dp [ i ] = 0 ; }
dp [ 0 ] = 1 ;
for ( let i = 1 ; i <= k ; i ++ ) {
let numWays = 0 ;
for ( let j = 0 ; j < n ; j ++ ) { numWays += dp [ j ] ; }
for ( let j = 0 ; j < n ; j ++ ) { dp [ j ] = numWays - dp [ j ] ; } }
document . write ( dp [ 0 ] ) ; }
let N = 5 ; let K = 3 ;
numberOfWays ( N , K ) ;
let M = 1000000007 ; function waysOfDecoding ( s ) { let first = 1 , second = s [ 0 ] == ' ' ? 9 : s [ 0 ] == ' ' ? 0 : 1 ; for ( let i = 1 ; i < s . length ; i ++ ) { let temp = second ;
if ( s [ i ] == ' ' ) { second = 9 * second ;
if ( s [ i - 1 ] == ' ' ) second = ( second + 9 * first ) % M ;
else if ( s [ i - 1 ] == ' ' ) second = ( second + 6 * first ) % M ;
else if ( s [ i - 1 ] == ' ' ) second = ( second + 15 * first ) % M ; }
else { second = s [ i ] != ' ' ? second : 0 ;
if ( s [ i - 1 ] == ' ' ) second = ( second + first ) % M ;
else if ( s [ i - 1 ] == ' ' && s [ i ] <= ' ' ) second = ( second + first ) % M ;
else if ( s [ i - 1 ] == ' ' ) second = ( second + ( s [ i ] <= ' ' ? 2 : 1 ) * first ) % M ; } first = temp ; } return second ; }
let s = " " ; document . write ( waysOfDecoding ( s ) ) ;
function findMinCost ( arr , X , n , i = 0 ) {
if ( X <= 0 ) return 0 ; if ( i >= n ) return Number . MAX_SAFE_INTEGER ;
let inc = findMinCost ( arr , X - arr [ i ] [ 0 ] , n , i + 1 ) ; if ( inc != Number . MAX_SAFE_INTEGER ) inc += arr [ i ] [ 1 ] ;
let exc = findMinCost ( arr , X , n , i + 1 ) ;
return Math . min ( inc , exc ) ; }
let arr = [ [ 4 , 3 ] , [ 3 , 2 ] , [ 2 , 4 ] , [ 1 , 3 ] , [ 4 , 2 ] ] ; let X = 7 ;
let n = arr . length ; let ans = findMinCost ( arr , X , n ) ;
if ( ans != Number . MAX_SAFE_INTEGER ) document . write ( ans ) else document . write ( - 1 )
function find ( N , sum ) {
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return 1.0 / 6 ; else return 0 ; } let s = 0 ; for ( let i = 1 ; i <= 6 ; i ++ ) s = s + find ( N - 1 , sum - i ) / 6 ; return s ; }
let N = 4 , a = 13 , b = 17 ; let probability = 0.0 ; for ( let sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
document . write ( probability . toFixed ( 6 ) ) ;
function minDays ( n ) {
if ( n < 1 ) return n ;
var cnt = 1 + Math . min ( n % 2 + minDays ( parseInt ( n / 2 ) ) , n % 3 + minDays ( parseInt ( n / 3 ) ) ) ;
return cnt ; }
var N = 6 ;
document . write ( minDays ( N ) ) ;
