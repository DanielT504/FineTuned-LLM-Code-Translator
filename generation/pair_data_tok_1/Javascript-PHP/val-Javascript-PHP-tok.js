function Conversion ( centi ) { let pixels = ( 96 * centi ) / 2.54 ; document . write ( pixels ) ; return 0 ; }
let centi = 15 ; Conversion ( centi )
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
function findNumbers ( n , w ) { let x = 0 , sum = 0 ;
if ( w >= 0 && w <= 8 ) {
x = 9 - w ; }
else if ( w >= - 9 && w <= - 1 ) {
x = 10 + w ; } sum = Math . pow ( 10 , n - 2 ) ; sum = ( x * sum ) ; return sum ; }
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
function printKDistinct ( arr , n , k ) { var dist_count = 0 ; for ( var i = 0 ; i < n ; i ++ ) {
var j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return - 1 ; }
var ar = [ 1 , 2 , 1 , 3 , 4 , 2 ] ; var n = ar . length ; var k = 2 ; document . write ( printKDistinct ( ar , n , k ) ) ;
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
function print2Smallest ( arr , arr_size ) { let i , first , second ;
if ( arr_size < 2 ) { document . write ( " " ) ; return ; } first = Number . MAX_VALUE ; second = Number . MAX_VALUE ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == Number . MAX_VALUE ) document . write ( " " ) ; else document . write ( " " + first + " " + " " + second + ' ' ) ; }
let arr = [ 12 , 13 , 1 , 10 , 34 , 1 ] ; let n = arr . length ; print2Smallest ( arr , n ) ;
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
function isSubsetSum ( set , n , sum ) {
let subset = new Array ( sum + 1 ) ; for ( let i = 0 ; i < sum + 1 ; i ++ ) { subset [ i ] = new Array ( sum + 1 ) ; for ( let j = 0 ; j < n + 1 ; j ++ ) { subset [ i ] [ j ] = 0 ; } }
for ( let i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( let i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( let i = 1 ; i <= sum ; i ++ ) { for ( let j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
for ( int i = 0 ; i <= sum ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) System . out . println ( subset [ i ] [ j ] ) ; } return subset [ sum ] [ n ] ; }
let set = [ 3 , 34 , 4 , 12 , 5 , 2 ] ; let sum = 9 ; let n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
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
function isPowerOfTwo ( x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
document . write ( isPowerOfTwo ( 31 ) ? " " : " " ) ; document . write ( " " + ( isPowerOfTwo ( 64 ) ? " " : " " ) ) ;
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
function printTetra ( n ) { if ( n < 0 ) return ;
var first = 0 , second = 1 ; var third = 1 , fourth = 2 ;
var curr ; if ( n == 0 ) cout << first ; else if ( n == 1 n == 2 ) cout << second ; else if ( n == 3 ) cout << fourth ; else {
for ( var i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } document . write ( curr ) ; } }
var n = 10 ; printTetra ( n ) ;
function countWays ( n ) { let res = new Array ( n + 2 ) ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( let i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
let n = 4 ; document . write ( countWays ( n ) ) ;
function maxTasks ( high , low , n ) {
if ( n <= 0 ) return 0 ;
return Math . max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
let n = 5 ; let high = [ 3 , 6 , 8 , 7 , 6 ] ; let low = [ 1 , 5 , 4 , 5 , 3 ] ; document . write ( maxTasks ( high , low , n ) ) ; ;
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
function printTetraRec ( n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
function printTetra ( n ) { document . write ( printTetraRec ( n ) + " " + " " ) ; }
let n = 10 ; printTetra ( n ) ;
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
function minOperations ( str , n ) {
var i , lastUpper = - 1 , firstLower = - 1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( isupper ( str [ i ] ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( islower ( str [ i ] ) ) { firstLower = i ; break ; } }
if ( lastUpper === - 1 firstLower === - 1 ) return 0 ;
var countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( isupper ( str [ i ] ) ) { countUpper ++ ; } }
var countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( islower ( str [ i ] ) ) { countLower ++ ; } }
return Math . min ( countLower , countUpper ) ; }
var str = " " ; var n = str . length ; document . write ( minOperations ( str , n ) + " " ) ;
function rainDayProbability ( a , n ) { let count = 0 , m ;
for ( let i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
let a = [ 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 ] ; let n = a . length ; document . write ( rainDayProbability ( a , n ) ) ;
function Series ( n ) { let sums = 0.0 ; for ( let i = 1 ; i < n + 1 ; i ++ ) { ser = 1 / Math . pow ( i , i ) ; sums += ser ; } return sums ; }
let n = 3 ; let res = Math . round ( Series ( n ) * 100000 ) / 100000 ; document . write ( res ) ;
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
function checkHV ( arr , N , M ) {
let horizontal = true ; let vertical = true ;
for ( let i = 0 , k = N - 1 ; i < parseInt ( N / 2 , 10 ) ; i ++ , k -- ) {
for ( let j = 0 ; j < M ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } }
for ( let i = 0 , k = M - 1 ; i < parseInt ( M / 2 , 10 ) ; i ++ , k -- ) {
for ( let j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } } if ( ! horizontal && ! vertical ) document . write ( " " ) ; else if ( horizontal && ! vertical ) document . write ( " " ) ; else if ( vertical && ! horizontal ) document . write ( " " ) ; else document . write ( " " ) ; }
let mat = [ [ 1 , 0 , 1 ] , [ 0 , 0 , 0 ] , [ 1 , 0 , 1 ] ] ; checkHV ( mat , 3 , 3 ) ;
let N = 4 ;
function add ( A , B , C ) { let i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] ; }
let A = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] ; let B = [ [ 1 , 1 , 1 , 1 ] , [ 2 , 2 , 2 , 2 ] , [ 3 , 3 , 3 , 3 ] , [ 4 , 4 , 4 , 4 ] ] ; let C = new Array ( N ) ; for ( let k = 0 ; k < N ; k ++ ) C [ k ] = new Array ( N ) ; let i , j ; add ( A , B , C ) ; document . write ( " " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) document . write ( C [ i ] [ j ] + " " ) ; document . write ( " " ) ; }
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
function sort ( arr ) { var n = arr . length ;
var output = Array . from ( { length : n } , ( _ , i ) => 0 ) ;
var count = Array . from ( { length : 256 } , ( _ , i ) => 0 ) ;
for ( var i = 0 ; i < n ; ++ i ) ++ count [ arr [ i ] . charCodeAt ( 0 ) ] ;
for ( var i = 1 ; i <= 255 ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( var i = n - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] . charCodeAt ( 0 ) ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] . charCodeAt ( 0 ) ] ; }
for ( var i = 0 ; i < n ; ++ i ) arr [ i ] = output [ i ] ; return arr ; }
var arr = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ] ; arr = sort ( arr ) ; document . write ( " " ) ; for ( var i = 0 ; i < arr . length ; ++ i ) document . write ( arr [ i ] ) ; cript
function binomialCoeff ( n , k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
var n = 5 , k = 2 ; document . write ( " " + n + " " + k + " " + binomialCoeff ( n , k ) ) ;
function binomialCoeff ( n , k ) { let C = new Array ( k + 1 ) ; C . fill ( 0 ) ;
C [ 0 ] = 1 ; for ( let i = 1 ; i <= n ; i ++ ) {
for ( let j = Math . min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
let n = 5 , k = 2 ; document . write ( " " + n + " " + k + " " + binomialCoeff ( n , k ) ) ;
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
for ( int i = 0 ; i <= sum ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) System . out . println ( subset [ i ] [ j ] ) ; } return subset [ sum ] [ n ] ; }
let set = [ 3 , 34 , 4 , 12 , 5 , 2 ] ; let sum = 9 ; let n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) document . write ( " " + " " ) ; else document . write ( " " + " " ) ;
function findoptimal ( N ) {
if ( N <= 6 ) return N ;
let max = 0 ;
let b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
let curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
let N ;
for ( N = 1 ; N <= 20 ; N ++ ) document . write ( " " + N + " " + findoptimal ( N ) + " " ) ;
function power ( x , y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , parseInt ( y / 2 , 10 ) ) * power ( x , parseInt ( y / 2 , 10 ) ) ; else return x * power ( x , parseInt ( y / 2 , 10 ) ) * power ( x , parseInt ( y / 2 , 10 ) ) ; }
let x = 2 ; let y = 3 ; document . write ( power ( x , y ) ) ;
function power ( x , y ) { var temp ; if ( y == 0 ) return 1 ; temp = power ( x , parseInt ( y / 2 ) ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
var x = 2 ; var y = - 3 ; document . write ( power ( x , y ) . toFixed ( 6 ) ) ;
function squareRoot ( n ) {
let x = n ; let y = 1 ; let e = 0.000001 ;
while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
let n = 50 ; document . write ( " " + n + " " + squareRoot ( n ) . toFixed ( 6 ) ) ;
function getAvg ( prev_avg , x , n ) { return ( prev_avg * n + x ) / ( n + 1 ) ; }
function streamAvg ( arr , n ) { let avg = 0 ; for ( let i = 0 ; i < n ; i ++ ) { avg = getAvg ( avg , arr [ i ] , i ) ; document . write ( " " + ( i + 1 ) + " " + avg . toFixed ( 6 ) + " " ) ; } return ; }
let arr = [ 10 , 20 , 30 , 40 , 50 , 60 ] ; let n = arr . length ; streamAvg ( arr , n ) ;
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
function maxIndexDiff ( arr , n ) { let maxDiff = - 1 ; let i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
let arr = [ 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ] ; let n = arr . length ; let maxDiff = maxIndexDiff ( arr , n ) ; document . write ( maxDiff ) ;
function missingK ( a , k , n ) { let difference = 0 , ans = 0 , count = k ; let flag = false ;
for ( let i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = true ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return - 1 ; }
let a = [ 1 , 5 , 11 , 19 ] ;
let k = 11 ; let n = a . length ;
let missing = missingK ( a , k , n ) ; document . write ( missing ) ;
function findKth ( arr , n , k ) { var missing = new Set ( ) ; var count = 0 ;
for ( var i = 0 ; i < n ; i ++ ) missing . add ( arr [ i ] ) ;
var maxm = arr . reduce ( ( a , b ) => Math . max ( a , b ) ) ; var minm = arr . reduce ( ( a , b ) => Math . min ( a , b ) ) ;
for ( var i = minm + 1 ; i < maxm ; i ++ ) {
if ( ! missing . has ( i ) ) count ++ ;
if ( count == k ) return i ; }
return - 1 ; }
var arr = [ 2 , 10 , 9 , 4 ] ; var n = arr . length ; var k = 5 ; document . write ( findKth ( arr , n , k ) ) ;
function waysToKAdjacentSetBits ( n , k , currentIndex , adjacentSetBits , lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } let noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( ! lastBit ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
let n = 5 , k = 2 ;
let totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; document . write ( " " + totalWays ) ;
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
function findRepeatFirstN2 ( s ) {
let p = - 1 , i , j ; for ( i = 0 ; i < s . length ; i ++ ) { for ( j = i + 1 ; j < s . length ; j ++ ) { if ( s [ i ] == s [ j ] ) { p = i ; break ; } } if ( p != - 1 ) break ; } return p ; }
let str = " " ; let pos = findRepeatFirstN2 ( str ) ; if ( pos == - 1 ) document . write ( " " ) ; else document . write ( str [ pos ] ) ;
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
function isPalindrome ( num ) { let reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp = Math . floor ( temp / 10 ) ; }
if ( reverse_num == num ) { return true ; } return false ; }
function isOddLength ( num ) { let count = 0 ; while ( num > 0 ) { num = Math . floor ( num / 10 ) ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
function sumOfAllPalindrome ( L , R ) { let sum = 0 ; if ( L <= R ) for ( let i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
let L = 110 , R = 1130 ; document . write ( sumOfAllPalindrome ( L , R ) ) ;
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
function solve ( a , n ) { let max1 = Number . MIN_VALUE ; for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) { if ( Math . abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
let arr = [ - 1 , 2 , 3 , - 4 , - 10 , 22 ] ; let size = arr . length ; document . write ( " " + solve ( arr , size ) ) ;
function solve ( a , n ) { let min1 = a [ 0 ] ; let max1 = a [ 0 ] ;
for ( let i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . abs ( min1 - max1 ) ; }
let arr = [ - 1 , 2 , 3 , 4 , - 10 ] ; let size = arr . length ; document . write ( " " + solve ( arr , size ) ) ;
function minElements ( arr , n ) {
let halfSum = 0 ; for ( let i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = parseInt ( halfSum / 2 , 10 ) ;
arr . sort ( function ( a , b ) { return a - b } ) ; arr . reverse ( ) ; let res = 0 , curr_sum = 0 ; for ( let i = 0 ; i < n ; i ++ ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
let arr = [ 3 , 1 , 7 , 1 ] ; let n = arr . length ; document . write ( minElements ( arr , n ) ) ;
function minCost ( N , P , Q ) {
var cost = 0 ;
while ( N > 0 ) { if ( N & 1 ) { cost += P ; N -- ; } else { var temp = parseInt ( N / 2 ) ;
if ( temp * P > Q ) cost += Q ;
else cost += P * temp ; N = parseInt ( N / 2 ) ; } }
return cost ; }
var N = 9 , P = 5 , Q = 1 ; document . write ( minCost ( N , P , Q ) ) ;
