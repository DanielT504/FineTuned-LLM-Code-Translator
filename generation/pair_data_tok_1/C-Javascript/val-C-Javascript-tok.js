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
function pell ( n ) { if ( n <= 2 ) return n ; let a = 1 ; let b = 2 ; let c ; for ( let i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
let n = 4 ; document . write ( pell ( n ) ) ;
function factorial ( n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
let num = 5 ; document . write ( " " + num + " " + factorial ( num ) ) ;
function findSubArray ( arr , n ) { let sum = 0 ; let maxsize = - 1 , startindex = 0 ; let endindex = 0 ;
for ( let i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? - 1 : 1 ;
for ( let j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] == 0 ) sum += - 1 ; else sum += 1 ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } endindex = startindex + maxsize - 1 ; if ( maxsize == - 1 ) document . write ( " " ) ; else document . write ( startindex + " " + endindex ) ; return maxsize ; }
let arr = [ 1 , 0 , 0 , 1 , 0 , 1 , 1 ] ; let size = arr . length ; findSubArray ( arr , size ) ;
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
function isPowerOfTwo ( x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
document . write ( isPowerOfTwo ( 31 ) ? " " : " " ) ; document . write ( " " + ( isPowerOfTwo ( 64 ) ? " " : " " ) ) ;
function nextPowerOf2 ( n ) { var count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
var n = 0 ; document . write ( nextPowerOf2 ( n ) ) ;
function countWays ( n ) { let res = new Array ( n + 2 ) ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( let i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
let n = 4 ; document . write ( countWays ( n ) ) ;
function maxTasks ( high , low , n ) {
if ( n <= 0 ) return 0 ;
return Math . max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
let n = 5 ; let high = [ 3 , 6 , 8 , 7 , 6 ] ; let low = [ 1 , 5 , 4 , 5 , 3 ] ; document . write ( maxTasks ( high , low , n ) ) ; ;
var OUT = 0 ; var IN = 1 ;
function countWords ( str ) { var state = OUT ;
var wc = 0 ; var i = 0 ;
while ( i < str . length ) {
if ( str [ i ] == ' ' str [ i ] == ' ' str [ i ] == ' ' ) state = OUT ;
else if ( state == OUT ) { state = IN ; ++ wc ; }
++ i ; } return wc ; }
var str = " " ; document . write ( " " + countWords ( str ) ) ;
function max ( x , y ) { return ( x > y ? x : y ) ; }
function maxTasks ( high , low , n ) {
var task_dp = Array . from ( { length : n + 1 } , ( _ , i ) => 0 ) ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( i = 2 ; i <= n ; i ++ ) task_dp [ i ] = Math . max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
var n = 5 ; var high = [ 3 , 6 , 8 , 7 , 6 ] ; var low = [ 1 , 5 , 4 , 5 , 3 ] ; document . write ( maxTasks ( high , low , n ) ) ;
function findPartition ( arr , n ) { var sum = 0 ; var i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; var part = Array ( parseInt ( sum / 2 ) + 1 ) . fill ( ) . map ( ( ) => Array ( n + 1 ) . fill ( 0 ) ) ;
for ( i = 0 ; i <= n ; i ++ ) part [ 0 ] [ i ] = true ;
for ( i = 1 ; i <= parseInt ( sum / 2 ) ; i ++ ) part [ i ] [ 0 ] = false ;
for ( i = 1 ; i <= parseInt ( sum / 2 ) ; i ++ ) { for ( j = 1 ; j <= n ; j ++ ) { part [ i ] [ j ] = part [ i ] [ j - 1 ] ; if ( i >= arr [ j - 1 ] ) part [ i ] [ j ] = part [ i ] [ j ] || part [ i - arr [ j - 1 ] ] [ j - 1 ] ; } }
return part [ parseInt ( sum / 2 ) ] [ n ] ; }
var arr = [ 3 , 1 , 1 , 2 , 2 , 1 ] ; var n = arr . length ;
if ( findPartition ( arr , n ) == true ) document . write ( " " ) ; else document . write ( " " ) ;
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
function power ( x , y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , parseInt ( y / 2 , 10 ) ) * power ( x , parseInt ( y / 2 , 10 ) ) ; else return x * power ( x , parseInt ( y / 2 , 10 ) ) * power ( x , parseInt ( y / 2 , 10 ) ) ; }
let x = 2 ; let y = 3 ; document . write ( power ( x , y ) ) ;
function power ( x , y ) { var temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
function power ( x , y ) { var temp ; if ( y == 0 ) return 1 ; temp = power ( x , parseInt ( y / 2 ) ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
var x = 2 ; var y = - 3 ; document . write ( power ( x , y ) . toFixed ( 6 ) ) ;
function squareRoot ( n ) {
let x = n ; let y = 1 ; let e = 0.000001 ;
while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
let n = 50 ; document . write ( " " + n + " " + squareRoot ( n ) . toFixed ( 6 ) ) ;
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
function segregate0and1 ( arr , size ) {
let left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
let arr = [ 0 , 1 , 0 , 1 , 1 , 1 ] ; let i , arr_size = arr . length ; segregate0and1 ( arr , arr_size ) ; document . write ( " " ) ; for ( i = 0 ; i < 6 ; i ++ ) document . write ( arr [ i ] + " " ) ;
function maxIndexDiff ( arr , n ) { let maxDiff = - 1 ; let i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
let arr = [ 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 ] ; let n = arr . length ; let maxDiff = maxIndexDiff ( arr , n ) ; document . write ( maxDiff ) ;
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
function isPalindrome ( num ) { let reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp = Math . floor ( temp / 10 ) ; }
if ( reverse_num == num ) { return true ; } return false ; }
function isOddLength ( num ) { let count = 0 ; while ( num > 0 ) { num = Math . floor ( num / 10 ) ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
function sumOfAllPalindrome ( L , R ) { let sum = 0 ; if ( L <= R ) for ( let i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
let L = 110 , R = 1130 ; document . write ( sumOfAllPalindrome ( L , R ) ) ;
function subtractOne ( x ) { let m = 1 ;
while ( ! ( x & m ) ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
document . write ( subtractOne ( 13 ) ) ;
function findSum ( n , a , b ) { let sum = 0 ; for ( let i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
let n = 10 ; let a = 3 ; let b = 5 ; document . write ( findSum ( n , a , b ) ) ;
function pell ( n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
let n = 4 ; document . write ( pell ( n ) ) ;
function largestPower ( n , p ) {
let x = 0 ;
while ( n ) { n = parseInt ( n / p ) ; x += n ; } return Math . floor ( x ) ; }
let n = 10 ; let p = 3 ; document . write ( " " + p ) ; document . write ( " " + n + " " ) ; document . write ( largestPower ( n , p ) ) ;
function factorial ( n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
var num = 5 ; document . write ( " " + num + " " + factorial ( num ) ) ;
function bitExtracted ( number , k , p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
let number = 171 , k = 5 , p = 2 ; document . write ( " " , bitExtracted ( number , k , p ) ) ;
function solve ( a , n ) { let max1 = Number . MIN_VALUE ; for ( let i = 0 ; i < n ; i ++ ) { for ( let j = 0 ; j < n ; j ++ ) { if ( Math . abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
let arr = [ - 1 , 2 , 3 , - 4 , - 10 , 22 ] ; let size = arr . length ; document . write ( " " + solve ( arr , size ) ) ;
function solve ( a , n ) { let min1 = a [ 0 ] ; let max1 = a [ 0 ] ;
for ( let i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . abs ( min1 - max1 ) ; }
let arr = [ - 1 , 2 , 3 , 4 , - 10 ] ; let size = arr . length ; document . write ( " " + solve ( arr , size ) ) ;
