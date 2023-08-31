import java . util . * ; class solution {
static void Alphabet_N_Pattern ( int N ) { int index , side_index , size ;
int Right = 1 , Left = 1 , Diagonal = 2 ;
for ( index = 0 ; index < N ; index ++ ) {
System . out . print ( Left ++ ) ;
for ( side_index = 0 ; side_index < 2 * ( index ) ; side_index ++ ) System . out . print ( " ▁ " ) ;
if ( index != 0 && index != N - 1 ) System . out . print ( Diagonal ++ ) ; else System . out . print ( " ▁ " ) ;
for ( side_index = 0 ; side_index < 2 * ( N - index - 1 ) ; side_index ++ ) System . out . print ( " ▁ " ) ;
System . out . print ( Right ++ ) ; System . out . println ( ) ; } }
public static void main ( String args [ ] ) {
int Size = 6 ;
Alphabet_N_Pattern ( Size ) ; } }
import java . io . * ; import java . math . * ; class GFG {
static int permutationCoeff ( int n , int k ) { int P [ ] [ ] = new int [ n + 2 ] [ k + 2 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= Math . min ( i , k ) ; j ++ ) {
if ( j == 0 ) P [ i ] [ j ] = 1 ;
else P [ i ] [ j ] = P [ i - 1 ] [ j ] + ( j * P [ i - 1 ] [ j - 1 ] ) ;
P [ i ] [ j + 1 ] = 0 ; } } return P [ n ] [ k ] ; }
public static void main ( String args [ ] ) { int n = 10 , k = 2 ; System . out . println ( " Value ▁ of ▁ P ( ▁ " + n + " , " + k + " ) " + " ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
import java . io . * ; public class GFG {
static int permutationCoeff ( int n , int k ) { int [ ] fact = new int [ n + 1 ] ;
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = i * fact [ i - 1 ] ;
return fact [ n ] / fact [ n - k ] ; }
static public void main ( String [ ] args ) { int n = 10 , k = 2 ; System . out . println ( " Value ▁ of " + " ▁ P ( ▁ " + n + " , ▁ " + k + " ) ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
class GFG {
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
class PellNumber {
public static int pell ( int n ) { if ( n <= 2 ) return n ; int a = 1 ; int b = 2 ; int c ; for ( int i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( pell ( n ) ) ; } }
class Test {
static int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
public static void main ( String [ ] args ) { int num = 5 ; System . out . println ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + factorial ( 5 ) ) ; } }
class LargestSubArray {
int findSubArray ( int arr [ ] , int n ) { int sum = 0 ; int maxsize = - 1 , startindex = 0 ; int endindex = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? - 1 : 1 ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] == 0 ) sum += - 1 ; else sum += 1 ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } endindex = startindex + maxsize - 1 ; if ( maxsize == - 1 ) System . out . println ( " No ▁ such ▁ subarray " ) ; else System . out . println ( startindex + " ▁ to ▁ " + endindex ) ; return maxsize ; }
public static void main ( String [ ] args ) { LargestSubArray sub ; sub = new LargestSubArray ( ) ; int arr [ ] = { 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int size = arr . length ; sub . findSubArray ( arr , size ) ; } }
class GFG {
static int ternarySearch ( int l , int r , int key , int ar [ ] ) { while ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
r = mid1 - 1 ; } else if ( key > ar [ mid2 ] ) {
l = mid2 + 1 ; } else {
l = mid1 + 1 ; r = mid2 - 1 ; } }
return - 1 ; }
public static void main ( String args [ ] ) { int l , r , p , key ;
int ar [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
System . out . println ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
System . out . println ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class Minimum { static int findMin ( int arr [ ] , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
public static void main ( String [ ] args ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = arr1 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr1 , 0 , n1 - 1 ) ) ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = arr2 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr2 , 0 , n2 - 1 ) ) ; int arr3 [ ] = { 1 } ; int n3 = arr3 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr3 , 0 , n3 - 1 ) ) ; int arr4 [ ] = { 1 , 2 } ; int n4 = arr4 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr4 , 0 , n4 - 1 ) ) ; int arr5 [ ] = { 2 , 1 } ; int n5 = arr5 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr5 , 0 , n5 - 1 ) ) ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = arr6 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr6 , 0 , n6 - 1 ) ) ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = arr7 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr7 , 0 , n7 - 1 ) ) ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = arr8 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr8 , 0 , n8 - 1 ) ) ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = arr9 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr9 , 0 , n9 - 1 ) ) ; } }
import java . io . * ; class SecondSmallest {
static void print2Smallest ( int arr [ ] ) { int first , second , arr_size = arr . length ;
if ( arr_size < 2 ) { System . out . println ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = Integer . MAX_VALUE ; for ( int i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == Integer . MAX_VALUE ) System . out . println ( " There ▁ is ▁ no ▁ second " + " smallest ▁ element " ) ; else System . out . println ( " The ▁ smallest ▁ element ▁ is ▁ " + first + " ▁ and ▁ second ▁ Smallest " + " ▁ element ▁ is ▁ " + second ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 12 , 13 , 1 , 10 , 34 , 1 } ; print2Smallest ( arr ) ; } }
static boolean isSubsetSum ( int arr [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ 2 ] [ sum + 1 ] ; for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) {
if ( j == 0 ) subset [ i % 2 ] [ j ] = true ;
else if ( i == 0 ) subset [ i % 2 ] [ j ] = false ; else if ( arr [ i - 1 ] <= j ) subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] || subset [ ( i + 1 ) % 2 ] [ j ] ; else subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] ; } } return subset [ n % 2 ] [ sum ] ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 2 , 5 } ; int sum = 7 ; int n = arr . length ; if ( isSubsetSum ( arr , n , sum ) == true ) System . out . println ( " There ▁ exists ▁ a ▁ subset ▁ with " + " given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ exists ▁ with " + " given ▁ sum " ) ; } }
class MajorityElement {
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
boolean isMajority ( int a [ ] , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) { if ( a [ i ] == cand ) count ++ ; } if ( count > size / 2 ) return true ; else return false ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) System . out . println ( " ▁ " + cand + " ▁ " ) ; else System . out . println ( " No ▁ Majority ▁ Element " ) ; }
public static void main ( String [ ] args ) { MajorityElement majorelement = new MajorityElement ( ) ; int a [ ] = new int [ ] { 1 , 3 , 3 , 1 , 2 } ;
int size = a . length ; majorelement . printMajority ( a , size ) ; } }
class GFG {
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ sum + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
for ( int i = 0 ; i <= sum ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) System . out . println ( subset [ i ] [ j ] ) ; } return subset [ sum ] [ n ] ; }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
class Test {
static boolean isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void main ( String [ ] args ) { System . out . println ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; System . out . println ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
import java . io . * ; class GFG { static int nextPowerOf2 ( int n ) { int count = 0 ;
if ( n > 0 && ( n & ( n - 1 ) ) == 0 ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
public static void main ( String args [ ] ) { int n = 0 ; System . out . println ( nextPowerOf2 ( n ) ) ; } }
import java . lang . * ; import java . util . * ; public class GfG {
public static int countWays ( int n ) { int [ ] res = new int [ n + 1 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
public static void main ( String argc [ ] ) { int n = 4 ; System . out . println ( countWays ( n ) ) ; } }
class GFG {
static int maxTasks ( int high [ ] , int low [ ] , int n ) {
if ( n <= 0 ) return 0 ;
return Math . max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; System . out . println ( maxTasks ( high , low , n ) ) ; } }
public class GFG { static final int OUT = 0 ; static final int IN = 1 ;
static int countWords ( String str ) { int state = OUT ;
int wc = 0 ; int i = 0 ;
while ( i < str . length ( ) ) {
if ( str . charAt ( i ) == ' ▁ ' || str . charAt ( i ) == 'NEW_LINE' || str . charAt ( i ) == ' TABSYMBOL ' ) state = OUT ;
else if ( state == OUT ) { state = IN ; ++ wc ; }
++ i ; } return wc ; }
public static void main ( String args [ ] ) { String str = "One twothree four five "; System . out . println ( " No ▁ of ▁ words ▁ : ▁ " + countWords ( str ) ) ; } }
class GFG {
static int max ( int x , int y ) { return ( x > y ? x : y ) ; }
static int maxTasks ( int [ ] high , int [ ] low , int n ) {
int [ ] task_dp = new int [ n + 1 ] ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( int i = 2 ; i <= n ; i ++ ) task_dp [ i ] = Math . max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
public static void main ( String [ ] args ) { int n = 5 ; int [ ] high = { 3 , 6 , 8 , 7 , 6 } ; int [ ] low = { 1 , 5 , 4 , 5 , 3 } ; System . out . println ( maxTasks ( high , low , n ) ) ; } }
import java . io . * ; class Partition {
static boolean findPartition ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; boolean part [ ] [ ] = new boolean [ sum / 2 + 1 ] [ n + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) part [ 0 ] [ i ] = true ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) part [ i ] [ 0 ] = false ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) { for ( j = 1 ; j <= n ; j ++ ) { part [ i ] [ j ] = part [ i ] [ j - 1 ] ; if ( i >= arr [ j - 1 ] ) part [ i ] [ j ] = part [ i ] [ j ] || part [ i - arr [ j - 1 ] ] [ j - 1 ] ; } }
return part [ sum / 2 ] [ n ] ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 1 , 1 , 2 , 2 , 1 } ; int n = arr . length ;
if ( findPartition ( arr , n ) == true ) System . out . println ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " " subsets ▁ of ▁ equal ▁ sum " ) ; else System . out . println ( " Can ▁ not ▁ be ▁ divided ▁ into " " ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
import java . util . * ; class GFG {
static int dfa = 0 ;
static void start ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ; }
static void state1 ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ;
else if ( c == ' h ' c == ' H ' ) dfa = 2 ;
else dfa = 0 ; }
static void state2 ( char c ) {
if ( c == ' e ' c == ' E ' ) dfa = 3 ; else dfa = 0 ; }
static void state3 ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ; else dfa = 0 ; } static boolean isAccepted ( char str [ ] ) {
int len = str . length ; for ( int i = 0 ; i < len ; i ++ ) { if ( dfa == 0 ) start ( str [ i ] ) ; else if ( dfa == 1 ) state1 ( str [ i ] ) ; else if ( dfa == 2 ) state2 ( str [ i ] ) ; else state3 ( str [ i ] ) ; } return ( dfa != 3 ) ; }
public static void main ( String [ ] args ) { char str [ ] = " forTHEgeeks " . toCharArray ( ) ; if ( isAccepted ( str ) == true ) System . out . println ( "ACCEPTEDNEW_LINE"); else System . out . println ( "NOT ACCEPTEDNEW_LINE"); } }
import java . io . * ; class Maths {
static double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / Math . pow ( i , i ) ; sums += ser ; } return sums ; }
public static void main ( String [ ] args ) { int n = 3 ; double res = Series ( n ) ; res = Math . round ( res * 100000.0 ) / 100000.0 ; System . out . println ( res ) ; } }
class GFG {
static int ternarySearch ( int l , int r , int key , int ar [ ] ) { if ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
return ternarySearch ( l , mid1 - 1 , key , ar ) ; } else if ( key > ar [ mid2 ] ) {
return ternarySearch ( mid2 + 1 , r , key , ar ) ; } else {
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) ; } }
return - 1 ; }
public static void main ( String args [ ] ) { int l , r , p , key ;
int ar [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
System . out . println ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
System . out . println ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ; } }
class GFG { static final int N = 4 ;
static void add ( int A [ ] [ ] , int B [ ] [ ] , int C [ ] [ ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] ; }
public static void main ( String [ ] args ) { int A [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ ] [ ] = new int [ N ] [ N ] ; int i , j ; add ( A , B , C ) ; System . out . print ( "Result matrix is NEW_LINE"); for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) System . out . print ( C [ i ] [ j ] + " ▁ " ) ; System . out . print ( "NEW_LINE"); } } }
class GFG { static final int N = 4 ;
static void subtract ( int A [ ] [ ] , int B [ ] [ ] , int C [ ] [ ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] ; }
public static void main ( String [ ] args ) { int A [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ ] [ ] = new int [ N ] [ N ] ; int i , j ; subtract ( A , B , C ) ; System . out . print ( "Result matrix is NEW_LINE"); for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) System . out . print ( C [ i ] [ j ] + " ▁ " ) ; System . out . print ( "NEW_LINE"); } } }
class Main { static int linearSearch ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == i ) return i ; }
return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] = { - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = arr . length ; System . out . println ( " Fixed ▁ Point ▁ is ▁ " + linearSearch ( arr , n ) ) ; } }
class Main { static int binarySearch ( int arr [ ] , int low , int high ) { if ( high >= low ) {
int mid = ( low + high ) / 2 ; if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] = { - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = arr . length ; System . out . println ( " Fixed ▁ Point ▁ is ▁ " + binarySearch ( arr , 0 , n - 1 ) ) ; } }
class GFG { public static int search ( int arr [ ] , int x ) { int n = arr . length ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == x ) return i ; } return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ;
int result = search ( arr , x ) ; if ( result == - 1 ) System . out . print ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) ; else System . out . print ( " Element ▁ is ▁ present ▁ at ▁ index ▁ " + result ) ; } }
class CountingSort {
void sort ( char arr [ ] ) { int n = arr . length ;
char output [ ] = new char [ n ] ;
int count [ ] = new int [ 256 ] ; for ( int i = 0 ; i < 256 ; ++ i ) count [ i ] = 0 ;
for ( int i = 0 ; i < n ; ++ i ) ++ count [ arr [ i ] ] ;
for ( int i = 1 ; i <= 255 ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( int i = 0 ; i < n ; ++ i ) arr [ i ] = output [ i ] ; }
public static void main ( String args [ ] ) { CountingSort ob = new CountingSort ( ) ; char arr [ ] = { ' g ' , ' e ' , ' e ' , ' k ' , ' s ' , ' f ' , ' o ' , ' r ' , ' g ' , ' e ' , ' e ' , ' k ' , ' s ' } ; ob . sort ( arr ) ; System . out . print ( " Sorted ▁ character ▁ array ▁ is ▁ " ) ; for ( int i = 0 ; i < arr . length ; ++ i ) System . out . print ( arr [ i ] ) ; } }
import java . util . * ; class GFG {
static int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
public static void main ( String [ ] args ) { int n = 5 , k = 2 ; System . out . printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; } }
import java . io . * ; class GFG {
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
public static void main ( String [ ] args ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) System . out . println ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ keystrokes ▁ is ▁ " + N + findoptimal ( N ) ) ; } }
import java . io . * ; class GFG {
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int screen [ ] = new int [ N ] ;
int b ;
int n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = 0 ;
for ( b = n - 3 ; b >= 1 ; b -- ) {
int curr = ( n - b - 1 ) * screen [ b - 1 ] ; if ( curr > screen [ n - 1 ] ) screen [ n - 1 ] = curr ; } } return screen [ N - 1 ] ; }
public static void main ( String [ ] args ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) System . out . println ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ keystrokes ▁ is ▁ " + N + findoptimal ( N ) ) ; } }
class GFG {
static int power ( int x , int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; }
public static void main ( String [ ] args ) { int x = 2 ; int y = 3 ; System . out . printf ( " % d " , power ( x , y ) ) ; } }
static int power ( int x , int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
class GFG { static float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
public static void main ( String [ ] args ) { float x = 2 ; int y = - 3 ; System . out . printf ( " % f " , power ( x , y ) ) ; } }
class GFG {
static float squareRoot ( float n ) {
float x = n ; float y = 1 ;
double e = 0.000001 ; while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
public static void main ( String [ ] args ) { int n = 50 ; System . out . printf ( " Square ▁ root ▁ of ▁ " + n + " ▁ is ▁ " + squareRoot ( n ) ) ; } }
class GFG { static int sum , n ;
static float getAvg ( int x ) { sum += x ; return ( ( ( float ) sum ) / ++ n ) ; }
static void streamAvg ( float [ ] arr , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( ( int ) arr [ i ] ) ; System . out . println ( " Average ▁ of ▁ " + ( i + 1 ) + " ▁ numbers ▁ is ▁ " + avg ) ; } return ; }
public static void main ( String [ ] args ) { float [ ] arr = new float [ ] { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . length ; streamAvg ( arr , n ) ; } }
class BinomialCoefficient {
static int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
public static void main ( String [ ] args ) { int n = 8 ; int k = 2 ; System . out . println ( " Value ▁ of ▁ C ( " + n + " , ▁ " + k + " ) ▁ " + " is " + " ▁ " + binomialCoeff ( n , k ) ) ; } }
import java . io . * ; import java . lang . Math ; class GFG {
public static void primeFactors ( int n ) {
while ( n % 2 == 0 ) { System . out . print ( 2 + " ▁ " ) ; n /= 2 ; }
for ( int i = 3 ; i <= Math . sqrt ( n ) ; i += 2 ) {
while ( n % i == 0 ) { System . out . print ( i + " ▁ " ) ; n /= i ; } }
if ( n > 2 ) System . out . print ( n ) ; }
public static void main ( String [ ] args ) { int n = 315 ; primeFactors ( n ) ; } }
import java . io . * ; class Combination {
static void printCombination ( int arr [ ] , int n , int r ) {
int data [ ] = new int [ r ] ;
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; }
static void combinationUtil ( int arr [ ] , int data [ ] , int start , int end , int index , int r ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) System . out . print ( data [ j ] + " ▁ " ) ; System . out . println ( " " ) ; return ; }
for ( int i = start ; i <= end && end - i + 1 >= r - index ; i ++ ) { data [ index ] = arr [ i ] ; combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . length ; printCombination ( arr , n , r ) ; } }
import java . io . * ; class Combination {
static void printCombination ( int arr [ ] , int n , int r ) {
int data [ ] = new int [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
static void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) System . out . print ( data [ j ] + " ▁ " ) ; System . out . println ( " " ) ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . length ; printCombination ( arr , n , r ) ; } }
class FindGroups {
int findgroups ( int arr [ ] , int n ) {
int c [ ] = new int [ ] { 0 , 0 , 0 } ; int i ;
int res = 0 ;
for ( i = 0 ; i < n ; i ++ ) c [ arr [ i ] % 3 ] ++ ;
res += ( ( c [ 0 ] * ( c [ 0 ] - 1 ) ) >> 1 ) ;
res += c [ 1 ] * c [ 2 ] ;
res += ( c [ 0 ] * ( c [ 0 ] - 1 ) * ( c [ 0 ] - 2 ) ) / 6 ;
res += ( c [ 1 ] * ( c [ 1 ] - 1 ) * ( c [ 1 ] - 2 ) ) / 6 ;
res += ( ( c [ 2 ] * ( c [ 2 ] - 1 ) * ( c [ 2 ] - 2 ) ) / 6 ) ;
res += c [ 0 ] * c [ 1 ] * c [ 2 ] ;
return res ; }
public static void main ( String [ ] args ) { FindGroups groups = new FindGroups ( ) ; int arr [ ] = { 3 , 6 , 7 , 2 , 9 } ; int n = arr . length ; System . out . println ( " Required ▁ number ▁ of ▁ groups ▁ are ▁ " + groups . findgroups ( arr , n ) ) ; } }
import java . io . * ; class GFG { static int nextPowerOf2 ( int n ) { int count = 0 ;
if ( n > 0 && ( n & ( n - 1 ) ) == 0 ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
public static void main ( String args [ ] ) { int n = 0 ; System . out . println ( nextPowerOf2 ( n ) ) ; } }
import java . io . * ; class GFG { static int nextPowerOf2 ( int n ) { int p = 1 ; if ( n > 0 && ( n & ( n - 1 ) ) == 0 ) return n ; while ( p < n ) p <<= 1 ; return p ; }
public static void main ( String args [ ] ) { int n = 5 ; System . out . println ( nextPowerOf2 ( n ) ) ; } }
import java . io . * ; class GFG {
static int nextPowerOf2 ( int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
public static void main ( String args [ ] ) { int n = 5 ; System . out . println ( nextPowerOf2 ( n ) ) ; } }
class Segregate {
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
public static void main ( String [ ] args ) { Segregate seg = new Segregate ( ) ; int arr [ ] = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = arr . length ; seg . segregate0and1 ( arr , arr_size ) ; System . out . print ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
class FindMaximum {
int maxIndexDiff ( int arr [ ] , int n ) { int maxDiff = - 1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
public static void main ( String [ ] args ) { FindMaximum max = new FindMaximum ( ) ; int arr [ ] = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = arr . length ; int maxDiff = max . maxIndexDiff ( arr , n ) ; System . out . println ( maxDiff ) ; } }
import java . lang . * ; import java . util . * ; public class GfG {
public static int findStep ( int n ) { if ( n == 1 n == 0 ) return 1 ; else if ( n == 2 ) return 2 ; else return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) ; }
public static void main ( String argc [ ] ) { int n = 4 ; System . out . println ( findStep ( n ) ) ; } }
import java . io . * ; class Partition {
static boolean isSubsetSum ( int arr [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 && sum != 0 ) return false ;
if ( arr [ n - 1 ] > sum ) return isSubsetSum ( arr , n - 1 , sum ) ;
return isSubsetSum ( arr , n - 1 , sum ) || isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) ; }
static boolean findPartition ( int arr [ ] , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( sum % 2 != 0 ) return false ;
return isSubsetSum ( arr , n , sum / 2 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 1 , 5 , 9 , 12 } ; int n = arr . length ;
if ( findPartition ( arr , n ) == true ) System . out . println ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " + " subsets ▁ of ▁ equal ▁ sum " ) ; else System . out . println ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " + " two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int findRepeatFirstN2 ( String s ) {
int p = - 1 , i , j ; for ( i = 0 ; i < s . length ( ) ; i ++ ) { for ( j = i + 1 ; j < s . length ( ) ; j ++ ) { if ( s . charAt ( i ) == s . charAt ( j ) ) { p = i ; break ; } } if ( p != - 1 ) break ; } return p ; }
static public void main ( String [ ] args ) { String str = " geeksforgeeks " ; int pos = findRepeatFirstN2 ( str ) ; if ( pos == - 1 ) System . out . println ( " Not ▁ found " ) ; else System . out . println ( str . charAt ( pos ) ) ; } }
class GFG {
static boolean isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
static boolean isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
static long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
public static void main ( String [ ] args ) { int L = 110 , R = 1130 ; System . out . println ( sumOfAllPalindrome ( L , R ) ) ; } }
import java . io . * ; class GFG { static int subtractOne ( int x ) { int m = 1 ;
while ( ! ( ( x & m ) > 0 ) ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
public static void main ( String [ ] args ) { System . out . println ( subtractOne ( 13 ) ) ; } }
import java . io . * ; class GFG {
static int findSum ( int n , int a , int b ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
public static void main ( String [ ] args ) { int n = 10 , a = 3 , b = 5 ; System . out . println ( findSum ( n , a , b ) ) ; } }
class PellNumber {
public static int pell ( int n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( pell ( n ) ) ; } }
import java . io . * ; class GFG {
static int Largestpower ( int n , int p ) {
int ans = 0 ;
while ( n > 0 ) { n /= p ; ans += n ; } return ans ; }
public static void main ( String [ ] args ) { int n = 10 ; int p = 3 ; System . out . println ( " ▁ The ▁ largest ▁ power ▁ of ▁ " + p + " ▁ that ▁ divides ▁ " + n + " ! ▁ is ▁ " + Largestpower ( n , p ) ) ; } }
class Factorial { int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
public static void main ( String args [ ] ) { Factorial obj = new Factorial ( ) ; int num = 5 ; System . out . println ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + obj . factorial ( num ) ) ; } }
class GFG {
static int bitExtracted ( int number , int k , int p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int number = 171 , k = 5 , p = 2 ; System . out . println ( " The ▁ extracted ▁ number ▁ is ▁ " + bitExtracted ( number , k , p ) ) ; } }
import java . io . * ; class GFG {
static int solve ( int [ ] a , int n ) { int max1 = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( Math . abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
static public void main ( String [ ] args ) { int [ ] arr = { - 1 , 2 , 3 , - 4 , - 10 , 22 } ; int size = arr . length ; System . out . println ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
import java . io . * ; class GFG {
static int solve ( int a [ ] , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . abs ( min1 - max1 ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { - 1 , 2 , 3 , 4 , - 10 } ; int size = arr . length ; System . out . println ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
