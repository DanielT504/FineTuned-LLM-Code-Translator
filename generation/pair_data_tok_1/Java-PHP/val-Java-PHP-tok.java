static double Conversion ( double centi ) { double pixels = ( 96 * centi ) / 2.54 ; System . out . println ( pixels ) ; return 0 ; }
public static void main ( String args [ ] ) { int centi = 15 ; Conversion ( centi ) ; } }
static int maxOfMin ( int a [ ] , int n , int S ) {
int mi = Integer . MAX_VALUE ;
int s1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { s1 += a [ i ] ; mi = Math . min ( a [ i ] , mi ) ; }
if ( s1 < S ) return - 1 ;
if ( s1 == S ) return 0 ;
int low = 0 ;
int high = mi ;
int ans = 0 ;
while ( low <= high ) { int mid = ( low + high ) / 2 ;
if ( s1 - ( mid * n ) >= S ) { ans = mid ; low = mid + 1 ; }
else high = mid - 1 ; }
return ans ; }
public static void main ( String [ ] args ) { int a [ ] = { 10 , 10 , 10 , 10 , 10 } ; int S = 10 ; int n = a . length ; System . out . println ( maxOfMin ( a , n , S ) ) ; } }
static void Alphabet_N_Pattern ( int N ) { int index , side_index , size ;
int Right = 1 , Left = 1 , Diagonal = 2 ;
for ( index = 0 ; index < N ; index ++ ) {
System . out . print ( Left ++ ) ;
for ( side_index = 0 ; side_index < 2 * ( index ) ; side_index ++ ) System . out . print ( " ▁ " ) ;
if ( index != 0 && index != N - 1 ) System . out . print ( Diagonal ++ ) ; else System . out . print ( " ▁ " ) ;
for ( side_index = 0 ; side_index < 2 * ( N - index - 1 ) ; side_index ++ ) System . out . print ( " ▁ " ) ;
System . out . print ( Right ++ ) ; System . out . println ( ) ; } }
int Size = 6 ;
Alphabet_N_Pattern ( Size ) ; } }
static int isSumDivides ( int N ) { int temp = N ; int sum = 0 ;
while ( temp > 0 ) { sum += temp % 10 ; temp /= 10 ; } if ( N % sum == 0 ) return 1 ; else return 0 ; }
public static void main ( String args [ ] ) { int N = 12 ; if ( isSumDivides ( N ) == 1 ) System . out . print ( " YES " ) ; else System . out . print ( " NO " ) ; } }
static int sum ( int N ) { int S1 , S2 , S3 ; S1 = ( ( N / 3 ) ) * ( 2 * 3 + ( N / 3 - 1 ) * 3 ) / 2 ; S2 = ( ( N / 4 ) ) * ( 2 * 4 + ( N / 4 - 1 ) * 4 ) / 2 ; S3 = ( ( N / 12 ) ) * ( 2 * 12 + ( N / 12 - 1 ) * 12 ) / 2 ; return S1 + S2 - S3 ; }
public static void main ( String [ ] args ) { int N = 20 ; System . out . print ( sum ( 12 ) ) ; } }
static int nextGreater ( int N ) { int power_of_2 = 1 , shift_count = 0 ;
while ( true ) {
if ( ( ( N >> shift_count ) & 1 ) % 2 == 0 ) break ;
shift_count ++ ;
power_of_2 = power_of_2 * 2 ; }
return ( N + power_of_2 ) ; }
public static void main ( String [ ] a ) { int N = 11 ;
System . out . println ( " The ▁ next ▁ number ▁ is ▁ = ▁ " + nextGreater ( N ) ) ; } }
static void printTetra ( int n ) { int [ ] dp = new int [ n + 5 ] ;
dp [ 0 ] = 0 ; dp [ 1 ] = dp [ 2 ] = 1 ; dp [ 3 ] = 2 ; for ( int i = 4 ; i <= n ; i ++ ) dp [ i ] = dp [ i - 1 ] + dp [ i - 2 ] + dp [ i - 3 ] + dp [ i - 4 ] ; System . out . print ( dp [ n ] ) ; }
public static void main ( String [ ] args ) { int n = 10 ; printTetra ( n ) ; } }
static int maxSum1 ( int arr [ ] , int n ) { int dp [ ] = new int [ n ] ; int maxi = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
dp [ i ] = arr [ i ] ;
if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( int i = 2 ; i < n - 1 ; i ++ ) {
for ( int j = 0 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < dp [ j ] + arr [ i ] ) { dp [ i ] = dp [ j ] + arr [ i ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; }
static int maxSum2 ( int arr [ ] , int n ) { int dp [ ] = new int [ n ] ; int maxi = 0 ; for ( int i = 1 ; i < n ; i ++ ) { dp [ i ] = arr [ i ] ; if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( int i = 3 ; i < n ; i ++ ) {
for ( int j = 1 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < arr [ i ] + dp [ j ] ) { dp [ i ] = arr [ i ] + dp [ j ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; } static int findMaxSum ( int arr [ ] , int n ) { int t = Math . max ( maxSum1 ( arr , n ) , maxSum2 ( arr , n ) ) ; return t ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 1 } ; int n = arr . length ; System . out . println ( findMaxSum ( arr , n ) ) ; } }
static int permutationCoeff ( int n , int k ) { int P [ ] [ ] = new int [ n + 2 ] [ k + 2 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= Math . min ( i , k ) ; j ++ ) {
if ( j == 0 ) P [ i ] [ j ] = 1 ;
else P [ i ] [ j ] = P [ i - 1 ] [ j ] + ( j * P [ i - 1 ] [ j - 1 ] ) ;
P [ i ] [ j + 1 ] = 0 ; } } return P [ n ] [ k ] ; }
public static void main ( String args [ ] ) { int n = 10 , k = 2 ; System . out . println ( " Value ▁ of ▁ P ( ▁ " + n + " , " + k + " ) " + " ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
static int permutationCoeff ( int n , int k ) { int [ ] fact = new int [ n + 1 ] ;
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = i * fact [ i - 1 ] ;
return fact [ n ] / fact [ n - k ] ; }
static public void main ( String [ ] args ) { int n = 10 , k = 2 ; System . out . println ( " Value ▁ of " + " ▁ P ( ▁ " + n + " , ▁ " + k + " ) ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
static int no_of_ways ( String s ) { int n = s . length ( ) ;
int count_left = 0 , count_right = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { if ( s . charAt ( i ) == s . charAt ( 0 ) ) { ++ count_left ; } else break ; }
for ( int i = n - 1 ; i >= 0 ; -- i ) { if ( s . charAt ( i ) == s . charAt ( n - 1 ) ) { ++ count_right ; } else break ; }
if ( s . charAt ( 0 ) == s . charAt ( n - 1 ) ) return ( ( count_left + 1 ) * ( count_right + 1 ) ) ;
else return ( count_left + count_right + 1 ) ; }
public static void main ( String args [ ] ) { String s = " geeksforgeeks " ; System . out . println ( no_of_ways ( s ) ) ; } }
static void preCompute ( int n , String s , int pref [ ] ) { pref [ 0 ] = 0 ; for ( int i = 1 ; i < n ; i ++ ) { pref [ i ] = pref [ i - 1 ] ; if ( s . charAt ( i - 1 ) == s . charAt ( i ) ) pref [ i ] ++ ; } }
static int query ( int pref [ ] , int l , int r ) { return pref [ r ] - pref [ l ] ; }
public static void main ( String [ ] args ) { String s = " ggggggg " ; int n = s . length ( ) ; int pref [ ] = new int [ n ] ; preCompute ( n , s , pref ) ;
int l = 1 ; int r = 2 ; System . out . println ( query ( pref , l , r ) ) ;
l = 1 ; r = 5 ; System . out . println ( query ( pref , l , r ) ) ; } }
static String findDirection ( String s ) { int count = 0 ; String d = " " ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s . charAt ( 0 ) == 'NEW_LINE') return null ; if ( s . charAt ( i ) == ' L ' ) count -- ; else { if ( s . charAt ( i ) == ' R ' ) count ++ ; } }
if ( count > 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == 1 ) d = " E " ; else if ( count % 4 == 2 ) d = " S " ; else if ( count % 4 == 3 ) d = " W " ; }
if ( count < 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == - 1 ) d = " W " ; else if ( count % 4 == - 2 ) d = " S " ; else if ( count % 4 == - 3 ) d = " E " ; } return d ; }
public static void main ( String [ ] args ) { String s = " LLRLRRL " ; System . out . println ( findDirection ( s ) ) ; s = " LL " ; System . out . println ( findDirection ( s ) ) ; } }
static void encode ( String s , int k ) {
String newS = " " ;
for ( int i = 0 ; i < s . length ( ) ; ++ i ) {
int val = s . charAt ( i ) ;
int dup = k ;
if ( val + k > 122 ) { k -= ( 122 - val ) ; k = k % 26 ; newS += ( char ) ( 96 + k ) ; } else { newS += ( char ) ( val + k ) ; } k = dup ; }
System . out . println ( newS ) ; }
public static void main ( String [ ] args ) { String str = " abc " ; int k = 28 ;
encode ( str , k ) ; } }
static boolean isVowel ( char x ) { if ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) return true ; else return false ; }
static String updateSandwichedVowels ( String a ) { int n = a . length ( ) ;
String updatedString = " " ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i == 0 i == n - 1 ) { updatedString += a . charAt ( i ) ; continue ; }
if ( isVowel ( a . charAt ( i ) ) == true && isVowel ( a . charAt ( i - 1 ) ) == false && isVowel ( a . charAt ( i + 1 ) ) == false ) { continue ; }
updatedString += a . charAt ( i ) ; } return updatedString ; }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ;
String updatedString = updateSandwichedVowels ( str ) ; System . out . print ( updatedString ) ; } }
static int findNumbers ( int n , int w ) { int x = 0 , sum = 0 ;
if ( w >= 0 && w <= 8 ) {
x = 9 - w ; }
else if ( w >= - 9 && w <= - 1 ) {
x = 10 + w ; } sum = ( int ) Math . pow ( 10 , n - 2 ) ; sum = ( x * sum ) ; return sum ; }
n = 3 ; w = 4 ;
System . out . println ( findNumbers ( n , w ) ) ; } }
import java . io . * ; class GFG { static int MaximumHeight ( int [ ] a , int n ) { int result = 1 ; for ( int i = 1 ; i <= n ; ++ i ) {
int y = ( i * ( i + 1 ) ) / 2 ;
if ( y < n ) result = i ;
else break ; } return result ; }
public static void main ( String [ ] args ) { int [ ] arr = { 40 , 100 , 20 , 30 } ; int n = arr . length ; System . out . println ( MaximumHeight ( arr , n ) ) ; } }
import java . util . * ; class GFG { static int findK ( int n , int k ) { ArrayList < Integer > a = new ArrayList < Integer > ( n ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( i % 2 == 1 ) a . add ( i ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( i % 2 == 0 ) a . add ( i ) ; return ( a . get ( k - 1 ) ) ; }
public static void main ( String [ ] args ) { int n = 10 , k = 3 ; System . out . println ( findK ( n , k ) ) ; } }
import java . io . * ; class GFG { static int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
public static void main ( String [ ] args ) { int num = 5 ; System . out . println ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + factorial ( num ) ) ; } }
public static int pell ( int n ) { if ( n <= 2 ) return n ; int a = 1 ; int b = 2 ; int c ; for ( int i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( pell ( n ) ) ; } }
static boolean isMultipleOf10 ( int n ) { if ( n % 15 == 0 ) return true ; return false ; }
public static void main ( String [ ] args ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int countOddPrimeFactors ( int n ) { int result = 1 ;
while ( n % 2 == 0 ) n /= 2 ;
for ( int i = 3 ; i * i <= n ; i += 2 ) { int divCount = 0 ;
while ( n % i == 0 ) { n /= i ; ++ divCount ; } result *= divCount + 1 ; }
if ( n > 2 ) result *= 2 ; return result ; } static int politness ( int n ) { return countOddPrimeFactors ( n ) - 1 ; }
public static void main ( String [ ] args ) { int n = 90 ; System . out . println ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; n = 15 ; System . out . println ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; } }
import java . util . * ; class GFG { static int MAX = 1000000 ;
static ArrayList < Integer > primes = new ArrayList < Integer > ( ) ;
static void Sieve ( ) { int n = MAX ;
int nNew = ( int ) Math . sqrt ( n ) ;
int [ ] marked = new int [ n / 2 + 500 ] ;
for ( int i = 1 ; i <= ( nNew - 1 ) / 2 ; i ++ ) for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= n / 2 ; j = j + 2 * i + 1 ) marked [ j ] = 1 ;
primes . add ( 2 ) ;
for ( int i = 1 ; i <= n / 2 ; i ++ ) if ( marked [ i ] == 0 ) primes . add ( 2 * i + 1 ) ; }
static int binarySearch ( int left , int right , int n ) { if ( left <= right ) { int mid = ( left + right ) / 2 ;
if ( mid == 0 || mid == primes . size ( ) - 1 ) return primes . get ( mid ) ;
if ( primes . get ( mid ) == n ) return primes . get ( mid - 1 ) ;
if ( primes . get ( mid ) < n && primes . get ( mid + 1 ) > n ) return primes . get ( mid ) ; if ( n < primes . get ( mid ) ) return binarySearch ( left , mid - 1 , n ) ; else return binarySearch ( mid + 1 , right , n ) ; } return 0 ; }
public static void main ( String [ ] args ) { Sieve ( ) ; int n = 17 ; System . out . println ( binarySearch ( 0 , primes . size ( ) - 1 , n ) ) ; } }
static int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
public static void main ( String [ ] args ) { int num = 5 ; System . out . println ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + factorial ( 5 ) ) ; } }
static int printKDistinct ( int arr [ ] , int n , int k ) { int dist_count = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return - 1 ; }
public static void main ( String [ ] args ) { int ar [ ] = { 1 , 2 , 1 , 3 , 4 , 2 } ; int n = ar . length ; int k = 2 ; System . out . print ( printKDistinct ( ar , n , k ) ) ; } }
static int calculate ( int a [ ] , int n ) {
Arrays . sort ( a ) ; int count = 1 ; int answer = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( a [ i ] == a [ i - 1 ] ) {
count += 1 ; } else {
answer = answer + ( count * ( count - 1 ) ) / 2 ; count = 1 ; } } answer = answer + ( count * ( count - 1 ) ) / 2 ; return answer ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 1 , 2 , 4 } ; int n = a . length ;
System . out . println ( calculate ( a , n ) ) ; } }
static int calculate ( int a [ ] , int n ) {
int maximum = Arrays . stream ( a ) . max ( ) . getAsInt ( ) ;
int frequency [ ] = new int [ maximum + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
frequency [ a [ i ] ] += 1 ; } int answer = 0 ;
for ( int i = 0 ; i < ( maximum ) + 1 ; i ++ ) {
answer = answer + frequency [ i ] * ( frequency [ i ] - 1 ) ; } return answer / 2 ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 1 , 2 , 4 } ; int n = a . length ;
System . out . println ( calculate ( a , n ) ) ; } }
int findSubArray ( int arr [ ] , int n ) { int sum = 0 ; int maxsize = - 1 , startindex = 0 ; int endindex = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? - 1 : 1 ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] == 0 ) sum += - 1 ; else sum += 1 ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } endindex = startindex + maxsize - 1 ; if ( maxsize == - 1 ) System . out . println ( " No ▁ such ▁ subarray " ) ; else System . out . println ( startindex + " ▁ to ▁ " + endindex ) ; return maxsize ; }
public static void main ( String [ ] args ) { LargestSubArray sub ; sub = new LargestSubArray ( ) ; int arr [ ] = { 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int size = arr . length ; sub . findSubArray ( arr , size ) ; } }
static int findMax ( int arr [ ] , int low , int high ) {
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid == 0 && arr [ mid ] > arr [ mid + 1 ] ) { return arr [ mid ] ; }
if ( arr [ low ] > arr [ mid ] ) { return findMax ( arr , low , mid - 1 ) ; } else { return findMax ( arr , mid + 1 , high ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 6 , 5 , 4 , 3 , 2 , 1 } ; int n = arr . length ; System . out . println ( findMax ( arr , 0 , n - 1 ) ) ; } }
static int search ( int arr [ ] , int l , int h , int key ) { if ( l > h ) return - 1 ; int mid = ( l + h ) / 2 ; if ( arr [ mid ] == key ) return mid ;
if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
return search ( arr , mid + 1 , h , key ) ; }
if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 4 , 5 , 6 , 7 , 8 , 9 , 1 , 2 , 3 } ; int n = arr . length ; int key = 6 ; int i = search ( arr , 0 , n - 1 , key ) ; if ( i != - 1 ) System . out . println ( " Index : ▁ " + i ) ; else System . out . println ( " Key ▁ not ▁ found " ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class Minimum { static int findMin ( int arr [ ] , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
public static void main ( String [ ] args ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = arr1 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr1 , 0 , n1 - 1 ) ) ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = arr2 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr2 , 0 , n2 - 1 ) ) ; int arr3 [ ] = { 1 } ; int n3 = arr3 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr3 , 0 , n3 - 1 ) ) ; int arr4 [ ] = { 1 , 2 } ; int n4 = arr4 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr4 , 0 , n4 - 1 ) ) ; int arr5 [ ] = { 2 , 1 } ; int n5 = arr5 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr5 , 0 , n5 - 1 ) ) ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = arr6 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr6 , 0 , n6 - 1 ) ) ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = arr7 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr7 , 0 , n7 - 1 ) ) ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = arr8 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr8 , 0 , n8 - 1 ) ) ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = arr9 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr9 , 0 , n9 - 1 ) ) ; } }
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
static int findMaxSum ( int [ ] arr , int n ) { int res = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { int prefix_sum = arr [ i ] ; for ( int j = 0 ; j < i ; j ++ ) prefix_sum += arr [ j ] ; int suffix_sum = arr [ i ] ; for ( int j = n - 1 ; j > i ; j -- ) suffix_sum += arr [ j ] ; if ( prefix_sum == suffix_sum ) res = Math . max ( res , prefix_sum ) ; } return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . length ; System . out . println ( findMaxSum ( arr , n ) ) ; } }
static int findMaxSum ( int [ ] arr , int n ) {
int [ ] preSum = new int [ n ] ;
int [ ] suffSum = new int [ n ] ;
int ans = Integer . MIN_VALUE ;
preSum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) preSum [ i ] = preSum [ i - 1 ] + arr [ i ] ;
suffSum [ n - 1 ] = arr [ n - 1 ] ; if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) ans = Math . max ( ans , preSum [ n - 1 ] ) ; for ( int i = n - 2 ; i >= 0 ; i -- ) { suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] ; if ( suffSum [ i ] == preSum [ i ] ) ans = Math . max ( ans , preSum [ i ] ) ; } return ans ; }
static public void main ( String [ ] args ) { int [ ] arr = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . length ; System . out . println ( findMaxSum ( arr , n ) ) ; } }
static void findMajority ( int arr [ ] , int n ) { int maxCount = 0 ;
int index = - 1 ; for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) count ++ ; }
if ( count > maxCount ) { maxCount = count ; index = i ; } }
if ( maxCount > n / 2 ) System . out . println ( arr [ index ] ) ; else System . out . println ( " No ▁ Majority ▁ Element " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = arr . length ;
findMajority ( arr , n ) ; } }
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
boolean isMajority ( int a [ ] , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) { if ( a [ i ] == cand ) count ++ ; } if ( count > size / 2 ) return true ; else return false ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) System . out . println ( " ▁ " + cand + " ▁ " ) ; else System . out . println ( " No ▁ Majority ▁ Element " ) ; }
public static void main ( String [ ] args ) { MajorityElement majorelement = new MajorityElement ( ) ; int a [ ] = new int [ ] { 1 , 3 , 3 , 1 , 2 } ;
int size = a . length ; majorelement . printMajority ( a , size ) ; } }
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ sum + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
for ( int i = 0 ; i <= sum ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) System . out . println ( subset [ i ] [ j ] ) ; } return subset [ sum ] [ n ] ; }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static void print_gcd_online ( int n , int m , int [ ] [ ] query , int [ ] arr ) {
int max_gcd = 0 ; int i = 0 ;
for ( i = 0 ; i < n ; i ++ ) max_gcd = gcd ( max_gcd , arr [ i ] ) ;
for ( i = 0 ; i < m ; i ++ ) {
query [ i ] [ 0 ] -- ;
arr [ query [ i ] [ 0 ] ] /= query [ i ] [ 1 ] ;
max_gcd = gcd ( arr [ query [ i ] [ 0 ] ] , max_gcd ) ;
System . out . println ( max_gcd ) ; } }
public static void main ( String [ ] args ) { int n = 3 ; int m = 3 ; int [ ] [ ] query = new int [ m ] [ 2 ] ; int [ ] arr = new int [ ] { 36 , 24 , 72 } ; query [ 0 ] [ 0 ] = 1 ; query [ 0 ] [ 1 ] = 3 ; query [ 1 ] [ 0 ] = 3 ; query [ 1 ] [ 1 ] = 12 ; query [ 2 ] [ 0 ] = 2 ; query [ 2 ] [ 1 ] = 4 ; print_gcd_online ( n , m , query , arr ) ; } }
class GFG { static final int MAX = 1000000 ;
static boolean [ ] prime = new boolean [ MAX + 1 ] ;
static int [ ] sum = new int [ MAX + 1 ] ;
static void SieveOfEratosthenes ( ) {
for ( int i = 0 ; i <= MAX ; i ++ ) prime [ i ] = true ; for ( int i = 0 ; i <= MAX ; i ++ ) sum [ i ] = 0 ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( int i = 1 ; i <= MAX ; i ++ ) { if ( prime [ i ] == true ) sum [ i ] = 1 ; sum [ i ] += sum [ i - 1 ] ; } }
SieveOfEratosthenes ( ) ;
int l = 3 , r = 9 ;
int c = ( sum [ r ] - sum [ l - 1 ] ) ;
System . out . println ( " Count : ▁ " + c ) ; } }
static float area ( float r ) {
if ( r < 0 ) return - 1 ;
float area = ( float ) ( 3.14 * Math . pow ( r / ( 2 * Math . sqrt ( 2 ) ) , 2 ) ) ; return area ; }
public static void main ( String [ ] args ) { float a = 5 ; System . out . println ( area ( a ) ) ; } }
import java . io . * ; class GFG { static int N = 100005 ;
static boolean prime [ ] = new boolean [ N ] ; static void SieveOfEratosthenes ( ) { for ( int i = 0 ; i < N ; i ++ ) prime [ i ] = true ; prime [ 1 ] = false ; for ( int p = 2 ; p * p < N ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < N ; i += p ) prime [ i ] = false ; } } }
static int almostPrimes ( int n ) {
int ans = 0 ;
for ( int i = 6 ; i <= n ; i ++ ) {
int c = 0 ; for ( int j = 2 ; j * j <= i ; j ++ ) { if ( i % j == 0 ) {
if ( j * j == i ) { if ( prime [ j ] ) c ++ ; } else { if ( prime [ j ] ) c ++ ; if ( prime [ i / j ] ) c ++ ; } } }
if ( c == 2 ) ans ++ ; } return ans ; }
public static void main ( String [ ] args ) { SieveOfEratosthenes ( ) ; int n = 21 ; System . out . println ( almostPrimes ( n ) ) ; } }
static int sumOfDigitsSingle ( int x ) { int ans = 0 ; while ( x != 0 ) { ans += x % 10 ; x /= 10 ; } return ans ; }
static int closest ( int x ) { int ans = 0 ; while ( ans * 10 + 9 <= x ) ans = ans * 10 + 9 ; return ans ; } static int sumOfDigitsTwoParts ( int N ) { int A = closest ( N ) ; return sumOfDigitsSingle ( A ) + sumOfDigitsSingle ( N - A ) ; }
public static void main ( String args [ ] ) { int N = 35 ; System . out . print ( sumOfDigitsTwoParts ( N ) ) ; } }
static boolean isPrime ( int p ) {
double checkNumber = Math . pow ( 2 , p ) - 1 ;
double nextval = 4 % checkNumber ;
for ( int i = 1 ; i < p - 1 ; i ++ ) nextval = ( nextval * nextval - 2 ) % checkNumber ;
return ( nextval == 0 ) ; }
int p = 7 ; double checkNumber = Math . pow ( 2 , p ) - 1 ; if ( isPrime ( p ) ) System . out . println ( ( int ) checkNumber + " ▁ is ▁ Prime . " ) ; else System . out . println ( ( int ) checkNumber + " ▁ is ▁ not ▁ Prime . " ) ; } }
static void sieve ( int n , boolean prime [ ] ) { for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < n ; i += p ) prime [ i ] = false ; } } } static void printSophieGermanNumber ( int n ) {
boolean prime [ ] = new boolean [ 2 * n + 1 ] ; Arrays . fill ( prime , true ) ; sieve ( 2 * n + 1 , prime ) ; for ( int i = 2 ; i < n ; ++ i ) {
if ( prime [ i ] && prime [ 2 * i + 1 ] ) System . out . print ( i + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int n = 25 ; printSophieGermanNumber ( n ) ; } }
static double ucal ( double u , int n ) { if ( n == 0 ) return 1 ; double temp = u ; for ( int i = 1 ; i <= n / 2 ; i ++ ) temp = temp * ( u - i ) ; for ( int i = 1 ; i < n / 2 ; i ++ ) temp = temp * ( u + i ) ; return temp ; }
static int fact ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
int n = 6 ; double x [ ] = { 25 , 26 , 27 , 28 , 29 , 30 } ;
double [ ] [ ] y = new double [ n ] [ n ] ; y [ 0 ] [ 0 ] = 4.000 ; y [ 1 ] [ 0 ] = 3.846 ; y [ 2 ] [ 0 ] = 3.704 ; y [ 3 ] [ 0 ] = 3.571 ; y [ 4 ] [ 0 ] = 3.448 ; y [ 5 ] [ 0 ] = 3.333 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < n - i ; j ++ ) y [ j ] [ i ] = y [ j + 1 ] [ i - 1 ] - y [ j ] [ i - 1 ] ;
DecimalFormat df = new DecimalFormat ( " # . # # # # # # # # " ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n - i ; j ++ ) System . out . print ( y [ i ] [ j ] + " TABSYMBOL " ) ; System . out . println ( " " ) ; }
double value = 27.4 ;
double sum = ( y [ 2 ] [ 0 ] + y [ 3 ] [ 0 ] ) / 2 ;
int k ;
k = n / 2 ; else
double u = ( value - x [ k ] ) / ( x [ 1 ] - x [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) { if ( ( i % 2 ) > 0 ) sum = sum + ( ( u - 0.5 ) * ucal ( u , i - 1 ) * y [ k ] [ i ] ) / fact ( i ) ; else sum = sum + ( ucal ( u , i ) * ( y [ k ] [ i ] + y [ -- k ] [ i ] ) / ( fact ( i ) * 2 ) ) ; } System . out . printf ( " Value ▁ at ▁ " + value + " ▁ is ▁ % .5f " , sum ) ; } }
class Fibonacci { static int fibonacci ( int n ) { int a = 0 ; int b = 1 ; int c = 0 ; if ( n <= 1 ) return n ; for ( int i = 2 ; i <= n ; i ++ ) { c = a + b ; a = b ; b = c ; } return c ; }
static boolean isMultipleOf10 ( int n ) { int f = fibonacci ( 30 ) ; return ( f % 10 == 0 ) ; }
public static void main ( String [ ] args ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void main ( String [ ] args ) { System . out . println ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; System . out . println ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
static int nextPowerOf2 ( int n ) {
int p = 1 ;
if ( n != 0 && ( ( n & ( n - 1 ) ) == 0 ) ) return n ;
while ( p < n ) p <<= 1 ; return p ; }
static int memoryUsed ( int arr [ ] , int n ) {
int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
int nearest = nextPowerOf2 ( sum ) ; return nearest ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 2 } ; int n = arr . length ; System . out . println ( memoryUsed ( arr , n ) ) ; } }
class Toggle { static int toggleKthBit ( int n , int k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int n = 5 , k = 1 ; System . out . println ( toggleKthBit ( n , k ) ) ; } }
import java . io . * ; class GFG { static int nextPowerOf2 ( int n ) { int count = 0 ;
if ( n > 0 && ( n & ( n - 1 ) ) == 0 ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
public static void main ( String args [ ] ) { int n = 0 ; System . out . println ( nextPowerOf2 ( n ) ) ; } }
static void printTetra ( int n ) { if ( n < 0 ) return ;
int first = 0 , second = 1 ; int third = 1 , fourth = 2 ;
int curr = 0 ; if ( n == 0 ) System . out . print ( first ) ; else if ( n == 1 n == 2 ) System . out . print ( second ) ; else if ( n == 3 ) System . out . print ( fourth ) ; else {
for ( int i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } System . out . print ( curr ) ; } }
public static void main ( String [ ] args ) { int n = 10 ; printTetra ( n ) ; } }
public static int countWays ( int n ) { int [ ] res = new int [ n + 1 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
public static void main ( String argc [ ] ) { int n = 4 ; System . out . println ( countWays ( n ) ) ; } }
static int maxTasks ( int high [ ] , int low [ ] , int n ) {
if ( n <= 0 ) return 0 ;
return Math . max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; System . out . println ( maxTasks ( high , low , n ) ) ; } }
static int countSubstr ( String str , int n , char x , char y ) {
int tot_count = 0 ;
int count_x = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str . charAt ( i ) == x ) count_x ++ ;
if ( str . charAt ( i ) == y ) tot_count += count_x ; }
return tot_count ; }
public static void main ( String args [ ] ) { String str = " abbcaceghcak " ; int n = str . length ( ) ; char x = ' a ' , y = ' c ' ; System . out . print ( " Count ▁ = ▁ " + countSubstr ( str , n , x , y ) ) ; } }
public class GFG { static final int OUT = 0 ; static final int IN = 1 ;
static int countWords ( String str ) { int state = OUT ;
int wc = 0 ; int i = 0 ;
while ( i < str . length ( ) ) {
if ( str . charAt ( i ) == ' ▁ ' || str . charAt ( i ) == 'NEW_LINE' || str . charAt ( i ) == ' TABSYMBOL ' ) state = OUT ;
else if ( state == OUT ) { state = IN ; ++ wc ; }
++ i ; } return wc ; }
public static void main ( String args [ ] ) { String str = "One twothree four five "; System . out . println ( " No ▁ of ▁ words ▁ : ▁ " + countWords ( str ) ) ; } }
static int nthEnneadecagonal ( int n ) {
return ( 17 * n * n - 15 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 6 ; System . out . print ( n + " th ▁ Enneadecagonal ▁ number ▁ : " ) ; System . out . println ( nthEnneadecagonal ( n ) ) ; } }
import java . io . * ; class Gfg {
static float areacircumscribed ( float a ) { float PI = 3.14159265f ; return ( a * a * ( PI / 2 ) ) ; }
public static void main ( String arg [ ] ) { float a = 6 ; System . out . print ( " Area ▁ of ▁ an ▁ circumscribed " + " circle ▁ is ▁ : " ) ; System . out . println ( areacircumscribed ( a ) ) ; } }
static int printTetraRec ( int n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
static void printTetra ( int n ) { System . out . println ( printTetraRec ( n ) + " ▁ " ) ; }
public static void main ( String [ ] args ) { int n = 10 ; printTetra ( n ) ; } }
class GFG {
static int max ( int x , int y ) { return ( x > y ? x : y ) ; }
static int maxTasks ( int [ ] high , int [ ] low , int n ) {
int [ ] task_dp = new int [ n + 1 ] ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( int i = 2 ; i <= n ; i ++ ) task_dp [ i ] = Math . max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
public static void main ( String [ ] args ) { int n = 5 ; int [ ] high = { 3 , 6 , 8 , 7 , 6 } ; int [ ] low = { 1 , 5 , 4 , 5 , 3 } ; System . out . println ( maxTasks ( high , low , n ) ) ; } }
import java . io . * ; class GFG { static int PermutationCoeff ( int n , int k ) { int Fn = 1 , Fk = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { Fn *= i ; if ( i == n - k ) Fk = Fn ; } int coeff = Fn / Fk ; return coeff ; }
public static void main ( String args [ ] ) { int n = 10 , k = 2 ; System . out . println ( " Value ▁ of ▁ P ( ▁ " + n + " , " + k + " ) ▁ is ▁ " + PermutationCoeff ( n , k ) ) ; } }
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
static boolean startsWith ( String str , String pre ) { int strLen = str . length ( ) ; int preLen = pre . length ( ) ; int i = 0 , j = 0 ;
while ( i < strLen && j < preLen ) {
if ( str . charAt ( i ) != pre . charAt ( j ) ) return false ; i ++ ; j ++ ; }
return true ; }
static boolean endsWith ( String str , String suff ) { int i = str . length ( ) - 1 ; int j = suff . length ( ) - 1 ;
while ( i >= 0 && j >= 0 ) {
if ( str . charAt ( i ) != suff . charAt ( j ) ) return false ; i -- ; j -- ; }
return true ; }
static boolean checkString ( String str , String a , String b ) {
if ( str . length ( ) != a . length ( ) + b . length ( ) ) return false ;
if ( startsWith ( str , a ) ) {
if ( endsWith ( str , b ) ) return true ; }
if ( startsWith ( str , b ) ) {
if ( endsWith ( str , a ) ) return true ; } return false ; }
public static void main ( String args [ ] ) { String str = " GeeksforGeeks " ; String a = " Geeksfo " ; String b = " rGeeks " ; if ( checkString ( str , a , b ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int minOperations ( String str , int n ) {
int i , lastUpper = - 1 , firstLower = - 1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( Character . isUpperCase ( str . charAt ( i ) ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( Character . isLowerCase ( str . charAt ( i ) ) ) { firstLower = i ; break ; } }
if ( lastUpper == - 1 firstLower == - 1 ) return 0 ;
int countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( Character . isUpperCase ( str . charAt ( i ) ) ) { countUpper ++ ; } }
int countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( Character . isLowerCase ( str . charAt ( i ) ) ) { countLower ++ ; } }
return Math . min ( countLower , countUpper ) ; }
public static void main ( String args [ ] ) { String str = " geEksFOrGEekS " ; int n = str . length ( ) ; System . out . println ( minOperations ( str , n ) ) ; } }
static float rainDayProbability ( int a [ ] , int n ) { float count = 0 , m ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
public static void main ( String args [ ] ) { int a [ ] = { 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 } ; int n = a . length ; System . out . print ( rainDayProbability ( a , n ) ) ; } }
static double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / Math . pow ( i , i ) ; sums += ser ; } return sums ; }
public static void main ( String [ ] args ) { int n = 3 ; double res = Series ( n ) ; res = Math . round ( res * 100000.0 ) / 100000.0 ; System . out . println ( res ) ; } }
static int ternarySearch ( int l , int r , int key , int ar [ ] ) { if ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
return ternarySearch ( l , mid1 - 1 , key , ar ) ; } else if ( key > ar [ mid2 ] ) {
return ternarySearch ( mid2 + 1 , r , key , ar ) ; } else {
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) ; } }
return - 1 ; }
int ar [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
System . out . println ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
System . out . println ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ; } }
public class Char_frequency { static final int SIZE = 26 ;
static void printCharWithFreq ( String str ) {
int n = str . length ( ) ;
int [ ] freq = new int [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str . charAt ( i ) - ' a ' ] != 0 ) {
System . out . print ( str . charAt ( i ) ) ; System . out . print ( freq [ str . charAt ( i ) - ' a ' ] + " ▁ " ) ;
freq [ str . charAt ( i ) - ' a ' ] = 0 ; } } }
public static void main ( String args [ ] ) { String str = " geeksforgeeks " ; printCharWithFreq ( str ) ; } }
import java . io . * ; public class GFG { static void checkHV ( int [ ] [ ] arr , int N , int M ) {
boolean horizontal = true ; boolean vertical = true ;
for ( int i = 0 , k = N - 1 ; i < N / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < M ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } }
for ( int i = 0 , k = M - 1 ; i < M / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } } if ( ! horizontal && ! vertical ) System . out . println ( " NO " ) ; else if ( horizontal && ! vertical ) System . out . println ( " HORIZONTAL " ) ; else if ( vertical && ! horizontal ) System . out . println ( " VERTICAL " ) ; else System . out . println ( " BOTH " ) ; }
static public void main ( String [ ] args ) { int [ ] [ ] mat = { { 1 , 0 , 1 } , { 0 , 0 , 0 } , { 1 , 0 , 1 } } ; checkHV ( mat , 3 , 3 ) ; } }
class GFG { static final int N = 4 ;
static void add ( int A [ ] [ ] , int B [ ] [ ] , int C [ ] [ ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] ; }
public static void main ( String [ ] args ) { int A [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ ] [ ] = new int [ N ] [ N ] ; int i , j ; add ( A , B , C ) ; System . out . print ( "Result matrix is NEW_LINE"); for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) System . out . print ( C [ i ] [ j ] + " ▁ " ) ; System . out . print ( "NEW_LINE"); } } }
static void subtract ( int A [ ] [ ] , int B [ ] [ ] , int C [ ] [ ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] ; }
public static void main ( String [ ] args ) { int A [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ ] [ ] = new int [ N ] [ N ] ; int i , j ; subtract ( A , B , C ) ; System . out . print ( "Result matrix is NEW_LINE"); for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) System . out . print ( C [ i ] [ j ] + " ▁ " ) ; System . out . print ( "NEW_LINE"); } } }
class Main { static int linearSearch ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == i ) return i ; }
return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] = { - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = arr . length ; System . out . println ( " Fixed ▁ Point ▁ is ▁ " + linearSearch ( arr , n ) ) ; } }
class Main { static int binarySearch ( int arr [ ] , int low , int high ) { if ( high >= low ) {
int mid = ( low + high ) / 2 ; if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] = { - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = arr . length ; System . out . println ( " Fixed ▁ Point ▁ is ▁ " + binarySearch ( arr , 0 , n - 1 ) ) ; } }
import java . io . * ; class GFG { static int maxTripletSum ( int arr [ ] , int n ) {
int sum = - 1000000 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) for ( int k = j + 1 ; k < n ; k ++ ) if ( sum < arr [ i ] + arr [ j ] + arr [ k ] ) sum = arr [ i ] + arr [ j ] + arr [ k ] ; return sum ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . length ; System . out . println ( maxTripletSum ( arr , n ) ) ; } }
static int maxTripletSum ( int arr [ ] , int n ) {
Arrays . sort ( arr ) ;
return arr [ n - 1 ] + arr [ n - 2 ] + arr [ n - 3 ] ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . length ; System . out . println ( maxTripletSum ( arr , n ) ) ; } }
static int maxTripletSum ( int arr [ ] , int n ) {
int maxA = - 100000000 , maxB = - 100000000 ; int maxC = - 100000000 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > maxA ) { maxC = maxB ; maxB = maxA ; maxA = arr [ i ] ; }
else if ( arr [ i ] > maxB ) { maxC = maxB ; maxB = arr [ i ] ; }
else if ( arr [ i ] > maxC ) maxC = arr [ i ] ; } return ( maxA + maxB + maxC ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . length ; System . out . println ( maxTripletSum ( arr , n ) ) ; } }
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
static int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
public static void main ( String [ ] args ) { int n = 5 , k = 2 ; System . out . printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; } }
import java . util . * ; class GFG { static int binomialCoeff ( int n , int k ) { int C [ ] = new int [ k + 1 ] ;
C [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = Math . min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
public static void main ( String [ ] args ) { int n = 5 , k = 2 ; System . out . printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; } }
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ sum + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
for ( int i = 0 ; i <= sum ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) System . out . println ( subset [ i ] [ j ] ) ; } return subset [ sum ] [ n ] ; }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
public static void main ( String [ ] args ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) System . out . println ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ keystrokes ▁ is ▁ " + N + findoptimal ( N ) ) ; } }
static int power ( int x , int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; }
public static void main ( String [ ] args ) { int x = 2 ; int y = 3 ; System . out . printf ( " % d " , power ( x , y ) ) ; } }
class GFG { static float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
public static void main ( String [ ] args ) { float x = 2 ; int y = - 3 ; System . out . printf ( " % f " , power ( x , y ) ) ; } }
static float squareRoot ( float n ) {
float x = n ; float y = 1 ;
double e = 0.000001 ; while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
public static void main ( String [ ] args ) { int n = 50 ; System . out . printf ( " Square ▁ root ▁ of ▁ " + n + " ▁ is ▁ " + squareRoot ( n ) ) ; } }
static float getAvg ( float prev_avg , float x , int n ) { return ( prev_avg * n + x ) / ( n + 1 ) ; }
static void streamAvg ( float arr [ ] , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( avg , arr [ i ] , i ) ; System . out . printf ( "Average of %d numbers is %f NEW_LINE", i + 1, avg); } return ; }
public static void main ( String [ ] args ) { float arr [ ] = { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . length ; streamAvg ( arr , n ) ; } }
static float getAvg ( int x ) { sum += x ; return ( ( ( float ) sum ) / ++ n ) ; }
static void streamAvg ( float [ ] arr , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( ( int ) arr [ i ] ) ; System . out . println ( " Average ▁ of ▁ " + ( i + 1 ) + " ▁ numbers ▁ is ▁ " + avg ) ; } return ; }
public static void main ( String [ ] args ) { float [ ] arr = new float [ ] { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . length ; streamAvg ( arr , n ) ; } }
static int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
public static void main ( String [ ] args ) { int n = 8 ; int k = 2 ; System . out . println ( " Value ▁ of ▁ C ( " + n + " , ▁ " + k + " ) ▁ " + " is " + " ▁ " + binomialCoeff ( n , k ) ) ; } }
public static void primeFactors ( int n ) {
while ( n % 2 == 0 ) { System . out . print ( 2 + " ▁ " ) ; n /= 2 ; }
for ( int i = 3 ; i <= Math . sqrt ( n ) ; i += 2 ) {
while ( n % i == 0 ) { System . out . print ( i + " ▁ " ) ; n /= i ; } }
if ( n > 2 ) System . out . print ( n ) ; }
public static void main ( String [ ] args ) { int n = 315 ; primeFactors ( n ) ; } }
static void printCombination ( int arr [ ] , int n , int r ) {
int data [ ] = new int [ r ] ;
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; }
static void combinationUtil ( int arr [ ] , int data [ ] , int start , int end , int index , int r ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) System . out . print ( data [ j ] + " ▁ " ) ; System . out . println ( " " ) ; return ; }
for ( int i = start ; i <= end && end - i + 1 >= r - index ; i ++ ) { data [ index ] = arr [ i ] ; combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . length ; printCombination ( arr , n , r ) ; } }
static void printCombination ( int arr [ ] , int n , int r ) {
int data [ ] = new int [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
static void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) System . out . print ( data [ j ] + " ▁ " ) ; System . out . println ( " " ) ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . length ; printCombination ( arr , n , r ) ; } }
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
static int nextPowerOf2 ( int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
public static void main ( String args [ ] ) { int n = 5 ; System . out . println ( nextPowerOf2 ( n ) ) ; } }
static void segregate0and1 ( int arr [ ] , int n ) {
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 0 ) count ++ ; }
for ( int i = 0 ; i < count ; i ++ ) arr [ i ] = 0 ;
for ( int i = count ; i < n ; i ++ ) arr [ i ] = 1 ; }
static void print ( int arr [ ] , int n ) { System . out . print ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int n = arr . length ; segregate0and1 ( arr , n ) ; print ( arr , n ) ; } }
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
public static void main ( String [ ] args ) { Segregate seg = new Segregate ( ) ; int arr [ ] = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = arr . length ; seg . segregate0and1 ( arr , arr_size ) ; System . out . print ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
static void segregate0and1 ( int arr [ ] ) { int type0 = 0 ; int type1 = arr . length - 1 ; while ( type0 < type1 ) { if ( arr [ type0 ] == 1 ) { arr [ type1 ] = arr [ type1 ] + arr [ type0 ] ; arr [ type0 ] = arr [ type1 ] - arr [ type0 ] ; arr [ type1 ] = arr [ type1 ] - arr [ type0 ] ; type1 -- ; } else { type0 ++ ; } } }
public static void main ( String [ ] args ) { int [ ] array = { 0 , 1 , 0 , 1 , 1 , 1 } ; segregate0and1 ( array ) ; for ( int a : array ) { System . out . print ( a + " ▁ " ) ; } } }
int maxIndexDiff ( int arr [ ] , int n ) { int maxDiff = - 1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
public static void main ( String [ ] args ) { FindMaximum max = new FindMaximum ( ) ; int arr [ ] = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = arr . length ; int maxDiff = max . maxIndexDiff ( arr , n ) ; System . out . println ( maxDiff ) ; } }
static int missingK ( int [ ] a , int k , int n ) { int difference = 0 , ans = 0 , count = k ; boolean flag = false ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = true ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return - 1 ; }
int [ ] a = { 1 , 5 , 11 , 19 } ;
int k = 11 ; int n = a . length ;
int missing = missingK ( a , k , n ) ; System . out . print ( missing ) ; } }
static int findRotations ( String str ) {
String tmp = str + str ; int n = str . length ( ) ; for ( int i = 1 ; i <= n ; i ++ ) {
String substring = tmp . substring ( i , i + str . length ( ) ) ;
if ( str . equals ( substring ) ) return i ; } return n ; }
public static void main ( String [ ] args ) { String str = " aaaa " ; System . out . println ( findRotations ( str ) ) ; } }
static int findKth ( int arr [ ] , int n , int k ) { HashSet < Integer > missing = new HashSet < > ( ) ; int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { missing . add ( arr [ i ] ) ; }
int maxm = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ; int minm = Arrays . stream ( arr ) . min ( ) . getAsInt ( ) ;
for ( int i = minm + 1 ; i < maxm ; i ++ ) {
if ( ! missing . contains ( i ) ) { count ++ ; }
if ( count == k ) { return i ; } }
return - 1 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 10 , 9 , 4 } ; int n = arr . length ; int k = 5 ; System . out . println ( findKth ( arr , n , k ) ) ; } }
static int waysToKAdjacentSetBits ( int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( lastBit == 0 ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
public static void main ( String args [ ] ) { int n = 5 , k = 2 ;
int totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; System . out . println ( " Number ▁ of ▁ ways ▁ = ▁ " + totalWays ) ; } }
public static int findStep ( int n ) { if ( n == 1 n == 0 ) return 1 ; else if ( n == 2 ) return 2 ; else return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) ; }
public static void main ( String argc [ ] ) { int n = 4 ; System . out . println ( findStep ( n ) ) ; } }
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
static int possibleStrings ( int n , int r , int b , int g ) {
int fact [ ] = new int [ n + 1 ] ; fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
int left = n - ( r + g + b ) ; int sum = 0 ;
for ( int i = 0 ; i <= left ; i ++ ) { for ( int j = 0 ; j <= left - i ; j ++ ) { int k = left - ( i + j ) ;
sum = sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ; } }
return sum ; }
public static void main ( String [ ] args ) { int n = 4 , r = 2 ; int b = 0 , g = 1 ; System . out . println ( possibleStrings ( n , r , b , g ) ) ; } }
static int remAnagram ( String str1 , String str2 ) {
int count1 [ ] = new int [ 26 ] ; int count2 [ ] = new int [ 26 ] ;
for ( int i = 0 ; i < str1 . length ( ) ; i ++ ) count1 [ str1 . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < str2 . length ( ) ; i ++ ) count2 [ str2 . charAt ( i ) - ' a ' ] ++ ;
int result = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) result += Math . abs ( count1 [ i ] - count2 [ i ] ) ; return result ; }
public static void main ( String [ ] args ) { String str1 = " bcadeh " , str2 = " hea " ; System . out . println ( remAnagram ( str1 , str2 ) ) ; } }
static void printPath ( Vector < Integer > res , int nThNode , int kThNode ) {
if ( kThNode > nThNode ) return ;
res . add ( kThNode ) ;
for ( int i = 0 ; i < res . size ( ) ; i ++ ) System . out . print ( res . get ( i ) + " ▁ " ) ; System . out . print ( "NEW_LINE");
printPath ( res , nThNode , kThNode * 2 ) ;
printPath ( res , nThNode , kThNode * 2 + 1 ) ; res . remove ( res . size ( ) - 1 ) ; }
static void printPathToCoverAllNodeUtil ( int nThNode ) {
Vector < Integer > res = new Vector < Integer > ( ) ;
printPath ( res , nThNode , 1 ) ; }
int nThNode = 7 ;
printPathToCoverAllNodeUtil ( nThNode ) ; } }
static void shortestLength ( int n , int x [ ] , int y [ ] ) { int answer = 0 ;
int i = 0 ; while ( n != 0 && i < x . length ) {
if ( x [ i ] + y [ i ] > answer ) answer = x [ i ] + y [ i ] ; i ++ ; }
System . out . println ( " Length ▁ - > ▁ " + answer ) ; System . out . println ( " Path ▁ - > ▁ " + " ( ▁ 1 , ▁ " + answer + " ▁ ) " + " and ▁ ( ▁ " + answer + " , ▁ 1 ▁ ) " ) ; }
int n = 4 ;
int x [ ] = new int [ ] { 1 , 4 , 2 , 1 } ; int y [ ] = new int [ ] { 4 , 1 , 1 , 2 } ; shortestLength ( n , x , y ) ; } }
static void FindPoints ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 , int x4 , int y4 ) {
int x5 = Math . max ( x1 , x3 ) ; int y5 = Math . max ( y1 , y3 ) ;
int x6 = Math . min ( x2 , x4 ) ; int y6 = Math . min ( y2 , y4 ) ;
if ( x5 > x6 y5 > y6 ) { System . out . println ( " No ▁ intersection " ) ; return ; } System . out . print ( " ( " + x5 + " , ▁ " + y5 + " ) ▁ " ) ; System . out . print ( " ( " + x6 + " , ▁ " + y6 + " ) ▁ " ) ;
int x7 = x5 ; int y7 = y6 ; System . out . print ( " ( " + x7 + " , ▁ " + y7 + " ) ▁ " ) ;
int x8 = x6 ; int y8 = y5 ; System . out . print ( " ( " + x8 + " , ▁ " + y8 + " ) ▁ " ) ; }
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x3 = 2 , y3 = 3 , x4 = 7 , y4 = 9 ;
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) ; } }
static double area ( double a , double b , double c ) { double d = Math . abs ( ( c * c ) / ( 2 * a * b ) ) ; return d ; }
public static void main ( String [ ] args ) { double a = - 2 , b = 4 , c = 3 ; System . out . println ( area ( a , b , c ) ) ; } }
static ArrayList < Integer > addToArrayForm ( ArrayList < Integer > A , int K ) {
ArrayList < Integer > v = new ArrayList < Integer > ( ) ; ArrayList < Integer > ans = new ArrayList < Integer > ( ) ;
int rem = 0 ; int i = 0 ;
for ( i = A . size ( ) - 1 ; i >= 0 ; i -- ) {
int my = A . get ( i ) + K % 10 + rem ; if ( my > 9 ) {
rem = 1 ;
v . add ( my % 10 ) ; } else { v . add ( my ) ; rem = 0 ; } K = K / 10 ; }
while ( K > 0 ) {
int my = K % 10 + rem ; v . add ( my % 10 ) ;
if ( my / 10 > 0 ) rem = 1 ; else rem = 0 ; K = K / 10 ; } if ( rem > 0 ) v . add ( rem ) ;
for ( int j = v . size ( ) - 1 ; j >= 0 ; j -- ) ans . add ( v . get ( j ) ) ; return ans ; }
public static void main ( String [ ] args ) { ArrayList < Integer > A = new ArrayList < Integer > ( ) ; A . add ( 2 ) ; A . add ( 7 ) ; A . add ( 4 ) ; int K = 181 ; ArrayList < Integer > ans = addToArrayForm ( A , K ) ;
for ( int i = 0 ; i < ans . size ( ) ; i ++ ) System . out . print ( ans . get ( i ) ) ; } }
static int findThirdDigit ( int n ) {
if ( n < 3 ) return 0 ;
return ( n & 1 ) > 0 ? 1 : 6 ; }
public static void main ( String args [ ] ) { int n = 7 ; System . out . println ( findThirdDigit ( n ) ) ; } }
static double getProbability ( int a , int b , int c , int d ) {
double p = ( double ) a / ( double ) b ; double q = ( double ) c / ( double ) d ;
double ans = p * ( 1 / ( 1 - ( 1 - q ) * ( 1 - p ) ) ) ; return ans ; }
public static void main ( String [ ] args ) { int a = 1 , b = 2 , c = 10 , d = 11 ; System . out . printf ( " % .5f " , getProbability ( a , b , c , d ) ) ; } }
static boolean isPalindrome ( int n ) {
int divisor = 1 ; while ( n / divisor >= 10 ) divisor *= 10 ; while ( n != 0 ) { int leading = n / divisor ; int trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = ( n % divisor ) / 10 ;
divisor = divisor / 100 ; } return true ; }
static int largestPalindrome ( int [ ] A , int n ) { int currentMax = - 1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( A [ i ] > currentMax && isPalindrome ( A [ i ] ) ) currentMax = A [ i ] ; }
return currentMax ; }
public static void main ( String [ ] args ) { int [ ] A = { 1 , 232 , 54545 , 999991 } ; int n = A . length ;
System . out . println ( largestPalindrome ( A , n ) ) ; } }
public static long getFinalElement ( long n ) { long finalNum ; for ( finalNum = 2 ; finalNum * 2 <= n ; finalNum *= 2 ) ; return finalNum ; }
public static void main ( String [ ] args ) { int N = 12 ; System . out . println ( getFinalElement ( N ) ) ; } }
static boolean isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
static boolean isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
static long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
public static void main ( String [ ] args ) { int L = 110 , R = 1130 ; System . out . println ( sumOfAllPalindrome ( L , R ) ) ; } }
static double calculateAlternateSum ( int n ) { if ( n <= 0 ) return 0 ; int fibo [ ] = new int [ n + 1 ] ; fibo [ 0 ] = 0 ; fibo [ 1 ] = 1 ;
double sum = Math . pow ( fibo [ 0 ] , 2 ) + Math . pow ( fibo [ 1 ] , 2 ) ;
for ( int i = 2 ; i <= n ; i ++ ) { fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] ;
if ( i % 2 == 0 ) sum -= fibo [ i ] ;
else sum += fibo [ i ] ; }
return sum ; }
int n = 8 ;
System . out . println ( " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " + n + " ▁ terms : ▁ " + calculateAlternateSum ( n ) ) ; } }
static int getValue ( int n ) { int i = 0 , k = 1 ; while ( i < n ) { i = i + k ; k = k * 2 ; } return k / 2 ; }
int n = 9 ;
System . out . println ( getValue ( n ) ) ;
n = 1025 ;
System . out . println ( getValue ( n ) ) ; } }
static void countDigits ( double val , long [ ] arr ) { while ( ( long ) val > 0 ) { long digit = ( long ) val % 10 ; arr [ ( int ) digit ] ++ ; val = ( long ) val / 10 ; } return ; } static void countFrequency ( int x , int n ) {
long [ ] freq_count = new long [ 10 ] ;
for ( int i = 1 ; i <= n ; i ++ ) {
double val = Math . pow ( ( double ) x , ( double ) i ) ;
countDigits ( val , freq_count ) ; }
for ( int i = 0 ; i <= 9 ; i ++ ) { System . out . print ( freq_count [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int x = 15 , n = 3 ; countFrequency ( x , n ) ; } }
static int countSolutions ( int a ) { int count = 0 ;
for ( int i = 0 ; i <= a ; i ++ ) { if ( a == ( i + ( a ^ i ) ) ) count ++ ; } return count ; }
public static void main ( String [ ] args ) { int a = 3 ; System . out . println ( countSolutions ( a ) ) ; } }
static int countSolutions ( int a ) { int count = Integer . bitCount ( a ) ; count = ( int ) Math . pow ( 2 , count ) ; return count ; }
public static void main ( String [ ] args ) { int a = 3 ; System . out . println ( countSolutions ( a ) ) ; } }
static int calculateAreaSum ( int l , int b ) { int size = 1 ;
int maxSize = Math . min ( l , b ) ; int totalArea = 0 ; for ( int i = 1 ; i <= maxSize ; i ++ ) {
int totalSquares = ( l - size + 1 ) * ( b - size + 1 ) ;
int area = totalSquares * size * size ;
totalArea += area ;
size ++ ; } return totalArea ; }
public static void main ( String [ ] args ) { int l = 4 , b = 3 ; System . out . println ( calculateAreaSum ( l , b ) ) ; } }
class GFG { static long boost_hyperfactorial ( long num ) {
long val = 1 ; for ( int i = 1 ; i <= num ; i ++ ) { val = val * ( long ) Math . pow ( i , i ) ; }
return val ; }
public static void main ( String args [ ] ) { int num = 5 ; System . out . println ( boost_hyperfactorial ( num ) ) ; } }
static int boost_hyperfactorial ( int num ) {
int val = 1 ; for ( int i = 1 ; i <= num ; i ++ ) { for ( int j = 1 ; j <= i ; j ++ ) {
val *= i ; } }
return val ; }
public static void main ( String [ ] args ) { int num = 5 ; System . out . println ( boost_hyperfactorial ( num ) ) ; } }
import java . io . * ; class GFG { static int subtractOne ( int x ) { int m = 1 ;
while ( ! ( ( x & m ) > 0 ) ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
public static void main ( String [ ] args ) { System . out . println ( subtractOne ( 13 ) ) ; } }
import java . io . * ; class GFG { static int rows = 3 ; static int cols = 3 ;
static void meanVector ( int mat [ ] [ ] ) { System . out . print ( " [ ▁ " ) ;
for ( int i = 0 ; i < rows ; i ++ ) {
double mean = 0.00 ;
int sum = 0 ; for ( int j = 0 ; j < cols ; j ++ ) sum += mat [ j ] [ i ] ; mean = sum / rows ; System . out . print ( ( int ) mean + " ▁ " ) ; } System . out . print ( " ] " ) ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 2 , 3 } , { 4 , 5 , 6 } , { 7 , 8 , 9 } } ; meanVector ( mat ) ; } }
static List < Integer > primeFactors ( int n ) { List < Integer > res = new ArrayList < Integer > ( ) ; if ( n % 2 == 0 ) { while ( n % 2 == 0 ) n = n / 2 ; res . add ( 2 ) ; }
for ( int i = 3 ; i <= Math . sqrt ( n ) ; i = i + 2 ) {
if ( n % i == 0 ) { while ( n % i == 0 ) n = n / i ; res . add ( i ) ; } }
if ( n > 2 ) res . add ( n ) ; return res ; }
static boolean isHoax ( int n ) {
List < Integer > pf = primeFactors ( n ) ;
if ( pf . get ( 0 ) == n ) return false ;
int all_pf_sum = 0 ; for ( int i = 0 ; i < pf . size ( ) ; i ++ ) {
int pf_sum ; for ( pf_sum = 0 ; pf . get ( i ) > 0 ; pf_sum += pf . get ( i ) % 10 , pf . set ( i , pf . get ( i ) / 10 ) ) ; all_pf_sum += pf_sum ; }
int sum_n ; for ( sum_n = 0 ; n > 0 ; sum_n += n % 10 , n /= 10 ) ;
return sum_n == all_pf_sum ; }
public static void main ( String args [ ] ) { int n = 84 ; if ( isHoax ( n ) ) System . out . print ( "A Hoax NumberNEW_LINE"); else System . out . print ( "Not a Hoax NumberNEW_LINE"); } }
static int modInverse ( int a , int prime ) { a = a % prime ; for ( int x = 1 ; x < prime ; x ++ ) if ( ( a * x ) % prime == 1 ) return x ; return - 1 ; } static void printModIverses ( int n , int prime ) { for ( int i = 1 ; i <= n ; i ++ ) System . out . print ( modInverse ( i , prime ) + " ▁ " ) ; }
public static void main ( String args [ ] ) { int n = 10 , prime = 17 ; printModIverses ( n , prime ) ; } }
static int minOp ( int num ) {
int rem ; int count = 0 ;
while ( num > 0 ) { rem = num % 10 ; if ( ! ( rem == 3 rem == 8 ) ) count ++ ; num /= 10 ; } return count ; }
public static void main ( String [ ] args ) { int num = 234198 ; System . out . print ( " Minimum ▁ Operations ▁ = " + minOp ( num ) ) ; } }
static int sumOfDigits ( int a ) { int sum = 0 ; while ( a != 0 ) { sum += a % 10 ; a /= 10 ; } return sum ; }
static int findMax ( int x ) {
int b = 1 , ans = x ;
while ( x != 0 ) {
int cur = ( x - 1 ) * b + ( b - 1 ) ;
if ( sumOfDigits ( cur ) > sumOfDigits ( ans ) || ( sumOfDigits ( cur ) == sumOfDigits ( ans ) && cur > ans ) ) ans = cur ;
x /= 10 ; b *= 10 ; } return ans ; }
public static void main ( String [ ] args ) { int n = 521 ; System . out . println ( findMax ( n ) ) ; } }
static int median ( int a [ ] , int l , int r ) { int n = r - l + 1 ; n = ( n + 1 ) / 2 - 1 ; return n + l ; }
static int IQR ( int [ ] a , int n ) { Arrays . sort ( a ) ;
int mid_index = median ( a , 0 , n ) ;
int Q1 = a [ median ( a , 0 , mid_index ) ] ;
int Q3 = a [ mid_index + median ( a , mid_index + 1 , n ) ] ;
return ( Q3 - Q1 ) ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 } ; int n = a . length ; System . out . println ( IQR ( a , n ) ) ; } }
static boolean isPalindrome ( int n ) {
int divisor = 1 ; while ( n / divisor >= 10 ) divisor *= 10 ; while ( n != 0 ) { int leading = n / divisor ; int trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = ( n % divisor ) / 10 ;
divisor = divisor / 100 ; } return true ; }
static int largestPalindrome ( int [ ] A , int n ) {
Arrays . sort ( A ) ; for ( int i = n - 1 ; i >= 0 ; -- i ) {
if ( isPalindrome ( A [ i ] ) ) return A [ i ] ; }
return - 1 ; }
public static void main ( String [ ] args ) { int [ ] A = { 1 , 232 , 54545 , 999991 } ; int n = A . length ;
System . out . println ( largestPalindrome ( A , n ) ) ; } }
static int findSum ( int n , int a , int b ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
public static void main ( String [ ] args ) { int n = 10 , a = 3 , b = 5 ; System . out . println ( findSum ( n , a , b ) ) ; } }
class GFG { static int subtractOne ( int x ) { return ( ( x << 1 ) + ( ~ x ) ) ; } public static void main ( String [ ] args ) { System . out . printf ( " % d " , subtractOne ( 13 ) ) ; } }
public static int pell ( int n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( pell ( n ) ) ; } }
static long LCM ( int arr [ ] , int n ) {
int max_num = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( max_num < arr [ i ] ) { max_num = arr [ i ] ; } }
long res = 1 ;
while ( x <= max_num ) {
Vector < Integer > indexes = new Vector < > ( ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] % x == 0 ) { indexes . add ( indexes . size ( ) , j ) ; } }
if ( indexes . size ( ) >= 2 ) {
for ( int j = 0 ; j < indexes . size ( ) ; j ++ ) { arr [ indexes . get ( j ) ] = arr [ indexes . get ( j ) ] / x ; } res = res * x ; } else { x ++ ; } }
for ( int i = 0 ; i < n ; i ++ ) { res = res * arr [ i ] ; } return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 } ; int n = arr . length ; System . out . println ( LCM ( arr , n ) ) ; } }
import java . util . * ; class GFG { static int MAX = 10000 ;
static ArrayList < Integer > primes = new ArrayList < Integer > ( ) ;
static void sieveSundaram ( ) {
boolean [ ] marked = new boolean [ MAX / 2 + 100 ] ;
for ( int i = 1 ; i <= ( Math . sqrt ( MAX ) - 1 ) / 2 ; i ++ ) for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= MAX / 2 ; j = j + 2 * i + 1 ) marked [ j ] = true ;
primes . add ( 2 ) ;
for ( int i = 1 ; i <= MAX / 2 ; i ++ ) if ( marked [ i ] == false ) primes . add ( 2 * i + 1 ) ; }
static void findPrimes ( int n ) {
if ( n <= 2 n % 2 != 0 ) { System . out . println ( " Invalid ▁ Input ▁ " ) ; return ; }
for ( int i = 0 ; primes . get ( i ) <= n / 2 ; i ++ ) {
int diff = n - primes . get ( i ) ;
if ( primes . contains ( diff ) ) {
System . out . println ( primes . get ( i ) + " ▁ + ▁ " + diff + " ▁ = ▁ " + n ) ; return ; } } }
sieveSundaram ( ) ;
findPrimes ( 4 ) ; findPrimes ( 38 ) ; findPrimes ( 100 ) ; } }
static int kPrimeFactor ( int n , int k ) {
while ( n % 2 == 0 ) { k -- ; n = n / 2 ; if ( k == 0 ) return 2 ; }
for ( int i = 3 ; i <= Math . sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { if ( k == 1 ) return i ; k -- ; n = n / i ; } }
if ( n > 2 && k == 1 ) return n ; return - 1 ; }
public static void main ( String args [ ] ) { int n = 12 , k = 3 ; System . out . println ( kPrimeFactor ( n , k ) ) ; n = 14 ; k = 3 ; System . out . println ( kPrimeFactor ( n , k ) ) ; } }
class GFG { static int MAX = 10001 ;
static void sieveOfEratosthenes ( int [ ] s ) {
boolean [ ] prime = new boolean [ MAX + 1 ] ;
for ( int i = 2 ; i <= MAX ; i += 2 ) s [ i ] = 2 ;
for ( int i = 3 ; i <= MAX ; i += 2 ) { if ( prime [ i ] == false ) {
s [ i ] = i ;
for ( int j = i ; j * i <= MAX ; j += 2 ) { if ( prime [ i * j ] == false ) { prime [ i * j ] = true ;
s [ i * j ] = i ; } } } } }
static int kPrimeFactor ( int n , int k , int [ ] s ) {
while ( n > 1 ) { if ( k == 1 ) return s [ n ] ;
k -- ;
n /= s [ n ] ; } return - 1 ; }
int [ ] s = new int [ MAX + 1 ] ; sieveOfEratosthenes ( s ) ; int n = 12 , k = 3 ; System . out . println ( kPrimeFactor ( n , k , s ) ) ; n = 14 ; k = 3 ; System . out . println ( kPrimeFactor ( n , k , s ) ) ; } }
static boolean squareRootExists ( int n , int p ) { n = n % p ;
for ( int x = 2 ; x < p ; x ++ ) if ( ( x * x ) % p == n ) return true ; return false ; }
public static void main ( String [ ] args ) { int p = 7 ; int n = 2 ; if ( squareRootExists ( n , p ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static int Largestpower ( int n , int p ) {
int ans = 0 ;
while ( n > 0 ) { n /= p ; ans += n ; } return ans ; }
public static void main ( String [ ] args ) { int n = 10 ; int p = 3 ; System . out . println ( " ▁ The ▁ largest ▁ power ▁ of ▁ " + p + " ▁ that ▁ divides ▁ " + n + " ! ▁ is ▁ " + Largestpower ( n , p ) ) ; } }
class Factorial { int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
public static void main ( String args [ ] ) { Factorial obj = new Factorial ( ) ; int num = 5 ; System . out . println ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + obj . factorial ( num ) ) ; } }
public static int reverseBits ( int n ) { int rev = 0 ;
while ( n > 0 ) {
rev <<= 1 ;
if ( ( int ) ( n & 1 ) == 1 ) rev ^= 1 ;
n >>= 1 ; }
return rev ; }
public static void main ( String [ ] args ) { int n = 11 ; System . out . println ( reverseBits ( n ) ) ; } }
static int countgroup ( int a [ ] , int n ) { int xs = 0 ; for ( int i = 0 ; i < n ; i ++ ) xs = xs ^ a [ i ] ;
if ( xs == 0 ) return ( 1 << ( n - 1 ) ) - 1 ; return 0 ; }
public static void main ( String args [ ] ) { int a [ ] = { 1 , 2 , 3 } ; int n = a . length ; System . out . println ( countgroup ( a , n ) ) ; } }
static int bitExtracted ( int number , int k , int p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int number = 171 , k = 5 , p = 2 ; System . out . println ( " The ▁ extracted ▁ number ▁ is ▁ " + bitExtracted ( number , k , p ) ) ; } }
static boolean isAMultipleOf4 ( int n ) {
if ( ( n & 3 ) == 0 ) return true ;
return false ; }
public static void main ( String [ ] args ) { int n = 16 ; System . out . println ( isAMultipleOf4 ( n ) ? " Yes " : " No " ) ; } }
import java . io . * ; class GFG { public static int square ( int n ) {
if ( n < 0 ) n = - n ;
int res = n ;
for ( int i = 1 ; i < n ; i ++ ) res += n ; return res ; }
public static void main ( String [ ] args ) { for ( int n = 1 ; n <= 5 ; n ++ ) System . out . println ( " n ▁ = ▁ " + n + " , ▁ n ^ 2 ▁ = ▁ " + square ( n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int PointInKSquares ( int n , int a [ ] , int k ) { Arrays . sort ( a ) ; return a [ n - k ] ; }
public static void main ( String [ ] args ) { int k = 2 ; int [ ] a = { 1 , 2 , 3 , 4 } ; int n = a . length ; int x = PointInKSquares ( n , a , k ) ; System . out . println ( " ( " + x + " , ▁ " + x + " ) " ) ; } }
static long answer ( int n ) {
int [ ] dp = new int [ 10 ] ;
int [ ] prev = new int [ 10 ] ;
if ( n == 1 ) return 10 ;
for ( int j = 0 ; j <= 9 ; j ++ ) dp [ j ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= 9 ; j ++ ) { prev [ j ] = dp [ j ] ; } for ( int j = 0 ; j <= 9 ; j ++ ) {
if ( j == 0 ) dp [ j ] = prev [ j + 1 ] ;
else if ( j == 9 ) dp [ j ] = prev [ j - 1 ] ;
else dp [ j ] = prev [ j - 1 ] + prev [ j + 1 ] ; } }
long sum = 0 ; for ( int j = 1 ; j <= 9 ; j ++ ) sum += dp [ j ] ; return sum ; }
public static void main ( String [ ] args ) { int n = 2 ; System . out . println ( answer ( n ) ) ; } }
class GFG1 { static int MAX = 100000 ;
static long catalan [ ] = new long [ MAX ] ;
static void catalanDP ( long n ) {
catalan [ 0 ] = catalan [ 1 ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { catalan [ i ] = 0 ; for ( int j = 0 ; j < i ; j ++ ) { catalan [ i ] += catalan [ j ] * catalan [ i - j - 1 ] ; } } }
static int CatalanSequence ( int arr [ ] , int n ) {
catalanDP ( n ) ; HashSet < Integer > s = new HashSet < Integer > ( ) ;
int a = 1 , b = 1 ; int c ;
s . add ( a ) ; if ( n >= 2 ) { s . add ( b ) ; } for ( int i = 2 ; i < n ; i ++ ) { s . add ( ( int ) catalan [ i ] ) ; } for ( int i = 0 ; i < n ; i ++ ) {
if ( s . contains ( arr [ i ] ) ) { s . remove ( arr [ i ] ) ; } }
return s . size ( ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 2 , 5 , 41 } ; int n = arr . length ; System . out . print ( CatalanSequence ( arr , n ) ) ; } }
static int solve ( int [ ] a , int n ) { int max1 = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( Math . abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
static public void main ( String [ ] args ) { int [ ] arr = { - 1 , 2 , 3 , - 4 , - 10 , 22 } ; int size = arr . length ; System . out . println ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
static int solve ( int a [ ] , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . abs ( min1 - max1 ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { - 1 , 2 , 3 , 4 , - 10 } ; int size = arr . length ; System . out . println ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
static int minElements ( int arr [ ] , int n ) {
int halfSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = halfSum / 2 ;
Arrays . sort ( arr ) ; int res = 0 , curr_sum = 0 ; for ( int i = n - 1 ; i >= 0 ; i -- ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 1 , 7 , 1 } ; int n = arr . length ; System . out . println ( minElements ( arr , n ) ) ; } }
static int minCost ( int N , int P , int Q ) {
int cost = 0 ;
while ( N > 0 ) { if ( ( N & 1 ) > 0 ) { cost += P ; N -- ; } else { int temp = N / 2 ;
if ( temp * P > Q ) cost += Q ;
else cost += P * temp ; N /= 2 ; } }
return cost ; }
public static void main ( String [ ] args ) { int N = 9 , P = 5 , Q = 1 ; System . out . println ( minCost ( N , P , Q ) ) ; } }
