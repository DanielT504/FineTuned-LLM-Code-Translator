static double Conversion ( double centi ) { double pixels = ( 96 * centi ) / 2.54 ; Console . WriteLine ( pixels ) ; return 0 ; }
public static void Main ( ) { double centi = 15 ; Conversion ( centi ) ; } }
static int maxOfMin ( int [ ] a , int n , int S ) {
int mi = int . MaxValue ;
int s1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { s1 += a [ i ] ; mi = Math . Min ( a [ i ] , mi ) ; }
if ( s1 < S ) return - 1 ;
if ( s1 == S ) return 0 ;
int low = 0 ;
int high = mi ;
int ans = 0 ;
while ( low <= high ) { int mid = ( low + high ) / 2 ;
if ( s1 - ( mid * n ) >= S ) { ans = mid ; low = mid + 1 ; }
else high = mid - 1 ; }
return ans ; }
public static void Main ( ) { int [ ] a = { 10 , 10 , 10 , 10 , 10 } ; int S = 10 ; int n = a . Length ; Console . WriteLine ( maxOfMin ( a , n , S ) ) ; } }
public static void Alphabet_N_Pattern ( int N ) { int index , side_index ;
int Right = 1 , Left = 1 , Diagonal = 2 ;
for ( index = 0 ; index < N ; index ++ ) {
Console . Write ( Left ++ ) ;
for ( side_index = 0 ; side_index < 2 * ( index ) ; side_index ++ ) Console . Write ( " ▁ " ) ;
if ( index != 0 && index != N - 1 ) Console . Write ( Diagonal ++ ) ; else Console . Write ( " ▁ " ) ;
for ( side_index = 0 ; side_index < 2 * ( N - index - 1 ) ; side_index ++ ) Console . Write ( " ▁ " ) ;
Console . Write ( Right ++ ) ; Console . Write ( " STRNEWLINE " ) ; } }
int Size = 6 ;
Alphabet_N_Pattern ( Size ) ; } }
class GFG { public int isSumDivides ( int N ) { int temp = N , sum = 0 ;
while ( temp > 0 ) { sum += temp % 10 ; temp /= 10 ; } if ( N % sum == 0 ) return 1 ; else return 0 ; }
public static void Main ( ) { GFG g = new GFG ( ) ; int N = 12 ; if ( g . isSumDivides ( N ) > 0 ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
static int sum ( int N ) { int S1 , S2 , S3 ; S1 = ( ( N / 3 ) ) * ( 2 * 3 + ( N / 3 - 1 ) * 3 ) / 2 ; S2 = ( ( N / 4 ) ) * ( 2 * 4 + ( N / 4 - 1 ) * 4 ) / 2 ; S3 = ( ( N / 12 ) ) * ( 2 * 12 + ( N / 12 - 1 ) * 12 ) / 2 ; return S1 + S2 - S3 ; }
public static void Main ( ) { int N = 20 ; Console . WriteLine ( sum ( 12 ) ) ; } }
static int nextGreater ( int N ) { int power_of_2 = 1 , shift_count = 0 ;
while ( true ) {
if ( ( ( N >> shift_count ) & 1 ) % 2 == 0 ) break ;
shift_count ++ ;
power_of_2 = power_of_2 * 2 ; }
return ( N + power_of_2 ) ; }
public static void Main ( ) { int N = 11 ;
Console . WriteLine ( " The ▁ next ▁ number ▁ is ▁ = ▁ " + nextGreater ( N ) ) ; } }
static void printTetra ( int n ) { int [ ] dp = new int [ n + 5 ] ;
dp [ 0 ] = 0 ; dp [ 1 ] = dp [ 2 ] = 1 ; dp [ 3 ] = 2 ; for ( int i = 4 ; i <= n ; i ++ ) dp [ i ] = dp [ i - 1 ] + dp [ i - 2 ] + dp [ i - 3 ] + dp [ i - 4 ] ; System . Console . WriteLine ( dp [ n ] ) ; }
static void Main ( ) { int n = 10 ; printTetra ( n ) ; } }
static int maxSum1 ( int [ ] arr , int n ) { int [ ] dp = new int [ n ] ; int maxi = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
dp [ i ] = arr [ i ] ;
if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( int i = 2 ; i < n - 1 ; i ++ ) {
for ( int j = 0 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < dp [ j ] + arr [ i ] ) { dp [ i ] = dp [ j ] + arr [ i ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; }
static int maxSum2 ( int [ ] arr , int n ) { int [ ] dp = new int [ n ] ; int maxi = 0 ; for ( int i = 1 ; i < n ; i ++ ) { dp [ i ] = arr [ i ] ; if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( int i = 3 ; i < n ; i ++ ) {
for ( int j = 1 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < arr [ i ] + dp [ j ] ) { dp [ i ] = arr [ i ] + dp [ j ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; } static int findMaxSum ( int [ ] arr , int n ) { int t = Math . Max ( maxSum1 ( arr , n ) , maxSum2 ( arr , n ) ) ; return t ; }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 1 } ; int n = arr . Length ; Console . WriteLine ( findMaxSum ( arr , n ) ) ; } }
static int permutationCoeff ( int n , int k ) { int [ , ] P = new int [ n + 2 , k + 2 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= Math . Min ( i , k ) ; j ++ ) {
if ( j == 0 ) P [ i , j ] = 1 ;
else P [ i , j ] = P [ i - 1 , j ] + ( j * P [ i - 1 , j - 1 ] ) ;
P [ i , j + 1 ] = 0 ; } } return P [ n , k ] ; }
public static void Main ( ) { int n = 10 , k = 2 ; Console . WriteLine ( " Value ▁ of ▁ P ( ▁ " + n + " , " + k + " ) " + " ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
static int permutationCoeff ( int n , int k ) { int [ ] fact = new int [ n + 1 ] ;
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = i * fact [ i - 1 ] ;
return fact [ n ] / fact [ n - k ] ; }
static public void Main ( ) { int n = 10 , k = 2 ; Console . WriteLine ( " Value ▁ of " + " ▁ P ( ▁ " + n + " , ▁ " + k + " ) ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
static int no_of_ways ( string s ) { int n = s . Length ;
int count_left = 0 , count_right = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { if ( s [ i ] == s [ 0 ] ) { ++ count_left ; } else break ; }
for ( int i = n - 1 ; i >= 0 ; -- i ) { if ( s [ i ] == s [ n - 1 ] ) { ++ count_right ; } else break ; }
if ( s [ 0 ] == s [ n - 1 ] ) return ( ( count_left + 1 ) * ( count_right + 1 ) ) ;
else return ( + count_right + 1 ) ; }
public static void Main ( ) { string s = " geeksforgeeks " ; Console . WriteLine ( no_of_ways ( s ) ) ; } }
static void preCompute ( int n , string s , int [ ] pref ) { pref [ 0 ] = 0 ; for ( int i = 1 ; i < n ; i ++ ) { pref [ i ] = pref [ i - 1 ] ; if ( s [ i - 1 ] == s [ i ] ) pref [ i ] ++ ; } }
static int query ( int [ ] pref , int l , int r ) { return pref [ r ] - pref [ l ] ; }
public static void Main ( ) { string s = " ggggggg " ; int n = s . Length ; int [ ] pref = new int [ n ] ; preCompute ( n , s , pref ) ;
int l = 1 ; int r = 2 ; Console . WriteLine ( query ( pref , l , r ) ) ;
l = 1 ; r = 5 ; Console . WriteLine ( query ( pref , l , r ) ) ; } }
static String findDirection ( String s ) { int count = 0 ; String d = " " ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( s [ 0 ] == ' STRNEWLINE ' ) return null ; if ( s [ i ] == ' L ' ) count -- ; else { if ( s [ i ] == ' R ' ) count ++ ; } }
if ( count > 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == 1 ) d = " E " ; else if ( count % 4 == 2 ) d = " S " ; else if ( count % 4 == 3 ) d = " W " ; }
if ( count < 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == - 1 ) d = " W " ; else if ( count % 4 == - 2 ) d = " S " ; else if ( count % 4 == - 3 ) d = " E " ; } return d ; }
public static void Main ( ) { String s = " LLRLRRL " ; Console . WriteLine ( findDirection ( s ) ) ; s = " LL " ; Console . WriteLine ( findDirection ( s ) ) ; } }
static void encode ( String s , int k ) {
String newS = " " ;
for ( int i = 0 ; i < s . Length ; ++ i ) {
int val = s [ i ] ;
int dup = k ;
if ( val + k > 122 ) { k -= ( 122 - val ) ; k = k % 26 ; newS += ( char ) ( 96 + k ) ; } else { newS += ( char ) ( 96 + k ) ; } k = dup ; }
Console . Write ( newS ) ; }
public static void Main ( ) { String str = " abc " ; int k = 28 ;
encode ( str , k ) ; } }
static bool isVowel ( char x ) { if ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) return true ; else return false ; }
static String updateSandwichedVowels ( String a ) { int n = a . Length ;
String updatedString = " " ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i == 0 i == n - 1 ) { updatedString += a [ i ] ; continue ; }
if ( ( isVowel ( a [ i ] ) ) == true && isVowel ( a [ i - 1 ] ) == false && isVowel ( a [ i + 1 ] ) == false ) { continue ; }
updatedString += a [ i ] ; } return updatedString ; }
public static void Main ( ) { String str = " geeksforgeeks " ;
String updatedString = updateSandwichedVowels ( str ) ; Console . WriteLine ( updatedString ) ; }
static int findNumbers ( int n , int w ) { int x = 0 , sum = 0 ;
if ( w >= 0 && w <= 8 ) {
x = 9 - w ; }
else if ( w >= - 9 && w <= - 1 ) {
x = 10 + w ; } sum = ( int ) Math . Pow ( 10 , n - 2 ) ; sum = ( x * sum ) ; return sum ; }
n = 3 ; w = 4 ;
Console . WriteLine ( findNumbers ( n , w ) ) ; } }
using System ; class GFG { static int MaximumHeight ( int [ ] a , int n ) { int result = 1 ; for ( int i = 1 ; i <= n ; ++ i ) {
int y = ( i * ( i + 1 ) ) / 2 ;
if ( y < n ) result = i ;
else break ; } return result ; }
static public void Main ( ) { int [ ] arr = { 40 , 100 , 20 , 30 } ; int n = arr . Length ; Console . WriteLine ( MaximumHeight ( arr , n ) ) ; } }
using System ; using System . Collections ; class GFG { static int findK ( int n , int k ) { ArrayList a = new ArrayList ( n ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( i % 2 == 1 ) a . Add ( i ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( i % 2 == 0 ) a . Add ( i ) ; return ( int ) ( a [ k - 1 ] ) ; }
static void Main ( ) { int n = 10 , k = 3 ; Console . WriteLine ( findK ( n , k ) ) ; } }
using System ; class GFG {
static int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; } public static void Main ( ) { int num = 5 ; Console . WriteLine ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + factorial ( num ) ) ; } }
public static int pell ( int n ) { if ( n <= 2 ) return n ; int a = 1 ; int b = 2 ; int c ; for ( int i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
public static void Main ( ) { int n = 4 ; Console . Write ( pell ( n ) ) ; } }
static bool isMultipleOf10 ( int n ) { if ( n % 15 == 0 ) return true ; return false ; }
public static void Main ( ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
static int countOddPrimeFactors ( int n ) { int result = 1 ;
while ( n % 2 == 0 ) n /= 2 ;
for ( int i = 3 ; i * i <= n ; i += 2 ) { int divCount = 0 ;
while ( n % i == 0 ) { n /= i ; ++ divCount ; } result *= divCount + 1 ; }
if ( n > 2 ) result *= 2 ; return result ; } static int politness ( int n ) { return countOddPrimeFactors ( n ) - 1 ; }
public static void Main ( ) { int n = 90 ; Console . WriteLine ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; n = 15 ; Console . WriteLine ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; } }
using System ; using System . Collections ; class GFG { static int MAX = 1000000 ;
static ArrayList primes = new ArrayList ( ) ;
static void Sieve ( ) { int n = MAX ;
int nNew = ( int ) Math . Sqrt ( n ) ;
int [ ] marked = new int [ n / 2 + 500 ] ;
for ( int i = 1 ; i <= ( nNew - 1 ) / 2 ; i ++ ) for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= n / 2 ; j = j + 2 * i + 1 ) marked [ j ] = 1 ;
primes . Add ( 2 ) ;
for ( int i = 1 ; i <= n / 2 ; i ++ ) if ( marked [ i ] == 0 ) primes . Add ( 2 * i + 1 ) ; }
static int binarySearch ( int left , int right , int n ) { if ( left <= right ) { int mid = ( left + right ) / 2 ;
if ( mid == 0 mid == primes . Count - 1 ) return ( int ) primes [ mid ] ;
if ( ( int ) primes [ mid ] == n ) return ( int ) primes [ mid - 1 ] ;
if ( ( int ) primes [ mid ] < n && ( int ) primes [ mid + 1 ] > n ) return ( int ) primes [ mid ] ; if ( n < ( int ) primes [ mid ] ) return binarySearch ( left , mid - 1 , n ) ; else return binarySearch ( mid + 1 , right , n ) ; } return 0 ; }
static void Main ( ) { Sieve ( ) ; int n = 17 ; Console . WriteLine ( binarySearch ( 0 , primes . Count - 1 , n ) ) ; } }
static int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
public static void Main ( ) { int num = 5 ; Console . WriteLine ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + factorial ( 5 ) ) ; } }
static int printKDistinct ( int [ ] arr , int n , int k ) { int dist_count = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return - 1 ; }
public static void Main ( ) { int [ ] ar = { 1 , 2 , 1 , 3 , 4 , 2 } ; int n = ar . Length ; int k = 2 ; Console . Write ( printKDistinct ( ar , n , k ) ) ; } }
static int calculate ( int [ ] a , int n ) {
Array . Sort ( a ) ; int count = 1 ; int answer = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( a [ i ] == a [ i - 1 ] ) {
count += 1 ; } else {
answer = answer + ( count * ( count - 1 ) ) / 2 ; count = 1 ; } } answer = answer + ( count * ( count - 1 ) ) / 2 ; return answer ; }
public static void Main ( ) { int [ ] a = { 1 , 2 , 1 , 2 , 4 } ; int n = a . Length ;
Console . WriteLine ( calculate ( a , n ) ) ; } }
static int calculate ( int [ ] a , int n ) {
int maximum = a . Max ( ) ;
int [ ] frequency = new int [ maximum + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
frequency [ a [ i ] ] += 1 ; } int answer = 0 ;
for ( int i = 0 ; i < ( maximum ) + 1 ; i ++ ) {
answer = answer + frequency [ i ] * ( frequency [ i ] - 1 ) ; } return answer / 2 ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 2 , 1 , 2 , 4 } ; int n = a . Length ;
Console . WriteLine ( calculate ( a , n ) ) ; } }
static int findSubArray ( int [ ] arr , int n ) { int sum = 0 ; int maxsize = - 1 , startindex = 0 ; int endindex = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? - 1 : 1 ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] == 0 ) sum += - 1 ; else sum += 1 ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } endindex = startindex + maxsize - 1 ; if ( maxsize == - 1 ) Console . WriteLine ( " No ▁ such ▁ subarray " ) ; else Console . WriteLine ( startindex + " ▁ to ▁ " + endindex ) ; return maxsize ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int size = arr . Length ; findSubArray ( arr , size ) ; } }
static int findMax ( int [ ] arr , int low , int high ) {
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid == 0 && arr [ mid ] > arr [ mid + 1 ] ) return arr [ mid ] ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] && mid > 0 && arr [ mid ] > arr [ mid - 1 ] ) { return arr [ mid ] ; }
if ( arr [ low ] > arr [ mid ] ) { return findMax ( arr , low , mid - 1 ) ; } else { return findMax ( arr , mid + 1 , high ) ; } }
public static void Main ( ) { int [ ] arr = { 6 , 5 , 1 , 2 , 3 , 4 } ; int n = arr . Length ; Console . WriteLine ( findMax ( arr , 0 , n - 1 ) ) ; } }
static int search ( int [ ] arr , int l , int h , int key ) { if ( l > h ) return - 1 ; int mid = ( l + h ) / 2 ; if ( arr [ mid ] == key ) return mid ;
if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
return search ( arr , mid + 1 , h , key ) ; }
if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
public static void Main ( ) { int [ ] arr = { 4 , 5 , 6 , 7 , 8 , 9 , 1 , 2 , 3 } ; int n = arr . Length ; int key = 6 ; int i = search ( arr , 0 , n - 1 , key ) ; if ( i != - 1 ) Console . WriteLine ( " Index : ▁ " + i ) ; else Console . WriteLine ( " Key ▁ not ▁ found " ) ; } }
using System ; class Minimum { static int findMin ( int [ ] arr , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
public static void Main ( ) { int [ ] arr1 = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = arr1 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr1 , 0 , n1 - 1 ) ) ; int [ ] arr2 = { 1 , 2 , 3 , 4 } ; int n2 = arr2 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr2 , 0 , n2 - 1 ) ) ; int [ ] arr3 = { 1 } ; int n3 = arr3 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr3 , 0 , n3 - 1 ) ) ; int [ ] arr4 = { 1 , 2 } ; int n4 = arr4 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr4 , 0 , n4 - 1 ) ) ; int [ ] arr5 = { 2 , 1 } ; int n5 = arr5 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr5 , 0 , n5 - 1 ) ) ; int [ ] arr6 = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = arr6 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr6 , 0 , n1 - 1 ) ) ; int [ ] arr7 = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = arr7 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr7 , 0 , n7 - 1 ) ) ; int [ ] arr8 = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = arr8 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr8 , 0 , n8 - 1 ) ) ; int [ ] arr9 = { 3 , 4 , 5 , 1 , 2 } ; int n9 = arr9 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr9 , 0 , n9 - 1 ) ) ; } }
static void print2Smallest ( int [ ] arr ) { int first , second , arr_size = arr . Length ;
if ( arr_size < 2 ) { Console . Write ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = int . MaxValue ; for ( int i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) = arr [ i ] ; } if ( second == int . MaxValue ) Console . Write ( " There ▁ is ▁ no ▁ second " + " smallest ▁ element " ) ; else . Write ( " The ▁ smallest ▁ element ▁ is ▁ " + first + " ▁ and ▁ second ▁ Smallest " + " ▁ element ▁ is ▁ " + second ) ; }
public static void Main ( ) { int [ ] arr = { 12 , 13 , 1 , 10 , 34 , 1 } ; print2Smallest ( arr ) ; } }
static bool isSubsetSum ( int [ ] arr , int n , int sum ) {
bool [ , ] subset = new bool [ 2 , sum + 1 ] ; for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) {
if ( j == 0 ) subset [ i % 2 , j ] = true ;
else if ( i = = 0 ) subset [ i % 2 , j ] = false ; else if ( arr [ i - 1 ] <= j ) [ i % 2 , j ] = subset [ ( i + 1 ) % 2 , j - arr [ i - 1 ] ] || subset [ ( i + 1 ) % 2 , j ] ; else [ i % 2 , j ] = subset [ ( i + 1 ) % 2 , j ] ; } } return [ n % 2 , sum ] ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 5 } ; int sum = 7 ; int n = arr . Length ; if ( isSubsetSum ( arr , n , sum ) == true ) Console . WriteLine ( " There ▁ exists ▁ a ▁ subset ▁ with " + " given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ exists ▁ with " + " given ▁ sum " ) ; } }
static int findMaxSum ( int [ ] arr , int n ) { int res = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) { int prefix_sum = arr [ i ] ; for ( int j = 0 ; j < i ; j ++ ) prefix_sum += arr [ j ] ; int suffix_sum = arr [ i ] ; for ( int j = n - 1 ; j > i ; j -- ) suffix_sum += arr [ j ] ; if ( prefix_sum == suffix_sum ) res = Math . Max ( res , prefix_sum ) ; } return res ; }
public static void Main ( ) { int [ ] arr = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( findMaxSum ( arr , n ) ) ; } }
static int findMaxSum ( int [ ] arr , int n ) {
int [ ] preSum = new int [ n ] ;
int [ ] suffSum = new int [ n ] ;
int ans = int . MinValue ;
preSum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) preSum [ i ] = preSum [ i - 1 ] + arr [ i ] ;
suffSum [ n - 1 ] = arr [ n - 1 ] ; if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) ans = Math . Max ( ans , preSum [ n - 1 ] ) ; for ( int i = n - 2 ; i >= 0 ; i -- ) { suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] ; if ( suffSum [ i ] == preSum [ i ] ) ans = Math . Max ( ans , preSum [ i ] ) ; } return ans ; }
static public void Main ( ) { int [ ] arr = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( findMaxSum ( arr , n ) ) ; } }
static void findMajority ( int [ ] arr , int n ) { int maxCount = 0 ;
int index = - 1 ; for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) count ++ ; }
if ( count > maxCount ) { maxCount = count ; index = i ; } }
if ( maxCount > n / 2 ) Console . WriteLine ( arr [ index ] ) ; else Console . WriteLine ( " No ▁ Majority ▁ Element " ) ; }
static public void Main ( ) { int [ ] arr = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = arr . Length ;
findMajority ( arr , n ) ; } }
static int findCandidate ( int [ ] a , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
static bool isMajority ( int [ ] a , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) { if ( a [ i ] == cand ) count ++ ; } if ( count > size / 2 ) return true ; else return false ; }
static void printMajority ( int [ ] a , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) Console . Write ( " ▁ " + cand + " ▁ " ) ; else Console . Write ( " No ▁ Majority ▁ Element " ) ; }
public static void Main ( ) { int [ ] a = { 1 , 3 , 3 , 1 , 2 } ; int size = a . Length ;
printMajority ( a , size ) ; } }
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
bool [ , ] subset = new bool [ sum + 1 , n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 , i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i , 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i , j ] = subset [ i , j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i , j ] = subset [ i , j ] || subset [ i - set [ j - 1 ] , j - 1 ] ; } } return subset [ sum , n ] ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static void print_gcd_online ( int n , int m , int [ , ] query , int [ ] arr ) {
int max_gcd = 0 ; int i = 0 ;
for ( i = 0 ; i < n ; i ++ ) max_gcd = gcd ( max_gcd , arr [ i ] ) ;
for ( i = 0 ; i < m ; i ++ ) {
query [ i , 0 ] -- ;
arr [ query [ i , 0 ] ] /= query [ i , 1 ] ;
max_gcd = gcd ( arr [ query [ i , 0 ] ] , max_gcd ) ;
Console . WriteLine ( max_gcd ) ; } }
public static void Main ( ) { int n = 3 ; int m = 3 ; int [ , ] query = new int [ m , 2 ] ; int [ ] arr = new int [ ] { 36 , 24 , 72 } ; query [ 0 , 0 ] = 1 ; query [ 0 , 1 ] = 3 ; query [ 1 , 0 ] = 3 ; query [ 1 , 1 ] = 12 ; query [ 2 , 0 ] = 2 ; query [ 2 , 1 ] = 4 ; print_gcd_online ( n , m , query , arr ) ; } }
using System ; class GFG { static int MAX = 1000000 ;
static bool [ ] prime = new bool [ MAX + 1 ] ;
static int [ ] sum = new int [ MAX + 1 ] ;
static void SieveOfEratosthenes ( ) {
for ( int i = 0 ; i <= MAX ; i ++ ) prime [ i ] = true ; for ( int i = 0 ; i <= MAX ; i ++ ) sum [ i ] = 0 ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( int i = 1 ; i <= MAX ; i ++ ) { if ( prime [ i ] == true ) sum [ i ] = 1 ; sum [ i ] += sum [ i - 1 ] ; } }
SieveOfEratosthenes ( ) ;
int l = 3 , r = 9 ;
int c = ( sum [ r ] - sum [ l - 1 ] ) ;
Console . WriteLine ( " Count : ▁ " + c ) ; } }
static float area ( float r ) {
if ( r < 0 ) return - 1 ;
float area = ( float ) ( 3.14 * Math . Pow ( r / ( 2 * Math . Sqrt ( 2 ) ) , 2 ) ) ; return area ; }
static public void Main ( String [ ] args ) { float a = 5 ; Console . WriteLine ( area ( a ) ) ; } }
using System ; class GFG { static int N = 100005 ;
static bool [ ] prime = new bool [ N ] ; static void SieveOfEratosthenes ( ) { for ( int i = 0 ; i < N ; i ++ ) prime [ i ] = true ; prime [ 1 ] = false ; for ( int p = 2 ; p * p < N ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < N ; i += p ) prime [ i ] = false ; } } }
static int almostPrimes ( int n ) {
int ans = 0 ;
for ( int i = 6 ; i <= n ; i ++ ) {
int c = 0 ; for ( int j = 2 ; j * j <= i ; j ++ ) { if ( i % j == 0 ) {
if ( j * j == i ) { if ( prime [ j ] ) c ++ ; } else { if ( prime [ j ] ) c ++ ; if ( prime [ i / j ] ) c ++ ; } } }
if ( c == 2 ) ans ++ ; } return ans ; }
public static void Main ( ) { SieveOfEratosthenes ( ) ; int n = 21 ; Console . WriteLine ( almostPrimes ( n ) ) ; } }
static int sumOfDigitsSingle ( int x ) { int ans = 0 ; while ( x != 0 ) { ans += x % 10 ; x /= 10 ; } return ans ; }
static int closest ( int x ) { int ans = 0 ; while ( ans * 10 + 9 <= x ) ans = ans * 10 + 9 ; return ans ; } static int sumOfDigitsTwoParts ( int N ) { int A = closest ( N ) ; return sumOfDigitsSingle ( A ) + sumOfDigitsSingle ( N - A ) ; }
public static void Main ( ) { int N = 35 ; Console . Write ( sumOfDigitsTwoParts ( N ) ) ; } }
static bool isPrime ( int p ) {
double checkNumber = Math . Pow ( 2 , p ) - 1 ;
double nextval = 4 % checkNumber ;
for ( int i = 1 ; i < p - 1 ; i ++ ) nextval = ( nextval * nextval - 2 ) % checkNumber ;
return ( nextval == 0 ) ; }
int p = 7 ; double checkNumber = Math . Pow ( 2 , p ) - 1 ; if ( isPrime ( p ) ) Console . WriteLine ( ( int ) checkNumber + " ▁ is ▁ Prime . " ) ; else Console . WriteLine ( ( int ) checkNumber + " ▁ is ▁ not ▁ Prime . " ) ; } }
static void sieve ( int n , bool [ ] prime ) { for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < n ; i += p ) prime [ i ] = false ; } } } static void printSophieGermanNumber ( int n ) {
bool [ ] prime = new bool [ 2 * n + 1 ] ; for ( int i = 0 ; i < prime . Length ; i ++ ) { prime [ i ] = true ; } sieve ( 2 * n + 1 , prime ) ; for ( int i = 2 ; i < n ; ++ i ) {
if ( prime [ i ] && prime [ 2 * i + 1 ] ) Console . Write ( i + " ▁ " ) ; } }
static void Main ( ) { int n = 25 ; printSophieGermanNumber ( n ) ; } }
static double ucal ( double u , int n ) { if ( n == 0 ) return 1 ; double temp = u ; for ( int i = 1 ; i <= n / 2 ; i ++ ) temp = temp * ( u - i ) ; for ( int i = 1 ; i < n / 2 ; i ++ ) temp = temp * ( u + i ) ; return temp ; }
static int fact ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
int n = 6 ; double [ ] x = { 25 , 26 , 27 , 28 , 29 , 30 } ;
double [ , ] y = new double [ n , n ] ; y [ 0 , 0 ] = 4.000 ; y [ 1 , 0 ] = 3.846 ; y [ 2 , 0 ] = 3.704 ; y [ 3 , 0 ] = 3.571 ; y [ 4 , 0 ] = 3.448 ; y [ 5 , 0 ] = 3.333 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < n - i ; j ++ ) y [ j , i ] = y [ j + 1 , i - 1 ] - y [ j , i - 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n - i ; j ++ ) System . Console . Write ( y [ i , j ] + " TABSYMBOL " ) ; System . Console . WriteLine ( " " ) ; }
double value = 27.4 ;
double sum = ( y [ 2 , 0 ] + y [ 3 , 0 ] ) / 2 ;
int k ;
k = n / 2 ; else
double u = ( value - x [ k ] ) / ( x [ 1 ] - x [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) { if ( ( i % 2 ) > 0 ) sum = sum + ( ( u - 0.5 ) * ucal ( u , i - 1 ) * y [ k , i ] ) / fact ( i ) ; else sum = sum + ( ucal ( u , i ) * ( y [ k , i ] + y [ -- k , i ] ) / ( fact ( i ) * 2 ) ) ; } System . Console . WriteLine ( " Value ▁ at ▁ " + value + " ▁ is ▁ " + System . Math . Round ( sum , 5 ) ) ; } }
using System ; class GFG { static int fibonacci ( int n ) { int a = 0 ; int b = 1 ; int c = 0 ; if ( n <= 1 ) return n ; for ( int i = 2 ; i <= n ; i ++ ) { c = a + b ; a = b ; b = c ; } return c ; }
static bool isMultipleOf10 ( int n ) { int f = fibonacci ( 30 ) ; return ( f % 10 == 0 ) ; }
public static void Main ( ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
static bool isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void Main ( ) { Console . WriteLine ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; Console . WriteLine ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
static int nextPowerOf2 ( int n ) {
int p = 1 ;
if ( n != 0 && ( ( n & ( n - 1 ) ) == 0 ) ) return n ;
while ( p < n ) p <<= 1 ; return p ; }
static int memoryUsed ( int [ ] arr , int n ) {
int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
int nearest = nextPowerOf2 ( sum ) ; return nearest ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 2 } ; int n = arr . Length ; Console . WriteLine ( memoryUsed ( arr , n ) ) ; } }
using System ; class GFG { static int toggleKthBit ( int n , int k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
public static void Main ( ) { int n = 5 , k = 1 ; Console . WriteLine ( toggleKthBit ( n , k ) ) ; } }
using System ; class GFG { static int nextPowerOf2 ( int n ) { int count = 0 ;
if ( n > 0 && ( n & ( n - 1 ) ) == 0 ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
public static void Main ( ) { int n = 0 ; Console . WriteLine ( nextPowerOf2 ( n ) ) ; } }
static void printTetra ( int n ) { if ( n < 0 ) return ;
int first = 0 , second = 1 ; int third = 1 , fourth = 2 ;
int curr = 0 ; if ( n == 0 ) Console . Write ( first ) ; else if ( n == 1 n == 2 ) Console . Write ( second ) ; else if ( n == 3 ) Console . Write ( fourth ) ; else {
for ( int i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } Console . Write ( curr ) ; } }
static public void Main ( ) { int n = 10 ; printTetra ( n ) ; } }
public static int countWays ( int n ) { int [ ] res = new int [ n + 2 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
public static void Main ( ) { int n = 4 ; Console . WriteLine ( countWays ( n ) ) ; } }
static int maxTasks ( int [ ] high , int [ ] low , int n ) {
if ( n <= 0 ) return 0 ;
return Math . Max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
public static void Main ( ) { int n = 5 ; int [ ] high = { 3 , 6 , 8 , 7 , 6 } ; int [ ] low = { 1 , 5 , 4 , 5 , 3 } ; Console . Write ( maxTasks ( high , low , n ) ) ; } }
static int countSubstr ( string str , int n , char x , char y ) {
int tot_count = 0 ;
int count_x = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str [ i ] == x ) count_x ++ ;
if ( str [ i ] == y ) tot_count += count_x ; }
return tot_count ; }
public static void Main ( ) { string str = " abbcaceghcak " ; int n = str . Length ; char x = ' a ' , y = ' c ' ; Console . Write ( " Count ▁ = ▁ " + countSubstr ( str , n , x , y ) ) ; } }
using System ; class GFG { static int OUT = 0 ; static int IN = 1 ;
static int countWords ( String str ) { int state = OUT ;
int wc = 0 ; int i = 0 ;
while ( i < str . Length ) {
if ( str [ i ] == ' ▁ ' str [ i ] == ' STRNEWLINE ' str [ i ] == ' TABSYMBOL ' ) state = OUT ;
else if ( state = = OUT ) { state = IN ; ++ wc ; }
++ i ; } return wc ; }
public static void Main ( ) { String str = " One ▁ twothree STRNEWLINE ▁ four TABSYMBOL five ▁ " ; Console . WriteLine ( " No ▁ of ▁ words ▁ : ▁ " + countWords ( str ) ) ; } }
static int nthEnneadecagonal ( int n ) {
return ( 17 * n * n - 15 * n ) / 2 ; }
static public void Main ( ) { int n = 6 ; Console . Write ( n + " th ▁ Enneadecagonal ▁ number ▁ : " ) ; Console . WriteLine ( nthEnneadecagonal ( n ) ) ; } }
using System ; class GFG { public static double PI = 3.14159265 ;
static float areacircumscribed ( float a ) { return ( a * a * ( float ) ( PI / 2 ) ) ; }
public static void Main ( ) { float a = 6 ; Console . Write ( " ▁ Area ▁ of ▁ an ▁ circumscribed " + " ▁ circle ▁ is ▁ : ▁ { 0 } " , Math . Round ( areacircumscribed ( a ) , 2 ) ) ; } }
static int printTetraRec ( int n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
static void printTetra ( int n ) { System . Console . WriteLine ( printTetraRec ( n ) + " ▁ " ) ; }
static void Main ( ) { int n = 10 ; printTetra ( n ) ; } }
using System ; class GFG {
static int max ( int x , int y ) { return ( x > y ? x : y ) ; }
static int maxTasks ( int [ ] high , int [ ] low , int n ) {
int [ ] task_dp = new int [ n + 1 ] ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( int i = 2 ; i <= n ; i ++ ) task_dp [ i ] = max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
static void Main ( ) { int n = 5 ; int [ ] high = { 3 , 6 , 8 , 7 , 6 } ; int [ ] low = { 1 , 5 , 4 , 5 , 3 } ; Console . WriteLine ( maxTasks ( high , low , n ) ) ; } }
using System ; class GFG { static int PermutationCoeff ( int n , int k ) { int Fn = 1 , Fk = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { Fn *= i ; if ( i == n - k ) Fk = Fn ; } int coeff = Fn / Fk ; return coeff ; }
public static void Main ( ) { int n = 10 , k = 2 ; Console . WriteLine ( " Value ▁ of ▁ P ( ▁ " + n + " , " + k + " ) ▁ is ▁ " + PermutationCoeff ( n , k ) ) ; } }
static int dfa = 0 ;
static void start ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ; }
static void state1 ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ;
else if ( c = = ' h ' c == ' H ' ) dfa = 2 ;
else dfa = 0 ; }
static void state2 ( char c ) {
if ( c == ' e ' c == ' E ' ) dfa = 3 ; else dfa = 0 ; }
static void state3 ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ; else dfa = 0 ; } static bool isAccepted ( char [ ] str ) {
int len = str . Length ; for ( int i = 0 ; i < len ; i ++ ) { if ( dfa == 0 ) start ( str [ i ] ) ; else if ( dfa == 1 ) state1 ( str [ i ] ) ; else if ( dfa == 2 ) state2 ( str [ i ] ) ; else state3 ( str [ i ] ) ; } return ( dfa != 3 ) ; }
static public void Main ( ) { char [ ] str = " forTHEgeeks " . ToCharArray ( ) ; if ( isAccepted ( str ) == true ) Console . WriteLine ( " ACCEPTED STRNEWLINE " ) ; else Console . WriteLine ( " NOT ▁ ACCEPTED STRNEWLINE " ) ; } }
static Boolean startsWith ( String str , String pre ) { int strLen = str . Length ; int preLen = pre . Length ; int i = 0 , j = 0 ;
while ( i < strLen && j < preLen ) {
if ( str [ i ] != pre [ j ] ) return false ; i ++ ; j ++ ; }
return true ; }
static Boolean endsWith ( String str , String suff ) { int i = str . Length - 1 ; int j = suff . Length - 1 ;
while ( i >= 0 && j >= 0 ) {
if ( str [ i ] != suff [ j ] ) return false ; i -- ; j -- ; }
return true ; }
static Boolean checkString ( String str , String a , String b ) {
if ( str . Length != a . Length + b . Length ) return false ;
if ( startsWith ( str , a ) ) {
if ( endsWith ( str , b ) ) return true ; }
if ( startsWith ( str , b ) ) {
if ( endsWith ( str , a ) ) return true ; } return false ; }
public static void Main ( String [ ] args ) { String str = " GeeksforGeeks " ; String a = " Geeksfo " ; String b = " rGeeks " ; if ( checkString ( str , a , b ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
static int minOperations ( string str , int n ) {
int i , lastUpper = - 1 , firstLower = - 1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( Char . IsUpper ( str [ i ] ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( Char . IsLower ( str [ i ] ) ) { firstLower = i ; break ; } }
if ( lastUpper == - 1 firstLower == - 1 ) return 0 ;
int countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( Char . IsUpper ( str [ i ] ) ) { countUpper ++ ; } }
int countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( Char . IsLower ( str [ i ] ) ) { countLower ++ ; } }
return Math . Min ( countLower , countUpper ) ; }
public static void Main ( ) { string str = " geEksFOrGEekS " ; int n = str . Length ; Console . WriteLine ( minOperations ( str , n ) ) ; } }
static float rainDayProbability ( int [ ] a , int n ) { float count = 0 , m ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
public static void Main ( ) { int [ ] a = { 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 } ; int n = a . Length ; Console . WriteLine ( rainDayProbability ( a , n ) ) ; } }
static double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / Math . Pow ( i , i ) ; sums += ser ; } return sums ; }
public static void Main ( ) { int n = 3 ; double res = Series ( n ) ; res = Math . Round ( res * 100000.0 ) / 100000.0 ; Console . Write ( res ) ; } }
static int ternarySearch ( int l , int r , int key , int [ ] ar ) { if ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
return ternarySearch ( l , mid1 - 1 , key , ar ) ; } else if ( key > ar [ mid2 ] ) {
return ternarySearch ( mid2 + 1 , r , key , ar ) ; } else {
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) ; } }
return - 1 ; }
int [ ] ar = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
Console . WriteLine ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
Console . WriteLine ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ; } }
using System ; class GFG { static int SIZE = 26 ;
static void printCharWithFreq ( String str ) {
int n = str . Length ;
int [ ] freq = new int [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] != 0 ) {
Console . Write ( str [ i ] ) ; Console . Write ( freq [ str [ i ] - ' a ' ] + " ▁ " ) ;
freq [ str [ i ] - ' a ' ] = 0 ; } } }
public static void Main ( ) { String str = " geeksforgeeks " ; printCharWithFreq ( str ) ; } }
using System ; public class GFG { static void checkHV ( int [ , ] arr , int N , int M ) {
bool horizontal = true ; bool vertical = true ;
for ( int i = 0 , k = N - 1 ; i < N / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < M ; j ++ ) {
if ( arr [ i , j ] != arr [ k , j ] ) { horizontal = false ; break ; } } }
for ( int i = 0 , k = M - 1 ; i < M / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i , j ] != arr [ k , j ] ) { horizontal = false ; break ; } } } if ( ! horizontal && ! vertical ) Console . WriteLine ( " NO " ) ; else if ( horizontal && ! vertical ) Console . WriteLine ( " HORIZONTAL " ) ; else if ( vertical && ! horizontal ) Console . WriteLine ( " VERTICAL " ) ; else Console . WriteLine ( " BOTH " ) ; }
static public void Main ( ) { int [ , ] mat = { { 1 , 0 , 1 } , { 0 , 0 , 0 } , { 1 , 0 , 1 } } ; checkHV ( mat , 3 , 3 ) ; } }
using System ; class GFG { static int N = 4 ;
static void add ( int [ , ] A , int [ , ] B , int [ , ] C ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i , j ] = A [ i , j ] + B [ i , j ] ; }
public static void Main ( ) { int [ , ] A = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int [ , ] B = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int [ , ] C = new int [ N , N ] ; int i , j ; add ( A , B , C ) ; Console . WriteLine ( " Result ▁ matrix ▁ is ▁ " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) Console . Write ( C [ i , j ] + " ▁ " ) ; Console . WriteLine ( ) ; } } }
public static void subtract ( int [ ] [ ] A , int [ ] [ ] B , int [ , ] C ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) { C [ i , j ] = A [ i ] [ j ] - B [ i ] [ j ] ; } } }
public static void Main ( string [ ] args ) { int [ ] [ ] A = new int [ ] [ ] { new int [ ] { 1 , 1 , 1 , 1 } , new int [ ] { 2 , 2 , 2 , 2 } , new int [ ] { 3 , 3 , 3 , 3 } , new int [ ] { 4 , 4 , 4 , 4 } } ; int [ ] [ ] B = new int [ ] [ ] { new int [ ] { 1 , 1 , 1 , 1 } , new int [ ] { 2 , 2 , 2 , 2 } , new int [ ] { 3 , 3 , 3 , 3 } , new int [ ] { 4 , 4 , 4 , 4 } } ; int [ , ] C = new int [ N , N ] ; int i , j ; subtract ( A , B , C ) ; Console . Write ( " Result ▁ matrix ▁ is ▁ STRNEWLINE " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) { Console . Write ( C [ i , j ] + " ▁ " ) ; } Console . Write ( " STRNEWLINE " ) ; } } }
using System ; class GFG { static int linearSearch ( int [ ] arr , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == i ) return i ; }
return - 1 ; }
public static void Main ( ) { int [ ] arr = { - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = arr . Length ; Console . Write ( " Fixed ▁ Point ▁ is ▁ " + linearSearch ( arr , n ) ) ; } }
using System ; class GFG { static int binarySearch ( int [ ] arr , int low , int high ) { if ( high >= low ) {
int mid = ( low + high ) / 2 ; if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return - 1 ; }
public static void Main ( ) { int [ ] arr = { - 10 , - 1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = arr . Length ; Console . Write ( " Fixed ▁ Point ▁ is ▁ " + binarySearch ( arr , 0 , n - 1 ) ) ; } }
using System ; class GFG { static int maxTripletSum ( int [ ] arr , int n ) {
int sum = - 1000000 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) for ( int k = j + 1 ; k < n ; k ++ ) if ( sum < arr [ i ] + arr [ j ] + arr [ k ] ) sum = arr [ i ] + arr [ j ] + arr [ k ] ; return sum ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( maxTripletSum ( arr , n ) ) ; } }
static int maxTripletSum ( int [ ] arr , int n ) {
Array . Sort ( arr ) ;
return arr [ n - 1 ] + arr [ n - 2 ] + arr [ n - 3 ] ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( maxTripletSum ( arr , n ) ) ; } }
static int maxTripletSum ( int [ ] arr , int n ) {
int maxA = - 100000000 , maxB = - 100000000 ; int maxC = - 100000000 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > maxA ) { maxC = maxB ; maxB = maxA ; maxA = arr [ i ] ; }
else if ( arr [ i ] > maxB ) { maxC = maxB ; maxB = arr [ i ] ; }
else if ( arr [ i ] > maxC ) maxC = arr [ i ] ; } return ( maxA + maxB + maxC ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( maxTripletSum ( arr , n ) ) ; } }
using System ; class GFG { public static int search ( int [ ] arr , int x ) { int n = arr . Length ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == x ) return i ; } return - 1 ; }
public static void Main ( ) { int [ ] arr = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ;
int result = search ( arr , x ) ; if ( result == - 1 ) Console . WriteLine ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) ; else Console . WriteLine ( " Element ▁ is ▁ present ▁ at ▁ index ▁ " + result ) ; } }
using System ; class GFG {
static void countsort ( char [ ] arr ) { int n = arr . Length ;
char [ ] output = new char [ n ] ;
int [ ] count = new int [ 256 ] ; for ( int i = 0 ; i < 256 ; ++ i ) count [ i ] = 0 ;
for ( int i = 0 ; i < n ; ++ i ) ++ count [ arr [ i ] ] ;
for ( int i = 1 ; i <= 255 ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( int i = 0 ; i < n ; ++ i ) arr [ i ] = output [ i ] ; }
public static void Main ( ) { char [ ] arr = { ' g ' , ' e ' , ' e ' , ' k ' , ' s ' , ' f ' , ' o ' , ' r ' , ' g ' , ' e ' , ' e ' , ' k ' , ' s ' } ; countsort ( arr ) ; Console . Write ( " Sorted ▁ character ▁ array ▁ is ▁ " ) ; for ( int i = 0 ; i < arr . Length ; ++ i ) Console . Write ( arr [ i ] ) ; } }
static int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
public static void Main ( ) { int n = 5 , k = 2 ; Console . Write ( " Value ▁ of ▁ C ( " + n + " , " + k + " ) ▁ is ▁ " + binomialCoeff ( n , k ) ) ; } }
using System ; class GFG { static int binomialCoeff ( int n , int k ) { int [ ] C = new int [ k + 1 ] ;
C [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = Math . Min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
public static void Main ( ) { int n = 5 , k = 2 ; Console . WriteLine ( " Value ▁ of ▁ C ( " + n + " ▁ " + k + " ) ▁ is ▁ " + binomialCoeff ( n , k ) ) ; } }
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
bool [ , ] subset = new bool [ sum + 1 , n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 , i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i , 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i , j ] = subset [ i , j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i , j ] = subset [ i , j ] || subset [ i - set [ j - 1 ] , j - 1 ] ; } } return subset [ sum , n ] ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
static void Main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) Console . WriteLine ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " + N + " ▁ keystrokes ▁ is ▁ " + findoptimal ( N ) ) ; } }
static int power ( int x , int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; }
public static void Main ( ) { int x = 2 ; int y = 3 ; Console . Write ( power ( x , y ) ) ; } }
using System ; public class GFG { static float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
public static void Main ( ) { float x = 2 ; int y = - 3 ; Console . Write ( power ( x , y ) ) ; } }
static float squareRoot ( float n ) {
float x = n ; float y = 1 ;
double e = 0.000001 ; while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
public static void Main ( ) { int n = 50 ; Console . Write ( " Square ▁ root ▁ of ▁ " + n + " ▁ is ▁ " + squareRoot ( n ) ) ; } }
static float getAvg ( float prev_avg , float x , int n ) { return ( prev_avg * n + x ) / ( n + 1 ) ; }
static void streamAvg ( float [ ] arr , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( avg , arr [ i ] , i ) ; Console . WriteLine ( " Average ▁ of ▁ { 0 } ▁ " + " numbers ▁ is ▁ { 1 } " , i + 1 , avg ) ; } return ; }
public static void Main ( String [ ] args ) { float [ ] arr = { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . Length ; streamAvg ( arr , n ) ; } }
static float getAvg ( int x ) { sum += x ; return ( ( ( float ) sum ) / ++ n ) ; }
static void streamAvg ( float [ ] arr , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( ( int ) arr [ i ] ) ; Console . WriteLine ( " Average ▁ of ▁ { 0 } ▁ numbers ▁ " + " is ▁ { 1 } " , ( i + 1 ) , avg ) ; } return ; }
static int Main ( ) { float [ ] arr = new float [ ] { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . Length ; streamAvg ( arr , n ) ; return 0 ; } }
static int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
public static void Main ( ) { int n = 8 ; int k = 2 ; Console . Write ( " Value ▁ of ▁ C ( " + n + " , ▁ " + k + " ) ▁ " + " is " + " ▁ " + binomialCoeff ( n , k ) ) ; } }
public static void primeFactors ( int n ) {
while ( n % 2 == 0 ) { Console . Write ( 2 + " ▁ " ) ; n /= 2 ; }
for ( int i = 3 ; i <= Math . Sqrt ( n ) ; i += 2 ) {
while ( n % i == 0 ) { Console . Write ( i + " ▁ " ) ; n /= i ; } }
if ( n > 2 ) Console . Write ( n ) ; }
public static void Main ( ) { int n = 315 ; primeFactors ( n ) ; } } }
static void printCombination ( int [ ] arr , int n , int r ) {
int [ ] data = new int [ r ] ;
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; }
static void combinationUtil ( int [ ] arr , int [ ] data , int start , int end , int index , int r ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) Console . Write ( data [ j ] + " ▁ " ) ; Console . WriteLine ( " " ) ; return ; }
for ( int i = start ; i <= end && end - i + 1 >= r - index ; i ++ ) { data [ index ] = arr [ i ] ; combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; } }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . Length ; printCombination ( arr , n , r ) ; } }
static void printCombination ( int [ ] arr , int n , int r ) {
int [ ] data = new int [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
static void combinationUtil ( int [ ] arr , int n , int r , int index , int [ ] data , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) Console . Write ( data [ j ] + " ▁ " ) ; Console . WriteLine ( " " ) ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . Length ; printCombination ( arr , n , r ) ; } }
int findgroups ( int [ ] arr , int n ) {
int [ ] c = new int [ ] { 0 , 0 , 0 } ; int i ;
int res = 0 ;
for ( i = 0 ; i < n ; i ++ ) c [ arr [ i ] % 3 ] ++ ;
res += ( ( c [ 0 ] * ( c [ 0 ] - 1 ) ) >> 1 ) ;
res += c [ 1 ] * c [ 2 ] ;
res += ( c [ 0 ] * ( c [ 0 ] - 1 ) * ( c [ 0 ] - 2 ) ) / 6 ;
res += ( c [ 1 ] * ( c [ 1 ] - 1 ) * ( c [ 1 ] - 2 ) ) / 6 ;
res += ( ( c [ 2 ] * ( c [ 2 ] - 1 ) * ( c [ 2 ] - 2 ) ) / 6 ) ;
res += c [ 0 ] * c [ 1 ] * c [ 2 ] ;
return res ; }
public static void Main ( ) { FindGroups groups = new FindGroups ( ) ; int [ ] arr = { 3 , 6 , 7 , 2 , 9 } ; int n = arr . Length ; Console . Write ( " Required ▁ number ▁ of ▁ groups ▁ are ▁ " + groups . findgroups ( arr , n ) ) ; } }
using System ; class GFG { static int nextPowerOf2 ( int n ) { int count = 0 ;
if ( n > 0 && ( n & ( n - 1 ) ) == 0 ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
public static void Main ( ) { int n = 0 ; Console . WriteLine ( nextPowerOf2 ( n ) ) ; } }
using System ; class GFG { static int nextPowerOf2 ( int n ) { int p = 1 ; if ( n > 0 && ( n & ( n - 1 ) ) == 0 ) return n ; while ( p < n ) p <<= 1 ; return p ; }
public static void Main ( ) { int n = 5 ; Console . Write ( nextPowerOf2 ( n ) ) ; } }
static int nextPowerOf2 ( int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
public static void Main ( ) { int n = 5 ; Console . WriteLine ( nextPowerOf2 ( n ) ) ; } }
static void segregate0and1 ( int [ ] arr , int n ) {
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 0 ) count ++ ; }
for ( int i = 0 ; i < count ; i ++ ) arr [ i ] = 0 ;
for ( int i = count ; i < n ; i ++ ) arr [ i ] = 1 ; }
static void print ( int [ ] arr , int n ) { Console . WriteLine ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int n = arr . Length ; segregate0and1 ( arr , n ) ; print ( arr , n ) ; } }
void segregate0and1 ( int [ ] arr , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
public static void Main ( ) { Segregate seg = new Segregate ( ) ; int [ ] arr = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = arr . Length ; seg . segregate0and1 ( arr , arr_size ) ; Console . WriteLine ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
static void segregate0and1 ( int [ ] arr ) { int type0 = 0 ; int type1 = arr . Length - 1 ; while ( type0 < type1 ) { if ( arr [ type0 ] == 1 ) { arr [ type1 ] = arr [ type1 ] + arr [ type0 ] ; arr [ type0 ] = arr [ type1 ] - arr [ type0 ] ; arr [ type1 ] = arr [ type1 ] - arr [ type0 ] ; type1 -- ; } else { type0 ++ ; } } }
public static void Main ( string [ ] args ) { int [ ] array = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; segregate0and1 ( array ) ; Console . Write ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; foreach ( int a in array ) { Console . Write ( a + " ▁ " ) ; } } }
static int maxIndexDiff ( int [ ] arr , int n ) { int maxDiff = - 1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
public static void Main ( ) { int [ ] arr = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = arr . Length ; int maxDiff = maxIndexDiff ( arr , n ) ; Console . Write ( maxDiff ) ; } }
static int missingK ( int [ ] a , int k , int n ) { int difference = 0 , ans = 0 , count = k ; bool flag = false ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = true ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return - 1 ; }
int [ ] a = { 1 , 5 , 11 , 19 } ;
int k = 11 ; int n = a . Length ;
int missing = missingK ( a , k , n ) ; Console . Write ( missing ) ; } }
static int findRotations ( String str ) {
String tmp = str + str ; int n = str . Length ; for ( int i = 1 ; i <= n ; i ++ ) {
String substring = tmp . Substring ( i , str . Length ) ;
if ( str == substring ) return i ; } return n ; }
public static void Main ( ) { String str = " abc " ; Console . Write ( findRotations ( str ) ) ; } }
static int findKth ( int [ ] arr , int n , int k ) { HashSet < int > missing = new HashSet < int > ( ) ; int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { missing . Add ( arr [ i ] ) ; }
int maxm = arr . Max ( ) ; int minm = arr . Min ( ) ;
for ( int i = minm + 1 ; i < maxm ; i ++ ) {
if ( ! missing . Contains ( i ) ) { count ++ ; }
if ( count == k ) { return i ; } }
return - 1 ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 10 , 9 , 4 } ; int n = arr . Length ; int k = 5 ; Console . WriteLine ( findKth ( arr , n , k ) ) ; } }
static int waysToKAdjacentSetBits ( int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( != 1 ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
public static void Main ( ) { int n = 5 , k = 2 ;
int totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; Console . WriteLine ( " Number ▁ of ▁ ways ▁ = ▁ " + totalWays ) ; } }
public static int findStep ( int n ) { if ( n == 1 n == 0 ) return 1 ; else if ( n == 2 ) return 2 ; else return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) ; }
public static void Main ( ) { int n = 4 ; Console . WriteLine ( findStep ( n ) ) ; } }
static bool isSubsetSum ( int [ ] arr , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 && sum != 0 ) return false ;
if ( arr [ n - 1 ] > sum ) return isSubsetSum ( arr , n - 1 , sum ) ;
return isSubsetSum ( arr , n - 1 , sum ) || isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) ; }
static bool findPartition ( int [ ] arr , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( sum % 2 != 0 ) return false ;
return isSubsetSum ( arr , n , sum / 2 ) ; }
public static void Main ( ) { int [ ] arr = { 3 , 1 , 5 , 9 , 12 } ; int n = arr . Length ;
if ( findPartition ( arr , n ) == true ) Console . Write ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " + " subsets ▁ of ▁ equal ▁ sum " ) ; else Console . Write ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " + " two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
using System ; class GFG { static int findRepeatFirstN2 ( string s ) {
int p = - 1 , i , j ; for ( i = 0 ; i < s . Length ; i ++ ) { for ( j = i + 1 ; j < s . Length ; j ++ ) { if ( s [ i ] == s [ j ] ) { p = i ; break ; } } if ( p != - 1 ) break ; } return p ; }
static public void Main ( ) { string str = " geeksforgeeks " ; int pos = findRepeatFirstN2 ( str ) ; if ( pos == - 1 ) Console . WriteLine ( " Not ▁ found " ) ; else Console . WriteLine ( str [ pos ] ) ; } }
static int possibleStrings ( int n , int r , int b , int g ) {
int [ ] fact = new int [ n + 1 ] ; fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
int left = n - ( r + g + b ) ; int sum = 0 ;
for ( int i = 0 ; i <= left ; i ++ ) { for ( int j = 0 ; j <= left - i ; j ++ ) { int k = left - ( i + j ) ;
sum = sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ; } }
return sum ; }
public static void Main ( ) { int n = 4 , r = 2 ; int b = 0 , g = 1 ; Console . WriteLine ( possibleStrings ( n , r , b , g ) ) ; } }
static int remAnagram ( string str1 , string str2 ) {
int [ ] count1 = new int [ 26 ] ; int [ ] count2 = new int [ 26 ] ;
for ( int i = 0 ; i < str1 . Length ; i ++ ) count1 [ str1 [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < str2 . Length ; i ++ ) count2 [ str2 [ i ] - ' a ' ] ++ ;
int result = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) result += Math . Abs ( count1 [ i ] - count2 [ i ] ) ; return result ; }
public static void Main ( ) { string str1 = " bcadeh " , str2 = " hea " ; Console . Write ( remAnagram ( str1 , str2 ) ) ; } }
static void printPath ( List < int > res , int nThNode , int kThNode ) {
if ( kThNode > nThNode ) return ;
res . Add ( kThNode ) ;
for ( int i = 0 ; i < res . Count ; i ++ ) Console . Write ( res [ i ] + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ;
printPath ( res , nThNode , kThNode * 2 ) ;
printPath ( res , nThNode , kThNode * 2 + 1 ) ; res . RemoveAt ( res . Count - 1 ) ; }
static void printPathToCoverAllNodeUtil ( int nThNode ) {
List < int > res = new List < int > ( ) ;
printPath ( res , nThNode , 1 ) ; }
int nThNode = 7 ;
printPathToCoverAllNodeUtil ( nThNode ) ; } }
static void shortestLength ( int n , int [ ] x , int [ ] y ) { int answer = 0 ;
int i = 0 ; while ( n != 0 && i < x . Length ) {
if ( x [ i ] + y [ i ] > answer ) answer = x [ i ] + y [ i ] ; i ++ ; }
Console . WriteLine ( " Length ▁ - > ▁ " + answer ) ; Console . WriteLine ( " Path ▁ - > ▁ " + " ( ▁ 1 , ▁ " + answer + " ▁ ) " + " and ▁ ( ▁ " + answer + " , ▁ 1 ▁ ) " ) ; }
int n = 4 ;
int [ ] x = new int [ ] { 1 , 4 , 2 , 1 } ; int [ ] y = new int [ ] { 4 , 1 , 1 , 2 } ; shortestLength ( n , x , y ) ; } }
static void FindPoints ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 , int x4 , int y4 ) {
int x5 = Math . Max ( x1 , x3 ) ; int y5 = Math . Max ( y1 , y3 ) ;
int x6 = Math . Min ( x2 , x4 ) ; int y6 = Math . Min ( y2 , y4 ) ;
if ( x5 > x6 y5 > y6 ) { Console . WriteLine ( " No ▁ intersection " ) ; return ; } Console . Write ( " ( " + x5 + " , ▁ " + y5 + " ) ▁ " ) ; Console . Write ( " ( " + x6 + " , ▁ " + y6 + " ) ▁ " ) ;
int x7 = x5 ; int y7 = y6 ; Console . Write ( " ( " + x7 + " , ▁ " + y7 + " ) ▁ " ) ;
int x8 = x6 ; int y8 = y5 ; Console . Write ( " ( " + x8 + " , ▁ " + y8 + " ) ▁ " ) ; }
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x3 = 2 , y3 = 3 , x4 = 7 , y4 = 9 ;
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) ; } }
static double area ( double a , double b , double c ) { double d = Math . Abs ( ( c * c ) / ( 2 * a * b ) ) ; return d ; }
static public void Main ( ) { double a = - 2 , b = 4 , c = 3 ; Console . WriteLine ( area ( a , b , c ) ) ; } }
static ArrayList addToArrayForm ( ArrayList A , int K ) {
ArrayList v = new ArrayList ( ) ; ArrayList ans = new ArrayList ( ) ;
int rem = 0 ; int i = 0 ;
for ( i = A . Count - 1 ; i >= 0 ; i -- ) {
int my = ( int ) A [ i ] + K % 10 + rem ; if ( my > 9 ) {
rem = 1 ;
v . Add ( my % 10 ) ; } else { v . Add ( my ) ; rem = 0 ; } K = K / 10 ; }
while ( K > 0 ) {
int my = K % 10 + rem ; v . Add ( my % 10 ) ;
if ( my / 10 > 0 ) rem = 1 ; else rem = 0 ; K = K / 10 ; } if ( rem > 0 ) v . Add ( rem ) ;
for ( int j = v . Count - 1 ; j >= 0 ; j -- ) ans . Add ( ( int ) v [ j ] ) ; return ans ; }
static void Main ( ) { ArrayList A = new ArrayList ( ) ; A . Add ( 2 ) ; A . Add ( 7 ) ; A . Add ( 4 ) ; int K = 181 ; ArrayList ans = addToArrayForm ( A , K ) ;
for ( int i = 0 ; i < ans . Count ; i ++ ) Console . Write ( ( int ) ans [ i ] ) ; } }
static int findThirdDigit ( int n ) {
if ( n < 3 ) return 0 ;
return ( n & 1 ) > 0 ? 1 : 6 ; }
static void Main ( ) { int n = 7 ; Console . WriteLine ( findThirdDigit ( n ) ) ; } }
public static double getProbability ( int a , int b , int c , int d ) {
double p = ( double ) a / ( double ) b ; double q = ( double ) c / ( double ) d ;
double ans = p * ( 1 / ( 1 - ( 1 - q ) * ( 1 - p ) ) ) ; return ans ; }
public static void Main ( string [ ] args ) { int a = 1 , b = 2 , c = 10 , d = 11 ; Console . Write ( " { 0 : F5 } " , getProbability ( a , b , c , d ) ) ; } }
static bool isPalindrome ( int n ) {
int divisor = 1 ; while ( n / divisor >= 10 ) divisor *= 10 ; while ( n != 0 ) { int leading = n / divisor ; int trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = ( n % divisor ) / 10 ;
divisor = divisor / 100 ; } return true ; }
static int largestPalindrome ( int [ ] A , int n ) { int currentMax = - 1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( A [ i ] > currentMax && isPalindrome ( A [ i ] ) ) currentMax = A [ i ] ; }
return currentMax ; }
public static void Main ( ) { int [ ] A = { 1 , 232 , 54545 , 999991 } ; int n = A . Length ;
Console . WriteLine ( largestPalindrome ( A , n ) ) ; } }
public static long getFinalElement ( long n ) { long finalNum ; for ( finalNum = 2 ; finalNum * 2 <= n ; finalNum *= 2 ) ; return finalNum ; }
static public void Main ( ) { int N = 12 ; Console . WriteLine ( getFinalElement ( N ) ) ; } }
static bool isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
static bool isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
static long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
public static void Main ( String [ ] args ) { int L = 110 , R = 1130 ; Console . WriteLine ( sumOfAllPalindrome ( L , R ) ) ; } }
static double calculateAlternateSum ( int n ) { if ( n <= 0 ) return 0 ; int [ ] fibo = new int [ n + 1 ] ; fibo [ 0 ] = 0 ; fibo [ 1 ] = 1 ;
double sum = Math . Pow ( fibo [ 0 ] , 2 ) + Math . Pow ( fibo [ 1 ] , 2 ) ;
for ( int i = 2 ; i <= n ; i ++ ) { fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] ;
if ( i % 2 == 0 ) sum -= fibo [ i ] ;
else sum += [ i ] ; }
return sum ; }
int n = 8 ;
Console . WriteLine ( " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " + n + " ▁ terms : ▁ " + calculateAlternateSum ( n ) ) ; } }
static int getValue ( int n ) { int i = 0 , k = 1 ; while ( i < n ) { i = i + k ; k = k * 2 ; } return k / 2 ; }
int n = 9 ;
Console . WriteLine ( getValue ( n ) ) ;
n = 1025 ;
Console . WriteLine ( getValue ( n ) ) ; } }
static void countDigits ( double val , long [ ] arr ) { while ( ( long ) val > 0 ) { long digit = ( long ) val % 10 ; arr [ ( int ) digit ] ++ ; val = ( long ) val / 10 ; } return ; } static void countFrequency ( int x , int n ) {
long [ ] freq_count = new long [ 10 ] ;
for ( int i = 1 ; i <= n ; i ++ ) {
double val = Math . Pow ( ( double ) x , ( double ) i ) ;
countDigits ( val , freq_count ) ; }
for ( int i = 0 ; i <= 9 ; i ++ ) { Console . Write ( freq_count [ i ] + " ▁ " ) ; } }
public static void Main ( ) { int x = 15 , n = 3 ; countFrequency ( x , n ) ; } }
static int countSolutions ( int a ) { int count = 0 ;
for ( int i = 0 ; i <= a ; i ++ ) { if ( a == ( i + ( a ^ i ) ) ) count ++ ; } return count ; }
public static void Main ( ) { int a = 3 ; Console . WriteLine ( countSolutions ( a ) ) ; } }
static int countSolutions ( int a ) { int count = bitCount ( a ) ; count = ( int ) System . Math . Pow ( 2 , count ) ; return count ; } static int bitCount ( int n ) { int count = 0 ; while ( n != 0 ) { count ++ ; n &= ( n - 1 ) ; } return count ; }
public static void Main ( ) { int a = 3 ; System . Console . WriteLine ( countSolutions ( a ) ) ; } }
static int calculateAreaSum ( int l , int b ) { int size = 1 ;
int maxSize = Math . Min ( l , b ) ; int totalArea = 0 ; for ( int i = 1 ; i <= maxSize ; i ++ ) {
int totalSquares = ( l - size + 1 ) * ( b - size + 1 ) ;
int area = totalSquares * size * size ;
totalArea += area ;
size ++ ; } return totalArea ; }
public static void Main ( ) { int l = 4 , b = 3 ; Console . Write ( calculateAreaSum ( l , b ) ) ; } }
static long boost_hyperfactorial ( long num ) {
long val = 1 ; for ( long i = 1 ; i <= num ; i ++ ) { val = val * ( long ) Math . Pow ( i , i ) ; }
return val ; }
public static void Main ( ) { int num = 5 ; Console . WriteLine ( boost_hyperfactorial ( num ) ) ; } }
static int boost_hyperfactorial ( int num ) {
int val = 1 ; for ( int i = 1 ; i <= num ; i ++ ) { for ( int j = 1 ; j <= i ; j ++ ) {
val *= i ; } }
return val ; }
public static void Main ( ) { int num = 5 ; Console . WriteLine ( boost_hyperfactorial ( num ) ) ; } }
using System ; class GFG { static int subtractOne ( int x ) { int m = 1 ;
while ( ! ( ( x & m ) > 0 ) ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
public static void Main ( ) { Console . WriteLine ( subtractOne ( 13 ) ) ; } }
using System ; class GFG { static int rows = 3 ; static int cols = 3 ;
static void meanVector ( int [ , ] mat ) { Console . Write ( " [ ▁ " ) ;
for ( int i = 0 ; i < rows ; i ++ ) {
double mean = 0.00 ;
int sum = 0 ; for ( int j = 0 ; j < cols ; j ++ ) sum += mat [ j , i ] ; mean = sum / rows ; Console . Write ( ( int ) mean + " ▁ " ) ; } Console . Write ( " ] " ) ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 2 , 3 } , { 4 , 5 , 6 } , { 7 , 8 , 9 } } ; meanVector ( mat ) ; } }
static List < int > primeFactors ( int n ) { List < int > res = new List < int > ( ) ; if ( n % 2 == 0 ) { while ( n % 2 == 0 ) n = n / 2 ; res . Add ( 2 ) ; }
for ( int i = 3 ; i <= Math . Sqrt ( n ) ; i = i + 2 ) {
if ( n % i == 0 ) { while ( n % i == 0 ) n = n / i ; res . Add ( i ) ; } }
if ( n > 2 ) res . Add ( n ) ; return res ; }
static bool isHoax ( int n ) {
List < int > pf = primeFactors ( n ) ;
if ( pf [ 0 ] == n ) return false ;
int all_pf_sum = 0 ; for ( int i = 0 ; i < pf . Count ; i ++ ) {
int pf_sum ; for ( pf_sum = 0 ; pf [ i ] > 0 ; pf_sum += pf [ i ] % 10 , pf [ i ] /= 10 ) ; all_pf_sum += pf_sum ; }
int sum_n ; for ( sum_n = 0 ; n > 0 ; sum_n += n % 10 , n /= 10 ) ;
return sum_n == all_pf_sum ; }
public static void Main ( ) { int n = 84 ; if ( isHoax ( n ) ) Console . Write ( " A ▁ Hoax ▁ Number STRNEWLINE " ) ; else Console . Write ( " Not ▁ a ▁ Hoax ▁ Number STRNEWLINE " ) ; } }
static int modInverse ( int a , int prime ) { a = a % prime ; for ( int x = 1 ; x < prime ; x ++ ) if ( ( a * x ) % prime == 1 ) return x ; return - 1 ; } static void printModIverses ( int n , int prime ) { for ( int i = 1 ; i <= n ; i ++ ) Console . Write ( modInverse ( i , prime ) + " ▁ " ) ; }
public static void Main ( ) { int n = 10 , prime = 17 ; printModIverses ( n , prime ) ; } }
static int minOp ( int num ) {
int rem ; int count = 0 ;
while ( num > 0 ) { rem = num % 10 ; if ( ! ( rem == 3 rem == 8 ) ) count ++ ; num /= 10 ; } return count ; }
public static void Main ( ) { int num = 234198 ; Console . WriteLine ( " Minimum ▁ Operations ▁ = " + minOp ( num ) ) ; } }
static int sumOfDigits ( int a ) { int sum = 0 ; while ( a != 0 ) { sum += a % 10 ; a /= 10 ; } return sum ; }
static int findMax ( int x ) {
int b = 1 , ans = x ;
while ( x != 0 ) {
int cur = ( x - 1 ) * b + ( b - 1 ) ;
if ( sumOfDigits ( cur ) > sumOfDigits ( ans ) || ( sumOfDigits ( cur ) == sumOfDigits ( ans ) && cur > ans ) ) ans = cur ;
x /= 10 ; b *= 10 ; } return ans ; }
public static void Main ( ) { int n = 521 ; Console . WriteLine ( findMax ( n ) ) ; } }
static int median ( int [ ] a , int l , int r ) { int n = r - l + 1 ; n = ( n + 1 ) / 2 - 1 ; return n + l ; }
static int IQR ( int [ ] a , int n ) { Array . Sort ( a ) ;
int mid_index = median ( a , 0 , n ) ;
int Q1 = a [ median ( a , 0 , mid_index ) ] ;
int Q3 = a [ mid_index + median ( a , mid_index + 1 , n ) ] ;
return ( Q3 - Q1 ) ; }
public static void Main ( ) { int [ ] a = { 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 } ; int n = a . Length ; Console . WriteLine ( IQR ( a , n ) ) ; } }
static bool isPalindrome ( int n ) {
int divisor = 1 ; while ( n / divisor >= 10 ) divisor *= 10 ; while ( n != 0 ) { int leading = n / divisor ; int trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = ( n % divisor ) / 10 ;
divisor = divisor / 100 ; } return true ; }
static int largestPalindrome ( int [ ] A , int n ) {
Array . Sort ( A ) ; for ( int i = n - 1 ; i >= 0 ; -- i ) {
if ( isPalindrome ( A [ i ] ) ) return A [ i ] ; }
return - 1 ; }
public static void Main ( ) { int [ ] A = { 1 , 232 , 54545 , 999991 } ; int n = A . Length ;
Console . WriteLine ( largestPalindrome ( A , n ) ) ; } }
static int findSum ( int n , int a , int b ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
static void Main ( ) { int n = 10 , a = 3 , b = 5 ; Console . WriteLine ( findSum ( n , a , b ) ) ; } }
using System ; class GFG { static int subtractOne ( int x ) { return ( ( x << 1 ) + ( ~ x ) ) ; } public static void Main ( String [ ] args ) { Console . Write ( " { 0 } " , subtractOne ( 13 ) ) ; } }
public static int pell ( int n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
public static void Main ( ) { int n = 4 ; Console . Write ( pell ( n ) ) ; } }
static long LCM ( int [ ] arr , int n ) {
int max_num = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( max_num < arr [ i ] ) { max_num = arr [ i ] ; } }
long res = 1 ;
while ( x <= max_num ) {
ArrayList indexes = new ArrayList ( ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] % x == 0 ) { indexes . Add ( j ) ; } }
if ( indexes . Count >= 2 ) {
for ( int j = 0 ; j < indexes . Count ; j ++ ) { arr [ ( int ) indexes [ j ] ] = arr [ ( int ) indexes [ j ] ] / x ; } res = res * x ; } else { x ++ ; } }
for ( int i = 0 ; i < n ; i ++ ) { res = res * arr [ i ] ; } return res ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 } ; int n = arr . Length ; Console . WriteLine ( LCM ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 10000 ;
static List < int > primes = new List < int > ( ) ;
static void sieveSundaram ( ) {
Boolean [ ] marked = new Boolean [ MAX / 2 + 100 ] ;
for ( int i = 1 ; i <= ( Math . Sqrt ( MAX ) - 1 ) / 2 ; i ++ ) for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= MAX / 2 ; j = j + 2 * i + 1 ) marked [ j ] = true ;
primes . Add ( 2 ) ;
for ( int i = 1 ; i <= MAX / 2 ; i ++ ) if ( marked [ i ] == false ) primes . Add ( 2 * i + 1 ) ; }
static void findPrimes ( int n ) {
if ( n <= 2 n % 2 != 0 ) { Console . WriteLine ( " Invalid ▁ Input ▁ " ) ; return ; }
for ( int i = 0 ; primes [ i ] <= n / 2 ; i ++ ) {
int diff = n - primes [ i ] ;
if ( primes . Contains ( diff ) ) {
Console . WriteLine ( primes [ i ] + " ▁ + ▁ " + diff + " ▁ = ▁ " + n ) ; return ; } } }
sieveSundaram ( ) ;
findPrimes ( 4 ) ; findPrimes ( 38 ) ; findPrimes ( 100 ) ; } }
static int kPrimeFactor ( int n , int k ) {
while ( n % 2 == 0 ) { k -- ; n = n / 2 ; if ( k == 0 ) return 2 ; }
for ( int i = 3 ; i <= Math . Sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { if ( k == 1 ) return i ; k -- ; n = n / i ; } }
if ( n > 2 && k == 1 ) return n ; return - 1 ; }
public static void Main ( ) { int n = 12 , k = 3 ; Console . WriteLine ( kPrimeFactor ( n , k ) ) ; n = 14 ; k = 3 ; Console . WriteLine ( kPrimeFactor ( n , k ) ) ; } }
using System ; class GFG { static int MAX = 10001 ;
static void sieveOfEratosthenes ( int [ ] s ) {
bool [ ] prime = new bool [ MAX + 1 ] ;
for ( int i = 2 ; i <= MAX ; i += 2 ) s [ i ] = 2 ;
for ( int i = 3 ; i <= MAX ; i += 2 ) { if ( prime [ i ] == false ) {
s [ i ] = i ;
for ( int j = i ; j * i <= MAX ; j += 2 ) { if ( prime [ i * j ] == false ) { prime [ i * j ] = true ;
s [ i * j ] = i ; } } } } }
static int kPrimeFactor ( int n , int k , int [ ] s ) {
while ( n > 1 ) { if ( k == 1 ) return s [ n ] ;
k -- ;
n /= s [ n ] ; } return - 1 ; }
int [ ] s = new int [ MAX + 1 ] ; sieveOfEratosthenes ( s ) ; int n = 12 , k = 3 ; Console . WriteLine ( kPrimeFactor ( n , k , s ) ) ; n = 14 ; k = 3 ; Console . WriteLine ( kPrimeFactor ( n , k , s ) ) ; } }
static bool squareRootExists ( int n , int p ) { n = n % p ;
for ( int x = 2 ; x < p ; x ++ ) if ( ( x * x ) % p == n ) return true ; return false ; }
public static void Main ( ) { int p = 7 ; int n = 2 ; if ( squareRootExists ( n , p ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
static int Largestpower ( int n , int p ) {
int ans = 0 ;
while ( n > 0 ) { n /= p ; ans += n ; } return ans ; }
public static void Main ( ) { int n = 10 ; int p = 3 ; Console . Write ( " ▁ The ▁ largest ▁ power ▁ of ▁ " + p + " ▁ that ▁ divides ▁ " + n + " ! ▁ is ▁ " + Largestpower ( n , p ) ) ; } }
using System ; class Factorial { int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
public static void Main ( ) { Factorial obj = new Factorial ( ) ; int num = 5 ; Console . WriteLine ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + obj . factorial ( num ) ) ; } }
public static int reverseBits ( int n ) { int rev = 0 ;
while ( n > 0 ) {
rev <<= 1 ;
if ( ( int ) ( n & 1 ) == 1 ) rev ^= 1 ;
n >>= 1 ; }
return rev ; }
public static void Main ( ) { int n = 11 ; Console . WriteLine ( reverseBits ( n ) ) ; } }
static int countgroup ( int [ ] a , int n ) { int xs = 0 ; for ( int i = 0 ; i < n ; i ++ ) xs = xs ^ a [ i ] ;
if ( xs == 0 ) return ( 1 << ( n - 1 ) ) - 1 ; return 0 ; }
public static void Main ( ) { int [ ] a = { 1 , 2 , 3 } ; int n = a . Length ; Console . WriteLine ( countgroup ( a , n ) ) ; } }
static int bitExtracted ( int number , int k , int p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
public static void Main ( ) { int number = 171 , k = 5 , p = 2 ; Console . WriteLine ( " The ▁ extracted ▁ number ▁ is ▁ " + bitExtracted ( number , k , p ) ) ; } }
static bool isAMultipleOf4 ( int n ) {
if ( ( n & 3 ) == 0 ) return true ;
return false ; }
public static void Main ( ) { int n = 16 ; Console . WriteLine ( isAMultipleOf4 ( n ) ? " Yes " : " No " ) ; } }
using System ; class GFG { public static int square ( int n ) {
if ( n < 0 ) n = - n ;
int res = n ;
for ( int i = 1 ; i < n ; i ++ ) res += n ; return res ; }
public static void Main ( ) { for ( int n = 1 ; n <= 5 ; n ++ ) Console . WriteLine ( " n ▁ = ▁ " + n + " , ▁ n ^ 2 ▁ = ▁ " + square ( n ) ) ; } }
using System ; class GFG { static int PointInKSquares ( int n , int [ ] a , int k ) { Array . Sort ( a ) ; return a [ n - k ] ; }
public static void Main ( String [ ] args ) { int k = 2 ; int [ ] a = { 1 , 2 , 3 , 4 } ; int n = a . Length ; int x = PointInKSquares ( n , a , k ) ; Console . WriteLine ( " ( " + x + " , ▁ " + x + " ) " ) ; } }
static long answer ( int n ) {
int [ ] dp = new int [ 10 ] ;
int [ ] prev = new int [ 10 ] ;
if ( n == 1 ) return 10 ;
for ( int j = 0 ; j <= 9 ; j ++ ) dp [ j ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= 9 ; j ++ ) { prev [ j ] = dp [ j ] ; } for ( int j = 0 ; j <= 9 ; j ++ ) {
if ( j == 0 ) dp [ j ] = prev [ j + 1 ] ;
else if ( j = = 9 ) dp [ j ] = prev [ j - 1 ] ;
else dp [ j ] = prev [ j - 1 ] + prev [ j + 1 ] ; } }
long sum = 0 ; for ( int j = 1 ; j <= 9 ; j ++ ) sum += dp [ j ] ; return sum ; }
static void Main ( ) { int n = 2 ; Console . WriteLine ( answer ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG1 { static int MAX = 100000 ;
static long [ ] catalan = new long [ MAX ] ;
static void catalanDP ( long n ) {
catalan [ 0 ] = catalan [ 1 ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { catalan [ i ] = 0 ; for ( int j = 0 ; j < i ; j ++ ) { catalan [ i ] += catalan [ j ] * catalan [ i - j - 1 ] ; } } }
static int CatalanSequence ( int [ ] arr , int n ) {
catalanDP ( n ) ; HashSet < int > s = new HashSet < int > ( ) ;
int a = 1 , b = 1 ;
s . Add ( a ) ; if ( n >= 2 ) { s . Add ( b ) ; } for ( int i = 2 ; i < n ; i ++ ) { s . Add ( ( int ) catalan [ i ] ) ; } for ( int i = 0 ; i < n ; i ++ ) {
if ( s . Contains ( arr [ i ] ) ) { s . Remove ( arr [ i ] ) ; } }
return s . Count ; }
public static void Main ( ) { int [ ] arr = { 1 , 1 , 2 , 5 , 41 } ; int n = arr . Length ; Console . WriteLine ( CatalanSequence ( arr , n ) ) ; } }
static int solve ( int [ ] a , int n ) { int max1 = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( Math . Abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . Abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
static public void Main ( ) { int [ ] arr = { - 1 , 2 , 3 , - 4 , - 10 , 22 } ; int size = arr . Length ; Console . WriteLine ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
static int solve ( int [ ] a , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . Abs ( min1 - max1 ) ; }
public static void Main ( ) { int [ ] arr = { - 1 , 2 , 3 , 4 , - 10 } ; int size = arr . Length ; Console . WriteLine ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
static int minElements ( int [ ] arr , int n ) {
int halfSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = halfSum / 2 ;
Array . Sort ( arr ) ; int res = 0 , curr_sum = 0 ; for ( int i = n - 1 ; i >= 0 ; i -- ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
public static void Main ( ) { int [ ] arr = { 3 , 1 , 7 , 1 } ; int n = arr . Length ; Console . WriteLine ( minElements ( arr , n ) ) ; } }
static int minCost ( int N , int P , int Q ) {
int cost = 0 ;
while ( N > 0 ) { if ( ( N & 1 ) > 0 ) { cost += P ; N -- ; } else { int temp = N / 2 ;
if ( temp * P > Q ) cost += Q ;
else cost += * temp ; N /= 2 ; } }
return cost ; }
static void Main ( ) { int N = 9 , P = 5 , Q = 1 ; System . Console . WriteLine ( minCost ( N , P , Q ) ) ; } }
