using System ; class GFG {
static double Conversion ( double centi ) { double pixels = ( 96 * centi ) / 2.54 ; Console . WriteLine ( pixels ) ; return 0 ; }
public static void Main ( ) { double centi = 15 ; Conversion ( centi ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int xor_operations ( int N , int [ ] arr , int M , int K ) {
if ( M < 0 M >= N ) return - 1 ;
if ( K < 0 K >= N - M ) return - 1 ;
for ( int p = 0 ; p < M ; p ++ ) {
List < int > temp = new List < int > ( ) ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
int value = arr [ i ] ^ arr [ i + 1 ] ;
temp . Add ( value ) ;
arr [ i ] = temp [ i ] ; } }
int ans = arr [ K ] ; return ans ; }
public static void Main ( String [ ] args ) {
int N = 5 ;
int [ ] arr = { 1 , 4 , 5 , 6 , 7 } ; int M = 1 , K = 2 ;
Console . Write ( xor_operations ( N , arr , M , K ) ) ; } }
using System ; class GFG {
public static void canBreakN ( long n ) {
for ( long i = 2 ; ; i ++ ) {
long m = i * ( i + 1 ) / 2 ;
if ( m > n ) break ; long k = n - m ;
if ( k % i != 0 ) continue ;
Console . Write ( i ) ; return ; }
Console . Write ( " - 1" ) ; }
public static void Main ( string [ ] args ) {
long N = 12 ;
canBreakN ( N ) ; } }
using System ; class GFG {
public static void findCoprimePair ( int N ) {
for ( int x = 2 ; x <= Math . Sqrt ( N ) ; x ++ ) { if ( N % x == 0 ) {
while ( N % x == 0 ) { N /= x ; } if ( N > 1 ) {
Console . WriteLine ( x + " ▁ " + N ) ; return ; } } }
Console . WriteLine ( - 1 ) ; }
public static void Main ( String [ ] args ) {
int N = 45 ; findCoprimePair ( N ) ;
N = 25 ; findCoprimePair ( N ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 10000 ;
static List < int > primes = new List < int > ( ) ;
static void sieveSundaram ( ) {
bool [ ] marked = new bool [ MAX / 2 + 1 ] ;
for ( int i = 1 ; i <= ( Math . Sqrt ( MAX ) - 1 ) / 2 ; i ++ ) { for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= MAX / 2 ; j = j + 2 * i + 1 ) { marked [ j ] = true ; } }
primes . Add ( 2 ) ;
for ( int i = 1 ; i <= MAX / 2 ; i ++ ) if ( marked [ i ] == false ) primes . Add ( 2 * i + 1 ) ; }
static bool isWasteful ( int n ) { if ( n == 1 ) return false ;
int original_no = n ; int sumDigits = 0 ; while ( original_no > 0 ) { sumDigits ++ ; original_no = original_no / 10 ; } int pDigit = 0 , count_exp = 0 , p = 0 ;
for ( int i = 0 ; primes [ i ] <= n / 2 ; i ++ ) {
while ( n % primes [ i ] == 0 ) {
p = primes [ i ] ; n = n / p ;
count_exp ++ ; }
while ( p > 0 ) { pDigit ++ ; p = p / 10 ; }
while ( count_exp > 1 ) { pDigit ++ ; count_exp = count_exp / 10 ; } }
if ( n != 1 ) { while ( n > 0 ) { pDigit ++ ; n = n / 10 ; } }
return ( pDigit > sumDigits ) ; }
static void Solve ( int N ) {
for ( int i = 1 ; i < N ; i ++ ) { if ( isWasteful ( i ) ) { Console . Write ( i + " ▁ " ) ; } } }
public static void Main ( String [ ] args ) {
sieveSundaram ( ) ; int N = 10 ;
Solve ( N ) ; } }
using System ; class GFG {
static int printhexaRec ( int n ) { if ( n == 0 n == 1 n == 2 n == 3 n == 4 n == 5 ) return 0 ; else if ( n == 6 ) return 1 ; else return ( printhexaRec ( n - 1 ) + printhexaRec ( n - 2 ) + printhexaRec ( n - 3 ) + printhexaRec ( n - 4 ) + printhexaRec ( n - 5 ) + printhexaRec ( n - 6 ) ) ; } static void printhexa ( int n ) { Console . Write ( printhexaRec ( n ) + " STRNEWLINE " ) ; }
public static void Main ( ) { int n = 11 ; printhexa ( n ) ; } }
using System ; class GFG {
static void printhexa ( int n ) { if ( n < 0 ) return ;
int first = 0 ; int second = 0 ; int third = 0 ; int fourth = 0 ; int fifth = 0 ; int sixth = 1 ;
int curr = 0 ; if ( n < 6 ) Console . WriteLine ( first ) ; else if ( n == 6 ) Console . WriteLine ( sixth ) ; else {
for ( int i = 6 ; i < n ; i ++ ) { curr = first + second + third + fourth + fifth + sixth ; first = second ; second = third ; third = fourth ; fourth = fifth ; fifth = sixth ; sixth = curr ; } } Console . WriteLine ( curr ) ; }
public static void Main ( String [ ] args ) { int n = 11 ; printhexa ( n ) ; } }
using System ; class GFG {
static void smallestNumber ( int N ) { Console . WriteLine ( ( N % 9 + 1 ) * Math . Pow ( 10 , ( N / 9 ) ) - 1 ) ; }
public static void Main ( ) { int N = 10 ; smallestNumber ( N ) ; } }
using System ; using System . Collections . Generic ; class GFG { static List < int > compo = new List < int > ( ) ;
static bool isComposite ( int n ) {
if ( n <= 3 ) return false ;
if ( n % 2 == 0 n % 3 == 0 ) return true ; int i = 5 ; while ( i * i <= n ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) return true ; i = i + 6 ; } return false ; }
static void Compositorial_list ( int n ) { int l = 0 ; for ( int i = 4 ; i < 1000000 ; i ++ ) { if ( l < n ) { if ( isComposite ( i ) ) { compo . Add ( i ) ; l += 1 ; } } } }
static int calculateCompositorial ( int n ) {
int result = 1 ; for ( int i = 0 ; i < n ; i ++ ) result = result * compo [ i ] ; return result ; }
public static void Main ( String [ ] args ) { int n = 5 ;
Compositorial_list ( n ) ; Console . Write ( ( calculateCompositorial ( n ) ) ) ; } }
using System ; class GFG {
static int [ ] b = new int [ 50 ] ;
static int PowerArray ( int n , int k ) {
int count = 0 ;
while ( k > 0 ) { if ( k % n == 0 ) { k /= n ; count ++ ; }
else if ( k % n == 1 ) { k -= 1 ; b [ count ] ++ ;
if ( b [ count ] > 1 ) { Console . Write ( - 1 ) ; return 0 ; } }
else { Console . Write ( - 1 ) ; return 0 ; } }
for ( int i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] != 0 ) { Console . Write ( i + " , ▁ " ) ; } } return int . MinValue ; }
public static void Main ( String [ ] args ) { int N = 3 ; int K = 40 ; PowerArray ( N , K ) ; } }
using System ; public class GFG {
static int findSum ( int N , int k ) {
int sum = 0 ; for ( int i = 1 ; i <= N ; i ++ ) {
sum += ( int ) Math . Pow ( i , k ) ; }
return sum ; }
public static void Main ( string [ ] args ) { int N = 8 , k = 4 ;
Console . WriteLine ( findSum ( N , k ) ) ; } }
using System ; class GFG {
static int countIndices ( int [ ] arr , int n ) {
int cnt = 0 ;
int max = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( max < arr [ i ] ) {
max = arr [ i ] ;
cnt ++ ; } } return cnt ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 } ; int n = arr . Length ; Console . WriteLine ( countIndices ( arr , n ) ) ; } }
using System ; class GFG {
static String [ ] bin = { "000" , "001" , "010" , "011" , "100" , "101" , "110" , "111" } ;
static int maxFreq ( String s ) {
String binary = " " ;
for ( int K = 0 ; K < s . Length ; K ++ ) { binary += bin [ s [ K ] - '0' ] ; }
binary = binary . Substring ( 0 , binary . Length - 1 ) ; int count = 1 , prev = - 1 , i , j = 0 ; for ( i = binary . Length - 1 ; i >= 0 ; i -- , j ++ )
if ( binary [ i ] == '1' ) {
count = Math . Max ( count , j - prev ) ; prev = j ; } return count ; }
public static void Main ( String [ ] args ) { String octal = "13" ; Console . WriteLine ( maxFreq ( octal ) ) ; } }
using System ; class GFG { static int sz = 100000 ; static bool [ ] isPrime = new bool [ sz + 1 ] ;
static void sieve ( ) { for ( int i = 0 ; i <= sz ; i ++ ) isPrime [ i ] = true ; isPrime [ 0 ] = isPrime [ 1 ] = false ; for ( int i = 2 ; i * i <= sz ; i ++ ) { if ( isPrime [ i ] ) { for ( int j = i * i ; j < sz ; j += i ) { isPrime [ j ] = false ; } } } }
static void findPrimesD ( int d ) {
int left = ( int ) Math . Pow ( 10 , d - 1 ) ; int right = ( int ) Math . Pow ( 10 , d ) - 1 ;
for ( int i = left ; i <= right ; i ++ ) {
if ( isPrime [ i ] ) { Console . Write ( i + " ▁ " ) ; } } }
static public void Main ( ) {
sieve ( ) ; int d = 1 ; findPrimesD ( d ) ; } }
using System ; class GFG {
public static int Cells ( int n , int x ) { if ( n <= 0 x <= 0 x > n * n ) return 0 ; int i = 0 , count = 0 ; while ( ++ i * i < x ) if ( x % i == 0 && x <= n * i ) count += 2 ; return i * i == x ? count + 1 : count ; }
static public void Main ( ) { int n = 6 , x = 12 ;
Console . WriteLine ( Cells ( n , x ) ) ; } }
using System ; class GFG {
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
using System ; class GFG {
public static void Alphabet_N_Pattern ( int N ) { int index , side_index ;
int Right = 1 , Left = 1 , Diagonal = 2 ;
for ( index = 0 ; index < N ; index ++ ) {
Console . Write ( Left ++ ) ;
for ( side_index = 0 ; side_index < 2 * ( index ) ; side_index ++ ) Console . Write ( " ▁ " ) ;
if ( index != 0 && index != N - 1 ) Console . Write ( Diagonal ++ ) ; else Console . Write ( " ▁ " ) ;
for ( side_index = 0 ; side_index < 2 * ( N - index - 1 ) ; side_index ++ ) Console . Write ( " ▁ " ) ;
Console . Write ( Right ++ ) ; Console . Write ( " STRNEWLINE " ) ; } }
static void Main ( ) {
int Size = 6 ;
Alphabet_N_Pattern ( Size ) ; } }
using System ;
class GFG { public int isSumDivides ( int N ) { int temp = N , sum = 0 ;
while ( temp > 0 ) { sum += temp % 10 ; temp /= 10 ; } if ( N % sum == 0 ) return 1 ; else return 0 ; }
public static void Main ( ) { GFG g = new GFG ( ) ; int N = 12 ; if ( g . isSumDivides ( N ) > 0 ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; class GFG {
static int sum ( int N ) { int S1 , S2 , S3 ; S1 = ( ( N / 3 ) ) * ( 2 * 3 + ( N / 3 - 1 ) * 3 ) / 2 ; S2 = ( ( N / 4 ) ) * ( 2 * 4 + ( N / 4 - 1 ) * 4 ) / 2 ; S3 = ( ( N / 12 ) ) * ( 2 * 12 + ( N / 12 - 1 ) * 12 ) / 2 ; return S1 + S2 - S3 ; }
public static void Main ( ) { int N = 20 ; Console . WriteLine ( sum ( 12 ) ) ; } }
using System ; class GFG {
static int nextGreater ( int N ) { int power_of_2 = 1 , shift_count = 0 ;
while ( true ) {
if ( ( ( N >> shift_count ) & 1 ) % 2 == 0 ) break ;
shift_count ++ ;
power_of_2 = power_of_2 * 2 ; }
return ( N + power_of_2 ) ; }
public static void Main ( ) { int N = 11 ;
Console . WriteLine ( " The ▁ next ▁ number ▁ is ▁ = ▁ " + nextGreater ( N ) ) ; } }
using System ; class GFG {
static int countWays ( int n ) {
if ( n == 0 ) return 1 ; if ( n <= 2 ) return n ;
int f0 = 1 , f1 = 1 , f2 = 2 ; int ans = 0 ;
for ( int i = 3 ; i <= n ; i ++ ) { ans = f0 + f1 + f2 ; f0 = f1 ; f1 = f2 ; f2 = ans ; }
return ans ; }
public static void Main ( String [ ] args ) { int n = 4 ; Console . WriteLine ( countWays ( n ) ) ; } }
using System ; class GFG { static int n = 6 , m = 6 ;
static void maxSum ( long [ , ] arr ) {
long [ , ] dp = new long [ n + 1 , 3 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
long m1 = 0 , m2 = 0 , m3 = 0 ; for ( int j = 0 ; j < m ; j ++ ) {
if ( ( j / ( m / 3 ) ) == 0 ) { m1 = Math . Max ( m1 , arr [ i , j ] ) ; }
else if ( ( j / ( m / 3 ) ) == 1 ) { m2 = Math . Max ( m2 , arr [ i , j ] ) ; }
else if ( ( j / ( m / 3 ) ) == 2 ) { m3 = Math . Max ( m3 , arr [ i , j ] ) ; } }
dp [ i + 1 , 0 ] = Math . Max ( dp [ i , 1 ] , dp [ i , 2 ] ) + m1 ; dp [ i + 1 , 1 ] = Math . Max ( dp [ i , 0 ] , dp [ i , 2 ] ) + m2 ; dp [ i + 1 , 2 ] = Math . Max ( dp [ i , 1 ] , dp [ i , 0 ] ) + m3 ; }
Console . Write ( Math . Max ( Math . Max ( dp [ n , 0 ] , dp [ n , 1 ] ) , dp [ n , 2 ] ) + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { long [ , ] arr = { { 1 , 3 , 5 , 2 , 4 , 6 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 1 , 3 , 5 , 2 , 4 , 6 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 1 , 3 , 5 , 2 , 4 , 6 } } ; maxSum ( arr ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void solve ( char [ ] s ) { int n = s . Length ;
int [ , ] dp = new int [ n , n ] ;
for ( int len = n - 1 ; len >= 0 ; -- len ) {
for ( int i = 0 ; i + len < n ; ++ i ) {
int j = i + len ;
if ( i == 0 && j == n - 1 ) { if ( s [ i ] == s [ j ] ) dp [ i , j ] = 2 ; else if ( s [ i ] != s [ j ] ) dp [ i , j ] = 1 ; } else { if ( s [ i ] == s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i , j ] += dp [ i - 1 , j ] ; } if ( j + 1 <= n - 1 ) { dp [ i , j ] += dp [ i , j + 1 ] ; } if ( i - 1 < 0 j + 1 >= n ) {
dp [ i , j ] += 1 ; } } else if ( s [ i ] != s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i , j ] += dp [ i - 1 , j ] ; } if ( j + 1 <= n - 1 ) { dp [ i , j ] += dp [ i , j + 1 ] ; } if ( i - 1 >= 0 && j + 1 <= n - 1 ) {
dp [ i , j ] -= dp [ i - 1 , j + 1 ] ; } } } } } List < int > ways = new List < int > ( ) ; for ( int i = 0 ; i < n ; ++ i ) { if ( i == 0 i == n - 1 ) {
ways . Add ( 1 ) ; } else {
int total = dp [ i - 1 , i + 1 ] ; ways . Add ( total ) ; } } for ( int i = 0 ; i < ways . Capacity ; ++ i ) { Console . Write ( ways [ i ] + " ▁ " ) ; } }
public static void Main ( ) { char [ ] s = " xyxyx " . ToCharArray ( ) ; solve ( s ) ; } }
using System ; class GFG {
static long getChicks ( int n ) {
int size = Math . Max ( n , 7 ) ; long [ ] dp = new long [ size ] ; dp [ 0 ] = 0 ; dp [ 1 ] = 1 ;
for ( int i = 2 ; i < 6 ; i ++ ) { dp [ i ] = dp [ i - 1 ] * 3 ; }
dp [ 6 ] = 726 ;
for ( int i = 8 ; i <= n ; i ++ ) {
dp [ i ] = ( dp [ i - 1 ] - ( 2 * dp [ i - 6 ] / 3 ) ) * 3 ; } return dp [ n ] ; }
static public void Main ( ) { int n = 3 ; Console . WriteLine ( getChicks ( n ) ) ; } }
using System ; class GFG {
static int getChicks ( int n ) { int chicks = ( int ) Math . Pow ( 3 , n - 1 ) ; return chicks ; }
public static void Main ( ) { int n = 3 ; Console . WriteLine ( getChicks ( n ) ) ; } }
using System ; class GFG { static int n = 3 ;
static int [ , ] dp = new int [ n , n ] ;
static int [ , ] v = new int [ n , n ] ;
static int minSteps ( int i , int j , int [ , ] arr ) {
if ( i == n - 1 && j == n - 1 ) { return 0 ; } if ( i > n - 1 j > n - 1 ) { return 9999999 ; }
if ( v [ i , j ] == 1 ) { return dp [ i , j ] ; } v [ i , j ] = 1 ; dp [ i , j ] = 9999999 ;
for ( int k = Math . Max ( 0 , arr [ i , j ] + j - n + 1 ) ; k <= Math . Min ( n - i - 1 , arr [ i , j ] ) ; k ++ ) { dp [ i , j ] = Math . Min ( dp [ i , j ] , minSteps ( i + k , j + arr [ i , j ] - k , arr ) ) ; } dp [ i , j ] ++ ; return dp [ i , j ] ; }
static public void Main ( ) { int [ , ] arr = { { 4 , 1 , 2 } , { 1 , 1 , 1 } , { 2 , 1 , 1 } } ; int ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) { Console . WriteLine ( - 1 ) ; } else { Console . WriteLine ( ans ) ; } } }
using System ; class GFG { static int n = 3 ;
static int [ , ] dp = new int [ n , n ] ;
static int [ , ] v = new int [ n , n ] ;
static int minSteps ( int i , int j , int [ , ] arr ) {
if ( i == n - 1 && j == n - 1 ) { return 0 ; } if ( i > n - 1 j > n - 1 ) { return 9999999 ; }
if ( v [ i , j ] == 1 ) { return dp [ i , j ] ; } v [ i , j ] = 1 ;
dp [ i , j ] = 1 + Math . Min ( minSteps ( i + arr [ i , j ] , j , arr ) , minSteps ( i , j + arr [ i , j ] , arr ) ) ; return dp [ i , j ] ; }
static public void Main ( ) { int [ , ] arr = { { 2 , 1 , 2 } , { 1 , 1 , 1 } , { 1 , 1 , 1 } } ; int ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) { Console . WriteLine ( - 1 ) ; } else { Console . WriteLine ( ans ) ; } } }
using System ; class GFG { static int MAX = 1001 ; static int [ , ] dp = new int [ MAX , MAX ] ;
static int MaxProfit ( int [ ] treasure , int [ ] color , int n , int k , int col , int A , int B ) {
return dp [ k , col ] = 0 ; if ( dp [ k , col ] != - 1 ) return dp [ k , col ] ; int sum = 0 ;
if ( col == color [ k ] ) sum += Math . Max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += Math . Max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return dp [ k , col ] = sum ; }
public static void Main ( String [ ] args ) { int A = - 5 , B = 7 ; int [ ] treasure = { 4 , 8 , 2 , 9 } ; int [ ] color = { 2 , 2 , 6 , 2 } ; int n = color . Length ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < MAX ; j ++ ) dp [ i , j ] = - 1 ; Console . Write ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) ; } }
class GFG {
static void printTetra ( int n ) { int [ ] dp = new int [ n + 5 ] ;
dp [ 0 ] = 0 ; dp [ 1 ] = dp [ 2 ] = 1 ; dp [ 3 ] = 2 ; for ( int i = 4 ; i <= n ; i ++ ) dp [ i ] = dp [ i - 1 ] + dp [ i - 2 ] + dp [ i - 3 ] + dp [ i - 4 ] ; System . Console . WriteLine ( dp [ n ] ) ; }
static void Main ( ) { int n = 10 ; printTetra ( n ) ; } }
using System ; class GFG {
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
using System ; class GFG {
static int permutationCoeff ( int n , int k ) { int [ , ] P = new int [ n + 2 , k + 2 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= Math . Min ( i , k ) ; j ++ ) {
if ( j == 0 ) P [ i , j ] = 1 ;
else P [ i , j ] = P [ i - 1 , j ] + ( j * P [ i - 1 , j - 1 ] ) ;
P [ i , j + 1 ] = 0 ; } } return P [ n , k ] ; }
public static void Main ( ) { int n = 10 , k = 2 ; Console . WriteLine ( " Value ▁ of ▁ P ( ▁ " + n + " , " + k + " ) " + " ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
using System ; public class GFG {
static int permutationCoeff ( int n , int k ) { int [ ] fact = new int [ n + 1 ] ;
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = i * fact [ i - 1 ] ;
return fact [ n ] / fact [ n - k ] ; }
static public void Main ( ) { int n = 10 , k = 2 ; Console . WriteLine ( " Value ▁ of " + " ▁ P ( ▁ " + n + " , ▁ " + k + " ) ▁ is ▁ " + permutationCoeff ( n , k ) ) ; } }
using System ; class GFG {
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
using System ; class GFG {
public static void compute_z ( string s , int [ ] z ) { int l = 0 , r = 0 ; int n = s . Length ; for ( int i = 1 ; i <= n - 1 ; i ++ ) { if ( i > r ) { l = i ; r = i ; while ( r < n && s [ r - l ] == s [ r ] ) { r ++ ; } z [ i ] = r - l ; r -- ; } else { int k = i - l ; if ( z [ k ] < r - i + 1 ) { z [ i ] = z [ k ] ; } else { l = i ; while ( r < n && s [ r - l ] == s [ r ] ) { r ++ ; } z [ i ] = r - l ; r -- ; } } } }
public static int countPermutation ( string a , string b ) {
b = b + b ;
b = b . Substring ( 0 , b . Length - 1 ) ;
int ans = 0 ; string s = a + " $ " + b ; int n = s . Length ;
int [ ] z = new int [ n ] ; compute_z ( s , z ) ; for ( int i = 1 ; i <= n - 1 ; i ++ ) {
if ( z [ i ] == a . Length ) { ans ++ ; } } return ans ; }
public static void Main ( string [ ] args ) { string a = "101" ; string b = "101" ; Console . WriteLine ( countPermutation ( a , b ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static void smallestSubsequence ( char [ ] S , int K ) {
int N = S . Length ;
Stack < char > answer = new Stack < char > ( ) ;
for ( int i = 0 ; i < N ; ++ i ) {
if ( answer . Count == 0 ) { answer . Push ( S [ i ] ) ; } else {
while ( ( answer . Count != 0 ) && ( S [ i ] < answer . Peek ( ) )
& & ( answer . Count - 1 + N - i >= K ) ) { answer . Pop ( ) ; }
if ( answer . Count == 0 answer . Count < K ) {
answer . Push ( S [ i ] ) ; } } }
String ret = " " ;
while ( answer . Count != 0 ) { ret += ( answer . Peek ( ) ) ; answer . Pop ( ) ; }
ret = reverse ( ret ) ;
Console . Write ( ret ) ; } static String reverse ( String input ) { char [ ] a = input . ToCharArray ( ) ; int l , r = a . Length - 1 ; for ( l = 0 ; l < r ; l ++ , r -- ) { char temp = a [ l ] ; a [ l ] = a [ r ] ; a [ r ] = temp ; } return String . Join ( " " , a ) ; }
public static void Main ( String [ ] args ) { String S = " aabdaabc " ; int K = 3 ; smallestSubsequence ( S . ToCharArray ( ) , K ) ; } }
using System ; class GFG {
public static bool is_rtol ( String s ) { int tmp = ( int ) ( Math . Sqrt ( s . Length ) ) - 1 ; char first = s [ tmp ] ;
for ( int pos = tmp ; pos < s . Length - 1 ; pos += tmp ) {
if ( s [ pos ] != first ) { return false ; } } return true ; }
public static void Main ( String [ ] args ) {
String str = " abcxabxcaxbcxabc " ;
if ( is_rtol ( str ) ) { Console . Write ( " Yes " ) ; } else { Console . Write ( " No " ) ; } } }
using System ; class GFG {
static bool check ( string str , int K ) {
if ( str . Length % K == 0 ) { int sum = 0 , i ;
for ( i = 0 ; i < K ; i ++ ) { sum += str [ i ] ; }
for ( int j = i ; j < str . Length ; j += K ) { int s_comp = 0 ; for ( int p = j ; p < j + K ; p ++ ) s_comp += str [ p ] ;
if ( s_comp != sum )
return false ; }
return true ; }
return false ; }
public static void Main ( string [ ] args ) { int K = 3 ; string str = " abdcbbdba " ; if ( check ( str , K ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static int maxSum ( string str ) { int maximumSum = 0 ;
int totalOnes = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( str [ i ] == '1' ) { totalOnes ++ ; } }
int zero = 0 , ones = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( str [ i ] == '0' ) { zero ++ ; } else { ones ++ ; }
maximumSum = Math . Max ( maximumSum , zero + ( totalOnes - ones ) ) ; } return maximumSum ; }
public static void Main ( string [ ] args ) {
string str = "011101" ;
Console . Write ( maxSum ( str ) ) ; } }
using System ; class GFG {
static int maxLenSubStr ( String s ) {
if ( s . Length < 3 ) return s . Length ;
int temp = 2 ; int ans = 2 ;
for ( int i = 2 ; i < s . Length ; i ++ ) {
if ( s [ i ] != s [ i - 1 ] s [ i ] != s [ i - 2 ] ) temp ++ ;
else { ans = Math . Max ( temp , ans ) ; temp = 2 ; } } ans = Math . Max ( temp , ans ) ; return ans ; }
static public void Main ( ) { String s = " baaabbabbb " ; Console . Write ( maxLenSubStr ( s ) ) ; } }
using System ; class GFG {
static int no_of_ways ( string s ) { int n = s . Length ;
int count_left = 0 , count_right = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { if ( s [ i ] == s [ 0 ] ) { ++ count_left ; } else break ; }
for ( int i = n - 1 ; i >= 0 ; -- i ) { if ( s [ i ] == s [ n - 1 ] ) { ++ count_right ; } else break ; }
if ( s [ 0 ] == s [ n - 1 ] ) return ( ( count_left + 1 ) * ( count_right + 1 ) ) ;
else return ( + count_right + 1 ) ; }
public static void Main ( ) { string s = " geeksforgeeks " ; Console . WriteLine ( no_of_ways ( s ) ) ; } }
using System ; class GFG {
static void preCompute ( int n , string s , int [ ] pref ) { pref [ 0 ] = 0 ; for ( int i = 1 ; i < n ; i ++ ) { pref [ i ] = pref [ i - 1 ] ; if ( s [ i - 1 ] == s [ i ] ) pref [ i ] ++ ; } }
static int query ( int [ ] pref , int l , int r ) { return pref [ r ] - pref [ l ] ; }
public static void Main ( ) { string s = " ggggggg " ; int n = s . Length ; int [ ] pref = new int [ n ] ; preCompute ( n , s , pref ) ;
int l = 1 ; int r = 2 ; Console . WriteLine ( query ( pref , l , r ) ) ;
l = 1 ; r = 5 ; Console . WriteLine ( query ( pref , l , r ) ) ; } }
using System ; class GFG {
static String findDirection ( String s ) { int count = 0 ; String d = " " ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( s [ 0 ] == ' STRNEWLINE ' ) return null ; if ( s [ i ] == ' L ' ) count -- ; else { if ( s [ i ] == ' R ' ) count ++ ; } }
if ( count > 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == 1 ) d = " E " ; else if ( count % 4 == 2 ) d = " S " ; else if ( count % 4 == 3 ) d = " W " ; }
if ( count < 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == - 1 ) d = " W " ; else if ( count % 4 == - 2 ) d = " S " ; else if ( count % 4 == - 3 ) d = " E " ; } return d ; }
public static void Main ( ) { String s = " LLRLRRL " ; Console . WriteLine ( findDirection ( s ) ) ; s = " LL " ; Console . WriteLine ( findDirection ( s ) ) ; } }
using System ; class GFG {
static bool isCheck ( string str ) { int len = str . Length ; string lowerStr = " " , upperStr = " " ; char [ ] str1 = str . ToCharArray ( ) ;
for ( int i = 0 ; i < len ; i ++ ) {
if ( ( int ) ( str1 [ i ] ) >= 65 && ( int ) str1 [ i ] <= 91 ) upperStr = upperStr + str1 [ i ] ; else lowerStr = lowerStr + str1 [ i ] ; }
String transformStr = lowerStr . ToUpper ( ) ; return ( transformStr . Equals ( upperStr ) ) ; }
public static void Main ( String [ ] args ) { String str = " geeGkEEsKS " ; if ( isCheck ( str ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; public class GFG {
static void encode ( String s , int k ) {
String newS = " " ;
for ( int i = 0 ; i < s . Length ; ++ i ) {
int val = s [ i ] ;
int dup = k ;
if ( val + k > 122 ) { k -= ( 122 - val ) ; k = k % 26 ; newS += ( char ) ( 96 + k ) ; } else { newS += ( char ) ( 96 + k ) ; } k = dup ; }
Console . Write ( newS ) ; }
public static void Main ( ) { String str = " abc " ; int k = 28 ;
encode ( str , k ) ; } }
using System ; class GFG {
static bool isVowel ( char x ) { if ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) return true ; else return false ; }
static String updateSandwichedVowels ( String a ) { int n = a . Length ;
String updatedString = " " ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i == 0 i == n - 1 ) { updatedString += a [ i ] ; continue ; }
if ( ( isVowel ( a [ i ] ) ) == true && isVowel ( a [ i - 1 ] ) == false && isVowel ( a [ i + 1 ] ) == false ) { continue ; }
updatedString += a [ i ] ; } return updatedString ; }
public static void Main ( ) { String str = " geeksforgeeks " ;
String updatedString = updateSandwichedVowels ( str ) ; Console . WriteLine ( updatedString ) ; }
using System ; using System . Collections . Generic ; class GFG {
class Node { public int data ; public Node left , right ; } ; static int ans ;
static Node newNode ( int data ) { Node newNode = new Node ( ) ; newNode . data = data ; newNode . left = newNode . right = null ; return ( newNode ) ; }
static void findPathUtil ( Node root , int k , List < int > path , int flag ) { if ( root == null ) return ;
if ( root . data >= k ) flag = 1 ;
if ( root . left == null && root . right == null ) { if ( flag == 1 ) { ans = 1 ; Console . Write ( " ( " ) ; for ( int i = 0 ; i < path . Count ; i ++ ) { Console . Write ( path [ i ] + " , ▁ " ) ; } Console . Write ( root . data + " ) , ▁ " ) ; } return ; }
path . Add ( root . data ) ;
findPathUtil ( root . left , k , path , flag ) ; findPathUtil ( root . right , k , path , flag ) ;
path . RemoveAt ( path . Count - 1 ) ; }
static void findPath ( Node root , int k ) {
int flag = 0 ;
ans = 0 ; List < int > v = new List < int > ( ) ;
findPathUtil ( root , k , v , flag ) ;
if ( ans == 0 ) Console . Write ( " - 1" ) ; }
public static void Main ( String [ ] args ) { int K = 25 ;
Node root = newNode ( 10 ) ; root . left = newNode ( 5 ) ; root . right = newNode ( 8 ) ; root . left . left = newNode ( 29 ) ; root . left . right = newNode ( 2 ) ; root . right . right = newNode ( 98 ) ; root . right . left = newNode ( 1 ) ; root . right . right . right = newNode ( 50 ) ; root . left . left . left = newNode ( 20 ) ; findPath ( root , K ) ; } }
using System ; class GFG {
static int Tridecagonal_num ( int n ) {
return ( 11 * n * n - 9 * n ) / 2 ; }
public static void Main ( String [ ] args ) { int n = 3 ; Console . Write ( Tridecagonal_num ( n ) + " STRNEWLINE " ) ; n = 10 ; Console . Write ( Tridecagonal_num ( n ) + " STRNEWLINE " ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
public class Node { public int data ; public Node left ; public Node right ; } ;
static Node newNode ( int k ) { Node node = new Node ( ) ; node . data = k ; node . right = node . left = null ; return node ; } static bool isHeap ( Node root ) { Queue < Node > q = new Queue < Node > ( ) ; q . Enqueue ( root ) ; bool nullish = false ; while ( q . Count != 0 ) { Node temp = q . Peek ( ) ; q . Dequeue ( ) ; if ( temp . left != null ) { if ( nullish temp . left . data >= temp . data ) { return false ; } q . Enqueue ( temp . left ) ; } else { nullish = true ; } if ( temp . right != null ) { if ( nullish temp . right . data >= temp . data ) { return false ; } q . Enqueue ( temp . right ) ; } else { nullish = true ; } } return true ; }
public static void Main ( String [ ] args ) { Node root = null ; root = newNode ( 10 ) ; root . left = newNode ( 9 ) ; root . right = newNode ( 8 ) ; root . left . left = newNode ( 7 ) ; root . left . right = newNode ( 6 ) ; root . right . left = newNode ( 5 ) ; root . right . right = newNode ( 4 ) ; root . left . left . left = newNode ( 3 ) ; root . left . left . right = newNode ( 2 ) ; root . left . right . left = newNode ( 1 ) ;
if ( isHeap ( root ) ) Console . Write ( " Given ▁ binary ▁ tree ▁ is ▁ a ▁ Heap STRNEWLINE " ) ; else Console . Write ( " Given ▁ binary ▁ tree ▁ is ▁ not ▁ a ▁ Heap STRNEWLINE " ) ; } }
using System ; class GFG {
static int findNumbers ( int n , int w ) { int x = 0 , sum = 0 ;
if ( w >= 0 && w <= 8 ) {
x = 9 - w ; }
else if ( w >= - 9 && w <= - 1 ) {
x = 10 + w ; } sum = ( int ) Math . Pow ( 10 , n - 2 ) ; sum = ( x * sum ) ; return sum ; }
static public void Main ( ) { int n , w ;
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
using System ; class PellNumber {
public static int pell ( int n ) { if ( n <= 2 ) return n ; int a = 1 ; int b = 2 ; int c ; for ( int i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
public static void Main ( ) { int n = 4 ; Console . Write ( pell ( n ) ) ; } }
using System ; class GFG {
static bool isMultipleOf10 ( int n ) { if ( n % 15 == 0 ) return true ; return false ; }
public static void Main ( ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; public class GFG {
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
using System ; class Test {
static int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
public static void Main ( ) { int num = 5 ; Console . WriteLine ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + factorial ( 5 ) ) ; } }
using System ; class GFG {
static int FlipBits ( int n ) { return n -= ( n & ( - n ) ) ; }
public static void Main ( String [ ] args ) { int N = 12 ; Console . Write ( " The ▁ number ▁ after " + " unsetting ▁ the ▁ " ) ; Console . Write ( " rightmost ▁ set ▁ bit : ▁ " + FlipBits ( N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void Maximum_xor_Triplet ( int n , int [ ] a ) {
HashSet < int > s = new HashSet < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i ; j < n ; j ++ ) {
s . Add ( a [ i ] ^ a [ j ] ) ; } } int ans = 0 ; foreach ( int i in s ) { for ( int j = 0 ; j < n ; j ++ ) {
ans = Math . Max ( ans , i ^ a [ j ] ) ; } } Console . WriteLine ( ans ) ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 3 , 8 , 15 } ; int n = a . Length ; Maximum_xor_Triplet ( n , a ) ; } }
using System ; class GFG {
static void printMissing ( int [ ] ar , int low , int high ) { Array . Sort ( ar ) ;
int index = ceilindex ( ar , low , 0 , ar . Length - 1 ) ; int x = low ;
while ( index < ar . Length && x <= high ) {
if ( ar [ index ] != x ) { Console . Write ( x + " ▁ " ) ; }
else index ++ ;
x ++ ; }
while ( x <= high ) { Console . Write ( x + " ▁ " ) ; x ++ ; } }
static int ceilindex ( int [ ] ar , int val , int low , int high ) { if ( val < ar [ 0 ] ) return 0 ; if ( val > ar [ ar . Length - 1 ] ) return ar . Length ; int mid = ( low + high ) / 2 ; if ( ar [ mid ] == val ) return mid ; if ( ar [ mid ] < val ) { if ( mid + 1 < high && ar [ mid + 1 ] >= val ) return mid + 1 ; return ceilindex ( ar , val , mid + 1 , high ) ; } else { if ( mid - 1 >= low && ar [ mid - 1 ] < val ) return mid ; return ceilindex ( ar , val , low , mid - 1 ) ; } }
static public void Main ( ) { int [ ] arr = { 1 , 3 , 5 , 4 } ; int low = 1 , high = 10 ; printMissing ( arr , low , high ) ; } }
using System ; class GFG {
static void printMissing ( int [ ] arr , int n , int low , int high ) {
bool [ ] points_of_range = new bool [ high - low + 1 ] ; for ( int i = 0 ; i < high - low + 1 ; i ++ ) points_of_range [ i ] = false ; for ( int i = 0 ; i < n ; i ++ ) {
if ( low <= arr [ i ] && arr [ i ] <= high ) points_of_range [ arr [ i ] - low ] = true ; }
for ( int x = 0 ; x <= high - low ; x ++ ) { if ( points_of_range [ x ] == false ) Console . Write ( " { 0 } ▁ " , low + x ) ; } }
public static void Main ( ) { int [ ] arr = { 1 , 3 , 5 , 4 } ; int n = arr . Length ; int low = 1 , high = 10 ; printMissing ( arr , n , low , high ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void printMissing ( int [ ] arr , int n , int low , int high ) {
HashSet < int > s = new HashSet < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { s . Add ( arr [ i ] ) ; }
for ( int x = low ; x <= high ; x ++ ) if ( ! s . Contains ( x ) ) Console . Write ( x + " ▁ " ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 3 , 5 , 4 } ; int n = arr . Length ; int low = 1 , high = 10 ; printMissing ( arr , n , low , high ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int find ( int [ ] a , int [ ] b , int k , int n1 , int n2 ) {
HashSet < int > s = new HashSet < int > ( ) ; for ( int i = 0 ; i < n2 ; i ++ ) s . Add ( b [ i ] ) ;
int missing = 0 ; for ( int i = 0 ; i < n1 ; i ++ ) { if ( ! s . Contains ( a [ i ] ) ) missing ++ ; if ( missing == k ) return a [ i ] ; } return - 1 ; }
public static void Main ( String [ ] args ) { int [ ] a = { 0 , 2 , 4 , 6 , 8 , 10 , 12 , 14 , 15 } ; int [ ] b = { 4 , 10 , 6 , 8 , 12 } ; int n1 = a . Length ; int n2 = b . Length ; int k = 3 ; Console . WriteLine ( find ( a , b , k , n1 , n2 ) ) ; } }
using System ; class GFG {
static void findString ( string S , int N ) {
int [ ] amounts = new int [ 26 ] ;
for ( int i = 0 ; i < 26 ; i ++ ) { amounts [ i ] = 0 ; }
for ( int i = 0 ; i < S . Length ; i ++ ) { amounts [ ( int ) ( S [ i ] - 97 ) ] ++ ; } int count = 0 ;
for ( int i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) count ++ ; }
if ( count > N ) { Console . Write ( " - 1" ) ; }
else { string ans = " " ; int high = 100001 ; int low = 0 ; int mid , total ;
while ( ( high - low ) > 1 ) { total = 0 ;
mid = ( high + low ) / 2 ;
for ( int i = 0 ; i < 26 ; i ++ ) {
if ( amounts [ i ] > 0 ) { total += ( amounts [ i ] - 1 ) / mid + 1 ; } }
if ( total <= N ) { high = mid ; } else { low = mid ; } } Console . Write ( high + " ▁ " ) ; total = 0 ;
for ( int i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) { total += ( amounts [ i ] - 1 ) / high + 1 ; for ( int j = 0 ; j < ( ( amounts [ i ] - 1 ) / high + 1 ) ; j ++ ) {
ans += ( char ) ( i + 97 ) ; } } }
for ( int i = total ; i < N ; i ++ ) { ans += ' a ' ; } string reverse = " " ; int Len = ans . Length - 1 ; while ( Len >= 0 ) { reverse = reverse + ans [ Len ] ; Len -- ; }
Console . Write ( reverse ) ; } }
public static void Main ( ) { string S = " toffee " ; int K = 4 ; findString ( S , K ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
public static void printFirstRepeating ( int [ ] arr ) {
int min = - 1 ;
HashSet < int > set = new HashSet < int > ( ) ;
for ( int i = arr . Length - 1 ; i >= 0 ; i -- ) {
if ( set . Contains ( arr [ i ] ) ) { min = i ; }
else { set . Add ( arr [ i ] ) ; } }
if ( min != - 1 ) { Console . WriteLine ( " The ▁ first ▁ repeating ▁ element ▁ is ▁ " + arr [ min ] ) ; } else { Console . WriteLine ( " There ▁ are ▁ no ▁ repeating ▁ elements " ) ; } }
public static void Main ( string [ ] args ) { int [ ] arr = new int [ ] { 10 , 5 , 3 , 4 , 3 , 5 , 6 } ; printFirstRepeating ( arr ) ; } }
using System ; class GFG {
static void printFirstRepeating ( int [ ] arr , int n ) {
int k = 0 ;
int max = n ; for ( int i = 0 ; i < n ; i ++ ) if ( max < arr [ i ] ) max = arr [ i ] ;
int [ ] a = new int [ max + 1 ] ;
int [ ] b = new int [ max + 1 ] ; for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ arr [ i ] ] != 0 ) { b [ arr [ i ] ] = 1 ; k = 1 ; continue ; } else
a [ arr [ i ] ] = i ; } if ( k == 0 ) Console . WriteLine ( " No ▁ repeating ▁ element ▁ found " ) ; else { int min = max + 1 ;
for ( int i = 0 ; i < max + 1 ; i ++ ) if ( ( a [ i ] != 0 ) && min > a [ i ] && ( b [ i ] != 0 ) ) min = a [ i ] ; Console . Write ( arr [ min ] ) ; } Console . WriteLine ( ) ; }
static void Main ( ) { int [ ] arr = { 10 , 5 , 3 , 4 , 3 , 5 , 6 } ; int n = arr . Length ; printFirstRepeating ( arr , n ) ; } }
using System ; class GFG {
static int printKDistinct ( int [ ] arr , int n , int k ) { int dist_count = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return - 1 ; }
public static void Main ( ) { int [ ] ar = { 1 , 2 , 1 , 3 , 4 , 2 } ; int n = ar . Length ; int k = 2 ; Console . Write ( printKDistinct ( ar , n , k ) ) ; } }
using System ; class GFG {
static void countSubarrays ( int [ ] A ) {
int res = 0 ;
int curr = A [ 0 ] ; int [ ] cnt = new int [ A . Length ] ; cnt [ 0 ] = 1 ; for ( int c = 1 ; c < A . Length ; c ++ ) {
if ( A == curr )
cnt ++ ; else
curr = A ; cnt = 1 ; }
for ( int i = 1 ; i < cnt . Length ; i ++ ) {
res += Math . Min ( cnt [ i - 1 ] , cnt [ i ] ) ; } Console . WriteLine ( res - 1 ) ; }
public static void Main ( String [ ] args ) {
int [ ] A = { 1 , 1 , 0 , 0 , 1 , 0 } ;
countSubarrays ( A ) ; } }
using System ; using System . Collections . Generic ; class GfG {
class Node { public int val ; public Node left , right ; }
static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . val = data ; temp . left = null ; temp . right = null ; return temp ; }
static bool isEvenOddBinaryTree ( Node root ) { if ( root == null ) return true ;
Queue < Node > q = new Queue < Node > ( ) ; q . Enqueue ( root ) ;
int level = 0 ;
while ( q . Count != 0 ) {
int size = q . Count ; for ( int i = 0 ; i < size ; i ++ ) { Node node = q . Dequeue ( ) ;
if ( level % 2 == 0 ) { if ( node . val % 2 == 1 ) return false ; } else if ( level % 2 == 1 ) { if ( node . val % 2 == 0 ) return false ; }
if ( node . left != null ) { q . Enqueue ( node . left ) ; } if ( node . right != null ) { q . Enqueue ( node . right ) ; } }
level ++ ; } return true ; }
public static void Main ( String [ ] args ) {
Node root = null ; root = newNode ( 2 ) ; root . left = newNode ( 3 ) ; root . right = newNode ( 9 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 10 ) ; root . right . right = newNode ( 6 ) ;
if ( isEvenOddBinaryTree ( root ) ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
using System ; class GFG { static int findMaxLen ( int [ ] a , int n ) {
int [ ] freq = new int [ n + 1 ] ; for ( int i = 0 ; i < n ; ++ i ) { freq [ a [ i ] ] ++ ; } int maxFreqElement = int . MinValue ; int maxFreqCount = 1 ; for ( int i = 1 ; i <= n ; ++ i ) {
if ( freq [ i ] > maxFreqElement ) { maxFreqElement = freq [ i ] ; maxFreqCount = 1 ; }
else if ( freq [ i ] == maxFreqElement ) ++ ; } int ;
if ( maxFreqElement == 1 ) ans = 0 ; else {
ans = ( ( n - maxFreqCount ) / ( maxFreqElement - 1 ) ) ; }
return ans ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 2 , 1 , 2 } ; int n = a . Length ; Console . Write ( findMaxLen ( a , n ) ) ; } }
using System ; class GFG {
static int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
static int MaxUtil ( int [ ] st , int ss , int se , int l , int r , int node ) {
if ( l <= ss && r >= se )
return st [ node ] ;
if ( se < l ss > r ) return - 1 ;
int mid = getMid ( ss , se ) ; return Math . Max ( MaxUtil ( st , ss , mid , l , r , 2 * node + 1 ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 2 ) ) ; }
static int getMax ( int [ ] st , int n , int l , int r ) {
if ( l < 0 r > n - 1 l > r ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
static int constructSTUtil ( int [ ] arr , int ss , int se , int [ ] st , int si ) {
if ( ss == se ) { st [ si ] = arr [ ss ] ; return arr [ ss ] ; }
int mid = getMid ( ss , se ) ;
st [ si ] = Math . Max ( constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) ,
constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ) ; return st [ si ] ; }
static int [ ] constructST ( int [ ] arr , int n ) {
int x = ( int ) ( Math . Ceiling ( Math . Log ( n ) ) ) ;
int max_size = 2 * ( int ) Math . Pow ( 2 , x ) - 1 ;
int [ ] st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 2 , 3 , 0 } ; int n = arr . Length ;
int [ ] st = constructST ( arr , n ) ; int [ , ] Q = { { 1 , 3 } , { 0 , 2 } } ; for ( int i = 0 ; i < Q . GetLength ( 0 ) ; i ++ ) { int max = getMax ( st , n , Q [ i , 0 ] , Q [ i , 1 ] ) ; int ok = 0 ; for ( int j = 30 ; j >= 0 ; j -- ) { if ( ( max & ( 1 << j ) ) != 0 ) ok = 1 ; if ( ok <= 0 ) continue ; max |= ( 1 << j ) ; } Console . Write ( max + " ▁ " ) ; } } }
using System ; class GFG {
static int calculate ( int [ ] a , int n ) {
Array . Sort ( a ) ; int count = 1 ; int answer = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( a [ i ] == a [ i - 1 ] ) {
count += 1 ; } else {
answer = answer + ( count * ( count - 1 ) ) / 2 ; count = 1 ; } } answer = answer + ( count * ( count - 1 ) ) / 2 ; return answer ; }
public static void Main ( ) { int [ ] a = { 1 , 2 , 1 , 2 , 4 } ; int n = a . Length ;
Console . WriteLine ( calculate ( a , n ) ) ; } }
using System ; using System . Linq ; class GFG {
static int calculate ( int [ ] a , int n ) {
int maximum = a . Max ( ) ;
int [ ] frequency = new int [ maximum + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
frequency [ a [ i ] ] += 1 ; } int answer = 0 ;
for ( int i = 0 ; i < ( maximum ) + 1 ; i ++ ) {
answer = answer + frequency [ i ] * ( frequency [ i ] - 1 ) ; } return answer / 2 ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 2 , 1 , 2 , 4 } ; int n = a . Length ;
Console . WriteLine ( calculate ( a , n ) ) ; } }
using System ; class GFG {
static int findSubArray ( int [ ] arr , int n ) { int sum = 0 ; int maxsize = - 1 , startindex = 0 ; int endindex = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? - 1 : 1 ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] == 0 ) sum += - 1 ; else sum += 1 ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } endindex = startindex + maxsize - 1 ; if ( maxsize == - 1 ) Console . WriteLine ( " No ▁ such ▁ subarray " ) ; else Console . WriteLine ( startindex + " ▁ to ▁ " + endindex ) ; return maxsize ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int size = arr . Length ; findSubArray ( arr , size ) ; } }
using System ; class GFG {
static int findMax ( int [ ] arr , int low , int high ) {
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid == 0 && arr [ mid ] > arr [ mid + 1 ] ) return arr [ mid ] ;
if ( arr [ low ] > arr [ mid ] ) { return findMax ( arr , low , mid - 1 ) ; } else { return findMax ( arr , mid + 1 , high ) ; } }
public static void Main ( ) { int [ ] arr = { 6 , 5 , 1 , 2 , 3 , 4 } ; int n = arr . Length ; Console . WriteLine ( findMax ( arr , 0 , n - 1 ) ) ; } }
using System ; public class GFG {
static int ternarySearch ( int l , int r , int key , int [ ] ar ) { while ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
r = mid1 - 1 ; } else if ( key > ar [ mid2 ] ) {
l = mid2 + 1 ; } else {
l = mid1 + 1 ; r = mid2 - 1 ; } }
return - 1 ; }
public static void Main ( String [ ] args ) { int l , r , p , key ;
int [ ] ar = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
Console . WriteLine ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
Console . WriteLine ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int majorityNumber ( int [ ] arr , int n ) { int ans = - 1 ; Dictionary < int , int > freq = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( freq . ContainsKey ( arr [ i ] ) ) { freq [ arr [ i ] ] = freq [ arr [ i ] ] + 1 ; } else { freq . Add ( arr [ i ] , 1 ) ; } if ( freq [ arr [ i ] ] > n / 2 ) ans = arr [ i ] ; } return ans ; }
public static void Main ( String [ ] args ) { int [ ] a = { 2 , 2 , 1 , 1 , 1 , 2 , 2 } ; int n = a . Length ; Console . WriteLine ( majorityNumber ( a , n ) ) ; } }
using System ; class GFG {
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
using System ; class GFG {
public static int findMin ( int [ ] arr , int low , int high ) { while ( low < high ) { int mid = low + ( high - low ) / 2 ; if ( arr [ mid ] == arr [ high ] ) high -- ; else if ( arr [ mid ] > arr [ high ] ) low = mid + 1 ; else high = mid ; } return arr [ high ] ; }
public static void Main ( String [ ] args ) { int [ ] arr1 = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = arr1 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr1 , 0 , n1 - 1 ) ) ; int [ ] arr2 = { 1 , 2 , 3 , 4 } ; int n2 = arr2 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr2 , 0 , n2 - 1 ) ) ; int [ ] arr3 = { 1 } ; int n3 = arr3 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr3 , 0 , n3 - 1 ) ) ; int [ ] arr4 = { 1 , 2 } ; int n4 = arr4 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr4 , 0 , n4 - 1 ) ) ; int [ ] arr5 = { 2 , 1 } ; int n5 = arr5 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr5 , 0 , n5 - 1 ) ) ; int [ ] arr6 = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = arr6 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr6 , 0 , n6 - 1 ) ) ; int [ ] arr7 = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = arr7 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr7 , 0 , n7 - 1 ) ) ; int [ ] arr8 = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = arr8 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr8 , 0 , n8 - 1 ) ) ; int [ ] arr9 = { 3 , 4 , 5 , 1 , 2 } ; int n9 = arr9 . Length ; Console . WriteLine ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr9 , 0 , n9 - 1 ) ) ; } }
using System ; class GFG {
static int countPairs ( int [ ] a , int n , int mid ) { int res = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int ub = upperbound ( a , n , a [ i ] + mid ) ; res += ( ub - ( i - 1 ) ) ; } return res ; }
static int upperbound ( int [ ] a , int n , int value ) { int low = 0 ; int high = n ; while ( low < high ) { int mid = ( low + high ) / 2 ; if ( value >= a [ mid ] ) low = mid + 1 ; else high = mid ; } return low ; }
static int kthDiff ( int [ ] a , int n , int k ) {
Array . Sort ( a ) ;
int low = a [ 1 ] - a [ 0 ] ; for ( int i = 1 ; i <= n - 2 ; ++ i ) low = Math . Min ( low , a [ i + 1 ] - a [ i ] ) ;
int high = a [ n - 1 ] - a [ 0 ] ;
while ( low < high ) { int mid = ( low + high ) >> 1 ; if ( countPairs ( a , n , mid ) < k ) low = mid + 1 ; else high = mid ; } return low ; }
public static void Main ( String [ ] args ) { int k = 3 ; int [ ] a = { 1 , 2 , 3 , 4 } ; int n = a . Length ; Console . WriteLine ( kthDiff ( a , n , k ) ) ; } }
using System ; class GFG {
static void print2Smallest ( int [ ] arr ) { int first , second , arr_size = arr . Length ;
if ( arr_size < 2 ) { Console . Write ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = int . MaxValue ; for ( int i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) = arr [ i ] ; } if ( second == int . MaxValue ) Console . Write ( " There ▁ is ▁ no ▁ second " + " smallest ▁ element " ) ; else . Write ( " The ▁ smallest ▁ element ▁ is ▁ " + first + " ▁ and ▁ second ▁ Smallest " + " ▁ element ▁ is ▁ " + second ) ; }
public static void Main ( ) { int [ ] arr = { 12 , 13 , 1 , 10 , 34 , 1 } ; print2Smallest ( arr ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int MAX = 1000 ;
static int [ ] tree = new int [ 4 * MAX ] ;
static int [ ] arr = new int [ MAX ] ;
static int gcd ( int a , int b ) { if ( a == 0 ) { return b ; } return gcd ( b % a , a ) ; }
static int lcm ( int a , int b ) { return a * b / gcd ( a , b ) ; }
static void build ( int node , int start , int end ) {
if ( start == end ) { tree [ node ] = arr [ start ] ; return ; } int mid = ( start + end ) / 2 ;
build ( 2 * node , start , mid ) ; build ( 2 * node + 1 , mid + 1 , end ) ;
int left_lcm = tree [ 2 * node ] ; int right_lcm = tree [ 2 * node + 1 ] ; tree [ node ] = lcm ( left_lcm , right_lcm ) ; }
static int query ( int node , int start , int end , int l , int r ) {
if ( end < l start > r ) { return 1 ; }
if ( l <= start && r >= end ) { return tree [ node ] ; }
int mid = ( start + end ) / 2 ; int left_lcm = query ( 2 * node , start , mid , l , r ) ; int right_lcm = query ( 2 * node + 1 , mid + 1 , end , l , r ) ; return lcm ( left_lcm , right_lcm ) ; }
public static void Main ( String [ ] args ) {
arr [ 0 ] = 5 ; arr [ 1 ] = 7 ; arr [ 2 ] = 5 ; arr [ 3 ] = 2 ; arr [ 4 ] = 10 ; arr [ 5 ] = 12 ; arr [ 6 ] = 11 ; arr [ 7 ] = 17 ; arr [ 8 ] = 14 ; arr [ 9 ] = 1 ; arr [ 10 ] = 44 ;
build ( 1 , 0 , 10 ) ;
Console . WriteLine ( query ( 1 , 0 , 10 , 2 , 5 ) ) ;
Console . WriteLine ( query ( 1 , 0 , 10 , 5 , 10 ) ) ;
Console . WriteLine ( query ( 1 , 0 , 10 , 0 , 10 ) ) ; } }
using System ; class GFG { static int M = 1000000007 ; static int waysOfDecoding ( String s ) { long [ ] dp = new long [ s . Length + 1 ] ; dp [ 0 ] = 1 ;
dp [ 1 ] = s [ 0 ] == ' * ' ? 9 : s [ 0 ] == '0' ? 0 : 1 ;
for ( int i = 1 ; i < s . Length ; i ++ ) {
if ( s [ i ] == ' * ' ) { dp [ i + 1 ] = 9 * dp [ i ] ;
if ( s [ i - 1 ] == '1' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 9 * dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == '2' ) [ i + 1 ] = ( dp [ i + 1 ] + 6 * dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' * ' ) [ i + 1 ] = ( dp [ i + 1 ] + 15 * dp [ i - 1 ] ) % M ; } else {
dp [ i + 1 ] = s [ i ] != '0' ? dp [ i ] : 0 ;
if ( s [ i - 1 ] == '1' ) dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == '2' && s [ i ] <= '6' ) [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' * ' ) [ i + 1 ] = ( dp [ i + 1 ] + ( s [ i ] <= '6' ? 2 : 1 ) * dp [ i - 1 ] ) % M ; } } return ( int ) dp [ s . Length ] ; }
public static void Main ( ) { String s = "12" ; Console . WriteLine ( waysOfDecoding ( s ) ) ; } }
using System ; public class GFG {
static int countSubset ( int [ ] arr , int n , int diff ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; sum += diff ; sum = sum / 2 ;
int [ , ] t = new int [ n + 1 , sum + 1 ] ;
for ( int j = 0 ; j <= sum ; j ++ ) t [ 0 , j ] = 0 ;
for ( int i = 0 ; i <= n ; i ++ ) t [ i , 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) {
if ( arr [ i - 1 ] > j ) t [ i , j ] = t [ i - 1 , j ] ; else { t [ i , j ] = t [ i - 1 , j ] + t [ i - 1 , j - arr [ i - 1 ] ] ; } } }
return t [ n , sum ] ; }
public static void Main ( string [ ] args ) {
int diff = 1 , n = 4 ; int [ ] arr = { 1 , 1 , 2 , 3 } ;
Console . Write ( countSubset ( arr , n , diff ) ) ; } }
using System ; public class GFG { static float [ , ] dp = new float [ 105 , 605 ] ;
static float find ( int N , int a , int b ) { float probability = 0.0f ;
for ( int i = 1 ; i <= 6 ; i ++ ) dp [ 1 , i ] = ( float ) ( 1.0 / 6 ) ; for ( int i = 2 ; i <= N ; i ++ ) { for ( int j = i ; j <= 6 * i ; j ++ ) { for ( int k = 1 ; k <= 6 && k <= j ; k ++ ) { dp [ i , j ] = dp [ i , j ] + dp [ i - 1 , j - k ] / 6 ; } } }
for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + dp [ N , sum ] ; return probability ; }
public static void Main ( String [ ] args ) { int N = 4 , a = 13 , b = 17 ; float probability = find ( N , a , b ) ;
Console . Write ( " { 0 : F6 } " , probability ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node left , right ; public Node ( int item ) { data = item ; left = right = null ; } }
public static int getSumAlternate ( Node root ) { if ( root == null ) return 0 ; int sum = root . data ; if ( root . left != null ) { sum += getSum ( root . left . left ) ; sum += getSum ( root . left . right ) ; } if ( root . right != null ) { sum += getSum ( root . right . left ) ; sum += getSum ( root . right . right ) ; } return sum ; }
public static int getSum ( Node root ) { if ( root == null ) return 0 ;
return Math . Max ( getSumAlternate ( root ) , ( getSumAlternate ( root . left ) + getSumAlternate ( root . right ) ) ) ; }
public static void Main ( ) { Node root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . right . left = new Node ( 4 ) ; root . right . left . right = new Node ( 5 ) ; root . right . left . right . left = new Node ( 6 ) ; Console . WriteLine ( getSum ( root ) ) ; } }
using System ; public class Subset_sum {
static bool isSubsetSum ( int [ ] arr , int n , int sum ) {
bool [ , ] subset = new bool [ 2 , sum + 1 ] ; for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) {
if ( j == 0 ) subset [ i % 2 , j ] = true ;
else if ( i = = 0 ) subset [ i % 2 , j ] = false ; else if ( arr [ i - 1 ] <= j ) [ i % 2 , j ] = subset [ ( i + 1 ) % 2 , j - arr [ i - 1 ] ] || subset [ ( i + 1 ) % 2 , j ] ; else [ i % 2 , j ] = subset [ ( i + 1 ) % 2 , j ] ; } } return [ n % 2 , sum ] ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 5 } ; int sum = 7 ; int n = arr . Length ; if ( isSubsetSum ( arr , n , sum ) == true ) Console . WriteLine ( " There ▁ exists ▁ a ▁ subset ▁ with " + " given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ exists ▁ with " + " given ▁ sum " ) ; } }
using System ; class GFG {
static int findMaxSum ( int [ ] arr , int n ) { int res = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) { int prefix_sum = arr [ i ] ; for ( int j = 0 ; j < i ; j ++ ) prefix_sum += arr [ j ] ; int suffix_sum = arr [ i ] ; for ( int j = n - 1 ; j > i ; j -- ) suffix_sum += arr [ j ] ; if ( prefix_sum == suffix_sum ) res = Math . Max ( res , prefix_sum ) ; } return res ; }
public static void Main ( ) { int [ ] arr = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( findMaxSum ( arr , n ) ) ; } }
using System ; public class GFG {
static int findMaxSum ( int [ ] arr , int n ) {
int [ ] preSum = new int [ n ] ;
int [ ] suffSum = new int [ n ] ;
int ans = int . MinValue ;
preSum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) preSum [ i ] = preSum [ i - 1 ] + arr [ i ] ;
suffSum [ n - 1 ] = arr [ n - 1 ] ; if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) ans = Math . Max ( ans , preSum [ n - 1 ] ) ; for ( int i = n - 2 ; i >= 0 ; i -- ) { suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] ; if ( suffSum [ i ] == preSum [ i ] ) ans = Math . Max ( ans , preSum [ i ] ) ; } return ans ; }
static public void Main ( ) { int [ ] arr = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( findMaxSum ( arr , n ) ) ; } }
using System . Linq ; using System ; class GFG { static int Add ( int x , int y ) { return x + y ; }
static int findMaxSum ( int [ ] arr , int n ) { int sum = arr . Aggregate ( func : Add ) ; int prefix_sum = 0 , res = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) { prefix_sum += arr [ i ] ; if ( prefix_sum == sum ) res = Math . Max ( res , prefix_sum ) ; sum -= arr [ i ] ; } return res ; }
public static void Main ( ) { int [ ] arr = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . Length ; Console . Write ( findMaxSum ( arr , n ) ) ; } }
using System ; public class GFG {
static void findMajority ( int [ ] arr , int n ) { int maxCount = 0 ;
int index = - 1 ; for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) count ++ ; }
if ( count > maxCount ) { maxCount = count ; index = i ; } }
if ( maxCount > n / 2 ) Console . WriteLine ( arr [ index ] ) ; else Console . WriteLine ( " No ▁ Majority ▁ Element " ) ; }
static public void Main ( ) { int [ ] arr = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = arr . Length ;
findMajority ( arr , n ) ; } }
using System ; public class Node { public int key ; public int c = 0 ; public Node left , right ; } class GFG { static int ma = 0 ;
static Node newNode ( int item ) { Node temp = new Node ( ) ; temp . key = item ; temp . c = 1 ; temp . left = temp . right = null ; return temp ; }
static Node insert ( Node node , int key ) {
if ( node == null ) { if ( ma == 0 ) ma = 1 ; return newNode ( key ) ; }
if ( key < node . key ) node . left = insert ( node . left , key ) ; else if ( key > node . key ) node . right = insert ( node . right , key ) ; else node . c ++ ;
ma = Math . Max ( ma , node . c ) ;
return node ; }
static void inorder ( Node root , int s ) { if ( root != null ) { inorder ( root . left , s ) ; if ( root . c > ( s / 2 ) ) Console . WriteLine ( root . key + " STRNEWLINE " ) ; inorder ( root . right , s ) ; } }
static public void Main ( ) { int [ ] a = { 1 , 3 , 3 , 3 , 2 } ; int size = a . Length ; Node root = null ; for ( int i = 0 ; i < size ; i ++ ) { root = insert ( root , a [ i ] ) ; }
if ( ma > ( size / 2 ) ) inorder ( root , size ) ; else Console . WriteLine ( " No ▁ majority ▁ element STRNEWLINE " ) ; } }
using System ; class GFG {
static int findCandidate ( int [ ] a , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
static bool isMajority ( int [ ] a , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) { if ( a [ i ] == cand ) count ++ ; } if ( count > size / 2 ) return true ; else return false ; }
static void printMajority ( int [ ] a , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) Console . Write ( " ▁ " + cand + " ▁ " ) ; else Console . Write ( " No ▁ Majority ▁ Element " ) ; }
public static void Main ( ) { int [ ] a = { 1 , 3 , 3 , 1 , 2 } ; int size = a . Length ;
printMajority ( a , size ) ; } }
using System ; using System . Collections . Generic ; class GFG { private static void findMajority ( int [ ] arr ) { Dictionary < int , int > map = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < arr . Length ; i ++ ) { if ( map . ContainsKey ( arr [ i ] ) ) { int count = map [ arr [ i ] ] + 1 ; if ( count > arr . Length / 2 ) { Console . WriteLine ( " Majority ▁ found ▁ : - ▁ " + arr [ i ] ) ; return ; } else { map [ arr [ i ] ] = count ; } } else { map [ arr [ i ] ] = 1 ; } } Console . WriteLine ( " ▁ No ▁ Majority ▁ element " ) ; }
public static void Main ( string [ ] args ) { int [ ] a = new int [ ] { 2 , 2 , 2 , 2 , 5 , 5 , 2 , 3 , 3 } ;
findMajority ( a ) ; } }
using System ; class GFG {
public static int majorityElement ( int [ ] arr , int n ) {
Array . Sort ( arr ) ; int count = 1 , max_ele = - 1 , temp = arr [ 0 ] , ele = 0 , f = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( temp == arr [ i ] ) { count ++ ; } else { count = 1 ; temp = arr [ i ] ; }
if ( max_ele < count ) { max_ele = count ; ele = arr [ i ] ; if ( max_ele > ( n / 2 ) ) { f = 1 ; break ; } } }
return ( f == 1 ? ele : - 1 ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = 7 ;
Console . WriteLine ( majorityElement ( arr , n ) ) ; } }
using System ; class GFG {
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
bool [ , ] subset = new bool [ sum + 1 , n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 , i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i , 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i , j ] = subset [ i , j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i , j ] = subset [ i , j ] || subset [ i - set [ j - 1 ] , j - 1 ] ; } } return subset [ sum , n ] ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
using System ; class GFG {
static int subsetSum ( int [ ] a , int n , int sum ) {
int [ , ] tab = new int [ n + 1 , sum + 1 ] ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) { tab [ i , j ] = - 1 ; } }
if ( sum == 0 ) return 1 ; if ( n <= 0 ) return 0 ;
if ( tab [ n - 1 , sum ] != - 1 ) return tab [ n - 1 , sum ] ;
if ( a [ n - 1 ] > sum ) return tab [ n - 1 , sum ] = subsetSum ( a , n - 1 , sum ) ; else {
if ( subsetSum ( a , n - 1 , sum ) != 0 || subsetSum ( a , n - 1 , sum - a [ n - 1 ] ) != 0 ) { return tab [ n - 1 , sum ] = 1 ; } else return tab [ n - 1 , sum ] = 0 ; } }
public static void Main ( String [ ] args ) { int n = 5 ; int [ ] a = { 1 , 5 , 3 , 7 , 4 } ; int sum = 12 ; if ( subsetSum ( a , n , sum ) != 0 ) { Console . Write ( " YES STRNEWLINE " ) ; } else Console . Write ( " NO STRNEWLINE " ) ; } }
using System ; class GFG {
static int binpow ( int a , int b ) { int res = 1 ; while ( b > 0 ) { if ( b % 2 == 1 ) res = res * a ; a = a * a ; b /= 2 ; } return res ; }
static int find ( int x ) { if ( x == 0 ) return 0 ; int p = ( int ) ( Math . Log ( x ) / Math . Log ( 2 ) ) ; return binpow ( 2 , p + 1 ) - 1 ; }
static String getBinary ( int n ) {
String ans = " " ;
while ( n > 0 ) { int dig = n % 2 ; ans += dig ; n /= 2 ; }
return ans ; }
static int totalCountDifference ( int n ) {
string ans = getBinary ( n ) ;
int req = 0 ;
for ( int i = 0 ; i < ans . Length ; i ++ ) {
if ( ans [ i ] == '1' ) { req += find ( binpow ( 2 , i ) ) ; } } return req ; }
public static void Main ( ) {
int n = 5 ;
Console . Write ( totalCountDifference ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int Maximum_Length ( List < int > a ) {
int [ ] counts = new int [ 11 ] ;
int ans = 0 ; for ( int index = 0 ; index < a . Count ; index ++ ) {
counts [ a [ index ] ] += 1 ;
List < int > k = new List < int > ( ) ; foreach ( int i in counts ) if ( i != 0 ) k . Add ( i ) ; k . Sort ( ) ;
if ( k . Count == 1 || ( k [ 0 ] == k [ k . Count - 2 ] && k [ k . Count - 1 ] - k [ k . Count - 2 ] == 1 ) || ( k [ 0 ] == 1 && k [ 1 ] == k [ k . Count - 1 ] ) ) ans = index ; }
return ans + 1 ; }
static void Main ( ) { List < int > a = new List < int > ( new int [ ] { 1 , 1 , 1 , 2 , 2 , 2 } ) ; Console . Write ( Maximum_Length ( a ) ) ; } }
using System ; class GFG {
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
public static void Main ( ) {
SieveOfEratosthenes ( ) ;
int l = 3 , r = 9 ;
int c = ( sum [ r ] - sum [ l - 1 ] ) ;
Console . WriteLine ( " Count : ▁ " + c ) ; } }
using System ; class GFG {
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
using System ; class GFG {
static int sumOfDigitsSingle ( int x ) { int ans = 0 ; while ( x != 0 ) { ans += x % 10 ; x /= 10 ; } return ans ; }
static int closest ( int x ) { int ans = 0 ; while ( ans * 10 + 9 <= x ) ans = ans * 10 + 9 ; return ans ; } static int sumOfDigitsTwoParts ( int N ) { int A = closest ( N ) ; return sumOfDigitsSingle ( A ) + sumOfDigitsSingle ( N - A ) ; }
public static void Main ( ) { int N = 35 ; Console . Write ( sumOfDigitsTwoParts ( N ) ) ; } }
using System ; class GFG {
static bool isPrime ( int p ) {
double checkNumber = Math . Pow ( 2 , p ) - 1 ;
double nextval = 4 % checkNumber ;
for ( int i = 1 ; i < p - 1 ; i ++ ) nextval = ( nextval * nextval - 2 ) % checkNumber ;
return ( nextval == 0 ) ; }
static void Main ( ) {
int p = 7 ; double checkNumber = Math . Pow ( 2 , p ) - 1 ; if ( isPrime ( p ) ) Console . WriteLine ( ( int ) checkNumber + " ▁ is ▁ Prime . " ) ; else Console . WriteLine ( ( int ) checkNumber + " ▁ is ▁ not ▁ Prime . " ) ; } }
using System ; class GFG {
static void sieve ( int n , bool [ ] prime ) { for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < n ; i += p ) prime [ i ] = false ; } } } static void printSophieGermanNumber ( int n ) {
bool [ ] prime = new bool [ 2 * n + 1 ] ; for ( int i = 0 ; i < prime . Length ; i ++ ) { prime [ i ] = true ; } sieve ( 2 * n + 1 , prime ) ; for ( int i = 2 ; i < n ; ++ i ) {
if ( prime [ i ] && prime [ 2 * i + 1 ] ) Console . Write ( i + " ▁ " ) ; } }
static void Main ( ) { int n = 25 ; printSophieGermanNumber ( n ) ; } }
class GFG {
static double ucal ( double u , int n ) { if ( n == 0 ) return 1 ; double temp = u ; for ( int i = 1 ; i <= n / 2 ; i ++ ) temp = temp * ( u - i ) ; for ( int i = 1 ; i < n / 2 ; i ++ ) temp = temp * ( u + i ) ; return temp ; }
static int fact ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
public static void Main ( ) {
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
using System ; class GFG {
static bool powerOf2 ( int n ) {
if ( n == 1 ) return true ;
else if ( n % 2 != 0 n == 0 ) return false ;
return powerOf2 ( n / 2 ) ; }
static void Main ( ) {
int n = 64 ;
int m = 12 ; if ( powerOf2 ( n ) ) { Console . Write ( " True " + " STRNEWLINE " ) ; } else { Console . Write ( " False " + " STRNEWLINE " ) ; } if ( powerOf2 ( m ) ) { Console . Write ( " True " ) ; } else { Console . Write ( " False " ) ; } } }
using System ; class GFG {
static bool isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void Main ( ) { Console . WriteLine ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; Console . WriteLine ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
using System ; public class GFG {
static bool isPowerofTwo ( int n ) { if ( n == 0 ) return false ; if ( ( n & ( ~ ( n - 1 ) ) ) == n ) return true ; return false ; }
public static void Main ( String [ ] args ) { if ( isPowerofTwo ( 30 ) == true ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; if ( isPowerofTwo ( 128 ) == true ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
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
using System ; class GFG {
static int gcd ( int A , int B ) { if ( B == 0 ) return A ; return gcd ( B , A % B ) ; }
static int lcm ( int A , int B ) { return ( A * B ) / gcd ( A , B ) ; }
static int checkA ( int A , int B , int C , int K ) {
int start = 1 ; int end = K ;
int ans = - 1 ; while ( start <= end ) { int mid = ( start + end ) / 2 ; int value = A * mid ; int divA = mid - 1 ; int divB = ( value % B == 0 ) ? value / B - 1 : value / B ; int divC = ( value % C == 0 ) ? value / C - 1 : value / C ; int divAB = ( value % lcm ( A , B ) == 0 ) ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ; int divBC = ( value % lcm ( C , B ) == 0 ) ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ; int divAC = ( value % lcm ( A , C ) == 0 ) ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ; int divABC = ( value % lcm ( A , lcm ( B , C ) ) == 0 ) ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ;
int elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem == ( K - 1 ) ) { ans = value ; break ; }
else if ( elem > ( K - 1 ) ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
static int checkB ( int A , int B , int C , int K ) {
int start = 1 ; int end = K ;
int ans = - 1 ; while ( start <= end ) { int mid = ( start + end ) / 2 ; int value = B * mid ; int divB = mid - 1 ; int divA = ( value % A == 0 ) ? value / A - 1 : value / A ; int divC = ( value % C == 0 ) ? value / C - 1 : value / C ; int divAB = ( value % lcm ( A , B ) == 0 ) ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ; int divBC = ( value % lcm ( C , B ) == 0 ) ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ; int divAC = ( value % lcm ( A , C ) == 0 ) ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ; int divABC = ( value % lcm ( A , lcm ( B , C ) ) == 0 ) ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ;
int elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem == ( K - 1 ) ) { ans = value ; break ; }
else if ( elem > ( K - 1 ) ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
static int checkC ( int A , int B , int C , int K ) {
int start = 1 ; int end = K ;
int ans = - 1 ; while ( start <= end ) { int mid = ( start + end ) / 2 ; int value = C * mid ; int divC = mid - 1 ; int divB = ( value % B == 0 ) ? value / B - 1 : value / B ; int divA = ( value % A == 0 ) ? value / A - 1 : value / A ; int divAB = ( value % lcm ( A , B ) == 0 ) ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ; int divBC = ( value % lcm ( C , B ) == 0 ) ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ; int divAC = ( value % lcm ( A , C ) == 0 ) ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ; int divABC = ( value % lcm ( A , lcm ( B , C ) ) == 0 ) ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ;
int elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem == ( K - 1 ) ) { ans = value ; break ; }
else if ( elem > ( K - 1 ) ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
static int findKthMultiple ( int A , int B , int C , int K ) {
int res = checkA ( A , B , C , K ) ;
if ( res == - 1 ) res = checkB ( A , B , C , K ) ;
if ( res == - 1 ) res = checkC ( A , B , C , K ) ; return res ; }
public static void Main ( String [ ] args ) { int A = 2 , B = 4 , C = 5 , K = 5 ; Console . WriteLine ( findKthMultiple ( A , B , C , K ) ) ; } }
using System ; using System . Collections . Generic ; using System . Linq ; class GFG {
static void variationStalinsort ( List < int > arr ) { int j = 0 ; while ( true ) { int moved = 0 ; for ( int i = 0 ; i < ( arr . Count - 1 - j ) ; i ++ ) { if ( arr [ i ] > arr [ i + 1 ] ) {
int index ; int temp ; index = arr [ i ] ; temp = arr [ i + 1 ] ; arr . Remove ( index ) ; arr . Insert ( i , temp ) ; arr . Remove ( temp ) ; arr . Insert ( i + 1 , index ) ; moved ++ ; } } j ++ ; if ( moved == 0 ) { break ; } } foreach ( int i in arr ) Console . Write ( i + " ▁ " ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 2 , 1 , 4 , 3 , 6 , 5 , 8 , 7 , 10 , 9 } ; List < int > arr1 = new List < int > ( ) ; for ( int i = 0 ; i < arr . Length ; i ++ ) arr1 . Add ( arr [ i ] ) ;
variationStalinsort ( arr1 ) ; } }
using System ; class GFG {
public static void printArray ( int [ ] arr , int N ) {
for ( int i = 0 ; i < N ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void sortArray ( int [ ] arr , int N ) {
for ( int i = 0 ; i < N ; ) {
if ( arr [ i ] == i + 1 ) { i ++ ; }
else {
int temp1 = arr [ i ] ; int temp2 = arr [ arr [ i ] - 1 ] ; arr [ i ] = temp2 ; arr [ temp1 - 1 ] = temp1 ; } } }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 1 , 5 , 3 , 4 } ; int N = arr . Length ;
sortArray ( arr , N ) ;
printArray ( arr , N ) ; } }
using System ; public class GFG {
static int maximum ( int [ ] value , int [ ] weight , int weight1 , int flag , int K , int index ) {
if ( index >= value . Length ) { return 0 ; }
if ( flag == K ) {
int skip = maximum ( value , weight , weight1 , flag , K , index + 1 ) ; int full = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 ) ; }
return Math . Max ( full , skip ) ; }
else {
int skip = maximum ( value , weight , weight1 , flag , K , index + 1 ) ; int full = 0 ; int half = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 ) ; }
if ( weight [ index ] / 2 <= weight1 ) { half = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] / 2 , flag , K , index + 1 ) ; }
return Math . Max ( full , Math . Max ( skip , half ) ) ; } }
public static void Main ( String [ ] args ) { int [ ] value = { 17 , 20 , 10 , 15 } ; int [ ] weight = { 4 , 2 , 7 , 5 } ; int K = 1 ; int W = 4 ; Console . WriteLine ( maximum ( value , weight , W , 0 , K , 0 ) ) ; } }
using System ; class GFG { static readonly int N = 1005 ;
public class Node { public int data ; public Node left , right ; } ;
public static Node newNode ( int data ) { Node node = new Node ( ) ; node . data = data ; node . left = node . right = null ; return node ; }
static int [ , , ] dp = new int [ N , 5 , 5 ] ;
static int minDominatingSet ( Node root , int covered , int compulsory ) {
if ( root == null ) return 0 ;
if ( root . left != null && root . right != null && covered > 0 ) compulsory = 1 ;
if ( dp [ root . data , covered , compulsory ] != - 1 ) return dp [ root . data , covered , compulsory ] ;
if ( compulsory > 0 ) {
return dp [ root . data , covered , compulsory ] = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; }
if ( covered > 0 ) { return dp [ root . data , covered , compulsory ] = Math . Min ( 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; }
int ans = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; if ( root . left != null ) { ans = Math . Min ( ans , minDominatingSet ( root . left , 0 , 1 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; } if ( root . right != null ) { ans = Math . Min ( ans , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 1 ) ) ; }
return dp [ root . data , covered , compulsory ] = ans ; }
public static void Main ( String [ ] args ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < 5 ; j ++ ) { for ( int l = 0 ; l < 5 ; l ++ ) dp [ i , j , l ] = - 1 ; } }
Node root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . left . left = newNode ( 3 ) ; root . left . right = newNode ( 4 ) ; root . left . left . left = newNode ( 5 ) ; root . left . left . left . left = newNode ( 6 ) ; root . left . left . left . right = newNode ( 7 ) ; root . left . left . left . right . right = newNode ( 10 ) ; root . left . left . left . left . left = newNode ( 8 ) ; root . left . left . left . left . right = newNode ( 9 ) ; Console . Write ( minDominatingSet root , 0 , 0 ) + " STRNEWLINE " ) ; } }
using System ; class GFG { static int maxSum = 100 ; static int arrSize = 51 ;
static int [ , ] dp = new int [ arrSize , maxSum ] ; static bool [ , ] visit = new bool [ arrSize , maxSum ] ;
static int SubsetCnt ( int i , int s , int [ ] arr , int n ) {
if ( i == n ) { if ( s == 0 ) { return 1 ; } else { return 0 ; } }
if ( visit [ i , s + arrSize ] ) { return dp [ i , s + arrSize ] ; }
visit [ i , s + arrSize ] = true ;
dp [ i , s + arrSize ] = SubsetCnt ( i + 1 , s + arr [ i ] , arr , n ) + SubsetCnt ( i + 1 , s , arr , n ) ;
return dp [ i , s + arrSize ] ; }
public static void Main ( ) { int [ ] arr = { 2 , 2 , 2 , - 4 , - 4 } ; int n = arr . Length ; Console . WriteLine ( SubsetCnt ( 0 , 0 , arr , n ) ) ; } }
using System ; class GFG { static readonly int MAX = 1000 ;
static int waysToKAdjacentSetBits ( int [ , , ] dp , int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } if ( dp [ currentIndex , adjacentSetBits , lastBit ] != - 1 ) { return dp [ currentIndex , adjacentSetBits , lastBit ] ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( lastBit = = 0 ) { noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } dp [ currentIndex , adjacentSetBits , lastBit ] = noOfWays ; return noOfWays ; }
public static void Main ( String [ ] args ) { int n = 5 , k = 2 ;
int [ , , ] dp = new int [ MAX , MAX , 2 ] ;
for ( int i = 0 ; i < MAX ; i ++ ) for ( int j = 0 ; j < MAX ; j ++ ) for ( int k1 = 0 ; k1 < 2 ; k1 ++ ) dp [ i , j , k1 ] = - 1 ;
int totalWays = waysToKAdjacentSetBits ( dp , n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( dp , n , k , 1 , 0 , 0 ) ; Console . Write ( " Number ▁ of ▁ ways ▁ = ▁ " + totalWays + " STRNEWLINE " ) ; } }
using System ; class GFG {
static void printTetra ( int n ) { if ( n < 0 ) return ;
int first = 0 , second = 1 ; int third = 1 , fourth = 2 ;
int curr = 0 ; if ( n == 0 ) Console . Write ( first ) ; else if ( n == 1 n == 2 ) Console . Write ( second ) ; else if ( n == 3 ) Console . Write ( fourth ) ; else {
for ( int i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } Console . Write ( curr ) ; } }
static public void Main ( ) { int n = 10 ; printTetra ( n ) ; } }
using System ; public class GfG {
public static int countWays ( int n ) { int [ ] res = new int [ n + 2 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
public static void Main ( ) { int n = 4 ; Console . WriteLine ( countWays ( n ) ) ; } }
using System ; class GFG {
static int countWays ( int n ) {
int a = 1 , b = 2 , c = 4 ;
int d = 0 ; if ( n == 0 n == 1 n == 2 ) return n ; if ( n == 3 ) return c ;
for ( int i = 4 ; i <= n ; i ++ ) { d = c + b + a ; a = b ; b = c ; c = d ; } return d ; }
public static void Main ( String [ ] args ) { int n = 4 ; Console . Write ( countWays ( n ) ) ; } }
using System ; class GFG { static Boolean isPossible ( int [ ] elements , int sum ) { int [ ] dp = new int [ sum + 1 ] ;
dp [ 0 ] = 1 ;
for ( int i = 0 ; i < elements . Length ; i ++ ) {
for ( int j = sum ; j >= elements [ i ] ; j -- ) { if ( dp [ j - elements [ i ] ] == 1 ) dp [ j ] = 1 ; } }
if ( dp [ sum ] == 1 ) return true ; return false ; }
public static void Main ( String [ ] args ) { int [ ] elements = { 6 , 2 , 5 } ; int sum = 7 ; if ( isPossible ( elements , sum ) ) Console . Write ( " YES " ) ; else Console . Write ( " NO " ) ; } }
using System ; class GFG {
static int maxTasks ( int [ ] high , int [ ] low , int n ) {
if ( n <= 0 ) return 0 ;
return Math . Max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
public static void Main ( ) { int n = 5 ; int [ ] high = { 3 , 6 , 8 , 7 , 6 } ; int [ ] low = { 1 , 5 , 4 , 5 , 3 } ; Console . Write ( maxTasks ( high , low , n ) ) ; } }
using System ; class GFG {
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; } static int nCr ( int n , int r ) {
if ( r > n ) return 0 ;
if ( r > n - r ) r = n - r ; int mod = 1000000007 ;
int [ ] arr = new int [ r ] ; for ( int i = n - r + 1 ; i <= n ; i ++ ) { arr [ i + r - n - 1 ] = i ; } long ans = 1 ;
for ( int k = 1 ; k < r + 1 ; k ++ ) { int j = 0 , i = k ; while ( j < arr . Length ) { int x = gcd ( i , arr [ j ] ) ; if ( x > 1 ) {
arr [ j ] /= x ; i /= x ; } if ( i == 1 )
break ; j += 1 ; } }
foreach ( int i in arr ) ans = ( ans * i ) % mod ; return ( int ) ans ; }
static public void Main ( ) { int n = 5 , r = 2 ; Console . WriteLine ( " Value ▁ of ▁ C ( " + n + " , ▁ " + r + " ) ▁ is ▁ " + nCr ( n , r ) + " STRNEWLINE " ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static char FindKthChar ( string str , int K , int X ) {
char ans = ' ▁ ' ; int sum = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) {
int digit = ( int ) str [ i ] - 48 ;
int range = ( int ) Math . Pow ( digit , X ) ; sum += range ;
if ( K <= sum ) { ans = str [ i ] ; break ; } }
return ans ; }
public static void Main ( ) {
string str = "123" ; int K = 9 ; int X = 3 ;
char ans = FindKthChar ( str , K , X ) ; Console . Write ( ans ) ; } }
using System ; using System . Linq ; class GFG {
static int totalPairs ( string s1 , string s2 ) { int count = 0 ; int [ ] arr1 = new int [ 7 ] ; int [ ] arr2 = new int [ 7 ] ;
for ( int i = 0 ; i < s1 . Length ; i ++ ) { int set_bits = Convert . ToString ( ( int ) s1 [ i ] , 2 ) . Count ( c => c == '1' ) ; arr1 [ set_bits ] ++ ; }
for ( int i = 0 ; i < s2 . Length ; i ++ ) { int set_bits = Convert . ToString ( ( int ) s2 [ i ] , 2 ) . Count ( c => c == '1' ) ; arr2 [ set_bits ] ++ ; }
for ( int i = 1 ; i <= 6 ; i ++ ) count += ( arr1 [ i ] * arr2 [ i ] ) ;
return count ; }
static void Main ( ) { string s1 = " geeks " ; string s2 = " forgeeks " ; Console . WriteLine ( totalPairs ( s1 , s2 ) ) ; } }
using System ; class GFG {
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
using System ; class GFG {
static int nthEnneadecagonal ( int n ) {
return ( 17 * n * n - 15 * n ) / 2 ; }
static public void Main ( ) { int n = 6 ; Console . Write ( n + " th ▁ Enneadecagonal ▁ number ▁ : " ) ; Console . WriteLine ( nthEnneadecagonal ( n ) ) ; } }
using System ; class GFG { public static double PI = 3.14159265 ;
static float areacircumscribed ( float a ) { return ( a * a * ( float ) ( PI / 2 ) ) ; }
public static void Main ( ) { float a = 6 ; Console . Write ( " ▁ Area ▁ of ▁ an ▁ circumscribed " + " ▁ circle ▁ is ▁ : ▁ { 0 } " , Math . Round ( areacircumscribed ( a ) , 2 ) ) ; } }
using System ; class GFG {
static int itemType ( int n ) {
int count = 0 ; int day = 1 ;
while ( count + day * ( day + 1 ) / 2 < n ) {
count += day * ( day + 1 ) / 2 ; day ++ ; } for ( int type = day ; type > 0 ; type -- ) {
count += type ;
if ( count >= n ) { return type ; } } return 0 ; }
public static void Main ( String [ ] args ) { int N = 10 ; Console . Write ( itemType ( N ) ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node next ; } ;
static bool isSortedDesc ( Node head ) { if ( head == null ) return true ;
for ( Node t = head ; t . next != null ; t = t . next ) if ( t . data <= t . next . data ) return false ; return true ; } static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . next = null ; temp . data = data ; return temp ; }
public static void Main ( String [ ] args ) { Node head = newNode ( 7 ) ; head . next = newNode ( 5 ) ; head . next . next = newNode ( 4 ) ; head . next . next . next = newNode ( 3 ) ; if ( isSortedDesc ( head ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; public class GFG {
static int maxLength ( String str , int n , char c , int k ) {
int ans = - 1 ;
int cnt = 0 ;
int left = 0 ; for ( int right = 0 ; right < n ; right ++ ) { if ( str [ right ] == c ) { cnt ++ ; }
while ( cnt > k ) { if ( str [ left ] == c ) { cnt -- ; }
left ++ ; }
ans = Math . Max ( ans , right - left + 1 ) ; } return ans ; }
static int maxConsecutiveSegment ( String S , int K ) { int N = S . Length ;
return Math . Max ( maxLength ( S , N , '0' , K ) , maxLength ( S , N , '1' , K ) ) ; }
public static void Main ( ) { String S = "1001" ; int K = 1 ; Console . WriteLine ( maxConsecutiveSegment ( S , K ) ) ; } }
using System ; public class GFG {
static void find ( int N ) { int T , F , O ;
F = ( int ) ( ( N - 4 ) / 5 ) ;
if ( ( ( N - 5 * F ) % 2 ) == 0 ) { O = 2 ; } else { O = 1 ; }
T = ( int ) Math . Floor ( ( double ) ( N - 5 * F - O ) / 2 ) ; Console . WriteLine ( " Count ▁ of ▁ 5 ▁ valueds ▁ coins : ▁ " + F ) ; Console . WriteLine ( " Count ▁ of ▁ 2 ▁ valueds ▁ coins : ▁ " + T ) ; Console . WriteLine ( " Count ▁ of ▁ 1 ▁ valueds ▁ coins : ▁ " + O ) ; }
public static void Main ( String [ ] args ) { int N = 8 ; find ( N ) ; } }
using System ; class GFG {
static void findMaxOccurence ( char [ ] str , int N ) {
for ( int i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' ? ' ) {
str [ i ] = '0' ; } } Console . Write ( str ) ; }
public static void Main ( String [ ] args ) {
String str = "10?0?11" ; int N = str . Length ; findMaxOccurence ( str . ToCharArray ( ) , N ) ; } }
using System ; class GFG {
public static void checkInfinite ( String s ) {
bool flag = true ; int N = s . Length ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( s [ i ] == ( char ) ( ( int ) ( s [ i + 1 ] ) + 1 ) ) { continue ; }
else if ( s [ i ] == ' a ' && s [ i + 1 ] == ' z ' ) { continue ; }
else { flag = false ; break ; } }
if ( ! flag ) Console . Write ( " NO " ) ; else Console . Write ( " YES " ) ; }
public static void Main ( String [ ] args ) {
String s = " ecbaz " ;
checkInfinite ( s ) ; } }
using System ; public class GFG {
static int minChangeInLane ( int [ ] barrier , int n ) { int [ ] dp = { 1 , 0 , 1 } ; for ( int j = 0 ; j < n ; j ++ ) {
int val = barrier [ j ] ; if ( val > 0 ) { dp [ val - 1 ] = ( int ) 1e6 ; } for ( int i = 0 ; i < 3 ; i ++ ) {
if ( val != i + 1 ) { dp [ i ] = Math . Min ( dp [ i ] , Math . Min ( dp [ ( i + 1 ) % 3 ] , dp [ ( i + 2 ) % 3 ] ) + 1 ) ; } } }
return Math . Min ( dp [ 0 ] , Math . Min ( dp [ 1 ] , dp [ 2 ] ) ) ; }
static public void Main ( ) { int [ ] barrier = { 0 , 1 , 2 , 3 , 0 } ; int N = barrier . Length ; Console . Write ( minChangeInLane ( barrier , N ) ) ; } }
using System ; class GFG {
public static void numWays ( int [ , ] ratings , int [ , ] queries , int n , int k ) {
int [ , ] dp = new int [ n , 10000 + 2 ] ;
for ( int i = 0 ; i < k ; i ++ ) dp [ 0 , ratings [ 0 , i ] ] += 1 ;
for ( int i = 1 ; i < n ; i ++ ) {
for ( int sum = 0 ; sum <= 10000 ; sum ++ ) {
for ( int j = 0 ; j < k ; j ++ ) {
if ( sum >= ratings [ i , j ] ) dp [ i , sum ] += dp [ i - 1 , sum - ratings [ i , j ] ] ; } } }
for ( int sum = 1 ; sum <= 10000 ; sum ++ ) { dp [ n - 1 , sum ] += dp [ n - 1 , sum - 1 ] ; }
for ( int q = 0 ; q < queries . GetLength ( 0 ) ; q ++ ) { int a = queries [ q , 0 ] ; int b = queries [ q , 1 ] ;
Console . Write ( dp [ n - 1 , b ] - dp [ n - 1 , a - 1 ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) {
int N = 2 , K = 3 ;
int [ , ] ratings = { { 1 , 2 , 3 } , { 4 , 5 , 6 } } ;
int [ , ] queries = { { 6 , 6 } , { 1 , 6 } } ;
numWays ( ratings , queries , N , K ) ; } }
using System ; class GFG {
static void numberOfPermWithKInversion ( int N , int K ) {
int [ , ] dp = new int [ 2 , K + 1 ] ; int mod = 1000000007 ; for ( int i = 1 ; i <= N ; i ++ ) { for ( int j = 0 ; j <= K ; j ++ ) {
if ( i == 1 ) { dp [ i % 2 , j ] = ( j == 0 ) ? 1 : 0 ; }
else if ( j = = 0 ) dp [ i % 2 , j ] = 1 ;
else dp [ i % 2 , j ] = ( dp [ i % 2 , j - 1 ] % mod + ( dp [ 1 - i % 2 , j ] - ( ( Math . Max ( j - ( i - 1 ) , 0 ) == 0 ) ? 0 : dp [ 1 - i % 2 , Math . Max ( j - ( i - 1 ) , 0 ) - 1 ] ) + mod ) % mod ) % mod ; } }
Console . WriteLine ( dp [ N % 2 , K ] ) ; }
public static void Main ( ) {
int N = 3 , K = 2 ;
numberOfPermWithKInversion ( N , K ) ; } }
using System ; class GFG { static readonly int N = 100 ; static int n , m ;
static int [ , ] a = new int [ N , N ] ;
static int [ , ] dp = new int [ N , N ] ; static int [ , ] visited = new int [ N , N ] ;
static int current_sum = 0 ;
static int total_sum = 0 ;
static void inputMatrix ( ) { n = 3 ; m = 3 ; a [ 0 , 0 ] = 500 ; a [ 0 , 1 ] = 100 ; a [ 0 , 2 ] = 230 ; a [ 1 , 0 ] = 1000 ; a [ 1 , 1 ] = 300 ; a [ 1 , 2 ] = 100 ; a [ 2 , 0 ] = 200 ; a [ 2 , 1 ] = 1000 ; a [ 2 , 2 ] = 200 ; }
static int maximum_sum_path ( int i , int j ) {
if ( i == n - 1 && j == m - 1 ) return a [ i , j ] ;
if ( visited [ i , j ] != 0 ) return dp [ i , j ] ;
visited [ i , j ] = 1 ; int total_sum = 0 ;
if ( i < n - 1 & j < m - 1 ) { int current_sum = Math . Max ( maximum_sum_path ( i , j + 1 ) , Math . Max ( maximum_sum_path ( i + 1 , j + 1 ) , maximum_sum_path ( i + 1 , j ) ) ) ; total_sum = a [ i , j ] + current_sum ; }
else if ( i = = n - 1 ) total_sum = a [ i , j ] + maximum_sum_path ( i , j + 1 ) ;
else total_sum = a [ i , j ] + maximum_sum_path ( i + 1 , j ) ;
dp [ i , j ] = total_sum ;
return total_sum ; }
public static void Main ( String [ ] args ) { inputMatrix ( ) ;
int maximum_sum = maximum_sum_path ( 0 , 0 ) ; Console . WriteLine ( maximum_sum ) ; } }
using System ; class GFG { static int MaxProfit ( int [ ] treasure , int [ ] color , int n , int k , int col , int A , int B ) { int sum = 0 ;
if ( k == n ) return 0 ;
if ( col == color [ k ] ) sum += Math . Max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += Math . Max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return sum ; }
public static void Main ( String [ ] args ) { int A = - 5 , B = 7 ; int [ ] treasure = { 4 , 8 , 2 , 9 } ; int [ ] color = { 2 , 2 , 6 , 2 } ; int n = color . Length ;
Console . Write ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) ; } }
class GFG {
static int printTetraRec ( int n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
static void printTetra ( int n ) { System . Console . WriteLine ( printTetraRec ( n ) + " ▁ " ) ; }
static void Main ( ) { int n = 10 ; printTetra ( n ) ; } }
using System ; class GFG {
static int sum = 0 ; static void Combination ( int [ ] a , int [ ] combi , int n , int r , int depth , int index ) {
if ( index == r ) {
int product = 1 ; for ( int i = 0 ; i < r ; i ++ ) product = product * combi [ i ] ;
sum += product ; return ; }
for ( int i = depth ; i < n ; i ++ ) { combi [ index ] = a [ i ] ; Combination ( a , combi , n , r , i + 1 , index + 1 ) ; } }
static void allCombination ( int [ ] a , int n ) { for ( int i = 1 ; i <= n ; i ++ ) {
int [ ] combi = new int [ i ] ;
Combination ( a , combi , n , i , 0 , 0 ) ;
Console . Write ( " f ( " + i + " ) ▁ - - > ▁ " + sum + " STRNEWLINE " ) ; sum = 0 ; } }
static void Main ( ) { int n = 5 ; int [ ] a = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) a [ i ] = i + 1 ;
allCombination ( a , n ) ; } }
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
using System ; class GFG {
static bool findPartition ( int [ ] arr , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; bool [ , ] part = new bool [ sum / 2 + 1 , n + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) part [ 0 , i ] = true ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) part [ i , 0 ] = false ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) { for ( j = 1 ; j <= n ; j ++ ) { part [ i , j ] = part [ i , j - 1 ] ; if ( i >= arr [ j - 1 ] ) part [ i , j ] = part [ i , j - 1 ] || part [ i - arr [ j - 1 ] , j - 1 ] ; } }
return part [ sum / 2 , n ] ; }
public static void Main ( ) { int [ ] arr = { 3 , 1 , 1 , 2 , 2 , 1 } ; int n = arr . Length ;
if ( findPartition ( arr , n ) == true ) Console . Write ( " Can ▁ be ▁ divided " + " ▁ into ▁ two ▁ subsets ▁ of " + " ▁ equal ▁ sum " ) ; else Console . Write ( " Can ▁ not ▁ be ▁ " + " divided ▁ into ▁ two ▁ subsets " + " ▁ of ▁ equal ▁ sum " ) ; } }
using System ; class GFG {
static void minimumOperations ( string orig_str , int m , int n ) {
string orig = orig_str ;
int turn = 1 ; int j = 1 ;
for ( int i = 0 ; i < orig_str . Length ; i ++ ) {
string m_cut = orig_str . Substring ( orig_str . Length - m ) ; orig_str = orig_str . Substring ( 0 , orig_str . Length - m ) ;
orig_str = m_cut + orig_str ;
j = j + 1 ;
if ( ! orig . Equals ( orig_str ) ) { turn = turn + 1 ;
String n_cut = orig_str . Substring ( orig_str . Length - n ) ; orig_str = orig_str . Substring ( 0 , orig_str . Length - n ) ;
orig_str = n_cut + orig_str ;
j = j + 1 ; }
if ( orig . Equals ( orig_str ) ) { break ; }
turn = turn + 1 ; } Console . WriteLine ( turn ) ; }
public static void Main ( ) {
string S = " GeeksforGeeks " ; int X = 5 , Y = 3 ;
minimumOperations ( S , X , Y ) ; } }
using System ; class GFG {
static int KMPSearch ( char [ ] pat , char [ ] txt ) { int M = pat . Length ; int N = txt . Length ;
int [ ] lps = new int [ M ] ;
computeLPSArray ( pat , M , lps ) ;
int i = 0 ; int j = 0 ; while ( i < N ) { if ( pat [ j ] == txt [ i ] ) { j ++ ; i ++ ; } if ( j == M ) { return i - j ; }
else if ( i < N && [ j ] != txt [ i ] ) {
if ( j != 0 ) j = lps [ j - 1 ] ; else i = i + 1 ; } } return 0 ; }
static void computeLPSArray ( char [ ] pat , int M , int [ ] lps ) {
int len = 0 ;
lps [ 0 ] = 0 ;
int i = 1 ; while ( i < M ) { if ( pat [ i ] == pat [ len ] ) { len ++ ; lps [ i ] = len ; i ++ ; }
else {
if ( len != 0 ) { len = lps [ len - 1 ] ; } else { lps [ i ] = 0 ; i ++ ; } } } }
static int countRotations ( string s ) {
string s1 = s . Substring ( 1 , s . Length - 1 ) + s ;
char [ ] pat = s . ToCharArray ( ) ; char [ ] text = s1 . ToCharArray ( ) ;
return 1 + KMPSearch ( pat , text ) ; }
public static void Main ( params string [ ] args ) { string s1 = " geeks " ; Console . Write ( countRotations ( s1 ) ) ; } }
using System ; class GFG {
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
using System ; using System . Collections . Generic ; class GFG { static int [ ] parent = new int [ 26 ] ;
static int find ( int x ) { if ( x != parent [ x ] ) return parent [ x ] = find ( parent [ x ] ) ; return x ; }
static void join ( int x , int y ) { int px = find ( x ) ; int pz = find ( y ) ; if ( px != pz ) { parent [ pz ] = px ; } }
static bool convertible ( String s1 , String s2 ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < s1 . Length ; i ++ ) { if ( ! mp . ContainsKey ( s1 [ i ] - ' a ' ) ) { mp . Add ( s1 [ i ] - ' a ' , s2 [ i ] - ' a ' ) ; } else { if ( mp [ s1 [ i ] - ' a ' ] != s2 [ i ] - ' a ' ) return false ; } }
foreach ( KeyValuePair < int , int > it in mp ) { if ( it . Key == it . Value ) continue ; else { if ( find ( it . Key ) == find ( it . Value ) ) return false ; else join ( it . Key , it . Value ) ; } } return true ; }
static void initialize ( ) { for ( int i = 0 ; i < 26 ; i ++ ) { parent [ i ] = i ; } }
public static void Main ( String [ ] args ) { String s1 , s2 ; s1 = " abbcaa " ; s2 = " bccdbb " ; initialize ( ) ; if ( convertible ( s1 , s2 ) ) Console . Write ( " Yes " + " STRNEWLINE " ) ; else Console . Write ( " No " + " STRNEWLINE " ) ; } }
using System ; class GFG { static int SIZE = 26 ;
static void SieveOfEratosthenes ( bool [ ] prime , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i < p_size ; i += p ) prime [ i ] = false ; } } }
static void printChar ( string str , int n ) { bool [ ] prime = new bool [ n + 1 ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) prime [ i ] = true ;
SieveOfEratosthenes ( prime , str . Length + 1 ) ;
int [ ] freq = new int [ SIZE ] ;
for ( int i = 0 ; i < SIZE ; i ++ ) freq [ i ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( prime [ freq [ str [ i ] - ' a ' ] ] ) { Console . Write ( str [ i ] ) ; } } }
public static void Main ( String [ ] args ) { String str = " geeksforgeeks " ; int n = str . Length ; printChar ( str , n ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool prime ( int n ) { if ( n <= 1 ) return false ; int max_div = ( int ) Math . Floor ( Math . Sqrt ( n ) ) ; for ( int i = 2 ; i < 1 + max_div ; i ++ ) { if ( n % i == 0 ) return false ; } return true ; } static void checkString ( string s ) {
Dictionary < char , int > freq = new Dictionary < char , int > ( ) ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( ! freq . ContainsKey ( s [ i ] ) ) freq [ s [ i ] ] = 0 ; freq [ s [ i ] ] += 1 ; }
for ( int i = 0 ; i < s . Length ; i ++ ) { if ( prime ( freq [ s [ i ] ] ) ) Console . Write ( s [ i ] ) ; } }
public static void Main ( ) { string s = " geeksforgeeks " ;
checkString ( s ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int SIZE = 26 ;
static void printChar ( String str , int n ) {
int [ ] freq = new int [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] % 2 == 0 ) { Console . Write ( str [ i ] ) ; } } }
public static void Main ( String [ ] args ) { String str = " geeksforgeeks " ; int n = str . Length ; printChar ( str , n ) ; } }
using System ; class GFG {
static bool CompareAlphanumeric ( char [ ] str1 , char [ ] str2 ) {
int i , j ; i = 0 ; j = 0 ;
int len1 = str1 . Length ;
int len2 = str2 . Length ;
while ( i <= len1 && j <= len2 ) {
while ( i < len1 && ( ! ( ( str1 [ i ] >= ' a ' && str1 [ i ] <= ' z ' ) || ( str1 [ i ] >= ' A ' && str1 [ i ] <= ' Z ' ) || ( str1 [ i ] >= '0' && str1 [ i ] <= '9' ) ) ) ) { i ++ ; }
while ( j < len2 && ( ! ( ( str2 [ j ] >= ' a ' && str2 [ j ] <= ' z ' ) || ( str2 [ j ] >= ' A ' && str2 [ j ] <= ' Z ' ) || ( str2 [ j ] >= '0' && str2 [ j ] <= '9' ) ) ) ) { j ++ ; }
if ( i == len1 && j == len2 ) { return true ; }
else if ( str1 [ i ] != str2 [ j ] ) { return false ; }
else { i ++ ; j ++ ; } }
return false ; }
static void CompareAlphanumericUtil ( string str1 , string str2 ) { bool res ;
res = CompareAlphanumeric ( str1 . ToCharArray ( ) , str2 . ToCharArray ( ) ) ;
if ( res == true ) { Console . WriteLine ( " Equal " ) ; }
else { Console . WriteLine ( " Unequal " ) ; } }
public static void Main ( ) { string str1 , str2 ; str1 = " Ram , ▁ Shyam " ; str2 = " ▁ Ram ▁ - ▁ Shyam . " ; CompareAlphanumericUtil ( str1 , str2 ) ; str1 = " abc123" ; str2 = "123abc " ; CompareAlphanumericUtil ( str1 , str2 ) ; } }
using System ; class GFG {
static void solveQueries ( String str , int [ , ] query ) {
int len = str . Length ;
int Q = query . GetLength ( 0 ) ;
int [ , ] pre = new int [ len , 26 ] ;
for ( int i = 0 ; i < len ; i ++ ) {
pre [ i , str [ i ] - ' a ' ] ++ ;
if ( i > 0 ) {
for ( int j = 0 ; j < 26 ; j ++ ) pre [ i , j ] += pre [ i - 1 , j ] ; } }
for ( int i = 0 ; i < Q ; i ++ ) {
int l = query [ i , 0 ] ; int r = query [ i , 1 ] ; int maxi = 0 ; char c = ' a ' ;
for ( int j = 0 ; j < 26 ; j ++ ) {
int times = pre [ r , j ] ;
if ( l > 0 ) times -= pre [ l - 1 , j ] ;
if ( times > maxi ) { maxi = times ; c = ( char ) ( ' a ' + j ) ; } }
Console . WriteLine ( " Query " + ( i + 1 ) + " : ▁ " + c ) ; } }
public static void Main ( String [ ] args ) { String str = " striver " ; int [ , ] query = { { 0 , 1 } , { 1 , 6 } , { 5 , 6 } } ; solveQueries ( str , query ) ; } }
using System ; class GFG {
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
using System ; class GFG {
public static void printChar ( String str , int n ) {
int [ ] freq = new int [ 26 ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] % 2 == 1 ) { Console . Write ( str [ i ] ) ; } } }
public static void Main ( String [ ] args ) { String str = " geeksforgeeks " ; int n = str . Length ; printChar ( str , n ) ; } }
using System ; class GFG {
static int minOperations ( string str , int n ) {
int i , lastUpper = - 1 , firstLower = - 1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( Char . IsUpper ( str [ i ] ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( Char . IsLower ( str [ i ] ) ) { firstLower = i ; break ; } }
if ( lastUpper == - 1 firstLower == - 1 ) return 0 ;
int countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( Char . IsUpper ( str [ i ] ) ) { countUpper ++ ; } }
int countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( Char . IsLower ( str [ i ] ) ) { countLower ++ ; } }
return Math . Min ( countLower , countUpper ) ; }
public static void Main ( ) { string str = " geEksFOrGEekS " ; int n = str . Length ; Console . WriteLine ( minOperations ( str , n ) ) ; } }
using System ; using System . Collections ; class GFG {
public static int Betrothed_Sum ( int n ) {
ArrayList set = new ArrayList ( ) ; for ( int number_1 = 1 ; number_1 < n ; number_1 ++ ) {
int sum_divisor_1 = 1 ;
int i = 2 ; while ( i * i <= number_1 ) { if ( number_1 % i == 0 ) { sum_divisor_1 = sum_divisor_1 + i ; if ( i * i != number_1 ) sum_divisor_1 += number_1 / i ; } i ++ ; } if ( sum_divisor_1 > number_1 ) { int number_2 = sum_divisor_1 - 1 ; int sum_divisor_2 = 1 ; int j = 2 ; while ( j * j <= number_2 ) { if ( number_2 % j == 0 ) { sum_divisor_2 += j ; if ( j * j != number_2 ) sum_divisor_2 += number_2 / j ; } j = j + 1 ; } if ( sum_divisor_2 == number_1 + 1 && number_1 <= n && number_2 <= n ) { set . Add ( number_1 ) ; set . Add ( number_2 ) ; } } }
int Summ = 0 ; for ( int i = 0 ; i < set . Count ; i ++ ) { if ( ( int ) set [ i ] <= n ) Summ += ( int ) set [ i ] ; } return Summ ; }
static public void Main ( ) { int n = 78 ; Console . WriteLine ( Betrothed_Sum ( n ) ) ; } }
using System ; class GFG {
static float rainDayProbability ( int [ ] a , int n ) { float count = 0 , m ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
public static void Main ( ) { int [ ] a = { 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 } ; int n = a . Length ; Console . WriteLine ( rainDayProbability ( a , n ) ) ; } }
using System ; class Maths {
static double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / Math . Pow ( i , i ) ; sums += ser ; } return sums ; }
public static void Main ( ) { int n = 3 ; double res = Series ( n ) ; res = Math . Round ( res * 100000.0 ) / 100000.0 ; Console . Write ( res ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static string lexicographicallyMaximum ( string S , int N ) {
Dictionary < char , int > M = new Dictionary < char , int > ( ) ;
for ( int i = 0 ; i < N ; ++ i ) { if ( M . ContainsKey ( S [ i ] ) ) M [ S [ i ] ] ++ ; else M . Add ( S [ i ] , 1 ) ; }
List < char > V = new List < char > ( ) ; for ( char i = ' a ' ; i < ( char ) ( ' a ' + Math . Min ( N , 25 ) ) ; ++ i ) { if ( M . ContainsKey ( i ) == false ) { V . Add ( i ) ; } }
int j = V . Count - 1 ;
for ( int i = 0 ; i < N ; ++ i ) {
if ( S [ i ] >= ( ' a ' + Math . Min ( N , 25 ) ) || ( M . ContainsKey ( S [ i ] ) && M [ S [ i ] ] > 1 ) ) { if ( V [ j ] < S [ i ] ) continue ;
M [ S [ i ] ] -- ;
S = S . Substring ( 0 , i ) + V [ j ] + S . Substring ( i + 1 ) ;
j -- ; } if ( j < 0 ) break ; } int l = 0 ;
for ( int i = N - 1 ; i >= 0 ; i -- ) { if ( l > j ) break ; if ( S [ i ] >= ( ' a ' + Math . Min ( N , 25 ) ) || M . ContainsKey ( S [ i ] ) && M [ S [ i ] ] > 1 ) {
M [ S [ i ] ] -- ;
S = S . Substring ( 0 , i ) + V [ l ] + S . Substring ( i + 1 ) ;
l ++ ; } }
return S ; }
public static void Main ( ) {
string S = " abccefghh " ; int N = S . Length ;
Console . Write ( lexicographicallyMaximum ( S , N ) ) ; } }
using System ; using System . Collections . Generic ; using System . Linq ; class GFG {
static bool isConsistingSubarrayUtil ( int [ ] arr , int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n ; ++ i ) {
if ( mp . ContainsKey ( arr [ i ] ) == true ) mp [ arr [ i ] ] += 1 ; else mp [ arr [ i ] ] = 1 ; } var val = mp . Keys . ToList ( ) ;
foreach ( var key in val ) {
if ( mp [ key ] > 1 ) { return true ; } }
return false ; }
static void isConsistingSubarray ( int [ ] arr , int N ) { if ( isConsistingSubarrayUtil ( arr , N ) ) { Console . Write ( " Yes " ) ; } else { Console . Write ( " No " ) ; } }
public static void Main ( ) {
int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 1 } ;
int N = arr . Length ;
isConsistingSubarray ( arr , N ) ; } }
using System ; using System . Collections . Generic ; class GFG { static bool [ ] isPrime ;
static HashSet < int > createhashmap ( int Max ) {
HashSet < int > hashmap = new HashSet < int > ( ) ;
int curr = 1 ;
int prev = 0 ;
hashmap . Add ( prev ) ;
while ( curr < Max ) {
hashmap . Add ( curr ) ;
int temp = curr ;
curr = curr + prev ;
prev = temp ; } return hashmap ; }
static void SieveOfEratosthenes ( int Max ) {
isPrime = new bool [ Max ] ; for ( int i = 0 ; i < Max ; i ++ ) isPrime [ i ] = true ; isPrime [ 0 ] = false ; isPrime [ 1 ] = false ;
for ( int p = 2 ; p * p <= Max ; p ++ ) {
if ( isPrime [ p ] ) {
for ( int i = p * p ; i <= Max ; i += p ) {
isPrime [ i ] = false ; } } } }
static void cntFibonacciPrime ( int [ ] arr , int N ) {
int Max = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) {
Max = Math . Max ( Max , arr [ i ] ) ; }
SieveOfEratosthenes ( Max ) ;
HashSet < int > hashmap = createhashmap ( Max ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 1 ) continue ;
if ( ( hashmap . Contains ( arr [ i ] ) ) && ! isPrime [ arr [ i ] ] ) {
Console . Write ( arr [ i ] + " ▁ " ) ; } } }
public static void Main ( String [ ] args ) { int [ ] arr = { 13 , 55 , 7 , 3 , 5 , 21 , 233 , 144 , 89 } ; int N = arr . Length ; cntFibonacciPrime ( arr , N ) ; } }
using System ; class GFG {
static int key ( int N ) {
String num = " " + N ; int ans = 0 ; int j = 0 ;
for ( j = 0 ; j < num . Length ; j ++ ) {
if ( ( num [ j ] - 48 ) % 2 == 0 ) { int add = 0 ; int i ;
for ( i = j ; j < num . Length ; j ++ ) { add += num [ j ] - 48 ;
if ( add % 2 == 1 ) break ; } if ( add == 0 ) { ans *= 10 ; } else { int digit = ( int ) Math . Floor ( Math . Log10 ( add ) + 1 ) ; ans *= ( int ) ( Math . Pow ( 10 , digit ) ) ;
ans += add ; }
i = j ; } else {
int add = 0 ; int i ;
for ( i = j ; j < num . Length ; j ++ ) { add += num [ j ] - 48 ;
if ( add % 2 == 0 ) { break ; } } if ( add == 0 ) { ans *= 10 ; } else { int digit = ( int ) Math . Floor ( Math . Log10 ( add ) + 1 ) ; ans *= ( int ) ( Math . Pow ( 10 , digit ) ) ;
ans += add ; }
i = j ; } }
if ( j + 1 >= num . Length ) { return ans ; } else { return ans += num [ num . Length - 1 ] - 48 ; } }
public static void Main ( String [ ] args ) { int N = 1667848271 ; Console . Write ( key ( N ) ) ; } }
using System ; class GFG {
static void sentinelSearch ( int [ ] arr , int n , int key ) {
int last = arr [ n - 1 ] ;
arr [ n - 1 ] = key ; int i = 0 ; while ( arr [ i ] != key ) i ++ ;
arr [ n - 1 ] = last ; if ( ( i < n - 1 ) || ( arr [ n - 1 ] == key ) ) Console . WriteLine ( key + " ▁ is ▁ present " + " ▁ at ▁ index ▁ " + i ) ; else Console . WriteLine ( " Element ▁ Not ▁ found " ) ; }
public static void Main ( ) { int [ ] arr = { 10 , 20 , 180 , 30 , 60 , 50 , 110 , 100 , 70 } ; int n = arr . Length ; int key = 180 ; sentinelSearch ( arr , n , key ) ; } }
using System ; class GFG {
static int maximum_middle_value ( int n , int k , int [ ] arr ) {
int ans = - 1 ;
int low = ( n + 1 - k ) / 2 ; int high = ( n + 1 - k ) / 2 + k ;
for ( int i = low ; i <= high ; i ++ ) {
ans = Math . Max ( ans , arr [ i - 1 ] ) ; }
return ans ; }
static public void Main ( ) { int n = 5 , k = 2 ; int [ ] arr = { 9 , 5 , 3 , 7 , 10 } ; Console . WriteLine ( maximum_middle_value ( n , k , arr ) ) ; n = 9 ; k = 3 ; int [ ] arr1 = { 2 , 4 , 3 , 9 , 5 , 8 , 7 , 6 , 10 } ; Console . WriteLine ( maximum_middle_value ( n , k , arr1 ) ) ; } }
using System ; class GFG {
static int ternarySearch ( int l , int r , int key , int [ ] ar ) { if ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
return ternarySearch ( l , mid1 - 1 , key , ar ) ; } else if ( key > ar [ mid2 ] ) {
return ternarySearch ( mid2 + 1 , r , key , ar ) ; } else {
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) ; } }
return - 1 ; }
public static void Main ( ) { int l , r , p , key ;
int [ ] ar = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
Console . WriteLine ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
Console . WriteLine ( " Index ▁ of ▁ " + key + " ▁ is ▁ " + p ) ; } }
using System ; class GFG {
public class Point { public int x , y ; public Point ( int x , int y ) { this . x = x ; this . y = y ; } } ;
static int findmin ( Point [ ] p , int n ) { int a = 0 , b = 0 , c = 0 , d = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( p [ i ] . x <= 0 ) a ++ ;
else if ( p [ i ] . x >= 0 ) ++ ;
if ( p [ i ] . y >= 0 ) c ++ ;
else if ( p [ i ] . y <= 0 ) ++ ; } return . Min ( Math . Min ( a , b ) , Math . Min ( c , d ) ) ; }
public static void Main ( String [ ] args ) { Point [ ] p = { new Point ( 1 , 1 ) , new Point ( 2 , 2 ) , new Point ( - 1 , - 1 ) , new Point ( - 2 , 2 ) } ; int n = p . Length ; Console . WriteLine ( findmin ( p , n ) ) ; } }
using System ; class GFG {
static void maxOps ( int a , int b , int c ) {
int [ ] arr = { a , b , c } ;
int count = 0 ; while ( 1 != 0 ) {
Array . Sort ( arr ) ;
if ( arr [ 0 ] == 0 && arr [ 1 ] == 0 ) break ;
arr [ 1 ] -= 1 ; arr [ 2 ] -= 1 ;
count += 1 ; }
Console . WriteLine ( count ) ; }
public static void Main ( String [ ] args ) {
int a = 4 , b = 3 , c = 2 ; maxOps ( a , b , c ) ; } }
using System ; class GFG { static int MAX = 26 ;
int [ ] lower = new int [ MAX ] ; int [ ] upper = new int [ MAX ] ; int i = 0 , j = 0 ; for ( i = 0 ; i < n ; i ++ ) {
if ( char . IsLower ( s [ i ] ) ) lower [ s [ i ] - ' a ' ] ++ ;
else if ( char . IsUpper ( s [ i ] ) ) upper [ s [ i ] - ' A ' ] ++ ; }
i = 0 ; while ( i < MAX && lower [ i ] == 0 ) i ++ ; while ( j < MAX && upper [ j ] == 0 ) j ++ ;
for ( int k = 0 ; k < n ; k ++ ) {
if ( char . IsLower ( s [ k ] ) ) { while ( lower [ i ] == 0 ) i ++ ; s [ k ] = ( char ) ( i + ' a ' ) ;
lower [ i ] -- ; }
else if ( char . IsUpper ( s [ k ] ) ) { while ( upper [ j ] == 0 ) j ++ ; s [ k ] = ( char ) ( j + ' A ' ) ;
upper [ j ] -- ; } }
return String . Join ( " " , s ) ; }
public static void Main ( String [ ] args ) { String s = " gEeksfOrgEEkS " ; int n = s . Length ; Console . WriteLine ( getSortedString ( s . ToCharArray ( ) , n ) ) ; } }
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
using System ; public class ReverseWords { public static void Main ( ) { string [ ] s = " i ▁ like ▁ this ▁ program ▁ very ▁ much " . Split ( ' ▁ ' ) ; string ans = " " ; for ( int i = s . Length - 1 ; i >= 0 ; i -- ) { ans += s [ i ] + " ▁ " ; } Console . Write ( " Reversed ▁ String : STRNEWLINE " ) ; Console . Write ( ans . Substring ( 0 , ans . Length - 1 ) ) ; } }
using System ; class GFG {
public static void SieveOfEratosthenes ( bool [ ] prime , int n ) { for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } }
public static void segregatePrimeNonPrime ( bool [ ] prime , int [ ] arr , int N ) {
SieveOfEratosthenes ( prime , 10000000 ) ;
int left = 0 , right = N - 1 ;
while ( left < right ) {
while ( prime [ arr [ left ] ] ) left ++ ;
while ( ! prime [ arr [ right ] ] ) right -- ;
if ( left < right ) {
int temp = arr [ left ] ; arr [ left ] = arr [ right ] ; arr [ right ] = temp ; left ++ ; right -- ; } }
for ( int i = 0 ; i < N ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { bool [ ] prime = new bool [ 10000001 ] ; for ( int i = 0 ; i < prime . Length ; i ++ ) prime [ i ] = true ; int [ ] arr = { 2 , 3 , 4 , 6 , 7 , 8 , 9 , 10 } ; int N = arr . Length ;
segregatePrimeNonPrime ( prime , arr , N ) ; } }
using System ; class GFG {
static int findDepthRec ( char [ ] tree , int n , int index ) { if ( index >= n tree [ index ] == ' l ' ) return 0 ;
index ++ ; int left = findDepthRec ( tree , n , index ) ;
index ++ ; int right = findDepthRec ( tree , n , index ) ; return Math . Max ( left , right ) + 1 ; }
static int findDepth ( char [ ] tree , int n ) { int index = 0 ; return ( findDepthRec ( tree , n , index ) ) ; }
static public void Main ( ) { char [ ] tree = " nlnnlll " . ToCharArray ( ) ; int n = tree . Length ; Console . WriteLine ( findDepth ( tree , n ) ) ; } }
using System ; class GFG {
class Node { public int key ; public Node left , right ; }
static Node newNode ( int item ) { Node temp = new Node ( ) ; temp . key = item ; temp . left = null ; temp . right = null ; return temp ; }
static Node insert ( Node node , int key ) {
if ( node == null ) return newNode ( key ) ;
if ( key < node . key ) node . left = insert ( node . left , key ) ; else if ( key > node . key ) node . right = insert ( node . right , key ) ;
return node ; }
static int findMaxforN ( Node root , int N ) {
if ( root == null ) return - 1 ; if ( root . key == N ) return N ;
else if ( root . key < N ) { int k = findMaxforN ( root . right , N ) ; if ( k == - 1 ) return root . key ; else return k ; }
else if ( root . key > N ) return findMaxforN ( root . left , N ) ; return - 1 ; }
public static void Main ( String [ ] args ) { int N = 4 ;
Node root = null ; root = insert ( root , 25 ) ; insert ( root , 2 ) ; insert ( root , 1 ) ; insert ( root , 3 ) ; insert ( root , 12 ) ; insert ( root , 9 ) ; insert ( root , 21 ) ; insert ( root , 19 ) ; insert ( root , 25 ) ; Console . WriteLine ( findMaxforN ( root , N ) ) ; } }
public class Solution { public class Node { public Node left , right ; public int data ; }
public static Node createNode ( int x ) { Node p = new Node ( ) ; p . data = x ; p . left = p . right = null ; return p ; }
public static void insertNode ( Node root , int x ) { Node p = root , q = null ; while ( p != null ) { q = p ; if ( p . data < x ) { p = p . right ; } else { p = p . left ; } } if ( q == null ) { p = createNode ( x ) ; } else { if ( q . data < x ) { q . right = createNode ( x ) ; } else { q . left = createNode ( x ) ; } } }
public static int maxelpath ( Node q , int x ) { Node p = q ; int mx = - 1 ;
while ( p . data != x ) { if ( p . data > x ) { mx = Math . Max ( mx , p . data ) ; p = p . left ; } else { mx = Math . Max ( mx , p . data ) ; p = p . right ; } } return Math . Max ( mx , x ) ; }
public static int maximumElement ( Node root , int x , int y ) { Node p = root ;
while ( ( x < p . data && y < p . data ) || ( x > p . data && y > p . data ) ) {
if ( x < p . data && y < p . data ) { p = p . left ; }
else if ( x > p . data && y > p . data ) { p = p . right ; } }
return Math . Max ( maxelpath ( p , x ) , maxelpath ( p , y ) ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = new int [ ] { 18 , 36 , 9 , 6 , 12 , 10 , 1 , 8 } ; int a = 1 , b = 10 ; int n = arr . Length ;
Node root = createNode ( arr [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) { insertNode ( root , arr [ i ] ) ; } Console . WriteLine ( maximumElement ( root , a , b ) ) ; } }
public class solution { public class Node { public Node left , right ; public int info ;
public bool lthread ;
public bool rthread ; }
public static Node insert ( Node root , int ikey ) {
Node ptr = root ;
Node par = null ; while ( ptr != null ) {
if ( ikey == ( ptr . info ) ) { Console . Write ( " Duplicate ▁ Key ▁ ! STRNEWLINE " ) ; return root ; }
par = ptr ;
if ( ikey < ptr . info ) { if ( ptr . lthread == false ) { ptr = ptr . left ; } else { break ; } }
else { if ( ptr . rthread == false ) { ptr = ptr . right ; } else { break ; } } }
Node tmp = new Node ( ) ; tmp . info = ikey ; tmp . lthread = true ; tmp . rthread = true ; if ( par == null ) { root = tmp ; tmp . left = null ; tmp . right = null ; } else if ( ikey < ( par . info ) ) { tmp . left = par . left ; tmp . right = par ; par . lthread = false ; par . left = tmp ; } else { tmp . left = par ; tmp . right = par . right ; par . rthread = false ; par . right = tmp ; } return root ; }
public static Node inorderSuccessor ( Node ptr ) {
if ( ptr . rthread == true ) { return ptr . right ; }
ptr = ptr . right ; while ( ptr . lthread == false ) { ptr = ptr . left ; } return ptr ; }
public static void inorder ( Node root ) { if ( root == null ) { Console . Write ( " Tree ▁ is ▁ empty " ) ; }
Node ptr = root ; while ( ptr . lthread == false ) { ptr = ptr . left ; }
while ( ptr != null ) { Console . Write ( " { 0 : D } ▁ " , ptr . info ) ; ptr = inorderSuccessor ( ptr ) ; } }
public static void Main ( string [ ] args ) { Node root = null ; root = insert ( root , 20 ) ; root = insert ( root , 10 ) ; root = insert ( root , 30 ) ; root = insert ( root , 5 ) ; root = insert ( root , 16 ) ; root = insert ( root , 14 ) ; root = insert ( root , 17 ) ; root = insert ( root , 13 ) ; inorder ( root ) ; } }
public class Node { public Node left , right ; public int info ;
public bool lthread ;
public bool rthread ; } ;
using System ; class GFG { public class Node { public Node left , right ; public int info ;
public bool lthread ;
public bool rthread ; } ;
static Node insert ( Node root , int ikey ) {
Node ptr = root ;
Node par = null ; while ( ptr != null ) {
if ( ikey == ( ptr . info ) ) { Console . Write ( " Duplicate ▁ Key ▁ ! STRNEWLINE " ) ; return root ; }
par = ptr ;
if ( ikey < ptr . info ) { if ( ptr . lthread == false ) ptr = ptr . left ; else break ; }
else { if ( ptr . rthread == false ) ptr = ptr . right ; else break ; } }
Node tmp = new Node ( ) ; tmp . info = ikey ; tmp . lthread = true ; tmp . rthread = true ; if ( par == null ) { root = tmp ; tmp . left = null ; tmp . right = null ; } else if ( ikey < ( par . info ) ) { tmp . left = par . left ; tmp . right = par ; par . lthread = false ; par . left = tmp ; } else { tmp . left = par ; tmp . right = par . right ; par . rthread = false ; par . right = tmp ; } return root ; }
static Node inSucc ( Node ptr ) { if ( ptr . rthread == true ) return ptr . right ; ptr = ptr . right ; while ( ptr . lthread == false ) ptr = ptr . left ; return ptr ; }
static Node inorderSuccessor ( Node ptr ) {
if ( ptr . rthread == true ) return ptr . right ;
ptr = ptr . right ; while ( ptr . lthread == false ) ptr = ptr . left ; return ptr ; }
static void inorder ( Node root ) { if ( root == null ) Console . Write ( " Tree ▁ is ▁ empty " ) ;
Node ptr = root ; while ( ptr . lthread == false ) ptr = ptr . left ;
while ( ptr != null ) { Console . Write ( " { 0 } ▁ " , ptr . info ) ; ptr = inorderSuccessor ( ptr ) ; } } static Node inPred ( Node ptr ) { if ( ptr . lthread == true ) return ptr . left ; ptr = ptr . left ; while ( ptr . rthread == false ) ptr = ptr . right ; return ptr ; }
static Node caseA ( Node root , Node par , Node ptr ) {
if ( par == null ) root = null ;
else if ( ptr = = par . left ) { par . lthread = true ; par . left = ptr . left ; } else { par . rthread = true ; par . right = ptr . right ; } return root ; }
static Node caseB ( Node root , Node par , Node ptr ) { Node child ;
if ( ptr . lthread == false ) child = ptr . left ;
else child = ptr . right ;
if ( par == null ) root = child ;
else if ( ptr = = par . left ) par . left = child ; else par . = child ;
Node s = inSucc ( ptr ) ; Node p = inPred ( ptr ) ;
if ( ptr . lthread == false ) p . right = s ;
else { if ( ptr . rthread == false ) s . left = p ; } return root ; }
static Node caseC ( Node root , Node par , Node ptr ) {
Node parsucc = ptr ; Node succ = ptr . right ;
while ( succ . lthread == false ) { parsucc = succ ; succ = succ . left ; } ptr . info = succ . info ; if ( succ . lthread == true && succ . rthread == true ) root = caseA ( root , parsucc , succ ) ; else root = caseB ( root , parsucc , succ ) ; return root ; }
static Node delThreadedBST ( Node root , int dkey ) {
Node par = null , ptr = root ;
int found = 0 ;
while ( ptr != null ) { if ( dkey == ptr . info ) { found = 1 ; break ; } par = ptr ; if ( dkey < ptr . info ) { if ( ptr . lthread == false ) ptr = ptr . left ; else break ; } else { if ( ptr . rthread == false ) ptr = ptr . right ; else break ; } } if ( found == 0 ) Console . Write ( " dkey ▁ not ▁ present ▁ in ▁ tree STRNEWLINE " ) ;
else if ( ptr . lthread == false && ptr . rthread == false ) root = caseC ( root , par , ptr ) ;
else if ( ptr . lthread == false ) root = caseB ( root , par , ptr ) ;
else if ( ptr . rthread == false ) root = caseB ( root , par , ptr ) ;
else root = caseA ( root , par , ptr ) ; return root ; }
public static void Main ( String [ ] args ) { Node root = null ; root = insert ( root , 20 ) ; root = insert ( root , 10 ) ; root = insert ( root , 30 ) ; root = insert ( root , 5 ) ; root = insert ( root , 16 ) ; root = insert ( root , 14 ) ; root = insert ( root , 17 ) ; root = insert ( root , 13 ) ; root = delThreadedBST ( root , 20 ) ; inorder ( root ) ; } }
using System ; public class GFG { static void checkHV ( int [ , ] arr , int N , int M ) {
bool horizontal = true ; bool vertical = true ;
for ( int i = 0 , k = N - 1 ; i < N / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < M ; j ++ ) {
if ( arr [ i , j ] != arr [ k , j ] ) { horizontal = false ; break ; } } }
for ( int i = 0 , k = M - 1 ; i < M / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i , j ] != arr [ k , j ] ) { horizontal = false ; break ; } } } if ( ! horizontal && ! vertical ) Console . WriteLine ( " NO " ) ; else if ( horizontal && ! vertical ) Console . WriteLine ( " HORIZONTAL " ) ; else if ( vertical && ! horizontal ) Console . WriteLine ( " VERTICAL " ) ; else Console . WriteLine ( " BOTH " ) ; }
static public void Main ( ) { int [ , ] mat = { { 1 , 0 , 1 } , { 0 , 0 , 0 } , { 1 , 0 , 1 } } ; checkHV ( mat , 3 , 3 ) ; } }
using System ; class GFG { static int R = 3 ; static int C = 4 ;
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
static void replacematrix ( int [ , ] mat , int n , int m ) { int [ ] rgcd = new int [ R ] ; int [ ] cgcd = new int [ C ] ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { rgcd [ i ] = gcd ( rgcd [ i ] , mat [ i , j ] ) ; cgcd [ j ] = gcd ( cgcd [ j ] , mat [ i , j ] ) ; } }
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < m ; j ++ ) mat [ i , j ] = Math . Max ( rgcd [ i ] , cgcd [ j ] ) ; }
static public void Main ( ) { int [ , ] m = { { 1 , 2 , 3 , 3 } , { 4 , 5 , 6 , 6 } , { 7 , 8 , 9 , 9 } , } ; replacematrix ( m , R , C ) ; for ( int i = 0 ; i < R ; i ++ ) { for ( int j = 0 ; j < C ; j ++ ) Console . Write ( m [ i , j ] + " ▁ " ) ; Console . WriteLine ( ) ; } } }
using System ; class GFG { static int N = 4 ;
static void add ( int [ , ] A , int [ , ] B , int [ , ] C ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i , j ] = A [ i , j ] + B [ i , j ] ; }
public static void Main ( ) { int [ , ] A = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int [ , ] B = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int [ , ] C = new int [ N , N ] ; int i , j ; add ( A , B , C ) ; Console . WriteLine ( " Result ▁ matrix ▁ is ▁ " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) Console . Write ( C [ i , j ] + " ▁ " ) ; Console . WriteLine ( ) ; } } }
using System ; class GFG { static int N = 4 ;
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
using System ; class GFG {
static int maxTripletSum ( int [ ] arr , int n ) {
Array . Sort ( arr ) ;
return arr [ n - 1 ] + arr [ n - 2 ] + arr [ n - 3 ] ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( maxTripletSum ( arr , n ) ) ; } }
using System ; class GFG {
static int maxTripletSum ( int [ ] arr , int n ) {
int maxA = - 100000000 , maxB = - 100000000 ; int maxC = - 100000000 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > maxA ) { maxC = maxB ; maxB = maxA ; maxA = arr [ i ] ; }
else if ( arr [ i ] > maxB ) { maxC = maxB ; maxB = arr [ i ] ; }
else if ( arr [ i ] > maxC ) maxC = arr [ i ] ; } return ( maxA + maxB + maxC ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . Length ; Console . WriteLine ( maxTripletSum ( arr , n ) ) ; } }
using System ; class GFG { public static int search ( int [ ] arr , int x ) { int n = arr . Length ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == x ) return i ; } return - 1 ; }
public static void Main ( ) { int [ ] arr = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ;
int result = search ( arr , x ) ; if ( result == - 1 ) Console . WriteLine ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) ; else Console . WriteLine ( " Element ▁ is ▁ present ▁ at ▁ index ▁ " + result ) ; } }
using System ; class GFG { public static void search ( int [ ] arr , int search_Element ) { int left = 0 ; int length = arr . Length ; int right = length - 1 ; int position = - 1 ;
for ( left = 0 ; left <= right ; ) {
if ( arr [ left ] == search_Element ) { position = left ; Console . WriteLine ( " Element ▁ found ▁ in ▁ Array ▁ at ▁ " + ( position + 1 ) + " ▁ Position ▁ with ▁ " + ( left + 1 ) + " ▁ Attempt " ) ; break ; }
if ( arr [ right ] == search_Element ) { position = right ; Console . WriteLine ( " Element ▁ found ▁ in ▁ Array ▁ at ▁ " + ( position + 1 ) + " ▁ Position ▁ with ▁ " + ( length - right ) + " ▁ Attempt " ) ; break ; } left ++ ; right -- ; }
if ( position == - 1 ) Console . WriteLine ( " Not ▁ found ▁ in ▁ Array ▁ with ▁ " + left + " ▁ Attempt " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int search_element = 5 ;
search ( arr , search_element ) ; } }
using System ; class GFG {
static void countsort ( char [ ] arr ) { int n = arr . Length ;
char [ ] output = new char [ n ] ;
int [ ] count = new int [ 256 ] ; for ( int i = 0 ; i < 256 ; ++ i ) count [ i ] = 0 ;
for ( int i = 0 ; i < n ; ++ i ) ++ count [ arr [ i ] ] ;
for ( int i = 1 ; i <= 255 ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( int i = 0 ; i < n ; ++ i ) arr [ i ] = output [ i ] ; }
public static void Main ( ) { char [ ] arr = { ' g ' , ' e ' , ' e ' , ' k ' , ' s ' , ' f ' , ' o ' , ' r ' , ' g ' , ' e ' , ' e ' , ' k ' , ' s ' } ; countsort ( arr ) ; Console . Write ( " Sorted ▁ character ▁ array ▁ is ▁ " ) ; for ( int i = 0 ; i < arr . Length ; ++ i ) Console . Write ( arr [ i ] ) ; } }
using System ; using System . Collections . Generic ; using System . Linq ; class GFG {
static void countSort ( int [ ] arr ) { int max = arr . Max ( ) ; int min = arr . Min ( ) ; int range = max - min + 1 ; int [ ] count = new int [ range ] ; int [ ] output = new int [ arr . Length ] ; for ( int i = 0 ; i < arr . Length ; i ++ ) { count [ arr [ i ] - min ] ++ ; } for ( int i = 1 ; i < count . Length ; i ++ ) { count [ i ] += count [ i - 1 ] ; } for ( int i = arr . Length - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] - min ] - 1 ] = arr [ i ] ; count [ arr [ i ] - min ] -- ; } for ( int i = 0 ; i < arr . Length ; i ++ ) { arr [ i ] = output [ i ] ; } }
static void printArray ( int [ ] arr ) { for ( int i = 0 ; i < arr . Length ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } Console . WriteLine ( " " ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { - 5 , - 10 , 0 , - 3 , 8 , 5 , - 1 , 10 } ; countSort ( arr ) ; printArray ( arr ) ; } }
using System ; class GFG {
static int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
public static void Main ( ) { int n = 5 , k = 2 ; Console . Write ( " Value ▁ of ▁ C ( " + n + " , " + k + " ) ▁ is ▁ " + binomialCoeff ( n , k ) ) ; } }
using System ; class GFG { static int binomialCoeff ( int n , int k ) { int [ ] C = new int [ k + 1 ] ;
C [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = Math . Min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
public static void Main ( ) { int n = 5 , k = 2 ; Console . WriteLine ( " Value ▁ of ▁ C ( " + n + " ▁ " + k + " ) ▁ is ▁ " + binomialCoeff ( n , k ) ) ; } }
using System ; public class GFG {
static int binomialCoeff ( int n , int r ) { if ( r > n ) return 0 ; long m = 1000000007 ; long [ ] inv = new long [ r + 1 ] ; inv [ 0 ] = 1 ; if ( r + 1 >= 2 ) inv [ 1 ] = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { inv [ i ] = m - ( m / i ) * inv [ ( int ) ( m % i ) ] % m ; } int ans = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { ans = ( int ) ( ( ( ans % m ) * ( inv [ i ] % m ) ) % m ) ; }
for ( int i = n ; i >= ( n - r + 1 ) ; i -- ) { ans = ( int ) ( ( ( ans % m ) * ( i % m ) ) % m ) ; } return ans ; }
public static void Main ( String [ ] args ) { int n = 5 , r = 2 ; Console . Write ( " Value ▁ of ▁ C ( " + n + " , ▁ " + r + " ) ▁ is ▁ " + binomialCoeff ( n , r ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static bool findPartiion ( int [ ] arr , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; bool [ ] part = new bool [ sum / 2 + 1 ] ;
for ( i = 0 ; i <= sum / 2 ; i ++ ) { part [ i ] = false ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = sum / 2 ; j >= arr [ i ] ; j -- ) {
if ( part [ j - arr [ i ] ] == true j == arr [ i ] ) part [ j ] = true ; } } return part [ sum / 2 ] ; }
static void Main ( ) { int [ ] arr = { 1 , 3 , 3 , 2 , 3 , 2 } ; int n = 6 ;
if ( findPartiion ( arr , n ) == true ) Console . WriteLine ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " + " subsets ▁ of ▁ equal ▁ sum " ) ; else Console . WriteLine ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " + " two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
using System ; class GFG {
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
using System ; class GFG {
static bool isSubsetSum ( int [ ] set , int n , int sum ) {
bool [ , ] subset = new bool [ sum + 1 , n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 , i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i , 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i , j ] = subset [ i , j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i , j ] = subset [ i , j ] || subset [ i - set [ j - 1 ] , j - 1 ] ; } } return subset [ sum , n ] ; }
public static void Main ( ) { int [ ] set = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . Length ; if ( isSubsetSum ( set , n , sum ) == true ) Console . WriteLine ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else Console . WriteLine ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; } }
using System ; class GFG {
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
static void Main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) Console . WriteLine ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " + N + " ▁ keystrokes ▁ is ▁ " + findoptimal ( N ) ) ; } }
using System ; public class GFG {
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int [ ] screen = new int [ N ] ;
int b ;
int n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = 0 ;
for ( b = n - 3 ; b >= 1 ; b -- ) {
int curr = ( n - b - 1 ) * screen [ b - 1 ] ; if ( curr > screen [ n - 1 ] ) screen [ n - 1 ] = curr ; } } return screen [ N - 1 ] ; }
public static void Main ( String [ ] args ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) Console . WriteLine ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ { 0 } ▁ keystrokes ▁ is ▁ { 1 } STRNEWLINE " , N , findoptimal ( N ) ) ; } }
using System ; class GFG {
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int [ ] screen = new int [ N ] ;
int n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = Math . Max ( 2 * screen [ n - 4 ] , Math . Max ( 3 * screen [ n - 5 ] , 4 * screen [ n - 6 ] ) ) ; } return screen [ N - 1 ] ; }
public static void Main ( String [ ] args ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) Console . Write ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with " + " ▁ { 0 } ▁ keystrokes ▁ is ▁ { 1 } STRNEWLINE " , N , findoptimal ( N ) ) ; } }
using System ; public class GFG {
static int power ( int x , int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; }
public static void Main ( ) { int x = 2 ; int y = 3 ; Console . Write ( power ( x , y ) ) ; } }
static int power ( int x , int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
using System ; public class GFG { static float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
public static void Main ( ) { float x = 2 ; int y = - 3 ; Console . Write ( power ( x , y ) ) ; } }
using System ; class GFG { public static int power ( int x , int y ) {
if ( y == 0 ) return 1 ;
if ( x == 0 ) return 0 ;
return x * power ( x , y - 1 ) ; }
public static void Main ( String [ ] args ) { int x = 2 ; int y = 3 ; Console . WriteLine ( power ( x , y ) ) ; } }
using System ; public class GFG { public static int power ( int x , int y ) {
return ( int ) Math . Pow ( x , y ) ; }
static public void Main ( ) { int x = 2 ; int y = 3 ; Console . WriteLine ( power ( x , y ) ) ; } }
using System ; class GFG {
static float squareRoot ( float n ) {
float x = n ; float y = 1 ;
double e = 0.000001 ; while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
public static void Main ( ) { int n = 50 ; Console . Write ( " Square ▁ root ▁ of ▁ " + n + " ▁ is ▁ " + squareRoot ( n ) ) ; } }
using System ; class GFG {
static float getAvg ( float prev_avg , float x , int n ) { return ( prev_avg * n + x ) / ( n + 1 ) ; }
static void streamAvg ( float [ ] arr , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( avg , arr [ i ] , i ) ; Console . WriteLine ( " Average ▁ of ▁ { 0 } ▁ " + " numbers ▁ is ▁ { 1 } " , i + 1 , avg ) ; } return ; }
public static void Main ( String [ ] args ) { float [ ] arr = { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . Length ; streamAvg ( arr , n ) ; } }
using System ; class GFG { static int sum , n ;
static float getAvg ( int x ) { sum += x ; return ( ( ( float ) sum ) / ++ n ) ; }
static void streamAvg ( float [ ] arr , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( ( int ) arr [ i ] ) ; Console . WriteLine ( " Average ▁ of ▁ { 0 } ▁ numbers ▁ " + " is ▁ { 1 } " , ( i + 1 ) , avg ) ; } return ; }
static int Main ( ) { float [ ] arr = new float [ ] { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . Length ; streamAvg ( arr , n ) ; return 0 ; } }
using System ; class BinomialCoefficient {
static int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
public static void Main ( ) { int n = 8 ; int k = 2 ; Console . Write ( " Value ▁ of ▁ C ( " + n + " , ▁ " + k + " ) ▁ " + " is " + " ▁ " + binomialCoeff ( n , k ) ) ; } }
using System ; namespace prime { public class GFG {
public static void primeFactors ( int n ) {
while ( n % 2 == 0 ) { Console . Write ( 2 + " ▁ " ) ; n /= 2 ; }
for ( int i = 3 ; i <= Math . Sqrt ( n ) ; i += 2 ) {
while ( n % i == 0 ) { Console . Write ( i + " ▁ " ) ; n /= i ; } }
if ( n > 2 ) Console . Write ( n ) ; }
public static void Main ( ) { int n = 315 ; primeFactors ( n ) ; } } }
using System ; class GFG {
static void printCombination ( int [ ] arr , int n , int r ) {
int [ ] data = new int [ r ] ;
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; }
static void combinationUtil ( int [ ] arr , int [ ] data , int start , int end , int index , int r ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) Console . Write ( data [ j ] + " ▁ " ) ; Console . WriteLine ( " " ) ; return ; }
for ( int i = start ; i <= end && end - i + 1 >= r - index ; i ++ ) { data [ index ] = arr [ i ] ; combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; } }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . Length ; printCombination ( arr , n , r ) ; } }
using System ; class GFG {
static void printCombination ( int [ ] arr , int n , int r ) {
int [ ] data = new int [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
static void combinationUtil ( int [ ] arr , int n , int r , int index , int [ ] data , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) Console . Write ( data [ j ] + " ▁ " ) ; Console . WriteLine ( " " ) ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = arr . Length ; printCombination ( arr , n , r ) ; } }
using System ; class FindGroups {
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
using System ; class GFG {
static int nextPowerOf2 ( int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
public static void Main ( ) { int n = 5 ; Console . WriteLine ( nextPowerOf2 ( n ) ) ; } }
using System ; class GFG {
static void segregate0and1 ( int [ ] arr , int n ) {
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 0 ) count ++ ; }
for ( int i = 0 ; i < count ; i ++ ) arr [ i ] = 0 ;
for ( int i = count ; i < n ; i ++ ) arr [ i ] = 1 ; }
static void print ( int [ ] arr , int n ) { Console . WriteLine ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int n = arr . Length ; segregate0and1 ( arr , n ) ; print ( arr , n ) ; } }
using System ; class Segregate {
void segregate0and1 ( int [ ] arr , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
public static void Main ( ) { Segregate seg = new Segregate ( ) ; int [ ] arr = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = arr . Length ; seg . segregate0and1 ( arr , arr_size ) ; Console . WriteLine ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
using System ; class GFG {
static void segregate0and1 ( int [ ] arr ) { int type0 = 0 ; int type1 = arr . Length - 1 ; while ( type0 < type1 ) { if ( arr [ type0 ] == 1 ) { arr [ type1 ] = arr [ type1 ] + arr [ type0 ] ; arr [ type0 ] = arr [ type1 ] - arr [ type0 ] ; arr [ type1 ] = arr [ type1 ] - arr [ type0 ] ; type1 -- ; } else { type0 ++ ; } } }
public static void Main ( string [ ] args ) { int [ ] array = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; segregate0and1 ( array ) ; Console . Write ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; foreach ( int a in array ) { Console . Write ( a + " ▁ " ) ; } } }
using System ; using System . Collections . Generic ; class GFG { public static void distinctAdjacentElement ( int [ ] a , int n ) {
Dictionary < int , int > m = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n ; ++ i ) { if ( m . ContainsKey ( a [ i ] ) ) { int x = m [ a [ i ] ] + 1 ; m [ a [ i ] ] = x ; } else { m [ a [ i ] ] = 1 ; } }
int mx = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { if ( mx < m [ a [ i ] ] ) { mx = m [ a [ i ] ] ; } }
if ( mx > ( n + 1 ) / 2 ) { Console . WriteLine ( " NO " ) ; } else { Console . WriteLine ( " YES " ) ; } }
public static void Main ( string [ ] args ) { int [ ] a = new int [ ] { 7 , 7 , 7 , 7 } ; int n = 4 ; distinctAdjacentElement ( a , n ) ; } }
using System ; class GFG {
static int maxIndexDiff ( int [ ] arr , int n ) { int maxDiff = - 1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
public static void Main ( ) { int [ ] arr = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = arr . Length ; int maxDiff = maxIndexDiff ( arr , n ) ; Console . Write ( maxDiff ) ; } }
using System ; class GFG { public static void Main ( String [ ] args ) { int [ ] v = { 34 , 8 , 10 , 3 , 2 , 80 , 30 , 33 , 1 } ; int n = v . Length ; int [ ] maxFromEnd = new int [ n + 1 ] ; for ( int i = 0 ; i < maxFromEnd . Length ; i ++ ) maxFromEnd [ i ] = int . MinValue ;
for ( int i = v . Length - 1 ; i >= 0 ; i -- ) { maxFromEnd [ i ] = Math . Max ( maxFromEnd [ i + 1 ] , v [ i ] ) ; } int result = 0 ; for ( int i = 0 ; i < v . Length ; i ++ ) { int low = i + 1 , high = v . Length - 1 , ans = i ; while ( low <= high ) { int mid = ( low + high ) / 2 ; if ( v [ i ] <= maxFromEnd [ mid ] ) {
ans = Math . Max ( ans , mid ) ; low = mid + 1 ; } else { high = mid - 1 ; } }
result = Math . Max ( result , ans - i ) ; } Console . Write ( result + " STRNEWLINE " ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static int maxIndexDiff ( List < int > arr , int n ) {
Dictionary < int , List < int > > hashmap = new Dictionary < int , List < int > > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( hashmap . ContainsKey ( arr [ i ] ) ) { hashmap [ arr [ i ] ] . Add ( i ) ; } else { hashmap . Add ( arr [ i ] , new List < int > ( ) ) ; hashmap [ arr [ i ] ] . Add ( i ) ; } }
arr . Sort ( ) ; int maxDiff = - 1 ; int temp = n ;
for ( int i = 0 ; i < n ; i ++ ) { if ( temp > hashmap [ arr [ i ] ] [ 0 ] ) { temp = hashmap [ arr [ i ] ] [ 0 ] ; } maxDiff = Math . Max ( maxDiff , hashmap [ arr [ i ] ] [ hashmap [ arr [ i ] ] . Count - 1 ] - temp ) ; } return maxDiff ; }
static public void Main ( ) { int n = 9 ; List < int > arr = new List < int > ( ) ; arr . Add ( 34 ) ; arr . Add ( 8 ) ; arr . Add ( 10 ) ; arr . Add ( 3 ) ; arr . Add ( 2 ) ; arr . Add ( 80 ) ; arr . Add ( 30 ) ; arr . Add ( 33 ) ; arr . Add ( 1 ) ;
int ans = maxIndexDiff ( arr , n ) ; Console . WriteLine ( " The ▁ maxIndexDiff ▁ is ▁ : ▁ " + ans ) ; } }
using System ; using System . Collections . Generic ; using System . Linq ; class GFG { static void printRepeating ( int [ ] arr , int size ) {
SortedSet < int > s = new SortedSet < int > ( arr ) ;
foreach ( var n in s ) { Console . Write ( n + " ▁ " ) ; } }
public static void Main ( ) { int [ ] arr = { 1 , 3 , 2 , 2 , 1 } ; int n = arr . Length ; printRepeating ( arr , n ) ; } }
using System ; using System . Collections . Generic ; using System . Linq ;
public class GFG { static int minSwapsToSort ( int [ ] arr , int n ) {
List < List < int > > arrPos = new List < List < int > > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { arrPos . Add ( new List < int > ( ) { arr [ i ] , i } ) ; }
arrPos = arrPos . OrderBy ( x => x [ 0 ] ) . ToList ( ) ;
bool [ ] vis = new bool [ n ] ; Array . Fill ( vis , false ) ;
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( vis [ i ] arrPos [ i ] [ 1 ] == i ) continue ;
int cycle_size = 0 ; int j = i ; while ( ! vis [ j ] ) { vis [ j ] = true ;
j = arrPos [ j ] [ 1 ] ; cycle_size ++ ; }
ans += ( cycle_size - 1 ) ; }
return ans ; }
static int minSwapToMakeArraySame ( int [ ] a , int [ ] b , int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { mp . Add ( b [ i ] , i ) ; }
for ( int i = 0 ; i < n ; i ++ ) { b [ i ] = mp [ a [ i ] ] ; }
return minSwapsToSort ( b , n ) ; }
static public void Main ( ) { int [ ] a = { 3 , 6 , 4 , 8 } ; int [ ] b = { 4 , 6 , 8 , 3 } ; int n = a . Length ; Console . WriteLine ( minSwapToMakeArraySame ( a , b , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int missingK ( int [ ] a , int k , int n ) { int difference = 0 , ans = 0 , count = k ; bool flag = false ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = true ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return - 1 ; }
public static void Main ( ) {
int [ ] a = { 1 , 5 , 11 , 19 } ;
int k = 11 ; int n = a . Length ;
int missing = missingK ( a , k , n ) ; Console . Write ( missing ) ; } }
using System ; class GFG {
static int missingK ( int [ ] arr , int k ) { int n = arr . Length ; int l = 0 , u = n - 1 , mid ; while ( l <= u ) { mid = ( l + u ) / 2 ; int numbers_less_than_mid = arr [ mid ] - ( mid + 1 ) ;
if ( numbers_less_than_mid == k ) {
if ( mid > 0 && ( arr [ mid - 1 ] - ( mid ) ) == k ) { u = mid - 1 ; continue ; }
return arr [ mid ] - 1 ; }
if ( numbers_less_than_mid < k ) { l = mid + 1 ; } else if ( k < numbers_less_than_mid ) { u = mid - 1 ; } }
if ( u < 0 ) return k ;
int less = arr [ u ] - ( u + 1 ) ; k -= less ;
return arr [ u ] + k ; }
static void Main ( ) { int [ ] arr = { 2 , 3 , 4 , 7 , 11 } ; int k = 5 ;
Console . WriteLine ( " Missing ▁ kth ▁ number ▁ = ▁ " + missingK ( arr , k ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public class Node { public int data ; public Node next ; }
static void printList ( Node node ) { while ( node != null ) { Console . Write ( node . data + " ▁ " ) ; node = node . next ; } Console . WriteLine ( ) ; }
static Node newNode ( int key ) { Node temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
static Node insertBeg ( Node head , int val ) { Node temp = newNode ( val ) ; temp . next = head ; head = temp ; return head ; }
static void rearrangeOddEven ( Node head ) { Stack < Node > odd = new Stack < Node > ( ) ; Stack < Node > even = new Stack < Node > ( ) ; int i = 1 ; while ( head != null ) { if ( head . data % 2 != 0 && i % 2 == 0 ) {
odd . Push ( head ) ; } else if ( head . data % 2 == 0 && i % 2 != 0 ) {
even . Push ( head ) ; } head = head . next ; i ++ ; } while ( odd . Count > 0 && even . Count > 0 ) {
int k = odd . Peek ( ) . data ; odd . Peek ( ) . data = even . Peek ( ) . data ; even . Peek ( ) . data = k ; odd . Pop ( ) ; even . Pop ( ) ; } }
public static void Main ( String [ ] args ) { Node head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 1 ) ; Console . WriteLine ( " Linked ▁ List : " ) ; printList ( head ) ; rearrangeOddEven ( head ) ; Console . WriteLine ( " Linked ▁ List ▁ after ▁ " + " Rearranging : " ) ; printList ( head ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node next ; } ;
static void printList ( Node node ) { while ( node != null ) { Console . Write ( node . data + " ▁ " ) ; node = node . next ; } Console . WriteLine ( ) ; }
static Node newNode ( int key ) { Node temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
static Node insertBeg ( Node head , int val ) { Node temp = newNode ( val ) ; temp . next = head ; head = temp ; return head ; }
static Node rearrange ( Node head ) {
Node even ; Node temp , prev_temp ; Node i , j , k , l , ptr = null ;
temp = ( head ) . next ; prev_temp = head ; while ( temp != null ) {
Node x = temp . next ;
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
public static void Main ( String [ ] args ) { Node head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 1 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 10 ) ; Console . WriteLine ( " Linked ▁ List : " ) ; printList ( head ) ; Console . WriteLine ( " Rearranged ▁ List " ) ; head = rearrange ( head ) ; printList ( head ) ; } }
using System ; public class GFG {
static void print ( int [ , ] mat ) {
for ( int i = 0 ; i < mat . GetLength ( 0 ) ; i ++ ) {
for ( int j = 0 ; j < mat . GetLength ( 1 ) ; j ++ )
Console . Write ( mat [ i , j ] + " ▁ " ) ; Console . WriteLine ( ) ; } }
static void performSwap ( int [ , ] mat , int i , int j ) { int N = mat . GetLength ( 0 ) ;
int ei = N - 1 - i ;
int ej = N - 1 - j ;
int temp = mat [ i , j ] ; mat [ i , j ] = mat [ ej , i ] ; mat [ ej , i ] = mat [ ei , ej ] ; mat [ ei , ej ] = mat [ j , ei ] ; mat [ j , ei ] = temp ; }
static void rotate ( int [ , ] mat , int N , int K ) {
K = K % 4 ;
while ( K -- > 0 ) {
for ( int i = 0 ; i < N / 2 ; i ++ ) {
for ( int j = i ; j < N - i - 1 ; j ++ ) {
if ( i != j && ( i + j ) != N - 1 ) {
performSwap ( mat , i , j ) ; } } } }
print ( mat ) ; }
public static void Main ( string [ ] args ) { int K = 5 ; int [ , ] mat = { { 1 , 2 , 3 , 4 } , { 6 , 7 , 8 , 9 } , { 11 , 12 , 13 , 14 } , { 16 , 17 , 18 , 19 } , } ; int N = mat . GetLength ( 0 ) ; rotate ( mat , N , K ) ; } }
using System ; class GFG {
static int findRotations ( String str ) {
String tmp = str + str ; int n = str . Length ; for ( int i = 1 ; i <= n ; i ++ ) {
String substring = tmp . Substring ( i , str . Length ) ;
if ( str == substring ) return i ; } return n ; }
public static void Main ( ) { String str = " abc " ; Console . Write ( findRotations ( str ) ) ; } }
using System ; class GFG { static int MAX = 10000 ;
static int [ ] prefix = new int [ MAX + 1 ] ; static bool isPowerOfTwo ( int x ) { if ( x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ) return true ; return false ; }
static void computePrefix ( int n , int [ ] a ) {
if ( isPowerOfTwo ( a [ 0 ] ) ) prefix [ 0 ] = 1 ; for ( int i = 1 ; i < n ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] ; if ( isPowerOfTwo ( a [ i ] ) ) prefix [ i ] ++ ; } }
static int query ( int L , int R ) { if ( L == 0 ) return prefix [ R ] ; return prefix [ R ] - prefix [ L - 1 ] ; }
public static void Main ( ) { int [ ] A = { 3 , 8 , 5 , 2 , 5 , 10 } ; int N = A . Length ; computePrefix ( N , A ) ; Console . WriteLine ( query ( 0 , 4 ) ) ; Console . WriteLine ( query ( 3 , 5 ) ) ; } }
using System ; class GFG {
static void countIntgralPoints ( int x1 , int y1 , int x2 , int y2 ) { Console . WriteLine ( ( y2 - y1 - 1 ) * ( x2 - x1 - 1 ) ) ; }
static void Main ( ) { int x1 = 1 , y1 = 1 ; int x2 = 4 , y2 = 4 ; countIntgralPoints ( x1 , y1 , x2 , y2 ) ; } }
using System ; class GFG {
static void findNextNumber ( int n ) { int [ ] h = new int [ 10 ] ; int i = 0 , msb = n , rem = 0 ; int next_num = - 1 , count = 0 ;
while ( msb > 9 ) { rem = msb % 10 ; h [ rem ] = 1 ; msb /= 10 ; count ++ ; } h [ msb ] = 1 ; count ++ ;
for ( i = msb + 1 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; break ; } }
if ( next_num == - 1 ) { for ( i = 1 ; i < msb ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; count ++ ; break ; } } }
if ( next_num > 0 ) {
for ( i = 0 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { msb = i ; break ; } }
for ( i = 1 ; i < count ; i ++ ) { next_num = ( ( next_num * 10 ) + msb ) ; }
if ( next_num > n ) Console . WriteLine ( next_num ) ; else Console . WriteLine ( " Not ▁ Possible " ) ; } else { Console . WriteLine ( " Not ▁ Possible " ) ; } }
public static void Main ( string [ ] args ) { int n = 2019 ; findNextNumber ( n ) ; } }
using System ; class GFG {
static void CalculateValues ( int N ) { int A = 0 , B = 0 , C = 0 ;
for ( C = 0 ; C < N / 7 ; C ++ ) {
for ( B = 0 ; B < N / 5 ; B ++ ) {
A = N - 7 * C - 5 * B ;
if ( A >= 0 && A % 3 == 0 ) { Console . Write ( " A ▁ = ▁ " + A / 3 + " , ▁ B ▁ = ▁ " + B + " , ▁ C ▁ = ▁ " + C ) ; return ; } } }
Console . WriteLine ( - 1 ) ; }
static public void Main ( ) { int N = 19 ; CalculateValues ( 19 ) ; } }
using System ; using System . Linq ; class GFG {
static void minimumTime ( int [ ] arr , int n ) {
int sum = 0 ;
int T = arr . Min ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
Console . WriteLine ( Math . Max ( 2 * T , sum ) ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 8 , 3 } ; int N = arr . Length ;
minimumTime ( arr , N ) ; } }
using System ; using System . Text ; class GFG {
static void lexicographicallyMax ( String s ) {
int n = s . Length ;
for ( int i = 0 ; i < n ; i ++ ) {
int count = 0 ;
int beg = i ;
int end = i ;
if ( s [ i ] == '1' ) count ++ ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( s [ j ] == '1' ) count ++ ; if ( count % 2 == 0 && count != 0 ) { end = j ; break ; } }
s = reverse ( s , beg , end + 1 ) ; }
Console . WriteLine ( s ) ; } static String reverse ( String s , int beg , int end ) { StringBuilder x = new StringBuilder ( " " ) ; for ( int i = 0 ; i < beg ; i ++ ) x . Append ( s [ i ] ) ; for ( int i = end - 1 ; i >= beg ; i -- ) x . Append ( s [ i ] ) ; for ( int i = end ; i < s . Length ; i ++ ) x . Append ( s [ i ] ) ; return x . ToString ( ) ; }
public static void Main ( String [ ] args ) { String S = "0101" ; lexicographicallyMax ( S ) ; } }
using System ; class GFG {
public static void maxPairs ( int [ ] nums , int k ) {
Array . Sort ( nums ) ;
int result = 0 ;
int start = 0 , end = nums . Length - 1 ;
while ( start < end ) { if ( nums [ start ] + nums [ end ] > k )
end -- ; else if ( nums [ start ] + nums [ end ] < k )
start ++ ;
else { start ++ ; end -- ; result ++ ; } }
Console . Write ( result ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 } ; int K = 5 ;
maxPairs ( arr , K ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static void maxPairs ( int [ ] nums , int k ) {
Dictionary < int , int > map = new Dictionary < int , int > ( ) ;
int result = 0 ;
foreach ( int i in nums ) {
if ( map . ContainsKey ( i ) && map [ i ] > 0 ) { map [ i ] = map [ i ] - 1 ; result ++ ; }
else { if ( ! map . ContainsKey ( k - i ) ) map . Add ( k - i , 1 ) ; else map [ i ] = map [ i ] + 1 ; } }
Console . WriteLine ( result ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 } ; int K = 5 ;
maxPairs ( arr , K ) ; } }
using System ; class GFG {
static void removeIndicesToMakeSumEqual ( int [ ] arr ) {
int N = arr . Length ;
int [ ] odd = new int [ N ] ;
int [ ] even = new int [ N ] ;
even [ 0 ] = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) {
odd [ i ] = odd [ i - 1 ] ;
even [ i ] = even [ i - 1 ] ;
if ( i % 2 == 0 ) {
even [ i ] += arr [ i ] ; }
else {
odd [ i ] += arr [ i ] ; } }
bool find = false ;
int p = odd [ N - 1 ] ;
int q = even [ N - 1 ] - arr [ 0 ] ;
if ( p == q ) { Console . Write ( "0 ▁ " ) ; find = true ; }
for ( int i = 1 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) {
p = even [ N - 1 ] - even [ i - 1 ] - arr [ i ] + odd [ i - 1 ] ;
q = odd [ N - 1 ] - odd [ i - 1 ] + even [ i - 1 ] ; } else {
q = odd [ N - 1 ] - odd [ i - 1 ] - arr [ i ] + even [ i - 1 ] ;
p = even [ N - 1 ] - even [ i - 1 ] + odd [ i - 1 ] ; }
if ( p == q ) {
find = true ;
Console . Write ( i + " ▁ " ) ; } }
if ( ! find ) {
Console . Write ( - 1 ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 4 , 1 , 6 , 2 } ; removeIndicesToMakeSumEqual ( arr ) ; } }
using System ; class GFG {
static void min_element_removal ( int [ ] arr , int N ) {
int [ ] left = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) left [ i ] = 1 ;
int [ ] right = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) right [ i ] = 1 ;
for ( int i = 1 ; i < N ; i ++ ) {
for ( int j = 0 ; j < i ; j ++ ) {
if ( arr [ j ] < arr [ i ] ) {
left [ i ] = Math . Max ( left [ i ] , left [ j ] + 1 ) ; } } }
for ( int i = N - 2 ; i >= 0 ; i -- ) {
for ( int j = N - 1 ; j > i ; j -- ) {
if ( arr [ i ] > arr [ j ] ) {
right [ i ] = Math . Max ( right [ i ] , right [ j ] + 1 ) ; } } }
int maxLen = 0 ;
for ( int i = 1 ; i < N - 1 ; i ++ ) {
maxLen = Math . Max ( maxLen , left [ i ] + right [ i ] - 1 ) ; } Console . WriteLine ( N - maxLen ) ; }
static void makeBitonic ( int [ ] arr , int N ) { if ( N == 1 ) { Console . WriteLine ( "0" ) ; return ; } if ( N == 2 ) { if ( arr [ 0 ] != arr [ 1 ] ) Console . WriteLine ( "0" ) ; else Console . WriteLine ( "1" ) ; return ; } min_element_removal ( arr , N ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 1 , 1 , 5 , 6 , 2 , 3 , 1 } ; int N = arr . Length ; makeBitonic ( arr , N ) ; } }
using System ; class GFG {
static void countSubarrays ( int [ ] A , int N ) {
int ans = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( A [ i ] != A [ i + 1 ] ) {
ans ++ ;
for ( int j = i - 1 , k = i + 2 ; j >= 0 && k < N && A [ j ] == A [ i ] && A [ k ] == A [ i + 1 ] ; j -- , k ++ ) {
ans ++ ; } } }
Console . Write ( ans + " STRNEWLINE " ) ; }
public static void Main ( ) { int [ ] A = { 1 , 1 , 0 , 0 , 1 , 0 } ; int N = A . Length ;
countSubarrays ( A , N ) ; } }
using System ; class GFG { static int maxN = 2002 ;
static int [ , ] lcount = new int [ maxN , maxN ] ;
static int [ , ] rcount = new int [ maxN , maxN ] ;
static void fill_counts ( int [ ] a , int n ) { int i , j ;
int maxA = a [ 0 ] ; for ( i = 0 ; i < n ; i ++ ) { if ( a [ i ] > maxA ) { maxA = a [ i ] ; } } for ( i = 0 ; i < n ; i ++ ) { lcount [ a [ i ] , i ] = 1 ; rcount [ a [ i ] , i ] = 1 ; } for ( i = 0 ; i <= maxA ; i ++ ) {
for ( j = 1 ; j < n ; j ++ ) { lcount [ i , j ] = lcount [ i , j - 1 ] + lcount [ i , j ] ; }
for ( j = n - 2 ; j >= 0 ; j -- ) { rcount [ i , j ] = rcount [ i , j + 1 ] + rcount [ i , j ] ; } } }
static int countSubsequence ( int [ ] a , int n ) { int i , j ; fill_counts ( a , n ) ; int answer = 0 ; for ( i = 1 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n - 1 ; j ++ ) { answer += lcount [ a [ j ] , i - 1 ] * rcount [ a [ i ] , j + 1 ] ; } } return answer ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 2 , 3 , 2 , 1 , 3 , 2 } ; Console . Write ( countSubsequence ( a , a . Length ) ) ; } }
using System ; class GFG {
static string removeOuterParentheses ( string S ) {
string res = " " ;
int count = 0 ;
for ( int c = 0 ; c < S . Length ; c ++ ) {
if ( S == ' ( ' && count ++ > 0 )
res += S ;
if ( S == ' ) ' && count -- > 1 )
res += S ; }
return res ; }
public static void Main ( ) { string S = " ( ( ) ( ) ) ( ( ) ) ( ) " ; Console . Write ( removeOuterParentheses ( S ) ) ; } }
using System ; class GFG {
public static int maxiConsecutiveSubarray ( int [ ] arr , int N ) {
int maxi = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) {
int cnt = 1 , j ; for ( j = i ; j < N - 1 ; j ++ ) {
if ( arr [ j + 1 ] == arr [ j ] + 1 ) { cnt ++ ; }
else { break ; } }
maxi = Math . Max ( maxi , cnt ) ; i = j ; }
return maxi ; }
public static void Main ( String [ ] args ) { int N = 11 ; int [ ] arr = { 1 , 3 , 4 , 2 , 3 , 4 , 2 , 3 , 5 , 6 , 7 } ; Console . WriteLine ( maxiConsecutiveSubarray ( arr , N ) ) ; } }
using System . Collections . Generic ; using System ; class GFG { static int N = 100005 ;
static void SieveOfEratosthenes ( bool [ ] prime , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
static int digitSum ( int number ) {
int sum = 0 ; while ( number > 0 ) {
sum += ( number % 10 ) ; number /= 10 ; }
return sum ; }
static void longestCompositeDigitSumSubsequence ( int [ ] arr , int n ) { int count = 0 ; bool [ ] prime = new bool [ N + 1 ] ; for ( int i = 0 ; i <= N ; i ++ ) prime [ i ] = true ; SieveOfEratosthenes ( prime , N ) ; for ( int i = 0 ; i < n ; i ++ ) {
int res = digitSum ( arr [ i ] ) ;
if ( res == 1 ) { continue ; }
if ( prime [ res ] == false ) { count ++ ; } } Console . WriteLine ( count ) ; }
public static void Main ( ) { int [ ] arr = { 13 , 55 , 7 , 3 , 5 , 1 , 10 , 21 , 233 , 144 , 89 } ; int n = arr . Length ;
longestCompositeDigitSumSubsequence ( arr , n ) ; } }
using System ; class GFG { static int sum ;
class Node { public int data ; public Node left , right ; } ;
static Node newnode ( int data ) { Node temp = new Node ( ) ; temp . data = data ; temp . left = null ; temp . right = null ;
return temp ; }
static Node insert ( String s , int i , int N , Node root , Node temp ) { if ( i == N ) return temp ;
if ( s [ i ] == ' L ' ) root . left = insert ( s , i + 1 , N , root . left , temp ) ;
else root . = insert ( s , i + 1 , N , root . right , temp ) ;
return root ; }
static int SBTUtil ( Node root ) {
if ( root == null ) return 0 ; if ( root . left == null && root . right == null ) return root . data ;
int left = SBTUtil ( root . left ) ;
int right = SBTUtil ( root . right ) ;
if ( root . left != null && root . right != null ) {
if ( ( left % 2 == 0 && right % 2 != 0 ) || ( left % 2 != 0 && right % 2 == 0 ) ) { sum += root . data ; } }
return left + right + root . data ; }
static Node build_tree ( int R , int N , String [ ] str , int [ ] values ) {
Node root = newnode ( R ) ; int i ;
for ( i = 0 ; i < N - 1 ; i ++ ) { String s = str [ i ] ; int x = values [ i ] ;
Node temp = newnode ( x ) ;
root = insert ( s , 0 , s . Length , root , temp ) ; }
return root ; }
static void speciallyBalancedNodes ( int R , int N , String [ ] str , int [ ] values ) {
Node root = build_tree ( R , N , str , values ) ;
sum = 0 ;
SBTUtil ( root ) ;
Console . Write ( sum + " ▁ " ) ; }
public static void Main ( String [ ] args ) {
int N = 7 ;
int R = 12 ;
String [ ] str = { " L " , " R " , " RL " , " RR " , " RLL " , " RLR " } ;
int [ ] values = { 17 , 16 , 4 , 9 , 2 , 3 } ;
speciallyBalancedNodes ( R , N , str , values ) ; } }
using System ; class GFG {
static void position ( int [ , ] arr , int N ) {
int pos = - 1 ;
int count ;
for ( int i = 0 ; i < N ; i ++ ) {
count = 0 ; for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i , 0 ] <= arr [ j , 0 ] && arr [ i , 1 ] >= arr [ j , 1 ] ) { count ++ ; } }
if ( count == N ) { pos = i ; } }
if ( pos == - 1 ) { Console . Write ( pos ) ; }
else { Console . Write ( pos + 1 ) ; } }
public static void Main ( ) {
int [ , ] arr = { { 3 , 3 } , { 1 , 3 } , { 2 , 2 } , { 2 , 3 } , { 1 , 2 } } ; int N = arr . GetLength ( 0 ) ;
position ( arr , N ) ; } }
using System ; class GFG {
static void position ( int [ , ] arr , int N ) {
int pos = - 1 ;
int right = int . MinValue ;
int left = int . MaxValue ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i , 1 ] > right ) { right = arr [ i , 1 ] ; }
if ( arr [ i , 0 ] < left ) { left = arr [ i , 0 ] ; } }
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i , 0 ] == left && arr [ i , 1 ] == right ) { pos = i + 1 ; } }
Console . Write ( pos + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) {
int [ , ] arr = { { 3 , 3 } , { 1 , 3 } , { 2 , 2 } , { 2 , 3 } , { 1 , 2 } } ; int N = arr . GetLength ( 0 ) ;
position ( arr , N ) ; } }
using System ; class GFG {
static int ctMinEdits ( string str1 , string str2 ) { int N1 = str1 . Length ; int N2 = str2 . Length ;
int [ ] freq1 = new int [ 256 ] ; freq1 [ 0 ] = str1 [ 0 ] ; for ( int i = 0 ; i < N1 ; i ++ ) { freq1 [ str1 [ i ] ] ++ ; }
int [ ] freq2 = new int [ 256 ] ; freq2 [ 0 ] = str2 [ 0 ] ; for ( int i = 0 ; i < N2 ; i ++ ) { freq2 [ str2 [ i ] ] ++ ; }
for ( int i = 0 ; i < 256 ; i ++ ) {
if ( freq1 [ i ] > freq2 [ i ] ) { freq1 [ i ] = freq1 [ i ] - freq2 [ i ] ; freq2 [ i ] = 0 ; }
else { freq2 [ i ] = freq2 [ i ] - freq1 [ i ] ; freq1 [ i ] = 0 ; } }
int sum1 = 0 ;
int sum2 = 0 ; for ( int i = 0 ; i < 256 ; i ++ ) { sum1 += freq1 [ i ] ; sum2 += freq2 [ i ] ; } return Math . Max ( sum1 , sum2 ) ; }
public static void Main ( ) { string str1 = " geeksforgeeks " ; string str2 = " geeksforcoder " ; Console . WriteLine ( ctMinEdits ( str1 , str2 ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void CountPairs ( int [ ] a , int [ ] b , int n ) {
int [ ] C = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) { C [ i ] = a [ i ] + b [ i ] ; }
Dictionary < int , int > freqCount = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( ! freqCount . ContainsKey ( C [ i ] ) ) freqCount . Add ( C [ i ] , 1 ) ; else freqCount [ C [ i ] ] = freqCount [ C [ i ] ] + 1 ; }
int NoOfPairs = 0 ; foreach ( KeyValuePair < int , int > x in freqCount ) { int y = x . Value ;
NoOfPairs = NoOfPairs + y * ( y - 1 ) / 2 ; }
Console . WriteLine ( NoOfPairs ) ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 1 , 4 , 20 , 3 , 10 , 5 } ; int [ ] brr = { 9 , 6 , 1 , 7 , 11 , 6 } ;
int N = arr . Length ;
CountPairs ( arr , brr , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void medianChange ( List < int > arr1 , List < int > arr2 ) { int N = arr1 . Count ;
List < double > median = new List < double > ( ) ;
if ( ( N & 1 ) != 0 ) { median . Add ( arr1 [ N / 2 ] * 1.0 ) ; }
else { median . Add ( ( arr1 [ N / 2 ] + arr1 [ ( N - 1 ) / 2 ] ) / 2.0 ) ; } foreach ( int x in arr2 ) {
int it = arr1 . IndexOf ( x ) ;
arr1 . RemoveAt ( it ) ;
N -- ;
if ( ( N & 1 ) != 0 ) { median . Add ( arr1 [ N / 2 ] * 1.0 ) ; }
else { median . Add ( ( arr1 [ N / 2 ] + arr1 [ ( N - 1 ) / 2 ] ) / 2.0 ) ; } }
for ( int i = 0 ; i < median . Count - 1 ; i ++ ) { Console . Write ( median [ i + 1 ] - median [ i ] + " ▁ " ) ; } }
static void Main ( ) {
List < int > arr1 = new List < int > ( new int [ ] { 2 , 4 , 6 , 8 , 10 } ) ; List < int > arr2 = new List < int > ( new int [ ] { 4 , 6 } ) ;
medianChange ( arr1 , arr2 ) ; } }
using System ; class GFG {
static int nfa = 1 ;
static int flag = 0 ;
static void state1 ( char c ) {
if ( c == ' a ' ) nfa = 2 ; else if ( c == ' b ' c == ' c ' ) nfa = 1 ; else flag = 1 ; }
static void state2 ( char c ) {
if ( c == ' a ' ) nfa = 3 ; else if ( c == ' b ' c == ' c ' ) nfa = 2 ; else flag = 1 ; }
static void state3 ( char c ) {
if ( c == ' a ' ) nfa = 1 ; else if ( c == ' b ' c == ' c ' ) nfa = 3 ; else flag = 1 ; }
static void state4 ( char c ) {
if ( c == ' b ' ) nfa = 5 ; else if ( c == ' a ' c == ' c ' ) nfa = 4 ; else flag = 1 ; }
static void state5 ( char c ) {
if ( c == ' b ' ) nfa = 6 ; else if ( c == ' a ' c == ' c ' ) nfa = 5 ; else flag = 1 ; }
static void state6 ( char c ) {
if ( c == ' b ' ) nfa = 4 ; else if ( c == ' a ' c == ' c ' ) nfa = 6 ; else flag = 1 ; }
static void state7 ( char c ) {
if ( c == ' c ' ) nfa = 8 ; else if ( c == ' b ' c == ' a ' ) nfa = 7 ; else flag = 1 ; }
static void state8 ( char c ) {
if ( c == ' c ' ) nfa = 9 ; else if ( c == ' b ' c == ' a ' ) nfa = 8 ; else flag = 1 ; }
static void state9 ( char c ) {
if ( c == ' c ' ) nfa = 7 ; else if ( c == ' b ' c == ' a ' ) nfa = 9 ; else flag = 1 ; }
static bool checkA ( String s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 1 ) state1 ( s [ i ] ) ; else if ( nfa == 2 ) state2 ( s [ i ] ) ; else if ( nfa == 3 ) state3 ( s [ i ] ) ; } if ( nfa == 1 ) { return true ; } else { nfa = 4 ; } return false ; }
static bool checkB ( String s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 4 ) state4 ( s [ i ] ) ; else if ( nfa == 5 ) state5 ( s [ i ] ) ; else if ( nfa == 6 ) state6 ( s [ i ] ) ; } if ( nfa == 4 ) { return true ; } else { nfa = 7 ; } return false ; }
static bool checkC ( String s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 7 ) state7 ( s [ i ] ) ; else if ( nfa == 8 ) state8 ( s [ i ] ) ; else if ( nfa == 9 ) state9 ( s [ i ] ) ; } if ( nfa == 7 ) { return true ; } return false ; }
public static void Main ( String [ ] args ) { String s = " bbbca " ; int x = 5 ;
if ( checkA ( s , x ) || checkB ( s , x ) || checkC ( s , x ) ) { Console . WriteLine ( " ACCEPTED " ) ; } else { if ( flag == 0 ) { Console . WriteLine ( " NOT ▁ ACCEPTED " ) ; } else { Console . WriteLine ( " INPUT ▁ OUT ▁ OF ▁ DICTIONARY . " ) ; } } } }
using System ; class GFG {
static int getPositionCount ( int [ ] a , int n ) {
int count = 1 ;
int min = a [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) {
if ( a [ i ] <= min ) {
min = a [ i ] ;
count ++ ; } } return count ; }
public static void Main ( ) { int [ ] a = { 5 , 4 , 6 , 1 , 3 , 1 } ; int n = a . Length ; Console . WriteLine ( getPositionCount ( a , n ) ) ; } }
using System ; class GFG {
static int maxSum ( int [ ] arr , int n , int k ) {
if ( n < k ) { return - 1 ; }
int res = 0 ; for ( int i = 0 ; i < k ; i ++ ) res += arr [ i ] ;
int curr_sum = res ; for ( int i = k ; i < n ; i ++ ) { curr_sum += arr [ i ] - arr [ i - k ] ; res = Math . Max ( res , curr_sum ) ; } return res ; }
static int solve ( int [ ] arr , int n , int k ) { int max_len = 0 , l = 0 , r = n , m ;
while ( l <= r ) { m = ( l + r ) / 2 ;
if ( maxSum ( arr , n , m ) > k ) r = m - 1 ; else { l = m + 1 ;
max_len = m ; } } return max_len ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . Length ; int k = 10 ; Console . WriteLine ( solve ( arr , n , k ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 100001 ; static int ROW = 10 ; static int COl = 3 ; static List < int > [ ] indices = new List < int > [ MAX ] ;
static int [ , ] test = { { 2 , 3 , 6 } , { 2 , 4 , 4 } , { 2 , 6 , 3 } , { 3 , 2 , 6 } , { 3 , 3 , 3 } , { 3 , 6 , 2 } , { 4 , 2 , 4 } , { 4 , 4 , 2 } , { 6 , 2 , 3 } , { 6 , 3 , 2 } } ;
static int find_triplet ( int [ ] array , int n ) { int answer = 0 ; for ( int i = 0 ; i < MAX ; i ++ ) { indices [ i ] = new List < int > ( ) ; }
for ( int i = 0 ; i < n ; i ++ ) { indices [ array [ i ] ] . Add ( i ) ; } for ( int i = 0 ; i < n ; i ++ ) { int y = array [ i ] ; for ( int j = 0 ; j < ROW ; j ++ ) { int s = test [ j , 1 ] * y ;
if ( s % test [ j , 0 ] != 0 ) continue ; if ( s % test [ j , 2 ] != 0 ) continue ; int x = s / test [ j , 0 ] ; int z = s / test [ j , 2 ] ; if ( x > MAX z > MAX ) continue ; int l = 0 ; int r = indices [ x ] . Count - 1 ; int first = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( indices [ x ] [ m ] < i ) { first = m ; l = m + 1 ; } else { r = m - 1 ; } } l = 0 ; r = indices [ z ] . Count - 1 ; int third = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( indices [ z ] [ m ] > i ) { third = m ; r = m - 1 ; } else { l = m + 1 ; } } if ( first != - 1 && third != - 1 ) {
answer += ( first + 1 ) * ( indices [ z ] . Count - third ) ; } } } return answer ; }
public static void Main ( String [ ] args ) { int [ ] array = { 2 , 4 , 5 , 6 , 7 } ; int n = array . Length ; Console . WriteLine ( find_triplet ( array , n ) ) ; } }
using System ; class GFG { static int distinct ( int [ ] arr , int n ) { int count = 0 ;
if ( n == 1 ) return 1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( i == 0 ) { if ( arr [ i ] != arr [ i + 1 ] ) count += 1 ; }
else { if ( arr [ i ] != arr [ i + 1 ] arr [ i ] != arr [ i - 1 ] ) count += 1 ; } }
if ( arr [ n - 1 ] != arr [ n - 2 ] ) count += 1 ; return count ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 0 , 0 , 0 , 0 , 0 , 1 , 0 } ; int n = arr . Length ; Console . WriteLine ( distinct ( arr , n ) ) ; } }
using System ; class GFG {
static bool isSorted ( int [ , ] arr , int N ) {
for ( int i = 1 ; i < N ; i ++ ) { if ( arr [ i , 0 ] > arr [ i - 1 , 0 ] ) { return false ; } }
return true ; }
static string isPossibleToSort ( int [ , ] arr , int N ) {
int group = arr [ 0 , 1 ] ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( arr [ i , 1 ] != group ) { return " Yes " ; } }
if ( isSorted ( arr , N ) ) { return " Yes " ; } else { return " No " ; } }
public static void Main ( ) { int [ , ] arr = { { 340000 , 2 } , { 45000 , 1 } , { 30000 , 2 } , { 50000 , 4 } } ; int N = arr . GetLength ( 0 ) ; Console . WriteLine ( isPossibleToSort ( arr , N ) ) ; } }
using System ;
class Node { public Node left , right ; public int data ; public Node ( int data ) { this . data = data ; left = null ; right = null ; } } class AlphaScore { Node root ; AlphaScore ( ) { root = null ; } static long sum = 0 , total_sum = 0 ; static long mod = 1000000007 ;
static long getAlphaScore ( Node node ) {
if ( node . left != null ) getAlphaScore ( node . left ) ;
sum = ( sum + node . data ) % mod ;
total_sum = ( total_sum + sum ) % mod ;
if ( node . right != null ) getAlphaScore ( node . right ) ;
return total_sum ; }
static Node constructBST ( int [ ] arr , int start , int end , Node root ) { if ( start > end ) return null ; int mid = ( start + end ) / 2 ;
if ( root == null ) root = new Node ( arr [ mid ] ) ;
root . left = constructBST ( arr , start , mid - 1 , root . left ) ;
root . right = constructBST ( arr , mid + 1 , end , root . right ) ;
return root ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 10 , 11 , 12 } ; int length = arr . Length ;
Array . Sort ( arr ) ; Node root = null ;
root = constructBST ( arr , 0 , length - 1 , root ) ; Console . WriteLine ( getAlphaScore ( root ) ) ; } }
using System ; class GFG {
static int sortByFreq ( int [ ] arr , int n ) {
int maxE = - 1 ;
for ( int i = 0 ; i < n ; i ++ ) { maxE = Math . Max ( maxE , arr [ i ] ) ; }
int [ ] freq = new int [ maxE + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; }
int cnt = 0 ;
for ( int i = 0 ; i <= maxE ; i ++ ) {
if ( freq [ i ] > 0 ) { int value = 100000 - i ; arr [ cnt ] = 100000 * freq [ i ] + value ; cnt ++ ; } }
return cnt ; }
static void printSortedArray ( int [ ] arr , int cnt ) {
for ( int i = 0 ; i < cnt ; i ++ ) {
int frequency = arr [ i ] / 100000 ;
int value = 100000 - ( arr [ i ] % 100000 ) ;
for ( int j = 0 ; j < frequency ; j ++ ) { Console . Write ( value + " ▁ " ) ; } } }
public static void Main ( ) { int [ ] arr = { 4 , 4 , 5 , 6 , 4 , 2 , 2 , 8 , 5 } ;
int n = arr . Length ;
int cnt = sortByFreq ( arr , n ) ;
Array . Sort ( arr ) ; Array . Reverse ( arr ) ;
printSortedArray ( arr , cnt ) ; } }
using System ; class GFG {
static bool checkRectangles ( int [ ] arr , int n ) { bool ans = true ;
Array . Sort ( arr ) ;
int area = arr [ 0 ] * arr [ 4 * n - 1 ] ;
for ( int i = 0 ; i < 2 * n ; i = i + 2 ) { if ( arr [ i ] != arr [ i + 1 ] arr [ 4 * n - i - 1 ] != arr [ 4 * n - i - 2 ] arr [ i ] * arr [ 4 * n - i - 1 ] != area ) {
ans = false ; break ; } }
if ( ans ) return true ; return false ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 8 , 2 , 1 , 2 , 4 , 4 , 8 } ; int n = 2 ; if ( checkRectangles ( arr , n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static int cntElements ( int [ ] arr , int n ) {
int [ ] copy_arr = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) copy_arr [ i ] = arr [ i ] ;
int count = 0 ;
Array . Sort ( arr ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != copy_arr [ i ] ) { count ++ ; } } return count ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 6 , 2 , 4 , 5 } ; int n = arr . Length ; Console . WriteLine ( cntElements ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { public class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static void findPairs ( int [ ] arr , int n , int k , int d ) {
if ( n < 2 * k ) { Console . Write ( - 1 ) ; return ; }
List < pair > pairs = new List < pair > ( ) ;
Array . Sort ( arr ) ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( arr [ n - k + i ] - arr [ i ] >= d ) {
pair p = new pair ( arr [ i ] , arr [ n - k + i ] ) ; pairs . Add ( p ) ; } }
if ( pairs . Count < k ) { Console . Write ( - 1 ) ; return ; }
foreach ( pair v in pairs ) { Console . WriteLine ( " ( " + v . first + " , ▁ " + v . second + " ) " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 4 , 6 , 10 , 23 , 14 , 7 , 2 , 20 , 9 } ; int n = arr . Length ; int k = 4 , d = 3 ; findPairs ( arr , n , k , d ) ; } }
using System ; class GFG {
static int pairs_count ( int [ ] arr , int n , int sum ) {
int ans = 0 ;
Array . Sort ( arr ) ;
int i = 0 , j = n - 1 ; while ( i < j ) {
if ( arr [ i ] + arr [ j ] < sum ) i ++ ;
else if ( arr [ i ] + arr [ j ] > sum ) -- ;
else {
int x = arr [ i ] , xx = i ; while ( ( i < j ) && ( arr [ i ] == x ) ) i ++ ;
int y = arr [ j ] , yy = j ; while ( ( j >= i ) && ( arr [ j ] == y ) ) j -- ;
if ( x == y ) { int temp = i - xx + yy - j - 1 ; ans += ( temp * ( temp + 1 ) ) / 2 ; } else ans += ( i - xx ) * ( yy - j ) ; } }
return ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 5 , 7 , 5 , - 1 } ; int n = arr . Length ; int sum = 6 ; Console . WriteLine ( pairs_count ( arr , n , sum ) ) ; } }
using System ; class GFG { static bool check ( string str ) { int min = Int32 . MaxValue ; int max = Int32 . MinValue ; int sum = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) {
int ascii = ( int ) str [ i ] ;
if ( ascii < 96 ascii > 122 ) return false ;
sum += ascii ;
if ( min > ascii ) min = ascii ;
if ( max < ascii ) max = ascii ; }
min -= 1 ;
int eSum = ( ( max * ( max + 1 ) ) / 2 ) - ( ( min * ( min + 1 ) ) / 2 ) ;
return sum == eSum ; }
static void Main ( ) {
string str = " dcef " ; if ( check ( str ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ;
string str1 = " xyza " ; if ( check ( str1 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; using System . Linq ; using System . Collections . Generic ; class GFG {
static int findKth ( int [ ] arr , int n , int k ) { HashSet < int > missing = new HashSet < int > ( ) ; int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { missing . Add ( arr [ i ] ) ; }
int maxm = arr . Max ( ) ; int minm = arr . Min ( ) ;
for ( int i = minm + 1 ; i < maxm ; i ++ ) {
if ( ! missing . Contains ( i ) ) { count ++ ; }
if ( count == k ) { return i ; } }
return - 1 ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 10 , 9 , 4 } ; int n = arr . Length ; int k = 5 ; Console . WriteLine ( findKth ( arr , n , k ) ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node next ; } ; static ;
static void sortList ( Node head ) { int startVal = 1 ; while ( head != null ) { head . data = startVal ; startVal ++ ; head = head . next ; } }
static void push ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = head_ref ;
head_ref = new_node ; start = head_ref ; }
static void printList ( Node node ) { while ( node != null ) { Console . Write ( node . data + " ▁ " ) ; node = node . next ; } }
public static void Main ( String [ ] args ) { start = null ;
push ( start , 2 ) ; push ( start , 1 ) ; push ( start , 6 ) ; push ( start , 4 ) ; push ( start , 5 ) ; push ( start , 3 ) ; sortList ( start ) ; printList ( start ) ; } }
using System ; class GfG {
public class Node { public int data ; public Node next ; }
static bool isSortedDesc ( Node head ) {
if ( head == null head . next == null ) return true ;
return ( head . data > head . next . data && isSortedDesc ( head . next ) ) ; } static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . next = null ; temp . data = data ; return temp ; }
public static void Main ( String [ ] args ) { Node head = newNode ( 7 ) ; head . next = newNode ( 5 ) ; head . next . next = newNode ( 4 ) ; head . next . next . next = newNode ( 3 ) ; if ( isSortedDesc ( head ) == true ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int minSum ( int [ ] arr , int n ) {
List < int > evenArr = new List < int > ( ) ; List < int > oddArr = new List < int > ( ) ; int i ;
Array . Sort ( arr ) ;
for ( i = 0 ; i < n ; i ++ ) { if ( i < n / 2 ) { oddArr . Add ( arr [ i ] ) ; } else { evenArr . Add ( arr [ i ] ) ; } }
evenArr . Sort ( ) ; evenArr . Reverse ( ) ;
int k = 0 , sum = 0 ; for ( int j = 0 ; j < evenArr . Count ; j ++ ) { arr [ k ++ ] = evenArr [ j ] ; arr [ k ++ ] = oddArr [ j ] ; sum += evenArr [ j ] * oddArr [ j ] ; } return sum ; }
public static void Main ( ) { int [ ] arr = { 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 } ; int n = arr . Length ; Console . WriteLine ( " Minimum ▁ required ▁ sum ▁ = ▁ " + minSum ( arr , n ) ) ; Console . WriteLine ( " Sorted ▁ array ▁ in ▁ " + " required ▁ format ▁ : ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } } }
using System ; using System . Collections . Generic ; class GFG {
static void minTime ( string word ) { int ans = 0 ;
int curr = 0 ; for ( int i = 0 ; i < word . Length ; i ++ ) {
int k = ( int ) word [ i ] - 97 ;
int a = Math . Abs ( curr - k ) ;
int b = 26 - Math . Abs ( curr - k ) ;
ans += Math . Min ( a , b ) ;
ans ++ ; curr = ( int ) word [ i ] - 97 ; }
Console . Write ( ans ) ; }
public static void Main ( ) {
string str = " zjpc " ;
minTime ( str ) ; } }
using System ; class GFG {
static int reduceToOne ( long N ) {
int cnt = 0 ; while ( N != 1 ) {
if ( N == 2 || ( N % 2 == 1 ) ) {
N = N - 1 ;
cnt ++ ; }
else if ( N % 2 == 0 ) {
N = N / ( N / 2 ) ;
cnt ++ ; } }
return cnt ; }
public static void Main ( ) { long N = 35 ; Console . WriteLine ( reduceToOne ( N ) ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG {
static void maxDiamonds ( int [ ] A , int N , int K ) {
var pq = new List < int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { pq . Add ( A [ i ] ) ; }
int ans = 0 ;
while ( pq . Count != 0 && K -- > 0 ) { pq . Sort ( ) ;
int top = pq [ pq . Count - 1 ] ;
pq . RemoveAt ( pq . Count - 1 ) ;
ans += top ;
top = top / 2 ; pq . Add ( top ) ; }
Console . WriteLine ( ans ) ; }
public static void Main ( string [ ] args ) { int [ ] A = { 2 , 1 , 7 , 4 , 2 } ; int K = 3 ; int N = A . Length ; maxDiamonds ( A , N , K ) ; } }
using System ; class GFG {
static int MinimumCost ( int [ ] A , int [ ] B , int N ) {
int totalCost = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int mod_A = B [ i ] % A [ i ] ; int totalCost_A = Math . Min ( mod_A , A [ i ] - mod_A ) ;
int mod_B = A [ i ] % B [ i ] ; int totalCost_B = Math . Min ( mod_B , B [ i ] - mod_B ) ;
totalCost += Math . Min ( totalCost_A , totalCost_B ) ; }
return totalCost ; }
public static void Main ( ) { int [ ] A = { 3 , 6 , 3 } ; int [ ] B = { 4 , 8 , 13 } ; int N = A . Length ; Console . Write ( MinimumCost ( A , B , N ) ) ; } }
using System ; public class GFG {
static void printLargestDivisible ( int [ ] arr , int N ) { int i , count0 = 0 , count7 = 0 ; for ( i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 0 ) count0 ++ ; else count7 ++ ; }
if ( count7 % 50 == 0 ) { while ( count7 != 0 ) { Console . Write ( 7 ) ; count7 -= 1 ; } while ( count0 != 0 ) { Console . Write ( 0 ) ; count0 -= 1 ; } }
else if ( count7 < 5 ) { if ( count0 == 0 ) Console . Write ( " No " ) ; else Console . Write ( "0" ) ; }
else {
count7 = count7 - count7 % 5 ; while ( count7 != 0 ) { Console . Write ( 7 ) ; count7 -= 1 ; } while ( count0 != 0 ) { Console . Write ( 0 ) ; count0 -= 1 ; } } }
public static void Main ( String [ ] args ) {
int [ ] arr = { 0 , 7 , 0 , 7 , 7 , 7 , 7 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 7 , 7 } ;
int N = arr . Length ; printLargestDivisible ( arr , N ) ; } }
using System ; class GFG {
static int findMaxValByRearrArr ( int [ ] arr , int N ) {
Array . Sort ( arr ) ;
int res = 0 ;
do {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += __gcd ( i + 1 , arr [ i ] ) ; }
res = Math . Max ( res , sum ) ; } while ( next_permutation ( arr ) ) ; return res ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; } static bool next_permutation ( int [ ] p ) { for ( int a = p . Length - 2 ; a >= 0 ; -- a ) if ( p [ a ] < p [ a + 1 ] ) for ( int b = p . Length - 1 ; ; -- b ) if ( p [ b ] > p [ a ] ) { int t = p [ a ] ; p [ a ] = p [ b ] ; p [ b ] = t ; for ( ++ a , b = p . Length - 1 ; a < b ; ++ a , -- b ) { t = p [ a ] ; p [ a ] = p [ b ] ; p [ b ] = t ; } return true ; } return false ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 2 , 1 } ; int N = arr . Length ; Console . Write ( findMaxValByRearrArr ( arr , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int min_elements ( int [ ] arr , int N ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( mp . ContainsKey ( arr [ i ] ) ) { mp [ arr [ i ] ] ++ ; } else { mp [ arr [ i ] ] = 1 ; } }
int cntMinRem = 0 ;
foreach ( KeyValuePair < int , int > it in mp ) {
int i = it . Key ;
if ( mp [ i ] < i ) {
cntMinRem += mp [ i ] ; }
else if ( mp [ i ] > i ) {
cntMinRem += ( mp [ i ] - i ) ; } } return cntMinRem ; }
static void Main ( ) { int [ ] arr = { 2 , 4 , 1 , 4 , 2 } ; int N = arr . Length ; Console . Write ( min_elements ( arr , N ) ) ; } }
using System ; class GFG {
static bool CheckAllarrayEqual ( int [ ] arr , int N ) {
if ( N == 1 ) { return true ; }
int totalSum = arr [ 0 ] ;
int secMax = Int32 . MinValue ;
int Max = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) { if ( arr [ i ] >= Max ) {
secMax = Max ;
Max = arr [ i ] ; } else if ( arr [ i ] > secMax ) {
secMax = arr [ i ] ; }
totalSum += arr [ i ] ; }
if ( ( secMax * ( N - 1 ) ) > totalSum ) { return false ; }
if ( totalSum % ( N - 1 ) != 0 ) { return false ; } return true ; }
public static void Main ( ) { int [ ] arr = { 6 , 2 , 2 , 2 } ; int N = arr . Length ; if ( CheckAllarrayEqual ( arr , N ) ) { Console . Write ( " YES " ) ; } else { Console . Write ( " NO " ) ; } } }
using System ; class GFG {
static void Remove_one_element ( int [ ] arr , int n ) {
int post_odd = 0 , post_even = 0 ;
int curr_odd = 0 , curr_even = 0 ;
int res = 0 ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( i % 2 != 0 ) post_odd ^= arr [ i ] ;
else post_even ^= [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( i % 2 != 0 ) post_odd ^= arr [ i ] ;
else post_even ^= [ i ] ;
int X = curr_odd ^ post_even ;
int Y = curr_even ^ post_odd ;
if ( X == Y ) res ++ ;
if ( i % 2 != 0 ) curr_odd ^= arr [ i ] ;
else curr_even ^= [ i ] ; }
Console . WriteLine ( res ) ; }
public static void Main ( ) {
int [ ] arr = { 1 , 0 , 1 , 0 , 1 } ;
int N = arr . Length ;
Remove_one_element ( arr , N ) ; } }
using System ; class GFG {
static int cntIndexesToMakeBalance ( int [ ] arr , int n ) {
if ( n == 1 ) { return 1 ; }
if ( n == 2 ) return 0 ;
int sumEven = 0 ;
int sumOdd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i % 2 == 0 ) {
sumEven += arr [ i ] ; }
else {
sumOdd += arr [ i ] ; } }
int currOdd = 0 ;
int currEven = arr [ 0 ] ;
int res = 0 ;
int newEvenSum = 0 ;
int newOddSum = 0 ;
for ( int i = 1 ; i < n - 1 ; i ++ ) {
if ( i % 2 != 0 ) {
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
public static void Main ( ) { int [ ] arr = { 1 , 1 , 1 } ; int n = arr . Length ; Console . WriteLine ( cntIndexesToMakeBalance ( arr , n ) ) ; } }
using System ; class GFG {
static void findNums ( int X , int Y ) {
int A , B ;
if ( X < Y ) { A = - 1 ; B = - 1 ; }
else if ( ( ( Math . Abs ( X - Y ) ) & 1 ) != 0 ) { A = - 1 ; B = - 1 ; }
else if ( X = = Y ) { A = 0 ; B = Y ; }
else {
A = ( X - Y ) / 2 ;
if ( ( A & Y ) == 0 ) {
B = ( A + Y ) ; }
else { A = - 1 ; B = - 1 ; } }
Console . Write ( A + " ▁ " + B ) ; }
public static void Main ( String [ ] args ) {
int X = 17 , Y = 13 ;
findNums ( X , Y ) ; } }
using System ; class GFG {
static void checkCount ( int [ ] A , int [ , ] Q , int q ) {
for ( int i = 0 ; i < q ; i ++ ) { int L = Q [ i , 0 ] ; int R = Q [ i , 1 ] ;
L -- ; R -- ;
if ( ( A [ L ] < A [ L + 1 ] ) != ( A [ R - 1 ] < A [ R ] ) ) { Console . WriteLine ( " Yes " ) ; } else { Console . WriteLine ( " No " ) ; } } }
public static void Main ( ) { int [ ] arr = { 11 , 13 , 12 , 14 } ; int [ , ] Q = { { 1 , 4 } , { 2 , 4 } } ; int q = Q . GetLength ( 0 ) ; checkCount ( arr , Q , q ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static double pairProductMean ( int [ ] arr , int N ) {
List < int > pairArray = new List < int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) { int pairProduct = arr [ i ] * arr [ j ] ;
pairArray . Add ( pairProduct ) ; } }
int length = pairArray . Count ;
float sum = 0 ; for ( int i = 0 ; i < length ; i ++ ) sum += pairArray [ i ] ;
float mean ;
if ( length != 0 ) mean = sum / length ; else mean = 0 ;
return mean ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 1 , 2 , 4 , 8 } ; int N = arr . Length ;
Console . WriteLine ( " { 0 : F2 } " , pairProductMean ( arr , N ) ) ; } }
using System ; class GFG {
static void findPlayer ( string [ ] str , int n ) {
int move_first = 0 ;
int move_sec = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str [ i ] [ 0 ] == str [ i ] [ str [ i ] . Length - 1 ] ) {
if ( ( str [ i ] [ 0 ] ) == 48 ) move_first ++ ; else move_sec ++ ; } }
if ( move_first <= move_sec ) { Console . Write ( " Player ▁ 2 ▁ wins " ) ; } else { Console . Write ( " Player ▁ 1 ▁ wins " ) ; } }
public static void Main ( ) {
string [ ] str = { "010" , "101" } ; int N = str . Length ;
findPlayer ( str , N ) ; } }
using System ; class GFG {
static int find_next ( int n , int k ) {
int M = n + 1 ; while ( true ) {
if ( ( M & ( 1L << k ) ) > 0 ) break ;
M ++ ; }
return M ; }
public static void Main ( String [ ] args ) {
int N = 15 , K = 2 ;
Console . Write ( find_next ( N , K ) ) ; } }
using System ; class GFG {
static int find_next ( int n , int k ) {
int ans = 0 ;
if ( ( n & ( 1L << k ) ) == 0 ) { int cur = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( ( n & ( 1L << i ) ) > 0 ) cur += ( int ) 1L << i ; }
ans = ( int ) ( n - cur + ( 1L << k ) ) ; }
else { int first_unset_bit = - 1 , cur = 0 ; for ( int i = 0 ; i < 64 ; i ++ ) {
if ( ( n & ( 1L << i ) ) == 0 ) { first_unset_bit = i ; break ; }
else cur += ( int ) ( 1L << i ) ; }
ans = ( int ) ( n - cur + ( 1L << first_unset_bit ) ) ;
if ( ( ans & ( 1L << k ) ) == 0 ) ans += ( int ) ( 1L << k ) ; }
return ans ; }
public static void Main ( String [ ] args ) { int N = 15 , K = 2 ;
Console . Write ( find_next ( N , K ) ) ; } }
using System ; class GFG { static String largestString ( String num , int k ) {
String ans = " " ; foreach ( char i in num . ToCharArray ( ) ) {
while ( ans . Length > 0 && ans [ ans . Length - 1 ] < i && k > 0 ) {
ans = ans . Substring ( 0 , ans . Length - 1 ) ;
k -- ; }
ans += i ; }
while ( ans . Length > 0 && k -- > 0 ) { ans = ans . Substring ( 0 , ans . Length - 1 ) ; }
return ans ; }
public static void Main ( String [ ] args ) { String str = " zyxedcba " ; int k = 1 ; Console . Write ( largestString ( str , k ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static void maxLengthSubArray ( int [ ] A , int N ) {
int [ ] forward = new int [ N ] ; int [ ] backward = new int [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) { if ( i == 0 A [ i ] != A [ i - 1 ] ) { forward [ i ] = 1 ; } else forward [ i ] = forward [ i - 1 ] + 1 ; }
for ( int i = N - 1 ; i >= 0 ; i -- ) { if ( i == N - 1 A [ i ] != A [ i + 1 ] ) { backward [ i ] = 1 ; } else backward [ i ] = backward [ i + 1 ] + 1 ; }
int ans = 0 ;
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( A [ i ] != A [ i + 1 ] ) ans = Math . Max ( ans , Math . Min ( forward [ i ] , backward [ i + 1 ] ) * 2 ) ; }
Console . WriteLine ( ans ) ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 1 , 2 , 3 , 4 , 4 , 4 , 6 , 6 , 6 , 9 } ;
int N = arr . Length ;
maxLengthSubArray ( arr , N ) ; } }
using System ; class GFG {
static void minNum ( int n ) { if ( n < 3 ) Console . WriteLine ( - 1 ) ; else Console . WriteLine ( 210 * ( ( int ) ( Math . Pow ( 10 , n - 1 ) / 210 ) + 1 ) ) ; }
public static void Main ( String [ ] args ) { int n = 5 ; minNum ( n ) ; } }
using System ; using System . Text ; using System . Collections ; class GFG {
static string helper ( int d , int s ) {
StringBuilder ans = new StringBuilder ( ) ; for ( int i = 0 ; i < d ; i ++ ) { ans . Append ( "0" ) ; } for ( int i = d - 1 ; i >= 0 ; i -- ) {
if ( s >= 9 ) { ans [ i ] = '9' ; s -= 9 ; }
else { char c = ( char ) ( s + ( int ) '0' ) ; ans [ i ] = c ; s = 0 ; } } return ans . ToString ( ) ; }
static string findMin ( int x , int Y ) {
string y = Y . ToString ( ) ; int n = y . Length ; ArrayList p = new ArrayList ( ) ; for ( int i = 0 ; i < n ; i ++ ) { p . Add ( 0 ) ; }
for ( int i = 0 ; i < n ; i ++ ) { p [ i ] = ( int ) ( ( int ) y [ i ] - ( int ) '0' ) ; if ( i > 0 ) { p [ i ] = ( int ) p [ i ] + ( int ) p [ i - 1 ] ; } }
for ( int i = n - 1 , k = 0 ; ; i -- , k ++ ) {
int d = 0 ; if ( i >= 0 ) { d = ( int ) y [ i ] - ( int ) '0' ; }
for ( int j = d + 1 ; j <= 9 ; j ++ ) { int r = j ;
if ( i > 0 ) { r += ( int ) p [ i - 1 ] ; }
if ( x - r >= 0 && x - r <= 9 * k ) {
string suf = helper ( k , x - r ) ; string pre = " " ; if ( i > 0 ) pre = y . Substring ( 0 , i ) ;
char cur = ( char ) ( j + ( int ) '0' ) ; pre += cur ;
return pre + suf ; } } } }
public static void Main ( string [ ] arg ) {
int x = 18 ; int y = 99 ;
Console . Write ( findMin ( x , y ) ) ; } }
using System ; class GFG {
public static void largestNumber ( int n , int X , int Y ) { int maxm = Math . Max ( X , Y ) ;
Y = X + Y - maxm ;
X = maxm ;
int Xs = 0 ; int Ys = 0 ; while ( n > 0 ) {
if ( n % Y == 0 ) {
Xs += n ;
n = 0 ; } else {
n -= X ;
Ys += X ; } }
if ( n == 0 ) { while ( Xs -- > 0 ) Console . Write ( X ) ; while ( Ys -- > 0 ) Console . Write ( Y ) ; }
else Console . ( " - 1" ) ; }
public static void Main ( String [ ] args ) { int n = 19 , X = 7 , Y = 5 ; largestNumber ( n , X , Y ) ; } }
using System ; class GFG { static int minChanges ( String str , int N ) { int res ; int count0 = 0 , count1 = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( str [ i ] == '0' ) count0 ++ ; } res = count0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( str [ i ] == '0' ) count0 -- ; if ( str [ i ] == '1' ) count1 ++ ; res = Math . Min ( res , count1 + count0 ) ; } return res ; }
public static void Main ( ) { int N = 9 ; String str = "000101001" ; Console . Write ( minChanges ( str , N ) ) ; } }
using System ; class GFG {
static int missingnumber ( int n , int [ ] arr ) { int mn = Int32 . MaxValue , mx = Int32 . MinValue ;
for ( int i = 0 ; i < n ; i ++ ) { if ( i > 0 && arr [ i ] == - 1 && arr [ i - 1 ] != - 1 ) { mn = Math . Min ( mn , arr [ i - 1 ] ) ; mx = Math . Max ( mx , arr [ i - 1 ] ) ; } if ( i < ( n - 1 ) && arr [ i ] == - 1 && arr [ i + 1 ] != - 1 ) { mn = Math . Min ( mn , arr [ i + 1 ] ) ; mx = Math . Max ( mx , arr [ i + 1 ] ) ; } } int res = ( mx + mn ) / 2 ; return res ; }
public static void Main ( ) { int n = 5 ; int [ ] arr = new int [ ] { - 1 , 10 , - 1 , 12 , - 1 } ;
int res = missingnumber ( n , arr ) ; Console . WriteLine ( res ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int lcsubtr ( char [ ] a , char [ ] b , int length1 , int length2 ) {
int [ , ] dp = new int [ length1 + 1 , length2 + 1 ] ; int max = 0 ;
for ( int i = 0 ; i <= length1 ; ++ i ) { for ( int j = 0 ; j <= length2 ; ++ j ) {
if ( i == 0 j == 0 ) { dp [ i , j ] = 0 ; }
else if ( a [ i - 1 ] == b [ j - 1 ] ) { dp [ i , j ] = dp [ i - 1 , j - 1 ] + 1 ; = Math . Max ( dp [ i , j ] , max ) ; }
else { dp [ i , j ] = 0 ; } } }
return max ; }
public static void Main ( ) { string m = "0110" ; string n = "1101" ; char [ ] m1 = m . ToCharArray ( ) ; char [ ] m2 = n . ToCharArray ( ) ;
Console . Write ( lcsubtr ( m1 , m2 , m1 . Length , m2 . Length ) ) ; } }
using System ; class GFG { static int maxN = 20 ; static int maxSum = 50 ; static int minSum = 50 ; static int Base = 50 ;
static int [ , ] dp = new int [ maxN , maxSum + minSum ] ; static bool [ , ] v = new bool [ maxN , maxSum + minSum ] ;
static int findCnt ( int [ ] arr , int i , int required_sum , int n ) {
if ( i == n ) { if ( required_sum == 0 ) return 1 ; else return 0 ; }
if ( v [ i , required_sum + Base ] ) return dp [ i , required_sum + Base ] ;
v [ i , required_sum + Base ] = true ;
dp [ i , required_sum + Base ] = findCnt ( arr , i + 1 , required_sum , n ) + findCnt ( arr , i + 1 , required_sum - arr [ i ] , n ) ; return dp [ i , required_sum + Base ] ; }
static void countSubsets ( int [ ] arr , int K , int n ) {
int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
int S1 = ( sum + K ) / 2 ;
Console . Write ( findCnt ( arr , 0 , S1 , n ) ) ; }
static void Main ( ) { int [ ] arr = { 1 , 1 , 2 , 3 } ; int N = arr . Length ; int K = 1 ;
countSubsets ( arr , K , N ) ; } }
using System ; using System . Collections . Generic ; public class GFG { static float [ , ] dp = new float [ 105 , 605 ] ;
static float find ( int N , int sum ) { if ( N < 0 sum < 0 ) return 0 ; if ( dp [ N , sum ] > 0 ) return dp [ N , sum ] ;
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return ( float ) ( 1.0 / 6 ) ; else return 0 ; } for ( int i = 1 ; i <= 6 ; i ++ ) dp [ N , sum ] = dp [ N , sum ] + find ( N - 1 , sum - i ) / 6 ; return dp [ N , sum ] ; }
public static void Main ( String [ ] args ) { int N = 4 , a = 13 , b = 17 ; float probability = 0.0f ;
for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
Console . Write ( " { 0 : F6 } " , probability ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static int count ( int n ) {
Dictionary < int , int > dp = new Dictionary < int , int > ( ) ;
dp . Add ( 0 , 0 ) ; dp . Add ( 1 , 1 ) ;
if ( ! dp . ContainsKey ( n ) ) dp . Add ( n , 1 + Math . Min ( n % 2 + count ( n / 2 ) , n % 3 + count ( n / 3 ) ) ) ;
return dp [ n ] ; }
public static void Main ( String [ ] args ) {
int N = 6 ;
Console . WriteLine ( String . Join ( " " , ( count ( N ) ) ) ) ; } }
using System ; class GFG {
static void find_minimum_operations ( int n , int [ ] b , int k ) {
int [ ] d = new int [ n + 1 ] ;
int i , operations = 0 , need ; for ( i = 0 ; i < n ; i ++ ) {
if ( i > 0 ) { d [ i ] += d [ i - 1 ] ; }
if ( b [ i ] > d [ i ] ) {
operations += b [ i ] - d [ i ] ; need = b [ i ] - d [ i ] ;
d [ i ] += need ;
if ( i + k <= n ) { d [ i + k ] -= need ; } } } Console . Write ( operations ) ; }
public static void Main ( string [ ] args ) { int n = 5 ; int [ ] b = { 1 , 2 , 3 , 4 , 5 } ; int k = 2 ;
find_minimum_operations ( n , b , k ) ; } }
using System ; class GFG {
static int ways ( int [ , ] arr , int K ) { int R = arr . GetLength ( 0 ) ; int C = arr . GetLength ( 1 ) ; int [ , ] preSum = new int [ R , C ] ;
for ( int r = R - 1 ; r >= 0 ; r -- ) { for ( int c = C - 1 ; c >= 0 ; c -- ) { preSum [ r , c ] = arr [ r , c ] ; if ( r + 1 < R ) preSum [ r , c ] += preSum [ r + 1 , c ] ; if ( c + 1 < C ) preSum [ r , c ] += preSum [ r , c + 1 ] ; if ( r + 1 < R && c + 1 < C ) preSum [ r , c ] -= preSum [ r + 1 , c + 1 ] ; } }
int [ , , ] dp = new int [ K + 1 , R , C ] ;
for ( int k = 1 ; k <= K ; k ++ ) { for ( int r = R - 1 ; r >= 0 ; r -- ) { for ( int c = C - 1 ; c >= 0 ; c -- ) { if ( k == 1 ) { dp [ k , r , c ] = ( preSum [ r , c ] > 0 ) ? 1 : 0 ; } else { dp [ k , r , c ] = 0 ; for ( int r1 = r + 1 ; r1 < R ; r1 ++ ) {
if ( preSum [ r , c ] - preSum [ r1 , c ] > 0 ) dp [ k , r , c ] += dp [ k - 1 , r1 , c ] ; } for ( int c1 = c + 1 ; c1 < C ; c1 ++ ) {
if ( preSum [ r , c ] - preSum [ r , c1 ] > 0 ) dp [ k , r , c ] += dp [ k - 1 , r , c1 ] ; } } } } } return dp [ K , 0 , 0 ] ; }
public static void Main ( string [ ] args ) { int [ , ] arr = { { 1 , 0 , 0 } , { 1 , 1 , 1 } , { 0 , 0 , 0 } } ; int k = 3 ;
Console . WriteLine ( ways ( arr , k ) ) ; } }
using System ; class GFG { static int p = 1000000007 ;
static int power ( int x , int y , int p ) { int res = 1 ; x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
static void nCr ( int n , int p , int [ , ] f , int m ) { for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= m ; j ++ ) {
if ( j > i ) { f [ i , j ] = 0 ; }
else if ( j = = 0 j == i ) { f [ i , j ] = 1 ; } else { f [ i , j ] = ( f [ i - 1 , j ] + f [ i - 1 , j - 1 ] ) % p ; } } } }
static void ProductOfSubsets ( int [ ] arr , int n , int m ) { int [ , ] f = new int [ n + 1 , 100 ] ; nCr ( n , p - 1 , f , m ) ; Array . Sort ( arr ) ;
long ans = 1 ; for ( int i = 0 ; i < n ; i ++ ) {
int x = 0 ; for ( int j = 1 ; j <= m ; j ++ ) {
if ( m % j == 0 ) {
x = ( x + ( f [ n - i - 1 , m - j ] * f [ i , j - 1 ] ) % ( p - 1 ) ) % ( p - 1 ) ; } } ans = ( ( ans * power ( arr [ i ] , x , p ) ) % p ) ; } Console . Write ( ans + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 4 , 5 , 7 , 9 , 3 } ; int K = 4 ; int N = arr . Length ; ProductOfSubsets ( arr , N , K ) ; } }
using System ; class GFG {
static int countWays ( int n , int m ) {
int [ , ] dp = new int [ m + 1 , n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { dp [ 1 , i ] = 1 ; }
int sum ; for ( int i = 2 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) { sum = 0 ;
for ( int k = 0 ; k <= j ; k ++ ) { sum += dp [ i - 1 , k ] ; }
dp [ i , j ] = sum ; } }
return dp [ m , n ] ; }
public static void Main ( String [ ] args ) { int N = 2 , K = 3 ;
Console . Write ( countWays ( N , K ) ) ; } }
using System ; class GFG {
static int countWays ( int n , int m ) {
int [ , ] dp = new int [ m + 1 , n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { dp [ 1 , i ] = 1 ; if ( i != 0 ) { dp [ 1 , i ] += dp [ 1 , i - 1 ] ; } }
for ( int i = 2 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) {
if ( j == 0 ) { dp [ i , j ] = dp [ i - 1 , j ] ; }
else { dp [ i , j ] = dp [ i - 1 , j ] ;
if ( i == m && j == n ) { return dp [ i , j ] ; }
dp [ i , j ] += dp [ i , j - 1 ] ; } } } return Int32 . MinValue ; }
public static void Main ( ) { int N = 2 , K = 3 ;
Console . Write ( countWays ( N , K ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static void SieveOfEratosthenes ( int MAX , List < int > primes ) { Boolean [ ] prime = new Boolean [ MAX + 1 ] ; for ( int i = 0 ; i < MAX + 1 ; i ++ ) prime [ i ] = true ;
for ( int p = 2 ; p * p <= MAX ; p ++ ) { if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( int i = 2 ; i <= MAX ; i ++ ) { if ( prime [ i ] ) primes . Add ( i ) ; } }
public static int findLongest ( int [ ] A , int n ) {
Dictionary < int , int > mpp = new Dictionary < int , int > ( ) ; List < int > primes = new List < int > ( ) ;
SieveOfEratosthenes ( A [ n - 1 ] , primes ) ; int [ ] dp = new int [ n ] ;
dp [ n - 1 ] = 1 ; mpp . Add ( A [ n - 1 ] , n - 1 ) ;
for ( int i = n - 2 ; i >= 0 ; i -- ) {
int num = A [ i ] ;
dp [ i ] = 1 ;
foreach ( int it in primes ) {
int xx = num * it ;
if ( xx > A [ n - 1 ] ) break ;
else if ( mpp . ContainsKey ( xx ) & & mpp [ xx ] != 0 ) {
dp [ i ] = Math . Max ( dp [ i ] , 1 + dp [ mpp [ xx ] ] ) ; } }
if ( mpp . ContainsKey ( A [ i ] ) ) mpp [ A [ i ] ] = i ; else mpp . Add ( A [ i ] , i ) ; } int ans = 1 ;
for ( int i = 0 ; i < n ; i ++ ) ans = Math . Max ( ans , dp [ i ] ) ; return ans ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 2 , 5 , 6 , 12 , 35 , 60 , 385 } ; int n = a . Length ; Console . WriteLine ( findLongest ( a , n ) ) ; } }
using System ; class GFG {
static int waysToKAdjacentSetBits ( int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( != 1 ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
public static void Main ( ) { int n = 5 , k = 2 ;
int totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; Console . WriteLine ( " Number ▁ of ▁ ways ▁ = ▁ " + totalWays ) ; } }
using System ; class GFG {
static void postfix ( int [ ] a , int n ) { for ( int i = n - 1 ; i > 0 ; i -- ) { a [ i - 1 ] = a [ i - 1 ] + a [ i ] ; } }
static void modify ( int [ ] a , int n ) { for ( int i = 1 ; i < n ; i ++ ) { a [ i - 1 ] = i * a [ i ] ; } }
static void allCombination ( int [ ] a , int n ) { int sum = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { sum += i ; } Console . WriteLine ( " f ( 1 ) ▁ - - > ▁ " + sum ) ;
for ( int i = 1 ; i < n ; i ++ ) {
postfix ( a , n - i + 1 ) ;
sum = 0 ; for ( int j = 1 ; j <= n - i ; j ++ ) { sum += ( j * a [ j ] ) ; } Console . WriteLine ( " f ( " + ( i + 1 ) + " ) ▁ - - > ▁ " + sum ) ;
modify ( a , n ) ; } }
public static void Main ( String [ ] args ) { int n = 5 ; int [ ] a = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) { a [ i ] = i + 1 ; }
allCombination ( a , n ) ; } }
using System ; public class GfG {
public static int findStep ( int n ) { if ( n == 1 n == 0 ) return 1 ; else if ( n == 2 ) return 2 ; else return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) ; }
public static void Main ( ) { int n = 4 ; Console . WriteLine ( findStep ( n ) ) ; } }
using System ; class GFG {
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
using System ; class GFG {
static bool findPartiion ( int [ ] arr , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; bool [ ] part = new bool [ sum / 2 + 1 ] ;
for ( i = 0 ; i <= sum / 2 ; i ++ ) { part [ i ] = false ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = sum / 2 ; j >= arr [ i ] ; j -- ) {
if ( part [ j - arr [ i ] ] == true j == arr [ i ] ) part [ j ] = true ; } } return part [ sum / 2 ] ; }
static void Main ( ) { int [ ] arr = { 1 , 3 , 3 , 2 , 3 , 2 } ; int n = 6 ;
if ( findPartiion ( arr , n ) == true ) Console . WriteLine ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " + " subsets ▁ of ▁ equal ▁ sum " ) ; else Console . WriteLine ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " + " two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
using System ; public class GFG {
static int binomialCoeff ( int n , int r ) { if ( r > n ) return 0 ; long m = 1000000007 ; long [ ] inv = new long [ r + 1 ] ; inv [ 0 ] = 1 ; if ( r + 1 >= 2 ) inv [ 1 ] = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { inv [ i ] = m - ( m / i ) * inv [ ( int ) ( m % i ) ] % m ; } int ans = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { ans = ( int ) ( ( ( ans % m ) * ( inv [ i ] % m ) ) % m ) ; }
for ( int i = n ; i >= ( n - r + 1 ) ; i -- ) { ans = ( int ) ( ( ( ans % m ) * ( i % m ) ) % m ) ; } return ans ; }
public static void Main ( String [ ] args ) { int n = 5 , r = 2 ; Console . Write ( " Value ▁ of ▁ C ( " + n + " , ▁ " + r + " ) ▁ is ▁ " + binomialCoeff ( n , r ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
public static int gcd ( int a , int b ) {
if ( a < b ) { int t = a ; a = b ; b = t ; } if ( a % b == 0 ) return b ;
return gcd ( b , a % b ) ; }
static void printAnswer ( int x , int y ) {
int val = gcd ( x , y ) ;
if ( ( val & ( val - 1 ) ) == 0 ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; }
public static void Main ( ) {
int x = 4 ; int y = 7 ;
printAnswer ( x , y ) ; } }
using System ; class GFG {
static int getElement ( int N , int r , int c ) {
if ( r > c ) return 0 ;
if ( r == 1 ) { return c ; }
int a = ( r + 1 ) * ( int ) ( Math . Pow ( 2 , ( r - 2 ) ) ) ;
int d = ( int ) ( Math . Pow ( 2 , ( r - 1 ) ) ) ;
c = c - r ; int element = a + d * c ; return element ; }
public static void Main ( String [ ] args ) { int N = 4 , R = 3 , C = 4 ; Console . WriteLine ( getElement ( N , R , C ) ) ; } }
using System ; class GFG {
static String MinValue ( string number , int x ) {
int length = number . Length ;
int position = length + 1 ;
if ( number [ 0 ] == ' - ' ) {
for ( int i = number . Length - 1 ; i >= 1 ; -- i ) { if ( ( number [ i ] - 48 ) < x ) { position = i ; } } } else {
for ( int i = number . Length - 1 ; i >= 0 ; -- i ) { if ( ( number [ i ] - 48 ) > x ) { position = i ; } } }
number = number . Substring ( 0 , position ) + x + number . Substring ( position , number . Length ) ;
return number . ToString ( ) ; }
public static void Main ( ) {
string number = "89" ; int x = 1 ;
Console . WriteLine ( MinValue ( number , x ) ) ; } }
using System ; class GFG {
public static String divisibleByk ( String s , int n , int k ) {
int [ ] poweroftwo = new int [ n ] ;
poweroftwo [ 0 ] = 1 % k ; for ( int i = 1 ; i < n ; i ++ ) {
poweroftwo [ i ] = ( poweroftwo [ i - 1 ] * ( 2 % k ) ) % k ; }
int rem = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ n - i - 1 ] == '1' ) {
rem += ( poweroftwo [ i ] ) ; rem %= k ; } }
if ( rem == 0 ) { return " Yes " ; }
else return " No " ; }
public static void Main ( String [ ] args ) {
String s = "1010001" ; int k = 9 ;
int n = s . Length ;
Console . Write ( divisibleByk ( s , n , k ) ) ; } }
using System ; public class GFG {
static int maxSumbySplittingString ( String str , int N ) {
int cntOne = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == '1' ) {
cntOne ++ ; } }
int zero = 0 ;
int one = 0 ;
int res = 0 ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( str [ i ] == '0' ) {
zero ++ ; }
else {
one ++ ; }
res = Math . Max ( res , zero + cntOne - one ) ; } return res ; }
public static void Main ( String [ ] args ) { String str = "00111" ; int N = str . Length ; Console . Write ( maxSumbySplittingString ( str , N ) ) ; } }
using System ; class GFG {
static void cntBalancedParenthesis ( String s , int N ) {
int cntPairs = 0 ;
int cntCurly = 0 ;
int cntSml = 0 ;
int cntSqr = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( s [ i ] == ' { ' ) {
cntCurly ++ ; } else if ( s [ i ] == ' ( ' ) {
cntSml ++ ; } else if ( s [ i ] == ' [ ' ) {
cntSqr ++ ; } else if ( s [ i ] == ' } ' && cntCurly > 0 ) {
cntCurly -- ;
cntPairs ++ ; } else if ( s [ i ] == ' ) ' && cntSml > 0 ) {
cntSml -- ;
cntPairs ++ ; } else if ( s [ i ] == ' ] ' && cntSqr > 0 ) {
cntSqr -- ;
cntPairs ++ ; } } Console . WriteLine ( cntPairs ) ; }
static public void Main ( ) {
String s = " { ( } ) " ; int N = s . Length ;
cntBalancedParenthesis ( s , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int arcIntersection ( String S , int len ) { Stack < char > stk = new Stack < char > ( ) ;
for ( int i = 0 ; i < len ; i ++ ) {
stk . Push ( S [ i ] ) ; if ( stk . Count >= 2 ) {
char temp = stk . Peek ( ) ;
stk . Pop ( ) ;
if ( stk . Peek ( ) == temp ) { stk . Pop ( ) ; }
else { stk . Push ( temp ) ; } } }
if ( stk . Count == 0 ) return 1 ; return 0 ; }
static void countString ( String [ ] arr , int N ) {
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int len = arr [ i ] . Length ;
count += arcIntersection ( arr [ i ] , len ) ; }
Console . Write ( count + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { String [ ] arr = { "0101" , "0011" , "0110" } ; int N = arr . Length ;
countString ( arr , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static String ConvertequivalentBase8 ( String S ) {
Dictionary < String , char > mp = new Dictionary < String , char > ( ) ;
mp . Add ( "000" , '0' ) ; mp . Add ( "001" , '1' ) ; mp . Add ( "010" , '2' ) ; mp . Add ( "011" , '3' ) ; mp . Add ( "100" , '4' ) ; mp . Add ( "101" , '5' ) ; mp . Add ( "110" , '6' ) ; mp . Add ( "111" , '7' ) ;
int N = S . Length ; if ( N % 3 == 2 ) {
S = "0" + S ; } else if ( % 3 == 1 ) {
S = "00" + S ; }
N = S . Length ;
String oct = " " ;
for ( int i = 0 ; i < N ; i += 3 ) {
String temp = S . Substring ( 0 , N ) ;
if ( mp . ContainsKey ( temp ) ) oct += mp [ temp ] ; } return oct ; }
static String binString_div_9 ( String S , int N ) {
String oct = " " ; oct = ConvertequivalentBase8 ( S ) ;
int oddSum = 0 ;
int evenSum = 0 ;
int M = oct . Length ;
for ( int i = 0 ; i < M ; i += 2 )
oddSum += ( oct [ i ] - '0' ) ;
for ( int i = 1 ; i < M ; i += 2 ) {
evenSum += ( oct [ i ] - '0' ) ; }
int Oct_9 = 11 ;
if ( Math . Abs ( oddSum - evenSum ) % Oct_9 == 0 ) { return " Yes " ; } return " No " ; }
public static void Main ( String [ ] args ) { String S = "1010001" ; int N = S . Length ; Console . WriteLine ( binString_div_9 ( S , N ) ) ; } }
using System ; class GFG {
static int min_cost ( String S ) {
int cost = 0 ;
int F = 0 ;
int B = 0 ; int count = 0 ; foreach ( char c in S . ToCharArray ( ) ) if ( c == ' ▁ ' ) count ++ ;
int n = S . Length - count ;
if ( n == 1 ) return cost ;
foreach ( char inn in S . ToCharArray ( ) ) {
if ( inn != ' ▁ ' ) {
if ( B != 0 ) {
cost += Math . Min ( n - F , F ) * B ; B = 0 ; }
F += 1 ; }
else {
B += 1 ; } }
return cost ; }
public static void Main ( String [ ] args ) { String S = " ▁ @ $ " ; Console . WriteLine ( min_cost ( S ) ) ; } }
using System ; class GFG {
static bool isVowel ( char ch ) { if ( ch == ' a ' ch == ' e ' ch == ' i ' ch == ' o ' ch == ' u ' ) return true ; else return false ; }
static int minCost ( String S ) {
int cA = 0 ; int cE = 0 ; int cI = 0 ; int cO = 0 ; int cU = 0 ;
for ( int i = 0 ; i < S . Length ; i ++ ) {
if ( isVowel ( S [ i ] ) ) {
cA += Math . Abs ( S [ i ] - ' a ' ) ; cE += Math . Abs ( S [ i ] - ' e ' ) ; cI += Math . Abs ( S [ i ] - ' i ' ) ; cO += Math . Abs ( S [ i ] - ' o ' ) ; cU += Math . Abs ( S [ i ] - ' u ' ) ; } }
return Math . Min ( Math . Min ( Math . Min ( Math . Min ( cA , cE ) , cI ) , cO ) , cU ) ; }
public static void Main ( String [ ] args ) { String S = " geeksforgeeks " ; Console . WriteLine ( minCost ( S ) ) ; } }
using System ; class GFG {
public static void decode_String ( String str , int K ) { String ans = " " ;
for ( int i = 0 ; i < str . Length ; i += K )
ans += str [ i ] ;
for ( int i = str . Length - ( K - 1 ) ; i < str . Length ; i ++ ) ans += str [ i ] ; Console . WriteLine ( ans ) ; }
public static void Main ( String [ ] args ) { int K = 3 ; String str = " abcbcscsesesesd " ; decode_String ( str , K ) ; } }
using System ; class GFG {
static string maxVowelSubString ( string str , int K ) {
int N = str . Length ;
int [ ] pref = new int [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' a ' str [ i ] == ' e ' str [ i ] == ' i ' str [ i ] == ' o ' str [ i ] == ' u ' ) pref [ i ] = 1 ;
else pref [ i ] = 0 ;
if ( i != 0 ) pref [ i ] += pref [ i - 1 ] ; }
int maxCount = pref [ K - 1 ] ;
string res = str . Substring ( 0 , K ) ;
for ( int i = K ; i < N ; i ++ ) {
int currCount = pref [ i ] - pref [ i - K ] ;
if ( currCount > maxCount ) { maxCount = currCount ; res = str . Substring ( i - K + 1 , K ) ; }
else if ( currCount = = maxCount ) { string temp = str . Substring ( i - K + 1 , K ) ; if ( string . Compare ( temp , res ) == - 1 ) res = temp ; } }
return res ; }
public static void Main ( ) { string str = " ceebbaceeffo " ; int K = 3 ; Console . Write ( maxVowelSubString ( str , K ) ) ; } }
using System ; class GFG {
static void decodeStr ( String str , int len ) {
char [ ] c = new char [ len ] ; int med , pos = 1 , k ;
if ( len % 2 == 1 ) med = len / 2 ; else med = len / 2 - 1 ;
c [ med ] = str [ 0 ] ;
if ( len % 2 == 0 ) c [ med + 1 ] = str [ 1 ] ;
if ( len % 2 == 1 ) k = 1 ; else k = 2 ; for ( int i = k ; i < len ; i += 2 ) { c [ med - pos ] = str [ i ] ;
if ( len % 2 == 1 ) c [ med + pos ] = str [ i + 1 ] ;
else c [ med + pos + 1 ] = str [ i + 1 ] ; pos ++ ; }
for ( int i = 0 ; i < len ; i ++ ) Console . Write ( c [ i ] ) ; }
public static void Main ( String [ ] args ) { String str = " ofrsgkeeeekgs " ; int len = str . Length ; decodeStr ( str , len ) ; } }
using System ; class GFG { static void findCount ( String s , int L , int R ) {
int distinct = 0 ;
int [ ] frequency = new int [ 26 ] ;
for ( int i = L ; i <= R ; i ++ ) {
frequency [ s [ i ] - ' a ' ] ++ ; } for ( int i = 0 ; i < 26 ; i ++ ) {
if ( frequency [ i ] > 0 ) distinct ++ ; } Console . Write ( distinct + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { String s = " geeksforgeeksisa " + " computerscienceportal " ; int queries = 3 ; int [ , ] Q = { { 0 , 10 } , { 15 , 18 } , { 12 , 20 } } ; for ( int i = 0 ; i < queries ; i ++ ) findCount ( s , Q [ i , 0 ] , Q [ i , 1 ] ) ; } }
using System ; class GFG {
static string ReverseComplement ( char [ ] s , int n , int k ) {
int rev = ( k + 1 ) / 2 ;
int complement = k - rev ;
if ( rev % 2 == 1 ) s = reverse ( s ) ;
if ( complement % 2 == 1 ) { for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == '0' ) s [ i ] = '1' ; else s [ i ] = '0' ; } }
return ( new string ( s ) ) ; } static char [ ] reverse ( char [ ] a ) { int i , n = a . Length ; char t ; for ( i = 0 ; i < n / 2 ; i ++ ) { t = a [ i ] ; a [ i ] = a [ n - i - 1 ] ; a [ n - i - 1 ] = t ; } return a ; }
public static void Main ( ) { string str = "10011" ; int k = 5 ; int n = str . Length ;
Console . Write ( ReverseComplement ( str . ToCharArray ( ) , n , k ) ) ; } }
using System ; class GFG {
static bool repeatingString ( String s , int n , int k ) {
if ( n % k != 0 ) { return false ; }
int [ ] frequency = new int [ 123 ] ;
for ( int i = 0 ; i < 123 ; i ++ ) { frequency [ i ] = 0 ; }
for ( int i = 0 ; i < n ; i ++ ) { frequency [ s [ i ] ] ++ ; } int repeat = n / k ;
for ( int i = 0 ; i < 123 ; i ++ ) { if ( frequency [ i ] % repeat != 0 ) { return false ; } } return true ; }
public static void Main ( String [ ] args ) { String s = " abcdcba " ; int n = s . Length ; int k = 3 ; if ( repeatingString ( s , n , k ) ) { Console . Write ( " Yes " + " STRNEWLINE " ) ; } else { Console . Write ( " No " + " STRNEWLINE " ) ; } } }
using System ; class GFG {
static void findPhoneNumber ( int n ) { int temp = n ; int sum = 0 ;
while ( temp != 0 ) { sum += temp % 10 ; temp = temp / 10 ; }
if ( sum < 10 ) Console . Write ( n + "0" + sum ) ;
else Console . ( n + "" + sum ) ; }
static public void Main ( ) { int n = 98765432 ; findPhoneNumber ( n ) ; } }
using System ; class GFG { static int maxN = 20 ; static int maxM = 64 ;
static int cntSplits ( String s ) {
if ( s [ s . Length - 1 ] == '1' ) return 0 ;
int c_zero = 0 ;
for ( int i = 0 ; i < s . Length ; i ++ ) c_zero += ( s [ i ] == '0' ) ? 1 : 0 ;
return ( int ) Math . Pow ( 2 , c_zero - 1 ) ; }
public static void Main ( String [ ] args ) { String s = "10010" ; Console . WriteLine ( cntSplits ( s ) ) ; } }
using System ; class GFG {
static void findNumbers ( String s ) {
int n = s . Length ;
int count = 1 ; int result = 0 ;
int left = 0 ; int right = 1 ; while ( right < n ) {
if ( s [ left ] == s [ right ] ) count ++ ;
else {
result += count * ( count + 1 ) / 2 ;
left = right ; count = 1 ; } right ++ ; }
result += count * ( count + 1 ) / 2 ; Console . WriteLine ( result ) ; }
public static void Main ( String [ ] args ) { String s = " bbbcbb " ; findNumbers ( s ) ; } }
using System ; class GFG {
static bool isVowel ( char ch ) { ch = char . ToUpper ( ch ) ; return ( ch == ' A ' ch == ' E ' ch == ' I ' ch == ' O ' ch == ' U ' ) ; }
static String duplicateVowels ( String str ) { int t = str . Length ;
String res = " " ;
for ( int i = 0 ; i < t ; i ++ ) { if ( isVowel ( str [ i ] ) ) res += str [ i ] ; res += str [ i ] ; } return res ; }
public static void Main ( String [ ] args ) { String str = " helloworld " ;
Console . WriteLine ( " Original ▁ String : ▁ " + str ) ; String res = duplicateVowels ( str ) ;
Console . WriteLine ( " String ▁ with ▁ Vowels ▁ duplicated : ▁ " + res ) ; } }
using System ; class GFG {
static int stringToInt ( String str ) {
if ( str . Length == 1 ) return ( str [ 0 ] - '0' ) ;
double y = stringToInt ( str . Substring ( 1 ) ) ;
double x = str [ 0 ] - '0' ;
x = x * Math . Pow ( 10 , str . Length - 1 ) + y ; return ( int ) ( x ) ; }
public static void Main ( String [ ] args ) { String str = "1235" ; Console . Write ( stringToInt ( str ) ) ; } }
using System ; class GFG { static int MAX = 26 ;
static int largestSubSeq ( string [ ] arr , int n ) {
int [ ] count = new int [ MAX ] ;
for ( int i = 0 ; i < n ; i ++ ) { string str = arr [ i ] ;
bool [ ] hash = new bool [ MAX ] ; for ( int j = 0 ; j < str . Length ; j ++ ) { hash [ str [ j ] - ' a ' ] = true ; } for ( int j = 0 ; j < MAX ; j ++ ) {
if ( hash [ j ] ) count [ j ] ++ ; } } int max = - 1 ; for ( int i = 0 ; i < MAX ; i ++ ) { if ( max < count [ i ] ) max = count [ i ] ; } return max ; }
public static void Main ( ) { string [ ] arr = { " ab " , " bc " , " de " } ; int n = arr . Length ; Console . WriteLine ( largestSubSeq ( arr , n ) ) ; } }
using System ; class GFG {
static bool isPalindrome ( string str ) { int len = str . Length ; for ( int i = 0 ; i < len / 2 ; i ++ ) { if ( str [ i ] != str [ len - 1 - i ] ) return false ; } return true ; }
static bool createStringAndCheckPalindrome ( int N ) {
string sub = " " + N , res_str = " " ; int sum = 0 ;
while ( N > 0 ) { int digit = N % 10 ; sum += digit ; N = N / 10 ; }
while ( res_str . Length < sum ) res_str += sub ;
if ( res_str . Length > sum ) res_str = res_str . Substring ( 0 , sum ) ;
if ( isPalindrome ( res_str ) ) return true ; return false ; }
public static void Main ( ) { int N = 10101 ; if ( createStringAndCheckPalindrome ( N ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static int minimumLength ( String s ) { int maxOcc = 0 , n = s . Length ; int [ ] arr = new int [ 26 ] ;
for ( int i = 0 ; i < n ; i ++ ) arr [ s [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < 26 ; i ++ ) if ( arr [ i ] > maxOcc ) maxOcc = arr [ i ] ;
return ( n - maxOcc ) ; }
public static void Main ( String [ ] args ) { String str = " afddewqd " ; Console . WriteLine ( minimumLength ( str ) ) ; } }
using System ; class GFG {
static void removeSpecialCharacter ( string s ) { for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( s [ i ] < ' A ' s [ i ] > ' Z ' && s [ i ] < ' a ' s [ i ] > ' z ' ) {
s = s . Remove ( i , 1 ) ; i -- ; } } Console . Write ( s ) ; }
public static void Main ( ) { string s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " ; removeSpecialCharacter ( s ) ; } }
using System ; public class GFG {
static void removeSpecialCharacter ( String str ) { char [ ] s = str . ToCharArray ( ) ; int j = 0 ; for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( ( s [ i ] >= ' A ' && s [ i ] <= ' Z ' ) || ( s [ i ] >= ' a ' && s [ i ] <= ' z ' ) ) { s [ j ] = s [ i ] ; j ++ ; } } Console . WriteLine ( String . Join ( " " , s ) . Substring ( 0 , j ) ) ; }
public static void Main ( ) { String s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " ; removeSpecialCharacter ( s ) ; } }
using System ; class GFG { static int findRepeatFirstN2 ( string s ) {
int p = - 1 , i , j ; for ( i = 0 ; i < s . Length ; i ++ ) { for ( j = i + 1 ; j < s . Length ; j ++ ) { if ( s [ i ] == s [ j ] ) { p = i ; break ; } } if ( p != - 1 ) break ; } return p ; }
static public void Main ( ) { string str = " geeksforgeeks " ; int pos = findRepeatFirstN2 ( str ) ; if ( pos == - 1 ) Console . WriteLine ( " Not ▁ found " ) ; else Console . WriteLine ( str [ pos ] ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG { public static void prCharWithFreq ( string s ) {
Dictionary < char , int > d = new Dictionary < char , int > ( ) ; foreach ( char i in s ) { if ( d . ContainsKey ( i ) ) { d [ i ] ++ ; } else { d [ i ] = 1 ; } }
foreach ( char i in s ) {
if ( d [ i ] != 0 ) { Console . Write ( i + d [ i ] . ToString ( ) + " ▁ " ) ; d [ i ] = 0 ; } } }
public static void Main ( string [ ] args ) { string s = " geeksforgeeks " ; prCharWithFreq ( s ) ; } }
using System ; class GFG {
static int possibleStrings ( int n , int r , int b , int g ) {
int [ ] fact = new int [ n + 1 ] ; fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
int left = n - ( r + g + b ) ; int sum = 0 ;
for ( int i = 0 ; i <= left ; i ++ ) { for ( int j = 0 ; j <= left - i ; j ++ ) { int k = left - ( i + j ) ;
sum = sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ; } }
return sum ; }
public static void Main ( ) { int n = 4 , r = 2 ; int b = 0 , g = 1 ; Console . WriteLine ( possibleStrings ( n , r , b , g ) ) ; } }
using System ; class GFG {
static int remAnagram ( string str1 , string str2 ) {
int [ ] count1 = new int [ 26 ] ; int [ ] count2 = new int [ 26 ] ;
for ( int i = 0 ; i < str1 . Length ; i ++ ) count1 [ str1 [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < str2 . Length ; i ++ ) count2 [ str2 [ i ] - ' a ' ] ++ ;
int result = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) result += Math . Abs ( count1 [ i ] - count2 [ i ] ) ; return result ; }
public static void Main ( ) { string str1 = " bcadeh " , str2 = " hea " ; Console . Write ( remAnagram ( str1 , str2 ) ) ; } }
using System ; public class GFG {
static int CHARS = 26 ;
static bool isValidString ( String str ) { int [ ] freq = new int [ CHARS ] ; int i = 0 ;
for ( i = 0 ; i < str . Length ; i ++ ) { freq [ str [ i ] - ' a ' ] ++ ; }
int freq1 = 0 , count_freq1 = 0 ; for ( i = 0 ; i < CHARS ; i ++ ) { if ( freq [ i ] != 0 ) { freq1 = freq [ i ] ; count_freq1 = 1 ; break ; } }
int j , freq2 = 0 , count_freq2 = 0 ; for ( j = i + 1 ; j < CHARS ; j ++ ) { if ( freq [ j ] != 0 ) { if ( freq [ j ] == freq1 ) { count_freq1 ++ ; } else { count_freq2 = 1 ; freq2 = freq [ j ] ; break ; } } }
for ( int k = j + 1 ; k < CHARS ; k ++ ) { if ( freq [ k ] != 0 ) { if ( freq [ k ] == freq1 ) { count_freq1 ++ ; } if ( freq [ k ] == freq2 ) { count_freq2 ++ ;
{ return false ; } }
if ( count_freq1 > 1 && count_freq2 > 1 ) { return false ; } }
return true ; }
public static void Main ( ) { String str = " abcbc " ; if ( isValidString ( str ) ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
using System ; using System . Collections . Generic ; public class AllCharsWithSameFrequencyWithOneVarAllowed {
public static bool checkForVariation ( String str ) { if ( str == null str . Length != 0 ) { return true ; } Dictionary < char , int > map = new Dictionary < char , int > ( ) ;
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( map . ContainsKey ( str [ i ] ) ) map [ str [ i ] ] = map [ str [ i ] ] + 1 ; else map . Add ( str [ i ] , 1 ) ; }
bool first = true , second = true ; int val1 = 0 , val2 = 0 ; int countOfVal1 = 0 , countOfVal2 = 0 ; foreach ( KeyValuePair < char , int > itr in map ) { int i = itr . Key ;
if ( first ) { val1 = i ; first = false ; countOfVal1 ++ ; continue ; } if ( i == val1 ) { countOfVal1 ++ ; continue ; }
if ( second ) { val2 = i ; countOfVal2 ++ ; second = false ; continue ; } if ( i == val2 ) { countOfVal2 ++ ; continue ; } return false ; } if ( countOfVal1 > 1 && countOfVal2 > 1 ) { return false ; } else { return true ; } }
public static void Main ( String [ ] args ) { Console . WriteLine ( checkForVariation ( " abcbc " ) ) ; } }
using System ; class GFG {
static int countCompletePairs ( string [ ] set1 , string [ ] set2 , int n , int m ) { int result = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
string concat = set1 [ i ] + set2 [ j ] ;
int [ ] frequency = new int [ 26 ] ; for ( int k = 0 ; k < concat . Length ; k ++ ) { frequency [ concat [ k ] - ' a ' ] ++ ; }
int l ; for ( l = 0 ; l < 26 ; l ++ ) { if ( frequency [ l ] < 1 ) { break ; } } if ( l == 26 ) { result ++ ; } } } return result ; }
static public void Main ( ) { string [ ] set1 = { " abcdefgh " , " geeksforgeeks " , " lmnopqrst " , " abc " } ; string [ ] set2 = { " ijklmnopqrstuvwxyz " , " abcdefghijklmnopqrstuvwxyz " , " defghijklmnopqrstuvwxyz " } ; int n = set1 . Length ; int m = set2 . Length ; Console . Write ( countCompletePairs ( set1 , set2 , n , m ) ) ; } }
using System ; class GFG {
static int countCompletePairs ( String [ ] set1 , String [ ] set2 , int n , int m ) { int result = 0 ;
int [ ] con_s1 = new int [ n ] ; int [ ] con_s2 = new int [ m ] ;
for ( int i = 0 ; i < n ; i ++ ) {
con_s1 [ i ] = 0 ; for ( int j = 0 ; j < set1 [ i ] . Length ; j ++ ) {
con_s1 [ i ] = con_s1 [ i ] | ( 1 << ( set1 [ i ] [ j ] - ' a ' ) ) ; } }
for ( int i = 0 ; i < m ; i ++ ) {
con_s2 [ i ] = 0 ; for ( int j = 0 ; j < set2 [ i ] . Length ; j ++ ) {
con_s2 [ i ] = con_s2 [ i ] | ( 1 << ( set2 [ i ] [ j ] - ' a ' ) ) ; } }
long complete = ( 1 << 26 ) - 1 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
if ( ( con_s1 [ i ] con_s2 [ j ] ) == complete ) { result ++ ; } } } return result ; }
public static void Main ( String [ ] args ) { String [ ] set1 = { " abcdefgh " , " geeksforgeeks " , " lmnopqrst " , " abc " } ; String [ ] set2 = { " ijklmnopqrstuvwxyz " , " abcdefghijklmnopqrstuvwxyz " , " defghijklmnopqrstuvwxyz " } ; int n = set1 . Length ; int m = set2 . Length ; Console . WriteLine ( countCompletePairs ( set1 , set2 , n , m ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static String encodeString ( String str ) { Dictionary < char , int > map = new Dictionary < char , int > ( ) ; String res = " " ; int i = 0 ;
char ch ; for ( int j = 0 ; j < str . Length ; j ++ ) { ch = str [ j ] ;
if ( ! map . ContainsKey ( ch ) ) map . Add ( ch , i ++ ) ;
res += map [ ch ] ; } return res ; }
static void findMatchedWords ( String [ ] dict , String pattern ) {
int len = pattern . Length ;
String hash = encodeString ( pattern ) ;
foreach ( String word in dict ) {
if ( word . Length == len && encodeString ( word ) . Equals ( hash ) ) Console . Write ( word + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { String [ ] dict = { " abb " , " abc " , " xyz " , " xyy " } ; String pattern = " foo " ; findMatchedWords ( dict , pattern ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG { static bool check ( string pattern , string word ) { if ( pattern . Length != word . Length ) return false ; int [ ] ch = new int [ 128 ] ; int Len = word . Length ; for ( int i = 0 ; i < Len ; i ++ ) { if ( ch [ ( int ) pattern [ i ] ] == 0 ) { ch [ ( int ) pattern [ i ] ] = word [ i ] ; } else if ( ch [ ( int ) pattern [ i ] ] != word [ i ] ) { return false ; } } return true ; }
static void findMatchedWords ( HashSet < string > dict , string pattern ) {
int Len = pattern . Length ;
string result = " ▁ " ; foreach ( string word in dict ) { if ( check ( pattern , word ) ) { result = word + " ▁ " + result ; } } Console . Write ( result ) ; }
static void Main ( ) { HashSet < string > dict = new HashSet < string > ( new string [ ] { " abb " , " abc " , " xyz " , " xyy " } ) ; string pattern = " foo " ; findMatchedWords ( dict , pattern ) ; } }
using System ; public class GFG {
static int countWords ( String str ) {
if ( str == null ) { return 0 ; } int wordCount = 0 ; bool isWord = false ; int endOfLine = str . Length - 1 ;
char [ ] ch = str . ToCharArray ( ) ; for ( int i = 0 ; i < ch . Length ; i ++ ) {
if ( Char . IsLetter ( ch [ i ] ) && i != endOfLine ) { isWord = true ; }
else if ( ! Char . IsLetter ( ch [ i ] ) && ) { wordCount ++ ; isWord = false ; }
else if ( Char . IsLetter ( ch [ i ] ) & & i == endOfLine ) { wordCount ++ ; } }
return wordCount ; }
static public void Main ( ) {
string str = " One ▁ twothree STRNEWLINE ▁ four TABSYMBOL five ▁ " ;
Console . WriteLine ( " No ▁ of ▁ words ▁ : ▁ " + countWords ( str ) ) ; } }
using System ; class GFG {
public static String [ ] RevString ( String [ ] s , int l ) {
if ( l % 2 == 0 ) {
int j = l / 2 ;
while ( j <= l - 1 ) { String temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } }
else {
int j = ( l / 2 ) + 1 ;
while ( j <= l - 1 ) { String temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } }
return s ; }
public static void Main ( String [ ] args ) { String s = " getting ▁ good ▁ at ▁ coding ▁ " + " needs ▁ a ▁ lot ▁ of ▁ practice " ; String [ ] words = s . Split ( " \\ s " ) ; words = RevString ( words , words . Length ) ; s = String . Join ( " ▁ " , words ) ; Console . WriteLine ( s ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void printPath ( List < int > res , int nThNode , int kThNode ) {
if ( kThNode > nThNode ) return ;
res . Add ( kThNode ) ;
for ( int i = 0 ; i < res . Count ; i ++ ) Console . Write ( res [ i ] + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ;
printPath ( res , nThNode , kThNode * 2 ) ;
printPath ( res , nThNode , kThNode * 2 + 1 ) ; res . RemoveAt ( res . Count - 1 ) ; }
static void printPathToCoverAllNodeUtil ( int nThNode ) {
List < int > res = new List < int > ( ) ;
printPath ( res , nThNode , 1 ) ; }
public static void Main ( String [ ] args ) {
int nThNode = 7 ;
printPathToCoverAllNodeUtil ( nThNode ) ; } }
using System ; class GFG {
static int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
static bool isArmstrong ( int x ) { int n = String . Join ( " " , x ) . Length ; int sum1 = 0 ; int temp = x ; while ( temp > 0 ) { int digit = temp % 10 ; sum1 += ( int ) Math . Pow ( digit , n ) ; temp /= 10 ; } if ( sum1 == x ) return true ; return false ; }
static int MaxUtil ( int [ ] st , int ss , int se , int l , int r , int node ) {
if ( l <= ss && r >= se ) return st [ node ] ;
if ( se < l ss > r ) return - 1 ;
int mid = getMid ( ss , se ) ; return Math . Max ( MaxUtil ( st , ss , mid , l , r , 2 * node ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 1 ) ) ; }
static void updateValue ( int [ ] arr , int [ ] st , int ss , int se , int index , int value , int node ) { if ( index < ss index > se ) { Console . Write ( " Invalid ▁ Input " + " STRNEWLINE " ) ; return ; } if ( ss == se ) {
arr [ index ] = value ; if ( isArmstrong ( value ) ) st [ node ] = value ; else st [ node ] = - 1 ; } else { int mid = getMid ( ss , se ) ; if ( index >= ss && index <= mid ) updateValue ( arr , st , ss , mid , index , value , 2 * node ) ; else updateValue ( arr , st , mid + 1 , se , index , value , 2 * node + 1 ) ; st [ node ] = Math . Max ( st [ 2 * node + 1 ] , st [ 2 * node + 2 ] ) ; } return ; }
static int getMax ( int [ ] st , int n , int l , int r ) {
if ( l < 0 r > n - 1 l > r ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
static int constructSTUtil ( int [ ] arr , int ss , int se , int [ ] st , int si ) {
if ( ss == se ) { if ( isArmstrong ( arr [ ss ] ) ) st [ si ] = arr [ ss ] ; else st [ si ] = - 1 ; return st [ si ] ; }
int mid = getMid ( ss , se ) ; st [ si ] = Math . Max ( constructSTUtil ( arr , ss , mid , st , si * 2 ) , constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 1 ) ) ; return st [ si ] ; }
static int [ ] constructST ( int [ ] arr , int n ) {
int x = ( int ) ( Math . Ceiling ( Math . Log ( n ) ) ) ;
int max_size = 2 * ( int ) Math . Pow ( 2 , x ) - 1 ;
int [ ] st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 192 , 113 , 535 , 7 , 19 , 111 } ; int n = arr . Length ;
int [ ] st = constructST ( arr , n ) ;
Console . Write ( " Maximum ▁ armstrong ▁ " + " number ▁ in ▁ given ▁ range ▁ = ▁ " + getMax ( st , n , 1 , 3 ) + " STRNEWLINE " ) ;
updateValue ( arr , st , 0 , n - 1 , 1 , 153 , 0 ) ;
Console . Write ( " Updated ▁ Maximum ▁ armstrong ▁ " + " number ▁ in ▁ given ▁ range ▁ = ▁ " + getMax ( st , n , 1 , 3 ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static void maxRegions ( int n ) { int num ; num = n * ( n + 1 ) / 2 + 1 ;
Console . WriteLine ( num ) ; }
public static void Main ( String [ ] args ) { int n = 10 ; maxRegions ( n ) ; } }
using System ; class GFG {
static void checkSolveable ( int n , int m ) {
if ( n == 1 m == 1 ) Console . WriteLine ( " YES " ) ;
else if ( m = = 2 && n == 2 ) Console . WriteLine ( " YES " ) ; else Console . ( " NO " ) ; }
public static void Main ( ) { int n = 1 , m = 3 ; checkSolveable ( n , m ) ; } }
using System ; class GFG {
static int GCD ( int a , int b ) {
if ( b == 0 ) return a ;
else return GCD ( b , a % b ) ; }
static void check ( int x , int y ) {
if ( GCD ( x , y ) == 1 ) { Console . WriteLine ( " Yes " ) ; } else { Console . WriteLine ( " No " ) ; } }
public static void Main ( ) {
int X = 2 , Y = 7 ;
check ( X , Y ) ; } }
using System ; class GFG { static readonly int size = 1000001 ;
static void seiveOfEratosthenes ( int [ ] prime ) { prime [ 0 ] = 1 ; prime [ 1 ] = 0 ; for ( int i = 2 ; i * i < 1000001 ; i ++ ) {
if ( prime [ i ] == 0 ) { for ( int j = i * i ; j < 1000001 ; j += i ) {
prime [ j ] = 1 ; } } } }
static float probabiltyEuler ( int [ ] prime , int L , int R , int M ) { int [ ] arr = new int [ size ] ; int [ ] eulerTotient = new int [ size ] ; int count = 0 ;
for ( int i = L ; i <= R ; i ++ ) {
eulerTotient [ i - L ] = i ; arr [ i - L ] = i ; } for ( int i = 2 ; i < 1000001 ; i ++ ) {
if ( prime [ i ] == 0 ) {
for ( int j = ( L / i ) * i ; j <= R ; j += i ) { if ( j - L >= 0 ) {
eulerTotient [ j - L ] = eulerTotient [ j - L ] / i * ( i - 1 ) ; while ( arr [ j - L ] % i == 0 ) { arr [ j - L ] /= i ; } } } } }
for ( int i = L ; i <= R ; i ++ ) { if ( arr [ i - L ] > 1 ) { eulerTotient [ i - L ] = ( eulerTotient [ i - L ] / arr [ i - L ] ) * ( arr [ i - L ] - 1 ) ; } } for ( int i = L ; i <= R ; i ++ ) {
if ( ( eulerTotient [ i - L ] % M ) == 0 ) { count ++ ; } }
return ( float ) ( 1.0 * count / ( R + 1 - L ) ) ; }
public static void Main ( String [ ] args ) { int [ ] prime = new int [ size ] ; seiveOfEratosthenes ( prime ) ; int L = 1 , R = 7 , M = 3 ; Console . Write ( probabiltyEuler ( prime , L , R , M ) ) ; } }
using System ; class GFG {
public static void findWinner ( int n , int k ) { int cnt = 0 ;
if ( n == 1 ) Console . Write ( " No " ) ;
else if ( ( n & 1 ) != 0 n == 2 ) Console . Write ( " Yes " ) ; else { int = n ; int val = 1 ;
while ( tmp > k && tmp % 2 == 0 ) { tmp /= 2 ; val *= 2 ; }
for ( int i = 3 ; i <= Math . Sqrt ( tmp ) ; i ++ ) { while ( tmp % i == 0 ) { cnt ++ ; tmp /= i ; } } if ( tmp > 1 ) cnt ++ ;
if ( val == n ) Console . Write ( " No " ) ; else if ( n / tmp == 2 && cnt == 1 ) Console . Write ( " No " ) ;
else Console . ( " Yes " ) ; } }
public static void Main ( string [ ] args ) { int n = 1 , k = 1 ; findWinner ( n , k ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void pen_hex ( long n ) { long pn = 1 ; for ( long i = 1 ; ; i ++ ) {
pn = i * ( 3 * i - 1 ) / 2 ; if ( pn > n ) break ;
double seqNum = ( 1 + Math . Sqrt ( 8 * pn + 1 ) ) / 4 ; if ( seqNum == ( long ) ( seqNum ) ) { Console . Write ( pn + " , ▁ " ) ; } } }
public static void Main ( string [ ] args ) { long N = 1000000 ; pen_hex ( N ) ; } }
using System ; class GFG {
static bool isPal ( int [ , ] a , int n , int m ) {
for ( int i = 0 ; i < n / 2 ; i ++ ) { for ( int j = 0 ; j < m - 1 ; j ++ ) { if ( a [ i , j ] != a [ n - 1 - i , m - 1 - j ] ) return false ; } } return true ; }
public static void Main ( String [ ] args ) { int n = 3 , m = 3 ; int [ , ] a = { { 1 , 2 , 3 } , { 4 , 5 , 4 } , { 3 , 2 , 1 } } ; if ( isPal ( a , n , m ) ) { Console . Write ( " YES " + " STRNEWLINE " ) ; } else { Console . Write ( " NO " + " STRNEWLINE " ) ; } } }
using System ; class GFG {
static int getSum ( int n ) { int sum = 0 ; while ( n != 0 ) { sum = sum + n % 10 ; n = n / 10 ; } return sum ; }
static void smallestNumber ( int N ) { int i = 1 ; while ( 1 != 0 ) {
if ( getSum ( i ) == N ) { Console . Write ( i ) ; break ; } i ++ ; } }
public static void Main ( String [ ] args ) { int N = 10 ; smallestNumber ( N ) ; } }
using System ; class GFG {
static int reversDigits ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; } return rev_num ; }
static bool isPerfectSquare ( double x ) {
double sr = Math . Sqrt ( x ) ;
return ( ( sr - Math . Floor ( sr ) ) == 0 ) ; }
static bool isRare ( int N ) {
int reverseN = reversDigits ( N ) ;
if ( reverseN == N ) return false ; return isPerfectSquare ( N + reverseN ) && isPerfectSquare ( N - reverseN ) ; }
public static void Main ( String [ ] args ) { int n = 65 ; if ( isRare ( n ) ) { Console . WriteLine ( " Yes " ) ; } else { Console . WriteLine ( " No " ) ; } } }
using System ; using System . Collections . Generic ; class GFG {
static void calc_ans ( int l , int r ) { List < int > power2 = new List < int > ( ) , power3 = new List < int > ( ) ;
int mul2 = 1 ; while ( mul2 <= r ) { power2 . Add ( mul2 ) ; mul2 *= 2 ; }
int mul3 = 1 ; while ( mul3 <= r ) { power3 . Add ( mul3 ) ; mul3 *= 3 ; }
List < int > power23 = new List < int > ( ) ; for ( int x = 0 ; x < power2 . Count ; x ++ ) { for ( int y = 0 ; y < power3 . Count ; y ++ ) { int mul = power2 [ x ] * power3 [ y ] ; if ( mul == 1 ) continue ;
if ( mul <= r ) power23 . Add ( mul ) ; } }
int ans = 0 ; foreach ( int x in power23 ) { if ( x >= l && x <= r ) ans ++ ; }
Console . Write ( ans + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { int l = 1 , r = 10 ; calc_ans ( l , r ) ; } }
using System ; class GFG {
static int nCr ( int n , int r ) { if ( r > n ) return 0 ; return fact ( n ) / ( fact ( r ) * fact ( n - r ) ) ; }
static int fact ( int n ) { int res = 1 ; for ( int i = 2 ; i <= n ; i ++ ) res = res * i ; return res ; }
static int countSubsequences ( int [ ] arr , int n , int k ) { int countOdd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] % 2 == 1 ) countOdd ++ ; } int ans = nCr ( n , k ) - nCr ( countOdd , k ) ; return ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 4 } ; int K = 1 ; int N = arr . Length ; Console . WriteLine ( countSubsequences ( arr , N , K ) ) ; } }
using System ; class GFG {
static void first_digit ( int x , int y ) {
int length = ( int ) ( Math . Log ( x ) / Math . Log ( y ) + 1 ) ;
int first_digit = ( int ) ( x / Math . Pow ( y , length - 1 ) ) ; Console . Write ( first_digit ) ; }
public static void Main ( ) { int X = 55 , Y = 3 ; first_digit ( X , Y ) ; } }
using System ; class GFG {
static void checkIfCurzonNumber ( long N ) { double powerTerm , productTerm ;
powerTerm = Math . Pow ( 2 , N ) + 1 ;
productTerm = 2 * N + 1 ;
if ( powerTerm % productTerm == 0 ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; }
static public void Main ( ) { long N = 5 ; checkIfCurzonNumber ( N ) ; N = 10 ; checkIfCurzonNumber ( N ) ; } }
using System ; class GFG {
static int minCount ( int n ) {
int [ ] hasharr = { 10 , 3 , 6 , 9 , 2 , 5 , 8 , 1 , 4 , 7 } ;
if ( n > 69 ) return hasharr [ n % 10 ] ; else {
if ( n >= hasharr [ n % 10 ] * 7 ) return ( hasharr [ n % 10 ] ) ; else return - 1 ; } }
public static void Main ( String [ ] args ) { int n = 38 ; Console . WriteLine ( minCount ( n ) ) ; } }
using System ; class GFG {
static void modifiedBinaryPattern ( int n ) {
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) {
if ( j == 1 j == i ) Console . Write ( 1 ) ;
else Console . ( 0 ) ; }
Console . WriteLine ( ) ; } }
public static void Main ( ) { int n = 7 ;
modifiedBinaryPattern ( n ) ; } }
using System ; class GFG {
static void findRealAndImag ( String s ) {
int l = s . Length ;
int i ;
if ( s . IndexOf ( ' + ' ) != - 1 ) { i = s . IndexOf ( ' + ' ) ; }
else { i = s . IndexOf ( ' - ' ) ; }
String real = s . Substring ( 0 , i ) ;
String imaginary = s . Substring ( i + 1 , l - i - 2 ) ; Console . WriteLine ( " Real ▁ part : ▁ " + real ) ; Console . WriteLine ( " Imaginary ▁ part : ▁ " + imaginary ) ; }
public static void Main ( String [ ] args ) { String s = "3 + 4i " ; findRealAndImag ( s ) ; } }
using System ; public class GFG {
static int highestPower ( int n , int k ) { int i = 0 ; int a = ( int ) Math . Pow ( n , i ) ;
while ( a <= k ) { i += 1 ; a = ( int ) Math . Pow ( n , i ) ; } return i - 1 ; }
static int [ ] b = new int [ 50 ] ;
static int PowerArray ( int n , int k ) { while ( k > 0 ) {
int t = highestPower ( n , k ) ;
if ( b [ t ] > 0 ) {
Console . Write ( - 1 ) ; return 0 ; }
b [ t ] = 1 ;
k -= ( int ) Math . Pow ( n , t ) ; }
for ( int i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] > 0 ) { Console . Write ( i + " , ▁ " ) ; } } return 0 ; }
public static void Main ( String [ ] args ) { int N = 3 ; int K = 40 ; PowerArray ( N , K ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int N = 10005 ;
static void SieveOfEratosthenes ( List < Boolean > composite ) { for ( int i = 0 ; i < N ; i ++ ) { composite . Insert ( i , false ) ; } for ( int p = 2 ; p * p < N ; p ++ ) {
if ( ! composite [ p ] ) {
for ( int i = p * 2 ; i < N ; i += p ) { composite . Insert ( i , true ) ; } } } }
static int sumOfElements ( int [ ] arr , int n ) { List < Boolean > composite = new List < Boolean > ( ) ; for ( int i = 0 ; i < N ; i ++ ) composite . Add ( false ) ; SieveOfEratosthenes ( composite ) ;
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) if ( mp . ContainsKey ( arr [ i ] ) ) { mp [ arr [ i ] ] = mp [ arr [ i ] ] + 1 ; } else { mp . Add ( arr [ i ] , 1 ) ; }
int sum = 0 ;
foreach ( KeyValuePair < int , int > it in mp ) {
if ( composite [ it . Value ] ) { sum += ( it . Key ) ; } } return sum ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 1 , 1 , 1 , 3 , 3 , 2 , 4 } ; int n = arr . Length ;
Console . Write ( sumOfElements ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void remove ( int [ ] arr , int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( mp . ContainsKey ( arr [ i ] ) ) { mp [ arr [ i ] ] = mp [ arr [ i ] ] + 1 ; } else { mp . Add ( arr [ i ] , 1 ) ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( ( mp . ContainsKey ( arr [ i ] ) && mp [ arr [ i ] ] % 2 == 1 ) ) continue ; Console . Write ( arr [ i ] + " , ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 3 , 3 , 2 , 2 , 4 , 7 , 7 } ; int n = arr . Length ;
remove ( arr , n ) ; } }
using System ; class GFG {
static void getmax ( int [ ] arr , int n , int x ) {
int s = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { s = s + arr [ i ] ; }
Console . WriteLine ( Math . Min ( s , x ) ) ; }
static void Main ( ) { int [ ] arr = new int [ ] { 1 , 2 , 3 , 4 } ; int x = 5 ; int arr_size = arr . Length ; getmax ( arr , arr_size , x ) ; } }
using System ; class GFG {
static void shortestLength ( int n , int [ ] x , int [ ] y ) { int answer = 0 ;
int i = 0 ; while ( n != 0 && i < x . Length ) {
if ( x [ i ] + y [ i ] > answer ) answer = x [ i ] + y [ i ] ; i ++ ; }
Console . WriteLine ( " Length ▁ - > ▁ " + answer ) ; Console . WriteLine ( " Path ▁ - > ▁ " + " ( ▁ 1 , ▁ " + answer + " ▁ ) " + " and ▁ ( ▁ " + answer + " , ▁ 1 ▁ ) " ) ; }
static public void Main ( ) {
int n = 4 ;
int [ ] x = new int [ ] { 1 , 4 , 2 , 1 } ; int [ ] y = new int [ ] { 4 , 1 , 1 , 2 } ; shortestLength ( n , x , y ) ; } }
using System ; class GFG {
static void FindPoints ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 , int x4 , int y4 ) {
int x5 = Math . Max ( x1 , x3 ) ; int y5 = Math . Max ( y1 , y3 ) ;
int x6 = Math . Min ( x2 , x4 ) ; int y6 = Math . Min ( y2 , y4 ) ;
if ( x5 > x6 y5 > y6 ) { Console . WriteLine ( " No ▁ intersection " ) ; return ; } Console . Write ( " ( " + x5 + " , ▁ " + y5 + " ) ▁ " ) ; Console . Write ( " ( " + x6 + " , ▁ " + y6 + " ) ▁ " ) ;
int x7 = x5 ; int y7 = y6 ; Console . Write ( " ( " + x7 + " , ▁ " + y7 + " ) ▁ " ) ;
int x8 = x6 ; int y8 = y5 ; Console . Write ( " ( " + x8 + " , ▁ " + y8 + " ) ▁ " ) ; }
public static void Main ( ) {
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x3 = 2 , y3 = 3 , x4 = 7 , y4 = 9 ;
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) ; } }
using System ; class GFG {
public class Point { public float x , y ; public Point ( ) { x = y = 0 ; } public Point ( float a , float b ) { x = a ; y = b ; } } ;
static void printCorners ( Point p , Point q , float l ) { Point a = new Point ( ) , b = new Point ( ) , c = new Point ( ) , d = new Point ( ) ;
if ( p . x == q . x ) { a . x = ( float ) ( p . x - ( l / 2.0 ) ) ; a . y = p . y ; d . x = ( float ) ( p . x + ( l / 2.0 ) ) ; d . y = p . y ; b . x = ( float ) ( q . x - ( l / 2.0 ) ) ; b . y = q . y ; c . x = ( float ) ( q . x + ( l / 2.0 ) ) ; c . y = q . y ; }
else if ( p . y == q . y ) { a . y = ( float ) ( p . y - ( l / 2.0 ) ) ; a . x = p . x ; d . y = ( float ) ( p . y + ( l / 2.0 ) ) ; d . x = p . x ; b . y = ( float ) ( q . y - ( l / 2.0 ) ) ; b . x = q . x ; c . y = ( float ) ( q . y + ( l / 2.0 ) ) ; c . x = q . x ; }
else {
float m = ( p . x - q . x ) / ( q . y - p . y ) ;
float dx = ( float ) ( ( l / Math . Sqrt ( 1 + ( m * m ) ) ) * 0.5 ) ; float dy = m * dx ; a . x = p . x - dx ; a . y = p . y - dy ; d . x = p . x + dx ; d . y = p . y + dy ; b . x = q . x - dx ; b . y = q . y - dy ; c . x = q . x + dx ; c . y = q . y + dy ; } Console . Write ( ( int ) a . x + " , ▁ " + ( int ) a . y + " ▁ STRNEWLINE " + ( int ) b . x + " , ▁ " + ( int ) b . y + " STRNEWLINE " + ( int ) c . x + " , ▁ " + ( int ) c . y + " ▁ STRNEWLINE " + ( int ) d . x + " , ▁ " + ( int ) d . y + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { Point p1 = new Point ( 1 , 0 ) , q1 = new Point ( 1 , 2 ) ; printCorners ( p1 , q1 , 2 ) ; Point p = new Point ( 1 , 1 ) , q = new Point ( - 1 , - 1 ) ; printCorners ( p , q , ( float ) ( 2 * Math . Sqrt ( 2 ) ) ) ; } }
using System ; public class GFG {
public static int minimumCost ( int [ ] arr , int N , int X , int Y ) {
int even_count = 0 , odd_count = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( ( arr [ i ] & 1 ) > 0 && ( i % 2 == 0 ) ) { odd_count ++ ; }
if ( ( arr [ i ] % 2 ) == 0 && ( i & 1 ) > 0 ) { even_count ++ ; } }
int cost1 = X * Math . Min ( odd_count , even_count ) ;
int cost2 = Y * ( Math . Max ( odd_count , even_count ) - Math . Min ( odd_count , even_count ) ) ;
int cost3 = ( odd_count + even_count ) * Y ;
return Math . Min ( cost1 + cost2 , cost3 ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 5 , 3 , 7 , 2 , 1 } ; int X = 10 , Y = 2 ; int N = arr . Length ; Console . WriteLine ( minimumCost ( arr , N , X , Y ) ) ; } }
using System ; public class GFG {
static int findMinMax ( int [ ] a ) {
int min_val = 1000000000 ;
for ( int i = 1 ; i < a . Length ; ++ i ) {
min_val = Math . Min ( min_val , a [ i ] * a [ i - 1 ] ) ; }
return min_val ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 6 , 4 , 5 , 6 , 2 , 4 , 1 } ; Console . WriteLine ( findMinMax ( arr ) ) ; } }
using System ; public class GFG { static int sum ;
public class TreeNode { public int data ; public TreeNode left ; public TreeNode right ;
public TreeNode ( int data ) { this . data = data ; this . left = null ; this . right = null ; } } ;
static void kDistanceDownSum ( TreeNode root , int k ) {
if ( root == null k < 0 ) return ;
if ( k == 0 ) { sum += root . data ; return ; }
kDistanceDownSum ( root . left , k - 1 ) ; kDistanceDownSum ( root . right , k - 1 ) ; }
static int kDistanceSum ( TreeNode root , int target , int k ) {
if ( root == null ) return - 1 ;
if ( root . data == target ) { kDistanceDownSum ( root . left , k - 1 ) ; return 0 ; }
int dl = - 1 ;
if ( target < root . data ) { dl = kDistanceSum ( root . left , target , k ) ; }
if ( dl != - 1 ) {
if ( dl + 1 == k ) sum += root . data ;
return - 1 ; }
int dr = - 1 ; if ( target > root . data ) { dr = kDistanceSum ( root . right , target , k ) ; } if ( dr != - 1 ) {
if ( dr + 1 == k ) sum += root . data ;
else kDistanceDownSum ( root . left , k - dr - 2 ) ; return 1 + dr ; }
return - 1 ; }
static TreeNode insertNode ( int data , TreeNode root ) {
if ( root == null ) { TreeNode node = new TreeNode ( data ) ; return node ; }
else if ( data > root . data ) { root . right = insertNode ( data , root . right ) ; }
else if ( data < = root . data ) { root . left = insertNode ( data , root . left ) ; }
return root ; }
static void findSum ( TreeNode root , int target , int K ) {
sum = 0 ; kDistanceSum ( root , target , K ) ;
Console . Write ( sum ) ; }
public static void Main ( String [ ] args ) { TreeNode root = null ; int N = 11 ; int [ ] tree = { 3 , 1 , 7 , 0 , 2 , 5 , 10 , 4 , 6 , 9 , 8 } ;
for ( int i = 0 ; i < N ; i ++ ) { root = insertNode ( tree [ i ] , root ) ; } int target = 7 ; int K = 2 ; findSum ( root , target , K ) ; } }
using System ; public class GFG {
static int itemType ( int n ) {
int count = 0 ;
for ( int day = 1 ; ; day ++ ) {
for ( int type = day ; type > 0 ; type -- ) { count += type ;
if ( count >= n ) return type ; } } }
static public void Main ( ) { int N = 10 ; Console . WriteLine ( itemType ( N ) ) ; } }
using System ; class GFG {
static int FindSum ( int [ ] arr , int N ) {
int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int power = ( int ) ( Math . Log ( arr [ i ] ) / Math . Log ( 2 ) ) ;
int LesserValue = ( int ) Math . Pow ( 2 , power ) ;
int LargerValue = ( int ) Math . Pow ( 2 , power + 1 ) ;
if ( ( arr [ i ] - LesserValue ) == ( LargerValue - arr [ i ] ) ) {
res += arr [ i ] ; } }
return res ; }
public static void Main ( ) { int [ ] arr = { 10 , 24 , 17 , 3 , 8 } ; int N = arr . Length ; Console . WriteLine ( FindSum ( arr , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void findLast ( int [ , ] mat ) { int m = 3 ; int n = 3 ;
HashSet < int > rows = new HashSet < int > ( ) ; HashSet < int > cols = new HashSet < int > ( ) ; for ( int i = 0 ; i < m ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( ( mat [ i , j ] > 0 ) ) { rows . Add ( i ) ; cols . Add ( j ) ; } } }
int avRows = m - rows . Count ; int avCols = n - cols . Count ;
int choices = Math . Min ( avRows , avCols ) ;
if ( ( choices & 1 ) != 0 )
Console . WriteLine ( " P1" ) ;
else Console . ( " P2" ) ; }
static public void Main ( ) { int [ , ] mat = { { 1 , 0 , 0 } , { 0 , 0 , 0 } , { 0 , 0 , 1 } } ; findLast ( mat ) ; } }
using System ; class GFG { const int MOD = 1000000007 ;
static void sumOfBinaryNumbers ( int n ) {
int ans = 0 ; int one = 1 ;
while ( true ) {
if ( n <= 1 ) { ans = ( ans + n ) % MOD ; break ; }
int x = ( int ) Math . Log ( n , 2 ) ; int cur = 0 ; int add = ( one << ( x - 1 ) ) ;
for ( int i = 1 ; i <= x ; i ++ ) {
cur = ( cur + add ) % MOD ; add = ( add * 10 % MOD ) ; }
ans = ( ans + cur ) % MOD ;
int rem = n - ( one << x ) + 1 ;
int p = ( int ) Math . Pow ( 10 , x ) ; p = ( p * ( rem % MOD ) ) % MOD ; ans = ( ans + p ) % MOD ;
n = rem - 1 ; }
Console . WriteLine ( ans ) ; }
public static void Main ( ) { int N = 3 ; sumOfBinaryNumbers ( N ) ; } }
using System ; class GFG {
static void nearestFibonacci ( int num ) {
if ( num == 0 ) { Console . Write ( 0 ) ; return ; }
int first = 0 , second = 1 ;
int third = first + second ;
while ( third <= num ) {
first = second ;
second = third ;
third = first + second ; }
int ans = ( Math . Abs ( third - num ) >= Math . Abs ( second - num ) ) ? second : third ;
Console . Write ( ans ) ; }
public static void Main ( string [ ] args ) { int N = 17 ; nearestFibonacci ( N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool checkPermutation ( int [ ] ans , int [ ] a , int n ) {
int Max = Int32 . MinValue ;
for ( int i = 0 ; i < n ; i ++ ) {
Max = Math . Max ( Max , ans [ i ] ) ;
if ( Max != a [ i ] ) return false ; }
return true ; }
static void findPermutation ( int [ ] a , int n ) {
int [ ] ans = new int [ n ] ;
Dictionary < int , int > um = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( ! um . ContainsKey ( a [ i ] ) ) {
ans [ i ] = a [ i ] ; um [ a [ i ] ] = i ; } }
List < int > v = new List < int > ( ) ; int j = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) {
if ( ! um . ContainsKey ( i ) ) { v . Add ( i ) ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( ans [ i ] == 0 ) { ans [ i ] = v [ j ] ; j ++ ; } }
if ( checkPermutation ( ans , a , n ) ) {
for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( ans [ i ] + " ▁ " ) ; } }
else Console . ( " - 1" ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 3 , 4 , 5 , 5 } ; int N = arr . Length ;
findPermutation ( arr , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void countEqualElementPairs ( int [ ] arr , int N ) {
Dictionary < int , int > map = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { if ( ! map . ContainsKey ( arr [ i ] ) ) map [ arr [ i ] ] = 1 ; else map [ arr [ i ] ] ++ ; }
int total = 0 ;
foreach ( KeyValuePair < int , int > e in map ) {
total += ( e . Value * ( e . Value - 1 ) ) / 2 ; }
for ( int i = 0 ; i < N ; i ++ ) {
Console . Write ( total - ( map [ arr [ i ] ] - 1 ) + " ▁ " ) ; } }
public static void Main ( ) {
int [ ] arr = { 1 , 1 , 2 , 1 , 2 } ;
int N = 5 ; countEqualElementPairs ( arr , N ) ; } }
using System ; class GFG {
static int count ( int N ) { int sum = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { sum += ( int ) ( 7 * Math . Pow ( 8 , i - 1 ) ) ; } return sum ; }
public static void Main ( ) { int N = 4 ; Console . WriteLine ( count ( N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool isPalindrome ( int n ) {
String str = String . Join ( " " , n ) ;
int s = 0 , e = str . Length - 1 ; while ( s < e ) {
if ( str [ s ] != str [ e ] ) { return false ; } s ++ ; e -- ; } return true ; }
static void palindromicDivisors ( int n ) {
List < int > PalindromDivisors = new List < int > ( ) ; for ( int i = 1 ; i <= Math . Sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) {
if ( n / i == i ) {
if ( isPalindrome ( i ) ) { PalindromDivisors . Add ( i ) ; } } else {
if ( isPalindrome ( i ) ) { PalindromDivisors . Add ( i ) ; }
if ( isPalindrome ( n / i ) ) { PalindromDivisors . Add ( n / i ) ; } } } }
PalindromDivisors . Sort ( ) ; for ( int i = 0 ; i < PalindromDivisors . Count ; i ++ ) { Console . Write ( PalindromDivisors [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int n = 66 ;
palindromicDivisors ( n ) ; } }
using System ; class GFG {
static int findMinDel ( int [ ] arr , int n ) {
int min_num = int . MaxValue ;
for ( int i = 0 ; i < n ; i ++ ) min_num = Math . Min ( arr [ i ] , min_num ) ;
int cnt = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] == min_num ) cnt ++ ;
return n - cnt ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 3 , 2 } ; int n = arr . Length ; Console . Write ( findMinDel ( arr , n ) ) ; } }
using System ; class GFG {
static int cntSubArr ( int [ ] arr , int n ) {
int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int curr_gcd = 0 ;
for ( int j = i ; j < n ; j ++ ) { curr_gcd = __gcd ( curr_gcd , arr [ j ] ) ;
ans += ( curr_gcd == 1 ) ? 1 : 0 ; } }
return ans ; } static int __gcd ( int a , int b ) { if ( b == 0 ) return a ; return __gcd ( b , a % b ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 1 , 1 } ; int n = arr . Length ; Console . WriteLine ( cntSubArr ( arr , n ) ) ; } }
using System ; class GFG {
static void print_primes_till_N ( int N ) {
int i , j , flag ;
Console . Write ( " Prime ▁ numbers ▁ between ▁ 1 ▁ and ▁ " + N + " ▁ are : STRNEWLINE " ) ;
for ( i = 1 ; i <= N ; i ++ ) {
if ( i == 1 i == 0 ) continue ;
flag = 1 ; for ( j = 2 ; j <= i / 2 ; ++ j ) { if ( i % j == 0 ) { flag = 0 ; break ; } }
if ( flag == 1 ) Console . Write ( i + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int N = 100 ; print_primes_till_N ( N ) ; } }
using System ; class GFG { static int MAX = 32 ;
static int findX ( int A , int B ) { int X = 0 ;
for ( int bit = 0 ; bit < MAX ; bit ++ ) {
int tempBit = 1 << bit ;
int bitOfX = A & B & tempBit ;
X += bitOfX ; } return X ; }
public static void Main ( String [ ] args ) { int A = 11 , B = 13 ; Console . WriteLine ( findX ( A , B ) ) ; } }
using System ; using System . Linq ; class GFG {
static int cntSubSets ( int [ ] arr , int n ) {
int maxVal = arr . Max ( ) ;
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == maxVal ) cnt ++ ; }
return ( int ) ( Math . Pow ( 2 , cnt ) - 1 ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 1 , 2 } ; int n = arr . Length ; Console . WriteLine ( cntSubSets ( arr , n ) ) ; } }
using System ; class GFG {
static float findProb ( int [ ] arr , int n ) {
long maxSum = int . MinValue , maxCount = 0 , totalPairs = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
int sum = arr [ i ] + arr [ j ] ;
if ( sum == maxSum ) {
maxCount ++ ; }
else if ( sum > maxSum ) {
maxSum = sum ; maxCount = 1 ; } totalPairs ++ ; } }
float prob = ( float ) maxCount / ( float ) totalPairs ; return prob ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 1 , 1 , 2 , 2 , 2 } ; int n = arr . Length ; Console . WriteLine ( findProb ( arr , n ) ) ; } }
using System ; class GFG { static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
static int maxCommonFactors ( int a , int b ) {
int __gcd = gcd ( a , b ) ;
int ans = 1 ;
for ( int i = 2 ; i * i <= __gcd ; i ++ ) { if ( __gcd % i == 0 ) { ans ++ ; while ( __gcd % i == 0 ) __gcd /= i ; } }
if ( __gcd != 1 ) ans ++ ;
return ans ; }
public static void Main ( String [ ] args ) { int a = 12 , b = 18 ; Console . WriteLine ( maxCommonFactors ( a , b ) ) ; } }
using System ; class GFG { static int [ ] days = { 31 , 28 , 31 , 30 , 31 , 30 , 31 , 31 , 30 , 31 , 30 , 31 } ;
static int dayOfYear ( string date ) {
int year = Int32 . Parse ( date . Substring ( 0 , 4 ) ) ; int month = Int32 . Parse ( date . Substring ( 5 , 2 ) ) ; int day = Int32 . Parse ( date . Substring ( 8 ) ) ;
if ( month > 2 && year % 4 == 0 && ( year % 100 != 0 year % 400 == 0 ) ) { ++ day ; }
while ( -- month > 0 ) { day = day + days [ month - 1 ] ; } return day ; }
public static void Main ( ) { String date = "2019-01-09" ; Console . WriteLine ( dayOfYear ( date ) ) ; } }
using System ; class GFG {
static int Cells ( int n , int x ) { int ans = 0 ; for ( int i = 1 ; i <= n ; i ++ ) if ( x % i == 0 && x / i <= n ) ans ++ ; return ans ; }
public static void Main ( ) { int n = 6 , x = 12 ;
Console . WriteLine ( Cells ( n , x ) ) ; } }
using System ; class GFG {
static int nextPowerOfFour ( int n ) { int x = ( int ) Math . Floor ( Math . Sqrt ( Math . Sqrt ( n ) ) ) ;
if ( Math . Pow ( x , 4 ) == n ) return n ; else { x = x + 1 ; return ( int ) Math . Pow ( x , 4 ) ; } }
public static void Main ( ) { int n = 122 ; Console . WriteLine ( nextPowerOfFour ( n ) ) ; } }
using System ; class GFG {
static int minOperations ( int x , int y , int p , int q ) {
if ( y % x != 0 ) return - 1 ; int d = y / x ;
int a = 0 ;
while ( d % p == 0 ) { d /= p ; a ++ ; }
int b = 0 ;
while ( d % q == 0 ) { d /= q ; b ++ ; }
if ( d != 1 ) return - 1 ;
return ( a + b ) ; }
public static void Main ( ) { int x = 12 , y = 2592 , p = 2 , q = 3 ; Console . Write ( minOperations ( x , y , p , q ) ) ; } }
using System ; class GFG {
static int nCr ( int n ) {
if ( n < 4 ) return 0 ; int answer = n * ( n - 1 ) * ( n - 2 ) * ( n - 3 ) ; answer /= 24 ; return answer ; }
static int countQuadruples ( int N , int K ) {
int M = N / K ; int answer = nCr ( M ) ;
for ( int i = 2 ; i < M ; i ++ ) { int j = i ;
int temp2 = M / i ;
int count = 0 ;
int check = 0 ; int temp = j ; while ( j % 2 == 0 ) { count ++ ; j /= 2 ; if ( count >= 2 ) break ; } if ( count >= 2 ) { check = 1 ; } for ( int k = 3 ; k <= Math . Sqrt ( temp ) ; k += 2 ) { int cnt = 0 ; while ( j % k == 0 ) { cnt ++ ; j /= k ; if ( cnt >= 2 ) break ; } if ( cnt >= 2 ) { check = 1 ; break ; } else if ( cnt == 1 ) count ++ ; } if ( j > 2 ) { count ++ ; }
if ( check == 1 ) continue ; else {
if ( count % 2 == 1 ) { answer -= nCr ( temp2 ) ; } else { answer += nCr ( temp2 ) ; } } } return answer ; }
public static void Main ( String [ ] args ) { int N = 10 , K = 2 ; Console . WriteLine ( countQuadruples ( N , K ) ) ; } }
using System ; class GFG {
static int getX ( int a , int b , int c , int d ) { int X = ( b * c - a * d ) / ( d - c ) ; return X ; }
static public void Main ( ) { int a = 2 , b = 3 , c = 4 , d = 5 ; Console . Write ( getX ( a , b , c , d ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool isVowel ( char ch ) { if ( ch == ' a ' ch == ' e ' ch == ' i ' ch == ' o ' ch == ' u ' ) return true ; else return false ; }
static long fact ( long n ) { if ( n < 2 ) { return 1 ; } return n * fact ( n - 1 ) ; }
static long only_vowels ( Dictionary < char , int > freq ) { long denom = 1 ; long cnt_vwl = 0 ;
foreach ( KeyValuePair < char , int > itr in freq ) { if ( isVowel ( itr . Key ) ) { denom *= fact ( itr . Value ) ; cnt_vwl += itr . Value ; } } return fact ( cnt_vwl ) / denom ; }
static long all_vowels_together ( Dictionary < char , int > freq ) {
long vow = only_vowels ( freq ) ;
long denom = 1 ;
long cnt_cnst = 0 ; foreach ( KeyValuePair < char , int > itr in freq ) { if ( ! isVowel ( itr . Key ) ) { denom *= fact ( itr . Value ) ; cnt_cnst += itr . Value ; } }
long ans = fact ( cnt_cnst + 1 ) / denom ; return ( ans * vow ) ; }
static long total_permutations ( Dictionary < char , int > freq ) {
long cnt = 0 ;
long denom = 1 ; foreach ( KeyValuePair < char , int > itr in freq ) { denom *= fact ( itr . Value ) ; cnt += itr . Value ; }
return fact ( cnt ) / denom ; }
static long no_vowels_together ( string word ) {
Dictionary < char , int > freq = new Dictionary < char , int > ( ) ;
for ( int i = 0 ; i < word . Length ; i ++ ) { char ch = Char . ToLower ( word [ i ] ) ; if ( freq . ContainsKey ( ch ) ) { freq [ ch ] ++ ; } else { freq [ ch ] = 1 ; } }
long total = total_permutations ( freq ) ;
long vwl_tgthr = all_vowels_together ( freq ) ;
long res = total - vwl_tgthr ;
return res ; }
static void Main ( ) { string word = " allahabad " ; long ans = no_vowels_together ( word ) ; Console . WriteLine ( ans ) ; word = " geeksforgeeks " ; ans = no_vowels_together ( word ) ; Console . WriteLine ( ans ) ; word = " abcd " ; ans = no_vowels_together ( word ) ; Console . WriteLine ( ans ) ; } }
using System ; class GFG {
static int numberOfMen ( int D , int m , int d ) { int Men = ( m * ( D - d ) ) / d ; return Men ; }
public static void Main ( ) { int D = 5 , m = 4 , d = 4 ; Console . WriteLine ( numberOfMen ( D , m , d ) ) ; } }
using System ; class GFG {
static double area ( double a , double b , double c ) { double d = Math . Abs ( ( c * c ) / ( 2 * a * b ) ) ; return d ; }
static public void Main ( ) { double a = - 2 , b = 4 , c = 3 ; Console . WriteLine ( area ( a , b , c ) ) ; } }
using System ; using System . Collections ; class GFG {
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
using System ; class GFG { static int MAX = 100005 ;
static int kadaneAlgorithm ( int [ ] ar , int n ) { int sum = 0 , maxSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) { sum += ar [ i ] ; if ( sum < 0 ) sum = 0 ; maxSum = Math . Max ( maxSum , sum ) ; } return maxSum ; }
static int maxFunction ( int [ ] arr , int n ) { int [ ] b = new int [ MAX ] ; int [ ] c = new int [ MAX ] ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { if ( i % 2 == 1 ) { b [ i ] = Math . Abs ( arr [ i + 1 ] - arr [ i ] ) ; c [ i ] = - b [ i ] ; } else { c [ i ] = Math . Abs ( arr [ i + 1 ] - arr [ i ] ) ; b [ i ] = - c [ i ] ; } }
int ans = kadaneAlgorithm ( b , n - 1 ) ; ans = Math . Max ( ans , kadaneAlgorithm ( c , n - 1 ) ) ; return ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 5 , 4 , 7 } ; int n = arr . Length ; Console . WriteLine ( maxFunction ( arr , n ) ) ; } }
using System ; class GFG {
static int findThirdDigit ( int n ) {
if ( n < 3 ) return 0 ;
return ( n & 1 ) > 0 ? 1 : 6 ; }
static void Main ( ) { int n = 7 ; Console . WriteLine ( findThirdDigit ( n ) ) ; } }
using System ; class GFG {
public static double getProbability ( int a , int b , int c , int d ) {
double p = ( double ) a / ( double ) b ; double q = ( double ) c / ( double ) d ;
double ans = p * ( 1 / ( 1 - ( 1 - q ) * ( 1 - p ) ) ) ; return ans ; }
public static void Main ( string [ ] args ) { int a = 1 , b = 2 , c = 10 , d = 11 ; Console . Write ( " { 0 : F5 } " , getProbability ( a , b , c , d ) ) ; } }
using System ; class GFG {
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
using System ; public class GFG {
public static long getFinalElement ( long n ) { long finalNum ; for ( finalNum = 2 ; finalNum * 2 <= n ; finalNum *= 2 ) ; return finalNum ; }
static public void Main ( ) { int N = 12 ; Console . WriteLine ( getFinalElement ( N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void SieveOfEratosthenes ( bool [ ] prime , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
static int sumOfElements ( int [ ] arr , int n ) { bool [ ] prime = new bool [ n + 1 ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) prime [ i ] = true ; SieveOfEratosthenes ( prime , n + 1 ) ;
Dictionary < int , int > m = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( m . ContainsKey ( arr [ i ] ) ) { var val = m [ arr [ i ] ] ; m . Remove ( arr [ i ] ) ; m . Add ( arr [ i ] , val + 1 ) ; } else { m . Add ( arr [ i ] , 1 ) ; } } int sum = 0 ;
foreach ( KeyValuePair < int , int > entry in m ) { int key = entry . Key ; int value = entry . Value ;
if ( prime [ value ] ) { sum += ( key ) ; } } return sum ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 4 , 6 , 5 , 4 , 6 } ; int n = arr . Length ; Console . WriteLine ( sumOfElements ( arr , n ) ) ; } }
using System ; public class GFG {
static bool isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
static bool isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
static long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
public static void Main ( String [ ] args ) { int L = 110 , R = 1130 ; Console . WriteLine ( sumOfAllPalindrome ( L , R ) ) ; } }
using System . Collections . Generic ; using System ; class GFG {
static int fact ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f = f * i ; return f ; }
static int waysOfConsonants ( int size1 , int [ ] freq ) { int ans = fact ( size1 ) ; for ( int i = 0 ; i < 26 ; i ++ ) {
if ( i == 0 i == 4 i == 8 i == 14 i == 20 ) continue ; else ans = ans / fact ( freq [ i ] ) ; } return ans ; }
static int waysOfVowels ( int size2 , int [ ] freq ) { return fact ( size2 ) / ( fact ( freq [ 0 ] ) * fact ( freq [ 4 ] ) * fact ( freq [ 8 ] ) * fact ( freq [ 14 ] ) * fact ( freq [ 20 ] ) ) ; }
static int countWays ( string str ) { int [ ] freq = new int [ 200 ] ; for ( int i = 0 ; i < 200 ; i ++ ) freq [ i ] = 0 ; for ( int i = 0 ; i < str . Length ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
int vowel = 0 , consonant = 0 ; for ( int i = 0 ; i < str . Length ; i ++ ) { if ( str [ i ] != ' a ' && str [ i ] != ' e ' && str [ i ] != ' i ' && str [ i ] != ' o ' && str [ i ] != ' u ' ) consonant ++ ; else vowel ++ ; }
return waysOfConsonants ( consonant + 1 , freq ) * waysOfVowels ( vowel , freq ) ; }
public static void Main ( ) { string str = " geeksforgeeks " ; Console . WriteLine ( countWays ( str ) ) ; } }
using System ; class GFG {
static double calculateAlternateSum ( int n ) { if ( n <= 0 ) return 0 ; int [ ] fibo = new int [ n + 1 ] ; fibo [ 0 ] = 0 ; fibo [ 1 ] = 1 ;
double sum = Math . Pow ( fibo [ 0 ] , 2 ) + Math . Pow ( fibo [ 1 ] , 2 ) ;
for ( int i = 2 ; i <= n ; i ++ ) { fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] ;
if ( i % 2 == 0 ) sum -= fibo [ i ] ;
else sum += [ i ] ; }
return sum ; }
public static void Main ( ) {
int n = 8 ;
Console . WriteLine ( " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " + n + " ▁ terms : ▁ " + calculateAlternateSum ( n ) ) ; } }
using System ; class GFG {
static int getValue ( int n ) { int i = 0 , k = 1 ; while ( i < n ) { i = i + k ; k = k * 2 ; } return k / 2 ; }
public static void Main ( ) {
int n = 9 ;
Console . WriteLine ( getValue ( n ) ) ;
n = 1025 ;
Console . WriteLine ( getValue ( n ) ) ; } }
using System ; class GFG {
static void countDigits ( double val , long [ ] arr ) { while ( ( long ) val > 0 ) { long digit = ( long ) val % 10 ; arr [ ( int ) digit ] ++ ; val = ( long ) val / 10 ; } return ; } static void countFrequency ( int x , int n ) {
long [ ] freq_count = new long [ 10 ] ;
for ( int i = 1 ; i <= n ; i ++ ) {
double val = Math . Pow ( ( double ) x , ( double ) i ) ;
countDigits ( val , freq_count ) ; }
for ( int i = 0 ; i <= 9 ; i ++ ) { Console . Write ( freq_count [ i ] + " ▁ " ) ; } }
public static void Main ( ) { int x = 15 , n = 3 ; countFrequency ( x , n ) ; } }
using System ; class GFG {
static int countSolutions ( int a ) { int count = 0 ;
for ( int i = 0 ; i <= a ; i ++ ) { if ( a == ( i + ( a ^ i ) ) ) count ++ ; } return count ; }
public static void Main ( ) { int a = 3 ; Console . WriteLine ( countSolutions ( a ) ) ; } }
class GFG {
static int countSolutions ( int a ) { int count = bitCount ( a ) ; count = ( int ) System . Math . Pow ( 2 , count ) ; return count ; } static int bitCount ( int n ) { int count = 0 ; while ( n != 0 ) { count ++ ; n &= ( n - 1 ) ; } return count ; }
public static void Main ( ) { int a = 3 ; System . Console . WriteLine ( countSolutions ( a ) ) ; } }
using System ; class GFG {
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
using System ; class GFG {
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
using System ; using System . Collections . Generic ; class GFG {
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
using System ; using System . Collections . Generic ; class GFG {
static void LucasLehmer ( int n ) {
long current_val = 4 ;
List < long > series = new List < long > ( ) ;
series . Add ( current_val ) ; for ( int i = 0 ; i < n ; i ++ ) { current_val = current_val * current_val - 2 ; series . Add ( current_val ) ; }
for ( int i = 0 ; i <= n ; i ++ ) Console . WriteLine ( " Term ▁ " + i + " : ▁ " + series [ i ] ) ; }
static void Main ( ) { int n = 5 ; LucasLehmer ( n ) ; } }
using System ; class GFG {
static int modInverse ( int a , int prime ) { a = a % prime ; for ( int x = 1 ; x < prime ; x ++ ) if ( ( a * x ) % prime == 1 ) return x ; return - 1 ; } static void printModIverses ( int n , int prime ) { for ( int i = 1 ; i <= n ; i ++ ) Console . Write ( modInverse ( i , prime ) + " ▁ " ) ; }
public static void Main ( ) { int n = 10 , prime = 17 ; printModIverses ( n , prime ) ; } }
using System ; class GFG {
static int minOp ( int num ) {
int rem ; int count = 0 ;
while ( num > 0 ) { rem = num % 10 ; if ( ! ( rem == 3 rem == 8 ) ) count ++ ; num /= 10 ; } return count ; }
public static void Main ( ) { int num = 234198 ; Console . WriteLine ( " Minimum ▁ Operations ▁ = " + minOp ( num ) ) ; } }
using System ; class GFG {
static int sumOfDigits ( int a ) { int sum = 0 ; while ( a != 0 ) { sum += a % 10 ; a /= 10 ; } return sum ; }
static int findMax ( int x ) {
int b = 1 , ans = x ;
while ( x != 0 ) {
int cur = ( x - 1 ) * b + ( b - 1 ) ;
if ( sumOfDigits ( cur ) > sumOfDigits ( ans ) || ( sumOfDigits ( cur ) == sumOfDigits ( ans ) && cur > ans ) ) ans = cur ;
x /= 10 ; b *= 10 ; } return ans ; }
public static void Main ( ) { int n = 521 ; Console . WriteLine ( findMax ( n ) ) ; } }
using System ; class GFG {
static int median ( int [ ] a , int l , int r ) { int n = r - l + 1 ; n = ( n + 1 ) / 2 - 1 ; return n + l ; }
static int IQR ( int [ ] a , int n ) { Array . Sort ( a ) ;
int mid_index = median ( a , 0 , n ) ;
int Q1 = a [ median ( a , 0 , mid_index ) ] ;
int Q3 = a [ mid_index + median ( a , mid_index + 1 , n ) ] ;
return ( Q3 - Q1 ) ; }
public static void Main ( ) { int [ ] a = { 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 } ; int n = a . Length ; Console . WriteLine ( IQR ( a , n ) ) ; } }
using System ; class GFG {
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
using System ; class GFG {
static int findSum ( int n , int a , int b ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
static void Main ( ) { int n = 10 , a = 3 , b = 5 ; Console . WriteLine ( findSum ( n , a , b ) ) ; } }
using System ; class GFG { static int subtractOne ( int x ) { return ( ( x << 1 ) + ( ~ x ) ) ; } public static void Main ( String [ ] args ) { Console . Write ( " { 0 } " , subtractOne ( 13 ) ) ; } }
using System ; class PellNumber {
public static int pell ( int n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
public static void Main ( ) { int n = 4 ; Console . Write ( pell ( n ) ) ; } }
using System ; using System . Collections ; class GFG {
static long LCM ( int [ ] arr , int n ) {
int max_num = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( max_num < arr [ i ] ) { max_num = arr [ i ] ; } }
long res = 1 ;
while ( x <= max_num ) {
ArrayList indexes = new ArrayList ( ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] % x == 0 ) { indexes . Add ( j ) ; } }
if ( indexes . Count >= 2 ) {
for ( int j = 0 ; j < indexes . Count ; j ++ ) { arr [ ( int ) indexes [ j ] ] = arr [ ( int ) indexes [ j ] ] / x ; } res = res * x ; } else { x ++ ; } }
for ( int i = 0 ; i < n ; i ++ ) { res = res * arr [ i ] ; } return res ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 } ; int n = arr . Length ; Console . WriteLine ( LCM ( arr , n ) ) ; } }
using System ; public class GFG {
static int politness ( int n ) { int count = 0 ;
for ( int i = 2 ; i <= Math . Sqrt ( 2 * n ) ; i ++ ) { int a ; if ( ( 2 * n ) % i != 0 ) continue ; a = 2 * n ; a /= i ; a -= ( i - 1 ) ; if ( a % 2 != 0 ) continue ; a /= 2 ; if ( a > 0 ) { count ++ ; } } return count ; }
public static void Main ( String [ ] args ) { int n = 90 ; Console . WriteLine ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; n = 15 ; Console . WriteLine ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; } }
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
public static void Main ( String [ ] args ) {
sieveSundaram ( ) ;
findPrimes ( 4 ) ; findPrimes ( 38 ) ; findPrimes ( 100 ) ; } }
using System ; class GFG {
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
static void Main ( ) {
int [ ] s = new int [ MAX + 1 ] ; sieveOfEratosthenes ( s ) ; int n = 12 , k = 3 ; Console . WriteLine ( kPrimeFactor ( n , k , s ) ) ; n = 14 ; k = 3 ; Console . WriteLine ( kPrimeFactor ( n , k , s ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static int sumDivisorsOfDivisors ( int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int j = 2 ; j <= Math . Sqrt ( n ) ; j ++ ) { int count = 0 ; while ( n % j == 0 ) { n /= j ; count ++ ; } if ( count != 0 ) mp . Add ( j , count ) ; }
if ( n != 1 ) mp . Add ( n , 1 ) ;
int ans = 1 ; foreach ( KeyValuePair < int , int > entry in mp ) { int pw = 1 ; int sum = 0 ; for ( int i = entry . Value + 1 ; i >= 1 ; i -- ) { sum += ( i * pw ) ; pw = entry . Key ; } ans *= sum ; } return ans ; }
public static void Main ( String [ ] args ) { int n = 10 ; Console . WriteLine ( sumDivisorsOfDivisors ( n ) ) ; } }
using System ; class GFG {
static int prime ( int n ) {
if ( n % 2 != 0 ) n -= 2 ; else n -- ; int i , j ; for ( i = n ; i >= 2 ; i -= 2 ) { if ( i % 2 == 0 ) continue ; for ( j = 3 ; j <= Math . Sqrt ( i ) ; j += 2 ) { if ( i % j == 0 ) break ; } if ( j > Math . Sqrt ( i ) ) return i ; }
return 2 ; }
public static void Main ( ) { int n = 17 ; Console . Write ( prime ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static string fractionToDecimal ( int numr , int denr ) {
string res = " " ;
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ;
int rem = numr % denr ;
while ( ( rem != 0 ) && ( ! mp . ContainsValue ( rem ) ) ) {
mp [ rem ] = res . Length ;
rem = rem * 10 ;
int res_part = rem / denr ; res += res_part . ToString ( ) ;
rem = rem % denr ; } if ( rem == 0 ) return " " ; else if ( mp . ContainsKey ( rem ) ) return res . Substring ( mp [ rem ] ) ; return " " ; }
public static void Main ( string [ ] args ) { int numr = 50 , denr = 22 ; string res = fractionToDecimal ( numr , denr ) ; if ( res == " " ) Console . Write ( " No ▁ recurring ▁ sequence " ) ; else Console . Write ( " Recurring ▁ sequence ▁ is ▁ " + res ) ; } }
using System ; class GFG {
static int has0 ( int x ) {
while ( x != 0 ) {
if ( x % 10 == 0 ) return 1 ; x /= 10 ; } return 0 ; }
static int getCount ( int n ) {
int count = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) count += has0 ( i ) ; return count ; }
public static void Main ( ) { int n = 107 ; Console . WriteLine ( " Count ▁ of ▁ numbers ▁ from ▁ 1" + " ▁ to ▁ " + n + " ▁ is ▁ " + getCount ( n ) ) ; } }
using System ; class GFG {
static bool squareRootExists ( int n , int p ) { n = n % p ;
for ( int x = 2 ; x < p ; x ++ ) if ( ( x * x ) % p == n ) return true ; return false ; }
public static void Main ( ) { int p = 7 ; int n = 2 ; if ( squareRootExists ( n , p ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; public class GFG {
static int Largestpower ( int n , int p ) {
int ans = 0 ;
while ( n > 0 ) { n /= p ; ans += n ; } return ans ; }
public static void Main ( ) { int n = 10 ; int p = 3 ; Console . Write ( " ▁ The ▁ largest ▁ power ▁ of ▁ " + p + " ▁ that ▁ divides ▁ " + n + " ! ▁ is ▁ " + Largestpower ( n , p ) ) ; } }
using System ; class Factorial { int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
public static void Main ( ) { Factorial obj = new Factorial ( ) ; int num = 5 ; Console . WriteLine ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + obj . factorial ( num ) ) ; } }
static bool getBit ( int num , int i ) {
return ( ( num & ( 1 << i ) ) != 0 ) ; }
static int clearBit ( int num , int i ) {
int mask = ~ ( 1 << i ) ;
return num & mask ; }
using System ; class GFG {
static public void Main ( ) {
int [ ] arr1 = { 1 , 2 , 3 } ;
int [ ] arr2 = { 1 , 2 , 3 } ;
int N = arr1 . Length ;
int M = arr2 . Length ;
Bitwise_AND_sum_i ( arr1 , arr2 , M , N ) ; }
static void Bitwise_AND_sum_i ( int [ ] arr1 , int [ ] arr2 , int M , int N ) {
int [ ] frequency = new int [ 32 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
int bit_position = 0 ; int num = arr1 [ i ] ;
while ( num != 0 ) {
if ( ( num & 1 ) != 0 ) {
frequency [ bit_position ] += 1 ; }
bit_position += 1 ;
num >>= 1 ; } }
for ( int i = 0 ; i < M ; i ++ ) { int num = arr2 [ i ] ;
int value_at_that_bit = 1 ;
int bitwise_AND_sum = 0 ;
for ( int bit_position = 0 ; bit_position < 32 ; bit_position ++ ) {
if ( ( num & 1 ) != 0 ) {
bitwise_AND_sum += frequency [ bit_position ] * value_at_that_bit ; }
num >>= 1 ;
value_at_that_bit <<= 1 ; }
Console . Write ( bitwise_AND_sum + " ▁ " ) ; } } }
using System ; class GFG {
static void FlipBits ( int n ) { for ( int bit = 0 ; bit < 32 ; bit ++ ) {
if ( ( n >> bit ) % 2 > 0 ) {
n = n ^ ( 1 << bit ) ; break ; } } Console . Write ( " The ▁ number ▁ after ▁ unsetting ▁ the ▁ " ) ; Console . Write ( " rightmost ▁ set ▁ bit ▁ " + n ) ; }
static void Main ( ) { int N = 12 ; FlipBits ( N ) ; } }
using System ; class GFG {
static int bitwiseAndOdd ( int n ) {
int result = 1 ;
for ( int i = 3 ; i <= n ; i = i + 2 ) { result = ( result & i ) ; } return result ; }
public static void Main ( ) { int n = 10 ; Console . WriteLine ( bitwiseAndOdd ( n ) ) ; } }
using System ; class GFG {
static int bitwiseAndOdd ( int n ) { return 1 ; }
public static void Main ( ) { int n = 10 ; Console . WriteLine ( bitwiseAndOdd ( n ) ) ; } }
using System ; class GFG {
public static int reverseBits ( int n ) { int rev = 0 ;
while ( n > 0 ) {
rev <<= 1 ;
if ( ( int ) ( n & 1 ) == 1 ) rev ^= 1 ;
n >>= 1 ; }
return rev ; }
public static void Main ( ) { int n = 11 ; Console . WriteLine ( reverseBits ( n ) ) ; } }
using System ; class GFG {
static int countgroup ( int [ ] a , int n ) { int xs = 0 ; for ( int i = 0 ; i < n ; i ++ ) xs = xs ^ a [ i ] ;
if ( xs == 0 ) return ( 1 << ( n - 1 ) ) - 1 ; return 0 ; }
public static void Main ( ) { int [ ] a = { 1 , 2 , 3 } ; int n = a . Length ; Console . WriteLine ( countgroup ( a , n ) ) ; } }
using System ; class GFG {
static int bitExtracted ( int number , int k , int p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
public static void Main ( ) { int number = 171 , k = 5 , p = 2 ; Console . WriteLine ( " The ▁ extracted ▁ number ▁ is ▁ " + bitExtracted ( number , k , p ) ) ; } }
using System ; public class GFG { static int findMax ( int num ) { byte size_of_int = 4 ; int num_copy = num ;
int j = size_of_int * 8 - 1 ; int i = 0 ; while ( i < j ) {
int m = ( num_copy >> i ) & 1 ; int n = ( num_copy >> j ) & 1 ;
if ( m > n ) { int x = ( 1 << i 1 << j ) ; num = num ^ x ; } i ++ ; j -- ; } return num ; }
static public void Main ( ) { int num = 4 ; Console . Write ( findMax ( num ) ) ; } }
using System ; class GFG {
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
using System ; class GFG {
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
using System ; class GFG {
static int composite ( int n ) { int flag = 0 ; int c = 0 ;
for ( int j = 1 ; j <= n ; j ++ ) { if ( n % j == 0 ) { c += 1 ; } }
if ( c >= 3 ) flag = 1 ; return flag ; }
static void odd_indices ( int [ ] arr , int n ) { int sum = 0 ;
for ( int k = 0 ; k < n ; k += 2 ) { int check = composite ( arr [ k ] ) ;
if ( check == 1 ) sum += arr [ k ] ; }
Console . Write ( sum + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 13 , 5 , 8 , 16 , 25 } ; int n = arr . Length ; odd_indices ( arr , n ) ; } }
using System ; class GFG {
public static void preprocess ( int [ ] p , int [ ] x , int [ ] y , int n ) { for ( int i = 0 ; i < n ; i ++ ) p [ i ] = x [ i ] * x [ i ] + y [ i ] * y [ i ] ; Array . Sort ( p ) ; }
public static int query ( int [ ] p , int n , int rad ) { int start = 0 , end = n - 1 ; while ( ( end - start ) > 1 ) { int mid = ( start + end ) / 2 ; double tp = Math . Sqrt ( p [ mid ] ) ; if ( tp > ( rad * 1.0 ) ) end = mid - 1 ; else start = mid ; } double tp1 = Math . Sqrt ( p [ start ] ) ; double tp2 = Math . Sqrt ( p [ end ] ) ; if ( tp1 > ( rad * 1.0 ) ) return 0 ; else if ( tp2 <= ( rad * 1.0 ) ) return end + 1 ; else return start + 1 ; }
public static void Main ( ) { int [ ] x = { 1 , 2 , 3 , - 1 , 4 } ; int [ ] y = { 1 , 2 , 3 , - 1 , 4 } ; int n = x . Length ;
int [ ] p = new int [ n ] ; preprocess ( p , x , y , n ) ;
Console . WriteLine ( query ( p , n , 3 ) ) ;
Console . WriteLine ( query ( p , n , 32 ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int find_Numb_ways ( int n ) {
int odd_indices = n / 2 ;
int even_indices = ( n / 2 ) + ( n % 2 ) ;
int arr_odd = ( int ) Math . Pow ( 4 , odd_indices ) ;
int arr_even = ( int ) Math . Pow ( 5 , even_indices ) ;
return arr_odd * arr_even ; }
public static void Main ( ) { int n = 4 ; Console . Write ( find_Numb_ways ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool isSpiralSorted ( int [ ] arr , int n ) {
int start = 0 ;
int end = n - 1 ; while ( start < end ) {
if ( arr [ start ] > arr [ end ] ) { return false ; }
start ++ ;
if ( arr [ end ] > arr [ start ] ) { return false ; }
end -- ; } return true ; }
static void Main ( ) { int [ ] arr = { 1 , 10 , 14 , 20 , 18 , 12 , 5 } ; int N = arr . Length ;
if ( isSpiralSorted ( arr , N ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void findWordsSameRow ( List < string > arr ) {
Dictionary < char , int > mp = new Dictionary < char , int > ( ) ; mp . Add ( ' q ' , 1 ) ; mp . Add ( ' w ' , 1 ) ; mp . Add ( ' e ' , 1 ) ; mp . Add ( ' r ' , 1 ) ; mp . Add ( ' t ' , 1 ) ; mp . Add ( ' y ' , 1 ) ; mp . Add ( ' u ' , 1 ) ; mp . Add ( ' i ' , 1 ) ; mp . Add ( ' o ' , 1 ) ; mp . Add ( ' p ' , 1 ) ; mp . Add ( ' a ' , 2 ) ; mp . Add ( ' s ' , 2 ) ; mp . Add ( ' d ' , 2 ) ; mp . Add ( ' f ' , 2 ) ; mp . Add ( ' g ' , 2 ) ; mp . Add ( ' h ' , 2 ) ; mp . Add ( ' j ' , 2 ) ; mp . Add ( ' k ' , 2 ) ; mp . Add ( ' l ' , 2 ) ; mp . Add ( ' z ' , 3 ) ; mp . Add ( ' x ' , 3 ) ; mp . Add ( ' c ' , 3 ) ; mp . Add ( ' v ' , 3 ) ; mp . Add ( ' b ' , 3 ) ; mp . Add ( ' n ' , 3 ) ; mp . Add ( ' m ' , 3 ) ;
foreach ( string word in arr ) {
if ( word . Length != 0 ) {
bool flag = true ;
int rowNum = mp [ char . ToLower ( word [ 0 ] ) ] ;
int M = word . Length ;
for ( int i = 1 ; i < M ; i ++ ) {
if ( mp [ Char . ToLower ( word [ i ] ) ] != rowNum ) {
flag = false ; break ; } }
if ( flag ) {
Console . Write ( word + " ▁ " ) ; } } } }
public static void Main ( String [ ] args ) { List < string > words = new List < string > ( new string [ ] { " Yeti " , " Had " , " GFG " , " comment " } ) ; findWordsSameRow ( words ) ; } }
using System ; class GFG {
static int countSubsequece ( int [ ] a , int n ) { int i , j , k , l ;
int answer = 0 ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { for ( k = j + 1 ; k < n ; k ++ ) { for ( l = k + 1 ; l < n ; l ++ ) {
if ( a [ j ] == a [ l ] &&
a [ i ] == a [ k ] ) { answer ++ ; } } } } } return answer ; }
public static void Main ( ) { int [ ] a = { 1 , 2 , 3 , 2 , 1 , 3 , 2 } ; Console . WriteLine ( countSubsequece ( a , 7 ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static char minDistChar ( char [ ] s ) { int n = s . Length ;
int [ ] first = new int [ 26 ] ; int [ ] last = new int [ 26 ] ;
for ( int i = 0 ; i < 26 ; i ++ ) { first [ i ] = - 1 ; last [ i ] = - 1 ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( first [ s [ i ] - ' a ' ] == - 1 ) { first [ s [ i ] - ' a ' ] = i ; }
last [ s [ i ] - ' a ' ] = i ; }
int min = int . MaxValue ; char ans = '1' ;
for ( int i = 0 ; i < 26 ; i ++ ) {
if ( last [ i ] == first [ i ] ) continue ;
if ( min > last [ i ] - first [ i ] ) { min = last [ i ] - first [ i ] ; ans = ( char ) ( i + ' a ' ) ; } }
return ans ; }
public static void Main ( string [ ] args ) { String str = " geeksforgeeks " ;
Console . Write ( minDistChar ( str . ToCharArray ( ) ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int n = 3 ; public class Pair { public int first , second ; public Pair ( int a , int b ) { first = a ; second = b ; } }
static int minSteps ( int [ , ] arr ) {
Boolean [ , ] v = new Boolean [ n , n ] ;
Queue < Pair > q = new Queue < Pair > ( ) ;
q . Enqueue ( new Pair ( 0 , 0 ) ) ;
int depth = 0 ;
while ( q . Count != 0 ) {
int x = q . Count ; while ( x -- > 0 ) {
Pair y = q . Peek ( ) ;
int i = y . first , j = y . second ; q . Dequeue ( ) ;
if ( v [ i , j ] ) continue ;
if ( i == n - 1 && j == n - 1 ) return depth ;
v [ i , j ] = true ;
if ( i + arr [ i , j ] < n ) q . Enqueue ( new Pair ( i + arr [ i , j ] , j ) ) ; if ( j + arr [ i , j ] < n ) q . Enqueue ( new Pair ( i , j + arr [ i , j ] ) ) ; } depth ++ ; } return - 1 ; }
public static void Main ( ) { int [ , ] arr = { { 1 , 1 , 1 } , { 1 , 1 , 1 } , { 1 , 1 , 1 } } ; Console . WriteLine ( minSteps ( arr ) ) ; } }
using System ; class GFG {
static int solve ( int [ ] a , int n ) { int max1 = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( Math . Abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . Abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
static public void Main ( ) { int [ ] arr = { - 1 , 2 , 3 , - 4 , - 10 , 22 } ; int size = arr . Length ; Console . WriteLine ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
using System ; class GFG {
static int solve ( int [ ] a , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . Abs ( min1 - max1 ) ; }
public static void Main ( ) { int [ ] arr = { - 1 , 2 , 3 , 4 , - 10 } ; int size = arr . Length ; Console . WriteLine ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
using System ; class GFG {
static void replaceOriginal ( String s , int n ) {
char [ ] r = new char [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) {
r [ i ] = s [ n - 1 - i ] ;
if ( s [ i ] != ' a ' && s [ i ] != ' e ' && s [ i ] != ' i ' && s [ i ] != ' o ' && s [ i ] != ' u ' ) { Console . Write ( r [ i ] ) ; } } Console . WriteLine ( " " ) ; }
public static void Main ( String [ ] args ) { String s = " geeksforgeeks " ; int n = s . Length ; replaceOriginal ( s , n ) ; } }
using System ; class GFG {
static bool sameStrings ( string str1 , string str2 ) { int N = str1 . Length ; int M = str2 . Length ;
if ( N != M ) { return false ; }
int [ ] a = new int [ 256 ] ; int [ ] b = new int [ 256 ] ;
for ( int j = 0 ; j < N ; j ++ ) { a [ str1 [ j ] - ' a ' ] ++ ; b [ str2 [ j ] - ' a ' ] ++ ; }
int i = 0 ; while ( i < 256 ) { if ( ( a [ i ] == 0 && b [ i ] == 0 ) || ( a [ i ] != 0 && b [ i ] != 0 ) ) { i ++ ; }
else { return false ; } }
Array . Sort ( a ) ; Array . Sort ( b ) ;
for ( int j = 0 ; j < 256 ; j ++ ) {
if ( a [ j ] != b [ j ] ) return false ; }
return true ; }
static public void Main ( ) { string S1 = " cabbba " , S2 = " abbccc " ; if ( sameStrings ( S1 , S2 ) ) Console . Write ( " YES " + " STRNEWLINE " ) ; else Console . Write ( " ▁ NO " + " STRNEWLINE " ) ; } }
using System ; class GFG {
public static int solution ( int A , int B , int C ) { int [ ] arr = new int [ 3 ] ;
arr [ 0 ] = A ; arr [ 1 ] = B ; arr [ 2 ] = C ;
Array . Sort ( arr ) ;
if ( arr [ 2 ] < arr [ 0 ] + arr [ 1 ] ) return ( ( arr [ 0 ] + arr [ 1 ] + arr [ 2 ] ) / 2 ) ;
else return ( arr [ 0 ] + arr [ 1 ] ) ; }
public static void Main ( String [ ] args ) {
int A = 8 , B = 1 , C = 5 ;
Console . WriteLine ( solution ( A , B , C ) ) ; } }
using System ; class GFG {
static int search ( int [ ] arr , int l , int h , int key ) { if ( l > h ) return - 1 ; int mid = ( l + h ) / 2 ; if ( arr [ mid ] == key ) return mid ;
if ( ( arr [ l ] == arr [ mid ] ) && ( arr [ h ] == arr [ mid ] ) ) { ++ l ; -- h ; return search ( arr , l , h , key ) }
if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
return search ( arr , mid + 1 , h , key ) ; }
if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
public static void Main ( ) { int [ ] arr = { 3 , 3 , 1 , 2 , 3 , 3 } ; int n = arr . Length ; int key = 3 ; Console . WriteLine ( search ( arr , 0 , n - 1 , key ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static String getSortedString ( char [ ] s , int n ) {
List < char > v1 = new List < char > ( ) ; List < char > v2 = new List < char > ( ) ; int i = 0 ; for ( i = 0 ; i < n ; i ++ ) { if ( s [ i ] > ' a ' && s [ i ] <= ' z ' ) v1 . Add ( s [ i ] ) ; if ( s [ i ] > ' A ' && s [ i ] <= ' z ' ) v2 . Add ( s [ i ] ) ; }
v1 . Sort ( ) ; v2 . Sort ( ) ; int j = 0 ; i = 0 ; for ( int k = 0 ; k < n ; k ++ ) {
if ( s [ k ] > ' a ' && s [ k ] <= ' z ' ) { s [ k ] = v1 [ i ] ; ++ i ; }
else if ( s [ k ] > ' A ' && [ ] <= ' Z ' ) { s [ k ] = v2 [ j ] ; ++ j ; } }
return String . Join ( " " , s ) ; }
public static void Main ( String [ ] args ) { String s = " gEeksfOrgEEkS " ; int n = s . Length ; Console . WriteLine ( getSortedString ( s . ToCharArray ( ) , n ) ) ; } }
using System ; using System . Collections ; class GfG {
static bool check ( char [ ] s ) {
int l = s . Length ;
Array . Sort ( s ) ;
for ( int i = 1 ; i < l ; i ++ ) {
if ( s [ i ] - s [ i - 1 ] != 1 ) return false ; } return true ; }
public static void Main ( ) {
string str = " dcef " ; if ( check ( str . ToCharArray ( ) ) == true ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ;
String str1 = " xyza " ; if ( check ( str1 . ToCharArray ( ) ) == true ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static int minElements ( int [ ] arr , int n ) {
int halfSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = halfSum / 2 ;
Array . Sort ( arr ) ; int res = 0 , curr_sum = 0 ; for ( int i = n - 1 ; i >= 0 ; i -- ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
public static void Main ( ) { int [ ] arr = { 3 , 1 , 7 , 1 } ; int n = arr . Length ; Console . WriteLine ( minElements ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static void arrayElementEqual ( int [ ] arr , int N ) {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
if ( sum % N == 0 ) { Console . WriteLine ( " Yes " ) ; }
else { Console . Write ( " No " + " STRNEWLINE " ) ; } }
static public void Main ( ) {
int [ ] arr = { 1 , 5 , 6 , 4 } ;
int N = arr . Length ; arrayElementEqual ( arr , N ) ; } }
using System ; class GFG {
static int findMaxValByRearrArr ( int [ ] arr , int N ) {
int res = 0 ;
res = ( N * ( N + 1 ) ) / 2 ; return res ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 2 , 1 } ; int N = arr . Length ; Console . Write ( findMaxValByRearrArr ( arr , N ) ) ; } }
using System ; class GFG {
static int MaximumSides ( int n ) {
if ( n < 4 ) return - 1 ;
return n % 2 == 0 ? n / 2 : - 1 ; }
public static void Main ( String [ ] args ) {
int N = 8 ;
Console . Write ( MaximumSides ( N ) ) ; } }
using System ; class GFG {
static double pairProductMean ( int [ ] arr , int N ) {
int [ ] suffixSumArray = new int [ N ] ; suffixSumArray [ N - 1 ] = arr [ N - 1 ] ;
for ( int i = N - 2 ; i >= 0 ; i -- ) { suffixSumArray [ i ] = suffixSumArray [ i + 1 ] + arr [ i ] ; }
int length = ( N * ( N - 1 ) ) / 2 ;
double res = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) { res += arr [ i ] * suffixSumArray [ i + 1 ] ; }
double mean ;
if ( length != 0 ) mean = res / length ; else mean = 0 ;
return mean ; }
public static void Main ( ) {
int [ ] arr = { 1 , 2 , 4 , 8 } ; int N = arr . Length ;
Console . WriteLine ( string . Format ( " { 0:0.00 } " , pairProductMean ( arr , N ) ) ) ; } }
using System ; class GFG {
static int ncr ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static int countPath ( int N , int M , int K ) { int answer ; if ( K >= 2 ) answer = 0 ; else if ( K == 0 ) answer = ncr ( N + M - 2 , N - 1 ) ; else {
answer = ncr ( N + M - 2 , N - 1 ) ;
int X = ( N - 1 ) / 2 + ( M - 1 ) / 2 ; int Y = ( N - 1 ) / 2 ; int midCount = ncr ( X , Y ) ;
X = ( ( N - 1 ) - ( N - 1 ) / 2 ) + ( ( M - 1 ) - ( M - 1 ) / 2 ) ; Y = ( ( N - 1 ) - ( N - 1 ) / 2 ) ; midCount *= ncr ( X , Y ) ; answer -= midCount ; } return answer ; }
public static void Main ( String [ ] args ) { int N = 3 ; int M = 3 ; int K = 1 ; Console . Write ( countPath ( N , M , K ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static int find_max ( List < pair > v , int n ) {
int count = 0 ; if ( n >= 2 ) count = 2 ; else count = 1 ;
for ( int i = 1 ; i < n - 1 ; i ++ ) {
if ( v [ i - 1 ] . first < ( v [ i ] . first - v [ i ] . second ) ) count ++ ;
else if ( v [ i + 1 ] . first > ( v [ i ] . first + v [ i ] . second ) ) { count ++ ; v [ i ] . first = v [ i ] . first + v [ i ] . second ; }
else continue ; }
return count ; }
public static void Main ( String [ ] args ) { int n = 3 ; List < pair > v = new List < pair > ( ) ; v . Add ( new pair ( 10 , 20 ) ) ; v . Add ( new pair ( 15 , 10 ) ) ; v . Add ( new pair ( 20 , 16 ) ) ; Console . Write ( find_max ( v , n ) ) ; } }
using System ; class GFG {
public static void numberofsubstrings ( String str , int k , char [ ] charArray ) { int N = str . Length ;
int [ ] available = new int [ 26 ] ;
for ( int i = 0 ; i < k ; i ++ ) { available [ charArray [ i ] - ' a ' ] = 1 ; }
int lastPos = - 1 ;
int ans = ( N * ( N + 1 ) ) / 2 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( available [ str [ i ] - ' a ' ] == 0 ) {
ans -= ( ( i - lastPos ) * ( N - i ) ) ;
lastPos = i ; } }
Console . WriteLine ( ans ) ; }
public static void Main ( String [ ] args ) {
String str = " abcb " ; int k = 2 ;
char [ ] charArray = { ' a ' , ' b ' } ;
numberofsubstrings ( str , k , charArray ) ; } }
class GFG {
static int minCost ( int N , int P , int Q ) {
int cost = 0 ;
while ( N > 0 ) { if ( ( N & 1 ) > 0 ) { cost += P ; N -- ; } else { int temp = N / 2 ;
if ( temp * P > Q ) cost += Q ;
else cost += * temp ; N /= 2 ; } }
return cost ; }
static void Main ( ) { int N = 9 , P = 5 , Q = 1 ; System . Console . WriteLine ( minCost ( N , P , Q ) ) ; } }
using System ; class GFG {
static void numberOfWays ( int n , int k ) {
int [ ] dp = new int [ 1000 ] ;
for ( int i = 0 ; i < n ; i ++ ) { dp [ i ] = 0 ; }
dp [ 0 ] = 1 ;
for ( int i = 1 ; i <= k ; i ++ ) {
int numWays = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { numWays += dp [ j ] ; }
for ( int j = 0 ; j < n ; j ++ ) { dp [ j ] = numWays - dp [ j ] ; } }
Console . Write ( dp [ 0 ] ) ; }
static public void Main ( ) {
int N = 5 , K = 3 ;
numberOfWays ( N , K ) ; } }
using System ; class GFG { static int M = 1000000007 ; static int waysOfDecoding ( string s ) { long first = 1 , second = s [ 0 ] == ' * ' ? 9 : s [ 0 ] == '0' ? 0 : 1 ; for ( int i = 1 ; i < s . Length ; i ++ ) { long temp = second ;
if ( s [ i ] == ' * ' ) { second = 9 * second ;
if ( s [ i - 1 ] == '1' ) second = ( second + 9 * first ) % M ;
else if ( s [ i - 1 ] == '2' ) = ( second + 6 * first ) % M ;
else if ( s [ i - 1 ] == ' * ' ) = ( second + 15 * first ) % M ; }
else { second = s [ i ] != '0' ? second : 0 ;
if ( s [ i - 1 ] == '1' ) second = ( second + first ) % M ;
else if ( s [ i - 1 ] == '2' && s [ i ] <= '6' ) = ( second + first ) % M ;
else if ( s [ i - 1 ] == ' * ' ) = ( second + ( s [ i ] <= '6' ? 2 : 1 ) * first ) % M ; } = temp ; } return ( int ) second ; }
static public void Main ( ) { string s = " * " ; Console . WriteLine ( waysOfDecoding ( s ) ) ; } }
using System ; class GFG {
static int findMinCost ( int [ , ] arr , int X , int n , int i = 0 ) {
if ( X <= 0 ) return 0 ; if ( i >= n ) return Int32 . MaxValue ;
int inc = findMinCost ( arr , X - arr [ i , 0 ] , n , i + 1 ) ; if ( inc != Int32 . MaxValue ) inc += arr [ i , 1 ] ;
int exc = findMinCost ( arr , X , n , i + 1 ) ;
return Math . Min ( inc , exc ) ; }
public static void Main ( ) {
int [ , ] arr = { { 4 , 3 } , { 3 , 2 } , { 2 , 4 } , { 1 , 3 } , { 4 , 2 } } ; int X = 7 ;
int n = arr . GetLength ( 0 ) ; int ans = findMinCost ( arr , X , n ) ;
if ( ans != Int32 . MaxValue ) Console . Write ( ans ) ; else Console . Write ( - 1 ) ; } }
using System ; class GFG {
static double find ( int N , int sum ) {
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return 1.0 / 6 ; else return 0 ; } double s = 0 ; for ( int i = 1 ; i <= 6 ; i ++ ) s = s + find ( N - 1 , sum - i ) / 6 ; return s ; }
static void Main ( ) { int N = 4 , a = 13 , b = 17 ; double probability = 0.0 ; for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
Console . WriteLine ( Math . Round ( probability , 6 ) ) ; } }
using System ; class GFG {
static int minDays ( int n ) {
if ( n < 1 ) return n ;
int cnt = 1 + Math . Min ( n % 2 + minDays ( n / 2 ) , n % 3 + minDays ( n / 3 ) ) ;
return cnt ; }
public static void Main ( String [ ] args ) {
int N = 6 ;
Console . Write ( minDays ( N ) ) ; } }
