import java . io . * ; class GFG {
static double Conversion ( double centi ) { double pixels = ( 96 * centi ) / 2.54 ; System . out . println ( pixels ) ; return 0 ; }
public static void main ( String args [ ] ) { int centi = 15 ; Conversion ( centi ) ; } }
import java . util . * ; class GFG {
static int xor_operations ( int N , int arr [ ] , int M , int K ) {
if ( M < 0 M >= N ) return - 1 ;
if ( K < 0 K >= N - M ) return - 1 ;
for ( int p = 0 ; p < M ; p ++ ) {
Vector < Integer > temp = new Vector < Integer > ( ) ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
int value = arr [ i ] ^ arr [ i + 1 ] ;
temp . add ( value ) ;
arr [ i ] = temp . get ( i ) ; } }
int ans = arr [ K ] ; return ans ; }
public static void main ( String [ ] args ) {
int N = 5 ;
int arr [ ] = { 1 , 4 , 5 , 6 , 7 } ; int M = 1 , K = 2 ;
System . out . print ( xor_operations ( N , arr , M , K ) ) ; } }
class GFG {
public static void canBreakN ( long n ) {
for ( long i = 2 ; ; i ++ ) {
long m = i * ( i + 1 ) / 2 ;
if ( m > n ) break ; long k = n - m ;
if ( k % i != 0 ) continue ;
System . out . println ( i ) ; return ; }
System . out . println ( " - 1" ) ; }
public static void main ( String [ ] args ) {
long N = 12 ;
canBreakN ( N ) ; } }
import java . util . * ; class GFG {
public static void findCoprimePair ( int N ) {
for ( int x = 2 ; x <= Math . sqrt ( N ) ; x ++ ) { if ( N % x == 0 ) {
while ( N % x == 0 ) { N /= x ; } if ( N > 1 ) {
System . out . println ( x + " ▁ " + N ) ; return ; } } }
System . out . println ( - 1 ) ; }
public static void main ( String [ ] args ) {
int N = 45 ; findCoprimePair ( N ) ;
N = 25 ; findCoprimePair ( N ) ; } }
import java . util . * ; class GFG { static int MAX = 10000 ;
static Vector < Integer > primes = new Vector < Integer > ( ) ;
static void sieveSundaram ( ) {
boolean marked [ ] = new boolean [ MAX / 2 + 1 ] ;
for ( int i = 1 ; i <= ( Math . sqrt ( MAX ) - 1 ) / 2 ; i ++ ) { for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= MAX / 2 ; j = j + 2 * i + 1 ) { marked [ j ] = true ; } }
primes . add ( 2 ) ;
for ( int i = 1 ; i <= MAX / 2 ; i ++ ) if ( marked [ i ] == false ) primes . add ( 2 * i + 1 ) ; }
static boolean isWasteful ( int n ) { if ( n == 1 ) return false ;
int original_no = n ; int sumDigits = 0 ; while ( original_no > 0 ) { sumDigits ++ ; original_no = original_no / 10 ; } int pDigit = 0 , count_exp = 0 , p = 0 ;
for ( int i = 0 ; primes . get ( i ) <= n / 2 ; i ++ ) {
while ( n % primes . get ( i ) == 0 ) {
p = primes . get ( i ) ; n = n / p ;
count_exp ++ ; }
while ( p > 0 ) { pDigit ++ ; p = p / 10 ; }
while ( count_exp > 1 ) { pDigit ++ ; count_exp = count_exp / 10 ; } }
if ( n != 1 ) { while ( n > 0 ) { pDigit ++ ; n = n / 10 ; } }
return ( pDigit > sumDigits ) ; }
static void Solve ( int N ) {
for ( int i = 1 ; i < N ; i ++ ) { if ( isWasteful ( i ) ) { System . out . print ( i + " ▁ " ) ; } } }
public static void main ( String [ ] args ) {
sieveSundaram ( ) ; int N = 10 ;
Solve ( N ) ; } }
import java . util . * ; class GFG {
static int printhexaRec ( int n ) { if ( n == 0 n == 1 n == 2 n == 3 n == 4 n == 5 ) return 0 ; else if ( n == 6 ) return 1 ; else return ( printhexaRec ( n - 1 ) + printhexaRec ( n - 2 ) + printhexaRec ( n - 3 ) + printhexaRec ( n - 4 ) + printhexaRec ( n - 5 ) + printhexaRec ( n - 6 ) ) ; } static void printhexa ( int n ) { System . out . print ( printhexaRec ( n ) + "NEW_LINE"); }
public static void main ( String [ ] args ) { int n = 11 ; printhexa ( n ) ; } }
class GFG {
static void printhexa ( int n ) { if ( n < 0 ) return ;
int first = 0 ; int second = 0 ; int third = 0 ; int fourth = 0 ; int fifth = 0 ; int sixth = 1 ;
int curr = 0 ; if ( n < 6 ) System . out . println ( first ) ; else if ( n == 6 ) System . out . println ( sixth ) ; else {
for ( int i = 6 ; i < n ; i ++ ) { curr = first + second + third + fourth + fifth + sixth ; first = second ; second = third ; third = fourth ; fourth = fifth ; fifth = sixth ; sixth = curr ; } } System . out . println ( curr ) ; }
public static void main ( String [ ] args ) { int n = 11 ; printhexa ( n ) ; } }
class GFG {
static void smallestNumber ( int N ) { System . out . print ( ( N % 9 + 1 ) * Math . pow ( 10 , ( N / 9 ) ) - 1 ) ; }
public static void main ( String [ ] args ) { int N = 10 ; smallestNumber ( N ) ; } }
import java . util . * ; class GFG { static Vector < Integer > compo = new Vector < Integer > ( ) ;
static boolean isComposite ( int n ) {
if ( n <= 3 ) return false ;
if ( n % 2 == 0 n % 3 == 0 ) return true ; int i = 5 ; while ( i * i <= n ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) return true ; i = i + 6 ; } return false ; }
static void Compositorial_list ( int n ) { int l = 0 ; for ( int i = 4 ; i < 1000000 ; i ++ ) { if ( l < n ) { if ( isComposite ( i ) ) { compo . add ( i ) ; l += 1 ; } } } }
static int calculateCompositorial ( int n ) {
int result = 1 ; for ( int i = 0 ; i < n ; i ++ ) result = result * compo . get ( i ) ; return result ; }
public static void main ( String [ ] args ) { int n = 5 ;
Compositorial_list ( n ) ; System . out . print ( ( calculateCompositorial ( n ) ) ) ; } }
class GFG {
static int b [ ] = new int [ 50 ] ;
static int PowerArray ( int n , int k ) {
int count = 0 ;
while ( k > 0 ) { if ( k % n == 0 ) { k /= n ; count ++ ; }
else if ( k % n == 1 ) { k -= 1 ; b [ count ] ++ ;
if ( b [ count ] > 1 ) { System . out . print ( - 1 ) ; return 0 ; } }
else { System . out . print ( - 1 ) ; return 0 ; } }
for ( int i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] != 0 ) { System . out . print ( i + " , ▁ " ) ; } } return Integer . MIN_VALUE ; }
public static void main ( String [ ] args ) { int N = 3 ; int K = 40 ; PowerArray ( N , K ) ; } }
class GFG {
static int findSum ( int N , int k ) {
int sum = 0 ; for ( int i = 1 ; i <= N ; i ++ ) {
sum += ( int ) Math . pow ( i , k ) ; }
return sum ; }
public static void main ( String [ ] args ) { int N = 8 , k = 4 ;
System . out . println ( findSum ( N , k ) ) ; } }
class GFG {
static int countIndices ( int arr [ ] , int n ) {
int cnt = 0 ;
int max = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( max < arr [ i ] ) {
max = arr [ i ] ;
cnt ++ ; } } return cnt ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int n = arr . length ; System . out . println ( countIndices ( arr , n ) ) ; } }
class GFG {
static String bin [ ] = { "000" , "001" , "010" , "011" , "100" , "101" , "110" , "111" } ;
static int maxFreq ( String s ) {
String binary = " " ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) { binary += bin [ s . charAt ( i ) - '0' ] ; }
binary = binary . substring ( 0 , binary . length ( ) - 1 ) ; int count = 1 , prev = - 1 , i , j = 0 ; for ( i = binary . length ( ) - 1 ; i >= 0 ; i -- , j ++ )
if ( binary . charAt ( i ) == '1' ) {
count = Math . max ( count , j - prev ) ; prev = j ; } return count ; }
public static void main ( String [ ] args ) { String octal = "13" ; System . out . println ( maxFreq ( octal ) ) ; } }
import java . util . * ; class GFG { static int sz = 100000 ; static boolean isPrime [ ] = new boolean [ sz + 1 ] ;
static void sieve ( ) { for ( int i = 0 ; i <= sz ; i ++ ) isPrime [ i ] = true ; isPrime [ 0 ] = isPrime [ 1 ] = false ; for ( int i = 2 ; i * i <= sz ; i ++ ) { if ( isPrime [ i ] ) { for ( int j = i * i ; j < sz ; j += i ) { isPrime [ j ] = false ; } } } }
static void findPrimesD ( int d ) {
int left = ( int ) Math . pow ( 10 , d - 1 ) ; int right = ( int ) Math . pow ( 10 , d ) - 1 ;
for ( int i = left ; i <= right ; i ++ ) {
if ( isPrime [ i ] ) { System . out . print ( i + " ▁ " ) ; } } }
public static void main ( String args [ ] ) {
sieve ( ) ; int d = 1 ; findPrimesD ( d ) ; } }
class GFG {
public static int Cells ( int n , int x ) { if ( n <= 0 x <= 0 x > n * n ) return 0 ; int i = 0 , count = 0 ; while ( ++ i * i < x ) if ( x % i == 0 && x <= n * i ) count += 2 ; return i * i == x ? count + 1 : count ; }
public static void main ( String [ ] args ) { int n = 6 , x = 12 ;
System . out . println ( Cells ( n , x ) ) ; } }
import java . io . * ; class GFG {
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
import java . util . * ; import java . lang . * ; class GFG {
static int isSumDivides ( int N ) { int temp = N ; int sum = 0 ;
while ( temp > 0 ) { sum += temp % 10 ; temp /= 10 ; } if ( N % sum == 0 ) return 1 ; else return 0 ; }
public static void main ( String args [ ] ) { int N = 12 ; if ( isSumDivides ( N ) == 1 ) System . out . print ( " YES " ) ; else System . out . print ( " NO " ) ; } }
class GFG {
static int sum ( int N ) { int S1 , S2 , S3 ; S1 = ( ( N / 3 ) ) * ( 2 * 3 + ( N / 3 - 1 ) * 3 ) / 2 ; S2 = ( ( N / 4 ) ) * ( 2 * 4 + ( N / 4 - 1 ) * 4 ) / 2 ; S3 = ( ( N / 12 ) ) * ( 2 * 12 + ( N / 12 - 1 ) * 12 ) / 2 ; return S1 + S2 - S3 ; }
public static void main ( String [ ] args ) { int N = 20 ; System . out . print ( sum ( 12 ) ) ; } }
class GFG {
static int nextGreater ( int N ) { int power_of_2 = 1 , shift_count = 0 ;
while ( true ) {
if ( ( ( N >> shift_count ) & 1 ) % 2 == 0 ) break ;
shift_count ++ ;
power_of_2 = power_of_2 * 2 ; }
return ( N + power_of_2 ) ; }
public static void main ( String [ ] a ) { int N = 11 ;
System . out . println ( " The ▁ next ▁ number ▁ is ▁ = ▁ " + nextGreater ( N ) ) ; } }
import java . io . * ; class GFG {
static int countWays ( int n ) {
if ( n == 0 ) return 1 ; if ( n <= 2 ) return n ;
int f0 = 1 , f1 = 1 , f2 = 2 ; int ans = 0 ;
for ( int i = 3 ; i <= n ; i ++ ) { ans = f0 + f1 + f2 ; f0 = f1 ; f1 = f2 ; f2 = ans ; }
return ans ; }
public static void main ( String [ ] args ) { int n = 4 ; System . out . println ( countWays ( n ) ) ; } }
class GFG { static int n = 6 , m = 6 ;
static void maxSum ( long arr [ ] [ ] ) {
long [ ] [ ] dp = new long [ n + 1 ] [ 3 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
long m1 = 0 , m2 = 0 , m3 = 0 ; for ( int j = 0 ; j < m ; j ++ ) {
if ( ( j / ( m / 3 ) ) == 0 ) { m1 = Math . max ( m1 , arr [ i ] [ j ] ) ; }
else if ( ( j / ( m / 3 ) ) == 1 ) { m2 = Math . max ( m2 , arr [ i ] [ j ] ) ; }
else if ( ( j / ( m / 3 ) ) == 2 ) { m3 = Math . max ( m3 , arr [ i ] [ j ] ) ; } }
dp [ i + 1 ] [ 0 ] = Math . max ( dp [ i ] [ 1 ] , dp [ i ] [ 2 ] ) + m1 ; dp [ i + 1 ] [ 1 ] = Math . max ( dp [ i ] [ 0 ] , dp [ i ] [ 2 ] ) + m2 ; dp [ i + 1 ] [ 2 ] = Math . max ( dp [ i ] [ 1 ] , dp [ i ] [ 0 ] ) + m3 ; }
System . out . print ( Math . max ( Math . max ( dp [ n ] [ 0 ] , dp [ n ] [ 1 ] ) , dp [ n ] [ 2 ] ) + "NEW_LINE"); }
public static void main ( String [ ] args ) { long arr [ ] [ ] = { { 1 , 3 , 5 , 2 , 4 , 6 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 1 , 3 , 5 , 2 , 4 , 6 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 1 , 3 , 5 , 2 , 4 , 6 } } ; maxSum ( arr ) ; } }
import java . util . * ; class GFG {
static void solve ( char [ ] s ) { int n = s . length ;
int [ ] [ ] dp = new int [ n ] [ n ] ;
for ( int len = n - 1 ; len >= 0 ; -- len ) {
for ( int i = 0 ; i + len < n ; ++ i ) {
int j = i + len ;
if ( i == 0 && j == n - 1 ) { if ( s [ i ] == s [ j ] ) dp [ i ] [ j ] = 2 ; else if ( s [ i ] != s [ j ] ) dp [ i ] [ j ] = 1 ; } else { if ( s [ i ] == s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i ] [ j ] += dp [ i - 1 ] [ j ] ; } if ( j + 1 <= n - 1 ) { dp [ i ] [ j ] += dp [ i ] [ j + 1 ] ; } if ( i - 1 < 0 j + 1 >= n ) {
dp [ i ] [ j ] += 1 ; } } else if ( s [ i ] != s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i ] [ j ] += dp [ i - 1 ] [ j ] ; } if ( j + 1 <= n - 1 ) { dp [ i ] [ j ] += dp [ i ] [ j + 1 ] ; } if ( i - 1 >= 0 && j + 1 <= n - 1 ) {
dp [ i ] [ j ] -= dp [ i - 1 ] [ j + 1 ] ; } } } } } Vector < Integer > ways = new Vector < > ( ) ; for ( int i = 0 ; i < n ; ++ i ) { if ( i == 0 i == n - 1 ) {
ways . add ( 1 ) ; } else {
int total = dp [ i - 1 ] [ i + 1 ] ; ways . add ( total ) ; } } for ( int i = 0 ; i < ways . size ( ) ; ++ i ) { System . out . print ( ways . get ( i ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) { char [ ] s = " xyxyx " . toCharArray ( ) ; solve ( s ) ; } }
import java . util . * ; public class GFG {
static long getChicks ( int n ) {
int size = Math . max ( n , 7 ) ; long [ ] dp = new long [ size ] ; dp [ 0 ] = 0 ; dp [ 1 ] = 1 ;
for ( int i = 2 ; i < 6 ; i ++ ) { dp [ i ] = dp [ i - 1 ] * 3 ; }
dp [ 6 ] = 726 ;
for ( int i = 8 ; i <= n ; i ++ ) {
dp [ i ] = ( dp [ i - 1 ] - ( 2 * dp [ i - 6 ] / 3 ) ) * 3 ; } return dp [ n ] ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( getChicks ( n ) ) ; } }
import java . io . * ; class GFG {
static int getChicks ( int n ) { int chicks = ( int ) Math . pow ( 3 , n - 1 ) ; return chicks ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( getChicks ( n ) ) ; } }
class GFG { static int n = 3 ;
static int [ ] [ ] dp = new int [ n ] [ n ] ;
static int [ ] [ ] v = new int [ n ] [ n ] ;
static int minSteps ( int i , int j , int arr [ ] [ ] ) {
if ( i == n - 1 && j == n - 1 ) { return 0 ; } if ( i > n - 1 j > n - 1 ) { return 9999999 ; }
if ( v [ i ] [ j ] == 1 ) { return dp [ i ] [ j ] ; } v [ i ] [ j ] = 1 ; dp [ i ] [ j ] = 9999999 ;
for ( int k = Math . max ( 0 , arr [ i ] [ j ] + j - n + 1 ) ; k <= Math . min ( n - i - 1 , arr [ i ] [ j ] ) ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , minSteps ( i + k , j + arr [ i ] [ j ] - k , arr ) ) ; } dp [ i ] [ j ] ++ ; return dp [ i ] [ j ] ; }
public static void main ( String [ ] args ) { int arr [ ] [ ] = { { 4 , 1 , 2 } , { 1 , 1 , 1 } , { 2 , 1 , 1 } } ; int ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) { System . out . println ( - 1 ) ; } else { System . out . println ( ans ) ; } } }
class GFG { static int n = 3 ;
static int dp [ ] [ ] = new int [ n ] [ n ] ;
static int [ ] [ ] v = new int [ n ] [ n ] ;
static int minSteps ( int i , int j , int arr [ ] [ ] ) {
if ( i == n - 1 && j == n - 1 ) { return 0 ; } if ( i > n - 1 j > n - 1 ) { return 9999999 ; }
if ( v [ i ] [ j ] == 1 ) { return dp [ i ] [ j ] ; } v [ i ] [ j ] = 1 ;
dp [ i ] [ j ] = 1 + Math . min ( minSteps ( i + arr [ i ] [ j ] , j , arr ) , minSteps ( i , j + arr [ i ] [ j ] , arr ) ) ; return dp [ i ] [ j ] ; }
public static void main ( String [ ] args ) { int arr [ ] [ ] = { { 2 , 1 , 2 } , { 1 , 1 , 1 } , { 1 , 1 , 1 } } ; int ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) { System . out . println ( - 1 ) ; } else { System . out . println ( ans ) ; } } }
import java . util . * ; class GFG { static int MAX = 1001 ; static int [ ] [ ] dp = new int [ MAX ] [ MAX ] ;
static int MaxProfit ( int treasure [ ] , int color [ ] , int n , int k , int col , int A , int B ) {
return dp [ k ] [ col ] = 0 ; if ( dp [ k ] [ col ] != - 1 ) return dp [ k ] [ col ] ; int sum = 0 ;
if ( col == color [ k ] ) sum += Math . max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += Math . max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return dp [ k ] [ col ] = sum ; }
public static void main ( String [ ] args ) { int A = - 5 , B = 7 ; int treasure [ ] = { 4 , 8 , 2 , 9 } ; int color [ ] = { 2 , 2 , 6 , 2 } ; int n = color . length ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < MAX ; j ++ ) dp [ i ] [ j ] = - 1 ; System . out . print ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) ; } }
class GFG {
static void printTetra ( int n ) { int [ ] dp = new int [ n + 5 ] ;
dp [ 0 ] = 0 ; dp [ 1 ] = dp [ 2 ] = 1 ; dp [ 3 ] = 2 ; for ( int i = 4 ; i <= n ; i ++ ) dp [ i ] = dp [ i - 1 ] + dp [ i - 2 ] + dp [ i - 3 ] + dp [ i - 4 ] ; System . out . print ( dp [ n ] ) ; }
public static void main ( String [ ] args ) { int n = 10 ; printTetra ( n ) ; } }
import java . io . * ; class GFG {
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
public class GFG {
static void compute_z ( String s , int z [ ] ) { int l = 0 , r = 0 ; int n = s . length ( ) ; for ( int i = 1 ; i <= n - 1 ; i ++ ) { if ( i > r ) { l = i ; r = i ; while ( r < n && s . charAt ( r - l ) == s . charAt ( r ) ) r ++ ; z [ i ] = r - l ; r -- ; } else { int k = i - l ; if ( z [ k ] < r - i + 1 ) { z [ i ] = z [ k ] ; } else { l = i ; while ( r < n && s . charAt ( r - l ) == s . charAt ( r ) ) r ++ ; z [ i ] = r - l ; r -- ; } } } }
static int countPermutation ( String a , String b ) {
b = b + b ;
b = b . substring ( 0 , b . length ( ) - 1 ) ;
int ans = 0 ; String s = a + " $ " + b ; int n = s . length ( ) ;
int z [ ] = new int [ n ] ; compute_z ( s , z ) ; for ( int i = 1 ; i <= n - 1 ; i ++ ) {
if ( z [ i ] == a . length ( ) ) ans ++ ; } return ans ; }
public static void main ( String [ ] args ) { String a = "101" ; String b = "101" ; System . out . println ( countPermutation ( a , b ) ) ; } }
import java . util . * ; class GFG {
static void smallestSubsequence ( char [ ] S , int K ) {
int N = S . length ;
Stack < Character > answer = new Stack < > ( ) ;
for ( int i = 0 ; i < N ; ++ i ) {
if ( answer . isEmpty ( ) ) { answer . add ( S [ i ] ) ; } else {
while ( ( ! answer . isEmpty ( ) ) && ( S [ i ] < answer . peek ( ) )
&& ( answer . size ( ) - 1 + N - i >= K ) ) { answer . pop ( ) ; }
if ( answer . isEmpty ( ) || answer . size ( ) < K ) {
answer . add ( S [ i ] ) ; } } }
String ret = " " ;
while ( ! answer . isEmpty ( ) ) { ret += ( answer . peek ( ) ) ; answer . pop ( ) ; }
ret = reverse ( ret ) ;
System . out . print ( ret ) ; } static String reverse ( String input ) { char [ ] a = input . toCharArray ( ) ; int l , r = a . length - 1 ; for ( l = 0 ; l < r ; l ++ , r -- ) { char temp = a [ l ] ; a [ l ] = a [ r ] ; a [ r ] = temp ; } return String . valueOf ( a ) ; }
public static void main ( String [ ] args ) { String S = " aabdaabc " ; int K = 3 ; smallestSubsequence ( S . toCharArray ( ) , K ) ; } }
import java . io . * ; class GFG {
public static boolean is_rtol ( String s ) { int tmp = ( int ) ( Math . sqrt ( s . length ( ) ) ) - 1 ; char first = s . charAt ( tmp ) ;
for ( int pos = tmp ; pos < s . length ( ) - 1 ; pos += tmp ) {
if ( s . charAt ( pos ) != first ) { return false ; } } return true ; }
public static void main ( String args [ ] ) {
String str = " abcxabxcaxbcxabc " ;
if ( is_rtol ( str ) ) { System . out . print ( " Yes " ) ; } else { System . out . print ( " No " ) ; } } }
class GFG {
static boolean check ( String str , int K ) {
if ( str . length ( ) % K == 0 ) { int sum = 0 , i ;
for ( i = 0 ; i < K ; i ++ ) { sum += str . charAt ( i ) ; }
for ( int j = i ; j < str . length ( ) ; j += K ) { int s_comp = 0 ; for ( int p = j ; p < j + K ; p ++ ) s_comp += str . charAt ( p ) ;
if ( s_comp != sum )
return false ; }
return true ; }
return false ; }
public static void main ( String args [ ] ) { int K = 3 ; String str = " abdcbbdba " ; if ( check ( str , K ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . util . * ; class GFG {
static int maxSum ( String str ) { int maximumSum = 0 ;
int totalOnes = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str . charAt ( i ) == '1' ) { totalOnes ++ ; } }
int zero = 0 , ones = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str . charAt ( i ) == '0' ) { zero ++ ; } else { ones ++ ; }
maximumSum = Math . max ( maximumSum , zero + ( totalOnes - ones ) ) ; } return maximumSum ; }
public static void main ( String args [ ] ) {
String str = "011101" ;
System . out . println ( maxSum ( str ) ) ; } }
import java . util . * ; class GFG {
static int maxLenSubStr ( String s ) {
if ( s . length ( ) < 3 ) return s . length ( ) ;
int temp = 2 ; int ans = 2 ;
for ( int i = 2 ; i < s . length ( ) ; i ++ ) {
if ( s . charAt ( i ) != s . charAt ( i - 1 ) || s . charAt ( i ) != s . charAt ( i - 2 ) ) temp ++ ;
else { ans = Math . max ( temp , ans ) ; temp = 2 ; } } ans = Math . max ( temp , ans ) ; return ans ; }
public static void main ( String [ ] args ) { String s = " baaabbabbb " ; System . out . println ( maxLenSubStr ( s ) ) ; } }
import java . util . * ; class solution {
static int no_of_ways ( String s ) { int n = s . length ( ) ;
int count_left = 0 , count_right = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { if ( s . charAt ( i ) == s . charAt ( 0 ) ) { ++ count_left ; } else break ; }
for ( int i = n - 1 ; i >= 0 ; -- i ) { if ( s . charAt ( i ) == s . charAt ( n - 1 ) ) { ++ count_right ; } else break ; }
if ( s . charAt ( 0 ) == s . charAt ( n - 1 ) ) return ( ( count_left + 1 ) * ( count_right + 1 ) ) ;
else return ( count_left + count_right + 1 ) ; }
public static void main ( String args [ ] ) { String s = " geeksforgeeks " ; System . out . println ( no_of_ways ( s ) ) ; } }
import java . io . * ; class GFG {
static void preCompute ( int n , String s , int pref [ ] ) { pref [ 0 ] = 0 ; for ( int i = 1 ; i < n ; i ++ ) { pref [ i ] = pref [ i - 1 ] ; if ( s . charAt ( i - 1 ) == s . charAt ( i ) ) pref [ i ] ++ ; } }
static int query ( int pref [ ] , int l , int r ) { return pref [ r ] - pref [ l ] ; }
public static void main ( String [ ] args ) { String s = " ggggggg " ; int n = s . length ( ) ; int pref [ ] = new int [ n ] ; preCompute ( n , s , pref ) ;
int l = 1 ; int r = 2 ; System . out . println ( query ( pref , l , r ) ) ;
l = 1 ; r = 5 ; System . out . println ( query ( pref , l , r ) ) ; } }
import java . util . * ; class GFG {
static String findDirection ( String s ) { int count = 0 ; String d = " " ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s . charAt ( 0 ) == 'NEW_LINE') return null ; if ( s . charAt ( i ) == ' L ' ) count -- ; else { if ( s . charAt ( i ) == ' R ' ) count ++ ; } }
if ( count > 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == 1 ) d = " E " ; else if ( count % 4 == 2 ) d = " S " ; else if ( count % 4 == 3 ) d = " W " ; }
if ( count < 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == - 1 ) d = " W " ; else if ( count % 4 == - 2 ) d = " S " ; else if ( count % 4 == - 3 ) d = " E " ; } return d ; }
public static void main ( String [ ] args ) { String s = " LLRLRRL " ; System . out . println ( findDirection ( s ) ) ; s = " LL " ; System . out . println ( findDirection ( s ) ) ; } }
public class GFG {
static boolean isCheck ( String str ) { int len = str . length ( ) ; String lowerStr = " " , upperStr = " " ; char [ ] str1 = str . toCharArray ( ) ;
for ( int i = 0 ; i < len ; i ++ ) {
if ( ( int ) ( str1 [ i ] ) >= 65 && ( int ) str1 [ i ] <= 91 ) upperStr = upperStr + str1 [ i ] ; else lowerStr = lowerStr + str1 [ i ] ; }
String transformStr = lowerStr . toUpperCase ( ) ; return ( transformStr . equals ( upperStr ) ) ; }
public static void main ( String [ ] args ) { String str = " geeGkEEsKS " ; if ( isCheck ( str ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
class GFG {
static void encode ( String s , int k ) {
String newS = " " ;
for ( int i = 0 ; i < s . length ( ) ; ++ i ) {
int val = s . charAt ( i ) ;
int dup = k ;
if ( val + k > 122 ) { k -= ( 122 - val ) ; k = k % 26 ; newS += ( char ) ( 96 + k ) ; } else { newS += ( char ) ( val + k ) ; } k = dup ; }
System . out . println ( newS ) ; }
public static void main ( String [ ] args ) { String str = " abc " ; int k = 28 ;
encode ( str , k ) ; } }
import java . io . * ; import java . util . * ; import java . lang . * ; class GFG {
static boolean isVowel ( char x ) { if ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) return true ; else return false ; }
static String updateSandwichedVowels ( String a ) { int n = a . length ( ) ;
String updatedString = " " ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i == 0 i == n - 1 ) { updatedString += a . charAt ( i ) ; continue ; }
if ( isVowel ( a . charAt ( i ) ) == true && isVowel ( a . charAt ( i - 1 ) ) == false && isVowel ( a . charAt ( i + 1 ) ) == false ) { continue ; }
updatedString += a . charAt ( i ) ; } return updatedString ; }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ;
String updatedString = updateSandwichedVowels ( str ) ; System . out . print ( updatedString ) ; } }
import java . util . * ; class GFG {
static class Node { int data ; Node left , right ; } ; static int ans ;
static Node newNode ( int data ) { Node newNode = new Node ( ) ; newNode . data = data ; newNode . left = newNode . right = null ; return ( newNode ) ; }
static void findPathUtil ( Node root , int k , Vector < Integer > path , int flag ) { if ( root == null ) return ;
if ( root . data >= k ) flag = 1 ;
if ( root . left == null && root . right == null ) { if ( flag == 1 ) { ans = 1 ; System . out . print ( " ( " ) ; for ( int i = 0 ; i < path . size ( ) ; i ++ ) { System . out . print ( path . get ( i ) + " , ▁ " ) ; } System . out . print ( root . data + " ) , ▁ " ) ; } return ; }
path . add ( root . data ) ;
findPathUtil ( root . left , k , path , flag ) ; findPathUtil ( root . right , k , path , flag ) ;
path . remove ( path . size ( ) - 1 ) ; }
static void findPath ( Node root , int k ) {
int flag = 0 ;
ans = 0 ; Vector < Integer > v = new Vector < Integer > ( ) ;
findPathUtil ( root , k , v , flag ) ;
if ( ans == 0 ) System . out . print ( " - 1" ) ; }
public static void main ( String [ ] args ) { int K = 25 ;
Node root = newNode ( 10 ) ; root . left = newNode ( 5 ) ; root . right = newNode ( 8 ) ; root . left . left = newNode ( 29 ) ; root . left . right = newNode ( 2 ) ; root . right . right = newNode ( 98 ) ; root . right . left = newNode ( 1 ) ; root . right . right . right = newNode ( 50 ) ; root . left . left . left = newNode ( 20 ) ; findPath ( root , K ) ; } }
class GFG {
static int Tridecagonal_num ( int n ) {
return ( 11 * n * n - 9 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . print ( Tridecagonal_num ( n ) + "NEW_LINE"); n = 10 ; System . out . print ( Tridecagonal_num ( n ) + "NEW_LINE"); } }
import java . util . * ; class GFG {
static class Node { int data ; Node left ; Node right ; } ;
static Node newNode ( int k ) { Node node = new Node ( ) ; node . data = k ; node . right = node . left = null ; return node ; } static boolean isHeap ( Node root ) { Queue < Node > q = new LinkedList < > ( ) ; q . add ( root ) ; boolean nullish = false ; while ( ! q . isEmpty ( ) ) { Node temp = q . peek ( ) ; q . remove ( ) ; if ( temp . left != null ) { if ( nullish temp . left . data >= temp . data ) { return false ; } q . add ( temp . left ) ; } else { nullish = true ; } if ( temp . right != null ) { if ( nullish temp . right . data >= temp . data ) { return false ; } q . add ( temp . right ) ; } else { nullish = true ; } } return true ; }
public static void main ( String [ ] args ) { Node root = null ; root = newNode ( 10 ) ; root . left = newNode ( 9 ) ; root . right = newNode ( 8 ) ; root . left . left = newNode ( 7 ) ; root . left . right = newNode ( 6 ) ; root . right . left = newNode ( 5 ) ; root . right . right = newNode ( 4 ) ; root . left . left . left = newNode ( 3 ) ; root . left . left . right = newNode ( 2 ) ; root . left . right . left = newNode ( 1 ) ;
if ( isHeap ( root ) ) System . out . print ( "Given binary tree is a HeapNEW_LINE"); else System . out . print ( "Given binary tree is not a HeapNEW_LINE"); } }
class GFG {
static int findNumbers ( int n , int w ) { int x = 0 , sum = 0 ;
if ( w >= 0 && w <= 8 ) {
x = 9 - w ; }
else if ( w >= - 9 && w <= - 1 ) {
x = 10 + w ; } sum = ( int ) Math . pow ( 10 , n - 2 ) ; sum = ( x * sum ) ; return sum ; }
public static void main ( String args [ ] ) { int n , w ;
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
class PellNumber {
public static int pell ( int n ) { if ( n <= 2 ) return n ; int a = 1 ; int b = 2 ; int c ; for ( int i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( pell ( n ) ) ; } }
class Fibonacci {
static boolean isMultipleOf10 ( int n ) { if ( n % 15 == 0 ) return true ; return false ; }
public static void main ( String [ ] args ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
public class Politeness {
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
class Test {
static int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
public static void main ( String [ ] args ) { int num = 5 ; System . out . println ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + factorial ( 5 ) ) ; } }
import java . util . * ; class GFG {
static int FlipBits ( int n ) { return n -= ( n & ( - n ) ) ; }
public static void main ( String [ ] args ) { int N = 12 ; System . out . print ( " The ▁ number ▁ after ▁ unsetting ▁ the ▁ " ) ; System . out . print ( " rightmost ▁ set ▁ bit : ▁ " + FlipBits ( N ) ) ; } }
import java . util . HashSet ; class GFG {
static void Maximum_xor_Triplet ( int n , int a [ ] ) {
HashSet < Integer > s = new HashSet < Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i ; j < n ; j ++ ) {
s . add ( a [ i ] ^ a [ j ] ) ; } } int ans = 0 ; for ( Integer i : s ) { for ( int j = 0 ; j < n ; j ++ ) {
ans = Math . max ( ans , i ^ a [ j ] ) ; } } System . out . println ( ans ) ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 3 , 8 , 15 } ; int n = a . length ; Maximum_xor_Triplet ( n , a ) ; } }
import java . util . Arrays ; public class PrintMissing {
static void printMissing ( int ar [ ] , int low , int high ) { Arrays . sort ( ar ) ;
int index = ceilindex ( ar , low , 0 , ar . length - 1 ) ; int x = low ;
while ( index < ar . length && x <= high ) {
if ( ar [ index ] != x ) { System . out . print ( x + " ▁ " ) ; }
else index ++ ;
x ++ ; }
while ( x <= high ) { System . out . print ( x + " ▁ " ) ; x ++ ; } }
static int ceilindex ( int ar [ ] , int val , int low , int high ) { if ( val < ar [ 0 ] ) return 0 ; if ( val > ar [ ar . length - 1 ] ) return ar . length ; int mid = ( low + high ) / 2 ; if ( ar [ mid ] == val ) return mid ; if ( ar [ mid ] < val ) { if ( mid + 1 < high && ar [ mid + 1 ] >= val ) return mid + 1 ; return ceilindex ( ar , val , mid + 1 , high ) ; } else { if ( mid - 1 >= low && ar [ mid - 1 ] < val ) return mid ; return ceilindex ( ar , val , low , mid - 1 ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 5 , 4 } ; int low = 1 , high = 10 ; printMissing ( arr , low , high ) ; } }
import java . util . Arrays ; public class Print {
static void printMissing ( int arr [ ] , int low , int high ) {
boolean [ ] points_of_range = new boolean [ high - low + 1 ] ; for ( int i = 0 ; i < arr . length ; i ++ ) {
if ( low <= arr [ i ] && arr [ i ] <= high ) points_of_range [ arr [ i ] - low ] = true ; }
for ( int x = 0 ; x <= high - low ; x ++ ) { if ( points_of_range [ x ] == false ) System . out . print ( ( low + x ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 5 , 4 } ; int low = 1 , high = 10 ; printMissing ( arr , low , high ) ; } }
import java . util . Arrays ; import java . util . HashSet ; public class Print {
static void printMissing ( int ar [ ] , int low , int high ) { HashSet < Integer > hs = new HashSet < > ( ) ;
for ( int i = 0 ; i < ar . length ; i ++ ) hs . add ( ar [ i ] ) ;
for ( int i = low ; i <= high ; i ++ ) { if ( ! hs . contains ( i ) ) { System . out . print ( i + " ▁ " ) ; } } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 5 , 4 } ; int low = 1 , high = 10 ; printMissing ( arr , low , high ) ; } }
import java . util . * ; class GFG {
static int find ( int a [ ] , int b [ ] , int k , int n1 , int n2 ) {
LinkedHashSet < Integer > s = new LinkedHashSet < > ( ) ; for ( int i = 0 ; i < n2 ; i ++ ) s . add ( b [ i ] ) ;
int missing = 0 ; for ( int i = 0 ; i < n1 ; i ++ ) { if ( ! s . contains ( a [ i ] ) ) missing ++ ; if ( missing == k ) return a [ i ] ; } return - 1 ; }
public static void main ( String [ ] args ) { int a [ ] = { 0 , 2 , 4 , 6 , 8 , 10 , 12 , 14 , 15 } ; int b [ ] = { 4 , 10 , 6 , 8 , 12 } ; int n1 = a . length ; int n2 = b . length ; int k = 3 ; System . out . println ( find ( a , b , k , n1 , n2 ) ) ; } }
import java . util . * ; class GFG {
static void findString ( String S , int N ) {
int [ ] amounts = new int [ 26 ] ;
for ( int i = 0 ; i < 26 ; i ++ ) { amounts [ i ] = 0 ; }
for ( int i = 0 ; i < S . length ( ) ; i ++ ) { amounts [ ( int ) ( S . charAt ( i ) - 97 ) ] ++ ; } int count = 0 ;
for ( int i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) count ++ ; }
if ( count > N ) { System . out . print ( " - 1" ) ; }
else { String ans = " " ; int high = 100001 ; int low = 0 ; int mid , total ;
while ( ( high - low ) > 1 ) { total = 0 ;
mid = ( high + low ) / 2 ;
for ( int i = 0 ; i < 26 ; i ++ ) {
if ( amounts [ i ] > 0 ) { total += ( amounts [ i ] - 1 ) / mid + 1 ; } }
if ( total <= N ) { high = mid ; } else { low = mid ; } } System . out . print ( high + " ▁ " ) ; total = 0 ;
for ( int i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) { total += ( amounts [ i ] - 1 ) / high + 1 ; for ( int j = 0 ; j < ( ( amounts [ i ] - 1 ) / high + 1 ) ; j ++ ) {
ans += ( char ) ( i + 97 ) ; } } }
for ( int i = total ; i < N ; i ++ ) { ans += ' a ' ; } String reverse = " " ; int Len = ans . length ( ) - 1 ; while ( Len >= 0 ) { reverse = reverse + ans . charAt ( Len ) ; Len -- ; }
System . out . print ( reverse ) ; } }
public static void main ( String [ ] args ) { String S = " toffee " ; int K = 4 ; findString ( S , K ) ; } }
import java . util . * ; class Main {
static void printFirstRepeating ( int arr [ ] ) {
int min = - 1 ;
HashSet < Integer > set = new HashSet < > ( ) ;
for ( int i = arr . length - 1 ; i >= 0 ; i -- ) {
if ( set . contains ( arr [ i ] ) ) min = i ;
else set . add ( arr [ i ] ) ; }
if ( min != - 1 ) System . out . println ( " The ▁ first ▁ repeating ▁ element ▁ is ▁ " + arr [ min ] ) ; else System . out . println ( " There ▁ are ▁ no ▁ repeating ▁ elements " ) ; }
public static void main ( String [ ] args ) throws java . lang . Exception { int arr [ ] = { 10 , 5 , 3 , 4 , 3 , 5 , 6 } ; printFirstRepeating ( arr ) ; } }
public class GFG {
static void printFirstRepeating ( int [ ] arr , int n ) {
int k = 0 ;
int max = n ; for ( int i = 0 ; i < n ; i ++ ) if ( max < arr [ i ] ) max = arr [ i ] ;
int [ ] a = new int [ max + 1 ] ;
int [ ] b = new int [ max + 1 ] ; for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ arr [ i ] ] != 0 ) { b [ arr [ i ] ] = 1 ; k = 1 ; continue ; } else
a [ arr [ i ] ] = i ; } if ( k == 0 ) System . out . println ( " No ▁ repeating ▁ element ▁ found " ) ; else { int min = max + 1 ;
for ( int i = 0 ; i < max + 1 ; i ++ ) if ( a [ i ] != 0 && min > a [ i ] && b [ i ] != 0 ) min = a [ i ] ; System . out . print ( arr [ min ] ) ; } System . out . println ( ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 10 , 5 , 3 , 4 , 3 , 5 , 6 } ; int n = arr . length ; printFirstRepeating ( arr , n ) ; } }
class GFG {
static int printKDistinct ( int arr [ ] , int n , int k ) { int dist_count = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return - 1 ; }
public static void main ( String [ ] args ) { int ar [ ] = { 1 , 2 , 1 , 3 , 4 , 2 } ; int n = ar . length ; int k = 2 ; System . out . print ( printKDistinct ( ar , n , k ) ) ; } }
import java . util . Vector ; class GFG {
static void countSubarrays ( int [ ] A ) {
int res = 0 ;
int curr = A [ 0 ] ; int [ ] cnt = new int [ A . length ] ; cnt [ 0 ] = 1 ; for ( int c = 1 ; c < A . length ; c ++ ) {
if ( A == curr )
cnt ++ ; else
curr = A ; cnt = 1 ; }
for ( int i = 1 ; i < cnt . length ; i ++ ) {
res += Math . min ( cnt [ i - 1 ] , cnt [ i ] ) ; } System . out . println ( res - 1 ) ; }
public static void main ( String [ ] args ) {
int [ ] A = { 1 , 1 , 0 , 0 , 1 , 0 } ;
countSubarrays ( A ) ; } }
import java . util . * ; class GfG {
static class Node { int val ; Node left , right ; }
static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . val = data ; temp . left = null ; temp . right = null ; return temp ; }
public static boolean isEvenOddBinaryTree ( Node root ) { if ( root == null ) return true ;
Queue < Node > q = new LinkedList < > ( ) ; q . add ( root ) ;
int level = 0 ;
while ( ! q . isEmpty ( ) ) {
int size = q . size ( ) ; for ( int i = 0 ; i < size ; i ++ ) { Node node = q . poll ( ) ;
if ( level % 2 == 0 ) { if ( node . val % 2 == 1 ) return false ; } else if ( level % 2 == 1 ) { if ( node . val % 2 == 0 ) return false ; }
if ( node . left != null ) { q . add ( node . left ) ; } if ( node . right != null ) { q . add ( node . right ) ; } }
level ++ ; } return true ; }
public static void main ( String [ ] args ) {
Node root = null ; root = newNode ( 2 ) ; root . left = newNode ( 3 ) ; root . right = newNode ( 9 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 10 ) ; root . right . right = newNode ( 6 ) ;
if ( isEvenOddBinaryTree ( root ) ) { System . out . println ( " YES " ) ; } else { System . out . println ( " NO " ) ; } } }
class GFG { static int findMaxLen ( int a [ ] , int n ) {
int freq [ ] = new int [ n + 1 ] ; for ( int i = 0 ; i < n ; ++ i ) { freq [ a [ i ] ] ++ ; } int maxFreqElement = Integer . MIN_VALUE ; int maxFreqCount = 1 ; for ( int i = 1 ; i <= n ; ++ i ) {
if ( freq [ i ] > maxFreqElement ) { maxFreqElement = freq [ i ] ; maxFreqCount = 1 ; }
else if ( freq [ i ] == maxFreqElement ) maxFreqCount ++ ; } int ans ;
if ( maxFreqElement == 1 ) ans = 0 ; else {
ans = ( ( n - maxFreqCount ) / ( maxFreqElement - 1 ) ) ; }
return ans ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 1 , 2 } ; int n = a . length ; System . out . print ( findMaxLen ( a , n ) ) ; } }
import java . util . * ; class GFG {
static int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
static int MaxUtil ( int [ ] st , int ss , int se , int l , int r , int node ) {
if ( l <= ss && r >= se )
return st [ node ] ;
if ( se < l ss > r ) return - 1 ;
int mid = getMid ( ss , se ) ; return Math . max ( MaxUtil ( st , ss , mid , l , r , 2 * node + 1 ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 2 ) ) ; }
static int getMax ( int [ ] st , int n , int l , int r ) {
if ( l < 0 r > n - 1 l > r ) { System . out . printf ( " Invalid ▁ Input " ) ; return - 1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
static int constructSTUtil ( int arr [ ] , int ss , int se , int [ ] st , int si ) {
if ( ss == se ) { st [ si ] = arr [ ss ] ; return arr [ ss ] ; }
int mid = getMid ( ss , se ) ;
st [ si ] = Math . max ( constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) ,
constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ) ; return st [ si ] ; }
static int [ ] constructST ( int arr [ ] , int n ) {
int x = ( int ) ( Math . ceil ( Math . log ( n ) ) ) ;
int max_size = 2 * ( int ) Math . pow ( 2 , x ) - 1 ;
int [ ] st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 3 , 0 } ; int n = arr . length ;
int [ ] st = constructST ( arr , n ) ; int [ ] [ ] Q = { { 1 , 3 } , { 0 , 2 } } ; for ( int i = 0 ; i < Q . length ; i ++ ) { int max = getMax ( st , n , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) ; int ok = 0 ; for ( int j = 30 ; j >= 0 ; j -- ) { if ( ( max & ( 1 << j ) ) != 0 ) ok = 1 ; if ( ok <= 0 ) continue ; max |= ( 1 << j ) ; } System . out . print ( max + " ▁ " ) ; } } }
import java . util . * ; class GFG {
static int calculate ( int a [ ] , int n ) {
Arrays . sort ( a ) ; int count = 1 ; int answer = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( a [ i ] == a [ i - 1 ] ) {
count += 1 ; } else {
answer = answer + ( count * ( count - 1 ) ) / 2 ; count = 1 ; } } answer = answer + ( count * ( count - 1 ) ) / 2 ; return answer ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 1 , 2 , 4 } ; int n = a . length ;
System . out . println ( calculate ( a , n ) ) ; } }
import java . util . * ; class GFG {
static int calculate ( int a [ ] , int n ) {
int maximum = Arrays . stream ( a ) . max ( ) . getAsInt ( ) ;
int frequency [ ] = new int [ maximum + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
frequency [ a [ i ] ] += 1 ; } int answer = 0 ;
for ( int i = 0 ; i < ( maximum ) + 1 ; i ++ ) {
answer = answer + frequency [ i ] * ( frequency [ i ] - 1 ) ; } return answer / 2 ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 1 , 2 , 4 } ; int n = a . length ;
System . out . println ( calculate ( a , n ) ) ; } }
class LargestSubArray {
int findSubArray ( int arr [ ] , int n ) { int sum = 0 ; int maxsize = - 1 , startindex = 0 ; int endindex = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? - 1 : 1 ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] == 0 ) sum += - 1 ; else sum += 1 ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } endindex = startindex + maxsize - 1 ; if ( maxsize == - 1 ) System . out . println ( " No ▁ such ▁ subarray " ) ; else System . out . println ( startindex + " ▁ to ▁ " + endindex ) ; return maxsize ; }
public static void main ( String [ ] args ) { LargestSubArray sub ; sub = new LargestSubArray ( ) ; int arr [ ] = { 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int size = arr . length ; sub . findSubArray ( arr , size ) ; } }
class GFG {
static int findMax ( int arr [ ] , int low , int high ) {
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid == 0 && arr [ mid ] > arr [ mid + 1 ] ) { return arr [ mid ] ; }
if ( arr [ low ] > arr [ mid ] ) { return findMax ( arr , low , mid - 1 ) ; } else { return findMax ( arr , mid + 1 , high ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 6 , 5 , 4 , 3 , 2 , 1 } ; int n = arr . length ; System . out . println ( findMax ( arr , 0 , n - 1 ) ) ; } }
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
import java . util . * ; class GFG {
static int majorityNumber ( int arr [ ] , int n ) { int ans = - 1 ; HashMap < Integer , Integer > freq = new HashMap < Integer , Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( freq . containsKey ( arr [ i ] ) ) { freq . put ( arr [ i ] , freq . get ( arr [ i ] ) + 1 ) ; } else { freq . put ( arr [ i ] , 1 ) ; } if ( freq . get ( arr [ i ] ) > n / 2 ) ans = arr [ i ] ; } return ans ; }
public static void main ( String [ ] args ) { int a [ ] = { 2 , 2 , 1 , 1 , 1 , 2 , 2 } ; int n = a . length ; System . out . println ( majorityNumber ( a , n ) ) ; } }
class Main {
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
import java . util . * ; import java . lang . * ; class GFG {
public static int findMin ( int arr [ ] , int low , int high ) { while ( low < high ) { int mid = low + ( high - low ) / 2 ; if ( arr [ mid ] == arr [ high ] ) high -- ; else if ( arr [ mid ] > arr [ high ] ) low = mid + 1 ; else high = mid ; } return arr [ high ] ; }
public static void main ( String args [ ] ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = arr1 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr1 , 0 , n1 - 1 ) ) ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = arr2 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr2 , 0 , n2 - 1 ) ) ; int arr3 [ ] = { 1 } ; int n3 = arr3 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr3 , 0 , n3 - 1 ) ) ; int arr4 [ ] = { 1 , 2 } ; int n4 = arr4 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr4 , 0 , n4 - 1 ) ) ; int arr5 [ ] = { 2 , 1 } ; int n5 = arr5 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr5 , 0 , n5 - 1 ) ) ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = arr6 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr6 , 0 , n6 - 1 ) ) ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = arr7 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr7 , 0 , n7 - 1 ) ) ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = arr8 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr8 , 0 , n8 - 1 ) ) ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = arr9 . length ; System . out . println ( " The ▁ minimum ▁ element ▁ is ▁ " + findMin ( arr9 , 0 , n9 - 1 ) ) ; } }
import java . util . Scanner ; import java . util . Arrays ; class GFG {
static int countPairs ( int [ ] a , int n , int mid ) { int res = 0 , value ; for ( int i = 0 ; i < n ; i ++ ) {
int ub = upperbound ( a , n , a [ i ] + mid ) ; res += ( ub - ( i - 1 ) ) ; } return res ; }
static int upperbound ( int a [ ] , int n , int value ) { int low = 0 ; int high = n ; while ( low < high ) { final int mid = ( low + high ) / 2 ; if ( value >= a [ mid ] ) low = mid + 1 ; else high = mid ; } return low ; }
static int kthDiff ( int a [ ] , int n , int k ) {
Arrays . sort ( a ) ;
int low = a [ 1 ] - a [ 0 ] ; for ( int i = 1 ; i <= n - 2 ; ++ i ) low = Math . min ( low , a [ i + 1 ] - a [ i ] ) ;
int high = a [ n - 1 ] - a [ 0 ] ;
while ( low < high ) { int mid = ( low + high ) >> 1 ; if ( countPairs ( a , n , mid ) < k ) low = mid + 1 ; else high = mid ; } return low ; }
public static void main ( String args [ ] ) { Scanner s = new Scanner ( System . in ) ; int k = 3 ; int a [ ] = { 1 , 2 , 3 , 4 } ; int n = a . length ; System . out . println ( kthDiff ( a , n , k ) ) ; } }
import java . io . * ; class SecondSmallest {
static void print2Smallest ( int arr [ ] ) { int first , second , arr_size = arr . length ;
if ( arr_size < 2 ) { System . out . println ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = Integer . MAX_VALUE ; for ( int i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == Integer . MAX_VALUE ) System . out . println ( " There ▁ is ▁ no ▁ second " + " smallest ▁ element " ) ; else System . out . println ( " The ▁ smallest ▁ element ▁ is ▁ " + first + " ▁ and ▁ second ▁ Smallest " + " ▁ element ▁ is ▁ " + second ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 12 , 13 , 1 , 10 , 34 , 1 } ; print2Smallest ( arr ) ; } }
class GFG { static final int MAX = 1000 ;
static int tree [ ] = new int [ 4 * MAX ] ;
static int arr [ ] = new int [ MAX ] ;
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
public static void main ( String [ ] args ) {
arr [ 0 ] = 5 ; arr [ 1 ] = 7 ; arr [ 2 ] = 5 ; arr [ 3 ] = 2 ; arr [ 4 ] = 10 ; arr [ 5 ] = 12 ; arr [ 6 ] = 11 ; arr [ 7 ] = 17 ; arr [ 8 ] = 14 ; arr [ 9 ] = 1 ; arr [ 10 ] = 44 ;
build ( 1 , 0 , 10 ) ;
System . out . println ( query ( 1 , 0 , 10 , 2 , 5 ) ) ;
System . out . println ( query ( 1 , 0 , 10 , 5 , 10 ) ) ;
System . out . println ( query ( 1 , 0 , 10 , 0 , 10 ) ) ; } }
import java . io . * ; class GFG { static int M = 1000000007 ; static int waysOfDecoding ( String s ) { long [ ] dp = new long [ s . length ( ) + 1 ] ; dp [ 0 ] = 1 ;
dp [ 1 ] = s . charAt ( 0 ) == ' * ' ? 9 : s . charAt ( 0 ) == '0' ? 0 : 1 ;
for ( int i = 1 ; i < s . length ( ) ; i ++ ) {
if ( s . charAt ( i ) == ' * ' ) { dp [ i + 1 ] = 9 * dp [ i ] ;
if ( s . charAt ( i - 1 ) == '1' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 9 * dp [ i - 1 ] ) % M ;
else if ( s . charAt ( i - 1 ) == '2' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 6 * dp [ i - 1 ] ) % M ;
else if ( s . charAt ( i - 1 ) == ' * ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 15 * dp [ i - 1 ] ) % M ; } else {
dp [ i + 1 ] = s . charAt ( i ) != '0' ? dp [ i ] : 0 ;
if ( s . charAt ( i - 1 ) == '1' ) dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s . charAt ( i - 1 ) == '2' && s . charAt ( i ) <= '6' ) dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s . charAt ( i - 1 ) == ' * ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + ( s . charAt ( i ) <= '6' ? 2 : 1 ) * dp [ i - 1 ] ) % M ; } } return ( int ) dp [ s . length ( ) ] ; }
public static void main ( String [ ] args ) { String s = "12" ; System . out . println ( waysOfDecoding ( s ) ) ; } }
import java . io . * ; public class GFG {
static int countSubset ( int [ ] arr , int n , int diff ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; sum += diff ; sum = sum / 2 ;
int t [ ] [ ] = new int [ n + 1 ] [ sum + 1 ] ;
for ( int j = 0 ; j <= sum ; j ++ ) t [ 0 ] [ j ] = 0 ;
for ( int i = 0 ; i <= n ; i ++ ) t [ i ] [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) {
if ( arr [ i - 1 ] > j ) t [ i ] [ j ] = t [ i - 1 ] [ j ] ; else { t [ i ] [ j ] = t [ i - 1 ] [ j ] + t [ i - 1 ] [ j - arr [ i - 1 ] ] ; } } }
return t [ n ] [ sum ] ; }
public static void main ( String [ ] args ) {
int diff = 1 , n = 4 ; int arr [ ] = { 1 , 1 , 2 , 3 } ;
System . out . print ( countSubset ( arr , n , diff ) ) ; } }
import java . util . * ; class GFG { static float [ ] [ ] dp = new float [ 105 ] [ 605 ] ;
static float find ( int N , int a , int b ) { float probability = 0.0f ;
for ( int i = 1 ; i <= 6 ; i ++ ) dp [ 1 ] [ i ] = ( float ) ( 1.0 / 6 ) ; for ( int i = 2 ; i <= N ; i ++ ) { for ( int j = i ; j <= 6 * i ; j ++ ) { for ( int k = 1 ; k <= 6 && k <= j ; k ++ ) { dp [ i ] [ j ] = dp [ i ] [ j ] + dp [ i - 1 ] [ j - k ] / 6 ; } } }
for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + dp [ N ] [ sum ] ; return probability ; }
public static void main ( String [ ] args ) { int N = 4 , a = 13 , b = 17 ; float probability = find ( N , a , b ) ;
System . out . printf ( " % .6f " , probability ) ; } }
import java . util . * ; public class Main {
static class Node { int data ; Node left , right ; Node ( int item ) { data = item ; left = right = null ; } }
public static int getSumAlternate ( Node root ) { if ( root == null ) return 0 ; int sum = root . data ; if ( root . left != null ) { sum += getSum ( root . left . left ) ; sum += getSum ( root . left . right ) ; } if ( root . right != null ) { sum += getSum ( root . right . left ) ; sum += getSum ( root . right . right ) ; } return sum ; }
public static int getSum ( Node root ) { if ( root == null ) return 0 ;
return Math . max ( getSumAlternate ( root ) , ( getSumAlternate ( root . left ) + getSumAlternate ( root . right ) ) ) ; }
public static void main ( String [ ] args ) { Node root = new Node ( 1 ) ; root . left = new Node ( 2 ) ; root . right = new Node ( 3 ) ; root . right . left = new Node ( 4 ) ; root . right . left . right = new Node ( 5 ) ; root . right . left . right . left = new Node ( 6 ) ; System . out . println ( getSum ( root ) ) ; } }
public class Subset_sum {
static boolean isSubsetSum ( int arr [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ 2 ] [ sum + 1 ] ; for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) {
if ( j == 0 ) subset [ i % 2 ] [ j ] = true ;
else if ( i == 0 ) subset [ i % 2 ] [ j ] = false ; else if ( arr [ i - 1 ] <= j ) subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] || subset [ ( i + 1 ) % 2 ] [ j ] ; else subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] ; } } return subset [ n % 2 ] [ sum ] ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 2 , 5 } ; int sum = 7 ; int n = arr . length ; if ( isSubsetSum ( arr , n , sum ) == true ) System . out . println ( " There ▁ exists ▁ a ▁ subset ▁ with " + " given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ exists ▁ with " + " given ▁ sum " ) ; } }
import java . io . * ; class GFG {
static int findMaxSum ( int [ ] arr , int n ) { int res = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { int prefix_sum = arr [ i ] ; for ( int j = 0 ; j < i ; j ++ ) prefix_sum += arr [ j ] ; int suffix_sum = arr [ i ] ; for ( int j = n - 1 ; j > i ; j -- ) suffix_sum += arr [ j ] ; if ( prefix_sum == suffix_sum ) res = Math . max ( res , prefix_sum ) ; } return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . length ; System . out . println ( findMaxSum ( arr , n ) ) ; } }
import java . io . * ; public class GFG {
static int findMaxSum ( int [ ] arr , int n ) {
int [ ] preSum = new int [ n ] ;
int [ ] suffSum = new int [ n ] ;
int ans = Integer . MIN_VALUE ;
preSum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) preSum [ i ] = preSum [ i - 1 ] + arr [ i ] ;
suffSum [ n - 1 ] = arr [ n - 1 ] ; if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) ans = Math . max ( ans , preSum [ n - 1 ] ) ; for ( int i = n - 2 ; i >= 0 ; i -- ) { suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] ; if ( suffSum [ i ] == preSum [ i ] ) ans = Math . max ( ans , preSum [ i ] ) ; } return ans ; }
static public void main ( String [ ] args ) { int [ ] arr = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . length ; System . out . println ( findMaxSum ( arr , n ) ) ; } }
import java . lang . Math . * ; import java . util . stream . * ; class GFG {
static int findMaxSum ( int arr [ ] , int n ) { int sum = IntStream . of ( arr ) . sum ( ) ; int prefix_sum = 0 , res = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { prefix_sum += arr [ i ] ; if ( prefix_sum == sum ) res = Math . max ( res , prefix_sum ) ; sum -= arr [ i ] ; } return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { - 2 , 5 , 3 , 1 , 2 , 6 , - 4 , 2 } ; int n = arr . length ; System . out . print ( findMaxSum ( arr , n ) ) ; } }
import java . io . * ; class GFG {
static void findMajority ( int arr [ ] , int n ) { int maxCount = 0 ;
int index = - 1 ; for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) count ++ ; }
if ( count > maxCount ) { maxCount = count ; index = i ; } }
if ( maxCount > n / 2 ) System . out . println ( arr [ index ] ) ; else System . out . println ( " No ▁ Majority ▁ Element " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = arr . length ;
findMajority ( arr , n ) ; } }
import java . io . * ; class Node { int key ; int c = 0 ; Node left , right ; } class GFG { static int ma = 0 ;
static Node newNode ( int item ) { Node temp = new Node ( ) ; temp . key = item ; temp . c = 1 ; temp . left = temp . right = null ; return temp ; }
static Node insert ( Node node , int key ) {
if ( node == null ) { if ( ma == 0 ) ma = 1 ; return newNode ( key ) ; }
if ( key < node . key ) node . left = insert ( node . left , key ) ; else if ( key > node . key ) node . right = insert ( node . right , key ) ; else node . c ++ ;
ma = Math . max ( ma , node . c ) ;
return node ; }
static void inorder ( Node root , int s ) { if ( root != null ) { inorder ( root . left , s ) ; if ( root . c > ( s / 2 ) ) System . out . println ( root . key + "NEW_LINE"); inorder ( root . right , s ) ; } }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 3 , 3 , 3 , 2 } ; int size = a . length ; Node root = null ; for ( int i = 0 ; i < size ; i ++ ) { root = insert ( root , a [ i ] ) ; }
if ( ma > ( size / 2 ) ) inorder ( root , size ) ; else System . out . println ( "No majority elementNEW_LINE"); } }
class MajorityElement {
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
boolean isMajority ( int a [ ] , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) { if ( a [ i ] == cand ) count ++ ; } if ( count > size / 2 ) return true ; else return false ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) System . out . println ( " ▁ " + cand + " ▁ " ) ; else System . out . println ( " No ▁ Majority ▁ Element " ) ; }
public static void main ( String [ ] args ) { MajorityElement majorelement = new MajorityElement ( ) ; int a [ ] = new int [ ] { 1 , 3 , 3 , 1 , 2 } ;
int size = a . length ; majorelement . printMajority ( a , size ) ; } }
import java . util . HashMap ; class MajorityElement { private static void findMajority ( int [ ] arr ) { HashMap < Integer , Integer > map = new HashMap < Integer , Integer > ( ) ; for ( int i = 0 ; i < arr . length ; i ++ ) { if ( map . containsKey ( arr [ i ] ) ) { int count = map . get ( arr [ i ] ) + 1 ; if ( count > arr . length / 2 ) { System . out . println ( " Majority ▁ found ▁ : - ▁ " + arr [ i ] ) ; return ; } else map . put ( arr [ i ] , count ) ; } else map . put ( arr [ i ] , 1 ) ; } System . out . println ( " ▁ No ▁ Majority ▁ element " ) ; }
public static void main ( String [ ] args ) { int a [ ] = new int [ ] { 2 , 2 , 2 , 2 , 5 , 5 , 2 , 3 , 3 } ;
findMajority ( a ) ; } }
import java . io . * ; import java . util . * ; class GFG {
public static int majorityElement ( int [ ] arr , int n ) {
Arrays . sort ( arr ) ; int count = 1 , max_ele = - 1 , temp = arr [ 0 ] , ele = 0 , f = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( temp == arr [ i ] ) { count ++ ; } else { count = 1 ; temp = arr [ i ] ; }
if ( max_ele < count ) { max_ele = count ; ele = arr [ i ] ; if ( max_ele > ( n / 2 ) ) { f = 1 ; break ; } } }
return ( f == 1 ? ele : - 1 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = 7 ;
System . out . println ( majorityElement ( arr , n ) ) ; } }
class GFG {
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ sum + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
class GFG {
static int subsetSum ( int a [ ] , int n , int sum ) {
int tab [ ] [ ] = new int [ n + 1 ] [ sum + 1 ] ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) { tab [ i ] [ j ] = - 1 ; } }
if ( sum == 0 ) return 1 ; if ( n <= 0 ) return 0 ;
if ( tab [ n - 1 ] [ sum ] != - 1 ) return tab [ n - 1 ] [ sum ] ;
if ( a [ n - 1 ] > sum ) return tab [ n - 1 ] [ sum ] = subsetSum ( a , n - 1 , sum ) ; else {
if ( subsetSum ( a , n - 1 , sum ) != 0 || subsetSum ( a , n - 1 , sum - a [ n - 1 ] ) != 0 ) { return tab [ n - 1 ] [ sum ] = 1 ; } else return tab [ n - 1 ] [ sum ] = 0 ; } }
public static void main ( String [ ] args ) { int n = 5 ; int a [ ] = { 1 , 5 , 3 , 7 , 4 } ; int sum = 12 ; if ( subsetSum ( a , n , sum ) != 0 ) { System . out . println ( "YESNEW_LINE"); } else System . out . println ( "NONEW_LINE"); } }
import java . io . * ; import java . lang . Math ; class GFG {
static int binpow ( int a , int b ) { int res = 1 ; while ( b > 0 ) { if ( b % 2 == 1 ) res = res * a ; a = a * a ; b /= 2 ; } return res ; }
static int find ( int x ) { if ( x == 0 ) return 0 ; int p = ( int ) ( Math . log ( x ) / Math . log ( 2 ) ) ; return binpow ( 2 , p + 1 ) - 1 ; }
static String getBinary ( int n ) {
String ans = " " ;
while ( n > 0 ) { int dig = n % 2 ; ans += dig ; n /= 2 ; }
return ans ; }
static int totalCountDifference ( int n ) {
String ans = getBinary ( n ) ;
int req = 0 ;
for ( int i = 0 ; i < ans . length ( ) ; i ++ ) {
if ( ans . charAt ( i ) == '1' ) { req += find ( binpow ( 2 , i ) ) ; } } return req ; }
public static void main ( String [ ] args ) {
int n = 5 ;
System . out . print ( totalCountDifference ( n ) ) ; } }
import java . util . * ; public class Main {
public static int Maximum_Length ( Vector < Integer > a ) {
int [ ] counts = new int [ 11 ] ;
int ans = 0 ; for ( int index = 0 ; index < a . size ( ) ; index ++ ) {
counts [ a . get ( index ) ] += 1 ;
Vector < Integer > k = new Vector < Integer > ( ) ; for ( int i : counts ) if ( i != 0 ) k . add ( i ) ; Collections . sort ( k ) ;
if ( k . size ( ) == 1 || ( k . get ( 0 ) == k . get ( k . size ( ) - 2 ) && k . get ( k . size ( ) - 1 ) - k . get ( k . size ( ) - 2 ) == 1 ) || ( k . get ( 0 ) == 1 && k . get ( 1 ) == k . get ( k . size ( ) - 1 ) ) ) ans = index ; }
return ans + 1 ; }
public static void main ( String [ ] args ) { Vector < Integer > a = new Vector < Integer > ( ) ; a . add ( 1 ) ; a . add ( 1 ) ; a . add ( 1 ) ; a . add ( 2 ) ; a . add ( 2 ) ; a . add ( 2 ) ; System . out . println ( Maximum_Length ( a ) ) ; } }
class GFG {
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
public static void main ( String [ ] args ) {
SieveOfEratosthenes ( ) ;
int l = 3 , r = 9 ;
int c = ( sum [ r ] - sum [ l - 1 ] ) ;
System . out . println ( " Count : ▁ " + c ) ; } }
import java . io . * ; class GFG {
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
import java . util . * ; import java . lang . * ; import java . io . * ; class GFG {
static int sumOfDigitsSingle ( int x ) { int ans = 0 ; while ( x != 0 ) { ans += x % 10 ; x /= 10 ; } return ans ; }
static int closest ( int x ) { int ans = 0 ; while ( ans * 10 + 9 <= x ) ans = ans * 10 + 9 ; return ans ; } static int sumOfDigitsTwoParts ( int N ) { int A = closest ( N ) ; return sumOfDigitsSingle ( A ) + sumOfDigitsSingle ( N - A ) ; }
public static void main ( String args [ ] ) { int N = 35 ; System . out . print ( sumOfDigitsTwoParts ( N ) ) ; } }
class GFG {
static boolean isPrime ( int p ) {
double checkNumber = Math . pow ( 2 , p ) - 1 ;
double nextval = 4 % checkNumber ;
for ( int i = 1 ; i < p - 1 ; i ++ ) nextval = ( nextval * nextval - 2 ) % checkNumber ;
return ( nextval == 0 ) ; }
public static void main ( String [ ] args ) {
int p = 7 ; double checkNumber = Math . pow ( 2 , p ) - 1 ; if ( isPrime ( p ) ) System . out . println ( ( int ) checkNumber + " ▁ is ▁ Prime . " ) ; else System . out . println ( ( int ) checkNumber + " ▁ is ▁ not ▁ Prime . " ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static void sieve ( int n , boolean prime [ ] ) { for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < n ; i += p ) prime [ i ] = false ; } } } static void printSophieGermanNumber ( int n ) {
boolean prime [ ] = new boolean [ 2 * n + 1 ] ; Arrays . fill ( prime , true ) ; sieve ( 2 * n + 1 , prime ) ; for ( int i = 2 ; i < n ; ++ i ) {
if ( prime [ i ] && prime [ 2 * i + 1 ] ) System . out . print ( i + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int n = 25 ; printSophieGermanNumber ( n ) ; } }
import java . text . * ; class GFG {
static double ucal ( double u , int n ) { if ( n == 0 ) return 1 ; double temp = u ; for ( int i = 1 ; i <= n / 2 ; i ++ ) temp = temp * ( u - i ) ; for ( int i = 1 ; i < n / 2 ; i ++ ) temp = temp * ( u + i ) ; return temp ; }
static int fact ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
public static void main ( String [ ] args ) {
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
import java . util . * ; class GFG {
static boolean powerOf2 ( int n ) {
if ( n == 1 ) return true ;
else if ( n % 2 != 0 n == 0 ) return false ;
return powerOf2 ( n / 2 ) ; }
public static void main ( String [ ] args ) {
int n = 64 ;
int m = 12 ; if ( powerOf2 ( n ) == true ) System . out . print ( " True " + "NEW_LINE"); else System . out . print ( " False " + "NEW_LINE"); if ( powerOf2 ( m ) == true ) System . out . print ( " True " + "NEW_LINE"); else System . out . print ( " False " + "NEW_LINE"); } }
class Test {
static boolean isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void main ( String [ ] args ) { System . out . println ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; System . out . println ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
import java . io . * ; class GFG {
static boolean isPowerofTwo ( int n ) { if ( n == 0 ) return false ; if ( ( n & ( ~ ( n - 1 ) ) ) == n ) return true ; return false ; }
public static void main ( String [ ] args ) { if ( isPowerofTwo ( 30 ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; if ( isPowerofTwo ( 128 ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
class GFG {
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
class GFG {
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
public static void main ( String args [ ] ) { int A = 2 , B = 4 , C = 5 , K = 5 ; System . out . println ( findKthMultiple ( A , B , C , K ) ) ; } }
import java . util . * ; class GFG {
static void variationStalinsort ( Vector < Integer > arr ) { int j = 0 ; while ( true ) { int moved = 0 ; for ( int i = 0 ; i < ( arr . size ( ) - 1 - j ) ; i ++ ) { if ( arr . get ( i ) > arr . get ( i + 1 ) ) {
int index ; int temp ; index = arr . get ( i ) ; temp = arr . get ( i + 1 ) ; arr . removeElement ( index ) ; arr . add ( i , temp ) ; arr . removeElement ( temp ) ; arr . add ( i + 1 , index ) ; moved ++ ; } } j ++ ; if ( moved == 0 ) { break ; } } System . out . print ( arr ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 2 , 1 , 4 , 3 , 6 , 5 , 8 , 7 , 10 , 9 } ; Vector < Integer > arr1 = new Vector < > ( ) ; for ( int i = 0 ; i < arr . length ; i ++ ) arr1 . add ( arr [ i ] ) ;
variationStalinsort ( arr1 ) ; } }
class Main {
public static void printArray ( int arr [ ] , int N ) {
for ( int i = 0 ; i < N ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void sortArray ( int arr [ ] , int N ) {
for ( int i = 0 ; i < N ; ) {
if ( arr [ i ] == i + 1 ) { i ++ ; }
else {
int temp1 = arr [ i ] ; int temp2 = arr [ arr [ i ] - 1 ] ; arr [ i ] = temp2 ; arr [ temp1 - 1 ] = temp1 ; } } }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 1 , 5 , 3 , 4 } ; int N = arr . length ;
sortArray ( arr , N ) ;
printArray ( arr , N ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int maximum ( int value [ ] , int weight [ ] , int weight1 , int flag , int K , int index ) {
if ( index >= value . length ) { return 0 ; }
if ( flag == K ) {
int skip = maximum ( value , weight , weight1 , flag , K , index + 1 ) ; int full = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 ) ; }
return Math . max ( full , skip ) ; }
else {
int skip = maximum ( value , weight , weight1 , flag , K , index + 1 ) ; int full = 0 ; int half = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 ) ; }
if ( weight [ index ] / 2 <= weight1 ) { half = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] / 2 , flag , K , index + 1 ) ; }
return Math . max ( full , Math . max ( skip , half ) ) ; } }
public static void main ( String [ ] args ) throws Exception { int value [ ] = { 17 , 20 , 10 , 15 } ; int weight [ ] = { 4 , 2 , 7 , 5 } ; int K = 1 ; int W = 4 ; System . out . println ( maximum ( value , weight , W , 0 , K , 0 ) ) ; } }
import java . util . * ; class GFG { static final int N = 1005 ;
static class Node { int data ; Node left , right ; } ;
static Node newNode ( int data ) { Node node = new Node ( ) ; node . data = data ; node . left = node . right = null ; return node ; }
static int [ ] [ ] [ ] dp = new int [ N ] [ 5 ] [ 5 ] ;
static int minDominatingSet ( Node root , int covered , int compulsory ) {
if ( root == null ) return 0 ;
if ( root . left != null && root . right != null && covered > 0 ) compulsory = 1 ;
if ( dp [ root . data ] [ covered ] [ compulsory ] != - 1 ) return dp [ root . data ] [ covered ] [ compulsory ] ;
if ( compulsory > 0 ) {
return dp [ root . data ] [ covered ] [ compulsory ] = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; }
if ( covered > 0 ) { return dp [ root . data ] [ covered ] [ compulsory ] = Math . min ( 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; }
int ans = 1 + minDominatingSet ( root . left , 1 , 0 ) + minDominatingSet ( root . right , 1 , 0 ) ; if ( root . left != null ) { ans = Math . min ( ans , minDominatingSet ( root . left , 0 , 1 ) + minDominatingSet ( root . right , 0 , 0 ) ) ; } if ( root . right != null ) { ans = Math . min ( ans , minDominatingSet ( root . left , 0 , 0 ) + minDominatingSet ( root . right , 0 , 1 ) ) ; }
return dp [ root . data ] [ covered ] [ compulsory ] = ans ; }
public static void main ( String [ ] args ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < 5 ; j ++ ) { for ( int l = 0 ; l < 5 ; l ++ ) dp [ i ] [ j ] [ l ] = - 1 ; } }
Node root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . left . left = newNode ( 3 ) ; root . left . right = newNode ( 4 ) ; root . left . left . left = newNode ( 5 ) ; root . left . left . left . left = newNode ( 6 ) ; root . left . left . left . right = newNode ( 7 ) ; root . left . left . left . right . right = newNode ( 10 ) ; root . left . left . left . left . left = newNode ( 8 ) ; root . left . left . left . left . right = newNode ( 9 ) ; System . out . print ( minDominatingSet ( root , 0 , 0 ) + "NEW_LINE"); } }
class GFG { static int maxSum = 100 ; static int arrSize = 51 ;
static int [ ] [ ] dp = new int [ arrSize ] [ maxSum ] ; static boolean [ ] [ ] visit = new boolean [ arrSize ] [ maxSum ] ;
static int SubsetCnt ( int i , int s , int arr [ ] , int n ) {
if ( i == n ) { if ( s == 0 ) { return 1 ; } else { return 0 ; } }
if ( visit [ i ] [ s + arrSize ] ) { return dp [ i ] [ s + arrSize ] ; }
visit [ i ] [ s + arrSize ] = true ;
dp [ i ] [ s + arrSize ] = SubsetCnt ( i + 1 , s + arr [ i ] , arr , n ) + SubsetCnt ( i + 1 , s , arr , n ) ;
return dp [ i ] [ s + arrSize ] ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 2 , 2 , - 4 , - 4 } ; int n = arr . length ; System . out . println ( SubsetCnt ( 0 , 0 , arr , n ) ) ; } }
class solution { static final int MAX = 1000 ;
static int waysToKAdjacentSetBits ( int dp [ ] [ ] [ ] , int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } if ( dp [ currentIndex ] [ adjacentSetBits ] [ lastBit ] != - 1 ) { return dp [ currentIndex ] [ adjacentSetBits ] [ lastBit ] ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( lastBit == 0 ) { noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( dp , n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } dp [ currentIndex ] [ adjacentSetBits ] [ lastBit ] = noOfWays ; return noOfWays ; }
public static void main ( String args [ ] ) { int n = 5 , k = 2 ;
int dp [ ] [ ] [ ] = new int [ MAX ] [ MAX ] [ 2 ] ;
for ( int i = 0 ; i < MAX ; i ++ ) for ( int j = 0 ; j < MAX ; j ++ ) for ( int k1 = 0 ; k1 < 2 ; k1 ++ ) dp [ i ] [ j ] [ k1 ] = - 1 ;
int totalWays = waysToKAdjacentSetBits ( dp , n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( dp , n , k , 1 , 0 , 0 ) ; System . out . print ( " Number ▁ of ▁ ways ▁ = ▁ " + totalWays + "NEW_LINE"); } }
import java . io . * ; import java . util . * ; import java . lang . * ; class GFG {
static void printTetra ( int n ) { if ( n < 0 ) return ;
int first = 0 , second = 1 ; int third = 1 , fourth = 2 ;
int curr = 0 ; if ( n == 0 ) System . out . print ( first ) ; else if ( n == 1 n == 2 ) System . out . print ( second ) ; else if ( n == 3 ) System . out . print ( fourth ) ; else {
for ( int i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } System . out . print ( curr ) ; } }
public static void main ( String [ ] args ) { int n = 10 ; printTetra ( n ) ; } }
import java . lang . * ; import java . util . * ; public class GfG {
public static int countWays ( int n ) { int [ ] res = new int [ n + 1 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
public static void main ( String argc [ ] ) { int n = 4 ; System . out . println ( countWays ( n ) ) ; } }
import java . io . * ; class GFG {
static int countWays ( int n ) {
int a = 1 , b = 2 , c = 4 ;
int d = 0 ; if ( n == 0 n == 1 n == 2 ) return n ; if ( n == 3 ) return c ;
for ( int i = 4 ; i <= n ; i ++ ) { d = c + b + a ; a = b ; b = c ; c = d ; } return d ; }
public static void main ( String [ ] args ) { int n = 4 ; System . out . println ( countWays ( n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static boolean isPossible ( int elements [ ] , int sum ) { int dp [ ] = new int [ sum + 1 ] ;
dp [ 0 ] = 1 ;
for ( int i = 0 ; i < elements . length ; i ++ ) {
for ( int j = sum ; j >= elements [ i ] ; j -- ) { if ( dp [ j - elements [ i ] ] == 1 ) dp [ j ] = 1 ; } }
if ( dp [ sum ] == 1 ) return true ; return false ; }
public static void main ( String [ ] args ) throws Exception { int elements [ ] = { 6 , 2 , 5 } ; int sum = 7 ; if ( isPossible ( elements , sum ) ) System . out . println ( " YES " ) ; else System . out . println ( " NO " ) ; } }
class GFG {
static int maxTasks ( int high [ ] , int low [ ] , int n ) {
if ( n <= 0 ) return 0 ;
return Math . max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; System . out . println ( maxTasks ( high , low , n ) ) ; } }
import java . util . * ; class GFG {
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; } static int nCr ( int n , int r ) {
if ( r > n ) return 0 ;
if ( r > n - r ) C ( n , r ) = C ( n , n - r ) r = n - r ; int mod = 1000000007 ;
int [ ] arr = new int [ r ] ; for ( int i = n - r + 1 ; i <= n ; i ++ ) { arr [ i + r - n - 1 ] = i ; } long ans = 1 ;
for ( int k = 1 ; k < r + 1 ; k ++ ) { int j = 0 , i = k ; while ( j < arr . length ) { int x = gcd ( i , arr [ j ] ) ; if ( x > 1 ) {
arr [ j ] /= x ; i /= x ; } if ( i == 1 )
break ; j += 1 ; } }
ans = ( ans * i ) % mod ; return ( int ) ans ; }
public static void main ( String [ ] args ) { int n = 5 , r = 2 ; System . out . print ( " Value ▁ of ▁ C ( " + n + " , ▁ " + r + " ) ▁ is ▁ " + nCr ( n , r ) + "NEW_LINE"); } }
class GFG {
static char FindKthChar ( String str , int K , int X ) {
char ans = ' ▁ ' ; int sum = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
int digit = ( int ) str . charAt ( i ) - 48 ;
int range = ( int ) Math . pow ( digit , X ) ; sum += range ;
if ( K <= sum ) { ans = str . charAt ( i ) ; break ; } }
return ans ; }
public static void main ( String [ ] args ) {
String str = "123" ; int K = 9 ; int X = 3 ;
char ans = FindKthChar ( str , K , X ) ; System . out . println ( ans ) ; } }
class GFG {
static int totalPairs ( String s1 , String s2 ) { int count = 0 ; int [ ] arr1 = new int [ 7 ] ; int [ ] arr2 = new int [ 7 ] ;
for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) { int set_bits = Integer . bitCount ( s1 . charAt ( i ) ) ; arr1 [ set_bits ] ++ ; }
for ( int i = 0 ; i < s2 . length ( ) ; i ++ ) { int set_bits = Integer . bitCount ( s2 . charAt ( i ) ) ; arr2 [ set_bits ] ++ ; }
for ( int i = 1 ; i <= 6 ; i ++ ) { count += ( arr1 [ i ] * arr2 [ i ] ) ; }
return count ; }
public static void main ( String [ ] args ) { String s1 = " geeks " ; String s2 = " forgeeks " ; System . out . println ( totalPairs ( s1 , s2 ) ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class GFG {
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
import java . io . * ; class GFG {
static int nthEnneadecagonal ( int n ) {
return ( 17 * n * n - 15 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 6 ; System . out . print ( n + " th ▁ Enneadecagonal ▁ number ▁ : " ) ; System . out . println ( nthEnneadecagonal ( n ) ) ; } }
import java . io . * ; class Gfg {
static float areacircumscribed ( float a ) { float PI = 3.14159265f ; return ( a * a * ( PI / 2 ) ) ; }
public static void main ( String arg [ ] ) { float a = 6 ; System . out . print ( " Area ▁ of ▁ an ▁ circumscribed " + " circle ▁ is ▁ : " ) ; System . out . println ( areacircumscribed ( a ) ) ; } }
import java . util . * ; class GFG {
static int itemType ( int n ) {
int count = 0 ; int day = 1 ;
while ( count + day * ( day + 1 ) / 2 < n ) {
count += day * ( day + 1 ) / 2 ; day ++ ; } for ( int type = day ; type > 0 ; type -- ) {
count += type ;
if ( count >= n ) { return type ; } } return 0 ; }
public static void main ( String [ ] args ) { int N = 10 ; System . out . println ( itemType ( N ) ) ; } }
class GFG {
static class Node { int data ; Node next ; } ;
static boolean isSortedDesc ( Node head ) { if ( head == null ) return true ;
for ( Node t = head ; t . next != null ; t = t . next ) if ( t . data <= t . next . data ) return false ; return true ; } static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . next = null ; temp . data = data ; return temp ; }
public static void main ( String [ ] args ) { Node head = newNode ( 7 ) ; head . next = newNode ( 5 ) ; head . next . next = newNode ( 4 ) ; head . next . next . next = newNode ( 3 ) ; if ( isSortedDesc ( head ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
public class GFG {
static int maxLength ( String str , int n , char c , int k ) {
int ans = - 1 ;
int cnt = 0 ;
int left = 0 ; for ( int right = 0 ; right < n ; right ++ ) { if ( str . charAt ( right ) == c ) { cnt ++ ; }
while ( cnt > k ) { if ( str . charAt ( left ) == c ) { cnt -- ; }
left ++ ; }
ans = Math . max ( ans , right - left + 1 ) ; } return ans ; }
static int maxConsecutiveSegment ( String S , int K ) { int N = S . length ( ) ;
return Math . max ( maxLength ( S , N , '0' , K ) , maxLength ( S , N , '1' , K ) ) ; }
int main ( ) { return 0 ; } public static void main ( String [ ] args ) { String S = "1001" ; int K = 1 ; System . out . println ( maxConsecutiveSegment ( S , K ) ) ; } }
import java . util . * ; class GFG {
static void find ( int N ) { int T , F , O ;
F = ( int ) ( ( N - 4 ) / 5 ) ;
if ( ( ( N - 5 * F ) % 2 ) == 0 ) { O = 2 ; } else { O = 1 ; }
T = ( int ) Math . floor ( ( N - 5 * F - O ) / 2 ) ; System . out . println ( " Count ▁ of ▁ 5 ▁ valueds ▁ coins : ▁ " + F ) ; System . out . println ( " Count ▁ of ▁ 2 ▁ valueds ▁ coins : ▁ " + T ) ; System . out . println ( " Count ▁ of ▁ 1 ▁ valueds ▁ coins : ▁ " + O ) ; }
public static void main ( String args [ ] ) { int N = 8 ; find ( N ) ; } }
class GFG {
static void findMaxOccurence ( char [ ] str , int N ) {
for ( int i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' ? ' ) {
str [ i ] = '0' ; } } System . out . print ( str ) ; }
public static void main ( String [ ] args ) {
String str = "10?0?11" ; int N = str . length ( ) ; findMaxOccurence ( str . toCharArray ( ) , N ) ; } }
class GFG {
public static void checkInfinite ( String s ) {
boolean flag = true ; int N = s . length ( ) ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( s . charAt ( i ) == ( char ) ( ( int ) ( s . charAt ( i + 1 ) ) + 1 ) ) { continue ; }
else if ( s . charAt ( i ) == ' a ' && s . charAt ( i + 1 ) == ' z ' ) { continue ; }
else { flag = false ; break ; } }
if ( ! flag ) System . out . print ( " NO " ) ; else System . out . print ( " YES " ) ; }
public static void main ( String [ ] args ) {
String s = " ecbaz " ;
checkInfinite ( s ) ; } }
class GFG {
static int minChangeInLane ( int barrier [ ] , int n ) { int dp [ ] = { 1 , 0 , 1 } ; for ( int j = 0 ; j < n ; j ++ ) {
int val = barrier [ j ] ; if ( val > 0 ) { dp [ val - 1 ] = ( int ) 1e6 ; } for ( int i = 0 ; i < 3 ; i ++ ) {
if ( val != i + 1 ) { dp [ i ] = Math . min ( dp [ i ] , Math . min ( dp [ ( i + 1 ) % 3 ] , dp [ ( i + 2 ) % 3 ] ) + 1 ) ; } } }
return Math . min ( dp [ 0 ] , Math . min ( dp [ 1 ] , dp [ 2 ] ) ) ; }
public static void main ( String [ ] args ) { int barrier [ ] = { 0 , 1 , 2 , 3 , 0 } ; int N = barrier . length ; System . out . print ( minChangeInLane ( barrier , N ) ) ; } }
import java . util . * ; public class Main {
public static void numWays ( int [ ] [ ] ratings , int queries [ ] [ ] , int n , int k ) {
int dp [ ] [ ] = new int [ n ] [ 10000 + 2 ] ;
for ( int i = 0 ; i < k ; i ++ ) dp [ 0 ] [ ratings [ 0 ] [ i ] ] += 1 ;
for ( int i = 1 ; i < n ; i ++ ) {
for ( int sum = 0 ; sum <= 10000 ; sum ++ ) {
for ( int j = 0 ; j < k ; j ++ ) {
if ( sum >= ratings [ i ] [ j ] ) dp [ i ] [ sum ] += dp [ i - 1 ] [ sum - ratings [ i ] [ j ] ] ; } } }
for ( int sum = 1 ; sum <= 10000 ; sum ++ ) { dp [ n - 1 ] [ sum ] += dp [ n - 1 ] [ sum - 1 ] ; }
for ( int q = 0 ; q < queries . length ; q ++ ) { int a = queries [ q ] [ 0 ] ; int b = queries [ q ] [ 1 ] ;
System . out . print ( dp [ n - 1 ] [ b ] - dp [ n - 1 ] [ a - 1 ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) {
int N = 2 , K = 3 ;
int ratings [ ] [ ] = { { 1 , 2 , 3 } , { 4 , 5 , 6 } } ;
int queries [ ] [ ] = { { 6 , 6 } , { 1 , 6 } } ;
numWays ( ratings , queries , N , K ) ; } }
import java . io . * ; class GFG {
static void numberOfPermWithKInversion ( int N , int K ) {
int [ ] [ ] dp = new int [ 2 ] [ K + 1 ] ; int mod = 1000000007 ; for ( int i = 1 ; i <= N ; i ++ ) { for ( int j = 0 ; j <= K ; j ++ ) {
if ( i == 1 ) { dp [ i % 2 ] [ j ] = ( j == 0 ) ? 1 : 0 ; }
else if ( j == 0 ) dp [ i % 2 ] [ j ] = 1 ;
else { int maxm = Math . max ( j - ( i - 1 ) ) ; dp [ i % 2 ] [ j ] = ( dp [ i % 2 ] [ j - 1 ] % mod + ( dp [ 1 - i % 2 ] [ j ] - ( ( Math . max ( j - ( i - 1 ) , 0 ) == 0 ) ? 0 : dp [ 1 - i % 2 ] [ maxm , 0 ) - 1 ] ) + mod ) % mod ) % mod ; } } }
System . out . println ( dp [ N % 2 ] [ K ] ) ; }
public static void main ( String [ ] args ) {
int N = 3 , K = 2 ;
numberOfPermWithKInversion ( N , K ) ; } }
class GFG { static final int N = 100 ; static int n , m ;
static int a [ ] [ ] = new int [ N ] [ N ] ;
static int dp [ ] [ ] = new int [ N ] [ N ] ; static int visited [ ] [ ] = new int [ N ] [ N ] ;
static int current_sum = 0 ;
static int total_sum = 0 ;
static void inputMatrix ( ) { n = 3 ; m = 3 ; a [ 0 ] [ 0 ] = 500 ; a [ 0 ] [ 1 ] = 100 ; a [ 0 ] [ 2 ] = 230 ; a [ 1 ] [ 0 ] = 1000 ; a [ 1 ] [ 1 ] = 300 ; a [ 1 ] [ 2 ] = 100 ; a [ 2 ] [ 0 ] = 200 ; a [ 2 ] [ 1 ] = 1000 ; a [ 2 ] [ 2 ] = 200 ; }
static int maximum_sum_path ( int i , int j ) {
if ( i == n - 1 && j == m - 1 ) return a [ i ] [ j ] ;
if ( visited [ i ] [ j ] != 0 ) return dp [ i ] [ j ] ;
visited [ i ] [ j ] = 1 ; int total_sum = 0 ;
if ( i < n - 1 & j < m - 1 ) { int current_sum = Math . max ( maximum_sum_path ( i , j + 1 ) , Math . max ( maximum_sum_path ( i + 1 , j + 1 ) , maximum_sum_path ( i + 1 , j ) ) ) ; total_sum = a [ i ] [ j ] + current_sum ; }
else if ( i == n - 1 ) total_sum = a [ i ] [ j ] + maximum_sum_path ( i , j + 1 ) ;
else total_sum = a [ i ] [ j ] + maximum_sum_path ( i + 1 , j ) ;
dp [ i ] [ j ] = total_sum ;
return total_sum ; }
public static void main ( String [ ] args ) { inputMatrix ( ) ;
int maximum_sum = maximum_sum_path ( 0 , 0 ) ; System . out . println ( maximum_sum ) ; } }
class GFG { static int MaxProfit ( int treasure [ ] , int color [ ] , int n , int k , int col , int A , int B ) { int sum = 0 ;
if ( k == n ) return 0 ;
if ( col == color [ k ] ) sum += Math . max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += Math . max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return sum ; }
public static void main ( String [ ] args ) { int A = - 5 , B = 7 ; int treasure [ ] = { 4 , 8 , 2 , 9 } ; int color [ ] = { 2 , 2 , 6 , 2 } ; int n = color . length ;
System . out . print ( MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ) ; } }
class GFG {
static int printTetraRec ( int n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
static void printTetra ( int n ) { System . out . println ( printTetraRec ( n ) + " ▁ " ) ; }
public static void main ( String [ ] args ) { int n = 10 ; printTetra ( n ) ; } }
import java . io . * ; class GFG {
static int sum = 0 ; static void Combination ( int [ ] a , int [ ] combi , int n , int r , int depth , int index ) {
if ( index == r ) {
int product = 1 ; for ( int i = 0 ; i < r ; i ++ ) product = product * combi [ i ] ;
sum += product ; return ; }
for ( int i = depth ; i < n ; i ++ ) { combi [ index ] = a [ i ] ; Combination ( a , combi , n , r , i + 1 , index + 1 ) ; } }
static void allCombination ( int [ ] a , int n ) { for ( int i = 1 ; i <= n ; i ++ ) {
int [ ] combi = new int [ i ] ;
Combination ( a , combi , n , i , 0 , 0 ) ;
System . out . print ( " f ( " + i + " ) ▁ - - > ▁ " + sum + "NEW_LINE"); sum = 0 ; } }
public static void main ( String args [ ] ) { int n = 5 ; int [ ] a = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) a [ i ] = i + 1 ;
allCombination ( a , n ) ; } }
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
import java . io . * ; class Partition {
static boolean findPartition ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; boolean part [ ] [ ] = new boolean [ sum / 2 + 1 ] [ n + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) part [ 0 ] [ i ] = true ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) part [ i ] [ 0 ] = false ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) { for ( j = 1 ; j <= n ; j ++ ) { part [ i ] [ j ] = part [ i ] [ j - 1 ] ; if ( i >= arr [ j - 1 ] ) part [ i ] [ j ] = part [ i ] [ j ] || part [ i - arr [ j - 1 ] ] [ j - 1 ] ; } }
return part [ sum / 2 ] [ n ] ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 1 , 1 , 2 , 2 , 1 } ; int n = arr . length ;
if ( findPartition ( arr , n ) == true ) System . out . println ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " " subsets ▁ of ▁ equal ▁ sum " ) ; else System . out . println ( " Can ▁ not ▁ be ▁ divided ▁ into " " ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static void minimumOperations ( String orig_str , int m , int n ) {
String orig = orig_str ;
int turn = 1 ; int j = 1 ;
for ( int i = 0 ; i < orig_str . length ( ) ; i ++ ) {
String m_cut = orig_str . substring ( orig_str . length ( ) - m ) ; orig_str = orig_str . substring ( 0 , orig_str . length ( ) - m ) ;
orig_str = m_cut + orig_str ;
j = j + 1 ;
if ( ! orig . equals ( orig_str ) ) { turn = turn + 1 ;
String n_cut = orig_str . substring ( orig_str . length ( ) - n ) ; orig_str = orig_str . substring ( 0 , orig_str . length ( ) - n ) ;
orig_str = n_cut + orig_str ;
j = j + 1 ; }
if ( orig . equals ( orig_str ) ) { break ; }
turn = turn + 1 ; } System . out . println ( turn ) ; }
public static void main ( String [ ] args ) {
String S = " GeeksforGeeks " ; int X = 5 , Y = 3 ;
minimumOperations ( S , X , Y ) ; } }
class GFG {
static int KMPSearch ( char [ ] pat , char [ ] txt ) { int M = pat . length ; int N = txt . length ;
int lps [ ] = new int [ M ] ;
computeLPSArray ( pat , M , lps ) ;
int i = 0 ; int j = 0 ; while ( i < N ) { if ( pat [ j ] == txt [ i ] ) { j ++ ; i ++ ; } if ( j == M ) { return i - j + 1 ; }
else if ( i < N && pat [ j ] != txt [ i ] ) {
if ( j != 0 ) j = lps [ j - 1 ] ; else i = i + 1 ; } } return 0 ; }
static void computeLPSArray ( char [ ] pat , int M , int [ ] lps ) {
int len = 0 ;
lps [ 0 ] = 0 ;
int i = 1 ; while ( i < M ) { if ( pat [ i ] == pat [ len ] ) { len ++ ; lps [ i ] = len ; i ++ ; }
else {
if ( len != 0 ) { len = lps [ len - 1 ] ; } else { lps [ i ] = 0 ; i ++ ; } } } }
static int countRotations ( String s ) {
String s1 = s . substring ( 1 , s . length ( ) - 1 ) + s ;
char [ ] pat = s . toCharArray ( ) ; char [ ] text = s1 . toCharArray ( ) ;
return 1 + KMPSearch ( pat , text ) ; }
public static void main ( String [ ] args ) { String s1 = " geeks " ; System . out . print ( countRotations ( s1 ) ) ; } }
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
import java . util . * ; class GFG { static int [ ] parent = new int [ 26 ] ;
static int find ( int x ) { if ( x != parent [ x ] ) return parent [ x ] = find ( parent [ x ] ) ; return x ; }
static void join ( int x , int y ) { int px = find ( x ) ; int pz = find ( y ) ; if ( px != pz ) { parent [ pz ] = px ; } }
static boolean convertible ( String s1 , String s2 ) {
HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ; for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) { if ( ! mp . containsKey ( s1 . charAt ( i ) - ' a ' ) ) { mp . put ( s1 . charAt ( i ) - ' a ' , s2 . charAt ( i ) - ' a ' ) ; } else { if ( mp . get ( s1 . charAt ( i ) - ' a ' ) != s2 . charAt ( i ) - ' a ' ) return false ; } }
for ( Map . Entry < Integer , Integer > it : mp . entrySet ( ) ) { if ( it . getKey ( ) == it . getValue ( ) ) continue ; else { if ( find ( it . getKey ( ) ) == find ( it . getValue ( ) ) ) return false ; else join ( it . getKey ( ) , it . getValue ( ) ) ; } } return true ; }
static void initialize ( ) { for ( int i = 0 ; i < 26 ; i ++ ) { parent [ i ] = i ; } }
public static void main ( String [ ] args ) { String s1 , s2 ; s1 = " abbcaa " ; s2 = " bccdbb " ; initialize ( ) ; if ( convertible ( s1 , s2 ) ) System . out . print ( " Yes " + "NEW_LINE"); else System . out . print ( " No " + "NEW_LINE"); } }
class GFG { static int SIZE = 26 ;
static void SieveOfEratosthenes ( boolean [ ] prime , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i < p_size ; i += p ) prime [ i ] = false ; } } }
static void printChar ( String str , int n ) { boolean [ ] prime = new boolean [ n + 1 ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) prime [ i ] = true ;
SieveOfEratosthenes ( prime , str . length ( ) + 1 ) ;
int [ ] freq = new int [ SIZE ] ;
for ( int i = 0 ; i < SIZE ; i ++ ) freq [ i ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( prime [ freq [ str . charAt ( i ) - ' a ' ] ] ) { System . out . print ( str . charAt ( i ) ) ; } } }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ; int n = str . length ( ) ; printChar ( str , n ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static boolean prime ( int n ) { if ( n <= 1 ) return false ; int max_div = ( int ) Math . floor ( Math . sqrt ( n ) ) ; for ( int i = 2 ; i < 1 + max_div ; i ++ ) { if ( n % i == 0 ) return false ; } return true ; } static void checkString ( String s ) {
Map < Character , Integer > freq = new HashMap < Character , Integer > ( ) ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( ! freq . containsKey ( s . charAt ( i ) ) ) freq . put ( s . charAt ( i ) , 0 ) ; freq . put ( s . charAt ( i ) , freq . get ( s . charAt ( i ) ) + 1 ) ; }
for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( prime ( freq . get ( s . charAt ( i ) ) ) ) System . out . print ( s . charAt ( i ) ) ; } }
public static void main ( String [ ] args ) { String s = " geeksforgeeks " ;
checkString ( s ) ; } }
import java . util . * ; class GFG { static int SIZE = 26 ;
static void printChar ( String str , int n ) {
int [ ] freq = new int [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str . charAt ( i ) - ' a ' ] % 2 == 0 ) { System . out . print ( str . charAt ( i ) ) ; } } }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ; int n = str . length ( ) ; printChar ( str , n ) ; } }
import java . util . * ; class GFG {
static boolean CompareAlphanumeric ( char [ ] str1 , char [ ] str2 ) {
int i , j ; i = 0 ; j = 0 ;
int len1 = str1 . length ;
int len2 = str2 . length ;
while ( i <= len1 && j <= len2 ) {
while ( i < len1 && ( ! ( ( str1 [ i ] >= ' a ' && str1 [ i ] <= ' z ' ) || ( str1 [ i ] >= ' A ' && str1 [ i ] <= ' Z ' ) || ( str1 [ i ] >= '0' && str1 [ i ] <= '9' ) ) ) ) { i ++ ; }
while ( j < len2 && ( ! ( ( str2 [ j ] >= ' a ' && str2 [ j ] <= ' z ' ) || ( str2 [ j ] >= ' A ' && str2 [ j ] <= ' Z ' ) || ( str2 [ j ] >= '0' && str2 [ j ] <= '9' ) ) ) ) { j ++ ; }
if ( i == len1 && j == len2 ) { return true ; }
else if ( str1 [ i ] != str2 [ j ] ) { return false ; }
else { i ++ ; j ++ ; } }
return false ; }
static void CompareAlphanumericUtil ( String str1 , String str2 ) { boolean res ;
res = CompareAlphanumeric ( str1 . toCharArray ( ) , str2 . toCharArray ( ) ) ;
if ( res == true ) { System . out . println ( " Equal " ) ; }
else { System . out . println ( " Unequal " ) ; } }
public static void main ( String [ ] args ) { String str1 , str2 ; str1 = " Ram , ▁ Shyam " ; str2 = " ▁ Ram ▁ - ▁ Shyam . " ; CompareAlphanumericUtil ( str1 , str2 ) ; str1 = " abc123" ; str2 = "123abc " ; CompareAlphanumericUtil ( str1 , str2 ) ; } }
class GFG {
static void solveQueries ( String str , int [ ] [ ] query ) {
int len = str . length ( ) ;
int Q = query . length ;
int [ ] [ ] pre = new int [ len ] [ 26 ] ;
for ( int i = 0 ; i < len ; i ++ ) {
pre [ i ] [ str . charAt ( i ) - ' a ' ] ++ ;
if ( i > 0 ) {
for ( int j = 0 ; j < 26 ; j ++ ) pre [ i ] [ j ] += pre [ i - 1 ] [ j ] ; } }
for ( int i = 0 ; i < Q ; i ++ ) {
int l = query [ i ] [ 0 ] ; int r = query [ i ] [ 1 ] ; int maxi = 0 ; char c = ' a ' ;
for ( int j = 0 ; j < 26 ; j ++ ) {
int times = pre [ r ] [ j ] ;
if ( l > 0 ) times -= pre [ l - 1 ] [ j ] ;
if ( times > maxi ) { maxi = times ; c = ( char ) ( ' a ' + j ) ; } }
System . out . println ( " Query " + ( i + 1 ) + " : ▁ " + c ) ; } }
public static void main ( String [ ] args ) { String str = " striver " ; int [ ] [ ] query = { { 0 , 1 } , { 1 , 6 } , { 5 , 6 } } ; solveQueries ( str , query ) ; } }
import java . util . * ; class GFG {
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
class GFG {
public static void printChar ( String str , int n ) {
int [ ] freq = new int [ 26 ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str . charAt ( i ) - ' a ' ] % 2 == 1 ) { System . out . print ( str . charAt ( i ) ) ; } } }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ; int n = str . length ( ) ; printChar ( str , n ) ; } }
class GFG {
static int minOperations ( String str , int n ) {
int i , lastUpper = - 1 , firstLower = - 1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( Character . isUpperCase ( str . charAt ( i ) ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( Character . isLowerCase ( str . charAt ( i ) ) ) { firstLower = i ; break ; } }
if ( lastUpper == - 1 firstLower == - 1 ) return 0 ;
int countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( Character . isUpperCase ( str . charAt ( i ) ) ) { countUpper ++ ; } }
int countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( Character . isLowerCase ( str . charAt ( i ) ) ) { countLower ++ ; } }
return Math . min ( countLower , countUpper ) ; }
public static void main ( String args [ ] ) { String str = " geEksFOrGEekS " ; int n = str . length ( ) ; System . out . println ( minOperations ( str , n ) ) ; } }
import java . util . * ; class GFG {
public static int Betrothed_Sum ( int n ) {
Vector < Integer > Set = new Vector < Integer > ( ) ; for ( int number_1 = 1 ; number_1 < n ; number_1 ++ ) {
int sum_divisor_1 = 1 ;
int i = 2 ; while ( i * i <= number_1 ) { if ( number_1 % i == 0 ) { sum_divisor_1 = sum_divisor_1 + i ; if ( i * i != number_1 ) sum_divisor_1 += number_1 / i ; } i ++ ; } if ( sum_divisor_1 > number_1 ) { int number_2 = sum_divisor_1 - 1 ; int sum_divisor_2 = 1 ; int j = 2 ; while ( j * j <= number_2 ) { if ( number_2 % j == 0 ) { sum_divisor_2 += j ; if ( j * j != number_2 ) sum_divisor_2 += number_2 / j ; } j = j + 1 ; } if ( sum_divisor_2 == number_1 + 1 && number_1 <= n && number_2 <= n ) { Set . add ( number_1 ) ; Set . add ( number_2 ) ; } } }
int Summ = 0 ; for ( int i = 0 ; i < Set . size ( ) ; i ++ ) { if ( Set . get ( i ) <= n ) Summ += Set . get ( i ) ; } return Summ ; }
public static void main ( String [ ] args ) { int n = 78 ; System . out . println ( Betrothed_Sum ( n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static float rainDayProbability ( int a [ ] , int n ) { float count = 0 , m ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
public static void main ( String args [ ] ) { int a [ ] = { 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 } ; int n = a . length ; System . out . print ( rainDayProbability ( a , n ) ) ; } }
import java . io . * ; class Maths {
static double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / Math . pow ( i , i ) ; sums += ser ; } return sums ; }
public static void main ( String [ ] args ) { int n = 3 ; double res = Series ( n ) ; res = Math . round ( res * 100000.0 ) / 100000.0 ; System . out . println ( res ) ; } }
import java . util . * ; public class Main {
static String lexicographicallyMaximum ( String S , int N ) {
HashMap < Character , Integer > M = new HashMap < > ( ) ;
for ( int i = 0 ; i < N ; ++ i ) { if ( M . containsKey ( S . charAt ( i ) ) ) M . put ( S . charAt ( i ) , M . get ( S . charAt ( i ) ) + 1 ) ; else M . put ( S . charAt ( i ) , 1 ) ; }
Vector < Character > V = new Vector < Character > ( ) ; for ( char i = ' a ' ; i < ( char ) ( ' a ' + Math . min ( N , 25 ) ) ; ++ i ) { if ( M . containsKey ( i ) == false ) { V . add ( i ) ; } }
int j = V . size ( ) - 1 ;
for ( int i = 0 ; i < N ; ++ i ) {
if ( S . charAt ( i ) >= ( ' a ' + Math . min ( N , 25 ) ) || ( M . containsKey ( S . charAt ( i ) ) && M . get ( S . charAt ( i ) ) > 1 ) ) { if ( V . get ( j ) < S . charAt ( i ) ) continue ;
M . put ( S . charAt ( i ) , M . get ( S . charAt ( i ) ) - 1 ) ;
S = S . substring ( 0 , i ) + V . get ( j ) + S . substring ( i + 1 ) ;
j -- ; } if ( j < 0 ) break ; } int l = 0 ;
for ( int i = N - 1 ; i >= 0 ; i -- ) { if ( l > j ) break ; if ( S . charAt ( i ) >= ( ' a ' + Math . min ( N , 25 ) ) || M . containsKey ( S . charAt ( i ) ) && M . get ( S . charAt ( i ) ) > 1 ) {
M . put ( S . charAt ( i ) , M . get ( S . charAt ( i ) ) - 1 ) ;
S = S . substring ( 0 , i ) + V . get ( l ) + S . substring ( i + 1 ) ;
l ++ ; } }
return S ; }
public static void main ( String [ ] args ) {
String S = " abccefghh " ; int N = S . length ( ) ;
System . out . println ( lexicographicallyMaximum ( S , N ) ) ; } }
import java . util . * ; class GFG {
static boolean isConsistingSubarrayUtil ( int arr [ ] , int n ) {
TreeMap < Integer , Integer > mp = new TreeMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < n ; ++ i ) {
mp . put ( arr [ i ] , mp . getOrDefault ( arr [ i ] , 0 ) + 1 ) ; }
for ( Map . Entry < Integer , Integer > it : mp . entrySet ( ) ) {
if ( it . getValue ( ) > 1 ) { return true ; } }
return false ; }
static void isConsistingSubarray ( int arr [ ] , int N ) { if ( isConsistingSubarrayUtil ( arr , N ) ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } }
public static void main ( String args [ ] ) {
int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 1 } ;
int N = arr . length ;
isConsistingSubarray ( arr , N ) ; } }
import java . util . * ; class GFG { static boolean [ ] isPrime ;
static HashSet < Integer > createhashmap ( int Max ) {
HashSet < Integer > hashmap = new HashSet < > ( ) ;
int curr = 1 ;
int prev = 0 ;
hashmap . add ( prev ) ;
while ( curr < Max ) {
hashmap . add ( curr ) ;
int temp = curr ;
curr = curr + prev ;
prev = temp ; } return hashmap ; }
static void SieveOfEratosthenes ( int Max ) {
isPrime = new boolean [ Max ] ; Arrays . fill ( isPrime , true ) ; isPrime [ 0 ] = false ; isPrime [ 1 ] = false ;
for ( int p = 2 ; p * p <= Max ; p ++ ) {
if ( isPrime [ p ] ) {
for ( int i = p * p ; i <= Max ; i += p ) {
isPrime [ i ] = false ; } } } }
static void cntFibonacciPrime ( int arr [ ] , int N ) {
int Max = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) {
Max = Math . max ( Max , arr [ i ] ) ; }
SieveOfEratosthenes ( Max ) ;
HashSet < Integer > hashmap = createhashmap ( Max ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 1 ) continue ;
if ( ( hashmap . contains ( arr [ i ] ) ) && ! isPrime [ arr [ i ] ] ) {
System . out . print ( arr [ i ] + " ▁ " ) ; } } }
public static void main ( String [ ] args ) { int arr [ ] = { 13 , 55 , 7 , 3 , 5 , 21 , 233 , 144 , 89 } ; int N = arr . length ; cntFibonacciPrime ( arr , N ) ; } }
import java . io . * ; import java . util . * ; import java . lang . * ; public class Main {
static int key ( int N ) {
String num = " " + N ; int ans = 0 ; int j = 0 ;
for ( j = 0 ; j < num . length ( ) ; j ++ ) {
if ( ( num . charAt ( j ) - 48 ) % 2 == 0 ) { int add = 0 ; int i ;
for ( i = j ; j < num . length ( ) ; j ++ ) { add += num . charAt ( j ) - 48 ;
if ( add % 2 == 1 ) break ; } if ( add == 0 ) { ans *= 10 ; } else { int digit = ( int ) Math . floor ( Math . log10 ( add ) + 1 ) ; ans *= ( Math . pow ( 10 , digit ) ) ;
ans += add ; }
i = j ; } else {
int add = 0 ; int i ;
for ( i = j ; j < num . length ( ) ; j ++ ) { add += num . charAt ( j ) - 48 ;
if ( add % 2 == 0 ) { break ; } } if ( add == 0 ) { ans *= 10 ; } else { int digit = ( int ) Math . floor ( Math . log10 ( add ) + 1 ) ; ans *= ( Math . pow ( 10 , digit ) ) ;
ans += add ; }
i = j ; } }
if ( j + 1 >= num . length ( ) ) { return ans ; } else { return ans += num . charAt ( num . length ( ) - 1 ) - 48 ; } }
public static void main ( String [ ] args ) { int N = 1667848271 ; System . out . print ( key ( N ) ) ; } }
class GFG {
static void sentinelSearch ( int arr [ ] , int n , int key ) {
int last = arr [ n - 1 ] ;
arr [ n - 1 ] = key ; int i = 0 ; while ( arr [ i ] != key ) i ++ ;
arr [ n - 1 ] = last ; if ( ( i < n - 1 ) || ( arr [ n - 1 ] == key ) ) System . out . println ( key + " ▁ is ▁ present ▁ at ▁ index ▁ " + i ) ; else System . out . println ( " Element ▁ Not ▁ found " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 10 , 20 , 180 , 30 , 60 , 50 , 110 , 100 , 70 } ; int n = arr . length ; int key = 180 ; sentinelSearch ( arr , n , key ) ; } }
import java . util . * ; class GFG {
static int maximum_middle_value ( int n , int k , int arr [ ] ) {
int ans = - 1 ;
int low = ( n + 1 - k ) / 2 ; int high = ( n + 1 - k ) / 2 + k ;
for ( int i = low ; i <= high ; i ++ ) {
ans = Math . max ( ans , arr [ i - 1 ] ) ; }
return ans ; }
public static void main ( String args [ ] ) { int n = 5 , k = 2 ; int arr [ ] = { 9 , 5 , 3 , 7 , 10 } ; System . out . println ( maximum_middle_value ( n , k , arr ) ) ; n = 9 ; k = 3 ; int arr1 [ ] = { 2 , 4 , 3 , 9 , 5 , 8 , 7 , 6 , 10 } ; System . out . println ( maximum_middle_value ( n , k , arr1 ) ) ; } }
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
import java . util . * ; class GFG {
static class Point { int x , y ; public Point ( int x , int y ) { this . x = x ; this . y = y ; } } ;
static int findmin ( Point p [ ] , int n ) { int a = 0 , b = 0 , c = 0 , d = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( p [ i ] . x <= 0 ) a ++ ;
else if ( p [ i ] . x >= 0 ) b ++ ;
if ( p [ i ] . y >= 0 ) c ++ ;
else if ( p [ i ] . y <= 0 ) d ++ ; } return Math . min ( Math . min ( a , b ) , Math . min ( c , d ) ) ; }
public static void main ( String [ ] args ) { Point p [ ] = { new Point ( 1 , 1 ) , new Point ( 2 , 2 ) , new Point ( - 1 , - 1 ) , new Point ( - 2 , 2 ) } ; int n = p . length ; System . out . println ( findmin ( p , n ) ) ; } }
import java . util . * ; class GFG {
static void maxOps ( int a , int b , int c ) {
int arr [ ] = { a , b , c } ;
int count = 0 ; while ( 1 != 0 ) {
Arrays . sort ( arr ) ;
if ( arr [ 0 ] == 0 && arr [ 1 ] == 0 ) break ;
arr [ 1 ] -= 1 ; arr [ 2 ] -= 1 ;
count += 1 ; }
System . out . print ( count ) ; }
public static void main ( String [ ] args ) {
int a = 4 , b = 3 , c = 2 ; maxOps ( a , b , c ) ; } }
import java . lang . Character ; class GFG { static int MAX = 26 ; public static String getSortedString ( StringBuilder s , int n ) {
int [ ] lower = new int [ MAX ] ; int [ ] upper = new int [ MAX ] ; for ( int i = 0 ; i < n ; i ++ ) {
if ( Character . isLowerCase ( s . charAt ( i ) ) ) lower [ s . charAt ( i ) - ' a ' ] ++ ;
else if ( Character . isUpperCase ( s . charAt ( i ) ) ) upper [ s . charAt ( i ) - ' A ' ] ++ ; }
int i = 0 , j = 0 ; while ( i < MAX && lower [ i ] == 0 ) i ++ ; while ( j < MAX && upper [ j ] == 0 ) j ++ ;
for ( int k = 0 ; k < n ; k ++ ) {
if ( Character . isLowerCase ( s . charAt ( k ) ) ) { while ( lower [ i ] == 0 ) i ++ ; s . setCharAt ( k , ( char ) ( i + ' a ' ) ) ;
lower [ i ] -- ; }
else if ( Character . isUpperCase ( s . charAt ( k ) ) ) { while ( upper [ j ] == 0 ) j ++ ; s . setCharAt ( k , ( char ) ( j + ' A ' ) ) ;
upper [ j ] -- ; } }
return s . toString ( ) ; }
public static void main ( String [ ] args ) { StringBuilder s = new StringBuilder ( " gEeksfOrgEEkS " ) ; int n = s . length ( ) ; System . out . println ( getSortedString ( s , n ) ) ; } }
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
public class ReverseWords { public static void main ( String [ ] args ) { String s [ ] = " i ▁ like ▁ this ▁ program ▁ very ▁ much " . split ( " ▁ " ) ; String ans = " " ; for ( int i = s . length - 1 ; i >= 0 ; i -- ) { ans += s [ i ] + " ▁ " ; } System . out . println ( " Reversed ▁ String : " ) ; System . out . println ( ans . substring ( 0 , ans . length ( ) - 1 ) ) ; } }
import java . util . * ; class GFG {
public static void SieveOfEratosthenes ( boolean [ ] prime , int n ) { for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } }
public static void segregatePrimeNonPrime ( boolean [ ] prime , int arr [ ] , int N ) {
SieveOfEratosthenes ( prime , 10000000 ) ;
int left = 0 , right = N - 1 ;
while ( left < right ) {
while ( prime [ arr [ left ] ] ) left ++ ;
while ( ! prime [ arr [ right ] ] ) right -- ;
if ( left < right ) {
int temp = arr [ left ] ; arr [ left ] = arr [ right ] ; arr [ right ] = temp ; left ++ ; right -- ; } }
for ( int i = 0 ; i < N ; i ++ ) System . out . printf ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { boolean [ ] prime = new boolean [ 10000001 ] ; Arrays . fill ( prime , true ) ; int arr [ ] = { 2 , 3 , 4 , 6 , 7 , 8 , 9 , 10 } ; int N = arr . length ;
segregatePrimeNonPrime ( prime , arr , N ) ; } }
import java . io . * ; class GFG {
static int findDepthRec ( String tree , int n , int index ) { if ( index >= n || tree . charAt ( index ) == ' l ' ) return 0 ;
index ++ ; int left = findDepthRec ( tree , n , index ) ;
index ++ ; int right = findDepthRec ( tree , n , index ) ; return Math . max ( left , right ) + 1 ; }
static int findDepth ( String tree , int n ) { int index = 0 ; return ( findDepthRec ( tree , n , index ) ) ; }
static public void main ( String [ ] args ) { String tree = " nlnnlll " ; int n = tree . length ( ) ; System . out . println ( findDepth ( tree , n ) ) ; } }
class GfG {
static class Node { int key ; Node left , right ; }
static Node newNode ( int item ) { Node temp = new Node ( ) ; temp . key = item ; temp . left = null ; temp . right = null ; return temp ; }
static Node insert ( Node node , int key ) {
if ( node == null ) return newNode ( key ) ;
if ( key < node . key ) node . left = insert ( node . left , key ) ; else if ( key > node . key ) node . right = insert ( node . right , key ) ;
return node ; }
static int findMaxforN ( Node root , int N ) {
if ( root == null ) return - 1 ; if ( root . key == N ) return N ;
else if ( root . key < N ) { int k = findMaxforN ( root . right , N ) ; if ( k == - 1 ) return root . key ; else return k ; }
else if ( root . key > N ) return findMaxforN ( root . left , N ) ; return - 1 ; }
public static void main ( String [ ] args ) { int N = 4 ;
Node root = null ; root = insert ( root , 25 ) ; insert ( root , 2 ) ; insert ( root , 1 ) ; insert ( root , 3 ) ; insert ( root , 12 ) ; insert ( root , 9 ) ; insert ( root , 21 ) ; insert ( root , 19 ) ; insert ( root , 25 ) ; System . out . println ( findMaxforN ( root , N ) ) ; } }
class Solution { static class Node { Node left , right ; int data ; }
static Node createNode ( int x ) { Node p = new Node ( ) ; p . data = x ; p . left = p . right = null ; return p ; }
static void insertNode ( Node root , int x ) { Node p = root , q = null ; while ( p != null ) { q = p ; if ( p . data < x ) p = p . right ; else p = p . left ; } if ( q == null ) p = createNode ( x ) ; else { if ( q . data < x ) q . right = createNode ( x ) ; else q . left = createNode ( x ) ; } }
static int maxelpath ( Node q , int x ) { Node p = q ; int mx = - 1 ;
while ( p . data != x ) { if ( p . data > x ) { mx = Math . max ( mx , p . data ) ; p = p . left ; } else { mx = Math . max ( mx , p . data ) ; p = p . right ; } } return Math . max ( mx , x ) ; }
static int maximumElement ( Node root , int x , int y ) { Node p = root ;
while ( ( x < p . data && y < p . data ) || ( x > p . data && y > p . data ) ) {
if ( x < p . data && y < p . data ) p = p . left ;
else if ( x > p . data && y > p . data ) p = p . right ; }
return Math . max ( maxelpath ( p , x ) , maxelpath ( p , y ) ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 18 , 36 , 9 , 6 , 12 , 10 , 1 , 8 } ; int a = 1 , b = 10 ; int n = arr . length ;
Node root = createNode ( arr [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) insertNode ( root , arr [ i ] ) ; System . out . println ( maximumElement ( root , a , b ) ) ; } }
import java . util . * ; class solution { static class Node { Node left , right ; int info ;
boolean lthread ;
boolean rthread ; } ;
static Node insert ( Node root , int ikey ) {
Node ptr = root ;
Node par = null ; while ( ptr != null ) {
if ( ikey == ( ptr . info ) ) { System . out . printf ( "Duplicate Key !NEW_LINE"); return root ; }
par = ptr ;
if ( ikey < ptr . info ) { if ( ptr . lthread == false ) ptr = ptr . left ; else break ; }
else { if ( ptr . rthread == false ) ptr = ptr . right ; else break ; } }
Node tmp = new Node ( ) ; tmp . info = ikey ; tmp . lthread = true ; tmp . rthread = true ; if ( par == null ) { root = tmp ; tmp . left = null ; tmp . right = null ; } else if ( ikey < ( par . info ) ) { tmp . left = par . left ; tmp . right = par ; par . lthread = false ; par . left = tmp ; } else { tmp . left = par ; tmp . right = par . right ; par . rthread = false ; par . right = tmp ; } return root ; }
static Node inorderSuccessor ( Node ptr ) {
if ( ptr . rthread == true ) return ptr . right ;
ptr = ptr . right ; while ( ptr . lthread == false ) ptr = ptr . left ; return ptr ; }
static void inorder ( Node root ) { if ( root == null ) System . out . printf ( " Tree ▁ is ▁ empty " ) ;
Node ptr = root ; while ( ptr . lthread == false ) ptr = ptr . left ;
while ( ptr != null ) { System . out . printf ( " % d ▁ " , ptr . info ) ; ptr = inorderSuccessor ( ptr ) ; } }
public static void main ( String [ ] args ) { Node root = null ; root = insert ( root , 20 ) ; root = insert ( root , 10 ) ; root = insert ( root , 30 ) ; root = insert ( root , 5 ) ; root = insert ( root , 16 ) ; root = insert ( root , 14 ) ; root = insert ( root , 17 ) ; root = insert ( root , 13 ) ; inorder ( root ) ; } }
static class Node { Node left , right ; int info ;
boolean lthread ;
boolean rthread ; } ;
import java . util . * ; class solution { static class Node { Node left , right ; int info ;
boolean lthread ;
boolean rthread ; } ;
static Node insert ( Node root , int ikey ) {
Node ptr = root ;
Node par = null ; while ( ptr != null ) {
if ( ikey == ( ptr . info ) ) { System . out . printf ( "Duplicate Key !NEW_LINE"); return root ; }
par = ptr ;
if ( ikey < ptr . info ) { if ( ptr . lthread == false ) ptr = ptr . left ; else break ; }
else { if ( ptr . rthread == false ) ptr = ptr . right ; else break ; } }
Node tmp = new Node ( ) ; tmp . info = ikey ; tmp . lthread = true ; tmp . rthread = true ; if ( par == null ) { root = tmp ; tmp . left = null ; tmp . right = null ; } else if ( ikey < ( par . info ) ) { tmp . left = par . left ; tmp . right = par ; par . lthread = false ; par . left = tmp ; } else { tmp . left = par ; tmp . right = par . right ; par . rthread = false ; par . right = tmp ; } return root ; }
static Node inSucc ( Node ptr ) { if ( ptr . rthread == true ) return ptr . right ; ptr = ptr . right ; while ( ptr . lthread == false ) ptr = ptr . left ; return ptr ; }
static Node inorderSuccessor ( Node ptr ) {
if ( ptr . rthread == true ) return ptr . right ;
ptr = ptr . right ; while ( ptr . lthread == false ) ptr = ptr . left ; return ptr ; }
static void inorder ( Node root ) { if ( root == null ) System . out . printf ( " Tree ▁ is ▁ empty " ) ;
Node ptr = root ; while ( ptr . lthread == false ) ptr = ptr . left ;
while ( ptr != null ) { System . out . printf ( " % d ▁ " , ptr . info ) ; ptr = inorderSuccessor ( ptr ) ; } } static Node inPred ( Node ptr ) { if ( ptr . lthread == true ) return ptr . left ; ptr = ptr . left ; while ( ptr . rthread == false ) ptr = ptr . right ; return ptr ; }
static Node caseA ( Node root , Node par , Node ptr ) {
if ( par == null ) root = null ;
else if ( ptr == par . left ) { par . lthread = true ; par . left = ptr . left ; } else { par . rthread = true ; par . right = ptr . right ; } return root ; }
static Node caseB ( Node root , Node par , Node ptr ) { Node child ;
if ( ptr . lthread == false ) child = ptr . left ;
else child = ptr . right ;
if ( par == null ) root = child ;
else if ( ptr == par . left ) par . left = child ; else par . right = child ;
Node s = inSucc ( ptr ) ; Node p = inPred ( ptr ) ;
if ( ptr . lthread == false ) p . right = s ;
else { if ( ptr . rthread == false ) s . left = p ; } return root ; }
static Node caseC ( Node root , Node par , Node ptr ) {
Node parsucc = ptr ; Node succ = ptr . right ;
while ( succ . lthread == false ) { parsucc = succ ; succ = succ . left ; } ptr . info = succ . info ; if ( succ . lthread == true && succ . rthread == true ) root = caseA ( root , parsucc , succ ) ; else root = caseB ( root , parsucc , succ ) ; return root ; }
static Node delThreadedBST ( Node root , int dkey ) {
Node par = null , ptr = root ;
int found = 0 ;
while ( ptr != null ) { if ( dkey == ptr . info ) { found = 1 ; break ; } par = ptr ; if ( dkey < ptr . info ) { if ( ptr . lthread == false ) ptr = ptr . left ; else break ; } else { if ( ptr . rthread == false ) ptr = ptr . right ; else break ; } } if ( found == 0 ) System . out . printf ( "dkey not present in treeNEW_LINE");
else if ( ptr . lthread == false && ptr . rthread == false ) root = caseC ( root , par , ptr ) ;
else if ( ptr . lthread == false ) root = caseB ( root , par , ptr ) ;
else if ( ptr . rthread == false ) root = caseB ( root , par , ptr ) ;
else root = caseA ( root , par , ptr ) ; return root ; }
public static void main ( String args [ ] ) { Node root = null ; root = insert ( root , 20 ) ; root = insert ( root , 10 ) ; root = insert ( root , 30 ) ; root = insert ( root , 5 ) ; root = insert ( root , 16 ) ; root = insert ( root , 14 ) ; root = insert ( root , 17 ) ; root = insert ( root , 13 ) ; root = delThreadedBST ( root , 20 ) ; inorder ( root ) ; } }
import java . io . * ; public class GFG { static void checkHV ( int [ ] [ ] arr , int N , int M ) {
boolean horizontal = true ; boolean vertical = true ;
for ( int i = 0 , k = N - 1 ; i < N / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < M ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } }
for ( int i = 0 , k = M - 1 ; i < M / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } } if ( ! horizontal && ! vertical ) System . out . println ( " NO " ) ; else if ( horizontal && ! vertical ) System . out . println ( " HORIZONTAL " ) ; else if ( vertical && ! horizontal ) System . out . println ( " VERTICAL " ) ; else System . out . println ( " BOTH " ) ; }
static public void main ( String [ ] args ) { int [ ] [ ] mat = { { 1 , 0 , 1 } , { 0 , 0 , 0 } , { 1 , 0 , 1 } } ; checkHV ( mat , 3 , 3 ) ; } }
import java . io . * ; class GFG { static int R = 3 ; static int C = 4 ;
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
static void replacematrix ( int [ ] [ ] mat , int n , int m ) { int [ ] rgcd = new int [ R ] ; int [ ] cgcd = new int [ C ] ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { rgcd [ i ] = gcd ( rgcd [ i ] , mat [ i ] [ j ] ) ; cgcd [ j ] = gcd ( cgcd [ j ] , mat [ i ] [ j ] ) ; } }
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < m ; j ++ ) mat [ i ] [ j ] = Math . max ( rgcd [ i ] , cgcd [ j ] ) ; }
static public void main ( String [ ] args ) { int [ ] [ ] m = { { 1 , 2 , 3 , 3 } , { 4 , 5 , 6 , 6 } , { 7 , 8 , 9 , 9 } , } ; replacematrix ( m , R , C ) ; for ( int i = 0 ; i < R ; i ++ ) { for ( int j = 0 ; j < C ; j ++ ) System . out . print ( m [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } } }
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
import java . io . * ; class GFG { static int maxTripletSum ( int arr [ ] , int n ) {
int sum = - 1000000 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) for ( int k = j + 1 ; k < n ; k ++ ) if ( sum < arr [ i ] + arr [ j ] + arr [ k ] ) sum = arr [ i ] + arr [ j ] + arr [ k ] ; return sum ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . length ; System . out . println ( maxTripletSum ( arr , n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int maxTripletSum ( int arr [ ] , int n ) {
Arrays . sort ( arr ) ;
return arr [ n - 1 ] + arr [ n - 2 ] + arr [ n - 3 ] ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . length ; System . out . println ( maxTripletSum ( arr , n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int maxTripletSum ( int arr [ ] , int n ) {
int maxA = - 100000000 , maxB = - 100000000 ; int maxC = - 100000000 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > maxA ) { maxC = maxB ; maxB = maxA ; maxA = arr [ i ] ; }
else if ( arr [ i ] > maxB ) { maxC = maxB ; maxB = arr [ i ] ; }
else if ( arr [ i ] > maxC ) maxC = arr [ i ] ; } return ( maxA + maxB + maxC ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = arr . length ; System . out . println ( maxTripletSum ( arr , n ) ) ; } }
class GFG { public static int search ( int arr [ ] , int x ) { int n = arr . length ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == x ) return i ; } return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ;
int result = search ( arr , x ) ; if ( result == - 1 ) System . out . print ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) ; else System . out . print ( " Element ▁ is ▁ present ▁ at ▁ index ▁ " + result ) ; } }
import java . io . * ; class GFG { public static void search ( int arr [ ] , int search_Element ) { int left = 0 ; int length = arr . length ; int right = length - 1 ; int position = - 1 ;
for ( left = 0 ; left <= right ; ) {
if ( arr [ left ] == search_Element ) { position = left ; System . out . println ( " Element ▁ found ▁ in ▁ Array ▁ at ▁ " + ( position + 1 ) + " ▁ Position ▁ with ▁ " + ( left + 1 ) + " ▁ Attempt " ) ; break ; }
if ( arr [ right ] == search_Element ) { position = right ; System . out . println ( " Element ▁ found ▁ in ▁ Array ▁ at ▁ " + ( position + 1 ) + " ▁ Position ▁ with ▁ " + ( length - right ) + " ▁ Attempt " ) ; break ; } left ++ ; right -- ; }
if ( position == - 1 ) System . out . println ( " Not ▁ found ▁ in ▁ Array ▁ with ▁ " + left + " ▁ Attempt " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int search_element = 5 ;
search ( arr , search_element ) ; } }
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
static void countSort ( int [ ] arr ) { int max = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ; int min = Arrays . stream ( arr ) . min ( ) . getAsInt ( ) ; int range = max - min + 1 ; int count [ ] = new int [ range ] ; int output [ ] = new int [ arr . length ] ; for ( int i = 0 ; i < arr . length ; i ++ ) { count [ arr [ i ] - min ] ++ ; } for ( int i = 1 ; i < count . length ; i ++ ) { count [ i ] += count [ i - 1 ] ; } for ( int i = arr . length - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] - min ] - 1 ] = arr [ i ] ; count [ arr [ i ] - min ] -- ; } for ( int i = 0 ; i < arr . length ; i ++ ) { arr [ i ] = output [ i ] ; } }
static void printArray ( int [ ] arr ) { for ( int i = 0 ; i < arr . length ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { - 5 , - 10 , 0 , - 3 , 8 , 5 , - 1 , 10 } ; countSort ( arr ) ; printArray ( arr ) ; } }
import java . util . * ; class GFG {
static int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
public static void main ( String [ ] args ) { int n = 5 , k = 2 ; System . out . printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; } }
import java . util . * ; class GFG { static int binomialCoeff ( int n , int k ) { int C [ ] = new int [ k + 1 ] ;
C [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = Math . min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
public static void main ( String [ ] args ) { int n = 5 , k = 2 ; System . out . printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; } }
import java . util . * ; class GFG {
static int binomialCoeff ( int n , int r ) { if ( r > n ) return 0 ; long m = 1000000007 ; long inv [ ] = new long [ r + 1 ] ; inv [ 0 ] = 1 ; if ( r + 1 >= 2 ) inv [ 1 ] = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { inv [ i ] = m - ( m / i ) * inv [ ( int ) ( m % i ) ] % m ; } int ans = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { ans = ( int ) ( ( ( ans % m ) * ( inv [ i ] % m ) ) % m ) ; }
for ( int i = n ; i >= ( n - r + 1 ) ; i -- ) { ans = ( int ) ( ( ( ans % m ) * ( i % m ) ) % m ) ; } return ans ; }
public static void main ( String [ ] args ) { int n = 5 , r = 2 ; System . out . print ( " Value ▁ of ▁ C ( " + n + " , ▁ " + r + " ) ▁ is ▁ " + binomialCoeff ( n , r ) + "NEW_LINE"); } }
import java . io . * ; class GFG {
public static boolean findPartiion ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; boolean [ ] part = new boolean [ sum / 2 + 1 ] ;
for ( i = 0 ; i <= sum / 2 ; i ++ ) { part [ i ] = false ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = sum / 2 ; j >= arr [ i ] ; j -- ) {
if ( part [ j - arr [ i ] ] == true j == arr [ i ] ) part [ j ] = true ; } } return part [ sum / 2 ] ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 3 , 2 , 3 , 2 } ; int n = 6 ;
if ( findPartiion ( arr , n ) == true ) System . out . println ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " + " subsets ▁ of ▁ equal ▁ sum " ) ; else System . out . println ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " + " two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
class GFG {
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
class GFG {
static boolean isSubsetSum ( int set [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ sum + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ i ] [ 0 ] = false ;
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; } }
public static void main ( String args [ ] ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = set . length ; if ( isSubsetSum ( set , n , sum ) == true ) System . out . println ( " Found ▁ a ▁ subset " + " ▁ with ▁ given ▁ sum " ) ; else System . out . println ( " No ▁ subset ▁ with " + " ▁ given ▁ sum " ) ; } }
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
static int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int [ ] screen = new int [ N ] ;
int n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = Math . max ( 2 * screen [ n - 4 ] , Math . max ( 3 * screen [ n - 5 ] , 4 * screen [ n - 6 ] ) ) ; } return screen [ N - 1 ] ; }
public static void main ( String [ ] args ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) System . out . printf ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with " + " %d keystrokes is %dNEW_LINE", N , findoptimal ( N ) ) ; } }
class GFG {
static int power ( int x , int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; }
public static void main ( String [ ] args ) { int x = 2 ; int y = 3 ; System . out . printf ( " % d " , power ( x , y ) ) ; } }
static int power ( int x , int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
class GFG { static float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
public static void main ( String [ ] args ) { float x = 2 ; int y = - 3 ; System . out . printf ( " % f " , power ( x , y ) ) ; } }
import java . io . * ; class GFG { public static int power ( int x , int y ) {
if ( y == 0 ) return 1 ;
if ( x == 0 ) return 0 ;
return x * power ( x , y - 1 ) ; }
public static void main ( String [ ] args ) { int x = 2 ; int y = 3 ; System . out . println ( power ( x , y ) ) ; } }
import java . io . * ; class GFG { public static int power ( int x , int y ) {
return ( int ) Math . pow ( x , y ) ; }
public static void main ( String [ ] args ) { int x = 2 ; int y = 3 ; System . out . println ( power ( x , y ) ) ; } }
class GFG {
static float squareRoot ( float n ) {
float x = n ; float y = 1 ;
double e = 0.000001 ; while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; }
public static void main ( String [ ] args ) { int n = 50 ; System . out . printf ( " Square ▁ root ▁ of ▁ " + n + " ▁ is ▁ " + squareRoot ( n ) ) ; } }
class GFG {
static float getAvg ( float prev_avg , float x , int n ) { return ( prev_avg * n + x ) / ( n + 1 ) ; }
static void streamAvg ( float arr [ ] , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( avg , arr [ i ] , i ) ; System . out . printf ( "Average of %d numbers is %f NEW_LINE", i + 1, avg); } return ; }
public static void main ( String [ ] args ) { float arr [ ] = { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = arr . length ; streamAvg ( arr , n ) ; } }
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
class GFG {
static void segregate0and1 ( int arr [ ] , int n ) {
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 0 ) count ++ ; }
for ( int i = 0 ; i < count ; i ++ ) arr [ i ] = 0 ;
for ( int i = count ; i < n ; i ++ ) arr [ i ] = 1 ; }
static void print ( int arr [ ] , int n ) { System . out . print ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int n = arr . length ; segregate0and1 ( arr , n ) ; print ( arr , n ) ; } }
class Segregate {
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
public static void main ( String [ ] args ) { Segregate seg = new Segregate ( ) ; int arr [ ] = new int [ ] { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = arr . length ; seg . segregate0and1 ( arr , arr_size ) ; System . out . print ( " Array ▁ after ▁ segregation ▁ is ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
import java . util . * ; class GFG {
static void segregate0and1 ( int arr [ ] ) { int type0 = 0 ; int type1 = arr . length - 1 ; while ( type0 < type1 ) { if ( arr [ type0 ] == 1 ) { arr [ type1 ] = arr [ type1 ] + arr [ type0 ] ; arr [ type0 ] = arr [ type1 ] - arr [ type0 ] ; arr [ type1 ] = arr [ type1 ] - arr [ type0 ] ; type1 -- ; } else { type0 ++ ; } } }
public static void main ( String [ ] args ) { int [ ] array = { 0 , 1 , 0 , 1 , 1 , 1 } ; segregate0and1 ( array ) ; for ( int a : array ) { System . out . print ( a + " ▁ " ) ; } } }
import java . io . * ; import java . util . HashMap ; import java . util . Map ; class GFG { static void distinctAdjacentElement ( int a [ ] , int n ) {
HashMap < Integer , Integer > m = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < n ; ++ i ) { if ( m . containsKey ( a [ i ] ) ) { int x = m . get ( a [ i ] ) + 1 ; m . put ( a [ i ] , x ) ; } else { m . put ( a [ i ] , 1 ) ; } }
int mx = 0 ;
for ( int i = 0 ; i < n ; ++ i ) if ( mx < m . get ( a [ i ] ) ) mx = m . get ( a [ i ] ) ;
if ( mx > ( n + 1 ) / 2 ) System . out . println ( " NO " ) ; else System . out . println ( " YES " ) ; }
public static void main ( String [ ] args ) { int a [ ] = { 7 , 7 , 7 , 7 } ; int n = 4 ; distinctAdjacentElement ( a , n ) ; } }
class FindMaximum {
int maxIndexDiff ( int arr [ ] , int n ) { int maxDiff = - 1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
public static void main ( String [ ] args ) { FindMaximum max = new FindMaximum ( ) ; int arr [ ] = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = arr . length ; int maxDiff = max . maxIndexDiff ( arr , n ) ; System . out . println ( maxDiff ) ; } }
import java . util . * ; class GFG { public static void main ( String [ ] args ) { int [ ] v = { 34 , 8 , 10 , 3 , 2 , 80 , 30 , 33 , 1 } ; int n = v . length ; int [ ] maxFromEnd = new int [ n + 1 ] ; Arrays . fill ( maxFromEnd , Integer . MIN_VALUE ) ;
for ( int i = v . length - 1 ; i >= 0 ; i -- ) { maxFromEnd [ i ] = Math . max ( maxFromEnd [ i + 1 ] , v [ i ] ) ; } int result = 0 ; for ( int i = 0 ; i < v . length ; i ++ ) { int low = i + 1 , high = v . length - 1 , ans = i ; while ( low <= high ) { int mid = ( low + high ) / 2 ; if ( v [ i ] <= maxFromEnd [ mid ] ) {
ans = Math . max ( ans , mid ) ; low = mid + 1 ; } else { high = mid - 1 ; } }
result = Math . max ( result , ans - i ) ; } System . out . print ( result + "NEW_LINE"); } }
import java . io . * ; import java . util . * ; class GFG {
static int maxIndexDiff ( ArrayList < Integer > arr , int n ) {
Map < Integer , ArrayList < Integer > > hashmap = new HashMap < Integer , ArrayList < Integer > > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( hashmap . containsKey ( arr . get ( i ) ) ) { hashmap . get ( arr . get ( i ) ) . add ( i ) ; } else { hashmap . put ( arr . get ( i ) , new ArrayList < Integer > ( ) ) ; hashmap . get ( arr . get ( i ) ) . add ( i ) ; } }
Collections . sort ( arr ) ; int maxDiff = Integer . MIN_VALUE ; int temp = n ;
for ( int i = 0 ; i < n ; i ++ ) { if ( temp > hashmap . get ( arr . get ( i ) ) . get ( 0 ) ) { temp = hashmap . get ( arr . get ( i ) ) . get ( 0 ) ; } maxDiff = Math . max ( maxDiff , hashmap . get ( arr . get ( i ) ) . get ( hashmap . get ( arr . get ( i ) ) . size ( ) - 1 ) - temp ) ; } return maxDiff ; }
public static void main ( String [ ] args ) { int n = 9 ; ArrayList < Integer > arr = new ArrayList < Integer > ( Arrays . asList ( 34 , 8 , 10 , 3 , 2 , 80 , 30 , 33 , 1 ) ) ;
int ans = maxIndexDiff ( arr , n ) ; System . out . println ( " The ▁ maxIndexDiff ▁ is ▁ : ▁ " + ans ) ; } }
import java . io . * ; import java . util . * ; public class GFG { static void printRepeating ( Integer [ ] arr , int size ) {
SortedSet < Integer > s = new TreeSet < > ( ) ; Collections . addAll ( s , arr ) ;
System . out . print ( s ) ; }
public static void main ( String args [ ] ) { Integer [ ] arr = { 1 , 3 , 2 , 2 , 1 } ; int n = arr . length ; printRepeating ( arr , n ) ; } }
import java . io . * ; import java . util . * ;
class GFG { static int minSwapsToSort ( int arr [ ] , int n ) {
ArrayList < ArrayList < Integer > > arrPos = new ArrayList < ArrayList < Integer > > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { arrPos . add ( new ArrayList < Integer > ( Arrays . asList ( arr [ i ] , i ) ) ) ; }
Collections . sort ( arrPos , new Comparator < ArrayList < Integer > > ( ) { @ Override public int compare ( ArrayList < Integer > o1 , ArrayList < Integer > o2 ) { return o1 . get ( 0 ) . compareTo ( o2 . get ( 0 ) ) ; } } ) ;
boolean [ ] vis = new boolean [ n ] ;
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( vis [ i ] || arrPos . get ( i ) . get ( 1 ) == i ) continue ;
int cycle_size = 0 ; int j = i ; while ( ! vis [ j ] ) { vis [ j ] = true ;
j = arrPos . get ( j ) . get ( 1 ) ; cycle_size ++ ; }
ans += ( cycle_size - 1 ) ; }
return ans ; }
static int minSwapToMakeArraySame ( int a [ ] , int b [ ] , int n ) {
Map < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { mp . put ( b [ i ] , i ) ; }
for ( int i = 0 ; i < n ; i ++ ) b [ i ] = mp . get ( a [ i ] ) ;
return minSwapsToSort ( b , n ) ; }
public static void main ( String [ ] args ) { int a [ ] = { 3 , 6 , 4 , 8 } ; int b [ ] = { 4 , 6 , 8 , 3 } ; int n = a . length ; System . out . println ( minSwapToMakeArraySame ( a , b , n ) ) ; } }
import java . io . * ; import java . util . * ; public class GFG {
static int missingK ( int [ ] a , int k , int n ) { int difference = 0 , ans = 0 , count = k ; boolean flag = false ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = true ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return - 1 ; }
public static void main ( String args [ ] ) {
int [ ] a = { 1 , 5 , 11 , 19 } ;
int k = 11 ; int n = a . length ;
int missing = missingK ( a , k , n ) ; System . out . print ( missing ) ; } }
public class GFG {
static int missingK ( int [ ] arr , int k ) { int n = arr . length ; int l = 0 , u = n - 1 , mid ; while ( l <= u ) { mid = ( l + u ) / 2 ; int numbers_less_than_mid = arr [ mid ] - ( mid + 1 ) ;
if ( numbers_less_than_mid == k ) {
if ( mid > 0 && ( arr [ mid - 1 ] - ( mid ) ) == k ) { u = mid - 1 ; continue ; }
return arr [ mid ] - 1 ; }
if ( numbers_less_than_mid < k ) { l = mid + 1 ; } else if ( k < numbers_less_than_mid ) { u = mid - 1 ; } }
if ( u < 0 ) return k ;
int less = arr [ u ] - ( u + 1 ) ; k -= less ;
return arr [ u ] + k ; }
public static void main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 4 , 7 , 11 } ; int k = 5 ;
System . out . println ( " Missing ▁ kth ▁ number ▁ = ▁ " + missingK ( arr , k ) ) ; } }
import java . util . * ; class GFG {
static class Node { int data ; Node next ; }
static void printList ( Node node ) { while ( node != null ) { System . out . print ( node . data + " ▁ " ) ; node = node . next ; } System . out . println ( ) ; }
static Node newNode ( int key ) { Node temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
static Node insertBeg ( Node head , int val ) { Node temp = newNode ( val ) ; temp . next = head ; head = temp ; return head ; }
static void rearrangeOddEven ( Node head ) { Stack < Node > odd = new Stack < Node > ( ) ; Stack < Node > even = new Stack < Node > ( ) ; int i = 1 ; while ( head != null ) { if ( head . data % 2 != 0 && i % 2 == 0 ) {
odd . push ( head ) ; } else if ( head . data % 2 == 0 && i % 2 != 0 ) {
even . push ( head ) ; } head = head . next ; i ++ ; } while ( odd . size ( ) > 0 && even . size ( ) > 0 ) {
int k = odd . peek ( ) . data ; odd . peek ( ) . data = even . peek ( ) . data ; even . peek ( ) . data = k ; odd . pop ( ) ; even . pop ( ) ; } }
public static void main ( String args [ ] ) { Node head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 1 ) ; System . out . println ( " Linked ▁ List : " ) ; printList ( head ) ; rearrangeOddEven ( head ) ; System . out . println ( " Linked ▁ List ▁ after ▁ " + " Rearranging : " ) ; printList ( head ) ; } }
class GFG {
static class Node { int data ; Node next ; } ;
static void printList ( Node node ) { while ( node != null ) { System . out . print ( node . data + " ▁ " ) ; node = node . next ; } System . out . println ( ) ; }
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
public static void main ( String args [ ] ) { Node head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 1 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 10 ) ; System . out . println ( " Linked ▁ List : " ) ; printList ( head ) ; System . out . println ( " Rearranged ▁ List " ) ; head = rearrange ( head ) ; printList ( head ) ; } }
import java . io . * ; import java . lang . * ; import java . util . * ; public class GFG {
static void print ( int mat [ ] [ ] ) {
for ( int i = 0 ; i < mat . length ; i ++ ) {
for ( int j = 0 ; j < mat [ 0 ] . length ; j ++ )
System . out . print ( mat [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
static void performSwap ( int mat [ ] [ ] , int i , int j ) { int N = mat . length ;
int ei = N - 1 - i ;
int ej = N - 1 - j ;
int temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ ej ] [ i ] ; mat [ ej ] [ i ] = mat [ ei ] [ ej ] ; mat [ ei ] [ ej ] = mat [ j ] [ ei ] ; mat [ j ] [ ei ] = temp ; }
static void rotate ( int mat [ ] [ ] , int N , int K ) {
K = K % 4 ;
while ( K -- > 0 ) {
for ( int i = 0 ; i < N / 2 ; i ++ ) {
for ( int j = i ; j < N - i - 1 ; j ++ ) {
if ( i != j && ( i + j ) != N - 1 ) {
performSwap ( mat , i , j ) ; } } } }
print ( mat ) ; }
public static void main ( String [ ] args ) { int K = 5 ; int mat [ ] [ ] = { { 1 , 2 , 3 , 4 } , { 6 , 7 , 8 , 9 } , { 11 , 12 , 13 , 14 } , { 16 , 17 , 18 , 19 } , } ; int N = mat . length ; rotate ( mat , N , K ) ; } }
import java . util . * ; class GFG {
static int findRotations ( String str ) {
String tmp = str + str ; int n = str . length ( ) ; for ( int i = 1 ; i <= n ; i ++ ) {
String substring = tmp . substring ( i , i + str . length ( ) ) ;
if ( str . equals ( substring ) ) return i ; } return n ; }
public static void main ( String [ ] args ) { String str = " aaaa " ; System . out . println ( findRotations ( str ) ) ; } }
import java . util . * ; class GFG { static final int MAX = 10000 ;
static int [ ] prefix = new int [ MAX + 1 ] ; static boolean isPowerOfTwo ( int x ) { if ( x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ) return true ; return false ; }
static void computePrefix ( int n , int a [ ] ) {
if ( isPowerOfTwo ( a [ 0 ] ) ) prefix [ 0 ] = 1 ; for ( int i = 1 ; i < n ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] ; if ( isPowerOfTwo ( a [ i ] ) ) prefix [ i ] ++ ; } }
static int query ( int L , int R ) { if ( L == 0 ) return prefix [ R ] ; return prefix [ R ] - prefix [ L - 1 ] ; }
public static void main ( String [ ] args ) { int A [ ] = { 3 , 8 , 5 , 2 , 5 , 10 } ; int N = A . length ; int Q = 2 ; computePrefix ( N , A ) ; System . out . println ( query ( 0 , 4 ) ) ; System . out . println ( query ( 3 , 5 ) ) ; } }
class GFG {
static void countIntgralPoints ( int x1 , int y1 , int x2 , int y2 ) { System . out . println ( ( y2 - y1 - 1 ) * ( x2 - x1 - 1 ) ) ; }
public static void main ( String args [ ] ) { int x1 = 1 , y1 = 1 ; int x2 = 4 , y2 = 4 ; countIntgralPoints ( x1 , y1 , x2 , y2 ) ; } }
class GFG {
static void findNextNumber ( int n ) { int h [ ] = new int [ 10 ] ; int i = 0 , msb = n , rem = 0 ; int next_num = - 1 , count = 0 ;
while ( msb > 9 ) { rem = msb % 10 ; h [ rem ] = 1 ; msb /= 10 ; count ++ ; } h [ msb ] = 1 ; count ++ ;
for ( i = msb + 1 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; break ; } }
if ( next_num == - 1 ) { for ( i = 1 ; i < msb ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; count ++ ; break ; } } }
if ( next_num > 0 ) {
for ( i = 0 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { msb = i ; break ; } }
for ( i = 1 ; i < count ; i ++ ) { next_num = ( ( next_num * 10 ) + msb ) ; }
if ( next_num > n ) System . out . print ( next_num + "NEW_LINE"); else System . out . print ( "Not Possible NEW_LINE"); } else { System . out . print ( "Not Possible NEW_LINE"); } }
public static void main ( String [ ] args ) { int n = 2019 ; findNextNumber ( n ) ; } }
import java . util . * ; class GFG {
static void CalculateValues ( int N ) { int A = 0 , B = 0 , C = 0 ;
for ( C = 0 ; C < N / 7 ; C ++ ) {
for ( B = 0 ; B < N / 5 ; B ++ ) {
A = N - 7 * C - 5 * B ;
if ( A >= 0 && A % 3 == 0 ) { System . out . print ( " A ▁ = ▁ " + A / 3 + " , ▁ B ▁ = ▁ " + B + " , ▁ C ▁ = ▁ " + C ) ; return ; } } }
System . out . println ( - 1 ) ; }
public static void main ( String [ ] args ) { int N = 19 ; CalculateValues ( 19 ) ; } }
import java . util . * ; class GFG {
static void minimumTime ( int [ ] arr , int n ) {
int sum = 0 ;
int T = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
System . out . println ( Math . max ( 2 * T , sum ) ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 8 , 3 } ; int N = arr . length ;
minimumTime ( arr , N ) ; } }
import java . util . * ; class GFG {
static void lexicographicallyMax ( String s ) {
int n = s . length ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
int count = 0 ;
int beg = i ;
int end = i ;
if ( s . charAt ( i ) == '1' ) count ++ ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( s . charAt ( j ) == '1' ) count ++ ; if ( count % 2 == 0 && count != 0 ) { end = j ; break ; } }
s = reverse ( s , beg , end + 1 ) ; }
System . out . println ( s ) ; } static String reverse ( String s , int beg , int end ) { StringBuilder x = new StringBuilder ( " " ) ; for ( int i = 0 ; i < beg ; i ++ ) x . append ( s . charAt ( i ) ) ; for ( int i = end - 1 ; i >= beg ; i -- ) x . append ( s . charAt ( i ) ) ; for ( int i = end ; i < s . length ( ) ; i ++ ) x . append ( s . charAt ( i ) ) ; return x . toString ( ) ; }
public static void main ( String args [ ] ) { String S = "0101" ; lexicographicallyMax ( S ) ; } }
import java . io . * ; import java . util . * ; class GFG {
public static void maxPairs ( int [ ] nums , int k ) {
Arrays . sort ( nums ) ;
int result = 0 ;
int start = 0 , end = nums . length - 1 ;
while ( start < end ) { if ( nums [ start ] + nums [ end ] > k )
end -- ; else if ( nums [ start ] + nums [ end ] < k )
start ++ ;
else { start ++ ; end -- ; result ++ ; } }
System . out . println ( result ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 } ; int K = 5 ;
maxPairs ( arr , K ) ; } }
import java . io . * ; import java . util . * ; class GFG {
public static void maxPairs ( int [ ] nums , int k ) {
Map < Integer , Integer > map = new HashMap < > ( ) ;
int result = 0 ;
for ( int i : nums ) {
if ( map . containsKey ( i ) && map . get ( i ) > 0 ) { map . put ( i , map . get ( i ) - 1 ) ; result ++ ; }
else { map . put ( k - i , map . getOrDefault ( k - i , 0 ) + 1 ) ; } }
System . out . println ( result ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 } ; int K = 5 ;
maxPairs ( arr , K ) ; } }
import java . util . * ; class GFG {
static void removeIndicesToMakeSumEqual ( int [ ] arr ) {
int N = arr . length ;
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
boolean find = false ;
int p = odd [ N - 1 ] ;
int q = even [ N - 1 ] - arr [ 0 ] ;
if ( p == q ) { System . out . print ( "0 ▁ " ) ; find = true ; }
for ( int i = 1 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) {
p = even [ N - 1 ] - even [ i - 1 ] - arr [ i ] + odd [ i - 1 ] ;
q = odd [ N - 1 ] - odd [ i - 1 ] + even [ i - 1 ] ; } else {
q = odd [ N - 1 ] - odd [ i - 1 ] - arr [ i ] + even [ i - 1 ] ;
p = even [ N - 1 ] - even [ i - 1 ] + odd [ i - 1 ] ; }
if ( p == q ) {
find = true ;
System . out . print ( i + " ▁ " ) ; } }
if ( ! find ) {
System . out . print ( - 1 ) ; } }
public static void main ( String [ ] args ) { int [ ] arr = { 4 , 1 , 6 , 2 } ; removeIndicesToMakeSumEqual ( arr ) ; } }
class GFG {
static void min_element_removal ( int arr [ ] , int N ) {
int left [ ] = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) left [ i ] = 1 ;
int right [ ] = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) right [ i ] = 1 ;
for ( int i = 1 ; i < N ; i ++ ) {
for ( int j = 0 ; j < i ; j ++ ) {
if ( arr [ j ] < arr [ i ] ) {
left [ i ] = Math . max ( left [ i ] , left [ j ] + 1 ) ; } } }
for ( int i = N - 2 ; i >= 0 ; i -- ) {
for ( int j = N - 1 ; j > i ; j -- ) {
if ( arr [ i ] > arr [ j ] ) {
right [ i ] = Math . max ( right [ i ] , right [ j ] + 1 ) ; } } }
int maxLen = 0 ;
for ( int i = 1 ; i < N - 1 ; i ++ ) {
maxLen = Math . max ( maxLen , left [ i ] + right [ i ] - 1 ) ; } System . out . println ( N - maxLen ) ; }
static void makeBitonic ( int arr [ ] , int N ) { if ( N == 1 ) { System . out . println ( "0" ) ; return ; } if ( N == 2 ) { if ( arr [ 0 ] != arr [ 1 ] ) System . out . println ( "0" ) ; else System . out . println ( "1" ) ; return ; } min_element_removal ( arr , N ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 1 , 1 , 5 , 6 , 2 , 3 , 1 } ; int N = arr . length ; makeBitonic ( arr , N ) ; } }
import java . util . * ; class GFG {
static void countSubarrays ( int A [ ] , int N ) {
int ans = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( A [ i ] != A [ i + 1 ] ) {
ans ++ ;
for ( int j = i - 1 , k = i + 2 ; j >= 0 && k < N && A [ j ] == A [ i ] && A [ k ] == A [ i + 1 ] ; j -- , k ++ ) {
ans ++ ; } } }
System . out . print ( ans + "NEW_LINE"); }
public static void main ( String [ ] args ) { int A [ ] = { 1 , 1 , 0 , 0 , 1 , 0 } ; int N = A . length ;
countSubarrays ( A , N ) ; } }
import java . util . * ; class GFG { static int maxN = 2002 ;
static int [ ] [ ] lcount = new int [ maxN ] [ maxN ] ;
static int [ ] [ ] rcount = new int [ maxN ] [ maxN ] ;
static void fill_counts ( int a [ ] , int n ) { int i , j ;
int maxA = a [ 0 ] ; for ( i = 0 ; i < n ; i ++ ) { if ( a [ i ] > maxA ) { maxA = a [ i ] ; } } for ( i = 0 ; i < n ; i ++ ) { lcount [ a [ i ] ] [ i ] = 1 ; rcount [ a [ i ] ] [ i ] = 1 ; } for ( i = 0 ; i <= maxA ; i ++ ) {
for ( j = 1 ; j < n ; j ++ ) { lcount [ i ] [ j ] = lcount [ i ] [ j - 1 ] + lcount [ i ] [ j ] ; }
for ( j = n - 2 ; j >= 0 ; j -- ) { rcount [ i ] [ j ] = rcount [ i ] [ j + 1 ] + rcount [ i ] [ j ] ; } } }
static int countSubsequence ( int a [ ] , int n ) { int i , j ; fill_counts ( a , n ) ; int answer = 0 ; for ( i = 1 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n - 1 ; j ++ ) { answer += lcount [ a [ j ] ] [ i - 1 ] * rcount [ a [ i ] ] [ j + 1 ] ; } } return answer ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 3 , 2 , 1 , 3 , 2 } ; System . out . print ( countSubsequence ( a , a . length ) ) ; } }
import java . io . * ; class GFG {
static String removeOuterParentheses ( String S ) {
String res = " " ;
int count = 0 ;
for ( int c = 0 ; c < S . length ( ) ; c ++ ) {
if ( S . charAt ( c ) == ' ( ' && count ++ > 0 )
res += S . charAt ( c ) ;
if ( S . charAt ( c ) == ' ) ' && count -- > 1 )
res += S . charAt ( c ) ; }
return res ; }
public static void main ( String [ ] args ) { String S = " ( ( ) ( ) ) ( ( ) ) ( ) " ; System . out . print ( removeOuterParentheses ( S ) ) ; } }
import java . util . * ; class GFG {
public static int maxiConsecutiveSubarray ( int arr [ ] , int N ) {
int maxi = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) {
int cnt = 1 , j ; for ( j = i ; j < N - 1 ; j ++ ) {
if ( arr [ j + 1 ] == arr [ j ] + 1 ) { cnt ++ ; }
else { break ; } }
maxi = Math . max ( maxi , cnt ) ; i = j ; }
return maxi ; }
public static void main ( String args [ ] ) { int N = 11 ; int arr [ ] = { 1 , 3 , 4 , 2 , 3 , 4 , 2 , 3 , 5 , 6 , 7 } ; System . out . println ( maxiConsecutiveSubarray ( arr , N ) ) ; } }
import java . util . * ; class GFG { static int N = 100005 ;
static void SieveOfEratosthenes ( boolean [ ] prime , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
static int digitSum ( int number ) {
int sum = 0 ; while ( number > 0 ) {
sum += ( number % 10 ) ; number /= 10 ; }
return sum ; }
static void longestCompositeDigitSumSubsequence ( int [ ] arr , int n ) { int count = 0 ; boolean [ ] prime = new boolean [ N + 1 ] ; for ( int i = 0 ; i <= N ; i ++ ) prime [ i ] = true ; SieveOfEratosthenes ( prime , N ) ; for ( int i = 0 ; i < n ; i ++ ) {
int res = digitSum ( arr [ i ] ) ;
if ( res == 1 ) { continue ; }
if ( prime [ res ] == false ) { count ++ ; } } System . out . println ( count ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 13 , 55 , 7 , 3 , 5 , 1 , 10 , 21 , 233 , 144 , 89 } ; int n = arr . length ;
longestCompositeDigitSumSubsequence ( arr , n ) ; } }
import java . util . * ; class GFG { static int sum ;
static class Node { int data ; Node left , right ; } ;
static Node newnode ( int data ) { Node temp = new Node ( ) ; temp . data = data ; temp . left = null ; temp . right = null ;
return temp ; }
static Node insert ( String s , int i , int N , Node root , Node temp ) { if ( i == N ) return temp ;
if ( s . charAt ( i ) == ' L ' ) root . left = insert ( s , i + 1 , N , root . left , temp ) ;
else root . right = insert ( s , i + 1 , N , root . right , temp ) ;
return root ; }
static int SBTUtil ( Node root ) {
if ( root == null ) return 0 ; if ( root . left == null && root . right == null ) return root . data ;
int left = SBTUtil ( root . left ) ;
int right = SBTUtil ( root . right ) ;
if ( root . left != null && root . right != null ) {
if ( ( left % 2 == 0 && right % 2 != 0 ) || ( left % 2 != 0 && right % 2 == 0 ) ) { sum += root . data ; } }
return left + right + root . data ; }
static Node build_tree ( int R , int N , String str [ ] , int values [ ] ) {
Node root = newnode ( R ) ; int i ;
for ( i = 0 ; i < N - 1 ; i ++ ) { String s = str [ i ] ; int x = values [ i ] ;
Node temp = newnode ( x ) ;
root = insert ( s , 0 , s . length ( ) , root , temp ) ; }
return root ; }
static void speciallyBalancedNodes ( int R , int N , String str [ ] , int values [ ] ) {
Node root = build_tree ( R , N , str , values ) ;
sum = 0 ;
SBTUtil ( root ) ;
System . out . print ( sum + " ▁ " ) ; }
public static void main ( String [ ] args ) {
int N = 7 ;
int R = 12 ;
String str [ ] = { " L " , " R " , " RL " , " RR " , " RLL " , " RLR " } ;
int values [ ] = { 17 , 16 , 4 , 9 , 2 , 3 } ;
speciallyBalancedNodes ( R , N , str , values ) ; } }
import java . util . * ; class GFG {
static void position ( int arr [ ] [ ] , int N ) {
int pos = - 1 ;
int count ;
for ( int i = 0 ; i < N ; i ++ ) {
count = 0 ; for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ 0 ] <= arr [ j ] [ 0 ] && arr [ i ] [ 1 ] >= arr [ j ] [ 1 ] ) { count ++ ; } }
if ( count == N ) { pos = i ; } }
if ( pos == - 1 ) { System . out . print ( pos ) ; }
else { System . out . print ( pos + 1 ) ; } }
public static void main ( String [ ] args ) {
int arr [ ] [ ] = { { 3 , 3 } , { 1 , 3 } , { 2 , 2 } , { 2 , 3 } , { 1 , 2 } } ; int N = arr . length ;
position ( arr , N ) ; } }
import java . util . * ; class GFG {
static void position ( int arr [ ] [ ] , int N ) {
int pos = - 1 ;
int right = Integer . MIN_VALUE ;
int left = Integer . MAX_VALUE ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] [ 1 ] > right ) { right = arr [ i ] [ 1 ] ; }
if ( arr [ i ] [ 0 ] < left ) { left = arr [ i ] [ 0 ] ; } }
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] [ 0 ] == left && arr [ i ] [ 1 ] == right ) { pos = i + 1 ; } }
System . out . print ( pos + "NEW_LINE"); }
public static void main ( String [ ] args ) {
int arr [ ] [ ] = { { 3 , 3 } , { 1 , 3 } , { 2 , 2 } , { 2 , 3 } , { 1 , 2 } } ; int N = arr . length ;
position ( arr , N ) ; } }
import java . util . * ; import java . io . * ; import java . lang . Math ; class GFG {
static int ctMinEdits ( String str1 , String str2 ) { int N1 = str1 . length ( ) ; int N2 = str2 . length ( ) ;
int freq1 [ ] = new int [ 256 ] ; Arrays . fill ( freq1 , 0 ) ; for ( int i = 0 ; i < N1 ; i ++ ) { freq1 [ str1 . charAt ( i ) ] ++ ; }
int freq2 [ ] = new int [ 256 ] ; Arrays . fill ( freq2 , 0 ) ; for ( int i = 0 ; i < N2 ; i ++ ) { freq2 [ str2 . charAt ( i ) ] ++ ; }
for ( int i = 0 ; i < 256 ; i ++ ) {
if ( freq1 [ i ] > freq2 [ i ] ) { freq1 [ i ] = freq1 [ i ] - freq2 [ i ] ; freq2 [ i ] = 0 ; }
else { freq2 [ i ] = freq2 [ i ] - freq1 [ i ] ; freq1 [ i ] = 0 ; } }
int sum1 = 0 ;
int sum2 = 0 ; for ( int i = 0 ; i < 256 ; i ++ ) { sum1 += freq1 [ i ] ; sum2 += freq2 [ i ] ; } return Math . max ( sum1 , sum2 ) ; }
public static void main ( final String [ ] args ) { String str1 = " geeksforgeeks " ; String str2 = " geeksforcoder " ; System . out . println ( ctMinEdits ( str1 , str2 ) ) ; } }
import java . util . * ; import java . io . * ; class GFG {
static void CountPairs ( int a [ ] , int b [ ] , int n ) {
int C [ ] = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) { C [ i ] = a [ i ] + b [ i ] ; }
HashMap < Integer , Integer > freqCount = new HashMap < > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( ! freqCount . containsKey ( C [ i ] ) ) freqCount . put ( C [ i ] , 1 ) ; else freqCount . put ( C [ i ] , freqCount . get ( C [ i ] ) + 1 ) ; }
int NoOfPairs = 0 ; for ( Map . Entry < Integer , Integer > x : freqCount . entrySet ( ) ) { int y = x . getValue ( ) ;
NoOfPairs = NoOfPairs + y * ( y - 1 ) / 2 ; }
System . out . println ( NoOfPairs ) ; }
public static void main ( String args [ ] ) {
int arr [ ] = { 1 , 4 , 20 , 3 , 10 , 5 } ; int brr [ ] = { 9 , 6 , 1 , 7 , 11 , 6 } ;
int N = arr . length ;
CountPairs ( arr , brr , N ) ; } }
import java . util . * ; class GFG {
public static void medianChange ( List < Integer > arr1 , List < Integer > arr2 ) { int N = arr1 . size ( ) ;
List < Integer > median = new ArrayList < > ( ) ;
if ( ( N & 1 ) != 0 ) median . add ( arr1 . get ( N / 2 ) * 1 ) ;
else median . add ( ( arr1 . get ( N / 2 ) + arr1 . get ( ( N - 1 ) / 2 ) ) / 2 ) ; for ( int x = 0 ; x < arr2 . size ( ) ; x ++ ) {
int it = arr1 . indexOf ( arr2 . get ( x ) ) ;
arr1 . remove ( it ) ;
N -- ;
if ( ( N & 1 ) != 0 ) { median . add ( arr1 . get ( N / 2 ) * 1 ) ; }
else { median . add ( ( arr1 . get ( N / 2 ) + arr1 . get ( ( N - 1 ) / 2 ) ) / 2 ) ; } }
for ( int i = 0 ; i < median . size ( ) - 1 ; i ++ ) { System . out . print ( median . get ( i + 1 ) - median . get ( i ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) {
List < Integer > arr1 = new ArrayList < Integer > ( ) { { add ( 2 ) ; add ( 4 ) ; add ( 6 ) ; add ( 8 ) ; add ( 10 ) ; } } ; List < Integer > arr2 = new ArrayList < Integer > ( ) { { add ( 4 ) ; add ( 6 ) ; } } ;
medianChange ( arr1 , arr2 ) ; } }
class GFG {
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
static boolean checkA ( String s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 1 ) state1 ( s . charAt ( i ) ) ; else if ( nfa == 2 ) state2 ( s . charAt ( i ) ) ; else if ( nfa == 3 ) state3 ( s . charAt ( i ) ) ; } if ( nfa == 1 ) { return true ; } else { nfa = 4 ; } return false ; }
static boolean checkB ( String s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 4 ) state4 ( s . charAt ( i ) ) ; else if ( nfa == 5 ) state5 ( s . charAt ( i ) ) ; else if ( nfa == 6 ) state6 ( s . charAt ( i ) ) ; } if ( nfa == 4 ) { return true ; } else { nfa = 7 ; } return false ; }
static boolean checkC ( String s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 7 ) state7 ( s . charAt ( i ) ) ; else if ( nfa == 8 ) state8 ( s . charAt ( i ) ) ; else if ( nfa == 9 ) state9 ( s . charAt ( i ) ) ; } if ( nfa == 7 ) { return true ; } return false ; }
public static void main ( String [ ] args ) { String s = " bbbca " ; int x = 5 ;
if ( checkA ( s , x ) || checkB ( s , x ) || checkC ( s , x ) ) { System . out . println ( " ACCEPTED " ) ; } else { if ( flag == 0 ) { System . out . println ( " NOT ▁ ACCEPTED " ) ; } else { System . out . println ( " INPUT ▁ OUT ▁ OF ▁ DICTIONARY . " ) ; } } } }
class GFG {
static int getPositionCount ( int a [ ] , int n ) {
int count = 1 ;
int min = a [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) {
if ( a [ i ] <= min ) {
min = a [ i ] ;
count ++ ; } } return count ; }
public static void main ( String [ ] args ) { int a [ ] = { 5 , 4 , 6 , 1 , 3 , 1 } ; int n = a . length ; System . out . print ( getPositionCount ( a , n ) ) ; } }
class GFG {
static int maxSum ( int arr [ ] , int n , int k ) {
if ( n < k ) { return - 1 ; }
int res = 0 ; for ( int i = 0 ; i < k ; i ++ ) res += arr [ i ] ;
int curr_sum = res ; for ( int i = k ; i < n ; i ++ ) { curr_sum += arr [ i ] - arr [ i - k ] ; res = Math . max ( res , curr_sum ) ; } return res ; }
static int solve ( int arr [ ] , int n , int k ) { int max_len = 0 , l = 0 , r = n , m ;
while ( l <= r ) { m = ( l + r ) / 2 ;
if ( maxSum ( arr , n , m ) > k ) r = m - 1 ; else { l = m + 1 ;
max_len = m ; } } return max_len ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . length ; int k = 10 ; System . out . println ( solve ( arr , n , k ) ) ; } }
import java . util . * ; class GFG { static int MAX = 100001 ; static int ROW = 10 ; static int COl = 3 ; static Vector < Integer > [ ] indices = new Vector [ MAX ] ;
static int test [ ] [ ] = { { 2 , 3 , 6 } , { 2 , 4 , 4 } , { 2 , 6 , 3 } , { 3 , 2 , 6 } , { 3 , 3 , 3 } , { 3 , 6 , 2 } , { 4 , 2 , 4 } , { 4 , 4 , 2 } , { 6 , 2 , 3 } , { 6 , 3 , 2 } } ;
static int find_triplet ( int array [ ] , int n ) { int answer = 0 ; for ( int i = 0 ; i < MAX ; i ++ ) { indices [ i ] = new Vector < > ( ) ; }
for ( int i = 0 ; i < n ; i ++ ) { indices [ array [ i ] ] . add ( i ) ; } for ( int i = 0 ; i < n ; i ++ ) { int y = array [ i ] ; for ( int j = 0 ; j < ROW ; j ++ ) { int s = test [ j ] [ 1 ] * y ;
if ( s % test [ j ] [ 0 ] != 0 ) continue ; if ( s % test [ j ] [ 2 ] != 0 ) continue ; int x = s / test [ j ] [ 0 ] ; int z = s / test [ j ] [ 2 ] ; if ( x > MAX z > MAX ) continue ; int l = 0 ; int r = indices [ x ] . size ( ) - 1 ; int first = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( indices [ x ] . get ( m ) < i ) { first = m ; l = m + 1 ; } else { r = m - 1 ; } } l = 0 ; r = indices [ z ] . size ( ) - 1 ; int third = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( indices [ z ] . get ( m ) > i ) { third = m ; r = m - 1 ; } else { l = m + 1 ; } } if ( first != - 1 && third != - 1 ) {
answer += ( first + 1 ) * ( indices [ z ] . size ( ) - third ) ; } } } return answer ; }
public static void main ( String [ ] args ) { int array [ ] = { 2 , 4 , 5 , 6 , 7 } ; int n = array . length ; System . out . println ( find_triplet ( array , n ) ) ; } }
class GFG { static int distinct ( int [ ] arr , int n ) { int count = 0 ;
if ( n == 1 ) return 1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( i == 0 ) { if ( arr [ i ] != arr [ i + 1 ] ) count += 1 ; }
else { if ( arr [ i ] != arr [ i + 1 ] arr [ i ] != arr [ i - 1 ] ) count += 1 ; } }
if ( arr [ n - 1 ] != arr [ n - 2 ] ) count += 1 ; return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 0 , 0 , 0 , 0 , 0 , 1 , 0 } ; int n = arr . length ; System . out . println ( distinct ( arr , n ) ) ; } }
import java . io . * ; import java . lang . * ; import java . util . * ; public class GFG {
static boolean isSorted ( int [ ] [ ] arr , int N ) {
for ( int i = 1 ; i < N ; i ++ ) { if ( arr [ i ] [ 0 ] > arr [ i - 1 ] [ 0 ] ) { return false ; } }
return true ; }
static String isPossibleToSort ( int [ ] [ ] arr , int N ) {
int group = arr [ 0 ] [ 1 ] ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] [ 1 ] != group ) { return " Yes " ; } }
if ( isSorted ( arr , N ) ) { return " Yes " ; } else { return " No " ; } }
public static void main ( String [ ] args ) { int arr [ ] [ ] = { { 340000 , 2 } , { 45000 , 1 } , { 30000 , 2 } , { 50000 , 4 } } ; int N = arr . length ; System . out . print ( isPossibleToSort ( arr , N ) ) ; } }
import java . lang . * ; import java . util . * ;
class Node { Node left , right ; int data ; public Node ( int data ) { this . data = data ; left = null ; right = null ; } } class AlphaScore { Node root ; AlphaScore ( ) { root = null ; } static long sum = 0 , total_sum = 0 ; static long mod = 1000000007 ;
public static long getAlphaScore ( Node node ) {
if ( node . left != null ) getAlphaScore ( node . left ) ;
sum = ( sum + node . data ) % mod ;
total_sum = ( total_sum + sum ) % mod ;
if ( node . right != null ) getAlphaScore ( node . right ) ;
return total_sum ; }
public static Node constructBST ( int [ ] arr , int start , int end , Node root ) { if ( start > end ) return null ; int mid = ( start + end ) / 2 ;
if ( root == null ) root = new Node ( arr [ mid ] ) ;
root . left = constructBST ( arr , start , mid - 1 , root . left ) ;
root . right = constructBST ( arr , mid + 1 , end , root . right ) ;
return root ; }
public static void main ( String args [ ] ) { int arr [ ] = { 10 , 11 , 12 } ; int length = arr . length ;
Arrays . sort ( arr ) ; Node root = null ;
root = constructBST ( arr , 0 , length - 1 , root ) ; System . out . println ( getAlphaScore ( root ) ) ; } }
import java . util . * ; class GFG {
static int sortByFreq ( Integer [ ] arr , int n ) {
int maxE = - 1 ;
for ( int i = 0 ; i < n ; i ++ ) { maxE = Math . max ( maxE , arr [ i ] ) ; }
int freq [ ] = new int [ maxE + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; }
int cnt = 0 ;
for ( int i = 0 ; i <= maxE ; i ++ ) {
if ( freq [ i ] > 0 ) { int value = 100000 - i ; arr [ cnt ] = 100000 * freq [ i ] + value ; cnt ++ ; } }
return cnt ; }
static void printSortedArray ( Integer [ ] arr , int cnt ) {
for ( int i = 0 ; i < cnt ; i ++ ) {
int frequency = arr [ i ] / 100000 ;
int value = 100000 - ( arr [ i ] % 100000 ) ;
for ( int j = 0 ; j < frequency ; j ++ ) { System . out . print ( value + " ▁ " ) ; } } }
public static void main ( String [ ] args ) { Integer arr [ ] = { 4 , 4 , 5 , 6 , 4 , 2 , 2 , 8 , 5 } ;
int n = arr . length ;
int cnt = sortByFreq ( arr , n ) ;
Arrays . sort ( arr , Collections . reverseOrder ( ) ) ;
printSortedArray ( arr , cnt ) ; } }
import java . util . * ; class GFG {
static boolean checkRectangles ( int [ ] arr , int n ) { boolean ans = true ;
Arrays . sort ( arr ) ;
int area = arr [ 0 ] * arr [ 4 * n - 1 ] ;
for ( int i = 0 ; i < 2 * n ; i = i + 2 ) { if ( arr [ i ] != arr [ i + 1 ] arr [ 4 * n - i - 1 ] != arr [ 4 * n - i - 2 ] arr [ i ] * arr [ 4 * n - i - 1 ] != area ) {
ans = false ; break ; } }
if ( ans ) return true ; return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 8 , 2 , 1 , 2 , 4 , 4 , 8 } ; int n = 2 ; if ( checkRectangles ( arr , n ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
import java . util . * ; class GFG {
static int cntElements ( int arr [ ] , int n ) {
int copy_arr [ ] = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) copy_arr [ i ] = arr [ i ] ;
int count = 0 ;
Arrays . sort ( arr ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != copy_arr [ i ] ) { count ++ ; } } return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 6 , 2 , 4 , 5 } ; int n = arr . length ; System . out . println ( cntElements ( arr , n ) ) ; } }
import java . util . * ; class GFG { static class pair { int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static void findPairs ( int arr [ ] , int n , int k , int d ) {
if ( n < 2 * k ) { System . out . print ( - 1 ) ; return ; }
Vector < pair > pairs = new Vector < pair > ( ) ;
Arrays . sort ( arr ) ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( arr [ n - k + i ] - arr [ i ] >= d ) {
pair p = new pair ( arr [ i ] , arr [ n - k + i ] ) ; pairs . add ( p ) ; } }
if ( pairs . size ( ) < k ) { System . out . print ( - 1 ) ; return ; }
for ( pair v : pairs ) { System . out . println ( " ( " + v . first + " , ▁ " + v . second + " ) " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 6 , 10 , 23 , 14 , 7 , 2 , 20 , 9 } ; int n = arr . length ; int k = 4 , d = 3 ; findPairs ( arr , n , k , d ) ; } }
import java . util . Arrays ; import java . io . * ; class GFG {
static int pairs_count ( int arr [ ] , int n , int sum ) {
int ans = 0 ;
Arrays . sort ( arr ) ;
int i = 0 , j = n - 1 ; while ( i < j ) {
if ( arr [ i ] + arr [ j ] < sum ) i ++ ;
else if ( arr [ i ] + arr [ j ] > sum ) j -- ;
else {
int x = arr [ i ] , xx = i ; while ( ( i < j ) && ( arr [ i ] == x ) ) i ++ ;
int y = arr [ j ] , yy = j ; while ( ( j >= i ) && ( arr [ j ] == y ) ) j -- ;
if ( x == y ) { int temp = i - xx + yy - j - 1 ; ans += ( temp * ( temp + 1 ) ) / 2 ; } else ans += ( i - xx ) * ( yy - j ) ; } }
return ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 5 , 7 , 5 , - 1 } ; int n = arr . length ; int sum = 6 ; System . out . println ( pairs_count ( arr , n , sum ) ) ; } }
public class GFG { public static boolean check ( String str ) { int min = Integer . MAX_VALUE ; int max = Integer . MIN_VALUE ; int sum = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
int ascii = ( int ) str . charAt ( i ) ;
if ( ascii < 96 ascii > 122 ) return false ;
sum += ascii ;
if ( min > ascii ) min = ascii ;
if ( max < ascii ) max = ascii ; }
min -= 1 ;
int eSum = ( ( max * ( max + 1 ) ) / 2 ) - ( ( min * ( min + 1 ) ) / 2 ) ;
return sum == eSum ; }
public static void main ( String [ ] args ) {
String str = " dcef " ; if ( check ( str ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ;
String str1 = " xyza " ; if ( check ( str1 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . util . * ; class GFG {
static int findKth ( int arr [ ] , int n , int k ) { HashSet < Integer > missing = new HashSet < > ( ) ; int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { missing . add ( arr [ i ] ) ; }
int maxm = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ; int minm = Arrays . stream ( arr ) . min ( ) . getAsInt ( ) ;
for ( int i = minm + 1 ; i < maxm ; i ++ ) {
if ( ! missing . contains ( i ) ) { count ++ ; }
if ( count == k ) { return i ; } }
return - 1 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 10 , 9 , 4 } ; int n = arr . length ; int k = 5 ; System . out . println ( findKth ( arr , n , k ) ) ; } }
import java . util . * ; class GFG {
static class Node { int data ; Node next ; } ; static Node start ;
static void sortList ( Node head ) { int startVal = 1 ; while ( head != null ) { head . data = startVal ; startVal ++ ; head = head . next ; } }
static void push ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = head_ref ;
head_ref = new_node ; start = head_ref ; }
static void printList ( Node node ) { while ( node != null ) { System . out . print ( node . data + " ▁ " ) ; node = node . next ; } }
public static void main ( String [ ] args ) { start = null ;
push ( start , 2 ) ; push ( start , 1 ) ; push ( start , 6 ) ; push ( start , 4 ) ; push ( start , 5 ) ; push ( start , 3 ) ; sortList ( start ) ; printList ( start ) ; } }
class GfG {
static class Node { int data ; Node next ; }
static boolean isSortedDesc ( Node head ) {
if ( head == null head . next == null ) return true ;
return ( head . data > head . next . data && isSortedDesc ( head . next ) ) ; } static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . next = null ; temp . data = data ; return temp ; }
public static void main ( String [ ] args ) { Node head = newNode ( 7 ) ; head . next = newNode ( 5 ) ; head . next . next = newNode ( 4 ) ; head . next . next . next = newNode ( 3 ) ; if ( isSortedDesc ( head ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . util . Arrays ; import java . util . Collections ; import java . util . Comparator ; import java . util . Vector ; class GFG { static int minSum ( int arr [ ] , int n ) {
Vector < Integer > evenArr = new Vector < > ( ) ; Vector < Integer > oddArr = new Vector < > ( ) ;
Arrays . sort ( arr ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( i < n / 2 ) { oddArr . add ( arr [ i ] ) ; } else { evenArr . add ( arr [ i ] ) ; } }
Comparator comparator = Collections . reverseOrder ( ) ; Collections . sort ( evenArr , comparator ) ;
int i = 0 , sum = 0 ; for ( int j = 0 ; j < evenArr . size ( ) ; j ++ ) { arr [ i ++ ] = evenArr . get ( j ) ; arr [ i ++ ] = oddArr . get ( j ) ; sum += evenArr . get ( j ) * oddArr . get ( j ) ; } return sum ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 } ; int n = arr . length ; System . out . println ( " Minimum ▁ required ▁ sum ▁ = ▁ " + minSum ( arr , n ) ) ; System . out . println ( " Sorted ▁ array ▁ in ▁ required ▁ format ▁ : ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } } }
class GFG {
static void minTime ( String word ) { int ans = 0 ;
int curr = 0 ; for ( int i = 0 ; i < word . length ( ) ; i ++ ) {
int k = ( int ) word . charAt ( i ) - 97 ;
int a = Math . abs ( curr - k ) ;
int b = 26 - Math . abs ( curr - k ) ;
ans += Math . min ( a , b ) ;
ans ++ ; curr = ( int ) word . charAt ( i ) - 97 ; }
System . out . print ( ans ) ; }
public static void main ( String [ ] args ) {
String str = " zjpc " ;
minTime ( str ) ; } }
import java . io . * ; class GFG {
static int reduceToOne ( long N ) {
int cnt = 0 ; while ( N != 1 ) {
if ( N == 2 || ( N % 2 == 1 ) ) {
N = N - 1 ;
cnt ++ ; }
else if ( N % 2 == 0 ) {
N = N / ( N / 2 ) ;
cnt ++ ; } }
return cnt ; }
public static void main ( String [ ] args ) { long N = 35 ; System . out . println ( reduceToOne ( N ) ) ; } }
import java . util . * ; class GFG {
static void maxDiamonds ( int A [ ] , int N , int K ) {
PriorityQueue < Integer > pq = new PriorityQueue < > ( ( a , b ) -> b - a ) ;
for ( int i = 0 ; i < N ; i ++ ) { pq . add ( A [ i ] ) ; }
int ans = 0 ;
while ( ! pq . isEmpty ( ) && K -- > 0 ) {
int top = pq . peek ( ) ;
pq . remove ( ) ;
ans += top ;
top = top / 2 ; pq . add ( top ) ; }
System . out . print ( ans ) ; }
public static void main ( String [ ] args ) { int A [ ] = { 2 , 1 , 7 , 4 , 2 } ; int K = 3 ; int N = A . length ; maxDiamonds ( A , N , K ) ; } }
import java . io . * ; class GFG {
static int MinimumCost ( int A [ ] , int B [ ] , int N ) {
int totalCost = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int mod_A = B [ i ] % A [ i ] ; int totalCost_A = Math . min ( mod_A , A [ i ] - mod_A ) ;
int mod_B = A [ i ] % B [ i ] ; int totalCost_B = Math . min ( mod_B , B [ i ] - mod_B ) ;
totalCost += Math . min ( totalCost_A , totalCost_B ) ; }
return totalCost ; }
public static void main ( String [ ] args ) { int A [ ] = { 3 , 6 , 3 } ; int B [ ] = { 4 , 8 , 13 } ; int N = A . length ; System . out . print ( MinimumCost ( A , B , N ) ) ; } }
import java . io . * ; class GFG {
static void printLargestDivisible ( int arr [ ] , int N ) { int i , count0 = 0 , count7 = 0 ; for ( i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 0 ) count0 ++ ; else count7 ++ ; }
if ( count7 % 50 == 0 ) { while ( count7 != 0 ) { System . out . print ( 7 ) ; count7 -= 1 ; } while ( count0 != 0 ) { System . out . print ( 0 ) ; count0 -= 1 ; } }
else if ( count7 < 5 ) { if ( count0 == 0 ) System . out . print ( " No " ) ; else System . out . print ( "0" ) ; }
else {
count7 = count7 - count7 % 5 ; while ( count7 != 0 ) { System . out . print ( 7 ) ; count7 -= 1 ; } while ( count0 != 0 ) { System . out . print ( 0 ) ; count0 -= 1 ; } } }
public static void main ( String [ ] args ) {
int arr [ ] = { 0 , 7 , 0 , 7 , 7 , 7 , 7 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 7 , 7 } ;
int N = arr . length ; printLargestDivisible ( arr , N ) ; } }
import java . util . * ; class GFG {
static int findMaxValByRearrArr ( int arr [ ] , int N ) {
Arrays . sort ( arr ) ;
int res = 0 ;
do {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += __gcd ( i + 1 , arr [ i ] ) ; }
res = Math . max ( res , sum ) ; } while ( next_permutation ( arr ) ) ; return res ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; } static boolean next_permutation ( int [ ] p ) { for ( int a = p . length - 2 ; a >= 0 ; -- a ) if ( p [ a ] < p [ a + 1 ] ) for ( int b = p . length - 1 ; ; -- b ) if ( p [ b ] > p [ a ] ) { int t = p [ a ] ; p [ a ] = p [ b ] ; p [ b ] = t ; for ( ++ a , b = p . length - 1 ; a < b ; ++ a , -- b ) { t = p [ a ] ; p [ a ] = p [ b ] ; p [ b ] = t ; } return true ; } return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 2 , 1 } ; int N = arr . length ; System . out . print ( findMaxValByRearrArr ( arr , N ) ) ; } }
import java . util . * ; class GFG {
public static int min_elements ( int arr [ ] , int N ) {
Map < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
mp . put ( arr [ i ] , mp . getOrDefault ( arr [ i ] , 0 ) + 1 ) ; }
int cntMinRem = 0 ;
for ( int key : mp . keySet ( ) ) {
int i = key ; int val = mp . get ( i ) ;
if ( val < i ) {
cntMinRem += val ; }
else if ( val > i ) {
cntMinRem += ( val - i ) ; } } return cntMinRem ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 1 , 4 , 2 } ; System . out . println ( min_elements ( arr , arr . length ) ) ; } }
import java . util . * ; class GFG {
static boolean CheckAllarrayEqual ( int [ ] arr , int N ) {
if ( N == 1 ) { return true ; }
int totalSum = arr [ 0 ] ;
int secMax = Integer . MIN_VALUE ;
int Max = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) { if ( arr [ i ] >= Max ) {
secMax = Max ;
Max = arr [ i ] ; } else if ( arr [ i ] > secMax ) {
secMax = arr [ i ] ; }
totalSum += arr [ i ] ; }
if ( ( secMax * ( N - 1 ) ) > totalSum ) { return false ; }
if ( totalSum % ( N - 1 ) != 0 ) { return false ; } return true ; }
public static void main ( String [ ] args ) { int [ ] arr = { 6 , 2 , 2 , 2 } ; int N = arr . length ; if ( CheckAllarrayEqual ( arr , N ) ) { System . out . print ( " YES " ) ; } else { System . out . print ( " NO " ) ; } } }
class GFG {
static void Remove_one_element ( int arr [ ] , int n ) {
int post_odd = 0 , post_even = 0 ;
int curr_odd = 0 , curr_even = 0 ;
int res = 0 ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( i % 2 != 0 ) post_odd ^= arr [ i ] ;
else post_even ^= arr [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( i % 2 != 0 ) post_odd ^= arr [ i ] ;
else post_even ^= arr [ i ] ;
int X = curr_odd ^ post_even ;
int Y = curr_even ^ post_odd ;
if ( X == Y ) res ++ ;
if ( i % 2 != 0 ) curr_odd ^= arr [ i ] ;
else curr_even ^= arr [ i ] ; }
System . out . println ( res ) ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 0 , 1 , 0 , 1 } ;
int N = arr . length ;
Remove_one_element ( arr , N ) ; } }
class GFG {
static int cntIndexesToMakeBalance ( int arr [ ] , int n ) {
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
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 1 } ; int n = arr . length ; System . out . println ( cntIndexesToMakeBalance ( arr , n ) ) ; } }
import java . util . * ; class GFG {
static void findNums ( int X , int Y ) {
int A , B ;
if ( X < Y ) { A = - 1 ; B = - 1 ; }
else if ( ( ( Math . abs ( X - Y ) ) & 1 ) != 0 ) { A = - 1 ; B = - 1 ; }
else if ( X == Y ) { A = 0 ; B = Y ; }
else {
A = ( X - Y ) / 2 ;
if ( ( A & Y ) == 0 ) {
B = ( A + Y ) ; }
else { A = - 1 ; B = - 1 ; } }
System . out . print ( A + " ▁ " + B ) ; }
public static void main ( String [ ] args ) {
int X = 17 , Y = 13 ;
findNums ( X , Y ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static void checkCount ( int A [ ] , int Q [ ] [ ] , int q ) {
for ( int i = 0 ; i < q ; i ++ ) { int L = Q [ i ] [ 0 ] ; int R = Q [ i ] [ 1 ] ;
L -- ; R -- ;
if ( ( A [ L ] < A [ L + 1 ] ) != ( A [ R - 1 ] < A [ R ] ) ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
public static void main ( String [ ] args ) { int arr [ ] = { 11 , 13 , 12 , 14 } ; int Q [ ] [ ] = { { 1 , 4 } , { 2 , 4 } } ; int q = Q . length ; checkCount ( arr , Q , q ) ; } }
import java . util . * ; class GFG {
static double pairProductMean ( int arr [ ] , int N ) {
Vector < Integer > pairArray = new Vector < > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) { int pairProduct = arr [ i ] * arr [ j ] ;
pairArray . add ( pairProduct ) ; } }
int length = pairArray . size ( ) ;
float sum = 0 ; for ( int i = 0 ; i < length ; i ++ ) sum += pairArray . get ( i ) ;
float mean ;
if ( length != 0 ) mean = sum / length ; else mean = 0 ;
return mean ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 2 , 4 , 8 } ; int N = arr . length ;
System . out . format ( " % .2f " , pairProductMean ( arr , N ) ) ; } }
import java . util . * ; class GFG {
static void findPlayer ( String str [ ] , int n ) {
int move_first = 0 ;
int move_sec = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( str [ i ] . charAt ( 0 ) == str [ i ] . charAt ( str [ i ] . length ( ) - 1 ) ) {
if ( str [ i ] . charAt ( 0 ) == 48 ) move_first ++ ; else move_sec ++ ; } }
if ( move_first <= move_sec ) { System . out . print ( " Player ▁ 2 ▁ wins " ) ; } else { System . out . print ( " Player ▁ 1 ▁ wins " ) ; } }
public static void main ( String [ ] args ) {
String str [ ] = { "010" , "101" } ; int N = str [ 0 ] . length ( ) ;
findPlayer ( str , N ) ; } }
import java . util . * ; class GFG {
static int find_next ( int n , int k ) {
int M = n + 1 ; while ( true ) {
if ( ( M & ( 1L << k ) ) > 0 ) break ;
M ++ ; }
return M ; }
public static void main ( String [ ] args ) {
int N = 15 , K = 2 ;
System . out . print ( find_next ( N , K ) ) ; } }
import java . util . * ; class GFG {
static int find_next ( int n , int k ) {
int ans = 0 ;
if ( ( n & ( 1L << k ) ) == 0 ) { int cur = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( ( n & ( 1L << i ) ) > 0 ) cur += 1L << i ; }
ans = ( int ) ( n - cur + ( 1L << k ) ) ; }
else { int first_unset_bit = - 1 , cur = 0 ; for ( int i = 0 ; i < 64 ; i ++ ) {
if ( ( n & ( 1L << i ) ) == 0 ) { first_unset_bit = i ; break ; }
else cur += ( 1L << i ) ; }
ans = ( int ) ( n - cur + ( 1L << first_unset_bit ) ) ;
if ( ( ans & ( 1L << k ) ) == 0 ) ans += ( 1L << k ) ; }
return ans ; }
public static void main ( String [ ] args ) { int N = 15 , K = 2 ;
System . out . print ( find_next ( N , K ) ) ; } }
class GFG { static String largestString ( String num , int k ) {
String ans = " " ; for ( char i : num . toCharArray ( ) ) {
while ( ans . length ( ) > 0 && ans . charAt ( ans . length ( ) - 1 ) < i && k > 0 ) {
ans = ans . substring ( 0 , ans . length ( ) - 1 ) ;
k -- ; }
ans += i ; }
while ( ans . length ( ) > 0 && k -- > 0 ) { ans = ans . substring ( 0 , ans . length ( ) - 1 ) ; }
return ans ; }
public static void main ( String [ ] args ) { String str = " zyxedcba " ; int k = 1 ; System . out . print ( largestString ( str , k ) + "NEW_LINE"); } }
class GFG {
static void maxLengthSubArray ( int A [ ] , int N ) {
int forward [ ] = new int [ N ] ; int backward [ ] = new int [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) { if ( i == 0 A [ i ] != A [ i - 1 ] ) { forward [ i ] = 1 ; } else forward [ i ] = forward [ i - 1 ] + 1 ; }
for ( int i = N - 1 ; i >= 0 ; i -- ) { if ( i == N - 1 A [ i ] != A [ i + 1 ] ) { backward [ i ] = 1 ; } else backward [ i ] = backward [ i + 1 ] + 1 ; }
int ans = 0 ;
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( A [ i ] != A [ i + 1 ] ) ans = Math . max ( ans , Math . min ( forward [ i ] , backward [ i + 1 ] ) * 2 ) ; }
System . out . println ( ans ) ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 2 , 3 , 4 , 4 , 4 , 6 , 6 , 6 , 9 } ;
int N = arr . length ;
maxLengthSubArray ( arr , N ) ; } }
class GFG {
static void minNum ( int n ) { if ( n < 3 ) System . out . println ( - 1 ) ; else System . out . println ( 210 * ( ( int ) ( Math . pow ( 10 , n - 1 ) / 210 ) + 1 ) ) ; }
public static void main ( String [ ] args ) { int n = 5 ; minNum ( n ) ; } }
import java . util . * ; @ SuppressWarnings ( " unchecked " ) class GFG {
static String helper ( int d , int s ) {
StringBuilder ans = new StringBuilder ( ) ; for ( int i = 0 ; i < d ; i ++ ) { ans . append ( "0" ) ; } for ( int i = d - 1 ; i >= 0 ; i -- ) {
if ( s >= 9 ) { ans . setCharAt ( i , '9' ) ; s -= 9 ; }
else { char c = ( char ) ( s + ( int ) '0' ) ; ans . setCharAt ( i , c ) ; s = 0 ; } } return ans . toString ( ) ; }
static String findMin ( int x , int Y ) {
String y = Integer . toString ( Y ) ; int n = y . length ( ) ; ArrayList p = new ArrayList ( ) ; for ( int i = 0 ; i < n ; i ++ ) { p . add ( 0 ) ; }
for ( int i = 0 ; i < n ; i ++ ) { p . add ( i , ( int ) ( ( int ) y . charAt ( i ) - ( int ) '0' ) ) ; if ( i > 0 ) { p . add ( i , ( int ) p . get ( i ) + ( int ) p . get ( i - 1 ) ) ; } }
for ( int i = n - 1 , k = 0 ; ; i -- , k ++ ) {
int d = 0 ; if ( i >= 0 ) { d = ( int ) y . charAt ( i ) - ( int ) '0' ; }
for ( int j = d + 1 ; j <= 9 ; j ++ ) { int r = j ;
if ( i > 0 ) { r += ( int ) p . get ( i - 1 ) ; }
if ( x - r >= 0 && x - r <= 9 * k ) {
String suf = helper ( k , x - r ) ; String pre = " " ; if ( i > 0 ) pre = y . substring ( 0 , i ) ;
char cur = ( char ) ( j + ( int ) '0' ) ; pre += cur ;
return pre + suf ; } } } }
public static void main ( String [ ] arg ) {
int x = 18 ; int y = 99 ;
System . out . print ( findMin ( x , y ) ) ; } }
import java . util . * ; class GFG {
public static void largestNumber ( int n , int X , int Y ) { int maxm = Math . max ( X , Y ) ;
Y = X + Y - maxm ;
X = maxm ;
int Xs = 0 ; int Ys = 0 ; while ( n > 0 ) {
if ( n % Y == 0 ) {
Xs += n ;
n = 0 ; } else {
n -= X ;
Ys += X ; } }
if ( n == 0 ) { while ( Xs -- > 0 ) System . out . print ( X ) ; while ( Ys -- > 0 ) System . out . print ( Y ) ; }
else System . out . print ( " - 1" ) ; }
public static void main ( String [ ] args ) { int n = 19 , X = 7 , Y = 5 ; largestNumber ( n , X , Y ) ; } }
import java . io . * ; class GFG { static int minChanges ( String str , int N ) { int res ; int count0 = 0 , count1 = 0 ;
for ( char x : str . toCharArray ( ) ) { if ( x == '0' ) count0 ++ ; } res = count0 ;
for ( char x : str . toCharArray ( ) ) { if ( x == '0' ) count0 -- ; if ( x == '1' ) count1 ++ ; res = Math . min ( res , count1 + count0 ) ; } return res ; }
public static void main ( String [ ] args ) { int N = 9 ; String str = "000101001" ; System . out . println ( minChanges ( str , N ) ) ; } }
import java . util . * ; class GFG {
static int missingnumber ( int n , int arr [ ] ) { int mn = Integer . MAX_VALUE , mx = Integer . MIN_VALUE ;
for ( int i = 0 ; i < n ; i ++ ) { if ( i > 0 && arr [ i ] == - 1 && arr [ i - 1 ] != - 1 ) { mn = Math . min ( mn , arr [ i - 1 ] ) ; mx = Math . max ( mx , arr [ i - 1 ] ) ; } if ( i < ( n - 1 ) && arr [ i ] == - 1 && arr [ i + 1 ] != - 1 ) { mn = Math . min ( mn , arr [ i + 1 ] ) ; mx = Math . max ( mx , arr [ i + 1 ] ) ; } } int res = ( mx + mn ) / 2 ; return res ; }
public static void main ( String [ ] args ) { int n = 5 ; int arr [ ] = { - 1 , 10 , - 1 , 12 , - 1 } ;
int res = missingnumber ( n , arr ) ; System . out . print ( res ) ; } }
class GFG {
static int lcsubtr ( char a [ ] , char b [ ] , int length1 , int length2 ) {
int dp [ ] [ ] = new int [ length1 + 1 ] [ length2 + 1 ] ; int max = 0 ;
for ( int i = 0 ; i <= length1 ; ++ i ) { for ( int j = 0 ; j <= length2 ; ++ j ) {
if ( i == 0 j == 0 ) { dp [ i ] [ j ] = 0 ; }
else if ( a [ i - 1 ] == b [ j - 1 ] ) { dp [ i ] [ j ] = dp [ i - 1 ] [ j - 1 ] + 1 ; max = Math . max ( dp [ i ] [ j ] , max ) ; }
else { dp [ i ] [ j ] = 0 ; } } }
return max ; }
public static void main ( String [ ] args ) { String m = "0110" ; String n = "1101" ; char m1 [ ] = m . toCharArray ( ) ; char m2 [ ] = n . toCharArray ( ) ;
System . out . println ( lcsubtr ( m1 , m2 , m1 . length , m2 . length ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int maxN = 20 ; static int maxSum = 50 ; static int minSum = 50 ; static int Base = 50 ;
static int [ ] [ ] dp = new int [ maxN ] [ maxSum + minSum ] ; static boolean [ ] [ ] v = new boolean [ maxN ] [ maxSum + minSum ] ;
static int findCnt ( int [ ] arr , int i , int required_sum , int n ) {
if ( i == n ) { if ( required_sum == 0 ) return 1 ; else return 0 ; }
if ( v [ i ] [ required_sum + Base ] ) return dp [ i ] [ required_sum + Base ] ;
v [ i ] [ required_sum + Base ] = true ;
dp [ i ] [ required_sum + Base ] = findCnt ( arr , i + 1 , required_sum , n ) + findCnt ( arr , i + 1 , required_sum - arr [ i ] , n ) ; return dp [ i ] [ required_sum + Base ] ; }
static void countSubsets ( int [ ] arr , int K , int n ) {
int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
int S1 = ( sum + K ) / 2 ;
System . out . print ( findCnt ( arr , 0 , S1 , n ) ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 1 , 2 , 3 } ; int N = arr . length ; int K = 1 ;
countSubsets ( arr , K , N ) ; } }
class GFG { static float [ ] [ ] dp = new float [ 105 ] [ 605 ] ;
static float find ( int N , int sum ) { if ( N < 0 sum < 0 ) return 0 ; if ( dp [ N ] [ sum ] > 0 ) return dp [ N ] [ sum ] ;
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return ( float ) ( 1.0 / 6 ) ; else return 0 ; } for ( int i = 1 ; i <= 6 ; i ++ ) dp [ N ] [ sum ] = dp [ N ] [ sum ] + find ( N - 1 , sum - i ) / 6 ; return dp [ N ] [ sum ] ; }
public static void main ( String [ ] args ) { int N = 4 , a = 13 , b = 17 ; float probability = 0.0f ;
for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
System . out . printf ( " % .6f " , probability ) ; } }
import java . util . HashMap ; class GFG {
static int count ( int n ) {
HashMap < Integer , Integer > dp = new HashMap < Integer , Integer > ( ) ;
dp . put ( 0 , 0 ) ; dp . put ( 1 , 1 ) ;
if ( ! dp . containsKey ( n ) ) dp . put ( n , 1 + Math . min ( n % 2 + count ( n / 2 ) , n % 3 + count ( n / 3 ) ) ) ;
return dp . get ( n ) ; }
public static void main ( String [ ] args ) {
int N = 6 ;
System . out . println ( String . valueOf ( ( count ( N ) ) ) ) ; } }
class GFG {
static void find_minimum_operations ( int n , int b [ ] , int k ) {
int d [ ] = new int [ n + 1 ] ;
int i , operations = 0 , need ; for ( i = 0 ; i < n ; i ++ ) {
if ( i > 0 ) { d [ i ] += d [ i - 1 ] ; }
if ( b [ i ] > d [ i ] ) {
operations += b [ i ] - d [ i ] ; need = b [ i ] - d [ i ] ;
d [ i ] += need ;
if ( i + k <= n ) { d [ i + k ] -= need ; } } } System . out . println ( operations ) ; }
public static void main ( String [ ] args ) { int n = 5 ; int b [ ] = { 1 , 2 , 3 , 4 , 5 } ; int k = 2 ;
find_minimum_operations ( n , b , k ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class GFG {
static int ways ( int [ ] [ ] arr , int K ) { int R = arr . length ; int C = arr [ 0 ] . length ; int [ ] [ ] preSum = new int [ R ] [ C ] ;
for ( int r = R - 1 ; r >= 0 ; r -- ) { for ( int c = C - 1 ; c >= 0 ; c -- ) { preSum [ r ] = arr [ r ] ; if ( r + 1 < R ) preSum [ r ] += preSum [ r + 1 ] ; if ( c + 1 < C ) preSum [ r ] += preSum [ r ] ; if ( r + 1 < R && c + 1 < C ) preSum [ r ] -= preSum [ r + 1 ] ; } }
int [ ] [ ] [ ] dp = new int [ K + 1 ] [ R ] [ C ] ;
for ( int k = 1 ; k <= K ; k ++ ) { for ( int r = R - 1 ; r >= 0 ; r -- ) { for ( int c = C - 1 ; c >= 0 ; c -- ) { if ( k == 1 ) { dp [ k ] [ r ] = ( preSum [ r ] > 0 ) ? 1 : 0 ; } else { dp [ k ] [ r ] = 0 ; for ( int r1 = r + 1 ; r1 < R ; r1 ++ ) {
if ( preSum [ r ] - preSum [ r1 ] > 0 ) dp [ k ] [ r ] += dp [ k - 1 ] [ r1 ] ; } for ( int c1 = c + 1 ; c1 < C ; c1 ++ ) {
if ( preSum [ r ] - preSum [ r ] [ c1 ] > 0 ) dp [ k ] [ r ] += dp [ k - 1 ] [ r ] [ c1 ] ; } } } } } return dp [ K ] [ 0 ] [ 0 ] ; }
public static void main ( String [ ] args ) { int [ ] [ ] arr = { { 1 , 0 , 0 } , { 1 , 1 , 1 } , { 0 , 0 , 0 } } ; int k = 3 ;
System . out . println ( ways ( arr , k ) ) ; } }
import java . util . * ; class GFG { static int p = 1000000007 ;
static int power ( int x , int y , int p ) { int res = 1 ; x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
static void nCr ( int n , int p , int f [ ] [ ] , int m ) { for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= m ; j ++ ) {
if ( j > i ) { f [ i ] [ j ] = 0 ; }
else if ( j == 0 j == i ) { f [ i ] [ j ] = 1 ; } else { f [ i ] [ j ] = ( f [ i - 1 ] [ j ] + f [ i - 1 ] [ j - 1 ] ) % p ; } } } }
static void ProductOfSubsets ( int arr [ ] , int n , int m ) { int [ ] [ ] f = new int [ n + 1 ] [ 100 ] ; nCr ( n , p - 1 , f , m ) ; Arrays . sort ( arr ) ;
long ans = 1 ; for ( int i = 0 ; i < n ; i ++ ) {
int x = 0 ; for ( int j = 1 ; j <= m ; j ++ ) {
if ( m % j == 0 ) {
x = ( x + ( f [ n - i - 1 ] [ m - j ] * f [ i ] [ j - 1 ] ) % ( p - 1 ) ) % ( p - 1 ) ; } } ans = ( ( ans * power ( arr [ i ] , x , p ) ) % p ) ; } System . out . print ( ans + "NEW_LINE"); }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 5 , 7 , 9 , 3 } ; int K = 4 ; int N = arr . length ; ProductOfSubsets ( arr , N , K ) ; } }
import java . util . * ; class GFG {
static int countWays ( int n , int m ) {
int [ ] [ ] dp = new int [ m + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { dp [ 1 ] [ i ] = 1 ; }
int sum ; for ( int i = 2 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) { sum = 0 ;
for ( int k = 0 ; k <= j ; k ++ ) { sum += dp [ i - 1 ] [ k ] ; }
dp [ i ] [ j ] = sum ; } }
return dp [ m ] [ n ] ; }
public static void main ( String [ ] args ) { int N = 2 , K = 3 ;
System . out . print ( countWays ( N , K ) ) ; } }
import java . util . * ; class GFG {
static int countWays ( int n , int m ) {
int [ ] [ ] dp = new int [ m + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { dp [ 1 ] [ i ] = 1 ; if ( i != 0 ) { dp [ 1 ] [ i ] += dp [ 1 ] [ i - 1 ] ; } }
for ( int i = 2 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) {
if ( j == 0 ) { dp [ i ] [ j ] = dp [ i - 1 ] [ j ] ; }
else { dp [ i ] [ j ] = dp [ i - 1 ] [ j ] ;
if ( i == m && j == n ) { return dp [ i ] [ j ] ; }
dp [ i ] [ j ] += dp [ i ] [ j - 1 ] ; } } } return Integer . MIN_VALUE ; }
public static void main ( String [ ] args ) { int N = 2 , K = 3 ;
System . out . print ( countWays ( N , K ) ) ; } }
import java . util . HashMap ; import java . util . Vector ; class GFG {
public static void SieveOfEratosthenes ( int MAX , Vector < Integer > primes ) { boolean [ ] prime = new boolean [ MAX + 1 ] ; for ( int i = 0 ; i < MAX + 1 ; i ++ ) prime [ i ] = true ;
for ( int p = 2 ; p * p <= MAX ; p ++ ) { if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( int i = 2 ; i <= MAX ; i ++ ) { if ( prime [ i ] ) primes . add ( i ) ; } }
public static int findLongest ( int [ ] A , int n ) {
HashMap < Integer , Integer > mpp = new HashMap < > ( ) ; Vector < Integer > primes = new Vector < > ( ) ;
SieveOfEratosthenes ( A [ n - 1 ] , primes ) ; int [ ] dp = new int [ n ] ;
dp [ n - 1 ] = 1 ; mpp . put ( A [ n - 1 ] , n - 1 ) ;
for ( int i = n - 2 ; i >= 0 ; i -- ) {
int num = A [ i ] ;
dp [ i ] = 1 ; int maxi = 0 ;
for ( int it : primes ) {
int xx = num * it ;
if ( xx > A [ n - 1 ] ) break ;
else if ( mpp . get ( xx ) != null && mpp . get ( xx ) != 0 ) {
dp [ i ] = Math . max ( dp [ i ] , 1 + dp [ mpp . get ( xx ) ] ) ; } }
mpp . put ( A [ i ] , i ) ; } int ans = 1 ;
for ( int i = 0 ; i < n ; i ++ ) ans = Math . max ( ans , dp [ i ] ) ; return ans ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 2 , 5 , 6 , 12 , 35 , 60 , 385 } ; int n = a . length ; System . out . println ( findLongest ( a , n ) ) ; } }
import java . util . * ; class solution {
static int waysToKAdjacentSetBits ( int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( lastBit == 0 ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
public static void main ( String args [ ] ) { int n = 5 , k = 2 ;
int totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; System . out . println ( " Number ▁ of ▁ ways ▁ = ▁ " + totalWays ) ; } }
import java . util . * ; class GFG {
static void postfix ( int a [ ] , int n ) { for ( int i = n - 1 ; i > 0 ; i -- ) { a [ i - 1 ] = a [ i - 1 ] + a [ i ] ; } }
static void modify ( int a [ ] , int n ) { for ( int i = 1 ; i < n ; i ++ ) { a [ i - 1 ] = i * a [ i ] ; } }
static void allCombination ( int a [ ] , int n ) { int sum = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { sum += i ; } System . out . println ( " f ( 1 ) ▁ - - > ▁ " + sum ) ;
for ( int i = 1 ; i < n ; i ++ ) {
postfix ( a , n - i + 1 ) ;
sum = 0 ; for ( int j = 1 ; j <= n - i ; j ++ ) { sum += ( j * a [ j ] ) ; } System . out . println ( " f ( " + ( i + 1 ) + " ) ▁ - - > ▁ " + sum ) ;
modify ( a , n ) ; } }
public static void main ( String [ ] args ) { int n = 5 ; int [ ] a = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) { a [ i ] = i + 1 ; }
allCombination ( a , n ) ; } }
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
import java . io . * ; class GFG {
public static boolean findPartiion ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; boolean [ ] part = new boolean [ sum / 2 + 1 ] ;
for ( i = 0 ; i <= sum / 2 ; i ++ ) { part [ i ] = false ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = sum / 2 ; j >= arr [ i ] ; j -- ) {
if ( part [ j - arr [ i ] ] == true j == arr [ i ] ) part [ j ] = true ; } } return part [ sum / 2 ] ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 3 , 2 , 3 , 2 } ; int n = 6 ;
if ( findPartiion ( arr , n ) == true ) System . out . println ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ " + " subsets ▁ of ▁ equal ▁ sum " ) ; else System . out . println ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ " + " two ▁ subsets ▁ of ▁ equal ▁ sum " ) ; } }
import java . util . * ; class GFG {
static int binomialCoeff ( int n , int r ) { if ( r > n ) return 0 ; long m = 1000000007 ; long inv [ ] = new long [ r + 1 ] ; inv [ 0 ] = 1 ; if ( r + 1 >= 2 ) inv [ 1 ] = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { inv [ i ] = m - ( m / i ) * inv [ ( int ) ( m % i ) ] % m ; } int ans = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { ans = ( int ) ( ( ( ans % m ) * ( inv [ i ] % m ) ) % m ) ; }
for ( int i = n ; i >= ( n - r + 1 ) ; i -- ) { ans = ( int ) ( ( ( ans % m ) * ( i % m ) ) % m ) ; } return ans ; }
public static void main ( String [ ] args ) { int n = 5 , r = 2 ; System . out . print ( " Value ▁ of ▁ C ( " + n + " , ▁ " + r + " ) ▁ is ▁ " + binomialCoeff ( n , r ) + "NEW_LINE"); } }
import java . io . * ; class GFG {
public static int gcd ( int a , int b ) {
if ( a < b ) { int t = a ; a = b ; b = t ; } if ( a % b == 0 ) return b ;
return gcd ( b , a % b ) ; }
static void printAnswer ( int x , int y ) {
int val = gcd ( x , y ) ;
if ( ( val & ( val - 1 ) ) == 0 ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; }
public static void main ( String [ ] args ) {
int x = 4 ; int y = 7 ;
printAnswer ( x , y ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int getElement ( int N , int r , int c ) {
if ( r > c ) return 0 ;
if ( r == 1 ) { return c ; }
int a = ( r + 1 ) * ( int ) ( Math . pow ( 2 , ( r - 2 ) ) ) ;
int d = ( int ) ( Math . pow ( 2 , ( r - 1 ) ) ) ;
c = c - r ; int element = a + d * c ; return element ; }
public static void main ( String [ ] args ) { int N = 4 , R = 3 , C = 4 ; System . out . println ( getElement ( N , R , C ) ) ; } }
import java . io . * ; import java . lang . * ; import java . util . * ; public class GFG {
static String MinValue ( String number , int x ) {
int length = number . length ( ) ;
int position = length + 1 ;
if ( number . charAt ( 0 ) == ' - ' ) {
for ( int i = number . length ( ) - 1 ; i >= 1 ; -- i ) { if ( ( number . charAt ( i ) - 48 ) < x ) { position = i ; } } } else {
for ( int i = number . length ( ) - 1 ; i >= 0 ; -- i ) { if ( ( number . charAt ( i ) - 48 ) > x ) { position = i ; } } }
number = number . substring ( 0 , position ) + x + number . substring ( position , number . length ( ) ) ;
return number . toString ( ) ; }
public static void main ( String [ ] args ) {
String number = "89" ; int x = 1 ;
System . out . print ( MinValue ( number , x ) ) ; } }
class GFG {
public static String divisibleByk ( String s , int n , int k ) {
int [ ] poweroftwo = new int [ n ] ;
poweroftwo [ 0 ] = 1 % k ; for ( int i = 1 ; i < n ; i ++ ) {
poweroftwo [ i ] = ( poweroftwo [ i - 1 ] * ( 2 % k ) ) % k ; }
int rem = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( s . charAt ( n - i - 1 ) == '1' ) {
rem += ( poweroftwo [ i ] ) ; rem %= k ; } }
if ( rem == 0 ) { return " Yes " ; }
else return " No " ; }
public static void main ( String args [ ] ) {
String s = "1010001" ; int k = 9 ;
int n = s . length ( ) ;
System . out . println ( divisibleByk ( s , n , k ) ) ; } }
import java . util . * ; class GFG {
static int maxSumbySplittingString ( String str , int N ) {
int cntOne = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( str . charAt ( i ) == '1' ) {
cntOne ++ ; } }
int zero = 0 ;
int one = 0 ;
int res = 0 ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( str . charAt ( i ) == '0' ) {
zero ++ ; }
else {
one ++ ; }
res = Math . max ( res , zero + cntOne - one ) ; } return res ; }
public static void main ( String [ ] args ) { String str = "00111" ; int N = str . length ( ) ; System . out . print ( maxSumbySplittingString ( str , N ) ) ; } }
import java . io . * ; class GFG {
static void cntBalancedParenthesis ( String s , int N ) {
int cntPairs = 0 ;
int cntCurly = 0 ;
int cntSml = 0 ;
int cntSqr = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( s . charAt ( i ) == ' { ' ) {
cntCurly ++ ; } else if ( s . charAt ( i ) == ' ( ' ) {
cntSml ++ ; } else if ( s . charAt ( i ) == ' [ ' ) {
cntSqr ++ ; } else if ( s . charAt ( i ) == ' } ' && cntCurly > 0 ) {
cntCurly -- ;
cntPairs ++ ; } else if ( s . charAt ( i ) == ' ) ' && cntSml > 0 ) {
cntSml -- ;
cntPairs ++ ; } else if ( s . charAt ( i ) == ' ] ' && cntSqr > 0 ) {
cntSqr -- ;
cntPairs ++ ; } } System . out . println ( cntPairs ) ; }
public static void main ( String [ ] args ) {
String s = " { ( } ) " ; int N = s . length ( ) ;
cntBalancedParenthesis ( s , N ) ; } }
import java . util . * ; class GFG {
static int arcIntersection ( String S , int len ) { Stack < Character > stk = new Stack < > ( ) ;
for ( int i = 0 ; i < len ; i ++ ) {
stk . push ( S . charAt ( i ) ) ; if ( stk . size ( ) >= 2 ) {
char temp = stk . peek ( ) ;
stk . pop ( ) ;
if ( stk . peek ( ) == temp ) { stk . pop ( ) ; }
else { stk . add ( temp ) ; } } }
if ( stk . isEmpty ( ) ) return 1 ; return 0 ; }
static void countString ( String arr [ ] , int N ) {
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int len = arr [ i ] . length ( ) ;
count += arcIntersection ( arr [ i ] , len ) ; }
System . out . print ( count + "NEW_LINE"); }
public static void main ( String [ ] args ) { String arr [ ] = { "0101" , "0011" , "0110" } ; int N = arr . length ;
countString ( arr , N ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class GFG {
static String ConvertequivalentBase8 ( String S ) {
HashMap < String , Character > mp = new HashMap < String , Character > ( ) ;
mp . put ( "000" , '0' ) ; mp . put ( "001" , '1' ) ; mp . put ( "010" , '2' ) ; mp . put ( "011" , '3' ) ; mp . put ( "100" , '4' ) ; mp . put ( "101" , '5' ) ; mp . put ( "110" , '6' ) ; mp . put ( "111" , '7' ) ;
int N = S . length ( ) ; if ( N % 3 == 2 ) {
S = "0" + S ; } else if ( N % 3 == 1 ) {
S = "00" + S ; }
N = S . length ( ) ;
String oct = " " ;
for ( int i = 0 ; i < N ; i += 3 ) {
String temp = S . substring ( i , i + 3 ) ;
oct += mp . get ( temp ) ; } return oct ; }
static String binString_div_9 ( String S , int N ) {
String oct = " " ; oct = ConvertequivalentBase8 ( S ) ;
int oddSum = 0 ;
int evenSum = 0 ;
int M = oct . length ( ) ;
for ( int i = 0 ; i < M ; i += 2 )
oddSum += ( oct . charAt ( i ) - '0' ) ;
for ( int i = 1 ; i < M ; i += 2 ) {
evenSum += ( oct . charAt ( i ) - '0' ) ; }
int Oct_9 = 11 ;
if ( Math . abs ( oddSum - evenSum ) % Oct_9 == 0 ) { return " Yes " ; } return " No " ; }
public static void main ( String [ ] args ) { String S = "1010001" ; int N = S . length ( ) ; System . out . println ( binString_div_9 ( S , N ) ) ; } }
import java . util . * ; import java . lang . * ; class GFG {
static int min_cost ( String S ) {
int cost = 0 ;
int F = 0 ;
int B = 0 ; int count = 0 ; for ( char c : S . toCharArray ( ) ) if ( c == ' ▁ ' ) count ++ ;
int n = S . length ( ) - count ;
if ( n == 1 ) return cost ;
for ( char in : S . toCharArray ( ) ) {
if ( in != ' ▁ ' ) {
if ( B != 0 ) {
cost += Math . min ( n - F , F ) * B ; B = 0 ; }
F += 1 ; }
else {
B += 1 ; } }
return cost ; }
public static void main ( String [ ] args ) { String S = " ▁ @ $ " ; System . out . println ( min_cost ( S ) ) ; } }
import java . util . * ; class GFG {
static boolean isVowel ( char ch ) { if ( ch == ' a ' ch == ' e ' ch == ' i ' ch == ' o ' ch == ' u ' ) return true ; else return false ; }
static int minCost ( String S ) {
int cA = 0 ; int cE = 0 ; int cI = 0 ; int cO = 0 ; int cU = 0 ;
for ( int i = 0 ; i < S . length ( ) ; i ++ ) {
if ( isVowel ( S . charAt ( i ) ) ) {
cA += Math . abs ( S . charAt ( i ) - ' a ' ) ; cE += Math . abs ( S . charAt ( i ) - ' e ' ) ; cI += Math . abs ( S . charAt ( i ) - ' i ' ) ; cO += Math . abs ( S . charAt ( i ) - ' o ' ) ; cU += Math . abs ( S . charAt ( i ) - ' u ' ) ; } }
return Math . min ( Math . min ( Math . min ( Math . min ( cA , cE ) , cI ) , cO ) , cU ) ; }
public static void main ( String [ ] args ) { String S = " geeksforgeeks " ; System . out . println ( minCost ( S ) ) ; } }
class GFG {
public static void decode_String ( String str , int K ) { String ans = " " ;
for ( int i = 0 ; i < str . length ( ) ; i += K )
ans += str . charAt ( i ) ;
for ( int i = str . length ( ) - ( K - 1 ) ; i < str . length ( ) ; i ++ ) ans += str . charAt ( i ) ; System . out . println ( ans ) ; }
public static void main ( String [ ] args ) { int K = 3 ; String str = " abcbcscsesesesd " ; decode_String ( str , K ) ; } }
class GFG {
static String maxVowelSubString ( String str , int K ) {
int N = str . length ( ) ;
int [ ] pref = new int [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( str . charAt ( i ) == ' a ' || str . charAt ( i ) == ' e ' || str . charAt ( i ) == ' i ' || str . charAt ( i ) == ' o ' || str . charAt ( i ) == ' u ' ) pref [ i ] = 1 ;
else pref [ i ] = 0 ;
if ( i != 0 ) pref [ i ] += pref [ i - 1 ] ; }
int maxCount = pref [ K - 1 ] ;
String res = str . substring ( 0 , K ) ;
for ( int i = K ; i < N ; i ++ ) {
int currCount = pref [ i ] - pref [ i - K ] ;
if ( currCount > maxCount ) { maxCount = currCount ; res = str . substring ( i - K + 1 , i + 1 ) ; }
else if ( currCount == maxCount ) { String temp = str . substring ( i - K + 1 , i + 1 ) ; if ( temp . compareTo ( res ) < 0 ) res = temp ; } }
return res ; }
public static void main ( String [ ] args ) { String str = " ceebbaceeffo " ; int K = 3 ; System . out . print ( maxVowelSubString ( str , K ) ) ; } }
class GFG {
static void decodeStr ( String str , int len ) {
char [ ] c = new char [ len ] ; int med , pos = 1 , k ;
if ( len % 2 == 1 ) med = len / 2 ; else med = len / 2 - 1 ;
c [ med ] = str . charAt ( 0 ) ;
if ( len % 2 == 0 ) c [ med + 1 ] = str . charAt ( 1 ) ;
if ( len % 2 == 1 ) k = 1 ; else k = 2 ; for ( int i = k ; i < len ; i += 2 ) { c [ med - pos ] = str . charAt ( i ) ;
if ( len % 2 == 1 ) c [ med + pos ] = str . charAt ( i + 1 ) ;
else c [ med + pos + 1 ] = str . charAt ( i + 1 ) ; pos ++ ; }
for ( int i = 0 ; i < len ; i ++ ) System . out . print ( c [ i ] ) ; }
public static void main ( String [ ] args ) { String str = " ofrsgkeeeekgs " ; int len = str . length ( ) ; decodeStr ( str , len ) ; } }
class GFG { static void findCount ( String s , int L , int R ) {
int distinct = 0 ;
int [ ] frequency = new int [ 26 ] ;
for ( int i = L ; i <= R ; i ++ ) {
frequency [ s . charAt ( i ) - ' a ' ] ++ ; } for ( int i = 0 ; i < 26 ; i ++ ) {
if ( frequency [ i ] > 0 ) distinct ++ ; } System . out . print ( distinct + "NEW_LINE"); }
public static void main ( String [ ] args ) { String s = " geeksforgeeksisa " + " computerscienceportal " ; int queries = 3 ; int Q [ ] [ ] = { { 0 , 10 } , { 15 , 18 } , { 12 , 20 } } ; for ( int i = 0 ; i < queries ; i ++ ) findCount ( s , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) ; } }
class GFG {
static String ReverseComplement ( char [ ] s , int n , int k ) {
int rev = ( k + 1 ) / 2 ;
int complement = k - rev ;
if ( rev % 2 == 1 ) s = reverse ( s ) ;
if ( complement % 2 == 1 ) { for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == '0' ) s [ i ] = '1' ; else s [ i ] = '0' ; } }
return String . valueOf ( s ) ; } static char [ ] reverse ( char a [ ] ) { int i , n = a . length ; char t ; for ( i = 0 ; i < n / 2 ; i ++ ) { t = a [ i ] ; a [ i ] = a [ n - i - 1 ] ; a [ n - i - 1 ] = t ; } return a ; }
public static void main ( String [ ] args ) { String str = "10011" ; int k = 5 ; int n = str . length ( ) ;
System . out . print ( ReverseComplement ( str . toCharArray ( ) , n , k ) ) ; } }
class GFG {
static boolean repeatingString ( String s , int n , int k ) {
if ( n % k != 0 ) { return false ; }
int [ ] frequency = new int [ 123 ] ;
for ( int i = 0 ; i < 123 ; i ++ ) { frequency [ i ] = 0 ; }
for ( int i = 0 ; i < n ; i ++ ) { frequency [ s . charAt ( i ) ] ++ ; } int repeat = n / k ;
for ( int i = 0 ; i < 123 ; i ++ ) { if ( frequency [ i ] % repeat != 0 ) { return false ; } } return true ; }
public static void main ( String [ ] args ) { String s = " abcdcba " ; int n = s . length ( ) ; int k = 3 ; if ( repeatingString ( s , n , k ) ) { System . out . print ( " Yes " + "NEW_LINE"); } else { System . out . print ( " No " + "NEW_LINE"); } } }
class GFG {
static void findPhoneNumber ( int n ) { int temp = n ; int sum = 0 ;
while ( temp != 0 ) { sum += temp % 10 ; temp = temp / 10 ; }
if ( sum < 10 ) System . out . print ( n + "0" + sum ) ;
else System . out . print ( n + " " + sum ) ; }
public static void main ( String [ ] args ) { int n = 98765432 ; findPhoneNumber ( n ) ; } }
class GFG { static int maxN = 20 ; static int maxM = 64 ;
static int cntSplits ( String s ) {
if ( s . charAt ( s . length ( ) - 1 ) == '1' ) return 0 ;
int c_zero = 0 ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) c_zero += ( s . charAt ( i ) == '0' ) ? 1 : 0 ;
return ( int ) Math . pow ( 2 , c_zero - 1 ) ; }
public static void main ( String [ ] args ) { String s = "10010" ; System . out . println ( cntSplits ( s ) ) ; } }
class GFG {
static void findNumbers ( String s ) {
int n = s . length ( ) ;
int count = 1 ; int result = 0 ;
int left = 0 ; int right = 1 ; while ( right < n ) {
if ( s . charAt ( left ) == s . charAt ( right ) ) { count ++ ; }
else {
result += count * ( count + 1 ) / 2 ;
left = right ; count = 1 ; } right ++ ; }
result += count * ( count + 1 ) / 2 ; System . out . println ( result ) ; }
public static void main ( String [ ] args ) { String s = " bbbcbb " ; findNumbers ( s ) ; } }
import java . util . * ; class GFG {
static boolean isVowel ( char ch ) { ch = Character . toUpperCase ( ch ) ; return ( ch == ' A ' ch == ' E ' ch == ' I ' ch == ' O ' ch == ' U ' ) ; }
static String duplicateVowels ( String str ) { int t = str . length ( ) ;
String res = " " ;
for ( int i = 0 ; i < t ; i ++ ) { if ( isVowel ( str . charAt ( i ) ) ) res += str . charAt ( i ) ; res += str . charAt ( i ) ; } return res ; }
public static void main ( String [ ] args ) { String str = " helloworld " ;
System . out . println ( " Original ▁ String : ▁ " + str ) ; String res = duplicateVowels ( str ) ;
System . out . println ( " String ▁ with ▁ Vowels ▁ duplicated : ▁ " + res ) ; } }
public class GFG {
static int stringToInt ( String str ) {
if ( str . length ( ) == 1 ) return ( str . charAt ( 0 ) - '0' ) ;
double y = stringToInt ( str . substring ( 1 ) ) ;
double x = str . charAt ( 0 ) - '0' ;
x = x * Math . pow ( 10 , str . length ( ) - 1 ) + y ; return ( int ) ( x ) ; }
public static void main ( String [ ] args ) { String str = "1235" ; System . out . print ( stringToInt ( str ) ) ; } }
class GFG { static int MAX = 26 ;
static int largestSubSeq ( String arr [ ] , int n ) {
int [ ] count = new int [ MAX ] ;
for ( int i = 0 ; i < n ; i ++ ) { String str = arr [ i ] ;
boolean [ ] hash = new boolean [ MAX ] ; for ( int j = 0 ; j < str . length ( ) ; j ++ ) { hash [ str . charAt ( j ) - ' a ' ] = true ; } for ( int j = 0 ; j < MAX ; j ++ ) {
if ( hash [ j ] ) count [ j ] ++ ; } } int max = - 1 ; for ( int i = 0 ; i < MAX ; i ++ ) { if ( max < count [ i ] ) max = count [ i ] ; } return max ; }
public static void main ( String [ ] args ) { String arr [ ] = { " ab " , " bc " , " de " } ; int n = arr . length ; System . out . println ( largestSubSeq ( arr , n ) ) ; } }
class GFG {
static boolean isPalindrome ( String str ) { int len = str . length ( ) ; for ( int i = 0 ; i < len / 2 ; i ++ ) { if ( str . charAt ( i ) != str . charAt ( len - 1 - i ) ) return false ; } return true ; }
static boolean createStringAndCheckPalindrome ( int N ) {
String sub = " " + N , res_str = " " ; int sum = 0 ;
while ( N > 0 ) { int digit = N % 10 ; sum += digit ; N = N / 10 ; }
while ( res_str . length ( ) < sum ) res_str += sub ;
if ( res_str . length ( ) > sum ) res_str = res_str . substring ( 0 , sum ) ;
if ( isPalindrome ( res_str ) ) return true ; return false ; }
public static void main ( String args [ ] ) { int N = 10101 ; if ( createStringAndCheckPalindrome ( N ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . io . * ; class GFG {
static int minimumLength ( String s ) { int maxOcc = 0 , n = s . length ( ) ; int arr [ ] = new int [ 26 ] ;
for ( int i = 0 ; i < n ; i ++ ) arr [ s . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < 26 ; i ++ ) if ( arr [ i ] > maxOcc ) maxOcc = arr [ i ] ;
return ( n - maxOcc ) ; }
public static void main ( String [ ] args ) { String str = " afddewqd " ; System . out . println ( minimumLength ( str ) ) ; } }
class GFG {
static void removeSpecialCharacter ( String s ) { for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( s . charAt ( i ) < ' A ' || s . charAt ( i ) > ' Z ' && s . charAt ( i ) < ' a ' || s . charAt ( i ) > ' z ' ) {
s = s . substring ( 0 , i ) + s . substring ( i + 1 ) ; i -- ; } } System . out . print ( s ) ; }
public static void main ( String [ ] args ) { String s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " ; removeSpecialCharacter ( s ) ; } }
class GFG {
static void removeSpecialCharacter ( String str ) { char [ ] s = str . toCharArray ( ) ; int j = 0 ; for ( int i = 0 ; i < s . length ; i ++ ) {
if ( ( s [ i ] >= ' A ' && s [ i ] <= ' Z ' ) || ( s [ i ] >= ' a ' && s [ i ] <= ' z ' ) ) { s [ j ] = s [ i ] ; j ++ ; } } System . out . println ( String . valueOf ( s ) . substring ( 0 , j ) ) ; }
public static void main ( String [ ] args ) { String s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " ; removeSpecialCharacter ( s ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int findRepeatFirstN2 ( String s ) {
int p = - 1 , i , j ; for ( i = 0 ; i < s . length ( ) ; i ++ ) { for ( j = i + 1 ; j < s . length ( ) ; j ++ ) { if ( s . charAt ( i ) == s . charAt ( j ) ) { p = i ; break ; } } if ( p != - 1 ) break ; } return p ; }
static public void main ( String [ ] args ) { String str = " geeksforgeeks " ; int pos = findRepeatFirstN2 ( str ) ; if ( pos == - 1 ) System . out . println ( " Not ▁ found " ) ; else System . out . println ( str . charAt ( pos ) ) ; } }
import java . util . * ; class Gfg { public static void prCharWithFreq ( String s ) {
Map < Character , Integer > d = new HashMap < Character , Integer > ( ) ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( d . containsKey ( s . charAt ( i ) ) ) { d . put ( s . charAt ( i ) , d . get ( s . charAt ( i ) ) + 1 ) ; } else { d . put ( s . charAt ( i ) , 1 ) ; } }
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( d . get ( s . charAt ( i ) ) != 0 ) { System . out . print ( s . charAt ( i ) ) ; System . out . print ( d . get ( s . charAt ( i ) ) + " ▁ " ) ; d . put ( s . charAt ( i ) , 0 ) ; } } }
public static void main ( String [ ] args ) { String S = " geeksforgeeks " ; prCharWithFreq ( S ) ; } }
class GFG {
static int possibleStrings ( int n , int r , int b , int g ) {
int fact [ ] = new int [ n + 1 ] ; fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
int left = n - ( r + g + b ) ; int sum = 0 ;
for ( int i = 0 ; i <= left ; i ++ ) { for ( int j = 0 ; j <= left - i ; j ++ ) { int k = left - ( i + j ) ;
sum = sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ; } }
return sum ; }
public static void main ( String [ ] args ) { int n = 4 , r = 2 ; int b = 0 , g = 1 ; System . out . println ( possibleStrings ( n , r , b , g ) ) ; } }
import java . util . * ; class GFG {
static int remAnagram ( String str1 , String str2 ) {
int count1 [ ] = new int [ 26 ] ; int count2 [ ] = new int [ 26 ] ;
for ( int i = 0 ; i < str1 . length ( ) ; i ++ ) count1 [ str1 . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < str2 . length ( ) ; i ++ ) count2 [ str2 . charAt ( i ) - ' a ' ] ++ ;
int result = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) result += Math . abs ( count1 [ i ] - count2 [ i ] ) ; return result ; }
public static void main ( String [ ] args ) { String str1 = " bcadeh " , str2 = " hea " ; System . out . println ( remAnagram ( str1 , str2 ) ) ; } }
public class GFG {
static int CHARS = 26 ;
static boolean isValidString ( String str ) { int freq [ ] = new int [ CHARS ] ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { freq [ str . charAt ( i ) - ' a ' ] ++ ; }
int i , freq1 = 0 , count_freq1 = 0 ; for ( i = 0 ; i < CHARS ; i ++ ) { if ( freq [ i ] != 0 ) { freq1 = freq [ i ] ; count_freq1 = 1 ; break ; } }
int j , freq2 = 0 , count_freq2 = 0 ; for ( j = i + 1 ; j < CHARS ; j ++ ) { if ( freq [ j ] != 0 ) { if ( freq [ j ] == freq1 ) { count_freq1 ++ ; } else { count_freq2 = 1 ; freq2 = freq [ j ] ; break ; } } }
for ( int k = j + 1 ; k < CHARS ; k ++ ) { if ( freq [ k ] != 0 ) { if ( freq [ k ] == freq1 ) { count_freq1 ++ ; } if ( freq [ k ] == freq2 ) { count_freq2 ++ ;
{ return false ; } }
if ( count_freq1 > 1 && count_freq2 > 1 ) { return false ; } }
return true ; }
public static void main ( String [ ] args ) { String str = " abcbc " ; if ( isValidString ( str ) ) { System . out . println ( " YES " ) ; } else { System . out . println ( " NO " ) ; } } }
import java . util . HashMap ; import java . util . Iterator ; import java . util . Map ; public class AllCharsWithSameFrequencyWithOneVarAllowed {
public static boolean checkForVariation ( String str ) { if ( str == null || str . isEmpty ( ) ) { return true ; } Map < Character , Integer > map = new HashMap < > ( ) ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { map . put ( str . charAt ( i ) , map . getOrDefault ( str . charAt ( i ) , 0 ) + 1 ) ; } Iterator < Integer > itr = map . values ( ) . iterator ( ) ;
boolean first = true , second = true ; int val1 = 0 , val2 = 0 ; int countOfVal1 = 0 , countOfVal2 = 0 ; while ( itr . hasNext ( ) ) { int i = itr . next ( ) ;
if ( first ) { val1 = i ; first = false ; countOfVal1 ++ ; continue ; } if ( i == val1 ) { countOfVal1 ++ ; continue ; }
if ( second ) { val2 = i ; countOfVal2 ++ ; second = false ; continue ; } if ( i == val2 ) { countOfVal2 ++ ; continue ; } return false ; } if ( countOfVal1 > 1 && countOfVal2 > 1 ) { return false ; } else { return true ; } }
public static void main ( String [ ] args ) { System . out . println ( checkForVariation ( " abcbc " ) ) ; } }
class GFG {
static int countCompletePairs ( String set1 [ ] , String set2 [ ] , int n , int m ) { int result = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
String concat = set1 [ i ] + set2 [ j ] ;
int frequency [ ] = new int [ 26 ] ; for ( int k = 0 ; k < concat . length ( ) ; k ++ ) { frequency [ concat . charAt ( k ) - ' a ' ] ++ ; }
int k ; for ( k = 0 ; k < 26 ; k ++ ) { if ( frequency [ k ] < 1 ) { break ; } } if ( k == 26 ) { result ++ ; } } } return result ; }
static public void main ( String [ ] args ) { String set1 [ ] = { " abcdefgh " , " geeksforgeeks " , " lmnopqrst " , " abc " } ; String set2 [ ] = { " ijklmnopqrstuvwxyz " , " abcdefghijklmnopqrstuvwxyz " , " defghijklmnopqrstuvwxyz " } ; int n = set1 . length ; int m = set2 . length ; System . out . println ( countCompletePairs ( set1 , set2 , n , m ) ) ; } }
class GFG {
static int countCompletePairs ( String set1 [ ] , String set2 [ ] , int n , int m ) { int result = 0 ;
int [ ] con_s1 = new int [ n ] ; int [ ] con_s2 = new int [ m ] ;
for ( int i = 0 ; i < n ; i ++ ) {
con_s1 [ i ] = 0 ; for ( int j = 0 ; j < set1 [ i ] . length ( ) ; j ++ ) {
con_s1 [ i ] = con_s1 [ i ] | ( 1 << ( set1 [ i ] . charAt ( j ) - ' a ' ) ) ; } }
for ( int i = 0 ; i < m ; i ++ ) {
con_s2 [ i ] = 0 ; for ( int j = 0 ; j < set2 [ i ] . length ( ) ; j ++ ) {
con_s2 [ i ] = con_s2 [ i ] | ( 1 << ( set2 [ i ] . charAt ( j ) - ' a ' ) ) ; } }
long complete = ( 1 << 26 ) - 1 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
if ( ( con_s1 [ i ] con_s2 [ j ] ) == complete ) { result ++ ; } } } return result ; }
public static void main ( String args [ ] ) { String set1 [ ] = { " abcdefgh " , " geeksforgeeks " , " lmnopqrst " , " abc " } ; String set2 [ ] = { " ijklmnopqrstuvwxyz " , " abcdefghijklmnopqrstuvwxyz " , " defghijklmnopqrstuvwxyz " } ; int n = set1 . length ; int m = set2 . length ; System . out . println ( countCompletePairs ( set1 , set2 , n , m ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static String encodeString ( String str ) { HashMap < Character , Integer > map = new HashMap < > ( ) ; String res = " " ; int i = 0 ;
char ch ; for ( int j = 0 ; j < str . length ( ) ; j ++ ) { ch = str . charAt ( j ) ;
if ( ! map . containsKey ( ch ) ) map . put ( ch , i ++ ) ;
res += map . get ( ch ) ; } return res ; }
static void findMatchedWords ( String [ ] dict , String pattern ) {
int len = pattern . length ( ) ;
String hash = encodeString ( pattern ) ;
for ( String word : dict ) {
if ( word . length ( ) == len && encodeString ( word ) . equals ( hash ) ) System . out . print ( word + " ▁ " ) ; } }
public static void main ( String args [ ] ) { String [ ] dict = { " abb " , " abc " , " xyz " , " xyy " } ; String pattern = " foo " ; findMatchedWords ( dict , pattern ) ; } }
import java . util . * ; class GFG { static boolean check ( String pattern , String word ) { if ( pattern . length ( ) != word . length ( ) ) return false ; int [ ] ch = new int [ 128 ] ; int Len = word . length ( ) ; for ( int i = 0 ; i < Len ; i ++ ) { if ( ch [ ( int ) pattern . charAt ( i ) ] == 0 ) { ch [ ( int ) pattern . charAt ( i ) ] = word . charAt ( i ) ; } else if ( ch [ ( int ) pattern . charAt ( i ) ] != word . charAt ( i ) ) { return false ; } } return true ; }
static void findMatchedWords ( HashSet < String > dict , String pattern ) {
int Len = pattern . length ( ) ;
String result = " ▁ " ; for ( String word : dict ) { if ( check ( pattern , word ) ) { result = word + " ▁ " + result ; } } System . out . print ( result ) ; }
public static void main ( String [ ] args ) { HashSet < String > dict = new HashSet < String > ( ) ; dict . add ( " abb " ) ; dict . add ( " abc " ) ; dict . add ( " xyz " ) ; dict . add ( " xyy " ) ; String pattern = " foo " ; findMatchedWords ( dict , pattern ) ; } }
class GFG {
public static int countWords ( String str ) {
if ( str == null || str . isEmpty ( ) ) return 0 ; int wordCount = 0 ; boolean isWord = false ; int endOfLine = str . length ( ) - 1 ;
char [ ] ch = str . toCharArray ( ) ; for ( int i = 0 ; i < ch . length ; i ++ ) {
if ( Character . isLetter ( ch [ i ] ) && i != endOfLine ) isWord = true ;
else if ( ! Character . isLetter ( ch [ i ] ) && isWord ) { wordCount ++ ; isWord = false ; }
else if ( Character . isLetter ( ch [ i ] ) && i == endOfLine ) wordCount ++ ; }
return wordCount ; }
public static void main ( String args [ ] ) {
String str = "One twothree four five ";
System . out . println ( " No ▁ of ▁ words ▁ : ▁ " + countWords ( str ) ) ; } }
class GFG {
public static String [ ] RevString ( String [ ] s , int l ) {
if ( l % 2 == 0 ) {
int j = l / 2 ;
while ( j <= l - 1 ) { String temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } }
else {
int j = ( l / 2 ) + 1 ;
while ( j <= l - 1 ) { String temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } }
return s ; }
public static void main ( String [ ] args ) { String s = " getting ▁ good ▁ at ▁ coding ▁ " + " needs ▁ a ▁ lot ▁ of ▁ practice " ; String [ ] words = s . split ( " \\ s " ) ; words = RevString ( words , words . length ) ; s = String . join ( " ▁ " , words ) ; System . out . println ( s ) ; } }
import java . util . * ; class GFG {
static void printPath ( Vector < Integer > res , int nThNode , int kThNode ) {
if ( kThNode > nThNode ) return ;
res . add ( kThNode ) ;
for ( int i = 0 ; i < res . size ( ) ; i ++ ) System . out . print ( res . get ( i ) + " ▁ " ) ; System . out . print ( "NEW_LINE");
printPath ( res , nThNode , kThNode * 2 ) ;
printPath ( res , nThNode , kThNode * 2 + 1 ) ; res . remove ( res . size ( ) - 1 ) ; }
static void printPathToCoverAllNodeUtil ( int nThNode ) {
Vector < Integer > res = new Vector < Integer > ( ) ;
printPath ( res , nThNode , 1 ) ; }
public static void main ( String args [ ] ) {
int nThNode = 7 ;
printPathToCoverAllNodeUtil ( nThNode ) ; } }
import java . util . * ; class GFG {
static int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
static boolean isArmstrong ( int x ) { int n = String . valueOf ( x ) . length ( ) ; int sum1 = 0 ; int temp = x ; while ( temp > 0 ) { int digit = temp % 10 ; sum1 += Math . pow ( digit , n ) ; temp /= 10 ; } if ( sum1 == x ) return true ; return false ; }
static int MaxUtil ( int [ ] st , int ss , int se , int l , int r , int node ) {
if ( l <= ss && r >= se ) return st [ node ] ;
if ( se < l ss > r ) return - 1 ;
int mid = getMid ( ss , se ) ; return Math . max ( MaxUtil ( st , ss , mid , l , r , 2 * node ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 1 ) ) ; }
static void updateValue ( int arr [ ] , int [ ] st , int ss , int se , int index , int value , int node ) { if ( index < ss index > se ) { System . out . print ( " Invalid ▁ Input " + "NEW_LINE"); return ; } if ( ss == se ) {
arr [ index ] = value ; if ( isArmstrong ( value ) ) st [ node ] = value ; else st [ node ] = - 1 ; } else { int mid = getMid ( ss , se ) ; if ( index >= ss && index <= mid ) updateValue ( arr , st , ss , mid , index , value , 2 * node ) ; else updateValue ( arr , st , mid + 1 , se , index , value , 2 * node + 1 ) ; st [ node ] = Math . max ( st [ 2 * node + 1 ] , st [ 2 * node + 2 ] ) ; } return ; }
static int getMax ( int [ ] st , int n , int l , int r ) {
if ( l < 0 r > n - 1 l > r ) { System . out . printf ( " Invalid ▁ Input " ) ; return - 1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
static int constructSTUtil ( int arr [ ] , int ss , int se , int [ ] st , int si ) {
if ( ss == se ) { if ( isArmstrong ( arr [ ss ] ) ) st [ si ] = arr [ ss ] ; else st [ si ] = - 1 ; return st [ si ] ; }
int mid = getMid ( ss , se ) ; st [ si ] = Math . max ( constructSTUtil ( arr , ss , mid , st , si * 2 ) , constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 1 ) ) ; return st [ si ] ; }
static int [ ] constructST ( int arr [ ] , int n ) {
int x = ( int ) ( Math . ceil ( Math . log ( n ) ) ) ;
int max_size = 2 * ( int ) Math . pow ( 2 , x ) - 1 ;
int [ ] st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
public static void main ( String [ ] args ) { int arr [ ] = { 192 , 113 , 535 , 7 , 19 , 111 } ; int n = arr . length ;
int [ ] st = constructST ( arr , n ) ;
System . out . print ( " Maximum ▁ armstrong ▁ " + " number ▁ in ▁ given ▁ range ▁ = ▁ " + getMax ( st , n , 1 , 3 ) + "NEW_LINE");
updateValue ( arr , st , 0 , n - 1 , 1 , 153 , 0 ) ;
System . out . print ( " Updated ▁ Maximum ▁ armstrong ▁ " + " number ▁ in ▁ given ▁ range ▁ = ▁ " + getMax ( st , n , 1 , 3 ) + "NEW_LINE"); } }
class GFG {
static void maxRegions ( int n ) { int num ; num = n * ( n + 1 ) / 2 + 1 ;
System . out . println ( num ) ; ; }
public static void main ( String [ ] args ) { int n = 10 ; maxRegions ( n ) ; } }
import java . util . * ; class GFG {
static void checkSolveable ( int n , int m ) {
if ( n == 1 m == 1 ) System . out . print ( " YES " ) ;
else if ( m == 2 && n == 2 ) System . out . print ( " YES " ) ; else System . out . print ( " NO " ) ; }
public static void main ( String [ ] args ) { int n = 1 , m = 3 ; checkSolveable ( n , m ) ; } }
import java . util . * ; class GFG {
static int GCD ( int a , int b ) {
if ( b == 0 ) return a ;
else return GCD ( b , a % b ) ; }
static void check ( int x , int y ) {
if ( GCD ( x , y ) == 1 ) { System . out . print ( " Yes " ) ; } else { System . out . print ( " No " ) ; } }
public static void main ( String [ ] args ) {
int X = 2 , Y = 7 ;
check ( X , Y ) ; } }
import java . util . * ; class GFG { static final int size = 1000001 ;
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
public static void main ( String [ ] args ) { int [ ] prime = new int [ size ] ; seiveOfEratosthenes ( prime ) ; int L = 1 , R = 7 , M = 3 ; System . out . print ( probabiltyEuler ( prime , L , R , M ) ) ; } }
import java . util . * ; class GFG {
public static void findWinner ( int n , int k ) { int cnt = 0 ;
if ( n == 1 ) System . out . println ( " No " ) ;
else if ( ( n & 1 ) != 0 n == 2 ) System . out . println ( " Yes " ) ; else { int tmp = n ; int val = 1 ;
while ( tmp > k && tmp % 2 == 0 ) { tmp /= 2 ; val *= 2 ; }
for ( int i = 3 ; i <= Math . sqrt ( tmp ) ; i ++ ) { while ( tmp % i == 0 ) { cnt ++ ; tmp /= i ; } } if ( tmp > 1 ) cnt ++ ;
if ( val == n ) System . out . println ( " No " ) ; else if ( n / tmp == 2 && cnt == 1 ) System . out . println ( " No " ) ;
else System . out . println ( " Yes " ) ; } }
public static void main ( String [ ] args ) { int n = 1 , k = 1 ; findWinner ( n , k ) ; } }
import java . util . * ; class GFG {
static void pen_hex ( long n ) { long pn = 1 ; for ( long i = 1 ; i < n ; i ++ ) {
pn = i * ( 3 * i - 1 ) / 2 ; if ( pn > n ) break ;
double seqNum = ( 1 + Math . sqrt ( 8 * pn + 1 ) ) / 4 ; if ( seqNum == ( long ) seqNum ) System . out . print ( pn + " , ▁ " ) ; } }
public static void main ( String [ ] args ) { long N = 1000000 ; pen_hex ( N ) ; } }
import java . util . * ; class GFG {
static boolean isPal ( int a [ ] [ ] , int n , int m ) {
for ( int i = 0 ; i < n / 2 ; i ++ ) { for ( int j = 0 ; j < m - 1 ; j ++ ) { if ( a [ i ] [ j ] != a [ n - 1 - i ] [ m - 1 - j ] ) return false ; } } return true ; }
public static void main ( String [ ] args ) { int n = 3 , m = 3 ; int a [ ] [ ] = { { 1 , 2 , 3 } , { 4 , 5 , 4 } , { 3 , 2 , 1 } } ; if ( isPal ( a , n , m ) ) { System . out . print ( " YES " + "NEW_LINE"); } else { System . out . print ( " NO " + "NEW_LINE"); } } }
class GFG {
static int getSum ( int n ) { int sum = 0 ; while ( n != 0 ) { sum = sum + n % 10 ; n = n / 10 ; } return sum ; }
static void smallestNumber ( int N ) { int i = 1 ; while ( 1 != 0 ) {
if ( getSum ( i ) == N ) { System . out . print ( i ) ; break ; } i ++ ; } }
public static void main ( String [ ] args ) { int N = 10 ; smallestNumber ( N ) ; } }
class GFG {
static int reversDigits ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; } return rev_num ; }
static boolean isPerfectSquare ( double x ) {
double sr = Math . sqrt ( x ) ;
return ( ( sr - Math . floor ( sr ) ) == 0 ) ; }
static boolean isRare ( int N ) {
int reverseN = reversDigits ( N ) ;
if ( reverseN == N ) return false ; return isPerfectSquare ( N + reverseN ) && isPerfectSquare ( N - reverseN ) ; }
public static void main ( String [ ] args ) { int n = 65 ; if ( isRare ( n ) ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
import java . util . * ; class GFG {
static void calc_ans ( int l , int r ) { Vector < Integer > power2 = new Vector < Integer > ( ) , power3 = new Vector < Integer > ( ) ;
int mul2 = 1 ; while ( mul2 <= r ) { power2 . add ( mul2 ) ; mul2 *= 2 ; }
int mul3 = 1 ; while ( mul3 <= r ) { power3 . add ( mul3 ) ; mul3 *= 3 ; }
Vector < Integer > power23 = new Vector < Integer > ( ) ; for ( int x = 0 ; x < power2 . size ( ) ; x ++ ) { for ( int y = 0 ; y < power3 . size ( ) ; y ++ ) { int mul = power2 . get ( x ) * power3 . get ( y ) ; if ( mul == 1 ) continue ;
if ( mul <= r ) power23 . add ( mul ) ; } }
int ans = 0 ; for ( int x : power23 ) { if ( x >= l && x <= r ) ans ++ ; }
System . out . print ( ans + "NEW_LINE"); }
public static void main ( String [ ] args ) { int l = 1 , r = 10 ; calc_ans ( l , r ) ; } }
import java . util . * ; class GFG {
static int nCr ( int n , int r ) { if ( r > n ) return 0 ; return fact ( n ) / ( fact ( r ) * fact ( n - r ) ) ; }
static int fact ( int n ) { int res = 1 ; for ( int i = 2 ; i <= n ; i ++ ) res = res * i ; return res ; }
static int countSubsequences ( int arr [ ] , int n , int k ) { int countOdd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] % 2 == 1 ) countOdd ++ ; } int ans = nCr ( n , k ) - nCr ( countOdd , k ) ; return ans ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , 4 } ; int K = 1 ; int N = arr . length ; System . out . println ( countSubsequences ( arr , N , K ) ) ; } }
import java . util . * ; class GFG {
static void first_digit ( int x , int y ) {
int length = ( int ) ( Math . log ( x ) / Math . log ( y ) + 1 ) ;
int first_digit = ( int ) ( x / Math . pow ( y , length - 1 ) ) ; System . out . println ( first_digit ) ; }
public static void main ( String args [ ] ) { int X = 55 , Y = 3 ; first_digit ( X , Y ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static void checkIfCurzonNumber ( long N ) { double powerTerm , productTerm ;
powerTerm = Math . pow ( 2 , N ) + 1 ;
productTerm = 2 * N + 1 ;
if ( powerTerm % productTerm == 0 ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; }
public static void main ( String [ ] args ) { long N = 5 ; checkIfCurzonNumber ( N ) ; N = 10 ; checkIfCurzonNumber ( N ) ; } }
class GFG {
static int minCount ( int n ) {
int [ ] hasharr = { 10 , 3 , 6 , 9 , 2 , 5 , 8 , 1 , 4 , 7 } ;
if ( n > 69 ) return hasharr [ n % 10 ] ; else {
if ( n >= hasharr [ n % 10 ] * 7 ) return ( hasharr [ n % 10 ] ) ; else return - 1 ; } }
public static void main ( String [ ] args ) { int n = 38 ; System . out . println ( minCount ( n ) ) ; } }
import java . io . * ; class GFG {
static void modifiedBinaryPattern ( int n ) {
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) {
if ( j == 1 j == i ) System . out . print ( 1 ) ;
else System . out . print ( 0 ) ; }
System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int n = 7 ;
modifiedBinaryPattern ( n ) ; } }
class GFG {
static void findRealAndImag ( String s ) {
int l = s . length ( ) ;
int i ;
if ( s . indexOf ( ' + ' ) != - 1 ) { i = s . indexOf ( ' + ' ) ; }
else { i = s . indexOf ( ' - ' ) ; }
String real = s . substring ( 0 , i ) ;
String imaginary = s . substring ( i + 1 , l - 1 ) ; System . out . println ( " Real ▁ part : ▁ " + real ) ; System . out . println ( " Imaginary ▁ part : ▁ " + imaginary ) ; }
public static void main ( String [ ] args ) { String s = "3 + 4i " ; findRealAndImag ( s ) ; } }
class GFG {
static int highestPower ( int n , int k ) { int i = 0 ; int a = ( int ) Math . pow ( n , i ) ;
while ( a <= k ) { i += 1 ; a = ( int ) Math . pow ( n , i ) ; } return i - 1 ; }
static int b [ ] = new int [ 50 ] ;
static int PowerArray ( int n , int k ) { while ( k > 0 ) {
int t = highestPower ( n , k ) ;
if ( b [ t ] > 0 ) {
System . out . print ( - 1 ) ; return 0 ; } else
b [ t ] = 1 ;
k -= Math . pow ( n , t ) ; }
for ( int i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] > 0 ) { System . out . print ( i + " , ▁ " ) ; } } return 0 ; }
public static void main ( String [ ] args ) { int N = 3 ; int K = 40 ; PowerArray ( N , K ) ; } }
import java . util . * ; class GFG { static final int N = 10005 ;
static void SieveOfEratosthenes ( Vector < Boolean > composite ) { for ( int i = 0 ; i < N ; i ++ ) { composite . add ( i , false ) ; } for ( int p = 2 ; p * p < N ; p ++ ) {
if ( ! composite . get ( p ) ) {
for ( int i = p * 2 ; i < N ; i += p ) { composite . add ( i , true ) ; } } } }
static int sumOfElements ( int arr [ ] , int n ) { Vector < Boolean > composite = new Vector < Boolean > ( ) ; for ( int i = 0 ; i < N ; i ++ ) composite . add ( false ) ; SieveOfEratosthenes ( composite ) ;
HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) if ( mp . containsKey ( arr [ i ] ) ) { mp . put ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . put ( arr [ i ] , 1 ) ; }
int sum = 0 ;
for ( Map . Entry < Integer , Integer > it : mp . entrySet ( ) ) {
if ( composite . get ( it . getValue ( ) ) ) { sum += ( it . getKey ( ) ) ; } } return sum ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 1 , 1 , 1 , 3 , 3 , 2 , 4 } ; int n = arr . length ;
System . out . print ( sumOfElements ( arr , n ) ) ; } }
import java . util . * ; class GFG {
static void remove ( int arr [ ] , int n ) {
HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( mp . containsKey ( arr [ i ] ) ) { mp . put ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . put ( arr [ i ] , 1 ) ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( ( mp . containsKey ( arr [ i ] ) && mp . get ( arr [ i ] ) % 2 == 1 ) ) continue ; System . out . print ( arr [ i ] + " , ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 3 , 3 , 2 , 2 , 4 , 7 , 7 } ; int n = arr . length ;
remove ( arr , n ) ; } }
import java . util . * ; class GFG {
static void getmax ( int arr [ ] , int n , int x ) {
int s = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { s = s + arr [ i ] ; }
System . out . print ( Math . min ( s , x ) ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int x = 5 ; int arr_size = arr . length ; getmax ( arr , arr_size , x ) ; } }
class GFG {
static void shortestLength ( int n , int x [ ] , int y [ ] ) { int answer = 0 ;
int i = 0 ; while ( n != 0 && i < x . length ) {
if ( x [ i ] + y [ i ] > answer ) answer = x [ i ] + y [ i ] ; i ++ ; }
System . out . println ( " Length ▁ - > ▁ " + answer ) ; System . out . println ( " Path ▁ - > ▁ " + " ( ▁ 1 , ▁ " + answer + " ▁ ) " + " and ▁ ( ▁ " + answer + " , ▁ 1 ▁ ) " ) ; }
public static void main ( String [ ] args ) {
int n = 4 ;
int x [ ] = new int [ ] { 1 , 4 , 2 , 1 } ; int y [ ] = new int [ ] { 4 , 1 , 1 , 2 } ; shortestLength ( n , x , y ) ; } }
class GFG {
static void FindPoints ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 , int x4 , int y4 ) {
int x5 = Math . max ( x1 , x3 ) ; int y5 = Math . max ( y1 , y3 ) ;
int x6 = Math . min ( x2 , x4 ) ; int y6 = Math . min ( y2 , y4 ) ;
if ( x5 > x6 y5 > y6 ) { System . out . println ( " No ▁ intersection " ) ; return ; } System . out . print ( " ( " + x5 + " , ▁ " + y5 + " ) ▁ " ) ; System . out . print ( " ( " + x6 + " , ▁ " + y6 + " ) ▁ " ) ;
int x7 = x5 ; int y7 = y6 ; System . out . print ( " ( " + x7 + " , ▁ " + y7 + " ) ▁ " ) ;
int x8 = x6 ; int y8 = y5 ; System . out . print ( " ( " + x8 + " , ▁ " + y8 + " ) ▁ " ) ; }
public static void main ( String args [ ] ) {
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x3 = 2 , y3 = 3 , x4 = 7 , y4 = 9 ;
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) ; } }
class GFG {
static class Point { float x , y ; Point ( ) { x = y = 0 ; } Point ( float a , float b ) { x = a ; y = b ; } } ;
static void printCorners ( Point p , Point q , float l ) { Point a = new Point ( ) , b = new Point ( ) , c = new Point ( ) , d = new Point ( ) ;
if ( p . x == q . x ) { a . x = ( float ) ( p . x - ( l / 2.0 ) ) ; a . y = p . y ; d . x = ( float ) ( p . x + ( l / 2.0 ) ) ; d . y = p . y ; b . x = ( float ) ( q . x - ( l / 2.0 ) ) ; b . y = q . y ; c . x = ( float ) ( q . x + ( l / 2.0 ) ) ; c . y = q . y ; }
else if ( p . y == q . y ) { a . y = ( float ) ( p . y - ( l / 2.0 ) ) ; a . x = p . x ; d . y = ( float ) ( p . y + ( l / 2.0 ) ) ; d . x = p . x ; b . y = ( float ) ( q . y - ( l / 2.0 ) ) ; b . x = q . x ; c . y = ( float ) ( q . y + ( l / 2.0 ) ) ; c . x = q . x ; }
else {
float m = ( p . x - q . x ) / ( q . y - p . y ) ;
float dx = ( float ) ( ( l / Math . sqrt ( 1 + ( m * m ) ) ) * 0.5 ) ; float dy = m * dx ; a . x = p . x - dx ; a . y = p . y - dy ; d . x = p . x + dx ; d . y = p . y + dy ; b . x = q . x - dx ; b . y = q . y - dy ; c . x = q . x + dx ; c . y = q . y + dy ; } System . out . print ( ( int ) a . x + " , ▁ " + ( int ) a . y + " NEW_LINE" + ( int ) b . x + " , ▁ " + ( int ) b . y + "NEW_LINE" + ( int ) c . x + " , ▁ " + ( int ) c . y + " NEW_LINE" + ( int ) d . x + " , ▁ " + ( int ) d . y + "NEW_LINE"); }
public static void main ( String [ ] args ) { Point p1 = new Point ( 1 , 0 ) , q1 = new Point ( 1 , 2 ) ; printCorners ( p1 , q1 , 2 ) ; Point p = new Point ( 1 , 1 ) , q = new Point ( - 1 , - 1 ) ; printCorners ( p , q , ( float ) ( 2 * Math . sqrt ( 2 ) ) ) ; } }
class GFG {
public static int minimumCost ( int arr [ ] , int N , int X , int Y ) {
int even_count = 0 , odd_count = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( ( arr [ i ] & 1 ) > 0 && ( i % 2 == 0 ) ) { odd_count ++ ; }
if ( ( arr [ i ] % 2 ) == 0 && ( i & 1 ) > 0 ) { even_count ++ ; } }
int cost1 = X * Math . min ( odd_count , even_count ) ;
int cost2 = Y * ( Math . max ( odd_count , even_count ) - Math . min ( odd_count , even_count ) ) ;
int cost3 = ( odd_count + even_count ) * Y ;
return Math . min ( cost1 + cost2 , cost3 ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 5 , 3 , 7 , 2 , 1 } , X = 10 , Y = 2 ; int N = arr . length ; System . out . println ( minimumCost ( arr , N , X , Y ) ) ; } }
import java . io . * ; class GFG {
static int findMinMax ( int [ ] a ) {
int min_val = 1000000000 ;
for ( int i = 1 ; i < a . length ; ++ i ) {
min_val = Math . min ( min_val , a [ i ] * a [ i - 1 ] ) ; }
return min_val ; }
public static void main ( String [ ] args ) { int [ ] arr = { 6 , 4 , 5 , 6 , 2 , 4 , 1 } ; System . out . println ( findMinMax ( arr ) ) ; } }
import java . util . * ; public class GFG { static int sum ;
static class TreeNode { int data ; TreeNode left ; TreeNode right ;
TreeNode ( int data ) { this . data = data ; this . left = null ; this . right = null ; } } ;
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
else if ( data <= root . data ) { root . left = insertNode ( data , root . left ) ; }
return root ; }
static void findSum ( TreeNode root , int target , int K ) {
sum = 0 ; kDistanceSum ( root , target , K ) ;
System . out . print ( sum ) ; }
public static void main ( String [ ] args ) { TreeNode root = null ; int N = 11 ; int tree [ ] = { 3 , 1 , 7 , 0 , 2 , 5 , 10 , 4 , 6 , 9 , 8 } ;
for ( int i = 0 ; i < N ; i ++ ) { root = insertNode ( tree [ i ] , root ) ; } int target = 7 ; int K = 2 ; findSum ( root , target , K ) ; } }
import java . io . * ; class GFG {
static int itemType ( int n ) {
int count = 0 ;
for ( int day = 1 ; ; day ++ ) {
for ( int type = day ; type > 0 ; type -- ) { count += type ;
if ( count >= n ) return type ; } } }
public static void main ( String [ ] args ) { int N = 10 ; System . out . println ( itemType ( N ) ) ; } }
class GFG {
static int FindSum ( int [ ] arr , int N ) {
int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int power = ( int ) ( Math . log ( arr [ i ] ) / Math . log ( 2 ) ) ;
int LesserValue = ( int ) Math . pow ( 2 , power ) ;
int LargerValue = ( int ) Math . pow ( 2 , power + 1 ) ;
if ( ( arr [ i ] - LesserValue ) == ( LargerValue - arr [ i ] ) ) {
res += arr [ i ] ; } }
return res ; }
public static void main ( String [ ] args ) { int [ ] arr = { 10 , 24 , 17 , 3 , 8 } ; int N = arr . length ; System . out . println ( FindSum ( arr , N ) ) ; } }
import java . util . * ; import java . lang . * ; class GFG {
static void findLast ( int mat [ ] [ ] ) { int m = 3 ; int n = 3 ;
Set < Integer > rows = new HashSet < Integer > ( ) ; Set < Integer > cols = new HashSet < Integer > ( ) ; for ( int i = 0 ; i < m ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( ( mat [ i ] [ j ] > 0 ) ) { rows . add ( i ) ; cols . add ( j ) ; } } }
int avRows = m - rows . size ( ) ; int avCols = n - cols . size ( ) ;
int choices = Math . min ( avRows , avCols ) ;
if ( ( choices & 1 ) != 0 )
System . out . println ( " P1" ) ;
else System . out . println ( " P2" ) ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 0 , 0 } , { 0 , 0 , 0 } , { 0 , 0 , 1 } } ; findLast ( mat ) ; } }
import java . io . * ; import java . lang . * ; class GFG { static final int MOD = 1000000007 ;
static void sumOfBinaryNumbers ( int n ) {
int ans = 0 ; int one = 1 ;
while ( true ) {
if ( n <= 1 ) { ans = ( ans + n ) % MOD ; break ; }
int x = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) ; int cur = 0 ; int add = ( int ) ( Math . pow ( 2 , ( x - 1 ) ) ) ;
for ( int i = 1 ; i <= x ; i ++ ) {
cur = ( cur + add ) % MOD ; add = ( add * 10 % MOD ) ; }
ans = ( ans + cur ) % MOD ;
int rem = n - ( int ) ( Math . pow ( 2 , x ) ) + 1 ;
int p = ( int ) Math . pow ( 10 , x ) ; p = ( p * ( rem % MOD ) ) % MOD ; ans = ( ans + p ) % MOD ;
n = rem - 1 ; }
System . out . println ( ans ) ; }
public static void main ( String [ ] args ) { int N = 3 ; sumOfBinaryNumbers ( N ) ; } }
class GFG {
static void nearestFibonacci ( int num ) {
if ( num == 0 ) { System . out . print ( 0 ) ; return ; }
int first = 0 , second = 1 ;
int third = first + second ;
while ( third <= num ) {
first = second ;
second = third ;
third = first + second ; }
int ans = ( Math . abs ( third - num ) >= Math . abs ( second - num ) ) ? second : third ;
System . out . print ( ans ) ; }
public static void main ( String [ ] args ) { int N = 17 ; nearestFibonacci ( N ) ; } }
import java . io . * ; import java . lang . * ; import java . util . * ; class GFG {
static boolean checkPermutation ( int ans [ ] , int a [ ] , int n ) {
int Max = Integer . MIN_VALUE ;
for ( int i = 0 ; i < n ; i ++ ) {
Max = Math . max ( Max , ans [ i ] ) ;
if ( Max != a [ i ] ) return false ; }
return true ; }
static void findPermutation ( int a [ ] , int n ) {
int ans [ ] = new int [ n ] ;
HashMap < Integer , Integer > um = new HashMap < > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( ! um . containsKey ( a [ i ] ) ) {
ans [ i ] = a [ i ] ; um . put ( a [ i ] , i ) ; } }
ArrayList < Integer > v = new ArrayList < > ( ) ; int j = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) {
if ( ! um . containsKey ( i ) ) { v . add ( i ) ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( ans [ i ] == 0 ) { ans [ i ] = v . get ( j ) ; j ++ ; } }
if ( checkPermutation ( ans , a , n ) ) {
for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( ans [ i ] + " ▁ " ) ; } }
else System . out . println ( " - 1" ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 4 , 5 , 5 } ; int N = arr . length ;
findPermutation ( arr , N ) ; } }
import java . io . * ; import java . util . Map ; import java . util . HashMap ; class GFG {
public static void countEqualElementPairs ( int arr [ ] , int N ) {
HashMap < Integer , Integer > map = new HashMap < > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { Integer k = map . get ( arr [ i ] ) ; map . put ( arr [ i ] , ( k == null ) ? 1 : k + 1 ) ; }
int total = 0 ;
for ( Map . Entry < Integer , Integer > e : map . entrySet ( ) ) {
total += ( e . getValue ( ) * ( e . getValue ( ) - 1 ) ) / 2 ; }
for ( int i = 0 ; i < N ; i ++ ) {
System . out . print ( total - ( map . get ( arr [ i ] ) - 1 ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 1 , 2 , 1 , 2 } ;
int N = 5 ; countEqualElementPairs ( arr , N ) ; } }
public class GFG {
static int count ( int N ) { int sum = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { sum += 7 * Math . pow ( 8 , i - 1 ) ; } return sum ; }
public static void main ( String [ ] args ) { int N = 4 ; System . out . println ( count ( N ) ) ; } }
import java . util . * ; class GFG {
static boolean isPalindrome ( int n ) {
String str = String . valueOf ( n ) ;
int s = 0 , e = str . length ( ) - 1 ; while ( s < e ) {
if ( str . charAt ( s ) != str . charAt ( e ) ) { return false ; } s ++ ; e -- ; } return true ; }
static void palindromicDivisors ( int n ) {
Vector < Integer > PalindromDivisors = new Vector < Integer > ( ) ; for ( int i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) {
if ( n / i == i ) {
if ( isPalindrome ( i ) ) { PalindromDivisors . add ( i ) ; } } else {
if ( isPalindrome ( i ) ) { PalindromDivisors . add ( i ) ; }
if ( isPalindrome ( n / i ) ) { PalindromDivisors . add ( n / i ) ; } } } }
Collections . sort ( PalindromDivisors ) ; for ( int i = 0 ; i < PalindromDivisors . size ( ) ; i ++ ) { System . out . print ( PalindromDivisors . get ( i ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int n = 66 ;
palindromicDivisors ( n ) ; } }
class GFG {
static int findMinDel ( int [ ] arr , int n ) {
int min_num = Integer . MAX_VALUE ;
for ( int i = 0 ; i < n ; i ++ ) min_num = Math . min ( arr [ i ] , min_num ) ;
int cnt = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] == min_num ) cnt ++ ;
return n - cnt ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 3 , 2 } ; int n = arr . length ; System . out . print ( findMinDel ( arr , n ) ) ; } }
class GFG {
static int cntSubArr ( int [ ] arr , int n ) {
int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int curr_gcd = 0 ;
for ( int j = i ; j < n ; j ++ ) { curr_gcd = __gcd ( curr_gcd , arr [ j ] ) ;
ans += ( curr_gcd == 1 ) ? 1 : 0 ; } }
return ans ; } static int __gcd ( int a , int b ) { if ( b == 0 ) return a ; return __gcd ( b , a % b ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 1 } ; int n = arr . length ; System . out . println ( cntSubArr ( arr , n ) ) ; } }
class GFG {
static void print_primes_till_N ( int N ) {
int i , j , flag ;
System . out . println ( " Prime ▁ numbers ▁ between ▁ 1 ▁ and ▁ " + N + " ▁ are : " ) ;
for ( i = 1 ; i <= N ; i ++ ) {
if ( i == 1 i == 0 ) continue ;
flag = 1 ; for ( j = 2 ; j <= i / 2 ; ++ j ) { if ( i % j == 0 ) { flag = 0 ; break ; } }
if ( flag == 1 ) System . out . print ( i + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int N = 100 ; print_primes_till_N ( N ) ; } }
class GFG { static int MAX = 32 ;
static int findX ( int A , int B ) { int X = 0 ;
for ( int bit = 0 ; bit < MAX ; bit ++ ) {
int tempBit = 1 << bit ;
int bitOfX = A & B & tempBit ;
X += bitOfX ; } return X ; }
public static void main ( String [ ] args ) { int A = 11 , B = 13 ; System . out . println ( findX ( A , B ) ) ; } }
import java . util . * ; class GFG {
static int cntSubSets ( int arr [ ] , int n ) {
int maxVal = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ;
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == maxVal ) cnt ++ ; }
return ( int ) ( Math . pow ( 2 , cnt ) - 1 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 1 , 2 } ; int n = arr . length ; System . out . println ( cntSubSets ( arr , n ) ) ; } }
import java . util . * ; class GFG {
static float findProb ( int arr [ ] , int n ) {
long maxSum = Integer . MIN_VALUE , maxCount = 0 , totalPairs = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
int sum = arr [ i ] + arr [ j ] ;
if ( sum == maxSum ) {
maxCount ++ ; }
else if ( sum > maxSum ) {
maxSum = sum ; maxCount = 1 ; } totalPairs ++ ; } }
float prob = ( float ) maxCount / ( float ) totalPairs ; return prob ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 1 , 1 , 2 , 2 , 2 } ; int n = arr . length ; System . out . println ( findProb ( arr , n ) ) ; } }
class GFG { static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
static int maxCommonFactors ( int a , int b ) {
int __gcd = gcd ( a , b ) ;
int ans = 1 ;
for ( int i = 2 ; i * i <= __gcd ; i ++ ) { if ( __gcd % i == 0 ) { ans ++ ; while ( __gcd % i == 0 ) __gcd /= i ; } }
if ( __gcd != 1 ) ans ++ ;
return ans ; }
public static void main ( String [ ] args ) { int a = 12 , b = 18 ; System . out . println ( maxCommonFactors ( a , b ) ) ; } }
class GFG { static int days [ ] = { 31 , 28 , 31 , 30 , 31 , 30 , 31 , 31 , 30 , 31 , 30 , 31 } ;
static int dayOfYear ( String date ) {
int year = Integer . parseInt ( date . substring ( 0 , 4 ) ) ; int month = Integer . parseInt ( date . substring ( 5 , 7 ) ) ; int day = Integer . parseInt ( date . substring ( 8 ) ) ;
if ( month > 2 && year % 4 == 0 && ( year % 100 != 0 year % 400 == 0 ) ) { ++ day ; }
while ( -- month > 0 ) { day = day + days [ month - 1 ] ; } return day ; }
public static void main ( String [ ] args ) { String date = "2019-01-09" ; System . out . println ( dayOfYear ( date ) ) ; } }
class GFG {
public static int Cells ( int n , int x ) { int ans = 0 ; for ( int i = 1 ; i <= n ; i ++ ) if ( x % i == 0 && x / i <= n ) ans ++ ; return ans ; }
public static void main ( String [ ] args ) { int n = 6 , x = 12 ;
System . out . println ( Cells ( n , x ) ) ; } }
import java . util . * ; import java . lang . Math ; import java . io . * ; class GFG {
static int nextPowerOfFour ( int n ) { int x = ( int ) Math . floor ( Math . sqrt ( Math . sqrt ( n ) ) ) ;
if ( Math . pow ( x , 4 ) == n ) return n ; else { x = x + 1 ; return ( int ) Math . pow ( x , 4 ) ; } }
public static void main ( String [ ] args ) throws java . lang . Exception { int n = 122 ; System . out . println ( nextPowerOfFour ( n ) ) ; } }
class GFG {
static int minOperations ( int x , int y , int p , int q ) {
if ( y % x != 0 ) return - 1 ; int d = y / x ;
int a = 0 ;
while ( d % p == 0 ) { d /= p ; a ++ ; }
int b = 0 ;
while ( d % q == 0 ) { d /= q ; b ++ ; }
if ( d != 1 ) return - 1 ;
return ( a + b ) ; }
public static void main ( String [ ] args ) { int x = 12 , y = 2592 , p = 2 , q = 3 ; System . out . println ( minOperations ( x , y , p , q ) ) ; } }
import java . util . * ; class GFG {
static int nCr ( int n ) {
if ( n < 4 ) return 0 ; int answer = n * ( n - 1 ) * ( n - 2 ) * ( n - 3 ) ; answer /= 24 ; return answer ; }
static int countQuadruples ( int N , int K ) {
int M = N / K ; int answer = nCr ( M ) ;
for ( int i = 2 ; i < M ; i ++ ) { int j = i ;
int temp2 = M / i ;
int count = 0 ;
int check = 0 ; int temp = j ; while ( j % 2 == 0 ) { count ++ ; j /= 2 ; if ( count >= 2 ) break ; } if ( count >= 2 ) { check = 1 ; } for ( int k = 3 ; k <= Math . sqrt ( temp ) ; k += 2 ) { int cnt = 0 ; while ( j % k == 0 ) { cnt ++ ; j /= k ; if ( cnt >= 2 ) break ; } if ( cnt >= 2 ) { check = 1 ; break ; } else if ( cnt == 1 ) count ++ ; } if ( j > 2 ) { count ++ ; }
if ( check == 1 ) continue ; else {
if ( count % 2 == 1 ) { answer -= nCr ( temp2 ) ; } else { answer += nCr ( temp2 ) ; } } } return answer ; }
public static void main ( String [ ] args ) { int N = 10 , K = 2 ; System . out . println ( countQuadruples ( N , K ) ) ; } }
import java . io . * ; class GFG {
static int getX ( int a , int b , int c , int d ) { int X = ( b * c - a * d ) / ( d - c ) ; return X ; }
public static void main ( String [ ] args ) { int a = 2 , b = 3 , c = 4 , d = 5 ; System . out . println ( getX ( a , b , c , d ) ) ; } }
import java . util . * ; import java . lang . * ; class GFG {
static boolean isVowel ( char ch ) { if ( ch == ' a ' ch == ' e ' ch == ' i ' ch == ' o ' ch == ' u ' ) return true ; else return false ; }
static long fact ( long n ) { if ( n < 2 ) { return 1 ; } return n * fact ( n - 1 ) ; }
static long only_vowels ( HashMap < Character , Integer > freq ) { long denom = 1 ; long cnt_vwl = 0 ;
for ( Map . Entry < Character , Integer > itr : freq . entrySet ( ) ) { if ( isVowel ( itr . getKey ( ) ) ) { denom *= fact ( itr . getValue ( ) ) ; cnt_vwl += itr . getValue ( ) ; } } return fact ( cnt_vwl ) / denom ; }
static long all_vowels_together ( HashMap < Character , Integer > freq ) {
long vow = only_vowels ( freq ) ;
long denom = 1 ;
long cnt_cnst = 0 ; for ( Map . Entry < Character , Integer > itr : freq . entrySet ( ) ) { if ( ! isVowel ( itr . getKey ( ) ) ) { denom *= fact ( itr . getValue ( ) ) ; cnt_cnst += itr . getValue ( ) ; } }
long ans = fact ( cnt_cnst + 1 ) / denom ; return ( ans * vow ) ; }
static long total_permutations ( HashMap < Character , Integer > freq ) {
long cnt = 0 ;
long denom = 1 ; for ( Map . Entry < Character , Integer > itr : freq . entrySet ( ) ) { denom *= fact ( itr . getValue ( ) ) ; cnt += itr . getValue ( ) ; }
return fact ( cnt ) / denom ; }
static long no_vowels_together ( String word ) {
HashMap < Character , Integer > freq = new HashMap < > ( ) ;
for ( int i = 0 ; i < word . length ( ) ; i ++ ) { char ch = Character . toLowerCase ( word . charAt ( i ) ) ; if ( freq . containsKey ( ch ) ) { freq . put ( ch , freq . get ( ch ) + 1 ) ; } else { freq . put ( ch , 1 ) ; } }
long total = total_permutations ( freq ) ;
long vwl_tgthr = all_vowels_together ( freq ) ;
long res = total - vwl_tgthr ;
return res ; }
public static void main ( String [ ] args ) { String word = " allahabad " ; long ans = no_vowels_together ( word ) ; System . out . println ( ans ) ; word = " geeksforgeeks " ; ans = no_vowels_together ( word ) ; System . out . println ( ans ) ; word = " abcd " ; ans = no_vowels_together ( word ) ; System . out . println ( ans ) ; } }
import java . util . * ; class GFG {
static int numberOfMen ( int D , int m , int d ) { int Men = ( m * ( D - d ) ) / d ; return Men ; }
public static void main ( String args [ ] ) { int D = 5 , m = 4 , d = 4 ; System . out . println ( numberOfMen ( D , m , d ) ) ; } }
import java . io . * ; class GFG {
static double area ( double a , double b , double c ) { double d = Math . abs ( ( c * c ) / ( 2 * a * b ) ) ; return d ; }
public static void main ( String [ ] args ) { double a = - 2 , b = 4 , c = 3 ; System . out . println ( area ( a , b , c ) ) ; } }
import java . util . * ; class GFG {
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
import java . util . * ; class GFG { static int MAX = 100005 ;
static int kadaneAlgorithm ( int [ ] ar , int n ) { int sum = 0 , maxSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) { sum += ar [ i ] ; if ( sum < 0 ) sum = 0 ; maxSum = Math . max ( maxSum , sum ) ; } return maxSum ; }
static int maxFunction ( int [ ] arr , int n ) { int [ ] b = new int [ MAX ] ; int [ ] c = new int [ MAX ] ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { if ( i % 2 == 1 ) { b [ i ] = Math . abs ( arr [ i + 1 ] - arr [ i ] ) ; c [ i ] = - b [ i ] ; } else { c [ i ] = Math . abs ( arr [ i + 1 ] - arr [ i ] ) ; b [ i ] = - c [ i ] ; } }
int ans = kadaneAlgorithm ( b , n - 1 ) ; ans = Math . max ( ans , kadaneAlgorithm ( c , n - 1 ) ) ; return ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 5 , 4 , 7 } ; int n = arr . length ; System . out . println ( maxFunction ( arr , n ) ) ; } }
class GFG {
static int findThirdDigit ( int n ) {
if ( n < 3 ) return 0 ;
return ( n & 1 ) > 0 ? 1 : 6 ; }
public static void main ( String args [ ] ) { int n = 7 ; System . out . println ( findThirdDigit ( n ) ) ; } }
class GFG {
static double getProbability ( int a , int b , int c , int d ) {
double p = ( double ) a / ( double ) b ; double q = ( double ) c / ( double ) d ;
double ans = p * ( 1 / ( 1 - ( 1 - q ) * ( 1 - p ) ) ) ; return ans ; }
public static void main ( String [ ] args ) { int a = 1 , b = 2 , c = 10 , d = 11 ; System . out . printf ( " % .5f " , getProbability ( a , b , c , d ) ) ; } }
import java . util . * ; class GFG {
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
class OddPosition {
public static long getFinalElement ( long n ) { long finalNum ; for ( finalNum = 2 ; finalNum * 2 <= n ; finalNum *= 2 ) ; return finalNum ; }
public static void main ( String [ ] args ) { int N = 12 ; System . out . println ( getFinalElement ( N ) ) ; } }
import java . util . * ; class GFG {
static void SieveOfEratosthenes ( boolean prime [ ] , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
static int sumOfElements ( int arr [ ] , int n ) { boolean prime [ ] = new boolean [ n + 1 ] ; Arrays . fill ( prime , true ) ; SieveOfEratosthenes ( prime , n + 1 ) ; int i , j ;
HashMap < Integer , Integer > m = new HashMap < > ( ) ; for ( i = 0 ; i < n ; i ++ ) { if ( m . containsKey ( arr [ i ] ) ) m . put ( arr [ i ] , m . get ( arr [ i ] ) + 1 ) ; else m . put ( arr [ i ] , 1 ) ; } int sum = 0 ;
for ( Map . Entry < Integer , Integer > entry : m . entrySet ( ) ) { int key = entry . getKey ( ) ; int value = entry . getValue ( ) ;
if ( prime [ value ] ) { sum += ( key ) ; } } return sum ; }
public static void main ( String args [ ] ) { int arr [ ] = { 5 , 4 , 6 , 5 , 4 , 6 } ; int n = arr . length ; System . out . println ( sumOfElements ( arr , n ) ) ; } }
class GFG {
static boolean isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
static boolean isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
static long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
public static void main ( String [ ] args ) { int L = 110 , R = 1130 ; System . out . println ( sumOfAllPalindrome ( L , R ) ) ; } }
import java . util . * ; class GFG {
static int fact ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f = f * i ; return f ; }
static int waysOfConsonants ( int size1 , int [ ] freq ) { int ans = fact ( size1 ) ; for ( int i = 0 ; i < 26 ; i ++ ) {
if ( i == 0 i == 4 i == 8 i == 14 i == 20 ) continue ; else ans = ans / fact ( freq [ i ] ) ; } return ans ; }
static int waysOfVowels ( int size2 , int [ ] freq ) { return fact ( size2 ) / ( fact ( freq [ 0 ] ) * fact ( freq [ 4 ] ) * fact ( freq [ 8 ] ) * fact ( freq [ 14 ] ) * fact ( freq [ 20 ] ) ) ; }
static int countWays ( String str ) { int [ ] freq = new int [ 200 ] ; for ( int i = 0 ; i < 200 ; i ++ ) freq [ i ] = 0 ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) freq [ str . charAt ( i ) - ' a ' ] ++ ;
int vowel = 0 , consonant = 0 ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str . charAt ( i ) != ' a ' && str . charAt ( i ) != ' e ' && str . charAt ( i ) != ' i ' && str . charAt ( i ) != ' o ' && str . charAt ( i ) != ' u ' ) consonant ++ ; else vowel ++ ; }
return waysOfConsonants ( consonant + 1 , freq ) * waysOfVowels ( vowel , freq ) ; }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ; System . out . println ( countWays ( str ) ) ; } }
public class GFG {
static double calculateAlternateSum ( int n ) { if ( n <= 0 ) return 0 ; int fibo [ ] = new int [ n + 1 ] ; fibo [ 0 ] = 0 ; fibo [ 1 ] = 1 ;
double sum = Math . pow ( fibo [ 0 ] , 2 ) + Math . pow ( fibo [ 1 ] , 2 ) ;
for ( int i = 2 ; i <= n ; i ++ ) { fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] ;
if ( i % 2 == 0 ) sum -= fibo [ i ] ;
else sum += fibo [ i ] ; }
return sum ; }
public static void main ( String args [ ] ) {
int n = 8 ;
System . out . println ( " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " + n + " ▁ terms : ▁ " + calculateAlternateSum ( n ) ) ; } }
class GFG {
static int getValue ( int n ) { int i = 0 , k = 1 ; while ( i < n ) { i = i + k ; k = k * 2 ; } return k / 2 ; }
public static void main ( String [ ] args ) {
int n = 9 ;
System . out . println ( getValue ( n ) ) ;
n = 1025 ;
System . out . println ( getValue ( n ) ) ; } }
import java . io . * ; import java . util . * ; public class GFG {
static void countDigits ( double val , long [ ] arr ) { while ( ( long ) val > 0 ) { long digit = ( long ) val % 10 ; arr [ ( int ) digit ] ++ ; val = ( long ) val / 10 ; } return ; } static void countFrequency ( int x , int n ) {
long [ ] freq_count = new long [ 10 ] ;
for ( int i = 1 ; i <= n ; i ++ ) {
double val = Math . pow ( ( double ) x , ( double ) i ) ;
countDigits ( val , freq_count ) ; }
for ( int i = 0 ; i <= 9 ; i ++ ) { System . out . print ( freq_count [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int x = 15 , n = 3 ; countFrequency ( x , n ) ; } }
import java . io . * ; class GFG {
static int countSolutions ( int a ) { int count = 0 ;
for ( int i = 0 ; i <= a ; i ++ ) { if ( a == ( i + ( a ^ i ) ) ) count ++ ; } return count ; }
public static void main ( String [ ] args ) { int a = 3 ; System . out . println ( countSolutions ( a ) ) ; } }
import java . io . * ; class GFG {
static int countSolutions ( int a ) { int count = Integer . bitCount ( a ) ; count = ( int ) Math . pow ( 2 , count ) ; return count ; }
public static void main ( String [ ] args ) { int a = 3 ; System . out . println ( countSolutions ( a ) ) ; } }
class GFG {
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
import java . io . * ; class GFG {
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
import java . io . * ; import java . util . * ; public class GFG {
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
import java . util . * ; class GFG {
static void LucasLehmer ( int n ) {
long current_val = 4 ;
ArrayList < Long > series = new ArrayList < > ( ) ;
series . add ( current_val ) ; for ( int i = 0 ; i < n ; i ++ ) { current_val = current_val * current_val - 2 ; series . add ( current_val ) ; }
for ( int i = 0 ; i <= n ; i ++ ) { System . out . println ( " Term ▁ " + i + " : ▁ " + series . get ( i ) ) ; } }
public static void main ( String [ ] args ) { int n = 5 ; LucasLehmer ( n ) ; } }
import java . io . * ; class GFG {
static int modInverse ( int a , int prime ) { a = a % prime ; for ( int x = 1 ; x < prime ; x ++ ) if ( ( a * x ) % prime == 1 ) return x ; return - 1 ; } static void printModIverses ( int n , int prime ) { for ( int i = 1 ; i <= n ; i ++ ) System . out . print ( modInverse ( i , prime ) + " ▁ " ) ; }
public static void main ( String args [ ] ) { int n = 10 , prime = 17 ; printModIverses ( n , prime ) ; } }
class GFG {
static int minOp ( int num ) {
int rem ; int count = 0 ;
while ( num > 0 ) { rem = num % 10 ; if ( ! ( rem == 3 rem == 8 ) ) count ++ ; num /= 10 ; } return count ; }
public static void main ( String [ ] args ) { int num = 234198 ; System . out . print ( " Minimum ▁ Operations ▁ = " + minOp ( num ) ) ; } }
import java . io . * ; class GFG {
static int sumOfDigits ( int a ) { int sum = 0 ; while ( a != 0 ) { sum += a % 10 ; a /= 10 ; } return sum ; }
static int findMax ( int x ) {
int b = 1 , ans = x ;
while ( x != 0 ) {
int cur = ( x - 1 ) * b + ( b - 1 ) ;
if ( sumOfDigits ( cur ) > sumOfDigits ( ans ) || ( sumOfDigits ( cur ) == sumOfDigits ( ans ) && cur > ans ) ) ans = cur ;
x /= 10 ; b *= 10 ; } return ans ; }
public static void main ( String [ ] args ) { int n = 521 ; System . out . println ( findMax ( n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int median ( int a [ ] , int l , int r ) { int n = r - l + 1 ; n = ( n + 1 ) / 2 - 1 ; return n + l ; }
static int IQR ( int [ ] a , int n ) { Arrays . sort ( a ) ;
int mid_index = median ( a , 0 , n ) ;
int Q1 = a [ median ( a , 0 , mid_index ) ] ;
int Q3 = a [ mid_index + median ( a , mid_index + 1 , n ) ] ;
return ( Q3 - Q1 ) ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 } ; int n = a . length ; System . out . println ( IQR ( a , n ) ) ; } }
import java . util . * ; class GFG {
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
import java . io . * ; class GFG {
static int findSum ( int n , int a , int b ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
public static void main ( String [ ] args ) { int n = 10 , a = 3 , b = 5 ; System . out . println ( findSum ( n , a , b ) ) ; } }
class GFG { static int subtractOne ( int x ) { return ( ( x << 1 ) + ( ~ x ) ) ; } public static void main ( String [ ] args ) { System . out . printf ( " % d " , subtractOne ( 13 ) ) ; } }
class PellNumber {
public static int pell ( int n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( pell ( n ) ) ; } }
import java . util . Vector ; class GFG {
static long LCM ( int arr [ ] , int n ) {
int max_num = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( max_num < arr [ i ] ) { max_num = arr [ i ] ; } }
long res = 1 ;
while ( x <= max_num ) {
Vector < Integer > indexes = new Vector < > ( ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] % x == 0 ) { indexes . add ( indexes . size ( ) , j ) ; } }
if ( indexes . size ( ) >= 2 ) {
for ( int j = 0 ; j < indexes . size ( ) ; j ++ ) { arr [ indexes . get ( j ) ] = arr [ indexes . get ( j ) ] / x ; } res = res * x ; } else { x ++ ; } }
for ( int i = 0 ; i < n ; i ++ ) { res = res * arr [ i ] ; } return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 } ; int n = arr . length ; System . out . println ( LCM ( arr , n ) ) ; } }
import java . lang . Math ; public class Main {
static int politness ( int n ) { int count = 0 ;
for ( int i = 2 ; i <= Math . sqrt ( 2 * n ) ; i ++ ) { int a ; if ( ( 2 * n ) % i != 0 ) continue ; a = 2 * n ; a /= i ; a -= ( i - 1 ) ; if ( a % 2 != 0 ) continue ; a /= 2 ; if ( a > 0 ) { count ++ ; } } return count ; }
public static void main ( String [ ] args ) { int n = 90 ; System . out . println ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; n = 15 ; System . out . println ( " Politness ▁ of ▁ " + n + " ▁ = ▁ " + politness ( n ) ) ; } }
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
public static void main ( String [ ] args ) {
sieveSundaram ( ) ;
findPrimes ( 4 ) ; findPrimes ( 38 ) ; findPrimes ( 100 ) ; } }
import java . io . * ; import java . math . * ; class GFG {
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
public static void main ( String [ ] args ) {
int [ ] s = new int [ MAX + 1 ] ; sieveOfEratosthenes ( s ) ; int n = 12 , k = 3 ; System . out . println ( kPrimeFactor ( n , k , s ) ) ; n = 14 ; k = 3 ; System . out . println ( kPrimeFactor ( n , k , s ) ) ; } }
import java . util . HashMap ; class GFG {
public static int sumDivisorsOfDivisors ( int n ) {
HashMap < Integer , Integer > mp = new HashMap < > ( ) ; for ( int j = 2 ; j <= Math . sqrt ( n ) ; j ++ ) { int count = 0 ; while ( n % j == 0 ) { n /= j ; count ++ ; } if ( count != 0 ) mp . put ( j , count ) ; }
if ( n != 1 ) mp . put ( n , 1 ) ;
int ans = 1 ; for ( HashMap . Entry < Integer , Integer > entry : mp . entrySet ( ) ) { int pw = 1 ; int sum = 0 ; for ( int i = entry . getValue ( ) + 1 ; i >= 1 ; i -- ) { sum += ( i * pw ) ; pw = entry . getKey ( ) ; } ans *= sum ; } return ans ; }
public static void main ( String [ ] args ) { int n = 10 ; System . out . println ( sumDivisorsOfDivisors ( n ) ) ; } }
import java . io . * ; class GFG {
static int prime ( int n ) {
if ( n % 2 != 0 ) n -= 2 ; else n -- ; int i , j ; for ( i = n ; i >= 2 ; i -= 2 ) { if ( i % 2 == 0 ) continue ; for ( j = 3 ; j <= Math . sqrt ( i ) ; j += 2 ) { if ( i % j == 0 ) break ; } if ( j > Math . sqrt ( i ) ) return i ; }
return 2 ; }
public static void main ( String [ ] args ) { int n = 17 ; System . out . print ( prime ( n ) ) ; } }
import java . util . * ; class GFG {
static String fractionToDecimal ( int numr , int denr ) {
String res = " " ;
HashMap < Integer , Integer > mp = new HashMap < > ( ) ; mp . clear ( ) ;
int rem = numr % denr ;
while ( ( rem != 0 ) && ( ! mp . containsKey ( rem ) ) ) {
mp . put ( rem , res . length ( ) ) ;
rem = rem * 10 ;
int res_part = rem / denr ; res += String . valueOf ( res_part ) ;
rem = rem % denr ; } if ( rem == 0 ) return " " ; else if ( mp . containsKey ( rem ) ) return res . substring ( mp . get ( rem ) ) ; return " " ; }
public static void main ( String [ ] args ) { int numr = 50 , denr = 22 ; String res = fractionToDecimal ( numr , denr ) ; if ( res == " " ) System . out . print ( " No ▁ recurring ▁ sequence " ) ; else System . out . print ( " Recurring ▁ sequence ▁ is ▁ " + res ) ; } }
import java . io . * ; class GFG {
static int has0 ( int x ) {
while ( x != 0 ) {
if ( x % 10 == 0 ) return 1 ; x /= 10 ; } return 0 ; }
static int getCount ( int n ) {
int count = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) count += has0 ( i ) ; return count ; }
public static void main ( String args [ ] ) { int n = 107 ; System . out . println ( " Count ▁ of ▁ numbers ▁ from ▁ 1" + " ▁ to ▁ " + n + " ▁ is ▁ " + getCount ( n ) ) ; } }
class GFG {
static boolean squareRootExists ( int n , int p ) { n = n % p ;
for ( int x = 2 ; x < p ; x ++ ) if ( ( x * x ) % p == n ) return true ; return false ; }
public static void main ( String [ ] args ) { int p = 7 ; int n = 2 ; if ( squareRootExists ( n , p ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
import java . io . * ; class GFG {
static int Largestpower ( int n , int p ) {
int ans = 0 ;
while ( n > 0 ) { n /= p ; ans += n ; } return ans ; }
public static void main ( String [ ] args ) { int n = 10 ; int p = 3 ; System . out . println ( " ▁ The ▁ largest ▁ power ▁ of ▁ " + p + " ▁ that ▁ divides ▁ " + n + " ! ▁ is ▁ " + Largestpower ( n , p ) ) ; } }
class Factorial { int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
public static void main ( String args [ ] ) { Factorial obj = new Factorial ( ) ; int num = 5 ; System . out . println ( " Factorial ▁ of ▁ " + num + " ▁ is ▁ " + obj . factorial ( num ) ) ; } }
static boolean getBit ( int num , int i ) {
return ( ( num & ( 1 << i ) ) != 0 ) ; }
static int clearBit ( int num , int i ) {
int mask = ~ ( 1 << i ) ;
return num & mask ; }
import java . io . * ; class GFG {
public static void main ( String [ ] args ) {
int [ ] arr1 = { 1 , 2 , 3 } ;
int [ ] arr2 = { 1 , 2 , 3 } ;
int N = arr1 . length ;
int M = arr2 . length ;
Bitwise_AND_sum_i ( arr1 , arr2 , M , N ) ; }
static void Bitwise_AND_sum_i ( int arr1 [ ] , int arr2 [ ] , int M , int N ) {
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
System . out . print ( bitwise_AND_sum + " ▁ " ) ; } } }
import java . util . * ; class GFG {
static void FlipBits ( int n ) { for ( int bit = 0 ; bit < 32 ; bit ++ ) {
if ( ( n >> bit ) % 2 > 0 ) {
n = n ^ ( 1 << bit ) ; break ; } } System . out . print ( " The ▁ number ▁ after ▁ unsetting ▁ the " ) ; System . out . print ( " ▁ rightmost ▁ set ▁ bit ▁ " + n ) ; }
public static void main ( String [ ] args ) { int N = 12 ; FlipBits ( N ) ; } }
class GFG {
static int bitwiseAndOdd ( int n ) {
int result = 1 ;
for ( int i = 3 ; i <= n ; i = i + 2 ) { result = ( result & i ) ; } return result ; }
public static void main ( String [ ] args ) { int n = 10 ; System . out . println ( bitwiseAndOdd ( n ) ) ; } }
class GFG {
static int bitwiseAndOdd ( int n ) { return 1 ; }
public static void main ( String [ ] args ) { int n = 10 ; System . out . println ( bitwiseAndOdd ( n ) ) ; } }
class GFG {
public static int reverseBits ( int n ) { int rev = 0 ;
while ( n > 0 ) {
rev <<= 1 ;
if ( ( int ) ( n & 1 ) == 1 ) rev ^= 1 ;
n >>= 1 ; }
return rev ; }
public static void main ( String [ ] args ) { int n = 11 ; System . out . println ( reverseBits ( n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int countgroup ( int a [ ] , int n ) { int xs = 0 ; for ( int i = 0 ; i < n ; i ++ ) xs = xs ^ a [ i ] ;
if ( xs == 0 ) return ( 1 << ( n - 1 ) ) - 1 ; return 0 ; }
public static void main ( String args [ ] ) { int a [ ] = { 1 , 2 , 3 } ; int n = a . length ; System . out . println ( countgroup ( a , n ) ) ; } }
class GFG {
static int bitExtracted ( int number , int k , int p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int number = 171 , k = 5 , p = 2 ; System . out . println ( " The ▁ extracted ▁ number ▁ is ▁ " + bitExtracted ( number , k , p ) ) ; } }
class GFG { static int findMax ( int num ) { byte size_of_int = 4 ; int num_copy = num ;
int j = size_of_int * 8 - 1 ; int i = 0 ; while ( i < j ) {
int m = ( num_copy >> i ) & 1 ; int n = ( num_copy >> j ) & 1 ;
if ( m > n ) { int x = ( 1 << i 1 << j ) ; num = num ^ x ; } i ++ ; j -- ; } return num ; }
static public void main ( String [ ] args ) { int num = 4 ; System . out . println ( findMax ( num ) ) ; } }
class GFG {
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
class GFG {
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
class GFG {
static int composite ( int n ) { int flag = 0 ; int c = 0 ;
for ( int j = 1 ; j <= n ; j ++ ) { if ( n % j == 0 ) { c += 1 ; } }
if ( c >= 3 ) flag = 1 ; return flag ; }
static void odd_indices ( int arr [ ] , int n ) { int sum = 0 ;
for ( int k = 0 ; k < n ; k += 2 ) { int check = composite ( arr [ k ] ) ;
if ( check == 1 ) sum += arr [ k ] ; }
System . out . print ( sum + "NEW_LINE"); }
public static void main ( String [ ] args ) { int arr [ ] = { 13 , 5 , 8 , 16 , 25 } ; int n = arr . length ; odd_indices ( arr , n ) ; } }
import java . util . * ; class GFG {
public static void preprocess ( int p [ ] , int x [ ] , int y [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) p [ i ] = x [ i ] * x [ i ] + y [ i ] * y [ i ] ; Arrays . sort ( p ) ; }
public static int query ( int p [ ] , int n , int rad ) { int start = 0 , end = n - 1 ; while ( ( end - start ) > 1 ) { int mid = ( start + end ) / 2 ; double tp = Math . sqrt ( p [ mid ] ) ; if ( tp > ( rad * 1.0 ) ) end = mid - 1 ; else start = mid ; } double tp1 = Math . sqrt ( p [ start ] ) ; double tp2 = Math . sqrt ( p [ end ] ) ; if ( tp1 > ( rad * 1.0 ) ) return 0 ; else if ( tp2 <= ( rad * 1.0 ) ) return end + 1 ; else return start + 1 ; }
public static void main ( String [ ] args ) { int x [ ] = { 1 , 2 , 3 , - 1 , 4 } ; int y [ ] = { 1 , 2 , 3 , - 1 , 4 } ; int n = x . length ;
int p [ ] = new int [ n ] ; preprocess ( p , x , y , n ) ;
System . out . println ( query ( p , n , 3 ) ) ;
System . out . println ( query ( p , n , 32 ) ) ; } }
import java . util . * ; class GFG {
static int find_Numb_ways ( int n ) {
int odd_indices = n / 2 ;
int even_indices = ( n / 2 ) + ( n % 2 ) ;
int arr_odd = ( int ) Math . pow ( 4 , odd_indices ) ;
int arr_even = ( int ) Math . pow ( 5 , even_indices ) ;
return arr_odd * arr_even ; }
public static void main ( String [ ] args ) { int n = 4 ; System . out . print ( find_Numb_ways ( n ) ) ; } }
import java . util . * ; class GFG {
static boolean isSpiralSorted ( int [ ] arr , int n ) {
int start = 0 ;
int end = n - 1 ; while ( start < end ) {
if ( arr [ start ] > arr [ end ] ) { return false ; }
start ++ ;
if ( arr [ end ] > arr [ start ] ) { return false ; }
end -- ; } return true ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 10 , 14 , 20 , 18 , 12 , 5 } ; int N = arr . length ;
if ( isSpiralSorted ( arr , N ) != false ) System . out . print ( " YES " ) ; else System . out . print ( " NO " ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static void findWordsSameRow ( List < String > arr ) {
Map < Character , Integer > mp = new HashMap < Character , Integer > ( ) ; mp . put ( ' q ' , 1 ) ; mp . put ( ' w ' , 1 ) ; mp . put ( ' e ' , 1 ) ; mp . put ( ' r ' , 1 ) ; mp . put ( ' t ' , 1 ) ; mp . put ( ' y ' , 1 ) ; mp . put ( ' u ' , 1 ) ; mp . put ( ' i ' , 1 ) ; mp . put ( ' o ' , 1 ) ; mp . put ( ' p ' , 1 ) ; mp . put ( ' a ' , 2 ) ; mp . put ( ' s ' , 2 ) ; mp . put ( ' d ' , 2 ) ; mp . put ( ' f ' , 2 ) ; mp . put ( ' g ' , 2 ) ; mp . put ( ' h ' , 2 ) ; mp . put ( ' j ' , 2 ) ; mp . put ( ' k ' , 2 ) ; mp . put ( ' l ' , 2 ) ; mp . put ( ' z ' , 3 ) ; mp . put ( ' x ' , 3 ) ; mp . put ( ' c ' , 3 ) ; mp . put ( ' v ' , 3 ) ; mp . put ( ' b ' , 3 ) ; mp . put ( ' n ' , 3 ) ; mp . put ( ' m ' , 3 ) ;
for ( String word : arr ) {
if ( word . length ( ) != 0 ) {
boolean flag = true ;
int rowNum = mp . get ( Character . toLowerCase ( word . charAt ( 0 ) ) ) ;
int M = word . length ( ) ;
for ( int i = 1 ; i < M ; i ++ ) {
if ( mp . get ( Character . toLowerCase ( word . charAt ( i ) ) ) != rowNum ) {
flag = false ; break ; } }
if ( flag ) {
System . out . print ( word + " ▁ " ) ; } } } }
public static void main ( String [ ] args ) { List < String > words = Arrays . asList ( " Yeti " , " Had " , " GFG " , " comment " ) ; findWordsSameRow ( words ) ; } }
import java . util . * ; class GFG {
static int countSubsequece ( int a [ ] , int n ) { int i , j , k , l ;
int answer = 0 ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { for ( k = j + 1 ; k < n ; k ++ ) { for ( l = k + 1 ; l < n ; l ++ ) {
if ( a [ j ] == a [ l ] &&
a [ i ] == a [ k ] ) { answer ++ ; } } } } } return answer ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 2 , 3 , 2 , 1 , 3 , 2 } ; System . out . print ( countSubsequece ( a , 7 ) ) ; } }
import java . util . * ; class GFG {
static char minDistChar ( char [ ] s ) { int n = s . length ;
int [ ] first = new int [ 26 ] ; int [ ] last = new int [ 26 ] ;
for ( int i = 0 ; i < 26 ; i ++ ) { first [ i ] = - 1 ; last [ i ] = - 1 ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( first [ s [ i ] - ' a ' ] == - 1 ) { first [ s [ i ] - ' a ' ] = i ; }
last [ s [ i ] - ' a ' ] = i ; }
int min = Integer . MAX_VALUE ; char ans = '1' ;
for ( int i = 0 ; i < 26 ; i ++ ) {
if ( last [ i ] == first [ i ] ) continue ;
if ( min > last [ i ] - first [ i ] ) { min = last [ i ] - first [ i ] ; ans = ( char ) ( i + ' a ' ) ; } }
return ans ; }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ;
System . out . print ( minDistChar ( str . toCharArray ( ) ) ) ; } }
import java . util . * ; class GFG { static int n = 3 ; static class Pair { int first , second ; Pair ( int a , int b ) { first = a ; second = b ; } }
static int minSteps ( int arr [ ] [ ] ) {
boolean v [ ] [ ] = new boolean [ n ] [ n ] ;
Queue < Pair > q = new LinkedList < Pair > ( ) ;
q . add ( new Pair ( 0 , 0 ) ) ;
int depth = 0 ;
while ( q . size ( ) != 0 ) {
int x = q . size ( ) ; while ( x -- > 0 ) {
Pair y = q . peek ( ) ;
int i = y . first , j = y . second ; q . remove ( ) ;
if ( v [ i ] [ j ] ) continue ;
if ( i == n - 1 && j == n - 1 ) return depth ;
v [ i ] [ j ] = true ;
if ( i + arr [ i ] [ j ] < n ) q . add ( new Pair ( i + arr [ i ] [ j ] , j ) ) ; if ( j + arr [ i ] [ j ] < n ) q . add ( new Pair ( i , j + arr [ i ] [ j ] ) ) ; } depth ++ ; } return - 1 ; }
public static void main ( String args [ ] ) { int arr [ ] [ ] = { { 1 , 1 , 1 } , { 1 , 1 , 1 } , { 1 , 1 , 1 } } ; System . out . println ( minSteps ( arr ) ) ; } }
import java . io . * ; class GFG {
static int solve ( int [ ] a , int n ) { int max1 = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( Math . abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = Math . abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
static public void main ( String [ ] args ) { int [ ] arr = { - 1 , 2 , 3 , - 4 , - 10 , 22 } ; int size = arr . length ; System . out . println ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
import java . io . * ; class GFG {
static int solve ( int a [ ] , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return Math . abs ( min1 - max1 ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { - 1 , 2 , 3 , 4 , - 10 } ; int size = arr . length ; System . out . println ( " Largest ▁ gap ▁ is ▁ : ▁ " + solve ( arr , size ) ) ; } }
class GFG {
static void replaceOriginal ( String s , int n ) {
char r [ ] = new char [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) {
r [ i ] = s . charAt ( n - 1 - i ) ;
if ( s . charAt ( i ) != ' a ' && s . charAt ( i ) != ' e ' && s . charAt ( i ) != ' i ' && s . charAt ( i ) != ' o ' && s . charAt ( i ) != ' u ' ) { System . out . print ( r [ i ] ) ; } } System . out . println ( " " ) ; }
public static void main ( String [ ] args ) { String s = " geeksforgeeks " ; int n = s . length ( ) ; replaceOriginal ( s , n ) ; } }
import java . util . * ; class GFG {
static boolean sameStrings ( String str1 , String str2 ) { int N = str1 . length ( ) ; int M = str2 . length ( ) ;
if ( N != M ) { return false ; }
int [ ] a = new int [ 256 ] ; int [ ] b = new int [ 256 ] ;
for ( int i = 0 ; i < N ; i ++ ) { a [ str1 . charAt ( i ) - ' a ' ] ++ ; b [ str2 . charAt ( i ) - ' a ' ] ++ ; }
int i = 0 ; while ( i < 256 ) { if ( ( a [ i ] == 0 && b [ i ] == 0 ) || ( a [ i ] != 0 && b [ i ] != 0 ) ) { i ++ ; }
else { return false ; } }
Arrays . sort ( a ) ; Arrays . sort ( b ) ;
for ( i = 0 ; i < 256 ; i ++ ) {
if ( a [ i ] != b [ i ] ) return false ; }
return true ; }
public static void main ( String [ ] args ) { String S1 = " cabbba " , S2 = " abbccc " ; if ( sameStrings ( S1 , S2 ) ) System . out . print ( " YES " + "NEW_LINE"); else System . out . print ( " ▁ NO " + "NEW_LINE"); } }
import java . util . * ; class GFG {
public static int solution ( int A , int B , int C ) { int arr [ ] = new int [ 3 ] ;
arr [ 0 ] = A ; arr [ 1 ] = B ; arr [ 2 ] = C ;
Arrays . sort ( arr ) ;
if ( arr [ 2 ] < arr [ 0 ] + arr [ 1 ] ) return ( ( arr [ 0 ] + arr [ 1 ] + arr [ 2 ] ) / 2 ) ;
else return ( arr [ 0 ] + arr [ 1 ] ) ; }
public static void main ( String [ ] args ) {
int A = 8 , B = 1 , C = 5 ;
System . out . println ( solution ( A , B , C ) ) ; } }
class GFG {
static int search ( int arr [ ] , int l , int h , int key ) { if ( l > h ) return - 1 ; int mid = ( l + h ) / 2 ; if ( arr [ mid ] == key ) return mid ;
if ( ( arr [ l ] == arr [ mid ] ) && ( arr [ h ] == arr [ mid ] ) ) { l ++ ; h -- ; return search ( arr , l , h , key ) ; }
else if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
else return search ( arr , mid + 1 , h , key ) ; }
else if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 3 , 1 , 2 , 3 , 3 } ; int n = arr . length ; int key = 3 ; System . out . println ( search ( arr , 0 , n - 1 , key ) ) ; } }
import java . util . Collections ; import java . util . Vector ; class GFG {
public static String getSortedString ( StringBuilder s , int n ) {
Vector < Character > v1 = new Vector < > ( ) ; Vector < Character > v2 = new Vector < > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( s . charAt ( i ) >= ' a ' && s . charAt ( i ) <= ' z ' ) v1 . add ( s . charAt ( i ) ) ; if ( s . charAt ( i ) >= ' A ' && s . charAt ( i ) <= ' z ' ) v2 . add ( s . charAt ( i ) ) ; }
Collections . sort ( v1 ) ; Collections . sort ( v2 ) ; int i = 0 , j = 0 ; for ( int k = 0 ; k < n ; k ++ ) {
if ( s . charAt ( k ) > = ' a ' && s . charAt ( k ) <= ' z ' ) { s . setCharAt ( k , v1 . elementAt ( i ) ) ; ++ i ; }
else if ( s . charAt ( k ) > = ' A ' && s . charAt ( k ) <= ' Z ' ) { s . setCharAt ( k , v2 . elementAt ( j ) ) ; ++ j ; } }
return s . toString ( ) ; }
public static void main ( String [ ] args ) { StringBuilder s = new StringBuilder ( " gEeksfOrgEEkS " ) ; int n = s . length ( ) ; System . out . println ( getSortedString ( s , n ) ) ; } }
import java . util . * ; class GfG {
static boolean check ( char s [ ] ) {
int l = s . length ;
Arrays . sort ( s ) ;
for ( int i = 1 ; i < l ; i ++ ) {
if ( s [ i ] - s [ i - 1 ] != 1 ) return false ; } return true ; }
public static void main ( String [ ] args ) {
String str = " dcef " ; if ( check ( str . toCharArray ( ) ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ;
String str1 = " xyza " ; if ( check ( str1 . toCharArray ( ) ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static int minElements ( int arr [ ] , int n ) {
int halfSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = halfSum / 2 ;
Arrays . sort ( arr ) ; int res = 0 , curr_sum = 0 ; for ( int i = n - 1 ; i >= 0 ; i -- ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 1 , 7 , 1 } ; int n = arr . length ; System . out . println ( minElements ( arr , n ) ) ; } }
import java . util . * ; class GFG {
static void arrayElementEqual ( int arr [ ] , int N ) {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
if ( sum % N == 0 ) { System . out . print ( " Yes " ) ; }
else { System . out . print ( " No " + "NEW_LINE"); } }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 5 , 6 , 4 } ;
int N = arr . length ; arrayElementEqual ( arr , N ) ; } }
import java . util . * ; class GFG {
static int findMaxValByRearrArr ( int arr [ ] , int N ) {
int res = 0 ;
res = ( N * ( N + 1 ) ) / 2 ; return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 2 , 1 } ; int N = arr . length ; System . out . print ( findMaxValByRearrArr ( arr , N ) ) ; } }
import java . util . * ; class GFG {
static int MaximumSides ( int n ) {
if ( n < 4 ) return - 1 ;
return n % 2 == 0 ? n / 2 : - 1 ; }
public static void main ( String [ ] args ) {
int N = 8 ;
System . out . print ( MaximumSides ( N ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static float pairProductMean ( int arr [ ] , int N ) {
int suffixSumArray [ ] = new int [ N ] ; suffixSumArray [ N - 1 ] = arr [ N - 1 ] ;
for ( int i = N - 2 ; i >= 0 ; i -- ) { suffixSumArray [ i ] = suffixSumArray [ i + 1 ] + arr [ i ] ; }
int length = ( N * ( N - 1 ) ) / 2 ;
float res = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) { res += arr [ i ] * suffixSumArray [ i + 1 ] ; }
float mean ;
if ( length != 0 ) mean = res / length ; else mean = 0 ;
return mean ; }
public static void main ( String [ ] args ) {
int arr [ ] = { 1 , 2 , 4 , 8 } ; int N = arr . length ;
System . out . format ( " % .2f " , pairProductMean ( arr , N ) ) ; } }
import java . util . * ; class GFG {
static int ncr ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static int countPath ( int N , int M , int K ) { int answer ; if ( K >= 2 ) answer = 0 ; else if ( K == 0 ) answer = ncr ( N + M - 2 , N - 1 ) ; else {
answer = ncr ( N + M - 2 , N - 1 ) ;
int X = ( N - 1 ) / 2 + ( M - 1 ) / 2 ; int Y = ( N - 1 ) / 2 ; int midCount = ncr ( X , Y ) ;
X = ( ( N - 1 ) - ( N - 1 ) / 2 ) + ( ( M - 1 ) - ( M - 1 ) / 2 ) ; Y = ( ( N - 1 ) - ( N - 1 ) / 2 ) ; midCount *= ncr ( X , Y ) ; answer -= midCount ; } return answer ; }
public static void main ( String [ ] args ) { int N = 3 ; int M = 3 ; int K = 1 ; System . out . print ( countPath ( N , M , K ) ) ; } }
import java . util . * ; class GFG { static class pair { int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static int find_max ( Vector < pair > v , int n ) {
int count = 0 ; if ( n >= 2 ) count = 2 ; else count = 1 ;
for ( int i = 1 ; i < n - 1 ; i ++ ) {
if ( v . get ( i - 1 ) . first < ( v . get ( i ) . first - v . get ( i ) . second ) ) count ++ ;
else if ( v . get ( i + 1 ) . first > ( v . get ( i ) . first + v . get ( i ) . second ) ) { count ++ ; v . get ( i ) . first = v . get ( i ) . first + v . get ( i ) . second ; }
else continue ; }
return count ; }
public static void main ( String [ ] args ) { int n = 3 ; Vector < pair > v = new Vector < > ( ) ; v . add ( new pair ( 10 , 20 ) ) ; v . add ( new pair ( 15 , 10 ) ) ; v . add ( new pair ( 20 , 16 ) ) ; System . out . print ( find_max ( v , n ) ) ; } }
import java . util . Arrays ; class GFG {
public static void numberofsubstrings ( String str , int k , char charArray [ ] ) { int N = str . length ( ) ;
int available [ ] = new int [ 26 ] ; Arrays . fill ( available , 0 ) ;
for ( int i = 0 ; i < k ; i ++ ) { available [ charArray [ i ] - ' a ' ] = 1 ; }
int lastPos = - 1 ;
int ans = ( N * ( N + 1 ) ) / 2 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( available [ str . charAt ( i ) - ' a ' ] == 0 ) {
ans -= ( ( i - lastPos ) * ( N - i ) ) ;
lastPos = i ; } }
System . out . println ( ans ) ; }
public static void main ( String args [ ] ) {
String str = " abcb " ; int k = 2 ;
char [ ] charArray = { ' a ' , ' b ' } ;
numberofsubstrings ( str , k , charArray ) ; } }
class GFG {
static int minCost ( int N , int P , int Q ) {
int cost = 0 ;
while ( N > 0 ) { if ( ( N & 1 ) > 0 ) { cost += P ; N -- ; } else { int temp = N / 2 ;
if ( temp * P > Q ) cost += Q ;
else cost += P * temp ; N /= 2 ; } }
return cost ; }
public static void main ( String [ ] args ) { int N = 9 , P = 5 , Q = 1 ; System . out . println ( minCost ( N , P , Q ) ) ; } }
class GFG {
static void numberOfWays ( int n , int k ) {
int [ ] dp = new int [ 1000 ] ;
for ( int i = 0 ; i < n ; i ++ ) { dp [ i ] = 0 ; }
dp [ 0 ] = 1 ;
for ( int i = 1 ; i <= k ; i ++ ) {
int numWays = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { numWays += dp [ j ] ; }
for ( int j = 0 ; j < n ; j ++ ) { dp [ j ] = numWays - dp [ j ] ; } }
System . out . println ( dp [ 0 ] + "NEW_LINE"); }
public static void main ( String args [ ] ) {
int N = 5 , K = 3 ;
numberOfWays ( N , K ) ; } }
import java . io . * ; class GFG { static int M = 1000000007 ; static int waysOfDecoding ( String s ) { long first = 1 , second = s . charAt ( 0 ) == ' * ' ? 9 : s . charAt ( 0 ) == '0' ? 0 : 1 ; for ( int i = 1 ; i < s . length ( ) ; i ++ ) { long temp = second ;
if ( s . charAt ( i ) == ' * ' ) { second = 9 * second ;
if ( s . charAt ( i - 1 ) == '1' ) second = ( second + 9 * first ) % M ;
else if ( s . charAt ( i - 1 ) == '2' ) second = ( second + 6 * first ) % M ;
else if ( s . charAt ( i - 1 ) == ' * ' ) second = ( second + 15 * first ) % M ; }
else { second = s . charAt ( i ) != '0' ? second : 0 ;
if ( s . charAt ( i - 1 ) == '1' ) second = ( second + first ) % M ;
else if ( s . charAt ( i - 1 ) == '2' && s . charAt ( i ) <= '6' ) second = ( second + first ) % M ;
else if ( s . charAt ( i - 1 ) == ' * ' ) second = ( second + ( s . charAt ( i ) <= '6' ? 2 : 1 ) * first ) % M ; } first = temp ; } return ( int ) second ; }
public static void main ( String [ ] args ) { String s = " * " ; System . out . println ( waysOfDecoding ( s ) ) ; } }
class GFG {
static int findMinCost ( int [ ] [ ] arr , int X , int n , int i ) {
if ( X <= 0 ) return 0 ; if ( i >= n ) return Integer . MAX_VALUE ;
int inc = findMinCost ( arr , X - arr [ i ] [ 0 ] , n , i + 1 ) ; if ( inc != Integer . MAX_VALUE ) inc += arr [ i ] [ 1 ] ;
int exc = findMinCost ( arr , X , n , i + 1 ) ;
return Math . min ( inc , exc ) ; }
public static void main ( String [ ] args ) {
int [ ] [ ] arr = { { 4 , 3 } , { 3 , 2 } , { 2 , 4 } , { 1 , 3 } , { 4 , 2 } } ; int X = 7 ;
int n = arr . length ; int ans = findMinCost ( arr , X , n , 0 ) ;
if ( ans != Integer . MAX_VALUE ) System . out . println ( ans ) ; else System . out . println ( - 1 ) ; } }
import java . util . * ; class GFG {
static double find ( int N , int sum ) {
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return 1.0 / 6 ; else return 0 ; } double s = 0 ; for ( int i = 1 ; i <= 6 ; i ++ ) s = s + find ( N - 1 , sum - i ) / 6 ; return s ; }
public static void main ( String [ ] args ) { int N = 4 , a = 13 , b = 17 ; double probability = 0.0 ; for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
System . out . format ( " % .6f " , probability ) ; } }
class GFG {
static int minDays ( int n ) {
if ( n < 1 ) return n ;
int cnt = 1 + Math . min ( n % 2 + minDays ( n / 2 ) , n % 3 + minDays ( n / 3 ) ) ;
return cnt ; }
public static void main ( String [ ] args ) {
int N = 6 ;
System . out . print ( minDays ( N ) ) ; } }
