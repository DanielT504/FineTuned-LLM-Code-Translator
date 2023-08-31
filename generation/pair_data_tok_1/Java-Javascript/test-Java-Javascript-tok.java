static int minSum ( int A [ ] , int N ) {
HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ; int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += A [ i ] ;
if ( mp . containsKey ( A [ i ] ) ) { mp . put ( A [ i ] , mp . get ( A [ i ] ) + 1 ) ; } else { mp . put ( A [ i ] , 1 ) ; } }
int minSum = Integer . MAX_VALUE ;
for ( Map . Entry < Integer , Integer > it : mp . entrySet ( ) ) {
minSum = Math . min ( minSum , sum - ( it . getKey ( ) * it . getValue ( ) ) ) ; }
return minSum ; }
int arr [ ] = { 4 , 5 , 6 , 6 } ;
int N = arr . length ; System . out . print ( minSum ( arr , N ) + "NEW_LINE"); } }
static void maxAdjacent ( int [ ] arr , int N ) { ArrayList < Integer > res = new ArrayList < Integer > ( ) ;
for ( int i = 1 ; i < N - 1 ; i ++ ) { int prev = arr [ 0 ] ;
int maxi = Integer . MIN_VALUE ;
for ( int j = 1 ; j < N ; j ++ ) {
if ( i == j ) continue ;
maxi = Math . max ( maxi , Math . abs ( arr [ j ] - prev ) ) ;
prev = arr [ j ] ; }
res . add ( maxi ) ; }
for ( int x : res ) { System . out . print ( x + " ▁ " ) ; } System . out . println ( ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 3 , 4 , 7 , 8 } ; int N = arr . length ; maxAdjacent ( arr , N ) ; } }
static int findSize ( int N ) {
if ( N == 0 ) return 1 ; if ( N == 1 ) return 1 ; int Size = 2 * findSize ( N / 2 ) + 1 ;
return Size ; }
static int CountOnes ( int N , int L , int R ) { if ( L > R ) { return 0 ; }
if ( N <= 1 ) { return N ; } int ret = 0 ; int M = N / 2 ; int Siz_M = findSize ( M ) ;
if ( L <= Siz_M ) {
ret += CountOnes ( N / 2 , L , Math . min ( Siz_M , R ) ) ; }
if ( L <= Siz_M + 1 && Siz_M + 1 <= R ) { ret += N % 2 ; }
if ( Siz_M + 1 < R ) { ret += CountOnes ( N / 2 , Math . max ( 1 , L - Siz_M - 1 ) , R - Siz_M - 1 ) ; } return ret ; }
int N = 7 , L = 2 , R = 5 ;
System . out . println ( CountOnes ( N , L , R ) ) ; } }
static boolean prime ( int n ) {
if ( n == 1 ) return false ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; }
return true ; }
static void minDivisior ( int n ) {
if ( prime ( n ) ) { System . out . print ( 1 + " ▁ " + ( n - 1 ) ) ; }
else { for ( int i = 2 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
System . out . print ( n / i + " ▁ " + ( n / i * ( i - 1 ) ) ) ; break ; } } } }
public static void main ( String [ ] args ) { int N = 4 ;
minDivisior ( N ) ; } }
static int Landau = Integer . MIN_VALUE ;
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static int lcm ( int a , int b ) { return ( a * b ) / gcd ( a , b ) ; }
static void findLCM ( Vector < Integer > arr ) { int nth_lcm = arr . get ( 0 ) ; for ( int i = 1 ; i < arr . size ( ) ; i ++ ) nth_lcm = lcm ( nth_lcm , arr . get ( i ) ) ;
Landau = Math . max ( Landau , nth_lcm ) ; }
static void findWays ( Vector < Integer > arr , int i , int n ) {
if ( n == 0 ) findLCM ( arr ) ;
for ( int j = i ; j <= n ; j ++ ) {
arr . add ( j ) ;
findWays ( arr , j , n - j ) ;
arr . remove ( arr . size ( ) - 1 ) ; } }
static void Landau_function ( int n ) { Vector < Integer > arr = new Vector < > ( ) ;
findWays ( arr , 1 , n ) ;
System . out . print ( Landau ) ; }
int N = 4 ;
Landau_function ( N ) ; } }
static boolean isPrime ( int n ) {
if ( n == 1 ) return true ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ;
for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static void checkExpression ( int n ) { if ( isPrime ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; }
public static void main ( String [ ] args ) { int N = 3 ; checkExpression ( N ) ; } }
static boolean checkArray ( int n , int k , int arr [ ] ) {
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( ( arr [ i ] & 1 ) != 0 ) cnt += 1 ; }
if ( cnt >= k && cnt % 2 == k % 2 ) return true ; else return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 4 , 7 , 5 , 3 , 1 } ; int n = arr . length ; int k = 4 ; if ( checkArray ( n , k , arr ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static long func ( int arr [ ] , int n ) { double ans = 0 ; int maxx = 0 ; double freq [ ] = new double [ 100005 ] ; int temp ;
for ( int i = 0 ; i < n ; i ++ ) { temp = arr [ i ] ; freq [ temp ] ++ ; maxx = Math . max ( maxx , temp ) ; }
for ( int i = 1 ; i <= maxx ; i ++ ) { freq [ i ] += freq [ i - 1 ] ; } for ( int i = 1 ; i <= maxx ; i ++ ) { if ( freq [ i ] != 0 ) { double j ;
double cur = Math . ceil ( 0.5 * i ) - 1.0 ; for ( j = 1.5 ; ; j ++ ) { int val = Math . min ( maxx , ( int ) ( Math . ceil ( i * j ) - 1.0 ) ) ; int times = ( int ) ( freq [ i ] - freq [ i - 1 ] ) , con = ( int ) ( j - 0.5 ) ;
ans += times * con * ( freq [ ( int ) val ] - freq [ ( int ) cur ] ) ; cur = val ; if ( val == maxx ) break ; } } }
return ( long ) ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 } ; int n = arr . length ; System . out . print ( func ( arr , n ) + "NEW_LINE"); } }
static void insert_element ( int a [ ] , int n ) {
int Xor = 0 ;
int Sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { Xor ^= a [ i ] ; Sum += a [ i ] ; }
if ( Sum == 2 * Xor ) {
System . out . println ( "0" ) ; return ; }
if ( Xor == 0 ) { System . out . println ( "1" ) ; System . out . println ( Sum ) ; return ; }
int num1 = Sum + Xor ; int num2 = Xor ;
System . out . print ( "2" ) ;
System . out . println ( num1 + " ▁ " + num2 ) ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 3 } ; int n = a . length ; insert_element ( a , n ) ; } }
static void checkSolution ( int a , int b , int c ) { if ( a == c ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; }
public static void main ( String [ ] args ) { int a = 2 , b = 0 , c = 2 ; checkSolution ( a , b , c ) ; } }
static boolean isPerfectSquare ( double x ) {
double sr = Math . sqrt ( x ) ;
return ( ( sr - Math . floor ( sr ) ) == 0 ) ; }
static void checkSunnyNumber ( int N ) {
if ( isPerfectSquare ( N + 1 ) ) { System . out . println ( " Yes " ) ; }
else { System . out . println ( " No " ) ; } }
int N = 8 ;
checkSunnyNumber ( N ) ; } }
static int countValues ( int n ) { int answer = 0 ;
for ( int i = 2 ; i <= n ; i ++ ) { int k = n ;
while ( k >= i ) { if ( k % i == 0 ) k /= i ; else k -= i ; }
if ( k == 1 ) answer ++ ; } return answer ; }
public static void main ( String args [ ] ) { int N = 6 ; System . out . print ( countValues ( N ) ) ; } }
static void printKNumbers ( int N , int K ) {
for ( int i = 0 ; i < K - 1 ; i ++ ) System . out . print ( 1 + " ▁ " ) ;
System . out . print ( N - K + 1 ) ; }
public static void main ( String [ ] args ) { int N = 10 , K = 3 ; printKNumbers ( N , K ) ; } }
static int NthSmallest ( int K ) {
Queue < Integer > Q = new LinkedList < > ( ) ; int x = 0 ;
for ( int i = 1 ; i < 10 ; i ++ ) Q . add ( i ) ;
for ( int i = 1 ; i <= K ; i ++ ) {
x = Q . peek ( ) ;
Q . remove ( ) ;
if ( x % 10 != 0 ) {
Q . add ( x * 10 + x % 10 - 1 ) ; }
Q . add ( x * 10 + x % 10 ) ;
if ( x % 10 != 9 ) {
Q . add ( x * 10 + x % 10 + 1 ) ; } }
return x ; }
int N = 16 ; System . out . print ( NthSmallest ( N ) ) ; } }
static int nearest ( int n ) {
int prevSquare = ( int ) Math . sqrt ( n ) ; int nextSquare = prevSquare + 1 ; prevSquare = prevSquare * prevSquare ; nextSquare = nextSquare * nextSquare ;
int ans = ( n - prevSquare ) < ( nextSquare - n ) ? ( prevSquare - n ) : ( nextSquare - n ) ;
return ans ; }
public static void main ( String [ ] args ) { int n = 14 ; System . out . println ( nearest ( n ) ) ; n = 16 ; System . out . println ( nearest ( n ) ) ; n = 18 ; System . out . println ( nearest ( n ) ) ; } }
static void printValueOfPi ( int N ) {
double pi = 2 * Math . acos ( 0.0 ) ;
System . out . println ( pi ) ; }
public static void main ( String [ ] args ) { int N = 4 ;
printValueOfPi ( N ) ; } }
static void decBinary ( int arr [ ] , int n ) { int k = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n /= 2 ; } }
static int binaryDec ( int arr [ ] , int n ) { int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
static int getNum ( int n , int k ) {
int l = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) + 1 ;
int a [ ] = new int [ l ] ; decBinary ( a , n ) ;
if ( k > l ) return n ;
a [ k - 1 ] = ( a [ k - 1 ] == 0 ) ? 1 : 0 ;
return binaryDec ( a , l ) ; }
public static void main ( String [ ] args ) { int n = 56 ; int k = 2 ; System . out . println ( getNum ( n , k ) ) ; } }
import java . io . * ; class GFG { static int MAX = 1000000 ; static int MOD = 10000007 ;
static int [ ] result = new int [ MAX + 1 ] ; static int [ ] fact = new int [ MAX + 1 ] ;
static void preCompute ( ) {
fact [ 0 ] = 1 ; result [ 0 ] = 1 ;
for ( int i = 1 ; i <= MAX ; i ++ ) {
fact [ i ] = ( ( fact [ i - 1 ] % MOD ) * i ) % MOD ;
result [ i ] = ( ( result [ i - 1 ] % MOD ) * ( fact [ i ] % MOD ) ) % MOD ; } }
static void performQueries ( int q [ ] , int n ) {
preCompute ( ) ;
for ( int i = 0 ; i < n ; i ++ ) System . out . println ( result [ q [ i ] ] ) ; }
public static void main ( String [ ] args ) { int q [ ] = { 4 , 5 } ; int n = q . length ; performQueries ( q , n ) ; } }
static long gcd ( long a , long b ) { if ( a == 0 ) { return b ; } return gcd ( b % a , a ) ; }
static long divTermCount ( long a , long b , long c , long num ) {
return ( ( num / a ) + ( num / b ) + ( num / c ) - ( num / ( ( a * b ) / gcd ( a , b ) ) ) - ( num / ( ( c * b ) / gcd ( c , b ) ) ) - ( num / ( ( a * c ) / gcd ( a , c ) ) ) + ( num / ( ( a * b * c ) / gcd ( gcd ( a , b ) , c ) ) ) ) ; }
static long findNthTerm ( int a , int b , int c , long n ) {
long low = 1 , high = Long . MAX_VALUE , mid ; while ( low < high ) { mid = low + ( high - low ) / 2 ;
if ( divTermCount ( a , b , c , mid ) < n ) { low = mid + 1 ; }
else { high = mid ; } } return low ; }
public static void main ( String args [ ] ) { int a = 2 , b = 3 , c = 5 , n = 100 ; System . out . println ( findNthTerm ( a , b , c , n ) ) ; } }
static double calculate_angle ( int n , int i , int j , int k ) {
int x , y ;
if ( i < j ) x = j - i ; else x = j + n - i ; if ( j < k ) y = k - j ; else y = k + n - j ;
double ang1 = ( 180 * x ) / n ; double ang2 = ( 180 * y ) / n ;
double ans = 180 - ang1 - ang2 ; return ans ; }
public static void main ( String [ ] args ) { int n = 5 ; int a1 = 1 ; int a2 = 2 ; int a3 = 5 ; System . out . println ( ( int ) calculate_angle ( n , a1 , a2 , a3 ) ) ; } }
static void Loss ( int SP , int P ) { float loss = 0 ; loss = ( float ) ( 2 * P * P * SP ) / ( 100 * 100 - P * P ) ; System . out . println ( " Loss ▁ = ▁ " + loss ) ; }
public static void main ( String [ ] args ) { int SP = 2400 , P = 30 ;
Loss ( SP , P ) ; } }
class GFG { static int MAXN = 1000001 ;
static int [ ] spf = new int [ MAXN ] ;
static int [ ] hash1 = new int [ MAXN ] ;
static void sieve ( ) { spf [ 1 ] = 1 ; for ( int i = 2 ; i < MAXN ; i ++ )
spf [ i ] = i ;
for ( int i = 4 ; i < MAXN ; i += 2 ) spf [ i ] = 2 ;
for ( int i = 3 ; i * i < MAXN ; i ++ ) {
if ( spf [ i ] == i ) { for ( int j = i * i ; j < MAXN ; j += i )
if ( spf [ j ] == j ) spf [ j ] = i ; } } }
static void getFactorization ( int x ) { int temp ; while ( x != 1 ) { temp = spf [ x ] ; if ( x % temp == 0 ) {
hash1 [ spf [ x ] ] ++ ; x = x / spf [ x ] ; } while ( x % temp == 0 ) x = x / temp ; } }
static boolean check ( int x ) { int temp ; while ( x != 1 ) { temp = spf [ x ] ;
if ( x % temp == 0 && hash1 [ temp ] > 1 ) return false ; while ( x % temp == 0 ) x = x / temp ; } return true ; }
static boolean hasValidNum ( int [ ] arr , int n ) {
sieve ( ) ; for ( int i = 0 ; i < n ; i ++ ) getFactorization ( arr [ i ] ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( check ( arr [ i ] ) ) return true ; return false ; }
public static void main ( String [ ] args ) { int [ ] arr = { 2 , 8 , 4 , 10 , 6 , 7 } ; int n = arr . length ; if ( hasValidNum ( arr , n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int countWays ( int N ) {
int E = ( N * ( N - 1 ) ) / 2 ; if ( N == 1 ) return 0 ; return ( int ) Math . pow ( 2 , E - 1 ) ; }
public static void main ( String [ ] args ) { int N = 4 ; System . out . println ( countWays ( N ) ) ; } }
static int l [ ] [ ] = new int [ 1001 ] [ 1001 ] ; static void initialize ( ) {
l [ 0 ] [ 0 ] = 1 ; for ( int i = 1 ; i < 1001 ; i ++ ) {
l [ i ] [ 0 ] = 1 ; for ( int j = 1 ; j < i + 1 ; j ++ ) {
l [ i ] [ j ] = ( l [ i - 1 ] [ j - 1 ] + l [ i - 1 ] [ j ] ) ; } } }
static int nCr ( int n , int r ) {
return l [ n ] [ r ] ; }
initialize ( ) ; int n = 8 ; int r = 3 ; System . out . println ( nCr ( n , r ) ) ; } }
static int minAbsDiff ( int n ) { int mod = n % 4 ; if ( mod == 0 mod == 3 ) { return 0 ; } return 1 ; }
public static void main ( String [ ] args ) { int n = 5 ; System . out . println ( minAbsDiff ( n ) ) ; } }
class GFG { static boolean check ( int s ) {
int [ ] freq = new int [ 10 ] ; int r , i ; for ( i = 0 ; i < 10 ; i ++ ) { freq [ i ] = 0 ; } while ( s != 0 ) {
r = s % 10 ;
s = ( int ) ( s / 10 ) ;
freq [ r ] += 1 ; } int xor__ = 0 ;
for ( i = 0 ; i < 10 ; i ++ ) { xor__ = xor__ ^ freq [ i ] ; if ( xor__ == 0 ) return true ; else return false ; } return true ; }
public static void main ( String [ ] args ) { int s = 122233 ; if ( check ( s ) ) System . out . println ( "YesNEW_LINE"); else System . out . println ( "NoNEW_LINE"); } }
static void printLines ( int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) { System . out . println ( k * ( 6 * i + 1 ) + " ▁ " + k * ( 6 * i + 2 ) + " ▁ " + k * ( 6 * i + 3 ) + " ▁ " + k * ( 6 * i + 5 ) ) ; } }
public static void main ( String args [ ] ) { int n = 2 , k = 2 ; printLines ( n , k ) ; } }
import java . util . * ; class GFG { static int calculateSum ( int n ) {
return ( ( int ) Math . pow ( 2 , n + 1 ) + n - 2 ) ; }
int n = 4 ;
System . out . println ( " Sum ▁ = ▁ " + calculateSum ( n ) ) ; } }
import java . util . * ; class GFG { static final int mod = 1000000007 ;
static int count_special ( int n ) {
int [ ] fib = new int [ n + 1 ] ;
fib [ 0 ] = 1 ;
fib [ 1 ] = 2 ; for ( int i = 2 ; i <= n ; i ++ ) {
fib [ i ] = ( fib [ i - 1 ] % mod + fib [ i - 2 ] % mod ) % mod ; }
return fib [ n ] ; }
int n = 3 ; System . out . print ( count_special ( n ) + "NEW_LINE"); } }
import java . io . * ; class GFG { static int mod = 1000000000 ;
static int ways ( int i , int arr [ ] , int n ) {
if ( i == n - 1 ) return 1 ; int sum = 0 ;
for ( int j = 1 ; j + i < n && j <= arr [ i ] ; j ++ ) { sum += ( ways ( i + j , arr , n ) ) % mod ; sum %= mod ; } return sum % mod ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 3 , 1 , 4 , 3 } ; int n = arr . length ; System . out . println ( ways ( 0 , arr , n ) ) ; } }
class GFG { static final int mod = ( int ) ( 1e9 + 7 ) ;
static int ways ( int arr [ ] , int n ) {
int dp [ ] = new int [ n + 1 ] ;
dp [ n - 1 ] = 1 ;
for ( int i = n - 2 ; i >= 0 ; i -- ) { dp [ i ] = 0 ;
for ( int j = 1 ; ( ( j + i ) < n && j <= arr [ i ] ) ; j ++ ) { dp [ i ] += dp [ i + j ] ; dp [ i ] %= mod ; } }
return dp [ 0 ] % mod ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 3 , 1 , 4 , 3 } ; int n = arr . length ; System . out . println ( ways ( arr , n ) % mod ) ; } }
class GFG { static class pair { int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static pair countSum ( int arr [ ] , int n ) { int result = 0 ;
int count_odd , count_even ;
count_odd = 0 ; count_even = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) {
if ( arr [ i - 1 ] % 2 == 0 ) { count_even = count_even + count_even + 1 ; count_odd = count_odd + count_odd ; }
else { int temp = count_even ; count_even = count_even + count_odd ; count_odd = count_odd + temp + 1 ; } } return new pair ( count_even , count_odd ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 2 , 3 } ; int n = arr . length ;
pair ans = countSum ( arr , n ) ; System . out . print ( " EvenSum ▁ = ▁ " + ans . first ) ; System . out . print ( " ▁ OddSum ▁ = ▁ " + ans . second ) ; } }
import java . util . * ; class GFG { static int MAX = 10 ;
static Vector < Integer > numToVec ( int N ) { Vector < Integer > digit = new Vector < Integer > ( ) ;
while ( N != 0 ) { digit . add ( N % 10 ) ; N = N / 10 ; }
if ( digit . size ( ) == 0 ) digit . add ( 0 ) ;
Collections . reverse ( digit ) ;
return digit ; }
static int solve ( Vector < Integer > A , int B , int C ) { Vector < Integer > digit = new Vector < Integer > ( ) ; int d , d2 ;
digit = numToVec ( C ) ; d = A . size ( ) ;
if ( B > digit . size ( ) d == 0 ) return 0 ;
else if ( B < digit . size ( ) ) {
if ( A . get ( 0 ) == 0 && B != 1 ) return ( int ) ( ( d - 1 ) * Math . pow ( d , B - 1 ) ) ; else return ( int ) Math . pow ( d , B ) ; }
else { int [ ] dp = new int [ B + 1 ] ; int [ ] lower = new int [ MAX + 1 ] ;
for ( int i = 0 ; i < d ; i ++ ) lower [ A . get ( i ) + 1 ] = 1 ; for ( int i = 1 ; i <= MAX ; i ++ ) lower [ i ] = lower [ i - 1 ] + lower [ i ] ; boolean flag = true ; dp [ 0 ] = 0 ; for ( int i = 1 ; i <= B ; i ++ ) { d2 = lower [ digit . get ( i - 1 ) ] ; dp [ i ] = dp [ i - 1 ] * d ;
if ( i == 1 && A . get ( 0 ) == 0 && B != 1 ) d2 = d2 - 1 ;
if ( flag ) dp [ i ] += d2 ;
flag = ( flag & ( lower [ digit . get ( i - 1 ) + 1 ] == lower [ digit . get ( i - 1 ) ] + 1 ) ) ; } return dp [ B ] ; } }
public static void main ( String [ ] args ) { Integer arr [ ] = { 0 , 1 , 2 , 5 } ; Vector < Integer > A = new Vector < > ( Arrays . asList ( arr ) ) ; int N = 2 ; int k = 21 ; System . out . println ( solve ( A , N , k ) ) ; } }
public static int solve ( int [ ] [ ] dp , int wt , int K , int M , int used ) {
if ( wt < 0 ) { return 0 ; } if ( wt == 0 ) {
if ( used == 1 ) { return 1 ; } return 0 ; } if ( dp [ wt ] [ used ] != - 1 ) { return dp [ wt ] [ used ] ; } int ans = 0 ; for ( int i = 1 ; i <= K ; i ++ ) {
if ( i >= M ) { ans += solve ( dp , wt - i , K , M , used 1 ) ; } else { ans += solve ( dp , wt - i , K , M , used ) ; } } return dp [ wt ] [ used ] = ans ; }
public static void main ( String [ ] args ) { int W = 3 , K = 3 , M = 2 ; int [ ] [ ] dp = new int [ W + 1 ] [ 2 ] ; for ( int i = 0 ; i < W + 1 ; i ++ ) { for ( int j = 0 ; j < 2 ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } System . out . print ( solve ( dp , W , K , M , 0 ) + "NEW_LINE"); } }
static long partitions ( int n ) { long p [ ] = new long [ n + 1 ] ;
p [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; ++ i ) { int k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 != 0 ? 1 : - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) { k *= - 1 ; } else { k = 1 - k ; } } } return p [ n ] ; }
public static void main ( String [ ] args ) { int N = 20 ; System . out . println ( partitions ( N ) ) ; } }
static int LIP ( int dp [ ] [ ] , int mat [ ] [ ] , int n , int m , int x , int y ) {
if ( dp [ x ] [ y ] < 0 ) { int result = 0 ;
if ( x == n - 1 && y == m - 1 ) return dp [ x ] [ y ] = 1 ;
if ( x == n - 1 y == m - 1 ) result = 1 ;
if ( x + 1 < n && mat [ x ] [ y ] < mat [ x + 1 ] [ y ] ) result = 1 + LIP ( dp , mat , n , m , x + 1 , y ) ;
if ( y + 1 < m && mat [ x ] [ y ] < mat [ x ] [ y + 1 ] ) result = Math . max ( result , 1 + LIP ( dp , mat , n , m , x , y + 1 ) ) ; dp [ x ] [ y ] = result ; } return dp [ x ] [ y ] ; }
static int wrapper ( int mat [ ] [ ] , int n , int m ) { int dp [ ] [ ] = new int [ 10 ] [ 10 ] ; for ( int i = 0 ; i < 10 ; i ++ ) Arrays . fill ( dp [ i ] , - 1 ) ; return LIP ( dp , mat , n , m , 0 , 0 ) ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 2 , 3 , 4 } , { 2 , 2 , 3 , 4 } , { 3 , 2 , 3 , 4 } , { 4 , 5 , 6 , 7 } , } ; int n = 4 , m = 4 ; System . out . println ( wrapper ( mat , n , m ) ) ; } }
static int countPaths ( int n , int m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
public static void main ( String [ ] args ) { int n = 3 , m = 2 ; System . out . println ( " ▁ Number ▁ of ▁ Paths ▁ " + countPaths ( n , m ) ) ; } }
import java . util . Arrays ; class GFG { static final int MAX = 100 ;
static int getMaxGold ( int gold [ ] [ ] , int m , int n ) {
int goldTable [ ] [ ] = new int [ m ] [ n ] ; for ( int [ ] rows : goldTable ) Arrays . fill ( rows , 0 ) ; for ( int col = n - 1 ; col >= 0 ; col -- ) { for ( int row = 0 ; row < m ; row ++ ) {
int right = ( col == n - 1 ) ? 0 : goldTable [ row ] [ col + 1 ] ;
int right_up = ( row == 0 col == n - 1 ) ? 0 : goldTable [ row - 1 ] [ col + 1 ] ;
int right_down = ( row == m - 1 col == n - 1 ) ? 0 : goldTable [ row + 1 ] [ col + 1 ] ;
goldTable [ row ] [ col ] = gold [ row ] [ col ] + Math . max ( right , Math . max ( right_up , right_down ) ) ; } }
int res = goldTable [ 0 ] [ 0 ] ; for ( int i = 1 ; i < m ; i ++ ) res = Math . max ( res , goldTable [ i ] [ 0 ] ) ; return res ; }
public static void main ( String arg [ ] ) { int gold [ ] [ ] = { { 1 , 3 , 1 , 5 } , { 2 , 2 , 4 , 1 } , { 5 , 0 , 2 , 3 } , { 0 , 6 , 1 , 2 } } ; int m = 4 , n = 4 ; System . out . print ( getMaxGold ( gold , m , n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { public static int M = 100 ;
static int minAdjustmentCost ( int A [ ] , int n , int target ) {
int [ ] [ ] dp = new int [ n ] [ M + 1 ] ;
for ( int j = 0 ; j <= M ; j ++ ) dp [ 0 ] [ j ] = Math . abs ( j - A [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) {
for ( int j = 0 ; j <= M ; j ++ ) {
dp [ i ] [ j ] = Integer . MAX_VALUE ;
int k = Math . max ( j - target , 0 ) ; for ( ; k <= Math . min ( M , j + target ) ; k ++ ) dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , dp [ i - 1 ] [ k ] + Math . abs ( A [ i ] - j ) ) ; } }
int res = Integer . MAX_VALUE ; for ( int j = 0 ; j <= M ; j ++ ) res = Math . min ( res , dp [ n - 1 ] [ j ] ) ; return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 55 , 77 , 52 , 61 , 39 , 6 , 25 , 60 , 49 , 47 } ; int n = arr . length ; int target = 10 ; System . out . println ( " Minimum ▁ adjustment ▁ cost ▁ is ▁ " + minAdjustmentCost ( arr , n , target ) ) ; } }
static int totalCombination ( int L , int R ) {
int count = 0 ;
int K = R - L ;
if ( K < L ) return 0 ;
int ans = K - L ;
count = ( ( ans + 1 ) * ( ans + 2 ) ) / 2 ;
return count ; }
public static void main ( String [ ] args ) { int L = 2 , R = 6 ; System . out . print ( totalCombination ( L , R ) ) ; } }
static void printArrays ( int n ) {
ArrayList < Integer > A = new ArrayList < Integer > ( ) ; ArrayList < Integer > B = new ArrayList < Integer > ( ) ;
for ( int i = 1 ; i <= 2 * n ; i ++ ) {
if ( i % 2 == 0 ) A . add ( i ) ; else B . add ( i ) ; }
System . out . print ( " { ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( A . get ( i ) ) ; if ( i != n - 1 ) System . out . print ( " , ▁ " ) ; } System . out . print ( " }NEW_LINE");
System . out . print ( " { ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( B . get ( i ) ) ; if ( i != n - 1 ) System . out . print ( " , ▁ " ) ; } System . out . print ( " ▁ } " ) ; }
public static void main ( String [ ] args ) { int N = 5 ;
printArrays ( N ) ; } }
static void flipBitsOfAandB ( int A , int B ) {
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( ( A & ( 1 << i ) ) & ( B & ( 1 << i ) ) ) != 0 ) {
A = A ^ ( 1 << i ) ;
B = B ^ ( 1 << i ) ; } }
System . out . print ( A + " ▁ " + B ) ; }
public static void main ( String [ ] args ) { int A = 7 , B = 4 ; flipBitsOfAandB ( A , B ) ; } }
static int findDistinctSums ( int N ) { return ( 2 * N - 1 ) ; }
public static void main ( String [ ] args ) { int N = 3 ; System . out . print ( findDistinctSums ( N ) ) ; } }
public static int countSubstrings ( String str ) {
int [ ] freq = new int [ 3 ] ;
int count = 0 ; int i = 0 ;
for ( int j = 0 ; j < str . length ( ) ; j ++ ) {
freq [ str . charAt ( j ) - '0' ] ++ ;
while ( freq [ 0 ] > 0 && freq [ 1 ] > 0 && freq [ 2 ] > 0 ) { freq [ str . charAt ( i ++ ) - '0' ] -- ; }
count += i ; }
return count ; }
public static void main ( String [ ] args ) { String str = "00021" ; System . out . println ( countSubstrings ( str ) ) ; } }
static int minFlips ( String str ) {
int count = 0 ;
if ( str . length ( ) <= 2 ) { return 0 ; }
for ( int i = 0 ; i < str . length ( ) - 2 {
if ( str . charAt ( i ) == str . charAt ( i + 1 ) && str . charAt ( i + 2 ) == str . charAt ( i + 1 ) ) { i = i + 3 ; count ++ ; } else { i ++ ; } }
return count ; }
public static void main ( String [ ] args ) { String S = "0011101" ; System . out . println ( minFlips ( S ) ) ; } }
static String convertToHex ( int num ) { StringBuilder temp = new StringBuilder ( ) ; while ( num != 0 ) { int rem = num % 16 ; char c ; if ( rem < 10 ) { c = ( char ) ( rem + 48 ) ; } else { c = ( char ) ( rem + 87 ) ; } temp . append ( c ) ; num = num / 16 ; } return temp . toString ( ) ; }
static String encryptString ( String S , int N ) { StringBuilder ans = new StringBuilder ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { char ch = S . charAt ( i ) ; int count = 0 ; String hex ;
while ( i < N && S . charAt ( i ) == ch ) {
count ++ ; i ++ ; }
i -- ;
hex = convertToHex ( count ) ;
ans . append ( ch ) ;
ans . append ( hex ) ; }
ans . reverse ( ) ;
return ans . toString ( ) ; }
String S = " abc " ; int N = S . length ( ) ;
System . out . println ( encryptString ( S , N ) ) ; } }
static int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static int countOfString ( int N ) {
int Stotal = ( int ) Math . pow ( 2 , N ) ;
int Sequal = 0 ;
if ( N % 2 == 0 ) Sequal = binomialCoeff ( N , N / 2 ) ; int S1 = ( Stotal - Sequal ) / 2 ; return S1 ; }
public static void main ( String [ ] args ) { int N = 3 ; System . out . print ( countOfString ( N ) ) ; } }
static String removeCharRecursive ( String str , char X ) {
if ( str . length ( ) == 0 ) { return " " ; }
if ( str . charAt ( 0 ) == X ) {
return removeCharRecursive ( str . substring ( 1 ) , X ) ; }
return str . charAt ( 0 ) + removeCharRecursive ( str . substring ( 1 ) , X ) ; }
String str = " geeksforgeeks " ;
char X = ' e ' ;
str = removeCharRecursive ( str , X ) ; System . out . println ( str ) ; } }
static boolean isValid ( char a1 , char a2 , String str , int flag ) { char v1 , v2 ;
if ( flag == 0 ) { v1 = str . charAt ( 4 ) ; v2 = str . charAt ( 3 ) ; } else {
v1 = str . charAt ( 1 ) ; v2 = str . charAt ( 0 ) ; }
if ( v1 != a1 && v1 != ' ? ' ) return false ; if ( v2 != a2 && v2 != ' ? ' ) return false ; return true ; }
static boolean inRange ( int hh , int mm , int L , int R ) { int a = Math . abs ( hh - mm ) ;
if ( a < L a > R ) return false ; return true ; }
static void displayTime ( int hh , int mm ) { if ( hh > 10 ) System . out . print ( hh + " : " ) ; else if ( hh < 10 ) System . out . print ( "0" + hh + " : " ) ; if ( mm > 10 ) System . out . println ( mm ) ; else if ( mm < 10 ) System . out . println ( "0" + mm ) ; }
static void maximumTimeWithDifferenceInRange ( String str , int L , int R ) { int i = 0 , j = 0 ; int h1 , h2 , m1 , m2 ;
for ( i = 23 ; i >= 0 ; i -- ) { h1 = i % 10 ; h2 = i / 10 ;
if ( ! isValid ( ( char ) h1 , ( char ) h2 , str , 1 ) ) { continue ; }
for ( j = 59 ; j >= 0 ; j -- ) { m1 = j % 10 ; m2 = j / 10 ;
if ( ! isValid ( ( char ) m1 , ( char ) m2 , str , 0 ) ) { continue ; } if ( inRange ( i , j , L , R ) ) { displayTime ( i , j ) ; return ; } } } if ( inRange ( i , j , L , R ) ) displayTime ( i , j ) ; else System . out . println ( " - 1" ) ; }
String timeValue = " ? ? : ? ? " ;
int L = 20 , R = 39 ; maximumTimeWithDifferenceInRange ( timeValue , L , R ) ; } }
static boolean check ( String s , int n ) {
Stack < Character > st = new Stack < Character > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( ! st . isEmpty ( ) && st . peek ( ) == s . charAt ( i ) ) st . pop ( ) ;
else st . add ( s . charAt ( i ) ) ; }
if ( st . isEmpty ( ) ) { return true ; }
else { return false ; } }
String str = " aanncddc " ; int n = str . length ( ) ;
if ( check ( str , n ) ) { System . out . print ( " Yes " + "NEW_LINE"); } else { System . out . print ( " No " + "NEW_LINE"); } } }
import java . util . * ; class GFG { static void findNumOfValidWords ( Vector < String > w , Vector < String > p ) {
HashMap < Integer , Integer > m = new HashMap < > ( ) ;
Vector < Integer > res = new Vector < > ( ) ;
for ( String s : w ) { int val = 0 ;
for ( char c : s . toCharArray ( ) ) { val = val | ( 1 << ( c - ' a ' ) ) ; }
if ( m . containsKey ( val ) ) m . put ( val , m . get ( val ) + 1 ) ; else m . put ( val , 1 ) ; }
for ( String s : p ) { int val = 0 ;
for ( char c : s . toCharArray ( ) ) { val = val | ( 1 << ( c - ' a ' ) ) ; } int temp = val ; int first = s . charAt ( 0 ) - ' a ' ; int count = 0 ; while ( temp != 0 ) {
if ( ( ( temp >> first ) & 1 ) == 1 ) { if ( m . containsKey ( temp ) ) { count += m . get ( temp ) ; } }
temp = ( temp - 1 ) & val ; }
res . add ( count ) ; }
for ( int it : res ) { System . out . println ( it ) ; } }
public static void main ( String [ ] args ) { Vector < String > arr1 = new Vector < > ( ) ; arr1 . add ( " aaaa " ) ; arr1 . add ( " asas " ) ; arr1 . add ( " able " ) ; arr1 . add ( " ability " ) ; arr1 . add ( " actt " ) ; arr1 . add ( " actor " ) ; arr1 . add ( " access " ) ; Vector < String > arr2 = new Vector < > ( ) ; arr2 . add ( " aboveyz " ) ; arr2 . add ( " abrodyz " ) ; arr2 . add ( " absolute " ) ; arr2 . add ( " absoryz " ) ; arr2 . add ( " actresz " ) ; arr2 . add ( " gaswxyz " ) ;
findNumOfValidWords ( arr1 , arr2 ) ; } }
static void flip ( String s ) { StringBuilder sb = new StringBuilder ( s ) ; for ( int i = 0 ; i < sb . length ( ) ; i ++ ) {
if ( sb . charAt ( i ) == '0' ) {
while ( sb . charAt ( i ) == '0' ) {
sb . setCharAt ( i , '1' ) ; i ++ ; }
break ; } } System . out . println ( sb . toString ( ) ) ; }
public static void main ( String [ ] args ) { String s = "100010001" ; flip ( s ) ; } }
static void getOrgString ( String s ) {
System . out . print ( s . charAt ( 0 ) ) ;
int i = 1 ; while ( i < s . length ( ) ) {
if ( s . charAt ( i ) >= ' A ' && s . charAt ( i ) <= ' Z ' ) System . out . print ( " ▁ " + Character . toLowerCase ( s . charAt ( i ) ) ) ;
else System . out . print ( s . charAt ( i ) ) ; i ++ ; } }
public static void main ( String [ ] args ) { String s = " ILoveGeeksForGeeks " ; getOrgString ( s ) ; } }
static int countChar ( String str , char x ) { int count = 0 ; int n = 10 ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) if ( str . charAt ( i ) == x ) count ++ ;
int repetitions = n / str . length ( ) ; count = count * repetitions ;
for ( int i = 0 ; i < n % str . length ( ) ; i ++ ) { if ( str . charAt ( i ) == x ) count ++ ; } return count ; }
public static void main ( String args [ ] ) { String str = " abcac " ; System . out . println ( countChar ( str , ' a ' ) ) ; } }
class GFG { static void countFreq ( int arr [ ] , int n , int limit ) {
int [ ] count = new int [ limit + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) count [ arr [ i ] ] ++ ; for ( int i = 0 ; i <= limit ; i ++ ) if ( count [ i ] > 0 ) System . out . println ( i + " ▁ " + count [ i ] ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 5 , 6 , 6 , 5 , 6 , 1 , 2 , 3 , 10 , 10 } ; int n = arr . length ; int limit = 10 ; countFreq ( arr , n , limit ) ; } }
static boolean check ( String s , int m ) {
int l = s . length ( ) ;
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( s . charAt ( i ) == '0' ) { c2 = 0 ;
c1 ++ ; } else { c1 = 0 ;
c2 ++ ; } if ( c1 == m c2 == m ) return true ; } return false ; }
public static void main ( String [ ] args ) { String s = "001001" ; int m = 2 ;
if ( check ( s , m ) ) System . out . println ( " YES " ) ; else System . out . println ( " NO " ) ; } }
static int productAtKthLevel ( String tree , int k ) { int level = - 1 ;
int product = 1 ; int n = tree . length ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( tree . charAt ( i ) == ' ( ' ) level ++ ;
else if ( tree . charAt ( i ) == ' ) ' ) level -- ; else {
if ( level == k ) product *= ( tree . charAt ( i ) - '0' ) ; } }
return product ; }
public static void main ( String [ ] args ) { String tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; System . out . println ( productAtKthLevel ( tree , k ) ) ; } }
static void findDuplciates ( String [ ] a , int n , int m ) {
boolean [ ] [ ] isPresent = new boolean [ n ] [ m ] ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { isPresent [ i ] [ j ] = false ; } } for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
for ( int k = 0 ; k < n ; k ++ ) { if ( a [ i ] . charAt ( j ) == a [ k ] . charAt ( j ) && i != k ) { isPresent [ i ] [ j ] = true ; isPresent [ k ] [ j ] = true ; } }
for ( int k = 0 ; k < m ; k ++ ) { if ( a [ i ] . charAt ( j ) == a [ i ] . charAt ( k ) && j != k ) { isPresent [ i ] [ j ] = true ; isPresent [ i ] [ k ] = true ; } } } } for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < m ; j ++ )
if ( isPresent [ i ] [ j ] == false ) System . out . print ( a [ i ] . charAt ( j ) ) ; }
public static void main ( String [ ] args ) { int n = 2 , m = 2 ;
String [ ] a = new String [ ] { " zx " , " xz " } ;
findDuplciates ( a , n , m ) ; } }
class GFG { static boolean isValidISBN ( String isbn ) {
int n = isbn . length ( ) ; if ( n != 10 ) return false ;
int sum = 0 ; for ( int i = 0 ; i < 9 ; i ++ ) { int digit = isbn . charAt ( i ) - '0' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
char last = isbn . charAt ( 9 ) ; if ( last != ' X ' && ( last < '0' last > '9' ) ) return false ;
sum += ( ( last == ' X ' ) ? 10 : ( last - '0' ) ) ;
return ( sum % 11 == 0 ) ; }
public static void main ( String [ ] args ) { String isbn = "007462542X " ; if ( isValidISBN ( isbn ) ) System . out . print ( " Valid " ) ; else System . out . print ( " Invalid " ) ; } }
static boolean isVowel ( char c ) { return ( c == ' a ' c == ' A ' c == ' e ' c == ' E ' c == ' i ' c == ' I ' c == ' o ' c == ' O ' c == ' u ' c == ' U ' ) ; }
static String reverseVowel ( String str1 ) { int j = 0 ;
char [ ] str = str1 . toCharArray ( ) ; String vowel = " " ; for ( int i = 0 ; i < str . length ; i ++ ) { if ( isVowel ( str [ i ] ) ) { j ++ ; vowel += str [ i ] ; } }
for ( int i = 0 ; i < str . length ; i ++ ) { if ( isVowel ( str [ i ] ) ) { str [ i ] = vowel . charAt ( -- j ) ; } } return String . valueOf ( str ) ; }
public static void main ( String [ ] args ) { String str = " hello ▁ world " ; System . out . println ( reverseVowel ( str ) ) ; } }
static String firstLetterWord ( String str ) { String result = " " ;
boolean v = true ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
if ( str . charAt ( i ) == ' ▁ ' ) { v = true ; }
else if ( str . charAt ( i ) != ' ▁ ' && v == true ) { result += ( str . charAt ( i ) ) ; v = false ; } } return result ; }
public static void main ( String [ ] args ) { String str = " geeks ▁ for ▁ geeks " ; System . out . println ( firstLetterWord ( str ) ) ; } }
import java . util . Arrays ; class GFG { static int ans = 0 ;
static void dfs ( int i , int j , int [ ] [ ] grid , boolean [ ] [ ] vis , int z , int z_count ) { int n = grid . length , m = grid [ 0 ] . length ;
vis [ i ] [ j ] = true ; if ( grid [ i ] [ j ] == 0 )
z ++ ;
if ( grid [ i ] [ j ] == 2 ) {
if ( z == z_count ) ans ++ ; vis [ i ] [ j ] = false ; return ; }
if ( i >= 1 && ! vis [ i - 1 ] [ j ] && grid [ i - 1 ] [ j ] != - 1 ) dfs ( i - 1 , j , grid , vis , z , z_count ) ;
if ( i < n - 1 && ! vis [ i + 1 ] [ j ] && grid [ i + 1 ] [ j ] != - 1 ) dfs ( i + 1 , j , grid , vis , z , z_count ) ;
if ( j >= 1 && ! vis [ i ] [ j - 1 ] && grid [ i ] [ j - 1 ] != - 1 ) dfs ( i , j - 1 , grid , vis , z , z_count ) ;
if ( j < m - 1 && ! vis [ i ] [ j + 1 ] && grid [ i ] [ j + 1 ] != - 1 ) dfs ( i , j + 1 , grid , vis , z , z_count ) ;
vis [ i ] [ j ] = false ; }
static int uniquePaths ( int [ ] [ ] grid ) {
int n = grid . length , m = grid [ 0 ] . length ; boolean [ ] [ ] vis = new boolean [ n ] [ m ] ; for ( int i = 0 ; i < n ; i ++ ) { Arrays . fill ( vis [ i ] , false ) ; } int x = 0 , y = 0 ; for ( int i = 0 ; i < n ; ++ i ) { for ( int j = 0 ; j < m ; ++ j ) {
if ( grid [ i ] [ j ] == 0 ) z_count ++ ; else if ( grid [ i ] [ j ] == 1 ) {
x = i ; y = j ; } } } dfs ( x , y , grid , vis , 0 , z_count ) ; return ans ; }
public static void main ( String [ ] args ) { int [ ] [ ] grid = { { 1 , 0 , 0 , 0 } , { 0 , 0 , 0 , 0 } , { 0 , 0 , 2 , - 1 } } ; System . out . println ( uniquePaths ( grid ) ) ; } }
static int numPairs ( int a [ ] , int n ) { int ans , i , index ;
ans = 0 ;
for ( i = 0 ; i < n ; i ++ ) a [ i ] = Math . abs ( a [ i ] ) ;
Arrays . sort ( a ) ;
for ( i = 0 ; i < n ; i ++ ) { index = 2 ; ans += index - i - 1 ; }
return ans ; }
public static void main ( String [ ] args ) { int a [ ] = new int [ ] { 3 , 6 } ; int n = a . length ; System . out . println ( numPairs ( a , n ) ) ; } }
static int areaOfSquare ( int S ) {
int area = S * S ; return area ; }
int S = 5 ;
System . out . println ( areaOfSquare ( S ) ) ; } }
class GFG { static int maxPointOfIntersection ( int x , int y ) { int k = y * ( y - 1 ) / 2 ; k = k + x * ( 2 * y + x - 1 ) ; return k ; }
int x = 3 ;
int y = 4 ;
System . out . print ( maxPointOfIntersection ( x , y ) ) ; } }
static int Icosihenagonal_num ( int n ) {
return ( 19 * n * n - 17 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . print ( Icosihenagonal_num ( n ) + "NEW_LINE"); n = 10 ; System . out . print ( Icosihenagonal_num ( n ) + "NEW_LINE"); } }
class GFG { static double [ ] find_Centroid ( double v [ ] [ ] ) { double [ ] ans = new double [ 2 ] ; int n = v . length ; double signedArea = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { double x0 = v [ i ] [ 0 ] , y0 = v [ i ] [ 1 ] ; double x1 = v [ ( i + 1 ) % n ] [ 0 ] , y1 = v [ ( i + 1 ) % n ] [ 1 ] ;
double A = ( x0 * y1 ) - ( x1 * y0 ) ; signedArea += A ;
ans [ 0 ] += ( x0 + x1 ) * A ; ans [ 1 ] += ( y0 + y1 ) * A ; } signedArea *= 0.5 ; ans [ 0 ] = ( ans [ 0 ] ) / ( 6 * signedArea ) ; ans [ 1 ] = ( ans [ 1 ] ) / ( 6 * signedArea ) ; return ans ; }
double vp [ ] [ ] = { { 1 , 2 } , { 3 , - 4 } , { 6 , - 7 } } ; double [ ] ans = find_Centroid ( vp ) ; System . out . println ( ans [ 0 ] + " ▁ " + ans [ 1 ] ) ; } }
public static void main ( String [ ] args ) { int d = 10 ; double a ;
a = ( double ) ( 360 - ( 6 * d ) ) / 4 ;
System . out . print ( a + " , ▁ " + ( a + d ) + " , ▁ " + ( a + ( 2 * d ) ) + " , ▁ " + ( a + ( 3 * d ) ) ) ; } }
static void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = Math . abs ( ( c2 * z1 + d2 ) ) / ( float ) ( Math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; System . out . println ( " Perpendicular ▁ distance ▁ is ▁ " + d ) ; } else System . out . println ( " Planes ▁ are ▁ not ▁ parallel " ) ; }
public static void main ( String [ ] args ) { float a1 = 1 ; float b1 = 2 ; float c1 = - 1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = - 3 ; float d2 = - 4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ; } }
static int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
static long numOfNecklace ( int N ) {
long ans = factorial ( N ) / ( factorial ( N / 2 ) * factorial ( N / 2 ) ) ;
ans = ans * factorial ( N / 2 - 1 ) ; ans = ans * factorial ( N / 2 - 1 ) ;
ans /= 2 ;
return ans ; }
int N = 4 ;
System . out . println ( numOfNecklace ( N ) ) ; } }
static String isDivisibleByDivisor ( int S , int D ) {
S %= D ;
Set < Integer > hashMap = new HashSet < > ( ) ; hashMap . add ( S ) ; for ( int i = 0 ; i <= D ; i ++ ) {
S += ( S % D ) ; S %= D ;
if ( hashMap . contains ( S ) ) {
if ( S == 0 ) { return " Yes " ; } return " No " ; }
else hashMap . add ( S ) ; } return " Yes " ; }
public static void main ( String [ ] args ) { int S = 3 , D = 6 ; System . out . println ( isDivisibleByDivisor ( S , D ) ) ; } }
static void minimumSteps ( int x , int y ) {
int cnt = 0 ;
while ( x != 0 && y != 0 ) {
if ( x > y ) {
cnt += x / y ; x %= y ; }
else {
cnt += y / x ; y %= x ; } } cnt -- ;
if ( x > 1 y > 1 ) cnt = - 1 ;
System . out . println ( cnt ) ; }
int x = 3 , y = 1 ; minimumSteps ( x , y ) ; } }
import java . io . * ; import java . util . * ; public class GFG { static void printLeast ( String arr ) {
int min_avail = 1 , pos_of_I = 0 ;
ArrayList < Integer > al = new ArrayList < > ( ) ;
if ( arr . charAt ( 0 ) == ' I ' ) { al . add ( 1 ) ; al . add ( 2 ) ; min_avail = 3 ; pos_of_I = 1 ; } else { al . add ( 2 ) ; al . add ( 1 ) ; min_avail = 3 ; pos_of_I = 0 ; }
for ( int i = 1 ; i < arr . length ( ) ; i ++ ) { if ( arr . charAt ( i ) == ' I ' ) { al . add ( min_avail ) ; min_avail ++ ; pos_of_I = i + 1 ; } else { al . add ( al . get ( i ) ) ; for ( int j = pos_of_I ; j <= i ; j ++ ) al . set ( j , al . get ( j ) + 1 ) ; min_avail ++ ; } }
for ( int i = 0 ; i < al . size ( ) ; i ++ ) System . out . print ( al . get ( i ) + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String args [ ] ) { printLeast ( " IDID " ) ; printLeast ( " I " ) ; printLeast ( " DD " ) ; printLeast ( " II " ) ; printLeast ( " DIDI " ) ; printLeast ( " IIDDD " ) ; printLeast ( " DDIDDIID " ) ; } }
static void PrintMinNumberForPattern ( String seq ) {
String result = " " ;
Stack < Integer > stk = new Stack < Integer > ( ) ;
for ( int i = 0 ; i <= seq . length ( ) ; i ++ ) {
stk . push ( i + 1 ) ;
if ( i == seq . length ( ) || seq . charAt ( i ) == ' I ' ) {
while ( ! stk . empty ( ) ) {
result += String . valueOf ( stk . peek ( ) ) ; result += " ▁ " ; stk . pop ( ) ; } } } System . out . println ( result ) ; }
public static void main ( String [ ] args ) { PrintMinNumberForPattern ( " IDID " ) ; PrintMinNumberForPattern ( " I " ) ; PrintMinNumberForPattern ( " DD " ) ; PrintMinNumberForPattern ( " II " ) ; PrintMinNumberForPattern ( " DIDI " ) ; PrintMinNumberForPattern ( " IIDDD " ) ; PrintMinNumberForPattern ( " DDIDDIID " ) ; } }
static String getMinNumberForPattern ( String seq ) { int n = seq . length ( ) ; if ( n >= 9 ) return " - 1" ; char result [ ] = new char [ n + 1 ] ; int count = 1 ;
for ( int i = 0 ; i <= n ; i ++ ) { if ( i == n || seq . charAt ( i ) == ' I ' ) { for ( int j = i - 1 ; j >= - 1 ; j -- ) { result [ j + 1 ] = ( char ) ( ( int ) '0' + count ++ ) ; if ( j >= 0 && seq . charAt ( j ) == ' I ' ) break ; } } } return new String ( result ) ; }
public static void main ( String [ ] args ) throws IOException { String inputs [ ] = { " IDID " , " I " , " DD " , " II " , " DIDI " , " IIDDD " , " DDIDDIID " } ; for ( String input : inputs ) { System . out . println ( getMinNumberForPattern ( input ) ) ; } } }
static boolean isPrime ( int n ) { int i , c = 0 ; for ( i = 1 ; i < n / 2 ; i ++ ) { if ( n % i == 0 ) c ++ ; } if ( c == 1 ) { return true ; } else { return false ; } }
static void findMinNum ( int arr [ ] , int n ) {
int first = 0 , last = 0 , num , rev , i ; int hash [ ] = new int [ 10 ] ;
for ( i = 0 ; i < n ; i ++ ) { hash [ arr [ i ] ] ++ ; }
System . out . print ( " Minimum ▁ number : ▁ " ) ; for ( i = 0 ; i <= 9 ; i ++ ) {
for ( int j = 0 ; j < hash [ i ] ; j ++ ) System . out . print ( i ) ; } System . out . println ( ) ; System . out . println ( ) ;
for ( i = 0 ; i <= 9 ; i ++ ) { if ( hash [ i ] != 0 ) { first = i ; break ; } }
for ( i = 9 ; i >= 0 ; i -- ) { if ( hash [ i ] != 0 ) { last = i ; break ; } } num = first * 10 + last ; rev = last * 10 + first ;
System . out . print ( " Prime ▁ combinations : ▁ " ) ; if ( isPrime ( num ) && isPrime ( rev ) ) { System . out . println ( num + " ▁ " + rev ) ; } else if ( isPrime ( num ) ) { System . out . println ( num ) ; } else if ( isPrime ( rev ) ) { System . out . println ( rev ) ; } else { System . out . println ( " No ▁ combinations ▁ exist " ) ; } }
public static void main ( String [ ] args ) { SmallPrime smallprime = new SmallPrime ( ) ; int arr [ ] = { 1 , 2 , 4 , 7 , 8 } ; smallprime . findMinNum ( arr , 5 ) ; } }
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static boolean coprime ( int a , int b ) {
return ( gcd ( a , b ) == 1 ) ; }
static void possibleTripletInRange ( int L , int R ) { boolean flag = false ; int possibleA = 0 , possibleB = 0 , possibleC = 0 ;
for ( int a = L ; a <= R ; a ++ ) { for ( int b = a + 1 ; b <= R ; b ++ ) { for ( int c = b + 1 ; c <= R ; c ++ ) {
if ( coprime ( a , b ) && coprime ( b , c ) && ! coprime ( a , c ) ) { flag = true ; possibleA = a ; possibleB = b ; possibleC = c ; break ; } } } }
if ( flag == true ) { System . out . println ( " ( " + possibleA + " , ▁ " + possibleB + " , ▁ " + possibleC + " ) " + " ▁ is ▁ one ▁ such ▁ possible ▁ triplet ▁ " + " between ▁ " + L + " ▁ and ▁ " + R ) ; } else { System . out . println ( " No ▁ Such ▁ Triplet ▁ exists " + " between ▁ " + L + " ▁ and ▁ " + R ) ; } }
public static void main ( String [ ] args ) { int L , R ;
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ; } }
static boolean possibleToReach ( int a , int b ) {
int c = ( int ) Math . cbrt ( a * b ) ;
int re1 = a / c ; int re2 = b / c ;
if ( ( re1 * re1 * re2 == a ) && ( re2 * re2 * re1 == b ) ) return true ; else return false ; }
public static void main ( String [ ] args ) { int A = 60 , B = 450 ; if ( possibleToReach ( A , B ) ) System . out . println ( " yes " ) ; else System . out . println ( " no " ) ; } }
import java . util . * ; class GFG { public static boolean isUndulating ( String n ) {
if ( n . length ( ) <= 2 ) return false ;
for ( int i = 2 ; i < n . length ( ) ; i ++ ) if ( n . charAt ( i - 2 ) != n . charAt ( i ) ) return false ; return true ; }
public static void main ( String [ ] args ) { String n = "1212121" ; if ( isUndulating ( n ) == true ) System . out . println ( " yes " ) ; else System . out . println ( " no " ) ; } }
static int Series ( int n ) { int i ; int sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
public static void main ( String [ ] args ) { int n = 3 ; int res = Series ( n ) ; System . out . println ( res ) ; } }
public static int counLastDigitK ( int low , int high , int k ) { int mlow = 10 * ( int ) Math . ceil ( low / 10.0 ) ; int mhigh = 10 * ( int ) Math . floor ( high / 10.0 ) ; int count = ( mhigh - mlow ) / 10 ; if ( high % 10 >= k ) count ++ ; if ( low % 10 <= k && ( low % 10 ) > 0 ) count ++ ; return count ; }
public static void main ( String argc [ ] ) { int low = 3 , high = 35 , k = 3 ; System . out . println ( counLastDigitK ( low , high , k ) ) ; } }
static int sum ( int L , int R ) {
int p = R / 6 ;
int q = ( L - 1 ) / 6 ;
int sumR = 3 * ( p * ( p + 1 ) ) ;
int sumL = ( q * ( q + 1 ) ) * 3 ;
return sumR - sumL ; }
public static void main ( String [ ] args ) { int L = 1 , R = 20 ; System . out . println ( sum ( L , R ) ) ; } }
static String prevNum ( String str ) { int len = str . length ( ) ; int index = - 1 ;
for ( int i = len - 2 ; i >= 0 ; i -- ) { if ( str . charAt ( i ) > str . charAt ( i + 1 ) ) { index = i ; break ; } }
int smallGreatDgt = - 1 ; for ( int i = len - 1 ; i > index ; i -- ) { if ( str . charAt ( i ) < str . charAt ( index ) ) { if ( smallGreatDgt == - 1 ) { smallGreatDgt = i ; } else if ( str . charAt ( i ) >= str . charAt ( smallGreatDgt ) ) { smallGreatDgt = i ; } } }
if ( index == - 1 ) { return " - 1" ; }
if ( smallGreatDgt != - 1 ) { str = swap ( str , index , smallGreatDgt ) ; return str ; } return " - 1" ; } static String swap ( String str , int i , int j ) { char ch [ ] = str . toCharArray ( ) ; char temp = ch [ i ] ; ch [ i ] = ch [ j ] ; ch [ j ] = temp ; return String . valueOf ( ch ) ; }
public static void main ( String [ ] args ) { String str = "34125" ; System . out . println ( prevNum ( str ) ) ; } }
static int horner ( int poly [ ] , int n , int x ) {
int result = poly [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) result = result * x + poly [ i ] ; return result ; }
static int findSign ( int poly [ ] , int n , int x ) { int result = horner ( poly , n , x ) ; if ( result > 0 ) return 1 ; else if ( result < 0 ) return - 1 ; return 0 ; }
int poly [ ] = { 2 , - 6 , 2 , - 1 } ; int x = 3 ; int n = poly . length ; System . out . print ( " Sign ▁ of ▁ polynomial ▁ is ▁ " + findSign ( poly , n , x ) ) ; } }
class GFG { static int MAX = 100005 ;
static boolean [ ] isPrime = new boolean [ MAX ] ;
static void sieveOfEratostheneses ( ) { isPrime [ 1 ] = true ; for ( int i = 2 ; i * i < MAX ; i ++ ) { if ( ! isPrime [ i ] ) { for ( int j = 2 * i ; j < MAX ; j += i ) isPrime [ j ] = true ; } } }
static int findPrime ( int n ) { int num = n + 1 ;
while ( num > 0 ) {
if ( ! isPrime [ num ] ) return num ;
num = num + 1 ; } return 0 ; }
static int minNumber ( int arr [ ] , int n ) {
sieveOfEratostheneses ( ) ; int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( ! isPrime [ sum ] ) return 0 ;
int num = findPrime ( sum ) ;
return num - sum ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 6 , 8 , 12 } ; int n = arr . length ; System . out . println ( minNumber ( arr , n ) ) ; } }
public static long SubArraySum ( int arr [ ] , int n ) { long result = 0 , temp = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
temp = 0 ; for ( int j = i ; j < n ; j ++ ) {
temp += arr [ j ] ; result += temp ; } } return result ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 } ; int n = arr . length ; System . out . println ( " Sum ▁ of ▁ SubArray ▁ : ▁ " + SubArraySum ( arr , n ) ) ; } }
import java . io . * ; class GFG { static int highestPowerof2 ( int n ) { int p = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) ; return ( int ) Math . pow ( 2 , p ) ; }
public static void main ( String [ ] args ) { int n = 10 ; System . out . println ( highestPowerof2 ( n ) ) ; } }
static int aModM ( String s , int mod ) { int number = 0 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
number = ( number * 10 ) ; int x = Character . getNumericValue ( s . charAt ( i ) ) ; number = number + x ; number %= mod ; } return number ; }
static int ApowBmodM ( String a , int b , int m ) {
int ans = aModM ( a , m ) ; int mul = ans ;
for ( int i = 1 ; i < b ; i ++ ) ans = ( ans * mul ) % m ; return ans ; }
public static void main ( String args [ ] ) { String a = "987584345091051645734583954832576" ; int b = 3 , m = 11 ; System . out . println ( ApowBmodM ( a , b , m ) ) ; } }
static class Data { int x , y ; public Data ( int x , int y ) { super ( ) ; this . x = x ; this . y = y ; } } ;
static double interpolate ( Data f [ ] , int xi , int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
double term = f [ i ] . y ; for ( int j = 0 ; j < n ; j ++ ) { if ( j != i ) term = term * ( xi - f [ j ] . x ) / ( f [ i ] . x - f [ j ] . x ) ; }
result += term ; } return result ; }
Data f [ ] = { new Data ( 0 , 2 ) , new Data ( 1 , 3 ) , new Data ( 2 , 12 ) , new Data ( 5 , 147 ) } ;
System . out . print ( " Value ▁ of ▁ f ( 3 ) ▁ is ▁ : ▁ " + ( int ) interpolate ( f , 3 , 4 ) ) ; } }
static int SieveOfSundaram ( int n ) {
int nNew = ( n - 1 ) / 2 ;
boolean marked [ ] = new boolean [ nNew + 1 ] ;
Arrays . fill ( marked , false ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) for ( int j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) System . out . print ( 2 + " ▁ " ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) System . out . print ( 2 * i + 1 + " ▁ " ) ; return - 1 ; }
public static void main ( String [ ] args ) { int n = 20 ; SieveOfSundaram ( n ) ; } }
static void constructArray ( int A [ ] , int N , int K ) {
int B [ ] = new int [ N ] ;
int totalXOR = A [ 0 ] ^ K ;
for ( int i = 0 ; i < N ; i ++ ) B [ i ] = totalXOR ^ A [ i ] ;
for ( int i = 0 ; i < N ; i ++ ) { System . out . print ( B [ i ] + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int A [ ] = { 13 , 14 , 10 , 6 } , K = 2 ; int N = A . length ;
constructArray ( A , N , K ) ; } }
static int extraElement ( int A [ ] , int B [ ] , int n ) {
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) ans ^= A [ i ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) ans ^= B [ i ] ; return ans ; }
public static void main ( String [ ] args ) { int A [ ] = { 10 , 15 , 5 } ; int B [ ] = { 10 , 100 , 15 , 5 } ; int n = A . length ; System . out . println ( extraElement ( A , B , n ) ) ; } }
static int hammingDistance ( int n1 , int n2 ) { int x = n1 ^ n2 ; int setBits = 0 ; while ( x > 0 ) { setBits += x & 1 ; x >>= 1 ; } return setBits ; }
public static void main ( String [ ] args ) { int n1 = 9 , n2 = 14 ; System . out . println ( hammingDistance ( n1 , n2 ) ) ; } }
static void printSubsets ( int n ) { for ( int i = 0 ; i <= n ; i ++ ) if ( ( n & i ) == i ) System . out . print ( i + " ▁ " ) ; }
public static void main ( String [ ] args ) { int n = 9 ; printSubsets ( n ) ; } }
class GFG { static int setBitNumber ( int n ) {
int k = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) ;
return 1 << k ; }
public static void main ( String arg [ ] ) { int n = 273 ; System . out . print ( setBitNumber ( n ) ) ; } }
public static int subset ( int ar [ ] , int n ) {
int res = 0 ;
Arrays . sort ( ar ) ;
for ( int i = 0 ; i < n ; i ++ ) { int count = 1 ;
for ( ; i < n - 1 ; i ++ ) { if ( ar [ i ] == ar [ i + 1 ] ) count ++ ; else break ; }
res = Math . max ( res , count ) ; } return res ; }
public static void main ( String argc [ ] ) { int arr [ ] = { 5 , 6 , 9 , 3 , 4 , 3 , 4 } ; int n = 7 ; System . out . println ( subset ( arr , n ) ) ; } }
static ArrayList < Integer > psquare = new ArrayList < > ( ) ;
static void calcPsquare ( int N ) { for ( int i = 1 ; i * i <= N ; i ++ ) psquare . add ( i * i ) ; }
static int countWays ( int index , int target ) {
if ( target == 0 ) return 1 ; if ( index < 0 target < 0 ) return 0 ;
int inc = countWays ( index , target - psquare . get ( index ) ) ;
int exc = countWays ( index - 1 , target ) ;
return inc + exc ; }
int N = 9 ;
calcPsquare ( N ) ;
System . out . print ( countWays ( psquare . size ( ) - 1 , N ) ) ; } }
import java . util . * ; class GFG { static class pair { int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static class TreeNode { int data , size ; TreeNode left ; TreeNode right ; } ;
static TreeNode newNode ( int data ) { TreeNode Node = new TreeNode ( ) ; Node . data = data ; Node . left = null ; Node . right = null ;
static pair sumofsubtree ( TreeNode root ) {
pair p = new pair ( 1 , 0 ) ;
if ( root . left != null ) { pair ptemp = sumofsubtree ( root . left ) ; p . second += ptemp . first + ptemp . second ; p . first += ptemp . first ; }
if ( root . right != null ) { pair ptemp = sumofsubtree ( root . right ) ; p . second += ptemp . first + ptemp . second ; p . first += ptemp . first ; }
root . size = p . first ; return p ; }
static int sum = 0 ;
static void distance ( TreeNode root , int target , int distancesum , int n ) {
if ( root . data == target ) { sum = distancesum ; }
if ( root . left != null ) {
int tempsum = distancesum - root . left . size + ( n - root . left . size ) ;
distance ( root . left , target , tempsum , n ) ; }
if ( root . right != null ) {
int tempsum = distancesum - root . right . size + ( n - root . right . size ) ;
distance ( root . right , target , tempsum , n ) ; } }
TreeNode root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . right = newNode ( 3 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 5 ) ; root . right . left = newNode ( 6 ) ; root . right . right = newNode ( 7 ) ; root . left . left . left = newNode ( 8 ) ; root . left . left . right = newNode ( 9 ) ; int target = 3 ; pair p = sumofsubtree ( root ) ;
int totalnodes = p . first ; distance ( root , target , p . second , totalnodes ) ;
System . out . print ( sum + "NEW_LINE"); } }
static int [ ] reverse ( int a [ ] ) { int i , n = a . length , t ; for ( i = 0 ; i < n / 2 ; i ++ ) { t = a [ i ] ; a [ i ] = a [ n - i - 1 ] ; a [ n - i - 1 ] = t ; } return a ; }
static void rearrangeArray ( int A [ ] , int B [ ] , int N , int K ) {
Arrays . sort ( B ) ; B = reverse ( B ) ; boolean flag = true ; for ( int i = 0 ; i < N ; i ++ ) {
if ( A [ i ] + B [ i ] > K ) { flag = false ; break ; } } if ( ! flag ) { System . out . print ( " - 1" + "NEW_LINE"); } else {
for ( int i = 0 ; i < N ; i ++ ) { System . out . print ( B [ i ] + " ▁ " ) ; } } }
int A [ ] = { 1 , 2 , 3 , 4 , 2 } ; int B [ ] = { 1 , 2 , 3 , 1 , 1 } ; int N = A . length ; int K = 5 ; rearrangeArray ( A , B , N , K ) ; } }
import java . io . * ; class GFG {
static void countRows ( int [ ] [ ] mat ) {
int count = 0 ;
int totalSum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { totalSum += mat [ i ] [ j ] ; } }
for ( int i = 0 ; i < n ; i ++ ) {
int currSum = 0 ;
for ( int j = 0 ; j < m ; j ++ ) { currSum += mat [ i ] [ j ] ; }
if ( currSum > totalSum - currSum )
count ++ ; }
System . out . println ( count ) ; }
int [ ] [ ] mat = { { 2 , - 1 , 5 } , { - 3 , 0 , - 2 } , { 5 , 1 , 2 } } ;
countRows ( mat ) ; } }
static boolean areElementsContiguous ( int arr [ ] , int n ) {
Arrays . sort ( arr ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] > 1 ) return false ; return true ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . length ; if ( areElementsContiguous ( arr , n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean areElementsContiguous ( int arr [ ] , int n ) {
int max = Integer . MIN_VALUE ; int min = Integer . MAX_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { max = Math . max ( max , arr [ i ] ) ; min = Math . min ( min , arr [ i ] ) ; } int m = max - min + 1 ;
if ( m > n ) return false ;
boolean visited [ ] = new boolean [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) visited [ arr [ i ] - min ] = true ;
for ( int i = 0 ; i < m ; i ++ ) if ( visited [ i ] == false ) return false ; return true ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . length ; if ( areElementsContiguous ( arr , n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static Boolean areElementsContiguous ( int arr [ ] , int n ) {
HashSet < Integer > us = new HashSet < Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) us . add ( arr [ i ] ) ;
int count = 1 ;
int curr_ele = arr [ 0 ] - 1 ;
while ( us . contains ( curr_ele ) == true ) {
count ++ ;
curr_ele -- ; }
curr_ele = arr [ 0 ] + 1 ;
while ( us . contains ( curr_ele ) == true ) {
count ++ ;
curr_ele ++ ; }
return ( count == ( us . size ( ) ) ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . length ; if ( areElementsContiguous ( arr , n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static void longest ( int a [ ] , int n , int k ) { int [ ] freq = new int [ 7 ] ; int start = 0 , end = 0 , now = 0 , l = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
freq [ a [ i ] ] ++ ;
if ( freq [ a [ i ] ] == 1 ) now ++ ;
while ( now > k ) {
freq [ a [ l ] ] -- ;
if ( freq [ a [ l ] ] == 0 ) now -- ;
l ++ ; }
if ( i - l + 1 >= end - start + 1 ) { end = i ; start = l ; } }
for ( int i = start ; i <= end ; i ++ ) System . out . print ( a [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int a [ ] = { 6 , 5 , 1 , 2 , 3 , 2 , 1 , 4 , 5 } ; int n = a . length ; int k = 3 ; longest ( a , n , k ) ; } }
static boolean kOverlap ( ArrayList < Pair > pairs , int k ) {
ArrayList < Pair > vec = new ArrayList < > ( ) ; for ( int i = 0 ; i < pairs . size ( ) ; i ++ ) {
vec . add ( new Pair ( pairs . get ( i ) . first , - 1 ) ) ; vec . add ( new Pair ( pairs . get ( i ) . second , + 1 ) ) ; }
Collections . sort ( vec , new Comparator < Pair > ( ) {
Stack < Pair > st = new Stack < > ( ) ; for ( int i = 0 ; i < vec . size ( ) ; i ++ ) {
Pair cur = vec . get ( i ) ;
if ( cur . second == - 1 ) {
st . push ( cur ) ; }
else {
st . pop ( ) ; }
if ( st . size ( ) >= k ) { return true ; } } return false ; }
public static void main ( String [ ] args ) { ArrayList < Pair > pairs = new ArrayList < > ( ) ; pairs . add ( new Pair ( 1 , 3 ) ) ; pairs . add ( new Pair ( 2 , 4 ) ) ; pairs . add ( new Pair ( 3 , 5 ) ) ; pairs . add ( new Pair ( 7 , 10 ) ) ; int n = pairs . size ( ) , k = 3 ; if ( kOverlap ( pairs , k ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
class GFG { static final int N = 5 ;
static int ptr [ ] = new int [ 501 ] ;
static void findSmallestRange ( int arr [ ] [ ] , int n , int k ) { int i , minval , maxval , minrange , minel = 0 , maxel = 0 , flag , minind ;
for ( i = 0 ; i <= k ; i ++ ) { ptr [ i ] = 0 ; } minrange = Integer . MAX_VALUE ; while ( true ) {
minind = - 1 ; minval = Integer . MAX_VALUE ; maxval = Integer . MIN_VALUE ; flag = 0 ;
for ( i = 0 ; i < k ; i ++ ) {
if ( ptr [ i ] == n ) { flag = 1 ; break ; }
if ( ptr [ i ] < n && arr [ i ] [ ptr [ i ] ] < minval ) {
minind = i ; minval = arr [ i ] [ ptr [ i ] ] ; }
if ( ptr [ i ] < n && arr [ i ] [ ptr [ i ] ] > maxval ) { maxval = arr [ i ] [ ptr [ i ] ] ; } }
if ( flag == 1 ) { break ; } ptr [ minind ] ++ ;
if ( ( maxval - minval ) < minrange ) { minel = minval ; maxel = maxval ; minrange = maxel - minel ; } } System . out . printf ( "The smallest range is [%d, %d]NEW_LINE", minel, maxel); }
public static void main ( String [ ] args ) { int arr [ ] [ ] = { { 4 , 7 , 9 , 12 , 15 } , { 0 , 8 , 10 , 14 , 20 } , { 6 , 12 , 16 , 30 , 50 } } ; int k = arr . length ; findSmallestRange ( arr , N , k ) ; } }
static int findLargestd ( int [ ] S , int n ) { boolean found = false ;
Arrays . sort ( S ) ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { for ( int j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( int k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( int l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return Integer . MAX_VALUE ; return - 1 ; }
public static void main ( String [ ] args ) { int [ ] S = new int [ ] { 2 , 3 , 5 , 7 , 12 } ; int n = S . length ; int ans = findLargestd ( S , n ) ; if ( ans == Integer . MAX_VALUE ) System . out . println ( " No ▁ Solution " ) ; else System . out . println ( " Largest ▁ d ▁ such ▁ that ▁ " + " a ▁ + ▁ " + " b ▁ + ▁ c ▁ = ▁ d ▁ is ▁ " + ans ) ; } }
class Indexes { int i , j ; Indexes ( int i , int j ) { this . i = i ; this . j = j ; } int getI ( ) { return i ; } int getJ ( ) { return j ; } } class GFG {
static int findFourElements ( int [ ] arr , int n ) { HashMap < Integer , Indexes > map = new HashMap < > ( ) ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) { map . put ( arr [ i ] + arr [ j ] , new Indexes ( i , j ) ) ; } } int d = Integer . MIN_VALUE ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) { int abs_diff = Math . abs ( arr [ i ] - arr [ j ] ) ;
if ( map . containsKey ( abs_diff ) ) { Indexes indexes = map . get ( abs_diff ) ;
if ( indexes . getI ( ) != i && indexes . getI ( ) != j && indexes . getJ ( ) != i && indexes . getJ ( ) != j ) { d = Math . max ( d , Math . max ( arr [ i ] , arr [ j ] ) ) ; } } } } return d ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 3 , 5 , 7 , 12 } ; int n = arr . length ; int res = findFourElements ( arr , n ) ; if ( res == Integer . MIN_VALUE ) System . out . println ( " No ▁ Solution " ) ; else System . out . println ( res ) ; } }
static int CountMaximum ( int arr [ ] , int n , int k ) {
Arrays . sort ( arr ) ; int sum = 0 , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 30 , 30 , 10 , 10 } ; int n = 4 ; int k = 50 ;
System . out . println ( CountMaximum ( arr , n , k ) ) ; } }
void leftRotatebyOne ( int arr [ ] , int n ) { int i , temp ; temp = arr [ 0 ] ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
void leftRotate ( int arr [ ] , int d , int n ) { for ( int i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { RotateArray rotate = new RotateArray ( ) ; int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; rotate . leftRotate ( arr , 2 , 7 ) ; rotate . printArray ( arr , 7 ) ; } }
static void partSort ( int [ ] arr , int N , int a , int b ) {
int l = Math . min ( a , b ) ; int r = Math . max ( a , b ) ;
int [ ] temp = new int [ r - l + 1 ] ; int j = 0 ; for ( int i = l ; i <= r ; i ++ ) { temp [ j ] = arr [ i ] ; j ++ ; }
Arrays . sort ( temp ) ;
j = 0 ; for ( int i = l ; i <= r ; i ++ ) { arr [ i ] = temp [ j ] ; j ++ ; }
for ( int i = 0 ; i < N ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int [ ] arr = { 7 , 8 , 4 , 5 , 2 } ; int a = 1 , b = 4 ;
int N = arr . length ; partSort ( arr , N , a , b ) ; } }
import java . util . * ; class GFG { static int MAX_SIZE = 10 ;
static void sortByRow ( int [ ] [ ] mat , int n , boolean descending ) { int temp = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( descending == true ) { int t = i ; for ( int p = 0 ; p < n ; p ++ ) { for ( int j = p + 1 ; j < n ; j ++ ) { if ( mat [ t ] [ p ] < mat [ t ] [ j ] ) { temp = mat [ t ] [ p ] ; mat [ t ] [ p ] = mat [ t ] [ j ] ; mat [ t ] [ j ] = temp ; } } } } else Arrays . sort ( mat [ i ] ) ; } }
static void transpose ( int mat [ ] [ ] , int n ) { int temp = 0 ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ j ] [ i ] ; mat [ j ] [ i ] = temp ; } } }
static void sortMatRowAndColWise ( int mat [ ] [ ] , int n ) {
sortByRow ( mat , n , true ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n , false ) ;
transpose ( mat , n ) ; }
static void printMat ( int mat [ ] [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) System . out . print ( mat [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String args [ ] ) { int n = 3 ; int [ ] [ ] mat = { { 3 , 2 , 1 } , { 9 , 8 , 7 } , { 6 , 5 , 4 } } ; System . out . println ( " Original ▁ Matrix : " ) ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; System . out . println ( " " ▁ + ▁ " Matrix After Sorting : "); printMat ( mat , n ) ; } }
static void pushZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = arr . length ; pushZerosToEnd ( arr , n ) ; System . out . println ( " Array ▁ after ▁ pushing ▁ zeros ▁ to ▁ the ▁ back : ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
static void moveZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ; int temp ;
for ( int i = 0 ; i < n ; i ++ ) { if ( ( arr [ i ] != 0 ) ) { temp = arr [ count ] ; arr [ count ] = arr [ i ] ; arr [ i ] = temp ; count = count + 1 ; } } }
static void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 0 , 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = arr . length ; System . out . print ( " Original ▁ array : ▁ " ) ; printArray ( arr , n ) ; moveZerosToEnd ( arr , n ) ; System . out . print ( " Modified array : "); printArray ( arr , n ) ; } }
static void pushZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
static void modifyAndRearrangeArr ( int arr [ ] , int n ) {
if ( n == 1 ) return ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( ( arr [ i ] != 0 ) && ( arr [ i ] == arr [ i + 1 ] ) ) {
arr [ i ] = 2 * arr [ i ] ;
arr [ i + 1 ] = 0 ;
i ++ ; } }
pushZerosToEnd ( arr , n ) ; }
static void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 0 , 2 , 2 , 2 , 0 , 6 , 6 , 0 , 0 , 8 } ; int n = arr . length ; System . out . print ( " Original ▁ array : ▁ " ) ; printArray ( arr , n ) ; modifyAndRearrangeArr ( arr , n ) ; System . out . print ( " Modified ▁ array : ▁ " ) ; printArray ( arr , n ) ; } }
public static void swap ( int [ ] A , int i , int j ) { int temp = A [ i ] ; A [ i ] = A [ j ] ; A [ j ] = temp ; }
static void shiftAllZeroToLeft ( int array [ ] , int n ) {
int lastSeenNonZero = 0 ; for ( int index = 0 ; index < n ; index ++ ) {
if ( array [ index ] != 0 ) {
swap ( array , array [ index ] , array [ lastSeenNonZero ] ) ;
lastSeenNonZero ++ ; } } } }
static void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; }
static void RearrangePosNeg ( int arr [ ] , int n ) { int key , j ; for ( int i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
public static void main ( String [ ] args ) { int arr [ ] = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; int n = arr . length ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ; } }
static void printArray ( int A [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) System . out . print ( A [ i ] + " ▁ " ) ; System . out . println ( " " ) ; ; }
static void reverse ( int arr [ ] , int l , int r ) { if ( l < r ) { arr = swap ( arr , l , r ) ; reverse ( arr , ++ l , -- r ) ; } }
static void merge ( int arr [ ] , int l , int m , int r ) {
int i = l ;
int j = m + 1 ; while ( i <= m && arr [ i ] < 0 ) i ++ ;
while ( j <= r && arr [ j ] < 0 ) j ++ ;
reverse ( arr , i , m ) ;
reverse ( arr , m + 1 , j - 1 ) ;
reverse ( arr , i , j - 1 ) ; }
static void RearrangePosNeg ( int arr [ ] , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
RearrangePosNeg ( arr , l , m ) ; RearrangePosNeg ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } } static int [ ] swap ( int [ ] arr , int i , int j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; return arr ; }
public static void main ( String [ ] args ) { int arr [ ] = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; int arr_size = arr . length ; RearrangePosNeg ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ; } }
import java . io . * ; class GFG { public static void RearrangePosNeg ( int arr [ ] ) { int i = 0 ; int j = arr . length - 1 ; while ( true ) {
while ( arr [ i ] < 0 && i < arr . length ) i ++ ;
while ( arr [ j ] > 0 && j >= 0 ) j -- ;
if ( i < j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } else break ; } }
public static void main ( String [ ] args ) { int arr [ ] = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; RearrangePosNeg ( arr ) ; for ( int i = 0 ; i < arr . length ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
static void winner ( int arr [ ] , int N ) {
if ( N % 2 == 1 ) { System . out . print ( " A " ) ; }
else { System . out . print ( " B " ) ; } }
int arr [ ] = { 24 , 45 , 45 , 24 } ;
int N = arr . length ; winner ( arr , N ) ; } }
import java . util . * ; import java . io . * ; class GFG { static void findElements ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . length ; findElements ( arr , n ) ; } }
import java . util . * ; import java . io . * ; class GFG { static void findElements ( int arr [ ] , int n ) { Arrays . sort ( arr ) ; for ( int i = 0 ; i < n - 2 ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . length ; findElements ( arr , n ) ; } }
import java . util . * ; import java . io . * ; class GFG { static void findElements ( int arr [ ] , int n ) { int first = Integer . MIN_VALUE ; int second = Integer . MAX_VALUE ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . length ; findElements ( arr , n ) ; } }
public static int getMinOps ( int [ ] arr ) {
int res = 0 ; for ( int i = 0 ; i < arr . length - 1 ; i ++ ) {
res += Math . max ( arr [ i + 1 ] - arr [ i ] , 0 ) ; }
return res ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 3 , 4 , 1 , 2 } ; System . out . println ( getMinOps ( arr ) ) ; } }
int findFirstMissing ( int array [ ] , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
public static void main ( String [ ] args ) { SmallestMissing small = new SmallestMissing ( ) ; int arr [ ] = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = arr . length ; System . out . println ( " First ▁ Missing ▁ element ▁ is ▁ : ▁ " + small . findFirstMissing ( arr , 0 , n - 1 ) ) ; } }
int findFirstMissing ( int [ ] arr , int start , int end , int first ) { if ( start < end ) { int mid = ( start + end ) / 2 ;
if ( arr [ mid ] != mid + first ) return findFirstMissing ( arr , start , mid , first ) ; else return findFirstMissing ( arr , mid + 1 , end , first ) ; } return start + first ; }
int findSmallestMissinginSortedArray ( int [ ] arr ) {
if ( arr [ 0 ] != 0 ) return 0 ;
if ( arr [ arr . length - 1 ] == arr . length - 1 ) return arr . length ; int first = arr [ 0 ] ; return findFirstMissing ( arr , 0 , arr . length - 1 , first ) ; }
public static void main ( String [ ] args ) { GFG small = new GFG ( ) ; int arr [ ] = { 0 , 1 , 2 , 3 , 4 , 5 , 7 } ; int n = arr . length ;
System . out . println ( " First ▁ Missing ▁ element ▁ is ▁ : ▁ " + small . findSmallestMissinginSortedArray ( arr ) ) ; } }
int FindMaxSum ( int arr [ ] , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
public static void main ( String [ ] args ) { MaximumSum sum = new MaximumSum ( ) ; int arr [ ] = new int [ ] { 5 , 5 , 10 , 100 , 10 , 5 } ; System . out . println ( sum . FindMaxSum ( arr , arr . length ) ) ; } }
import java . util . * ; class GFG { static final int N = 7 ;
static int countChanges ( int matrix [ ] [ ] , int n , int m ) {
int dist = n + m - 1 ;
int [ ] [ ] freq = new int [ dist ] [ 10 ] ;
for ( int i = 0 ; i < dist ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) freq [ i ] [ j ] = 0 ; }
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
freq [ i + j ] [ matrix [ i ] [ j ] ] ++ ; } } int min_changes_sum = 0 ; for ( int i = 0 ; i < dist / 2 ; i ++ ) { int maximum = 0 ; int total_values = 0 ;
for ( int j = 0 ; j < 10 ; j ++ ) { maximum = Math . max ( maximum , freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) ; total_values += ( freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) ; }
min_changes_sum += ( total_values - maximum ) ; }
return min_changes_sum ; }
int mat [ ] [ ] = { { 1 , 2 } , { 3 , 5 } } ;
System . out . print ( countChanges ( mat , 2 , 2 ) ) ; } }
import java . io . * ; class GFG { static int MAX = 500 ;
static int [ ] [ ] lookup = new int [ MAX ] [ MAX ] ;
static void buildSparseTable ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) lookup [ i ] [ 0 ] = arr [ i ] ;
for ( int j = 1 ; ( 1 << j ) <= n ; j ++ ) {
for ( int i = 0 ; ( i + ( 1 << j ) - 1 ) < n ; i ++ ) {
if ( lookup [ i ] [ j - 1 ] < lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ) lookup [ i ] [ j ] = lookup [ i ] [ j - 1 ] ; else lookup [ i ] [ j ] = lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ; } } }
static int query ( int L , int R ) {
int j = ( int ) Math . log ( R - L + 1 ) ;
if ( lookup [ L ] [ j ] <= lookup [ R - ( 1 << j ) + 1 ] [ j ] ) return lookup [ L ] [ j ] ; else return lookup [ R - ( 1 << j ) + 1 ] [ j ] ; }
public static void main ( String [ ] args ) { int a [ ] = { 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 } ; int n = a . length ; buildSparseTable ( a , n ) ; System . out . println ( query ( 0 , 4 ) ) ; System . out . println ( query ( 4 , 7 ) ) ; System . out . println ( query ( 7 , 8 ) ) ; } }
static void minimizeWithKSwaps ( int arr [ ] , int n , int k ) { for ( int i = 0 ; i < n - 1 && k > 0 ; ++ i ) {
int pos = i ; for ( int j = i + 1 ; j < n ; ++ j ) {
if ( j - i > k ) break ;
if ( arr [ j ] < arr [ pos ] ) pos = j ; }
int temp ; for ( int j = pos ; j > i ; -- j ) { temp = arr [ j ] ; arr [ j ] = arr [ j - 1 ] ; arr [ j - 1 ] = temp ; }
k -= pos - i ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 7 , 6 , 9 , 2 , 1 } ; int n = arr . length ; int k = 3 ;
minimizeWithKSwaps ( arr , n , k ) ;
for ( int i = 0 ; i < n ; ++ i ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
static int findMaxAverage ( int [ ] arr , int n , int k ) {
if ( k > n ) return - 1 ;
int [ ] csum = new int [ n ] ; csum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) csum [ i ] = csum [ i - 1 ] + arr [ i ] ;
int max_sum = csum [ k - 1 ] , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { int curr_sum = csum [ i ] - csum [ i - k ] ; if ( curr_sum > max_sum ) { max_sum = curr_sum ; max_end = i ; } }
return max_end - k + 1 ; }
static public void main ( String [ ] args ) { int [ ] arr = { 1 , 12 , - 5 , - 6 , 50 , 3 } ; int k = 4 ; int n = arr . length ; System . out . println ( " The ▁ maximum ▁ " + " average ▁ subarray ▁ of ▁ length ▁ " + k + " ▁ begins ▁ at ▁ index ▁ " + findMaxAverage ( arr , n , k ) ) ; } }
static int findMaxAverage ( int arr [ ] , int n , int k ) {
if ( k > n ) return - 1 ;
int sum = arr [ 0 ] ; for ( int i = 1 ; i < k ; i ++ ) sum += arr [ i ] ; int max_sum = sum , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { sum = sum + arr [ i ] - arr [ i - k ] ; if ( sum > max_sum ) { max_sum = sum ; max_end = i ; } }
return max_end - k + 1 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 12 , - 5 , - 6 , 50 , 3 } ; int k = 4 ; int n = arr . length ; System . out . println ( " The ▁ maximum ▁ average " + " ▁ subarray ▁ of ▁ length ▁ " + k + " ▁ begins ▁ at ▁ index ▁ " + findMaxAverage ( arr , n , k ) ) ; } }
static HashMap < R , Integer > m = new HashMap < > ( ) ;
public static int findMinimum ( int [ ] arr , int N , int pos , int turn ) {
R x = new R ( pos , turn ) ; if ( m . containsKey ( x ) ) { return m . get ( x ) ; }
if ( pos >= N - 1 ) { return 0 ; }
if ( turn == 0 ) {
int ans = Math . min ( findMinimum ( arr , N , pos + 1 , 1 ) + arr [ pos ] , findMinimum ( arr , N , pos + 2 , 1 ) + arr [ pos ] + arr [ pos + 1 ] ) ;
R v = new R ( pos , turn ) ; m . put ( v , ans ) ;
return ans ; }
if ( turn != 0 ) {
int ans = Math . min ( findMinimum ( arr , N , pos + 1 , 0 ) , findMinimum ( arr , N , pos + 2 , 0 ) ) ;
R v = new R ( pos , turn ) ; m . put ( v , ans ) ;
return ans ; } return 0 ; }
public static int countPenality ( int [ ] arr , int N ) {
int pos = 0 ;
int turn = 0 ;
return findMinimum ( arr , N , pos , turn ) + 1 ; }
public static void printAnswer ( int [ ] arr , int N ) {
int a = countPenality ( arr , N ) ;
int sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
System . out . println ( a ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 } ; int N = 8 ; printAnswer ( arr , N ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int MAX = 1000001 ; static int prime [ ] = new int [ MAX ] ;
static void SieveOfEratosthenes ( ) {
Arrays . fill ( prime , 1 ) ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( prime [ p ] == 1 ) {
for ( int i = p * p ; i <= MAX - 1 ; i += p ) prime [ i ] = 0 ; } } }
static int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
static int getSumUtil ( int [ ] st , int ss , int se , int qs , int qe , int si ) {
if ( qs <= ss && qe >= se ) return st [ si ] ;
if ( se < qs ss > qe ) return 0 ;
int mid = getMid ( ss , se ) ; return getSumUtil ( st , ss , mid , qs , qe , 2 * si + 1 ) + getSumUtil ( st , mid + 1 , se , qs , qe , 2 * si + 2 ) ; }
static void updateValueUtil ( int [ ] st , int ss , int se , int i , int diff , int si ) {
if ( i < ss i > se ) return ;
st [ si ] = st [ si ] + diff ; if ( se != ss ) { int mid = getMid ( ss , se ) ; updateValueUtil ( st , ss , mid , i , diff , 2 * si + 1 ) ; updateValueUtil ( st , mid + 1 , se , i , diff , 2 * si + 2 ) ; } }
static void updateValue ( int arr [ ] , int [ ] st , int n , int i , int new_val ) {
if ( i < 0 i > n - 1 ) { System . out . print ( " - 1" ) ; return ; }
int diff = new_val - arr [ i ] ; int prev_val = arr [ i ] ;
arr [ i ] = new_val ;
if ( ( prime [ new_val ] prime [ prev_val ] ) != 0 ) {
if ( prime [ prev_val ] == 0 ) updateValueUtil ( st , 0 , n - 1 , i , new_val , 0 ) ;
else if ( prime [ new_val ] == 0 ) updateValueUtil ( st , 0 , n - 1 , i , - prev_val , 0 ) ;
else updateValueUtil ( st , 0 , n - 1 , i , diff , 0 ) ; } }
static int getSum ( int [ ] st , int n , int qs , int qe ) {
if ( qs < 0 qe > n - 1 qs > qe ) { System . out . println ( " - 1" ) ; return - 1 ; } return getSumUtil ( st , 0 , n - 1 , qs , qe , 0 ) ; }
static int constructSTUtil ( int arr [ ] , int ss , int se , int [ ] st , int si ) {
if ( ss == se ) {
if ( prime [ arr [ ss ] ] != 0 ) st [ si ] = arr [ ss ] ; else st [ si ] = 0 ; return st [ si ] ; }
int mid = getMid ( ss , se ) ; st [ si ] = constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) + constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ; return st [ si ] ; }
static int [ ] constructST ( int arr [ ] , int n ) {
int x = ( int ) ( Math . ceil ( Math . log ( n ) / Math . log ( 2 ) ) ) ;
int max_size = 2 * ( int ) Math . pow ( 2 , x ) - 1 ;
int [ ] st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 5 , 7 , 9 , 11 } ; int n = arr . length ; int Q [ ] [ ] = { { 1 , 1 , 3 } , { 2 , 1 , 10 } , { 1 , 1 , 3 } } ;
SieveOfEratosthenes ( ) ;
int [ ] st = constructST ( arr , n ) ;
System . out . println ( getSum ( st , n , 1 , 3 ) ) ;
updateValue ( arr , st , n , 1 , 10 ) ;
System . out . println ( getSum ( st , n , 1 , 3 ) ) ; } }
import java . util . * ; class GFG { static int mod = 1000000007 ; static int [ ] [ ] dp = new int [ 1000 ] [ 1000 ] ; static int calculate ( int pos , int prev , String s , Vector < Integer > index ) {
if ( pos == s . length ( ) ) return 1 ;
if ( dp [ pos ] [ prev ] != - 1 ) return dp [ pos ] [ prev ] ;
int answer = 0 ; for ( int i = 0 ; i < index . size ( ) ; i ++ ) { if ( index . get ( i ) . compareTo ( prev ) >= 0 ) { answer = ( answer % mod + calculate ( pos + 1 , index . get ( i ) , s , index ) % mod ) % mod ; } }
return dp [ pos ] [ prev ] = answer ; } static int countWays ( Vector < String > a , String s ) { int n = a . size ( ) ;
Vector < Integer > [ ] index = new Vector [ 26 ] ; for ( int i = 0 ; i < 26 ; i ++ ) index [ i ] = new Vector < Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < a . get ( i ) . length ( ) ; j ++ ) {
index [ a . get ( i ) . charAt ( j ) - ' a ' ] . add ( j + 1 ) ; } }
for ( int i = 0 ; i < 1000 ; i ++ ) { for ( int j = 0 ; j < 1000 ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } return calculate ( 0 , 0 , s , index [ 0 ] ) ; }
public static void main ( String [ ] args ) { Vector < String > A = new Vector < String > ( ) ; A . add ( " adc " ) ; A . add ( " aec " ) ; A . add ( " erg " ) ; String S = " ac " ; System . out . print ( countWays ( A , S ) ) ; } }
import java . util . * ; class GFG { static final int MAX = 100005 ; static final int MOD = 1000000007 ;
static int [ ] [ ] [ ] dp = new int [ MAX ] [ 101 ] [ 2 ] ;
static int countNum ( int idx , int sum , int tight , Vector < Integer > num , int len , int k ) { if ( len == idx ) { if ( sum == 0 ) return 1 ; else return 0 ; } if ( dp [ idx ] [ sum ] [ tight ] != - 1 ) return dp [ idx ] [ sum ] [ tight ] ; int res = 0 , limit ;
if ( tight == 0 ) { limit = num . get ( idx ) ; }
else { limit = 9 ; } for ( int i = 0 ; i <= limit ; i ++ ) {
int new_tight = tight ; if ( tight == 0 && i < limit ) new_tight = 1 ; res += countNum ( idx + 1 , ( sum + i ) % k , new_tight , num , len , k ) ; res %= MOD ; }
if ( res < 0 ) res += MOD ; return dp [ idx ] [ sum ] [ tight ] = res ; }
static Vector < Integer > process ( String s ) { Vector < Integer > num = new Vector < Integer > ( ) ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { num . add ( s . charAt ( i ) - '0' ) ; } return num ; }
String n = "98765432109876543210" ;
int len = n . length ( ) ; int k = 58 ;
Vector < Integer > num = process ( n ) ; System . out . print ( countNum ( 0 , 0 , 0 , num , len , k ) ) ; } }
import java . util . * ; import java . lang . * ; class GFG { public static int [ ] countSum ( int arr [ ] , int n ) { int result = 0 ;
int [ ] countODD = new int [ n + 1 ] ; int [ ] countEVEN = new int [ n + 1 ] ;
countODD [ 0 ] = 0 ; countEVEN [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) {
if ( arr [ i - 1 ] % 2 == 0 ) { countEVEN [ i ] = countEVEN [ i - 1 ] + countEVEN [ i - 1 ] + 1 ; countODD [ i ] = countODD [ i - 1 ] + countODD [ i - 1 ] ; }
else { countEVEN [ i ] = countEVEN [ i - 1 ] + countODD [ i - 1 ] ; countODD [ i ] = countODD [ i - 1 ] + countEVEN [ i - 1 ] + 1 ; } } int [ ] ans = new int [ 2 ] ; ans [ 0 ] = countEVEN [ n ] ; ans [ 1 ] = countODD [ n ] ; return ans ; }
public static void main ( String [ ] args ) { int [ ] arr = new int [ ] { 1 , 2 , 2 , 3 } ; int n = 4 ; int [ ] ans = countSum ( arr , n ) ; System . out . println ( " EvenSum ▁ = ▁ " + ans [ 0 ] ) ; System . out . println ( " OddSum ▁ = ▁ " + ans [ 1 ] ) ; } }
class GFG { static int maxN = 31 ; static int maxW = 31 ;
static int dp [ ] [ ] [ ] = new int [ maxN ] [ maxW ] [ maxW ] ;
static int maxWeight ( int arr [ ] , int n , int w1_r , int w2_r , int i ) {
if ( i == n ) return 0 ; if ( dp [ i ] [ w1_r ] [ w2_r ] != - 1 ) return dp [ i ] [ w1_r ] [ w2_r ] ;
int fill_w1 = 0 , fill_w2 = 0 , fill_none = 0 ; if ( w1_r >= arr [ i ] ) fill_w1 = arr [ i ] + maxWeight ( arr , n , w1_r - arr [ i ] , w2_r , i + 1 ) ; if ( w2_r >= arr [ i ] ) fill_w2 = arr [ i ] + maxWeight ( arr , n , w1_r , w2_r - arr [ i ] , i + 1 ) ; fill_none = maxWeight ( arr , n , w1_r , w2_r , i + 1 ) ;
dp [ i ] [ w1_r ] [ w2_r ] = Math . max ( fill_none , Math . max ( fill_w1 , fill_w2 ) ) ; return dp [ i ] [ w1_r ] [ w2_r ] ; }
int arr [ ] = { 8 , 2 , 3 } ;
int n = arr . length ;
int w1 = 10 , w2 = 3 ;
System . out . println ( maxWeight ( arr , n , w1 , w2 , 0 ) ) ; } }
import java . util . * ; class GFG { static int n = 3 ;
static void findPrefixCount ( int p_arr [ ] [ ] , boolean set_bit [ ] [ ] ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = n - 1 ; j >= 0 ; j -- ) { if ( ! set_bit [ i ] [ j ] ) continue ; if ( j != n - 1 ) p_arr [ i ] [ j ] += p_arr [ i ] [ j + 1 ] ; p_arr [ i ] [ j ] += ( set_bit [ i ] [ j ] ) ? 1 : 0 ; } } } static class pair { int first , second ; pair ( ) { } pair ( int a , int b ) { first = a ; second = b ; } }
static int matrixAllOne ( boolean set_bit [ ] [ ] ) {
int p_arr [ ] [ ] = new int [ n ] [ n ] ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) p_arr [ i ] [ j ] = 0 ; findPrefixCount ( p_arr , set_bit ) ;
int ans = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { int i = n - 1 ;
Stack < pair > q = new Stack < pair > ( ) ;
int to_sum = 0 ; while ( i >= 0 ) { int c = 0 ; while ( q . size ( ) != 0 && q . peek ( ) . first > p_arr [ i ] [ j ] ) { to_sum -= ( q . peek ( ) . second + 1 ) * ( q . peek ( ) . first - p_arr [ i ] [ j ] ) ; c += q . peek ( ) . second + 1 ; q . pop ( ) ; } to_sum += p_arr [ i ] [ j ] ; ans += to_sum ; q . push ( new pair ( p_arr [ i ] [ j ] , c ) ) ; i -- ; } } return ans ; }
static int sumAndMatrix ( int arr [ ] [ ] ) { int sum = 0 ; int mul = 1 ; for ( int i = 0 ; i < 30 ; i ++ ) {
boolean set_bit [ ] [ ] = new boolean [ n ] [ n ] ; for ( int R = 0 ; R < n ; R ++ ) for ( int C = 0 ; C < n ; C ++ ) set_bit [ R ] [ C ] = ( ( arr [ R ] [ C ] & ( 1 << i ) ) != 0 ) ; sum += ( mul * matrixAllOne ( set_bit ) ) ; mul *= 2 ; } return sum ; }
public static void main ( String args [ ] ) { int arr [ ] [ ] = { { 9 , 7 , 4 } , { 8 , 9 , 2 } , { 11 , 11 , 5 } } ; System . out . println ( sumAndMatrix ( arr ) ) ; } }
import java . io . * ; class GFG { static int CountWays ( int n ) {
int noOfWays [ ] = new int [ n + 3 ] ; noOfWays [ 0 ] = 1 ; noOfWays [ 1 ] = 1 ; noOfWays [ 2 ] = 1 + 1 ;
for ( int i = 3 ; i < n + 1 ; i ++ ) { noOfWays [ i ] =
noOfWays [ 3 - 1 ]
+ noOfWays [ 3 - 3 ] ;
noOfWays [ 0 ] = noOfWays [ 1 ] ; noOfWays [ 1 ] = noOfWays [ 2 ] ; noOfWays [ 2 ] = noOfWays [ i ] ; } return noOfWays [ n ] ; }
public static void main ( String [ ] args ) { int n = 5 ; System . out . println ( CountWays ( n ) ) ; } }
import java . util . * ; public class Main { static class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } } static int MAX = 105 , q = 0 ; static int [ ] prime = new int [ MAX ] ; static void sieve ( ) { for ( int i = 2 ; i * i < MAX ; i ++ ) { if ( prime [ i ] == 0 ) { for ( int j = i * i ; j < MAX ; j += i ) prime [ j ] = 1 ; } } }
static void dfs ( int i , int j , int k , int n , int m , int [ ] [ ] mappedMatrix , int [ ] [ ] mark , pair [ ] ans ) {
if ( ( mappedMatrix [ i ] [ j ] == 0 ? true : false ) || ( i > n ? true : false ) || ( j > m ? true : false ) || ( mark [ i ] [ j ] != 0 ? true : false ) || ( q != 0 ? true : false ) ) return ;
mark [ i ] [ j ] = 1 ;
ans [ k ] = new pair ( i , j ) ;
if ( i == n && j == m ) {
( q ) = k ; return ; }
dfs ( i + 1 , j + 1 , k + 1 , n , m , mappedMatrix , mark , ans ) ;
dfs ( i + 1 , j , k + 1 , n , m , mappedMatrix , mark , ans ) ;
dfs ( i , j + 1 , k + 1 , n , m , mappedMatrix , mark , ans ) ; }
static void lexicographicalPath ( int n , int m , int [ ] [ ] mappedMatrix ) {
pair [ ] ans = new pair [ MAX ] ;
int [ ] [ ] mark = new int [ MAX ] [ MAX ] ;
dfs ( 1 , 1 , 1 , n , m , mappedMatrix , mark , ans ) ; int [ ] [ ] anss = { { 1 , 1 } , { 2 , 1 } , { 3 , 2 } , { 3 , 3 } } ;
for ( int i = 0 ; i < 4 ; i ++ ) System . out . println ( anss [ i ] [ 0 ] + " ▁ " + anss [ i ] [ 1 ] ) ; }
static void countPrimePath ( int [ ] [ ] mappedMatrix , int n , int m ) { int [ ] [ ] dp = new int [ MAX ] [ MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) { for ( int j = 0 ; j < MAX ; j ++ ) { dp [ i ] [ j ] = 0 ; } } dp [ 1 ] [ 1 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= m ; j ++ ) {
if ( i == 1 && j == 1 ) continue ; dp [ i ] [ j ] = ( dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] + dp [ i - 1 ] [ j - 1 ] ) ;
if ( mappedMatrix [ i ] [ j ] == 0 ) dp [ i ] [ j ] = 0 ; } } System . out . println ( dp [ n ] [ m ] ) ; }
static void preprocessMatrix ( int [ ] [ ] mappedMatrix , int [ ] [ ] a , int n , int m ) {
sieve ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
if ( prime [ a [ i ] [ j ] ] == 0 ) mappedMatrix [ i + 1 ] [ j + 1 ] = 1 ;
else mappedMatrix [ i + 1 ] [ j + 1 ] = 0 ; } } }
public static void main ( String [ ] args ) { int n = 3 ; int m = 3 ; int [ ] [ ] a = { { 2 , 3 , 7 } , { 5 , 4 , 2 } , { 3 , 7 , 11 } } ; int [ ] [ ] mappedMatrix = new int [ MAX ] [ MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) { for ( int j = 0 ; j < MAX ; j ++ ) { mappedMatrix [ i ] [ j ] = 0 ; } } preprocessMatrix ( mappedMatrix , a , n , m ) ; countPrimePath ( mappedMatrix , n , m ) ; lexicographicalPath ( n , m , mappedMatrix ) ; } }
static int isSubsetSum ( int set [ ] , int n , int sum ) {
boolean subset [ ] [ ] = new boolean [ sum + 1 ] [ n + 1 ] ; int count [ ] [ ] = new int [ sum + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { subset [ 0 ] [ i ] = true ; count [ 0 ] [ i ] = 0 ; }
for ( int i = 1 ; i <= sum ; i ++ ) { subset [ i ] [ 0 ] = false ; count [ i ] [ 0 ] = - 1 ; }
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; count [ i ] [ j ] = count [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) { subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; if ( subset [ i ] [ j ] ) count [ i ] [ j ] = Math . max ( count [ i ] [ j - 1 ] , count [ i - set [ j - 1 ] ] [ j - 1 ] + 1 ) ; } } } return count [ sum ] [ n ] ; }
public static void main ( String args [ ] ) { int set [ ] = { 2 , 3 , 5 , 10 } ; int sum = 20 ; int n = set . length ; System . out . println ( isSubsetSum ( set , n , sum ) ) ; } }
class GFG { static int MAX = 100 ;
static int lcslen = 0 ;
static int [ ] [ ] dp = new int [ MAX ] [ MAX ] ;
static int lcs ( String str1 , String str2 , int len1 , int len2 , int i , int j ) { int ret = dp [ i ] [ j ] ;
if ( i == len1 j == len2 ) return ret = 0 ;
if ( ret != - 1 ) return ret ; ret = 0 ;
if ( str1 . charAt ( i ) == str2 . charAt ( j ) ) ret = 1 + lcs ( str1 , str2 , len1 , len2 , i + 1 , j + 1 ) ; else ret = Math . max ( lcs ( str1 , str2 , len1 , len2 , i + 1 , j ) , lcs ( str1 , str2 , len1 , len2 , i , j + 1 ) ) ; return ret ; }
static void printAll ( String str1 , String str2 , int len1 , int len2 , char [ ] data , int indx1 , int indx2 , int currlcs ) {
if ( currlcs == lcslen ) { data [ currlcs ] = ' \0' ; System . out . println ( new String ( data ) ) ; return ; }
if ( indx1 == len1 indx2 == len2 ) return ;
for ( char ch = ' a ' ; ch <= ' z ' ; ch ++ ) {
boolean done = false ; for ( int i = indx1 ; i < len1 ; i ++ ) {
if ( ch == str1 . charAt ( i ) ) { for ( int j = indx2 ; j < len2 ; j ++ ) {
if ( ch == str2 . charAt ( j ) && dp [ i ] [ j ] == lcslen - currlcs ) { data [ currlcs ] = ch ; printAll ( str1 , str2 , len1 , len2 , data , i + 1 , j + 1 , currlcs + 1 ) ; done = true ; break ; } } }
if ( done ) break ; } } }
static void prinlAllLCSSorted ( String str1 , String str2 ) {
int len1 = str1 . length ( ) , len2 = str2 . length ( ) ;
for ( int i = 0 ; i < MAX ; i ++ ) { for ( int j = 0 ; j < MAX ; j ++ ) { dp [ i ] [ j ] = - 1 ; } } lcslen = lcs ( str1 , str2 , len1 , len2 , 0 , 0 ) ;
char [ ] data = new char [ MAX ] ; printAll ( str1 , str2 , len1 , len2 , data , 0 , 0 , 0 ) ; }
public static void main ( String [ ] args ) { String str1 = " abcabcaa " , str2 = " acbacba " ; prinlAllLCSSorted ( str1 , str2 ) ; } }
import java . io . * ; class Majority { static boolean isMajority ( int arr [ ] , int n , int x ) { int i , last_index = 0 ;
last_index = ( n % 2 == 0 ) ? n / 2 : n / 2 + 1 ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + n / 2 ] == x ) return true ; } return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 4 , 4 , 4 } ; int n = arr . length ; int x = 4 ; if ( isMajority ( arr , n , x ) == true ) System . out . println ( x + " ▁ appears ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; else System . out . println ( x + " ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; } }
static int _binarySearch ( int arr [ ] , int low , int high , int x ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( ( mid == 0 x > arr [ mid - 1 ] ) && ( arr [ mid ] == x ) ) return mid ; else if ( x > arr [ mid ] ) return _binarySearch ( arr , ( mid + 1 ) , high , x ) ; else return _binarySearch ( arr , low , ( mid - 1 ) , x ) ; } return - 1 ; }
static boolean isMajority ( int arr [ ] , int n , int x ) {
int i = _binarySearch ( arr , 0 , n - 1 , x ) ;
if ( i == - 1 ) return false ;
if ( ( ( i + n / 2 ) <= ( n - 1 ) ) && arr [ i + n / 2 ] == x ) return true ; else return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = arr . length ; int x = 3 ; if ( isMajority ( arr , n , x ) == true ) System . out . println ( x + " ▁ appears ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; else System . out . println ( x + " ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; } }
import java . util . * ; class GFG { static boolean isMajorityElement ( int arr [ ] , int n , int key ) { if ( arr [ n / 2 ] == key ) return true ; else return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = arr . length ; int x = 3 ; if ( isMajorityElement ( arr , n , x ) ) System . out . printf ( " % d ▁ appears ▁ more ▁ than ▁ % d ▁ " + " times ▁ in ▁ arr [ ] " , x , n / 2 ) ; else System . out . printf ( " % d ▁ does ▁ not ▁ appear ▁ more ▁ " + " than ▁ % d ▁ times ▁ in ▁ " + " arr [ ] " , x , n / 2 ) ; } }
static int cutRod ( int price [ ] , int n ) { int val [ ] = new int [ n + 1 ] ; val [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { int max_val = Integer . MIN_VALUE ; for ( int j = 0 ; j < i ; j ++ ) max_val = Math . max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . length ; System . out . println ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
public static boolean isPossible ( int [ ] target ) {
int max = 0 ;
int index = 0 ;
for ( int i = 0 ; i < target . length ; i ++ ) {
if ( max < target [ i ] ) { max = target [ i ] ; index = i ; } }
if ( max == 1 ) return true ;
for ( int i = 0 ; i < target . length ; i ++ ) {
if ( i != index ) {
max -= target [ i ] ;
if ( max <= 0 ) return false ; } }
target [ index ] = max ;
return isPossible ( target ) ; }
public static void main ( String [ ] args ) { int [ ] target = { 9 , 3 , 5 } ; boolean res = isPossible ( target ) ; if ( res ) { System . out . println ( " YES " ) ; } else { System . out . println ( " NO " ) ; } } }
static int nCr ( int n , int r ) {
int res = 1 ;
if ( r > n - r ) r = n - r ;
for ( int i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
public static void main ( String [ ] args ) { int n = 3 , m = 2 , k = 2 ; System . out . print ( nCr ( n + m , k ) ) ; } }
static void Is_possible ( long N ) { long C = 0 ; long D = 0 ;
while ( N % 10 == 0 ) { N = N / 10 ; C += 1 ; }
if ( Math . pow ( 2 , ( long ) ( Math . log ( N ) / ( Math . log ( 2 ) ) ) ) == N ) { D = ( long ) ( Math . log ( N ) / ( Math . log ( 2 ) ) ) ;
if ( C >= D ) System . out . print ( " YES " ) ; else System . out . print ( " NO " ) ; } else System . out . print ( " NO " ) ; }
public static void main ( String args [ ] ) { long N = 2000000000000L ; Is_possible ( N ) ; } }
static void findNthTerm ( int n ) { System . out . println ( n * n - n + 1 ) ; }
public static void main ( String [ ] args ) { int N = 4 ; findNthTerm ( N ) ; } }
static int rev ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; }
return rev_num ; }
static int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= Math . sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += rev ( i ) ; else result += ( rev ( i ) + rev ( num / i ) ) ; } }
return ( result + 1 ) ; }
static boolean isAntiPerfect ( int n ) { return divSum ( n ) == n ; }
int N = 244 ;
if ( isAntiPerfect ( N ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static void printSeries ( int n , int a , int b , int c ) { int d ;
if ( n == 1 ) { System . out . print ( a + " ▁ " ) ; return ; } if ( n == 2 ) { System . out . print ( a + " ▁ " + b + " ▁ " ) ; return ; } System . out . print ( a + " ▁ " + b + " ▁ " + c + " ▁ " ) ; for ( int i = 4 ; i <= n ; i ++ ) { d = a + b + c ; System . out . print ( d + " ▁ " ) ; a = b ; b = c ; c = d ; } }
public static void main ( String [ ] args ) { int N = 7 , a = 1 , b = 3 ; int c = 4 ;
printSeries ( N , a , b , c ) ; } }
static int diameter ( int n ) {
int L , H , templen ; L = 1 ;
H = 0 ;
if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 2 ; } if ( n == 3 ) { return 3 ; }
while ( L * 2 <= n ) { L *= 2 ; H ++ ; }
if ( n >= L * 2 - 1 ) return 2 * H + 1 ; else if ( n >= L + ( L / 2 ) - 1 ) return 2 * H ; return 2 * H - 1 ; }
public static void main ( String [ ] args ) { int n = 15 ; System . out . println ( diameter ( n ) ) ; } }
static void compareValues ( int a , int b , int c , int d ) {
double log1 = Math . log10 ( a ) ; double num1 = log1 * b ;
double log2 = Math . log10 ( c ) ; double num2 = log2 * d ;
if ( num1 > num2 ) System . out . println ( a + " ^ " + b ) ; else System . out . println ( c + " ^ " + d ) ; }
public static void main ( String [ ] args ) { int a = 8 , b = 29 , c = 60 , d = 59 ; compareValues ( a , b , c , d ) ; } }
import java . util . * ; class GFG { static int MAX = 100005 ;
static Vector < Integer > addPrimes ( ) { int n = MAX ; boolean [ ] prime = new boolean [ n + 1 ] ; Arrays . fill ( prime , true ) ; for ( int p = 2 ; p * p <= n ; p ++ ) { if ( prime [ p ] == true ) { for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } Vector < Integer > ans = new Vector < Integer > ( ) ;
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) ans . add ( p ) ; return ans ; }
static boolean is_prime ( int n ) { return ( n == 3 n == 5 n == 7 ) ; }
static int find_Sum ( int n ) {
int sum = 0 ;
Vector < Integer > v = addPrimes ( ) ;
for ( int i = 0 ; i < v . size ( ) && n > 0 ; i ++ ) {
int flag = 1 ; int a = v . get ( i ) ;
while ( a != 0 ) { int d = a % 10 ; a = a / 10 ; if ( is_prime ( d ) ) { flag = 0 ; break ; } }
if ( flag == 1 ) { n -- ; sum = sum + v . get ( i ) ; } }
return sum ; }
public static void main ( String [ ] args ) { int n = 7 ;
System . out . println ( find_Sum ( n ) ) ; } }
static int primeCount ( int arr [ ] , int n ) {
int max_val = max_element ( arr ) ;
boolean prime [ ] = new boolean [ max_val + 1 ] ; for ( int p = 0 ; p <= max_val ; p ++ ) prime [ p ] = true ;
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= max_val ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i <= max_val ; i += p ) prime [ i ] = false ; } }
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( prime [ arr [ i ] ] ) count ++ ; return count ; }
static int [ ] getPrefixArray ( int arr [ ] , int n , int pre [ ] ) {
pre [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { pre [ i ] = pre [ i - 1 ] + arr [ i ] ; } return pre ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 4 , 8 , 4 } ; int n = arr . length ;
int pre [ ] = new int [ n ] ; pre = getPrefixArray ( arr , n , pre ) ;
System . out . println ( primeCount ( pre , n ) ) ; } }
static int minValue ( int n , int x , int y ) {
float val = ( y * n ) / 100 ;
if ( x >= val ) return 0 ; else return ( int ) ( Math . ceil ( val ) - x ) ; }
public static void main ( String [ ] args ) { int n = 10 , x = 2 , y = 40 ; System . out . println ( minValue ( n , x , y ) ) ; } }
static boolean isPrime ( long n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static boolean isFactorialPrime ( long n ) {
if ( ! isPrime ( n ) ) return false ; long fact = 1 ; int i = 1 ; while ( fact <= n + 1 ) {
fact = fact * i ;
if ( n + 1 == fact n - 1 == fact ) return true ; i ++ ; }
return false ; }
public static void main ( String args [ ] ) { int n = 23 ; if ( isFactorialPrime ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
public static void main ( String [ ] args ) {
long n = 5 ;
long fac1 = 1 ; for ( int i = 2 ; i <= n - 1 ; i ++ ) fac1 = fac1 * i ;
long fac2 = fac1 * n ;
long totalWays = fac1 * fac2 ;
System . out . println ( totalWays ) ; } }
import java . util . * ; class GFG { static final int MAX = 10000 ; static Vector < Integer > arr = new Vector < Integer > ( ) ;
static void SieveOfEratosthenes ( ) {
boolean [ ] prime = new boolean [ MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) prime [ i ] = true ; for ( int p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
for ( int p = 2 ; p < MAX ; p ++ ) if ( prime [ p ] ) arr . add ( p ) ; }
static boolean isEuclid ( long n ) { long product = 1 ; int i = 0 ; while ( product < n ) {
product = product * arr . get ( i ) ; if ( product + 1 == n ) return true ; i ++ ; } return false ; }
SieveOfEratosthenes ( ) ;
long n = 31 ;
if ( isEuclid ( n ) ) System . out . println ( " YES " ) ; else System . out . println ( " NO " ) ;
n = 42 ;
if ( isEuclid ( n ) ) System . out . println ( " YES " ) ; else System . out . println ( " NO " ) ; } }
static int nextPerfectCube ( int N ) { int nextN = ( int ) Math . floor ( Math . cbrt ( N ) ) + 1 ; return nextN * nextN * nextN ; }
public static void main ( String args [ ] ) { int n = 35 ; System . out . print ( nextPerfectCube ( n ) ) ; } }
static boolean isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static int SumOfPrimeDivisors ( int n ) { int sum = 0 ;
int root_n = ( int ) Math . sqrt ( n ) ; for ( int i = 1 ; i <= root_n ; i ++ ) { if ( n % i == 0 ) {
if ( i == n / i && isPrime ( i ) ) { sum += i ; } else {
if ( isPrime ( i ) ) { sum += i ; } if ( isPrime ( n / i ) ) { sum += ( n / i ) ; } } } } return sum ; }
public static void main ( String [ ] args ) { int n = 60 ; System . out . println ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " + SumOfPrimeDivisors ( n ) ) ; } }
class GFG { static int findpos ( String n ) { int pos = 0 ; for ( int i = 0 ; i < n . length ( ) ; i ++ ) { switch ( n . charAt ( i ) ) {
case '2' : pos = pos * 4 + 1 ; break ;
case '3' : pos = pos * 4 + 2 ; break ;
case '5' : pos = pos * 4 + 3 ; break ;
case '7' : pos = pos * 4 + 4 ; break ; } } return pos ; }
public static void main ( String args [ ] ) { String n = "777" ; System . out . println ( findpos ( n ) ) ; } }
static void possibleTripletInRange ( int L , int R ) { boolean flag = false ; int possibleA = 0 , possibleB = 0 , possibleC = 0 ; int numbersInRange = ( R - L + 1 ) ;
if ( numbersInRange < 3 ) { flag = false ; }
else if ( numbersInRange > 3 ) { flag = true ;
if ( L % 2 > 0 ) { L ++ ; } possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
if ( ! ( L % 2 > 0 ) ) { flag = true ; possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
flag = false ; } }
if ( flag == true ) { System . out . println ( " ( " + possibleA + " , ▁ " + possibleB + " , ▁ " + possibleC + " ) " + " ▁ is ▁ one ▁ such ▁ possible " + " ▁ triplet ▁ between ▁ " + L + " ▁ and ▁ " + R ) ; } else { System . out . println ( " No ▁ Such ▁ Triplet " + " ▁ exists ▁ between ▁ " + L + " ▁ and ▁ " + R ) ; } }
public static void main ( String [ ] args ) { int L , R ;
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ; } }
import java . lang . * ; class GFG { static final int mod = 1000000007 ;
static int digitNumber ( long n ) {
if ( n == 0 ) return 1 ;
if ( n == 1 ) return 9 ;
if ( n % 2 != 0 ) {
int temp = digitNumber ( ( n - 1 ) / 2 ) % mod ; return ( 9 * ( temp * temp ) % mod ) % mod ; } else {
int temp = digitNumber ( n / 2 ) % mod ; return ( temp * temp ) % mod ; } } static int countExcluding ( int n , int d ) {
if ( d == 0 ) return ( 9 * digitNumber ( n - 1 ) ) % mod ; else return ( 8 * digitNumber ( n - 1 ) ) % mod ; }
int d = 9 ; int n = 3 ; System . out . println ( countExcluding ( n , d ) ) ; } }
public static boolean isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
public static boolean isEmirp ( int n ) {
if ( isPrime ( n ) == false ) return false ;
int rev = 0 ; while ( n != 0 ) { int d = n % 10 ; rev = rev * 10 + d ; n /= 10 ; }
return isPrime ( rev ) ; }
int n = 13 ; if ( isEmirp ( n ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static double Convert ( double radian ) { double pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
public static void main ( String [ ] args ) { double radian = 5.0 ; double degree = Convert ( radian ) ; System . out . println ( " degree ▁ = ▁ " + degree ) ; } }
static int sn ( int n , int an ) { return ( n * ( 1 + an ) ) / 2 ; }
static int trace ( int n , int m ) {
int an = 1 + ( n - 1 ) * ( m + 1 ) ;
int rowmajorSum = sn ( n , an ) ;
an = 1 + ( n - 1 ) * ( n + 1 ) ;
int colmajorSum = sn ( n , an ) ; return rowmajorSum + colmajorSum ; }
static public void main ( String [ ] args ) { int N = 3 , M = 3 ; System . out . println ( trace ( N , M ) ) ; } }
static void max_area ( int n , int m , int k ) { if ( k > ( n + m - 2 ) ) System . out . println ( " Not ▁ possible " ) ; else { int result ;
if ( k < Math . max ( m , n ) - 1 ) { result = Math . max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; }
else { result = Math . max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; }
System . out . println ( result ) ; } }
public static void main ( String [ ] args ) { int n = 3 , m = 4 , k = 1 ; max_area ( n , m , k ) ; } }
static int area_fun ( int side ) { int area = side * side ; return area ; }
public static void main ( String arg [ ] ) { int side = 4 ; int area = area_fun ( side ) ; System . out . println ( area ) ; } }
static int countConsecutive ( int N ) {
int count = 0 ; for ( int L = 1 ; L * ( L + 1 ) < 2 * N ; L ++ ) { double a = ( double ) ( ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) ) ; if ( a - ( int ) a == 0.0 ) count ++ ; } return count ; }
public static void main ( String [ ] args ) { int N = 15 ; System . out . println ( countConsecutive ( N ) ) ; N = 10 ; System . out . println ( countConsecutive ( N ) ) ; } }
static boolean isAutomorphic ( int N ) {
int sq = N * N ;
while ( N > 0 ) {
if ( N % 10 != sq % 10 ) return false ;
N /= 10 ; sq /= 10 ; } return true ; }
public static void main ( String [ ] args ) { int N = 5 ; System . out . println ( isAutomorphic ( N ) ? " Automorphic " : " Not ▁ Automorphic " ) ; } }
static int maxPrimefactorNum ( int N ) {
boolean arr [ ] = new boolean [ N + 5 ] ;
for ( int i = 3 ; i * i <= N ; i += 2 ) { if ( ! arr [ i ] ) { for ( int j = i * i ; j <= N ; j += i ) { arr [ j ] = true ; } } }
Vector < Integer > prime = new Vector < > ( ) ; prime . add ( prime . size ( ) , 2 ) ; for ( int i = 3 ; i <= N ; i += 2 ) { if ( ! arr [ i ] ) { prime . add ( prime . size ( ) , i ) ; } }
int i = 0 , ans = 1 ; while ( ans * prime . get ( i ) <= N && i < prime . size ( ) ) { ans *= prime . get ( i ) ; i ++ ; } return ans ; }
public static void main ( String [ ] args ) { int N = 40 ; System . out . println ( maxPrimefactorNum ( N ) ) ; } }
import java . io . * ; class GFG { static int highestPowerof2 ( int x ) {
x |= x >> 1 ; x |= x >> 2 ; x |= x >> 4 ; x |= x >> 8 ; x |= x >> 16 ;
return x ^ ( x >> 1 ) ; }
public static void main ( String [ ] args ) { int n = 10 ; System . out . println ( highestPowerof2 ( n ) ) ; } }
static int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= Math . sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; }
public static void main ( String [ ] args ) { int num = 36 ; System . out . println ( divSum ( num ) ) ; } }
static int power ( int x , int y , int p ) {
while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
static void squareRoot ( int n , int p ) { if ( p % 4 != 3 ) { System . out . print ( " Invalid ▁ Input " ) ; return ; }
n = n % p ; int x = power ( n , ( p + 1 ) / 4 , p ) ; if ( ( x * x ) % p == n ) { System . out . print ( " Square ▁ root ▁ is ▁ " + x ) ; return ; }
x = p - x ; if ( ( x * x ) % p == n ) { System . out . print ( " Square ▁ root ▁ is ▁ " + x ) ; return ; }
System . out . print ( " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ) ; }
static public void main ( String [ ] args ) { int p = 7 ; int n = 2 ; squareRoot ( n , p ) ; } }
static int power ( int x , int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
static boolean miillerTest ( int d , int n ) {
int a = 2 + ( int ) ( Math . random ( ) % ( n - 4 ) ) ;
int x = power ( a , d , n ) ; if ( x == 1 x == n - 1 ) return true ;
while ( d != n - 1 ) { x = ( x * x ) % n ; d *= 2 ; if ( x == 1 ) return false ; if ( x == n - 1 ) return true ; }
return false ; }
static boolean isPrime ( int n , int k ) {
if ( n <= 1 n == 4 ) return false ; if ( n <= 3 ) return true ;
int d = n - 1 ; while ( d % 2 == 0 ) d /= 2 ;
for ( int i = 0 ; i < k ; i ++ ) if ( ! miillerTest ( d , n ) ) return false ; return true ; }
public static void main ( String args [ ] ) { int k = 4 ; System . out . println ( " All ▁ primes ▁ smaller ▁ " + " than ▁ 100 : ▁ " ) ; for ( int n = 1 ; n < 100 ; n ++ ) if ( isPrime ( n , k ) ) System . out . print ( n + " ▁ " ) ; } }
private static int maxConsecutiveOnes ( int x ) {
int count = 0 ;
while ( x != 0 ) {
x = ( x & ( x << 1 ) ) ; count ++ ; } return count ; }
public static void main ( String strings [ ] ) { System . out . println ( maxConsecutiveOnes ( 14 ) ) ; System . out . println ( maxConsecutiveOnes ( 222 ) ) ; } }
import java . io . * ; class GFG { static int subtract ( int x , int y ) {
while ( y != 0 ) {
int borrow = ( ~ x ) & y ;
x = x ^ y ;
y = borrow << 1 ; } return x ; }
public static void main ( String [ ] args ) { int x = 29 , y = 13 ; System . out . println ( " x ▁ - ▁ y ▁ is ▁ " + subtract ( x , y ) ) ; } }
class GFG { static int subtract ( int x , int y ) { if ( y == 0 ) return x ; return subtract ( x ^ y , ( ~ x & y ) << 1 ) ; }
public static void main ( String [ ] args ) { int x = 29 , y = 13 ; System . out . printf ( " x ▁ - ▁ y ▁ is ▁ % d " , subtract ( x , y ) ) ; } }
static void addEdge ( Vector < Integer > v [ ] , int x , int y ) { v [ x ] . add ( y ) ; v [ y ] . add ( x ) ; }
static void dfs ( Vector < Integer > tree [ ] , Vector < Integer > temp , int ancestor [ ] , int u , int parent , int k ) {
temp . add ( u ) ;
for ( int i : tree [ u ] ) { if ( i == parent ) continue ; dfs ( tree , temp , ancestor , i , u , k ) ; } temp . remove ( temp . size ( ) - 1 ) ;
if ( temp . size ( ) < k ) { ancestor [ u ] = - 1 ; } else {
ancestor [ u ] = temp . get ( temp . size ( ) - k ) ; } }
static void KthAncestor ( int N , int K , int E , int edges [ ] [ ] ) {
@ SuppressWarnings ( " unchecked " ) Vector < Integer > [ ] tree = new Vector [ N + 1 ] ; for ( int i = 0 ; i < tree . length ; i ++ ) tree [ i ] = new Vector < Integer > ( ) ; for ( int i = 0 ; i < E ; i ++ ) { addEdge ( tree , edges [ i ] [ 0 ] , edges [ i ] [ 1 ] ) ; }
Vector < Integer > temp = new Vector < Integer > ( ) ;
int [ ] ancestor = new int [ N + 1 ] ; dfs ( tree , temp , ancestor , 1 , 0 , K ) ;
for ( int i = 1 ; i <= N ; i ++ ) { System . out . print ( ancestor [ i ] + " ▁ " ) ; } }
int N = 9 ; int K = 2 ;
int E = 8 ; int edges [ ] [ ] = { { 1 , 2 } , { 1 , 3 } , { 2 , 4 } , { 2 , 5 } , { 2 , 6 } , { 3 , 7 } , { 3 , 8 } , { 3 , 9 } } ;
KthAncestor ( N , K , E , edges ) ; } }
static void build ( Vector < Integer > sum , Vector < Integer > a , int l , int r , int rt ) {
if ( l == r ) { sum . set ( rt , a . get ( l - 1 ) ) ; return ; }
int m = ( l + r ) >> 1 ;
build ( sum , a , l , m , rt << 1 ) ; build ( sum , a , m + 1 , r , rt << 1 1 ) ; }
static void pushDown ( Vector < Integer > sum , Vector < Integer > add , int rt , int ln , int rn ) { if ( add . get ( rt ) != 0 ) { add . set ( rt << 1 , add . get ( rt ) ) ; add . set ( rt << 1 | 1 , add . get ( rt ) ) ; sum . set ( rt << 1 , sum . get ( rt << 1 ) + add . get ( rt ) * ln ) ; sum . set ( rt << 1 | 1 , sum . get ( rt << 1 1 ) + add . get ( rt ) * rn ) ; add . set ( rt , 0 ) ; } }
static void update ( Vector < Integer > sum , Vector < Integer > add , int L , int R , int C , int l , int r , int rt ) {
if ( L <= l && r <= R ) { sum . set ( rt , sum . get ( rt ) + C * ( r - l + 1 ) ) ; add . set ( rt , add . get ( rt ) + C ) ; return ; }
int m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ;
if ( L <= m ) { update ( sum , add , L , R , C , l , m , rt << 1 ) ; } if ( R > m ) { update ( sum , add , L , R , C , m + 1 , r , rt << 1 1 ) ; } }
static int query ( Vector < Integer > sum , Vector < Integer > add , int L , int R , int l , int r , int rt ) {
if ( L <= l && r <= R ) { return sum . get ( rt ) ; }
int m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ; int ans = 0 ;
if ( L <= m ) { ans += query ( sum , add , L , R , l , m , rt << 1 ) ; } if ( R > m ) { ans += query ( sum , add , L , R , m + 1 , r , rt << 1 1 ) ; }
return ans ; }
static void sequenceMaintenance ( int n , int q , Vector < Integer > a , Vector < Integer > b , int m ) {
Collections . sort ( a ) ;
Vector < Integer > sum = new Vector < Integer > ( ) ; Vector < Integer > ad = new Vector < Integer > ( ) ; Vector < Integer > ans = new Vector < Integer > ( ) ; for ( int i = 0 ; i < ( n << 2 ) ; i ++ ) { sum . add ( 0 ) ; ad . add ( 0 ) ; }
build ( sum , a , 1 , n , 1 ) ;
for ( int i = 0 ; i < q ; i ++ ) { int l = 1 , r = n , pos = - 1 ; while ( l <= r ) { m = ( l + r ) >> 1 ; if ( query ( sum , ad , m , m , 1 , n , 1 ) >= b . get ( i ) ) { r = m - 1 ; pos = m ; } else { l = m + 1 ; } } if ( pos == - 1 ) { ans . add ( 0 ) ; } else {
ans . add ( n - pos + 1 ) ;
update ( sum , ad , pos , n , - m , 1 , n , 1 ) ; } }
for ( int i = 0 ; i < ans . size ( ) ; i ++ ) { System . out . print ( ans . get ( i ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int N = 4 ; int Q = 3 ; int M = 1 ; Vector < Integer > arr = new Vector < Integer > ( ) ; arr . add ( 1 ) ; arr . add ( 2 ) ; arr . add ( 3 ) ; arr . add ( 4 ) ; Vector < Integer > query = new Vector < Integer > ( ) ; query . add ( 4 ) ; query . add ( 3 ) ; query . add ( 1 ) ;
sequenceMaintenance ( N , Q , arr , query , M ) ; } }
static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
static boolean hasCoprimePair ( int [ ] arr , int n ) {
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
if ( ( __gcd ( arr [ i ] , arr [ j ] ) ) == 1 ) { return true ; } } }
return false ; }
public static void main ( String [ ] args ) { int n = 3 ; int [ ] arr = { 6 , 9 , 15 } ;
if ( hasCoprimePair ( arr , n ) ) { System . out . print ( 1 + "NEW_LINE"); }
else { System . out . print ( n + "NEW_LINE"); } } }
static int Numberofways ( int n ) { int count = 0 ; for ( int a = 1 ; a < n ; a ++ ) { for ( int b = 0 ; b < n ; b ++ ) { int c = n - ( a + b ) ;
if ( a + b > c && a + c > b && b + c > a ) { count ++ ; } } }
return count ; }
public static void main ( String [ ] args ) { int n = 15 ; System . out . println ( Numberofways ( n ) ) ; } }
static void countPairs ( int N , int [ ] arr ) { int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( i == arr [ arr [ i ] - 1 ] - 1 ) {
count ++ ; } }
System . out . println ( count / 2 ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 2 , 1 , 4 , 3 } ; int N = arr . length ; countPairs ( N , arr ) ; } }
static int LongestFibSubseq ( int A [ ] , int n ) {
TreeSet < Integer > S = new TreeSet < > ( ) ; for ( int t : A ) { S . add ( t ) ; } int maxLen = 0 , x , y ; for ( int i = 0 ; i < n ; ++ i ) { for ( int j = i + 1 ; j < n ; ++ j ) { x = A [ j ] ; y = A [ i ] + A [ j ] ; int length = 3 ;
while ( S . contains ( y ) && ( y != S . last ( ) ) ) {
int z = x + y ; x = y ; y = z ; maxLen = Math . max ( maxLen , ++ length ) ; } } } return maxLen >= 3 ? maxLen : 0 ; }
public static void main ( String [ ] args ) { int A [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 } ; int n = A . length ; System . out . print ( LongestFibSubseq ( A , n ) ) ; } }
static int CountMaximum ( int arr [ ] , int n , int k ) {
Arrays . sort ( arr ) ; int sum = 0 , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 30 , 30 , 10 , 10 } ; int n = 4 ; int k = 50 ;
System . out . println ( CountMaximum ( arr , n , k ) ) ; } }
public static int num_candyTypes ( int [ ] candies ) {
Dictionary < Integer , Integer > s = new Hashtable < Integer , Integer > ( ) ;
for ( int i = 0 ; i < candies . length ; i ++ ) { s . put ( candies [ i ] , 1 ) ; }
return s . size ( ) ; }
public static void distribute_candies ( int [ ] candies ) {
int allowed = candies . length / 2 ;
int types = num_candyTypes ( candies ) ;
if ( types < allowed ) System . out . println ( types ) ; else System . out . println ( allowed ) ; }
int candies [ ] = { 4 , 4 , 5 , 5 , 3 , 3 } ;
distribute_candies ( candies ) ; } }
static double [ ] Length_Diagonals ( int a , double theta ) { double p = a * Math . sqrt ( 2 + ( 2 * Math . cos ( theta * ( Math . PI / 180 ) ) ) ) ; double q = a * Math . sqrt ( 2 - ( 2 * Math . cos ( theta * ( Math . PI / 180 ) ) ) ) ; return new double [ ] { p , q } ; }
public static void main ( String [ ] args ) { int A = 6 ; double theta = 45 ; double [ ] ans = Length_Diagonals ( A , theta ) ; System . out . printf ( " % .2f " + " ▁ " + " % .2f " , ans [ 0 ] , ans [ 1 ] ) ; } }
static int __builtin_popcount ( int n ) { int count = 0 ; while ( n > 0 ) { count += n & 1 ; n >>= 1 ; } return count ; }
static void countEvenOdd ( int arr [ ] , int n , int K ) { int even = 0 , odd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } int y ;
y = __builtin_popcount ( K ) ;
if ( ( y & 1 ) != 0 ) { System . out . println ( " Even ▁ = ▁ " + odd + " , ▁ Odd ▁ = ▁ " + even ) ; }
else { System . out . println ( " Even ▁ = ▁ " + even + " , ▁ Odd ▁ = ▁ " + odd ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 2 , 15 , 9 , 8 , 8 } ; int K = 3 ; int n = arr . length ;
countEvenOdd ( arr , n , K ) ; } }
public static void main ( String args [ ] ) { int N = 6 ; int Even = N / 2 ; int Odd = N - Even ; System . out . println ( Even * Odd ) ; } }
static int countTriplets ( int [ ] A ) {
int cnt = 0 ;
HashMap < Integer , Integer > tuples = new HashMap < Integer , Integer > ( ) ;
for ( int a : A )
for ( int b : A ) { if ( tuples . containsKey ( a & b ) ) tuples . put ( a & b , tuples . get ( a & b ) + 1 ) ; else tuples . put ( a & b , 1 ) ; }
for ( int a : A )
for ( Map . Entry < Integer , Integer > t : tuples . entrySet ( ) )
if ( ( t . getKey ( ) & a ) == 0 ) cnt += t . getValue ( ) ;
return cnt ; }
int [ ] A = { 2 , 1 , 3 } ;
System . out . print ( countTriplets ( A ) ) ; } }
import java . util . Arrays ; class GfG { static int CountWays ( int n ) {
int noOfWays [ ] = new int [ n + 3 ] ; noOfWays [ 0 ] = 1 ; noOfWays [ 1 ] = 1 ; noOfWays [ 2 ] = 1 + 1 ;
for ( int i = 3 ; i < n + 1 ; i ++ ) {
noOfWays [ i ] = noOfWays [ i - 1 ] + noOfWays [ i - 3 ] ; } return noOfWays [ n ] ; }
public static void main ( String [ ] args ) { int n = 5 ; System . out . println ( CountWays ( n ) ) ; } }
public class GFG { public static void printSpiral ( int size ) {
int row = 0 , col = 0 ; int boundary = size - 1 ; int sizeLeft = size - 1 ; int flag = 1 ;
char move = ' r ' ;
int matrix [ ] [ ] = new int [ size ] [ size ] ; for ( int i = 1 ; i < size * size + 1 ; i ++ ) {
matrix [ row ] [ col ] = i ;
switch ( move ) {
case ' r ' : col += 1 ; break ;
case ' l ' : col -= 1 ; break ;
case ' u ' : row -= 1 ; break ;
case ' d ' : row += 1 ; break ; }
if ( i == boundary ) {
boundary += sizeLeft ;
if ( flag != 2 ) { flag = 2 ; } else { flag = 1 ; sizeLeft -= 1 ; }
switch ( move ) {
case ' r ' : move = ' d ' ; break ;
case ' d ' : move = ' l ' ; break ;
case ' l ' : move = ' u ' ; break ;
case ' u ' : move = ' r ' ; break ; } } }
for ( row = 0 ; row < size ; row ++ ) { for ( col = 0 ; col < size ; col ++ ) { int n = matrix [ row ] [ col ] ; System . out . print ( ( n < 10 ) ? ( n + " ▁ " ) : ( n + " ▁ " ) ) ; } System . out . println ( ) ; } }
int size = 5 ;
printSpiral ( size ) ; } }
static void findWinner ( String a , int n ) {
Vector < Integer > v = new Vector < Integer > ( ) ;
int c = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a . charAt ( i ) == '0' ) { c ++ ; }
else { if ( c != 0 ) v . add ( c ) ; c = 0 ; } } if ( c != 0 ) v . add ( c ) ;
if ( v . size ( ) == 0 ) { System . out . print ( " Player ▁ B " ) ; return ; }
if ( v . size ( ) == 1 ) { if ( ( v . get ( 0 ) & 1 ) != 0 ) System . out . print ( " Player ▁ A " ) ;
else System . out . print ( " Player ▁ B " ) ; return ; }
int first = Integer . MIN_VALUE ; int second = Integer . MIN_VALUE ;
for ( int i = 0 ; i < v . size ( ) ; i ++ ) {
if ( a . charAt ( i ) > first ) { second = first ; first = a . charAt ( i ) ; }
else if ( a . charAt ( i ) > second && a . charAt ( i ) != first ) second = a . charAt ( i ) ; }
if ( ( first & 1 ) != 0 && ( first + 1 ) / 2 > second ) System . out . print ( " Player ▁ A " ) ; else System . out . print ( " Player ▁ B " ) ; }
public static void main ( String [ ] args ) { String S = "1100011" ; int N = S . length ( ) ; findWinner ( S , N ) ; } }
static boolean can_Construct ( String S , int K ) {
Map < Character , Integer > m = new HashMap < > ( ) ; int p = 0 ;
if ( S . length ( ) == K ) return true ;
for ( int i = 0 ; i < S . length ( ) ; i ++ ) m . put ( S . charAt ( i ) , m . getOrDefault ( S . charAt ( i ) , 0 ) + 1 ) ;
if ( K > S . length ( ) ) return false ; else {
for ( Integer h : m . values ( ) ) { if ( h % 2 != 0 ) p = p + 1 ; } }
if ( K < p ) return false ; return true ; }
public static void main ( String [ ] args ) { String S = " annabelle " ; int K = 4 ; if ( can_Construct ( S , K ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean equalIgnoreCase ( String str1 , String str2 ) { int i = 0 ;
str1 = str1 . toLowerCase ( ) ; str2 = str2 . toLowerCase ( ) ;
int x = str1 . compareTo ( str2 ) ;
return x == 0 ; }
static void equalIgnoreCaseUtil ( String str1 , String str2 ) { boolean res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) System . out . println ( " Same " ) ; else System . out . println ( " Not ▁ Same " ) ; }
public static void main ( String [ ] args ) { String str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; } }
static void steps ( String str , int n ) {
boolean flag = false ; int x = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
if ( x == 0 ) flag = true ;
if ( x == n - 1 ) flag = false ;
for ( int j = 0 ; j < x ; j ++ ) System . out . print ( " * " ) ; System . out . print ( str . charAt ( i ) + "NEW_LINE");
if ( flag == true ) x ++ ; else x -- ; } }
int n = 4 ; String str = " GeeksForGeeks " ; System . out . println ( " String : ▁ " + str ) ; System . out . println ( " Max ▁ Length ▁ of ▁ Steps : ▁ " + n ) ;
steps ( str , n ) ; } }
import java . util . * ; class GFG { static void countFreq ( int arr [ ] , int n ) {
boolean [ ] visited = new boolean [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( visited [ i ] == true ) continue ;
int count = 1 ; for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) { visited [ j ] = true ; count ++ ; } } System . out . println ( arr [ i ] + " ▁ " + count ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 10 , 20 , 20 , 10 , 10 , 20 , 5 , 20 } ; int n = arr . length ; countFreq ( arr , n ) ; } }
static boolean isDivisible ( String str , int k ) { int n = str . length ( ) ; int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) if ( str . charAt ( n - i - 1 ) == '0' ) c ++ ;
return ( c == k ) ; }
String str1 = "10101100" ; int k = 2 ; if ( isDivisible ( str1 , k ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ;
String str2 = "111010100" ; k = 2 ; if ( isDivisible ( str2 , k ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
public class GFG { static final int NO_OF_CHARS = 256 ;
static boolean canFormPalindrome ( String str ) {
int [ ] count = new int [ NO_OF_CHARS ] ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) count [ str . charAt ( i ) ] ++ ;
int odd = 0 ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) { if ( ( count [ i ] & 1 ) != 0 ) odd ++ ; if ( odd > 1 ) return false ; }
return true ; }
public static void main ( String args [ ] ) { System . out . println ( canFormPalindrome ( " geeksforgeeks " ) ? " Yes " : " No " ) ; System . out . println ( canFormPalindrome ( " geeksogeeks " ) ? " Yes " : " No " ) ; } }
static boolean isNumber ( String s ) { for ( int i = 0 ; i < s . length ( ) ; i ++ ) if ( Character . isDigit ( s . charAt ( i ) ) == false ) return false ; return true ; }
String str = "6790" ;
if ( isNumber ( str ) ) System . out . println ( " Integer " ) ;
else System . out . println ( " String " ) ; } }
void reverse ( String str ) { if ( ( str == null ) || ( str . length ( ) <= 1 ) ) System . out . println ( str ) ; else { System . out . print ( str . charAt ( str . length ( ) - 1 ) ) ; reverse ( str . substring ( 0 , str . length ( ) - 1 ) ) ; } }
public static void main ( String [ ] args ) { String str = " Geeks ▁ for ▁ Geeks " ; StringReverse obj = new StringReverse ( ) ; obj . reverse ( str ) ; } }
static int box1 = 0 ;
static int box2 = 0 ; static int [ ] fact = new int [ 11 ] ;
public static double getProbability ( int [ ] balls ) {
factorial ( 10 ) ;
box2 = balls . length ;
int K = 0 ;
for ( int i = 0 ; i < balls . length ; i ++ ) K += balls [ i ] ;
if ( K % 2 == 1 ) return 0 ;
long all = comb ( K , K / 2 ) ;
long validPermutations = validPermutations ( K / 2 , balls , 0 , 0 ) ;
return ( double ) validPermutations / all ; }
static long validPermutations ( int n , int [ ] balls , int usedBalls , int i ) {
if ( usedBalls == n ) {
return box1 == box2 ? 1 : 0 ; }
if ( i >= balls . length ) return 0 ;
long res = validPermutations ( n , balls , usedBalls , i + 1 ) ;
box1 ++ ;
for ( int j = 1 ; j <= balls [ i ] ; j ++ ) {
if ( j == balls [ i ] ) box2 -- ;
long combinations = comb ( balls [ i ] , j ) ;
res += combinations * validPermutations ( n , balls , usedBalls + j , i + 1 ) ; }
box1 -- ;
box2 ++ ; return res ; }
static void factorial ( int N ) {
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ; }
static long comb ( int n , int r ) { long res = fact [ n ] / fact [ r ] ; res /= fact [ n - r ] ; return res ; }
public static void main ( String [ ] args ) { int [ ] arr = { 2 , 1 , 1 } ; int N = 4 ;
System . out . println ( getProbability ( arr ) ) ; } }
static double polyarea ( double n , double r ) {
if ( r < 0 && n < 0 ) return - 1 ;
double A = ( ( r * r * n ) * Math . sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
public static void main ( String [ ] args ) { float r = 9 , n = 6 ; System . out . println ( polyarea ( n , r ) ) ; } }
class GFG { static void is_partition_possible ( int n , int x [ ] , int y [ ] , int w [ ] ) { Map < Integer , Integer > weight_at_x = new HashMap < Integer , Integer > ( ) ; int max_x = ( int ) - 2e3 , min_x = ( int ) 2e3 ;
for ( int i = 0 ; i < n ; i ++ ) { int new_x = x [ i ] - y [ i ] ; max_x = Math . max ( max_x , new_x ) ; min_x = Math . min ( min_x , new_x ) ;
if ( weight_at_x . containsKey ( new_x ) ) { weight_at_x . put ( new_x , weight_at_x . get ( new_x ) + w [ i ] ) ; } else { weight_at_x . put ( new_x , w [ i ] ) ; } } Vector < Integer > sum_till = new Vector < > ( ) ; sum_till . add ( 0 ) ;
for ( int s = min_x ; s <= max_x ; s ++ ) { if ( weight_at_x . get ( s ) == null ) sum_till . add ( sum_till . lastElement ( ) ) ; else sum_till . add ( sum_till . lastElement ( ) + weight_at_x . get ( s ) ) ; } int total_sum = sum_till . lastElement ( ) ; int partition_possible = 0 ; for ( int i = 1 ; i < sum_till . size ( ) ; i ++ ) { if ( sum_till . get ( i ) == total_sum - sum_till . get ( i ) ) partition_possible = 1 ;
if ( sum_till . get ( i - 1 ) == total_sum - sum_till . get ( i ) ) partition_possible = 1 ; } System . out . printf ( partition_possible == 1 ? "YES " ▁ : ▁ " NO "); }
public static void main ( String [ ] args ) { int n = 3 ; int x [ ] = { - 1 , - 2 , 1 } ; int y [ ] = { 1 , 1 , - 1 } ; int w [ ] = { 3 , 1 , 4 } ; is_partition_possible ( n , x , y , w ) ; } }
static double findPCSlope ( double m ) { return - 1.0 / m ; }
public static void main ( String [ ] args ) { double m = 2.0 ; System . out . println ( findPCSlope ( m ) ) ; } }
class GFG { static float pi = 3.14159f ; static float
area_of_segment ( float radius , float angle ) {
float area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
float area_of_triangle = ( float ) 1 / 2 * ( radius * radius ) * ( float ) Math . sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
public static void main ( String [ ] args ) { float radius = 10.0f , angle = 90.0f ; System . out . println ( " Area ▁ of ▁ minor ▁ segment ▁ = ▁ " + area_of_segment ( radius , angle ) ) ; System . out . println ( " Area ▁ of ▁ major ▁ segment ▁ = ▁ " + area_of_segment ( radius , ( 360 - angle ) ) ) ; } }
class GFG { static void SectorArea ( double radius , double angle ) { if ( angle >= 360 ) System . out . println ( " Angle ▁ not ▁ possible " ) ;
else { double sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; System . out . println ( sector ) ; } }
public static void main ( String [ ] args ) { double radius = 9 ; double angle = 60 ; SectorArea ( radius , angle ) ; } }
import java . util . * ; public class Main { static int gcd ( int a , int b ) {
if ( b == 0 ) { return a ; } return gcd ( b , a % b ) ; }
static HashMap < Integer , Integer > PrimeFactor ( int N ) { HashMap < Integer , Integer > primef = new HashMap < Integer , Integer > ( ) ;
while ( N % 2 == 0 ) { if ( primef . containsKey ( 2 ) ) { primef . put ( 2 , primef . get ( 2 ) + 1 ) ; } else { primef . put ( 2 , 1 ) ; }
N = N / 2 ; }
for ( int i = 3 ; i <= Math . sqrt ( N ) ; i ++ ) {
while ( N % i == 0 ) { if ( primef . containsKey ( i ) ) { primef . put ( i , primef . get ( i ) + 1 ) ; } else { primef . put ( i , 1 ) ; }
N = N / 2 ; } } if ( N > 2 ) { primef . put ( N , 1 ) ; } return primef ; }
static int CountToMakeEqual ( int X , int Y ) {
int gcdofXY = gcd ( X , Y ) ;
int newX = Y / gcdofXY ; int newY = X / gcdofXY ;
HashMap < Integer , Integer > primeX = PrimeFactor ( newX ) ; HashMap < Integer , Integer > primeY = PrimeFactor ( newY ) ;
int ans = 0 ;
for ( Map . Entry keys : primeX . entrySet ( ) ) { if ( X % ( int ) keys . getKey ( ) != 0 ) { return - 1 ; } ans += primeX . get ( keys . getKey ( ) ) ; }
for ( Map . Entry keys : primeY . entrySet ( ) ) { if ( Y % ( int ) keys . getKey ( ) != 0 ) { return - 1 ; } ans += primeY . get ( keys . getKey ( ) ) ; }
return ans ; }
int X = 36 ; int Y = 48 ;
int ans = CountToMakeEqual ( X , Y ) ; System . out . println ( ans ) ; } }
static class Node { int L , R , V ; Node ( int L , int R , int V ) { this . L = L ; this . R = R ; this . V = V ; } }
static boolean check ( ArrayList < Integer > Adj [ ] , int Src , int N , boolean visited [ ] ) { int color [ ] = new int [ N ] ;
visited [ Src ] = true ; ArrayDeque < Integer > q = new ArrayDeque < > ( ) ;
q . addLast ( Src ) ; while ( ! q . isEmpty ( ) ) {
int u = q . removeFirst ( ) ;
int Col = color [ u ] ;
for ( int x : Adj [ u ] ) {
if ( visited [ x ] == true && color [ x ] == Col ) { return false ; } else if ( visited [ x ] == false ) {
visited [ x ] = true ;
q . addLast ( x ) ;
color [ x ] = 1 - Col ; } } }
return true ; }
static void addEdge ( ArrayList < Integer > Adj [ ] , int u , int v ) { Adj [ u ] . add ( v ) ; Adj [ v ] . add ( u ) ; }
static void isPossible ( Node Arr [ ] , int N ) {
@ SuppressWarnings ( " unchecked " ) ArrayList < Integer > [ ] Adj = ( ArrayList < Integer > [ ] ) new ArrayList [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) Adj [ i ] = new ArrayList < > ( ) ;
for ( int i = 0 ; i < N - 1 ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) {
if ( Arr [ i ] . R < Arr [ j ] . L Arr [ i ] . L > Arr [ j ] . R ) { continue ; }
else { if ( Arr [ i ] . V == Arr [ j ] . V ) {
addEdge ( Adj , i , j ) ; } } } }
boolean visited [ ] = new boolean [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) { if ( visited [ i ] == false && Adj [ i ] . size ( ) > 0 ) {
if ( check ( Adj , i , N , visited ) == false ) { System . out . println ( " No " ) ; return ; } } }
System . out . println ( " Yes " ) ; }
public static void main ( String [ ] args ) { Node arr [ ] = { new Node ( 5 , 7 , 2 ) , new Node ( 4 , 6 , 1 ) , new Node ( 1 , 5 , 2 ) , new Node ( 6 , 5 , 1 ) } ; int N = arr . length ; isPossible ( arr , N ) ; } }
import java . io . * ; import java . util . * ; class GFG { public static void lexNumbers ( int n ) { List < Integer > sol = new ArrayList < > ( ) ; dfs ( 1 , n , sol ) ; System . out . println ( sol ) ; } public static void dfs ( int temp , int n , List < Integer > sol ) { if ( temp > n ) return ; sol . add ( temp ) ; dfs ( temp * 10 , n , sol ) ; if ( temp % 10 != 9 ) dfs ( temp + 1 , n , sol ) ; }
public static void main ( String [ ] args ) { int n = 15 ; lexNumbers ( n ) ; } }
static int minimumSwaps ( int [ ] arr ) {
int count = 0 ; int i = 0 ; while ( i < arr . length ) {
if ( arr [ i ] != i + 1 ) { while ( arr [ i ] != i + 1 ) { int temp = 0 ;
temp = arr [ arr [ i ] - 1 ] ; arr [ arr [ i ] - 1 ] = arr [ i ] ; arr [ i ] = temp ; count ++ ; } }
i ++ ; } return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 3 , 4 , 1 , 5 } ;
System . out . println ( minimumSwaps ( arr ) ) ; } }
class GFG {
static class Node { int data ; Node next ; Node prev ; } ;
static Node append ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ; Node last = head_ref ;
new_node . data = new_data ;
new_node . next = null ;
if ( head_ref == null ) { new_node . prev = null ; head_ref = new_node ; return head_ref ; }
while ( last . next != null ) last = last . next ;
last . next = new_node ;
new_node . prev = last ; return head_ref ; }
static void printList ( Node node ) { Node last ;
while ( node != null ) { System . out . print ( node . data + " ▁ " ) ; last = node ; node = node . next ; } }
static Node mergeList ( Node p , Node q ) { Node s = null ;
if ( p == null q == null ) { return ( p == null ? q : p ) ; }
if ( p . data < q . data ) { p . prev = s ; s = p ; p = p . next ; } else { q . prev = s ; s = q ; q = q . next ; }
Node head = s ; while ( p != null && q != null ) { if ( p . data < q . data ) {
s . next = p ; p . prev = s ; s = s . next ; p = p . next ; } else {
s . next = q ; q . prev = s ; s = s . next ; q = q . next ; } }
if ( p == null ) { s . next = q ; q . prev = s ; } if ( q == null ) { s . next = p ; p . prev = s ; }
return head ; }
static Node mergeAllList ( Node head [ ] , int k ) { Node finalList = null ; for ( int i = 0 ; i < k ; i ++ ) {
finalList = mergeList ( finalList , head [ i ] ) ; }
return finalList ; }
public static void main ( String args [ ] ) { int k = 3 ; Node head [ ] = new Node [ k ] ;
for ( int i = 0 ; i < k ; i ++ ) { head [ i ] = null ; }
head [ 0 ] = append ( head [ 0 ] , 1 ) ; head [ 0 ] = append ( head [ 0 ] , 5 ) ; head [ 0 ] = append ( head [ 0 ] , 9 ) ;
head [ 1 ] = append ( head [ 1 ] , 2 ) ; head [ 1 ] = append ( head [ 1 ] , 3 ) ; head [ 1 ] = append ( head [ 1 ] , 7 ) ; head [ 1 ] = append ( head [ 1 ] , 12 ) ;
head [ 2 ] = append ( head [ 2 ] , 8 ) ; head [ 2 ] = append ( head [ 2 ] , 11 ) ; head [ 2 ] = append ( head [ 2 ] , 13 ) ; head [ 2 ] = append ( head [ 2 ] , 18 ) ;
Node finalList = mergeAllList ( head , k ) ;
printList ( finalList ) ; } }
static void insertionSortRecursive ( int arr [ ] , int n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
int last = arr [ n - 1 ] ; int j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
public static void main ( String [ ] args ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; insertionSortRecursive ( arr , arr . length ) ; System . out . println ( Arrays . toString ( arr ) ) ; } }
static void bubbleSort ( int arr [ ] , int n ) {
if ( n == 1 ) return ;
for ( int i = 0 ; i < n - 1 ; i ++ ) if ( arr [ i ] > arr [ i + 1 ] ) {
int temp = arr [ i ] ; arr [ i ] = arr [ i + 1 ] ; arr [ i + 1 ] = temp ; }
bubbleSort ( arr , n - 1 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; bubbleSort ( arr , arr . length ) ; System . out . println ( " Sorted ▁ array ▁ : ▁ " ) ; System . out . println ( Arrays . toString ( arr ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int maxSumAfterPartition ( int arr [ ] , int n ) {
ArrayList < Integer > pos = new ArrayList < Integer > ( ) ;
ArrayList < Integer > neg = new ArrayList < Integer > ( ) ;
int zero = 0 ;
int pos_sum = 0 ;
int neg_sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > 0 ) { pos . add ( arr [ i ] ) ; pos_sum += arr [ i ] ; } else if ( arr [ i ] < 0 ) { neg . add ( arr [ i ] ) ; neg_sum += arr [ i ] ; } else { zero ++ ; } }
int ans = 0 ;
Collections . sort ( pos ) ;
Collections . sort ( neg ) ;
if ( pos . size ( ) > 0 && neg . size ( ) > 0 ) { ans = ( pos_sum - neg_sum ) ; } else if ( pos . size ( ) > 0 ) { if ( zero > 0 ) {
ans = ( pos_sum ) ; } else {
ans = ( pos_sum - 2 * pos . get ( 0 ) ) ; } } else { if ( zero > 0 ) {
ans = ( - 1 * neg_sum ) ; } else {
ans = ( neg . get ( 0 ) - ( neg_sum - neg . get ( 0 ) ) ) ; } } return ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , - 5 , - 7 } ; int n = 5 ; System . out . println ( maxSumAfterPartition ( arr , n ) ) ; } }
static int MaxXOR ( int arr [ ] , int N ) {
int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { res |= arr [ i ] ; }
return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 5 , 7 } ; int N = arr . length ; System . out . println ( MaxXOR ( arr , N ) ) ; } }
static int countEqual ( int A [ ] , int B [ ] , int N ) {
int first = 0 ; int second = N - 1 ;
int count = 0 ; while ( first < N && second >= 0 ) {
if ( A [ first ] < B [ second ] ) {
first ++ ; }
else if ( B [ second ] < A [ first ] ) {
second -- ; }
else {
count ++ ;
first ++ ;
second -- ; } }
return count ; }
public static void main ( String [ ] args ) { int A [ ] = { 2 , 4 , 5 , 8 , 12 , 13 , 17 , 18 , 20 , 22 , 309 , 999 } ; int B [ ] = { 109 , 99 , 68 , 54 , 22 , 19 , 17 , 13 , 11 , 5 , 3 , 1 } ; int N = A . length ; System . out . println ( countEqual ( A , B , N ) ) ; } }
class GFG { static int [ ] arr = new int [ 100005 ] ;
static boolean isPalindrome ( int N ) {
int temp = N ;
int res = 0 ;
while ( temp != 0 ) { int rem = temp % 10 ; res = res * 10 + rem ; temp /= 10 ; }
if ( res == N ) { return true ; } else { return false ; } }
static int sumOfDigits ( int N ) {
int sum = 0 ; while ( N != 0 ) {
sum += N % 10 ;
N /= 10 ; }
return sum ; }
static boolean isPrime ( int n ) {
if ( n <= 1 ) { return false ; }
for ( int i = 2 ; i <= n / 2 ; ++ i ) {
if ( n % i == 0 ) return false ; } return true ; }
static void precompute ( ) {
for ( int i = 1 ; i <= 100000 ; i ++ ) {
if ( isPalindrome ( i ) ) {
int sum = sumOfDigits ( i ) ;
if ( isPrime ( sum ) ) arr [ i ] = 1 ; else arr [ i ] = 0 ; } else arr [ i ] = 0 ; }
for ( int i = 1 ; i <= 100000 ; i ++ ) { arr [ i ] = arr [ i ] + arr [ i - 1 ] ; } }
static void countNumbers ( int [ ] [ ] Q , int N ) {
precompute ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
System . out . println ( ( arr [ Q [ i ] [ 1 ] ] - arr [ Q [ i ] [ 0 ] - 1 ] ) ) ; } }
public static void main ( String [ ] args ) { int [ ] [ ] Q = { { 5 , 9 } , { 1 , 101 } } ; int N = Q . length ;
countNumbers ( Q , N ) ; } }
static int sum ( int n ) { int res = 0 ; while ( n > 0 ) { res += n % 10 ; n /= 10 ; } return res ; }
static int smallestNumber ( int n , int s ) {
if ( sum ( n ) <= s ) { return n ; }
int ans = n , k = 1 ; for ( int i = 0 ; i < 9 ; ++ i ) {
int digit = ( ans / k ) % 10 ;
int add = k * ( ( 10 - digit ) % 10 ) ; ans += add ;
if ( sum ( ans ) <= s ) { break ; }
k *= 10 ; } return ans ; }
int N = 3 , S = 2 ;
System . out . println ( smallestNumber ( N , S ) ) ; } }
static int maxSubsequences ( int arr [ ] , int n ) {
HashMap < Integer , Integer > map = new HashMap < > ( ) ;
int maxCount = 0 ;
int count ; for ( int i = 0 ; i < n ; i ++ ) {
if ( map . containsKey ( arr [ i ] ) ) {
count = map . get ( arr [ i ] ) ;
if ( count > 1 ) {
map . put ( arr [ i ] , count - 1 ) ; }
else map . remove ( arr [ i ] ) ;
if ( arr [ i ] - 1 > 0 ) map . put ( arr [ i ] - 1 , map . getOrDefault ( arr [ i ] - 1 , 0 ) + 1 ) ; } else {
maxCount ++ ;
if ( arr [ i ] - 1 > 0 ) map . put ( arr [ i ] - 1 , map . getOrDefault ( arr [ i ] - 1 , 0 ) + 1 ) ; } }
return maxCount ; }
public static void main ( String [ ] args ) { int n = 5 ; int arr [ ] = { 4 , 5 , 2 , 1 , 4 } ; System . out . println ( maxSubsequences ( arr , n ) ) ; } }
static String removeOcc ( String s , char ch ) {
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( s . charAt ( i ) == ch ) { s = s . substring ( 0 , i ) + s . substring ( i + 1 ) ; break ; } }
for ( int i = s . length ( ) - 1 ; i > - 1 ; i -- ) {
if ( s . charAt ( i ) == ch ) { s = s . substring ( 0 , i ) + s . substring ( i + 1 ) ; break ; } } return s ; }
public static void main ( String [ ] args ) { String s = " hello ▁ world " ; char ch = ' l ' ; System . out . print ( removeOcc ( s , ch ) ) ; } }
public static void minSteps ( int N , int [ ] increasing , int [ ] decreasing ) {
int min = Integer . MAX_VALUE ;
for ( int i : increasing ) { if ( min > i ) min = i ; }
int max = Integer . MIN_VALUE ;
for ( int i : decreasing ) { if ( max < i ) max = i ; }
int minSteps = Math . max ( max , N - min ) ;
System . out . println ( minSteps ) ; }
int N = 7 ;
int increasing [ ] = { 3 , 5 } ; int decreasing [ ] = { 6 } ;
minSteps ( N , increasing , decreasing ) ; } }
static void solve ( int P [ ] , int n ) {
int arr [ ] = new int [ n + 1 ] ; arr [ 0 ] = 0 ; for ( int i = 0 ; i < n ; i ++ ) arr [ i + 1 ] = P [ i ] ;
int cnt = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] == i ) { int t = arr [ i + 1 ] ; arr [ i + 1 ] = arr [ i ] ; arr [ i ] = t ; cnt ++ ; } }
if ( arr [ n ] == n ) {
int t = arr [ n - 1 ] ; arr [ n - 1 ] = arr [ n ] ; arr [ n ] = t ; cnt ++ ; }
System . out . println ( cnt ) ; }
int N = 9 ;
int P [ ] = new int [ ] { 1 , 2 , 4 , 9 , 5 , 8 , 7 , 3 , 6 } ;
solve ( P , N ) ; } }
static boolean isWaveArray ( int arr [ ] , int n ) { boolean result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
int arr [ ] = { 1 , 3 , 2 , 4 } ; int n = arr . length ; if ( isWaveArray ( arr , n ) ) { System . out . println ( " YES " ) ; } else { System . out . println ( " NO " ) ; } } }
static void countPossiblities ( int arr [ ] , int n ) {
int [ ] lastOccur = new int [ 100000 ] ; for ( int i = 0 ; i < n ; i ++ ) { lastOccur [ i ] = - 1 ; }
int [ ] dp = new int [ n + 1 ] ;
dp [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) { int curEle = arr [ i - 1 ] ;
dp [ i ] = dp [ i - 1 ] ;
if ( lastOccur [ curEle ] != - 1 & lastOccur [ curEle ] < i - 1 ) { dp [ i ] += dp [ lastOccur [ curEle ] ] ; }
lastOccur [ curEle ] = i ; }
System . out . println ( dp [ n ] ) ; } public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 1 , 2 , 2 } ; int N = arr . length ; countPossiblities ( arr , N ) ; } }
static void maxSum ( int [ ] [ ] arr , int n , int m ) {
int [ ] [ ] dp = new int [ n ] [ m + 1 ] ;
for ( int i = 0 ; i < 2 ; i ++ ) { for ( int j = 0 ; j <= m ; j ++ ) { dp [ i ] [ j ] = 0 ; } }
dp [ 0 ] [ m - 1 ] = arr [ 0 ] [ m - 1 ] ; dp [ 1 ] [ m - 1 ] = arr [ 1 ] [ m - 1 ] ;
for ( int j = m - 2 ; j >= 0 ; j -- ) {
for ( int i = 0 ; i < 2 ; i ++ ) { if ( i == 1 ) { dp [ i ] [ j ] = Math . max ( arr [ i ] [ j ] + dp [ 0 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 0 ] [ j + 2 ] ) ; } else { dp [ i ] [ j ] = Math . max ( arr [ i ] [ j ] + dp [ 1 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 1 ] [ j + 2 ] ) ; } } }
System . out . println ( Math . max ( dp [ 0 ] [ 0 ] , dp [ 1 ] [ 0 ] ) ) ; }
int [ ] [ ] arr = { { 1 , 50 , 21 , 5 } , { 2 , 10 , 10 , 5 } } ;
int N = arr [ 0 ] . length ;
maxSum ( arr , 2 , N ) ; } }
static void maxSum ( int [ ] [ ] arr , int n ) {
int r1 = 0 , r2 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int temp = r1 ; r1 = Math . max ( r1 , r2 + arr [ 0 ] [ i ] ) ; r2 = Math . max ( r2 , temp + arr [ 1 ] [ i ] ) ; }
System . out . println ( Math . max ( r1 , r2 ) ) ; }
public static void main ( String args [ ] ) { int [ ] [ ] arr = { { 1 , 50 , 21 , 5 } , { 2 , 10 , 10 , 5 } } ;
int n = arr [ 0 ] . length ; maxSum ( arr , n ) ; } }
class GFG { static int mod = ( int ) ( 1e9 + 7 ) ; static int mx = ( int ) 1e6 ; static int [ ] fact = new int [ ( int ) mx + 1 ] ;
static void Calculate_factorial ( ) { fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= mx ; i ++ ) { fact [ i ] = i * fact [ i - 1 ] ; fact [ i ] %= mod ; } }
static int UniModal_per ( int a , int b ) { int res = 1 ;
while ( b > 0 ) {
if ( b % 2 != 0 ) res = res * a ; res %= mod ; a = a * a ; a %= mod ;
b /= 2 ; }
return res ; }
static void countPermutations ( int n ) {
Calculate_factorial ( ) ;
int uni_modal = UniModal_per ( 2 , n - 1 ) ;
int nonuni_modal = fact [ n ] - uni_modal ; System . out . print ( uni_modal + " ▁ " + nonuni_modal ) ; return ; }
public static void main ( String [ ] args ) {
countPermutations ( N ) ; } }
import java . io . * ; class GFG { static void longestSubseq ( String s , int length ) {
int [ ] ones = new int [ length + 1 ] ; int [ ] zeroes = new int [ length + 1 ] ;
for ( int i = 0 ; i < length ; i ++ ) {
if ( s . charAt ( i ) == '1' ) { ones [ i + 1 ] = ones [ i ] + 1 ; zeroes [ i + 1 ] = zeroes [ i ] ; }
else { zeroes [ i + 1 ] = zeroes [ i ] + 1 ; ones [ i + 1 ] = ones [ i ] ; } } int answer = Integer . MIN_VALUE ; int x = 0 ; for ( int i = 0 ; i <= length ; i ++ ) { for ( int j = i ; j <= length ; j ++ ) {
x += ones [ i ] ;
x += ( zeroes [ j ] - zeroes [ i ] ) ;
x += ( ones [ length ] - ones [ j ] ) ;
answer = Math . max ( answer , x ) ; x = 0 ; } }
System . out . println ( answer ) ; }
public static void main ( String [ ] args ) { String s = "10010010111100101" ; int length = s . length ( ) ; longestSubseq ( s , length ) ; } }
class GFG { static int MAX = 100 ;
static void largestSquare ( int matrix [ ] [ ] , int R , int C , int q_i [ ] , int q_j [ ] , int K , int Q ) {
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ; int min_dist = Math . min ( Math . min ( i , j ) , Math . min ( R - i - 1 , C - j - 1 ) ) ; int ans = - 1 ; for ( int k = 0 ; k <= min_dist ; k ++ ) { int count = 0 ;
for ( int row = i - k ; row <= i + k ; row ++ ) for ( int col = j - k ; col <= j + k ; col ++ ) count += matrix [ row ] [ col ] ;
if ( count > K ) break ; ans = 2 * k + 1 ; } System . out . print ( ans + "NEW_LINE"); } }
public static void main ( String [ ] args ) { int matrix [ ] [ ] = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int q_i [ ] = { 1 } ; int q_j [ ] = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; } }
import java . util . * ; class GFG { static int MAX = 100 ;
static void largestSquare ( int matrix [ ] [ ] , int R , int C , int q_i [ ] , int q_j [ ] , int K , int Q ) { int [ ] [ ] countDP = new int [ R ] [ C ] ;
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] ; for ( int i = 1 ; i < R ; i ++ ) countDP [ i ] [ 0 ] = countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ; for ( int j = 1 ; j < C ; j ++ ) countDP [ 0 ] [ j ] = countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ; for ( int i = 1 ; i < R ; i ++ ) for ( int j = 1 ; j < C ; j ++ ) countDP [ i ] [ j ] = matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ;
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ;
int min_dist = Math . min ( Math . min ( i , j ) , Math . min ( R - i - 1 , C - j - 1 ) ) ; int ans = - 1 ; for ( int k = 0 ; k <= min_dist ; k ++ ) { int x1 = i - k , x2 = i + k ; int y1 = j - k , y2 = j + k ;
int count = countDP [ x2 ] [ y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 ] [ y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 ] [ y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 ] [ y1 - 1 ] ; if ( count > K ) break ; ans = 2 * k + 1 ; } System . out . print ( ans + "NEW_LINE"); } }
public static void main ( String [ ] args ) { int matrix [ ] [ ] = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int q_i [ ] = { 1 } ; int q_j [ ] = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; } }
static int MinCost ( int arr [ ] , int n ) {
int [ ] [ ] dp = new int [ n + 5 ] [ n + 5 ] ; int [ ] [ ] sum = new int [ n + 5 ] [ n + 5 ] ;
for ( int i = 0 ; i < n ; i ++ ) { int k = arr [ i ] ; for ( int j = i ; j < n ; j ++ ) { if ( i == j ) sum [ i ] [ j ] = k ; else { k += arr [ j ] ; sum [ i ] [ j ] = k ; } } }
for ( int i = n - 1 ; i >= 0 ; i -- ) {
for ( int j = i ; j < n ; j ++ ) { dp [ i ] [ j ] = Integer . MAX_VALUE ;
if ( i == j ) dp [ i ] [ j ] = 0 ; else { for ( int k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , dp [ i ] [ k ] + dp [ k + 1 ] [ j ] + sum [ i ] [ j ] ) ; } } } } return dp [ 0 ] [ n - 1 ] ; }
public static void main ( String [ ] args ) { int arr [ ] = { 7 , 6 , 8 , 6 , 1 , 1 } ; int n = arr . length ; System . out . println ( MinCost ( arr , n ) ) ; } }
static int f ( int i , int state , int A [ ] , int dp [ ] [ ] , int N ) { if ( i >= N ) return 0 ;
else if ( dp [ i ] [ state ] != - 1 ) { return dp [ i ] [ state ] ; }
else { if ( i == N - 1 ) dp [ i ] [ state ] = 1 ; else if ( state == 1 && A [ i ] > A [ i + 1 ] ) dp [ i ] [ state ] = 1 ; else if ( state == 2 && A [ i ] < A [ i + 1 ] ) dp [ i ] [ state ] = 1 ; else if ( state == 1 && A [ i ] <= A [ i + 1 ] ) dp [ i ] [ state ] = 1 + f ( i + 1 , 2 , A , dp , N ) ; else if ( state == 2 && A [ i ] >= A [ i + 1 ] ) dp [ i ] [ state ] = 1 + f ( i + 1 , 1 , A , dp , N ) ; return dp [ i ] [ state ] ; } }
static int maxLenSeq ( int A [ ] , int N ) { int i , j , tmp , y , ans ;
int dp [ ] [ ] = new int [ 1000 ] [ 3 ] ;
for ( i = 0 ; i < 1000 ; i ++ ) for ( j = 0 ; j < 3 ; j ++ ) dp [ i ] [ j ] = - 1 ;
for ( i = 0 ; i < N ; i ++ ) { tmp = f ( i , 1 , A , dp , N ) ; tmp = f ( i , 2 , A , dp , N ) ; }
ans = - 1 ; for ( i = 0 ; i < N ; i ++ ) {
y = dp [ i ] [ 1 ] ; if ( i + y >= N ) ans = Math . max ( ans , dp [ i ] [ 1 ] + 1 ) ;
else if ( y % 2 == 0 ) { ans = Math . max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 2 ] ) ; }
else if ( y % 2 == 1 ) { ans = Math . max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 1 ] ) ; } } return ans ; }
public static void main ( String [ ] args ) { int A [ ] = { 1 , 10 , 3 , 20 , 25 , 24 } ; int n = A . length ; System . out . println ( maxLenSeq ( A , n ) ) ; } }
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
static int MaxGCD ( int a [ ] , int n ) {
int Prefix [ ] = new int [ n + 2 ] ; int Suffix [ ] = new int [ n + 2 ] ;
Prefix [ 1 ] = a [ 0 ] ; for ( int i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = gcd ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( int i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = gcd ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
int ans = Math . max ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( int i = 2 ; i < n ; i += 1 ) { ans = Math . max ( ans , gcd ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; }
public static void main ( String [ ] args ) { int a [ ] = { 14 , 17 , 28 , 70 } ; int n = a . length ; System . out . println ( MaxGCD ( a , n ) ) ; } }
import java . util . Arrays ; class GFG { static int right = 2 ; static int left = 4 ; static int [ ] [ ] dp = new int [ left ] [ right ] ;
static int findSubarraySum ( int ind , int flips , int n , int [ ] a , int k ) {
if ( flips > k ) return ( int ) ( - 1e9 ) ;
if ( ind == n ) return 0 ;
if ( dp [ ind ] [ flips ] != - 1 ) return dp [ ind ] [ flips ] ;
int ans = 0 ;
ans = Math . max ( 0 , a [ ind ] + findSubarraySum ( ind + 1 , flips , n , a , k ) ) ; ans = Math . max ( ans , - a [ ind ] + findSubarraySum ( ind + 1 , flips + 1 , n , a , k ) ) ;
return dp [ ind ] [ flips ] = ans ; }
static int findMaxSubarraySum ( int [ ] a , int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < k + 1 ; j ++ ) dp [ i ] [ j ] = - 1 ; int ans = ( int ) ( - 1e9 ) ;
for ( int i = 0 ; i < n ; i ++ ) ans = Math . max ( ans , findSubarraySum ( i , 0 , n , a , k ) ) ;
if ( ans == 0 && k == 0 ) return Arrays . stream ( a ) . max ( ) . getAsInt ( ) ; return ans ; }
public static void main ( String [ ] args ) { int [ ] a = { - 1 , - 2 , - 100 , - 10 } ; int n = a . length ; int k = 1 ; System . out . println ( findMaxSubarraySum ( a , n , k ) ) ; } }
import java . io . * ; class GFG { static int mod = 1000000007 ;
static int sumOddFibonacci ( int n ) { int Sum [ ] = new int [ n + 1 ] ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( int i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
public static void main ( String [ ] args ) { int n = 6 ; System . out . println ( sumOddFibonacci ( n ) ) ; }
public class GFG { static long fun ( int marks [ ] , int n ) {
long dp [ ] = new long [ n ] ; int temp ; for ( int i = 0 ; i < n ; i ++ ) dp [ i ] = 1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( marks [ i ] > marks [ i + 1 ] ) { temp = i ; while ( true ) { if ( ( marks [ temp ] > marks [ temp + 1 ] ) && temp >= 0 ) { if ( dp [ temp ] > dp [ temp + 1 ] ) { temp -= 1 ; continue ; } else { dp [ temp ] = dp [ temp + 1 ] + 1 ; temp -= 1 ; } } else break ; } }
else if ( marks [ i ] < marks [ i + 1 ] ) dp [ i + 1 ] = dp [ i ] + 1 ; } int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += dp [ i ] ; return sum ; }
int n = 6 ;
int marks [ ] = { 1 , 4 , 5 , 2 , 2 , 1 } ;
System . out . println ( fun ( marks , n ) ) ; } }
class GFG { static int solve ( int N , int K ) {
int [ ] combo ; combo = new int [ 50 ] ;
combo [ 0 ] = 1 ;
for ( int i = 1 ; i <= K ; i ++ ) {
for ( int j = 0 ; j <= N ; j ++ ) {
if ( j >= i ) {
combo [ j ] += combo [ j - i ] ; } } }
return combo [ N ] ; }
int N = 29 ; int K = 5 ; System . out . println ( solve ( N , K ) ) ; solve ( N , K ) ; } }
static int computeLIS ( int circBuff [ ] , int start , int end , int n ) { int LIS [ ] = new int [ n + end - start ] ;
for ( int i = start ; i < end ; i ++ ) LIS [ i ] = 1 ;
for ( int i = start + 1 ; i < end ; i ++ )
for ( int j = start ; j < i ; j ++ ) if ( circBuff [ i ] > circBuff [ j ] && LIS [ i ] < LIS [ j ] + 1 ) LIS [ i ] = LIS [ j ] + 1 ;
int res = Integer . MIN_VALUE ; for ( int i = start ; i < end ; i ++ ) res = Math . max ( res , LIS [ i ] ) ; return res ; }
static int LICS ( int arr [ ] , int n ) {
int circBuff [ ] = new int [ 2 * n ] ; for ( int i = 0 ; i < n ; i ++ ) circBuff [ i ] = arr [ i ] ; for ( int i = n ; i < 2 * n ; i ++ ) circBuff [ i ] = arr [ i - n ] ;
int res = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) res = Math . max ( computeLIS ( circBuff , i , i + n , n ) , res ) ; return res ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 4 , 6 , 2 , 3 } ; System . out . println ( " Length ▁ of ▁ LICS ▁ is ▁ " + LICS ( arr , arr . length ) ) ; } }
static int min ( int a , int b ) { return a < b ? a : b ; }
static int binomialCoeff ( int n , int k ) { int C [ ] = new int [ k + 1 ] ; C [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
public static void main ( String [ ] args ) { int n = 3 , m = 2 ; System . out . println ( " Number ▁ of ▁ Paths : ▁ " + binomialCoeff ( n + m , n ) ) ; } }
static int LCIS ( int arr1 [ ] , int n , int arr2 [ ] , int m ) {
int table [ ] = new int [ m ] ; for ( int j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int current = 0 ;
for ( int j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
int result = 0 ; for ( int i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
public static void main ( String [ ] args ) { int arr1 [ ] = { 3 , 4 , 9 , 1 } ; int arr2 [ ] = { 5 , 3 , 8 , 9 , 10 , 2 , 1 } ; int n = arr1 . length ; int m = arr2 . length ; System . out . println ( " Length ▁ of ▁ LCIS ▁ is ▁ " + LCIS ( arr1 , n , arr2 , m ) ) ; } }
static int longComPre ( String arr [ ] , int N ) {
int [ ] [ ] freq = new int [ N ] [ 256 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
int M = arr [ i ] . length ( ) ;
for ( int j = 0 ; j < M ; j ++ ) {
freq [ i ] [ arr [ i ] . charAt ( j ) ] ++ ; } }
int maxLen = 0 ;
for ( int j = 0 ; j < 256 ; j ++ ) {
int minRowVal = Integer . MAX_VALUE ;
for ( int i = 0 ; i < N ; i ++ ) {
minRowVal = Math . min ( minRowVal , freq [ i ] [ j ] ) ; }
maxLen += minRowVal ; } return maxLen ; }
public static void main ( String [ ] args ) { String arr [ ] = { " aabdc " , " abcd " , " aacd " } ; int N = 3 ; System . out . print ( longComPre ( arr , N ) ) ; } }
import java . util . * ; class GFG { static int MAX_CHAR = 26 ;
static String removeChars ( char arr [ ] , int k ) {
int [ ] hash = new int [ MAX_CHAR ] ;
int n = arr . length ; for ( int i = 0 ; i < n ; ++ i ) hash [ arr [ i ] - ' a ' ] ++ ;
String ans = " " ;
for ( int i = 0 ; i < n ; ++ i ) {
if ( hash [ arr [ i ] - ' a ' ] != k ) { ans += arr [ i ] ; } } return ans ; }
public static void main ( String [ ] args ) { char str [ ] = " geeksforgeeks " . toCharArray ( ) ; int k = 2 ;
System . out . print ( removeChars ( str , k ) ) ; } }
static void sub_segments ( String str , int n ) { int l = str . length ( ) ; for ( int x = 0 ; x < l ; x += n ) { String newlist = str . substring ( x , x + n ) ;
List < Character > arr = new ArrayList < Character > ( ) ; for ( char y : newlist . toCharArray ( ) ) {
if ( ! arr . contains ( y ) ) arr . add ( y ) ; } for ( char y : arr ) System . out . print ( y ) ; System . out . println ( ) ; } }
public static void main ( String [ ] args ) { String str = " geeksforgeeksgfg " ; int n = 4 ; sub_segments ( str , n ) ; } }
class GFG { static boolean equalIgnoreCase ( String str1 , String str2 ) { int i = 0 ;
int len1 = str1 . length ( ) ;
int len2 = str2 . length ( ) ;
if ( len1 != len2 ) return false ;
while ( i < len1 ) {
if ( str1 . charAt ( i ) == str2 . charAt ( i ) ) { i ++ ; }
else if ( ! ( ( str1 . charAt ( i ) >= ' a ' && str1 . charAt ( i ) <= ' z ' ) || ( str1 . charAt ( i ) >= ' A ' && str1 . charAt ( i ) <= ' Z ' ) ) ) { return false ; }
else if ( ! ( ( str2 . charAt ( i ) >= ' a ' && str2 . charAt ( i ) <= ' z ' ) || ( str2 . charAt ( i ) >= ' A ' && str2 . charAt ( i ) <= ' Z ' ) ) ) { return false ; }
else {
if ( str1 . charAt ( i ) >= ' a ' && str1 . charAt ( i ) <= ' z ' ) { if ( str1 . charAt ( i ) - 32 != str2 . charAt ( i ) ) return false ; } else if ( str1 . charAt ( i ) >= ' A ' && str1 . charAt ( i ) <= ' Z ' ) { if ( str1 . charAt ( i ) + 32 != str2 . charAt ( i ) ) return false ; }
i ++ ;
return true ;
static void equalIgnoreCaseUtil ( String str1 , String str2 ) { boolean res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) System . out . println ( " Same " ) ; else System . out . println ( " Not ▁ Same " ) ; }
public static void main ( String args [ ] ) { String str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; } }
static String maxValue ( char [ ] a , char [ ] b ) {
Arrays . sort ( b ) ; int n = a . length ; int m = b . length ;
int j = m - 1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( j < 0 ) break ; if ( b [ j ] > a [ i ] ) { a [ i ] = b [ j ] ;
j -- ; } }
return String . valueOf ( a ) ; }
public static void main ( String [ ] args ) { String a = "1234" ; String b = "4321" ; System . out . print ( maxValue ( a . toCharArray ( ) , b . toCharArray ( ) ) ) ; } }
static boolean checkIfUnequal ( int n , int q ) {
String s1 = Integer . toString ( n ) ; int a [ ] = new int [ 26 ] ;
for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) a [ s1 . charAt ( i ) - '0' ] ++ ;
int prod = n * q ;
String s2 = Integer . toString ( prod ) ;
for ( int i = 0 ; i < s2 . length ( ) ; i ++ ) {
if ( a [ s2 . charAt ( i ) - '0' ] > 0 ) return false ; }
return true ; }
static int countInRange ( int l , int r , int q ) { int count = 0 ; for ( int i = l ; i <= r ; i ++ ) {
if ( checkIfUnequal ( i , q ) ) count ++ ; } return count ; }
public static void main ( String [ ] args ) { int l = 10 , r = 12 , q = 2 ;
System . out . println ( countInRange ( l , r , q ) ) ; } }
public static boolean is_possible ( String s ) {
int l = s . length ( ) ; int one = 0 , zero = 0 ; for ( int i = 0 ; i < l ; i ++ ) {
if ( s . charAt ( i ) == '0' ) zero ++ ;
else one ++ ; }
if ( l % 2 == 0 ) return ( one == zero ) ;
else return ( Math . abs ( one - zero ) == 1 ) ; } public static void main ( String [ ] args ) { String s = "100110" ; if ( is_possible ( s ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
class GFG { static int limit = 255 ; static void countFreq ( String str ) {
int [ ] count = new int [ limit + 1 ] ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) count [ str . charAt ( i ) ] ++ ; for ( int i = 0 ; i <= limit ; i ++ ) if ( count [ i ] > 0 ) System . out . println ( ( char ) i + " ▁ " + count [ i ] ) ; }
public static void main ( String [ ] args ) { String str = " GeeksforGeeks " ; countFreq ( str ) ; } }
static int __builtin_popcount ( int n ) { int count = 0 ; while ( n > 0 ) { count += n & 1 ; n >>= 1 ; } return count ; }
static void countEvenOdd ( int arr [ ] , int n , int K ) { int even = 0 , odd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } int y ;
y = __builtin_popcount ( K ) ;
if ( ( y & 1 ) != 0 ) { System . out . println ( " Even ▁ = ▁ " + odd + " , ▁ Odd ▁ = ▁ " + even ) ; }
else { System . out . println ( " Even ▁ = ▁ " + even + " , ▁ Odd ▁ = ▁ " + odd ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 2 , 15 , 9 , 8 , 8 } ; int K = 3 ; int n = arr . length ;
countEvenOdd ( arr , n , K ) ; } }
static String convert ( String s ) { int n = s . length ( ) ; String s1 = " " ; s1 = s1 + Character . toLowerCase ( s . charAt ( 0 ) ) ; for ( int i = 1 ; i < n ; i ++ ) {
if ( s . charAt ( i ) == ' ▁ ' && i < n ) {
s1 = s1 + " ▁ " + Character . toLowerCase ( s . charAt ( i + 1 ) ) ; i ++ ; }
else s1 = s1 + Character . toUpperCase ( s . charAt ( i ) ) ; }
return s1 ; }
public static void main ( String [ ] args ) { String str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " ; System . out . println ( convert ( str ) ) ; } }
static int reverse ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; } return rev_num ; }
static int properDivSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= Math . sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; } static boolean isTcefrep ( int n ) { return properDivSum ( n ) == reverse ( n ) ; }
int N = 6 ;
if ( isTcefrep ( N ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
public class GFG { static String findNthNo ( int n ) { String res = " " ; while ( n >= 1 ) {
if ( ( n & 1 ) == 1 ) { res = res + "3" ; n = ( n - 1 ) / 2 ; }
else { res = res + "5" ; n = ( n - 2 ) / 2 ; } }
StringBuilder sb = new StringBuilder ( res ) ; sb . reverse ( ) ; return new String ( sb ) ; }
public static void main ( String args [ ] ) { int n = 5 ; System . out . print ( findNthNo ( n ) ) ; } }
static int findNthNonSquare ( int n ) {
double x = ( double ) n ;
double ans = x + Math . floor ( 0.5 + Math . sqrt ( x ) ) ; return ( int ) ans ; }
int n = 16 ;
System . out . print ( " The ▁ " + n + " th ▁ Non - Square ▁ number ▁ is ▁ " ) ; System . out . print ( findNthNonSquare ( n ) ) ; } }
static int seiresSum ( int n , int [ ] a ) { return n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ; }
public static void main ( String args [ ] ) { int n = 2 ; int a [ ] = { 1 , 2 , 3 , 4 } ; System . out . println ( seiresSum ( n , a ) ) ; } }
public static boolean checkdigit ( int n , int k ) { while ( n != 0 ) {
int rem = n % 10 ;
if ( rem == k ) return true ; n = n / 10 ; } return false ; }
public static int findNthNumber ( int n , int k ) {
for ( int i = k + 1 , count = 1 ; count < n ; i ++ ) {
if ( checkdigit ( i , k ) || ( i % k == 0 ) ) count ++ ; if ( count == n ) return i ; } return - 1 ; }
public static void main ( String [ ] args ) { int n = 10 , k = 2 ; System . out . println ( findNthNumber ( n , k ) ) ; } }
import java . util . * ; class GFG { public static int find_permutations ( Vector < Integer > arr ) { int cnt = 0 ; int max_ind = - 1 , min_ind = 10000000 ; int n = arr . size ( ) ; HashMap < Integer , Integer > index_of = new HashMap < > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { index_of . put ( arr . get ( i ) , i + 1 ) ; } for ( int i = 1 ; i <= n ; i ++ ) {
max_ind = Math . max ( max_ind , index_of . get ( i ) ) ; min_ind = Math . min ( min_ind , index_of . get ( i ) ) ; if ( max_ind - min_ind + 1 == i ) cnt ++ ; } return cnt ; }
public static void main ( String [ ] args ) { Vector < Integer > nums = new Vector < Integer > ( ) ; nums . add ( 2 ) ; nums . add ( 3 ) ; nums . add ( 1 ) ; nums . add ( 5 ) ; nums . add ( 4 ) ; System . out . print ( find_permutations ( nums ) ) ; } }
class GFG {
static int getCount ( int [ ] a , int n ) {
int gcd = 0 ; for ( int i = 0 ; i < n ; i ++ ) gcd = calgcd ( gcd , a [ i ] ) ;
int cnt = 0 ; for ( int i = 1 ; i * i <= gcd ; i ++ ) { if ( gcd % i == 0 ) {
if ( i * i == gcd ) cnt ++ ;
else cnt += 2 ; } } return cnt ; }
public static void main ( String [ ] args ) { int [ ] a = { 4 , 16 , 1024 , 48 } ; int n = a . length ; System . out . println ( getCount ( a , n ) ) ; } }
public static int delCost ( String s , int [ ] cost ) {
boolean visited [ ] = new boolean [ s . length ( ) ] ;
int ans = 0 ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( visited [ i ] ) { continue ; }
int maxDel = 0 ;
int totalCost = 0 ;
visited [ i ] = true ;
for ( int j = i ; j < s . length ( ) ; j ++ ) {
if ( s . charAt ( i ) == s . charAt ( j ) ) {
maxDel = Math . max ( maxDel , cost [ j ] ) ; totalCost += cost [ j ] ;
visited [ j ] = true ; } }
ans += totalCost - maxDel ; }
return ans ; }
String s = " AAABBB " ;
int [ ] cost = { 1 , 2 , 3 , 4 , 5 , 6 } ;
System . out . println ( delCost ( s , cost ) ) ; } }
static void checkXOR ( int arr [ ] , int N ) {
if ( N % 2 == 0 ) {
int xro = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
xro ^= arr [ i ] ; }
if ( xro != 0 ) { System . out . println ( - 1 ) ; return ; }
for ( int i = 0 ; i < N - 3 ; i += 2 ) { System . out . println ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( i + 2 ) ) ; }
for ( int i = 0 ; i < N - 3 ; i += 2 ) { System . out . println ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( N - 1 ) ) ; } } else {
for ( int i = 0 ; i < N - 2 ; i += 2 ) { System . out . println ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( i + 2 ) ) ; }
for ( int i = 0 ; i < N - 2 ; i += 2 ) { System . out . println ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( N - 1 ) ) ; } } }
int arr [ ] = { 4 , 2 , 1 , 7 , 2 } ;
int N = arr . length ;
checkXOR ( arr , N ) ; } }
static int make_array_element_even ( int arr [ ] , int N ) {
int res = 0 ;
int odd_cont_seg = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
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
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 5 , 11 , 6 } ; int N = arr . length ; System . out . print ( make_array_element_even ( arr , N ) ) ; } }
public static int zvalue ( int [ ] nums ) {
int m = max_element ( nums ) ; int cnt = 0 ;
for ( int i = 0 ; i <= m ; i ++ ) { cnt = 0 ;
for ( int j = 0 ; j < nums . length ; j ++ ) {
if ( nums [ j ] >= i ) cnt ++ ; }
if ( cnt == i ) return i ; }
return - 1 ; }
public static int max_element ( int [ ] nums ) { int max = nums [ 0 ] ; for ( int i = 1 ; i < nums . length ; i ++ ) max = Math . max ( max , nums [ i ] ) ; return max ; }
public static void main ( String args [ ] ) { int [ ] nums = { 7 , 8 , 9 , 0 , 0 , 1 } ; System . out . println ( zvalue ( nums ) ) ; } }
static String [ ] lexico_smallest ( String s1 , String s2 ) {
Map < Character , Integer > M = new HashMap < > ( ) ; Set < Character > S = new TreeSet < > ( ) ;
for ( int i = 0 ; i <= s1 . length ( ) - 1 ; ++ i ) {
if ( ! M . containsKey ( s1 . charAt ( i ) ) ) M . put ( s1 . charAt ( i ) , 1 ) ; else M . replace ( s1 . charAt ( i ) , M . get ( s1 . charAt ( i ) ) + 1 ) ;
S . add ( s1 . charAt ( i ) ) ; }
for ( int i = 0 ; i <= s2 . length ( ) - 1 ; ++ i ) { if ( M . containsKey ( s2 . charAt ( i ) ) ) M . replace ( s2 . charAt ( i ) , M . get ( s2 . charAt ( i ) ) - 1 ) ; } char c = s2 . charAt ( 0 ) ; int index = 0 ; String res = " " ;
Iterator < Character > it = S . iterator ( ) ; while ( it . hasNext ( ) ) { char x = it . next ( ) ;
if ( x != c ) { for ( int i = 1 ; i <= M . get ( x ) ; ++ i ) { res += x ; } } else {
int j = 0 ; index = res . length ( ) ;
while ( s2 . charAt ( j ) == x ) { j ++ ; }
if ( s2 . charAt ( j ) < c ) { res += s2 ; for ( int i = 1 ; i <= M . get ( x ) ; ++ i ) { res += x ; } } else { for ( int i = 1 ; i <= M . get ( x ) ; ++ i ) { res += x ; } index += M . get ( x ) ; res += s2 ; } } } String pr [ ] = { res , index + " " } ; return pr ; }
return pr ; }
static String lexico_largest ( String s1 , String s2 ) {
String pr [ ] = lexico_smallest ( s1 , s2 ) ;
String d1 = " " ; for ( int i = Integer . valueOf ( pr [ 1 ] ) - 1 ; i >= 0 ; i -- ) { d1 += pr [ 0 ] . charAt ( i ) ; }
String d2 = " " ; for ( int i = pr [ 0 ] . length ( ) - 1 ; i >= Integer . valueOf ( pr [ 1 ] ) + s2 . length ( ) ; -- i ) { d2 += pr [ 0 ] . charAt ( i ) ; } String res = d2 + s2 + d1 ;
return res ; }
String s1 = " ethgakagmenpgs " ; String s2 = " geeks " ;
System . out . println ( lexico_smallest ( s1 , s2 ) [ 0 ] ) ; System . out . println ( lexico_largest ( s1 , s2 ) ) ; } }
import java . util . * ; class GFG { static int sz = ( int ) 1e5 ;
static Vector < Integer > [ ] tree = new Vector [ sz ] ;
static int n ;
static boolean [ ] vis = new boolean [ sz ] ;
static int [ ] subtreeSize = new int [ sz ] ;
static void addEdge ( int a , int b ) {
tree [ a ] . add ( b ) ;
tree [ b ] . add ( a ) ; }
static void dfs ( int x ) {
vis [ x ] = true ;
subtreeSize [ x ] = 1 ;
for ( int i : tree [ x ] ) { if ( ! vis [ i ] ) { dfs ( i ) ; subtreeSize [ x ] += subtreeSize [ i ] ; } } }
static void countPairs ( int a , int b ) { int sub = Math . min ( subtreeSize [ a ] , subtreeSize [ b ] ) ; System . out . print ( sub * ( n - sub ) + "NEW_LINE"); }
n = 6 ; for ( int i = 0 ; i < tree . length ; i ++ ) tree [ i ] = new Vector < Integer > ( ) ; addEdge ( 0 , 1 ) ; addEdge ( 0 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 3 , 4 ) ; addEdge ( 3 , 5 ) ;
dfs ( 0 ) ;
countPairs ( 1 , 3 ) ; countPairs ( 0 , 2 ) ; } }
static int findPermutation ( Set < Integer > arr , int N ) { int pos = arr . size ( ) + 1 ;
if ( pos > N ) return 1 ; int res = 0 ; for ( int i = 1 ; i <= N ; i ++ ) {
if ( ! arr . contains ( i ) ) {
if ( i % pos == 0 pos % i == 0 ) {
arr . add ( i ) ;
res += findPermutation ( arr , N ) ;
arr . remove ( i ) ; } } }
return res ; }
public static void main ( String [ ] args ) { int N = 5 ; Set < Integer > arr = new HashSet < Integer > ( ) ; System . out . print ( findPermutation ( arr , N ) ) ; } }
static void solve ( int arr [ ] , int n , int X , int Y ) {
int diff = Y - X ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] != 1 ) { diff = diff % ( arr [ i ] - 1 ) ; } }
if ( diff == 0 ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 7 , 9 , 10 } ; int n = arr . length ; int X = 11 , Y = 13 ; solve ( arr , n , X , Y ) ; } }
import java . util . * ; class GFG { static final int maxN = 100001 ;
@ SuppressWarnings ( " unchecked " ) static Vector < Integer > [ ] adj = new Vector [ maxN ] ;
static int [ ] height = new int [ maxN ] ;
static int [ ] dist = new int [ maxN ] ;
static void addEdge ( int u , int v ) {
adj [ u ] . add ( v ) ;
adj [ v ] . add ( u ) ; }
static void dfs1 ( int cur , int par ) {
for ( int u : adj [ cur ] ) { if ( u != par ) {
dfs1 ( u , cur ) ;
height [ cur ] = Math . max ( height [ cur ] , height [ u ] ) ; } }
height [ cur ] += 1 ; }
static void dfs2 ( int cur , int par ) { int max1 = 0 ; int max2 = 0 ;
for ( int u : adj [ cur ] ) { if ( u != par ) {
if ( height [ u ] >= max1 ) { max2 = max1 ; max1 = height [ u ] ; } else if ( height [ u ] > max2 ) { max2 = height [ u ] ; } } } int sum = 0 ; for ( int u : adj [ cur ] ) { if ( u != par ) {
sum = ( ( max1 == height [ u ] ) ? max2 : max1 ) ; if ( max1 == height [ u ] ) dist [ u ] = 1 + Math . max ( 1 + max2 , dist [ cur ] ) ; else dist [ u ] = 1 + Math . max ( 1 + max1 , dist [ cur ] ) ;
dfs2 ( u , cur ) ; } } }
public static void main ( String [ ] args ) { int n = 6 ; for ( int i = 0 ; i < adj . length ; i ++ ) adj [ i ] = new Vector < Integer > ( ) ; addEdge ( 1 , 2 ) ; addEdge ( 2 , 3 ) ; addEdge ( 2 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 5 , 6 ) ;
dfs1 ( 1 , 0 ) ;
dfs2 ( 1 , 0 ) ;
for ( int i = 1 ; i <= n ; i ++ ) System . out . print ( ( Math . max ( dist [ i ] , height [ i ] ) - 1 ) + " ▁ " ) ; } }
import java . util . * ; class Middle {
public static int middleOfThree ( int a , int b , int c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && a < c ) || ( c < a && a < b ) ) return a ; else return c ; }
public static void main ( String [ ] args ) { int a = 20 , b = 30 , c = 40 ; System . out . println ( middleOfThree ( a , b , c ) ) ; } }
static void selectionSort ( int arr [ ] , int n ) { int i , j , min_idx ;
for ( i = 0 ; i < n - 1 ; i ++ ) {
min_idx = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
int temp = arr [ min_idx ] ; arr [ min_idx ] = arr [ i ] ; arr [ i ] = temp ; } }
static void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } System . out . println ( ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 64 , 25 , 12 , 22 , 11 } ; int n = arr . length ;
selectionSort ( arr , n ) ; System . out . print ( "Sorted array: NEW_LINE");
printArray ( arr , n ) ; } }
import java . util . * ; class GFG { static boolean checkStr1CanConStr2 ( String str1 , String str2 ) {
int N = str1 . length ( ) ;
int M = str2 . length ( ) ;
HashSet < Integer > st1 = new HashSet < > ( ) ;
HashSet < Integer > st2 = new HashSet < > ( ) ;
int hash1 [ ] = new int [ 256 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
hash1 [ str1 . charAt ( i ) ] ++ ; }
for ( int i = 0 ; i < N ; i ++ ) {
st1 . add ( ( int ) str1 . charAt ( i ) ) ; }
for ( int i = 0 ; i < M ; i ++ ) {
st2 . add ( ( int ) str2 . charAt ( i ) ) ; }
if ( ! st1 . equals ( st2 ) ) { return false ; }
int hash2 [ ] = new int [ 256 ] ;
for ( int i = 0 ; i < M ; i ++ ) {
hash2 [ str2 . charAt ( i ) ] ++ ; }
Arrays . sort ( hash1 ) ;
Arrays . sort ( hash2 ) ;
for ( int i = 0 ; i < 256 ; i ++ ) {
if ( hash1 [ i ] != hash2 [ i ] ) { return false ; } } return true ; }
public static void main ( String [ ] args ) { String str1 = " xyyzzlll " ; String str2 = " yllzzxxx " ; if ( checkStr1CanConStr2 ( str1 , str2 ) ) { System . out . print ( " True " ) ; } else { System . out . print ( " False " ) ; } } }
static void partSort ( int [ ] arr , int N , int a , int b ) {
int l = Math . min ( a , b ) ; int r = Math . max ( a , b ) ;
Arrays . sort ( arr , l , r + 1 ) ;
for ( int i = 0 ; i < N ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int [ ] arr = { 7 , 8 , 4 , 5 , 2 } ; int a = 1 , b = 4 ; int N = arr . length ; partSort ( arr , N , a , b ) ; } }
class shortest_path { static int INF = Integer . MAX_VALUE , N = 4 ;
static int minCost ( int cost [ ] [ ] ) {
int dist [ ] = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) dist [ j ] = dist [ i ] + cost [ i ] [ j ] ; return dist [ N - 1 ] ; }
public static void main ( String args [ ] ) { int cost [ ] [ ] = { { 0 , 15 , 80 , 90 } , { INF , 0 , 40 , 50 } , { INF , INF , 0 , 70 } , { INF , INF , INF , 0 } } ; System . out . println ( " The ▁ Minimum ▁ cost ▁ to ▁ reach ▁ station ▁ " + N + " ▁ is ▁ " + minCost ( cost ) ) ; } }
static int numOfways ( int n , int k ) { int p = 1 ; if ( k % 2 != 0 ) p = - 1 ; return ( int ) ( Math . pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
public static void main ( String args [ ] ) { int n = 4 , k = 2 ; System . out . println ( numOfways ( n , k ) ) ; } }
static char largest_alphabet ( String a , int n ) {
char max = ' A ' ;
for ( int i = 0 ; i < n ; i ++ ) if ( a . charAt ( i ) > max ) max = a . charAt ( i ) ;
return max ; }
static char smallest_alphabet ( String a , int n ) {
char min = ' z ' ;
for ( int i = 0 ; i < n - 1 ; i ++ ) if ( a . charAt ( i ) < min ) min = a . charAt ( i ) ;
return min ; }
String a = " GeEksforGeeks " ;
int size = a . length ( ) ;
System . out . print ( " Largest ▁ and ▁ smallest ▁ alphabet ▁ is ▁ : ▁ " ) ; System . out . print ( largest_alphabet ( a , size ) + " ▁ and ▁ " ) ; System . out . println ( smallest_alphabet ( a , size ) ) ; } }
static String maximumPalinUsingKChanges ( String str , int k ) { char palin [ ] = str . toCharArray ( ) ; String ans = " " ;
int l = 0 ; int r = str . length ( ) - 1 ;
while ( l < r ) {
if ( str . charAt ( l ) != str . charAt ( r ) ) { palin [ l ] = palin [ r ] = ( char ) Math . max ( str . charAt ( l ) , str . charAt ( r ) ) ; k -- ; } l ++ ; r -- ; }
if ( k < 0 ) { return " Not ▁ possible " ; } l = 0 ; r = str . length ( ) - 1 ; while ( l <= r ) {
if ( l == r ) { if ( k > 0 ) { palin [ l ] = '9' ; } }
if ( palin [ l ] < '9' ) {
if ( k >= 2 && palin [ l ] == str . charAt ( l ) && palin [ r ] == str . charAt ( r ) ) { k -= 2 ; palin [ l ] = palin [ r ] = '9' ; }
else if ( k >= 1 && ( palin [ l ] != str . charAt ( l ) || palin [ r ] != str . charAt ( r ) ) ) { k -- ; palin [ l ] = palin [ r ] = '9' ; } } l ++ ; r -- ; } for ( int i = 0 ; i < palin . length ; i ++ ) ans += palin [ i ] ; return ans ; }
public static void main ( String [ ] args ) throws ParseException { String str = "43435" ; int k = 3 ; System . out . println ( maximumPalinUsingKChanges ( str , k ) ) ; } }
static int countTriplets ( int [ ] A ) {
int cnt = 0 ;
HashMap < Integer , Integer > tuples = new HashMap < Integer , Integer > ( ) ;
for ( int a : A )
for ( int b : A ) { if ( tuples . containsKey ( a & b ) ) tuples . put ( a & b , tuples . get ( a & b ) + 1 ) ; else tuples . put ( a & b , 1 ) ; }
for ( int a : A )
for ( Map . Entry < Integer , Integer > t : tuples . entrySet ( ) )
if ( ( t . getKey ( ) & a ) == 0 ) cnt += t . getValue ( ) ;
return cnt ; }
int [ ] A = { 2 , 1 , 3 } ;
System . out . print ( countTriplets ( A ) ) ; } }
import java . util . * ; public class Main { static int min ;
static void parity ( List < Integer > even , List < Integer > odd , List < Integer > v , int i ) {
if ( i == v . size ( ) || even . size ( ) == 0 && odd . size ( ) == 0 ) { int count = 0 ; for ( int j = 0 ; j < v . size ( ) - 1 ; j ++ ) { if ( v . get ( j ) % 2 != v . get ( j + 1 ) % 2 ) count ++ ; } if ( count < min ) min = count ; return ; }
if ( v . get ( i ) != - 1 ) parity ( even , odd , v , i + 1 ) ;
else { if ( even . size ( ) != 0 ) { int x = even . get ( even . size ( ) - 1 ) ; even . remove ( even . size ( ) - 1 ) ; v . set ( i , x ) ; parity ( even , odd , v , i + 1 ) ;
even . add ( x ) ; } if ( odd . size ( ) != 0 ) { int x = odd . get ( odd . size ( ) - 1 ) ; odd . remove ( odd . size ( ) - 1 ) ; v . set ( i , x ) ; parity ( even , odd , v , i + 1 ) ;
odd . add ( x ) ; } } }
static void minDiffParity ( List < Integer > v , int n ) {
List < Integer > even = new ArrayList < Integer > ( ) ;
List < Integer > odd = new ArrayList < Integer > ( ) ; HashMap < Integer , Integer > m = new HashMap < > ( ) ; for ( int i = 1 ; i <= n ; i ++ ) { if ( m . containsKey ( i ) ) { m . replace ( i , 1 ) ; } else { m . put ( i , 1 ) ; } } for ( int i = 0 ; i < v . size ( ) ; i ++ ) {
if ( v . get ( i ) != - 1 ) m . remove ( v . get ( i ) ) ; }
for ( Map . Entry < Integer , Integer > i : m . entrySet ( ) ) { if ( i . getKey ( ) % 2 == 0 ) { even . add ( i . getKey ( ) ) ; } else { odd . add ( i . getKey ( ) ) ; } } min = 1000 ; parity ( even , odd , v , 0 ) ; System . out . println ( min ) ; }
public static void main ( String [ ] args ) { int n = 8 ; List < Integer > v = new ArrayList < Integer > ( ) ; v . add ( 2 ) ; v . add ( 1 ) ; v . add ( 4 ) ; v . add ( - 1 ) ; v . add ( - 1 ) ; v . add ( 6 ) ; v . add ( - 1 ) ; v . add ( 8 ) ; minDiffParity ( v , n ) ; } }
import java . util . * ; class GFG { static int MAX = 100005 ; static Vector < Vector < Integer > > adjacent = new Vector < Vector < Integer > > ( ) ; static boolean visited [ ] = new boolean [ MAX ] ;
static int startnode , endnode , thirdnode ; static int maxi = - 1 , N ;
static int parent [ ] = new int [ MAX ] ;
static boolean vis [ ] = new boolean [ MAX ] ;
static void dfs ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent . get ( u ) . size ( ) ; i ++ ) { if ( ! visited [ adjacent . get ( u ) . get ( i ) ] ) { temp ++ ; dfs ( adjacent . get ( u ) . get ( i ) , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; startnode = u ; } } }
static void dfs1 ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent . get ( u ) . size ( ) ; i ++ ) { if ( ! visited [ adjacent . get ( u ) . get ( i ) ] ) { temp ++ ; parent [ adjacent . get ( u ) . get ( i ) ] = u ; dfs1 ( adjacent . get ( u ) . get ( i ) , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; endnode = u ; } } }
static void dfs2 ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent . get ( u ) . size ( ) ; i ++ ) { if ( ! visited [ adjacent . get ( u ) . get ( i ) ] && ! vis [ adjacent . get ( u ) . get ( i ) ] ) { temp ++ ; dfs2 ( adjacent . get ( u ) . get ( i ) , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; thirdnode = u ; } } }
static void findNodes ( ) {
dfs ( 1 , 0 ) ; for ( int i = 0 ; i <= N ; i ++ ) visited [ i ] = false ; maxi = - 1 ;
dfs1 ( startnode , 0 ) ; for ( int i = 0 ; i <= N ; i ++ ) visited [ i ] = false ;
int x = endnode ; vis [ startnode ] = true ;
while ( x != startnode ) { vis [ x ] = true ; x = parent [ x ] ; } maxi = - 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( vis [ i ] ) dfs2 ( i , 0 ) ; } }
public static void main ( String args [ ] ) { for ( int i = 0 ; i < MAX ; i ++ ) adjacent . add ( new Vector < Integer > ( ) ) ; N = 4 ; adjacent . get ( 1 ) . add ( 2 ) ; adjacent . get ( 2 ) . add ( 1 ) ; adjacent . get ( 1 ) . add ( 3 ) ; adjacent . get ( 3 ) . add ( 1 ) ; adjacent . get ( 1 ) . add ( 4 ) ; adjacent . get ( 4 ) . add ( 1 ) ; findNodes ( ) ; System . out . print ( " ( " + startnode + " , ▁ " + endnode + " , ▁ " + thirdnode + " ) " ) ; } }
import java . io . * ; class GFG { static void newvol ( double x ) { System . out . print ( " percentage ▁ increase ▁ in ▁ the " + " ▁ volume ▁ of ▁ the ▁ sphere ▁ is ▁ " + ( Math . pow ( x , 3 ) / 10000 + 3 * x + ( 3 * Math . pow ( x , 2 ) ) / 100 ) + " % " ) ; }
public static void main ( String [ ] args ) { double x = 10 ; newvol ( x ) ; } }
static void length_of_chord ( double r , double x ) { System . out . println ( " The ▁ length ▁ of ▁ the ▁ chord " + " ▁ of ▁ the ▁ circle ▁ is ▁ " + 2 * r * Math . sin ( x * ( 3.14 / 180 ) ) ) ; }
public static void main ( String [ ] args ) { double r = 4 , x = 63 ; length_of_chord ( r , x ) ; } }
static float area ( float a ) {
if ( a < 0 ) return - 1 ;
float area = ( float ) Math . sqrt ( a ) / 6 ; return area ; }
public static void main ( String [ ] args ) { float a = 10 ; System . out . println ( area ( a ) ) ; } }
static double longestRodInCuboid ( int length , int breadth , int height ) { double result ; int temp ;
temp = length * length + breadth * breadth + height * height ;
result = Math . sqrt ( temp ) ; return result ; }
public static void main ( String [ ] args ) { int length = 12 , breadth = 9 , height = 8 ;
System . out . println ( ( int ) longestRodInCuboid ( length , breadth , height ) ) ; } }
static boolean LiesInsieRectangle ( int a , int b , int x , int y ) { if ( x - y - b <= 0 && x - y + b >= 0 && x + y - 2 * a + b <= 0 && x + y - b >= 0 ) return true ; return false ; }
public static void main ( String [ ] args ) { int a = 7 , b = 2 , x = 4 , y = 5 ; if ( LiesInsieRectangle ( a , b , x , y ) ) System . out . println ( " Given ▁ point ▁ lies ▁ " + " inside ▁ the ▁ rectangle " ) ; else System . out . println ( " Given ▁ point ▁ does ▁ not ▁ " + " lie ▁ on ▁ the ▁ rectangle " ) ; } }
static int maxvolume ( int s ) { int maxvalue = 0 ;
for ( int i = 1 ; i <= s - 2 ; i ++ ) {
for ( int j = 1 ; j <= s - 1 ; j ++ ) {
int k = s - i - j ;
maxvalue = Math . max ( maxvalue , i * j * k ) ; } } return maxvalue ; }
public static void main ( String [ ] args ) { int s = 8 ; System . out . println ( maxvolume ( s ) ) ; } }
static int maxvolume ( int s ) {
int length = s / 3 ; s -= length ;
int breadth = s / 2 ;
int height = s - breadth ; return length * breadth * height ; }
public static void main ( String [ ] args ) { int s = 8 ; System . out . println ( maxvolume ( s ) ) ; } }
public static double hexagonArea ( double s ) { return ( ( 3 * Math . sqrt ( 3 ) * ( s * s ) ) / 2 ) ; }
double s = 4 ; System . out . print ( " Area : ▁ " + hexagonArea ( s ) ) ; } }
static int maxSquare ( int b , int m ) {
return ( b / m - 1 ) * ( b / m ) / 2 ; }
public static void main ( String args [ ] ) { int b = 10 , m = 2 ; System . out . println ( maxSquare ( b , m ) ) ; } }
static void findRightAngle ( double A , double H ) {
double D = Math . pow ( H , 4 ) - 16 * A * A ; if ( D >= 0 ) {
double root1 = ( H * H + Math . sqrt ( D ) ) / 2 ; double root2 = ( H * H - Math . sqrt ( D ) ) / 2 ; double a = Math . sqrt ( root1 ) ; double b = Math . sqrt ( root2 ) ; if ( b >= a ) System . out . print ( a + " ▁ " + b + " ▁ " + H ) ; else System . out . print ( b + " ▁ " + a + " ▁ " + H ) ; } else System . out . print ( " - 1" ) ; }
public static void main ( String arg [ ] ) { findRightAngle ( 6 , 5 ) ; } }
class Squares { public static int numberOfSquares ( int base ) {
base = ( base - 2 ) ;
base = Math . floorDiv ( base , 2 ) ; return base * ( base + 1 ) / 2 ; }
public static void main ( String args [ ] ) { int base = 8 ; System . out . println ( numberOfSquares ( base ) ) ; } }
static void performQuery ( int arr [ ] , int Q [ ] [ ] ) {
for ( int i = 0 ; i < Q . length ; i ++ ) {
int or = 0 ;
int x = Q [ i ] [ 0 ] ; arr [ x - 1 ] = Q [ i ] [ 1 ] ;
for ( int j = 0 ; j < arr . length ; j ++ ) { or = or | arr [ j ] ; }
System . out . print ( or + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 } ; int Q [ ] [ ] = { { 1 , 4 } , { 3 , 0 } } ; performQuery ( arr , Q ) ; } }
static int smallest ( int k , int d ) { int cnt = 1 ; int m = d % k ;
int [ ] v = new int [ k ] ; Arrays . fill ( v , 0 ) ; v [ m ] = 1 ;
while ( 1 != 0 ) { if ( m == 0 ) return cnt ; m = ( ( ( m * ( 10 % k ) ) % k ) + ( d % k ) ) % k ;
if ( v [ m ] == 1 ) return - 1 ; v [ m ] = 1 ; cnt ++ ; } }
public static void main ( String [ ] args ) { int d = 1 ; int k = 41 ; System . out . println ( smallest ( k , d ) ) ; } }
static int fib ( int n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
static int findVertices ( int n ) {
return fib ( n + 2 ) ; } public static void main ( String args [ ] ) {
int n = 3 ; System . out . println ( findVertices ( n ) ) ; } }
static void checkCommonDivisor ( int [ ] arr , int N , int X ) {
int G = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { G = gcd ( G , arr [ i ] ) ; } int copy_G = G ; for ( int divisor = 2 ; divisor <= X ; divisor ++ ) {
while ( G % divisor == 0 ) {
G = G / divisor ; } }
if ( G <= X ) { System . out . println ( " Yes " ) ;
for ( int i = 0 ; i < N ; i ++ ) System . out . print ( ( arr [ i ] / copy_G ) + " ▁ " ) ; System . out . println ( ) ; }
else System . out . println ( " No " ) ; }
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
int [ ] arr = { 6 , 15 , 6 } ; int X = 6 ;
int N = arr . length ; checkCommonDivisor ( arr , N , X ) ; } }
public class GFG { public static void printSpiral ( int size ) { int row = 0 , col = 0 ; int boundary = size - 1 ; int sizeLeft = size - 1 ; int flag = 1 ;
char move = ' r ' ;
int matrix [ ] [ ] = new int [ size ] [ size ] ; for ( int i = 1 ; i < size * size + 1 ; i ++ ) {
matrix [ row ] [ col ] = i ;
switch ( move ) {
case ' r ' : col += 1 ; break ;
case ' l ' : col -= 1 ; break ;
case ' u ' : row -= 1 ; break ;
case ' d ' : row += 1 ; break ; }
if ( i == boundary ) {
boundary += sizeLeft ;
if ( flag != 2 ) { flag = 2 ; } else { flag = 1 ; sizeLeft -= 1 ; }
switch ( move ) {
case ' r ' : move = ' d ' ; break ;
case ' d ' : move = ' l ' ; break ;
case ' l ' : move = ' u ' ; break ;
case ' u ' : move = ' r ' ; break ; } } }
for ( row = 0 ; row < size ; row ++ ) { for ( col = 0 ; col < size ; col ++ ) { int n = matrix [ row ] [ col ] ; System . out . print ( ( n < 10 ) ? ( n + " ▁ " ) : ( n + " ▁ " ) ) ; } System . out . println ( ) ; } }
int size = 5 ;
printSpiral ( size ) ; } }
static class Node { int data ; Node next ; Node prev ; }
static Node reverse ( Node head_ref ) { Node temp = null ; Node current = head_ref ;
while ( current != null ) { temp = current . prev ; current . prev = current . next ; current . next = temp ; current = current . prev ; }
if ( temp != null ) head_ref = temp . prev ; return head_ref ; }
static Node merge ( Node first , Node second ) {
if ( first == null ) return second ;
if ( second == null ) return first ;
if ( first . data < second . data ) { first . next = merge ( first . next , second ) ; first . next . prev = first ; first . prev = null ; return first ; } else { second . next = merge ( first , second . next ) ; second . next . prev = second ; second . prev = null ; return second ; } }
static Node sort ( Node head ) {
if ( head == null head . next == null ) return head ; Node current = head . next ; while ( current != null ) {
if ( current . data < current . prev . data ) break ;
current = current . next ; }
if ( current == null ) return head ;
current . prev . next = null ; current . prev = null ;
current = reverse ( current ) ;
return merge ( head , current ) ; }
static Node push ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . prev = null ;
new_node . next = ( head_ref ) ;
if ( ( head_ref ) != null ) ( head_ref ) . prev = new_node ;
( head_ref ) = new_node ; return head_ref ; }
static void printList ( Node head ) {
if ( head == null ) System . out . println ( " Doubly ▁ Linked ▁ list ▁ empty " ) ; while ( head != null ) { System . out . print ( head . data + " ▁ " ) ; head = head . next ; } }
public static void main ( String args [ ] ) { Node head = null ;
head = push ( head , 1 ) ; head = push ( head , 4 ) ; head = push ( head , 6 ) ; head = push ( head , 10 ) ; head = push ( head , 12 ) ; head = push ( head , 7 ) ; head = push ( head , 5 ) ; head = push ( head , 2 ) ; System . out . println ( " Original ▁ Doubly ▁ linked ▁ list : n " ) ; printList ( head ) ;
head = sort ( head ) ; System . out . println ( " Doubly linked list after sorting : n "); printList ( head ) ; } }
static class Node { char data ; Node next ; }
static Node newNode ( char key ) { Node temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
static void printlist ( Node head ) { if ( head == null ) { System . out . println ( " Empty ▁ List " ) ; return ; } while ( head != null ) { System . out . print ( head . data + " ▁ " ) ; if ( head . next != null ) System . out . print ( " - > ▁ " ) ; head = head . next ; } System . out . println ( ) ; }
static boolean isVowel ( char x ) { return ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) ; }
static Node arrange ( Node head ) { Node newHead = head ;
Node latestVowel ; Node curr = head ;
if ( head == null ) return null ;
if ( isVowel ( head . data ) == true )
latestVowel = head ; else {
while ( curr . next != null && ! isVowel ( curr . next . data ) ) curr = curr . next ;
if ( curr . next == null ) return head ;
latestVowel = newHead = curr . next ; curr . next = curr . next . next ; latestVowel . next = head ; }
while ( curr != null && curr . next != null ) { if ( isVowel ( curr . next . data ) == true ) {
if ( curr == latestVowel ) {
latestVowel = curr = curr . next ; } else {
Node temp = latestVowel . next ;
latestVowel . next = curr . next ;
latestVowel = latestVowel . next ;
curr . next = curr . next . next ;
latestVowel . next = temp ; } } else {
curr = curr . next ; } } return newHead ; }
public static void main ( String [ ] args ) { Node head = newNode ( ' a ' ) ; head . next = newNode ( ' b ' ) ; head . next . next = newNode ( ' c ' ) ; head . next . next . next = newNode ( ' e ' ) ; head . next . next . next . next = newNode ( ' d ' ) ; head . next . next . next . next . next = newNode ( ' o ' ) ; head . next . next . next . next . next . next = newNode ( ' x ' ) ; head . next . next . next . next . next . next . next = newNode ( ' i ' ) ; System . out . println ( " Linked ▁ list ▁ before ▁ : ▁ " ) ; printlist ( head ) ; head = arrange ( head ) ; System . out . println ( " Linked ▁ list ▁ after ▁ : " ) ; printlist ( head ) ; } }
static class Node { int data ; Node left , right ; }
static Node newNode ( int data ) { Node temp = new Node ( ) ; temp . data = data ; temp . right = null ; temp . left = null ; return temp ; } static Node KthLargestUsingMorrisTraversal ( Node root , int k ) { Node curr = root ; Node Klargest = null ;
int count = 0 ; while ( curr != null ) {
if ( curr . right == null ) {
if ( ++ count == k ) Klargest = curr ;
curr = curr . left ; } else {
Node succ = curr . right ; while ( succ . left != null && succ . left != curr ) succ = succ . left ; if ( succ . left == null ) {
succ . left = curr ;
curr = curr . right ; }
else { succ . left = null ; if ( ++ count == k ) Klargest = curr ;
curr = curr . left ; } } } return Klargest ; }
Node root = newNode ( 4 ) ; root . left = newNode ( 2 ) ; root . right = newNode ( 7 ) ; root . left . left = newNode ( 1 ) ; root . left . right = newNode ( 3 ) ; root . right . left = newNode ( 6 ) ; root . right . right = newNode ( 10 ) ; System . out . println ( " Finding ▁ K - th ▁ largest ▁ Node ▁ in ▁ BST ▁ : ▁ " + KthLargestUsingMorrisTraversal ( root , 2 ) . data ) ; } }
import java . util . Arrays ; import java . util . Collections ; class GFG { static int MAX_SIZE = 10 ;
static void sortByRow ( Integer mat [ ] [ ] , int n , boolean ascending ) { for ( int i = 0 ; i < n ; i ++ ) { if ( ascending ) Arrays . sort ( mat [ i ] ) ; else Arrays . sort ( mat [ i ] , Collections . reverseOrder ( ) ) ; } }
static void transpose ( Integer mat [ ] [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) {
int temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ j ] [ i ] ; mat [ j ] [ i ] = temp ; } }
static void sortMatRowAndColWise ( Integer mat [ ] [ ] , int n ) {
sortByRow ( mat , n , true ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n , false ) ;
transpose ( mat , n ) ; }
static void printMat ( Integer mat [ ] [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) System . out . print ( mat [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int n = 3 ; Integer mat [ ] [ ] = { { 3 , 2 , 1 } , { 9 , 8 , 7 } , { 6 , 5 , 4 } } ; System . out . print ( "Original Matrix:NEW_LINE"); printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; System . out . print ( " Matrix After Sorting : "); printMat ( mat , n ) ; } }
import java . util . Arrays ; class GFG { static final int MAX_SIZE = 10 ;
static void sortByRow ( int mat [ ] [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ )
Arrays . sort ( mat [ i ] ) ; }
static void transpose ( int mat [ ] [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) {
int temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ j ] [ i ] ; mat [ j ] [ i ] = temp ; } }
static void sortMatRowAndColWise ( int mat [ ] [ ] , int n ) {
sortByRow ( mat , n ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n ) ;
transpose ( mat , n ) ; }
static void printMat ( int mat [ ] [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) System . out . print ( mat [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 4 , 1 , 3 } , { 9 , 6 , 8 } , { 5 , 2 , 7 } } ; int n = 3 ; System . out . print ( "Original Matrix:NEW_LINE"); printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; System . out . print ( " Matrix After Sorting : "); printMat ( mat , n ) ; } }
static void doublyEven ( int n ) { int [ ] [ ] arr = new int [ n ] [ n ] ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) for ( j = 0 ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * i ) + j + 1 ;
for ( i = 0 ; i < n / 4 ; i ++ ) for ( j = 0 ; j < n / 4 ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 0 ; i < n / 4 ; i ++ ) for ( j = 3 * ( n / 4 ) ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 3 * n / 4 ; i < n ; i ++ ) for ( j = 0 ; j < n / 4 ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 3 * n / 4 ; i < n ; i ++ ) for ( j = 3 * n / 4 ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = n / 4 ; i < 3 * n / 4 ; i ++ ) for ( j = n / 4 ; j < 3 * n / 4 ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = 0 ; j < n ; j ++ ) System . out . print ( arr [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int n = 8 ;
doublyEven ( n ) ; } }
static int cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
static void Kroneckerproduct ( int A [ ] [ ] , int B [ ] [ ] ) { int [ ] [ ] C = new int [ rowa * rowb ] [ cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; System . out . print ( C [ i + l + 1 ] [ j + k + 1 ] + " ▁ " ) ; } } System . out . println ( ) ; } } }
public static void main ( String [ ] args ) { int A [ ] [ ] = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } ; int B [ ] [ ] = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; } }
import java . io . * ; class Lower_triangular { int N = 4 ;
boolean isLowerTriangularMatrix ( int mat [ ] [ ] ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != 0 ) return false ; return true ; }
public static void main ( String args [ ] ) { Lower_triangular ob = new Lower_triangular ( ) ; int mat [ ] [ ] = { { 1 , 0 , 0 , 0 } , { 1 , 4 , 0 , 0 } , { 4 , 6 , 2 , 0 } , { 0 , 4 , 7 , 6 } } ;
if ( ob . isLowerTriangularMatrix ( mat ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . util . * ; import java . lang . * ; public class GfG { private static final int N = 4 ;
public static Boolean isUpperTriangularMatrix ( int mat [ ] [ ] ) { for ( int i = 1 ; i < N ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( mat [ i ] [ j ] != 0 ) return false ; return true ; }
public static void main ( String argc [ ] ) { int [ ] [ ] mat = { { 1 , 3 , 5 , 3 } , { 0 , 4 , 6 , 2 } , { 0 , 0 , 2 , 5 } , { 0 , 0 , 0 , 6 } } ; if ( isUpperTriangularMatrix ( mat ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static final int m = 3 ;
static final int n = 2 ;
static long countSets ( int a [ ] [ ] ) {
long res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < m ; j ++ ) { if ( a [ i ] [ j ] == 1 ) u ++ ; else v ++ ; } res += Math . pow ( 2 , u ) - 1 + Math . pow ( 2 , v ) - 1 ; }
for ( int i = 0 ; i < m ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( a [ j ] [ i ] == 1 ) u ++ ; else v ++ ; } res += Math . pow ( 2 , u ) - 1 + Math . pow ( 2 , v ) - 1 ; }
return res - ( n * m ) ; }
public static void main ( String [ ] args ) { int a [ ] [ ] = { { 1 , 0 , 1 } , { 0 , 1 , 0 } } ; System . out . print ( countSets ( a ) ) ; } }
import java . io . * ; class GFG { static int MAX = 100 ;
static void transpose ( int mat [ ] [ ] , int tr [ ] [ ] , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) tr [ i ] [ j ] = mat [ j ] [ i ] ; }
static boolean isSymmetric ( int mat [ ] [ ] , int N ) { int tr [ ] [ ] = new int [ N ] [ MAX ] ; transpose ( mat , tr , N ) ; for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != tr [ i ] [ j ] ) return false ; return true ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . io . * ; class GFG { static int MAX = 100 ;
static boolean isSymmetric ( int mat [ ] [ ] , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != mat [ j ] [ i ] ) return false ; return true ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " NO " ) ; } }
static int MAX = 100 ;
static int findNormal ( int mat [ ] [ ] , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) sum += mat [ i ] [ j ] * mat [ i ] [ j ] ; return ( int ) Math . sqrt ( sum ) ; }
static int findTrace ( int mat [ ] [ ] , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += mat [ i ] [ i ] ; return sum ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; System . out . println ( " Trace ▁ of ▁ Matrix ▁ = ▁ " + findTrace ( mat , 5 ) ) ; System . out . println ( " Normal ▁ of ▁ Matrix ▁ = ▁ " + findNormal ( mat , 5 ) ) ; } }
static int maxDet ( int n ) { return ( 2 * n * n * n ) ; }
void resMatrix ( int n ) { for ( int i = 0 ; i < 3 ; i ++ ) { for ( int j = 0 ; j < 3 ; j ++ ) {
if ( i == 0 && j == 2 ) System . out . print ( "0 ▁ " ) ; else if ( i == 1 && j == 0 ) System . out . print ( "0 ▁ " ) ; else if ( i == 2 && j == 1 ) System . out . print ( "0 ▁ " ) ;
else System . out . print ( n + " ▁ " ) ; } System . out . println ( " " ) ; } }
static public void main ( String [ ] args ) { int n = 15 ; GFG geeks = new GFG ( ) ; System . out . println ( " Maximum ▁ Determinant ▁ = ▁ " + maxDet ( n ) ) ; System . out . println ( " Resultant ▁ Matrix ▁ : " ) ; geeks . resMatrix ( n ) ; } }
import java . util . * ; import java . lang . * ; import java . io . * ; class GFG { static int countNegative ( int M [ ] [ ] , int n , int m ) { int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { if ( M [ i ] [ j ] < 0 ) count += 1 ;
else break ; } } return count ; }
public static void main ( String [ ] args ) { int M [ ] [ ] = { { - 3 , - 2 , - 1 , 1 } , { - 2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; System . out . println ( countNegative ( M , 3 , 4 ) ) ; } }
static int countNegative ( int M [ ] [ ] , int n , int m ) {
int count = 0 ;
int i = 0 ; int j = m - 1 ;
while ( j >= 0 && i < n ) { if ( M [ i ] [ j ] < 0 ) {
count += j + 1 ;
i += 1 ; }
else j -= 1 ; } return count ; }
public static void main ( String [ ] args ) { int M [ ] [ ] = { { - 3 , - 2 , - 1 , 1 } , { - 2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; System . out . println ( countNegative ( M , 3 , 4 ) ) ; } }
static int findMaxValue ( int N , int mat [ ] [ ] ) {
int maxValue = Integer . MIN_VALUE ;
for ( int a = 0 ; a < N - 1 ; a ++ ) for ( int b = 0 ; b < N - 1 ; b ++ ) for ( int d = a + 1 ; d < N ; d ++ ) for ( int e = b + 1 ; e < N ; e ++ ) if ( maxValue < ( mat [ d ] [ e ] - mat [ a ] [ b ] ) ) maxValue = mat [ d ] [ e ] - mat [ a ] [ b ] ; return maxValue ; }
public static void main ( String [ ] args ) { int N = 5 ; int mat [ ] [ ] = { { 1 , 2 , - 1 , - 4 , - 20 } , { - 8 , - 3 , 4 , 2 , 1 } , { 3 , 8 , 6 , 1 , 3 } , { - 4 , - 1 , 1 , 7 , - 6 } , { 0 , - 4 , 10 , - 5 , 1 } } ; System . out . print ( " Maximum ▁ Value ▁ is ▁ " + findMaxValue ( N , mat ) ) ; } }
static int findMaxValue ( int N , int mat [ ] [ ] ) {
int maxValue = Integer . MIN_VALUE ;
int maxArr [ ] [ ] = new int [ N ] [ N ] ;
maxArr [ N - 1 ] [ N - 1 ] = mat [ N - 1 ] [ N - 1 ] ;
int maxv = mat [ N - 1 ] [ N - 1 ] ; for ( int j = N - 2 ; j >= 0 ; j -- ) { if ( mat [ N - 1 ] [ j ] > maxv ) maxv = mat [ N - 1 ] [ j ] ; maxArr [ N - 1 ] [ j ] = maxv ; }
maxv = mat [ N - 1 ] [ N - 1 ] ; for ( int i = N - 2 ; i >= 0 ; i -- ) { if ( mat [ i ] [ N - 1 ] > maxv ) maxv = mat [ i ] [ N - 1 ] ; maxArr [ i ] [ N - 1 ] = maxv ; }
for ( int i = N - 2 ; i >= 0 ; i -- ) { for ( int j = N - 2 ; j >= 0 ; j -- ) {
if ( maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] > maxValue ) maxValue = maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] ;
maxArr [ i ] [ j ] = Math . max ( mat [ i ] [ j ] , Math . max ( maxArr [ i ] [ j + 1 ] , maxArr [ i + 1 ] [ j ] ) ) ; } } return maxValue ; }
public static void main ( String [ ] args ) { int N = 5 ; int mat [ ] [ ] = { { 1 , 2 , - 1 , - 4 , - 20 } , { - 8 , - 3 , 4 , 2 , 1 } , { 3 , 8 , 6 , 1 , 3 } , { - 4 , - 1 , 1 , 7 , - 6 } , { 0 , - 4 , 10 , - 5 , 1 } } ; System . out . print ( " Maximum ▁ Value ▁ is ▁ " + findMaxValue ( N , mat ) ) ; } }
class GFG { static final int INF = Integer . MAX_VALUE ; static final int N = 4 ;
static void youngify ( int mat [ ] [ ] , int i , int j ) {
int downVal = ( i + 1 < N ) ? mat [ i + 1 ] [ j ] : INF ; int rightVal = ( j + 1 < N ) ? mat [ i ] [ j + 1 ] : INF ;
if ( downVal == INF && rightVal == INF ) { return ; }
if ( downVal < rightVal ) { mat [ i ] [ j ] = downVal ; mat [ i + 1 ] [ j ] = INF ; youngify ( mat , i + 1 , j ) ; } else { mat [ i ] [ j ] = rightVal ; mat [ i ] [ j + 1 ] = INF ; youngify ( mat , i , j + 1 ) ; } }
static int extractMin ( int mat [ ] [ ] ) { int ret = mat [ 0 ] [ 0 ] ; mat [ 0 ] [ 0 ] = INF ; youngify ( mat , 0 , 0 ) ; return ret ; }
static void printSorted ( int mat [ ] [ ] ) { System . out . println ( " Elements ▁ of ▁ matrix ▁ in ▁ sorted ▁ order ▁ n " ) ; for ( int i = 0 ; i < N * N ; i ++ ) { System . out . print ( extractMin ( mat ) + " ▁ " ) ; } }
static final int n = 5 ;
static void printSumSimple ( int mat [ ] [ ] , int k ) {
if ( k > n ) return ;
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
for ( int j = 0 ; j < n - k + 1 ; j ++ ) {
int sum = 0 ; for ( int p = i ; p < k + i ; p ++ ) for ( int q = j ; q < k + j ; q ++ ) sum += mat [ p ] [ q ] ; System . out . print ( sum + " ▁ " ) ; }
System . out . println ( ) ; } }
public static void main ( String arg [ ] ) { int mat [ ] [ ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } } ; int k = 3 ; printSumSimple ( mat , k ) ; } }
static int n = 5 ;
static void printSumTricky ( int mat [ ] [ ] , int k ) {
if ( k > n ) return ;
int stripSum [ ] [ ] = new int [ n ] [ n ] ;
for ( int j = 0 ; j < n ; j ++ ) {
int sum = 0 ; for ( int i = 0 ; i < k ; i ++ ) sum += mat [ i ] [ j ] ; stripSum [ 0 ] [ j ] = sum ;
for ( int i = 1 ; i < n - k + 1 ; i ++ ) { sum += ( mat [ i + k - 1 ] [ j ] - mat [ i - 1 ] [ j ] ) ; stripSum [ i ] [ j ] = sum ; } }
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
int sum = 0 ; for ( int j = 0 ; j < k ; j ++ ) sum += stripSum [ i ] [ j ] ; System . out . print ( sum + " ▁ " ) ;
for ( int j = 1 ; j < n - k + 1 ; j ++ ) { sum += ( stripSum [ i ] [ j + k - 1 ] - stripSum [ i ] [ j - 1 ] ) ; System . out . print ( sum + " ▁ " ) ; } System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; int k = 3 ; printSumTricky ( mat , k ) ; } }
class GFG { static final int M = 3 ; static final int N = 4 ;
static void transpose ( int A [ ] [ ] , int B [ ] [ ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < M ; j ++ ) B [ i ] [ j ] = A [ j ] [ i ] ; }
public static void main ( String [ ] args ) { int A [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } } ; int B [ ] [ ] = new int [ N ] [ M ] , i , j ; transpose ( A , B ) ; System . out . print ( "Result matrix is NEW_LINE"); for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < M ; j ++ ) System . out . print ( B [ i ] [ j ] + " ▁ " ) ; System . out . print ( "NEW_LINE"); } } }
class GFG { static final int N = 4 ;
static void transpose ( int A [ ] [ ] ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) { int temp = A [ i ] [ j ] ; A [ i ] [ j ] = A [ j ] [ i ] ; A [ j ] [ i ] = temp ; } }
public static void main ( String [ ] args ) { int A [ ] [ ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; transpose ( A ) ; System . out . print ( "Modified matrix is NEW_LINE"); for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) System . out . print ( A [ i ] [ j ] + " ▁ " ) ; System . out . print ( "NEW_LINE"); } } }
class GFG { static final int R = 3 ; static final int C = 3 ;
static int pathCountRec ( int mat [ ] [ ] , int m , int n , int k ) {
if ( m < 0 n < 0 ) { return 0 ; } if ( m == 0 && n == 0 && ( k == mat [ m ] [ n ] ) ) { return 1 ; }
return pathCountRec ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountRec ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ; }
static int pathCount ( int mat [ ] [ ] , int k ) { return pathCountRec ( mat , R - 1 , C - 1 , k ) ; }
public static void main ( String [ ] args ) { int k = 12 ; int mat [ ] [ ] = { { 1 , 2 , 3 } , { 4 , 6 , 5 } , { 3 , 2 , 1 } } ; System . out . println ( pathCount ( mat , k ) ) ; } }
class GFG { static final int R = 3 ; static final int C = 3 ; static final int MAX_K = 100 ; static int [ ] [ ] [ ] dp = new int [ R ] [ C ] [ MAX_K ] ; static int pathCountDPRecDP ( int [ ] [ ] mat , int m , int n , int k ) {
if ( m < 0 n < 0 ) return 0 ; if ( m == 0 && n == 0 ) return ( k == mat [ m ] [ n ] ? 1 : 0 ) ;
if ( dp [ m ] [ n ] [ k ] != - 1 ) return dp [ m ] [ n ] [ k ] ;
dp [ m ] [ n ] [ k ] = pathCountDPRecDP ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountDPRecDP ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ; return dp [ m ] [ n ] [ k ] ; }
static int pathCountDP ( int [ ] [ ] mat , int k ) { for ( int i = 0 ; i < R ; i ++ ) for ( int j = 0 ; j < C ; j ++ ) for ( int l = 0 ; l < MAX_K ; l ++ ) dp [ i ] [ j ] [ l ] = - 1 ; return pathCountDPRecDP ( mat , R - 1 , C - 1 , k ) ; }
public static void main ( String [ ] args ) { int k = 12 ; int [ ] [ ] mat = new int [ ] [ ] { new int [ ] { 1 , 2 , 3 } , new int [ ] { 4 , 6 , 5 } , new int [ ] { 3 , 2 , 1 } } ; System . out . println ( pathCountDP ( mat , k ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int SIZE = 10 ;
static void sortMat ( int mat [ ] [ ] , int n ) {
int temp [ ] = new int [ n * n ] ; int k = 0 ;
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) temp [ k ++ ] = mat [ i ] [ j ] ;
Arrays . sort ( temp ) ;
k = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) mat [ i ] [ j ] = temp [ k ++ ] ; }
static void printMat ( int mat [ ] [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) System . out . print ( mat [ i ] [ j ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String args [ ] ) { int mat [ ] [ ] = { { 5 , 4 , 7 } , { 1 , 3 , 8 } , { 2 , 9 , 6 } } ; int n = 3 ; System . out . println ( " Original ▁ Matrix : " ) ; printMat ( mat , n ) ; sortMat ( mat , n ) ; System . out . println ( " Matrix ▁ After ▁ Sorting : " ) ; printMat ( mat , n ) ; } }
int findCrossOver ( int arr [ ] , int low , int high , int x ) {
if ( arr [ high ] <= x ) return high ;
if ( arr [ low ] > x ) return low ;
int mid = ( low + high ) / 2 ;
if ( arr [ mid ] <= x && arr [ mid + 1 ] > x ) return mid ;
if ( arr [ mid ] < x ) return findCrossOver ( arr , mid + 1 , high , x ) ; return findCrossOver ( arr , low , mid - 1 , x ) ; }
void printKclosest ( int arr [ ] , int x , int k , int n ) {
int l = findCrossOver ( arr , 0 , n - 1 , x ) ;
int r = l + 1 ;
int count = 0 ;
if ( arr [ l ] == x ) l -- ;
while ( l >= 0 && r < n && count < k ) { if ( x - arr [ l ] < arr [ r ] - x ) System . out . print ( arr [ l -- ] + " ▁ " ) ; else System . out . print ( arr [ r ++ ] + " ▁ " ) ; count ++ ; }
while ( count < k && l >= 0 ) { System . out . print ( arr [ l -- ] + " ▁ " ) ; count ++ ; }
while ( count < k && r < n ) { System . out . print ( arr [ r ++ ] + " ▁ " ) ; count ++ ; } }
public static void main ( String args [ ] ) { KClosest ob = new KClosest ( ) ; int arr [ ] = { 12 , 16 , 22 , 30 , 35 , 39 , 42 , 45 , 48 , 50 , 53 , 55 , 56 } ; int n = arr . length ; int x = 35 , k = 4 ; ob . printKclosest ( arr , x , 4 , n ) ; } }
public class LinkedlistIS { node head ; node sorted ; class node { int val ; node next ; public node ( int val ) { this . val = val ; } }
void push ( int val ) {
node newnode = new node ( val ) ;
newnode . next = head ;
head = newnode ; }
void insertionSort ( node headref ) {
sorted = null ; node current = headref ;
while ( current != null ) {
node next = current . next ;
sortedInsert ( current ) ;
current = next ; }
head = sorted ; }
void sortedInsert ( node newnode ) {
if ( sorted == null sorted . val >= newnode . val ) { newnode . next = sorted ; sorted = newnode ; } else { node current = sorted ;
while ( current . next != null && current . next . val < newnode . val ) { current = current . next ; } newnode . next = current . next ; current . next = newnode ; } }
void printlist ( node head ) { while ( head != null ) { System . out . print ( head . val + " ▁ " ) ; head = head . next ; } }
public static void main ( String [ ] args ) { LinkedlistIS list = new LinkedlistIS ( ) ; list . push ( 5 ) ; list . push ( 20 ) ; list . push ( 4 ) ; list . push ( 3 ) ; list . push ( 30 ) ; System . out . println ( " Linked ▁ List ▁ before ▁ Sorting . . " ) ; list . printlist ( list . head ) ; list . insertionSort ( list . head ) ; System . out . println ( " LinkedList After sorting "); list . printlist ( list . head ) ; } }
public static int count ( int S [ ] , int m , int n ) {
int table [ ] = new int [ n + 1 ] ;
table [ 0 ] = 1 ;
for ( int i = 0 ; i < m ; i ++ ) for ( int j = S [ i ] ; j <= n ; j ++ ) table [ j ] += table [ j - S [ i ] ] ; return table [ n ] ; }
import java . io . * ; import java . util . * ; class GFG { static int [ ] [ ] dp = new int [ 100 ] [ 100 ] ;
static int matrixChainMemoised ( int [ ] p , int i , int j ) { if ( i == j ) { return 0 ; } if ( dp [ i ] [ j ] != - 1 ) { return dp [ i ] [ j ] ; } dp [ i ] [ j ] = Integer . MAX_VALUE ; for ( int k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i ] [ j ] ; } static int MatrixChainOrder ( int [ ] p , int n ) { int i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int n = arr . length ; for ( int [ ] row : dp ) Arrays . fill ( row , - 1 ) ; System . out . println ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , n ) ) ; } }
static int MatrixChainOrder ( int p [ ] , int n ) {
int m [ ] [ ] = new int [ n ] [ n ] ; int i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i ] [ i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; if ( j == n ) continue ; m [ i ] [ j ] = Integer . MAX_VALUE ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i ] [ j ] ) m [ i ] [ j ] = q ; } } } return m [ 1 ] [ n - 1 ] ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 2 , 3 , 4 } ; int size = arr . length ; System . out . println ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , size ) ) ; } }
static int cutRod ( int price [ ] , int n ) { if ( n <= 0 ) return 0 ; int max_val = Integer . MIN_VALUE ;
for ( int i = 0 ; i < n ; i ++ ) max_val = Math . max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) ; return max_val ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . length ; System . out . println ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
static int cutRod ( int price [ ] , int n ) { int val [ ] = new int [ n + 1 ] ; val [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { int max_val = Integer . MIN_VALUE ; for ( int j = 0 ; j < i ; j ++ ) max_val = Math . max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . length ; System . out . println ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
static int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; return - 1 ; }
public static void main ( String [ ] args ) { System . out . print ( "NEW_LINE" + multiply(5, -11)); } }
class SieveOfEratosthenes { void sieveOfEratosthenes ( int n ) {
boolean prime [ ] = new boolean [ n + 1 ] ; for ( int i = 0 ; i <= n ; i ++ ) prime [ i ] = true ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( int i = 2 ; i <= n ; i ++ ) { if ( prime [ i ] == true ) System . out . print ( i + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int n = 30 ; System . out . print ( " Following ▁ are ▁ the ▁ prime ▁ numbers ▁ " ) ; System . out . println ( " smaller ▁ than ▁ or ▁ equal ▁ to ▁ " + n ) ; SieveOfEratosthenes g = new SieveOfEratosthenes ( ) ; g . sieveOfEratosthenes ( n ) ; } }
static int binomialCoeff ( int n , int k ) { int res = 1 ; if ( k > n - k ) k = n - k ; for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static void printPascal ( int n ) {
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) System . out . print ( binomialCoeff ( line , i ) + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String args [ ] ) { int n = 7 ; printPascal ( n ) ; } }
public static void printPascal ( int n ) {
int [ ] [ ] arr = new int [ n ] [ n ] ;
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) {
if ( line == i i == 0 ) arr [ line ] [ i ] = 1 ;
else arr [ line ] [ i ] = arr [ line - 1 ] [ i - 1 ] + arr [ line - 1 ] [ i ] ; System . out . print ( arr [ line ] [ i ] ) ; } System . out . println ( " " ) ; } } }
public static void main ( String [ ] args ) { int n = 5 ; printPascal ( n ) ; }
import java . io . * ; class GFG { public static void printPascal ( int n ) { for ( int line = 1 ; line <= n ; line ++ ) {
int C = 1 ; for ( int i = 1 ; i <= line ; i ++ ) {
System . out . print ( C + " ▁ " ) ; C = C * ( line - i ) / i ; } System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int n = 5 ; printPascal ( n ) ; } }
import java . io . * ; class GFG { static int Add ( int x , int y ) {
while ( y != 0 ) {
int carry = x & y ;
x = x ^ y ;
y = carry << 1 ; } return x ; }
public static void main ( String arg [ ] ) { System . out . println ( Add ( 15 , 32 ) ) ; } }
static int Add ( int x , int y ) { if ( y == 0 ) return x ; else return Add ( x ^ y , ( x & y ) << 1 ) ; }
static int getModulo ( int n , int d ) { return ( n & ( d - 1 ) ) ; }
public static void main ( String [ ] args ) { int n = 6 ;
int d = 4 ; System . out . println ( n + " ▁ moduo ▁ " + d + " ▁ is ▁ " + getModulo ( n , d ) ) ; } }
static int countSetBits ( int n ) { int count = 0 ; while ( n > 0 ) { count += n & 1 ; n >>= 1 ; } return count ; }
public static void main ( String args [ ] ) { int i = 9 ; System . out . println ( countSetBits ( i ) ) ; } }
public static int countSetBits ( int n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
int n = 9 ;
System . out . println ( countSetBits ( n ) ) ; } }
static int [ ] BitsSetTable256 = new int [ 256 ] ;
public static void initialize ( ) {
BitsSetTable256 [ 0 ] = 0 ; for ( int i = 0 ; i < 256 ; i ++ ) { BitsSetTable256 [ i ] = ( i & 1 ) + BitsSetTable256 [ i / 2 ] ; } }
public static int countSetBits ( int n ) { return ( BitsSetTable256 [ n & 0xff ] + BitsSetTable256 [ ( n >> 8 ) & 0xff ] + BitsSetTable256 [ ( n >> 16 ) & 0xff ] + BitsSetTable256 [ n >> 24 ] ) ; }
initialize ( ) ; int n = 9 ; System . out . print ( countSetBits ( n ) ) ; } }
public static void main ( String [ ] args ) { System . out . println ( Integer . bitCount ( 4 ) ) ; System . out . println ( Integer . bitCount ( 15 ) ) ; } }
class GFG { static int [ ] num_to_bits = new int [ ] { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
static int countSetBitsRec ( int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
public static void main ( String [ ] args ) { int num = 31 ; System . out . println ( countSetBitsRec ( num ) ) ; } }
static int countSetBits ( int N ) { int count = 0 ;
for ( int i = 0 ; i < 4 * 8 ; i ++ ) { if ( ( N & ( 1 << i ) ) != 0 ) count ++ ; } return count ; }
public static void main ( String [ ] args ) { int N = 15 ; System . out . println ( countSetBits ( N ) ) ; } }
static boolean getParity ( int n ) { boolean parity = false ; while ( n != 0 ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
public static void main ( String [ ] args ) { int n = 12 ; System . out . println ( " Parity ▁ of ▁ no ▁ " + n + " ▁ = ▁ " + ( getParity ( n ) ? " odd " : " even " ) ) ; } }
static boolean isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( int ) ( Math . ceil ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) == ( int ) ( Math . floor ( ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) ) ; }
public static void main ( String [ ] args ) { if ( isPowerOfTwo ( 31 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; if ( isPowerOfTwo ( 64 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; while ( n != 1 ) { if ( n % 2 != 0 ) return false ; n = n / 2 ; } return true ; }
public static void main ( String args [ ] ) { if ( isPowerOfTwo ( 31 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; if ( isPowerOfTwo ( 64 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean powerOf2 ( int n ) {
if ( n == 1 ) return true ;
else if ( n % 2 != 0 n == 0 ) return false ;
return powerOf2 ( n / 2 ) ; }
int n = 64 ;
int m = 12 ; if ( powerOf2 ( n ) == true ) System . out . print ( " True " + "NEW_LINE"); else System . out . print ( " False " + "NEW_LINE"); if ( powerOf2 ( m ) == true ) System . out . print ( " True " + "NEW_LINE"); else System . out . print ( " False " + "NEW_LINE"); } }
static boolean isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void main ( String [ ] args ) { System . out . println ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; System . out . println ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
static int maxRepeating ( int arr [ ] , int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) arr [ ( arr [ i ] % k ) ] += k ;
int max = arr [ 0 ] , result = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; result = i ; } }
return result ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 } ; int n = arr . length ; int k = 8 ; System . out . println ( " Maximum ▁ repeating ▁ element ▁ is : ▁ " + maxRepeating ( arr , n , k ) ) ; } }
static int fun ( int x ) { int y = ( x / 4 ) * 4 ;
int ans = 0 ; for ( int i = y ; i <= x ; i ++ ) ans ^= i ; return ans ; }
static int query ( int x ) {
if ( x == 0 ) return 0 ; int k = ( x + 1 ) / 2 ;
return ( ( x %= 2 ) != 0 ) ? 2 * fun ( k ) : ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) ; } static void allQueries ( int q , int l [ ] , int r [ ] ) { for ( int i = 0 ; i < q ; i ++ ) System . out . println ( ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) ) ; }
public static void main ( String [ ] args ) { int q = 3 ; int [ ] l = { 2 , 2 , 5 } ; int [ ] r = { 4 , 8 , 9 } ; allQueries ( q , l , r ) ; } }
static void prefixXOR ( int arr [ ] , int preXOR [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { while ( arr [ i ] % 2 != 1 ) arr [ i ] /= 2 ; preXOR [ i ] = arr [ i ] ; }
for ( int i = 1 ; i < n ; i ++ ) preXOR [ i ] = preXOR [ i - 1 ] ^ preXOR [ i ] ; }
static int query ( int preXOR [ ] , int l , int r ) { if ( l == 0 ) return preXOR [ r ] ; else return preXOR [ r ] ^ preXOR [ l - 1 ] ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 4 , 5 } ; int n = arr . length ; int preXOR [ ] = new int [ n ] ; prefixXOR ( arr , preXOR , n ) ; System . out . println ( query ( preXOR , 0 , 2 ) ) ; System . out . println ( query ( preXOR , 1 , 2 ) ) ; } }
static int findMinSwaps ( int arr [ ] , int n ) {
int noOfZeroes [ ] = new int [ n ] ; int i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
public static void main ( String args [ ] ) { int ar [ ] = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; System . out . println ( findMinSwaps ( ar , ar . length ) ) ; } }
import java . io . * ; class GFG { public static int minswaps ( int arr [ ] , int n ) { int count = 0 ; int num_unplaced_zeros = 0 ; for ( int index = n - 1 ; index >= 0 ; index -- ) { if ( arr [ index ] == 0 ) num_unplaced_zeros += 1 ; else count += num_unplaced_zeros ; } return count ; }
public static void main ( String [ ] args ) { int [ ] arr = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; System . out . println ( minswaps ( arr , 9 ) ) ; } }
static boolean arraySortedOrNot ( int arr [ ] , int n ) {
if ( n == 0 n == 1 ) return true ; for ( int i = 1 ; i < n ; i ++ )
if ( arr [ i - 1 ] > arr [ i ] ) return false ;
return true ; }
public static void main ( String [ ] args ) { int arr [ ] = { 20 , 23 , 23 , 45 , 78 , 88 } ; int n = arr . length ; if ( arraySortedOrNot ( arr , n ) ) System . out . print ( "YesNEW_LINE"); else System . out . print ( "NoNEW_LINE"); } }
static void printTwoOdd ( int arr [ ] , int size ) {
int xor2 = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( ( arr [ i ] & set_bit_no ) > 0 ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } System . out . println ( " The ▁ two ▁ ODD ▁ elements ▁ are ▁ " + x + " ▁ & ▁ " + y ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = arr . length ; printTwoOdd ( arr , arr_size ) ; } }
static boolean findPair ( int arr [ ] , int n ) { int size = arr . length ;
int i = 0 , j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { System . out . print ( " Pair ▁ Found : ▁ " + " ( ▁ " + arr [ i ] + " , ▁ " + arr [ j ] + " ▁ ) " ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } System . out . print ( " No ▁ such ▁ pair " ) ; return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 8 , 30 , 40 , 100 } ; int n = 60 ; findPair ( arr , n ) ; } }
public static void printMax ( int arr [ ] , int k , int n ) {
Integer [ ] brr = new Integer [ n ] ; for ( int i = 0 ; i < n ; i ++ ) brr [ i ] = arr [ i ] ;
Arrays . sort ( brr , Collections . reverseOrder ( ) ) ;
for ( int i = 0 ; i < n ; ++ i ) if ( Arrays . binarySearch ( brr , arr [ i ] , Collections . reverseOrder ( ) ) >= 0 && Arrays . binarySearch ( brr , arr [ i ] , Collections . reverseOrder ( ) ) < k ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 50 , 8 , 45 , 12 , 25 , 40 , 84 } ; int n = arr . length ; int k = 3 ; printMax ( arr , k , n ) ; } }
static void printSmall ( int arr [ ] , int asize , int n ) {
int [ ] copy_arr = Arrays . copyOf ( arr , asize ) ;
Arrays . sort ( copy_arr ) ;
for ( int i = 0 ; i < asize ; ++ i ) { if ( Arrays . binarySearch ( copy_arr , 0 , n , arr [ i ] ) > - 1 ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 } ; int asize = arr . length ; int n = 5 ; printSmall ( arr , asize , n ) ; } }
static boolean checkIsAP ( int arr [ ] , int n ) { if ( n == 1 ) return true ;
Arrays . sort ( arr ) ;
int d = arr [ 1 ] - arr [ 0 ] ; for ( int i = 2 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] != d ) return false ; return true ; }
public static void main ( String [ ] args ) { int arr [ ] = { 20 , 15 , 5 , 0 , 10 } ; int n = arr . length ; if ( checkIsAP ( arr , n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . util . * ; class GFG { static int countPairs ( int a [ ] , int n ) {
int mn = Integer . MAX_VALUE ; int mx = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) { mn = Math . min ( mn , a [ i ] ) ; mx = Math . max ( mx , a [ i ] ) ; }
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == mn ) c1 ++ ; if ( a [ i ] == mx ) c2 ++ ; }
if ( mn == mx ) return n * ( n - 1 ) / 2 ; else return c1 * c2 ; }
public static void main ( String [ ] args ) { int a [ ] = { 3 , 2 , 1 , 1 , 3 } ; int n = a . length ; System . out . print ( countPairs ( a , n ) ) ; } }
class Geeks { static class Node { int data ; Node next ; }
static Node rearrange ( Node head ) {
if ( head == null ) return null ;
Node prev = head , curr = head . next ; while ( curr != null ) {
if ( prev . data > curr . data ) { int t = prev . data ; prev . data = curr . data ; curr . data = t ; }
if ( curr . next != null && curr . next . data > curr . data ) { int t = curr . next . data ; curr . next . data = curr . data ; curr . data = t ; } prev = curr . next ; if ( curr . next == null ) break ; curr = curr . next . next ; } return head ; }
static Node push ( Node head , int k ) { Node tem = new Node ( ) ; tem . data = k ; tem . next = head ; head = tem ; return head ; }
static void display ( Node head ) { Node curr = head ; while ( curr != null ) { System . out . printf ( " % d ▁ " , curr . data ) ; curr = curr . next ; } }
head = push ( head , 7 ) ; head = push ( head , 3 ) ; head = push ( head , 8 ) ; head = push ( head , 6 ) ; head = push ( head , 9 ) ; head = rearrange ( head ) ; display ( head ) ; } }
class Node { int data ; Node next ; Node ( int key ) { data = key ; next = null ; } }
class GFG { Node left = null ;
void printlist ( Node head ) { while ( head != null ) { System . out . print ( head . data + " ▁ " ) ; if ( head . next != null ) { System . out . print ( " - > " ) ; } head = head . next ; } System . out . println ( ) ; }
void rearrange ( Node head ) { if ( head != null ) { left = head ; reorderListUtil ( left ) ; } } void reorderListUtil ( Node right ) { if ( right == null ) { return ; } reorderListUtil ( right . next ) ;
if ( left == null ) { return ; }
if ( left != right && left . next != right ) { Node temp = left . next ; left . next = right ; right . next = temp ; left = temp ; }
else { if ( left . next == right ) {
left . next . next = null ; left = null ; } else {
left . next = null ; left = null ; } } }
public static void main ( String [ ] args ) { Node head = new Node ( 1 ) ; head . next = new Node ( 2 ) ; head . next . next = new Node ( 3 ) ; head . next . next . next = new Node ( 4 ) ; head . next . next . next . next = new Node ( 5 ) ; GFG gfg = new GFG ( ) ;
gfg . printlist ( head ) ;
gfg . rearrange ( head ) ;
gfg . printlist ( head ) ; } }
static Node head ; boolean borrow ;
static class Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } }
int getLength ( Node node ) { int size = 0 ; while ( node != null ) { node = node . next ; size ++ ; } return size ; }
Node paddZeros ( Node sNode , int diff ) { if ( sNode == null ) return null ; Node zHead = new Node ( 0 ) ; diff -- ; Node temp = zHead ; while ( ( diff -- ) != 0 ) { temp . next = new Node ( 0 ) ; temp = temp . next ; } temp . next = sNode ; return zHead ; }
Node subtractLinkedListHelper ( Node l1 , Node l2 ) { if ( l1 == null && l2 == null && borrow == false ) return null ; Node previous = subtractLinkedListHelper ( ( l1 != null ) ? l1 . next : null , ( l2 != null ) ? l2 . next : null ) ; int d1 = l1 . data ; int d2 = l2 . data ; int sub = 0 ;
if ( borrow ) { d1 -- ; borrow = false ; }
if ( d1 < d2 ) { borrow = true ; d1 = d1 + 10 ; }
sub = d1 - d2 ;
Node current = new Node ( sub ) ;
current . next = previous ; return current ; }
Node subtractLinkedList ( Node l1 , Node l2 ) {
if ( l1 == null && l2 == null ) return null ;
int len1 = getLength ( l1 ) ; int len2 = getLength ( l2 ) ; Node lNode = null , sNode = null ; Node temp1 = l1 ; Node temp2 = l2 ;
if ( len1 != len2 ) { lNode = len1 > len2 ? l1 : l2 ; sNode = len1 > len2 ? l2 : l1 ; sNode = paddZeros ( sNode , Math . abs ( len1 - len2 ) ) ; } else {
while ( l1 != null && l2 != null ) { if ( l1 . data != l2 . data ) { lNode = l1 . data > l2 . data ? temp1 : temp2 ; sNode = l1 . data > l2 . data ? temp2 : temp1 ; break ; } l1 = l1 . next ; l2 = l2 . next ; } }
borrow = false ; return subtractLinkedListHelper ( lNode , sNode ) ; }
static void printList ( Node head ) { Node temp = head ; while ( temp != null ) { System . out . print ( temp . data + " ▁ " ) ; temp = temp . next ; } }
public static void main ( String [ ] args ) { Node head = new Node ( 1 ) ; head . next = new Node ( 0 ) ; head . next . next = new Node ( 0 ) ; Node head2 = new Node ( 1 ) ; LinkedList ob = new LinkedList ( ) ; Node result = ob . subtractLinkedList ( head , head2 ) ; printList ( result ) ; } }
static Node head ;
static class Node { int data ; Node next ;
Node ( int d ) { data = d ; next = null ; } }
static void insertAtMid ( int x ) {
if ( head == null ) head = new Node ( x ) ; else {
Node newNode = new Node ( x ) ; Node ptr = head ; int len = 0 ;
while ( ptr != null ) { len ++ ; ptr = ptr . next ; }
int count = ( ( len % 2 ) == 0 ) ? ( len / 2 ) : ( len + 1 ) / 2 ; ptr = head ;
while ( count -- > 1 ) ptr = ptr . next ;
newNode . next = ptr . next ; ptr . next = newNode ; } }
static void display ( ) { Node temp = head ; while ( temp != null ) { System . out . print ( temp . data + " ▁ " ) ; temp = temp . next ; } }
head = null ; head = new Node ( 1 ) ; head . next = new Node ( 2 ) ; head . next . next = new Node ( 4 ) ; head . next . next . next = new Node ( 5 ) ; System . out . println ( " Linked ▁ list ▁ before ▁ " + " insertion : ▁ " ) ; display ( ) ; int x = 3 ; insertAtMid ( x ) ; System . out . println ( " Linked list after " + ▁ " insertion : "); display ( ) ; } }
static Node head ;
static class Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } }
static void insertAtMid ( int x ) {
if ( head == null ) head = new Node ( x ) ; else {
Node newNode = new Node ( x ) ;
Node slow = head ; Node fast = head . next ; while ( fast != null && fast . next != null ) {
slow = slow . next ;
fast = fast . next . next ; }
newNode . next = slow . next ; slow . next = newNode ; } }
static void display ( ) { Node temp = head ; while ( temp != null ) { System . out . print ( temp . data + " ▁ " ) ; temp = temp . next ; } }
head = null ; head = new Node ( 1 ) ; head . next = new Node ( 2 ) ; head . next . next = new Node ( 4 ) ; head . next . next . next = new Node ( 5 ) ; System . out . println ( " Linked ▁ list ▁ before " + " ▁ insertion : ▁ " ) ; display ( ) ; int x = 3 ; insertAtMid ( x ) ; System . out . println ( " Linked list after " + ▁ " insertion : "); display ( ) ; } }
static class Node { int data ; Node prev , next ; } ;
static Node getNode ( int data ) {
Node newNode = new Node ( ) ;
newNode . data = data ; newNode . prev = newNode . next = null ; return newNode ; }
static Node sortedInsert ( Node head_ref , Node newNode ) { Node current ;
if ( head_ref == null ) head_ref = newNode ;
else if ( ( head_ref ) . data >= newNode . data ) { newNode . next = head_ref ; newNode . next . prev = newNode ; head_ref = newNode ; } else { current = head_ref ;
while ( current . next != null && current . next . data < newNode . data ) current = current . next ;
newNode . next = current . next ;
if ( current . next != null ) newNode . next . prev = newNode ; current . next = newNode ; newNode . prev = current ; } return head_ref ; }
static Node insertionSort ( Node head_ref ) {
Node sorted = null ;
Node current = head_ref ; while ( current != null ) {
Node next = current . next ;
current . prev = current . next = null ;
sorted = sortedInsert ( sorted , current ) ;
current = next ; }
head_ref = sorted ; return head_ref ; }
static void printList ( Node head ) { while ( head != null ) { System . out . print ( head . data + " ▁ " ) ; head = head . next ; } }
static Node push ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = ( head_ref ) ; new_node . prev = null ;
if ( ( head_ref ) != null ) ( head_ref ) . prev = new_node ;
( head_ref ) = new_node ; return head_ref ; }
Node head = null ;
head = push ( head , 9 ) ; head = push ( head , 3 ) ; head = push ( head , 5 ) ; head = push ( head , 10 ) ; head = push ( head , 12 ) ; head = push ( head , 8 ) ; System . out . println ( "Doubly Linked List Before SortingNEW_LINE"); printList ( head ) ; head = insertionSort ( head ) ; System . out . println ( " Doubly Linked List After Sorting "); printList ( head ) ; } }
static int arr [ ] = new int [ 10000 ] ;
public static void reverse ( int arr [ ] , int s , int e ) { while ( s < e ) { int tem = arr [ s ] ; arr [ s ] = arr [ e ] ; arr [ e ] = tem ; s = s + 1 ; e = e - 1 ; } }
public static void fun ( int arr [ ] , int k ) { int n = 4 - 1 ; int v = n - k ; if ( v >= 0 ) { reverse ( arr , 0 , v ) ; reverse ( arr , v + 1 , n ) ; reverse ( arr , 0 , n ) ; } }
public static void main ( String args [ ] ) { arr [ 0 ] = 1 ; arr [ 1 ] = 2 ; arr [ 2 ] = 3 ; arr [ 3 ] = 4 ; for ( int i = 0 ; i < 4 ; i ++ ) { fun ( arr , i ) ; System . out . print ( " [ " ) ; for ( int j = 0 ; j < 4 ; j ++ ) { System . out . print ( arr [ j ] + " , ▁ " ) ; } System . out . print ( " ] " ) ; } } }
import java . util . * ; class GFG { static int MAX = 100005 ;
static int [ ] seg = new int [ 4 * MAX ] ;
static void build ( int node , int l , int r , int a [ ] ) { if ( l == r ) seg [ node ] = a [ l ] ; else { int mid = ( l + r ) / 2 ; build ( 2 * node , l , mid , a ) ; build ( 2 * node + 1 , mid + 1 , r , a ) ; seg [ node ] = ( seg [ 2 * node ] seg [ 2 * node + 1 ] ) ; } }
static int query ( int node , int l , int r , int start , int end , int a [ ] ) {
if ( l > end r < start ) return 0 ; if ( start <= l && r <= end ) return seg [ node ] ;
int mid = ( l + r ) / 2 ;
return ( ( query ( 2 * node , l , mid , start , end , a ) ) | ( query ( 2 * node + 1 , mid + 1 , r , start , end , a ) ) ) ; }
static void orsum ( int a [ ] , int n , int q , int k [ ] ) {
build ( 1 , 0 , n - 1 , a ) ;
for ( int j = 0 ; j < q ; j ++ ) {
int i = k [ j ] % ( n / 2 ) ;
int sec = query ( 1 , 0 , n - 1 , n / 2 - i , n - i - 1 , a ) ;
int first = ( query ( 1 , 0 , n - 1 , 0 , n / 2 - 1 - i , a ) | query ( 1 , 0 , n - 1 , n - i , n - 1 , a ) ) ; int temp = sec + first ;
System . out . print ( temp + "NEW_LINE"); } }
public static void main ( String [ ] args ) { int a [ ] = { 7 , 44 , 19 , 86 , 65 , 39 , 75 , 101 } ; int n = a . length ; int q = 2 ; int k [ ] = { 4 , 2 } ; orsum ( a , n , q , k ) ; } }
static void maximumEqual ( int a [ ] , int b [ ] , int n ) {
int store [ ] = new int [ ( int ) 1e5 ] ;
for ( int i = 0 ; i < n ; i ++ ) { store [ b [ i ] ] = i + 1 ; }
int ans [ ] = new int [ ( int ) 1e5 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
int d = Math . abs ( store [ a [ i ] ] - ( i + 1 ) ) ;
if ( store [ a [ i ] ] < i + 1 ) { d = n - d ; }
ans [ d ] ++ ; } int finalans = 0 ;
for ( int i = 0 ; i < 1e5 ; i ++ ) finalans = Math . max ( finalans , ans [ i ] ) ;
System . out . print ( finalans + "NEW_LINE"); }
int A [ ] = { 6 , 7 , 3 , 9 , 5 } ; int B [ ] = { 7 , 3 , 9 , 5 , 6 } ; int size = A . length ;
maximumEqual ( A , B , size ) ; } }
static void RightRotate ( int a [ ] , int n , int k ) {
k = k % n ; for ( int i = 0 ; i < n ; i ++ ) { if ( i < k ) {
System . out . print ( a [ n + i - k ] + " ▁ " ) ; } else {
System . out . print ( a [ i - k ] + " ▁ " ) ; } } System . out . println ( ) ; }
public static void main ( String args [ ] ) { int Array [ ] = { 1 , 2 , 3 , 4 , 5 } ; int N = Array . length ; int K = 2 ; RightRotate ( Array , N , K ) ; } }
static void restoreSortedArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > arr [ i + 1 ] ) {
reverse ( arr , 0 , i ) ; reverse ( arr , i + 1 , n ) ; reverse ( arr , 0 , n ) ; } } } static void reverse ( int [ ] arr , int i , int j ) { int temp ; while ( i < j ) { temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; i ++ ; j -- ; } }
static void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n = arr . length ; restoreSortedArray ( arr , n - 1 ) ; printArray ( arr , n ) ; } }
static int findStartIndexOfArray ( int arr [ ] , int low , int high ) { if ( low > high ) { return - 1 ; } if ( low == high ) { return low ; } int mid = low + ( high - low ) / 2 ; if ( arr [ mid ] > arr [ mid + 1 ] ) { return mid + 1 ; } if ( arr [ mid - 1 ] > arr [ mid ] ) { return mid ; } if ( arr [ low ] > arr [ mid ] ) { return findStartIndexOfArray ( arr , low , mid - 1 ) ; } else { return findStartIndexOfArray ( arr , mid + 1 , high ) ; } }
static void restoreSortedArray ( int arr [ ] , int n ) {
if ( arr [ 0 ] < arr [ n - 1 ] ) { return ; } int start = findStartIndexOfArray ( arr , 0 , n - 1 ) ;
Arrays . sort ( arr , 0 , start ) ; Arrays . sort ( arr , start , n ) ; Arrays . sort ( arr ) ; }
static void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . length ; restoreSortedArray ( arr , n ) ; printArray ( arr , n ) ; } }
static String leftrotate ( String str , int d ) { String ans = str . substring ( d ) + str . substring ( 0 , d ) ; return ans ; }
static String rightrotate ( String str , int d ) { return leftrotate ( str , str . length ( ) - d ) ; }
public static void main ( String args [ ] ) { String str1 = " GeeksforGeeks " ; System . out . println ( leftrotate ( str1 , 2 ) ) ; String str2 = " GeeksforGeeks " ; System . out . println ( rightrotate ( str2 , 2 ) ) ; } }
static class Node { int data ; Node next ; Node prev ; } ;
static Node insertNode ( Node start , int value ) {
if ( start == null ) { Node new_node = new Node ( ) ; new_node . data = value ; new_node . next = new_node . prev = new_node ; start = new_node ; return new_node ; }
Node last = ( start ) . prev ;
Node new_node = new Node ( ) ; new_node . data = value ;
new_node . next = start ;
( start ) . prev = new_node ;
new_node . prev = last ;
last . next = new_node ; return start ; }
static void displayList ( Node start ) { Node temp = start ; while ( temp . next != start ) { System . out . printf ( " % d ▁ " , temp . data ) ; temp = temp . next ; } System . out . printf ( " % d ▁ " , temp . data ) ; }
static int searchList ( Node start , int search ) {
Node temp = start ;
int count = 0 , flag = 0 , value ;
if ( temp == null ) return - 1 ; else {
while ( temp . next != start ) {
count ++ ;
if ( temp . data == search ) { flag = 1 ; count -- ; break ; }
temp = temp . next ; }
if ( temp . data == search ) { count ++ ; flag = 1 ; }
if ( flag == 1 ) System . out . println ( " " + search ▁ + " found at location "+ count); else System . out . println ( " " + search ▁ + " not found "); } return - 1 ; }
Node start = null ;
start = insertNode ( start , 4 ) ;
start = insertNode ( start , 5 ) ;
start = insertNode ( start , 7 ) ;
start = insertNode ( start , 8 ) ;
start = insertNode ( start , 6 ) ; System . out . printf ( " Created ▁ circular ▁ doubly ▁ linked ▁ list ▁ is : ▁ " ) ; displayList ( start ) ; searchList ( start , 5 ) ; } }
static class Node { int data ; Node next , prev ; } ;
static Node getNode ( int data ) { Node newNode = new Node ( ) ; newNode . data = data ; return newNode ; }
static Node insertEnd ( Node head , Node new_node ) {
if ( head == null ) { new_node . next = new_node . prev = new_node ; head = new_node ; return head ; }
Node last = ( head ) . prev ;
new_node . next = head ;
( head ) . prev = new_node ;
new_node . prev = last ;
last . next = new_node ; return head ; }
static Node reverse ( Node head ) { if ( head == null ) return null ;
Node new_head = null ;
Node last = head . prev ;
Node curr = last , prev ;
while ( curr . prev != last ) { prev = curr . prev ;
new_head = insertEnd ( new_head , curr ) ; curr = prev ; } new_head = insertEnd ( new_head , curr ) ;
return new_head ; }
static void display ( Node head ) { if ( head == null ) return ; Node temp = head ; System . out . print ( " Forward ▁ direction : ▁ " ) ; while ( temp . next != head ) { System . out . print ( temp . data + " ▁ " ) ; temp = temp . next ; } System . out . print ( temp . data + " ▁ " ) ; Node last = head . prev ; temp = last ; System . out . print ( " Backward direction : "); while ( temp . prev != last ) { System . out . print ( temp . data + " ▁ " ) ; temp = temp . prev ; } System . out . print ( temp . data + " ▁ " ) ; }
public static void main ( String args [ ] ) { Node head = null ; head = insertEnd ( head , getNode ( 1 ) ) ; head = insertEnd ( head , getNode ( 2 ) ) ; head = insertEnd ( head , getNode ( 3 ) ) ; head = insertEnd ( head , getNode ( 4 ) ) ; head = insertEnd ( head , getNode ( 5 ) ) ; System . out . print ( "Current list:NEW_LINE"); display ( head ) ; head = reverse ( head ) ; System . out . print ( " Reversed list : "); display ( head ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int MAXN = 1001 ;
static int [ ] depth = new int [ MAXN ] ;
static int [ ] parent = new int [ MAXN ] ; @ SuppressWarnings ( " unchecked " ) static Vector < Integer > [ ] adj = new Vector [ MAXN ] ; static { for ( int i = 0 ; i < MAXN ; i ++ ) adj [ i ] = new Vector < > ( ) ; } static void addEdge ( int u , int v ) { adj [ u ] . add ( v ) ; adj [ v ] . add ( u ) ; } static void dfs ( int cur , int prev ) {
parent [ cur ] = prev ;
depth [ cur ] = depth [ prev ] + 1 ;
for ( int i = 0 ; i < adj [ cur ] . size ( ) ; i ++ ) if ( adj [ cur ] . elementAt ( i ) != prev ) dfs ( adj [ cur ] . elementAt ( i ) , cur ) ; } static void preprocess ( ) {
depth [ 0 ] = - 1 ;
dfs ( 1 , 0 ) ; }
static int LCANaive ( int u , int v ) { if ( u == v ) return u ; if ( depth [ u ] > depth [ v ] ) { int temp = u ; u = v ; v = temp ; } v = parent [ v ] ; return LCANaive ( u , v ) ; }
public static void main ( String [ ] args ) {
addEdge ( 1 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 1 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 2 , 6 ) ; addEdge ( 3 , 7 ) ; addEdge ( 4 , 8 ) ; addEdge ( 4 , 9 ) ; addEdge ( 9 , 10 ) ; addEdge ( 9 , 11 ) ; addEdge ( 7 , 12 ) ; addEdge ( 7 , 13 ) ; preprocess ( ) ; System . out . println ( " LCA ( 11,8 ) ▁ : ▁ " + LCANaive ( 11 , 8 ) ) ; System . out . println ( " LCA ( 3,13 ) ▁ : ▁ " + LCANaive ( 3 , 13 ) ) ; } }
import java . util . * ; class GFG { static final int MAXN = 1001 ;
static int block_sz ;
static int [ ] depth = new int [ MAXN ] ;
static int [ ] parent = new int [ MAXN ] ;
static int [ ] jump_parent = new int [ MAXN ] ; static Vector < Integer > [ ] adj = new Vector [ MAXN ] ; static void addEdge ( int u , int v ) { adj [ u ] . add ( v ) ; adj [ v ] . add ( u ) ; } static int LCANaive ( int u , int v ) { if ( u == v ) return u ; if ( depth [ u ] > depth [ v ] ) { int t = u ; u = v ; v = t ; } v = parent [ v ] ; return LCANaive ( u , v ) ; }
static void dfs ( int cur , int prev ) {
depth [ cur ] = depth [ prev ] + 1 ;
parent [ cur ] = prev ;
if ( depth [ cur ] % block_sz == 0 )
jump_parent [ cur ] = parent [ cur ] ; else
jump_parent [ cur ] = jump_parent [ prev ] ;
for ( int i = 0 ; i < adj [ cur ] . size ( ) ; ++ i ) if ( adj [ cur ] . get ( i ) != prev ) dfs ( adj [ cur ] . get ( i ) , cur ) ; }
static int LCASQRT ( int u , int v ) { while ( jump_parent [ u ] != jump_parent [ v ] ) { if ( depth [ u ] > depth [ v ] ) {
int t = u ; u = v ; v = t ; }
v = jump_parent [ v ] ; }
return LCANaive ( u , v ) ; } static void preprocess ( int height ) { block_sz = ( int ) Math . sqrt ( height ) ; depth [ 0 ] = - 1 ;
dfs ( 1 , 0 ) ; }
public static void main ( String [ ] args ) { for ( int i = 0 ; i < adj . length ; i ++ ) adj [ i ] = new Vector < Integer > ( ) ;
addEdge ( 1 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 1 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 2 , 6 ) ; addEdge ( 3 , 7 ) ; addEdge ( 4 , 8 ) ; addEdge ( 4 , 9 ) ; addEdge ( 9 , 10 ) ; addEdge ( 9 , 11 ) ; addEdge ( 7 , 12 ) ; addEdge ( 7 , 13 ) ;
int height = 4 ; preprocess ( height ) ; System . out . print ( " LCA ( 11,8 ) ▁ : ▁ " + LCASQRT ( 11 , 8 ) + "NEW_LINE"); System . out . print ( " LCA ( 3,13 ) ▁ : ▁ " + LCASQRT ( 3 , 13 ) + "NEW_LINE"); } }
public static void main ( String [ ] args ) { int N = 3 ;
System . out . print ( Math . pow ( 2 , N + 1 ) - 2 ) ; } }
class GFG { static int countOfNum ( int n , int a , int b ) { int cnt_of_a , cnt_of_b , cnt_of_ab , sum ;
cnt_of_a = n / a ;
cnt_of_b = n / b ;
sum = cnt_of_b + cnt_of_a ;
cnt_of_ab = n / ( a * b ) ;
sum = sum - cnt_of_ab ; return sum ; }
static int sumOfNum ( int n , int a , int b ) { int i ; int sum = 0 ;
Set < Integer > ans = new HashSet < Integer > ( ) ;
for ( i = a ; i <= n ; i = i + a ) { ans . add ( i ) ; }
for ( i = b ; i <= n ; i = i + b ) { ans . add ( i ) ; }
for ( Integer it : ans ) { sum = sum + it ; } return sum ; }
public static void main ( String [ ] args ) { int N = 88 ; int A = 11 ; int B = 8 ; int count = countOfNum ( N , A , B ) ; int sumofnum = sumOfNum ( N , A , B ) ; System . out . print ( sumofnum % count ) ; } }
static double get ( double L , double R ) {
double x = 1.0 / L ;
double y = 1.0 / ( R + 1.0 ) ; return ( x - y ) ; }
public static void main ( String [ ] args ) { int L = 6 , R = 12 ;
double ans = get ( L , R ) ; System . out . printf ( " % .2f " , ans ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int MAX = 100000 ;
static ArrayList < Integer > v = new ArrayList < Integer > ( ) ; public static int upper_bound ( ArrayList < Integer > ar , int k ) { int s = 0 ; int e = ar . size ( ) ; while ( s != e ) { int mid = s + e >> 1 ; if ( ar . get ( mid ) <= k ) { s = mid + 1 ; } else { e = mid ; } } if ( s == ar . size ( ) ) { return - 1 ; } return s ; }
static int consecutiveOnes ( int x ) {
int p = 0 ; while ( x > 0 ) {
if ( x % 2 == 1 && p == 1 ) { return 1 ; }
p = x % 2 ;
x /= 2 ; } return 0 ; }
static void preCompute ( ) {
for ( int i = 0 ; i <= MAX ; i ++ ) { if ( consecutiveOnes ( i ) == 0 ) { v . add ( i ) ; } } }
static int nextValid ( int n ) {
int it = upper_bound ( v , n ) ; int val = v . get ( it ) ; return val ; }
static void performQueries ( int queries [ ] , int q ) { for ( int i = 0 ; i < q ; i ++ ) { System . out . println ( nextValid ( queries [ i ] ) ) ; } }
public static void main ( String [ ] args ) { int queries [ ] = { 4 , 6 } ; int q = queries . length ;
preCompute ( ) ;
performQueries ( queries , q ) ; } }
static int changeToOnes ( String str ) {
int i , l , ctr = 0 ; l = str . length ( ) ;
for ( i = l - 1 ; i >= 0 ; i -- ) {
if ( str . charAt ( i ) == '1' ) ctr ++ ;
else break ; }
return l - ctr ; }
static String removeZeroesFromFront ( String str ) { String s ; int i = 0 ;
while ( i < str . length ( ) && str . charAt ( i ) == '0' ) i ++ ;
if ( i == str . length ( ) ) s = "0" ;
else s = str . substring ( i , str . length ( ) - i ) ; return s ; }
public static void main ( String [ ] args ) { String str = "10010111" ;
str = removeZeroesFromFront ( str ) ; System . out . println ( changeToOnes ( str ) ) ; } }
static int MinDeletion ( int a [ ] , int n ) {
Map < Integer , Integer > mp = new HashMap < > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( mp . containsKey ( a [ i ] ) ) { mp . put ( a [ i ] , mp . get ( a [ i ] ) + 1 ) ; } else { mp . put ( a [ i ] , 1 ) ; } }
int ans = 0 ; for ( Map . Entry < Integer , Integer > i : mp . entrySet ( ) ) {
int x = i . getKey ( ) ;
int frequency = i . getValue ( ) ;
if ( x <= frequency ) {
ans += ( frequency - x ) ; }
else ans += frequency ; } return ans ; }
public static void main ( String [ ] args ) { int a [ ] = { 2 , 3 , 2 , 3 , 4 , 4 , 4 , 4 , 5 } ; int n = a . length ; System . out . println ( MinDeletion ( a , n ) ) ; } }
static int maxCountAB ( String s [ ] , int n ) {
int A = 0 , B = 0 , BA = 0 , ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) { String S = s [ i ] ; int L = S . length ( ) ; for ( int j = 0 ; j < L - 1 ; j ++ ) {
if ( S . charAt ( j ) == ' A ' && S . charAt ( j + 1 ) == ' B ' ) { ans ++ ; } }
if ( S . charAt ( 0 ) == ' B ' && S . charAt ( L - 1 ) == ' A ' ) BA ++ ;
else if ( S . charAt ( 0 ) == ' B ' ) B ++ ;
else if ( S . charAt ( L - 1 ) == ' A ' ) A ++ ; }
if ( BA == 0 ) ans += Math . min ( B , A ) ; else if ( A + B == 0 ) ans += BA - 1 ; else ans += BA + Math . min ( B , A ) ; return ans ; }
public static void main ( String [ ] args ) { String s [ ] = { " ABCA " , " BOOK " , " BAND " } ; int n = s . length ; System . out . println ( maxCountAB ( s , n ) ) ; } }
static int MinOperations ( int n , int x , int [ ] arr ) {
int total = 0 ; for ( int i = 0 ; i < n ; ++ i ) {
if ( arr [ i ] > x ) { int difference = arr [ i ] - x ; total = total + difference ; arr [ i ] = x ; } }
for ( int i = 1 ; i < n ; ++ i ) { int LeftNeigbouringSum = arr [ i ] + arr [ i - 1 ] ;
if ( LeftNeigbouringSum > x ) { int current_diff = LeftNeigbouringSum - x ; arr [ i ] = Math . max ( 0 , arr [ i ] - current_diff ) ; total = total + current_diff ; } } return total ; }
public static void main ( String args [ ] ) { int X = 1 ; int arr [ ] = { 1 , 6 , 1 , 2 , 0 , 4 } ; int N = arr . length ; System . out . println ( MinOperations ( N , X , arr ) ) ; } }
static void findNumbers ( int arr [ ] , int n ) {
int sumN = ( n * ( n + 1 ) ) / 2 ;
int sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
int sum = 0 , sumSq = 0 , i ; for ( i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq += Math . pow ( arr [ i ] , 2 ) ; } int B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; int A = sum - sumN + B ; System . out . println ( " A ▁ = ▁ " + A + " B = " + B); }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 2 , 3 , 4 } ; int n = arr . length ; findNumbers ( arr , n ) ; } }
static boolean is_prefix ( String temp , String str ) {
if ( temp . length ( ) < str . length ( ) ) return false ; else {
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str . charAt ( i ) != temp . charAt ( i ) ) return false ; } return true ; } }
static String lexicographicallyString ( String [ ] input , int n , String str ) {
Arrays . sort ( input ) ; for ( int i = 0 ; i < n ; i ++ ) { String temp = input [ i ] ;
if ( is_prefix ( temp , str ) ) { return temp ; } }
return " - 1" ; }
public static void main ( String args [ ] ) { String [ ] arr = { " apple " , " appe " , " apl " , " aapl " , " appax " } ; String S = " app " ; int N = 5 ; System . out . println ( lexicographicallyString ( arr , N , S ) ) ; } }
static void Rearrange ( int arr [ ] , int K , int N ) {
int ans [ ] = new int [ N + 1 ] ;
int f = - 1 ; for ( int i = 0 ; i < N ; i ++ ) { ans [ i ] = - 1 ; }
for ( int i = 0 ; i < arr . length ; i ++ ) { if ( arr [ i ] == K ) { K = i ; break ; } }
Vector < Integer > smaller = new Vector < Integer > ( ) ; Vector < Integer > greater = new Vector < Integer > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] < arr [ K ] ) smaller . add ( arr [ i ] ) ;
else if ( arr [ i ] > arr [ K ] ) greater . add ( arr [ i ] ) ; } int low = 0 , high = N - 1 ;
while ( low <= high ) {
int mid = ( low + high ) / 2 ;
if ( mid == K ) { ans [ mid ] = arr [ K ] ; f = 1 ; break ; }
else if ( mid < K ) { if ( smaller . size ( ) == 0 ) { break ; } ans [ mid ] = smaller . lastElement ( ) ; smaller . remove ( smaller . size ( ) - 1 ) ; low = mid + 1 ; }
else { if ( greater . size ( ) == 0 ) { break ; } ans [ mid ] = greater . lastElement ( ) ; greater . remove ( greater . size ( ) - 1 ) ; high = mid - 1 ; } }
if ( f == - 1 ) { System . out . println ( - 1 ) ; return ; }
for ( int i = 0 ; i < N ; i ++ ) {
if ( ans [ i ] == - 1 ) { if ( smaller . size ( ) > 0 ) { ans [ i ] = smaller . lastElement ( ) ; smaller . remove ( smaller . size ( ) - 1 ) ; } else if ( greater . size ( ) > 0 ) { ans [ i ] = greater . lastElement ( ) ; greater . remove ( greater . size ( ) - 1 ) ; } } }
for ( int i = 0 ; i < N ; i ++ ) System . out . print ( ans [ i ] + " ▁ " ) ; System . out . println ( ) ; }
int arr [ ] = { 10 , 7 , 2 , 5 , 3 , 8 } ; int K = 7 ; int N = arr . length ;
Rearrange ( arr , K , N ) ; } }
static void minimumK ( ArrayList < Integer > arr , int M , int N ) {
int good = ( int ) ( ( N * 1.0 ) / ( ( M + 1 ) * 1.0 ) ) + 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { int K = i ;
int candies = N ;
int taken = 0 ; while ( candies > 0 ) {
taken += Math . min ( K , candies ) ; candies -= Math . min ( K , candies ) ;
for ( int j = 0 ; j < M ; j ++ ) {
int consume = ( arr . get ( j ) * candies ) / 100 ;
candies -= consume ; } }
if ( taken >= good ) { System . out . print ( i ) ; return ; } } }
public static void main ( String [ ] args ) { int N = 13 , M = 1 ; ArrayList < Integer > arr = new ArrayList < Integer > ( ) ; arr . add ( 50 ) ; minimumK ( arr , M , N ) ; } }
static void calcTotalTime ( String path ) {
int time = 0 ;
int x = 0 , y = 0 ;
Set < String > s = new HashSet < > ( ) ; for ( int i = 0 ; i < path . length ( ) ; i ++ ) { int p = x ; int q = y ; if ( path . charAt ( i ) == ' N ' ) y ++ ; else if ( path . charAt ( i ) == ' S ' ) y -- ; else if ( path . charAt ( i ) == ' E ' ) x ++ ; else if ( path . charAt ( i ) == ' W ' ) x -- ;
String o = ( p + x ) + " ▁ " + ( q + y ) ; if ( ! s . contains ( o ) ) {
time += 2 ;
s . add ( o ) ; } else time += 1 ; }
System . out . println ( time ) ; }
public static void main ( String [ ] args ) { String path = " NSE " ; calcTotalTime ( path ) ; } }
static int findCost ( int [ ] A , int N ) {
int totalCost = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( A [ i ] == 0 ) {
A [ i ] = 1 ;
totalCost += i ; } }
return totalCost ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 0 , 1 , 0 , 1 , 0 } ; int N = arr . length ; System . out . println ( findCost ( arr , N ) ) ; } }
public static int peakIndex ( int [ ] arr ) { int N = arr . length ;
if ( arr . length < 3 ) return - 1 ; int i = 0 ;
while ( i + 1 < N ) {
if ( arr [ i + 1 ] < arr [ i ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; } if ( i == 0 i == N - 1 ) return - 1 ;
int ans = i ;
while ( i < N - 1 ) {
if ( arr [ i ] < arr [ i + 1 ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; }
if ( i == N - 1 ) return ans ;
return - 1 ; }
public static void main ( String [ ] args ) { int [ ] arr = { 0 , 1 , 0 } ; System . out . println ( peakIndex ( arr ) ) ; } }
static void hasArrayTwoPairs ( int nums [ ] , int n , int target ) {
Arrays . sort ( nums ) ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = target - nums [ i ] ;
int low = 0 , high = n - 1 ; while ( low <= high ) {
int mid = low + ( ( high - low ) / 2 ) ;
if ( nums [ mid ] > x ) { high = mid - 1 ; }
else if ( nums [ mid ] < x ) { low = mid + 1 ; }
else {
if ( mid == i ) { if ( ( mid - 1 >= 0 ) && nums [ mid - 1 ] == x ) { System . out . print ( nums [ i ] + " , ▁ " ) ; System . out . print ( nums [ mid - 1 ] ) ; return ; } if ( ( mid + 1 < n ) && nums [ mid + 1 ] == x ) { System . out . print ( nums [ i ] + " , ▁ " ) ; System . out . print ( nums [ mid + 1 ] ) ; return ; } break ; }
else { System . out . print ( nums [ i ] + " , ▁ " ) ; System . out . print ( nums [ mid ] ) ; return ; } } } }
System . out . print ( - 1 ) ; }
public static void main ( String [ ] args ) { int A [ ] = { 0 , - 1 , 2 , - 3 , 1 } ; int X = - 2 ; int N = A . length ;
hasArrayTwoPairs ( A , N , X ) ; } }
static void findClosest ( int N , int target ) { int closest = - 1 ; int diff = Integer . MAX_VALUE ;
for ( int i = 1 ; i <= ( int ) Math . sqrt ( N ) ; i ++ ) { if ( N % i == 0 ) {
if ( N / i == i ) {
if ( Math . abs ( target - i ) < diff ) { diff = Math . abs ( target - i ) ; closest = i ; } } else {
if ( Math . abs ( target - i ) < diff ) { diff = Math . abs ( target - i ) ; closest = i ; }
if ( Math . abs ( target - N / i ) < diff ) { diff = Math . abs ( target - N / i ) ; closest = N / i ; } } } }
System . out . println ( closest ) ; }
int N = 16 , X = 5 ;
findClosest ( N , X ) ; } }
static int power ( int A , int N ) {
int count = 0 ; if ( A == 1 ) return 0 ; while ( N > 0 ) {
count ++ ;
N /= A ; } return count ; }
static void Pairs ( int N , int A , int B ) { int powerA , powerB ;
powerA = power ( A , N ) ;
powerB = power ( B , N ) ;
int intialB = B , intialA = A ;
A = 1 ; for ( int i = 0 ; i <= powerA ; i ++ ) { B = 1 ; for ( int j = 0 ; j <= powerB ; j ++ ) {
if ( B == N - A ) { System . out . println ( i + " ▁ " + j ) ; return ; }
B *= intialB ; }
A *= intialA ; }
System . out . println ( " - 1" ) ; return ; }
int N = 106 , A = 3 , B = 5 ;
Pairs ( N , A , B ) ; } }
public static int findNonMultiples ( int [ ] arr , int n , int k ) {
Set < Integer > multiples = new HashSet < Integer > ( ) ;
for ( int i = 0 ; i < n ; ++ i ) {
if ( ! multiples . contains ( arr [ i ] ) ) {
for ( int j = 1 ; j <= k / arr [ i ] ; j ++ ) { multiples . add ( arr [ i ] * j ) ; } } }
return k - multiples . size ( ) ; }
public static int countValues ( int [ ] arr , int N , int L , int R ) {
return findNonMultiples ( arr , N , R ) - findNonMultiples ( arr , N , L - 1 ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 4 , 5 , 6 } ; int N = arr . length ; int L = 1 ; int R = 20 ;
System . out . println ( countValues ( arr , N , L , R ) ) ; } }
static void minCollectingSpeed ( int [ ] piles , int H ) {
int ans = - 1 ; int low = 1 , high ;
high = Arrays . stream ( piles ) . max ( ) . getAsInt ( ) ;
while ( low <= high ) {
int K = low + ( high - low ) / 2 ; int time = 0 ;
for ( int ai : piles ) { time += ( ai + K - 1 ) / K ; }
if ( time <= H ) { ans = K ; high = K - 1 ; }
else { low = K + 1 ; } }
System . out . print ( ans ) ; }
static public void main ( String args [ ] ) { int [ ] arr = { 3 , 6 , 7 , 11 } ; int H = 8 ;
minCollectingSpeed ( arr , H ) ; } }
static int cntDisPairs ( int arr [ ] , int N , int K ) {
int cntPairs = 0 ;
Arrays . sort ( arr ) ;
int i = 0 ;
int j = N - 1 ;
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
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 6 , 5 , 7 , 7 , 8 } ; int N = arr . length ; int K = 13 ; System . out . print ( cntDisPairs ( arr , N , K ) ) ; }
static void longestSubsequence ( int N , int Q , int arr [ ] , int Queries [ ] [ ] ) { for ( int i = 0 ; i < Q ; i ++ ) {
int x = Queries [ i ] [ 0 ] ; int y = Queries [ i ] [ 1 ] ;
arr [ x - 1 ] = y ;
int count = 1 ; for ( int j = 1 ; j < N ; j ++ ) {
if ( arr [ j ] != arr [ j - 1 ] ) { count += 1 ; } }
System . out . print ( count + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 2 , 5 , 2 } ; int N = arr . length ; int Q = 2 ; int Queries [ ] [ ] = { { 1 , 3 } , { 4 , 2 } } ;
longestSubsequence ( N , Q , arr , Queries ) ; } }
import java . util . * ; class GFG { static void longestSubsequence ( int N , int Q , int arr [ ] , int Queries [ ] [ ] ) { int count = 1 ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] != arr [ i - 1 ] ) { count += 1 ; } }
for ( int i = 0 ; i < Q ; i ++ ) {
int x = Queries [ i ] [ 0 ] ; int y = Queries [ i ] [ 1 ] ;
if ( x > 1 ) {
if ( arr [ x - 1 ] != arr [ x - 2 ] ) { count -= 1 ; }
if ( arr [ x - 2 ] != y ) { count += 1 ; } }
if ( x < N ) {
if ( arr [ x ] != arr [ x - 1 ] ) { count -= 1 ; }
if ( y != arr [ x ] ) { count += 1 ; } } System . out . print ( count + " ▁ " ) ;
arr [ x - 1 ] = y ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 1 , 2 , 5 , 2 } ; int N = arr . length ; int Q = 2 ; int Queries [ ] [ ] = { { 1 , 3 } , { 4 , 2 } } ;
longestSubsequence ( N , Q , arr , Queries ) ; } }
static void sum ( int arr [ ] , int n ) {
HashMap < Integer , Vector < Integer > > mp = new HashMap < > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { Vector < Integer > v = new Vector < > ( ) ; v . add ( i ) ; if ( mp . containsKey ( arr [ i ] ) ) v . addAll ( mp . get ( arr [ i ] ) ) ; mp . put ( arr [ i ] , v ) ; }
int [ ] ans = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) {
int sum = 0 ;
for ( int it : mp . get ( arr [ i ] ) ) {
sum += Math . abs ( it - i ) ; }
ans [ i ] = sum ; }
for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( ans [ i ] + " ▁ " ) ; } return ; }
int arr [ ] = { 1 , 3 , 1 , 1 , 2 } ;
int n = arr . length ;
sum ( arr , n ) ; } }
static void conVowUpp ( char [ ] str ) {
int N = str . length ; for ( int i = 0 ; i < N ; i ++ ) { if ( str [ i ] == ' a ' str [ i ] == ' e ' str [ i ] == ' i ' str [ i ] == ' o ' str [ i ] == ' u ' ) { char c = Character . toUpperCase ( str [ i ] ) ; str [ i ] = c ; } } for ( char c : str ) System . out . print ( c ) ; }
public static void main ( String [ ] args ) { String str = " eutopia " ; conVowUpp ( str . toCharArray ( ) ) ; } }
static HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ; static int N , P ;
static boolean helper ( int mid ) { int cnt = 0 ; for ( Map . Entry < Integer , Integer > i : mp . entrySet ( ) ) { int temp = i . getValue ( ) ; while ( temp >= mid ) { temp -= mid ; cnt ++ ; } }
return cnt >= N ; }
static int findMaximumDays ( int arr [ ] ) {
for ( int i = 0 ; i < P ; i ++ ) { if ( mp . containsKey ( arr [ i ] ) ) { mp . put ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . put ( arr [ i ] , 1 ) ; } }
int start = 0 , end = P , ans = 0 ; while ( start <= end ) {
int mid = start + ( ( end - start ) / 2 ) ;
if ( mid != 0 && helper ( mid ) ) { ans = mid ;
start = mid + 1 ; } else if ( mid == 0 ) { start = mid + 1 ; } else { end = mid - 1 ; } } return ans ; }
public static void main ( String [ ] args ) { N = 3 ; P = 10 ; int arr [ ] = { 1 , 2 , 2 , 1 , 1 , 3 , 3 , 3 , 2 , 4 } ;
System . out . print ( findMaximumDays ( arr ) ) ; } }
static void countSubarrays ( int a [ ] , int n , int k ) {
int ans = 0 ;
ArrayList < Integer > pref = new ArrayList < > ( ) ; pref . add ( 0 ) ;
for ( int i = 0 ; i < n ; i ++ ) pref . add ( ( a [ i ] + pref . get ( i ) ) % k ) ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) {
if ( ( pref . get ( j ) - pref . get ( i - 1 ) + k ) % k == j - i + 1 ) { ans ++ ; } } }
System . out . println ( ans ) ; }
int arr [ ] = { 2 , 3 , 5 , 3 , 1 , 5 } ;
int N = arr . length ;
int K = 4 ;
countSubarrays ( arr , N , K ) ; } }
static boolean check ( String s , int k ) { int n = s . length ( ) ;
for ( int i = 0 ; i < k ; i ++ ) { for ( int j = i ; j < n ; j += k ) {
if ( s . charAt ( i ) != s . charAt ( j ) ) return false ; } } int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( s . charAt ( i ) == '0' )
c ++ ;
else
c -- ; }
if ( c == 0 ) return true ; else return false ; }
public static void main ( String [ ] args ) { String s = "101010" ; int k = 2 ; if ( check ( s , k ) ) System . out . print ( " Yes " + "NEW_LINE"); else System . out . print ( " No " + "NEW_LINE"); } }
static boolean isSame ( String str , int n ) {
HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( mp . containsKey ( str . charAt ( i ) - ' a ' ) ) { mp . put ( str . charAt ( i ) - ' a ' , mp . get ( str . charAt ( i ) - ' a ' ) + 1 ) ; } else { mp . put ( str . charAt ( i ) - ' a ' , 1 ) ; } } for ( Map . Entry < Integer , Integer > it : mp . entrySet ( ) ) {
if ( ( it . getValue ( ) ) >= n ) { return true ; } }
return false ; }
public static void main ( String [ ] args ) { String str = " ccabcba " ; int n = 4 ;
if ( isSame ( str , n ) ) { System . out . print ( " Yes " ) ; } else { System . out . print ( " No " ) ; } } }
import java . util . * ; import java . lang . * ; class GFG { static final double eps = 1e-6 ;
static double func ( double a , double b , double c , double x ) { return a * x * x + b * x + c ; }
static double findRoot ( double a , double b , double c , double low , double high ) { double x = - 1 ;
while ( Math . abs ( high - low ) > eps ) {
x = ( low + high ) / 2 ;
if ( func ( a , b , c , low ) * func ( a , b , c , x ) <= 0 ) { high = x ; }
else { low = x ; } }
return x ; }
static void solve ( double a , double b , double c , double A , double B ) {
if ( func ( a , b , c , A ) * func ( a , b , c , B ) > 0 ) { System . out . println ( " No ▁ solution " ) ; }
else { System . out . format ( " % .4f " , findRoot ( a , b , c , A , B ) ) ; } }
double a = 2 , b = - 3 , c = - 2 , A = 0 , B = 3 ;
solve ( a , b , c , A , B ) ; } }
static boolean possible ( long mid , int [ ] a ) {
long n = a . length ;
long total = ( n * ( n - 1 ) ) / 2 ;
long need = ( total + 1 ) / 2 ; long count = 0 ; long start = 0 , end = 1 ;
while ( end < n ) { if ( a [ ( int ) end ] - a [ ( int ) start ] <= mid ) { end ++ ; } else { count += ( end - start - 1 ) ; start ++ ; } }
if ( end == n && start < end && a [ ( int ) end - 1 ] - a [ ( int ) start ] <= mid ) { long t = end - start - 1 ; count += ( t * ( t + 1 ) / 2 ) ; }
if ( count >= need ) return true ; else return false ; }
static long findMedian ( int [ ] a ) {
long n = a . length ;
long low = 0 , high = a [ ( int ) n - 1 ] - a [ 0 ] ;
while ( low <= high ) {
long mid = ( low + high ) / 2 ;
if ( possible ( mid , a ) ) high = mid - 1 ; else low = mid + 1 ; }
return high + 1 ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 7 , 5 , 2 } ; Arrays . sort ( a ) ; System . out . println ( findMedian ( a ) ) ; } }
static void UniversalSubset ( List < String > A , List < String > B ) {
int n1 = A . size ( ) ; int n2 = B . size ( ) ;
List < String > res = new ArrayList < > ( ) ;
int [ ] [ ] A_fre = new int [ n1 ] [ 26 ] ; for ( int i = 0 ; i < n1 ; i ++ ) { for ( int j = 0 ; j < 26 ; j ++ ) A_fre [ i ] [ j ] = 0 ; }
for ( int i = 0 ; i < n1 ; i ++ ) { for ( int j = 0 ; j < A . get ( i ) . length ( ) ; j ++ ) { A_fre [ i ] [ A . get ( i ) . charAt ( j ) - ' a ' ] ++ ; } }
int [ ] B_fre = new int [ 26 ] ; for ( int i = 0 ; i < n2 ; i ++ ) { int [ ] arr = new int [ 26 ] ; for ( int j = 0 ; j < B . get ( i ) . length ( ) ; j ++ ) { arr [ B . get ( i ) . charAt ( j ) - ' a ' ] ++ ; B_fre [ B . get ( i ) . charAt ( j ) - ' a ' ] = Math . max ( B_fre [ B . get ( i ) . charAt ( j ) - ' a ' ] , arr [ B . get ( i ) . charAt ( j ) - ' a ' ] ) ; } } for ( int i = 0 ; i < n1 ; i ++ ) { int flag = 0 ; for ( int j = 0 ; j < 26 ; j ++ ) {
if ( A_fre [ i ] [ j ] < B_fre [ j ] ) {
flag = 1 ; break ; } }
if ( flag == 0 )
res . add ( A . get ( i ) ) ; }
if ( res . size ( ) != 0 ) {
for ( int i = 0 ; i < res . size ( ) ; i ++ ) { for ( int j = 0 ; j < res . get ( i ) . length ( ) ; j ++ ) System . out . print ( res . get ( i ) . charAt ( j ) ) ; } System . out . print ( " ▁ " ) ; }
else System . out . print ( " - 1" ) ; }
public static void main ( String [ ] args ) { List < String > A = Arrays . asList ( " geeksforgeeks " , " topcoder " , " leetcode " ) ; List < String > B = Arrays . asList ( " geek " , " ee " ) ; UniversalSubset ( A , B ) ; } }
public static void findPair ( int a [ ] , int n ) {
int min_dist = Integer . MAX_VALUE ; int index_a = - 1 , index_b = - 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i + 1 ; j < n ; j ++ ) {
if ( j - i < min_dist ) {
if ( a [ i ] % a [ j ] == 0 a [ j ] % a [ i ] == 0 ) {
min_dist = j - i ;
index_a = i ; index_b = j ; } } } }
if ( index_a == - 1 ) { System . out . println ( " - 1" ) ; }
else { System . out . print ( " ( " + a [ index_a ] + " , ▁ " + a [ index_b ] + " ) " ) ; } }
int a [ ] = { 2 , 3 , 4 , 5 , 6 } ; int n = a . length ;
findPair ( a , n ) ; } }
static void printNum ( int L , int R ) {
for ( int i = L ; i <= R ; i ++ ) { int temp = i ; int c = 10 ; int flag = 0 ;
while ( temp > 0 ) {
if ( temp % 10 >= c ) { flag = 1 ; break ; } c = temp % 10 ; temp /= 10 ; }
if ( flag == 0 ) System . out . print ( i + " ▁ " ) ; } }
int L = 10 , R = 15 ;
printNum ( L , R ) ; } }
static int findMissing ( int arr [ ] , int left , int right , int diff ) {
if ( right <= left ) return 0 ;
int mid = left + ( right - left ) / 2 ;
if ( arr [ mid + 1 ] - arr [ mid ] != diff ) return ( arr [ mid ] + diff ) ;
if ( mid > 0 && arr [ mid ] - arr [ mid - 1 ] != diff ) return ( arr [ mid - 1 ] + diff ) ;
if ( arr [ mid ] == arr [ 0 ] + mid * diff ) return findMissing ( arr , mid + 1 , right , diff ) ;
return findMissing ( arr , left , mid - 1 , diff ) ; }
static int missingElement ( int arr [ ] , int n ) {
Arrays . sort ( arr ) ;
int diff = ( arr [ n - 1 ] - arr [ 0 ] ) / n ;
return findMissing ( arr , 0 , n - 1 , diff ) ; }
int arr [ ] = new int [ ] { 2 , 8 , 6 , 10 } ; int n = arr . length ;
System . out . println ( missingElement ( arr , n ) ) ; } }
static int power ( int x , int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
static int nthRootSearch ( int low , int high , int N , int K ) {
if ( low <= high ) {
int mid = ( low + high ) / 2 ;
if ( ( power ( mid , K ) <= N ) && ( power ( mid + 1 , K ) > N ) ) { return mid ; }
else if ( power ( mid , K ) < N ) { return nthRootSearch ( mid + 1 , high , N , K ) ; } else { return nthRootSearch ( low , mid - 1 , N , K ) ; } } return low ; }
int N = 16 , K = 4 ;
System . out . println ( nthRootSearch ( 0 , N , N , K ) ) ; } }
static int get_subset_count ( int arr [ ] , int K , int N ) {
Arrays . sort ( arr ) ; int left , right ; left = 0 ; right = N - 1 ;
int ans = 0 ; while ( left <= right ) { if ( arr [ left ] + arr [ right ] < K ) {
ans += 1 << ( right - left ) ; left ++ ; } else {
right -- ; } } return ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 5 , 7 } ; int K = 8 ; int N = arr . length ; System . out . print ( get_subset_count ( arr , K , N ) ) ; } }
import java . util . * ; class GFG { static int minMaxDiff ( int arr [ ] , int n , int k ) { int max_adj_dif = Integer . MIN_VALUE ;
for ( int i = 0 ; i < n - 1 ; i ++ ) max_adj_dif = Math . max ( max_adj_dif , Math . abs ( arr [ i ] - arr [ i + 1 ] ) ) ;
if ( max_adj_dif == 0 ) return 0 ;
int best = 1 ; int worst = max_adj_dif ; int mid , required ; while ( best < worst ) { mid = ( best + worst ) / 2 ;
required = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) { required += ( Math . abs ( arr [ i ] - arr [ i + 1 ] ) - 1 ) / mid ; }
if ( required > k ) best = mid + 1 ;
else worst = mid ; } return worst ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 12 , 25 , 50 } ; int n = arr . length ; int k = 7 ; System . out . println ( minMaxDiff ( arr , n , k ) ) ; } }
static void checkMin ( int arr [ ] , int len ) {
int smallest = Integer . MAX_VALUE ; int secondSmallest = Integer . MAX_VALUE ; for ( int i = 0 ; i < len ; i ++ ) {
if ( arr [ i ] < smallest ) { secondSmallest = smallest ; smallest = arr [ i ] ; }
else if ( arr [ i ] < secondSmallest ) { secondSmallest = arr [ i ] ; } } if ( 2 * smallest <= secondSmallest ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 3 , 4 , 5 } ; int len = arr . length ; checkMin ( arr , len ) ; } }
static void createHash ( HashSet < Integer > hash , int maxElement ) {
int prev = 0 , curr = 1 ; hash . add ( prev ) ; hash . add ( curr ) ; while ( curr <= maxElement ) {
int temp = curr + prev ; hash . add ( temp ) ;
prev = curr ; curr = temp ; } }
static void fibonacci ( int arr [ ] , int n ) {
int max_val = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ;
HashSet < Integer > hash = new HashSet < Integer > ( ) ; createHash ( hash , max_val ) ;
int minimum = Integer . MAX_VALUE ; int maximum = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) {
if ( hash . contains ( arr [ i ] ) ) {
minimum = Math . min ( minimum , arr [ i ] ) ; maximum = Math . max ( maximum , arr [ i ] ) ; } } System . out . print ( minimum + " , ▁ " + maximum + "NEW_LINE"); }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n = arr . length ; fibonacci ( arr , n ) ; } }
static boolean isValidLen ( String s , int len , int k ) {
int n = s . length ( ) ;
Map < Character , Integer > mp = new HashMap < Character , Integer > ( ) ; int right = 0 ;
while ( right < len ) { if ( mp . containsKey ( s . charAt ( right ) ) ) { mp . put ( s . charAt ( right ) , mp . get ( s . charAt ( right ) ) + 1 ) ; } else { mp . put ( s . charAt ( right ) , 1 ) ; } right ++ ; } if ( mp . size ( ) <= k ) return true ;
while ( right < n ) {
if ( mp . containsKey ( s . charAt ( right ) ) ) { mp . put ( s . charAt ( right ) , mp . get ( s . charAt ( right ) ) + 1 ) ; } else { mp . put ( s . charAt ( right ) , 1 ) ; }
if ( mp . containsKey ( s . charAt ( right - len ) ) ) { mp . put ( s . charAt ( right - len ) , mp . get ( s . charAt ( right - len ) ) - 1 ) ; }
if ( mp . get ( s . charAt ( right - len ) ) == 0 ) mp . remove ( s . charAt ( right - len ) ) ; if ( mp . size ( ) <= k ) return true ; right ++ ; } return mp . size ( ) <= k ; }
static int maxLenSubStr ( String s , int k ) {
Set < Character > uni = new HashSet < Character > ( ) ; for ( Character x : s . toCharArray ( ) ) uni . add ( x ) ; if ( uni . size ( ) < k ) return - 1 ;
int n = s . length ( ) ;
int lo = - 1 , hi = n + 1 ; while ( hi - lo > 1 ) { int mid = lo + hi >> 1 ; if ( isValidLen ( s , mid , k ) ) lo = mid ; else hi = mid ; } return lo ; }
public static void main ( String [ ] args ) { String s = " aabacbebebe " ; int k = 3 ; System . out . print ( maxLenSubStr ( s , k ) ) ; } }
static boolean isSquarePossible ( int arr [ ] , int n , int l ) {
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] >= l ) cnt ++ ;
if ( cnt >= l ) return true ; } return false ; }
static int maxArea ( int arr [ ] , int n ) { int l = 0 , r = n ; int len = 0 ; while ( l <= r ) { int m = l + ( ( r - l ) / 2 ) ;
if ( isSquarePossible ( arr , n , m ) ) { len = m ; l = m + 1 ; }
else r = m - 1 ; }
return ( len * len ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 4 , 5 , 5 } ; int n = arr . length ; System . out . println ( maxArea ( arr , n ) ) ; } }
static void insertNames ( String arr [ ] , int n ) {
HashSet < String > set = new HashSet < String > ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( ! set . contains ( arr [ i ] ) ) { System . out . print ( "NoNEW_LINE"); set . add ( arr [ i ] ) ; } else { System . out . print ( "YesNEW_LINE"); } } }
public static void main ( String [ ] args ) { String arr [ ] = { " geeks " , " for " , " geeks " } ; int n = arr . length ; insertNames ( arr , n ) ; } }
static int countLessThan ( int arr [ ] , int n , int key ) { int l = 0 , r = n - 1 ; int index = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( arr [ m ] < key ) { l = m + 1 ; index = m ; } else { r = m - 1 ; } } return ( index + 1 ) ; }
static int countGreaterThan ( int arr [ ] , int n , int key ) { int l = 0 , r = n - 1 ; int index = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( arr [ m ] <= key ) { l = m + 1 ; } else { r = m - 1 ; index = m ; } } if ( index == - 1 ) return 0 ; return ( n - index ) ; }
static int countTriplets ( int n , int a [ ] , int b [ ] , int c [ ] ) {
Arrays . sort ( a ) ; Arrays . sort ( b ) ; Arrays . sort ( c ) ; int count = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { int current = b [ i ] ;
int low = countLessThan ( a , n , current ) ;
int high = countGreaterThan ( c , n , current ) ;
count += ( low * high ) ; } return count ; }
public static void main ( String args [ ] ) { int a [ ] = { 1 , 5 } ; int b [ ] = { 2 , 4 } ; int c [ ] = { 3 , 6 } ; int size = a . length ; System . out . println ( countTriplets ( size , a , b , c ) ) ; } }
import java . io . * ; class GFG { static int costToBalance ( String s ) { if ( s . length ( ) == 0 ) System . out . println ( 0 ) ;
int ans = 0 ;
int o = 0 , c = 0 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s . charAt ( i ) == ' ( ' ) o ++ ; if ( s . charAt ( i ) == ' ) ' ) c ++ ; } if ( o != c ) return - 1 ; int [ ] a = new int [ s . length ( ) ] ; if ( s . charAt ( 0 ) == ' ( ' ) a [ 0 ] = 1 ; else a [ 0 ] = - 1 ; if ( a [ 0 ] < 0 ) ans += Math . abs ( a [ 0 ] ) ; for ( int i = 1 ; i < s . length ( ) ; i ++ ) { if ( s . charAt ( i ) == ' ( ' ) a [ i ] = a [ i - 1 ] + 1 ; else a [ i ] = a [ i - 1 ] - 1 ; if ( a [ i ] < 0 ) ans += Math . abs ( a [ i ] ) ; } return ans ; }
public static void main ( String args [ ] ) { String s ; s = " ) ) ) ( ( ( " ; System . out . println ( costToBalance ( s ) ) ; s = " ) ) ( ( " ; System . out . println ( costToBalance ( s ) ) ; } }
public static int middleOfThree ( int a , int b , int c ) {
int x = a - b ;
int y = b - c ;
int z = a - c ;
if ( x * y > 0 ) return b ;
else if ( x * z > 0 ) return c ; else return a ; }
public static void main ( String [ ] args ) { int a = 20 , b = 30 , c = 40 ; System . out . println ( middleOfThree ( a , b , c ) ) ; } }
public static void missing4 ( int [ ] arr ) {
int [ ] helper = new int [ 4 ] ;
for ( int i = 0 ; i < arr . length ; i ++ ) { int temp = Math . abs ( arr [ i ] ) ;
if ( temp <= arr . length ) arr [ temp - 1 ] *= ( - 1 ) ;
else if ( temp > arr . length ) { if ( temp % arr . length != 0 ) helper [ temp % arr . length - 1 ] = - 1 ; else helper [ ( temp % arr . length ) + arr . length - 1 ] = - 1 ; } }
for ( int i = 0 ; i < arr . length ; i ++ ) if ( arr [ i ] > 0 ) System . out . print ( i + 1 + " ▁ " ) ; for ( int i = 0 ; i < helper . length ; i ++ ) if ( helper [ i ] >= 0 ) System . out . print ( arr . length + i + 1 + " ▁ " ) ; return ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 } ; missing4 ( arr ) ; } }
static void lexiMiddleSmallest ( int K , int N ) {
if ( K % 2 == 0 ) {
System . out . print ( K / 2 + " ▁ " ) ;
for ( int i = 0 ; i < N - 1 ; ++ i ) { System . out . print ( K + " ▁ " ) ; } System . out . println ( ) ; return ; }
ArrayList < Integer > a = new ArrayList < Integer > ( ) ;
for ( int i = 0 ; i < N / 2 ; ++ i ) {
if ( a . get ( a . size ( ) - 1 ) == 1 ) {
a . remove ( a . size ( ) - 1 ) ; }
else {
int t = a . get ( a . size ( ) - 1 ) - 1 ; a . set ( a . get ( a . size ( ) - 1 ) , t ) ;
while ( a . size ( ) < N ) { a . add ( K ) ; } } }
for ( int i : a ) { System . out . print ( i + " ▁ " ) ; } System . out . println ( ) ; }
public static void main ( String [ ] args ) { int K = 2 , N = 4 ; lexiMiddleSmallest ( K , N ) ; } }
static void findLastElement ( int arr [ ] , int N ) {
Arrays . sort ( arr ) ; int i = 0 ;
for ( i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] - arr [ i - 1 ] != 0 && arr [ i ] - arr [ i - 1 ] != 2 ) { System . out . println ( " - 1" ) ; return ; } }
System . out . println ( arr [ N - 1 ] ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 6 , 8 , 0 , 8 } ; int N = arr . length ; findLastElement ( arr , N ) ; } }
static void maxDivisions ( Integer arr [ ] , int N , int X ) {
Arrays . sort ( arr , Collections . reverseOrder ( ) ) ;
int maxSub = 0 ;
int size = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
size ++ ;
if ( arr [ i ] * size >= X ) {
maxSub ++ ;
size = 0 ; } } System . out . print ( maxSub + "NEW_LINE"); }
Integer arr [ ] = { 1 , 3 , 3 , 7 } ;
int N = arr . length ;
int X = 3 ; maxDivisions ( arr , N , X ) ; } }
public static void maxPossibleSum ( int [ ] arr , int N ) {
Arrays . sort ( arr ) ; int sum = 0 ; int j = N - 3 ; while ( j >= 0 ) {
sum += arr [ j ] ; j -= 3 ; }
System . out . println ( sum ) ; }
int [ ] arr = { 7 , 4 , 5 , 2 , 3 , 1 , 5 , 9 } ;
int N = arr . length ; maxPossibleSum ( arr , N ) ; } }
static void insertionSort ( int arr [ ] , int n ) { int i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
static void printArray ( int arr [ ] , int n ) { int i ;
for ( i = 0 ; i < n ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } System . out . println ( ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; int N = arr . length ;
insertionSort ( arr , N ) ; printArray ( arr , N ) ; } }
static void getPairs ( int arr [ ] , int N , int K ) {
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) {
if ( arr [ i ] > K * arr [ i + 1 ] ) count ++ ; } } System . out . print ( count ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 6 , 2 , 1 } ; int N = arr . length ; int K = 2 ;
getPairs ( arr , N , K ) ; } }
static int merge ( int arr [ ] , int temp [ ] , int l , int m , int r , int K ) {
int i = l ;
int j = m + 1 ;
int cnt = 0 ; for ( i = l ; i <= m ; i ++ ) { boolean found = false ;
while ( j <= r ) {
if ( arr [ i ] >= K * arr [ j ] ) { found = true ; } else break ; j ++ ; }
if ( found == true ) { cnt += j - ( m + 1 ) ; j -- ; } }
int k = l ; i = l ; j = m + 1 ; while ( i <= m && j <= r ) { if ( arr [ i ] <= arr [ j ] ) temp [ k ++ ] = arr [ i ++ ] ; else temp [ k ++ ] = arr [ j ++ ] ; }
while ( i <= m ) temp [ k ++ ] = arr [ i ++ ] ;
while ( j <= r ) temp [ k ++ ] = arr [ j ++ ] ; for ( i = l ; i <= r ; i ++ ) arr [ i ] = temp [ i ] ;
return cnt ; }
static int mergeSortUtil ( int arr [ ] , int temp [ ] , int l , int r , int K ) { int cnt = 0 ; if ( l < r ) {
int m = ( l + r ) / 2 ;
cnt += mergeSortUtil ( arr , temp , l , m , K ) ; cnt += mergeSortUtil ( arr , temp , m + 1 , r , K ) ;
cnt += merge ( arr , temp , l , m , r , K ) ; } return cnt ; }
static void mergeSort ( int arr [ ] , int N , int K ) { int temp [ ] = new int [ N ] ; System . out . print ( mergeSortUtil ( arr , temp , 0 , N - 1 , K ) ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 6 , 2 , 5 } ; int N = arr . length ; int K = 2 ;
mergeSort ( arr , N , K ) ; } }
static void minRemovals ( int [ ] A , int N ) {
Arrays . sort ( A ) ;
int mx = A [ N - 1 ] ;
int sum = 1 ;
for ( int i = 0 ; i < N ; i ++ ) { sum += A [ i ] ; } if ( sum - mx >= mx ) { System . out . println ( 0 ) ; } else { System . out . println ( 2 * mx - sum ) ; } }
public static void main ( String [ ] args ) { int [ ] A = { 3 , 3 , 2 } ; int N = A . length ;
minRemovals ( A , N ) ; } }
static void rearrangeArray ( int a [ ] , int n ) {
Arrays . sort ( a ) ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( a [ i ] == i + 1 ) {
int temp = a [ i ] ; a [ i ] = a [ i + 1 ] ; a [ i + 1 ] = temp ; } }
if ( a [ n - 1 ] == n ) {
int temp = a [ n - 1 ] ; a [ n - 1 ] = a [ n - 2 ] ; a [ n - 2 ] = temp ; }
for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( a [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 5 , 3 , 2 , 4 } ; int N = arr . length ;
rearrangeArray ( arr , N ) ; } }
static int minOperations ( int arr1 [ ] , int arr2 [ ] , int i , int j ) {
if ( arr1 . equals ( arr2 ) ) return 0 ; if ( i >= arr1 . length j >= arr2 . length ) return 0 ;
if ( arr1 [ i ] < arr2 [ j ] )
return 1 + minOperations ( arr1 , arr2 , i + 1 , j + 1 ) ;
return Math . max ( minOperations ( arr1 , arr2 , i , j + 1 ) , minOperations ( arr1 , arr2 , i + 1 , j ) ) ; }
static void minOperationsUtil ( int [ ] arr ) { int brr [ ] = new int [ arr . length ] ; for ( int i = 0 ; i < arr . length ; i ++ ) brr [ i ] = arr [ i ] ; Arrays . sort ( brr ) ;
if ( arr . equals ( brr ) )
System . out . print ( "0" ) ;
else
System . out . println ( minOperations ( arr , brr , 0 , 0 ) ) ; }
public static void main ( final String [ ] args ) { int arr [ ] = { 4 , 7 , 2 , 3 , 9 } ; minOperationsUtil ( arr ) ; } }
static void canTransform ( String s , String t ) { int n = s . length ( ) ;
Vector < Integer > occur [ ] = new Vector [ 26 ] ; for ( int i = 0 ; i < occur . length ; i ++ ) occur [ i ] = new Vector < Integer > ( ) ; for ( int x = 0 ; x < n ; x ++ ) { char ch = ( char ) ( s . charAt ( x ) - ' a ' ) ; occur [ ch ] . add ( x ) ; }
int [ ] idx = new int [ 26 ] ; boolean poss = true ; for ( int x = 0 ; x < n ; x ++ ) { char ch = ( char ) ( t . charAt ( x ) - ' a ' ) ;
if ( idx [ ch ] >= occur [ ch ] . size ( ) ) {
poss = false ; break ; } for ( int small = 0 ; small < ch ; small ++ ) {
if ( idx [ small ] < occur [ small ] . size ( ) && occur [ small ] . get ( idx [ small ] ) < occur [ ch ] . get ( idx [ ch ] ) ) {
poss = false ; break ; } } idx [ ch ] ++ ; }
if ( poss ) { System . out . print ( " Yes " + "NEW_LINE"); } else { System . out . print ( " No " + "NEW_LINE"); } }
public static void main ( String [ ] args ) { String s , t ; s = " hdecb " ; t = " cdheb " ; canTransform ( s , t ) ; } }
static int inversionCount ( String s ) {
int [ ] freq = new int [ 26 ] ; int inv = 0 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { int temp = 0 ;
for ( int j = 0 ; j < ( int ) ( s . charAt ( i ) - ' a ' ) ; j ++ )
temp += freq [ j ] ; inv += ( i - temp ) ;
freq [ s . charAt ( i ) - ' a ' ] ++ ; } return inv ; }
static boolean haveRepeated ( String S1 , String S2 ) { int [ ] freq = new int [ 26 ] ; for ( char i : S1 . toCharArray ( ) ) { if ( freq [ i - ' a ' ] > 0 ) return true ; freq [ i - ' a ' ] ++ ; } for ( int i = 0 ; i < 26 ; i ++ ) freq [ i ] = 0 ; for ( char i : S2 . toCharArray ( ) ) { if ( freq [ i - ' a ' ] > 0 ) return true ; freq [ i - ' a ' ] ++ ; } return false ; }
static void checkToMakeEqual ( String S1 , String S2 ) {
int [ ] freq = new int [ 26 ] ; for ( int i = 0 ; i < S1 . length ( ) ; i ++ ) {
freq [ S1 . charAt ( i ) - ' a ' ] ++ ; } boolean flag = false ; for ( int i = 0 ; i < S2 . length ( ) ; i ++ ) { if ( freq [ S2 . charAt ( i ) - ' a ' ] == 0 ) {
flag = true ; break ; }
freq [ S2 . charAt ( i ) - ' a ' ] -- ; } if ( flag == true ) {
System . out . println ( " No " ) ; return ; }
int invCount1 = inversionCount ( S1 ) ; int invCount2 = inversionCount ( S2 ) ; if ( invCount1 == invCount2 || ( invCount1 & 1 ) == ( invCount2 & 1 ) || haveRepeated ( S1 , S2 ) ) {
System . out . println ( " Yes " ) ; } else System . out . println ( " No " ) ; }
public static void main ( String [ ] args ) { String S1 = " abbca " , S2 = " acabb " ; checkToMakeEqual ( S1 , S2 ) ; } }
static void sortArr ( int a [ ] , int n ) { int i , k ;
k = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) ; k = ( int ) Math . pow ( 2 , k ) ;
while ( k > 0 ) { for ( i = 0 ; i + k < n ; i ++ ) if ( a [ i ] > a [ i + k ] ) { int tmp = a [ i ] ; a [ i ] = a [ i + k ] ; a [ i + k ] = tmp ; }
k = k / 2 ; }
for ( i = 0 ; i < n ; i ++ ) { System . out . print ( a [ i ] + " ▁ " ) ; } }
int arr [ ] = { 5 , 20 , 30 , 40 , 36 , 33 , 25 , 15 , 10 } ; int n = arr . length ;
sortArr ( arr , n ) ; } }
static void maximumSum ( int arr [ ] , int n , int k ) {
int elt = n / k ; int sum = 0 ;
Arrays . sort ( arr ) ; int count = 0 ; int i = n - 1 ;
while ( count < k ) { sum += arr [ i ] ; i -- ; count ++ ; } count = 0 ; i = 0 ;
while ( count < k ) { sum += arr [ i ] ; i += elt - 1 ; count ++ ; }
System . out . println ( sum ) ; }
public static void main ( String [ ] args ) { int Arr [ ] = { 1 , 13 , 7 , 17 , 6 , 5 } ; int K = 2 ; int size = Arr . length ; maximumSum ( Arr , size , K ) ; } }
static int findMinSum ( int [ ] arr , int K , int L , int size ) { if ( K * L > size ) return - 1 ; int minsum = 0 ;
Arrays . sort ( arr ) ;
for ( int i = 0 ; i < K ; i ++ ) minsum += arr [ i ] ;
return minsum ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , 15 , 5 , 1 , 35 , 16 , 67 , 10 } ; int K = 3 ; int L = 2 ; int length = arr . length ; System . out . print ( findMinSum ( arr , K , L , length ) ) ; } }
static int findKthSmallest ( int [ ] arr , int n , int k ) {
int max = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; }
int [ ] counter = new int [ max + 1 ] ;
int smallest = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { counter [ arr [ i ] ] ++ ; }
for ( int num = 1 ; num <= max ; num ++ ) {
if ( counter [ num ] > 0 ) {
smallest += counter [ num ] ; }
if ( smallest >= k ) {
return num ; } } return - 1 ; }
int [ ] arr = { 7 , 1 , 4 , 4 , 20 , 15 , 8 } ; int N = arr . length ; int K = 5 ;
System . out . print ( findKthSmallest ( arr , N , K ) ) ; } }
static void lexNumbers ( int n ) { Vector < String > s = new Vector < String > ( ) ; for ( int i = 1 ; i <= n ; i ++ ) { s . add ( String . valueOf ( i ) ) ; } Collections . sort ( s ) ; Vector < Integer > ans = new Vector < Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) ans . add ( Integer . valueOf ( s . get ( i ) ) ) ; for ( int i = 0 ; i < n ; i ++ ) System . out . print ( ans . get ( i ) + " ▁ " ) ; }
public static void main ( String [ ] args ) { int n = 15 ; lexNumbers ( n ) ; } }
class GFG { static int N = 4 ; static void func ( int a [ ] [ ] ) { int i , j , k ;
for ( i = 0 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) { for ( j = 0 ; j < N ; j ++ ) { for ( k = j + 1 ; k < N ; ++ k ) {
if ( a [ i ] [ j ] > a [ i ] [ k ] ) {
int temp = a [ i ] [ j ] ; a [ i ] [ j ] = a [ i ] [ k ] ; a [ i ] [ k ] = temp ; } } } }
else { for ( j = 0 ; j < N ; j ++ ) { for ( k = j + 1 ; k < N ; ++ k ) {
if ( a [ i ] [ j ] < a [ i ] [ k ] ) {
int temp = a [ i ] [ j ] ; a [ i ] [ j ] = a [ i ] [ k ] ; a [ i ] [ k ] = temp ; } } } } }
for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) { System . out . print ( a [ i ] [ j ] + " ▁ " ) ; } System . out . print ( "NEW_LINE"); } }
public static void main ( String [ ] args ) { int a [ ] [ ] = { { 5 , 7 , 3 , 4 } , { 9 , 5 , 8 , 2 } , { 6 , 3 , 8 , 1 } , { 5 , 8 , 9 , 3 } } ; func ( a ) ; } }
static HashMap < Integer , Integer > [ ] g = new HashMap [ 200005 ] ; static HashSet < Integer > s = new HashSet < > ( ) ; static HashSet < Integer > ns = new HashSet < > ( ) ;
static void dfs ( int x ) { Vector < Integer > v = new Vector < > ( ) ; v . clear ( ) ; ns . clear ( ) ;
for ( int it : s ) {
if ( g [ x ] . get ( it ) != null ) { v . add ( it ) ; } else { ns . add ( it ) ; } } s = ns ; for ( int i : v ) { dfs ( i ) ; } }
static void weightOfMST ( int N ) {
int cnt = 0 ;
for ( int i = 1 ; i <= N ; ++ i ) { s . add ( i ) ; } Vector < Integer > qt = new Vector < > ( ) ; for ( int t : s ) qt . add ( t ) ;
while ( ! qt . isEmpty ( ) ) {
++ cnt ; int t = qt . get ( 0 ) ; qt . remove ( 0 ) ;
dfs ( t ) ; } System . out . print ( cnt - 4 ) ; }
public static void main ( String [ ] args ) { int N = 6 , M = 11 ; int edges [ ] [ ] = { { 1 , 3 } , { 1 , 4 } , { 1 , 5 } , { 1 , 6 } , { 2 , 3 } , { 2 , 4 } , { 2 , 5 } , { 2 , 6 } , { 3 , 4 } , { 3 , 5 } , { 3 , 6 } } ; for ( int i = 0 ; i < g . length ; i ++ ) g [ i ] = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < M ; ++ i ) { int u = edges [ i ] [ 0 ] ; int v = edges [ i ] [ 1 ] ; g [ u ] . put ( v , 1 ) ; g [ v ] . put ( u , 1 ) ; }
weightOfMST ( N ) ; } }
static int countPairs ( int [ ] A , int [ ] B ) { int n = A . length ; int ans = 0 ; Arrays . sort ( A ) ; Arrays . sort ( B ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( A [ i ] > B [ ans ] ) { ans ++ ; } } return ans ; }
public static void main ( String [ ] args ) { int [ ] A = { 30 , 28 , 45 , 22 } ; int [ ] B = { 35 , 25 , 22 , 48 } ; System . out . print ( countPairs ( A , B ) ) ; } }
static int maxMod ( int arr [ ] , int n ) { int maxVal = max_element ( arr , n ) ; int secondMax = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] < maxVal && arr [ i ] > secondMax ) { secondMax = arr [ i ] ; } } return secondMax ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 1 , 5 , 3 , 6 } ; int n = arr . length ; System . out . println ( maxMod ( arr , n ) ) ; } }
static boolean isPossible ( int A [ ] , int B [ ] , int n , int m , int x , int y ) {
if ( x > n y > m ) return false ;
Arrays . sort ( A ) ; Arrays . sort ( B ) ;
if ( A [ x - 1 ] < B [ m - y ] ) return true ; else return false ; }
public static void main ( String [ ] args ) { int A [ ] = { 1 , 1 , 1 , 1 , 1 } ; int B [ ] = { 2 , 2 } ; int n = A . length ; int m = B . length ; ; int x = 3 , y = 1 ; if ( isPossible ( A , B , n , m , x , y ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
import java . util . * ; class GFG { static int MAX = 100005 ;
static int Min_Replace ( int [ ] arr , int n , int k ) { Arrays . sort ( arr ) ;
Integer [ ] freq = new Integer [ MAX ] ; Arrays . fill ( freq , 0 ) ; int p = 0 ; freq [ p ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] == arr [ i - 1 ] ) ++ freq [ p ] ; else ++ freq [ ++ p ] ; }
Arrays . sort ( freq , Collections . reverseOrder ( ) ) ;
int ans = 0 ; for ( int i = k ; i <= p ; i ++ ) ans += freq [ i ] ;
return ans ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 7 , 8 , 2 , 3 , 2 , 3 } ; int n = arr . length ; int k = 2 ; System . out . println ( Min_Replace ( arr , n , k ) ) ; } }
static int Segment ( int x [ ] , int l [ ] , int n ) {
if ( n == 1 ) return 1 ;
int ans = 2 ; for ( int i = 1 ; i < n - 1 ; i ++ ) {
if ( x [ i ] - l [ i ] > x [ i - 1 ] ) ans ++ ;
else if ( x [ i ] + l [ i ] < x [ i + 1 ] ) {
x [ i ] = x [ i ] + l [ i ] ; ans ++ ; } }
return ans ; }
public static void main ( String [ ] args ) { int x [ ] = { 1 , 3 , 4 , 5 , 8 } , l [ ] = { 10 , 1 , 2 , 2 , 5 } ; int n = x . length ;
System . out . println ( Segment ( x , l , n ) ) ; } }
static int MinimizeleftOverSum ( int a [ ] , int n ) { Vector < Integer > v1 = new Vector < Integer > ( ) , v2 = new Vector < Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] % 2 == 1 ) v1 . add ( a [ i ] ) ; else v2 . add ( a [ i ] ) ; }
if ( v1 . size ( ) > v2 . size ( ) ) {
Collections . sort ( v1 ) ; Collections . sort ( v2 ) ;
int x = v1 . size ( ) - v2 . size ( ) - 1 ; int sum = 0 ; int i = 0 ;
while ( i < x ) { sum += v1 . get ( i ++ ) ; }
return sum ; }
else if ( v2 . size ( ) > v1 . size ( ) ) {
Collections . sort ( v1 ) ; Collections . sort ( v2 ) ;
int x = v2 . size ( ) - v1 . size ( ) - 1 ; int sum = 0 ; int i = 0 ;
while ( i < x ) { sum += v2 . get ( i ++ ) ; }
return sum ; }
else return 0 ; }
public static void main ( String [ ] args ) { int a [ ] = { 2 , 2 , 2 , 2 } ; int n = a . length ; System . out . println ( MinimizeleftOverSum ( a , n ) ) ; } }
static void minOperation ( String S , int N , int K ) {
if ( N % K != 0 ) { System . out . println ( " Not ▁ Possible " ) ; } else {
int [ ] count = new int [ 26 ] ; for ( int i = 0 ; i < N ; i ++ ) { count [ ( S . charAt ( i ) - 97 ) ] ++ ; } int E = N / K ; Vector < Integer > greaterE = new Vector < > ( ) ; Vector < Integer > lessE = new Vector < > ( ) ; for ( int i = 0 ; i < 26 ; i ++ ) {
if ( count [ i ] < E ) lessE . add ( E - count [ i ] ) ; else greaterE . add ( count [ i ] - E ) ; } Collections . sort ( greaterE ) ; Collections . sort ( lessE ) ; int mi = Integer . MAX_VALUE ; for ( int i = 0 ; i <= K ; i ++ ) {
int set1 = i ; int set2 = K - i ; if ( greaterE . size ( ) >= set1 && lessE . size ( ) >= set2 ) { int step1 = 0 ; int step2 = 0 ; for ( int j = 0 ; j < set1 ; j ++ ) step1 += greaterE . get ( j ) ; for ( int j = 0 ; j < set2 ; j ++ ) step2 += lessE . get ( j ) ; mi = Math . min ( mi , Math . max ( step1 , step2 ) ) ; } } System . out . println ( mi ) ; } }
public static void main ( String [ ] args ) { String S = " accb " ; int N = S . length ( ) ; int K = 2 ; minOperation ( S , N , K ) ; } }
static int minMovesToSort ( int arr [ ] , int n ) { int moves = 0 ; int i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
} return moves ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 5 , 2 , 8 , 4 } ; int n = arr . length ; System . out . println ( minMovesToSort ( arr , n ) ) ; } }
import java . util . * ; class GFG { static boolean prime [ ] = new boolean [ 100005 ] ; static void SieveOfEratosthenes ( int n ) { Arrays . fill ( prime , true ) ;
prime [ 1 ] = false ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i < n ; i += p ) { prime [ i ] = false ; } } } }
static void sortPrimes ( int arr [ ] , int n ) { SieveOfEratosthenes ( 100005 ) ;
Vector < Integer > v = new Vector < Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( prime [ arr [ i ] ] ) { v . add ( arr [ i ] ) ; } } Comparator comparator = Collections . reverseOrder ( ) ; Collections . sort ( v , comparator ) ; int j = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( prime [ arr [ i ] ] ) { arr [ i ] = v . get ( j ++ ) ; } } }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 3 , 2 , 6 , 100 , 17 } ; int n = arr . length ; sortPrimes ( arr , n ) ;
for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } } }
import java . io . * ; import java . util . Arrays ; class GFG { static void findOptimalPairs ( int arr [ ] , int N ) { Arrays . sort ( arr ) ;
for ( int i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) System . out . print ( " ( " + arr [ i ] + " , ▁ " + arr [ j ] + " ) " + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 9 , 6 , 5 , 1 } ; int N = arr . length ; findOptimalPairs ( arr , N ) ; } }
static int countBits ( int a ) { int count = 0 ; while ( a > 0 ) { if ( ( a & 1 ) > 0 ) count += 1 ; a = a >> 1 ; } return count ; }
static void insertionSort ( int arr [ ] , int aux [ ] , int n ) { for ( int i = 1 ; i < n ; i ++ ) {
int key1 = aux [ i ] ; int key2 = arr [ i ] ; int j = i - 1 ;
while ( j >= 0 && aux [ j ] < key1 ) { aux [ j + 1 ] = aux [ j ] ; arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } aux [ j + 1 ] = key1 ; arr [ j + 1 ] = key2 ; } }
static void sortBySetBitCount ( int arr [ ] , int n ) {
int aux [ ] = new int [ n ] ; for ( int i = 0 ; i < n ; i ++ ) aux [ i ] = countBits ( arr [ i ] ) ;
insertionSort ( arr , aux , n ) ; }
static void printArr ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = arr . length ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ; } }
static int countBits ( int a ) { int count = 0 ; while ( a > 0 ) { if ( ( a & 1 ) > 0 ) count += 1 ; a = a >> 1 ; } return count ; }
static void sortBySetBitCount ( int arr [ ] , int n ) { Vector < Integer > [ ] count = new Vector [ 32 ] ; for ( int i = 0 ; i < count . length ; i ++ ) count [ i ] = new Vector < Integer > ( ) ; int setbitcount = 0 ; for ( int i = 0 ; i < n ; i ++ ) { setbitcount = countBits ( arr [ i ] ) ; count [ setbitcount ] . add ( arr [ i ] ) ; }
int j = 0 ;
for ( int i = 31 ; i >= 0 ; i -- ) { Vector < Integer > v1 = count [ i ] ; for ( int p = 0 ; p < v1 . size ( ) ; p ++ ) arr [ j ++ ] = v1 . get ( p ) ; } }
static void printArr ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = arr . length ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ; } }
static void generateString ( int k1 , int k2 , char [ ] s ) {
int C1s = 0 , C0s = 0 ; int flag = 0 ; Vector < Integer > pos = new Vector < Integer > ( ) ;
for ( int i = 0 ; i < s . length ; i ++ ) { if ( s [ i ] == '0' ) { C0s ++ ;
if ( ( i + 1 ) % k1 != 0 && ( i + 1 ) % k2 != 0 ) { pos . add ( i ) ; } } else { C1s ++ ; } if ( C0s >= C1s ) {
if ( pos . size ( ) == 0 ) { System . out . print ( - 1 ) ; flag = 1 ; break ; }
else { int k = pos . get ( pos . size ( ) - 1 ) ; s [ k ] = '1' ; C0s -- ; C1s ++ ; pos . remove ( pos . size ( ) - 1 ) ; } } }
if ( flag == 0 ) { System . out . print ( s ) ; } }
public static void main ( String [ ] args ) { int K1 = 2 , K2 = 4 ; String S = "11000100" ; generateString ( K1 , K2 , S . toCharArray ( ) ) ; } }
static void maximizeProduct ( int N ) {
int MSB = ( int ) ( Math . log ( N ) / Math . log ( 2 ) ) ;
int X = 1 << MSB ;
int Y = N - ( 1 << MSB ) ;
for ( int i = 0 ; i < MSB ; i ++ ) {
if ( ( N & ( 1 << i ) ) == 0 ) {
X += 1 << i ;
Y += 1 << i ; } }
System . out . println ( X + " ▁ " + Y ) ; }
public static void main ( String [ ] args ) { int N = 45 ; maximizeProduct ( N ) ; } }
static boolean check ( int num ) {
int sm = 0 ;
int num2 = num * num ; while ( num > 0 ) { sm += num % 10 ; num /= 10 ; }
int sm2 = 0 ; while ( num2 > 0 ) { sm2 += num2 % 10 ; num2 /= 10 ; } return ( ( sm * sm ) == sm2 ) ; }
static int convert ( String s ) { int val = 0 ; s = reverse ( s ) ; int cur = 1 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { val += ( s . charAt ( i ) - '0' ) * cur ; cur *= 10 ; } return val ; }
static void generate ( String s , int len , HashSet < Integer > uniq ) {
if ( s . length ( ) == len ) {
if ( check ( convert ( s ) ) ) { uniq . add ( convert ( s ) ) ; } return ; }
for ( int i = 0 ; i <= 3 ; i ++ ) { generate ( s + ( char ) ( i + '0' ) , len , uniq ) ; } } static String reverse ( String input ) { char [ ] a = input . toCharArray ( ) ; int l , r = a . length - 1 ; for ( l = 0 ; l < r ; l ++ , r -- ) { char temp = a [ l ] ; a [ l ] = a [ r ] ; a [ r ] = temp ; } return String . valueOf ( a ) ; }
static int totalNumbers ( int L , int R ) {
int ans = 0 ;
int max_len = ( int ) ( Math . log10 ( R ) + 1 ) ;
HashSet < Integer > uniq = new HashSet < Integer > ( ) ; for ( int i = 1 ; i <= max_len ; i ++ ) {
generate ( " " , i , uniq ) ; }
for ( int x : uniq ) { if ( x >= L && x <= R ) { ans ++ ; } } return ans ; }
public static void main ( String [ ] args ) { int L = 22 , R = 22 ; System . out . print ( totalNumbers ( L , R ) ) ; } }
static void convertXintoY ( int X , int Y ) {
while ( Y > X ) {
if ( Y % 2 == 0 ) Y /= 2 ;
else if ( Y % 10 == 1 ) Y /= 10 ;
else break ; }
if ( X == Y ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; }
public static void main ( String [ ] args ) { int X = 100 , Y = 40021 ; convertXintoY ( X , Y ) ; } }
static void generateString ( int K ) {
String s = " " ;
for ( int i = 97 ; i < 97 + K ; i ++ ) { s = s + ( char ) ( i ) ;
for ( int j = i + 1 ; j < 97 + K ; j ++ ) { s += ( char ) ( i ) ; s += ( char ) ( j ) ; } }
s += ( char ) ( 97 ) ;
System . out . println ( s ) ; }
public static void main ( String [ ] args ) { int K = 4 ; generateString ( K ) ; } }
public static void findEquation ( int S , int M ) {
System . out . println ( "1 ▁ " + ( ( - 1 ) * S ) + " ▁ " + M ) ; }
public static void main ( String [ ] args ) { int S = 5 , M = 6 ; findEquation ( S , M ) ; } }
static int minSteps ( ArrayList < Integer > a , int n ) {
int [ ] prefix_sum = new int [ n ] ; prefix_sum [ 0 ] = a . get ( 0 ) ;
for ( int i = 1 ; i < n ; i ++ ) prefix_sum [ i ] += prefix_sum [ i - 1 ] + a . get ( i ) ;
int mx = - 1 ;
for ( int subgroupsum : prefix_sum ) { int sum = 0 ; int i = 0 ; int grp_count = 0 ;
while ( i < n ) { sum += a . get ( i ) ;
if ( sum == subgroupsum ) {
grp_count += 1 ; sum = 0 ; }
else if ( sum > subgroupsum ) { grp_count = - 1 ; break ; } i += 1 ; }
if ( grp_count > mx ) mx = grp_count ; }
return n - mx ; }
public static void main ( String [ ] args ) { ArrayList < Integer > A = new ArrayList < Integer > ( ) ; A . add ( 1 ) ; A . add ( 2 ) ; A . add ( 3 ) ; A . add ( 2 ) ; A . add ( 1 ) ; A . add ( 3 ) ; int N = A . size ( ) ;
System . out . print ( minSteps ( A , N ) ) ; } }
public static void maxOccuringCharacter ( String s ) {
int count0 = 0 , count1 = 0 ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( s . charAt ( i ) == '1' ) { count1 ++ ; }
else if ( s . charAt ( i ) == '0' ) { count0 ++ ; } }
int prev = - 1 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s . charAt ( i ) == '1' ) { prev = i ; break ; } }
for ( int i = prev + 1 ; i < s . length ( ) ; i ++ ) {
if ( s . charAt ( i ) != ' X ' ) {
if ( s . charAt ( i ) == '1' ) { count1 += i - prev - 1 ; prev = i ; }
else {
boolean flag = true ; for ( int j = i + 1 ; j < s . length ( ) ; j ++ ) { if ( s . charAt ( j ) == '1' ) { flag = false ; prev = j ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . length ( ) ; } } } }
prev = - 1 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s . charAt ( i ) == '0' ) { prev = i ; break ; } }
for ( int i = prev + 1 ; i < s . length ( ) ; i ++ ) {
if ( s . charAt ( i ) != ' X ' ) {
if ( s . charAt ( i ) == '0' ) {
count0 += i - prev - 1 ;
prev = i ; }
else {
boolean flag = true ; for ( int j = i + 1 ; j < s . length ( ) ; j ++ ) { if ( s . charAt ( j ) == '0' ) { prev = j ; flag = false ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . length ( ) ; } } } }
if ( s . charAt ( 0 ) == ' X ' ) {
int count = 0 ; int i = 0 ; while ( s . charAt ( i ) == ' X ' ) { count ++ ; i ++ ; }
if ( s . charAt ( i ) == '1' ) { count1 += count ; } }
if ( s . charAt ( s . length ( ) - 1 ) == ' X ' ) {
int count = 0 ; int i = s . length ( ) - 1 ; while ( s . charAt ( i ) == ' X ' ) { count ++ ; i -- ; }
if ( s . charAt ( i ) == '0' ) { count0 += count ; } }
if ( count0 == count1 ) { System . out . println ( " X " ) ; }
else if ( count0 > count1 ) { System . out . println ( 0 ) ; }
else System . out . println ( 1 ) ; }
public static void main ( String [ ] args ) { String S = " XX10XX10XXX1XX " ; maxOccuringCharacter ( S ) ; } }
static int maxSheets ( int A , int B ) { int area = A * B ;
int count = 1 ;
while ( area % 2 == 0 ) {
area /= 2 ;
count *= 2 ; } return count ; }
public static void main ( String args [ ] ) { int A = 5 , B = 10 ; System . out . println ( maxSheets ( A , B ) ) ; } }
static void findMinMoves ( int a , int b ) {
int ans = 0 ;
if ( a == b || Math . abs ( a - b ) == 1 ) { ans = a + b ; } else {
int k = Math . min ( a , b ) ;
int j = Math . max ( a , b ) ; ans = 2 * k + 2 * ( j - k ) - 1 ; }
System . out . print ( ans ) ; }
int a = 3 , b = 5 ;
findMinMoves ( a , b ) ; } }
static long cntEvenSumPairs ( long X , long Y ) {
long cntXEvenNums = X / 2 ;
long cntXOddNums = ( X + 1 ) / 2 ;
long cntYEvenNums = Y / 2 ;
long cntYOddNums = ( Y + 1 ) / 2 ;
long cntPairs = ( cntXEvenNums * cntYEvenNums ) + ( cntXOddNums * cntYOddNums ) ;
return cntPairs ; }
public static void main ( String [ ] args ) { long X = 2 ; long Y = 3 ; System . out . println ( cntEvenSumPairs ( X , Y ) ) ; } }
static int minMoves ( int [ ] arr ) { int N = arr . length ;
if ( N <= 2 ) return 0 ;
int ans = Integer . MAX_VALUE ;
for ( int i = - 1 ; i <= 1 ; i ++ ) { for ( int j = - 1 ; j <= 1 ; j ++ ) {
int num1 = arr [ 0 ] + i ;
int num2 = arr [ 1 ] + j ; int flag = 1 ; int moves = Math . abs ( i ) + Math . abs ( j ) ;
for ( int idx = 2 ; idx < N ; idx ++ ) {
int num = num1 + num2 ;
if ( Math . abs ( arr [ idx ] - num ) > 1 ) flag = 0 ;
else moves += Math . abs ( arr [ idx ] - num ) ; num1 = num2 ; num2 = num ; }
if ( flag > 0 ) ans = Math . min ( ans , moves ) ; } }
if ( ans == Integer . MAX_VALUE ) return - 1 ; return ans ; }
public static void main ( String [ ] args ) { int [ ] arr = { 4 , 8 , 9 , 17 , 27 } ; System . out . print ( minMoves ( arr ) ) ; } }
static void querySum ( int arr [ ] , int N , int Q [ ] [ ] , int M ) {
for ( int i = 0 ; i < M ; i ++ ) { int x = Q [ i ] [ 0 ] ; int y = Q [ i ] [ 1 ] ;
int sum = 0 ;
while ( x < N ) {
sum += arr [ x ] ;
x += y ; } System . out . print ( sum + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 7 , 5 , 4 } ; int Q [ ] [ ] = { { 2 , 1 } , { 3 , 2 } } ; int N = arr . length ; int M = Q . length ; querySum ( arr , N , Q , M ) ; } }
static int findBitwiseORGivenXORAND ( int X , int Y ) { return X + Y ; }
public static void main ( String [ ] args ) { int X = 5 , Y = 2 ; System . out . print ( findBitwiseORGivenXORAND ( X , Y ) ) ; } }
static int GCD ( int a , int b ) {
if ( b == 0 ) return a ;
return GCD ( b , a % b ) ; }
static void canReach ( int N , int A , int B , int K ) {
int gcd = GCD ( N , K ) ;
if ( Math . abs ( A - B ) % gcd == 0 ) { System . out . println ( " Yes " ) ; }
else { System . out . println ( " No " ) ; } }
public static void main ( String args [ ] ) { int N = 5 , A = 2 , B = 1 , K = 2 ;
canReach ( N , A , B , K ) ; } }
static void countOfSubarray ( int arr [ ] , int N ) {
Map < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ;
int answer = 0 ;
int sum = 0 ;
if ( mp . get ( 1 ) != null ) mp . put ( 1 , mp . get ( 1 ) + 1 ) ; else mp . put ( 1 , 1 ) ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += arr [ i ] ; if ( mp . get ( sum - i ) != null ) answer += mp . get ( sum - i ) ;
if ( mp . get ( sum - i ) != null ) mp . put ( sum - i , mp . get ( sum - i ) + 1 ) ; else mp . put ( sum - i , 1 ) ; }
System . out . print ( answer ) ; }
int arr [ ] = { 1 , 0 , 2 , 1 , 2 , - 2 , 2 , 4 } ;
int N = arr . length ;
countOfSubarray ( arr , N ) ; } }
static int minAbsDiff ( int N ) {
int sumSet1 = 0 ;
int sumSet2 = 0 ;
for ( int i = N ; i > 0 ; i -- ) {
if ( sumSet1 <= sumSet2 ) { sumSet1 += i ; } else { sumSet2 += i ; } } return Math . abs ( sumSet1 - sumSet2 ) ; }
public static void main ( String [ ] args ) { int N = 6 ; System . out . println ( minAbsDiff ( N ) ) ; } }
static boolean checkDigits ( int n ) {
do { int r = n % 10 ;
if ( r == 3 r == 4 r == 6 r == 7 r == 9 ) return false ; n /= 10 ; } while ( n != 0 ) ; return true ; }
static boolean isPrime ( int n ) { if ( n <= 1 ) return false ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; } return true ; }
static boolean isAllPrime ( int n ) { return isPrime ( n ) && checkDigits ( n ) ; }
public static void main ( String [ ] args ) { int N = 101 ; if ( isAllPrime ( N ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static void minCost ( String str , int a , int b ) {
int openUnbalanced = 0 ;
int closedUnbalanced = 0 ;
int openCount = 0 ;
int closedCount = 0 ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
if ( str . charAt ( i ) == ' ( ' ) { openUnbalanced ++ ; openCount ++ ; }
else {
if ( openUnbalanced == 0 )
closedUnbalanced ++ ;
else
openUnbalanced -- ;
closedCount ++ ; } }
int result = a * ( Math . abs ( openCount - closedCount ) ) ;
if ( closedCount > openCount ) closedUnbalanced -= ( closedCount - openCount ) ; if ( openCount > closedCount ) openUnbalanced -= ( openCount - closedCount ) ;
result += Math . min ( a * ( openUnbalanced + closedUnbalanced ) , b * closedUnbalanced ) ;
System . out . print ( result + "NEW_LINE"); }
public static void main ( String [ ] args ) { String str = " ) ) ( ) ( ( ) ( ) ( " ; int A = 1 , B = 3 ; minCost ( str , A , B ) ; } }
public static void countEvenSum ( int low , int high , int k ) {
int even_count = high / 2 - ( low - 1 ) / 2 ; int odd_count = ( high + 1 ) / 2 - low / 2 ; long even_sum = 1 ; long odd_sum = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
long prev_even = even_sum ; long prev_odd = odd_sum ;
even_sum = ( prev_even * even_count ) + ( prev_odd * odd_count ) ;
odd_sum = ( prev_even * odd_count ) + ( prev_odd * even_count ) ; }
System . out . println ( even_sum ) ; }
int low = 4 ; int high = 5 ;
int K = 3 ;
countEvenSum ( low , high , K ) ; } }
public static void count ( int n , int k ) { long count = ( long ) ( Math . pow ( 10 , k ) - Math . pow ( 10 , k - 1 ) ) ;
System . out . print ( count ) ; }
public static void main ( String [ ] args ) { int n = 2 , k = 1 ; count ( n , k ) ; } }
static int func ( int N , int P ) {
int sumUptoN = ( N * ( N + 1 ) / 2 ) ; int sumOfMultiplesOfP ;
if ( N < P ) { return sumUptoN ; }
else if ( ( N / P ) == 1 ) { return sumUptoN - P + 1 ; }
sumOfMultiplesOfP = ( ( N / P ) * ( 2 * P + ( N / P - 1 ) * P ) ) / 2 ;
return ( sumUptoN + func ( N / P , P ) - sumOfMultiplesOfP ) ; }
int N = 10 , P = 5 ;
System . out . println ( func ( N , P ) ) ; } }
public static void findShifts ( int [ ] A , int N ) {
int [ ] shift = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) {
if ( i == A [ i ] - 1 ) shift [ i ] = 0 ;
else
shift [ i ] = ( A [ i ] - 1 - i + N ) % N ; }
for ( int i = 0 ; i < N ; i ++ ) System . out . print ( shift [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 4 , 3 , 2 , 5 } ; int N = arr . length ; findShifts ( arr , N ) ; } }
public static void constructmatrix ( int N ) { boolean check = true ; for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( i == j ) { System . out . print ( "1 ▁ " ) ; } else if ( check ) {
System . out . print ( "2 ▁ " ) ; check = false ; } else {
System . out . print ( " - 2 ▁ " ) ; check = true ; } } System . out . println ( ) ; } }
public static void main ( String [ ] args ) { int N = 5 ; constructmatrix ( 5 ) ; } }
static int check ( int unit_digit , int X ) { int times , digit ;
for ( times = 1 ; times <= 10 ; times ++ ) { digit = ( X * times ) % 10 ; if ( digit == unit_digit ) return times ; }
return - 1 ; }
static int getNum ( int N , int X ) { int unit_digit ;
unit_digit = N % 10 ;
int times = check ( unit_digit , X ) ;
if ( times == - 1 ) return times ;
else {
if ( N >= ( times * X ) )
return times ;
else return - 1 ; } }
public static void main ( String [ ] args ) { int N = 58 , X = 7 ; System . out . println ( getNum ( N , X ) ) ; } }
static int minPoints ( int n , int m ) { int ans = 0 ;
if ( ( n % 2 != 0 ) && ( m % 2 != 0 ) ) { ans = ( ( n * m ) / 2 ) + 1 ; } else { ans = ( n * m ) / 2 ; }
return ans ; }
int N = 5 , M = 7 ;
System . out . print ( minPoints ( N , M ) ) ; } }
static String getLargestString ( String s , int k ) {
int [ ] frequency_array = new int [ 26 ] ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) { frequency_array [ s . charAt ( i ) - ' a ' ] ++ ; }
String ans = " " ;
for ( int i = 25 ; i >= 0 {
if ( frequency_array [ i ] > k ) {
int temp = k ; String st = String . valueOf ( ( char ) ( i + ' a ' ) ) ; while ( temp > 0 ) {
ans += st ; temp -- ; } frequency_array [ i ] -= k ;
int j = i - 1 ; while ( frequency_array [ j ] <= 0 && j >= 0 ) { j -- ; }
if ( frequency_array [ j ] > 0 && j >= 0 ) { String str = String . valueOf ( ( char ) ( j + ' a ' ) ) ; ans += str ; frequency_array [ j ] -= 1 ; } else {
break ; } }
else if ( frequency_array [ i ] > 0 ) {
int temp = frequency_array [ i ] ; frequency_array [ i ] -= temp ; String st = String . valueOf ( ( char ) ( i + ' a ' ) ) ; while ( temp > 0 ) { ans += st ; temp -- ; } }
else { i -- ; } } return ans ; }
public static void main ( String [ ] args ) { String S = " xxxxzza " ; int k = 3 ; System . out . print ( getLargestString ( S , k ) ) ; } }
static int minOperations ( int a [ ] , int b [ ] , int n ) {
int minA = Arrays . stream ( a ) . min ( ) . getAsInt ( ) ;
for ( int x = minA ; x >= 0 ; x -- ) {
boolean check = true ;
int operations = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( x % b [ i ] == a [ i ] % b [ i ] ) { operations += ( a [ i ] - x ) / b [ i ] ; }
else { check = false ; break ; } } if ( check ) return operations ; } return - 1 ; }
public static void main ( String [ ] args ) { int N = 5 ; int A [ ] = { 5 , 7 , 10 , 5 , 15 } ; int B [ ] = { 2 , 2 , 1 , 3 , 5 } ; System . out . print ( minOperations ( A , B , N ) ) ; } }
static int getLargestSum ( int N ) {
int max_sum = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { for ( int j = i + 1 ; j <= N ; j ++ ) {
if ( i * j % ( i + j ) == 0 )
max_sum = Math . max ( max_sum , i + j ) ; } }
return max_sum ; }
public static void main ( String [ ] args ) { int N = 25 ; int max_sum = getLargestSum ( N ) ; System . out . print ( max_sum ) ; } }
static int maxSubArraySum ( int a [ ] , int size ) { int max_so_far = Integer . MIN_VALUE , max_ending_here = 0 ;
for ( int i = 0 ; i < size ; i ++ ) { max_ending_here = max_ending_here + a [ i ] ; if ( max_ending_here < 0 ) max_ending_here = 0 ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; } return max_so_far ; }
static int maxSum ( int a [ ] , int n ) {
int S = 0 ; int i ;
for ( i = 0 ; i < n ; i ++ ) S += a [ i ] ; int X = maxSubArraySum ( a , n ) ;
return 2 * X - S ; }
public static void main ( String [ ] args ) { int a [ ] = { - 1 , - 2 , - 3 } ; int n = a . length ; int max_sum = maxSum ( a , n ) ; System . out . print ( max_sum ) ; } }
static boolean isPrime ( int n ) { int flag = 1 ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) { flag = 0 ; break ; } } return ( flag == 1 ? true : false ) ; }
static boolean isPerfectSquare ( int x ) {
double sr = Math . sqrt ( x ) ;
return ( ( sr - Math . floor ( sr ) ) == 0 ) ; }
static int countInterestingPrimes ( int n ) { int answer = 0 ; for ( int i = 2 ; i <= n ; i ++ ) {
if ( isPrime ( i ) ) {
for ( int j = 1 ; j * j * j * j <= i ; j ++ ) {
if ( isPerfectSquare ( i - j * j * j * j ) ) { answer ++ ; break ; } } } }
return answer ; }
public static void main ( String [ ] args ) { int N = 10 ; System . out . print ( countInterestingPrimes ( N ) ) ; } }
static void decBinary ( int arr [ ] , int n ) { int k = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n /= 2 ; } }
static int binaryDec ( int arr [ ] , int n ) { int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
static int maxNum ( int n , int k ) {
int l = ( int ) ( Math . log ( n ) / Math . log ( 2 ) ) + 1 ;
int a [ ] = new int [ l ] ; decBinary ( a , n ) ;
int cn = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( a [ i ] == 0 && cn < k ) { a [ i ] = 1 ; cn ++ ; } }
return binaryDec ( a , l ) ; }
public static void main ( String [ ] args ) { int n = 4 , k = 1 ; System . out . println ( maxNum ( n , k ) ) ; } }
static void findSubSeq ( int arr [ ] , int n , int sum ) { for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( sum < arr [ i ] ) arr [ i ] = - 1 ;
else sum -= arr [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != - 1 ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 17 , 25 , 46 , 94 , 201 , 400 } ; int n = arr . length ; int sum = 272 ; findSubSeq ( arr , n , sum ) ; } }
class GFG { static int MAX = 26 ;
static char maxAlpha ( String str , int len ) {
int [ ] first = new int [ MAX ] ; int [ ] last = new int [ MAX ] ;
for ( int i = 0 ; i < MAX ; i ++ ) { first [ i ] = - 1 ; last [ i ] = - 1 ; }
for ( int i = 0 ; i < len ; i ++ ) { int index = ( str . charAt ( i ) - ' a ' ) ;
if ( first [ index ] == - 1 ) first [ index ] = i ; last [ index ] = i ; }
int ans = - 1 , maxVal = - 1 ;
for ( int i = 0 ; i < MAX ; i ++ ) {
if ( first [ i ] == - 1 ) continue ;
if ( ( last [ i ] - first [ i ] ) > maxVal ) { maxVal = last [ i ] - first [ i ] ; ans = i ; } } return ( char ) ( ans + ' a ' ) ; }
public static void main ( String [ ] args ) { String str = " abbba " ; int len = str . length ( ) ; System . out . print ( maxAlpha ( str , len ) ) ; } }
static void find_distinct ( int a [ ] , int n , int q , int queries [ ] ) { int [ ] check = new int [ MAX ] ; int [ ] idx = new int [ MAX ] ; int cnt = 1 ; for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( check [ a [ i ] ] == 0 ) {
idx [ i ] = cnt ; check [ a [ i ] ] = 1 ; cnt ++ ; } else {
idx [ i ] = cnt - 1 ; } }
for ( int i = 0 ; i < q ; i ++ ) { int m = queries [ i ] ; System . out . print ( idx [ m ] + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 3 , 1 , 2 , 3 , 4 , 5 } ; int n = a . length ; int queries [ ] = { 0 , 3 , 5 , 7 } ; int q = queries . length ; find_distinct ( a , n , q , queries ) ; } }
import java . io . * ; class GFG { static int MAX = 24 ;
static int countOp ( int x ) {
int arr [ ] = new int [ MAX ] ; arr [ 0 ] = 1 ; for ( int i = 1 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] * 2 ;
int temp = x ; boolean flag = true ;
int ans = 0 ;
int operations = 0 ; boolean flag2 = false ; for ( int i = 0 ; i < MAX ; i ++ ) { if ( arr [ i ] - 1 == x ) flag2 = true ;
if ( arr [ i ] > x ) { ans = i ; break ; } }
if ( flag2 ) return 0 ; while ( flag ) {
if ( arr [ ans ] < x ) ans ++ ; operations ++ ;
for ( int i = 0 ; i < MAX ; i ++ ) { int take = x ^ ( arr [ i ] - 1 ) ; if ( take <= arr [ ans ] - 1 ) {
if ( take > temp ) temp = take ; } }
if ( temp == arr [ ans ] - 1 ) { flag = false ; break ; } temp ++ ; operations ++ ; x = temp ; if ( x == arr [ ans ] - 1 ) flag = false ; }
return operations ; }
public static void main ( String [ ] args ) { int x = 39 ; System . out . println ( countOp ( x ) ) ; } }
static int minOperations ( int [ ] arr , int n ) { int maxi , result = 0 ;
int [ ] freq = new int [ 1000001 ] ; for ( int i = 0 ; i < n ; i ++ ) { int x = arr [ i ] ; freq [ x ] ++ ; }
maxi = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ; for ( int i = 1 ; i <= maxi ; i ++ ) { if ( freq [ i ] != 0 ) {
for ( int j = i * 2 ; j <= maxi ; j = j + i ) {
freq [ j ] = 0 ; }
result ++ ; } } return result ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 2 , 4 , 4 , 4 } ; int n = arr . length ; System . out . println ( minOperations ( arr , n ) ) ; } }
static int __gcd ( int a , int b ) { if ( a == 0 ) return b ; return __gcd ( b % a , a ) ; } static int minGCD ( int arr [ ] , int n ) { int minGCD = 0 ;
for ( int i = 0 ; i < n ; i ++ ) minGCD = __gcd ( minGCD , arr [ i ] ) ; return minGCD ; }
static int minLCM ( int arr [ ] , int n ) { int minLCM = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) minLCM = Math . min ( minLCM , arr [ i ] ) ; return minLCM ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 66 , 14 , 521 } ; int n = arr . length ; System . out . println ( " LCM ▁ = ▁ " + minLCM ( arr , n ) + " ▁ GCD ▁ = ▁ " + minGCD ( arr , n ) ) ; } }
static String formStringMinOperations ( char [ ] s ) {
int count [ ] = new int [ 3 ] ; for ( char c : s ) { count [ ( int ) c - 48 ] += 1 ; }
int processed [ ] = new int [ 3 ] ;
int reqd = ( int ) s . length / 3 ; for ( int i = 0 ; i < s . length ; i ++ ) {
if ( count [ s [ i ] - '0' ] == reqd ) { continue ; }
if ( s [ i ] == '0' && count [ 0 ] > reqd && processed [ 0 ] >= reqd ) {
if ( count [ 1 ] < reqd ) { s [ i ] = '1' ; count [ 1 ] ++ ; count [ 0 ] -- ; }
else if ( count [ 2 ] < reqd ) { s [ i ] = '2' ; count [ 2 ] ++ ; count [ 0 ] -- ; } }
if ( s [ i ] == '1' && count [ 1 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = '0' ; count [ 0 ] ++ ; count [ 1 ] -- ; } else if ( count [ 2 ] < reqd && processed [ 1 ] >= reqd ) { s [ i ] = '2' ; count [ 2 ] ++ ; count [ 1 ] -- ; } }
if ( s [ i ] == '2' && count [ 2 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = '0' ; count [ 0 ] ++ ; count [ 2 ] -- ; } else if ( count [ 1 ] < reqd ) { s [ i ] = '1' ; count [ 1 ] ++ ; count [ 2 ] -- ; } }
processed [ s [ i ] - '0' ] ++ ; } return String . valueOf ( s ) ; }
public static void main ( String [ ] args ) { String s = "011200" ; System . out . println ( formStringMinOperations ( s . toCharArray ( ) ) ) ; } }
static int findMinimumAdjacentSwaps ( int arr [ ] , int N ) {
boolean [ ] visited = new boolean [ N + 1 ] ; int minimumSwaps = 0 ; Arrays . fill ( visited , false ) ; for ( int i = 0 ; i < 2 * N ; i ++ ) {
if ( visited [ arr [ i ] ] == false ) { visited [ arr [ i ] ] = true ;
int count = 0 ; for ( int j = i + 1 ; j < 2 * N ; j ++ ) {
if ( visited [ arr [ j ] ] == false ) count ++ ;
else if ( arr [ i ] == arr [ j ] ) minimumSwaps += count ; } } } return minimumSwaps ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 2 , 3 , 3 , 1 , 2 } ; int N = arr . length ; N /= 2 ; System . out . println ( findMinimumAdjacentSwaps ( arr , N ) ) ; } }
static boolean possibility ( HashMap < Integer , Integer > m , int length , String s ) {
int countodd = 0 ; for ( int i = 0 ; i < length ; i ++ ) {
if ( m . get ( s . charAt ( i ) - '0' ) % 2 == 1 ) countodd ++ ;
if ( countodd > 1 ) return false ; } return true ; }
static void largestPalindrome ( String s ) {
int l = s . length ( ) ;
HashMap < Integer , Integer > m = new HashMap < > ( ) ; for ( int i = 0 ; i < l ; i ++ ) if ( m . containsKey ( s . charAt ( i ) - '0' ) ) m . put ( s . charAt ( i ) - '0' , m . get ( s . charAt ( i ) - '0' ) + 1 ) ; else m . put ( s . charAt ( i ) - '0' , 1 ) ;
if ( possibility ( m , l , s ) == false ) { System . out . print ( " Palindrome ▁ cannot ▁ be ▁ formed " ) ; return ; }
char [ ] largest = new char [ l ] ;
int front = 0 ;
for ( int i = 9 ; i >= 0 ; i -- ) {
if ( m . containsKey ( i ) && m . get ( i ) % 2 == 1 ) {
largest [ l / 2 ] = ( char ) ( i + 48 ) ;
m . put ( i , m . get ( i ) - 1 ) ;
while ( m . get ( i ) > 0 ) { largest [ front ] = ( char ) ( i + 48 ) ; largest [ l - front - 1 ] = ( char ) ( i + 48 ) ; m . put ( i , m . get ( i ) - 2 ) ; front ++ ; } } else {
while ( m . containsKey ( i ) && m . get ( i ) > 0 ) {
largest [ front ] = ( char ) ( i + 48 ) ; largest [ l - front - 1 ] = ( char ) ( i + 48 ) ;
m . put ( i , m . get ( i ) - 2 ) ;
front ++ ; } } }
for ( int i = 0 ; i < l ; i ++ ) System . out . print ( largest [ i ] ) ; }
public static void main ( String [ ] args ) { String s = "313551" ; largestPalindrome ( s ) ; } }
public static long swapCount ( String s ) {
Vector < Integer > pos = new Vector < Integer > ( ) ; for ( int i = 0 ; i < s . length ( ) ; ++ i ) if ( s . charAt ( i ) == ' [ ' ) pos . add ( i ) ;
int count = 0 ;
int p = 0 ;
long sum = 0 ; char [ ] S = s . toCharArray ( ) ; for ( int i = 0 ; i < s . length ( ) ; ++ i ) {
if ( S [ i ] == ' [ ' ) { ++ count ; ++ p ; } else if ( S [ i ] == ' ] ' ) -- count ;
if ( count < 0 ) {
sum += pos . get ( p ) - i ; char temp = S [ i ] ; S [ i ] = S [ pos . get ( p ) ] ; S [ pos . get ( p ) ] = temp ; ++ p ;
count = 1 ; } } return sum ; }
public static void main ( String [ ] args ) { String s = " [ ] ] [ ] [ " ; System . out . println ( swapCount ( s ) ) ; s = " [ [ ] [ ] ] " ; System . out . println ( swapCount ( s ) ) ; } }
static int minimumCostOfBreaking ( Integer X [ ] , Integer Y [ ] , int m , int n ) { int res = 0 ;
Arrays . sort ( X , Collections . reverseOrder ( ) ) ;
Arrays . sort ( Y , Collections . reverseOrder ( ) ) ;
int hzntl = 1 , vert = 1 ;
int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( X [ i ] > Y [ j ] ) { res += X [ i ] * vert ;
hzntl ++ ; i ++ ; } else { res += Y [ j ] * hzntl ;
vert ++ ; j ++ ; } }
int total = 0 ; while ( i < m ) total += X [ i ++ ] ; res += total * vert ;
total = 0 ; while ( j < n ) total += Y [ j ++ ] ; res += total * hzntl ; return res ; }
public static void main ( String arg [ ] ) { int m = 6 , n = 4 ; Integer X [ ] = { 2 , 1 , 3 , 1 , 4 } ; Integer Y [ ] = { 4 , 1 , 2 } ; System . out . print ( minimumCostOfBreaking ( X , Y , m - 1 , n - 1 ) ) ; } }
static int getMin ( int x , int y , int z ) { return Math . min ( Math . min ( x , y ) , z ) ; }
static int editDistance ( String str1 , String str2 , int m , int n ) {
int [ ] [ ] dp = new int [ m + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) {
if ( i == 0 )
dp [ i ] [ j ] = j ;
else if ( j == 0 )
dp [ i ] [ j ] = i ;
else if ( str1 . charAt ( i - 1 ) == str2 . charAt ( j - 1 ) ) dp [ i ] [ j ] = dp [ i - 1 ] [ j - 1 ] ;
else {
dp [ i ] [ j ] = 1 + getMin ( dp [ i ] [ j - 1 ] , dp [ i - 1 ] [ j ] , dp [ i - 1 ] [ j - 1 ] ) ; } } }
return dp [ m ] [ n ] ; }
static void minimumSteps ( String S , int N ) {
int ans = Integer . MAX_VALUE ;
for ( int i = 1 ; i < N ; i ++ ) { String S1 = S . substring ( 0 , i ) ; String S2 = S . substring ( i ) ;
int count = editDistance ( S1 , S2 , S1 . length ( ) , S2 . length ( ) ) ;
ans = Math . min ( ans , count ) ; }
System . out . print ( ans ) ; }
public static void main ( String [ ] args ) { String S = " aabb " ; int N = S . length ( ) ; minimumSteps ( S , N ) ; } }
static int minimumOperations ( int N ) {
int [ ] dp = new int [ N + 1 ] ; int i ;
for ( i = 0 ; i <= N ; i ++ ) { dp [ i ] = ( int ) 1e9 ; }
dp [ 2 ] = 0 ;
for ( i = 2 ; i <= N ; i ++ ) {
if ( dp [ i ] == ( int ) 1e9 ) continue ;
if ( i * 5 <= N ) { dp [ i * 5 ] = Math . min ( dp [ i * 5 ] , dp [ i ] + 1 ) ; }
if ( i + 3 <= N ) { dp [ i + 3 ] = Math . min ( dp [ i + 3 ] , dp [ i ] + 1 ) ; } }
if ( dp [ N ] == 1e9 ) return - 1 ;
return dp [ N ] ; }
public static void main ( String [ ] args ) { int N = 25 ; System . out . println ( minimumOperations ( N ) ) ; } }
static int MaxProfit ( int arr [ ] , int n , int transactionFee ) { int buy = - arr [ 0 ] ; int sell = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { int temp = buy ;
buy = Math . max ( buy , sell - arr [ i ] ) ; sell = Math . max ( sell , temp + arr [ i ] - transactionFee ) ; }
return Math . max ( sell , buy ) ; }
int arr [ ] = { 6 , 1 , 7 , 2 , 8 , 4 } ; int n = arr . length ; int transactionFee = 2 ;
System . out . println ( MaxProfit ( arr , n , transactionFee ) ) ; } }
static int start [ ] [ ] = new int [ 3 ] [ 3 ] ;
static int ending [ ] [ ] = new int [ 3 ] [ 3 ] ;
static void calculateStart ( int n , int m ) {
for ( int i = 1 ; i < m ; ++ i ) { start [ 0 ] [ i ] += start [ 0 ] [ i - 1 ] ; }
for ( int i = 1 ; i < n ; ++ i ) { start [ i ] [ 0 ] += start [ i - 1 ] [ 0 ] ; }
for ( int i = 1 ; i < n ; ++ i ) { for ( int j = 1 ; j < m ; ++ j ) {
start [ i ] [ j ] += Math . max ( start [ i - 1 ] [ j ] , start [ i ] [ j - 1 ] ) ; } } }
static void calculateEnd ( int n , int m ) {
for ( int i = n - 2 ; i >= 0 ; -- i ) { ending [ i ] [ m - 1 ] += ending [ i + 1 ] [ m - 1 ] ; }
for ( int i = m - 2 ; i >= 0 ; -- i ) { ending [ n - 1 ] [ i ] += ending [ n - 1 ] [ i + 1 ] ; }
for ( int i = n - 2 ; i >= 0 ; -- i ) { for ( int j = m - 2 ; j >= 0 ; -- j ) {
ending [ i ] [ j ] += Math . max ( ending [ i + 1 ] [ j ] , ending [ i ] [ j + 1 ] ) ; } } }
static void maximumPathSum ( int mat [ ] [ ] , int n , int m , int q , int coordinates [ ] [ ] ) {
for ( int i = 0 ; i < n ; ++ i ) { for ( int j = 0 ; j < m ; ++ j ) { start [ i ] [ j ] = mat [ i ] [ j ] ; ending [ i ] [ j ] = mat [ i ] [ j ] ; } }
calculateStart ( n , m ) ;
calculateEnd ( n , m ) ;
int ans = 0 ;
for ( int i = 0 ; i < q ; ++ i ) { int X = coordinates [ i ] [ 0 ] - 1 ; int Y = coordinates [ i ] [ 1 ] - 1 ;
ans = Math . max ( ans , start [ X ] [ Y ] + ending [ X ] [ Y ] - mat [ X ] [ Y ] ) ; }
System . out . print ( ans ) ; }
public static void main ( String [ ] args ) { int mat [ ] [ ] = { { 1 , 2 , 3 } , { 4 , 5 , 6 } , { 7 , 8 , 9 } } ; int N = 3 ; int M = 3 ; int Q = 2 ; int coordinates [ ] [ ] = { { 1 , 2 } , { 2 , 2 } } ; maximumPathSum ( mat , N , M , Q , coordinates ) ; } }
static int MaxSubsetlength ( String arr [ ] , int A , int B ) {
int dp [ ] [ ] = new int [ A + 1 ] [ B + 1 ] ;
for ( String str : arr ) {
int zeros = 0 , ones = 0 ; for ( char ch : str . toCharArray ( ) ) { if ( ch == '0' ) zeros ++ ; else ones ++ ; }
for ( int i = A ; i >= zeros ; i -- )
for ( int j = B ; j >= ones ; j -- )
dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - zeros ] [ j - ones ] + 1 ) ; }
return dp [ A ] [ B ] ; }
public static void main ( String [ ] args ) { String arr [ ] = { "1" , "0" , "0001" , "10" , "111001" } ; int A = 5 , B = 3 ; System . out . println ( MaxSubsetlength ( arr , A , B ) ) ; } }
static int numOfWays ( int a [ ] [ ] , int n , int i , HashSet < Integer > blue ) {
if ( i == n ) return 1 ;
int count = 0 ;
for ( int j = 0 ; j < n ; j ++ ) {
if ( a [ i ] [ j ] == 1 && ! blue . contains ( j ) ) { blue . add ( j ) ; count += numOfWays ( a , n , i + 1 , blue ) ; blue . remove ( j ) ; } } return count ; }
public static void main ( String [ ] args ) { int n = 3 ; int mat [ ] [ ] = { { 0 , 1 , 1 } , { 1 , 0 , 1 } , { 1 , 1 , 1 } } ; HashSet < Integer > mpp = new HashSet < > ( ) ; System . out . println ( ( numOfWays ( mat , n , 0 , mpp ) ) ) ; } }
static void minCost ( int arr [ ] , int n ) {
if ( n < 3 ) { System . out . println ( arr [ 0 ] ) ; return ; }
int dp [ ] = new int [ n ] ;
dp [ 0 ] = arr [ 0 ] ; dp [ 1 ] = dp [ 0 ] + arr [ 1 ] + arr [ 2 ] ;
for ( int i = 2 ; i < n - 1 ; i ++ ) dp [ i ] = Math . min ( dp [ i - 2 ] + arr [ i ] , dp [ i - 1 ] + arr [ i ] + arr [ i + 1 ] ) ;
dp [ n - 1 ] = Math . min ( dp [ n - 2 ] , dp [ n - 3 ] + arr [ n - 1 ] ) ;
System . out . println ( dp [ n - 1 ] ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 9 , 4 , 6 , 8 , 5 } ; int N = arr . length ; minCost ( arr , N ) ; } }
import java . util . * ; class GFG { static int M = 1000000007 ;
static int power ( int X , int Y ) {
int res = 1 ;
X = X % M ;
if ( X == 0 ) return 0 ;
while ( Y > 0 ) {
if ( ( Y & 1 ) != 0 ) {
res = ( res * X ) % M ; }
Y = Y >> 1 ;
X = ( X * X ) % M ; } return res ; }
static int findValue ( int n ) {
int X = 0 ;
int pow_10 = 1 ;
while ( n != 0 ) {
if ( ( n & 1 ) != 0 ) {
X += pow_10 ; }
pow_10 *= 10 ;
n /= 2 ; }
X = ( X * 2 ) % M ;
int res = power ( 2 , X ) ; return res ; }
public static void main ( String [ ] args ) { int n = 2 ; System . out . println ( findValue ( n ) ) ; } }
static int findWays ( int N ) {
if ( N == 0 ) { return 1 ; }
int cnt = 0 ;
for ( int i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i ) ; } }
return cnt ; }
public static void main ( String [ ] args ) { int N = 4 ;
System . out . print ( findWays ( N ) ) ; } }
static int checkEqualSumUtil ( int arr [ ] , int N , int sm1 , int sm2 , int sm3 , int j ) {
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; } else {
int l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
int m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
int r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
return Math . max ( Math . max ( l , m ) , r ) ; } }
static void checkEqualSum ( int arr [ ] , int N ) {
int sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { System . out . print ( " Yes " ) ; } else { System . out . print ( " No " ) ; } }
int arr [ ] = { 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 } ; int N = arr . length ;
checkEqualSum ( arr , N ) ; } }
import java . util . * ; class GFG { static HashMap < String , Integer > dp = new HashMap < String , Integer > ( ) ;
static int checkEqualSumUtil ( int arr [ ] , int N , int sm1 , int sm2 , int sm3 , int j ) { String s = String . valueOf ( sm1 ) + " _ " + String . valueOf ( sm2 ) + String . valueOf ( j ) ;
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; }
if ( dp . containsKey ( s ) ) return dp . get ( s ) ; else {
int l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
int m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
int r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
dp . put ( s , Math . max ( Math . max ( l , m ) , r ) ) ; return dp . get ( s ) ; } }
static void checkEqualSum ( int arr [ ] , int N ) {
int sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { System . out . print ( " Yes " ) ; } else { System . out . print ( " No " ) ; } }
int arr [ ] = { 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 } ; int N = arr . length ;
checkEqualSum ( arr , N ) ; } }
static void precompute ( int nextpos [ ] , int arr [ ] , int N ) {
nextpos [ N - 1 ] = N ; for ( int i = N - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] == arr [ i + 1 ] ) nextpos [ i ] = nextpos [ i + 1 ] ; else nextpos [ i ] = i + 1 ; } }
static void findIndex ( int query [ ] [ ] , int arr [ ] , int N , int Q ) {
int [ ] nextpos = new int [ N ] ; precompute ( nextpos , arr , N ) ; for ( int i = 0 ; i < Q ; i ++ ) { int l , r , x ; l = query [ i ] [ 0 ] ; r = query [ i ] [ 1 ] ; x = query [ i ] [ 2 ] ; int ans = - 1 ;
if ( arr [ l ] != x ) ans = l ;
else {
int d = nextpos [ l ] ;
if ( d <= r ) ans = d ; } System . out . print ( ans + "NEW_LINE"); } }
public static void main ( String [ ] args ) { int N , Q ; N = 6 ; Q = 3 ; int arr [ ] = { 1 , 2 , 1 , 1 , 3 , 5 } ; int query [ ] [ ] = { { 0 , 3 , 1 } , { 1 , 5 , 2 } , { 2 , 3 , 1 } } ; findIndex ( query , arr , N , Q ) ; } }
class GFG { static long mod = 10000000007L ;
static long countWays ( String s , String t , int k ) {
int n = s . length ( ) ;
int a = 0 , b = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { String p = s . substring ( i , n - i ) + s . substring ( 0 , i ) ;
if ( p == t ) a ++ ; else b ++ ; }
long dp1 [ ] = new long [ k + 1 ] ; long dp2 [ ] = new long [ k + 1 ] ; if ( s == t ) { dp1 [ 0 ] = 1 ; dp2 [ 0 ] = 0 ; } else { dp1 [ 0 ] = 0 ; dp2 [ 0 ] = 1 ; }
for ( int i = 1 ; i <= k ; i ++ ) { dp1 [ i ] = ( ( dp1 [ i - 1 ] * ( a - 1 ) ) % mod + ( dp2 [ i - 1 ] * a ) % mod ) % mod ; dp2 [ i ] = ( ( dp1 [ i - 1 ] * ( b ) ) % mod + ( dp2 [ i - 1 ] * ( b - 1 ) ) % mod ) % mod ; }
return dp1 [ k ] ; }
String S = " ab " , T = " ab " ;
int K = 2 ;
System . out . print ( countWays ( S , T , K ) ) ; } }
static int minOperation ( int k ) {
int dp [ ] = new int [ k + 1 ] ; for ( int i = 1 ; i <= k ; i ++ ) { dp [ i ] = dp [ i - 1 ] + 1 ;
if ( i % 2 == 0 ) { dp [ i ] = Math . min ( dp [ i ] , dp [ i / 2 ] + 1 ) ; } } return dp [ k ] ; }
public static void main ( String [ ] args ) { int K = 12 ; System . out . print ( minOperation ( K ) ) ; } }
static int maxSum ( int p0 , int p1 , int a [ ] , int pos , int n ) { if ( pos == n ) { if ( p0 == p1 ) return p0 ; else return 0 ; }
int ans = maxSum ( p0 , p1 , a , pos + 1 , n ) ;
ans = Math . max ( ans , maxSum ( p0 + a [ pos ] , p1 , a , pos + 1 , n ) ) ;
ans = Math . max ( ans , maxSum ( p0 , p1 + a [ pos ] , a , pos + 1 , n ) ) ; return ans ; }
int n = 4 ; int a [ ] = { 1 , 2 , 3 , 6 } ; System . out . println ( maxSum ( 0 , 0 , a , 0 , n ) ) ; } }
static int maxSum ( int a [ ] , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += a [ i ] ; int limit = 2 * sum + 1 ;
int dp [ ] [ ] = new int [ n + 1 ] [ limit ] ;
for ( int i = 0 ; i < n + 1 ; i ++ ) { for ( int j = 0 ; j < limit ; j ++ ) dp [ i ] [ j ] = INT_MIN ; }
dp [ 0 ] [ sum ] = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 0 ; j < limit ; j ++ ) {
if ( ( j - a [ i - 1 ] ) >= 0 && dp [ i - 1 ] [ j - a [ i - 1 ] ] != INT_MIN ) dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j - a [ i - 1 ] ] + a [ i - 1 ] ) ;
if ( ( j + a [ i - 1 ] ) < limit && dp [ i - 1 ] [ j + a [ i - 1 ] ] != INT_MIN ) dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j + a [ i - 1 ] ] ) ;
if ( dp [ i - 1 ] [ j ] != INT_MIN ) dp [ i ] [ j ] = Math . max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j ] ) ; } } return dp [ n ] [ sum ] ; }
public static void main ( String [ ] args ) { int n = 4 ; int [ ] a = { 1 , 2 , 3 , 6 } ; System . out . println ( maxSum ( a , n ) ) ; } }
static int fib [ ] = new int [ 100005 ] ;
static void computeFibonacci ( ) { fib [ 0 ] = 1 ; fib [ 1 ] = 1 ; for ( int i = 2 ; i < 100005 ; i ++ ) { fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ; } }
static int countString ( String str ) {
int ans = 1 ; int cnt = 1 ; for ( int i = 1 ; i < str . length ( ) ; i ++ ) {
if ( str . charAt ( i ) == str . charAt ( i - 1 ) ) { cnt ++ ; }
else { ans = ans * fib [ cnt ] ; cnt = 1 ; } }
ans = ans * fib [ cnt ] ;
return ans ; }
public static void main ( String [ ] args ) { String str = " abdllldefkkkk " ;
computeFibonacci ( ) ;
System . out . println ( countString ( str ) ) ; } }
import java . util . * ; class GFG { static int MAX = 1000 ;
static void printGolombSequence ( int N ) {
int [ ] arr = new int [ MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) arr [ i ] = 0 ;
int cnt = 0 ;
arr [ 0 ] = 0 ; arr [ 1 ] = 1 ;
Map < Integer , Integer > M = new HashMap < Integer , Integer > ( ) ;
M . put ( 2 , 2 ) ;
for ( int i = 2 ; i <= N ; i ++ ) {
if ( cnt == 0 ) { arr [ i ] = 1 + arr [ i - 1 ] ; cnt = M . get ( arr [ i ] ) ; cnt -- ; }
else { arr [ i ] = arr [ i - 1 ] ; cnt -- ; }
M . put ( i , arr [ i ] ) ; }
for ( int i = 1 ; i <= N ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int N = 11 ; printGolombSequence ( N ) ; } }
static int number_of_ways ( int n ) {
int [ ] includes_3 = new int [ n + 1 ] ;
int [ ] not_includes_3 = new int [ n + 1 ] ;
includes_3 [ 3 ] = 1 ; not_includes_3 [ 1 ] = 1 ; not_includes_3 [ 2 ] = 2 ; not_includes_3 [ 3 ] = 3 ;
for ( int i = 4 ; i <= n ; i ++ ) { includes_3 [ i ] = includes_3 [ i - 1 ] + includes_3 [ i - 2 ] + not_includes_3 [ i - 3 ] ; not_includes_3 [ i ] = not_includes_3 [ i - 1 ] + not_includes_3 [ i - 2 ] ; } return includes_3 [ n ] ; }
public static void main ( String [ ] args ) { int n = 7 ; System . out . print ( number_of_ways ( n ) ) ; } }
import java . util . * ; class GFG { static int MAX = 100000 ;
static int [ ] divisors = new int [ MAX ] ;
static void generateDivisors ( int n ) { for ( int i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) { if ( n / i == i ) { divisors [ i ] ++ ; } else { divisors [ i ] ++ ; divisors [ n / i ] ++ ; } } } }
static int findMaxMultiples ( int [ ] arr , int n ) {
int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
ans = Math . max ( divisors [ arr [ i ] ] , ans ) ;
generateDivisors ( arr [ i ] ) ; } return ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 8 , 1 , 28 , 4 , 2 , 6 , 7 } ; int n = arr . length ; System . out . print ( findMaxMultiples ( arr , n ) ) ; } }
class GFG { static int n = 3 ; static int maxV = 20 ;
static int [ ] [ ] [ ] dp = new int [ n ] [ n ] [ maxV ] ;
static int [ ] [ ] [ ] v = new int [ n ] [ n ] [ maxV ] ;
static int countWays ( int i , int j , int x , int arr [ ] [ ] ) {
if ( i == n j == n ) { return 0 ; } x = ( x & arr [ i ] [ j ] ) ; if ( x == 0 ) { return 0 ; } if ( i == n - 1 && j == n - 1 ) { return 1 ; }
if ( v [ i ] [ j ] [ x ] == 1 ) { return dp [ i ] [ j ] [ x ] ; } v [ i ] [ j ] [ x ] = 1 ;
dp [ i ] [ j ] [ x ] = countWays ( i + 1 , j , x , arr ) + countWays ( i , j + 1 , x , arr ) ; return dp [ i ] [ j ] [ x ] ; }
public static void main ( String [ ] args ) { int arr [ ] [ ] = { { 1 , 2 , 1 } , { 1 , 1 , 0 } , { 2 , 1 , 1 } } ; System . out . println ( countWays ( 0 , 0 , arr [ 0 ] [ 0 ] , arr ) ) ; } }
class GFG { static int N = 3 ;
static int FindMaximumSum ( int ind , int kon , int a [ ] , int b [ ] , int c [ ] , int n , int dp [ ] [ ] ) {
if ( ind == n ) return 0 ;
if ( dp [ ind ] [ kon ] != - 1 ) return dp [ ind ] [ kon ] ; int ans = ( int ) ( - 1e9 + 5 ) ;
if ( kon == 0 ) { ans = Math . max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = Math . max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon == 1 ) { ans = Math . max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; ans = Math . max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon == 2 ) { ans = Math . max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = Math . max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; } return dp [ ind ] [ kon ] = ans ; }
public static void main ( String [ ] args ) { int a [ ] = { 6 , 8 , 2 , 7 , 4 , 2 , 7 } ; int b [ ] = { 7 , 8 , 5 , 8 , 6 , 3 , 5 } ; int c [ ] = { 8 , 3 , 2 , 6 , 8 , 4 , 1 } ; int n = a . length ; int dp [ ] [ ] = new int [ n ] [ N ] ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) { dp [ i ] [ j ] = - 1 ; } }
int x = FindMaximumSum ( 0 , 0 , a , b , c , n , dp ) ;
int y = FindMaximumSum ( 0 , 1 , a , b , c , n , dp ) ;
int z = FindMaximumSum ( 0 , 2 , a , b , c , n , dp ) ;
System . out . println ( Math . max ( x , Math . max ( y , z ) ) ) ; } }
import java . util . * ; class GFG { static int mod = 1000000007 ;
static int noOfBinaryStrings ( int N , int k ) { int dp [ ] = new int [ 100002 ] ; for ( int i = 1 ; i <= k - 1 ; i ++ ) { dp [ i ] = 1 ; } dp [ k ] = 2 ; for ( int i = k + 1 ; i <= N ; i ++ ) { dp [ i ] = ( dp [ i - 1 ] + dp [ i - k ] ) % mod ; } return dp [ N ] ; }
public static void main ( String [ ] args ) { int N = 4 ; int K = 2 ; System . out . println ( noOfBinaryStrings ( N , K ) ) ; } }
static int findWaysToPair ( int p ) {
int dp [ ] = new int [ p + 1 ] ; dp [ 1 ] = 1 ; dp [ 2 ] = 2 ;
for ( int i = 3 ; i <= p ; i ++ ) { dp [ i ] = dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ; } return dp [ p ] ; }
public static void main ( String args [ ] ) { int p = 3 ; System . out . println ( findWaysToPair ( p ) ) ; } }
import java . io . * ; class GFG { static int CountWays ( int n ) {
if ( n == 0 ) { return 1 ; } if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 1 + 1 ; }
return CountWays ( n - 1 ) + CountWays ( n - 3 ) ; }
public static void main ( String [ ] args ) { int n = 5 ; System . out . println ( CountWays ( n ) ) ; } }
static ArrayList < Integer > factors ( int n ) {
ArrayList < Integer > v = new ArrayList < Integer > ( ) ; v . add ( 1 ) ;
for ( int i = 2 ; i <= Math . sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) { v . add ( i ) ;
if ( n / i != i ) { v . add ( n / i ) ; } } }
return v ; }
static boolean checkAbundant ( int n ) { ArrayList < Integer > v ; int sum = 0 ;
v = factors ( n ) ;
for ( int i = 0 ; i < v . size ( ) ; i ++ ) { sum += v . get ( i ) ; }
if ( sum > n ) return true ; else return false ; }
static boolean checkSemiPerfect ( int n ) { ArrayList < Integer > v ;
v = factors ( n ) ;
Collections . sort ( v ) ; int r = v . size ( ) ;
boolean subset [ ] [ ] = new boolean [ r + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= r ; i ++ ) subset [ i ] [ 0 ] = true ;
for ( int i = 1 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = false ;
for ( int i = 1 ; i <= r ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) {
if ( j < v . get ( i - 1 ) ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] ; else { subset [ i ] [ j ] = subset [ i - 1 ] [ j ] || subset [ i - 1 ] [ j - v . get ( i - 1 ) ] ; } } }
if ( ( subset [ r ] [ n ] ) == false ) return false ; else return true ; }
static boolean checkweird ( int n ) { if ( checkAbundant ( n ) == true && checkSemiPerfect ( n ) == false ) return true ; else return false ; }
public static void main ( String args [ ] ) { int n = 70 ; if ( checkweird ( n ) ) System . out . println ( " Weird ▁ Number " ) ; else System . out . println ( " Not ▁ Weird ▁ Number " ) ; } }
static int maxSubArraySumRepeated ( int a [ ] , int n , int k ) { int max_so_far = 0 ; int INT_MIN , max_ending_here = 0 ; for ( int i = 0 ; i < n * k ; i ++ ) {
max_ending_here = max_ending_here + a [ i % n ] ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; if ( max_ending_here < 0 ) max_ending_here = 0 ; } return max_so_far ; }
public static void main ( String [ ] args ) { int a [ ] = { 10 , 20 , - 30 , - 1 } ; int n = a . length ; int k = 3 ; System . out . println ( " Maximum ▁ contiguous ▁ sum ▁ is ▁ " + maxSubArraySumRepeated ( a , n , k ) ) ; } }
public static int longOddEvenIncSeq ( int arr [ ] , int n ) {
int [ ] lioes = new int [ n ] ;
int maxLen = 0 ;
for ( int i = 0 ; i < n ; i ++ ) lioes [ i ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && ( arr [ i ] + arr [ j ] ) % 2 != 0 && lioes [ i ] < lioes [ j ] + 1 ) lioes [ i ] = lioes [ j ] + 1 ;
for ( int i = 0 ; i < n ; i ++ ) if ( maxLen < lioes [ i ] ) maxLen = lioes [ i ] ;
return maxLen ; }
public static void main ( String argc [ ] ) { int [ ] arr = new int [ ] { 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 } ; int n = 10 ; System . out . println ( " Longest ▁ Increasing ▁ Odd " + " ▁ Even ▁ Subsequence : ▁ " + longOddEvenIncSeq ( arr , n ) ) ; } }
static boolean isOperator ( char op ) { return ( op == ' + ' op == ' * ' ) ; }
static void printMinAndMaxValueOfExp ( String exp ) { Vector < Integer > num = new Vector < Integer > ( ) ; Vector < Character > opr = new Vector < Character > ( ) ; String tmp = " " ;
for ( int i = 0 ; i < exp . length ( ) ; i ++ ) { if ( isOperator ( exp . charAt ( i ) ) ) { opr . add ( exp . charAt ( i ) ) ; num . add ( Integer . parseInt ( tmp ) ) ; tmp = " " ; } else { tmp += exp . charAt ( i ) ; } }
num . add ( Integer . parseInt ( tmp ) ) ; int len = num . size ( ) ; int [ ] [ ] minVal = new int [ len ] [ len ] ; int [ ] [ ] maxVal = new int [ len ] [ len ] ;
for ( int i = 0 ; i < len ; i ++ ) { for ( int j = 0 ; j < len ; j ++ ) { minVal [ i ] [ j ] = Integer . MAX_VALUE ; maxVal [ i ] [ j ] = 0 ;
if ( i == j ) minVal [ i ] [ j ] = maxVal [ i ] [ j ] = num . get ( i ) ; } }
for ( int L = 2 ; L <= len ; L ++ ) { for ( int i = 0 ; i < len - L + 1 ; i ++ ) { int j = i + L - 1 ; for ( int k = i ; k < j ; k ++ ) { int minTmp = 0 , maxTmp = 0 ;
if ( opr . get ( k ) == ' + ' ) { minTmp = minVal [ i ] [ k ] + minVal [ k + 1 ] [ j ] ; maxTmp = maxVal [ i ] [ k ] + maxVal [ k + 1 ] [ j ] ; }
else if ( opr . get ( k ) == ' * ' ) { minTmp = minVal [ i ] [ k ] * minVal [ k + 1 ] [ j ] ; maxTmp = maxVal [ i ] [ k ] * maxVal [ k + 1 ] [ j ] ; }
if ( minTmp < minVal [ i ] [ j ] ) minVal [ i ] [ j ] = minTmp ; if ( maxTmp > maxVal [ i ] [ j ] ) maxVal [ i ] [ j ] = maxTmp ; } } }
System . out . print ( " Minimum ▁ value ▁ : ▁ " + minVal [ 0 ] [ len - 1 ] + " , ▁ Maximum ▁ value ▁ : ▁ " + maxVal [ 0 ] [ len - 1 ] ) ; }
public static void main ( String [ ] args ) { String expression = "1 + 2*3 + 4*5" ; printMinAndMaxValueOfExp ( expression ) ; } }
static int MatrixChainOrder ( int p [ ] , int i , int j ) { if ( i == j ) return 0 ; int min = Integer . MAX_VALUE ;
for ( int k = i ; k < j ; k ++ ) { int count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 2 , 3 , 4 , 3 } ; int n = arr . length ; System . out . println ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int [ ] [ ] dp = new int [ 100 ] [ 100 ] ;
static int matrixChainMemoised ( int [ ] p , int i , int j ) { if ( i == j ) { return 0 ; } if ( dp [ i ] [ j ] != - 1 ) { return dp [ i ] [ j ] ; } dp [ i ] [ j ] = Integer . MAX_VALUE ; for ( int k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = Math . min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i ] [ j ] ; } static int MatrixChainOrder ( int [ ] p , int n ) { int i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int n = arr . length ; for ( int [ ] row : dp ) Arrays . fill ( row , - 1 ) ; System . out . println ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , n ) ) ; } }
static void flipBitsOfAandB ( int A , int B ) {
A = A ^ ( A & B ) ;
B = B ^ ( A & B ) ;
System . out . print ( A + " ▁ " + B ) ; }
public static void main ( String [ ] args ) { int A = 10 , B = 20 ; flipBitsOfAandB ( A , B ) ; } }
static int TotalHammingDistance ( int n ) { int i = 1 , sum = 0 ; while ( n / i > 0 ) { sum = sum + n / i ; i = i * 2 ; } return sum ; }
public static void main ( String [ ] args ) { int N = 9 ; System . out . println ( TotalHammingDistance ( N ) ) ; } }
import java . util . * ; class GFG { static final int m = 1000000007 ;
static void solve ( long n ) {
long s = 0 ; for ( int l = 1 ; l <= n ; ) {
int r = ( int ) ( n / Math . floor ( n / l ) ) ; int x = ( ( ( r % m ) * ( ( r + 1 ) % m ) ) / 2 ) % m ; int y = ( ( ( l % m ) * ( ( l - 1 ) % m ) ) / 2 ) % m ; int p = ( int ) ( ( n / l ) % m ) ;
s = ( s + ( ( ( x - y ) % m ) * p ) % m + m ) % m ; s %= m ; l = r + 1 ; }
System . out . print ( ( s + m ) % m ) ; }
public static void main ( String [ ] args ) { long n = 12 ; solve ( n ) ; } }
static int min_time_to_cut ( int N ) { if ( N == 0 ) return 0 ;
return ( int ) Math . ceil ( Math . log ( N ) / Math . log ( 2 ) ) ; }
public static void main ( String [ ] args ) { int N = 100 ; System . out . print ( min_time_to_cut ( N ) ) ; } }
static int findDistinctSums ( int n ) {
HashSet < Integer > s = new HashSet < > ( ) ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) {
s . add ( i + j ) ; } }
return s . size ( ) ; }
public static void main ( String [ ] args ) { int N = 3 ; System . out . print ( findDistinctSums ( N ) ) ; } }
static int printPattern ( int i , int j , int n ) {
if ( j >= n ) { return 0 ; } if ( i >= n ) { return 1 ; }
if ( j == i j == n - 1 - i ) {
if ( i == n - 1 - j ) { System . out . print ( " / " ) ; }
else { System . out . print ( " \ \" ) ; } }
else { System . out . print ( " * " ) ; }
if ( printPattern ( i , j + 1 , n ) == 1 ) { return 1 ; } System . out . println ( ) ;
return printPattern ( i + 1 , 0 , n ) ; }
public static void main ( String [ ] args ) { int N = 9 ;
printPattern ( 0 , 0 , N ) ; } }
private static int [ ] zArray ( int arr [ ] ) { int z [ ] ; int n = arr . length ; z = new int [ n ] ; int r = 0 , l = 0 ;
for ( int k = 1 ; k < n ; k ++ ) {
if ( k > r ) { r = l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; }
else { int k1 = k - l ; if ( z [ k1 ] < r - k + 1 ) z [ k ] = z [ k1 ] ; else { l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; } } } return z ; }
private static int [ ] mergeArray ( int A [ ] , int B [ ] ) { int n = A . length ; int m = B . length ; int z [ ] ;
int c [ ] = new int [ n + m + 1 ] ;
for ( int i = 0 ; i < m ; i ++ ) c [ i ] = B [ i ] ;
c [ m ] = Integer . MAX_VALUE ;
for ( int i = 0 ; i < n ; i ++ ) c [ m + i + 1 ] = A [ i ] ;
z = zArray ( c ) ; return z ; }
private static void findZArray ( int A [ ] , int B [ ] , int n ) { int flag = 0 ; int z [ ] ; z = mergeArray ( A , B ) ;
for ( int i = 0 ; i < z . length ; i ++ ) { if ( z [ i ] == n ) { System . out . print ( ( i - n - 1 ) + " ▁ " ) ; flag = 1 ; } } if ( flag == 0 ) { System . out . println ( " Not ▁ Found " ) ; } }
public static void main ( String args [ ] ) { int A [ ] = { 1 , 2 , 3 , 2 , 3 , 2 } ; int B [ ] = { 2 , 3 } ; int n = B . length ; findZArray ( A , B , n ) ; } }
static int getCount ( String a , String b ) {
if ( b . length ( ) % a . length ( ) != 0 ) return - 1 ; int count = b . length ( ) / a . length ( ) ;
String str = " " ; for ( int i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str . equals ( b ) ) return count ; return - 1 ; }
public static void main ( String [ ] args ) { String a = " geeks " ; String b = " geeksgeeks " ; System . out . println ( getCount ( a , b ) ) ; } }
static boolean check ( String S1 , String S2 ) {
int n1 = S1 . length ( ) ; int n2 = S2 . length ( ) ;
HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < n1 ; i ++ ) { if ( mp . containsKey ( ( int ) S1 . charAt ( i ) ) ) { mp . put ( ( int ) S1 . charAt ( i ) , mp . get ( ( int ) S1 . charAt ( i ) ) + 1 ) ; } else { mp . put ( ( int ) S1 . charAt ( i ) , 1 ) ; } }
for ( int i = 0 ; i < n2 ; i ++ ) {
if ( mp . containsKey ( ( int ) S2 . charAt ( i ) ) ) { mp . put ( ( int ) S2 . charAt ( i ) , mp . get ( ( int ) S2 . charAt ( i ) ) - 1 ) ; }
else if ( mp . containsKey ( S2 . charAt ( i ) - 1 ) && mp . containsKey ( S2 . charAt ( i ) - 2 ) ) { mp . put ( ( S2 . charAt ( i ) - 1 ) , mp . get ( S2 . charAt ( i ) - 1 ) - 1 ) ; mp . put ( ( S2 . charAt ( i ) - 2 ) , mp . get ( S2 . charAt ( i ) - 2 ) - 1 ) ; } else { return false ; } } return true ; }
public static void main ( String [ ] args ) { String S1 = " abbat " ; String S2 = " cat " ;
if ( check ( S1 , S2 ) ) System . out . print ( " YES " ) ; else System . out . print ( " NO " ) ; } }
static int countPattern ( String str ) { int len = str . length ( ) ; boolean oneSeen = false ;
for ( int i = 0 ; i < len ; i ++ ) { char getChar = str . charAt ( i ) ;
if ( getChar == '1' && oneSeen == true ) { if ( str . charAt ( i - 1 ) == '0' ) count ++ ; }
if ( getChar == '1' && oneSeen == false ) oneSeen = true ;
if ( getChar != '0' && str . charAt ( i ) != '1' ) oneSeen = false ; } return count ; }
public static void main ( String [ ] args ) { String str = "100001abc101" ; System . out . println ( countPattern ( str ) ) ; } }
static String checkIfPossible ( int N , String [ ] arr , String T ) {
int [ ] freqS = new int [ 256 ] ;
int [ ] freqT = new int [ 256 ] ;
for ( char ch : T . toCharArray ( ) ) { freqT [ ch - ' a ' ] ++ ; }
for ( int i = 0 ; i < N ; i ++ ) {
for ( char ch : arr [ i ] . toCharArray ( ) ) { freqS [ ch - ' a ' ] ++ ; } } for ( int i = 0 ; i < 256 ; i ++ ) {
if ( freqT [ i ] == 0 && freqS [ i ] != 0 ) { return " No " ; }
else if ( freqS [ i ] == 0 && freqT [ i ] != 0 ) { return " No " ; }
else if ( freqT [ i ] != 0 && freqS [ i ] != ( freqT [ i ] * N ) ) { return " No " ; } }
return " Yes " ; }
public static void main ( String [ ] args ) { String [ ] arr = { " abc " , " abb " , " acc " } ; String T = " abc " ; int N = arr . length ; System . out . println ( checkIfPossible ( N , arr , T ) ) ; } }
static int groupsOfOnes ( String S , int N ) {
int count = 0 ;
Stack < Integer > st = new Stack < > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( S . charAt ( i ) == '1' ) st . push ( 1 ) ;
else {
if ( ! st . empty ( ) ) { count ++ ; while ( ! st . empty ( ) ) { st . pop ( ) ; } } } }
if ( ! st . empty ( ) ) count ++ ;
return count ; }
String S = "100110111" ; int N = S . length ( ) ;
System . out . println ( groupsOfOnes ( S , N ) ) ; } }
static void generatePalindrome ( String S ) {
HashMap < Character , Integer > Hash = new HashMap < > ( ) ;
for ( int i = 0 ; i < S . length ( ) ; i ++ ) { if ( Hash . containsKey ( S . charAt ( i ) ) ) Hash . put ( S . charAt ( i ) , Hash . get ( S . charAt ( i ) ) + 1 ) ; else Hash . put ( S . charAt ( i ) , 1 ) ; }
TreeSet < String > st = new TreeSet < String > ( ) ;
for ( char i = ' a ' ; i <= ' z ' ; i ++ ) {
if ( Hash . containsKey ( i ) && Hash . get ( i ) == 2 ) {
for ( char j = ' a ' ; j <= ' z ' ; j ++ ) {
String s = " " ; if ( Hash . containsKey ( j ) && i != j ) { s += i ; s += j ; s += i ;
st . add ( s ) ; } } }
if ( Hash . containsKey ( i ) && Hash . get ( i ) >= 3 ) {
for ( char j = ' a ' ; j <= ' z ' ; j ++ ) {
String s = " " ;
if ( Hash . containsKey ( j ) ) { s += i ; s += j ; s += i ;
st . add ( s ) ; } } } }
for ( String ans : st ) { System . out . println ( ans ) ; } }
public static void main ( String [ ] args ) { String S = " ddabdac " ; generatePalindrome ( S ) ; } }
static void countOccurrences ( String S , String X , String Y ) {
int count = 0 ;
int N = S . length ( ) , A = X . length ( ) ; int B = Y . length ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( S . substring ( i , Math . min ( N , i + B ) ) . equals ( Y ) ) count ++ ;
if ( S . substring ( i , Math . min ( N , i + A ) ) . equals ( X ) ) System . out . print ( count + " ▁ " ) ; } }
public static void main ( String [ ] args ) { String S = " abcdefdefabc " ; String X = " abc " ; String Y = " def " ; countOccurrences ( S , X , Y ) ; } }
static void DFA ( String str , int N ) {
if ( N <= 1 ) { System . out . print ( " No " ) ; return ; }
int count = 0 ;
if ( str . charAt ( 0 ) == ' C ' ) { count ++ ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( str . charAt ( i ) == ' A ' || str . charAt ( i ) == ' B ' ) count ++ ; else break ; } } else {
System . out . print ( " No " ) ; return ; }
if ( count == N ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; }
public static void main ( String [ ] args ) { String str = " CAABBAAB " ; int N = str . length ( ) ; DFA ( str , N ) ; } }
static void minMaxDigits ( String str , int N ) {
int arr [ ] = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) arr [ i ] = ( str . charAt ( i ) - '0' ) % 3 ;
int zero = 0 , one = 0 , two = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( arr [ i ] == 0 ) zero ++ ; if ( arr [ i ] == 1 ) one ++ ; if ( arr [ i ] == 2 ) two ++ ; }
int sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) { sum = ( sum + arr [ i ] ) % 3 ; }
if ( sum == 0 ) { System . out . print ( 0 + " ▁ " ) ; } if ( sum == 1 ) { if ( ( one != 0 ) && ( N > 1 ) ) System . out . print ( 1 + " ▁ " ) ; else if ( two > 1 && N > 2 ) System . out . print ( 2 + " ▁ " ) ; else System . out . print ( - 1 + " ▁ " ) ; } if ( sum == 2 ) { if ( two != 0 && N > 1 ) System . out . print ( 1 + " ▁ " ) ; else if ( one > 1 && N > 2 ) System . out . print ( 2 + " ▁ " ) ; else System . out . print ( - 1 + " ▁ " ) ; }
if ( zero > 0 ) System . out . print ( N - 1 + " ▁ " ) ; else if ( one > 0 && two > 0 ) System . out . print ( N - 2 + " ▁ " ) ; else if ( one > 2 two > 2 ) System . out . print ( N - 3 + " ▁ " ) ; else System . out . print ( - 1 + " ▁ " ) ; }
public static void main ( String [ ] args ) { String str = "12345" ; int N = str . length ( ) ;
minMaxDigits ( str , N ) ; } }
static int findMinimumChanges ( int N , int K , char [ ] S ) {
int ans = 0 ;
for ( int i = 0 ; i < ( K + 1 ) / 2 ; i ++ ) {
HashMap < Character , Integer > mp = new HashMap < > ( ) ;
for ( int j = i ; j < N ; j += K ) {
if ( mp . containsKey ( S [ j ] ) ) { mp . put ( S [ j ] , mp . get ( S [ j ] ) + 1 ) ; } else { mp . put ( S [ j ] , 1 ) ; } }
for ( int j = N - i - 1 ; j >= 0 ; j -= K ) {
if ( K % 2 == 1 && i == K / 2 ) break ;
if ( mp . containsKey ( S [ j ] ) ) { mp . put ( S [ j ] , mp . get ( S [ j ] ) + 1 ) ; } else { mp . put ( S [ j ] , 1 ) ; } }
int curr_max = Integer . MIN_VALUE ; for ( Map . Entry < Character , Integer > p : mp . entrySet ( ) ) { curr_max = Math . max ( curr_max , p . getValue ( ) ) ; }
if ( ( K % 2 == 1 ) && i == K / 2 ) ans += ( N / K - curr_max ) ;
else ans += ( N / K * 2 - curr_max ) ; }
return ans ; }
public static void main ( String [ ] args ) { String S = " aabbcbbcb " ; int N = S . length ( ) ; int K = 3 ;
System . out . print ( findMinimumChanges ( N , K , S . toCharArray ( ) ) ) ; } }
static String checkString ( String s , int K ) { int n = s . length ( ) ;
Map < Character , Integer > mp = new HashMap < > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { mp . put ( s . charAt ( i ) , i ) ; } int cnt = 0 , f = 0 ;
Set < Character > st = new HashSet < > ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
st . add ( s . charAt ( i ) ) ;
if ( st . size ( ) > K ) { f = 1 ; break ; }
if ( mp . get ( s . charAt ( i ) ) == i ) st . remove ( s . charAt ( i ) ) ; } return ( f == 1 ? " Yes " : " No " ) ; }
public static void main ( String [ ] args ) { String s = " aabbcdca " ; int k = 2 ; System . out . println ( checkString ( s , k ) ) ; } }
public static void distinct ( String [ ] S , int M ) { int count = 0 ;
for ( int i = 0 ; i < S . length ; i ++ ) {
Set < Character > set = new HashSet < > ( ) ; for ( int j = 0 ; j < S [ i ] . length ( ) ; j ++ ) { if ( ! set . contains ( S [ i ] . charAt ( j ) ) ) set . add ( S [ i ] . charAt ( j ) ) ; } int c = set . size ( ) ;
if ( c <= M ) count += 1 ; } System . out . println ( count ) ; }
public static void main ( String [ ] args ) { String S [ ] = { " HERBIVORES " , " AEROPLANE " , " GEEKSFORGEEKS " } ; int M = 7 ; distinct ( S , M ) ; } }
static String removeOddFrequencyCharacters ( String s ) {
HashMap < Character , Integer > m = new HashMap < Character , Integer > ( ) ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { char p = s . charAt ( i ) ; Integer count = m . get ( p ) ; if ( count == null ) { count = 0 ; m . put ( p , 1 ) ; } else m . put ( p , count + 1 ) ; }
String new_string = " " ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( ( m . get ( s . charAt ( i ) ) & 1 ) == 1 ) continue ;
new_string += s . charAt ( i ) ; }
return new_string ; }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ;
str = removeOddFrequencyCharacters ( str ) ; System . out . print ( str ) ; } }
class GFG { static int i ;
static int productAtKthLevel ( String tree , int k , int level ) { if ( tree . charAt ( i ++ ) == ' ( ' ) {
if ( tree . charAt ( i ) == ' ) ' ) return 1 ; int product = 1 ;
if ( level == k ) product = tree . charAt ( i ) - '0' ;
++ i ; int leftproduct = productAtKthLevel ( tree , k , level + 1 ) ;
++ i ; int rightproduct = productAtKthLevel ( tree , k , level + 1 ) ;
++ i ; return product * leftproduct * rightproduct ; } return Integer . MIN_VALUE ; }
public static void main ( String [ ] args ) { String tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) " + " ( 9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; i = 0 ; System . out . print ( productAtKthLevel ( tree , k , 0 ) ) ; } }
static void findMostOccurringChar ( Vector < String > str ) {
int [ ] hash = new int [ 26 ] ;
for ( int i = 0 ; i < str . size ( ) ; i ++ ) {
for ( int j = 0 ; j < str . get ( i ) . length ( ) ; j ++ ) {
hash [ str . get ( i ) . charAt ( j ) - 97 ] ++ ; } }
int max = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) { max = hash [ i ] > hash [ max ] ? i : max ; } System . out . print ( ( char ) ( max + 97 ) + "NEW_LINE"); }
Vector < String > str = new Vector < String > ( ) ; str . add ( " animal " ) ; str . add ( " zebra " ) ; str . add ( " lion " ) ; str . add ( " giraffe " ) ; findMostOccurringChar ( str ) ; } }
public static boolean isPalindrome ( float num ) {
String s = String . valueOf ( num ) ;
int low = 0 ; int high = s . length ( ) - 1 ; while ( low < high ) {
if ( s . charAt ( low ) != s . charAt ( high ) ) return false ;
low ++ ; high -- ; } return true ; }
public static void main ( String args [ ] ) { float n = 123.321f ; if ( isPalindrome ( n ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
class GFG { final static int MAX = 26 ;
static int maxSubStr ( char [ ] str1 , int len1 , char [ ] str2 , int len2 ) {
if ( len1 > len2 ) return 0 ;
int freq1 [ ] = new int [ MAX ] ; for ( int i = 0 ; i < len1 ; i ++ ) freq1 [ i ] = 0 ; for ( int i = 0 ; i < len1 ; i ++ ) freq1 [ str1 [ i ] - ' a ' ] ++ ;
int freq2 [ ] = new int [ MAX ] ; for ( int i = 0 ; i < len2 ; i ++ ) freq2 [ i ] = 0 ; for ( int i = 0 ; i < len2 ; i ++ ) freq2 [ str2 [ i ] - ' a ' ] ++ ;
int minPoss = Integer . MAX_VALUE ; for ( int i = 0 ; i < MAX ; i ++ ) {
if ( freq1 [ i ] == 0 ) continue ;
if ( freq1 [ i ] > freq2 [ i ] ) return 0 ;
minPoss = Math . min ( minPoss , freq2 [ i ] / freq1 [ i ] ) ; } return minPoss ; }
public static void main ( String [ ] args ) { String str1 = " geeks " , str2 = " gskefrgoekees " ; int len1 = str1 . length ( ) ; int len2 = str2 . length ( ) ; System . out . println ( maxSubStr ( str1 . toCharArray ( ) , len1 , str2 . toCharArray ( ) , len2 ) ) ; } }
static int cntWays ( String str , int n ) { int x = n + 1 ; int ways = x * x * ( x * x - 1 ) / 12 ; return ways ; }
public static void main ( String [ ] args ) { String str = " ab " ; int n = str . length ( ) ; System . out . println ( cntWays ( str , n ) ) ; } }
static Set < String > uSet = new HashSet < String > ( ) ;
static int minCnt = Integer . MAX_VALUE ;
static void findSubStr ( String str , int cnt , int start ) {
if ( start == str . length ( ) ) {
minCnt = Math . min ( cnt , minCnt ) ; }
for ( int len = 1 ; len <= ( str . length ( ) - start ) ; len ++ ) {
String subStr = str . substring ( start , start + len ) ;
if ( uSet . contains ( subStr ) ) {
findSubStr ( str , cnt + 1 , start + len ) ; } } }
static void findMinSubStr ( String arr [ ] , int n , String str ) {
for ( int i = 0 ; i < n ; i ++ ) uSet . add ( arr [ i ] ) ;
findSubStr ( str , 0 , 0 ) ; }
public static void main ( String args [ ] ) { String str = "123456" ; String arr [ ] = { "1" , "12345" , "2345" , "56" , "23" , "456" } ; int n = arr . length ; findMinSubStr ( arr , n , str ) ; System . out . print ( minCnt ) ; } }
static int countSubStr ( String s , int n ) { int c1 = 0 , c2 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i < n - 5 && " geeks " . equals ( s . substring ( i , i + 5 ) ) ) { c1 ++ ; }
if ( i < n - 3 && " for " . equals ( s . substring ( i , i + 3 ) ) ) { c2 = c2 + c1 ; } } return c2 ; }
public static void main ( String [ ] args ) { String s = " geeksforgeeksisforgeeks " ; int n = s . length ( ) ; System . out . println ( countSubStr ( s , n ) ) ; } }
String string = " { [ ( ) ] } [ ] " ;
char [ ] lst1 = { ' { ' , ' ( ' , ' [ ' } ;
char [ ] lst2 = { ' } ' , ' ) ' , ' ] ' } ;
Vector < Character > lst = new Vector < Character > ( ) ;
HashMap < Character , Character > Dict = new HashMap < > ( ) ; Dict . put ( ' ) ' , ' ( ' ) ; Dict . put ( ' } ' , ' { ' ) ; Dict . put ( ' ] ' , ' [ ' ) ; int a = 0 , b = 0 , c = 0 ;
if ( Arrays . asList ( lst2 ) . contains ( string . charAt ( 0 ) ) ) { System . out . println ( 1 ) ; } else { int k = 0 ;
for ( int i = 0 ; i < string . length ( ) ; i ++ ) { if ( Arrays . asList ( lst1 ) . contains ( string . charAt ( i ) ) ) { lst . add ( string . charAt ( i ) ) ; k = i + 2 ; } else {
if ( lst . size ( ) == 0 && Arrays . asList ( lst2 ) . contains ( string . charAt ( i ) ) ) { System . out . println ( ( i + 1 ) ) ; c = 1 ; break ; } else {
if ( lst . size ( ) > 0 && Dict . get ( string . charAt ( i ) ) == lst . get ( lst . size ( ) - 1 ) ) { lst . remove ( lst . size ( ) - 1 ) ; } else {
a = 1 ; break ; } } } }
if ( lst . size ( ) == 0 && c == 0 ) { System . out . println ( 0 ) ; b = 1 ; } if ( a == 0 && b == 0 && c == 0 ) { System . out . println ( k ) ; } } } }
public class GFG { static final int MAX = 26 ;
static String encryptStr ( String str , int n , int x ) {
x = x % MAX ; char arr [ ] = str . toCharArray ( ) ;
int freq [ ] = new int [ MAX ] ; for ( int i = 0 ; i < n ; i ++ ) freq [ arr [ i ] - ' a ' ] ++ ; for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ arr [ i ] - ' a ' ] % 2 == 0 ) { int pos = ( arr [ i ] - ' a ' + x ) % MAX ; arr [ i ] = ( char ) ( pos + ' a ' ) ; }
else { int pos = ( arr [ i ] - ' a ' - x ) ; if ( pos < 0 ) pos += MAX ; arr [ i ] = ( char ) ( pos + ' a ' ) ; } }
return String . valueOf ( arr ) ; }
public static void main ( String [ ] args ) { String s = " abcda " ; int n = s . length ( ) ; int x = 3 ; System . out . println ( encryptStr ( s , n , x ) ) ; } }
static boolean isPossible ( char [ ] str ) {
Map < Character , Integer > freq = new HashMap < > ( ) ;
int max_freq = 0 ; for ( int j = 0 ; j < ( str . length ) ; j ++ ) { if ( freq . containsKey ( str [ j ] ) ) { freq . put ( str [ j ] , freq . get ( str [ j ] ) + 1 ) ; if ( freq . get ( str [ j ] ) > max_freq ) max_freq = freq . get ( str [ j ] ) ; } else { freq . put ( str [ j ] , 1 ) ; if ( freq . get ( str [ j ] ) > max_freq ) max_freq = freq . get ( str [ j ] ) ; } }
if ( max_freq <= ( str . length - max_freq + 1 ) ) return true ; return false ; }
public static void main ( String [ ] args ) { String str = " geeksforgeeks " ; if ( isPossible ( str . toCharArray ( ) ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static void printUncommon ( String str1 , String str2 ) { int a1 = 0 , a2 = 0 ; for ( int i = 0 ; i < str1 . length ( ) ; i ++ ) {
int ch = ( str1 . charAt ( i ) ) - ' a ' ;
a1 = a1 | ( 1 << ch ) ; } for ( int i = 0 ; i < str2 . length ( ) ; i ++ ) {
int ch = ( str2 . charAt ( i ) ) - ' a ' ;
a2 = a2 | ( 1 << ch ) ; }
int ans = a1 ^ a2 ; int i = 0 ; while ( i < 26 ) { if ( ans % 2 == 1 ) { System . out . print ( ( char ) ( ' a ' + i ) ) ; } ans = ans / 2 ; i ++ ; } }
public static void main ( String [ ] args ) { String str1 = " geeksforgeeks " ; String str2 = " geeksquiz " ; printUncommon ( str1 , str2 ) ; } }
static int countMinReversals ( String expr ) { int len = expr . length ( ) ;
if ( len % 2 != 0 ) return - 1 ;
int ans = 0 ; int i ;
int open = 0 ;
int close = 0 ; for ( i = 0 ; i < len ; i ++ ) {
if ( expr . charAt ( i ) == ' { ' ) open ++ ;
else { if ( open == 0 ) close ++ ; else open -- ; } } ans = ( close / 2 ) + ( open / 2 ) ;
close %= 2 ; open %= 2 ; if ( close != 0 ) ans += 2 ; return ans ; }
public static void main ( String args [ ] ) { String expr = " } } { { " ; System . out . println ( countMinReversals ( expr ) ) ; } }
static int totalPairs ( String s1 , String s2 ) { int a1 = 0 , b1 = 0 ;
for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) { if ( ( int ) s1 . charAt ( i ) % 2 != 0 ) a1 ++ ; else b1 ++ ; } int a2 = 0 , b2 = 0 ;
for ( int i = 0 ; i < s2 . length ( ) ; i ++ ) { if ( ( int ) s2 . charAt ( i ) % 2 != 0 ) a2 ++ ; else b2 ++ ; }
return ( ( a1 * a2 ) + ( b1 * b2 ) ) ; }
public static void main ( String [ ] args ) { String s1 = " geeks " , s2 = " for " ; System . out . println ( totalPairs ( s1 , s2 ) ) ; } }
static int prefixOccurrences ( String str ) { char c = str . charAt ( 0 ) ; int countc = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str . charAt ( i ) == c ) countc ++ ; } return countc ; }
public static void main ( String args [ ] ) { String str = " abbcdabbcd " ; System . out . println ( prefixOccurrences ( str ) ) ; } }
static int minOperations ( String s , String t , int n ) { int ct0 = 0 , ct1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( s . charAt ( i ) == t . charAt ( i ) ) continue ;
if ( s . charAt ( i ) == '0' ) ct0 ++ ;
else ct1 ++ ; } return Math . max ( ct0 , ct1 ) ; }
public static void main ( String args [ ] ) { String s = "010" , t = "101" ; int n = s . length ( ) ; System . out . println ( minOperations ( s , t , n ) ) ; } }
static String decryptString ( String str , int n ) {
int i = 0 , jump = 1 ; String decryptedStr = " " ; while ( i < n ) { decryptedStr += str . charAt ( i ) ; i += jump ;
jump ++ ; } return decryptedStr ; }
public static void main ( String [ ] args ) { String str = " geeeeekkkksssss " ; int n = str . length ( ) ; System . out . println ( decryptString ( str , n ) ) ; } }
static char bitToBeFlipped ( String s ) {
char last = s . charAt ( s . length ( ) - 1 ) ; char first = s . charAt ( 0 ) ;
if ( last == first ) { if ( last == '0' ) { return '1' ; } else { return '0' ; } }
else if ( last != first ) { return last ; } return last ; }
public static void main ( String [ ] args ) { String s = "1101011000" ; System . out . println ( bitToBeFlipped ( s ) ) ; } }
static void SieveOfEratosthenes ( boolean prime [ ] , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i < p_size ; i += p ) { prime [ i ] = false ; } } } }
static void sumProdOfPrimeFreq ( char [ ] s ) { boolean [ ] prime = new boolean [ s . length + 1 ] ; Arrays . fill ( prime , true ) ; SieveOfEratosthenes ( prime , s . length + 1 ) ; int i , j ;
Map < Character , Integer > mp = new HashMap < > ( ) ; for ( i = 0 ; i < s . length ; i ++ ) { mp . put ( s [ i ] , mp . get ( s [ i ] ) == null ? 1 : mp . get ( s [ i ] ) + 1 ) ; } int sum = 0 , product = 1 ;
for ( Map . Entry < Character , Integer > it : mp . entrySet ( ) ) {
if ( prime [ it . getValue ( ) ] ) { sum += it . getValue ( ) ; product *= it . getValue ( ) ; } } System . out . print ( " Sum ▁ = ▁ " + sum ) ; System . out . println ( " Product = " + product); }
public static void main ( String [ ] args ) { String s = " geeksforgeeks " ; sumProdOfPrimeFreq ( s . toCharArray ( ) ) ; } }
public static boolean multipleOrFactor ( String s1 , String s2 ) {
HashMap < Character , Integer > m1 = new HashMap < > ( ) ; HashMap < Character , Integer > m2 = new HashMap < > ( ) ; for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) { if ( m1 . containsKey ( s1 . charAt ( i ) ) ) { int x = m1 . get ( s1 . charAt ( i ) ) ; m1 . put ( s1 . charAt ( i ) , ++ x ) ; } else m1 . put ( s1 . charAt ( i ) , 1 ) ; } for ( int i = 0 ; i < s2 . length ( ) ; i ++ ) { if ( m2 . containsKey ( s2 . charAt ( i ) ) ) { int x = m2 . get ( s2 . charAt ( i ) ) ; m2 . put ( s2 . charAt ( i ) , ++ x ) ; } else m2 . put ( s2 . charAt ( i ) , 1 ) ; } for ( HashMap . Entry < Character , Integer > entry : m1 . entrySet ( ) ) {
if ( ! m2 . containsKey ( entry . getKey ( ) ) ) continue ;
if ( m2 . get ( entry . getKey ( ) ) != null && ( m2 . get ( entry . getKey ( ) ) % entry . getValue ( ) == 0 || entry . getValue ( ) % m2 . get ( entry . getKey ( ) ) == 0 ) ) continue ;
else return false ; } return true ; }
public static void main ( String [ ] args ) { String s1 = " geeksforgeeks " , s2 = " geeks " ; if ( multipleOrFactor ( s1 , s2 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static void solve ( String s ) {
HashMap < Character , Integer > m = new HashMap < > ( ) ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( m . containsKey ( s . charAt ( i ) ) ) m . put ( s . charAt ( i ) , m . get ( s . charAt ( i ) ) + 1 ) ; else m . put ( s . charAt ( i ) , 1 ) ; }
String new_string = " " ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( m . get ( s . charAt ( i ) ) % 2 == 0 ) continue ;
new_string = new_string + s . charAt ( i ) ; }
System . out . println ( new_string ) ; }
public static void main ( String [ ] args ) { String s = " aabbbddeeecc " ;
solve ( s ) ; } }
static boolean isPalindrome ( String str ) { int i = 0 , j = str . length ( ) - 1 ;
while ( i < j ) {
if ( str . charAt ( i ++ ) != str . charAt ( j -- ) ) return false ; }
return true ; }
static String removePalinWords ( String str ) {
String final_str = " " , word = " " ;
str = str + " ▁ " ; int n = str . length ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str . charAt ( i ) != ' ▁ ' ) word = word + str . charAt ( i ) ; else {
if ( ! ( isPalindrome ( word ) ) ) final_str += word + " ▁ " ;
word = " " ; } }
return final_str ; }
public static void main ( String [ ] args ) { String str = " Text ▁ contains ▁ malayalam ▁ and ▁ level ▁ words " ; System . out . print ( removePalinWords ( str ) ) ; } }
static int findSubSequence ( String s , int num ) {
int res = 0 ;
int i = 0 ; while ( num > 0 ) {
if ( ( num & 1 ) == 1 ) res += s . charAt ( i ) - '0' ; i ++ ;
num = num >> 1 ; } return res ; }
static int combinedSum ( String s ) {
int n = s . length ( ) ;
int c_sum = 0 ;
int range = ( 1 << n ) - 1 ;
for ( int i = 0 ; i <= range ; i ++ ) c_sum += findSubSequence ( s , i ) ;
return c_sum ; }
public static void main ( String [ ] args ) { String s = "123" ; System . out . println ( combinedSum ( s ) ) ; } }
class GFG { static final int MAX_CHAR = 26 ;
static void findSubsequence ( String str , int k ) {
int a [ ] = new int [ MAX_CHAR ] ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) a [ str . charAt ( i ) - ' a ' ] ++ ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) if ( a [ str . charAt ( i ) - ' a ' ] >= k ) System . out . print ( str . charAt ( i ) ) ; }
public static void main ( String [ ] args ) { int k = 2 ; findSubsequence ( " geeksforgeeks " , k ) ; } }
class GFG { static String convert ( String str ) {
String w = " " , z = " " ;
str = str . toUpperCase ( ) + " ▁ " ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
char ch = str . charAt ( i ) ; if ( ch != ' ▁ ' ) w = w + ch ; else {
z = z + ( Character . toLowerCase ( w . charAt ( 0 ) ) ) + w . substring ( 1 ) + " ▁ " ; w = " " ; } } return z ; }
public static void main ( String [ ] args ) { String str = " I ▁ got ▁ intern ▁ at ▁ geeksforgeeks " ; System . out . println ( convert ( str ) ) ; } }
static boolean isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
static String encryptString ( char [ ] s , int n , int k ) {
int [ ] cv = new int [ n ] ; int [ ] cc = new int [ n ] ; if ( isVowel ( s [ 0 ] ) ) cv [ 0 ] = 1 ; else cc [ 0 ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { cv [ i ] = cv [ i - 1 ] + ( isVowel ( s [ i ] ) == true ? 1 : 0 ) ; cc [ i ] = cc [ i - 1 ] + ( isVowel ( s [ i ] ) == true ? 0 : 1 ) ; } String ans = " " ; int prod = 0 ; prod = cc [ k - 1 ] * cv [ k - 1 ] ; ans += String . valueOf ( prod ) ;
for ( int i = k ; i < s . length ; i ++ ) { prod = ( cc [ i ] - cc [ i - k ] ) * ( cv [ i ] - cv [ i - k ] ) ; ans += String . valueOf ( prod ) ; } return ans ; }
public static void main ( String [ ] args ) { String s = " hello " ; int n = s . length ( ) ; int k = 2 ; System . out . print ( encryptString ( s . toCharArray ( ) , n , k ) + "NEW_LINE"); } }
import java . io . * ; class GFG { static int countOccurrences ( String str , String word ) {
String a [ ] = str . split ( " ▁ " ) ;
int count = 0 ; for ( int i = 0 ; i < a . length ; i ++ ) {
if ( word . equals ( a [ i ] ) ) count ++ ; } return count ; }
public static void main ( String args [ ] ) { String str = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " ; String word = " portal " ; System . out . println ( countOccurrences ( str , word ) ) ; } }
static void permute ( String input ) { int n = input . length ( ) ;
int max = 1 << n ;
input = input . toLowerCase ( ) ;
for ( int i = 0 ; i < max ; i ++ ) { char combination [ ] = input . toCharArray ( ) ;
for ( int j = 0 ; j < n ; j ++ ) { if ( ( ( i >> j ) & 1 ) == 1 ) combination [ j ] = ( char ) ( combination [ j ] - 32 ) ; }
System . out . print ( combination ) ; System . out . print ( " ▁ " ) ; } }
public static void main ( String [ ] args ) { permute ( " ABC " ) ; } }
static void printString ( String str , char ch , int count ) { int occ = 0 , i ;
if ( count == 0 ) { System . out . println ( str ) ; return ; }
for ( i = 0 ; i < str . length ( ) ; i ++ ) {
if ( str . charAt ( i ) == ch ) occ ++ ;
if ( occ == count ) break ; }
if ( i < str . length ( ) - 1 ) System . out . println ( str . substring ( i + 1 ) ) ;
else System . out . println ( " Empty ▁ string " ) ; }
public static void main ( String [ ] args ) { String str = " geeks ▁ for ▁ geeks " ; printString ( str , ' e ' , 2 ) ; } }
static boolean isVowel ( char c ) { return ( c == ' a ' c == ' A ' c == ' e ' c == ' E ' c == ' i ' c == ' I ' c == ' o ' c == ' O ' c == ' u ' c == ' U ' ) ; }
static String reverseVowel ( String str ) {
int i = 0 ; int j = str . length ( ) - 1 ; char [ ] str1 = str . toCharArray ( ) ; while ( i < j ) { if ( ! isVowel ( str1 [ i ] ) ) { i ++ ; continue ; } if ( ! isVowel ( str1 [ j ] ) ) { j -- ; continue ; }
char t = str1 [ i ] ; str1 [ i ] = str1 [ j ] ; str1 [ j ] = t ; i ++ ; j -- ; } String str2 = String . copyValueOf ( str1 ) ; return str2 ; }
public static void main ( String [ ] args ) { String str = " hello ▁ world " ; System . out . println ( reverseVowel ( str ) ) ; } }
static boolean isPalindrome ( String str ) {
int l = 0 ; int h = str . length ( ) - 1 ;
while ( h > l ) if ( str . charAt ( l ++ ) != str . charAt ( h -- ) ) return false ; return true ; }
static int minRemovals ( String str ) {
if ( str . charAt ( 0 ) == '') return 0 ;
if ( isPalindrome ( str ) ) return 1 ;
return 2 ; }
public static void main ( String [ ] args ) { System . out . println ( minRemovals ( "010010" ) ) ; System . out . println ( minRemovals ( "0100101" ) ) ; } }
static int power ( int x , int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
static int findModuloByM ( int X , int N , int M ) {
if ( N < 6 ) {
String temp = " " ; for ( int i = 0 ; i < N ; i ++ ) temp = temp + ( char ) ( X + 48 ) ;
int res = Integer . parseInt ( temp ) % M ; return res ; }
if ( N % 2 == 0 ) {
int half = findModuloByM ( X , N / 2 , M ) % M ;
int res = ( half * power ( 10 , N / 2 , M ) + half ) % M ; return res ; } else {
int half = findModuloByM ( X , N / 2 , M ) % M ;
int res = ( half * power ( 10 , N / 2 + 1 , M ) + half * 10 + X ) % M ; return res ; } }
public static void main ( String [ ] args ) { int X = 6 , N = 14 , M = 9 ;
System . out . println ( findModuloByM ( X , N , M ) ) ; } }
static class circle { double x ; double y ; double r ; public circle ( int x , int y , int r ) { this . x = x ; this . y = y ; this . r = r ; } }
static boolean check ( circle C [ ] ) {
double C1C2 = Math . sqrt ( ( C [ 1 ] . x - C [ 0 ] . x ) * ( C [ 1 ] . x - C [ 0 ] . x ) + ( C [ 1 ] . y - C [ 0 ] . y ) * ( C [ 1 ] . y - C [ 0 ] . y ) ) ;
boolean flag = false ;
if ( C1C2 < ( C [ 0 ] . r + C [ 1 ] . r ) ) {
if ( ( C [ 0 ] . x + C [ 1 ] . x ) == 2 * C [ 2 ] . x && ( C [ 0 ] . y + C [ 1 ] . y ) == 2 * C [ 2 ] . y ) {
flag = true ; } }
return flag ; }
static boolean IsFairTriplet ( circle c [ ] ) { boolean f = false ;
f |= check ( c ) ; for ( int i = 0 ; i < 2 ; i ++ ) { swap ( c [ 0 ] , c [ 2 ] ) ;
f |= check ( c ) ; } return f ; } static void swap ( circle circle1 , circle circle2 ) { circle temp = circle1 ; circle1 = circle2 ; circle2 = temp ; }
public static void main ( String [ ] args ) { circle C [ ] = new circle [ 3 ] ; C [ 0 ] = new circle ( 0 , 0 , 8 ) ; C [ 1 ] = new circle ( 0 , 10 , 6 ) ; C [ 2 ] = new circle ( 0 , 5 , 5 ) ; if ( IsFairTriplet ( C ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static double eccHyperbola ( double A , double B ) {
double r = ( double ) B * B / A * A ;
r += 1 ;
return Math . sqrt ( r ) ; }
public static void main ( String [ ] args ) { double A = 3.0 , B = 2.0 ; System . out . print ( eccHyperbola ( A , B ) ) ; } }
static float calculateArea ( float A , float B , float C , float D ) {
float S = ( A + B + C + D ) / 2 ;
float area = ( float ) Math . sqrt ( ( S - A ) * ( S - B ) * ( S - C ) * ( S - D ) ) ;
return area ; }
public static void main ( String [ ] args ) { float A = 10 ; float B = 15 ; float C = 20 ; float D = 25 ; System . out . println ( calculateArea ( A , B , C , D ) ) ; } }
static void triangleArea ( int a , int b ) {
double ratio = ( double ) b / a ;
System . out . println ( ratio ) ; }
public static void main ( String args [ ] ) { int a = 1 , b = 2 ; triangleArea ( a , b ) ; } }
static float distance ( int m , int n , int p , int q ) { return ( float ) Math . sqrt ( Math . pow ( n - m , 2 ) + Math . pow ( q - p , 2 ) * 1.0 ) ; }
static void Excenters ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 ) {
float a = distance ( x2 , x3 , y2 , y3 ) ; float b = distance ( x3 , x1 , y3 , y1 ) ; float c = distance ( x1 , x2 , y1 , y2 ) ;
pair [ ] excenter = new pair [ 4 ] ;
excenter [ 1 ] = new pair ( ( - ( a * x1 ) + ( b * x2 ) + ( c * x3 ) ) / ( - a + b + c ) , ( - ( a * y1 ) + ( b * y2 ) + ( c * y3 ) ) / ( - a + b + c ) ) ;
excenter [ 2 ] = new pair ( ( ( a * x1 ) - ( b * x2 ) + ( c * x3 ) ) / ( a - b + c ) , ( ( a * y1 ) - ( b * y2 ) + ( c * y3 ) ) / ( a - b + c ) ) ;
excenter [ 3 ] = new pair ( ( ( a * x1 ) + ( b * x2 ) - ( c * x3 ) ) / ( a + b - c ) , ( ( a * y1 ) + ( b * y2 ) - ( c * y3 ) ) / ( a + b - c ) ) ;
for ( int i = 1 ; i <= 3 ; i ++ ) { System . out . println ( ( int ) excenter [ i ] . first + " ▁ " + ( int ) excenter [ i ] . second ) ; } }
public static void main ( String [ ] args ) { int x1 , x2 , x3 , y1 , y2 , y3 ; x1 = 0 ; x2 = 3 ; x3 = 0 ; y1 = 0 ; y2 = 0 ; y3 = 4 ; Excenters ( x1 , y1 , x2 , y2 , x3 , y3 ) ; } }
static int Icositetragonal_num ( int n ) {
return ( 22 * n * n - 20 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( Icositetragonal_num ( n ) ) ; n = 10 ; System . out . println ( Icositetragonal_num ( n ) ) ; } }
static double area_of_circle ( int m , int n ) {
int square_of_radius = ( m * n ) / 4 ; double area = ( 3.141 * square_of_radius ) ; return area ; }
public static void main ( String [ ] args ) { int n = 10 ; int m = 30 ; System . out . println ( area_of_circle ( m , n ) ) ; } }
static double area ( int R ) {
double base = 1.732 * R ; double height = ( 1.5 ) * R ;
double area = 0.5 * base * height ; return area ; }
public static void main ( String [ ] args ) { int R = 7 ; System . out . println ( area ( R ) ) ; } }
static float circlearea ( float R ) {
if ( R < 0 ) return - 1 ;
float a = ( float ) ( ( 3.14 * R * R ) / 4 ) ; return a ; }
public static void main ( String [ ] args ) { float R = 2 ; System . out . println ( circlearea ( R ) ) ; } }
static int countPairs ( int [ ] P , int [ ] Q , int N , int M ) {
int [ ] A = new int [ 2 ] , B = new int [ 2 ] ;
for ( int i = 0 ; i < N ; i ++ ) A [ P [ i ] % 2 ] ++ ;
for ( int i = 0 ; i < M ; i ++ ) B [ Q [ i ] % 2 ] ++ ;
return ( A [ 0 ] * B [ 0 ] + A [ 1 ] * B [ 1 ] ) ; }
public static void main ( String [ ] args ) { int [ ] P = { 1 , 3 , 2 } ; int [ ] Q = { 3 , 0 } ; int N = P . length ; int M = Q . length ; System . out . print ( countPairs ( P , Q , N , M ) ) ; } }
static int countIntersections ( int n ) { return n * ( n - 1 ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( countIntersections ( n ) ) ; } }
public class GFG { static double PI = 3.14159 ;
static double areaOfTriangle ( float d ) {
float c = ( float ) ( 1.618 * d ) ; float s = ( d + c + c ) / 2 ;
double area = Math . sqrt ( s * ( s - c ) * ( s - c ) * ( s - d ) ) ;
return 5 * area ; }
static double areaOfRegPentagon ( float d ) {
double cal = 4 * Math . tan ( PI / 5 ) ; double area = ( 5 * d * d ) / cal ;
return area ; }
static double areaOfPentagram ( float d ) {
return areaOfRegPentagon ( d ) + areaOfTriangle ( d ) ; }
public static void main ( String [ ] args ) { float d = 5 ; System . out . println ( areaOfPentagram ( d ) ) ; } }
import java . io . * ; class GFG { static void anglequichord ( int z ) { System . out . println ( " The ▁ angle ▁ is ▁ " + z + " ▁ degrees " ) ; }
public static void main ( String [ ] args ) { int z = 48 ; anglequichord ( z ) ; } }
static void convertToASCII ( int N ) { String num = Integer . toString ( N ) ; for ( char ch : num . toCharArray ( ) ) { System . out . print ( ch + " ▁ ( " + ( int ) ch + ")NEW_LINE"); } }
public static void main ( String [ ] args ) { int N = 36 ; convertToASCII ( N ) ; } }
static void productExceptSelf ( int arr [ ] , int N ) {
int product = 1 ;
int z = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] != 0 ) product *= arr [ i ] ;
if ( arr [ i ] == 0 ) z += 1 ; }
int a = Math . abs ( product ) ; for ( int i = 0 ; i < N ; i ++ ) {
if ( z == 1 ) {
if ( arr [ i ] != 0 ) arr [ i ] = 0 ;
else arr [ i ] = product ; continue ; }
else if ( z > 1 ) {
arr [ i ] = 0 ; continue ; }
int b = Math . abs ( arr [ i ] ) ;
int curr = ( int ) Math . round ( Math . exp ( Math . log ( a ) - Math . log ( b ) ) ) ;
if ( arr [ i ] < 0 && product < 0 ) arr [ i ] = curr ;
else if ( arr [ i ] > 0 && product > 0 ) arr [ i ] = curr ;
else arr [ i ] = - 1 * curr ; }
for ( int i = 0 ; i < N ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 10 , 3 , 5 , 6 , 2 } ; int N = arr . length ;
productExceptSelf ( arr , N ) ; } }
static void singleDigitSubarrayCount ( int arr [ ] , int N ) {
int res = 0 ;
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( arr [ i ] <= 9 ) {
count ++ ;
res += count ; } else {
count = 0 ; } } System . out . print ( res ) ; }
int arr [ ] = { 0 , 1 , 14 , 2 , 5 } ;
int N = arr . length ; singleDigitSubarrayCount ( arr , N ) ; } }
static int isPossible ( int N ) { return ( ( ( N & ( N - 1 ) ) & N ) ) ; }
static void countElements ( int N ) {
int count = 0 ; for ( int i = 1 ; i <= N ; i ++ ) { if ( isPossible ( i ) != 0 ) count ++ ; } System . out . println ( count ) ; }
public static void main ( String [ ] args ) { int N = 15 ; countElements ( N ) ; } }
static void countElements ( int N ) { int Cur_Ele = 1 ; int Count = 0 ;
while ( Cur_Ele <= N ) {
Count ++ ;
Cur_Ele = Cur_Ele * 2 ; } System . out . print ( N - Count ) ; }
public static void main ( String [ ] args ) { int N = 15 ; countElements ( N ) ; } }
static void maxAdjacent ( int [ ] arr , int N ) { Vector < Integer > res = new Vector < Integer > ( ) ; int arr_max = Integer . MIN_VALUE ;
for ( int i = 1 ; i < N ; i ++ ) { arr_max = Math . max ( arr_max , Math . abs ( arr [ i - 1 ] - arr [ i ] ) ) ; } for ( int i = 1 ; i < N - 1 ; i ++ ) { int curr_max = Math . abs ( arr [ i - 1 ] - arr [ i + 1 ] ) ;
int ans = Math . max ( curr_max , arr_max ) ;
res . add ( ans ) ; }
for ( int x : res ) System . out . print ( x + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 4 , 7 , 8 } ; int N = arr . length ; maxAdjacent ( arr , N ) ; } }
static int minimumIncrement ( int arr [ ] , int N ) {
if ( N % 2 != 0 ) { System . out . println ( " - 1" ) ; System . exit ( 0 ) ; }
int cntEven = 0 ;
int cntOdd = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 == 0 ) {
cntEven += 1 ; } }
cntOdd = N - cntEven ;
return Math . abs ( cntEven - cntOdd ) / 2 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 3 , 4 , 9 } ; int N = arr . length ;
System . out . println ( minimumIncrement ( arr , N ) ) ; } }
static void cntWaysConsArray ( int A [ ] , int N ) {
int total = 1 ;
int oddArray = 1 ;
for ( int i = 0 ; i < N ; i ++ ) {
total = total * 3 ;
if ( A [ i ] % 2 == 0 ) {
oddArray *= 2 ; } }
System . out . println ( total - oddArray ) ; }
public static void main ( String [ ] args ) { int A [ ] = { 2 , 4 } ; int N = A . length ; cntWaysConsArray ( A , N ) ; } }
static void countNumberHavingKthBitSet ( int N , int K ) {
int numbers_rightmost_setbit_K = 0 ; for ( int i = 1 ; i <= K ; i ++ ) {
int numbers_rightmost_bit_i = ( N + 1 ) / 2 ;
N -= numbers_rightmost_bit_i ;
if ( i == K ) { numbers_rightmost_setbit_K = numbers_rightmost_bit_i ; } } System . out . println ( numbers_rightmost_setbit_K ) ; }
static public void main ( String args [ ] ) { int N = 15 ; int K = 2 ; countNumberHavingKthBitSet ( N , K ) ; } }
static int countSetBits ( int N ) { int count = 0 ;
while ( N != 0 ) { N = N & ( N - 1 ) ; count ++ ; }
return count ; }
public static void main ( String [ ] args ) { int N = 4 ; int bits = countSetBits ( N ) ;
System . out . println ( " Odd ▁ " + " : ▁ " + ( int ) ( Math . pow ( 2 , bits ) ) ) ;
System . out . println ( " Even ▁ " + " : ▁ " + ( N + 1 - ( int ) ( Math . pow ( 2 , bits ) ) ) ) ; } }
static void minMoves ( int arr [ ] , int N ) {
int odd_element_cnt = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 != 0 ) { odd_element_cnt ++ ; } }
int moves = ( odd_element_cnt ) / 2 ;
if ( odd_element_cnt % 2 != 0 ) moves += 2 ;
System . out . print ( moves ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 6 , 3 , 7 , 20 } ; int N = arr . length ;
minMoves ( arr , N ) ; } }
static void minimumSubsetDifference ( int N ) {
int blockOfSize8 = N / 8 ;
String str = " ABBABAAB " ;
int subsetDifference = 0 ;
String partition = " " ; while ( blockOfSize8 -- > 0 ) { partition += str ; }
int A [ ] = new int [ N ] ; int B [ ] = new int [ N ] ; int x = 0 , y = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( partition . charAt ( i ) == ' A ' ) { A [ x ++ ] = ( ( i + 1 ) * ( i + 1 ) ) ; }
else { B [ y ++ ] = ( ( i + 1 ) * ( i + 1 ) ) ; } }
System . out . println ( subsetDifference ) ;
for ( int i = 0 ; i < x ; i ++ ) System . out . print ( A [ i ] + " ▁ " ) ; System . out . println ( ) ;
for ( int i = 0 ; i < y ; i ++ ) System . out . print ( B [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int N = 8 ;
minimumSubsetDifference ( N ) ; } }
static void findTheGreatestX ( int P , int Q ) {
HashMap < Integer , Integer > divisiors = new HashMap < > ( ) ; for ( int i = 2 ; i * i <= Q ; i ++ ) { while ( Q % i == 0 && Q > 1 ) { Q /= i ;
if ( divisiors . containsKey ( i ) ) { divisiors . put ( i , divisiors . get ( i ) + 1 ) ; } else { divisiors . put ( i , 1 ) ; } } }
if ( Q > 1 ) if ( divisiors . containsKey ( Q ) ) { divisiors . put ( Q , divisiors . get ( Q ) + 1 ) ; } else { divisiors . put ( Q , 1 ) ; }
int ans = 0 ;
for ( Map . Entry < Integer , Integer > i : divisiors . entrySet ( ) ) { int frequency = i . getValue ( ) ; int temp = P ;
int cur = 0 ; while ( temp % i . getKey ( ) == 0 ) { temp /= i . getKey ( ) ;
cur ++ ; }
if ( cur < frequency ) { ans = P ; break ; } temp = P ;
for ( int j = cur ; j >= frequency ; j -- ) { temp /= i . getKey ( ) ; }
ans = Math . max ( temp , ans ) ; }
System . out . print ( ans ) ; }
int P = 10 , Q = 4 ;
findTheGreatestX ( P , Q ) ; } }
static String checkRearrangements ( int [ ] [ ] mat , int N , int M ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 1 ; j < M ; j ++ ) { if ( mat [ i ] [ 0 ] != mat [ i ] [ j ] ) { return " Yes " ; } } } return " No " ; }
static String nonZeroXor ( int [ ] [ ] mat , int N , int M ) { int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { res = res ^ mat [ i ] [ 0 ] ; }
if ( res != 0 ) return " Yes " ;
else return checkRearrangements ( mat , N , M ) ; }
int [ ] [ ] mat = { { 1 , 1 , 2 } , { 2 , 2 , 2 } , { 3 , 3 , 3 } } ; int N = mat . length ; int M = mat [ 0 ] . length ;
System . out . print ( nonZeroXor ( mat , N , M ) ) ; } }
import java . util . * ; class GFG { static final int size_int = 32 ;
static int functionMax ( int arr [ ] , int n ) {
Vector < Integer > [ ] setBit = new Vector [ 32 + 1 ] ; for ( int i = 0 ; i < setBit . length ; i ++ ) setBit [ i ] = new Vector < Integer > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < size_int ; j ++ ) {
if ( ( arr [ i ] & ( 1 << j ) ) > 0 )
setBit [ j ] . add ( i ) ; } }
for ( int i = size_int ; i >= 0 ; i -- ) { if ( setBit [ i ] . size ( ) == 1 ) {
swap ( arr , 0 , setBit [ i ] . get ( 0 ) ) ; break ; } }
int maxAnd = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { maxAnd = maxAnd & ( ~ arr [ i ] ) ; }
return maxAnd ; } static int [ ] swap ( int [ ] arr , int i , int j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; return arr ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 4 , 8 , 16 } ; int n = arr . length ;
System . out . print ( functionMax ( arr , n ) ) ; } }
static int nCr ( int n , int r ) {
int res = 1 ;
if ( r > n - r ) r = n - r ;
for ( int i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static int solve ( int n , int m , int k ) {
int sum = 0 ;
for ( int i = 0 ; i <= k ; i ++ ) sum += nCr ( n , i ) * nCr ( m , k - i ) ; return sum ; }
public static void main ( String [ ] args ) { int n = 3 , m = 2 , k = 2 ; System . out . print ( solve ( n , m , k ) ) ; } }
static int powerOptimised ( int a , int n ) {
int ans = 1 ; while ( n > 0 ) { int last_bit = ( n & 1 ) ;
if ( last_bit > 0 ) { ans = ans * a ; } a = a * a ;
n = n >> 1 ; } return ans ; }
public static void main ( String [ ] args ) { int a = 3 , n = 5 ; System . out . print ( powerOptimised ( a , n ) ) ; } }
static int findMaximumGcd ( int n ) {
int max_gcd = 1 ;
for ( int i = 1 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
if ( i > max_gcd ) max_gcd = i ; if ( ( n / i != i ) && ( n / i != n ) && ( ( n / i ) > max_gcd ) ) max_gcd = n / i ; } }
return max_gcd ; }
int N = 10 ;
System . out . print ( findMaximumGcd ( N ) ) ; } }
import java . util . * ; class GFG { static final int x = 2000021 ;
static int [ ] v = new int [ x ] ;
static void sieve ( ) { v [ 1 ] = 1 ;
for ( int i = 2 ; i < x ; i ++ ) v [ i ] = i ;
for ( int i = 4 ; i < x ; i += 2 ) v [ i ] = 2 ; for ( int i = 3 ; i * i < x ; i ++ ) {
if ( v [ i ] == i ) {
for ( int j = i * i ; j < x ; j += i ) {
if ( v [ j ] == j ) { v [ j ] = i ; } } } } }
static int prime_factors ( int n ) { HashSet < Integer > s = new HashSet < Integer > ( ) ; while ( n != 1 ) { s . add ( v [ n ] ) ; n = n / v [ n ] ; } return s . size ( ) ; }
static void distinctPrimes ( int m , int k ) {
Vector < Integer > result = new Vector < Integer > ( ) ; for ( int i = 14 ; i < m + k ; i ++ ) {
long count = prime_factors ( i ) ;
if ( count == k ) { result . add ( i ) ; } } int p = result . size ( ) ; for ( int index = 0 ; index < p - 1 ; index ++ ) { long element = result . get ( index ) ; int count = 1 , z = index ;
while ( z < p - 1 && count <= k && result . get ( z ) + 1 == result . get ( z + 1 ) ) {
count ++ ; z ++ ; }
if ( count >= k ) System . out . print ( element + " ▁ " ) ; } }
sieve ( ) ;
int N = 1000 , K = 3 ;
distinctPrimes ( N , K ) ; } }
static void print_product ( int a , int b , int c , int d ) {
int prod1 = a * c ; int prod2 = b * d ; int prod3 = ( a + b ) * ( c + d ) ;
int real = prod1 - prod2 ;
int imag = prod3 - ( prod1 + prod2 ) ;
System . out . println ( real + " ▁ + ▁ " + imag + " i " ) ; }
public static void main ( String [ ] args ) {
int a = 2 ; int b = 3 ; int c = 4 ; int d = 5 ;
print_product ( a , b , c , d ) ; } }
static boolean isInsolite ( int n ) { int N = n ;
int sum = 0 ;
int product = 1 ; while ( n != 0 ) {
int r = n % 10 ; sum = sum + r * r ; product = product * r * r ; n = n / 10 ; } return ( N % sum == 0 ) && ( N % product == 0 ) ; }
public static void main ( String [ ] args ) { int N = 111 ;
if ( isInsolite ( N ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static int sigma ( int n ) { if ( n == 1 ) return 1 ;
int result = 0 ;
for ( int i = 2 ; i <= Math . sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) {
if ( i == ( n / i ) ) result += i ; else result += ( i + n / i ) ; } }
return ( result + n + 1 ) ; }
static boolean isSuperabundant ( int N ) {
for ( double i = 1 ; i < N ; i ++ ) { double x = sigma ( ( int ) ( i ) ) / i ; double y = sigma ( ( int ) ( N ) ) / ( N * 1.0 ) ; if ( x > y ) return false ; } return true ; }
public static void main ( String [ ] args ) { int N = 4 ; if ( isSuperabundant ( N ) ) System . out . print ( "YesNEW_LINE"); else System . out . print ( "NoNEW_LINE"); } }
static boolean isDNum ( int n ) {
if ( n < 4 ) return false ; int numerator = 0 , hcf = 0 ;
for ( int k = 2 ; k <= n ; k ++ ) { numerator = ( int ) ( Math . pow ( k , n - 2 ) - k ) ; hcf = __gcd ( n , k ) ; }
if ( hcf == 1 && ( numerator % n ) != 0 ) return false ; return true ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
public static void main ( String [ ] args ) { int n = 15 ; boolean a = isDNum ( n ) ; if ( a ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static int Sum ( int N ) { int SumOfPrimeDivisors [ ] = new int [ N + 1 ] ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( SumOfPrimeDivisors [ i ] == 1 ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
static boolean RuthAaronNumber ( int n ) { if ( Sum ( n ) == Sum ( n + 1 ) ) return true ; else return false ; }
public static void main ( String [ ] args ) { int N = 714 ; if ( RuthAaronNumber ( N ) ) { System . out . print ( " Yes " ) ; } else { System . out . print ( " No " ) ; } } }
static int maxAdjacentDifference ( int N , int K ) {
if ( N == 1 ) { return 0 ; }
if ( N == 2 ) { return K ; }
return 2 * K ; }
public static void main ( String [ ] args ) { int N = 6 ; int K = 11 ; System . out . print ( maxAdjacentDifference ( N , K ) ) ; } }
class GFG { static final int mod = 1000000007 ;
public static int linearSum ( int n ) { return ( n * ( n + 1 ) / 2 ) % mod ; }
public static int rangeSum ( int b , int a ) { return ( linearSum ( b ) - linearSum ( a ) ) % mod ; }
public static int totalSum ( int n ) {
int result = 0 ; int i = 1 ;
while ( true ) {
result += rangeSum ( n / i , n / ( i + 1 ) ) * ( i % mod ) % mod ; result %= mod ; if ( i == n ) break ; i = n / ( n / ( i + 1 ) ) ; } return result ; }
public static void main ( String [ ] args ) { int N = 4 ; System . out . println ( totalSum ( N ) ) ; N = 12 ; System . out . println ( totalSum ( N ) ) ; } }
static boolean isDouble ( int num ) { String s = Integer . toString ( num ) ; int l = s . length ( ) ;
if ( s . charAt ( 0 ) == s . charAt ( 1 ) ) return false ;
if ( l % 2 == 1 ) { s = s + s . charAt ( 1 ) ; l ++ ; }
String s1 = s . substring ( 0 , l / 2 ) ;
String s2 = s . substring ( l / 2 ) ;
return s1 . equals ( s2 ) ; }
static boolean isNontrivialUndulant ( int N ) { return N > 100 && isDouble ( N ) ; }
public static void main ( String [ ] args ) { int n = 121 ; if ( isNontrivialUndulant ( n ) ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
static int MegagonNum ( int n ) { return ( 999998 * n * n - 999996 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . print ( MegagonNum ( n ) ) ; } }
import java . util . * ; class GFG { static final int mod = 1000000007 ;
static int productPairs ( int arr [ ] , int n ) {
int product = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) {
product *= ( arr [ i ] % mod * arr [ j ] % mod ) % mod ; product = product % mod ; } }
return product % mod ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 } ; int n = arr . length ; System . out . print ( productPairs ( arr , n ) ) ; } }
import java . util . * ; class GFG { static final int mod = 1000000007 ;
static int power ( int x , int y ) { int p = 1000000007 ;
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ; y = y >> 1 ; x = ( x * x ) % p ; }
return res ; }
static int productPairs ( int arr [ ] , int n ) {
int product = 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
product = ( product % mod * ( int ) power ( arr [ i ] , ( 2 * n ) ) % mod ) % mod ; } return product % mod ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 } ; int n = arr . length ; System . out . print ( productPairs ( arr , n ) ) ; } }
static void constructArray ( int N ) { int arr [ ] = new int [ N ] ;
for ( int i = 1 ; i <= N ; i ++ ) { arr [ i - 1 ] = i ; }
for ( int i = 0 ; i < N ; i ++ ) { System . out . print ( arr [ i ] + " , ▁ " ) ; } }
public static void main ( String [ ] args ) { int N = 6 ; constructArray ( N ) ; } }
static boolean isPrime ( int n ) { if ( n <= 1 ) return false ; for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
static int countSubsequences ( int arr [ ] , int n ) {
int totalSubsequence = ( int ) ( Math . pow ( 2 , n ) - 1 ) ; int countPrime = 0 , countOnes = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) countOnes ++ ; else if ( isPrime ( arr [ i ] ) ) countPrime ++ ; } int compositeSubsequence ;
int onesSequence = ( int ) ( Math . pow ( 2 , countOnes ) - 1 ) ;
compositeSubsequence = totalSubsequence - countPrime - onesSequence - onesSequence * countPrime ; return compositeSubsequence ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 1 , 2 } ; int n = arr . length ; System . out . print ( countSubsequences ( arr , n ) ) ; } }
static void checksum ( int n , int k ) {
float first_term = ( float ) ( ( ( 2 * n ) / k + ( 1 - k ) ) / 2.0 ) ;
if ( first_term - ( int ) ( first_term ) == 0 ) {
for ( int i = ( int ) first_term ; i <= first_term + k - 1 ; i ++ ) { System . out . print ( i + " ▁ " ) ; } } else System . out . print ( " - 1" ) ; }
public static void main ( String [ ] args ) { int n = 33 , k = 6 ; checksum ( n , k ) ; } }
static void sumEvenNumbers ( int N , int K ) { int check = N - 2 * ( K - 1 ) ;
if ( check > 0 && check % 2 == 0 ) { for ( int i = 0 ; i < K - 1 ; i ++ ) { System . out . print ( "2 ▁ " ) ; } System . out . println ( check ) ; } else { System . out . println ( " - 1" ) ; } }
public static void main ( String args [ ] ) { int N = 8 ; int K = 2 ; sumEvenNumbers ( N , K ) ; } }
public static int [ ] calculateWays ( int n ) { int x = 0 ;
int [ ] v = new int [ n ] ; for ( int i = 0 ; i < n ; i ++ ) v [ i ] = 0 ;
for ( int i = 0 ; i < n / 2 ; i ++ ) {
if ( n % 2 == 0 && i == n / 2 ) break ;
x = n * ( i + 1 ) - ( i + 1 ) * i ;
v [ i ] = x ; v [ n - i - 1 ] = x ; } return v ; }
public static void printArray ( int [ ] v ) { for ( int i = 0 ; i < v . length ; i ++ ) System . out . print ( v [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int [ ] v ; v = calculateWays ( 4 ) ; printArray ( v ) ; } }
class GFG { static final int MAXN = 10000000 ;
static int sumOfDigits ( int n ) {
int sum = 0 ; while ( n > 0 ) {
sum += n % 10 ;
n /= 10 ; } return sum ; }
static int smallestNum ( int X , int Y ) {
int res = - 1 ;
for ( int i = X ; i < MAXN ; i ++ ) {
int sum_of_digit = sumOfDigits ( i ) ;
if ( sum_of_digit % Y == 0 ) { res = i ; break ; } } return res ; }
public static void main ( String [ ] args ) { int X = 5923 , Y = 13 ; System . out . print ( smallestNum ( X , Y ) ) ; } }
static int countValues ( int N ) { Vector < Integer > div = new Vector < > ( ) ;
for ( int i = 2 ; i * i <= N ; i ++ ) {
if ( N % i == 0 ) { div . add ( i ) ;
if ( N != i * i ) { div . add ( N / i ) ; } } } int answer = 0 ;
for ( int i = 1 ; i * i <= N - 1 ; i ++ ) {
if ( ( N - 1 ) % i == 0 ) { if ( i * i == N - 1 ) answer ++ ; else answer += 2 ; } }
for ( int d : div ) { int K = N ; while ( K % d == 0 ) K /= d ; if ( ( K - 1 ) % d == 0 ) answer ++ ; } return answer ; }
public static void main ( String [ ] args ) { int N = 6 ; System . out . print ( countValues ( N ) ) ; } }
static void findMaxPrimeDivisor ( int n ) { int max_possible_prime = 0 ;
while ( n % 2 == 0 ) { max_possible_prime ++ ; n = n / 2 ; }
for ( int i = 3 ; i * i <= n ; i = i + 2 ) { while ( n % i == 0 ) { max_possible_prime ++ ; n = n / i ; } }
if ( n > 2 ) { max_possible_prime ++ ; } System . out . print ( max_possible_prime + "NEW_LINE"); }
public static void main ( String [ ] args ) { int n = 4 ;
findMaxPrimeDivisor ( n ) ; } }
static int CountWays ( int n ) { int ans = ( n - 1 ) / 2 ; return ans ; }
public static void main ( String [ ] args ) { int N = 8 ; System . out . print ( CountWays ( N ) ) ; } }
static void Solve ( int arr [ ] , int size , int n ) { int [ ] v = new int [ n + 1 ] ;
for ( int i = 0 ; i < size ; i ++ ) v [ arr [ i ] ] ++ ;
int max1 = - 1 , mx = - 1 ; for ( int i = 0 ; i < v . length ; i ++ ) { if ( v [ i ] > mx ) { mx = v [ i ] ; max1 = i ; } }
int cnt = 0 ; for ( int i : v ) { if ( i == 0 ) ++ cnt ; } int diff1 = n + 1 - cnt ;
int max_size = Math . max ( Math . min ( v [ max1 ] - 1 , diff1 ) , Math . min ( v [ max1 ] , diff1 - 1 ) ) ; System . out . println ( " Maximum ▁ size ▁ is : ▁ " + max_size ) ;
System . out . println ( " First ▁ Array ▁ is " ) ; for ( int i = 0 ; i < max_size ; i ++ ) { System . out . print ( max1 + " ▁ " ) ; v [ max1 ] -= 1 ; } System . out . println ( ) ;
System . out . println ( " The ▁ Second ▁ Array ▁ Is ▁ : " ) ; for ( int i = 0 ; i < ( n + 1 ) ; i ++ ) { if ( v [ i ] > 0 ) { System . out . print ( i + " ▁ " ) ; max_size -- ; } if ( max_size < 1 ) break ; } System . out . println ( ) ; }
int n = 7 ;
int arr [ ] = new int [ ] { 1 , 2 , 1 , 5 , 1 , 6 , 7 , 2 } ;
int size = arr . length ; Solve ( arr , size , n ) ; } }
static int power ( int x , int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
static int modInverse ( int n , int p ) { return power ( n , p - 2 , p ) ; }
static int nCrModPFermat ( int n , int r , int p ) {
if ( r == 0 ) return 1 ; if ( n < r ) return 0 ;
int fac [ ] = new int [ n + 1 ] ; fac [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fac [ i ] = fac [ i - 1 ] * i % p ; return ( fac [ n ] * modInverse ( fac [ r ] , p ) % p * modInverse ( fac [ n - r ] , p ) % p ) % p ; }
static int SumOfXor ( int a [ ] , int n ) { int mod = 10037 ; int answer = 0 ;
for ( int k = 0 ; k < 32 ; k ++ ) {
int x = 0 , y = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( ( a [ i ] & ( 1 << k ) ) != 0 ) x ++ ; else y ++ ; }
answer += ( ( 1 << k ) % mod * ( nCrModPFermat ( x , 3 , mod ) + x * nCrModPFermat ( y , 2 , mod ) ) % mod ) % mod ; } return answer ; }
public static void main ( String [ ] args ) { int n = 5 ; int A [ ] = { 3 , 5 , 2 , 18 , 7 } ; System . out . println ( SumOfXor ( A , n ) ) ; } }
class GFG { public static float round ( float var , int digit ) { float value = ( int ) ( var * Math . pow ( 10 , digit ) + .5 ) ; return ( float ) value / ( float ) Math . pow ( 10 , digit ) ; }
public static int probability ( int N ) {
int a = 2 ; int b = 3 ;
if ( N == 1 ) { return a ; } else if ( N == 2 ) { return b ; } else {
for ( int i = 3 ; i <= N ; i ++ ) { int c = a + b ; a = b ; b = c ; } return b ; } }
public static float operations ( int N ) {
int x = probability ( N ) ;
int y = ( int ) Math . pow ( 2 , N ) ; return round ( ( float ) x / ( float ) y , 2 ) ; }
public static void main ( String [ ] args ) { int N = 10 ; System . out . println ( ( operations ( N ) ) ) ; } }
static boolean isPerfectCube ( int x ) { long cr = Math . round ( Math . cbrt ( x ) ) ; return ( cr * cr * cr == x ) ; }
static void checkCube ( int a , int b ) {
String s1 = Integer . toString ( a ) ; String s2 = Integer . toString ( b ) ;
int c = Integer . parseInt ( s1 + s2 ) ;
if ( isPerfectCube ( c ) ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } }
public static void main ( String [ ] args ) { int a = 6 ; int b = 4 ; checkCube ( a , b ) ; } }
static int largest_sum ( int arr [ ] , int n ) {
int maximum = - 1 ;
HashMap < Integer , Integer > m = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( m . containsKey ( arr [ i ] ) ) { m . put ( arr [ i ] , m . get ( arr [ i ] ) + 1 ) ; } else { m . put ( arr [ i ] , 1 ) ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( m . get ( arr [ i ] ) > 1 ) { if ( m . containsKey ( 2 * arr [ i ] ) ) {
m . put ( 2 * arr [ i ] , m . get ( 2 * arr [ i ] ) + m . get ( arr [ i ] ) / 2 ) ; } else { m . put ( 2 * arr [ i ] , m . get ( arr [ i ] ) / 2 ) ; }
if ( 2 * arr [ i ] > maximum ) maximum = 2 * arr [ i ] ; } }
return maximum ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 2 , 4 , 7 , 8 } ; int n = arr . length ;
System . out . println ( largest_sum ( arr , n ) ) ; } }
static void canBeReduced ( int x , int y ) { int maxi = Math . max ( x , y ) ; int mini = Math . min ( x , y ) ;
if ( ( ( x + y ) % 3 ) == 0 && maxi <= 2 * mini ) System . out . print ( " YES " + "NEW_LINE"); else System . out . print ( " NO " + "NEW_LINE"); }
public static void main ( String [ ] args ) { int x = 6 , y = 9 ;
canBeReduced ( x , y ) ; } }
static void isPrime ( int N ) { boolean isPrime = true ;
int [ ] arr = { 7 , 11 , 13 , 17 , 19 , 23 , 29 , 31 } ;
if ( N < 2 ) { isPrime = false ; }
if ( N % 2 == 0 N % 3 == 0 N % 5 == 0 ) { isPrime = false ; }
for ( int i = 0 ; i < Math . sqrt ( N ) ; i += 30 ) {
for ( int c : arr ) {
if ( c > Math . sqrt ( N ) ) { break ; }
else { if ( N % ( c + i ) == 0 ) { isPrime = false ; break ; } }
if ( ! isPrime ) break ; } } if ( isPrime ) System . out . println ( " Prime ▁ Number " ) ; else System . out . println ( " Not ▁ a ▁ Prime ▁ Number " ) ; }
public static void main ( String args [ ] ) { int N = 121 ;
isPrime ( N ) ; } }
static void printPairs ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { System . out . print ( " ( " + arr [ i ] + " , ▁ " + arr [ j ] + " ) " + " , ▁ " ) ; } } }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 } ; int n = arr . length ; printPairs ( arr , n ) ; } }
import java . io . * ; class GFG { static void circle ( int x1 , int y1 , int x2 , int y2 , int r1 , int r2 ) { int distSq = ( int ) Math . sqrt ( ( ( x1 - x2 ) * ( x1 - x2 ) ) + ( ( y1 - y2 ) * ( y1 - y2 ) ) ) ; if ( distSq + r2 == r1 ) { System . out . println ( " The ▁ smaller ▁ circle ▁ lies ▁ completely " + " ▁ inside ▁ the ▁ bigger ▁ circle ▁ with ▁ " + " touching ▁ each ▁ other ▁ " + " at ▁ a ▁ point ▁ of ▁ circumference . ▁ " ) ; } else if ( distSq + r2 < r1 ) { System . out . println ( " The ▁ smaller ▁ circle ▁ lies ▁ completely " + " ▁ inside ▁ the ▁ bigger ▁ circle ▁ without " + " ▁ touching ▁ each ▁ other ▁ " + " at ▁ a ▁ point ▁ of ▁ circumference . " ) ; } else { System . out . println ( " The ▁ smaller ▁ does ▁ not ▁ lies ▁ inside " + " ▁ the ▁ bigger ▁ circle ▁ completely . " ) ; } }
public static void main ( String [ ] args ) { int x1 = 10 , y1 = 8 ; int x2 = 1 , y2 = 2 ; int r1 = 30 , r2 = 10 ; circle ( x1 , y1 , x2 , y2 , r1 , r2 ) ; } }
static void lengtang ( double r1 , double r2 , double d ) { System . out . println ( " The ▁ length ▁ of ▁ the ▁ direct " + " ▁ common ▁ tangent ▁ is ▁ " + ( Math . sqrt ( Math . pow ( d , 2 ) - Math . pow ( ( r1 - r2 ) , 2 ) ) ) ) ; }
public static void main ( String [ ] args ) { double r1 = 4 , r2 = 6 , d = 3 ; lengtang ( r1 , r2 , d ) ; } }
static void rad ( double d , double h ) { System . out . println ( " The ▁ radius ▁ of ▁ the ▁ circle ▁ is ▁ " + ( ( d * d ) / ( 8 * h ) + h / 2 ) ) ; }
public static void main ( String [ ] args ) { double d = 4 , h = 1 ; rad ( d , h ) ; } }
static void shortdis ( double r , double d ) { System . out . println ( " The ▁ shortest ▁ distance ▁ " + " from ▁ the ▁ chord ▁ to ▁ centre ▁ " + ( Math . sqrt ( ( r * r ) - ( ( d * d ) / 4 ) ) ) ) ; }
public static void main ( String [ ] args ) { double r = 4 , d = 3 ; shortdis ( r , d ) ; } }
static void lengtang ( double r1 , double r2 , double d ) { System . out . println ( " The ▁ length ▁ of ▁ the ▁ direct " + " ▁ common ▁ tangent ▁ is ▁ " + ( Math . sqrt ( Math . pow ( d , 2 ) - Math . pow ( ( r1 - r2 ) , 2 ) ) ) ) ; }
public static void main ( String [ ] args ) { double r1 = 4 , r2 = 6 , d = 12 ; lengtang ( r1 , r2 , d ) ; } }
static double square ( double a ) {
if ( a < 0 ) return - 1 ;
double x = 0.464 * a ; return x ; }
public static void main ( String [ ] args ) { double a = 5 ; System . out . println ( square ( a ) ) ; } }
double polyapothem ( double n , double a ) {
if ( a < 0 && n < 0 ) return - 1 ;
return ( a / ( 2 * java . lang . Math . tan ( ( 180 / n ) * 3.14159 / 180 ) ) ) ; }
public static void main ( String args [ ] ) { double a = 9 , n = 6 ; GFG g = new GFG ( ) ; System . out . println ( g . polyapothem ( n , a ) ) ; } }
static float polyarea ( float n , float a ) {
if ( a < 0 && n < 0 ) return - 1 ;
float A = ( a * a * n ) / ( float ) ( 4 * Math . tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; return A ; }
public static void main ( String [ ] args ) { float a = 9 , n = 6 ; System . out . println ( polyarea ( n , a ) ) ; } }
static double calculateSide ( double n , double r ) { double theta , theta_in_radians ; theta = 360 / n ; theta_in_radians = theta * 3.14 / 180 ; return 2 * r * Math . sin ( theta_in_radians / 2 ) ; }
double n = 3 ;
double r = 5 ; System . out . println ( calculateSide ( n , r ) ) ; } }
static float cyl ( float r , float R , float h ) {
if ( h < 0 && r < 0 && R < 0 ) return - 1 ;
float r1 = r ;
float h1 = h ;
float V = ( float ) ( 3.14 * Math . pow ( r1 , 2 ) * h1 ) ; return V ; }
public static void main ( String [ ] args ) { float r = 7 , R = 11 , h = 6 ; System . out . print ( cyl ( r , R , h ) ) ; } }
static double Perimeter ( double s , int n ) { double perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
int n = 5 ;
double s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; System . out . println ( " Perimeter ▁ of ▁ Regular ▁ Polygon " + " ▁ with ▁ " + n + " ▁ sides ▁ of ▁ length ▁ " + s + " ▁ = ▁ " + peri ) ; } }
static float rhombusarea ( float l , float b ) {
if ( l < 0 b < 0 ) return - 1 ;
return ( l * b ) / 2 ; }
public static void main ( String [ ] args ) { float l = 16 , b = 6 ; System . out . println ( rhombusarea ( l , b ) ) ; } }
static boolean FindPoint ( int x1 , int y1 , int x2 , int y2 , int x , int y ) { if ( x > x1 && x < x2 && y > y1 && y < y2 ) return true ; return false ; }
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x = 1 , y = 5 ;
if ( FindPoint ( x1 , y1 , x2 , y2 , x , y ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static void shortest_distance ( float x1 , float y1 , float z1 , float a , float b , float c , float d ) { d = Math . abs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; float e = ( float ) Math . sqrt ( a * a + b * b + c * c ) ; System . out . println ( " Perpendicular ▁ distance ▁ " + " is ▁ " + d / e ) ; }
public static void main ( String [ ] args ) { float x1 = 4 ; float y1 = - 4 ; float z1 = 3 ; float a = 2 ; float b = - 2 ; float c = 5 ; float d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ; } }
static float findVolume ( float l , float b , float h ) {
float volume = ( l * b * h ) / 2 ; return volume ; }
public static void main ( String [ ] args ) { float l = 18 , b = 12 , h = 9 ;
System . out . println ( " Volume ▁ of ▁ triangular ▁ prism : ▁ " + findVolume ( l , b , h ) ) ; } }
static boolean isRectangle ( int a , int b , int c , int d ) {
if ( a == b && a == c && a == d && c == d && b == c && b == d ) return true ; else if ( a == b && c == d ) return true ; else if ( a == d && c == b ) return true ; else if ( a == c && d == b ) return true ; else return false ; }
public static void main ( String [ ] args ) { int a = 1 , b = 2 , c = 3 , d = 4 ; if ( isRectangle ( a , b , c , d ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static void midpoint ( int x1 , int x2 , int y1 , int y2 ) { System . out . print ( ( x1 + x2 ) / 2 + " ▁ , ▁ " + ( y1 + y2 ) / 2 ) ; }
public static void main ( String [ ] args ) { int x1 = - 1 , y1 = 2 ; int x2 = 3 , y2 = - 6 ; midpoint ( x1 , x2 , y1 , y2 ) ; } }
static double arcLength ( double diameter , double angle ) { double pi = 22.0 / 7.0 ; double arc ; if ( angle >= 360 ) { System . out . println ( " Angle ▁ cannot " + " ▁ be ▁ formed " ) ; return 0 ; } else { arc = ( pi * diameter ) * ( angle / 360.0 ) ; return arc ; } }
public static void main ( String args [ ] ) { double diameter = 25.0 ; double angle = 45.0 ; double arc_len = arcLength ( diameter , angle ) ; System . out . println ( arc_len ) ; } }
import java . io . * ; class GFG { static void checkCollision ( int a , int b , int c , int x , int y , int radius ) {
double dist = ( Math . abs ( a * x + b * y + c ) ) / Math . sqrt ( a * a + b * b ) ;
if ( radius == dist ) System . out . println ( " Touch " ) ; else if ( radius > dist ) System . out . println ( " Intersect " ) ; else System . out . println ( " Outside " ) ; }
public static void main ( String [ ] args ) { int radius = 5 ; int x = 0 , y = 0 ; int a = 3 , b = 4 , c = 25 ; checkCollision ( a , b , c , x , y , radius ) ; } }
static double polygonArea ( double X [ ] , double Y [ ] , int n ) {
double area = 0.0 ;
int j = n - 1 ; for ( int i = 0 ; i < n ; i ++ ) { area += ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) ;
j = i ; }
return Math . abs ( area / 2.0 ) ; }
public static void main ( String [ ] args ) { double X [ ] = { 0 , 2 , 4 } ; double Y [ ] = { 1 , 3 , 7 } ; int n = X . length ; System . out . println ( polygonArea ( X , Y , n ) ) ; } }
static int chk ( int n ) {
Vector < Integer > v = new Vector < Integer > ( ) ; while ( n != 0 ) { v . add ( n % 2 ) ; n = n / 2 ; } for ( int i = 0 ; i < v . size ( ) ; i ++ ) { if ( v . get ( i ) == 1 ) { return ( int ) Math . pow ( 2 , i ) ; } } return 0 ; }
static void sumOfLSB ( int arr [ ] , int N ) {
Vector < Integer > lsb_arr = new Vector < Integer > ( ) ; for ( int i = 0 ; i < N ; i ++ ) {
lsb_arr . add ( chk ( arr [ i ] ) ) ; }
Collections . sort ( lsb_arr ) ; int ans = 0 ; for ( int i = 0 ; i < N - 1 ; i += 2 ) {
ans += ( lsb_arr . get ( i + 1 ) ) ; }
System . out . print ( ans ) ; }
public static void main ( String [ ] args ) { int N = 5 ; int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ;
sumOfLSB ( arr , N ) ; } }
static int countSubsequences ( int arr [ ] , int N ) {
int odd = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( ( arr [ i ] & 1 ) % 2 == 1 ) odd ++ ; }
return ( 1 << odd ) - 1 ; }
public static void main ( String [ ] args ) { int N = 3 ; int arr [ ] = { 1 , 3 , 3 } ;
System . out . println ( countSubsequences ( arr , N ) ) ; } }
static int getPairsCount ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = arr [ i ] - ( i % arr [ i ] ) ; j < n ; j += arr [ i ] ) {
if ( i < j && Math . abs ( arr [ i ] - arr [ j ] ) >= Math . min ( arr [ i ] , arr [ j ] ) ) { count ++ ; } } }
return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 2 , 3 } ; int N = arr . length ; System . out . println ( getPairsCount ( arr , N ) ) ; } }
static void check ( int N ) { int twos = 0 , fives = 0 ;
while ( N % 2 == 0 ) { N /= 2 ; twos ++ ; }
while ( N % 5 == 0 ) { N /= 5 ; fives ++ ; } if ( N == 1 && twos <= fives ) { System . out . println ( 2 * fives - twos ) ; } else { System . out . println ( - 1 ) ; } }
public static void main ( String [ ] args ) { int N = 50 ; check ( N ) ; } }
static void rangeSum ( int arr [ ] , int N , int L , int R ) {
int sum = 0 ;
for ( int i = L - 1 ; i < R ; i ++ ) { sum += arr [ i % N ] ; }
System . out . println ( sum ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 6 , 9 } ; int L = 10 , R = 13 ; int N = arr . length ; rangeSum ( arr , N , L , R ) ; } }
static void rangeSum ( int arr [ ] , int N , int L , int R ) {
int prefix [ ] = new int [ N + 1 ] ; prefix [ 0 ] = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] + arr [ i - 1 ] ; }
int leftsum = ( ( L - 1 ) / N ) * prefix [ N ] + prefix [ ( L - 1 ) % N ] ;
int rightsum = ( R / N ) * prefix [ N ] + prefix [ R % N ] ;
System . out . print ( rightsum - leftsum ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 6 , 9 } ; int L = 10 , R = 13 ; int N = arr . length ; rangeSum ( arr , N , L , R ) ; } }
static int ExpoFactorial ( int N ) {
int res = 1 ; int mod = 1000000007 ;
for ( int i = 2 ; i < N + 1 ; i ++ )
res = ( int ) Math . pow ( i , res ) % mod ;
return res ; }
int N = 4 ;
System . out . println ( ( ExpoFactorial ( N ) ) ) ; } }
static int maxSubArraySumRepeated ( int [ ] arr , int N , int K ) {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) sum += arr [ i ] ; int curr = arr [ 0 ] ;
int ans = arr [ 0 ] ;
if ( K == 1 ) {
for ( int i = 1 ; i < N ; i ++ ) { curr = Math . max ( arr [ i ] , curr + arr [ i ] ) ; ans = Math . max ( ans , curr ) ; }
return ans ; }
ArrayList < Integer > V = new ArrayList < Integer > ( ) ;
for ( int i = 0 ; i < 2 * N ; i ++ ) { V . add ( arr [ i % N ] ) ; }
int maxSuf = V . get ( 0 ) ;
int maxPref = V . get ( 2 * N - 1 ) ; curr = V . get ( 0 ) ; for ( int i = 1 ; i < 2 * N ; i ++ ) { curr += V . get ( i ) ; maxPref = Math . max ( maxPref , curr ) ; } curr = V . get ( 2 * N - 1 ) ; for ( int i = 2 * N - 2 ; i >= 0 ; i -- ) { curr += V . get ( i ) ; maxSuf = Math . max ( maxSuf , curr ) ; } curr = V . get ( 0 ) ;
for ( int i = 1 ; i < 2 * N ; i ++ ) { curr = Math . max ( V . get ( i ) , curr + V . get ( i ) ) ; ans = Math . max ( ans , curr ) ; }
if ( sum > 0 ) { int temp = sum * ( K - 2 ) ; ans = Math . max ( ans , Math . max ( temp + maxPref , temp + maxSuf ) ) ; }
return ans ; }
int [ ] arr = { 10 , 20 , - 30 , - 1 , 40 } ; int N = arr . length ; int K = 10 ;
System . out . print ( maxSubArraySumRepeated ( arr , N , K ) ) ; } }
public static void countSubarray ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i ; j < n ; j ++ ) {
int mxSubarray = 0 ;
int mxOther = 0 ;
for ( int k = i ; k <= j ; k ++ ) { mxSubarray = Math . max ( mxSubarray , arr [ k ] ) ; }
for ( int k = 0 ; k < i ; k ++ ) { mxOther = Math . max ( mxOther , arr [ k ] ) ; } for ( int k = j + 1 ; k < n ; k ++ ) { mxOther = Math . max ( mxOther , arr [ k ] ) ; }
if ( mxSubarray > ( 2 * mxOther ) ) count ++ ; } }
System . out . println ( count ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 6 , 10 , 9 , 7 , 3 } ; int N = arr . length ; countSubarray ( arr , N ) ; } }
static void countSubarray ( int [ ] arr , int n ) { int L = 0 , R = 0 ;
int mx = Integer . MIN_VALUE ; for ( int i = 0 ; i < n ; i ++ ) mx = Math . max ( mx , arr [ i ] ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] * 2 > mx ) {
L = i ; break ; } } for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( arr [ i ] * 2 > mx ) {
R = i ; break ; } }
System . out . println ( ( L + 1 ) * ( n - R ) ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 6 , 10 , 9 , 7 , 3 } ; int N = arr . length ; countSubarray ( arr , N ) ; } }
static boolean isPrime ( int X ) { for ( int i = 2 ; i * i <= X ; i ++ )
if ( X % i == 0 ) return false ; return true ; }
static void printPrimes ( int A [ ] , int N ) {
for ( int i = 0 ; i < N ; i ++ ) {
for ( int j = A [ i ] - 1 ; ; j -- ) {
if ( isPrime ( j ) ) { System . out . print ( j + " ▁ " ) ; break ; } }
for ( int j = A [ i ] + 1 ; ; j ++ ) {
if ( isPrime ( j ) ) { System . out . print ( j + " ▁ " ) ; break ; } } System . out . println ( ) ; } }
int A [ ] = { 17 , 28 } ; int N = A . length ;
printPrimes ( A , N ) ; } }
static int KthSmallest ( int A [ ] , int B [ ] , int N , int K ) { int M = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { M = Math . max ( A [ i ] , M ) ; }
int freq [ ] = new int [ M + 1 ] ;
for ( int i = 0 ; i < N ; i ++ ) { freq [ A [ i ] ] += B [ i ] ; }
int sum = 0 ;
for ( int i = 0 ; i <= M ; i ++ ) {
sum += freq [ i ] ;
if ( sum >= K ) {
return i ; } }
return - 1 ; }
int A [ ] = { 3 , 4 , 5 } ; int B [ ] = { 2 , 1 , 3 } ; int N = A . length ; int K = 4 ;
System . out . println ( KthSmallest ( A , B , N , K ) ) ; } }
static void findbitwiseOR ( int [ ] a , int n ) {
int res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int curr_sub_array = a [ i ] ;
res = res | curr_sub_array ; for ( int j = i ; j < n ; j ++ ) {
curr_sub_array = curr_sub_array & a [ j ] ; res = res | curr_sub_array ; } }
System . out . println ( res ) ; }
public static void main ( String [ ] args ) { int A [ ] = { 1 , 2 , 3 } ; int N = A . length ; findbitwiseOR ( A , N ) ; } }
static void findbitwiseOR ( int [ ] a , int n ) {
int res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) res = res | a [ i ] ;
System . out . println ( res ) ; }
public static void main ( String [ ] args ) { int [ ] A = { 1 , 2 , 3 } ; int N = A . length ; findbitwiseOR ( A , N ) ; } }
static void check ( int n ) {
int sumOfDigit = 0 ; int prodOfDigit = 1 ; while ( n > 0 ) {
int rem ; rem = n % 10 ;
sumOfDigit += rem ;
prodOfDigit *= rem ;
n /= 10 ; }
if ( sumOfDigit > prodOfDigit ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; }
public static void main ( String [ ] args ) { int N = 1234 ; check ( N ) ; } }
static void evenOddBitwiseXOR ( int N ) { System . out . print ( " Even : ▁ " + 0 + " ▁ " ) ;
for ( int i = 4 ; i <= N ; i = i + 4 ) { System . out . print ( i + " ▁ " ) ; } System . out . print ( "NEW_LINE"); System . out . print ( " Odd : ▁ " + 1 + " ▁ " ) ;
for ( int i = 4 ; i <= N ; i = i + 4 ) { System . out . print ( i - 1 + " ▁ " ) ; } if ( N % 4 == 2 ) System . out . print ( N + 1 ) ; else if ( N % 4 == 3 ) System . out . print ( N ) ; }
public static void main ( String [ ] args ) { int N = 6 ; evenOddBitwiseXOR ( N ) ; } }
static void findPermutation ( int [ ] arr ) { int N = arr . length ; int i = N - 2 ;
while ( i >= 0 && arr [ i ] <= arr [ i + 1 ] ) i -- ;
if ( i == - 1 ) { System . out . print ( " - 1" ) ; return ; } int j = N - 1 ;
while ( j > i && arr [ j ] >= arr [ i ] ) j -- ;
while ( j > i && arr [ j ] == arr [ j - 1 ] ) {
j -- ; }
int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ;
for ( int it : arr ) { System . out . print ( it + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 5 , 3 , 4 , 6 } ; findPermutation ( arr ) ; } }
static void sieveOfEratosthenes ( int N , int s [ ] ) {
boolean [ ] prime = new boolean [ N + 1 ] ;
for ( int i = 2 ; i <= N ; i += 2 ) s [ i ] = 2 ;
for ( int i = 3 ; i <= N ; i += 2 ) {
if ( prime [ i ] == false ) { s [ i ] = i ;
for ( int j = i ; j * i <= N ; j += 2 ) {
if ( ! prime [ i * j ] ) { prime [ i * j ] = true ; s [ i * j ] = i ; } } } } }
static void findDifference ( int N ) {
int [ ] s = new int [ N + 1 ] ;
sieveOfEratosthenes ( N , s ) ;
int total = 1 , odd = 1 , even = 0 ;
int curr = s [ N ] ;
int cnt = 1 ;
while ( N > 1 ) { N /= s [ N ] ;
if ( curr == s [ N ] ) { cnt ++ ; continue ; }
if ( curr == 2 ) { total = total * ( cnt + 1 ) ; }
else { total = total * ( cnt + 1 ) ; odd = odd * ( cnt + 1 ) ; }
curr = s [ N ] ; cnt = 1 ; }
even = total - odd ;
System . out . print ( Math . abs ( even - odd ) ) ; }
public static void main ( String [ ] args ) { int N = 12 ; findDifference ( N ) ; } }
static void findMedian ( int Mean , int Mode ) {
double Median = ( 2 * Mean + Mode ) / 3.0 ;
System . out . print ( ( int ) Median ) ; }
public static void main ( String [ ] args ) { int mode = 6 , mean = 3 ; findMedian ( mean , mode ) ; } }
private static double vectorMagnitude ( int x , int y , int z ) {
int sum = x * x + y * y + z * z ;
return Math . sqrt ( sum ) ; }
public static void main ( String [ ] args ) { int x = 1 ; int y = 2 ; int z = 3 ; System . out . print ( vectorMagnitude ( x , y , z ) ) ; } }
static long multiplyByMersenne ( long N , long M ) {
long x = ( int ) ( Math . log ( M + 1 ) / Math . log ( 2 ) ) ;
return ( ( N << x ) - N ) ; }
public static void main ( String [ ] args ) { long N = 4 ; long M = 15 ; System . out . print ( multiplyByMersenne ( N , M ) ) ; } }
static int perfectSquare ( int num ) {
int sr = ( int ) ( Math . sqrt ( num ) ) ;
int a = sr * sr ; int b = ( sr + 1 ) * ( sr + 1 ) ;
if ( ( num - a ) < ( b - num ) ) { return a ; } else { return b ; } }
static int powerOfTwo ( int num ) {
int lg = ( int ) ( Math . log ( num ) / Math . log ( 2 ) ) ;
int p = ( int ) ( Math . pow ( 2 , lg ) ) ; return p ; }
static void uniqueElement ( int arr [ ] , int N ) { boolean ans = true ;
HashMap < Integer , Integer > freq = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { if ( freq . containsKey ( arr [ i ] ) ) { freq . put ( arr [ i ] , freq . get ( arr [ i ] ) + 1 ) ; } else { freq . put ( arr [ i ] , 1 ) ; } }
for ( Map . Entry < Integer , Integer > el : freq . entrySet ( ) ) {
if ( el . getValue ( ) == 1 ) { ans = false ;
int ps = perfectSquare ( el . getKey ( ) ) ;
System . out . print ( powerOfTwo ( ps ) + " ▁ " ) ; } }
if ( ans ) System . out . print ( " - 1" ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 11 , 4 , 3 , 4 } ; int N = arr . length ; uniqueElement ( arr , N ) ; } }
static void partitionArray ( int a [ ] , int n ) {
int min [ ] = new int [ n ] ;
int mini = Integer . MAX_VALUE ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
mini = Math . min ( mini , a [ i ] ) ;
min [ i ] = mini ; }
int maxi = Integer . MIN_VALUE ;
int ind = - 1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
maxi = Math . max ( maxi , a [ i ] ) ;
if ( maxi < min [ i + 1 ] ) {
ind = i ;
break ; } }
if ( ind != - 1 ) {
for ( int i = 0 ; i <= ind ; i ++ ) System . out . print ( a [ i ] + " ▁ " ) ; System . out . println ( ) ;
for ( int i = ind + 1 ; i < n ; i ++ ) System . out . print ( a [ i ] + " ▁ " ) ; }
else System . out . println ( " Impossible " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 3 , 2 , 7 , 9 } ; int N = arr . length ; partitionArray ( arr , N ) ; } }
static int countPrimeFactors ( int n ) { int count = 0 ;
while ( n % 2 == 0 ) { n = n / 2 ; count ++ ; }
for ( int i = 3 ; i <= ( int ) Math . sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { n = n / i ; count ++ ; } }
if ( n > 2 ) count ++ ; return ( count ) ; }
static int findSum ( int n ) {
int sum = 0 ; for ( int i = 1 , num = 2 ; i <= n ; num ++ ) {
if ( countPrimeFactors ( num ) == 2 ) { sum += num ;
i ++ ; } } return sum ; }
static void check ( int n , int k ) {
int s = findSum ( k - 1 ) ;
if ( s >= n ) System . out . print ( " No " ) ;
else System . out . print ( " Yes " ) ; }
public static void main ( String [ ] args ) { int n = 100 , k = 6 ; check ( n , k ) ; } }
static int gcd ( int a , int b ) {
while ( b > 0 ) { int rem = a % b ; a = b ; b = rem ; }
return a ; }
static int countNumberOfWays ( int n ) {
if ( n == 1 ) return - 1 ;
int g = 0 ; int power = 0 ;
while ( n % 2 == 0 ) { power ++ ; n /= 2 ; } g = gcd ( g , power ) ;
for ( int i = 3 ; i <= ( int ) Math . sqrt ( n ) ; i += 2 ) { power = 0 ;
while ( n % i == 0 ) { power ++ ; n /= i ; } g = gcd ( g , power ) ; }
if ( n > 2 ) g = gcd ( g , 1 ) ;
int ways = 1 ;
power = 0 ; while ( g % 2 == 0 ) { g /= 2 ; power ++ ; }
ways *= ( power + 1 ) ;
for ( int i = 3 ; i <= ( int ) Math . sqrt ( g ) ; i += 2 ) { power = 0 ;
while ( g % i == 0 ) { power ++ ; g /= i ; }
ways *= ( power + 1 ) ; }
if ( g > 2 ) ways *= 2 ;
return ways ; }
public static void main ( String [ ] args ) { int N = 64 ; System . out . print ( countNumberOfWays ( N ) ) ; } }
static int powOfPositive ( int n ) {
int pos = ( int ) Math . floor ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ; return ( int ) Math . pow ( 2 , pos ) ; }
static int powOfNegative ( int n ) {
int pos = ( int ) Math . ceil ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ; return ( int ) ( - 1 * Math . pow ( 2 , pos ) ) ; }
static void highestPowerOf2 ( int n ) {
if ( n > 0 ) { System . out . println ( powOfPositive ( n ) ) ; } else {
n = - n ; System . out . println ( powOfNegative ( n ) ) ; } }
public static void main ( String [ ] args ) { int n = - 24 ; highestPowerOf2 ( n ) ; } }
public static int noOfCards ( int n ) { return n * ( 3 * n + 1 ) / 2 ; }
public static void main ( String args [ ] ) { int n = 3 ; System . out . print ( noOfCards ( n ) ) ; } }
static String smallestPoss ( String s , int n ) {
String ans = " " ;
int arr [ ] = new int [ 10 ] ;
for ( int i = 0 ; i < n ; i ++ ) { arr [ s . charAt ( i ) - 48 ] ++ ; }
for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < arr [ i ] ; j ++ ) ans = ans + String . valueOf ( i ) ; }
return ans ; }
public static void main ( String [ ] args ) { int N = 15 ; String K = "325343273113434" ; System . out . print ( smallestPoss ( K , N ) ) ; } }
static int Count_subarray ( int arr [ ] , int n ) { int subarray_sum , remaining_sum , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i ; j < n ; j ++ ) {
subarray_sum = 0 ; remaining_sum = 0 ;
for ( int k = i ; k <= j ; k ++ ) { subarray_sum += arr [ k ] ; }
for ( int l = 0 ; l < i ; l ++ ) { remaining_sum += arr [ l ] ; } for ( int l = j + 1 ; l < n ; l ++ ) { remaining_sum += arr [ l ] ; }
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 10 , 9 , 12 , 6 } ; int n = arr . length ; System . out . print ( Count_subarray ( arr , n ) ) ; } }
class GFG { static int Count_subarray ( int arr [ ] , int n ) { int total_sum = 0 , subarray_sum , remaining_sum , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { total_sum += arr [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
subarray_sum = 0 ;
for ( int j = i ; j < n ; j ++ ) {
subarray_sum += arr [ j ] ; remaining_sum = total_sum - subarray_sum ;
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 10 , 9 , 12 , 6 } ; int n = arr . length ; System . out . print ( Count_subarray ( arr , n ) ) ; } }
static int maxXOR ( int arr [ ] , int n ) {
int xorArr = 0 ; for ( int i = 0 ; i < n ; i ++ ) xorArr ^= arr [ i ] ;
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) ans = Math . max ( ans , ( xorArr ^ arr [ i ] ) ) ;
return ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 1 , 3 } ; int n = arr . length ; System . out . println ( maxXOR ( arr , n ) ) ; } }
static boolean digitDividesK ( int num , int k ) { while ( num != 0 ) {
int d = num % 10 ;
if ( d != 0 && k % d == 0 ) return true ;
num = num / 10 ; }
return false ; }
static int findCount ( int l , int r , int k ) {
int count = 0 ;
for ( int i = l ; i <= r ; i ++ ) {
if ( digitDividesK ( i , k ) ) count ++ ; } return count ; }
public static void main ( String [ ] args ) { int l = 20 , r = 35 ; int k = 45 ; System . out . println ( findCount ( l , r , k ) ) ; } }
static boolean isFactorial ( int n ) { for ( int i = 1 ; ; i ++ ) { if ( n % i == 0 ) { n /= i ; } else { break ; } } if ( n == 1 ) { return true ; } else { return false ; } }
public static void main ( String [ ] args ) { int n = 24 ; boolean ans = isFactorial ( n ) ; if ( ans == true ) { System . out . println ( " Yes " ) ; } else { System . out . println ( " No " ) ; } } }
static int lcm ( int a , int b ) { int GCD = __gcd ( a , b ) ; return ( a * b ) / GCD ; }
static int MinLCM ( int a [ ] , int n ) {
int [ ] Prefix = new int [ n + 2 ] ; int [ ] Suffix = new int [ n + 2 ] ;
Prefix [ 1 ] = a [ 0 ] ; for ( int i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = lcm ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( int i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = lcm ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
int ans = Math . min ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( int i = 2 ; i < n ; i += 1 ) { ans = Math . min ( ans , lcm ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
public static void main ( String [ ] args ) { int a [ ] = { 5 , 15 , 9 , 36 } ; int n = a . length ; System . out . println ( MinLCM ( a , n ) ) ; } }
static int count ( int n ) { return n * ( 3 * n - 1 ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . println ( count ( n ) ) ; } }
static int findMinValue ( int arr [ ] , int n ) {
long sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
return ( ( int ) ( sum / n ) + 1 ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 4 , 2 , 1 , 10 , 6 } ; int n = arr . length ; System . out . print ( findMinValue ( arr , n ) ) ; } }
class GFG { static final int MOD = 1000000007 ;
static int modFact ( int n , int m ) { int result = 1 ; for ( int i = 1 ; i <= m ; i ++ ) result = ( result * i ) % MOD ; return result ; }
public static void main ( String [ ] args ) { int n = 3 , m = 2 ; System . out . println ( modFact ( n , m ) ) ; } }
class GFG { static final int mod = ( int ) ( 1e9 + 7 ) ;
static long power ( int p ) { long res = 1 ; for ( int i = 1 ; i <= p ; ++ i ) { res *= 2 ; res %= mod ; } return res % mod ; }
static long subset_square_sum ( int A [ ] ) { int n = A . length ; long ans = 0 ;
for ( int i : A ) { ans += ( 1 * i * i ) % mod ; ans %= mod ; } return ( 1 * ans * power ( n - 1 ) ) % mod ; }
public static void main ( String [ ] args ) { int A [ ] = { 3 , 7 } ; System . out . println ( subset_square_sum ( A ) ) ; } }
class GFG { static int N = 100050 ; static int [ ] lpf = new int [ N ] ; static int [ ] mobius = new int [ N ] ;
static void least_prime_factor ( ) { for ( int i = 2 ; i < N ; i ++ )
if ( lpf [ i ] == 0 ) for ( int j = i ; j < N ; j += i )
if ( lpf [ j ] == 0 ) lpf [ j ] = i ; }
static void Mobius ( ) { for ( int i = 1 ; i < N ; i ++ ) {
if ( i == 1 ) mobius [ i ] = 1 ; else {
if ( lpf [ i / lpf [ i ] ] == lpf [ i ] ) mobius [ i ] = 0 ;
else mobius [ i ] = - 1 * mobius [ i / lpf [ i ] ] ; } } }
static int gcd_pairs ( int a [ ] , int n ) {
int maxi = 0 ;
int [ ] fre = new int [ N ] ;
for ( int i = 0 ; i < n ; i ++ ) { fre [ a [ i ] ] ++ ; maxi = Math . max ( a [ i ] , maxi ) ; } least_prime_factor ( ) ; Mobius ( ) ;
int ans = 0 ;
for ( int i = 1 ; i <= maxi ; i ++ ) { if ( mobius [ i ] == 0 ) continue ; int temp = 0 ; for ( int j = i ; j <= maxi ; j += i ) temp += fre [ j ] ; ans += temp * ( temp - 1 ) / 2 * mobius [ i ] ; }
return ans ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = a . length ;
System . out . print ( gcd_pairs ( a , n ) ) ; } }
static void compareVal ( int x , int y ) {
double a = y * Math . log ( x ) ; double b = x * Math . log ( y ) ;
if ( a > b ) System . out . print ( x + " ^ " + y + " ▁ > ▁ " + y + " ^ " + x ) ; else if ( a < b ) System . out . print ( x + " ^ " + y + " ▁ < ▁ " + y + " ^ " + x ) ; else if ( a == b ) System . out . print ( x + " ^ " + y + " ▁ = ▁ " + y + " ^ " + x ) ; }
public static void main ( String [ ] args ) { int x = 4 , y = 5 ; compareVal ( x , y ) ; } }
static void ZigZag ( int n ) {
long [ ] fact = new long [ n + 1 ] ; long [ ] zig = new long [ n + 1 ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) zig [ i ] = 0 ;
fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
zig [ 0 ] = 1 ; zig [ 1 ] = 1 ; System . out . print ( " zig ▁ zag ▁ numbers : ▁ " ) ;
System . out . print ( zig [ 0 ] + " ▁ " + zig [ 1 ] + " ▁ " ) ;
for ( int i = 2 ; i < n ; i ++ ) { long sum = 0 ; for ( int k = 0 ; k <= i - 1 ; k ++ ) {
sum += ( fact [ i - 1 ] / ( fact [ i - 1 - k ] * fact [ k ] ) ) * zig [ k ] * zig [ i - 1 - k ] ; }
zig [ i ] = sum / 2 ;
System . out . print ( sum / 2 + " ▁ " ) ; } }
public static void main ( String [ ] args ) throws java . lang . Exception { int n = 10 ;
ZigZag ( n ) ; } }
static int find_count ( Vector < Integer > ele ) {
int count = 0 ; for ( int i = 0 ; i < ele . size ( ) ; i ++ ) {
Vector < Integer > p = new Vector < Integer > ( ) ;
int c = 0 ;
for ( int j = ele . size ( ) - 1 ; j >= ( ele . size ( ) - 1 - i ) && j >= 0 ; j -- ) { p . add ( ele . get ( j ) ) ; } int j = ele . size ( ) - 1 , k = 0 ;
while ( j >= 0 ) {
if ( ele . get ( j ) != p . get ( k ) ) { break ; } j -- ; k ++ ;
if ( k == p . size ( ) ) { c ++ ; k = 0 ; } } count = Math . max ( count , c ) ; }
return count ; }
static void solve ( int n ) {
int count = 1 ;
Vector < Integer > ele = new Vector < Integer > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( count + " , ▁ " ) ;
ele . add ( count ) ;
count = find_count ( ele ) ; } }
public static void main ( String [ ] args ) { int n = 10 ; solve ( n ) ; } }
static HashMap < Integer , Integer > store = new HashMap < Integer , Integer > ( ) ;
static int Wedderburn ( int n ) {
if ( n <= 2 ) return store . get ( n ) ;
else if ( n % 2 == 0 ) {
int x = n / 2 , ans = 0 ;
for ( int i = 1 ; i < x ; i ++ ) { ans += store . get ( i ) * store . get ( n - i ) ; }
ans += ( store . get ( x ) * ( store . get ( x ) + 1 ) ) / 2 ;
store . put ( n , ans ) ;
return ans ; } else {
int x = ( n + 1 ) / 2 , ans = 0 ;
for ( int i = 1 ; i < x ; i ++ ) { ans += store . get ( i ) * store . get ( n - i ) ; }
store . put ( n , ans ) ;
return ans ; } }
static void Wedderburn_Etherington ( int n ) {
store . put ( 0 , 0 ) ; store . put ( 1 , 1 ) ; store . put ( 2 , 1 ) ;
for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( Wedderburn ( i ) ) ; if ( i != n - 1 ) System . out . print ( " ▁ " ) ; } }
public static void main ( String [ ] args ) { int n = 10 ;
Wedderburn_Etherington ( n ) ; } }
static int Max_sum ( int a [ ] , int n ) {
int pos = 0 , neg = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] > 0 ) pos = 1 ;
else if ( a [ i ] < 0 ) neg = 1 ;
if ( ( pos == 1 ) && ( neg == 1 ) ) break ; }
int sum = 0 ; if ( ( pos == 1 ) && ( neg == 1 ) ) { for ( int i = 0 ; i < n ; i ++ ) sum += Math . abs ( a [ i ] ) ; } else if ( pos == 1 ) {
int mini = a [ 0 ] ; sum = a [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { mini = Math . min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; } else if ( neg == 1 ) {
for ( int i = 0 ; i < n ; i ++ ) a [ i ] = Math . abs ( a [ i ] ) ;
int mini = a [ 0 ] ; sum = a [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { mini = Math . min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; }
return sum ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 3 , 5 , - 2 , - 6 } ; int n = a . length ;
System . out . println ( Max_sum ( a , n ) ) ; } }
static void decimalToBinary ( int n ) {
if ( n == 0 ) { System . out . print ( "0" ) ; return ; }
decimalToBinary ( n / 2 ) ; System . out . print ( n % 2 ) ; }
public static void main ( String [ ] args ) { int n = 13 ; decimalToBinary ( n ) ; } }
static void MinimumValue ( int x , int y ) {
if ( x > y ) { int temp = x ; x = y ; y = temp ; }
int a = 1 ; int b = x - 1 ; int c = y - b ; System . out . print ( a + " ▁ " + b + " ▁ " + c ) ; }
public static void main ( String [ ] args ) { int x = 123 , y = 13 ;
MinimumValue ( x , y ) ; } }
static boolean canConvert ( int a , int b ) { while ( b > a ) {
if ( b % 10 == 1 ) { b /= 10 ; continue ; }
if ( b % 2 == 0 ) { b /= 2 ; continue ; }
return false ; }
if ( b == a ) return true ; return false ; }
public static void main ( String [ ] args ) { int A = 2 , B = 82 ; if ( canConvert ( A , B ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int count ( int N ) { int a = 0 ; a = ( N * ( N + 1 ) ) / 2 ; return a ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . print ( count ( n ) ) ; } }
static int numberOfDays ( int a , int b , int n ) { int Days = b * ( n + a ) / ( a + b ) ; return Days ; }
public static void main ( String [ ] args ) { int a = 10 , b = 20 , n = 5 ; System . out . println ( numberOfDays ( a , b , n ) ) ; } }
static int getAverage ( int x , int y ) {
int avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; return avg ; }
public static void main ( String [ ] args ) { int x = 10 , y = 9 ; System . out . print ( getAverage ( x , y ) ) ; } }
static int smallestIndex ( int [ ] a , int n ) {
int right1 = 0 , right0 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == 1 ) right1 = i ;
else right0 = i ; }
return Math . min ( right1 , right0 ) ; }
public static void main ( String [ ] args ) { int [ ] a = { 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int n = a . length ; System . out . println ( smallestIndex ( a , n ) ) ; } }
static int countSquares ( int r , int c , int m ) {
int squares = 0 ;
for ( int i = 1 ; i <= 8 ; i ++ ) { for ( int j = 1 ; j <= 8 ; j ++ ) {
if ( Math . max ( Math . abs ( i - r ) , Math . abs ( j - c ) ) <= m ) squares ++ ; } }
return squares ; }
public static void main ( String [ ] args ) { int r = 4 , c = 4 , m = 1 ; System . out . print ( countSquares ( r , c , m ) ) ; } }
static int countQuadruples ( int a [ ] , int n ) {
HashMap < Integer , Integer > mp = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( mp . containsKey ( a [ i ] ) ) { mp . put ( a [ i ] , mp . get ( a [ i ] ) + 1 ) ; } else { mp . put ( a [ i ] , 1 ) ; } int count = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { for ( int k = 0 ; k < n ; k ++ ) {
if ( j == k ) continue ;
mp . put ( a [ j ] , mp . get ( a [ j ] ) - 1 ) ; mp . put ( a [ k ] , mp . get ( a [ k ] ) - 1 ) ;
int first = a [ j ] - ( a [ k ] - a [ j ] ) ;
int fourth = ( a [ k ] * a [ k ] ) / a [ j ] ;
if ( ( a [ k ] * a [ k ] ) % a [ j ] == 0 ) {
if ( a [ j ] != a [ k ] ) { if ( mp . containsKey ( first ) && mp . containsKey ( fourth ) ) count += mp . get ( first ) * mp . get ( fourth ) ; }
else if ( mp . containsKey ( first ) && mp . containsKey ( fourth ) ) count += mp . get ( first ) * ( mp . get ( fourth ) - 1 ) ; }
if ( mp . containsKey ( a [ j ] ) ) { mp . put ( a [ j ] , mp . get ( a [ j ] ) + 1 ) ; } else { mp . put ( a [ j ] , 1 ) ; } if ( mp . containsKey ( a [ k ] ) ) { mp . put ( a [ k ] , mp . get ( a [ k ] ) + 1 ) ; } else { mp . put ( a [ k ] , 1 ) ; } } } return count ; }
public static void main ( String [ ] args ) { int a [ ] = { 2 , 6 , 4 , 9 , 2 } ; int n = a . length ; System . out . print ( countQuadruples ( a , n ) ) ; } }
static int countNumbers ( int L , int R , int K ) { if ( K == 9 ) { K = 0 ; }
int totalnumbers = R - L + 1 ;
int factor9 = totalnumbers / 9 ;
int rem = totalnumbers % 9 ;
int ans = factor9 ;
for ( int i = R ; i > R - rem ; i -- ) { int rem1 = i % 9 ; if ( rem1 == K ) { ans ++ ; } } return ans ; }
public static void main ( String [ ] args ) { int L = 10 ; int R = 22 ; int K = 3 ; System . out . println ( countNumbers ( L , R , K ) ) ; } }
static int EvenSum ( int [ ] A , int index , int value ) {
A [ index ] = A [ index ] + value ;
int sum = 0 ; for ( int i = 0 ; i < A . length ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; return sum ; }
static void BalanceArray ( int [ ] A , int [ ] [ ] Q ) {
int [ ] ANS = new int [ Q . length ] ; int i , sum ; for ( i = 0 ; i < Q . length ; i ++ ) { int index = Q [ i ] [ 0 ] ; int value = Q [ i ] [ 1 ] ;
sum = EvenSum ( A , index , value ) ;
ANS [ i ] = sum ; }
for ( i = 0 ; i < ANS . length ; i ++ ) System . out . print ( ANS [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int [ ] A = { 1 , 2 , 3 , 4 } ; int [ ] [ ] Q = { { 0 , 1 } , { 1 , - 3 } , { 0 , - 4 } , { 3 , 2 } } ; BalanceArray ( A , Q ) ; } }
static void BalanceArray ( int [ ] A , int [ ] [ ] Q ) { int [ ] ANS = new int [ A . length ] ; int i , sum = 0 ; for ( i = 0 ; i < A . length ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; for ( i = 0 ; i < Q . length ; i ++ ) { int index = Q [ i ] [ 0 ] ; int value = Q [ i ] [ 1 ] ;
if ( A [ index ] % 2 == 0 ) sum = sum - A [ index ] ; A [ index ] = A [ index ] + value ;
if ( A [ index ] % 2 == 0 ) sum = sum + A [ index ] ;
ANS [ i ] = sum ; }
for ( i = 0 ; i < ANS . length ; i ++ ) System . out . print ( ANS [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { int [ ] A = { 1 , 2 , 3 , 4 } ; int [ ] [ ] Q = { { 0 , 1 } , { 1 , - 3 } , { 0 , - 4 } , { 3 , 2 } } ; BalanceArray ( A , Q ) ; } }
static int Cycles ( int N ) { int fact = 1 , result = 0 ; result = N - 1 ;
int i = result ; while ( i > 0 ) { fact = fact * i ; i -- ; } return fact / 2 ; }
public static void main ( String [ ] args ) { int N = 5 ; int Number = Cycles ( N ) ; System . out . println ( " Hamiltonian ▁ cycles ▁ = ▁ " + Number ) ; } }
static boolean digitWell ( int n , int m , int k ) { int cnt = 0 ; while ( n > 0 ) { if ( n % 10 == m ) ++ cnt ; n /= 10 ; } return cnt == k ; }
static int findInt ( int n , int m , int k ) { int i = n + 1 ; while ( true ) { if ( digitWell ( i , m , k ) ) return i ; i ++ ; } }
public static void main ( String [ ] args ) { int n = 111 , m = 2 , k = 2 ; System . out . println ( findInt ( n , m , k ) ) ; } }
static int countOdd ( int [ ] arr , int n ) {
int odd = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] % 2 == 1 ) odd ++ ; } return odd ; }
static int countValidPairs ( int [ ] arr , int n ) { int odd = countOdd ( arr , n ) ; return ( odd * ( odd - 1 ) ) / 2 ; }
public static void main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . length ; System . out . println ( countValidPairs ( arr , n ) ) ; } }
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; else return gcd ( b , a % b ) ; }
static int lcmOfArray ( int arr [ ] , int n ) { if ( n < 1 ) return 0 ; int lcm = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) lcm = ( lcm * arr [ i ] ) / gcd ( lcm , arr [ i ] ) ;
return lcm ; }
static int minPerfectCube ( int arr [ ] , int n ) { int minPerfectCube ;
int lcm = lcmOfArray ( arr , n ) ; minPerfectCube = lcm ; int cnt = 0 ; while ( lcm > 1 && lcm % 2 == 0 ) { cnt ++ ; lcm /= 2 ; }
if ( cnt % 3 == 2 ) minPerfectCube *= 2 ; else if ( cnt % 3 == 1 ) minPerfectCube *= 4 ; int i = 3 ;
while ( lcm > 1 ) { cnt = 0 ; while ( lcm % i == 0 ) { cnt ++ ; lcm /= i ; } if ( cnt % 3 == 1 ) minPerfectCube *= i * i ; else if ( cnt % 3 == 2 ) minPerfectCube *= i ; i += 2 ; }
return minPerfectCube ; }
public static void main ( String args [ ] ) { int arr [ ] = { 10 , 125 , 14 , 42 , 100 } ; int n = arr . length ; System . out . println ( minPerfectCube ( arr , n ) ) ; } }
static boolean isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static boolean isStrongPrime ( int n ) {
if ( ! isPrime ( n ) n == 2 ) return false ;
int previous_prime = n - 1 ; int next_prime = n + 1 ;
while ( ! isPrime ( next_prime ) ) next_prime ++ ;
while ( ! isPrime ( previous_prime ) ) previous_prime -- ;
int mean = ( previous_prime + next_prime ) / 2 ;
if ( n > mean ) return true ; else return false ; }
public static void main ( String args [ ] ) { int n = 11 ; if ( isStrongPrime ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int countDigitsToBeRemoved ( int N , int K ) {
String s = Integer . toString ( N ) ;
int res = 0 ;
int f_zero = 0 ; for ( int i = s . length ( ) - 1 ; i >= 0 ; i -- ) { if ( K == 0 ) return res ; if ( s . charAt ( i ) == '0' ) {
f_zero = 1 ; K -- ; } else res ++ ; }
if ( K == 0 ) return res ; else if ( f_zero == 1 ) return s . length ( ) - 1 ; return - 1 ; }
public static void main ( String [ ] args ) { int N = 10904025 ; int K = 2 ; System . out . println ( countDigitsToBeRemoved ( N , K ) ) ; N = 1000 ; K = 5 ; System . out . println ( countDigitsToBeRemoved ( N , K ) ) ; N = 23985 ; K = 2 ; System . out . println ( countDigitsToBeRemoved ( N , K ) ) ; } }
public static float getSum ( int a , int n ) {
float sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) {
sum += ( i / Math . pow ( a , i ) ) ; } return sum ; }
public static void main ( String [ ] args ) { int a = 3 , n = 3 ;
System . out . println ( getSum ( a , n ) ) ; } }
static int largestPrimeFactor ( int n ) {
int max = - 1 ;
while ( n % 2 == 0 ) { max = 2 ;
}
for ( int i = 3 ; i <= Math . sqrt ( n ) ; i += 2 ) { while ( n % i == 0 ) { max = i ; n = n / i ; } }
if ( n > 2 ) max = n ; return max ; }
static boolean checkUnusual ( int n ) {
int factor = largestPrimeFactor ( n ) ;
if ( factor > Math . sqrt ( n ) ) { return true ; } else { return false ; } }
public static void main ( String [ ] args ) { int n = 14 ; if ( checkUnusual ( n ) ) { System . out . println ( " YES " ) ; } else { System . out . println ( " NO " ) ; } } }
static void isHalfReducible ( int arr [ ] , int n , int m ) { int frequencyHash [ ] = new int [ m + 1 ] ; int i ; for ( i = 0 ; i < frequencyHash . length ; i ++ ) frequencyHash [ i ] = 0 ; for ( i = 0 ; i < n ; i ++ ) { frequencyHash [ arr [ i ] % ( m + 1 ) ] ++ ; } for ( i = 0 ; i <= m ; i ++ ) { if ( frequencyHash [ i ] >= n / 2 ) break ; } if ( i <= m ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 8 , 16 , 32 , 3 , 12 } ; int n = arr . length ; int m = 7 ; isHalfReducible ( arr , n , m ) ; } }
import java . util . * ; class GFG { static Vector < Integer > arr = new Vector < Integer > ( ) ;
static void generateDivisors ( int n ) {
for ( int i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) arr . add ( i ) ;
{ arr . add ( i ) ; arr . add ( n / i ) ; } } } }
static double harmonicMean ( int n ) { generateDivisors ( n ) ;
double sum = 0.0 ; int len = arr . size ( ) ;
for ( int i = 0 ; i < len ; i ++ ) sum = sum + n / arr . get ( i ) ; sum = sum / n ;
return arr . size ( ) / sum ; }
static boolean isOreNumber ( int n ) {
double mean = harmonicMean ( n ) ;
if ( mean - Math . floor ( mean ) == 0 ) return true ; else return false ; }
public static void main ( String [ ] args ) { int n = 28 ; if ( isOreNumber ( n ) ) System . out . println ( " YES " ) ; else System . out . println ( " NO " ) ; } }
import java . util . * ; class GFG { static int MAX = 10000 ; static HashSet < Integer > s = new HashSet < Integer > ( ) ;
static void SieveOfEratosthenes ( ) {
boolean [ ] prime = new boolean [ MAX ] ; Arrays . fill ( prime , true ) ; prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
int product = 1 ; for ( int p = 2 ; p < MAX ; p ++ ) { if ( prime [ p ] ) {
product = product * p ;
s . add ( product + 1 ) ; } } }
static boolean isEuclid ( int n ) {
if ( s . contains ( n ) ) return true ; else return false ; }
SieveOfEratosthenes ( ) ;
int n = 31 ;
if ( isEuclid ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ;
n = 42 ;
if ( isEuclid ( n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) { return false ; } } return true ; }
static boolean isPowerOfTwo ( int n ) { return n != 0 && ( ( n & ( n - 1 ) ) == 0 ) ; }
public static void main ( String [ ] args ) { int n = 43 ;
if ( isPrime ( n ) && ( isPowerOfTwo ( n * 3 - 1 ) ) ) { System . out . println ( " YES " ) ; } else { System . out . println ( " NO " ) ; } } }
static float area ( float a ) {
if ( a < 0 ) return - 1 ;
float area = ( float ) Math . pow ( ( a * Math . sqrt ( 3 ) ) / ( Math . sqrt ( 2 ) ) , 2 ) ; return area ; }
public static void main ( String [ ] args ) { float a = 5 ; System . out . println ( area ( a ) ) ; } }
static int nthTerm ( int n ) { return 3 * ( int ) Math . pow ( n , 2 ) - 4 * n + 2 ; }
public static void main ( String args [ ] ) { int N = 4 ; System . out . println ( nthTerm ( N ) ) ; } }
static int calculateSum ( int n ) { return n * ( n + 1 ) / 2 + ( int ) Math . pow ( ( n * ( n + 1 ) / 2 ) , 2 ) ; }
int n = 3 ;
System . out . println ( " Sum ▁ = ▁ " + calculateSum ( n ) ) ; } }
static boolean arePermutations ( int a [ ] , int b [ ] , int n , int m ) { int sum1 = 0 , sum2 = 0 , mul1 = 1 , mul2 = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { sum1 += a [ i ] ; mul1 *= a [ i ] ; }
for ( int i = 0 ; i < m ; i ++ ) { sum2 += b [ i ] ; mul2 *= b [ i ] ; }
return ( ( sum1 == sum2 ) && ( mul1 == mul2 ) ) ; }
public static void main ( String [ ] args ) { int a [ ] = { 1 , 3 , 2 } ; int b [ ] = { 3 , 1 , 2 } ; int n = a . length ; int m = b . length ; if ( arePermutations ( a , b , n , m ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int Race ( int B , int C ) { int result = 0 ;
result = ( ( C * 100 ) / B ) ; return 100 - result ; }
public static void main ( String [ ] args ) { int B = 10 ; int C = 28 ;
B = 100 - B ; C = 100 - C ; System . out . println ( Race ( B , C ) + " ▁ meters " ) ; } }
static float Time ( float arr [ ] , int n , float Emptypipe ) { float fill = 0 ; for ( int i = 0 ; i < n ; i ++ ) fill += 1 / arr [ i ] ; fill = fill - ( 1 / ( float ) Emptypipe ) ; return 1 / fill ; }
public static void main ( String [ ] args ) { float arr [ ] = { 12 , 14 } ; float Emptypipe = 30 ; int n = arr . length ; System . out . println ( ( int ) ( Time ( arr , n , Emptypipe ) ) + " ▁ Hours " ) ; } }
static int check ( int n ) { int sum = 0 ;
while ( n != 0 ) { sum += n % 10 ; n = n / 10 ; }
if ( sum % 7 == 0 ) return 1 ; else return 0 ; }
int n = 25 ; String s = ( check ( n ) == 1 ) ? " YES " : " NO " ; System . out . println ( s ) ; } }
import java . io . * ; import java . util . * ; class GFG {
static boolean isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static int SumOfPrimeDivisors ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { if ( n % i == 0 ) { if ( isPrime ( i ) ) sum += i ; } } return sum ; }
public static void main ( String args [ ] ) { int n = 60 ; System . out . print ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " + SumOfPrimeDivisors ( n ) + "NEW_LINE"); } }
static int Sum ( int N ) { int SumOfPrimeDivisors [ ] = new int [ N + 1 ] ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( SumOfPrimeDivisors [ i ] == 0 ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
public static void main ( String args [ ] ) { int N = 60 ; System . out . print ( " Sum ▁ of ▁ prime ▁ " + " divisors ▁ of ▁ 60 ▁ is ▁ " + Sum ( N ) + "NEW_LINE"); } }
static String find_Square_369 ( String num ) { char a , b , c , d ;
if ( num . charAt ( 0 ) == '3' ) { a = '1' ; b = '0' ; c = '8' ; d = '9' ; }
else if ( num . charAt ( 0 ) == '6' ) { a = '4' ; b = '3' ; c = '5' ; d = '6' ; }
else { a = '9' ; b = '8' ; c = '0' ; d = '1' ; }
String result = " " ;
int size = num . length ( ) ;
for ( int i = 1 ; i < size ; i ++ ) result += a ;
result += b ;
for ( int i = 1 ; i < size ; i ++ ) result += c ;
result += d ;
return result ; }
public static void main ( String [ ] args ) { String num_3 , num_6 , num_9 ; num_3 = "3333" ; num_6 = "6666" ; num_9 = "9999" ; String result = " " ;
result = find_Square_369 ( num_3 ) ; System . out . println ( " Square ▁ of ▁ " + num_3 + " ▁ is ▁ : ▁ " + result ) ;
result = find_Square_369 ( num_6 ) ; System . out . println ( " Square ▁ of ▁ " + num_9 + " ▁ is ▁ : ▁ " + result ) ;
result = find_Square_369 ( num_9 ) ; System . out . println ( " Square ▁ of ▁ " + num_9 + " ▁ is ▁ : ▁ " + result ) ; } }
class GFG { public static void main ( String [ ] args ) { long ans = 1 ; long mod = ( long ) 1000000007 * 120 ; for ( int i = 0 ; i < 5 ; i ++ ) ans = ( ans * ( 55555 - i ) ) % mod ; ans = ans / 120 ; System . out . println ( " Answer ▁ using " + " ▁ shortcut : ▁ " + ans ) ; } }
static int fact ( int n ) { if ( n == 0 n == 1 ) return 1 ; int ans = 1 ; for ( int i = 1 ; i <= n ; i ++ ) ans = ans * i ; return ans ; }
static int nCr ( int n , int r ) { int Nr = n , Dr = 1 , ans = 1 ; for ( int i = 1 ; i <= r ; i ++ ) { ans = ( ans * Nr ) / ( Dr ) ; Nr -- ; Dr ++ ; } return ans ; }
static int solve ( int n ) { int N = 2 * n - 2 ; int R = n - 1 ; return nCr ( N , R ) * fact ( n - 1 ) ; }
public static void main ( String [ ] args ) { int n = 6 ; System . out . println ( solve ( n ) ) ; } }
class GFG { static void pythagoreanTriplet ( int n ) {
for ( int i = 1 ; i <= n / 3 ; i ++ ) {
for ( int j = i + 1 ; j <= n / 2 ; j ++ ) { int k = n - i - j ; if ( i * i + j * j == k * k ) { System . out . print ( i + " , ▁ " + j + " , ▁ " + k ) ; return ; } } } System . out . print ( " No ▁ Triplet " ) ; }
public static void main ( String arg [ ] ) { int n = 12 ; pythagoreanTriplet ( n ) ; } }
static int factorial ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
static void series ( int A , int X , int n ) {
int nFact = factorial ( n ) ;
for ( int i = 0 ; i < n + 1 ; i ++ ) {
int niFact = factorial ( n - i ) ; int iFact = factorial ( i ) ;
int aPow = ( int ) Math . pow ( A , n - i ) ; int xPow = ( int ) Math . pow ( X , i ) ;
System . out . print ( ( nFact * aPow * xPow ) / ( niFact * iFact ) + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; } }
static int seiresSum ( int n , int [ ] a ) { int res = 0 , i ; for ( i = 0 ; i < 2 * n ; i ++ ) { if ( i % 2 == 0 ) res += a [ i ] * a [ i ] ; else res -= a [ i ] * a [ i ] ; } return res ; }
public static void main ( String args [ ] ) { int n = 2 ; int a [ ] = { 1 , 2 , 3 , 4 } ; System . out . println ( seiresSum ( n , a ) ) ; } }
static int power ( int n , int r ) {
int count = 0 ; for ( int i = r ; ( n / i ) >= 1 ; i = i * r ) count += n / i ; return count ; }
public static void main ( String [ ] args ) { int n = 6 , r = 3 ; System . out . print ( power ( n , r ) ) ; } }
static int avg_of_odd_num ( int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += ( 2 * i + 1 ) ;
return sum / n ; }
public static void main ( String [ ] args ) { int n = 20 ; avg_of_odd_num ( n ) ; System . out . println ( avg_of_odd_num ( n ) ) ; } }
static int avg_of_odd_num ( int n ) { return n ; }
public static void main ( String [ ] args ) { int n = 8 ; System . out . println ( avg_of_odd_num ( n ) ) ; } }
static void fib ( int f [ ] , int N ) {
f [ 1 ] = 1 ; f [ 2 ] = 1 ; for ( int i = 3 ; i <= N ; i ++ )
f [ i ] = f [ i - 1 ] + f [ i - 2 ] ; } static void fiboTriangle ( int n ) {
int N = n * ( n + 1 ) / 2 ; int f [ ] = new int [ N + 1 ] ; fib ( f , N ) ;
int fiboNum = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) System . out . print ( f [ fiboNum ++ ] + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String args [ ] ) { int n = 5 ; fiboTriangle ( n ) ; } }
static int averageOdd ( int n ) { if ( n % 2 == 0 ) { System . out . println ( " Invalid ▁ Input " ) ; return - 1 ; } int sum = 0 , count = 0 ; while ( n >= 1 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
public static void main ( String args [ ] ) { int n = 15 ; System . out . println ( averageOdd ( n ) ) ; } }
static int averageOdd ( int n ) { if ( n % 2 == 0 ) { System . out . println ( " Invalid ▁ Input " ) ; return - 1 ; } return ( n + 1 ) / 2 ; }
public static void main ( String args [ ] ) { int n = 15 ; System . out . println ( averageOdd ( n ) ) ; } }
class GFG { static class Rational { int nume , deno ; public Rational ( int nume , int deno ) { this . nume = nume ; this . deno = deno ; } } ;
static int lcm ( int a , int b ) { return ( a * b ) / ( __gcd ( a , b ) ) ; }
static Rational maxRational ( Rational first , Rational sec ) {
int k = lcm ( first . deno , sec . deno ) ;
int nume1 = first . nume ; int nume2 = sec . nume ; nume1 *= k / ( first . deno ) ; nume2 *= k / ( sec . deno ) ; return ( nume2 < nume1 ) ? first : sec ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
public static void main ( String [ ] args ) { Rational first = new Rational ( 3 , 2 ) ; Rational sec = new Rational ( 3 , 4 ) ; Rational res = maxRational ( first , sec ) ; System . out . print ( res . nume + " / " + res . deno ) ; } }
public static int TrinomialValue ( int n , int k ) {
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ; }
public static void printTrinomial ( int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = - i ; j <= 0 ; j ++ ) System . out . print ( TrinomialValue ( i , j ) + " ▁ " ) ;
for ( int j = 1 ; j <= i ; j ++ ) System . out . print ( TrinomialValue ( i , j ) + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String argc [ ] ) { int n = 4 ; printTrinomial ( n ) ; } }
import java . util . * ; import java . lang . * ; public class GfG { private static final int MAX = 10 ;
public static int TrinomialValue ( int dp [ ] [ ] , int n , int k ) {
if ( k < 0 ) k = - k ;
if ( dp [ n ] [ k ] != 0 ) return dp [ n ] [ k ] ;
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return ( dp [ n ] [ k ] = TrinomialValue ( dp , n - 1 , k - 1 ) + TrinomialValue ( dp , n - 1 , k ) + TrinomialValue ( dp , n - 1 , k + 1 ) ) ; }
public static void printTrinomial ( int n ) { int [ ] [ ] dp = new int [ MAX ] [ MAX ] ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = - i ; j <= 0 ; j ++ ) System . out . print ( TrinomialValue ( dp , i , j ) + " ▁ " ) ;
for ( int j = 1 ; j <= i ; j ++ ) System . out . print ( TrinomialValue ( dp , i , j ) + " ▁ " ) ; System . out . println ( ) ; } }
public static void main ( String argc [ ] ) { int n = 4 ; printTrinomial ( n ) ; } }
static int sumOfLargePrimeFactor ( int n ) {
int prime [ ] = new int [ n + 1 ] , sum = 0 ; Arrays . fill ( prime , 0 ) ; int max = n / 2 ; for ( int p = 2 ; p <= max ; p ++ ) {
if ( prime [ p ] == 0 ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = p ; } }
for ( int p = 2 ; p <= n ; p ++ ) {
if ( prime [ p ] != 0 ) sum += prime [ p ] ;
else sum += p ; }
return sum ; }
public static void main ( String args [ ] ) { int n = 12 ; System . out . println ( " Sum ▁ = ▁ " + sumOfLargePrimeFactor ( n ) ) ; } }
static int calculate_sum ( int a , int N ) {
int m = N / a ;
int sum = m * ( m + 1 ) / 2 ;
int ans = a * sum ; return ans ; }
public static void main ( String [ ] args ) { int a = 7 , N = 49 ; System . out . println ( " Sum ▁ of ▁ multiples ▁ of ▁ " + a + " ▁ up ▁ to ▁ " + N + " ▁ = ▁ " + calculate_sum ( a , N ) ) ; } }
static long ispowerof2 ( long num ) { if ( ( num & ( num - 1 ) ) == 0 ) return 1 ; return 0 ; }
public static void main ( String [ ] args ) { long num = 549755813888L ; System . out . println ( ispowerof2 ( num ) ) ; } }
static int counDivisors ( int X ) {
int count = 0 ;
for ( int i = 1 ; i <= X ; ++ i ) { if ( X % i == 0 ) { count ++ ; } }
return count ; }
static int countDivisorsMult ( int arr [ ] , int n ) {
int mul = 1 ; for ( int i = 0 ; i < n ; ++ i ) mul *= arr [ i ] ;
return counDivisors ( mul ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 6 } ; int n = arr . length ; System . out . println ( countDivisorsMult ( arr , n ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static void SieveOfEratosthenes ( int largest , ArrayList < Integer > prime ) {
boolean [ ] isPrime = new boolean [ largest + 1 ] ; Arrays . fill ( isPrime , true ) ; for ( int p = 2 ; p * p <= largest ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( int i = p * 2 ; i <= largest ; i += p ) isPrime [ i ] = false ; } }
for ( int p = 2 ; p <= largest ; p ++ ) if ( isPrime [ p ] ) prime . add ( p ) ; }
static long countDivisorsMult ( int [ ] arr , int n ) {
int largest = 0 ; for ( int a : arr ) { largest = Math . max ( largest , a ) ; } ArrayList < Integer > prime = new ArrayList < Integer > ( ) ; SieveOfEratosthenes ( largest , prime ) ;
Map < Integer , Integer > mp = new HashMap < > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < prime . size ( ) ; j ++ ) { while ( arr [ i ] > 1 && arr [ i ] % prime . get ( j ) == 0 ) { arr [ i ] /= prime . get ( j ) ; if ( mp . containsKey ( prime . get ( j ) ) ) { mp . put ( prime . get ( j ) , mp . get ( prime . get ( j ) ) + 1 ) ; } else { mp . put ( prime . get ( j ) , 1 ) ; } } } if ( arr [ i ] != 1 ) { if ( mp . containsKey ( arr [ i ] ) ) { mp . put ( arr [ i ] , mp . get ( arr [ i ] ) + 1 ) ; } else { mp . put ( arr [ i ] , 1 ) ; } } }
long res = 1 ; for ( int it : mp . keySet ( ) ) res *= ( mp . get ( it ) + 1L ) ; return res ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 6 } ; int n = arr . length ; System . out . println ( countDivisorsMult ( arr , n ) ) ; } }
static void findPrimeNos ( int L , int R , Map < Integer , Integer > M , int K ) {
for ( int i = L ; i <= R ; i ++ ) { if ( M . get ( i ) != null ) M . put ( i , M . get ( i ) + 1 ) ; else M . put ( i , 1 ) ; }
if ( M . get ( 1 ) != null ) { M . remove ( 1 ) ; }
for ( int i = 2 ; i <= Math . sqrt ( R ) ; i ++ ) { int multiple = 2 ; while ( ( i * multiple ) <= R ) {
if ( M . get ( i * multiple ) != null ) {
M . remove ( i * multiple ) ; }
multiple ++ ; } }
for ( Map . Entry < Integer , Integer > entry : M . entrySet ( ) ) {
if ( M . get ( entry . getKey ( ) + K ) != null ) { System . out . print ( " ( " + entry . getKey ( ) + " , ▁ " + ( entry . getKey ( ) + K ) + " ) ▁ " ) ; } } }
static void getPrimePairs ( int L , int R , int K ) { Map < Integer , Integer > M = new HashMap < Integer , Integer > ( ) ;
findPrimeNos ( L , R , M , K ) ; }
int L = 1 , R = 19 ;
int K = 6 ;
getPrimePairs ( L , R , K ) ; } }
static int enneacontahexagonNum ( int n ) { return ( 94 * n * n - 92 * n ) / 2 ; }
public static void main ( String [ ] args ) { int n = 3 ; System . out . print ( enneacontahexagonNum ( n ) ) ; } }
static void find_composite_nos ( int n ) { System . out . println ( 9 * n + " ▁ " + 8 * n ) ; }
public static void main ( String [ ] args ) { int n = 4 ; find_composite_nos ( n ) ; } }
static int freqPairs ( int arr [ ] , int n ) {
int max = Arrays . stream ( arr ) . max ( ) . getAsInt ( ) ;
int freq [ ] = new int [ max + 1 ] ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; }
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 2 * arr [ i ] ; j <= max ; j += arr [ i ] ) {
if ( freq [ j ] >= 1 ) { count += freq [ j ] ; } }
if ( freq [ arr [ i ] ] > 1 ) { count += freq [ arr [ i ] ] - 1 ; freq [ arr [ i ] ] -- ; } } return count ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 2 , 4 , 2 , 6 } ; int n = arr . length ; System . out . println ( freqPairs ( arr , n ) ) ; } }
static double Nth_Term ( int n ) { return ( 2 * Math . pow ( n , 3 ) - 3 * Math . pow ( n , 2 ) + n + 6 ) / 6 ; }
static public void main ( String args [ ] ) { int N = 8 ; System . out . println ( Nth_Term ( N ) ) ; } }
static int printNthElement ( int n ) {
int arr [ ] = new int [ n + 1 ] ; arr [ 1 ] = 3 ; arr [ 2 ] = 5 ; for ( int i = 3 ; i <= n ; i ++ ) {
if ( i % 2 != 0 ) arr [ i ] = arr [ i / 2 ] * 10 + 3 ; else arr [ i ] = arr [ ( i / 2 ) - 1 ] * 10 + 5 ; } return arr [ n ] ; }
public static void main ( String [ ] args ) { int n = 6 ; System . out . println ( printNthElement ( n ) ) ; } }
return ( N * ( ( N / 2 ) + ( ( N % 2 ) * 2 ) + N ) ) ; } }
class GFG { public static void main ( String [ ] args ) {
Nth a = new Nth ( ) ;
System . out . println ( " Nth ▁ term ▁ for ▁ N ▁ = ▁ " + N + " ▁ : ▁ " + a . nthTerm ( N ) ) ; } }
static void series ( int A , int X , int n ) {
int term = ( int ) Math . pow ( A , n ) ; System . out . print ( term + " ▁ " ) ;
for ( int i = 1 ; i <= n ; i ++ ) {
term = term * X * ( n - i + 1 ) / ( i * A ) ; System . out . print ( term + " ▁ " ) ; } }
public static void main ( String [ ] args ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; } }
static boolean Div_by_8 ( int n ) { return ( ( ( n >> 3 ) << 3 ) == n ) ; }
public static void main ( String [ ] args ) { int n = 16 ; if ( Div_by_8 ( n ) ) System . out . println ( " YES " ) ; else System . out . println ( " NO " ) ; } }
static int averageEven ( int n ) { if ( n % 2 != 0 ) { System . out . println ( " Invalid ▁ Input " ) ; return - 1 ; } int sum = 0 , count = 0 ; while ( n >= 2 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
public static void main ( String args [ ] ) { int n = 16 ; System . out . println ( averageEven ( n ) ) ; } }
static int averageEven ( int n ) { if ( n % 2 != 0 ) { System . out . println ( " Invalid ▁ Input " ) ; return - 1 ; } return ( n + 2 ) / 2 ; }
public static void main ( String args [ ] ) { int n = 16 ; System . out . println ( averageEven ( n ) ) ; } }
static int gcd ( int a , int b ) {
if ( a == 0 b == 0 ) return 0 ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
static int cpFact ( int x , int y ) { while ( gcd ( x , y ) != 1 ) { x = x / gcd ( x , y ) ; } return x ; }
public static void main ( String [ ] args ) { int x = 15 ; int y = 3 ; System . out . println ( cpFact ( x , y ) ) ; x = 14 ; y = 28 ; System . out . println ( cpFact ( x , y ) ) ; x = 7 ; y = 3 ; System . out . println ( cpFact ( x , y ) ) ; } }
public static int counLastDigitK ( int low , int high , int k ) { int count = 0 ; for ( int i = low ; i <= high ; i ++ ) if ( i % 10 == k ) count ++ ; return count ; }
public static void main ( String args [ ] ) { int low = 3 , high = 35 , k = 3 ; System . out . println ( counLastDigitK ( low , high , k ) ) ; } }
import java . util . * ; class GFG { public static void printTaxicab2 ( int N ) {
int i = 1 , count = 0 ; while ( count < N ) { int int_count = 0 ;
for ( int j = 1 ; j <= Math . pow ( i , 1.0 / 3 ) ; j ++ ) for ( int k = j + 1 ; k <= Math . pow ( i , 1.0 / 3 ) ; k ++ ) if ( j * j * j + k * k * k == i ) int_count ++ ;
if ( int_count == 2 ) { count ++ ; System . out . println ( count + " ▁ " + i ) ; } i ++ ; } }
public static void main ( String [ ] args ) { int N = 5 ; printTaxicab2 ( N ) ; } }
import java . io . * ; class Composite { static boolean isComposite ( int n ) {
if ( n <= 1 ) System . out . println ( " False " ) ; if ( n <= 3 ) System . out . println ( " False " ) ;
if ( n % 2 == 0 n % 3 == 0 ) return true ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return true ; return false ; }
public static void main ( String args [ ] ) { System . out . println ( isComposite ( 11 ) ? " true " : " false " ) ; System . out . println ( isComposite ( 15 ) ? " true " : " false " ) ; } }
static boolean isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
static int findPrime ( int n ) { int num = n + 1 ;
while ( num > 0 ) {
if ( isPrime ( num ) ) return num ;
num = num + 1 ; } return 0 ; }
static int minNumber ( int arr [ ] , int n ) { int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( isPrime ( sum ) ) return 0 ;
int num = findPrime ( sum ) ;
return num - sum ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 4 , 6 , 8 , 12 } ; int n = arr . length ; System . out . println ( minNumber ( arr , n ) ) ; } }
static int fact ( int n ) { if ( n == 0 ) return 1 ; return n * fact ( n - 1 ) ; }
static int div ( int x ) { int ans = 0 ; for ( int i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
static int sumFactDiv ( int n ) { return div ( fact ( n ) ) ; }
public static void main ( String args [ ] ) { int n = 4 ; System . out . println ( sumFactDiv ( n ) ) ; } }
static ArrayList < Integer > allPrimes = new ArrayList < Integer > ( ) ;
static void sieve ( int n ) {
boolean [ ] prime = new boolean [ n + 1 ] ;
for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == false ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = true ; } }
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] == false ) allPrimes . add ( p ) ; }
static int factorialDivisors ( int n ) {
int result = 1 ;
for ( int i = 0 ; i < allPrimes . size ( ) ; i ++ ) {
int p = allPrimes . get ( i ) ;
int exp = 0 ; while ( p <= n ) { exp = exp + ( n / p ) ; p = p * allPrimes . get ( i ) ; }
result = result * ( ( int ) Math . pow ( allPrimes . get ( i ) , exp + 1 ) - 1 ) / ( allPrimes . get ( i ) - 1 ) ; }
return result ; }
public static void main ( String [ ] args ) { System . out . println ( factorialDivisors ( 4 ) ) ; } }
static boolean checkPandigital ( int b , String n ) {
if ( n . length ( ) < b ) return false ; boolean hash [ ] = new boolean [ b ] ; Arrays . fill ( hash , false ) ;
for ( int i = 0 ; i < n . length ( ) ; i ++ ) {
if ( n . charAt ( i ) >= '0' && n . charAt ( i ) <= '9' ) hash [ n . charAt ( i ) - '0' ] = true ;
else if ( n . charAt ( i ) - ' A ' <= b - 11 ) hash [ n . charAt ( i ) - ' A ' + 10 ] = true ; }
for ( int i = 0 ; i < b ; i ++ ) if ( hash [ i ] == false ) return false ; return true ; }
public static void main ( String [ ] args ) { int b = 13 ; String n = "1298450376ABC " ; if ( checkPandigital ( b , n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int convert ( int m , int n ) { if ( m == n ) return 0 ;
if ( m > n ) return m - n ;
if ( m <= 0 && n > 0 ) return - 1 ;
if ( n % 2 == 1 )
return 1 + convert ( m , n + 1 ) ;
else
return 1 + convert ( m , n / 2 ) ; }
public static void main ( String [ ] args ) { int m = 3 , n = 11 ; System . out . println ( " Minimum ▁ number ▁ of ▁ " + " operations ▁ : ▁ " + convert ( m , n ) ) ; } }
import java . util . * ; class GFg { static int MAX = 10000 ; static int [ ] prodDig = new int [ MAX ] ;
static int getDigitProduct ( int x ) {
if ( x < 10 ) return x ;
if ( prodDig [ x ] != 0 ) return prodDig [ x ] ;
int prod = ( x % 10 ) * getDigitProduct ( x / 10 ) ; return ( prodDig [ x ] = prod ) ; }
static void findSeed ( int n ) {
List < Integer > res = new ArrayList < Integer > ( ) ; for ( int i = 1 ; i <= n / 2 ; i ++ ) if ( i * getDigitProduct ( i ) == n ) res . add ( i ) ;
if ( res . size ( ) == 0 ) { System . out . println ( " NO ▁ seed ▁ exists " ) ; return ; }
for ( int i = 0 ; i < res . size ( ) ; i ++ ) System . out . print ( res . get ( i ) + " ▁ " ) ; }
public static void main ( String [ ] args ) { int n = 138 ; findSeed ( n ) ; } }
static int maxPrimefactorNum ( int N ) { int arr [ ] = new int [ N + 5 ] ; Arrays . fill ( arr , 0 ) ;
for ( int i = 2 ; i * i <= N ; i ++ ) { if ( arr [ i ] == 0 ) { for ( int j = 2 * i ; j <= N ; j += i ) { arr [ j ] ++ ; } } arr [ i ] = 1 ; } int maxval = 0 , maxint = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( arr [ i ] > maxval ) { maxval = arr [ i ] ; maxint = i ; } } return maxint ; }
public static void main ( String [ ] args ) { int N = 40 ; System . out . println ( maxPrimefactorNum ( N ) ) ; } }
public static long SubArraySum ( int arr [ ] , int n ) { long result = 0 ;
for ( int i = 0 ; i < n ; i ++ ) result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) ;
return result ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 } ; int n = arr . length ; System . out . println ( " Sum ▁ of ▁ SubArray ▁ " + SubArraySum ( arr , n ) ) ; } }
class GFG { static int highestPowerof2 ( int n ) { int res = 0 ; for ( int i = n ; i >= 1 ; i -- ) {
if ( ( i & ( i - 1 ) ) == 0 ) { res = i ; break ; } } return res ; }
public static void main ( String [ ] args ) { int n = 10 ; System . out . print ( highestPowerof2 ( n ) ) ; } }
static void findPairs ( int n ) {
int cubeRoot = ( int ) Math . pow ( n , 1.0 / 3.0 ) ;
int cube [ ] = new int [ cubeRoot + 1 ] ;
for ( int i = 1 ; i <= cubeRoot ; i ++ ) cube [ i ] = i * i * i ;
int l = 1 ; int r = cubeRoot ; while ( l < r ) { if ( cube [ l ] + cube [ r ] < n ) l ++ ; else if ( cube [ l ] + cube [ r ] > n ) r -- ; else { System . out . println ( " ( " + l + " , ▁ " + r + " ) " ) ; l ++ ; r -- ; } } }
public static void main ( String [ ] args ) { int n = 20683 ; findPairs ( n ) ; } }
static int gcd ( int a , int b ) { while ( b != 0 ) { int t = b ; b = a % b ; a = t ; } return a ; }
static int findMinDiff ( int a , int b , int x , int y ) {
int g = gcd ( a , b ) ;
int diff = Math . abs ( x - y ) % g ; return Math . min ( diff , g - diff ) ; }
public static void main ( String [ ] args ) { int a = 20 , b = 52 , x = 5 , y = 7 ; System . out . println ( findMinDiff ( a , b , x , y ) ) ; } }
static void printDivisors ( int n ) {
Vector < Integer > v = new Vector < > ( ) ; for ( int i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) System . out . printf ( " % d ▁ " , i ) ; else { System . out . printf ( " % d ▁ " , i ) ;
v . add ( n / i ) ; } } }
for ( int i = v . size ( ) - 1 ; i >= 0 ; i -- ) System . out . printf ( " % d ▁ " , v . get ( i ) ) ; }
public static void main ( String args [ ] ) { System . out . println ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " ) ; printDivisors ( 100 ) ; } }
public static void printDivisors ( int n ) { int i ; for ( i = 1 ; i * i < n ; i ++ ) { if ( n % i == 0 ) System . out . print ( i + " ▁ " ) ; } if ( i - ( n / i ) == 1 ) { i -- ; } for ( ; i >= 1 ; i -- ) { if ( n % i == 0 ) System . out . print ( n / i + " ▁ " ) ; } }
public static void main ( String [ ] args ) { System . out . println ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " ) ; printDivisors ( 100 ) ; } }
static void printDivisors ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) System . out . print ( i + " ▁ " ) ; }
public static void main ( String args [ ] ) { System . out . println ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " ) ; printDivisors ( 100 ) ; ; } }
static void printDivisors ( int n ) {
for ( int i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) System . out . print ( " ▁ " + i ) ;
System . out . print ( i + " ▁ " + n / i + " ▁ " ) ; } } }
public static void main ( String args [ ] ) { System . out . println ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ " ) ; printDivisors ( 100 ) ; ; } }
class GFG { static int SieveOfAtkin ( int limit ) {
if ( limit > 2 ) System . out . print ( 2 + " ▁ " ) ; if ( limit > 3 ) System . out . print ( 3 + " ▁ " ) ;
boolean sieve [ ] = new boolean [ limit ] ; for ( int i = 0 ; i < limit ; i ++ ) sieve [ i ] = false ;
for ( int x = 1 ; x * x < limit ; x ++ ) { for ( int y = 1 ; y * y < limit ; y ++ ) {
int n = ( 4 * x * x ) + ( y * y ) ; if ( n <= limit && ( n % 12 == 1 n % 12 == 5 ) ) sieve [ n ] ^= true ; n = ( 3 * x * x ) + ( y * y ) ; if ( n <= limit && n % 12 == 7 ) sieve [ n ] ^= true ; n = ( 3 * x * x ) - ( y * y ) ; if ( x > y && n <= limit && n % 12 == 11 ) sieve [ n ] ^= true ; } }
for ( int r = 5 ; r * r < limit ; r ++ ) { if ( sieve [ r ] ) { for ( int i = r * r ; i < limit ; i += r * r ) sieve [ i ] = false ; } }
for ( int a = 5 ; a < limit ; a ++ ) if ( sieve [ a ] ) System . out . print ( a + " ▁ " ) ; return 0 ; }
public static void main ( String [ ] args ) { int limit = 20 ; SieveOfAtkin ( limit ) ; } }
class GFG { static boolean isInside ( int circle_x , int circle_y , int rad , int x , int y ) {
if ( ( x - circle_x ) * ( x - circle_x ) + ( y - circle_y ) * ( y - circle_y ) <= rad * rad ) return true ; else return false ; }
public static void main ( String arg [ ] ) { int x = 1 , y = 1 ; int circle_x = 0 , circle_y = 1 , rad = 2 ; if ( isInside ( circle_x , circle_y , rad , x , y ) ) System . out . print ( " Inside " ) ; else System . out . print ( " Outside " ) ; } }
static int eval ( int a , char op , int b ) { if ( op == ' + ' ) { return a + b ; } if ( op == ' - ' ) { return a - b ; } if ( op == ' * ' ) { return a * b ; } return Integer . MAX_VALUE ; }
static Vector < Integer > evaluateAll ( String expr , int low , int high ) {
Vector < Integer > res = new Vector < Integer > ( ) ;
if ( low == high ) { res . add ( expr . charAt ( low ) - '0' ) ; return res ; }
if ( low == ( high - 2 ) ) { int num = eval ( expr . charAt ( low ) - '0' , expr . charAt ( low + 1 ) , expr . charAt ( low + 2 ) - '0' ) ; res . add ( num ) ; return res ; }
for ( int i = low + 1 ; i <= high ; i += 2 ) {
Vector < Integer > l = evaluateAll ( expr , low , i - 1 ) ;
Vector < Integer > r = evaluateAll ( expr , i + 1 , high ) ;
for ( int s1 = 0 ; s1 < l . size ( ) ; s1 ++ ) {
for ( int s2 = 0 ; s2 < r . size ( ) ; s2 ++ ) {
int val = eval ( l . get ( s1 ) , expr . charAt ( i ) , r . get ( s2 ) ) ; res . add ( val ) ; } } } return res ; }
public static void main ( String [ ] args ) { String expr = "1*2 + 3*4" ; int len = expr . length ( ) ; Vector < Integer > ans = evaluateAll ( expr , 0 , len - 1 ) ; for ( int i = 0 ; i < ans . size ( ) ; i ++ ) { System . out . println ( ans . get ( i ) ) ; } } }
static boolean isLucky ( int n ) {
boolean arr [ ] = new boolean [ 10 ] ; for ( int i = 0 ; i < 10 ; i ++ ) arr [ i ] = false ;
while ( n > 0 ) {
int digit = n % 10 ;
if ( arr [ digit ] ) return false ;
arr [ digit ] = true ;
n = n / 10 ; } return true ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1291 , 897 , 4566 , 1232 , 80 , 700 } ; int n = arr . length ; for ( int i = 0 ; i < n ; i ++ ) if ( isLucky ( arr [ i ] ) ) System . out . print ( arr [ i ] + " is Lucky NEW_LINE"); else System . out . print ( arr [ i ] + " is not Lucky NEW_LINE"); } }
import java . io . * ; class GFG { static void printSquares ( int n ) {
int square = 0 , odd = 1 ;
for ( int x = 0 ; x < n ; x ++ ) {
System . out . print ( square + " ▁ " ) ;
square = square + odd ; odd = odd + 2 ; } }
public static void main ( String [ ] args ) { int n = 5 ; printSquares ( n ) ; } }
class GFG { static int rev_num = 0 ; static int base_pos = 1 ; static int reversDigits ( int num ) { if ( num > 0 ) { reversDigits ( num / 10 ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
public static void main ( String [ ] args ) { int num = 4562 ; System . out . println ( reversDigits ( num ) ) ; } }
static int RecursiveFunction ( ArrayList < Integer > ref , int bit ) {
if ( ref . size ( ) == 0 bit < 0 ) return 0 ; ArrayList < Integer > curr_on = new ArrayList < > ( ) ; ArrayList < Integer > curr_off = new ArrayList < > ( ) ; for ( int i = 0 ; i < ref . size ( ) ; i ++ ) {
if ( ( ( ref . get ( i ) >> bit ) & 1 ) == 0 ) curr_off . add ( ref . get ( i ) ) ;
else curr_on . add ( ref . get ( i ) ) ; }
if ( curr_off . size ( ) == 0 ) return RecursiveFunction ( curr_on , bit - 1 ) ;
if ( curr_on . size ( ) == 0 ) return RecursiveFunction ( curr_off , bit - 1 ) ;
return Math . min ( RecursiveFunction ( curr_off , bit - 1 ) , RecursiveFunction ( curr_on , bit - 1 ) ) + ( 1 << bit ) ; }
static void PrintMinimum ( int a [ ] , int n ) { ArrayList < Integer > v = new ArrayList < > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) v . add ( a [ i ] ) ;
System . out . println ( RecursiveFunction ( v , 30 ) ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 2 , 1 } ; int size = arr . length ; PrintMinimum ( arr , size ) ; } }
static int cntElements ( int arr [ ] , int n ) {
int cnt = 0 ;
for ( int i = 0 ; i < n - 2 ; i ++ ) {
if ( arr [ i ] == ( arr [ i + 1 ] ^ arr [ i + 2 ] ) ) { cnt ++ ; } } return cnt ; }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 2 , 1 , 3 , 7 , 8 } ; int n = arr . length ; System . out . println ( cntElements ( arr , n ) ) ; } }
static int xor_triplet ( int arr [ ] , int n ) {
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i + 1 ; j < n ; j ++ ) {
for ( int k = j ; k < n ; k ++ ) { int xor1 = 0 , xor2 = 0 ;
for ( int x = i ; x < j ; x ++ ) { xor1 ^= arr [ x ] ; }
for ( int x = j ; x <= k ; x ++ ) { xor2 ^= arr [ x ] ; }
if ( xor1 == xor2 ) { ans ++ ; } } } } return ans ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . length ;
System . out . println ( xor_triplet ( arr , n ) ) ; } }
import java . util . * ; class GFG { static final int N = 100005 ; static int n , k ;
@ SuppressWarnings ( " unchecked " ) static Vector < Integer > [ ] al = new Vector [ N ] ; static long Ideal_pair ; static long [ ] bit = new long [ N ] ; static boolean [ ] root_node = new boolean [ N ] ;
static long bit_q ( int i , int j ) { long sum = 0 ; while ( j > 0 ) { sum += bit [ j ] ; j -= ( j & ( j * - 1 ) ) ; } i -- ; while ( i > 0 ) { sum -= bit [ i ] ; i -= ( i & ( i * - 1 ) ) ; } return sum ; }
static void bit_up ( int i , long diff ) { while ( i <= n ) { bit [ i ] += diff ; i += i & - i ; } }
static void dfs ( int node ) { Ideal_pair += bit_q ( Math . max ( 1 , node - k ) , Math . min ( n , node + k ) ) ; bit_up ( node , 1 ) ; for ( int i = 0 ; i < al [ node ] . size ( ) ; i ++ ) dfs ( al [ node ] . get ( i ) ) ; bit_up ( node , - 1 ) ; }
static void initialise ( ) { Ideal_pair = 0 ; for ( int i = 0 ; i <= n ; i ++ ) { root_node [ i ] = true ; bit [ i ] = 0 ; } }
static void Add_Edge ( int x , int y ) { al [ x ] . add ( y ) ; root_node [ y ] = false ; }
static long Idealpairs ( ) {
int r = - 1 ; for ( int i = 1 ; i <= n ; i ++ ) if ( root_node [ i ] ) { r = i ; break ; } dfs ( r ) ; return Ideal_pair ; }
public static void main ( String [ ] args ) { n = 6 ; k = 3 ; for ( int i = 0 ; i < al . length ; i ++ ) al [ i ] = new Vector < Integer > ( ) ; initialise ( ) ;
Add_Edge ( 1 , 2 ) ; Add_Edge ( 1 , 3 ) ; Add_Edge ( 3 , 4 ) ; Add_Edge ( 3 , 5 ) ; Add_Edge ( 3 , 6 ) ;
System . out . print ( Idealpairs ( ) ) ; } }
static void printSubsets ( int n ) { for ( int i = n ; i > 0 ; i = ( i - 1 ) & n ) System . out . print ( i + " ▁ " ) ; System . out . print ( " ▁ 0 ▁ " ) ; }
public static void main ( String [ ] args ) { int n = 9 ; printSubsets ( n ) ; } }
static boolean isDivisibleby17 ( int n ) {
if ( n == 0 n == 17 ) return true ;
if ( n < 17 ) return false ;
return isDivisibleby17 ( ( int ) ( n >> 4 ) - ( int ) ( n & 15 ) ) ; }
public static void main ( String [ ] args ) { int n = 35 ; if ( isDivisibleby17 ( n ) == true ) System . out . printf ( " % d ▁ is ▁ divisible ▁ by ▁ 17" , n ) ; else System . out . printf ( " % d ▁ is ▁ not ▁ divisible ▁ by ▁ 17" , n ) ; } }
static long answer ( long n ) {
long m = 2 ;
long ans = 1 ; long r = 1 ;
while ( r < n ) {
r = ( ( long ) Math . pow ( 2 , m ) - 1 ) * ( ( long ) Math . pow ( 2 , m - 1 ) ) ;
if ( r < n ) ans = r ;
m ++ ; } return ans ; }
public static void main ( String args [ ] ) { long n = 7 ; System . out . println ( answer ( n ) ) ; } }
import java . io . * ; class GFG { static int setBitNumber ( int n ) { if ( n == 0 ) return 0 ; int msb = 0 ; n = n / 2 ; while ( n != 0 ) { n = n / 2 ; msb ++ ; } return ( 1 << msb ) ; }
public static void main ( String [ ] args ) { int n = 0 ; System . out . println ( setBitNumber ( n ) ) ; } }
class GFG { static int setBitNumber ( int n ) {
n |= n >> 1 ;
n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ;
n = n + 1 ;
return ( n >> 1 ) ; }
public static void main ( String arg [ ] ) { int n = 273 ; System . out . print ( setBitNumber ( n ) ) ; } }
import java . io . * ; class GFG { public static int countTrailingZero ( int x ) { int count = 0 ; while ( ( x & 1 ) == 0 ) { x = x >> 1 ; count ++ ; } return count ; }
public static void main ( String [ ] args ) { System . out . println ( countTrailingZero ( 11 ) ) ; } }
import java . io . * ; class GFG { static int countTrailingZero ( int x ) {
int lookup [ ] = { 32 , 0 , 1 , 26 , 2 , 23 , 27 , 0 , 3 , 16 , 24 , 30 , 28 , 11 , 0 , 13 , 4 , 7 , 17 , 0 , 25 , 22 , 31 , 15 , 29 , 10 , 12 , 6 , 0 , 21 , 14 , 9 , 5 , 20 , 8 , 19 , 18 } ;
return lookup [ ( - x & x ) % 37 ] ; }
public static void main ( String [ ] args ) { System . out . println ( countTrailingZero ( 48 ) ) ; } }
import java . io . * ; class GFG { static int multiplyBySevenByEight ( int n ) {
return ( n - ( n >> 3 ) ) ; }
public static void main ( String args [ ] ) { int n = 9 ; System . out . println ( multiplyBySevenByEight ( n ) ) ; } }
import java . io . * ; class GFG { static int multiplyBySevenByEight ( int n ) {
return ( ( n << 3 ) - n ) >> 3 ; }
public static void main ( String args [ ] ) { int n = 15 ; System . out . println ( multiplyBySevenByEight ( n ) ) ; } }
static int countNumbers ( int L , int R , int K ) {
ArrayList < Integer > list = new ArrayList < > ( ) ;
for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) ) {
list . add ( i ) ; } }
int count = 0 ;
for ( int i = 0 ; i < list . size ( ) ; i ++ ) {
int right_index = search ( list , list . get ( i ) + K - 1 ) ;
if ( right_index != - 1 ) count = Math . max ( count , right_index - i + 1 ) ; }
return count ; }
static int search ( ArrayList < Integer > list , int num ) { int low = 0 , high = list . size ( ) - 1 ;
int ans = - 1 ; while ( low <= high ) {
int mid = low + ( high - low ) / 2 ;
if ( list . get ( mid ) <= num ) {
ans = mid ;
low = mid + 1 ; } else
high = mid - 1 ; }
return ans ; }
static boolean isPalindrome ( int n ) { int rev = 0 ; int temp = n ;
while ( n > 0 ) { rev = rev * 10 + n % 10 ; n /= 10 ; }
return rev == temp ; }
public static void main ( String args [ ] ) { int L = 98 , R = 112 ; int K = 13 ; System . out . print ( countNumbers ( L , R , K ) ) ; } }
public static int findMaximumSum ( int [ ] a , int n ) {
int prev_smaller [ ] = findPrevious ( a , n ) ;
int next_smaller [ ] = findNext ( a , n ) ; int max_value = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
max_value = Math . max ( max_value , a [ i ] * ( next_smaller [ i ] - prev_smaller [ i ] - 1 ) ) ; }
return max_value ; }
public static int [ ] findPrevious ( int [ ] a , int n ) { int ps [ ] = new int [ n ] ;
ps [ 0 ] = - 1 ;
Stack < Integer > stack = new Stack < > ( ) ;
stack . push ( 0 ) ; for ( int i = 1 ; i < a . length ; i ++ ) {
while ( stack . size ( ) > 0 && a [ stack . peek ( ) ] >= a [ i ] ) stack . pop ( ) ;
ps [ i ] = stack . size ( ) > 0 ? stack . peek ( ) : - 1 ;
stack . push ( i ) ; }
return ps ; }
public static int [ ] findNext ( int [ ] a , int n ) { int ns [ ] = new int [ n ] ; ns [ n - 1 ] = n ;
Stack < Integer > stack = new Stack < > ( ) ; stack . push ( n - 1 ) ;
for ( int i = n - 2 ; i >= 0 ; i -- ) {
while ( stack . size ( ) > 0 && a [ stack . peek ( ) ] >= a [ i ] ) stack . pop ( ) ;
ns [ i ] = stack . size ( ) > 0 ? stack . peek ( ) : a . length ;
stack . push ( i ) ; }
return ns ; }
public static void main ( String args [ ] ) { int n = 3 ; int a [ ] = { 80 , 48 , 82 } ; System . out . println ( findMaximumSum ( a , n ) ) ; } }
import java . util . * ; class GFG { static boolean compare ( int [ ] arr1 , int [ ] arr2 ) { for ( int i = 0 ; i < 256 ; i ++ ) if ( arr1 [ i ] != arr2 [ i ] ) return false ; return true ; }
static boolean search ( String pat , String txt ) { int M = pat . length ( ) ; int N = txt . length ( ) ;
int [ ] countP = new int [ 256 ] ; int [ ] countTW = new int [ 256 ] ; for ( int i = 0 ; i < 256 ; i ++ ) { countP [ i ] = 0 ; countTW [ i ] = 0 ; } for ( int i = 0 ; i < M ; i ++ ) { ( countP [ pat . charAt ( i ) ] ) ++ ; ( countTW [ txt . charAt ( i ) ] ) ++ ; }
for ( int i = M ; i < N ; i ++ ) {
if ( compare ( countP , countTW ) ) return true ;
( countTW [ txt . charAt ( i ) ] ) ++ ;
countTW [ txt . charAt ( i - M ) ] -- ; }
if ( compare ( countP , countTW ) ) return true ; return false ; }
public static void main ( String [ ] args ) { String txt = " BACDGABCDA " ; String pat = " ABCD " ; if ( search ( pat , txt ) ) System . out . println ( " Yes " ) ; else System . out . println ( " NO " ) ; } }
static double getMaxMedian ( int [ ] arr , int n , int k ) { int size = n + k ;
Arrays . sort ( arr ) ;
if ( size % 2 == 0 ) { double median = ( double ) ( arr [ ( size / 2 ) - 1 ] + arr [ size / 2 ] ) / 2 ; return median ; }
double median1 = arr [ size / 2 ] ; return median1 ; }
public static void main ( String [ ] args ) { int [ ] arr = { 3 , 2 , 3 , 4 , 2 } ; int n = arr . length ; int k = 2 ; System . out . print ( ( int ) getMaxMedian ( arr , n , k ) ) ; } }
class GFG { static void printSorted ( int a , int b , int c ) {
int get_max = Math . max ( a , Math . max ( b , c ) ) ;
int get_min = - Math . max ( - a , Math . max ( - b , - c ) ) ; int get_mid = ( a + b + c ) - ( get_max + get_min ) ; System . out . print ( get_min + " ▁ " + get_mid + " ▁ " + get_max ) ; }
public static void main ( String [ ] args ) { int a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ; } }
import java . io . * ; class GFG { static int binarySearch ( int a [ ] , int item , int low , int high ) { while ( low <= high ) { int mid = low + ( high - low ) / 2 ; if ( item == a [ mid ] ) return mid + 1 ; else if ( item > a [ mid ] ) low = mid + 1 ; else high = mid - 1 ; } return low ; }
static void insertionSort ( int a [ ] , int n ) { int i , loc , j , k , selected ; for ( i = 1 ; i < n ; ++ i ) { j = i - 1 ; selected = a [ i ] ;
loc = binarySearch ( a , selected , 0 , j ) ;
while ( j >= loc ) { a [ j + 1 ] = a [ j ] ; j -- ; } a [ j + 1 ] = selected ; } }
public static void main ( String [ ] args ) { int a [ ] = { 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 } ; int n = a . length , i ; insertionSort ( a , n ) ; System . out . println ( " Sorted ▁ array : " ) ; for ( i = 0 ; i < n ; i ++ ) System . out . print ( a [ i ] + " ▁ " ) ; } }
void sort ( int arr [ ] ) { int n = arr . length ; for ( int i = 1 ; i < n ; ++ i ) { int key = arr [ i ] ; int j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
static void printArray ( int arr [ ] ) { int n = arr . length ; for ( int i = 0 ; i < n ; ++ i ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; InsertionSort ob = new InsertionSort ( ) ; ob . sort ( arr ) ; printArray ( arr ) ; } }
import java . util . HashMap ; class GFG {
static int validPermutations ( String str ) { HashMap < Character , Integer > m = new HashMap < Character , Integer > ( ) ;
int count = str . length ( ) , ans = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { m . put ( str . charAt ( i ) , m . getOrDefault ( str . charAt ( i ) , 0 ) + 1 ) ; } for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
ans += count - m . get ( str . charAt ( i ) ) ;
m . put ( str . charAt ( i ) , m . get ( str . charAt ( i ) ) - 1 ) ; count -- ; }
return ans + 1 ; } public static void main ( String [ ] args ) { String str = " sstt " ; System . out . println ( validPermutations ( str ) ) ; } }
static int countPaths ( int n , int m ) { int dp [ ] [ ] = new int [ n + 1 ] [ m + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) dp [ i ] [ 0 ] = 1 ; for ( int i = 0 ; i <= m ; i ++ ) dp [ 0 ] [ i ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) for ( int j = 1 ; j <= m ; j ++ ) dp [ i ] [ j ] = dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] ; return dp [ n ] [ m ] ; }
public static void main ( String [ ] args ) { int n = 3 , m = 2 ; System . out . println ( " ▁ Number ▁ of ▁ Paths ▁ " + countPaths ( n , m ) ) ; } }
static int count ( int S [ ] , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 1 , 2 , 3 } ; int m = arr . length ; System . out . println ( count ( arr , m , 4 ) ) ; } }
class GFG { static boolean equalIgnoreCase ( String str1 , String str2 ) { int i = 0 ;
str1 = str1 . toUpperCase ( ) ; str2 = str2 . toUpperCase ( ) ;
int x = str1 . compareTo ( str2 ) ;
if ( x != 0 ) { return false ; } else { return true ; } }
static void equalIgnoreCaseUtil ( String str1 , String str2 ) { boolean res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) { System . out . println ( " Same " ) ; } else { System . out . println ( " Not ▁ Same " ) ; } }
public static void main ( String [ ] args ) { String str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; } }
static String replaceConsonants ( String str ) {
String res = " " ; int i = 0 , count = 0 ;
while ( i < str . length ( ) ) {
if ( str . charAt ( i ) != ' a ' && str . charAt ( i ) != ' e ' && str . charAt ( i ) != ' i ' && str . charAt ( i ) != ' o ' && str . charAt ( i ) != ' u ' ) { i ++ ; count ++ ; } else {
if ( count > 0 ) res += count ;
res += str . charAt ( i ) ; i ++ ; count = 0 ; } }
if ( count > 0 ) res += count ;
return res ; }
public static void main ( String [ ] args ) { String str = " abcdeiop " ; System . out . println ( replaceConsonants ( str ) ) ; } }
static boolean isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
static String encryptString ( String s , int n , int k ) { int countVowels = 0 ; int countConsonants = 0 ; String ans = " " ;
for ( int l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( int r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s . charAt ( r ) ) == true ) { countVowels ++ ; } else { countConsonants ++ ; } }
ans += String . valueOf ( countVowels * countConsonants ) ; } return ans ; }
static public void main ( String [ ] args ) { String s = " hello " ; int n = s . length ( ) ; int k = 2 ; System . out . println ( encryptString ( s , n , k ) ) ; } }
class GFG { private static StringBuilder charBuffer = new StringBuilder ( ) ; public static String processWords ( String input ) {
String s [ ] = input . split ( " ( \\ s ) + " ) ; for ( String values : s ) {
charBuffer . append ( values . charAt ( 0 ) ) ; } return charBuffer . toString ( ) ; }
public static void main ( String [ ] args ) { String input = " geeks ▁ forgeeks ▁ geeksfor ▁ geeks " ; System . out . println ( processWords ( input ) ) ; } }
public static String toString ( char [ ] a ) { String string = new String ( a ) ; return string ; } static void generate ( int k , char [ ] ch , int n ) {
if ( n == k ) {
System . out . print ( toString ( ch ) + " ▁ " ) ; return ; }
if ( ch [ n - 1 ] == '0' ) { ch [ n ] = '0' ; generate ( k , ch , n + 1 ) ; ch [ n ] = '1' ; generate ( k , ch , n + 1 ) ; }
if ( ch [ n - 1 ] == '1' ) { ch [ n ] = '0' ;
generate ( k , ch , n + 1 ) ; } } static void fun ( int k ) { if ( k <= 0 ) { return ; } char [ ] ch = new char [ k ] ;
ch [ 0 ] = '0' ;
generate ( k , ch , 1 ) ;
ch [ 0 ] = '1' ; generate ( k , ch , 1 ) ; } public static void main ( String args [ ] ) { int k = 3 ;
fun ( k ) ;
} }
static float findVolume ( float a ) {
if ( a < 0 ) return - 1 ;
float r = a / 2 ;
float h = a ;
float V = ( float ) ( 3.14 * Math . pow ( r , 2 ) * h ) ; return V ; }
public static void main ( String [ ] args ) { float a = 5 ; System . out . print ( findVolume ( a ) ) ; } }
public static float volumeTriangular ( int a , int b , int h ) { float vol = ( float ) ( 0.1666 ) * a * b * h ; return vol ; }
public static float volumeSquare ( int b , int h ) { float vol = ( float ) ( 0.33 ) * b * b * h ; return vol ; }
public static float volumePentagonal ( int a , int b , int h ) { float vol = ( float ) ( 0.83 ) * a * b * h ; return vol ; }
public static float volumeHexagonal ( int a , int b , int h ) { float vol = ( float ) a * b * h ; return vol ; }
public static void main ( String argc [ ] ) { int b = 4 , h = 9 , a = 4 ; System . out . println ( " Volume ▁ of ▁ triangular " + " ▁ base ▁ pyramid ▁ is ▁ " + volumeTriangular ( a , b , h ) ) ; System . out . println ( " Volume ▁ of ▁ square ▁ base " + " ▁ pyramid ▁ is ▁ " + volumeSquare ( b , h ) ) ; System . out . println ( " Volume ▁ of ▁ pentagonal " + " ▁ base ▁ pyramid ▁ is ▁ " + volumePentagonal ( a , b , h ) ) ; System . out . println ( " Volume ▁ of ▁ Hexagonal " + " ▁ base ▁ pyramid ▁ is ▁ " + volumeHexagonal ( a , b , h ) ) ; } }
static double Area ( int b1 , int b2 , int h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
public static void main ( String [ ] args ) { int base1 = 8 , base2 = 10 , height = 6 ; double area = Area ( base1 , base2 , height ) ; System . out . println ( " Area ▁ is : ▁ " + area ) ; } }
public class Diagonals { static int numberOfDiagonals ( int n ) { return n * ( n - 3 ) / 2 ; }
public static void main ( String [ ] args ) { int n = 5 ; System . out . print ( n + " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ) ; System . out . println ( numberOfDiagonals ( n ) + " ▁ diagonals " ) ; } }
static void maximumArea ( int l , int b , int x , int y ) {
int left , right , above , below ; left = x * b ; right = ( l - x - 1 ) * b ; above = l * y ; below = ( b - y - 1 ) * l ;
System . out . print ( Math . max ( Math . max ( left , right ) , Math . max ( above , below ) ) ) ; }
public static void main ( String [ ] args ) { int L = 8 , B = 8 ; int X = 0 , Y = 0 ;
maximumArea ( L , B , X , Y ) ; } }
static int delCost ( String s , int [ ] cost ) {
int ans = 0 ;
HashMap < Character , Integer > forMax = new HashMap < > ( ) ;
HashMap < Character , Integer > forTot = new HashMap < > ( ) ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( ! forMax . containsKey ( s . charAt ( i ) ) ) { forMax . put ( s . charAt ( i ) , cost [ i ] ) ; } else {
forMax . put ( s . charAt ( i ) , Math . max ( cost [ i ] , forMax . get ( s . charAt ( i ) ) ) ) ; }
if ( ! forTot . containsKey ( s . charAt ( i ) ) ) { forTot . put ( s . charAt ( i ) , cost [ i ] ) ; } else {
forTot . put ( s . charAt ( i ) , forTot . get ( s . charAt ( i ) ) + cost [ i ] ) ; } }
for ( Map . Entry < Character , Integer > i : forMax . entrySet ( ) ) {
ans += forTot . get ( i . getKey ( ) ) - i . getValue ( ) ; }
return ans ; }
String s = " AAABBB " ;
int [ ] cost = { 1 , 2 , 3 , 4 , 5 , 6 } ;
System . out . println ( delCost ( s , cost ) ) ; } }
static final int MAX = 10000 ; static Vector < Integer > [ ] divisors = new Vector [ MAX + 1 ] ;
static void computeDivisors ( ) { for ( int i = 1 ; i <= MAX ; i ++ ) { for ( int j = i ; j <= MAX ; j += i ) {
divisors [ j ] . add ( i ) ; } } }
static int getClosest ( int val1 , int val2 , int target ) { if ( target - val1 >= val2 - target ) return val2 ; else return val1 ; }
static int findClosest ( Vector < Integer > array , int n , int target ) { Integer [ ] arr = array . toArray ( new Integer [ array . size ( ) ] ) ;
if ( target <= arr [ 0 ] ) return arr [ 0 ] ; if ( target >= arr [ n - 1 ] ) return arr [ n - 1 ] ;
int i = 0 , j = n , mid = 0 ; while ( i < j ) { mid = ( i + j ) / 2 ; if ( arr [ mid ] == target ) return arr [ mid ] ;
if ( target < arr [ mid ] ) {
if ( mid > 0 && target > arr [ mid - 1 ] ) return getClosest ( arr [ mid - 1 ] , arr [ mid ] , target ) ;
j = mid ; }
else { if ( mid < n - 1 && target < arr [ mid + 1 ] ) return getClosest ( arr [ mid ] , arr [ mid + 1 ] , target ) ;
i = mid + 1 ; } }
return arr [ mid ] ; }
static void printClosest ( int N , int X ) {
computeDivisors ( ) ;
int ans = findClosest ( divisors [ N ] , divisors [ N ] . size ( ) , X ) ;
System . out . print ( ans ) ; }
int N = 16 , X = 5 ; for ( int i = 0 ; i < divisors . length ; i ++ ) divisors [ i ] = new Vector < Integer > ( ) ;
printClosest ( N , X ) ; } }
static int maxMatch ( int [ ] A , int [ ] B ) {
HashMap < Integer , Integer > Aindex = new HashMap < Integer , Integer > ( ) ;
HashMap < Integer , Integer > diff = new HashMap < Integer , Integer > ( ) ;
for ( int i = 0 ; i < A . length ; i ++ ) { Aindex . put ( A [ i ] , i ) ; }
for ( int i = 0 ; i < B . length ; i ++ ) {
if ( i - Aindex . get ( B [ i ] ) < 0 ) { if ( ! diff . containsKey ( A . length + i - Aindex . get ( B [ i ] ) ) ) { diff . put ( A . length + i - Aindex . get ( B [ i ] ) , 1 ) ; } else { diff . put ( A . length + i - Aindex . get ( B [ i ] ) , diff . get ( A . length + i - Aindex . get ( B [ i ] ) ) + 1 ) ; } }
else { if ( ! diff . containsKey ( i - Aindex . get ( B [ i ] ) ) ) { diff . put ( i - Aindex . get ( B [ i ] ) , 1 ) ; } else { diff . put ( i - Aindex . get ( B [ i ] ) , diff . get ( i - Aindex . get ( B [ i ] ) ) + 1 ) ; } } }
int max = 0 ; for ( Map . Entry < Integer , Integer > ele : diff . entrySet ( ) ) { if ( ele . getValue ( ) > max ) { max = ele . getValue ( ) ; } } return max ; }
public static void main ( String [ ] args ) { int [ ] A = { 5 , 3 , 7 , 9 , 8 } ; int [ ] B = { 8 , 7 , 3 , 5 , 9 } ;
System . out . println ( maxMatch ( A , B ) ) ; } }
import java . io . * ; import java . util . * ; class GFG { static int N = 9 ;
static boolean isinRange ( int [ ] [ ] board ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( board [ i ] [ j ] <= 0 board [ i ] [ j ] > 9 ) { return false ; } } } return true ; }
static boolean isValidSudoku ( int board [ ] [ ] ) {
if ( isinRange ( board ) == false ) { return false ; }
boolean [ ] unique = new boolean [ N + 1 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
Arrays . fill ( unique , false ) ;
for ( int j = 0 ; j < N ; j ++ ) {
int Z = board [ i ] [ j ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( int i = 0 ; i < N ; i ++ ) {
for ( int j = 0 ; j < N ; j ++ ) {
int Z = board [ j ] [ i ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( int i = 0 ; i < N - 2 ; i += 3 ) {
for ( int j = 0 ; j < N - 2 ; j += 3 ) {
Arrays . fill ( unique , false ) ;
for ( int k = 0 ; k < 3 ; k ++ ) { for ( int l = 0 ; l < 3 ; l ++ ) {
int X = i + k ;
int Y = j + l ;
int Z = board [ X ] [ Y ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } } } }
return true ; }
public static void main ( String [ ] args ) { int [ ] [ ] board = { { 7 , 9 , 2 , 1 , 5 , 4 , 3 , 8 , 6 } , { 6 , 4 , 3 , 8 , 2 , 7 , 1 , 5 , 9 } , { 8 , 5 , 1 , 3 , 9 , 6 , 7 , 2 , 4 } , { 2 , 6 , 5 , 9 , 7 , 3 , 8 , 4 , 1 } , { 4 , 8 , 9 , 5 , 6 , 1 , 2 , 7 , 3 } , { 3 , 1 , 7 , 4 , 8 , 2 , 9 , 6 , 5 } , { 1 , 3 , 6 , 7 , 4 , 8 , 5 , 9 , 2 } , { 9 , 7 , 4 , 2 , 1 , 5 , 6 , 3 , 8 } , { 5 , 2 , 8 , 6 , 3 , 9 , 4 , 1 , 7 } } ; if ( isValidSudoku ( board ) ) { System . out . println ( " Valid " ) ; } else { System . out . println ( " Not ▁ Valid " ) ; } } }
public static boolean palindrome ( int [ ] a , int i , int j ) { while ( i < j ) {
if ( a [ i ] != a [ j ] ) return false ;
i ++ ; j -- ; }
return true ; }
static int findSubArray ( int [ ] arr , int k ) { int n = arr . length ;
for ( int i = 0 ; i <= n - k ; i ++ ) { if ( palindrome ( arr , i , i + k - 1 ) ) return i ; }
return - 1 ; }
public static void main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 5 , 1 , 3 } ; int k = 4 ; int ans = findSubArray ( arr , k ) ; if ( ans == - 1 ) System . out . print ( - 1 + "NEW_LINE"); else { for ( int i = ans ; i < ans + k ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . print ( "NEW_LINE"); } } }
static void isCrossed ( String path ) { if ( path . length ( ) == 0 ) return ;
boolean ans = false ;
HashSet < Point > set = new HashSet < Point > ( ) ;
int x = 0 , y = 0 ; set . add ( new Point ( x , y ) ) ;
for ( int i = 0 ; i < path . length ( ) ; i ++ ) {
if ( path . charAt ( i ) == ' N ' ) set . add ( new Point ( x , y ++ ) ) ; if ( path . charAt ( i ) == ' S ' ) set . add ( new Point ( x , y -- ) ) ; if ( path . charAt ( i ) == ' E ' ) set . add ( new Point ( x ++ , y ) ) ; if ( path . charAt ( i ) == ' W ' ) set . add ( new Point ( x -- , y ) ) ;
if ( set . contains ( new Point ( x , y ) ) ) { ans = true ; break ; } }
if ( ans ) System . out . print ( " Crossed " ) ; else System . out . print ( " Not ▁ Crossed " ) ; }
String path = " NESW " ;
isCrossed ( path ) ; } }
static int maxWidth ( int N , int M , ArrayList < Integer > cost , ArrayList < ArrayList < Integer > > s ) {
ArrayList < ArrayList < Integer > > adj = new ArrayList < ArrayList < Integer > > ( ) ; for ( int i = 0 ; i < N ; i ++ ) { adj . add ( new ArrayList < Integer > ( ) ) ; } for ( int i = 0 ; i < M ; i ++ ) { adj . get ( s . get ( i ) . get ( 0 ) ) . add ( s . get ( i ) . get ( 1 ) ) ; }
int result = 0 ;
Queue < Integer > q = new LinkedList < > ( ) ;
q . add ( 0 ) ;
while ( q . size ( ) != 0 ) {
int count = q . size ( ) ;
result = Math . max ( count , result ) ;
while ( count -- > 0 ) {
int temp = q . remove ( ) ;
for ( int i = 0 ; i < adj . get ( temp ) . size ( ) ; i ++ ) { q . add ( adj . get ( temp ) . get ( i ) ) ; } } }
return result ; }
public static void main ( String [ ] args ) { int N = 11 , M = 10 ; ArrayList < ArrayList < Integer > > edges = new ArrayList < ArrayList < Integer > > ( ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 0 , 1 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 0 , 2 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 0 , 3 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 1 , 4 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 1 , 5 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 3 , 6 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 4 , 7 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 6 , 10 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 6 , 8 ) ) ) ; edges . add ( new ArrayList < Integer > ( Arrays . asList ( 6 , 9 ) ) ) ; ArrayList < Integer > cost = new ArrayList < Integer > ( Arrays . asList ( 1 , 2 , - 1 , 3 , 4 , 5 , 8 , 2 , 6 , 12 , 7 ) ) ;
System . out . println ( maxWidth ( N , M , cost , edges ) ) ; } }
import java . util . * ; class GFG { static final int MAX = 10000000 ;
static boolean [ ] isPrime = new boolean [ MAX + 1 ] ;
static Vector < Integer > primes = new Vector < Integer > ( ) ;
static void SieveOfEratosthenes ( ) { Arrays . fill ( isPrime , true ) ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( int i = p * p ; i <= MAX ; i += p ) isPrime [ i ] = false ; } }
for ( int p = 2 ; p <= MAX ; p ++ ) if ( isPrime [ p ] ) primes . add ( p ) ; }
static int prime_search ( Vector < Integer > primes , int diff ) {
int low = 0 ; int high = primes . size ( ) - 1 ; int res = - 1 ; while ( low <= high ) { int mid = ( low + high ) / 2 ;
if ( primes . get ( mid ) == diff ) {
return primes . get ( mid ) ; }
else if ( primes . get ( mid ) < diff ) {
low = mid + 1 ; }
else { res = primes . get ( mid ) ;
high = mid - 1 ; } }
return res ; }
static int minCost ( int arr [ ] , int n ) {
SieveOfEratosthenes ( ) ;
int res = 0 ;
for ( int i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] < arr [ i - 1 ] ) { int diff = arr [ i - 1 ] - arr [ i ] ;
int closest_prime = prime_search ( primes , diff ) ;
res += closest_prime ;
arr [ i ] += closest_prime ; } }
return res ; }
int arr [ ] = { 2 , 1 , 5 , 4 , 3 } ; int n = 5 ;
System . out . print ( minCost ( arr , n ) ) ; } }
static int count ( String s ) {
int cnt = 0 ;
for ( char c : s . toCharArray ( ) ) { cnt += c == '0' ? 1 : 0 ; }
if ( cnt % 3 != 0 ) return 0 ; int res = 0 , k = cnt / 3 , sum = 0 ;
Map < Integer , Integer > map = new HashMap < > ( ) ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
sum += s . charAt ( i ) == '0' ? 1 : 0 ;
if ( sum == 2 * k && map . containsKey ( k ) && i < s . length ( ) - 1 && i > 0 ) { res += map . get ( k ) ; }
map . put ( sum , map . getOrDefault ( sum , 0 ) + 1 ) ; }
return res ; }
String str = "01010" ;
System . out . println ( count ( str ) ) ; } }
static int splitstring ( String s ) { int n = s . length ( ) ;
int zeros = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( s . charAt ( i ) == '0' ) zeros ++ ;
if ( zeros % 3 != 0 ) return 0 ;
if ( zeros == 0 ) return ( ( n - 1 ) * ( n - 2 ) ) / 2 ;
int zerosInEachSubstring = zeros / 3 ;
int waysOfFirstCut = 0 ; int waysOfSecondCut = 0 ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( s . charAt ( i ) == '0' ) count ++ ;
if ( count == zerosInEachSubstring ) waysOfFirstCut ++ ;
else if ( count == 2 * zerosInEachSubstring ) waysOfSecondCut ++ ; }
return waysOfFirstCut * waysOfSecondCut ; }
public static void main ( String args [ ] ) { String s = "01010" ;
System . out . println ( " The ▁ number ▁ of ▁ " + " ways ▁ to ▁ split ▁ is ▁ " + splitstring ( s ) ) ; } }
static boolean canTransform ( String str1 , String str2 ) { String s1 = " " ; String s2 = " " ;
for ( char c : str1 . toCharArray ( ) ) { if ( c != ' C ' ) { s1 += c ; } } for ( char c : str2 . toCharArray ( ) ) { if ( c != ' C ' ) { s2 += c ; } }
if ( ! s1 . equals ( s2 ) ) return false ; int i = 0 ; int j = 0 ; int n = str1 . length ( ) ;
while ( i < n && j < n ) { if ( str1 . charAt ( i ) == ' C ' ) { i ++ ; } else if ( str2 . charAt ( j ) == ' C ' ) { j ++ ; }
else { if ( ( str1 . charAt ( i ) == ' A ' && i < j ) || ( str1 . charAt ( i ) == ' B ' && i > j ) ) { return false ; } i ++ ; j ++ ; } } return true ; }
public static void main ( String [ ] args ) { String str1 = " BCCABCBCA " ; String str2 = " CBACCBBAC " ;
if ( canTransform ( str1 , str2 ) ) { System . out . print ( " Yes " ) ; } else { System . out . print ( " No " ) ; } } }
static int maxsubStringLength ( char [ ] S , int N ) { int arr [ ] = new int [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) if ( S [ i ] == ' a ' S [ i ] == ' e ' S [ i ] == ' i ' S [ i ] == ' o ' S [ i ] == ' u ' ) arr [ i ] = 1 ; else arr [ i ] = - 1 ;
int maxLen = 0 ;
int curr_sum = 0 ;
HashMap < Integer , Integer > hash = new HashMap < > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { curr_sum += arr [ i ] ;
if ( curr_sum == 0 )
maxLen = Math . max ( maxLen , i + 1 ) ;
if ( hash . containsKey ( curr_sum ) ) maxLen = Math . max ( maxLen , i - hash . get ( curr_sum ) ) ;
else hash . put ( curr_sum , i ) ; }
return maxLen ; }
public static void main ( String [ ] args ) { String S = " geeksforgeeks " ; int n = S . length ( ) ; System . out . print ( maxsubStringLength ( S . toCharArray ( ) , n ) ) ; } }
import java . util . * ; class GFG { static class pair { int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } } static int [ ] [ ] mat = new int [ 1001 ] [ 1001 ] ; static int r , c , x , y ;
static int dx [ ] = { 0 , - 1 , - 1 , - 1 , 0 , 1 , 1 , 1 } ; static int dy [ ] = { 1 , 1 , 0 , - 1 , - 1 , - 1 , 0 , 1 } ;
static void FindMinimumDistance ( ) {
Queue < pair > q = new LinkedList < > ( ) ;
q . add ( new pair ( x , y ) ) ; mat [ x ] [ y ] = 0 ;
while ( ! q . isEmpty ( ) ) {
x = q . peek ( ) . first ; y = q . peek ( ) . second ;
q . remove ( ) ; for ( int i = 0 ; i < 8 ; i ++ ) { int a = x + dx [ i ] ; int b = y + dy [ i ] ;
if ( a < 0 a >= r b >= c b < 0 ) continue ;
if ( mat [ a ] [ b ] == 0 ) {
mat [ a ] [ b ] = mat [ x ] [ y ] + 1 ;
q . add ( new pair ( a , b ) ) ; } } } }
public static void main ( String [ ] args ) { r = 5 ; c = 5 ; x = 1 ; y = 1 ; int t = x ; int l = y ; mat [ x ] [ y ] = 0 ; FindMinimumDistance ( ) ; mat [ t ] [ l ] = 0 ;
for ( int i = 0 ; i < r ; i ++ ) { for ( int j = 0 ; j < c ; j ++ ) { System . out . print ( mat [ i ] [ j ] + " ▁ " ) ; } System . out . println ( ) ; } } }
public static int minOperations ( String S , int K ) {
int ans = 0 ;
for ( int i = 0 ; i < K ; i ++ ) {
int zero = 0 , one = 0 ;
for ( int j = i ; j < S . length ( ) ; j += K ) {
if ( S . charAt ( j ) == '0' ) zero ++ ;
else one ++ ; }
ans += Math . min ( zero , one ) ; }
return ans ; }
public static void main ( String args [ ] ) { String S = "110100101" ; int K = 3 ; System . out . println ( minOperations ( S , K ) ) ; } }
static int missingElement ( int arr [ ] , int n ) {
int max_ele = arr [ 0 ] ;
int min_ele = arr [ 0 ] ;
int x = 0 ;
int d ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max_ele ) max_ele = arr [ i ] ; if ( arr [ i ] < min_ele ) min_ele = arr [ i ] ; }
d = ( max_ele - min_ele ) / n ;
for ( int i = 0 ; i < n ; i ++ ) { x = x ^ arr [ i ] ; }
for ( int i = 0 ; i <= n ; i ++ ) { x = x ^ ( min_ele + ( i * d ) ) ; }
return x ; }
int arr [ ] = new int [ ] { 12 , 3 , 6 , 15 , 18 } ; int n = arr . length ;
int element = missingElement ( arr , n ) ;
System . out . print ( element ) ; } }
static void Printksubstring ( String str , int n , int k ) {
int total = ( n * ( n + 1 ) ) / 2 ;
if ( k > total ) { System . out . printf ( "-1NEW_LINE"); return ; }
int substring [ ] = new int [ n + 1 ] ; substring [ 0 ] = 0 ;
int temp = n ; for ( int i = 1 ; i <= n ; i ++ ) {
substring [ i ] = substring [ i - 1 ] + temp ; temp -- ; }
int l = 1 ; int h = n ; int start = 0 ; while ( l <= h ) { int m = ( l + h ) / 2 ; if ( substring [ m ] > k ) { start = m ; h = m - 1 ; } else if ( substring [ m ] < k ) { l = m + 1 ; } else { start = m ; break ; } }
int end = n - ( substring [ start ] - k ) ;
for ( int i = start - 1 ; i < end ; i ++ ) { System . out . print ( str . charAt ( i ) ) ; } }
public static void main ( String [ ] args ) { String str = " abc " ; int k = 4 ; int n = str . length ( ) ; Printksubstring ( str , n , k ) ; } }
static int LowerInsertionPoint ( int arr [ ] , int n , int X ) {
if ( X < arr [ 0 ] ) return 0 ; else if ( X > arr [ n - 1 ] ) return n ; int lowerPnt = 0 ; int i = 1 ; while ( i < n && arr [ i ] < X ) { lowerPnt = i ; i = i * 2 ; }
while ( lowerPnt < n && arr [ lowerPnt ] < X ) lowerPnt ++ ; return lowerPnt ; }
public static void main ( String [ ] args ) { int arr [ ] = { 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 } ; int n = arr . length ; int X = 4 ; System . out . println ( LowerInsertionPoint ( arr , n , X ) ) ; } }
static int getCount ( int M , int N ) { int count = 0 ;
if ( M == 1 ) return N ;
if ( N == 1 ) return M ; if ( N > M ) {
for ( int i = 1 ; i <= M ; i ++ ) { int numerator = N * i - N + M - i ; int denominator = M - 1 ;
if ( numerator % denominator == 0 ) { int j = numerator / denominator ;
if ( j >= 1 && j <= N ) count ++ ; } } } else {
for ( int j = 1 ; j <= N ; j ++ ) { int numerator = M * j - M + N - j ; int denominator = N - 1 ;
if ( numerator % denominator == 0 ) { int i = numerator / denominator ;
if ( i >= 1 && i <= M ) count ++ ; } } } return count ; }
public static void main ( String [ ] args ) { int M = 3 , N = 5 ; System . out . println ( getCount ( M , N ) ) ; } }
static boolean swapElement ( int [ ] arr1 , int [ ] arr2 , int n ) {
int wrongIdx = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr1 [ i ] < arr1 [ i - 1 ] ) { wrongIdx = i ; } } int maximum = Integer . MIN_VALUE ; int maxIdx = - 1 ; boolean res = false ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr2 [ i ] > maximum && arr2 [ i ] >= arr1 [ wrongIdx - 1 ] ) { if ( wrongIdx + 1 <= n - 1 && arr2 [ i ] <= arr1 [ wrongIdx + 1 ] ) { maximum = arr2 [ i ] ; maxIdx = i ; res = true ; } } }
if ( res ) { swap ( arr1 , wrongIdx , arr2 , maxIdx ) ; } return res ; } static void swap ( int [ ] a , int wrongIdx , int [ ] b , int maxIdx ) { int c = a [ wrongIdx ] ; a [ wrongIdx ] = b [ maxIdx ] ; b [ maxIdx ] = c ; }
static void getSortedArray ( int arr1 [ ] , int arr2 [ ] , int n ) { if ( swapElement ( arr1 , arr2 , n ) ) { for ( int i = 0 ; i < n ; i ++ ) { System . out . print ( arr1 [ i ] + " ▁ " ) ; } } else { System . out . println ( " Not ▁ Possible " ) ; } }
public static void main ( String [ ] args ) { int arr1 [ ] = { 1 , 3 , 7 , 4 , 10 } ; int arr2 [ ] = { 2 , 1 , 6 , 8 , 9 } ; int n = arr1 . length ; getSortedArray ( arr1 , arr2 , n ) ; } }
public static int middleOfThree ( int a , int b , int c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
public static void main ( String [ ] args ) { int a = 20 , b = 30 , c = 40 ; System . out . println ( middleOfThree ( a , b , c ) ) ; } }
static int [ ] [ ] transpose ( int [ ] [ ] mat , int row , int col ) {
int [ ] [ ] tr = new int [ col ] [ row ] ;
for ( int i = 0 ; i < row ; i ++ ) {
for ( int j = 0 ; j < col ; j ++ ) {
tr [ j ] [ i ] = mat [ i ] [ j ] ; } } return tr ; }
static void RowWiseSort ( int [ ] [ ] B ) {
for ( int i = 0 ; i < ( int ) B . length ; i ++ ) {
Arrays . sort ( B [ i ] ) ; } }
static void sortCol ( int [ ] [ ] mat , int N , int M ) {
int [ ] [ ] B = transpose ( mat , N , M ) ;
RowWiseSort ( B ) ;
mat = transpose ( B , M , N ) ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < M ; j ++ ) { System . out . print ( mat [ i ] [ j ] + " ▁ " ) ; } System . out . println ( ) ; } }
int [ ] [ ] mat = { { 1 , 6 , 10 } , { 8 , 5 , 9 } , { 9 , 4 , 15 } , { 7 , 3 , 60 } } ; int N = mat . length ; int M = mat [ 0 ] . length ;
sortCol ( mat , N , M ) ; } }
static void largestArea ( int N , int M , int [ ] H , int [ ] V ) {
Set < Integer > s1 = new HashSet < > ( ) ; Set < Integer > s2 = new HashSet < > ( ) ;
for ( int i = 1 ; i <= N + 1 ; i ++ ) s1 . add ( i ) ;
for ( int i = 1 ; i <= M + 1 ; i ++ ) s2 . add ( i ) ;
for ( int i = 0 ; i < H . length ; i ++ ) { s1 . remove ( H [ i ] ) ; }
for ( int i = 0 ; i < V . length ; i ++ ) { s2 . remove ( V [ i ] ) ; }
int [ ] list1 = new int [ s1 . size ( ) ] ; int [ ] list2 = new int [ s2 . size ( ) ] ; int i = 0 ; Iterator it1 = s1 . iterator ( ) ; while ( it1 . hasNext ( ) ) { list1 [ i ++ ] = ( int ) it1 . next ( ) ; } i = 0 ; Iterator it2 = s2 . iterator ( ) ; while ( it2 . hasNext ( ) ) { list2 [ i ++ ] = ( int ) it2 . next ( ) ; }
Arrays . sort ( list1 ) ; Arrays . sort ( list2 ) ; int maxH = 0 , p1 = 0 , maxV = 0 , p2 = 0 ;
for ( int j = 0 ; j < list1 . length ; j ++ ) { maxH = Math . max ( maxH , list1 [ j ] - p1 ) ; p1 = list1 [ j ] ; }
for ( int j = 0 ; j < list2 . length ; j ++ ) { maxV = Math . max ( maxV , list2 [ j ] - p2 ) ; p2 = list2 [ j ] ; }
System . out . println ( maxV * maxH ) ; }
int N = 3 , M = 3 ;
int [ ] H = { 2 } ; int [ ] V = { 2 } ;
largestArea ( N , M , H , V ) ; } }
static boolean checkifSorted ( int A [ ] , int B [ ] , int N ) {
boolean flag = false ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( A [ i ] > A [ i + 1 ] ) {
flag = true ; break ; } }
if ( ! flag ) { return true ; }
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( B [ i ] == 0 ) {
count ++ ; break ; } }
for ( int i = 0 ; i < N ; i ++ ) {
if ( B [ i ] == 1 ) { count ++ ; break ; } }
if ( count == 2 ) { return true ; } return false ; }
int A [ ] = { 3 , 1 , 2 } ;
int B [ ] = { 0 , 1 , 1 } ; int N = A . length ;
boolean check = checkifSorted ( A , B , N ) ;
if ( check ) { System . out . println ( " YES " ) ; }
else { System . out . println ( " NO " ) ; } } }
static int minSteps ( StringBuilder A , StringBuilder B , int M , int N ) { if ( A . charAt ( 0 ) > B . charAt ( 0 ) ) return 0 ; if ( B . charAt ( 0 ) > A . charAt ( 0 ) ) { return 1 ; }
if ( M <= N && A . charAt ( 0 ) == B . charAt ( 0 ) && count ( A , A . charAt ( 0 ) ) == M && count ( B , B . charAt ( 0 ) ) == N ) return - 1 ;
for ( int i = 1 ; i < N ; i ++ ) { if ( B . charAt ( i ) > B . charAt ( 0 ) ) return 1 ; }
for ( int i = 1 ; i < M ; i ++ ) { if ( A . charAt ( i ) < A . charAt ( 0 ) ) return 1 ; }
for ( int i = 1 ; i < M ; i ++ ) { if ( A . charAt ( i ) > A . charAt ( 0 ) ) { swap ( A , i , B , 0 ) ; swap ( A , 0 , B , 0 ) ; return 2 ; } }
for ( int i = 1 ; i < N ; i ++ ) { if ( B . charAt ( i ) < B . charAt ( 0 ) ) { swap ( A , 0 , B , i ) ; swap ( A , 0 , B , 0 ) ; return 2 ; } }
return 0 ; } static int count ( StringBuilder a , char c ) { int count = 0 ; for ( int i = 0 ; i < a . length ( ) ; i ++ ) if ( a . charAt ( i ) == c ) count ++ ; return count ; } static void swap ( StringBuilder s1 , int index1 , StringBuilder s2 , int index2 ) { char c = s1 . charAt ( index1 ) ; s1 . setCharAt ( index1 , s2 . charAt ( index2 ) ) ; s2 . setCharAt ( index2 , c ) ; }
public static void main ( String [ ] args ) { StringBuilder A = new StringBuilder ( " adsfd " ) ; StringBuilder B = new StringBuilder ( " dffff " ) ; int M = A . length ( ) ; int N = B . length ( ) ; System . out . println ( minSteps ( A , B , M , N ) ) ; } }
import java . util . * ; import java . lang . * ; class GFG { static final int maxN = 201 ;
static int n1 , n2 , n3 ;
static int [ ] [ ] [ ] dp = new int [ maxN ] [ maxN ] [ maxN ] ;
static int getMaxSum ( int i , int j , int k , int arr1 [ ] , int arr2 [ ] , int arr3 [ ] ) {
int cnt = 0 ; if ( i >= n1 ) cnt ++ ; if ( j >= n2 ) cnt ++ ; if ( k >= n3 ) cnt ++ ;
if ( cnt >= 2 ) return 0 ;
if ( dp [ i ] [ j ] [ k ] != - 1 ) return dp [ i ] [ j ] [ k ] ; int ans = 0 ;
if ( i < n1 && j < n2 )
ans = Math . max ( ans , getMaxSum ( i + 1 , j + 1 , k , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr2 [ j ] ) ; if ( i < n1 && k < n3 ) ans = Math . max ( ans , getMaxSum ( i + 1 , j , k + 1 , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr3 [ k ] ) ; if ( j < n2 && k < n3 ) ans = Math . max ( ans , getMaxSum ( i , j + 1 , k + 1 , arr1 , arr2 , arr3 ) + arr2 [ j ] * arr3 [ k ] ) ;
dp [ i ] [ j ] [ k ] = ans ;
return dp [ i ] [ j ] [ k ] ; } static void reverse ( int [ ] tmp ) { int i , k , t ; int n = tmp . length ; for ( i = 0 ; i < n / 2 ; i ++ ) { t = tmp [ i ] ; tmp [ i ] = tmp [ n - i - 1 ] ; tmp [ n - i - 1 ] = t ; } }
static int maxProductSum ( int arr1 [ ] , int arr2 [ ] , int arr3 [ ] ) {
Arrays . sort ( arr1 ) ; reverse ( arr1 ) ; Arrays . sort ( arr2 ) ; reverse ( arr2 ) ; Arrays . sort ( arr3 ) ; reverse ( arr3 ) ; return getMaxSum ( 0 , 0 , 0 , arr1 , arr2 , arr3 ) ; }
public static void main ( String [ ] args ) { n1 = 2 ; int arr1 [ ] = { 3 , 5 } ; n2 = 2 ; int arr2 [ ] = { 2 , 1 } ; n3 = 3 ; int arr3 [ ] = { 4 , 3 , 5 } ; System . out . println ( maxProductSum ( arr1 , arr2 , arr3 ) ) ; } }
static void findTriplet ( int arr [ ] , int N ) {
Arrays . sort ( arr ) ; int flag = 0 , i ;
for ( i = N - 1 ; i - 2 >= 0 ; i -- ) {
if ( arr [ i - 2 ] + arr [ i - 1 ] > arr [ i ] ) { flag = 1 ; break ; } }
if ( flag != 0 ) {
System . out . println ( arr [ i - 2 ] + " ▁ " + arr [ i - 1 ] + " ▁ " + arr [ i ] ) ; }
else { System . out . println ( - 1 ) ; } }
public static void main ( String [ ] args ) { int arr [ ] = { 4 , 2 , 10 , 3 , 5 } ; int N = arr . length ; findTriplet ( arr , N ) ; } }
static int numberofpairs ( int [ ] arr , int N ) {
int answer = 0 ;
Arrays . sort ( arr ) ;
int minDiff = 10000000 ; for ( int i = 0 ; i < N - 1 ; i ++ )
minDiff = Math . min ( minDiff , arr [ i + 1 ] - arr [ i ] ) ; for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( arr [ i + 1 ] - arr [ i ] == minDiff )
answer ++ ; }
return answer ; }
int arr [ ] = { 4 , 2 , 1 , 3 } ; int N = arr . length ;
System . out . print ( numberofpairs ( arr , N ) ) ; } }
static int max_length = 0 ;
static Vector < Integer > store = new Vector < Integer > ( ) ;
static Vector < Integer > ans = new Vector < Integer > ( ) ;
static void find_max_length ( int [ ] arr , int index , int sum , int k ) { sum = sum + arr [ index ] ; store . add ( arr [ index ] ) ; if ( sum == k ) { if ( max_length < store . size ( ) ) {
max_length = store . size ( ) ;
ans = store ; } } for ( int i = index + 1 ; i < arr . length ; i ++ ) { if ( sum + arr [ i ] <= k ) {
find_max_length ( arr , i , sum , k ) ;
store . remove ( store . size ( ) - 1 ) ; }
else return ; } return ; } static int longestSubsequence ( int [ ] arr , int n , int k ) {
Arrays . sort ( arr ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( max_length >= n - i ) break ; store . clear ( ) ; find_max_length ( arr , i , 0 , k ) ; } return max_length ; }
public static void main ( String [ ] args ) { int [ ] arr = { - 3 , 0 , 1 , 1 , 2 } ; int n = arr . length ; int k = 1 ; System . out . print ( longestSubsequence ( arr , n , k ) ) ; } }
static void sortArray ( int A [ ] , int N ) {
int x = 0 , y = 0 , z = 0 ;
if ( N % 4 == 0 N % 4 == 1 ) {
for ( int i = 0 ; i < N / 2 ; i ++ ) { x = i ; if ( i % 2 == 0 ) { y = N - i - 2 ; z = N - i - 1 ; }
A [ z ] = A [ y ] ; A [ y ] = A [ x ] ; A [ x ] = x + 1 ; }
System . out . print ( " Sorted ▁ Array : ▁ " ) ; for ( int i = 0 ; i < N ; i ++ ) System . out . print ( A [ i ] + " ▁ " ) ; }
else { System . out . print ( " - 1" ) ; } }
public static void main ( String [ ] args ) { int A [ ] = { 5 , 4 , 3 , 2 , 1 } ; int N = A . length ; sortArray ( A , N ) ; } }
static int findK ( int arr [ ] , int size , int N ) {
Arrays . sort ( arr ) ; int temp_sum = 0 ;
for ( int i = 0 ; i < size ; i ++ ) { temp_sum += arr [ i ] ;
if ( N - temp_sum == arr [ i ] * ( size - i - 1 ) ) { return arr [ i ] ; } } return - 1 ; }
public static void main ( String [ ] args ) { int [ ] arr = { 3 , 1 , 10 , 4 , 8 } ; int size = arr . length ; int N = 16 ; System . out . print ( findK ( arr , size , N ) ) ; } }
static boolean existsTriplet ( int a [ ] , int b [ ] , int c [ ] , int x , int l1 , int l2 , int l3 ) {
if ( l2 <= l1 && l2 <= l3 ) { swap ( l2 , l1 ) ; swap ( a , b ) ; } else if ( l3 <= l1 && l3 <= l2 ) { swap ( l3 , l1 ) ; swap ( a , c ) ; }
for ( int i = 0 ; i < l1 ; i ++ ) {
int j = 0 , k = l3 - 1 ; while ( j < l2 && k >= 0 ) {
if ( a [ i ] + b [ j ] + c [ k ] == x ) return true ; if ( a [ i ] + b [ j ] + c [ k ] < x ) j ++ ; else k -- ; } } return false ; }
public static void main ( String [ ] args ) { int a [ ] = { 2 , 7 , 8 , 10 , 15 } ; int b [ ] = { 1 , 6 , 7 , 8 } ; int c [ ] = { 4 , 5 , 5 } ; int l1 = a . length ; int l2 = b . length ; int l3 = c . length ; int x = 14 ; if ( existsTriplet ( a , b , c , x , l1 , l2 , l3 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
public static void printArr ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] ) ; }
public static int compare ( int num1 , int num2 ) {
String A = Integer . toString ( num1 ) ;
String B = Integer . toString ( num2 ) ;
return ( A + B ) . compareTo ( B + A ) ; }
public static void printSmallest ( int N , int [ ] arr ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) { if ( compare ( arr [ i ] , arr [ j ] ) > 0 ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } } }
printArr ( arr , N ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 2 , 9 , 21 , 1 } ; int N = arr . length ; printSmallest ( N , arr ) ; } }
class GFG { static void stableSelectionSort ( int [ ] a , int n ) {
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min = i ; for ( int j = i + 1 ; j < n ; j ++ ) if ( a [ min ] > a [ j ] ) min = j ;
int key = a [ min ] ; while ( min > i ) { a [ min ] = a [ min - 1 ] ; min -- ; } a [ i ] = key ; } } static void printArray ( int [ ] a , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( a [ i ] + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String [ ] args ) { int [ ] a = { 4 , 5 , 3 , 2 , 4 , 1 } ; int n = a . length ; stableSelectionSort ( a , n ) ; printArray ( a , n ) ; } }
static boolean isPossible ( Integer a [ ] , int b [ ] , int n , int k ) {
Arrays . sort ( a , Collections . reverseOrder ( ) ) ;
Arrays . sort ( b ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
public static void main ( String [ ] args ) { Integer a [ ] = { 2 , 1 , 3 } ; int b [ ] = { 7 , 8 , 9 } ; int k = 10 ; int n = a . length ; if ( isPossible ( a , b , n , k ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static int canReach ( String s , int L , int R ) {
int dp [ ] = new int [ s . length ( ) ] ;
int pre = 0 ;
for ( int i = 1 ; i < s . length ( ) ; i ++ ) {
if ( i >= L ) { pre += dp [ i - L ] ; }
if ( i > R ) { pre -= dp [ i - R - 1 ] ; } if ( pre > 0 && s . charAt ( i ) == '0' ) dp [ i ] = 1 ; else dp [ i ] = 0 ; }
return dp [ s . length ( ) - 1 ] ; }
public static void main ( String [ ] args ) { String S = "01101110" ; int L = 2 , R = 3 ; if ( canReach ( S , L , R ) == 1 ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int maxXORUtil ( int arr [ ] , int N , int xrr , int orr ) {
if ( N == 0 ) return xrr ^ orr ;
int x = maxXORUtil ( arr , N - 1 , xrr ^ orr , arr [ N - 1 ] ) ;
int y = maxXORUtil ( arr , N - 1 , xrr , orr arr [ N - 1 ] ) ;
return Math . max ( x , y ) ; }
static int maximumXOR ( int arr [ ] , int N ) {
return maxXORUtil ( arr , N , 0 , 0 ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 5 , 7 } ; int N = arr . length ; System . out . println ( maximumXOR ( arr , N ) ) ; } }
import java . lang . * ; import java . io . * ; import java . util . * ; class GFG { static int N = 100000 + 5 ;
static int visited [ ] = new int [ N ] ;
static void construct_tree ( int weights [ ] , int n ) { int minimum = Arrays . stream ( weights ) . min ( ) . getAsInt ( ) ; int maximum = Arrays . stream ( weights ) . max ( ) . getAsInt ( ) ;
if ( minimum == maximum ) {
System . out . println ( " No " ) ; return ; }
else {
System . out . println ( " Yes " ) ; }
int root = weights [ 0 ] ;
visited [ 1 ] = 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] != root && visited [ i + 1 ] == 0 ) { System . out . println ( 1 + " ▁ " + ( i + 1 ) + " ▁ " ) ;
visited [ i + 1 ] = 1 ; } }
int notroot = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( weights [ i ] != root ) { notroot = i + 1 ; break ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] == root && visited [ i + 1 ] == 0 ) { System . out . println ( notroot + " ▁ " + ( i + 1 ) ) ; visited [ i + 1 ] = 1 ; } } }
public static void main ( String [ ] args ) { int weights [ ] = { 1 , 2 , 1 , 2 , 5 } ; int N = weights . length ;
construct_tree ( weights , N ) ; } }
static void minCost ( String s , int k ) {
int n = s . length ( ) ;
int ans = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
int [ ] a = new int [ 26 ] ; for ( int j = i ; j < n ; j += k ) { a [ s . charAt ( j ) - ' a ' ] ++ ; }
int min_cost = Integer . MAX_VALUE ;
for ( int ch = 0 ; ch < 26 ; ch ++ ) { int cost = 0 ;
for ( int tr = 0 ; tr < 26 ; tr ++ ) cost += Math . abs ( ch - tr ) * a [ tr ] ;
min_cost = Math . min ( min_cost , cost ) ; }
ans += min_cost ; }
System . out . println ( ans ) ; }
String S = " abcdefabc " ; int K = 3 ;
minCost ( S , K ) ; } }
static int minAbsDiff ( int N ) { if ( N % 4 == 0 N % 4 == 3 ) { return 0 ; } return 1 ; }
public static void main ( String [ ] args ) { int N = 6 ; System . out . println ( minAbsDiff ( N ) ) ; } }
import java . util . * ; class GFG { static final int N = 10000 ;
@ SuppressWarnings ( " unchecked " ) static Vector < Integer > [ ] adj = new Vector [ N ] ; static int used [ ] = new int [ N ] ; static int max_matching ;
static void AddEdge ( int u , int v ) {
adj [ u ] . add ( v ) ;
adj [ v ] . add ( u ) ; }
static void Matching_dfs ( int u , int p ) { for ( int i = 0 ; i < adj [ u ] . size ( ) ; i ++ ) {
if ( adj [ u ] . get ( i ) != p ) { Matching_dfs ( adj [ u ] . get ( i ) , u ) ; } }
if ( used [ u ] == 0 && used [ p ] == 0 && p != 0 ) {
max_matching ++ ; used [ u ] = used [ p ] = 1 ; } }
static void maxMatching ( ) {
Matching_dfs ( 1 , 0 ) ;
System . out . print ( max_matching + "NEW_LINE"); }
public static void main ( String [ ] args ) { for ( int i = 0 ; i < adj . length ; i ++ ) adj [ i ] = new Vector < Integer > ( ) ;
AddEdge ( 1 , 2 ) ; AddEdge ( 1 , 3 ) ; AddEdge ( 3 , 4 ) ; AddEdge ( 3 , 5 ) ;
maxMatching ( ) ; } }
static int getMinCost ( int [ ] A , int [ ] B , int N ) { int mini = Integer . MAX_VALUE ; for ( int i = 0 ; i < N ; i ++ ) { mini = Math . min ( mini , Math . min ( A [ i ] , B [ i ] ) ) ; }
return mini * ( 2 * N - 1 ) ; }
public static void main ( String [ ] args ) { int N = 3 ; int [ ] A = { 1 , 4 , 2 } ; int [ ] B = { 10 , 6 , 12 } ; System . out . print ( getMinCost ( A , B , N ) ) ; } }
static void printVector ( ArrayList < Integer > arr ) { if ( arr . size ( ) != 1 ) {
for ( int i = 0 ; i < arr . size ( ) ; i ++ ) { System . out . print ( arr . get ( i ) + " ▁ " ) ; } System . out . println ( ) ; } }
static void findWays ( ArrayList < Integer > arr , int i , int n ) {
if ( n == 0 ) printVector ( arr ) ;
for ( int j = i ; j <= n ; j ++ ) {
arr . add ( j ) ;
findWays ( arr , j , n - j ) ;
arr . remove ( arr . size ( ) - 1 ) ; } }
int n = 4 ;
ArrayList < Integer > arr = new ArrayList < Integer > ( ) ;
findWays ( arr , 1 , n ) ; } }
public static void Maximum_subsequence ( int [ ] A , int N ) {
HashMap < Integer , Integer > frequency = new HashMap < > ( ) ;
int max_freq = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( frequency . containsKey ( A [ i ] ) ) { frequency . replace ( A [ i ] , frequency . get ( A [ i ] ) + 1 ) ; } else { frequency . put ( A [ i ] , 1 ) ; } } for ( Map . Entry it : frequency . entrySet ( ) ) {
if ( ( int ) it . getValue ( ) > max_freq ) { max_freq = ( int ) it . getValue ( ) ; } }
System . out . println ( max_freq ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 6 , 5 , 2 , 4 , 5 , 2 } ; int N = arr . length ; Maximum_subsequence ( arr , N ) ; } }
public static void DivideString ( String s , int n , int k ) { int i , c = 0 , no = 1 ; int c1 = 0 , c2 = 0 ;
int [ ] fr = new int [ 26 ] ; char [ ] ans = new char [ n ] ; for ( i = 0 ; i < n ; i ++ ) { fr [ s . charAt ( i ) - ' a ' ] ++ ; } char ch = ' a ' , ch1 = ' a ' ; for ( i = 0 ; i < 26 ; i ++ ) {
if ( fr [ i ] == k ) { c ++ ; }
if ( fr [ i ] > k && fr [ i ] != 2 * k ) { c1 ++ ; ch = ( char ) ( i + ' a ' ) ; } if ( fr [ i ] == 2 * k ) { c2 ++ ; ch1 = ( char ) ( i + ' a ' ) ; } } for ( i = 0 ; i < n ; i ++ ) ans [ i ] = '1' ; HashMap < Character , Integer > mp = new HashMap < > ( ) ; if ( c % 2 == 0 c1 > 0 c2 > 0 ) { for ( i = 0 ; i < n ; i ++ ) {
if ( fr [ s . charAt ( i ) - ' a ' ] == k ) { if ( mp . containsKey ( s . charAt ( i ) ) ) { ans [ i ] = '2' ; } else { if ( no <= ( c / 2 ) ) { ans [ i ] = '2' ; no ++ ; mp . replace ( s . charAt ( i ) , 1 ) ; } } } }
if ( ( c % 2 == 1 ) && ( c1 > 0 ) ) { no = 1 ; for ( i = 0 ; i < n ; i ++ ) { if ( s . charAt ( i ) == ch && no <= k ) { ans [ i ] = '2' ; no ++ ; } } }
if ( c % 2 == 1 && c1 == 0 ) { no = 1 ; int flag = 0 ; for ( i = 0 ; i < n ; i ++ ) { if ( s . charAt ( i ) == ch1 && no <= k ) { ans [ i ] = '2' ; no ++ ; } if ( fr [ s . charAt ( i ) - ' a ' ] == k && flag == 0 && ans [ i ] == '1' ) { ans [ i ] = '2' ; flag = 1 ; } } } System . out . println ( ans ) ; } else {
System . out . println ( " NO " ) ; } }
public static void main ( String [ ] args ) { String S = " abbbccc " ; int N = S . length ( ) ; int K = 1 ; DivideString ( S , N , K ) ; } }
static String check ( int S , int prices [ ] , int type [ ] , int n ) {
for ( int j = 0 ; j < n ; j ++ ) { for ( int k = j + 1 ; k < n ; k ++ ) {
if ( ( type [ j ] == 0 && type [ k ] == 1 ) || ( type [ j ] == 1 && type [ k ] == 0 ) ) { if ( prices [ j ] + prices [ k ] <= S ) { return " Yes " ; } } } } return " No " ; }
public static void main ( String [ ] args ) { int prices [ ] = { 3 , 8 , 6 , 5 } ; int type [ ] = { 0 , 1 , 1 , 0 } ; int S = 10 ; int n = 4 ;
System . out . print ( check ( S , prices , type , n ) ) ; } }
static int getLargestSum ( int N ) {
int max_sum = 0 ;
for ( int i = 1 ; i * i <= N ; i ++ ) { for ( int j = i + 1 ; j * j <= N ; j ++ ) {
int k = N / j ; int a = k * i ; int b = k * j ;
if ( a <= N && b <= N && a * b % ( a + b ) == 0 )
max_sum = Math . max ( max_sum , a + b ) ; } }
return max_sum ; }
public static void main ( String [ ] args ) { int N = 25 ; int max_sum = getLargestSum ( N ) ; System . out . print ( max_sum + "NEW_LINE"); } }
static String encryptString ( String str , int n ) { int i = 0 , cnt = 0 ; String encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- > 0 ) encryptedStr += str . charAt ( i ) ; i ++ ; } return encryptedStr ; }
public static void main ( String [ ] args ) { String str = " geeks " ; int n = str . length ( ) ; System . out . println ( encryptString ( str , n ) ) ; } }
static int minDiff ( int n , int x , int A [ ] ) { int mn = A [ 0 ] , mx = A [ 0 ] ;
for ( int i = 0 ; i < n ; ++ i ) { mn = Math . min ( mn , A [ i ] ) ; mx = Math . max ( mx , A [ i ] ) ; }
return Math . max ( 0 , mx - mn - 2 * x ) ; }
public static void main ( String [ ] args ) { int n = 3 , x = 3 ; int A [ ] = { 1 , 3 , 6 } ;
System . out . println ( minDiff ( n , x , A ) ) ; } }
public class BalanceParan { static long swapCount ( String s ) { char [ ] chars = s . toCharArray ( ) ;
int countLeft = 0 , countRight = 0 ;
int swap = 0 , imbalance = 0 ; for ( int i = 0 ; i < chars . length ; i ++ ) { if ( chars [ i ] == ' [ ' ) {
countLeft ++ ; if ( imbalance > 0 ) {
swap += imbalance ;
imbalance -- ; } } else if ( chars [ i ] == ' ] ' ) {
countRight ++ ;
imbalance = ( countRight - countLeft ) ; } } return swap ; }
public static void main ( String args [ ] ) { String s = " [ ] ] [ ] [ " ; System . out . println ( swapCount ( s ) ) ; s = " [ [ ] [ ] ] " ; System . out . println ( swapCount ( s ) ) ; } }
public static void longestSubSequence ( int [ ] [ ] A , int N ) {
int [ ] dp = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) {
dp [ i ] = 1 ; for ( int j = 0 ; j < i ; j ++ ) {
if ( A [ j ] [ 0 ] < A [ i ] [ 0 ] && A [ j ] [ 1 ] > A [ i ] [ 1 ] ) { dp [ i ] = Math . max ( dp [ i ] , dp [ j ] + 1 ) ; } } }
System . out . println ( dp [ N - 1 ] ) ; }
int [ ] [ ] A = { { 1 , 2 } , { 2 , 2 } , { 3 , 1 } } ; int N = A . length ;
longestSubSequence ( A , N ) ; } }
static int findWays ( int N , int dp [ ] ) {
if ( N == 0 ) { return 1 ; }
if ( dp [ N ] != - 1 ) { return dp [ N ] ; } int cnt = 0 ;
for ( int i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i , dp ) ; } }
return dp [ N ] = cnt ; }
int N = 4 ;
int [ ] dp = new int [ N + 1 ] ; for ( int i = 0 ; i < dp . length ; i ++ ) dp [ i ] = - 1 ;
System . out . print ( findWays ( N , dp ) ) ; } }
static void findWays ( int N ) {
int [ ] dp = new int [ N + 1 ] ; dp [ 0 ] = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { dp [ i ] = 0 ;
for ( int j = 1 ; j <= 6 ; j ++ ) { if ( i - j >= 0 ) { dp [ i ] = dp [ i ] + dp [ i - j ] ; } } }
System . out . print ( dp [ N ] ) ; }
int N = 4 ;
findWays ( N ) ; } }
import java . util . * ; class GFG { static int INF = ( int ) ( 1e9 + 9 ) ;
static class TrieNode { TrieNode [ ] child = new TrieNode [ 26 ] ; } ;
static void insert ( int idx , String s , TrieNode root ) { TrieNode temp = root ; for ( int i = idx ; i < s . length ( ) ; i ++ ) {
if ( temp . child [ s . charAt ( i ) - ' a ' ] == null )
temp . child [ s . charAt ( i ) - ' a ' ] = new TrieNode ( ) ; temp = temp . child [ s . charAt ( i ) - ' a ' ] ; } }
static int minCuts ( String S1 , String S2 ) { int n1 = S1 . length ( ) ; int n2 = S2 . length ( ) ;
TrieNode root = new TrieNode ( ) ; for ( int i = 0 ; i < n2 ; i ++ ) {
insert ( i , S2 , root ) ; }
int [ ] dp = new int [ n1 + 1 ] ; Arrays . fill ( dp , INF ) ;
dp [ 0 ] = 0 ; for ( int i = 0 ; i < n1 ; i ++ ) {
TrieNode temp = root ; for ( int j = i + 1 ; j <= n1 ; j ++ ) { if ( temp . child [ S1 . charAt ( j - 1 ) - ' a ' ] == null )
break ;
dp [ j ] = Math . min ( dp [ j ] , dp [ i ] + 1 ) ;
temp = temp . child [ S1 . charAt ( j - 1 ) - ' a ' ] ; } }
if ( dp [ n1 ] >= INF ) return - 1 ; else return dp [ n1 ] ; }
public static void main ( String [ ] args ) { String S1 = " abcdab " ; String S2 = " dabc " ; System . out . print ( minCuts ( S1 , S2 ) ) ; } }
static void largestSquare ( int matrix [ ] [ ] , int R , int C , int q_i [ ] , int q_j [ ] , int K , int Q ) { int countDP [ ] [ ] = new int [ R ] [ C ] ; for ( int i = 0 ; i < R ; i ++ ) for ( int j = 0 ; j < C ; j ++ ) countDP [ i ] [ j ] = 0 ;
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] ; for ( int i = 1 ; i < R ; i ++ ) countDP [ i ] [ 0 ] = countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ; for ( int j = 1 ; j < C ; j ++ ) countDP [ 0 ] [ j ] = countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ; for ( int i = 1 ; i < R ; i ++ ) for ( int j = 1 ; j < C ; j ++ ) countDP [ i ] [ j ] = matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ;
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ; int min_dist = Math . min ( Math . min ( i , j ) , Math . min ( R - i - 1 , C - j - 1 ) ) ; int ans = - 1 , l = 0 , u = min_dist ;
while ( l <= u ) { int mid = ( l + u ) / 2 ; int x1 = i - mid , x2 = i + mid ; int y1 = j - mid , y2 = j + mid ;
int count = countDP [ x2 ] [ y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 ] [ y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 ] [ y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 ] [ y1 - 1 ] ;
if ( count <= K ) { ans = 2 * mid + 1 ; l = mid + 1 ; } else u = mid - 1 ; } System . out . println ( ans ) ; } }
public static void main ( String args [ ] ) { int matrix [ ] [ ] = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int q_i [ ] = { 1 } ; int q_j [ ] = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; } }
