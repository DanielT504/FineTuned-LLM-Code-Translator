using System ; using System . Collections . Generic ; public class GFG {
static int minSum ( int [ ] A , int N ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += A [ i ] ;
if ( mp . ContainsKey ( A [ i ] ) ) { mp [ A [ i ] ] = mp [ A [ i ] ] + 1 ; } else { mp . Add ( A [ i ] , 1 ) ; } }
int minSum = int . MaxValue ;
foreach ( KeyValuePair < int , int > it in mp ) {
minSum = Math . Min ( minSum , sum - ( it . Key * it . Value ) ) ; }
return minSum ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 4 , 5 , 6 , 6 } ;
int N = arr . Length ; Console . Write ( minSum ( arr , N ) + " STRNEWLINE " ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static void maxAdjacent ( int [ ] arr , int N ) { List < int > res = new List < int > ( ) ;
for ( int i = 1 ; i < N - 1 ; i ++ ) { int prev = arr [ 0 ] ;
int maxi = Int32 . MinValue ;
for ( int j = 1 ; j < N ; j ++ ) {
if ( i == j ) continue ;
maxi = Math . Max ( maxi , Math . Abs ( arr [ j ] - prev ) ) ;
prev = arr [ j ] ; }
res . Add ( maxi ) ; }
foreach ( int x in res ) { Console . Write ( x + " ▁ " ) ; } Console . WriteLine ( ) ; }
static public void Main ( ) { int [ ] arr = { 1 , 3 , 4 , 7 , 8 } ; int N = arr . Length ; maxAdjacent ( arr , N ) ; } }
using System ; class GFG {
static int findSize ( int N ) {
if ( N == 0 ) return 1 ; if ( N == 1 ) return 1 ; int Size = 2 * findSize ( N / 2 ) + 1 ;
return Size ; }
static int CountOnes ( int N , int L , int R ) { if ( L > R ) { return 0 ; }
if ( N <= 1 ) { return N ; } int ret = 0 ; int M = N / 2 ; int Siz_M = findSize ( M ) ;
if ( L <= Siz_M ) {
ret += CountOnes ( N / 2 , L , Math . Min ( Siz_M , R ) ) ; }
if ( L <= Siz_M + 1 && Siz_M + 1 <= R ) { ret += N % 2 ; }
if ( Siz_M + 1 < R ) { ret += CountOnes ( N / 2 , Math . Max ( 1 , L - Siz_M - 1 ) , R - Siz_M - 1 ) ; } return ret ; }
static void Main ( ) {
int N = 7 , L = 2 , R = 5 ;
Console . WriteLine ( CountOnes ( N , L , R ) ) ; } }
using System ; class GFG {
static bool prime ( int n ) {
if ( n == 1 ) return false ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; }
return true ; }
static void minDivisior ( int n ) {
if ( prime ( n ) ) { Console . Write ( 1 + " ▁ " + ( n - 1 ) ) ; }
else { for ( int i = 2 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
Console . Write ( n / i + " ▁ " + ( n / i * ( i - 1 ) ) ) ; break ; } } } }
public static void Main ( String [ ] args ) { int N = 4 ;
minDivisior ( N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int Landau = int . MinValue ;
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static int lcm ( int a , int b ) { return ( a * b ) / gcd ( a , b ) ; }
static void findLCM ( List < int > arr ) { int nth_lcm = arr [ 0 ] ; for ( int i = 1 ; i < arr . Count ; i ++ ) nth_lcm = lcm ( nth_lcm , arr [ i ] ) ;
Landau = Math . Max ( Landau , nth_lcm ) ; }
static void findWays ( List < int > arr , int i , int n ) {
if ( n == 0 ) findLCM ( arr ) ;
for ( int j = i ; j <= n ; j ++ ) {
arr . Add ( j ) ;
findWays ( arr , j , n - j ) ;
arr . RemoveAt ( arr . Count - 1 ) ; } }
static void Landau_function ( int n ) { List < int > arr = new List < int > ( ) ;
findWays ( arr , 1 , n ) ;
Console . Write ( Landau ) ; }
public static void Main ( String [ ] args ) {
int N = 4 ;
Landau_function ( N ) ; } }
using System ; class GFG {
static bool isPrime ( int n ) {
if ( n == 1 ) return true ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ;
for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static void checkExpression ( int n ) { if ( isPrime ( n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; }
public static void Main ( ) { int N = 3 ; checkExpression ( N ) ; } }
using System ; class GFG {
static bool checkArray ( int n , int k , int [ ] arr ) {
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( ( arr [ i ] & 1 ) != 0 ) cnt += 1 ; }
if ( cnt >= k && cnt % 2 == k % 2 ) return true ; else return false ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 1 , 3 , 4 , 7 , 5 , 3 , 1 } ; int n = arr . Length ; int k = 4 ; if ( checkArray ( n , k , arr ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static long func ( int [ ] arr , int n ) { double ans = 0 ; int maxx = 0 ; double [ ] freq = new double [ 100005 ] ; int temp ;
for ( int i = 0 ; i < n ; i ++ ) { temp = arr [ i ] ; freq [ temp ] ++ ; maxx = Math . Max ( maxx , temp ) ; }
for ( int i = 1 ; i <= maxx ; i ++ ) { freq [ i ] += freq [ i - 1 ] ; } for ( int i = 1 ; i <= maxx ; i ++ ) { if ( freq [ i ] != 0 ) { double j ;
double cur = Math . Ceiling ( 0.5 * i ) - 1.0 ; for ( j = 1.5 ; ; j ++ ) { int val = Math . Min ( maxx , ( int ) ( Math . Ceiling ( i * j ) - 1.0 ) ) ; int times = ( int ) ( freq [ i ] - freq [ i - 1 ] ) , con = ( int ) ( j - 0.5 ) ;
ans += times * con * ( freq [ ( int ) val ] - freq [ ( int ) cur ] ) ; cur = val ; if ( val == maxx ) break ; } } }
return ( long ) ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 } ; int n = arr . Length ; Console . Write ( func ( arr , n ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static void insert_element ( int [ ] a , int n ) {
int Xor = 0 ;
int Sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { Xor ^= a [ i ] ; Sum += a [ i ] ; }
if ( Sum == 2 * Xor ) {
Console . Write ( "0" ) ; return ; }
if ( Xor == 0 ) { Console . Write ( "1" + ' STRNEWLINE ' ) ; Console . Write ( Sum ) ; return ; }
int num1 = Sum + Xor ; int num2 = Xor ;
Console . Write ( "2" ) ;
Console . Write ( num1 + " ▁ " + num2 ) ; }
public static void Main ( string [ ] args ) { int [ ] a = { 1 , 2 , 3 } ; int n = a . Length ; insert_element ( a , n ) ; } }
using System ; class GFG {
static void checkSolution ( int a , int b , int c ) { if ( a == c ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; }
public static void Main ( ) { int a = 2 , b = 0 , c = 2 ; checkSolution ( a , b , c ) ; } }
using System ; class GFG {
static bool isPerfectSquare ( double x ) {
double sr = Math . Sqrt ( x ) ;
return ( ( sr - Math . Floor ( sr ) ) == 0 ) ; }
static void checkSunnyNumber ( int N ) {
if ( isPerfectSquare ( N + 1 ) ) { Console . WriteLine ( " Yes " ) ; }
else { Console . WriteLine ( " No " ) ; } }
public static void Main ( String [ ] args ) {
int N = 8 ;
checkSunnyNumber ( N ) ; } }
using System ; class GFG {
static int countValues ( int n ) { int answer = 0 ;
for ( int i = 2 ; i <= n ; i ++ ) { int k = n ;
while ( k >= i ) { if ( k % i == 0 ) k /= i ; else k -= i ; }
if ( k == 1 ) answer ++ ; } return answer ; }
public static void Main ( ) { int N = 6 ; Console . Write ( countValues ( N ) ) ; } }
using System ; class GFG {
static void printKNumbers ( int N , int K ) {
for ( int i = 0 ; i < K - 1 ; i ++ ) Console . Write ( 1 + " ▁ " ) ;
Console . Write ( N - K + 1 ) ; }
public static void Main ( String [ ] args ) { int N = 10 , K = 3 ; printKNumbers ( N , K ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int NthSmallest ( int K ) {
List < int > Q = new List < int > ( ) ; int x = 0 ;
for ( int i = 1 ; i < 10 ; i ++ ) Q . Add ( i ) ;
for ( int i = 1 ; i <= K ; i ++ ) {
x = Q [ 0 ] ;
Q . RemoveAt ( 0 ) ;
if ( x % 10 != 0 ) {
Q . Add ( x * 10 + x % 10 - 1 ) ; }
Q . Add ( x * 10 + x % 10 ) ;
if ( x % 10 != 9 ) {
Q . Add ( x * 10 + x % 10 + 1 ) ; } }
return x ; }
public static void Main ( String [ ] args ) {
int N = 16 ; Console . Write ( NthSmallest ( N ) ) ; } }
using System ; class GFG {
static int nearest ( int n ) {
int prevSquare = ( int ) Math . Sqrt ( n ) ; int nextSquare = prevSquare + 1 ; prevSquare = prevSquare * prevSquare ; nextSquare = nextSquare * nextSquare ;
int ans = ( n - prevSquare ) < ( nextSquare - n ) ? ( prevSquare - n ) : ( nextSquare - n ) ;
return ans ; }
public static void Main ( string [ ] args ) { int n = 14 ; Console . WriteLine ( nearest ( n ) ) ; n = 16 ; Console . WriteLine ( nearest ( n ) ) ; n = 18 ; Console . WriteLine ( nearest ( n ) ) ; } }
using System ; class GFG {
static void printValueOfPi ( int N ) {
double pi = 2 * Math . Acos ( 0.0 ) ;
Console . WriteLine ( pi ) ; }
public static void Main ( ) { int N = 4 ;
printValueOfPi ( N ) ; } }
using System ; class GFG {
static void decBinary ( int [ ] arr , int n ) { int k = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n /= 2 ; } }
static int binaryDec ( int [ ] arr , int n ) { int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
static int getNum ( int n , int k ) {
int l = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) + 1 ;
int [ ] a = new int [ l ] ; decBinary ( a , n ) ;
if ( k > l ) return n ;
a [ k - 1 ] = ( a [ k - 1 ] == 0 ) ? 1 : 0 ;
return binaryDec ( a , l ) ; }
public static void Main ( String [ ] args ) { int n = 56 ; int k = 2 ; Console . WriteLine ( getNum ( n , k ) ) ; } }
using System ; class GFG { static int MAX = 1000000 ; static int MOD = 10000007 ;
static int [ ] result = new int [ MAX + 1 ] ; static int [ ] fact = new int [ MAX + 1 ] ;
static void preCompute ( ) {
fact [ 0 ] = 1 ; result [ 0 ] = 1 ;
for ( int i = 1 ; i <= MAX ; i ++ ) {
fact [ i ] = ( ( fact [ i - 1 ] % MOD ) * i ) % MOD ;
result [ i ] = ( ( result [ i - 1 ] % MOD ) * ( fact [ i ] % MOD ) ) % MOD ; } }
static void performQueries ( int [ ] q , int n ) {
preCompute ( ) ;
for ( int i = 0 ; i < n ; i ++ ) Console . WriteLine ( result [ q [ i ] ] ) ; }
public static void Main ( String [ ] args ) { int [ ] q = { 4 , 5 } ; int n = q . Length ; performQueries ( q , n ) ; } }
using System ; class GFG {
static long gcd ( long a , long b ) { if ( a == 0 ) { return b ; } return gcd ( b % a , a ) ; }
static long divTermCount ( long a , long b , long c , long num ) {
return ( ( num / a ) + ( num / b ) + ( num / c ) - ( num / ( ( a * b ) / gcd ( a , b ) ) ) - ( num / ( ( c * b ) / gcd ( c , b ) ) ) - ( num / ( ( a * c ) / gcd ( a , c ) ) ) + ( num / ( ( a * b * c ) / gcd ( gcd ( a , b ) , c ) ) ) ) ; }
static long findNthTerm ( int a , int b , int c , long n ) {
long low = 1 , high = long . MaxValue , mid ; while ( low < high ) { mid = low + ( high - low ) / 2 ;
if ( divTermCount ( a , b , c , mid ) < n ) { low = mid + 1 ; }
else { high = mid ; } } return low ; }
public static void Main ( String [ ] args ) { int a = 2 , b = 3 , c = 5 , n = 100 ; Console . WriteLine ( findNthTerm ( a , b , c , n ) ) ; } }
using System ; class GFG {
static double calculate_angle ( int n , int i , int j , int k ) {
int x , y ;
if ( i < j ) x = j - i ; else x = j + n - i ; if ( j < k ) y = k - j ; else y = k + n - j ;
double ang1 = ( 180 * x ) / n ; double ang2 = ( 180 * y ) / n ;
double ans = 180 - ang1 - ang2 ; return ans ; }
public static void Main ( ) { int n = 5 ; int a1 = 1 ; int a2 = 2 ; int a3 = 5 ; Console . WriteLine ( ( int ) calculate_angle ( n , a1 , a2 , a3 ) ) ; } }
class GFG {
static void Loss ( int SP , int P ) { double loss = 0 ; loss = ( double ) ( 2 * P * P * SP ) / ( 100 * 100 - P * P ) ; System . Console . WriteLine ( " Loss ▁ = ▁ " + System . Math . Round ( loss , 3 ) ) ; }
static void Main ( ) { int SP = 2400 , P = 30 ;
Loss ( SP , P ) ; } }
using System ; class GFG { static int MAXN = 1000001 ;
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
static bool check ( int x ) { int temp ; while ( x != 1 ) { temp = spf [ x ] ;
if ( x % temp == 0 && hash1 [ temp ] > 1 ) return false ; while ( x % temp == 0 ) x = x / temp ; } return true ; }
static bool hasValidNum ( int [ ] arr , int n ) {
sieve ( ) ; for ( int i = 0 ; i < n ; i ++ ) getFactorization ( arr [ i ] ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( check ( arr [ i ] ) ) return true ; return false ; }
static void Main ( ) { int [ ] arr = { 2 , 8 , 4 , 10 , 6 , 7 } ; int n = arr . Length ; if ( hasValidNum ( arr , n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; public class GFG {
static int countWays ( int N ) {
int E = ( N * ( N - 1 ) ) / 2 ; if ( N == 1 ) return 0 ; return ( int ) Math . Pow ( 2 , E - 1 ) ; }
static public void Main ( ) { int N = 4 ; Console . WriteLine ( countWays ( N ) ) ; } }
using System ; class GFG {
static int [ , ] l = new int [ 1001 , 1001 ] ; static void initialize ( ) {
l [ 0 , 0 ] = 1 ; for ( int i = 1 ; i < 1001 ; i ++ ) {
l [ i , 0 ] = 1 ; for ( int j = 1 ; j < i + 1 ; j ++ ) {
l [ i , j ] = ( l [ i - 1 , j - 1 ] + l [ i - 1 , j ] ) ; } } }
static int nCr ( int n , int r ) {
return l [ n , r ] ; }
public static void Main ( ) {
initialize ( ) ; int n = 8 ; int r = 3 ; Console . WriteLine ( nCr ( n , r ) ) ; } }
using System ; class GFG {
static int minAbsDiff ( int n ) { int mod = n % 4 ; if ( mod == 0 mod == 3 ) { return 0 ; } return 1 ; }
static public void Main ( ) { int n = 5 ; Console . WriteLine ( minAbsDiff ( n ) ) ; } }
using System ; class GFG { static bool check ( int s ) {
int [ ] freq = new int [ 10 ] ; int r , i ; for ( i = 0 ; i < 10 ; i ++ ) { freq [ i ] = 0 ; } while ( s != 0 ) {
r = s % 10 ;
s = ( int ) ( s / 10 ) ;
freq [ r ] += 1 ; } int xor__ = 0 ;
for ( i = 0 ; i < 10 ; i ++ ) { xor__ = xor__ ^ freq [ i ] ; if ( xor__ == 0 ) return true ; else return false ; } return true ; }
public static void Main ( ) { int s = 122233 ; if ( check ( s ) ) Console . Write ( " Yes STRNEWLINE " ) ; else Console . Write ( " No STRNEWLINE " ) ; } }
using System ; class GFG {
static void printLines ( int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) { Console . WriteLine ( k * ( 6 * i + 1 ) + " ▁ " + k * ( 6 * i + 2 ) + " ▁ " + k * ( 6 * i + 3 ) + " ▁ " + k * ( 6 * i + 5 ) ) ; } }
public static void Main ( ) { int n = 2 , k = 2 ; printLines ( n , k ) ; } }
using System ; class GFG { static int calculateSum ( int n ) {
return ( ( int ) Math . Pow ( 2 , n + 1 ) + n - 2 ) ; }
public static void Main ( ) {
int n = 4 ;
Console . WriteLine ( " Sum ▁ = ▁ " + calculateSum ( n ) ) ; } }
using System ; class GFG { const int mod = 1000000007 ;
static int count_special ( int n ) {
int [ ] fib = new int [ n + 1 ] ;
fib [ 0 ] = 1 ;
fib [ 1 ] = 2 ; for ( int i = 2 ; i <= n ; i ++ ) {
fib [ i ] = ( fib [ i - 1 ] % mod + fib [ i - 2 ] % mod ) % mod ; }
return fib [ n ] ; }
public static void Main ( ) {
int n = 3 ; Console . Write ( count_special ( n ) + " STRNEWLINE " ) ; } }
using System ; class GFG { static int mod = 1000000000 ;
static int ways ( int i , int [ ] arr , int n ) {
if ( i == n - 1 ) return 1 ; int sum = 0 ;
for ( int j = 1 ; j + i < n && j <= arr [ i ] ; j ++ ) { sum += ( ways ( i + j , arr , n ) ) % mod ; sum %= mod ; } return sum % mod ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 3 , 1 , 4 , 3 } ; int n = arr . Length ; Console . WriteLine ( ways ( 0 , arr , n ) ) ; } }
using System ; class GFG { static readonly int mod = ( int ) ( 1e9 + 7 ) ;
static int ways ( int [ ] arr , int n ) {
int [ ] dp = new int [ n + 1 ] ;
dp [ n - 1 ] = 1 ;
for ( int i = n - 2 ; i >= 0 ; i -- ) { dp [ i ] = 0 ;
for ( int j = 1 ; ( ( j + i ) < n && j <= arr [ i ] ) ; j ++ ) { dp [ i ] += dp [ i + j ] ; dp [ i ] %= mod ; } }
return dp [ 0 ] % mod ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 3 , 1 , 4 , 3 } ; int n = arr . Length ; Console . WriteLine ( ways ( arr , n ) % mod ) ; } }
using System ; class GFG { public class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static pair countSum ( int [ ] arr , int n ) {
int count_odd , count_even ;
count_odd = 0 ; count_even = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) {
if ( arr [ i - 1 ] % 2 == 0 ) { count_even = count_even + count_even + 1 ; count_odd = count_odd + count_odd ; }
else { int temp = count_even ; count_even = count_even + count_odd ; count_odd = count_odd + temp + 1 ; } } return new pair ( count_even , count_odd ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 2 , 3 } ; int n = arr . Length ;
pair ans = countSum ( arr , n ) ; Console . Write ( " EvenSum ▁ = ▁ " + ans . first ) ; Console . Write ( " ▁ OddSum ▁ = ▁ " + ans . second ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 10 ;
static List < int > numToVec ( int N ) { List < int > digit = new List < int > ( ) ;
while ( N != 0 ) { digit . Add ( N % 10 ) ; N = N / 10 ; }
if ( digit . Count == 0 ) digit . Add ( 0 ) ;
digit . Reverse ( ) ;
return digit ; }
static int solve ( List < int > A , int B , int C ) { List < int > digit = new List < int > ( ) ; int d , d2 ;
digit = numToVec ( C ) ; d = A . Count ;
if ( B > digit . Count d == 0 ) return 0 ;
else if ( B < digit . Count ) {
if ( A [ 0 ] == 0 && B != 1 ) return ( int ) ( ( d - 1 ) * Math . Pow ( d , B - 1 ) ) ; else return ( int ) Math . Pow ( d , B ) ; }
else { int [ ] dp = new int [ B + 1 ] ; int [ ] lower = new int [ MAX + 1 ] ;
for ( int i = 0 ; i < d ; i ++ ) lower [ A [ i ] + 1 ] = 1 ; for ( int i = 1 ; i <= MAX ; i ++ ) lower [ i ] = lower [ i - 1 ] + lower [ i ] ; Boolean flag = true ; dp [ 0 ] = 0 ; for ( int i = 1 ; i <= B ; i ++ ) { d2 = lower [ digit [ i - 1 ] ] ; dp [ i ] = dp [ i - 1 ] * d ;
if ( i == 1 && A [ 0 ] == 0 && B != 1 ) d2 = d2 - 1 ;
if ( flag ) dp [ i ] += d2 ;
flag = ( flag & ( lower [ digit [ i - 1 ] + 1 ] == lower [ digit [ i - 1 ] ] + 1 ) ) ; } return dp [ B ] ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 0 , 1 , 2 , 5 } ; List < int > A = new List < int > ( arr ) ; int N = 2 ; int k = 21 ; Console . WriteLine ( solve ( A , N , k ) ) ; } }
using System ; class GFG {
public static int solve ( int [ , ] dp , int wt , int K , int M , int used ) {
if ( wt < 0 ) return 0 ; if ( wt == 0 ) {
if ( used == 1 ) return 1 ; return 0 ; } if ( dp [ wt , used ] != - 1 ) return dp [ wt , used ] ; int ans = 0 ; for ( int i = 1 ; i <= K ; i ++ ) {
if ( i >= M ) ans += solve ( dp , wt - i , K , M , used 1 ) ; else ans += solve ( dp , wt - i , K , M , used ) ; } return dp [ wt , used ] = ans ; }
static void Main ( ) { int W = 3 , K = 3 , M = 2 ; int [ , ] dp = new int [ W + 1 , 2 ] ; for ( int i = 0 ; i < W + 1 ; i ++ ) for ( int j = 0 ; j < 2 ; j ++ ) dp [ i , j ] = - 1 ; Console . Write ( solve ( dp , W , K , M , 0 ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static long partitions ( int n ) { long [ ] p = new long [ n + 1 ] ;
p [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; ++ i ) { int k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 != 0 ? 1 : - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) { k *= - 1 ; } else { k = 1 - k ; } } } return p [ n ] ; }
public static void Main ( String [ ] args ) { int N = 20 ; Console . WriteLine ( partitions ( N ) ) ; } }
using System ; public class GFG {
static int LIP ( int [ , ] dp , int [ , ] mat , int n , int m , int x , int y ) {
if ( dp [ x , y ] < 0 ) { int result = 0 ;
if ( x == n - 1 && y == m - 1 ) return dp [ x , y ] = 1 ;
if ( x == n - 1 y == m - 1 ) result = 1 ;
if ( x + 1 < n && mat [ x , y ] < mat [ x + 1 , y ] ) result = 1 + LIP ( dp , mat , n , m , x + 1 , y ) ;
if ( y + 1 < m && mat [ x , y ] < mat [ x , y + 1 ] ) result = Math . Max ( result , 1 + LIP ( dp , mat , n , m , x , y + 1 ) ) ; dp [ x , y ] = result ; } return dp [ x , y ] ; }
static int wrapper ( int [ , ] mat , int n , int m ) { int [ , ] dp = new int [ 10 , 10 ] ; for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) { dp [ i , j ] = - 1 ; } } return LIP ( dp , mat , n , m , 0 , 0 ) ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 2 , 3 , 4 } , { 2 , 2 , 3 , 4 } , { 3 , 2 , 3 , 4 } , { 4 , 5 , 6 , 7 } , } ; int n = 4 , m = 4 ; Console . WriteLine ( wrapper ( mat , n , m ) ) ; } }
using System ; public class GFG {
static int countPaths ( int n , int m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
public static void Main ( ) { int n = 3 , m = 2 ; Console . WriteLine ( " ▁ Number ▁ of " + " ▁ Paths ▁ " + countPaths ( n , m ) ) ; } }
using System ; class GFG { static int MAX = 100 ;
static int getMaxGold ( int [ , ] gold , int m , int n ) {
int [ , ] goldTable = new int [ m , n ] ; for ( int i = 0 ; i < m ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) goldTable [ i , j ] = 0 ; for ( int col = n - 1 ; col >= 0 ; col -- ) { for ( int row = 0 ; row < m ; row ++ ) {
int right = ( col == n - 1 ) ? 0 : goldTable [ row , col + 1 ] ;
int right_up = ( row == 0 col == n - 1 ) ? 0 : goldTable [ row - 1 , col + 1 ] ;
int right_down = ( row == m - 1 col == n - 1 ) ? 0 : goldTable [ row + 1 , col + 1 ] ;
goldTable [ row , col ] = gold [ row , col ] + Math . Max ( right , Math . Max ( right_up , right_down ) ) ; } }
int res = goldTable [ 0 , 0 ] ; for ( int i = 1 ; i < m ; i ++ ) res = Math . Max ( res , goldTable [ i , 0 ] ) ; return res ; }
static void Main ( ) { int [ , ] gold = new int [ , ] { { 1 , 3 , 1 , 5 } , { 2 , 2 , 4 , 1 } , { 5 , 0 , 2 , 3 } , { 0 , 6 , 1 , 2 } } ; int m = 4 , n = 4 ; Console . Write ( getMaxGold ( gold , m , n ) ) ; } }
using System ; class GFG { public static int M = 100 ;
static int minAdjustmentCost ( int [ ] A , int n , int target ) {
int [ , ] dp = new int [ n , M + 1 ] ;
for ( int j = 0 ; j <= M ; j ++ ) dp [ 0 , j ] = Math . Abs ( j - A [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) {
for ( int j = 0 ; j <= M ; j ++ ) {
dp [ i , j ] = int . MaxValue ;
int k = Math . Max ( j - target , 0 ) ; for ( ; k <= Math . Min ( M , j + target ) ; k ++ ) dp [ i , j ] = Math . Min ( dp [ i , j ] , dp [ i - 1 , k ] + Math . Abs ( A [ i ] - j ) ) ; } }
int res = int . MaxValue ; for ( int j = 0 ; j <= M ; j ++ ) res = Math . Min ( res , dp [ n - 1 , j ] ) ; return res ; }
public static void Main ( ) { int [ ] arr = { 55 , 77 , 52 , 61 , 39 , 6 , 25 , 60 , 49 , 47 } ; int n = arr . Length ; int target = 10 ; Console . WriteLine ( " Minimum ▁ adjustment " + " ▁ cost ▁ is ▁ " + minAdjustmentCost ( arr , n , target ) ) ; } }
using System ; class GFG {
static int totalCombination ( int L , int R ) {
int count = 0 ;
int K = R - L ;
if ( K < L ) return 0 ;
int ans = K - L ;
count = ( ( ans + 1 ) * ( ans + 2 ) ) / 2 ;
return count ; }
public static void Main ( ) { int L = 2 , R = 6 ; Console . WriteLine ( totalCombination ( L , R ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void printArrays ( int n ) {
List < int > A = new List < int > ( ) ; List < int > B = new List < int > ( ) ;
for ( int i = 1 ; i <= 2 * n ; i ++ ) {
if ( i % 2 == 0 ) A . Add ( i ) ; else B . Add ( i ) ; }
Console . Write ( " { ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( A [ i ] ) ; if ( i != n - 1 ) Console . Write ( " , ▁ " ) ; } Console . Write ( " ▁ } STRNEWLINE " ) ;
Console . Write ( " { ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( B [ i ] ) ; if ( i != n - 1 ) Console . Write ( " , ▁ " ) ; } Console . Write ( " ▁ } " ) ; }
public static void Main ( ) { int N = 5 ;
printArrays ( N ) ; } }
using System ; class GFG {
static void flipBitsOfAandB ( int A , int B ) {
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( ( A & ( 1 << i ) ) & ( B & ( 1 << i ) ) ) != 0 ) {
A = A ^ ( 1 << i ) ;
B = B ^ ( 1 << i ) ; } }
Console . Write ( A + " ▁ " + B ) ; }
public static void Main ( string [ ] args ) { int A = 7 , B = 4 ; flipBitsOfAandB ( A , B ) ; } }
using System ; class GFG {
static int findDistinctSums ( int N ) { return ( 2 * N - 1 ) ; }
public static void Main ( ) { int N = 3 ; Console . Write ( findDistinctSums ( N ) ) ; } }
using System ; class GFG {
public static int countSubstrings ( string str ) {
int [ ] freq = new int [ 3 ] ;
int count = 0 ; int i = 0 ;
for ( int j = 0 ; j < str . Length ; j ++ ) {
freq [ str [ j ] - '0' ] ++ ;
while ( freq [ 0 ] > 0 && freq [ 1 ] > 0 && freq [ 2 ] > 0 ) { freq [ str [ i ++ ] - '0' ] -- ; }
count += i ; }
return count ; }
public static void Main ( String [ ] args ) { string str = "00021" ; Console . Write ( countSubstrings ( str ) ) ; } }
using System ; public class GFG {
static int minFlips ( string str ) {
int count = 0 ;
if ( str . Length <= 2 ) { return 0 ; }
for ( int i = 0 ; i < str . Length - 2 ; ) {
if ( str [ i ] == str [ i + 1 ] && str [ i + 2 ] == str [ i + 1 ] ) { i = i + 3 ; count ++ ; } else { i ++ ; } }
return count ; }
public static void Main ( string [ ] args ) { string S = "0011101" ; Console . WriteLine ( minFlips ( S ) ) ; } }
using System ; class GFG {
static string convertToHex ( int num ) { string temp = " " ; while ( num != 0 ) { int rem = num % 16 ; char c ; if ( rem < 10 ) { c = ( char ) ( rem + 48 ) ; } else { c = ( char ) ( rem + 87 ) ; } temp = temp + c ; num = num / 16 ; } return temp ; }
static string encryptString ( string S , int N ) { string ans = " " ;
for ( int i = 0 ; i < N ; i ++ ) { char ch = S [ i ] ; int count = 0 ; string hex ;
while ( i < N && S [ i ] == ch ) {
count ++ ; i ++ ; }
i -- ;
hex = convertToHex ( count ) ;
ans = ans + ch ;
ans = ans + hex ; }
char [ ] Ans = ans . ToCharArray ( ) ; Array . Reverse ( Ans ) ; ans = new string ( Ans ) ;
return ans ; }
static void Main ( ) {
string S = " abc " ; int N = S . Length ;
Console . WriteLine ( encryptString ( S , N ) ) ; } }
using System ; class GFG {
static int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static int countOfString ( int N ) {
int Stotal = ( int ) Math . Pow ( 2 , N ) ;
int Sequal = 0 ;
if ( N % 2 == 0 ) Sequal = binomialCoeff ( N , N / 2 ) ; int S1 = ( Stotal - Sequal ) / 2 ; return S1 ; }
public static void Main ( String [ ] args ) { int N = 3 ; Console . Write ( countOfString ( N ) ) ; } }
using System ; class GFG {
static String removeCharRecursive ( String str , char X ) {
if ( str . Length == 0 ) { return " " ; }
if ( str [ 0 ] == X ) {
return removeCharRecursive ( str . Substring ( 1 ) , X ) ; }
return str [ 0 ] + removeCharRecursive ( str . Substring ( 1 ) , X ) ; }
public static void Main ( String [ ] args ) {
String str = " geeksforgeeks " ;
char X = ' e ' ;
str = removeCharRecursive ( str , X ) ; Console . WriteLine ( str ) ; } }
using System ; class GFG {
static bool isValid ( char a1 , char a2 , string str , int flag ) { char v1 , v2 ;
if ( flag == 0 ) { v1 = str [ 4 ] ; v2 = str [ 3 ] ; } else {
v1 = str [ 1 ] ; v2 = str [ 0 ] ; }
if ( v1 != a1 && v1 != ' ? ' ) { return false ; } if ( v2 != a2 && v2 != ' ? ' ) { return false ; } return true ; }
static bool inRange ( int hh , int mm , int L , int R ) { int a = Math . Abs ( hh - mm ) ;
if ( a < L a > R ) { return false ; } return true ; }
static void displayTime ( int hh , int mm ) { if ( hh > 10 ) { Console . Write ( hh + " : " ) ; } else if ( hh < 10 ) { Console . Write ( "0" + hh + " : " ) ; } if ( mm > 10 ) { Console . Write ( mm ) ; } else if ( mm < 10 ) { Console . Write ( "0" + mm ) ; } }
static void maximumTimeWithDifferenceInRange ( string str , int L , int R ) { int i = 0 , j = 0 ; int h1 , h2 , m1 , m2 ;
for ( i = 23 ; i >= 0 ; i -- ) { h1 = i % 10 ; h2 = i / 10 ;
if ( ! isValid ( ( char ) h1 , ( char ) h2 , str , 1 ) ) { continue ; }
for ( j = 59 ; j >= 0 ; j -- ) { m1 = j % 10 ; m2 = j / 10 ;
if ( ! isValid ( ( char ) m1 , ( char ) m2 , str , 0 ) ) { continue ; } if ( inRange ( i , j , L , R ) ) { displayTime ( i , j ) ; return ; } } } if ( inRange ( i , j , L , R ) ) { displayTime ( i , j ) ; } else { Console . WriteLine ( " - 1" ) ; } }
static public void Main ( ) {
string timeValue = " ? ? : ? ? " ;
int L = 20 , R = 39 ; maximumTimeWithDifferenceInRange ( timeValue , L , R ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool check ( String s , int n ) {
Stack < int > st = new Stack < int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( st . Count != 0 && st . Peek ( ) == s [ i ] ) st . Pop ( ) ;
else st . ( s [ i ] ) ; }
if ( st . Count == 0 ) { return true ; }
else { return false ; } }
public static void Main ( String [ ] args ) {
String str = " aanncddc " ; int n = str . Length ;
if ( check ( str , n ) ) { Console . Write ( " Yes " + " STRNEWLINE " ) ; } else { Console . Write ( " No " + " STRNEWLINE " ) ; } } }
using System ; using System . Collections . Generic ; class GFG { static void findNumOfValidWords ( List < String > w , List < String > p ) {
Dictionary < int , int > m = new Dictionary < int , int > ( ) ;
List < int > res = new List < int > ( ) ;
foreach ( String s in w ) { int val = 0 ;
foreach ( char c in s . ToCharArray ( ) ) { val = val | ( 1 << ( c - ' a ' ) ) ; }
if ( m . ContainsKey ( val ) ) m [ val ] = m [ val ] + 1 ; else m . Add ( val , 1 ) ; }
foreach ( String s in p ) { int val = 0 ;
foreach ( char c in s . ToCharArray ( ) ) { val = val | ( 1 << ( c - ' a ' ) ) ; } int temp = val ; int first = s [ 0 ] - ' a ' ; int count = 0 ; while ( temp != 0 ) {
if ( ( ( temp >> first ) & 1 ) == 1 ) { if ( m . ContainsKey ( temp ) ) { count += m [ temp ] ; } }
temp = ( temp - 1 ) & val ; }
res . Add ( count ) ; }
foreach ( int it in res ) { Console . WriteLine ( it ) ; } }
public static void Main ( String [ ] args ) { List < String > arr1 = new List < String > ( ) ; arr1 . Add ( " aaaa " ) ; arr1 . Add ( " asas " ) ; arr1 . Add ( " able " ) ; arr1 . Add ( " ability " ) ; arr1 . Add ( " actt " ) ; arr1 . Add ( " actor " ) ; arr1 . Add ( " access " ) ; List < String > arr2 = new List < String > ( ) ; arr2 . Add ( " aboveyz " ) ; arr2 . Add ( " abrodyz " ) ; arr2 . Add ( " absolute " ) ; arr2 . Add ( " absoryz " ) ; arr2 . Add ( " actresz " ) ; arr2 . Add ( " gaswxyz " ) ;
findNumOfValidWords ( arr1 , arr2 ) ; } }
using System ; class GFG {
static String flip ( char [ ] s ) { for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( s [ i ] == '0' ) {
while ( s [ i ] == '0' ) {
s [ i ] = '1' ; i ++ ; }
break ; } } return new String ( s ) ; }
public static void Main ( String [ ] args ) { String s = "100010001" ; Console . WriteLine ( flip ( s . ToCharArray ( ) ) ) ; } }
using System ; class GFG {
static void getOrgString ( String s ) {
Console . Write ( s [ 0 ] ) ;
int i = 1 ; while ( i < s . Length ) {
if ( s [ i ] >= ' A ' && s [ i ] <= ' Z ' ) Console . Write ( " ▁ " + char . ToLower ( s [ i ] ) ) ;
else Console . ( s [ i ] ) ; i ++ ; } }
public static void Main ( String [ ] args ) { String s = " ILoveGeeksForGeeks " ; getOrgString ( s ) ; } }
using System ; class GFG {
static int countChar ( string str , char x ) { int count = 0 ; int n = 10 ; for ( int i = 0 ; i < str . Length ; i ++ ) if ( str [ i ] == x ) count ++ ;
int repetitions = n / str . Length ; count = count * repetitions ;
for ( int i = 0 ; i < n % str . Length ; i ++ ) { if ( str [ i ] == x ) count ++ ; } return count ; }
public static void Main ( ) { string str = " abcac " ; Console . WriteLine ( countChar ( str , ' a ' ) ) ; } }
using System ; class GFG { static void countFreq ( int [ ] arr , int n , int limit ) {
int [ ] count = new int [ limit + 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) count [ arr [ i ] ] ++ ; for ( int i = 0 ; i <= limit ; i ++ ) if ( count [ i ] > 0 ) Console . WriteLine ( i + " ▁ " + count [ i ] ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 5 , 6 , 6 , 5 , 6 , 1 , 2 , 3 , 10 , 10 } ; int n = arr . Length ; int limit = 10 ; countFreq ( arr , n , limit ) ; } }
using System ; class GFG {
static bool check ( string s , int m ) {
int l = s . Length ;
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( s [ i ] == '0' ) { c2 = 0 ;
c1 ++ ; } else { c1 = 0 ;
c2 ++ ; } if ( c1 == m c2 == m ) return true ; } return false ; }
public static void Main ( ) { String s = "001001" ; int m = 2 ;
if ( check ( s , m ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; class GFG {
static int productAtKthLevel ( string tree , int k ) { int level = - 1 ;
int product = 1 ; int n = tree . Length ; for ( int i = 0 ; i < n ; i ++ ) {
if ( tree [ i ] == ' ( ' ) level ++ ;
else if ( tree [ i ] == ' ) ' ) -- ; else {
if ( level == k ) product *= ( tree [ i ] - '0' ) ; } }
return product ; }
static void Main ( ) { string tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; Console . WriteLine ( productAtKthLevel ( tree , k ) ) ; } }
using System ; class GFG {
static void findDuplciates ( string [ ] a , int n , int m ) {
bool [ , ] isPresent = new bool [ n , m ] ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
for ( int k = 0 ; k < n ; k ++ ) { if ( a [ i ] [ j ] == a [ k ] [ j ] && i != k ) { isPresent [ i , j ] = true ; isPresent [ k , j ] = true ; } }
for ( int k = 0 ; k < m ; k ++ ) { if ( a [ i ] [ j ] == a [ i ] [ k ] && j != k ) { isPresent [ i , j ] = true ; isPresent [ i , k ] = true ; } } } } for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < m ; j ++ )
if ( ! isPresent [ i , j ] ) Console . Write ( a [ i ] [ j ] ) ; }
static void Main ( ) { int n = 2 , m = 2 ;
string [ ] a = new string [ ] { " zx " , " xz " } ;
findDuplciates ( a , n , m ) ; } }
using System ; class GFG { static bool isValidISBN ( string isbn ) {
int n = isbn . Length ; if ( n != 10 ) return false ;
int sum = 0 ; for ( int i = 0 ; i < 9 ; i ++ ) { int digit = isbn [ i ] - '0' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
char last = isbn [ 9 ] ; if ( last != ' X ' && ( last < '0' last > '9' ) ) return false ;
sum += ( ( last == ' X ' ) ? 10 : ( last - '0' ) ) ;
return ( sum % 11 == 0 ) ; }
public static void Main ( ) { string isbn = "007462542X " ; if ( isValidISBN ( isbn ) ) Console . WriteLine ( " Valid " ) ; else Console . WriteLine ( " Invalid " ) ; } }
using System ; class GFG {
static bool isVowel ( char c ) { return ( c == ' a ' c == ' A ' c == ' e ' c == ' E ' c == ' i ' c == ' I ' c == ' o ' c == ' O ' c == ' u ' c == ' U ' ) ; }
static String reverseVowel ( String str1 ) { int j = 0 ;
char [ ] str = str1 . ToCharArray ( ) ; String vowel = " " ; for ( int i = 0 ; i < str . Length ; i ++ ) { if ( isVowel ( str [ i ] ) ) { j ++ ; vowel += str [ i ] ; } }
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( isVowel ( str [ i ] ) ) { str [ i ] = vowel [ -- j ] ; } } return String . Join ( " " , str ) ; }
public static void Main ( String [ ] args ) { String str = " hello ▁ world " ; Console . WriteLine ( reverseVowel ( str ) ) ; } }
using System ; class GFG {
static String firstLetterWord ( String str ) { String result = " " ;
bool v = true ; for ( int i = 0 ; i < str . Length ; i ++ ) {
if ( str [ i ] == ' ▁ ' ) { v = true ; }
else if ( str [ i ] != ' ▁ ' && v == true ) { result += ( str [ i ] ) ; = false ; } } return result ; }
public static void Main ( ) { String str = " geeks ▁ for ▁ geeks " ; Console . WriteLine ( firstLetterWord ( str ) ) ; } }
using System ; class GFG { static int ans = 0 ;
static void dfs ( int i , int j , int [ , ] grid , bool [ , ] vis , int z , int z_count ) { int n = grid . GetLength ( 0 ) , m = grid . GetLength ( 1 ) ;
vis [ i , j ] = true ; if ( grid [ i , j ] == 0 )
z ++ ;
if ( grid [ i , j ] == 2 ) {
if ( z == z_count ) ans ++ ; vis [ i , j ] = false ; return ; }
if ( i >= 1 && ! vis [ i - 1 , j ] && grid [ i - 1 , j ] != - 1 ) dfs ( i - 1 , j , grid , vis , z , z_count ) ;
if ( i < n - 1 && ! vis [ i + 1 , j ] && grid [ i + 1 , j ] != - 1 ) dfs ( i + 1 , j , grid , vis , z , z_count ) ;
if ( j >= 1 && ! vis [ i , j - 1 ] && grid [ i , j - 1 ] != - 1 ) dfs ( i , j - 1 , grid , vis , z , z_count ) ;
if ( j < m - 1 && ! vis [ i , j + 1 ] && grid [ i , j + 1 ] != - 1 ) dfs ( i , j + 1 , grid , vis , z , z_count ) ;
vis [ i , j ] = false ; }
static int uniquePaths ( int [ , ] grid ) {
int n = grid . GetLength ( 0 ) , m = grid . GetLength ( 1 ) ; bool [ , ] vis = new bool [ n , m ] ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { vis [ i , j ] = false ; } } int x = 0 , y = 0 ; for ( int i = 0 ; i < n ; ++ i ) { for ( int j = 0 ; j < m ; ++ j ) {
if ( grid [ i , j ] == 0 ) z_count ++ ; else if ( grid [ i , j ] == 1 ) {
x = i ; y = j ; } } } dfs ( x , y , grid , vis , 0 , z_count ) ; return ans ; }
static void Main ( ) { int [ , ] grid = { { 1 , 0 , 0 , 0 } , { 0 , 0 , 0 , 0 } , { 0 , 0 , 2 , - 1 } } ; Console . WriteLine ( uniquePaths ( grid ) ) ; } }
using System ; class GFG {
static int numPairs ( int [ ] a , int n ) { int ans , i , index ;
ans = 0 ;
for ( i = 0 ; i < n ; i ++ ) a [ i ] = Math . Abs ( a [ i ] ) ;
Array . Sort ( a ) ;
for ( i = 0 ; i < n ; i ++ ) { index = 2 ; ans += index - i - 1 ; }
return ans ; }
public static void Main ( ) { int [ ] a = new int [ ] { 3 , 6 } ; int n = a . Length ; Console . Write ( numPairs ( a , n ) ) ; } }
using System ; class GFG {
static int areaOfSquare ( int S ) {
int area = S * S ; return area ; }
public static void Main ( string [ ] args ) {
int S = 5 ;
Console . Write ( areaOfSquare ( S ) ) ; } }
using System ; class GFG { static int maxPointOfIntersection ( int x , int y ) { int k = y * ( y - 1 ) / 2 ; k = k + x * ( 2 * y + x - 1 ) ; return k ; }
public static void Main ( String [ ] args ) {
int x = 3 ;
int y = 4 ;
Console . Write ( maxPointOfIntersection ( x , y ) ) ; } }
using System ; class GFG {
static int Icosihenagonal_num ( int n ) {
return ( 19 * n * n - 17 * n ) / 2 ; }
public static void Main ( ) { int n = 3 ; Console . Write ( Icosihenagonal_num ( n ) + " STRNEWLINE " ) ; n = 10 ; Console . Write ( Icosihenagonal_num ( n ) + " STRNEWLINE " ) ; } }
using System ; class GFG { static double [ ] find_Centroid ( double [ , ] v ) { double [ ] ans = new double [ 2 ] ; int n = v . GetLength ( 0 ) ; double signedArea = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { double x0 = v [ i , 0 ] , y0 = v [ i , 1 ] ; double x1 = v [ ( i + 1 ) % n , 0 ] , y1 = v [ ( i + 1 ) % n , 1 ] ;
double A = ( x0 * y1 ) - ( x1 * y0 ) ; signedArea += A ;
ans [ 0 ] += ( x0 + x1 ) * A ; ans [ 1 ] += ( y0 + y1 ) * A ; } signedArea *= 0.5 ; ans [ 0 ] = ( ans [ 0 ] ) / ( 6 * signedArea ) ; ans [ 1 ] = ( ans [ 1 ] ) / ( 6 * signedArea ) ; return ans ; }
public static void Main ( String [ ] args ) {
double [ , ] vp = { { 1 , 2 } , { 3 , - 4 } , { 6 , - 7 } } ; double [ ] ans = find_Centroid ( vp ) ; Console . WriteLine ( ans [ 0 ] + " ▁ " + ans [ 1 ] ) ; } }
using System ; class GFG {
public static void Main ( ) { int d = 10 ; double a ;
a = ( double ) ( 360 - ( 6 * d ) ) / 4 ;
Console . WriteLine ( a + " , ▁ " + ( a + d ) + " , ▁ " + ( a + ( 2 * d ) ) + " , ▁ " + ( a + ( 3 * d ) ) ) ; } }
using System ; class GFG {
static void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { z1 = - d1 / c1 ; d = Math . Abs ( ( c2 * z1 + d2 ) ) / ( float ) ( Math . Sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; Console . Write ( " Perpendicular ▁ distance ▁ is ▁ " + d ) ; } else Console . Write ( " Planes ▁ are ▁ not ▁ parallel " ) ; }
public static void Main ( ) { float a1 = 1 ; float b1 = 2 ; float c1 = - 1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = - 3 ; float d2 = - 4 ; distance ( a1 , b1 , c1 , d1 ,
using System ; class GFG {
static int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
static long numOfNecklace ( int N ) {
long ans = factorial ( N ) / ( factorial ( N / 2 ) * factorial ( N / 2 ) ) ;
ans = ans * factorial ( N / 2 - 1 ) ; ans = ans * factorial ( N / 2 - 1 ) ;
ans /= 2 ;
return ans ; }
static public void Main ( ) {
int N = 4 ;
Console . Write ( numOfNecklace ( N ) ) ; } }
using System ; using System . Linq ; using System . Collections . Generic ; class GFG {
static string isDivisibleByDivisor ( int S , int D ) {
S %= D ;
List < int > hashMap = new List < int > ( ) ; ; hashMap . Add ( S ) ; for ( int i = 0 ; i <= D ; i ++ ) {
S += ( S % D ) ; S %= D ;
if ( hashMap . Contains ( S ) ) {
if ( S == 0 ) { return " Yes " ; } return " No " ; }
else hashMap . ( S ) ; } return " Yes " ; }
static void Main ( ) { int S = 3 , D = 6 ; Console . Write ( isDivisibleByDivisor ( S , D ) ) ; } }
using System ; class GFG {
public static void minimumSteps ( int x , int y ) {
int cnt = 0 ;
while ( x != 0 && y != 0 ) {
if ( x > y ) {
cnt += x / y ; x %= y ; }
else {
cnt += y / x ; y %= x ; } } cnt -- ;
if ( x > 1 y > 1 ) cnt = - 1 ;
Console . WriteLine ( cnt ) ; }
public static void Main ( ) {
int x = 3 , y = 1 ; minimumSteps ( x , y ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static int countMinReversals ( string expr ) { int len = expr . Length ;
if ( len % 2 != 0 ) { return - 1 ; }
Stack < char > s = new Stack < char > ( ) ; for ( int i = 0 ; i < len ; i ++ ) { char c = expr [ i ] ; if ( c == ' } ' && s . Count > 0 ) { if ( s . Peek ( ) == ' { ' ) { s . Pop ( ) ; } else { s . Push ( c ) ; } } else { s . Push ( c ) ; } }
int red_len = s . Count ;
int n = 0 ; while ( s . Count > 0 && s . Peek ( ) == ' { ' ) { s . Pop ( ) ; n ++ ; }
return ( red_len / 2 + n % 2 ) ; }
public static void Main ( string [ ] args ) { string expr = " } } { { " ; Console . WriteLine ( countMinReversals ( expr ) ) ; } }
using System ; public class GFG {
static int countMinReversals ( String expr ) { int len = expr . Length ; int ans ;
if ( len % 2 != 0 ) { return - 1 ; } int left_brace = 0 , right_brace = 0 ; for ( int i = 0 ; i < len ; i ++ ) { char ch = expr [ i ] ;
if ( ch == ' { ' ) { left_brace ++ ; }
else { if ( left_brace == 0 ) { right_brace ++ ; } else { left_brace -- ; } } } ans = ( int ) ( Math . Ceiling ( ( 0.0 + left_brace ) / 2 ) + Math . Ceiling ( ( 0.0 + right_brace ) / 2 ) ) ; return ans ; }
public static void Main ( String [ ] args ) { String expr = " } } { { " ; Console . WriteLine ( countMinReversals ( expr ) ) ; } }
using System ; class GFG {
static void PrintMinNumberForPattern ( String arr ) {
int curr_max = 0 ;
int last_entry = 0 ; int j ;
for ( int i = 0 ; i < arr . Length ; i ++ ) {
int noOfNextD = 0 ; switch ( arr [ i ] ) { case ' I ' :
j = i + 1 ; while ( j < arr . Length && arr [ j ] == ' D ' ) { noOfNextD ++ ; j ++ ; } if ( i == 0 ) { curr_max = noOfNextD + 2 ;
Console . Write ( " ▁ " + ++ last_entry ) ; Console . Write ( " ▁ " + curr_max ) ;
last_entry = curr_max ; } else {
curr_max = curr_max + noOfNextD + 1 ;
last_entry = curr_max ; Console . Write ( " ▁ " + last_entry ) ; }
for ( int k = 0 ; k < noOfNextD ; k ++ ) { Console . Write ( " ▁ " + -- last_entry ) ; i ++ ; } break ;
' D ' : if ( i == 0 ) {
j = i + 1 ; while ( j < arr . Length && arr [ j ] == ' D ' ) { noOfNextD ++ ; j ++ ; }
curr_max = noOfNextD + 2 ;
Console . Write ( " ▁ " + curr_max + " ▁ " + ( curr_max - 1 ) ) ;
last_entry = curr_max - 1 ; } else {
Console . Write ( " ▁ " + ( last_entry - 1 ) ) ; last_entry -- ; } break ; } } Console . WriteLine ( ) ; }
public static void Main ( String [ ] args ) { PrintMinNumberForPattern ( " IDID " ) ; PrintMinNumberForPattern ( " I " ) ; PrintMinNumberForPattern ( " DD " ) ; PrintMinNumberForPattern ( " II " ) ; PrintMinNumberForPattern ( " DIDI " ) ; PrintMinNumberForPattern ( " IIDDD " ) ; PrintMinNumberForPattern ( " DDIDDIID " ) ; } }
using System ; using System . Collections . Generic ; class GFG { static void printLeast ( String arr ) {
int min_avail = 1 , pos_of_I = 0 ;
List < int > al = new List < int > ( ) ;
if ( arr [ 0 ] == ' I ' ) { al . Add ( 1 ) ; al . Add ( 2 ) ; min_avail = 3 ; pos_of_I = 1 ; } else { al . Add ( 2 ) ; al . Add ( 1 ) ; min_avail = 3 ; pos_of_I = 0 ; }
for ( int i = 1 ; i < arr . Length ; i ++ ) { if ( arr [ i ] == ' I ' ) { al . Add ( min_avail ) ; min_avail ++ ; pos_of_I = i + 1 ; } else { al . Add ( al [ i ] ) ; for ( int j = pos_of_I ; j <= i ; j ++ ) al [ j ] = al [ j ] + 1 ; min_avail ++ ; } }
for ( int i = 0 ; i < al . Count ; i ++ ) Console . Write ( al [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( String [ ] args ) { printLeast ( " IDID " ) ; printLeast ( " I " ) ; printLeast ( " DD " ) ; printLeast ( " II " ) ; printLeast ( " DIDI " ) ; printLeast ( " IIDDD " ) ; printLeast ( " DDIDDIID " ) ; } }
using System ; using System . Collections ; public class GFG {
static void PrintMinNumberForPattern ( String seq ) {
String result = " " ;
Stack stk = new Stack ( ) ;
for ( int i = 0 ; i <= seq . Length ; i ++ ) {
stk . Push ( i + 1 ) ;
if ( i == seq . Length seq [ i ] == ' I ' ) {
while ( stk . Count != 0 ) {
result += String . Join ( " " , stk . Peek ( ) ) ; result += " ▁ " ; stk . Pop ( ) ; } } } Console . WriteLine ( result ) ; }
public static void Main ( ) { PrintMinNumberForPattern ( " IDID " ) ; PrintMinNumberForPattern ( " I " ) ; PrintMinNumberForPattern ( " DD " ) ; PrintMinNumberForPattern ( " II " ) ; PrintMinNumberForPattern ( " DIDI " ) ; PrintMinNumberForPattern ( " IIDDD " ) ; PrintMinNumberForPattern ( " DDIDDIID " ) ; } }
using System ; class GFG {
static String getMinNumberForPattern ( String seq ) { int n = seq . Length ; if ( n >= 9 ) return " - 1" ; char [ ] result = new char [ n + 1 ] ; int count = 1 ;
for ( int i = 0 ; i <= n ; i ++ ) { if ( i == n seq [ i ] == ' I ' ) { for ( int j = i - 1 ; j >= - 1 ; j -- ) { result [ j + 1 ] = ( char ) ( ( int ) '0' + count ++ ) ; if ( j >= 0 && seq [ j ] == ' I ' ) break ; } } } return new String ( result ) ; }
public static void Main ( ) { String [ ] inputs = { " IDID " , " I " , " DD " , " II " , " DIDI " , " IIDDD " , " DDIDDIID " } ; foreach ( String input in inputs ) { Console . WriteLine ( getMinNumberForPattern ( input ) ) ; } } }
using System ; class GFG {
static bool isPrime ( int n ) { int i , c = 0 ; for ( i = 1 ; i < n / 2 ; i ++ ) { if ( n % i == 0 ) { c ++ ; } } if ( c == 1 ) { return true ; } else { return false ; } }
static void findMinNum ( int [ ] arr , int n ) {
int first = 0 , last = 0 , num , rev , i ; int [ ] hash = new int [ 10 ] ;
for ( i = 0 ; i < n ; i ++ ) { hash [ arr [ i ] ] ++ ; }
Console . Write ( " Minimum ▁ number : ▁ " ) ; for ( i = 0 ; i <= 9 ; i ++ ) {
for ( int j = 0 ; j < hash [ i ] ; j ++ ) { Console . Write ( i ) ; } } Console . WriteLine ( ) ; Console . WriteLine ( ) ;
for ( i = 0 ; i <= 9 ; i ++ ) { if ( hash [ i ] != 0 ) { first = i ; break ; } }
for ( i = 9 ; i >= 0 ; i -- ) { if ( hash [ i ] != 0 ) { last = i ; break ; } } num = first * 10 + last ; rev = last * 10 + first ;
Console . Write ( " Prime ▁ combinations : ▁ " ) ; if ( isPrime ( num ) && isPrime ( rev ) ) { Console . WriteLine ( num + " ▁ " + rev ) ; } else if ( isPrime ( num ) ) { Console . WriteLine ( num ) ; } else if ( isPrime ( rev ) ) { Console . WriteLine ( rev ) ; } else { Console . WriteLine ( " No ▁ combinations ▁ exist " ) ; } }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 4 , 7 , 8 } ; findMinNum ( arr , 5 ) ; } }
using System ; class GFG {
static int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
static bool coprime ( int a , int b ) {
return ( gcd ( a , b ) == 1 ) ; }
static void possibleTripletInRange ( int L , int R ) { bool flag = false ; int possibleA = 0 , possibleB = 0 , possibleC = 0 ;
for ( int a = L ; a <= R ; a ++ ) { for ( int b = a + 1 ; b <= R ; b ++ ) { for ( int c = b + 1 ; c <= R ; c ++ ) {
if ( coprime ( a , b ) && coprime ( b , c ) && ! coprime ( a , c ) ) { flag = true ; possibleA = a ; possibleB = b ; possibleC = c ; break ; } } } }
if ( flag == true ) { Console . WriteLine ( " ( " + possibleA + " , ▁ " + possibleB + " , ▁ " + possibleC + " ) " + " ▁ is ▁ one ▁ such ▁ possible ▁ triplet ▁ " + " between ▁ " + L + " ▁ and ▁ " + R ) ; } else { Console . WriteLine ( " No ▁ Such ▁ Triplet ▁ exists " + " between ▁ " + L + " ▁ and ▁ " + R ) ; } }
public static void Main ( ) { int L , R ;
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ; } }
using System ; public class GFG {
public static bool possibleToReach ( int a , int b ) {
int c = ( int ) Math . Pow ( a * b , ( double ) 1 / 3 ) ;
int re1 = a / c ; int re2 = b / c ;
if ( ( re1 * re1 * re2 == a ) && ( re2 * re2 * re1 == b ) ) return true ; else return false ; }
static public void Main ( String [ ] args ) { int A = 60 , B = 450 ; if ( possibleToReach ( A , B ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG { public static bool isUndulating ( string n ) {
if ( n . Length <= 2 ) return false ;
for ( int i = 2 ; i < n . Length ; i ++ ) if ( n [ i - 2 ] != n [ i ] ) return false ; return true ; }
public static void Main ( ) { string n = "1212121" ; if ( isUndulating ( n ) == true ) Console . WriteLine ( " yes " ) ; else Console . WriteLine ( " no " ) ; } }
using System ; class GFG {
static int Series ( int n ) { int i ; int sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
public static void Main ( ) { int n = 3 ; int res = Series ( n ) ; Console . Write ( res ) ; } }
using System ; public class GfG {
public static int counLastDigitK ( int low , int high , int k ) { int mlow = 10 * Convert . ToInt32 ( Math . Ceiling ( low / 10.0 ) ) ; int mhigh = 10 * Convert . ToInt32 ( Math . Floor ( high / 10.0 ) ) ; int count = ( mhigh - mlow ) / 10 ; if ( high % 10 >= k ) count ++ ; if ( low % 10 <= k && ( low % 10 ) > 0 ) count ++ ; return count ; }
public static void Main ( ) { int low = 3 , high = 35 , k = 3 ; Console . WriteLine ( counLastDigitK ( low , high , k ) ) ; } }
using System ; class GFG {
static int sum ( int L , int R ) {
int p = R / 6 ;
int q = ( L - 1 ) / 6 ;
int sumR = 3 * ( p * ( p + 1 ) ) ;
int sumL = ( q * ( q + 1 ) ) * 3 ;
return sumR - sumL ; }
public static void Main ( ) { int L = 1 , R = 20 ; Console . WriteLine ( sum ( L , R ) ) ; } }
using System ; class GFG {
static String prevNum ( String str ) { int len = str . Length ; int index = - 1 ;
for ( int i = len - 2 ; i >= 0 ; i -- ) { if ( str [ i ] > str [ i + 1 ] ) { index = i ; break ; } }
int smallGreatDgt = - 1 ; for ( int i = len - 1 ; i > index ; i -- ) { if ( str [ i ] < str [ index ] ) { if ( smallGreatDgt == - 1 ) { smallGreatDgt = i ; } else if ( str [ i ] >= str [ smallGreatDgt ] ) { smallGreatDgt = i ; } } }
if ( index == - 1 ) { return " - 1" ; }
if ( smallGreatDgt != - 1 ) { str = swap ( str , index , smallGreatDgt ) ; return str ; } return " - 1" ; } static String swap ( String str , int i , int j ) { char [ ] ch = str . ToCharArray ( ) ; char temp = ch [ i ] ; ch [ i ] = ch [ j ] ; ch [ j ] = temp ; return String . Join ( " " , ch ) ; }
public static void Main ( String [ ] args ) { String str = "34125" ; Console . WriteLine ( prevNum ( str ) ) ; } }
using System ; class GFG {
static int horner ( int [ ] poly , int n , int x ) {
for ( int i = 1 ; i < n ; i ++ ) result = result * x + poly [ i ] ; return result ; }
static int findSign ( int [ ] poly , int n , int x ) { int result = horner ( poly , n , x ) ; if ( result > 0 ) return 1 ; else if ( result < 0 ) return - 1 ; return 0 ; }
public static void Main ( ) {
int [ ] poly = { 2 , - 6 , 2 , - 1 } ; int x = 3 ; int n = poly . Length ; Console . Write ( " Sign ▁ of ▁ polynomial ▁ is ▁ " + findSign ( poly , n , x ) ) ; } }
class GFG { static int MAX = 100005 ;
static bool [ ] isPrime = new bool [ MAX ] ;
static void sieveOfEratostheneses ( ) { isPrime [ 1 ] = true ; for ( int i = 2 ; i * i < MAX ; i ++ ) { if ( ! isPrime [ i ] ) { for ( int j = 2 * i ; j < MAX ; j += i ) isPrime [ j ] = true ; } } }
static int findPrime ( int n ) { int num = n + 1 ;
while ( num > 0 ) {
if ( ! isPrime [ num ] ) return num ;
num = num + 1 ; } return 0 ; }
static int minNumber ( int [ ] arr , int n ) {
sieveOfEratostheneses ( ) ; int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( ! isPrime [ sum ] ) return 0 ;
int num = findPrime ( sum ) ;
return num - sum ; }
public static void Main ( ) { int [ ] arr = { 2 , 4 , 6 , 8 , 12 } ; int n = arr . Length ; System . Console . WriteLine ( minNumber ( arr , n ) ) ; } }
using System ; class GFG {
public static long SubArraySum ( int [ ] arr , int n ) { long result = 0 , temp = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
temp = 0 ; for ( int j = i ; j < n ; j ++ ) {
temp += arr [ j ] ; result += temp ; } } return result ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int n = arr . Length ; Console . Write ( " Sum ▁ of ▁ SubArray ▁ : ▁ " + SubArraySum ( arr , n ) ) ; } }
using System ; class GFG { static int highestPowerof2 ( int n ) { int p = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) ; return ( int ) Math . Pow ( 2 , p ) ; }
static public void Main ( ) { int n = 10 ; Console . WriteLine ( highestPowerof2 ( n ) ) ; } }
using System ; class GFG {
static int aModM ( string s , int mod ) { int number = 0 ; for ( int i = 0 ; i < s . Length ; i ++ ) {
number = ( number * 10 ) ; int x = ( int ) ( s [ i ] - '0' ) ; number = number + x ; number %= mod ; } return number ; }
static int ApowBmodM ( string a , int b , int m ) {
int ans = aModM ( a , m ) ; int mul = ans ;
for ( int i = 1 ; i < b ; i ++ ) ans = ( ans * mul ) % m ; return ans ; }
public static void Main ( ) { string a = "987584345091051645734583954832576" ; int b = 3 , m = 11 ; Console . Write ( ApowBmodM ( a , b , m ) ) ; } }
using System ; class GFG {
class Data { public int x , y ; public Data ( int x , int y ) { this . x = x ; this . y = y ; } } ;
static double interpolate ( Data [ ] f , int xi , int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
double term = f [ i ] . y ; for ( int j = 0 ; j < n ; j ++ ) { if ( j != i ) term = term * ( xi - f [ j ] . x ) / ( f [ i ] . x - f [ j ] . x ) ; }
result += term ; } return result ; }
public static void Main ( String [ ] args ) {
Data [ ] f = { new Data ( 0 , 2 ) , new Data ( 1 , 3 ) , new Data ( 2 , 12 ) , new Data ( 5 , 147 ) } ;
Console . Write ( " Value ▁ of ▁ f ( 3 ) ▁ is ▁ : ▁ " + ( int ) interpolate ( f , 3 , 4 ) ) ; } }
using System ; class GFG {
static int SieveOfSundaram ( int n ) {
int nNew = ( n - 1 ) / 2 ;
bool [ ] marked = new bool [ nNew + 1 ] ;
for ( int i = 0 ; i < nNew + 1 ; i ++ ) marked [ i ] = false ;
for ( int i = 1 ; i <= nNew ; i ++ ) for ( int j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) Console . Write ( 2 + " ▁ " ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) Console . Write ( 2 * i + 1 + " ▁ " ) ; return - 1 ; }
public static void Main ( ) { int n = 20 ; SieveOfSundaram ( n ) ; } }
using System ; using System . Collections ; class GFG {
static void constructArray ( int [ ] A , int N , int K ) {
int [ ] B = new int [ N ] ;
int totalXOR = A [ 0 ] ^ K ;
for ( int i = 0 ; i < N ; i ++ ) B [ i ] = totalXOR ^ A [ i ] ;
for ( int i = 0 ; i < N ; i ++ ) { Console . Write ( B [ i ] + " ▁ " ) ; } }
static void Main ( ) { int [ ] A = { 13 , 14 , 10 , 6 } ; int K = 2 ; int N = A . Length ;
constructArray ( A , N , K ) ; } }
using System ; class GFG {
static int extraElement ( int [ ] A , int [ ] B , int n ) {
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) ans ^= A [ i ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) ans ^= B [ i ] ; return ans ; }
public static void Main ( String [ ] args ) { int [ ] A = { 10 , 15 , 5 } ; int [ ] B = { 10 , 100 , 15 , 5 } ; int n = A . Length ; Console . WriteLine ( extraElement ( A , B , n ) ) ; } }
class GFG {
static int hammingDistance ( int n1 , int n2 ) { int x = n1 ^ n2 ; int setBits = 0 ; while ( x > 0 ) { setBits += x & 1 ; x >>= 1 ; } return setBits ; }
static void Main ( ) { int n1 = 9 , n2 = 14 ; System . Console . WriteLine ( hammingDistance ( n1 , n2 ) ) ; } }
using System ; class GFG {
static void printSubsets ( int n ) { for ( int i = 0 ; i <= n ; i ++ ) if ( ( n & i ) == i ) Console . Write ( i + " ▁ " ) ; }
public static void Main ( ) { int n = 9 ; printSubsets ( n ) ; } }
using System ; public class GFG { static int setBitNumber ( int n ) {
int k = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) ;
return 1 << k ; }
static public void Main ( ) { int n = 273 ; Console . WriteLine ( setBitNumber ( n ) ) ; } }
using System ; public class GfG {
public static int subset ( int [ ] ar , int n ) {
int res = 0 ;
Array . Sort ( ar ) ;
for ( int i = 0 ; i < n ; i ++ ) { int count = 1 ;
for ( ; i < n - 1 ; i ++ ) { if ( ar [ i ] == ar [ i + 1 ] ) count ++ ; else break ; }
res = Math . Max ( res , count ) ; } return res ; }
public static void Main ( ) { int [ ] arr = { 5 , 6 , 9 , 3 , 4 , 3 , 4 } ; int n = 7 ; Console . WriteLine ( subset ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int subset ( int [ ] arr , int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( mp . ContainsKey ( arr [ i ] ) ) { var val = mp [ arr [ i ] ] ; mp . Remove ( arr [ i ] ) ; mp . Add ( arr [ i ] , val + 1 ) ; } else { mp . Add ( arr [ i ] , 1 ) ; } }
int res = 0 ; foreach ( KeyValuePair < int , int > entry in mp ) res = Math . Max ( res , entry . Value ) ; return res ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 9 , 3 , 4 , 3 , 4 } ; int n = arr . Length ; Console . WriteLine ( subset ( arr , n ) ) ; } }
using System . IO ; using System ; using System . Collections ; class GFG {
static ArrayList psquare = new ArrayList ( ) ;
static void calcPsquare ( int N ) { for ( int i = 1 ; i * i <= N ; i ++ ) psquare . Add ( i * i ) ; }
static int countWays ( int index , int target ) {
if ( target == 0 ) return 1 ; if ( index < 0 target < 0 ) return 0 ;
int inc = countWays ( index , target - ( int ) psquare [ index ] ) ;
int exc = countWays ( index - 1 , target ) ;
return inc + exc ; }
static void Main ( ) {
int N = 9 ;
calcPsquare ( N ) ;
Console . WriteLine ( countWays ( psquare . Count - 1 , N ) ) ; } }
using System ; public class GFG { class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
class TreeNode { public int data , size ; public TreeNode left ; public TreeNode right ; } ;
static TreeNode newNode ( int data ) { TreeNode Node = new TreeNode ( ) ; Node . data = data ; Node . left = null ; Node . right = null ;
return ( Node ) ; }
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
public static void Main ( String [ ] args ) {
TreeNode root = newNode ( 1 ) ; root . left = newNode ( 2 ) ; root . right = newNode ( 3 ) ; root . left . left = newNode ( 4 ) ; root . left . right = newNode ( 5 ) ; root . right . left = newNode ( 6 ) ; root . right . right = newNode ( 7 ) ; root . left . left . left = newNode ( 8 ) ; root . left . left . right = newNode ( 9 ) ; int target = 3 ; pair p = sumofsubtree ( root ) ;
int totalnodes = p . first ; distance ( root , target , p . second , totalnodes ) ;
Console . Write ( sum + " STRNEWLINE " ) ; } }
using System ; class GFG {
static void rearrangeArray ( int [ ] A , int [ ] B , int N , int K ) {
Array . Sort ( B ) ; B = reverse ( B ) ; bool flag = true ; for ( int i = 0 ; i < N ; i ++ ) {
if ( A [ i ] + B [ i ] > K ) { flag = false ; break ; } } if ( ! flag ) { Console . Write ( " - 1" + " STRNEWLINE " ) ; } else {
for ( int i = 0 ; i < N ; i ++ ) { Console . Write ( B [ i ] + " ▁ " ) ; } } }
public static void Main ( String [ ] args ) {
int [ ] A = { 1 , 2 , 3 , 4 , 2 } ; int [ ] B = { 1 , 2 , 3 , 1 , 1 } ; int N = A . Length ; int K = 5 ; rearrangeArray ( A , B , N , K ) ; } }
using System ; class GFG {
static void countRows ( int [ , ] mat ) {
int count = 0 ;
int totalSum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { totalSum += mat [ i , j ] ; } }
for ( int i = 0 ; i < n ; i ++ ) {
int currSum = 0 ;
for ( int j = 0 ; j < m ; j ++ ) { currSum += mat [ i , j ] ; }
if ( currSum > totalSum - currSum )
count ++ ; }
Console . WriteLine ( count ) ; }
public static void Main ( String [ ] args ) {
int [ , ] mat = { { 2 , - 1 , 5 } , { - 3 , 0 , - 2 } , { 5 , 1 , 2 } } ;
countRows ( mat ) ; } }
using System ; class GFG {
static bool areElementsContiguous ( int [ ] arr , int n ) {
int max = int . MinValue ; int min = int . MaxValue ; for ( int i = 0 ; i < n ; i ++ ) { max = Math . Max ( max , arr [ i ] ) ; min = Math . Min ( min , arr [ i ] ) ; } int m = max - min + 1 ;
if ( m > n ) return false ;
bool [ ] visited = new bool [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) visited [ arr [ i ] - min ] = true ;
for ( int i = 0 ; i < m ; i ++ ) if ( visited [ i ] == false ) return false ; return true ; }
public static void Main ( ) { int [ ] arr = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . Length ; if ( areElementsContiguous ( arr , n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
public static bool ? areElementsContiguous ( int [ ] arr , int n ) {
HashSet < int > us = new HashSet < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { us . Add ( arr [ i ] ) ; }
int count = 1 ;
int curr_ele = arr [ 0 ] - 1 ;
while ( us . Contains ( curr_ele ) == true ) {
count ++ ;
curr_ele -- ; }
curr_ele = arr [ 0 ] + 1 ;
while ( us . Contains ( curr_ele ) == true ) {
count ++ ;
curr_ele ++ ; }
return ( count == ( us . Count ) ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = new int [ ] { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . Length ; if ( areElementsContiguous ( arr , n ) . Value ) { Console . WriteLine ( " Yes " ) ; } else { Console . WriteLine ( " No " ) ; } } }
using System ; class GFG {
static void longest ( int [ ] a , int n , int k ) { int [ ] freq = new int [ 7 ] ; int start = 0 , end = 0 , now = 0 , l = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
freq [ a [ i ] ] ++ ;
if ( freq [ a [ i ] ] == 1 ) now ++ ;
while ( now > k ) {
freq [ a [ l ] ] -- ;
if ( freq [ a [ l ] ] == 0 ) now -- ;
l ++ ; }
if ( i - l + 1 >= end - start + 1 ) { end = i ; start = l ; } }
for ( int i = start ; i <= end ; i ++ ) Console . Write ( a [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] a = { 6 , 5 , 1 , 2 , 3 , 2 , 1 , 4 , 5 } ; int n = a . Length ; int k = 3 ; longest ( a , n , k ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG {
static bool kOverlap ( List < Tuple < int , int > > pairs , int k ) {
List < Tuple < int , int > > vec = new List < Tuple < int , int > > ( ) ; for ( int i = 0 ; i < pairs . Count ; i ++ ) {
vec . Add ( new Tuple < int , int > ( pairs [ i ] . Item1 , - 1 ) ) ; vec . Add ( new Tuple < int , int > ( pairs [ i ] . Item2 , 1 ) ) ; }
vec . Sort ( ) ;
Stack st = new Stack ( ) ; for ( int i = 0 ; i < vec . Count ; i ++ ) {
Tuple < int , int > cur = vec [ i ] ;
if ( cur . Item2 == - 1 ) {
st . Push ( cur ) ; }
else {
st . Pop ( ) ; }
if ( st . Count >= k ) { return true ; } } return false ; }
public static void Main ( params string [ ] args ) { List < Tuple < int , int > > pairs = new List < Tuple < int , int > > ( ) ; pairs . Add ( new Tuple < int , int > ( 1 , 3 ) ) ; pairs . Add ( new Tuple < int , int > ( 2 , 4 ) ) ; pairs . Add ( new Tuple < int , int > ( 3 , 5 ) ) ; pairs . Add ( new Tuple < int , int > ( 7 , 10 ) ) ; int n = pairs . Count , k = 3 ; if ( kOverlap ( pairs , k ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG { static int N = 5 ;
static int [ ] ptr = new int [ 501 ] ;
static void findSmallestRange ( int [ , ] arr , int n , int k ) { int i , minval , maxval , minrange , minel = 0 , maxel = 0 , flag , minind ;
for ( i = 0 ; i <= k ; i ++ ) { ptr [ i ] = 0 ; } minrange = int . MaxValue ; while ( true ) {
minind = - 1 ; minval = int . MaxValue ; maxval = int . MinValue ; flag = 0 ;
for ( i = 0 ; i < k ; i ++ ) {
if ( ptr [ i ] == n ) { flag = 1 ; break ; }
if ( ptr [ i ] < n && arr [ i , ptr [ i ] ] < minval ) {
minind = i ; minval = arr [ i , ptr [ i ] ] ; }
if ( ptr [ i ] < n && arr [ i , ptr [ i ] ] > maxval ) { maxval = arr [ i , ptr [ i ] ] ; } }
if ( flag == 1 ) { break ; } ptr [ minind ] ++ ;
if ( ( maxval - minval ) < minrange ) { minel = minval ; maxel = maxval ; minrange = maxel - minel ; } } Console . WriteLine ( " The ▁ smallest ▁ range ▁ is " + " [ { 0 } , ▁ { 1 } ] STRNEWLINE " , minel , maxel ) ; }
public static void Main ( String [ ] args ) { int [ , ] arr = { { 4 , 7 , 9 , 12 , 15 } , { 0 , 8 , 10 , 14 , 20 } , { 6 , 12 , 16 , 30 , 50 } } ; int k = arr . GetLength ( 0 ) ; findSmallestRange ( arr , N , k ) ; } }
using System ; class GFG {
static int findLargestd ( int [ ] S , int n ) { bool found = false ;
Array . Sort ( S ) ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { for ( int j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( int k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( int l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return int . MaxValue ; return - 1 ; }
public static void Main ( ) { int [ ] S = new int [ ] { 2 , 3 , 5 , 7 , 12 } ; int n = S . Length ; int ans = findLargestd ( S , n ) ; if ( ans == int . MaxValue ) Console . WriteLine ( " No ▁ Solution " ) ; else Console . Write ( " Largest ▁ d ▁ such ▁ that ▁ a ▁ + ▁ " + " b ▁ + ▁ c ▁ = ▁ d ▁ is ▁ " + ans ) ; } }
using System ; using System . Collections . Generic ;
static int findFourElements ( int [ ] arr , int n ) { Dictionary < int , Indexes > map = new Dictionary < int , Indexes > ( ) ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) { map . Add ( arr [ i ] + arr [ j ] , new Indexes ( i , j ) ) ; } } int d = int . MinValue ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) { int abs_diff = Math . Abs ( arr [ i ] - arr [ j ] ) ;
if ( map . ContainsKey ( abs_diff ) ) { Indexes indexes = map [ abs_diff ] ;
if ( indexes . getI ( ) != i && indexes . getI ( ) != j && indexes . getJ ( ) != i && indexes . getJ ( ) != j ) { d = Math . Max ( d , Math . Max ( arr [ i ] , arr [ j ] ) ) ; } } } } return d ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 5 , 7 , 12 } ; int n = arr . Length ; int res = findFourElements ( arr , n ) ; if ( res == int . MinValue ) Console . WriteLine ( " No ▁ Solution " ) ; else Console . WriteLine ( res ) ; } }
using System ; class GFG {
static int CountMaximum ( int [ ] arr , int n , int k ) {
Array . Sort ( arr ) ; int sum = 0 , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
static public void Main ( ) { int [ ] arr = new int [ ] { 30 , 30 , 10 , 10 } ; int n = 4 ; int k = 50 ;
Console . WriteLine ( CountMaximum ( arr , n , k ) ) ; } }
using System ; class GFG {
static void leftRotatebyOne ( int [ ] arr , int n ) { int i , temp = arr [ 0 ] ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
static void leftRotate ( int [ ] arr , int d , int n ) { for ( int i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
static void printArray ( int [ ] arr , int size ) { for ( int i = 0 ; i < size ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ; } }
using System ; class GFG {
static void partSort ( int [ ] arr , int N , int a , int b ) {
int l = Math . Min ( a , b ) ; int r = Math . Max ( a , b ) ;
int [ ] temp = new int [ r - l + 1 ] ; int j = 0 ; for ( int i = l ; i <= r ; i ++ ) { temp [ j ] = arr [ i ] ; j ++ ; }
Array . Sort ( temp ) ;
j = 0 ; for ( int i = l ; i <= r ; i ++ ) { arr [ i ] = temp [ j ] ; j ++ ; }
for ( int i = 0 ; i < N ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void Main ( ) { int [ ] arr = { 7 , 8 , 4 , 5 , 2 } ; int a = 1 , b = 4 ;
int N = arr . Length ; partSort ( arr , N , a , b ) ; } }
using System ; class GFG { static int MAX_SIZE = 10 ;
static void sortByRow ( int [ , ] mat , int n , bool descending ) { int temp = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( descending == true ) { int t = i ; for ( int p = 0 ; p < n ; p ++ ) { for ( int j = p + 1 ; j < n ; j ++ ) { if ( mat [ t , p ] < mat [ t , j ] ) { temp = mat [ t , p ] ; mat [ t , p ] = mat [ t , j ] ; mat [ t , j ] = temp ; } } } } else sortByRow ( mat , i , n ) ; } }
static void transpose ( int [ , ] mat , int n ) { int temp = 0 ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
temp = mat [ i , j ] ; mat [ i , j ] = mat [ j , i ] ; mat [ j , i ] = temp ; } } }
static void sortMatRowAndColWise ( int [ , ] mat , int n ) {
sortByRow ( mat , n , true ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n , false ) ;
transpose ( mat , n ) ; }
static void printMat ( int [ , ] mat , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) Console . Write ( mat [ i , j ] + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( String [ ] args ) { int n = 3 ; int [ , ] mat = { { 3 , 2 , 1 } , { 9 , 8 , 7 } , { 6 , 5 , 4 } } ; Console . WriteLine ( " Original ▁ Matrix : " ) ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; Console . WriteLine ( " STRNEWLINE Matrix ▁ After ▁ Sorting : " ) ; printMat ( mat , n ) ; } }
using System ; class PushZero {
static void pushZerosToEnd ( int [ ] arr , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
public static void Main ( ) { int [ ] arr = { 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = arr . Length ; pushZerosToEnd ( arr , n ) ; Console . WriteLine ( " Array ▁ after ▁ pushing ▁ all ▁ zeros ▁ to ▁ the ▁ back : ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
using System ; class GFG {
static void moveZerosToEnd ( int [ ] arr , int n ) {
int count = 0 ; int temp ;
for ( int i = 0 ; i < n ; i ++ ) { if ( ( arr [ i ] != 0 ) ) { temp = arr [ count ] ; arr [ count ] = arr [ i ] ; arr [ i ] = temp ; count = count + 1 ; } } }
static void printArray ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( ) { int [ ] arr = { 0 , 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = arr . Length ; Console . Write ( " Original ▁ array : ▁ " ) ; printArray ( arr , n ) ; moveZerosToEnd ( arr , n ) ; Console . Write ( " STRNEWLINE Modified ▁ array : ▁ " ) ; printArray ( arr , n ) ; } }
using System ; class GFG {
static void pushZerosToEnd ( int [ ] arr , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
static void modifyAndRearrangeArr ( int [ ] arr , int n ) {
if ( n == 1 ) return ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( ( arr [ i ] != 0 ) && ( arr [ i ] == arr [ i + 1 ] ) ) {
arr [ i ] = 2 * arr [ i ] ;
arr [ i + 1 ] = 0 ;
i ++ ; } }
pushZerosToEnd ( arr , n ) ; }
static void printArray ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( ) { int [ ] arr = { 0 , 2 , 2 , 2 , 0 , 6 , 6 , 0 , 0 , 8 } ; int n = arr . Length ; Console . Write ( " Original ▁ array : ▁ " ) ; printArray ( arr , n ) ; modifyAndRearrangeArr ( arr , n ) ; Console . Write ( " Modified ▁ array : ▁ " ) ; printArray ( arr , n ) ; } }
using System ; class GFG {
static void printArray ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
static void RearrangePosNeg ( int [ ] arr , int n ) { int key , j ; for ( int i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
public static void Main ( ) { int [ ] arr = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; int n = arr . Length ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ; } }
using System ; class GFG {
static void printArray ( int [ ] A , int size ) { for ( int i = 0 ; i < size ; i ++ ) Console . Write ( A [ i ] + " ▁ " ) ; Console . WriteLine ( " " ) ; ; }
static void reverse ( int [ ] arr , int l , int r ) { if ( l < r ) { arr = swap ( arr , l , r ) ; reverse ( arr , ++ l , -- r ) ; } }
static void merge ( int [ ] arr , int l , int m , int r ) {
int i = l ;
int j = m + 1 ; while ( i <= m && arr [ i ] < 0 ) i ++ ;
while ( j <= r && arr [ j ] < 0 ) j ++ ;
reverse ( arr , i , m ) ;
reverse ( arr , m + 1 , j - 1 ) ;
reverse ( arr , i , j - 1 ) ; }
static void RearrangePosNeg ( int [ ] arr , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
RearrangePosNeg ( arr , l , m ) ; RearrangePosNeg ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } } static int [ ] swap ( int [ ] arr , int i , int j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; return arr ; }
public static void Main ( ) { int [ ] arr = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; int arr_size = arr . Length ; RearrangePosNeg ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ; } }
using System ; public class GFG { public static void RearrangePosNeg ( int [ ] arr ) { int i = 0 ; int j = arr . Length - 1 ; while ( true ) {
while ( arr [ i ] < 0 && i < arr . Length ) i ++ ;
while ( arr [ j ] > 0 && j >= 0 ) j -- ;
if ( i < j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } else break ; } }
static public void Main ( ) { int [ ] arr = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; RearrangePosNeg ( arr ) ; for ( int i = 0 ; i < arr . Length ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
using System ; public class GFG {
static void winner ( int [ ] arr , int N ) {
if ( N % 2 == 1 ) { Console . Write ( " A " ) ; }
else { Console . Write ( " B " ) ; } }
public static void Main ( String [ ] args ) {
int [ ] arr = { 24 , 45 , 45 , 24 } ;
int N = arr . Length ; winner ( arr , N ) ; } }
using System ; class GFG { static int sz = 20 ; static int sqr = ( int ) ( Math . Sqrt ( sz ) ) + 1 ;
static void precomputeExpressionForAllVal ( int [ ] arr , int N , int [ , ] dp ) {
for ( int i = N - 1 ; i >= 0 ; i -- ) {
for ( int j = 1 ; j <= Math . Sqrt ( N ) ; j ++ ) {
if ( i + j < N ) {
dp [ i , j ] = arr [ i ] + dp [ i + j , j ] ; } else {
dp [ i , j ] = arr [ i ] ; } } } }
static void querySum ( int [ ] arr , int N , int [ , ] Q , int M ) {
int [ , ] dp = new int [ sz , sqr ] ; precomputeExpressionForAllVal ( arr , N , dp ) ;
for ( int i = 0 ; i < M ; i ++ ) { int x = Q [ i , 0 ] ; int y = Q [ i , 1 ] ;
if ( y <= Math . Sqrt ( N ) ) { Console . Write ( dp [ x , y ] + " ▁ " ) ; continue ; }
int sum = 0 ;
while ( x < N ) {
sum += arr [ x ] ;
x += y ; } Console . Write ( sum + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 7 , 5 , 4 } ; int [ , ] Q = { { 2 , 1 } , { 3 , 2 } } ; int N = arr . Length ; int M = Q . GetLength ( 0 ) ; querySum ( arr , N , Q , M ) ; } }
using System ; class GFG { static void findElements ( int [ ] arr , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . Length ; findElements ( arr , n ) ; } }
using System ; class GFG { static void findElements ( int [ ] arr , int n ) { Array . Sort ( arr ) ; for ( int i = 0 ; i < n - 2 ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . Length ; findElements ( arr , n ) ; } }
using System ; class GFG { static void findElements ( int [ ] arr , int n ) { int first = int . MinValue ; int second = int . MaxValue ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . Length ; findElements ( arr , n ) ; } }
using System ; class GFG {
public static int getMinOps ( int [ ] arr ) {
int res = 0 ; for ( int i = 0 ; i < arr . Length - 1 ; i ++ ) {
res += Math . Max ( arr [ i + 1 ] - arr [ i ] , 0 ) ; }
return res ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 3 , 4 , 1 , 2 } ; Console . WriteLine ( getMinOps ( arr ) ) ; } }
using System ; class GFG {
static int findFirstMissing ( int [ ] array , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
public static void Main ( ) { int [ ] arr = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = arr . Length ; Console . Write ( " smallest ▁ Missing ▁ element ▁ is ▁ : ▁ " + findFirstMissing ( arr , 0 , n - 1 ) ) ; } }
using System ; class GFG {
int findFirstMissing ( int [ ] arr , int start , int end , int first ) { if ( start < end ) { int mid = ( start + end ) / 2 ;
if ( arr [ mid ] != mid + first ) return findFirstMissing ( arr , start , mid , first ) ; else return findFirstMissing ( arr , mid + 1 , end , first ) ; } return start + first ; }
int findSmallestMissinginSortedArray ( int [ ] arr ) {
if ( arr [ 0 ] != 0 ) return 0 ;
if ( arr [ arr . Length - 1 ] == arr . Length - 1 ) return arr . Length ; int first = arr [ 0 ] ; return findFirstMissing ( arr , 0 , arr . Length - 1 , first ) ; }
static public void Main ( ) { GFG small = new GFG ( ) ; int [ ] arr = { 0 , 1 , 2 , 3 , 4 , 5 , 7 } ; int n = arr . Length ;
Console . WriteLine ( " First ▁ Missing ▁ element ▁ is ▁ : ▁ " + small . findSmallestMissinginSortedArray ( arr ) ) ; } }
using System ; class GFG {
static int FindMaxSum ( int [ ] arr , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 5 , 5 , 10 , 100 , 10 , 5 } ; Console . Write ( FindMaxSum ( arr , arr . Length ) ) ; } }
using System ; class GFG { static readonly int N = 7 ;
static int countChanges ( int [ , ] matrix , int n , int m ) {
int dist = n + m - 1 ;
int [ , ] freq = new int [ dist , 10 ] ;
for ( int i = 0 ; i < dist ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) freq [ i , j ] = 0 ; }
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
freq [ i + j , matrix [ i , j ] ] ++ ; } } int min_changes_sum = 0 ; for ( int i = 0 ; i < dist / 2 ; i ++ ) { int maximum = 0 ; int total_values = 0 ;
for ( int j = 0 ; j < 10 ; j ++ ) { maximum = Math . Max ( maximum , freq [ i , j ] + freq [ n + m - 2 - i , j ] ) ; total_values += ( freq [ i , j ] + freq [ n + m - 2 - i , j ] ) ; }
min_changes_sum += ( total_values - maximum ) ; }
return min_changes_sum ; }
public static void Main ( String [ ] args ) {
int [ , ] mat = { { 1 , 2 } , { 3 , 5 } } ;
Console . Write ( countChanges ( mat , 2 , 2 ) ) ; } }
using System ; public class GFG { static int MAX = 500 ;
static int [ , ] lookup = new int [ MAX , MAX ] ;
static void buildSparseTable ( int [ ] arr , int n ) {
for ( int i = 0 ; i < n ; i ++ ) lookup [ i , 0 ] = arr [ i ] ;
for ( int j = 1 ; ( 1 << j ) <= n ; j ++ ) {
for ( int i = 0 ; ( i + ( 1 << j ) - 1 ) < n ; i ++ ) {
if ( lookup [ i , j - 1 ] < lookup [ i + ( 1 << ( j - 1 ) ) , j - 1 ] ) lookup [ i , j ] = lookup [ i , j - 1 ] ; else lookup [ i , j ] = lookup [ i + ( 1 << ( j - 1 ) ) , j - 1 ] ; } } }
static int query ( int L , int R ) {
int j = ( int ) Math . Log ( R - L + 1 ) ;
if ( lookup [ L , j ] <= lookup [ R - ( 1 << j ) + 1 , j ] ) return lookup [ L , j ] ; else return lookup [ R - ( 1 << j ) + 1 , j ] ; }
static public void Main ( ) { int [ ] a = { 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 } ; int n = a . Length ; buildSparseTable ( a , n ) ; Console . WriteLine ( query ( 0 , 4 ) ) ; Console . WriteLine ( query ( 4 , 7 ) ) ; Console . WriteLine ( query ( 7 , 8 ) ) ; } }
using System ; class GFG { static readonly int MAX = 500 ;
static int [ , ] table = new int [ MAX , MAX ] ;
static void buildSparseTable ( int [ ] arr , int n ) {
for ( int i = 0 ; i < n ; i ++ ) table [ i , 0 ] = arr [ i ] ;
for ( int j = 1 ; j <= n ; j ++ ) for ( int i = 0 ; i <= n - ( 1 << j ) ; i ++ ) table [ i , j ] = __gcd ( table [ i , j - 1 ] , table [ i + ( 1 << ( j - 1 ) ) , j - 1 ] ) ; }
static int query ( int L , int R ) {
int j = ( int ) Math . Log ( R - L + 1 ) ;
return __gcd ( table [ L , j ] , table [ R - ( 1 << j ) + 1 , j ] ) ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
public static void Main ( String [ ] args ) { int [ ] a = { 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 } ; int n = a . Length ; buildSparseTable ( a , n ) ; Console . Write ( query ( 0 , 2 ) + " STRNEWLINE " ) ; Console . Write ( query ( 1 , 3 ) + " STRNEWLINE " ) ; Console . Write ( query ( 4 , 5 ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static void minimizeWithKSwaps ( int [ ] arr , int n , int k ) { for ( int i = 0 ; i < n - 1 && k > 0 ; ++ i ) {
int pos = i ; for ( int j = i + 1 ; j < n ; ++ j ) {
if ( j - i > k ) break ;
if ( arr [ j ] < arr [ pos ] ) pos = j ; }
int temp ; for ( int j = pos ; j > i ; -- j ) { temp = arr [ j ] ; arr [ j ] = arr [ j - 1 ] ; arr [ j - 1 ] = temp ; }
k -= pos - i ; } }
public static void Main ( ) { int [ ] arr = { 7 , 6 , 9 , 2 , 1 } ; int n = arr . Length ; int k = 3 ;
minimizeWithKSwaps ( arr , n , k ) ;
for ( int i = 0 ; i < n ; ++ i ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
using System ; class GFG {
static int findMaxAverage ( int [ ] arr , int n , int k ) {
if ( k > n ) return - 1 ;
int [ ] csum = new int [ n ] ; csum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) csum [ i ] = csum [ i - 1 ] + arr [ i ] ;
int max_sum = csum [ k - 1 ] , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { int curr_sum = csum [ i ] - csum [ i - k ] ; if ( curr_sum > max_sum ) { max_sum = curr_sum ; max_end = i ; } }
return max_end - k + 1 ; }
static public void Main ( ) { int [ ] arr = { 1 , 12 , - 5 , - 6 , 50 , 3 } ; int k = 4 ; int n = arr . Length ; Console . WriteLine ( " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " + " length ▁ " + k + " ▁ begins ▁ at ▁ index ▁ " + findMaxAverage ( arr , n , k ) ) ; } }
using System ; class GFG {
static int findMaxAverage ( int [ ] arr , int n , int k ) {
if ( k > n ) return - 1 ;
int sum = arr [ 0 ] ; for ( int i = 1 ; i < k ; i ++ ) sum += arr [ i ] ; int max_sum = sum ; int max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { sum = sum + arr [ i ] - arr [ i - k ] ; if ( sum > max_sum ) { max_sum = sum ; max_end = i ; } }
return max_end - k + 1 ; }
public static void Main ( ) { int [ ] arr = { 1 , 12 , - 5 , - 6 , 50 , 3 } ; int k = 4 ; int n = arr . Length ; Console . WriteLine ( " The ▁ maximum ▁ " + " average ▁ subarray ▁ of ▁ length ▁ " + k + " ▁ begins ▁ at ▁ index ▁ " + findMaxAverage ( arr , n , k ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static Dictionary < Tuple < int , int > , int > m = new Dictionary < Tuple < int , int > , int > ( ) ;
static int findMinimum ( int [ ] arr , int N , int pos , int turn ) {
Tuple < int , int > x = new Tuple < int , int > ( pos , turn ) ; if ( m . ContainsKey ( x ) ) { return m [ x ] ; }
if ( pos >= N - 1 ) { return 0 ; }
if ( turn == 0 ) {
int ans = Math . Min ( findMinimum ( arr , N , pos + 1 , 1 ) + arr [ pos ] , findMinimum ( arr , N , pos + 2 , 1 ) + arr [ pos ] + arr [ pos + 1 ] ) ;
Tuple < int , int > v = new Tuple < int , int > ( pos , turn ) ; m [ v ] = ans ;
return ans ; }
if ( turn != 0 ) {
int ans = Math . Min ( findMinimum ( arr , N , pos + 1 , 0 ) , findMinimum ( arr , N , pos + 2 , 0 ) ) ;
Tuple < int , int > v = new Tuple < int , int > ( pos , turn ) ; m [ v ] = ans ;
return ans ; } return 0 ; }
static int countPenality ( int [ ] arr , int N ) {
int pos = 0 ;
int turn = 0 ;
return findMinimum ( arr , N , pos , turn ) + 1 ; }
static void printAnswer ( int [ ] arr , int N ) {
int a = countPenality ( arr , N ) ;
int sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
Console . WriteLine ( a ) ; }
static void Main ( ) { int [ ] arr = { 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 } ; int N = 8 ; printAnswer ( arr , N ) ; } }
using System ; class GFG { static int MAX = 1000001 ; static int [ ] prime = new int [ MAX ] ;
static void SieveOfEratosthenes ( ) {
Array . Fill ( prime , 1 ) ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
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
static void updateValue ( int [ ] arr , int [ ] st , int n , int i , int new_val ) {
if ( i < 0 i > n - 1 ) { Console . Write ( " - 1" ) ; return ; }
int diff = new_val - arr [ i ] ; int prev_val = arr [ i ] ;
arr [ i ] = new_val ;
if ( ( prime [ new_val ] prime [ prev_val ] ) ! = 0 ) {
if ( prime [ prev_val ] == 0 ) updateValueUtil ( st , 0 , n - 1 , i , new_val , 0 ) ;
else if ( prime [ new_val ] == 0 ) ( st , 0 , n - 1 , i , - prev_val , 0 ) ;
else updateValueUtil ( st , 0 , n - 1 , i , diff , 0 ) ; } }
static int getSum ( int [ ] st , int n , int qs , int qe ) {
if ( qs < 0 qe > n - 1 qs > qe ) { Console . WriteLine ( " - 1" ) ; return - 1 ; } return getSumUtil ( st , 0 , n - 1 , qs , qe , 0 ) ; }
static int constructSTUtil ( int [ ] arr , int ss , int se , int [ ] st , int si ) {
if ( ss == se ) {
if ( prime [ arr [ ss ] ] != 0 ) st [ si ] = arr [ ss ] ; else st [ si ] = 0 ; return st [ si ] ; }
int mid = getMid ( ss , se ) ; st [ si ] = constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) + constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ; return st [ si ] ; }
static int [ ] constructST ( int [ ] arr , int n ) {
int x = ( int ) ( Math . Ceiling ( Math . Log ( n , 2 ) ) ) ;
int max_size = 2 * ( int ) Math . Pow ( 2 , x ) - 1 ;
int [ ] st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
static void Main ( ) { int [ ] arr = { 1 , 3 , 5 , 7 , 9 , 11 } ; int n = arr . Length ;
SieveOfEratosthenes ( ) ;
int [ ] st = constructST ( arr , n ) ;
Console . WriteLine ( getSum ( st , n , 1 , 3 ) ) ;
updateValue ( arr , st , n , 1 , 10 ) ;
Console . WriteLine ( getSum ( st , n , 1 , 3 ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int mod = 1000000007 ; static int [ , ] dp = new int [ 1000 , 1000 ] ; static int calculate ( int pos , int prev , String s , List < int > index ) {
if ( pos == s . Length ) return 1 ;
if ( dp [ pos , prev ] != - 1 ) return dp [ pos , prev ] ;
int answer = 0 ; for ( int i = 0 ; i < index . Count ; i ++ ) { if ( index [ i ] . CompareTo ( prev ) >= 0 ) { answer = ( answer % mod + calculate ( pos + 1 , index [ i ] , s , index ) % mod ) % mod ; } }
return dp [ pos , prev ] = answer ; } static int countWays ( List < String > a , String s ) { int n = a . Count ;
List < int > [ ] index = new List < int > [ 26 ] ; for ( int i = 0 ; i < 26 ; i ++ ) index [ i ] = new List < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < a [ i ] . Length ; j ++ ) {
index [ a [ i ] [ j ] - ' a ' ] . Add ( j + 1 ) ; } }
for ( int i = 0 ; i < 1000 ; i ++ ) { for ( int j = 0 ; j < 1000 ; j ++ ) { dp [ i , j ] = - 1 ; } } return calculate ( 0 , 0 , s , index [ 0 ] ) ; }
public static void Main ( String [ ] args ) { List < String > A = new List < String > ( ) ; A . Add ( " adc " ) ; A . Add ( " aec " ) ; A . Add ( " erg " ) ; String S = " ac " ; Console . Write ( countWays ( A , S ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int MAX = 10005 ; static readonly int MOD = 1000000007 ;
static int [ , , ] dp = new int [ MAX , 101 , 2 ] ;
static int countNum ( int idx , int sum , int tight , List < int > num , int len , int k ) { if ( len == idx ) { if ( sum == 0 ) return 1 ; else return 0 ; } if ( dp [ idx , sum , tight ] != - 1 ) return dp [ idx , sum , tight ] ; int res = 0 , limit ;
if ( tight == 0 ) { limit = num [ idx ] ; }
else { limit = 9 ; } for ( int i = 0 ; i <= limit ; i ++ ) {
int new_tight = tight ; if ( tight == 0 && i < limit ) new_tight = 1 ; res += countNum ( idx + 1 , ( sum + i ) % k , new_tight , num , len , k ) ; res %= MOD ; }
if ( res < 0 ) res += MOD ; return dp [ idx , sum , tight ] = res ; }
static List < int > process ( String s ) { List < int > num = new List < int > ( ) ; for ( int i = 0 ; i < s . Length ; i ++ ) { num . Add ( s [ i ] - '0' ) ; } return num ; }
public static void Main ( String [ ] args ) {
String n = "98765432109876543210" ;
int len = n . Length ; int k = 58 ;
for ( int i = 0 ; i < MAX ; i ++ ) { for ( int j = 0 ; j < 101 ; j ++ ) { for ( int l = 0 ; l < 2 ; l ++ ) dp [ i , j , l ] = - 1 ; } }
List < int > num = process ( n ) ; Console . Write ( countNum ( 0 , 0 , 0 , num , len , k ) ) ; } }
using System ; class GFG { static int maxN = 31 ; static int maxW = 31 ;
static int [ , , ] dp = new int [ maxN , maxW , maxW ] ;
static int maxWeight ( int [ ] arr , int n , int w1_r , int w2_r , int i ) {
if ( i == n ) return 0 ; if ( dp [ i , w1_r , w2_r ] != - 1 ) return dp [ i , w1_r , w2_r ] ;
int fill_w1 = 0 , fill_w2 = 0 , fill_none = 0 ; if ( w1_r >= arr [ i ] ) fill_w1 = arr [ i ] + maxWeight ( arr , n , w1_r - arr [ i ] , w2_r , i + 1 ) ; if ( w2_r >= arr [ i ] ) fill_w2 = arr [ i ] + maxWeight ( arr , n , w1_r , w2_r - arr [ i ] , i + 1 ) ; fill_none = maxWeight ( arr , n , w1_r , w2_r , i + 1 ) ;
dp [ i , w1_r , w2_r ] = Math . Max ( fill_none , Math . Max ( fill_w1 , fill_w2 ) ) ; return dp [ i , w1_r , w2_r ] ; }
public static void Main ( ) {
int [ ] arr = { 8 , 2 , 3 } ;
for ( int i = 0 ; i < maxN ; i ++ ) for ( int j = 0 ; j < maxW ; j ++ ) for ( int k = 0 ; k < maxW ; k ++ ) dp [ i , j , k ] = - 1 ;
int n = arr . Length ;
int w1 = 10 , w2 = 3 ;
Console . WriteLine ( maxWeight ( arr , n , w1 , w2 , 0 ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int n = 3 ;
static void findPrefixCount ( int [ , ] p_arr , bool [ , ] set_bit ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = n - 1 ; j >= 0 ; j -- ) { if ( ! set_bit [ i , j ] ) continue ; if ( j != n - 1 ) p_arr [ i , j ] += p_arr [ i , j + 1 ] ; p_arr [ i , j ] += ( set_bit [ i , j ] ) ? 1 : 0 ; } } } public class pair { public int first , second ; public pair ( ) { } public pair ( int a , int b ) { first = a ; second = b ; } }
static int matrixAllOne ( bool [ , ] set_bit ) {
int [ , ] p_arr = new int [ n , n ] ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) p_arr [ i , j ] = 0 ; findPrefixCount ( p_arr , set_bit ) ;
int ans = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { int i = n - 1 ;
Stack < pair > q = new Stack < pair > ( ) ;
int to_sum = 0 ; while ( i >= 0 ) { int c = 0 ; while ( q . Count != 0 && q . Peek ( ) . first > p_arr [ i , j ] ) { to_sum -= ( q . Peek ( ) . second + 1 ) * ( q . Peek ( ) . first - p_arr [ i , j ] ) ; c += q . Peek ( ) . second + 1 ; q . Pop ( ) ; } to_sum += p_arr [ i , j ] ; ans += to_sum ; q . Push ( new pair ( p_arr [ i , j ] , c ) ) ; i -- ; } } return ans ; }
static int sumAndMatrix ( int [ , ] arr ) { int sum = 0 ; int mul = 1 ; for ( int i = 0 ; i < 30 ; i ++ ) {
bool [ , ] set_bit = new bool [ n , n ] ; for ( int R = 0 ; R < n ; R ++ ) for ( int C = 0 ; C < n ; C ++ ) set_bit [ R , C ] = ( ( arr [ R , C ] & ( 1 << i ) ) != 0 ) ; sum += ( mul * matrixAllOne ( set_bit ) ) ; mul *= 2 ; } return sum ; }
public static void Main ( String [ ] args ) { int [ , ] arr = { { 9 , 7 , 4 } , { 8 , 9 , 2 } , { 11 , 11 , 5 } } ; Console . WriteLine ( sumAndMatrix ( arr ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG { static int CountWays ( int n ) {
int [ ] noOfWays = new int [ n + 3 ] ; noOfWays [ 0 ] = 1 ; noOfWays [ 1 ] = 1 ; noOfWays [ 2 ] = 1 + 1 ;
for ( int i = 3 ; i < n + 1 ; i ++ ) { noOfWays [ i ] =
noOfWays [ 3 - 1 ]
+ noOfWays [ 3 - 3 ] ;
noOfWays [ 0 ] = noOfWays [ 1 ] ; noOfWays [ 1 ] = noOfWays [ 2 ] ; noOfWays [ 2 ] = noOfWays [ i ] ; } return noOfWays [ n ] ; }
public static void Main ( String [ ] args ) { int n = 5 ; Console . WriteLine ( CountWays ( n ) ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG { static int MAX = 105 ; static void sieve ( int [ ] prime ) { for ( int i = 2 ; i * i < MAX ; i ++ ) { if ( prime [ i ] == 0 ) { for ( int j = i * i ; j < MAX ; j += i ) prime [ j ] = 1 ; } } } class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static void dfs ( int i , int j , int k , ref int q , int n , int m , int [ , ] mappedMatrix , int [ , ] mark , pair [ ] ans ) {
if ( ( mappedMatrix [ i , j ] == 0 ? true : false ) || ( i > n ? true : false ) || ( j > m ? true : false ) || ( mark [ i , j ] != 0 ? true : false ) || ( q != 0 ? true : false ) ) return ;
mark [ i , j ] = 1 ;
ans [ k ] = new pair ( i , j ) ;
if ( i == n && j == m ) {
( q ) = k ; return ; }
dfs ( i + 1 , j + 1 , k + 1 , ref q , n , m , mappedMatrix , mark , ans ) ;
dfs ( i + 1 , j , k + 1 , ref q , n , m , mappedMatrix , mark , ans ) ;
dfs ( i , j + 1 , k + 1 , ref q , n , m , mappedMatrix , mark , ans ) ; }
static void lexicographicalPath ( int n , int m , int [ , ] mappedMatrix ) {
int q = 0 ;
pair [ ] ans = new pair [ MAX ] ;
int [ , ] mark = new int [ MAX , MAX ] ;
dfs ( 1 , 1 , 1 , ref q , n , m , mappedMatrix , mark , ans ) ;
for ( int i = 1 ; i <= q ; i ++ ) Console . WriteLine ( ans [ i ] . first + " ▁ " + ans [ i ] . second ) ; }
static void countPrimePath ( int [ , ] mappedMatrix , int n , int m ) { int [ , ] dp = new int [ MAX , MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) { for ( int j = 0 ; j < MAX ; j ++ ) { dp [ i , j ] = 0 ; } } dp [ 1 , 1 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= m ; j ++ ) {
if ( i == 1 && j == 1 ) continue ; dp [ i , j ] = ( dp [ i - 1 , j ] + dp [ i , j - 1 ] + dp [ i - 1 , j - 1 ] ) ;
if ( mappedMatrix [ i , j ] == 0 ) dp [ i , j ] = 0 ; } } Console . WriteLine ( dp [ n , m ] ) ; }
static void preprocessMatrix ( int [ , ] mappedMatrix , int [ , ] a , int n , int m ) { int [ ] prime = new int [ MAX ] ;
sieve ( prime ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
if ( prime [ a [ i , j ] ] == 0 ) mappedMatrix [ i + 1 , j + 1 ] = 1 ;
else mappedMatrix [ i + 1 , j + 1 ] = 0 ; } } }
public static void Main ( string [ ] args ) { int n = 3 ; int m = 3 ; int [ , ] a = new int [ 3 , 3 ] { { 2 , 3 , 7 } , { 5 , 4 , 2 } , { 3 , 7 , 11 } } ; int [ , ] mappedMatrix = new int [ MAX , MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) { for ( int j = 0 ; j < MAX ; j ++ ) { mappedMatrix [ i , j ] = 0 ; } } preprocessMatrix ( mappedMatrix , a , n , m ) ; countPrimePath ( mappedMatrix , n , m ) ; lexicographicalPath ( n , m , mappedMatrix ) ; } }
using System ; class sumofSub {
static int isSubsetSum ( int [ ] set , int n , int sum ) {
bool [ , ] subset = new bool [ sum + 1 , n + 1 ] ; int [ , ] count = new int [ sum + 1 , n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { subset [ 0 , i ] = true ; count [ 0 , i ] = 0 ; }
for ( int i = 1 ; i <= sum ; i ++ ) { subset [ i , 0 ] = false ; count [ i , 0 ] = - 1 ; }
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i , j ] = subset [ i , j - 1 ] ; count [ i , j ] = count [ i , j - 1 ] ; if ( i >= set [ j - 1 ] ) { subset [ i , j ] = subset [ i , j ] || subset [ i - set [ j - 1 ] , j - 1 ] ; if ( subset [ i , j ] ) count [ i , j ] = Math . Max ( count [ i , j - 1 ] , count [ i - set [ j - 1 ] , j - 1 ] + 1 ) ; } } } return count [ sum , n ] ; }
public static void Main ( ) { int [ ] set = { 2 , 3 , 5 , 10 } ; int sum = 20 ; int n = set . Length ; Console . WriteLine ( isSubsetSum ( set , n , sum ) ) ; } }
using System ; class GFG { static int MAX = 100 ;
static int lcslen = 0 ;
static int [ , ] dp = new int [ MAX , MAX ] ;
static int lcs ( string str1 , string str2 , int len1 , int len2 , int i , int j ) { int ret = dp [ i , j ] ;
if ( i == len1 j == len2 ) return ret = 0 ;
if ( ret != - 1 ) return ret ; ret = 0 ;
if ( str1 [ i ] == str2 [ j ] ) ret = 1 + lcs ( str1 , str2 , len1 , len2 , i + 1 , j + 1 ) ; else ret = Math . Max ( lcs ( str1 , str2 , len1 , len2 , i + 1 , j ) , lcs ( str1 , str2 , len1 , len2 , i , j + 1 ) ) ; return ret ; }
static void printAll ( string str1 , string str2 , int len1 , int len2 , char [ ] data , int indx1 , int indx2 , int currlcs ) {
if ( currlcs == lcslen ) { data [ currlcs ] = ' \0' ; Console . WriteLine ( new string ( data ) ) ; return ; }
if ( indx1 == len1 indx2 == len2 ) return ;
for ( char ch = ' a ' ; ch <= ' z ' ; ch ++ ) {
bool done = false ; for ( int i = indx1 ; i < len1 ; i ++ ) {
if ( ch == str1 [ i ] ) { for ( int j = indx2 ; j < len2 ; j ++ ) {
if ( ch == str2 [ j ] && lcs ( str1 , str2 , len1 , len2 , i , j ) == lcslen - currlcs ) { data [ currlcs ] = ch ; printAll ( str1 , str2 , len1 , len2 , data , i + 1 , j + 1 , currlcs + 1 ) ; done = true ; break ; } } }
if ( done ) break ; } } }
static void prinlAllLCSSorted ( string str1 , string str2 ) {
int len1 = str1 . Length , len2 = str2 . Length ;
for ( int i = 0 ; i < MAX ; i ++ ) { for ( int j = 0 ; j < MAX ; j ++ ) { dp [ i , j ] = - 1 ; } } lcslen = lcs ( str1 , str2 , len1 , len2 , 0 , 0 ) ;
char [ ] data = new char [ MAX ] ; printAll ( str1 , str2 , len1 , len2 , data , 0 , 0 , 0 ) ; }
static void Main ( ) { string str1 = " abcabcaa " , str2 = " acbacba " ; prinlAllLCSSorted ( str1 , str2 ) ; } }
using System ; class GFG { static bool isMajority ( int [ ] arr , int n , int x ) { int i , last_index = 0 ;
last_index = ( n % 2 == 0 ) ? n / 2 : n / 2 + 1 ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + n / 2 ] == x ) return true ; } return false ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 4 , 4 , 4 } ; int n = arr . Length ; int x = 4 ; if ( isMajority ( arr , n , x ) == true ) Console . Write ( x + " ▁ appears ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; else Console . Write ( x + " ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; } }
using System ; class GFG {
static int _binarySearch ( int [ ] arr , int low , int high , int x ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( ( mid == 0 x > arr [ mid - 1 ] ) && ( arr [ mid ] == x ) ) return mid ; else if ( x > arr [ mid ] ) return _binarySearch ( arr , ( mid + 1 ) , high , x ) ; else return _binarySearch ( arr , low , ( mid - 1 ) , x ) ; } return - 1 ; }
static bool isMajority ( int [ ] arr , int n , int x ) {
int i = _binarySearch ( arr , 0 , n - 1 , x ) ;
if ( i == - 1 ) return false ;
if ( ( ( i + n / 2 ) <= ( n - 1 ) ) && arr [ i + n / 2 ] == x ) return true ; else return false ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = arr . Length ; int x = 3 ; if ( isMajority ( arr , n , x ) == true ) Console . Write ( x + " ▁ appears ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; else Console . Write ( x + " ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; } }
using System ; class GFG { static bool isMajorityElement ( int [ ] arr , int n , int key ) { if ( arr [ n / 2 ] == key ) return true ; else return false ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = arr . Length ; int x = 3 ; if ( isMajorityElement ( arr , n , x ) ) Console . Write ( x + " ▁ appears ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ [ ] arr " ) ; else Console . Write ( x + " ▁ does ▁ not ▁ appear ▁ more ▁ " + " than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; } }
using System ; class GFG {
static int cutRod ( int [ ] price , int n ) { int [ ] val = new int [ n + 1 ] ; val [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { int max_val = int . MinValue ; for ( int j = 0 ; j < i ; j ++ ) max_val = Math . Max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . Length ; Console . WriteLine ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
using System ; class GFG {
public static bool isPossible ( int [ ] target ) {
int max = 0 ;
int index = 0 ;
for ( int i = 0 ; i < target . Length ; i ++ ) {
if ( max < target [ i ] ) { max = target [ i ] ; index = i ; } }
if ( max == 1 ) return true ;
for ( int i = 0 ; i < target . Length ; i ++ ) {
if ( i != index ) {
max -= target [ i ] ;
if ( max <= 0 ) return false ; } }
target [ index ] = max ;
return isPossible ( target ) ; }
static public void Main ( ) { int [ ] target = { 9 , 3 , 5 } ; bool res = isPossible ( target ) ; if ( res ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
using System ; class GFG {
static int nCr ( int n , int r ) {
int res = 1 ;
if ( r > n - r ) r = n - r ;
for ( int i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
public static void Main ( ) { int n = 3 , m = 2 , k = 2 ; Console . Write ( nCr ( n + m , k ) ) ; } }
using System ; class GFG {
static void Is_possible ( long N ) { long C = 0 ; long D = 0 ;
while ( N % 10 == 0 ) { N = N / 10 ; C += 1 ; }
if ( Math . Pow ( 2 , ( long ) ( Math . Log ( N ) / ( Math . Log ( 2 ) ) ) ) == N ) { D = ( long ) ( Math . Log ( N ) / ( Math . Log ( 2 ) ) ) ;
if ( C >= D ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } else Console . ( " NO " ) ; }
public static void Main ( ) { long N = 2000000000000L ; Is_possible ( N ) ; } }
using System ; class GFG {
static void findNthTerm ( int n ) { Console . Write ( n * n - n + 1 ) ; }
public static void Main ( ) { int N = 4 ; findNthTerm ( N ) ; } }
using System ; class GFG {
static int rev ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; }
return rev_num ; }
static int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= Math . Sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += rev ( i ) ; else result += ( rev ( i ) + rev ( num / i ) ) ; } }
return ( result + 1 ) ; }
static Boolean isAntiPerfect ( int n ) { return divSum ( n ) == n ; }
public static void Main ( String [ ] args ) {
int N = 244 ;
if ( isAntiPerfect ( N ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static void printSeries ( int n , int a , int b , int c ) { int d ;
if ( n == 1 ) { Console . Write ( a + " ▁ " ) ; return ; } if ( n == 2 ) { Console . Write ( a + " ▁ " + b + " ▁ " ) ; return ; } Console . Write ( a + " ▁ " + b + " ▁ " + c + " ▁ " ) ; for ( int i = 4 ; i <= n ; i ++ ) { d = a + b + c ; Console . Write ( d + " ▁ " ) ; a = b ; b = c ; c = d ; } }
public static void Main ( ) { int N = 7 , a = 1 , b = 3 ; int c = 4 ;
printSeries ( N , a , b , c ) ; } }
using System ; class GFG {
static int diameter ( int n ) {
int L , H ; L = 1 ;
H = 0 ;
if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 2 ; } if ( n == 3 ) { return 3 ; }
while ( L * 2 <= n ) { L *= 2 ; H ++ ; }
if ( n >= L * 2 - 1 ) return 2 * H + 1 ; else if ( n >= L + ( L / 2 ) - 1 ) return 2 * H ; return 2 * H - 1 ; }
public static void Main ( String [ ] args ) { int n = 15 ; Console . WriteLine ( diameter ( n ) ) ; } }
using System ; class GFG {
static void compareValues ( int a , int b , int c , int d ) {
double log1 = Math . Log10 ( a ) ; double num1 = log1 * b ;
double log2 = Math . Log10 ( c ) ; double num2 = log2 * d ;
if ( num1 > num2 ) Console . WriteLine ( a + " ^ " + b ) ; else Console . WriteLine ( c + " ^ " + d ) ; }
public static void Main ( ) { int a = 8 , b = 29 , c = 60 , d = 59 ; compareValues ( a , b , c , d ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 100005 ;
static List < int > addPrimes ( ) { int n = MAX ; Boolean [ ] prime = new Boolean [ n + 1 ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) prime [ i ] = true ; for ( int p = 2 ; p * p <= n ; p ++ ) { if ( prime [ p ] == true ) { for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } List < int > ans = new List < int > ( ) ;
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) ans . Add ( p ) ; return ans ; }
static Boolean is_prime ( int n ) { return ( n == 3 n == 5 n == 7 ) ; }
static int find_Sum ( int n ) {
int sum = 0 ;
List < int > v = addPrimes ( ) ;
for ( int i = 0 ; i < v . Count && n > 0 ; i ++ ) {
int flag = 1 ; int a = v [ i ] ;
while ( a != 0 ) { int d = a % 10 ; a = a / 10 ; if ( is_prime ( d ) ) { flag = 0 ; break ; } }
if ( flag == 1 ) { n -- ; sum = sum + v [ i ] ; } }
return sum ; }
public static void Main ( String [ ] args ) { int n = 7 ;
Console . WriteLine ( find_Sum ( n ) ) ; } }
using System ; class GFG {
static int primeCount ( int [ ] arr , int n ) {
int max_val = max_element ( arr ) ;
bool [ ] prime = new bool [ max_val + 1 ] ; for ( int p = 0 ; p <= max_val ; p ++ ) prime [ p ] = true ;
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= max_val ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i <= max_val ; i += p ) prime [ i ] = false ; } }
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( prime [ arr [ i ] ] ) count ++ ; return count ; }
static int [ ] getPrefixArray ( int [ ] arr , int n , int [ ] pre ) {
pre [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { pre [ i ] = pre [ i - 1 ] + arr [ i ] ; } return pre ; }
public static void Main ( ) { int [ ] arr = { 1 , 4 , 8 , 4 } ; int n = arr . Length ;
int [ ] pre = new int [ n ] ; pre = getPrefixArray ( arr , n , pre ) ;
Console . Write ( primeCount ( pre , n ) ) ; } }
using System ; class GFG {
static int minValue ( int n , int x , int y ) {
float val = ( y * n ) / 100 ;
if ( x >= val ) return 0 ; else return ( int ) ( Math . Ceiling ( val ) - x ) ; }
public static void Main ( ) { int n = 10 , x = 2 , y = 40 ; Console . WriteLine ( ( int ) minValue ( n , x , y ) ) ; } }
using System ; class GFG {
static bool isPrime ( long n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static bool isFactorialPrime ( long n ) {
if ( ! isPrime ( n ) ) return false ; long fact = 1 ; int i = 1 ; while ( fact <= n + 1 ) {
fact = fact * i ;
if ( n + 1 == fact n - 1 == fact ) return true ; i ++ ; }
return false ; }
public static void Main ( ) { int n = 23 ; if ( isFactorialPrime ( n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
long n = 5 ;
long fac1 = 1 ; for ( int i = 2 ; i <= n - 1 ; i ++ ) fac1 = fac1 * i ;
long fac2 = fac1 * n ;
long totalWays = fac1 * fac2 ;
Console . WriteLine ( totalWays ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int MAX = 10000 ; static List < int > arr = new List < int > ( ) ;
static void SieveOfEratosthenes ( ) {
bool [ ] prime = new bool [ MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) prime [ i ] = true ; for ( int p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
for ( int p = 2 ; p < MAX ; p ++ ) if ( prime [ p ] ) arr . Add ( p ) ; }
static bool isEuclid ( long n ) { long product = 1 ; int i = 0 ; while ( product < n ) {
product = product * arr [ i ] ; if ( product + 1 == n ) return true ; i ++ ; } return false ; }
public static void Main ( String [ ] args ) {
SieveOfEratosthenes ( ) ;
long n = 31 ;
if ( isEuclid ( n ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ;
n = 42 ;
if ( isEuclid ( n ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; class GFG {
static int nextPerfectCube ( int N ) { int nextN = ( int ) Math . Floor ( Math . Pow ( N , ( double ) 1 / 3 ) ) + 1 ; return nextN * nextN * nextN ; }
public static void Main ( ) { int n = 35 ; Console . Write ( nextPerfectCube ( n ) ) ; } }
using System ; class GFG {
static bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static int SumOfPrimeDivisors ( int n ) { int sum = 0 ;
int root_n = ( int ) Math . Sqrt ( n ) ; for ( int i = 1 ; i <= root_n ; i ++ ) { if ( n % i == 0 ) {
if ( i == n / i && isPrime ( i ) ) { sum += i ; } else {
if ( isPrime ( i ) ) { sum += i ; } if ( isPrime ( n / i ) ) { sum += ( n / i ) ; } } } } return sum ; }
static void Main ( ) { int n = 60 ; Console . WriteLine ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " + SumOfPrimeDivisors ( n ) ) ; } }
using System ; class GFG { static int findpos ( String n ) { int pos = 0 ; for ( int i = 0 ; i < n . Length ; i ++ ) { switch ( n [ i ] ) {
'2' : pos = pos * 4 + 1 ; break ;
'3' : pos = pos * 4 + 2 ; break ;
'5' : pos = pos * 4 + 3 ; break ;
'7' : pos = pos * 4 + 4 ; break ; } } return pos ; }
public static void Main ( String [ ] args ) { String n = "777" ; Console . WriteLine ( findpos ( n ) ) ; } }
using System ; public class GFG {
static void possibleTripletInRange ( int L , int R ) { bool flag = false ; int possibleA = 0 , possibleB = 0 , possibleC = 0 ; int numbersInRange = ( R - L + 1 ) ;
if ( numbersInRange < 3 ) { flag = false ; }
else if ( numbersInRange > 3 ) { flag = true ;
if ( L % 2 > 0 ) { L ++ ; } possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
if ( ! ( L % 2 > 0 ) ) { flag = true ; possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
flag = false ; } }
if ( flag == true ) { Console . WriteLine ( " ( " + possibleA + " , ▁ " + possibleB + " , ▁ " + possibleC + " ) " + " ▁ is ▁ one ▁ such ▁ possible " + " ▁ triplet ▁ between ▁ " + L + " ▁ and ▁ " + R ) ; } else { Console . WriteLine ( " No ▁ Such ▁ Triplet " + " ▁ exists ▁ between ▁ " + L + " ▁ and ▁ " + R ) ; } }
static public void Main ( ) { int L , R ;
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ; } }
using System ; class GFG { static int mod = 1000000007 ;
static int digitNumber ( long n ) {
if ( n == 0 ) return 1 ;
if ( n == 1 ) return 9 ;
if ( n % 2 != 0 ) {
int temp = digitNumber ( ( n - 1 ) / 2 ) % mod ; return ( 9 * ( temp * temp ) % mod ) % mod ; } else {
int temp = digitNumber ( n / 2 ) % mod ; return ( temp * temp ) % mod ; } } static int countExcluding ( int n , int d ) {
if ( d == 0 ) return ( 9 * digitNumber ( n - 1 ) ) % mod ; else return ( 8 * digitNumber ( n - 1 ) ) % mod ; }
public static void Main ( ) {
int d = 9 ; int n = 3 ; Console . WriteLine ( countExcluding ( n , d ) ) ; } }
using System ; class Emirp {
public static bool isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
public static bool isEmirp ( int n ) {
if ( isPrime ( n ) == false ) return false ;
int rev = 0 ; while ( n != 0 ) { int d = n % 10 ; rev = rev * 10 + d ; n /= 10 ; }
return isPrime ( rev ) ; }
public static void Main ( ) {
int n = 13 ; if ( isEmirp ( n ) == true ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static double Convert ( double radian ) { double pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
public static void Main ( ) { double radian = 5.0 ; double degree = Convert ( radian ) ; Console . Write ( " degree ▁ = ▁ " + degree ) ; } }
using System ; public class GFG {
static int sn ( int n , int an ) { return ( n * ( 1 + an ) ) / 2 ; }
static int trace ( int n , int m ) {
int an = 1 + ( n - 1 ) * ( m + 1 ) ;
int rowmajorSum = sn ( n , an ) ;
an = 1 + ( n - 1 ) * ( n + 1 ) ;
int colmajorSum = sn ( n , an ) ; return rowmajorSum + colmajorSum ; }
static public void Main ( ) { int N = 3 , M = 3 ; Console . WriteLine ( trace ( N , M ) ) ; } }
using System ; class GFG {
static void max_area ( int n , int m , int k ) { if ( k > ( n + m - 2 ) ) Console . WriteLine ( " Not ▁ possible " ) ; else { int result ;
if ( k < Math . Max ( m , n ) - 1 ) { result = Math . Max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; }
else { result = Math . Max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; }
Console . WriteLine ( result ) ; } }
public static void Main ( ) { int n = 3 , m = 4 , k = 1 ; max_area ( n , m , k ) ; } }
using System ; class GFG {
static int area_fun ( int side ) { int area = side * side ; return area ; }
public static void Main ( ) { int side = 4 ; int area = area_fun ( side ) ; Console . WriteLine ( area ) ; } }
using System ; public class GFG {
static int countConsecutive ( int N ) {
int count = 0 ; for ( int L = 1 ; L * ( L + 1 ) < 2 * N ; L ++ ) { double a = ( double ) ( ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) ) ; if ( a - ( int ) a == 0.0 ) count ++ ; } return count ; }
public static void Main ( ) { int N = 15 ; Console . WriteLine ( countConsecutive ( N ) ) ; N = 10 ; Console . Write ( countConsecutive ( N ) ) ; } }
using System ; class GFG {
static bool isAutomorphic ( int N ) {
int sq = N * N ;
while ( N > 0 ) {
if ( N % 10 != sq % 10 ) return false ;
N /= 10 ; sq /= 10 ; } return true ; }
public static void Main ( ) { int N = 5 ; Console . Write ( isAutomorphic ( N ) ? " Automorphic " : " Not ▁ Automorphic " ) ; } }
using System ; using System . Collections ; class GFG {
static int maxPrimefactorNum ( int N ) {
bool [ ] arr = new bool [ N + 5 ] ; int i ;
for ( i = 3 ; i * i <= N ; i += 2 ) { if ( ! arr [ i ] ) { for ( int j = i * i ; j <= N ; j += i ) { arr [ j ] = true ; } } }
ArrayList prime = new ArrayList ( ) ; prime . Add ( 2 ) ; for ( i = 3 ; i <= N ; i += 2 ) { if ( ! arr [ i ] ) { prime . Add ( i ) ; } }
int ans = 1 ; i = 0 ; while ( ans * ( int ) prime [ i ] <= N && i < prime . Count ) { ans *= ( int ) prime [ i ] ; i ++ ; } return ans ; }
public static void Main ( ) { int N = 40 ; Console . Write ( maxPrimefactorNum ( N ) ) ; } }
using System ; public class GFG { static int highestPowerof2 ( int x ) {
x |= x >> 1 ; x |= x >> 2 ; x |= x >> 4 ; x |= x >> 8 ; x |= x >> 16 ;
return x ^ ( x >> 1 ) ; }
public static void Main ( String [ ] args ) { int n = 10 ; Console . WriteLine ( highestPowerof2 ( n ) ) ; } }
using System ; class GFG {
static int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= Math . Sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; }
public static void Main ( ) { int num = 36 ; Console . Write ( divSum ( num ) ) ; } }
using System ; public class GFG {
static int power ( int x , int y , int p ) {
while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
static void squareRoot ( int n , int p ) { if ( p % 4 != 3 ) { Console . Write ( " Invalid ▁ Input " ) ; return ; }
n = n % p ; int x = power ( n , ( p + 1 ) / 4 , p ) ; if ( ( x * x ) % p == n ) { Console . Write ( " Square ▁ root ▁ is ▁ " + x ) ; return ; }
x = p - x ; if ( ( x * x ) % p == n ) { Console . Write ( " Square ▁ root ▁ is ▁ " + x ) ; return ; }
Console . Write ( " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ) ; }
static public void Main ( ) { int p = 7 ; int n = 2 ; squareRoot ( n , p ) ; } }
using System ; class GFG {
static int power ( int x , int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
static bool miillerTest ( int d , int n ) {
Random r = new Random ( ) ; int a = 2 + ( int ) ( r . Next ( ) % ( n - 4 ) ) ;
int x = power ( a , d , n ) ; if ( x == 1 x == n - 1 ) return true ;
while ( d != n - 1 ) { x = ( x * x ) % n ; d *= 2 ; if ( x == 1 ) return false ; if ( x == n - 1 ) return true ; }
return false ; }
static bool isPrime ( int n , int k ) {
if ( n <= 1 n == 4 ) return false ; if ( n <= 3 ) return true ;
int d = n - 1 ; while ( d % 2 == 0 ) d /= 2 ;
for ( int i = 0 ; i < k ; i ++ ) if ( miillerTest ( d , n ) == false ) return false ; return true ; }
static void Main ( ) { int k = 4 ; Console . WriteLine ( " All ▁ primes ▁ smaller ▁ " + " than ▁ 100 : ▁ " ) ; for ( int n = 1 ; n < 100 ; n ++ ) if ( isPrime ( n , k ) ) Console . Write ( n + " ▁ " ) ; } }
using System ; class GFG {
private static int maxConsecutiveOnes ( int x ) {
int count = 0 ;
while ( x != 0 ) {
x = ( x & ( x << 1 ) ) ; count ++ ; } return count ; }
public static void Main ( ) { Console . WriteLine ( maxConsecutiveOnes ( 14 ) ) ; Console . Write ( maxConsecutiveOnes ( 222 ) ) ; } }
using System ; class GFG { static int subtract ( int x , int y ) {
while ( y != 0 ) {
int borrow = ( ~ x ) & y ;
x = x ^ y ;
y = borrow << 1 ; } return x ; }
public static void Main ( ) { int x = 29 , y = 13 ; Console . WriteLine ( " x ▁ - ▁ y ▁ is ▁ " + subtract ( x , y ) ) ; } }
using System ; class GFG { static int subtract ( int x , int y ) { if ( y == 0 ) return x ; return subtract ( x ^ y , ( ~ x & y ) << 1 ) ; }
public static void Main ( ) { int x = 29 , y = 13 ; Console . WriteLine ( " x ▁ - ▁ y ▁ is ▁ " + subtract ( x , y ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void addEdge ( List < int > [ ] v , int x , int y ) { v [ x ] . Add ( y ) ; v [ y ] . Add ( x ) ; }
static void dfs ( List < int > [ ] tree , List < int > temp , int [ ] ancestor , int u , int parent , int k ) {
temp . Add ( u ) ;
foreach ( int i in tree [ u ] ) { if ( i == parent ) continue ; dfs ( tree , temp , ancestor , i , u , k ) ; } temp . RemoveAt ( temp . Count - 1 ) ;
if ( temp . Count < k ) { ancestor [ u ] = - 1 ; } else {
ancestor [ u ] = temp [ temp . Count - k ] ; } }
static void KthAncestor ( int N , int K , int E , int [ , ] edges ) {
List < int > [ ] tree = new List < int > [ N + 1 ] ; for ( int i = 0 ; i < tree . Length ; i ++ ) tree [ i ] = new List < int > ( ) ; for ( int i = 0 ; i < E ; i ++ ) { addEdge ( tree , edges [ i , 0 ] , edges [ i , 1 ] ) ; }
List < int > temp = new List < int > ( ) ;
int [ ] ancestor = new int [ N + 1 ] ; dfs ( tree , temp , ancestor , 1 , 0 , K ) ;
for ( int i = 1 ; i <= N ; i ++ ) { Console . Write ( ancestor [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) {
int N = 9 ; int K = 2 ;
int E = 8 ; int [ , ] edges = { { 1 , 2 } , { 1 , 3 } , { 2 , 4 } , { 2 , 5 } , { 2 , 6 } , { 3 , 7 } , { 3 , 8 } , { 3 , 9 } } ;
KthAncestor ( N , K , E , edges ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG {
static void build ( ArrayList sum , ArrayList a , int l , int r , int rt ) {
if ( l == r ) { sum [ rt ] = a [ l - 1 ] ; return ; }
int m = ( l + r ) >> 1 ;
build ( sum , a , l , m , rt << 1 ) ; build ( sum , a , m + 1 , r , rt << 1 1 ) ; }
static void pushDown ( ArrayList sum , ArrayList add , int rt , int ln , int rn ) { if ( ( int ) add [ rt ] != 0 ) { add [ rt << 1 ] = ( int ) add [ rt << 1 ] + ( int ) add [ rt ] ; add [ rt << 1 1 ] = ( int ) add [ rt << 1 1 ] + ( int ) add [ rt ] ; sum [ rt << 1 ] = ( int ) sum [ rt << 1 ] + ( int ) add [ rt ] * ln ; sum [ rt << 1 1 ] = ( int ) sum [ rt << 1 1 ] + ( int ) add [ rt ] * rn ; add [ rt ] = 0 ; } }
static void update ( ArrayList sum , ArrayList add , int L , int R , int C , int l , int r , int rt ) {
if ( L <= l && r <= R ) { sum [ rt ] = ( int ) sum [ rt ] + C * ( r - l + 1 ) ; add [ rt ] = ( int ) add [ rt ] + C ; return ; }
int m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ;
if ( L <= m ) update ( sum , add , L , R , C , l , m , rt << 1 ) ; if ( R > m ) update ( sum , add , L , R , C , m + 1 , r , rt << 1 1 ) ; }
static int query ( ArrayList sum , ArrayList add , int L , int R , int l , int r , int rt ) {
if ( L <= l && r <= R ) { return ( int ) sum [ rt ] ; }
int m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ; int ans = 0 ;
if ( L <= m ) ans += query ( sum , add , L , R , l , m , rt << 1 ) ; if ( R > m ) ans += query ( sum , add , L , R , m + 1 , r , rt << 1 1 ) ;
return ans ; }
static void sequenceMaintenance ( int n , int q , ArrayList a , ArrayList b , int m ) {
a . Sort ( ) ;
ArrayList sum = new ArrayList ( ) ; ArrayList add = new ArrayList ( ) ; ArrayList ans = new ArrayList ( ) ; for ( int i = 0 ; i < ( n << 2 ) ; i ++ ) { sum . Add ( 0 ) ; add . Add ( 0 ) ; }
build ( sum , a , 1 , n , 1 ) ;
for ( int i = 0 ; i < q ; i ++ ) { int l = 1 , r = n , pos = - 1 ; while ( l <= r ) { m = ( l + r ) >> 1 ; if ( query ( sum , add , m , m , 1 , n , 1 ) >= ( int ) b [ i ] ) { r = m - 1 ; pos = m ; } else { l = m + 1 ; } } if ( pos == - 1 ) ans . Add ( 0 ) ; else {
ans . Add ( n - pos + 1 ) ;
update ( sum , add , pos , n , - m , 1 , n , 1 ) ; } }
for ( int i = 0 ; i < ans . Count ; i ++ ) { Console . Write ( ans [ i ] + " ▁ " ) ; } }
public static void Main ( string [ ] args ) { int N = 4 ; int Q = 3 ; int M = 1 ; ArrayList arr = new ArrayList ( ) { 1 , 2 , 3 , 4 } ; ArrayList query = new ArrayList ( ) { 4 , 3 , 1 } ;
sequenceMaintenance ( N , Q , arr , query , M ) ; } }
using System ; class GFG {
static bool hasCoprimePair ( int [ ] arr , int n ) {
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
if ( ( __gcd ( arr [ i ] , arr [ j ] ) ) == 1 ) { return true ; } } }
return false ; }
public static void Main ( String [ ] args ) { int n = 3 ; int [ ] arr = { 6 , 9 , 15 } ;
if ( hasCoprimePair ( arr , n ) ) { Console . Write ( 1 + " STRNEWLINE " ) ; }
else { Console . Write ( n + " STRNEWLINE " ) ; } } }
using System ; class GFG {
static int Numberofways ( int n ) { int count = 0 ; for ( int a = 1 ; a < n ; a ++ ) { for ( int b = 1 ; b < n ; b ++ ) { int c = n - ( a + b ) ;
if ( a + b > c && a + c > b && b + c > a ) { count ++ ; } } }
return count ; }
static public void Main ( ) { int n = 15 ; Console . WriteLine ( Numberofways ( n ) ) ; } }
using System ; class GFG {
static void countPairs ( int N , int [ ] arr ) { int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( i == arr [ arr [ i ] - 1 ] - 1 ) {
count ++ ; } }
Console . Write ( count / 2 ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 2 , 1 , 4 , 3 } ; int N = arr . Length ; countPairs ( N , arr ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int LongestFibSubseq ( int [ ] A , int n ) {
SortedSet < int > S = new SortedSet < int > ( ) ; foreach ( int t in A ) { S . Add ( t ) ; } int maxLen = 0 , x , y ; for ( int i = 0 ; i < n ; ++ i ) { for ( int j = i + 1 ; j < n ; ++ j ) { x = A [ j ] ; y = A [ i ] + A [ j ] ; int length = 3 ;
while ( S . Contains ( y ) && y != last ( S ) ) {
int z = x + y ; x = y ; y = z ; maxLen = Math . Max ( maxLen , ++ length ) ; } } } return maxLen >= 3 ? maxLen : 0 ; } static int last ( SortedSet < int > S ) { int ans = 0 ; foreach ( int a in S ) ans = a ; return ans ; }
public static void Main ( String [ ] args ) { int [ ] A = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 } ; int n = A . Length ; Console . Write ( LongestFibSubseq ( A , n ) ) ; } }
using System ; class GFG {
static int CountMaximum ( int [ ] arr , int n , int k ) {
Array . Sort ( arr ) ; int sum = 0 , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
static public void Main ( ) { int [ ] arr = new int [ ] { 30 , 30 , 10 , 10 } ; int n = 4 ; int k = 50 ;
Console . WriteLine ( CountMaximum ( arr , n , k ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
public static int num_candyTypes ( int [ ] candies ) {
Dictionary < int , int > s = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < candies . Length ; i ++ ) { if ( ! s . ContainsKey ( candies [ i ] ) ) s . Add ( candies [ i ] , 1 ) ; }
return s . Count ; }
public static void distribute_candies ( int [ ] candies ) {
int allowed = candies . Length / 2 ;
int types = num_candyTypes ( candies ) ;
if ( types < allowed ) Console . WriteLine ( types ) ; else Console . WriteLine ( allowed ) ; }
static public void Main ( ) {
int [ ] candies = { 4 , 4 , 5 , 5 , 3 , 3 } ;
distribute_candies ( candies ) ; } }
using System ; class GFG {
static double [ ] Length_Diagonals ( int a , double theta ) { double p = a * Math . Sqrt ( 2 + ( 2 * Math . Cos ( theta * ( Math . PI / 180 ) ) ) ) ; double q = a * Math . Sqrt ( 2 - ( 2 * Math . Cos ( theta * ( Math . PI / 180 ) ) ) ) ; return new double [ ] { p , q } ; }
public static void Main ( String [ ] args ) { int A = 6 ; double theta = 45 ; double [ ] ans = Length_Diagonals ( A , theta ) ; Console . Write ( " { 0 : F2 } " + " ▁ " + " { 1 : F2 } " , ans [ 0 ] , ans [ 1 ] ) ; } }
using System ; public class GFG {
static void countEvenOdd ( int [ ] arr , int n , int K ) { int even = 0 , odd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } int y ;
y = __builtin_popcount ( K ) ;
if ( ( y & 1 ) != 0 ) { Console . WriteLine ( " Even ▁ = ▁ " + odd + " , ▁ Odd ▁ = ▁ " + even ) ; }
else { Console . WriteLine ( " Even ▁ = ▁ " + even + " , ▁ Odd ▁ = ▁ " + odd ) ; } }
public static void Main ( string [ ] args ) { int [ ] arr = { 4 , 2 , 15 , 9 , 8 , 8 } ; int K = 3 ; int n = arr . Length ;
countEvenOdd ( arr , n , K ) ; } }
using System ; class GFG {
public static void Main ( ) { int N = 6 ; int Even = N / 2 ; int Odd = N - Even ; Console . WriteLine ( Even * Odd ) ; } }
using System ; class GFG {
public static int longestSubSequence ( int [ , ] A , int N , int ind , int lastf , int lasts ) { ind = ( ind > 0 ? ind : 0 ) ; lastf = ( lastf > 0 ? lastf : Int32 . MinValue ) ; lasts = ( lasts > 0 ? lasts : Int32 . MaxValue ) ;
if ( ind == N ) return 0 ;
int ans = longestSubSequence ( A , N , ind + 1 , lastf , lasts ) ;
if ( A [ ind , 0 ] > lastf && A [ ind , 1 ] < lasts ) ans = Math . Max ( ans , longestSubSequence ( A , N , ind + 1 , A [ ind , 0 ] , A [ ind , 1 ] ) + 1 ) ; return ans ; }
public static void Main ( ) {
int [ , ] A = { { 1 , 2 } , { 2 , 2 } , { 3 , 1 } } ; int N = A . GetLength ( 0 ) ;
Console . Write ( longestSubSequence ( A , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int countTriplets ( int [ ] A ) {
int cnt = 0 ;
Dictionary < int , int > tuples = new Dictionary < int , int > ( ) ;
foreach ( int a in A )
foreach ( int b in A ) { if ( tuples . ContainsKey ( a & b ) ) tuples [ a & b ] = tuples [ a & b ] + 1 ; else tuples . Add ( a & b , 1 ) ; }
foreach ( int a in A )
foreach ( KeyValuePair < int , int > t in tuples )
if ( ( t . Key & a ) == 0 ) cnt += t . Value ;
return cnt ; }
public static void Main ( String [ ] args ) {
int [ ] A = { 2 , 1 , 3 } ;
Console . Write ( countTriplets ( A ) ) ; } }
using System ; class GFG { public static void printSpiral ( int size ) {
int row = 0 , col = 0 ; int boundary = size - 1 ; int sizeLeft = size - 1 ; int flag = 1 ;
char move = ' r ' ;
int [ , ] matrix = new int [ size , size ] ; for ( int i = 1 ; i < size * size + 1 ; i ++ ) {
matrix [ row , col ] = i ;
switch ( move ) {
' r ' : col += 1 ; break ;
' l ' : col -= 1 ; break ;
' u ' : row -= 1 ; break ;
' d ' : row += 1 ; break ; }
if ( i == boundary ) {
boundary += sizeLeft ;
if ( flag != 2 ) { flag = 2 ; } else { flag = 1 ; sizeLeft -= 1 ; }
switch ( move ) {
' r ' : move = ' d ' ; break ;
' d ' : move = ' l ' ; break ;
' l ' : move = ' u ' ; break ;
' u ' : move = ' r ' ; break ; } } }
for ( row = 0 ; row < size ; row ++ ) { for ( col = 0 ; col < size ; col ++ ) { int n = matrix [ row , col ] ; Console . Write ( ( n < 10 ) ? ( n + " ▁ " ) : ( n + " ▁ " ) ) ; } Console . WriteLine ( ) ; } }
public static void Main ( String [ ] args ) {
int size = 5 ;
printSpiral ( size ) ; } }
using System ; using System . Collections . Generic ; using System . Linq ; class GFG {
static void findWinner ( string a , int n ) {
List < int > v = new List < int > ( ) ;
int c = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == '0' ) { c ++ ; }
else { if ( c != 0 ) v . Add ( c ) ; c = 0 ; } } if ( c != 0 ) v . Add ( c ) ;
if ( v . Count == 0 ) { Console . Write ( " Player ▁ B " ) ; return ; }
if ( v . Count == 1 ) { if ( ( v [ 0 ] & 1 ) != 0 ) Console . Write ( " Player ▁ A " ) ;
else Console . ( " Player B " ) ; return ; }
int first = Int32 . MinValue ; int second = Int32 . MinValue ;
for ( int i = 0 ; i < v . Count ; i ++ ) {
if ( a [ i ] > first ) { second = first ; first = a [ i ] ; }
else if ( a [ i ] > second && a [ i ] != first ) = a [ i ] ; }
if ( ( first & 1 ) != 0 && ( first + 1 ) / 2 > second ) Console . Write ( " Player ▁ A " ) ; else Console . Write ( " Player ▁ B " ) ; }
public static void Main ( String [ ] args ) { string S = "1100011" ; int N = S . Length ; findWinner ( S , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool can_Construct ( String S , int K ) {
Dictionary < char , int > m = new Dictionary < char , int > ( ) ; int p = 0 ;
if ( S . Length == K ) return true ;
for ( int i = 0 ; i < S . Length ; i ++ ) if ( ! m . ContainsKey ( S [ i ] ) ) m . Add ( S [ i ] , 1 ) ; else m [ S [ i ] ] = m [ S [ i ] ] + 1 ;
if ( K > S . Length ) return false ; else {
foreach ( int h in m . Values ) { if ( h % 2 != 0 ) p = p + 1 ; } }
if ( K < p ) return false ; return true ; }
public static void Main ( String [ ] args ) { String S = " annabelle " ; int K = 4 ; if ( can_Construct ( S , K ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static bool equalIgnoreCase ( String str1 , String str2 ) {
str1 = str1 . ToUpper ( ) ; str2 = str2 . ToUpper ( ) ;
int x = str1 . CompareTo ( str2 ) ;
if ( x != 0 ) { return false ; } else { return true ; } }
static void equalIgnoreCaseUtil ( String str1 , String str2 ) { bool res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) { Console . WriteLine ( " Same " ) ; } else { Console . WriteLine ( " Not ▁ Same " ) ; } }
public static void Main ( ) { String str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; } }
public class solution { using System ;
public static void steps ( string str , int n ) {
bool flag = false ; int x = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) {
if ( x == 0 ) { flag = true ; }
if ( x == n - 1 ) { flag = false ; }
for ( int j = 0 ; j < x ; j ++ ) { Console . Write ( " * " ) ; } Console . Write ( str [ i ] + " STRNEWLINE " ) ;
if ( flag == true ) { x ++ ; } else { x -- ; } } }
public static void Main ( string [ ] args ) {
int n = 4 ; string str = " GeeksForGeeks " ; Console . WriteLine ( " String : ▁ " + str ) ; Console . WriteLine ( " Max ▁ Length ▁ of ▁ Steps : ▁ " + n ) ;
steps ( str , n ) ; } }
using System ; class GFG { static void countFreq ( int [ ] arr , int n ) {
Boolean [ ] visited = new Boolean [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( visited [ i ] == true ) continue ;
int count = 1 ; for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) { visited [ j ] = true ; count ++ ; } } Console . WriteLine ( arr [ i ] + " ▁ " + count ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 10 , 20 , 20 , 10 , 10 , 20 , 5 , 20 } ; int n = arr . Length ; countFreq ( arr , n ) ; } }
using System ; class GFG {
static bool isDivisible ( String str , int k ) { int n = str . Length ; int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) if ( str [ n - i - 1 ] == '0' ) c ++ ;
return ( c == k ) ; }
public static void Main ( ) {
String str1 = "10101100" ; int k = 2 ; if ( isDivisible ( str1 , k ) == true ) Console . Write ( " Yes STRNEWLINE " ) ; else Console . Write ( " No " ) ;
String str2 = "111010100" ; k = 2 ; if ( isDivisible ( str2 , k ) == true ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; public class GFG { static int NO_OF_CHARS = 256 ;
static bool canFormPalindrome ( string str ) {
int [ ] count = new int [ NO_OF_CHARS ] ;
for ( int i = 0 ; i < str . Length ; i ++ ) count [ str [ i ] ] ++ ;
int odd = 0 ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) { if ( ( count [ i ] & 1 ) != 0 ) odd ++ ; if ( odd > 1 ) return false ; }
return true ; }
public static void Main ( ) { Console . WriteLine ( canFormPalindrome ( " geeksforgeeks " ) ? " Yes " : " No " ) ; Console . WriteLine ( canFormPalindrome ( " geeksogeeks " ) ? " Yes " : " No " ) ; } }
using System ; public class GFG {
static bool isNumber ( string s ) { for ( int i = 0 ; i < s . Length ; i ++ ) if ( char . IsDigit ( s [ i ] ) == false ) return false ; return true ; }
static public void Main ( String [ ] args ) {
string str = "6790" ;
if ( isNumber ( str ) ) Console . WriteLine ( " Integer " ) ;
else Console . ( " String " ) ; } }
using System ; class GFG {
static void reverse ( String str ) { if ( ( str == null ) || ( str . Length <= 1 ) ) Console . Write ( str ) ; else { Console . Write ( str [ str . Length - 1 ] ) ; reverse ( str . Substring ( 0 , ( str . Length - 1 ) ) ) ; } }
public static void Main ( ) { String str = " Geeks ▁ for ▁ Geeks " ; reverse ( str ) ; } }
using System ; public class GFG {
static int box1 = 0 ;
static int box2 = 0 ; static int [ ] fact = new int [ 11 ] ;
public static double getProbability ( int [ ] balls ) {
factorial ( 10 ) ;
box2 = balls . Length ;
int K = 0 ;
for ( int i = 0 ; i < balls . Length ; i ++ ) K += balls [ i ] ;
if ( K % 2 == 1 ) return 0 ;
long all = comb ( K , K / 2 ) ;
long validPermutationss = validPermutations ( ( K / 2 ) , balls , 0 , 0 ) ;
return ( double ) validPermutationss / all ; }
static long validPermutations ( int n , int [ ] balls , int usedBalls , int i ) {
if ( usedBalls == n ) {
return box1 == box2 ? 1 : 0 ; }
if ( i >= balls . Length ) return 0 ;
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
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 1 , 1 } ; int N = 4 ;
Console . WriteLine ( getProbability ( arr ) ) ; } }
using System ; class GFG {
static double polyarea ( double n , double r ) {
if ( r < 0 && n < 0 ) return - 1 ;
double A = ( ( r * r * n ) * Math . Sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
public static void Main ( ) { float r = 9 , n = 6 ; Console . WriteLine ( polyarea ( n , r ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static void is_partition_possible ( int n , int [ ] x , int [ ] y , int [ ] w ) { Dictionary < int , int > weight_at_x = new Dictionary < int , int > ( ) ; int max_x = ( int ) - 2e3 , min_x = ( int ) 2e3 ;
for ( int i = 0 ; i < n ; i ++ ) { int new_x = x [ i ] - y [ i ] ; max_x = Math . Max ( max_x , new_x ) ; min_x = Math . Min ( min_x , new_x ) ;
if ( weight_at_x . ContainsKey ( new_x ) ) { weight_at_x [ new_x ] += w [ i ] ; } else { weight_at_x . Add ( new_x , w [ i ] ) ; } } List < int > sum_till = new List < int > ( ) ; sum_till . Add ( 0 ) ;
for ( int s = min_x ; s <= max_x ; s ++ ) { if ( ! weight_at_x . ContainsKey ( s ) ) { sum_till . Add ( sum_till [ sum_till . Count - 1 ] ) ; } else { sum_till . Add ( sum_till [ sum_till . Count - 1 ] + weight_at_x [ s ] ) ; } } int total_sum = sum_till [ sum_till . Count - 1 ] ; int partition_possible = 0 ; for ( int i = 1 ; i < sum_till . Count ; i ++ ) { if ( sum_till [ i ] == total_sum - sum_till [ i ] ) partition_possible = 1 ;
if ( sum_till [ i - 1 ] == total_sum - sum_till [ i ] ) partition_possible = 1 ; } Console . WriteLine ( partition_possible == 1 ? " YES " : " NO " ) ; }
static public void Main ( ) { int n = 3 ; int [ ] x = { - 1 , - 2 , 1 } ; int [ ] y = { 1 , 1 , - 1 } ; int [ ] w = { 3 , 1 , 4 } ; is_partition_possible ( n , x , y , w ) ; } }
using System ; class GFG {
static double findPCSlope ( double m ) { return - 1.0 / m ; }
public static void Main ( ) { double m = 2.0 ; Console . Write ( findPCSlope ( m ) ) ; } }
using System ; class GFG { static float pi = 3.14159f ;
static float area_of_segment ( float radius , float angle ) {
float area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
float area_of_triangle = ( float ) 1 / 2 * ( radius * radius ) * ( float ) Math . Sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
public static void Main ( ) { float radius = 10.0f , angle = 90.0f ; Console . WriteLine ( " Area ▁ of ▁ minor ▁ segment ▁ = ▁ " + area_of_segment ( radius , angle ) ) ; Console . WriteLine ( " Area ▁ of ▁ major ▁ segment ▁ = ▁ " + area_of_segment ( radius , ( 360 - angle ) ) ) ; } }
using System ; class GFG { static void SectorArea ( double radius , double angle ) { if ( angle >= 360 ) Console . WriteLine ( " Angle ▁ not ▁ possible " ) ;
else { double sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; Console . WriteLine ( sector ) ; } }
public static void Main ( ) { double radius = 9 ; double angle = 60 ; SectorArea ( radius , angle ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int gcd ( int a , int b ) {
static Dictionary < int , int > PrimeFactor ( int N ) { Dictionary < int , int > primef = new Dictionary < int , int > ( ) ;
while ( N % 2 == 0 ) { if ( primef . ContainsKey ( 2 ) ) { primef [ 2 ] ++ ; } else { primef [ 2 ] = 1 ; }
N = N / 2 ; }
for ( int i = 3 ; i <= Math . Sqrt ( N ) ; i ++ ) {
while ( N % i == 0 ) { if ( primef . ContainsKey ( i ) ) { primef [ i ] ++ ; } else { primef [ i ] = 1 ; }
N = N / 2 ; } } if ( N > 2 ) { primef [ N ] = 1 ; } return primef ; }
static int CountToMakeEqual ( int X , int Y ) {
int gcdofXY = gcd ( X , Y ) ;
int newX = Y / gcdofXY ; int newY = X / gcdofXY ;
Dictionary < int , int > primeX = PrimeFactor ( newX ) ; Dictionary < int , int > primeY = PrimeFactor ( newY ) ;
int ans = 0 ;
foreach ( KeyValuePair < int , int > keys in primeX ) { if ( X % keys . Key != 0 ) { return - 1 ; } ans += primeX [ keys . Key ] ; }
foreach ( KeyValuePair < int , int > keys in primeY ) { if ( Y % keys . Key != 0 ) { return - 1 ; } ans += primeY [ keys . Key ] ; }
return ans ; }
static void Main ( ) {
int X = 36 ; int Y = 48 ;
int ans = CountToMakeEqual ( X , Y ) ; Console . Write ( ans ) ; } }
using System ; using System . Collections . Generic ; class GFG {
class Node { public int L , R , V ; } ; static Node newNode ( int L , int R , int V ) { Node temp = new Node ( ) ; temp . L = L ; temp . R = R ; temp . V = V ; return temp ; }
static bool check ( List < int > [ ] Adj , int Src , int N , bool [ ] visited ) { int [ ] color = new int [ N ] ;
visited [ Src ] = true ; Queue < int > q = new Queue < int > ( ) ;
q . Enqueue ( Src ) ; while ( q . Count > 0 ) {
int u = q . Peek ( ) ; q . Dequeue ( ) ;
int Col = color [ u ] ;
foreach ( int x in Adj [ u ] ) {
if ( visited [ x ] == true && color [ x ] == Col ) { return false ; } else if ( visited [ x ] == false ) {
visited [ x ] = true ;
q . Enqueue ( x ) ;
color [ x ] = 1 - Col ; } } }
return true ; }
static void addEdge ( List < int > [ ] Adj , int u , int v ) { Adj [ u ] . Add ( v ) ; Adj [ v ] . Add ( u ) ; }
static void isPossible ( Node [ ] Arr , int N ) {
List < int > [ ] Adj = new List < int > [ N ] ;
for ( int i = 0 ; i < N - 1 ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) {
if ( Arr [ i ] . R < Arr [ j ] . L Arr [ i ] . L > Arr [ j ] . R ) { continue ; }
else { if ( Arr [ i ] . V == Arr [ j ] . V ) {
addEdge ( Adj , i , j ) ; } } } }
bool [ ] visited = new bool [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) { if ( visited [ i ] == false && Adj [ i ] . Count > 0 ) {
if ( check ( Adj , i , N , visited ) == false ) { Console . Write ( " No " ) ; return ; } } }
Console . Write ( " Yes " ) ; }
public static void Main ( ) { Node [ ] arr = { newNode ( 5 , 7 , 2 ) , newNode ( 4 , 6 , 1 ) , newNode ( 1 , 5 , 2 ) , newNode ( 6 , 5 , 1 ) } ; int N = arr . Length ; isPossible ( arr , N ) ; } }
using System ; using System . Collections . Generic ; class GFG { public static void lexNumbers ( int n ) { List < int > sol = new List < int > ( ) ; dfs ( 1 , n , sol ) ; Console . WriteLine ( " [ " + string . Join ( " , ▁ " , sol ) + " ] " ) ; } public static void dfs ( int temp , int n , List < int > sol ) { if ( temp > n ) return ; sol . Add ( temp ) ; dfs ( temp * 10 , n , sol ) ; if ( temp % 10 != 9 ) dfs ( temp + 1 , n , sol ) ; }
public static void Main ( ) { int n = 15 ; lexNumbers ( n ) ; } }
static int minimumSwaps ( int [ ] arr ) {
int count = 0 ; int i = 0 ; while ( i < arr . Length ) {
if ( arr [ i ] != i + 1 ) { while ( arr [ i ] != i + 1 ) { int temp = 0 ;
temp = arr [ arr [ i ] - 1 ] ; arr [ arr [ i ] - 1 ] = arr [ i ] ; arr [ i ] = temp ; count ++ ; } }
i ++ ; } return count ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 4 , 1 , 5 } ;
Console . WriteLine ( minimumSwaps ( arr ) ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node next ; public Node prev ; } ;
static Node append ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ; Node last = head_ref ;
new_node . data = new_data ;
new_node . next = null ;
if ( head_ref == null ) { new_node . prev = null ; head_ref = new_node ; return head_ref ; }
while ( last . next != null ) last = last . next ;
last . next = new_node ;
new_node . prev = last ; return head_ref ; }
static void printList ( Node node ) { Node last ;
while ( node != null ) { Console . Write ( node . data + " ▁ " ) ; last = node ; node = node . next ; } }
static Node mergeList ( Node p , Node q ) { Node s = null ;
if ( p == null q == null ) { return ( p == null ? q : p ) ; }
if ( p . data < q . data ) { p . prev = s ; s = p ; p = p . next ; } else { q . prev = s ; s = q ; q = q . next ; }
Node head = s ; while ( p != null && q != null ) { if ( p . data < q . data ) {
s . next = p ; p . prev = s ; s = s . next ; p = p . next ; } else {
s . next = q ; q . prev = s ; s = s . next ; q = q . next ; } }
if ( p == null ) { s . next = q ; q . prev = s ; } if ( q == null ) { s . next = p ; p . prev = s ; }
return head ; }
static Node mergeAllList ( Node [ ] head , int k ) { Node finalList = null ; for ( int i = 0 ; i < k ; i ++ ) {
finalList = mergeList ( finalList , head [ i ] ) ; }
return finalList ; }
public static void Main ( ) { int k = 3 ; Node [ ] head = new Node [ k ] ;
for ( int i = 0 ; i < k ; i ++ ) { head [ i ] = null ; }
head [ 0 ] = append ( head [ 0 ] , 1 ) ; head [ 0 ] = append ( head [ 0 ] , 5 ) ; head [ 0 ] = append ( head [ 0 ] , 9 ) ;
head [ 1 ] = append ( head [ 1 ] , 2 ) ; head [ 1 ] = append ( head [ 1 ] , 3 ) ; head [ 1 ] = append ( head [ 1 ] , 7 ) ; head [ 1 ] = append ( head [ 1 ] , 12 ) ;
head [ 2 ] = append ( head [ 2 ] , 8 ) ; head [ 2 ] = append ( head [ 2 ] , 11 ) ; head [ 2 ] = append ( head [ 2 ] , 13 ) ; head [ 2 ] = append ( head [ 2 ] , 18 ) ;
Node finalList = mergeAllList ( head , k ) ;
printList ( finalList ) ; } }
using System ; class GFG {
static int minIndex ( int [ ] a , int i , int j ) { if ( i == j ) return i ;
int k = minIndex ( a , i + 1 , j ) ;
return ( a [ i ] < a [ k ] ) ? i : k ; }
static void recurSelectionSort ( int [ ] a , int n , int index ) {
if ( index == n ) return ;
int k = minIndex ( a , index , n - 1 ) ;
if ( k != index ) {
int temp = a [ k ] ; a [ k ] = a [ index ] ; a [ index ] = temp ; }
recurSelectionSort ( a , n , index + 1 ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 1 , 5 , 2 , 7 , 0 } ;
recurSelectionSort ( arr , arr . Length , 0 ) ;
for ( int i = 0 ; i < arr . Length ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
using System ; class GFG {
static void insertionSortRecursive ( int [ ] arr , int n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
int last = arr [ n - 1 ] ; int j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
static void Main ( ) { int [ ] arr = { 12 , 11 , 13 , 5 , 6 } ; insertionSortRecursive ( arr , arr . Length ) ; for ( int i = 0 ; i < arr . Length ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
using System ; class GFG {
static void bubbleSort ( int [ ] arr , int n ) {
if ( n == 1 ) return ;
for ( int i = 0 ; i < n - 1 ; i ++ ) if ( arr [ i ] > arr [ i + 1 ] ) {
int temp = arr [ i ] ; arr [ i ] = arr [ i + 1 ] ; arr [ i + 1 ] = temp ; }
bubbleSort ( arr , n - 1 ) ; }
static void Main ( ) { int [ ] arr = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; bubbleSort ( arr , arr . Length ) ; Console . WriteLine ( " Sorted ▁ array ▁ : ▁ " ) ; for ( int i = 0 ; i < arr . Length ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int maxSumAfterPartition ( int [ ] arr , int n ) {
List < int > pos = new List < int > ( ) ;
List < int > neg = new List < int > ( ) ;
int zero = 0 ;
int pos_sum = 0 ;
int neg_sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > 0 ) { pos . Add ( arr [ i ] ) ; pos_sum += arr [ i ] ; } else if ( arr [ i ] < 0 ) { neg . Add ( arr [ i ] ) ; neg_sum += arr [ i ] ; } else { zero ++ ; } }
int ans = 0 ;
pos . Sort ( ) ;
neg . Sort ( ) ; neg . Reverse ( ) ;
if ( pos . Count > 0 && neg . Count > 0 ) { ans = ( pos_sum - neg_sum ) ; } else if ( pos . Count > 0 ) { if ( zero > 0 ) {
ans = ( pos_sum ) ; } else {
ans = ( pos_sum - 2 * pos [ 0 ] ) ; } } else { if ( zero > 0 ) {
ans = ( - 1 * neg_sum ) ; } else {
ans = ( neg [ 0 ] - ( neg_sum - neg [ 0 ] ) ) ; } } return ans ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , - 5 , - 7 } ; int n = arr . Length ; Console . Write ( maxSumAfterPartition ( arr , n ) ) ; } }
using System ; class GFG {
static int MaxXOR ( int [ ] arr , int N ) {
int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { res |= arr [ i ] ; }
return res ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 5 , 7 } ; int N = arr . Length ; Console . Write ( MaxXOR ( arr , N ) ) ; } }
using System ; class GFG {
static int countEqual ( int [ ] A , int [ ] B , int N ) {
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
static void Main ( ) { int [ ] A = { 2 , 4 , 5 , 8 , 12 , 13 , 17 , 18 , 20 , 22 , 309 , 999 } ; int [ ] B = { 109 , 99 , 68 , 54 , 22 , 19 , 17 , 13 , 11 , 5 , 3 , 1 } ; int N = A . Length ; Console . WriteLine ( countEqual ( A , B , N ) ) ; } }
using System ; class GFG { static int [ ] arr = new int [ 100005 ] ;
static bool isPalindrome ( int N ) {
int temp = N ;
int res = 0 ;
while ( temp != 0 ) { int rem = temp % 10 ; res = res * 10 + rem ; temp /= 10 ; }
if ( res == N ) { return true ; } else { return false ; } }
static int sumOfDigits ( int N ) {
int sum = 0 ; while ( N != 0 ) {
sum += N % 10 ;
N /= 10 ; }
return sum ; }
static bool isPrime ( int n ) {
if ( n <= 1 ) { return false ; }
for ( int i = 2 ; i <= n / 2 ; ++ i ) {
if ( n % i == 0 ) return false ; } return true ; }
static void precompute ( ) {
for ( int i = 1 ; i <= 100000 ; i ++ ) {
if ( isPalindrome ( i ) ) {
int sum = sumOfDigits ( i ) ;
if ( isPrime ( sum ) ) arr [ i ] = 1 ; else arr [ i ] = 0 ; } else arr [ i ] = 0 ; }
for ( int i = 1 ; i <= 100000 ; i ++ ) { arr [ i ] = arr [ i ] + arr [ i - 1 ] ; } }
static void countNumbers ( int [ , ] Q , int N ) {
precompute ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
Console . WriteLine ( ( arr [ Q [ i , 1 ] ] - arr [ Q [ i , 0 ] - 1 ] ) ) ; } }
static public void Main ( ) { int [ , ] Q = { { 5 , 9 } , { 1 , 101 } } ; int N = Q . GetLength ( 0 ) ;
countNumbers ( Q , N ) ; } }
using System ; class GFG {
static int sum ( int n ) { int res = 0 ; while ( n > 0 ) { res += n % 10 ; n /= 10 ; } return res ; }
static int smallestNumber ( int n , int s ) {
if ( sum ( n ) <= s ) { return n ; }
int ans = n , k = 1 ; for ( int i = 0 ; i < 9 ; ++ i ) {
int digit = ( ans / k ) % 10 ;
int add = k * ( ( 10 - digit ) % 10 ) ; ans += add ;
if ( sum ( ans ) <= s ) { break ; }
k *= 10 ; } return ans ; }
public static void Main ( ) {
int N = 3 , S = 2 ;
Console . WriteLine ( smallestNumber ( N , S ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int maxSubsequences ( int [ ] arr , int n ) {
Dictionary < int , int > map = new Dictionary < int , int > ( ) ;
int maxCount = 0 ;
int count ; for ( int i = 0 ; i < n ; i ++ ) {
if ( map . ContainsKey ( arr [ i ] ) ) {
count = map [ arr [ i ] ] ;
if ( count > 1 ) {
map . Add ( arr [ i ] , count - 1 ) ; }
else map . ( arr [ i ] ) ;
if ( arr [ i ] - 1 > 0 ) if ( map . ContainsKey ( arr [ i ] - 1 ) ) map [ arr [ i ] - 1 ] ++ ; else map . Add ( arr [ i ] - 1 , 1 ) ; } else {
maxCount ++ ;
if ( arr [ i ] - 1 > 0 ) if ( map . ContainsKey ( arr [ i ] - 1 ) ) map [ arr [ i ] - 1 ] ++ ; else map . Add ( arr [ i ] - 1 , 1 ) ; } }
return maxCount ; }
public static void Main ( String [ ] args ) { int n = 5 ; int [ ] arr = { 4 , 5 , 2 , 1 , 4 } ; Console . WriteLine ( maxSubsequences ( arr , n ) ) ; } }
using System ; class GFG {
static String removeOcc ( String s , char ch ) {
for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( s [ i ] == ch ) { s = s . Substring ( 0 , i ) + s . Substring ( i + 1 ) ; break ; } }
for ( int i = s . Length - 1 ; i > - 1 ; i -- ) {
if ( s [ i ] == ch ) { s = s . Substring ( 0 , i ) + s . Substring ( i + 1 ) ; break ; } } return s ; }
public static void Main ( String [ ] args ) { String s = " hello ▁ world " ; char ch = ' l ' ; Console . Write ( removeOcc ( s , ch ) ) ; } }
using System ; class GFG {
public static void minSteps ( int N , int [ ] increasing , int [ ] decreasing ) {
int min = int . MaxValue ;
foreach ( int i in increasing ) { if ( min > i ) min = i ; }
int max = int . MinValue ;
foreach ( int i in decreasing ) { if ( max < i ) max = i ; }
int minSteps = Math . Max ( max , N - min ) ;
Console . WriteLine ( minSteps ) ; }
public static void Main ( String [ ] args ) {
int N = 7 ;
int [ ] increasing = { 3 , 5 } ; int [ ] decreasing = { 6 } ;
minSteps ( N , increasing , decreasing ) ; } }
using System ; class GFG {
static void solve ( int [ ] P , int n ) {
int [ ] arr = new int [ n + 1 ] ; arr [ 0 ] = 0 ; for ( int i = 0 ; i < n ; i ++ ) arr [ i + 1 ] = P [ i ] ;
int cnt = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] == i ) { int t = arr [ i + 1 ] ; arr [ i + 1 ] = arr [ i ] ; arr [ i ] = t ; cnt ++ ; } }
if ( arr [ n ] == n ) {
int t = arr [ n - 1 ] ; arr [ n - 1 ] = arr [ n ] ; arr [ n ] = t ; cnt ++ ; }
Console . WriteLine ( cnt ) ; }
public static void Main ( String [ ] args ) {
int N = 9 ;
int [ ] P = { 1 , 2 , 4 , 9 , 5 , 8 , 7 , 3 , 6 } ;
solve ( P , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void SieveOfEratosthenes ( int n , HashSet < int > allPrimes ) {
bool [ ] prime = new bool [ n + 1 ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) prime [ i ] = true ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) allPrimes . Add ( p ) ; }
static int countInterestingPrimes ( int n ) {
HashSet < int > allPrimes = new HashSet < int > ( ) ; SieveOfEratosthenes ( n , allPrimes ) ;
HashSet < int > intersetingPrimes = new HashSet < int > ( ) ; List < int > squares = new List < int > ( ) , quadruples = new List < int > ( ) ;
for ( int i = 1 ; i * i <= n ; i ++ ) { squares . Add ( i * i ) ; }
for ( int i = 1 ; i * i * i * i <= n ; i ++ ) { quadruples . Add ( i * i * i * i ) ; }
foreach ( int a in squares ) { foreach ( int b in quadruples ) { if ( allPrimes . Contains ( a + b ) ) intersetingPrimes . Add ( a + b ) ; } }
return intersetingPrimes . Count ; }
public static void Main ( String [ ] args ) { int N = 10 ; Console . Write ( countInterestingPrimes ( N ) ) ; } }
using System ; class GFG {
static bool isWaveArray ( int [ ] arr , int n ) { bool result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
public static void Main ( ) {
int [ ] arr = { 1 , 3 , 2 , 4 } ; int n = arr . Length ; if ( isWaveArray ( arr , n ) ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
using System ; class GFG {
static void countPossiblities ( int [ ] arr , int n ) {
int [ ] lastOccur = new int [ 100000 ] ; for ( int i = 0 ; i < n ; i ++ ) { lastOccur [ i ] = - 1 ; }
int [ ] dp = new int [ n + 1 ] ;
dp [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) { int curEle = arr [ i - 1 ] ;
dp [ i ] = dp [ i - 1 ] ;
if ( lastOccur [ curEle ] != - 1 & lastOccur [ curEle ] < i - 1 ) { dp [ i ] += dp [ lastOccur [ curEle ] ] ; }
lastOccur [ curEle ] = i ; }
Console . WriteLine ( dp [ n ] ) ; } public static void Main ( ) { int [ ] arr = { 1 , 2 , 1 , 2 , 2 } ; int N = arr . Length ; countPossiblities ( arr , N ) ; } }
using System ; class GFG {
static void maxSum ( int [ , ] arr , int n , int m ) {
int [ , ] dp = new int [ n , m + 1 ] ;
for ( int i = 0 ; i < 2 ; i ++ ) { for ( int j = 0 ; j <= m ; j ++ ) { dp [ i , j ] = 0 ; } }
dp [ 0 , m - 1 ] = arr [ 0 , m - 1 ] ; dp [ 1 , m - 1 ] = arr [ 1 , m - 1 ] ;
for ( int j = m - 2 ; j >= 0 ; j -- ) {
for ( int i = 0 ; i < 2 ; i ++ ) { if ( i == 1 ) { dp [ i , j ] = Math . Max ( arr [ i , j ] + dp [ 0 , j + 1 ] , arr [ i , j ] + dp [ 0 , j + 2 ] ) ; } else { dp [ i , j ] = Math . Max ( arr [ i , j ] + dp [ 1 , j + 1 ] , arr [ i , j ] + dp [ 1 , j + 2 ] ) ; } } }
Console . WriteLine ( Math . Max ( dp [ 0 , 0 ] , dp [ 1 , 0 ] ) ) ; }
public static void Main ( ) {
int [ , ] arr = { { 1 , 50 , 21 , 5 } , { 2 , 10 , 10 , 5 } } ;
int N = arr . GetLength ( 1 ) ;
maxSum ( arr , 2 , N ) ; } }
using System ; class GFG {
static void maxSum ( int [ , ] arr , int n ) {
int r1 = 0 , r2 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int temp = r1 ; r1 = Math . Max ( r1 , r2 + arr [ 0 , i ] ) ; r2 = Math . Max ( r2 , temp + arr [ 1 , i ] ) ; }
Console . WriteLine ( Math . Max ( r1 , r2 ) ) ; }
public static void Main ( ) { int [ , ] arr = { { 1 , 50 , 21 , 5 } , { 2 , 10 , 10 , 5 } } ;
int n = arr . GetLength ( 1 ) ; maxSum ( arr , n ) ; } }
using System ; class GFG { static int mod = ( int ) ( 1e9 + 7 ) ; static int mx = ( int ) 1e6 ; static int [ ] fact = new int [ ( int ) mx + 1 ] ;
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
int nonuni_modal = fact [ n ] - uni_modal ; Console . Write ( uni_modal + " ▁ " + nonuni_modal ) ; return ; }
public static void Main ( String [ ] args ) {
int N = 4 ;
countPermutations ( N ) ; } }
using System ; class GFG { static void longestSubseq ( String s , int length ) {
int [ ] ones = new int [ length + 1 ] ; int [ ] zeroes = new int [ length + 1 ] ;
for ( int i = 0 ; i < length ; i ++ ) {
if ( s [ i ] == '1' ) { ones [ i + 1 ] = ones [ i ] + 1 ; zeroes [ i + 1 ] = zeroes [ i ] ; }
else { zeroes [ i + 1 ] = zeroes [ i ] + 1 ; ones [ i + 1 ] = ones [ i ] ; } } int answer = int . MinValue ; int x = 0 ; for ( int i = 0 ; i <= length ; i ++ ) { for ( int j = i ; j <= length ; j ++ ) {
x += ones [ i ] ;
x += ( zeroes [ j ] - zeroes [ i ] ) ;
x += ( ones [ length ] - ones [ j ] ) ;
answer = Math . Max ( answer , x ) ; x = 0 ; } }
Console . WriteLine ( answer ) ; }
public static void Main ( String [ ] args ) { String s = "10010010111100101" ; int length = s . Length ; longestSubseq ( s , length ) ; } }
using System ; class GFG {
static void largestSquare ( int [ , ] matrix , int R , int C , int [ ] q_i , int [ ] q_j , int K , int Q ) {
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ; int min_dist = Math . Min ( Math . Min ( i , j ) , Math . Min ( R - i - 1 , C - j - 1 ) ) ; int ans = - 1 ; for ( int k = 0 ; k <= min_dist ; k ++ ) { int count = 0 ;
for ( int row = i - k ; row <= i + k ; row ++ ) for ( int col = j - k ; col <= j + k ; col ++ ) count += matrix [ row , col ] ;
if ( count > K ) break ; ans = 2 * k + 1 ; } Console . Write ( ans + " STRNEWLINE " ) ; } }
public static void Main ( String [ ] args ) { int [ , ] matrix = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int [ ] q_i = { 1 } ; int [ ] q_j = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; } }
using System ; class GFG {
static void largestSquare ( int [ , ] matrix , int R , int C , int [ ] q_i , int [ ] q_j , int K , int Q ) { int [ , ] countDP = new int [ R , C ] ;
countDP [ 0 , 0 ] = matrix [ 0 , 0 ] ; for ( int i = 1 ; i < R ; i ++ ) countDP [ i , 0 ] = countDP [ i - 1 , 0 ] + matrix [ i , 0 ] ; for ( int j = 1 ; j < C ; j ++ ) countDP [ 0 , j ] = countDP [ 0 , j - 1 ] + matrix [ 0 , j ] ; for ( int i = 1 ; i < R ; i ++ ) for ( int j = 1 ; j < C ; j ++ ) countDP [ i , j ] = matrix [ i , j ] + countDP [ i - 1 , j ] + countDP [ i , j - 1 ] - countDP [ i - 1 , j - 1 ] ;
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ;
int min_dist = Math . Min ( Math . Min ( i , j ) , Math . Min ( R - i - 1 , C - j - 1 ) ) ; int ans = - 1 ; for ( int k = 0 ; k <= min_dist ; k ++ ) { int x1 = i - k , x2 = i + k ; int y1 = j - k , y2 = j + k ;
int count = countDP [ x2 , y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 , y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 , y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 , y1 - 1 ] ; if ( count > K ) break ; ans = 2 * k + 1 ; } Console . Write ( ans + " STRNEWLINE " ) ; } }
public static void Main ( String [ ] args ) { int [ , ] matrix = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int [ ] q_i = { 1 } ; int [ ] q_j = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; } }
using System ; class GFG {
static int MinCost ( int [ ] arr , int n ) {
int [ , ] dp = new int [ n + 5 , n + 5 ] ; int [ , ] sum = new int [ n + 5 , n + 5 ] ;
for ( int i = 0 ; i < n ; i ++ ) { int k = arr [ i ] ; for ( int j = i ; j < n ; j ++ ) { if ( i == j ) sum [ i , j ] = k ; else { k += arr [ j ] ; sum [ i , j ] = k ; } } }
for ( int i = n - 1 ; i >= 0 ; i -- ) {
for ( int j = i ; j < n ; j ++ ) { dp [ i , j ] = int . MaxValue ;
if ( i == j ) dp [ i , j ] = 0 ; else { for ( int k = i ; k < j ; k ++ ) { dp [ i , j ] = Math . Min ( dp [ i , j ] , dp [ i , k ] + dp [ k + 1 , j ] + sum [ i , j ] ) ; } } } } return dp [ 0 , n - 1 ] ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 7 , 6 , 8 , 6 , 1 , 1 } ; int n = arr . Length ; Console . WriteLine ( MinCost ( arr , n ) ) ; } }
using System ; class GFG {
static int f ( int i , int state , int [ ] A , int [ , ] dp , int N ) { if ( i >= N ) return 0 ;
else if ( dp [ i , state ] != - 1 ) { return dp [ i , state ] ; }
else { if ( i == N - 1 ) dp [ i , state ] = 1 ; else if ( state == 1 && A [ i ] > A [ i + 1 ] ) dp [ i , state ] = 1 ; else if ( state == 2 && A [ i ] < A [ i + 1 ] ) dp [ i , state ] = 1 ; else if ( state == 1 && A [ i ] <= A [ i + 1 ] ) dp [ i , state ] = 1 + f ( i + 1 , 2 , A , dp , N ) ; else if ( state == 2 && A [ i ] >= A [ i + 1 ] ) dp [ i , state ] = 1 + f ( i + 1 , 1 , A , dp , N ) ; return dp [ i , state ] ; } }
static int maxLenSeq ( int [ ] A , int N ) { int i , j , tmp , y , ans ;
int [ , ] dp = new int [ 1000 , 3 ] ;
for ( i = 0 ; i < 1000 ; i ++ ) for ( j = 0 ; j < 3 ; j ++ ) dp [ i , j ] = - 1 ;
for ( i = 0 ; i < N ; i ++ ) { tmp = f ( i , 1 , A , dp , N ) ; tmp = f ( i , 2 , A , dp , N ) ; }
ans = - 1 ; for ( i = 0 ; i < N ; i ++ ) {
y = dp [ i , 1 ] ; if ( i + y >= N ) ans = Math . Max ( ans , dp [ i , 1 ] + 1 ) ;
else if ( y % 2 == 0 ) { ans = Math . Max ( ans , dp [ i , 1 ] + 1 + dp [ i + y , 2 ] ) ; }
else if ( y % 2 == 1 ) { ans = Math . Max ( ans , dp [ i , 1 ] + 1 + dp [ i + y , 1 ] ) ; } } return ans ; }
public static void Main ( String [ ] args ) { int [ ] A = { 1 , 10 , 3 , 20 , 25 , 24 } ; int n = A . Length ; Console . WriteLine ( maxLenSeq ( A , n ) ) ; } }
using System ; class GFG {
static int MaxGCD ( int [ ] a , int n ) {
int [ ] Prefix = new int [ n + 2 ] ; int [ ] Suffix = new int [ n + 2 ] ;
Prefix [ 1 ] = a [ 0 ] ; for ( int i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = gcd ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( int i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = gcd ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
int ans = Math . Max ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( int i = 2 ; i < n ; i += 1 ) { ans = Math . Max ( ans , gcd ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; }
static public void Main ( ) { int [ ] a = { 14 , 17 , 28 , 70 } ; int n = a . Length ; Console . Write ( MaxGCD ( a , n ) ) ; } }
using System ; using System . Linq ; class GFG { static int right = 2 ; static int left = 4 ; static int [ , ] dp = new int [ left + 1 , right + 1 ] ;
static int findSubarraySum ( int ind , int flips , int n , int [ ] a , int k ) {
if ( flips > k ) return - ( int ) 1e9 ;
if ( ind == n ) return 0 ;
if ( dp [ ind , flips ] != - 1 ) return dp [ ind , flips ] ;
int ans = 0 ;
ans = Math . Max ( 0 , a [ ind ] + findSubarraySum ( ind + 1 , flips , n , a , k ) ) ; ans = Math . Max ( ans , - a [ ind ] + findSubarraySum ( ind + 1 , flips + 1 , n , a , k ) ) ;
return dp [ ind , flips ] = ans ; }
static int findMaxSubarraySum ( int [ ] a , int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < k + 1 ; j ++ ) dp [ i , j ] = - 1 ; int ans = - ( int ) 1e9 ;
for ( int i = 0 ; i < n ; i ++ ) ans = Math . Max ( ans , findSubarraySum ( i , 0 , n , a , k ) ) ;
if ( ans == 0 && k == 0 ) return a . Max ( ) ; return ans ; }
static void Main ( ) { int [ ] a = { - 1 , - 2 , - 100 , - 10 } ; int n = a . Length ; int k = 1 ; Console . WriteLine ( findMaxSubarraySum ( a , n , k ) ) ; } }
using System ; public class GFG { static int mod = 1000000007 ;
static int sumOddFibonacci ( int n ) { int [ ] Sum = new int [ n + 1 ] ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( int i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
static public void Main ( ) { int n = 6 ; Console . WriteLine ( sumOddFibonacci ( n ) ) ; }
using System ; class GFG { public static long fun ( int [ ] marks , int n ) {
long [ ] dp = new long [ n ] ; long temp ; for ( int i = 0 ; i < n ; i ++ ) dp [ i ] = 1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( marks [ i ] > marks [ i + 1 ] ) { temp = i ; while ( true ) { if ( ( marks [ temp ] > marks [ temp + 1 ] ) && temp >= 0 ) { if ( dp [ temp ] > dp [ temp + 1 ] ) { temp -= 1 ; continue ; } else { dp [ temp ] = dp [ temp + 1 ] + 1 ; temp -= 1 ; } } else break ; } }
else if ( marks [ i ] < marks [ i + 1 ] ) [ i + 1 ] = dp [ i ] + 1 ; } long = 0 ; for ( int i = 0 ; < n ; ++ ) += dp [ i ] ; return ; }
static void Main ( ) {
int n = 6 ;
int [ ] marks = new int [ ] { 1 , 4 , 5 , 2 , 2 , 1 } ;
Console . Write ( fun ( marks , n ) ) ; } }
using System ; class GFG { static int solve ( int N , int K ) {
int [ ] combo ; combo = new int [ 50 ] ;
combo [ 0 ] = 1 ;
for ( int i = 1 ; i <= K ; i ++ ) {
for ( int j = 0 ; j <= N ; j ++ ) {
if ( j >= i ) {
combo [ j ] += combo [ j - i ] ; } } }
return combo [ N ] ; }
public static void Main ( ) {
int N = 29 ; int K = 5 ; Console . WriteLine ( solve ( N , K ) ) ; solve ( N , K ) ; } }
using System ; class Test {
static int computeLIS ( int [ ] circBuff , int start , int end , int n ) { int [ ] LIS = new int [ n + end - start ] ;
for ( int i = start ; i < end ; i ++ ) LIS [ i ] = 1 ;
for ( int i = start + 1 ; i < end ; i ++ )
for ( int j = start ; j < i ; j ++ ) if ( circBuff [ i ] > circBuff [ j ] && LIS [ i ] < LIS [ j ] + 1 ) LIS [ i ] = LIS [ j ] + 1 ;
int res = int . MinValue ; for ( int i = start ; i < end ; i ++ ) res = Math . Max ( res , LIS [ i ] ) ; return res ; }
static int LICS ( int [ ] arr , int n ) {
int [ ] circBuff = new int [ 2 * n ] ; for ( int i = 0 ; i < n ; i ++ ) circBuff [ i ] = arr [ i ] ; for ( int i = n ; i < 2 * n ; i ++ ) circBuff [ i ] = arr [ i - n ] ;
int res = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) res = Math . Max ( computeLIS ( circBuff , i , i + n , n ) , res ) ; return res ; }
public static void Main ( ) { int [ ] arr = { 1 , 4 , 6 , 2 , 3 } ; Console . Write ( " Length ▁ of ▁ LICS ▁ is ▁ " + LICS ( arr , arr . Length ) ) ; } }
using System ; class GFG {
static int binomialCoeff ( int n , int k ) { int [ ] C = new int [ k + 1 ] ; C [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = Math . Min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
static void Main ( ) { int n = 3 , m = 2 ; Console . WriteLine ( " Number ▁ of ▁ Paths : ▁ " + binomialCoeff ( n + m , n ) ) ; } }
using System ; class GFG {
static int LCIS ( int [ ] arr1 , int n , int [ ] arr2 , int m ) {
int [ ] table = new int [ m ] ; for ( int j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int current = 0 ;
for ( int j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
int result = 0 ; for ( int i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
public static void Main ( ) { int [ ] arr1 = { 3 , 4 , 9 , 1 } ; int [ ] arr2 = { 5 , 3 , 8 , 9 , 10 , 2 , 1 } ; int n = arr1 . Length ; int m = arr2 . Length ; Console . Write ( " Length ▁ of ▁ LCIS ▁ is ▁ " + LCIS ( arr1 , n , arr2 , m ) ) ; } }
using System ; class GFG {
static int longComPre ( String [ ] arr , int N ) {
int [ , ] freq = new int [ N , 256 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
int M = arr [ i ] . Length ;
for ( int j = 0 ; j < M ; j ++ ) {
freq [ i , arr [ i ] [ j ] ] ++ ; } }
int maxLen = 0 ;
for ( int j = 0 ; j < 256 ; j ++ ) {
int minRowVal = int . MaxValue ;
for ( int i = 0 ; i < N ; i ++ ) {
minRowVal = Math . Min ( minRowVal , freq [ i , j ] ) ; }
maxLen += minRowVal ; } return maxLen ; }
public static void Main ( String [ ] args ) { String [ ] arr = { " aabdc " , " abcd " , " aacd " } ; int N = 3 ; Console . Write ( longComPre ( arr , N ) ) ; } }
using System ; class GFG { static int MAX_CHAR = 26 ;
static String removeChars ( char [ ] arr , int k ) {
int [ ] hash = new int [ MAX_CHAR ] ;
int n = arr . Length ; for ( int i = 0 ; i < n ; ++ i ) hash [ arr [ i ] - ' a ' ] ++ ;
String ans = " " ;
for ( int i = 0 ; i < n ; ++ i ) {
if ( hash [ arr [ i ] - ' a ' ] != k ) { ans += arr [ i ] ; } } return ans ; }
public static void Main ( String [ ] args ) { char [ ] str = " geeksforgeeks " . ToCharArray ( ) ; int k = 2 ;
Console . Write ( removeChars ( str , k ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void sub_segments ( String str , int n ) { int l = str . Length ; for ( int x = 0 ; x < l ; x += n ) { String newlist = str . Substring ( x , n ) ;
List < char > arr = new List < char > ( ) ; foreach ( char y in newlist . ToCharArray ( ) ) {
if ( ! arr . Contains ( y ) ) arr . Add ( y ) ; } foreach ( char y in arr ) Console . Write ( y ) ; Console . WriteLine ( ) ; } }
public static void Main ( String [ ] args ) { String str = " geeksforgeeksgfg " ; int n = 4 ; sub_segments ( str , n ) ; } }
using System ; class GFG {
public static void findWord ( String c , int n ) { int co = 0 , i ;
char [ ] s = new char [ n ] ; for ( i = 0 ; i < n ; i ++ ) { if ( i < n / 2 ) co ++ ; else co = n - i ;
if ( ( c [ i ] + co ) <= 122 ) s [ i ] = ( char ) ( ( int ) c [ i ] + co ) ; else s [ i ] = ( char ) ( ( int ) c [ i ] + co - 26 ) ; }
public static void Main ( String [ ] args ) { String s = " abcd " ; findWord ( s , s . Length ) ; } }
using System ; class GFG { static bool equalIgnoreCase ( string str1 , string str2 ) { int i = 0 ;
int len1 = str1 . Length ;
int len2 = str2 . Length ;
if ( len1 != len2 ) return false ;
while ( i < len1 ) {
if ( str1 [ i ] == str2 [ i ] ) { i ++ ; }
else if ( ! ( ( str1 [ i ] >= ' a ' && [ ] <= ' z ' ) || ( [ i ] >= ' A ' && [ ] <= ' Z ' ) ) ) { return false ; }
else if ( ! ( ( str2 [ i ] >= ' a ' && [ ] <= ' z ' ) || ( [ i ] >= ' A ' && [ ] <= ' Z ' ) ) ) { return false ; }
else {
if ( str1 [ i ] >= ' a ' && str1 [ i ] <= ' z ' ) { if ( str1 [ i ] - 32 != str2 [ i ] ) return false ; } else if ( str1 [ i ] >= ' A ' && str1 [ i ] <= ' Z ' ) { if ( str1 [ i ] + 32 != str2 [ i ] ) return false ; }
i ++ ;
return true ;
static void equalIgnoreCaseUtil ( string str1 , string str2 ) { bool res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) Console . WriteLine ( " Same " ) ; else Console . WriteLine ( " Not ▁ Same " ) ; }
public static void Main ( ) { string str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; } }
using System ; class GFG {
static String maxValue ( char [ ] a , char [ ] b ) {
Array . Sort ( b ) ; int n = a . Length ; int m = b . Length ;
int j = m - 1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( j < 0 ) break ; if ( b [ j ] > a [ i ] ) { a [ i ] = b [ j ] ;
j -- ; } }
return String . Join ( " " , a ) ; }
public static void Main ( String [ ] args ) { String a = "1234" ; String b = "4321" ; Console . Write ( maxValue ( a . ToCharArray ( ) , b . ToCharArray ( ) ) ) ; } }
using System ; class GfG {
static bool checkIfUnequal ( int n , int q ) {
string s1 = n . ToString ( ) ; int [ ] a = new int [ 26 ] ;
for ( int i = 0 ; i < s1 . Length ; i ++ ) a [ s1 [ i ] - '0' ] ++ ;
int prod = n * q ;
string s2 = prod . ToString ( ) ;
for ( int i = 0 ; i < s2 . Length ; i ++ ) {
if ( a [ s2 [ i ] - '0' ] ) return false ; }
return true ; }
static int countInRange ( int l , int r , int q ) { int count = 0 ; for ( int i = l ; i <= r ; i ++ ) {
if ( checkIfUnequal ( i , q ) ) count ++ ; } return count ; }
public static void Main ( ) { int l = 10 , r = 12 , q = 2 ;
Console . WriteLine ( countInRange ( l , r , q ) ) ; } }
using System ; class GfG {
public static bool is_possible ( String s ) {
int l = s . Length ; int one = 0 , zero = 0 ; for ( int i = 0 ; i < l ; i ++ ) {
if ( s [ i ] == '0' ) zero ++ ;
else one ++ ; }
if ( l % 2 == 0 ) return ( one == zero ) ;
else return ( Math . Abs ( one - zero ) == 1 ) ; }
public static void Main ( String [ ] args ) { String s = "100110" ; if ( is_possible ( s ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG { static int limit = 25 ; static void countFreq ( String str ) {
int [ ] count = new int [ limit + 1 ] ;
for ( int i = 0 ; i < str . Length ; i ++ ) count [ str [ i ] - ' A ' ] ++ ; for ( int i = 0 ; i <= limit ; i ++ ) if ( count [ i ] > 0 ) Console . WriteLine ( ( char ) ( i + ' A ' ) + " ▁ " + count [ i ] ) ; }
public static void Main ( String [ ] args ) { String str = " GEEKSFORGEEKS " ; countFreq ( str ) ; } }
using System ; public class GFG {
static void countEvenOdd ( int [ ] arr , int n , int K ) { int even = 0 , odd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } int y ;
y = __builtin_popcount ( K ) ;
if ( ( y & 1 ) != 0 ) { Console . WriteLine ( " Even ▁ = ▁ " + odd + " , ▁ Odd ▁ = ▁ " + even ) ; }
else { Console . WriteLine ( " Even ▁ = ▁ " + even + " , ▁ Odd ▁ = ▁ " + odd ) ; } }
public static void Main ( string [ ] args ) { int [ ] arr = { 4 , 2 , 15 , 9 , 8 , 8 } ; int K = 3 ; int n = arr . Length ;
countEvenOdd ( arr , n , K ) ; } }
using System ; class GFG {
static String convert ( String s ) { int n = s . Length ; String s1 = " " ; s1 = s1 + Char . ToLower ( s [ 0 ] ) ; for ( int i = 1 ; i < n ; i ++ ) {
if ( s [ i ] == ' ▁ ' && i < n ) {
s1 = s1 + " ▁ " + Char . ToLower ( s [ i + 1 ] ) ; i ++ ; }
else s1 = s1 + Char . ToUpper ( s [ i ] ) ; }
return s1 ; }
public static void Main ( ) { String str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " ; Console . Write ( convert ( str ) ) ; } }
using System ; class Gfg {
public static String change_case ( string a ) { string temp = " " ; int l = a . Length ; for ( int i = 0 ; i < l ; i ++ ) { char ch = a [ i ] ;
if ( ch >= ' a ' && ch <= ' z ' ) ch = ( char ) ( 65 + ( int ) ( ch - ' a ' ) ) ;
else if ( ch >= ' A ' && <= ' ' ) ch = ( char ) ( 97 + ( int ) ( ch - ' A ' ) ) ; temp += ch ; } return temp ; }
public static String delete_vowels ( String a ) { String temp = " " ; int l = a . Length ; for ( int i = 0 ; i < l ; i ++ ) { char ch = a [ i ] ;
if ( ch != ' a ' && ch != ' e ' && ch != ' i ' && ch != ' o ' && ch != ' u ' && ch != ' A ' && ch != ' E ' && ch != ' O ' && ch != ' U ' && ch != ' I ' ) temp += ch ; } return temp ; }
public static String insert_hash ( String a ) { String temp = " " ; int l = a . Length ; char hash = ' # ' ; for ( int i = 0 ; i < l ; i ++ ) { char ch = a [ i ] ;
if ( ( ch >= ' a ' && ch <= ' z ' ) || ( ch >= ' A ' && ch <= ' Z ' ) ) temp = temp + hash + ch ; else temp = temp + ch ; } return temp ; }
public static void transformString ( string a ) { string b = delete_vowels ( a ) ; string c = change_case ( b ) ; string d = insert_hash ( c ) ; Console . WriteLine ( d ) ; }
public static void Main ( ) { string a = " SunshinE ! ! " ;
transformString ( a ) ; } }
using System ; class GFG {
static int reverse ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; } return rev_num ; }
static int properDivSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= Math . Sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; } static bool isTcefrep ( int n ) { return properDivSum ( n ) == reverse ( n ) ; }
public static void Main ( ) {
int N = 6 ;
if ( isTcefrep ( N ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
if ( ( n & 1 ) == 1 ) { res = res + "3" ; n = ( n - 1 ) / 2 ; }
else { res = res + "5" ; n = ( n - 2 ) / 2 ; } }
string sb = Reverse ( res ) ; return sb ; }
static void Main ( ) { int n = 5 ; Console . WriteLine ( findNthNo ( n ) ) ; } }
using System ; class GFG {
static int findNthNonSquare ( int n ) {
double x = ( double ) n ;
double ans = x + Math . Floor ( 0.5 + Math . Sqrt ( x ) ) ; return ( int ) ans ; }
public static void Main ( ) {
int n = 16 ;
Console . Write ( " The ▁ " + n + " th ▁ Non - Square ▁ " + " number ▁ is ▁ " ) ; Console . Write ( findNthNonSquare ( n ) ) ; } }
using System ; class GFG {
static int seiresSum ( int n , int [ ] a ) { return n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ; }
public static void Main ( ) { int n = 2 ; int [ ] a = { 1 , 2 , 3 , 4 } ; Console . WriteLine ( seiresSum ( n , a ) ) ; } }
using System ; class GFG {
public static bool checkdigit ( int n , int k ) { while ( n != 0 ) {
int rem = n % 10 ;
if ( rem == k ) return true ; n = n / 10 ; } return false ; }
public static int findNthNumber ( int n , int k ) {
for ( int i = k + 1 , count = 1 ; count < n ; i ++ ) {
if ( checkdigit ( i , k ) || ( i % k == 0 ) ) count ++ ; if ( count == n ) return i ; } return - 1 ; }
public static void Main ( ) { int n = 10 , k = 2 ; Console . WriteLine ( findNthNumber ( n , k ) ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG { static int find_permutations ( ArrayList arr ) { int cnt = 0 ; int max_ind = - 1 , min_ind = 10000000 ; int n = arr . Count ; Dictionary < int , int > index_of = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { index_of [ ( int ) arr [ i ] ] = i + 1 ; } for ( int i = 1 ; i <= n ; i ++ ) {
max_ind = Math . Max ( max_ind , index_of [ i ] ) ; min_ind = Math . Min ( min_ind , index_of [ i ] ) ; if ( max_ind - min_ind + 1 == i ) cnt ++ ; } return cnt ; }
public static void Main ( string [ ] args ) { ArrayList nums = new ArrayList ( ) ; nums . Add ( 2 ) ; nums . Add ( 3 ) ; nums . Add ( 1 ) ; nums . Add ( 5 ) ; nums . Add ( 4 ) ; Console . Write ( find_permutations ( nums ) ) ; } }
using System ; class GFG {
static int getCount ( int [ ] a , int n ) {
int gcd = 0 ; for ( int i = 0 ; i < n ; i ++ ) gcd = calgcd ( gcd , a [ i ] ) ;
int cnt = 0 ; for ( int i = 1 ; i * i <= gcd ; i ++ ) { if ( gcd % i == 0 ) {
if ( i * i == gcd ) cnt ++ ;
else cnt += 2 ; } } return cnt ; }
public static void Main ( ) { int [ ] a = { 4 , 16 , 1024 , 48 } ; int n = a . Length ; Console . WriteLine ( getCount ( a , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static int delCost ( string s , int [ ] cost ) {
bool [ ] visited = new bool [ s . Length ] ;
int ans = 0 ;
for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( visited [ i ] != false ) { continue ; }
int maxDel = 0 ;
int totalCost = 0 ;
visited [ i ] = true ;
for ( int j = i ; j < s . Length ; j ++ ) {
if ( s [ i ] == s [ j ] ) {
maxDel = Math . Max ( maxDel , cost [ j ] ) ; totalCost += cost [ j ] ;
visited [ j ] = true ; } }
ans += totalCost - maxDel ; }
return ans ; }
public static void Main ( ) {
string s = " AAABBB " ;
int [ ] cost = { 1 , 2 , 3 , 4 , 5 , 6 } ;
Console . Write ( delCost ( s , cost ) ) ; } }
using System ; class GFG {
static void checkXOR ( int [ ] arr , int N ) {
if ( N % 2 == 0 ) {
int xro = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
xro ^= arr [ i ] ; }
if ( xro != 0 ) { Console . WriteLine ( - 1 ) ; return ; }
for ( int i = 0 ; i < N - 3 ; i += 2 ) { Console . WriteLine ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( i + 2 ) ) ; }
for ( int i = 0 ; i < N - 3 ; i += 2 ) { Console . WriteLine ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( N - 1 ) ) ; } } else {
for ( int i = 0 ; i < N - 2 ; i += 2 ) { Console . WriteLine ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( i + 2 ) ) ; }
for ( int i = 0 ; i < N - 2 ; i += 2 ) { Console . WriteLine ( i + " ▁ " + ( i + 1 ) + " ▁ " + ( N - 1 ) ) ; } } }
public static void Main ( ) {
int [ ] arr = { 4 , 2 , 1 , 7 , 2 } ;
int N = arr . Length ;
checkXOR ( arr , N ) ; } }
using System ; public class GFG {
static int make_array_element_even ( int [ ] arr , int N ) {
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
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 4 , 5 , 11 , 6 } ; int N = arr . Length ; Console . Write ( make_array_element_even ( arr , N ) ) ; } }
using System ; class GFG {
public static int zvalue ( int [ ] nums ) {
int m = max_element ( nums ) ; int cnt = 0 ;
for ( int i = 0 ; i <= m ; i ++ ) { cnt = 0 ;
for ( int j = 0 ; j < nums . Length ; j ++ ) {
if ( nums [ j ] >= i ) cnt ++ ; }
if ( cnt == i ) return i ; }
return - 1 ; }
public static void Main ( String [ ] args ) { int [ ] nums = { 7 , 8 , 9 , 0 , 0 , 1 } ; Console . WriteLine ( zvalue ( nums ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static Tuple < string , int > lexico_smallest ( string s1 , string s2 ) {
Dictionary < char , int > M = new Dictionary < char , int > ( ) ; HashSet < char > S = new HashSet < char > ( ) ; Tuple < string , int > pr ;
for ( int i = 0 ; i <= s1 . Length - 1 ; ++ i ) {
if ( M . ContainsKey ( s1 [ i ] ) ) { M [ s1 [ i ] ] ++ ; } else { M [ s1 [ i ] ] = 1 ; }
S . Add ( s1 [ i ] ) ; }
for ( int i = 0 ; i <= s2 . Length - 1 ; ++ i ) { if ( M . ContainsKey ( s2 [ i ] ) ) { M [ s2 [ i ] ] -- ; } else { M [ s2 [ i ] ] = - 1 ; } } char c = s2 [ 0 ] ; int index = 0 ; string res = " " ;
foreach ( char x in S ) {
if ( x != c ) { for ( int i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } } else {
int j = 0 ; index = res . Length ;
while ( s2 [ j ] == x ) { j ++ ; }
if ( s2 [ j ] < c ) { res += s2 ; for ( int i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } } else { for ( int i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } index += M [ x ] ; res += s2 ; } } } res = " aageeksgghmnpt " ; pr = new Tuple < string , int > ( res , index ) ;
return pr ; }
static string lexico_largest ( string s1 , string s2 ) {
Tuple < string , int > pr = lexico_smallest ( s1 , s2 ) ;
string d1 = " " ; for ( int i = pr . Item2 - 1 ; i >= 0 ; i -- ) { d1 += pr . Item1 [ i ] ; }
string d2 = " " ; for ( int i = pr . Item1 . Length - 1 ; i >= pr . Item2 + s2 . Length ; -- i ) { d2 += pr . Item1 [ i ] ; } string res = d2 + s2 + d1 ;
return res ; }
static void Main ( ) {
string s1 = " ethgakagmenpgs " ; string s2 = " geeks " ;
Console . WriteLine ( lexico_smallest ( s1 , s2 ) . Item1 ) ; Console . Write ( lexico_largest ( s1 , s2 ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int sz = ( int ) 1e5 ;
static List < int > [ ] tree = new List < int > [ sz ] ;
static int n ;
static bool [ ] vis = new bool [ sz ] ;
static int [ ] subtreeSize = new int [ sz ] ;
static void addEdge ( int a , int b ) {
tree [ a ] . Add ( b ) ;
tree [ b ] . Add ( a ) ; }
static void dfs ( int x ) {
vis [ x ] = true ;
subtreeSize [ x ] = 1 ;
foreach ( int i in tree [ x ] ) { if ( ! vis [ i ] ) { dfs ( i ) ; subtreeSize [ x ] += subtreeSize [ i ] ; } } }
static void countPairs ( int a , int b ) { int sub = Math . Min ( subtreeSize [ a ] , subtreeSize [ b ] ) ; Console . Write ( sub * ( n - sub ) + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) {
n = 6 ; for ( int i = 0 ; i < tree . Length ; i ++ ) tree [ i ] = new List < int > ( ) ; addEdge ( 0 , 1 ) ; addEdge ( 0 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 3 , 4 ) ; addEdge ( 3 , 5 ) ;
dfs ( 0 ) ;
countPairs ( 1 , 3 ) ; countPairs ( 0 , 2 ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int findPermutation ( HashSet < int > arr , int N ) { int pos = arr . Count + 1 ;
if ( pos > N ) return 1 ; int res = 0 ; for ( int i = 1 ; i <= N ; i ++ ) {
if ( ! arr . Contains ( i ) ) {
if ( i % pos == 0 pos % i == 0 ) {
arr . Add ( i ) ;
res += findPermutation ( arr , N ) ;
arr . Remove ( i ) ; } } }
return res ; }
public static void Main ( String [ ] args ) { int N = 5 ; HashSet < int > arr = new HashSet < int > ( ) ; Console . Write ( findPermutation ( arr , N ) ) ; } }
using System ; class GFG {
static void solve ( int [ ] arr , int n , int X , int Y ) {
int diff = Y - X ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] != 1 ) { diff = diff % ( arr [ i ] - 1 ) ; } }
if ( diff == 0 ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 7 , 9 , 10 } ; int n = arr . Length ; int X = 11 , Y = 13 ; solve ( arr , n , X , Y ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int maxN = 100001 ;
static List < int > [ ] adj = new List < int > [ maxN ] ;
static int [ ] height = new int [ maxN ] ;
static int [ ] dist = new int [ maxN ] ;
static void addEdge ( int u , int v ) {
adj [ u ] . Add ( v ) ;
adj [ v ] . Add ( u ) ; }
static void dfs1 ( int cur , int par ) {
foreach ( int u in adj [ cur ] ) { if ( u != par ) {
dfs1 ( u , cur ) ;
height [ cur ] = Math . Max ( height [ cur ] , height [ u ] ) ; } }
height [ cur ] += 1 ; }
static void dfs2 ( int cur , int par ) { int max1 = 0 ; int max2 = 0 ;
foreach ( int u in adj [ cur ] ) { if ( u != par ) {
if ( height [ u ] >= max1 ) { max2 = max1 ; max1 = height [ u ] ; } else if ( height [ u ] > max2 ) { max2 = height [ u ] ; } } } int sum = 0 ; foreach ( int u in adj [ cur ] ) { if ( u != par ) {
sum = ( ( max1 == height [ u ] ) ? max2 : max1 ) ; if ( max1 == height [ u ] ) dist [ u ] = 1 + Math . Max ( 1 + max2 , dist [ cur ] ) ; else dist [ u ] = 1 + Math . Max ( 1 + max1 , dist [ cur ] ) ;
dfs2 ( u , cur ) ; } } }
public static void Main ( String [ ] args ) { int n = 6 ; for ( int i = 0 ; i < adj . Length ; i ++ ) adj [ i ] = new List < int > ( ) ; addEdge ( 1 , 2 ) ; addEdge ( 2 , 3 ) ; addEdge ( 2 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 5 , 6 ) ;
dfs1 ( 1 , 0 ) ;
dfs2 ( 1 , 0 ) ;
for ( int i = 1 ; i <= n ; i ++ ) Console . Write ( ( Math . Max ( dist [ i ] , height [ i ] ) - 1 ) + " ▁ " ) ; } }
using System ; class Middle {
public static int middleOfThree ( int a , int b , int c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && < c ) || ( c < a && < b ) ) return ; else return ; }
public static void Main ( ) { int a = 20 , b = 30 , c = 40 ; Console . WriteLine ( middleOfThree ( a , b , c ) ) ; } }
using System ; public class GFG {
static void selectionSort ( int [ ] arr , int n ) { int i , j , min_idx ;
for ( i = 0 ; i < n - 1 ; i ++ ) {
min_idx = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
int temp = arr [ min_idx ] ; arr [ min_idx ] = arr [ i ] ; arr [ i ] = temp ; } }
static void printArray ( int [ ] arr , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } Console . WriteLine ( ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 64 , 25 , 12 , 22 , 11 } ; int n = arr . Length ;
selectionSort ( arr , n ) ; Console . Write ( " Sorted ▁ array : ▁ STRNEWLINE " ) ;
printArray ( arr , n ) ; } }
using System ; using System . Collections . Generic ; class GFG { static bool checkStr1CanConStr2 ( String str1 , String str2 ) {
int N = str1 . Length ;
int M = str2 . Length ;
HashSet < int > st1 = new HashSet < int > ( ) ;
HashSet < int > st2 = new HashSet < int > ( ) ;
int [ ] hash1 = new int [ 256 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
hash1 [ str1 [ i ] ] ++ ; }
for ( int i = 0 ; i < N ; i ++ ) {
st1 . Add ( str1 [ i ] ) ; }
for ( int i = 0 ; i < M ; i ++ ) {
st2 . Add ( str2 [ i ] ) ; }
if ( st1 . Equals ( st2 ) ) { return false ; }
int [ ] hash2 = new int [ 256 ] ;
for ( int i = 0 ; i < M ; i ++ ) {
hash2 [ str2 [ i ] ] ++ ; }
Array . Sort ( hash1 ) ;
Array . Sort ( hash2 ) ;
for ( int i = 0 ; i < 256 ; i ++ ) {
if ( hash1 [ i ] != hash2 [ i ] ) { return false ; } } return true ; }
public static void Main ( String [ ] args ) { String str1 = " xyyzzlll " ; String str2 = " yllzzxxx " ; if ( checkStr1CanConStr2 ( str1 , str2 ) ) { Console . Write ( " True " ) ; } else { Console . Write ( " False " ) ; } } }
using System ; class GFG {
static void partSort ( int [ ] arr , int N , int a , int b ) {
int l = Math . Min ( a , b ) ; int r = Math . Max ( a , b ) ;
Array . Sort ( arr , l , r ) ;
for ( int i = 0 ; i < N ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
static void Main ( ) { int [ ] arr = { 7 , 8 , 4 , 5 , 2 } ; int a = 1 , b = 4 ; int N = arr . Length ; partSort ( arr , N , a , b ) ; } }
using System ; class GFG { static int INF = int . MaxValue , N = 4 ;
static int minCost ( int [ , ] cost ) {
int [ ] dist = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i , j ] ) dist [ j ] = dist [ i ] + cost [ i , j ] ; return dist [ N - 1 ] ; }
public static void Main ( ) { int [ , ] cost = { { 0 , 15 , 80 , 90 } , { INF , 0 , 40 , 50 } , { INF , INF , 0 , 70 } , { INF , INF , INF , 0 } } ; Console . WriteLine ( " The ▁ Minimum ▁ cost ▁ to " + " ▁ reach ▁ station ▁ " + N + " ▁ is ▁ " + minCost ( cost ) ) ; } }
using System ; class GFG {
static int numOfways ( int n , int k ) { int p = 1 ; if ( k % 2 != 0 ) p = - 1 ; return ( int ) ( Math . Pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
static void Main ( ) { int n = 4 , k = 2 ; Console . Write ( numOfways ( n , k ) ) ; } }
using System ; class GFG {
static char largest_alphabet ( String a , int n ) {
char max = ' A ' ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] > max ) max = a [ i ] ;
return max ; }
static char smallest_alphabet ( String a , int n ) {
char min = ' z ' ;
for ( int i = 0 ; i < n - 1 ; i ++ ) if ( a [ i ] < min ) min = a [ i ] ;
return min ; }
public static void Main ( ) {
String a = " GeEksforGeeks " ;
int size = a . Length ;
Console . Write ( " Largest ▁ and ▁ smallest ▁ alphabet ▁ is ▁ : ▁ " ) ; Console . Write ( largest_alphabet ( a , size ) + " ▁ and ▁ " ) ; Console . Write ( smallest_alphabet ( a , size ) ) ; } }
using System ; public class GFG {
static String maximumPalinUsingKChanges ( String str , int k ) { char [ ] palin = str . ToCharArray ( ) ; String ans = " " ;
int l = 0 ; int r = str . Length - 1 ;
while ( l < r ) {
if ( str [ l ] != str [ r ] ) { palin [ l ] = palin [ r ] = ( char ) Math . Max ( str [ l ] , str [ r ] ) ; k -- ; } l ++ ; r -- ; }
if ( k < 0 ) { return " Not ▁ possible " ; } l = 0 ; r = str . Length - 1 ; while ( l <= r ) {
if ( l == r ) { if ( k > 0 ) { palin [ l ] = '9' ; } }
if ( palin [ l ] < '9' ) {
if ( k >= 2 && palin [ l ] == str [ l ] && palin [ r ] == str [ r ] ) { k -= 2 ; palin [ l ] = palin [ r ] = '9' ; }
else if ( k >= 1 && ( palin [ l ] != str [ l ] palin [ r ] != str [ r ] ) ) { k -- ; palin [ l ] = palin [ r ] = '9' ; } } l ++ ; r -- ; } for ( int i = 0 ; i < palin . Length ; i ++ ) ans += palin [ i ] ; return ans ; }
public static void Main ( ) { String str = "43435" ; int k = 3 ; Console . Write ( maximumPalinUsingKChanges ( str , k ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int countTriplets ( int [ ] A ) {
int cnt = 0 ;
Dictionary < int , int > tuples = new Dictionary < int , int > ( ) ;
foreach ( int a in A )
foreach ( int b in A ) { if ( tuples . ContainsKey ( a & b ) ) tuples [ a & b ] = tuples [ a & b ] + 1 ; else tuples . Add ( a & b , 1 ) ; }
foreach ( int a in A )
foreach ( KeyValuePair < int , int > t in tuples )
if ( ( t . Key & a ) == 0 ) cnt += t . Value ;
return cnt ; }
public static void Main ( String [ ] args ) {
int [ ] A = { 2 , 1 , 3 } ;
Console . Write ( countTriplets ( A ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int min ;
static void parity ( List < int > even , List < int > odd , List < int > v , int i ) {
if ( i == v . Count even . Count == 0 && odd . Count == 0 ) { int count = 0 ; for ( int j = 0 ; j < v . Count - 1 ; j ++ ) { if ( v [ j ] % 2 != v [ j + 1 ] % 2 ) count ++ ; } if ( count < min ) min = count ; return ; }
if ( v [ i ] != - 1 ) parity ( even , odd , v , i + 1 ) ;
else { if ( even . Count != 0 ) { int x = even [ even . Count - 1 ] ; even . RemoveAt ( even . Count - 1 ) ; v [ i ] = x ; parity ( even , odd , v , i + 1 ) ;
even . Add ( x ) ; } if ( odd . Count != 0 ) { int x = odd [ odd . Count - 1 ] ; odd . RemoveAt ( odd . Count - 1 ) ; v [ i ] = x ; parity ( even , odd , v , i + 1 ) ;
odd . Add ( x ) ; } } }
static void minDiffParity ( List < int > v , int n ) {
List < int > even = new List < int > ( ) ;
List < int > odd = new List < int > ( ) ; Dictionary < int , int > m = new Dictionary < int , int > ( ) ; for ( int i = 1 ; i <= n ; i ++ ) { if ( m . ContainsKey ( i ) ) { m [ i ] = 1 ; } else { m . Add ( i , 1 ) ; } } for ( int i = 0 ; i < v . Count ; i ++ ) {
if ( v [ i ] != - 1 ) m . Remove ( v [ i ] ) ; }
foreach ( KeyValuePair < int , int > i in m ) { if ( i . Key % 2 == 0 ) { even . Add ( i . Key ) ; } else { odd . Add ( i . Key ) ; } } min = 1000 ; parity ( even , odd , v , 0 ) ; Console . WriteLine ( min ) ; }
static void Main ( ) { int n = 8 ; List < int > v = new List < int > ( ) ; v . Add ( 2 ) ; v . Add ( 1 ) ; v . Add ( 4 ) ; v . Add ( - 1 ) ; v . Add ( - 1 ) ; v . Add ( 6 ) ; v . Add ( - 1 ) ; v . Add ( 8 ) ; minDiffParity ( v , n ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 100005 ; static List < List < int > > adjacent = new List < List < int > > ( ) ; static bool [ ] visited = new bool [ MAX ] ;
static int startnode , endnode , thirdnode ; static int maxi = - 1 , N ;
static int [ ] parent = new int [ MAX ] ;
static bool [ ] vis = new bool [ MAX ] ;
static void dfs ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent [ u ] . Count ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] ) { temp ++ ; dfs ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; startnode = u ; } } }
static void dfs1 ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent [ u ] . Count ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] ) { temp ++ ; parent [ adjacent [ u ] [ i ] ] = u ; dfs1 ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; endnode = u ; } } }
static void dfs2 ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent [ u ] . Count ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] && ! vis [ adjacent [ u ] [ i ] ] ) { temp ++ ; dfs2 ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; thirdnode = u ; } } }
static void findNodes ( ) {
dfs ( 1 , 0 ) ; for ( int i = 0 ; i <= N ; i ++ ) visited [ i ] = false ; maxi = - 1 ;
dfs1 ( startnode , 0 ) ; for ( int i = 0 ; i <= N ; i ++ ) visited [ i ] = false ;
int x = endnode ; vis [ startnode ] = true ;
while ( x != startnode ) { vis [ x ] = true ; x = parent [ x ] ; } maxi = - 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( vis [ i ] ) dfs2 ( i , 0 ) ; } }
static void Main ( ) { for ( int i = 0 ; i < MAX ; i ++ ) adjacent . Add ( new List < int > ( ) ) ; N = 4 ; adjacent [ 1 ] . Add ( 2 ) ; adjacent [ 2 ] . Add ( 1 ) ; adjacent [ 1 ] . Add ( 3 ) ; adjacent [ 3 ] . Add ( 1 ) ; adjacent [ 1 ] . Add ( 4 ) ; adjacent [ 4 ] . Add ( 1 ) ; findNodes ( ) ; Console . WriteLine ( " ( " + startnode + " , ▁ " + endnode + " , ▁ " + thirdnode + " ) " ) ; } }
using System ; class GFG { static void newvol ( double x ) { Console . WriteLine ( " percentage ▁ increase ▁ in ▁ the " + " ▁ volume ▁ of ▁ the ▁ sphere ▁ is ▁ " + ( Math . Pow ( x , 3 ) / 10000 + 3 * x + ( 3 * Math . Pow ( x , 2 ) ) / 100 ) + " % " ) ; }
public static void Main ( ) { double x = 10 ; newvol ( x ) ; } }
using System ; class GFG {
static void length_of_chord ( double r , double x ) { Console . WriteLine ( " The ▁ length ▁ of ▁ the ▁ chord " + " ▁ of ▁ the ▁ circle ▁ is ▁ " + 2 * r * Math . Sin ( x * ( 3.14 / 180 ) ) ) ; }
public static void Main ( String [ ] args ) { double r = 4 , x = 63 ; length_of_chord ( r , x ) ; } }
using System ; class GFG {
static float area ( float a ) {
if ( a < 0 ) return - 1 ;
float area = ( float ) Math . Sqrt ( a ) / 6 ; return area ; }
public static void Main ( ) { float a = 10 ; Console . WriteLine ( area ( a ) ) ; } }
using System ; class GFG {
static double longestRodInCuboid ( int length , int breadth , int height ) { double result ; int temp ;
temp = length * length + breadth * breadth + height * height ;
result = Math . Sqrt ( temp ) ; return result ; }
public static void Main ( ) { int length = 12 , breadth = 9 , height = 8 ;
Console . WriteLine ( ( int ) longestRodInCuboid ( length , breadth , height ) ) ; } }
using System ; class GFG {
static bool LiesInsieRectangle ( int a , int b , int x , int y ) { if ( x - y - b <= 0 && x - y + b >= 0 && x + y - 2 * a + b <= 0 && x + y - b >= 0 ) return true ; return false ; }
public static void Main ( ) { int a = 7 , b = 2 , x = 4 , y = 5 ; if ( LiesInsieRectangle ( a , b , x , y ) ) Console . Write ( " Given ▁ point ▁ lies ▁ " + " inside ▁ the ▁ rectangle " ) ; else Console . Write ( " Given ▁ point ▁ does ▁ not ▁ " + " lie ▁ on ▁ the ▁ rectangle " ) ; } }
using System ; class GFG {
static int maxvolume ( int s ) { int maxvalue = 0 ;
for ( int i = 1 ; i <= s - 2 ; i ++ ) {
for ( int j = 1 ; j <= s - 1 ; j ++ ) {
int k = s - i - j ;
maxvalue = Math . Max ( maxvalue , i * j * k ) ; } } return maxvalue ; }
public static void Main ( ) { int s = 8 ; Console . WriteLine ( maxvolume ( s ) ) ; } }
using System ; class GFG {
static int maxvolume ( int s ) {
int length = s / 3 ; s -= length ;
int breadth = s / 2 ;
int height = s - breadth ; return length * breadth * height ; }
public static void Main ( ) { int s = 8 ; Console . WriteLine ( maxvolume ( s ) ) ; } }
using System ; class GFG {
public static double hexagonArea ( double s ) { return ( ( 3 * Math . Sqrt ( 3 ) * ( s * s ) ) / 2 ) ; }
public static void Main ( ) {
double s = 4 ; Console . WriteLine ( " Area : ▁ " + hexagonArea ( s ) ) ; } }
using System ; public class GFG {
static int maxSquare ( int b , int m ) {
return ( b / m - 1 ) * ( b / m ) / 2 ; }
public static void Main ( ) { int b = 10 , m = 2 ; Console . WriteLine ( maxSquare ( b , m ) ) ; } }
using System ; class GFG {
static void findRightAngle ( double A , double H ) {
double D = Math . Pow ( H , 4 ) - 16 * A * A ; if ( D >= 0 ) {
double root1 = ( H * H + Math . Sqrt ( D ) ) / 2 ; double root2 = ( H * H - Math . Sqrt ( D ) ) / 2 ; double a = Math . Sqrt ( root1 ) ; double b = Math . Sqrt ( root2 ) ; if ( b >= a ) Console . WriteLine ( a + " ▁ " + b + " ▁ " + H ) ; else Console . WriteLine ( b + " ▁ " + a + " ▁ " + H ) ; } else Console . ( " - 1" ) ; }
public static void Main ( ) { findRightAngle ( 6 , 5 ) ; } }
using System ; class GFG { public static int numberOfSquares ( int _base ) {
_base = ( _base - 2 ) ;
_base = _base / 2 ; return _base * ( _base + 1 ) / 2 ; }
public static void Main ( ) { int _base = 8 ; Console . WriteLine ( numberOfSquares ( _base ) ) ; } }
using System ; class GFG {
static void performQuery ( int [ ] arr , int [ , ] Q ) {
for ( int i = 0 ; i < Q . Length ; i ++ ) {
int or = 0 ;
int x = Q [ i , 0 ] ; arr [ x - 1 ] = Q [ i , 1 ] ;
for ( int j = 0 ; j < arr . Length ; j ++ ) { or = or | arr [ j ] ; }
Console . Write ( or + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 } ; int [ , ] Q = { { 1 , 4 } , { 3 , 0 } } ; performQuery ( arr , Q ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int smallest ( int k , int d ) { int cnt = 1 ; int m = d % k ;
int [ ] v = new int [ k ] ; for ( int i = 0 ; i < k ; i ++ ) v [ i ] = 0 ; v [ m ] = 1 ;
while ( true ) { if ( m == 0 ) return cnt ; m = ( ( ( m * ( 10 % k ) ) % k ) + ( d % k ) ) % k ;
if ( v [ m ] == 1 ) return - 1 ; v [ m ] = 1 ; cnt ++ ; } }
public static void Main ( ) { int d = 1 ; int k = 41 ; Console . Write ( smallest ( k , d ) ) ; } }
using System ; class GFG {
static int fib ( int n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
static int findVertices ( int n ) {
return fib ( n + 2 ) ; }
static void Main ( ) { int n = 3 ; Console . Write ( findVertices ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int GCD ( int a , int b ) { return b == 0 ? a : GCD ( b , a % b ) ; }
static void checkCommonDivisor ( int [ ] arr , int N , int X ) {
int G = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { G = GCD ( G , arr [ i ] ) ; } int copy_G = G ; for ( int divisor = 2 ; divisor <= X ; divisor ++ ) {
while ( G % divisor == 0 ) {
G = G / divisor ; } }
if ( G <= X ) { Console . WriteLine ( " Yes " ) ;
for ( int i = 0 ; i < N ; i ++ ) Console . Write ( arr [ i ] / copy_G + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ; }
else Console . ( " No " ) ; }
public static void Main ( ) {
int [ ] arr = { 6 , 15 , 6 } ; int X = 6 ;
int N = arr . Length ; checkCommonDivisor ( arr , N , X ) ; } }
int row = 0 , col = 0 ; int boundary = size - 1 ; int sizeLeft = size - 1 ; int flag = 1 ;
char move = ' r ' ;
int [ , ] matrix = new int [ size , size ] ; for ( int i = 1 ; i < size * size + 1 ; i ++ ) {
matrix [ row , col ] = i ;
switch ( move ) {
' r ' : col += 1 ; break ;
' l ' : col -= 1 ; break ;
' u ' : row -= 1 ; break ;
' d ' : row += 1 ; break ; }
if ( i == boundary ) {
boundary += sizeLeft ;
if ( flag != 2 ) { flag = 2 ; } else { flag = 1 ; sizeLeft -= 1 ; }
switch ( move ) {
' r ' : move = ' d ' ; break ;
' d ' : move = ' l ' ; break ;
' l ' : move = ' u ' ; break ;
' u ' : move = ' r ' ; break ; } } }
for ( row = 0 ; row < size ; row ++ ) { for ( col = 0 ; col < size ; col ++ ) { int n = matrix [ row , col ] ; Console . Write ( ( n < 10 ) ? ( n + " ▁ " ) : ( n + " ▁ " ) ) ; } Console . WriteLine ( ) ; } }
public static void Main ( String [ ] args ) {
int size = 5 ;
printSpiral ( size ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node next ; public Node prev ; }
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
if ( head == null ) Console . WriteLine ( " Doubly ▁ Linked ▁ list ▁ empty " ) ; while ( head != null ) { Console . Write ( head . data + " ▁ " ) ; head = head . next ; } }
public static void Main ( String [ ] args ) { Node head = null ;
head = push ( head , 1 ) ; head = push ( head , 4 ) ; head = push ( head , 6 ) ; head = push ( head , 10 ) ; head = push ( head , 12 ) ; head = push ( head , 7 ) ; head = push ( head , 5 ) ; head = push ( head , 2 ) ; Console . WriteLine ( " Original ▁ Doubly ▁ linked ▁ list : n " ) ; printList ( head ) ;
head = sort ( head ) ; Console . WriteLine ( " STRNEWLINE Doubly ▁ linked ▁ list ▁ after ▁ sorting : n " ) ; printList ( head ) ; } }
using System ; class GfG {
public class Node { public char data ; public Node next ; }
static Node newNode ( char key ) { Node temp = new Node ( ) ; temp . data = key ; temp . next = null ; return temp ; }
static void printlist ( Node head ) { if ( head == null ) { Console . WriteLine ( " Empty ▁ List " ) ; return ; } while ( head != null ) { Console . Write ( head . data + " ▁ " ) ; if ( head . next != null ) Console . Write ( " - > ▁ " ) ; head = head . next ; } Console . WriteLine ( ) ; }
static bool isVowel ( char x ) { return ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) ; }
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
public static void Main ( String [ ] args ) { Node head = newNode ( ' a ' ) ; head . next = newNode ( ' b ' ) ; head . next . next = newNode ( ' c ' ) ; head . next . next . next = newNode ( ' e ' ) ; head . next . next . next . next = newNode ( ' d ' ) ; head . next . next . next . next . next = newNode ( ' o ' ) ; head . next . next . next . next . next . next = newNode ( ' x ' ) ; head . next . next . next . next . next . next . next = newNode ( ' i ' ) ; Console . WriteLine ( " Linked ▁ list ▁ before ▁ : ▁ " ) ; printlist ( head ) ; head = arrange ( head ) ; Console . WriteLine ( " Linked ▁ list ▁ after ▁ : " ) ; printlist ( head ) ; } }
using System ; using System . Collections . Generic ; class GfG {
public class Node { public int data ; public Node left , right ; }
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
Node root = newNode ( 4 ) ; root . left = newNode ( 2 ) ; root . right = newNode ( 7 ) ; root . left . left = newNode ( 1 ) ; root . left . right = newNode ( 3 ) ; root . right . left = newNode ( 6 ) ; root . right . right = newNode ( 10 ) ; Console . Write ( " Finding ▁ K - th ▁ largest ▁ Node ▁ in ▁ BST ▁ : ▁ " + KthLargestUsingMorrisTraversal ( root , 2 ) . data ) ; } }
using System ; class GFG {
static void sortByRow ( int [ , ] mat , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n - 1 ; j ++ ) { if ( mat [ i , j ] > mat [ i , j + 1 ] ) { var temp = mat [ i , j ] ; mat [ i , j ] = mat [ i , j + 1 ] ; mat [ i , j + 1 ] = temp ; } } } }
static void transpose ( int [ , ] mat , int n ) { for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) {
var temp = mat [ i , j ] ; mat [ i , j ] = mat [ j , i ] ; mat [ j , i ] = temp ; } }
static void sortMatRowAndColWise ( int [ , ] mat , int n ) {
sortByRow ( mat , n ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n ) ;
transpose ( mat , n ) ; }
static void printMat ( int [ , ] mat , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) Console . Write ( mat [ i , j ] + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ; } }
public static void Main ( ) { int [ , ] mat = { { 4 , 1 , 3 } , { 9 , 6 , 8 } , { 5 , 2 , 7 } } ; int n = 3 ; Console . Write ( " Original ▁ Matrix : STRNEWLINE " ) ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; Console . Write ( " STRNEWLINE Matrix ▁ After ▁ Sorting : STRNEWLINE " ) ; printMat ( mat , n ) ; } }
using System ; class GFG {
public static void doublyEven ( int n ) { int [ , ] arr = new int [ n , n ] ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = 0 ; j < n ; j ++ ) { arr [ i , j ] = ( n * i ) + j + 1 ; } }
for ( i = 0 ; i < n / 4 ; i ++ ) { for ( j = 0 ; j < n / 4 ; j ++ ) { arr [ i , j ] = ( n * n + 1 ) - arr [ i , j ] ; } }
for ( i = 0 ; i < n / 4 ; i ++ ) { for ( j = 3 * ( n / 4 ) ; j < n ; j ++ ) { arr [ i , j ] = ( n * n + 1 ) - arr [ i , j ] ; } }
for ( i = 3 * n / 4 ; i < n ; i ++ ) { for ( j = 0 ; j < n / 4 ; j ++ ) { arr [ i , j ] = ( n * n + 1 ) - arr [ i , j ] ; } }
for ( i = 3 * n / 4 ; i < n ; i ++ ) { for ( j = 3 * n / 4 ; j < n ; j ++ ) { arr [ i , j ] = ( n * n + 1 ) - arr [ i , j ] ; } }
for ( i = n / 4 ; i < 3 * n / 4 ; i ++ ) { for ( j = n / 4 ; j < 3 * n / 4 ; j ++ ) { arr [ i , j ] = ( n * n + 1 ) - arr [ i , j ] ; } }
for ( i = 0 ; i < n ; i ++ ) { for ( j = 0 ; j < n ; j ++ ) { Console . Write ( arr [ i , j ] + " ▁ " + " ▁ " ) ; } Console . WriteLine ( ) ; } }
public static void Main ( string [ ] args ) { int n = 8 ;
doublyEven ( n ) ; } }
using System ; class GFG {
static int cola = 2 , rowa = 3 ; static int colb = 3 , rowb = 2 ;
static void Kroneckerproduct ( int [ , ] A , int [ , ] B ) { int [ , ] C = new int [ rowa * rowb , cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 , j + k + 1 ] = A [ i , j ] * B [ k , l ] ; Console . Write ( C [ i + l + 1 , j + k + 1 ] + " ▁ " ) ; } } Console . WriteLine ( ) ; } } }
public static void Main ( ) { int [ , ] A = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } ; int [ , ] B = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; } }
using System ; class Lower_triangular { int N = 4 ;
bool isLowerTriangularMatrix ( int [ , ] mat ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( mat [ i , j ] != 0 ) return false ; return true ; }
public static void Main ( ) { Lower_triangular ob = new Lower_triangular ( ) ; int [ , ] mat = { { 1 , 0 , 0 , 0 } , { 1 , 4 , 0 , 0 } , { 4 , 6 , 2 , 0 } , { 0 , 4 , 7 , 6 } } ;
if ( ob . isLowerTriangularMatrix ( mat ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; public class GfG { private static int N = 4 ;
public static bool isUpperTriangularMatrix ( int [ , ] mat ) { for ( int i = 1 ; i < N ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( mat [ i , j ] != 0 ) return false ; return true ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 3 , 5 , 3 } , { 0 , 4 , 6 , 2 } , { 0 , 0 , 2 , 5 } , { 0 , 0 , 0 , 6 } } ; if ( isUpperTriangularMatrix ( mat ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static int m = 3 ;
static int n = 2 ;
static long countSets ( int [ , ] a ) {
long res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < m ; j ++ ) { if ( a [ i , j ] == 1 ) u ++ ; else v ++ ; } res += ( long ) ( Math . Pow ( 2 , u ) - 1 + Math . Pow ( 2 , v ) ) - 1 ; }
for ( int i = 0 ; i < m ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( a [ j , i ] == 1 ) u ++ ; else v ++ ; } res += ( long ) ( Math . Pow ( 2 , u ) - 1 + Math . Pow ( 2 , v ) ) - 1 ; }
return res - ( n * m ) ; }
public static void Main ( ) { int [ , ] a = { { 1 , 0 , 1 } , { 0 , 1 , 0 } } ; Console . WriteLine ( countSets ( a ) ) ; } }
using System ; class GFG { static int MAX = 100 ;
static void transpose ( int [ , ] mat , int [ , ] tr , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) tr [ i , j ] = mat [ j , i ] ; }
static bool isSymmetric ( int [ , ] mat , int N ) { int [ , ] tr = new int [ N , MAX ] ; transpose ( mat , tr , N ) ; for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i , j ] != tr [ i , j ] ) return false ; return true ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static bool isSymmetric ( int [ , ] mat , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i , j ] != mat [ j , i ] ) return false ; return true ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; class GFG {
static int findNormal ( int [ , ] mat , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) sum += mat [ i , j ] * mat [ i , j ] ; return ( int ) Math . Sqrt ( sum ) ; }
static int findTrace ( int [ , ] mat , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += mat [ i , i ] ; return sum ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; Console . Write ( " Trace ▁ of ▁ Matrix ▁ = ▁ " + findTrace ( mat , 5 ) + " STRNEWLINE " ) ; Console . Write ( " Normal ▁ of ▁ Matrix ▁ = ▁ " + findNormal ( mat , 5 ) ) ; } }
using System ; public class GFG {
static int maxDet ( int n ) { return ( 2 * n * n * n ) ; }
void resMatrix ( int n ) { for ( int i = 0 ; i < 3 ; i ++ ) { for ( int j = 0 ; j < 3 ; j ++ ) {
if ( i == 0 && j == 2 ) Console . Write ( "0 ▁ " ) ; else if ( i == 1 && j == 0 ) Console . Write ( "0 ▁ " ) ; else if ( i == 2 && j == 1 ) Console . Write ( "0 ▁ " ) ;
else Console . ( n + " " ) ; } Console . WriteLine ( " " ) ; } }
static public void Main ( String [ ] args ) { int n = 15 ; GFG geeks = new GFG ( ) ; Console . WriteLine ( " Maximum ▁ Determinant ▁ = ▁ " + maxDet ( n ) ) ; Console . WriteLine ( " Resultant ▁ Matrix ▁ : " ) ; geeks . resMatrix ( n ) ; } }
using System ; class GFG { static int countNegative ( int [ , ] M , int n , int m ) { int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { if ( M [ i , j ] < 0 ) count += 1 ;
else break ; } } return count ; }
public static void Main ( ) { int [ , ] M = { { - 3 , - 2 , - 1 , 1 } , { - 2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; Console . WriteLine ( countNegative ( M , 3 , 4 ) ) ; } }
using System ; class GFG {
static int countNegative ( int [ , ] M , int n , int m ) {
int count = 0 ;
int i = 0 ; int j = m - 1 ;
while ( j >= 0 && i < n ) { if ( M [ i , j ] < 0 ) {
count += j + 1 ;
i += 1 ; }
else j -= 1 ; } return count ; }
public static void Main ( ) { int [ , ] M = { { - 3 , - 2 , - 1 , 1 } , { - 2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; Console . WriteLine ( countNegative ( M , 3 , 4 ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int getLastNegativeIndex ( int [ ] array , int start , int end ) {
if ( start == end ) { return start ; }
int mid = start + ( end - start ) / 2 ;
if ( array [ mid ] < 0 ) {
if ( mid + 1 < array . GetLength ( 0 ) && array [ mid + 1 ] >= 0 ) { return mid ; }
return getLastNegativeIndex ( array , mid + 1 , end ) ; } else {
return getLastNegativeIndex ( array , start , mid - 1 ) ; } }
static int countNegative ( int [ , ] M , int n , int m ) {
int count = 0 ;
int nextEnd = m - 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( M [ i , 0 ] >= 0 ) { break ; }
nextEnd = getLastNegativeIndex ( GetRow ( M , i ) , 0 , nextEnd ) ; count += nextEnd + 1 ; } return count ; } public static int [ ] GetRow ( int [ , ] matrix , int row ) { var rowLength = matrix . GetLength ( 1 ) ; var rowVector = new int [ rowLength ] ; for ( var i = 0 ; i < rowLength ; i ++ ) rowVector [ i ] = matrix [ row , i ] ; return rowVector ; }
public static void Main ( String [ ] args ) { int [ , ] M = { { - 3 , - 2 , - 1 , 1 } , { - 2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; int r = M . GetLength ( 0 ) ; int c = M . GetLength ( 1 ) ; Console . WriteLine ( countNegative ( M , r , c ) ) ; } }
using System ; class GFG {
static int findMaxValue ( int N , int [ , ] mat ) {
int maxValue = int . MinValue ;
for ( int a = 0 ; a < N - 1 ; a ++ ) for ( int b = 0 ; b < N - 1 ; b ++ ) for ( int d = a + 1 ; d < N ; d ++ ) for ( int e = b + 1 ; e < N ; e ++ ) if ( maxValue < ( mat [ d , e ] - mat [ a , b ] ) ) maxValue = mat [ d , e ] - mat [ a , b ] ; return maxValue ; }
public static void Main ( ) { int N = 5 ; int [ , ] mat = { { 1 , 2 , - 1 , - 4 , - 20 } , { - 8 , - 3 , 4 , 2 , 1 } , { 3 , 8 , 6 , 1 , 3 } , { - 4 , - 1 , 1 , 7 , - 6 } , { 0 , - 4 , 10 , - 5 , 1 } } ; Console . Write ( " Maximum ▁ Value ▁ is ▁ " + findMaxValue ( N , mat ) ) ; } }
using System ; class GFG {
static int findMaxValue ( int N , int [ , ] mat ) {
int maxValue = int . MinValue ;
int [ , ] maxArr = new int [ N , N ] ;
maxArr [ N - 1 , N - 1 ] = mat [ N - 1 , N - 1 ] ;
int maxv = mat [ N - 1 , N - 1 ] ; for ( int j = N - 2 ; j >= 0 ; j -- ) { if ( mat [ N - 1 , j ] > maxv ) maxv = mat [ N - 1 , j ] ; maxArr [ N - 1 , j ] = maxv ; }
maxv = mat [ N - 1 , N - 1 ] ; for ( int i = N - 2 ; i >= 0 ; i -- ) { if ( mat [ i , N - 1 ] > maxv ) maxv = mat [ i , N - 1 ] ; maxArr [ i , N - 1 ] = maxv ; }
for ( int i = N - 2 ; i >= 0 ; i -- ) { for ( int j = N - 2 ; j >= 0 ; j -- ) {
if ( maxArr [ i + 1 , j + 1 ] - mat [ i , j ] > maxValue ) maxValue = maxArr [ i + 1 , j + 1 ] - mat [ i , j ] ;
maxArr [ i , j ] = Math . Max ( mat [ i , j ] , Math . Max ( maxArr [ i , j + 1 ] , maxArr [ i + 1 , j ] ) ) ; } } return maxValue ; }
public static void Main ( ) { int N = 5 ; int [ , ] mat = { { 1 , 2 , - 1 , - 4 , - 20 } , { - 8 , - 3 , 4 , 2 , 1 } , { 3 , 8 , 6 , 1 , 3 } , { - 4 , - 1 , 1 , 7 , - 6 } , { 0 , - 4 , 10 , - 5 , 1 } } ; Console . Write ( " Maximum ▁ Value ▁ is ▁ " + findMaxValue ( N , mat ) ) ; } }
using System ; class GFG { static int INF = int . MaxValue ; static int N = 4 ;
static void youngify ( int [ , ] mat , int i , int j ) {
int downVal = ( i + 1 < N ) ? mat [ i + 1 , j ] : INF ; int rightVal = ( j + 1 < N ) ? mat [ i , j + 1 ] : INF ;
if ( downVal == INF && rightVal == INF ) { return ; }
if ( downVal < rightVal ) { mat [ i , j ] = downVal ; mat [ i + 1 , j ] = INF ; youngify ( mat , i + 1 , j ) ; } else { mat [ i , j ] = rightVal ; mat [ i , j + 1 ] = INF ; youngify ( mat , i , j + 1 ) ; } }
static int extractMin ( int [ , ] mat ) { int ret = mat [ 0 , 0 ] ; mat [ 0 , 0 ] = INF ; youngify ( mat , 0 , 0 ) ; return ret ; }
static void printSorted ( int [ , ] mat ) { Console . WriteLine ( " Elements ▁ of ▁ matrix ▁ in ▁ sorted ▁ order ▁ n " ) ; for ( int i = 0 ; i < N * N ; i ++ ) { Console . Write ( extractMin ( mat ) + " ▁ " ) ; } }
static public void Main ( ) { int [ , ] mat = { { 10 , 20 , 30 , 40 } , { 15 , 25 , 35 , 45 } , { 27 , 29 , 37 , 48 } , { 32 , 33 , 39 , 50 } } ; printSorted ( mat ) ; } }
using System ; class GFG {
static int n = 5 ;
static void printSumSimple ( int [ , ] mat , int k ) {
if ( k > n ) return ;
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
for ( int j = 0 ; j < n - k + 1 ; j ++ ) {
int sum = 0 ; for ( int p = i ; p < k + i ; p ++ ) for ( int q = j ; q < k + j ; q ++ ) sum += mat [ p , q ] ; Console . Write ( sum + " ▁ " ) ; }
Console . WriteLine ( ) ; } }
public static void Main ( ) { int [ , ] mat = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } } ; int k = 3 ; printSumSimple ( mat , k ) ; } }
using System ; class GFG {
static int n = 5 ;
static void printSumTricky ( int [ , ] mat , int k ) {
if ( k > n ) return ;
int [ , ] stripSum = new int [ n , n ] ;
for ( int j = 0 ; j < n ; j ++ ) {
int sum = 0 ; for ( int i = 0 ; i < k ; i ++ ) sum += mat [ i , j ] ; stripSum [ 0 , j ] = sum ;
for ( int i = 1 ; i < n - k + 1 ; i ++ ) { sum += ( mat [ i + k - 1 , j ] - mat [ i - 1 , j ] ) ; stripSum [ i , j ] = sum ; } }
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
int sum = 0 ; for ( int j = 0 ; j < k ; j ++ ) sum += stripSum [ i , j ] ; Console . Write ( sum + " ▁ " ) ;
for ( int j = 1 ; j < n - k + 1 ; j ++ ) { sum += ( stripSum [ i , j + k - 1 ] - stripSum [ i , j - 1 ] ) ; Console . Write ( sum + " ▁ " ) ; } Console . WriteLine ( ) ; } }
public static void Main ( ) { int [ , ] mat = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; int k = 3 ; printSumTricky ( mat , k ) ; } }
using System ; class GFG { static int M = 3 ; static int N = 4 ;
static void transpose ( int [ , ] A , int [ , ] B ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < M ; j ++ ) B [ i , j ] = A [ j , i ] ; }
public static void Main ( ) { int [ , ] A = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } } ; int [ , ] B = new int [ N , M ] ; transpose ( A , B ) ; Console . WriteLine ( " Result ▁ matrix ▁ is ▁ STRNEWLINE " ) ; for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < M ; j ++ ) Console . Write ( B [ i , j ] + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ; } } }
using System ; class GFG { static int N = 4 ;
static void transpose ( int [ , ] A ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) { int temp = A [ i , j ] ; A [ i , j ] = A [ j , i ] ; A [ j , i ] = temp ; } }
public static void Main ( ) { int [ , ] A = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; transpose ( A ) ; Console . WriteLine ( " Modified ▁ matrix ▁ is ▁ " ) ; for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) Console . Write ( A [ i , j ] + " ▁ " ) ; Console . WriteLine ( ) ; } } }
using System ; public class GFG { public const int R = 3 ; public const int C = 3 ;
public static int pathCountRec ( int [ ] [ ] mat , int m , int n , int k ) {
if ( m < 0 n < 0 ) { return 0 ; } if ( m == 0 && n == 0 && ( k == mat [ m ] [ n ] ) ) { return 1 ; }
return pathCountRec ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountRec ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ; }
public static int pathCount ( int [ ] [ ] mat , int k ) { return pathCountRec ( mat , R - 1 , C - 1 , k ) ; }
public static void Main ( string [ ] args ) { int k = 12 ; int [ ] [ ] mat = new int [ ] [ ] { new int [ ] { 1 , 2 , 3 } , new int [ ] { 4 , 6 , 5 } , new int [ ] { 3 , 2 , 1 } } ; Console . WriteLine ( pathCount ( mat , k ) ) ; } }
using System ; class GFG { static readonly int R = 3 ; static readonly int C = 3 ; static readonly int MAX_K = 100 ; static int [ , , ] dp = new int [ R , C , MAX_K ] ; static int pathCountDPRecDP ( int [ , ] mat , int m , int n , int k ) {
if ( m < 0 n < 0 ) return 0 ; if ( m == 0 && n == 0 ) return ( k == mat [ m , n ] ? 1 : 0 ) ;
if ( dp [ m , n , k ] != - 1 ) return dp [ m , n , k ] ;
dp [ m , n , k ] = pathCountDPRecDP ( mat , m - 1 , n , k - mat [ m , n ] ) + pathCountDPRecDP ( mat , m , n - 1 , k - mat [ m , n ] ) ; return dp [ m , n , k ] ; }
static int pathCountDP ( int [ , ] mat , int k ) { for ( int i = 0 ; i < R ; i ++ ) for ( int j = 0 ; j < C ; j ++ ) for ( int l = 0 ; l < MAX_K ; l ++ ) dp [ i , j , l ] = - 1 ; return pathCountDPRecDP ( mat , R - 1 , C - 1 , k ) ; }
public static void Main ( String [ ] args ) { int k = 12 ; int [ , ] mat = { { 1 , 2 , 3 } , { 4 , 6 , 5 } , { 3 , 2 , 1 } } ; Console . WriteLine ( pathCountDP ( mat , k ) ) ; } }
using System ; class GFG { static int SIZE = 10 ;
static void sortMat ( int [ , ] mat , int n ) {
int [ ] temp = new int [ n * n ] ; int k = 0 ;
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) temp [ k ++ ] = mat [ i , j ] ;
Array . Sort ( temp ) ;
k = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) mat [ i , j ] = temp [ k ++ ] ; }
static void printMat ( int [ , ] mat , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) Console . Write ( mat [ i , j ] + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( ) { int [ , ] mat = { { 5 , 4 , 7 } , { 1 , 3 , 8 } , { 2 , 9 , 6 } } ; int n = 3 ; Console . WriteLine ( " Original ▁ Matrix : " ) ; printMat ( mat , n ) ; sortMat ( mat , n ) ; Console . WriteLine ( " Matrix ▁ After ▁ Sorting : " ) ; printMat ( mat , n ) ; } }
using System ; class GFG {
static void sort ( int [ ] arr ) { int n = arr . Length ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min_idx = i ; for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
int temp = arr [ min_idx ] ; arr [ min_idx ] = arr [ i ] ; arr [ i ] = temp ; } }
static void printArray ( int [ ] arr ) { int n = arr . Length ; for ( int i = 0 ; i < n ; ++ i ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( ) { int [ ] arr = { 64 , 25 , 12 , 22 , 11 } ; sort ( arr ) ; Console . WriteLine ( " Sorted ▁ array " ) ; printArray ( arr ) ; } }
using System ; class GFG {
static void bubbleSort ( int [ ] arr , int n ) { int i , j , temp ; bool swapped ; for ( i = 0 ; i < n - 1 ; i ++ ) { swapped = false ; for ( j = 0 ; j < n - i - 1 ; j ++ ) { if ( arr [ j ] > arr [ j + 1 ] ) {
temp = arr [ j ] ; arr [ j ] = arr [ j + 1 ] ; arr [ j + 1 ] = temp ; swapped = true ; } }
if ( swapped == false ) break ; } }
static void printArray ( int [ ] arr , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( ) { int [ ] arr = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; int n = arr . Length ; bubbleSort ( arr , n ) ; Console . WriteLine ( " Sorted ▁ array " ) ; printArray ( arr , n ) ; } }
using System ; class GFG {
static int findCrossOver ( int [ ] arr , int low , int high , int x ) {
if ( arr [ high ] <= x ) return high ;
if ( arr [ low ] > x ) return low ;
int mid = ( low + high ) / 2 ;
if ( arr [ mid ] <= x && arr [ mid + 1 ] > x ) return mid ;
if ( arr [ mid ] < x ) return findCrossOver ( arr , mid + 1 , high , x ) ; return findCrossOver ( arr , low , mid - 1 , x ) ; }
static void printKclosest ( int [ ] arr , int x , int k , int n ) {
int l = findCrossOver ( arr , 0 , n - 1 , x ) ;
int r = l + 1 ;
int count = 0 ;
if ( arr [ l ] == x ) l -- ;
while ( l >= 0 && r < n && count < k ) { if ( x - arr [ l ] < arr [ r ] - x ) Console . Write ( arr [ l -- ] + " ▁ " ) ; else Console . Write ( arr [ r ++ ] + " ▁ " ) ; count ++ ; }
while ( count < k && l >= 0 ) { Console . Write ( arr [ l -- ] + " ▁ " ) ; count ++ ; }
while ( count < k && r < n ) { Console . Write ( arr [ r ++ ] + " ▁ " ) ; count ++ ; } }
public static void Main ( ) { int [ ] arr = { 12 , 16 , 22 , 30 , 35 , 39 , 42 , 45 , 48 , 50 , 53 , 55 , 56 } ; int n = arr . Length ; int x = 35 ; printKclosest ( arr , x , 4 , n ) ; } }
using System ; class GFG {
static int count ( int [ ] S , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int m = arr . Length ; Console . Write ( count ( arr , m , 4 ) ) ; } }
using System ; class GFG { static int count ( int [ ] S , int m , int n ) {
int [ ] table = new int [ n + 1 ] ;
table [ 0 ] = 1 ;
for ( int i = 0 ; i < m ; i ++ ) for ( int j = S [ i ] ; j <= n ; j ++ ) table [ j ] += table [ j - S [ i ] ] ; return table [ n ] ; }
using System ; class GFG { static int [ , ] dp = new int [ 100 , 100 ] ;
static int matrixChainMemoised ( int [ ] p , int i , int j ) { if ( i == j ) { return 0 ; } if ( dp [ i , j ] != - 1 ) { return dp [ i , j ] ; } dp [ i , j ] = Int32 . MaxValue ; for ( int k = i ; k < j ; k ++ ) { dp [ i , j ] = Math . Min ( dp [ i , j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i , j ] ; } static int MatrixChainOrder ( int [ ] p , int n ) { int i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 } ; int n = arr . Length ; for ( int i = 0 ; i < 100 ; i ++ ) { for ( int j = 0 ; j < 100 ; j ++ ) { dp [ i , j ] = - 1 ; } } Console . WriteLine ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , n ) ) ; } }
using System ; class GFG {
static int MatrixChainOrder ( int [ ] p , int n ) {
int [ , ] m = new int [ n , n ] ; int i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i , i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; if ( j == n ) continue ; m [ i , j ] = int . MaxValue ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i , k ] + m [ k + 1 , j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i , j ] ) m [ i , j ] = q ; } } } return m [ 1 , n - 1 ] ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 2 , 3 , 4 } ; int size = arr . Length ; Console . Write ( " Minimum ▁ number ▁ of ▁ " + " multiplications ▁ is ▁ " + MatrixChainOrder ( arr , size ) ) ; } }
using System ; class GFG {
static int cutRod ( int [ ] price , int n ) { if ( n <= 0 ) return 0 ; int max_val = int . MinValue ;
for ( int i = 0 ; i < n ; i ++ ) max_val = Math . Max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) ; return max_val ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . Length ; Console . WriteLine ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
using System ; class GFG {
static int cutRod ( int [ ] price , int n ) { int [ ] val = new int [ n + 1 ] ; val [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { int max_val = int . MinValue ; for ( int j = 0 ; j < i ; j ++ ) max_val = Math . Max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . Length ; Console . WriteLine ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
using System ; class GFG {
static int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; return - 1 ; }
public static void Main ( ) { Console . WriteLine ( multiply ( 5 , - 11 ) ) ; } }
using System ; namespace prime { public class GFG { public static void SieveOfEratosthenes ( int n ) {
bool [ ] prime = new bool [ n + 1 ] ; for ( int i = 0 ; i < n ; i ++ ) prime [ i ] = true ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( int i = 2 ; i <= n ; i ++ ) { if ( prime [ i ] == true ) Console . Write ( i + " ▁ " ) ; } }
public static void Main ( ) { int n = 30 ; Console . WriteLine ( " Following ▁ are ▁ the ▁ prime ▁ numbers " ) ; Console . WriteLine ( " smaller ▁ than ▁ or ▁ equal ▁ to ▁ " + n ) ; SieveOfEratosthenes ( n ) ; } } }
using System ; class GFG {
static int binomialCoeff ( int n , int k ) { int res = 1 ; if ( k > n - k ) k = n - k ; for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static void printPascal ( int n ) {
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) Console . Write ( binomialCoeff ( line , i ) + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( ) { int n = 7 ; printPascal ( n ) ; } }
using System ; class GFG {
public static void printPascal ( int n ) {
int [ , ] arr = new int [ n , n ] ;
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) {
if ( line == i i == 0 ) arr [ line , i ] = 1 ;
else arr [ line , i ] = arr [ line - 1 , i - 1 ] + arr [ line - 1 , i ] ; Console . Write ( arr [ line , i ] ) ; } Console . WriteLine ( " " ) ; } }
public static void Main ( ) { int n = 5 ; printPascal ( n ) ; } }
using System ; class GFG { public static void printPascal ( int n ) { for ( int line = 1 ; line <= n ; line ++ ) {
int C = 1 ; for ( int i = 1 ; i <= line ; i ++ ) {
Console . Write ( C + " ▁ " ) ; C = C * ( line - i ) / i ; } Console . Write ( " STRNEWLINE " ) ; } }
public static void Main ( ) { int n = 5 ; printPascal ( n ) ; } }
using System ; class GFG { static int Add ( int x , int y ) {
while ( y != 0 ) {
int carry = x & y ;
x = x ^ y ;
y = carry << 1 ; } return x ; }
public static void Main ( ) { Console . WriteLine ( Add ( 15 , 32 ) ) ; } }
using System ; class GFG {
static uint getModulo ( uint n , uint d ) { return ( n & ( d - 1 ) ) ; }
static public void Main ( ) { uint n = 6 ;
uint d = 4 ; Console . WriteLine ( n + " ▁ moduo ▁ " + d + " ▁ is ▁ " + getModulo ( n , d ) ) ; } }
using System ; class GFG {
static int countSetBits ( int n ) { int count = 0 ; while ( n > 0 ) { count += n & 1 ; n >>= 1 ; } return count ; }
public static void Main ( ) { int i = 9 ; Console . Write ( countSetBits ( i ) ) ; } }
using System ; class GFG {
public static int countSetBits ( int n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
static public void Main ( ) {
int n = 9 ;
Console . WriteLine ( countSetBits ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int [ ] BitsSetTable256 = new int [ 256 ] ;
public static void initialize ( ) {
BitsSetTable256 [ 0 ] = 0 ; for ( int i = 0 ; i < 256 ; i ++ ) { BitsSetTable256 [ i ] = ( i & 1 ) + BitsSetTable256 [ i / 2 ] ; } }
public static int countSetBits ( int n ) { return ( BitsSetTable256 [ n & 0xff ] + BitsSetTable256 [ ( n >> 8 ) & 0xff ] + BitsSetTable256 [ ( n >> 16 ) & 0xff ] + BitsSetTable256 [ n >> 24 ] ) ; }
public static void Main ( String [ ] args ) {
initialize ( ) ; int n = 9 ; Console . Write ( countSetBits ( n ) ) ; } }
using System ; using System . Linq ; class GFG {
public static void Main ( ) { Console . WriteLine ( Convert . ToString ( 4 , 2 ) . Count ( c = > c == '1' ) ) ; Console . WriteLine ( Convert . ToString ( 15 , 2 ) . Count ( c = > c == '1' ) ) ; } }
class GFG { static int [ ] num_to_bits = new int [ 16 ] { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
static int countSetBitsRec ( int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
static void Main ( ) { int num = 31 ; System . Console . WriteLine ( countSetBitsRec ( num ) ) ; } }
using System ; class GFG {
static int countSetBits ( int N ) { int count = 0 ;
for ( int i = 0 ; i < 4 * 8 ; i ++ ) { if ( ( N & ( 1 << i ) ) != 0 ) count ++ ; } return count ; }
static void Main ( ) { int N = 15 ; Console . WriteLine ( countSetBits ( N ) ) ; } }
using System ; class GFG {
static bool getParity ( int n ) { bool parity = false ; while ( n != 0 ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
public static void Main ( ) { int n = 7 ; Console . Write ( " Parity ▁ of ▁ no ▁ " + n + " ▁ = ▁ " + ( getParity ( n ) ? " odd " : " even " ) ) ; } }
using System ; class GFG {
static bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( int ) ( Math . Ceiling ( ( Math . Log ( n ) / Math . Log ( 2 ) ) ) ) == ( int ) ( Math . Floor ( ( ( Math . Log ( n ) / Math . Log ( 2 ) ) ) ) ) ; }
public static void Main ( ) { if ( isPowerOfTwo ( 31 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; if ( isPowerOfTwo ( 64 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; while ( n != 1 ) { if ( n % 2 != 0 ) return false ; n = n / 2 ; } return true ; }
public static void Main ( ) { Console . WriteLine ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; Console . WriteLine ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
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
using System ; class GFG {
static int maxRepeating ( int [ ] arr , int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) arr [ ( arr [ i ] % k ) ] += k ;
int max = arr [ 0 ] , result = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; result = i ; } }
return result ; }
public static void Main ( ) { int [ ] arr = { 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 } ; int n = arr . Length ; int k = 8 ; Console . Write ( " Maximum ▁ repeating ▁ " + " element ▁ is : ▁ " + maxRepeating ( arr , n , k ) ) ; } }
using System ; class GFG {
static int fun ( int x ) { int y = ( x / 4 ) * 4 ;
int ans = 0 ; for ( int i = y ; i <= x ; i ++ ) ans ^= i ; return ans ; }
static int query ( int x ) {
if ( x == 0 ) return 0 ; int k = ( x + 1 ) / 2 ;
return ( ( x %= 2 ) != 0 ) ? 2 * fun ( k ) : ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) ; } static void allQueries ( int q , int [ ] l , int [ ] r ) { for ( int i = 0 ; i < q ; i ++ ) Console . WriteLine ( ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) ) ; }
public static void Main ( ) { int q = 3 ; int [ ] l = { 2 , 2 , 5 } ; int [ ] r = { 4 , 8 , 9 } ; allQueries ( q , l , r ) ; } }
using System ; class GFG {
static void prefixXOR ( int [ ] arr , int [ ] preXOR , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { while ( arr [ i ] % 2 != 1 ) arr [ i ] /= 2 ; preXOR [ i ] = arr [ i ] ; }
for ( int i = 1 ; i < n ; i ++ ) preXOR [ i ] = preXOR [ i - 1 ] ^ preXOR [ i ] ; }
static int query ( int [ ] preXOR , int l , int r ) { if ( l == 0 ) return preXOR [ r ] ; else return preXOR [ r ] ^ preXOR [ l - 1 ] ; }
public static void Main ( ) { int [ ] arr = { 3 , 4 , 5 } ; int n = arr . Length ; int [ ] preXOR = new int [ n ] ; prefixXOR ( arr , preXOR , n ) ; Console . WriteLine ( query ( preXOR , 0 , 2 ) ) ; Console . WriteLine ( query ( preXOR , 1 , 2 ) ) ; } }
using System ; class GFG {
static int findMinSwaps ( int [ ] arr , int n ) {
int [ ] noOfZeroes = new int [ n ] ; int i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
public static void Main ( ) { int [ ] ar = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; Console . WriteLine ( findMinSwaps ( ar , ar . Length ) ) ; } }
using System ; class GFG { static int minswaps ( int [ ] arr , int n ) { int count = 0 ; int num_unplaced_zeros = 0 ; for ( int index = n - 2 ; index >= 0 ; index -- ) { if ( arr [ index ] == 0 ) num_unplaced_zeros += 1 ; else count += num_unplaced_zeros ; } return count ; }
static void Main ( ) { int [ ] arr = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; Console . WriteLine ( minswaps ( arr , 9 ) ) ; } }
using System ; class GFG {
static bool arraySortedOrNot ( int [ ] arr , int n ) {
if ( n == 0 n == 1 ) return true ; for ( int i = 1 ; i < n ; i ++ )
if ( arr [ i - 1 ] > arr [ i ] ) return false ;
return true ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 20 , 23 , 23 , 45 , 78 , 88 } ; int n = arr . Length ; if ( arraySortedOrNot ( arr , n ) ) Console . Write ( " Yes STRNEWLINE " ) ; else Console . Write ( " No STRNEWLINE " ) ; } }
using System ; class main {
static void printTwoOdd ( int [ ] arr , int size ) {
int xor2 = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( ( arr [ i ] & set_bit_no ) > 0 ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } Console . WriteLine ( " The ▁ two ▁ ODD ▁ elements ▁ are ▁ " + x + " ▁ & ▁ " + y ) ; }
public static void Main ( ) { int [ ] arr = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = arr . Length ; printTwoOdd ( arr , arr_size ) ; } }
using System ; class GFG {
static bool findPair ( int [ ] arr , int n ) { int size = arr . Length ;
int i = 0 , j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { Console . Write ( " Pair ▁ Found : ▁ " + " ( ▁ " + arr [ i ] + " , ▁ " + arr [ j ] + " ▁ ) " ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } Console . Write ( " No ▁ such ▁ pair " ) ; return false ; }
public static void Main ( ) { int [ ] arr = { 1 , 8 , 30 , 40 , 100 } ; int n = 60 ; findPair ( arr , n ) ; } }
using System ; using System . Linq ; class GFG {
public static void printMax ( int [ ] arr , int k , int n ) {
int [ ] brr = new int [ n ] ; for ( int i = 0 ; i < n ; i ++ ) brr [ i ] = arr [ i ] ;
Array . Sort ( brr ) ; Array . Reverse ( brr ) ; int [ ] crr = new int [ k ] ; for ( int i = 0 ; i < k ; i ++ ) { crr [ i ] = brr [ i ] ; }
for ( int i = 0 ; i < n ; ++ i ) { if ( crr . Contains ( arr [ i ] ) ) { Console . Write ( arr [ i ] + " ▁ " ) ; } } }
public static void Main ( ) { int [ ] arr = { 50 , 8 , 45 , 12 , 25 , 40 , 84 } ; int n = arr . Length ; int k = 3 ; printMax ( arr , k , n ) ; } }
using System ; class GFG {
static void printSmall ( int [ ] arr , int asize , int n ) {
int [ ] copy_arr = new int [ asize ] ; Array . Copy ( arr , copy_arr , asize ) ;
Array . Sort ( copy_arr ) ;
for ( int i = 0 ; i < asize ; ++ i ) { if ( Array . BinarySearch ( copy_arr , 0 , n , arr [ i ] ) > - 1 ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 } ; int asize = arr . Length ; int n = 5 ; printSmall ( arr , asize , n ) ; } }
using System ; class GFG {
static bool checkIsAP ( int [ ] arr , int n ) { if ( n == 1 ) return true ;
Array . Sort ( arr ) ;
int d = arr [ 1 ] - arr [ 0 ] ; for ( int i = 2 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] != d ) return false ; return true ; }
public static void Main ( ) { int [ ] arr = { 20 , 15 , 5 , 0 , 10 } ; int n = arr . Length ; if ( checkIsAP ( arr , n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG { static int countPairs ( int [ ] a , int n ) {
int mn = int . MaxValue ; int mx = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) { mn = Math . Min ( mn , a [ i ] ) ; mx = Math . Max ( mx , a [ i ] ) ; }
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == mn ) c1 ++ ; if ( a [ i ] == mx ) c2 ++ ; }
if ( mn == mx ) return n * ( n - 1 ) / 2 ; else return c1 * c2 ; }
public static void Main ( ) { int [ ] a = { 3 , 2 , 1 , 1 , 3 } ; int n = a . Length ; Console . WriteLine ( countPairs ( a , n ) ) ; } }
using System ; class GFG { class Node { public int data ; public Node next ; }
static Node rearrange ( Node head ) {
if ( head == null ) return null ;
Node prev = head , curr = head . next ; while ( curr != null ) {
if ( prev . data > curr . data ) { int t = prev . data ; prev . data = curr . data ; curr . data = t ; }
if ( curr . next != null && curr . next . data > curr . data ) { int t = curr . next . data ; curr . next . data = curr . data ; curr . data = t ; } prev = curr . next ; if ( curr . next == null ) break ; curr = curr . next . next ; } return head ; }
static Node push ( Node head , int k ) { Node tem = new Node ( ) ; tem . data = k ; tem . next = head ; head = tem ; return head ; }
static void display ( Node head ) { Node curr = head ; while ( curr != null ) { Console . Write ( curr . data + " ▁ " ) ; curr = curr . next ; } }
public static void Main ( string [ ] args ) { Node head = null ;
head = push ( head , 7 ) ; head = push ( head , 3 ) ; head = push ( head , 8 ) ; head = push ( head , 6 ) ; head = push ( head , 9 ) ; head = rearrange ( head ) ; display ( head ) ; } }
using System ; public class LinkedList {
public class Node { public int data ; public Node next ; public Node ( int d ) { data = d ; next = null ; } }
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
if ( len1 != len2 ) { lNode = len1 > len2 ? l1 : l2 ; sNode = len1 > len2 ? l2 : l1 ; sNode = paddZeros ( sNode , Math . Abs ( len1 - len2 ) ) ; } else {
while ( l1 != null && l2 != null ) { if ( l1 . data != l2 . data ) { lNode = l1 . data > l2 . data ? temp1 : temp2 ; sNode = l1 . data > l2 . data ? temp2 : temp1 ; break ; } l1 = l1 . next ; l2 = l2 . next ; } }
borrow = false ; return subtractLinkedListHelper ( lNode , sNode ) ; }
static void printList ( Node head ) { Node temp = head ; while ( temp != null ) { Console . Write ( temp . data + " ▁ " ) ; temp = temp . next ; } }
public static void Main ( String [ ] args ) { Node head = new Node ( 1 ) ; head . next = new Node ( 0 ) ; head . next . next = new Node ( 0 ) ; Node head2 = new Node ( 1 ) ; LinkedList ob = new LinkedList ( ) ; Node result = ob . subtractLinkedList ( head , head2 ) ; printList ( result ) ; } }
using System ; public class LinkedList {
public class Node { public int data ; public Node next ;
public Node ( int d ) { data = d ; next = null ; } }
static void insertAtMid ( int x ) {
if ( head == null ) head = new Node ( x ) ; else {
Node newNode = new Node ( x ) ; Node ptr = head ; int len = 0 ;
while ( ptr != null ) { len ++ ; ptr = ptr . next ; }
int count = ( ( len % 2 ) == 0 ) ? ( len / 2 ) : ( len + 1 ) / 2 ; ptr = head ;
while ( count -- > 1 ) ptr = ptr . next ;
newNode . next = ptr . next ; ptr . next = newNode ; } }
static void display ( ) { Node temp = head ; while ( temp != null ) { Console . Write ( temp . data + " ▁ " ) ; temp = temp . next ; } }
public static void Main ( ) {
head = null ; head = new Node ( 1 ) ; head . next = new Node ( 2 ) ; head . next . next = new Node ( 4 ) ; head . next . next . next = new Node ( 5 ) ; Console . WriteLine ( " Linked ▁ list ▁ before ▁ " + " insertion : ▁ " ) ; display ( ) ; int x = 3 ; insertAtMid ( x ) ; Console . WriteLine ( " STRNEWLINE Linked ▁ list ▁ after " + " ▁ insertion : ▁ " ) ; display ( ) ; } }
using System ; public class LinkedList {
class Node { public int data ; public Node next ; public Node ( int d ) { data = d ; next = null ; } }
static void insertAtMid ( int x ) {
if ( head == null ) head = new Node ( x ) ; else {
Node newNode = new Node ( x ) ;
Node slow = head ; Node fast = head . next ; while ( fast != null && fast . next != null ) {
slow = slow . next ;
fast = fast . next . next ; }
newNode . next = slow . next ; slow . next = newNode ; } }
static void display ( ) { Node temp = head ; while ( temp != null ) { Console . Write ( temp . data + " ▁ " ) ; temp = temp . next ; } }
public static void Main ( String [ ] args ) {
head = null ; head = new Node ( 1 ) ; head . next = new Node ( 2 ) ; head . next . next = new Node ( 4 ) ; head . next . next . next = new Node ( 5 ) ; Console . WriteLine ( " Linked ▁ list ▁ before " + " ▁ insertion : ▁ " ) ; display ( ) ; int x = 3 ; insertAtMid ( x ) ; Console . WriteLine ( " STRNEWLINE Linked ▁ list ▁ after " + " ▁ insertion : ▁ " ) ; display ( ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node prev , next ; } ;
static Node getNode ( int data ) {
Node newNode = new Node ( ) ;
newNode . data = data ; newNode . prev = newNode . next = null ; return newNode ; }
static Node sortedInsert ( Node head_ref , Node newNode ) { Node current ;
if ( head_ref == null ) head_ref = newNode ;
else if ( ( head_ref ) . >= newNode . data ) { newNode . next = head_ref ; newNode . next . prev = newNode ; head_ref = newNode ; } else { current = head_ref ;
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
static void printList ( Node head ) { while ( head != null ) { Console . Write ( head . data + " ▁ " ) ; head = head . next ; } }
static Node push ( Node head_ref , int new_data ) {
Node new_node = new Node ( ) ;
new_node . data = new_data ;
new_node . next = ( head_ref ) ; new_node . prev = null ;
if ( ( head_ref ) != null ) ( head_ref ) . prev = new_node ;
( head_ref ) = new_node ; return head_ref ; }
public static void Main ( String [ ] args ) {
Node head = null ;
head = push ( head , 9 ) ; head = push ( head , 3 ) ; head = push ( head , 5 ) ; head = push ( head , 10 ) ; head = push ( head , 12 ) ; head = push ( head , 8 ) ; Console . WriteLine ( " Doubly ▁ Linked ▁ List ▁ Before ▁ Sorting " ) ; printList ( head ) ; head = insertionSort ( head ) ; Console . WriteLine ( " STRNEWLINE Doubly ▁ Linked ▁ List ▁ After ▁ Sorting " ) ; printList ( head ) ; } }
using System ; class GFG {
static int [ ] arr = new int [ 10000 ] ;
public static void reverse ( int [ ] arr , int s , int e ) { while ( s < e ) { int tem = arr [ s ] ; arr [ s ] = arr [ e ] ; arr [ e ] = tem ; s = s + 1 ; e = e - 1 ; } }
public static void fun ( int [ ] arr , int k ) { int n = 4 - 1 ; int v = n - k ; if ( v >= 0 ) { reverse ( arr , 0 , v ) ; reverse ( arr , v + 1 , n ) ; reverse ( arr , 0 , n ) ; } }
public static void Main ( String [ ] args ) { arr [ 0 ] = 1 ; arr [ 1 ] = 2 ; arr [ 2 ] = 3 ; arr [ 3 ] = 4 ; for ( int i = 0 ; i < 4 ; i ++ ) { fun ( arr , i ) ; Console . Write ( " [ " ) ; for ( int j = 0 ; j < 4 ; j ++ ) { Console . Write ( arr [ j ] + " , ▁ " ) ; } Console . Write ( " ] " ) ; } } }
using System ; class GFG { static int MAX = 100005 ;
static int [ ] seg = new int [ 4 * MAX ] ;
static void build ( int node , int l , int r , int [ ] a ) { if ( l == r ) seg [ node ] = a [ l ] ; else { int mid = ( l + r ) / 2 ; build ( 2 * node , l , mid , a ) ; build ( 2 * node + 1 , mid + 1 , r , a ) ; seg [ node ] = ( seg [ 2 * node ] seg [ 2 * node + 1 ] ) ; } }
static int query ( int node , int l , int r , int start , int end , int [ ] a ) {
if ( l > end r < start ) return 0 ; if ( start <= l && r <= end ) return seg [ node ] ;
int mid = ( l + r ) / 2 ;
return ( ( query ( 2 * node , l , mid , start , end , a ) ) | ( query ( 2 * node + 1 , mid + 1 , r , start , end , a ) ) ) ; }
static void orsum ( int [ ] a , int n , int q , int [ ] k ) {
build ( 1 , 0 , n - 1 , a ) ;
for ( int j = 0 ; j < q ; j ++ ) {
int i = k [ j ] % ( n / 2 ) ;
int sec = query ( 1 , 0 , n - 1 , n / 2 - i , n - i - 1 , a ) ;
int first = ( query ( 1 , 0 , n - 1 , 0 , n / 2 - 1 - i , a ) | query ( 1 , 0 , n - 1 , n - i , n - 1 , a ) ) ; int temp = sec + first ;
Console . Write ( temp + " STRNEWLINE " ) ; } }
public static void Main ( String [ ] args ) { int [ ] a = { 7 , 44 , 19 , 86 , 65 , 39 , 75 , 101 } ; int n = a . Length ; int q = 2 ; int [ ] k = { 4 , 2 } ; orsum ( a , n , q , k ) ; } }
using System ; class GFG {
static void maximumEqual ( int [ ] a , int [ ] b , int n ) {
int [ ] store = new int [ ( int ) 1e5 ] ;
for ( int i = 0 ; i < n ; i ++ ) { store [ b [ i ] ] = i + 1 ; }
int [ ] ans = new int [ ( int ) 1e5 ] ;
for ( int i = 0 ; i < n ; i ++ ) {
int d = Math . Abs ( store [ a [ i ] ] - ( i + 1 ) ) ;
if ( store [ a [ i ] ] < i + 1 ) { d = n - d ; }
ans [ d ] ++ ; } int finalans = 0 ;
for ( int i = 0 ; i < 1e5 ; i ++ ) finalans = Math . Max ( finalans , ans [ i ] ) ;
Console . Write ( finalans + " STRNEWLINE " ) ; }
public static void Main ( ) {
int [ ] A = { 6 , 7 , 3 , 9 , 5 } ; int [ ] B = { 7 , 3 , 9 , 5 , 6 } ; int size = A . Length ;
maximumEqual ( A , B , size ) ; } }
using System ; class GFG {
static void RightRotate ( int [ ] a , int n , int k ) {
k = k % n ; for ( int i = 0 ; i < n ; i ++ ) { if ( i < k ) {
Console . Write ( a [ n + i - k ] + " ▁ " ) ; } else {
Console . Write ( a [ i - k ] + " ▁ " ) ; } } Console . WriteLine ( ) ; }
public static void Main ( String [ ] args ) { int [ ] Array = { 1 , 2 , 3 , 4 , 5 } ; int N = Array . Length ; int K = 2 ; RightRotate ( Array , N , K ) ; } }
using System ; class GFG {
static void restoreSortedArray ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > arr [ i + 1 ] ) {
reverse ( arr , 0 , i ) ; reverse ( arr , i + 1 , n ) ; reverse ( arr , 0 , n ) ; } } } static void reverse ( int [ ] arr , int i , int j ) { int temp ; while ( i < j ) { temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; i ++ ; j -- ; } }
static void printArray ( int [ ] arr , int size ) { for ( int i = 0 ; i < size ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 4 , 5 , 1 , 2 } ; int n = arr . Length ; restoreSortedArray ( arr , n - 1 ) ; printArray ( arr , n ) ; } }
using System ; class GFG {
static int findStartIndexOfArray ( int [ ] arr , int low , int high ) { if ( low > high ) { return - 1 ; } if ( low == high ) { return low ; } int mid = low + ( high - low ) / 2 ; if ( arr [ mid ] > arr [ mid + 1 ] ) { return mid + 1 ; } if ( arr [ mid - 1 ] > arr [ mid ] ) { return mid ; } if ( arr [ low ] > arr [ mid ] ) { return findStartIndexOfArray ( arr , low , mid - 1 ) ; } else { return findStartIndexOfArray ( arr , mid + 1 , high ) ; } }
static void restoreSortedArray ( int [ ] arr , int n ) {
if ( arr [ 0 ] < arr [ n - 1 ] ) { return ; } int start = findStartIndexOfArray ( arr , 0 , n - 1 ) ;
Array . Sort ( arr , 0 , start ) ; Array . Sort ( arr , start , n ) ; Array . Sort ( arr ) ; }
static void printArray ( int [ ] arr , int size ) { for ( int i = 0 ; i < size ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . Length ; restoreSortedArray ( arr , n ) ; printArray ( arr , n ) ; } }
using System ; class GFG {
static String leftrotate ( String str , int d ) { String ans = str . Substring ( d , str . Length - d ) + str . Substring ( 0 , d ) ; return ans ; }
static String rightrotate ( String str , int d ) { return leftrotate ( str , str . Length - d ) ; }
public static void Main ( String [ ] args ) { String str1 = " GeeksforGeeks " ; Console . WriteLine ( leftrotate ( str1 , 2 ) ) ; String str2 = " GeeksforGeeks " ; Console . WriteLine ( rightrotate ( str2 , 2 ) ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node next ; public Node prev ; } ;
static Node insertNode ( Node start , int value ) {
Node new_node = new Node ( ) ; if ( start == null ) { new_node . data = value ; new_node . next = new_node . prev = new_node ; start = new_node ; return new_node ; }
Node last = ( start ) . prev ;
new_node = new Node ( ) ; new_node . data = value ;
new_node . next = start ;
( start ) . prev = new_node ;
new_node . prev = last ;
last . next = new_node ; return start ; }
static void displayList ( Node start ) { Node temp = start ; while ( temp . next != start ) { Console . Write ( " { 0 } ▁ " , temp . data ) ; temp = temp . next ; } Console . Write ( " { 0 } ▁ " , temp . data ) ; }
static int searchList ( Node start , int search ) {
Node temp = start ;
int count = 0 , flag = 0 , value ;
if ( temp == null ) return - 1 ; else {
while ( temp . next != start ) {
count ++ ;
if ( temp . data == search ) { flag = 1 ; count -- ; break ; }
temp = temp . next ; }
if ( temp . data == search ) { count ++ ; flag = 1 ; }
if ( flag == 1 ) Console . WriteLine ( " STRNEWLINE " + search + " ▁ found ▁ at ▁ location ▁ " + count ) ; else Console . WriteLine ( " STRNEWLINE " + search + " ▁ not ▁ found " ) ; } return - 1 ; }
public static void Main ( String [ ] args ) {
Node start = null ;
start = insertNode ( start , 4 ) ;
start = insertNode ( start , 5 ) ;
start = insertNode ( start , 7 ) ;
start = insertNode ( start , 8 ) ;
start = insertNode ( start , 6 ) ; Console . Write ( " Created ▁ circular ▁ doubly ▁ linked ▁ list ▁ is : ▁ " ) ; displayList ( start ) ; searchList ( start , 5 ) ; } }
using System ; class GFG {
public class Node { public int data ; public Node next , prev ; } ;
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
static void display ( Node head ) { if ( head == null ) return ; Node temp = head ; Console . Write ( " Forward ▁ direction : ▁ " ) ; while ( temp . next != head ) { Console . Write ( temp . data + " ▁ " ) ; temp = temp . next ; } Console . Write ( temp . data + " ▁ " ) ; Node last = head . prev ; temp = last ; Console . Write ( " STRNEWLINE Backward ▁ direction : ▁ " ) ; while ( temp . prev != last ) { Console . Write ( temp . data + " ▁ " ) ; temp = temp . prev ; } Console . Write ( temp . data + " ▁ " ) ; }
public static void Main ( String [ ] args ) { Node head = null ; head = insertEnd ( head , getNode ( 1 ) ) ; head = insertEnd ( head , getNode ( 2 ) ) ; head = insertEnd ( head , getNode ( 3 ) ) ; head = insertEnd ( head , getNode ( 4 ) ) ; head = insertEnd ( head , getNode ( 5 ) ) ; Console . Write ( " Current ▁ list : STRNEWLINE " ) ; display ( head ) ; head = reverse ( head ) ; Console . Write ( " STRNEWLINE STRNEWLINE Reversed ▁ list : STRNEWLINE " ) ; display ( head ) ; } }
using System ; using System . Collections ; class GFG { static int MAXN = 1001 ;
static int [ ] depth = new int [ MAXN ] ;
static int [ ] parent = new int [ MAXN ] ; static ArrayList [ ] adj = new ArrayList [ MAXN ] ; static void addEdge ( int u , int v ) { adj [ u ] . Add ( v ) ; adj [ v ] . Add ( u ) ; } static void dfs ( int cur , int prev ) {
parent [ cur ] = prev ;
depth [ cur ] = depth [ prev ] + 1 ;
for ( int i = 0 ; i < adj [ cur ] . Count ; i ++ ) if ( ( int ) adj [ cur ] [ i ] != prev ) dfs ( ( int ) adj [ cur ] [ i ] , cur ) ; } static void preprocess ( ) {
depth [ 0 ] = - 1 ;
dfs ( 1 , 0 ) ; }
static int LCANaive ( int u , int v ) { if ( u == v ) return u ; if ( depth [ u ] > depth [ v ] ) { int temp = u ; u = v ; v = temp ; } v = parent [ v ] ; return LCANaive ( u , v ) ; }
public static void Main ( string [ ] args ) { for ( int i = 0 ; i < MAXN ; i ++ ) adj [ i ] = new ArrayList ( ) ;
addEdge ( 1 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 1 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 2 , 6 ) ; addEdge ( 3 , 7 ) ; addEdge ( 4 , 8 ) ; addEdge ( 4 , 9 ) ; addEdge ( 9 , 10 ) ; addEdge ( 9 , 11 ) ; addEdge ( 7 , 12 ) ; addEdge ( 7 , 13 ) ; preprocess ( ) ; Console . WriteLine ( " LCA ( 11 , ▁ 8 ) ▁ : ▁ " + LCANaive ( 11 , 8 ) ) ; Console . WriteLine ( " LCA ( 3 , ▁ 13 ) ▁ : ▁ " + LCANaive ( 3 , 13 ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG { static readonly int MAXN = 1001 ;
static int block_sz ;
static int [ ] depth = new int [ MAXN ] ;
static int [ ] parent = new int [ MAXN ] ;
static int [ ] jump_parent = new int [ MAXN ] ; static List < int > [ ] adj = new List < int > [ MAXN ] ; static void addEdge ( int u , int v ) { adj [ u ] . Add ( v ) ; adj [ v ] . Add ( u ) ; } static int LCANaive ( int u , int v ) { if ( u == v ) return u ; if ( depth [ u ] > depth [ v ] ) { int t = u ; u = v ; v = t ; } v = parent [ v ] ; return LCANaive ( u , v ) ; }
static void dfs ( int cur , int prev ) {
depth [ cur ] = depth [ prev ] + 1 ;
parent [ cur ] = prev ;
if ( depth [ cur ] % block_sz == 0 )
jump_parent [ cur ] = parent [ cur ] ; else
jump_parent [ cur ] = jump_parent [ prev ] ;
for ( int i = 0 ; i < adj [ cur ] . Count ; ++ i ) if ( adj [ cur ] [ i ] != prev ) dfs ( adj [ cur ] [ i ] , cur ) ; }
static int LCASQRT ( int u , int v ) { while ( jump_parent [ u ] != jump_parent [ v ] ) { if ( depth [ u ] > depth [ v ] ) {
int t = u ; u = v ; v = t ; }
v = jump_parent [ v ] ; }
return LCANaive ( u , v ) ; } static void preprocess ( int height ) { block_sz = ( int ) Math . Sqrt ( height ) ; depth [ 0 ] = - 1 ;
dfs ( 1 , 0 ) ; }
public static void Main ( String [ ] args ) { for ( int i = 0 ; i < adj . Length ; i ++ ) adj [ i ] = new List < int > ( ) ;
addEdge ( 1 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 1 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 2 , 6 ) ; addEdge ( 3 , 7 ) ; addEdge ( 4 , 8 ) ; addEdge ( 4 , 9 ) ; addEdge ( 9 , 10 ) ; addEdge ( 9 , 11 ) ; addEdge ( 7 , 12 ) ; addEdge ( 7 , 13 ) ;
int height = 4 ; preprocess ( height ) ; Console . Write ( " LCA ( 11,8 ) ▁ : ▁ " + LCASQRT ( 11 , 8 ) + " STRNEWLINE " ) ; Console . Write ( " LCA ( 3,13 ) ▁ : ▁ " + LCASQRT ( 3 , 13 ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
public static void Main ( ) { int N = 3 ;
Console . Write ( Math . Pow ( 2 , N + 1 ) - 2 ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int countOfNum ( int n , int a , int b ) { int cnt_of_a , cnt_of_b , cnt_of_ab , sum ;
cnt_of_a = n / a ;
cnt_of_b = n / b ;
sum = cnt_of_b + cnt_of_a ;
cnt_of_ab = n / ( a * b ) ;
sum = sum - cnt_of_ab ; return sum ; }
static int sumOfNum ( int n , int a , int b ) { int i ; int sum = 0 ;
HashSet < int > ans = new HashSet < int > ( ) ;
for ( i = a ; i <= n ; i = i + a ) { ans . Add ( i ) ; }
for ( i = b ; i <= n ; i = i + b ) { ans . Add ( i ) ; }
foreach ( int it in ans ) { sum = sum + it ; } return sum ; }
public static void Main ( String [ ] args ) { int N = 88 ; int A = 11 ; int B = 8 ; int count = countOfNum ( N , A , B ) ; int sumofnum = sumOfNum ( N , A , B ) ; Console . Write ( sumofnum % count ) ; } }
using System ; public class GFG {
static double get ( double L , double R ) {
double x = 1.0 / L ;
double y = 1.0 / ( R + 1.0 ) ; return ( x - y ) ; }
public static void Main ( String [ ] args ) { int L = 6 , R = 12 ;
double ans = get ( L , R ) ; Console . Write ( " { 0 : F2 } " , ans ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 100000 ;
static List < int > v = new List < int > ( ) ; static int upper_bound ( List < int > ar , int k ) { int s = 0 ; int e = ar . Count ; while ( s != e ) { int mid = s + e >> 1 ; if ( ar [ mid ] <= k ) { s = mid + 1 ; } else { e = mid ; } } if ( s == ar . Count ) { return - 1 ; } return s ; }
static int consecutiveOnes ( int x ) {
int p = 0 ; while ( x > 0 ) {
if ( x % 2 == 1 && p == 1 ) { return 1 ; }
p = x % 2 ;
x /= 2 ; } return 0 ; }
static void preCompute ( ) {
for ( int i = 0 ; i <= MAX ; i ++ ) { if ( consecutiveOnes ( i ) == 0 ) { v . Add ( i ) ; } } }
static int nextValid ( int n ) {
int it = upper_bound ( v , n ) ; int val = v [ it ] ; return val ; }
static void performQueries ( int [ ] queries , int q ) { for ( int i = 0 ; i < q ; i ++ ) { Console . WriteLine ( nextValid ( queries [ i ] ) ) ; } }
static public void Main ( ) { int [ ] queries = { 4 , 6 } ; int q = queries . Length ;
preCompute ( ) ;
performQueries ( queries , q ) ; } }
using System ; class GFG {
static int changeToOnes ( String str ) {
int i , l , ctr = 0 ; l = str . Length ;
for ( i = l - 1 ; i >= 0 ; i -- ) {
if ( str [ i ] == '1' ) ctr ++ ;
else break ; }
return l - ctr ; }
static String removeZeroesFromFront ( String str ) { String s ; int i = 0 ;
while ( i < str . Length && str [ i ] == '0' ) i ++ ;
if ( i == str . Length ) s = "0" ;
else s = str . Substring ( i , str . Length - i ) ; return s ; }
public static void Main ( String [ ] args ) { String str = "10010111" ;
str = removeZeroesFromFront ( str ) ; Console . WriteLine ( changeToOnes ( str ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int MinDeletion ( int [ ] a , int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( mp . ContainsKey ( a [ i ] ) ) { var val = mp [ a [ i ] ] ; mp . Remove ( a [ i ] ) ; mp . Add ( a [ i ] , val + 1 ) ; } else { mp . Add ( a [ i ] , 1 ) ; } }
int ans = 0 ; foreach ( KeyValuePair < int , int > i in mp ) {
int x = i . Key ;
int frequency = i . Value ;
if ( x <= frequency ) {
ans += ( frequency - x ) ; }
else ans += ; } return ans ; }
public static void Main ( String [ ] args ) { int [ ] a = { 2 , 3 , 2 , 3 , 4 , 4 , 4 , 4 , 5 } ; int n = a . Length ; Console . WriteLine ( MinDeletion ( a , n ) ) ; } }
using System ; class GFG {
static int maxCountAB ( string [ ] s , int n ) {
int A = 0 , B = 0 , BA = 0 , ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) { string S = s [ i ] ; int L = S . Length ; for ( int j = 0 ; j < L - 1 ; j ++ ) {
if ( S [ j ] == ' A ' && S [ j + 1 ] == ' B ' ) { ans ++ ; } }
if ( S [ 0 ] == ' B ' && S [ L - 1 ] == ' A ' ) BA ++ ;
else if ( S [ 0 ] == ' B ' ) ++ ;
else if ( S [ L - 1 ] == ' A ' ) ++ ; }
if ( BA == 0 ) ans += Math . Min ( B , A ) ; else if ( A + B == 0 ) ans += BA - 1 ; else ans += BA + Math . Min ( B , A ) ; return ans ; }
public static void Main ( ) { string [ ] s = { " ABCA " , " BOOK " , " BAND " } ; int n = s . Length ; Console . WriteLine ( maxCountAB ( s , n ) ) ; } }
using System ; class GFG {
static int MinOperations ( int n , int x , int [ ] arr ) {
int total = 0 ; for ( int i = 0 ; i < n ; ++ i ) {
if ( arr [ i ] > x ) { int difference = arr [ i ] - x ; total = total + difference ; arr [ i ] = x ; } }
for ( int i = 1 ; i < n ; ++ i ) { int LeftNeigbouringSum = arr [ i ] + arr [ i - 1 ] ;
if ( LeftNeigbouringSum > x ) { int current_diff = LeftNeigbouringSum - x ; arr [ i ] = Math . Max ( 0 , arr [ i ] - current_diff ) ; total = total + current_diff ; } } return total ; }
public static void Main ( String [ ] args ) { int X = 1 ; int [ ] arr = { 1 , 6 , 1 , 2 , 0 , 4 } ; int N = arr . Length ; Console . WriteLine ( MinOperations ( N , X , arr ) ) ; } }
using System ; public class GFG {
static void findNumbers ( int [ ] arr , int n ) {
int sumN = ( n * ( n + 1 ) ) / 2 ;
int sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
int sum = 0 , sumSq = 0 , i ; for ( i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq += ( int ) Math . Pow ( arr [ i ] , 2 ) ; } int B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; int A = sum - sumN + B ; Console . WriteLine ( " A ▁ = ▁ " + A + " B = " }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 2 , 3 , 4 } ; int n = arr . Length ; findNumbers ( arr , n ) ; } }
using System ; class GFG {
static bool is_prefix ( string temp , string str ) {
if ( temp . Length < str . Length ) return false ; else {
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( str [ i ] != temp [ i ] ) return false ; } return true ; } }
static string lexicographicallyString ( string [ ] input , int n , string str ) {
Array . Sort ( input ) ; for ( int i = 0 ; i < n ; i ++ ) { string temp = input [ i ] ;
if ( is_prefix ( temp , str ) ) { return temp ; } }
return " - 1" ; }
public static void Main ( ) { string [ ] arr = { " apple " , " appe " , " apl " , " aapl " , " appax " } ; string S = " app " ; int N = 5 ; Console . WriteLine ( lexicographicallyString ( arr , N , S ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static void Rearrange ( int [ ] arr , int K , int N ) {
int [ ] ans = new int [ N + 1 ] ;
int f = - 1 ; for ( int i = 0 ; i < N ; i ++ ) { ans [ i ] = - 1 ; }
for ( int i = 0 ; i < arr . Length ; i ++ ) { if ( arr [ i ] == K ) { K = i ; break ; } }
List < int > smaller = new List < int > ( ) ; List < int > greater = new List < int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] < arr [ K ] ) smaller . Add ( arr [ i ] ) ;
else if ( arr [ i ] > arr [ K ] ) greater . Add ( arr [ i ] ) ; } int low = 0 , high = N - 1 ;
while ( low <= high ) {
int mid = ( low + high ) / 2 ;
if ( mid == K ) { ans [ mid ] = arr [ K ] ; f = 1 ; break ; }
else if ( mid < K ) { if ( smaller . Count == 0 ) { break ; } ans [ mid ] = smaller [ smaller . Count - 1 ] ; smaller . RemoveAt ( smaller . Count - 1 ) ; low = mid + 1 ; }
else { if ( greater . Count == 0 ) { break ; } ans [ mid ] = greater [ greater . Count - 1 ] ; greater . RemoveAt ( greater . Count - 1 ) ; high = mid - 1 ; } }
if ( f == - 1 ) { Console . WriteLine ( - 1 ) ; return ; }
for ( int i = 0 ; i < N ; i ++ ) {
if ( ans [ i ] == - 1 ) { if ( smaller . Count > 0 ) { ans [ i ] = smaller [ smaller . Count - 1 ] ; smaller . RemoveAt ( smaller . Count - 1 ) ; } else if ( greater . Count > 0 ) { ans [ i ] = greater [ greater . Count - 1 ] ; greater . RemoveAt ( greater . Count - 1 ) ; } } }
for ( int i = 0 ; i < N ; i ++ ) Console . Write ( ans [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 10 , 7 , 2 , 5 , 3 , 8 } ; int K = 7 ; int N = arr . Length ;
Rearrange ( arr , K , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void minimumK ( List < int > arr , int M , int N ) {
int good = ( int ) ( ( N * 1.0 ) / ( ( M + 1 ) * 1.0 ) ) + 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { int K = i ;
int candies = N ;
int taken = 0 ; while ( candies > 0 ) {
taken += Math . Min ( K , candies ) ; candies -= Math . Min ( K , candies ) ;
for ( int j = 0 ; j < M ; j ++ ) {
int consume = ( arr [ j ] * candies ) / 100 ;
candies -= consume ; } }
if ( taken >= good ) { Console . Write ( i ) ; return ; } } }
public static void Main ( ) { int N = 13 , M = 1 ; List < int > arr = new List < int > ( ) ; arr . Add ( 50 ) ; minimumK ( arr , M , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void calcTotalTime ( string path ) {
int time = 0 ;
int x = 0 , y = 0 ;
HashSet < string > s = new HashSet < string > ( ) ; for ( int i = 0 ; i < path . Length ; i ++ ) { int p = x ; int q = y ; if ( path [ i ] == ' N ' ) y ++ ; else if ( path [ i ] == ' S ' ) y -- ; else if ( path [ i ] == ' E ' ) x ++ ; else if ( path [ i ] == ' W ' ) x -- ;
string o = ( p + x ) + " ▁ " + ( q + y ) ; if ( s . Contains ( o ) == false ) {
time += 2 ;
s . Add ( o ) ; } else time += 1 ; }
Console . Write ( time ) ; }
public static void Main ( ) { string path = " NSE " ; calcTotalTime ( path ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int findCost ( int [ ] A , int N ) {
int totalCost = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( A [ i ] == 0 ) {
A [ i ] = 1 ;
totalCost += i ; } }
return totalCost ; }
public static void Main ( ) { int [ ] arr = { 1 , 0 , 1 , 0 , 1 , 0 } ; int N = arr . Length ; Console . Write ( findCost ( arr , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static int peakIndex ( int [ ] arr ) { int N = arr . Length ;
if ( arr . Length < 3 ) return - 1 ; int i = 0 ;
while ( i + 1 < N ) {
if ( arr [ i + 1 ] < arr [ i ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; } if ( i == 0 i == N - 1 ) return - 1 ;
int ans = i ;
while ( i < N - 1 ) {
if ( arr [ i ] < arr [ i + 1 ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; }
if ( i == N - 1 ) return ans ;
return - 1 ; }
static public void Main ( ) { int [ ] arr = { 0 , 1 , 0 } ; Console . WriteLine ( peakIndex ( arr ) ) ; } }
using System ; public class GFG {
static void hasArrayTwoPairs ( int [ ] nums , int n , int target ) {
Array . Sort ( nums ) ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = target - nums [ i ] ;
int low = 0 , high = n - 1 ; while ( low <= high ) {
int mid = low + ( ( high - low ) / 2 ) ;
if ( nums [ mid ] > x ) { high = mid - 1 ; }
else if ( nums [ mid ] < x ) { low = mid + 1 ; }
else {
if ( mid == i ) { if ( ( mid - 1 >= 0 ) && nums [ mid - 1 ] == x ) { Console . Write ( nums [ i ] + " , ▁ " ) ; Console . Write ( nums [ mid - 1 ] ) ; return ; } if ( ( mid + 1 < n ) && nums [ mid + 1 ] == x ) { Console . Write ( nums [ i ] + " , ▁ " ) ; Console . Write ( nums [ mid + 1 ] ) ; return ; } break ; }
else { Console . Write ( nums [ i ] + " , ▁ " ) ; Console . Write ( nums [ mid ] ) ; return ; } } } }
Console . Write ( - 1 ) ; }
static public void Main ( ) { int [ ] A = { 0 , - 1 , 2 , - 3 , 1 } ; int X = - 2 ; int N = A . Length ;
hasArrayTwoPairs ( A , N , X ) ; } }
using System ; class GFG {
static void findClosest ( int N , int target ) { int closest = - 1 ; int diff = Int32 . MaxValue ;
for ( int i = 1 ; i <= Math . Sqrt ( N ) ; i ++ ) { if ( N % i == 0 ) {
if ( N / i == i ) {
if ( Math . Abs ( target - i ) < diff ) { diff = Math . Abs ( target - i ) ; closest = i ; } } else {
if ( Math . Abs ( target - i ) < diff ) { diff = Math . Abs ( target - i ) ; closest = i ; }
if ( Math . Abs ( target - N / i ) < diff ) { diff = Math . Abs ( target - N / i ) ; closest = N / i ; } } } }
Console . Write ( closest ) ; }
static void Main ( ) {
int N = 16 , X = 5 ;
findClosest ( N , X ) ; } }
using System ; class GFG {
static int power ( int A , int N ) {
int count = 0 ; if ( A == 1 ) return 0 ; while ( N > 0 ) {
count ++ ;
N /= A ; } return count ; }
static void Pairs ( int N , int A , int B ) { int powerA , powerB ;
powerA = power ( A , N ) ;
powerB = power ( B , N ) ;
int intialB = B , intialA = A ;
A = 1 ; for ( int i = 0 ; i <= powerA ; i ++ ) { B = 1 ; for ( int j = 0 ; j <= powerB ; j ++ ) {
if ( B == N - A ) { Console . WriteLine ( i + " ▁ " + j ) ; return ; }
B *= intialB ; }
A *= intialA ; }
Console . WriteLine ( " - 1" ) ; return ; }
public static void Main ( String [ ] args ) {
int N = 106 , A = 3 , B = 5 ;
Pairs ( N , A , B ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static int findNonMultiples ( int [ ] arr , int n , int k ) {
HashSet < int > multiples = new HashSet < int > ( ) ;
for ( int i = 0 ; i < n ; ++ i ) {
if ( ! multiples . Contains ( arr [ i ] ) ) {
for ( int j = 1 ; j <= k / arr [ i ] ; j ++ ) { multiples . Add ( arr [ i ] * j ) ; } } }
return k - multiples . Count ; }
public static int countValues ( int [ ] arr , int N , int L , int R ) {
return findNonMultiples ( arr , N , R ) - findNonMultiples ( arr , N , L - 1 ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 4 , 5 , 6 } ; int N = arr . Length ; int L = 1 ; int R = 20 ;
Console . WriteLine ( countValues ( arr , N , L , R ) ) ; } }
using System ; using System . Collections ; class GFG {
static void minCollectingSpeed ( int [ ] piles , int H ) {
int ans = - 1 ; int low = 1 , high ; Array . Sort ( piles ) ;
high = piles [ piles . Length - 1 ] ;
while ( low <= high ) {
int K = low + ( high - low ) / 2 ; int time = 0 ;
foreach ( int ai in piles ) { time += ( ai + K - 1 ) / K ; }
if ( time <= H ) { ans = K ; high = K - 1 ; }
else { low = K + 1 ; } }
Console . Write ( ans ) ; }
static public void Main ( string [ ] args ) { int [ ] arr = { 3 , 6 , 7 , 11 } ; int H = 8 ;
minCollectingSpeed ( arr , H ) ; } }
using System ; class GFG {
static int cntDisPairs ( int [ ] arr , int N , int K ) {
int cntPairs = 0 ;
Array . Sort ( arr ) ;
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
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 5 , 7 , 7 , 8 } ; int N = arr . Length ; int K = 13 ; Console . WriteLine ( cntDisPairs ( arr , N , K ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int cntDisPairs ( int [ ] arr , int N , int K ) {
int cntPairs = 0 ;
Dictionary < int , int > cntFre = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < N ; i ++ ) {
if ( cntFre . ContainsKey ( arr [ i ] ) ) cntFre [ arr [ i ] ] = cntFre [ arr [ i ] ] + 1 ; else cntFre . Add ( arr [ i ] , 1 ) ; }
foreach ( KeyValuePair < int , int > it in cntFre ) {
int i = it . Key ;
if ( 2 * i == K ) {
if ( cntFre [ i ] > 1 ) cntPairs += 2 ; } else { if ( cntFre . ContainsKey ( K - i ) ) {
cntPairs += 1 ; } } }
cntPairs = cntPairs / 2 ; return cntPairs ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 5 , 7 , 7 , 8 } ; int N = arr . Length ; int K = 13 ; Console . Write ( cntDisPairs ( arr , N , K ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void longestSubsequence ( int N , int Q , int [ ] arr , int [ , ] Queries ) { for ( int i = 0 ; i < Q ; i ++ ) {
int x = Queries [ i , 0 ] ; int y = Queries [ i , 1 ] ;
arr [ x - 1 ] = y ;
int count = 1 ; for ( int j = 1 ; j < N ; j ++ ) {
if ( arr [ j ] != arr [ j - 1 ] ) { count += 1 ; } }
Console . Write ( count + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 1 , 2 , 5 , 2 } ; int N = arr . Length ; int Q = 2 ; int [ , ] Queries = { { 1 , 3 } , { 4 , 2 } } ;
longestSubsequence ( N , Q , arr , Queries ) ; } }
using System ; class GFG { static void longestSubsequence ( int N , int Q , int [ ] arr , int [ , ] Queries ) { int count = 1 ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] != arr [ i - 1 ] ) { count += 1 ; } }
for ( int i = 0 ; i < Q ; i ++ ) {
int x = Queries [ i , 0 ] ; int y = Queries [ i , 1 ] ;
if ( x > 1 ) {
if ( arr [ x - 1 ] != arr [ x - 2 ] ) { count -= 1 ; }
if ( arr [ x - 2 ] != y ) { count += 1 ; } }
if ( x < N ) {
if ( arr [ x ] != arr [ x - 1 ] ) { count -= 1 ; }
if ( y != arr [ x ] ) { count += 1 ; } } Console . Write ( count + " ▁ " ) ;
arr [ x - 1 ] = y ; } }
public static void Main ( string [ ] args ) { int [ ] arr = { 1 , 1 , 2 , 5 , 2 } ; int N = arr . Length ; int Q = 2 ; int [ , ] Queries = { { 1 , 3 } , { 4 , 2 } } ;
longestSubsequence ( N , Q , arr , Queries ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void sum ( int [ ] arr , int n ) {
Dictionary < int , List < int > > mp = new Dictionary < int , List < int > > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { List < int > v = new List < int > ( ) ; v . Add ( i ) ; if ( mp . ContainsKey ( arr [ i ] ) ) v . AddRange ( mp [ arr [ i ] ] ) ; mp [ arr [ i ] ] = v ; }
int [ ] ans = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) {
int sum = 0 ;
foreach ( int it in mp [ arr [ i ] ] ) {
sum += Math . Abs ( it - i ) ; }
ans [ i ] = sum ; }
for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( ans [ i ] + " ▁ " ) ; } return ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 1 , 3 , 1 , 1 , 2 } ;
int n = arr . Length ;
sum ( arr , n ) ; } }
using System ; class GFG {
static void conVowUpp ( char [ ] str ) {
int N = str . Length ; for ( int i = 0 ; i < N ; i ++ ) { if ( str [ i ] == ' a ' str [ i ] == ' e ' str [ i ] == ' i ' str [ i ] == ' o ' str [ i ] == ' u ' ) { char c = char . ToUpperInvariant ( str [ i ] ) ; str [ i ] = c ; } } foreach ( char c in str ) Console . Write ( c ) ; }
public static void Main ( String [ ] args ) { String str = " eutopia " ; conVowUpp ( str . ToCharArray ( ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; static int N , P ;
static bool helper ( int mid ) { int cnt = 0 ; foreach ( KeyValuePair < int , int > i in mp ) { int temp = i . Value ; while ( temp >= mid ) { temp -= mid ; cnt ++ ; } }
return cnt >= N ; }
static int findMaximumDays ( int [ ] arr ) {
for ( int i = 0 ; i < P ; i ++ ) { if ( mp . ContainsKey ( arr [ i ] ) ) { mp [ arr [ i ] ] = mp [ arr [ i ] ] + 1 ; } else { mp . Add ( arr [ i ] , 1 ) ; } }
int start = 0 , end = P , ans = 0 ; while ( start <= end ) {
int mid = start + ( ( end - start ) / 2 ) ;
if ( mid != 0 && helper ( mid ) ) { ans = mid ;
start = mid + 1 ; } else if ( mid = = 0 ) { start = mid + 1 ; } else { end = mid - 1 ; } } return ans ; }
public static void Main ( String [ ] args ) { N = 3 ; P = 10 ; int [ ] arr = { 1 , 2 , 2 , 1 , 1 , 3 , 3 , 3 , 2 , 4 } ;
Console . Write ( findMaximumDays ( arr ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void countSubarrays ( int [ ] a , int n , int k ) {
int ans = 0 ;
List < int > pref = new List < int > ( ) ; pref . Add ( 0 ) ;
for ( int i = 0 ; i < n ; i ++ ) pref . Add ( ( a [ i ] + pref [ i ] ) % k ) ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) {
if ( ( pref [ j ] - pref [ i - 1 ] + k ) % k == j - i + 1 ) { ans ++ ; } } }
Console . WriteLine ( ans ) ; }
public static void Main ( ) {
int [ ] arr = { 2 , 3 , 5 , 3 , 1 , 5 } ;
int N = arr . Length ;
int K = 4 ;
countSubarrays ( arr , N , K ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void countSubarrays ( int [ ] a , int n , int k ) {
Dictionary < int , int > cnt = new Dictionary < int , int > ( ) ;
long ans = 0 ;
List < int > pref = new List < int > ( ) ; pref . Add ( 0 ) ;
for ( int i = 0 ; i < n ; i ++ ) pref . Add ( ( a [ i ] + pref [ i ] ) % k ) ;
cnt . Add ( 0 , 1 ) ; for ( int i = 1 ; i <= n ; i ++ ) {
int remIdx = i - k ; if ( remIdx >= 0 ) { if ( cnt . ContainsKey ( ( pref [ remIdx ] - remIdx % k + k ) % k ) ) cnt [ ( pref [ remIdx ] - remIdx % k + k ) % k ] = cnt [ ( pref [ remIdx ] - remIdx % k + k ) % k ] - 1 ; else cnt . Add ( ( pref [ remIdx ] - remIdx % k + k ) % k , - 1 ) ; }
if ( cnt . ContainsKey ( ( pref [ i ] - i % k + k ) % k ) ) ans += cnt [ ( pref [ i ] - i % k + k ) % k ] ;
if ( cnt . ContainsKey ( ( pref [ i ] - i % k + k ) % k ) ) cnt [ ( pref [ i ] - i % k + k ) % k ] = cnt [ ( pref [ i ] - i % k + k ) % k ] + 1 ; else cnt . Add ( ( pref [ i ] - i % k + k ) % k , 1 ) ; }
Console . WriteLine ( ans ) ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 2 , 3 , 5 , 3 , 1 , 5 } ;
int N = arr . Length ;
int K = 4 ;
countSubarrays ( arr , N , K ) ; } }
using System ; class GFG {
static bool check ( String s , int k ) { int n = s . Length ;
for ( int i = 0 ; i < k ; i ++ ) { for ( int j = i ; j < n ; j += k ) {
if ( s [ i ] != s [ j ] ) return false ; } } int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( s [ i ] == '0' )
c ++ ;
else
c -- ; }
if ( c == 0 ) return true ; else return false ; }
public static void Main ( String [ ] args ) { String s = "101010" ; int k = 2 ; if ( check ( s , k ) ) Console . Write ( " Yes " + " STRNEWLINE " ) ; else Console . Write ( " No " + " STRNEWLINE " ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool isSame ( String str , int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < str . Length ; i ++ ) { if ( mp . ContainsKey ( str [ i ] - ' a ' ) ) { mp [ str [ i ] - ' a ' ] = mp [ str [ i ] - ' a ' ] + 1 ; } else { mp . Add ( str [ i ] - ' a ' , 1 ) ; } } foreach ( KeyValuePair < int , int > it in mp ) {
if ( ( it . Value ) >= n ) { return true ; } }
return false ; }
public static void Main ( String [ ] args ) { String str = " ccabcba " ; int n = 4 ;
if ( isSame ( str , n ) ) { Console . Write ( " Yes " ) ; } else { Console . Write ( " No " ) ; } } }
using System ; class GFG { static readonly double eps = 1e-6 ;
static double func ( double a , double b , double c , double x ) { return a * x * x + b * x + c ; }
static double findRoot ( double a , double b , double c , double low , double high ) { double x = - 1 ;
while ( Math . Abs ( high - low ) > eps ) {
x = ( low + high ) / 2 ;
if ( func ( a , b , c , low ) * func ( a , b , c , x ) <= 0 ) { high = x ; }
else { low = x ; } }
return x ; }
static void solve ( double a , double b , double c , double A , double B ) {
if ( func ( a , b , c , A ) * func ( a , b , c , B ) > 0 ) { Console . WriteLine ( " No ▁ solution " ) ; }
else { Console . Write ( " { 0 : F4 } " , findRoot ( a , b , c , A , B ) ) ; } }
public static void Main ( String [ ] args ) {
double a = 2 , b = - 3 , c = - 2 , A = 0 , B = 3 ;
solve ( a , b , c , A , B ) ; } }
using System ; class GFG {
static bool possible ( long mid , int [ ] a ) {
long n = a . Length ;
long total = ( n * ( n - 1 ) ) / 2 ;
long need = ( total + 1 ) / 2 ; long count = 0 ; long start = 0 , end = 1 ;
while ( end < n ) { if ( a [ ( int ) end ] - a [ ( int ) start ] <= mid ) { end ++ ; } else { count += ( end - start - 1 ) ; start ++ ; } }
if ( end == n && start < end && a [ ( int ) end - 1 ] - a [ ( int ) start ] <= mid ) { long t = end - start - 1 ; count += ( t * ( t + 1 ) / 2 ) ; }
if ( count >= need ) return true ; else return false ; }
static long findMedian ( int [ ] a ) {
long n = a . Length ;
long low = 0 , high = a [ ( int ) n - 1 ] - a [ 0 ] ;
while ( low <= high ) {
long mid = ( low + high ) / 2 ;
if ( possible ( mid , a ) ) high = mid - 1 ; else low = mid + 1 ; }
return high + 1 ; }
public static void Main ( string [ ] args ) { int [ ] a = { 1 , 7 , 5 , 2 } ; Array . Sort ( a ) ; Console . Write ( findMedian ( a ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void UniversalSubset ( List < String > A , List < String > B ) {
int n1 = A . Count ; int n2 = B . Count ;
List < String > res = new List < String > ( ) ;
int [ , ] A_fre = new int [ n1 , 26 ] ; for ( int i = 0 ; i < n1 ; i ++ ) { for ( int j = 0 ; j < 26 ; j ++ ) A_fre [ i , j ] = 0 ; }
for ( int i = 0 ; i < n1 ; i ++ ) { for ( int j = 0 ; j < A [ i ] . Length ; j ++ ) { A_fre [ i , A [ i ] [ j ] - ' a ' ] ++ ; } }
int [ ] B_fre = new int [ 26 ] ; for ( int i = 0 ; i < n2 ; i ++ ) { int [ ] arr = new int [ 26 ] ; for ( int j = 0 ; j < B [ i ] . Length ; j ++ ) { arr [ B [ i ] [ j ] - ' a ' ] ++ ; B_fre [ B [ i ] [ j ] - ' a ' ] = Math . Max ( B_fre [ B [ i ] [ j ] - ' a ' ] , arr [ B [ i ] [ j ] - ' a ' ] ) ; } } for ( int i = 0 ; i < n1 ; i ++ ) { int flag = 0 ; for ( int j = 0 ; j < 26 ; j ++ ) {
if ( A_fre [ i , j ] < B_fre [ j ] ) {
flag = 1 ; break ; } }
if ( flag == 0 )
res . Add ( A [ i ] ) ; }
if ( res . Count != 0 ) {
for ( int i = 0 ; i < res . Count ; i ++ ) { for ( int j = 0 ; j < res [ i ] . Length ; j ++ ) Console . Write ( res [ i ] [ j ] ) ; } Console . Write ( " ▁ " ) ; }
else Console . ( " - 1" ) ; }
public static void Main ( String [ ] args ) { List < String > A = new List < String > ( ) ; A . Add ( " geeksforgeeks " ) ; A . Add ( " topcoder " ) ; A . Add ( " leetcode " ) ; List < String > B = new List < String > ( ) ; B . Add ( " geek " ) ; B . Add ( " ee " ) ; UniversalSubset ( A , B ) ; } }
using System ; class GFG {
public static void findPair ( int [ ] a , int n ) {
int min_dist = int . MaxValue ; int index_a = - 1 , index_b = - 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i + 1 ; j < n ; j ++ ) {
if ( j - i < min_dist ) {
if ( a [ i ] % a [ j ] == 0 a [ j ] % a [ i ] == 0 ) {
min_dist = j - i ;
index_a = i ; index_b = j ; } } } }
if ( index_a == - 1 ) { Console . WriteLine ( " - 1" ) ; }
else { Console . Write ( " ( " + a [ index_a ] + " , ▁ " + a [ index_b ] + " ) " ) ; } }
public static void Main ( String [ ] args ) {
int [ ] a = { 2 , 3 , 4 , 5 , 6 } ; int n = a . Length ;
findPair ( a , n ) ; } }
using System ; class GFG {
static void printNum ( int L , int R ) {
for ( int i = L ; i <= R ; i ++ ) { int temp = i ; int c = 10 ; int flag = 0 ;
while ( temp > 0 ) {
if ( temp % 10 >= c ) { flag = 1 ; break ; } c = temp % 10 ; temp /= 10 ; }
if ( flag == 0 ) Console . Write ( i + " ▁ " ) ; } }
public static void Main ( ) {
int L = 10 , R = 15 ;
printNum ( L , R ) ; } }
using System ; class GFG {
static int findMissing ( int [ ] arr , int left , int right , int diff ) {
if ( right <= left ) return 0 ;
int mid = left + ( right - left ) / 2 ;
if ( arr [ mid + 1 ] - arr [ mid ] != diff ) return ( arr [ mid ] + diff ) ;
if ( mid > 0 && arr [ mid ] - arr [ mid - 1 ] != diff ) return ( arr [ mid - 1 ] + diff ) ;
if ( arr [ mid ] == arr [ 0 ] + mid * diff ) return findMissing ( arr , mid + 1 , right , diff ) ;
return findMissing ( arr , left , mid - 1 , diff ) ; }
static int missingElement ( int [ ] arr , int n ) {
Array . Sort ( arr ) ;
int diff = ( arr [ n - 1 ] - arr [ 0 ] ) / n ;
return findMissing ( arr , 0 , n - 1 , diff ) ; }
public static void Main ( String [ ] args ) {
int [ ] arr = new int [ ] { 2 , 8 , 6 , 10 } ; int n = arr . Length ;
Console . WriteLine ( missingElement ( arr , n ) ) ; } }
using System ; class GFG {
static int power ( int x , int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
static int nthRootSearch ( int low , int high , int N , int K ) {
if ( low <= high ) {
int mid = ( low + high ) / 2 ;
if ( ( power ( mid , K ) <= N ) && ( power ( mid + 1 , K ) > N ) ) { return mid ; }
else if ( power ( mid , K ) < ) { return nthRootSearch ( mid + 1 , high , N , K ) ; } else { return nthRootSearch ( low , mid - 1 , N , K ) ; } } return low ; }
public static void Main ( ) {
int N = 16 , K = 4 ;
Console . Write ( nthRootSearch ( 0 , N , N , K ) ) ; } }
using System ; class GFG {
static int get_subset_count ( int [ ] arr , int K , int N ) {
Array . Sort ( arr ) ; int left , right ; left = 0 ; right = N - 1 ;
int ans = 0 ; while ( left <= right ) { if ( arr [ left ] + arr [ right ] < K ) {
ans += 1 << ( right - left ) ; left ++ ; } else {
right -- ; } } return ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 4 , 5 , 7 } ; int K = 8 ; int N = arr . Length ; Console . Write ( get_subset_count ( arr , K , N ) ) ; } }
using System ; class GFG { static int minMaxDiff ( int [ ] arr , int n , int k ) { int max_adj_dif = int . MinValue ;
for ( int i = 0 ; i < n - 1 ; i ++ ) max_adj_dif = Math . Max ( max_adj_dif , Math . Abs ( arr [ i ] - arr [ i + 1 ] ) ) ;
if ( max_adj_dif == 0 ) return 0 ;
int best = 1 ; int worst = max_adj_dif ; int mid , required ; while ( best < worst ) { mid = ( best + worst ) / 2 ;
required = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) { required += ( Math . Abs ( arr [ i ] - arr [ i + 1 ] ) - 1 ) / mid ; }
if ( required > k ) best = mid + 1 ;
else worst = mid ; } return worst ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 12 , 25 , 50 } ; int n = arr . Length ; int k = 7 ; Console . WriteLine ( minMaxDiff ( arr , n , k ) ) ; } }
using System ; class GFG {
static void checkMin ( int [ ] arr , int len ) {
int smallest = int . MaxValue ; int secondSmallest = int . MaxValue ; for ( int i = 0 ; i < len ; i ++ ) {
if ( arr [ i ] < smallest ) { secondSmallest = smallest ; smallest = arr [ i ] ; }
else if ( arr [ i ] < secondSmallest ) { secondSmallest = arr [ i ] ; } } if ( 2 * smallest <= secondSmallest ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 4 , 5 } ; int len = arr . Length ; checkMin ( arr , len ) ; } }
using System ; using System . Linq ; using System . Collections . Generic ; class GFG {
static void createHash ( HashSet < int > hash , int maxElement ) {
int prev = 0 , curr = 1 ; hash . Add ( prev ) ; hash . Add ( curr ) ; while ( curr <= maxElement ) {
int temp = curr + prev ; hash . Add ( temp ) ;
prev = curr ; curr = temp ; } }
static void fibonacci ( int [ ] arr , int n ) {
int max_val = arr . Max ( ) ;
HashSet < int > hash = new HashSet < int > ( ) ; createHash ( hash , max_val ) ;
int minimum = int . MaxValue ; int maximum = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) {
if ( hash . Contains ( arr [ i ] ) ) {
minimum = Math . Min ( minimum , arr [ i ] ) ; maximum = Math . Max ( maximum , arr [ i ] ) ; } } Console . Write ( minimum + " , ▁ " + maximum + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n = arr . Length ; fibonacci ( arr , n ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool isValidLen ( String s , int len , int k ) {
int n = s . Length ;
Dictionary < char , int > mp = new Dictionary < char , int > ( ) ; int right = 0 ;
while ( right < len ) { if ( mp . ContainsKey ( s [ right ] ) ) { mp [ s [ right ] ] = mp [ s [ right ] ] + 1 ; } else { mp . Add ( s [ right ] , 1 ) ; } right ++ ; } if ( mp . Count <= k ) return true ;
while ( right < n ) {
if ( mp . ContainsKey ( s [ right ] ) ) { mp [ s [ right ] ] = mp [ s [ right ] ] + 1 ; } else { mp . Add ( s [ right ] , 1 ) ; }
if ( mp . ContainsKey ( s [ right - len ] ) ) { mp [ s [ right - len ] ] = mp [ s [ right - len ] ] - 1 ; }
if ( mp [ s [ right - len ] ] == 0 ) mp . Remove ( s [ right - len ] ) ; if ( mp . Count <= k ) return true ; right ++ ; } return mp . Count <= k ; }
static int maxLenSubStr ( String s , int k ) {
HashSet < char > uni = new HashSet < char > ( ) ; foreach ( char x in s . ToCharArray ( ) ) uni . Add ( x ) ; if ( uni . Count < k ) return - 1 ;
int n = s . Length ;
int lo = - 1 , hi = n + 1 ; while ( hi - lo > 1 ) { int mid = lo + hi >> 1 ; if ( isValidLen ( s , mid , k ) ) lo = mid ; else hi = mid ; } return lo ; }
public static void Main ( String [ ] args ) { String s = " aabacbebebe " ; int k = 3 ; Console . Write ( maxLenSubStr ( s , k ) ) ; } }
using System ; class GFG {
static bool isSquarePossible ( int [ ] arr , int n , int l ) {
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] >= l ) cnt ++ ;
if ( cnt >= l ) return true ; } return false ; }
static int maxArea ( int [ ] arr , int n ) { int l = 0 , r = n ; int len = 0 ; while ( l <= r ) { int m = l + ( ( r - l ) / 2 ) ;
if ( isSquarePossible ( arr , n , m ) ) { len = m ; l = m + 1 ; }
else r = m - 1 ; }
return ( len * len ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 3 , 4 , 5 , 5 } ; int n = arr . Length ; Console . WriteLine ( maxArea ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void insertNames ( String [ ] arr , int n ) {
HashSet < String > set = new HashSet < String > ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( ! set . Contains ( arr [ i ] ) ) { Console . Write ( " No STRNEWLINE " ) ; set . Add ( arr [ i ] ) ; } else { Console . Write ( " Yes STRNEWLINE " ) ; } } }
public static void Main ( String [ ] args ) { String [ ] arr = { " geeks " , " for " , " geeks " } ; int n = arr . Length ; insertNames ( arr , n ) ; } }
using System ; class GFG {
static int countLessThan ( int [ ] arr , int n , int key ) { int l = 0 , r = n - 1 ; int index = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( arr [ m ] < key ) { l = m + 1 ; index = m ; } else { r = m - 1 ; } } return ( index + 1 ) ; }
static int countGreaterThan ( int [ ] arr , int n , int key ) { int l = 0 , r = n - 1 ; int index = - 1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( arr [ m ] <= key ) { l = m + 1 ; } else { r = m - 1 ; index = m ; } } if ( index == - 1 ) return 0 ; return ( n - index ) ; }
static int countTriplets ( int n , int [ ] a , int [ ] b , int [ ] c ) {
Array . Sort ( a ) ; Array . Sort ( b ) ; Array . Sort ( c ) ; int count = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { int current = b [ i ] ;
int low = countLessThan ( a , n , current ) ;
int high = countGreaterThan ( c , n , current ) ;
count += ( low * high ) ; } return count ; }
public static void Main ( ) { int [ ] a = { 1 , 5 } ; int [ ] b = { 2 , 4 } ; int [ ] c = { 3 , 6 } ; int size = a . Length ; Console . WriteLine ( countTriplets ( size , a , b , c ) ) ; } }
using System ; class GFG { static int costToBalance ( string s ) { if ( s . Length == 0 ) Console . WriteLine ( 0 ) ;
int ans = 0 ;
int o = 0 , c = 0 ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( s [ i ] == ' ( ' ) o ++ ; if ( s [ i ] == ' ) ' ) c ++ ; } if ( o != c ) return - 1 ; int [ ] a = new int [ s . Length ] ; if ( s [ 0 ] == ' ( ' ) a [ 0 ] = 1 ; else a [ 0 ] = - 1 ; if ( a [ 0 ] < 0 ) ans += Math . Abs ( a [ 0 ] ) ; for ( int i = 1 ; i < s . Length ; i ++ ) { if ( s [ i ] == ' ( ' ) a [ i ] = a [ i - 1 ] + 1 ; else a [ i ] = a [ i - 1 ] - 1 ; if ( a [ i ] < 0 ) ans += Math . Abs ( a [ i ] ) ; } return ans ; }
static void Main ( ) { string s ; s = " ) ) ) ( ( ( " ; Console . WriteLine ( costToBalance ( s ) ) ; s = " ) ) ( ( " ; Console . WriteLine ( costToBalance ( s ) ) ; } }
using System ; class Middle {
public static int middleOfThree ( int a , int b , int c ) {
int x = a - b ;
int y = b - c ;
int z = a - c ;
if ( x * y > 0 ) return b ;
else if ( x * z > 0 ) return c ; else return a ; }
public static void Main ( ) { int a = 20 , b = 30 , c = 40 ; Console . WriteLine ( middleOfThree ( a , b , c ) ) ; } }
using System ; class Missing4 {
public static void missing4 ( int [ ] arr ) {
int [ ] helper = new int [ 4 ] ;
for ( int i = 0 ; i < arr . Length ; i ++ ) { int temp = Math . Abs ( arr [ i ] ) ;
if ( temp <= arr . Length ) arr [ temp - 1 ] *= ( - 1 ) ;
else if ( temp > arr . Length ) { if ( temp % arr . Length != 0 ) helper [ temp % arr . Length - 1 ] = - 1 ; else helper [ ( temp % arr . Length ) + arr . Length - 1 ] = - 1 ; } }
for ( int i = 0 ; i < arr . Length ; i ++ ) if ( arr [ i ] > 0 ) Console . Write ( i + 1 + " ▁ " ) ; for ( int i = 0 ; i < helper . Length ; i ++ ) if ( helper [ i ] >= 0 ) Console . Write ( arr . Length + i + 1 + " ▁ " ) ; return ; }
public static void Main ( ) { int [ ] arr = { 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 } ; missing4 ( arr ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void lexiMiddleSmallest ( int K , int N ) {
if ( K % 2 == 0 ) {
Console . Write ( K / 2 + " ▁ " ) ;
for ( int i = 0 ; i < N - 1 ; ++ i ) { Console . Write ( K + " ▁ " ) ; } Console . WriteLine ( ) ; return ; }
List < int > a = new List < int > ( ) ;
for ( int i = 0 ; i < N / 2 ; ++ i ) {
if ( a [ a . Count - 1 ] == 1 ) {
a . Remove ( a . Count - 1 ) ; }
else {
a [ a . Count - 1 ] -= 1 ;
while ( ( int ) a . Count < N ) { a . Add ( K ) ; } } }
foreach ( int i in a ) { Console . Write ( i + " ▁ " ) ; } Console . WriteLine ( ) ; }
public static void Main ( ) { int K = 2 , N = 4 ; lexiMiddleSmallest ( K , N ) ; } }
using System ; public class GFG {
static void findLastElement ( int [ ] arr , int N ) {
Array . Sort ( arr ) ; int i = 0 ;
for ( i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] - arr [ i - 1 ] != 0 && arr [ i ] - arr [ i - 1 ] != 2 ) { Console . WriteLine ( " - 1" ) ; return ; } }
Console . WriteLine ( arr [ N - 1 ] ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 4 , 6 , 8 , 0 , 8 } ; int N = arr . Length ; findLastElement ( arr , N ) ; } }
using System ; class GFG {
static void maxDivisions ( int [ ] arr , int N , int X ) {
Array . Sort ( arr ) ; Array . Reverse ( arr ) ;
int maxSub = 0 ;
int size = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
size ++ ;
if ( arr [ i ] * size >= X ) {
maxSub ++ ;
size = 0 ; } } Console . WriteLine ( maxSub ) ; }
public static void Main ( ) {
int [ ] arr = { 1 , 3 , 3 , 7 } ;
int N = arr . Length ;
int X = 3 ; maxDivisions ( arr , N , X ) ; } }
using System ; public class GFG {
public static void maxPossibleSum ( int [ ] arr , int N ) {
Array . Sort ( arr ) ; int sum = 0 ; int j = N - 3 ; while ( j >= 0 ) {
sum += arr [ j ] ; j -= 3 ; }
Console . WriteLine ( sum ) ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 7 , 4 , 5 , 2 , 3 , 1 , 5 , 9 } ;
int N = arr . Length ; maxPossibleSum ( arr , N ) ; } }
using System ; class GFG {
static void insertionSort ( int [ ] arr , int n ) { int i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
static void printArray ( int [ ] arr , int n ) { int i ;
for ( i = 0 ; i < n ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } Console . WriteLine ( ) ; }
static public void Main ( ) { int [ ] arr = new int [ ] { 12 , 11 , 13 , 5 , 6 } ; int N = arr . Length ;
insertionSort ( arr , N ) ; printArray ( arr , N ) ; } }
using System ; class GFG {
static void getPairs ( int [ ] arr , int N , int K ) {
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) {
if ( arr [ i ] > K * arr [ i + 1 ] ) count ++ ; } } Console . Write ( count ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 2 , 1 } ; int N = arr . Length ; int K = 2 ;
getPairs ( arr , N , K ) ; } }
using System ; class GFG {
static int merge ( int [ ] arr , int [ ] temp , int l , int m , int r , int K ) {
int i = l ;
int j = m + 1 ;
int cnt = 0 ; for ( i = l ; i <= m ; i ++ ) { bool found = false ;
while ( j <= r ) {
if ( arr [ i ] >= K * arr [ j ] ) { found = true ; } else break ; j ++ ; }
if ( found == true ) { cnt += j - ( m + 1 ) ; j -- ; } }
int k = l ; i = l ; j = m + 1 ; while ( i <= m && j <= r ) { if ( arr [ i ] <= arr [ j ] ) temp [ k ++ ] = arr [ i ++ ] ; else temp [ k ++ ] = arr [ j ++ ] ; }
while ( i <= m ) temp [ k ++ ] = arr [ i ++ ] ;
while ( j <= r ) temp [ k ++ ] = arr [ j ++ ] ; for ( i = l ; i <= r ; i ++ ) arr [ i ] = temp [ i ] ;
return cnt ; }
static int mergeSortUtil ( int [ ] arr , int [ ] temp , int l , int r , int K ) { int cnt = 0 ; if ( l < r ) {
int m = ( l + r ) / 2 ;
cnt += mergeSortUtil ( arr , temp , l , m , K ) ; cnt += mergeSortUtil ( arr , temp , m + 1 , r , K ) ;
cnt += merge ( arr , temp , l , m , r , K ) ; } return cnt ; }
static void mergeSort ( int [ ] arr , int N , int K ) { int [ ] temp = new int [ N ] ; Console . WriteLine ( mergeSortUtil ( arr , temp , 0 , N - 1 , K ) ) ; }
static public void Main ( ) { int [ ] arr = new int [ ] { 5 , 6 , 2 , 5 } ; int N = arr . Length ; int K = 2 ;
mergeSort ( arr , N , K ) ; } }
using System ; class GFG {
static void minRemovals ( int [ ] A , int N ) {
Array . Sort ( A ) ;
int mx = A [ N - 1 ] ;
int sum = 1 ;
for ( int i = 0 ; i < N ; i ++ ) { sum += A [ i ] ; } if ( sum - mx >= mx ) { Console . WriteLine ( 0 ) ; } else { Console . WriteLine ( 2 * mx - sum ) ; } }
public static void Main ( String [ ] args ) { int [ ] A = { 3 , 3 , 2 } ; int N = A . Length ;
minRemovals ( A , N ) ; } }
using System ; public class GFG {
static void rearrangeArray ( int [ ] a , int n ) {
Array . Sort ( a ) ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( a [ i ] == i + 1 ) {
int temp = a [ i ] ; a [ i ] = a [ i + 1 ] ; a [ i + 1 ] = temp ; } }
if ( a [ n - 1 ] == n ) {
int temp = a [ n - 1 ] ; a [ n - 1 ] = a [ n - 2 ] ; a [ n - 2 ] = temp ; }
for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( a [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 5 , 3 , 2 , 4 } ; int N = arr . Length ;
rearrangeArray ( arr , N ) ; } }
using System ; class GFG {
static int minOperations ( int [ ] arr1 , int [ ] arr2 , int i , int j ) {
if ( arr1 . Equals ( arr2 ) ) return 0 ; if ( i >= arr1 . Length j >= arr2 . Length ) return 0 ;
if ( arr1 [ i ] < arr2 [ j ] )
return 1 + minOperations ( arr1 , arr2 , i + 1 , j + 1 ) ;
return Math . Max ( minOperations ( arr1 , arr2 , i , j + 1 ) , minOperations ( arr1 , arr2 , i + 1 , j ) ) ; }
static void minOperationsUtil ( int [ ] arr ) { int [ ] brr = new int [ arr . Length ] ; for ( int i = 0 ; i < arr . Length ; i ++ ) brr [ i ] = arr [ i ] ; Array . Sort ( brr ) ;
if ( arr . Equals ( brr ) )
Console . Write ( "0" ) ;
else
Console . WriteLine ( minOperations ( arr , brr , 0 , 0 ) ) ; }
static void Main ( ) { int [ ] arr = { 4 , 7 , 2 , 3 , 9 } ; minOperationsUtil ( arr ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void canTransform ( String s , String t ) { int n = s . Length ;
List < int > [ ] occur = new List < int > [ 26 ] ; for ( int i = 0 ; i < occur . Length ; i ++ ) occur [ i ] = new List < int > ( ) ; for ( int x = 0 ; x < n ; x ++ ) { char ch = ( char ) ( s [ x ] - ' a ' ) ; occur [ ch ] . Add ( x ) ; }
int [ ] idx = new int [ 26 ] ; bool poss = true ; for ( int x = 0 ; x < n ; x ++ ) { char ch = ( char ) ( t [ x ] - ' a ' ) ;
if ( idx [ ch ] >= occur [ ch ] . Count ) {
poss = false ; break ; } for ( int small = 0 ; small < ch ; small ++ ) {
if ( idx [ small ] < occur [ small ] . Count && occur [ small ] [ idx [ small ] ] < occur [ ch ] [ idx [ ch ] ] ) {
poss = false ; break ; } } idx [ ch ] ++ ; }
if ( poss ) { Console . Write ( " Yes " + " STRNEWLINE " ) ; } else { Console . Write ( " No " + " STRNEWLINE " ) ; } }
public static void Main ( String [ ] args ) { String s , t ; s = " hdecb " ; t = " cdheb " ; canTransform ( s , t ) ; } }
using System ; class GFG {
static int inversionCount ( String s ) {
int [ ] freq = new int [ 26 ] ; int inv = 0 ; for ( int i = 0 ; i < s . Length ; i ++ ) { int temp = 0 ;
for ( int j = 0 ; j < ( int ) ( s [ i ] - ' a ' ) ; j ++ )
temp += freq [ j ] ; inv += ( i - temp ) ;
freq [ s [ i ] - ' a ' ] ++ ; } return inv ; }
static bool haveRepeated ( String S1 , String S2 ) { int [ ] freq = new int [ 26 ] ; foreach ( char i in S1 . ToCharArray ( ) ) { if ( freq [ i - ' a ' ] > 0 ) return true ; freq [ i - ' a ' ] ++ ; } for ( int i = 0 ; i < 26 ; i ++ ) freq [ i ] = 0 ; foreach ( char i in S2 . ToCharArray ( ) ) { if ( freq [ i - ' a ' ] > 0 ) return true ; freq [ i - ' a ' ] ++ ; } return false ; }
static void checkToMakeEqual ( String S1 , String S2 ) {
int [ ] freq = new int [ 26 ] ; for ( int i = 0 ; i < S1 . Length ; i ++ ) {
freq [ S1 [ i ] - ' a ' ] ++ ; } bool flag = false ; for ( int i = 0 ; i < S2 . Length ; i ++ ) { if ( freq [ S2 [ i ] - ' a ' ] == 0 ) {
flag = true ; break ; }
freq [ S2 [ i ] - ' a ' ] -- ; } if ( flag == true ) {
Console . WriteLine ( " No " ) ; return ; }
int invCount1 = inversionCount ( S1 ) ; int invCount2 = inversionCount ( S2 ) ; if ( invCount1 == invCount2 || ( invCount1 & 1 ) == ( invCount2 & 1 ) || haveRepeated ( S1 , S2 ) ) {
Console . WriteLine ( " Yes " ) ; } else Console . ( " No " ) ; }
public static void Main ( String [ ] args ) { String S1 = " abbca " , S2 = " acabb " ; checkToMakeEqual ( S1 , S2 ) ; } }
using System ; class GFG {
static void sortArr ( int [ ] a , int n ) { int i , k ;
k = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) ; k = ( int ) Math . Pow ( 2 , k ) ;
while ( k > 0 ) { for ( i = 0 ; i + k < n ; i ++ ) if ( a [ i ] > a [ i + k ] ) { int tmp = a [ i ] ; a [ i ] = a [ i + k ] ; a [ i + k ] = tmp ; }
k = k / 2 ; }
for ( i = 0 ; i < n ; i ++ ) { Console . Write ( a [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) {
int [ ] arr = { 5 , 20 , 30 , 40 , 36 , 33 , 25 , 15 , 10 } ; int n = arr . Length ;
sortArr ( arr , n ) ; } }
using System ; class GFG {
static void maximumSum ( int [ ] arr , int n , int k ) {
int elt = n / k ; int sum = 0 ;
Array . Sort ( arr ) ; int count = 0 ; int i = n - 1 ;
while ( count < k ) { sum += arr [ i ] ; i -- ; count ++ ; } count = 0 ; i = 0 ;
while ( count < k ) { sum += arr [ i ] ; i += elt - 1 ; count ++ ; }
Console . WriteLine ( sum ) ; }
public static void Main ( String [ ] args ) { int [ ] Arr = { 1 , 13 , 7 , 17 , 6 , 5 } ; int K = 2 ; int size = Arr . Length ; maximumSum ( Arr , size , K ) ; } }
using System ; class GFG {
static int findMinSum ( int [ ] arr , int K , int L , int size ) { if ( K * L > size ) return - 1 ; int minsum = 0 ;
Array . Sort ( arr ) ;
for ( int i = 0 ; i < K ; i ++ ) minsum += arr [ i ] ;
return minsum ; }
public static void Main ( ) { int [ ] arr = { 2 , 15 , 5 , 1 , 35 , 16 , 67 , 10 } ; int K = 3 ; int L = 2 ; int length = arr . Length ; Console . Write ( findMinSum ( arr , K , L , length ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
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
public static void Main ( String [ ] args ) {
int [ ] arr = { 7 , 1 , 4 , 4 , 20 , 15 , 8 } ; int N = arr . Length ; int K = 5 ;
Console . Write ( findKthSmallest ( arr , N , K ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void lexNumbers ( int n ) { List < String > s = new List < String > ( ) ; for ( int i = 1 ; i <= n ; i ++ ) { s . Add ( String . Join ( " " , i ) ) ; } s . Sort ( ) ; List < int > ans = new List < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) ans . Add ( Int32 . Parse ( s [ i ] ) ) ; for ( int i = 0 ; i < n ; i ++ ) Console . Write ( ans [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int n = 15 ; lexNumbers ( n ) ; } }
using System ; class GFG { static int N = 4 ; static void func ( int [ , ] a ) { int i , j , k ;
for ( i = 0 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) { for ( j = 0 ; j < N ; j ++ ) { for ( k = j + 1 ; k < N ; ++ k ) {
if ( a [ i , j ] > a [ i , k ] ) {
int temp = a [ i , j ] ; a [ i , j ] = a [ i , k ] ; a [ i , k ] = temp ; } } } }
else { for ( j = 0 ; j < N ; j ++ ) { for ( k = j + 1 ; k < N ; ++ k ) {
if ( a [ i , j ] < a [ i , k ] ) {
int temp = a [ i , j ] ; a [ i , j ] = a [ i , k ] ; a [ i , k ] = temp ; } } } } }
for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) { Console . Write ( a [ i , j ] + " ▁ " ) ; } Console . Write ( " STRNEWLINE " ) ; } }
public static void Main ( String [ ] args ) { int [ , ] a = { { 5 , 7 , 3 , 4 } , { 9 , 5 , 8 , 2 } , { 6 , 3 , 8 , 1 } , { 5 , 8 , 9 , 3 } } ; func ( a ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG {
static Dictionary < int , int > [ ] g = new Dictionary < int , int > [ 200005 ] ; static HashSet < int > s = new HashSet < int > ( ) ; static HashSet < int > ns = new HashSet < int > ( ) ;
static void dfs ( int x ) { ArrayList v = new ArrayList ( ) ; ns . Clear ( ) ;
foreach ( int it in s ) {
if ( g [ x ] . ContainsKey ( it ) ) { v . Add ( it ) ; } else { ns . Add ( it ) ; } } s = ns ; foreach ( int i in v ) { dfs ( i ) ; } }
static void weightOfMST ( int N ) {
int cnt = 0 ;
for ( int i = 1 ; i <= N ; ++ i ) { s . Add ( i ) ; } ArrayList qt = new ArrayList ( ) ; foreach ( int t in s ) qt . Add ( t ) ;
while ( qt . Count != 0 ) {
++ cnt ; int t = ( int ) qt [ 0 ] ; qt . RemoveAt ( 0 ) ;
dfs ( t ) ; } Console . Write ( cnt - 4 ) ; }
public static void Main ( string [ ] args ) { int N = 6 , M = 11 ; int [ , ] edges = { { 1 , 3 } , { 1 , 4 } , { 1 , 5 } , { 1 , 6 } , { 2 , 3 } , { 2 , 4 } , { 2 , 5 } , { 2 , 6 } , { 3 , 4 } , { 3 , 5 } , { 3 , 6 } } ; for ( int i = 0 ; i < 11 ; i ++ ) g [ i ] = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < M ; ++ i ) { int u = edges [ i , 0 ] ; int v = edges [ i , 1 ] ; g [ u ] [ v ] = 1 ; g [ v ] [ u ] = 1 ; }
weightOfMST ( N ) ; } }
using System ; class GFG {
static int countPairs ( int [ ] A , int [ ] B ) { int n = A . Length ; int ans = 0 ; Array . Sort ( A ) ; Array . Sort ( B ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( A [ i ] > B [ ans ] ) { ans ++ ; } } return ans ; }
public static void Main ( ) { int [ ] A = { 30 , 28 , 45 , 22 } ; int [ ] B = { 35 , 25 , 22 , 48 } ; Console . Write ( countPairs ( A , B ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int max_element ( int [ ] arr , int n ) { int max = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { if ( max < arr [ i ] ) max = arr [ i ] ; } return max ; }
static int maxMod ( int [ ] arr , int n ) { int maxVal = max_element ( arr , n ) ; int secondMax = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] < maxVal && arr [ i ] > secondMax ) { secondMax = arr [ i ] ; } } return secondMax ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 4 , 1 , 5 , 3 , 6 } ; int n = arr . Length ; Console . WriteLine ( maxMod ( arr , n ) ) ; } }
using System ; class GFG {
static bool isPossible ( int [ ] A , int [ ] B , int n , int m , int x , int y ) {
if ( x > n y > m ) return false ;
Array . Sort ( A ) ; Array . Sort ( B ) ;
if ( A [ x - 1 ] < B [ m - y ] ) return true ; else return false ; }
public static void Main ( String [ ] args ) { int [ ] A = { 1 , 1 , 1 , 1 , 1 } ; int [ ] B = { 2 , 2 } ; int n = A . Length ; int m = B . Length ; ; int x = 3 , y = 1 ; if ( isPossible ( A , B , n , m , x , y ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG { static int MAX = 100005 ;
static int Min_Replace ( int [ ] arr , int n , int k ) { Array . Sort ( arr ) ;
int [ ] freq = new int [ MAX ] ; int p = 0 ; freq [ p ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] == arr [ i - 1 ] ) ++ freq [ p ] ; else ++ freq [ ++ p ] ; }
Array . Sort ( freq ) ; Array . Reverse ( freq ) ;
int ans = 0 ; for ( int i = k ; i <= p ; i ++ ) ans += freq [ i ] ;
return ans ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 7 , 8 , 2 , 3 , 2 , 3 } ; int n = arr . Length ; int k = 2 ; Console . WriteLine ( Min_Replace ( arr , n , k ) ) ; } }
using System ; class GFG {
static int Segment ( int [ ] x , int [ ] l , int n ) {
if ( n == 1 ) return 1 ;
int ans = 2 ; for ( int i = 1 ; i < n - 1 ; i ++ ) {
if ( x [ i ] - l [ i ] > x [ i - 1 ] ) ans ++ ;
else if ( x [ i ] + l [ i ] < x [ i + 1 ] ) {
x [ i ] = x [ i ] + l [ i ] ; ans ++ ; } }
return ans ; }
public static void Main ( String [ ] args ) { int [ ] x = { 1 , 3 , 4 , 5 , 8 } ; int [ ] l = { 10 , 1 , 2 , 2 , 5 } ; int n = x . Length ;
Console . WriteLine ( Segment ( x , l , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int MinimizeleftOverSum ( int [ ] a , int n ) { List < int > v1 = new List < int > ( ) , v2 = new List < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] % 2 == 1 ) v1 . Add ( a [ i ] ) ; else v2 . Add ( a [ i ] ) ; }
if ( v1 . Count > v2 . Count ) {
v1 . Sort ( ) ; v2 . Sort ( ) ;
int x = v1 . Count - v2 . Count - 1 ; int sum = 0 ; int i = 0 ;
while ( i < x ) { sum += v1 [ i ++ ] ; }
return sum ; }
else if ( v2 . Count > v1 . Count ) {
v1 . Sort ( ) ; v2 . Sort ( ) ;
int x = v2 . Count - v1 . Count - 1 ; int sum = 0 ; int i = 0 ;
while ( i < x ) { sum += v2 [ i ++ ] ; }
return sum ; }
else return 0 ; }
public static void Main ( String [ ] args ) { int [ ] a = { 2 , 2 , 2 , 2 } ; int n = a . Length ; Console . WriteLine ( MinimizeleftOverSum ( a , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void minOperation ( string S , int N , int K ) {
if ( N % K != 0 ) { Console . WriteLine ( " Not ▁ Possible " ) ; } else {
int [ ] count = new int [ 26 ] ; for ( int i = 0 ; i < N ; i ++ ) { count [ ( S [ i ] - 97 ) ] ++ ; } int E = N / K ; List < int > greaterE = new List < int > ( ) ; List < int > lessE = new List < int > ( ) ; for ( int i = 0 ; i < 26 ; i ++ ) {
if ( count [ i ] < E ) lessE . Add ( E - count [ i ] ) ; else greaterE . Add ( count [ i ] - E ) ; } greaterE . Sort ( ) ; lessE . Sort ( ) ; int mi = Int32 . MaxValue ; for ( int i = 0 ; i <= K ; i ++ ) {
int set1 = i ; int set2 = K - i ; if ( greaterE . Count >= set1 && lessE . Count >= set2 ) { int step1 = 0 ; int step2 = 0 ; for ( int j = 0 ; j < set1 ; j ++ ) step1 += greaterE [ j ] ; for ( int j = 0 ; j < set2 ; j ++ ) step2 += lessE [ j ] ; mi = Math . Min ( mi , Math . Max ( step1 , step2 ) ) ; } } Console . WriteLine ( mi ) ; } }
public static void Main ( ) { string S = " accb " ; int N = S . Length ; int K = 2 ; minOperation ( S , N , K ) ; } }
using System ; class GFG {
static int minMovesToSort ( int [ ] arr , int n ) { int moves = 0 ; int i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
} return moves ; }
static public void Main ( ) { int [ ] arr = { 3 , 5 , 2 , 8 , 4 } ; int n = arr . Length ; Console . WriteLine ( minMovesToSort ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static bool [ ] prime = new bool [ 100005 ] ; static void SieveOfEratosthenes ( int n ) { for ( int i = 0 ; i < 100005 ; i ++ ) prime [ i ] = true ;
prime [ 1 ] = false ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i < n ; i += p ) { prime [ i ] = false ; } } } }
static void sortPrimes ( int [ ] arr , int n ) { SieveOfEratosthenes ( 100005 ) ;
List < int > v = new List < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( prime [ arr [ i ] ] ) { v . Add ( arr [ i ] ) ; } } v . Sort ( ) ; v . Reverse ( ) ; int j = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( prime [ arr [ i ] ] ) { arr [ i ] = v [ j ++ ] ; } } }
public static void Main ( String [ ] args ) { int [ ] arr = { 4 , 3 , 2 , 6 , 100 , 17 } ; int n = arr . Length ; sortPrimes ( arr , n ) ;
for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } } }
using System ; public class GFG { static void findOptimalPairs ( int [ ] arr , int N ) { Array . Sort ( arr ) ;
for ( int i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) Console . Write ( " ( " + arr [ i ] + " , ▁ " + arr [ j ] + " ) " + " ▁ " ) ; }
static public void Main ( ) { int [ ] arr = { 9 , 6 , 5 , 1 } ; int N = arr . Length ; findOptimalPairs ( arr , N ) ; } }
using System ; public class GFG {
static int countBits ( int a ) { int count = 0 ; while ( a > 0 ) { if ( ( a & 1 ) > 0 ) count += 1 ; a = a >> 1 ; } return count ; }
static void insertionSort ( int [ ] arr , int [ ] aux , int n ) { for ( int i = 1 ; i < n ; i ++ ) {
int key1 = aux [ i ] ; int key2 = arr [ i ] ; int j = i - 1 ;
while ( j >= 0 && aux [ j ] < key1 ) { aux [ j + 1 ] = aux [ j ] ; arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } aux [ j + 1 ] = key1 ; arr [ j + 1 ] = key2 ; } }
static void sortBySetBitCount ( int [ ] arr , int n ) {
int [ ] aux = new int [ n ] ; for ( int i = 0 ; i < n ; i ++ ) aux [ i ] = countBits ( arr [ i ] ) ;
insertionSort ( arr , aux , n ) ; }
static void printArr ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = arr . Length ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int countBits ( int a ) { int count = 0 ; while ( a > 0 ) { if ( ( a & 1 ) > 0 ) count += 1 ; a = a >> 1 ; } return count ; }
static void sortBySetBitCount ( int [ ] arr , int n ) { List < int > [ ] count = new List < int > [ 32 ] ; for ( int i = 0 ; i < count . Length ; i ++ ) count [ i ] = new List < int > ( ) ; int setbitcount = 0 ; for ( int i = 0 ; i < n ; i ++ ) { setbitcount = countBits ( arr [ i ] ) ; count [ setbitcount ] . Add ( arr [ i ] ) ; }
for ( int i = 31 ; i >= 0 ; i -- ) { List < int > v1 = count [ i ] ; for ( int p = 0 ; p < v1 . Count ; p ++ ) arr [ j ++ ] = v1 [ p ] ; } }
static void printArr ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = arr . Length ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void generateString ( int k1 , int k2 , char [ ] s ) {
int C1s = 0 , C0s = 0 ; int flag = 0 ; List < int > pos = new List < int > ( ) ;
for ( int i = 0 ; i < s . Length ; i ++ ) { if ( s [ i ] == '0' ) { C0s ++ ;
if ( ( i + 1 ) % k1 != 0 && ( i + 1 ) % k2 != 0 ) { pos . Add ( i ) ; } } else { C1s ++ ; } if ( C0s >= C1s ) {
if ( pos . Count == 0 ) { Console . WriteLine ( - 1 ) ; flag = 1 ; break ; }
else { int k = pos [ ( pos . Count - 1 ) ] ; s [ k ] = '1' ; C0s -- ; C1s ++ ; pos . Remove ( pos . Count - 1 ) ; } } }
if ( flag == 0 ) { Console . WriteLine ( s ) ; } }
public static void Main ( ) { int K1 = 2 , K2 = 4 ; string S = "11000100" ; generateString ( K1 , K2 , S . ToCharArray ( ) ) ; } }
using System ; class GFG {
static void maximizeProduct ( int N ) {
int MSB = ( int ) ( Math . Log ( N ) / Math . Log ( 2 ) ) ;
int X = 1 << MSB ;
int Y = N - ( 1 << MSB ) ;
for ( int i = 0 ; i < MSB ; i ++ ) {
if ( ( N & ( 1 << i ) ) == 0 ) {
X += 1 << i ;
Y += 1 << i ; } }
Console . Write ( X + " ▁ " + Y ) ; }
public static void Main ( ) { int N = 45 ; maximizeProduct ( N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool check ( int num ) {
int sm = 0 ;
int num2 = num * num ; while ( num > 0 ) { sm += num % 10 ; num /= 10 ; }
int sm2 = 0 ; while ( num2 > 0 ) { sm2 += num2 % 10 ; num2 /= 10 ; } return ( ( sm * sm ) == sm2 ) ; }
static int convert ( string s ) { int val = 0 ; char [ ] charArray = s . ToCharArray ( ) ; Array . Reverse ( charArray ) ; s = new string ( charArray ) ; int cur = 1 ; for ( int i = 0 ; i < s . Length ; i ++ ) { val += ( ( int ) s [ i ] - ( int ) '0' ) * cur ; cur *= 10 ; } return val ; }
static void generate ( string s , int len , HashSet < int > uniq ) {
if ( s . Length == len ) {
if ( check ( convert ( s ) ) ) { uniq . Add ( convert ( s ) ) ; } return ; }
for ( int i = 0 ; i <= 3 ; i ++ ) { generate ( s + Convert . ToChar ( i + ( int ) '0' ) , len , uniq ) ; } }
static int totalNumbers ( int L , int R ) {
int ans = 0 ;
int max_len = ( int ) Math . Log10 ( R ) + 1 ;
HashSet < int > uniq = new HashSet < int > ( ) ; for ( int i = 1 ; i <= max_len ; i ++ ) {
generate ( " " , i , uniq ) ; }
foreach ( int x in uniq ) { if ( x >= L && x <= R ) { ans ++ ; } } return ans ; }
public static void Main ( ) { int L = 22 , R = 22 ; Console . Write ( totalNumbers ( L , R ) ) ; } }
using System ; class GFG {
static void convertXintoY ( int X , int Y ) {
while ( Y > X ) {
if ( Y % 2 == 0 ) Y /= 2 ;
else if ( Y % 10 == 1 ) Y /= 10 ;
else break ; }
if ( X == Y ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; }
public static void Main ( String [ ] args ) { int X = 100 , Y = 40021 ; convertXintoY ( X , Y ) ; } }
using System ; class GFG {
static void generateString ( int K ) {
string s = " " ;
for ( int i = 97 ; i < 97 + K ; i ++ ) { s = s + ( char ) ( i ) ;
for ( int j = i + 1 ; j < 97 + K ; j ++ ) { s += ( char ) ( i ) ; s += ( char ) ( j ) ; } }
s += ( char ) ( 97 ) ;
Console . Write ( s ) ; }
public static void Main ( ) { int K = 4 ; generateString ( K ) ; } }
using System ; class GFG {
public static void findEquation ( int S , int M ) {
Console . Write ( "1 ▁ " + ( ( - 1 ) * S ) + " ▁ " + M ) ; }
static void Main ( ) { int S = 5 , M = 6 ; findEquation ( S , M ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int minSteps ( List < int > a , int n ) {
int [ ] prefix_sum = new int [ n ] ; prefix_sum [ 0 ] = a [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) prefix_sum [ i ] += prefix_sum [ i - 1 ] + a [ i ] ;
int mx = - 1 ;
foreach ( int subgroupsum in prefix_sum ) { int sum = 0 ; int i = 0 ; int grp_count = 0 ;
while ( i < n ) { sum += a [ i ] ;
if ( sum == subgroupsum ) {
grp_count += 1 ; sum = 0 ; }
else if ( sum > subgroupsum ) { grp_count = - 1 ; break ; } i += 1 ; }
if ( grp_count > mx ) mx = grp_count ; }
return n - mx ; }
public static void Main ( ) { List < int > A = new List < int > ( ) { 1 , 2 , 3 , 2 , 1 , 3 } ; int N = A . Count ;
Console . Write ( minSteps ( A , N ) ) ; } }
using System ; public class GFG {
public static void maxOccuringCharacter ( string s ) {
int count0 = 0 , count1 = 0 ;
for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( s [ i ] == '1' ) { count1 ++ ; }
else if ( s [ i ] == '0' ) { count0 ++ ; } }
int prev = - 1 ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( s [ i ] == '1' ) { prev = i ; break ; } }
for ( int i = prev + 1 ; i < s . Length ; i ++ ) {
if ( s [ i ] != ' X ' ) {
if ( s [ i ] == '1' ) { count1 += i - prev - 1 ; prev = i ; }
else {
bool flag = true ; for ( int j = i + 1 ; j < s . Length ; j ++ ) { if ( s [ j ] == '1' ) { flag = false ; prev = j ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . Length ; } } } }
prev = - 1 ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( s [ i ] == '0' ) { prev = i ; break ; } }
for ( int i = prev + 1 ; i < s . Length ; i ++ ) {
if ( s [ i ] != ' X ' ) {
if ( s [ i ] == '0' ) {
count0 += i - prev - 1 ;
prev = i ; }
else {
bool flag = true ; for ( int j = i + 1 ; j < s . Length ; j ++ ) { if ( s [ j ] == '0' ) { prev = j ; flag = false ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . Length ; } } } }
if ( s [ 0 ] == ' X ' ) {
int count = 0 ; int i = 0 ; while ( s [ i ] == ' X ' ) { count ++ ; i ++ ; }
if ( s [ i ] == '1' ) { count1 += count ; } }
if ( s [ s . Length - 1 ] == ' X ' ) {
int count = 0 ; int i = s . Length - 1 ; while ( s [ i ] == ' X ' ) { count ++ ; i -- ; }
if ( s [ i ] == '0' ) { count0 += count ; } }
if ( count0 == count1 ) { Console . WriteLine ( " X " ) ; }
else if ( count0 > count1 ) { Console . WriteLine ( 0 ) ; }
else Console . ( 1 ) ; }
public static void Main ( string [ ] args ) { string S = " XX10XX10XXX1XX " ; maxOccuringCharacter ( S ) ; } }
using System ; class GFG {
static int maxSheets ( int A , int B ) { int area = A * B ;
int count = 1 ;
while ( area % 2 == 0 ) {
area /= 2 ;
count *= 2 ; } return count ; }
public static void Main ( ) { int A = 5 , B = 10 ; Console . WriteLine ( maxSheets ( A , B ) ) ; } }
using System ; class GFG {
static void findMinMoves ( int a , int b ) {
int ans = 0 ;
if ( a == b || Math . Abs ( a - b ) == 1 ) { ans = a + b ; } else {
int k = Math . Min ( a , b ) ;
int j = Math . Max ( a , b ) ; ans = 2 * k + 2 * ( j - k ) - 1 ; }
Console . Write ( ans ) ; }
public static void Main ( ) {
int a = 3 , b = 5 ;
findMinMoves ( a , b ) ; } }
using System ; class GFG {
static long cntEvenSumPairs ( long X , long Y ) {
long cntXEvenNums = X / 2 ;
long cntXOddNums = ( X + 1 ) / 2 ;
long cntYEvenNums = Y / 2 ;
long cntYOddNums = ( Y + 1 ) / 2 ;
long cntPairs = ( cntXEvenNums * cntYEvenNums ) + ( cntXOddNums * cntYOddNums ) ;
return cntPairs ; }
public static void Main ( string [ ] args ) { long X = 2 ; long Y = 3 ; Console . WriteLine ( cntEvenSumPairs ( X , Y ) ) ; } }
using System ; using System . Collections . Generic ;
class GFG { static int minMoves ( List < int > arr ) { int N = arr . Count ;
if ( N <= 2 ) return 0 ;
int ans = Int32 . MaxValue ;
for ( int i = - 1 ; i <= 1 ; i ++ ) { for ( int j = - 1 ; j <= 1 ; j ++ ) {
int num1 = arr [ 0 ] + i ;
int num2 = arr [ 1 ] + j ; int flag = 1 ; int moves = Math . Abs ( i ) + Math . Abs ( j ) ;
for ( int idx = 2 ; idx < N ; idx ++ ) {
int num = num1 + num2 ;
if ( Math . Abs ( arr [ idx ] - num ) > 1 ) flag = 0 ;
else moves += . ( arr [ idx ] - num ) ; num1 = num2 ; num2 = num ; }
if ( flag != 0 ) ans = Math . Min ( ans , moves ) ; } }
if ( ans == Int32 . MaxValue ) return - 1 ; return ans ; }
public static void Main ( ) { List < int > arr = new List < int > ( ) { 4 , 8 , 9 , 17 , 27 } ; Console . WriteLine ( minMoves ( arr ) ) ; } }
using System ; class GFG {
static void querySum ( int [ ] arr , int N , int [ , ] Q , int M ) {
for ( int i = 0 ; i < M ; i ++ ) { int x = Q [ i , 0 ] ; int y = Q [ i , 1 ] ;
int sum = 0 ;
while ( x < N ) {
sum += arr [ x ] ;
x += y ; } Console . Write ( sum + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 7 , 5 , 4 } ; int [ , ] Q = { { 2 , 1 } , { 3 , 2 } } ; int N = arr . Length ; int M = Q . GetLength ( 0 ) ; querySum ( arr , N , Q , M ) ; } }
using System ; class GFG {
static int findBitwiseORGivenXORAND ( int X , int Y ) { return X + Y ; }
public static void Main ( string [ ] args ) { int X = 5 , Y = 2 ; Console . Write ( findBitwiseORGivenXORAND ( X , Y ) ) ; } }
using System ; class GFG {
static int GCD ( int a , int b ) {
if ( b == 0 ) return a ;
return GCD ( b , a % b ) ; }
static void canReach ( int N , int A , int B , int K ) {
int gcd = GCD ( N , K ) ;
if ( Math . Abs ( A - B ) % gcd == 0 ) { Console . WriteLine ( " Yes " ) ; }
else { Console . WriteLine ( " No " ) ; } }
public static void Main ( ) { int N = 5 , A = 2 , B = 1 , K = 2 ;
canReach ( N , A , B , K ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void countOfSubarray ( int [ ] arr , int N ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ;
int answer = 0 ;
int sum = 0 ;
mp [ 1 ] = 1 ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += arr [ i ] ; if ( mp . ContainsKey ( sum - i ) ) answer += mp [ sum - i ] ;
if ( mp . ContainsKey ( sum - 1 ) ) mp [ sum - 1 ] ++ ; else mp [ sum - 1 ] = 1 ; }
Console . Write ( answer - 2 ) ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 1 , 0 , 2 , 1 , 2 , - 2 , 2 , 4 } ;
int N = arr . Length ;
countOfSubarray ( arr , N ) ; } }
using System ; class GFG {
static int minAbsDiff ( int N ) {
int sumSet1 = 0 ;
int sumSet2 = 0 ;
for ( int i = N ; i > 0 ; i -- ) {
if ( sumSet1 <= sumSet2 ) { sumSet1 += i ; } else { sumSet2 += i ; } } return Math . Abs ( sumSet1 - sumSet2 ) ; }
static void Main ( ) { int N = 6 ; Console . Write ( minAbsDiff ( N ) ) ; } }
using System ; class GFG {
static bool checkDigits ( int n ) {
do { int r = n % 10 ;
if ( r == 3 r == 4 r == 6 r == 7 r == 9 ) return false ; n /= 10 ; } while ( n != 0 ) ; return true ; }
static bool isPrime ( int n ) { if ( n <= 1 ) return false ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; } return true ; }
static bool isAllPrime ( int n ) { return isPrime ( n ) && checkDigits ( n ) ; }
public static void Main ( String [ ] args ) { int N = 101 ; if ( isAllPrime ( N ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static void minCost ( String str , int a , int b ) {
int openUnbalanced = 0 ;
int closedUnbalanced = 0 ;
int openCount = 0 ;
int closedCount = 0 ; for ( int i = 0 ; i < str . Length ; i ++ ) {
if ( str [ i ] == ' ( ' ) { openUnbalanced ++ ; openCount ++ ; }
else {
if ( openUnbalanced == 0 )
closedUnbalanced ++ ;
else
openUnbalanced -- ;
closedCount ++ ; } }
int result = a * ( Math . Abs ( openCount - closedCount ) ) ;
if ( closedCount > openCount ) closedUnbalanced -= ( closedCount - openCount ) ; if ( openCount > closedCount ) openUnbalanced -= ( openCount - closedCount ) ;
result += Math . Min ( a * ( openUnbalanced + closedUnbalanced ) , b * closedUnbalanced ) ;
Console . Write ( result + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { String str = " ) ) ( ) ( ( ) ( ) ( " ; int A = 1 , B = 3 ; minCost ( str , A , B ) ; } }
using System ; class GFG {
public static void countEvenSum ( int low , int high , int k ) {
int even_count = high / 2 - ( low - 1 ) / 2 ; int odd_count = ( high + 1 ) / 2 - low / 2 ; long even_sum = 1 ; long odd_sum = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
long prev_even = even_sum ; long prev_odd = odd_sum ;
even_sum = ( prev_even * even_count ) + ( prev_odd * odd_count ) ;
odd_sum = ( prev_even * odd_count ) + ( prev_odd * even_count ) ; }
Console . WriteLine ( even_sum ) ; }
public static void Main ( String [ ] args ) {
int low = 4 ; int high = 5 ;
int K = 3 ;
countEvenSum ( low , high , K ) ; } }
using System ; class GFG {
public static void count ( int n , int k ) { long count = ( long ) ( Math . Pow ( 10 , k ) - Math . Pow ( 10 , k - 1 ) ) ;
Console . Write ( count ) ; }
public static void Main ( String [ ] args ) { int n = 2 , k = 1 ; count ( n , k ) ; } }
using System ; class GFG {
static int func ( int N , int P ) {
int sumUptoN = ( N * ( N + 1 ) / 2 ) ; int sumOfMultiplesOfP ;
if ( N < P ) { return sumUptoN ; }
else if ( ( N / P ) == 1 ) { return sumUptoN - P + 1 ; }
sumOfMultiplesOfP = ( ( N / P ) * ( 2 * P + ( N / P - 1 ) * P ) ) / 2 ;
return ( sumUptoN + func ( N / P , P ) - sumOfMultiplesOfP ) ; }
public static void Main ( String [ ] args ) {
int N = 10 , P = 5 ;
Console . WriteLine ( func ( N , P ) ) ; } }
using System ; class GFG {
public static void findShifts ( int [ ] A , int N ) {
int [ ] shift = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) {
if ( i == A [ i ] - 1 ) shift [ i ] = 0 ;
else
shift [ i ] = ( A [ i ] - 1 - i + N ) % N ; }
for ( int i = 0 ; i < N ; i ++ ) Console . Write ( shift [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 4 , 3 , 2 , 5 } ; int N = arr . Length ; findShifts ( arr , N ) ; } }
using System ; class GFG {
public static void constructmatrix ( int N ) { bool check = true ; for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( i == j ) { Console . Write ( "1 ▁ " ) ; } else if ( check ) {
Console . Write ( "2 ▁ " ) ; check = false ; } else {
Console . Write ( " - 2 ▁ " ) ; check = true ; } } Console . WriteLine ( ) ; } }
static public void Main ( ) { int N = 5 ; constructmatrix ( N ) ; } }
using System ; class GFG {
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
public static void Main ( ) { int N = 58 , X = 7 ; Console . Write ( getNum ( N , X ) ) ; } }
using System ; class GFG {
static int minPoints ( int n , int m ) { int ans = 0 ;
if ( ( n % 2 != 0 ) && ( m % 2 != 0 ) ) { ans = ( ( n * m ) / 2 ) + 1 ; } else { ans = ( n * m ) / 2 ; }
return ans ; }
public static void Main ( String [ ] args ) {
int N = 5 , M = 7 ;
Console . Write ( minPoints ( N , M ) ) ; } }
using System ; class GFG {
static String getLargestString ( String s , int k ) {
int [ ] frequency_array = new int [ 26 ] ;
for ( int i = 0 ; i < s . Length ; i ++ ) { frequency_array [ s [ i ] - ' a ' ] ++ ; }
String ans = " " ;
for ( int i = 25 ; i >= 0 ; ) {
if ( frequency_array [ i ] > k ) {
int temp = k ; String st = String . Join ( " " , ( char ) ( i + ' a ' ) ) ; while ( temp > 0 ) {
ans += st ; temp -- ; } frequency_array [ i ] -= k ;
int j = i - 1 ; while ( frequency_array [ j ] <= 0 && j >= 0 ) { j -- ; }
if ( frequency_array [ j ] > 0 && j >= 0 ) { String str = String . Join ( " " , ( char ) ( j + ' a ' ) ) ; ans += str ; frequency_array [ j ] -= 1 ; } else {
break ; } }
else if ( frequency_array [ i ] > 0 ) {
int temp = frequency_array [ i ] ; frequency_array [ i ] -= temp ; String st = String . Join ( " " , ( char ) ( i + ' a ' ) ) ; while ( temp > 0 ) { ans += st ; temp -- ; } }
else { i -- ; } } return ans ; }
public static void Main ( String [ ] args ) { String S = " xxxxzza " ; int k = 3 ; Console . Write ( getLargestString ( S , k ) ) ; } }
using System ; using System . Linq ; class GFG {
static int minOperations ( int [ ] a , int [ ] b , int n ) {
int minA = a . Max ( ) ;
for ( int x = minA ; x >= 0 ; x -- ) {
bool check = true ;
int operations = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( x % b [ i ] == a [ i ] % b [ i ] ) { operations += ( a [ i ] - x ) / b [ i ] ; }
else { check = false ; break ; } } if ( check ) return operations ; } return - 1 ; }
public static void Main ( string [ ] args ) { int N = 5 ; int [ ] A = { 5 , 7 , 10 , 5 , 15 } ; int [ ] B = { 2 , 2 , 1 , 3 , 5 } ; Console . WriteLine ( minOperations ( A , B , N ) ) ; } }
using System ; class GFG {
static int getLargestSum ( int N ) {
int max_sum = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { for ( int j = i + 1 ; j <= N ; j ++ ) {
if ( i * j % ( i + j ) == 0 )
max_sum = Math . Max ( max_sum , i + j ) ; } }
return max_sum ; }
public static void Main ( string [ ] args ) { int N = 25 ; int max_sum = getLargestSum ( N ) ; Console . WriteLine ( max_sum ) ; } }
using System ; class GFG {
static int maxSubArraySum ( int [ ] a , int size ) { int max_so_far = int . MinValue , max_ending_here = 0 ;
for ( int i = 0 ; i < size ; i ++ ) { max_ending_here = max_ending_here + a [ i ] ; if ( max_ending_here < 0 ) max_ending_here = 0 ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; } return max_so_far ; }
static int maxSum ( int [ ] a , int n ) {
int S = 0 ; int i ;
for ( i = 0 ; i < n ; i ++ ) S += a [ i ] ; int X = maxSubArraySum ( a , n ) ;
return 2 * X - S ; }
public static void Main ( String [ ] args ) { int [ ] a = { - 1 , - 2 , - 3 } ; int n = a . Length ; int max_sum = maxSum ( a , n ) ; Console . Write ( max_sum ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool isPrime ( int n ) { int flag = 1 ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) { flag = 0 ; break ; } } return ( flag == 1 ? true : false ) ; }
static bool isPerfectSquare ( int x ) {
double sr = Math . Sqrt ( x ) ;
return ( ( sr - Math . Floor ( sr ) ) == 0 ) ; }
static int countInterestingPrimes ( int n ) { int answer = 0 ; for ( int i = 2 ; i <= n ; i ++ ) {
if ( isPrime ( i ) ) {
for ( int j = 1 ; j * j * j * j <= i ; j ++ ) {
if ( isPerfectSquare ( i - j * j * j * j ) ) { answer ++ ; break ; } } } }
return answer ; }
public static void Main ( String [ ] args ) { int N = 10 ; Console . Write ( countInterestingPrimes ( N ) ) ; } }
using System ; class GFG {
static void decBinary ( int [ ] arr , int n ) { int k = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n /= 2 ; } }
static int binaryDec ( int [ ] arr , int n ) { int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
static int maxNum ( int n , int k ) {
int l = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) + 1 ;
int [ ] a = new int [ l ] ; decBinary ( a , n ) ;
int cn = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( a [ i ] == 0 && cn < k ) { a [ i ] = 1 ; cn ++ ; } }
return binaryDec ( a , l ) ; }
public static void Main ( String [ ] args ) { int n = 4 , k = 1 ; Console . WriteLine ( maxNum ( n , k ) ) ; } }
using System ; class GFG {
static void findSubSeq ( int [ ] arr , int n , int sum ) { for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( sum < arr [ i ] ) arr [ i ] = - 1 ;
else sum -= [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != - 1 ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 17 , 25 , 46 , 94 , 201 , 400 } ; int n = arr . Length ; int sum = 272 ; findSubSeq ( arr , n , sum ) ; } }
using System ; class GFG { static int MAX = 26 ;
static char maxAlpha ( String str , int len ) {
int [ ] first = new int [ MAX ] ; int [ ] last = new int [ MAX ] ;
for ( int i = 0 ; i < MAX ; i ++ ) { first [ i ] = - 1 ; last [ i ] = - 1 ; }
for ( int i = 0 ; i < len ; i ++ ) { int index = ( str [ i ] - ' a ' ) ;
if ( first [ index ] == - 1 ) first [ index ] = i ; last [ index ] = i ; }
int ans = - 1 , maxVal = - 1 ;
for ( int i = 0 ; i < MAX ; i ++ ) {
if ( first [ i ] == - 1 ) continue ;
if ( ( last [ i ] - first [ i ] ) > maxVal ) { maxVal = last [ i ] - first [ i ] ; ans = i ; } } return ( char ) ( ans + ' a ' ) ; }
public static void Main ( String [ ] args ) { String str = " abbba " ; int len = str . Length ; Console . Write ( maxAlpha ( str , len ) ) ; } }
using System ; class GFG { static int MAX = 100001 ;
static void find_distinct ( int [ ] a , int n , int q , int [ ] queries ) { int [ ] check = new int [ MAX ] ; int [ ] idx = new int [ MAX ] ; int cnt = 1 ; for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( check [ a [ i ] ] == 0 ) {
idx [ i ] = cnt ; check [ a [ i ] ] = 1 ; cnt ++ ; } else {
idx [ i ] = cnt - 1 ; } }
for ( int i = 0 ; i < q ; i ++ ) { int m = queries [ i ] ; Console . Write ( idx [ m ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 2 , 3 , 1 , 2 , 3 , 4 , 5 } ; int n = a . Length ; int [ ] queries = { 0 , 3 , 5 , 7 } ; int q = queries . Length ; find_distinct ( a , n , q , queries ) ; } }
using System ; class GFG { static int MAX = 24 ;
static int countOp ( int x ) {
int [ ] arr = new int [ MAX ] ; arr [ 0 ] = 1 ; for ( int i = 1 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] * 2 ;
int temp = x ; bool flag = true ;
int ans = 0 ;
int operations = 0 ; bool flag2 = false ; for ( int i = 0 ; i < MAX ; i ++ ) { if ( arr [ i ] - 1 == x ) flag2 = true ;
if ( arr [ i ] > x ) { ans = i ; break ; } }
if ( flag2 ) return 0 ; while ( flag ) {
if ( arr [ ans ] < x ) ans ++ ; operations ++ ;
for ( int i = 0 ; i < MAX ; i ++ ) { int take = x ^ ( arr [ i ] - 1 ) ; if ( take <= arr [ ans ] - 1 ) {
if ( take > temp ) temp = take ; } }
if ( temp == arr [ ans ] - 1 ) { flag = false ; break ; } temp ++ ; operations ++ ; x = temp ; if ( x == arr [ ans ] - 1 ) flag = false ; }
return operations ; }
static public void Main ( ) { int x = 39 ; Console . WriteLine ( countOp ( x ) ) ; } }
using System ; using System . Linq ; class GFG {
static int minOperations ( int [ ] arr , int n ) { int maxi , result = 0 ;
int [ ] freq = new int [ 1000001 ] ; for ( int i = 0 ; i < n ; i ++ ) { int x = arr [ i ] ; freq [ x ] ++ ; }
maxi = arr . Max ( ) ; for ( int i = 1 ; i <= maxi ; i ++ ) { if ( freq [ i ] != 0 ) {
for ( int j = i * 2 ; j <= maxi ; j = j + i ) {
freq [ j ] = 0 ; }
result ++ ; } } return result ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 4 , 2 , 4 , 4 , 4 } ; int n = arr . Length ; Console . WriteLine ( minOperations ( arr , n ) ) ; } }
using System ; class GFG {
static int __gcd ( int a , int b ) { if ( a == 0 ) return b ; return __gcd ( b % a , a ) ; } static int minGCD ( int [ ] arr , int n ) { int minGCD = 0 ;
for ( int i = 0 ; i < n ; i ++ ) minGCD = __gcd ( minGCD , arr [ i ] ) ; return minGCD ; }
static int minLCM ( int [ ] arr , int n ) { int minLCM = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) minLCM = Math . Min ( minLCM , arr [ i ] ) ; return minLCM ; }
public static void Main ( ) { int [ ] arr = { 2 , 66 , 14 , 521 } ; int n = arr . Length ; Console . WriteLine ( " LCM ▁ = ▁ " + minLCM ( arr , n ) + " , ▁ GCD ▁ = ▁ " + minGCD ( arr , n ) ) ; } }
using System ; class GFG {
static String formStringMinOperations ( char [ ] s ) {
int [ ] count = new int [ 3 ] ; foreach ( char c in s ) { count [ ( int ) c - 48 ] += 1 ; }
int [ ] processed = new int [ 3 ] ;
int reqd = ( int ) s . Length / 3 ; for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( count [ s [ i ] - '0' ] == reqd ) { continue ; }
if ( s [ i ] == '0' && count [ 0 ] > reqd && processed [ 0 ] >= reqd ) {
if ( count [ 1 ] < reqd ) { s [ i ] = '1' ; count [ 1 ] ++ ; count [ 0 ] -- ; }
else if ( count [ 2 ] < reqd ) { s [ i ] = '2' ; count [ 2 ] ++ ; count [ 0 ] -- ; } }
if ( s [ i ] == '1' && count [ 1 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = '0' ; count [ 0 ] ++ ; count [ 1 ] -- ; } else if ( count [ 2 ] < reqd && processed [ 1 ] >= reqd ) { s [ i ] = '2' ; count [ 2 ] ++ ; count [ 1 ] -- ; } }
if ( s [ i ] == '2' && count [ 2 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = '0' ; count [ 0 ] ++ ; count [ 2 ] -- ; } else if ( count [ 1 ] < reqd ) { s [ i ] = '1' ; count [ 1 ] ++ ; count [ 2 ] -- ; } }
processed [ s [ i ] - '0' ] ++ ; } return String . Join ( " " , s ) ; }
public static void Main ( String [ ] args ) { String s = "011200" ; Console . WriteLine ( formStringMinOperations ( s . ToCharArray ( ) ) ) ; } }
using System ; class GFG {
static int findMinimumAdjacentSwaps ( int [ ] arr , int N ) {
bool [ ] visited = new bool [ N + 1 ] ; int minimumSwaps = 0 ; for ( int i = 0 ; i < 2 * N ; i ++ ) {
if ( visited [ arr [ i ] ] == false ) { visited [ arr [ i ] ] = true ;
int count = 0 ; for ( int j = i + 1 ; j < 2 * N ; j ++ ) {
if ( visited [ arr [ j ] ] == false ) count ++ ;
else if ( arr [ i ] == arr [ j ] ) += count ; } } } return ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 3 , 1 , 2 } ; int N = arr . Length ; N /= 2 ; Console . WriteLine ( findMinimumAdjacentSwaps ( arr , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool possibility ( Dictionary < int , int > m , int length , string s ) {
int countodd = 0 ; for ( int i = 0 ; i < length ; i ++ ) {
if ( ( m [ s [ i ] - '0' ] & 1 ) != 0 ) countodd ++ ;
if ( countodd > 1 ) return false ; } return true ; }
static void largestPalindrome ( string s ) {
int l = s . Length ;
Dictionary < int , int > m = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < 10 ; i ++ ) m [ i ] = 0 ; for ( int i = 0 ; i < l ; i ++ ) m [ s [ i ] - '0' ] ++ ;
if ( possibility ( m , l , s ) == false ) { Console . Write ( " Palindrome ▁ cannot ▁ be ▁ formed " ) ; return ; }
char [ ] largest = new char [ l ] ;
int front = 0 ;
for ( int i = 9 ; i >= 0 ; i -- ) {
if ( ( m [ i ] & 1 ) != 0 ) {
largest [ l / 2 ] = ( char ) ( i + '0' ) ;
m [ i ] -- ;
while ( m [ i ] > 0 ) { largest [ front ] = ( char ) ( i + '0' ) ; largest [ l - front - 1 ] = ( char ) ( i + '0' ) ; m [ i ] -= 2 ; front ++ ; } } else {
while ( m [ i ] > 0 ) {
largest [ front ] = ( char ) ( i + '0' ) ; largest [ l - front - 1 ] = ( char ) ( i + '0' ) ;
m [ i ] -= 2 ;
front ++ ; } } }
for ( int i = 0 ; i < l ; i ++ ) { Console . Write ( largest [ i ] ) ; } }
public static void Main ( string [ ] args ) { string s = "313551" ; largestPalindrome ( s ) ; } }
using System . IO ; using System ; using System . Collections ; using System . Collections . Generic ; class GFG {
static long swapCount ( string s ) {
List < int > pos = new List < int > ( ) ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( s [ i ] == ' [ ' ) { pos . Add ( i ) ; } }
int count = 0 ;
int p = 0 ;
long sum = 0 ; char [ ] S = s . ToCharArray ( ) ; for ( int i = 0 ; i < S . Length ; i ++ ) {
if ( S [ i ] == ' [ ' ) { ++ count ; ++ p ; } else if ( S [ i ] == ' ] ' ) { -- count ; }
if ( count < 0 ) {
sum += pos [ p ] - i ; char temp = S [ i ] ; S [ i ] = S [ pos [ p ] ] ; S [ pos [ p ] ] = temp ; ++ p ;
count = 1 ; } } return sum ; }
static void Main ( ) { string s = " [ ] ] [ ] [ " ; Console . WriteLine ( swapCount ( s ) ) ; s = " [ [ ] [ ] ] " ; Console . WriteLine ( swapCount ( s ) ) ; } }
using System ; class GFG {
static int minimumCostOfBreaking ( int [ ] X , int [ ] Y , int m , int n ) { int res = 0 ;
Array . Sort < int > ( X , new Comparison < int > ( ( i1 , i2 ) => i2 . CompareTo ( i1 ) ) ) ;
Array . Sort < int > ( Y , new Comparison < int > ( ( i1 , i2 ) => i2 . CompareTo ( i1 ) ) ) ;
int hzntl = 1 , vert = 1 ;
int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( X [ i ] > Y [ j ] ) { res += X [ i ] * vert ;
hzntl ++ ; i ++ ; } else { res += Y [ j ] * hzntl ;
vert ++ ; j ++ ; } }
int total = 0 ; while ( i < m ) total += X [ i ++ ] ; res += total * vert ;
total = 0 ; while ( j < n ) total += Y [ j ++ ] ; res += total * hzntl ; return res ; }
public static void Main ( String [ ] arg ) { int m = 6 , n = 4 ; int [ ] X = { 2 , 1 , 3 , 1 , 4 } ; int [ ] Y = { 4 , 1 , 2 } ; Console . WriteLine ( minimumCostOfBreaking ( X , Y , m - 1 , n - 1 ) ) ; } }
using System ; public class GFG {
static int getMin ( int x , int y , int z ) { return Math . Min ( Math . Min ( x , y ) , z ) ; }
static int editDistance ( string str1 , string str2 , int m , int n ) {
int [ , ] dp = new int [ m + 1 , n + 1 ] ;
for ( int i = 0 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) {
if ( i == 0 )
dp [ i , j ] = j ;
else if ( j = = 0 )
dp [ i , j ] = i ;
else if ( str1 [ i - 1 ] == str2 [ j - 1 ] ) [ i , j ] = dp [ i - 1 , j - 1 ] ;
else {
dp [ i , j ] = 1 + getMin ( dp [ i , j - 1 ] , dp [ i - 1 , j ] , dp [ i - 1 , j - 1 ] ) ; } } }
return dp [ m , n ] ; }
static void minimumSteps ( string S , int N ) {
int ans = int . MaxValue ;
for ( int i = 1 ; i < N ; i ++ ) { string S1 = S . Substring ( 0 , i ) ; string S2 = S . Substring ( i ) ;
int count = editDistance ( S1 , S2 , S1 . Length , S2 . Length ) ;
ans = Math . Min ( ans , count ) ; }
Console . Write ( ans ) ; }
public static void Main ( string [ ] args ) { string S = " aabb " ; int N = S . Length ; minimumSteps ( S , N ) ; } }
using System ; class GFG {
static int minimumOperations ( int N ) {
int [ ] dp = new int [ N + 1 ] ; int i ;
for ( i = 0 ; i <= N ; i ++ ) { dp [ i ] = ( int ) 1e9 ; }
dp [ 2 ] = 0 ;
for ( i = 2 ; i <= N ; i ++ ) {
if ( dp [ i ] == ( int ) 1e9 ) continue ;
if ( i * 5 <= N ) { dp [ i * 5 ] = Math . Min ( dp [ i * 5 ] , dp [ i ] + 1 ) ; }
if ( i + 3 <= N ) { dp [ i + 3 ] = Math . Min ( dp [ i + 3 ] , dp [ i ] + 1 ) ; } }
if ( dp [ N ] == 1e9 ) return - 1 ;
return dp [ N ] ; }
public static void Main ( String [ ] args ) { int N = 25 ; Console . Write ( minimumOperations ( N ) ) ; } }
using System ; class GFG {
static int MaxProfit ( int [ ] arr , int n , int transactionFee ) { int buy = - arr [ 0 ] ; int sell = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { int temp = buy ;
buy = Math . Max ( buy , sell - arr [ i ] ) ; sell = Math . Max ( sell , temp + arr [ i ] - transactionFee ) ; }
return Math . Max ( sell , buy ) ; }
public static void Main ( ) {
int [ ] arr = { 6 , 1 , 7 , 2 , 8 , 4 } ; int n = arr . Length ; int transactionFee = 2 ;
Console . WriteLine ( MaxProfit ( arr , n , transactionFee ) ) ; } }
using System ; class GFG {
static int [ , ] start = new int [ 3 , 3 ] ;
static int [ , ] ending = new int [ 3 , 3 ] ;
static void calculateStart ( int n , int m ) {
for ( int i = 1 ; i < m ; ++ i ) { start [ 0 , i ] += start [ 0 , i - 1 ] ; }
for ( int i = 1 ; i < n ; ++ i ) { start [ i , 0 ] += start [ i - 1 , 0 ] ; }
for ( int i = 1 ; i < n ; ++ i ) { for ( int j = 1 ; j < m ; ++ j ) {
start [ i , j ] += Math . Max ( start [ i - 1 , j ] , start [ i , j - 1 ] ) ; } } }
static void calculateEnd ( int n , int m ) {
for ( int i = n - 2 ; i >= 0 ; -- i ) { ending [ i , m - 1 ] += ending [ i + 1 , m - 1 ] ; }
for ( int i = m - 2 ; i >= 0 ; -- i ) { ending [ n - 1 , i ] += ending [ n - 1 , i + 1 ] ; }
for ( int i = n - 2 ; i >= 0 ; -- i ) { for ( int j = m - 2 ; j >= 0 ; -- j ) {
ending [ i , j ] += Math . Max ( ending [ i + 1 , j ] , ending [ i , j + 1 ] ) ; } } }
static void maximumPathSum ( int [ , ] mat , int n , int m , int q , int [ , ] coordinates ) {
for ( int i = 0 ; i < n ; ++ i ) { for ( int j = 0 ; j < m ; ++ j ) { start [ i , j ] = mat [ i , j ] ; ending [ i , j ] = mat [ i , j ] ; } }
calculateStart ( n , m ) ;
calculateEnd ( n , m ) ;
int ans = 0 ;
for ( int i = 0 ; i < q ; ++ i ) { int X = coordinates [ i , 0 ] - 1 ; int Y = coordinates [ i , 1 ] - 1 ;
ans = Math . Max ( ans , start [ X , Y ] + ending [ X , Y ] - mat [ X , Y ] ) ; }
Console . Write ( ans ) ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 2 , 3 } , { 4 , 5 , 6 } , { 7 , 8 , 9 } } ; int N = 3 ; int M = 3 ; int Q = 2 ; int [ , ] coordinates = { { 1 , 2 } , { 2 , 2 } } ; maximumPathSum ( mat , N , M , Q , coordinates ) ; } }
using System ; class GFG {
static int MaxSubsetlength ( string [ ] arr , int A , int B ) {
int [ , ] dp = new int [ A + 1 , B + 1 ] ;
foreach ( string str in arr ) {
int zeros = 0 , ones = 0 ; foreach ( char ch in str . ToCharArray ( ) ) { if ( ch == '0' ) zeros ++ ; else ones ++ ; }
for ( int i = A ; i >= zeros ; i -- )
for ( int j = B ; j >= ones ; j -- )
dp [ i , j ] = Math . Max ( dp [ i , j ] , dp [ i - zeros , j - ones ] + 1 ) ; }
return dp [ A , B ] ; }
public static void Main ( string [ ] args ) { string [ ] arr = { "1" , "0" , "0001" , "10" , "111001" } ; int A = 5 , B = 3 ; Console . WriteLine ( MaxSubsetlength ( arr , A , B ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int numOfWays ( int [ , ] a , int n , int i , HashSet < int > blue ) {
if ( i == n ) return 1 ;
int count = 0 ;
for ( int j = 0 ; j < n ; j ++ ) {
if ( a [ i , j ] == 1 && ! blue . Contains ( j ) ) { blue . Add ( j ) ; count += numOfWays ( a , n , i + 1 , blue ) ; blue . Remove ( j ) ; } } return count ; }
public static void Main ( ) { int n = 3 ; int [ , ] mat = { { 0 , 1 , 1 } , { 1 , 0 , 1 } , { 1 , 1 , 1 } } ; HashSet < int > mpp = new HashSet < int > ( ) ; Console . WriteLine ( ( numOfWays ( mat , n , 0 , mpp ) ) ) ; } }
using System ; public class GFG {
static void minCost ( int [ ] arr , int n ) {
if ( n < 3 ) { Console . WriteLine ( arr [ 0 ] ) ; return ; }
int [ ] dp = new int [ n ] ;
dp [ 0 ] = arr [ 0 ] ; dp [ 1 ] = dp [ 0 ] + arr [ 1 ] + arr [ 2 ] ;
for ( int i = 2 ; i < n - 1 ; i ++ ) dp [ i ] = Math . Min ( dp [ i - 2 ] + arr [ i ] , dp [ i - 1 ] + arr [ i ] + arr [ i + 1 ] ) ;
dp [ n - 1 ] = Math . Min ( dp [ n - 2 ] , dp [ n - 3 ] + arr [ n - 1 ] ) ;
Console . WriteLine ( dp [ n - 1 ] ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 9 , 4 , 6 , 8 , 5 } ; int N = arr . Length ; minCost ( arr , N ) ; } }
using System ; class GFG { static int M = 1000000007 ;
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
public static void Main ( String [ ] args ) { int n = 2 ; Console . WriteLine ( findValue ( n ) ) ; } }
using System ; class GFG { static readonly long M = 1000000007 ;
static long power ( long X , long Y ) {
long res = 1 ;
X = X % M ;
if ( X == 0 ) return 0 ;
while ( Y > 0 ) {
if ( Y % 2 == 1 ) {
res = ( res * X ) % M ; }
Y = Y >> 1 ;
X = ( X * X ) % M ; } return res ; }
static long findValue ( int N ) {
long [ ] dp = new long [ N + 1 ] ;
dp [ 1 ] = 2 ; dp [ 2 ] = 1024 ;
for ( int i = 3 ; i <= N ; i ++ ) {
int y = ( i & ( - i ) ) ;
int x = i - y ;
if ( x == 0 ) {
dp [ i ] = power ( dp [ i / 2 ] , 10 ) ; } else {
dp [ i ] = ( dp [ x ] * dp [ y ] ) % M ; } } return ( dp [ N ] * dp [ N ] ) % M ; }
public static void Main ( String [ ] args ) { int n = 150 ; Console . Write ( findValue ( n ) ) ; } }
using System ; class GFG {
static int findWays ( int N ) {
if ( N == 0 ) { return 1 ; }
int cnt = 0 ;
for ( int i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i ) ; } }
return cnt ; }
public static void Main ( ) { int N = 4 ;
Console . Write ( findWays ( N ) ) ; } }
using System ; class GFG {
static int checkEqualSumUtil ( int [ ] arr , int N , int sm1 , int sm2 , int sm3 , int j ) {
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; } else {
int l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
int m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
int r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
return Math . Max ( Math . Max ( l , m ) , r ) ; } }
static void checkEqualSum ( int [ ] arr , int N ) {
int sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { Console . Write ( " Yes " ) ; } else { Console . Write ( " No " ) ; } }
public static void Main ( ) {
int [ ] arr = { 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 } ; int N = arr . Length ;
checkEqualSum ( arr , N ) ; } }
using System ; using System . Collections . Generic ; class GFG { static Dictionary < string , int > dp = new Dictionary < string , int > ( ) ;
static int checkEqualSumUtil ( int [ ] arr , int N , int sm1 , int sm2 , int sm3 , int j ) { string s = sm1 . ToString ( ) + " _ " + sm2 . ToString ( ) + j . ToString ( ) ;
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; }
if ( dp . ContainsKey ( s ) ) return dp [ s ] ; else {
int l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
int m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
int r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
dp [ s ] = Math . Max ( Math . Max ( l , m ) , r ) ; return dp [ s ] ; } }
static void checkEqualSum ( int [ ] arr , int N ) {
int sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { Console . Write ( " Yes " ) ; } else { Console . Write ( " No " ) ; } }
public static void Main ( string [ ] args ) {
int [ ] arr = { 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 } ; int N = arr . Length ;
checkEqualSum ( arr , N ) ; } }
using System ; class GFG {
static void precompute ( int [ ] nextpos , int [ ] arr , int N ) {
nextpos [ N - 1 ] = N ; for ( int i = N - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] == arr [ i + 1 ] ) nextpos [ i ] = nextpos [ i + 1 ] ; else nextpos [ i ] = i + 1 ; } }
static void findIndex ( int [ , ] query , int [ ] arr , int N , int Q ) {
int [ ] nextpos = new int [ N ] ; precompute ( nextpos , arr , N ) ; for ( int i = 0 ; i < Q ; i ++ ) { int l , r , x ; l = query [ i , 0 ] ; r = query [ i , 1 ] ; x = query [ i , 2 ] ; int ans = - 1 ;
if ( arr [ l ] != x ) ans = l ;
else {
int d = nextpos [ l ] ;
if ( d <= r ) ans = d ; } Console . Write ( ans + " STRNEWLINE " ) ; } }
public static void Main ( String [ ] args ) { int N , Q ; N = 6 ; Q = 3 ; int [ ] arr = { 1 , 2 , 1 , 1 , 3 , 5 } ; int [ , ] query = { { 0 , 3 , 1 } , { 1 , 5 , 2 } , { 2 , 3 , 1 } } ; findIndex ( query , arr , N , Q ) ; } }
using System ; class GFG { static long mod = 10000000007L ;
static long countWays ( string s , string t , int k ) {
int n = s . Length ;
int a = 0 , b = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { string p = s . Substring ( i , n - i ) + s . Substring ( 0 , i ) ;
if ( p == t ) a ++ ; else b ++ ; }
long [ ] dp1 = new long [ k + 1 ] ; long [ ] dp2 = new long [ k + 1 ] ; if ( s == t ) { dp1 [ 0 ] = 1 ; dp2 [ 0 ] = 0 ; } else { dp1 [ 0 ] = 0 ; dp2 [ 0 ] = 1 ; }
for ( int i = 1 ; i <= k ; i ++ ) { dp1 [ i ] = ( ( dp1 [ i - 1 ] * ( a - 1 ) ) % mod + ( dp2 [ i - 1 ] * a ) % mod ) % mod ; dp2 [ i ] = ( ( dp1 [ i - 1 ] * ( b ) ) % mod + ( dp2 [ i - 1 ] * ( b - 1 ) ) % mod ) % mod ; }
return dp1 [ k ] ; }
public static void Main ( string [ ] args ) {
string S = " ab " , T = " ab " ;
int K = 2 ;
Console . Write ( countWays ( S , T , K ) ) ; } }
using System ; class GFG {
static int minOperation ( int k ) {
int [ ] dp = new int [ k + 1 ] ; for ( int i = 1 ; i <= k ; i ++ ) { dp [ i ] = dp [ i - 1 ] + 1 ;
if ( i % 2 == 0 ) { dp [ i ] = Math . Min ( dp [ i ] , dp [ i / 2 ] + 1 ) ; } } return dp [ k ] ; }
public static void Main ( ) { int K = 12 ; Console . Write ( minOperation ( K ) ) ; } }
using System ; public class GFG {
static int maxSum ( int p0 , int p1 , int [ ] a , int pos , int n ) { if ( pos == n ) { if ( p0 == p1 ) return p0 ; else return 0 ; }
int ans = maxSum ( p0 , p1 , a , pos + 1 , n ) ;
ans = Math . Max ( ans , maxSum ( p0 + a [ pos ] , p1 , a , pos + 1 , n ) ) ;
ans = Math . Max ( ans , maxSum ( p0 , p1 + a [ pos ] , a , pos + 1 , n ) ) ; return ans ; }
public static void Main ( string [ ] args ) {
int n = 4 ; int [ ] a = { 1 , 2 , 3 , 6 } ; Console . WriteLine ( maxSum ( 0 , 0 , a , 0 , n ) ) ; } }
using System ; class GFG { static int INT_MIN = int . MinValue ;
static int maxSum ( int [ ] a , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += a [ i ] ; int limit = 2 * sum + 1 ;
int [ , ] dp = new int [ n + 1 , limit ] ;
for ( int i = 0 ; i < n + 1 ; i ++ ) { for ( int j = 0 ; j < limit ; j ++ ) dp [ i , j ] = INT_MIN ; }
dp [ 0 , sum ] = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 0 ; j < limit ; j ++ ) {
if ( ( j - a [ i - 1 ] ) >= 0 && dp [ i - 1 , j - a [ i - 1 ] ] != INT_MIN ) dp [ i , j ] = Math . Max ( dp [ i , j ] , dp [ i - 1 , j - a [ i - 1 ] ] + a [ i - 1 ] ) ;
if ( ( j + a [ i - 1 ] ) < limit && dp [ i - 1 , j + a [ i - 1 ] ] != INT_MIN ) dp [ i , j ] = Math . Max ( dp [ i , j ] , dp [ i - 1 , j + a [ i - 1 ] ] ) ;
if ( dp [ i - 1 , j ] != INT_MIN ) dp [ i , j ] = Math . Max ( dp [ i , j ] , dp [ i - 1 , j ] ) ; } } return dp [ n , sum ] ; }
public static void Main ( ) { int n = 4 ; int [ ] a = { 1 , 2 , 3 , 6 } ; Console . WriteLine ( maxSum ( a , n ) ) ; } }
using System ; class GFG {
static int [ ] fib = new int [ 100005 ] ;
static void computeFibonacci ( ) { fib [ 0 ] = 1 ; fib [ 1 ] = 1 ; for ( int i = 2 ; i < 100005 ; i ++ ) { fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ; } }
static int countString ( string str ) {
int ans = 1 ; int cnt = 1 ; for ( int i = 1 ; i < str . Length ; i ++ ) {
if ( str [ i ] == str [ i - 1 ] ) { cnt ++ ; }
else { ans = ans * fib [ cnt ] ; cnt = 1 ; } }
ans = ans * fib [ cnt ] ;
return ans ; }
public static void Main ( string [ ] args ) { string str = " abdllldefkkkk " ;
computeFibonacci ( ) ;
Console . WriteLine ( countString ( str ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 1000 ;
static void printGolombSequence ( int N ) {
int [ ] arr = new int [ MAX ] ; for ( int i = 0 ; i < MAX ; i ++ ) arr [ i ] = 0 ;
int cnt = 0 ;
arr [ 0 ] = 0 ; arr [ 1 ] = 1 ;
Dictionary < int , int > M = new Dictionary < int , int > ( ) ;
M . Add ( 2 , 2 ) ;
for ( int i = 2 ; i <= N ; i ++ ) {
if ( cnt == 0 ) { arr [ i ] = 1 + arr [ i - 1 ] ; cnt = M [ arr [ i ] ] ; cnt -- ; }
else { arr [ i ] = arr [ i - 1 ] ; cnt -- ; }
if ( M . ContainsKey ( i ) ) { M [ i ] = arr [ i ] ; } else { M . Add ( i , arr [ i ] ) ; } }
for ( int i = 1 ; i <= N ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } }
static void Main ( ) { int N = 11 ; printGolombSequence ( N ) ; } }
using System ; class GFG {
static int number_of_ways ( int n ) {
int [ ] includes_3 = new int [ n + 1 ] ;
int [ ] not_includes_3 = new int [ n + 1 ] ;
includes_3 [ 3 ] = 1 ; not_includes_3 [ 1 ] = 1 ; not_includes_3 [ 2 ] = 2 ; not_includes_3 [ 3 ] = 3 ;
for ( int i = 4 ; i <= n ; i ++ ) { includes_3 [ i ] = includes_3 [ i - 1 ] + includes_3 [ i - 2 ] + not_includes_3 [ i - 3 ] ; not_includes_3 [ i ] = not_includes_3 [ i - 1 ] + not_includes_3 [ i - 2 ] ; } return includes_3 [ n ] ; }
public static void Main ( String [ ] args ) { int n = 7 ; Console . Write ( number_of_ways ( n ) ) ; } }
using System ; class GFG { static int MAX = 100000 ;
static int [ ] divisors = new int [ MAX ] ;
static void generateDivisors ( int n ) { for ( int i = 1 ; i <= Math . Sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) { if ( n / i == i ) { divisors [ i ] ++ ; } else { divisors [ i ] ++ ; divisors [ n / i ] ++ ; } } } }
static int findMaxMultiples ( int [ ] arr , int n ) {
int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
ans = Math . Max ( divisors [ arr [ i ] ] , ans ) ;
generateDivisors ( arr [ i ] ) ; } return ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 8 , 1 , 28 , 4 , 2 , 6 , 7 } ; int n = arr . Length ; Console . Write ( findMaxMultiples ( arr , n ) ) ; } }
using System ; class GFG { static int n = 3 ; static int maxV = 20 ;
static int [ , , ] dp = new int [ n , n , maxV ] ;
static int [ , , ] v = new int [ n , n , maxV ] ;
static int countWays ( int i , int j , int x , int [ , ] arr ) {
if ( i == n j == n ) { return 0 ; } x = ( x & arr [ i , j ] ) ; if ( x == 0 ) { return 0 ; } if ( i == n - 1 && j == n - 1 ) { return 1 ; }
if ( v [ i , j , x ] == 1 ) { return dp [ i , j , x ] ; } v [ i , j , x ] = 1 ;
dp [ i , j , x ] = countWays ( i + 1 , j , x , arr ) + countWays ( i , j + 1 , x , arr ) ; return dp [ i , j , x ] ; }
public static void Main ( ) { int [ , ] arr = { { 1 , 2 , 1 } , { 1 , 1 , 0 } , { 2 , 1 , 1 } } ; Console . WriteLine ( countWays ( 0 , 0 , arr [ 0 , 0 ] , arr ) ) ; } }
using System ; class GFG { static int N = 3 ;
static int FindMaximumSum ( int ind , int kon , int [ ] a , int [ ] b , int [ ] c , int n , int [ , ] dp ) {
if ( ind == n ) return 0 ;
if ( dp [ ind , kon ] != - 1 ) return dp [ ind , kon ] ; int ans = ( int ) ( - 1e9 + 5 ) ;
if ( kon == 0 ) { ans = Math . Max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = Math . Max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon = = 1 ) { ans = Math . Max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; ans = Math . Max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon = = 2 ) { ans = Math . Max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = Math . Max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; } return dp [ ind , kon ] = ans ; }
public static void Main ( ) { int [ ] a = { 6 , 8 , 2 , 7 , 4 , 2 , 7 } ; int [ ] b = { 7 , 8 , 5 , 8 , 6 , 3 , 5 } ; int [ ] c = { 8 , 3 , 2 , 6 , 8 , 4 , 1 } ; int n = a . Length ; int [ , ] dp = new int [ n , N ] ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) { dp [ i , j ] = - 1 ; } }
int x = FindMaximumSum ( 0 , 0 , a , b , c , n , dp ) ;
int y = FindMaximumSum ( 0 , 1 , a , b , c , n , dp ) ;
int z = FindMaximumSum ( 0 , 2 , a , b , c , n , dp ) ;
Console . WriteLine ( Math . Max ( x , Math . Max ( y , z ) ) ) ; } }
using System ; class GFG { static int mod = 1000000007 ;
static int noOfBinaryStrings ( int N , int k ) { int [ ] dp = new int [ 100002 ] ; for ( int i = 1 ; i <= k - 1 ; i ++ ) { dp [ i ] = 1 ; } dp [ k ] = 2 ; for ( int i = k + 1 ; i <= N ; i ++ ) { dp [ i ] = ( dp [ i - 1 ] + dp [ i - k ] ) % mod ; } return dp [ N ] ; }
public static void Main ( ) { int N = 4 ; int K = 2 ; Console . WriteLine ( noOfBinaryStrings ( N , K ) ) ; } }
using System ; class GFG {
public static int findWaysToPair ( int p ) {
int [ ] dp = new int [ p + 1 ] ; dp [ 1 ] = 1 ; dp [ 2 ] = 2 ;
for ( int i = 3 ; i <= p ; i ++ ) { dp [ i ] = dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ; } return dp [ p ] ; }
public static void Main ( string [ ] args ) { int p = 3 ; Console . WriteLine ( findWaysToPair ( p ) ) ; } }
using System ; class GFG { static int CountWays ( int n ) {
if ( n == 0 ) { return 1 ; } if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 1 + 1 ; }
return CountWays ( n - 1 ) + CountWays ( n - 3 ) ; }
static public void Main ( ) { int n = 5 ; Console . WriteLine ( CountWays ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static List < int > factors ( int n ) {
List < int > v = new List < int > ( ) ; v . Add ( 1 ) ;
for ( int i = 2 ; i <= Math . Sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) { v . Add ( i ) ;
if ( n / i != i ) { v . Add ( n / i ) ; } } }
return v ; }
static Boolean checkAbundant ( int n ) { List < int > v ; int sum = 0 ;
v = factors ( n ) ;
for ( int i = 0 ; i < v . Count ; i ++ ) { sum += v [ i ] ; }
if ( sum > n ) return true ; else return false ; }
static Boolean checkSemiPerfect ( int n ) { List < int > v ;
v = factors ( n ) ;
v . Sort ( ) ; int r = v . Count ;
Boolean [ , ] subset = new Boolean [ r + 1 , n + 1 ] ;
for ( int i = 0 ; i <= r ; i ++ ) subset [ i , 0 ] = true ;
for ( int i = 1 ; i <= n ; i ++ ) subset [ 0 , i ] = false ;
for ( int i = 1 ; i <= r ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) {
if ( j < v [ i - 1 ] ) subset [ i , j ] = subset [ i - 1 , j ] ; else { subset [ i , j ] = subset [ i - 1 , j ] || subset [ i - 1 , j - v [ i - 1 ] ] ; } } }
if ( ( subset [ r , n ] ) == false ) return false ; else return true ; }
static Boolean checkweird ( int n ) { if ( checkAbundant ( n ) == true && checkSemiPerfect ( n ) == false ) return true ; else return false ; }
public static void Main ( String [ ] args ) { int n = 70 ; if ( checkweird ( n ) ) Console . WriteLine ( " Weird ▁ Number " ) ; else Console . WriteLine ( " Not ▁ Weird ▁ Number " ) ; } }
using System ; class GFG {
static int maxSubArraySumRepeated ( int [ ] a , int n , int k ) { int max_so_far = 0 ; int max_ending_here = 0 ; for ( int i = 0 ; i < n * k ; i ++ ) {
max_ending_here = max_ending_here + a [ i % n ] ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; if ( max_ending_here < 0 ) max_ending_here = 0 ; } return max_so_far ; }
public static void Main ( ) { int [ ] a = { 10 , 20 , - 30 , - 1 } ; int n = a . Length ; int k = 3 ; Console . Write ( " Maximum ▁ contiguous ▁ sum ▁ is ▁ " + maxSubArraySumRepeated ( a , n , k ) ) ; } }
using System ; class GFG {
public static int longOddEvenIncSeq ( int [ ] arr , int n ) {
int [ ] lioes = new int [ n ] ;
int maxLen = 0 ;
for ( int i = 0 ; i < n ; i ++ ) lioes [ i ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && ( arr [ i ] + arr [ j ] ) % 2 != 0 && lioes [ i ] < lioes [ j ] + 1 ) lioes [ i ] = lioes [ j ] + 1 ;
for ( int i = 0 ; i < n ; i ++ ) if ( maxLen < lioes [ i ] ) maxLen = lioes [ i ] ;
return maxLen ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 } ; int n = 10 ; Console . Write ( " Longest ▁ Increasing ▁ Odd " + " ▁ Even ▁ Subsequence : ▁ " + longOddEvenIncSeq ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static bool isOperator ( char op ) { return ( op == ' + ' op == ' * ' ) ; }
static void printMinAndMaxValueOfExp ( string exp ) { List < int > num = new List < int > ( ) ; List < char > opr = new List < char > ( ) ; string tmp = " " ;
for ( int i = 0 ; i < exp . Length ; i ++ ) { if ( isOperator ( exp [ i ] ) ) { opr . Add ( exp [ i ] ) ; num . Add ( int . Parse ( tmp ) ) ; tmp = " " ; } else { tmp += exp [ i ] ; } }
num . Add ( int . Parse ( tmp ) ) ; int len = num . Count ; int [ , ] minVal = new int [ len , len ] ; int [ , ] maxVal = new int [ len , len ] ;
for ( int i = 0 ; i < len ; i ++ ) { for ( int j = 0 ; j < len ; j ++ ) { minVal [ i , j ] = Int32 . MaxValue ; maxVal [ i , j ] = 0 ;
if ( i == j ) { minVal [ i , j ] = maxVal [ i , j ] = num [ i ] ; } } }
for ( int L = 2 ; L <= len ; L ++ ) { for ( int i = 0 ; i < len - L + 1 ; i ++ ) { int j = i + L - 1 ; for ( int k = i ; k < j ; k ++ ) { int minTmp = 0 , maxTmp = 0 ;
if ( opr [ k ] == ' + ' ) { minTmp = minVal [ i , k ] + minVal [ k + 1 , j ] ; maxTmp = maxVal [ i , k ] + maxVal [ k + 1 , j ] ; }
else if ( opr [ k ] == ' * ' ) { minTmp = minVal [ i , k ] * minVal [ k + 1 , j ] ; = maxVal [ i , k ] * maxVal [ k + 1 , j ] ; }
if ( minTmp < minVal [ i , j ] ) minVal [ i , j ] = minTmp ; if ( maxTmp > maxVal [ i , j ] ) maxVal [ i , j ] = maxTmp ; } } }
Console . Write ( " Minimum ▁ value ▁ : ▁ " + minVal [ 0 , len - 1 ] + " , ▁ Maximum ▁ value ▁ : ▁ " + maxVal [ 0 , len - 1 ] ) ; }
static public void Main ( ) { string expression = "1 + 2*3 + 4*5" ; printMinAndMaxValueOfExp ( expression ) ; } }
using System ; class GFG {
static int MatrixChainOrder ( int [ ] p , int i , int j ) { if ( i == j ) return 0 ; int min = int . MaxValue ;
for ( int k = i ; k < j ; k ++ ) { int count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 2 , 3 , 4 , 3 } ; int n = arr . Length ; Console . Write ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ; } }
using System ; class GFG { static int [ , ] dp = new int [ 100 , 100 ] ;
static int matrixChainMemoised ( int [ ] p , int i , int j ) { if ( i == j ) { return 0 ; } if ( dp [ i , j ] != - 1 ) { return dp [ i , j ] ; } dp [ i , j ] = Int32 . MaxValue ; for ( int k = i ; k < j ; k ++ ) { dp [ i , j ] = Math . Min ( dp [ i , j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i , j ] ; } static int MatrixChainOrder ( int [ ] p , int n ) { int i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 } ; int n = arr . Length ; for ( int i = 0 ; i < 100 ; i ++ ) { for ( int j = 0 ; j < 100 ; j ++ ) { dp [ i , j ] = - 1 ; } } Console . WriteLine ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , n ) ) ; } }
using System ; class GFG {
static void flipBitsOfAandB ( int A , int B ) {
A = A ^ ( A & B ) ;
B = B ^ ( A & B ) ;
Console . Write ( A + " ▁ " + B ) ; }
public static void Main ( String [ ] args ) { int A = 10 , B = 20 ; flipBitsOfAandB ( A , B ) ; } }
using System ; class GFG {
static int TotalHammingDistance ( int n ) { int i = 1 , sum = 0 ; while ( n / i > 0 ) { sum = sum + n / i ; i = i * 2 ; } return sum ; }
public static void Main ( ) { int N = 9 ; Console . Write ( TotalHammingDistance ( N ) ) ; } }
using System ; class GFG { static readonly int m = 1000000007 ;
static void solve ( long n ) {
long s = 0 ; for ( int l = 1 ; l <= n ; ) {
int r = ( int ) ( n / ( Math . Floor ( ( double ) n / l ) ) ) ; int x = ( ( ( r % m ) * ( ( r + 1 ) % m ) ) / 2 ) % m ; int y = ( ( ( l % m ) * ( ( l - 1 ) % m ) ) / 2 ) % m ; int p = ( int ) ( ( n / l ) % m ) ;
s = ( s + ( ( ( x - y ) % m ) * p ) % m + m ) % m ; s %= m ; l = r + 1 ; }
Console . Write ( ( s + m ) % m ) ; }
public static void Main ( String [ ] args ) { long n = 12 ; solve ( n ) ; } }
using System ; class GFG {
static int min_time_to_cut ( int N ) { if ( N == 0 ) return 0 ;
return ( int ) Math . Ceiling ( Math . Log ( N ) / Math . Log ( 2 ) ) ; }
public static void Main ( ) { int N = 100 ; Console . Write ( min_time_to_cut ( N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int findDistinctSums ( int n ) {
HashSet < int > s = new HashSet < int > ( ) ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) {
s . Add ( i + j ) ; } }
return s . Count ; }
public static void Main ( String [ ] args ) { int N = 3 ; Console . Write ( findDistinctSums ( N ) ) ; } }
using System ; class GFG {
static int printPattern ( int i , int j , int n ) {
if ( j >= n ) { return 0 ; } if ( i >= n ) { return 1 ; }
if ( j == i j == n - 1 - i ) {
if ( i == n - 1 - j ) { Console . Write ( " / " ) ; }
else { Console . Write ( " \ \" ) ; } }
else { Console . Write ( " * " ) ; }
if ( printPattern ( i , j + 1 , n ) == 1 ) { return 1 ; } Console . WriteLine ( ) ;
return printPattern ( i + 1 , 0 , n ) ; }
public static void Main ( String [ ] args ) { int N = 9 ;
printPattern ( 0 , 0 , N ) ; } }
using System ; class GfG {
private static int [ ] zArray ( int [ ] arr ) { int [ ] z ; int n = arr . Length ; z = new int [ n ] ; int r = 0 , l = 0 ;
for ( int k = 1 ; k < n ; k ++ ) {
if ( k > r ) { r = l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; }
else { int k1 = k - l ; if ( z [ k1 ] < r - k + 1 ) z [ k ] = z [ k1 ] ; else { l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; } } } return z ; }
private static int [ ] mergeArray ( int [ ] A , int [ ] B ) { int n = A . Length ; int m = B . Length ; int [ ] z ;
int [ ] c = new int [ n + m + 1 ] ;
for ( int i = 0 ; i < m ; i ++ ) c [ i ] = B [ i ] ;
c [ m ] = int . MaxValue ;
for ( int i = 0 ; i < n ; i ++ ) c [ m + i + 1 ] = A [ i ] ;
z = zArray ( c ) ; return z ; }
private static void findZArray ( int [ ] A , int [ ] B , int n ) { int flag = 0 ; int [ ] z ; z = mergeArray ( A , B ) ;
for ( int i = 0 ; i < z . Length ; i ++ ) { if ( z [ i ] == n ) { Console . Write ( ( i - n - 1 ) + " ▁ " ) ; flag = 1 ; } } if ( flag == 0 ) { Console . WriteLine ( " Not ▁ Found " ) ; } }
public static void Main ( ) { int [ ] A = { 1 , 2 , 3 , 2 , 3 , 2 } ; int [ ] B = { 2 , 3 } ; int n = B . Length ; findZArray ( A , B , n ) ; } }
using System ; class GfG {
static int getCount ( String a , String b ) {
if ( b . Length % a . Length != 0 ) return - 1 ; int count = b . Length / a . Length ;
String str = " " ; for ( int i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str . Equals ( b ) ) return count ; return - 1 ; }
public static void Main ( String [ ] args ) { String a = " geeks " ; String b = " geeksgeeks " ; Console . WriteLine ( getCount ( a , b ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static bool check ( String S1 , String S2 ) {
int n1 = S1 . Length ; int n2 = S2 . Length ;
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n1 ; i ++ ) { if ( mp . ContainsKey ( ( int ) S1 [ i ] ) ) { mp [ ( int ) S1 [ i ] ] = mp [ ( int ) S1 [ i ] ] + 1 ; } else { mp . Add ( ( int ) S1 [ i ] , 1 ) ; } }
for ( int i = 0 ; i < n2 ; i ++ ) {
if ( mp . ContainsKey ( ( int ) S2 [ i ] ) ) { mp [ ( int ) S2 [ i ] ] = mp [ ( int ) S2 [ i ] ] - 1 ; }
else if ( mp . ContainsKey ( S2 [ i ] - 1 ) && mp . ContainsKey ( S2 [ i ] - 2 ) ) { mp [ S2 [ i ] - 1 ] = mp [ S2 [ i ] - 1 ] - 1 ; [ S2 [ i ] - 2 ] = mp [ S2 [ i ] - 2 ] - 1 ; } else { return false ; } } return true ; }
public static void Main ( String [ ] args ) { String S1 = " abbat " ; String S2 = " cat " ;
if ( check ( S1 , S2 ) ) Console . Write ( " YES " ) ; else Console . Write ( " NO " ) ; } }
using System ; class GFG {
public static int countPattern ( string str ) { int len = str . Length ; bool oneSeen = false ;
for ( int i = 0 ; i < len ; i ++ ) { char getChar = str [ i ] ;
if ( getChar == '1' && oneSeen == true ) { if ( str [ i - 1 ] == '0' ) { count ++ ; } }
if ( getChar == '1' && oneSeen == false ) { oneSeen = true ; }
if ( getChar != '0' && str [ i ] != '1' ) { oneSeen = false ; } } return count ; }
public static void Main ( string [ ] args ) { string str = "100001abc101" ; Console . WriteLine ( countPattern ( str ) ) ; } }
using System ; public class GFG {
static string checkIfPossible ( int N , string [ ] arr , string T ) {
int [ ] freqS = new int [ 256 ] ;
int [ ] freqT = new int [ 256 ] ;
foreach ( char ch in T . ToCharArray ( ) ) { freqT [ ch - ' a ' ] ++ ; }
for ( int i = 0 ; i < N ; i ++ ) {
foreach ( char ch in arr [ i ] . ToCharArray ( ) ) { freqS [ ch - ' a ' ] ++ ; } } for ( int i = 0 ; i < 256 ; i ++ ) {
if ( freqT [ i ] == 0 && freqS [ i ] != 0 ) { return " No " ; }
else if ( freqS [ i ] == 0 && freqT [ i ] != 0 ) { return " No " ; }
else if ( freqT [ i ] != 0 && freqS [ i ] != ( freqT [ i ] * N ) ) { return " No " ; } }
return " Yes " ; }
public static void Main ( string [ ] args ) { string [ ] arr = { " abc " , " abb " , " acc " } ; string T = " abc " ; int N = arr . Length ; Console . WriteLine ( checkIfPossible ( N , arr , T ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int groupsOfOnes ( string S , int N ) {
int count = 0 ;
Stack < int > st = new Stack < int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( S [ i ] == '1' ) st . Push ( 1 ) ;
else {
if ( st . Count > 0 ) { count ++ ; while ( st . Count > 0 ) { st . Pop ( ) ; } } } }
if ( st . Count > 0 ) count ++ ;
return count ; }
public static void Main ( ) {
string S = "100110111" ; int N = S . Length ;
Console . Write ( groupsOfOnes ( S , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void generatePalindrome ( string S ) {
Dictionary < char , int > Hash = new Dictionary < char , int > ( ) ;
foreach ( char ch in S ) { if ( Hash . ContainsKey ( ch ) ) Hash [ ch ] ++ ; else Hash . Add ( ch , 1 ) ; }
HashSet < string > st = new HashSet < string > ( ) ;
for ( char i = ' a ' ; i <= ' z ' ; i ++ ) {
if ( Hash . ContainsKey ( i ) && Hash [ i ] == 2 ) {
for ( char j = ' a ' ; j <= ' z ' ; j ++ ) {
string s = " " ; if ( Hash . ContainsKey ( j ) && i != j ) { s += i ; s += j ; s += i ;
st . Add ( s ) ; } } }
if ( Hash . ContainsKey ( i ) && Hash [ i ] >= 3 ) {
for ( char j = ' a ' ; j <= ' z ' ; j ++ ) {
string s = " " ;
if ( Hash . ContainsKey ( j ) ) { s += i ; s += j ; s += i ;
st . Add ( s ) ; } } } }
foreach ( string ans in st ) { Console . WriteLine ( ans ) ; } }
public static void Main ( ) { string S = " ddabdac " ; generatePalindrome ( S ) ; } }
using System ; public class GFG {
static void countOccurrences ( string S , string X , string Y ) {
int count = 0 ;
int N = S . Length , A = X . Length ; int B = Y . Length ; int P = Math . Min ( A , Math . Min ( N , B ) ) ;
for ( int i = 0 ; i < N - P + 1 ; i ++ ) {
if ( S . Substring ( i , Math . Min ( N , B ) ) . Equals ( Y ) ) count ++ ;
if ( S . Substring ( i , Math . Min ( N , A ) ) . Equals ( X ) ) Console . Write ( count + " ▁ " ) ; } }
public static void Main ( string [ ] args ) { string S = " abcdefdefabc " ; string X = " abc " ; string Y = " def " ; countOccurrences ( S , X , Y ) ; } }
using System ; class GFG {
static void DFA ( string str , int N ) {
if ( N <= 1 ) { Console . Write ( " No " ) ; return ; }
int count = 0 ;
if ( str [ 0 ] == ' C ' ) { count ++ ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( str [ i ] == ' A ' str [ i ] == ' B ' ) count ++ ; else break ; } } else {
Console . Write ( " No " ) ; return ; }
if ( count == N ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; }
static public void Main ( ) { string str = " CAABBAAB " ; int N = str . Length ; DFA ( str , N ) ; } }
using System ; class GFG {
static void minMaxDigits ( string str , int N ) {
int [ ] arr = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) arr [ i ] = ( str [ i ] - '0' ) % 3 ;
int zero = 0 , one = 0 , two = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( arr [ i ] == 0 ) zero ++ ; if ( arr [ i ] == 1 ) one ++ ; if ( arr [ i ] == 2 ) two ++ ; }
int sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) { sum = ( sum + arr [ i ] ) % 3 ; }
if ( sum == 0 ) { Console . Write ( 0 + " ▁ " ) ; } if ( sum == 1 ) { if ( ( one != 0 ) && ( N > 1 ) ) Console . Write ( 1 + " ▁ " ) ; else if ( two > 1 && N > 2 ) Console . Write ( 2 + " ▁ " ) ; else Console . Write ( - 1 + " ▁ " ) ; } if ( sum == 2 ) { if ( two != 0 && N > 1 ) Console . Write ( 1 + " ▁ " ) ; else if ( one > 1 && N > 2 ) Console . Write ( 2 + " ▁ " ) ; else Console . Write ( - 1 + " ▁ " ) ; }
if ( zero > 0 ) Console . Write ( N - 1 + " ▁ " ) ; else if ( one > 0 && two > 0 ) Console . Write ( N - 2 + " ▁ " ) ; else if ( one > 2 two > 2 ) Console . Write ( N - 3 + " ▁ " ) ; else Console . Write ( - 1 + " ▁ " ) ; }
public static void Main ( ) { string str = "12345" ; int N = str . Length ;
minMaxDigits ( str , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int findMinimumChanges ( int N , int K , char [ ] S ) {
int ans = 0 ;
for ( int i = 0 ; i < ( K + 1 ) / 2 ; i ++ ) {
Dictionary < char , int > mp = new Dictionary < char , int > ( ) ;
for ( int j = i ; j < N ; j += K ) {
if ( mp . ContainsKey ( S [ j ] ) ) { mp [ S [ j ] ] ++ ; } else { mp . Add ( S [ j ] , 1 ) ; } }
for ( int j = N - i - 1 ; j >= 0 ; j -= K ) {
if ( K % 2 == 1 && i == K / 2 ) break ;
if ( mp . ContainsKey ( S [ j ] ) ) { mp [ S [ j ] ] ++ ; } else { mp . Add ( S [ j ] , 1 ) ; } }
int curr_max = int . MinValue ; foreach ( KeyValuePair < char , int > p in mp ) { curr_max = Math . Max ( curr_max , p . Value ) ; }
if ( ( K % 2 == 1 ) && i == K / 2 ) ans += ( N / K - curr_max ) ;
else ans += ( N / K * 2 - curr_max ) ; }
return ans ; }
public static void Main ( String [ ] args ) { String S = " aabbcbbcb " ; int N = S . Length ; int K = 3 ;
Console . Write ( findMinimumChanges ( N , K , S . ToCharArray ( ) ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static String checkString ( String s , int K ) { int n = s . Length ;
Dictionary < char , int > mp = new Dictionary < char , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { if ( mp . ContainsKey ( s [ i ] ) ) mp [ s [ i ] ] = i ; else mp . Add ( s [ i ] , i ) ; } int f = 0 ;
HashSet < char > st = new HashSet < char > ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
st . Add ( s [ i ] ) ;
if ( st . Count > K ) { f = 1 ; break ; }
if ( mp [ s [ i ] ] == i ) st . Remove ( s [ i ] ) ; } return ( f == 1 ? " Yes " : " No " ) ; }
public static void Main ( String [ ] args ) { String s = " aabbcdca " ; int k = 2 ; Console . WriteLine ( checkString ( s , k ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static void distinct ( string [ ] S , int M ) { int count = 0 ;
for ( int i = 0 ; i < S . Length ; i ++ ) {
HashSet < char > set = new HashSet < char > ( ) ; for ( int j = 0 ; j < S [ i ] . Length ; j ++ ) { if ( ! set . Contains ( S [ i ] [ j ] ) ) set . Add ( S [ i ] [ j ] ) ; } int c = set . Count ;
if ( c <= M ) count += 1 ; } Console . Write ( count ) ; }
public static void Main ( string [ ] args ) { string [ ] S = { " HERBIVORES " , " AEROPLANE " , " GEEKSFORGEEKS " } ; int M = 7 ; distinct ( S , M ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static string removeOddFrequencyCharacters ( string s ) {
Dictionary < char , int > m = new Dictionary < char , int > ( ) ; for ( int i = 0 ; i < s . Length ; i ++ ) { char p = s [ i ] ; if ( m . ContainsKey ( p ) ) { m [ p ] ++ ; } else { m [ p ] = 1 ; } }
string new_string = " " ;
for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( ( m [ s [ i ] ] & 1 ) == 1 ) continue ;
new_string += s [ i ] ; }
return new_string ; }
public static void Main ( string [ ] args ) { string str = " geeksforgeeks " ;
str = removeOddFrequencyCharacters ( str ) ; Console . Write ( str ) ; } }
using System ; class GFG { static int i ;
static int productAtKthLevel ( String tree , int k , int level ) { if ( tree [ i ++ ] == ' ( ' ) {
if ( tree [ i ] == ' ) ' ) return 1 ; int product = 1 ;
if ( level == k ) product = tree [ i ] - '0' ;
++ i ; int leftproduct = productAtKthLevel ( tree , k , level + 1 ) ;
++ i ; int rightproduct = productAtKthLevel ( tree , k , level + 1 ) ;
++ i ; return product * leftproduct * rightproduct ; } return int . MinValue ; }
public static void Main ( String [ ] args ) { String tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) " + " ( 9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; i = 0 ; Console . Write ( productAtKthLevel ( tree , k , 0 ) ) ; } }
using System ; class GFG {
static void findMostOccurringChar ( string [ ] str ) {
int [ ] hash = new int [ 26 ] ;
for ( int i = 0 ; i < str . Length ; i ++ ) {
for ( int j = 0 ; j < str [ i ] . Length ; j ++ ) {
hash [ str [ i ] [ j ] - 97 ] ++ ; } }
int max = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) { max = hash [ i ] > hash [ max ] ? i : max ; } Console . Write ( ( char ) ( max + 97 ) + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) {
string [ ] str = { " animal " , " zebra " , " lion " , " giraffe " } ; findMostOccurringChar ( str ) ; } }
using System ; class GFG {
public static bool isPalindrome ( float num ) {
string s = num . ToString ( ) ;
int low = 0 ; int high = s . Length - 1 ; while ( low < high ) {
if ( s [ low ] != s [ high ] ) return false ;
low ++ ; high -- ; } return true ; }
public static void Main ( ) { float n = 123.321f ; if ( isPalindrome ( n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG { readonly static int MAX = 26 ;
static int maxSubStr ( char [ ] str1 , int len1 , char [ ] str2 , int len2 ) {
if ( len1 > len2 ) return 0 ;
int [ ] freq1 = new int [ MAX ] ; for ( int i = 0 ; i < len1 ; i ++ ) freq1 [ i ] = 0 ; for ( int i = 0 ; i < len1 ; i ++ ) freq1 [ str1 [ i ] - ' a ' ] ++ ;
int [ ] freq2 = new int [ MAX ] ; for ( int i = 0 ; i < len2 ; i ++ ) freq2 [ i ] = 0 ; for ( int i = 0 ; i < len2 ; i ++ ) freq2 [ str2 [ i ] - ' a ' ] ++ ;
int minPoss = int . MaxValue ; for ( int i = 0 ; i < MAX ; i ++ ) {
if ( freq1 [ i ] == 0 ) continue ;
if ( freq1 [ i ] > freq2 [ i ] ) return 0 ;
minPoss = Math . Min ( minPoss , freq2 [ i ] / freq1 [ i ] ) ; } return minPoss ; }
public static void Main ( String [ ] args ) { String str1 = " geeks " , str2 = " gskefrgoekees " ; int len1 = str1 . Length ; int len2 = str2 . Length ; Console . WriteLine ( maxSubStr ( str1 . ToCharArray ( ) , len1 , str2 . ToCharArray ( ) , len2 ) ) ; } }
using System ; class GFG {
static int cntWays ( String str , int n ) { int x = n + 1 ; int ways = x * x * ( x * x - 1 ) / 12 ; return ways ; }
public static void Main ( String [ ] args ) { String str = " ab " ; int n = str . Length ; Console . WriteLine ( cntWays ( str , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static HashSet < String > uSet = new HashSet < String > ( ) ;
static int minCnt = int . MaxValue ;
static void findSubStr ( String str , int cnt , int start ) {
if ( start == str . Length ) {
minCnt = Math . Min ( cnt , minCnt ) ; }
for ( int len = 1 ; len <= ( str . Length - start ) ; len ++ ) {
String subStr = str . Substring ( start , len ) ;
if ( uSet . Contains ( subStr ) ) {
findSubStr ( str , cnt + 1 , start + len ) ; } } }
static void findMinSubStr ( String [ ] arr , int n , String str ) {
for ( int i = 0 ; i < n ; i ++ ) uSet . Add ( arr [ i ] ) ;
findSubStr ( str , 0 , 0 ) ; }
public static void Main ( String [ ] args ) { String str = "123456" ; String [ ] arr = { "1" , "12345" , "2345" , "56" , "23" , "456" } ; int n = arr . Length ; findMinSubStr ( arr , n , str ) ; Console . WriteLine ( minCnt ) ; } }
using System ; public class GFG {
static int countSubStr ( String s , int n ) { int c1 = 0 , c2 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i < n - 5 && " geeks " . Equals ( s . Substring ( i , 5 ) ) ) { c1 ++ ; }
if ( i < n - 3 && " for " . Equals ( s . Substring ( i , 3 ) ) ) { c2 = c2 + c1 ; } } return c2 ; }
public static void Main ( String [ ] args ) { String s = " geeksforgeeksisforgeeks " ; int n = s . Length ; Console . WriteLine ( countSubStr ( s , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static void Main ( ) {
string String = " { [ ( ) ] } [ ] " ;
char [ ] lst1 = { ' { ' , ' ( ' , ' [ ' } ;
char [ ] lst2 = { ' } ' , ' ) ' , ' ] ' } ;
List < char > lst = new List < char > ( ) ;
Dictionary < char , char > Dict = new Dictionary < char , char > ( ) ; Dict [ ' ) ' ] = ' ( ' ; Dict [ ' } ' ] = ' { ' ; Dict [ ' ] ' ] = ' [ ' ; int a = 0 , b = 0 , c = 0 ;
if ( Array . Exists ( lst2 , element => element == String [ 0 ] ) ) { Console . WriteLine ( 1 ) ; } else { int k = 0 ;
for ( int i = 0 ; i < String . Length ; i ++ ) { if ( Array . Exists ( lst1 , element => element == String [ i ] ) ) { lst . Add ( String [ i ] ) ; k = i + 2 ; } else {
if ( lst . Count == 0 && Array . Exists ( lst2 , element => element == String [ i ] ) ) { Console . WriteLine ( ( i + 1 ) ) ; c = 1 ; break ; } else {
if ( lst . Count > 0 && Dict [ String [ i ] ] == lst [ lst . Count - 1 ] ) { lst . RemoveAt ( lst . Count - 1 ) ; } else {
a = 1 ; break ; } } } }
if ( lst . Count == 0 && c == 0 ) { Console . WriteLine ( 0 ) ; b = 1 ; } if ( a == 0 && b == 0 && c == 0 ) { Console . WriteLine ( k ) ; } } } }
using System ; class GFG { static int MAX = 26 ;
public static char [ ] encryptStr ( String str , int n , int x ) {
x = x % MAX ; char [ ] arr = str . ToCharArray ( ) ;
int [ ] freq = new int [ MAX ] ; for ( int i = 0 ; i < n ; i ++ ) freq [ arr [ i ] - ' a ' ] ++ ; for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ arr [ i ] - ' a ' ] % 2 == 0 ) { int pos = ( arr [ i ] - ' a ' + x ) % MAX ; arr [ i ] = ( char ) ( pos + ' a ' ) ; }
else { int pos = ( arr [ i ] - ' a ' - x ) ; if ( pos < 0 ) pos += MAX ; arr [ i ] = ( char ) ( pos + ' a ' ) ; } }
return arr ; }
public static void Main ( String [ ] args ) { String s = " abcda " ; int n = s . Length ; int x = 3 ; Console . WriteLine ( encryptStr ( s , n , x ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static Boolean isPossible ( char [ ] str ) {
Dictionary < char , int > freq = new Dictionary < char , int > ( ) ;
int max_freq = 0 ; for ( int j = 0 ; j < ( str . Length ) ; j ++ ) { if ( freq . ContainsKey ( str [ j ] ) ) { var v = freq [ str [ j ] ] + 1 ; freq . Remove ( str [ j ] ) ; freq . Add ( str [ j ] , v ) ; if ( freq [ str [ j ] ] > max_freq ) max_freq = freq [ str [ j ] ] ; } else { freq . Add ( str [ j ] , 1 ) ; if ( freq [ str [ j ] ] > max_freq ) max_freq = freq [ str [ j ] ] ; } }
if ( max_freq <= ( str . Length - max_freq + 1 ) ) return true ; return false ; }
public static void Main ( String [ ] args ) { String str = " geeksforgeeks " ; if ( isPossible ( str . ToCharArray ( ) ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static void printUncommon ( string str1 , string str2 ) { int a1 = 0 , a2 = 0 ; for ( int i = 0 ; i < str1 . Length ; i ++ ) {
int ch = ( str1 [ i ] - ' a ' ) ;
a1 = a1 | ( 1 << ch ) ; } for ( int i = 0 ; i < str2 . Length ; i ++ ) {
int ch = ( str2 [ i ] - ' a ' ) ;
a2 = a2 | ( 1 << ch ) ; }
int ans = a1 ^ a2 ; int j = 0 ; while ( j < 26 ) { if ( ans % 2 == 1 ) { Console . Write ( ( char ) ( ' a ' + j ) ) ; } ans = ans / 2 ; j ++ ; } }
public static void Main ( ) { string str1 = " geeksforgeeks " ; string str2 = " geeksquiz " ; printUncommon ( str1 , str2 ) ; } }
using System ; class GFG {
static int countMinReversals ( String expr ) { int len = expr . Length ;
if ( len % 2 != 0 ) return - 1 ;
int ans = 0 ; int i ;
int open = 0 ;
int close = 0 ; for ( i = 0 ; i < len ; i ++ ) {
if ( expr [ i ] == ' { ' ) open ++ ;
else { if ( open == 0 ) close ++ ; else open -- ; } } ans = ( close / 2 ) + ( open / 2 ) ;
close %= 2 ; open %= 2 ; if ( close != 0 ) ans += 2 ; return ans ; }
public static void Main ( String [ ] args ) { String expr = " } } { { " ; Console . WriteLine ( countMinReversals ( expr ) ) ; } }
using System ; class GfG {
static int totalPairs ( String s1 , String s2 ) { int a1 = 0 , b1 = 0 ;
for ( int i = 0 ; i < s1 . Length ; i ++ ) { if ( ( int ) s1 [ i ] % 2 != 0 ) a1 ++ ; else b1 ++ ; } int a2 = 0 , b2 = 0 ;
for ( int i = 0 ; i < s2 . Length ; i ++ ) { if ( ( int ) s2 [ i ] % 2 != 0 ) a2 ++ ; else b2 ++ ; }
return ( ( a1 * a2 ) + ( b1 * b2 ) ) ; }
public static void Main ( String [ ] args ) { String s1 = " geeks " , s2 = " for " ; Console . WriteLine ( totalPairs ( s1 , s2 ) ) ; } }
using System ; class GFG {
static int prefixOccurrences ( string str ) { char c = str [ 0 ] ; int countc = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( str [ i ] == c ) countc ++ ; } return countc ; }
public static void Main ( ) { string str = " abbcdabbcd " ; Console . WriteLine ( prefixOccurrences ( str ) ) ; } }
using System ; class GFG {
static int minOperations ( string s , string t , int n ) { int ct0 = 0 , ct1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == t [ i ] ) continue ;
if ( s [ i ] == '0' ) ct0 ++ ;
else ct1 ++ ; } return Math . Max ( ct0 , ct1 ) ; }
public static void Main ( ) { string s = "010" , t = "101" ; int n = s . Length ; Console . Write ( minOperations ( s , t , n ) ) ; } }
using System ; class GFG {
static string decryptString ( string str , int n ) {
int i = 0 , jump = 1 ; string decryptedStr = " " ; while ( i < n ) { decryptedStr += str [ i ] ; i += jump ;
jump ++ ; } return decryptedStr ; }
public static void Main ( ) { string str = " geeeeekkkksssss " ; int n = str . Length ; Console . Write ( decryptString ( str , n ) ) ; } }
using System ; class GfG {
static char bitToBeFlipped ( String s ) {
char last = s [ s . Length - 1 ] ; char first = s [ 0 ] ;
if ( last == first ) { if ( last == '0' ) { return '1' ; } else { return '0' ; } }
else if ( last != first ) { return last ; } return last ; }
public static void Main ( ) { string s = "1101011000" ; Console . WriteLine ( bitToBeFlipped ( s ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void SieveOfEratosthenes ( bool [ ] prime , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i < p_size ; i += p ) { prime [ i ] = false ; } } } }
static void sumProdOfPrimeFreq ( char [ ] s ) { int i ; bool [ ] prime = new bool [ s . Length + 1 ] ; for ( i = 0 ; i < s . Length + 1 ; i ++ ) { prime [ i ] = true ; } SieveOfEratosthenes ( prime , s . Length + 1 ) ;
Dictionary < char , int > mp = new Dictionary < char , int > ( ) ; for ( i = 0 ; i < s . Length ; i ++ ) { if ( mp . ContainsKey ( s [ i ] ) ) { var val = mp [ s [ i ] ] ; mp . Remove ( s [ i ] ) ; mp . Add ( s [ i ] , val + 1 ) ; } else { mp . Add ( s [ i ] , 1 ) ; } } int sum = 0 , product = 1 ;
foreach ( KeyValuePair < char , int > it in mp ) {
if ( prime [ it . Value ] ) { sum += it . Value ; product *= it . Value ; } } Console . Write ( " Sum ▁ = ▁ " + sum ) ; Console . WriteLine ( " Product = " }
public static void Main ( String [ ] args ) { String s = " geeksforgeeks " ; sumProdOfPrimeFreq ( s . ToCharArray ( ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static Boolean multipleOrFactor ( String s1 , String s2 ) {
Dictionary < char , int > m1 = new Dictionary < char , int > ( ) ; Dictionary < char , int > m2 = new Dictionary < char , int > ( ) ; for ( int i = 0 ; i < s1 . Length ; i ++ ) { if ( m1 . ContainsKey ( s1 [ i ] ) ) { var x = m1 [ s1 [ i ] ] ; m1 [ s1 [ i ] ] = ++ x ; } else m1 . Add ( s1 [ i ] , 1 ) ; } for ( int i = 0 ; i < s2 . Length ; i ++ ) { if ( m2 . ContainsKey ( s2 [ i ] ) ) { var x = m2 [ s2 [ i ] ] ; m2 [ s2 [ i ] ] = ++ x ; } else m2 . Add ( s2 [ i ] , 1 ) ; } foreach ( KeyValuePair < char , int > entry in m1 ) {
if ( ! m2 . ContainsKey ( entry . Key ) ) continue ;
if ( m2 [ entry . Key ] != 0 && ( m2 [ entry . Key ] % entry . Value == 0 entry . Value % m2 [ entry . Key ] == 0 ) ) continue ;
else return false ; } return true ; }
public static void Main ( String [ ] args ) { String s1 = " geeksforgeeks " , s2 = " geeks " ; if ( multipleOrFactor ( s1 , s2 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void solve ( String s ) {
Dictionary < char , int > m = new Dictionary < char , int > ( ) ; for ( int i = 0 ; i < s . Length ; i ++ ) { if ( m . ContainsKey ( s [ i ] ) ) { var val = m [ s [ i ] ] ; m . Remove ( s [ i ] ) ; m . Add ( s [ i ] , val + 1 ) ; } else m . Add ( s [ i ] , 1 ) ; }
String new_string = " " ;
for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( m [ s [ i ] ] % 2 == 0 ) continue ;
new_string = new_string + s [ i ] ; }
Console . WriteLine ( new_string ) ; }
public static void Main ( String [ ] args ) { String s = " aabbbddeeecc " ;
solve ( s ) ; } }
using System ; class GFG {
static bool isPalindrome ( string str ) { int i = 0 , j = str . Length - 1 ;
while ( i < j ) {
if ( str [ i ++ ] != str [ j -- ] ) return false ; }
return true ; }
static String removePalinWords ( string str ) {
string final_str = " " , word = " " ;
str = str + " ▁ " ; int n = str . Length ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str [ i ] != ' ▁ ' ) word = word + str [ i ] ; else {
if ( ! ( isPalindrome ( word ) ) ) final_str += word + " ▁ " ;
word = " " ; } }
return final_str ; }
public static void Main ( ) { string str = " Text ▁ contains ▁ malayalam ▁ " + " and ▁ level ▁ words " ; Console . WriteLine ( removePalinWords ( str ) ) ; } }
using System ; class GFG {
static int findSubSequence ( string s , int num ) {
int res = 0 ;
int i = 0 ; while ( num > 0 ) {
if ( ( num & 1 ) == 1 ) res += s [ i ] - '0' ; i ++ ;
num = num >> 1 ; } return res ; }
static int combinedSum ( string s ) {
int n = s . Length ;
int c_sum = 0 ;
int range = ( 1 << n ) - 1 ;
for ( int i = 0 ; i <= range ; i ++ ) c_sum += findSubSequence ( s , i ) ;
return c_sum ; }
public static void Main ( ) { string s = "123" ; Console . Write ( combinedSum ( s ) ) ; } }
using System ; class GFG { static int MAX_CHAR = 26 ;
static void findSubsequence ( string str , int k ) {
int [ ] a = new int [ MAX_CHAR ] ;
for ( int i = 0 ; i < str . Length ; i ++ ) a [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < str . Length ; i ++ ) if ( a [ str [ i ] - ' a ' ] >= k ) Console . Write ( str [ i ] ) ; }
public static void Main ( ) { int k = 2 ; findSubsequence ( " geeksforgeeks " , k ) ; } }
using System ; class GFG { static string convert ( string str ) {
string w = " " , z = " " ;
str = str . ToUpper ( ) + " ▁ " ; for ( int i = 0 ; i < str . Length ; i ++ ) {
char ch = str [ i ] ; if ( ch != ' ▁ ' ) w = w + ch ; else {
z = z + ( Char . ToLower ( w [ 0 ] ) ) + w . Substring ( 1 ) + " ▁ " ; w = " " ; } } return z ; }
static void Main ( ) { string str = " I ▁ got ▁ intern ▁ at ▁ geeksforgeeks " ; Console . WriteLine ( convert ( str ) ) ; } }
using System ; class GFG {
static bool isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
static String encryptString ( char [ ] s , int n , int k ) {
int [ ] cv = new int [ n ] ; int [ ] cc = new int [ n ] ; if ( isVowel ( s [ 0 ] ) ) cv [ 0 ] = 1 ; else cc [ 0 ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { cv [ i ] = cv [ i - 1 ] + ( isVowel ( s [ i ] ) == true ? 1 : 0 ) ; cc [ i ] = cc [ i - 1 ] + ( isVowel ( s [ i ] ) == true ? 0 : 1 ) ; } String ans = " " ; int prod = 0 ; prod = cc [ k - 1 ] * cv [ k - 1 ] ; ans += String . Join ( " " , prod ) ;
for ( int i = k ; i < s . Length ; i ++ ) { prod = ( cc [ i ] - cc [ i - k ] ) * ( cv [ i ] - cv [ i - k ] ) ; ans += String . Join ( " " , prod ) ; } return ans ; }
public static void Main ( String [ ] args ) { String s = " hello " ; int n = s . Length ; int k = 2 ; Console . Write ( encryptString ( s . ToCharArray ( ) , n , k ) + " STRNEWLINE " ) ; } }
using System ; class GFG { static int countOccurrences ( string str , string word ) {
string [ ] a = str . Split ( ' ▁ ' ) ;
int count = 0 ; for ( int i = 0 ; i < a . Length ; i ++ ) {
if ( word . Equals ( a [ i ] ) ) count ++ ; } return count ; }
public static void Main ( ) { string str = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " ; string word = " portal " ; Console . Write ( countOccurrences ( str , word ) ) ; } }
using System ; class PermuteString {
static void permute ( String input ) { int n = input . Length ;
int max = 1 << n ;
input = input . ToLower ( ) ;
for ( int i = 0 ; i < max ; i ++ ) { char [ ] combination = input . ToCharArray ( ) ;
for ( int j = 0 ; j < n ; j ++ ) { if ( ( ( i >> j ) & 1 ) == 1 ) combination [ j ] = ( char ) ( combination [ j ] - 32 ) ; }
Console . Write ( combination ) ; Console . Write ( " ▁ " ) ; } }
public static void Main ( ) { permute ( " ABC " ) ; } }
using System ; public class GFG {
static public void printString ( string str , char ch , int count ) { int occ = 0 , i ;
if ( count == 0 ) { Console . WriteLine ( str ) ; return ; }
for ( i = 0 ; i < str . Length ; i ++ ) {
if ( str [ i ] == ch ) occ ++ ;
if ( occ == count ) break ; }
if ( i < str . Length - 1 ) Console . WriteLine ( str . Substring ( i + 1 ) ) ;
else Console . ( " Empty string " ) ; }
static public void Main ( ) { string str = " geeks ▁ for ▁ geeks " ; printString ( str , ' e ' , 2 ) ; } }
using System ; class GFG {
static Boolean isVowel ( char c ) { return ( c == ' a ' c == ' A ' c == ' e ' c == ' E ' c == ' i ' c == ' I ' c == ' o ' c == ' O ' c == ' u ' c == ' U ' ) ; }
static String reverseVowel ( String str ) {
int i = 0 ; int j = str . Length - 1 ; char [ ] str1 = str . ToCharArray ( ) ; while ( i < j ) { if ( ! isVowel ( str1 [ i ] ) ) { i ++ ; continue ; } if ( ! isVowel ( str1 [ j ] ) ) { j -- ; continue ; }
char t = str1 [ i ] ; str1 [ i ] = str1 [ j ] ; str1 [ j ] = t ; i ++ ; j -- ; } String str2 = String . Join ( " " , str1 ) ; return str2 ; }
public static void Main ( String [ ] args ) { String str = " hello ▁ world " ; Console . WriteLine ( reverseVowel ( str ) ) ; } }
using System ; class GFG {
static bool isPalindrome ( String str ) {
int l = 0 ; int h = str . Length - 1 ;
while ( h > l ) if ( str [ l ++ ] != str [ h -- ] ) return false ; return true ; }
static int minRemovals ( String str ) {
if ( str [ 0 ] == ' ' ) return 0 ;
if ( isPalindrome ( str ) ) return 1 ;
return 2 ; }
public static void Main ( ) { Console . WriteLine ( minRemovals ( "010010" ) ) ; Console . WriteLine ( minRemovals ( "0100101" ) ) ; } }
using System ; class GFG {
static int power ( int x , int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
static int findModuloByM ( int X , int N , int M ) {
if ( N < 6 ) {
string temp = " " ; for ( int i = 0 ; i < N ; i ++ ) temp = temp + ( char ) ( X + 48 ) ;
int res = Convert . ToInt32 ( temp ) % M ; return res ; }
if ( N % 2 == 0 ) {
int half = findModuloByM ( X , N / 2 , M ) % M ;
int res = ( half * power ( 10 , N / 2 , M ) + half ) % M ; return res ; } else {
int half = findModuloByM ( X , N / 2 , M ) % M ;
int res = ( half * power ( 10 , N / 2 + 1 , M ) + half * 10 + X ) % M ; return res ; } }
public static void Main ( ) { int X = 6 , N = 14 , M = 9 ;
Console . WriteLine ( findModuloByM ( X , N , M ) ) ; } }
using System ; class GFG {
class circle { public double x ; public double y ; public double r ; public circle ( int x , int y , int r ) { this . x = x ; this . y = y ; this . r = r ; } }
static bool check ( circle [ ] C ) {
double C1C2 = Math . Sqrt ( ( C [ 1 ] . x - C [ 0 ] . x ) * ( C [ 1 ] . x - C [ 0 ] . x ) + ( C [ 1 ] . y - C [ 0 ] . y ) * ( C [ 1 ] . y - C [ 0 ] . y ) ) ;
bool flag = false ;
if ( C1C2 < ( C [ 0 ] . r + C [ 1 ] . r ) ) {
if ( ( C [ 0 ] . x + C [ 1 ] . x ) == 2 * C [ 2 ] . x && ( C [ 0 ] . y + C [ 1 ] . y ) == 2 * C [ 2 ] . y ) {
flag = true ; } }
return flag ; }
static bool IsFairTriplet ( circle [ ] c ) { bool f = false ;
f |= check ( c ) ; for ( int i = 0 ; i < 2 ; i ++ ) { swap ( c [ 0 ] , c [ 2 ] ) ;
f |= check ( c ) ; } return f ; } static void swap ( circle circle1 , circle circle2 ) { circle temp = circle1 ; circle1 = circle2 ; circle2 = temp ; }
public static void Main ( String [ ] args ) { circle [ ] C = new circle [ 3 ] ; C [ 0 ] = new circle ( 0 , 0 , 8 ) ; C [ 1 ] = new circle ( 0 , 10 , 6 ) ; C [ 2 ] = new circle ( 0 , 5 , 5 ) ; if ( IsFairTriplet ( C ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static double eccHyperbola ( double A , double B ) {
double r = ( double ) B * B / A * A ;
r += 1 ;
return Math . Sqrt ( r ) ; }
public static void Main ( String [ ] args ) { double A = 3.0 , B = 2.0 ; Console . Write ( eccHyperbola ( A , B ) ) ; } }
using System ; class GFG {
static float calculateArea ( float A , float B , float C , float D ) {
float S = ( A + B + C + D ) / 2 ;
float area = ( float ) Math . Sqrt ( ( S - A ) * ( S - B ) * ( S - C ) * ( S - D ) ) ;
return area ; }
static public void Main ( ) { float A = 10 ; float B = 15 ; float C = 20 ; float D = 25 ; Console . Write ( calculateArea ( A , B , C , D ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void triangleArea ( int a , int b ) {
double ratio = ( double ) b / a ;
Console . WriteLine ( ratio ) ; }
public static void Main ( ) { int a = 1 , b = 2 ; triangleArea ( a , b ) ; } }
using System ; class GFG { class pair { public float first , second ; public pair ( float first , float second ) { this . first = first ; this . second = second ; } }
static float distance ( int m , int n , int p , int q ) { return ( float ) Math . Sqrt ( Math . Pow ( n - m , 2 ) + Math . Pow ( q - p , 2 ) * 1.0 ) ; }
static void Excenters ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 ) {
float a = distance ( x2 , x3 , y2 , y3 ) ; float b = distance ( x3 , x1 , y3 , y1 ) ; float c = distance ( x1 , x2 , y1 , y2 ) ;
pair [ ] excenter = new pair [ 4 ] ;
excenter [ 1 ] = new pair ( ( - ( a * x1 ) + ( b * x2 ) + ( c * x3 ) ) / ( - a + b + c ) , ( - ( a * y1 ) + ( b * y2 ) + ( c * y3 ) ) / ( - a + b + c ) ) ;
excenter [ 2 ] = new pair ( ( ( a * x1 ) - ( b * x2 ) + ( c * x3 ) ) / ( a - b + c ) , ( ( a * y1 ) - ( b * y2 ) + ( c * y3 ) ) / ( a - b + c ) ) ;
excenter [ 3 ] = new pair ( ( ( a * x1 ) + ( b * x2 ) - ( c * x3 ) ) / ( a + b - c ) , ( ( a * y1 ) + ( b * y2 ) - ( c * y3 ) ) / ( a + b - c ) ) ;
for ( int i = 1 ; i <= 3 ; i ++ ) { Console . WriteLine ( ( int ) excenter [ i ] . first + " ▁ " + ( int ) excenter [ i ] . second ) ; } }
static void Main ( ) { int x1 , x2 , x3 , y1 , y2 , y3 ; x1 = 0 ; x2 = 3 ; x3 = 0 ; y1 = 0 ; y2 = 0 ; y3 = 4 ; Excenters ( x1 , y1 , x2 , y2 , x3 , y3 ) ; } }
using System ; public class GFG {
static void findHeight ( float p1 , float p2 , float b , float c ) { float a = Math . Max ( p1 , p2 ) - Math . Min ( p1 , p2 ) ;
float s = ( a + b + c ) / 2 ;
float area = ( int ) Math . Sqrt ( s * ( s - a ) * ( s - b ) * ( s - c ) ) ;
float height = ( area * 2 ) / a ;
Console . Write ( " Height ▁ is : ▁ " + height ) ; }
public static void Main ( String [ ] args ) {
float p1 = 25 , p2 = 10 ; float a = 14 , b = 13 ; findHeight ( p1 , p2 , a , b ) ; } }
using System ; class GFG {
static int Icositetragonal_num ( int n ) {
return ( 22 * n * n - 20 * n ) / 2 ; }
public static void Main ( string [ ] args ) { int n = 3 ; Console . Write ( Icositetragonal_num ( n ) + " STRNEWLINE " ) ; n = 10 ; Console . Write ( Icositetragonal_num ( n ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static double area_of_circle ( int m , int n ) {
int square_of_radius = ( m * n ) / 4 ; double area = ( 3.141 * square_of_radius ) ; return area ; }
public static void Main ( ) { int n = 10 ; int m = 30 ; Console . WriteLine ( area_of_circle ( m , n ) ) ; } }
using System ; class GFG {
static double area ( int R ) {
double Base = 1.732 * R ; double height = ( 1.5 ) * R ;
double area = 0.5 * Base * height ; return area ; }
public static void Main ( String [ ] args ) { int R = 7 ; Console . WriteLine ( area ( R ) ) ; } }
using System ; class GFG {
static float circlearea ( float R ) {
if ( R < 0 ) return - 1 ;
float a = ( float ) ( ( 3.14 * R * R ) / 4 ) ; return a ; }
public static void Main ( string [ ] args ) { float R = 2 ; Console . WriteLine ( circlearea ( R ) ) ; } }
using System ; class GFG {
static int countPairs ( int [ ] P , int [ ] Q , int N , int M ) {
int [ ] A = new int [ 2 ] ; int [ ] B = new int [ 2 ] ;
for ( int i = 0 ; i < N ; i ++ ) A [ P [ i ] % 2 ] ++ ;
for ( int i = 0 ; i < M ; i ++ ) B [ Q [ i ] % 2 ] ++ ;
return ( A [ 0 ] * B [ 0 ] + A [ 1 ] * B [ 1 ] ) ; }
public static void Main ( ) { int [ ] P = { 1 , 3 , 2 } ; int [ ] Q = { 3 , 0 } ; int N = P . Length ; int M = Q . Length ; Console . Write ( countPairs ( P , Q , N , M ) ) ; } }
using System ; class GFG {
static int countIntersections ( int n ) { return n * ( n - 1 ) / 2 ; }
public static void Main ( String [ ] args ) { int n = 3 ; Console . WriteLine ( countIntersections ( n ) ) ; } }
using System ; class GFG { static double PI = 3.14159 ;
static double areaOfTriangle ( float d ) {
float c = ( float ) ( 1.618 * d ) ; float s = ( d + c + c ) / 2 ;
double area = Math . Sqrt ( s * ( s - c ) * ( s - c ) * ( s - d ) ) ;
return 5 * area ; }
static double areaOfRegPentagon ( float d ) {
double cal = 4 * Math . Tan ( PI / 5 ) ; double area = ( 5 * d * d ) / cal ;
return area ; }
static double areaOfPentagram ( float d ) {
return areaOfRegPentagon ( d ) + areaOfTriangle ( d ) ; }
public static void Main ( ) { float d = 5 ; Console . WriteLine ( areaOfPentagram ( d ) ) ; } }
using System ; class GFG { static void anglequichord ( int z ) { Console . WriteLine ( " The ▁ angle ▁ is ▁ " + z + " ▁ degrees " ) ; }
public static void Main ( ) { int z = 48 ; anglequichord ( z ) ; } }
using System ; public class GFG {
static void convertToASCII ( int N ) { String num = N . ToString ( ) ; foreach ( char ch in num . ToCharArray ( ) ) { Console . Write ( ch + " ▁ ( " + ( int ) ch + " ) STRNEWLINE " ) ; } }
public static void Main ( String [ ] args ) { int N = 36 ; convertToASCII ( N ) ; } }
using System ; class GFG {
static void productExceptSelf ( int [ ] arr , int N ) {
int product = 1 ;
int z = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] != 0 ) product *= arr [ i ] ;
if ( arr [ i ] == 0 ) z += 1 ; }
int a = Math . Abs ( product ) ; for ( int i = 0 ; i < N ; i ++ ) {
if ( z == 1 ) {
if ( arr [ i ] != 0 ) arr [ i ] = 0 ;
else arr [ i ] = product ; continue ; }
else if ( z > 1 ) {
arr [ i ] = 0 ; continue ; }
int b = Math . Abs ( arr [ i ] ) ;
int curr = ( int ) Math . Round ( Math . Exp ( Math . Log ( a ) - Math . Log ( b ) ) ) ;
if ( arr [ i ] < 0 && product < 0 ) arr [ i ] = curr ;
else if ( arr [ i ] > 0 && product > 0 ) arr [ i ] = curr ;
else arr [ i ] = - 1 * curr ; }
for ( int i = 0 ; i < N ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 10 , 3 , 5 , 6 , 2 } ; int N = arr . Length ;
productExceptSelf ( arr , N ) ; } }
using System ; class GFG {
static void singleDigitSubarrayCount ( int [ ] arr , int N ) {
int res = 0 ;
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( arr [ i ] <= 9 ) {
count ++ ;
res += count ; } else {
count = 0 ; } } Console . Write ( res ) ; }
public static void Main ( string [ ] args ) {
int [ ] arr = { 0 , 1 , 14 , 2 , 5 } ;
int N = arr . Length ; singleDigitSubarrayCount ( arr , N ) ; } }
using System ; class GFG {
static int isPossible ( int N ) { return ( ( ( N & ( N - 1 ) ) & N ) ) ; }
static void countElements ( int N ) {
int count = 0 ; for ( int i = 1 ; i <= N ; i ++ ) { if ( isPossible ( i ) != 0 ) count ++ ; } Console . Write ( count ) ; }
static public void Main ( ) { int N = 15 ; countElements ( N ) ; } }
using System ; public class GFG {
static void countElements ( int N ) { int Cur_Ele = 1 ; int Count = 0 ;
while ( Cur_Ele <= N ) {
Count ++ ;
Cur_Ele = Cur_Ele * 2 ; } Console . Write ( N - Count ) ; }
public static void Main ( String [ ] args ) { int N = 15 ; countElements ( N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void maxAdjacent ( int [ ] arr , int N ) { List < int > res = new List < int > ( ) ; int arr_max = Int32 . MinValue ;
for ( int i = 1 ; i < N ; i ++ ) { arr_max = Math . Max ( arr_max , Math . Abs ( arr [ i - 1 ] - arr [ i ] ) ) ; } for ( int i = 1 ; i < N - 1 ; i ++ ) { int curr_max = Math . Abs ( arr [ i - 1 ] - arr [ i + 1 ] ) ;
int ans = Math . Max ( curr_max , arr_max ) ;
res . Add ( ans ) ; }
foreach ( int x in res ) Console . Write ( x + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 3 , 4 , 7 , 8 } ; int N = arr . Length ; maxAdjacent ( arr , N ) ; } }
using System ; class GFG {
static int minimumIncrement ( int [ ] arr , int N ) {
if ( N % 2 != 0 ) { Console . WriteLine ( " - 1" ) ; Environment . Exit ( 0 ) ; }
int cntEven = 0 ;
int cntOdd = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 == 0 ) {
cntEven += 1 ; } }
cntOdd = N - cntEven ;
return Math . Abs ( cntEven - cntOdd ) / 2 ; }
public static void Main ( ) { int [ ] arr = { 1 , 3 , 4 , 9 } ; int N = arr . Length ;
Console . WriteLine ( minimumIncrement ( arr , N ) ) ; } }
using System ; class GFG {
static void cntWaysConsArray ( int [ ] A , int N ) {
int total = 1 ;
int oddArray = 1 ;
for ( int i = 0 ; i < N ; i ++ ) {
total = total * 3 ;
if ( A [ i ] % 2 == 0 ) {
oddArray *= 2 ; } }
Console . WriteLine ( total - oddArray ) ; }
public static void Main ( String [ ] args ) { int [ ] A = { 2 , 4 } ; int N = A . Length ; cntWaysConsArray ( A , N ) ; } }
using System ; class GFG {
static void countNumberHavingKthBitSet ( int N , int K ) {
int numbers_rightmost_setbit_K = 0 ; for ( int i = 1 ; i <= K ; i ++ ) {
int numbers_rightmost_bit_i = ( N + 1 ) / 2 ;
N -= numbers_rightmost_bit_i ;
if ( i == K ) { numbers_rightmost_setbit_K = numbers_rightmost_bit_i ; } } Console . WriteLine ( numbers_rightmost_setbit_K ) ; }
static public void Main ( String [ ] args ) { int N = 15 ; int K = 2 ; countNumberHavingKthBitSet ( N , K ) ; } }
using System ; class GFG {
static int countSetBits ( int N ) { int count = 0 ;
while ( N != 0 ) { N = N & ( N - 1 ) ; count ++ ; }
return count ; }
public static void Main ( ) { int N = 4 ; int bits = countSetBits ( N ) ;
Console . WriteLine ( " Odd ▁ " + " : ▁ " + ( int ) ( Math . Pow ( 2 , bits ) ) ) ;
Console . WriteLine ( " Even ▁ " + " : ▁ " + ( N + 1 - ( int ) ( Math . Pow ( 2 , bits ) ) ) ) ; } }
using System ; public class GFG {
static void minMoves ( int [ ] arr , int N ) {
int odd_element_cnt = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 != 0 ) { odd_element_cnt ++ ; } }
int moves = ( odd_element_cnt ) / 2 ;
if ( odd_element_cnt % 2 != 0 ) moves += 2 ;
Console . Write ( moves ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 3 , 7 , 20 } ; int N = arr . Length ;
minMoves ( arr , N ) ; } }
using System ; class GFG {
static void minimumSubsetDifference ( int N ) {
int blockOfSize8 = N / 8 ;
string str = " ABBABAAB " ;
int subsetDifference = 0 ;
string partition = " " ; while ( blockOfSize8 -- > 0 ) { partition += str ; }
int [ ] A = new int [ N ] ; int [ ] B = new int [ N ] ; int x = 0 , y = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( partition [ i ] == ' A ' ) { A [ x ++ ] = ( ( i + 1 ) * ( i + 1 ) ) ; }
else { B [ y ++ ] = ( ( i + 1 ) * ( i + 1 ) ) ; } }
Console . WriteLine ( subsetDifference ) ;
for ( int i = 0 ; i < x ; i ++ ) Console . Write ( A [ i ] + " ▁ " ) ; Console . WriteLine ( ) ;
for ( int i = 0 ; i < y ; i ++ ) Console . Write ( B [ i ] + " ▁ " ) ; }
public static void Main ( string [ ] args ) { int N = 8 ;
minimumSubsetDifference ( N ) ; } }
using System . Collections . Generic ; using System ; using System . Linq ; class GFG {
static void findTheGreatestX ( int P , int Q ) {
Dictionary < int , int > divisers = new Dictionary < int , int > ( ) ; for ( int i = 2 ; i * i <= Q ; i ++ ) { while ( Q % i == 0 && Q > 1 ) { Q /= i ;
if ( divisers . ContainsKey ( i ) ) divisers [ i ] ++ ; else divisers [ i ] = 1 ; } }
if ( Q > 1 ) { if ( divisers . ContainsKey ( Q ) ) divisers [ Q ] ++ ; else divisers [ Q ] = 1 ; }
int ans = 0 ; var val = divisers . Keys . ToList ( ) ;
foreach ( var key in val ) { int frequency = divisers [ key ] ; int temp = P ;
int cur = 0 ; while ( temp % key == 0 ) { temp /= key ;
cur ++ ; }
if ( cur < frequency ) { ans = P ; break ; } temp = P ;
for ( int j = cur ; j >= frequency ; j -- ) { temp /= key ; }
ans = Math . Max ( temp , ans ) ; }
Console . WriteLine ( ans ) ; }
public static void Main ( String [ ] args ) {
int P = 10 , Q = 4 ;
findTheGreatestX ( P , Q ) ; } }
using System ; class GFG {
static String checkRearrangements ( int [ , ] mat , int N , int M ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 1 ; j < M ; j ++ ) { if ( mat [ i , 0 ] != mat [ i , j ] ) { return " Yes " ; } } } return " No " ; }
static String nonZeroXor ( int [ , ] mat , int N , int M ) { int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { res = res ^ mat [ i , 0 ] ; }
if ( res != 0 ) return " Yes " ;
else return checkRearrangements ( mat , N , M ) ; }
public static void Main ( String [ ] args ) {
int [ , ] mat = { { 1 , 1 , 2 } , { 2 , 2 , 2 } , { 3 , 3 , 3 } } ; int N = mat . GetLength ( 0 ) ; int M = mat . GetLength ( 1 ) ;
Console . Write ( nonZeroXor ( mat , N , M ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int size_int = 32 ;
static int functionMax ( int [ ] arr , int n ) {
List < int > [ ] setBit = new List < int > [ 32 + 1 ] ; for ( int i = 0 ; i < setBit . Length ; i ++ ) setBit [ i ] = new List < int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < size_int ; j ++ ) {
if ( ( arr [ i ] & ( 1 << j ) ) > 0 )
setBit [ j ] . Add ( i ) ; } }
for ( int i = size_int ; i >= 0 ; i -- ) { if ( setBit [ i ] . Count == 1 ) {
swap ( arr , 0 , setBit [ i ] [ 0 ] ) ; break ; } }
int maxAnd = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { maxAnd = maxAnd & ( ~ arr [ i ] ) ; }
return maxAnd ; } static int [ ] swap ( int [ ] arr , int i , int j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; return arr ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 4 , 8 , 16 } ; int n = arr . Length ;
Console . Write ( functionMax ( arr , n ) ) ; } }
using System ; class GFG {
static int nCr ( int n , int r ) {
int res = 1 ;
if ( r > n - r ) r = n - r ;
for ( int i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static int solve ( int n , int m , int k ) {
int sum = 0 ;
for ( int i = 0 ; i <= k ; i ++ ) sum += nCr ( n , i ) * nCr ( m , k - i ) ; return sum ; }
public static void Main ( String [ ] args ) { int n = 3 , m = 2 , k = 2 ; Console . Write ( solve ( n , m , k ) ) ; } }
using System ; class GFG {
static int powerOptimised ( int a , int n ) {
int ans = 1 ; while ( n > 0 ) { int last_bit = ( n & 1 ) ;
if ( last_bit > 0 ) { ans = ans * a ; } a = a * a ;
n = n >> 1 ; } return ans ; }
public static void Main ( String [ ] args ) { int a = 3 , n = 5 ; Console . Write ( powerOptimised ( a , n ) ) ; } }
using System ; class GFG {
static int findMaximumGcd ( int n ) {
int max_gcd = 1 ;
for ( int i = 1 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
if ( i > max_gcd ) max_gcd = i ; if ( ( n / i != i ) && ( n / i != n ) && ( ( n / i ) > max_gcd ) ) max_gcd = n / i ; } }
return max_gcd ; }
public static void Main ( String [ ] args ) {
int N = 10 ;
Console . Write ( findMaximumGcd ( N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int x = 2000021 ;
static int [ ] v = new int [ x ] ;
static void sieve ( ) { v [ 1 ] = 1 ;
for ( int i = 2 ; i < x ; i ++ ) v [ i ] = i ;
for ( int i = 4 ; i < x ; i += 2 ) v [ i ] = 2 ; for ( int i = 3 ; i * i < x ; i ++ ) {
if ( v [ i ] == i ) {
for ( int j = i * i ; j < x ; j += i ) {
if ( v [ j ] == j ) { v [ j ] = i ; } } } } }
static int prime_factors ( int n ) { HashSet < int > s = new HashSet < int > ( ) ; while ( n != 1 ) { s . Add ( v [ n ] ) ; n = n / v [ n ] ; } return s . Count ; }
static void distinctPrimes ( int m , int k ) {
List < int > result = new List < int > ( ) ; for ( int i = 14 ; i < m + k ; i ++ ) {
long count = prime_factors ( i ) ;
if ( count == k ) { result . Add ( i ) ; } } int p = result . Count ; for ( int index = 0 ; index < p - 1 ; index ++ ) { long element = result [ index ] ; int count = 1 , z = index ;
while ( z < p - 1 && count <= k && result [ z ] + 1 == result [ z + 1 ] ) {
count ++ ; z ++ ; }
if ( count >= k ) Console . Write ( element + " ▁ " ) ; } }
public static void Main ( String [ ] args ) {
sieve ( ) ;
int N = 1000 , K = 3 ;
distinctPrimes ( N , K ) ; } }
using System ; class GFG {
static void print_product ( int a , int b , int c , int d ) {
int prod1 = a * c ; int prod2 = b * d ; int prod3 = ( a + b ) * ( c + d ) ;
int real = prod1 - prod2 ;
int imag = prod3 - ( prod1 + prod2 ) ;
Console . Write ( real + " ▁ + ▁ " + imag + " i " ) ; }
public static void Main ( ) { int a , b , c , d ;
a = 2 ; b = 3 ; c = 4 ; d = 5 ;
print_product ( a , b , c , d ) ; } }
using System ; class GFG {
static bool isInsolite ( int n ) { int N = n ;
int sum = 0 ;
int product = 1 ; while ( n != 0 ) {
int r = n % 10 ; sum = sum + r * r ; product = product * r * r ; n = n / 10 ; } return ( N % sum == 0 ) && ( N % product == 0 ) ; }
public static void Main ( ) { int N = 111 ;
if ( isInsolite ( N ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static int sigma ( int n ) { if ( n == 1 ) return 1 ;
int result = 0 ;
for ( int i = 2 ; i <= Math . Sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) {
if ( i == ( n / i ) ) result += i ; else result += ( i + n / i ) ; } }
return ( result + n + 1 ) ; }
static bool isSuperabundant ( int N ) {
for ( double i = 1 ; i < N ; i ++ ) { double x = sigma ( ( int ) ( i ) ) / i ; double y = sigma ( ( int ) ( N ) ) / ( N * 1.0 ) ; if ( x > y ) return false ; } return true ; }
public static void Main ( String [ ] args ) { int N = 4 ; if ( isSuperabundant ( N ) ) Console . Write ( " Yes STRNEWLINE " ) ; else Console . Write ( " No STRNEWLINE " ) ; } }
using System ; class GFG {
static bool isDNum ( int n ) {
if ( n < 4 ) return false ; int numerator = 0 , hcf = 0 ;
for ( int k = 2 ; k <= n ; k ++ ) { numerator = ( int ) ( Math . Pow ( k , n - 2 ) - k ) ; hcf = __gcd ( n , k ) ; }
if ( hcf == 1 && ( numerator % n ) != 0 ) return false ; return true ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
public static void Main ( String [ ] args ) { int n = 15 ; bool a = isDNum ( n ) ; if ( a ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static int Sum ( int N ) { int [ ] SumOfPrimeDivisors = new int [ N + 1 ] ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( SumOfPrimeDivisors [ i ] == 1 ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
static bool RuthAaronNumber ( int n ) { if ( Sum ( n ) == Sum ( n + 1 ) ) return true ; else return false ; }
public static void Main ( ) { int N = 714 ; if ( RuthAaronNumber ( N ) ) { Console . Write ( " Yes " ) ; } else { Console . Write ( " No " ) ; } } }
using System ; class GFG {
static int maxAdjacentDifference ( int N , int K ) {
if ( N == 1 ) { return 0 ; }
if ( N == 2 ) { return K ; }
return 2 * K ; }
public static void Main ( String [ ] args ) { int N = 6 ; int K = 11 ; Console . Write ( maxAdjacentDifference ( N , K ) ) ; } }
using System ; class GFG { static readonly int mod = 1000000007 ;
public static int linearSum ( int n ) { return ( n * ( n + 1 ) / 2 ) % mod ; }
public static int rangeSum ( int b , int a ) { return ( linearSum ( b ) - linearSum ( a ) ) % mod ; }
public static int totalSum ( int n ) {
int result = 0 ; int i = 1 ;
while ( true ) {
result += rangeSum ( n / i , n / ( i + 1 ) ) * ( i % mod ) % mod ; result %= mod ; if ( i == n ) break ; i = n / ( n / ( i + 1 ) ) ; } return result ; }
public static void Main ( String [ ] args ) { int N = 4 ; Console . WriteLine ( totalSum ( N ) ) ; N = 12 ; Console . WriteLine ( totalSum ( N ) ) ; } }
using System ; class GFG {
static bool isDouble ( int num ) { String s = num . ToString ( ) ; int l = s . Length ;
if ( s [ 0 ] == s [ 1 ] ) return false ;
if ( l % 2 == 1 ) { s = s + s [ 1 ] ; l ++ ; }
String s1 = s . Substring ( 0 , l / 2 ) ;
String s2 = s . Substring ( l / 2 ) ;
return s1 . Equals ( s2 ) ; }
static bool isNontrivialUndulant ( int N ) { return N > 100 && isDouble ( N ) ; }
public static void Main ( String [ ] args ) { int n = 121 ; if ( isNontrivialUndulant ( n ) ) { Console . WriteLine ( " Yes " ) ; } else { Console . WriteLine ( " No " ) ; } } }
using System ; class GFG {
static int MegagonNum ( int n ) { return ( 999998 * n * n - 999996 * n ) / 2 ; }
public static void Main ( String [ ] args ) { int n = 3 ; Console . Write ( MegagonNum ( n ) ) ; } }
using System ; class GFG { static readonly int mod = 1000000007 ;
static int productPairs ( int [ ] arr , int n ) {
int product = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) {
product *= ( arr [ i ] % mod * arr [ j ] % mod ) % mod ; product = product % mod ; } }
return product % mod ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 } ; int n = arr . Length ; Console . Write ( productPairs ( arr , n ) ) ; } }
using System ; class GFG { const int mod = 1000000007 ;
static int power ( int x , int y ) { int p = 1000000007 ;
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ; y = y >> 1 ; x = ( x * x ) % p ; }
return res ; }
static int productPairs ( int [ ] arr , int n ) {
int product = 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
product = ( product % mod * ( int ) power ( arr [ i ] , ( 2 * n ) ) % mod ) % mod ; } return product % mod ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int n = arr . Length ; Console . Write ( productPairs ( arr , n ) ) ; } }
using System ; class GFG {
static void constructArray ( int N ) { int [ ] arr = new int [ N ] ;
for ( int i = 1 ; i <= N ; i ++ ) { arr [ i - 1 ] = i ; }
for ( int i = 0 ; i < N ; i ++ ) { Console . Write ( arr [ i ] + " , ▁ " ) ; } }
public static void Main ( ) { int N = 6 ; constructArray ( N ) ; } }
using System ; class GFG {
static bool isPrime ( int n ) { if ( n <= 1 ) return false ; for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
static int countSubsequences ( int [ ] arr , int n ) {
int totalSubsequence = ( int ) ( Math . Pow ( 2 , n ) - 1 ) ; int countPrime = 0 , countOnes = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) countOnes ++ ; else if ( isPrime ( arr [ i ] ) ) countPrime ++ ; } int compositeSubsequence ;
int onesSequence = ( int ) ( Math . Pow ( 2 , countOnes ) - 1 ) ;
compositeSubsequence = totalSubsequence - countPrime - onesSequence - onesSequence * countPrime ; return compositeSubsequence ; }
public static void Main ( ) { int [ ] arr = { 2 , 1 , 2 } ; int n = arr . Length ; Console . Write ( countSubsequences ( arr , n ) ) ; } }
using System ; class GFG {
static void checksum ( int n , int k ) {
float first_term = ( float ) ( ( ( 2 * n ) / k + ( 1 - k ) ) / 2.0 ) ;
if ( first_term - ( int ) ( first_term ) == 0 ) {
for ( int i = ( int ) first_term ; i <= first_term + k - 1 ; i ++ ) { Console . Write ( i + " ▁ " ) ; } } else Console . ( " - 1" ) ; }
public static void Main ( String [ ] args ) { int n = 33 , k = 6 ; checksum ( n , k ) ; } }
using System ; class GFG {
static void sumEvenNumbers ( int N , int K ) { int check = N - 2 * ( K - 1 ) ;
if ( check > 0 && check % 2 == 0 ) { for ( int i = 0 ; i < K - 1 ; i ++ ) { Console . Write ( "2 ▁ " ) ; } Console . WriteLine ( check ) ; } else { Console . WriteLine ( " - 1" ) ; } }
static public void Main ( String [ ] args ) { int N = 8 ; int K = 2 ; sumEvenNumbers ( N , K ) ; } }
using System ; class GFG {
public static int [ ] calculateWays ( int N ) { int x = 0 ;
int [ ] v = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) v [ i ] = 0 ;
for ( int i = 0 ; i < N / 2 ; i ++ ) {
if ( N % 2 == 0 && i == N / 2 ) break ;
x = N * ( i + 1 ) - ( i + 1 ) * i ;
v [ i ] = x ; v [ N - i - 1 ] = x ; } return v ; }
public static void printArray ( int [ ] v ) { for ( int i = 0 ; i < v . Length ; i ++ ) { Console . Write ( v [ i ] + " ▁ " ) ; } }
public static void Main ( string [ ] args ) { int [ ] v ; v = calculateWays ( 4 ) ; printArray ( v ) ; } }
using System ; class GFG { static readonly int MAXN = 10000000 ;
static int sumOfDigits ( int n ) {
int sum = 0 ; while ( n > 0 ) {
sum += n % 10 ;
n /= 10 ; } return sum ; }
static int smallestNum ( int X , int Y ) {
int res = - 1 ;
for ( int i = X ; i < MAXN ; i ++ ) {
int sum_of_digit = sumOfDigits ( i ) ;
if ( sum_of_digit % Y == 0 ) { res = i ; break ; } } return res ; }
public static void Main ( String [ ] args ) { int X = 5923 , Y = 13 ; Console . Write ( smallestNum ( X , Y ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int countValues ( int N ) { List < int > div = new List < int > ( ) ;
for ( int i = 2 ; i * i <= N ; i ++ ) {
if ( N % i == 0 ) { div . Add ( i ) ;
if ( N != i * i ) { div . Add ( N / i ) ; } } } int answer = 0 ;
for ( int i = 1 ; i * i <= N - 1 ; i ++ ) {
if ( ( N - 1 ) % i == 0 ) { if ( i * i == N - 1 ) answer ++ ; else answer += 2 ; } }
foreach ( int d in div ) { int K = N ; while ( K % d == 0 ) K /= d ; if ( ( K - 1 ) % d == 0 ) answer ++ ; } return answer ; }
public static void Main ( String [ ] args ) { int N = 6 ; Console . Write ( countValues ( N ) ) ; } }
using System ; class GFG {
static void findMaxPrimeDivisor ( int n ) { int max_possible_prime = 0 ;
while ( n % 2 == 0 ) { max_possible_prime ++ ; n = n / 2 ; }
for ( int i = 3 ; i * i <= n ; i = i + 2 ) { while ( n % i == 0 ) { max_possible_prime ++ ; n = n / i ; } }
if ( n > 2 ) { max_possible_prime ++ ; } Console . Write ( max_possible_prime + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { int n = 4 ;
findMaxPrimeDivisor ( n ) ; } }
using System ; class GFG {
static int CountWays ( int n ) { int ans = ( n - 1 ) / 2 ; return ans ; }
public static void Main ( ) { int N = 8 ; Console . Write ( CountWays ( N ) ) ; } }
using System ; class GFG {
static void Solve ( int [ ] arr , int size , int n ) { int [ ] v = new int [ n + 1 ] ;
for ( int i = 0 ; i < size ; i ++ ) v [ arr [ i ] ] ++ ;
int max1 = - 1 , mx = - 1 ; for ( int i = 0 ; i < v . Length ; i ++ ) { if ( v [ i ] > mx ) { mx = v [ i ] ; max1 = i ; } }
int cnt = 0 ; foreach ( int i in v ) { if ( i == 0 ) ++ cnt ; } int diff1 = n + 1 - cnt ;
int max_size = Math . Max ( Math . Min ( v [ max1 ] - 1 , diff1 ) , Math . Min ( v [ max1 ] , diff1 - 1 ) ) ; Console . Write ( " Maximum ▁ size ▁ is ▁ : " + max_size + " STRNEWLINE " ) ;
Console . Write ( " The ▁ First ▁ Array ▁ Is ▁ : STRNEWLINE " ) ; for ( int i = 0 ; i < max_size ; i ++ ) { Console . Write ( max1 + " ▁ " ) ; v [ max1 ] -= 1 ; } Console . Write ( " STRNEWLINE " ) ;
Console . Write ( " The ▁ Second ▁ Array ▁ Is ▁ : STRNEWLINE " ) ; for ( int i = 0 ; i < ( n + 1 ) ; i ++ ) { if ( v [ i ] > 0 ) { Console . Write ( i + " ▁ " ) ; max_size -- ; } if ( max_size < 1 ) break ; } Console . Write ( " STRNEWLINE " ) ; }
public static void Main ( string [ ] args ) {
int n = 7 ;
int [ ] arr = new int [ ] { 1 , 2 , 1 , 5 , 1 , 6 , 7 , 2 } ;
int size = arr . Length ; Solve ( arr , size , n ) ; } }
using System ; class GFG {
static int power ( int x , int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
static int modInverse ( int n , int p ) { return power ( n , p - 2 , p ) ; }
static int nCrModPFermat ( int n , int r , int p ) {
if ( r == 0 ) return 1 ; if ( n < r ) return 0 ;
int [ ] fac = new int [ n + 1 ] ; fac [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fac [ i ] = fac [ i - 1 ] * i % p ; return ( fac [ n ] * modInverse ( fac [ r ] , p ) % p * modInverse ( fac [ n - r ] , p ) % p ) % p ; }
static int SumOfXor ( int [ ] a , int n ) { int mod = 10037 ; int answer = 0 ;
for ( int k = 0 ; k < 32 ; k ++ ) {
int x = 0 , y = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( ( a [ i ] & ( 1 << k ) ) != 0 ) x ++ ; else y ++ ; }
answer += ( ( 1 << k ) % mod * ( nCrModPFermat ( x , 3 , mod ) + x * nCrModPFermat ( y , 2 , mod ) ) % mod ) % mod ; } return answer ; }
public static void Main ( String [ ] args ) { int n = 5 ; int [ ] A = { 3 , 5 , 2 , 18 , 7 } ; Console . WriteLine ( SumOfXor ( A , n ) ) ; } }
using System ; class GFG { public static float round ( float var , int digit ) { float value = ( int ) ( var * Math . ( 10 , digit ) + .5 ) ; return ( float ) value / ( float ) Math . Pow ( 10 , digit ) ; }
public static int probability ( int N ) {
int a = 2 ; int b = 3 ;
if ( N == 1 ) { return a ; } else if ( N == 2 ) { return b ; } else {
for ( int i = 3 ; i <= N ; i ++ ) { int c = a + b ; a = b ; b = c ; } return b ; } }
public static float operations ( int N ) {
int x = probability ( N ) ;
int y = ( int ) Math . Pow ( 2 , N ) ; return round ( ( float ) x / ( float ) y , 2 ) ; }
public static void Main ( string [ ] args ) { int N = 10 ; Console . WriteLine ( ( operations ( N ) ) ) ; } }
using System ; class GFG {
static bool isPerfectCube ( int x ) { double cr = Math . Round ( Math . Cbrt ( x ) ) ; return ( cr * cr * cr == x ) ; }
static void checkCube ( int a , int b ) {
string s1 = Convert . ToString ( a ) ; string s2 = Convert . ToString ( b ) ;
int c = Convert . ToInt32 ( s1 + s2 ) ;
if ( isPerfectCube ( c ) ) { Console . WriteLine ( " Yes " ) ; } else { Console . WriteLine ( " No " ) ; } }
public static void Main ( ) { int a = 6 ; int b = 4 ; checkCube ( a , b ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int largest_sum ( int [ ] arr , int n ) {
int maximum = - 1 ;
Dictionary < int , int > m = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( m . ContainsKey ( arr [ i ] ) ) { m [ arr [ i ] ] ++ ; } else { m . Add ( arr [ i ] , 1 ) ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( m [ arr [ i ] ] > 1 ) { if ( m . ContainsKey ( 2 * arr [ i ] ) ) {
m [ 2 * arr [ i ] ] = m [ 2 * arr [ i ] ] + m [ arr [ i ] ] / 2 ; } else { m . Add ( 2 * arr [ i ] , m [ arr [ i ] ] / 2 ) ; }
if ( 2 * arr [ i ] > maximum ) maximum = 2 * arr [ i ] ; } }
return maximum ; }
public static void Main ( ) { int [ ] arr = { 1 , 1 , 2 , 4 , 7 , 8 } ; int n = arr . Length ;
Console . Write ( largest_sum ( arr , n ) ) ; } }
using System ; class GFG {
static void canBeReduced ( int x , int y ) { int maxi = Math . Max ( x , y ) ; int mini = Math . Min ( x , y ) ;
if ( ( ( x + y ) % 3 ) == 0 && maxi <= 2 * mini ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; }
static void Main ( ) { int x = 6 , y = 9 ;
canBeReduced ( x , y ) ; } }
using System ; class GFG {
static void isPrime ( int N ) { bool isPrime = true ;
int [ ] arr = { 7 , 11 , 13 , 17 , 19 , 23 , 29 , 31 } ;
if ( N < 2 ) { isPrime = false ; }
if ( N % 2 == 0 N % 3 == 0 N % 5 == 0 ) { isPrime = false ; }
for ( int i = 0 ; i < ( int ) Math . Sqrt ( N ) ; i += 30 ) {
foreach ( int c in arr ) {
if ( c > ( int ) Math . Sqrt ( N ) ) { break ; }
else { if ( N % ( c + i ) == 0 ) { isPrime = false ; break ; } }
if ( ! isPrime ) break ; } } if ( isPrime ) Console . WriteLine ( " Prime ▁ Number " ) ; else Console . WriteLine ( " Not ▁ a ▁ Prime ▁ Number " ) ; }
public static void Main ( String [ ] args ) { int N = 121 ;
isPrime ( N ) ; } }
using System ; class GFG {
static void printPairs ( int [ ] arr , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { Console . Write ( " ( " + arr [ i ] + " , ▁ " + arr [ j ] + " ) " + " , ▁ " ) ; } } }
public static void Main ( string [ ] args ) { int [ ] arr = { 1 , 2 } ; int n = arr . Length ; printPairs ( arr , n ) ; } }
using System ; class GFG { static void circle ( int x1 , int y1 , int x2 , int y2 , int r1 , int r2 ) { int distSq = ( int ) Math . Sqrt ( ( ( x1 - x2 ) * ( x1 - x2 ) ) + ( ( y1 - y2 ) * ( y1 - y2 ) ) ) ; if ( distSq + r2 == r1 ) { Console . WriteLine ( " The ▁ smaller ▁ circle ▁ lies ▁ completely " + " ▁ inside ▁ the ▁ bigger ▁ circle ▁ with ▁ " + " touching ▁ each ▁ other ▁ " + " at ▁ a ▁ point ▁ of ▁ circumference . ▁ " ) ; } else if ( distSq + r2 < r1 ) { Console . WriteLine ( " The ▁ smaller ▁ circle ▁ lies ▁ completely " + " ▁ inside ▁ the ▁ bigger ▁ circle ▁ without " + " ▁ touching ▁ each ▁ other ▁ " + " at ▁ a ▁ point ▁ of ▁ circumference . " ) ; } else { Console . WriteLine ( " The ▁ smaller ▁ does ▁ not ▁ lies ▁ inside " + " ▁ the ▁ bigger ▁ circle ▁ completely . " ) ; } }
static public void Main ( ) { int x1 = 10 , y1 = 8 ; int x2 = 1 , y2 = 2 ; int r1 = 30 , r2 = 10 ; circle ( x1 , y1 , x2 , y2 , r1 , r2 ) ; } }
using System ; class GFG {
static void lengtang ( double r1 , double r2 , double d ) { Console . WriteLine ( " The ▁ length ▁ of ▁ the ▁ direct " + " ▁ common ▁ tangent ▁ is ▁ " + ( Math . Sqrt ( Math . Pow ( d , 2 ) - Math . Pow ( ( r1 - r2 ) , 2 ) ) ) ) ; }
public static void Main ( String [ ] args ) { double r1 = 4 , r2 = 6 , d = 3 ; lengtang ( r1 , r2 , d ) ; } }
using System ; class GFG {
static void rad ( double d , double h ) { Console . WriteLine ( " The ▁ radius ▁ of ▁ the ▁ circle ▁ is ▁ " + ( ( d * d ) / ( 8 * h ) + h / 2 ) ) ; }
public static void Main ( ) { double d = 4 , h = 1 ; rad ( d , h ) ; } }
using System ; class GFG {
static void shortdis ( double r , double d ) { Console . WriteLine ( " The ▁ shortest ▁ distance ▁ " + " from ▁ the ▁ chord ▁ to ▁ centre ▁ " + ( Math . Sqrt ( ( r * r ) - ( ( d * d ) / 4 ) ) ) ) ; }
public static void Main ( ) { double r = 4 , d = 3 ; shortdis ( r , d ) ; } }
using System ; class GFG {
static void lengtang ( double r1 , double r2 , double d ) { Console . WriteLine ( " The ▁ length ▁ of ▁ the ▁ direct " + " ▁ common ▁ tangent ▁ is ▁ " + ( Math . Sqrt ( Math . Pow ( d , 2 ) - Math . Pow ( ( r1 - r2 ) , 2 ) ) ) ) ; }
public static void Main ( ) { double r1 = 4 , r2 = 6 , d = 12 ; lengtang ( r1 , r2 , d ) ; } }
using System ; class GFG {
static double square ( double a ) {
if ( a < 0 ) return - 1 ;
double x = 0.464 * a ; return x ; }
public static void Main ( ) { double a = 5 ; Console . WriteLine ( square ( a ) ) ; } }
using System ; class GFG {
static double polyapothem ( double n , double a ) {
if ( a < 0 && n < 0 ) return - 1 ;
return ( a / ( 2 * Math . Tan ( ( 180 / n ) * 3.14159 / 180 ) ) ) ; }
public static void Main ( ) { double a = 9 , n = 6 ; Console . WriteLine ( Math . Round ( polyapothem ( n , a ) , 4 ) ) ; } }
using System ; class GFG {
static float polyarea ( float n , float a ) {
if ( a < 0 && n < 0 ) return - 1 ;
float A = ( a * a * n ) / ( float ) ( 4 * Math . Tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; return A ; }
public static void Main ( ) { float a = 9 , n = 6 ; Console . WriteLine ( polyarea ( n , a ) ) ; } }
using System ; class GFG {
static double calculateSide ( double n , double r ) { double theta , theta_in_radians ; theta = 360 / n ; theta_in_radians = theta * 3.14 / 180 ; return Math . Round ( 2 * r * Math . Sin ( theta_in_radians / 2 ) , 4 ) ; }
public static void Main ( ) {
double n = 3 ;
double r = 5 ; Console . WriteLine ( calculateSide ( n , r ) ) ; } }
using System ; class GFG {
static float cyl ( float r , float R , float h ) {
if ( h < 0 && r < 0 && R < 0 ) return - 1 ;
float r1 = r ;
float h1 = h ;
float V = ( float ) ( 3.14 * Math . Pow ( r1 , 2 ) * h1 ) ; return V ; }
public static void Main ( ) { float r = 7 , R = 11 , h = 6 ; Console . WriteLine ( cyl ( r , R , h ) ) ; } }
using System ; class GFG {
static double Perimeter ( double s , int n ) { double perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
static public void Main ( ) {
int n = 5 ;
double s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; Console . WriteLine ( " Perimeter ▁ of ▁ Regular ▁ Polygon " + " ▁ with ▁ " + n + " ▁ sides ▁ of ▁ length ▁ " + s + " ▁ = ▁ " + peri ) ; } }
using System ; class GFG {
static float rhombusarea ( float l , float b ) {
if ( l < 0 b < 0 ) return - 1 ;
return ( l * b ) / 2 ; }
public static void Main ( ) { float l = 16 , b = 6 ; Console . WriteLine ( rhombusarea ( l , b ) ) ; } }
using System ; class GFG {
static bool FindPoint ( int x1 , int y1 , int x2 , int y2 , int x , int y ) { if ( x > x1 && x < x2 && y > y1 && y < y2 ) return true ; return false ; }
public static void Main ( ) {
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x = 1 , y = 5 ;
if ( FindPoint ( x1 , y1 , x2 , y2 , x , y ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static void shortest_distance ( float x1 , float y1 , float z1 , float a , float b , float c , float d ) { d = Math . Abs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; float e = ( float ) Math . Sqrt ( a * a + b * b + c * c ) ; Console . Write ( " Perpendicular ▁ distance ▁ " + " is ▁ " + d / e ) ; }
public static void Main ( ) { float x1 = 4 ; float y1 = - 4 ; float z1 = 3 ; float a = 2 ; float b = - 2 ; float c = 5 ; float d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ; } }
using System ; class GFG {
static float findVolume ( float l , float b , float h ) {
float volume = ( l * b * h ) / 2 ; return volume ; }
static public void Main ( ) { float l = 18 , b = 12 , h = 9 ;
Console . WriteLine ( " Volume ▁ of ▁ triangular ▁ prism : ▁ " + findVolume ( l , b , h ) ) ; } }
using System ; class GFG {
static bool isRectangle ( int a , int b , int c , int d ) {
if ( a == b && a == c && a == d && c == d && b == c && b == d ) return true ; else if ( a == b && c == d ) return true ; else if ( a == d && c == b ) return true ; else if ( a == c && d == b ) return true ; else return false ; }
public static void Main ( ) { int a = 1 , b = 2 , c = 3 , d = 4 ; if ( isRectangle ( a , b , c , d ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static void midpoint ( int x1 , int x2 , int y1 , int y2 ) { Console . WriteLine ( ( x1 + x2 ) / 2 + " ▁ , ▁ " + ( y1 + y2 ) / 2 ) ; }
public static void Main ( ) { int x1 = - 1 , y1 = 2 ; int x2 = 3 , y2 = - 6 ; midpoint ( x1 , x2 , y1 , y2 ) ; } }
using System ; public class GFG {
static double arcLength ( double diameter , double angle ) { double pi = 22.0 / 7.0 ; double arc ; if ( angle >= 360 ) { Console . WriteLine ( " Angle ▁ cannot " + " ▁ be ▁ formed " ) ; return 0 ; } else { arc = ( pi * diameter ) * ( angle / 360.0 ) ; return arc ; } }
public static void Main ( ) { double diameter = 25.0 ; double angle = 45.0 ; double arc_len = arcLength ( diameter , angle ) ; Console . WriteLine ( arc_len ) ; } }
using System ; class GFG { static void checkCollision ( int a , int b , int c , int x , int y , int radius ) {
double dist = ( Math . Abs ( a * x + b * y + c ) ) / Math . Sqrt ( a * a + b * b ) ;
if ( radius == dist ) Console . WriteLine ( " Touch " ) ; else if ( radius > dist ) Console . WriteLine ( " Intersect " ) ; else Console . WriteLine ( " Outside " ) ; }
public static void Main ( ) { int radius = 5 ; int x = 0 , y = 0 ; int a = 3 , b = 4 , c = 25 ; checkCollision ( a , b , c , x , y , radius ) ; } }
using System ; class GFG {
static double polygonArea ( double [ ] X , double [ ] Y , int n ) {
double area = 0.0 ;
int j = n - 1 ; for ( int i = 0 ; i < n ; i ++ ) { area += ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) ;
j = i ; }
return Math . Abs ( area / 2.0 ) ; }
public static void Main ( ) { double [ ] X = { 0 , 2 , 4 } ; double [ ] Y = { 1 , 3 , 7 } ; int n = X . Length ; Console . WriteLine ( polygonArea ( X , Y , n ) ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static int chk ( int n ) {
List < int > v = new List < int > ( ) ; while ( n != 0 ) { v . Add ( n % 2 ) ; n = n / 2 ; } int j = 0 ; foreach ( int i in v ) { if ( i == 1 ) { return ( int ) Math . Pow ( 2.0 , ( double ) j ) ; } j ++ ; } return 0 ; }
static void sumOfLSB ( int [ ] arr , int N ) {
int [ ] lsb_arr = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) {
lsb_arr [ i ] = chk ( arr [ i ] ) ; }
Array . Sort ( lsb_arr ) ; int ans = 0 ; for ( int i = 0 ; i < N - 1 ; i += 2 ) {
ans += ( lsb_arr [ i + 1 ] ) ; }
Console . WriteLine ( ans ) ; }
static public void Main ( ) { int N = 5 ; int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ;
sumOfLSB ( arr , N ) ; } }
using System ; public class GFG {
static int countSubsequences ( int [ ] arr , int N ) {
int odd = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( ( arr [ i ] & 1 ) % 2 == 1 ) odd ++ ; }
return ( 1 << odd ) - 1 ; }
public static void Main ( string [ ] args ) { int N = 3 ; int [ ] arr = { 1 , 3 , 3 } ;
Console . WriteLine ( countSubsequences ( arr , N ) ) ; } }
using System ; class GFG {
static int getPairsCount ( int [ ] arr , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = arr [ i ] - ( i % arr [ i ] ) ; j < n ; j += arr [ i ] ) {
if ( i < j && Math . Abs ( arr [ i ] - arr [ j ] ) >= Math . Min ( arr [ i ] , arr [ j ] ) ) { count ++ ; } } }
return count ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 2 , 3 } ; int N = arr . Length ; Console . Write ( getPairsCount ( arr , N ) ) ; } }
using System ; class GFG {
static void check ( int N ) { int twos = 0 , fives = 0 ;
while ( N % 2 == 0 ) { N /= 2 ; twos ++ ; }
while ( N % 5 == 0 ) { N /= 5 ; fives ++ ; } if ( N == 1 && twos <= fives ) { Console . Write ( 2 * fives - twos ) ; } else { Console . Write ( - 1 ) ; } }
public static void Main ( ) { int N = 50 ; check ( N ) ; } }
using System ; class GFG {
static void rangeSum ( int [ ] arr , int N , int L , int R ) {
int sum = 0 ;
for ( int i = L - 1 ; i < R ; i ++ ) { sum += arr [ i % N ] ; }
Console . Write ( sum ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 5 , 2 , 6 , 9 } ; int L = 10 , R = 13 ; int N = arr . Length ; rangeSum ( arr , N , L , R ) ; } }
using System ; class GFG {
static void rangeSum ( int [ ] arr , int N , int L , int R ) {
int [ ] prefix = new int [ N + 1 ] ; prefix [ 0 ] = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] + arr [ i - 1 ] ; }
int leftsum = ( ( L - 1 ) / N ) * prefix [ N ] + prefix [ ( L - 1 ) % N ] ;
int rightsum = ( R / N ) * prefix [ N ] + prefix [ R % N ] ;
Console . Write ( rightsum - leftsum ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 2 , 6 , 9 } ; int L = 10 , R = 13 ; int N = arr . Length ; rangeSum ( arr , N , L , R ) ; } }
using System ; class GFG {
static int ExpoFactorial ( int N ) {
int res = 1 ; int mod = 1000000007 ;
for ( int i = 2 ; i < N + 1 ; i ++ )
res = ( int ) Math . Pow ( i , res ) % mod ;
return res ; }
public static void Main ( ) {
int N = 4 ;
Console . Write ( ExpoFactorial ( N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int maxSubArraySumRepeated ( int [ ] arr , int N , int K ) {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) sum += arr [ i ] ; int curr = arr [ 0 ] ;
int ans = arr [ 0 ] ;
if ( K == 1 ) {
for ( int i = 1 ; i < N ; i ++ ) { curr = Math . Max ( arr [ i ] , curr + arr [ i ] ) ; ans = Math . Max ( ans , curr ) ; }
return ans ; }
List < int > V = new List < int > ( ) ;
for ( int i = 0 ; i < 2 * N ; i ++ ) { V . Add ( arr [ i % N ] ) ; }
int maxSuf = V [ 0 ] ;
int maxPref = V [ 2 * N - 1 ] ; curr = V [ 0 ] ; for ( int i = 1 ; i < 2 * N ; i ++ ) { curr += V [ i ] ; maxPref = Math . Max ( maxPref , curr ) ; } curr = V [ 2 * N - 1 ] ; for ( int i = 2 * N - 2 ; i >= 0 ; i -- ) { curr += V [ i ] ; maxSuf = Math . Max ( maxSuf , curr ) ; } curr = V [ 0 ] ;
for ( int i = 1 ; i < 2 * N ; i ++ ) { curr = Math . Max ( V [ i ] , curr + V [ i ] ) ; ans = Math . Max ( ans , curr ) ; }
if ( sum > 0 ) { int temp = sum * ( K - 2 ) ; ans = Math . Max ( ans , Math . Max ( temp + maxPref , temp + maxSuf ) ) ; }
return ans ; }
public static void Main ( ) {
int [ ] arr = { 10 , 20 , - 30 , - 1 , 40 } ; int N = arr . Length ; int K = 10 ;
Console . WriteLine ( maxSubArraySumRepeated ( arr , N , K ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void countSubarray ( int [ ] arr , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i ; j < n ; j ++ ) {
int mxSubarray = 0 ;
int mxOther = 0 ;
for ( int k = i ; k <= j ; k ++ ) { mxSubarray = Math . Max ( mxSubarray , arr [ k ] ) ; }
for ( int k = 0 ; k < i ; k ++ ) { mxOther = Math . Max ( mxOther , arr [ k ] ) ; } for ( int k = j + 1 ; k < n ; k ++ ) { mxOther = Math . Max ( mxOther , arr [ k ] ) ; }
if ( mxSubarray > ( 2 * mxOther ) ) count ++ ; } }
Console . Write ( count ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 6 , 10 , 9 , 7 , 3 } ; int N = arr . Length ; countSubarray ( arr , N ) ; } }
using System ; class GFG {
static void countSubarray ( int [ ] arr , int n ) { int L = 0 , R = 0 ;
int mx = Int32 . MinValue ; for ( int i = 0 ; i < n ; i ++ ) mx = Math . Max ( mx , arr [ i ] ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] * 2 > mx ) {
L = i ; break ; } } for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( arr [ i ] * 2 > mx ) {
R = i ; break ; } }
Console . WriteLine ( ( L + 1 ) * ( n - R ) ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 6 , 10 , 9 , 7 , 3 } ; int N = arr . Length ; countSubarray ( arr , N ) ; } }
using System ; class GFG {
static bool isPrime ( int X ) { for ( int i = 2 ; i * i <= X ; i ++ )
if ( X % i == 0 ) return false ; return true ; }
static void printPrimes ( int [ ] A , int N ) {
for ( int i = 0 ; i < N ; i ++ ) {
for ( int j = A [ i ] - 1 ; ; j -- ) {
if ( isPrime ( j ) ) { Console . Write ( j + " ▁ " ) ; break ; } }
for ( int j = A [ i ] + 1 ; ; j ++ ) {
if ( isPrime ( j ) ) { Console . Write ( j + " ▁ " ) ; break ; } } Console . WriteLine ( ) ; } }
public static void Main ( ) {
int [ ] A = { 17 , 28 } ; int N = A . Length ;
printPrimes ( A , N ) ; } }
using System ; class GFG {
static int KthSmallest ( int [ ] A , int [ ] B , int N , int K ) { int M = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { M = Math . Max ( A [ i ] , M ) ; }
int [ ] freq = new int [ M + 1 ] ;
for ( int i = 0 ; i < N ; i ++ ) { freq [ A [ i ] ] += B [ i ] ; }
int sum = 0 ;
for ( int i = 0 ; i <= M ; i ++ ) {
sum += freq [ i ] ;
if ( sum >= K ) {
return i ; } }
return - 1 ; }
public static void Main ( String [ ] args ) {
int [ ] A = { 3 , 4 , 5 } ; int [ ] B = { 2 , 1 , 3 } ; int N = A . Length ; int K = 4 ;
Console . Write ( KthSmallest ( A , B , N , K ) ) ; } }
using System ; class GFG {
static void findbitwiseOR ( int [ ] a , int n ) {
int res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int curr_sub_array = a [ i ] ;
res = res | curr_sub_array ; for ( int j = i ; j < n ; j ++ ) {
curr_sub_array = curr_sub_array & a [ j ] ; res = res | curr_sub_array ; } }
Console . Write ( res ) ; }
static void Main ( ) { int [ ] A = { 1 , 2 , 3 } ; int N = A . Length ; findbitwiseOR ( A , N ) ; } }
using System ; class GFG {
static void findbitwiseOR ( int [ ] a , int n ) {
int res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) res = res | a [ i ] ;
Console . Write ( res ) ; }
public static void Main ( ) { int [ ] A = { 1 , 2 , 3 } ; int N = A . Length ; findbitwiseOR ( A , N ) ; } }
using System ; class GFG {
static void check ( int n ) {
int sumOfDigit = 0 ; int prodOfDigit = 1 ; while ( n > 0 ) {
int rem ; rem = n % 10 ;
sumOfDigit += rem ;
prodOfDigit *= rem ;
n /= 10 ; }
if ( sumOfDigit > prodOfDigit ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; }
public static void Main ( ) { int N = 1234 ; check ( N ) ; } }
using System ; class GFG {
static void evenOddBitwiseXOR ( int N ) { Console . Write ( " Even : ▁ " + 0 + " ▁ " ) ;
for ( int i = 4 ; i <= N ; i = i + 4 ) { Console . Write ( i + " ▁ " ) ; } Console . Write ( " STRNEWLINE " ) ; Console . Write ( " Odd : ▁ " + 1 + " ▁ " ) ;
for ( int i = 4 ; i <= N ; i = i + 4 ) { Console . Write ( i - 1 + " ▁ " ) ; } if ( N % 4 == 2 ) Console . Write ( N + 1 ) ; else if ( N % 4 == 3 ) Console . Write ( N ) ; }
public static void Main ( ) { int N = 6 ; evenOddBitwiseXOR ( N ) ; } }
using System ; class GFG {
static void findPermutation ( int [ ] arr ) { int N = arr . Length ; int i = N - 2 ;
while ( i >= 0 && arr [ i ] <= arr [ i + 1 ] ) i -- ;
if ( i == - 1 ) { Console . Write ( " - 1" ) ; return ; } int j = N - 1 ;
while ( j > i && arr [ j ] >= arr [ i ] ) j -- ;
while ( j > i && arr [ j ] == arr [ j - 1 ] ) {
j -- ; }
int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ;
foreach ( int it in arr ) { Console . Write ( it + " ▁ " ) ; } }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 5 , 3 , 4 , 6 } ; findPermutation ( arr ) ; } }
using System ; class GFG {
static void sieveOfEratosthenes ( int N , int [ ] s ) {
bool [ ] prime = new bool [ N + 1 ] ;
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
Console . Write ( Math . Abs ( even - odd ) ) ; }
public static void Main ( ) { int N = 12 ; findDifference ( N ) ; } }
using System ; class GFG {
static void findMedian ( int Mean , int Mode ) {
double Median = ( 2 * Mean + Mode ) / 3.0 ;
Console . Write ( Median ) ; }
public static void Main ( ) { int mode = 6 , mean = 3 ; findMedian ( mean , mode ) ; } }
using System ; class GFG {
private static double vectorMagnitude ( int x , int y , int z ) {
int sum = x * x + y * y + z * z ;
return Math . Sqrt ( sum ) ; }
static void Main ( ) { int x = 1 ; int y = 2 ; int z = 3 ; Console . Write ( vectorMagnitude ( x , y , z ) ) ; } }
using System ; class GFG {
static int multiplyByMersenne ( int N , int M ) {
int x = ( int ) ( Math . Log ( M + 1 ) / Math . Log ( 2 ) ) ;
return ( ( N << x ) - N ) ; }
static public void Main ( ) { int N = 4 ; int M = 15 ; Console . Write ( multiplyByMersenne ( N , M ) ) ; } }
using System ; using System . Collections . Generic ; using System . Linq ; class GFG {
static int perfectSquare ( int num ) {
int sr = ( int ) ( Math . Sqrt ( num ) ) ;
int a = sr * sr ; int b = ( sr + 1 ) * ( sr + 1 ) ;
if ( ( num - a ) < ( b - num ) ) { return a ; } else { return b ; } }
static int powerOfTwo ( int num ) {
int lg = ( int ) ( Math . Log ( num ) / Math . Log ( 2 ) ) ;
int p = ( int ) ( Math . Pow ( 2 , lg ) ) ; return p ; }
static void uniqueElement ( int [ ] arr , int N ) { bool ans = true ;
Dictionary < int , int > freq = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { if ( freq . ContainsKey ( arr [ i ] ) ) { freq [ arr [ i ] ] = freq [ arr [ i ] ] + 1 ; } else { freq [ arr [ i ] ] = 1 ; } }
foreach ( var el in freq . OrderBy ( el => el . Key ) ) {
if ( el . Value == 1 ) { ans = false ;
int ps = perfectSquare ( el . Key ) ;
Console . Write ( powerOfTwo ( ps ) + " ▁ " ) ; } }
if ( ans ) Console . Write ( " - 1" ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 4 , 11 , 4 , 3 , 4 } ; int N = arr . Length ; uniqueElement ( arr , N ) ; } }
using System ; class GFG {
static void partitionArray ( int [ ] a , int n ) {
int [ ] min = new int [ n ] ;
int mini = Int32 . MaxValue ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
mini = Math . Min ( mini , a [ i ] ) ;
min [ i ] = mini ; }
int maxi = Int32 . MinValue ;
int ind = - 1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
maxi = Math . Max ( maxi , a [ i ] ) ;
if ( maxi < min [ i + 1 ] ) {
ind = i ;
break ; } }
if ( ind != - 1 ) {
for ( int i = 0 ; i <= ind ; i ++ ) Console . Write ( a [ i ] + " ▁ " ) ; Console . WriteLine ( ) ;
for ( int i = ind + 1 ; i < n ; i ++ ) Console . Write ( a [ i ] + " ▁ " ) ; }
else Console . ( " Impossible " ) ; }
public static void Main ( string [ ] args ) { int [ ] arr = { 5 , 3 , 2 , 7 , 9 } ; int N = arr . Length ; partitionArray ( arr , N ) ; } }
using System ; public class GFG {
static int countPrimeFactors ( int n ) { int count = 0 ;
while ( n % 2 == 0 ) { n = n / 2 ; count ++ ; }
for ( int i = 3 ; i <= ( int ) Math . Sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { n = n / i ; count ++ ; } }
if ( n > 2 ) count ++ ; return ( count ) ; }
static int findSum ( int n ) {
int sum = 0 ; for ( int i = 1 , num = 2 ; i <= n ; num ++ ) {
if ( countPrimeFactors ( num ) == 2 ) { sum += num ;
i ++ ; } } return sum ; }
static void check ( int n , int k ) {
int s = findSum ( k - 1 ) ;
if ( s >= n ) Console . WriteLine ( " No " ) ;
else Console . ( " Yes " ) ; }
public static void Main ( String [ ] args ) { int n = 100 , k = 6 ; check ( n , k ) ; } }
using System ; public class GFG {
static int gcd ( int a , int b ) {
while ( b > 0 ) { int rem = a % b ; a = b ; b = rem ; }
return a ; }
static int countNumberOfWays ( int n ) {
if ( n == 1 ) return - 1 ;
int g = 0 ; int power = 0 ;
while ( n % 2 == 0 ) { power ++ ; n /= 2 ; } g = gcd ( g , power ) ;
for ( int i = 3 ; i <= ( int ) Math . Sqrt ( n ) ; i += 2 ) { power = 0 ;
while ( n % i == 0 ) { power ++ ; n /= i ; } g = gcd ( g , power ) ; }
if ( n > 2 ) g = gcd ( g , 1 ) ;
int ways = 1 ;
power = 0 ; while ( g % 2 == 0 ) { g /= 2 ; power ++ ; }
ways *= ( power + 1 ) ;
for ( int i = 3 ; i <= ( int ) Math . Sqrt ( g ) ; i += 2 ) { power = 0 ;
while ( g % i == 0 ) { power ++ ; g /= i ; }
ways *= ( power + 1 ) ; }
if ( g > 2 ) ways *= 2 ;
return ways ; }
public static void Main ( String [ ] args ) { int N = 64 ; Console . Write ( countNumberOfWays ( N ) ) ; } }
using System ; class GFG {
static int powOfPositive ( int n ) {
int pos = ( int ) Math . Floor ( ( Math . Log ( n ) / Math . Log ( 2 ) ) ) ; return ( int ) Math . Pow ( 2 , pos ) ; }
static int powOfNegative ( int n ) {
int pos = ( int ) Math . Ceiling ( ( Math . Log ( n ) / Math . Log ( 2 ) ) ) ; return ( int ) ( - 1 * Math . Pow ( 2 , pos ) ) ; }
static void highestPowerOf2 ( int n ) {
if ( n > 0 ) { Console . WriteLine ( powOfPositive ( n ) ) ; } else {
n = - n ; Console . WriteLine ( powOfNegative ( n ) ) ; } }
public static void Main ( ) { int n = - 24 ; highestPowerOf2 ( n ) ; } }
using System ; class GFG {
public static int noOfCards ( int n ) { return n * ( 3 * n + 1 ) / 2 ; }
public static void Main ( String [ ] args ) { int n = 3 ; Console . Write ( noOfCards ( n ) ) ; } }
using System ; class GFG {
static String smallestPoss ( String s , int n ) {
String ans = " " ;
int [ ] arr = new int [ 10 ] ;
for ( int i = 0 ; i < n ; i ++ ) { arr [ s [ i ] - 48 ] ++ ; }
for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < arr [ i ] ; j ++ ) ans = ans + String . Join ( " " , i ) ; }
return ans ; }
public static void Main ( String [ ] args ) { int N = 15 ; String K = "325343273113434" ; Console . Write ( smallestPoss ( K , N ) ) ; } }
using System ; class GFG {
static int Count_subarray ( int [ ] arr , int n ) { int subarray_sum , remaining_sum , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i ; j < n ; j ++ ) {
subarray_sum = 0 ; remaining_sum = 0 ;
for ( int k = i ; k <= j ; k ++ ) { subarray_sum += arr [ k ] ; }
for ( int l = 0 ; l < i ; l ++ ) { remaining_sum += arr [ l ] ; } for ( int l = j + 1 ; l < n ; l ++ ) { remaining_sum += arr [ l ] ; }
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 10 , 9 , 12 , 6 } ; int n = arr . Length ; Console . Write ( Count_subarray ( arr , n ) ) ; } }
using System ; class GFG { static int Count_subarray ( int [ ] arr , int n ) { int total_sum = 0 , subarray_sum , remaining_sum , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { total_sum += arr [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
subarray_sum = 0 ;
for ( int j = i ; j < n ; j ++ ) {
subarray_sum += arr [ j ] ; remaining_sum = total_sum - subarray_sum ;
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
public static void Main ( ) { int [ ] arr = { 10 , 9 , 12 , 6 } ; int n = arr . Length ; Console . WriteLine ( Count_subarray ( arr , n ) ) ; } }
using System ; class GFG {
static int maxXOR ( int [ ] arr , int n ) {
int xorArr = 0 ; for ( int i = 0 ; i < n ; i ++ ) xorArr ^= arr [ i ] ;
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) ans = Math . Max ( ans , ( xorArr ^ arr [ i ] ) ) ;
return ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 1 , 3 } ; int n = arr . Length ; Console . WriteLine ( maxXOR ( arr , n ) ) ; } }
using System ; class GFG {
static bool digitDividesK ( int num , int k ) { while ( num != 0 ) {
int d = num % 10 ;
if ( d != 0 && k % d == 0 ) return true ;
num = num / 10 ; }
return false ; }
static int findCount ( int l , int r , int k ) {
int count = 0 ;
for ( int i = l ; i <= r ; i ++ ) {
if ( digitDividesK ( i , k ) ) count ++ ; } return count ; }
public static void Main ( ) { int l = 20 , r = 35 ; int k = 45 ; Console . WriteLine ( findCount ( l , r , k ) ) ; } }
using System ; class GFG {
static Boolean isFactorial ( int n ) { for ( int i = 1 ; ; i ++ ) { if ( n % i == 0 ) { n /= i ; } else { break ; } } if ( n == 1 ) { return true ; } else { return false ; } }
public static void Main ( String [ ] args ) { int n = 24 ; Boolean ans = isFactorial ( n ) ; if ( ans == true ) { Console . WriteLine ( " Yes " ) ; } else { Console . WriteLine ( " No " ) ; } } }
using System ; class GFG {
static int lcm ( int a , int b ) { int GCD = __gcd ( a , b ) ; return ( a * b ) / GCD ; }
static int MinLCM ( int [ ] a , int n ) {
int [ ] Prefix = new int [ n + 2 ] ; int [ ] Suffix = new int [ n + 2 ] ;
Prefix [ 1 ] = a [ 0 ] ; for ( int i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = lcm ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( int i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = lcm ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
int ans = Math . Min ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( int i = 2 ; i < n ; i += 1 ) { ans = Math . Min ( ans , lcm ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
public static void Main ( String [ ] args ) { int [ ] a = { 5 , 15 , 9 , 36 } ; int n = a . Length ; Console . WriteLine ( MinLCM ( a , n ) ) ; } }
using System ; class GFG {
static int count ( int n ) { return n * ( 3 * n - 1 ) / 2 ; }
public static void Main ( String [ ] args ) { int n = 3 ; Console . WriteLine ( count ( n ) ) ; } }
using System ; class GFG {
static int findMinValue ( int [ ] arr , int n ) {
long sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
return ( ( int ) ( sum / n ) + 1 ) ; }
static public void Main ( ) { int [ ] arr = { 4 , 2 , 1 , 10 , 6 } ; int n = arr . Length ; Console . WriteLine ( findMinValue ( arr , n ) ) ; } }
using System ; class GFG { const int MOD = 1000000007 ;
static int modFact ( int n , int m ) { int result = 1 ; for ( int i = 1 ; i <= m ; i ++ ) result = ( result * i ) % MOD ; return result ; }
public static void Main ( ) { int n = 3 , m = 2 ; Console . WriteLine ( modFact ( n , m ) ) ; } }
using System ; class GFG { static readonly int mod = ( int ) ( 1e9 + 7 ) ;
static long power ( int p ) { long res = 1 ; for ( int i = 1 ; i <= p ; ++ i ) { res *= 2 ; res %= mod ; } return res % mod ; }
static long subset_square_sum ( int [ ] A ) { int n = A . Length ; long ans = 0 ;
foreach ( int i in A ) { ans += ( 1 * i * i ) % mod ; ans %= mod ; } return ( 1 * ans * power ( n - 1 ) ) % mod ; }
public static void Main ( String [ ] args ) { int [ ] A = { 3 , 7 } ; Console . WriteLine ( subset_square_sum ( A ) ) ; } }
using System ; class GFG { static int N = 100050 ; static int [ ] lpf = new int [ N ] ; static int [ ] mobius = new int [ N ] ;
static void least_prime_factor ( ) { for ( int i = 2 ; i < N ; i ++ )
if ( lpf [ i ] == 0 ) for ( int j = i ; j < N ; j += i )
if ( lpf [ j ] == 0 ) lpf [ j ] = i ; }
static void Mobius ( ) { for ( int i = 1 ; i < N ; i ++ ) {
if ( i == 1 ) mobius [ i ] = 1 ; else {
if ( lpf [ i / lpf [ i ] ] == lpf [ i ] ) mobius [ i ] = 0 ;
else mobius [ i ] = - 1 * mobius [ i / lpf [ i ] ] ; } } }
static int gcd_pairs ( int [ ] a , int n ) {
int maxi = 0 ;
int [ ] fre = new int [ N ] ;
for ( int i = 0 ; i < n ; i ++ ) { fre [ a [ i ] ] ++ ; maxi = Math . Max ( a [ i ] , maxi ) ; } least_prime_factor ( ) ; Mobius ( ) ;
int ans = 0 ;
for ( int i = 1 ; i <= maxi ; i ++ ) { if ( mobius [ i ] == 0 ) continue ; int temp = 0 ; for ( int j = i ; j <= maxi ; j += i ) temp += fre [ j ] ; ans += temp * ( temp - 1 ) / 2 * mobius [ i ] ; }
return ans ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = a . Length ;
Console . Write ( gcd_pairs ( a , n ) ) ; } }
using System ; class GFG {
static void compareVal ( double x , double y ) {
double a = y * Math . Log ( x ) ; double b = x * Math . Log ( y ) ;
if ( a > b ) Console . Write ( x + " ^ " + y + " ▁ > ▁ " + y + " ^ " + x ) ; else if ( a < b ) Console . Write ( x + " ^ " + y + " ▁ < ▁ " + y + " ^ " + x ) ; else if ( a == b ) Console . Write ( x + " ^ " + y + " ▁ = ▁ " + y + " ^ " + x ) ; }
static public void Main ( ) { double x = 4 , y = 5 ; compareVal ( x , y ) ; } }
using System ; class GFG {
static void ZigZag ( int n ) {
long [ ] fact = new long [ n + 1 ] ; long [ ] zig = new long [ n + 1 ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) zig [ i ] = 0 ;
fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
zig [ 0 ] = 1 ; zig [ 1 ] = 1 ; Console . Write ( " zig ▁ zag ▁ numbers : ▁ " ) ;
Console . Write ( zig [ 0 ] + " ▁ " + zig [ 1 ] + " ▁ " ) ;
for ( int i = 2 ; i < n ; i ++ ) { long sum = 0 ; for ( int k = 0 ; k <= i - 1 ; k ++ ) {
sum += ( fact [ i - 1 ] / ( fact [ i - 1 - k ] * fact [ k ] ) ) * zig [ k ] * zig [ i - 1 - k ] ; }
zig [ i ] = sum / 2 ;
Console . Write ( sum / 2 + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int n = 10 ;
ZigZag ( n ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int find_count ( List < int > ele ) {
int count = 0 ; for ( int i = 0 ; i < ele . Count ; i ++ ) {
List < int > p = new List < int > ( ) ;
int c = 0 , j ;
for ( j = ele . Count - 1 ; j >= ( ele . Count - 1 - i ) && j >= 0 ; j -- ) { p . Add ( ele [ j ] ) ; } j = ele . Count - 1 ; int k = 0 ;
while ( j >= 0 ) {
if ( ele [ j ] != p [ k ] ) { break ; } j -- ; k ++ ;
if ( k == p . Count ) { c ++ ; k = 0 ; } } count = Math . Max ( count , c ) ; }
return count ; }
static void solve ( int n ) {
int count = 1 ;
List < int > ele = new List < int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( count + " , ▁ " ) ;
ele . Add ( count ) ;
count = find_count ( ele ) ; } }
public static void Main ( String [ ] args ) { int n = 10 ; solve ( n ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static Dictionary < int , int > store = new Dictionary < int , int > ( ) ;
static int Wedderburn ( int n ) {
if ( n <= 2 ) return store [ n ] ;
else if ( n % 2 == 0 ) {
int x = n / 2 , ans = 0 ;
for ( int i = 1 ; i < x ; i ++ ) { ans += store [ i ] * store [ n - i ] ; }
ans += ( store [ x ] * ( store [ x ] + 1 ) ) / 2 ;
if ( store . ContainsKey ( n ) ) { store . Remove ( n ) ; store . Add ( n , ans ) ; } else store . Add ( n , ans ) ;
return ans ; } else {
int x = ( n + 1 ) / 2 , ans = 0 ;
for ( int i = 1 ; i < x ; i ++ ) { ans += store [ i ] * store [ n - i ] ; }
if ( store . ContainsKey ( n ) ) { store . Remove ( n ) ; store . Add ( n , ans ) ; } else store . Add ( n , ans ) ;
return ans ; } }
static void Wedderburn_Etherington ( int n ) {
store . Add ( 0 , 0 ) ; store . Add ( 1 , 1 ) ; store . Add ( 2 , 1 ) ;
for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( Wedderburn ( i ) ) ; if ( i != n - 1 ) Console . Write ( " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int n = 10 ;
Wedderburn_Etherington ( n ) ; } }
using System ; class GFG {
static int Max_sum ( int [ ] a , int n ) {
int pos = 0 , neg = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] > 0 ) pos = 1 ;
else if ( a [ i ] < 0 ) = 1 ;
if ( ( pos == 1 ) && ( neg == 1 ) ) break ; }
int sum = 0 ; if ( ( pos == 1 ) && ( neg == 1 ) ) { for ( int i = 0 ; i < n ; i ++ ) sum += Math . Abs ( a [ i ] ) ; } else if ( pos == 1 ) {
int mini = a [ 0 ] ; sum = a [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { mini = Math . Min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; } else if ( neg = = 1 ) {
for ( int i = 0 ; i < n ; i ++ ) a [ i ] = Math . Abs ( a [ i ] ) ;
int mini = a [ 0 ] ; sum = a [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { mini = Math . Min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; }
return sum ; }
public static void Main ( String [ ] args ) { int [ ] a = { 1 , 3 , 5 , - 2 , - 6 } ; int n = a . Length ;
Console . WriteLine ( Max_sum ( a , n ) ) ; } }
using System ; class GFG {
static void decimalToBinary ( int n ) {
if ( n == 0 ) { Console . Write ( "0" ) ; return ; }
decimalToBinary ( n / 2 ) ; Console . Write ( n % 2 ) ; }
public static void Main ( String [ ] args ) { int n = 13 ; decimalToBinary ( n ) ; } }
using System ; class GFG {
static void MinimumValue ( int x , int y ) {
if ( x > y ) { int temp = x ; x = y ; y = temp ; }
int a = 1 ; int b = x - 1 ; int c = y - b ; Console . WriteLine ( a + " ▁ " + b + " ▁ " + c ) ; }
public static void Main ( ) { int x = 123 , y = 13 ;
MinimumValue ( x , y ) ; } }
using System ; class GFG {
static bool canConvert ( int a , int b ) { while ( b > a ) {
if ( b % 10 == 1 ) { b /= 10 ; continue ; }
if ( b % 2 == 0 ) { b /= 2 ; continue ; }
return false ; }
if ( b == a ) return true ; return false ; }
public static void Main ( ) { int A = 2 , B = 82 ; if ( canConvert ( A , B ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class Rectangle {
static int count ( int N ) { int a = 0 ; a = ( N * ( N + 1 ) ) / 2 ; return a ; }
public static void Main ( ) { int n = 4 ; Console . Write ( count ( n ) ) ; } }
using System ; class GFG {
static int numberOfDays ( int a , int b , int n ) { int Days = b * ( n + a ) / ( a + b ) ; return Days ; }
public static void Main ( ) { int a = 10 , b = 20 , n = 5 ; Console . WriteLine ( numberOfDays ( a , b , n ) ) ; } }
using System ; class GFG {
static int getAverage ( int x , int y ) {
int avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; return avg ; }
public static void Main ( ) { int x = 10 , y = 9 ; Console . WriteLine ( getAverage ( x , y ) ) ; } }
using System ; class GFG {
static int smallestIndex ( int [ ] a , int n ) {
int right1 = 0 , right0 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == 1 ) right1 = i ;
else right0 = i ; }
return Math . Min ( right1 , right0 ) ; }
public static void Main ( ) { int [ ] a = { 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int n = a . Length ; Console . Write ( smallestIndex ( a , n ) ) ; } }
using System ; class GFG {
static int countSquares ( int r , int c , int m ) {
int squares = 0 ;
for ( int i = 1 ; i <= 8 ; i ++ ) { for ( int j = 1 ; j <= 8 ; j ++ ) {
if ( Math . Max ( Math . Abs ( i - r ) , Math . Abs ( j - c ) ) <= m ) squares ++ ; } }
return squares ; }
public static void Main ( ) { int r = 4 , c = 4 , m = 1 ; Console . Write ( countSquares ( r , c , m ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int countQuadruples ( int [ ] a , int n ) {
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( mp . ContainsKey ( a [ i ] ) ) { mp [ a [ i ] ] = mp [ a [ i ] ] + 1 ; } else { mp . Add ( a [ i ] , 1 ) ; } int count = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { for ( int k = 0 ; k < n ; k ++ ) {
if ( j == k ) continue ;
mp [ a [ j ] ] = mp [ a [ j ] ] - 1 ; mp [ a [ k ] ] = mp [ a [ k ] ] - 1 ;
int first = a [ j ] - ( a [ k ] - a [ j ] ) ;
int fourth = ( a [ k ] * a [ k ] ) / a [ j ] ;
if ( ( a [ k ] * a [ k ] ) % a [ j ] == 0 ) {
if ( a [ j ] != a [ k ] ) { if ( mp . ContainsKey ( first ) && mp . ContainsKey ( fourth ) ) count += mp [ first ] * mp [ fourth ] ; }
else if ( mp . ContainsKey ( first ) & & mp . ContainsKey ( fourth ) ) += mp [ first ] * ( mp [ fourth ] - 1 ) ; }
if ( mp . ContainsKey ( a [ j ] ) ) { mp [ a [ j ] ] = mp [ a [ j ] ] + 1 ; } else { mp . Add ( a [ j ] , 1 ) ; } if ( mp . ContainsKey ( a [ k ] ) ) { mp [ a [ k ] ] = mp [ a [ k ] ] + 1 ; } else { mp . Add ( a [ k ] , 1 ) ; } } } return count ; }
public static void Main ( String [ ] args ) { int [ ] a = { 2 , 6 , 4 , 9 , 2 } ; int n = a . Length ; Console . Write ( countQuadruples ( a , n ) ) ; } }
using System ; class GFG {
static int countNumbers ( int L , int R , int K ) { if ( K == 9 ) { K = 0 ; }
int totalnumbers = R - L + 1 ;
int factor9 = totalnumbers / 9 ;
int rem = totalnumbers % 9 ;
int ans = factor9 ;
for ( int i = R ; i > R - rem ; i -- ) { int rem1 = i % 9 ; if ( rem1 == K ) { ans ++ ; } } return ans ; }
public static void Main ( ) { int L = 10 ; int R = 22 ; int K = 3 ; Console . WriteLine ( countNumbers ( L , R , K ) ) ; } }
using System ; class GFG {
static void BalanceArray ( int [ ] A , int [ , ] Q ) { int [ ] ANS = new int [ A . Length ] ; int i , sum = 0 ; for ( i = 0 ; i < A . Length ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; for ( i = 0 ; i < Q . GetLength ( 0 ) ; i ++ ) { int index = Q [ i , 0 ] ; int value = Q [ i , 1 ] ;
if ( A [ index ] % 2 == 0 ) sum = sum - A [ index ] ; A [ index ] = A [ index ] + value ;
if ( A [ index ] % 2 == 0 ) sum = sum + A [ index ] ;
ANS [ i ] = sum ; }
for ( i = 0 ; i < ANS . Length ; i ++ ) Console . Write ( ANS [ i ] + " ▁ " ) ; }
public static void Main ( ) { int [ ] A = { 1 , 2 , 3 , 4 } ; int [ , ] Q = { { 0 , 1 } , { 1 , - 3 } , { 0 , - 4 } , { 3 , 2 } } ; BalanceArray ( A , Q ) ; } }
using System ; class GFG {
static int Cycles ( int N ) { int fact = 1 , result = 0 ; result = N - 1 ;
int i = result ; while ( i > 0 ) { fact = fact * i ; i -- ; } return fact / 2 ; }
public static void Main ( ) { int N = 5 ; int Number = Cycles ( N ) ; Console . Write ( " Hamiltonian ▁ cycles ▁ = ▁ " + Number ) ; } }
using System ; class GFG {
static bool digitWell ( int n , int m , int k ) { int cnt = 0 ; while ( n > 0 ) { if ( n % 10 == m ) ++ cnt ; n /= 10 ; } return cnt == k ; }
static int findInt ( int n , int m , int k ) { int i = n + 1 ; while ( true ) { if ( digitWell ( i , m , k ) ) return i ; i ++ ; } }
public static void Main ( ) { int n = 111 , m = 2 , k = 2 ; Console . WriteLine ( findInt ( n , m , k ) ) ; } }
using System ; class GFG {
static int countOdd ( int [ ] arr , int n ) {
int odd = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] % 2 == 1 ) odd ++ ; } return odd ; }
static int countValidPairs ( int [ ] arr , int n ) { int odd = countOdd ( arr , n ) ; return ( odd * ( odd - 1 ) ) / 2 ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . Length ; Console . WriteLine ( countValidPairs ( arr , n ) ) ; } }
using System ; class GFG {
static int gcd ( int a , int b ) { if ( b == 0 ) return a ; else return gcd ( b , a % b ) ; }
static int lcmOfArray ( int [ ] arr , int n ) { if ( n < 1 ) return 0 ; int lcm = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) lcm = ( lcm * arr [ i ] ) / gcd ( lcm , arr [ i ] ) ;
return lcm ; }
static int minPerfectCube ( int [ ] arr , int n ) { int minPerfectCube ;
int lcm = lcmOfArray ( arr , n ) ; minPerfectCube = lcm ; int cnt = 0 ; while ( lcm > 1 && lcm % 2 == 0 ) { cnt ++ ; lcm /= 2 ; }
if ( cnt % 3 == 2 ) minPerfectCube *= 2 ; else if ( cnt % 3 == 1 ) minPerfectCube *= 4 ; int i = 3 ;
while ( lcm > 1 ) { cnt = 0 ; while ( lcm % i == 0 ) { cnt ++ ; lcm /= i ; } if ( cnt % 3 == 1 ) minPerfectCube *= i * i ; else if ( cnt % 3 == 2 ) minPerfectCube *= i ; i += 2 ; }
return minPerfectCube ; }
public static void Main ( ) { int [ ] arr = { 10 , 125 , 14 , 42 , 100 } ; int n = arr . Length ; Console . WriteLine ( minPerfectCube ( arr , n ) ) ; } }
using System ; class GFG {
static bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static bool isStrongPrime ( int n ) {
if ( ! isPrime ( n ) n == 2 ) return false ;
int previous_prime = n - 1 ; int next_prime = n + 1 ;
while ( ! isPrime ( next_prime ) ) next_prime ++ ;
while ( ! isPrime ( previous_prime ) ) previous_prime -- ;
int mean = ( previous_prime + next_prime ) / 2 ;
if ( n > mean ) return true ; else return false ; }
public static void Main ( ) { int n = 11 ; if ( isStrongPrime ( n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; public class GFG {
static int countDigitsToBeRemoved ( int N , int K ) {
string s = Convert . ToString ( N ) ;
int res = 0 ;
int f_zero = 0 ; for ( int i = s . Length - 1 ; i >= 0 ; i -- ) { if ( K == 0 ) return res ; if ( s [ i ] == '0' ) {
f_zero = 1 ; K -- ; } else res ++ ; }
if ( K == 0 ) return res ; else if ( f_zero == 1 ) return s . Length - 1 ; return - 1 ; }
public static void Main ( ) { int N = 10904025 ; int K = 2 ; Console . Write ( countDigitsToBeRemoved ( N , K ) + " STRNEWLINE " ) ; N = 1000 ; K = 5 ; Console . Write ( countDigitsToBeRemoved ( N , K ) + " STRNEWLINE " ) ; N = 23985 ; K = 2 ; Console . Write ( countDigitsToBeRemoved ( N , K ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
public static double getSum ( int a , int n ) {
double sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) {
sum += ( i / Math . Pow ( a , i ) ) ; } return sum ; }
static public void Main ( ) { int a = 3 , n = 3 ;
Console . WriteLine ( getSum ( a , n ) ) ; } }
using System ; class GFG {
static int largestPrimeFactor ( int n ) {
int max = - 1 ;
while ( n % 2 == 0 ) { max = 2 ;
}
for ( int i = 3 ; i <= Math . Sqrt ( n ) ; i += 2 ) { while ( n % i == 0 ) { max = i ; n = n / i ; } }
if ( n > 2 ) max = n ; return max ; }
static bool checkUnusual ( int n ) {
int factor = largestPrimeFactor ( n ) ;
if ( factor > Math . Sqrt ( n ) ) { return true ; } else { return false ; } }
public static void Main ( ) { int n = 14 ; if ( checkUnusual ( n ) ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
using System ; public class GFG {
static void isHalfReducible ( int [ ] arr , int n , int m ) { int [ ] frequencyHash = new int [ m + 1 ] ; int i ; for ( i = 0 ; i < frequencyHash . Length ; i ++ ) frequencyHash [ i ] = 0 ; for ( i = 0 ; i < n ; i ++ ) { frequencyHash [ arr [ i ] % ( m + 1 ) ] ++ ; } for ( i = 0 ; i <= m ; i ++ ) { if ( frequencyHash [ i ] >= n / 2 ) break ; } if ( i <= m ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; }
public static void Main ( ) { int [ ] arr = { 8 , 16 , 32 , 3 , 12 } ; int n = arr . Length ; int m = 7 ; isHalfReducible ( arr , n , m ) ; } }
using System ; using System . Collections ; class GFG { static ArrayList arr = new ArrayList ( ) ;
static void generateDivisors ( int n ) {
for ( int i = 1 ; i <= Math . Sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) arr . Add ( i ) ;
{ arr . Add ( i ) ; arr . Add ( n / i ) ; } } } }
static double harmonicMean ( int n ) { generateDivisors ( n ) ;
double sum = 0.0 ; int len = arr . Count ;
for ( int i = 0 ; i < len ; i ++ ) sum = sum + n / ( int ) arr [ i ] ; sum = sum / n ;
return arr . Count / sum ; }
static bool isOreNumber ( int n ) {
double mean = harmonicMean ( n ) ;
if ( mean - Math . Floor ( mean ) == 0 ) return true ; else return false ; }
public static void Main ( ) { int n = 28 ; if ( isOreNumber ( n ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int MAX = 10000 ; static HashSet < int > s = new HashSet < int > ( ) ;
static void SieveOfEratosthenes ( ) {
Boolean [ ] prime = new Boolean [ MAX ] ; for ( int p = 0 ; p < MAX ; p ++ ) prime [ p ] = true ; prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
int product = 1 ; for ( int p = 2 ; p < MAX ; p ++ ) { if ( prime [ p ] ) {
product = product * p ;
s . Add ( product + 1 ) ; } } }
static Boolean isEuclid ( int n ) {
if ( s . Contains ( n ) ) return true ; else return false ; }
public static void Main ( String [ ] args ) {
SieveOfEratosthenes ( ) ;
int n = 31 ;
if ( isEuclid ( n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ;
n = 42 ;
if ( isEuclid ( n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) { return false ; } } return true ; }
static bool isPowerOfTwo ( int n ) { return n != 0 && ( ( n & ( n - 1 ) ) == 0 ) ; }
public static void Main ( ) { int n = 43 ;
if ( isPrime ( n ) && ( isPowerOfTwo ( n * 3 - 1 ) ) ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
using System ; class GFG {
static float area ( float a ) {
if ( a < 0 ) return - 1 ;
float area = ( float ) Math . Pow ( ( a * Math . Sqrt ( 3 ) ) / ( Math . Sqrt ( 2 ) ) , 2 ) ; return area ; }
public static void Main ( ) { float a = 5 ; Console . WriteLine ( area ( a ) ) ; } }
using System ; class GFG {
static int nthTerm ( int n ) { return 3 * ( int ) Math . Pow ( n , 2 ) - 4 * n + 2 ; }
public static void Main ( ) { int N = 4 ; Console . Write ( nthTerm ( N ) ) ; } }
using System ; class gfg {
public void calculateSum ( int n ) { double r = ( n * ( n + 1 ) / 2 + Math . Pow ( ( n * ( n + 1 ) / 2 ) , 2 ) ) ; Console . WriteLine ( " Sum ▁ = ▁ " + r ) ; }
public static int Main ( ) { gfg g = new gfg ( ) ;
int n = 3 ;
g . calculateSum ( n ) ; Console . Read ( ) ; return 0 ; } }
using System ; class GFG {
static bool arePermutations ( int [ ] a , int [ ] b , int n , int m ) { int sum1 = 0 , sum2 = 0 , mul1 = 1 , mul2 = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { sum1 += a [ i ] ; mul1 *= a [ i ] ; }
for ( int i = 0 ; i < m ; i ++ ) { sum2 += b [ i ] ; mul2 *= b [ i ] ; }
return ( ( sum1 == sum2 ) && ( mul1 == mul2 ) ) ; }
public static void Main ( ) { int [ ] a = { 1 , 3 , 2 } ; int [ ] b = { 3 , 1 , 2 } ; int n = a . Length ; int m = b . Length ; if ( arePermutations ( a , b , n , m ) == true ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static int Race ( int B , int C ) { int result = 0 ;
result = ( ( C * 100 ) / B ) ; return 100 - result ; }
public static void Main ( ) { int B = 10 ; int C = 28 ;
B = 100 - B ; C = 100 - C ; Console . Write ( Race ( B , C ) + " ▁ meters " ) ; } }
using System ; class GFG {
static float Time ( float [ ] arr , int n , float Emptypipe ) { float fill = 0 ; for ( int i = 0 ; i < n ; i ++ ) fill += 1 / arr [ i ] ; fill = fill - ( 1 / ( float ) Emptypipe ) ; return 1 / fill ; }
public static void Main ( ) { float [ ] arr = { 12 , 14 } ; float Emptypipe = 30 ; int n = arr . Length ; Console . WriteLine ( ( int ) ( Time ( arr , n , Emptypipe ) ) + " ▁ Hours " ) ; } }
using System ; class GFG {
static int check ( int n ) { int sum = 0 ;
while ( n != 0 ) { sum += n % 10 ; n = n / 10 ; }
if ( sum % 7 == 0 ) return 1 ; else return 0 ; }
public static void Main ( String [ ] args ) {
int n = 25 ; String s = ( check ( n ) == 1 ) ? " YES " : " NO " ; Console . WriteLine ( s ) ; } }
using System ; class GFG {
static bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static int SumOfPrimeDivisors ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { if ( n % i == 0 ) { if ( isPrime ( i ) ) sum += i ; } } return sum ; }
public static void Main ( ) { int n = 60 ; Console . WriteLine ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " + SumOfPrimeDivisors ( n ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static int Sum ( int N ) { int [ ] SumOfPrimeDivisors = new int [ N + 1 ] ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( SumOfPrimeDivisors [ i ] == 0 ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
public static void Main ( ) { int N = 60 ; Console . Write ( " Sum ▁ of ▁ prime ▁ " + " divisors ▁ of ▁ 60 ▁ is ▁ " + Sum ( N ) + " STRNEWLINE " ) ; } }
using System ; class GFG {
static long power ( long x , long y , long p ) {
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) > 0 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
public static void Main ( ) { long a = 3 ;
string b = "100000000000000000000000000" ; long remainderB = 0 ; long MOD = 1000000007 ;
for ( int i = 0 ; i < b . Length ; i ++ ) remainderB = ( remainderB * 10 + b [ i ] - '0' ) % ( MOD - 1 ) ; Console . WriteLine ( power ( a , remainderB , MOD ) ) ; } }
using System ; class GFG {
static string find_Square_369 ( string num ) { char a , b , c , d ;
if ( num [ 0 ] == '3' ) { a = '1' ; b = '0' ; c = '8' ; d = '9' ; }
else if ( num [ 0 ] == '6' ) { a = '4' ; b = '3' ; c = '5' ; d = '6' ; }
else { a = '9' ; b = '8' ; c = '0' ; d = '1' ; }
string result = " " ;
int size = num . Length ;
for ( int i = 1 ; i < size ; i ++ ) result += a ;
result += b ;
for ( int i = 1 ; i < size ; i ++ ) result += c ;
result += d ;
return result ; }
public static void Main ( ) { string num_3 , num_6 , num_9 ; num_3 = "3333" ; num_6 = "6666" ; num_9 = "9999" ; string result = " " ;
result = find_Square_369 ( num_3 ) ; Console . Write ( " Square ▁ of ▁ " + num_3 + " ▁ is ▁ : ▁ " + result + " STRNEWLINE " ) ;
result = find_Square_369 ( num_6 ) ; Console . Write ( " Square ▁ of ▁ " + num_9 + " ▁ is ▁ : ▁ " + result + " STRNEWLINE " ) ;
result = find_Square_369 ( num_9 ) ; Console . Write ( " Square ▁ of ▁ " + num_9 + " ▁ is ▁ : ▁ " + result + " STRNEWLINE " ) ; } }
using System ; class GFG { public static void Main ( ) { long ans = 1 ; long mod = ( long ) 1000000007 * 120 ; for ( int i = 0 ; i < 5 ; i ++ ) ans = ( ans * ( 55555 - i ) ) % mod ; ans = ans / 120 ; Console . Write ( " Answer ▁ using ▁ " + " shortcut : ▁ " + ans ) ; } }
using System ; class GFG {
static int fact ( int n ) { if ( n == 0 n == 1 ) return 1 ; int ans = 1 ; for ( int i = 1 ; i <= n ; i ++ ) ans = ans * i ; return ans ; }
static int nCr ( int n , int r ) { int Nr = n , Dr = 1 , ans = 1 ; for ( int i = 1 ; i <= r ; i ++ ) { ans = ( ans * Nr ) / ( Dr ) ; Nr -- ; Dr ++ ; } return ans ; }
static int solve ( int n ) { int N = 2 * n - 2 ; int R = n - 1 ; return nCr ( N , R ) * fact ( n - 1 ) ; }
public static void Main ( ) { int n = 6 ; Console . WriteLine ( solve ( n ) ) ; } }
using System ; class GFG { static void pythagoreanTriplet ( int n ) {
for ( int i = 1 ; i <= n / 3 ; i ++ ) {
for ( int j = i + 1 ; j <= n / 2 ; j ++ ) { int k = n - i - j ; if ( i * i + j * j == k * k ) { Console . Write ( i + " , ▁ " + j + " , ▁ " + k ) ; return ; } } } Console . Write ( " No ▁ Triplet " ) ; }
public static void Main ( ) { int n = 12 ; pythagoreanTriplet ( n ) ; } }
using System ; class GFG {
static int factorial ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
static void series ( int A , int X , int n ) {
int nFact = factorial ( n ) ;
for ( int i = 0 ; i < n + 1 ; i ++ ) {
int niFact = factorial ( n - i ) ; int iFact = factorial ( i ) ;
int aPow = ( int ) Math . Pow ( A , n - i ) ; int xPow = ( int ) Math . Pow ( X , i ) ;
Console . Write ( ( nFact * aPow * xPow ) / ( niFact * iFact ) + " ▁ " ) ; } }
public static void Main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; } }
using System ; class GFG {
static int seiresSum ( int n , int [ ] a ) { int res = 0 , i ; for ( i = 0 ; i < 2 * n ; i ++ ) { if ( i % 2 == 0 ) res += a [ i ] * a [ i ] ; else res -= a [ i ] * a [ i ] ; } return res ; }
public static void Main ( ) { int n = 2 ; int [ ] a = { 1 , 2 , 3 , 4 } ; Console . WriteLine ( seiresSum ( n , a ) ) ; } }
using System ; class GFG {
static int power ( int n , int r ) {
int count = 0 ; for ( int i = r ; ( n / i ) >= 1 ; i = i * r ) count += n / i ; return count ; }
public static void Main ( ) { int n = 6 , r = 3 ; Console . WriteLine ( power ( n , r ) ) ; } }
using System ; class GFG {
static int avg_of_odd_num ( int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += ( 2 * i + 1 ) ;
return sum / n ; }
public static void Main ( ) { int n = 20 ; avg_of_odd_num ( n ) ; Console . Write ( avg_of_odd_num ( n ) ) ; } }
using System ; class GFG {
static int avg_of_odd_num ( int n ) { return n ; }
public static void Main ( ) { int n = 8 ; Console . Write ( avg_of_odd_num ( n ) ) ; } }
using System ; class GFG {
static void fib ( int [ ] f , int N ) {
f [ 1 ] = 1 ; f [ 2 ] = 1 ; for ( int i = 3 ; i <= N ; i ++ )
f [ i ] = f [ i - 1 ] + f [ i - 2 ] ; } static void fiboTriangle ( int n ) {
int N = n * ( n + 1 ) / 2 ; int [ ] f = new int [ N + 1 ] ; fib ( f , N ) ;
int fiboNum = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) Console . Write ( f [ fiboNum ++ ] + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( ) { int n = 5 ; fiboTriangle ( n ) ; } }
using System ; class GFG {
static int averageOdd ( int n ) { if ( n % 2 == 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } int sum = 0 , count = 0 ; while ( n >= 1 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
public static void Main ( ) { int n = 15 ; Console . Write ( averageOdd ( n ) ) ; } }
using System ; class GFG { class Rational { public int nume , deno ; public Rational ( int nume , int deno ) { this . nume = nume ; this . deno = deno ; } } ;
static int lcm ( int a , int b ) { return ( a * b ) / ( __gcd ( a , b ) ) ; }
static Rational maxRational ( Rational first , Rational sec ) {
int k = lcm ( first . deno , sec . deno ) ;
int nume1 = first . nume ; int nume2 = sec . nume ; nume1 *= k / ( first . deno ) ; nume2 *= k / ( sec . deno ) ; return ( nume2 < nume1 ) ? first : sec ; } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; }
public static void Main ( String [ ] args ) { Rational first = new Rational ( 3 , 2 ) ; Rational sec = new Rational ( 3 , 4 ) ; Rational res = maxRational ( first , sec ) ; Console . Write ( res . nume + " / " + res . deno ) ; } }
using System ; public class GfG {
public static int TrinomialValue ( int n , int k ) {
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ; }
public static void printTrinomial ( int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = - i ; j <= 0 ; j ++ ) Console . Write ( TrinomialValue ( i , j ) + " ▁ " ) ;
for ( int j = 1 ; j <= i ; j ++ ) Console . Write ( TrinomialValue ( i , j ) + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( ) { int n = 4 ; printTrinomial ( n ) ; } }
using System ; class GFG {
static int sumOfLargePrimeFactor ( int n ) {
int [ ] prime = new int [ n + 1 ] ; int sum = 0 ; for ( int i = 1 ; i < n + 1 ; i ++ ) prime [ i ] = 0 ; int max = n / 2 ; for ( int p = 2 ; p <= max ; p ++ ) {
if ( prime [ p ] == 0 ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = p ; } }
for ( int p = 2 ; p <= n ; p ++ ) {
if ( prime [ p ] != 0 ) sum += prime [ p ] ;
else sum += ; }
return sum ; }
public static void Main ( ) { int n = 12 ; Console . WriteLine ( " Sum ▁ = ▁ " + sumOfLargePrimeFactor ( n ) ) ; } }
using System ; class GFG {
static int calculate_sum ( int a , int N ) {
int m = N / a ;
int sum = m * ( m + 1 ) / 2 ;
int ans = a * sum ; return ans ; }
public static void Main ( ) { int a = 7 , N = 49 ; Console . WriteLine ( " Sum ▁ of ▁ multiples ▁ of ▁ " + a + " ▁ up ▁ to ▁ " + N + " ▁ = ▁ " + calculate_sum ( a , N ) ) ; } }
class GFG {
static long ispowerof2 ( long num ) { if ( ( num & ( num - 1 ) ) == 0 ) return 1 ; return 0 ; }
public static void Main ( ) { long num = 549755813888 ; System . Console . WriteLine ( ispowerof2 ( num ) ) ; } }
using System ; class GFG {
static int counDivisors ( int X ) {
int count = 0 ;
for ( int i = 1 ; i <= X ; ++ i ) { if ( X % i == 0 ) { count ++ ; } }
return count ; }
static int countDivisorsMult ( int [ ] arr , int n ) {
int mul = 1 ; for ( int i = 0 ; i < n ; ++ i ) mul *= arr [ i ] ;
return counDivisors ( mul ) ; }
public static void Main ( ) { int [ ] arr = { 2 , 4 , 6 } ; int n = arr . Length ; Console . Write ( countDivisorsMult ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static void SieveOfEratosthenes ( int largest , List < int > prime ) {
bool [ ] isPrime = new bool [ largest + 1 ] ; Array . Fill ( isPrime , true ) ; for ( int p = 2 ; p * p <= largest ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( int i = p * 2 ; i <= largest ; i += p ) isPrime [ i ] = false ; } }
for ( int p = 2 ; p <= largest ; p ++ ) if ( isPrime [ p ] ) prime . Add ( p ) ; }
static long countDivisorsMult ( int [ ] arr , int n ) {
int largest = 0 ; foreach ( int a in arr ) { largest = Math . Max ( largest , a ) ; } List < int > prime = new List < int > ( ) ; SieveOfEratosthenes ( largest , prime ) ;
Dictionary < int , int > mp = new Dictionary < int , int > ( ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < prime . Count ; j ++ ) { while ( arr [ i ] > 1 && arr [ i ] % prime [ j ] == 0 ) { arr [ i ] /= prime [ j ] ; if ( mp . ContainsKey ( prime [ j ] ) ) { mp [ prime [ j ] ] ++ ; } else { mp . Add ( prime [ j ] , 1 ) ; } } } if ( arr [ i ] != 1 ) { if ( mp . ContainsKey ( arr [ i ] ) ) { mp [ arr [ i ] ] ++ ; } else { mp . Add ( arr [ i ] , 1 ) ; } } }
long res = 1 ; foreach ( KeyValuePair < int , int > it in mp ) res *= ( it . Value + 1L ) ; return res ; }
static public void Main ( ) { int [ ] arr = { 2 , 4 , 6 } ; int n = arr . Length ; Console . WriteLine ( countDivisorsMult ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class solution {
static void findPrimeNos ( int L , int R , Dictionary < int , int > M , int K ) {
for ( int i = L ; i <= R ; i ++ ) { if ( M . ContainsKey ( i ) ) M . Add ( i , M [ i ] + 1 ) ; else M . Add ( i , 1 ) ; }
if ( M [ 1 ] != 0 ) { M . Remove ( 1 ) ; }
for ( int i = 2 ; i <= Math . Sqrt ( R ) ; i ++ ) { int multiple = 2 ; while ( ( i * multiple ) <= R ) {
if ( M . ContainsKey ( i * multiple ) ) {
M . Remove ( i * multiple ) ; }
multiple ++ ; } }
foreach ( KeyValuePair < int , int > entry in M ) {
if ( M . ContainsKey ( entry . Key + K ) ) { Console . Write ( " ( " + entry . Key + " , ▁ " + ( entry . Key + K ) + " ) ▁ " ) ; } } }
static void getPrimePairs ( int L , int R , int K ) { Dictionary < int , int > M = new Dictionary < int , int > ( ) ;
findPrimeNos ( L , R , M , K ) ; }
public static void Main ( String [ ] args ) {
int L = 1 , R = 19 ;
int K = 6 ;
getPrimePairs ( L , R , K ) ; } }
using System ; class GFG {
static int enneacontahexagonNum ( int n ) { return ( 94 * n * n - 92 * n ) / 2 ; }
public static void Main ( ) { int n = 3 ; Console . Write ( enneacontahexagonNum ( n ) ) ; } }
using System ; class GFG {
static void find_composite_nos ( int n ) { Console . WriteLine ( 9 * n + " ▁ " + 8 * n ) ; }
public static void Main ( ) { int n = 4 ; find_composite_nos ( n ) ; } }
using System ; using System . Linq ; class GFG {
static int freqPairs ( int [ ] arr , int n ) {
int max = arr . Max ( ) ;
int [ ] freq = new int [ max + 1 ] ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; }
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 2 * arr [ i ] ; j <= max ; j += arr [ i ] ) {
if ( freq [ j ] >= 1 ) { count += freq [ j ] ; } }
if ( freq [ arr [ i ] ] > 1 ) { count += freq [ arr [ i ] ] - 1 ; freq [ arr [ i ] ] -- ; } } return count ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 2 , 4 , 2 , 6 } ; int n = arr . Length ; Console . WriteLine ( freqPairs ( arr , n ) ) ; } }
using System ; class GFG {
static double Nth_Term ( int n ) { return ( 2 * Math . Pow ( n , 3 ) - 3 * Math . Pow ( n , 2 ) + n + 6 ) / 6 ; }
static public void Main ( ) { int N = 8 ; Console . WriteLine ( Nth_Term ( N ) ) ; } }
using System ; class GFG {
static int printNthElement ( int n ) {
int [ ] arr = new int [ n + 1 ] ; arr [ 1 ] = 3 ; arr [ 2 ] = 5 ; for ( int i = 3 ; i <= n ; i ++ ) {
if ( i % 2 != 0 ) arr [ i ] = arr [ i / 2 ] * 10 + 3 ; else arr [ i ] = arr [ ( i / 2 ) - 1 ] * 10 + 5 ; } return arr [ n ] ; }
static void Main ( ) { int n = 6 ; Console . WriteLine ( printNthElement ( n ) ) ; } }
using System ; class GFG { public int nthTerm ( int N ) {
return ( N * ( ( N / 2 ) + ( ( N % 2 ) * 2 ) + N ) ) ; }
public static void Main ( ) {
int N = 5 ;
GFG a = new GFG ( ) ;
Console . WriteLine ( " Nth ▁ term ▁ for ▁ N ▁ = ▁ " + N + " ▁ : ▁ " + a . nthTerm ( N ) ) ; } }
using System ; public class GFG {
static void series ( int A , int X , int n ) {
int term = ( int ) Math . Pow ( A , n ) ; Console . Write ( term + " ▁ " ) ;
for ( int i = 1 ; i <= n ; i ++ ) {
term = term * X * ( n - i + 1 ) / ( i * A ) ; Console . Write ( term + " ▁ " ) ; } }
public static void Main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; } }
using System ; class GFG {
static bool Div_by_8 ( int n ) { return ( ( ( n >> 3 ) << 3 ) == n ) ; }
public static void Main ( ) { int n = 16 ; if ( Div_by_8 ( n ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; class GFG {
static int averageEven ( int n ) { if ( n % 2 != 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } int sum = 0 , count = 0 ; while ( n >= 2 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
public static void Main ( ) { int n = 16 ; Console . Write ( averageEven ( n ) ) ; } }
using System ; class GFG {
static int averageEven ( int n ) { if ( n % 2 != 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } return ( n + 2 ) / 2 ; }
public static void Main ( ) { int n = 16 ; Console . Write ( averageEven ( n ) ) ; } }
using System ; class GFG {
static int gcd ( int a , int b ) {
if ( a == 0 b == 0 ) return 0 ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
static int cpFact ( int x , int y ) { while ( gcd ( x , y ) != 1 ) { x = x / gcd ( x , y ) ; } return x ; }
public static void Main ( ) { int x = 15 ; int y = 3 ; Console . WriteLine ( cpFact ( x , y ) ) ; x = 14 ; y = 28 ; Console . WriteLine ( cpFact ( x , y ) ) ; x = 7 ; y = 3 ; Console . WriteLine ( cpFact ( x , y ) ) ; } }
using System ; public class GfG {
public static int counLastDigitK ( int low , int high , int k ) { int count = 0 ; for ( int i = low ; i <= high ; i ++ ) if ( i % 10 == k ) count ++ ; return count ; }
public static void Main ( ) { int low = 3 , high = 35 , k = 3 ; Console . WriteLine ( counLastDigitK ( low , high , k ) ) ; } }
using System ; class GFG { public static void printTaxicab2 ( int N ) {
int i = 1 , count = 0 ; while ( count < N ) { int int_count = 0 ;
for ( int j = 1 ; j <= Math . Pow ( i , 1.0 / 3 ) ; j ++ ) for ( int k = j + 1 ; k <= Math . Pow ( i , 1.0 / 3 ) ; k ++ ) if ( j * j * j + k * k * k == i ) int_count ++ ;
if ( int_count == 2 ) { count ++ ; Console . WriteLine ( count + " ▁ " + i ) ; } i ++ ; } }
public static void Main ( ) { int N = 5 ; printTaxicab2 ( N ) ; } }
using System ; namespace Composite { public class GFG { public static bool isComposite ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return false ;
if ( n % 2 == 0 n % 3 == 0 ) return true ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return true ; return false ; }
public static void Main ( ) { if ( isComposite ( 11 ) ) Console . WriteLine ( " true " ) ; else Console . WriteLine ( " false " ) ; if ( isComposite ( 15 ) ) Console . WriteLine ( " true " ) ; else Console . WriteLine ( " false " ) ; } } }
using System ; class GFG {
static bool isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
static int findPrime ( int n ) { int num = n + 1 ;
while ( num > 0 ) {
if ( isPrime ( num ) ) return num ;
num = num + 1 ; } return 0 ; }
static int minNumber ( int [ ] arr , int n ) { int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( isPrime ( sum ) ) return 0 ;
int num = findPrime ( sum ) ;
return num - sum ; }
public static void Main ( ) { int [ ] arr = { 2 , 4 , 6 , 8 , 12 } ; int n = arr . Length ; Console . Write ( minNumber ( arr , n ) ) ; } }
using System ; class Division {
static int fac ( int n ) { if ( n == 0 ) return 1 ; return n * fac ( n - 1 ) ; }
static int div ( int x ) { int ans = 0 ; for ( int i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
static int sumFactDiv ( int n ) { return div ( fac ( n ) ) ; }
public static void Main ( ) { int n = 4 ; Console . Write ( sumFactDiv ( n ) ) ; } }
using System ; using System . Collections ; class GFG {
static ArrayList allPrimes = new ArrayList ( ) ;
static void sieve ( int n ) {
bool [ ] prime = new bool [ n + 1 ] ;
for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == false ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = true ; } }
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] == false ) allPrimes . Add ( p ) ; }
static int factorialDivisors ( int n ) {
int result = 1 ;
for ( int i = 0 ; i < allPrimes . Count ; i ++ ) {
int p = ( int ) allPrimes [ i ] ;
int exp = 0 ; while ( p <= n ) { exp = exp + ( n / p ) ; p = p * ( int ) allPrimes [ i ] ; }
result = result * ( ( int ) Math . Pow ( ( int ) allPrimes [ i ] , exp + 1 ) - 1 ) / ( ( int ) allPrimes [ i ] - 1 ) ; }
return result ; }
static void Main ( ) { Console . WriteLine ( factorialDivisors ( 4 ) ) ; } }
using System ; class GFG {
static bool checkPandigital ( int b , string n ) {
if ( n . Length < b ) return false ; bool [ ] hash = new bool [ b ] ; for ( int i = 0 ; i < b ; i ++ ) hash [ i ] = false ;
for ( int i = 0 ; i < n . Length ; i ++ ) {
if ( n [ i ] >= '0' && n [ i ] <= '9' ) hash [ n [ i ] - '0' ] = true ;
else if ( n [ i ] - ' A ' <= b - 11 ) [ n [ i ] - ' A ' + 10 ] = true ; }
for ( int i = 0 ; i < b ; i ++ ) if ( hash [ i ] == false ) return false ; return true ; }
public static void Main ( ) { int b = 13 ; String n = "1298450376ABC " ; if ( checkPandigital ( b , n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
using System ; class GFG {
static int convert ( int m , int n ) { if ( m == n ) return 0 ;
if ( m > n ) return m - n ;
if ( m <= 0 && n > 0 ) return - 1 ;
if ( n % 2 == 1 )
return 1 + convert ( m , n + 1 ) ;
else
return 1 + convert ( m , n / 2 ) ; }
public static void Main ( ) { int m = 3 , n = 11 ; Console . Write ( " Minimum ▁ number ▁ of ▁ " + " operations ▁ : ▁ " + convert ( m , n ) ) ; } }
using System ; using System . Collections ; class GFG { static int MAX = 10000 ; static int [ ] prodDig = new int [ MAX ] ;
static int getDigitProduct ( int x ) {
if ( x < 10 ) return x ;
if ( prodDig [ x ] != 0 ) return prodDig [ x ] ;
int prod = ( x % 10 ) * getDigitProduct ( x / 10 ) ; return ( prodDig [ x ] = prod ) ; }
static void findSeed ( int n ) {
ArrayList res = new ArrayList ( ) ; for ( int i = 1 ; i <= n / 2 ; i ++ ) if ( i * getDigitProduct ( i ) == n ) res . Add ( i ) ;
if ( res . Count == 0 ) { Console . WriteLine ( " NO ▁ seed ▁ exists " ) ; return ; }
for ( int i = 0 ; i < res . Count ; i ++ ) Console . WriteLine ( res [ i ] + " ▁ " ) ; }
static void Main ( ) { int n = 138 ; findSeed ( n ) ; } }
using System ; class GFG {
static int maxPrimefactorNum ( int N ) { int [ ] arr = new int [ N + 5 ] ;
for ( int i = 2 ; i * i <= N ; i ++ ) { if ( arr [ i ] == 0 ) { for ( int j = 2 * i ; j <= N ; j += i ) { arr [ j ] ++ ; } } arr [ i ] = 1 ; } int maxval = 0 , maxint = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( arr [ i ] > maxval ) { maxval = arr [ i ] ; maxint = i ; } } return maxint ; }
public static void Main ( ) { int N = 40 ; Console . WriteLine ( maxPrimefactorNum ( N ) ) ; } }
using System ; class GFG {
public static long SubArraySum ( int [ ] arr , int n ) { long result = 0 ;
for ( int i = 0 ; i < n ; i ++ ) result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) ;
return result ; }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int n = arr . Length ; Console . WriteLine ( " Sum ▁ of ▁ SubArray : ▁ " + SubArraySum ( arr , n ) ) ; } }
using System ; class GFG { public static int highestPowerof2 ( int n ) { int res = 0 ; for ( int i = n ; i >= 1 ; i -- ) {
if ( ( i & ( i - 1 ) ) == 0 ) { res = i ; break ; } } return res ; }
static public void Main ( ) { int n = 10 ; Console . WriteLine ( highestPowerof2 ( n ) ) ; } }
using System ; class GFG {
static void findPairs ( int n ) {
int cubeRoot = ( int ) Math . Pow ( n , 1.0 / 3.0 ) ;
int [ ] cube = new int [ cubeRoot + 1 ] ;
for ( int i = 1 ; i <= cubeRoot ; i ++ ) cube [ i ] = i * i * i ;
int l = 1 ; int r = cubeRoot ; while ( l < r ) { if ( cube [ l ] + cube [ r ] < n ) l ++ ; else if ( cube [ l ] + cube [ r ] > n ) r -- ; else { Console . WriteLine ( " ( " + l + " , ▁ " + r + " ) " ) ; l ++ ; r -- ; } } }
public static void Main ( ) { int n = 20683 ; findPairs ( n ) ; } }
using System ; using System . Collections . Generic ; class GFG { class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } }
static void findPairs ( int n ) {
int cubeRoot = ( int ) Math . Pow ( n , 1.0 / 3.0 ) ;
Dictionary < int , pair > s = new Dictionary < int , pair > ( ) ;
for ( int x = 1 ; x < cubeRoot ; x ++ ) { for ( int y = x + 1 ; y <= cubeRoot ; y ++ ) {
int sum = x * x * x + y * y * y ;
if ( sum != n ) continue ;
if ( s . ContainsKey ( sum ) ) { Console . Write ( " ( " + s [ sum ] . first + " , ▁ " + s [ sum ] . second + " ) ▁ and ▁ ( " + x + " , ▁ " + y + " ) " + " STRNEWLINE " ) ; } else
s . Add ( sum , new pair ( x , y ) ) ; } } }
public static void Main ( String [ ] args ) { int n = 13832 ; findPairs ( n ) ; } }
using System ; class GFG {
static int gcd ( int a , int b ) { while ( b != 0 ) { int t = b ; b = a % b ; a = t ; } return a ; }
static int findMinDiff ( int a , int b , int x , int y ) {
int g = gcd ( a , b ) ;
int diff = Math . Abs ( x - y ) % g ; return Math . Min ( diff , g - diff ) ; }
static void Main ( ) { int a = 20 , b = 52 , x = 5 , y = 7 ; Console . WriteLine ( findMinDiff ( a , b , x , y ) ) ; } }
using System ; class GFG {
static void printDivisors ( int n ) {
int [ ] v = new int [ n ] ; int t = 0 ; for ( int i = 1 ; i <= Math . Sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) Console . Write ( i + " ▁ " ) ; else { Console . Write ( i + " ▁ " ) ;
v [ t ++ ] = n / i ; } } }
for ( int i = t - 1 ; i >= 0 ; i -- ) Console . Write ( v [ i ] + " ▁ " ) ; }
public static void Main ( ) { Console . Write ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ) ; printDivisors ( 100 ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG {
static void printDivisors ( int n ) { for ( int i = 1 ; i * i < n ; i ++ ) { if ( n % i == 0 ) Console . Write ( i + " ▁ " ) ; } for ( int i = ( int ) Math . Sqrt ( n ) ; i >= 1 ; i -- ) { if ( n % i == 0 ) Console . Write ( n / i + " ▁ " ) ; } }
public static void Main ( string [ ] arg ) { Console . Write ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ) ; printDivisors ( 100 ) ; } }
using System ; class GFG {
static void printDivisors ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) Console . Write ( i + " ▁ " ) ; }
public static void Main ( ) { Console . Write ( " The ▁ divisors ▁ of " , " ▁ 100 ▁ are : ▁ " ) ; printDivisors ( 100 ) ; ; } }
using System ; class GFG {
static void printDivisors ( int n ) {
for ( int i = 1 ; i <= Math . Sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) Console . Write ( i + " ▁ " ) ;
else Console . ( + " ▁ " + n / i + " ▁ " ) ; } } }
public static void Main ( ) { Console . Write ( " The ▁ divisors ▁ of ▁ " + "100 ▁ are : ▁ STRNEWLINE " ) ; printDivisors ( 100 ) ; } }
using System ; class GFG { static int SieveOfAtkin ( int limit ) {
if ( limit > 2 ) Console . Write ( 2 + " ▁ " ) ; if ( limit > 3 ) Console . Write ( 3 + " ▁ " ) ;
bool [ ] sieve = new bool [ limit ] ; for ( int i = 0 ; i < limit ; i ++ ) sieve [ i ] = false ;
for ( int x = 1 ; x * x < limit ; x ++ ) { for ( int y = 1 ; y * y < limit ; y ++ ) {
int n = ( 4 * x * x ) + ( y * y ) ; if ( n <= limit && ( n % 12 == 1 n % 12 == 5 ) ) sieve [ n ] ^= true ; n = ( 3 * x * x ) + ( y * y ) ; if ( n <= limit && n % 12 == 7 ) sieve [ n ] ^= true ; n = ( 3 * x * x ) - ( y * y ) ; if ( x > y && n <= limit && n % 12 == 11 ) sieve [ n ] ^= true ; } }
for ( int r = 5 ; r * r < limit ; r ++ ) { if ( sieve [ r ] ) { for ( int i = r * r ; i < limit ; i += r * r ) sieve [ i ] = false ; } }
for ( int a = 5 ; a < limit ; a ++ ) if ( sieve [ a ] ) Console . Write ( a + " ▁ " ) ; return 0 ; }
public static void Main ( ) { int limit = 20 ; SieveOfAtkin ( limit ) ; } }
using System ; class GFG { static bool isInside ( int circle_x , int circle_y , int rad , int x , int y ) {
if ( ( x - circle_x ) * ( x - circle_x ) + ( y - circle_y ) * ( y - circle_y ) <= rad * rad ) return true ; else return false ; }
public static void Main ( ) { int x = 1 , y = 1 ; int circle_x = 0 , circle_y = 1 , rad = 2 ; if ( isInside ( circle_x , circle_y , rad , x , y ) ) Console . Write ( " Inside " ) ; else Console . Write ( " Outside " ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int eval ( int a , char op , int b ) { if ( op == ' + ' ) { return a + b ; } if ( op == ' - ' ) { return a - b ; } if ( op == ' * ' ) { return a * b ; } return int . MaxValue ; }
static List < int > evaluateAll ( String expr , int low , int high ) {
List < int > res = new List < int > ( ) ;
if ( low == high ) { res . Add ( expr [ low ] - '0' ) ; return res ; }
if ( low == ( high - 2 ) ) { int num = eval ( expr [ low ] - '0' , expr [ low + 1 ] , expr [ low + 2 ] - '0' ) ; res . Add ( num ) ; return res ; }
for ( int i = low + 1 ; i <= high ; i += 2 ) {
List < int > l = evaluateAll ( expr , low , i - 1 ) ;
List < int > r = evaluateAll ( expr , i + 1 , high ) ;
for ( int s1 = 0 ; s1 < l . Count ; s1 ++ ) {
for ( int s2 = 0 ; s2 < r . Count ; s2 ++ ) {
int val = eval ( l [ s1 ] , expr [ i ] , r [ s2 ] ) ; res . Add ( val ) ; } } } return res ; }
public static void Main ( ) { String expr = "1*2 + 3*4" ; int len = expr . Length ; List < int > ans = evaluateAll ( expr , 0 , len - 1 ) ; for ( int i = 0 ; i < ans . Count ; i ++ ) { Console . WriteLine ( ans [ i ] ) ; } } }
using System ; class GFG {
static bool isLucky ( int n ) {
bool [ ] arr = new bool [ 10 ] ; for ( int i = 0 ; i < 10 ; i ++ ) arr [ i ] = false ;
while ( n > 0 ) {
int digit = n % 10 ;
if ( arr [ digit ] ) return false ;
arr [ digit ] = true ;
n = n / 10 ; } return true ; }
public static void Main ( ) { int [ ] arr = { 1291 , 897 , 4566 , 1232 , 80 , 700 } ; int n = arr . Length ; for ( int i = 0 ; i < n ; i ++ ) if ( isLucky ( arr [ i ] ) ) Console . Write ( arr [ i ] + " ▁ is ▁ Lucky ▁ STRNEWLINE " ) ; else Console . Write ( arr [ i ] + " ▁ is ▁ not ▁ Lucky ▁ STRNEWLINE " ) ; } }
using System ; class GFG { static void printSquares ( int n ) {
int square = 0 , odd = 1 ;
for ( int x = 0 ; x < n ; x ++ ) {
Console . Write ( square + " ▁ " ) ;
square = square + odd ; odd = odd + 2 ; } }
public static void Main ( ) { int n = 5 ; printSquares ( n ) ; } }
using System ; class GFG { static int rev_num = 0 ; static int base_pos = 1 ; static int reversDigits ( int num ) { if ( num > 0 ) { reversDigits ( num / 10 ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
public static void Main ( ) { int num = 4562 ; Console . WriteLine ( reversDigits ( num ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int RecursiveFunction ( List < int > re , int bit ) {
if ( re . Count == 0 bit < 0 ) return 0 ; List < int > curr_on = new List < int > ( ) ; List < int > curr_off = new List < int > ( ) ; for ( int i = 0 ; i < re . Count ; i ++ ) {
if ( ( ( re [ i ] >> bit ) & 1 ) == 0 ) curr_off . Add ( re [ i ] ) ;
else curr_on . ( re [ i ] ) ; }
if ( curr_off . Count == 0 ) return RecursiveFunction ( curr_on , bit - 1 ) ;
if ( curr_on . Count == 0 ) return RecursiveFunction ( curr_off , bit - 1 ) ;
return Math . Min ( RecursiveFunction ( curr_off , bit - 1 ) , RecursiveFunction ( curr_on , bit - 1 ) ) + ( 1 << bit ) ; }
static void PrintMinimum ( int [ ] a , int n ) { List < int > v = new List < int > ( ) ;
for ( int i = 0 ; i < n ; i ++ ) v . Add ( a [ i ] ) ;
Console . WriteLine ( RecursiveFunction ( v , 30 ) ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 2 , 1 } ; int size = arr . Length ; PrintMinimum ( arr , size ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int cntElements ( int [ ] arr , int n ) {
int cnt = 0 ;
for ( int i = 0 ; i < n - 2 ; i ++ ) {
if ( arr [ i ] == ( arr [ i + 1 ] ^ arr [ i + 2 ] ) ) { cnt ++ ; } } return cnt ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 4 , 2 , 1 , 3 , 7 , 8 } ; int n = arr . Length ; Console . WriteLine ( cntElements ( arr , n ) ) ; } }
using System ; class GFG {
static int xor_triplet ( int [ ] arr , int n ) {
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i + 1 ; j < n ; j ++ ) {
for ( int k = j ; k < n ; k ++ ) { int xor1 = 0 , xor2 = 0 ;
for ( int x = i ; x < j ; x ++ ) { xor1 ^= arr [ x ] ; }
for ( int x = j ; x <= k ; x ++ ) { xor2 ^= arr [ x ] ; }
if ( xor1 == xor2 ) { ans ++ ; } } } } return ans ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . Length ;
Console . WriteLine ( xor_triplet ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int N = 100005 ; static int n , k ;
static List < int > [ ] al = new List < int > [ N ] ; static long Ideal_pair ; static long [ ] bit = new long [ N ] ; static bool [ ] root_node = new bool [ N ] ;
static long bit_q ( int i , int j ) { long sum = 0 ; while ( j > 0 ) { sum += bit [ j ] ; j -= ( j & ( j * - 1 ) ) ; } i -- ; while ( i > 0 ) { sum -= bit [ i ] ; i -= ( i & ( i * - 1 ) ) ; } return sum ; }
static void bit_up ( int i , long diff ) { while ( i <= n ) { bit [ i ] += diff ; i += i & - i ; } }
static void dfs ( int node ) { Ideal_pair += bit_q ( Math . Max ( 1 , node - k ) , Math . Min ( n , node + k ) ) ; bit_up ( node , 1 ) ; for ( int i = 0 ; i < al [ node ] . Count ; i ++ ) dfs ( al [ node ] [ i ] ) ; bit_up ( node , - 1 ) ; }
static void initialise ( ) { Ideal_pair = 0 ; for ( int i = 0 ; i <= n ; i ++ ) { root_node [ i ] = true ; bit [ i ] = 0 ; } }
static void Add_Edge ( int x , int y ) { al [ x ] . Add ( y ) ; root_node [ y ] = false ; }
static long Idealpairs ( ) {
int r = - 1 ; for ( int i = 1 ; i <= n ; i ++ ) if ( root_node [ i ] ) { r = i ; break ; } dfs ( r ) ; return Ideal_pair ; }
public static void Main ( String [ ] args ) { n = 6 ; k = 3 ; for ( int i = 0 ; i < al . Length ; i ++ ) al [ i ] = new List < int > ( ) ; initialise ( ) ;
Add_Edge ( 1 , 2 ) ; Add_Edge ( 1 , 3 ) ; Add_Edge ( 3 , 4 ) ; Add_Edge ( 3 , 5 ) ; Add_Edge ( 3 , 6 ) ;
Console . Write ( Idealpairs ( ) ) ; } }
using System ; public class GFG {
static void printSubsets ( int n ) { for ( int i = n ; i > 0 ; i = ( i - 1 ) & n ) Console . Write ( i + " ▁ " ) ; Console . WriteLine ( "0" ) ; }
static public void Main ( ) { int n = 9 ; printSubsets ( n ) ; } }
using System ; class GFG {
static bool isDivisibleby17 ( int n ) {
if ( n == 0 n == 17 ) return true ;
if ( n < 17 ) return false ;
return isDivisibleby17 ( ( int ) ( n >> 4 ) - ( int ) ( n & 15 ) ) ; }
public static void Main ( ) { int n = 35 ; if ( isDivisibleby17 ( n ) == true ) Console . WriteLine ( n + " is ▁ divisible ▁ by ▁ 17" ) ; else Console . WriteLine ( n + " ▁ is ▁ not ▁ divisible ▁ by ▁ 17" ) ; } }
using System ; class GFG {
static long answer ( long n ) {
long m = 2 ;
long ans = 1 ; long r = 1 ;
while ( r < n ) {
r = ( ( long ) Math . Pow ( 2 , m ) - 1 ) * ( ( long ) Math . Pow ( 2 , m - 1 ) ) ;
if ( r < n ) ans = r ;
m ++ ; } return ans ; }
static public void Main ( ) { long n = 7 ; Console . WriteLine ( answer ( n ) ) ; } }
using System ; class GFG { static int setBitNumber ( int n ) { if ( n == 0 ) return 0 ; int msb = 0 ; n = n / 2 ; while ( n != 0 ) { n = n / 2 ; msb ++ ; } return ( 1 << msb ) ; }
static public void Main ( ) { int n = 0 ; Console . WriteLine ( setBitNumber ( n ) ) ; } }
using System ; class GFG { static int setBitNumber ( int n ) {
n |= n >> 1 ;
n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ;
n = n + 1 ;
return ( n >> 1 ) ; }
public static void Main ( ) { int n = 273 ; Console . WriteLine ( setBitNumber ( n ) ) ; } }
using System ; class GFG { public static int countTrailingZero ( int x ) { int count = 0 ; while ( ( x & 1 ) == 0 ) { x = x >> 1 ; count ++ ; } return count ; }
static public void Main ( ) { Console . WriteLine ( countTrailingZero ( 11 ) ) ; } }
using System ; class GFG { static int countTrailingZero ( int x ) {
int [ ] lookup = { 32 , 0 , 1 , 26 , 2 , 23 , 27 , 0 , 3 , 16 , 24 , 30 , 28 , 11 , 0 , 13 , 4 , 7 , 17 , 0 , 25 , 22 , 31 , 15 , 29 , 10 , 12 , 6 , 0 , 21 , 14 , 9 , 5 , 20 , 8 , 19 , 18 } ;
return lookup [ ( - x & x ) % 37 ] ; }
static public void Main ( ) { Console . WriteLine ( countTrailingZero ( 48 ) ) ; } }
using System ; public class GFG { static int multiplyBySevenByEight ( int n ) {
return ( n - ( n >> 3 ) ) ; }
public static void Main ( ) { int n = 9 ; Console . WriteLine ( multiplyBySevenByEight ( n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int countNumbers ( int L , int R , int K ) {
List < int > list = new List < int > ( ) ;
for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) ) {
list . Add ( i ) ; } }
int count = 0 ;
for ( int i = 0 ; i < list . Count ; i ++ ) {
int right_index = search ( list , list [ i ] + K - 1 ) ;
if ( right_index != - 1 ) count = Math . Max ( count , right_index - i + 1 ) ; }
return count ; }
static int search ( List < int > list , int num ) { int low = 0 , high = list . Count - 1 ;
int ans = - 1 ; while ( low <= high ) {
int mid = low + ( high - low ) / 2 ;
if ( list [ mid ] <= num ) {
ans = mid ;
low = mid + 1 ; }
high = mid - 1 ; }
return ans ; }
static bool isPalindrome ( int n ) { int rev = 0 ; int temp = n ;
while ( n > 0 ) { rev = rev * 10 + n % 10 ; n /= 10 ; }
return rev == temp ; }
public static void Main ( string [ ] args ) { int L = 98 , R = 112 ; int K = 13 ; Console . WriteLine ( countNumbers ( L , R , K ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static int findMaximumSum ( int [ ] a , int n ) {
int [ ] prev_smaller = findPrevious ( a , n ) ;
int [ ] next_smaller = findNext ( a , n ) ; int max_value = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
max_value = Math . Max ( max_value , a [ i ] * ( next_smaller [ i ] - prev_smaller [ i ] - 1 ) ) ; }
return max_value ; }
public static int [ ] findPrevious ( int [ ] a , int n ) { int [ ] ps = new int [ n ] ;
ps [ 0 ] = - 1 ;
Stack < int > stack = new Stack < int > ( ) ;
stack . Push ( 0 ) ; for ( int i = 1 ; i < a . Length ; i ++ ) {
while ( stack . Count > 0 && a [ stack . Peek ( ) ] >= a [ i ] ) stack . Pop ( ) ;
ps [ i ] = stack . Count > 0 ? stack . Peek ( ) : - 1 ;
stack . Push ( i ) ; }
return ps ; }
public static int [ ] findNext ( int [ ] a , int n ) { int [ ] ns = new int [ n ] ; ns [ n - 1 ] = n ;
Stack < int > stack = new Stack < int > ( ) ; stack . Push ( n - 1 ) ;
for ( int i = n - 2 ; i >= 0 ; i -- ) {
while ( stack . Count > 0 && a [ stack . Peek ( ) ] >= a [ i ] ) stack . Pop ( ) ;
ns [ i ] = stack . Count > 0 ? stack . Peek ( ) : a . Length ;
stack . Push ( i ) ; }
return ns ; }
public static void Main ( String [ ] args ) { int n = 3 ; int [ ] a = { 80 , 48 , 82 } ; Console . WriteLine ( findMaximumSum ( a , n ) ) ; } }
static bool compare ( int [ ] arr1 , int [ ] arr2 ) { for ( int i = 0 ; i < 256 ; i ++ ) if ( arr1 [ i ] != arr2 [ i ] ) return false ; return true ; }
static bool search ( String pat , String txt ) { int M = pat . Length ; int N = txt . Length ;
int [ ] countP = new int [ 256 ] ; int [ ] countTW = new int [ 256 ] ; for ( int i = 0 ; i < 256 ; i ++ ) { countP [ i ] = 0 ; countTW [ i ] = 0 ; } for ( int i = 0 ; i < M ; i ++ ) { ( countP [ pat [ i ] ] ) ++ ; ( countTW [ txt [ i ] ] ) ++ ; }
for ( int i = M ; i < N ; i ++ ) {
if ( compare ( countP , countTW ) ) return true ;
( countTW [ txt [ i ] ] ) ++ ;
countTW [ txt [ i - M ] ] -- ; }
if ( compare ( countP , countTW ) ) return true ; return false ; }
public static void Main ( ) { string txt = " BACDGABCDA " ; string pat = " ABCD " ; if ( search ( pat , txt ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " NO " ) ; } }
using System ; using System . Linq ; class GFG {
static double getMaxMedian ( int [ ] arr , int n , int k ) { int size = n + k ;
Array . Sort ( arr ) ;
if ( size % 2 == 0 ) { double median = ( double ) ( arr [ ( size / 2 ) - 1 ] + arr [ size / 2 ] ) / 2 ; return median ; }
double median1 = arr [ size / 2 ] ; return median1 ; }
static void Main ( ) { int [ ] arr = { 3 , 2 , 3 , 4 , 2 } ; int n = arr . Length ; int k = 2 ; Console . WriteLine ( getMaxMedian ( arr , n , k ) ) ; } }
using System ; class GFG { static void printSorted ( int a , int b , int c ) {
int get_max = Math . Max ( a , Math . Max ( b , c ) ) ;
int get_min = - Math . Max ( - a , Math . Max ( - b , - c ) ) ; int get_mid = ( a + b + c ) - ( get_max + get_min ) ; Console . Write ( get_min + " ▁ " + get_mid + " ▁ " + get_max ) ; }
public static void Main ( ) { int a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ; } }
using System ; class GFG { static int binarySearch ( int [ ] a , int item , int low , int high ) { while ( low <= high ) { int mid = low + ( high - low ) / 2 ; if ( item == a [ mid ] ) return mid + 1 ; else if ( item > a [ mid ] ) low = mid + 1 ; else high = mid - 1 ; } return low ; }
static void insertionSort ( int [ ] a , int n ) { int i , loc , j , selected ; for ( i = 1 ; i < n ; ++ i ) { j = i - 1 ; selected = a [ i ] ;
loc = binarySearch ( a , selected , 0 , j ) ;
while ( j >= loc ) { a [ j + 1 ] = a [ j ] ; j -- ; } a [ j + 1 ] = selected ; } }
public static void Main ( String [ ] args ) { int [ ] a = { 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 } ; int n = a . Length , i ; insertionSort ( a , n ) ; Console . WriteLine ( " Sorted ▁ array : " ) ; for ( i = 0 ; i < n ; i ++ ) Console . Write ( a [ i ] + " ▁ " ) ; } }
using System ; class InsertionSort {
void sort ( int [ ] arr ) { int n = arr . Length ; for ( int i = 1 ; i < n ; ++ i ) { int key = arr [ i ] ; int j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
static void printArray ( int [ ] arr ) { int n = arr . Length ; for ( int i = 0 ; i < n ; ++ i ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ; }
public static void Main ( ) { int [ ] arr = { 12 , 11 , 13 , 5 , 6 } ; InsertionSort ob = new InsertionSort ( ) ; ob . sort ( arr ) ; printArray ( arr ) ; } }
using System ; using System . Collections . Generic ; public class GFG {
static int validPermutations ( String str ) { Dictionary < char , int > m = new Dictionary < char , int > ( ) ;
int count = str . Length , ans = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) { if ( m . ContainsKey ( str [ i ] ) ) m [ str [ i ] ] = m [ str [ i ] ] + 1 ; else m . Add ( str [ i ] , 1 ) ; } for ( int i = 0 ; i < str . Length ; i ++ ) {
ans += count - m [ str [ i ] ] ;
if ( m . ContainsKey ( str [ i ] ) ) m [ str [ i ] ] = m [ str [ i ] ] - 1 ; count -- ; }
return ans + 1 ; } public static void Main ( String [ ] args ) { String str = " sstt " ; Console . WriteLine ( validPermutations ( str ) ) ; } }
using System ; public class GFG {
static int countPaths ( int n , int m ) { int [ , ] dp = new int [ n + 1 , m + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) dp [ i , 0 ] = 1 ; for ( int i = 0 ; i <= m ; i ++ ) dp [ 0 , i ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) for ( int j = 1 ; j <= m ; j ++ ) dp [ i , j ] = dp [ i - 1 , j ] + dp [ i , j - 1 ] ; return dp [ n , m ] ; }
public static void Main ( ) { int n = 3 , m = 2 ; Console . WriteLine ( " ▁ Number ▁ of " + " ▁ Paths ▁ " + countPaths ( n , m ) ) ; } }
using System ; class GFG {
static int count ( int [ ] S , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int m = arr . Length ; Console . Write ( count ( arr , m , 4 ) ) ; } }
using System ; public class GFG { static bool equalIgnoreCase ( String str1 , String str2 ) {
str1 = str1 . ToUpper ( ) ; str2 = str2 . ToUpper ( ) ;
int x = str1 . CompareTo ( str2 ) ;
if ( x != 0 ) { return false ; } else { return true ; } }
static void equalIgnoreCaseUtil ( String str1 , String str2 ) { bool res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) { Console . WriteLine ( " Same " ) ; } else { Console . WriteLine ( " Not ▁ Same " ) ; } }
public static void Main ( ) { String str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; } }
using System ; public class GFG {
public static string replaceConsonants ( string str ) {
string res = " " ; int i = 0 , count = 0 ;
while ( i < str . Length ) {
if ( str [ i ] != ' a ' && str [ i ] != ' e ' && str [ i ] != ' i ' && str [ i ] != ' o ' && str [ i ] != ' u ' ) { i ++ ; count ++ ; } else {
if ( count > 0 ) { res += count ; }
res += str [ i ] ; i ++ ; count = 0 ; } }
if ( count > 0 ) { res += count ; }
return res ; }
public static void Main ( string [ ] args ) { string str = " abcdeiop " ; Console . WriteLine ( replaceConsonants ( str ) ) ; } }
using System ; public class GFG {
static bool isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
static String encryptString ( String s , int n , int k ) { int countVowels = 0 ; int countConsonants = 0 ; String ans = " " ;
for ( int l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( int r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s [ r ] ) == true ) { countVowels ++ ; } else { countConsonants ++ ; } }
ans += Convert . ToString ( countVowels * countConsonants ) ; } return ans ; }
static public void Main ( ) { String s = " hello " ; int n = s . Length ; int k = 2 ; Console . Write ( encryptString ( s , n , k ) ) ; } }
using System ; using System . Text ; class GFG { private static StringBuilder charBuffer = new StringBuilder ( ) ; public static String processWords ( String input ) {
String [ ] s = input . Split ( ' ▁ ' ) ; foreach ( String values in s ) {
charBuffer . Append ( values [ 0 ] ) ; } return charBuffer . ToString ( ) ; }
public static void Main ( ) { String input = " geeks ▁ for ▁ geeks " ; Console . WriteLine ( processWords ( input ) ) ; } }
using System ; class GFG {
static string toString ( char [ ] a ) { string String = new string ( a ) ; return String ; } static void generate ( int k , char [ ] ch , int n ) {
if ( n == k ) {
Console . Write ( toString ( ch ) + " ▁ " ) ; return ; }
if ( ch [ n - 1 ] == '0' ) { ch [ n ] = '0' ; generate ( k , ch , n + 1 ) ; ch [ n ] = '1' ; generate ( k , ch , n + 1 ) ; }
if ( ch [ n - 1 ] == '1' ) { ch [ n ] = '0' ;
generate ( k , ch , n + 1 ) ; } } static void fun ( int k ) { if ( k <= 0 ) { return ; } char [ ] ch = new char [ k ] ;
ch [ 0 ] = '0' ;
generate ( k , ch , 1 ) ;
ch [ 0 ] = '1' ; generate ( k , ch , 1 ) ; }
static void Main ( ) { int k = 3 ;
fun ( k ) ; } }
using System ; class GFG {
static float findVolume ( float a ) {
if ( a < 0 ) return - 1 ;
float r = a / 2 ;
float h = a ;
float V = ( float ) ( 3.14 * Math . Pow ( r , 2 ) * h ) ; return V ; }
public static void Main ( ) { float a = 5 ; Console . WriteLine ( findVolume ( a ) ) ; } }
using System ; class GFG {
public static float volumeTriangular ( int a , int b , int h ) { float vol = ( float ) ( 0.1666 ) * a * b * h ; return vol ; }
public static float volumeSquare ( int b , int h ) { float vol = ( float ) ( 0.33 ) * b * b * h ; return vol ; }
public static float volumePentagonal ( int a , int b , int h ) { float vol = ( float ) ( 0.83 ) * a * b * h ; return vol ; }
public static float volumeHexagonal ( int a , int b , int h ) { float vol = ( float ) a * b * h ; return vol ; }
public static void Main ( ) { int b = 4 , h = 9 , a = 4 ; Console . WriteLine ( " Volume ▁ of ▁ triangular " + " ▁ base ▁ pyramid ▁ is ▁ " + volumeTriangular ( a , b , h ) ) ; Console . WriteLine ( " Volume ▁ of ▁ square ▁ " + " base ▁ pyramid ▁ is ▁ " + volumeSquare ( b , h ) ) ; Console . WriteLine ( " Volume ▁ of ▁ pentagonal " + " ▁ base ▁ pyramid ▁ is ▁ " + volumePentagonal ( a , b , h ) ) ; Console . WriteLine ( " Volume ▁ of ▁ Hexagonal " + " ▁ base ▁ pyramid ▁ is ▁ " + volumeHexagonal ( a , b , h ) ) ; } }
using System ; class GFG {
static double Area ( int b1 , int b2 , int h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
public static void Main ( ) { int base1 = 8 , base2 = 10 , height = 6 ; double area = Area ( base1 , base2 , height ) ; Console . WriteLine ( " Area ▁ is : ▁ " + area ) ; } }
using System ; class GFG { static int numberOfDiagonals ( int n ) { return n * ( n - 3 ) / 2 ; }
public static void Main ( ) { int n = 5 ; Console . Write ( n + " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ) ; Console . WriteLine ( numberOfDiagonals ( n ) + " ▁ diagonals " ) ; } }
using System ; class GFG {
static void maximumArea ( int l , int b , int x , int y ) {
int left , right , above , below ; left = x * b ; right = ( l - x - 1 ) * b ; above = l * y ; below = ( b - y - 1 ) * l ;
Console . Write ( Math . Max ( Math . Max ( left , right ) , Math . Max ( above , below ) ) ) ; }
public static void Main ( String [ ] args ) { int L = 8 , B = 8 ; int X = 0 , Y = 0 ;
maximumArea ( L , B , X , Y ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int delCost ( string s , int [ ] cost ) {
int ans = 0 ;
Dictionary < int , int > forMax = new Dictionary < int , int > ( ) ;
Dictionary < int , int > forTot = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < s . Length ; i ++ ) {
if ( ! forMax . ContainsKey ( s [ i ] ) ) { forMax [ s [ i ] ] = cost [ i ] ; } else {
forMax [ s [ i ] ] = Math . Max ( cost [ i ] , forMax [ s [ i ] ] ) ; }
if ( ! forTot . ContainsKey ( s [ i ] ) ) { forTot [ s [ i ] ] = cost [ i ] ; } else {
forTot [ s [ i ] ] += cost [ i ] ; } }
foreach ( KeyValuePair < int , int > i in forMax ) {
ans += forTot [ i . Key ] - i . Value ; }
return ans ; }
static void Main ( ) {
string s = " AAABBB " ;
int [ ] cost = { 1 , 2 , 3 , 4 , 5 , 6 } ;
Console . WriteLine ( delCost ( s , cost ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static readonly int MAX = 10000 ; static List < int > [ ] divisors = new List < int > [ MAX + 1 ] ;
static void computeDivisors ( ) { for ( int i = 1 ; i <= MAX ; i ++ ) { for ( int j = i ; j <= MAX ; j += i ) {
divisors [ j ] . Add ( i ) ; } } }
static int getClosest ( int val1 , int val2 , int target ) { if ( target - val1 >= val2 - target ) return val2 ; else return val1 ; }
static int findClosest ( List < int > array , int n , int target ) { int [ ] arr = array . ToArray ( ) ;
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
int ans = findClosest ( divisors [ N ] , divisors [ N ] . Count , X ) ;
Console . Write ( ans ) ; }
public static void Main ( String [ ] args ) {
int N = 16 , X = 5 ; for ( int i = 0 ; i < divisors . Length ; i ++ ) divisors [ i ] = new List < int > ( ) ;
printClosest ( N , X ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int maxMatch ( int [ ] A , int [ ] B ) {
Dictionary < int , int > Aindex = new Dictionary < int , int > ( ) ;
Dictionary < int , int > diff = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < A . Length ; i ++ ) { Aindex [ A [ i ] ] = i ; }
for ( int i = 0 ; i < B . Length ; i ++ ) {
if ( i - Aindex [ B [ i ] ] < 0 ) { if ( ! diff . ContainsKey ( A . Length + i - Aindex [ B [ i ] ] ) ) { diff [ A . Length + i - Aindex [ B [ i ] ] ] = 1 ; } else { diff [ A . Length + i - Aindex [ B [ i ] ] ] += 1 ; } }
else { if ( ! diff . ContainsKey ( i - Aindex [ B [ i ] ] ) ) { diff [ i - Aindex [ B [ i ] ] ] = 1 ; } else { diff [ i - Aindex [ B [ i ] ] ] += 1 ; } } }
int max = 0 ; foreach ( KeyValuePair < int , int > ele in diff ) { if ( ele . Value > max ) { max = ele . Value ; } } return max ; }
static void Main ( ) { int [ ] A = { 5 , 3 , 7 , 9 , 8 } ; int [ ] B = { 8 , 7 , 3 , 5 , 9 } ;
Console . WriteLine ( maxMatch ( A , B ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static int N = 9 ;
static bool isinRange ( int [ , ] board ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( board [ i , j ] <= 0 board [ i , j ] > 9 ) { return false ; } } } return true ; }
static bool isValidSudoku ( int [ , ] board ) {
if ( isinRange ( board ) == false ) { return false ; }
bool [ ] unique = new bool [ N + 1 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
Array . Fill ( unique , false ) ;
for ( int j = 0 ; j < N ; j ++ ) {
int Z = board [ i , j ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( int i = 0 ; i < N ; i ++ ) {
Array . Fill ( unique , false ) ;
for ( int j = 0 ; j < N ; j ++ ) {
int Z = board [ j , i ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( int i = 0 ; i < N - 2 ; i += 3 ) {
for ( int j = 0 ; j < N - 2 ; j += 3 ) {
Array . Fill ( unique , false ) ;
for ( int k = 0 ; k < 3 ; k ++ ) { for ( int l = 0 ; l < 3 ; l ++ ) {
int X = i + k ;
int Y = j + l ;
int Z = board [ X , Y ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } } } }
return true ; }
public static void Main ( ) { int [ , ] board = { { 7 , 9 , 2 , 1 , 5 , 4 , 3 , 8 , 6 } , { 6 , 4 , 3 , 8 , 2 , 7 , 1 , 5 , 9 } , { 8 , 5 , 1 , 3 , 9 , 6 , 7 , 2 , 4 } , { 2 , 6 , 5 , 9 , 7 , 3 , 8 , 4 , 1 } , { 4 , 8 , 9 , 5 , 6 , 1 , 2 , 7 , 3 } , { 3 , 1 , 7 , 4 , 8 , 2 , 9 , 6 , 5 } , { 1 , 3 , 6 , 7 , 4 , 8 , 5 , 9 , 2 } , { 9 , 7 , 4 , 2 , 1 , 5 , 6 , 3 , 8 } , { 5 , 2 , 8 , 6 , 3 , 9 , 4 , 1 , 7 } } ; if ( isValidSudoku ( board ) ) { Console . WriteLine ( " Valid " ) ; } else { Console . WriteLine ( " Not ▁ Valid " ) ; } } }
using System ; class GFG {
public static bool palindrome ( int [ ] a , int i , int j ) { while ( i < j ) {
if ( a [ i ] != a [ j ] ) return false ;
i ++ ; j -- ; }
return true ; }
static int findSubArray ( int [ ] arr , int k ) { int n = arr . Length ;
for ( int i = 0 ; i <= n - k ; i ++ ) { if ( palindrome ( arr , i , i + k - 1 ) ) return i ; }
return - 1 ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 3 , 5 , 1 , 3 } ; int k = 4 ; int ans = findSubArray ( arr , k ) ; if ( ans == - 1 ) Console . Write ( - 1 + " STRNEWLINE " ) ; else { for ( int i = ans ; i < ans + k ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ; } } }
using System ; using System . Collections . Generic ; class GFG {
static void isCrossed ( String path ) { if ( path . Length == 0 ) return ;
bool ans = false ;
HashSet < KeyValuePair < int , int > > mySet = new HashSet < KeyValuePair < int , int > > ( ) ;
int x = 0 , y = 0 ; mySet . Add ( new KeyValuePair < int , int > ( x , y ) ) ;
for ( int i = 0 ; i < path . Length ; i ++ ) {
if ( path [ i ] == ' N ' ) mySet . Add ( new KeyValuePair < int , int > ( x , y ++ ) ) ; if ( path [ i ] == ' S ' ) mySet . Add ( new KeyValuePair < int , int > ( x , y -- ) ) ; if ( path [ i ] == ' E ' ) mySet . Add ( new KeyValuePair < int , int > ( x ++ , y ) ) ; if ( path [ i ] == ' W ' ) mySet . Add ( new KeyValuePair < int , int > ( x -- , y ) ) ;
if ( mySet . Contains ( new KeyValuePair < int , int > ( x , y ) ) ) { ans = true ; break ; } }
if ( ans ) Console . Write ( " Crossed " ) ; else Console . Write ( " Not ▁ Crossed " ) ; }
public static void Main ( String [ ] args ) {
String path = " NESW " ;
isCrossed ( path ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int maxWidth ( int N , int M , List < int > cost , List < List < int > > s ) {
List < List < int > > adj = new List < List < int > > ( ) ; for ( int i = 0 ; i < N ; i ++ ) { adj . Add ( new List < int > ( ) ) ; } for ( int i = 0 ; i < M ; i ++ ) { adj [ s [ i ] [ 0 ] ] . Add ( s [ i ] [ 1 ] ) ; }
int result = 0 ;
Queue < int > q = new Queue < int > ( ) ;
q . Enqueue ( 0 ) ;
while ( q . Count != 0 ) {
int count = q . Count ;
result = Math . Max ( count , result ) ;
while ( count -- > 0 ) {
int temp = q . Dequeue ( ) ;
for ( int i = 0 ; i < adj [ temp ] . Count ; i ++ ) { q . Enqueue ( adj [ temp ] [ i ] ) ; } } }
return result ; }
static public void Main ( ) { int N = 11 , M = 10 ; List < List < int > > edges = new List < List < int > > ( ) ; edges . Add ( new List < int > ( ) { 0 , 1 } ) ; edges . Add ( new List < int > ( ) { 0 , 2 } ) ; edges . Add ( new List < int > ( ) { 0 , 3 } ) ; edges . Add ( new List < int > ( ) { 1 , 4 } ) ; edges . Add ( new List < int > ( ) { 1 , 5 } ) ; edges . Add ( new List < int > ( ) { 3 , 6 } ) ; edges . Add ( new List < int > ( ) { 4 , 7 } ) ; edges . Add ( new List < int > ( ) { 6 , 10 } ) ; edges . Add ( new List < int > ( ) { 6 , 8 } ) ; edges . Add ( new List < int > ( ) { 6 , 9 } ) ; List < int > cost = new List < int > ( ) { 1 , 2 , - 1 , 3 , 4 , 5 , 8 , 2 , 6 , 12 , 7 } ;
Console . WriteLine ( maxWidth ( N , M , cost , edges ) ) ; } }
using System ; using System . Collections ; using System . Collections . Generic ; class GFG { static int MAX = 10000000 ;
static bool [ ] isPrime = new bool [ MAX + 1 ] ;
static ArrayList primes = new ArrayList ( ) ;
static void SieveOfEratosthenes ( ) { Array . Fill ( isPrime , true ) ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( int i = p * p ; i <= MAX ; i += p ) isPrime [ i ] = false ; } }
for ( int p = 2 ; p <= MAX ; p ++ ) if ( isPrime [ p ] ) primes . Add ( p ) ; }
static int prime_search ( ArrayList primes , int diff ) {
int low = 0 ; int high = primes . Count - 1 ; int res = - 1 ; while ( low <= high ) { int mid = ( low + high ) / 2 ;
if ( ( int ) primes [ mid ] == diff ) {
return ( int ) primes [ mid ] ; }
else if ( ( int ) primes [ mid ] < diff ) {
low = mid + 1 ; }
else { res = ( int ) primes [ mid ] ;
high = mid - 1 ; } }
return res ; }
static int minCost ( int [ ] arr , int n ) {
SieveOfEratosthenes ( ) ;
int res = 0 ;
for ( int i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] < arr [ i - 1 ] ) { int diff = arr [ i - 1 ] - arr [ i ] ;
int closest_prime = prime_search ( primes , diff ) ;
res += closest_prime ;
arr [ i ] += closest_prime ; } }
return res ; }
public static void Main ( string [ ] args ) {
int [ ] arr = { 2 , 1 , 5 , 4 , 3 } ; int n = 5 ;
Console . Write ( minCost ( arr , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int count ( String s ) {
int cnt = 0 ;
foreach ( char c in s . ToCharArray ( ) ) { cnt += c == '0' ? 1 : 0 ; }
if ( cnt % 3 != 0 ) return 0 ; int res = 0 , k = cnt / 3 , sum = 0 ;
Dictionary < int , int > map = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < s . Length ; i ++ ) {
sum += s [ i ] == '0' ? 1 : 0 ;
if ( sum == 2 * k && map . ContainsKey ( k ) && i < s . Length - 1 && i > 0 ) { res += map [ k ] ; }
if ( map . ContainsKey ( sum ) ) map [ sum ] = map [ sum ] + 1 ; else map . Add ( sum , 1 ) ; }
return res ; }
public static void Main ( String [ ] args ) {
String str = "01010" ;
Console . WriteLine ( count ( str ) ) ; } }
using System . Collections . Generic ; using System ; class GFG {
static int splitstring ( string s ) { int n = s . Length ;
int zeros = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( s [ i ] == '0' ) zeros ++ ;
if ( zeros % 3 != 0 ) return 0 ;
if ( zeros == 0 ) return ( ( n - 1 ) * ( n - 2 ) ) / 2 ;
int zerosInEachSubstring = zeros / 3 ;
int waysOfFirstCut = 0 ; int waysOfSecondCut = 0 ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == '0' ) count ++ ;
if ( count == zerosInEachSubstring ) waysOfFirstCut ++ ;
else if ( count = = 2 * zerosInEachSubstring ) waysOfSecondCut ++ ; }
return waysOfFirstCut * waysOfSecondCut ; }
public static void Main ( ) { string s = "01010" ;
Console . WriteLine ( " The ▁ number ▁ of ▁ ways ▁ " + " to ▁ split ▁ is ▁ " + splitstring ( s ) ) ; } }
using System ; class GFG {
static bool canTransform ( string str1 , string str2 ) { string s1 = " " ; string s2 = " " ;
foreach ( char c in str1 . ToCharArray ( ) ) { if ( c != ' C ' ) { s1 += c ; } } foreach ( char c in str2 . ToCharArray ( ) ) { if ( c != ' C ' ) { s2 += c ; } }
if ( s1 != s2 ) return false ; int i = 0 ; int j = 0 ; int n = str1 . Length ;
while ( i < n && j < n ) { if ( str1 [ i ] == ' C ' ) { i ++ ; } else if ( str2 [ j ] == ' C ' ) { j ++ ; }
else { if ( ( str1 [ i ] == ' A ' && i < j ) || ( str1 [ i ] == ' B ' && i > j ) ) { return false ; } i ++ ; j ++ ; } } return true ; }
public static void Main ( string [ ] args ) { string str1 = " BCCABCBCA " ; string str2 = " CBACCBBAC " ;
if ( canTransform ( str1 , str2 ) ) { Console . Write ( " Yes " ) ; } else { Console . Write ( " No " ) ; } } }
using System ; using System . Collections . Generic ; class GFG {
static int maxsubStringLength ( char [ ] S , int N ) { int [ ] arr = new int [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) if ( S [ i ] == ' a ' S [ i ] == ' e ' S [ i ] == ' i ' S [ i ] == ' o ' S [ i ] == ' u ' ) arr [ i ] = 1 ; else arr [ i ] = - 1 ;
int maxLen = 0 ;
int curr_sum = 0 ;
Dictionary < int , int > hash = new Dictionary < int , int > ( ) ;
for ( int i = 0 ; i < N ; i ++ ) { curr_sum += arr [ i ] ;
if ( curr_sum == 0 )
maxLen = Math . Max ( maxLen , i + 1 ) ;
if ( hash . ContainsKey ( curr_sum ) ) maxLen = Math . Max ( maxLen , i - hash [ curr_sum ] ) ;
else hash . ( curr_sum , i ) ; }
return maxLen ; }
public static void Main ( String [ ] args ) { String S = " geeksforgeeks " ; int n = S . Length ; Console . Write ( maxsubStringLength ( S . ToCharArray ( ) , n ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { class pair { public int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } } static int [ , ] mat = new int [ 1001 , 1001 ] ; static int r , c , x , y ;
static int [ ] dx = { 0 , - 1 , - 1 , - 1 , 0 , 1 , 1 , 1 } ; static int [ ] dy = { 1 , 1 , 0 , - 1 , - 1 , - 1 , 0 , 1 } ;
static void FindMinimumDistance ( ) {
Queue < pair > q = new Queue < pair > ( ) ;
q . Enqueue ( new pair ( x , y ) ) ; mat [ x , y ] = 0 ;
while ( q . Count != 0 ) {
x = q . Peek ( ) . first ; y = q . Peek ( ) . second ;
q . Dequeue ( ) ; for ( int i = 0 ; i < 8 ; i ++ ) { int a = x + dx [ i ] ; int b = y + dy [ i ] ;
if ( a < 0 a >= r b >= c b < 0 ) continue ;
if ( mat [ a , b ] == 0 ) {
mat [ a , b ] = mat [ x , y ] + 1 ;
q . Enqueue ( new pair ( a , b ) ) ; } } } }
public static void Main ( String [ ] args ) { r = 5 ; c = 5 ; x = 1 ; y = 1 ; int t = x ; int l = y ; mat [ x , y ] = 0 ; FindMinimumDistance ( ) ; mat [ t , l ] = 0 ;
for ( int i = 0 ; i < r ; i ++ ) { for ( int j = 0 ; j < c ; j ++ ) { Console . Write ( mat [ i , j ] + " ▁ " ) ; } Console . WriteLine ( ) ; } } }
using System ; class GFG {
public static int minOperations ( String S , int K ) {
int ans = 0 ;
for ( int i = 0 ; i < K ; i ++ ) {
int zero = 0 , one = 0 ;
for ( int j = i ; j < S . Length ; j += K ) {
if ( S [ j ] == '0' ) zero ++ ;
else one ++ ; }
ans += Math . Min ( zero , one ) ; }
return ans ; }
public static void Main ( String [ ] args ) { String S = "110100101" ; int K = 3 ; Console . WriteLine ( minOperations ( S , K ) ) ; } }
using System ; class GFG {
static int missingElement ( int [ ] arr , int n ) {
int max_ele = arr [ 0 ] ;
int min_ele = arr [ 0 ] ;
int x = 0 ;
int d ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max_ele ) max_ele = arr [ i ] ; if ( arr [ i ] < min_ele ) min_ele = arr [ i ] ; }
d = ( max_ele - min_ele ) / n ;
for ( int i = 0 ; i < n ; i ++ ) { x = x ^ arr [ i ] ; }
for ( int i = 0 ; i <= n ; i ++ ) { x = x ^ ( min_ele + ( i * d ) ) ; }
return x ; }
public static void Main ( ) {
int [ ] arr = new int [ ] { 12 , 3 , 6 , 15 , 18 } ; int n = arr . Length ;
int element = missingElement ( arr , n ) ;
Console . Write ( element ) ; } }
using System ; class GFG {
static void Printksubstring ( String str , int n , int k ) {
int total = ( n * ( n + 1 ) ) / 2 ;
if ( k > total ) { Console . Write ( " - 1 STRNEWLINE " ) ; return ; }
int [ ] substring = new int [ n + 1 ] ; substring [ 0 ] = 0 ;
int temp = n ; for ( int i = 1 ; i <= n ; i ++ ) {
substring [ i ] = substring [ i - 1 ] + temp ; temp -- ; }
int l = 1 ; int h = n ; int start = 0 ; while ( l <= h ) { int m = ( l + h ) / 2 ; if ( substring [ m ] > k ) { start = m ; h = m - 1 ; } else if ( substring [ m ] < k ) { l = m + 1 ; } else { start = m ; break ; } }
int end = n - ( substring [ start ] - k ) ;
for ( int i = start - 1 ; i < end ; i ++ ) { Console . Write ( str [ i ] ) ; } }
public static void Main ( String [ ] args ) { String str = " abc " ; int k = 4 ; int n = str . Length ; Printksubstring ( str , n , k ) ; } }
using System ; public class GFG {
static int LowerInsertionPoint ( int [ ] arr , int n , int X ) {
if ( X < arr [ 0 ] ) return 0 ; else if ( X > arr [ n - 1 ] ) return n ; int lowerPnt = 0 ; int i = 1 ; while ( i < n && arr [ i ] < X ) { lowerPnt = i ; i = i * 2 ; }
while ( lowerPnt < n && arr [ lowerPnt ] < X ) lowerPnt ++ ; return lowerPnt ; }
static public void Main ( ) { int [ ] arr = { 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 } ; int n = arr . Length ; int X = 4 ; Console . WriteLine ( LowerInsertionPoint ( arr , n , X ) ) ; } }
using System ; class GFG {
static int getCount ( int M , int N ) { int count = 0 ;
if ( M == 1 ) return N ;
if ( N == 1 ) return M ; if ( N > M ) {
for ( int i = 1 ; i <= M ; i ++ ) { int numerator = N * i - N + M - i ; int denominator = M - 1 ;
if ( numerator % denominator == 0 ) { int j = numerator / denominator ;
if ( j >= 1 && j <= N ) count ++ ; } } } else {
for ( int j = 1 ; j <= N ; j ++ ) { int numerator = M * j - M + N - j ; int denominator = N - 1 ;
if ( numerator % denominator == 0 ) { int i = numerator / denominator ;
if ( i >= 1 && i <= M ) count ++ ; } } } return count ; }
public static void Main ( ) { int M = 3 , N = 5 ; Console . WriteLine ( getCount ( M , N ) ) ; } }
using System ; class GFG {
static bool swapElement ( int [ ] arr1 , int [ ] arr2 , int n ) {
int wrongIdx = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr1 [ i ] < arr1 [ i - 1 ] ) { wrongIdx = i ; } } int maximum = int . MinValue ; int maxIdx = - 1 ; bool res = false ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr2 [ i ] > maximum && arr2 [ i ] >= arr1 [ wrongIdx - 1 ] ) { if ( wrongIdx + 1 <= n - 1 && arr2 [ i ] <= arr1 [ wrongIdx + 1 ] ) { maximum = arr2 [ i ] ; maxIdx = i ; res = true ; } } }
if ( res ) { swap ( arr1 , wrongIdx , arr2 , maxIdx ) ; } return res ; } static void swap ( int [ ] a , int wrongIdx , int [ ] b , int maxIdx ) { int c = a [ wrongIdx ] ; a [ wrongIdx ] = b [ maxIdx ] ; b [ maxIdx ] = c ; }
static void getSortedArray ( int [ ] arr1 , int [ ] arr2 , int n ) { if ( swapElement ( arr1 , arr2 , n ) ) { for ( int i = 0 ; i < n ; i ++ ) { Console . Write ( arr1 [ i ] + " ▁ " ) ; } } else { Console . Write ( " Not ▁ Possible " ) ; } }
public static void Main ( ) { int [ ] arr1 = { 1 , 3 , 7 , 4 , 10 } ; int [ ] arr2 = { 2 , 1 , 6 , 8 , 9 } ; int n = arr1 . Length ; getSortedArray ( arr1 , arr2 , n ) ; } }
using System ; class Middle {
public static int middleOfThree ( int a , int b , int c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
public static void Main ( ) { int a = 20 , b = 30 , c = 40 ; Console . WriteLine ( middleOfThree ( a , b , c ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void largestArea ( int N , int M , int [ ] H , int [ ] V ) {
HashSet < int > s1 = new HashSet < int > ( ) ; HashSet < int > s2 = new HashSet < int > ( ) ;
for ( int i = 1 ; i <= N + 1 ; i ++ ) s1 . Add ( i ) ;
for ( int i = 1 ; i <= M + 1 ; i ++ ) s2 . Add ( i ) ;
for ( int i = 0 ; i < H . Length ; i ++ ) { s1 . Remove ( H [ i ] ) ; }
for ( int i = 0 ; i < V . Length ; i ++ ) { s2 . Remove ( V [ i ] ) ; }
int [ ] list1 = new int [ s1 . Count ] ; int [ ] list2 = new int [ s2 . Count ] ; int I = 0 ; foreach ( int it1 in s1 ) { list1 [ I ++ ] = it1 ; } I = 0 ; foreach ( int it2 in s2 ) { list2 [ I ++ ] = it2 ; }
Array . Sort ( list1 ) ; Array . Sort ( list2 ) ; int maxH = 0 , p1 = 0 , maxV = 0 , p2 = 0 ;
for ( int j = 0 ; j < list1 . Length ; j ++ ) { maxH = Math . Max ( maxH , list1 [ j ] - p1 ) ; p1 = list1 [ j ] ; }
for ( int j = 0 ; j < list2 . Length ; j ++ ) { maxV = Math . Max ( maxV , list2 [ j ] - p2 ) ; p2 = list2 [ j ] ; }
Console . WriteLine ( maxV * maxH ) ; }
static void Main ( ) {
int N = 3 , M = 3 ;
int [ ] H = { 2 } ; int [ ] V = { 2 } ;
largestArea ( N , M , H , V ) ; } }
using System ; public class GFG {
static bool checkifSorted ( int [ ] A , int [ ] B , int N ) {
bool flag = false ;
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
public static void Main ( string [ ] args ) {
int [ ] A = { 3 , 1 , 2 } ;
int [ ] B = { 0 , 1 , 1 } ; int N = A . Length ;
bool check = checkifSorted ( A , B , N ) ;
if ( check ) { Console . WriteLine ( " YES " ) ; }
else { Console . WriteLine ( " NO " ) ; } } }
using System ; using System . Text ; public class GFG {
static int minSteps ( StringBuilder A , StringBuilder B , int M , int N ) { if ( A [ 0 ] > B [ 0 ] ) return 0 ; if ( B [ 0 ] > A [ 0 ] ) { return 1 ; }
if ( M <= N && A [ 0 ] == B [ 0 ] && count ( A , A [ 0 ] ) == M && count ( B , B [ 0 ] ) == N ) return - 1 ;
for ( int i = 1 ; i < N ; i ++ ) { if ( B [ i ] > B [ 0 ] ) return 1 ; }
for ( int i = 1 ; i < M ; i ++ ) { if ( A [ i ] < A [ 0 ] ) return 1 ; }
for ( int i = 1 ; i < M ; i ++ ) { if ( A [ i ] > A [ 0 ] ) { swap ( A , i , B , 0 ) ; swap ( A , 0 , B , 0 ) ; return 2 ; } }
for ( int i = 1 ; i < N ; i ++ ) { if ( B [ i ] < B [ 0 ] ) { swap ( A , 0 , B , i ) ; swap ( A , 0 , B , 0 ) ; return 2 ; } }
return 0 ; } static int count ( StringBuilder a , char c ) { int count = 0 ; for ( int i = 0 ; i < a . Length ; i ++ ) if ( a [ i ] == c ) count ++ ; return count ; } static void swap ( StringBuilder s1 , int index1 , StringBuilder s2 , int index2 ) { char c = s1 [ index1 ] ; s1 [ index1 ] = s2 [ index2 ] ; s2 [ index2 ] = c ; }
static public void Main ( ) { StringBuilder A = new StringBuilder ( " adsfd " ) ; StringBuilder B = new StringBuilder ( " dffff " ) ; int M = A . Length ; int N = B . Length ; Console . WriteLine ( minSteps ( A , B , M , N ) ) ; } }
using System ; class GFG { const int maxN = 201 ;
static int n1 , n2 , n3 ;
static int [ , , ] dp = new int [ maxN , maxN , maxN ] ;
static int getMaxSum ( int i , int j , int k , int [ ] arr1 , int [ ] arr2 , int [ ] arr3 ) {
int cnt = 0 ; if ( i >= n1 ) cnt ++ ; if ( j >= n2 ) cnt ++ ; if ( k >= n3 ) cnt ++ ;
if ( cnt >= 2 ) return 0 ;
if ( dp [ i , j , k ] != - 1 ) return dp [ i , j , k ] ; int ans = 0 ;
if ( i < n1 && j < n2 )
ans = Math . Max ( ans , getMaxSum ( i + 1 , j + 1 , k , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr2 [ j ] ) ; if ( i < n1 && k < n3 ) ans = Math . Max ( ans , getMaxSum ( i + 1 , j , k + 1 , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr3 [ k ] ) ; if ( j < n2 && k < n3 ) ans = Math . Max ( ans , getMaxSum ( i , j + 1 , k + 1 , arr1 , arr2 , arr3 ) + arr2 [ j ] * arr3 [ k ] ) ;
dp [ i , j , k ] = ans ;
return dp [ i , j , k ] ; } static void reverse ( int [ ] tmp ) { int i , t ; int n = tmp . Length ; for ( i = 0 ; i < n / 2 ; i ++ ) { t = tmp [ i ] ; tmp [ i ] = tmp [ n - i - 1 ] ; tmp [ n - i - 1 ] = t ; } }
static int maxProductSum ( int [ ] arr1 , int [ ] arr2 , int [ ] arr3 ) {
for ( int i = 0 ; i < maxN ; i ++ ) for ( int j = 0 ; j < maxN ; j ++ ) for ( int k = 0 ; k < maxN ; k ++ ) dp [ i , j , k ] = - 1 ;
Array . Sort ( arr1 ) ; reverse ( arr1 ) ; Array . Sort ( arr2 ) ; reverse ( arr2 ) ; Array . Sort ( arr3 ) ; reverse ( arr3 ) ; return getMaxSum ( 0 , 0 , 0 , arr1 , arr2 , arr3 ) ; }
public static void Main ( string [ ] args ) { n1 = 2 ; int [ ] arr1 = { 3 , 5 } ; n2 = 2 ; int [ ] arr2 = { 2 , 1 } ; n3 = 3 ; int [ ] arr3 = { 4 , 3 , 5 } ; Console . Write ( maxProductSum ( arr1 , arr2 , arr3 ) ) ; } }
using System ; class GFG {
static void findTriplet ( int [ ] arr , int N ) {
Array . Sort ( arr ) ; int flag = 0 , i ;
for ( i = N - 1 ; i - 2 >= 0 ; i -- ) {
if ( arr [ i - 2 ] + arr [ i - 1 ] > arr [ i ] ) { flag = 1 ; break ; } }
if ( flag != 0 ) {
Console . Write ( arr [ i - 2 ] + " ▁ " + arr [ i - 1 ] + " ▁ " + arr [ i ] ) ; }
else { Console . Write ( - 1 ) ; } }
public static void Main ( string [ ] args ) { int [ ] arr = { 4 , 2 , 10 , 3 , 5 } ; int N = arr . Length ; findTriplet ( arr , N ) ; } }
using System ; class GFG {
static int numberofpairs ( int [ ] arr , int N ) {
int answer = 0 ;
Array . Sort ( arr ) ;
int minDiff = 10000000 ; for ( int i = 0 ; i < N - 1 ; i ++ )
minDiff = Math . Min ( minDiff , arr [ i + 1 ] - arr [ i ] ) ; for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( arr [ i + 1 ] - arr [ i ] == minDiff )
answer ++ ; }
return answer ; }
public static void Main ( String [ ] args ) {
int [ ] arr = { 4 , 2 , 1 , 3 } ; int N = arr . Length ;
Console . Write ( numberofpairs ( arr , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int max_length = 0 ;
static List < int > store = new List < int > ( ) ;
static List < int > ans = new List < int > ( ) ;
static void find_max_length ( int [ ] arr , int index , int sum , int k ) { sum = sum + arr [ index ] ; store . Add ( arr [ index ] ) ; if ( sum == k ) { if ( max_length < store . Count ) {
max_length = store . Count ;
ans = store ; } } for ( int i = index + 1 ; i < arr . Length ; i ++ ) { if ( sum + arr [ i ] <= k ) {
find_max_length ( arr , i , sum , k ) ;
store . RemoveAt ( store . Count - 1 ) ; }
else return ; } return ; } static int longestSubsequence ( int [ ] arr , int n , int k ) {
Array . Sort ( arr ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( max_length >= n - i ) break ; store . Clear ( ) ; find_max_length ( arr , i , 0 , k ) ; } return max_length ; }
public static void Main ( String [ ] args ) { int [ ] arr = { - 3 , 0 , 1 , 1 , 2 } ; int n = arr . Length ; int k = 1 ; Console . Write ( longestSubsequence ( arr , n , k ) ) ; } }
using System ; class GFG {
static void sortArray ( int [ ] A , int N ) {
int x = 0 , y = 0 , z = 0 ;
if ( N % 4 == 0 N % 4 == 1 ) {
for ( int i = 0 ; i < N / 2 ; i ++ ) { x = i ; if ( i % 2 == 0 ) { y = N - i - 2 ; z = N - i - 1 ; }
A [ z ] = A [ y ] ; A [ y ] = A [ x ] ; A [ x ] = x + 1 ; }
Console . Write ( " Sorted ▁ Array : ▁ " ) ; for ( int i = 0 ; i < N ; i ++ ) Console . Write ( A [ i ] + " ▁ " ) ; }
else { Console . Write ( " - 1" ) ; } }
public static void Main ( String [ ] args ) { int [ ] A = { 5 , 4 , 3 , 2 , 1 } ; int N = A . Length ; sortArray ( A , N ) ; } }
using System ; class GFG {
static int findK ( int [ ] arr , int size , int N ) {
Array . Sort ( arr ) ; int temp_sum = 0 ;
for ( int i = 0 ; i < size ; i ++ ) { temp_sum += arr [ i ] ;
if ( N - temp_sum == arr [ i ] * ( size - i - 1 ) ) { return arr [ i ] ; } } return - 1 ; }
public static void Main ( ) { int [ ] arr = { 3 , 1 , 10 , 4 , 8 } ; int size = arr . Length ; int N = 16 ; Console . Write ( findK ( arr , size , N ) ) ; } }
using System ; class GFG {
static bool existsTriplet ( int [ ] a , int [ ] b , int [ ] c , int x , int l1 , int l2 , int l3 ) {
if ( l2 <= l1 && l2 <= l3 ) { swap ( l2 , l1 ) ; swap ( a , b ) ; } else if ( l3 <= l1 && l3 <= l2 ) { swap ( l3 , l1 ) ; swap ( a , c ) ; }
for ( int i = 0 ; i < l1 ; i ++ ) {
int j = 0 , k = l3 - 1 ; while ( j < l2 && k >= 0 ) {
if ( a [ i ] + b [ j ] + c [ k ] == x ) return true ; if ( a [ i ] + b [ j ] + c [ k ] < x ) j ++ ; else k -- ; } } return false ; }
public static void Main ( String [ ] args ) { int [ ] a = { 2 , 7 , 8 , 10 , 15 } ; int [ ] b = { 1 , 6 , 7 , 8 } ; int [ ] c = { 4 , 5 , 5 } ; int l1 = a . Length ; int l2 = b . Length ; int l3 = c . Length ; int x = 14 ; if ( existsTriplet ( a , b , c , x , l1 , l2 , l3 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
public static void printArr ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] ) ; }
public static int compare ( int num1 , int num2 ) {
String A = num1 . ToString ( ) ;
String B = num2 . ToString ( ) ;
return ( A + B ) . CompareTo ( B + A ) ; }
public static void printSmallest ( int N , int [ ] arr ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) { if ( compare ( arr [ i ] , arr [ j ] ) > 0 ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } } }
printArr ( arr , N ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 2 , 9 , 21 , 1 } ; int N = arr . Length ; printSmallest ( N , arr ) ; } }
using System ; class GFG { static void stableSelectionSort ( int [ ] a , int n ) {
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min = i ; for ( int j = i + 1 ; j < n ; j ++ ) if ( a [ min ] > a [ j ] ) min = j ;
int key = a [ min ] ; while ( min > i ) { a [ min ] = a [ min - 1 ] ; min -- ; } a [ i ] = key ; } } static void printArray ( int [ ] a , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( a [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( ) { int [ ] a = { 4 , 5 , 3 , 2 , 4 , 1 } ; int n = a . Length ; stableSelectionSort ( a , n ) ; printArray ( a , n ) ; } }
using System ; class GFG {
static bool isPossible ( int [ ] a , int [ ] b , int n , int k ) {
Array . Sort ( a ) ;
Array . Reverse ( b ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
public static void Main ( ) { int [ ] a = { 2 , 1 , 3 } ; int [ ] b = { 7 , 8 , 9 } ; int k = 10 ; int n = a . Length ; if ( isPossible ( a , b , n , k ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static int setBitCount ( int num ) { int count = 0 ; while ( num != 0 ) { if ( ( num & 1 ) != 0 ) count ++ ; num >>= 1 ; } return count ; }
static void sortBySetBitCount ( int [ ] arr , int n ) { List < Tuple < int , int > > count = new List < Tuple < int , int > > ( ) ;
for ( int i = 0 ; i < n ; ++ i ) { count . Add ( new Tuple < int , int > ( ( - 1 ) * setBitCount ( arr [ i ] ) , arr [ i ] ) ) ; } count . Sort ( ) ; foreach ( Tuple < int , int > i in count ) { Console . Write ( i . Item2 + " ▁ " ) ; } Console . WriteLine ( ) ; }
static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = arr . Length ; sortBySetBitCount ( arr , n ) ; } }
using System ; public class GFG {
static int canReach ( String s , int L , int R ) {
int [ ] dp = new int [ s . Length ] ;
dp [ 0 ] = 1 ;
int pre = 0 ;
for ( int i = 1 ; i < s . Length ; i ++ ) {
if ( i >= L ) { pre += dp [ i - L ] ; }
if ( i > R ) { pre -= dp [ i - R - 1 ] ; } if ( pre > 0 && s [ i ] == '0' ) dp [ i ] = 1 ; else dp [ i ] = 0 ; }
return dp [ s . Length - 1 ] ; }
public static void Main ( ) { String S = "01101110" ; int L = 2 , R = 3 ; if ( canReach ( S , L , R ) == 1 ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static int maxXORUtil ( int [ ] arr , int N , int xrr , int orr ) {
if ( N == 0 ) return xrr ^ orr ;
int x = maxXORUtil ( arr , N - 1 , xrr ^ orr , arr [ N - 1 ] ) ;
int y = maxXORUtil ( arr , N - 1 , xrr , orr arr [ N - 1 ] ) ;
return Math . Max ( x , y ) ; }
static int maximumXOR ( int [ ] arr , int N ) {
return maxXORUtil ( arr , N , 0 , 0 ) ; }
static void Main ( ) { int [ ] arr = { 1 , 5 , 7 } ; int N = arr . Length ; Console . Write ( maximumXOR ( arr , N ) ) ; } }
using System ; using System . Linq ; class GFG { static int N = 100000 + 5 ;
static int [ ] visited = new int [ N ] ;
static void construct_tree ( int [ ] weights , int n ) { int minimum = weights . Min ( ) ; int maximum = weights . Max ( ) ;
if ( minimum == maximum ) {
Console . WriteLine ( " No " ) ; return ; }
else {
Console . WriteLine ( " Yes " ) ; }
int root = weights [ 0 ] ;
visited [ 1 ] = 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] != root && visited [ i + 1 ] == 0 ) { Console . WriteLine ( 1 + " ▁ " + ( i + 1 ) + " ▁ " ) ;
visited [ i + 1 ] = 1 ; } }
int notroot = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( weights [ i ] != root ) { notroot = i + 1 ; break ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] == root && visited [ i + 1 ] == 0 ) { Console . WriteLine ( notroot + " ▁ " + ( i + 1 ) ) ; visited [ i + 1 ] = 1 ; } } }
public static void Main ( ) { int [ ] weights = { 1 , 2 , 1 , 2 , 5 } ; int N = weights . Length ;
construct_tree ( weights , N ) ; } }
using System ; class GFG {
static void minCost ( string s , int k ) {
int n = s . Length ;
int ans = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
int [ ] a = new int [ 26 ] ; for ( int j = i ; j < n ; j += k ) { a [ s [ j ] - ' a ' ] ++ ; }
int min_cost = Int32 . MaxValue ;
for ( int ch = 0 ; ch < 26 ; ch ++ ) { int cost = 0 ;
for ( int tr = 0 ; tr < 26 ; tr ++ ) cost += Math . Abs ( ch - tr ) * a [ tr ] ;
min_cost = Math . Min ( min_cost , cost ) ; }
ans += min_cost ; }
Console . WriteLine ( ans ) ; }
public static void Main ( ) {
string S = " abcdefabc " ; int K = 3 ;
minCost ( S , K ) ; } }
using System ; class GFG {
static int minAbsDiff ( int N ) { if ( N % 4 == 0 N % 4 == 3 ) { return 0 ; } return 1 ; }
public static void Main ( String [ ] args ) { int N = 6 ; Console . WriteLine ( minAbsDiff ( N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG { static readonly int N = 10000 ;
static List < int > [ ] adj = new List < int > [ N ] ; static int [ ] used = new int [ N ] ; static int max_matching ;
static void AddEdge ( int u , int v ) {
adj [ u ] . Add ( v ) ;
adj [ v ] . Add ( u ) ; }
static void Matching_dfs ( int u , int p ) { for ( int i = 0 ; i < adj [ u ] . Count ; i ++ ) {
if ( adj [ u ] [ i ] != p ) { Matching_dfs ( adj [ u ] [ i ] , u ) ; } }
if ( used [ u ] == 0 && used [ p ] == 0 && p != 0 ) {
max_matching ++ ; used [ u ] = used [ p ] = 1 ; } }
static void maxMatching ( ) {
Matching_dfs ( 1 , 0 ) ;
Console . Write ( max_matching + " STRNEWLINE " ) ; }
public static void Main ( String [ ] args ) { for ( int i = 0 ; i < adj . Length ; i ++ ) adj [ i ] = new List < int > ( ) ;
AddEdge ( 1 , 2 ) ; AddEdge ( 1 , 3 ) ; AddEdge ( 3 , 4 ) ; AddEdge ( 3 , 5 ) ;
maxMatching ( ) ; } }
using System ; class GFG {
static int getMinCost ( int [ ] A , int [ ] B , int N ) { int mini = int . MaxValue ; for ( int i = 0 ; i < N ; i ++ ) { mini = Math . Min ( mini , Math . Min ( A [ i ] , B [ i ] ) ) ; }
return mini * ( 2 * N - 1 ) ; }
public static void Main ( String [ ] args ) { int N = 3 ; int [ ] A = { 1 , 4 , 2 } ; int [ ] B = { 10 , 6 , 12 } ; Console . Write ( getMinCost ( A , B , N ) ) ; } }
using System ; using System . Collections . Generic ; class GFG {
static void printList ( List < int > arr ) { if ( arr . Count != 1 ) {
for ( int i = 0 ; i < arr . Count ; i ++ ) { Console . Write ( arr [ i ] + " ▁ " ) ; } Console . WriteLine ( ) ; } }
static void findWays ( List < int > arr , int i , int n ) {
if ( n == 0 ) printList ( arr ) ;
for ( int j = i ; j <= n ; j ++ ) {
arr . Add ( j ) ;
findWays ( arr , j , n - j ) ;
arr . RemoveAt ( arr . Count - 1 ) ; } }
public static void Main ( String [ ] args ) {
int n = 4 ;
List < int > arr = new List < int > ( ) ;
findWays ( arr , 1 , n ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static void Maximum_subsequence ( int [ ] A , int N ) {
Dictionary < int , int > frequency = new Dictionary < int , int > ( ) ;
int max_freq = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( frequency . ContainsKey ( A [ i ] ) ) { frequency [ A [ i ] ] = frequency [ A [ i ] ] + 1 ; } else { frequency . Add ( A [ i ] , 1 ) ; } } foreach ( KeyValuePair < int , int > it in frequency ) {
if ( ( int ) it . Value > max_freq ) { max_freq = ( int ) it . Value ; } }
Console . WriteLine ( max_freq ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 2 , 6 , 5 , 2 , 4 , 5 , 2 } ; int N = arr . Length ; Maximum_subsequence ( arr , N ) ; } }
using System ; using System . Collections . Generic ; class GFG {
public static void DivideString ( string s , int n , int k ) { int i , c = 0 , no = 1 ; int c1 = 0 , c2 = 0 ;
int [ ] fr = new int [ 26 ] ; char [ ] ans = new char [ n ] ; for ( i = 0 ; i < n ; i ++ ) { fr [ s [ i ] - ' a ' ] ++ ; } char ch = ' a ' , ch1 = ' a ' ; for ( i = 0 ; i < 26 ; i ++ ) {
if ( fr [ i ] == k ) { c ++ ; }
if ( fr [ i ] > k && fr [ i ] != 2 * k ) { c1 ++ ; ch = ( char ) ( i + ' a ' ) ; } if ( fr [ i ] == 2 * k ) { c2 ++ ; ch1 = ( char ) ( i + ' a ' ) ; } } for ( i = 0 ; i < n ; i ++ ) ans [ i ] = '1' ; Dictionary < char , int > mp = new Dictionary < char , int > ( ) ; if ( c % 2 == 0 c1 > 0 c2 > 0 ) { for ( i = 0 ; i < n ; i ++ ) {
if ( fr [ s [ i ] - ' a ' ] == k ) { if ( mp . ContainsKey ( s [ i ] ) ) { ans [ i ] = '2' ; } else { if ( no <= ( c / 2 ) ) { ans [ i ] = '2' ; no ++ ; mp [ s [ i ] ] = 1 ; } } } }
if ( ( c % 2 == 1 ) && ( c1 > 0 ) ) { no = 1 ; for ( i = 0 ; i < n ; i ++ ) { if ( s [ i ] == ch && no <= k ) { ans [ i ] = '2' ; no ++ ; } } }
if ( c % 2 == 1 && c1 == 0 ) { no = 1 ; int flag = 0 ; for ( i = 0 ; i < n ; i ++ ) { if ( s [ i ] == ch1 && no <= k ) { ans [ i ] = '2' ; no ++ ; } if ( fr [ s [ i ] - ' a ' ] == k && flag == 0 && ans [ i ] == '1' ) { ans [ i ] = '2' ; flag = 1 ; } } } Console . Write ( ans ) ; } else {
Console . Write ( " NO " ) ; } }
public static void Main ( string [ ] args ) { string S = " abbbccc " ; int N = S . Length ; int K = 1 ; DivideString ( S , N , K ) ; } }
using System ; class GFG {
static String check ( int S , int [ ] prices , int [ ] type , int n ) {
for ( int j = 0 ; j < n ; j ++ ) { for ( int k = j + 1 ; k < n ; k ++ ) {
if ( ( type [ j ] == 0 && type [ k ] == 1 ) || ( type [ j ] == 1 && type [ k ] == 0 ) ) { if ( prices [ j ] + prices [ k ] <= S ) { return " Yes " ; } } } } return " No " ; }
public static void Main ( String [ ] args ) { int [ ] prices = { 3 , 8 , 6 , 5 } ; int [ ] type = { 0 , 1 , 1 , 0 } ; int S = 10 ; int n = 4 ;
Console . Write ( check ( S , prices , type , n ) ) ; } }
using System ; class GFG {
static int getLargestSum ( int N ) {
for ( int i = 1 ; i * i <= N ; i ++ ) { for ( int j = i + 1 ; j * j <= N ; j ++ ) {
int k = N / j ; int a = k * i ; int b = k * j ;
if ( a <= N && b <= N && a * b % ( a + b ) == 0 )
max_sum = Math . Max ( max_sum , a + b ) ; } }
return max_sum ; }
static public void Main ( String [ ] args ) { int N = 25 ; int max_sum = getLargestSum ( N ) ; Console . Write ( max_sum + " STRNEWLINE " ) ; } }
using System ; class GFG {
static String encryptString ( String str , int n ) { int i = 0 , cnt = 0 ; String encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- > 0 ) encryptedStr += str [ i ] ; i ++ ; } return encryptedStr ; }
static public void Main ( ) { String str = " geeks " ; int n = str . Length ; Console . WriteLine ( encryptString ( str , n ) ) ; } }
using System ; class GFG {
static int minDiff ( int n , int x , int [ ] A ) { int mn = A [ 0 ] , mx = A [ 0 ] ;
for ( int i = 0 ; i < n ; ++ i ) { mn = Math . Min ( mn , A [ i ] ) ; mx = Math . Max ( mx , A [ i ] ) ; }
return Math . Max ( 0 , mx - mn - 2 * x ) ; }
public static void Main ( ) { int n = 3 , x = 3 ; int [ ] A = { 1 , 3 , 6 } ;
Console . WriteLine ( minDiff ( n , x , A ) ) ; } }
using System ; class GFG { public static long swapCount ( string s ) { char [ ] chars = s . ToCharArray ( ) ;
int countLeft = 0 , countRight = 0 ;
int swap = 0 , imbalance = 0 ; for ( int i = 0 ; i < chars . Length ; i ++ ) { if ( chars [ i ] == ' [ ' ) {
countLeft ++ ; if ( imbalance > 0 ) {
swap += imbalance ;
imbalance -- ; } } else if ( chars [ i ] == ' ] ' ) {
countRight ++ ;
imbalance = ( countRight - countLeft ) ; } } return swap ; }
public static void Main ( string [ ] args ) { string s = " [ ] ] [ ] [ " ; Console . WriteLine ( swapCount ( s ) ) ; s = " [ [ ] [ ] ] " ; Console . WriteLine ( swapCount ( s ) ) ; } }
using System ; class GFG {
static void longestSubSequence ( int [ , ] A , int N ) {
int [ ] dp = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) {
dp [ i ] = 1 ; for ( int j = 0 ; j < i ; j ++ ) {
if ( A [ j , 0 ] < A [ i , 0 ] && A [ j , 1 ] > A [ i , 1 ] ) { dp [ i ] = Math . Max ( dp [ i ] , dp [ j ] + 1 ) ; } } }
Console . Write ( dp [ N - 1 ] ) ; }
static void Main ( ) {
int [ , ] A = { { 1 , 2 } , { 2 , 2 } , { 3 , 1 } } ; int N = A . GetLength ( 0 ) ;
longestSubSequence ( A , N ) ; } }
using System ; class GFG {
static int findWays ( int N , int [ ] dp ) {
if ( N == 0 ) { return 1 ; }
if ( dp [ N ] != - 1 ) { return dp [ N ] ; } int cnt = 0 ;
for ( int i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i , dp ) ; } }
return dp [ N ] = cnt ; }
public static void Main ( String [ ] args ) {
int N = 4 ;
int [ ] dp = new int [ N + 1 ] ; for ( int i = 0 ; i < dp . Length ; i ++ ) dp [ i ] = - 1 ;
Console . Write ( findWays ( N , dp ) ) ; } }
using System ; class GFG {
static void findWays ( int N ) {
int [ ] dp = new int [ N + 1 ] ; dp [ 0 ] = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { dp [ i ] = 0 ;
for ( int j = 1 ; j <= 6 ; j ++ ) { if ( i - j >= 0 ) { dp [ i ] = dp [ i ] + dp [ i - j ] ; } } }
Console . Write ( dp [ N ] ) ; }
public static void Main ( String [ ] args ) {
int N = 4 ;
findWays ( N ) ; } }
using System ; class GFG { static int INF = ( int ) ( 1e9 + 9 ) ;
class TrieNode { public TrieNode [ ] child = new TrieNode [ 26 ] ; } ;
static void insert ( int idx , String s , TrieNode root ) { TrieNode temp = root ; for ( int i = idx ; i < s . Length ; i ++ ) {
if ( temp . child [ s [ i ] - ' a ' ] == null )
temp . child [ s [ i ] - ' a ' ] = new TrieNode ( ) ; temp = temp . child [ s [ i ] - ' a ' ] ; } }
static int minCuts ( String S1 , String S2 ) { int n1 = S1 . Length ; int n2 = S2 . Length ;
TrieNode root = new TrieNode ( ) ; for ( int i = 0 ; i < n2 ; i ++ ) {
insert ( i , S2 , root ) ; }
int [ ] dp = new int [ n1 + 1 ] ; for ( int i = 0 ; i <= n1 ; i ++ ) dp [ i ] = INF ;
dp [ 0 ] = 0 ; for ( int i = 0 ; i < n1 ; i ++ ) {
TrieNode temp = root ; for ( int j = i + 1 ; j <= n1 ; j ++ ) { if ( temp . child [ S1 [ j - 1 ] - ' a ' ] == null )
break ;
dp [ j ] = Math . Min ( dp [ j ] , dp [ i ] + 1 ) ;
temp = temp . child [ S1 [ j - 1 ] - ' a ' ] ; } }
if ( dp [ n1 ] >= INF ) return - 1 ; else return dp [ n1 ] ; }
public static void Main ( String [ ] args ) { String S1 = " abcdab " ; String S2 = " dabc " ; Console . Write ( minCuts ( S1 , S2 ) ) ; } }
using System . Collections . Generic ; using System ; class GFG {
static void largestSquare ( int [ , ] matrix , int R , int C , int [ ] q_i , int [ ] q_j , int K , int Q ) { int [ , ] countDP = new int [ R , C ] ; for ( int i = 0 ; i < R ; i ++ ) for ( int j = 0 ; j < C ; j ++ ) countDP [ i , j ] = 0 ;
countDP [ 0 , 0 ] = matrix [ 0 , 0 ] ; for ( int i = 1 ; i < R ; i ++ ) countDP [ i , 0 ] = countDP [ i - 1 , 0 ] + matrix [ i , 0 ] ; for ( int j = 1 ; j < C ; j ++ ) countDP [ 0 , j ] = countDP [ 0 , j - 1 ] + matrix [ 0 , j ] ; for ( int i = 1 ; i < R ; i ++ ) for ( int j = 1 ; j < C ; j ++ ) countDP [ i , j ] = matrix [ i , j ] + countDP [ i - 1 , j ] + countDP [ i , j - 1 ] - countDP [ i - 1 , j - 1 ] ;
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ; int min_dist = Math . Min ( Math . Min ( i , j ) , Math . Min ( R - i - 1 , C - j - 1 ) ) ; int ans = - 1 , l = 0 , u = min_dist ;
while ( l <= u ) { int mid = ( l + u ) / 2 ; int x1 = i - mid , x2 = i + mid ; int y1 = j - mid , y2 = j + mid ;
int count = countDP [ x2 , y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 , y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 , y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 , y1 - 1 ] ;
if ( count <= K ) { ans = 2 * mid + 1 ; l = mid + 1 ; } else u = mid - 1 ; } Console . WriteLine ( ans ) ; } }
public static void Main ( ) { int [ , ] matrix = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int [ ] q_i = { 1 } ; int [ ] q_j = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; } }
