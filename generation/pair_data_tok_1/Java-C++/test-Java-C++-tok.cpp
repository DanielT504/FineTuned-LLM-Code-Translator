#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minSum ( int A [ ] , int N ) {
map < int , int > mp ; int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += A [ i ] ;
mp [ A [ i ] ] ++ ; }
int minSum = INT_MAX ;
for ( auto it : mp ) {
minSum = min ( minSum , sum - ( it . first * it . second ) ) ; }
return minSum ; }
int main ( ) {
int arr [ ] = { 4 , 5 , 6 , 6 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minSum ( arr , N ) << " STRNEWLINE " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maxAdjacent ( int * arr , int N ) { vector < int > res ;
for ( int i = 1 ; i < N - 1 ; i ++ ) { int prev = arr [ 0 ] ;
int maxi = INT_MIN ;
for ( int j = 1 ; j < N ; j ++ ) {
if ( i == j ) continue ;
maxi = max ( maxi , abs ( arr [ j ] - prev ) ) ;
prev = arr [ j ] ; }
res . push_back ( maxi ) ; }
for ( auto x : res ) cout << x << " ▁ " ; cout << endl ; }
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 7 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; maxAdjacent ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findSize ( int N ) {
if ( N == 0 ) return 1 ; if ( N == 1 ) return 1 ; int Size = 2 * findSize ( N / 2 ) + 1 ;
return Size ; }
int CountOnes ( int N , int L , int R ) { if ( L > R ) { return 0 ; }
if ( N <= 1 ) { return N ; } int ret = 0 ; int M = N / 2 ; int Siz_M = findSize ( M ) ;
if ( L <= Siz_M ) {
ret += CountOnes ( N / 2 , L , min ( Siz_M , R ) ) ; }
if ( L <= Siz_M + 1 && Siz_M + 1 <= R ) { ret += N % 2 ; }
if ( Siz_M + 1 < R ) { ret += CountOnes ( N / 2 , max ( 1 , L - Siz_M - 1 ) , R - Siz_M - 1 ) ; } return ret ; }
int main ( ) {
int N = 7 , L = 2 , R = 5 ;
cout << CountOnes ( N , L , R ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool prime ( int n ) {
if ( n == 1 ) return false ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; }
return true ; }
void minDivisior ( int n ) {
if ( prime ( n ) ) { cout << 1 << " ▁ " << n - 1 ; }
else { for ( int i = 2 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
cout << n / i << " ▁ " << n / i * ( i - 1 ) ; break ; } } } }
int main ( ) { int N = 4 ;
minDivisior ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Landau = INT_MIN ;
int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
int lcm ( int a , int b ) { return ( a * b ) / gcd ( a , b ) ; }
void findLCM ( vector < int > & arr ) { int nth_lcm = arr [ 0 ] ; for ( int i = 1 ; i < arr . size ( ) ; i ++ ) nth_lcm = lcm ( nth_lcm , arr [ i ] ) ;
Landau = max ( Landau , nth_lcm ) ; }
void findWays ( vector < int > & arr , int i , int n ) {
if ( n == 0 ) findLCM ( arr ) ;
for ( int j = i ; j <= n ; j ++ ) {
arr . push_back ( j ) ;
findWays ( arr , j , n - j ) ;
arr . pop_back ( ) ; } }
void Landau_function ( int n ) { vector < int > arr ;
findWays ( arr , 1 , n ) ;
cout << Landau ; }
int main ( ) {
int N = 4 ;
Landau_function ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) {
if ( n == 1 ) return true ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ;
for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
void checkExpression ( int n ) { if ( isPrime ( n ) ) cout << " Yes " ; else cout << " No " ; }
int main ( ) { int N = 3 ; checkExpression ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkArray ( int n , int k , int arr [ ] ) {
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] & 1 ) cnt += 1 ; }
if ( cnt >= k && cnt % 2 == k % 2 ) return true ; else return false ; }
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 7 , 5 , 3 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 4 ; if ( checkArray ( n , k , arr ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define ll  long long NEW_LINE using namespace std ;
int func ( int arr [ ] , int n ) { double ans = 0 ; int maxx = 0 ; double freq [ 100005 ] = { 0 } ; int temp ;
for ( int i = 0 ; i < n ; i ++ ) { temp = arr [ i ] ; freq [ temp ] ++ ; maxx = max ( maxx , temp ) ; }
for ( int i = 1 ; i <= maxx ; i ++ ) { freq [ i ] += freq [ i - 1 ] ; } for ( int i = 1 ; i <= maxx ; i ++ ) { if ( freq [ i ] ) { i = ( double ) i ; double j ; ll value = 0 ;
double cur = ceil ( 0.5 * i ) - 1.0 ; for ( j = 1.5 ; ; j ++ ) { int val = min ( maxx , ( int ) ( ceil ( i * j ) - 1.0 ) ) ; int times = ( freq [ i ] - freq [ i - 1 ] ) , con = j - 0.5 ;
ans += times * con * ( freq [ ( int ) val ] - freq [ ( int ) cur ] ) ; cur = val ; if ( val == maxx ) break ; } } }
return ( ll ) ans ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << func ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void insert_element ( int a [ ] , int n ) {
int Xor = 0 ;
int Sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { Xor ^= a [ i ] ; Sum += a [ i ] ; }
if ( Sum == 2 * Xor ) {
cout << "0" << endl ; return ; }
if ( Xor == 0 ) { cout << "1" << endl ; cout << Sum << endl ; return ; }
int num1 = Sum + Xor ; int num2 = Xor ;
cout << "2" ;
cout << num1 << " ▁ " << num2 << endl ; }
int main ( ) { int a [ ] = { 1 , 2 , 3 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; insert_element ( a , n ) ; }
#include <iostream> NEW_LINE using namespace std ;
void checkSolution ( int a , int b , int c ) { if ( a == c ) cout << " Yes " ; else cout << " No " ; }
int main ( ) { int a = 2 , b = 0 , c = 2 ; checkSolution ( a , b , c ) ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE using namespace std ;
bool isPerfectSquare ( long double x ) {
long double sr = sqrt ( x ) ;
return ( ( sr - floor ( sr ) ) == 0 ) ; }
void checkSunnyNumber ( int N ) {
if ( isPerfectSquare ( N + 1 ) ) { cout << " Yes STRNEWLINE " ; }
else { cout << " No STRNEWLINE " ; } }
int main ( ) {
int N = 8 ;
checkSunnyNumber ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countValues ( int n ) { int answer = 0 ;
for ( int i = 2 ; i <= n ; i ++ ) { int k = n ;
while ( k >= i ) { if ( k % i == 0 ) k /= i ; else k -= i ; }
if ( k == 1 ) answer ++ ; } return answer ; }
int main ( ) { int N = 6 ; cout << countValues ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printKNumbers ( int N , int K ) {
for ( int i = 0 ; i < K - 1 ; i ++ ) cout << 1 << " ▁ " ;
cout << ( N - K + 1 ) ; }
int main ( ) { int N = 10 , K = 3 ; printKNumbers ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int NthSmallest ( int K ) {
queue < int > Q ; int x ;
for ( int i = 1 ; i < 10 ; i ++ ) Q . push ( i ) ;
for ( int i = 1 ; i <= K ; i ++ ) {
x = Q . front ( ) ;
Q . pop ( ) ;
if ( x % 10 != 0 ) {
Q . push ( x * 10 + x % 10 - 1 ) ; }
Q . push ( x * 10 + x % 10 ) ;
if ( x % 10 != 9 ) {
Q . push ( x * 10 + x % 10 + 1 ) ; } }
return x ; }
int main ( ) {
int N = 16 ; cout << NthSmallest ( N ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int nearest ( int n ) {
int prevSquare = sqrt ( n ) ; int nextSquare = prevSquare + 1 ; prevSquare = prevSquare * prevSquare ; nextSquare = nextSquare * nextSquare ;
int ans = ( n - prevSquare ) < ( nextSquare - n ) ? ( prevSquare - n ) : ( nextSquare - n ) ;
return ans ; }
int main ( ) { int n = 14 ; cout << nearest ( n ) << endl ; n = 16 ; cout << nearest ( n ) << endl ; n = 18 ; cout << nearest ( n ) << endl ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE using namespace std ;
void printValueOfPi ( int N ) {
double pi = 2 * acos ( 0.0 ) ;
printf ( " % . * lf STRNEWLINE " , N , pi ) ; }
int main ( ) { int N = 45 ;
printValueOfPi ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void decBinary ( int arr [ ] , int n ) { int k = log2 ( n ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n /= 2 ; } }
int binaryDec ( int arr [ ] , int n ) { int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
int getNum ( int n , int k ) {
int l = log2 ( n ) + 1 ;
int a [ l ] = { 0 } ; decBinary ( a , n ) ;
if ( k > l ) return n ;
a [ k - 1 ] = ( a [ k - 1 ] == 0 ) ? 1 : 0 ;
return binaryDec ( a , l ) ; }
int main ( ) { int n = 56 , k = 2 ; cout << getNum ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ll  long long NEW_LINE #define MAX  1000000 NEW_LINE const ll MOD = 1e9 + 7 ;
ll result [ MAX + 1 ] ; ll fact [ MAX + 1 ] ;
void preCompute ( ) {
fact [ 0 ] = 1 ; result [ 0 ] = 1 ;
for ( int i = 1 ; i <= MAX ; i ++ ) {
fact [ i ] = ( ( fact [ i - 1 ] % MOD ) * i ) % MOD ;
result [ i ] = ( ( result [ i - 1 ] % MOD ) * ( fact [ i ] % MOD ) ) % MOD ; } }
void performQueries ( int q [ ] , int n ) {
preCompute ( ) ;
for ( int i = 0 ; i < n ; i ++ ) cout << result [ q [ i ] ] << " STRNEWLINE " ; }
int main ( ) { int q [ ] = { 4 , 5 } ; int n = sizeof ( q ) / sizeof ( q [ 0 ] ) ; performQueries ( q , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
long divTermCount ( long a , long b , long c , long num ) {
return ( ( num / a ) + ( num / b ) + ( num / c ) - ( num / ( ( a * b ) / gcd ( a , b ) ) ) - ( num / ( ( c * b ) / gcd ( c , b ) ) ) - ( num / ( ( a * c ) / gcd ( a , c ) ) ) + ( num / ( ( ( ( a * b ) / gcd ( a , b ) ) * c ) / gcd ( ( ( a * b ) / gcd ( a , b ) ) , c ) ) ) ) ; }
int findNthTerm ( int a , int b , int c , long n ) {
long low = 1 , high = LONG_MAX , mid ; while ( low < high ) { mid = low + ( high - low ) / 2 ;
if ( divTermCount ( a , b , c , mid ) < n ) low = mid + 1 ;
else high = mid ; } return low ; }
int main ( ) { long a = 2 , b = 3 , c = 5 , n = 100 ; cout << findNthTerm ( a , b , c , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double calculate_angle ( int n , int i , int j , int k ) {
int x , y ;
if ( i < j ) x = j - i ; else x = j + n - i ; if ( j < k ) y = k - j ; else y = k + n - j ;
double ang1 = ( 180 * x ) / n ; double ang2 = ( 180 * y ) / n ;
double ans = 180 - ang1 - ang2 ; return ans ; }
int main ( ) { int n = 5 ; int a1 = 1 ; int a2 = 2 ; int a3 = 5 ; cout << calculate_angle ( n , a1 , a2 , a3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void Loss ( int SP , int P ) { float loss = 0 ; loss = ( 2 * P * P * SP ) / float ( 100 * 100 - P * P ) ; cout << " Loss ▁ = ▁ " << loss ; }
int main ( ) { int SP = 2400 , P = 30 ;
Loss ( SP , P ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAXN  1000001
int spf [ MAXN ] ;
int hash1 [ MAXN ] = { 0 } ;
void sieve ( ) { spf [ 1 ] = 1 ; for ( int i = 2 ; i < MAXN ; i ++ )
spf [ i ] = i ;
for ( int i = 4 ; i < MAXN ; i += 2 ) spf [ i ] = 2 ;
for ( int i = 3 ; i * i < MAXN ; i ++ ) {
if ( spf [ i ] == i ) { for ( int j = i * i ; j < MAXN ; j += i )
if ( spf [ j ] == j ) spf [ j ] = i ; } } }
void getFactorization ( int x ) { int temp ; while ( x != 1 ) { temp = spf [ x ] ; if ( x % temp == 0 ) {
hash1 [ spf [ x ] ] ++ ; x = x / spf [ x ] ; } while ( x % temp == 0 ) x = x / temp ; } }
bool check ( int x ) { int temp ; while ( x != 1 ) { temp = spf [ x ] ;
if ( x % temp == 0 && hash1 [ temp ] > 1 ) return false ; while ( x % temp == 0 ) x = x / temp ; } return true ; }
bool hasValidNum ( int arr [ ] , int n ) {
sieve ( ) ; for ( int i = 0 ; i < n ; i ++ ) getFactorization ( arr [ i ] ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( check ( arr [ i ] ) ) return true ; return false ; }
int main ( ) { int arr [ ] = { 2 , 8 , 4 , 10 , 6 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; if ( hasValidNum ( arr , n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countWays ( int N ) {
int E = ( N * ( N - 1 ) ) / 2 ; if ( N == 1 ) return 0 ; return pow ( 2 , E - 1 ) ; }
int main ( ) { int N = 4 ; cout << countWays ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int l [ 1001 ] [ 1001 ] = { 0 } ; void initialize ( ) {
l [ 0 ] [ 0 ] = 1 ; for ( int i = 1 ; i < 1001 ; i ++ ) {
l [ i ] [ 0 ] = 1 ; for ( int j = 1 ; j < i + 1 ; j ++ ) {
l [ i ] [ j ] = ( l [ i - 1 ] [ j - 1 ] + l [ i - 1 ] [ j ] ) ; } } }
int nCr ( int n , int r ) {
return l [ n ] [ r ] ; }
int main ( ) {
initialize ( ) ; int n = 8 ; int r = 3 ; cout << nCr ( n , r ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minAbsDiff ( int n ) { int mod = n % 4 ; if ( mod == 0 mod == 3 ) return 0 ; return 1 ; }
int main ( ) { int n = 5 ; cout << minAbsDiff ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool check ( int s ) {
int freq [ 10 ] = { 0 } , r ; while ( s != 0 ) {
r = s % 10 ;
s = int ( s / 10 ) ;
freq [ r ] += 1 ; } int xor__ = 0 ;
for ( int i = 0 ; i < 10 ; i ++ ) { xor__ = xor__ ^ freq [ i ] ; if ( xor__ == 0 ) return true ; else return false ; } }
int main ( ) { int s = 122233 ; if ( check ( s ) ) cout << " Yes " << endl ; else cout << " No " << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printLines ( int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) { cout << k * ( 6 * i + 1 ) << " ▁ " << k * ( 6 * i + 2 ) << " ▁ " << k * ( 6 * i + 3 ) << " ▁ " << k * ( 6 * i + 5 ) << endl ; } }
int main ( ) { int n = 2 , k = 2 ; printLines ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int calculateSum ( int n ) {
return ( pow ( 2 , n + 1 ) + n - 2 ) ; }
int main ( ) {
int n = 4 ;
cout << " Sum ▁ = ▁ " << calculateSum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define mod  1000000007
long count_special ( long n ) {
long fib [ n + 1 ] ;
fib [ 0 ] = 1 ;
fib [ 1 ] = 2 ; for ( int i = 2 ; i <= n ; i ++ ) {
fib [ i ] = ( fib [ i - 1 ] % mod + fib [ i - 2 ] % mod ) % mod ; }
return fib [ n ] ; }
int main ( ) {
long n = 3 ; cout << count_special ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int mod = 1e9 + 7 ;
int ways ( int i , int arr [ ] , int n ) {
if ( i == n - 1 ) return 1 ; int sum = 0 ;
for ( int j = 1 ; j + i < n && j <= arr [ i ] ; j ++ ) { sum += ( ways ( i + j , arr , n ) ) % mod ; sum %= mod ; } return sum % mod ; }
int main ( ) { int arr [ ] = { 5 , 3 , 1 , 4 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << ways ( 0 , arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int mod = 1e9 + 7 ;
int ways ( int arr [ ] , int n ) {
int dp [ n + 1 ] ;
dp [ n - 1 ] = 1 ;
for ( int i = n - 2 ; i >= 0 ; i -- ) { dp [ i ] = 0 ;
for ( int j = 1 ; ( ( j + i ) < n && j <= arr [ i ] ) ; j ++ ) { dp [ i ] += dp [ i + j ] ; dp [ i ] %= mod ; } }
return dp [ 0 ] % mod ; }
int main ( ) { int arr [ ] = { 5 , 3 , 1 , 4 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << ways ( arr , n ) % mod << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
pair < int , int > countSum ( int arr [ ] , int n ) { int result = 0 ;
int count_odd , count_even ;
count_odd = 0 ; count_even = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) {
if ( arr [ i - 1 ] % 2 == 0 ) { count_even = count_even + count_even + 1 ; count_odd = count_odd + count_odd ; }
else { int temp = count_even ; count_even = count_even + count_odd ; count_odd = count_odd + temp + 1 ; } } return { count_even , count_odd } ; }
int main ( ) { int arr [ ] = { 1 , 2 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
pair < int , int > ans = countSum ( arr , n ) ; cout << " EvenSum ▁ = ▁ " << ans . first ; cout << " ▁ OddSum ▁ = ▁ " << ans . second ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  10
vector < int > numToVec ( int N ) { vector < int > digit ;
while ( N != 0 ) { digit . push_back ( N % 10 ) ; N = N / 10 ; }
if ( digit . size ( ) == 0 ) digit . push_back ( 0 ) ;
reverse ( digit . begin ( ) , digit . end ( ) ) ;
return digit ; }
int solve ( vector < int > & A , int B , int C ) { vector < int > digit ; int d , d2 ;
digit = numToVec ( C ) ; d = A . size ( ) ;
if ( B > digit . size ( ) d == 0 ) return 0 ;
else if ( B < digit . size ( ) ) {
if ( A [ 0 ] == 0 && B != 1 ) return ( d - 1 ) * pow ( d , B - 1 ) ; else return pow ( d , B ) ; }
else { int dp [ B + 1 ] = { 0 } ; int lower [ MAX + 1 ] = { 0 } ;
for ( int i = 0 ; i < d ; i ++ ) lower [ A [ i ] + 1 ] = 1 ; for ( int i = 1 ; i <= MAX ; i ++ ) lower [ i ] = lower [ i - 1 ] + lower [ i ] ; bool flag = true ; dp [ 0 ] = 0 ; for ( int i = 1 ; i <= B ; i ++ ) { d2 = lower [ digit [ i - 1 ] ] ; dp [ i ] = dp [ i - 1 ] * d ;
if ( i == 1 && A [ 0 ] == 0 && B != 1 ) d2 = d2 - 1 ;
if ( flag ) dp [ i ] += d2 ;
flag = ( flag & ( lower [ digit [ i - 1 ] + 1 ] == lower [ digit [ i - 1 ] ] + 1 ) ) ; } return dp [ B ] ; } }
int main ( ) { vector < int > A = { 0 , 1 , 2 , 5 } ; int N = 2 ; int k = 21 ; cout << solve ( A , N , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int solve ( int dp [ ] [ 2 ] , int wt , int K , int M , int used ) {
if ( wt < 0 ) return 0 ; if ( wt == 0 ) {
if ( used ) return 1 ; return 0 ; } if ( dp [ wt ] [ used ] != -1 ) return dp [ wt ] [ used ] ; int ans = 0 ; for ( int i = 1 ; i <= K ; i ++ ) {
if ( i >= M ) ans += solve ( dp , wt - i , K , M , used 1 ) ; else ans += solve ( dp , wt - i , K , M , used ) ; } return dp [ wt ] [ used ] = ans ; }
int main ( ) { int W = 3 , K = 3 , M = 2 ; int dp [ W + 1 ] [ 2 ] ; memset ( dp , -1 , sizeof ( dp ) ) ; cout << solve ( dp , W , K , M , 0 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long partitions ( int n ) { vector < long long > p ( n + 1 , 0 ) ;
p [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; ++ i ) { int k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 ? 1 : -1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) k *= -1 ; else k = 1 - k ; } } return p [ n ] ; }
int main ( ) { int N = 20 ; cout << partitions ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define MAX  10 NEW_LINE using namespace std ;
int LIP ( int dp [ ] [ MAX ] , int mat [ ] [ MAX ] , int n , int m , int x , int y ) {
if ( dp [ x ] [ y ] < 0 ) { int result = 0 ;
if ( x == n - 1 && y == m - 1 ) return dp [ x ] [ y ] = 1 ;
if ( x == n - 1 y == m - 1 ) result = 1 ;
if ( mat [ x ] [ y ] < mat [ x + 1 ] [ y ] ) result = 1 + LIP ( dp , mat , n , m , x + 1 , y ) ;
if ( mat [ x ] [ y ] < mat [ x ] [ y + 1 ] ) result = max ( result , 1 + LIP ( dp , mat , n , m , x , y + 1 ) ) ; dp [ x ] [ y ] = result ; } return dp [ x ] [ y ] ; }
int wrapper ( int mat [ ] [ MAX ] , int n , int m ) { int dp [ MAX ] [ MAX ] ; memset ( dp , -1 , sizeof dp ) ; return LIP ( dp , mat , n , m , 0 , 0 ) ; }
int main ( ) { int mat [ ] [ MAX ] = { { 1 , 2 , 3 , 4 } , { 2 , 2 , 3 , 4 } , { 3 , 2 , 3 , 4 } , { 4 , 5 , 6 , 7 } , } ; int n = 4 , m = 4 ; cout << wrapper ( mat , n , m ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countPaths ( int n , int m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
int main ( ) { int n = 3 , m = 2 ; cout << " ▁ Number ▁ of ▁ Paths ▁ " << countPaths ( n , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 100 ;
int getMaxGold ( int gold [ ] [ MAX ] , int m , int n ) {
int goldTable [ m ] [ n ] ; memset ( goldTable , 0 , sizeof ( goldTable ) ) ; for ( int col = n - 1 ; col >= 0 ; col -- ) { for ( int row = 0 ; row < m ; row ++ ) {
int right = ( col == n - 1 ) ? 0 : goldTable [ row ] [ col + 1 ] ;
int right_up = ( row == 0 col == n - 1 ) ? 0 : goldTable [ row - 1 ] [ col + 1 ] ;
int right_down = ( row == m - 1 col == n - 1 ) ? 0 : goldTable [ row + 1 ] [ col + 1 ] ;
goldTable [ row ] [ col ] = gold [ row ] [ col ] + max ( right , max ( right_up , right_down ) ) ; } }
int res = goldTable [ 0 ] [ 0 ] ; for ( int i = 1 ; i < m ; i ++ ) res = max ( res , goldTable [ i ] [ 0 ] ) ; return res ; }
int main ( ) { int gold [ MAX ] [ MAX ] = { { 1 , 3 , 1 , 5 } , { 2 , 2 , 4 , 1 } , { 5 , 0 , 2 , 3 } , { 0 , 6 , 1 , 2 } } ; int m = 4 , n = 4 ; cout << getMaxGold ( gold , m , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define M  100
int minAdjustmentCost ( int A [ ] , int n , int target ) {
int dp [ n ] [ M + 1 ] ;
for ( int j = 0 ; j <= M ; j ++ ) dp [ 0 ] [ j ] = abs ( j - A [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) {
for ( int j = 0 ; j <= M ; j ++ ) {
dp [ i ] [ j ] = INT_MAX ;
for ( int k = max ( j - target , 0 ) ; k <= min ( M , j + target ) ; k ++ ) dp [ i ] [ j ] = min ( dp [ i ] [ j ] , dp [ i - 1 ] [ k ] + abs ( A [ i ] - j ) ) ; } }
int res = INT_MAX ; for ( int j = 0 ; j <= M ; j ++ ) res = min ( res , dp [ n - 1 ] [ j ] ) ; return res ; }
int main ( ) { int arr [ ] = { 55 , 77 , 52 , 61 , 39 , 6 , 25 , 60 , 49 , 47 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int target = 10 ; cout << " Minimum ▁ adjustment ▁ cost ▁ is ▁ " << minAdjustmentCost ( arr , n , target ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int totalCombination ( int L , int R ) {
int count = 0 ;
int K = R - L ;
if ( K < L ) return 0 ;
int ans = K - L ;
count = ( ( ans + 1 ) * ( ans + 2 ) ) / 2 ;
return count ; }
int main ( ) { int L = 2 , R = 6 ; cout << totalCombination ( L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printArrays ( int n ) {
vector < int > A , B ;
for ( int i = 1 ; i <= 2 * n ; i ++ ) {
if ( i % 2 == 0 ) A . push_back ( i ) ; else B . push_back ( i ) ; }
cout << " { ▁ " ; for ( int i = 0 ; i < n ; i ++ ) { cout << A [ i ] ; if ( i != n - 1 ) cout << " , ▁ " ; } cout << " ▁ } STRNEWLINE " ;
cout << " { ▁ " ; for ( int i = 0 ; i < n ; i ++ ) { cout << B [ i ] ; if ( i != n - 1 ) cout << " , ▁ " ; } cout << " ▁ } " ; }
int main ( ) { int N = 5 ;
printArrays ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void flipBitsOfAandB ( int A , int B ) {
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( A & ( 1 << i ) ) && ( B & ( 1 << i ) ) ) {
A = A ^ ( 1 << i ) ;
B = B ^ ( 1 << i ) ; } }
cout << A << " ▁ " << B ; }
int main ( ) { int A = 7 , B = 4 ; flipBitsOfAandB ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findDistinctSums ( int N ) { return ( 2 * N - 1 ) ; }
int main ( ) { int N = 3 ; cout << findDistinctSums ( N ) ; return 0 ; }
#include <iostream> NEW_LINE #include <string> NEW_LINE using namespace std ;
int countSubstrings ( string & str ) {
int freq [ 3 ] = { 0 } ;
int count = 0 ; int i = 0 ;
for ( int j = 0 ; j < str . length ( ) ; j ++ ) {
freq [ str [ j ] - '0' ] ++ ;
while ( freq [ 0 ] > 0 && freq [ 1 ] > 0 && freq [ 2 ] > 0 ) { freq [ str [ i ++ ] - '0' ] -- ; }
count += i ; }
return count ; }
int main ( ) { string str = "00021" ; int count = countSubstrings ( str ) ; cout << count ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minFlips ( string str ) {
int count = 0 ;
if ( str . size ( ) <= 2 ) { return 0 ; }
for ( int i = 0 ; i < str . size ( ) - 2 ; ) {
if ( str [ i ] == str [ i + 1 ] && str [ i + 2 ] == str [ i + 1 ] ) { i = i + 3 ; count ++ ; } else { i ++ ; } }
return count ; }
int main ( ) { string S = "0011101" ; cout << minFlips ( S ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string convertToHex ( int num ) { string temp = " " ; while ( num != 0 ) { int rem = num % 16 ; char c ; if ( rem < 10 ) { c = rem + 48 ; } else { c = rem + 87 ; } temp += c ; num = num / 16 ; } return temp ; }
string encryptString ( string S , int N ) { string ans = " " ;
for ( int i = 0 ; i < N ; i ++ ) { char ch = S [ i ] ; int count = 0 ; string hex ;
while ( i < N && S [ i ] == ch ) {
count ++ ; i ++ ; }
i -- ;
hex = convertToHex ( count ) ;
ans += ch ;
ans += hex ; }
reverse ( ans . begin ( ) , ans . end ( ) ) ;
return ans ; }
int main ( ) {
string S = " abc " ; int N = S . size ( ) ;
cout << encryptString ( S , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unsigned long int binomialCoeff ( unsigned long int n , unsigned long int k ) { unsigned long int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
unsigned long int countOfString ( int N ) {
unsigned long int Stotal = pow ( 2 , N ) ;
unsigned long int Sequal = 0 ;
if ( N % 2 == 0 ) Sequal = binomialCoeff ( N , N / 2 ) ; unsigned long int S1 = ( Stotal - Sequal ) / 2 ; return S1 ; }
int main ( ) { int N = 3 ; cout << countOfString ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string removeCharRecursive ( string str , char X ) {
if ( str . length ( ) == 0 ) { return " " ; }
if ( str [ 0 ] == X ) {
return removeCharRecursive ( str . substr ( 1 ) , X ) ; }
return str [ 0 ] + removeCharRecursive ( str . substr ( 1 ) , X ) ; }
int main ( ) {
string str = " geeksforgeeks " ;
char X = ' e ' ;
str = removeCharRecursive ( str , X ) ; cout << str ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isValid ( char a1 , char a2 , string str , int flag ) { char v1 , v2 ;
if ( flag == 0 ) { v1 = str [ 4 ] ; v2 = str [ 3 ] ; } else {
v1 = str [ 1 ] ; v2 = str [ 0 ] ; }
if ( v1 != a1 && v1 != ' ? ' ) return false ; if ( v2 != a2 && v2 != ' ? ' ) return false ; return true ; }
bool inRange ( int hh , int mm , int L , int R ) { int a = abs ( hh - mm ) ;
if ( a < L a > R ) return false ; return true ; }
void displayTime ( int hh , int mm ) { if ( hh > 10 ) cout << hh << " : " ; else if ( hh < 10 ) cout << "0" << hh << " : " ; if ( mm > 10 ) cout << mm << endl ; else if ( mm < 10 ) cout << "0" << mm << endl ; }
void maximumTimeWithDifferenceInRange ( string str , int L , int R ) { int i , j ; int h1 , h2 , m1 , m2 ;
for ( i = 23 ; i >= 0 ; i -- ) { h1 = i % 10 ; h2 = i / 10 ;
if ( ! isValid ( h1 + '0' , h2 + '0' , str , 1 ) ) { continue ; }
for ( j = 59 ; j >= 0 ; j -- ) { m1 = j % 10 ; m2 = j / 10 ;
if ( ! isValid ( m1 + '0' , m2 + '0' , str , 0 ) ) { continue ; } if ( inRange ( i , j , L , R ) ) { displayTime ( i , j ) ; return ; } } } if ( inRange ( i , j , L , R ) ) displayTime ( i , j ) ; else cout << " - 1" << endl ; }
int main ( ) {
string timeValue = " ? ? : ? ? " ;
int L = 20 , R = 39 ; maximumTimeWithDifferenceInRange ( timeValue , L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool check ( string s , int n ) {
stack < char > st ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( ! st . empty ( ) && st . top ( ) == s [ i ] ) st . pop ( ) ;
else st . push ( s [ i ] ) ; }
if ( st . empty ( ) ) { return true ; }
else { return false ; } }
int main ( ) {
string str = " aanncddc " ; int n = str . length ( ) ;
if ( check ( str , n ) ) { cout << " Yes " << endl ; } else { cout << " No " << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findNumOfValidWords ( vector < string > & w , vector < string > & p ) {
unordered_map < int , int > m ;
vector < int > res ;
for ( string & s : w ) { int val = 0 ;
for ( char c : s ) { val = val | ( 1 << ( c - ' a ' ) ) ; }
m [ val ] ++ ; }
for ( string & s : p ) { int val = 0 ;
for ( char c : s ) { val = val | ( 1 << ( c - ' a ' ) ) ; } int temp = val ; int first = s [ 0 ] - ' a ' ; int count = 0 ; while ( temp != 0 ) {
if ( ( ( temp >> first ) & 1 ) == 1 ) { if ( m . find ( temp ) != m . end ( ) ) { count += m [ temp ] ; } }
temp = ( temp - 1 ) & val ; }
res . push_back ( count ) ; }
for ( auto & it : res ) { cout << it << ' ' ; } }
int main ( ) { vector < string > arr1 ; arr1 = { " aaaa " , " asas " , " able " , " ability " , " actt " , " actor " , " access " } ; vector < string > arr2 ; arr2 = { " aboveyz " , " abrodyz " , " absolute " , " absoryz " , " actresz " , " gaswxyz " } ;
findNumOfValidWords ( arr1 , arr2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void flip ( string & s ) { for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( s [ i ] == '0' ) {
while ( s [ i ] == '0' ) {
s [ i ] = '1' ; i ++ ; }
break ; } } }
int main ( ) { string s = "100010001" ; flip ( s ) ; cout << s ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void getOrgString ( string s ) {
cout << s [ 0 ] ;
int i = 1 ; while ( i < s . length ( ) ) {
if ( s [ i ] >= ' A ' && s [ i ] <= ' Z ' ) cout << " ▁ " << ( char ) tolower ( s [ i ] ) ;
else cout < < s [ i ] ; i ++ ; } }
int main ( ) { string s = " ILoveGeeksForGeeks " ; getOrgString ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countChar ( string str , char x ) { int count = 0 , n = 10 ; for ( int i = 0 ; i < str . size ( ) ; i ++ ) if ( str [ i ] == x ) count ++ ;
int repetitions = n / str . size ( ) ; count = count * repetitions ;
for ( int i = 0 ; i < n % str . size ( ) ; i ++ ) { if ( str [ i ] == x ) count ++ ; } return count ; }
int main ( ) { string str = " abcac " ; cout << countChar ( str , ' a ' ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void countFreq ( int arr [ ] , int n , int limit ) {
vector < int > count ( limit + 1 , 0 ) ;
for ( int i = 0 ; i < n ; i ++ ) count [ arr [ i ] ] ++ ; for ( int i = 0 ; i <= limit ; i ++ ) if ( count [ i ] > 0 ) cout << i << " ▁ " << count [ i ] << endl ; }
int main ( ) { int arr [ ] = { 5 , 5 , 6 , 6 , 5 , 6 , 1 , 2 , 3 , 10 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int limit = 10 ; countFreq ( arr , n , limit ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <stdio.h> NEW_LINE using namespace std ;
bool check ( string s , int m ) {
int l = s . length ( ) ;
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( s [ i ] == '0' ) { c2 = 0 ;
c1 ++ ; } else { c1 = 0 ;
c2 ++ ; } if ( c1 == m c2 == m ) return true ; } return false ; }
int main ( ) { string s = "001001" ; int m = 2 ;
if ( check ( s , m ) ) cout << " YES " ; else cout << " NO " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int productAtKthLevel ( string tree , int k ) { int level = -1 ;
int n = tree . length ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( tree [ i ] == ' ( ' ) level ++ ;
else if ( tree [ i ] == ' ) ' ) level -- ; else {
if ( level == k ) product *= ( tree [ i ] - '0' ) ; } }
return product ; }
int main ( ) { string tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; cout << productAtKthLevel ( tree , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findDuplciates ( string a [ ] , int n , int m ) {
bool isPresent [ n ] [ m ] ; memset ( isPresent , 0 , sizeof ( isPresent ) ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
for ( int k = 0 ; k < n ; k ++ ) { if ( a [ i ] [ j ] == a [ k ] [ j ] && i != k ) { isPresent [ i ] [ j ] = true ; isPresent [ k ] [ j ] = true ; } }
for ( int k = 0 ; k < m ; k ++ ) { if ( a [ i ] [ j ] == a [ i ] [ k ] && j != k ) { isPresent [ i ] [ j ] = true ; isPresent [ i ] [ k ] = true ; } } } } for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < m ; j ++ )
if ( ! isPresent [ i ] [ j ] ) printf ( " % c " , a [ i ] [ j ] ) ; }
int main ( ) { int n = 2 , m = 5 ;
string a [ ] = { " zx " , " xz " } ;
findDuplciates ( a , n , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isValidISBN ( string & isbn ) {
int n = isbn . length ( ) ; if ( n != 10 ) return false ;
int sum = 0 ; for ( int i = 0 ; i < 9 ; i ++ ) { int digit = isbn [ i ] - '0' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
char last = isbn [ 9 ] ; if ( last != ' X ' && ( last < '0' last > '9' ) ) return false ;
sum += ( ( last == ' X ' ) ? 10 : ( last - '0' ) ) ;
return ( sum % 11 == 0 ) ; }
int main ( ) { string isbn = "007462542X " ; if ( isValidISBN ( isbn ) ) cout << " Valid " ; else cout << " Invalid " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isVowel ( char c ) { return ( c == ' a ' c == ' A ' c == ' e ' c == ' E ' c == ' i ' c == ' I ' c == ' o ' c == ' O ' c == ' u ' c == ' U ' ) ; }
string reverseVowel ( string str ) { int j = 0 ;
string vowel ; for ( int i = 0 ; str [ i ] != ' \0' ; i ++ ) if ( isVowel ( str [ i ] ) ) vowel [ j ++ ] = str [ i ] ;
for ( int i = 0 ; str [ i ] != ' \0' ; i ++ ) if ( isVowel ( str [ i ] ) ) str [ i ] = vowel [ -- j ] ; return str ; }
int main ( ) { string str = " hello ▁ world " ; cout << reverseVowel ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string firstLetterWord ( string str ) { string result = " " ;
bool v = true ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
if ( str [ i ] == ' ▁ ' ) v = true ;
else if ( str [ i ] != ' ▁ ' && v == true ) { result . push_back ( str [ i ] ) ; v = false ; } } return result ; }
int main ( ) { string str = " geeks ▁ for ▁ geeks " ; cout << firstLetterWord ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void dfs ( int i , int j , vector < vector < int > > & grid , vector < vector < bool > > & vis , int & ans , int z , int z_count ) { int n = grid . size ( ) , m = grid [ 0 ] . size ( ) ;
vis [ i ] [ j ] = 1 ; if ( grid [ i ] [ j ] == 0 )
z ++ ;
if ( grid [ i ] [ j ] == 2 ) {
if ( z == z_count ) ans ++ ; vis [ i ] [ j ] = 0 ; return ; }
if ( i >= 1 && ! vis [ i - 1 ] [ j ] && grid [ i - 1 ] [ j ] != -1 ) dfs ( i - 1 , j , grid , vis , ans , z , z_count ) ;
if ( i < n - 1 && ! vis [ i + 1 ] [ j ] && grid [ i + 1 ] [ j ] != -1 ) dfs ( i + 1 , j , grid , vis , ans , z , z_count ) ;
if ( j >= 1 && ! vis [ i ] [ j - 1 ] && grid [ i ] [ j - 1 ] != -1 ) dfs ( i , j - 1 , grid , vis , ans , z , z_count ) ;
if ( j < m - 1 && ! vis [ i ] [ j + 1 ] && grid [ i ] [ j + 1 ] != -1 ) dfs ( i , j + 1 , grid , vis , ans , z , z_count ) ;
vis [ i ] [ j ] = 0 ; }
int uniquePaths ( vector < vector < int > > & grid ) {
int n = grid . size ( ) , m = grid [ 0 ] . size ( ) ; int ans = 0 ; vector < vector < bool > > vis ( n , vector < bool > ( m , 0 ) ) ; int x , y ; for ( int i = 0 ; i < n ; ++ i ) { for ( int j = 0 ; j < m ; ++ j ) {
if ( grid [ i ] [ j ] == 0 ) z_count ++ ; else if ( grid [ i ] [ j ] == 1 ) {
x = i , y = j ; } } } dfs ( x , y , grid , vis , ans , 0 , z_count ) ; return ans ; }
int main ( ) { vector < vector < int > > grid { { 1 , 0 , 0 , 0 } , { 0 , 0 , 0 , 0 } , { 0 , 0 , 2 , -1 } } ; cout << uniquePaths ( grid ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int numPairs ( int a [ ] , int n ) { int ans , i , index ;
ans = 0 ;
for ( i = 0 ; i < n ; i ++ ) a [ i ] = abs ( a [ i ] ) ;
sort ( a , a + n ) ;
for ( i = 0 ; i < n ; i ++ ) { index = upper_bound ( a , a + n , 2 * a [ i ] ) - a ; ans += index - i - 1 ; }
return ans ; }
int main ( ) { int a [ ] = { 3 , 6 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << numPairs ( a , n ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int areaOfSquare ( int S ) {
int area = S * S ; return area ; }
int main ( ) {
int S = 5 ;
cout << areaOfSquare ( S ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int maxPointOfIntersection ( int x , int y ) { int k = y * ( y - 1 ) / 2 ; k = k + x * ( 2 * y + x - 1 ) ; return k ; }
int main ( ) {
int x = 3 ;
int y = 4 ;
cout << ( maxPointOfIntersection ( x , y ) ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Icosihenagonal_num ( int n ) {
return ( 19 * n * n - 17 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << Icosihenagonal_num ( n ) << endl ; n = 10 ; cout << Icosihenagonal_num ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; pair < double , double > find_Centroid ( vector < pair < double , double > > & v ) { pair < double , double > ans = { 0 , 0 } ; int n = v . size ( ) ; double signedArea = 0 ;
for ( int i = 0 ; i < v . size ( ) ; i ++ ) { double x0 = v [ i ] . first , y0 = v [ i ] . second ; double x1 = v [ ( i + 1 ) % n ] . first , y1 = v [ ( i + 1 ) % n ] . second ;
double A = ( x0 * y1 ) - ( x1 * y0 ) ; signedArea += A ;
ans . first += ( x0 + x1 ) * A ; ans . second += ( y0 + y1 ) * A ; } signedArea *= 0.5 ; ans . first = ( ans . first ) / ( 6 * signedArea ) ; ans . second = ( ans . second ) / ( 6 * signedArea ) ; return ans ; }
int main ( ) {
vector < pair < double , double > > vp = { { 1 , 2 } , { 3 , -4 } , { 6 , -7 } } ; pair < double , double > ans = find_Centroid ( vp ) ; cout << setprecision ( 12 ) << ans . first << " ▁ " << ans . second << ' ' ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define ll  long long int NEW_LINE using namespace std ;
int main ( ) { int d = 10 ; double a ;
a = ( double ) ( 360 - ( 6 * d ) ) / 4 ;
cout << a << " , ▁ " << a + d << " , ▁ " << a + ( 2 * d ) << " , ▁ " << a + ( 3 * d ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = fabs ( ( c2 * z1 + d2 ) ) / ( sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; cout << " Perpendicular ▁ distance ▁ is ▁ " << d << endl ; } else cout << " Planes ▁ are ▁ not ▁ parallel " ; return ; }
int main ( ) { float a1 = 1 ; float b1 = 2 ; float c1 = -1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = -3 ; float d2 = -4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int factorial ( int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
long long numOfNecklace ( int N ) {
long long ans = factorial ( N ) / ( factorial ( N / 2 ) * factorial ( N / 2 ) ) ;
ans = ans * factorial ( N / 2 - 1 ) ; ans = ans * factorial ( N / 2 - 1 ) ;
ans /= 2 ;
return ans ; }
int main ( ) {
int N = 4 ;
cout << numOfNecklace ( N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string isDivisibleByDivisor ( int S , int D ) {
S %= D ;
unordered_set < int > hashMap ; hashMap . insert ( S ) ; for ( int i = 0 ; i <= D ; i ++ ) {
S += ( S % D ) ; S %= D ;
if ( hashMap . find ( S ) != hashMap . end ( ) ) {
if ( S == 0 ) { return " Yes " ; } return " No " ; }
else hashMap . insert ( S ) ; } return " Yes " ; }
int main ( ) { int S = 3 , D = 6 ; cout << isDivisibleByDivisor ( S , D ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minimumSteps ( int x , int y ) {
int cnt = 0 ;
while ( x != 0 && y != 0 ) {
if ( x > y ) {
cnt += x / y ; x %= y ; }
else {
cnt += y / x ; y %= x ; } } cnt -- ;
if ( x > 1 y > 1 ) cnt = -1 ;
cout << cnt ; }
int main ( ) {
int x = 3 , y = 1 ; minimumSteps ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool check ( int A [ ] , int N ) {
stack < int > S ;
int B_end = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( ! S . empty ( ) ) {
int top = S . top ( ) ;
while ( top == B_end + 1 ) {
B_end = B_end + 1 ;
S . pop ( ) ;
if ( S . empty ( ) ) { break ; }
top = S . top ( ) ; }
if ( S . empty ( ) ) { S . push ( A [ i ] ) ; } else { top = S . top ( ) ;
if ( A [ i ] < top ) { S . push ( A [ i ] ) ; }
else {
return false ; } } } else {
S . push ( A [ i ] ) ; } }
return true ; }
int main ( ) { int A [ ] = { 4 , 1 , 2 , 3 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; check ( A , N ) ? cout << " YES " : cout << " NO " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countMinReversals ( string expr ) { int len = expr . length ( ) ;
if ( len % 2 ) return -1 ;
stack < char > s ; for ( int i = 0 ; i < len ; i ++ ) { if ( expr [ i ] == ' } ' && ! s . empty ( ) ) { if ( s . top ( ) == ' { ' ) s . pop ( ) ; else s . push ( expr [ i ] ) ; } else s . push ( expr [ i ] ) ; }
int red_len = s . size ( ) ;
int n = 0 ; while ( ! s . empty ( ) && s . top ( ) == ' { ' ) { s . pop ( ) ; n ++ ; }
return ( red_len / 2 + n % 2 ) ; }
int main ( ) { string expr = " } } { { " ; cout << countMinReversals ( expr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countMinReversals ( string expr ) { int len = expr . length ( ) ;
if ( len % 2 != 0 ) { return -1 ; } int left_brace = 0 , right_brace = 0 ; int ans ; for ( int i = 0 ; i < len ; i ++ ) {
if ( expr [ i ] == ' { ' ) { left_brace ++ ; }
else { if ( left_brace == 0 ) { right_brace ++ ; } else { left_brace -- ; } } } ans = ceil ( left_brace / 2 ) + ceil ( right_brace / 2 ) ; return ans ; }
int main ( ) { string expr = " } } { { " ; cout << countMinReversals ( expr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void PrintMinNumberForPattern ( string arr ) {
int curr_max = 0 ;
int last_entry = 0 ; int j ;
for ( int i = 0 ; i < arr . length ( ) ; i ++ ) {
int noOfNextD = 0 ; switch ( arr [ i ] ) { case ' I ' :
j = i + 1 ; while ( arr [ j ] == ' D ' && j < arr . length ( ) ) { noOfNextD ++ ; j ++ ; } if ( i == 0 ) { curr_max = noOfNextD + 2 ;
cout << " ▁ " << ++ last_entry ; cout << " ▁ " << curr_max ;
last_entry = curr_max ; } else {
curr_max = curr_max + noOfNextD + 1 ;
last_entry = curr_max ; cout << " ▁ " << last_entry ; }
for ( int k = 0 ; k < noOfNextD ; k ++ ) { cout << " ▁ " << -- last_entry ; i ++ ; } break ;
case ' D ' : if ( i == 0 ) {
j = i + 1 ; while ( arr [ j ] == ' D ' && j < arr . length ( ) ) { noOfNextD ++ ; j ++ ; }
curr_max = noOfNextD + 2 ;
cout << " ▁ " << curr_max << " ▁ " << curr_max - 1 ;
last_entry = curr_max - 1 ; } else {
cout << " ▁ " << last_entry - 1 ; last_entry -- ; } break ; } } cout << endl ; }
int main ( ) { PrintMinNumberForPattern ( " IDID " ) ; PrintMinNumberForPattern ( " I " ) ; PrintMinNumberForPattern ( " DD " ) ; PrintMinNumberForPattern ( " II " ) ; PrintMinNumberForPattern ( " DIDI " ) ; PrintMinNumberForPattern ( " IIDDD " ) ; PrintMinNumberForPattern ( " DDIDDIID " ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printLeast ( string arr ) {
int min_avail = 1 , pos_of_I = 0 ;
vector < int > v ;
if ( arr [ 0 ] == ' I ' ) { v . push_back ( 1 ) ; v . push_back ( 2 ) ; min_avail = 3 ; pos_of_I = 1 ; } else { v . push_back ( 2 ) ; v . push_back ( 1 ) ; min_avail = 3 ; pos_of_I = 0 ; }
for ( int i = 1 ; i < arr . length ( ) ; i ++ ) { if ( arr [ i ] == ' I ' ) { v . push_back ( min_avail ) ; min_avail ++ ; pos_of_I = i + 1 ; } else { v . push_back ( v [ i ] ) ; for ( int j = pos_of_I ; j <= i ; j ++ ) v [ j ] ++ ; min_avail ++ ; } }
for ( int i = 0 ; i < v . size ( ) ; i ++ ) cout << v [ i ] << " ▁ " ; cout << endl ; }
int main ( ) { printLeast ( " IDID " ) ; printLeast ( " I " ) ; printLeast ( " DD " ) ; printLeast ( " II " ) ; printLeast ( " DIDI " ) ; printLeast ( " IIDDD " ) ; printLeast ( " DDIDDIID " ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void PrintMinNumberForPattern ( string seq ) {
string result ;
stack < int > stk ;
for ( int i = 0 ; i <= seq . length ( ) ; i ++ ) {
stk . push ( i + 1 ) ;
if ( i == seq . length ( ) seq [ i ] == ' I ' ) {
while ( ! stk . empty ( ) ) {
result += to_string ( stk . top ( ) ) ; result += " ▁ " ; stk . pop ( ) ; } } } cout << result << endl ; }
int main ( ) { PrintMinNumberForPattern ( " IDID " ) ; PrintMinNumberForPattern ( " I " ) ; PrintMinNumberForPattern ( " DD " ) ; PrintMinNumberForPattern ( " II " ) ; PrintMinNumberForPattern ( " DIDI " ) ; PrintMinNumberForPattern ( " IIDDD " ) ; PrintMinNumberForPattern ( " DDIDDIID " ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string getMinNumberForPattern ( string seq ) { int n = seq . length ( ) ; if ( n >= 9 ) return " - 1" ; string result ( n + 1 , ' ▁ ' ) ; int count = 1 ;
for ( int i = 0 ; i <= n ; i ++ ) { if ( i == n seq [ i ] == ' I ' ) { for ( int j = i - 1 ; j >= -1 ; j -- ) { result [ j + 1 ] = '0' + count ++ ; if ( j >= 0 && seq [ j ] == ' I ' ) break ; } } } return result ; }
int main ( ) { string inputs [ ] = { " IDID " , " I " , " DD " , " II " , " DIDI " , " IIDDD " , " DDIDDIID " } ; for ( string input : inputs ) { cout << getMinNumberForPattern ( input ) << " STRNEWLINE " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int isPrime ( int n ) { int i , c = 0 ; for ( i = 1 ; i < n / 2 ; i ++ ) { if ( n % i == 0 ) c ++ ; } if ( c == 1 ) return 1 ; else return 0 ; }
void findMinNum ( int arr [ ] , int n ) {
int first = 0 , last = 0 , num , rev , i ; int hash [ 10 ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) { hash [ arr [ i ] ] ++ ; }
cout << " Minimum ▁ number : ▁ " ; for ( int i = 0 ; i <= 9 ; i ++ ) {
for ( int j = 0 ; j < hash [ i ] ; j ++ ) cout << i ; } cout << endl ;
for ( i = 0 ; i <= 9 ; i ++ ) { if ( hash [ i ] != 0 ) { first = i ; break ; } }
for ( i = 9 ; i >= 0 ; i -- ) { if ( hash [ i ] != 0 ) { last = i ; break ; } } num = first * 10 + last ; rev = last * 10 + first ;
cout << " Prime ▁ combinations : ▁ " ; if ( isPrime ( num ) && isPrime ( rev ) ) cout << num << " ▁ " << rev ; else if ( isPrime ( num ) ) cout << num ; else if ( isPrime ( rev ) ) cout << rev ; else cout << " No ▁ combinations ▁ exist " ; }
int main ( ) { int arr [ ] = { 1 , 2 , 4 , 7 , 8 } ; findMinNum ( arr , 5 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
bool coprime ( int a , int b ) {
return ( gcd ( a , b ) == 1 ) ; }
void possibleTripletInRange ( int L , int R ) { bool flag = false ; int possibleA , possibleB , possibleC ;
for ( int a = L ; a <= R ; a ++ ) { for ( int b = a + 1 ; b <= R ; b ++ ) { for ( int c = b + 1 ; c <= R ; c ++ ) {
if ( coprime ( a , b ) && coprime ( b , c ) && ! coprime ( a , c ) ) { flag = true ; possibleA = a ; possibleB = b ; possibleC = c ; break ; } } } }
if ( flag == true ) { cout << " ( " << possibleA << " , ▁ " << possibleB << " , ▁ " << possibleC << " ) " << " ▁ is ▁ one ▁ such ▁ possible ▁ triplet ▁ between ▁ " << L << " ▁ and ▁ " << R << " STRNEWLINE " ; } else { cout << " No ▁ Such ▁ Triplet ▁ exists ▁ between ▁ " << L << " ▁ and ▁ " << R << " STRNEWLINE " ; } }
int main ( ) { int L , R ;
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool possibleToReach ( int a , int b ) {
int c = cbrt ( a * b ) ;
int re1 = a / c ; int re2 = b / c ;
if ( ( re1 * re1 * re2 == a ) && ( re2 * re2 * re1 == b ) ) return true ; else return false ; }
int main ( ) { int A = 60 , B = 450 ; if ( possibleToReach ( A , B ) ) cout << " yes " ; else cout << " no " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isUndulating ( string n ) {
if ( n . length ( ) <= 2 ) return false ;
for ( int i = 2 ; i < n . length ( ) ; i ++ ) if ( n [ i - 2 ] != n [ i ] ) false ; return true ; }
int main ( ) { string n = "1212121" ; if ( isUndulating ( n ) ) cout << " Yes " ; else cout << " No " ; }
#include <iostream> NEW_LINE using namespace std ;
int Series ( int n ) { int i ; int sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
int main ( ) { int n = 3 ; int res = Series ( n ) ; cout << res << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countLastDigitK ( long long low , long long high , long long K ) { long long mlow = 10 * ceil ( low / 10.0 ) ; long long mhigh = 10 * floor ( high / 10.0 ) ; int count = ( mhigh - mlow ) / 10 ; if ( high % 10 >= K ) count ++ ; if ( low % 10 <= K && ( low % 10 ) ) count ++ ; return count ; }
int main ( ) { int low = 3 , high = 35 , k = 3 ; cout << countLastDigitK ( low , high , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int sum ( int L , int R ) {
int p = R / 6 ;
int q = ( L - 1 ) / 6 ;
int sumR = 3 * ( p * ( p + 1 ) ) ;
int sumL = ( q * ( q + 1 ) ) * 3 ;
return sumR - sumL ; }
int main ( ) { int L = 1 , R = 20 ; cout << sum ( L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string prevNum ( string str ) { int len = str . length ( ) ; int index = -1 ;
for ( int i = len - 2 ; i >= 0 ; i -- ) { if ( str [ i ] > str [ i + 1 ] ) { index = i ; break ; } }
int smallGreatDgt = -1 ; for ( int i = len - 1 ; i > index ; i -- ) { if ( str [ i ] < str [ index ] ) { if ( smallGreatDgt == -1 ) smallGreatDgt = i ; else if ( str [ i ] >= str [ smallGreatDgt ] ) smallGreatDgt = i ; } }
if ( index == -1 ) return " - 1" ;
if ( smallGreatDgt != -1 ) { swap ( str [ index ] , str [ smallGreatDgt ] ) ; return str ; } return " - 1" ; }
int main ( ) { string str = "34125" ; cout << prevNum ( str ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int horner ( int poly [ ] , int n , int x ) {
for ( int i = 1 ; i < n ; i ++ ) result = result * x + poly [ i ] ; return result ; }
int findSign ( int poly [ ] , int n , int x ) { int result = horner ( poly , n , x ) ; if ( result > 0 ) return 1 ; else if ( result < 0 ) return -1 ; return 0 ; }
int main ( ) {
int poly [ ] = { 2 , -6 , 2 , -1 } ; int x = 3 ; int n = sizeof ( poly ) / sizeof ( poly [ 0 ] ) ; cout << " Sign ▁ of ▁ polynomial ▁ is ▁ " << findSign ( poly , n , x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100005
bool isPrime [ MAX ] ;
void sieveOfEratostheneses ( ) { memset ( isPrime , true , sizeof ( isPrime ) ) ; isPrime [ 1 ] = false ; for ( int i = 2 ; i * i < MAX ; i ++ ) { if ( isPrime [ i ] ) { for ( int j = 2 * i ; j < MAX ; j += i ) isPrime [ j ] = false ; } } }
int findPrime ( int n ) { int num = n + 1 ;
while ( num ) {
if ( isPrime [ num ] ) return num ;
num = num + 1 ; } return 0 ; }
int minNumber ( int arr [ ] , int n ) {
sieveOfEratostheneses ( ) ; int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( isPrime [ sum ] ) return 0 ;
int num = findPrime ( sum ) ;
return num - sum ; }
int main ( ) { int arr [ ] = { 2 , 4 , 6 , 8 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minNumber ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long int SubArraySum ( int arr [ ] , int n ) { long int result = 0 , temp = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
temp = 0 ; for ( int j = i ; j < n ; j ++ ) {
temp += arr [ j ] ; result += temp ; } } return result ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Sum ▁ of ▁ SubArray ▁ : ▁ " << SubArraySum ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int highestPowerof2 ( int n ) { int p = ( int ) log2 ( n ) ; return ( int ) pow ( 2 , p ) ; }
int main ( ) { int n = 10 ; cout << highestPowerof2 ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX_CHAR = 26 ; struct Key {
int freq ; char ch ;
void rearrangeString ( string str ) { int n = str . length ( ) ;
int count [ MAX_CHAR ] = { 0 } ; for ( int i = 0 ; i < n ; i ++ ) count [ str [ i ] - ' a ' ] ++ ;
priority_queue < Key > pq ; for ( char c = ' a ' ; c <= ' z ' ; c ++ ) { int val = c - ' a ' ; if ( count [ val ] ) { pq . push ( Key { count [ val ] , c } ) ; } }
str = " " ;
Key prev { -1 , ' # ' } ;
while ( ! pq . empty ( ) ) {
Key k = pq . top ( ) ; pq . pop ( ) ; str = str + k . ch ;
if ( prev . freq > 0 ) pq . push ( prev ) ;
( k . freq ) -- ; prev = k ; }
if ( n != str . length ( ) ) cout << " ▁ Not ▁ valid ▁ String ▁ " << endl ;
else cout < < str << endl ; }
int main ( ) { string str = " bbbaa " ; rearrangeString ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unsigned int aModM ( string s , unsigned int mod ) { unsigned int number = 0 ; for ( unsigned int i = 0 ; i < s . length ( ) ; i ++ ) {
number = ( number * 10 + ( s [ i ] - '0' ) ) ; number %= mod ; } return number ; }
unsigned int ApowBmodM ( string & a , unsigned int b , unsigned int m ) {
unsigned int ans = aModM ( a , m ) ; unsigned int mul = ans ;
for ( unsigned int i = 1 ; i < b ; i ++ ) ans = ( ans * mul ) % m ; return ans ; }
int main ( ) { string a = "987584345091051645734583954832576" ; unsigned int b = 3 , m = 11 ; cout << ApowBmodM ( a , b , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Data { int x , y ; } ;
double interpolate ( Data f [ ] , int xi , int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
double term = f [ i ] . y ; for ( int j = 0 ; j < n ; j ++ ) { if ( j != i ) term = term * ( xi - f [ j ] . x ) / double ( f [ i ] . x - f [ j ] . x ) ; }
result += term ; } return result ; }
int main ( ) {
Data f [ ] = { { 0 , 2 } , { 1 , 3 } , { 2 , 12 } , { 5 , 147 } } ;
cout << " Value ▁ of ▁ f ( 3 ) ▁ is ▁ : ▁ " << interpolate ( f , 3 , 5 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int SieveOfSundaram ( int n ) {
int nNew = ( n - 1 ) / 2 ;
bool marked [ nNew + 1 ] ;
memset ( marked , false , sizeof ( marked ) ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) for ( int j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) cout << 2 << " ▁ " ;
for ( int i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) cout << 2 * i + 1 << " ▁ " ; }
int main ( void ) { int n = 20 ; SieveOfSundaram ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void constructArray ( int A [ ] , int N , int K ) {
int B [ N ] ;
int totalXOR = A [ 0 ] ^ K ;
for ( int i = 0 ; i < N ; i ++ ) B [ i ] = totalXOR ^ A [ i ] ;
for ( int i = 0 ; i < N ; i ++ ) { cout << B [ i ] << " ▁ " ; } }
int main ( ) { int A [ ] = { 13 , 14 , 10 , 6 } , K = 2 ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
constructArray ( A , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int extraElement ( int A [ ] , int B [ ] , int n ) {
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) ans ^= A [ i ] ; for ( int i = 0 ; i < n + 1 ; i ++ ) ans ^= B [ i ] ; return ans ; }
int main ( ) { int A [ ] = { 10 , 15 , 5 } ; int B [ ] = { 10 , 100 , 15 , 5 } ; int n = sizeof ( A ) / sizeof ( int ) ; cout << extraElement ( A , B , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int hammingDistance ( int n1 , int n2 ) { int x = n1 ^ n2 ; int setBits = 0 ; while ( x > 0 ) { setBits += x & 1 ; x >>= 1 ; } return setBits ; }
int main ( ) { int n1 = 9 , n2 = 14 ; cout << hammingDistance ( 9 , 14 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printSubsets ( int n ) { for ( int i = 0 ; i <= n ; i ++ ) if ( ( n & i ) == i ) cout << i << " ▁ " ; }
int main ( ) { int n = 9 ; printSubsets ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int setBitNumber ( int n ) {
int k = ( int ) ( log2 ( n ) ) ;
return 1 << k ; }
int main ( ) { int n = 273 ; cout << setBitNumber ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int subset ( int ar [ ] , int n ) {
int res = 0 ;
sort ( ar , ar + n ) ;
for ( int i = 0 ; i < n ; i ++ ) { int count = 1 ;
for ( ; i < n - 1 ; i ++ ) { if ( ar [ i ] == ar [ i + 1 ] ) count ++ ; else break ; }
res = max ( res , count ) ; } return res ; }
int main ( ) { int arr [ ] = { 5 , 6 , 9 , 3 , 4 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << subset ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int subset ( int arr [ ] , int n ) {
unordered_map < int , int > mp ; for ( int i = 0 ; i < n ; i ++ ) mp [ arr [ i ] ] ++ ;
int res = 0 ; for ( auto x : mp ) res = max ( res , x . second ) ; return res ; }
int main ( ) { int arr [ ] = { 5 , 6 , 9 , 3 , 4 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << subset ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > psquare ;
void calcPsquare ( int N ) { for ( int i = 1 ; i * i <= N ; i ++ ) psquare . push_back ( i * i ) ; }
int countWays ( int index , int target ) {
if ( target == 0 ) return 1 ; if ( index < 0 target < 0 ) return 0 ;
int inc = countWays ( index , target - psquare [ index ] ) ;
int exc = countWays ( index - 1 , target ) ;
return inc + exc ; }
int main ( ) {
int N = 9 ;
calcPsquare ( N ) ;
cout << countWays ( psquare . size ( ) - 1 , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class TreeNode { public : int data , size ; TreeNode * left ; TreeNode * right ; } ;
TreeNode * newNode ( int data ) { TreeNode * Node = new TreeNode ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ;
return ( Node ) ; }
pair < int , int > sumofsubtree ( TreeNode * root ) {
pair < int , int > p = make_pair ( 1 , 0 ) ;
if ( root -> left ) { pair < int , int > ptemp = sumofsubtree ( root -> left ) ; p . second += ptemp . first + ptemp . second ; p . first += ptemp . first ; }
if ( root -> right ) { pair < int , int > ptemp = sumofsubtree ( root -> right ) ; p . second += ptemp . first + ptemp . second ; p . first += ptemp . first ; }
root -> size = p . first ; return p ; }
int sum = 0 ;
void distance ( TreeNode * root , int target , int distancesum , int n ) {
if ( root -> data == target ) { sum = distancesum ; }
if ( root -> left ) {
int tempsum = distancesum - root -> left -> size + ( n - root -> left -> size ) ;
distance ( root -> left , target , tempsum , n ) ; }
if ( root -> right ) {
int tempsum = distancesum - root -> right -> size + ( n - root -> right -> size ) ;
distance ( root -> right , target , tempsum , n ) ; } }
int main ( ) {
TreeNode * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> left = newNode ( 6 ) ; root -> right -> right = newNode ( 7 ) ; root -> left -> left -> left = newNode ( 8 ) ; root -> left -> left -> right = newNode ( 9 ) ; int target = 3 ; pair < int , int > p = sumofsubtree ( root ) ;
int totalnodes = p . first ; distance ( root , target , p . second , totalnodes ) ;
cout << sum << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void rearrangeArray ( int A [ ] , int B [ ] , int N , int K ) {
sort ( B , B + N , greater < int > ( ) ) ; bool flag = true ; for ( int i = 0 ; i < N ; i ++ ) {
if ( A [ i ] + B [ i ] > K ) { flag = false ; break ; } } if ( ! flag ) { cout << " - 1" << endl ; } else {
for ( int i = 0 ; i < N ; i ++ ) { cout << B [ i ] << " ▁ " ; } } }
int main ( ) {
int A [ ] = { 1 , 2 , 3 , 4 , 2 } ; int B [ ] = { 1 , 2 , 3 , 1 , 1 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int K = 5 ; rearrangeArray ( A , B , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  3 NEW_LINE #define M  3
void countRows ( int mat [ M ] [ N ] ) {
int count = 0 ;
int totalSum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < M ; j ++ ) { totalSum += mat [ i ] [ j ] ; } }
for ( int i = 0 ; i < N ; i ++ ) {
int currSum = 0 ;
for ( int j = 0 ; j < M ; j ++ ) { currSum += mat [ i ] [ j ] ; }
if ( currSum > totalSum - currSum )
count ++ ; }
cout << count ; }
int main ( ) {
int mat [ N ] [ M ] = { { 2 , -1 , 5 } , { -3 , 0 , -2 } , { 5 , 1 , 2 } } ;
countRows ( mat ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool areElementsContiguous ( int arr [ ] , int n ) {
int max = * max_element ( arr , arr + n ) ; int min = * min_element ( arr , arr + n ) ; int m = max - min + 1 ;
if ( m > n ) return false ;
bool visited [ m ] ; memset ( visited , false , sizeof ( visited ) ) ;
for ( int i = 0 ; i < n ; i ++ ) visited [ arr [ i ] - min ] = true ;
for ( int i = 0 ; i < m ; i ++ ) if ( visited [ i ] == false ) return false ; return true ; }
int main ( ) { int arr [ ] = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; if ( areElementsContiguous ( arr , n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool areElementsContiguous ( int arr [ ] , int n ) {
unordered_set < int > us ; for ( int i = 0 ; i < n ; i ++ ) us . insert ( arr [ i ] ) ;
int count = 1 ;
int curr_ele = arr [ 0 ] - 1 ;
while ( us . find ( curr_ele ) != us . end ( ) ) {
count ++ ;
curr_ele -- ; }
curr_ele = arr [ 0 ] + 1 ;
while ( us . find ( curr_ele ) != us . end ( ) ) {
count ++ ;
curr_ele ++ ; }
return ( count == ( int ) ( us . size ( ) ) ) ; }
int main ( ) { int arr [ ] = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; if ( areElementsContiguous ( arr , n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void longest ( int a [ ] , int n , int k ) { unordered_map < int , int > freq ; int start = 0 , end = 0 , now = 0 , l = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
freq [ a [ i ] ] ++ ;
if ( freq [ a [ i ] ] == 1 ) now ++ ;
while ( now > k ) {
freq [ a [ l ] ] -- ;
if ( freq [ a [ l ] ] == 0 ) now -- ;
l ++ ; }
if ( i - l + 1 >= end - start + 1 ) end = i , start = l ; }
for ( int i = start ; i <= end ; i ++ ) cout << a [ i ] << " ▁ " ; }
int main ( ) { int a [ ] = { 6 , 5 , 1 , 2 , 3 , 2 , 1 , 4 , 5 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int k = 3 ; longest ( a , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool sortby ( const pair < int , int > & a , const pair < int , int > & b ) { if ( a . first != b . first ) return a . first < b . first ; return ( a . second < b . second ) ; }
bool kOverlap ( vector < pair < int , int > > pairs , int k ) { vector < pair < int , int > > vec ; for ( int i = 0 ; i < pairs . size ( ) ; i ++ ) {
vec . push_back ( { pairs [ i ] . first , -1 } ) ; vec . push_back ( { pairs [ i ] . second , +1 } ) ; }
sort ( vec . begin ( ) , vec . end ( ) ) ;
stack < pair < int , int > > st ; for ( int i = 0 ; i < vec . size ( ) ; i ++ ) {
pair < int , int > cur = vec [ i ] ;
if ( cur . second == -1 ) {
st . push ( cur ) ; }
else {
st . pop ( ) ; }
if ( st . size ( ) >= k ) { return true ; } } return false ; }
int main ( ) { vector < pair < int , int > > pairs ; pairs . push_back ( make_pair ( 1 , 3 ) ) ; pairs . push_back ( make_pair ( 2 , 4 ) ) ; pairs . push_back ( make_pair ( 3 , 5 ) ) ; pairs . push_back ( make_pair ( 7 , 10 ) ) ; int n = pairs . size ( ) , k = 3 ; if ( kOverlap ( pairs , k ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  5
int ptr [ 501 ] ;
void findSmallestRange ( int arr [ ] [ N ] , int n , int k ) { int i , minval , maxval , minrange , minel , maxel , flag , minind ;
for ( i = 0 ; i <= k ; i ++ ) ptr [ i ] = 0 ; minrange = INT_MAX ; while ( 1 ) {
minind = -1 ; minval = INT_MAX ; maxval = INT_MIN ; flag = 0 ;
for ( i = 0 ; i < k ; i ++ ) {
if ( ptr [ i ] == n ) { flag = 1 ; break ; }
if ( ptr [ i ] < n && arr [ i ] [ ptr [ i ] ] < minval ) {
minind = i ; minval = arr [ i ] [ ptr [ i ] ] ; }
if ( ptr [ i ] < n && arr [ i ] [ ptr [ i ] ] > maxval ) { maxval = arr [ i ] [ ptr [ i ] ] ; } }
if ( flag ) break ; ptr [ minind ] ++ ;
if ( ( maxval - minval ) < minrange ) { minel = minval ; maxel = maxval ; minrange = maxel - minel ; } } printf ( " The ▁ smallest ▁ range ▁ is ▁ [ % d , ▁ % d ] STRNEWLINE " , minel , maxel ) ; }
int main ( ) { int arr [ ] [ N ] = { { 4 , 7 , 9 , 12 , 15 } , { 0 , 8 , 10 , 14 , 20 } , { 6 , 12 , 16 , 30 , 50 } } ; int k = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findSmallestRange ( arr , N , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findLargestd ( int S [ ] , int n ) { bool found = false ;
sort ( S , S + n ) ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { for ( int j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( int k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( int l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return INT_MIN ; }
int main ( ) { int S [ ] = { 2 , 3 , 5 , 7 , 12 } ; int n = sizeof ( S ) / sizeof ( S [ 0 ] ) ; int ans = findLargestd ( S , n ) ; if ( ans == INT_MIN ) cout << " No ▁ Solution " << endl ; else cout << " Largest ▁ d ▁ such ▁ that ▁ a ▁ + ▁ b ▁ + ▁ " << " c ▁ = ▁ d ▁ is ▁ " << ans << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findFourElements ( int arr [ ] , int n ) { unordered_map < int , pair < int , int > > mp ;
for ( int i = 0 ; i < n - 1 ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) mp [ arr [ i ] + arr [ j ] ] = { i , j } ;
int d = INT_MIN ; for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) { int abs_diff = abs ( arr [ i ] - arr [ j ] ) ;
if ( mp . find ( abs_diff ) != mp . end ( ) ) {
pair < int , int > p = mp [ abs_diff ] ; if ( p . first != i && p . first != j && p . second != i && p . second != j ) d = max ( d , max ( arr [ i ] , arr [ j ] ) ) ; } } } return d ; }
int main ( ) { int arr [ ] = { 2 , 3 , 5 , 7 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int res = findFourElements ( arr , n ) ; if ( res == INT_MIN ) cout << " No ▁ Solution . " ; else cout << res ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int CountMaximum ( int arr [ ] , int n , int k ) {
sort ( arr , arr + n ) ; int sum = 0 , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
int main ( ) { int arr [ ] = { 30 , 30 , 10 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 50 ;
cout << CountMaximum ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void leftRotatebyOne ( int arr [ ] , int n ) { int temp = arr [ 0 ] , i ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
void leftRotate ( int arr [ ] , int d , int n ) { for ( int i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; leftRotate ( arr , 2 , n ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void partSort ( int arr [ ] , int N , int a , int b ) {
int l = min ( a , b ) ; int r = max ( a , b ) ;
int temp [ r - l + 1 ] ; int j = 0 ; for ( int i = l ; i <= r ; i ++ ) { temp [ j ] = arr [ i ] ; j ++ ; }
sort ( temp , temp + r - l + 1 ) ;
j = 0 ; for ( int i = l ; i <= r ; i ++ ) { arr [ i ] = temp [ j ] ; j ++ ; }
for ( int i = 0 ; i < N ; i ++ ) { cout << arr [ i ] << " ▁ " ; } }
int main ( ) { int arr [ ] = { 7 , 8 , 4 , 5 , 2 } ; int a = 1 ; int b = 4 ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; partSort ( arr , N , a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX_SIZE  10
void sortByRow ( int mat [ ] [ MAX_SIZE ] , int n , bool descending ) { for ( int i = 0 ; i < n ; i ++ ) { if ( descending == true ) sort ( mat [ i ] , mat [ i ] + n , greater < int > ( ) ) ; else sort ( mat [ i ] , mat [ i ] + n ) ; } }
void transpose ( int mat [ ] [ MAX_SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ )
swap ( mat [ i ] [ j ] , mat [ j ] [ i ] ) ; }
void sortMatRowAndColWise ( int mat [ ] [ MAX_SIZE ] , int n ) {
sortByRow ( mat , n , true ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n , false ) ;
transpose ( mat , n ) ; }
void printMat ( int mat [ ] [ MAX_SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) cout << mat [ i ] [ j ] << " ▁ " ; cout << endl ; } }
int main ( ) { int n = 3 ; int mat [ n ] [ MAX_SIZE ] = { { 3 , 2 , 1 } , { 9 , 8 , 7 } , { 6 , 5 , 4 } } ; cout << " Original ▁ Matrix : STRNEWLINE " ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; cout << " Matrix After Sorting : " ; printMat ( mat , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void pushZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
int main ( ) { int arr [ ] = { 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; pushZerosToEnd ( arr , n ) ; cout << " Array ▁ after ▁ pushing ▁ all ▁ zeros ▁ to ▁ end ▁ of ▁ array ▁ : STRNEWLINE " ; for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void moveZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 ) swap ( arr [ count ++ ] , arr [ i ] ) ; }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 0 , 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Original ▁ array : ▁ " ; printArray ( arr , n ) ; moveZerosToEnd ( arr , n ) ; cout << " Modified array : " printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void pushZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
void modifyAndRearrangeArr ( int arr [ ] , int n ) {
if ( n == 1 ) return ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( ( arr [ i ] != 0 ) && ( arr [ i ] == arr [ i + 1 ] ) ) {
arr [ i ] = 2 * arr [ i ] ;
arr [ i + 1 ] = 0 ;
i ++ ; } }
pushZerosToEnd ( arr , n ) ; }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 0 , 2 , 2 , 2 , 0 , 6 , 6 , 0 , 0 , 8 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Original ▁ array : ▁ " ; printArray ( arr , n ) ; modifyAndRearrangeArr ( arr , n ) ; cout << " Modified array : " printArray ( arr , n ) ; return 0 ; }
void swap ( int & a , int & b ) { a = b + a - ( b = a ) ; }
void shiftAllZeroToLeft ( int array [ ] , int n ) {
int lastSeenNonZero = 0 ; for ( index = 0 ; index < n ; index ++ ) {
if ( array [ index ] != 0 ) {
swap ( array [ index ] , array [ lastSeenNonZero ] ) ;
lastSeenNonZero ++ ; } } }
#include <stdio.h>
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
void RearrangePosNeg ( int arr [ ] , int n ) { int key , j ; for ( int i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
int main ( ) { int arr [ ] = { -12 , 11 , -13 , -5 , 6 , -7 , 5 , -3 , -6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printArray ( int A [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) cout << A [ i ] << " ▁ " ; cout << endl ; }
void reverse ( int arr [ ] , int l , int r ) { if ( l < r ) { swap ( arr [ l ] , arr [ r ] ) ; reverse ( arr , ++ l , -- r ) ; } }
void merge ( int arr [ ] , int l , int m , int r ) {
int i = l ;
int j = m + 1 ; while ( i <= m && arr [ i ] < 0 ) i ++ ;
while ( j <= r && arr [ j ] < 0 ) j ++ ;
reverse ( arr , i , m ) ;
reverse ( arr , m + 1 , j - 1 ) ;
reverse ( arr , i , j - 1 ) ; }
void RearrangePosNeg ( int arr [ ] , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
RearrangePosNeg ( arr , l , m ) ; RearrangePosNeg ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } }
int main ( ) { int arr [ ] = { -12 , 11 , -13 , -5 , 6 , -7 , 5 , -3 , -6 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; RearrangePosNeg ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; void RearrangePosNeg ( int arr [ ] , int n ) { int i = 0 ; int j = n - 1 ; while ( true ) {
while ( arr [ i ] < 0 && i < n ) i ++ ;
while ( arr [ j ] > 0 && j >= 0 ) j -- ;
if ( i < j ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } else break ; } }
int main ( ) { int arr [ ] = { -12 , 11 , -13 , -5 , 6 , -7 , 5 , -3 , -6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; RearrangePosNeg ( arr , n ) ; for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void winner ( int arr [ ] , int N ) {
if ( N % 2 == 1 ) { cout << " A " ; }
else { cout << " B " ; } }
int main ( ) {
int arr [ ] = { 24 , 45 , 45 , 24 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; winner ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int sz = 20 ; const int sqr = int ( sqrt ( sz ) ) + 1 ;
void precomputeExpressionForAllVal ( int arr [ ] , int N , int dp [ sz ] [ sqr ] ) {
for ( int i = N - 1 ; i >= 0 ; i -- ) {
for ( int j = 1 ; j <= sqrt ( N ) ; j ++ ) {
if ( i + j < N ) {
dp [ i ] [ j ] = arr [ i ] + dp [ i + j ] [ j ] ; } else {
dp [ i ] [ j ] = arr [ i ] ; } } } }
int querySum ( int arr [ ] , int N , int Q [ ] [ 2 ] , int M ) {
int dp [ sz ] [ sqr ] ; precomputeExpressionForAllVal ( arr , N , dp ) ;
for ( int i = 0 ; i < M ; i ++ ) { int x = Q [ i ] [ 0 ] ; int y = Q [ i ] [ 1 ] ;
if ( y <= sqrt ( N ) ) { cout << dp [ x ] [ y ] << " ▁ " ; continue ; }
int sum = 0 ;
while ( x < N ) {
sum += arr [ x ] ;
x += y ; } cout << sum << " ▁ " ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 7 , 5 , 4 } ; int Q [ ] [ 2 ] = { { 2 , 1 } , { 3 , 2 } } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int M = sizeof ( Q ) / sizeof ( Q [ 0 ] ) ; querySum ( arr , N , Q , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findElements ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) cout << arr [ i ] << " ▁ " ; } }
int main ( ) { int arr [ ] = { 2 , -6 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findElements ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findElements ( int arr [ ] , int n ) { sort ( arr , arr + n ) ; for ( int i = 0 ; i < n - 2 ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 2 , -6 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findElements ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findElements ( int arr [ ] , int n ) { int first = INT_MIN , second = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 2 , -6 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findElements ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getMinOps ( int arr [ ] , int n ) {
int res = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
res += max ( arr [ i + 1 ] - arr [ i ] , 0 ) ; }
return res ; }
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 1 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << getMinOps ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findFirstMissing ( int array [ ] , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
int main ( ) { int arr [ ] = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Smallest ▁ missing ▁ element ▁ is ▁ " << findFirstMissing ( arr , 0 , n - 1 ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findSmallestMissinginSortedArray ( vector < int > arr ) {
if ( arr [ 0 ] != 0 ) return 0 ;
if ( arr [ arr . size ( ) - 1 ] == arr . size ( ) - 1 ) return arr . size ( ) ; int first = arr [ 0 ] ; return findFirstMissing ( arr , 0 , arr . size ( ) - 1 , first ) ; }
int findFirstMissing ( vector < int > arr , int start , int end , int first ) { if ( start < end ) { int mid = ( start + end ) / 2 ;
if ( arr [ mid ] != mid + first ) return findFirstMissing ( arr , start , mid , first ) ; else return findFirstMissing ( arr , mid + 1 , end , first ) ; } return start + first ; }
int main ( ) { vector < int > arr = { 0 , 1 , 2 , 3 , 4 , 5 , 7 } ; int n = arr . size ( ) ;
cout << " First ▁ Missing ▁ element ▁ is ▁ : ▁ " << findSmallestMissinginSortedArray ( arr ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int FindMaxSum ( vector < int > arr , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
int main ( ) { vector < int > arr = { 5 , 5 , 10 , 100 , 10 , 5 } ; cout << FindMaxSum ( arr , arr . size ( ) ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  7
int countChanges ( int matrix [ ] [ N ] , int n , int m ) {
int dist = n + m - 1 ;
int freq [ dist ] [ 10 ] ;
for ( int i = 0 ; i < dist ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) freq [ i ] [ j ] = 0 ; }
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
freq [ i + j ] [ matrix [ i ] [ j ] ] ++ ; } } int min_changes_sum = 0 ; for ( int i = 0 ; i < dist / 2 ; i ++ ) { int maximum = 0 ; int total_values = 0 ;
for ( int j = 0 ; j < 10 ; j ++ ) { maximum = max ( maximum , freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) ; total_values += ( freq [ i ] [ j ] + freq [ n + m - 2 - i ] [ j ] ) ; }
min_changes_sum += ( total_values - maximum ) ; }
return min_changes_sum ; }
int main ( ) {
int mat [ ] [ N ] = { { 1 , 2 } , { 3 , 5 } } ;
cout << countChanges ( mat , 2 , 2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  500
int lookup [ MAX ] [ MAX ] ;
void buildSparseTable ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) lookup [ i ] [ 0 ] = arr [ i ] ;
for ( int j = 1 ; ( 1 << j ) <= n ; j ++ ) {
for ( int i = 0 ; ( i + ( 1 << j ) - 1 ) < n ; i ++ ) {
if ( lookup [ i ] [ j - 1 ] < lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ) lookup [ i ] [ j ] = lookup [ i ] [ j - 1 ] ; else lookup [ i ] [ j ] = lookup [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ; } } }
int query ( int L , int R ) {
int j = ( int ) log2 ( R - L + 1 ) ;
if ( lookup [ L ] [ j ] <= lookup [ R - ( 1 << j ) + 1 ] [ j ] ) return lookup [ L ] [ j ] ; else return lookup [ R - ( 1 << j ) + 1 ] [ j ] ; }
int main ( ) { int a [ ] = { 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; buildSparseTable ( a , n ) ; cout << query ( 0 , 4 ) << endl ; cout << query ( 4 , 7 ) << endl ; cout << query ( 7 , 8 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  500
int table [ MAX ] [ MAX ] ;
void buildSparseTable ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) table [ i ] [ 0 ] = arr [ i ] ;
for ( int j = 1 ; j <= n ; j ++ ) for ( int i = 0 ; i <= n - ( 1 << j ) ; i ++ ) table [ i ] [ j ] = __gcd ( table [ i ] [ j - 1 ] , table [ i + ( 1 << ( j - 1 ) ) ] [ j - 1 ] ) ; }
int query ( int L , int R ) {
int j = ( int ) log2 ( R - L + 1 ) ;
return __gcd ( table [ L ] [ j ] , table [ R - ( 1 << j ) + 1 ] [ j ] ) ; }
int main ( ) { int a [ ] = { 7 , 2 , 3 , 0 , 5 , 10 , 3 , 12 , 18 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; buildSparseTable ( a , n ) ; cout << query ( 0 , 2 ) << endl ; cout << query ( 1 , 3 ) << endl ; cout << query ( 4 , 5 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minimizeWithKSwaps ( int arr [ ] , int n , int k ) { for ( int i = 0 ; i < n - 1 && k > 0 ; ++ i ) {
int pos = i ; for ( int j = i + 1 ; j < n ; ++ j ) {
if ( j - i > k ) break ;
if ( arr [ j ] < arr [ pos ] ) pos = j ; }
for ( int j = pos ; j > i ; -- j ) swap ( arr [ j ] , arr [ j - 1 ] ) ;
k -= pos - i ; } }
int main ( ) { int arr [ ] = { 7 , 6 , 9 , 2 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 3 ;
minimizeWithKSwaps ( arr , n , k ) ;
for ( int i = 0 ; i < n ; ++ i ) cout << arr [ i ] << " ▁ " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMaxAverage ( int arr [ ] , int n , int k ) {
if ( k > n ) return -1 ;
int * csum = new int [ n ] ; csum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) csum [ i ] = csum [ i - 1 ] + arr [ i ] ;
int max_sum = csum [ k - 1 ] , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { int curr_sum = csum [ i ] - csum [ i - k ] ; if ( curr_sum > max_sum ) { max_sum = curr_sum ; max_end = i ; } }
return max_end - k + 1 ; }
int main ( ) { int arr [ ] = { 1 , 12 , -5 , -6 , 50 , 3 } ; int k = 4 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " " length ▁ " << k << " ▁ begins ▁ at ▁ index ▁ " << findMaxAverage ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMaxAverage ( int arr [ ] , int n , int k ) {
if ( k > n ) return -1 ;
int sum = arr [ 0 ] ; for ( int i = 1 ; i < k ; i ++ ) sum += arr [ i ] ; int max_sum = sum , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { int sum = sum + arr [ i ] - arr [ i - k ] ; if ( sum > max_sum ) { max_sum = sum ; max_end = i ; } }
return max_end - k + 1 ; }
int main ( ) { int arr [ ] = { 1 , 12 , -5 , -6 , 50 , 3 } ; int k = 4 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " " length ▁ " << k << " ▁ begins ▁ at ▁ index ▁ " << findMaxAverage ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
map < pair < int , int > , int > m ;
int findMinimum ( int a [ ] , int n , int pos , int myturn ) {
if ( m . find ( { pos , myturn } ) != m . end ( ) ) { return m [ { pos , myturn } ] ; }
if ( pos >= n ) { return 0 ; }
if ( ! myturn ) {
int ans = min ( findMinimum ( a , n , pos + 1 , ! myturn ) + a [ pos ] , findMinimum ( a , n , pos + 2 , ! myturn ) + a [ pos ] + a [ pos + 1 ] ) ;
m [ { pos , myturn } ] = ans ;
return ans ; }
if ( myturn ) {
int ans = min ( findMinimum ( a , n , pos + 1 , ! myturn ) , findMinimum ( a , n , pos + 2 , ! myturn ) ) ;
m [ { pos , myturn } ] = ans ;
return ans ; } return 0 ; }
int countPenality ( int arr [ ] , int N ) {
int pos = 0 ;
int turn = 0 ;
return findMinimum ( arr , N , pos , turn ) ; }
void printAnswer ( int * arr , int N ) {
int a = countPenality ( arr , N ) ;
int sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
cout << a ; }
int main ( ) { int arr [ ] = { 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printAnswer ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int const MAX = 1000001 ; bool prime [ MAX ] ;
void SieveOfEratosthenes ( ) {
memset ( prime , true , sizeof ( prime ) ) ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= MAX ; i += p ) prime [ i ] = false ; } } }
int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
int getSumUtil ( int * st , int ss , int se , int qs , int qe , int si ) {
if ( qs <= ss && qe >= se ) return st [ si ] ;
if ( se < qs ss > qe ) return 0 ;
int mid = getMid ( ss , se ) ; return getSumUtil ( st , ss , mid , qs , qe , 2 * si + 1 ) + getSumUtil ( st , mid + 1 , se , qs , qe , 2 * si + 2 ) ; }
void updateValueUtil ( int * st , int ss , int se , int i , int diff , int si ) {
if ( i < ss i > se ) return ;
st [ si ] = st [ si ] + diff ; if ( se != ss ) { int mid = getMid ( ss , se ) ; updateValueUtil ( st , ss , mid , i , diff , 2 * si + 1 ) ; updateValueUtil ( st , mid + 1 , se , i , diff , 2 * si + 2 ) ; } }
void updateValue ( int arr [ ] , int * st , int n , int i , int new_val ) {
if ( i < 0 i > n - 1 ) { cout << " - 1" ; return ; }
int diff = new_val - arr [ i ] ; int prev_val = arr [ i ] ;
arr [ i ] = new_val ;
if ( prime [ new_val ] prime [ prev_val ] ) {
if ( ! prime [ prev_val ] ) updateValueUtil ( st , 0 , n - 1 , i , new_val , 0 ) ;
else if ( ! prime [ new_val ] ) updateValueUtil ( st , 0 , n - 1 , i , - prev_val , 0 ) ;
else updateValueUtil ( st , 0 , n - 1 , i , diff , 0 ) ; } }
int getSum ( int * st , int n , int qs , int qe ) {
if ( qs < 0 qe > n - 1 qs > qe ) { cout << " - 1" ; return -1 ; } return getSumUtil ( st , 0 , n - 1 , qs , qe , 0 ) ; }
int constructSTUtil ( int arr [ ] , int ss , int se , int * st , int si ) {
if ( ss == se ) {
if ( prime [ arr [ ss ] ] ) st [ si ] = arr [ ss ] ; else st [ si ] = 0 ; return st [ si ] ; }
int mid = getMid ( ss , se ) ; st [ si ] = constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) + constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ; return st [ si ] ; }
int * constructST ( int arr [ ] , int n ) {
int x = ( int ) ( ceil ( log2 ( n ) ) ) ;
int max_size = 2 * ( int ) pow ( 2 , x ) - 1 ;
int * st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
int main ( ) { int arr [ ] = { 1 , 3 , 5 , 7 , 9 , 11 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int Q [ 3 ] [ 3 ] = { { 1 , 1 , 3 } , { 2 , 1 , 10 } , { 1 , 1 , 3 } } ;
SieveOfEratosthenes ( ) ;
int * st = constructST ( arr , n ) ;
cout << getSum ( st , n , 1 , 3 ) << endl ;
updateValue ( arr , st , n , 1 , 10 ) ;
cout << getSum ( st , n , 1 , 3 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int mod = 1000000007 ; int dp [ 1000 ] [ 1000 ] ; int calculate ( int pos , int prev , string s , vector < int > * index ) {
if ( pos == s . length ( ) ) return 1 ;
if ( dp [ pos ] [ prev ] != -1 ) return dp [ pos ] [ prev ] ;
int answer = 0 ; for ( int i = 0 ; i < index . size ( ) ; i ++ ) { if ( index [ i ] > prev ) { answer = ( answer % mod + calculate ( pos + 1 , index [ i ] , s , index ) % mod ) % mod ; } }
return dp [ pos ] [ prev ] = answer ; } int countWays ( vector < string > & a , string s ) { int n = a . size ( ) ;
vector < int > index [ 26 ] ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < a [ i ] . length ( ) ; j ++ ) {
index [ a [ i ] [ j ] - ' a ' ] . push_back ( j + 1 ) ; } }
memset ( dp , -1 , sizeof ( dp ) ) ; return calculate ( 0 , 0 , s , index ) ; }
int main ( ) { vector < string > A ; A . push_back ( " adc " ) ; A . push_back ( " aec " ) ; A . push_back ( " erg " ) ; string S = " ac " ; cout << countWays ( A , S ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100005 NEW_LINE #define MOD  1000000007
int dp [ MAX ] [ 101 ] [ 2 ] ;
int countNum ( int idx , int sum , int tight , vector < int > num , int len , int k ) { if ( len == idx ) { if ( sum == 0 ) return 1 ; else return 0 ; } if ( dp [ idx ] [ sum ] [ tight ] != -1 ) return dp [ idx ] [ sum ] [ tight ] ; int res = 0 , limit ;
if ( tight == 0 ) { limit = num [ idx ] ; }
else { limit = 9 ; } for ( int i = 0 ; i <= limit ; i ++ ) {
int new_tight = tight ; if ( tight == 0 && i < limit ) new_tight = 1 ; res += countNum ( idx + 1 , ( sum + i ) % k , new_tight , num , len , k ) ; res %= MOD ; }
if ( res < 0 ) res += MOD ; return dp [ idx ] [ sum ] [ tight ] = res ; }
vector < int > process ( string s ) { vector < int > num ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { num . push_back ( s [ i ] - '0' ) ; } return num ; }
int main ( ) {
string n = "98765432109876543210" ;
int len = n . length ( ) ; int k = 58 ;
memset ( dp , -1 , sizeof ( dp ) ) ;
vector < int > num = process ( n ) ; cout << countNum ( 0 , 0 , 0 , num , len , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define maxN  31 NEW_LINE #define maxW  31 NEW_LINE using namespace std ;
int dp [ maxN ] [ maxW ] [ maxW ] ;
int maxWeight ( int * arr , int n , int w1_r , int w2_r , int i ) {
if ( i == n ) return 0 ; if ( dp [ i ] [ w1_r ] [ w2_r ] != -1 ) return dp [ i ] [ w1_r ] [ w2_r ] ;
int fill_w1 = 0 , fill_w2 = 0 , fill_none = 0 ; if ( w1_r >= arr [ i ] ) fill_w1 = arr [ i ] + maxWeight ( arr , n , w1_r - arr [ i ] , w2_r , i + 1 ) ; if ( w2_r >= arr [ i ] ) fill_w2 = arr [ i ] + maxWeight ( arr , n , w1_r , w2_r - arr [ i ] , i + 1 ) ; fill_none = maxWeight ( arr , n , w1_r , w2_r , i + 1 ) ;
dp [ i ] [ w1_r ] [ w2_r ] = max ( fill_none , max ( fill_w1 , fill_w2 ) ) ; return dp [ i ] [ w1_r ] [ w2_r ] ; }
int main ( ) {
int arr [ ] = { 8 , 2 , 3 } ;
memset ( dp , -1 , sizeof ( dp ) ) ;
int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int w1 = 10 , w2 = 3 ;
cout << maxWeight ( arr , n , w1 , w2 , 0 ) ; return 0 ; }
#include <iostream> NEW_LINE #include <stack> NEW_LINE using namespace std ; #define n  3
void findPrefixCount ( int p_arr [ ] [ n ] , bool set_bit [ ] [ n ] ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = n - 1 ; j >= 0 ; j -- ) { if ( ! set_bit [ i ] [ j ] ) continue ; if ( j != n - 1 ) p_arr [ i ] [ j ] += p_arr [ i ] [ j + 1 ] ; p_arr [ i ] [ j ] += ( int ) set_bit [ i ] [ j ] ; } } }
int matrixAllOne ( bool set_bit [ ] [ n ] ) {
int p_arr [ n ] [ n ] = { 0 } ; findPrefixCount ( p_arr , set_bit ) ;
int ans = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { int i = n - 1 ;
stack < pair < int , int > > q ;
int to_sum = 0 ; while ( i >= 0 ) { int c = 0 ; while ( q . size ( ) != 0 and q . top ( ) . first > p_arr [ i ] [ j ] ) { to_sum -= ( q . top ( ) . second + 1 ) * ( q . top ( ) . first - p_arr [ i ] [ j ] ) ; c += q . top ( ) . second + 1 ; q . pop ( ) ; } to_sum += p_arr [ i ] [ j ] ; ans += to_sum ; q . push ( { p_arr [ i ] [ j ] , c } ) ; i -- ; } } return ans ; }
int sumAndMatrix ( int arr [ ] [ n ] ) { int sum = 0 ; int mul = 1 ; for ( int i = 0 ; i < 30 ; i ++ ) {
bool set_bit [ n ] [ n ] ; for ( int R = 0 ; R < n ; R ++ ) for ( int C = 0 ; C < n ; C ++ ) set_bit [ R ] [ C ] = ( ( arr [ R ] [ C ] & ( 1 << i ) ) != 0 ) ; sum += ( mul * matrixAllOne ( set_bit ) ) ; mul *= 2 ; } return sum ; }
int main ( ) { int arr [ ] [ n ] = { { 9 , 7 , 4 } , { 8 , 9 , 2 } , { 11 , 11 , 5 } } ; cout << sumAndMatrix ( arr ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int CountWays ( int n ) {
int noOfWays [ 3 ] ; noOfWays [ 0 ] = 1 ; noOfWays [ 1 ] = 1 ; noOfWays [ 2 ] = 1 + 1 ;
for ( int i = 3 ; i < n + 1 ; i ++ ) { noOfWays [ i ] =
noOfWays [ 3 - 1 ]
+ noOfWays [ 3 - 3 ] ;
noOfWays [ 0 ] = noOfWays [ 1 ] ; noOfWays [ 1 ] = noOfWays [ 2 ] ; noOfWays [ 2 ] = noOfWays [ i ] ; } return noOfWays [ n ] ; }
int main ( ) { int n = 5 ; cout << CountWays ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  105 NEW_LINE void sieve ( int prime [ ] ) { for ( int i = 2 ; i * i <= MAX ; i ++ ) { if ( prime [ i ] == 0 ) { for ( int j = i * i ; j <= MAX ; j += i ) prime [ j ] = 1 ; } } }
void dfs ( int i , int j , int k , int * q , int n , int m , int mappedMatrix [ ] [ MAX ] , int mark [ ] [ MAX ] , pair < int , int > ans [ ] ) {
if ( mappedMatrix [ i ] [ j ] == 0 || i > n || j > m || mark [ i ] [ j ] || ( * q ) ) return ;
mark [ i ] [ j ] = 1 ;
ans [ k ] = make_pair ( i , j ) ;
if ( i == n && j == m ) {
( * q ) = k ; return ; }
dfs ( i + 1 , j + 1 , k + 1 , q , n , m , mappedMatrix , mark , ans ) ;
dfs ( i + 1 , j , k + 1 , q , n , m , mappedMatrix , mark , ans ) ;
dfs ( i , j + 1 , k + 1 , q , n , m , mappedMatrix , mark , ans ) ; }
void lexicographicalPath ( int n , int m , int mappedMatrix [ ] [ MAX ] ) {
int q = 0 ;
pair < int , int > ans [ MAX ] ;
int mark [ MAX ] [ MAX ] ;
dfs ( 1 , 1 , 1 , & q , n , m , mappedMatrix , mark , ans ) ;
for ( int i = 1 ; i <= q ; i ++ ) cout << ans [ i ] . first << " ▁ " << ans [ i ] . second << " STRNEWLINE " ; }
void countPrimePath ( int mappedMatrix [ ] [ MAX ] , int n , int m ) { int dp [ MAX ] [ MAX ] = { 0 } ; dp [ 1 ] [ 1 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= m ; j ++ ) {
if ( i == 1 && j == 1 ) continue ; dp [ i ] [ j ] = ( dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] + dp [ i - 1 ] [ j - 1 ] ) ;
if ( mappedMatrix [ i ] [ j ] == 0 ) dp [ i ] [ j ] = 0 ; } } cout << dp [ n ] [ m ] << " STRNEWLINE " ; }
void preprocessMatrix ( int mappedMatrix [ ] [ MAX ] , int a [ ] [ MAX ] , int n , int m ) { int prime [ MAX ] ;
sieve ( prime ) ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
if ( prime [ a [ i ] [ j ] ] == 0 ) mappedMatrix [ i + 1 ] [ j + 1 ] = 1 ;
else mappedMatrix [ i + 1 ] [ j + 1 ] = 0 ; } } }
int main ( ) { int n = 3 ; int m = 3 ; int a [ MAX ] [ MAX ] = { { 2 , 3 , 7 } , { 5 , 4 , 2 } , { 3 , 7 , 11 } } ; int mappedMatrix [ MAX ] [ MAX ] = { 0 } ; preprocessMatrix ( mappedMatrix , a , n , m ) ; countPrimePath ( mappedMatrix , n , m ) ; lexicographicalPath ( n , m , mappedMatrix ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int isSubsetSum ( int set [ ] , int n , int sum ) {
bool subset [ sum + 1 ] [ n + 1 ] ; int count [ sum + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { subset [ 0 ] [ i ] = true ; count [ 0 ] [ i ] = 0 ; }
for ( int i = 1 ; i <= sum ; i ++ ) { subset [ i ] [ 0 ] = false ; count [ i ] [ 0 ] = -1 ; }
for ( int i = 1 ; i <= sum ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) { subset [ i ] [ j ] = subset [ i ] [ j - 1 ] ; count [ i ] [ j ] = count [ i ] [ j - 1 ] ; if ( i >= set [ j - 1 ] ) { subset [ i ] [ j ] = subset [ i ] [ j ] || subset [ i - set [ j - 1 ] ] [ j - 1 ] ; if ( subset [ i ] [ j ] ) count [ i ] [ j ] = max ( count [ i ] [ j - 1 ] , count [ i - set [ j - 1 ] ] [ j - 1 ] + 1 ) ; } } } return count [ sum ] [ n ] ; }
int main ( ) { int set [ ] = { 2 , 3 , 5 , 10 } ; int sum = 20 ; int n = 4 ; cout << isSubsetSum ( set , n , sum ) ; }
#include <bits/stdc++.h> NEW_LINE #define MAX  100 NEW_LINE using namespace std ;
int lcslen = 0 ;
int dp [ MAX ] [ MAX ] ;
int lcs ( string str1 , string str2 , int len1 , int len2 , int i , int j ) { int & ret = dp [ i ] [ j ] ;
if ( i == len1 j == len2 ) return ret = 0 ;
if ( ret != -1 ) return ret ; ret = 0 ;
if ( str1 [ i ] == str2 [ j ] ) ret = 1 + lcs ( str1 , str2 , len1 , len2 , i + 1 , j + 1 ) ; else ret = max ( lcs ( str1 , str2 , len1 , len2 , i + 1 , j ) , lcs ( str1 , str2 , len1 , len2 , i , j + 1 ) ) ; return ret ; }
void printAll ( string str1 , string str2 , int len1 , int len2 , char data [ ] , int indx1 , int indx2 , int currlcs ) {
if ( currlcs == lcslen ) { data [ currlcs ] = ' \0' ; puts ( data ) ; return ; }
if ( indx1 == len1 indx2 == len2 ) return ;
for ( char ch = ' a ' ; ch <= ' z ' ; ch ++ ) {
bool done = false ; for ( int i = indx1 ; i < len1 ; i ++ ) {
if ( ch == str1 [ i ] ) { for ( int j = indx2 ; j < len2 ; j ++ ) {
if ( ch == str2 [ j ] && dp [ i ] [ j ] == lcslen - currlcs ) { data [ currlcs ] = ch ; printAll ( str1 , str2 , len1 , len2 , data , i + 1 , j + 1 , currlcs + 1 ) ; done = true ; break ; } } }
if ( done ) break ; } } }
void prinlAllLCSSorted ( string str1 , string str2 ) {
int len1 = str1 . length ( ) , len2 = str2 . length ( ) ;
memset ( dp , -1 , sizeof ( dp ) ) ; lcslen = lcs ( str1 , str2 , len1 , len2 , 0 , 0 ) ;
char data [ MAX ] ; printAll ( str1 , str2 , len1 , len2 , data , 0 , 0 , 0 ) ; }
int main ( ) { string str1 = " abcabcaa " , str2 = " acbacba " ; prinlAllLCSSorted ( str1 , str2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isMajority ( int arr [ ] , int n , int x ) { int i ;
int last_index = n % 2 ? ( n / 2 + 1 ) : ( n / 2 ) ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + n / 2 ] == x ) return 1 ; } return 0 ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 4 , 4 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 4 ; if ( isMajority ( arr , n , x ) ) cout << x << " ▁ appears ▁ more ▁ than ▁ " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; else cout << x << " ▁ does ▁ not ▁ appear ▁ more ▁ than " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int _binarySearch ( int arr [ ] , int low , int high , int x ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( ( mid == 0 x > arr [ mid - 1 ] ) && ( arr [ mid ] == x ) ) return mid ; else if ( x > arr [ mid ] ) return _binarySearch ( arr , ( mid + 1 ) , high , x ) ; else return _binarySearch ( arr , low , ( mid - 1 ) , x ) ; } return -1 ; }
bool isMajority ( int arr [ ] , int n , int x ) {
int i = _binarySearch ( arr , 0 , n - 1 , x ) ;
if ( i == -1 ) return false ;
if ( ( ( i + n / 2 ) <= ( n - 1 ) ) && arr [ i + n / 2 ] == x ) return true ; else return false ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 3 ; if ( isMajority ( arr , n , x ) ) cout << x << " ▁ appears ▁ more ▁ than ▁ " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; else cout << x << " ▁ does ▁ not ▁ appear ▁ more ▁ than " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; bool isMajorityElement ( int arr [ ] , int n , int key ) { if ( arr [ n / 2 ] == key ) return true ; else return false ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 3 ; if ( isMajorityElement ( arr , n , x ) ) cout << x << " ▁ appears ▁ more ▁ than ▁ " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; else cout << x << " ▁ does ▁ not ▁ appear ▁ more ▁ than " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; return 0 ; }
#include <iostream> NEW_LINE #include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
int cutRod ( int price [ ] , int n ) { int val [ n + 1 ] ; val [ 0 ] = 0 ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { int max_val = INT_MIN ; for ( j = 0 ; j < i ; j ++ ) max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " << cutRod ( arr , size ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPossible ( int target [ ] , int n ) {
int max = 0 ;
int index = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( max < target [ i ] ) { max = target [ i ] ; index = i ; } }
if ( max == 1 ) return true ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( i != index ) {
max -= target [ i ] ;
if ( max <= 0 ) return false ; } }
target [ index ] = max ;
return isPossible ( target , n ) ; }
int main ( ) { int target [ ] = { 9 , 3 , 5 } ;
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int nCr ( int n , int r ) {
int res = 1 ;
if ( r > n - r ) r = n - r ;
for ( int i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
int main ( ) { int n = 3 , m = 2 , k = 2 ; cout << nCr ( n + m , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void Is_possible ( long long int N ) { int C = 0 ; int D = 0 ;
while ( N % 10 == 0 ) { N = N / 10 ; C += 1 ; }
if ( pow ( 2 , ( int ) log2 ( N ) ) == N ) { D = ( int ) log2 ( N ) ;
if ( C >= D ) cout << " YES " ; else cout << " NO " ; } else cout << " NO " ; }
int main ( ) { long long int N = 2000000000000 ; Is_possible ( N ) ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void findNthTerm ( int n ) { cout << n * n - n + 1 << endl ; }
int main ( ) { int N = 4 ; findNthTerm ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int rev ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; }
return rev_num ; }
int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += rev ( i ) ; else result += ( rev ( i ) + rev ( num / i ) ) ; } }
return ( result + 1 ) ; }
bool isAntiPerfect ( int n ) { return divSum ( n ) == n ; }
int main ( ) {
int N = 244 ;
if ( isAntiPerfect ( N ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE using namespace std ;
void printSeries ( int n , int a , int b , int c ) { int d ;
if ( n == 1 ) { cout << a << " ▁ " ; return ; } if ( n == 2 ) { cout << a << " ▁ " << b << " ▁ " ; return ; } cout << a << " ▁ " << b << " ▁ " << c << " ▁ " ; for ( int i = 4 ; i <= n ; i ++ ) { d = a + b + c ; cout << d << " ▁ " ; a = b ; b = c ; c = d ; } }
int main ( ) { int N = 7 , a = 1 , b = 3 ; int c = 4 ;
printSeries ( N , a , b , c ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int diameter ( int n ) {
int L , H , templen ; L = 1 ;
H = 0 ;
if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 2 ; } if ( n == 3 ) { return 3 ; }
while ( L * 2 <= n ) { L *= 2 ; H ++ ; }
if ( n >= L * 2 - 1 ) return 2 * H + 1 ; else if ( n >= L + ( L / 2 ) - 1 ) return 2 * H ; return 2 * H - 1 ; }
int main ( ) { int n = 15 ; cout << diameter ( n ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void compareValues ( int a , int b , int c , int d ) {
double log1 = log10 ( a ) ; double num1 = log1 * b ;
double log2 = log10 ( c ) ; double num2 = log2 * d ;
if ( num1 > num2 ) cout << a << " ^ " << b ; else cout << c << " ^ " << d ; }
int main ( ) { int a = 8 , b = 29 , c = 60 , d = 59 ; compareValues ( a , b , c , d ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100005
vector < int > addPrimes ( ) { int n = MAX ; bool prime [ n + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ; for ( int p = 2 ; p * p <= n ; p ++ ) { if ( prime [ p ] == true ) { for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } vector < int > ans ;
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) ans . push_back ( p ) ; return ans ; }
bool is_prime ( int n ) { return ( n == 3 n == 5 n == 7 ) ; }
int find_Sum ( int n ) {
int sum = 0 ;
vector < int > v = addPrimes ( ) ;
for ( int i = 0 ; i < v . size ( ) and n ; i ++ ) {
int flag = 1 ; int a = v [ i ] ;
while ( a != 0 ) { int d = a % 10 ; a = a / 10 ; if ( is_prime ( d ) ) { flag = 0 ; break ; } }
if ( flag == 1 ) { n -- ; sum = sum + v [ i ] ; } }
return sum ; }
int main ( ) { int n = 7 ;
cout << find_Sum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int primeCount ( int arr [ ] , int n ) {
int max_val = * max_element ( arr , arr + n ) ;
vector < bool > prime ( max_val + 1 , true ) ;
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= max_val ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i <= max_val ; i += p ) prime [ i ] = false ; } }
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( prime [ arr [ i ] ] ) count ++ ; return count ; }
void getPrefixArray ( int arr [ ] , int n , int pre [ ] ) {
pre [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { pre [ i ] = pre [ i - 1 ] + arr [ i ] ; } }
int main ( ) { int arr [ ] = { 1 , 4 , 8 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int pre [ n ] ; getPrefixArray ( arr , n , pre ) ;
cout << primeCount ( pre , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minValue ( int n , int x , int y ) {
float val = ( y * n ) / 100 ;
if ( x >= val ) return 0 ; else return ( ceil ( val ) - x ) ; }
int main ( ) { int n = 10 , x = 2 , y = 40 ; cout << minValue ( n , x , y ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
bool isFactorialPrime ( long n ) {
if ( ! isPrime ( n ) ) return false ; long fact = 1 ; int i = 1 ; while ( fact <= n + 1 ) {
fact = fact * i ;
if ( n + 1 == fact n - 1 == fact ) return true ; i ++ ; }
return false ; }
int main ( ) { int n = 23 ; if ( isFactorialPrime ( n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ll  long int
int main ( ) {
ll n = 5 ;
ll fac1 = 1 ; for ( int i = 2 ; i <= n - 1 ; i ++ ) fac1 = fac1 * i ;
ll fac2 = fac1 * n ;
ll totalWays = fac1 * fac2 ;
cout << totalWays << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  10000 NEW_LINE vector < int > arr ;
void SieveOfEratosthenes ( ) {
bool prime [ MAX ] ; memset ( prime , true , sizeof ( prime ) ) ; for ( int p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
for ( int p = 2 ; p < MAX ; p ++ ) if ( prime [ p ] ) arr . push_back ( p ) ; }
bool isEuclid ( long n ) { long long product = 1 ; int i = 0 ; while ( product < n ) {
product = product * arr [ i ] ; if ( product + 1 == n ) return true ; i ++ ; } return false ; }
int main ( ) {
SieveOfEratosthenes ( ) ;
long n = 31 ;
if ( isEuclid ( n ) ) cout << " YES STRNEWLINE " ; else cout << " NO STRNEWLINE " ;
n = 42 ;
if ( isEuclid ( n ) ) cout << " YES STRNEWLINE " ; else cout << " NO STRNEWLINE " ; return 0 ; }
#include <cmath> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
int nextPerfectCube ( int N ) { int nextN = floor ( cbrt ( N ) ) + 1 ; return nextN * nextN * nextN ; }
int main ( ) { int n = 35 ; cout << nextPerfectCube ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
int SumOfPrimeDivisors ( int n ) { int sum = 0 ;
int root_n = ( int ) sqrt ( n ) ; for ( int i = 1 ; i <= root_n ; i ++ ) { if ( n % i == 0 ) {
if ( i == n / i && isPrime ( i ) ) { sum += i ; } else {
if ( isPrime ( i ) ) { sum += i ; } if ( isPrime ( n / i ) ) { sum += ( n / i ) ; } } } } return sum ; }
int main ( ) { int n = 60 ; cout << " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " << SumOfPrimeDivisors ( n ) << endl ; }
#include <algorithm> NEW_LINE #include <iostream> NEW_LINE using namespace std ; int findpos ( string n ) { int pos = 0 ; for ( int i = 0 ; n [ i ] != ' \0' ; i ++ ) { switch ( n [ i ] ) {
case '2' : pos = pos * 4 + 1 ; break ;
case '3' : pos = pos * 4 + 2 ; break ;
case '5' : pos = pos * 4 + 3 ; break ;
case '7' : pos = pos * 4 + 4 ; break ; } } return pos ; }
int main ( ) { string n = "777" ; cout << findpos ( n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void possibleTripletInRange ( int L , int R ) { bool flag = false ; int possibleA , possibleB , possibleC ; int numbersInRange = ( R - L + 1 ) ;
if ( numbersInRange < 3 ) { flag = false ; }
else if ( numbersInRange > 3 ) { flag = true ;
if ( L % 2 ) { L ++ ; } possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
if ( ! ( L % 2 ) ) { flag = true ; possibleA = L ; possibleB = L + 1 ; possibleC = L + 2 ; } else {
flag = false ; } }
if ( flag == true ) { cout << " ( " << possibleA << " , ▁ " << possibleB << " , ▁ " << possibleC << " ) " << " ▁ is ▁ one ▁ such ▁ possible ▁ triplet ▁ between ▁ " << L << " ▁ and ▁ " << R << " STRNEWLINE " ; } else { cout << " No ▁ Such ▁ Triplet ▁ exists ▁ between ▁ " << L << " ▁ and ▁ " << R << " STRNEWLINE " ; } }
int main ( ) { int L , R ;
L = 2 ; R = 10 ; possibleTripletInRange ( L , R ) ;
L = 23 ; R = 46 ; possibleTripletInRange ( L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define mod  1000000007 NEW_LINE using namespace std ;
long long digitNumber ( long long n ) {
if ( n == 0 ) return 1 ;
if ( n == 1 ) return 9 ;
if ( n % 2 ) {
long long temp = digitNumber ( ( n - 1 ) / 2 ) % mod ; return ( 9 * ( temp * temp ) % mod ) % mod ; } else {
long long temp = digitNumber ( n / 2 ) % mod ; return ( temp * temp ) % mod ; } } int countExcluding ( int n , int d ) {
if ( d == 0 ) return ( 9 * digitNumber ( n - 1 ) ) % mod ; else return ( 8 * digitNumber ( n - 1 ) ) % mod ; }
int main ( ) {
long long d = 9 ; int n = 3 ; cout << countExcluding ( n , d ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
bool isEmirp ( int n ) {
if ( isPrime ( n ) == false ) return false ;
int rev = 0 ; while ( n != 0 ) { int d = n % 10 ; rev = rev * 10 + d ; n /= 10 ; }
return isPrime ( rev ) ; }
int main ( ) {
int n = 13 ; if ( isEmirp ( n ) == true ) cout << " Yes " ; else cout << " No " ; }
#include <iostream> NEW_LINE using namespace std ;
double Convert ( double radian ) { double pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
int main ( ) { double radian = 5.0 ; double degree = Convert ( radian ) ; cout << degree ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int sn ( int n , int an ) { return ( n * ( 1 + an ) ) / 2 ; }
int trace ( int n , int m ) {
int an = 1 + ( n - 1 ) * ( m + 1 ) ;
int rowmajorSum = sn ( n , an ) ;
an = 1 + ( n - 1 ) * ( n + 1 ) ;
int colmajorSum = sn ( n , an ) ; return rowmajorSum + colmajorSum ; }
int main ( ) { int N = 3 , M = 3 ; cout << trace ( N , M ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void max_area ( int n , int m , int k ) { if ( k > ( n + m - 2 ) ) cout << " Not ▁ possible " << endl ; else { int result ;
if ( k < max ( m , n ) - 1 ) { result = max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; }
else { result = max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; }
cout << result << endl ; } }
int main ( ) { int n = 3 , m = 4 , k = 1 ; max_area ( n , m , k ) ; }
#include <iostream> NEW_LINE using namespace std ;
int area_fun ( int side ) { int area = side * side ; return area ; }
int main ( ) { int side = 4 ; int area = area_fun ( side ) ; cout << area ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long int countConsecutive ( long int N ) {
long int count = 0 ; for ( long int L = 1 ; L * ( L + 1 ) < 2 * N ; L ++ ) { double a = ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) ; if ( a - ( int ) a == 0.0 ) count ++ ; } return count ; }
int main ( ) { long int N = 15 ; cout << countConsecutive ( N ) << endl ; N = 10 ; cout << countConsecutive ( N ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isAutomorphic ( int N ) {
int sq = N * N ;
while ( N > 0 ) {
if ( N % 10 != sq % 10 ) return false ;
N /= 10 ; sq /= 10 ; } return true ; }
int main ( ) { int N = 5 ; isAutomorphic ( N ) ? cout << " Automorphic " : cout << " Not ▁ Automorphic " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxPrimefactorNum ( int N ) {
bool arr [ N + 5 ] ; memset ( arr , true , sizeof ( arr ) ) ;
for ( int i = 3 ; i * i <= N ; i += 2 ) { if ( arr [ i ] ) for ( int j = i * i ; j <= N ; j += i ) arr [ j ] = false ; }
vector < int > prime ; prime . push_back ( 2 ) ; for ( int i = 3 ; i <= N ; i += 2 ) if ( arr [ i ] ) prime . push_back ( i ) ;
int i = 0 , ans = 1 ; while ( ans * prime [ i ] <= N && i < prime . size ( ) ) { ans *= prime [ i ] ; i ++ ; } return ans ; }
int main ( ) { int N = 40 ; cout << maxPrimefactorNum ( N ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; unsigned highestPowerof2 ( unsigned x ) {
x |= x >> 1 ; x |= x >> 2 ; x |= x >> 4 ; x |= x >> 8 ; x |= x >> 16 ;
return x ^ ( x >> 1 ) ; }
int main ( ) { int n = 10 ; cout << highestPowerof2 ( n ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; }
int main ( ) { int num = 36 ; cout << divSum ( num ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int power ( int x , int y , int p ) {
while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
void squareRoot ( int n , int p ) { if ( p % 4 != 3 ) { cout << " Invalid ▁ Input " ; return ; }
n = n % p ; int x = power ( n , ( p + 1 ) / 4 , p ) ; if ( ( x * x ) % p == n ) { cout << " Square ▁ root ▁ is ▁ " << x ; return ; }
x = p - x ; if ( ( x * x ) % p == n ) { cout << " Square ▁ root ▁ is ▁ " << x ; return ; }
cout << " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ; }
int main ( ) { int p = 7 ; int n = 2 ; squareRoot ( n , p ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int power ( int x , unsigned int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
bool miillerTest ( int d , int n ) {
int a = 2 + rand ( ) % ( n - 4 ) ;
int x = power ( a , d , n ) ; if ( x == 1 x == n - 1 ) return true ;
while ( d != n - 1 ) { x = ( x * x ) % n ; d *= 2 ; if ( x == 1 ) return false ; if ( x == n - 1 ) return true ; }
return false ; }
bool isPrime ( int n , int k ) {
if ( n <= 1 n == 4 ) return false ; if ( n <= 3 ) return true ;
int d = n - 1 ; while ( d % 2 == 0 ) d /= 2 ;
for ( int i = 0 ; i < k ; i ++ ) if ( ! miillerTest ( d , n ) ) return false ; return true ; }
int main ( ) { int k = 4 ; cout << " All ▁ primes ▁ smaller ▁ than ▁ 100 : ▁ STRNEWLINE " ; for ( int n = 1 ; n < 100 ; n ++ ) if ( isPrime ( n , k ) ) cout << n << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxConsecutiveOnes ( int x ) {
int count = 0 ;
while ( x != 0 ) {
x = ( x & ( x << 1 ) ) ; count ++ ; } return count ; }
int main ( ) { cout << maxConsecutiveOnes ( 14 ) << endl ; cout << maxConsecutiveOnes ( 222 ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int subtract ( int x , int y ) {
while ( y != 0 ) {
int borrow = ( ~ x ) & y ;
x = x ^ y ;
y = borrow << 1 ; } return x ; }
int main ( ) { int x = 29 , y = 13 ; cout << " x ▁ - ▁ y ▁ is ▁ " << subtract ( x , y ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int subtract ( int x , int y ) { if ( y == 0 ) return x ; return subtract ( x ^ y , ( ~ x & y ) << 1 ) ; }
int main ( ) { int x = 29 , y = 13 ; cout << " x ▁ - ▁ y ▁ is ▁ " << subtract ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void addEdge ( vector < int > v [ ] , int x , int y ) { v [ x ] . push_back ( y ) ; v [ y ] . push_back ( x ) ; }
void dfs ( vector < int > tree [ ] , vector < int > & temp , int ancestor [ ] , int u , int parent , int k ) {
temp . push_back ( u ) ;
for ( auto i : tree [ u ] ) { if ( i == parent ) continue ; dfs ( tree , temp , ancestor , i , u , k ) ; } temp . pop_back ( ) ;
if ( temp . size ( ) < k ) { ancestor [ u ] = -1 ; } else {
ancestor [ u ] = temp [ temp . size ( ) - k ] ; } }
void KthAncestor ( int N , int K , int E , int edges [ ] [ 2 ] ) {
vector < int > tree [ N + 1 ] ; for ( int i = 0 ; i < E ; i ++ ) { addEdge ( tree , edges [ i ] [ 0 ] , edges [ i ] [ 1 ] ) ; }
vector < int > temp ;
int ancestor [ N + 1 ] ; dfs ( tree , temp , ancestor , 1 , 0 , K ) ;
for ( int i = 1 ; i <= N ; i ++ ) { cout << ancestor [ i ] << " ▁ " ; } }
int main ( ) {
int N = 9 ; int K = 2 ;
int E = 8 ; int edges [ 8 ] [ 2 ] = { { 1 , 2 } , { 1 , 3 } , { 2 , 4 } , { 2 , 5 } , { 2 , 6 } , { 3 , 7 } , { 3 , 8 } , { 3 , 9 } } ;
KthAncestor ( N , K , E , edges ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void build ( vector < int > & sum , vector < int > & a , int l , int r , int rt ) {
if ( l == r ) { sum [ rt ] = a [ l - 1 ] ; return ; }
int m = ( l + r ) >> 1 ;
build ( sum , a , l , m , rt << 1 ) ; build ( sum , a , m + 1 , r , rt << 1 1 ) ; }
void pushDown ( vector < int > & sum , vector < int > & add , int rt , int ln , int rn ) { if ( add [ rt ] ) { add [ rt << 1 ] += add [ rt ] ; add [ rt << 1 1 ] += add [ rt ] ; sum [ rt << 1 ] += add [ rt ] * ln ; sum [ rt << 1 1 ] += add [ rt ] * rn ; add [ rt ] = 0 ; } }
void update ( vector < int > & sum , vector < int > & add , int L , int R , int C , int l , int r , int rt ) {
if ( L <= l && r <= R ) { sum [ rt ] += C * ( r - l + 1 ) ; add [ rt ] += C ; return ; }
int m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ;
if ( L <= m ) update ( sum , add , L , R , C , l , m , rt << 1 ) ; if ( R > m ) update ( sum , add , L , R , C , m + 1 , r , rt << 1 1 ) ; }
int query ( vector < int > & sum , vector < int > & add , int L , int R , int l , int r , int rt ) {
if ( L <= l && r <= R ) { return sum [ rt ] ; }
int m = ( l + r ) >> 1 ;
pushDown ( sum , add , rt , m - l + 1 , r - m ) ; int ans = 0 ;
if ( L <= m ) ans += query ( sum , add , L , R , l , m , rt << 1 ) ; if ( R > m ) ans += query ( sum , add , L , R , m + 1 , r , rt << 1 1 ) ;
return ans ; }
void sequenceMaintenance ( int n , int q , vector < int > & a , vector < int > & b , int m ) {
sort ( a . begin ( ) , a . end ( ) ) ;
vector < int > sum , add , ans ; sum . assign ( n << 2 , 0 ) ; add . assign ( n << 2 , 0 ) ;
build ( sum , a , 1 , n , 1 ) ;
for ( int i = 0 ; i < q ; i ++ ) { int l = 1 , r = n , pos = -1 ; while ( l <= r ) { int m = ( l + r ) >> 1 ; if ( query ( sum , add , m , m , 1 , n , 1 ) >= b [ i ] ) { r = m - 1 ; pos = m ; } else { l = m + 1 ; } } if ( pos == -1 ) ans . push_back ( 0 ) ; else {
ans . push_back ( n - pos + 1 ) ;
update ( sum , add , pos , n , - m , 1 , n , 1 ) ; } }
for ( int i = 0 ; i < ans . size ( ) ; i ++ ) { cout << ans [ i ] << " ▁ " ; } }
int main ( ) { int N = 4 ; int Q = 3 ; int M = 1 ; vector < int > arr = { 1 , 2 , 3 , 4 } ; vector < int > query = { 4 , 3 , 1 } ;
sequenceMaintenance ( N , Q , arr , query , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool hasCoprimePair ( vector < int > & arr , int n ) {
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
if ( __gcd ( arr [ i ] , arr [ j ] ) == 1 ) { return true ; } } }
return false ; }
int main ( ) { int n = 3 ; vector < int > arr = { 6 , 9 , 15 } ;
if ( hasCoprimePair ( arr , n ) ) { cout << 1 << endl ; }
else { cout << n << endl ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Numberofways ( int n ) { int count = 0 ; for ( int a = 1 ; a < n ; a ++ ) { for ( int b = 1 ; b < n ; b ++ ) { int c = n - ( a + b ) ;
if ( a + b > c && a + c > b && b + c > a ) { count ++ ; } } }
return count ; }
int main ( ) { int n = 15 ; cout << Numberofways ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countPairs ( int N , int arr [ ] ) { int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( i == arr [ arr [ i ] - 1 ] - 1 ) {
count ++ ; } }
cout << ( count / 2 ) << endl ; }
int main ( ) { int arr [ ] = { 2 , 1 , 4 , 3 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; countPairs ( N , arr ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int LongestFibSubseq ( int A [ ] , int n ) {
unordered_set < int > S ( A , A + n ) ; int maxLen = 0 , x , y ; for ( int i = 0 ; i < n ; ++ i ) { for ( int j = i + 1 ; j < n ; ++ j ) { x = A [ j ] ; y = A [ i ] + A [ j ] ; int length = 2 ;
while ( S . find ( y ) != S . end ( ) ) {
int z = x + y ; x = y ; y = z ; maxLen = max ( maxLen , ++ length ) ; } } } return maxLen >= 3 ? maxLen : 0 ; }
int main ( ) { int A [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; cout << LongestFibSubseq ( A , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int CountMaximum ( int arr [ ] , int n , int k ) {
sort ( arr , arr + n ) ; int sum = 0 , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ;
if ( sum > k ) break ;
count ++ ; }
return count ; }
int main ( ) { int arr [ ] = { 30 , 30 , 10 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 50 ;
cout << CountMaximum ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int num_candyTypes ( vector < int > & candies ) {
unordered_set < int > s ;
for ( int i = 0 ; i < candies . size ( ) ; i ++ ) { s . insert ( candies [ i ] ) ; }
return s . size ( ) ; }
void distribute_candies ( vector < int > & candies ) {
int allowed = candies . size ( ) / 2 ;
int types = num_candyTypes ( candies ) ;
if ( types < allowed ) cout << types ; else cout << allowed ; }
int main ( ) {
vector < int > candies = { 4 , 4 , 5 , 5 , 3 , 3 } ;
distribute_candies ( candies ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double Length_Diagonals ( int a , double theta ) { double p = a * sqrt ( 2 + ( 2 * cos ( theta * ( 3.141 / 180 ) ) ) ) ; double q = a * sqrt ( 2 - ( 2 * cos ( theta * ( 3.141 / 180 ) ) ) ) ; cout << fixed << setprecision ( 2 ) << p << " ▁ " << q ; }
int main ( ) { int a = 6 ; int theta = 45 ; Length_Diagonals ( a , theta ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countEvenOdd ( int arr [ ] , int n , int K ) { int even = 0 , odd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } int y ;
y = __builtin_popcount ( K ) ;
if ( y & 1 ) { cout << " Even ▁ = ▁ " << odd << " , ▁ Odd ▁ = ▁ " << even ; }
else { cout << " Even ▁ = ▁ " << even << " , ▁ Odd ▁ = ▁ " << odd ; } }
int main ( void ) { int arr [ ] = { 4 , 2 , 15 , 9 , 8 , 8 } ; int K = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
countEvenOdd ( arr , n , K ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int main ( ) { int N = 6 ; int Even = N / 2 ; int Odd = N - Even ; cout << Even * Odd ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int longestSubSequence ( pair < int , int > A [ ] , int N , int ind = 0 , int lastf = INT_MIN , int lasts = INT_MAX ) {
if ( ind == N ) return 0 ;
int ans = longestSubSequence ( A , N , ind + 1 , lastf , lasts ) ;
if ( A [ ind ] . first > lastf && A [ ind ] . second < lasts ) ans = max ( ans , longestSubSequence ( A , N , ind + 1 , A [ ind ] . first , A [ ind ] . second ) + 1 ) ; return ans ; }
int main ( ) {
pair < int , int > A [ ] = { { 1 , 2 } , { 2 , 2 } , { 3 , 1 } } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
cout << longestSubSequence ( A , N ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
int countTriplets ( vector < int > & A ) {
int cnt = 0 ;
unordered_map < int , int > tuples ;
for ( auto a : A )
for ( auto b : A ) ++ tuples [ a & b ] ;
for ( auto a : A )
for ( auto t : tuples )
if ( ( t . first & a ) == 0 ) cnt += t . second ;
return cnt ; }
int main ( ) {
vector < int > A = { 2 , 1 , 3 } ;
cout << countTriplets ( A ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int CountWays ( int n ) {
int noOfWays [ n + 3 ] ; noOfWays [ 0 ] = 1 ; noOfWays [ 1 ] = 1 ; noOfWays [ 2 ] = 1 + 1 ;
for ( int i = 3 ; i < n + 1 ; i ++ ) {
noOfWays [ i ] = noOfWays [ i - 1 ] + noOfWays [ i - 3 ] ; } return noOfWays [ n ] ; }
int main ( ) { int n = 0 ; cout << CountWays ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; void printSpiral ( int size ) {
int row = 0 , col = 0 ; int boundary = size - 1 ; int sizeLeft = size - 1 ; int flag = 1 ;
char move = ' r ' ;
int matrix [ size ] [ size ] = { 0 } ; for ( int i = 1 ; i < size * size + 1 ; i ++ ) {
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
for ( row = 0 ; row < size ; row ++ ) { for ( col = 0 ; col < size ; col ++ ) { int n = matrix [ row ] [ col ] ; if ( n < 10 ) cout << n << " ▁ " ; else cout << n << " ▁ " ; } cout << endl ; } }
int main ( ) {
int size = 5 ;
printSpiral ( size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findWinner ( string a , int n ) {
vector < int > v ;
int c = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == '0' ) { c ++ ; }
else { if ( c != 0 ) v . push_back ( c ) ; c = 0 ; } } if ( c != 0 ) v . push_back ( c ) ;
if ( v . size ( ) == 0 ) { cout << " Player ▁ B " ; return ; }
if ( v . size ( ) == 1 ) { if ( v [ 0 ] & 1 ) cout << " Player ▁ A " ;
else cout < < " Player ▁ B " ; return ; }
int first = INT_MIN ; int second = INT_MIN ;
for ( int i = 0 ; i < v . size ( ) ; i ++ ) {
if ( a [ i ] > first ) { second = first ; first = a [ i ] ; }
else if ( a [ i ] > second && a [ i ] != first ) second = a [ i ] ; }
if ( ( first & 1 ) && ( first + 1 ) / 2 > second ) cout << " Player ▁ A " ; else cout << " Player ▁ B " ; }
int main ( ) { string S = "1100011" ; int N = S . length ( ) ; findWinner ( S , N ) ; return 0 ; }
#include <iostream> NEW_LINE #include <map> NEW_LINE using namespace std ;
bool can_Construct ( string S , int K ) {
map < int , int > m ; int i = 0 , j = 0 , p = 0 ;
if ( S . length ( ) == K ) { return true ; }
for ( i = 0 ; i < S . length ( ) ; i ++ ) { m [ S [ i ] ] = m [ S [ i ] ] + 1 ; }
if ( K > S . length ( ) ) { return false ; } else {
for ( h = m . begin ( ) ; h != m . end ( ) ; h ++ ) { if ( m [ h -> first ] % 2 != 0 ) { p = p + 1 ; } } }
if ( K < p ) { return false ; } return true ; }
int main ( ) { string S = " annabelle " ; int K = 4 ; if ( can_Construct ( S , K ) ) { cout << " Yes " ; } else { cout << " No " ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool equalIgnoreCase ( string str1 , string str2 ) { int i = 0 ;
transform ( str1 . begin ( ) , str1 . end ( ) , str1 . begin ( ) , :: tolower ) ; transform ( str2 . begin ( ) , str2 . end ( ) , str2 . begin ( ) , :: tolower ) ;
int x = str1 . compare ( str2 ) ;
if ( x != 0 ) return false ; else return true ; }
void equalIgnoreCaseUtil ( string str1 , string str2 ) { bool res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) cout << " Same " << endl ; else cout << " Not ▁ Same " << endl ; }
int main ( ) { string str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void steps ( string str , int n ) {
bool flag ; int x = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
if ( x == 0 ) flag = true ;
if ( x == n - 1 ) flag = false ;
for ( int j = 0 ; j < x ; j ++ ) cout << " * " ; cout << str [ i ] << " STRNEWLINE " ;
if ( flag == true ) x ++ ; else x -- ; } }
int main ( ) {
int n = 4 ; string str = " GeeksForGeeks " ; cout << " String : ▁ " << str << endl ; cout << " Max ▁ Length ▁ of ▁ Steps : ▁ " << n << endl ;
steps ( str , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void countFreq ( int arr [ ] , int n ) {
vector < int > visited ( n , false ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( visited [ i ] == true ) continue ;
int count = 1 ; for ( int j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) { visited [ j ] = true ; count ++ ; } } cout << arr [ i ] << " ▁ " << count << endl ; } }
int main ( ) { int arr [ ] = { 10 , 20 , 20 , 10 , 10 , 20 , 5 , 20 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; countFreq ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isDivisible ( char str [ ] , int k ) { int n = strlen ( str ) ; int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) if ( str [ n - i - 1 ] == '0' ) c ++ ;
return ( c == k ) ; }
int main ( ) {
char str1 [ ] = "10101100" ; int k = 2 ; if ( isDivisible ( str1 , k ) ) cout << " Yes " << endl ; else cout << " No " << " STRNEWLINE " ;
char str2 [ ] = "111010100" ; k = 2 ; if ( isDivisible ( str2 , k ) ) cout << " Yes " << endl ; else cout << " No " << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #define NO_OF_CHARS  256
bool canFormPalindrome ( string str ) {
int count [ NO_OF_CHARS ] = { 0 } ;
for ( int i = 0 ; str [ i ] ; i ++ ) count [ str [ i ] ] ++ ;
int odd = 0 ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) { if ( count [ i ] & 1 ) odd ++ ; if ( odd > 1 ) return false ; }
return true ; }
int main ( ) { canFormPalindrome ( " geeksforgeeks " ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; canFormPalindrome ( " geeksogeeks " ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isNumber ( string s ) { for ( int i = 0 ; i < s . length ( ) ; i ++ ) if ( isdigit ( s [ i ] ) == false ) return false ; return true ; }
int main ( ) {
string str = "6790" ;
if ( isNumber ( str ) ) cout << " Integer " ;
else cout < < " String " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void reverse ( string str ) { if ( str . size ( ) == 0 ) { return ; } reverse ( str . substr ( 1 ) ) ; cout << str [ 0 ] ; }
int main ( ) { string a = " Geeks ▁ for ▁ Geeks " ; reverse ( a ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
static int box1 = 0 ;
static int box2 = 0 ; static int fact [ 11 ] ;
double getProbability ( int balls [ ] , int M ) {
factorial ( 10 ) ;
box2 = M ;
int K = 0 ;
for ( int i = 0 ; i < M ; i ++ ) K += balls [ i ] ;
if ( K % 2 == 1 ) return 0 ;
long all = comb ( K , K / 2 ) ;
long validPermutation = validPermutations ( K / 2 , balls , 0 , 0 , M ) ;
return ( double ) validPermutation / all ; }
long validPermutations ( int n , int balls [ ] , int usedBalls , int i , int M ) {
if ( usedBalls == n ) {
return box1 == box2 ? 1 : 0 ; }
if ( i >= M ) return 0 ;
long res = validPermutations ( n , balls , usedBalls , i + 1 , M ) ;
box1 ++ ;
for ( int j = 1 ; j <= balls [ i ] ; j ++ ) {
if ( j == balls [ i ] ) box2 -- ;
long combinations = comb ( balls [ i ] , j ) ;
res += combinations * validPermutations ( n , balls , usedBalls + j , i + 1 , M ) ; }
box1 -- ;
box2 ++ ; return res ; }
void factorial ( int N ) {
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ; }
long comb ( int n , int r ) { long res = fact [ n ] / fact [ r ] ; res /= fact [ n - r ] ; return res ; }
int main ( ) { int arr [ ] = { 2 , 1 , 1 } ; int N = 4 ; int M = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << ( getProbability ( arr , M ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float polyarea ( float n , float r ) {
if ( r < 0 && n < 0 ) return -1 ;
float A = ( ( r * r * n ) * sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
int main ( ) { float r = 9 , n = 6 ; cout << polyarea ( n , r ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void is_partition_possible ( int n , int x [ ] , int y [ ] , int w [ ] ) { map < int , int > weight_at_x ; int max_x = -2e3 , min_x = 2e3 ;
for ( int i = 0 ; i < n ; i ++ ) { int new_x = x [ i ] - y [ i ] ; max_x = max ( max_x , new_x ) ; min_x = min ( min_x , new_x ) ;
weight_at_x [ new_x ] += w [ i ] ; } vector < int > sum_till ; sum_till . push_back ( 0 ) ;
for ( int x = min_x ; x <= max_x ; x ++ ) { sum_till . push_back ( sum_till . back ( ) + weight_at_x [ x ] ) ; } int total_sum = sum_till . back ( ) ; int partition_possible = false ; for ( int i = 1 ; i < sum_till . size ( ) ; i ++ ) { if ( sum_till [ i ] == total_sum - sum_till [ i ] ) partition_possible = true ;
if ( sum_till [ i - 1 ] == total_sum - sum_till [ i ] ) partition_possible = true ; } printf ( partition_possible ? " YES STRNEWLINE " : " NO STRNEWLINE " ) ; }
int main ( ) { int n = 3 ; int x [ ] = { -1 , -2 , 1 } ; int y [ ] = { 1 , 1 , -1 } ; int w [ ] = { 3 , 1 , 4 } ; is_partition_possible ( n , x , y , w ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double findPCSlope ( double m ) { return -1.0 / m ; }
int main ( ) { double m = 2.0 ; cout << findPCSlope ( m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; float pi = 3.14159 ;
float area_of_segment ( float radius , float angle ) {
float area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
float area_of_triangle = ( float ) 1 / 2 * ( radius * radius ) * sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
int main ( ) { float radius = 10.0 , angle = 90.0 ; cout << " Area ▁ of ▁ minor ▁ segment ▁ = ▁ " << area_of_segment ( radius , angle ) << endl ; cout << " Area ▁ of ▁ major ▁ segment ▁ = ▁ " << area_of_segment ( radius , ( 360 - angle ) ) ; }
#include <iostream> NEW_LINE using namespace std ; void SectorArea ( double radius , double angle ) { if ( angle >= 360 ) cout << " Angle ▁ not ▁ possible " ;
else { double sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; cout << sector ; } }
int main ( ) { double radius = 9 ; double angle = 60 ; SectorArea ( radius , angle ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unordered_map < int , int > PrimeFactor ( int N ) { unordered_map < int , int > primef ;
while ( N % 2 == 0 ) { if ( primef . count ( 2 ) ) { primef [ 2 ] += 1 ; } else { primef [ 2 ] = 1 ; }
N /= 2 ; }
for ( int i = 3 ; i <= sqrt ( N ) ; i ++ ) {
while ( N % i == 0 ) { if ( primef . count ( i ) ) { primef [ i ] += 1 ; } else { primef [ i ] = 1 ; }
N /= 2 ; } } if ( N > 2 ) { primef [ N ] = 1 ; } return primef ; }
int CountToMakeEqual ( int X , int Y ) {
int gcdofXY = __gcd ( X , Y ) ;
int newX = Y / gcdofXY ; int newY = X / gcdofXY ;
unordered_map < int , int > primeX ; unordered_map < int , int > primeY ; primeX = PrimeFactor ( newX ) ; primeY = PrimeFactor ( newY ) ;
int ans = 0 ;
for ( auto c : primeX ) { if ( X % c . first != 0 ) { return -1 ; } ans += primeX [ c . first ] ; }
for ( auto c : primeY ) { if ( Y % c . first != 0 ) { return -1 ; } ans += primeY [ c . first ] ; }
return ans ; }
int main ( ) {
int X = 36 ; int Y = 48 ;
int ans = CountToMakeEqual ( X , Y ) ; cout << ans << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int L , R , V ; } ;
bool check ( vector < int > Adj [ ] , int Src , int N , bool visited [ ] ) { int color [ N ] = { 0 } ;
visited [ Src ] = true ; queue < int > q ;
q . push ( Src ) ; while ( ! q . empty ( ) ) {
int u = q . front ( ) ; q . pop ( ) ;
int Col = color [ u ] ;
for ( int x : Adj [ u ] ) {
if ( visited [ x ] == true && color [ x ] == Col ) { return false ; } else if ( visited [ x ] == false ) {
visited [ x ] = true ;
q . push ( x ) ;
color [ x ] = 1 - Col ; } } }
return true ; }
void addEdge ( vector < int > Adj [ ] , int u , int v ) { Adj [ u ] . push_back ( v ) ; Adj [ v ] . push_back ( u ) ; }
void isPossible ( struct Node Arr [ ] , int N ) {
vector < int > Adj [ N ] ;
for ( int i = 0 ; i < N - 1 ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) {
if ( Arr [ i ] . R < Arr [ j ] . L Arr [ i ] . L > Arr [ j ] . R ) { continue ; }
else { if ( Arr [ i ] . V == Arr [ j ] . V ) {
addEdge ( Adj , i , j ) ; } } } }
bool visited [ N ] = { false } ;
for ( int i = 0 ; i < N ; i ++ ) { if ( visited [ i ] == false && Adj [ i ] . size ( ) > 0 ) {
if ( check ( Adj , i , N , visited ) == false ) { cout << " No " ; return ; } } }
cout << " Yes " ; }
int main ( ) { struct Node arr [ ] = { { 5 , 7 , 2 } , { 4 , 6 , 1 } , { 1 , 5 , 2 } , { 6 , 5 , 1 } } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; isPossible ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void dfs ( int temp , int n , vector < int > & sol ) ; void lexNumbers ( int n ) { vector < int > sol ; dfs ( 1 , n , sol ) ; cout << " [ " << sol [ 0 ] ; for ( int i = 1 ; i < sol . size ( ) ; i ++ ) cout << " , ▁ " << sol [ i ] ; cout << " ] " ; } void dfs ( int temp , int n , vector < int > & sol ) { if ( temp > n ) return ; sol . push_back ( temp ) ; dfs ( temp * 10 , n , sol ) ; if ( temp % 10 != 9 ) dfs ( temp + 1 , n , sol ) ; }
int main ( ) { int n = 15 ; lexNumbers ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int minimumSwaps ( int arr [ ] , int n ) {
int count = 0 ; int i = 0 ; while ( i < n ) {
if ( arr [ i ] != i + 1 ) { while ( arr [ i ] != i + 1 ) { int temp = 0 ;
temp = arr [ arr [ i ] - 1 ] ; arr [ arr [ i ] - 1 ] = arr [ i ] ; arr [ i ] = temp ; count ++ ; } }
i ++ ; } return count ; }
int main ( ) { int arr [ ] = { 2 , 3 , 4 , 1 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << minimumSwaps ( arr , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; Node * next ; Node * prev ; } ;
void append ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; struct Node * last = * head_ref ;
new_node -> data = new_data ;
new_node -> next = NULL ;
if ( * head_ref == NULL ) { new_node -> prev = NULL ; * head_ref = new_node ; return ; }
while ( last -> next != NULL ) last = last -> next ;
last -> next = new_node ;
new_node -> prev = last ; return ; }
void printList ( Node * node ) { Node * last ;
while ( node != NULL ) { cout << node -> data << " ▁ " ; last = node ; node = node -> next ; } }
Node * mergeList ( Node * p , Node * q ) { Node * s = NULL ;
if ( p == NULL q == NULL ) { return ( p == NULL ? q : p ) ; }
if ( p -> data < q -> data ) { p -> prev = s ; s = p ; p = p -> next ; } else { q -> prev = s ; s = q ; q = q -> next ; }
Node * head = s ; while ( p != NULL && q != NULL ) { if ( p -> data < q -> data ) {
s -> next = p ; p -> prev = s ; s = s -> next ; p = p -> next ; } else {
s -> next = q ; q -> prev = s ; s = s -> next ; q = q -> next ; } }
if ( p == NULL ) { s -> next = q ; q -> prev = s ; } if ( q == NULL ) { s -> next = p ; p -> prev = s ; }
return head ; }
Node * mergeAllList ( Node * head [ ] , int k ) { Node * finalList = NULL ; for ( int i = 0 ; i < k ; i ++ ) {
finalList = mergeList ( finalList , head [ i ] ) ; }
return finalList ; }
int main ( ) { int k = 3 ; Node * head [ k ] ;
for ( int i = 0 ; i < k ; i ++ ) { head [ i ] = NULL ; }
append ( & head [ 0 ] , 1 ) ; append ( & head [ 0 ] , 5 ) ; append ( & head [ 0 ] , 9 ) ;
append ( & head [ 1 ] , 2 ) ; append ( & head [ 1 ] , 3 ) ; append ( & head [ 1 ] , 7 ) ; append ( & head [ 1 ] , 12 ) ;
append ( & head [ 2 ] , 8 ) ; append ( & head [ 2 ] , 11 ) ; append ( & head [ 2 ] , 13 ) ; append ( & head [ 2 ] , 18 ) ;
Node * finalList = mergeAllList ( head , k ) ;
printList ( finalList ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int minIndex ( int a [ ] , int i , int j ) { if ( i == j ) return i ;
int k = minIndex ( a , i + 1 , j ) ;
return ( a [ i ] < a [ k ] ) ? i : k ; }
void recurSelectionSort ( int a [ ] , int n , int index = 0 ) {
if ( index == n ) return ;
int k = minIndex ( a , index , n - 1 ) ;
if ( k != index )
swap ( a [ k ] , a [ index ] ) ;
recurSelectionSort ( a , n , index + 1 ) ; }
int main ( ) { int arr [ ] = { 3 , 1 , 5 , 2 , 7 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
recurSelectionSort ( arr , n ) ;
for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void insertionSortRecursive ( int arr [ ] , int n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
int last = arr [ n - 1 ] ; int j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; insertionSortRecursive ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void bubbleSort ( int arr [ ] , int n ) {
if ( n == 1 ) return ;
for ( int i = 0 ; i < n - 1 ; i ++ ) if ( arr [ i ] > arr [ i + 1 ] )
swap ( arr [ i ] , arr [ i + 1 ] ) ;
bubbleSort ( arr , n - 1 ) ; }
int main ( ) { int arr [ ] = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; bubbleSort ( arr , n ) ; printf ( " Sorted ▁ array ▁ : ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int maxSumAfterPartition ( int arr [ ] , int n ) {
vector < int > pos ;
vector < int > neg ;
int zero = 0 ;
int pos_sum = 0 ;
int neg_sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > 0 ) { pos . push_back ( arr [ i ] ) ; pos_sum += arr [ i ] ; } else if ( arr [ i ] < 0 ) { neg . push_back ( arr [ i ] ) ; neg_sum += arr [ i ] ; } else { zero ++ ; } }
int ans = 0 ;
sort ( pos . begin ( ) , pos . end ( ) ) ;
sort ( neg . begin ( ) , neg . end ( ) , greater < int > ( ) ) ;
if ( pos . size ( ) > 0 && neg . size ( ) > 0 ) { ans = ( pos_sum - neg_sum ) ; } else if ( pos . size ( ) > 0 ) { if ( zero > 0 ) {
ans = ( pos_sum ) ; } else {
ans = ( pos_sum - 2 * pos [ 0 ] ) ; } } else { if ( zero > 0 ) {
ans = ( -1 * neg_sum ) ; } else {
ans = ( neg [ 0 ] - ( neg_sum - neg [ 0 ] ) ) ; } } return ans ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , -5 , -7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << maxSumAfterPartition ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MaxXOR ( int arr [ ] , int N ) {
int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { res |= arr [ i ] ; }
return res ; }
int main ( ) { int arr [ ] = { 1 , 5 , 7 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << MaxXOR ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countEqual ( int A [ ] , int B [ ] , int N ) {
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
int main ( ) { int A [ ] = { 2 , 4 , 5 , 8 , 12 , 13 , 17 , 18 , 20 , 22 , 309 , 999 } ; int B [ ] = { 109 , 99 , 68 , 54 , 22 , 19 , 17 , 13 , 11 , 5 , 3 , 1 } ; int N = sizeof ( A ) / sizeof ( int ) ; cout << countEqual ( A , B , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int arr [ 100005 ] ;
bool isPalindrome ( int N ) {
int temp = N ;
int res = 0 ;
while ( temp != 0 ) { int rem = temp % 10 ; res = res * 10 + rem ; temp /= 10 ; }
if ( res == N ) { return true ; } else { return false ; } }
int sumOfDigits ( int N ) {
int sum = 0 ; while ( N != 0 ) {
sum += N % 10 ;
N /= 10 ; }
return sum ; }
bool isPrime ( int n ) {
if ( n <= 1 ) { return false ; }
for ( int i = 2 ; i <= n / 2 ; ++ i ) {
if ( n % i == 0 ) return false ; } return true ; }
void precompute ( ) {
for ( int i = 1 ; i <= 100000 ; i ++ ) {
if ( isPalindrome ( i ) ) {
int sum = sumOfDigits ( i ) ;
if ( isPrime ( sum ) ) arr [ i ] = 1 ; else arr [ i ] = 0 ; } else arr [ i ] = 0 ; }
for ( int i = 1 ; i <= 100000 ; i ++ ) { arr [ i ] = arr [ i ] + arr [ i - 1 ] ; } }
void countNumbers ( int Q [ ] [ 2 ] , int N ) {
precompute ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
cout << ( arr [ Q [ i ] [ 1 ] ] - arr [ Q [ i ] [ 0 ] - 1 ] ) ; cout << endl ; } }
int main ( ) { int Q [ ] [ 2 ] = { { 5 , 9 } , { 1 , 101 } } ; int N = sizeof ( Q ) / sizeof ( Q [ 0 ] ) ;
countNumbers ( Q , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int sum ( int n ) { int res = 0 ; while ( n > 0 ) { res += n % 10 ; n /= 10 ; } return res ; }
int smallestNumber ( int n , int s ) {
if ( sum ( n ) <= s ) { return n ; }
int ans = n , k = 1 ; for ( int i = 0 ; i < 9 ; ++ i ) {
int digit = ( ans / k ) % 10 ;
int add = k * ( ( 10 - digit ) % 10 ) ; ans += add ;
if ( sum ( ans ) <= s ) { break ; }
k *= 10 ; } return ans ; }
int main ( ) {
int N = 3 , S = 2 ;
cout << smallestNumber ( N , S ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSubsequences ( int arr [ ] , int n ) {
unordered_map < int , int > m ;
int maxCount = 0 ;
int count ; for ( int i = 0 ; i < n ; i ++ ) {
if ( m . find ( arr [ i ] ) != m . end ( ) ) {
count = m [ arr [ i ] ] ;
if ( count > 1 ) {
m [ arr [ i ] ] = count - 1 ; }
else m . erase ( arr [ i ] ) ;
if ( arr [ i ] - 1 > 0 ) m [ arr [ i ] - 1 ] += 1 ; } else {
maxCount ++ ;
if ( arr [ i ] - 1 > 0 ) m [ arr [ i ] - 1 ] += 1 ; } }
return maxCount ; }
int main ( ) { int n = 5 ; int arr [ ] = { 4 , 5 , 2 , 1 , 4 } ; cout << maxSubsequences ( arr , n ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string removeOcc ( string & s , char ch ) {
for ( int i = 0 ; s [ i ] ; i ++ ) {
if ( s [ i ] == ch ) { s . erase ( s . begin ( ) + i ) ; break ; } }
for ( int i = s . length ( ) - 1 ; i > -1 ; i -- ) {
if ( s [ i ] == ch ) { s . erase ( s . begin ( ) + i ) ; break ; } } return s ; }
int main ( ) { string s = " hello ▁ world " ; char ch = ' l ' ; cout << removeOcc ( s , ch ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minSteps ( int N , int increasing [ ] , int decreasing [ ] , int m1 , int m2 ) {
int mini = INT_MAX ;
for ( int i = 0 ; i < m1 ; i ++ ) { if ( mini > increasing [ i ] ) mini = increasing [ i ] ; }
int maxi = INT_MIN ;
for ( int i = 0 ; i < m2 ; i ++ ) { if ( maxi < decreasing [ i ] ) maxi = decreasing [ i ] ; }
int minSteps = max ( maxi , N - mini ) ;
cout << minSteps << endl ; }
int main ( ) {
int N = 7 ;
int increasing [ ] = { 3 , 5 } ; int decreasing [ ] = { 6 } ;
minSteps ( N , increasing , decreasing , m1 , m2 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void solve ( vector < int > & P , int n ) {
vector < int > arr ; arr . push_back ( 0 ) ; for ( auto x : P ) arr . push_back ( x ) ;
int cnt = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] == i ) { swap ( arr [ i ] , arr [ i + 1 ] ) ; cnt ++ ; } }
if ( arr [ n ] == n ) {
swap ( arr [ n - 1 ] , arr [ n ] ) ; cnt ++ ; }
cout << cnt << endl ; }
signed main ( ) {
int N = 9 ;
vector < int > P = { 1 , 2 , 4 , 9 , 5 , 8 , 7 , 3 , 6 } ;
solve ( P , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void SieveOfEratosthenes ( int n , unordered_set < int > & allPrimes ) {
bool prime [ n + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) allPrimes . insert ( p ) ; }
int countInterestingPrimes ( int n ) {
unordered_set < int > allPrimes ; SieveOfEratosthenes ( n , allPrimes ) ;
unordered_set < int > intersetingPrimes ; vector < int > squares , quadruples ;
for ( int i = 1 ; i * i <= n ; i ++ ) { squares . push_back ( i * i ) ; }
for ( int i = 1 ; i * i * i * i <= n ; i ++ ) { quadruples . push_back ( i * i * i * i ) ; }
for ( auto a : squares ) { for ( auto b : quadruples ) { if ( allPrimes . count ( a + b ) ) intersetingPrimes . insert ( a + b ) ; } }
return intersetingPrimes . size ( ) ; }
int main ( ) { int N = 10 ; cout << countInterestingPrimes ( N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isWaveArray ( int arr [ ] , int n ) { bool result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
int main ( ) {
int arr [ ] = { 1 , 3 , 2 , 4 } ; int n = sizeof ( arr ) / sizeof ( int ) ; if ( isWaveArray ( arr , n ) ) { cout << " YES " << endl ; } else { cout << " NO " << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countPossiblities ( int arr [ ] , int n ) {
int lastOccur [ 100000 ] ; for ( int i = 0 ; i < n ; i ++ ) { lastOccur [ i ] = -1 ; }
int dp [ n + 1 ] ;
dp [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) { int curEle = arr [ i - 1 ] ;
dp [ i ] = dp [ i - 1 ] ;
if ( lastOccur [ curEle ] != -1 & lastOccur [ curEle ] < i - 1 ) { dp [ i ] += dp [ lastOccur [ curEle ] ] ; }
lastOccur [ curEle ] = i ; }
cout << dp [ n ] << endl ; }
#include <iostream> NEW_LINE #include <vector> NEW_LINE using namespace std ;
void maxSum ( vector < vector < int > > arr , int n , int m ) {
vector < vector < int > > dp ( n ) ;
for ( int i = 0 ; i < 2 ; i ++ ) { dp [ i ] = vector < int > ( m ) ; for ( int j = 0 ; j < m ; j ++ ) { dp [ i ] [ j ] = 0 ; } }
dp [ 0 ] [ m - 1 ] = arr [ 0 ] [ m - 1 ] ; dp [ 1 ] [ m - 1 ] = arr [ 1 ] [ m - 1 ] ;
for ( int j = m - 2 ; j >= 0 ; j -- ) {
for ( int i = 0 ; i < 2 ; i ++ ) { if ( i == 1 ) { dp [ i ] [ j ] = max ( arr [ i ] [ j ] + dp [ 0 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 0 ] [ j + 2 ] ) ; } else { dp [ i ] [ j ] = max ( arr [ i ] [ j ] + dp [ 1 ] [ j + 1 ] , arr [ i ] [ j ] + dp [ 1 ] [ j + 2 ] ) ; } } }
cout << max ( dp [ 0 ] [ 0 ] , dp [ 1 ] [ 0 ] ) ; }
int main ( ) {
vector < vector < int > > arr = { { 1 , 50 , 21 , 5 } , { 2 , 10 , 10 , 5 } } ;
int N = arr [ 0 ] . size ( ) ;
maxSum ( arr , 2 , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maxSum ( vector < vector < int > > arr , int n ) {
int r1 = 0 , r2 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int temp = r1 ; r1 = max ( r1 , r2 + arr [ 0 ] [ i ] ) ; r2 = max ( r2 , temp + arr [ 1 ] [ i ] ) ; }
cout << max ( r1 , r2 ) ; }
int main ( ) { vector < vector < int > > arr = { { 1 , 50 , 21 , 5 } , { 2 , 10 , 10 , 5 } } ;
int n = arr [ 0 ] . size ( ) ; maxSum ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int mod = 1e9 + 7 ; const int mx = 1e6 ; int fact [ mx + 1 ] ;
void Calculate_factorial ( ) { fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= mx ; i ++ ) { fact [ i ] = i * fact [ i - 1 ] ; fact [ i ] %= mod ; } }
int UniModal_per ( int a , int b ) { long long int res = 1 ;
while ( b ) {
if ( b % 2 ) res = res * a ; res %= mod ; a = a * a ; a %= mod ;
b /= 2 ; }
return res ; }
void countPermutations ( int n ) {
Calculate_factorial ( ) ;
int uni_modal = UniModal_per ( 2 , n - 1 ) ;
int nonuni_modal = fact [ n ] - uni_modal ; cout << uni_modal << " ▁ " << nonuni_modal ; return ; }
int main ( ) {
int N = 4 ;
countPermutations ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int longestSubseq ( string s , int length ) {
int ones [ length + 1 ] , zeroes [ length + 1 ] ;
for ( int i = 0 ; i < length ; i ++ ) {
if ( s [ i ] == '1' ) { ones [ i + 1 ] = ones [ i ] + 1 ; zeroes [ i + 1 ] = zeroes [ i ] ; }
else { zeroes [ i + 1 ] = zeroes [ i ] + 1 ; ones [ i + 1 ] = ones [ i ] ; } } int answer = INT_MIN ; int x = 0 ; for ( int i = 0 ; i <= length ; i ++ ) { for ( int j = i ; j <= length ; j ++ ) {
x += ones [ i ] ;
x += ( zeroes [ j ] - zeroes [ i ] ) ;
x += ( ones [ length ] - ones [ j ] ) ;
answer = max ( answer , x ) ; x = 0 ; } }
cout << answer << endl ; }
int main ( ) { string s = "10010010111100101" ; int length = s . length ( ) ; longestSubseq ( s , length ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 100 ;
void largestSquare ( int matrix [ ] [ MAX ] , int R , int C , int q_i [ ] , int q_j [ ] , int K , int Q ) {
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ; int min_dist = min ( min ( i , j ) , min ( R - i - 1 , C - j - 1 ) ) ; int ans = -1 ; for ( int k = 0 ; k <= min_dist ; k ++ ) { int count = 0 ;
for ( int row = i - k ; row <= i + k ; row ++ ) for ( int col = j - k ; col <= j + k ; col ++ ) count += matrix [ row ] [ col ] ;
if ( count > K ) break ; ans = 2 * k + 1 ; } cout << ans << " STRNEWLINE " ; } }
int main ( ) { int matrix [ ] [ MAX ] = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int q_i [ ] = { 1 } ; int q_j [ ] = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 100 ;
void largestSquare ( int matrix [ ] [ MAX ] , int R , int C , int q_i [ ] , int q_j [ ] , int K , int Q ) { int countDP [ R ] [ C ] ; memset ( countDP , 0 , sizeof ( countDP ) ) ;
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] ; for ( int i = 1 ; i < R ; i ++ ) countDP [ i ] [ 0 ] = countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ; for ( int j = 1 ; j < C ; j ++ ) countDP [ 0 ] [ j ] = countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ; for ( int i = 1 ; i < R ; i ++ ) for ( int j = 1 ; j < C ; j ++ ) countDP [ i ] [ j ] = matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ;
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ;
int min_dist = min ( min ( i , j ) , min ( R - i - 1 , C - j - 1 ) ) ; int ans = -1 ; for ( int k = 0 ; k <= min_dist ; k ++ ) { int x1 = i - k , x2 = i + k ; int y1 = j - k , y2 = j + k ;
int count = countDP [ x2 ] [ y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 ] [ y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 ] [ y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 ] [ y1 - 1 ] ; if ( count > K ) break ; ans = 2 * k + 1 ; } cout << ans << " STRNEWLINE " ; } }
int main ( ) { int matrix [ ] [ MAX ] = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int q_i [ ] = { 1 } ; int q_j [ ] = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MinCost ( int arr [ ] , int n ) {
int dp [ n + 5 ] [ n + 5 ] , sum [ n + 5 ] [ n + 5 ] ;
memset ( sum , 0 , sizeof ( 0 ) ) ; for ( int i = 0 ; i < n ; i ++ ) { int k = arr [ i ] ; for ( int j = i ; j < n ; j ++ ) { if ( i == j ) sum [ i ] [ j ] = k ; else { k += arr [ j ] ; sum [ i ] [ j ] = k ; } } }
for ( int i = n - 1 ; i >= 0 ; i -- ) {
for ( int j = i ; j < n ; j ++ ) { dp [ i ] [ j ] = INT_MAX ;
if ( i == j ) dp [ i ] [ j ] = 0 ; else { for ( int k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = min ( dp [ i ] [ j ] , dp [ i ] [ k ] + dp [ k + 1 ] [ j ] + sum [ i ] [ j ] ) ; } } } } return dp [ 0 ] [ n - 1 ] ; }
int main ( ) { int arr [ ] = { 7 , 6 , 8 , 6 , 1 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << MinCost ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int f ( int i , int state , int A [ ] , int dp [ ] [ 3 ] , int N ) { if ( i >= N ) return 0 ;
else if ( dp [ i ] [ state ] != -1 ) { return dp [ i ] [ state ] ; }
else { if ( i == N - 1 ) dp [ i ] [ state ] = 1 ; else if ( state == 1 && A [ i ] > A [ i + 1 ] ) dp [ i ] [ state ] = 1 ; else if ( state == 2 && A [ i ] < A [ i + 1 ] ) dp [ i ] [ state ] = 1 ; else if ( state == 1 && A [ i ] <= A [ i + 1 ] ) dp [ i ] [ state ] = 1 + f ( i + 1 , 2 , A , dp , N ) ; else if ( state == 2 && A [ i ] >= A [ i + 1 ] ) dp [ i ] [ state ] = 1 + f ( i + 1 , 1 , A , dp , N ) ; return dp [ i ] [ state ] ; } }
int maxLenSeq ( int A [ ] , int N ) { int i , tmp , y , ans ;
int dp [ 1000 ] [ 3 ] ;
memset ( dp , -1 , sizeof dp ) ;
for ( i = 0 ; i < N ; i ++ ) { tmp = f ( i , 1 , A , dp , N ) ; tmp = f ( i , 2 , A , dp , N ) ; }
ans = -1 ; for ( i = 0 ; i < N ; i ++ ) {
y = dp [ i ] [ 1 ] ; if ( i + y >= N ) ans = max ( ans , dp [ i ] [ 1 ] + 1 ) ;
else if ( y % 2 == 0 ) { ans = max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 2 ] ) ; }
else if ( y % 2 == 1 ) { ans = max ( ans , dp [ i ] [ 1 ] + 1 + dp [ i + y ] [ 1 ] ) ; } } return ans ; }
int main ( ) { int A [ ] = { 1 , 10 , 3 , 20 , 25 , 24 } ; int n = sizeof ( A ) / sizeof ( int ) ; cout << maxLenSeq ( A , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MaxGCD ( int a [ ] , int n ) {
int Prefix [ n + 2 ] ; int Suffix [ n + 2 ] ;
Prefix [ 1 ] = a [ 0 ] ; for ( int i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = __gcd ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( int i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = __gcd ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
int ans = max ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( int i = 2 ; i < n ; i += 1 ) { ans = max ( ans , __gcd ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; }
int main ( ) { int a [ ] = { 14 , 17 , 28 , 70 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << MaxGCD ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define right  2 NEW_LINE #define left  4 NEW_LINE int dp [ left ] [ right ] ;
int findSubarraySum ( int ind , int flips , int n , int a [ ] , int k ) {
if ( flips > k ) return -1e9 ;
if ( ind == n ) return 0 ;
if ( dp [ ind ] [ flips ] != -1 ) return dp [ ind ] [ flips ] ;
int ans = 0 ;
ans = max ( 0 , a [ ind ] + findSubarraySum ( ind + 1 , flips , n , a , k ) ) ; ans = max ( ans , - a [ ind ] + findSubarraySum ( ind + 1 , flips + 1 , n , a , k ) ) ;
return dp [ ind ] [ flips ] = ans ; }
int findMaxSubarraySum ( int a [ ] , int n , int k ) {
memset ( dp , -1 , sizeof ( dp ) ) ; int ans = -1e9 ;
for ( int i = 0 ; i < n ; i ++ ) ans = max ( ans , findSubarraySum ( i , 0 , n , a , k ) ) ;
if ( ans == 0 && k == 0 ) return * max_element ( a , a + n ) ; return ans ; }
int main ( ) { int a [ ] = { -1 , -2 , -100 , -10 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int k = 1 ; cout << findMaxSubarraySum ( a , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define mod  1000000007
long long sumOddFibonacci ( int n ) { long long Sum [ n + 1 ] ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( int i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
int main ( ) { long long n = 6 ; cout << sumOddFibonacci ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; long long fun ( int marks [ ] , int n ) {
long long dp [ n ] , temp ; fill ( dp , dp + n , 1 ) ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( marks [ i ] > marks [ i + 1 ] ) { temp = i ; while ( true ) { if ( ( marks [ temp ] > marks [ temp + 1 ] ) && temp >= 0 ) { if ( dp [ temp ] > dp [ temp + 1 ] ) { temp -= 1 ; continue ; } else { dp [ temp ] = dp [ temp + 1 ] + 1 ; temp -= 1 ; } } else break ; } }
else if ( marks [ i ] < marks [ i + 1 ] ) dp [ i + 1 ] = dp [ i ] + 1 ; } int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += dp [ i ] ; return sum ; }
int main ( ) {
int n = 6 ;
int marks [ 6 ] = { 1 , 4 , 5 , 2 , 2 , 1 } ;
cout << fun ( marks , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int solve ( int N , int K ) {
int combo [ N + 1 ] = { 0 } ;
combo [ 0 ] = 1 ;
for ( int i = 1 ; i <= K ; i ++ ) {
for ( int j = 0 ; j <= N ; j ++ ) {
if ( j >= i ) {
combo [ j ] += combo [ j - i ] ; } } }
return combo [ N ] ; }
int main ( ) {
int N = 29 ; int K = 5 ; cout << solve ( N , K ) ; solve ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int computeLIS ( int circBuff [ ] , int start , int end , int n ) { int LIS [ end - start ] ;
for ( int i = start ; i < end ; i ++ ) LIS [ i ] = 1 ;
for ( int i = start + 1 ; i < end ; i ++ )
for ( int j = start ; j < i ; j ++ ) if ( circBuff [ i ] > circBuff [ j ] && LIS [ i ] < LIS [ j ] + 1 ) LIS [ i ] = LIS [ j ] + 1 ;
int res = INT_MIN ; for ( int i = start ; i < end ; i ++ ) res = max ( res , LIS [ i ] ) ; return res ; }
int LICS ( int arr [ ] , int n ) {
int circBuff [ 2 * n ] ; for ( int i = 0 ; i < n ; i ++ ) circBuff [ i ] = arr [ i ] ; for ( int i = n ; i < 2 * n ; i ++ ) circBuff [ i ] = arr [ i - n ] ;
int res = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) res = max ( computeLIS ( circBuff , i , i + n , n ) , res ) ; return res ; }
int main ( ) { int arr [ ] = { 1 , 4 , 6 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Length ▁ of ▁ LICS ▁ is ▁ " << LICS ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE #include <bits/stdc++.h> NEW_LINE using namespace std ;
int binomialCoeff ( int n , int k ) { int C [ k + 1 ] ; memset ( C , 0 , sizeof ( C ) ) ; C [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
int main ( ) { int n = 3 , m = 2 ; cout << " Number ▁ of ▁ Paths : ▁ " << binomialCoeff ( n + m , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int LCIS ( int arr1 [ ] , int n , int arr2 [ ] , int m ) {
int table [ m ] ; for ( int j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int current = 0 ;
for ( int j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
int result = 0 ; for ( int i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
int main ( ) { int arr1 [ ] = { 3 , 4 , 9 , 1 } ; int arr2 [ ] = { 5 , 3 , 8 , 9 , 10 , 2 , 1 } ; int n = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; int m = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; cout << " Length ▁ of ▁ LCIS ▁ is ▁ " << LCIS ( arr1 , n , arr2 , m ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int longComPre ( string arr [ ] , int N ) {
int freq [ N ] [ 256 ] ; for ( let String of DistString ) {
for ( int i = 0 ; i < N ; i ++ ) {
int M = arr [ i ] . length ( ) ;
for ( int j = 0 ; j < M ; j ++ ) {
freq [ i ] [ arr [ i ] [ j ] ] ++ ; } }
int maxLen = 0 ;
for ( int j = 0 ; j < 256 ; j ++ ) {
int minRowVal = INT_MAX ;
for ( int i = 0 ; i < N ; i ++ ) {
minRowVal = min ( minRowVal , freq [ i ] [ j ] ) ; }
maxLen += minRowVal ; } return maxLen ; }
int main ( ) { string arr [ ] = { " aabdc " , " abcd " , " aacd " } ; int N = 3 ; cout << longComPre ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX_CHAR = 26 ;
string removeChars ( char arr [ ] , int k ) {
int hash [ MAX_CHAR ] = { 0 } ;
int n = strlen ( arr ) ; for ( int i = 0 ; i < n ; ++ i ) hash [ arr [ i ] - ' a ' ] ++ ;
string ans = " " ;
int index = 0 ; for ( int i = 0 ; i < n ; ++ i ) {
if ( hash [ arr [ i ] - ' a ' ] != k ) { ans += arr [ i ] ; } } return ans ; }
int main ( ) { char str [ ] = " geeksforgeeks " ; int k = 2 ;
cout << removeChars ( str , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void sub_segments ( string str , int n ) { int l = str . length ( ) ; for ( int x = 0 ; x < l ; x += n ) { string newlist = str . substr ( x , n ) ;
list < char > arr ; list < char > :: iterator it ; for ( auto y : newlist ) { it = find ( arr . begin ( ) , arr . end ( ) , y ) ;
if ( it == arr . end ( ) ) arr . push_back ( y ) ; } for ( auto y : arr ) cout << y ; cout << endl ; } }
int main ( ) { string str = " geeksforgeeksgfg " ; int n = 4 ; sub_segments ( str , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findWord ( string c , int n ) { int co = 0 , i ;
string s ( n , ' ▁ ' ) ; for ( i = 0 ; i < n ; i ++ ) { if ( i < n / 2 ) co ++ ; else co = n - i ;
if ( c [ i ] + co <= 122 ) s [ i ] = ( char ) ( ( int ) c [ i ] + co ) ; else s [ i ] = ( char ) ( ( int ) c [ i ] + co - 26 ) ; } cout << s ; }
int main ( ) { string s = " abcd " ; findWord ( s , s . length ( ) ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; bool equalIgnoreCase ( string str1 , string str2 ) { int i = 0 ;
int len1 = str1 . size ( ) ;
int len2 = str2 . size ( ) ;
if ( len1 != len2 ) return false ;
while ( i < len1 ) {
if ( str1 [ i ] == str2 [ i ] ) { i ++ ; }
else if ( ! ( ( str1 [ i ] >= ' a ' && str1 [ i ] <= ' z ' ) || ( str1 [ i ] >= ' A ' && str1 [ i ] <= ' Z ' ) ) ) { return false ; }
else if ( ! ( ( str2 [ i ] >= ' a ' && str2 [ i ] <= ' z ' ) || ( str2 [ i ] >= ' A ' && str2 [ i ] <= ' Z ' ) ) ) { return false ; }
else {
if ( str1 [ i ] >= ' a ' && str1 [ i ] <= ' z ' ) { if ( str1 [ i ] - 32 != str2 [ i ] ) return false ; } else if ( str1 [ i ] >= ' A ' && str1 [ i ] <= ' Z ' ) { if ( str1 [ i ] + 32 != str2 [ i ] ) return false ; }
i ++ ;
return true ;
void equalIgnoreCaseUtil ( string str1 , string str2 ) { bool res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) cout << " Same " << endl ; else cout << " Not ▁ Same " << endl ; }
int main ( ) { string str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string maxValue ( string a , string b ) {
sort ( b . begin ( ) , b . end ( ) ) ; int n = a . length ( ) ; int m = b . length ( ) ;
int j = m - 1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( j < 0 ) break ; if ( b [ j ] > a [ i ] ) { a [ i ] = b [ j ] ;
j -- ; } }
return a ; }
int main ( ) { string a = "1234" ; string b = "4321" ; cout << maxValue ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkIfUnequal ( int n , int q ) {
string s1 = to_string ( n ) ; int a [ 26 ] = { 0 } ;
for ( int i = 0 ; i < s1 . size ( ) ; i ++ ) a [ s1 [ i ] - '0' ] ++ ;
int prod = n * q ;
string s2 = to_string ( prod ) ;
for ( int i = 0 ; i < s2 . size ( ) ; i ++ ) {
if ( a [ s2 [ i ] - '0' ] ) return false ; }
return true ; }
int countInRange ( int l , int r , int q ) { int count = 0 ; for ( int i = l ; i <= r ; i ++ ) {
if ( checkIfUnequal ( i , q ) ) count ++ ; } return count ; }
int main ( ) { int l = 10 , r = 12 , q = 2 ;
cout << countInRange ( l , r , q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool is_possible ( string s ) {
int l = s . length ( ) ; int one = 0 , zero = 0 ; for ( int i = 0 ; i < l ; i ++ ) {
if ( s [ i ] == '0' ) zero ++ ;
else one ++ ; }
if ( l % 2 == 0 ) return ( one == zero ) ;
else return ( abs ( one - zero ) == 1 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int limit = 255 ; void countFreq ( string str ) {
vector < int > count ( limit + 1 , 0 ) ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) count [ str [ i ] ] ++ ; for ( int i = 0 ; i <= limit ; i ++ ) if ( count [ i ] > 0 ) cout << ( char ) i << " ▁ " << count [ i ] << endl ; }
int main ( ) { string str = " GeeksforGeeks " ; countFreq ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countEvenOdd ( int arr [ ] , int n , int K ) { int even = 0 , odd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = __builtin_popcount ( arr [ i ] ) ; if ( x % 2 == 0 ) even ++ ; else odd ++ ; } int y ;
y = __builtin_popcount ( K ) ;
if ( y & 1 ) { cout << " Even ▁ = ▁ " << odd << " , ▁ Odd ▁ = ▁ " << even ; }
else { cout << " Even ▁ = ▁ " << even << " , ▁ Odd ▁ = ▁ " << odd ; } }
int main ( void ) { int arr [ ] = { 4 , 2 , 15 , 9 , 8 , 8 } ; int K = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
countEvenOdd ( arr , n , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string convert ( string s ) { int n = s . length ( ) ; s [ 0 ] = tolower ( s [ 0 ] ) ; for ( int i = 1 ; i < n ; i ++ ) {
if ( s [ i ] == ' ▁ ' && i < n ) {
s [ i + 1 ] = tolower ( s [ i + 1 ] ) ; i ++ ; }
else s [ i ] = toupper ( s [ i ] ) ; }
return s ; }
int main ( ) { string str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " ; cout << convert ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string change_case ( string a ) { int l = a . length ( ) ; for ( int i = 0 ; i < l ; i ++ ) {
if ( a [ i ] >= ' a ' && a [ i ] <= ' z ' ) a [ i ] = ( char ) ( 65 + ( int ) ( a [ i ] - ' a ' ) ) ;
else if ( a [ i ] >= ' A ' && a [ i ] <= ' Z ' ) a [ i ] = ( char ) ( 97 + ( int ) ( a [ i ] - ' A ' ) ) ; } return a ; }
string delete_vowels ( string a ) { string temp = " " ; int l = a . length ( ) ; for ( int i = 0 ; i < l ; i ++ ) {
if ( a [ i ] != ' a ' && a [ i ] != ' e ' && a [ i ] != ' i ' && a [ i ] != ' o ' && a [ i ] != ' u ' && a [ i ] != ' A ' && a [ i ] != ' E ' && a [ i ] != ' O ' && a [ i ] != ' U ' && a [ i ] != ' I ' ) temp += a [ i ] ; } return temp ; }
string insert_hash ( string a ) { string temp = " " ; int l = a . length ( ) ; for ( int i = 0 ; i < l ; i ++ ) {
if ( ( a [ i ] >= ' a ' && a [ i ] <= ' z ' ) || ( a [ i ] >= ' A ' && a [ i ] <= ' Z ' ) ) temp = temp + ' # ' + a [ i ] ; else temp = temp + a [ i ] ; } return temp ; }
void transformSting ( string a ) { string b = delete_vowels ( a ) ; string c = change_case ( b ) ; string d = insert_hash ( c ) ; cout << d ; }
int main ( ) { string a = " SunshinE ! ! " ;
transformSting ( a ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int reverse ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; } return rev_num ; }
int properDivSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; } bool isTcefrep ( int n ) { return properDivSum ( n ) == reverse ( n ) ; }
int main ( ) {
int N = 6 ;
if ( isTcefrep ( N ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; string findNthNo ( int n ) { string res = " " ; while ( n >= 1 ) {
if ( n & 1 ) { res = res + "3" ; n = ( n - 1 ) / 2 ; }
else { res = res + "5" ; n = ( n - 2 ) / 2 ; } }
reverse ( res . begin ( ) , res . end ( ) ) ; return res ; }
int main ( ) { int n = 5 ; cout << findNthNo ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findNthNonSquare ( int n ) {
long double x = ( long double ) n ;
long double ans = x + floor ( 0.5 + sqrt ( x ) ) ; return ( int ) ans ; }
int main ( ) {
int n = 16 ;
cout << " The ▁ " << n << " th ▁ Non - Square ▁ number ▁ is ▁ " ; cout << findNthNonSquare ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int seiresSum ( int n , int a [ ] ) { return n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ; }
int main ( ) { int n = 2 ; int a [ ] = { 1 , 2 , 3 , 4 } ; cout << seiresSum ( n , a ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int checkdigit ( int n , int k ) { while ( n ) {
int rem = n % 10 ;
if ( rem == k ) return 1 ; n = n / 10 ; } return 0 ; }
int findNthNumber ( int n , int k ) {
for ( int i = k + 1 , count = 1 ; count < n ; i ++ ) {
if ( checkdigit ( i , k ) || ( i % k == 0 ) ) count ++ ; if ( count == n ) return i ; } return -1 ; }
int main ( ) { int n = 10 , k = 2 ; cout << findNthNumber ( n , k ) << endl ; return 0 ; }
#include <iostream> NEW_LINE #include <unordered_map> NEW_LINE #include <vector> NEW_LINE using namespace std ; int find_permutations ( vector < int > & arr ) { int cnt = 0 ; int max_ind = -1 , min_ind = 10000000 ; int n = arr . size ( ) ; unordered_map < int , int > index_of ;
for ( int i = 0 ; i < n ; i ++ ) { index_of [ arr [ i ] ] = i + 1 ; } for ( int i = 1 ; i <= n ; i ++ ) {
max_ind = max ( max_ind , index_of [ i ] ) ; min_ind = min ( min_ind , index_of [ i ] ) ; if ( max_ind - min_ind + 1 == i ) cnt ++ ; } return cnt ; }
int main ( ) { vector < int > nums ; nums . push_back ( 2 ) ; nums . push_back ( 3 ) ; nums . push_back ( 1 ) ; nums . push_back ( 5 ) ; nums . push_back ( 4 ) ; cout << find_permutations ( nums ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getCount ( int a [ ] , int n ) {
int gcd = 0 ; for ( int i = 0 ; i < n ; i ++ ) gcd = __gcd ( gcd , a [ i ] ) ;
int cnt = 0 ; for ( int i = 1 ; i * i <= gcd ; i ++ ) { if ( gcd % i == 0 ) {
if ( i * i == gcd ) cnt ++ ;
else cnt += 2 ; } } return cnt ; }
int main ( ) { int a [ ] = { 4 , 16 , 1024 , 48 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << getCount ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int delCost ( string s , int cost [ ] , int l1 , int l2 ) {
bool visited [ l1 ] ; memset ( visited , 0 , sizeof ( visited ) ) ;
int ans = 0 ;
for ( int i = 0 ; i < l1 ; i ++ ) {
if ( visited [ i ] ) { continue ; }
int maxDel = 0 ;
int totalCost = 0 ;
visited [ i ] = 1 ;
for ( int j = i ; j < l1 ; j ++ ) {
if ( s [ i ] == s [ j ] ) {
maxDel = max ( maxDel , cost [ j ] ) ; totalCost += cost [ j ] ;
visited [ j ] = 1 ; } }
ans += totalCost - maxDel ; }
return ans ; }
int main ( ) {
string s = " AAABBB " ; int l1 = s . size ( ) ;
int cost [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int l2 = sizeof ( cost ) / sizeof ( cost [ 0 ] ) ;
cout << delCost ( s , cost , l1 , l2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void checkXOR ( int arr [ ] , int N ) {
if ( N % 2 == 0 ) {
int xro = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
xro ^= arr [ i ] ; }
if ( xro != 0 ) { cout << -1 << endl ; return ; }
for ( int i = 0 ; i < N - 3 ; i += 2 ) { cout << i << " ▁ " << i + 1 << " ▁ " << i + 2 << endl ; }
for ( int i = 0 ; i < N - 3 ; i += 2 ) { cout << i << " ▁ " << i + 1 << " ▁ " << N - 1 << endl ; } } else {
for ( int i = 0 ; i < N - 2 ; i += 2 ) { cout << i << " ▁ " << i + 1 << " ▁ " << i + 2 << endl ; }
for ( int i = 0 ; i < N - 2 ; i += 2 ) { cout << i << " ▁ " << i + 1 << " ▁ " << N - 1 << endl ; } } }
int main ( ) {
int arr [ ] = { 4 , 2 , 1 , 7 , 2 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
checkXOR ( arr , N ) ; }
#include <iostream> NEW_LINE using namespace std ;
int make_array_element_even ( int arr [ ] , int N ) {
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
int main ( ) { int arr [ ] = { 2 , 4 , 5 , 11 , 6 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << make_array_element_even ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int zvalue ( vector < int > & nums ) {
int m = * max_element ( nums . begin ( ) , nums . end ( ) ) ; int cnt = 0 ;
for ( int i = 0 ; i <= m ; i ++ ) { cnt = 0 ;
for ( int j = 0 ; j < nums . size ( ) ; j ++ ) {
if ( nums [ j ] >= i ) cnt ++ ; }
if ( cnt == i ) return i ; }
return -1 ; }
int main ( ) { vector < int > nums = { 7 , 8 , 9 , 0 , 0 , 1 } ; cout << zvalue ( nums ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
pair < string , int > lexico_smallest ( string s1 , string s2 ) {
map < char , int > M ; set < char > S ; pair < string , int > pr ;
for ( int i = 0 ; i <= s1 . size ( ) - 1 ; ++ i ) {
M [ s1 [ i ] ] ++ ;
S . insert ( s1 [ i ] ) ; }
for ( int i = 0 ; i <= s2 . size ( ) - 1 ; ++ i ) { M [ s2 [ i ] ] -- ; } char c = s2 [ 0 ] ; int index = 0 ; string res = " " ;
for ( auto x : S ) {
if ( x != c ) { for ( int i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } } else {
int j = 0 ; index = res . size ( ) ;
while ( s2 [ j ] == x ) { j ++ ; }
if ( s2 [ j ] < c ) { res += s2 ; for ( int i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } } else { for ( int i = 1 ; i <= M [ x ] ; ++ i ) { res += x ; } index += M [ x ] ; res += s2 ; } } } pr . first = res ; pr . second = index ;
return pr ; }
string lexico_largest ( string s1 , string s2 ) {
pair < string , int > pr = lexico_smallest ( s1 , s2 ) ;
string d1 = " " ; for ( int i = pr . second - 1 ; i >= 0 ; i -- ) { d1 += pr . first [ i ] ; }
string d2 = " " ; for ( int i = pr . first . size ( ) - 1 ; i >= pr . second + s2 . size ( ) ; -- i ) { d2 += pr . first [ i ] ; } string res = d2 + s2 + d1 ;
return res ; }
int main ( ) {
string s1 = " ethgakagmenpgs " ; string s2 = " geeks " ;
cout << lexico_smallest ( s1 , s2 ) . first << " STRNEWLINE " ; cout << lexico_largest ( s1 , s2 ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int sz = 1e5 ;
vector < int > tree [ sz ] ;
int n ;
bool vis [ sz ] ;
int subtreeSize [ sz ] ;
void addEdge ( int a , int b ) {
tree [ a ] . push_back ( b ) ;
tree [ b ] . push_back ( a ) ; }
void dfs ( int x ) {
vis [ x ] = true ;
subtreeSize [ x ] = 1 ;
for ( auto i : tree [ x ] ) { if ( ! vis [ i ] ) { dfs ( i ) ; subtreeSize [ x ] += subtreeSize [ i ] ; } } }
void countPairs ( int a , int b ) { int sub = min ( subtreeSize [ a ] , subtreeSize [ b ] ) ; cout << sub * ( n - sub ) << endl ; }
int main ( ) {
n = 6 ; addEdge ( 0 , 1 ) ; addEdge ( 0 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 3 , 4 ) ; addEdge ( 3 , 5 ) ;
dfs ( 0 ) ;
countPairs ( 1 , 3 ) ; countPairs ( 0 , 2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findPermutation ( unordered_set < int > & arr , int N ) { int pos = arr . size ( ) + 1 ;
if ( pos > N ) return 1 ; int res = 0 ; for ( int i = 1 ; i <= N ; i ++ ) {
if ( arr . find ( i ) == arr . end ( ) ) {
if ( i % pos == 0 or pos % i == 0 ) {
arr . insert ( i ) ;
res += findPermutation ( arr , N ) ;
arr . erase ( arr . find ( i ) ) ; } } }
return res ; }
int main ( ) { int N = 5 ; unordered_set < int > arr ; cout << findPermutation ( arr , N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void solve ( int arr [ ] , int n , int X , int Y ) {
int diff = Y - X ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] != 1 ) { diff = diff % ( arr [ i ] - 1 ) ; } }
if ( diff == 0 ) cout << " Yes " ; else cout << " No " ; }
int main ( ) { int arr [ ] = { 1 , 2 , 7 , 9 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int X = 11 , Y = 13 ; solve ( arr , n , X , Y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define maxN  100001
vector < int > adj [ maxN ] ;
int height [ maxN ] ;
int dist [ maxN ] ;
void addEdge ( int u , int v ) {
adj [ u ] . push_back ( v ) ;
adj [ v ] . push_back ( u ) ; }
void dfs1 ( int cur , int par ) {
for ( auto u : adj [ cur ] ) { if ( u != par ) {
dfs1 ( u , cur ) ;
height [ cur ] = max ( height [ cur ] , height [ u ] ) ; } }
height [ cur ] += 1 ; }
void dfs2 ( int cur , int par ) { int max1 = 0 ; int max2 = 0 ;
for ( auto u : adj [ cur ] ) { if ( u != par ) {
if ( height [ u ] >= max1 ) { max2 = max1 ; max1 = height [ u ] ; } else if ( height [ u ] > max2 ) { max2 = height [ u ] ; } } } int sum = 0 ; for ( auto u : adj [ cur ] ) { if ( u != par ) {
sum = ( ( max1 == height [ u ] ) ? max2 : max1 ) ; if ( max1 == height [ u ] ) dist [ u ] = 1 + max ( 1 + max2 , dist [ cur ] ) ; else dist [ u ] = 1 + max ( 1 + max1 , dist [ cur ] ) ;
dfs2 ( u , cur ) ; } } }
int main ( ) { int n = 6 ; addEdge ( 1 , 2 ) ; addEdge ( 2 , 3 ) ; addEdge ( 2 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 5 , 6 ) ;
dfs1 ( 1 , 0 ) ;
dfs2 ( 1 , 0 ) ;
for ( int i = 1 ; i <= n ; i ++ ) cout << ( max ( dist [ i ] , height [ i ] ) - 1 ) << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int middleOfThree ( int a , int b , int c ) {
int middleOfThree ( int a , int b , int c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && a < c ) || ( c < a && a < b ) ) return a ; else return c ; }
int main ( ) { int a = 20 , b = 30 , c = 40 ; cout << middleOfThree ( a , b , c ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void swap ( int * xp , int * yp ) { int temp = * xp ; * xp = * yp ; * yp = temp ; } void selectionSort ( int arr [ ] , int n ) { int i , j , min_idx ;
for ( i = 0 ; i < n - 1 ; i ++ ) {
min_idx = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
swap ( & arr [ min_idx ] , & arr [ i ] ) ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) { cout << arr [ i ] << " ▁ " ; } cout << endl ; }
int main ( ) { int arr [ ] = { 64 , 25 , 12 , 22 , 11 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
selectionSort ( arr , n ) ; cout << " Sorted ▁ array : ▁ STRNEWLINE " ;
printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool checkStr1CanConStr2 ( string & str1 , string & str2 ) {
int N = str1 . length ( ) ;
int M = str2 . length ( ) ;
set < int > st1 ;
set < int > st2 ;
int hash1 [ 256 ] = { 0 } ;
for ( int i = 0 ; i < N ; i ++ ) {
hash1 [ str1 [ i ] ] ++ ; }
for ( int i = 0 ; i < N ; i ++ ) {
st1 . insert ( str1 [ i ] ) ; }
for ( int i = 0 ; i < M ; i ++ ) {
st2 . insert ( str2 [ i ] ) ; }
if ( st1 != st2 ) { return false ; }
int hash2 [ 256 ] = { 0 } ;
for ( int i = 0 ; i < M ; i ++ ) {
hash2 [ str2 [ i ] ] ++ ; }
sort ( hash1 , hash1 + 256 ) ;
sort ( hash2 , hash2 + 256 ) ;
for ( int i = 0 ; i < 256 ; i ++ ) {
if ( hash1 [ i ] != hash2 [ i ] ) { return false ; } } return true ; }
int main ( ) { string str1 = " xyyzzlll " ; string str2 = " yllzzxxx " ; if ( checkStr1CanConStr2 ( str1 , str2 ) ) { cout << " True " ; } else { cout << " False " ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void partSort ( int arr [ ] , int N , int a , int b ) {
int l = min ( a , b ) ; int r = max ( a , b ) ; vector < int > v ( arr , arr + N ) ;
sort ( v . begin ( ) + l , v . begin ( ) + r + 1 ) ;
for ( int i = 0 ; i < N ; i ++ ) cout << v [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 7 , 8 , 4 , 5 , 2 } ; int a = 1 , b = 4 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; partSort ( arr , N , a , b ) ; }
#include <iostream> NEW_LINE #include <climits> NEW_LINE using namespace std ; #define INF  INT_MAX NEW_LINE #define N  4
int minCost ( int cost [ ] [ N ] ) {
int dist [ N ] ; for ( int i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) dist [ j ] = dist [ i ] + cost [ i ] [ j ] ; return dist [ N - 1 ] ; }
int main ( ) { int cost [ N ] [ N ] = { { 0 , 15 , 80 , 90 } , { INF , 0 , 40 , 50 } , { INF , INF , 0 , 70 } , { INF , INF , INF , 0 } } ; cout << " The ▁ Minimum ▁ cost ▁ to ▁ reach ▁ station ▁ " << N << " ▁ is ▁ " << minCost ( cost ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int numOfways ( int n , int k ) { int p = 1 ; if ( k % 2 ) p = -1 ; return ( pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
int main ( ) { int n = 4 , k = 2 ; cout << numOfways ( n , k ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
char largest_alphabet ( char a [ ] , int n ) {
char max = ' A ' ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] > max ) max = a [ i ] ;
return max ; }
char smallest_alphabet ( char a [ ] , int n ) {
char min = ' z ' ;
for ( int i = 0 ; i < n - 1 ; i ++ ) if ( a [ i ] < min ) min = a [ i ] ;
return min ; }
int main ( ) {
char a [ ] = " GeEksforGeeks " ;
int size = sizeof ( a ) / sizeof ( a [ 0 ] ) ;
cout << " Largest ▁ and ▁ smallest ▁ alphabet ▁ is ▁ : ▁ " ; cout << largest_alphabet ( a , size ) << " ▁ and ▁ " ; cout << smallest_alphabet ( a , size ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string maximumPalinUsingKChanges ( string str , int k ) { string palin = str ;
int l = 0 ; int r = str . length ( ) - 1 ;
while ( l < r ) {
if ( str [ l ] != str [ r ] ) { palin [ l ] = palin [ r ] = max ( str [ l ] , str [ r ] ) ; k -- ; } l ++ ; r -- ; }
if ( k < 0 ) return " Not ▁ possible " ; l = 0 ; r = str . length ( ) - 1 ; while ( l <= r ) {
if ( l == r ) { if ( k > 0 ) palin [ l ] = '9' ; }
if ( palin [ l ] < '9' ) {
if ( k >= 2 && palin [ l ] == str [ l ] && palin [ r ] == str [ r ] ) { k -= 2 ; palin [ l ] = palin [ r ] = '9' ; }
else if ( k >= 1 && ( palin [ l ] != str [ l ] palin [ r ] != str [ r ] ) ) { k -- ; palin [ l ] = palin [ r ] = '9' ; } } l ++ ; r -- ; } return palin ; }
int main ( ) { string str = "43435" ; int k = 3 ; cout << maximumPalinUsingKChanges ( str , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
int countTriplets ( vector < int > & A ) {
int cnt = 0 ;
unordered_map < int , int > tuples ;
for ( auto a : A )
for ( auto b : A ) ++ tuples [ a & b ] ;
for ( auto a : A )
for ( auto t : tuples )
if ( ( t . first & a ) == 0 ) cnt += t . second ;
return cnt ; }
int main ( ) {
vector < int > A = { 2 , 1 , 3 } ;
cout << countTriplets ( A ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void parity ( vector < int > even , vector < int > odd , vector < int > v , int i , int & min ) {
if ( i == v . size ( ) || even . size ( ) == 0 && odd . size ( ) == 0 ) { int count = 0 ; for ( int j = 0 ; j < v . size ( ) - 1 ; j ++ ) { if ( v [ j ] % 2 != v [ j + 1 ] % 2 ) count ++ ; } if ( count < min ) min = count ; return ; }
if ( v [ i ] != -1 ) parity ( even , odd , v , i + 1 , min ) ;
else { if ( even . size ( ) != 0 ) { int x = even . back ( ) ; even . pop_back ( ) ; v [ i ] = x ; parity ( even , odd , v , i + 1 , min ) ;
even . push_back ( x ) ; } if ( odd . size ( ) != 0 ) { int x = odd . back ( ) ; odd . pop_back ( ) ; v [ i ] = x ; parity ( even , odd , v , i + 1 , min ) ;
odd . push_back ( x ) ; } } }
void minDiffParity ( vector < int > v , int n ) {
vector < int > even ;
vector < int > odd ; unordered_map < int , int > m ; for ( int i = 1 ; i <= n ; i ++ ) m [ i ] = 1 ; for ( int i = 0 ; i < v . size ( ) ; i ++ ) {
if ( v [ i ] != -1 ) m . erase ( v [ i ] ) ; }
for ( auto i : m ) { if ( i . first % 2 == 0 ) even . push_back ( i . first ) ; else odd . push_back ( i . first ) ; } int min = 1000 ; parity ( even , odd , v , 0 , min ) ; cout << min << endl ; }
int main ( ) { int n = 8 ; vector < int > v = { 2 , 1 , 4 , -1 , -1 , 6 , -1 , 8 } ; minDiffParity ( v , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define ll  long long int NEW_LINE #define MAX  100005 NEW_LINE using namespace std ; vector < int > adjacent [ MAX ] ; bool visited [ MAX ] ;
int startnode , endnode , thirdnode ; int maxi = -1 , N ;
int parent [ MAX ] ;
bool vis [ MAX ] ;
void dfs ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent [ u ] . size ( ) ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] ) { temp ++ ; dfs ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; startnode = u ; } } }
void dfs1 ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent [ u ] . size ( ) ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] ) { temp ++ ; parent [ adjacent [ u ] [ i ] ] = u ; dfs1 ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; endnode = u ; } } }
void dfs2 ( int u , int count ) { visited [ u ] = true ; int temp = 0 ; for ( int i = 0 ; i < adjacent [ u ] . size ( ) ; i ++ ) { if ( ! visited [ adjacent [ u ] [ i ] ] && ! vis [ adjacent [ u ] [ i ] ] ) { temp ++ ; dfs2 ( adjacent [ u ] [ i ] , count + 1 ) ; } } if ( temp == 0 ) { if ( maxi < count ) { maxi = count ; thirdnode = u ; } } }
void findNodes ( ) {
dfs ( 1 , 0 ) ; for ( int i = 0 ; i <= N ; i ++ ) visited [ i ] = false ; maxi = -1 ;
dfs1 ( startnode , 0 ) ; for ( int i = 0 ; i <= N ; i ++ ) visited [ i ] = false ;
int x = endnode ; vis [ startnode ] = true ;
while ( x != startnode ) { vis [ x ] = true ; x = parent [ x ] ; } maxi = -1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( vis [ i ] ) dfs2 ( i , 0 ) ; } }
int main ( ) { N = 4 ; adjacent [ 1 ] . push_back ( 2 ) ; adjacent [ 2 ] . push_back ( 1 ) ; adjacent [ 1 ] . push_back ( 3 ) ; adjacent [ 3 ] . push_back ( 1 ) ; adjacent [ 1 ] . push_back ( 4 ) ; adjacent [ 4 ] . push_back ( 1 ) ; findNodes ( ) ; cout << " ( " << startnode << " , ▁ " << endnode << " , ▁ " << thirdnode << " ) " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void newvol ( double x ) { cout << " percentage ▁ increase ▁ in ▁ the " << " ▁ volume ▁ of ▁ the ▁ sphere ▁ is ▁ " << pow ( x , 3 ) / 10000 + 3 * x + ( 3 * pow ( x , 2 ) ) / 100 << " % " << endl ; }
int main ( ) { double x = 10 ; newvol ( x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void length_of_chord ( double r , double x ) { cout << " The ▁ length ▁ of ▁ the ▁ chord " << " ▁ of ▁ the ▁ circle ▁ is ▁ " << 2 * r * sin ( x * ( 3.14 / 180 ) ) << endl ; }
int main ( ) { double r = 4 , x = 63 ; length_of_chord ( r , x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float area ( float a ) {
if ( a < 0 ) return -1 ;
float area = sqrt ( a ) / 6 ; return area ; }
int main ( ) { float a = 10 ; cout << area ( a ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double longestRodInCuboid ( int length , int breadth , int height ) { double result ; int temp ;
temp = length * length + breadth * breadth + height * height ;
result = sqrt ( temp ) ; return result ; }
int main ( ) { int length = 12 , breadth = 9 , height = 8 ;
cout << longestRodInCuboid ( length , breadth , height ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool LiesInsieRectangle ( int a , int b , int x , int y ) { if ( x - y - b <= 0 && x - y + b >= 0 && x + y - 2 * a + b <= 0 && x + y - b >= 0 ) return true ; return false ; }
int main ( ) { int a = 7 , b = 2 , x = 4 , y = 5 ; if ( LiesInsieRectangle ( a , b , x , y ) ) cout << " Given ▁ point ▁ lies ▁ inside ▁ the ▁ rectangle " ; else cout << " Given ▁ point ▁ does ▁ not ▁ lie ▁ on ▁ the ▁ rectangle " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxvolume ( int s ) { int maxvalue = 0 ;
for ( int i = 1 ; i <= s - 2 ; i ++ ) {
for ( int j = 1 ; j <= s - 1 ; j ++ ) {
int k = s - i - j ;
maxvalue = max ( maxvalue , i * j * k ) ; } } return maxvalue ; }
int main ( ) { int s = 8 ; cout << maxvolume ( s ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxvolume ( int s ) {
int length = s / 3 ; s -= length ;
int breadth = s / 2 ;
int height = s - breadth ; return length * breadth * height ; }
int main ( ) { int s = 8 ; cout << maxvolume ( s ) << endl ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
double hexagonArea ( double s ) { return ( ( 3 * sqrt ( 3 ) * ( s * s ) ) / 2 ) ; }
int main ( ) {
double s = 4 ; cout << " Area ▁ : ▁ " << hexagonArea ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSquare ( int b , int m ) {
return ( b / m - 1 ) * ( b / m ) / 2 ; }
int main ( ) { int b = 10 , m = 2 ; cout << maxSquare ( b , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findRightAngle ( int A , int H ) {
long D = pow ( H , 4 ) - 16 * A * A ; if ( D >= 0 ) {
long root1 = ( H * H + sqrt ( D ) ) / 2 ; long root2 = ( H * H - sqrt ( D ) ) / 2 ; long a = sqrt ( root1 ) ; long b = sqrt ( root2 ) ; if ( b >= a ) cout << a << " ▁ " << b << " ▁ " << H ; else cout << b << " ▁ " << a << " ▁ " << H ; } else cout << " - 1" ; }
int main ( ) { findRightAngle ( 6 , 5 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int numberOfSquares ( int base ) {
base = ( base - 2 ) ;
base = floor ( base / 2 ) ; return base * ( base + 1 ) / 2 ; }
int main ( ) { int base = 8 ; cout << numberOfSquares ( base ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void performQuery ( vector < int > arr , vector < vector < int > > Q ) {
for ( int i = 0 ; i < Q . size ( ) ; i ++ ) {
int or1 = 0 ;
int x = Q [ i ] [ 0 ] ; arr [ x - 1 ] = Q [ i ] [ 1 ] ;
for ( int j = 0 ; j < arr . size ( ) ; j ++ ) { or1 = or1 | arr [ j ] ; }
cout << or1 << " ▁ " ; } }
int main ( ) { vector < int > arr ( { 1 , 2 , 3 } ) ; vector < int > v1 ( { 1 , 4 } ) ; vector < int > v2 ( { 3 , 0 } ) ; vector < vector < int > > Q ; Q . push_back ( v1 ) ; Q . push_back ( v2 ) ; performQuery ( arr , Q ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int smallest ( int k , int d ) { int cnt = 1 ; int m = d % k ;
vector < int > v ( k , 0 ) ; v [ m ] = 1 ;
while ( 1 ) { if ( m == 0 ) return cnt ; m = ( ( ( m * ( 10 % k ) ) % k ) + ( d % k ) ) % k ;
if ( v [ m ] == 1 ) return -1 ; v [ m ] = 1 ; cnt ++ ; } return -1 ; }
int main ( ) { int d = 1 ; int k = 41 ; cout << smallest ( k , d ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int fib ( int n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
int findVertices ( int n ) {
return fib ( n + 2 ) ; }
int main ( ) { int n = 3 ; cout << findVertices ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void checkCommonDivisor ( int arr [ ] , int N , int X ) {
int G = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { G = __gcd ( G , arr [ i ] ) ; } int copy_G = G ; for ( int divisor = 2 ; divisor <= X ; divisor ++ ) {
while ( G % divisor == 0 ) {
G = G / divisor ; } }
if ( G <= X ) { cout << " Yes STRNEWLINE " ;
for ( int i = 0 ; i < N ; i ++ ) cout << arr [ i ] / copy_G << " ▁ " ; cout << endl ; }
else cout < < " No " ; }
int main ( ) {
int arr [ ] = { 6 , 15 , 6 } , X = 6 ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; checkCommonDivisor ( arr , N , X ) ; }
#include <iostream> NEW_LINE using namespace std ; void printSpiral ( int size ) { int row = 0 , col = 0 ; int boundary = size - 1 ; int sizeLeft = size - 1 ; int flag = 1 ;
char move = ' r ' ;
int matrix [ size ] [ size ] = { 0 } ; for ( int i = 1 ; i < size * size + 1 ; i ++ ) {
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
for ( row = 0 ; row < size ; row ++ ) { for ( col = 0 ; col < size ; col ++ ) { int n = matrix [ row ] [ col ] ; if ( n < 10 ) cout << n << " ▁ " ; else cout << n << " ▁ " ; } cout << endl ; } }
int main ( ) {
int size = 5 ;
printSpiral ( size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * next ; struct Node * prev ; } ;
void reverse ( struct Node * * head_ref ) { struct Node * temp = NULL ; struct Node * current = * head_ref ;
while ( current != NULL ) { temp = current -> prev ; current -> prev = current -> next ; current -> next = temp ; current = current -> prev ; }
if ( temp != NULL ) * head_ref = temp -> prev ; }
struct Node * merge ( struct Node * first , struct Node * second ) {
if ( ! first ) return second ;
if ( ! second ) return first ;
if ( first -> data < second -> data ) { first -> next = merge ( first -> next , second ) ; first -> next -> prev = first ; first -> prev = NULL ; return first ; } else { second -> next = merge ( first , second -> next ) ; second -> next -> prev = second ; second -> prev = NULL ; return second ; } }
struct Node * sort ( struct Node * head ) {
if ( head == NULL head -> next == NULL ) return head ; struct Node * current = head -> next ; while ( current != NULL ) {
if ( current -> data < current -> prev -> data ) break ;
current = current -> next ; }
if ( current == NULL ) return head ;
current -> prev -> next = NULL ; current -> prev = NULL ;
reverse ( & current ) ;
return merge ( head , current ) ; }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ;
new_node -> data = new_data ;
new_node -> prev = NULL ;
new_node -> next = ( * head_ref ) ;
if ( ( * head_ref ) != NULL ) ( * head_ref ) -> prev = new_node ;
( * head_ref ) = new_node ; }
void printList ( struct Node * head ) {
if ( head == NULL ) cout << " Doubly ▁ Linked ▁ list ▁ empty " ; while ( head != NULL ) { cout << head -> data << " ▁ " ; head = head -> next ; } }
int main ( ) { struct Node * head = NULL ;
push ( & head , 1 ) ; push ( & head , 4 ) ; push ( & head , 6 ) ; push ( & head , 10 ) ; push ( & head , 12 ) ; push ( & head , 7 ) ; push ( & head , 5 ) ; push ( & head , 2 ) ; cout << " Original ▁ Doubly ▁ linked ▁ list : n " ; printList ( head ) ;
head = sort ( head ) ; cout << " Doubly linked list after sorting : n " ; printList ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { char data ; struct Node * next ; } ;
Node * newNode ( char key ) { Node * temp = new Node ; temp -> data = key ; temp -> next = NULL ; return temp ; }
void printlist ( Node * head ) { if ( ! head ) { cout << " Empty ▁ List STRNEWLINE " ; return ; } while ( head != NULL ) { cout << head -> data << " ▁ " ; if ( head -> next ) cout << " - > ▁ " ; head = head -> next ; } cout << endl ; }
bool isVowel ( char x ) { return ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) ; }
Node * arrange ( Node * head ) { Node * newHead = head ;
Node * latestVowel ; Node * curr = head ;
if ( head == NULL ) return NULL ;
if ( isVowel ( head -> data ) )
latestVowel = head ; else {
while ( curr -> next != NULL && ! isVowel ( curr -> next -> data ) ) curr = curr -> next ;
if ( curr -> next == NULL ) return head ;
latestVowel = newHead = curr -> next ; curr -> next = curr -> next -> next ; latestVowel -> next = head ; }
while ( curr != NULL && curr -> next != NULL ) { if ( isVowel ( curr -> next -> data ) ) {
if ( curr == latestVowel ) {
latestVowel = curr = curr -> next ; } else {
Node * temp = latestVowel -> next ;
latestVowel -> next = curr -> next ;
latestVowel = latestVowel -> next ;
curr -> next = curr -> next -> next ;
latestVowel -> next = temp ; } } else {
curr = curr -> next ; } } return newHead ; }
int main ( ) { Node * head = newNode ( ' a ' ) ; head -> next = newNode ( ' b ' ) ; head -> next -> next = newNode ( ' c ' ) ; head -> next -> next -> next = newNode ( ' e ' ) ; head -> next -> next -> next -> next = newNode ( ' d ' ) ; head -> next -> next -> next -> next -> next = newNode ( ' o ' ) ; head -> next -> next -> next -> next -> next -> next = newNode ( ' x ' ) ; head -> next -> next -> next -> next -> next -> next -> next = newNode ( ' i ' ) ; printf ( " Linked ▁ list ▁ before ▁ : STRNEWLINE " ) ; printlist ( head ) ; head = arrange ( head ) ; printf ( " Linked ▁ list ▁ after ▁ : STRNEWLINE " ) ; printlist ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * left , * right ; } ;
Node * newNode ( int data ) { Node * temp = new Node ; temp -> data = data ; temp -> right = temp -> left = NULL ; return temp ; } Node * KthLargestUsingMorrisTraversal ( Node * root , int k ) { Node * curr = root ; Node * Klargest = NULL ;
int count = 0 ; while ( curr != NULL ) {
if ( curr -> right == NULL ) {
if ( ++ count == k ) Klargest = curr ;
curr = curr -> left ; } else {
Node * succ = curr -> right ; while ( succ -> left != NULL && succ -> left != curr ) succ = succ -> left ; if ( succ -> left == NULL ) {
succ -> left = curr ;
curr = curr -> right ; }
else { succ -> left = NULL ; if ( ++ count == k ) Klargest = curr ;
curr = curr -> left ; } } } return Klargest ; } int main ( ) {
Node * root = newNode ( 4 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 7 ) ; root -> left -> left = newNode ( 1 ) ; root -> left -> right = newNode ( 3 ) ; root -> right -> left = newNode ( 6 ) ; root -> right -> right = newNode ( 10 ) ; cout << " Finding ▁ K - th ▁ largest ▁ Node ▁ in ▁ BST ▁ : ▁ " << KthLargestUsingMorrisTraversal ( root , 2 ) -> data ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX_SIZE  10
void sortByRow ( int mat [ ] [ MAX_SIZE ] , int n , bool ascending ) { for ( int i = 0 ; i < n ; i ++ ) { if ( ascending ) sort ( mat [ i ] , mat [ i ] + n ) ; else sort ( mat [ i ] , mat [ i ] + n , greater < int > ( ) ) ; } }
void transpose ( int mat [ ] [ MAX_SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ )
swap ( mat [ i ] [ j ] , mat [ j ] [ i ] ) ; }
void sortMatRowAndColWise ( int mat [ ] [ MAX_SIZE ] , int n ) {
sortByRow ( mat , n , true ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n , false ) ;
transpose ( mat , n ) ; }
void printMat ( int mat [ ] [ MAX_SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) cout << mat [ i ] [ j ] << " ▁ " ; cout << endl ; } }
int main ( ) { int n = 3 ; int mat [ n ] [ MAX_SIZE ] = { { 3 , 2 , 1 } , { 9 , 8 , 7 } , { 6 , 5 , 4 } } ; cout << " Original ▁ Matrix : STRNEWLINE " ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; cout << " Matrix After Sorting : " ; printMat ( mat , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX_SIZE  10
void sortByRow ( int mat [ MAX_SIZE ] [ MAX_SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ )
sort ( mat [ i ] , mat [ i ] + n ) ; }
void transpose ( int mat [ MAX_SIZE ] [ MAX_SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ )
swap ( mat [ i ] [ j ] , mat [ j ] [ i ] ) ; }
void sortMatRowAndColWise ( int mat [ MAX_SIZE ] [ MAX_SIZE ] , int n ) {
sortByRow ( mat , n ) ;
transpose ( mat , n ) ;
sortByRow ( mat , n ) ;
transpose ( mat , n ) ; }
void printMat ( int mat [ MAX_SIZE ] [ MAX_SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) cout << mat [ i ] [ j ] << " ▁ " ; cout << endl ; } }
int main ( ) { int mat [ MAX_SIZE ] [ MAX_SIZE ] = { { 4 , 1 , 3 } , { 9 , 6 , 8 } , { 5 , 2 , 7 } } ; int n = 3 ; cout << " Original ▁ Matrix : STRNEWLINE " ; printMat ( mat , n ) ; sortMatRowAndColWise ( mat , n ) ; cout << " Matrix After Sorting : " ; printMat ( mat , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void doublyEven ( int n ) { int arr [ n ] [ n ] , i , j ;
for ( i = 0 ; i < n ; i ++ ) for ( j = 0 ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * i ) + j + 1 ;
for ( i = 0 ; i < n / 4 ; i ++ ) for ( j = 0 ; j < n / 4 ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 0 ; i < n / 4 ; i ++ ) for ( j = 3 * ( n / 4 ) ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 3 * n / 4 ; i < n ; i ++ ) for ( j = 0 ; j < n / 4 ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 3 * n / 4 ; i < n ; i ++ ) for ( j = 3 * n / 4 ; j < n ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = n / 4 ; i < 3 * n / 4 ; i ++ ) for ( j = n / 4 ; j < 3 * n / 4 ; j ++ ) arr [ i ] [ j ] = ( n * n + 1 ) - arr [ i ] [ j ] ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = 0 ; j < n ; j ++ ) cout << arr [ i ] [ j ] << " ▁ " ; cout << " STRNEWLINE " ; } }
int main ( ) { int n = 8 ;
doublyEven ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
const int cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
void Kroneckerproduct ( int A [ ] [ cola ] , int B [ ] [ colb ] ) { int C [ rowa * rowb ] [ cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; cout << C [ i + l + 1 ] [ j + k + 1 ] << " ▁ " ; } } cout << endl ; } } }
int main ( ) { int A [ 3 ] [ 2 ] = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } , B [ 2 ] [ 3 ] = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define N  4 NEW_LINE using namespace std ;
bool isLowerTriangularMatrix ( int mat [ N ] [ N ] ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != 0 ) return false ; return true ; }
int main ( ) { int mat [ N ] [ N ] = { { 1 , 0 , 0 , 0 } , { 1 , 4 , 0 , 0 } , { 4 , 6 , 2 , 0 } , { 0 , 4 , 7 , 6 } } ;
if ( isLowerTriangularMatrix ( mat ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define N  4 NEW_LINE using namespace std ;
bool isUpperTriangularMatrix ( int mat [ N ] [ N ] ) { for ( int i = 1 ; i < N ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( mat [ i ] [ j ] != 0 ) return false ; return true ; }
int main ( ) { int mat [ N ] [ N ] = { { 1 , 3 , 5 , 3 } , { 0 , 4 , 6 , 2 } , { 0 , 0 , 2 , 5 } , { 0 , 0 , 0 , 6 } } ; if ( isUpperTriangularMatrix ( mat ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
const int m = 3 ;
const int n = 2 ;
long long countSets ( int a [ n ] [ m ] ) {
long long res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < m ; j ++ ) a [ i ] [ j ] ? u ++ : v ++ ; res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 ; }
for ( int i = 0 ; i < m ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < n ; j ++ ) a [ j ] [ i ] ? u ++ : v ++ ; res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 ; }
return res - ( n * m ) ; }
int main ( ) { int a [ ] [ 3 ] = { ( 1 , 0 , 1 ) , ( 0 , 1 , 0 ) } ; cout << countSets ( a ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; const int MAX = 100 ;
void transpose ( int mat [ ] [ MAX ] , int tr [ ] [ MAX ] , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) tr [ i ] [ j ] = mat [ j ] [ i ] ; }
bool isSymmetric ( int mat [ ] [ MAX ] , int N ) { int tr [ N ] [ MAX ] ; transpose ( mat , tr , N ) ; for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != tr [ i ] [ j ] ) return false ; return true ; }
int main ( ) { int mat [ ] [ MAX ] = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; const int MAX = 100 ;
bool isSymmetric ( int mat [ ] [ MAX ] , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != mat [ j ] [ i ] ) return false ; return true ; }
int main ( ) { int mat [ ] [ MAX ] = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
const int MAX = 100 ;
int findNormal ( int mat [ ] [ MAX ] , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) sum += mat [ i ] [ j ] * mat [ i ] [ j ] ; return sqrt ( sum ) ; }
int findTrace ( int mat [ ] [ MAX ] , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += mat [ i ] [ i ] ; return sum ; }
int main ( ) { int mat [ ] [ MAX ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; cout << " Trace ▁ of ▁ Matrix ▁ = ▁ " << findTrace ( mat , 5 ) << endl ; cout << " Normal ▁ of ▁ Matrix ▁ = ▁ " << findNormal ( mat , 5 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxDet ( int n ) { return ( 2 * n * n * n ) ; }
void resMatrix ( int n ) { for ( int i = 0 ; i < 3 ; i ++ ) { for ( int j = 0 ; j < 3 ; j ++ ) {
if ( i == 0 && j == 2 ) cout << "0 ▁ " ; else if ( i == 1 && j == 0 ) cout << "0 ▁ " ; else if ( i == 2 && j == 1 ) cout << "0 ▁ " ;
else cout < < n << " ▁ " ; } cout << " STRNEWLINE " ; } }
int main ( ) { int n = 15 ; cout << " Maximum ▁ Determinant ▁ = ▁ " << maxDet ( n ) ; cout << " Resultant Matrix : " resMatrix ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countNegative ( int M [ ] [ 4 ] , int n , int m ) { int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { if ( M [ i ] [ j ] < 0 ) count += 1 ;
else break ; } } return count ; }
int main ( ) { int M [ 3 ] [ 4 ] = { { -3 , -2 , -1 , 1 } , { -2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; cout << countNegative ( M , 3 , 4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countNegative ( int M [ ] [ 4 ] , int n , int m ) {
int count = 0 ;
int i = 0 ; int j = m - 1 ;
while ( j >= 0 && i < n ) { if ( M [ i ] [ j ] < 0 ) {
count += j + 1 ;
i += 1 ; }
else j -= 1 ; } return count ; }
int main ( ) { int M [ 3 ] [ 4 ] = { { -3 , -2 , -1 , 1 } , { -2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; cout << countNegative ( M , 3 , 4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getLastNegativeIndex ( int array [ ] , int start , int end , int n ) {
if ( start == end ) { return start ; }
int mid = start + ( end - start ) / 2 ;
if ( array [ mid ] < 0 ) {
if ( mid + 1 < n && array [ mid + 1 ] >= 0 ) { return mid ; }
return getLastNegativeIndex ( array , mid + 1 , end , n ) ; } else {
return getLastNegativeIndex ( array , start , mid - 1 , n ) ; } }
int countNegative ( int M [ ] [ 4 ] , int n , int m ) {
int count = 0 ;
int nextEnd = m - 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( M [ i ] [ 0 ] >= 0 ) { break ; }
nextEnd = getLastNegativeIndex ( M [ i ] , 0 , nextEnd , 4 ) ; count += nextEnd + 1 ; } return count ; }
int main ( ) { int M [ ] [ 4 ] = { { -3 , -2 , -1 , 1 } , { -2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; int r = 3 ; int c = 4 ; cout << ( countNegative ( M , r , c ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  5
int findMaxValue ( int mat [ ] [ N ] ) {
int maxValue = INT_MIN ;
for ( int a = 0 ; a < N - 1 ; a ++ ) for ( int b = 0 ; b < N - 1 ; b ++ ) for ( int d = a + 1 ; d < N ; d ++ ) for ( int e = b + 1 ; e < N ; e ++ ) if ( maxValue < ( mat [ d ] [ e ] - mat [ a ] [ b ] ) ) maxValue = mat [ d ] [ e ] - mat [ a ] [ b ] ; return maxValue ; }
int main ( ) { int mat [ N ] [ N ] = { { 1 , 2 , -1 , -4 , -20 } , { -8 , -3 , 4 , 2 , 1 } , { 3 , 8 , 6 , 1 , 3 } , { -4 , -1 , 1 , 7 , -6 } , { 0 , -4 , 10 , -5 , 1 } } ; cout << " Maximum ▁ Value ▁ is ▁ " << findMaxValue ( mat ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  5
int findMaxValue ( int mat [ ] [ N ] ) {
int maxValue = INT_MIN ;
int maxArr [ N ] [ N ] ;
maxArr [ N - 1 ] [ N - 1 ] = mat [ N - 1 ] [ N - 1 ] ;
int maxv = mat [ N - 1 ] [ N - 1 ] ; for ( int j = N - 2 ; j >= 0 ; j -- ) { if ( mat [ N - 1 ] [ j ] > maxv ) maxv = mat [ N - 1 ] [ j ] ; maxArr [ N - 1 ] [ j ] = maxv ; }
maxv = mat [ N - 1 ] [ N - 1 ] ; for ( int i = N - 2 ; i >= 0 ; i -- ) { if ( mat [ i ] [ N - 1 ] > maxv ) maxv = mat [ i ] [ N - 1 ] ; maxArr [ i ] [ N - 1 ] = maxv ; }
for ( int i = N - 2 ; i >= 0 ; i -- ) { for ( int j = N - 2 ; j >= 0 ; j -- ) {
if ( maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] > maxValue ) maxValue = maxArr [ i + 1 ] [ j + 1 ] - mat [ i ] [ j ] ;
maxArr [ i ] [ j ] = max ( mat [ i ] [ j ] , max ( maxArr [ i ] [ j + 1 ] , maxArr [ i + 1 ] [ j ] ) ) ; } } return maxValue ; }
int main ( ) { int mat [ N ] [ N ] = { { 1 , 2 , -1 , -4 , -20 } , { -8 , -3 , 4 , 2 , 1 } , { 3 , 8 , 6 , 1 , 3 } , { -4 , -1 , 1 , 7 , -6 } , { 0 , -4 , 10 , -5 , 1 } } ; cout << " Maximum ▁ Value ▁ is ▁ " << findMaxValue ( mat ) ; return 0 ; }
#include <iostream> NEW_LINE #include <climits> NEW_LINE using namespace std ; #define INF  INT_MAX NEW_LINE #define N  4
void youngify ( int mat [ ] [ N ] , int i , int j ) {
int downVal = ( i + 1 < N ) ? mat [ i + 1 ] [ j ] : INF ; int rightVal = ( j + 1 < N ) ? mat [ i ] [ j + 1 ] : INF ;
if ( downVal == INF && rightVal == INF ) return ;
if ( downVal < rightVal ) { mat [ i ] [ j ] = downVal ; mat [ i + 1 ] [ j ] = INF ; youngify ( mat , i + 1 , j ) ; } else { mat [ i ] [ j ] = rightVal ; mat [ i ] [ j + 1 ] = INF ; youngify ( mat , i , j + 1 ) ; } }
int extractMin ( int mat [ ] [ N ] ) { int ret = mat [ 0 ] [ 0 ] ; mat [ 0 ] [ 0 ] = INF ; youngify ( mat , 0 , 0 ) ; return ret ; }
void printSorted ( int mat [ ] [ N ] ) { cout << " Elements ▁ of ▁ matrix ▁ in ▁ sorted ▁ order ▁ n " ; for ( int i = 0 ; i < N * N ; i ++ ) cout << extractMin ( mat ) << " ▁ " ; }
int main ( ) { int mat [ N ] [ N ] = { { 10 , 20 , 30 , 40 } , { 15 , 25 , 35 , 45 } , { 27 , 29 , 37 , 48 } , { 32 , 33 , 39 , 50 } , } ; printSorted ( mat ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
#define n  5
void printSumSimple ( int mat [ ] [ n ] , int k ) {
if ( k > n ) return ;
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
for ( int j = 0 ; j < n - k + 1 ; j ++ ) {
int sum = 0 ; for ( int p = i ; p < k + i ; p ++ ) for ( int q = j ; q < k + j ; q ++ ) sum += mat [ p ] [ q ] ; cout << sum << " ▁ " ; }
cout << endl ; } }
int main ( ) { int mat [ n ] [ n ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; int k = 3 ; printSumSimple ( mat , k ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
#define n  5
void printSumTricky ( int mat [ ] [ n ] , int k ) {
if ( k > n ) return ;
int stripSum [ n ] [ n ] ;
for ( int j = 0 ; j < n ; j ++ ) {
int sum = 0 ; for ( int i = 0 ; i < k ; i ++ ) sum += mat [ i ] [ j ] ; stripSum [ 0 ] [ j ] = sum ;
for ( int i = 1 ; i < n - k + 1 ; i ++ ) { sum += ( mat [ i + k - 1 ] [ j ] - mat [ i - 1 ] [ j ] ) ; stripSum [ i ] [ j ] = sum ; } }
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
int sum = 0 ; for ( int j = 0 ; j < k ; j ++ ) sum += stripSum [ i ] [ j ] ; cout << sum << " ▁ " ;
for ( int j = 1 ; j < n - k + 1 ; j ++ ) { sum += ( stripSum [ i ] [ j + k - 1 ] - stripSum [ i ] [ j - 1 ] ) ; cout << sum << " ▁ " ; } cout << endl ; } }
int main ( ) { int mat [ n ] [ n ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; int k = 3 ; printSumTricky ( mat , k ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define M  3 NEW_LINE #define N  4
void transpose ( int A [ ] [ N ] , int B [ ] [ M ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < M ; j ++ ) B [ i ] [ j ] = A [ j ] [ i ] ; }
int main ( ) { int A [ M ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } } ; int B [ N ] [ M ] , i , j ; transpose ( A , B ) ; printf ( " Result ▁ matrix ▁ is ▁ STRNEWLINE " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < M ; j ++ ) printf ( " % d ▁ " , B [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  4
void transpose ( int A [ ] [ N ] ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) swap ( A [ i ] [ j ] , A [ j ] [ i ] ) ; }
int main ( ) { int A [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; transpose ( A ) ; printf ( " Modified ▁ matrix ▁ is ▁ STRNEWLINE " ) ; for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) printf ( " % d ▁ " , A [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE using namespace std ;
int pathCountRec ( int mat [ ] [ C ] , int m , int n , int k ) {
if ( m < 0 n < 0 ) return 0 ; if ( m == 0 && n == 0 ) return ( k == mat [ m ] [ n ] ) ;
return pathCountRec ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountRec ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ; }
int pathCount ( int mat [ ] [ C ] , int k ) { return pathCountRec ( mat , R - 1 , C - 1 , k ) ; }
int main ( ) { int k = 12 ; int mat [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 6 , 5 } , { 3 , 2 , 1 } } ; cout << pathCount ( mat , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE #define MAX_K  1000 NEW_LINE using namespace std ; int dp [ R ] [ C ] [ MAX_K ] ; int pathCountDPRecDP ( int mat [ ] [ C ] , int m , int n , int k ) {
if ( m < 0 n < 0 ) return 0 ; if ( m == 0 && n == 0 ) return ( k == mat [ m ] [ n ] ) ;
if ( dp [ m ] [ n ] [ k ] != -1 ) return dp [ m ] [ n ] [ k ] ;
dp [ m ] [ n ] [ k ] = pathCountDPRecDP ( mat , m - 1 , n , k - mat [ m ] [ n ] ) + pathCountDPRecDP ( mat , m , n - 1 , k - mat [ m ] [ n ] ) ; return dp [ m ] [ n ] [ k ] ; }
int pathCountDP ( int mat [ ] [ C ] , int k ) { memset ( dp , -1 , sizeof dp ) ; return pathCountDPRecDP ( mat , R - 1 , C - 1 , k ) ; }
int main ( ) { int k = 12 ; int mat [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 6 , 5 } , { 3 , 2 , 1 } } ; cout << pathCountDP ( mat , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define SIZE  10
void sortMat ( int mat [ SIZE ] [ SIZE ] , int n ) {
int temp [ n * n ] ; int k = 0 ;
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) temp [ k ++ ] = mat [ i ] [ j ] ;
sort ( temp , temp + k ) ;
k = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) mat [ i ] [ j ] = temp [ k ++ ] ; }
void printMat ( int mat [ SIZE ] [ SIZE ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) cout << mat [ i ] [ j ] << " ▁ " ; cout << endl ; } }
int main ( ) { int mat [ SIZE ] [ SIZE ] = { { 5 , 4 , 7 } , { 1 , 3 , 8 } , { 2 , 9 , 6 } } ; int n = 3 ; cout << " Original ▁ Matrix : STRNEWLINE " ; printMat ( mat , n ) ; sortMat ( mat , n ) ; cout << " Matrix After Sorting : " ; printMat ( mat , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void selectionSort ( int arr [ ] , int n ) { int i , j , min_idx ;
for ( i = 0 ; i < n - 1 ; i ++ ) {
min_idx = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
swap ( & arr [ min_idx ] , & arr [ i ] ) ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; }
int main ( ) { int arr [ ] = { 64 , 25 , 12 , 22 , 11 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; selectionSort ( arr , n ) ; cout << " Sorted ▁ array : ▁ STRNEWLINE " ; printArray ( arr , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE void swap ( int * xp , int * yp ) { int temp = * xp ; * xp = * yp ; * yp = temp ; }
void bubbleSort ( int arr [ ] , int n ) { int i , j ; bool swapped ; for ( i = 0 ; i < n - 1 ; i ++ ) { swapped = false ; for ( j = 0 ; j < n - i - 1 ; j ++ ) { if ( arr [ j ] > arr [ j + 1 ] ) {
swap ( & arr [ j ] , & arr [ j + 1 ] ) ; swapped = true ; } }
if ( swapped == false ) break ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " n " ) ; }
int main ( ) { int arr [ ] = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; bubbleSort ( arr , n ) ; printf ( " Sorted ▁ array : ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; return 0 ; }
#include <stdio.h>
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
while ( l >= 0 && r < n && count < k ) { if ( x - arr [ l ] < arr [ r ] - x ) printf ( " % d ▁ " , arr [ l -- ] ) ; else printf ( " % d ▁ " , arr [ r ++ ] ) ; count ++ ; }
while ( count < k && l >= 0 ) printf ( " % d ▁ " , arr [ l -- ] ) , count ++ ;
while ( count < k && r < n ) printf ( " % d ▁ " , arr [ r ++ ] ) , count ++ ; }
int main ( ) { int arr [ ] = { 12 , 16 , 22 , 30 , 35 , 39 , 42 , 45 , 48 , 50 , 53 , 55 , 56 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 35 , k = 4 ; printKclosest ( arr , x , 4 , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE int count ( int S [ ] , int m , int n ) {
int table [ n + 1 ] ; memset ( table , 0 , sizeof ( table ) ) ;
table [ 0 ] = 1 ;
for ( int i = 0 ; i < m ; i ++ ) for ( int j = S [ i ] ; j <= n ; j ++ ) table [ j ] += table [ j - S [ i ] ] ; return table [ n ] ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int dp [ 100 ] [ 100 ] ;
int matrixChainMemoised ( int * p , int i , int j ) { if ( i == j ) { return 0 ; } if ( dp [ i ] [ j ] != -1 ) { return dp [ i ] [ j ] ; } dp [ i ] [ j ] = INT_MAX ; for ( int k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i ] [ j ] ; } int MatrixChainOrder ( int * p , int n ) { int i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; memset ( dp , -1 , sizeof dp ) ; cout << " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " << MatrixChainOrder ( arr , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MatrixChainOrder ( int p [ ] , int n ) {
int m [ n ] [ n ] ; int i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i ] [ i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; m [ i ] [ j ] = INT_MAX ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i ] [ j ] ) m [ i ] [ j ] = q ; } } } return m [ 1 ] [ n - 1 ] ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " << MatrixChainOrder ( arr , size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h>
int cutRod ( int price [ ] , int n ) { if ( n <= 0 ) return 0 ; int max_val = INT_MIN ;
for ( int i = 0 ; i < n ; i ++ ) max_val = max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) ; return max_val ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ % dn " , cutRod ( arr , size ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h>
int cutRod ( int price [ ] , int n ) { int val [ n + 1 ] ; val [ 0 ] = 0 ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { int max_val = INT_MIN ; for ( j = 0 ; j < i ; j ++ ) max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ % dn " , cutRod ( arr , size ) ) ; getchar ( ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; class GFG {
public : int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; } } ;
int main ( ) { GFG g ; cout << endl << g . multiply ( 5 , -11 ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void SieveOfEratosthenes ( int n ) {
bool prime [ n + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) cout << p << " ▁ " ; }
int main ( ) { int n = 30 ; cout << " Following ▁ are ▁ the ▁ prime ▁ numbers ▁ smaller ▁ " << " ▁ than ▁ or ▁ equal ▁ to ▁ " << n << endl ; SieveOfEratosthenes ( n ) ; return 0 ; }
#include <stdio.h>
int binomialCoeff ( int n , int k ) ; int binomialCoeff ( int n , int k ) { int res = 1 ; if ( k > n - k ) k = n - k ; for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
void printPascal ( int n ) {
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) printf ( " % d ▁ " , binomialCoeff ( line , i ) ) ; printf ( " STRNEWLINE " ) ; } }
int main ( ) { int n = 7 ; printPascal ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printPascal ( int n ) {
int arr [ n ] [ n ] ;
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) {
if ( line == i i == 0 ) arr [ line ] [ i ] = 1 ;
else arr [ line ] [ i ] = arr [ line - 1 ] [ i - 1 ] + arr [ line - 1 ] [ i ] ; cout << arr [ line ] [ i ] << " ▁ " ; } cout << " STRNEWLINE " ; } }
int main ( ) { int n = 5 ; printPascal ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printPascal ( int n ) { for ( int line = 1 ; line <= n ; line ++ ) {
int C = 1 ; for ( int i = 1 ; i <= line ; i ++ ) {
cout << C << " ▁ " ; C = C * ( line - i ) / i ; } cout << " STRNEWLINE " ; } }
int main ( ) { int n = 5 ; printPascal ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int Add ( int x , int y ) {
while ( y != 0 ) {
int carry = x & y ;
x = x ^ y ;
y = carry << 1 ; } return x ; }
int main ( ) { cout << Add ( 15 , 32 ) ; return 0 ; }
#include <stdio.h>
unsigned int getModulo ( unsigned int n , unsigned int d ) { return ( n & ( d - 1 ) ) ; }
int main ( ) { unsigned int n = 6 ;
unsigned int d = 4 ; printf ( " % u ▁ moduo ▁ % u ▁ is ▁ % u " , n , d , getModulo ( n , d ) ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unsigned int countSetBits ( unsigned int n ) { unsigned int count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
int main ( ) { int i = 9 ; cout << countSetBits ( i ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countSetBits ( int n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
int main ( ) {
int n = 9 ;
cout << countSetBits ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int BitsSetTable256 [ 256 ] ;
void initialize ( ) {
BitsSetTable256 [ 0 ] = 0 ; for ( int i = 0 ; i < 256 ; i ++ ) { BitsSetTable256 [ i ] = ( i & 1 ) + BitsSetTable256 [ i / 2 ] ; } }
int countSetBits ( int n ) { return ( BitsSetTable256 [ n & 0xff ] + BitsSetTable256 [ ( n >> 8 ) & 0xff ] + BitsSetTable256 [ ( n >> 16 ) & 0xff ] + BitsSetTable256 [ n >> 24 ] ) ; }
int main ( ) {
initialize ( ) ; int n = 9 ; cout << countSetBits ( n ) ; }
#include <iostream>
using namespace std ; int main ( ) { cout << __builtin_popcount ( 4 ) << endl ; cout << __builtin_popcount ( 15 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int num_to_bits [ 16 ] = { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
unsigned int countSetBitsRec ( unsigned int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
int main ( ) { int num = 31 ; cout << countSetBitsRec ( num ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int countSetBits ( int N ) { int count = 0 ;
for ( int i = 0 ; i < sizeof ( int ) * 8 ; i ++ ) { if ( N & ( 1 << i ) ) count ++ ; } return count ; }
int main ( ) { int N = 15 ; cout << countSetBits ( N ) << endl ; return 0 ; }
# include <bits/stdc++.h> NEW_LINE # define bool  int NEW_LINE using namespace std ;
bool getParity ( unsigned int n ) { bool parity = 0 ; while ( n ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
int main ( ) { unsigned int n = 7 ; cout << " Parity ▁ of ▁ no ▁ " << n << " ▁ = ▁ " << ( getParity ( n ) ? " odd " : " even " ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes " << endl : cout << " No " << endl ; isPowerOfTwo ( 64 ) ? cout << " Yes " << endl : cout << " No " << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 2 != 0 ) return 0 ; n = n / 2 ; } return 1 ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerOfTwo ( 64 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool powerOf2 ( int n ) {
if ( n == 1 ) return true ;
else if ( n % 2 != 0 n == 0 ) return false ;
return powerOf2 ( n / 2 ) ; }
int main ( ) {
int n = 64 ;
int m = 12 ; if ( powerOf2 ( n ) == 1 ) cout << " True " << endl ; else cout << " False " << endl ; if ( powerOf2 ( m ) == 1 ) cout << " True " << endl ; else cout << " False " << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define bool  int
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerOfTwo ( 64 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int maxRepeating ( int * arr , int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) arr [ arr [ i ] % k ] += k ;
int max = arr [ 0 ] , result = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; result = i ; } }
return result ; }
int main ( ) { int arr [ ] = { 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 8 ; cout << " The ▁ maximum ▁ repeating ▁ number ▁ is ▁ " << maxRepeating ( arr , n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int fun ( int x ) { int y = ( x / 4 ) * 4 ;
int ans = 0 ; for ( int i = y ; i <= x ; i ++ ) ans ^= i ; return ans ; }
int query ( int x ) {
if ( x == 0 ) return 0 ; int k = ( x + 1 ) / 2 ;
return ( x %= 2 ) ? 2 * fun ( k ) : ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) ; } void allQueries ( int q , int l [ ] , int r [ ] ) { for ( int i = 0 ; i < q ; i ++ ) cout << ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) << endl ; }
int main ( ) { int q = 3 ; int l [ ] = { 2 , 2 , 5 } ; int r [ ] = { 4 , 8 , 9 } ; allQueries ( q , l , r ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void prefixXOR ( int arr [ ] , int preXOR [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { while ( arr [ i ] % 2 != 1 ) arr [ i ] /= 2 ; preXOR [ i ] = arr [ i ] ; }
for ( int i = 1 ; i < n ; i ++ ) preXOR [ i ] = preXOR [ i - 1 ] ^ preXOR [ i ] ; }
int query ( int preXOR [ ] , int l , int r ) { if ( l == 0 ) return preXOR [ r ] ; else return preXOR [ r ] ^ preXOR [ l - 1 ] ; }
int main ( ) { int arr [ ] = { 3 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int preXOR [ n ] ; prefixXOR ( arr , preXOR , n ) ; cout << query ( preXOR , 0 , 2 ) << endl ; cout << query ( preXOR , 1 , 2 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinSwaps ( int arr [ ] , int n ) {
int noOfZeroes [ n ] ; memset ( noOfZeroes , 0 , sizeof ( noOfZeroes ) ) ; int i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
int main ( ) { int arr [ ] = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMinSwaps ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int minswaps ( int arr [ ] , int n ) { int count = 0 ; int num_unplaced_zeros = 0 ; for ( int index = n - 1 ; index >= 0 ; index -- ) { if ( arr [ index ] == 0 ) num_unplaced_zeros += 1 ; else count += num_unplaced_zeros ; } return count ; }
int main ( ) { int arr [ ] = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; cout << minswaps ( arr , 9 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool arraySortedOrNot ( int arr [ ] , int n ) {
if ( n == 0 n == 1 ) return true ; for ( int i = 1 ; i < n ; i ++ )
if ( arr [ i - 1 ] > arr [ i ] ) return false ;
return true ; }
int main ( ) { int arr [ ] = { 20 , 23 , 23 , 45 , 78 , 88 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; if ( arraySortedOrNot ( arr , n ) ) cout << " Yes STRNEWLINE " ; else cout << " No STRNEWLINE " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printTwoOdd ( int arr [ ] , int size ) { int xor2 = arr [ 0 ] ;
int set_bit_no ;
int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } cout << " The ▁ two ▁ ODD ▁ elements ▁ are ▁ " << x << " ▁ & ▁ " << y ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoOdd ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool findPair ( int arr [ ] , int size , int n ) {
int i = 0 ; int j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { cout << " Pair ▁ Found : ▁ ( " << arr [ i ] << " , ▁ " << arr [ j ] << " ) " ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } cout << " No ▁ such ▁ pair " ; return false ; }
int main ( ) { int arr [ ] = { 1 , 8 , 30 , 40 , 100 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 60 ; findPair ( arr , size , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printMax ( int arr [ ] , int k , int n ) {
vector < int > brr ( arr , arr + n ) ;
sort ( brr . begin ( ) , brr . end ( ) , greater < int > ( ) ) ;
for ( int i = 0 ; i < n ; ++ i ) if ( binary_search ( brr . begin ( ) , brr . begin ( ) + k , arr [ i ] , greater < int > ( ) ) ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 50 , 8 , 45 , 12 , 25 , 40 , 84 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 3 ; printMax ( arr , k , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printSmall ( int arr [ ] , int asize , int n ) {
vector < int > copy_arr ( arr , arr + asize ) ;
sort ( copy_arr . begin ( ) , copy_arr . begin ( ) + asize ) ;
for ( int i = 0 ; i < asize ; ++ i ) if ( binary_search ( copy_arr . begin ( ) , copy_arr . begin ( ) + n , arr [ i ] ) ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 } ; int asize = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 5 ; printSmall ( arr , asize , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkIsAP ( int arr [ ] , int n ) { if ( n == 1 ) return true ;
sort ( arr , arr + n ) ;
int d = arr [ 1 ] - arr [ 0 ] ; for ( int i = 2 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] != d ) return false ; return true ; }
int main ( ) { int arr [ ] = { 20 , 15 , 5 , 0 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; ( checkIsAP ( arr , n ) ) ? ( cout << " Yes " << endl ) : ( cout << " No " << endl ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countPairs ( int a [ ] , int n ) {
int mn = INT_MAX ; int mx = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) { mn = min ( mn , a [ i ] ) ; mx = max ( mx , a [ i ] ) ; }
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == mn ) c1 ++ ; if ( a [ i ] == mx ) c2 ++ ; }
if ( mn == mx ) return n * ( n - 1 ) / 2 ; else return c1 * c2 ; }
int main ( ) { int a [ ] = { 3 , 2 , 1 , 1 , 3 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << countPairs ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; struct node { int data ; struct node * next ; } ; typedef struct node Node ;
void rearrange ( Node * head ) {
if ( head == NULL ) return ;
Node * prev = head , * curr = head -> next ; while ( curr ) {
if ( prev -> data > curr -> data ) swap ( prev -> data , curr -> data ) ;
if ( curr -> next && curr -> next -> data > curr -> data ) swap ( curr -> next -> data , curr -> data ) ; prev = curr -> next ; if ( ! curr -> next ) break ; curr = curr -> next -> next ; } }
void push ( Node * * head , int k ) { Node * tem = ( Node * ) malloc ( sizeof ( Node ) ) ; tem -> data = k ; tem -> next = * head ; * head = tem ; }
void display ( Node * head ) { Node * curr = head ; while ( curr != NULL ) { printf ( " % d ▁ " , curr -> data ) ; curr = curr -> next ; } }
int main ( ) { Node * head = NULL ;
push ( & head , 7 ) ; push ( & head , 3 ) ; push ( & head , 8 ) ; push ( & head , 6 ) ; push ( & head , 9 ) ; rearrange ( head ) ; display ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * next ; } ; Node * newNode ( int data ) { Node * temp = new Node ; temp -> data = data ; temp -> next = NULL ; return temp ; }
int getLength ( Node * Node ) { int size = 0 ; while ( Node != NULL ) { Node = Node -> next ; size ++ ; } return size ; }
Node * paddZeros ( Node * sNode , int diff ) { if ( sNode == NULL ) return NULL ; Node * zHead = newNode ( 0 ) ; diff -- ; Node * temp = zHead ; while ( diff -- ) { temp -> next = newNode ( 0 ) ; temp = temp -> next ; } temp -> next = sNode ; return zHead ; }
Node * subtractLinkedListHelper ( Node * l1 , Node * l2 , bool & borrow ) { if ( l1 == NULL && l2 == NULL && borrow == 0 ) return NULL ; Node * previous = subtractLinkedListHelper ( l1 ? l1 -> next : NULL , l2 ? l2 -> next : NULL , borrow ) ; int d1 = l1 -> data ; int d2 = l2 -> data ; int sub = 0 ;
if ( borrow ) { d1 -- ; borrow = false ; }
if ( d1 < d2 ) { borrow = true ; d1 = d1 + 10 ; }
sub = d1 - d2 ;
Node * current = newNode ( sub ) ;
current -> next = previous ; return current ; }
Node * subtractLinkedList ( Node * l1 , Node * l2 ) {
if ( l1 == NULL && l2 == NULL ) return NULL ;
int len1 = getLength ( l1 ) ; int len2 = getLength ( l2 ) ; Node * lNode = NULL , * sNode = NULL ; Node * temp1 = l1 ; Node * temp2 = l2 ;
if ( len1 != len2 ) { lNode = len1 > len2 ? l1 : l2 ; sNode = len1 > len2 ? l2 : l1 ; sNode = paddZeros ( sNode , abs ( len1 - len2 ) ) ; } else {
while ( l1 && l2 ) { if ( l1 -> data != l2 -> data ) { lNode = l1 -> data > l2 -> data ? temp1 : temp2 ; sNode = l1 -> data > l2 -> data ? temp2 : temp1 ; break ; } l1 = l1 -> next ; l2 = l2 -> next ; } }
bool borrow = false ; return subtractLinkedListHelper ( lNode , sNode , borrow ) ; }
void printList ( struct Node * Node ) { while ( Node != NULL ) { printf ( " % d ▁ " , Node -> data ) ; Node = Node -> next ; } printf ( " STRNEWLINE " ) ; }
int main ( ) { Node * head1 = newNode ( 1 ) ; head1 -> next = newNode ( 0 ) ; head1 -> next -> next = newNode ( 0 ) ; Node * head2 = newNode ( 1 ) ; Node * result = subtractLinkedList ( head1 , head2 ) ; printList ( result ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; Node * next ; } ;
Node * getNode ( int data ) { Node * newNode = ( Node * ) malloc ( sizeof ( Node ) ) ; newNode -> data = data ; newNode -> next = NULL ; return newNode ; }
void insertAtMid ( Node * * head_ref , int x ) {
if ( * head_ref == NULL ) * head_ref = getNode ( x ) ; else {
Node * newNode = getNode ( x ) ; Node * ptr = * head_ref ; int len = 0 ;
while ( ptr != NULL ) { len ++ ; ptr = ptr -> next ; }
int count = ( ( len % 2 ) == 0 ) ? ( len / 2 ) : ( len + 1 ) / 2 ; ptr = * head_ref ;
while ( count -- > 1 ) ptr = ptr -> next ;
newNode -> next = ptr -> next ; ptr -> next = newNode ; } }
void display ( Node * head ) { while ( head != NULL ) { cout << head -> data << " ▁ " ; head = head -> next ; } }
int main ( ) {
Node * head = NULL ; head = getNode ( 1 ) ; head -> next = getNode ( 2 ) ; head -> next -> next = getNode ( 4 ) ; head -> next -> next -> next = getNode ( 5 ) ; cout << " Linked ▁ list ▁ before ▁ insertion : ▁ " ; display ( head ) ; int x = 3 ; insertAtMid ( & head , x ) ; cout << " Linked list after insertion : " display ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; Node * next ; } ; Node * getNode ( int data ) { Node * newNode = ( Node * ) malloc ( sizeof ( Node ) ) ; newNode -> data = data ; newNode -> next = NULL ; return newNode ; }
void insertAtMid ( Node * * head_ref , int x ) {
if ( * head_ref == NULL ) * head_ref = getNode ( x ) ; else {
Node * newNode = getNode ( x ) ;
Node * slow = * head_ref ; Node * fast = ( * head_ref ) -> next ; while ( fast && fast -> next ) {
slow = slow -> next ;
fast = fast -> next -> next ; }
newNode -> next = slow -> next ; slow -> next = newNode ; } }
void display ( Node * head ) { while ( head != NULL ) { cout << head -> data << " ▁ " ; head = head -> next ; } }
int main ( ) {
Node * head = NULL ; head = getNode ( 1 ) ; head -> next = getNode ( 2 ) ; head -> next -> next = getNode ( 4 ) ; head -> next -> next -> next = getNode ( 5 ) ; cout << " Linked ▁ list ▁ before ▁ insertion : ▁ " ; display ( head ) ; int x = 3 ; insertAtMid ( & head , x ) ; cout << " Linked list after insertion : " display ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * prev , * next ; } ;
struct Node * getNode ( int data ) {
struct Node * newNode = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ;
newNode -> data = data ; newNode -> prev = newNode -> next = NULL ; return newNode ; }
void sortedInsert ( struct Node * * head_ref , struct Node * newNode ) { struct Node * current ;
if ( * head_ref == NULL ) * head_ref = newNode ;
else if ( ( * head_ref ) -> data >= newNode -> data ) { newNode -> next = * head_ref ; newNode -> next -> prev = newNode ; * head_ref = newNode ; } else { current = * head_ref ;
while ( current -> next != NULL && current -> next -> data < newNode -> data ) current = current -> next ;
newNode -> next = current -> next ;
if ( current -> next != NULL ) newNode -> next -> prev = newNode ; current -> next = newNode ; newNode -> prev = current ; } }
void insertionSort ( struct Node * * head_ref ) {
struct Node * sorted = NULL ;
struct Node * current = * head_ref ; while ( current != NULL ) {
struct Node * next = current -> next ;
current -> prev = current -> next = NULL ;
sortedInsert ( & sorted , current ) ;
current = next ; }
* head_ref = sorted ; }
void printList ( struct Node * head ) { while ( head != NULL ) { cout << head -> data << " ▁ " ; head = head -> next ; } }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ;
new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ; new_node -> prev = NULL ;
if ( ( * head_ref ) != NULL ) ( * head_ref ) -> prev = new_node ;
( * head_ref ) = new_node ; }
int main ( ) {
struct Node * head = NULL ;
push ( & head , 9 ) ; push ( & head , 3 ) ; push ( & head , 5 ) ; push ( & head , 10 ) ; push ( & head , 12 ) ; push ( & head , 8 ) ; cout << " Doubly ▁ Linked ▁ List ▁ Before ▁ Sortingn " ; printList ( head ) ; insertionSort ( & head ) ; cout << " nDoubly ▁ Linked ▁ List ▁ After ▁ Sortingn " ; printList ( head ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int arr [ 10000 ] ;
void reverse ( int arr [ ] , int s , int e ) { while ( s < e ) { int tem = arr [ s ] ; arr [ s ] = arr [ e ] ; arr [ e ] = tem ; s = s + 1 ; e = e - 1 ; } }
void fun ( int arr [ ] , int k ) { int n = 4 - 1 ; int v = n - k ; if ( v >= 0 ) { reverse ( arr , 0 , v ) ; reverse ( arr , v + 1 , n ) ; reverse ( arr , 0 , n ) ; } }
int main ( ) { arr [ 0 ] = 1 ; arr [ 1 ] = 2 ; arr [ 2 ] = 3 ; arr [ 3 ] = 4 ; for ( int i = 0 ; i < 4 ; i ++ ) { fun ( arr , i ) ; cout << ( " [ " ) ; for ( int j = 0 ; j < 4 ; j ++ ) { cout << ( arr [ j ] ) << " , ▁ " ; } cout << ( " ] " ) ; } }
#include <bits/stdc++.h> NEW_LINE const int MAX = 100005 ; using namespace std ;
int seg [ 4 * MAX ] ;
void build ( int node , int l , int r , int a [ ] ) { if ( l == r ) seg [ node ] = a [ l ] ; else { int mid = ( l + r ) / 2 ; build ( 2 * node , l , mid , a ) ; build ( 2 * node + 1 , mid + 1 , r , a ) ; seg [ node ] = ( seg [ 2 * node ] seg [ 2 * node + 1 ] ) ; } }
int query ( int node , int l , int r , int start , int end , int a [ ] ) {
if ( l > end or r < start ) return 0 ; if ( start <= l and r <= end ) return seg [ node ] ;
int mid = ( l + r ) / 2 ;
return ( ( query ( 2 * node , l , mid , start , end , a ) ) | ( query ( 2 * node + 1 , mid + 1 , r , start , end , a ) ) ) ; }
void orsum ( int a [ ] , int n , int q , int k [ ] ) {
build ( 1 , 0 , n - 1 , a ) ;
for ( int j = 0 ; j < q ; j ++ ) {
int i = k [ j ] % ( n / 2 ) ;
int sec = query ( 1 , 0 , n - 1 , n / 2 - i , n - i - 1 , a ) ;
int first = ( query ( 1 , 0 , n - 1 , 0 , n / 2 - 1 - i , a ) | query ( 1 , 0 , n - 1 , n - i , n - 1 , a ) ) ; int temp = sec + first ;
cout << temp << endl ; } }
int main ( ) { int a [ ] = { 7 , 44 , 19 , 86 , 65 , 39 , 75 , 101 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int q = 2 ; int k [ q ] = { 4 , 2 } ; orsum ( a , n , q , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maximumEqual ( int a [ ] , int b [ ] , int n ) {
vector < int > store ( 1e5 ) ;
for ( int i = 0 ; i < n ; i ++ ) { store [ b [ i ] ] = i + 1 ; }
vector < int > ans ( 1e5 ) ;
for ( int i = 0 ; i < n ; i ++ ) {
int d = abs ( store [ a [ i ] ] - ( i + 1 ) ) ;
if ( store [ a [ i ] ] < i + 1 ) { d = n - d ; }
ans [ d ] ++ ; } int finalans = 0 ;
for ( int i = 0 ; i < 1e5 ; i ++ ) finalans = max ( finalans , ans [ i ] ) ;
cout << finalans << " STRNEWLINE " ; }
int main ( ) {
int A [ ] = { 6 , 7 , 3 , 9 , 5 } ; int B [ ] = { 7 , 3 , 9 , 5 , 6 } ; int size = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
maximumEqual ( A , B , size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void RightRotate ( int a [ ] , int n , int k ) {
k = k % n ; for ( int i = 0 ; i < n ; i ++ ) { if ( i < k ) {
cout << a [ n + i - k ] << " ▁ " ; } else {
cout << ( a [ i - k ] ) << " ▁ " ; } } cout << " STRNEWLINE " ; }
int main ( ) { int Array [ ] = { 1 , 2 , 3 , 4 , 5 } ; int N = sizeof ( Array ) / sizeof ( Array [ 0 ] ) ; int K = 2 ; RightRotate ( Array , N , K ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void restoreSortedArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > arr [ i + 1 ] ) {
reverse ( arr , arr + i + 1 ) ; reverse ( arr + i + 1 , arr + n ) ; reverse ( arr , arr + n ) ; } } }
void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; restoreSortedArray ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findStartIndexOfArray ( int arr [ ] , int low , int high ) { if ( low > high ) { return -1 ; } if ( low == high ) { return low ; } int mid = low + ( high - low ) / 2 ; if ( arr [ mid ] > arr [ mid + 1 ] ) return mid + 1 ; if ( arr [ mid - 1 ] > arr [ mid ] ) return mid ; if ( arr [ low ] > arr [ mid ] ) return findStartIndexOfArray ( arr , low , mid - 1 ) ; else return findStartIndexOfArray ( arr , mid + 1 , high ) ; }
void restoreSortedArray ( int arr [ ] , int n ) {
if ( arr [ 0 ] < arr [ n - 1 ] ) return ; int start = findStartIndexOfArray ( arr , 0 , n - 1 ) ;
reverse ( arr , arr + start ) ; reverse ( arr + start , arr + n ) ; reverse ( arr , arr + n ) ; }
void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; restoreSortedArray ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void leftrotate ( string & s , int d ) { reverse ( s . begin ( ) , s . begin ( ) + d ) ; reverse ( s . begin ( ) + d , s . end ( ) ) ; reverse ( s . begin ( ) , s . end ( ) ) ; }
void rightrotate ( string & s , int d ) { leftrotate ( s , s . length ( ) - d ) ; }
int main ( ) { string str1 = " GeeksforGeeks " ; leftrotate ( str1 , 2 ) ; cout << str1 << endl ; string str2 = " GeeksforGeeks " ; rightrotate ( str2 , 2 ) ; cout << str2 << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * next ; struct Node * prev ; } ;
void insertNode ( struct Node * * start , int value ) {
if ( * start == NULL ) { struct Node * new_node = new Node ; new_node -> data = value ; new_node -> next = new_node -> prev = new_node ; * start = new_node ; return ; }
Node * last = ( * start ) -> prev ;
struct Node * new_node = new Node ; new_node -> data = value ;
new_node -> next = * start ;
( * start ) -> prev = new_node ;
new_node -> prev = last ;
last -> next = new_node ; }
void displayList ( struct Node * start ) { struct Node * temp = start ; while ( temp -> next != start ) { printf ( " % d ▁ " , temp -> data ) ; temp = temp -> next ; } printf ( " % d ▁ " , temp -> data ) ; }
int searchList ( struct Node * start , int search ) {
struct Node * temp = start ;
int count = 0 , flag = 0 , value ;
if ( temp == NULL ) return -1 ; else {
while ( temp -> next != start ) {
count ++ ;
if ( temp -> data == search ) { flag = 1 ; count -- ; break ; }
temp = temp -> next ; }
if ( temp -> data == search ) { count ++ ; flag = 1 ; }
if ( flag == 1 ) cout << " STRNEWLINE " << search << " ▁ found ▁ at ▁ location ▁ " << count << endl ; else cout << " STRNEWLINE " << search << " ▁ not ▁ found " << endl ; } }
int main ( ) {
struct Node * start = NULL ;
insertNode ( & start , 4 ) ;
insertNode ( & start , 5 ) ;
insertNode ( & start , 7 ) ;
insertNode ( & start , 8 ) ;
insertNode ( & start , 6 ) ; printf ( " Created ▁ circular ▁ doubly ▁ linked ▁ list ▁ is : ▁ " ) ; displayList ( start ) ; searchList ( start , 5 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; Node * next , * prev ; } ;
Node * getNode ( int data ) { Node * newNode = ( Node * ) malloc ( sizeof ( Node ) ) ; newNode -> data = data ; return newNode ; }
void insertEnd ( Node * * head , Node * new_node ) {
if ( * head == NULL ) { new_node -> next = new_node -> prev = new_node ; * head = new_node ; return ; }
Node * last = ( * head ) -> prev ;
new_node -> next = * head ;
( * head ) -> prev = new_node ;
new_node -> prev = last ;
last -> next = new_node ; }
Node * reverse ( Node * head ) { if ( ! head ) return NULL ;
Node * new_head = NULL ;
Node * last = head -> prev ;
Node * curr = last , * prev ;
while ( curr -> prev != last ) { prev = curr -> prev ;
insertEnd ( & new_head , curr ) ; curr = prev ; } insertEnd ( & new_head , curr ) ;
return new_head ; }
void display ( Node * head ) { if ( ! head ) return ; Node * temp = head ; cout << " Forward ▁ direction : ▁ " ; while ( temp -> next != head ) { cout << temp -> data << " ▁ " ; temp = temp -> next ; } cout << temp -> data ; Node * last = head -> prev ; temp = last ; cout << " Backward direction : " while ( temp -> prev != last ) { cout << temp -> data << " ▁ " ; temp = temp -> prev ; } cout << temp -> data ; }
int main ( ) { Node * head = NULL ; insertEnd ( & head , getNode ( 1 ) ) ; insertEnd ( & head , getNode ( 2 ) ) ; insertEnd ( & head , getNode ( 3 ) ) ; insertEnd ( & head , getNode ( 4 ) ) ; insertEnd ( & head , getNode ( 5 ) ) ; cout << " Current ▁ list : STRNEWLINE " ; display ( head ) ; head = reverse ( head ) ; cout << " Reversed list : " display ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAXN  1001
int depth [ MAXN ] ;
int parent [ MAXN ] ; vector < int > adj [ MAXN ] ; void addEdge ( int u , int v ) { adj [ u ] . push_back ( v ) ; adj [ v ] . push_back ( u ) ; } void dfs ( int cur , int prev ) {
parent [ cur ] = prev ;
depth [ cur ] = depth [ prev ] + 1 ;
for ( int i = 0 ; i < adj [ cur ] . size ( ) ; i ++ ) if ( adj [ cur ] [ i ] != prev ) dfs ( adj [ cur ] [ i ] , cur ) ; } void preprocess ( ) {
depth [ 0 ] = -1 ;
dfs ( 1 , 0 ) ; }
int LCANaive ( int u , int v ) { if ( u == v ) return u ; if ( depth [ u ] > depth [ v ] ) swap ( u , v ) ; v = parent [ v ] ; return LCANaive ( u , v ) ; }
int main ( int argc , char const * argv [ ] ) {
addEdge ( 1 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 1 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 2 , 6 ) ; addEdge ( 3 , 7 ) ; addEdge ( 4 , 8 ) ; addEdge ( 4 , 9 ) ; addEdge ( 9 , 10 ) ; addEdge ( 9 , 11 ) ; addEdge ( 7 , 12 ) ; addEdge ( 7 , 13 ) ; preprocess ( ) ; cout << " LCA ( 11,8 ) ▁ : ▁ " << LCANaive ( 11 , 8 ) << endl ; cout << " LCA ( 3,13 ) ▁ : ▁ " << LCANaive ( 3 , 13 ) << endl ; return 0 ; }
#include " iostream " NEW_LINE #include " vector " NEW_LINE #include " math . h " NEW_LINE using namespace std ; #define MAXN  1001
int block_sz ;
int depth [ MAXN ] ;
int parent [ MAXN ] ;
int jump_parent [ MAXN ] ; vector < int > adj [ MAXN ] ; void addEdge ( int u , int v ) { adj [ u ] . push_back ( v ) ; adj [ v ] . push_back ( u ) ; } int LCANaive ( int u , int v ) { if ( u == v ) return u ; if ( depth [ u ] > depth [ v ] ) swap ( u , v ) ; v = parent [ v ] ; return LCANaive ( u , v ) ; }
void dfs ( int cur , int prev ) {
depth [ cur ] = depth [ prev ] + 1 ;
parent [ cur ] = prev ;
if ( depth [ cur ] % block_sz == 0 )
jump_parent [ cur ] = parent [ cur ] ; else
jump_parent [ cur ] = jump_parent [ prev ] ;
for ( int i = 0 ; i < adj [ cur ] . size ( ) ; ++ i ) if ( adj [ cur ] [ i ] != prev ) dfs ( adj [ cur ] [ i ] , cur ) ; }
int LCASQRT ( int u , int v ) { while ( jump_parent [ u ] != jump_parent [ v ] ) { if ( depth [ u ] > depth [ v ] )
swap ( u , v ) ;
v = jump_parent [ v ] ; }
return LCANaive ( u , v ) ; } void preprocess ( int height ) { block_sz = sqrt ( height ) ; depth [ 0 ] = -1 ;
dfs ( 1 , 0 ) ; }
int main ( int argc , char const * argv [ ] ) {
addEdge ( 1 , 2 ) ; addEdge ( 1 , 3 ) ; addEdge ( 1 , 4 ) ; addEdge ( 2 , 5 ) ; addEdge ( 2 , 6 ) ; addEdge ( 3 , 7 ) ; addEdge ( 4 , 8 ) ; addEdge ( 4 , 9 ) ; addEdge ( 9 , 10 ) ; addEdge ( 9 , 11 ) ; addEdge ( 7 , 12 ) ; addEdge ( 7 , 13 ) ;
int height = 4 ; preprocess ( height ) ; cout << " LCA ( 11,8 ) ▁ : ▁ " << LCASQRT ( 11 , 8 ) << endl ; cout << " LCA ( 3,13 ) ▁ : ▁ " << LCASQRT ( 3 , 13 ) << endl ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE using namespace std ;
int main ( ) { int N = 3 ;
cout << pow ( 2 , N + 1 ) - 2 ; return 0 ; }
#include <algorithm> NEW_LINE #include <iostream> NEW_LINE #include <set> NEW_LINE #define ll  long long NEW_LINE using namespace std ;
ll int countOfNum ( ll int n , ll int a , ll int b ) { ll int cnt_of_a , cnt_of_b , cnt_of_ab , sum ;
cnt_of_a = n / a ;
cnt_of_b = n / b ;
sum = cnt_of_b + cnt_of_a ;
cnt_of_ab = n / ( a * b ) ;
sum = sum - cnt_of_ab ; return sum ; }
ll int sumOfNum ( ll int n , ll int a , ll int b ) { ll int i ; ll int sum = 0 ;
set < ll int > ans ;
for ( i = a ; i <= n ; i = i + a ) { ans . insert ( i ) ; }
for ( i = b ; i <= n ; i = i + b ) { ans . insert ( i ) ; }
for ( auto it = ans . begin ( ) ; it != ans . end ( ) ; it ++ ) { sum = sum + * it ; } return sum ; }
int main ( ) { ll int N = 88 ; ll int A = 11 ; ll int B = 8 ; ll int count = countOfNum ( N , A , B ) ; ll int sumofnum = sumOfNum ( N , A , B ) ; cout << sumofnum % count << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double get ( double L , double R ) {
double x = 1.0 / L ;
double y = 1.0 / ( R + 1.0 ) ; return ( x - y ) ; }
int main ( ) { int L = 6 , R = 12 ;
double ans = get ( L , R ) ; cout << fixed << setprecision ( 2 ) << ans ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 100000 ;
vector < int > v ;
int consecutiveOnes ( int x ) {
int p = 0 ; while ( x > 0 ) {
if ( x % 2 == 1 and p == 1 ) return true ;
p = x % 2 ;
x /= 2 ; } return false ; }
void preCompute ( ) {
for ( int i = 0 ; i <= MAX ; i ++ ) { if ( ! consecutiveOnes ( i ) ) v . push_back ( i ) ; } }
int nextValid ( int n ) {
int it = upper_bound ( v . begin ( ) , v . end ( ) , n ) - v . begin ( ) ; int val = v [ it ] ; return val ; }
void performQueries ( int queries [ ] , int q ) { for ( int i = 0 ; i < q ; i ++ ) cout << nextValid ( queries [ i ] ) << " STRNEWLINE " ; }
int main ( ) { int queries [ ] = { 4 , 6 } ; int q = sizeof ( queries ) / sizeof ( int ) ;
preCompute ( ) ;
performQueries ( queries , q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int changeToOnes ( string str ) {
int i , l , ctr = 0 ; l = str . length ( ) ;
for ( i = l - 1 ; i >= 0 ; i -- ) {
if ( str [ i ] == '1' ) ctr ++ ;
else break ; }
return l - ctr ; }
string removeZeroesFromFront ( string str ) { string s ; int i = 0 ;
while ( i < str . length ( ) && str [ i ] == '0' ) i ++ ;
if ( i == str . length ( ) ) s = "0" ;
else s = str . substr ( i , str . length ( ) - i ) ; return s ; }
int main ( ) { string str = "10010111" ;
str = removeZeroesFromFront ( str ) ; cout << changeToOnes ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MinDeletion ( int a [ ] , int n ) {
unordered_map < int , int > map ;
for ( int i = 0 ; i < n ; i ++ ) map [ a [ i ] ] ++ ;
int ans = 0 ; for ( auto i : map ) {
int x = i . first ;
int frequency = i . second ;
if ( x <= frequency ) {
ans += ( frequency - x ) ; }
else ans += frequency ; } return ans ; }
int main ( ) { int a [ ] = { 2 , 3 , 2 , 3 , 4 , 4 , 4 , 4 , 5 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << MinDeletion ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxCountAB ( string s [ ] , int n ) {
int A = 0 , B = 0 , BA = 0 , ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) { string S = s [ i ] ; int L = S . size ( ) ; for ( int j = 0 ; j < L - 1 ; j ++ ) {
if ( S . at ( j ) == ' A ' && S . at ( j + 1 ) == ' B ' ) { ans ++ ; } }
if ( S . at ( 0 ) == ' B ' && S . at ( L - 1 ) == ' A ' ) BA ++ ;
else if ( S . at ( 0 ) == ' B ' ) B ++ ;
else if ( S . at ( L - 1 ) == ' A ' ) A ++ ; }
if ( BA == 0 ) ans += min ( B , A ) ; else if ( A + B == 0 ) ans += BA - 1 ; else ans += BA + min ( B , A ) ; return ans ; }
int main ( ) { string s [ ] = { " ABCA " , " BOOK " , " BAND " } ; int n = sizeof ( s ) / sizeof ( s [ 0 ] ) ; cout << maxCountAB ( s , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MinOperations ( int n , int x , int * arr ) {
int total = 0 ; for ( int i = 0 ; i < n ; ++ i ) {
if ( arr [ i ] > x ) { int difference = arr [ i ] - x ; total = total + difference ; arr [ i ] = x ; } }
for ( int i = 1 ; i < n ; ++ i ) { int LeftNeigbouringSum = arr [ i ] + arr [ i - 1 ] ;
if ( LeftNeigbouringSum > x ) { int current_diff = LeftNeigbouringSum - x ; arr [ i ] = max ( 0 , arr [ i ] - current_diff ) ; total = total + current_diff ; } } return total ; }
int main ( ) { int X = 1 ; int arr [ ] = { 1 , 6 , 1 , 2 , 0 , 4 } ; int N = sizeof ( arr ) / sizeof ( int ) ; cout << MinOperations ( N , X , arr ) ; return 0 ; }
#include <cmath> NEW_LINE #include <bits/stdc++.h> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
void findNumbers ( int arr [ ] , int n ) {
int sumN = ( n * ( n + 1 ) ) / 2 ;
int sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
int sum = 0 , sumSq = 0 , i ; for ( i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq = sumSq + ( pow ( arr [ i ] , 2 ) ) ; } int B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; int A = sum - sumN + B ; cout << " A ▁ = ▁ " ; cout << A << endl ; cout << " B ▁ = ▁ " ; cout << B << endl ; }
int main ( ) { int arr [ ] = { 1 , 2 , 2 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findNumbers ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool is_prefix ( string temp , string str ) {
if ( temp . length ( ) < str . length ( ) ) return 0 ; else {
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str [ i ] != temp [ i ] ) return 0 ; } return 1 ; } }
string lexicographicallyString ( string input [ ] , int n , string str ) {
sort ( input , input + n ) ; for ( int i = 0 ; i < n ; i ++ ) { string temp = input [ i ] ;
if ( is_prefix ( temp , str ) ) { return temp ; } }
return " - 1" ; }
int main ( ) { string arr [ ] = { " apple " , " appe " , " apl " , " aapl " , " appax " } ; string S = " app " ; int N = 5 ; cout << lexicographicallyString ( arr , N , S ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void Rearrange ( int arr [ ] , int K , int N ) {
int ans [ N + 1 ] ;
int f = -1 ; for ( int i = 0 ; i < N ; i ++ ) { ans [ i ] = -1 ; }
K = find ( arr , arr + N , K ) - arr ;
vector < int > smaller , greater ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] < arr [ K ] ) smaller . push_back ( arr [ i ] ) ;
else if ( arr [ i ] > arr [ K ] ) greater . push_back ( arr [ i ] ) ; } int low = 0 , high = N - 1 ;
while ( low <= high ) {
int mid = ( low + high ) / 2 ;
if ( mid == K ) { ans [ mid ] = arr [ K ] ; f = 1 ; break ; }
else if ( mid < K ) { if ( smaller . size ( ) == 0 ) { break ; } ans [ mid ] = smaller . back ( ) ; smaller . pop_back ( ) ; low = mid + 1 ; }
else { if ( greater . size ( ) == 0 ) { break ; } ans [ mid ] = greater . back ( ) ; greater . pop_back ( ) ; high = mid - 1 ; } }
if ( f == -1 ) { cout << -1 << endl ; return ; }
for ( int i = 0 ; i < N ; i ++ ) {
if ( ans [ i ] == -1 ) { if ( smaller . size ( ) ) { ans [ i ] = smaller . back ( ) ; smaller . pop_back ( ) ; } else if ( greater . size ( ) ) { ans [ i ] = greater . back ( ) ; greater . pop_back ( ) ; } } }
for ( int i = 0 ; i < N ; i ++ ) cout << ans [ i ] << " ▁ " ; cout << endl ; }
int main ( ) {
int arr [ ] = { 10 , 7 , 2 , 5 , 3 , 8 } ; int K = 7 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
Rearrange ( arr , K , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minimumK ( vector < int > & arr , int M , int N ) {
int good = ceil ( ( N * 1.0 ) / ( ( M + 1 ) * 1.0 ) ) ;
for ( int i = 1 ; i <= N ; i ++ ) { int K = i ;
int candies = N ;
int taken = 0 ; while ( candies > 0 ) {
taken += min ( K , candies ) ; candies -= min ( K , candies ) ;
for ( int j = 0 ; j < M ; j ++ ) {
int consume = ( arr [ j ] * candies ) / 100 ;
candies -= consume ; } }
if ( taken >= good ) { cout << i ; return ; } } }
int main ( ) { int N = 13 , M = 1 ; vector < int > arr = { 50 } ; minimumK ( arr , M , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool check ( int K , int n , int m , vector < int > arr , int good_share ) { int candies = n , taken = 0 ; while ( candies > 0 ) {
taken += min ( K , candies ) ; candies -= min ( K , candies ) ;
for ( int j = 0 ; j < m ; j ++ ) {
int consume = ( arr [ j ] * candies ) / 100 ;
candies -= consume ; } }
return ( taken >= good_share ) ; }
void minimumK ( vector < int > & arr , int N , int M ) {
int good_share = ceil ( ( N * 1.0 ) / ( ( M + 1 ) * 1.0 ) ) ; int lo = 1 , hi = N ;
while ( lo < hi ) {
int mid = ( lo + hi ) / 2 ;
if ( check ( mid , N , M , arr , good_share ) ) {
hi = mid ; }
else { lo = mid + 1 ; } }
cout << hi ; }
int main ( ) { int N = 13 , M = 1 ; vector < int > arr = { 50 } ; minimumK ( arr , N , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void calcTotalTime ( string path ) {
int time = 0 ;
int x = 0 , y = 0 ;
set < pair < int , int > > s ; for ( int i = 0 ; i < path . size ( ) ; i ++ ) { int p = x ; int q = y ; if ( path [ i ] == ' N ' ) y ++ ; else if ( path [ i ] == ' S ' ) y -- ; else if ( path [ i ] == ' E ' ) x ++ ; else if ( path [ i ] == ' W ' ) x -- ;
if ( s . find ( { p + x , q + y } ) == s . end ( ) ) {
time += 2 ;
s . insert ( { p + x , q + y } ) ; } else time += 1 ; }
cout << time << endl ; }
int main ( ) { string path = " NSE " ; calcTotalTime ( path ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findCost ( int A [ ] , int N ) {
int totalCost = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( A [ i ] == 0 ) {
A [ i ] = 1 ;
totalCost += i ; } }
return totalCost ; }
int main ( ) { int arr [ ] = { 1 , 0 , 1 , 0 , 1 , 0 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findCost ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int peakIndex ( int arr [ ] , int N ) {
if ( N < 3 ) return -1 ; int i = 0 ;
while ( i + 1 < N ) {
if ( arr [ i + 1 ] < arr [ i ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; } if ( i == 0 i == N - 1 ) return -1 ;
int ans = i ;
while ( i < N - 1 ) {
if ( arr [ i ] < arr [ i + 1 ] arr [ i ] == arr [ i + 1 ] ) break ; i ++ ; }
if ( i == N - 1 ) return ans ;
return -1 ; }
int main ( ) { int arr [ ] = { 0 , 1 , 0 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << peakIndex ( arr , N ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void hasArrayTwoPairs ( int nums [ ] , int n , int target ) {
sort ( nums , nums + n ) ;
for ( int i = 0 ; i < n ; i ++ ) {
int x = target - nums [ i ] ;
int low = 0 , high = n - 1 ; while ( low <= high ) {
int mid = low + ( ( high - low ) / 2 ) ;
if ( nums [ mid ] > x ) { high = mid - 1 ; }
else if ( nums [ mid ] < x ) { low = mid + 1 ; }
else {
if ( mid == i ) { if ( ( mid - 1 >= 0 ) && nums [ mid - 1 ] == x ) { cout << nums [ i ] << " , ▁ " ; cout << nums [ mid - 1 ] ; return ; } if ( ( mid + 1 < n ) && nums [ mid + 1 ] == x ) { cout << nums [ i ] << " , ▁ " ; cout << nums [ mid + 1 ] ; return ; } break ; }
else { cout << nums [ i ] << " , ▁ " ; cout << nums [ mid ] ; return ; } } } }
cout << -1 ; }
int main ( ) { int A [ ] = { 0 , -1 , 2 , -3 , 1 } ; int X = -2 ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
hasArrayTwoPairs ( A , N , X ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findClosest ( int N , int target ) { int closest = -1 ; int diff = INT_MAX ;
for ( int i = 1 ; i <= sqrt ( N ) ; i ++ ) { if ( N % i == 0 ) {
if ( N / i == i ) {
if ( abs ( target - i ) < diff ) { diff = abs ( target - i ) ; closest = i ; } } else {
if ( abs ( target - i ) < diff ) { diff = abs ( target - i ) ; closest = i ; }
if ( abs ( target - N / i ) < diff ) { diff = abs ( target - N / i ) ; closest = N / i ; } } } }
cout << closest ; }
int main ( ) {
int N = 16 , X = 5 ;
findClosest ( N , X ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int power ( long long int A , long long int N ) {
int count = 0 ; if ( A == 1 ) return 0 ; while ( N ) {
count ++ ;
N /= A ; } return count ; }
void Pairs ( long long int N , long long int A , long long int B ) { int powerA , powerB ;
powerA = power ( A , N ) ;
powerB = power ( B , N ) ;
long long int intialB = B , intialA = A ;
A = 1 ; for ( int i = 0 ; i <= powerA ; i ++ ) { B = 1 ; for ( int j = 0 ; j <= powerB ; j ++ ) {
if ( B == N - A ) { cout << i << " ▁ " << j << endl ; return ; }
B *= intialB ; }
A *= intialA ; }
cout << -1 << endl ; return ; }
int main ( ) {
long long int N = 106 , A = 3 , B = 5 ;
Pairs ( N , A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findNonMultiples ( int arr [ ] , int n , int k ) {
set < int > multiples ;
for ( int i = 0 ; i < n ; ++ i ) {
if ( multiples . find ( arr [ i ] ) == multiples . end ( ) ) {
for ( int j = 1 ; j <= k / arr [ i ] ; j ++ ) { multiples . insert ( arr [ i ] * j ) ; } } }
return k - multiples . size ( ) ; }
int countValues ( int arr [ ] , int N , int L , int R ) {
return findNonMultiples ( arr , N , R ) - findNonMultiples ( arr , N , L - 1 ) ; }
int main ( ) { int arr [ ] = { 2 , 3 , 4 , 5 , 6 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int L = 1 , R = 20 ;
cout << countValues ( arr , N , L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minCollectingSpeed ( vector < int > & piles , int H ) {
int ans = -1 ; int low = 1 , high ;
high = * max_element ( piles . begin ( ) , piles . end ( ) ) ;
while ( low <= high ) {
int K = low + ( high - low ) / 2 ; int time = 0 ;
for ( int ai : piles ) { time += ( ai + K - 1 ) / K ; }
if ( time <= H ) { ans = K ; high = K - 1 ; }
else { low = K + 1 ; } }
cout << ans ; }
int main ( ) { vector < int > arr = { 3 , 6 , 7 , 11 } ; int H = 8 ;
minCollectingSpeed ( arr , H ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int cntDisPairs ( int arr [ ] , int N , int K ) {
int cntPairs = 0 ;
sort ( arr , arr + N ) ;
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
int main ( ) { int arr [ ] = { 5 , 6 , 5 , 7 , 7 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int K = 13 ; cout << cntDisPairs ( arr , N , K ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int cntDisPairs ( int arr [ ] , int N , int K ) {
int cntPairs = 0 ;
unordered_map < int , int > cntFre ; for ( int i = 0 ; i < N ; i ++ ) {
cntFre [ arr [ i ] ] ++ ; }
for ( auto it : cntFre ) {
int i = it . first ;
if ( 2 * i == K ) {
if ( cntFre [ i ] > 1 ) cntPairs += 2 ; } else { if ( cntFre [ K - i ] ) {
cntPairs += 1 ; } } }
cntPairs = cntPairs / 2 ; return cntPairs ; }
int main ( ) { int arr [ ] = { 5 , 6 , 5 , 7 , 7 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int K = 13 ; cout << cntDisPairs ( arr , N , K ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void longestSubsequence ( int N , int Q , int arr [ ] , int Queries [ ] [ 2 ] ) { for ( int i = 0 ; i < Q ; i ++ ) {
int x = Queries [ i ] [ 0 ] ; int y = Queries [ i ] [ 1 ] ;
arr [ x - 1 ] = y ;
int count = 1 ; for ( int j = 1 ; j < N ; j ++ ) {
if ( arr [ j ] != arr [ j - 1 ] ) { count += 1 ; } }
cout << count << ' ▁ ' ; } }
int main ( ) { int arr [ ] = { 1 , 1 , 2 , 5 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int Q = 2 ; int Queries [ Q ] [ 2 ] = { { 1 , 3 } , { 4 , 2 } } ;
longestSubsequence ( N , Q , arr , Queries ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void longestSubsequence ( int N , int Q , int arr [ ] , int Queries [ ] [ 2 ] ) { int count = 1 ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] != arr [ i - 1 ] ) { count += 1 ; } }
for ( int i = 0 ; i < Q ; i ++ ) {
int x = Queries [ i ] [ 0 ] ; int y = Queries [ i ] [ 1 ] ;
if ( x > 1 ) {
if ( arr [ x - 1 ] != arr [ x - 2 ] ) { count -= 1 ; }
if ( arr [ x - 2 ] != y ) { count += 1 ; } }
if ( x < N ) {
if ( arr [ x ] != arr [ x - 1 ] ) { count -= 1 ; }
if ( y != arr [ x ] ) { count += 1 ; } } cout << count << ' ▁ ' ;
arr [ x - 1 ] = y ; } }
int main ( ) { int arr [ ] = { 1 , 1 , 2 , 5 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int Q = 2 ; int Queries [ Q ] [ 2 ] = { { 1 , 3 } , { 4 , 2 } } ;
longestSubsequence ( N , Q , arr , Queries ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void sum ( int arr [ ] , int n ) {
map < int , vector < int > > mp ;
for ( int i = 0 ; i < n ; i ++ ) { mp [ arr [ i ] ] . push_back ( i ) ; }
int ans [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) {
int sum = 0 ;
for ( auto it : mp [ arr [ i ] ] ) {
sum += abs ( it - i ) ; }
ans [ i ] = sum ; }
for ( int i = 0 ; i < n ; i ++ ) { cout << ans [ i ] << " ▁ " ; } return ; }
int main ( ) {
int arr [ ] = { 1 , 3 , 1 , 1 , 2 } ;
int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
sum ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string conVowUpp ( string & str ) {
int N = str . length ( ) ; for ( int i = 0 ; i < N ; i ++ ) { if ( str [ i ] == ' a ' str [ i ] == ' e ' str [ i ] == ' i ' str [ i ] == ' o ' str [ i ] == ' u ' ) { str [ i ] = str [ i ] - ' a ' + ' A ' ; } } return str ; }
int main ( ) { string str = " eutopia " ; cout << conVowUpp ( str ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
map < int , int > mp ; int N , P ;
bool helper ( int mid ) { int cnt = 0 ; for ( auto i : mp ) { int temp = i . second ; while ( temp >= mid ) { temp -= mid ; cnt ++ ; } }
return cnt >= N ; }
int findMaximumDays ( int arr [ ] ) {
for ( int i = 0 ; i < P ; i ++ ) { mp [ arr [ i ] ] ++ ; }
int start = 0 , end = P , ans = 0 ; while ( start <= end ) {
int mid = start + ( ( end - start ) / 2 ) ;
if ( mid != 0 and helper ( mid ) ) { ans = mid ;
start = mid + 1 ; } else if ( mid == 0 ) { start = mid + 1 ; } else { end = mid - 1 ; } } return ans ; }
int main ( ) { N = 3 , P = 10 ; int arr [ ] = { 1 , 2 , 2 , 1 , 1 , 3 , 3 , 3 , 2 , 4 } ;
cout << findMaximumDays ( arr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long int countSubarrays ( int a [ ] , int n , int k ) {
int ans = 0 ;
vector < int > pref ; pref . push_back ( 0 ) ;
for ( int i = 0 ; i < n ; i ++ ) pref . push_back ( ( a [ i ] + pref [ i ] ) % k ) ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) {
if ( ( pref [ j ] - pref [ i - 1 ] + k ) % k == j - i + 1 ) { ans ++ ; } } }
cout << ans << ' ▁ ' ; }
int main ( ) {
int arr [ ] = { 2 , 3 , 5 , 3 , 1 , 5 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int K = 4 ;
countSubarrays ( arr , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long int countSubarrays ( int a [ ] , int n , int k ) {
unordered_map < int , int > cnt ;
long long int ans = 0 ;
vector < int > pref ; pref . push_back ( 0 ) ;
for ( int i = 0 ; i < n ; i ++ ) pref . push_back ( ( a [ i ] + pref [ i ] ) % k ) ;
cnt [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) {
int remIdx = i - k ; if ( remIdx >= 0 ) { cnt [ ( pref [ remIdx ] - remIdx % k + k ) % k ] -- ; }
ans += cnt [ ( pref [ i ] - i % k + k ) % k ] ;
cnt [ ( pref [ i ] - i % k + k ) % k ] ++ ; }
cout << ans << ' ▁ ' ; }
int main ( ) {
int arr [ ] = { 2 , 3 , 5 , 3 , 1 , 5 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int K = 4 ;
countSubarrays ( arr , N , K ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int check ( string & s , int k ) { int n = s . size ( ) ;
for ( int i = 0 ; i < k ; i ++ ) { for ( int j = i ; j < n ; j += k ) {
if ( s [ i ] != s [ j ] ) return false ; } } int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( s [ i ] == '0' )
c ++ ;
else
c -- ; }
if ( c == 0 ) return true ; else return false ; }
int main ( ) { string s = "101010" ; int k = 2 ; if ( check ( s , k ) ) cout << " Yes " << endl ; else cout << " No " << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isSame ( string str , int n ) {
map < int , int > mp ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) { mp [ str [ i ] - ' a ' ] ++ ; } for ( auto it : mp ) {
if ( ( it . second ) >= n ) { return true ; } }
return false ; }
int main ( ) { string str = " ccabcba " ; int n = 4 ;
if ( isSame ( str , n ) ) { cout << " Yes " ; } else { cout << " No " ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define eps  1e-6
double func ( double a , double b , double c , double x ) { return a * x * x + b * x + c ; }
double findRoot ( double a , double b , double c , double low , double high ) { double x ;
while ( fabs ( high - low ) > eps ) {
x = ( low + high ) / 2 ;
if ( func ( a , b , c , low ) * func ( a , b , c , x ) <= 0 ) { high = x ; }
else { low = x ; } }
return x ; }
void solve ( double a , double b , double c , double A , double B ) {
if ( func ( a , b , c , A ) * func ( a , b , c , B ) > 0 ) { cout << " No ▁ solution " ; }
else { cout << fixed << setprecision ( 4 ) << findRoot ( a , b , c , A , B ) ; } }
int main ( ) {
double a = 2 , b = -3 , c = -2 , A = 0 , B = 3 ;
solve ( a , b , c , A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define ll  long long NEW_LINE using namespace std ;
bool possible ( ll mid , vector < ll > & a ) {
ll n = a . size ( ) ;
ll total = ( n * ( n - 1 ) ) / 2 ;
ll need = ( total + 1 ) / 2 ; ll count = 0 ; ll start = 0 , end = 1 ;
while ( end < n ) { if ( a [ end ] - a [ start ] <= mid ) { end ++ ; } else { count += ( end - start - 1 ) ; start ++ ; } }
if ( end == n && start < end && a [ end - 1 ] - a [ start ] <= mid ) { ll t = end - start - 1 ; count += ( t * ( t + 1 ) / 2 ) ; }
if ( count >= need ) return true ; else return false ; }
ll findMedian ( vector < ll > & a ) {
ll n = a . size ( ) ;
ll low = 0 , high = a [ n - 1 ] - a [ 0 ] ;
while ( low <= high ) {
ll mid = ( low + high ) / 2 ;
if ( possible ( mid , a ) ) high = mid - 1 ; else low = mid + 1 ; }
return high + 1 ; }
int main ( ) { vector < ll > a = { 1 , 7 , 5 , 2 } ; sort ( a . begin ( ) , a . end ( ) ) ; cout << findMedian ( a ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void UniversalSubset ( vector < string > A , vector < string > B ) {
int n1 = A . size ( ) ; int n2 = B . size ( ) ;
vector < string > res ;
int A_fre [ n1 ] [ 26 ] ; for ( int i = 0 ; i < n1 ; i ++ ) { for ( int j = 0 ; j < 26 ; j ++ ) A_fre [ i ] [ j ] = 0 ; }
for ( int i = 0 ; i < n1 ; i ++ ) { for ( int j = 0 ; j < A [ i ] . size ( ) ; j ++ ) { A_fre [ i ] [ A [ i ] [ j ] - ' a ' ] ++ ; } }
int B_fre [ 26 ] = { 0 } ; for ( int i = 0 ; i < n2 ; i ++ ) { int arr [ 26 ] = { 0 } ; for ( int j = 0 ; j < B [ i ] . size ( ) ; j ++ ) { arr [ B [ i ] [ j ] - ' a ' ] ++ ; B_fre [ B [ i ] [ j ] - ' a ' ] = max ( B_fre [ B [ i ] [ j ] - ' a ' ] , arr [ B [ i ] [ j ] - ' a ' ] ) ; } } for ( int i = 0 ; i < n1 ; i ++ ) { int flag = 0 ; for ( int j = 0 ; j < 26 ; j ++ ) {
if ( A_fre [ i ] [ j ] < B_fre [ j ] ) {
flag = 1 ; break ; } }
if ( flag == 0 )
res . push_back ( A [ i ] ) ; }
if ( res . size ( ) ) {
for ( int i = 0 ; i < res . size ( ) ; i ++ ) { for ( int j = 0 ; j < res [ i ] . size ( ) ; j ++ ) cout << res [ i ] [ j ] ; } cout << " ▁ " ; }
else cout < < " - 1" ; }
int main ( ) { vector < string > A = { " geeksforgeeks " , " topcoder " , " leetcode " } ; vector < string > B = { " geek " , " ee " } ; UniversalSubset ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findPair ( int a [ ] , int n ) {
int min_dist = INT_MAX ; int index_a = -1 , index_b = -1 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i + 1 ; j < n ; j ++ ) {
if ( j - i < min_dist ) {
if ( a [ i ] % a [ j ] == 0 a [ j ] % a [ i ] == 0 ) {
min_dist = j - i ;
index_a = i ; index_b = j ; } } } }
if ( index_a == -1 ) { cout << ( " - 1" ) ; }
else { cout << " ( " << a [ index_a ] << " , ▁ " << a [ index_b ] << " ) " ; } }
int main ( ) {
int a [ ] = { 2 , 3 , 4 , 5 , 6 } ; int n = sizeof ( a ) / sizeof ( int ) ;
findPair ( a , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printNum ( int L , int R ) {
for ( int i = L ; i <= R ; i ++ ) { int temp = i ; int c = 10 ; int flag = 0 ;
while ( temp > 0 ) {
if ( temp % 10 >= c ) { flag = 1 ; break ; } c = temp % 10 ; temp /= 10 ; }
if ( flag == 0 ) cout << i << " ▁ " ; } }
int main ( ) {
int L = 10 , R = 15 ;
printNum ( L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMissing ( int arr [ ] , int left , int right , int diff ) {
if ( right <= left ) return INT_MAX ;
int mid = left + ( right - left ) / 2 ;
if ( arr [ mid + 1 ] - arr [ mid ] != diff ) return ( arr [ mid ] + diff ) ;
if ( mid > 0 && arr [ mid ] - arr [ mid - 1 ] != diff ) return ( arr [ mid - 1 ] + diff ) ;
if ( arr [ mid ] == arr [ 0 ] + mid * diff ) return findMissing ( arr , mid + 1 , right , diff ) ;
return findMissing ( arr , left , mid - 1 , diff ) ; }
int missingElement ( int arr [ ] , int n ) {
sort ( arr , arr + n ) ;
int diff = ( arr [ n - 1 ] - arr [ 0 ] ) / n ;
return findMissing ( arr , 0 , n - 1 , diff ) ; }
int main ( ) {
int arr [ ] = { 2 , 8 , 6 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << missingElement ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int power ( int x , unsigned int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
int nthRootSearch ( int low , int high , int N , int K ) {
if ( low <= high ) {
int mid = ( low + high ) / 2 ;
if ( ( power ( mid , K ) <= N ) && ( power ( mid + 1 , K ) > N ) ) { return mid ; }
else if ( power ( mid , K ) < N ) { return nthRootSearch ( mid + 1 , high , N , K ) ; } else { return nthRootSearch ( low , mid - 1 , N , K ) ; } } return low ; }
int main ( ) {
int N = 16 , K = 4 ;
cout << nthRootSearch ( 0 , N , N , K ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int get_subset_count ( int arr [ ] , int K , int N ) {
sort ( arr , arr + N ) ; int left , right ; left = 0 ; right = N - 1 ;
int ans = 0 ; while ( left <= right ) { if ( arr [ left ] + arr [ right ] < K ) {
ans += 1 << ( right - left ) ; left ++ ; } else {
right -- ; } } return ans ; }
int main ( ) { int arr [ ] = { 2 , 4 , 5 , 7 } ; int K = 8 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << get_subset_count ( arr , K , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int minMaxDiff ( int arr [ ] , int n , int k ) { int max_adj_dif = INT_MIN ;
for ( int i = 0 ; i < n - 1 ; i ++ ) max_adj_dif = max ( max_adj_dif , abs ( arr [ i ] - arr [ i + 1 ] ) ) ;
if ( max_adj_dif == 0 ) return 0 ;
int best = 1 ; int worst = max_adj_dif ; int mid , required ; while ( best < worst ) { mid = ( best + worst ) / 2 ;
required = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) { required += ( abs ( arr [ i ] - arr [ i + 1 ] ) - 1 ) / mid ; }
if ( required > k ) best = mid + 1 ;
else worst = mid ; } return worst ; }
int main ( ) { int arr [ ] = { 3 , 12 , 25 , 50 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 7 ; cout << minMaxDiff ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void checkMin ( int arr [ ] , int len ) {
int smallest = INT_MAX , secondSmallest = INT_MAX ; for ( int i = 0 ; i < len ; i ++ ) {
if ( arr [ i ] < smallest ) { secondSmallest = smallest ; smallest = arr [ i ] ; }
else if ( arr [ i ] < secondSmallest ) { secondSmallest = arr [ i ] ; } } if ( 2 * smallest <= secondSmallest ) cout << " Yes " ; else cout << " No " ; }
int main ( ) { int arr [ ] = { 2 , 3 , 4 , 5 } ; int len = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; checkMin ( arr , len ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void createHash ( set < int > & hash , int maxElement ) {
int prev = 0 , curr = 1 ; hash . insert ( prev ) ; hash . insert ( curr ) ; while ( curr <= maxElement ) {
int temp = curr + prev ; hash . insert ( temp ) ;
prev = curr ; curr = temp ; } }
void fibonacci ( int arr [ ] , int n ) {
int max_val = * max_element ( arr , arr + n ) ;
set < int > hash ; createHash ( hash , max_val ) ;
int minimum = INT_MAX ; int maximum = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) {
if ( hash . find ( arr [ i ] ) != hash . end ( ) ) {
minimum = min ( minimum , arr [ i ] ) ; maximum = max ( maximum , arr [ i ] ) ; } } cout << minimum << " , ▁ " << maximum << endl ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; fibonacci ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isValidLen ( string s , int len , int k ) {
int n = s . size ( ) ;
unordered_map < char , int > mp ; int right = 0 ;
while ( right < len ) { mp [ s [ right ] ] ++ ; right ++ ; } if ( mp . size ( ) <= k ) return true ;
while ( right < n ) {
mp [ s [ right ] ] ++ ;
mp [ s [ right - len ] ] -- ;
if ( mp [ s [ right - len ] ] == 0 ) mp . erase ( s [ right - len ] ) ; if ( mp . size ( ) <= k ) return true ; right ++ ; } return mp . size ( ) <= k ; }
int maxLenSubStr ( string s , int k ) {
set < char > uni ; for ( auto x : s ) uni . insert ( x ) ; if ( uni . size ( ) < k ) return -1 ;
int n = s . size ( ) ;
int lo = -1 , hi = n + 1 ; while ( hi - lo > 1 ) { int mid = lo + hi >> 1 ; if ( isValidLen ( s , mid , k ) ) lo = mid ; else hi = mid ; } return lo ; }
int main ( ) { string s = " aabacbebebe " ; int k = 3 ; cout << maxLenSubStr ( s , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isSquarePossible ( int arr [ ] , int n , int l ) {
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] >= l ) cnt ++ ;
if ( cnt >= l ) return true ; } return false ; }
int maxArea ( int arr [ ] , int n ) { int l = 0 , r = n ; int len = 0 ; while ( l <= r ) { int m = l + ( ( r - l ) / 2 ) ;
if ( isSquarePossible ( arr , n , m ) ) { len = m ; l = m + 1 ; }
else r = m - 1 ; }
return ( len * len ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 5 , 5 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << maxArea ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void insertNames ( string arr [ ] , int n ) {
unordered_set < string > set ; for ( int i = 0 ; i < n ; i ++ ) {
if ( set . find ( arr [ i ] ) == set . end ( ) ) { cout << " No STRNEWLINE " ; set . insert ( arr [ i ] ) ; } else { cout << " Yes STRNEWLINE " ; } } }
int main ( ) { string arr [ ] = { " geeks " , " for " , " geeks " } ; int n = sizeof ( arr ) / sizeof ( string ) ; insertNames ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countLessThan ( int arr [ ] , int n , int key ) { int l = 0 , r = n - 1 ; int index = -1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( arr [ m ] < key ) { l = m + 1 ; index = m ; } else { r = m - 1 ; } } return ( index + 1 ) ; }
int countGreaterThan ( int arr [ ] , int n , int key ) { int l = 0 , r = n - 1 ; int index = -1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( arr [ m ] <= key ) { l = m + 1 ; } else { r = m - 1 ; index = m ; } } if ( index == -1 ) return 0 ; return ( n - index ) ; }
int countTriplets ( int n , int * a , int * b , int * c ) {
sort ( a , a + n ) ; sort ( b , b + n ) ; sort ( c , c + n ) ; int count = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { int current = b [ i ] ; int a_index = -1 , c_index = -1 ;
int low = countLessThan ( a , n , current ) ;
int high = countGreaterThan ( c , n , current ) ;
count += ( low * high ) ; } return count ; }
int main ( ) { int a [ ] = { 1 , 5 } ; int b [ ] = { 2 , 4 } ; int c [ ] = { 3 , 6 } ; int size = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << countTriplets ( size , a , b , c ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int costToBalance ( string s ) { if ( s . length ( ) == 0 ) cout << 0 << endl ;
int ans = 0 ;
int o = 0 , c = 0 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s [ i ] == ' ( ' ) o ++ ; if ( s [ i ] == ' ) ' ) c ++ ; } if ( o != c ) return -1 ; int a [ s . size ( ) ] ; if ( s [ 0 ] == ' ( ' ) a [ 0 ] = 1 ; else a [ 0 ] = -1 ; if ( a [ 0 ] < 0 ) ans += abs ( a [ 0 ] ) ; for ( int i = 1 ; i < s . length ( ) ; i ++ ) { if ( s [ i ] == ' ( ' ) a [ i ] = a [ i - 1 ] + 1 ; else a [ i ] = a [ i - 1 ] - 1 ; if ( a [ i ] < 0 ) ans += abs ( a [ i ] ) ; } return ans ; }
int main ( ) { string s ; s = " ) ) ) ( ( ( " ; cout << costToBalance ( s ) << endl ; s = " ) ) ( ( " ; cout << costToBalance ( s ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int middleOfThree ( int a , int b , int c ) {
int x = a - b ;
int y = b - c ;
int z = a - c ;
if ( x * y > 0 ) return b ;
else if ( x * z > 0 ) return c ; else return a ; }
int main ( ) { int a = 20 , b = 30 , c = 40 ; cout << middleOfThree ( a , b , c ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void missing4 ( int arr [ ] , int n ) {
int helper [ 4 ] ;
for ( int i = 0 ; i < n ; i ++ ) { int temp = abs ( arr [ i ] ) ;
if ( temp <= n ) arr [ temp - 1 ] *= ( -1 ) ;
else if ( temp > n ) { if ( temp % n != 0 ) helper [ temp % n - 1 ] = -1 ; else helper [ ( temp % n ) + n - 1 ] = -1 ; } }
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] > 0 ) cout << ( i + 1 ) << " ▁ " ; for ( int i = 0 ; i < 4 ; i ++ ) if ( helper [ i ] >= 0 ) cout << ( n + i + 1 ) << " ▁ " ; return ; }
int main ( ) { int arr [ ] = { 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; missing4 ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void lexiMiddleSmallest ( int K , int N ) {
if ( K % 2 == 0 ) {
cout << K / 2 << " ▁ " ;
for ( int i = 0 ; i < N - 1 ; ++ i ) { cout << K << " ▁ " ; } cout << " STRNEWLINE " ; exit ( 0 ) ; }
vector < int > a ( N , ( K + 1 ) / 2 ) ;
for ( int i = 0 ; i < N / 2 ; ++ i ) {
if ( a . back ( ) == 1 ) {
a . pop_back ( ) ; }
else {
-- a . back ( ) ;
while ( ( int ) a . size ( ) < N ) { a . push_back ( K ) ; } } }
for ( auto i : a ) { cout << i << " ▁ " ; } cout << " STRNEWLINE " ; }
int main ( ) { int K = 2 , N = 4 ; lexiMiddleSmallest ( K , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findLastElement ( int arr [ ] , int N ) {
sort ( arr , arr + N ) ; int i = 0 ;
for ( i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] - arr [ i - 1 ] != 0 && arr [ i ] - arr [ i - 1 ] != 2 ) { cout << " - 1" << endl ; return ; } }
cout << arr [ N - 1 ] << endl ; }
int main ( ) { int arr [ ] = { 2 , 4 , 6 , 8 , 0 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findLastElement ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maxDivisions ( int arr [ ] , int N , int X ) {
sort ( arr , arr + N , greater < int > ( ) ) ;
int maxSub = 0 ;
int size = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
size ++ ;
if ( arr [ i ] * size >= X ) {
maxSub ++ ;
size = 0 ; } } cout << maxSub << endl ; }
int main ( ) {
int arr [ ] = { 1 , 3 , 3 , 7 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int X = 3 ; maxDivisions ( arr , N , X ) ; return 0 ; }
#include <iostream> NEW_LINE #include <bits/stdc++.h> NEW_LINE using namespace std ;
void maxPossibleSum ( int arr [ ] , int N ) {
sort ( arr , arr + N ) ; int sum = 0 ; int j = N - 3 ; while ( j >= 0 ) {
sum += arr [ j ] ; j -= 3 ; }
cout << sum ; }
int main ( ) {
int arr [ ] = { 7 , 4 , 5 , 2 , 3 , 1 , 5 , 9 } ;
int N = 8 ; maxPossibleSum ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void insertionSort ( int arr [ ] , int n ) { int i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
void printArray ( int arr [ ] , int n ) { int i ;
for ( i = 0 ; i < n ; i ++ ) { cout << arr [ i ] << " ▁ " ; } cout << endl ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
insertionSort ( arr , N ) ; printArray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getPairs ( int arr [ ] , int N , int K ) {
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) {
if ( arr [ i ] > K * arr [ i + 1 ] ) count ++ ; } } cout << count ; }
int main ( ) { int arr [ ] = { 5 , 6 , 2 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int K = 2 ;
getPairs ( arr , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int merge ( int arr [ ] , int temp [ ] , int l , int m , int r , int K ) {
int i = l ;
int j = m + 1 ;
int cnt = 0 ; for ( int l = 0 ; i <= m ; i ++ ) { bool found = false ;
while ( j <= r ) {
if ( arr [ i ] >= K * arr [ j ] ) { found = true ; } else break ; j ++ ; }
if ( found ) { cnt += j - ( m + 1 ) ; j -- ; } }
int k = l ; i = l ; j = m + 1 ; while ( i <= m && j <= r ) { if ( arr [ i ] <= arr [ j ] ) temp [ k ++ ] = arr [ i ++ ] ; else temp [ k ++ ] = arr [ j ++ ] ; }
while ( i <= m ) temp [ k ++ ] = arr [ i ++ ] ;
while ( j <= r ) temp [ k ++ ] = arr [ j ++ ] ; for ( int i = l ; i <= r ; i ++ ) arr [ i ] = temp [ i ] ;
return cnt ; }
int mergeSortUtil ( int arr [ ] , int temp [ ] , int l , int r , int K ) { int cnt = 0 ; if ( l < r ) {
int m = ( l + r ) / 2 ;
cnt += mergeSortUtil ( arr , temp , l , m , K ) ; cnt += mergeSortUtil ( arr , temp , m + 1 , r , K ) ;
cnt += merge ( arr , temp , l , m , r , K ) ; } return cnt ; }
int mergeSort ( int arr [ ] , int N , int K ) { int temp [ N ] ; cout << mergeSortUtil ( arr , temp , 0 , N - 1 , K ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 2 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int K = 2 ;
mergeSort ( arr , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minRemovals ( int A [ ] , int N ) {
sort ( A , A + N ) ;
int mx = A [ N - 1 ] ;
int sum = 1 ;
for ( int i = 0 ; i < N ; i ++ ) { sum += A [ i ] ; } if ( sum - mx >= mx ) { cout << 0 << " STRNEWLINE " ; } else { cout << 2 * mx - sum << " STRNEWLINE " ; } }
int main ( ) { int A [ ] = { 3 , 3 , 2 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
minRemovals ( A , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void rearrangeArray ( int a [ ] , int n ) {
sort ( a , a + n ) ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( a [ i ] == i + 1 ) {
swap ( a [ i ] , a [ i + 1 ] ) ; } }
if ( a [ n - 1 ] == n ) {
swap ( a [ n - 1 ] , a [ n - 2 ] ) ; }
for ( int i = 0 ; i < n ; i ++ ) { cout << a [ i ] << " ▁ " ; } }
int main ( ) { int arr [ ] = { 1 , 5 , 3 , 2 , 4 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
rearrangeArray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minOperations ( int arr1 [ ] , int arr2 [ ] , int i , int j , int n ) {
int f = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr1 [ i ] != arr2 [ i ] ) f = 1 ; break ; } if ( f == 0 ) return 0 ; if ( i >= n j >= n ) return 0 ;
if ( arr1 [ i ] < arr2 [ j ] )
return 1 + minOperations ( arr1 , arr2 , i + 1 , j + 1 , n ) ;
return max ( minOperations ( arr1 , arr2 , i , j + 1 , n ) , minOperations ( arr1 , arr2 , i + 1 , j , n ) ) ; }
void minOperationsUtil ( int arr [ ] , int n ) { int brr [ n ] ; for ( int i = 0 ; i < n ; i ++ ) brr [ i ] = arr [ i ] ; sort ( brr , brr + n ) ; int f = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] != brr [ i ] )
f = 1 ; break ; }
if ( f == 1 )
cout << ( minOperations ( arr , brr , 0 , 0 , n ) ) ; else cout < < "0" ; }
int main ( ) { int arr [ ] = { 4 , 7 , 2 , 3 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; minOperationsUtil ( arr , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void canTransform ( string & s , string & t ) { int n = s . length ( ) ;
vector < int > occur [ 26 ] ; for ( int x = 0 ; x < n ; x ++ ) { char ch = s [ x ] - ' a ' ; occur [ ch ] . push_back ( x ) ; }
vector < int > idx ( 26 , 0 ) ; bool poss = true ; for ( int x = 0 ; x < n ; x ++ ) { char ch = t [ x ] - ' a ' ;
if ( idx [ ch ] >= occur [ ch ] . size ( ) ) {
poss = false ; break ; } for ( int small = 0 ; small < ch ; small ++ ) {
if ( idx [ small ] < occur [ small ] . size ( ) && occur [ small ] [ idx [ small ] ] < occur [ ch ] [ idx [ ch ] ] ) {
poss = false ; break ; } } idx [ ch ] ++ ; }
if ( poss ) { cout << " Yes " << endl ; } else { cout << " No " << endl ; } }
int main ( ) { string s , t ; s = " hdecb " ; t = " cdheb " ; canTransform ( s , t ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int inversionCount ( string & s ) {
int freq [ 26 ] = { 0 } ; int inv = 0 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { int temp = 0 ;
for ( int j = 0 ; j < int ( s [ i ] - ' a ' ) ; j ++ )
temp += freq [ j ] ; inv += ( i - temp ) ;
freq [ s [ i ] - ' a ' ] ++ ; } return inv ; }
bool haveRepeated ( string & S1 , string & S2 ) { int freq [ 26 ] = { 0 } ; for ( char i : S1 ) { if ( freq [ i - ' a ' ] > 0 ) return true ; freq [ i - ' a ' ] ++ ; } for ( int i = 0 ; i < 26 ; i ++ ) freq [ i ] = 0 ; for ( char i : S2 ) { if ( freq [ i - ' a ' ] > 0 ) return true ; freq [ i - ' a ' ] ++ ; } return false ; }
void checkToMakeEqual ( string S1 , string S2 ) {
int freq [ 26 ] = { 0 } ; for ( int i = 0 ; i < S1 . length ( ) ; i ++ ) {
freq [ S1 [ i ] - ' a ' ] ++ ; } bool flag = 0 ; for ( int i = 0 ; i < S2 . length ( ) ; i ++ ) { if ( freq [ S2 [ i ] - ' a ' ] == 0 ) {
flag = true ; break ; }
freq [ S2 [ i ] - ' a ' ] -- ; } if ( flag == true ) {
cout << " No STRNEWLINE " ; return ; }
int invCount1 = inversionCount ( S1 ) ; int invCount2 = inversionCount ( S2 ) ; if ( invCount1 == invCount2 || ( invCount1 & 1 ) == ( invCount2 & 1 ) || haveRepeated ( S1 , S2 ) ) {
cout << " Yes STRNEWLINE " ; } else cout << " No STRNEWLINE " ; }
int main ( ) { string S1 = " abbca " , S2 = " acabb " ; checkToMakeEqual ( S1 , S2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void sortArr ( int a [ ] , int n ) { int i , k ;
k = ( int ) log2 ( n ) ; k = pow ( 2 , k ) ;
while ( k > 0 ) { for ( i = 0 ; i + k < n ; i ++ ) if ( a [ i ] > a [ i + k ] ) swap ( a [ i ] , a [ i + k ] ) ;
k = k / 2 ; }
for ( i = 0 ; i < n ; i ++ ) { cout << a [ i ] << " ▁ " ; } }
int main ( ) {
int arr [ ] = { 5 , 20 , 30 , 40 , 36 , 33 , 25 , 15 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
sortArr ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maximumSum ( int arr [ ] , int n , int k ) {
int elt = n / k ; int sum = 0 ;
sort ( arr , arr + n ) ; int count = 0 ; int i = n - 1 ;
while ( count < k ) { sum += arr [ i ] ; i -- ; count ++ ; } count = 0 ; i = 0 ;
while ( count < k ) { sum += arr [ i ] ; i += elt - 1 ; count ++ ; }
cout << sum << " STRNEWLINE " ; }
int main ( ) { int Arr [ ] = { 1 , 13 , 7 , 17 , 6 , 5 } ; int K = 2 ; int size = sizeof ( Arr ) / sizeof ( Arr [ 0 ] ) ; maximumSum ( Arr , size , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinSum ( int arr [ ] , int K , int L , int size ) { if ( K * L > size ) return -1 ; int minsum = 0 ;
sort ( arr , arr + size ) ;
for ( int i = 0 ; i < K ; i ++ ) minsum += arr [ i ] ;
return minsum ; }
int main ( ) { int arr [ ] = { 2 , 15 , 5 , 1 , 35 , 16 , 67 , 10 } ; int K = 3 ; int L = 2 ; int length = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMinSum ( arr , K , L , length ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int findKthSmallest ( int arr [ ] , int n , int k ) {
int max = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; }
int counter [ max + 1 ] = { 0 } ;
int smallest = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { counter [ arr [ i ] ] ++ ; }
for ( int num = 1 ; num <= max ; num ++ ) {
if ( counter [ num ] > 0 ) {
smallest += counter [ num ] ; }
if ( smallest >= k ) {
return num ; } } }
int main ( ) {
int arr [ ] = { 7 , 1 , 4 , 4 , 20 , 15 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int K = 5 ;
cout << findKthSmallest ( arr , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void lexNumbers ( int n ) { vector < string > s ; for ( int i = 1 ; i <= n ; i ++ ) { s . push_back ( to_string ( i ) ) ; } sort ( s . begin ( ) , s . end ( ) ) ; vector < int > ans ; for ( int i = 0 ; i < n ; i ++ ) ans . push_back ( stoi ( s [ i ] ) ) ; for ( int i = 0 ; i < n ; i ++ ) cout << ans [ i ] << " ▁ " ; }
int main ( ) { int n = 15 ; lexNumbers ( n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define N  4 NEW_LINE void func ( int a [ ] [ N ] ) {
for ( int i = 0 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) { for ( int j = 0 ; j < N ; j ++ ) { for ( int k = j + 1 ; k < N ; ++ k ) {
if ( a [ i ] [ j ] > a [ i ] [ k ] ) {
int temp = a [ i ] [ j ] ; a [ i ] [ j ] = a [ i ] [ k ] ; a [ i ] [ k ] = temp ; } } } }
else { for ( int j = 0 ; j < N ; j ++ ) { for ( int k = j + 1 ; k < N ; ++ k ) {
if ( a [ i ] [ j ] < a [ i ] [ k ] ) {
int temp = a [ i ] [ j ] ; a [ i ] [ j ] = a [ i ] [ k ] ; a [ i ] [ k ] = temp ; } } } } }
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) { printf ( " % d ▁ " , a [ i ] [ j ] ) ; } printf ( " STRNEWLINE " ) ; } }
int main ( ) { int a [ N ] [ N ] = { { 5 , 7 , 3 , 4 } , { 9 , 5 , 8 , 2 } , { 6 , 3 , 8 , 1 } , { 5 , 8 , 9 , 3 } } ; func ( a ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
map < int , int > g [ 200005 ] ; set < int > s , ns ;
void dfs ( int x ) { vector < int > v ; v . clear ( ) ; ns . clear ( ) ;
for ( int it : s ) {
if ( ! g [ x ] [ it ] ) { v . push_back ( it ) ; } else { ns . insert ( it ) ; } } s = ns ; for ( int i : v ) { dfs ( i ) ; } }
void weightOfMST ( int N ) {
int cnt = 0 ;
for ( int i = 1 ; i <= N ; ++ i ) { s . insert ( i ) ; }
for ( ; s . size ( ) ; ) {
++ cnt ; int t = * s . begin ( ) ; s . erase ( t ) ;
dfs ( t ) ; } cout << cnt - 1 ; }
int main ( ) { int N = 6 , M = 11 ; int edges [ ] [ ] = { { 1 , 3 } , { 1 , 4 } , { 1 , 5 } , { 1 , 6 } , { 2 , 3 } , { 2 , 4 } , { 2 , 5 } , { 2 , 6 } , { 3 , 4 } , { 3 , 5 } , { 3 , 6 } } ;
for ( int i = 0 ; i < M ; ++ i ) { int u = edges [ i ] [ 0 ] ; int v = edges [ i ] [ 1 ] ; g [ u ] [ v ] = 1 ; g [ v ] [ u ] = 1 ; }
weightOfMST ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countPairs ( vector < int > A , vector < int > B ) { int n = A . size ( ) ; sort ( A . begin ( ) , A . end ( ) ) ; sort ( B . begin ( ) , B . end ( ) ) ; int ans = 0 , i ; for ( int i = 0 ; i < n ; i ++ ) { if ( A [ i ] > B [ ans ] ) { ans ++ ; } } return ans ; }
int main ( ) { vector < int > A = { 30 , 28 , 45 , 22 } ; vector < int > B = { 35 , 25 , 22 , 48 } ; cout << countPairs ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxMod ( int arr [ ] , int n ) { int maxVal = * max_element ( arr , arr + n ) ; int secondMax = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] < maxVal && arr [ i ] > secondMax ) { secondMax = arr [ i ] ; } } return secondMax ; }
int main ( ) { int arr [ ] = { 2 , 4 , 1 , 5 , 3 , 6 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << maxMod ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPossible ( int A [ ] , int B [ ] , int n , int m , int x , int y ) {
if ( x > n y > m ) return false ;
sort ( A , A + n ) ; sort ( B , B + m ) ;
if ( A [ x - 1 ] < B [ m - y ] ) return true ; else return false ; }
int main ( ) { int A [ ] = { 1 , 1 , 1 , 1 , 1 } ; int B [ ] = { 2 , 2 } ; int n = sizeof ( A ) / sizeof ( int ) ; int m = sizeof ( B ) / sizeof ( int ) ; int x = 3 , y = 1 ; if ( isPossible ( A , B , n , m , x , y ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100005
int Min_Replace ( int arr [ ] , int n , int k ) { sort ( arr , arr + n ) ;
int freq [ MAX ] ; memset ( freq , 0 , sizeof freq ) ; int p = 0 ; freq [ p ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] == arr [ i - 1 ] ) ++ freq [ p ] ; else ++ freq [ ++ p ] ; }
sort ( freq , freq + n , greater < int > ( ) ) ;
int ans = 0 ; for ( int i = k ; i <= p ; i ++ ) ans += freq [ i ] ;
return ans ; }
int main ( ) { int arr [ ] = { 1 , 2 , 7 , 8 , 2 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 2 ; cout << Min_Replace ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Segment ( int x [ ] , int l [ ] , int n ) {
if ( n == 1 ) return 1 ;
int ans = 2 ; for ( int i = 1 ; i < n - 1 ; i ++ ) {
if ( x [ i ] - l [ i ] > x [ i - 1 ] ) ans ++ ;
else if ( x [ i ] + l [ i ] < x [ i + 1 ] ) {
x [ i ] = x [ i ] + l [ i ] ; ans ++ ; } }
return ans ; }
int main ( ) { int x [ ] = { 1 , 3 , 4 , 5 , 8 } , l [ ] = { 10 , 1 , 2 , 2 , 5 } ; int n = sizeof ( x ) / sizeof ( x [ 0 ] ) ;
cout << Segment ( x , l , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MinimizeleftOverSum ( int a [ ] , int n ) { vector < int > v1 , v2 ; for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] % 2 ) v1 . push_back ( a [ i ] ) ; else v2 . push_back ( a [ i ] ) ; }
if ( v1 . size ( ) > v2 . size ( ) ) {
sort ( v1 . begin ( ) , v1 . end ( ) ) ; sort ( v2 . begin ( ) , v2 . end ( ) ) ;
int x = v1 . size ( ) - v2 . size ( ) - 1 ; int sum = 0 ; int i = 0 ;
while ( i < x ) { sum += v1 [ i ++ ] ; }
return sum ; }
else if ( v2 . size ( ) > v1 . size ( ) ) {
sort ( v1 . begin ( ) , v1 . end ( ) ) ; sort ( v2 . begin ( ) , v2 . end ( ) ) ;
int x = v2 . size ( ) - v1 . size ( ) - 1 ; int sum = 0 ; int i = 0 ;
while ( i < x ) { sum += v2 [ i ++ ] ; }
return sum ; }
else return 0 ; }
int main ( ) { int a [ ] = { 2 , 2 , 2 , 2 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << MinimizeleftOverSum ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minOperation ( string S , int N , int K ) {
if ( N % K ) { cout << " Not ▁ Possible " << endl ; return ; }
int count [ 26 ] = { 0 } ; for ( int i = 0 ; i < N ; i ++ ) { count [ S [ i ] - 97 ] ++ ; } int E = N / K ; vector < int > greaterE ; vector < int > lessE ; for ( int i = 0 ; i < 26 ; i ++ ) {
if ( count [ i ] < E ) lessE . push_back ( E - count [ i ] ) ; else greaterE . push_back ( count [ i ] - E ) ; } sort ( greaterE . begin ( ) , greaterE . end ( ) ) ; sort ( lessE . begin ( ) , lessE . end ( ) ) ; int mi = INT_MAX ; for ( int i = 0 ; i <= K ; i ++ ) {
int set1 = i ; int set2 = K - i ; if ( greaterE . size ( ) >= set1 && lessE . size ( ) >= set2 ) { int step1 = 0 ; int step2 = 0 ; for ( int j = 0 ; j < set1 ; j ++ ) step1 += greaterE [ j ] ; for ( int j = 0 ; j < set2 ; j ++ ) step2 += lessE [ j ] ; mi = min ( mi , max ( step1 , step2 ) ) ; } } cout << mi << endl ; }
int main ( ) { string S = " accb " ; int N = S . size ( ) ; int K = 2 ; minOperation ( S , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minMovesToSort ( int arr [ ] , int n ) { int moves = 0 ; int i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
} return moves ; }
int main ( ) { int arr [ ] = { 3 , 5 , 2 , 8 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minMovesToSort ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool prime [ 100005 ] ; void SieveOfEratosthenes ( int n ) { memset ( prime , true , sizeof ( prime ) ) ;
prime [ 1 ] = false ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = false ; } } }
void sortPrimes ( int arr [ ] , int n ) { SieveOfEratosthenes ( 100005 ) ;
vector < int > v ; for ( int i = 0 ; i < n ; i ++ ) {
if ( prime [ arr [ i ] ] ) v . push_back ( arr [ i ] ) ; } sort ( v . begin ( ) , v . end ( ) , greater < int > ( ) ) ; int j = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( prime [ arr [ i ] ] ) arr [ i ] = v [ j ++ ] ; } }
int main ( ) { int arr [ ] = { 4 , 3 , 2 , 6 , 100 , 17 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; sortPrimes ( arr , n ) ;
for ( int i = 0 ; i < n ; i ++ ) { cout << arr [ i ] << " ▁ " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findOptimalPairs ( int arr [ ] , int N ) { sort ( arr , arr + N ) ;
for ( int i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) cout << " ( " << arr [ i ] << " , ▁ " << arr [ j ] << " ) " << " ▁ " ; }
int main ( ) { int arr [ ] = { 9 , 6 , 5 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findOptimalPairs ( arr , N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int countBits ( int a ) { int count = 0 ; while ( a ) { if ( a & 1 ) count += 1 ; a = a >> 1 ; } return count ; }
void insertionSort ( int arr [ ] , int aux [ ] , int n ) { for ( int i = 1 ; i < n ; i ++ ) {
int key1 = aux [ i ] ; int key2 = arr [ i ] ; int j = i - 1 ;
while ( j >= 0 && aux [ j ] < key1 ) { aux [ j + 1 ] = aux [ j ] ; arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } aux [ j + 1 ] = key1 ; arr [ j + 1 ] = key2 ; } }
void sortBySetBitCount ( int arr [ ] , int n ) {
int aux [ n ] ; for ( int i = 0 ; i < n ; i ++ ) aux [ i ] = countBits ( arr [ i ] ) ;
insertionSort ( arr , aux , n ) ; }
void printArr ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countBits ( int a ) { int count = 0 ; while ( a ) { if ( a & 1 ) count += 1 ; a = a >> 1 ; } return count ; }
void sortBySetBitCount ( int arr [ ] , int n ) { vector < vector < int > > count ( 32 ) ; int setbitcount = 0 ; for ( int i = 0 ; i < n ; i ++ ) { setbitcount = countBits ( arr [ i ] ) ; count [ setbitcount ] . push_back ( arr [ i ] ) ; }
for ( int i = 31 ; i >= 0 ; i -- ) { vector < int > v1 = count [ i ] ; for ( int i = 0 ; i < v1 . size ( ) ; i ++ ) arr [ j ++ ] = v1 [ i ] ; } }
void printArr ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; sortBySetBitCount ( arr , n ) ; printArr ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void generateString ( int k1 , int k2 , string s ) {
int C1s = 0 , C0s = 0 ; int flag = 0 ; vector < int > pos ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s [ i ] == '0' ) { C0s ++ ;
if ( ( i + 1 ) % k1 != 0 && ( i + 1 ) % k2 != 0 ) { pos . push_back ( i ) ; } } else { C1s ++ ; } if ( C0s >= C1s ) {
if ( pos . size ( ) == 0 ) { cout << -1 ; flag = 1 ; break ; }
else { int k = pos . back ( ) ; s [ k ] = '1' ; C0s -- ; C1s ++ ; pos . pop_back ( ) ; } } }
if ( flag == 0 ) { cout << s ; } }
int main ( ) { int K1 = 2 , K2 = 4 ; string S = "11000100" ; generateString ( K1 , K2 , S ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maximizeProduct ( int N ) {
int MSB = ( int ) log2 ( N ) ;
int X = 1 << MSB ;
int Y = N - ( 1 << MSB ) ;
for ( int i = 0 ; i < MSB ; i ++ ) {
if ( ! ( N & ( 1 << i ) ) ) {
X += 1 << i ;
Y += 1 << i ; } }
cout << X << " ▁ " << Y ; }
int main ( ) { int N = 45 ; maximizeProduct ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool check ( int num ) {
int sm = 0 ;
int num2 = num * num ; while ( num ) { sm += num % 10 ; num /= 10 ; }
int sm2 = 0 ; while ( num2 ) { sm2 += num2 % 10 ; num2 /= 10 ; } return ( ( sm * sm ) == sm2 ) ; }
int convert ( string s ) { int val = 0 ; reverse ( s . begin ( ) , s . end ( ) ) ; int cur = 1 ; for ( int i = 0 ; i < s . size ( ) ; i ++ ) { val += ( s [ i ] - '0' ) * cur ; cur *= 10 ; } return val ; }
void generate ( string s , int len , set < int > & uniq ) {
if ( s . size ( ) == len ) {
if ( check ( convert ( s ) ) ) { uniq . insert ( convert ( s ) ) ; } return ; }
for ( int i = 0 ; i <= 3 ; i ++ ) { generate ( s + char ( i + '0' ) , len , uniq ) ; } }
int totalNumbers ( int L , int R ) {
int ans = 0 ;
int max_len = log10 ( R ) + 1 ;
set < int > uniq ; for ( int i = 1 ; i <= max_len ; i ++ ) {
generate ( " " , i , uniq ) ; }
for ( auto x : uniq ) { if ( x >= L && x <= R ) { ans ++ ; } } return ans ; }
int main ( ) { int L = 22 , R = 22 ; cout << totalNumbers ( L , R ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void convertXintoY ( int X , int Y ) {
while ( Y > X ) {
if ( Y % 2 == 0 ) Y /= 2 ;
else if ( Y % 10 == 1 ) Y /= 10 ;
else break ; }
if ( X == Y ) cout << " Yes " ; else cout << " No " ; }
int main ( ) { int X = 100 , Y = 40021 ; convertXintoY ( X , Y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void generateString ( int K ) {
string s = " " ;
for ( int i = 97 ; i < 97 + K ; i ++ ) { s = s + char ( i ) ;
for ( int j = i + 1 ; j < 97 + K ; j ++ ) { s += char ( i ) ; s += char ( j ) ; } }
s += char ( 97 ) ;
cout << s ; }
int main ( ) { int K = 4 ; generateString ( K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findEquation ( int S , int M ) {
cout << "1 ▁ " << ( -1 ) * S << " ▁ " << M << endl ; }
int main ( ) { int S = 5 , M = 6 ; findEquation ( S , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minSteps ( vector < int > a , int n ) {
vector < int > prefix_sum ( n ) ; prefix_sum [ 0 ] = a [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) prefix_sum [ i ] += prefix_sum [ i - 1 ] + a [ i ] ;
int mx = -1 ;
for ( int subgroupsum : prefix_sum ) { int sum = 0 ; int i = 0 ; int grp_count = 0 ;
while ( i < n ) { sum += a [ i ] ;
if ( sum == subgroupsum ) {
grp_count += 1 ; sum = 0 ; }
else if ( sum > subgroupsum ) { grp_count = -1 ; break ; } i += 1 ; }
if ( grp_count > mx ) mx = grp_count ; }
return n - mx ; }
int main ( ) { vector < int > A = { 1 , 2 , 3 , 2 , 1 , 3 } ; int N = A . size ( ) ;
cout << minSteps ( A , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maxOccuringCharacter ( string s ) {
int count0 = 0 , count1 = 0 ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( s [ i ] == '1' ) { count1 ++ ; }
else if ( s [ i ] == '0' ) { count0 ++ ; } }
int prev = -1 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s [ i ] == '1' ) { prev = i ; break ; } }
for ( int i = prev + 1 ; i < s . length ( ) ; i ++ ) {
if ( s [ i ] != ' X ' ) {
if ( s [ i ] == '1' ) { count1 += i - prev - 1 ; prev = i ; }
else {
bool flag = true ; for ( int j = i + 1 ; j < s . length ( ) ; j ++ ) { if ( s [ j ] == '1' ) { flag = false ; prev = j ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . length ( ) ; } } } }
prev = -1 ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s [ i ] == '0' ) { prev = i ; break ; } }
for ( int i = prev + 1 ; i < s . length ( ) ; i ++ ) {
if ( s [ i ] != ' X ' ) {
if ( s [ i ] == '0' ) {
count0 += i - prev - 1 ;
prev = i ; }
else {
bool flag = true ; for ( int j = i + 1 ; j < s . length ( ) ; j ++ ) { if ( s [ j ] == '0' ) { prev = j ; flag = false ; break ; } }
if ( ! flag ) { i = prev ; }
else { i = s . length ( ) ; } } } }
if ( s [ 0 ] == ' X ' ) {
int count = 0 ; int i = 0 ; while ( s [ i ] == ' X ' ) { count ++ ; i ++ ; }
if ( s [ i ] == '1' ) { count1 += count ; } }
if ( s [ ( s . length ( ) - 1 ) ] == ' X ' ) {
int count = 0 ; int i = s . length ( ) - 1 ; while ( s [ i ] == ' X ' ) { count ++ ; i -- ; }
if ( s [ i ] == '0' ) { count0 += count ; } }
if ( count0 == count1 ) { cout << " X " << endl ; }
else if ( count0 > count1 ) { cout << 0 << endl ; }
else cout < < 1 << endl ; }
int main ( ) { string S = " XX10XX10XXX1XX " ; maxOccuringCharacter ( S ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSheets ( int A , int B ) { int area = A * B ;
int count = 1 ;
while ( area % 2 == 0 ) {
area /= 2 ;
count *= 2 ; } return count ; }
int main ( ) { int A = 5 , B = 10 ; cout << maxSheets ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findMinMoves ( int a , int b ) {
int ans = 0 ;
if ( a == b || abs ( a - b ) == 1 ) { ans = a + b ; } else {
int k = min ( a , b ) ;
int j = max ( a , b ) ; ans = 2 * k + 2 * ( j - k ) - 1 ; }
cout << ans ; }
int main ( ) {
int a = 3 , b = 5 ;
findMinMoves ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long cntEvenSumPairs ( long long X , long long Y ) {
long long cntXEvenNums = X / 2 ;
long long cntXOddNums = ( X + 1 ) / 2 ;
long long cntYEvenNums = Y / 2 ;
long long cntYOddNums = ( Y + 1 ) / 2 ;
long long cntPairs = ( cntXEvenNums * 1LL * cntYEvenNums ) + ( cntXOddNums * 1LL * cntYOddNums ) ;
return cntPairs ; }
int main ( ) { long long X = 2 ; long long Y = 3 ; cout << cntEvenSumPairs ( X , Y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minMoves ( vector < int > arr ) { int N = arr . size ( ) ;
if ( N <= 2 ) return 0 ;
int ans = INT_MAX ;
for ( int i = -1 ; i <= 1 ; i ++ ) { for ( int j = -1 ; j <= 1 ; j ++ ) {
int num1 = arr [ 0 ] + i ;
int num2 = arr [ 1 ] + j ; int flag = 1 ; int moves = abs ( i ) + abs ( j ) ;
for ( int idx = 2 ; idx < N ; idx ++ ) {
int num = num1 + num2 ;
if ( abs ( arr [ idx ] - num ) > 1 ) flag = 0 ;
else moves += abs ( arr [ idx ] - num ) ; num1 = num2 ; num2 = num ; }
if ( flag ) ans = min ( ans , moves ) ; } }
if ( ans == INT_MAX ) return -1 ; return ans ; }
int main ( ) { vector < int > arr = { 4 , 8 , 9 , 17 , 27 } ; cout << minMoves ( arr ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void querySum ( int arr [ ] , int N , int Q [ ] [ 2 ] , int M ) {
for ( int i = 0 ; i < M ; i ++ ) { int x = Q [ i ] [ 0 ] ; int y = Q [ i ] [ 1 ] ;
int sum = 0 ;
while ( x < N ) {
sum += arr [ x ] ;
x += y ; } cout << sum << " ▁ " ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 7 , 5 , 4 } ; int Q [ ] [ 2 ] = { { 2 , 1 } , { 3 , 2 } } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int M = sizeof ( Q ) / sizeof ( Q [ 0 ] ) ; querySum ( arr , N , Q , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findBitwiseORGivenXORAND ( int X , int Y ) { return X + Y ; }
int main ( ) { int X = 5 , Y = 2 ; cout << findBitwiseORGivenXORAND ( X , Y ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int GCD ( int a , int b ) {
if ( b == 0 ) return a ;
return GCD ( b , a % b ) ; }
void canReach ( int N , int A , int B , int K ) {
int gcd = GCD ( N , K ) ;
if ( abs ( A - B ) % gcd == 0 ) { cout << " Yes " ; }
else { cout << " No " ; } }
int main ( ) { int N = 5 , A = 2 , B = 1 , K = 2 ;
canReach ( N , A , B , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countOfSubarray ( int arr [ ] , int N ) {
unordered_map < int , int > mp ;
int answer = 0 ;
int sum = 0 ;
mp [ 1 ] ++ ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += arr [ i ] ; answer += mp [ sum - i ] ;
mp [ sum - i ] ++ ; }
cout << answer ; }
int main ( ) {
int arr [ ] = { 1 , 0 , 2 , 1 , 2 , -2 , 2 , 4 } ;
int N = sizeof arr / sizeof arr [ 0 ] ;
countOfSubarray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minAbsDiff ( int N ) {
int sumSet1 = 0 ;
int sumSet2 = 0 ;
for ( int i = N ; i > 0 ; i -- ) {
if ( sumSet1 <= sumSet2 ) { sumSet1 += i ; } else { sumSet2 += i ; } } return abs ( sumSet1 - sumSet2 ) ; }
int main ( ) { int N = 6 ; cout << minAbsDiff ( N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkDigits ( int n ) {
do { int r = n % 10 ;
if ( r == 3 r == 4 r == 6 r == 7 r == 9 ) return false ; n /= 10 ; } while ( n != 0 ) ; return true ; }
bool isPrime ( int n ) { if ( n <= 1 ) return false ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) return false ; } return true ; }
int isAllPrime ( int n ) { return isPrime ( n ) && checkDigits ( n ) ; }
int main ( ) { int N = 101 ; if ( isAllPrime ( N ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minCost ( string str , int a , int b ) {
int openUnbalanced = 0 ;
int closedUnbalanced = 0 ;
int openCount = 0 ;
int closedCount = 0 ; for ( int i = 0 ; str [ i ] != ' \0' ; i ++ ) {
if ( str [ i ] == ' ( ' ) { openUnbalanced ++ ; openCount ++ ; }
else {
if ( openUnbalanced == 0 )
closedUnbalanced ++ ;
else
openUnbalanced -- ;
closedCount ++ ; } }
int result = a * ( abs ( openCount - closedCount ) ) ;
if ( closedCount > openCount ) closedUnbalanced -= ( closedCount - openCount ) ; if ( openCount > closedCount ) openUnbalanced -= ( openCount - closedCount ) ;
result += min ( a * ( openUnbalanced + closedUnbalanced ) , b * closedUnbalanced ) ;
cout << result << endl ; }
int main ( ) { string str = " ) ) ( ) ( ( ) ( ) ( " ; int A = 1 , B = 3 ; minCost ( str , A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countEvenSum ( int low , int high , int k ) {
int even_count = high / 2 - ( low - 1 ) / 2 ; int odd_count = ( high + 1 ) / 2 - low / 2 ; long even_sum = 1 ; long odd_sum = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
long prev_even = even_sum ; long prev_odd = odd_sum ;
even_sum = ( prev_even * even_count ) + ( prev_odd * odd_count ) ;
odd_sum = ( prev_even * odd_count ) + ( prev_odd * even_count ) ; }
cout << ( even_sum ) ; }
int main ( ) {
int low = 4 ; int high = 5 ;
int K = 3 ;
countEvenSum ( low , high , K ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void count ( int n , int k ) { long count = ( long ) ( pow ( 10 , k ) - pow ( 10 , k - 1 ) ) ;
cout << ( count ) ; }
int main ( ) { int n = 2 , k = 1 ; count ( n , k ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int func ( int N , int P ) {
int sumUptoN = ( N * ( N + 1 ) / 2 ) ; int sumOfMultiplesOfP ;
if ( N < P ) { return sumUptoN ; }
else if ( ( N / P ) == 1 ) { return sumUptoN - P + 1 ; }
sumOfMultiplesOfP = ( ( N / P ) * ( 2 * P + ( N / P - 1 ) * P ) ) / 2 ;
return ( sumUptoN + func ( N / P , P ) - sumOfMultiplesOfP ) ; }
int main ( ) {
int N = 10 , P = 5 ;
cout << func ( N , P ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findShifts ( int A [ ] , int N ) {
int shift [ N ] ; for ( int i = 0 ; i < N ; i ++ ) {
if ( i == A [ i ] - 1 ) shift [ i ] = 0 ;
else
shift [ i ] = ( A [ i ] - 1 - i + N ) % N ; }
for ( int i = 0 ; i < N ; i ++ ) cout << shift [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 4 , 3 , 2 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findShifts ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void constructmatrix ( int N ) { bool check = true ; for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( i == j ) { cout << 1 << " ▁ " ; } else if ( check ) {
cout << 2 << " ▁ " ; check = false ; } else {
cout << -2 << " ▁ " ; check = true ; } } cout << endl ; } }
int main ( ) { int N = 5 ; constructmatrix ( 5 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int check ( int unit_digit , int X ) { int times , digit ;
for ( int times = 1 ; times <= 10 ; times ++ ) { digit = ( X * times ) % 10 ; if ( digit == unit_digit ) return times ; }
return -1 ; }
int getNum ( int N , int X ) { int unit_digit ;
unit_digit = N % 10 ;
int times = check ( unit_digit , X ) ;
if ( times == -1 ) return times ;
else {
if ( N >= ( times * X ) )
return times ;
else return -1 ; } }
int main ( ) { int N = 58 , X = 7 ; cout << getNum ( N , X ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minPoints ( int n , int m ) { int ans = 0 ;
if ( ( n % 2 != 0 ) && ( m % 2 != 0 ) ) { ans = ( ( n * m ) / 2 ) + 1 ; } else { ans = ( n * m ) / 2 ; }
return ans ; }
int main ( ) {
int N = 5 , M = 7 ;
cout << minPoints ( N , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ll  long long int
string getLargestString ( string s , ll k ) {
vector < int > frequency_array ( 26 , 0 ) ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) { frequency_array [ s [ i ] - ' a ' ] ++ ; }
string ans = " " ;
for ( int i = 25 ; i >= 0 ; ) {
if ( frequency_array [ i ] > k ) {
int temp = k ; string st ( 1 , i + ' a ' ) ; while ( temp > 0 ) {
ans += st ; temp -- ; } frequency_array [ i ] -= k ;
int j = i - 1 ; while ( frequency_array [ j ] <= 0 && j >= 0 ) { j -- ; }
if ( frequency_array [ j ] > 0 && j >= 0 ) { string str ( 1 , j + ' a ' ) ; ans += str ; frequency_array [ j ] -= 1 ; } else {
break ; } }
else if ( frequency_array [ i ] > 0 ) {
int temp = frequency_array [ i ] ; frequency_array [ i ] -= temp ; string st ( 1 , i + ' a ' ) ; while ( temp > 0 ) { ans += st ; temp -- ; } }
else { i -- ; } } return ans ; }
int main ( ) { string S = " xxxxzza " ; int k = 3 ; cout << getLargestString ( S , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minOperations ( int a [ ] , int b [ ] , int n ) {
int minA = * min_element ( a , a + n ) ;
for ( int x = minA ; x >= 0 ; x -- ) {
bool check = 1 ;
int operations = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( x % b [ i ] == a [ i ] % b [ i ] ) { operations += ( a [ i ] - x ) / b [ i ] ; }
else { check = 0 ; break ; } } if ( check ) return operations ; } return -1 ; }
int main ( ) { int N = 5 ; int A [ N ] = { 5 , 7 , 10 , 5 , 15 } ; int B [ N ] = { 2 , 2 , 1 , 3 , 5 } ; cout << minOperations ( A , B , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getLargestSum ( int N ) {
int max_sum = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { for ( int j = i + 1 ; j <= N ; j ++ ) {
if ( i * j % ( i + j ) == 0 )
max_sum = max ( max_sum , i + j ) ; } }
return max_sum ; }
int main ( ) { int N = 25 ; int max_sum = getLargestSum ( N ) ; cout << max_sum << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSubArraySum ( int a [ ] , int size ) { int max_so_far = INT_MIN , max_ending_here = 0 ;
for ( int i = 0 ; i < size ; i ++ ) { max_ending_here = max_ending_here + a [ i ] ; if ( max_ending_here < 0 ) max_ending_here = 0 ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; } return max_so_far ; }
int maxSum ( int a [ ] , int n ) {
int S = 0 ;
for ( int i = 0 ; i < n ; i ++ ) S += a [ i ] ; int X = maxSubArraySum ( a , n ) ;
return 2 * X - S ; }
int main ( ) { int a [ ] = { -1 , -2 , -3 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int max_sum = maxSum ( a , n ) ; cout << max_sum ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) { int flag = 1 ;
for ( int i = 2 ; i * i <= n ; i ++ ) { if ( n % i == 0 ) { flag = 0 ; break ; } } return ( flag == 1 ? true : false ) ; }
bool isPerfectSquare ( int x ) {
long double sr = sqrt ( x ) ;
return ( ( sr - floor ( sr ) ) == 0 ) ; }
int countInterestingPrimes ( int n ) { int answer = 0 ; for ( int i = 2 ; i <= n ; i ++ ) {
if ( isPrime ( i ) ) {
for ( int j = 1 ; j * j * j * j <= i ; j ++ ) {
if ( isPerfectSquare ( i - j * j * j * j ) ) { answer ++ ; break ; } } } }
return answer ; }
int main ( ) { int N = 10 ; cout << countInterestingPrimes ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void decBinary ( int arr [ ] , int n ) { int k = log2 ( n ) ; while ( n > 0 ) { arr [ k -- ] = n % 2 ; n /= 2 ; } }
int binaryDec ( int arr [ ] , int n ) { int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) ans += arr [ i ] << ( n - i - 1 ) ; return ans ; }
int maxNum ( int n , int k ) {
int l = log2 ( n ) + 1 ;
int a [ l ] = { 0 } ; decBinary ( a , n ) ;
int cn = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( a [ i ] == 0 && cn < k ) { a [ i ] = 1 ; cn ++ ; } }
return binaryDec ( a , l ) ; }
int main ( ) { int n = 4 , k = 1 ; cout << maxNum ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findSubSeq ( int arr [ ] , int n , int sum ) { for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( sum < arr [ i ] ) arr [ i ] = -1 ;
else sum -= arr [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != -1 ) cout << arr [ i ] << " ▁ " ; } }
int main ( ) { int arr [ ] = { 17 , 25 , 46 , 94 , 201 , 400 } ; int n = sizeof ( arr ) / sizeof ( int ) ; int sum = 272 ; findSubSeq ( arr , n , sum ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 26 ;
char maxAlpha ( string str , int len ) {
int first [ MAX ] , last [ MAX ] ;
for ( int i = 0 ; i < MAX ; i ++ ) { first [ i ] = -1 ; last [ i ] = -1 ; }
for ( int i = 0 ; i < len ; i ++ ) { int index = ( str [ i ] - ' a ' ) ;
if ( first [ index ] == -1 ) first [ index ] = i ; last [ index ] = i ; }
int ans = -1 , maxVal = -1 ;
for ( int i = 0 ; i < MAX ; i ++ ) {
if ( first [ i ] == -1 ) continue ;
if ( ( last [ i ] - first [ i ] ) > maxVal ) { maxVal = last [ i ] - first [ i ] ; ans = i ; } } return ( char ) ( ans + ' a ' ) ; }
int main ( ) { string str = " abbba " ; int len = str . length ( ) ; cout << maxAlpha ( str , len ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100001
void find_distinct ( int a [ ] , int n , int q , int queries [ ] ) { int check [ MAX ] = { 0 } ; int idx [ MAX ] ; int cnt = 1 ; for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( check [ a [ i ] ] == 0 ) {
idx [ i ] = cnt ; check [ a [ i ] ] = 1 ; cnt ++ ; } else {
idx [ i ] = cnt - 1 ; } }
for ( int i = 0 ; i < q ; i ++ ) { int m = queries [ i ] ; cout << idx [ m ] << " ▁ " ; } }
int main ( ) { int a [ ] = { 1 , 2 , 3 , 1 , 2 , 3 , 4 , 5 } ; int n = sizeof ( a ) / sizeof ( int ) ; int queries [ ] = { 0 , 3 , 5 , 7 } ; int q = sizeof ( queries ) / sizeof ( int ) ; find_distinct ( a , n , q , queries ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 24 ;
int countOp ( int x ) {
int arr [ MAX ] ; arr [ 0 ] = 1 ; for ( int i = 1 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] * 2 ;
int temp = x ; bool flag = true ;
int ans ;
int operations = 0 ; bool flag2 = false ; for ( int i = 0 ; i < MAX ; i ++ ) { if ( arr [ i ] - 1 == x ) flag2 = true ;
if ( arr [ i ] > x ) { ans = i ; break ; } }
if ( flag2 ) return 0 ; while ( flag ) {
if ( arr [ ans ] < x ) ans ++ ; operations ++ ;
for ( int i = 0 ; i < MAX ; i ++ ) { int take = x ^ ( arr [ i ] - 1 ) ; if ( take <= arr [ ans ] - 1 ) {
if ( take > temp ) temp = take ; } }
if ( temp == arr [ ans ] - 1 ) { flag = false ; break ; } temp ++ ; operations ++ ; x = temp ; if ( x == arr [ ans ] - 1 ) flag = false ; }
return operations ; }
int main ( ) { int x = 39 ; cout << countOp ( x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minOperations ( int * arr , int n ) { int maxi , result = 0 ;
vector < int > freq ( 1000001 , 0 ) ; for ( int i = 0 ; i < n ; i ++ ) { int x = arr [ i ] ; freq [ x ] ++ ; }
maxi = * ( max_element ( arr , arr + n ) ) ; for ( int i = 1 ; i <= maxi ; i ++ ) { if ( freq [ i ] != 0 ) {
for ( int j = i * 2 ; j <= maxi ; j = j + i ) {
freq [ j ] = 0 ; }
result ++ ; } } return result ; }
int main ( ) { int arr [ ] = { 2 , 4 , 2 , 4 , 4 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minOperations ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minGCD ( int arr [ ] , int n ) { int minGCD = 0 ;
for ( int i = 0 ; i < n ; i ++ ) minGCD = __gcd ( minGCD , arr [ i ] ) ; return minGCD ; }
int minLCM ( int arr [ ] , int n ) { int minLCM = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) minLCM = min ( minLCM , arr [ i ] ) ; return minLCM ; }
int main ( ) { int arr [ ] = { 2 , 66 , 14 , 521 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " LCM ▁ = ▁ " << minLCM ( arr , n ) << " , ▁ GCD ▁ = ▁ " << minGCD ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string formStringMinOperations ( string s ) {
int count [ 3 ] = { 0 } ; for ( auto & c : s ) count ++ ;
int processed [ 3 ] = { 0 } ;
int reqd = ( int ) s . size ( ) / 3 ; for ( int i = 0 ; i < s . size ( ) ; i ++ ) {
if ( count [ s [ i ] - '0' ] == reqd ) continue ;
if ( s [ i ] == '0' && count [ 0 ] > reqd && processed [ 0 ] >= reqd ) {
if ( count [ 1 ] < reqd ) { s [ i ] = '1' ; count [ 1 ] ++ ; count [ 0 ] -- ; }
else if ( count [ 2 ] < reqd ) { s [ i ] = '2' ; count [ 2 ] ++ ; count [ 0 ] -- ; } }
if ( s [ i ] == '1' && count [ 1 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = '0' ; count [ 0 ] ++ ; count [ 1 ] -- ; } else if ( count [ 2 ] < reqd && processed [ 1 ] >= reqd ) { s [ i ] = '2' ; count [ 2 ] ++ ; count [ 1 ] -- ; } }
if ( s [ i ] == '2' && count [ 2 ] > reqd ) { if ( count [ 0 ] < reqd ) { s [ i ] = '0' ; count [ 0 ] ++ ; count [ 2 ] -- ; } else if ( count [ 1 ] < reqd ) { s [ i ] = '1' ; count [ 1 ] ++ ; count [ 2 ] -- ; } }
processed [ s [ i ] - '0' ] ++ ; } return s ; }
int main ( ) { string s = "011200" ; cout << formStringMinOperations ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinimumAdjacentSwaps ( int arr [ ] , int N ) {
bool visited [ N + 1 ] ; int minimumSwaps = 0 ; memset ( visited , false , sizeof ( visited ) ) ; for ( int i = 0 ; i < 2 * N ; i ++ ) {
if ( visited [ arr [ i ] ] == false ) { visited [ arr [ i ] ] = true ;
int count = 0 ; for ( int j = i + 1 ; j < 2 * N ; j ++ ) {
if ( visited [ arr [ j ] ] == false ) count ++ ;
else if ( arr [ i ] == arr [ j ] ) minimumSwaps += count ; } } } return minimumSwaps ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 3 , 1 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; N /= 2 ; cout << findMinimumAdjacentSwaps ( arr , N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool possibility ( unordered_map < int , int > m , int length , string s ) {
int countodd = 0 ; for ( int i = 0 ; i < length ; i ++ ) {
if ( m [ s [ i ] - '0' ] & 1 ) countodd ++ ;
if ( countodd > 1 ) return false ; } return true ; }
void largestPalindrome ( string s ) {
int l = s . length ( ) ;
unordered_map < int , int > m ; for ( int i = 0 ; i < l ; i ++ ) m [ s [ i ] - '0' ] ++ ;
if ( possibility ( m , l , s ) == false ) { cout << " Palindrome ▁ cannot ▁ be ▁ formed " ; return ; }
char largest [ l ] ;
int front = 0 ;
for ( int i = 9 ; i >= 0 ; i -- ) {
if ( m [ i ] & 1 ) {
largest [ l / 2 ] = char ( i + 48 ) ;
m [ i ] -- ;
while ( m [ i ] > 0 ) { largest [ front ] = char ( i + 48 ) ; largest [ l - front - 1 ] = char ( i + 48 ) ; m [ i ] -= 2 ; front ++ ; } } else {
while ( m [ i ] > 0 ) {
largest [ front ] = char ( i + 48 ) ; largest [ l - front - 1 ] = char ( i + 48 ) ;
m [ i ] -= 2 ;
front ++ ; } } }
for ( int i = 0 ; i < l ; i ++ ) cout << largest [ i ] ; }
int main ( ) { string s = "313551" ; largestPalindrome ( s ) ; return 0 ; }
#include <iostream> NEW_LINE #include <vector> NEW_LINE #include <algorithm> NEW_LINE using namespace std ;
long swapCount ( string s ) {
vector < int > pos ; for ( int i = 0 ; i < s . length ( ) ; ++ i ) if ( s [ i ] == ' [ ' ) pos . push_back ( i ) ;
int count = 0 ;
int p = 0 ;
long sum = 0 ; for ( int i = 0 ; i < s . length ( ) ; ++ i ) {
if ( s [ i ] == ' [ ' ) { ++ count ; ++ p ; } else if ( s [ i ] == ' ] ' ) -- count ;
if ( count < 0 ) {
sum += pos [ p ] - i ; swap ( s [ i ] , s [ pos [ p ] ] ) ; ++ p ;
count = 1 ; } } return sum ; }
int main ( ) { string s = " [ ] ] [ ] [ " ; cout << swapCount ( s ) << " STRNEWLINE " ; s = " [ [ ] [ ] ] " ; cout << swapCount ( s ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minimumCostOfBreaking ( int X [ ] , int Y [ ] , int m , int n ) { int res = 0 ;
sort ( X , X + m , greater < int > ( ) ) ;
sort ( Y , Y + n , greater < int > ( ) ) ;
int hzntl = 1 , vert = 1 ;
int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( X [ i ] > Y [ j ] ) { res += X [ i ] * vert ;
hzntl ++ ; i ++ ; } else { res += Y [ j ] * hzntl ;
vert ++ ; j ++ ; } }
int total = 0 ; while ( i < m ) total += X [ i ++ ] ; res += total * vert ;
total = 0 ; while ( j < n ) total += Y [ j ++ ] ; res += total * hzntl ; return res ; }
int main ( ) { int m = 6 , n = 4 ; int X [ m - 1 ] = { 2 , 1 , 3 , 1 , 4 } ; int Y [ n - 1 ] = { 4 , 1 , 2 } ; cout << minimumCostOfBreaking ( X , Y , m - 1 , n - 1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getMin ( int x , int y , int z ) { return min ( min ( x , y ) , z ) ; }
int editDistance ( string str1 , string str2 , int m , int n ) {
int dp [ m + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) {
if ( i == 0 )
dp [ i ] [ j ] = j ;
else if ( j == 0 )
dp [ i ] [ j ] = i ;
else if ( str1 [ i - 1 ] == str2 [ j - 1 ] ) dp [ i ] [ j ] = dp [ i - 1 ] [ j - 1 ] ;
else {
dp [ i ] [ j ] = 1 + getMin ( dp [ i ] [ j - 1 ] , dp [ i - 1 ] [ j ] , dp [ i - 1 ] [ j - 1 ] ) ; } } }
return dp [ m ] [ n ] ; }
void minimumSteps ( string & S , int N ) {
int ans = INT_MAX ;
for ( int i = 1 ; i < N ; i ++ ) { string S1 = S . substr ( 0 , i ) ; string S2 = S . substr ( i ) ;
int count = editDistance ( S1 , S2 , S1 . length ( ) , S2 . length ( ) ) ;
ans = min ( ans , count ) ; }
cout << ans << ' ' }
int main ( ) { string S = " aabb " ; int N = S . length ( ) ; minimumSteps ( S , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minimumOperations ( int N ) {
int dp [ N + 1 ] ; int i ;
for ( int i = 0 ; i <= N ; i ++ ) { dp [ i ] = 1e9 ; }
dp [ 2 ] = 0 ;
for ( i = 2 ; i <= N ; i ++ ) {
if ( dp [ i ] == 1e9 ) continue ;
if ( i * 5 <= N ) { dp [ i * 5 ] = min ( dp [ i * 5 ] , dp [ i ] + 1 ) ; }
if ( i + 3 <= N ) { dp [ i + 3 ] = min ( dp [ i + 3 ] , dp [ i ] + 1 ) ; } }
if ( dp [ N ] == 1e9 ) return -1 ;
return dp [ N ] ; }
int main ( ) { int N = 25 ; cout << minimumOperations ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MaxProfit ( int arr [ ] , int n , int transactionFee ) { int buy = - arr [ 0 ] ; int sell = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { int temp = buy ;
buy = max ( buy , sell - arr [ i ] ) ; sell = max ( sell , temp + arr [ i ] - transactionFee ) ; }
return max ( sell , buy ) ; }
int main ( ) {
int arr [ ] = { 6 , 1 , 7 , 2 , 8 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int transactionFee = 2 ;
cout << MaxProfit ( arr , n , transactionFee ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int start [ 3 ] [ 3 ] ;
int ending [ 3 ] [ 3 ] ;
void calculateStart ( int n , int m ) {
for ( int i = 1 ; i < m ; ++ i ) { start [ 0 ] [ i ] += start [ 0 ] [ i - 1 ] ; }
for ( int i = 1 ; i < n ; ++ i ) { start [ i ] [ 0 ] += start [ i - 1 ] [ 0 ] ; }
for ( int i = 1 ; i < n ; ++ i ) { for ( int j = 1 ; j < m ; ++ j ) {
start [ i ] [ j ] += max ( start [ i - 1 ] [ j ] , start [ i ] [ j - 1 ] ) ; } } }
void calculateEnd ( int n , int m ) {
for ( int i = n - 2 ; i >= 0 ; -- i ) { ending [ i ] [ m - 1 ] += ending [ i + 1 ] [ m - 1 ] ; }
for ( int i = m - 2 ; i >= 0 ; -- i ) { ending [ n - 1 ] [ i ] += ending [ n - 1 ] [ i + 1 ] ; }
for ( int i = n - 2 ; i >= 0 ; -- i ) { for ( int j = m - 2 ; j >= 0 ; -- j ) {
ending [ i ] [ j ] += max ( ending [ i + 1 ] [ j ] , ending [ i ] [ j + 1 ] ) ; } } }
void maximumPathSum ( int mat [ ] [ 3 ] , int n , int m , int q , int coordinates [ ] [ 2 ] ) {
for ( int i = 0 ; i < n ; ++ i ) { for ( int j = 0 ; j < m ; ++ j ) { start [ i ] [ j ] = mat [ i ] [ j ] ; ending [ i ] [ j ] = mat [ i ] [ j ] ; } }
calculateStart ( n , m ) ;
calculateEnd ( n , m ) ;
int ans = 0 ;
for ( int i = 0 ; i < q ; ++ i ) { int X = coordinates [ i ] [ 0 ] - 1 ; int Y = coordinates [ i ] [ 1 ] - 1 ;
ans = max ( ans , start [ X ] [ Y ] + ending [ X ] [ Y ] - mat [ X ] [ Y ] ) ; }
cout << ans ; }
int main ( ) { int mat [ ] [ 3 ] = { { 1 , 2 , 3 } , { 4 , 5 , 6 } , { 7 , 8 , 9 } } ; int N = 3 ; int M = 3 ; int Q = 2 ; int coordinates [ ] [ 2 ] = { { 1 , 2 } , { 2 , 2 } } ; maximumPathSum ( mat , N , M , Q , coordinates ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MaxSubsetlength ( vector < string > arr , int A , int B ) {
int dp [ A + 1 ] [ B + 1 ] ; memset ( dp , 0 , sizeof ( dp ) ) ;
for ( auto & str : arr ) {
int zeros = count ( str . begin ( ) , str . end ( ) , '0' ) ; int ones = count ( str . begin ( ) , str . end ( ) , '1' ) ;
for ( int i = A ; i >= zeros ; i -- )
for ( int j = B ; j >= ones ; j -- )
dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - zeros ] [ j - ones ] + 1 ) ; }
return dp [ A ] [ B ] ; }
int main ( ) { vector < string > arr = { "1" , "0" , "0001" , "10" , "111001" } ; int A = 5 , B = 3 ; cout << MaxSubsetlength ( arr , A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int numOfWays ( vector < vector < int > > a , int n , int i , set < int > & blue ) {
if ( i == n ) return 1 ;
int count = 0 ;
for ( int j = 0 ; j < n ; j ++ ) {
if ( a [ i ] [ j ] == 1 && blue . find ( j ) == blue . end ( ) ) { blue . insert ( j ) ; count += numOfWays ( a , n , i + 1 , blue ) ; blue . erase ( j ) ; } } return count ; }
int main ( ) { int n = 3 ; vector < vector < int > > mat = { { 0 , 1 , 1 } , { 1 , 0 , 1 } , { 1 , 1 , 1 } } ; set < int > mpp ; cout << ( numOfWays ( mat , n , 0 , mpp ) ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minCost ( int arr [ ] , int n ) {
if ( n < 3 ) { cout << arr [ 0 ] ; return ; }
int * dp = new int [ n ] ;
dp [ 0 ] = arr [ 0 ] ; dp [ 1 ] = dp [ 0 ] + arr [ 1 ] + arr [ 2 ] ;
for ( int i = 2 ; i < n - 1 ; i ++ ) dp [ i ] = min ( dp [ i - 2 ] + arr [ i ] , dp [ i - 1 ] + arr [ i ] + arr [ i + 1 ] ) ;
dp [ n - 1 ] = min ( dp [ n - 2 ] , dp [ n - 3 ] + arr [ n - 1 ] ) ;
cout << dp [ n - 1 ] ; }
int main ( ) { int arr [ ] = { 9 , 4 , 6 , 8 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; minCost ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define M  1000000007
long long power ( long long X , long long Y ) {
long long res = 1 ;
X = X % M ;
if ( X == 0 ) return 0 ;
while ( Y > 0 ) {
if ( Y & 1 ) {
res = ( res * X ) % M ; }
Y = Y >> 1 ;
X = ( X * X ) % M ; } return res ; }
int findValue ( long long int n ) {
long long X = 0 ;
long long pow_10 = 1 ;
while ( n ) {
if ( n & 1 ) {
X += pow_10 ; }
pow_10 *= 10 ;
n /= 2 ; }
X = ( X * 2 ) % M ;
long long res = power ( 2 , X ) ; return res ; }
int main ( ) { long long n = 2 ; cout << findValue ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define M  1000000007
long long power ( long long X , long long Y ) {
long long res = 1 ;
X = X % M ;
if ( X == 0 ) return 0 ;
while ( Y > 0 ) {
if ( Y & 1 ) {
res = ( res * X ) % M ; }
Y = Y >> 1 ;
X = ( X * X ) % M ; } return res ; }
long long findValue ( long long N ) {
long long dp [ N + 1 ] ;
dp [ 1 ] = 2 ; dp [ 2 ] = 1024 ;
for ( int i = 3 ; i <= N ; i ++ ) {
int y = ( i & ( - i ) ) ;
int x = i - y ;
if ( x == 0 ) {
dp [ i ] = power ( dp [ i / 2 ] , 10 ) ; } else {
dp [ i ] = ( dp [ x ] * dp [ y ] ) % M ; } } return ( dp [ N ] * dp [ N ] ) % M ; }
int main ( ) { long long n = 150 ; cout << findValue ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findWays ( int N ) {
if ( N == 0 ) { return 1 ; }
int cnt = 0 ;
for ( int i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i ) ; } }
return cnt ; }
int main ( ) { int N = 4 ;
cout << findWays ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int checkEqualSumUtil ( int arr [ ] , int N , int sm1 , int sm2 , int sm3 , int j ) {
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; } else {
int l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
int m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
int r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
return max ( max ( l , m ) , r ) ; } }
void checkEqualSum ( int arr [ ] , int N ) {
int sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { cout << " Yes " ; } else { cout << " No " ; } }
int main ( ) {
int arr [ ] = { 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
checkEqualSum ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; map < string , int > dp ;
int checkEqualSumUtil ( int arr [ ] , int N , int sm1 , int sm2 , int sm3 , int j ) { string s = to_string ( sm1 ) + " _ " + to_string ( sm2 ) + to_string ( j ) ;
if ( j == N ) { if ( sm1 == sm2 && sm2 == sm3 ) return 1 ; else return 0 ; }
if ( dp . find ( s ) != dp . end ( ) ) return dp [ s ] ; else {
int l = checkEqualSumUtil ( arr , N , sm1 + arr [ j ] , sm2 , sm3 , j + 1 ) ;
int m = checkEqualSumUtil ( arr , N , sm1 , sm2 + arr [ j ] , sm3 , j + 1 ) ;
int r = checkEqualSumUtil ( arr , N , sm1 , sm2 , sm3 + arr [ j ] , j + 1 ) ;
return dp [ s ] = max ( max ( l , m ) , r ) ; } }
void checkEqualSum ( int arr [ ] , int N ) {
int sum1 , sum2 , sum3 ; sum1 = sum2 = sum3 = 0 ;
if ( checkEqualSumUtil ( arr , N , sum1 , sum2 , sum3 , 0 ) == 1 ) { cout << " Yes " ; } else { cout << " No " ; } }
int main ( ) {
int arr [ ] = { 17 , 34 , 59 , 23 , 17 , 67 , 57 , 2 , 18 , 59 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
checkEqualSum ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void precompute ( int nextpos [ ] , int arr [ ] , int N ) {
nextpos [ N - 1 ] = N ; for ( int i = N - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] == arr [ i + 1 ] ) nextpos [ i ] = nextpos [ i + 1 ] ; else nextpos [ i ] = i + 1 ; } }
void findIndex ( int query [ ] [ 3 ] , int arr [ ] , int N , int Q ) {
int nextpos [ N ] ; precompute ( nextpos , arr , N ) ; for ( int i = 0 ; i < Q ; i ++ ) { int l , r , x ; l = query [ i ] [ 0 ] ; r = query [ i ] [ 1 ] ; x = query [ i ] [ 2 ] ; int ans = -1 ;
if ( arr [ l ] != x ) ans = l ;
else {
int d = nextpos [ l ] ;
if ( d <= r ) ans = d ; } cout << ans << " STRNEWLINE " ; } }
int main ( ) { int N , Q ; N = 6 ; Q = 3 ; int arr [ ] = { 1 , 2 , 1 , 1 , 3 , 5 } ; int query [ Q ] [ 3 ] = { { 0 , 3 , 1 } , { 1 , 5 , 2 } , { 2 , 3 , 1 } } ; findIndex ( query , arr , N , Q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define mod  10000000007
long long countWays ( string s , string t , int k ) {
int n = s . size ( ) ;
int a = 0 , b = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { string p = s . substr ( i , n - i ) + s . substr ( 0 , i ) ;
if ( p == t ) a ++ ; else b ++ ; }
vector < long long > dp1 ( k + 1 ) , dp2 ( k + 1 ) ; if ( s == t ) { dp1 [ 0 ] = 1 ; dp2 [ 0 ] = 0 ; } else { dp1 [ 0 ] = 0 ; dp2 [ 0 ] = 1 ; }
for ( int i = 1 ; i <= k ; i ++ ) { dp1 [ i ] = ( ( dp1 [ i - 1 ] * ( a - 1 ) ) % mod + ( dp2 [ i - 1 ] * a ) % mod ) % mod ; dp2 [ i ] = ( ( dp1 [ i - 1 ] * ( b ) ) % mod + ( dp2 [ i - 1 ] * ( b - 1 ) ) % mod ) % mod ; }
return dp1 [ k ] ; }
int main ( ) {
string S = " ab " , T = " ab " ;
int K = 2 ;
cout << countWays ( S , T , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minOperation ( int k ) {
vector < int > dp ( k + 1 , 0 ) ; for ( int i = 1 ; i <= k ; i ++ ) { dp [ i ] = dp [ i - 1 ] + 1 ;
if ( i % 2 == 0 ) { dp [ i ] = min ( dp [ i ] , dp [ i / 2 ] + 1 ) ; } } return dp [ k ] ; }
int main ( ) { int K = 12 ; cout << minOperation ( k ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSum ( int p0 , int p1 , int a [ ] , int pos , int n ) { if ( pos == n ) { if ( p0 == p1 ) return p0 ; else return 0 ; }
int ans = maxSum ( p0 , p1 , a , pos + 1 , n ) ;
ans = max ( ans , maxSum ( p0 + a [ pos ] , p1 , a , pos + 1 , n ) ) ;
ans = max ( ans , maxSum ( p0 , p1 + a [ pos ] , a , pos + 1 , n ) ) ; return ans ; }
int main ( ) {
int n = 4 ; int a [ n ] = { 1 , 2 , 3 , 6 } ; cout << maxSum ( 0 , 0 , a , 0 , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSum ( int a [ ] , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += a [ i ] ; int limit = 2 * sum + 1 ;
int dp [ n + 1 ] [ limit ] ;
for ( int i = 0 ; i < n + 1 ; i ++ ) { for ( int j = 0 ; j < limit ; j ++ ) dp [ i ] [ j ] = INT_MIN ; }
dp [ 0 ] [ sum ] = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 0 ; j < limit ; j ++ ) {
if ( ( j - a [ i - 1 ] ) >= 0 && dp [ i - 1 ] [ j - a [ i - 1 ] ] != INT_MIN ) dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j - a [ i - 1 ] ] + a [ i - 1 ] ) ;
if ( ( j + a [ i - 1 ] ) < limit && dp [ i - 1 ] [ j + a [ i - 1 ] ] != INT_MIN ) dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j + a [ i - 1 ] ] ) ;
if ( dp [ i - 1 ] [ j ] != INT_MIN ) dp [ i ] [ j ] = max ( dp [ i ] [ j ] , dp [ i - 1 ] [ j ] ) ; } } return dp [ n ] [ sum ] ; }
int main ( ) { int n = 4 ; int a [ n ] = { 1 , 2 , 3 , 6 } ; cout << maxSum ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int fib [ 100005 ] ;
void computeFibonacci ( ) { fib [ 0 ] = 1 ; fib [ 1 ] = 1 ; for ( int i = 2 ; i < 100005 ; i ++ ) { fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ; } }
int countString ( string str ) {
int ans = 1 ; int cnt = 1 ; for ( int i = 1 ; str [ i ] ; i ++ ) {
if ( str [ i ] == str [ i - 1 ] ) { cnt ++ ; }
else { ans = ans * fib [ cnt ] ; cnt = 1 ; } }
ans = ans * fib [ cnt ] ;
return ans ; }
int main ( ) { string str = " abdllldefkkkk " ;
computeFibonacci ( ) ;
cout << countString ( str ) ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE #define MAX  100001 NEW_LINE using namespace std ;
void printGolombSequence ( int N ) {
int arr [ MAX ] ;
int cnt = 0 ;
arr [ 0 ] = 0 ; arr [ 1 ] = 1 ;
map < int , int > M ;
M [ 2 ] = 2 ;
for ( int i = 2 ; i <= N ; i ++ ) {
if ( cnt == 0 ) { arr [ i ] = 1 + arr [ i - 1 ] ; cnt = M [ arr [ i ] ] ; cnt -- ; }
else { arr [ i ] = arr [ i - 1 ] ; cnt -- ; }
M [ i ] = arr [ i ] ; }
for ( int i = 1 ; i <= N ; i ++ ) { cout << arr [ i ] << ' ▁ ' ; } }
int main ( ) { int N = 11 ; printGolombSequence ( N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int number_of_ways ( int n ) {
int includes_3 [ n + 1 ] = { } ;
int not_includes_3 [ n + 1 ] = { } ;
includes_3 [ 3 ] = 1 ; not_includes_3 [ 1 ] = 1 ; not_includes_3 [ 2 ] = 2 ; not_includes_3 [ 3 ] = 3 ;
for ( int i = 4 ; i <= n ; i ++ ) { includes_3 [ i ] = includes_3 [ i - 1 ] + includes_3 [ i - 2 ] + not_includes_3 [ i - 3 ] ; not_includes_3 [ i ] = not_includes_3 [ i - 1 ] + not_includes_3 [ i - 2 ] ; } return includes_3 [ n ] ; }
int main ( ) { int n = 7 ; cout << number_of_ways ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 100000 ;
int divisors [ MAX ] ;
int generateDivisors ( int n ) { for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) { if ( n / i == i ) { divisors [ i ] ++ ; } else { divisors [ i ] ++ ; divisors [ n / i ] ++ ; } } } }
int findMaxMultiples ( int * arr , int n ) {
int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
ans = max ( divisors [ arr [ i ] ] , ans ) ;
generateDivisors ( arr [ i ] ) ; } return ans ; }
int main ( ) { int arr [ ] = { 8 , 1 , 28 , 4 , 2 , 6 , 7 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << findMaxMultiples ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define n  3 NEW_LINE #define maxV  20 NEW_LINE using namespace std ;
int dp [ n ] [ n ] [ maxV ] ;
int v [ n ] [ n ] [ maxV ] ;
int countWays ( int i , int j , int x , int arr [ ] [ n ] ) {
if ( i == n j == n ) return 0 ; x = ( x & arr [ i ] [ j ] ) ; if ( x == 0 ) return 0 ; if ( i == n - 1 && j == n - 1 ) return 1 ;
if ( v [ i ] [ j ] [ x ] ) return dp [ i ] [ j ] [ x ] ; v [ i ] [ j ] [ x ] = 1 ;
dp [ i ] [ j ] [ x ] = countWays ( i + 1 , j , x , arr ) + countWays ( i , j + 1 , x , arr ) ; return dp [ i ] [ j ] [ x ] ; }
int main ( ) { int arr [ n ] [ n ] = { { 1 , 2 , 1 } , { 1 , 1 , 0 } , { 2 , 1 , 1 } } ; cout << countWays ( 0 , 0 , arr [ 0 ] [ 0 ] , arr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int N = 3 ;
int FindMaximumSum ( int ind , int kon , int a [ ] , int b [ ] , int c [ ] , int n , int dp [ ] [ N ] ) {
if ( ind == n ) return 0 ;
if ( dp [ ind ] [ kon ] != -1 ) return dp [ ind ] [ kon ] ; int ans = -1e9 + 5 ;
if ( kon == 0 ) { ans = max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon == 1 ) { ans = max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; ans = max ( ans , c [ ind ] + FindMaximumSum ( ind + 1 , 2 , a , b , c , n , dp ) ) ; }
else if ( kon == 2 ) { ans = max ( ans , a [ ind ] + FindMaximumSum ( ind + 1 , 1 , a , b , c , n , dp ) ) ; ans = max ( ans , b [ ind ] + FindMaximumSum ( ind + 1 , 0 , a , b , c , n , dp ) ) ; } return dp [ ind ] [ kon ] = ans ; }
int main ( ) { int a [ ] = { 6 , 8 , 2 , 7 , 4 , 2 , 7 } ; int b [ ] = { 7 , 8 , 5 , 8 , 6 , 3 , 5 } ; int c [ ] = { 8 , 3 , 2 , 6 , 8 , 4 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int dp [ n ] [ N ] ; memset ( dp , -1 , sizeof dp ) ;
int x = FindMaximumSum ( 0 , 0 , a , b , c , n , dp ) ;
int y = FindMaximumSum ( 0 , 1 , a , b , c , n , dp ) ;
int z = FindMaximumSum ( 0 , 2 , a , b , c , n , dp ) ;
cout << max ( x , max ( y , z ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int mod = 1000000007 ;
int noOfBinaryStrings ( int N , int k ) { int dp [ 100002 ] ; for ( int i = 1 ; i <= k - 1 ; i ++ ) { dp [ i ] = 1 ; } dp [ k ] = 2 ; for ( int i = k + 1 ; i <= N ; i ++ ) { dp [ i ] = ( dp [ i - 1 ] + dp [ i - k ] ) % mod ; } return dp [ N ] ; }
int main ( ) { int N = 4 ; int K = 2 ; cout << noOfBinaryStrings ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findWaysToPair ( int p ) {
int dp [ p + 1 ] ; dp [ 1 ] = 1 ; dp [ 2 ] = 2 ;
for ( int i = 3 ; i <= p ; i ++ ) { dp [ i ] = dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ; } return dp [ p ] ; }
int main ( ) { int p = 3 ; cout << findWaysToPair ( p ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int CountWays ( int n ) {
if ( n == 0 ) { return 1 ; } if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 1 + 1 ; }
return CountWays ( n - 1 ) + CountWays ( n - 3 ) ; }
int main ( ) { int n = 10 ; cout << CountWays ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > factors ( int n ) {
vector < int > v ; v . push_back ( 1 ) ;
for ( int i = 2 ; i <= sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) { v . push_back ( i ) ;
if ( n / i != i ) { v . push_back ( n / i ) ; } } }
return v ; }
bool checkAbundant ( int n ) { vector < int > v ; int sum = 0 ;
v = factors ( n ) ;
for ( int i = 0 ; i < v . size ( ) ; i ++ ) { sum += v [ i ] ; }
if ( sum > n ) return true ; else return false ; }
bool checkSemiPerfect ( int n ) { vector < int > v ;
v = factors ( n ) ;
sort ( v . begin ( ) , v . end ( ) ) ; int r = v . size ( ) ;
bool subset [ r + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= r ; i ++ ) subset [ i ] [ 0 ] = true ;
for ( int i = 1 ; i <= n ; i ++ ) subset [ 0 ] [ i ] = false ;
for ( int i = 1 ; i <= r ; i ++ ) { for ( int j = 1 ; j <= n ; j ++ ) {
if ( j < v [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] ; else { subset [ i ] [ j ] = subset [ i - 1 ] [ j ] || subset [ i - 1 ] [ j - v [ i - 1 ] ] ; } } }
if ( ( subset [ r ] [ n ] ) == 0 ) return false ; else return true ; }
bool checkweird ( int n ) { if ( checkAbundant ( n ) == true && checkSemiPerfect ( n ) == false ) return true ; else return false ; }
int main ( ) { int n = 70 ; if ( checkweird ( n ) ) cout << " Weird ▁ Number " ; else cout << " Not ▁ Weird ▁ Number " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSubArraySumRepeated ( int a [ ] , int n , int k ) { int max_so_far = INT_MIN , max_ending_here = 0 ; for ( int i = 0 ; i < n * k ; i ++ ) {
max_ending_here = max_ending_here + a [ i % n ] ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; if ( max_ending_here < 0 ) max_ending_here = 0 ; } return max_so_far ; }
int main ( ) { int a [ ] = { 10 , 20 , -30 , -1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int k = 3 ; cout << " Maximum ▁ contiguous ▁ sum ▁ is ▁ " << maxSubArraySumRepeated ( a , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int longOddEvenIncSeq ( int arr [ ] , int n ) {
int lioes [ n ] ;
int maxLen = 0 ;
for ( int i = 0 ; i < n ; i ++ ) lioes [ i ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && ( arr [ i ] + arr [ j ] ) % 2 != 0 && lioes [ i ] < lioes [ j ] + 1 ) lioes [ i ] = lioes [ j ] + 1 ;
for ( int i = 0 ; i < n ; i ++ ) if ( maxLen < lioes [ i ] ) maxLen = lioes [ i ] ;
return maxLen ; }
int main ( ) { int arr [ ] = { 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 } ; int n = sizeof ( arr ) / sizeof ( n ) ; cout << " Longest ▁ Increasing ▁ Odd ▁ Even ▁ " << " Subsequence : ▁ " << longOddEvenIncSeq ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isOperator ( char op ) { return ( op == ' + ' op == ' * ' ) ; }
void printMinAndMaxValueOfExp ( string exp ) { vector < int > num ; vector < char > opr ; string tmp = " " ;
for ( int i = 0 ; i < exp . length ( ) ; i ++ ) { if ( isOperator ( exp [ i ] ) ) { opr . push_back ( exp [ i ] ) ; num . push_back ( atoi ( tmp . c_str ( ) ) ) ; tmp = " " ; } else { tmp += exp [ i ] ; } }
num . push_back ( atoi ( tmp . c_str ( ) ) ) ; int len = num . size ( ) ; int minVal [ len ] [ len ] ; int maxVal [ len ] [ len ] ;
for ( int i = 0 ; i < len ; i ++ ) { for ( int j = 0 ; j < len ; j ++ ) { minVal [ i ] [ j ] = INT_MAX ; maxVal [ i ] [ j ] = 0 ;
if ( i == j ) minVal [ i ] [ j ] = maxVal [ i ] [ j ] = num [ i ] ; } }
for ( int L = 2 ; L <= len ; L ++ ) { for ( int i = 0 ; i < len - L + 1 ; i ++ ) { int j = i + L - 1 ; for ( int k = i ; k < j ; k ++ ) { int minTmp = 0 , maxTmp = 0 ;
if ( opr [ k ] == ' + ' ) { minTmp = minVal [ i ] [ k ] + minVal [ k + 1 ] [ j ] ; maxTmp = maxVal [ i ] [ k ] + maxVal [ k + 1 ] [ j ] ; }
else if ( opr [ k ] == ' * ' ) { minTmp = minVal [ i ] [ k ] * minVal [ k + 1 ] [ j ] ; maxTmp = maxVal [ i ] [ k ] * maxVal [ k + 1 ] [ j ] ; }
if ( minTmp < minVal [ i ] [ j ] ) minVal [ i ] [ j ] = minTmp ; if ( maxTmp > maxVal [ i ] [ j ] ) maxVal [ i ] [ j ] = maxTmp ; } } }
cout << " Minimum ▁ value ▁ : ▁ " << minVal [ 0 ] [ len - 1 ] << " , ▁ Maximum ▁ value ▁ : ▁ " << maxVal [ 0 ] [ len - 1 ] ; }
int main ( ) { string expression = "1 + 2*3 + 4*5" ; printMinAndMaxValueOfExp ( expression ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MatrixChainOrder ( int p [ ] , int i , int j ) { if ( i == j ) return 0 ; int k ; int min = INT_MAX ; int count ;
for ( k = i ; k < j ; k ++ ) { count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " << MatrixChainOrder ( arr , 1 , n - 1 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int dp [ 100 ] [ 100 ] ;
int matrixChainMemoised ( int * p , int i , int j ) { if ( i == j ) { return 0 ; } if ( dp [ i ] [ j ] != -1 ) { return dp [ i ] [ j ] ; } dp [ i ] [ j ] = INT_MAX ; for ( int k = i ; k < j ; k ++ ) { dp [ i ] [ j ] = min ( dp [ i ] [ j ] , matrixChainMemoised ( p , i , k ) + matrixChainMemoised ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ) ; } return dp [ i ] [ j ] ; } int MatrixChainOrder ( int * p , int n ) { int i = 1 , j = n - 1 ; return matrixChainMemoised ( p , i , j ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; memset ( dp , -1 , sizeof dp ) ; cout << " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " << MatrixChainOrder ( arr , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void flipBitsOfAandB ( int A , int B ) {
A = A ^ ( A & B ) ;
B = B ^ ( A & B ) ;
cout << A << " ▁ " << B ; }
int main ( ) { int A = 10 , B = 20 ; flipBitsOfAandB ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int TotalHammingDistance ( int n ) { int i = 1 , sum = 0 ; while ( n / i > 0 ) { sum = sum + n / i ; i = i * 2 ; } return sum ; }
int main ( ) { int N = 9 ; cout << TotalHammingDistance ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define int  long long int NEW_LINE #define m  1000000007
void solve ( long long n ) {
long long s = 0 ; for ( int l = 1 ; l <= n ; ) {
int r = n / floor ( n / l ) ; int x = ( ( ( r % m ) * ( ( r + 1 ) % m ) ) / 2 ) % m ; int y = ( ( ( l % m ) * ( ( l - 1 ) % m ) ) / 2 ) % m ; int p = ( ( n / l ) % m ) ;
s = ( s + ( ( ( x - y ) % m ) * p ) % m + m ) % m ; s %= m ; l = r + 1 ; }
cout << ( s + m ) % m ; }
signed main ( ) { long long n = 12 ; solve ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int min_time_to_cut ( int N ) { if ( N == 0 ) return 0 ;
return ceil ( log2 ( N ) ) ; }
int main ( ) { int N = 100 ; cout << min_time_to_cut ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findDistinctSums ( int n ) {
set < int > s ; for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = i ; j <= n ; j ++ ) {
s . insert ( i + j ) ; } }
return s . size ( ) ; }
int main ( ) { int N = 3 ; cout << findDistinctSums ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int printPattern ( int i , int j , int n ) {
if ( j >= n ) { return 0 ; } if ( i >= n ) { return 1 ; }
if ( j == i j == n - 1 - i ) {
if ( i == n - 1 - j ) { cout << " / " ; }
else { cout << " \ \" ; } }
else { cout << " * " ; }
if ( printPattern ( i , j + 1 , n ) == 1 ) { return 1 ; } cout << endl ;
return printPattern ( i + 1 , 0 , n ) ; }
int main ( ) { int N = 9 ;
printPattern ( 0 , 0 , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > zArray ( vector < int > arr ) { int n = arr . size ( ) ; vector < int > z ( n ) ; int r = 0 , l = 0 ;
for ( int k = 1 ; k < n ; k ++ ) {
if ( k > r ) { r = l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; }
else { int k1 = k - l ; if ( z [ k1 ] < r - k + 1 ) z [ k ] = z [ k1 ] ; else { l = k ; while ( r < n && arr [ r ] == arr [ r - l ] ) r ++ ; z [ k ] = r - l ; r -- ; } } } return z ; }
vector < int > mergeArray ( vector < int > A , vector < int > B ) { int n = A . size ( ) ; int m = B . size ( ) ; vector < int > z ;
vector < int > c ( n + m + 1 ) ;
for ( int i = 0 ; i < m ; i ++ ) c [ i ] = B [ i ] ;
c [ m ] = INT_MAX ;
for ( int i = 0 ; i < n ; i ++ ) c [ m + i + 1 ] = A [ i ] ;
z = zArray ( c ) ; return z ; }
void findZArray ( vector < int > A , vector < int > B , int n ) { int flag = 0 ; vector < int > z ; z = mergeArray ( A , B ) ;
for ( int i = 0 ; i < z . size ( ) ; i ++ ) { if ( z [ i ] == n ) { cout << ( i - n - 1 ) << " ▁ " ; flag = 1 ; } } if ( flag == 0 ) { cout << ( " Not ▁ Found " ) ; } }
int main ( ) { vector < int > A { 1 , 2 , 3 , 2 , 3 , 2 } ; vector < int > B { 2 , 3 } ; int n = B . size ( ) ; findZArray ( A , B , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getCount ( string a , string b ) {
if ( b . length ( ) % a . length ( ) != 0 ) return -1 ; int count = b . length ( ) / a . length ( ) ;
string str = " " ; for ( int i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str == b ) return count ; return -1 ; }
int main ( ) { string a = " geeks " ; string b = " geeksgeeks " ; cout << ( getCount ( a , b ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool check ( string S1 , string S2 ) {
int n1 = S1 . size ( ) ; int n2 = S2 . size ( ) ;
unordered_map < int , int > mp ;
for ( int i = 0 ; i < n1 ; i ++ ) { mp [ S1 [ i ] ] ++ ; }
for ( int i = 0 ; i < n2 ; i ++ ) {
if ( mp [ S2 [ i ] ] ) { mp [ S2 [ i ] ] -- ; }
else if ( mp [ S2 [ i ] - 1 ] && mp [ S2 [ i ] - 2 ] ) { mp [ S2 [ i ] - 1 ] -- ; mp [ S2 [ i ] - 2 ] -- ; } else { return false ; } } return true ; }
int main ( ) { string S1 = " abbat " ; string S2 = " cat " ;
if ( check ( S1 , S2 ) ) cout << " YES " ; else cout << " NO " ; }
#include <iostream> NEW_LINE using namespace std ;
int countPattern ( string str ) { int len = str . size ( ) ; bool oneSeen = 0 ;
for ( int i = 0 ; i < len ; i ++ ) {
if ( str [ i ] == '1' && oneSeen == 1 ) if ( str [ i - 1 ] == '0' ) count ++ ;
if ( str [ i ] == '1' && oneSeen == 0 ) { oneSeen = 1 ; continue ; }
if ( str [ i ] != '0' && str [ i ] != '1' ) oneSeen = 0 ; } return count ; }
int main ( ) { string str = "100001abc101" ; cout << countPattern ( str ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
string checkIfPossible ( int N , string arr [ ] , string T ) {
int freqS [ 256 ] = { 0 } ;
int freqT [ 256 ] = { 0 } ;
for ( char ch : T ) { freqT [ ch - ' a ' ] ++ ; }
for ( int i = 0 ; i < N ; i ++ ) {
for ( char ch : arr [ i ] ) { freqS [ ch - ' a ' ] ++ ; } } for ( int i = 0 ; i < 256 ; i ++ ) {
if ( freqT [ i ] == 0 && freqS [ i ] != 0 ) { return " No " ; }
else if ( freqS [ i ] == 0 && freqT [ i ] != 0 ) { return " No " ; }
else if ( freqT [ i ] != 0 && freqS [ i ] != ( freqT [ i ] * N ) ) { return " No " ; } }
return " Yes " ; }
int main ( ) { string arr [ ] = { " abc " , " abb " , " acc " } ; string T = " abc " ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << checkIfPossible ( N , arr , T ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int groupsOfOnes ( string S , int N ) {
int count = 0 ;
stack < int > st ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( S [ i ] == '1' ) st . push ( 1 ) ;
else {
if ( ! st . empty ( ) ) { count ++ ; while ( ! st . empty ( ) ) { st . pop ( ) ; } } } }
if ( ! st . empty ( ) ) count ++ ;
return count ; }
int main ( ) {
string S = "100110111" ; int N = S . length ( ) ;
cout << groupsOfOnes ( S , N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void generatePalindrome ( string S ) {
unordered_map < char , int > Hash ;
for ( auto ch : S ) { Hash [ ch ] ++ ; }
set < string > st ;
for ( char i = ' a ' ; i <= ' z ' ; i ++ ) {
if ( Hash [ i ] == 2 ) {
for ( char j = ' a ' ; j <= ' z ' ; j ++ ) {
string s = " " ; if ( Hash [ j ] && i != j ) { s += i ; s += j ; s += i ;
st . insert ( s ) ; } } }
if ( Hash [ i ] >= 3 ) {
for ( char j = ' a ' ; j <= ' z ' ; j ++ ) {
string s = " " ;
if ( Hash [ j ] ) { s += i ; s += j ; s += i ;
st . insert ( s ) ; } } } }
for ( auto ans : st ) { cout << ans << " STRNEWLINE " ; } }
int main ( ) { string S = " ddabdac " ; generatePalindrome ( S ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countOccurrences ( string S , string X , string Y ) {
int count = 0 ;
int N = S . length ( ) , A = X . length ( ) ; int B = Y . length ( ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( S . substr ( i , B ) == Y ) count ++ ;
if ( S . substr ( i , A ) == X ) cout << count << " ▁ " ; } }
int main ( ) { string S = " abcdefdefabc " ; string X = " abc " ; string Y = " def " ; countOccurrences ( S , X , Y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void DFA ( string str , int N ) {
if ( N <= 1 ) { cout << " No " ; return ; }
int count = 0 ;
if ( str [ 0 ] == ' C ' ) { count ++ ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( str [ i ] == ' A ' str [ i ] == ' B ' ) count ++ ; else break ; } } else {
cout << " No " ; return ; }
if ( count == N ) cout << " Yes " ; else cout << " No " ; }
int main ( ) { string str = " CAABBAAB " ; int N = str . size ( ) ; DFA ( str , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minMaxDigits ( string str , int N ) {
int arr [ N ] ; for ( int i = 0 ; i < N ; i ++ ) arr [ i ] = ( str [ i ] - '0' ) % 3 ;
int zero = 0 , one = 0 , two = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( arr [ i ] == 0 ) zero ++ ; if ( arr [ i ] == 1 ) one ++ ; if ( arr [ i ] == 2 ) two ++ ; }
int sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) { sum = ( sum + arr [ i ] ) % 3 ; }
if ( sum == 0 ) { cout << 0 << ' ▁ ' ; } if ( sum == 1 ) { if ( one && N > 1 ) cout << 1 << ' ▁ ' ; else if ( two > 1 && N > 2 ) cout << 2 << ' ▁ ' ; else cout << -1 << ' ▁ ' ; } if ( sum == 2 ) { if ( two && N > 1 ) cout << 1 << ' ▁ ' ; else if ( one > 1 && N > 2 ) cout << 2 << ' ▁ ' ; else cout << -1 << ' ▁ ' ; }
if ( zero > 0 ) cout << N - 1 << ' ▁ ' ; else if ( one > 0 && two > 0 ) cout << N - 2 << ' ▁ ' ; else if ( one > 2 two > 2 ) cout << N - 3 << ' ▁ ' ; else cout << -1 << ' ▁ ' ; }
int main ( ) { string str = "12345" ; int N = str . length ( ) ;
minMaxDigits ( str , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinimumChanges ( int N , int K , string S ) {
int ans = 0 ;
for ( int i = 0 ; i < ( K + 1 ) / 2 ; i ++ ) {
map < char , int > mp ;
for ( int j = i ; j < N ; j += K ) {
mp [ S [ j ] ] ++ ; }
for ( int j = N - i - 1 ; j >= 0 ; j -= K ) {
if ( K & 1 and i == K / 2 ) break ;
mp [ S [ j ] ] ++ ; }
int curr_max = INT_MIN ; for ( auto p : mp ) curr_max = max ( curr_max , p . second ) ;
if ( K & 1 and i == K / 2 ) ans += ( N / K - curr_max ) ;
else ans += ( N / K * 2 - curr_max ) ; }
return ans ; }
int main ( ) { string S = " aabbcbbcb " ; int N = S . length ( ) ; int K = 3 ;
cout << findMinimumChanges ( N , K , S ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string checkString ( string s , int K ) { int n = s . length ( ) ;
unordered_map < char , int > mp ; for ( int i = 0 ; i < n ; i ++ ) { mp [ s [ i ] ] = i ; } int cnt = 0 , f = 0 ;
unordered_set < int > st ; for ( int i = 0 ; i < n ; i ++ ) {
st . insert ( s [ i ] ) ;
if ( st . size ( ) > K ) { f = 1 ; break ; }
if ( mp [ s [ i ] ] == i ) st . erase ( s [ i ] ) ; } return ( f == 1 ? " Yes " : " No " ) ; }
int main ( ) { string s = " aabbcdca " ; int k = 2 ; cout << checkString ( s , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <set> NEW_LINE using namespace std ;
void distinct ( string S [ ] , int M , int n ) { int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
set < char > set1 ; for ( int j = 0 ; j < S [ i ] . length ( ) ; j ++ ) { if ( set1 . find ( S [ i ] [ j ] ) == set1 . end ( ) ) set1 . insert ( S [ i ] [ j ] ) ; } int c = set1 . size ( ) ;
if ( c <= M ) count += 1 ; } cout << ( count ) ; }
int main ( ) { string S [ ] = { " HERBIVORES " , " AEROPLANE " , " GEEKSFORGEEKS " } ; int M = 7 ; int n = sizeof ( S ) / sizeof ( S [ 0 ] ) ; distinct ( S , M , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string removeOddFrequencyCharacters ( string s ) {
unordered_map < char , int > m ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { m [ s [ i ] ] ++ ; }
string new_string = " " ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( m [ s [ i ] ] & 1 ) continue ;
new_string += s [ i ] ; }
return new_string ; }
int main ( ) { string str = " geeksforgeeks " ;
str = removeOddFrequencyCharacters ( str ) ; cout << str << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int productAtKthLevel ( string tree , int k , int & i , int level ) { if ( tree [ i ++ ] == ' ( ' ) {
if ( tree [ i ] == ' ) ' ) return 1 ; int product = 1 ;
if ( level == k ) product = tree [ i ] - '0' ;
int leftproduct = productAtKthLevel ( tree , k , ++ i , level + 1 ) ;
int rightproduct = productAtKthLevel ( tree , k , ++ i , level + 1 ) ;
++ i ; return product * leftproduct * rightproduct ; } }
int main ( ) { string tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) " " ( 9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; int i = 0 ; cout << productAtKthLevel ( tree , k , i , 0 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findMostOccurringChar ( vector < string > str ) {
int hash [ 26 ] = { 0 } ;
for ( int i = 0 ; i < str . size ( ) ; i ++ ) {
for ( int j = 0 ; j < str [ i ] . length ( ) ; j ++ ) {
hash [ str [ i ] [ j ] ] ++ ; } }
int max = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) { max = hash [ i ] > hash [ max ] ? i : max ; } cout << ( char ) ( max + 97 ) << endl ; }
int main ( ) {
vector < string > str ; str . push_back ( " animal " ) ; str . push_back ( " zebra " ) ; str . push_back ( " lion " ) ; str . push_back ( " giraffe " ) ; findMostOccurringChar ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPalindrome ( float num ) {
stringstream ss ; ss << num ; string s ; ss >> s ;
int low = 0 ; int high = s . size ( ) - 1 ; while ( low < high ) {
if ( s [ low ] != s [ high ] ) return false ;
low ++ ; high -- ; } return true ; }
int main ( ) { float n = 123.321f ; if ( isPalindrome ( n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 26 ;
int maxSubStr ( string str1 , int len1 , string str2 , int len2 ) {
if ( len1 > len2 ) return 0 ;
int freq1 [ MAX ] = { 0 } ; for ( int i = 0 ; i < len1 ; i ++ ) freq1 [ str1 [ i ] - ' a ' ] ++ ;
int freq2 [ MAX ] = { 0 } ; for ( int i = 0 ; i < len2 ; i ++ ) freq2 [ str2 [ i ] - ' a ' ] ++ ;
int minPoss = INT_MAX ; for ( int i = 0 ; i < MAX ; i ++ ) {
if ( freq1 [ i ] == 0 ) continue ;
if ( freq1 [ i ] > freq2 [ i ] ) return 0 ;
minPoss = min ( minPoss , freq2 [ i ] / freq1 [ i ] ) ; } return minPoss ; }
int main ( ) { string str1 = " geeks " , str2 = " gskefrgoekees " ; int len1 = str1 . length ( ) ; int len2 = str2 . length ( ) ; cout << maxSubStr ( str1 , len1 , str2 , len2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int cntWays ( string str , int n ) { int x = n + 1 ; int ways = x * x * ( x * x - 1 ) / 12 ; return ways ; }
int main ( ) { string str = " ab " ; int n = str . length ( ) ; cout << cntWays ( str , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unordered_set < string > uSet ;
int minCnt = INT_MAX ;
void findSubStr ( string str , int cnt , int start ) {
if ( start == str . length ( ) ) {
minCnt = min ( cnt , minCnt ) ; }
for ( int len = 1 ; len <= ( str . length ( ) - start ) ; len ++ ) {
string subStr = str . substr ( start , len ) ;
if ( uSet . find ( subStr ) != uSet . end ( ) ) {
findSubStr ( str , cnt + 1 , start + len ) ; } } }
void findMinSubStr ( string arr [ ] , int n , string str ) {
for ( int i = 0 ; i < n ; i ++ ) uSet . insert ( arr [ i ] ) ;
findSubStr ( str , 0 , 0 ) ; }
int main ( ) { string str = "123456" ; string arr [ ] = { "1" , "12345" , "2345" , "56" , "23" , "456" } ; int n = sizeof ( arr ) / sizeof ( string ) ; findMinSubStr ( arr , n , str ) ; cout << minCnt ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int countSubStr ( string s , int n ) { int c1 = 0 , c2 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( s . substr ( i , 5 ) == " geeks " ) c1 ++ ;
if ( s . substr ( i , 3 ) == " for " ) c2 = c2 + c1 ; } return c2 ; }
int main ( ) { string s = " geeksforgeeksisforgeeks " ; int n = s . size ( ) ; cout << countSubStr ( s , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) {
string String = " { [ ( ) ] } [ ] " ;
vector < char > lst1 = { ' { ' , ' ( ' , ' [ ' } ;
vector < char > lst2 = { ' } ' , ' ) ' , ' ] ' } ;
vector < char > lst ; int k ;
map < char , char > Dict ; Dict . insert ( pair < int , int > ( ' ) ' , ' ( ' ) ) ; Dict . insert ( pair < int , int > ( ' } ' , ' { ' ) ) ; Dict . insert ( pair < int , int > ( ' ] ' , ' [ ' ) ) ; int a = 0 , b = 0 , c = 0 ;
if ( count ( lst2 . begin ( ) , lst2 . end ( ) , String [ 0 ] ) ) { cout << 1 << endl ; } else {
for ( int i = 0 ; i < String . size ( ) ; i ++ ) { if ( count ( lst1 . begin ( ) , lst1 . end ( ) , String [ i ] ) ) { lst . push_back ( String [ i ] ) ; k = i + 2 ; } else {
if ( lst . size ( ) == 0 && ( count ( lst2 . begin ( ) , lst2 . end ( ) , String [ i ] ) ) ) { cout << ( i + 1 ) << endl ; c = 1 ; break ; } else {
if ( Dict [ String [ i ] ] == lst [ lst . size ( ) - 1 ] ) { lst . pop_back ( ) ; } else {
break ; cout << ( i + 1 ) << endl ; a = 1 ; } } } }
if ( lst . size ( ) == 0 && c == 0 ) { cout << 0 << endl ; b = 1 ; } if ( a == 0 && b == 0 && c == 0 ) { cout << k << endl ; } } return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define MAX  26 NEW_LINE using namespace std ;
string encryptStr ( string str , int n , int x ) {
x = x % MAX ;
int freq [ MAX ] = { 0 } ; for ( int i = 0 ; i < n ; i ++ ) { freq [ str [ i ] - ' a ' ] ++ ; } for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] % 2 == 0 ) { int pos = ( str [ i ] - ' a ' + x ) % MAX ; str [ i ] = ( char ) ( pos + ' a ' ) ; }
else { int pos = ( str [ i ] - ' a ' - x ) ; if ( pos < 0 ) { pos += MAX ; } str [ i ] = ( char ) ( pos + ' a ' ) ; } }
return str ; }
int main ( ) { string s = " abcda " ; int n = s . size ( ) ; int x = 3 ; cout << encryptStr ( s , n , x ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <time.h> NEW_LINE using namespace std ;
int isPossible ( string str ) {
unordered_map < char , int > freq ;
int max_freq = 0 ; for ( int j = 0 ; j < ( str . length ( ) ) ; j ++ ) { freq [ str [ j ] ] ++ ; if ( freq [ str [ j ] ] > max_freq ) max_freq = freq [ str [ j ] ] ; }
if ( max_freq <= ( str . length ( ) - max_freq + 1 ) ) return true ; return false ; }
int main ( ) { string str = " geeksforgeeks " ; if ( isPossible ( str ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printUncommon ( string str1 , string str2 ) { int a1 = 0 , a2 = 0 ; for ( int i = 0 ; i < str1 . length ( ) ; i ++ ) {
int ch = int ( str1 [ i ] ) - ' a ' ;
a1 = a1 | ( 1 << ch ) ; } for ( int i = 0 ; i < str2 . length ( ) ; i ++ ) {
int ch = int ( str2 [ i ] ) - ' a ' ;
a2 = a2 | ( 1 << ch ) ; }
int ans = a1 ^ a2 ; int i = 0 ; while ( i < 26 ) { if ( ans % 2 == 1 ) { cout << char ( ' a ' + i ) ; } ans = ans / 2 ; i ++ ; } }
int main ( ) { string str1 = " geeksforgeeks " ; string str2 = " geeksquiz " ; printUncommon ( str1 , str2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countMinReversals ( string expr ) { int len = expr . length ( ) ;
if ( len % 2 ) return -1 ;
int ans = 0 ; int i ;
int open = 0 ;
int close = 0 ; for ( i = 0 ; i < len ; i ++ ) {
if ( expr [ i ] == ' { ' ) open ++ ;
else { if ( ! open ) close ++ ; else open -- ; } } ans = ( close / 2 ) + ( open / 2 ) ;
close %= 2 ; open %= 2 ; if ( close ) ans += 2 ; return ans ; }
int main ( ) { string expr = " } } { { " ; cout << countMinReversals ( expr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int totalPairs ( string s1 , string s2 ) { int a1 = 0 , b1 = 0 ;
for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) { if ( int ( s1 [ i ] ) % 2 != 0 ) a1 ++ ; else b1 ++ ; } int a2 = 0 , b2 = 0 ;
for ( int i = 0 ; i < s2 . length ( ) ; i ++ ) { if ( int ( s2 [ i ] ) % 2 != 0 ) a2 ++ ; else b2 ++ ; }
return ( ( a1 * a2 ) + ( b1 * b2 ) ) ; }
int main ( ) { string s1 = " geeks " , s2 = " for " ; cout << totalPairs ( s1 , s2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int prefixOccurrences ( string str ) { char c = str [ 0 ] ; int countc = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str [ i ] == c ) countc ++ ; } return countc ; }
int main ( ) { string str = " abbcdabbcd " ; cout << prefixOccurrences ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minOperations ( string s , string t , int n ) { int ct0 = 0 , ct1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == t [ i ] ) continue ;
if ( s [ i ] == '0' ) ct0 ++ ;
else ct1 ++ ; } return max ( ct0 , ct1 ) ; }
int main ( ) { string s = "010" , t = "101" ; int n = s . length ( ) ; cout << minOperations ( s , t , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string decryptString ( string str , int n ) {
int i = 0 , jump = 1 ; string decryptedStr = " " ; while ( i < n ) { decryptedStr += str [ i ] ; i += jump ;
jump ++ ; } return decryptedStr ; }
int main ( ) { string str = " geeeeekkkksssss " ; int n = str . length ( ) ; cout << decryptString ( str , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
char bitToBeFlipped ( string s ) {
char last = s [ s . length ( ) - 1 ] ; char first = s [ 0 ] ;
if ( last == first ) { if ( last == '0' ) { return '1' ; } else { return '0' ; } }
else if ( last != first ) { return last ; } }
int main ( ) { string s = "1101011000" ; cout << bitToBeFlipped ( s ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void SieveOfEratosthenes ( bool prime [ ] , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
void sumProdOfPrimeFreq ( string s ) { bool prime [ s . length ( ) + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ; SieveOfEratosthenes ( prime , s . length ( ) + 1 ) ; int i , j ;
unordered_map < char , int > m ; for ( i = 0 ; i < s . length ( ) ; i ++ ) m [ s [ i ] ] ++ ; int sum = 0 , product = 1 ;
for ( auto it = m . begin ( ) ; it != m . end ( ) ; it ++ ) {
if ( prime [ it -> second ] ) { sum += it -> second ; product *= it -> second ; } } cout << " Sum ▁ = ▁ " << sum ; cout << " Product = " }
int main ( ) { string s = " geeksforgeeks " ; sumProdOfPrimeFreq ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool multipleOrFactor ( string s1 , string s2 ) {
map < char , int > m1 , m2 ; for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) m1 [ s1 [ i ] ] ++ ; for ( int i = 0 ; i < s2 . length ( ) ; i ++ ) m2 [ s2 [ i ] ] ++ ; map < char , int > :: iterator it ; for ( it = m1 . begin ( ) ; it != m1 . end ( ) ; it ++ ) {
if ( m2 . find ( ( * it ) . first ) == m2 . end ( ) ) continue ;
if ( m2 [ ( * it ) . first ] % ( * it ) . second == 0 || ( * it ) . second % m2 [ ( * it ) . first ] == 0 ) continue ;
else return false ; } }
int main ( ) { string s1 = " geeksforgeeks " ; string s2 = " geeks " ; multipleOrFactor ( s1 , s2 ) ? cout << " YES " : cout << " NO " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void solve ( string s ) {
unordered_map < char , int > m ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { m [ s [ i ] ] ++ ; }
string new_string = " " ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( m [ s [ i ] ] % 2 == 0 ) continue ;
new_string += s [ i ] ; }
cout << new_string << endl ; }
int main ( ) { string s = " aabbbddeeecc " ;
solve ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPalindrome ( string str ) { int i = 0 , j = str . size ( ) - 1 ;
while ( i < j )
if ( str [ i ++ ] != str [ j -- ] ) return false ;
return true ; }
string removePalinWords ( string str ) {
string final_str = " " , word = " " ;
str = str + " ▁ " ; int n = str . size ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str [ i ] != ' ▁ ' ) word = word + str [ i ] ; else {
if ( ! ( isPalindrome ( word ) ) ) final_str += word + " ▁ " ;
word = " " ; } }
return final_str ; }
int main ( ) { string str = " Text ▁ contains ▁ malayalam ▁ and ▁ level ▁ words " ; cout << removePalinWords ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findSubSequence ( string s , int num ) {
int res = 0 ;
int i = 0 ; while ( num ) {
if ( num & 1 ) res += s [ i ] - '0' ; i ++ ;
num = num >> 1 ; } return res ; }
int combinedSum ( string s ) {
int n = s . length ( ) ;
int c_sum = 0 ;
int range = ( 1 << n ) - 1 ;
for ( int i = 0 ; i <= range ; i ++ ) c_sum += findSubSequence ( s , i ) ;
return c_sum ; }
int main ( ) { string s = "123" ; cout << combinedSum ( s ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #define MAX_CHAR  26
void findSubsequence ( string str , int k ) {
int a [ MAX_CHAR ] = { 0 } ;
for ( int i = 0 ; i < str . size ( ) ; i ++ ) a [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < l ; i ++ ) if ( a [ str [ i ] - ' a ' ] >= k ) cout << str [ i ] ; }
int main ( ) { int k = 2 ; findSubsequence ( " geeksforgeeks " , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; string convert ( string str ) {
string w = " " , z = " " ;
transform ( str . begin ( ) , str . end ( ) , str . begin ( ) , :: toupper ) ; str += " ▁ " ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
char ch = str [ i ] ; if ( ch != ' ▁ ' ) { w = w + ch ; } else {
z = z + char ( tolower ( w [ 0 ] ) ) + w . substr ( 1 ) + " ▁ " ; w = " " ; } } return z ; }
int main ( ) { string str = " I ▁ got ▁ intern ▁ at ▁ geeksforgeeks " ; cout << convert ( str ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
string encryptString ( string s , int n , int k ) {
int cv [ n ] , cc [ n ] ; if ( isVowel ( s [ 0 ] ) ) cv [ 0 ] = 1 ; else cc [ 0 ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { cv [ i ] = cv [ i - 1 ] + isVowel ( s [ i ] ) ; cc [ i ] = cc [ i - 1 ] + ! isVowel ( s [ i ] ) ; } string ans = " " ; int prod = 0 ; prod = cc [ k - 1 ] * cv [ k - 1 ] ; ans += to_string ( prod ) ;
for ( int i = k ; i < s . length ( ) ; i ++ ) { prod = ( cc [ i ] - cc [ i - k ] ) * ( cv [ i ] - cv [ i - k ] ) ; ans += to_string ( prod ) ; } return ans ; }
int main ( ) { string s = " hello " ; int n = s . length ( ) ; int k = 2 ; cout << encryptString ( s , n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countOccurrences ( char * str , string word ) { char * p ;
vector < string > a ; p = strtok ( str , " ▁ " ) ; while ( p != NULL ) { a . push_back ( p ) ; p = strtok ( NULL , " ▁ " ) ; }
int c = 0 ; for ( int i = 0 ; i < a . size ( ) ; i ++ )
if ( word == a [ i ] ) c ++ ; return c ; }
int main ( ) { char str [ ] = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " ; string word = " portal " ; cout << countOccurrences ( str , word ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void permute ( string input ) { int n = input . length ( ) ;
int max = 1 << n ;
transform ( input . begin ( ) , input . end ( ) , input . begin ( ) , :: tolower ) ;
for ( int i = 0 ; i < max ; i ++ ) {
string combination = input ; for ( int j = 0 ; j < n ; j ++ ) if ( ( ( i >> j ) & 1 ) == 1 ) combination [ j ] = toupper ( input . at ( j ) ) ;
cout << combination << " ▁ " ; } }
int main ( ) { permute ( " ABC " ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void printString ( string str , char ch , int count ) { int occ = 0 , i ;
if ( count == 0 ) { cout << str ; return ; }
for ( i = 0 ; i < str . length ( ) ; i ++ ) {
if ( str [ i ] == ch ) occ ++ ;
if ( occ == count ) break ; }
if ( i < str . length ( ) - 1 ) cout << str . substr ( i + 1 , str . length ( ) - ( i + 1 ) ) ;
else cout < < " Empty ▁ string " ; }
int main ( ) { string str = " geeks ▁ for ▁ geeks " ; printString ( str , ' e ' , 2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isVowel ( char c ) { return ( c == ' a ' c == ' A ' c == ' e ' c == ' E ' c == ' i ' c == ' I ' c == ' o ' c == ' O ' c == ' u ' c == ' U ' ) ; }
string reverseVowel ( string str ) {
int i = 0 ; int j = str . length ( ) - 1 ; while ( i < j ) { if ( ! isVowel ( str [ i ] ) ) { i ++ ; continue ; } if ( ! isVowel ( str [ j ] ) ) { j -- ; continue ; }
swap ( str [ i ] , str [ j ] ) ; i ++ ; j -- ; } return str ; }
int main ( ) { string str = " hello ▁ world " ; cout << reverseVowel ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPalindrome ( const char * str ) {
int l = 0 ; int h = strlen ( str ) - 1 ;
while ( h > l ) if ( str [ l ++ ] != str [ h -- ] ) return false ; return true ; }
int minRemovals ( const char * str ) {
if ( str [ 0 ] == ' ' ) return 0 ;
if ( isPalindrome ( str ) ) return 1 ;
return 2 ; }
int main ( ) { cout << minRemovals ( "010010" ) << endl ; cout << minRemovals ( "0100101" ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int power ( int x , unsigned int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
int findModuloByM ( int X , int N , int M ) {
if ( N < 6 ) {
string temp ( N , ( char ) ( 48 + X ) ) ;
int res = stoi ( temp ) % M ; return res ; }
if ( N % 2 == 0 ) {
int half = findModuloByM ( X , N / 2 , M ) % M ;
int res = ( half * power ( 10 , N / 2 , M ) + half ) % M ; return res ; } else {
int half = findModuloByM ( X , N / 2 , M ) % M ;
int res = ( half * power ( 10 , N / 2 + 1 , M ) + half * 10 + X ) % M ; return res ; } }
int main ( ) { int X = 6 , N = 14 , M = 9 ;
cout << findModuloByM ( X , N , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class circle { public : double x ; double y ; double r ; } ;
bool check ( circle C [ ] ) {
double C1C2 = sqrt ( ( C [ 1 ] . x - C [ 0 ] . x ) * ( C [ 1 ] . x - C [ 0 ] . x ) + ( C [ 1 ] . y - C [ 0 ] . y ) * ( C [ 1 ] . y - C [ 0 ] . y ) ) ;
bool flag = 0 ;
if ( C1C2 < ( C [ 0 ] . r + C [ 1 ] . r ) ) {
if ( ( C [ 0 ] . x + C [ 1 ] . x ) == 2 * C [ 2 ] . x && ( C [ 0 ] . y + C [ 1 ] . y ) == 2 * C [ 2 ] . y ) {
flag = 1 ; } }
return flag ; }
bool IsFairTriplet ( circle c [ ] ) { bool f = false ;
f |= check ( c ) ; for ( int i = 0 ; i < 2 ; i ++ ) { swap ( c [ 0 ] , c [ 2 ] ) ;
f |= check ( c ) ; } return f ; }
int main ( ) { circle C [ 3 ] ; C [ 0 ] = { 0 , 0 , 8 } ; C [ 1 ] = { 0 , 10 , 6 } ; C [ 2 ] = { 0 , 5 , 5 } ; if ( IsFairTriplet ( C ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double eccHyperbola ( double A , double B ) {
double r = ( double ) B * B / A * A ;
r += 1 ;
return sqrt ( r ) ; }
int main ( ) { double A = 3.0 , B = 2.0 ; cout << eccHyperbola ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float calculateArea ( float A , float B , float C , float D ) {
float S = ( A + B + C + D ) / 2 ;
float area = sqrt ( ( S - A ) * ( S - B ) * ( S - C ) * ( S - D ) ) ;
return area ; }
int main ( ) { float A = 10 ; float B = 15 ; float C = 20 ; float D = 25 ; cout << calculateArea ( A , B , C , D ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void triangleArea ( int a , int b ) {
double ratio = ( double ) b / a ;
cout << ratio ; }
int main ( ) { int a = 1 , b = 2 ; triangleArea ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float distance ( int m , int n , int p , int q ) { return sqrt ( pow ( n - m , 2 ) + pow ( q - p , 2 ) * 1.0 ) ; }
void Excenters ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 ) {
float a = distance ( x2 , x3 , y2 , y3 ) ; float b = distance ( x3 , x1 , y3 , y1 ) ; float c = distance ( x1 , x2 , y1 , y2 ) ;
vector < pair < float , float > > excenter ( 4 ) ;
excenter [ 1 ] . first = ( - ( a * x1 ) + ( b * x2 ) + ( c * x3 ) ) / ( - a + b + c ) ; excenter [ 1 ] . second = ( - ( a * y1 ) + ( b * y2 ) + ( c * y3 ) ) / ( - a + b + c ) ;
excenter [ 2 ] . first = ( ( a * x1 ) - ( b * x2 ) + ( c * x3 ) ) / ( a - b + c ) ; excenter [ 2 ] . second = ( ( a * y1 ) - ( b * y2 ) + ( c * y3 ) ) / ( a - b + c ) ;
excenter [ 3 ] . first = ( ( a * x1 ) + ( b * x2 ) - ( c * x3 ) ) / ( a + b - c ) ; excenter [ 3 ] . second = ( ( a * y1 ) + ( b * y2 ) - ( c * y3 ) ) / ( a + b - c ) ;
for ( int i = 1 ; i <= 3 ; i ++ ) { cout << excenter [ i ] . first << " ▁ " << excenter [ i ] . second << endl ; } }
int main ( ) { float x1 , x2 , x3 , y1 , y2 , y3 ; x1 = 0 ; x2 = 3 ; x3 = 0 ; y1 = 0 ; y2 = 0 ; y3 = 4 ; Excenters ( x1 , y1 , x2 , y2 , x3 , y3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findHeight ( float p1 , float p2 , float b , float c ) { float a = max ( p1 , p2 ) - min ( p1 , p2 ) ;
float s = ( a + b + c ) / 2 ;
float area = sqrt ( s * ( s - a ) * ( s - b ) * ( s - c ) ) ;
float height = ( area * 2 ) / a ;
cout << " Height ▁ is : ▁ " << height ; }
int main ( ) {
float p1 = 25 , p2 = 10 ; float a = 14 , b = 13 ; findHeight ( p1 , p2 , a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Icositetragonal_num ( int n ) {
return ( 22 * n * n - 20 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << Icositetragonal_num ( n ) << endl ; n = 10 ; cout << Icositetragonal_num ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double area_of_circle ( int m , int n ) {
int square_of_radius = ( m * n ) / 4 ; double area = ( 3.141 * square_of_radius ) ; return area ; }
int main ( ) { int n = 10 ; int m = 30 ; cout << ( area_of_circle ( m , n ) ) ; }
#include <iostream> NEW_LINE using namespace std ;
double area ( int R ) {
double base = 1.732 * R ; double height = ( 1.5 ) * R ;
double area = 0.5 * base * height ; return area ; }
int main ( ) { int R = 7 ; cout << ( area ( R ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float circlearea ( float R ) {
if ( R < 0 ) return -1 ;
float a = 3.14 * R * R / 4 ; return a ; }
int main ( ) { float R = 2 ; cout << circlearea ( R ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countPairs ( int * P , int * Q , int N , int M ) {
int A [ 2 ] = { 0 } , B [ 2 ] = { 0 } ;
for ( int i = 0 ; i < N ; i ++ ) A [ P [ i ] % 2 ] ++ ;
for ( int i = 0 ; i < M ; i ++ ) B [ Q [ i ] % 2 ] ++ ;
return ( A [ 0 ] * B [ 0 ] + A [ 1 ] * B [ 1 ] ) ; }
int main ( ) { int P [ ] = { 1 , 3 , 2 } , Q [ ] = { 3 , 0 } ; int N = sizeof ( P ) / sizeof ( P [ 0 ] ) ; int M = sizeof ( Q ) / sizeof ( Q [ 0 ] ) ; cout << countPairs ( P , Q , N , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countIntersections ( int n ) { return n * ( n - 1 ) / 2 ; }
int main ( ) { int n = 3 ; cout << countIntersections ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define PI  3.14159 NEW_LINE using namespace std ;
double areaOfTriangle ( float d ) {
float c = 1.618 * d ; float s = ( d + c + c ) / 2 ;
double area = sqrt ( s * ( s - c ) * ( s - c ) * ( s - d ) ) ;
return 5 * area ; }
double areaOfRegPentagon ( float d ) {
double cal = 4 * tan ( PI / 5 ) ; double area = ( 5 * d * d ) / cal ;
return area ; }
double areaOfPentagram ( float d ) {
return areaOfRegPentagon ( d ) + areaOfTriangle ( d ) ; }
int main ( ) { float d = 5 ; cout << areaOfPentagram ( d ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void anglequichord ( int z ) { cout << " The ▁ angle ▁ is ▁ " << z << " ▁ degrees " << endl ; }
int main ( ) { int z = 48 ; anglequichord ( z ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int convertToASCII ( int N ) { string num = to_string ( N ) ; for ( char ch : num ) { cout << ch << " ▁ ( " << ( int ) ch << " ) STRNEWLINE " ; } }
int main ( ) { int N = 36 ; convertToASCII ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void productExceptSelf ( int arr [ ] , int N ) {
int product = 1 ;
int z = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] ) product *= arr [ i ] ;
z += ( arr [ i ] == 0 ) ; }
int a = abs ( product ) , b ; for ( int i = 0 ; i < N ; i ++ ) {
if ( z == 1 ) {
if ( arr [ i ] ) arr [ i ] = 0 ;
else arr [ i ] = product ; continue ; }
else if ( z > 1 ) {
arr [ i ] = 0 ; continue ; }
int b = abs ( arr [ i ] ) ;
int curr = round ( exp ( log ( a ) - log ( b ) ) ) ;
if ( arr [ i ] < 0 && product < 0 ) arr [ i ] = curr ;
else if ( arr [ i ] > 0 && product > 0 ) arr [ i ] = curr ;
else arr [ i ] = -1 * curr ; }
for ( int i = 0 ; i < N ; i ++ ) { cout << arr [ i ] << " ▁ " ; } }
int main ( ) { int arr [ ] = { 10 , 3 , 5 , 6 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
productExceptSelf ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int singleDigitSubarrayCount ( int arr [ ] , int N ) {
int res = 0 ;
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { if ( arr [ i ] <= 9 ) {
count ++ ;
res += count ; } else {
count = 0 ; } } cout << res ; }
int main ( ) {
int arr [ ] = { 0 , 1 , 14 , 2 , 5 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; singleDigitSubarrayCount ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPossible ( int N ) { return ( ( N & ( N - 1 ) ) && N ) ; }
void countElements ( int N ) {
int count = 0 ; for ( int i = 1 ; i <= N ; i ++ ) { if ( isPossible ( i ) ) count ++ ; } cout << count ; }
int main ( ) { int N = 15 ; countElements ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countElements ( int N ) { int Cur_Ele = 1 ; int Count = 0 ;
while ( Cur_Ele <= N ) {
Count ++ ;
Cur_Ele = Cur_Ele * 2 ; } cout << N - Count ; }
int main ( ) { int N = 15 ; countElements ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maxAdjacent ( int * arr , int N ) { vector < int > res ; int arr_max = INT_MIN ;
for ( int i = 1 ; i < N ; i ++ ) { arr_max = max ( arr_max , abs ( arr [ i - 1 ] - arr [ i ] ) ) ; } for ( int i = 1 ; i < N - 1 ; i ++ ) { int curr_max = abs ( arr [ i - 1 ] - arr [ i + 1 ] ) ;
int ans = max ( curr_max , arr_max ) ;
res . push_back ( ans ) ; }
for ( auto x : res ) cout << x << " ▁ " ; cout << endl ; }
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 7 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; maxAdjacent ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minimumIncrement ( int arr [ ] , int N ) {
if ( N % 2 != 0 ) { cout << " - 1" ; exit ( 0 ) ; }
int cntEven = 0 ;
int cntOdd = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 == 0 ) {
cntEven += 1 ; } }
cntOdd = N - cntEven ;
return abs ( cntEven - cntOdd ) / 2 ; }
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 9 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << minimumIncrement ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void cntWaysConsArray ( int A [ ] , int N ) {
int total = 1 ;
int oddArray = 1 ;
for ( int i = 0 ; i < N ; i ++ ) {
total = total * 3 ;
if ( A [ i ] % 2 == 0 ) {
oddArray *= 2 ; } }
cout << total - oddArray << " STRNEWLINE " ; }
int main ( ) { int A [ ] = { 2 , 4 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; cntWaysConsArray ( A , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countNumberHavingKthBitSet ( int N , int K ) {
int numbers_rightmost_setbit_K ; for ( int i = 1 ; i <= K ; i ++ ) {
int numbers_rightmost_bit_i = ( N + 1 ) / 2 ;
N -= numbers_rightmost_bit_i ;
if ( i == K ) { numbers_rightmost_setbit_K = numbers_rightmost_bit_i ; } } cout << numbers_rightmost_setbit_K ; }
int main ( ) { int N = 15 ; int K = 2 ; countNumberHavingKthBitSet ( N , K ) ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
int countSetBits ( int N ) { int count = 0 ;
while ( N ) { N = N & ( N - 1 ) ; count ++ ; }
return count ; }
int main ( ) { int N = 4 ; int bits = countSetBits ( N ) ;
cout << " Odd ▁ " << " : ▁ " << pow ( 2 , bits ) << " STRNEWLINE " ;
cout << " Even ▁ " << " : ▁ " << N + 1 - pow ( 2 , bits ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minMoves ( int arr [ ] , int N ) {
int odd_element_cnt = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] % 2 != 0 ) { odd_element_cnt ++ ; } }
int moves = ( odd_element_cnt ) / 2 ;
if ( odd_element_cnt % 2 != 0 ) moves += 2 ;
cout << moves ; }
int main ( ) { int arr [ ] = { 5 , 6 , 3 , 7 , 20 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
minMoves ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minimumSubsetDifference ( int N ) {
int blockOfSize8 = N / 8 ;
string str = " ABBABAAB " ;
int subsetDifference = 0 ;
string partition = " " ; while ( blockOfSize8 -- ) { partition += str ; }
vector < int > A , B ; for ( int i = 0 ; i < N ; i ++ ) {
if ( partition [ i ] == ' A ' ) { A . push_back ( ( i + 1 ) * ( i + 1 ) ) ; }
else { B . push_back ( ( i + 1 ) * ( i + 1 ) ) ; } }
cout << subsetDifference << " STRNEWLINE " ;
for ( int i = 0 ; i < A . size ( ) ; i ++ ) cout << A [ i ] << " ▁ " ; cout << " STRNEWLINE " ;
for ( int i = 0 ; i < B . size ( ) ; i ++ ) cout << B [ i ] << " ▁ " ; }
int main ( ) { int N = 8 ;
minimumSubsetDifference ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findTheGreatestX ( int P , int Q ) {
map < int , int > divisiors ; for ( int i = 2 ; i * i <= Q ; i ++ ) { while ( Q % i == 0 and Q > 1 ) { Q /= i ;
divisiors [ i ] ++ ; } }
if ( Q > 1 ) divisiors [ Q ] ++ ;
int ans = 0 ;
for ( auto i : divisiors ) { int frequency = i . second ; int temp = P ;
int cur = 0 ; while ( temp % i . first == 0 ) { temp /= i . first ;
cur ++ ; }
if ( cur < frequency ) { ans = P ; break ; } temp = P ;
for ( int j = cur ; j >= frequency ; j -- ) { temp /= i . first ; }
ans = max ( temp , ans ) ; }
cout << ans ; }
int main ( ) {
int P = 10 , Q = 4 ;
findTheGreatestX ( P , Q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string checkRearrangements ( vector < vector < int > > mat , int N , int M ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 1 ; j < M ; j ++ ) { if ( mat [ i ] [ 0 ] != mat [ i ] [ j ] ) { return " Yes " ; } } } return " No " ; }
string nonZeroXor ( vector < vector < int > > mat , int N , int M ) { int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { res = res ^ mat [ i ] [ 0 ] ; }
if ( res != 0 ) return " Yes " ;
else return checkRearrangements ( mat , N , M ) ; }
int main ( ) {
vector < vector < int > > mat = { { 1 , 1 , 2 } , { 2 , 2 , 2 } , { 3 , 3 , 3 } } ; int N = mat . size ( ) ; int M = mat [ 0 ] . size ( ) ;
cout << nonZeroXor ( mat , N , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define size_int  32
int functionMax ( int arr [ ] , int n ) {
vector < int > setBit [ 32 ] ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < size_int ; j ++ ) {
if ( arr [ i ] & ( 1 << j ) )
setBit [ j ] . push_back ( i ) ; } }
for ( int i = size_int ; i >= 0 ; i -- ) { if ( setBit [ i ] . size ( ) == 1 ) {
swap ( arr [ 0 ] , arr [ setBit [ i ] [ 0 ] ] ) ; break ; } }
int maxAnd = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { maxAnd = maxAnd & ( ~ arr [ i ] ) ; }
return maxAnd ; }
int main ( ) { int arr [ ] = { 1 , 2 , 4 , 8 , 16 } ; int n = sizeof arr / sizeof arr [ 0 ] ;
cout << functionMax ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int nCr ( int n , int r ) {
int res = 1 ;
if ( r > n - r ) r = n - r ;
for ( int i = 0 ; i < r ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
int solve ( int n , int m , int k ) {
int sum = 0 ;
for ( int i = 0 ; i <= k ; i ++ ) sum += nCr ( n , i ) * nCr ( m , k - i ) ; return sum ; }
int main ( ) { int n = 3 , m = 2 , k = 2 ; cout << solve ( n , m , k ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int powerOptimised ( int a , int n ) {
int ans = 1 ; while ( n > 0 ) { int last_bit = ( n & 1 ) ;
if ( last_bit ) { ans = ans * a ; } a = a * a ;
n = n >> 1 ; } return ans ; }
int main ( ) { int a = 3 , n = 5 ; cout << powerOptimised ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMaximumGcd ( int n ) {
int max_gcd = 1 ;
for ( int i = 1 ; i * i <= n ; i ++ ) {
if ( n % i == 0 ) {
if ( i > max_gcd ) max_gcd = i ; if ( ( n / i != i ) && ( n / i != n ) && ( ( n / i ) > max_gcd ) ) max_gcd = n / i ; } }
return max_gcd ; }
int main ( ) {
int N = 10 ;
cout << findMaximumGcd ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define x  2000021 NEW_LINE using namespace std ;
long long int v [ x ] ;
void sieve ( ) { v [ 1 ] = 1 ;
for ( long long int i = 2 ; i < x ; i ++ ) v [ i ] = i ;
for ( long long int i = 4 ; i < x ; i += 2 ) v [ i ] = 2 ; for ( long long int i = 3 ; i * i < x ; i ++ ) {
if ( v [ i ] == i ) {
for ( long long int j = i * i ; j < x ; j += i ) {
if ( v [ j ] == j ) { v [ j ] = i ; } } } } }
long long int prime_factors ( long long n ) { set < long long int > s ; while ( n != 1 ) { s . insert ( v [ n ] ) ; n = n / v [ n ] ; } return s . size ( ) ; }
void distinctPrimes ( long long int m , long long int k ) {
vector < long long int > result ; for ( long long int i = 14 ; i < m + k ; i ++ ) {
long long count = prime_factors ( i ) ;
if ( count == k ) { result . push_back ( i ) ; } } long long int p = result . size ( ) ; for ( long long int index = 0 ; index < p - 1 ; index ++ ) { long long element = result [ index ] ; long long count = 1 , z = index ;
while ( z < p - 1 && count <= k && result [ z ] + 1 == result [ z + 1 ] ) {
count ++ ; z ++ ; }
if ( count >= k ) cout << element << ' ▁ ' ; } }
int main ( ) {
sieve ( ) ;
long long int N = 1000 , K = 3 ;
distinctPrimes ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void print_product ( int a , int b , int c , int d ) {
int prod1 = a * c ; int prod2 = b * d ; int prod3 = ( a + b ) * ( c + d ) ;
int real = prod1 - prod2 ;
int imag = prod3 - ( prod1 + prod2 ) ;
cout << real << " ▁ + ▁ " << imag << " i " ; }
int main ( ) { int a , b , c , d ;
a = 2 ; b = 3 ; c = 4 ; d = 5 ;
print_product ( a , b , c , d ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isInsolite ( int n ) { int N = n ;
int sum = 0 ;
int product = 1 ; while ( n != 0 ) {
int r = n % 10 ; sum = sum + r * r ; product = product * r * r ; n = n / 10 ; } return ( N % sum == 0 ) && ( N % product == 0 ) ; }
int main ( ) { int N = 111 ;
if ( isInsolite ( N ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int sigma ( int n ) { if ( n == 1 ) return 1 ;
int result = 0 ;
for ( int i = 2 ; i <= sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) {
if ( i == ( n / i ) ) result += i ; else result += ( i + n / i ) ; } }
return ( result + n + 1 ) ; }
bool isSuperabundant ( int N ) {
for ( float i = 1 ; i < N ; i ++ ) { float x = sigma ( i ) / i ; float y = sigma ( N ) / ( N * 1.0 ) ; if ( x > y ) return false ; } return true ; }
int main ( ) { int N = 4 ; isSuperabundant ( N ) ? cout << " Yes " : cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int isDNum ( int n ) {
if ( n < 4 ) return false ; int numerator , hcf ;
for ( int k = 2 ; k <= n ; k ++ ) { numerator = pow ( k , n - 2 ) - k ; hcf = __gcd ( n , k ) ; }
if ( hcf == 1 && ( numerator % n ) != 0 ) return false ; return true ; }
int main ( ) { int n = 15 ; int a = isDNum ( n ) ; if ( a ) cout << " Yes " ; else cout << " No " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Sum ( int N ) { int SumOfPrimeDivisors [ N + 1 ] = { 0 } ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( ! SumOfPrimeDivisors [ i ] ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
bool RuthAaronNumber ( int n ) { if ( Sum ( n ) == Sum ( n + 1 ) ) return true ; else return false ; }
int main ( ) { int N = 714 ; if ( RuthAaronNumber ( N ) ) { cout << " Yes " ; } else { cout << " No " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxAdjacentDifference ( int N , int K ) {
if ( N == 1 ) { return 0 ; }
if ( N == 2 ) { return K ; }
return 2 * K ; }
int main ( ) { int N = 6 ; int K = 11 ; cout << maxAdjacentDifference ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int mod = 1000000007 ;
int linearSum ( int n ) { return ( n * ( n + 1 ) / 2 ) % mod ; }
int rangeSum ( int b , int a ) { return ( linearSum ( b ) - linearSum ( a ) ) % mod ; }
int totalSum ( int n ) {
int result = 0 ; int i = 1 ;
while ( true ) {
result += rangeSum ( n / i , n / ( i + 1 ) ) * ( i % mod ) % mod ; result %= mod ; if ( i == n ) break ; i = n / ( n / ( i + 1 ) ) ; } return result ; }
int main ( ) { int N = 4 ; cout << totalSum ( N ) << endl ; N = 12 ; cout << totalSum ( N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isDouble ( int num ) { string s = to_string ( num ) ; int l = s . length ( ) ;
if ( s [ 0 ] == s [ 1 ] ) return false ;
if ( l % 2 == 1 ) { s = s + s [ 1 ] ; l ++ ; }
string s1 = s . substr ( 0 , l / 2 ) ;
string s2 = s . substr ( l / 2 ) ;
return s1 == s2 ; }
bool isNontrivialUndulant ( int N ) { return N > 100 && isDouble ( N ) ; }
int main ( ) { int n = 121 ; if ( isNontrivialUndulant ( n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int MegagonNum ( int n ) { return ( 999998 * n * n - 999996 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << MegagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define mod  1000000007
int productPairs ( int arr [ ] , int n ) {
int product = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) {
product *= ( arr [ i ] % mod * arr [ j ] % mod ) % mod ; product = product % mod ; } }
return product % mod ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << productPairs ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define mod  1000000007 NEW_LINE #define ll  long long int
int power ( int x , unsigned int y ) { int p = 1000000007 ;
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ; y = y >> 1 ; x = ( x * x ) % p ; }
return res ; }
ll productPairs ( ll arr [ ] , ll n ) {
ll product = 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
product = ( product % mod * ( int ) power ( arr [ i ] , ( 2 * n ) ) % mod ) % mod ; } return product % mod ; }
int main ( ) { ll arr [ ] = { 1 , 2 , 3 } ; ll n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << productPairs ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void constructArray ( int N ) { int arr [ N ] ;
for ( int i = 1 ; i <= N ; i ++ ) { arr [ i - 1 ] = i ; }
for ( int i = 0 ; i < N ; i ++ ) { cout << arr [ i ] << " , ▁ " ; } }
int main ( ) { int N = 6 ; constructArray ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) { if ( n <= 1 ) return false ; for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
int countSubsequences ( int arr [ ] , int n ) {
int totalSubsequence = pow ( 2 , n ) - 1 ; int countPrime = 0 , countOnes = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) countOnes ++ ; else if ( isPrime ( arr [ i ] ) ) countPrime ++ ; } int compositeSubsequence ;
int onesSequence = pow ( 2 , countOnes ) - 1 ;
compositeSubsequence = totalSubsequence - countPrime - onesSequence - onesSequence * countPrime ; return compositeSubsequence ; }
int main ( ) { int arr [ ] = { 2 , 1 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << countSubsequences ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void checksum ( int n , int k ) {
float first_term = ( ( 2 * n ) / k + ( 1 - k ) ) / 2.0 ;
if ( first_term - int ( first_term ) == 0 ) {
for ( int i = first_term ; i <= first_term + k - 1 ; i ++ ) { cout << i << " ▁ " ; } } else cout << " - 1" ; }
int main ( ) { int n = 33 , k = 6 ; checksum ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void sumEvenNumbers ( int N , int K ) { int check = N - 2 * ( K - 1 ) ;
if ( check > 0 && check % 2 == 0 ) { for ( int i = 0 ; i < K - 1 ; i ++ ) { cout << "2 ▁ " ; } cout << check ; } else { cout << " - 1" ; } }
int main ( ) { int N = 8 ; int K = 2 ; sumEvenNumbers ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > calculateWays ( int N ) { int x = 0 ; vector < int > v ;
for ( int i = 0 ; i < N ; i ++ ) v . push_back ( 0 ) ;
for ( int i = 0 ; i <= N / 2 ; i ++ ) {
if ( N % 2 == 0 && i == N / 2 ) break ;
x = N * ( i + 1 ) - ( i + 1 ) * i ;
v [ i ] = x ; v [ N - i - 1 ] = x ; } return v ; }
void printArray ( vector < int > v ) { for ( int i = 0 ; i < v . size ( ) ; i ++ ) cout << v [ i ] << " ▁ " ; }
int main ( ) { vector < int > v ; v = calculateWays ( 4 ) ; printArray ( v ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAXN  10000000
int sumOfDigits ( int n ) {
int sum = 0 ; while ( n > 0 ) {
sum += n % 10 ;
n /= 10 ; } return sum ; }
int smallestNum ( int X , int Y ) {
int res = -1 ;
for ( int i = X ; i < MAXN ; i ++ ) {
int sum_of_digit = sumOfDigits ( i ) ;
if ( sum_of_digit % Y == 0 ) { res = i ; break ; } } return res ; }
int main ( ) { int X = 5923 , Y = 13 ; cout << smallestNum ( X , Y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countValues ( int N ) { vector < int > div ;
for ( int i = 2 ; i * i <= N ; i ++ ) {
if ( N % i == 0 ) { div . push_back ( i ) ;
if ( N != i * i ) { div . push_back ( N / i ) ; } } } int answer = 0 ;
for ( int i = 1 ; i * i <= N - 1 ; i ++ ) {
if ( ( N - 1 ) % i == 0 ) { if ( i * i == N - 1 ) answer ++ ; else answer += 2 ; } }
for ( auto d : div ) { int K = N ; while ( K % d == 0 ) K /= d ; if ( ( K - 1 ) % d == 0 ) answer ++ ; } return answer ; }
int main ( ) { int N = 6 ; cout << countValues ( N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #define ll  long long int
void findMaxPrimeDivisor ( int n ) { int max_possible_prime = 0 ;
while ( n % 2 == 0 ) { max_possible_prime ++ ; n = n / 2 ; }
for ( int i = 3 ; i * i <= n ; i = i + 2 ) { while ( n % i == 0 ) { max_possible_prime ++ ; n = n / i ; } }
if ( n > 2 ) { max_possible_prime ++ ; } cout << max_possible_prime << " STRNEWLINE " ; }
int main ( ) { int n = 4 ;
findMaxPrimeDivisor ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int CountWays ( int n ) { int ans = ( n - 1 ) / 2 ; return ans ; }
int main ( ) { int N = 8 ; cout << CountWays ( N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void Solve ( int arr [ ] , int size , int n ) { vector < int > v ( n + 1 ) ;
for ( int i = 0 ; i < size ; i ++ ) v [ arr [ i ] ] ++ ;
int max1 = ( max_element ( v . begin ( ) , v . end ( ) ) - v . begin ( ) ) ;
int diff1 = n + 1 - count ( v . begin ( ) , v . end ( ) , 0 ) ;
int max_size = max ( min ( v [ max1 ] - 1 , diff1 ) , min ( v [ max1 ] , diff1 - 1 ) ) ; cout << " Maximum ▁ size ▁ is ▁ : " << max_size << " STRNEWLINE " ;
cout << " The ▁ First ▁ Array ▁ Is ▁ : ▁ STRNEWLINE " ; for ( int i = 0 ; i < max_size ; i ++ ) { cout << max1 << " ▁ " ; v [ max1 ] -= 1 ; } cout << " STRNEWLINE " ;
cout << " The ▁ Second ▁ Array ▁ Is ▁ : ▁ STRNEWLINE " ; for ( int i = 0 ; i < ( n + 1 ) ; i ++ ) { if ( v [ i ] > 0 ) { cout << i << " ▁ " ; max_size -- ; } if ( max_size < 1 ) break ; } cout << " STRNEWLINE " ; }
int main ( ) {
int n = 7 ;
int arr [ ] = { 1 , 2 , 1 , 5 , 1 , 6 , 7 , 2 } ;
int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; Solve ( arr , size , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int power ( int x , int y , int p ) {
int res = 1 ;
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
int modInverse ( int n , int p ) { return power ( n , p - 2 , p ) ; }
int nCrModPFermat ( int n , int r , int p ) {
if ( r == 0 ) return 1 ; if ( n < r ) return 0 ;
int fac [ n + 1 ] ; fac [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fac [ i ] = fac [ i - 1 ] * i % p ; return ( fac [ n ] * modInverse ( fac [ r ] , p ) % p * modInverse ( fac [ n - r ] , p ) % p ) % p ; }
int SumOfXor ( int a [ ] , int n ) { int mod = 10037 ; int answer = 0 ;
for ( int k = 0 ; k < 32 ; k ++ ) {
int x = 0 , y = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] & ( 1 << k ) ) x ++ ; else y ++ ; }
answer += ( ( 1 << k ) % mod * ( nCrModPFermat ( x , 3 , mod ) + x * nCrModPFermat ( y , 2 , mod ) ) % mod ) % mod ; } return answer ; }
int main ( ) { int n = 5 ; int A [ n ] = { 3 , 5 , 2 , 18 , 7 } ; cout << SumOfXor ( A , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; float round ( float var , int digit ) { float value = ( int ) ( var * pow ( 10 , digit ) + .5 ) ; return ( float ) value / pow ( 10 , digit ) ; }
int probability ( int N ) {
int a = 2 ; int b = 3 ;
if ( N == 1 ) { return a ; } else if ( N == 2 ) { return b ; } else {
for ( int i = 3 ; i <= N ; i ++ ) { int c = a + b ; a = b ; b = c ; } return b ; } }
float operations ( int N ) {
int x = probability ( N ) ;
int y = pow ( 2 , N ) ; return round ( ( float ) x / ( float ) y , 2 ) ; }
int main ( ) { int N = 10 ; cout << ( operations ( N ) ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPerfectCube ( int x ) { long double cr = round ( cbrt ( x ) ) ; return ( cr * cr * cr == x ) ; }
void checkCube ( int a , int b ) {
string s1 = to_string ( a ) ; string s2 = to_string ( b ) ;
int c = stoi ( s1 + s2 ) ;
if ( isPerfectCube ( c ) ) { cout << " Yes " ; } else { cout << " No " ; } }
int main ( ) { int a = 6 ; int b = 4 ; checkCube ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int largest_sum ( int arr [ ] , int n ) {
int maximum = -1 ;
map < int , int > m ;
for ( int i = 0 ; i < n ; i ++ ) { m [ arr [ i ] ] ++ ; }
for ( auto j : m ) {
if ( j . second > 1 ) {
m [ 2 * j . first ] = m [ 2 * j . first ] + j . second / 2 ;
if ( 2 * j . first > maximum ) maximum = 2 * j . first ; } }
return maximum ; }
int main ( ) { int arr [ ] = { 1 , 1 , 2 , 4 , 7 , 8 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << largest_sum ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void canBeReduced ( int x , int y ) { int maxi = max ( x , y ) ; int mini = min ( x , y ) ;
if ( ( ( x + y ) % 3 ) == 0 && maxi <= 2 * mini ) cout << " YES " << endl ; else cout << " NO " << endl ; }
int main ( ) { int x = 6 , y = 9 ;
canBeReduced ( x , y ) ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE using namespace std ;
void isPrime ( int N ) { bool isPrime = true ;
int arr [ 8 ] = { 7 , 11 , 13 , 17 , 19 , 23 , 29 , 31 } ;
if ( N < 2 ) { isPrime = false ; }
if ( N % 2 == 0 N % 3 == 0 N % 5 == 0 ) { isPrime = false ; }
for ( int i = 0 ; i < sqrt ( N ) ; i += 30 ) {
for ( int c : arr ) {
if ( c > sqrt ( N ) ) { break ; }
else { if ( N % ( c + i ) == 0 ) { isPrime = false ; break ; } }
if ( ! isPrime ) break ; } } if ( isPrime ) cout << " Prime ▁ Number " ; else cout << " Not ▁ a ▁ Prime ▁ Number " ; }
int main ( ) { int N = 121 ;
isPrime ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printPairs ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { cout << " ( " << arr [ i ] << " , ▁ " << arr [ j ] << " ) " << " , ▁ " ; } } }
int main ( ) { int arr [ ] = { 1 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printPairs ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int nearest ( int n ) {
int prevCube = cbrt ( n ) ; int nextCube = prevCube + 1 ; prevCube = prevCube * prevCube * prevCube ; nextCube = nextCube * nextCube * nextCube ;
int ans = ( n - prevCube ) < ( nextCube - n ) ? ( prevCube - n ) : ( nextCube - n ) ;
return ans ; }
int main ( ) { int n = 25 ; cout << nearest ( n ) << endl ; n = 27 ; cout << nearest ( n ) << endl ; n = 40 ; cout << nearest ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void circle ( int x1 , int y1 , int x2 , int y2 , int r1 , int r2 ) { int distSq = sqrt ( ( ( x1 - x2 ) * ( x1 - x2 ) ) + ( ( y1 - y2 ) * ( y1 - y2 ) ) ) ; if ( distSq + r2 == r1 ) cout << " The ▁ smaller ▁ circle ▁ lies ▁ completely " << " ▁ inside ▁ the ▁ bigger ▁ circle ▁ with ▁ " << " touching ▁ each ▁ other ▁ " << " at ▁ a ▁ point ▁ of ▁ circumference . ▁ " << endl ; else if ( distSq + r2 < r1 ) cout << " The ▁ smaller ▁ circle ▁ lies ▁ completely " << " ▁ inside ▁ the ▁ bigger ▁ circle ▁ without " << " ▁ touching ▁ each ▁ other ▁ " << " at ▁ a ▁ point ▁ of ▁ circumference . ▁ " << endl ; else cout << " The ▁ smaller ▁ does ▁ not ▁ lies ▁ inside " << " ▁ the ▁ bigger ▁ circle ▁ completely . " << endl ; }
int main ( ) { int x1 = 10 , y1 = 8 ; int x2 = 1 , y2 = 2 ; int r1 = 30 , r2 = 10 ; circle ( x1 , y1 , x2 , y2 , r1 , r2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void lengtang ( double r1 , double r2 , double d ) { cout << " The ▁ length ▁ of ▁ the ▁ direct " << " ▁ common ▁ tangent ▁ is ▁ " << sqrt ( pow ( d , 2 ) - pow ( ( r1 - r2 ) , 2 ) ) << endl ; }
int main ( ) { double r1 = 4 , r2 = 6 , d = 3 ; lengtang ( r1 , r2 , d ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void rad ( double d , double h ) { cout << " The ▁ radius ▁ of ▁ the ▁ circle ▁ is ▁ " << ( ( d * d ) / ( 8 * h ) + h / 2 ) << endl ; }
int main ( ) { double d = 4 , h = 1 ; rad ( d , h ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void shortdis ( double r , double d ) { cout << " The ▁ shortest ▁ distance ▁ " << " from ▁ the ▁ chord ▁ to ▁ centre ▁ " << sqrt ( ( r * r ) - ( ( d * d ) / 4 ) ) << endl ; }
int main ( ) { double r = 4 , d = 3 ; shortdis ( r , d ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void lengtang ( double r1 , double r2 , double d ) { cout << " The ▁ length ▁ of ▁ the ▁ direct " << " ▁ common ▁ tangent ▁ is ▁ " << sqrt ( pow ( d , 2 ) - pow ( ( r1 - r2 ) , 2 ) ) << endl ; }
int main ( ) { double r1 = 4 , r2 = 6 , d = 12 ; lengtang ( r1 , r2 , d ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float square ( float a ) {
if ( a < 0 ) return -1 ;
float x = 0.464 * a ; return x ; }
int main ( ) { float a = 5 ; cout << square ( a ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float polyapothem ( float n , float a ) {
if ( a < 0 && n < 0 ) return -1 ;
return a / ( 2 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; }
int main ( ) { float a = 9 , n = 6 ; cout << polyapothem ( n , a ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float polyarea ( float n , float a ) {
if ( a < 0 && n < 0 ) return -1 ;
float A = ( a * a * n ) / ( 4 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; return A ; }
int main ( ) { float a = 9 , n = 6 ; cout << polyarea ( n , a ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float calculateSide ( float n , float r ) { float theta , theta_in_radians ; theta = 360 / n ; theta_in_radians = theta * 3.14 / 180 ; return 2 * r * sin ( theta_in_radians / 2 ) ; }
int main ( ) {
float n = 3 ;
float r = 5 ; cout << calculateSide ( n , r ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float cyl ( float r , float R , float h ) {
if ( h < 0 && r < 0 && R < 0 ) return -1 ;
float r1 = r ;
float h1 = h ;
float V = 3.14 * pow ( r1 , 2 ) * h1 ; return V ; }
int main ( ) { float r = 7 , R = 11 , h = 6 ; cout << cyl ( r , R , h ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
float Perimeter ( float s , int n ) { float perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
int main ( ) {
int n = 5 ;
float s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; cout << " Perimeter ▁ of ▁ Regular ▁ Polygon " << " ▁ with ▁ " << n << " ▁ sides ▁ of ▁ length ▁ " << s << " ▁ = ▁ " << peri << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float rhombusarea ( float l , float b ) {
if ( l < 0 b < 0 ) return -1 ;
return ( l * b ) / 2 ; }
int main ( ) { float l = 16 , b = 6 ; cout << rhombusarea ( l , b ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool FindPoint ( int x1 , int y1 , int x2 , int y2 , int x , int y ) { if ( x > x1 and x < x2 and y > y1 and y < y2 ) return true ; return false ; }
int main ( ) {
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x = 1 , y = 5 ;
if ( FindPoint ( x1 , y1 , x2 , y2 , x , y ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void shortest_distance ( float x1 , float y1 , float z1 , float a , float b , float c , float d ) { d = fabs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; float e = sqrt ( a * a + b * b + c * c ) ; cout << " Perpendicular ▁ distance ▁ is ▁ " << ( d / e ) ; return ; }
int main ( ) { float x1 = 4 ; float y1 = -4 ; float z1 = 3 ; float a = 2 ; float b = -2 ; float c = 5 ; float d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float findVolume ( float l , float b , float h ) {
float volume = ( l * b * h ) / 2 ; return volume ; }
int main ( ) { float l = 18 , b = 12 , h = 9 ;
cout << " Volume ▁ of ▁ triangular ▁ prism : ▁ " << findVolume ( l , b , h ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isRectangle ( int a , int b , int c , int d ) {
if ( a == b == c == d ) return true ; else if ( a == b && c == d ) return true ; else if ( a == d && c == b ) return true ; else if ( a == c && d == b ) return true ; else return false ; }
int main ( ) { int a , b , c , d ; a = 1 , b = 2 , c = 3 , d = 4 ; if ( isRectangle ( a , b , c , d ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void midpoint ( int x1 , int x2 , int y1 , int y2 ) { cout << ( float ) ( x1 + x2 ) / 2 << " ▁ , ▁ " << ( float ) ( y1 + y2 ) / 2 ; }
int main ( ) { int x1 = -1 , y1 = 2 ; int x2 = 3 , y2 = -6 ; midpoint ( x1 , x2 , y1 , y2 ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
double arcLength ( double diameter , double angle ) { double pi = 22.0 / 7.0 ; double arc ; if ( angle >= 360 ) { cout << " Angle ▁ cannot " , " ▁ be ▁ formed " ; return 0 ; } else { arc = ( pi * diameter ) * ( angle / 360.0 ) ; return arc ; } }
int main ( ) { double diameter = 25.0 ; double angle = 45.0 ; double arc_len = arcLength ( diameter , angle ) ; cout << ( arc_len ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void checkCollision ( int a , int b , int c , int x , int y , int radius ) {
int dist = ( abs ( a * x + b * y + c ) ) / sqrt ( a * a + b * b ) ;
if ( radius == dist ) cout << " Touch " << endl ; else if ( radius > dist ) cout << " Intersect " << endl ; else cout << " Outside " << endl ; }
int main ( ) { int radius = 5 ; int x = 0 , y = 0 ; int a = 3 , b = 4 , c = 25 ; checkCollision ( a , b , c , x , y , radius ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double polygonArea ( double X [ ] , double Y [ ] , int n ) {
double area = 0.0 ;
int j = n - 1 ; for ( int i = 0 ; i < n ; i ++ ) { area += ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) ;
}
return abs ( area / 2.0 ) ; }
int main ( ) { double X [ ] = { 0 , 2 , 4 } ; double Y [ ] = { 1 , 3 , 7 } ; int n = sizeof ( X ) / sizeof ( X [ 0 ] ) ; cout << polygonArea ( X , Y , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int chk ( int n ) {
vector < int > v ; while ( n != 0 ) { v . push_back ( n % 2 ) ; n = n / 2 ; } for ( int i = 0 ; i < v . size ( ) ; i ++ ) { if ( v [ i ] == 1 ) { return pow ( 2 , i ) ; } } return 0 ; }
void sumOfLSB ( int arr [ ] , int N ) {
vector < int > lsb_arr ; for ( int i = 0 ; i < N ; i ++ ) {
lsb_arr . push_back ( chk ( arr [ i ] ) ) ; }
sort ( lsb_arr . begin ( ) , lsb_arr . end ( ) , greater < int > ( ) ) ; int ans = 0 ; for ( int i = 0 ; i < N - 1 ; i += 2 ) {
ans += ( lsb_arr [ i + 1 ] ) ; }
cout << ( ans ) ; }
int main ( ) { int N = 5 ; int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ;
sumOfLSB ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countSubsequences ( vector < int > arr ) {
int odd = 0 ;
for ( int x : arr ) {
if ( x & 1 ) odd ++ ; }
return ( 1 << odd ) - 1 ; }
int main ( ) { vector < int > arr = { 1 , 3 , 3 } ;
cout << countSubsequences ( arr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getPairsCount ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = arr [ i ] - ( i % arr [ i ] ) ; j < n ; j += arr [ i ] ) {
if ( i < j && abs ( arr [ i ] - arr [ j ] ) >= min ( arr [ i ] , arr [ j ] ) ) { count ++ ; } } }
return count ; }
int main ( ) { int arr [ ] = { 1 , 2 , 2 , 3 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << getPairsCount ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void check ( int N ) { int twos = 0 , fives = 0 ;
while ( N % 2 == 0 ) { N /= 2 ; twos ++ ; }
while ( N % 5 == 0 ) { N /= 5 ; fives ++ ; } if ( N == 1 && twos <= fives ) { cout << 2 * fives - twos ; } else { cout << -1 ; } }
int main ( ) { int N = 50 ; check ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void rangeSum ( int arr [ ] , int N , int L , int R ) {
int sum = 0 ;
for ( int i = L - 1 ; i < R ; i ++ ) { sum += arr [ i % N ] ; }
cout << sum ; }
int main ( ) { int arr [ ] = { 5 , 2 , 6 , 9 } ; int L = 10 , R = 13 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; rangeSum ( arr , N , L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void rangeSum ( int arr [ ] , int N , int L , int R ) {
int prefix [ N + 1 ] ; prefix [ 0 ] = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] + arr [ i - 1 ] ; }
int leftsum = ( ( L - 1 ) / N ) * prefix [ N ] + prefix [ ( L - 1 ) % N ] ;
int rightsum = ( R / N ) * prefix [ N ] + prefix [ R % N ] ;
cout << rightsum - leftsum ; }
int main ( ) { int arr [ ] = { 5 , 2 , 6 , 9 } ; int L = 10 , R = 13 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; rangeSum ( arr , N , L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int ExpoFactorial ( int N ) {
int res = 1 ; int mod = 1000000007 ;
for ( int i = 2 ; i < N + 1 ; i ++ )
res = ( int ) pow ( i , res ) % mod ;
return res ; }
int main ( ) {
int N = 4 ;
cout << ( ExpoFactorial ( N ) ) ;
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSubArraySumRepeated ( int arr [ ] , int N , int K ) {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) sum += arr [ i ] ; int curr = arr [ 0 ] ;
int ans = arr [ 0 ] ;
if ( K == 1 ) {
for ( int i = 1 ; i < N ; i ++ ) { curr = max ( arr [ i ] , curr + arr [ i ] ) ; ans = max ( ans , curr ) ; }
return ans ; }
vector < int > V ;
for ( int i = 0 ; i < 2 * N ; i ++ ) { V . push_back ( arr [ i % N ] ) ; }
int maxSuf = V [ 0 ] ;
int maxPref = V [ 2 * N - 1 ] ; curr = V [ 0 ] ; for ( int i = 1 ; i < 2 * N ; i ++ ) { curr += V [ i ] ; maxPref = max ( maxPref , curr ) ; } curr = V [ 2 * N - 1 ] ; for ( int i = 2 * N - 2 ; i >= 0 ; i -- ) { curr += V [ i ] ; maxSuf = max ( maxSuf , curr ) ; } curr = V [ 0 ] ;
for ( int i = 1 ; i < 2 * N ; i ++ ) { curr = max ( V [ i ] , curr + V [ i ] ) ; ans = max ( ans , curr ) ; }
if ( sum > 0 ) { int temp = 1LL * sum * ( K - 2 ) ; ans = max ( ans , max ( temp + maxPref , temp + maxSuf ) ) ; }
return ans ; }
int main ( ) {
int arr [ ] = { 10 , 20 , -30 , -1 , 40 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int K = 10 ;
cout << maxSubArraySumRepeated ( arr , N , K ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void countSubarray ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i ; j < n ; j ++ ) {
int mxSubarray = 0 ;
int mxOther = 0 ;
for ( int k = i ; k <= j ; k ++ ) { mxSubarray = max ( mxSubarray , arr [ k ] ) ; }
for ( int k = 0 ; k < i ; k ++ ) { mxOther = max ( mxOther , arr [ k ] ) ; } for ( int k = j + 1 ; k < n ; k ++ ) { mxOther = max ( mxOther , arr [ k ] ) ; }
if ( mxSubarray > ( 2 * mxOther ) ) count ++ ; } }
cout << count ; }
int main ( ) { int arr [ ] = { 1 , 6 , 10 , 9 , 7 , 3 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; countSubarray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countSubarray ( int arr [ ] , int n ) { int count = 0 , L = 0 , R = 0 ;
int mx = * max_element ( arr , arr + n ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] * 2 > mx ) {
L = i ; break ; } } for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( arr [ i ] * 2 > mx ) {
R = i ; break ; } }
cout << ( L + 1 ) * ( n - R ) ; }
int main ( ) { int arr [ ] = { 1 , 6 , 10 , 9 , 7 , 3 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; countSubarray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int X ) { for ( int i = 2 ; i * i <= X ; i ++ )
return false ; return true ; }
void printPrimes ( int A [ ] , int N ) {
for ( int i = 0 ; i < N ; i ++ ) {
for ( int j = A [ i ] - 1 ; ; j -- ) {
if ( isPrime ( j ) ) { cout << j << " ▁ " ; break ; } }
for ( int j = A [ i ] + 1 ; ; j ++ ) {
if ( isPrime ( j ) ) { cout << j << " ▁ " ; break ; } } cout << endl ; } }
int main ( ) {
int A [ ] = { 17 , 28 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
printPrimes ( A , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int KthSmallest ( int A [ ] , int B [ ] , int N , int K ) { int M = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { M = max ( A [ i ] , M ) ; }
int freq [ M + 1 ] = { 0 } ;
for ( int i = 0 ; i < N ; i ++ ) { freq [ A [ i ] ] += B [ i ] ; }
int sum = 0 ;
for ( int i = 0 ; i <= M ; i ++ ) {
sum += freq [ i ] ;
if ( sum >= K ) {
return i ; } }
return -1 ; }
int main ( ) {
int A [ ] = { 3 , 4 , 5 } ; int B [ ] = { 2 , 1 , 3 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int K = 4 ;
cout << KthSmallest ( A , B , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findbitwiseOR ( int * a , int n ) {
int res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int curr_sub_array = a [ i ] ;
res = res | curr_sub_array ; for ( int j = i ; j < n ; j ++ ) {
curr_sub_array = curr_sub_array & a [ j ] ; res = res | curr_sub_array ; } }
cout << res ; }
int main ( ) { int A [ ] = { 1 , 2 , 3 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; findbitwiseOR ( A , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findbitwiseOR ( int * a , int n ) {
int res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) res = res | a [ i ] ;
cout << res ; }
int main ( ) { int A [ ] = { 1 , 2 , 3 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; findbitwiseOR ( A , N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void check ( int n ) {
int sumOfDigit = 0 ; int prodOfDigit = 1 ; while ( n > 0 ) {
int rem ; rem = n % 10 ;
sumOfDigit += rem ;
prodOfDigit *= rem ;
n /= 10 ; }
if ( sumOfDigit > prodOfDigit ) cout << " Yes " ; else cout << " No " ; }
int main ( ) { int N = 1234 ; check ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void evenOddBitwiseXOR ( int N ) { cout << " Even : ▁ " << 0 << " ▁ " ;
for ( int i = 4 ; i <= N ; i = i + 4 ) { cout << i << " ▁ " ; } cout << " STRNEWLINE " ; cout << " Odd : ▁ " << 1 << " ▁ " ;
for ( int i = 4 ; i <= N ; i = i + 4 ) { cout << i - 1 << " ▁ " ; } if ( N % 4 == 2 ) cout << N + 1 ; else if ( N % 4 == 3 ) cout << N ; }
int main ( ) { int N = 6 ; evenOddBitwiseXOR ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findPermutation ( vector < int > & arr ) { int N = arr . size ( ) ; int i = N - 2 ;
while ( i >= 0 && arr [ i ] <= arr [ i + 1 ] ) i -- ;
if ( i == -1 ) { cout << " - 1" ; return ; } int j = N - 1 ;
while ( j > i && arr [ j ] >= arr [ i ] ) j -- ;
while ( j > i && arr [ j ] == arr [ j - 1 ] ) {
j -- ; }
swap ( arr [ i ] , arr [ j ] ) ;
for ( auto & it : arr ) { cout << it << ' ▁ ' ; } }
int main ( ) { vector < int > arr = { 1 , 2 , 5 , 3 , 4 , 6 } ; findPermutation ( arr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void sieveOfEratosthenes ( int N , int s [ ] ) {
vector < bool > prime ( N + 1 , false ) ;
for ( int i = 2 ; i <= N ; i += 2 ) s [ i ] = 2 ;
for ( int i = 3 ; i <= N ; i += 2 ) {
if ( prime [ i ] == false ) { s [ i ] = i ;
for ( int j = i ; j * i <= N ; j += 2 ) {
if ( ! prime [ i * j ] ) { prime [ i * j ] = true ; s [ i * j ] = i ; } } } } }
void findDifference ( int N ) {
int s [ N + 1 ] ;
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
cout << abs ( even - odd ) ; }
int main ( ) { int N = 12 ; findDifference ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findMedian ( int Mean , int Mode ) {
double Median = ( 2 * Mean + Mode ) / 3.0 ;
cout << Median ; }
int main ( ) { int mode = 6 , mean = 3 ; findMedian ( mean , mode ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float vectorMagnitude ( int x , int y , int z ) {
int sum = x * x + y * y + z * z ;
return sqrt ( sum ) ; }
int main ( ) { int x = 1 ; int y = 2 ; int z = 3 ; cout << vectorMagnitude ( x , y , z ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long multiplyByMersenne ( long N , long M ) {
long x = log2 ( M + 1 ) ;
return ( ( N << x ) - N ) ; }
int main ( ) { long N = 4 ; long M = 15 ; cout << multiplyByMersenne ( N , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int perfectSquare ( int num ) {
int sr = sqrt ( num ) ;
int a = sr * sr ; int b = ( sr + 1 ) * ( sr + 1 ) ;
if ( ( num - a ) < ( b - num ) ) { return a ; } else { return b ; } }
int powerOfTwo ( int num ) {
int lg = log2 ( num ) ;
int p = pow ( 2 , lg ) ; return p ; }
void uniqueElement ( int arr [ ] , int N ) { bool ans = true ;
unordered_map < int , int > freq ;
for ( int i = 0 ; i < N ; i ++ ) { freq [ arr [ i ] ] ++ ; }
for ( auto el : freq ) {
if ( el . second == 1 ) { ans = false ;
int ps = perfectSquare ( el . first ) ;
cout << powerOfTwo ( ps ) << ' ▁ ' ; } }
if ( ans ) cout << " - 1" ; }
int main ( ) { int arr [ ] = { 4 , 11 , 4 , 3 , 4 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; uniqueElement ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void partitionArray ( int * a , int n ) {
int * Min = new int [ n ] ;
int Mini = INT_MAX ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
Mini = min ( Mini , a [ i ] ) ;
Min [ i ] = Mini ; }
int Maxi = INT_MIN ;
int ind = -1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
Maxi = max ( Maxi , a [ i ] ) ;
if ( Maxi < Min [ i + 1 ] ) {
ind = i ;
break ; } }
if ( ind != -1 ) {
for ( int i = 0 ; i <= ind ; i ++ ) cout << a [ i ] << " ▁ " ; cout << endl ;
for ( int i = ind + 1 ; i < n ; i ++ ) cout << a [ i ] << " ▁ " ; }
else cout < < " Impossible " ; }
int main ( ) { int arr [ ] = { 5 , 3 , 2 , 7 , 9 } ; int N = 5 ; partitionArray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countPrimeFactors ( int n ) { int count = 0 ;
while ( n % 2 == 0 ) { n = n / 2 ; count ++ ; }
for ( int i = 3 ; i <= sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { n = n / i ; count ++ ; } }
if ( n > 2 ) count ++ ; return ( count ) ; }
int findSum ( int n ) {
int sum = 0 ; for ( int i = 1 , num = 2 ; i <= n ; num ++ ) {
if ( countPrimeFactors ( num ) == 2 ) { sum += num ;
i ++ ; } } return sum ; }
void check ( int n , int k ) {
int s = findSum ( k - 1 ) ;
if ( s >= n ) cout << " No " ;
else cout < < " Yes " ; }
int main ( ) { int n = 100 , k = 6 ; check ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long int gcd ( long long int a , long long int b ) {
while ( b > 0 ) { long long int rem = a % b ; a = b ; b = rem ; }
return a ; }
int countNumberOfWays ( long long int n ) {
if ( n == 1 ) return -1 ;
long long int g = 0 ; int power = 0 ;
while ( n % 2 == 0 ) { power ++ ; n /= 2 ; } g = gcd ( g , power ) ;
for ( int i = 3 ; i <= sqrt ( n ) ; i += 2 ) { power = 0 ;
while ( n % i == 0 ) { power ++ ; n /= i ; } g = gcd ( g , power ) ; }
if ( n > 2 ) g = gcd ( g , 1 ) ;
int ways = 1 ;
power = 0 ; while ( g % 2 == 0 ) { g /= 2 ; power ++ ; }
ways *= ( power + 1 ) ;
for ( int i = 3 ; i <= sqrt ( g ) ; i += 2 ) { power = 0 ;
while ( g % i == 0 ) { power ++ ; g /= i ; }
ways *= ( power + 1 ) ; }
if ( g > 2 ) ways *= 2 ;
return ways ; }
int main ( ) { int N = 64 ; cout << countNumberOfWays ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int powOfPositive ( int n ) {
int pos = floor ( log2 ( n ) ) ; return pow ( 2 , pos ) ; }
int powOfNegative ( int n ) {
int pos = ceil ( log2 ( n ) ) ; return ( -1 * pow ( 2 , pos ) ) ; }
void highestPowerOf2 ( int n ) {
if ( n > 0 ) { cout << powOfPositive ( n ) ; } else {
n = - n ; cout << powOfNegative ( n ) ; } }
int main ( ) { int n = -24 ; highestPowerOf2 ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int noOfCards ( int n ) { return n * ( 3 * n + 1 ) / 2 ; }
int main ( ) { int n = 3 ; cout << noOfCards ( n ) << " , ▁ " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
string smallestPoss ( string s , int n ) {
string ans = " " ;
int arr [ 10 ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) { arr [ s [ i ] - 48 ] ++ ; }
for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < arr [ i ] ; j ++ ) ans = ans + to_string ( i ) ; }
return ans ; }
int main ( ) { int N = 15 ; string K = "325343273113434" ; cout << smallestPoss ( K , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Count_subarray ( int arr [ ] , int n ) { int subarray_sum , remaining_sum , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i ; j < n ; j ++ ) {
subarray_sum = 0 ; remaining_sum = 0 ;
for ( int k = i ; k <= j ; k ++ ) { subarray_sum += arr [ k ] ; }
for ( int l = 0 ; l < i ; l ++ ) { remaining_sum += arr [ l ] ; } for ( int l = j + 1 ; l < n ; l ++ ) { remaining_sum += arr [ l ] ; }
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
int main ( ) { int arr [ ] = { 10 , 9 , 12 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << Count_subarray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int Count_subarray ( int arr [ ] , int n ) { int total_sum = 0 , subarray_sum , remaining_sum , count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { total_sum += arr [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
subarray_sum = 0 ;
for ( int j = i ; j < n ; j ++ ) {
subarray_sum += arr [ j ] ; remaining_sum = total_sum - subarray_sum ;
if ( subarray_sum > remaining_sum ) { count += 1 ; } } } return count ; }
int main ( ) { int arr [ ] = { 10 , 9 , 12 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << Count_subarray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxXOR ( int * arr , int n ) {
int xorArr = 0 ; for ( int i = 0 ; i < n ; i ++ ) xorArr ^= arr [ i ] ;
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) ans = max ( ans , ( xorArr ^ arr [ i ] ) ) ;
return ans ; }
int main ( ) { int arr [ ] = { 1 , 1 , 3 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << maxXOR ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool digitDividesK ( int num , int k ) { while ( num ) {
int d = num % 10 ;
if ( d != 0 and k % d == 0 ) return true ;
num = num / 10 ; }
return false ; }
int findCount ( int l , int r , int k ) {
int count = 0 ;
for ( int i = l ; i <= r ; i ++ ) {
if ( digitDividesK ( i , k ) ) count ++ ; } return count ; }
int main ( ) { int l = 20 , r = 35 ; int k = 45 ; cout << findCount ( l , r , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isFactorial ( int n ) { for ( int i = 1 ; ; i ++ ) { if ( n % i == 0 ) { n /= i ; } else { break ; } } if ( n == 1 ) { return true ; } else { return false ; } }
int main ( ) { int n = 24 ; bool ans = isFactorial ( n ) ; if ( ans == 1 ) { cout << " Yes STRNEWLINE " ; } else { cout << " No STRNEWLINE " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int lcm ( int a , int b ) { int GCD = __gcd ( a , b ) ; return ( a * b ) / GCD ; }
int MinLCM ( int a [ ] , int n ) {
int Prefix [ n + 2 ] ; int Suffix [ n + 2 ] ;
Prefix [ 1 ] = a [ 0 ] ; for ( int i = 2 ; i <= n ; i += 1 ) { Prefix [ i ] = lcm ( Prefix [ i - 1 ] , a [ i - 1 ] ) ; }
Suffix [ n ] = a [ n - 1 ] ;
for ( int i = n - 1 ; i >= 1 ; i -= 1 ) { Suffix [ i ] = lcm ( Suffix [ i + 1 ] , a [ i - 1 ] ) ; }
int ans = min ( Suffix [ 2 ] , Prefix [ n - 1 ] ) ;
for ( int i = 2 ; i < n ; i += 1 ) { ans = min ( ans , lcm ( Prefix [ i - 1 ] , Suffix [ i + 1 ] ) ) ; }
return ans ; }
int main ( ) { int a [ ] = { 5 , 15 , 9 , 36 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << MinLCM ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int count ( int n ) { return n * ( 3 * n - 1 ) / 2 ; }
int main ( ) { int n = 3 ; cout << count ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinValue ( int arr [ ] , int n ) {
long sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
return ( ( sum / n ) + 1 ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 1 , 10 , 6 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << findMinValue ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MOD  1000000007
int modFact ( int n , int m ) { int result = 1 ; for ( int i = 1 ; i <= m ; i ++ ) result = ( result * i ) % MOD ; return result ; }
int main ( ) { int n = 3 , m = 2 ; cout << modFact ( n , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int mod = 1e9 + 7 ;
long long power ( int p ) { long long res = 1 ; for ( int i = 1 ; i <= p ; ++ i ) { res *= 2 ; res %= mod ; } return res % mod ; }
long long subset_square_sum ( vector < int > & A ) { int n = ( int ) A . size ( ) ; long long ans = 0 ;
for ( int i : A ) { ans += ( 1LL * i * i ) % mod ; ans %= mod ; } return ( 1LL * ans * power ( n - 1 ) ) % mod ; }
int main ( ) { vector < int > A = { 3 , 7 } ; cout << subset_square_sum ( A ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  100050 NEW_LINE int lpf [ N ] , mobius [ N ] ;
void least_prime_factor ( ) { for ( int i = 2 ; i < N ; i ++ )
if ( ! lpf [ i ] ) for ( int j = i ; j < N ; j += i )
if ( ! lpf [ j ] ) lpf [ j ] = i ; }
void Mobius ( ) { for ( int i = 1 ; i < N ; i ++ ) {
if ( i == 1 ) mobius [ i ] = 1 ; else {
if ( lpf [ i / lpf [ i ] ] == lpf [ i ] ) mobius [ i ] = 0 ;
else mobius [ i ] = -1 * mobius [ i / lpf [ i ] ] ; } } }
int gcd_pairs ( int a [ ] , int n ) {
int maxi = 0 ;
int fre [ N ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) { fre [ a [ i ] ] ++ ; maxi = max ( a [ i ] , maxi ) ; } least_prime_factor ( ) ; Mobius ( ) ;
int ans = 0 ;
for ( int i = 1 ; i <= maxi ; i ++ ) { if ( ! mobius [ i ] ) continue ; int temp = 0 ; for ( int j = i ; j <= maxi ; j += i ) temp += fre [ j ] ; ans += temp * ( temp - 1 ) / 2 * mobius [ i ] ; }
return ans ; }
int main ( ) { int a [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ;
cout << gcd_pairs ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void compareVal ( int x , int y ) {
long double a = y * log ( x ) ; long double b = x * log ( y ) ;
if ( a > b ) cout << x << " ^ " << y << " ▁ > ▁ " << y << " ^ " << x ; else if ( a < b ) cout << x << " ^ " << y << " ▁ < ▁ " << y << " ^ " << x ; else if ( a == b ) cout << x << " ^ " << y << " ▁ = ▁ " << y << " ^ " << x ; }
int main ( ) { long double x = 4 , y = 5 ; compareVal ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void ZigZag ( int n ) {
long long fact [ n + 1 ] , zig [ n + 1 ] = { 0 } ;
fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
zig [ 0 ] = 1 ; zig [ 1 ] = 1 ; cout << " zig ▁ zag ▁ numbers : ▁ " ;
cout << zig [ 0 ] << " ▁ " << zig [ 1 ] << " ▁ " ;
for ( int i = 2 ; i < n ; i ++ ) { long long sum = 0 ; for ( int k = 0 ; k <= i - 1 ; k ++ ) {
sum += ( fact [ i - 1 ] / ( fact [ i - 1 - k ] * fact [ k ] ) ) * zig [ k ] * zig [ i - 1 - k ] ; }
zig [ i ] = sum / 2 ;
cout << sum / 2 << " ▁ " ; } }
int main ( ) { int n = 10 ;
ZigZag ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int find_count ( vector < int > ele ) {
int count = 0 ; for ( int i = 0 ; i < ele . size ( ) ; i ++ ) {
vector < int > p ;
int c = 0 ;
for ( int j = ele . size ( ) - 1 ; j >= ( ele . size ( ) - 1 - i ) && j >= 0 ; j -- ) p . push_back ( ele [ j ] ) ; int j = ele . size ( ) - 1 , k = 0 ;
while ( j >= 0 ) {
if ( ele [ j ] != p [ k ] ) break ; j -- ; k ++ ;
if ( k == p . size ( ) ) { c ++ ; k = 0 ; } } count = max ( count , c ) ; }
return count ; }
void solve ( int n ) {
int count = 1 ;
vector < int > ele ;
for ( int i = 0 ; i < n ; i ++ ) { cout << count << " , ▁ " ;
ele . push_back ( count ) ;
count = find_count ( ele ) ; } }
int main ( ) { int n = 10 ; solve ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
map < int , int > store ;
int Wedderburn ( int n ) {
if ( n <= 2 ) return store [ n ] ;
else if ( n % 2 == 0 ) {
int x = n / 2 , ans = 0 ;
for ( int i = 1 ; i < x ; i ++ ) { ans += store [ i ] * store [ n - i ] ; }
ans += ( store [ x ] * ( store [ x ] + 1 ) ) / 2 ;
store [ n ] = ans ;
return ans ; } else {
int x = ( n + 1 ) / 2 , ans = 0 ;
for ( int i = 1 ; i < x ; i ++ ) { ans += store [ i ] * store [ n - i ] ; }
store [ n ] = ans ;
return ans ; } }
void Wedderburn_Etherington ( int n ) {
store [ 0 ] = 0 ; store [ 1 ] = 1 ; store [ 2 ] = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { cout << Wedderburn ( i ) ; if ( i != n - 1 ) cout << " , ▁ " ; } }
int main ( ) { int n = 10 ;
Wedderburn_Etherington ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Max_sum ( int a [ ] , int n ) {
int pos = 0 , neg = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] > 0 ) pos = 1 ;
else if ( a [ i ] < 0 ) neg = 1 ;
if ( pos == 1 and neg == 1 ) break ; }
int sum = 0 ; if ( pos == 1 and neg == 1 ) { for ( int i = 0 ; i < n ; i ++ ) sum += abs ( a [ i ] ) ; } else if ( pos == 1 ) {
int mini = a [ 0 ] ; sum = a [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { mini = min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; } else if ( neg == 1 ) {
for ( int i = 0 ; i < n ; i ++ ) a [ i ] = abs ( a [ i ] ) ;
int mini = a [ 0 ] ; sum = a [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) { mini = min ( mini , a [ i ] ) ; sum += a [ i ] ; }
sum -= 2 * mini ; }
return sum ; }
int main ( ) { int a [ ] = { 1 , 3 , 5 , -2 , -6 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ;
cout << Max_sum ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void decimalToBinary ( int n ) {
if ( n == 0 ) { cout << "0" ; return ; }
decimalToBinary ( n / 2 ) ; cout << n % 2 ; }
int main ( ) { int n = 13 ; decimalToBinary ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void MinimumValue ( int x , int y ) {
if ( x > y ) swap ( x , y ) ;
int a = 1 ; int b = x - 1 ; int c = y - b ; cout << a << " ▁ " << b << " ▁ " << c ; }
int main ( ) { int x = 123 , y = 13 ;
MinimumValue ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool canConvert ( int a , int b ) { while ( b > a ) {
if ( b % 10 == 1 ) { b /= 10 ; continue ; }
if ( b % 2 == 0 ) { b /= 2 ; continue ; }
return false ; }
if ( b == a ) return true ; return false ; }
int main ( ) { int A = 2 , B = 82 ; if ( canConvert ( A , B ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int count ( int N ) { int a = 0 ; a = ( N * ( N + 1 ) ) / 2 ; return a ; }
int main ( ) { int N = 4 ; cout << count ( N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int numberOfDays ( int a , int b , int n ) { int Days = b * ( n + a ) / ( a + b ) ; return Days ; }
int main ( ) { int a = 10 , b = 20 , n = 5 ; cout << numberOfDays ( a , b , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getAverage ( int x , int y ) {
int avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; return avg ; }
int main ( ) { int x = 10 , y = 9 ; cout << getAverage ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int smallestIndex ( int a [ ] , int n ) {
int right1 = 0 , right0 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == 1 ) right1 = i ;
else right0 = i ; }
return min ( right1 , right0 ) ; }
int main ( ) { int a [ ] = { 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << smallestIndex ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countSquares ( int r , int c , int m ) {
int squares = 0 ;
for ( int i = 1 ; i <= 8 ; i ++ ) { for ( int j = 1 ; j <= 8 ; j ++ ) {
if ( max ( abs ( i - r ) , abs ( j - c ) ) <= m ) squares ++ ; } }
return squares ; }
int main ( ) { int r = 4 , c = 4 , m = 1 ; cout << countSquares ( r , c , m ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countQuadruples ( int a [ ] , int n ) {
unordered_map < int , int > mpp ;
for ( int i = 0 ; i < n ; i ++ ) mpp [ a [ i ] ] ++ ; int count = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { for ( int k = 0 ; k < n ; k ++ ) {
if ( j == k ) continue ;
mpp [ a [ j ] ] -- ; mpp [ a [ k ] ] -- ;
int first = a [ j ] - ( a [ k ] - a [ j ] ) ;
int fourth = ( a [ k ] * a [ k ] ) / a [ j ] ;
if ( ( a [ k ] * a [ k ] ) % a [ j ] == 0 ) {
if ( a [ j ] != a [ k ] ) count += mpp [ first ] * mpp [ fourth ] ;
else count += mpp [ first ] * ( mpp [ fourth ] - 1 ) ; }
mpp [ a [ j ] ] ++ ; mpp [ a [ k ] ] ++ ; } } return count ; }
int main ( ) { int a [ ] = { 2 , 6 , 4 , 9 , 2 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << countQuadruples ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define ll  long long int NEW_LINE using namespace std ;
int countNumbers ( int L , int R , int K ) { if ( K == 9 ) K = 0 ;
int totalnumbers = R - L + 1 ;
int factor9 = totalnumbers / 9 ;
int rem = totalnumbers % 9 ;
int ans = factor9 ;
for ( int i = R ; i > R - rem ; i -- ) { int rem1 = i % 9 ; if ( rem1 == K ) ans ++ ; } return ans ; }
int main ( ) { int L = 10 ; int R = 22 ; int K = 3 ; cout << countNumbers ( L , R , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int EvenSum ( vector < int > & A , int index , int value ) {
A [ index ] = A [ index ] + value ;
int sum = 0 ; for ( int i = 0 ; i < A . size ( ) ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; return sum ; }
void BalanceArray ( vector < int > & A , vector < vector < int > > & Q ) {
vector < int > ANS ; int i , sum ; for ( i = 0 ; i < Q . size ( ) ; i ++ ) { int index = Q [ i ] [ 0 ] ; int value = Q [ i ] [ 1 ] ;
sum = EvenSum ( A , index , value ) ;
ANS . push_back ( sum ) ; }
for ( i = 0 ; i < ANS . size ( ) ; i ++ ) cout << ANS [ i ] << " ▁ " ; }
int main ( ) { vector < int > A = { 1 , 2 , 3 , 4 } ; vector < vector < int > > Q = { { 0 , 1 } , { 1 , -3 } , { 0 , -4 } , { 3 , 2 } } ; BalanceArray ( A , Q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void BalanceArray ( vector < int > & A , vector < vector < int > > & Q ) { vector < int > ANS ; int i , sum = 0 ; for ( i = 0 ; i < A . size ( ) ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; for ( i = 0 ; i < Q . size ( ) ; i ++ ) { int index = Q [ i ] [ 0 ] ; int value = Q [ i ] [ 1 ] ;
if ( A [ index ] % 2 == 0 ) sum = sum - A [ index ] ; A [ index ] = A [ index ] + value ;
if ( A [ index ] % 2 == 0 ) sum = sum + A [ index ] ;
ANS . push_back ( sum ) ; }
for ( i = 0 ; i < ANS . size ( ) ; i ++ ) cout << ANS [ i ] << " ▁ " ; }
int main ( ) { vector < int > A = { 1 , 2 , 3 , 4 } ; vector < vector < int > > Q = { { 0 , 1 } , { 1 , -3 } , { 0 , -4 } , { 3 , 2 } } ; BalanceArray ( A , Q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Cycles ( int N ) { int fact = 1 , result = 0 ; result = N - 1 ;
int i = result ; while ( i > 0 ) { fact = fact * i ; i -- ; } return fact / 2 ; }
int main ( ) { int N = 5 ; int Number = Cycles ( N ) ; cout << " Hamiltonian ▁ cycles ▁ = ▁ " << Number ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool digitWell ( int n , int m , int k ) { int cnt = 0 ; while ( n > 0 ) { if ( n % 10 == m ) ++ cnt ; n /= 10 ; } return cnt == k ; }
int findInt ( int n , int m , int k ) { int i = n + 1 ; while ( true ) { if ( digitWell ( i , m , k ) ) return i ; i ++ ; } }
int main ( ) { int n = 111 , m = 2 , k = 2 ; cout << findInt ( n , m , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countOdd ( int arr [ ] , int n ) {
int odd = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] % 2 == 1 ) odd ++ ; } return odd ; }
int countValidPairs ( int arr [ ] , int n ) { int odd = countOdd ( arr , n ) ; return ( odd * ( odd - 1 ) ) / 2 ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << countValidPairs ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ll  long long int
ll gcd ( ll a , ll b ) { if ( b == 0 ) return a ; else return gcd ( b , a % b ) ; }
ll lcmOfArray ( int arr [ ] , int n ) { if ( n < 1 ) return 0 ; ll lcm = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) lcm = ( lcm * arr [ i ] ) / gcd ( lcm , arr [ i ] ) ;
return lcm ; }
int minPerfectCube ( int arr [ ] , int n ) { ll minPerfectCube ;
ll lcm = lcmOfArray ( arr , n ) ; minPerfectCube = ( long long ) lcm ; int cnt = 0 ; while ( lcm > 1 && lcm % 2 == 0 ) { cnt ++ ; lcm /= 2 ; }
if ( cnt % 3 == 2 ) minPerfectCube *= 2 ; else if ( cnt % 3 == 1 ) minPerfectCube *= 4 ; int i = 3 ;
while ( lcm > 1 ) { cnt = 0 ; while ( lcm % i == 0 ) { cnt ++ ; lcm /= i ; } if ( cnt % 3 == 1 ) minPerfectCube *= i * i ; else if ( cnt % 3 == 2 ) minPerfectCube *= i ; i += 2 ; }
return minPerfectCube ; }
int main ( ) { int arr [ ] = { 10 , 125 , 14 , 42 , 100 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minPerfectCube ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static bool isStrongPrime ( int n ) {
if ( ! isPrime ( n ) n == 2 ) return false ;
int previous_prime = n - 1 ; int next_prime = n + 1 ;
while ( ! isPrime ( next_prime ) ) next_prime ++ ;
while ( ! isPrime ( previous_prime ) ) previous_prime -- ;
int mean = ( previous_prime + next_prime ) / 2 ;
if ( n > mean ) return true ; else return false ; }
int main ( ) { int n = 11 ; if ( isStrongPrime ( n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countDigitsToBeRemoved ( int N , int K ) {
string s = to_string ( N ) ;
int res = 0 ;
int f_zero = 0 ; for ( int i = s . size ( ) - 1 ; i >= 0 ; i -- ) { if ( K == 0 ) return res ; if ( s [ i ] == '0' ) {
f_zero = 1 ; K -- ; } else res ++ ; }
if ( ! K ) return res ; else if ( f_zero ) return s . size ( ) - 1 ; return -1 ; }
int main ( ) { int N = 10904025 , K = 2 ; cout << countDigitsToBeRemoved ( N , K ) << endl ; N = 1000 , K = 5 ; cout << countDigitsToBeRemoved ( N , K ) << endl ; N = 23985 , K = 2 ; cout << countDigitsToBeRemoved ( N , K ) << endl ; return 0 ; }
#include <stdio.h> NEW_LINE #include <math.h> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
float getSum ( int a , int n ) {
float sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) {
sum += ( i / pow ( a , i ) ) ; } return sum ; }
int main ( ) { int a = 3 , n = 3 ;
cout << ( getSum ( a , n ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int largestPrimeFactor ( int n ) {
int max = -1 ;
while ( n % 2 == 0 ) { max = 2 ;
}
for ( int i = 3 ; i <= sqrt ( n ) ; i += 2 ) { while ( n % i == 0 ) { max = i ; n = n / i ; } }
if ( n > 2 ) max = n ; return max ; }
bool checkUnusual ( int n ) {
int factor = largestPrimeFactor ( n ) ;
if ( factor > sqrt ( n ) ) { return true ; } else { return false ; } }
int main ( ) { int n = 14 ; if ( checkUnusual ( n ) ) { cout << " YES " << " STRNEWLINE " ; } else { cout << " NO " << " STRNEWLINE " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void isHalfReducible ( int arr [ ] , int n , int m ) { int frequencyHash [ m + 1 ] ; int i ; memset ( frequencyHash , 0 , sizeof ( frequencyHash ) ) ; for ( i = 0 ; i < n ; i ++ ) { frequencyHash [ arr [ i ] % ( m + 1 ) ] ++ ; } for ( i = 0 ; i <= m ; i ++ ) { if ( frequencyHash [ i ] >= n / 2 ) break ; } if ( i <= m ) cout << " Yes " << endl ; else cout << " No " << endl ; }
int main ( ) { int arr [ ] = { 8 , 16 , 32 , 3 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int m = 7 ; isHalfReducible ( arr , n , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; vector < int > arr ;
void generateDivisors ( int n ) {
for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) arr . push_back ( i ) ;
{ arr . push_back ( i ) ; arr . push_back ( n / i ) ; } } } }
double harmonicMean ( int n ) { generateDivisors ( n ) ;
double sum = 0.0 ; int len = arr . size ( ) ;
for ( int i = 0 ; i < len ; i ++ ) sum = sum + double ( n / arr [ i ] ) ; sum = double ( sum / n ) ;
return double ( arr . size ( ) / sum ) ; }
bool isOreNumber ( int n ) {
double mean = harmonicMean ( n ) ;
if ( mean - int ( mean ) == 0 ) return true ; else return false ; }
int main ( ) { int n = 28 ; if ( isOreNumber ( n ) ) cout << " YES " ; else cout << " NO " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  10000 NEW_LINE unordered_set < long long int > s ;
void SieveOfEratosthenes ( ) {
bool prime [ MAX ] ; memset ( prime , true , sizeof ( prime ) ) ; for ( int p = 2 ; p * p < MAX ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < MAX ; i += p ) prime [ i ] = false ; } }
long long int product = 1 ; for ( int p = 2 ; p < MAX ; p ++ ) { if ( prime [ p ] ) {
product = product * p ;
s . insert ( product + 1 ) ; } } }
bool isEuclid ( long n ) {
if ( s . find ( n ) != s . end ( ) ) return true ; else return false ; }
int main ( ) {
SieveOfEratosthenes ( ) ;
long n = 31 ;
if ( isEuclid ( n ) ) cout << " YES STRNEWLINE " ; else cout << " NO STRNEWLINE " ;
n = 42 ;
if ( isEuclid ( n ) ) cout << " YES STRNEWLINE " ; else cout << " NO STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) { return false ; } } return true ; }
bool isPowerOfTwo ( int n ) { return ( n && ! ( n & ( n - 1 ) ) ) ; }
int main ( ) { int n = 43 ;
if ( isPrime ( n ) && ( isPowerOfTwo ( n * 3 - 1 ) ) ) { cout << " YES STRNEWLINE " ; } else { cout << " NO STRNEWLINE " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float area ( float a ) {
if ( a < 0 ) return -1 ;
float area = pow ( ( a * sqrt ( 3 ) ) / ( sqrt ( 2 ) ) , 2 ) ; return area ; }
int main ( ) { float a = 5 ; cout << area ( a ) << endl ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
int nthTerm ( int n ) { return 3 * pow ( n , 2 ) - 4 * n + 2 ; }
int main ( ) { int N = 4 ; cout << nthTerm ( N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int calculateSum ( int n ) { return n * ( n + 1 ) / 2 + pow ( ( n * ( n + 1 ) / 2 ) , 2 ) ; }
int main ( ) {
int n = 3 ;
cout << " Sum ▁ = ▁ " << calculateSum ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool arePermutations ( int a [ ] , int b [ ] , int n , int m ) { int sum1 = 0 , sum2 = 0 , mul1 = 1 , mul2 = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { sum1 += a [ i ] ; mul1 *= a [ i ] ; }
for ( int i = 0 ; i < m ; i ++ ) { sum2 += b [ i ] ; mul2 *= b [ i ] ; }
return ( ( sum1 == sum2 ) && ( mul1 == mul2 ) ) ; }
int main ( ) { int a [ ] = { 1 , 3 , 2 } ; int b [ ] = { 3 , 1 , 2 } ; int n = sizeof ( a ) / sizeof ( int ) ; int m = sizeof ( b ) / sizeof ( int ) ; if ( arePermutations ( a , b , n , m ) ) cout << " Yes " << endl ; else cout << " No " << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Race ( int B , int C ) { int result = 0 ;
result = ( ( C * 100 ) / B ) ; return 100 - result ; }
int main ( ) { int B = 10 , C = 28 ;
B = 100 - B ; C = 100 - C ; cout << Race ( B , C ) << " ▁ meters " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float Time ( float arr [ ] , int n , int Emptypipe ) { float fill = 0 ; for ( int i = 0 ; i < n ; i ++ ) fill += 1 / arr [ i ] ; fill = fill - ( 1 / ( float ) Emptypipe ) ; return 1 / fill ; }
int main ( ) { float arr [ ] = { 12 , 14 } ; float Emptypipe = 30 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << floor ( Time ( arr , n , Emptypipe ) ) << " ▁ Hours " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int check ( int n ) { int sum = 0 ;
while ( n != 0 ) { sum += n % 10 ; n = n / 10 ; }
if ( sum % 7 == 0 ) return 1 ; else return 0 ; }
int main ( ) {
int n = 25 ; ( check ( n ) == 1 ) ? cout << " YES " : cout << " NO " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  1000005
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
int SumOfPrimeDivisors ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { if ( n % i == 0 ) { if ( isPrime ( i ) ) sum += i ; } } return sum ; }
int main ( ) { int n = 60 ; cout << " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " << SumOfPrimeDivisors ( n ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Sum ( int N ) { int SumOfPrimeDivisors [ N + 1 ] = { 0 } ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( ! SumOfPrimeDivisors [ i ] ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
int main ( ) { int N = 60 ; cout << " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " << Sum ( N ) << endl ; }
#include <bits/stdc++.h> NEW_LINE #define ll  long long int NEW_LINE using namespace std ;
ll power ( ll x , ll y , ll p ) {
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
int main ( ) { ll a = 3 ;
string b = "100000000000000000000000000" ; ll remainderB = 0 ; ll MOD = 1000000007 ;
for ( int i = 0 ; i < b . length ( ) ; i ++ ) remainderB = ( remainderB * 10 + b [ i ] - '0' ) % ( MOD - 1 ) ; cout << power ( a , remainderB , MOD ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
string find_Square_369 ( string num ) { char a , b , c , d ;
if ( num [ 0 ] == '3' ) a = '1' , b = '0' , c = '8' , d = '9' ;
else if ( num [ 0 ] == '6' ) a = '4' , b = '3' , c = '5' , d = '6' ;
else a = '9' , b = '8' , c = '0' , d = '1' ;
string result = " " ;
int size = num . size ( ) ;
for ( int i = 1 ; i < num . size ( ) ; i ++ ) result += a ;
result += b ;
for ( int i = 1 ; i < num . size ( ) ; i ++ ) result += c ;
result += d ;
return result ; }
int main ( ) { string num_3 , num_6 , num_9 ; num_3 = "3333" ; num_6 = "6666" ; num_9 = "9999" ; string result = " " ;
result = find_Square_369 ( num_3 ) ; cout << " Square ▁ of ▁ " << num_3 << " ▁ is ▁ : ▁ " << result << endl ;
result = find_Square_369 ( num_6 ) ; cout << " Square ▁ of ▁ " << num_6 << " ▁ is ▁ : ▁ " << result << endl ;
result = find_Square_369 ( num_9 ) ; cout << " Square ▁ of ▁ " << num_9 << " ▁ is ▁ : ▁ " << result << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) { long int ans = 1 ; long int mod = ( long int ) 1000000007 * 120 ; for ( int i = 0 ; i < 5 ; i ++ ) ans = ( ans * ( 55555 - i ) ) % mod ; ans = ans / 120 ; cout << " Answer ▁ using ▁ shortcut : ▁ " << ans ; return 0 ; }
# include <bits/stdc++.h> NEW_LINE using namespace std ;
int fact ( int n ) { if ( n == 0 n == 1 ) return 1 ; int ans = 1 ; for ( int i = 1 ; i <= n ; i ++ ) ans = ans * i ; return ans ; }
int nCr ( int n , int r ) { int Nr = n , Dr = 1 , ans = 1 ; for ( int i = 1 ; i <= r ; i ++ ) { ans = ( ans * Nr ) / ( Dr ) ; Nr -- ; Dr ++ ; } return ans ; }
int solve ( int n ) { int N = 2 * n - 2 ; int R = n - 1 ; return nCr ( N , R ) * fact ( n - 1 ) ; }
int main ( ) { int n = 6 ; cout << solve ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void pythagoreanTriplet ( int n ) {
for ( int i = 1 ; i <= n / 3 ; i ++ ) {
for ( int j = i + 1 ; j <= n / 2 ; j ++ ) { int k = n - i - j ; if ( i * i + j * j == k * k ) { cout << i << " , ▁ " << j << " , ▁ " << k ; return ; } } } cout << " No ▁ Triplet " ; }
int main ( ) { int n = 12 ; pythagoreanTriplet ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int factorial ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
void series ( int A , int X , int n ) {
int nFact = factorial ( n ) ;
for ( int i = 0 ; i < n + 1 ; i ++ ) {
int niFact = factorial ( n - i ) ; int iFact = factorial ( i ) ;
int aPow = pow ( A , n - i ) ; int xPow = pow ( X , i ) ;
cout << ( nFact * aPow * xPow ) / ( niFact * iFact ) << " ▁ " ; } }
int main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int seiresSum ( int n , int a [ ] ) { int res = 0 ; for ( int i = 0 ; i < 2 * n ; i ++ ) { if ( i % 2 == 0 ) res += a [ i ] * a [ i ] ; else res -= a [ i ] * a [ i ] ; } return res ; }
int main ( ) { int n = 2 ; int a [ ] = { 1 , 2 , 3 , 4 } ; cout << seiresSum ( n , a ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int power ( int n , int r ) {
int count = 0 ; for ( int i = r ; ( n / i ) >= 1 ; i = i * r ) count += n / i ; return count ; }
int main ( ) { int n = 6 , r = 3 ; printf ( " ▁ % d ▁ " , power ( n , r ) ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int avg_of_odd_num ( int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += ( 2 * i + 1 ) ;
return sum / n ; }
int main ( ) { int n = 20 ; cout << avg_of_odd_num ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int avg_of_odd_num ( int n ) { return n ; }
int main ( ) { int n = 8 ; cout << avg_of_odd_num ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void fib ( int f [ ] , int N ) {
f [ 1 ] = 1 ; f [ 2 ] = 1 ; for ( int i = 3 ; i <= N ; i ++ )
f [ i ] = f [ i - 1 ] + f [ i - 2 ] ; } void fiboTriangle ( int n ) {
int N = n * ( n + 1 ) / 2 ; int f [ N + 1 ] ; fib ( f , N ) ;
int fiboNum = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) cout << f [ fiboNum ++ ] << " ▁ " ; cout << endl ; } }
int main ( ) { int n = 5 ; fiboTriangle ( n ) ; return 0 ; }
#include <stdio.h>
int averageOdd ( int n ) { if ( n % 2 == 0 ) { printf ( " Invalid ▁ Input " ) ; return -1 ; } int sum = 0 , count = 0 ; while ( n >= 1 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
int main ( ) { int n = 15 ; printf ( " % d " , averageOdd ( n ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; struct Rational { int nume , deno ; } ;
int lcm ( int a , int b ) { return ( a * b ) / ( __gcd ( a , b ) ) ; }
Rational maxRational ( Rational first , Rational sec ) {
int k = lcm ( first . deno , sec . deno ) ;
int nume1 = first . nume ; int nume2 = sec . nume ; nume1 *= k / ( first . deno ) ; nume2 *= k / ( sec . deno ) ; return ( nume2 < nume1 ) ? first : sec ; }
int main ( ) { Rational first = { 3 , 2 } ; Rational sec = { 3 , 4 } ; Rational res = maxRational ( first , sec ) ; cout << res . nume << " / " << res . deno ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int TrinomialValue ( int n , int k ) {
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ; }
void printTrinomial ( int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = - i ; j <= 0 ; j ++ ) cout << TrinomialValue ( i , j ) << " ▁ " ;
for ( int j = 1 ; j <= i ; j ++ ) cout << TrinomialValue ( i , j ) << " ▁ " ; cout << endl ; } }
int main ( ) { int n = 4 ; printTrinomial ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int sumOfLargePrimeFactor ( int n ) {
int prime [ n + 1 ] , sum = 0 ; memset ( prime , 0 , sizeof ( prime ) ) ; int max = n / 2 ; for ( int p = 2 ; p <= max ; p ++ ) {
if ( prime [ p ] == 0 ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = p ; } }
for ( int p = 2 ; p <= n ; p ++ ) {
if ( prime [ p ] ) sum += prime [ p ] ;
else sum += p ; }
return sum ; }
int main ( ) { int n = 12 ; cout << " Sum ▁ = ▁ " << sumOfLargePrimeFactor ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int calculate_sum ( int a , int N ) {
int m = N / a ;
int sum = m * ( m + 1 ) / 2 ;
int ans = a * sum ; return ans ; }
int main ( ) { int a = 7 , N = 49 ; cout << " Sum ▁ of ▁ multiples ▁ of ▁ " << a << " ▁ up ▁ to ▁ " << N << " ▁ = ▁ " << calculate_sum ( a , N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <boost/multiprecision/cpp_int.hpp> NEW_LINE using namespace std ; using namespace boost :: multiprecision ;
bool ispowerof2 ( cpp_int num ) { if ( ( num & ( num - 1 ) ) == 0 ) return 1 ; return 0 ; }
int main ( ) { cpp_int num = 549755813888 ; cout << ispowerof2 ( num ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int counDivisors ( int X ) {
int count = 0 ;
for ( int i = 1 ; i <= X ; ++ i ) { if ( X % i == 0 ) { count ++ ; } }
return count ; }
int countDivisorsMult ( int arr [ ] , int n ) {
int mul = 1 ; for ( int i = 0 ; i < n ; ++ i ) mul *= arr [ i ] ;
return counDivisors ( mul ) ; }
int main ( ) { int arr [ ] = { 2 , 4 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << countDivisorsMult ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void SieveOfEratosthenes ( int largest , vector < int > & prime ) {
bool isPrime [ largest + 1 ] ; memset ( isPrime , true , sizeof ( isPrime ) ) ; for ( int p = 2 ; p * p <= largest ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( int i = p * 2 ; i <= largest ; i += p ) isPrime [ i ] = false ; } }
for ( int p = 2 ; p <= largest ; p ++ ) if ( isPrime [ p ] ) prime . push_back ( p ) ; }
int countDivisorsMult ( int arr [ ] , int n ) {
int largest = * max_element ( arr , arr + n ) ; vector < int > prime ; SieveOfEratosthenes ( largest , prime ) ;
unordered_map < int , int > mp ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < prime . size ( ) ; j ++ ) { while ( arr [ i ] > 1 && arr [ i ] % prime [ j ] == 0 ) { arr [ i ] /= prime [ j ] ; mp [ prime [ j ] ] ++ ; } } if ( arr [ i ] != 1 ) mp [ arr [ i ] ] ++ ; }
long long int res = 1 ; for ( auto it : mp ) res *= ( it . second + 1L ) ; return res ; }
int main ( ) { int arr [ ] = { 2 , 4 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << countDivisorsMult ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findPrimeNos ( int L , int R , unordered_map < int , int > & M ) {
for ( int i = L ; i <= R ; i ++ ) { M [ i ] ++ ; }
if ( M . find ( 1 ) != M . end ( ) ) { M . erase ( 1 ) ; }
for ( int i = 2 ; i <= sqrt ( R ) ; i ++ ) { int multiple = 2 ; while ( ( i * multiple ) <= R ) {
if ( M . find ( i * multiple ) != M . end ( ) ) {
M . erase ( i * multiple ) ; }
multiple ++ ; } } }
void getPrimePairs ( int L , int R , int K ) { unordered_map < int , int > M ;
findPrimeNos ( L , R , M ) ;
for ( auto & it : M ) {
if ( M . find ( it . first + K ) != M . end ( ) ) { cout << " ( " << it . first << " , ▁ " << it . first + K << " ) ▁ " ; } } }
int main ( ) {
int L = 1 , R = 19 ;
int K = 6 ;
getPrimePairs ( L , R , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int EnneacontahexagonNum ( int n ) { return ( 94 * n * n - 92 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << EnneacontahexagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void find_composite_nos ( int n ) { cout << 9 * n << " ▁ " << 8 * n ; }
int main ( ) { int n = 4 ; find_composite_nos ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int freqPairs ( int arr [ ] , int n ) {
int max = * ( std :: max_element ( arr , arr + n ) ) ;
int freq [ max + 1 ] = { 0 } ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) freq [ arr [ i ] ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 2 * arr [ i ] ; j <= max ; j += arr [ i ] ) {
if ( freq [ j ] >= 1 ) count += freq [ j ] ; }
if ( freq [ arr [ i ] ] > 1 ) { count += freq [ arr [ i ] ] - 1 ; freq [ arr [ i ] ] -- ; } } return count ; }
int main ( ) { int arr [ ] = { 3 , 2 , 4 , 2 , 6 } ; int n = ( sizeof ( arr ) / sizeof ( arr [ 0 ] ) ) ; cout << freqPairs ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
int Nth_Term ( int n ) { return ( 2 * pow ( n , 3 ) - 3 * pow ( n , 2 ) + n + 6 ) / 6 ; }
int main ( ) { int N = 8 ; cout << Nth_Term ( N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int printNthElement ( int n ) {
int arr [ n + 1 ] ; arr [ 1 ] = 3 ; arr [ 2 ] = 5 ; for ( int i = 3 ; i <= n ; i ++ ) {
if ( i % 2 != 0 ) arr [ i ] = arr [ i / 2 ] * 10 + 3 ; else arr [ i ] = arr [ ( i / 2 ) - 1 ] * 10 + 5 ; } return arr [ n ] ; }
int main ( ) { int n = 6 ; cout << printNthElement ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int nthTerm ( int N ) {
return ( N * ( ( N / 2 ) + ( ( N % 2 ) * 2 ) + N ) ) ; }
int main ( ) {
int N = 5 ;
cout << " Nth ▁ term ▁ for ▁ N ▁ = ▁ " << N << " ▁ : ▁ " << nthTerm ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void series ( int A , int X , int n ) {
int term = pow ( A , n ) ; cout << term << " ▁ " ;
for ( int i = 1 ; i <= n ; i ++ ) {
term = term * X * ( n - i + 1 ) / ( i * A ) ; cout << term << " ▁ " ; } }
int main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Div_by_8 ( int n ) { return ( ( ( n >> 3 ) << 3 ) == n ) ; }
int main ( ) { int n = 16 ; if ( Div_by_8 ( n ) ) cout << " YES " << endl ; else cout << " NO " << endl ; return 0 ; }
#include <stdio.h>
int averageEven ( int n ) { if ( n % 2 != 0 ) { printf ( " Invalid ▁ Input " ) ; return -1 ; } int sum = 0 , count = 0 ; while ( n >= 2 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
int main ( ) { int n = 16 ; printf ( " % d " , averageEven ( n ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int averageEven ( int n ) { if ( n % 2 != 0 ) { cout << " Invalid ▁ Input " ; return -1 ; } return ( n + 2 ) / 2 ; }
int main ( ) { int n = 16 ; cout << averageEven ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int gcd ( int a , int b ) {
if ( a == 0 b == 0 ) return 0 ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
int cpFact ( int x , int y ) { while ( gcd ( x , y ) != 1 ) { x = x / gcd ( x , y ) ; } return x ; }
int main ( ) { int x = 15 ; int y = 3 ; cout << cpFact ( x , y ) << endl ; x = 14 ; y = 28 ; cout << cpFact ( x , y ) << endl ; x = 7 ; y = 3 ; cout << cpFact ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int counLastDigitK ( int low , int high , int k ) { int count = 0 ; for ( int i = low ; i <= high ; i ++ ) if ( i % 10 == k ) count ++ ; return count ; }
int main ( ) { int low = 3 , high = 35 , k = 3 ; cout << counLastDigitK ( low , high , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printTaxicab2 ( int N ) {
int i = 1 , count = 0 ; while ( count < N ) { int int_count = 0 ;
for ( int j = 1 ; j <= pow ( i , 1.0 / 3 ) ; j ++ ) for ( int k = j + 1 ; k <= pow ( i , 1.0 / 3 ) ; k ++ ) if ( j * j * j + k * k * k == i ) int_count ++ ;
if ( int_count == 2 ) { count ++ ; cout << count << " ▁ " << i << endl ; } i ++ ; } }
int main ( ) { int N = 5 ; printTaxicab2 ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isComposite ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return false ;
if ( n % 2 == 0 n % 3 == 0 ) return true ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return true ; return false ; }
int main ( ) { isComposite ( 11 ) ? cout << " ▁ true STRNEWLINE " : cout << " ▁ false STRNEWLINE " ; isComposite ( 15 ) ? cout << " ▁ true STRNEWLINE " : cout << " ▁ false STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
int findPrime ( int n ) { int num = n + 1 ;
while ( num ) {
if ( isPrime ( num ) ) return num ;
num = num + 1 ; } return 0 ; }
int minNumber ( int arr [ ] , int n ) { int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( isPrime ( sum ) ) return 0 ;
int num = findPrime ( sum ) ;
return num - sum ; }
int main ( ) { int arr [ ] = { 2 , 4 , 6 , 8 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minNumber ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int fact ( int n ) { if ( n == 0 ) return 1 ; return n * fact ( n - 1 ) ; }
int div ( int x ) { int ans = 0 ; for ( int i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
int sumFactDiv ( int n ) { return div ( fact ( n ) ) ; }
int main ( ) { int n = 4 ; cout << sumFactDiv ( n ) ; }
#include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
vector < int > allPrimes ;
void sieve ( int n ) {
vector < bool > prime ( n + 1 , true ) ;
for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = false ; } }
for ( int p = 2 ; p <= n ; p ++ ) if ( prime [ p ] ) allPrimes . push_back ( p ) ; }
int factorialDivisors ( int n ) {
int result = 1 ;
for ( int i = 0 ; i < allPrimes . size ( ) ; i ++ ) {
int p = allPrimes [ i ] ;
int exp = 0 ; while ( p <= n ) { exp = exp + ( n / p ) ; p = p * allPrimes [ i ] ; }
result = result * ( pow ( allPrimes [ i ] , exp + 1 ) - 1 ) / ( allPrimes [ i ] - 1 ) ; }
return result ; }
int main ( ) { cout << factorialDivisors ( 4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkPandigital ( int b , char n [ ] ) {
if ( strlen ( n ) < b ) return false ; bool hash [ b ] ; memset ( hash , false , sizeof ( hash ) ) ;
for ( int i = 0 ; i < strlen ( n ) ; i ++ ) {
if ( n [ i ] >= '0' && n [ i ] <= '9' ) hash [ n [ i ] - '0' ] = true ;
else if ( n [ i ] - ' A ' <= b - 11 ) hash [ n [ i ] - ' A ' + 10 ] = true ; }
for ( int i = 0 ; i < b ; i ++ ) if ( hash [ i ] == false ) return false ; return true ; }
int main ( ) { int b = 13 ; char n [ ] = "1298450376ABC " ; ( checkPandigital ( b , n ) ) ? ( cout << " Yes " << endl ) : ( cout << " No " << endl ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int convert ( int m , int n ) { if ( m == n ) return 0 ;
if ( m > n ) return m - n ;
if ( m <= 0 && n > 0 ) return -1 ;
if ( n % 2 == 1 )
return 1 + convert ( m , n + 1 ) ;
else
return 1 + convert ( m , n / 2 ) ; }
int main ( ) { int m = 3 , n = 11 ; cout << " Minimum ▁ number ▁ of ▁ operations ▁ : ▁ " << convert ( m , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 10000 ; int prodDig [ MAX ] ;
int getDigitProduct ( int x ) {
if ( x < 10 ) return x ;
if ( prodDig [ x ] != 0 ) return prodDig [ x ] ;
int prod = ( x % 10 ) * getDigitProduct ( x / 10 ) ; return ( prodDig [ x ] = prod ) ; }
void findSeed ( int n ) {
vector < int > res ; for ( int i = 1 ; i <= n / 2 ; i ++ ) if ( i * getDigitProduct ( i ) == n ) res . push_back ( i ) ;
if ( res . size ( ) == 0 ) { cout << " NO ▁ seed ▁ exists STRNEWLINE " ; return ; }
for ( int i = 0 ; i < res . size ( ) ; i ++ ) cout << res [ i ] << " ▁ " ; }
int main ( ) { long long int n = 138 ; findSeed ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxPrimefactorNum ( int N ) { int arr [ N + 5 ] ; memset ( arr , 0 , sizeof ( arr ) ) ;
for ( int i = 2 ; i * i <= N ; i ++ ) { if ( ! arr [ i ] ) for ( int j = 2 * i ; j <= N ; j += i ) arr [ j ] ++ ; arr [ i ] = 1 ; } int maxval = 0 , maxint = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( arr [ i ] > maxval ) { maxval = arr [ i ] ; maxint = i ; } } return maxint ; }
int main ( ) { int N = 40 ; cout << maxPrimefactorNum ( N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long int SubArraySum ( int arr [ ] , int n ) { long int result = 0 ;
for ( int i = 0 ; i < n ; i ++ ) result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) ;
return result ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Sum ▁ of ▁ SubArray ▁ : ▁ " << SubArraySum ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int highestPowerof2 ( int n ) { int res = 0 ; for ( int i = n ; i >= 1 ; i -- ) {
if ( ( i & ( i - 1 ) ) == 0 ) { res = i ; break ; } } return res ; }
int main ( ) { int n = 10 ; cout << highestPowerof2 ( n ) ; return 0 ; }
#include <iostream> NEW_LINE #include <cmath> NEW_LINE using namespace std ;
void findPairs ( int n ) {
int cubeRoot = pow ( n , 1.0 / 3.0 ) ;
int cube [ cubeRoot + 1 ] ;
for ( int i = 1 ; i <= cubeRoot ; i ++ ) cube [ i ] = i * i * i ;
int l = 1 ; int r = cubeRoot ; while ( l < r ) { if ( cube [ l ] + cube [ r ] < n ) l ++ ; else if ( cube [ l ] + cube [ r ] > n ) r -- ; else { cout << " ( " << l << " , ▁ " << r << " ) " << endl ; l ++ ; r -- ; } } }
int main ( ) { int n = 20683 ; findPairs ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findPairs ( int n ) {
int cubeRoot = pow ( n , 1.0 / 3.0 ) ;
unordered_map < int , pair < int , int > > s ;
for ( int x = 1 ; x < cubeRoot ; x ++ ) { for ( int y = x + 1 ; y <= cubeRoot ; y ++ ) {
int sum = x * x * x + y * y * y ;
if ( sum != n ) continue ;
if ( s . find ( sum ) != s . end ( ) ) { cout << " ( " << s [ sum ] . first << " , ▁ " << s [ sum ] . second << " ) ▁ and ▁ ( " << x << " , ▁ " << y << " ) " << endl ; } else
s [ sum ] = make_pair ( x , y ) ; } } }
int main ( ) { int n = 13832 ; findPairs ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int gcd ( int a , int b ) { while ( b != 0 ) { int t = b ; b = a % b ; a = t ; } return a ; }
int findMinDiff ( int a , int b , int x , int y ) {
int g = gcd ( a , b ) ;
int diff = abs ( x - y ) % g ; return min ( diff , g - diff ) ; }
int main ( ) { int a = 20 , b = 52 , x = 5 , y = 7 ; cout << findMinDiff ( a , b , x , y ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printDivisors ( int n ) {
vector < int > v ; for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) printf ( " % d ▁ " , i ) ; else { printf ( " % d ▁ " , i ) ;
v . push_back ( n / i ) ; } } }
for ( int i = v . size ( ) - 1 ; i >= 0 ; i -- ) printf ( " % d ▁ " , v [ i ] ) ; }
int main ( ) { printf ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ n " ) ; printDivisors ( 100 ) ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void printDivisors ( int n ) { int i ; for ( i = 1 ; i * i < n ; i ++ ) { if ( n % i == 0 ) cout << i << " ▁ " ; } if ( i - ( n / i ) == 1 ) { i -- ; } for ( ; i >= 1 ; i -- ) { if ( n % i == 0 ) cout << n / i << " ▁ " ; } }
int main ( ) { cout << " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ; printDivisors ( 100 ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void printDivisors ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) cout << " ▁ " << i ; }
int main ( ) { cout << " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ; printDivisors ( 100 ) ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void printDivisors ( int n ) {
for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) cout << " ▁ " << i ;
cout << " ▁ " << i << " ▁ " << n / i ; } } }
int main ( ) { cout << " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ; printDivisors ( 100 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int SieveOfAtkin ( int limit ) {
if ( limit > 2 ) cout << 2 << " ▁ " ; if ( limit > 3 ) cout << 3 << " ▁ " ;
bool sieve [ limit ] ; for ( int i = 0 ; i < limit ; i ++ ) sieve [ i ] = false ;
for ( int x = 1 ; x * x < limit ; x ++ ) { for ( int y = 1 ; y * y < limit ; y ++ ) {
int n = ( 4 * x * x ) + ( y * y ) ; if ( n <= limit && ( n % 12 == 1 n % 12 == 5 ) ) sieve [ n ] ^= true ; n = ( 3 * x * x ) + ( y * y ) ; if ( n <= limit && n % 12 == 7 ) sieve [ n ] ^= true ; n = ( 3 * x * x ) - ( y * y ) ; if ( x > y && n <= limit && n % 12 == 11 ) sieve [ n ] ^= true ; } }
for ( int r = 5 ; r * r < limit ; r ++ ) { if ( sieve [ r ] ) { for ( int i = r * r ; i < limit ; i += r * r ) sieve [ i ] = false ; } }
for ( int a = 5 ; a < limit ; a ++ ) if ( sieve [ a ] ) cout << a << " ▁ " ; }
int main ( void ) { int limit = 20 ; SieveOfAtkin ( limit ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isInside ( int circle_x , int circle_y , int rad , int x , int y ) {
if ( ( x - circle_x ) * ( x - circle_x ) + ( y - circle_y ) * ( y - circle_y ) <= rad * rad ) return true ; else return false ; }
int main ( ) { int x = 1 , y = 1 ; int circle_x = 0 , circle_y = 1 , rad = 2 ; isInside ( circle_x , circle_y , rad , x , y ) ? cout << " Inside " : cout << " Outside " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int eval ( int a , char op , int b ) { if ( op == ' + ' ) return a + b ; if ( op == ' - ' ) return a - b ; if ( op == ' * ' ) return a * b ; }
vector < int > evaluateAll ( string expr , int low , int high ) {
vector < int > res ;
if ( low == high ) { res . push_back ( expr [ low ] - '0' ) ; return res ; }
if ( low == ( high - 2 ) ) { int num = eval ( expr [ low ] - '0' , expr [ low + 1 ] , expr [ low + 2 ] - '0' ) ; res . push_back ( num ) ; return res ; }
for ( int i = low + 1 ; i <= high ; i += 2 ) {
vector < int > l = evaluateAll ( expr , low , i - 1 ) ;
vector < int > r = evaluateAll ( expr , i + 1 , high ) ;
for ( int s1 = 0 ; s1 < l . size ( ) ; s1 ++ ) {
for ( int s2 = 0 ; s2 < r . size ( ) ; s2 ++ ) {
int val = eval ( l [ s1 ] , expr [ i ] , r [ s2 ] ) ; res . push_back ( val ) ; } } } return res ; }
int main ( ) { string expr = "1*2 + 3*4" ; int len = expr . length ( ) ; vector < int > ans = evaluateAll ( expr , 0 , len - 1 ) ; for ( int i = 0 ; i < ans . size ( ) ; i ++ ) cout << ans [ i ] << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isLucky ( int n ) {
bool arr [ 10 ] ; for ( int i = 0 ; i < 10 ; i ++ ) arr [ i ] = false ;
while ( n > 0 ) {
int digit = n % 10 ;
if ( arr [ digit ] ) return false ;
arr [ digit ] = true ;
n = n / 10 ; } return true ; }
int main ( ) { int arr [ ] = { 1291 , 897 , 4566 , 1232 , 80 , 700 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; for ( int i = 0 ; i < n ; i ++ ) isLucky ( arr [ i ] ) ? cout << arr [ i ] << " ▁ is ▁ Lucky ▁ STRNEWLINE " : cout << arr [ i ] << " ▁ is ▁ not ▁ Lucky ▁ STRNEWLINE " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; void printSquares ( int n ) {
int square = 0 , odd = 1 ;
for ( int x = 0 ; x < n ; x ++ ) {
cout << square << " ▁ " ;
square = square + odd ; odd = odd + 2 ; } }
int main ( ) { int n = 5 ; printSquares ( n ) ; }
int reversDigits ( int num ) { static int rev_num = 0 ; static int base_pos = 1 ; if ( num > 0 ) { reversDigits ( num / 10 ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
int main ( ) { int num = 4562 ; cout << " Reverse ▁ of ▁ no . ▁ is ▁ " << reversDigits ( num ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int RecursiveFunction ( vector < int > ref , int bit ) {
if ( ref . size ( ) == 0 bit < 0 ) return 0 ; vector < int > curr_on , curr_off ; for ( int i = 0 ; i < ref . size ( ) ; i ++ ) {
if ( ( ( ref [ i ] >> bit ) & 1 ) == 0 ) curr_off . push_back ( ref [ i ] ) ;
else curr_on . push_back ( ref [ i ] ) ; }
if ( curr_off . size ( ) == 0 ) return RecursiveFunction ( curr_on , bit - 1 ) ;
if ( curr_on . size ( ) == 0 ) return RecursiveFunction ( curr_off , bit - 1 ) ;
return min ( RecursiveFunction ( curr_off , bit - 1 ) , RecursiveFunction ( curr_on , bit - 1 ) ) + ( 1 << bit ) ; }
void PrintMinimum ( int a [ ] , int n ) { vector < int > v ;
for ( int i = 0 ; i < n ; i ++ ) v . push_back ( a [ i ] ) ;
cout << RecursiveFunction ( v , 30 ) << " STRNEWLINE " ; }
int main ( ) { int arr [ ] = { 3 , 2 , 1 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; PrintMinimum ( arr , size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int cntElements ( int arr [ ] , int n ) {
int cnt = 0 ;
for ( int i = 0 ; i < n - 2 ; i ++ ) {
if ( arr [ i ] == ( arr [ i + 1 ] ^ arr [ i + 2 ] ) ) { cnt ++ ; } } return cnt ; }
int main ( ) { int arr [ ] = { 4 , 2 , 1 , 3 , 7 , 8 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << cntElements ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int xor_triplet ( int arr [ ] , int n ) {
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i + 1 ; j < n ; j ++ ) {
for ( int k = j ; k < n ; k ++ ) { int xor1 = 0 , xor2 = 0 ;
for ( int x = i ; x < j ; x ++ ) { xor1 ^= arr [ x ] ; }
for ( int x = j ; x <= k ; x ++ ) { xor2 ^= arr [ x ] ; }
if ( xor1 == xor2 ) { ans ++ ; } } } } return ans ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << xor_triplet ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define lg  31
struct TrieNode {
TrieNode * children [ 2 ] ;
int sum_of_indexes ;
int number_of_indexes ;
TrieNode ( ) { this -> children [ 0 ] = nullptr ; this -> children [ 1 ] = nullptr ; this -> sum_of_indexes = 0 ; this -> number_of_indexes = 0 ; } } ;
void insert ( TrieNode * node , int num , int index ) {
for ( int bits = lg ; bits >= 0 ; bits -- ) {
int curr_bit = ( num >> bits ) & 1 ;
if ( node -> children [ curr_bit ] == nullptr ) { node -> children [ curr_bit ] = new TrieNode ( ) ; } node = node -> children [ curr_bit ] ; }
node -> sum_of_indexes += index ;
node -> number_of_indexes ++ ; }
int query ( TrieNode * node , int num , int index ) {
for ( int bits = lg ; bits >= 0 ; bits -- ) {
int curr_bit = ( num >> bits ) & 1 ;
if ( node -> children [ curr_bit ] == nullptr ) { return 0 ; } node = node -> children [ curr_bit ] ; }
int sz = node -> number_of_indexes ;
int sum = node -> sum_of_indexes ; int ans = ( sz * index ) - ( sum ) ; return ans ; }
int no_of_triplets ( int arr [ ] , int n ) {
int curr_xor = 0 ; int number_of_triplets = 0 ;
TrieNode * root = new TrieNode ( ) ; for ( int i = 0 ; i < n ; i ++ ) { int x = arr [ i ] ;
insert ( root , curr_xor , i ) ;
curr_xor ^= x ;
number_of_triplets += query ( root , curr_xor , i ) ; } return number_of_triplets ; }
int main ( ) {
int arr [ ] = { 5 , 2 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << no_of_triplets ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  100005 NEW_LINE int n , k ;
vector < int > al [ N ] ; long long Ideal_pair ; long long bit [ N ] ; bool root_node [ N ] ;
long long bit_q ( int i , int j ) { long long sum = 0ll ; while ( j > 0 ) { sum += bit [ j ] ; j -= ( j & ( j * -1 ) ) ; } i -- ; while ( i > 0 ) { sum -= bit [ i ] ; i -= ( i & ( i * -1 ) ) ; } return sum ; }
void bit_up ( int i , long long diff ) { while ( i <= n ) { bit [ i ] += diff ; i += i & - i ; } }
void dfs ( int node ) { Ideal_pair += bit_q ( max ( 1 , node - k ) , min ( n , node + k ) ) ; bit_up ( node , 1 ) ; for ( int i = 0 ; i < al [ node ] . size ( ) ; i ++ ) dfs ( al [ node ] [ i ] ) ; bit_up ( node , -1 ) ; }
void initialise ( ) { Ideal_pair = 0 ; for ( int i = 0 ; i <= n ; i ++ ) { root_node [ i ] = true ; bit [ i ] = 0LL ; } }
void Add_Edge ( int x , int y ) { al [ x ] . push_back ( y ) ; root_node [ y ] = false ; }
long long Idealpairs ( ) {
int r = -1 ; for ( int i = 1 ; i <= n ; i ++ ) if ( root_node [ i ] ) { r = i ; break ; } dfs ( r ) ; return Ideal_pair ; }
int main ( ) { n = 6 , k = 3 ; initialise ( ) ;
Add_Edge ( 1 , 2 ) ; Add_Edge ( 1 , 3 ) ; Add_Edge ( 3 , 4 ) ; Add_Edge ( 3 , 5 ) ; Add_Edge ( 3 , 6 ) ;
cout << Idealpairs ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printSubsets ( int n ) { for ( int i = n ; i > 0 ; i = ( i - 1 ) & n ) cout << i << " ▁ " ; cout << 0 ; }
int main ( ) { int n = 9 ; printSubsets ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isDivisibleby17 ( int n ) {
if ( n == 0 n == 17 ) return true ;
if ( n < 17 ) return false ;
return isDivisibleby17 ( ( int ) ( n >> 4 ) - ( int ) ( n & 15 ) ) ; }
int main ( ) { int n = 35 ; if ( isDivisibleby17 ( n ) ) cout << n << " ▁ is ▁ divisible ▁ by ▁ 17" ; else cout << n << " ▁ is ▁ not ▁ divisible ▁ by ▁ 17" ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long answer ( long long n ) {
long m = 2 ;
long long ans = 1 ; long long r = 1 ;
while ( r < n ) {
r = ( int ) ( pow ( 2 , m ) - 1 ) * ( pow ( 2 , m - 1 ) ) ;
if ( r < n ) ans = r ;
m ++ ; } return ans ; }
int main ( ) { long long n = 7 ; cout << answer ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int setBitNumber ( int n ) { if ( n == 0 ) return 0 ; int msb = 0 ; n = n / 2 ; while ( n != 0 ) { n = n / 2 ; msb ++ ; } return ( 1 << msb ) ; }
int main ( ) { int n = 0 ; cout << setBitNumber ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int setBitNumber ( int n ) {
n |= n >> 1 ;
n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ;
n = n + 1 ;
return ( n >> 1 ) ; }
int main ( ) { int n = 273 ; cout << setBitNumber ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countTrailingZero ( int x ) { int count = 0 ; while ( ( x & 1 ) == 0 ) { x = x >> 1 ; count ++ ; } return count ; }
int main ( ) { cout << countTrailingZero ( 11 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countTrailingZero ( int x ) {
static const int lookup [ ] = { 32 , 0 , 1 , 26 , 2 , 23 , 27 , 0 , 3 , 16 , 24 , 30 , 28 , 11 , 0 , 13 , 4 , 7 , 17 , 0 , 25 , 22 , 31 , 15 , 29 , 10 , 12 , 6 , 0 , 21 , 14 , 9 , 5 , 20 , 8 , 19 , 18 } ;
return lookup [ ( - x & x ) % 37 ] ; }
int main ( ) { cout << countTrailingZero ( 48 ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int multiplyBySevenByEight ( int n ) {
return ( n - ( n >> 3 ) ) ; }
int main ( ) { int n = 9 ; cout << multiplyBySevenByEight ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
static int search ( vector < int > list , int num ) { int low = 0 , high = list . size ( ) - 1 ;
int ans = -1 ; while ( low <= high ) {
int mid = low + ( high - low ) / 2 ;
if ( list [ mid ] <= num ) {
ans = mid ;
low = mid + 1 ; } else
high = mid - 1 ; }
return ans ; }
bool isPalindrome ( int n ) { int rev = 0 ; int temp = n ;
while ( n > 0 ) { rev = rev * 10 + n % 10 ; n /= 10 ; }
return rev == temp ; }
int countNumbers ( int L , int R , int K ) {
vector < int > list ;
for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) ) {
list . push_back ( i ) ; } }
int count = 0 ;
for ( int i = 0 ; i < list . size ( ) ; i ++ ) {
int right_index = search ( list , list [ i ] + K - 1 ) ;
if ( right_index != -1 ) count = max ( count , right_index - i + 1 ) ; }
return count ; }
int main ( ) { int L = 98 , R = 112 ; int K = 13 ; cout << countNumbers ( L , R , K ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > findPrevious ( vector < int > a , int n ) { vector < int > ps ( n ) ;
ps [ 0 ] = -1 ;
stack < int > Stack ;
Stack . push ( 0 ) ; for ( int i = 1 ; i < n ; i ++ ) {
while ( Stack . size ( ) > 0 && a [ Stack . top ( ) ] >= a [ i ] ) Stack . pop ( ) ;
ps [ i ] = Stack . size ( ) > 0 ? Stack . top ( ) : -1 ;
Stack . push ( i ) ; }
return ps ; }
vector < int > findNext ( vector < int > a , int n ) { vector < int > ns ( n ) ; ns [ n - 1 ] = n ;
stack < int > Stack ; Stack . push ( n - 1 ) ;
for ( int i = n - 2 ; i >= 0 ; i -- ) {
while ( Stack . size ( ) > 0 && a [ Stack . top ( ) ] >= a [ i ] ) Stack . pop ( ) ;
ns [ i ] = Stack . size ( ) > 0 ? Stack . top ( ) : n ;
Stack . push ( i ) ; }
return ns ; }
int findMaximumSum ( vector < int > a , int n ) {
vector < int > prev_smaller = findPrevious ( a , n ) ;
vector < int > next_smaller = findNext ( a , n ) ; int max_value = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
max_value = max ( max_value , a [ i ] * ( next_smaller [ i ] - prev_smaller [ i ] - 1 ) ) ; }
return max_value ; }
int main ( ) { int n = 3 ; vector < int > a { 80 , 48 , 82 } ; cout << findMaximumSum ( a , n ) ; return 0 ; }
#include <iostream> NEW_LINE #include <cstring> NEW_LINE #define MAX  256 NEW_LINE using namespace std ; bool compare ( char arr1 [ ] , char arr2 [ ] ) { for ( int i = 0 ; i < MAX ; i ++ ) if ( arr1 [ i ] != arr2 [ i ] ) return false ; return true ; }
bool search ( char * pat , char * txt ) { int M = strlen ( pat ) , N = strlen ( txt ) ;
char countP [ MAX ] = { 0 } , countTW [ MAX ] = { 0 } ; for ( int i = 0 ; i < M ; i ++ ) { ( countP [ pat [ i ] ] ) ++ ; ( countTW [ txt [ i ] ] ) ++ ; }
for ( int i = M ; i < N ; i ++ ) {
if ( compare ( countP , countTW ) ) return true ;
( countTW [ txt [ i ] ] ) ++ ;
countTW [ txt [ i - M ] ] -- ; }
if ( compare ( countP , countTW ) ) return true ; return false ; }
int main ( ) { char txt [ ] = " BACDGABCDA " ; char pat [ ] = " ABCD " ; if ( search ( pat , txt ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float getMaxMedian ( int arr [ ] , int n , int k ) { int size = n + k ;
sort ( arr , arr + n ) ;
if ( size % 2 == 0 ) { float median = ( float ) ( arr [ ( size / 2 ) - 1 ] + arr [ size / 2 ] ) / 2 ; return median ; }
float median = arr [ size / 2 ] ; return median ; }
int main ( ) { int arr [ ] = { 3 , 2 , 3 , 4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 2 ; cout << getMaxMedian ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printSorted ( int a , int b , int c ) {
int get_max = max ( a , max ( b , c ) ) ;
int get_min = - max ( - a , max ( - b , - c ) ) ; int get_mid = ( a + b + c ) - ( get_max + get_min ) ; cout << get_min << " ▁ " << get_mid << " ▁ " << get_max ; }
int main ( ) { int a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int binarySearch ( int a [ ] , int item , int low , int high ) { while ( low <= high ) { int mid = low + ( high - low ) / 2 ; if ( item == a [ mid ] ) return mid + 1 ; else if ( item > a [ mid ] ) low = mid + 1 ; else high = mid - 1 ; } return low ; }
void insertionSort ( int a [ ] , int n ) { int i , loc , j , k , selected ; for ( i = 1 ; i < n ; ++ i ) { j = i - 1 ; selected = a [ i ] ;
loc = binarySearch ( a , selected , 0 , j ) ;
while ( j >= loc ) { a [ j + 1 ] = a [ j ] ; j -- ; } a [ j + 1 ] = selected ; } }
int main ( ) { int a [ ] = { 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) , i ; insertionSort ( a , n ) ; cout << " Sorted ▁ array : ▁ STRNEWLINE " ; for ( i = 0 ; i < n ; i ++ ) cout << " ▁ " << a [ i ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void insertionSort ( int arr [ ] , int n ) { int i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
void printArray ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; insertionSort ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
int validPermutations ( string str ) { unordered_map < char , int > m ;
int count = str . length ( ) , ans = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { m [ str [ i ] ] ++ ; } for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
ans += count - m [ str [ i ] ] ;
m [ str [ i ] ] -- ; count -- ; }
return ans + 1 ; }
int main ( ) { string str = " sstt " ; cout << validPermutations ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countPaths ( int n , int m ) { int dp [ n + 1 ] [ m + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) dp [ i ] [ 0 ] = 1 ; for ( int i = 0 ; i <= m ; i ++ ) dp [ 0 ] [ i ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) for ( int j = 1 ; j <= m ; j ++ ) dp [ i ] [ j ] = dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] ; return dp [ n ] [ m ] ; }
int main ( ) { int n = 3 , m = 2 ; cout << " ▁ Number ▁ of ▁ Paths ▁ " << countPaths ( n , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int count ( int S [ ] , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
int main ( ) { int i , j ; int arr [ ] = { 1 , 2 , 3 } ; int m = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " ▁ " << count ( arr , m , 4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int coinchange ( vector < int > & a , int v , int n , vector < vector < int > > & dp ) { if ( v == 0 ) return dp [ n ] [ v ] = 1 ; if ( n == 0 ) return 0 ; if ( dp [ n ] [ v ] != -1 ) return dp [ n ] [ v ] ; if ( a [ n - 1 ] <= v ) { return dp [ n ] [ v ] = coinchange ( a , v - a [ n - 1 ] , n , dp ) + coinchange ( a , v , n - 1 , dp ) ; }
return dp [ n ] [ v ] = coinchange ( a , v , n - 1 , dp ) ; } int32_t main ( ) { int tc = 1 ;
while ( tc -- ) { int n , v ; n = 3 , v = 4 ; vector < int > a = { 1 , 2 , 3 } ; vector < vector < int > > dp ( n + 1 , vector < int > ( v + 1 , -1 ) ) ; int res = coinchange ( a , v , n , dp ) ; cout << res << endl ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool equalIgnoreCase ( string str1 , string str2 ) { int i = 0 ;
transform ( str1 . begin ( ) , str1 . end ( ) , str1 . begin ( ) , :: toupper ) ; transform ( str2 . begin ( ) , str2 . end ( ) , str2 . begin ( ) , :: toupper ) ;
int x = str1 . compare ( str2 ) ;
if ( x != 0 ) return false ; else return true ; }
void equalIgnoreCaseUtil ( string str1 , string str2 ) { bool res = equalIgnoreCase ( str1 , str2 ) ; if ( res == true ) cout << " Same " << endl ; else cout << " Not ▁ Same " << endl ; }
int main ( ) { string str1 , str2 ; str1 = " Geeks " ; str2 = " geeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; str1 = " Geek " ; str2 = " geeksforgeeks " ; equalIgnoreCaseUtil ( str1 , str2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string replaceConsonants ( string str ) {
string res = " " ; int i = 0 , count = 0 ;
while ( i < str . length ( ) ) {
if ( str [ i ] != ' a ' && str [ i ] != ' e ' && str [ i ] != ' i ' && str [ i ] != ' o ' && str [ i ] != ' u ' ) { i ++ ; count ++ ; } else {
if ( count > 0 ) res += to_string ( count ) ;
res += str [ i ] ; i ++ ; count = 0 ; } }
if ( count > 0 ) res += to_string ( count ) ;
return res ; }
int main ( ) { string str = " abcdeiop " ; cout << replaceConsonants ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
string encryptString ( string s , int n , int k ) { int countVowels = 0 ; int countConsonants = 0 ; string ans = " " ;
for ( int l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( int r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s [ r ] ) == true ) countVowels ++ ; else countConsonants ++ ; }
ans += to_string ( countVowels * countConsonants ) ; } return ans ; }
int main ( ) { string s = " hello " ; int n = s . length ( ) ; int k = 2 ; cout << encryptString ( s , n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; string processWords ( char * input ) {
char * p ; vector < string > s ; p = strtok ( input , " ▁ " ) ; while ( p != NULL ) { s . push_back ( p ) ; p = strtok ( NULL , " ▁ " ) ; } string charBuffer ; for ( string values : s )
charBuffer += values [ 0 ] ; return charBuffer ; }
int main ( ) { char input [ ] = " geeks ▁ for ▁ geeks " ; cout << processWords ( input ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void generateAllStringsUtil ( int K , char str [ ] , int n ) {
if ( n == K ) {
str [ n ] = ' \0' ; cout << str << " ▁ " ; return ; }
if ( str [ n - 1 ] == '1' ) { str [ n ] = '0' ; generateAllStringsUtil ( K , str , n + 1 ) ; }
if ( str [ n - 1 ] == '0' ) { str [ n ] = '0' ; generateAllStringsUtil ( K , str , n + 1 ) ; str [ n ] = '1' ; generateAllStringsUtil ( K , str , n + 1 ) ; } }
void generateAllStrings ( int K ) {
if ( K <= 0 ) return ;
char str [ K ] ;
str [ 0 ] = '0' ; generateAllStringsUtil ( K , str , 1 ) ;
str [ 0 ] = '1' ; generateAllStringsUtil ( K , str , 1 ) ; }
int main ( ) { int K = 3 ; generateAllStrings ( K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float findVolume ( float a ) {
if ( a < 0 ) return -1 ;
float r = a / 2 ;
float h = a ;
float V = 3.14 * pow ( r , 2 ) * h ; return V ; }
int main ( ) { float a = 5 ; cout << findVolume ( a ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float volumeTriangular ( int a , int b , int h ) { float vol = ( 0.1666 ) * a * b * h ; return vol ; }
float volumeSquare ( int b , int h ) { float vol = ( 0.33 ) * b * b * h ; return vol ; }
float volumePentagonal ( int a , int b , int h ) { float vol = ( 0.83 ) * a * b * h ; return vol ; }
float volumeHexagonal ( int a , int b , int h ) { float vol = a * b * h ; return vol ; }
int main ( ) { int b = 4 , h = 9 , a = 4 ; cout << " Volume ▁ of ▁ triangular " << " ▁ base ▁ pyramid ▁ is ▁ " << volumeTriangular ( a , b , h ) << endl ; cout << " Volume ▁ of ▁ square ▁ " << " ▁ base ▁ pyramid ▁ is ▁ " << volumeSquare ( b , h ) << endl ; cout << " Volume ▁ of ▁ pentagonal " << " ▁ base ▁ pyramid ▁ is ▁ " << volumePentagonal ( a , b , h ) << endl ; cout << " Volume ▁ of ▁ Hexagonal " << " ▁ base ▁ pyramid ▁ is ▁ " << volumeHexagonal ( a , b , h ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double Area ( int b1 , int b2 , int h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
int main ( ) { int base1 = 8 , base2 = 10 , height = 6 ; double area = Area ( base1 , base2 , height ) ; cout << " Area ▁ is : ▁ " << area ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int numberOfDiagonals ( int n ) { return n * ( n - 3 ) / 2 ; }
int main ( ) { int n = 5 ; cout << n << " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ; cout << numberOfDiagonals ( n ) << " ▁ diagonals " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void maximumArea ( int l , int b , int x , int y ) {
int left , right , above , below ; left = x * b ; right = ( l - x - 1 ) * b ; above = l * y ; below = ( b - y - 1 ) * l ;
cout << max ( max ( left , right ) , max ( above , below ) ) ; }
int main ( ) { int L = 8 , B = 8 ; int X = 0 , Y = 0 ;
maximumArea ( l , b , x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int delCost ( string s , int cost [ ] ) {
int ans = 0 ;
map < char , int > forMax ;
map < char , int > forTot ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
if ( ! forMax [ s [ i ] ] ) { forMax [ s [ i ] ] = cost [ i ] ; } else {
forMax [ s [ i ] ] = max ( cost [ i ] , forMax [ s [ i ] ] ) ; }
if ( ! forTot [ s [ i ] ] ) { forTot [ s [ i ] ] = cost [ i ] ; } else {
forTot [ s [ i ] ] = forTot [ s [ i ] ] + cost [ i ] ; } }
for ( auto i : forMax ) {
ans += forTot [ i . first ] - i . second ; }
return ans ; }
int main ( ) {
string s = " AAABBB " ;
int cost [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ;
cout << ( delCost ( s , cost ) ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define MAX  10000 NEW_LINE vector < vector < int > > divisors ( MAX + 1 ) ;
void computeDivisors ( ) { for ( int i = 1 ; i <= MAX ; i ++ ) { for ( int j = i ; j <= MAX ; j += i ) {
divisors [ j ] . push_back ( i ) ; } } }
int getClosest ( int val1 , int val2 , int target ) { if ( target - val1 >= val2 - target ) return val2 ; else return val1 ; }
int findClosest ( vector < int > & arr , int n , int target ) {
if ( target <= arr [ 0 ] ) return arr [ 0 ] ; if ( target >= arr [ n - 1 ] ) return arr [ n - 1 ] ;
int i = 0 , j = n , mid = 0 ; while ( i < j ) { mid = ( i + j ) / 2 ; if ( arr [ mid ] == target ) return arr [ mid ] ;
if ( target < arr [ mid ] ) {
if ( mid > 0 && target > arr [ mid - 1 ] ) return getClosest ( arr [ mid - 1 ] , arr [ mid ] , target ) ;
j = mid ; }
else { if ( mid < n - 1 && target < arr [ mid + 1 ] ) return getClosest ( arr [ mid ] , arr [ mid + 1 ] , target ) ;
i = mid + 1 ; } }
return arr [ mid ] ; }
void printClosest ( int N , int X ) {
computeDivisors ( ) ;
int ans = findClosest ( divisors [ N ] , divisors [ N ] . size ( ) , X ) ;
cout << ans ; }
int main ( ) {
int N = 16 , X = 5 ;
printClosest ( N , X ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxMatch ( int A [ ] , int B [ ] , int M , int N ) {
map < int , int > Aindex ;
map < int , int > diff ;
for ( int i = 0 ; i < M ; i ++ ) { Aindex [ A [ i ] ] = i ; }
for ( int i = 0 ; i < N ; i ++ ) {
if ( i - Aindex [ B [ i ] ] < 0 ) { diff [ M + i - Aindex [ B [ i ] ] ] += 1 ; }
else { diff [ i - Aindex [ B [ i ] ] ] += 1 ; } }
int max = 0 ; for ( auto ele = diff . begin ( ) ; ele != diff . end ( ) ; ele ++ ) { if ( ele -> second > max ) { max = ele -> second ; } } return max ; }
int main ( ) { int A [ ] = { 5 , 3 , 7 , 9 , 8 } ; int B [ ] = { 8 , 7 , 3 , 5 , 9 } ; int M = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int N = sizeof ( B ) / sizeof ( B [ 0 ] ) ;
cout << maxMatch ( A , B , M , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  9
bool isinRange ( int board [ ] [ N ] ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( board [ i ] [ j ] <= 0 board [ i ] [ j ] > 9 ) { return false ; } } } return true ; }
bool isValidSudoku ( int board [ ] [ N ] ) {
if ( isinRange ( board ) == false ) { return false ; }
bool unique [ N + 1 ] ;
for ( int i = 0 ; i < N ; i ++ ) {
memset ( unique , false , sizeof ( unique ) ) ;
for ( int j = 0 ; j < N ; j ++ ) {
int Z = board [ i ] [ j ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( int i = 0 ; i < N ; i ++ ) {
memset ( unique , false , sizeof ( unique ) ) ;
for ( int j = 0 ; j < N ; j ++ ) {
int Z = board [ j ] [ i ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } }
for ( int i = 0 ; i < N - 2 ; i += 3 ) {
for ( int j = 0 ; j < N - 2 ; j += 3 ) {
memset ( unique , false , sizeof ( unique ) ) ;
for ( int k = 0 ; k < 3 ; k ++ ) { for ( int l = 0 ; l < 3 ; l ++ ) {
int X = i + k ;
int Y = j + l ;
int Z = board [ X ] [ Y ] ;
if ( unique [ Z ] ) { return false ; } unique [ Z ] = true ; } } } }
return true ; }
int main ( ) { int board [ N ] [ N ] = { { 7 , 9 , 2 , 1 , 5 , 4 , 3 , 8 , 6 } , { 6 , 4 , 3 , 8 , 2 , 7 , 1 , 5 , 9 } , { 8 , 5 , 1 , 3 , 9 , 6 , 7 , 2 , 4 } , { 2 , 6 , 5 , 9 , 7 , 3 , 8 , 4 , 1 } , { 4 , 8 , 9 , 5 , 6 , 1 , 2 , 7 , 3 } , { 3 , 1 , 7 , 4 , 8 , 2 , 9 , 6 , 5 } , { 1 , 3 , 6 , 7 , 4 , 8 , 5 , 9 , 2 } , { 9 , 7 , 4 , 2 , 1 , 5 , 6 , 3 , 8 } , { 5 , 2 , 8 , 6 , 3 , 9 , 4 , 1 , 7 } } ; if ( isValidSudoku ( board ) ) { cout << " Valid " ; } else { cout << " Not ▁ Valid " ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool palindrome ( vector < int > a , int i , int j ) { while ( i < j ) {
if ( a [ i ] != a [ j ] ) return false ;
i ++ ; j -- ; }
return true ; }
int findSubArray ( vector < int > arr , int k ) { int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
for ( int i = 0 ; i <= n - k ; i ++ ) { if ( palindrome ( arr , i , i + k - 1 ) ) return i ; }
return -1 ; }
int main ( ) { vector < int > arr = { 2 , 3 , 5 , 1 , 3 } ; int k = 4 ; int ans = findSubArray ( arr , k ) ; if ( ans == -1 ) cout << -1 << " STRNEWLINE " ; else { for ( int i = ans ; i < ans + k ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << " STRNEWLINE " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isCrossed ( string path ) { if ( path . size ( ) == 0 ) return false ;
bool ans = false ;
set < pair < int , int > > set ;
int x = 0 , y = 0 ; set . insert ( { x , y } ) ;
for ( int i = 0 ; i < path . size ( ) ; i ++ ) {
if ( path [ i ] == ' N ' ) set . insert ( { x , y ++ } ) ; if ( path [ i ] == ' S ' ) set . insert ( { x , y -- } ) ; if ( path [ i ] == ' E ' ) set . insert ( { x ++ , y } ) ; if ( path [ i ] == ' W ' ) set . insert ( { x -- , y } ) ;
if ( set . find ( { x , y } ) != set . end ( ) ) { ans = true ; break ; } }
if ( ans ) cout << " Crossed " ; else cout << " Not ▁ Crossed " ; }
int main ( ) {
string path = " NESW " ;
isCrossed ( path ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxWidth ( int N , int M , vector < int > cost , vector < vector < int > > s ) {
vector < int > adj [ N ] ; for ( int i = 0 ; i < M ; i ++ ) { adj [ s [ i ] [ 0 ] ] . push_back ( s [ i ] [ 1 ] ) ; }
int result = 0 ;
queue < int > q ;
q . push ( 0 ) ;
while ( ! q . empty ( ) ) {
int count = q . size ( ) ;
result = max ( count , result ) ;
while ( count -- ) {
int temp = q . front ( ) ; q . pop ( ) ;
for ( int i = 0 ; i < adj [ temp ] . size ( ) ; i ++ ) { q . push ( adj [ temp ] [ i ] ) ; } } }
return result ; }
int main ( ) { int N = 11 , M = 10 ; vector < vector < int > > edges ; edges . push_back ( { 0 , 1 } ) ; edges . push_back ( { 0 , 2 } ) ; edges . push_back ( { 0 , 3 } ) ; edges . push_back ( { 1 , 4 } ) ; edges . push_back ( { 1 , 5 } ) ; edges . push_back ( { 3 , 6 } ) ; edges . push_back ( { 4 , 7 } ) ; edges . push_back ( { 6 , 10 } ) ; edges . push_back ( { 6 , 8 } ) ; edges . push_back ( { 6 , 9 } ) ; vector < int > cost = { 1 , 2 , -1 , 3 , 4 , 5 , 8 , 2 , 6 , 12 , 7 } ;
cout << maxWidth ( N , M , cost , edges ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  10000000
bool isPrime [ MAX ] ;
vector < int > primes ;
void SieveOfEratosthenes ( ) { memset ( isPrime , true , sizeof ( isPrime ) ) ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( isPrime [ p ] == true ) {
for ( int i = p * p ; i <= MAX ; i += p ) isPrime [ i ] = false ; } }
for ( int p = 2 ; p <= MAX ; p ++ ) if ( isPrime [ p ] ) primes . push_back ( p ) ; }
int prime_search ( vector < int > primes , int diff ) {
int low = 0 ; int high = primes . size ( ) - 1 ; int res ; while ( low <= high ) { int mid = ( low + high ) / 2 ;
if ( primes [ mid ] == diff ) {
return primes [ mid ] ; }
else if ( primes [ mid ] < diff ) {
low = mid + 1 ; }
else { res = primes [ mid ] ;
high = mid - 1 ; } }
return res ; }
int minCost ( int arr [ ] , int n ) {
SieveOfEratosthenes ( ) ;
int res = 0 ;
for ( int i = 1 ; i < n ; i ++ ) {
if ( arr [ i ] < arr [ i - 1 ] ) { int diff = arr [ i - 1 ] - arr [ i ] ;
int closest_prime = prime_search ( primes , diff ) ;
res += closest_prime ;
arr [ i ] += closest_prime ; } }
return res ; }
int main ( ) {
int arr [ ] = { 2 , 1 , 5 , 4 , 3 } ; int n = 5 ;
cout << minCost ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int count ( string s ) {
int cnt = 0 ;
for ( char c : s ) { cnt += c == '0' ? 1 : 0 ; }
if ( cnt % 3 != 0 ) return 0 ; int res = 0 , k = cnt / 3 , sum = 0 ;
map < int , int > mp ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) {
sum += s [ i ] == '0' ? 1 : 0 ;
if ( sum == 2 * k && mp . find ( k ) != mp . end ( ) && i < s . length ( ) - 1 && i > 0 ) { res += mp [ k ] ; }
mp [ sum ] ++ ; }
return res ; }
int main ( ) {
string str = "01010" ;
cout << count ( str ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int splitstring ( string s ) { int n = s . length ( ) ;
int zeros = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( s [ i ] == '0' ) zeros ++ ;
if ( zeros % 3 != 0 ) return 0 ;
if ( zeros == 0 ) return ( ( n - 1 ) * ( n - 2 ) ) / 2 ;
int zerosInEachSubstring = zeros / 3 ;
int waysOfFirstCut = 0 , waysOfSecondCut = 0 ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == '0' ) count ++ ;
if ( count == zerosInEachSubstring ) waysOfFirstCut ++ ;
else if ( count == 2 * zerosInEachSubstring ) waysOfSecondCut ++ ; }
return waysOfFirstCut * waysOfSecondCut ; }
int main ( ) { string s = "01010" ;
cout << " The ▁ number ▁ of ▁ ways ▁ to ▁ split ▁ is ▁ " << splitstring ( s ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool canTransform ( string str1 , string str2 ) { string s1 = " " ; string s2 = " " ;
for ( char c : str1 ) { if ( c != ' C ' ) { s1 += c ; } } for ( char c : str2 ) { if ( c != ' C ' ) { s2 += c ; } }
if ( s1 != s2 ) return false ; int i = 0 ; int j = 0 ; int n = str1 . length ( ) ;
while ( i < n and j < n ) { if ( str1 [ i ] == ' C ' ) { i ++ ; } else if ( str2 [ j ] == ' C ' ) { j ++ ; }
else { if ( ( str1 [ i ] == ' A ' and i < j ) or ( str1 [ i ] == ' B ' and i > j ) ) { return false ; } i ++ ; j ++ ; } } return true ; }
int main ( ) { string str1 = " BCCABCBCA " ; string str2 = " CBACCBBAC " ;
if ( canTransform ( str1 , str2 ) ) { cout << " Yes " ; } else { cout << " No " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxsubstringLength ( string S , int N ) { int arr [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) if ( S [ i ] == ' a ' S [ i ] == ' e ' S [ i ] == ' i ' S [ i ] == ' o ' S [ i ] == ' u ' ) arr [ i ] = 1 ; else arr [ i ] = -1 ;
int maxLen = 0 ;
int curr_sum = 0 ;
unordered_map < int , int > hash ;
for ( int i = 0 ; i < N ; i ++ ) { curr_sum += arr [ i ] ;
if ( curr_sum == 0 )
maxLen = max ( maxLen , i + 1 ) ;
if ( hash . find ( curr_sum ) != hash . end ( ) ) maxLen = max ( maxLen , i - hash [ curr_sum ] ) ;
else hash [ curr_sum ] = i ; }
return maxLen ; }
int main ( ) { string S = " geeksforgeeks " ; int n = sizeof ( S ) / sizeof ( S [ 0 ] ) ; cout << maxsubstringLength ( S , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int mat [ 1001 ] [ 1001 ] ; int r , c , x , y ;
int dx [ ] = { 0 , -1 , -1 , -1 , 0 , 1 , 1 , 1 } ; int dy [ ] = { 1 , 1 , 0 , -1 , -1 , -1 , 0 , 1 } ;
void FindMinimumDistance ( ) {
queue < pair < int , int > > q ;
q . push ( { x , y } ) ; mat [ x ] [ y ] = 0 ;
while ( ! q . empty ( ) ) {
x = q . front ( ) . first ; y = q . front ( ) . second ;
q . pop ( ) ; for ( int i = 0 ; i < 8 ; i ++ ) { int a = x + dx [ i ] ; int b = y + dy [ i ] ;
if ( a < 0 a > = r b >= c b < 0 ) continue ;
if ( mat [ a ] [ b ] == 0 ) {
mat [ a ] [ b ] = mat [ x ] [ y ] + 1 ;
q . push ( { a , b } ) ; } } } }
int main ( ) { r = 5 , c = 5 , x = 1 , y = 1 ; int t = x ; int l = y ; mat [ x ] [ y ] = 0 ; FindMinimumDistance ( ) ; mat [ t ] [ l ] = 0 ;
for ( int i = 0 ; i < r ; i ++ ) { for ( int j = 0 ; j < c ; j ++ ) { cout << mat [ i ] [ j ] << " ▁ " ; } cout << endl ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minOperations ( string S , int K ) {
int ans = 0 ;
for ( int i = 0 ; i < K ; i ++ ) {
int zero = 0 , one = 0 ;
for ( int j = i ; j < S . size ( ) ; j += K ) {
if ( S [ j ] == '0' ) zero ++ ;
else one ++ ; }
ans += min ( zero , one ) ; }
return ans ; }
int main ( ) { string S = "110100101" ; int K = 3 ; cout << minOperations ( S , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int missingElement ( int arr [ ] , int n ) {
int max_ele = arr [ 0 ] ;
int min_ele = arr [ 0 ] ;
int x = 0 ;
int d ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max_ele ) max_ele = arr [ i ] ; if ( arr [ i ] < min_ele ) min_ele = arr [ i ] ; }
d = ( max_ele - min_ele ) / n ;
for ( int i = 0 ; i < n ; i ++ ) { x = x ^ arr [ i ] ; }
for ( int i = 0 ; i <= n ; i ++ ) { x = x ^ ( min_ele + ( i * d ) ) ; }
return x ; }
int main ( ) {
int arr [ ] = { 12 , 3 , 6 , 15 , 18 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int element = missingElement ( arr , n ) ;
cout << element ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void Printksubstring ( string str , int n , int k ) {
int total = ( n * ( n + 1 ) ) / 2 ;
if ( k > total ) { printf ( " - 1 STRNEWLINE " ) ; return ; }
int substring [ n + 1 ] ; substring [ 0 ] = 0 ;
int temp = n ; for ( int i = 1 ; i <= n ; i ++ ) {
substring [ i ] = substring [ i - 1 ] + temp ; temp -- ; }
int l = 1 ; int h = n ; int start = 0 ; while ( l <= h ) { int m = ( l + h ) / 2 ; if ( substring [ m ] > k ) { start = m ; h = m - 1 ; } else if ( substring [ m ] < k ) l = m + 1 ; else { start = m ; break ; } }
int end = n - ( substring [ start ] - k ) ;
for ( int i = start - 1 ; i < end ; i ++ ) cout << str [ i ] ; }
int main ( ) { string str = " abc " ; int k = 4 ; int n = str . length ( ) ; Printksubstring ( str , n , k ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int LowerInsertionPoint ( int arr [ ] , int n , int X ) {
if ( X < arr [ 0 ] ) return 0 ; else if ( X > arr [ n - 1 ] ) return n ; int lowerPnt = 0 ; int i = 1 ; while ( i < n && arr [ i ] < X ) { lowerPnt = i ; i = i * 2 ; }
while ( lowerPnt < n && arr [ lowerPnt ] < X ) lowerPnt ++ ; return lowerPnt ; }
int main ( ) { int arr [ ] = { 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int X = 4 ; cout << LowerInsertionPoint ( arr , n , X ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getCount ( int M , int N ) { int count = 0 ;
if ( M == 1 ) return N ;
if ( N == 1 ) return M ; if ( N > M ) {
for ( int i = 1 ; i <= M ; i ++ ) { int numerator = N * i - N + M - i ; int denominator = M - 1 ;
if ( numerator % denominator == 0 ) { int j = numerator / denominator ;
if ( j >= 1 && j <= N ) count ++ ; } } } else {
for ( int j = 1 ; j <= N ; j ++ ) { int numerator = M * j - M + N - j ; int denominator = N - 1 ;
if ( numerator % denominator == 0 ) { int i = numerator / denominator ;
if ( i >= 1 && i <= M ) count ++ ; } } } return count ; }
int main ( ) { int M = 3 , N = 5 ; cout << getCount ( M , N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool swapElement ( int arr1 [ ] , int arr2 [ ] , int n ) {
int wrongIdx = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr1 [ i ] < arr1 [ i - 1 ] ) wrongIdx = i ; int maximum = INT_MIN ; int maxIdx = -1 ; bool res = false ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr2 [ i ] > maximum && arr2 [ i ] >= arr1 [ wrongIdx - 1 ] ) { if ( wrongIdx + 1 <= n - 1 && arr2 [ i ] <= arr1 [ wrongIdx + 1 ] ) { maximum = arr2 [ i ] ; maxIdx = i ; res = true ; } } }
if ( res ) swap ( arr1 [ wrongIdx ] , arr2 [ maxIdx ] ) ; return res ; }
void getSortedArray ( int arr1 [ ] , int arr2 [ ] , int n ) { if ( swapElement ( arr1 , arr2 , n ) ) for ( int i = 0 ; i < n ; i ++ ) cout << arr1 [ i ] << " ▁ " ; else cout << " Not ▁ Possible " << endl ; }
int main ( ) { int arr1 [ ] = { 1 , 3 , 7 , 4 , 10 } ; int arr2 [ ] = { 2 , 1 , 6 , 8 , 9 } ; int n = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; getSortedArray ( arr1 , arr2 , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int middleOfThree ( int a , int b , int c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
int main ( ) { int a = 20 , b = 30 , c = 40 ; cout << middleOfThree ( a , b , c ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < vector < int > > transpose ( vector < vector < int > > mat , int row , int col ) {
vector < vector < int > > tr ( col , vector < int > ( row ) ) ;
for ( int i = 0 ; i < row ; i ++ ) {
for ( int j = 0 ; j < col ; j ++ ) {
tr [ j ] [ i ] = mat [ i ] [ j ] ; } } return tr ; }
void RowWiseSort ( vector < vector < int > > & B ) {
for ( int i = 0 ; i < ( int ) B . size ( ) ; i ++ ) {
sort ( B [ i ] . begin ( ) , B [ i ] . end ( ) ) ; } }
void sortCol ( vector < vector < int > > mat , int N , int M ) {
vector < vector < int > > B = transpose ( mat , N , M ) ;
RowWiseSort ( B ) ;
mat = transpose ( B , M , N ) ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < M ; j ++ ) { cout << mat [ i ] [ j ] << " ▁ " ; } cout << ' ' ; } }
int main ( ) {
vector < vector < int > > mat = { { 1 , 6 , 10 } , { 8 , 5 , 9 } , { 9 , 4 , 15 } , { 7 , 3 , 60 } } ; int N = mat . size ( ) ; int M = mat [ 0 ] . size ( ) ;
sortCol ( mat , N , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void largestArea ( int N , int M , int H [ ] , int V [ ] , int h , int v ) {
set < int > s1 ; set < int > s2 ;
for ( int i = 1 ; i <= N + 1 ; i ++ ) s1 . insert ( i ) ;
for ( int i = 1 ; i <= M + 1 ; i ++ ) s2 . insert ( i ) ;
for ( int i = 0 ; i < h ; i ++ ) { s1 . erase ( H [ i ] ) ; }
for ( int i = 0 ; i < v ; i ++ ) { s2 . erase ( V [ i ] ) ; }
int list1 [ s1 . size ( ) ] ; int list2 [ s2 . size ( ) ] ; int i = 0 ; for ( auto it1 = s1 . begin ( ) ; it1 != s1 . end ( ) ; it1 ++ ) { list1 [ i ++ ] = * it1 ; } i = 0 ; for ( auto it2 = s2 . begin ( ) ; it2 != s2 . end ( ) ; it2 ++ ) { list2 [ i ++ ] = * it2 ; }
sort ( list1 , list1 + s1 . size ( ) ) ; sort ( list2 , list2 + s2 . size ( ) ) ; int maxH = 0 , p1 = 0 , maxV = 0 , p2 = 0 ;
for ( int j = 0 ; j < s1 . size ( ) ; j ++ ) { maxH = max ( maxH , list1 [ j ] - p1 ) ; p1 = list1 [ j ] ; }
for ( int j = 0 ; j < s2 . size ( ) ; j ++ ) { maxV = max ( maxV , list2 [ j ] - p2 ) ; p2 = list2 [ j ] ; }
cout << ( maxV * maxH ) << endl ; }
int main ( ) {
int N = 3 , M = 3 ;
int H [ ] = { 2 } ; int V [ ] = { 2 } ; int h = sizeof ( H ) / sizeof ( H [ 0 ] ) ; int v = sizeof ( V ) / sizeof ( V [ 0 ] ) ;
largestArea ( N , M , H , V , h , v ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkifSorted ( int A [ ] , int B [ ] , int N ) {
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
int main ( ) {
int A [ ] = { 3 , 1 , 2 } ;
int B [ ] = { 0 , 1 , 1 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
bool check = checkifSorted ( A , B , N ) ;
if ( check ) { cout << " YES " << endl ; }
else { cout << " NO " << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minSteps ( string A , string B , int M , int N ) { if ( A [ 0 ] > B [ 0 ] ) return 0 ; if ( B [ 0 ] > A [ 0 ] ) { return 1 ; }
if ( M <= N && A [ 0 ] == B [ 0 ] && count ( A . begin ( ) , A . end ( ) , A [ 0 ] ) == M && count ( B . begin ( ) , B . end ( ) , B [ 0 ] ) == N ) return -1 ;
for ( int i = 1 ; i < N ; i ++ ) { if ( B [ i ] > B [ 0 ] ) return 1 ; }
for ( int i = 1 ; i < M ; i ++ ) { if ( A [ i ] < A [ 0 ] ) return 1 ; }
for ( int i = 1 ; i < M ; i ++ ) { if ( A [ i ] > A [ 0 ] ) { swap ( A [ i ] , B [ 0 ] ) ; swap ( A [ 0 ] , B [ 0 ] ) ; return 2 ; } }
for ( int i = 1 ; i < N ; i ++ ) { if ( B [ i ] < B [ 0 ] ) { swap ( A [ 0 ] , B [ i ] ) ; swap ( A [ 0 ] , B [ 0 ] ) ; return 2 ; } }
return 0 ; }
int main ( ) { string A = " adsfd " ; string B = " dffff " ; int M = A . length ( ) ; int N = B . length ( ) ; cout << minSteps ( A , B , M , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <vector> NEW_LINE using namespace std ;
int main ( ) { int arr [ ] = { 4 , 7 , 2 , 3 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int count = minOperations ( arr , n ) ; cout << count ; }
int minOperations ( int arr [ ] , int n ) {
vector < pair < int , int > > vect ; for ( int i = 0 ; i < n ; i ++ ) { vect . push_back ( make_pair ( arr [ i ] , i ) ) ; }
sort ( vect . begin ( ) , vect . end ( ) ) ;
int res = 1 ; int streak = 1 ; int prev = vect [ 0 ] . second ; for ( int i = 1 ; i < n ; i ++ ) { if ( prev < vect [ i ] . second ) { res ++ ;
streak = max ( streak , res ) ; } else res = 1 ; prev = vect [ i ] . second ; }
return n - streak ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define maxN  201
int n1 , n2 , n3 ;
int dp [ maxN ] [ maxN ] [ maxN ] ;
int getMaxSum ( int i , int j , int k , int arr1 [ ] , int arr2 [ ] , int arr3 [ ] ) {
int cnt = 0 ; if ( i >= n1 ) cnt ++ ; if ( j >= n2 ) cnt ++ ; if ( k >= n3 ) cnt ++ ;
if ( cnt >= 2 ) return 0 ;
if ( dp [ i ] [ j ] [ k ] != -1 ) return dp [ i ] [ j ] [ k ] ; int ans = 0 ;
if ( i < n1 && j < n2 )
ans = max ( ans , getMaxSum ( i + 1 , j + 1 , k , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr2 [ j ] ) ; if ( i < n1 && k < n3 ) ans = max ( ans , getMaxSum ( i + 1 , j , k + 1 , arr1 , arr2 , arr3 ) + arr1 [ i ] * arr3 [ k ] ) ; if ( j < n2 && k < n3 ) ans = max ( ans , getMaxSum ( i , j + 1 , k + 1 , arr1 , arr2 , arr3 ) + arr2 [ j ] * arr3 [ k ] ) ;
dp [ i ] [ j ] [ k ] = ans ;
return dp [ i ] [ j ] [ k ] ; }
int maxProductSum ( int arr1 [ ] , int arr2 [ ] , int arr3 [ ] ) {
memset ( dp , -1 , sizeof ( dp ) ) ;
sort ( arr1 , arr1 + n1 ) ; reverse ( arr1 , arr1 + n1 ) ; sort ( arr2 , arr2 + n2 ) ; reverse ( arr2 , arr2 + n2 ) ; sort ( arr3 , arr3 + n3 ) ; reverse ( arr3 , arr3 + n3 ) ; return getMaxSum ( 0 , 0 , 0 , arr1 , arr2 , arr3 ) ; }
int main ( ) { n1 = 2 ; int arr1 [ ] = { 3 , 5 } ; n2 = 2 ; int arr2 [ ] = { 2 , 1 } ; n3 = 3 ; int arr3 [ ] = { 4 , 3 , 5 } ; cout << maxProductSum ( arr1 , arr2 , arr3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findTriplet ( int arr [ ] , int N ) {
sort ( arr , arr + N ) ; int flag = 0 , i ;
for ( i = N - 1 ; i - 2 >= 0 ; i -- ) {
if ( arr [ i - 2 ] + arr [ i - 1 ] > arr [ i ] ) { flag = 1 ; break ; } }
if ( flag ) {
cout << arr [ i - 2 ] << " ▁ " << arr [ i - 1 ] << " ▁ " << arr [ i ] << endl ; }
else { cout << -1 << endl ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 10 , 3 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findTriplet ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int numberofpairs ( int arr [ ] , int N ) {
int answer = 0 ;
sort ( arr , arr + N ) ;
int minDiff = INT_MAX ; for ( int i = 0 ; i < N - 1 ; i ++ )
minDiff = min ( minDiff , arr [ i + 1 ] - arr [ i ] ) ; for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( arr [ i + 1 ] - arr [ i ] == minDiff )
answer ++ ; }
return answer ; }
int main ( ) {
int arr [ ] = { 4 , 2 , 1 , 3 } ; int N = ( sizeof arr ) / ( sizeof arr [ 0 ] ) ;
cout << numberofpairs ( arr , N ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max_length = 0 ;
vector < int > store ;
vector < int > ans ;
void find_max_length ( vector < int > & arr , int index , int sum , int k ) { sum = sum + arr [ index ] ; store . push_back ( arr [ index ] ) ; if ( sum == k ) { if ( max_length < store . size ( ) ) {
max_length = store . size ( ) ;
ans = store ; } } for ( int i = index + 1 ; i < arr . size ( ) ; i ++ ) { if ( sum + arr [ i ] <= k ) {
find_max_length ( arr , i , sum , k ) ;
store . pop_back ( ) ; }
else return ; } return ; } int longestSubsequence ( vector < int > arr , int n , int k ) {
sort ( arr . begin ( ) , arr . end ( ) ) ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( max_length >= n - i ) break ; store . clear ( ) ; find_max_length ( arr , i , 0 , k ) ; } return max_length ; }
int main ( ) { vector < int > arr { -3 , 0 , 1 , 1 , 2 } ; int n = arr . size ( ) ; int k = 1 ; cout << longestSubsequence ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void sortArray ( int A [ ] , int N ) {
int x , y , z ;
if ( N % 4 == 0 N % 4 == 1 ) {
for ( int i = 0 ; i < N / 2 ; i ++ ) { x = i ; if ( i % 2 == 0 ) { y = N - i - 2 ; z = N - i - 1 ; }
A [ z ] = A [ y ] ; A [ y ] = A [ x ] ; A [ x ] = x + 1 ; }
cout << " Sorted ▁ Array : ▁ " ; for ( int i = 0 ; i < N ; i ++ ) cout << A [ i ] << " ▁ " ; }
else cout < < " - 1" ; }
int main ( ) { int A [ ] = { 5 , 4 , 3 , 2 , 1 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; sortArray ( A , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findK ( int arr [ ] , int size , int N ) {
sort ( arr , arr + size ) ; int temp_sum = 0 ;
for ( int i = 0 ; i < size ; i ++ ) { temp_sum += arr [ i ] ;
if ( N - temp_sum == arr [ i ] * ( size - i - 1 ) ) { return arr [ i ] ; } } return -1 ; }
int main ( ) { int arr [ ] = { 3 , 1 , 10 , 4 , 8 } ; int size = sizeof ( arr ) / sizeof ( int ) ; int N = 16 ; cout << findK ( arr , size , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool existsTriplet ( int a [ ] , int b [ ] , int c [ ] , int x , int l1 , int l2 , int l3 ) {
if ( l2 <= l1 and l2 <= l3 ) swap ( l2 , l1 ) , swap ( a , b ) ; else if ( l3 <= l1 and l3 <= l2 ) swap ( l3 , l1 ) , swap ( a , c ) ;
for ( int i = 0 ; i < l1 ; i ++ ) {
int j = 0 , k = l3 - 1 ; while ( j < l2 and k > = 0 ) {
if ( a [ i ] + b [ j ] + c [ k ] == x ) return true ; if ( a [ i ] + b [ j ] + c [ k ] < x ) j ++ ; else k -- ; } } return false ; }
int main ( ) { int a [ ] = { 2 , 7 , 8 , 10 , 15 } ; int b [ ] = { 1 , 6 , 7 , 8 } ; int c [ ] = { 4 , 5 , 5 } ; int l1 = sizeof ( a ) / sizeof ( int ) ; int l2 = sizeof ( b ) / sizeof ( int ) ; int l3 = sizeof ( c ) / sizeof ( int ) ; int x = 14 ; if ( existsTriplet ( a , b , c , x , l1 , l2 , l3 ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <algorithm> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
void printArr ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] ; }
bool compare ( int num1 , int num2 ) {
string A = to_string ( num1 ) ;
string B = to_string ( num2 ) ;
return ( A + B ) <= ( B + A ) ; }
void printSmallest ( int N , int arr [ ] ) {
sort ( arr , arr + N , compare ) ;
printArr ( arr , N ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 2 , 9 , 21 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printSmallest ( N , arr ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; void stableSelectionSort ( int a [ ] , int n ) {
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min = i ; for ( int j = i + 1 ; j < n ; j ++ ) if ( a [ min ] > a [ j ] ) min = j ;
int key = a [ min ] ; while ( min > i ) { a [ min ] = a [ min - 1 ] ; min -- ; } a [ i ] = key ; } } void printArray ( int a [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << a [ i ] << " ▁ " ; cout << endl ; }
int main ( ) { int a [ ] = { 4 , 5 , 3 , 2 , 4 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; stableSelectionSort ( a , n ) ; printArray ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPossible ( int a [ ] , int b [ ] , int n , int k ) {
sort ( a , a + n ) ;
sort ( b , b + n , greater < int > ( ) ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
int main ( ) { int a [ ] = { 2 , 1 , 3 } ; int b [ ] = { 7 , 8 , 9 } ; int k = 10 ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; isPossible ( a , b , n , k ) ? cout << " Yes " : cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int setBitCount ( int num ) { int count = 0 ; while ( num ) { if ( num & 1 ) count ++ ; num >>= 1 ; } return count ; }
void sortBySetBitCount ( int arr [ ] , int n ) { multimap < int , int > count ;
for ( int i = 0 ; i < n ; ++ i ) { count . insert ( { ( -1 ) * setBitCount ( arr [ i ] ) , arr [ i ] } ) ; } for ( auto i : count ) cout << i . second << " ▁ " ; cout << " STRNEWLINE " ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; sortBySetBitCount ( arr , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool canReach ( string s , int L , int R ) {
vector < int > dp ( s . length ( ) ) ;
dp [ 0 ] = 1 ;
int pre = 0 ;
for ( int i = 1 ; i < s . length ( ) ; i ++ ) {
if ( i >= L ) { pre += dp [ i - L ] ; }
if ( i > R ) { pre -= dp [ i - R - 1 ] ; } dp [ i ] = ( pre > 0 ) and ( s [ i ] == '0' ) ; }
return dp [ s . length ( ) - 1 ] ; }
int main ( ) { string S = "01101110" ; int L = 2 , R = 3 ; cout << ( canReach ( S , L , R ) ? " Yes " : " No " ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxXORUtil ( int arr [ ] , int N , int xrr , int orr ) {
if ( N == 0 ) return xrr ^ orr ;
int x = maxXORUtil ( arr , N - 1 , xrr ^ orr , arr [ N - 1 ] ) ;
int y = maxXORUtil ( arr , N - 1 , xrr , orr arr [ N - 1 ] ) ;
return max ( x , y ) ; }
int maximumXOR ( int arr [ ] , int N ) {
return maxXORUtil ( arr , N , 0 , 0 ) ; }
int main ( ) { int arr [ ] = { 1 , 5 , 7 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << maximumXOR ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int N = 1e5 + 5 ;
int visited [ N ] ;
void construct_tree ( int weights [ ] , int n ) { int minimum = * min_element ( weights , weights + n ) ; int maximum = * max_element ( weights , weights + n ) ;
if ( minimum == maximum ) {
cout << " No " ; return ; }
else {
cout << " Yes " << endl ; }
int root = weights [ 0 ] ;
visited [ 1 ] = 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] != root && visited [ i + 1 ] == 0 ) { cout << 1 << " ▁ " << i + 1 << " ▁ " << endl ;
visited [ i + 1 ] = 1 ; } }
int notroot = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( weights [ i ] != root ) { notroot = i + 1 ; break ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( weights [ i ] == root && visited [ i + 1 ] == 0 ) { cout << notroot << " ▁ " << i + 1 << endl ; visited [ i + 1 ] = 1 ; } } }
int main ( ) { int weights [ ] = { 1 , 2 , 1 , 2 , 5 } ; int N = sizeof ( weights ) / sizeof ( weights [ 0 ] ) ;
construct_tree ( weights , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minCost ( string s , int k ) {
int n = s . size ( ) ;
int ans = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
int a [ 26 ] ; for ( int p = 0 ; p < 26 ; p ++ ) { a [ p ] = 0 ; } for ( int j = i ; j < n ; j += k ) { a [ s [ j ] - ' a ' ] ++ ; }
int min_cost = INT_MAX ;
for ( int ch = 0 ; ch < 26 ; ch ++ ) { int cost = 0 ;
for ( int tr = 0 ; tr < 26 ; tr ++ ) cost += abs ( ch - tr ) * a [ tr ] ;
min_cost = min ( min_cost , cost ) ; }
ans += min_cost ; }
cout << ( ans ) ; }
int main ( ) {
string S = " abcdefabc " ; int K = 3 ;
minCost ( S , K ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minAbsDiff ( int N ) { if ( N % 4 == 0 N % 4 == 3 ) { return 0 ; } return 1 ; }
int main ( ) { int N = 6 ; cout << minAbsDiff ( N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  10000
vector < int > adj [ N ] ; int used [ N ] ; int max_matching ;
void AddEdge ( int u , int v ) {
adj [ u ] . push_back ( v ) ;
adj [ v ] . push_back ( u ) ; }
void Matching_dfs ( int u , int p ) { for ( int i = 0 ; i < adj [ u ] . size ( ) ; i ++ ) {
if ( adj [ u ] [ i ] != p ) { Matching_dfs ( adj [ u ] [ i ] , u ) ; } }
if ( ! used [ u ] and ! used [ p ] and p != 0 ) {
max_matching ++ ; used [ u ] = used [ p ] = 1 ; } }
void maxMatching ( ) {
Matching_dfs ( 1 , 0 ) ;
cout << max_matching << " STRNEWLINE " ; }
int main ( ) { int n = 5 ;
AddEdge ( 1 , 2 ) ; AddEdge ( 1 , 3 ) ; AddEdge ( 3 , 4 ) ; AddEdge ( 3 , 5 ) ;
maxMatching ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getMinCost ( vector < int > A , vector < int > B , int N ) { int mini = INT_MAX ; for ( int i = 0 ; i < N ; i ++ ) { mini = min ( mini , min ( A [ i ] , B [ i ] ) ) ; }
return mini * ( 2 * N - 1 ) ; }
int main ( ) { int N = 3 ; vector < int > A = { 1 , 4 , 2 } ; vector < int > B = { 10 , 6 , 12 } ; cout << getMinCost ( A , B , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printVector ( vector < int > & arr ) { if ( arr . size ( ) != 1 ) {
for ( int i = 0 ; i < arr . size ( ) ; i ++ ) { cout << arr [ i ] << " ▁ " ; } cout << endl ; } }
void findWays ( vector < int > & arr , int i , int n ) {
if ( n == 0 ) printVector ( arr ) ;
for ( int j = i ; j <= n ; j ++ ) {
arr . push_back ( j ) ;
findWays ( arr , j , n - j ) ;
arr . pop_back ( ) ; } }
int main ( ) {
int n = 4 ;
vector < int > arr ;
findWays ( arr , 1 , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void Maximum_subsequence ( int A [ ] , int N ) {
unordered_map < int , int > frequency ;
int max_freq = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
frequency [ A [ i ] ] ++ ; } for ( auto it : frequency ) {
if ( it . second > max_freq ) { max_freq = it . second ; } }
cout << max_freq << endl ; }
int main ( ) { int arr [ ] = { 5 , 2 , 6 , 5 , 2 , 4 , 5 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; Maximum_subsequence ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void DivideString ( string s , int n , int k ) { int i , c = 0 , no = 1 ; int c1 = 0 , c2 = 0 ;
int fr [ 26 ] = { 0 } ; string ans = " " ; for ( i = 0 ; i < n ; i ++ ) { fr [ s [ i ] - ' a ' ] ++ ; } char ch , ch1 ; for ( i = 0 ; i < 26 ; i ++ ) {
if ( fr [ i ] == k ) { c ++ ; }
if ( fr [ i ] > k && fr [ i ] != 2 * k ) { c1 ++ ; ch = i + ' a ' ; } if ( fr [ i ] == 2 * k ) { c2 ++ ; ch1 = i + ' a ' ; } } for ( i = 0 ; i < n ; i ++ ) ans = ans + "1" ; map < char , int > mp ; if ( c % 2 == 0 c1 > 0 c2 > 0 ) { for ( i = 0 ; i < n ; i ++ ) {
if ( fr [ s [ i ] - ' a ' ] == k ) { if ( mp . find ( s [ i ] ) != mp . end ( ) ) { ans [ i ] = '2' ; } else { if ( no <= ( c / 2 ) ) { ans [ i ] = '2' ; no ++ ; mp [ s [ i ] ] = 1 ; } } } }
if ( c % 2 == 1 && c1 > 0 ) { no = 1 ; for ( i = 0 ; i < n ; i ++ ) { if ( s [ i ] == ch && no <= k ) { ans [ i ] = '2' ; no ++ ; } } }
if ( c % 2 == 1 && c1 == 0 ) { no = 1 ; int flag = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( s [ i ] == ch1 && no <= k ) { ans [ i ] = '2' ; no ++ ; } if ( fr [ s [ i ] - ' a ' ] == k && flag == 0 && ans [ i ] == '1' ) { ans [ i ] = '2' ; flag = 1 ; } } } cout << ans << endl ; } else {
cout << " NO " << endl ; } }
int main ( ) { string S = " abbbccc " ; int N = S . size ( ) ; int K = 1 ; DivideString ( S , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string check ( int S , int prices [ ] , int type [ ] , int n ) {
for ( int j = 0 ; j < n ; j ++ ) { for ( int k = j + 1 ; k < n ; k ++ ) {
if ( ( type [ j ] == 0 && type [ k ] == 1 ) || ( type [ j ] == 1 && type [ k ] == 0 ) ) { if ( prices [ j ] + prices [ k ] <= S ) { return " Yes " ; } } } } return " No " ; }
int main ( ) { int prices [ ] = { 3 , 8 , 6 , 5 } ; int type [ ] = { 0 , 1 , 1 , 0 } ; int S = 10 ; int n = 4 ;
cout << check ( S , prices , type , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getLargestSum ( int N ) {
for ( int i = 1 ; i * i <= N ; i ++ ) { for ( int j = i + 1 ; j * j <= N ; j ++ ) {
int k = N / j ; int a = k * i ; int b = k * j ;
if ( a <= N && b <= N && a * b % ( a + b ) == 0 )
max_sum = max ( max_sum , a + b ) ; } }
return max_sum ; }
int main ( ) { int N = 25 ; int max_sum = getLargestSum ( N ) ; cout << max_sum << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
string encryptString ( string str , int n ) { int i = 0 , cnt = 0 ; string encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- ) encryptedStr += str [ i ] ; i ++ ; } return encryptedStr ; }
int main ( ) { string str = " geeks " ; int n = str . length ( ) ; cout << encryptString ( str , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minDiff ( int n , int x , int A [ ] ) { int mn = A [ 0 ] , mx = A [ 0 ] ;
for ( int i = 0 ; i < n ; ++ i ) { mn = min ( mn , A [ i ] ) ; mx = max ( mx , A [ i ] ) ; }
return max ( 0 , mx - mn - 2 * x ) ; }
int main ( ) { int n = 3 , x = 3 ; int A [ ] = { 1 , 3 , 6 } ;
cout << minDiff ( n , x , A ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; long swapCount ( string chars ) {
int countLeft = 0 , countRight = 0 ;
int swap = 0 , imbalance = 0 ; for ( int i = 0 ; i < chars . length ( ) ; i ++ ) { if ( chars [ i ] == ' [ ' ) {
countLeft ++ ; if ( imbalance > 0 ) {
swap += imbalance ;
imbalance -- ; } } else if ( chars [ i ] == ' ] ' ) {
countRight ++ ;
imbalance = ( countRight - countLeft ) ; } } return swap ; }
int main ( ) { string s = " [ ] ] [ ] [ " ; cout << swapCount ( s ) << endl ; s = " [ [ ] [ ] ] " ; cout << swapCount ( s ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void longestSubSequence ( pair < int , int > A [ ] , int N ) {
int dp [ N ] ; for ( int i = 0 ; i < N ; i ++ ) {
dp [ i ] = 1 ; for ( int j = 0 ; j < i ; j ++ ) {
if ( A [ j ] . first < A [ i ] . first && A [ j ] . second > A [ i ] . second ) { dp [ i ] = max ( dp [ i ] , dp [ j ] + 1 ) ; } } }
cout << dp [ N - 1 ] << endl ; }
int main ( ) {
pair < int , int > A [ ] = { { 1 , 2 } , { 2 , 2 } , { 3 , 1 } } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
longestSubSequence ( A , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findWays ( int N , int dp [ ] ) {
if ( N == 0 ) { return 1 ; }
if ( dp [ N ] != -1 ) { return dp [ N ] ; } int cnt = 0 ;
for ( int i = 1 ; i <= 6 ; i ++ ) { if ( N - i >= 0 ) { cnt = cnt + findWays ( N - i , dp ) ; } }
return dp [ N ] = cnt ; }
int main ( ) {
int N = 4 ;
int dp [ N + 1 ] ; memset ( dp , -1 , sizeof ( dp ) ) ;
cout << findWays ( N , dp ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findWays ( int N ) {
int dp [ N + 1 ] ; dp [ 0 ] = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { dp [ i ] = 0 ;
for ( int j = 1 ; j <= 6 ; j ++ ) { if ( i - j >= 0 ) { dp [ i ] = dp [ i ] + dp [ i - j ] ; } } }
cout << dp [ N ] ; }
int main ( ) {
int N = 4 ;
findWays ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int INF = 1e9 + 9 ;
struct TrieNode { TrieNode * child [ 26 ] = { NULL } ; } ;
void insert ( int idx , string & s , TrieNode * root ) { TrieNode * temp = root ; for ( int i = idx ; i < s . length ( ) ; i ++ ) {
if ( temp -> child [ s [ i ] - ' a ' ] == NULL )
temp -> child [ s [ i ] - ' a ' ] = new TrieNode ; temp = temp -> child [ s [ i ] - ' a ' ] ; } }
int minCuts ( string S1 , string S2 ) { int n1 = S1 . length ( ) ; int n2 = S2 . length ( ) ;
TrieNode * root = new TrieNode ; for ( int i = 0 ; i < n2 ; i ++ ) {
insert ( i , S2 , root ) ; }
vector < int > dp ( n1 + 1 , INF ) ;
dp [ 0 ] = 0 ; for ( int i = 0 ; i < n1 ; i ++ ) {
TrieNode * temp = root ; for ( int j = i + 1 ; j <= n1 ; j ++ ) { if ( temp -> child [ S1 [ j - 1 ] - ' a ' ] == NULL )
break ;
dp [ j ] = min ( dp [ j ] , dp [ i ] + 1 ) ;
temp = temp -> child [ S1 [ j - 1 ] - ' a ' ] ; } }
if ( dp [ n1 ] >= INF ) return -1 ; else return dp [ n1 ] ; }
int main ( ) { string S1 = " abcdab " ; string S2 = " dabc " ; cout << minCuts ( S1 , S2 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 100 ;
void largestSquare ( int matrix [ ] [ MAX ] , int R , int C , int q_i [ ] , int q_j [ ] , int K , int Q ) { int countDP [ R ] [ C ] ; memset ( countDP , 0 , sizeof ( countDP ) ) ;
countDP [ 0 ] [ 0 ] = matrix [ 0 ] [ 0 ] ; for ( int i = 1 ; i < R ; i ++ ) countDP [ i ] [ 0 ] = countDP [ i - 1 ] [ 0 ] + matrix [ i ] [ 0 ] ; for ( int j = 1 ; j < C ; j ++ ) countDP [ 0 ] [ j ] = countDP [ 0 ] [ j - 1 ] + matrix [ 0 ] [ j ] ; for ( int i = 1 ; i < R ; i ++ ) for ( int j = 1 ; j < C ; j ++ ) countDP [ i ] [ j ] = matrix [ i ] [ j ] + countDP [ i - 1 ] [ j ] + countDP [ i ] [ j - 1 ] - countDP [ i - 1 ] [ j - 1 ] ;
for ( int q = 0 ; q < Q ; q ++ ) { int i = q_i [ q ] ; int j = q_j [ q ] ; int min_dist = min ( min ( i , j ) , min ( R - i - 1 , C - j - 1 ) ) ; int ans = -1 , l = 0 , u = min_dist ;
while ( l <= u ) { int mid = ( l + u ) / 2 ; int x1 = i - mid , x2 = i + mid ; int y1 = j - mid , y2 = j + mid ;
int count = countDP [ x2 ] [ y2 ] ; if ( x1 > 0 ) count -= countDP [ x1 - 1 ] [ y2 ] ; if ( y1 > 0 ) count -= countDP [ x2 ] [ y1 - 1 ] ; if ( x1 > 0 && y1 > 0 ) count += countDP [ x1 - 1 ] [ y1 - 1 ] ;
if ( count <= K ) { ans = 2 * mid + 1 ; l = mid + 1 ; } else u = mid - 1 ; } cout << ans << " STRNEWLINE " ; } }
int main ( ) { int matrix [ ] [ MAX ] = { { 1 , 0 , 1 , 0 , 0 } , { 1 , 0 , 1 , 1 , 1 } , { 1 , 1 , 1 , 1 , 1 } , { 1 , 0 , 0 , 1 , 0 } } ; int K = 9 , Q = 1 ; int q_i [ ] = { 1 } ; int q_j [ ] = { 2 } ; largestSquare ( matrix , 4 , 5 , q_i , q_j , K , Q ) ; return 0 ; }
