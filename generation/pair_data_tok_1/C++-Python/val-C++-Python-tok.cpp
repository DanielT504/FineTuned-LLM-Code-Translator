void Conversion ( double centi ) { double pixels = ( 96 * centi ) / 2.54 ; cout << fixed << setprecision ( 2 ) << pixels ; }
int main ( ) { double centi = 15 ; Conversion ( centi ) ; return 0 ; }
int xor_operations ( int N , int arr [ ] , int M , int K ) {
if ( M < 0 or M > = N ) return -1 ;
if ( K < 0 or K > = N - M ) return -1 ;
for ( int p = 0 ; p < M ; p ++ ) {
vector < int > temp ;
for ( int i = 0 ; i < N ; i ++ ) {
int value = arr [ i ] ^ arr [ i + 1 ] ;
temp . push_back ( value ) ;
arr [ i ] = temp [ i ] ; } }
int ans = arr [ K ] ; return ans ; }
int N = 5 ;
int arr [ ] = { 1 , 4 , 5 , 6 , 7 } ; int M = 1 , K = 2 ;
cout << xor_operations ( N , arr , M , K ) ; return 0 ; }
void canBreakN ( long long n ) {
for ( long long i = 2 ; ; i ++ ) {
long long m = i * ( i + 1 ) / 2 ;
if ( m > n ) break ; long long k = n - m ;
if ( k % i ) continue ;
cout << i << endl ; return ; }
cout << " - 1" ; }
long long N = 12 ;
canBreakN ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findCoprimePair ( int N ) {
for ( int x = 2 ; x <= sqrt ( N ) ; x ++ ) { if ( N % x == 0 ) {
while ( N % x == 0 ) { N /= x ; } if ( N > 1 ) {
cout << x << " ▁ " << N << endl ; return ; } } }
cout << -1 << endl ; }
int N = 45 ; findCoprimePair ( N ) ;
N = 25 ; findCoprimePair ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 10000 ;
vector < int > primes ;
void sieveSundaram ( ) {
bool marked [ MAX / 2 + 1 ] = { 0 } ;
for ( int i = 1 ; i <= ( sqrt ( MAX ) - 1 ) / 2 ; i ++ ) { for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= MAX / 2 ; j = j + 2 * i + 1 ) { marked [ j ] = true ; } }
primes . push_back ( 2 ) ;
for ( int i = 1 ; i <= MAX / 2 ; i ++ ) if ( marked [ i ] == false ) primes . push_back ( 2 * i + 1 ) ; }
bool isWasteful ( int n ) { if ( n == 1 ) return false ;
int original_no = n ; int sumDigits = 0 ; while ( original_no > 0 ) { sumDigits ++ ; original_no = original_no / 10 ; } int pDigit = 0 , count_exp = 0 , p ;
for ( int i = 0 ; primes [ i ] <= n / 2 ; i ++ ) {
while ( n % primes [ i ] == 0 ) {
p = primes [ i ] ; n = n / p ;
count_exp ++ ; }
while ( p > 0 ) { pDigit ++ ; p = p / 10 ; }
while ( count_exp > 1 ) { pDigit ++ ; count_exp = count_exp / 10 ; } }
if ( n != 1 ) { while ( n > 0 ) { pDigit ++ ; n = n / 10 ; } }
return ( pDigit > sumDigits ) ; }
void Solve ( int N ) {
for ( int i = 1 ; i < N ; i ++ ) { if ( isWasteful ( i ) ) { cout << i << " ▁ " ; } } }
sieveSundaram ( ) ; int N = 10 ;
Solve ( N ) ; return 0 ; }
int printhexaRec ( int n ) { if ( n == 0 n == 1 n == 2 n == 3 n == 4 n == 5 ) return 0 ; else if ( n == 6 ) return 1 ; else return ( printhexaRec ( n - 1 ) + printhexaRec ( n - 2 ) + printhexaRec ( n - 3 ) + printhexaRec ( n - 4 ) + printhexaRec ( n - 5 ) + printhexaRec ( n - 6 ) ) ; } int printhexa ( int n ) { cout << printhexaRec ( n ) << endl ; }
int main ( ) { privatenthexa ( n ) ; }
void printhexa ( int n ) { if ( n < 0 ) return ;
int first = 0 ; int second = 0 ; int third = 0 ; int fourth = 0 ; int fifth = 0 ; int sixth = 1 ;
int curr = 0 ; if ( n < 6 ) cout << first << endl ; else if ( n == 6 ) cout << sixth << endl ; else {
for ( int i = 6 ; i < n ; i ++ ) { curr = first + second + third + fourth + fifth + sixth ; first = second ; second = third ; third = fourth ; fourth = fifth ; fifth = sixth ; sixth = curr ; } } cout << curr << endl ; }
int main ( ) { int n = 11 ; printhexa ( n ) ; return 0 ; }
void smallestNumber ( int N ) { cout << ( N % 9 + 1 ) * pow ( 10 , ( N / 9 ) ) - 1 ; }
int main ( ) { int N = 10 ; smallestNumber ( N ) ; return 0 ; }
bool isComposite ( int n ) {
if ( n <= 3 ) return false ;
if ( n % 2 == 0 or n % 3 == 0 ) return true ; int i = 5 ; while ( i * i <= n ) { if ( n % i == 0 or n % ( i + 2 ) == 0 ) return true ; i = i + 6 ; } return false ; }
void Compositorial_list ( int n ) { int l = 0 ; for ( int i = 4 ; i < 1000000 ; i ++ ) { if ( l < n ) { if ( isComposite ( i ) ) { compo . push_back ( i ) ; l += 1 ; } } } }
int calculateCompositorial ( int n ) {
int result = 1 ; for ( int i = 0 ; i < n ; i ++ ) result = result * compo [ i ] ; return result ; }
int main ( ) { int n = 5 ;
Compositorial_list ( n ) ; cout << ( calculateCompositorial ( n ) ) ; return 0 ; }
int b [ 50 ] = { 0 } ;
int PowerArray ( int n , int k ) {
int count = 0 ;
while ( k ) { if ( k % n == 0 ) { k /= n ; count ++ ; }
else if ( k % n == 1 ) { k -= 1 ; b [ count ] ++ ;
if ( b [ count ] > 1 ) { cout << -1 ; return 0 ; } }
else { cout << -1 ; return 0 ; } }
for ( int i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] ) { cout << i << " , ▁ " ; } } }
int main ( ) { int N = 3 ; int K = 40 ; PowerArray ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findSum ( int N , int k ) {
int sum = 0 ; for ( int i = 1 ; i <= N ; i ++ ) {
sum += pow ( i , k ) ; }
return sum ; }
int main ( ) { int N = 8 , k = 4 ;
cout << findSum ( N , k ) << endl ; return 0 ; }
int countIndices ( int arr [ ] , int n ) {
int cnt = 0 ;
int max = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( max < arr [ i ] ) {
max = arr [ i ] ;
cnt ++ ; } } return cnt ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << countIndices ( arr , n ) ; return 0 ; }
const string bin [ ] = { "000" , "001" , "010" , "011" , "100" , "101" , "110" , "111" } ;
int maxFreq ( string s ) {
string binary = " " ;
for ( int i = 0 ; i < s . length ( ) ; i ++ ) { binary += bin [ s [ i ] - '0' ] ; }
binary = binary . substr ( 0 , binary . length ( ) - 1 ) ; int count = 1 , prev = -1 , i , j = 0 ; for ( i = binary . length ( ) - 1 ; i >= 0 ; i -- , j ++ )
if ( binary [ i ] == '1' ) {
count = max ( count , j - prev ) ; prev = j ; } return count ; }
int main ( ) { string octal = "13" ; cout << maxFreq ( octal ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int sz = 1e5 ; bool isPrime [ sz + 1 ] ;
void sieve ( ) { memset ( isPrime , true , sizeof ( isPrime ) ) ; isPrime [ 0 ] = isPrime [ 1 ] = false ; for ( int i = 2 ; i * i <= sz ; i ++ ) { if ( isPrime [ i ] ) { for ( int j = i * i ; j < sz ; j += i ) { isPrime [ j ] = false ; } } } }
void findPrimesD ( int d ) {
int left = pow ( 10 , d - 1 ) ; int right = pow ( 10 , d ) - 1 ;
for ( int i = left ; i <= right ; i ++ ) {
if ( isPrime [ i ] ) { cout << i << " ▁ " ; } } }
int main ( ) {
sieve ( ) ; int d = 1 ; findPrimesD ( d ) ; return 0 ; }
int Cells ( int n , int x ) { if ( n <= 0 x <= 0 x > n * n ) return 0 ; int i = 0 , count = 0 ; while ( ++ i * i < x ) if ( x % i == 0 && x <= n * i ) count += 2 ; return i * i == x ? count + 1 : count ; }
int main ( ) { int n = 6 , x = 12 ;
cout << ( Cells ( n , x ) ) ; return 0 ; }
int maxOfMin ( int a [ ] , int n , int S ) {
int mi = INT_MAX ;
int s1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { s1 += a [ i ] ; mi = min ( a [ i ] , mi ) ; }
if ( s1 < S ) return -1 ;
if ( s1 == S ) return 0 ;
int low = 0 ;
int high = mi ;
int ans ;
while ( low <= high ) { int mid = ( low + high ) / 2 ;
if ( s1 - ( mid * n ) >= S ) { ans = mid ; low = mid + 1 ; }
else high = mid - 1 ; }
return ans ; }
int main ( ) { int a [ ] = { 10 , 10 , 10 , 10 , 10 } ; int S = 10 ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << maxOfMin ( a , n , S ) ; return 0 ; }
void Alphabet_N_Pattern ( int N ) { int index , side_index , size ;
int Right = 1 , Left = 1 , Diagonal = 2 ;
for ( index = 0 ; index < N ; index ++ ) {
cout << Left ++ ;
for ( side_index = 0 ; side_index < 2 * ( index ) ; side_index ++ ) cout << " ▁ " ;
if ( index != 0 && index != N - 1 ) cout << Diagonal ++ ; else cout << " ▁ " ;
for ( side_index = 0 ; side_index < 2 * ( N - index - 1 ) ; side_index ++ ) cout << " ▁ " ;
cout << Right ++ ; cout << endl ; } }
int main ( int argc , char * * argv ) {
int Size = 6 ;
Alphabet_N_Pattern ( Size ) ; }
int isSumDivides ( int N ) { int temp = N ; int sum = 0 ;
while ( temp ) { sum += temp % 10 ; temp /= 10 ; } if ( N % sum == 0 ) return 1 ; else return 0 ; }
int main ( ) { int N = 12 ; if ( isSumDivides ( N ) ) cout << " YES " ; else cout << " NO " ; return 0 ; }
int sum ( int N ) { int S1 , S2 , S3 ; S1 = ( ( N / 3 ) ) * ( 2 * 3 + ( N / 3 - 1 ) * 3 ) / 2 ; S2 = ( ( N / 4 ) ) * ( 2 * 4 + ( N / 4 - 1 ) * 4 ) / 2 ; S3 = ( ( N / 12 ) ) * ( 2 * 12 + ( N / 12 - 1 ) * 12 ) / 2 ; return S1 + S2 - S3 ; }
int main ( ) { int N = 20 ; cout << sum ( 12 ) ; return 0 ; }
long long nextGreater ( long long N ) { long long power_of_2 = 1 , shift_count = 0 ;
while ( true ) {
if ( ( ( N >> shift_count ) & 1 ) % 2 == 0 ) break ;
shift_count ++ ;
power_of_2 = power_of_2 * 2 ; }
return ( N + power_of_2 ) ; }
int main ( ) { long long N = 11 ;
cout << " The ▁ next ▁ number ▁ is ▁ = ▁ " << nextGreater ( N ) ; return 0 ; }
int countWays ( int n ) {
if ( n == 0 ) return 1 ; if ( n <= 2 ) return n ;
int f0 = 1 , f1 = 1 , f2 = 2 , ans ;
for ( int i = 3 ; i <= n ; i ++ ) { ans = f0 + f1 + f2 ; f0 = f1 ; f1 = f2 ; f2 = ans ; }
return ans ; }
int main ( ) { int n = 4 ; cout << countWays ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int n = 6 , m = 6 ;
void maxSum ( long arr [ n ] [ m ] ) {
long dp [ n + 1 ] [ 3 ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) {
long m1 = 0 , m2 = 0 , m3 = 0 ; for ( int j = 0 ; j < m ; j ++ ) {
if ( ( j / ( m / 3 ) ) == 0 ) { m1 = max ( m1 , arr [ i ] [ j ] ) ; }
else if ( ( j / ( m / 3 ) ) == 1 ) { m2 = max ( m2 , arr [ i ] [ j ] ) ; }
else if ( ( j / ( m / 3 ) ) == 2 ) { m3 = max ( m3 , arr [ i ] [ j ] ) ; } }
dp [ i + 1 ] [ 0 ] = max ( dp [ i ] [ 1 ] , dp [ i ] [ 2 ] ) + m1 ; dp [ i + 1 ] [ 1 ] = max ( dp [ i ] [ 0 ] , dp [ i ] [ 2 ] ) + m2 ; dp [ i + 1 ] [ 2 ] = max ( dp [ i ] [ 1 ] , dp [ i ] [ 0 ] ) + m3 ; }
cout << max ( max ( dp [ n ] [ 0 ] , dp [ n ] [ 1 ] ) , dp [ n ] [ 2 ] ) << ' ' }
int main ( ) { long arr [ n ] [ m ] = { { 1 , 3 , 5 , 2 , 4 , 6 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 1 , 3 , 5 , 2 , 4 , 6 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 6 , 4 , 5 , 1 , 3 , 2 } , { 1 , 3 , 5 , 2 , 4 , 6 } } ; maxSum ( arr ) ; return 0 ; }
void solve ( string & s ) { int n = s . length ( ) ;
int dp [ n ] [ n ] ; memset ( dp , 0 , sizeof dp ) ;
for ( int len = n - 1 ; len >= 0 ; -- len ) {
for ( int i = 0 ; i + len < n ; ++ i ) {
int j = i + len ;
if ( i == 0 and j == n - 1 ) { if ( s [ i ] == s [ j ] ) dp [ i ] [ j ] = 2 ; else if ( s [ i ] != s [ j ] ) dp [ i ] [ j ] = 1 ; } else { if ( s [ i ] == s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i ] [ j ] += dp [ i - 1 ] [ j ] ; } if ( j + 1 <= n - 1 ) { dp [ i ] [ j ] += dp [ i ] [ j + 1 ] ; } if ( i - 1 < 0 or j + 1 >= n ) {
dp [ i ] [ j ] += 1 ; } } else if ( s [ i ] != s [ j ] ) {
if ( i - 1 >= 0 ) { dp [ i ] [ j ] += dp [ i - 1 ] [ j ] ; } if ( j + 1 <= n - 1 ) { dp [ i ] [ j ] += dp [ i ] [ j + 1 ] ; } if ( i - 1 >= 0 and j + 1 <= n - 1 ) {
dp [ i ] [ j ] -= dp [ i - 1 ] [ j + 1 ] ; } } } } } vector < int > ways ; for ( int i = 0 ; i < n ; ++ i ) { if ( i == 0 or i == n - 1 ) {
ways . push_back ( 1 ) ; } else {
int total = dp [ i - 1 ] [ i + 1 ] ; ways . push_back ( total ) ; } } for ( int i = 0 ; i < ways . size ( ) ; ++ i ) { cout << ways [ i ] << " ▁ " ; } }
int main ( ) { string s = " xyxyx " ; solve ( s ) ; return 0 ; }
ll getChicks ( int n ) {
int size = max ( n , 7 ) ; ll dp [ size ] ; dp [ 0 ] = 0 ; dp [ 1 ] = 1 ;
for ( int i = 2 ; i <= 6 ; i ++ ) { dp [ i ] = dp [ i - 1 ] * 3 ; }
dp [ 7 ] = 726 ;
for ( int i = 8 ; i <= n ; i ++ ) {
dp [ i ] = ( dp [ i - 1 ] - ( 2 * dp [ i - 6 ] / 3 ) ) * 3 ; } return dp [ n ] ; }
int main ( ) { int n = 3 ; cout << getChicks ( n ) ; return 0 ; }
ll getChicks ( int n ) { ll chicks = ( ll ) pow ( 3 , n - 1 ) ; return chicks ; }
int main ( ) { int n = 3 ; cout << getChicks ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define n  3 NEW_LINE using namespace std ;
int dp [ n ] [ n ] ;
int v [ n ] [ n ] ;
int minSteps ( int i , int j , int arr [ ] [ n ] ) {
if ( i == n - 1 and j == n - 1 ) return 0 ; if ( i > n - 1 j > n - 1 ) return 9999999 ;
if ( v [ i ] [ j ] ) return dp [ i ] [ j ] ; v [ i ] [ j ] = 1 ; dp [ i ] [ j ] = 9999999 ;
for ( int k = max ( 0 , arr [ i ] [ j ] + j - n + 1 ) ; k <= min ( n - i - 1 , arr [ i ] [ j ] ) ; k ++ ) { dp [ i ] [ j ] = min ( dp [ i ] [ j ] , minSteps ( i + k , j + arr [ i ] [ j ] - k , arr ) ) ; } dp [ i ] [ j ] ++ ; return dp [ i ] [ j ] ; }
int main ( ) { int arr [ n ] [ n ] = { { 4 , 1 , 2 } , { 1 , 1 , 1 } , { 2 , 1 , 1 } } ; int ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) cout << -1 ; else cout << ans ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define n  3 NEW_LINE using namespace std ;
int dp [ n ] [ n ] ;
int v [ n ] [ n ] ;
int minSteps ( int i , int j , int arr [ ] [ n ] ) {
if ( i == n - 1 and j == n - 1 ) return 0 ; if ( i > n - 1 j > n - 1 ) return 9999999 ;
if ( v [ i ] [ j ] ) return dp [ i ] [ j ] ; v [ i ] [ j ] = 1 ;
dp [ i ] [ j ] = 1 + min ( minSteps ( i + arr [ i ] [ j ] , j , arr ) , minSteps ( i , j + arr [ i ] [ j ] , arr ) ) ; return dp [ i ] [ j ] ; }
int main ( ) { int arr [ n ] [ n ] = { { 2 , 1 , 2 } , { 1 , 1 , 1 } , { 1 , 1 , 1 } } ; int ans = minSteps ( 0 , 0 , arr ) ; if ( ans >= 9999999 ) cout << -1 ; else cout << ans ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 1001 ; int dp [ MAX ] [ MAX ] ;
int MaxProfit ( int treasure [ ] , int color [ ] , int n , int k , int col , int A , int B ) {
return dp [ k ] [ col ] = 0 ; if ( dp [ k ] [ col ] != -1 ) return dp [ k ] [ col ] ; int sum = 0 ;
sum += max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return dp [ k ] [ col ] = sum ; }
int main ( ) { int A = -5 , B = 7 ; int treasure [ ] = { 4 , 8 , 2 , 9 } ; int color [ ] = { 2 , 2 , 6 , 2 } ; int n = sizeof ( color ) / sizeof ( color [ 0 ] ) ; memset ( dp , -1 , sizeof ( dp ) ) ; cout << MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ; return 0 ; }
int printTetra ( int n ) { int dp [ n + 5 ] ;
dp [ 0 ] = 0 ; dp [ 1 ] = dp [ 2 ] = 1 ; dp [ 3 ] = 2 ; for ( int i = 4 ; i <= n ; i ++ ) dp [ i ] = dp [ i - 1 ] + dp [ i - 2 ] + dp [ i - 3 ] + dp [ i - 4 ] ; cout << dp [ n ] ; }
int main ( ) { int n = 10 ; printTetra ( n ) ; return 0 ; }
int maxSum1 ( int arr [ ] , int n ) { int dp [ n ] ; int maxi = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
dp [ i ] = arr [ i ] ;
if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( int i = 2 ; i < n - 1 ; i ++ ) {
for ( int j = 0 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < dp [ j ] + arr [ i ] ) { dp [ i ] = dp [ j ] + arr [ i ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; }
int maxSum2 ( int arr [ ] , int n ) { int dp [ n ] ; int maxi = 0 ; for ( int i = 1 ; i < n ; i ++ ) { dp [ i ] = arr [ i ] ; if ( maxi < arr [ i ] ) maxi = arr [ i ] ; }
for ( int i = 3 ; i < n ; i ++ ) {
for ( int j = 1 ; j < i - 1 ; j ++ ) {
if ( dp [ i ] < arr [ i ] + dp [ j ] ) { dp [ i ] = arr [ i ] + dp [ j ] ;
if ( maxi < dp [ i ] ) maxi = dp [ i ] ; } } }
return maxi ; } int findMaxSum ( int arr [ ] , int n ) { return max ( maxSum1 ( arr , n ) , maxSum2 ( arr , n ) ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMaxSum ( arr , n ) ; return 0 ; }
int permutationCoeff ( int n , int k ) { int fact [ n + 1 ] ;
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = i * fact [ i - 1 ] ;
return fact [ n ] / fact [ n - k ] ; }
int main ( ) { int n = 10 , k = 2 ; cout << " Value ▁ of ▁ P ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << permutationCoeff ( n , k ) ; return 0 ; }
bool isSubsetSum ( int set [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
int main ( ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = sizeof ( set ) / sizeof ( set [ 0 ] ) ; if ( isSubsetSum ( set , n , sum ) == true ) cout << " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else cout << " No ▁ subset ▁ with ▁ given ▁ sum " ; return 0 ; }
void compute_z ( string s , int z [ ] ) { int l = 0 , r = 0 ; int n = s . length ( ) ; for ( int i = 1 ; i <= n - 1 ; i ++ ) { if ( i > r ) { l = i , r = i ; while ( r < n && s [ r - l ] == s [ r ] ) r ++ ; z [ i ] = r - l ; r -- ; } else { int k = i - l ; if ( z [ k ] < r - i + 1 ) { z [ i ] = z [ k ] ; } else { l = i ; while ( r < n && s [ r - l ] == s [ r ] ) r ++ ; z [ i ] = r - l ; r -- ; } } } }
int countPermutation ( string a , string b ) {
b = b + b ;
b = b . substr ( 0 , b . size ( ) - 1 ) ;
int ans = 0 ; string s = a + " $ " + b ; int n = s . length ( ) ;
int z [ n ] ; compute_z ( s , z ) ; for ( int i = 1 ; i <= n - 1 ; i ++ ) {
if ( z [ i ] == a . length ( ) ) ans ++ ; } return ans ; }
int main ( ) { string a = "101" ; string b = "101" ; cout << countPermutation ( a , b ) << endl ; return 0 ; }
void smallestSubsequence ( string & S , int K ) {
int N = S . size ( ) ;
stack < char > answer ;
for ( int i = 0 ; i < N ; ++ i ) {
if ( answer . empty ( ) ) { answer . push ( S [ i ] ) ; } else {
while ( ( ! answer . empty ( ) ) && ( S [ i ] < answer . top ( ) )
if ( answer . empty ( ) || answer . size ( ) < K ) {
answer . push ( S [ i ] ) ; } } }
string ret ;
while ( ! answer . empty ( ) ) { ret . push_back ( answer . top ( ) ) ; answer . pop ( ) ; }
reverse ( ret . begin ( ) , ret . end ( ) ) ;
cout << ret ; }
int main ( ) { string S = " aabdaabc " ; int K = 3 ; smallestSubsequence ( S , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int is_rtol ( string s ) { int tmp = sqrt ( s . length ( ) ) - 1 ; char first = s [ tmp ] ;
for ( int pos = tmp ; pos < s . length ( ) - 1 ; pos += tmp ) {
if ( s [ pos ] != first ) { return false ; } } return true ; }
int main ( ) {
string str = " abcxabxcaxbcxabc " ;
if ( is_rtol ( str ) ) { cout << " Yes " << endl ; } else { cout << " No " << endl ; } return 0 ; }
bool check ( string str , int K ) {
if ( str . size ( ) % K == 0 ) { int sum = 0 , i ;
for ( i = 0 ; i < K ; i ++ ) { sum += str [ i ] ; }
for ( int j = i ; j < str . size ( ) ; j += K ) { int s_comp = 0 ; for ( int p = j ; p < j + K ; p ++ ) s_comp += str [ p ] ;
if ( s_comp != sum )
return false ; }
return true ; }
return false ; }
int main ( ) { int K = 3 ; string str = " abdcbbdba " ; if ( check ( str , K ) ) cout << " YES " << endl ; else cout << " NO " << endl ; }
int maxSum ( string & str ) { int maximumSum = 0 ;
totalOnes = count ( str . begin ( ) , str . end ( ) , '1' ) ;
int zero = 0 , ones = 0 ;
for ( int i = 0 ; str [ i ] ; i ++ ) { if ( str [ i ] == '0' ) { zero ++ ; } else { ones ++ ; }
maximumSum = max ( maximumSum , zero + ( totalOnes - ones ) ) ; } return maximumSum ; }
int main ( ) {
string str = "011101" ;
cout << maxSum ( str ) ; return 0 ; }
int maxLenSubStr ( string & s ) {
if ( s . length ( ) < 3 ) return s . length ( ) ;
int temp = 2 ; int ans = 2 ;
for ( int i = 2 ; i < s . length ( ) ; i ++ ) {
if ( s [ i ] != s [ i - 1 ] s [ i ] != s [ i - 2 ] ) temp ++ ;
else { ans = max ( temp , ans ) ; temp = 2 ; } } ans = max ( temp , ans ) ; return ans ; }
int main ( ) { string s = " baaabbabbb " ; cout << maxLenSubStr ( s ) ; return 0 ; }
int no_of_ways ( string s ) { int n = s . length ( ) ;
int count_left = 0 , count_right = 0 ;
for ( int i = 0 ; i < n ; ++ i ) { if ( s [ i ] == s [ 0 ] ) { ++ count_left ; } else break ; }
for ( int i = n - 1 ; i >= 0 ; -- i ) { if ( s [ i ] == s [ n - 1 ] ) { ++ count_right ; } else break ; }
if ( s [ 0 ] == s [ n - 1 ] ) return ( ( count_left + 1 ) * ( count_right + 1 ) ) ;
else return ( count_left + count_right + 1 ) ; }
int main ( ) { string s = " geeksforgeeks " ; cout << no_of_ways ( s ) ; return 0 ; }
void preCompute ( int n , string s , int pref [ ] ) { pref [ 0 ] = 0 ; for ( int i = 1 ; i < n ; i ++ ) { pref [ i ] = pref [ i - 1 ] ; if ( s [ i - 1 ] == s [ i ] ) pref [ i ] ++ ; } }
int query ( int pref [ ] , int l , int r ) { return pref [ r ] - pref [ l ] ; }
int main ( ) { string s = " ggggggg " ; int n = s . length ( ) ; int pref [ n ] ; preCompute ( n , s , pref ) ;
int l = 1 ; int r = 2 ; cout << query ( pref , l , r ) << endl ;
l = 1 ; r = 5 ; cout << query ( pref , l , r ) << endl ; return 0 ; }
string findDirection ( string s ) { int count = 0 ; string d = " " ; for ( int i = 0 ; i < s . length ( ) ; i ++ ) { if ( s [ 0 ] == ' ' ) return NULL ; if ( s [ i ] == ' L ' ) count -- ; else { if ( s [ i ] == ' R ' ) count ++ ; } }
if ( count > 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == 1 ) d = " E " ; else if ( count % 4 == 2 ) d = " S " ; else if ( count % 4 == 3 ) d = " W " ; }
if ( count < 0 ) { if ( count % 4 == 0 ) d = " N " ; else if ( count % 4 == -1 ) d = " W " ; else if ( count % 4 == -2 ) d = " S " ; else if ( count % 4 == -3 ) d = " E " ; } return d ; }
int main ( ) { string s = " LLRLRRL " ; cout << ( findDirection ( s ) ) << endl ; s = " LL " ; cout << ( findDirection ( s ) ) << endl ; }
bool isCheck ( string str ) { int len = str . length ( ) ; string lowerStr = " " , upperStr = " " ;
for ( int i = 0 ; i < len ; i ++ ) {
if ( str [ i ] >= 65 && str [ i ] <= 91 ) upperStr = upperStr + str [ i ] ; else lowerStr = lowerStr + str [ i ] ; }
transform ( lowerStr . begin ( ) , lowerStr . end ( ) , lowerStr . begin ( ) , :: toupper ) ; return lowerStr == upperStr ; }
int main ( ) { string str = " geeGkEEsKS " ; isCheck ( str ) ? cout << " Yes " : cout << " No " ; return 0 ; }
void encode ( string s , int k ) {
string newS ;
for ( int i = 0 ; i < s . length ( ) ; ++ i ) {
int val = int ( s [ i ] ) ;
int dup = k ;
if ( val + k > 122 ) { k -= ( 122 - val ) ; k = k % 26 ; newS += char ( 96 + k ) ; } else newS += char ( val + k ) ; k = dup ; }
cout << newS ; }
int main ( ) { string str = " abc " ; int k = 28 ;
encode ( str , k ) ; return 0 ; }
bool isVowel ( char x ) { if ( x == ' a ' x == ' e ' x == ' i ' x == ' o ' x == ' u ' ) return true ; else return false ; }
string updateSandwichedVowels ( string a ) { int n = a . length ( ) ;
string updatedString = " " ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( ! i i == n - 1 ) { updatedString += a [ i ] ; continue ; }
if ( isVowel ( a [ i ] ) && ! isVowel ( a [ i - 1 ] ) && ! isVowel ( a [ i + 1 ] ) ) { continue ; }
updatedString += a [ i ] ; } return updatedString ; }
int main ( ) { string str = " geeksforgeeks " ;
string updatedString = updateSandwichedVowels ( str ) ; cout << updatedString ; return 0 ; }
struct Node { int data ; struct Node * left , * right ; } ;
void findPathUtil ( Node * root , int k , vector < int > path , int flag , int & ans ) { if ( root == NULL ) return ;
if ( root -> data >= k ) flag = 1 ;
if ( root -> left == NULL && root -> right == NULL ) { if ( flag == 1 ) { ans = 1 ; cout << " ( " ; for ( int i = 0 ; i < path . size ( ) ; i ++ ) { cout << path [ i ] << " , ▁ " ; } cout << root -> data << " ) , ▁ " ; } return ; }
path . push_back ( root -> data ) ;
findPathUtil ( root -> left , k , path , flag , ans ) ; findPathUtil ( root -> right , k , path , flag , ans ) ;
path . pop_back ( ) ; }
void findPath ( Node * root , int k ) {
int flag = 0 ;
int ans = 0 ; vector < int > v ;
findPathUtil ( root , k , v , flag , ans ) ;
if ( ans == 0 ) cout << " - 1" ; }
int main ( void ) { int K = 25 ;
struct Node * root = newNode ( 10 ) ; root -> left = newNode ( 5 ) ; root -> right = newNode ( 8 ) ; root -> left -> left = newNode ( 29 ) ; root -> left -> right = newNode ( 2 ) ; root -> right -> right = newNode ( 98 ) ; root -> right -> left = newNode ( 1 ) ; root -> right -> right -> right = newNode ( 50 ) ; root -> left -> left -> left = newNode ( 20 ) ; findPath ( root , K ) ; return 0 ; }
int Tridecagonal_num ( int n ) {
return ( 11 * n * n - 9 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << Tridecagonal_num ( n ) << endl ; n = 10 ; cout << Tridecagonal_num ( n ) << endl ; return 0 ; }
int findNumbers ( int n , int w ) { int x = 0 , sum = 0 ;
if ( w >= 0 && w <= 8 ) {
x = 9 - w ; }
else if ( w >= -9 && w <= -1 ) {
x = 10 + w ; } sum = pow ( 10 , n - 2 ) ; sum = ( x * sum ) ; return sum ; }
n = 3 , w = 4 ;
cout << findNumbers ( n , w ) ; ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int MaximumHeight ( int a [ ] , int n ) { int result = 1 ; for ( int i = 1 ; i <= n ; ++ i ) {
long long y = ( i * ( i + 1 ) ) / 2 ;
if ( y < n ) result = i ;
else break ; } return result ; }
int main ( ) { int arr [ ] = { 40 , 100 , 20 , 30 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << MaximumHeight ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int findK ( int n , int k ) { vector < long > a ;
for ( int i = 1 ; i < n ; i ++ ) if ( i % 2 == 1 ) a . push_back ( i ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( i % 2 == 0 ) a . push_back ( i ) ; return ( a [ k - 1 ] ) ; }
int main ( ) { long n = 10 , k = 3 ; cout << findK ( n , k ) << endl ; return 0 ; }
#include <iostream> NEW_LINE int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
int main ( ) { int num = 5 ; printf ( " Factorial ▁ of ▁ % d ▁ is ▁ % d " , num , factorial ( num ) ) ; return 0 ; }
int pell ( int n ) { if ( n <= 2 ) return n ; int a = 1 ; int b = 2 ; int c , i ; for ( i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
int main ( ) { int n = 4 ; cout << pell ( n ) ; return 0 ; }
bool isMultipleOf10 ( int n ) { return ( n % 15 == 0 ) ; }
int main ( ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) printf ( " Yes STRNEWLINE " ) ; else printf ( " No STRNEWLINE " ) ; return 0 ; }
int countOddPrimeFactors ( int n ) { int result = 1 ;
while ( n % 2 == 0 ) n /= 2 ;
for ( int i = 3 ; i * i <= n ; i += 2 ) { int divCount = 0 ;
while ( n % i == 0 ) { n /= i ; ++ divCount ; } result *= divCount + 1 ; }
if ( n > 2 ) result *= 2 ; return result ; } int politness ( int n ) { return countOddPrimeFactors ( n ) - 1 ; }
int main ( ) { int n = 90 ; cout << " Politness ▁ of ▁ " << n << " ▁ = ▁ " << politness ( n ) << " STRNEWLINE " ; n = 15 ; cout << " Politness ▁ of ▁ " << n << " ▁ = ▁ " << politness ( n ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define MAX  1000000 NEW_LINE using namespace std ;
vector < int > primes ;
void Sieve ( ) { int n = MAX ;
int nNew = sqrt ( n ) ;
int marked [ n / 2 + 500 ] = { 0 } ;
for ( int i = 1 ; i <= ( nNew - 1 ) / 2 ; i ++ ) for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= n / 2 ; j = j + 2 * i + 1 ) marked [ j ] = 1 ;
primes . push_back ( 2 ) ;
for ( int i = 1 ; i <= n / 2 ; i ++ ) if ( marked [ i ] == 0 ) primes . push_back ( 2 * i + 1 ) ; }
int binarySearch ( int left , int right , int n ) { if ( left <= right ) { int mid = ( left + right ) / 2 ;
if ( mid == 0 || mid == primes . size ( ) - 1 ) return primes [ mid ] ;
if ( primes [ mid ] == n ) return primes [ mid - 1 ] ;
if ( primes [ mid ] < n && primes [ mid + 1 ] > n ) return primes [ mid ] ; if ( n < primes [ mid ] ) return binarySearch ( left , mid - 1 , n ) ; else return binarySearch ( mid + 1 , right , n ) ; } return 0 ; }
int main ( ) { Sieve ( ) ; int n = 17 ; cout << binarySearch ( 0 , primes . size ( ) - 1 , n ) ; return 0 ; }
unsigned int factorial ( unsigned int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
int main ( ) { int num = 5 ; cout << " Factorial ▁ of ▁ " << num << " ▁ is ▁ " << factorial ( num ) << endl ; return 0 ; }
int FlipBits ( unsigned int n ) { return n -= ( n & ( - n ) ) ; }
int main ( ) { int N = 12 ; cout << " The ▁ number ▁ after ▁ unsetting ▁ the " ; cout << " ▁ rightmost ▁ set ▁ bit : ▁ " << FlipBits ( N ) ; return 0 ; }
void Maximum_xor_Triplet ( int n , int a [ ] ) {
set < int > s ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = i ; j < n ; j ++ ) {
s . insert ( a [ i ] ^ a [ j ] ) ; } } int ans = 0 ; for ( auto i : s ) { for ( int j = 0 ; j < n ; j ++ ) {
ans = max ( ans , i ^ a [ j ] ) ; } } cout << ans << " STRNEWLINE " ; }
int main ( ) { int a [ ] = { 1 , 3 , 8 , 15 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; Maximum_xor_Triplet ( n , a ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printMissing ( int arr [ ] , int n , int low , int high ) { sort ( arr , arr + n ) ;
int * ptr = lower_bound ( arr , arr + n , low ) ; int index = ptr - arr ;
int i = index , x = low ; while ( i < n && x <= high ) {
if ( arr [ i ] != x ) cout << x << " ▁ " ;
else i ++ ;
x ++ ; }
while ( x <= high ) cout << x ++ << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 3 , 5 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int low = 1 , high = 10 ; printMissing ( arr , n , low , high ) ; return 0 ; }
void printMissing ( int arr [ ] , int n , int low , int high ) {
bool points_of_range [ high - low + 1 ] = { false } ; for ( int i = 0 ; i < n ; i ++ ) {
if ( low <= arr [ i ] && arr [ i ] <= high ) points_of_range [ arr [ i ] - low ] = true ; }
for ( int x = 0 ; x <= high - low ; x ++ ) { if ( points_of_range [ x ] == false ) cout << low + x << " ▁ " ; } }
int main ( ) { int arr [ ] = { 1 , 3 , 5 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int low = 1 , high = 10 ; printMissing ( arr , n , low , high ) ; return 0 ; }
void printMissing ( int arr [ ] , int n , int low , int high ) {
unordered_set < int > s ; for ( int i = 0 ; i < n ; i ++ ) s . insert ( arr [ i ] ) ;
for ( int x = low ; x <= high ; x ++ ) if ( s . find ( x ) == s . end ( ) ) cout << x << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 3 , 5 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int low = 1 , high = 10 ; printMissing ( arr , n , low , high ) ; return 0 ; }
int find ( int a [ ] , int b [ ] , int k , int n1 , int n2 ) {
unordered_set < int > s ; for ( int i = 0 ; i < n2 ; i ++ ) s . insert ( b [ i ] ) ;
int missing = 0 ; for ( int i = 0 ; i < n1 ; i ++ ) { if ( s . find ( a [ i ] ) == s . end ( ) ) missing ++ ; if ( missing == k ) return a [ i ] ; } return -1 ; }
int main ( ) { int a [ ] = { 0 , 2 , 4 , 6 , 8 , 10 , 12 , 14 , 15 } ; int b [ ] = { 4 , 10 , 6 , 8 , 12 } ; int n1 = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int n2 = sizeof ( b ) / sizeof ( b [ 0 ] ) ; int k = 3 ; cout << find ( a , b , k , n1 , n2 ) ; return 0 ; }
void findString ( string S , int N ) {
int amounts [ 26 ] ;
for ( int i = 0 ; i < S . length ( ) ; i ++ ) { amounts [ int ( S [ i ] ) - 97 ] ++ ; } int count = 0 ;
for ( int i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) count ++ ; }
if ( count > N ) { cout << " - 1" ; }
else { string ans = " " ; int high = 100001 ; int low = 0 ; int mid , total ;
while ( ( high - low ) > 1 ) { total = 0 ;
mid = ( high + low ) / 2 ;
for ( int i = 0 ; i < 26 ; i ++ ) {
if ( amounts [ i ] > 0 ) { total += ( amounts [ i ] - 1 ) / mid + 1 ; } }
if ( total <= N ) { high = mid ; } else { low = mid ; } } cout << high << " ▁ " ; total = 0 ;
for ( int i = 0 ; i < 26 ; i ++ ) { if ( amounts [ i ] > 0 ) { total += ( amounts [ i ] - 1 ) / high + 1 ; for ( int j = 0 ; j < ( ( amounts [ i ] - 1 ) / high + 1 ) ; j ++ ) {
ans += char ( i + 97 ) ; } } }
for ( int i = total ; i < N ; i ++ ) { ans += ' a ' ; } reverse ( ans . begin ( ) , ans . end ( ) ) ;
cout << ans ; } }
int main ( ) { string S = " toffee " ; int K = 4 ; findString ( S , K ) ; return 0 ; }
void printFirstRepeating ( int arr [ ] , int n ) {
int min = -1 ;
set < int > myset ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( myset . find ( arr [ i ] ) != myset . end ( ) ) min = i ;
else myset . insert ( arr [ i ] ) ; }
if ( min != -1 ) cout << " The ▁ first ▁ repeating ▁ element ▁ is ▁ " << arr [ min ] ; else cout << " There ▁ are ▁ no ▁ repeating ▁ elements " ; }
int main ( ) { int arr [ ] = { 10 , 5 , 3 , 4 , 3 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printFirstRepeating ( arr , n ) ; }
void printFirstRepeating ( int arr [ ] , int n ) {
int k = 0 ;
int max = n ; for ( int i = 0 ; i < n ; i ++ ) if ( max < arr [ i ] ) max = arr [ i ] ;
int a [ max + 1 ] = { } ;
int b [ max + 1 ] = { } ; for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ arr [ i ] ] ) { b [ arr [ i ] ] = 1 ; k = 1 ; continue ; } else
a [ arr [ i ] ] = i ; } if ( k == 0 ) cout << " No ▁ repeating ▁ element ▁ found " << endl ; else { int min = max + 1 ;
for ( int i = 0 ; i < max + 1 ; i ++ ) if ( a [ i ] && min > a [ i ] && b [ i ] ) min = a [ i ] ; cout << arr [ min ] ; } cout << endl ; }
int main ( ) { int arr [ ] = { 10 , 5 , 3 , 4 , 3 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printFirstRepeating ( arr , n ) ; }
int printKDistinct ( int arr [ ] , int n , int k ) { int dist_count = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return -1 ; }
int main ( ) { int ar [ ] = { 1 , 2 , 1 , 3 , 4 , 2 } ; int n = sizeof ( ar ) / sizeof ( ar [ 0 ] ) ; int k = 2 ; cout << printKDistinct ( ar , n , k ) ; return 0 ; }
void countSubarrays ( int A [ ] , int N ) {
int res = 0 ;
int curr = A [ 0 ] ; vector < int > cnt = { 1 } ; for ( int c = 1 ; c < N ; c ++ ) {
if ( A == curr )
cnt [ cnt . size ( ) - 1 ] ++ ; else
curr = A ; cnt . push_back ( 1 ) ; }
for ( int i = 1 ; i < cnt . size ( ) ; i ++ ) {
res += min ( cnt [ i - 1 ] , cnt [ i ] ) ; } cout << ( res - 1 ) ; }
int main ( ) {
int A [ ] = { 1 , 1 , 0 , 0 , 1 , 0 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
countSubarrays ( A , N ) ; return 0 ; }
struct Node { int val ; Node * left , * right ; } ;
struct Node * newNode ( int data ) { struct Node * temp = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; temp -> val = data ; temp -> left = NULL ; temp -> right = NULL ; return temp ; }
bool isEvenOddBinaryTree ( Node * root ) { if ( root == NULL ) return true ;
queue < Node * > q ; q . push ( root ) ;
int level = 0 ;
while ( ! q . empty ( ) ) {
int size = q . size ( ) ; for ( int i = 0 ; i < size ; i ++ ) { Node * node = q . front ( ) ;
if ( level % 2 == 0 ) { if ( node -> val % 2 == 1 ) return false ; } else if ( level % 2 == 1 ) { if ( node -> val % 2 == 0 ) return true ; }
if ( node -> left != NULL ) { q . push ( node -> left ) ; } if ( node -> right != NULL ) { q . push ( node -> right ) ; } }
level ++ ; } return true ; }
int main ( ) {
Node * root = NULL ; root = newNode ( 2 ) ; root -> left = newNode ( 3 ) ; root -> right = newNode ( 9 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 10 ) ; root -> right -> right = newNode ( 6 ) ;
if ( isEvenOddBinaryTree ( root ) ) cout << " YES " ; else cout << " NO " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int findMaxLen ( vector < int > & a ) {
int n = a . size ( ) ;
int freq [ n + 1 ] ; memset ( freq , 0 , sizeof freq ) ; for ( int i = 0 ; i < n ; ++ i ) { freq [ a [ i ] ] ++ ; } int maxFreqElement = INT_MIN ; int maxFreqCount = 1 ; for ( int i = 1 ; i <= n ; ++ i ) {
if ( freq [ i ] > maxFreqElement ) { maxFreqElement = freq [ i ] ; maxFreqCount = 1 ; }
else if ( freq [ i ] == maxFreqElement ) maxFreqCount ++ ; } int ans ;
if ( maxFreqElement == 1 ) ans = 0 ; else {
ans = ( ( n - maxFreqCount ) / ( maxFreqElement - 1 ) ) ; }
return ans ; }
int main ( ) { vector < int > a = { 1 , 2 , 1 , 2 } ; cout << findMaxLen ( a ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
int MaxUtil ( int * st , int ss , int se , int l , int r , int node ) {
if ( l <= ss && r >= se )
return st [ node ] ;
if ( se < l ss > r ) return -1 ;
int mid = getMid ( ss , se ) ; return max ( MaxUtil ( st , ss , mid , l , r , 2 * node + 1 ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 2 ) ) ; }
int getMax ( int * st , int n , int l , int r ) {
if ( l < 0 r > n - 1 l > r ) { printf ( " Invalid ▁ Input " ) ; return -1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
int constructSTUtil ( int arr [ ] , int ss , int se , int * st , int si ) {
if ( ss == se ) { st [ si ] = arr [ ss ] ; return arr [ ss ] ; }
int mid = getMid ( ss , se ) ;
int * constructST ( int arr [ ] , int n ) {
int x = ( int ) ( ceil ( log2 ( n ) ) ) ;
int max_size = 2 * ( int ) pow ( 2 , x ) - 1 ;
int * st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
int main ( ) { int arr [ ] = { 5 , 2 , 3 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int * st = constructST ( arr , n ) ; vector < vector < int > > Q = { { 1 , 3 } , { 0 , 2 } } ; for ( int i = 0 ; i < Q . size ( ) ; i ++ ) { int max = getMax ( st , n , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) ; int ok = 0 ; for ( int i = 30 ; i >= 0 ; i -- ) { if ( ( max & ( 1 << i ) ) != 0 ) ok = 1 ; if ( ! ok ) continue ; max |= ( 1 << i ) ; } cout << max << " ▁ " ; } return 0 ; }
int calculate ( int a [ ] , int n ) {
sort ( a , a + n ) ; int count = 1 ; int answer = 0 ;
for ( int i = 1 ; i < n ; i ++ ) { if ( a [ i ] == a [ i - 1 ] ) {
count += 1 ; } else {
answer = answer + ( count * ( count - 1 ) ) / 2 ; count = 1 ; } } answer = answer + ( count * ( count - 1 ) ) / 2 ; return answer ; }
int main ( ) { int a [ ] = { 1 , 2 , 1 , 2 , 4 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ;
cout << calculate ( a , n ) ; return 0 ; }
int calculate ( int a [ ] , int n ) {
int * maximum = max_element ( a , a + n ) ;
int frequency [ * maximum + 1 ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) {
frequency [ a [ i ] ] += 1 ; } int answer = 0 ;
for ( int i = 0 ; i < ( * maximum ) + 1 ; i ++ ) {
answer = answer + frequency [ i ] * ( frequency [ i ] - 1 ) ; } return answer / 2 ; }
int main ( ) { int a [ ] = { 1 , 2 , 1 , 2 , 4 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ;
cout << ( calculate ( a , n ) ) ; }
int findSubArray ( int arr [ ] , int n ) { int sum = 0 ; int maxsize = -1 , startindex ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? -1 : 1 ;
for ( int j = i + 1 ; j < n ; j ++ ) { ( arr [ j ] == 0 ) ? ( sum += -1 ) : ( sum += 1 ) ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } if ( maxsize == -1 ) cout << " No ▁ such ▁ subarray " ; else cout << startindex << " ▁ to ▁ " << startindex + maxsize - 1 ; return maxsize ; }
int main ( ) { int arr [ ] = { 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findSubArray ( arr , size ) ; return 0 ; }
int findMax ( int arr [ ] , int low , int high ) {
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid == 0 && arr [ mid ] > arr [ mid + 1 ] ) { return arr [ mid ] ; }
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] && mid > 0 && arr [ mid ] > arr [ mid - 1 ] ) { return arr [ mid ] ; }
if ( arr [ low ] > arr [ mid ] ) { return findMax ( arr , low , mid - 1 ) ; } else { return findMax ( arr , mid + 1 , high ) ; } }
int main ( ) { int arr [ ] = { 6 , 5 , 4 , 3 , 2 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMax ( arr , 0 , n - 1 ) ; return 0 ; }
int ternarySearch ( int l , int r , int key , int ar [ ] ) { while ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
r = mid1 - 1 ; } else if ( key > ar [ mid2 ] ) {
l = mid2 + 1 ; } else {
l = mid1 + 1 ; r = mid2 - 1 ; } }
return -1 ; }
int ar [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
cout << " Index ▁ of ▁ " << key << " ▁ is ▁ " << p << endl ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
cout << " Index ▁ of ▁ " << key << " ▁ is ▁ " << p ; }
int majorityNumber ( int arr [ ] , int n ) { int ans = -1 ; unordered_map < int , int > freq ; for ( int i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; if ( freq [ arr [ i ] ] > n / 2 ) ans = arr [ i ] ; } return ans ; }
int main ( ) { int a [ ] = { 2 , 2 , 1 , 1 , 1 , 2 , 2 } ; int n = sizeof ( a ) / sizeof ( int ) ; cout << majorityNumber ( a , n ) ; return 0 ; }
int search ( int arr [ ] , int l , int h , int key ) { if ( l > h ) return -1 ; int mid = ( l + h ) / 2 ; if ( arr [ mid ] == key ) return mid ;
if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
return search ( arr , mid + 1 , h , key ) ; }
if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
int main ( ) { int arr [ ] = { 4 , 5 , 6 , 7 , 8 , 9 , 1 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int key = 6 ; int i = search ( arr , 0 , n - 1 , key ) ; if ( i != -1 ) cout << " Index : ▁ " << i << endl ; else cout << " Key ▁ not ▁ found " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int findMin ( int arr [ ] , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
int main ( ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr1 , 0 , n1 - 1 ) << endl ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr2 , 0 , n2 - 1 ) << endl ; int arr3 [ ] = { 1 } ; int n3 = sizeof ( arr3 ) / sizeof ( arr3 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr3 , 0 , n3 - 1 ) << endl ; int arr4 [ ] = { 1 , 2 } ; int n4 = sizeof ( arr4 ) / sizeof ( arr4 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr4 , 0 , n4 - 1 ) << endl ; int arr5 [ ] = { 2 , 1 } ; int n5 = sizeof ( arr5 ) / sizeof ( arr5 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr5 , 0 , n5 - 1 ) << endl ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = sizeof ( arr6 ) / sizeof ( arr6 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr6 , 0 , n6 - 1 ) << endl ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = sizeof ( arr7 ) / sizeof ( arr7 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr7 , 0 , n7 - 1 ) << endl ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = sizeof ( arr8 ) / sizeof ( arr8 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr8 , 0 , n8 - 1 ) << endl ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = sizeof ( arr9 ) / sizeof ( arr9 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr9 , 0 , n9 - 1 ) << endl ; return 0 ; }
int findMin ( int arr [ ] , int low , int high ) { while ( low < high ) { int mid = low + ( high - low ) / 2 ; if ( arr [ mid ] == arr [ high ] ) high -- ; else if ( arr [ mid ] > arr [ high ] ) low = mid + 1 ; else high = mid ; } return arr [ high ] ; }
int main ( ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr1 , 0 , n1 - 1 ) << endl ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr2 , 0 , n2 - 1 ) << endl ; int arr3 [ ] = { 1 } ; int n3 = sizeof ( arr3 ) / sizeof ( arr3 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr3 , 0 , n3 - 1 ) << endl ; int arr4 [ ] = { 1 , 2 } ; int n4 = sizeof ( arr4 ) / sizeof ( arr4 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr4 , 0 , n4 - 1 ) << endl ; int arr5 [ ] = { 2 , 1 } ; int n5 = sizeof ( arr5 ) / sizeof ( arr5 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr5 , 0 , n5 - 1 ) << endl ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = sizeof ( arr6 ) / sizeof ( arr6 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr6 , 0 , n6 - 1 ) << endl ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = sizeof ( arr7 ) / sizeof ( arr7 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr7 , 0 , n7 - 1 ) << endl ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = sizeof ( arr8 ) / sizeof ( arr8 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr8 , 0 , n8 - 1 ) << endl ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = sizeof ( arr9 ) / sizeof ( arr9 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr9 , 0 , n9 - 1 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countPairs ( int * a , int n , int mid ) { int res = 0 ; for ( int i = 0 ; i < n ; ++ i )
res += upper_bound ( a + i , a + n , a [ i ] + mid ) - ( a + i + 1 ) ; return res ; }
int kthDiff ( int a [ ] , int n , int k ) {
sort ( a , a + n ) ;
int low = a [ 1 ] - a [ 0 ] ; for ( int i = 1 ; i <= n - 2 ; ++ i ) low = min ( low , a [ i + 1 ] - a [ i ] ) ;
int high = a [ n - 1 ] - a [ 0 ] ;
while ( low < high ) { int mid = ( low + high ) >> 1 ; if ( countPairs ( a , n , mid ) < k ) low = mid + 1 ; else high = mid ; } return low ; }
int main ( ) { int k = 3 ; int a [ ] = { 1 , 2 , 3 , 4 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << kthDiff ( a , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void print2Smallest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { cout << " ▁ Invalid ▁ Input ▁ " ; return ; } first = second = INT_MAX ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == INT_MAX ) cout << " There ▁ is ▁ no ▁ second ▁ smallest ▁ element STRNEWLINE " ; else cout << " The ▁ smallest ▁ element ▁ is ▁ " << first << " ▁ and ▁ second ▁ " " Smallest ▁ element ▁ is ▁ " << second << endl ; }
int main ( ) { int arr [ ] = { 12 , 13 , 1 , 10 , 34 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; print2Smallest ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  1000
int tree [ 4 * MAX ] ;
int arr [ MAX ] ;
int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
int lcm ( int a , int b ) { return a * b / gcd ( a , b ) ; }
void build ( int node , int start , int end ) {
if ( start == end ) { tree [ node ] = arr [ start ] ; return ; } int mid = ( start + end ) / 2 ;
build ( 2 * node , start , mid ) ; build ( 2 * node + 1 , mid + 1 , end ) ;
int left_lcm = tree [ 2 * node ] ; int right_lcm = tree [ 2 * node + 1 ] ; tree [ node ] = lcm ( left_lcm , right_lcm ) ; }
int query ( int node , int start , int end , int l , int r ) {
if ( end < l start > r ) return 1 ;
if ( l <= start && r >= end ) return tree [ node ] ;
int mid = ( start + end ) / 2 ; int left_lcm = query ( 2 * node , start , mid , l , r ) ; int right_lcm = query ( 2 * node + 1 , mid + 1 , end , l , r ) ; return lcm ( left_lcm , right_lcm ) ; }
int main ( ) {
arr [ 0 ] = 5 ; arr [ 1 ] = 7 ; arr [ 2 ] = 5 ; arr [ 3 ] = 2 ; arr [ 4 ] = 10 ; arr [ 5 ] = 12 ; arr [ 6 ] = 11 ; arr [ 7 ] = 17 ; arr [ 8 ] = 14 ; arr [ 9 ] = 1 ; arr [ 10 ] = 44 ;
build ( 1 , 0 , 10 ) ;
cout << query ( 1 , 0 , 10 , 2 , 5 ) << endl ;
cout << query ( 1 , 0 , 10 , 5 , 10 ) << endl ;
cout << query ( 1 , 0 , 10 , 0 , 10 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int M = 1000000007 ; int waysOfDecoding ( string s ) { vector < int > dp ( ( int ) s . size ( ) + 1 ) ; dp [ 0 ] = 1 ;
dp [ 1 ] = s [ 0 ] == ' * ' ? 9 : s [ 0 ] == '0' ? 0 : 1 ;
for ( int i = 1 ; i < ( int ) s . size ( ) ; i ++ ) {
if ( s [ i ] == ' * ' ) { dp [ i + 1 ] = 9 * dp [ i ] ;
if ( s [ i - 1 ] == '1' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 9 * dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == '2' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 6 * dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' * ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + 15 * dp [ i - 1 ] ) % M ; } else {
dp [ i + 1 ] = s [ i ] != '0' ? dp [ i ] : 0 ;
if ( s [ i - 1 ] == '1' ) dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == '2' && s [ i ] <= '6' ) dp [ i + 1 ] = ( dp [ i + 1 ] + dp [ i - 1 ] ) % M ;
else if ( s [ i - 1 ] == ' * ' ) dp [ i + 1 ] = ( dp [ i + 1 ] + ( s [ i ] <= '6' ? 2 : 1 ) * dp [ i - 1 ] ) % M ; } } return dp [ ( int ) s . size ( ) ] ; }
int main ( ) { string s = "12" ; cout << waysOfDecoding ( s ) ; return 0 ; }
int countSubset ( int arr [ ] , int n , int diff ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; sum += diff ; sum = sum / 2 ;
int t [ n + 1 ] [ sum + 1 ] ;
for ( int j = 0 ; j <= sum ; j ++ ) t [ 0 ] [ j ] = 0 ;
for ( int i = 0 ; i <= n ; i ++ ) t [ i ] [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) {
if ( arr [ i - 1 ] > j ) t [ i ] [ j ] = t [ i - 1 ] [ j ] ; else { t [ i ] [ j ] = t [ i - 1 ] [ j ] + t [ i - 1 ] [ j - arr [ i - 1 ] ] ; } } }
return t [ n ] [ sum ] ; }
int main ( ) {
int diff = 1 , n = 4 ; int arr [ ] = { 1 , 1 , 2 , 3 } ;
cout << countSubset ( arr , n , diff ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; float dp [ 105 ] [ 605 ] ;
float find ( int N , int a , int b ) { float probability = 0.0 ;
for ( int i = 1 ; i <= 6 ; i ++ ) dp [ 1 ] [ i ] = 1.0 / 6 ; for ( int i = 2 ; i <= N ; i ++ ) { for ( int j = i ; j <= 6 * i ; j ++ ) { for ( int k = 1 ; k <= 6 ; k ++ ) { dp [ i ] [ j ] = dp [ i ] [ j ] + dp [ i - 1 ] [ j - k ] / 6 ; } } }
for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + dp [ N ] [ sum ] ; return probability ; }
int main ( ) { int N = 4 , a = 13 , b = 17 ; float probability = find ( N , a , b ) ;
cout << fixed << setprecision ( 6 ) << probability ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; Node * left , * right ; Node ( int item ) { data = item ; } } ; int getSum ( Node * root ) ;
int getSumAlternate ( Node * root ) { if ( root == NULL ) return 0 ; int sum = root -> data ; if ( root -> left != NULL ) { sum += getSum ( root -> left -> left ) ; sum += getSum ( root -> left -> right ) ; } if ( root -> right != NULL ) { sum += getSum ( root -> right -> left ) ; sum += getSum ( root -> right -> right ) ; } return sum ; }
int getSum ( Node * root ) { if ( root == NULL ) return 0 ;
return max ( getSumAlternate ( root ) , ( getSumAlternate ( root -> left ) + getSumAlternate ( root -> right ) ) ) ; }
int main ( ) { Node * root = new Node ( 1 ) ; root -> left = new Node ( 2 ) ; root -> right = new Node ( 3 ) ; root -> right -> left = new Node ( 4 ) ; root -> right -> left -> right = new Node ( 5 ) ; root -> right -> left -> right -> left = new Node ( 6 ) ; cout << ( getSum ( root ) ) ; return 0 ; }
bool isSubsetSum ( int arr [ ] , int n , int sum ) {
bool subset [ 2 ] [ sum + 1 ] ; for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) {
if ( j == 0 ) subset [ i % 2 ] [ j ] = true ;
else if ( i == 0 ) subset [ i % 2 ] [ j ] = false ; else if ( arr [ i - 1 ] <= j ) subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] || subset [ ( i + 1 ) % 2 ] [ j ] ; else subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] ; } } return subset [ n % 2 ] [ sum ] ; }
int main ( ) { int arr [ ] = { 6 , 2 , 5 } ; int sum = 7 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; if ( isSubsetSum ( arr , n , sum ) == true ) cout << " There ▁ exists ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else cout << " No ▁ subset ▁ exists ▁ with ▁ given ▁ sum " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMaxSum ( int arr [ ] , int n ) { int res = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) { int prefix_sum = arr [ i ] ; for ( int j = 0 ; j < i ; j ++ ) prefix_sum += arr [ j ] ; int suffix_sum = arr [ i ] ; for ( int j = n - 1 ; j > i ; j -- ) suffix_sum += arr [ j ] ; if ( prefix_sum == suffix_sum ) res = max ( res , prefix_sum ) ; } return res ; }
int main ( ) { int arr [ ] = { -2 , 5 , 3 , 1 , 2 , 6 , -4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMaxSum ( arr , n ) ; return 0 ; }
int findMaxSum ( int arr [ ] , int n ) {
int preSum [ n ] ;
int suffSum [ n ] ;
int ans = INT_MIN ;
preSum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) preSum [ i ] = preSum [ i - 1 ] + arr [ i ] ;
suffSum [ n - 1 ] = arr [ n - 1 ] ; if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) ans = max ( ans , preSum [ n - 1 ] ) ; for ( int i = n - 2 ; i >= 0 ; i -- ) { suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] ; if ( suffSum [ i ] == preSum [ i ] ) ans = max ( ans , preSum [ i ] ) ; } return ans ; }
int main ( ) { int arr [ ] = { -2 , 5 , 3 , 1 , 2 , 6 , -4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMaxSum ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMaxSum ( int arr [ ] , int n ) { int sum = accumulate ( arr , arr + n , 0 ) ; int prefix_sum = 0 , res = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) { prefix_sum += arr [ i ] ; if ( prefix_sum == sum ) res = max ( res , prefix_sum ) ; sum -= arr [ i ] ; } return res ; }
int main ( ) { int arr [ ] = { -2 , 5 , 3 , 1 , 2 , 6 , -4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMaxSum ( arr , n ) ; return 0 ; }
void findMajority ( int arr [ ] , int n ) { int maxCount = 0 ;
int index = -1 ; for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ i ] == arr [ j ] ) count ++ ; }
if ( count > maxCount ) { maxCount = count ; index = i ; } }
if ( maxCount > n / 2 ) cout << arr [ index ] << endl ; else cout << " No ▁ Majority ▁ Element " << endl ; }
int main ( ) { int arr [ ] = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
findMajority ( arr , n ) ; return 0 ; }
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; for ( int i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
bool isMajority ( int a [ ] , int size , int cand ) { int count = 0 ; for ( int i = 0 ; i < size ; i ++ ) if ( a [ i ] == cand ) count ++ ; if ( count > size / 2 ) return 1 ; else return 0 ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) cout << " ▁ " << cand << " ▁ " ; else cout << " No ▁ Majority ▁ Element " ; }
int main ( ) { int a [ ] = { 1 , 3 , 3 , 1 , 2 } ; int size = ( sizeof ( a ) ) / sizeof ( a [ 0 ] ) ;
printMajority ( a , size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findMajority ( int arr [ ] , int size ) { unordered_map < int , int > m ; for ( int i = 0 ; i < size ; i ++ ) m [ arr [ i ] ] ++ ; int count = 0 ; for ( auto i : m ) { if ( i . second > size / 2 ) { count = 1 ; cout << " Majority ▁ found ▁ : - ▁ " << i . first << endl ; break ; } } if ( count == 0 ) cout << " No ▁ Majority ▁ element " << endl ; }
int main ( ) { int arr [ ] = { 2 , 2 , 2 , 2 , 5 , 5 , 2 , 3 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
findMajority ( arr , n ) ; return 0 ; }
int majorityElement ( int * arr , int n ) {
sort ( arr , arr + n ) ; int count = 1 , max_ele = -1 , temp = arr [ 0 ] , ele , f = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( temp == arr [ i ] ) { count ++ ; } else { count = 1 ; temp = arr [ i ] ; }
if ( max_ele < count ) { max_ele = count ; ele = arr [ i ] ; if ( max_ele > ( n / 2 ) ) { f = 1 ; break ; } } }
return ( f == 1 ? ele : -1 ) ; }
int main ( ) { int arr [ ] = { 1 , 1 , 2 , 1 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << majorityElement ( arr , n ) ; return 0 ; }
bool isSubsetSum ( int set [ ] , int n , int sum ) {
bool subset [ n + 1 ] [ sum + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ i ] [ 0 ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ 0 ] [ i ] = false ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) { if ( j < set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] ; if ( j >= set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] || subset [ i - 1 ] [ j - set [ i - 1 ] ] ; } }
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) printf ( " % 4d " , subset [ i ] [ j ] ) ; cout << " STRNEWLINE " ; } return subset [ n ] [ sum ] ; }
int main ( ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = sizeof ( set ) / sizeof ( set [ 0 ] ) ; if ( isSubsetSum ( set , n , sum ) == true ) cout << " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else cout << " No ▁ subset ▁ with ▁ given ▁ sum " ; return 0 ; }
int tab [ 2000 ] [ 2000 ] ;
int subsetSum ( int a [ ] , int n , int sum ) {
if ( sum == 0 ) return 1 ; if ( n <= 0 ) return 0 ;
if ( tab [ n - 1 ] [ sum ] != -1 ) return tab [ n - 1 ] [ sum ] ;
if ( a [ n - 1 ] > sum ) return tab [ n - 1 ] [ sum ] = subsetSum ( a , n - 1 , sum ) ; else {
return tab [ n - 1 ] [ sum ] = subsetSum ( a , n - 1 , sum ) || subsetSum ( a , n - 1 , sum - a [ n - 1 ] ) ; } }
int main ( ) { memset ( tab , -1 , sizeof ( tab ) ) ; int n = 5 ; int a [ ] = { 1 , 5 , 3 , 7 , 4 } ; int sum = 12 ; if ( subsetSum ( a , n , sum ) ) { cout << " YES " << endl ; } else cout << " NO " << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int binpow ( int a , int b ) { int res = 1 ; while ( b ) { if ( b & 1 ) res = res * a ; a = a * a ; b /= 2 ; } return res ; }
int find ( int x ) { if ( x == 0 ) return 0 ; int p = log2 ( x ) ; return binpow ( 2 , p + 1 ) - 1 ; }
string getBinary ( int n ) {
string ans = " " ;
while ( n ) { int dig = n % 2 ; ans += to_string ( dig ) ; n /= 2 ; }
return ans ; }
int totalCountDifference ( int n ) {
string ans = getBinary ( n ) ;
int req = 0 ;
for ( int i = 0 ; i < ans . size ( ) ; i ++ ) {
if ( ans [ i ] == '1' ) { req += find ( binpow ( 2 , i ) ) ; } } return req ; }
int N = 5 ;
cout << totalCountDifference ( N ) ; return 0 ; }
int Maximum_Length ( vector < int > a ) {
int counts [ 11 ] = { 0 } ;
int ans = 0 ; for ( int index = 0 ; index < a . size ( ) ; index ++ ) {
counts [ a [ index ] ] += 1 ;
vector < int > k ; for ( auto i : counts ) if ( i != 0 ) k . push_back ( i ) ; sort ( k . begin ( ) , k . end ( ) ) ;
if ( k . size ( ) == 1 || ( k [ 0 ] == k [ k . size ( ) - 2 ] && k . back ( ) - k [ k . size ( ) - 2 ] == 1 ) || ( k [ 0 ] == 1 and k [ 1 ] == k . back ( ) ) ) ans = index ; }
return ans + 1 ; }
int main ( ) { vector < int > a = { 1 , 1 , 1 , 2 , 2 , 2 } ; cout << ( Maximum_Length ( a ) ) ; }
void print_gcd_online ( int n , int m , int query [ ] [ 2 ] , int arr [ ] ) {
int max_gcd = 0 ; int i = 0 ;
for ( i = 0 ; i < n ; i ++ ) max_gcd = __gcd ( max_gcd , arr [ i ] ) ;
for ( i = 0 ; i < m ; i ++ ) {
query [ i ] [ 0 ] -- ;
arr [ query [ i ] [ 0 ] ] /= query [ i ] [ 1 ] ;
max_gcd = __gcd ( arr [ query [ i ] [ 0 ] ] , max_gcd ) ;
cout << max_gcd << endl ; } }
int main ( ) { int n = 3 ; int m = 3 ; int query [ m ] [ 2 ] ; int arr [ ] = { 36 , 24 , 72 } ; query [ 0 ] [ 0 ] = 1 ; query [ 0 ] [ 1 ] = 3 ; query [ 1 ] [ 0 ] = 3 ; query [ 1 ] [ 1 ] = 12 ; query [ 2 ] [ 0 ] = 2 ; query [ 2 ] [ 1 ] = 4 ; print_gcd_online ( n , m , query , arr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  1000000
bool prime [ MAX + 1 ] ;
int sum [ MAX + 1 ] ;
void SieveOfEratosthenes ( ) {
memset ( prime , true , sizeof ( prime ) ) ; memset ( sum , 0 , sizeof ( sum ) ) ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= MAX ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( int i = 1 ; i <= MAX ; i ++ ) { if ( prime [ i ] == true ) sum [ i ] = 1 ; sum [ i ] += sum [ i - 1 ] ; } }
int main ( ) {
SieveOfEratosthenes ( ) ;
int l = 3 , r = 9 ;
int c = ( sum [ r ] - sum [ l - 1 ] ) ;
cout << " Count : ▁ " << c << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float area ( float r ) {
if ( r < 0 ) return -1 ;
float area = 3.14 * pow ( r / ( 2 * sqrt ( 2 ) ) , 2 ) ; return area ; }
int main ( ) { float a = 5 ; cout << area ( a ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  100005
bool prime [ N ] ; void SieveOfEratosthenes ( ) { memset ( prime , true , sizeof ( prime ) ) ; prime [ 1 ] = false ; for ( int p = 2 ; p * p < N ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i < N ; i += p ) prime [ i ] = false ; } } }
int almostPrimes ( int n ) {
int ans = 0 ;
for ( int i = 6 ; i <= n ; i ++ ) {
int c = 0 ; for ( int j = 2 ; j * j <= i ; j ++ ) { if ( i % j == 0 ) {
if ( j * j == i ) { if ( prime [ j ] ) c ++ ; } else { if ( prime [ j ] ) c ++ ; if ( prime [ i / j ] ) c ++ ; } } }
if ( c == 2 ) ans ++ ; } return ans ; }
int main ( ) { SieveOfEratosthenes ( ) ; int n = 21 ; cout << almostPrimes ( n ) ; return 0 ; }
int sumOfDigitsSingle ( int x ) { int ans = 0 ; while ( x ) { ans += x % 10 ; x /= 10 ; } return ans ; }
int closest ( int x ) { int ans = 0 ; while ( ans * 10 + 9 <= x ) ans = ans * 10 + 9 ; return ans ; } int sumOfDigitsTwoParts ( int N ) { int A = closest ( N ) ; return sumOfDigitsSingle ( A ) + sumOfDigitsSingle ( N - A ) ; }
int main ( ) { int N = 35 ; cout << sumOfDigitsTwoParts ( N ) ; return 0 ; }
bool isPrime ( int p ) {
long long checkNumber = pow ( 2 , p ) - 1 ;
long long nextval = 4 % checkNumber ;
for ( int i = 1 ; i < p - 1 ; i ++ ) nextval = ( nextval * nextval - 2 ) % checkNumber ;
return ( nextval == 0 ) ; }
int p = 7 ; long long checkNumber = pow ( 2 , p ) - 1 ; if ( isPrime ( p ) ) cout << checkNumber << " ▁ is ▁ Prime . " ; else cout << checkNumber << " ▁ is ▁ not ▁ Prime . " ; return 0 ; }
bool sieve ( int n , bool prime [ ] ) { for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = false ; } } } void printSophieGermanNumber ( int n ) {
bool prime [ 2 * n + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ; sieve ( 2 * n + 1 , prime ) ; for ( int i = 2 ; i <= n ; ++ i ) {
if ( prime [ i ] && prime [ 2 * i + 1 ] ) cout << i << " ▁ " ; } }
int main ( ) { int n = 25 ; printSophieGermanNumber ( n ) ; return 0 ; }
float ucal ( float u , int n ) { if ( n == 0 ) return 1 ; float temp = u ; for ( int i = 1 ; i <= n / 2 ; i ++ ) temp = temp * ( u - i ) ; for ( int i = 1 ; i < n / 2 ; i ++ ) temp = temp * ( u + i ) ; return temp ; }
int fact ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
int n = 6 ; float x [ ] = { 25 , 26 , 27 , 28 , 29 , 30 } ;
float y [ n ] [ n ] ; y [ 0 ] [ 0 ] = 4.000 ; y [ 1 ] [ 0 ] = 3.846 ; y [ 2 ] [ 0 ] = 3.704 ; y [ 3 ] [ 0 ] = 3.571 ; y [ 4 ] [ 0 ] = 3.448 ; y [ 5 ] [ 0 ] = 3.333 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < n - i ; j ++ ) y [ j ] [ i ] = y [ j + 1 ] [ i - 1 ] - y [ j ] [ i - 1 ] ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n - i ; j ++ ) cout << setw ( 4 ) << y [ i ] [ j ] << " TABSYMBOL " ; cout << endl ; }
float value = 27.4 ;
float sum = ( y [ 2 ] [ 0 ] + y [ 3 ] [ 0 ] ) / 2 ;
int k ;
k = n / 2 ; else
float u = ( value - x [ k ] ) / ( x [ 1 ] - x [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) { if ( i % 2 ) sum = sum + ( ( u - 0.5 ) * ucal ( u , i - 1 ) * y [ k ] [ i ] ) / fact ( i ) ; else sum = sum + ( ucal ( u , i ) * ( y [ k ] [ i ] + y [ -- k ] [ i ] ) / ( fact ( i ) * 2 ) ) ; } cout << " Value ▁ at ▁ " << value << " ▁ is ▁ " << sum << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE int fibonacci ( int n ) { int a = 0 , b = 1 , c ; if ( n <= 1 ) return n ; for ( int i = 2 ; i <= n ; i ++ ) { c = a + b ; a = b ; b = c ; } return c ; }
bool isMultipleOf10 ( int n ) { int f = fibonacci ( 30 ) ; return ( f % 10 == 0 ) ; }
int main ( ) { int n = 30 ; if ( isMultipleOf10 ( n ) ) printf ( " Yes STRNEWLINE " ) ; else printf ( " No STRNEWLINE " ) ; }
bool powerOf2 ( int n ) {
if ( n == 1 ) return true ;
else if ( n % 2 != 0 n == 0 ) return false ;
return powerOf2 ( n / 2 ) ; }
int main ( ) {
int n = 64 ;
int m = 12 ; if ( powerOf2 ( n ) == 1 ) cout << " True " << endl ; else cout << " False " << endl ; if ( powerOf2 ( m ) == 1 ) cout << " True " << endl ; else cout << " False " << endl ; }
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerOfTwo ( 64 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
bool isPowerofTwo ( long long n ) { if ( n == 0 ) return 0 ; if ( ( n & ( ~ ( n - 1 ) ) ) == n ) return 1 ; return 0 ; }
int main ( ) { isPowerofTwo ( 30 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerofTwo ( 128 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
int nextPowerOf2 ( int n ) {
int p = 1 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ;
while ( p < n ) p <<= 1 ; return p ; }
int memoryUsed ( int arr [ ] , int n ) {
int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
int nearest = nextPowerOf2 ( sum ) ; return nearest ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << memoryUsed ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int toggleKthBit ( int n , int k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
int main ( ) { int n = 5 , k = 1 ; cout << toggleKthBit ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; unsigned int nextPowerOf2 ( unsigned int n ) { unsigned count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
int main ( ) { unsigned int n = 0 ; cout << nextPowerOf2 ( n ) ; return 0 ; }
int gcd ( int A , int B ) { if ( B == 0 ) return A ; return gcd ( B , A % B ) ; }
int lcm ( int A , int B ) { return ( A * B ) / gcd ( A , B ) ; }
int checkA ( int A , int B , int C , int K ) {
int start = 1 ; int end = K ;
int ans = -1 ; while ( start <= end ) { int mid = ( start + end ) / 2 ; int value = A * mid ; int divA = mid - 1 ; int divB = ( value % B == 0 ) ? value / B - 1 : value / B ; int divC = ( value % C == 0 ) ? value / C - 1 : value / C ; int divAB = ( value % lcm ( A , B ) == 0 ) ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ; int divBC = ( value % lcm ( C , B ) == 0 ) ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ; int divAC = ( value % lcm ( A , C ) == 0 ) ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ; int divABC = ( value % lcm ( A , lcm ( B , C ) ) == 0 ) ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ;
int elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem == ( K - 1 ) ) { ans = value ; break ; }
else if ( elem > ( K - 1 ) ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
int checkB ( int A , int B , int C , int K ) {
int start = 1 ; int end = K ;
int ans = -1 ; while ( start <= end ) { int mid = ( start + end ) / 2 ; int value = B * mid ; int divB = mid - 1 ; int divA = ( value % A == 0 ) ? value / A - 1 : value / A ; int divC = ( value % C == 0 ) ? value / C - 1 : value / C ; int divAB = ( value % lcm ( A , B ) == 0 ) ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ; int divBC = ( value % lcm ( C , B ) == 0 ) ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ; int divAC = ( value % lcm ( A , C ) == 0 ) ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ; int divABC = ( value % lcm ( A , lcm ( B , C ) ) == 0 ) ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ;
int elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem == ( K - 1 ) ) { ans = value ; break ; }
else if ( elem > ( K - 1 ) ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
int checkC ( int A , int B , int C , int K ) {
int start = 1 ; int end = K ;
int ans = -1 ; while ( start <= end ) { int mid = ( start + end ) / 2 ; int value = C * mid ; int divC = mid - 1 ; int divB = ( value % B == 0 ) ? value / B - 1 : value / B ; int divA = ( value % A == 0 ) ? value / A - 1 : value / A ; int divAB = ( value % lcm ( A , B ) == 0 ) ? value / lcm ( A , B ) - 1 : value / lcm ( A , B ) ; int divBC = ( value % lcm ( C , B ) == 0 ) ? value / lcm ( C , B ) - 1 : value / lcm ( C , B ) ; int divAC = ( value % lcm ( A , C ) == 0 ) ? value / lcm ( A , C ) - 1 : value / lcm ( A , C ) ; int divABC = ( value % lcm ( A , lcm ( B , C ) ) == 0 ) ? value / lcm ( A , lcm ( B , C ) ) - 1 : value / lcm ( A , lcm ( B , C ) ) ;
int elem = divA + divB + divC - divAC - divBC - divAB + divABC ; if ( elem == ( K - 1 ) ) { ans = value ; break ; }
else if ( elem > ( K - 1 ) ) { end = mid - 1 ; }
else { start = mid + 1 ; } } return ans ; }
int findKthMultiple ( int A , int B , int C , int K ) {
int res = checkA ( A , B , C , K ) ;
if ( res == -1 ) res = checkB ( A , B , C , K ) ;
if ( res == -1 ) res = checkC ( A , B , C , K ) ; return res ; }
int main ( ) { int A = 2 , B = 4 , C = 5 , K = 5 ; cout << findKthMultiple ( A , B , C , K ) ; return 0 ; }
void variationStalinsort ( vector < int > arr ) { int j = 0 ; while ( true ) { int moved = 0 ; for ( int i = 0 ; i < ( arr . size ( ) - 1 - j ) ; i ++ ) { if ( arr [ i ] > arr [ i + 1 ] ) {
vector < int > :: iterator index ; int temp ; index = arr . begin ( ) + i + 1 ; temp = arr [ i + 1 ] ; arr . erase ( index ) ; arr . insert ( arr . begin ( ) + moved , temp ) ; moved ++ ; } } j ++ ; if ( moved == 0 ) { break ; } } for ( int i = 0 ; i < arr . size ( ) ; i ++ ) { cout << arr [ i ] << " , ▁ " ; } }
int main ( ) { vector < int > arr = { 2 , 1 , 4 , 3 , 6 , 5 , 8 , 7 , 10 , 9 } ;
variationStalinsort ( arr ) ; }
void printArray ( int arr [ ] , int N ) {
for ( int i = 0 ; i < N ; i ++ ) { cout << arr [ i ] << ' ▁ ' ; } }
void sortArray ( int arr [ ] , int N ) {
for ( int i = 0 ; i < N ; ) {
if ( arr [ i ] == i + 1 ) { i ++ ; }
else { swap ( & arr [ i ] , & arr [ arr [ i ] - 1 ] ) ; } } }
int main ( ) { int arr [ ] = { 2 , 1 , 5 , 3 , 4 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
sortArray ( arr , N ) ;
printArray ( arr , N ) ; return 0 ; }
int maximum ( int value [ ] , int weight [ ] , int weight1 , int flag , int K , int index , int val_len ) {
if ( index >= val_len ) { return 0 ; }
if ( flag == K ) {
int skip = maximum ( value , weight , weight1 , flag , K , index + 1 , val_len ) ; int full = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 , val_len ) ; }
return max ( full , skip ) ; }
else {
int skip = maximum ( value , weight , weight1 , flag , K , index + 1 , val_len ) ; int full = 0 ; int half = 0 ;
if ( weight [ index ] <= weight1 ) { full = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] , flag , K , index + 1 , val_len ) ; }
if ( weight [ index ] / 2 <= weight1 ) { half = value [ index ] + maximum ( value , weight , weight1 - weight [ index ] / 2 , flag , K , index + 1 , val_len ) ; }
return max ( full , max ( skip , half ) ) ; } }
int main ( ) { int value [ ] = { 17 , 20 , 10 , 15 } ; int weight [ ] = { 4 , 2 , 7 , 5 } ; int K = 1 ; int W = 4 ; int val_len = sizeof ( value ) / sizeof ( value [ 0 ] ) ; cout << ( maximum ( value , weight , W , 0 , K , 0 , val_len ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  1005
struct Node { int data ; Node * left , * right ; } ;
Node * newNode ( int data ) { Node * node = new Node ( ) ; node -> data = data ; node -> left = node -> right = NULL ; return node ; }
int dp [ N ] [ 5 ] [ 5 ] ;
int minDominatingSet ( Node * root , int covered , int compulsory ) {
if ( ! root ) return 0 ;
if ( ! root -> left and ! root -> right and ! covered ) compulsory = true ;
if ( dp [ root -> data ] [ covered ] [ compulsory ] != -1 ) return dp [ root -> data ] [ covered ] [ compulsory ] ;
if ( compulsory ) {
return dp [ root -> data ] [ covered ] [ compulsory ] = 1 + minDominatingSet ( root -> left , 1 , 0 ) + minDominatingSet ( root -> right , 1 , 0 ) ; }
if ( covered ) { return dp [ root -> data ] [ covered ] [ compulsory ] = min ( 1 + minDominatingSet ( root -> left , 1 , 0 ) + minDominatingSet ( root -> right , 1 , 0 ) , minDominatingSet ( root -> left , 0 , 0 ) + minDominatingSet ( root -> right , 0 , 0 ) ) ; }
int ans = 1 + minDominatingSet ( root -> left , 1 , 0 ) + minDominatingSet ( root -> right , 1 , 0 ) ; if ( root -> left ) { ans = min ( ans , minDominatingSet ( root -> left , 0 , 1 ) + minDominatingSet ( root -> right , 0 , 0 ) ) ; } if ( root -> right ) { ans = min ( ans , minDominatingSet ( root -> left , 0 , 0 ) + minDominatingSet ( root -> right , 0 , 1 ) ) ; }
return dp [ root -> data ] [ covered ] [ compulsory ] = ans ; }
signed main ( ) {
Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> left -> left = newNode ( 3 ) ; root -> left -> right = newNode ( 4 ) ; root -> left -> left -> left = newNode ( 5 ) ; root -> left -> left -> left -> left = newNode ( 6 ) ; root -> left -> left -> left -> right = newNode ( 7 ) ; root -> left -> left -> left -> right -> right = newNode ( 10 ) ; root -> left -> left -> left -> left -> left = newNode ( 8 ) ; root -> left -> left -> left -> left -> right = newNode ( 9 ) ; cout << minDominatingSet ( root , 0 , 0 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define maxSum  100 NEW_LINE #define arrSize  51 NEW_LINE using namespace std ;
int dp [ arrSize ] [ maxSum ] ; bool visit [ arrSize ] [ maxSum ] ;
int SubsetCnt ( int i , int s , int arr [ ] , int n ) {
if ( i == n ) { if ( s == 0 ) return 1 ; else return 0 ; }
if ( visit [ i ] [ s + maxSum ] ) return dp [ i ] [ s + maxSum ] ;
visit [ i ] [ s + maxSum ] = 1 ;
dp [ i ] [ s + maxSum ] = SubsetCnt ( i + 1 , s + arr [ i ] , arr , n ) + SubsetCnt ( i + 1 , s , arr , n ) ;
return dp [ i ] [ s + maxSum ] ; }
int main ( ) { int arr [ ] = { 2 , 2 , 2 , -4 , -4 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << SubsetCnt ( 0 , 0 , arr , n ) ; }
void printTetra ( int n ) { if ( n < 0 ) return ;
int first = 0 , second = 1 ; int third = 1 , fourth = 2 ;
int curr ; if ( n == 0 ) cout << first ; else if ( n == 1 n == 2 ) cout << second ; else if ( n == 3 ) cout << fourth ; else {
for ( int i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } cout << curr ; } }
int main ( ) { int n = 10 ; printTetra ( n ) ; return 0 ; }
int countWays ( int n ) { int res [ n + 1 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
int main ( ) { int n = 4 ; cout << countWays ( n ) ; return 0 ; }
int countWays ( int n ) {
int a = 1 , b = 2 , c = 4 ;
int d = 0 ; if ( n == 0 n == 1 n == 2 ) return n ; if ( n == 3 ) return c ;
for ( int i = 4 ; i <= n ; i ++ ) { d = c + b + a ; a = b ; b = c ; c = d ; } return d ; }
int main ( ) { int n = 4 ; cout << countWays ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; bool isPossible ( int elements [ ] , int sum , int n ) { int dp [ sum + 1 ] ;
dp [ 0 ] = 1 ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = sum ; j >= elements [ i ] ; j -- ) { if ( dp [ j - elements [ i ] ] == 1 ) dp [ j ] = 1 ; } }
if ( dp [ sum ] == 1 ) return true ; return false ; }
int main ( ) { int elements [ ] = { 6 , 2 , 5 } ; int n = sizeof ( elements ) / sizeof ( elements [ 0 ] ) ; int sum = 7 ; if ( isPossible ( elements , sum , n ) ) cout << ( " YES " ) ; else cout << ( " NO " ) ; return 0 ; }
int maxTasks ( int high [ ] , int low [ ] , int n ) {
if ( n <= 0 ) return 0 ;
return max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
int main ( ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; cout << maxTasks ( high , low , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
char FindKthChar ( string str , long long K , int X ) {
char ans ; int sum = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
int digit = str [ i ] - '0' ;
int range = pow ( digit , X ) ; sum += range ;
if ( K <= sum ) { ans = str [ i ] ; break ; } }
return ans ; }
string str = "123" ; long long K = 9 ; int X = 3 ;
char ans = FindKthChar ( str , K , X ) ; cout << ans << " STRNEWLINE " ; return 0 ; }
int totalPairs ( string s1 , string s2 ) { int count = 0 ; int arr1 [ 7 ] , arr2 [ 7 ] ; for ( int i = 1 ; i <= 6 ; i ++ ) { arr1 [ i ] = 0 ; arr2 [ i ] = 0 ; }
for ( int i = 0 ; i < s1 . length ( ) ; i ++ ) { int set_bits = __builtin_popcount ( ( int ) s1 [ i ] ) ; arr1 [ set_bits ] ++ ; }
for ( int i = 0 ; i < s2 . length ( ) ; i ++ ) { int set_bits = __builtin_popcount ( ( int ) s2 [ i ] ) ; arr2 [ set_bits ] ++ ; }
for ( int i = 1 ; i <= 6 ; i ++ ) count += ( arr1 [ i ] * arr2 [ i ] ) ;
return count ; }
int main ( ) { string s1 = " geeks " ; string s2 = " forgeeks " ; cout << totalPairs ( s1 , s2 ) ; return 0 ; }
int countSubstr ( string str , int n , char x , char y ) {
int tot_count = 0 ;
int count_x = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str [ i ] == x ) count_x ++ ;
if ( str [ i ] == y ) tot_count += count_x ; }
return tot_count ; }
int main ( ) { string str = " abbcaceghcak " ; int n = str . size ( ) ; char x = ' a ' , y = ' c ' ; cout << " Count ▁ = ▁ " << countSubstr ( str , n , x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define OUT  0 NEW_LINE #define IN  1
unsigned countWords ( char * str ) { int state = OUT ;
unsigned wc = 0 ;
while ( * str ) {
if ( * str == ' ▁ ' * str == ' ' * str == ' TABSYMBOL ' ) state = OUT ;
else if ( state == OUT ) { state = IN ; ++ wc ; }
++ str ; } return wc ; }
int main ( void ) { char str [ ] = " One ▁ twothree STRNEWLINE ▁ four TABSYMBOL five ▁ " ; cout << " No ▁ of ▁ words ▁ : ▁ " << countWords ( str ) ; return 0 ; }
int nthEnneadecagonal ( long int n ) {
return ( 17 * n * n - 15 * n ) / 2 ; }
int main ( ) { long int n = 6 ; cout << n << " th ▁ Enneadecagonal ▁ number ▁ : " << nthEnneadecagonal ( n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define PI  3.14159265
float areacircumscribed ( float a ) { return ( a * a * ( PI / 2 ) ) ; }
int main ( ) { float a = 6 ; printf ( " ▁ Area ▁ of ▁ an ▁ circumscribed ▁ circle ▁ is ▁ : ▁ % .2f ▁ " , areacircumscribed ( a ) ) ; return 0 ; }
int itemType ( int n ) {
int count = 0 ; int day = 1 ;
while ( count + day * ( day + 1 ) / 2 < n ) {
count += day * ( day + 1 ) / 2 ; day ++ ; } for ( int type = day ; type > 0 ; type -- ) {
count += type ;
if ( count >= n ) { return type ; } } }
int main ( ) { int N = 10 ; cout << itemType ( N ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
bool isSortedDesc ( struct Node * head ) { if ( head == NULL ) return true ;
for ( Node * t = head ; t -> next != NULL ; t = t -> next ) if ( t -> data <= t -> next -> data ) return false ; return true ; } Node * newNode ( int data ) { Node * temp = new Node ; temp -> next = NULL ; temp -> data = data ; }
int main ( ) { struct Node * head = newNode ( 7 ) ; head -> next = newNode ( 5 ) ; head -> next -> next = newNode ( 4 ) ; head -> next -> next -> next = newNode ( 3 ) ; isSortedDesc ( head ) ? cout << " Yes " : cout << " No " ; return 0 ; }
int maxLength ( string str , int n , char c , int k ) {
int ans = -1 ;
int cnt = 0 ;
int left = 0 ; for ( int right = 0 ; right < n ; right ++ ) { if ( str [ right ] == c ) { cnt ++ ; }
while ( cnt > k ) { if ( str [ left ] == c ) { cnt -- ; }
left ++ ; }
ans = max ( ans , right - left + 1 ) ; } return ans ; }
int maxConsecutiveSegment ( string S , int K ) { int N = S . length ( ) ;
return max ( maxLength ( S , N , '0' , K ) , maxLength ( S , N , '1' , K ) ) ; }
int main ( ) { string S = "1001" ; int K = 1 ; cout << maxConsecutiveSegment ( S , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void find ( int N ) { int T , F , O ;
F = int ( ( N - 4 ) / 5 ) ;
if ( ( ( N - 5 * F ) % 2 ) == 0 ) { O = 2 ; } else { O = 1 ; }
T = floor ( ( N - 5 * F - O ) / 2 ) ; cout << " Count ▁ of ▁ 5 ▁ valueds ▁ coins : ▁ " << F << endl ; cout << " Count ▁ of ▁ 2 ▁ valueds ▁ coins : ▁ " << T << endl ; cout << " Count ▁ of ▁ 1 ▁ valueds ▁ coins : ▁ " << O << endl ; }
int main ( ) { int N = 8 ; find ( N ) ; return 0 ; }
void findMaxOccurence ( string str , int N ) {
for ( int i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' ? ' ) {
string str = "10?0?11" ; int N = str . length ( ) ; findMaxOccurence ( str , N ) ; return 0 ; }
void checkInfinite ( string s ) {
bool flag = 1 ; int N = s . length ( ) ;
for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( s [ i ] == char ( int ( s [ i + 1 ] ) + 1 ) ) { continue ; }
else if ( s [ i ] == ' a ' && s [ i + 1 ] == ' z ' ) { continue ; }
else { flag = 0 ; break ; } }
if ( flag == 0 ) cout << " NO " ; else cout << " YES " ; }
int main ( ) {
string s = " ecbaz " ;
checkInfinite ( s ) ; return 0 ; }
int minChangeInLane ( int barrier [ ] , int n ) { int dp [ ] = { 1 , 0 , 1 } ; for ( int j = 0 ; j < n ; j ++ ) {
int val = barrier [ j ] ; if ( val > 0 ) { dp [ val - 1 ] = 1e6 ; } for ( int i = 0 ; i < 3 ; i ++ ) {
if ( val != i + 1 ) { dp [ i ] = min ( dp [ i ] , min ( dp [ ( i + 1 ) % 3 ] , dp [ ( i + 2 ) % 3 ] ) + 1 ) ; } } }
return min ( dp [ 0 ] , min ( dp [ 1 ] , dp [ 2 ] ) ) ; }
int main ( ) { int barrier [ ] = { 0 , 1 , 2 , 3 , 0 } ; int N = sizeof ( barrier ) / sizeof ( barrier [ 0 ] ) ; cout << minChangeInLane ( barrier , N ) ; return 0 ; }
void numWays ( int ratings [ n ] [ k ] , int queries [ ] [ 2 ] ) {
int dp [ n ] [ 10000 + 2 ] ;
for ( int i = 0 ; i < k ; i ++ ) dp [ 0 ] [ ratings [ 0 ] [ i ] ] += 1 ;
for ( int i = 1 ; i < n ; i ++ ) {
for ( int sum = 0 ; sum <= 10000 ; sum ++ ) {
for ( int j = 0 ; j < k ; j ++ ) {
if ( sum >= ratings [ i ] [ j ] ) dp [ i ] [ sum ] += dp [ i - 1 ] [ sum - ratings [ i ] [ j ] ] ; } } }
for ( int sum = 1 ; sum <= 10000 ; sum ++ ) { dp [ n - 1 ] [ sum ] += dp [ n - 1 ] [ sum - 1 ] ; }
for ( int q = 0 ; q < 2 ; q ++ ) { int a = queries [ q ] [ 0 ] ; int b = queries [ q ] [ 1 ] ;
cout << dp [ n - 1 ] [ b ] - dp [ n - 1 ] [ a - 1 ] << " ▁ " ; } }
int main ( ) {
#define n  2 NEW_LINE #define k  3
int ratings [ n ] [ k ] = { { 1 , 2 , 3 } , { 4 , 5 , 6 } } ;
numWays ( ratings , queries ) ; return 0 ; }
int numberOfPermWithKInversion ( int N , int K ) {
int dp [ 2 ] [ K + 1 ] ; int mod = 1000000007 ; for ( int i = 1 ; i <= N ; i ++ ) { for ( int j = 0 ; j <= K ; j ++ ) {
if ( i == 1 ) dp [ i % 2 ] [ j ] = ( j == 0 ) ;
else if ( j == 0 ) dp [ i % 2 ] [ j ] = 1 ;
else dp [ i % 2 ] [ j ] = ( dp [ i % 2 ] [ j - 1 ] % mod + ( dp [ 1 - i % 2 ] [ j ] - ( ( max ( j - ( i - 1 ) , 0 ) == 0 ) ? 0 : dp [ 1 - i % 2 ] [ max ( j - ( i - 1 ) , 0 ) - 1 ] ) + mod ) % mod ) % mod ; ; } }
cout << dp [ N % 2 ] [ K ] ; }
int main ( ) {
int N = 3 , K = 2 ;
numberOfPermWithKInversion ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int MaxProfit ( int treasure [ ] , int color [ ] , int n , int k , int col , int A , int B ) { int sum = 0 ;
if ( k == n ) return 0 ;
if ( col == color [ k ] ) sum += max ( A * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ; else sum += max ( B * treasure [ k ] + MaxProfit ( treasure , color , n , k + 1 , color [ k ] , A , B ) , MaxProfit ( treasure , color , n , k + 1 , col , A , B ) ) ;
return sum ; }
int main ( ) { int A = -5 , B = 7 ; int treasure [ ] = { 4 , 8 , 2 , 9 } ; int color [ ] = { 2 , 2 , 6 , 2 } ; int n = sizeof ( color ) / sizeof ( color [ 0 ] ) ;
cout << MaxProfit ( treasure , color , n , 0 , 0 , A , B ) ; return 0 ; }
int printTetraRec ( int n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
void printTetra ( int n ) { cout << printTetraRec ( n ) << " ▁ " ; }
int main ( ) { int n = 10 ; printTetra ( n ) ; return 0 ; }
int sum = 0 ; void Combination ( int a [ ] , int combi [ ] , int n , int r , int depth , int index ) {
if ( index == r ) {
int product = 1 ; for ( int i = 0 ; i < r ; i ++ ) product = product * combi [ i ] ;
sum += product ; return ; }
for ( int i = depth ; i < n ; i ++ ) { combi [ index ] = a [ i ] ; Combination ( a , combi , n , r , i + 1 , index + 1 ) ; } }
void allCombination ( int a [ ] , int n ) { for ( int i = 1 ; i <= n ; i ++ ) {
int * combi = new int [ i ] ;
Combination ( a , combi , n , i , 0 , 0 ) ;
cout << " f ( " << i << " ) ▁ - - > ▁ " << sum << " STRNEWLINE " ; sum = 0 ; free ( combi ) ; } }
int main ( ) { int n = 5 ; int * a = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) a [ i ] = i + 1 ;
allCombination ( a , n ) ; return 0 ; }
int max ( int x , int y ) { return ( x > y ? x : y ) ; }
int maxTasks ( int high [ ] , int low [ ] , int n ) {
int task_dp [ n + 1 ] ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( int i = 2 ; i <= n ; i ++ ) task_dp [ i ] = max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
int main ( ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; cout << maxTasks ( high , low , n ) ; return 0 ; }
int main ( ) { int n = 10 , k = 2 ; cout << " Value ▁ of ▁ P ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << PermutationCoeff ( n , k ) ; return 0 ; }
bool findPartiion ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; bool part [ sum / 2 + 1 ] [ n + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) part [ 0 ] [ i ] = true ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) part [ i ] [ 0 ] = false ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) { for ( j = 1 ; j <= n ; j ++ ) { part [ i ] [ j ] = part [ i ] [ j - 1 ] ; if ( i >= arr [ j - 1 ] ) part [ i ] [ j ] = part [ i ] [ j ] || part [ i - arr [ j - 1 ] ] [ j - 1 ] ; } }
return part [ sum / 2 ] [ n ] ; }
int main ( ) { int arr [ ] = { 3 , 1 , 1 , 2 , 2 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) cout << " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ " " sum " ; else cout << " Can ▁ not ▁ be ▁ divided ▁ into " << " ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ; return 0 ; }
int minimumOperations ( string orig_str , int m , int n ) {
string orig = orig_str ;
int turn = 1 ; int j = 1 ;
for ( auto i : orig_str ) {
string m_cut = orig_str . substr ( orig_str . length ( ) - m ) ; orig_str . erase ( orig_str . length ( ) - m ) ;
orig_str = m_cut + orig_str ;
j = j + 1 ;
if ( orig != orig_str ) { turn = turn + 1 ;
string n_cut = orig_str . substr ( orig_str . length ( ) - n ) ; orig_str . erase ( orig_str . length ( ) - n ) ;
orig_str = n_cut + orig_str ;
j = j + 1 ; }
if ( orig == orig_str ) { break ; }
turn = turn + 1 ; } cout << turn ; }
string S = " GeeksforGeeks " ; int X = 5 , Y = 3 ;
minimumOperations ( S , X , Y ) ; return 0 ; }
int KMPSearch ( char * pat , char * txt ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ;
int lps [ M ] ;
computeLPSArray ( pat , M , lps ) ;
int i = 0 ; int j = 0 ; while ( i < N ) { if ( pat [ j ] == txt [ i ] ) { j ++ ; i ++ ; } if ( j == M ) { return i - j ; j = lps [ j - 1 ] ; }
else if ( i < N && pat [ j ] != txt [ i ] ) {
if ( j != 0 ) j = lps [ j - 1 ] ; else i = i + 1 ; } } }
void computeLPSArray ( char * pat , int M , int * lps ) {
int len = 0 ;
lps [ 0 ] = 0 ;
int i = 1 ; while ( i < M ) { if ( pat [ i ] == pat [ len ] ) { len ++ ; lps [ i ] = len ; i ++ ; }
else {
if ( len != 0 ) { len = lps [ len - 1 ] ; } else { lps [ i ] = 0 ; i ++ ; } } } }
int countRotations ( string s ) {
string s1 = s . substr ( 1 , s . size ( ) - 1 ) + s ;
char pat [ s . length ( ) ] , text [ s1 . length ( ) ] ; strcpy ( pat , s . c_str ( ) ) ; strcpy ( text , s1 . c_str ( ) ) ;
return 1 + KMPSearch ( pat , text ) ; }
int main ( ) { string s1 = " geeks " ; cout << countRotations ( s1 ) ; return 0 ; }
void start ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ; }
void state1 ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ;
else if ( c == ' h ' c == ' H ' ) dfa = 2 ;
else dfa = 0 ; }
void state2 ( char c ) {
if ( c == ' e ' c == ' E ' ) dfa = 3 ; else if ( c == ' t ' c == ' T ' ) dfa = 1 ; else dfa = 0 ; }
void state3 ( char c ) {
if ( c == ' t ' c == ' T ' ) dfa = 1 ; else dfa = 0 ; } bool isAccepted ( string str ) {
int len = str . length ( ) ; for ( int i = 0 ; i < len ; i ++ ) { if ( dfa == 0 ) start ( str [ i ] ) ; else if ( dfa == 1 ) state1 ( str [ i ] ) ; else if ( dfa == 2 ) state2 ( str [ i ] ) ; else state3 ( str [ i ] ) ; } return ( dfa != 3 ) ; }
int main ( ) { string str = " forTHEgeeks " ; if ( isAccepted ( str ) == true ) cout << " ACCEPTED STRNEWLINE " ; else cout << " NOT ▁ ACCEPTED STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int parent [ 26 ] ;
int find ( int x ) { if ( x != parent [ x ] ) return parent [ x ] = find ( parent [ x ] ) ; return x ; }
void join ( int x , int y ) { int px = find ( x ) ; int pz = find ( y ) ; if ( px != pz ) { parent [ pz ] = px ; } }
bool convertible ( string s1 , string s2 ) {
map < int , int > mp ; for ( int i = 0 ; i < s1 . size ( ) ; i ++ ) { if ( mp . find ( s1 [ i ] - ' a ' ) == mp . end ( ) ) { mp [ s1 [ i ] - ' a ' ] = s2 [ i ] - ' a ' ; } else { if ( mp [ s1 [ i ] - ' a ' ] != s2 [ i ] - ' a ' ) return false ; } }
for ( auto it : mp ) { if ( it . first == it . second ) continue ; else { if ( find ( it . first ) == find ( it . second ) ) return false ; else join ( it . first , it . second ) ; } } return true ; }
void initialize ( ) { for ( int i = 0 ; i < 26 ; i ++ ) { parent [ i ] = i ; } }
int main ( ) { string s1 , s2 ; s1 = " abbcaa " ; s2 = " bccdbb " ; initialize ( ) ; if ( convertible ( s1 , s2 ) ) cout << " Yes " << endl ; else cout << " No " << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define SIZE  26
void SieveOfEratosthenes ( bool prime [ ] , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
void printChar ( string str , int n ) { bool prime [ n + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ;
SieveOfEratosthenes ( prime , str . length ( ) + 1 ) ;
int freq [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( prime [ freq [ str [ i ] - ' a ' ] ] ) { cout << str [ i ] ; } } }
int main ( ) { string str = " geeksforgeeks " ; int n = str . length ( ) ; printChar ( str , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define SIZE  26
void printChar ( string str , int n ) {
int freq [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] % 2 == 0 ) { cout << str [ i ] ; } } }
int main ( ) { string str = " geeksforgeeks " ; int n = str . length ( ) ; printChar ( str , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; bool CompareAlphanumeric ( string & str1 , string & str2 ) {
int i , j ; i = 0 ; j = 0 ;
int len1 = str1 . size ( ) ;
int len2 = str2 . size ( ) ;
while ( i <= len1 && j <= len2 ) {
while ( i < len1 && ( ! ( ( str1 [ i ] >= ' a ' && str1 [ i ] <= ' z ' ) || ( str1 [ i ] >= ' A ' && str1 [ i ] <= ' Z ' ) || ( str1 [ i ] >= '0' && str1 [ i ] <= '9' ) ) ) ) { i ++ ; }
while ( j < len2 && ( ! ( ( str2 [ j ] >= ' a ' && str2 [ j ] <= ' z ' ) || ( str2 [ j ] >= ' A ' && str2 [ j ] <= ' Z ' ) || ( str2 [ j ] >= '0' && str2 [ j ] <= '9' ) ) ) ) { j ++ ; }
if ( i == len1 && j == len2 ) return true ;
else if ( str1 [ i ] != str2 [ j ] ) return false ;
else { i ++ ; j ++ ; } }
return false ; }
void CompareAlphanumericUtil ( string str1 , string str2 ) { bool res ;
res = CompareAlphanumeric ( str1 , str2 ) ;
if ( res == true ) cout << " Equal " << endl ;
else cout < < " Unequal " << endl ; }
int main ( ) { string str1 , str2 ; str1 = " Ram , ▁ Shyam " ; str2 = " ▁ Ram ▁ - ▁ Shyam . " ; CompareAlphanumericUtil ( str1 , str2 ) ; str1 = " abc123" ; str2 = "123abc " ; CompareAlphanumericUtil ( str1 , str2 ) ; return 0 ; }
void solveQueries ( string str , vector < vector < int > > & query ) {
int len = str . size ( ) ;
int Q = query . size ( ) ;
int pre [ len ] [ 26 ] ; memset ( pre , 0 , sizeof pre ) ;
for ( int i = 0 ; i < len ; i ++ ) {
pre [ i ] [ str [ i ] - ' a ' ] ++ ;
if ( i ) {
for ( int j = 0 ; j < 26 ; j ++ ) pre [ i ] [ j ] += pre [ i - 1 ] [ j ] ; } }
for ( int i = 0 ; i < Q ; i ++ ) {
int l = query [ i ] [ 0 ] ; int r = query [ i ] [ 1 ] ; int maxi = 0 ; char c = ' a ' ;
for ( int j = 0 ; j < 26 ; j ++ ) {
int times = pre [ r ] [ j ] ;
if ( l ) times -= pre [ l - 1 ] [ j ] ;
if ( times > maxi ) { maxi = times ; c = char ( ' a ' + j ) ; } }
cout << " Query ▁ " << i + 1 << " : ▁ " << c << endl ; } }
int main ( ) { string str = " striver " ; vector < vector < int > > query ; query . push_back ( { 0 , 1 } ) ; query . push_back ( { 1 , 6 } ) ; query . push_back ( { 5 , 6 } ) ; solveQueries ( str , query ) ; }
bool startsWith ( string str , string pre ) { int strLen = str . length ( ) ; int preLen = pre . length ( ) ; int i = 0 , j = 0 ;
while ( i < strLen && j < preLen ) {
if ( str [ i ] != pre [ j ] ) return false ; i ++ ; j ++ ; }
return true ; }
bool endsWith ( string str , string suff ) { int i = str . length ( ) - 0 ; int j = suff . length ( ) - 0 ;
while ( i >= 0 && j >= 0 ) {
if ( str [ i ] != suff [ j ] ) return false ; i -- ; j -- ; }
return true ; }
bool checkString ( string str , string a , string b ) {
if ( str . length ( ) != a . length ( ) + b . length ( ) ) return false ;
if ( startsWith ( str , a ) ) {
if ( endsWith ( str , b ) ) return true ; }
if ( startsWith ( str , b ) ) {
if ( endsWith ( str , a ) ) return true ; } return false ; }
int main ( ) { string str = " GeeksforGeeks " ; string a = " Geeksfo " ; string b = " rGeeks " ; if ( checkString ( str , a , b ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define SIZE  26
void printChar ( string str , int n ) {
int freq [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] % 2 == 1 ) { cout << str [ i ] ; } } }
int main ( ) { string str = " geeksforgeeks " ; int n = str . length ( ) ; printChar ( str , n ) ; return 0 ; }
int minOperations ( string str , int n ) {
int i , lastUpper = -1 , firstLower = -1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( isupper ( str [ i ] ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( islower ( str [ i ] ) ) { firstLower = i ; break ; } }
if ( lastUpper == -1 firstLower == -1 ) return 0 ;
int countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( isupper ( str [ i ] ) ) { countUpper ++ ; } }
int countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( islower ( str [ i ] ) ) { countLower ++ ; } }
return min ( countLower , countUpper ) ; }
int main ( ) { string str = " geEksFOrGEekS " ; int n = str . length ( ) ; cout << minOperations ( str , n ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int Betrothed_Sum ( int n ) {
vector < int > Set ; for ( int number_1 = 1 ; number_1 < n ; number_1 ++ ) {
int sum_divisor_1 = 1 ;
int i = 2 ; while ( i * i <= number_1 ) { if ( number_1 % i == 0 ) { sum_divisor_1 = sum_divisor_1 + i ; if ( i * i != number_1 ) sum_divisor_1 += number_1 / i ; } i ++ ; } if ( sum_divisor_1 > number_1 ) { int number_2 = sum_divisor_1 - 1 ; int sum_divisor_2 = 1 ; int j = 2 ; while ( j * j <= number_2 ) { if ( number_2 % j == 0 ) { sum_divisor_2 += j ; if ( j * j != number_2 ) sum_divisor_2 += number_2 / j ; } j = j + 1 ; } if ( sum_divisor_2 == number_1 + 1 and number_1 <= n && number_2 <= n ) { Set . push_back ( number_1 ) ; Set . push_back ( number_2 ) ; } } }
int Summ = 0 ; for ( auto i : Set ) { if ( i <= n ) Summ += i ; } return Summ ; }
int main ( ) { int n = 78 ; cout << Betrothed_Sum ( n ) ; return 0 ; }
float rainDayProbability ( int a [ ] , int n ) { float count = 0 , m ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
int main ( ) { int a [ ] = { 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << rainDayProbability ( a , n ) ; return 0 ; }
double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / pow ( i , i ) ; sums += ser ; } return sums ; }
int main ( ) { int n = 3 ; double res = Series ( n ) ; cout << res ; return 0 ; }
string lexicographicallyMaximum ( string S , int N ) {
unordered_map < char , int > M ;
for ( int i = 0 ; i < N ; ++ i ) { M [ S [ i ] ] ++ ; }
vector < char > V ; for ( char i = ' a ' ; i < ( char ) ( ' a ' + min ( N , 25 ) ) ; ++ i ) { if ( M [ i ] == 0 ) { V . push_back ( i ) ; } }
int j = V . size ( ) - 1 ;
for ( int i = 0 ; i < N ; ++ i ) {
if ( S [ i ] >= ( ' a ' + min ( N , 25 ) ) M [ S [ i ] ] > 1 ) { if ( V [ j ] < S [ i ] ) continue ;
M [ S [ i ] ] -- ;
S [ i ] = V [ j ] ;
j -- ; } if ( j < 0 ) break ; } int l = 0 ;
for ( int i = N - 1 ; i >= 0 ; i -- ) { if ( l > j ) break ; if ( S [ i ] >= ( ' a ' + min ( N , 25 ) ) M [ S [ i ] ] > 1 ) {
M [ S [ i ] ] -- ;
S [ i ] = V [ l ] ;
l ++ ; } }
return S ; }
string S = " abccefghh " ; int N = S . length ( ) ;
cout << lexicographicallyMaximum ( S , N ) ; return 0 ; }
bool isConsistingSubarrayUtil ( int arr [ ] , int n ) {
map < int , int > mp ;
for ( int i = 0 ; i < n ; ++ i ) {
mp [ arr [ i ] ] ++ ; }
map < int , int > :: iterator it ; for ( it = mp . begin ( ) ; it != mp . end ( ) ; ++ it ) {
if ( it -> second > 1 ) { return true ; } }
return false ; }
void isConsistingSubarray ( int arr [ ] , int N ) { if ( isConsistingSubarrayUtil ( arr , N ) ) { cout << " Yes " << endl ; } else { cout << " No " << endl ; } }
int main ( ) {
int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 1 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
isConsistingSubarray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
set < int > createhashmap ( int Max ) {
set < int > hashmap ;
int curr = 1 ;
int prev = 0 ;
hashmap . insert ( prev ) ;
while ( curr <= Max ) {
hashmap . insert ( curr ) ;
int temp = curr ;
curr = curr + prev ;
prev = temp ; } return hashmap ; }
vector < bool > SieveOfEratosthenes ( int Max ) {
vector < bool > isPrime ( Max , true ) ; isPrime [ 0 ] = false ; isPrime [ 1 ] = false ;
for ( int p = 2 ; p * p <= Max ; p ++ ) {
if ( isPrime [ p ] ) {
for ( int i = p * p ; i <= Max ; i += p ) {
int cntFibonacciPrime ( int arr [ ] , int N ) {
int Max = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) {
Max = max ( Max , arr [ i ] ) ; }
vector < bool > isPrime = SieveOfEratosthenes ( Max ) ;
set < int > hashmap = createhashmap ( Max ) ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 1 ) continue ;
if ( ( hashmap . count ( arr [ i ] ) ) && ! isPrime [ arr [ i ] ] ) {
cout << arr [ i ] << " ▁ " ; } } }
int main ( ) { int arr [ ] = { 13 , 55 , 7 , 3 , 5 , 21 , 233 , 144 , 89 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cntFibonacciPrime ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int key ( int N ) {
string num = " " + to_string ( N ) ; int ans = 0 ; int j = 0 ;
for ( j = 0 ; j < num . length ( ) ; j ++ ) {
if ( ( num [ j ] - 48 ) % 2 == 0 ) { int add = 0 ; int i ;
for ( i = j ; j < num . length ( ) ; j ++ ) { add += num [ j ] - 48 ;
if ( add % 2 == 1 ) break ; } if ( add == 0 ) { ans *= 10 ; } else { int digit = ( int ) floor ( log10 ( add ) + 1 ) ; ans *= ( pow ( 10 , digit ) ) ;
ans += add ; }
i = j ; } else {
int add = 0 ; int i ;
for ( i = j ; j < num . length ( ) ; j ++ ) { add += num [ j ] - 48 ;
if ( add % 2 == 0 ) { break ; } } if ( add == 0 ) { ans *= 10 ; } else { int digit = ( int ) floor ( log10 ( add ) + 1 ) ; ans *= ( pow ( 10 , digit ) ) ;
ans += add ; }
i = j ; } }
if ( j + 1 >= num . length ( ) ) { return ans ; } else { return ans += num [ num . length ( ) - 1 ] - 48 ; } }
int main ( ) { int N = 1667848271 ; cout << key ( N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void sentinelSearch ( int arr [ ] , int n , int key ) {
int last = arr [ n - 1 ] ;
arr [ n - 1 ] = key ; int i = 0 ; while ( arr [ i ] != key ) i ++ ;
arr [ n - 1 ] = last ; if ( ( i < n - 1 ) || ( arr [ n - 1 ] == key ) ) cout << key << " ▁ is ▁ present ▁ at ▁ index ▁ " << i ; else cout << " Element ▁ Not ▁ found " ; }
int main ( ) { int arr [ ] = { 10 , 20 , 180 , 30 , 60 , 50 , 110 , 100 , 70 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int key = 180 ; sentinelSearch ( arr , n , key ) ; return 0 ; }
int maximum_middle_value ( int n , int k , int arr [ ] ) {
int ans = -1 ;
int low = ( n + 1 - k ) / 2 ; int high = ( n + 1 - k ) / 2 + k ;
for ( int i = low ; i <= high ; i ++ ) {
ans = max ( ans , arr [ i - 1 ] ) ; }
return ans ; }
int main ( ) { int n = 5 , k = 2 ; int arr [ ] = { 9 , 5 , 3 , 7 , 10 } ; cout << maximum_middle_value ( n , k , arr ) << endl ; n = 9 ; k = 3 ; int arr1 [ ] = { 2 , 4 , 3 , 9 , 5 , 8 , 7 , 6 , 10 } ; cout << maximum_middle_value ( n , k , arr1 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int ternarySearch ( int l , int r , int key , int ar [ ] ) { if ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
return ternarySearch ( l , mid1 - 1 , key , ar ) ; } else if ( key > ar [ mid2 ] ) {
return ternarySearch ( mid2 + 1 , r , key , ar ) ; } else {
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) ; } }
return -1 ; }
int main ( ) { int l , r , p , key ;
int ar [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
cout << " Index ▁ of ▁ " << key << " ▁ is ▁ " << p << endl ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
cout << " Index ▁ of ▁ " << key << " ▁ is ▁ " << p << endl ; }
int findmin ( Point p [ ] , int n ) { int a = 0 , b = 0 , c = 0 , d = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( p [ i ] . x <= 0 ) a ++ ;
else if ( p [ i ] . x >= 0 ) b ++ ;
if ( p [ i ] . y >= 0 ) c ++ ;
else if ( p [ i ] . y <= 0 ) d ++ ; } return min ( { a , b , c , d } ) ; }
int main ( ) { Point p [ ] = { { 1 , 1 } , { 2 , 2 } , { -1 , -1 } , { -2 , 2 } } ; int n = sizeof ( p ) / sizeof ( p [ 0 ] ) ; cout << findmin ( p , n ) ; return 0 ; }
void maxOps ( int a , int b , int c ) {
int arr [ ] = { a , b , c } ;
int count = 0 ; while ( 1 ) {
sort ( arr , arr + 3 ) ;
if ( ! arr [ 0 ] && ! arr [ 1 ] ) break ;
arr [ 1 ] -= 1 ; arr [ 2 ] -= 1 ;
count += 1 ; }
cout << count ; }
int a = 4 , b = 3 , c = 2 ; maxOps ( a , b , c ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  26
string getSortedString ( string s , int n ) {
int lower [ MAX ] = { 0 } ; int upper [ MAX ] = { 0 } ; for ( int i = 0 ; i < n ; i ++ ) {
if ( islower ( s [ i ] ) ) lower [ s [ i ] - ' a ' ] ++ ;
else if ( isupper ( s [ i ] ) ) upper [ s [ i ] - ' A ' ] ++ ; }
int i = 0 , j = 0 ; while ( i < MAX && lower [ i ] == 0 ) i ++ ; while ( j < MAX && upper [ j ] == 0 ) j ++ ;
for ( int k = 0 ; k < n ; k ++ ) {
if ( islower ( s [ k ] ) ) { while ( lower [ i ] == 0 ) i ++ ; s [ k ] = ( char ) ( i + ' a ' ) ;
lower [ i ] -- ; }
else if ( isupper ( s [ k ] ) ) { while ( upper [ j ] == 0 ) j ++ ; s [ k ] = ( char ) ( j + ' A ' ) ;
upper [ j ] -- ; } }
return s ; }
int main ( ) { string s = " gEeksfOrgEEkS " ; int n = s . length ( ) ; cout << getSortedString ( s , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define SIZE  26
void printCharWithFreq ( string str ) {
int n = str . size ( ) ;
memset ( freq , 0 , sizeof ( freq ) ) ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] != 0 ) {
cout << str [ i ] << freq [ str [ i ] - ' a ' ] << " ▁ " ;
freq [ str [ i ] - ' a ' ] = 0 ; } } }
int main ( ) { string str = " geeksforgeeks " ; printCharWithFreq ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { string s [ ] = { " i " , " like " , " this " , " program " , " very " , " much " } ; string ans = " " ; for ( int i = 5 ; i >= 0 ; i -- ) { ans += s [ i ] + " ▁ " ; } cout << ( " Reversed ▁ String : " ) << endl ; cout << ( ans . substr ( 0 , ans . length ( ) - 1 ) ) << endl ; return 0 ; }
void SieveOfEratosthenes ( int n ) { memset ( prime , true , sizeof ( prime ) ) ; for ( int p = 2 ; p * p <= n ; p ++ ) {
if ( prime [ p ] == true ) {
for ( int i = p * p ; i <= n ; i += p ) prime [ i ] = false ; } } }
void segregatePrimeNonPrime ( int arr [ ] , int N ) {
SieveOfEratosthenes ( 10000000 ) ;
int left = 0 , right = N - 1 ;
while ( left < right ) {
while ( prime [ arr [ left ] ] ) left ++ ;
while ( ! prime [ arr [ right ] ] ) right -- ;
if ( left < right ) {
swap ( & arr [ left ] , & arr [ right ] ) ; left ++ ; right -- ; } }
for ( int i = 0 ; i < N ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 2 , 3 , 4 , 6 , 7 , 8 , 9 , 10 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
segregatePrimeNonPrime ( arr , N ) ; return 0 ; }
int findDepthRec ( char tree [ ] , int n , int & index ) { if ( index >= n tree [ index ] == ' l ' ) return 0 ;
index ++ ; int left = findDepthRec ( tree , n , index ) ;
index ++ ; int right = findDepthRec ( tree , n , index ) ; return max ( left , right ) + 1 ; }
int findDepth ( char tree [ ] , int n ) { int index = 0 ; findDepthRec ( tree , n , index ) ; }
int main ( ) { char tree [ ] = " nlnnlll " ; int n = strlen ( tree ) ; cout << findDepth ( tree , n ) << endl ; return 0 ; }
Node * newNode ( int item ) { Node * temp = new Node ; temp -> key = item ; temp -> left = temp -> right = NULL ; return temp ; }
Node * insert ( Node * node , int key ) {
if ( node == NULL ) return newNode ( key ) ;
if ( key < node -> key ) node -> left = insert ( node -> left , key ) ; else if ( key > node -> key ) node -> right = insert ( node -> right , key ) ;
return node ; }
int findMaxforN ( Node * root , int N ) {
if ( root == NULL ) return -1 ; if ( root -> key == N ) return N ;
else if ( root -> key < N ) { int k = findMaxforN ( root -> right , N ) ; if ( k == -1 ) return root -> key ; else return k ; }
else if ( root -> key > N ) return findMaxforN ( root -> left , N ) ; }
int main ( ) { int N = 4 ;
Node * root = insert ( root , 25 ) ; insert ( root , 2 ) ; insert ( root , 1 ) ; insert ( root , 3 ) ; insert ( root , 12 ) ; insert ( root , 9 ) ; insert ( root , 21 ) ; insert ( root , 19 ) ; insert ( root , 25 ) ; printf ( " % d " , findMaxforN ( root , N ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; struct Node { int key ; Node * left , * right ; } ;
Node * newNode ( int item ) { Node * temp = new Node ; temp -> key = item ; temp -> left = temp -> right = NULL ; return temp ; }
Node * insert ( Node * node , int key ) {
if ( node == NULL ) return newNode ( key ) ;
if ( key < node -> key ) node -> left = insert ( node -> left , key ) ; else if ( key > node -> key ) node -> right = insert ( node -> right , key ) ;
return node ; }
void findMaxforN ( Node * root , int N ) {
while ( root != NULL && root -> right != NULL ) {
if ( N > root -> key && N >= root -> right -> key ) root = root -> right ;
else if ( N < root -> key ) root = root -> left ; else break ; } if ( root == NULL root -> key > N ) cout << -1 ; else cout << root -> key ; }
int main ( ) { int N = 50 ; Node * root = insert ( root , 5 ) ; insert ( root , 2 ) ; insert ( root , 1 ) ; insert ( root , 3 ) ; insert ( root , 12 ) ; insert ( root , 9 ) ; insert ( root , 21 ) ; insert ( root , 19 ) ; insert ( root , 25 ) ; findMaxforN ( root , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; struct Node { struct Node * left , * right ; int data ; } ;
Node * createNode ( int x ) { Node * p = new Node ; p -> data = x ; p -> left = p -> right = NULL ; return p ; }
void insertNode ( struct Node * root , int x ) { Node * p = root , * q = NULL ; while ( p != NULL ) { q = p ; if ( p -> data < x ) p = p -> right ; else p = p -> left ; } if ( q == NULL ) p = createNode ( x ) ; else { if ( q -> data < x ) q -> right = createNode ( x ) ; else q -> left = createNode ( x ) ; } }
int maxelpath ( Node * q , int x ) { Node * p = q ; int mx = INT_MIN ;
while ( p -> data != x ) { if ( p -> data > x ) { mx = max ( mx , p -> data ) ; p = p -> left ; } else { mx = max ( mx , p -> data ) ; p = p -> right ; } } return max ( mx , x ) ; }
int maximumElement ( struct Node * root , int x , int y ) { Node * p = root ;
while ( ( x < p -> data && y < p -> data ) || ( x > p -> data && y > p -> data ) ) {
if ( x < p -> data && y < p -> data ) p = p -> left ;
else if ( x > p -> data && y > p -> data ) p = p -> right ; }
return max ( maxelpath ( p , x ) , maxelpath ( p , y ) ) ; }
int main ( ) { int arr [ ] = { 18 , 36 , 9 , 6 , 12 , 10 , 1 , 8 } ; int a = 1 , b = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
struct Node * root = createNode ( arr [ 0 ] ) ;
for ( int i = 1 ; i < n ; i ++ ) insertNode ( root , arr [ i ] ) ; cout << maximumElement ( root , a , b ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; struct Node { struct Node * left , * right ; int info ;
bool lthread ;
bool rthread ; } ;
struct Node * insert ( struct Node * root , int ikey ) {
Node * ptr = root ;
Node * par = NULL ; while ( ptr != NULL ) {
if ( ikey == ( ptr -> info ) ) { printf ( " Duplicate ▁ Key ▁ ! STRNEWLINE " ) ; return root ; }
par = ptr ;
if ( ikey < ptr -> info ) { if ( ptr -> lthread == false ) ptr = ptr -> left ; else break ; }
else { if ( ptr -> rthread == false ) ptr = ptr -> right ; else break ; } }
Node * tmp = new Node ; tmp -> info = ikey ; tmp -> lthread = true ; tmp -> rthread = true ; if ( par == NULL ) { root = tmp ; tmp -> left = NULL ; tmp -> right = NULL ; } else if ( ikey < ( par -> info ) ) { tmp -> left = par -> left ; tmp -> right = par ; par -> lthread = false ; par -> left = tmp ; } else { tmp -> left = par ; tmp -> right = par -> right ; par -> rthread = false ; par -> right = tmp ; } return root ; }
struct Node * inorderSuccessor ( struct Node * ptr ) {
if ( ptr -> rthread == true ) return ptr -> right ;
ptr = ptr -> right ; while ( ptr -> lthread == false ) ptr = ptr -> left ; return ptr ; }
void inorder ( struct Node * root ) { if ( root == NULL ) printf ( " Tree ▁ is ▁ empty " ) ;
struct Node * ptr = root ; while ( ptr -> lthread == false ) ptr = ptr -> left ;
while ( ptr != NULL ) { printf ( " % d ▁ " , ptr -> info ) ; ptr = inorderSuccessor ( ptr ) ; } }
int main ( ) { struct Node * root = NULL ; root = insert ( root , 20 ) ; root = insert ( root , 10 ) ; root = insert ( root , 30 ) ; root = insert ( root , 5 ) ; root = insert ( root , 16 ) ; root = insert ( root , 14 ) ; root = insert ( root , 17 ) ; root = insert ( root , 13 ) ; inorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define MAX  1000 NEW_LINE using namespace std ; void checkHV ( int arr [ ] [ MAX ] , int N , int M ) {
bool horizontal = true , vertical = true ;
for ( int i = 0 , k = N - 1 ; i < N / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < M ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } }
for ( int i = 0 , k = M - 1 ; i < M / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { vertical = false ; break ; } } } if ( ! horizontal && ! vertical ) cout << " NO STRNEWLINE " ; else if ( horizontal && ! vertical ) cout << " HORIZONTAL STRNEWLINE " ; else if ( vertical && ! horizontal ) cout << " VERTICAL STRNEWLINE " ; else cout << " BOTH STRNEWLINE " ; }
int main ( ) { int mat [ MAX ] [ MAX ] = { { 1 , 0 , 1 } , { 0 , 0 , 0 } , { 1 , 0 , 1 } } ; checkHV ( mat , 3 , 3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define R  3 NEW_LINE #define C  4
int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
void replacematrix ( int mat [ R ] [ C ] , int n , int m ) { int rgcd [ R ] = { 0 } , cgcd [ C ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { rgcd [ i ] = gcd ( rgcd [ i ] , mat [ i ] [ j ] ) ; cgcd [ j ] = gcd ( cgcd [ j ] , mat [ i ] [ j ] ) ; } }
for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < m ; j ++ ) mat [ i ] [ j ] = max ( rgcd [ i ] , cgcd [ j ] ) ; }
int main ( ) { int m [ R ] [ C ] = { 1 , 2 , 3 , 3 , 4 , 5 , 6 , 6 , 7 , 8 , 9 , 9 , } ; replacematrix ( m , R , C ) ; for ( int i = 0 ; i < R ; i ++ ) { for ( int j = 0 ; j < C ; j ++ ) cout << m [ i ] [ j ] << " ▁ " ; cout << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  4
void add ( int A [ ] [ N ] , int B [ ] [ N ] , int C [ ] [ N ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] ; }
int main ( ) { int A [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ N ] [ N ] ; int i , j ; add ( A , B , C ) ; cout << " Result ▁ matrix ▁ is ▁ " << endl ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) cout << C [ i ] [ j ] << " ▁ " ; cout << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  4
void subtract ( int A [ ] [ N ] , int B [ ] [ N ] , int C [ ] [ N ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] ; }
int main ( ) { int A [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ N ] [ N ] ; int i , j ; subtract ( A , B , C ) ; cout << " Result ▁ matrix ▁ is ▁ " << endl ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) cout << C [ i ] [ j ] << " ▁ " ; cout << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int linearSearch ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == i ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { -10 , -1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Fixed ▁ Point ▁ is ▁ " << linearSearch ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int binarySearch ( int arr [ ] , int low , int high ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return -1 ; }
int main ( ) { int arr [ 10 ] = { -10 , -1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Fixed ▁ Point ▁ is ▁ " << binarySearch ( arr , 0 , n - 1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int maxTripletSum ( int arr [ ] , int n ) {
int sum = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) for ( int k = j + 1 ; k < n ; k ++ ) if ( sum < arr [ i ] + arr [ j ] + arr [ k ] ) sum = arr [ i ] + arr [ j ] + arr [ k ] ; return sum ; }
int main ( ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << maxTripletSum ( arr , n ) ; return 0 ; }
int maxTripletSum ( int arr [ ] , int n ) {
sort ( arr , arr + n ) ;
return arr [ n - 1 ] + arr [ n - 2 ] + arr [ n - 3 ] ; }
int main ( ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << maxTripletSum ( arr , n ) ; return 0 ; }
int maxTripletSum ( int arr [ ] , int n ) {
int maxA = INT_MIN , maxB = INT_MIN , maxC = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > maxA ) { maxC = maxB ; maxB = maxA ; maxA = arr [ i ] ; }
else if ( arr [ i ] > maxB ) { maxC = maxB ; maxB = arr [ i ] ; }
else if ( arr [ i ] > maxC ) maxC = arr [ i ] ; } return ( maxA + maxB + maxC ) ; }
int main ( ) { int arr [ ] = { 1 , 0 , 8 , 6 , 4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << maxTripletSum ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == x ) return i ; return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int result = search ( arr , n , x ) ; ( result == -1 ) ? cout << " Element ▁ is ▁ not ▁ present ▁ in ▁ array " : cout << " Element ▁ is ▁ present ▁ at ▁ index ▁ " << result ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void search ( vector < int > arr , int search_Element ) { int left = 0 ; int length = arr . size ( ) ; int position = -1 ; int right = length - 1 ;
for ( left = 0 ; left <= right ; ) {
if ( arr [ left ] == search_Element ) { position = left ; cout << " Element ▁ found ▁ in ▁ Array ▁ at ▁ " << position + 1 << " ▁ Position ▁ with ▁ " << left + 1 << " ▁ Attempt " ; break ; }
if ( arr [ right ] == search_Element ) { position = right ; cout << " Element ▁ found ▁ in ▁ Array ▁ at ▁ " << position + 1 << " ▁ Position ▁ with ▁ " << length - right << " ▁ Attempt " ; break ; } left ++ ; right -- ; }
if ( position == -1 ) cout << " Not ▁ found ▁ in ▁ Array ▁ with ▁ " << left << " ▁ Attempt " ; }
int main ( ) { vector < int > arr { 1 , 2 , 3 , 4 , 5 } ; int search_element = 5 ;
search ( arr , search_element ) ; }
void countSort ( char arr [ ] ) {
char output [ strlen ( arr ) ] ;
int count [ RANGE + 1 ] , i ; memset ( count , 0 , sizeof ( count ) ) ;
for ( i = 0 ; arr [ i ] ; ++ i ) ++ count [ arr [ i ] ] ;
for ( i = 1 ; i <= RANGE ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( i = 0 ; arr [ i ] ; ++ i ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( i = 0 ; arr [ i ] ; ++ i ) arr [ i ] = output [ i ] ; }
int main ( ) { char arr [ ] = " geeksforgeeks " ; countSort ( arr ) ; cout << " Sorted ▁ character ▁ array ▁ is ▁ " << arr ; return 0 ; }
void countSort ( vector < int > & arr ) { int max = * max_element ( arr . begin ( ) , arr . end ( ) ) ; int min = * min_element ( arr . begin ( ) , arr . end ( ) ) ; int range = max - min + 1 ; vector < int > count ( range ) , output ( arr . size ( ) ) ; for ( int i = 0 ; i < arr . size ( ) ; i ++ ) count [ arr [ i ] - min ] ++ ; for ( int i = 1 ; i < count . size ( ) ; i ++ ) count [ i ] += count [ i - 1 ] ; for ( int i = arr . size ( ) - 1 ; i >= 0 ; i -- ) { output [ count [ arr [ i ] - min ] - 1 ] = arr [ i ] ; count [ arr [ i ] - min ] -- ; } for ( int i = 0 ; i < arr . size ( ) ; i ++ ) arr [ i ] = output [ i ] ; }
int main ( ) { vector < int > arr = { -5 , -10 , 0 , -3 , 8 , 5 , -1 , 10 } ; countSort ( arr ) ; printArray ( arr ) ; return 0 ; }
int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
int main ( ) { int n = 5 , k = 2 ; cout << " Value ▁ of ▁ C ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << binomialCoeff ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int binomialCoeff ( int n , int k ) { int C [ k + 1 ] ; memset ( C , 0 , sizeof ( C ) ) ;
C [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
int main ( ) { int n = 5 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
int binomialCoeff ( int n , int r ) { if ( r > n ) return 0 ; long long int m = 1000000007 ; long long int inv [ r + 1 ] = { 0 } ; inv [ 0 ] = 1 ; if ( r + 1 >= 2 ) inv [ 1 ] = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { inv [ i ] = m - ( m / i ) * inv [ m % i ] % m ; } int ans = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { ans = ( ( ans % m ) * ( inv [ i ] % m ) ) % m ; }
for ( int i = n ; i >= ( n - r + 1 ) ; i -- ) { ans = ( ( ans % m ) * ( i % m ) ) % m ; } return ans ; }
int main ( ) { int n = 5 , r = 2 ; cout << " Value ▁ of ▁ C ( " << n << " , ▁ " << r << " ) ▁ is ▁ " << binomialCoeff ( n , r ) << endl ; return 0 ; }
bool findPartiion ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; bool part [ sum / 2 + 1 ] ;
for ( i = 0 ; i <= sum / 2 ; i ++ ) { part [ i ] = 0 ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = sum / 2 ; j >= arr [ i ] ; j -- ) {
if ( part [ j - arr [ i ] ] == 1 j == arr [ i ] ) part [ j ] = 1 ; } } return part [ sum / 2 ] ; }
int main ( ) { int arr [ ] = { 1 , 3 , 3 , 2 , 3 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) cout << " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ " " sum " ; else cout << " Can ▁ not ▁ be ▁ divided ▁ into " << " ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ; return 0 ; }
bool isSubsetSum ( int set [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
int main ( ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = sizeof ( set ) / sizeof ( set [ 0 ] ) ; if ( isSubsetSum ( set , n , sum ) == true ) printf ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else printf ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; return 0 ; }
bool isSubsetSum ( int set [ ] , int n , int sum ) {
bool subset [ n + 1 ] [ sum + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ i ] [ 0 ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ 0 ] [ i ] = false ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) { if ( j < set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] ; if ( j >= set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] || subset [ i - 1 ] [ j - set [ i - 1 ] ] ; } }
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) printf ( " % 4d " , subset [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } return subset [ n ] [ sum ] ; }
int main ( ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = sizeof ( set ) / sizeof ( set [ 0 ] ) ; if ( isSubsetSum ( set , n , sum ) == true ) printf ( " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ) ; else printf ( " No ▁ subset ▁ with ▁ given ▁ sum " ) ; return 0 ; }
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) cout << " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " << N << " ▁ keystrokes ▁ is ▁ " << findoptimal ( N ) << endl ; }
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int screen [ N ] ;
int n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = 0 ;
for ( b = n - 3 ; b >= 1 ; b -- ) {
int curr = ( n - b - 1 ) * screen [ b - 1 ] ; if ( curr > screen [ n - 1 ] ) screen [ n - 1 ] = curr ; } } return screen [ N - 1 ] ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) cout << " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " << N << " ▁ keystrokes ▁ is ▁ " << findoptimal ( N ) << endl ; }
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int screen [ N ] ;
int n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = max ( 2 * screen [ n - 4 ] , max ( 3 * screen [ n - 5 ] , 4 * screen [ n - 6 ] ) ) ; } return screen [ N - 1 ] ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) printf ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ % d ▁ keystrokes ▁ is ▁ % d STRNEWLINE " , N , findoptimal ( N ) ) ; }
public : int power ( int x , unsigned int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; } } ;
int main ( ) { gfg g ; int x = 2 ; unsigned int y = 3 ; cout << g . power ( x , y ) ; return 0 ; }
int power ( int x , unsigned int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
int main ( ) { float x = 2 ; int y = -3 ; cout << power ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int power ( int x , int y ) {
if ( y == 0 ) return 1 ;
if ( x == 0 ) return 0 ;
return x * power ( x , y - 1 ) ; }
int main ( ) { int x = 2 ; int y = 3 ; cout << ( power ( x , y ) ) ; }
public : float squareRoot ( float n ) {
float x = n ; float y = 1 ; float e = 0.000001 ;
while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; } } ;
int main ( ) { gfg g ; int n = 50 ; cout << " Square ▁ root ▁ of ▁ " << n << " ▁ is ▁ " << g . squareRoot ( n ) ; getchar ( ) ; }
float getAvg ( float prev_avg , int x , int n ) { return ( prev_avg * n + x ) / ( n + 1 ) ; }
void streamAvg ( float arr [ ] , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( avg , arr [ i ] , i ) ; printf ( " Average ▁ of ▁ % d ▁ numbers ▁ is ▁ % f ▁ STRNEWLINE " , i + 1 , avg ) ; } return ; }
int main ( ) { float arr [ ] = { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; streamAvg ( arr , n ) ; return 0 ; }
float getAvg ( int x ) { static int sum , n ; sum += x ; return ( ( ( float ) sum ) / ++ n ) ; }
void streamAvg ( float arr [ ] , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( arr [ i ] ) ; cout << " Average ▁ of ▁ " << i + 1 << " ▁ numbers ▁ is ▁ " << fixed << setprecision ( 1 ) << avg << endl ; } return ; }
int main ( ) { float arr [ ] = { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; streamAvg ( arr , n ) ; return 0 ; }
int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
int main ( ) { int n = 8 , k = 2 ; cout << " Value ▁ of ▁ C ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << binomialCoeff ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void primeFactors ( int n ) {
while ( n % 2 == 0 ) { cout << 2 << " ▁ " ; n = n / 2 ; }
for ( int i = 3 ; i <= sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { cout << i << " ▁ " ; n = n / i ; } }
if ( n > 2 ) cout << n << " ▁ " ; }
int main ( ) { int n = 315 ; primeFactors ( n ) ; return 0 ; }
void printCombination ( int arr [ ] , int n , int r ) {
int data [ r ] ;
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; }
void combinationUtil ( int arr [ ] , int data [ ] , int start , int end , int index , int r ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) cout << data [ j ] << " ▁ " ; cout << endl ; return ; }
for ( int i = start ; i <= end && end - i + 1 >= r - index ; i ++ ) { data [ index ] = arr [ i ] ; combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printCombination ( arr , n , r ) ; }
void printCombination ( int arr [ ] , int n , int r ) {
int data [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) cout << data [ j ] << " ▁ " ; cout << endl ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printCombination ( arr , n , r ) ; return 0 ; }
int findgroups ( int arr [ ] , int n ) {
int c [ 3 ] = { 0 } , i ;
int res = 0 ;
for ( i = 0 ; i < n ; i ++ ) c [ arr [ i ] % 3 ] ++ ;
res += ( ( c [ 0 ] * ( c [ 0 ] - 1 ) ) >> 1 ) ;
res += c [ 1 ] * c [ 2 ] ;
res += ( c [ 0 ] * ( c [ 0 ] - 1 ) * ( c [ 0 ] - 2 ) ) / 6 ;
res += ( c [ 1 ] * ( c [ 1 ] - 1 ) * ( c [ 1 ] - 2 ) ) / 6 ;
res += ( ( c [ 2 ] * ( c [ 2 ] - 1 ) * ( c [ 2 ] - 2 ) ) / 6 ) ;
res += c [ 0 ] * c [ 1 ] * c [ 2 ] ;
return res ; }
int main ( ) { int arr [ ] = { 3 , 6 , 7 , 2 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Required ▁ number ▁ of ▁ groups ▁ are ▁ " << findgroups ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; unsigned int nextPowerOf2 ( unsigned int n ) { unsigned count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
int main ( ) { unsigned int n = 0 ; cout << nextPowerOf2 ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; unsigned int nextPowerOf2 ( unsigned int n ) { unsigned int p = 1 ; if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( p < n ) p <<= 1 ; return p ; }
int main ( ) { unsigned int n = 5 ; cout << nextPowerOf2 ( n ) ; return 0 ; }
unsigned int nextPowerOf2 ( unsigned int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
int main ( ) { unsigned int n = 5 ; cout << nextPowerOf2 ( n ) ; return 0 ; }
void segregate0and1 ( int arr [ ] , int n ) {
int count = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 0 ) count ++ ; }
for ( int i = 0 ; i < count ; i ++ ) arr [ i ] = 0 ;
for ( int i = count ; i < n ; i ++ ) arr [ i ] = 1 ; }
void print ( int arr [ ] , int n ) { cout << " Array ▁ after ▁ segregation ▁ is ▁ " ; for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 0 , 1 , 0 , 1 , 1 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; segregate0and1 ( arr , n ) ; print ( arr , n ) ; return 0 ; }
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
int main ( ) { int arr [ ] = { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; segregate0and1 ( arr , arr_size ) ; cout << " Array ▁ after ▁ segregation ▁ " ; for ( i = 0 ; i < 6 ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
void segregate0and1 ( int arr [ ] , int size ) { int type0 = 0 ; int type1 = size - 1 ; while ( type0 < type1 ) { if ( arr [ type0 ] == 1 ) { swap ( arr [ type0 ] , arr [ type1 ] ) ; type1 -- ; } else type0 ++ ; } }
int main ( ) { int arr [ ] = { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; segregate0and1 ( arr , arr_size ) ; cout << " Array ▁ after ▁ segregation ▁ is ▁ " ; for ( i = 0 ; i < arr_size ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void distinctAdjacentElement ( int a [ ] , int n ) {
map < int , int > m ;
for ( int i = 0 ; i < n ; ++ i ) m [ a [ i ] ] ++ ;
int mx = 0 ;
for ( int i = 0 ; i < n ; ++ i ) if ( mx < m [ a [ i ] ] ) mx = m [ a [ i ] ] ;
if ( mx > ( n + 1 ) / 2 ) cout << " NO " << endl ; else cout << " YES " << endl ; }
int main ( ) { int a [ ] = { 7 , 7 , 7 , 7 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; distinctAdjacentElement ( a , n ) ; return 0 ; }
int maxIndexDiff ( int arr [ ] , int n ) { int maxDiff = -1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
int main ( ) { int arr [ ] = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int maxDiff = maxIndexDiff ( arr , n ) ; cout << " STRNEWLINE " << maxDiff ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { vector < long long int > v { 34 , 8 , 10 , 3 , 2 , 80 , 30 , 33 , 1 } ; int n = v . size ( ) ; vector < long long int > maxFromEnd ( n + 1 , INT_MIN ) ;
for ( int i = v . size ( ) - 1 ; i >= 0 ; i -- ) { maxFromEnd [ i ] = max ( maxFromEnd [ i + 1 ] , v [ i ] ) ; } int result = 0 ; for ( int i = 0 ; i < v . size ( ) ; i ++ ) { int low = i + 1 , high = v . size ( ) - 1 , ans = i ; while ( low <= high ) { int mid = ( low + high ) / 2 ; if ( v [ i ] <= maxFromEnd [ mid ] ) {
ans = max ( ans , mid ) ; low = mid + 1 ; } else { high = mid - 1 ; } }
result = max ( result , ans - i ) ; } cout << result << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printRepeating ( int arr [ ] , int size ) {
set < int > s ( arr , arr + size ) ;
for ( auto x : s ) cout << x << " ▁ " ; }
int main ( ) { int arr [ ] = { 1 , 3 , 2 , 2 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , n ) ; return 0 ; }
int minSwapsToSort ( int arr [ ] , int n ) {
pair < int , int > arrPos [ n ] ; for ( int i = 0 ; i < n ; i ++ ) { arrPos [ i ] . first = arr [ i ] ; arrPos [ i ] . second = i ; }
sort ( arrPos , arrPos + n ) ;
vector < bool > vis ( n , false ) ;
int ans = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( vis [ i ] arrPos [ i ] . second == i ) continue ;
int cycle_size = 0 ; int j = i ; while ( ! vis [ j ] ) { vis [ j ] = 1 ;
j = arrPos [ j ] . second ; cycle_size ++ ; }
ans += ( cycle_size - 1 ) ; }
return ans ; }
int minSwapToMakeArraySame ( int a [ ] , int b [ ] , int n ) {
map < int , int > mp ; for ( int i = 0 ; i < n ; i ++ ) mp [ b [ i ] ] = i ;
for ( int i = 0 ; i < n ; i ++ ) b [ i ] = mp [ a [ i ] ] ;
return minSwapsToSort ( b , n ) ; }
int main ( ) { int a [ ] = { 3 , 6 , 4 , 8 } ; int b [ ] = { 4 , 6 , 8 , 3 } ; int n = sizeof ( a ) / sizeof ( int ) ; cout << minSwapToMakeArraySame ( a , b , n ) ; return 0 ; }
int missingK ( int a [ ] , int k , int n ) { int difference = 0 , ans = 0 , count = k ; bool flag = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = 1 ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return -1 ; }
int a [ ] = { 1 , 5 , 11 , 19 } ;
int k = 11 ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ;
int missing = missingK ( a , k , n ) ; cout << missing << endl ; return 0 ; }
int missingK ( vector < int > & arr , int k ) { int n = arr . size ( ) ; int l = 0 , u = n - 1 , mid ; while ( l <= u ) { mid = ( l + u ) / 2 ; int numbers_less_than_mid = arr [ mid ] - ( mid + 1 ) ;
if ( numbers_less_than_mid == k ) {
if ( mid > 0 && ( arr [ mid - 1 ] - ( mid ) ) == k ) { u = mid - 1 ; continue ; }
return arr [ mid ] - 1 ; }
if ( numbers_less_than_mid < k ) { l = mid + 1 ; } else if ( k < numbers_less_than_mid ) { u = mid - 1 ; } }
if ( u < 0 ) return k ;
int less = arr [ u ] - ( u + 1 ) ; k -= less ;
return arr [ u ] + k ; }
int main ( ) { vector < int > arr = { 2 , 3 , 4 , 7 , 11 } ; int k = 5 ;
cout << " Missing ▁ kth ▁ number ▁ = ▁ " << missingK ( arr , k ) << endl ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
void printList ( struct Node * node ) { while ( node != NULL ) { cout << node -> data << " ▁ " ; node = node -> next ; } cout << endl ; }
Node * newNode ( int key ) { Node * temp = new Node ; temp -> data = key ; temp -> next = NULL ; return temp ; }
Node * insertBeg ( Node * head , int val ) { Node * temp = newNode ( val ) ; temp -> next = head ; head = temp ; return head ; }
void rearrangeOddEven ( Node * head ) { stack < Node * > odd ; stack < Node * > even ; int i = 1 ; while ( head != nullptr ) { if ( head -> data % 2 != 0 && i % 2 == 0 ) {
odd . push ( head ) ; } else if ( head -> data % 2 == 0 && i % 2 != 0 ) {
even . push ( head ) ; } head = head -> next ; i ++ ; } while ( ! odd . empty ( ) && ! even . empty ( ) ) {
swap ( odd . top ( ) -> data , even . top ( ) -> data ) ; odd . pop ( ) ; even . pop ( ) ; } }
int main ( ) { Node * head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 1 ) ; cout << " Linked ▁ List : " << endl ; printList ( head ) ; rearrangeOddEven ( head ) ; cout << " Linked ▁ List ▁ after ▁ " << " Rearranging : " << endl ; printList ( head ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
void printList ( struct Node * node ) { while ( node != NULL ) { cout << node -> data << " ▁ " ; node = node -> next ; } cout << endl ; }
Node * newNode ( int key ) { Node * temp = new Node ; temp -> data = key ; temp -> next = NULL ; return temp ; }
Node * insertBeg ( Node * head , int val ) { Node * temp = newNode ( val ) ; temp -> next = head ; head = temp ; return head ; }
void rearrange ( Node * * head ) {
Node * even ; Node * temp , * prev_temp ; Node * i , * j , * k , * l , * ptr ;
temp = ( * head ) -> next ; prev_temp = * head ; while ( temp != nullptr ) {
Node * x = temp -> next ;
if ( temp -> data % 2 != 0 ) { prev_temp -> next = x ; temp -> next = ( * head ) ; ( * head ) = temp ; } else { prev_temp = temp ; }
temp = x ; }
temp = ( * head ) -> next ; prev_temp = ( * head ) ; while ( temp != nullptr && temp -> data % 2 != 0 ) { prev_temp = temp ; temp = temp -> next ; } even = temp ;
prev_temp -> next = nullptr ;
i = * head ; j = even ; while ( j != nullptr && i != nullptr ) {
k = i -> next ; l = j -> next ; i -> next = j ; j -> next = k ;
ptr = j ;
i = k ; j = l ; } if ( i == nullptr ) {
ptr -> next = j ; }
}
int main ( ) { Node * head = newNode ( 8 ) ; head = insertBeg ( head , 7 ) ; head = insertBeg ( head , 6 ) ; head = insertBeg ( head , 3 ) ; head = insertBeg ( head , 5 ) ; head = insertBeg ( head , 1 ) ; head = insertBeg ( head , 2 ) ; head = insertBeg ( head , 10 ) ; cout << " Linked ▁ List : " << endl ; printList ( head ) ; cout << " Rearranged ▁ List " << endl ; rearrange ( & head ) ; printList ( head ) ; }
void print ( vector < vector < int > > & mat ) {
for ( int i = 0 ; i < mat . size ( ) ; i ++ ) {
for ( int j = 0 ; j < mat [ 0 ] . size ( ) ; j ++ )
cout << setw ( 3 ) << mat [ i ] [ j ] ; cout << " STRNEWLINE " ; } }
void performSwap ( vector < vector < int > > & mat , int i , int j ) { int N = mat . size ( ) ;
int ei = N - 1 - i ;
int ej = N - 1 - j ;
int temp = mat [ i ] [ j ] ; mat [ i ] [ j ] = mat [ ej ] [ i ] ; mat [ ej ] [ i ] = mat [ ei ] [ ej ] ; mat [ ei ] [ ej ] = mat [ j ] [ ei ] ; mat [ j ] [ ei ] = temp ; }
void rotate ( vector < vector < int > > & mat , int N , int K ) {
K = K % 4 ;
while ( K -- ) {
for ( int i = 0 ; i < N / 2 ; i ++ ) {
for ( int j = i ; j < N - i - 1 ; j ++ ) {
if ( i != j && ( i + j ) != N - 1 ) {
performSwap ( mat , i , j ) ; } } } }
print ( mat ) ; }
int main ( ) { int K = 5 ; vector < vector < int > > mat = { { 1 , 2 , 3 , 4 } , { 6 , 7 , 8 , 9 } , { 11 , 12 , 13 , 14 } , { 16 , 17 , 18 , 19 } , } ; int N = mat . size ( ) ; rotate ( mat , N , K ) ; return 0 ; }
int findRotations ( string str ) {
string tmp = str + str ; int n = str . length ( ) ; for ( int i = 1 ; i <= n ; i ++ ) {
string substring = tmp . substr ( i , str . size ( ) ) ;
if ( str == substring ) return i ; } return n ; }
int main ( ) { string str = " abc " ; cout << findRotations ( str ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 10000 ;
int prefix [ MAX + 1 ] ; bool isPowerOfTwo ( int x ) { if ( x && ( ! ( x & ( x - 1 ) ) ) ) return true ; return false ; }
void computePrefix ( int n , int a [ ] ) {
if ( isPowerOfTwo ( a [ 0 ] ) ) prefix [ 0 ] = 1 ; for ( int i = 1 ; i < n ; i ++ ) { prefix [ i ] = prefix [ i - 1 ] ; if ( isPowerOfTwo ( a [ i ] ) ) prefix [ i ] ++ ; } }
int query ( int L , int R ) { return prefix [ R ] - prefix [ L - 1 ] ; }
int main ( ) { int A [ ] = { 3 , 8 , 5 , 2 , 5 , 10 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int Q = 2 ; computePrefix ( N , A ) ; cout << query ( 0 , 4 ) << " STRNEWLINE " ; cout << query ( 3 , 5 ) << " STRNEWLINE " ; return 0 ; }
void countIntgralPoints ( int x1 , int y1 , int x2 , int y2 ) { cout << ( y2 - y1 - 1 ) * ( x2 - x1 - 1 ) ; }
int main ( ) { int x1 = 1 , y1 = 1 ; int x2 = 4 , y2 = 4 ; countIntgralPoints ( x1 , y1 , x2 , y2 ) ; return 0 ; }
void findNextNumber ( int n ) { int h [ 10 ] = { 0 } ; int i = 0 , msb = n , rem = 0 ; int next_num = -1 , count = 0 ;
while ( msb > 9 ) { rem = msb % 10 ; h [ rem ] = 1 ; msb /= 10 ; count ++ ; } h [ msb ] = 1 ; count ++ ;
for ( i = msb + 1 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; break ; } }
if ( next_num == -1 ) { for ( i = 1 ; i < msb ; i ++ ) { if ( h [ i ] == 0 ) { next_num = i ; count ++ ; break ; } } }
if ( next_num > 0 ) {
for ( i = 0 ; i < 10 ; i ++ ) { if ( h [ i ] == 0 ) { msb = i ; break ; } }
for ( i = 1 ; i < count ; i ++ ) { next_num = ( ( next_num * 10 ) + msb ) ; }
if ( next_num > n ) cout << next_num << " STRNEWLINE " ; else cout << " Not ▁ Possible ▁ STRNEWLINE " ; } else { cout << " Not ▁ Possible ▁ STRNEWLINE " ; } }
int main ( ) { int n = 2019 ; findNextNumber ( n ) ; return 0 ; }
void CalculateValues ( int N ) { int A = 0 , B = 0 , C = 0 ;
for ( C = 0 ; C < N / 7 ; C ++ ) {
for ( B = 0 ; B < N / 5 ; B ++ ) {
int A = N - 7 * C - 5 * B ;
if ( A >= 0 && A % 3 == 0 ) { cout << " A ▁ = ▁ " << A / 3 << " , ▁ B ▁ = ▁ " << B << " , ▁ C ▁ = ▁ " << C << endl ; return ; } } }
cout << -1 << endl ; }
int main ( ) { int N = 19 ; CalculateValues ( 19 ) ; return 0 ; }
void minimumTime ( int * arr , int n ) {
int sum = 0 ;
int T = * max_element ( arr , arr + n ) ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
cout << max ( 2 * T , sum ) ; }
int main ( ) { int arr [ ] = { 2 , 8 , 3 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
minimumTime ( arr , N ) ; return 0 ; }
void lexicographicallyMax ( string s ) {
int n = s . size ( ) ;
for ( int i = 0 ; i < n ; i ++ ) {
int count = 0 ;
int beg = i ;
int end = i ;
if ( s [ i ] == '1' ) count ++ ;
for ( int j = i + 1 ; j < n ; j ++ ) { if ( s [ j ] == '1' ) count ++ ; if ( count % 2 == 0 && count != 0 ) { end = j ; break ; } }
reverse ( s . begin ( ) + beg , s . begin ( ) + end + 1 ) ; }
cout << s << " STRNEWLINE " ; }
int main ( ) { string S = "0101" ; lexicographicallyMax ( S ) ; return 0 ; }
void maxPairs ( int nums [ ] , int n , int k ) {
sort ( nums , nums + n ) ;
int result = 0 ;
int start = 0 , end = n - 1 ;
while ( start < end ) { if ( nums [ start ] + nums [ end ] > k )
end -- ; else if ( nums [ start ] + nums [ end ] < k )
start ++ ;
else { start ++ ; end -- ; result ++ ; } }
cout << result << endl ; ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int K = 5 ;
maxPairs ( arr , n , K ) ; return 0 ; }
void maxPairs ( vector < int > nums , int k ) {
map < int , int > m ;
int result = 0 ;
for ( auto i : nums ) {
if ( m . find ( i ) != m . end ( ) && m [ i ] > 0 ) { m [ i ] = m [ i ] - 1 ; result ++ ; }
else { m [ k - i ] = m [ k - i ] + 1 ; } }
cout << result ; }
int main ( ) { vector < int > arr = { 1 , 2 , 3 , 4 } ; int K = 5 ;
maxPairs ( arr , K ) ; }
void removeIndicesToMakeSumEqual ( vector < int > & arr ) {
int N = arr . size ( ) ;
vector < int > odd ( N , 0 ) ;
vector < int > even ( N , 0 ) ;
even [ 0 ] = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) {
odd [ i ] = odd [ i - 1 ] ;
even [ i ] = even [ i - 1 ] ;
if ( i % 2 == 0 ) {
even [ i ] += arr [ i ] ; }
else {
odd [ i ] += arr [ i ] ; } }
bool find = 0 ;
int p = odd [ N - 1 ] ;
int q = even [ N - 1 ] - arr [ 0 ] ;
if ( p == q ) { cout << "0 ▁ " ; find = 1 ; }
for ( int i = 1 ; i < N ; i ++ ) {
if ( i % 2 == 0 ) {
p = even [ N - 1 ] - even [ i - 1 ] - arr [ i ] + odd [ i - 1 ] ;
q = odd [ N - 1 ] - odd [ i - 1 ] + even [ i - 1 ] ; } else {
q = odd [ N - 1 ] - odd [ i - 1 ] - arr [ i ] + even [ i - 1 ] ;
p = even [ N - 1 ] - even [ i - 1 ] + odd [ i - 1 ] ; }
if ( p == q ) {
find = 1 ;
cout << i << " ▁ " ; } }
if ( ! find ) {
cout << -1 ; } }
int main ( ) { vector < int > arr = { 4 , 1 , 6 , 2 } ; removeIndicesToMakeSumEqual ( arr ) ; return 0 ; }
void min_element_removal ( int arr [ ] , int N ) {
vector < int > left ( N , 1 ) ;
vector < int > right ( N , 1 ) ;
for ( int i = 1 ; i < N ; i ++ ) {
for ( int j = 0 ; j < i ; j ++ ) {
if ( arr [ j ] < arr [ i ] ) {
left [ i ] = max ( left [ i ] , left [ j ] + 1 ) ; } } }
for ( int i = N - 2 ; i >= 0 ; i -- ) {
for ( int j = N - 1 ; j > i ; j -- ) {
if ( arr [ i ] > arr [ j ] ) {
right [ i ] = max ( right [ i ] , right [ j ] + 1 ) ; } } }
int maxLen = 0 ;
for ( int i = 1 ; i < N - 1 ; i ++ ) {
maxLen = max ( maxLen , left [ i ] + right [ i ] - 1 ) ; } cout << ( N - maxLen ) << " STRNEWLINE " ; }
void makeBitonic ( int arr [ ] , int N ) { if ( N == 1 ) { cout << "0" << endl ; return ; } if ( N == 2 ) { if ( arr [ 0 ] != arr [ 1 ] ) cout << "0" << endl ; else cout << "1" << endl ; return ; } min_element_removal ( arr , N ) ; }
int main ( ) { int arr [ ] = { 2 , 1 , 1 , 5 , 6 , 2 , 3 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; makeBitonic ( arr , N ) ; return 0 ; }
void countSubarrays ( int A [ ] , int N ) {
int ans = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) {
if ( A [ i ] != A [ i + 1 ] ) {
ans ++ ;
for ( int j = i - 1 , k = i + 2 ; j >= 0 && k < N && A [ j ] == A [ i ] && A [ k ] == A [ i + 1 ] ; j -- , k ++ ) {
ans ++ ; } } }
cout << ans << " STRNEWLINE " ; }
int main ( ) { int A [ ] = { 1 , 1 , 0 , 0 , 1 , 0 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
countSubarrays ( A , N ) ; return 0 ; }
#include <cstring> NEW_LINE #include <iostream> NEW_LINE using namespace std ; const int maxN = 2002 ;
int lcount [ maxN ] [ maxN ] ;
int rcount [ maxN ] [ maxN ] ;
void fill_counts ( int a [ ] , int n ) { int i , j ;
int maxA = a [ 0 ] ; for ( i = 0 ; i < n ; i ++ ) { if ( a [ i ] > maxA ) { maxA = a [ i ] ; } } memset ( lcount , 0 , sizeof ( lcount ) ) ; memset ( rcount , 0 , sizeof ( rcount ) ) ; for ( i = 0 ; i < n ; i ++ ) { lcount [ a [ i ] ] [ i ] = 1 ; rcount [ a [ i ] ] [ i ] = 1 ; } for ( i = 0 ; i <= maxA ; i ++ ) {
for ( j = 0 ; j < n ; j ++ ) { lcount [ i ] [ j ] = lcount [ i ] [ j - 1 ] + lcount [ i ] [ j ] ; }
for ( j = n - 2 ; j >= 0 ; j -- ) { rcount [ i ] [ j ] = rcount [ i ] [ j + 1 ] + rcount [ i ] [ j ] ; } } }
int countSubsequence ( int a [ ] , int n ) { int i , j ; fill_counts ( a , n ) ; int answer = 0 ; for ( i = 1 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n - 1 ; j ++ ) { answer += lcount [ a [ j ] ] [ i - 1 ] * rcount [ a [ i ] ] [ j + 1 ] ; } } return answer ; }
int main ( ) { int a [ 7 ] = { 1 , 2 , 3 , 2 , 1 , 3 , 2 } ; cout << countSubsequence ( a , 7 ) ; return 0 ; }
string removeOuterParentheses ( string S ) {
string res ;
int count = 0 ;
for ( char c : S ) {
if ( c == ' ( ' && count ++ > 0 )
res += c ;
if ( c == ' ) ' && count -- > 1 )
res += c ; }
return res ; }
int main ( ) { string S = " ( ( ) ( ) ) ( ( ) ) ( ) " ; cout << removeOuterParentheses ( S ) ; }
int maxiConsecutiveSubarray ( int arr [ ] , int N ) {
int maxi = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) {
int cnt = 1 , j ; for ( j = i ; j < N ; j ++ ) {
if ( arr [ j + 1 ] == arr [ j ] + 1 ) { cnt ++ ; }
else { break ; } }
maxi = max ( maxi , cnt ) ; i = j ; }
return maxi ; }
int main ( ) { int N = 11 ; int arr [ ] = { 1 , 3 , 4 , 2 , 3 , 4 , 2 , 3 , 5 , 6 , 7 } ; cout << maxiConsecutiveSubarray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  100005
void SieveOfEratosthenes ( bool prime [ ] , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
int digitSum ( int number ) {
int sum = 0 ; while ( number > 0 ) {
sum += ( number % 10 ) ; number /= 10 ; }
return sum ; }
void longestCompositeDigitSumSubsequence ( int arr [ ] , int n ) { int count = 0 ; bool prime [ N + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ; SieveOfEratosthenes ( prime , N ) ; for ( int i = 0 ; i < n ; i ++ ) {
int res = digitSum ( arr [ i ] ) ;
if ( res == 1 ) { continue ; }
if ( ! prime [ res ] ) { count ++ ; } } cout << count << endl ; }
int main ( ) { int arr [ ] = { 13 , 55 , 7 , 3 , 5 , 1 , 10 , 21 , 233 , 144 , 89 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
longestCompositeDigitSumSubsequence ( arr , n ) ; return 0 ; }
struct Node { int data ; Node * left , * right ; } ;
Node * newnode ( int data ) { Node * temp = new Node ; temp -> data = data ; temp -> left = NULL ; temp -> right = NULL ;
return temp ; }
Node * insert ( string s , int i , int N , Node * root , Node * temp ) { if ( i == N ) return temp ;
if ( s [ i ] == ' L ' ) root -> left = insert ( s , i + 1 , N , root -> left , temp ) ;
else root -> right = insert ( s , i + 1 , N , root -> right , temp ) ;
return root ; }
int SBTUtil ( Node * root , int & sum ) {
if ( root == NULL ) return 0 ; if ( root -> left == NULL && root -> right == NULL ) return root -> data ;
int left = SBTUtil ( root -> left , sum ) ;
int right = SBTUtil ( root -> right , sum ) ;
if ( root -> left && root -> right ) {
if ( ( left % 2 == 0 && right % 2 != 0 ) || ( left % 2 != 0 && right % 2 == 0 ) ) { sum += root -> data ; } }
return left + right + root -> data ; }
Node * build_tree ( int R , int N , string str [ ] , int values [ ] ) {
Node * root = newnode ( R ) ; int i ;
for ( i = 0 ; i < N - 1 ; i ++ ) { string s = str [ i ] ; int x = values [ i ] ;
Node * temp = newnode ( x ) ;
root = insert ( s , 0 , s . size ( ) , root , temp ) ; }
return root ; }
void speciallyBalancedNodes ( int R , int N , string str [ ] , int values [ ] ) {
Node * root = build_tree ( R , N , str , values ) ;
int sum = 0 ;
SBTUtil ( root , sum ) ;
cout << sum << " ▁ " ; }
int main ( ) {
int N = 7 ;
int R = 12 ;
string str [ N - 1 ] = { " L " , " R " , " RL " , " RR " , " RLL " , " RLR " } ;
int values [ N - 1 ] = { 17 , 16 , 4 , 9 , 2 , 3 } ;
speciallyBalancedNodes ( R , N , str , values ) ; return 0 ; }
void position ( int arr [ ] [ 2 ] , int N ) {
int pos = -1 ;
int count ;
for ( int i = 0 ; i < N ; i ++ ) {
count = 0 ; for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ 0 ] <= arr [ j ] [ 0 ] && arr [ i ] [ 1 ] >= arr [ j ] [ 1 ] ) { count ++ ; } }
if ( count == N ) { pos = i ; } }
if ( pos == -1 ) { cout << pos ; }
else { cout << pos + 1 ; } }
int main ( ) {
int arr [ ] [ 2 ] = { { 3 , 3 } , { 1 , 3 } , { 2 , 2 } , { 2 , 3 } , { 1 , 2 } } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
position ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void position ( int arr [ ] [ 2 ] , int N ) {
int pos = -1 ;
int right = INT_MIN ;
int left = INT_MAX ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] [ 1 ] > right ) { right = arr [ i ] [ 1 ] ; }
if ( arr [ i ] [ 0 ] < left ) { left = arr [ i ] [ 0 ] ; } }
for ( int i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] [ 0 ] == left && arr [ i ] [ 1 ] == right ) { pos = i + 1 ; } }
cout << pos << endl ; }
int main ( ) {
int arr [ ] [ 2 ] = { { 3 , 3 } , { 1 , 3 } , { 2 , 2 } , { 2 , 3 } , { 1 , 2 } } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
position ( arr , N ) ; }
int ctMinEdits ( string str1 , string str2 ) { int N1 = str1 . length ( ) ; int N2 = str2 . length ( ) ;
int freq1 [ 256 ] = { 0 } ; for ( int i = 0 ; i < N1 ; i ++ ) { freq1 [ str1 [ i ] ] ++ ; }
int freq2 [ 256 ] = { 0 } ; for ( int i = 0 ; i < N2 ; i ++ ) { freq2 [ str2 [ i ] ] ++ ; }
for ( int i = 0 ; i < 256 ; i ++ ) {
if ( freq1 [ i ] > freq2 [ i ] ) { freq1 [ i ] = freq1 [ i ] - freq2 [ i ] ; freq2 [ i ] = 0 ; }
else { freq2 [ i ] = freq2 [ i ] - freq1 [ i ] ; freq1 [ i ] = 0 ; } }
int sum1 = 0 ;
int sum2 = 0 ; for ( int i = 0 ; i < 256 ; i ++ ) { sum1 += freq1 [ i ] ; sum2 += freq2 [ i ] ; } return max ( sum1 , sum2 ) ; }
int main ( ) { string str1 = " geeksforgeeks " ; string str2 = " geeksforcoder " ; cout << ctMinEdits ( str1 , str2 ) ; }
int CountPairs ( int * a , int * b , int n ) {
int C [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) { C [ i ] = a [ i ] + b [ i ] ; }
map < int , int > freqCount ; for ( int i = 0 ; i < n ; i ++ ) { freqCount [ C [ i ] ] ++ ; }
int NoOfPairs = 0 ; for ( auto x : freqCount ) { int y = x . second ;
NoOfPairs = NoOfPairs + y * ( y - 1 ) / 2 ; }
cout << NoOfPairs ; }
int arr [ ] = { 1 , 4 , 20 , 3 , 10 , 5 } ; int brr [ ] = { 9 , 6 , 1 , 7 , 11 , 6 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
CountPairs ( arr , brr , N ) ; return 0 ; }
void medianChange ( vector < int > & arr1 , vector < int > & arr2 ) { int N = arr1 . size ( ) ;
vector < float > median ;
if ( N & 1 ) { median . push_back ( arr1 [ N / 2 ] * 1.0 ) ; }
else { median . push_back ( ( arr1 [ N / 2 ] + arr1 [ ( N - 1 ) / 2 ] ) / 2.0 ) ; } for ( auto & x : arr2 ) {
auto it = find ( arr1 . begin ( ) , arr1 . end ( ) , x ) ;
arr1 . erase ( it ) ;
N -- ;
if ( N & 1 ) { median . push_back ( arr1 [ N / 2 ] * 1.0 ) ; }
else { median . push_back ( ( arr1 [ N / 2 ] + arr1 [ ( N - 1 ) / 2 ] ) / 2.0 ) ; } }
for ( int i = 0 ; i < median . size ( ) - 1 ; i ++ ) { cout << median [ i + 1 ] - median [ i ] << ' ▁ ' ; } }
int main ( ) {
vector < int > arr1 = { 2 , 4 , 6 , 8 , 10 } ; vector < int > arr2 = { 4 , 6 } ;
medianChange ( arr1 , arr2 ) ; return 0 ; }
int nfa = 1 ;
int flag = 0 ; using namespace std ;
void state1 ( char c ) {
if ( c == ' a ' ) nfa = 2 ; else if ( c == ' b ' c == ' c ' ) nfa = 1 ; else flag = 1 ; }
void state2 ( char c ) {
if ( c == ' a ' ) nfa = 3 ; else if ( c == ' b ' c == ' c ' ) nfa = 2 ; else flag = 1 ; }
void state3 ( char c ) {
if ( c == ' a ' ) nfa = 1 ; else if ( c == ' b ' c == ' c ' ) nfa = 3 ; else flag = 1 ; }
void state4 ( char c ) {
if ( c == ' b ' ) nfa = 5 ; else if ( c == ' a ' c == ' c ' ) nfa = 4 ; else flag = 1 ; }
void state5 ( char c ) {
if ( c == ' b ' ) nfa = 6 ; else if ( c == ' a ' c == ' c ' ) nfa = 5 ; else flag = 1 ; }
void state6 ( char c ) {
if ( c == ' b ' ) nfa = 4 ; else if ( c == ' a ' c == ' c ' ) nfa = 6 ; else flag = 1 ; }
void state7 ( char c ) {
if ( c == ' c ' ) nfa = 8 ; else if ( c == ' b ' c == ' a ' ) nfa = 7 ; else flag = 1 ; }
void state8 ( char c ) {
if ( c == ' c ' ) nfa = 9 ; else if ( c == ' b ' c == ' a ' ) nfa = 8 ; else flag = 1 ; }
void state9 ( char c ) {
if ( c == ' c ' ) nfa = 7 ; else if ( c == ' b ' c == ' a ' ) nfa = 9 ; else flag = 1 ; }
bool checkA ( string s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 1 ) state1 ( s [ i ] ) ; else if ( nfa == 2 ) state2 ( s [ i ] ) ; else if ( nfa == 3 ) state3 ( s [ i ] ) ; } if ( nfa == 1 ) { return true ; } else { nfa = 4 ; } }
bool checkB ( string s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 4 ) state4 ( s [ i ] ) ; else if ( nfa == 5 ) state5 ( s [ i ] ) ; else if ( nfa == 6 ) state6 ( s [ i ] ) ; } if ( nfa == 4 ) { return true ; } else { nfa = 7 ; } }
bool checkC ( string s , int x ) { for ( int i = 0 ; i < x ; i ++ ) { if ( nfa == 7 ) state7 ( s [ i ] ) ; else if ( nfa == 8 ) state8 ( s [ i ] ) ; else if ( nfa == 9 ) state9 ( s [ i ] ) ; } if ( nfa == 7 ) { return true ; } }
int main ( ) { string s = " bbbca " ; int x = 5 ;
if ( checkA ( s , x ) || checkB ( s , x ) || checkC ( s , x ) ) { cout << " ACCEPTED " ; } else { if ( flag == 0 ) { cout << " NOT ▁ ACCEPTED " ; return 0 ; } else { cout << " INPUT ▁ OUT ▁ OF ▁ DICTIONARY . " ; return 0 ; } } }
int getPositionCount ( int a [ ] , int n ) {
int count = 1 ;
int min = a [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) {
if ( a [ i ] <= min ) {
min = a [ i ] ;
count ++ ; } } return count ; }
int main ( ) { int a [ ] = { 5 , 4 , 6 , 1 , 3 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << getPositionCount ( a , n ) ; return 0 ; }
int maxSum ( int arr [ ] , int n , int k ) {
if ( n < k ) { return -1 ; }
int res = 0 ; for ( int i = 0 ; i < k ; i ++ ) res += arr [ i ] ;
int curr_sum = res ; for ( int i = k ; i < n ; i ++ ) { curr_sum += arr [ i ] - arr [ i - k ] ; res = max ( res , curr_sum ) ; } return res ; }
int solve ( int arr [ ] , int n , int k ) { int max_len = 0 , l = 0 , r = n , m ;
while ( l <= r ) { m = ( l + r ) / 2 ;
if ( maxSum ( arr , n , m ) > k ) r = m - 1 ; else { l = m + 1 ;
max_len = m ; } } return max_len ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( int ) ; int k = 10 ; cout << solve ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ll  long long int NEW_LINE #define MAX  100001 NEW_LINE #define ROW  10 NEW_LINE #define COl  3 NEW_LINE vector < int > indices [ MAX ] ;
int test [ ROW ] [ COl ] = { { 2 , 3 , 6 } , { 2 , 4 , 4 } , { 2 , 6 , 3 } , { 3 , 2 , 6 } , { 3 , 3 , 3 } , { 3 , 6 , 2 } , { 4 , 2 , 4 } , { 4 , 4 , 2 } , { 6 , 2 , 3 } , { 6 , 3 , 2 } } ;
int find_triplet ( int array [ ] , int n ) { int answer = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { indices [ array [ i ] ] . push_back ( i ) ; } for ( int i = 0 ; i < n ; i ++ ) { int y = array [ i ] ; for ( int j = 0 ; j < ROW ; j ++ ) { int s = test [ j ] [ 1 ] * y ;
if ( s % test [ j ] [ 0 ] != 0 ) continue ; if ( s % test [ j ] [ 2 ] != 0 ) continue ; int x = s / test [ j ] [ 0 ] ; ll z = s / test [ j ] [ 2 ] ; if ( x > MAX z > MAX ) continue ; int l = 0 ; int r = indices [ x ] . size ( ) - 1 ; int first = -1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( indices [ x ] [ m ] < i ) { first = m ; l = m + 1 ; } else { r = m - 1 ; } } l = 0 ; r = indices [ z ] . size ( ) - 1 ; int third = -1 ;
while ( l <= r ) { int m = ( l + r ) / 2 ; if ( indices [ z ] [ m ] > i ) { third = m ; r = m - 1 ; } else { l = m + 1 ; } } if ( first != -1 && third != -1 ) {
answer += ( first + 1 ) * ( indices [ z ] . size ( ) - third ) ; } } } return answer ; }
int main ( ) { int array [ ] = { 2 , 4 , 5 , 6 , 7 } ; int n = sizeof ( array ) / sizeof ( array [ 0 ] ) ; cout << find_triplet ( array , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int distinct ( int arr [ ] , int n ) { int count = 0 ;
if ( n == 1 ) return 1 ; for ( int i = 0 ; i < n - 1 ; i ++ ) {
if ( i == 0 ) { if ( arr [ i ] != arr [ i + 1 ] ) count += 1 ; }
else { if ( arr [ i ] != arr [ i + 1 ] arr [ i ] != arr [ i - 1 ] ) count += 1 ; } }
if ( arr [ n - 1 ] != arr [ n - 2 ] ) count += 1 ; return count ; }
int main ( ) { int arr [ ] = { 0 , 0 , 0 , 0 , 0 , 1 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << distinct ( arr , n ) ; return 0 ; }
bool isSorted ( pair < int , int > * arr , int N ) {
for ( int i = 1 ; i < N ; i ++ ) { if ( arr [ i ] . first > arr [ i - 1 ] . first ) { return false ; } }
return true ; }
string isPossibleToSort ( pair < int , int > * arr , int N ) {
int group = arr [ 0 ] . second ;
for ( int i = 1 ; i < N ; i ++ ) {
if ( arr [ i ] . second != group ) { return " Yes " ; } }
if ( isSorted ( arr , N ) ) { return " Yes " ; } else { return " No " ; } }
int main ( ) { pair < int , int > arr [ ] = { { 340000 , 2 } , { 45000 , 1 } , { 30000 , 2 } , { 50000 , 4 } } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << isPossibleToSort ( arr , N ) ; return 0 ; }
struct Node { Node * left , * right ; int data ; Node ( int x ) { data = x ; left = NULL ; right = NULL ; } } ;
long getAlphaScore ( Node * node ) {
if ( node -> left != NULL ) getAlphaScore ( node -> left ) ;
sum = ( sum + node -> data ) % mod ;
total_sum = ( total_sum + sum ) % mod ;
if ( node -> right != NULL ) getAlphaScore ( node -> right ) ;
return total_sum ; }
Node * constructBST ( int arr [ ] , int start , int end , Node * root ) { if ( start > end ) return NULL ; int mid = ( start + end ) / 2 ;
if ( root == NULL ) root = new Node ( arr [ mid ] ) ;
root -> left = constructBST ( arr , start , mid - 1 , root -> left ) ;
root -> right = constructBST ( arr , mid + 1 , end , root -> right ) ;
return root ; }
int main ( ) { int arr [ ] = { 10 , 11 , 12 } ; int length = 3 ;
sort ( arr , arr + length ) ; Node * root = NULL ;
root = constructBST ( arr , 0 , length - 1 , root ) ; cout << ( getAlphaScore ( root ) ) ; }
int sortByFreq ( int * arr , int n ) {
int maxE = -1 ;
for ( int i = 0 ; i < n ; i ++ ) { maxE = max ( maxE , arr [ i ] ) ; }
int freq [ maxE + 1 ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; }
int cnt = 0 ;
for ( int i = 0 ; i <= maxE ; i ++ ) {
if ( freq [ i ] > 0 ) { int value = 100000 - i ; arr [ cnt ] = 100000 * freq [ i ] + value ; cnt ++ ; } }
return cnt ; }
void printSortedArray ( int * arr , int cnt ) {
for ( int i = 0 ; i < cnt ; i ++ ) {
int frequency = arr [ i ] / 100000 ;
int value = 100000 - ( arr [ i ] % 100000 ) ;
for ( int j = 0 ; j < frequency ; j ++ ) { cout << value << ' ▁ ' ; } } }
int main ( ) { int arr [ ] = { 4 , 4 , 5 , 6 , 4 , 2 , 2 , 8 , 5 } ;
int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int cnt = sortByFreq ( arr , n ) ;
sort ( arr , arr + cnt , greater < int > ( ) ) ;
printSortedArray ( arr , cnt ) ; return 0 ; }
bool checkRectangles ( int * arr , int n ) { bool ans = true ;
sort ( arr , arr + 4 * n ) ;
int area = arr [ 0 ] * arr [ 4 * n - 1 ] ;
for ( int i = 0 ; i < 2 * n ; i = i + 2 ) { if ( arr [ i ] != arr [ i + 1 ] arr [ 4 * n - i - 1 ] != arr [ 4 * n - i - 2 ] arr [ i ] * arr [ 4 * n - i - 1 ] != area ) {
ans = false ; break ; } }
if ( ans ) return true ; return false ; }
int main ( ) { int arr [ ] = { 1 , 8 , 2 , 1 , 2 , 4 , 4 , 8 } ; int n = 2 ; if ( checkRectangles ( arr , n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
int cntElements ( int arr [ ] , int n ) {
int copy_arr [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) copy_arr [ i ] = arr [ i ] ;
int count = 0 ;
sort ( arr , arr + n ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] != copy_arr [ i ] ) { count ++ ; } } return count ; }
int main ( ) { int arr [ ] = { 1 , 2 , 6 , 2 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << cntElements ( arr , n ) ; return 0 ; }
void findPairs ( int arr [ ] , int n , int k , int d ) {
if ( n < 2 * k ) { cout << -1 ; return ; }
vector < pair < int , int > > pairs ;
sort ( arr , arr + n ) ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( arr [ n - k + i ] - arr [ i ] >= d ) {
pair < int , int > p = make_pair ( arr [ i ] , arr [ n - k + i ] ) ; pairs . push_back ( p ) ; } }
if ( pairs . size ( ) < k ) { cout << -1 ; return ; }
for ( auto v : pairs ) { cout << " ( " << v . first << " , ▁ " << v . second << " ) " << endl ; } }
int main ( ) { int arr [ ] = { 4 , 6 , 10 , 23 , 14 , 7 , 2 , 20 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 4 , d = 3 ; findPairs ( arr , n , k , d ) ; return 0 ; }
int pairs_count ( int arr [ ] , int n , int sum ) {
int ans = 0 ;
sort ( arr , arr + n ) ;
int i = 0 , j = n - 1 ; while ( i < j ) {
if ( arr [ i ] + arr [ j ] < sum ) i ++ ;
else if ( arr [ i ] + arr [ j ] > sum ) j -- ;
else {
int x = arr [ i ] , xx = i ; while ( i < j and arr [ i ] == x ) i ++ ;
int y = arr [ j ] , yy = j ; while ( j >= i and arr [ j ] == y ) j -- ;
if ( x == y ) { int temp = i - xx + yy - j - 1 ; ans += ( temp * ( temp + 1 ) ) / 2 ; } else ans += ( i - xx ) * ( yy - j ) ; } }
return ans ; }
int main ( ) { int arr [ ] = { 1 , 5 , 7 , 5 , -1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int sum = 6 ; cout << pairs_count ( arr , n , sum ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool check ( string str ) { int min = INT_MAX ; int max = - INT_MAX ; int sum = 0 ;
for ( int i = 0 ; i < str . size ( ) ; i ++ ) {
int ascii = str [ i ] ;
if ( ascii < 96 ascii > 122 ) return false ;
sum += ascii ;
if ( min > ascii ) min = ascii ;
if ( max < ascii ) max = ascii ; }
min -= 1 ;
int eSum = ( ( max * ( max + 1 ) ) / 2 ) - ( ( min * ( min + 1 ) ) / 2 ) ;
return sum == eSum ; }
int main ( ) {
string str = " dcef " ; if ( check ( str ) ) cout << ( " Yes " ) ; else cout << ( " No " ) ;
string str1 = " xyza " ; if ( check ( str1 ) ) cout << ( " Yes " else cout << ( " No " }
int findKth ( int arr [ ] , int n , int k ) { unordered_set < int > missing ; int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) missing . insert ( arr [ i ] ) ;
int maxm = * max_element ( arr , arr + n ) ; int minm = * min_element ( arr , arr + n ) ;
for ( int i = minm + 1 ; i < maxm ; i ++ ) {
if ( missing . find ( i ) == missing . end ( ) ) count ++ ;
if ( count == k ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { 2 , 10 , 9 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 5 ; cout << findKth ( arr , n , k ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * next ; } ;
bool sortList ( struct Node * head ) { int startVal = 1 ; while ( head != NULL ) { head -> data = startVal ; startVal ++ ; head = head -> next ; } }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ;
new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( struct Node * node ) { while ( node != NULL ) { cout << node -> data << " ▁ " ; node = node -> next ; } }
int main ( ) { struct Node * start = NULL ;
push ( & start , 2 ) ; push ( & start , 1 ) ; push ( & start , 6 ) ; push ( & start , 4 ) ; push ( & start , 5 ) ; push ( & start , 3 ) ; sortList ( start ) ; printList ( start ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
bool isSortedDesc ( struct Node * head ) {
if ( head == NULL head -> next == NULL ) return true ;
return ( head -> data > head -> next -> data && isSortedDesc ( head -> next ) ) ; } Node * newNode ( int data ) { Node * temp = new Node ; temp -> next = NULL ; temp -> data = data ; }
int main ( ) { struct Node * head = newNode ( 7 ) ; head -> next = newNode ( 5 ) ; head -> next -> next = newNode ( 4 ) ; head -> next -> next -> next = newNode ( 3 ) ; isSortedDesc ( head ) ? cout << " Yes " : cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int minSum ( int arr [ ] , int n ) {
vector < int > evenArr ; vector < int > oddArr ;
sort ( arr , arr + n ) ;
for ( int i = 0 ; i < n ; i ++ ) { if ( i < n / 2 ) oddArr . push_back ( arr [ i ] ) ; else evenArr . push_back ( arr [ i ] ) ; }
sort ( evenArr . begin ( ) , evenArr . end ( ) , greater < int > ( ) ) ;
int i = 0 , sum = 0 ; for ( int j = 0 ; j < evenArr . size ( ) ; j ++ ) { arr [ i ++ ] = evenArr [ j ] ; arr [ i ++ ] = oddArr [ j ] ; sum += evenArr [ j ] * oddArr [ j ] ; } return sum ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 6 , 7 , 3 , 4 , 2 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Minimum ▁ required ▁ sum ▁ = ▁ " << minSum ( arr , n ) ; cout << " Sorted array in required format : " ; for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
void minTime ( string word ) { int ans = 0 ;
int curr = 0 ; for ( int i = 0 ; i < word . length ( ) ; i ++ ) {
int k = word [ i ] - ' a ' ;
int a = abs ( curr - k ) ;
int b = 26 - abs ( curr - k ) ;
ans += min ( a , b ) ;
ans ++ ; curr = word [ i ] - ' a ' ; }
cout << ans ; }
int main ( ) {
string str = " zjpc " ;
minTime ( str ) ; return 0 ; }
int reduceToOne ( long long int N ) {
int cnt = 0 ; while ( N != 1 ) {
if ( N == 2 or ( N % 2 == 1 ) ) {
N = N - 1 ;
cnt ++ ; }
else if ( N % 2 == 0 ) {
N = N / ( N / 2 ) ;
cnt ++ ; } }
return cnt ; }
int main ( ) { long long int N = 35 ; cout << reduceToOne ( N ) ; return 0 ; }
void maxDiamonds ( int A [ ] , int N , int K ) {
priority_queue < int > pq ;
for ( int i = 0 ; i < N ; i ++ ) { pq . push ( A [ i ] ) ; }
int ans = 0 ;
while ( ! pq . empty ( ) && K -- ) {
int top = pq . top ( ) ;
pq . pop ( ) ;
ans += top ;
top = top / 2 ; pq . push ( top ) ; }
cout << ans ; }
int main ( ) { int A [ ] = { 2 , 1 , 7 , 4 , 2 } ; int K = 3 ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; maxDiamonds ( A , N , K ) ; return 0 ; }
int MinimumCost ( int A [ ] , int B [ ] , int N ) {
int totalCost = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int mod_A = B [ i ] % A [ i ] ; int totalCost_A = min ( mod_A , A [ i ] - mod_A ) ;
int mod_B = A [ i ] % B [ i ] ; int totalCost_B = min ( mod_B , B [ i ] - mod_B ) ;
totalCost += min ( totalCost_A , totalCost_B ) ; }
return totalCost ; }
int main ( ) { int A [ ] = { 3 , 6 , 3 } ; int B [ ] = { 4 , 8 , 13 } ; int N = sizeof ( A ) / sizeof ( A [ 0 ] ) ; cout << MinimumCost ( A , B , N ) ; return 0 ; }
void printLargestDivisible ( int arr [ ] , int N ) { int i , count0 = 0 , count7 = 0 ; for ( i = 0 ; i < N ; i ++ ) {
if ( arr [ i ] == 0 ) count0 ++ ; else count7 ++ ; }
if ( count7 % 50 == 0 ) { while ( count7 -- ) cout << 7 ; while ( count0 -- ) cout << 0 ; }
else if ( count7 < 5 ) { if ( count0 == 0 ) cout << " No " ; else cout << "0" ; }
else {
count7 = count7 - count7 % 5 ; while ( count7 -- ) cout << 7 ; while ( count0 -- ) cout << 0 ; } }
int main ( ) {
int arr [ ] = { 0 , 7 , 0 , 7 , 7 , 7 , 7 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 7 , 7 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printLargestDivisible ( arr , N ) ; return 0 ; }
int findMaxValByRearrArr ( int arr [ ] , int N ) {
sort ( arr , arr + N ) ;
int res = 0 ;
do {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
sum += __gcd ( i + 1 , arr [ i ] ) ; }
res = max ( res , sum ) ; } while ( next_permutation ( arr , arr + N ) ) ; return res ; }
int main ( ) { int arr [ ] = { 3 , 2 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMaxValByRearrArr ( arr , N ) ; return 0 ; }
int min_elements ( int arr [ ] , int N ) {
unordered_map < int , int > mp ;
for ( int i = 0 ; i < N ; i ++ ) {
mp [ arr [ i ] ] ++ ; }
int cntMinRem = 0 ;
for ( auto it : mp ) {
int i = it . first ;
if ( mp [ i ] < i ) {
cntMinRem += mp [ i ] ; }
else if ( mp [ i ] > i ) {
cntMinRem += ( mp [ i ] - i ) ; } } return cntMinRem ; }
int main ( ) { int arr [ ] = { 2 , 4 , 1 , 4 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << min_elements ( arr , N ) ; return 0 ; }
bool CheckAllarrayEqual ( int arr [ ] , int N ) {
if ( N == 1 ) { return true ; }
int totalSum = arr [ 0 ] ;
int secMax = INT_MIN ;
int Max = arr [ 0 ] ;
for ( int i = 1 ; i < N ; i ++ ) { if ( arr [ i ] >= Max ) {
secMax = Max ;
Max = arr [ i ] ; } else if ( arr [ i ] > secMax ) {
secMax = arr [ i ] ; }
totalSum += arr [ i ] ; }
if ( ( secMax * ( N - 1 ) ) > totalSum ) { return false ; }
if ( totalSum % ( N - 1 ) ) { return false ; } return true ; }
int main ( ) { int arr [ ] = { 6 , 2 , 2 , 2 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; if ( CheckAllarrayEqual ( arr , N ) ) { cout << " YES " ; } else { cout << " NO " ; } }
void Remove_one_element ( int arr [ ] , int n ) {
int post_odd = 0 , post_even = 0 ;
int curr_odd = 0 , curr_even = 0 ;
int res = 0 ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
if ( i % 2 ) post_odd ^= arr [ i ] ;
else post_even ^= arr [ i ] ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( i % 2 ) post_odd ^= arr [ i ] ;
else post_even ^= arr [ i ] ;
int X = curr_odd ^ post_even ;
int Y = curr_even ^ post_odd ;
if ( X == Y ) res ++ ;
if ( i % 2 ) curr_odd ^= arr [ i ] ;
else curr_even ^= arr [ i ] ; }
cout << res << endl ; }
int main ( ) {
int arr [ ] = { 1 , 0 , 1 , 0 , 1 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
Remove_one_element ( arr , N ) ; return 0 ; }
int cntIndexesToMakeBalance ( int arr [ ] , int n ) {
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
if ( i % 2 ) {
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
int main ( ) { int arr [ ] = { 1 , 1 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << cntIndexesToMakeBalance ( arr , n ) ; return 0 ; }
void findNums ( int X , int Y ) {
int A , B ;
if ( X < Y ) { A = -1 ; B = -1 ; }
else if ( abs ( X - Y ) & 1 ) { A = -1 ; B = -1 ; }
else if ( X == Y ) { A = 0 ; B = Y ; }
else {
A = ( X - Y ) / 2 ;
if ( ( A & Y ) == 0 ) {
B = ( A + Y ) ; }
else { A = -1 ; B = -1 ; } }
cout << A << " ▁ " << B ; }
int main ( ) {
int X = 17 , Y = 13 ;
findNums ( X , Y ) ; return 0 ; }
void checkCount ( int A [ ] , int Q [ ] [ 2 ] , int q ) {
for ( int i = 0 ; i < q ; i ++ ) { int L = Q [ i ] [ 0 ] ; int R = Q [ i ] [ 1 ] ;
L -- , R -- ;
if ( ( A [ L ] < A [ L + 1 ] ) != ( A [ R - 1 ] < A [ R ] ) ) { cout << " Yes STRNEWLINE " ; } else { cout << " No STRNEWLINE " ; } } }
int main ( ) { int arr [ ] = { 11 , 13 , 12 , 14 } ; int Q [ ] [ 2 ] = { { 1 , 4 } , { 2 , 4 } } ; int q = sizeof ( Q ) / sizeof ( Q [ 0 ] ) ; checkCount ( arr , Q , q ) ; return 0 ; }
float pairProductMean ( int arr [ ] , int N ) {
vector < int > pairArray ;
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) { int pairProduct = arr [ i ] * arr [ j ] ;
pairArray . push_back ( pairProduct ) ; } }
int length = pairArray . size ( ) ;
float sum = 0 ; for ( int i = 0 ; i < length ; i ++ ) sum += pairArray [ i ] ;
float mean ;
if ( length != 0 ) mean = sum / length ; else mean = 0 ;
return mean ; }
int main ( ) {
int arr [ ] = { 1 , 2 , 4 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << fixed << setprecision ( 2 ) << pairProductMean ( arr , N ) ; return 0 ; }
void findPlayer ( string str [ ] , int n ) {
int move_first = 0 ;
int move_sec = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( str [ i ] [ 0 ] == str [ i ] [ str [ i ] . length ( ) - 1 ] ) {
if ( str [ i ] [ 0 ] == 48 ) move_first ++ ; else move_sec ++ ; } }
if ( move_first <= move_sec ) { cout << " Player ▁ 2 ▁ wins " ; } else { cout << " Player ▁ 1 ▁ wins " ; } }
string str [ ] = { "010" , "101" } ; int N = sizeof ( str ) / sizeof ( str [ 0 ] ) ;
findPlayer ( str , N ) ; return 0 ; }
int find_next ( int n , int k ) {
int M = n + 1 ; while ( 1 ) {
if ( M & ( 1ll << k ) ) break ;
M ++ ; }
return M ; }
int main ( ) {
int N = 15 , K = 2 ;
cout << find_next ( N , K ) ; return 0 ; }
int find_next ( int n , int k ) {
int ans = 0 ;
if ( ( n & ( 1ll << k ) ) == 0 ) { int cur = 0 ;
for ( int i = 0 ; i < k ; i ++ ) {
if ( n & ( 1ll << i ) ) cur += 1ll << i ; }
ans = n - cur + ( 1ll << k ) ; }
else { int first_unset_bit = -1 , cur = 0 ; for ( int i = 0 ; i < 64 ; i ++ ) {
if ( ( n & ( 1ll << i ) ) == 0 ) { first_unset_bit = i ; break ; }
else cur += ( 1ll << i ) ; }
ans = n - cur + ( 1ll << first_unset_bit ) ;
if ( ( ans & ( 1ll << k ) ) == 0 ) ans += ( 1ll << k ) ; }
return ans ; }
int main ( ) { int N = 15 , K = 2 ;
cout << find_next ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; string largestString ( string num , int k ) {
string ans = " " ; for ( auto i : num ) {
while ( ans . length ( ) && ans . back ( ) < i && k > 0 ) {
ans . pop_back ( ) ;
k -- ; }
ans . push_back ( i ) ; }
while ( ans . length ( ) and k -- ) { ans . pop_back ( ) ; }
return ans ; }
int main ( ) { string str = " zyxedcba " ; int k = 1 ; cout << largestString ( str , k ) << endl ; }
void maxLengthSubArray ( int A [ ] , int N ) {
int forward [ N ] , backward [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) { if ( i == 0 A [ i ] != A [ i - 1 ] ) { forward [ i ] = 1 ; } else forward [ i ] = forward [ i - 1 ] + 1 ; }
for ( int i = N - 1 ; i >= 0 ; i -- ) { if ( i == N - 1 A [ i ] != A [ i + 1 ] ) { backward [ i ] = 1 ; } else backward [ i ] = backward [ i + 1 ] + 1 ; }
int ans = 0 ;
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( A [ i ] != A [ i + 1 ] ) ans = max ( ans , min ( forward [ i ] , backward [ i + 1 ] ) * 2 ) ; }
cout << ans ; }
int arr [ ] = { 1 , 2 , 3 , 4 , 4 , 4 , 6 , 6 , 6 , 9 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
maxLengthSubArray ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void minNum ( int n ) { if ( n < 3 ) cout << -1 ; else cout << ( 210 * ( ( int ) ( pow ( 10 , n - 1 ) / 210 ) + 1 ) ) ; }
int main ( ) { int n = 5 ; minNum ( n ) ; return 0 ; }
string helper ( int d , int s ) {
string ans ( d , '0' ) ; for ( int i = d - 1 ; i >= 0 ; i -- ) {
if ( s >= 9 ) { ans [ i ] = '9' ; s -= 9 ; }
else { char c = ( char ) s + '0' ; ans [ i ] = c ; s = 0 ; } } return ans ; }
string findMin ( int x , int Y ) {
string y = to_string ( Y ) ; int n = y . size ( ) ; vector < int > p ( n ) ;
for ( int i = 0 ; i < n ; i ++ ) { p [ i ] = y [ i ] - '0' ; if ( i > 0 ) p [ i ] += p [ i - 1 ] ; }
for ( int i = n - 1 , k = 0 ; ; i -- , k ++ ) {
int d = 0 ; if ( i >= 0 ) d = y [ i ] - '0' ;
for ( int j = d + 1 ; j <= 9 ; j ++ ) {
int r = ( i > 0 ) * p [ i - 1 ] + j ;
if ( x - r >= 0 and x - r <= 9 * k ) {
string suf = helper ( k , x - r ) ; string pre = " " ; if ( i > 0 ) pre = y . substr ( 0 , i ) ;
char cur = ( char ) j + '0' ; pre += cur ;
return pre + suf ; } } } }
int main ( ) {
int x = 18 ; int y = 99 ;
cout << findMin ( x , y ) << endl ; return 0 ; }
void largestNumber ( int n , int X , int Y ) { int maxm = max ( X , Y ) ;
Y = X + Y - maxm ;
X = maxm ;
int Xs = 0 ; int Ys = 0 ; while ( n > 0 ) {
if ( n % Y == 0 ) {
Xs += n ;
n = 0 ; } else {
n -= X ;
Ys += X ; } }
if ( n == 0 ) { while ( Xs -- > 0 ) cout << X ; while ( Ys -- > 0 ) cout << Y ; }
else cout < < " - 1" ; }
int main ( ) { int n = 19 , X = 7 , Y = 5 ; largestNumber ( n , X , Y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int minChanges ( string str , int N ) { int res ; int count0 = 0 , count1 = 0 ;
for ( char x : str ) { count0 += ( x == '0' ) ; } res = count0 ;
for ( char x : str ) { count0 -= ( x == '0' ) ; count1 += ( x == '1' ) ; res = min ( res , count1 + count0 ) ; } return res ; }
int main ( ) { int N = 9 ; string str = "000101001" ; cout << minChanges ( str , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int missingnumber ( int n , int arr [ ] ) { int mn = INT_MAX , mx = INT_MIN ;
for ( int i = 0 ; i < n ; i ++ ) { if ( i > 0 && arr [ i ] == -1 && arr [ i - 1 ] != -1 ) { mn = min ( mn , arr [ i - 1 ] ) ; mx = max ( mx , arr [ i - 1 ] ) ; } if ( i < ( n - 1 ) && arr [ i ] == -1 && arr [ i + 1 ] != -1 ) { mn = min ( mn , arr [ i + 1 ] ) ; mx = max ( mx , arr [ i + 1 ] ) ; } } long long int res = ( mx + mn ) / 2 ; return res ; }
int main ( ) { int n = 5 ; int arr [ 5 ] = { -1 , 10 , -1 , 12 , -1 } ; int ans = 0 ;
int res = missingnumber ( n , arr ) ; cout << res ; return 0 ; }
int LCSubStr ( char * A , char * B , int m , int n ) {
int LCSuff [ m + 1 ] [ n + 1 ] ; int result = 0 ;
for ( int i = 0 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) {
if ( i == 0 j == 0 ) LCSuff [ i ] [ j ] = 0 ;
else if ( A [ i - 1 ] == B [ j - 1 ] ) { LCSuff [ i ] [ j ] = LCSuff [ i - 1 ] [ j - 1 ] + 1 ; result = max ( result , LCSuff [ i ] [ j ] ) ; }
else LCSuff [ i ] [ j ] = 0 ; } }
return result ; }
int main ( ) { char A [ ] = "0110" ; char B [ ] = "1101" ; int M = strlen ( A ) ; int N = strlen ( B ) ;
cout << LCSubStr ( A , B , M , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define maxN  20 NEW_LINE #define maxSum  50 NEW_LINE #define minSum  50 NEW_LINE #define base  50
int dp [ maxN ] [ maxSum + minSum ] ; bool v [ maxN ] [ maxSum + minSum ] ;
int findCnt ( int * arr , int i , int required_sum , int n ) {
if ( i == n ) { if ( required_sum == 0 ) return 1 ; else return 0 ; }
if ( v [ i ] [ required_sum + base ] ) return dp [ i ] [ required_sum + base ] ;
v [ i ] [ required_sum + base ] = 1 ;
dp [ i ] [ required_sum + base ] = findCnt ( arr , i + 1 , required_sum , n ) + findCnt ( arr , i + 1 , required_sum - arr [ i ] , n ) ; return dp [ i ] [ required_sum + base ] ; }
void countSubsets ( int * arr , int K , int n ) {
int sum = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
sum += arr [ i ] ; }
int S1 = ( sum + K ) / 2 ;
cout << findCnt ( arr , 0 , S1 , n ) ; }
int main ( ) { int arr [ ] = { 1 , 1 , 2 , 3 } ; int N = sizeof ( arr ) / sizeof ( int ) ; int K = 1 ;
countSubsets ( arr , K , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; float dp [ 105 ] [ 605 ] ;
float find ( int N , int sum ) { if ( dp [ N ] [ sum ] ) return dp [ N ] [ sum ] ;
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return 1.0 / 6 ; else return 0 ; } for ( int i = 1 ; i <= 6 ; i ++ ) dp [ N ] [ sum ] = dp [ N ] [ sum ] + find ( N - 1 , sum - i ) / 6 ; return dp [ N ] [ sum ] ; }
int main ( ) { int N = 4 , a = 13 , b = 17 ; float probability = 0.0 ;
for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
cout << fixed << setprecision ( 6 ) << probability ; return 0 ; }
int count ( int n ) {
map < int , int > dp ;
dp [ 0 ] = 0 ; dp [ 1 ] = 1 ;
if ( ( dp . find ( n ) == dp . end ( ) ) ) dp [ n ] = 1 + min ( n % 2 + count ( n / 2 ) , n % 3 + count ( n / 3 ) ) ;
return dp [ n ] ; }
int N = 6 ;
cout << count ( N ) ; }
int find_minimum_operations ( int n , int b [ ] , int k ) {
int d [ n + 1 ] = { 0 } ;
int operations = 0 , need ; for ( int i = 0 ; i < n ; i ++ ) {
if ( i > 0 ) { d [ i ] += d [ i - 1 ] ; }
if ( b [ i ] > d [ i ] ) {
operations += b [ i ] - d [ i ] ; need = b [ i ] - d [ i ] ;
d [ i ] += need ;
if ( i + k <= n ) { d [ i + k ] -= need ; } } } cout << operations << endl ; }
int main ( ) { int n = 5 ; int b [ ] = { 1 , 2 , 3 , 4 , 5 } ; int k = 2 ;
find_minimum_operations ( n , b , k ) ; return 0 ; }
int ways ( vector < vector < int > > & arr , int K ) { int R = arr . size ( ) ; int C = arr [ 0 ] . size ( ) ; int preSum [ R ] [ C ] ;
for ( int r = R - 1 ; r >= 0 ; r -- ) { for ( int c = C - 1 ; c >= 0 ; c -- ) { preSum [ r ] = arr [ r ] ; if ( r + 1 < R ) preSum [ r ] += preSum [ r + 1 ] ; if ( c + 1 < C ) preSum [ r ] += preSum [ r ] ; if ( r + 1 < R && c + 1 < C ) preSum [ r ] -= preSum [ r + 1 ] ; } }
int dp [ K + 1 ] [ R ] [ C ] ;
for ( int k = 1 ; k <= K ; k ++ ) { for ( int r = R - 1 ; r >= 0 ; r -- ) { for ( int c = C - 1 ; c >= 0 ; c -- ) { if ( k == 1 ) { dp [ k ] [ r ] = ( preSum [ r ] > 0 ) ? 1 : 0 ; } else { dp [ k ] [ r ] = 0 ; for ( int r1 = r + 1 ; r1 < R ; r1 ++ ) {
if ( preSum [ r ] - preSum [ r1 ] > 0 ) dp [ k ] [ r ] += dp [ k - 1 ] [ r1 ] ; } for ( int c1 = c + 1 ; c1 < C ; c1 ++ ) {
if ( preSum [ r ] - preSum [ r ] [ c1 ] > 0 ) dp [ k ] [ r ] += dp [ k - 1 ] [ r ] [ c1 ] ; } } } } } return dp [ K ] [ 0 ] [ 0 ] ; }
int main ( ) { vector < vector < int > > arr = { { 1 , 0 , 0 } , { 1 , 1 , 1 } , { 0 , 0 , 0 } } ; int k = 3 ;
cout << ways ( arr , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int p = 1000000007 ;
long long int power ( long long int x , long long int y , long long int p ) { long long int res = 1 ; x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
y = y >> 1 ; x = ( x * x ) % p ; } return res ; }
void nCr ( long long int n , long long int p , int f [ ] [ 100 ] , int m ) { for ( long long int i = 0 ; i <= n ; i ++ ) { for ( long long int j = 0 ; j <= m ; j ++ ) {
if ( j > i ) { f [ i ] [ j ] = 0 ; }
else if ( j == 0 j == i ) { f [ i ] [ j ] = 1 ; } else { f [ i ] [ j ] = ( f [ i - 1 ] [ j ] + f [ i - 1 ] [ j - 1 ] ) % p ; } } } }
void ProductOfSubsets ( int arr [ ] , int n , int m ) { int f [ n + 1 ] [ 100 ] ; nCr ( n , p - 1 , f , m ) ; sort ( arr , arr + n ) ;
long long int ans = 1 ; for ( long long int i = 0 ; i < n ; i ++ ) {
long long int x = 0 ; for ( long long int j = 1 ; j <= m ; j ++ ) {
if ( m % j == 0 ) {
x = ( x + ( f [ n - i - 1 ] [ m - j ] * f [ i ] [ j - 1 ] ) % ( p - 1 ) ) % ( p - 1 ) ; } } ans = ( ( ans * power ( arr [ i ] , x , p ) ) % p ) ; } cout << ans << endl ; }
int main ( ) { int arr [ ] = { 4 , 5 , 7 , 9 , 3 } ; int K = 4 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; ProductOfSubsets ( arr , N , K ) ; return 0 ; }
int countWays ( int n , int m ) {
int dp [ m + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { dp [ 1 ] [ i ] = 1 ; }
int sum ; for ( int i = 2 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) { sum = 0 ;
for ( int k = 0 ; k <= j ; k ++ ) { sum += dp [ i - 1 ] [ k ] ; }
dp [ i ] [ j ] = sum ; } }
return dp [ m ] [ n ] ; }
int main ( ) { int N = 2 , K = 3 ;
cout << countWays ( N , K ) ; return 0 ; }
int countWays ( int n , int m ) {
int dp [ m + 1 ] [ n + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) { dp [ 1 ] [ i ] = 1 ; if ( i != 0 ) { dp [ 1 ] [ i ] += dp [ 1 ] [ i - 1 ] ; } }
for ( int i = 2 ; i <= m ; i ++ ) { for ( int j = 0 ; j <= n ; j ++ ) {
if ( j == 0 ) { dp [ i ] [ j ] = dp [ i - 1 ] [ j ] ; }
else { dp [ i ] [ j ] = dp [ i - 1 ] [ j ] ;
if ( i == m && j == n ) { return dp [ i ] [ j ] ; }
dp [ i ] [ j ] += dp [ i ] [ j - 1 ] ; } } } }
int main ( ) { int N = 2 , K = 3 ;
cout << countWays ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void SieveOfEratosthenes ( int MAX , vector < int > & primes ) { bool prime [ MAX + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ;
for ( long long p = 2 ; p * p <= MAX ; p ++ ) { if ( prime [ p ] == true ) {
for ( long long i = p * p ; i <= MAX ; i += p ) prime [ i ] = false ; } }
for ( long long i = 2 ; i <= MAX ; i ++ ) { if ( prime [ i ] ) primes . push_back ( i ) ; } }
int findLongest ( int A [ ] , int n ) {
unordered_map < int , int > mpp ; vector < int > primes ;
SieveOfEratosthenes ( A [ n - 1 ] , primes ) ; int dp [ n ] ; memset ( dp , 0 , sizeof dp ) ;
dp [ n - 1 ] = 1 ; mpp [ A [ n - 1 ] ] = n - 1 ;
for ( int i = n - 2 ; i >= 0 ; i -- ) {
int num = A [ i ] ;
dp [ i ] = 1 ; int maxi = 0 ;
for ( auto it : primes ) {
int xx = num * it ;
if ( xx > A [ n - 1 ] ) break ;
else if ( mpp [ xx ] != 0 ) {
dp [ i ] = max ( dp [ i ] , 1 + dp [ mpp [ xx ] ] ) ; } }
mpp [ A [ i ] ] = i ; } int ans = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { ans = max ( ans , dp [ i ] ) ; } return ans ; }
int main ( ) { int a [ ] = { 1 , 2 , 5 , 6 , 12 , 35 , 60 , 385 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << findLongest ( a , n ) ; }
int waysToKAdjacentSetBits ( int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( ! lastBit ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
int main ( ) { int n = 5 , k = 2 ;
int totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; cout << " Number ▁ of ▁ ways ▁ = ▁ " << totalWays << " STRNEWLINE " ; return 0 ; }
void postfix ( int a [ ] , int n ) { for ( int i = n - 1 ; i > 0 ; i -- ) a [ i - 1 ] = a [ i - 1 ] + a [ i ] ; }
void modify ( int a [ ] , int n ) { for ( int i = 1 ; i < n ; i ++ ) a [ i - 1 ] = i * a [ i ] ; }
void allCombination ( int a [ ] , int n ) { int sum = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) sum += i ; cout << " f ( 1 ) ▁ - - > ▁ " << sum << " STRNEWLINE " ;
for ( int i = 1 ; i < n ; i ++ ) {
postfix ( a , n - i + 1 ) ;
sum = 0 ; for ( int j = 1 ; j <= n - i ; j ++ ) { sum += ( j * a [ j ] ) ; } cout << " f ( " << i + 1 << " ) ▁ - - > ▁ " << sum << " STRNEWLINE " ;
modify ( a , n ) ; } }
int main ( ) { int n = 5 ; int * a = new int [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) a [ i ] = i + 1 ;
allCombination ( a , n ) ; return 0 ; }
public : int findStep ( int n ) { if ( n == 1 n == 0 ) return 1 ; else if ( n == 2 ) return 2 ; else return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) ; } } ;
int main ( ) { GFG g ; int n = 4 ; cout << g . findStep ( n ) ; return 0 ; }
bool isSubsetSum ( int arr [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 && sum != 0 ) return false ;
if ( arr [ n - 1 ] > sum ) return isSubsetSum ( arr , n - 1 , sum ) ;
return isSubsetSum ( arr , n - 1 , sum ) || isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) ; }
bool findPartiion ( int arr [ ] , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( sum % 2 != 0 ) return false ;
return isSubsetSum ( arr , n , sum / 2 ) ; }
int main ( ) { int arr [ ] = { 3 , 1 , 5 , 9 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) cout << " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ " " of ▁ equal ▁ sum " ; else cout << " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets " " ▁ of ▁ equal ▁ sum " ; return 0 ; }
bool findPartiion ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; bool part [ sum / 2 + 1 ] ;
for ( i = 0 ; i <= sum / 2 ; i ++ ) { part [ i ] = 0 ; }
for ( i = 0 ; i < n ; i ++ ) {
for ( j = sum / 2 ; j >= arr [ i ] ;
if ( part [ j - arr [ i ] ] == 1 j == arr [ i ] ) part [ j ] = 1 ; } } return part [ sum / 2 ] ; }
int main ( ) { int arr [ ] = { 1 , 3 , 3 , 2 , 3 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) cout << " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ " " sum " ; else cout << " Can ▁ not ▁ be ▁ divided ▁ into " << " ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ; return 0 ; }
int binomialCoeff ( int n , int r ) { if ( r > n ) return 0 ; long long int m = 1000000007 ; long long int inv [ r + 1 ] = { 0 } ; inv [ 0 ] = 1 ; if ( r + 1 >= 2 ) inv [ 1 ] = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { inv [ i ] = m - ( m / i ) * inv [ m % i ] % m ; } int ans = 1 ;
for ( int i = 2 ; i <= r ; i ++ ) { ans = ( ( ans % m ) * ( inv [ i ] % m ) ) % m ; }
for ( int i = n ; i >= ( n - r + 1 ) ; i -- ) { ans = ( ( ans % m ) * ( i % m ) ) % m ; } return ans ; }
int main ( ) { int n = 5 , r = 2 ; cout << " Value ▁ of ▁ C ( " << n << " , ▁ " << r << " ) ▁ is ▁ " << binomialCoeff ( n , r ) << endl ; return 0 ; }
int gcd ( int a , int b ) {
if ( a < b ) { int t = a ; a = b ; b = t ; } if ( a % b == 0 ) return b ;
return gcd ( b , a % b ) ; }
void printAnswer ( int x , int y ) {
int val = gcd ( x , y ) ;
if ( ( val & ( val - 1 ) ) == 0 ) cout << " Yes " ; else cout << " No " ; }
int main ( ) {
int x = 4 ; int y = 7 ;
printAnswer ( x , y ) ; return 0 ; }
int getElement ( int N , int r , int c ) {
if ( r > c ) return 0 ;
if ( r == 1 ) { return c ; }
int a = ( r + 1 ) * pow ( 2 , r - 2 ) ;
int d = pow ( 2 , r - 1 ) ;
c = c - r ; int element = a + d * c ; return element ; }
int main ( ) { int N = 4 , R = 3 , C = 4 ; cout << getElement ( N , R , C ) ; return 0 ; }
string MinValue ( string N , int X ) {
int len = N . size ( ) ;
int position = len + 1 ;
if ( N [ 0 ] == ' - ' ) {
for ( int i = len - 1 ; i >= 1 ; i -- ) { if ( ( N [ i ] - '0' ) < X ) { position = i ; } } } else {
for ( int i = len - 1 ; i >= 0 ; i -- ) { if ( ( N [ i ] - '0' ) > X ) { position = i ; } } }
N . insert ( N . begin ( ) + position , X + '0' ) ;
return N ; }
string N = "89" ; int X = 1 ;
cout << MinValue ( N , X ) << " STRNEWLINE " ; }
string divisibleByk ( string s , int n , int k ) {
int poweroftwo [ n ] ;
poweroftwo [ 0 ] = 1 % k ; for ( int i = 1 ; i < n ; i ++ ) {
poweroftwo [ i ] = ( poweroftwo [ i - 1 ] * ( 2 % k ) ) % k ; }
int rem = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ n - i - 1 ] == '1' ) {
rem += ( poweroftwo [ i ] ) ; rem %= k ; } }
if ( rem == 0 ) { return " Yes " ; }
else return " No " ; }
int main ( ) {
string s = "1010001" ; int k = 9 ;
int n = s . length ( ) ;
cout << divisibleByk ( s , n , k ) ; return 0 ; }
int maxSumbySplittingstring ( string str , int N ) {
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
res = max ( res , zero + cntOne - one ) ; } return res ; }
int main ( ) { string str = "00111" ; int N = str . length ( ) ; cout << maxSumbySplittingstring ( str , N ) ; return 0 ; }
void cntBalancedParenthesis ( string s , int N ) {
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
cntPairs ++ ; } } cout << cntPairs ; }
int main ( ) {
string s = " { ( } ) " ; int N = s . length ( ) ;
cntBalancedParenthesis ( s , N ) ; return 0 ; }
int arcIntersection ( string S , int len ) { stack < char > stk ;
for ( int i = 0 ; i < len ; i ++ ) {
stk . push ( S [ i ] ) ; if ( stk . size ( ) >= 2 ) {
char temp = stk . top ( ) ;
stk . pop ( ) ;
if ( stk . top ( ) == temp ) { stk . pop ( ) ; }
else { stk . push ( temp ) ; } } }
if ( stk . empty ( ) ) return 1 ; return 0 ; }
void countString ( string arr [ ] , int N ) {
int count = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int len = arr [ i ] . length ( ) ;
count += arcIntersection ( arr [ i ] , len ) ; }
cout << count << endl ; }
int main ( ) { string arr [ ] = { "0101" , "0011" , "0110" } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
countString ( arr , N ) ; return 0 ; }
string ConvertequivalentBase8 ( string S ) {
map < string , char > mp ;
mp [ "000" ] = '0' ; mp [ "001" ] = '1' ; mp [ "010" ] = '2' ; mp [ "011" ] = '3' ; mp [ "100" ] = '4' ; mp [ "101" ] = '5' ; mp [ "110" ] = '6' ; mp [ "111" ] = '7' ;
int N = S . length ( ) ; if ( N % 3 == 2 ) {
S = "0" + S ; } else if ( N % 3 == 1 ) {
S = "00" + S ; }
N = S . length ( ) ;
string oct ;
for ( int i = 0 ; i < N ; i += 3 ) {
string temp = S . substr ( i , 3 ) ;
oct . push_back ( mp [ temp ] ) ; } return oct ; }
string binString_div_9 ( string S , int N ) {
string oct ; oct = ConvertequivalentBase8 ( S ) ;
int oddSum = 0 ;
int evenSum = 0 ;
int M = oct . length ( ) ;
for ( int i = 0 ; i < M ; i += 2 ) {
oddSum += int ( oct [ i ] - '0' ) ; }
for ( int i = 1 ; i < M ; i += 2 ) {
evenSum += int ( oct [ i ] - '0' ) ; }
int Oct_9 = 11 ;
if ( abs ( oddSum - evenSum ) % Oct_9 == 0 ) { return " Yes " ; } return " No " ; }
int main ( ) { string S = "1010001" ; int N = S . length ( ) ; cout << binString_div_9 ( S , N ) ; }
int min_cost ( string S ) {
int cost = 0 ;
int F = 0 ;
int B = 0 ; int count = 0 ; for ( char c : S ) if ( c == ' ▁ ' ) count ++ ;
int n = S . size ( ) - count ;
if ( n == 1 ) return cost ;
for ( char in : S ) {
if ( in != ' ▁ ' ) {
if ( B != 0 ) {
cost += min ( n - F , F ) * B ; B = 0 ; }
F += 1 ; }
else {
B += 1 ; } }
return cost ; }
int main ( ) { string S = " ▁ @ $ " ; cout << min_cost ( S ) ; return 0 ; }
bool isVowel ( char ch ) { if ( ch == ' a ' or ch == ' e ' or ch == ' i ' or ch == ' o ' or ch == ' u ' ) return true ; else return false ; }
int minCost ( string S ) {
int cA = 0 ; int cE = 0 ; int cI = 0 ; int cO = 0 ; int cU = 0 ;
for ( int i = 0 ; i < S . size ( ) ; i ++ ) {
if ( isVowel ( S [ i ] ) ) {
cA += abs ( S [ i ] - ' a ' ) ; cE += abs ( S [ i ] - ' e ' ) ; cI += abs ( S [ i ] - ' i ' ) ; cO += abs ( S [ i ] - ' o ' ) ; cU += abs ( S [ i ] - ' u ' ) ; } }
return min ( min ( min ( min ( cA , cE ) , cI ) , cO ) , cU ) ; }
int main ( ) { string S = " geeksforgeeks " ; cout << minCost ( S ) << endl ; return 0 ; }
void decode_String ( string str , int K ) { string ans = " " ;
for ( int i = 0 ; i < str . size ( ) ; i += K )
ans += str [ i ] ;
for ( int i = str . size ( ) - ( K - 1 ) ; i < str . size ( ) ; i ++ ) ans += str [ i ] ; cout << ans << endl ; }
int main ( ) { int K = 3 ; string str = " abcbcscsesesesd " ; decode_String ( str , K ) ; }
string maxVowelSubString ( string str , int K ) {
int N = str . length ( ) ;
int pref [ N ] ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( str [ i ] == ' a ' or str [ i ] == ' e ' or str [ i ] == ' i ' or str [ i ] == ' o ' or str [ i ] == ' u ' ) pref [ i ] = 1 ;
else pref [ i ] = 0 ;
if ( i ) pref [ i ] += pref [ i - 1 ] ; }
int maxCount = pref [ K - 1 ] ;
string res = str . substr ( 0 , K ) ;
for ( int i = K ; i < N ; i ++ ) {
int currCount = pref [ i ] - pref [ i - K ] ;
if ( currCount > maxCount ) { maxCount = currCount ; res = str . substr ( i - K + 1 , K ) ; }
else if ( currCount == maxCount ) { string temp = str . substr ( i - K + 1 , K ) ; if ( temp < res ) res = temp ; } }
return res ; }
int main ( ) { string str = " ceebbaceeffo " ; int K = 3 ; cout << maxVowelSubString ( str , K ) ; return 0 ; }
void decodeStr ( string str , int len ) {
char c [ len ] = " " ; int med , pos = 1 , k ;
if ( len % 2 == 1 ) med = len / 2 ; else med = len / 2 - 1 ;
c [ med ] = str [ 0 ] ;
if ( len % 2 == 0 ) c [ med + 1 ] = str [ 1 ] ;
if ( len & 1 ) k = 1 ; else k = 2 ; for ( int i = k ; i < len ; i += 2 ) { c [ med - pos ] = str [ i ] ;
if ( len % 2 == 1 ) c [ med + pos ] = str [ i + 1 ] ;
else c [ med + pos + 1 ] = str [ i + 1 ] ; pos ++ ; }
for ( int i = 0 ; i < len ; i ++ ) cout << c [ i ] ; }
int main ( ) { string str = " ofrsgkeeeekgs " ; int len = str . length ( ) ; decodeStr ( str , len ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findCount ( string s , int L , int R ) {
int distinct = 0 ;
int frequency [ 26 ] = { } ;
for ( int i = L ; i <= R ; i ++ ) {
frequency [ s [ i ] - ' a ' ] ++ ; } for ( int i = 0 ; i < 26 ; i ++ ) {
if ( frequency [ i ] > 0 ) distinct ++ ; } cout << distinct << endl ; }
int main ( ) { string s = " geeksforgeeksisacomputerscienceportal " ; int queries = 3 ; int Q [ queries ] [ 2 ] = { { 0 , 10 } , { 15 , 18 } , { 12 , 20 } } ; for ( int i = 0 ; i < queries ; i ++ ) findCount ( s , Q [ i ] [ 0 ] , Q [ i ] [ 1 ] ) ; return 0 ; }
string ReverseComplement ( string s , int n , int k ) {
int rev = ( k + 1 ) / 2 ;
int complement = k - rev ;
if ( rev % 2 ) reverse ( s . begin ( ) , s . end ( ) ) ;
if ( complement % 2 ) { for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == '0' ) s [ i ] = '1' ; else s [ i ] = '0' ; } }
return s ; }
int main ( ) { string str = "10011" ; int k = 5 ; int n = str . size ( ) ;
cout << ReverseComplement ( str , n , k ) ; return 0 ; }
bool repeatingString ( string s , int n , int k ) {
if ( n % k != 0 ) { return false ; }
int frequency [ 123 ] ;
for ( int i = 0 ; i < 123 ; i ++ ) { frequency [ i ] = 0 ; }
for ( int i = 0 ; i < n ; i ++ ) { frequency [ s [ i ] ] ++ ; } int repeat = n / k ;
for ( int i = 0 ; i < 123 ; i ++ ) { if ( frequency [ i ] % repeat != 0 ) { return false ; } } return true ; }
int main ( ) { string s = " abcdcba " ; int n = s . size ( ) ; int k = 3 ; if ( repeatingString ( s , n , k ) ) { cout << " Yes " << endl ; } else { cout << " No " << endl ; } return 0 ; }
void findPhoneNumber ( int n ) { int temp = n ; int sum ;
while ( temp != 0 ) { sum += temp % 10 ; temp = temp / 10 ; }
if ( sum < 10 ) cout << n << "0" << sum ;
else cout < < n << sum ; }
int main ( ) { long int n = 98765432 ; findPhoneNumber ( n ) ; return 0 ; }
int cntSplits ( string s ) {
if ( s [ s . size ( ) - 1 ] == '1' ) return 0 ;
int c_zero = 0 ;
for ( int i = 0 ; i < s . size ( ) ; i ++ ) c_zero += ( s [ i ] == '0' ) ;
return ( int ) pow ( 2 , c_zero - 1 ) ; }
int main ( ) { string s = "10010" ; cout << cntSplits ( s ) ; return 0 ; }
void findNumbers ( string s ) { if ( s . empty ( ) ) return 0 ;
int n = s . size ( ) ;
int count = 1 ; int result = 0 ;
int left = 0 ; int right = 1 ; while ( right < n ) {
if ( s [ left ] == s [ right ] ) { count ++ ; }
else {
result += count * ( count + 1 ) / 2 ;
left = right ; count = 1 ; } right ++ ; }
result += count * ( count + 1 ) / 2 ; cout << result << endl ; }
int main ( ) { string s = " bbbcbb " ; findNumbers ( s ) ; }
bool isVowel ( char ch ) { ch = toupper ( ch ) ; return ( ch == ' A ' ch == ' E ' ch == ' I ' ch == ' O ' ch == ' U ' ) ; }
string duplicateVowels ( string str ) { int t = str . length ( ) ;
string res = " " ;
for ( int i = 0 ; i < t ; i ++ ) { if ( isVowel ( str [ i ] ) ) { res += str [ i ] ; } res += str [ i ] ; } return res ; }
int main ( ) { string str = " helloworld " ;
cout << " Original ▁ String : ▁ " << str << endl ; string res = duplicateVowels ( str ) ;
cout << " String ▁ with ▁ Vowels ▁ duplicated : ▁ " << res << endl ; }
int stringToInt ( string str ) {
if ( str . length ( ) == 1 ) return ( str [ 0 ] - '0' ) ;
double y = stringToInt ( str . substr ( 1 ) ) ;
double x = str [ 0 ] - '0' ;
x = x * pow ( 10 , str . length ( ) - 1 ) + y ; return int ( x ) ; }
int main ( ) { string str = "1235" ; cout << ( stringToInt ( str ) ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  26
int largestSubSeq ( string arr [ ] , int n ) {
int count [ MAX ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) { string str = arr [ i ] ;
bool hash [ MAX ] = { 0 } ; for ( int j = 0 ; j < str . length ( ) ; j ++ ) { hash [ str [ j ] - ' a ' ] = true ; } for ( int j = 0 ; j < MAX ; j ++ ) {
if ( hash [ j ] ) count [ j ] ++ ; } } return * ( max_element ( count , count + MAX ) ) ; }
int main ( ) { string arr [ ] = { " ab " , " bc " , " de " } ; int n = sizeof ( arr ) / sizeof ( string ) ; cout << largestSubSeq ( arr , n ) ; return 0 ; }
bool isPalindrome ( string str ) { int len = str . length ( ) ; for ( int i = 0 ; i < len / 2 ; i ++ ) { if ( str [ i ] != str [ len - 1 - i ] ) return false ; } return true ; }
bool createStringAndCheckPalindrome ( int N ) {
ostringstream out ; out << N ; string result = out . str ( ) ; string sub = " " + result , res_str = " " ; int sum = 0 ;
while ( N > 0 ) { int digit = N % 10 ; sum += digit ; N = N / 10 ; }
while ( res_str . length ( ) < sum ) res_str += sub ;
if ( res_str . length ( ) > sum ) res_str = res_str . substr ( 0 , sum ) ;
if ( isPalindrome ( res_str ) ) return true ; return false ; }
int main ( ) { int N = 10101 ; if ( createStringAndCheckPalindrome ( N ) ) cout << ( " Yes " ) ; else cout << ( " No " ) ; }
int minimumLength ( string s ) { int maxOcc = 0 , n = s . length ( ) ; int arr [ 26 ] = { 0 } ;
for ( int i = 0 ; i < n ; i ++ ) arr [ s [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < 26 ; i ++ ) if ( arr [ i ] > maxOcc ) maxOcc = arr [ i ] ;
return ( n - maxOcc ) ; }
int main ( ) { string str = " afddewqd " ; cout << minimumLength ( str ) ; return 0 ; }
void removeSpecialCharacter ( string s ) { for ( int i = 0 ; i < s . size ( ) ; i ++ ) {
if ( s [ i ] < ' A ' s [ i ] > ' Z ' && s [ i ] < ' a ' s [ i ] > ' z ' ) {
s . erase ( i , 1 ) ; i -- ; } } cout << s ; }
int main ( ) { string s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " ; removeSpecialCharacter ( s ) ; return 0 ; }
void removeSpecialCharacter ( string s ) { int j = 0 ; for ( int i = 0 ; i < s . size ( ) ; i ++ ) {
if ( ( s [ i ] >= ' A ' && s [ i ] <= ' Z ' ) || ( s [ i ] >= ' a ' && s [ i ] <= ' z ' ) ) { s [ j ] = s [ i ] ; j ++ ; } } cout << s . substr ( 0 , j ) ; }
int main ( ) { string s = " $ Gee * k ; s . . fo , ▁ r ' Ge ^ eks ? " ; removeSpecialCharacter ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ; int findRepeatFirstN2 ( char * s ) {
int p = -1 , i , j ; for ( i = 0 ; i < strlen ( s ) ; i ++ ) { for ( j = i + 1 ; j < strlen ( s ) ; j ++ ) { if ( s [ i ] == s [ j ] ) { p = i ; break ; } } if ( p != -1 ) break ; } return p ; }
int main ( ) { char str [ ] = " geeksforgeeks " ; int pos = findRepeatFirstN2 ( str ) ; if ( pos == -1 ) cout << " Not ▁ found " ; else cout << str [ pos ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void prCharWithFreq ( string s ) {
unordered_map < char , int > d ; for ( char i : s ) { d [ i ] ++ ; }
for ( char i : s ) {
if ( d [ i ] != 0 ) { cout << i << d [ i ] << " ▁ " ; d [ i ] = 0 ; } } }
int main ( ) { string s = " geeksforgeeks " ; prCharWithFreq ( s ) ; }
int possibleStrings ( int n , int r , int b , int g ) {
int fact [ n + 1 ] ; fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
int left = n - ( r + g + b ) ; int sum = 0 ;
for ( int i = 0 ; i <= left ; i ++ ) { for ( int j = 0 ; j <= left - i ; j ++ ) { int k = left - ( i + j ) ;
sum = sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ; } }
return sum ; }
int main ( ) { int n = 4 , r = 2 ; int b = 0 , g = 1 ; cout << possibleStrings ( n , r , b , g ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int CHARS = 26 ;
int remAnagram ( string str1 , string str2 ) {
int count1 [ CHARS ] = { 0 } , count2 [ CHARS ] = { 0 } ;
for ( int i = 0 ; str1 [ i ] != ' \0' ; i ++ ) count1 [ str1 [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; str2 [ i ] != ' \0' ; i ++ ) count2 [ str2 [ i ] - ' a ' ] ++ ;
int result = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) result += abs ( count1 [ i ] - count2 [ i ] ) ; return result ; }
int main ( ) { string str1 = " bcadeh " , str2 = " hea " ; cout << remAnagram ( str1 , str2 ) ; return 0 ; }
const int CHARS = 26 ;
bool isValidString ( string str ) { int freq [ CHARS ] = { 0 } ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
int i , freq1 = 0 , count_freq1 = 0 ; for ( i = 0 ; i < CHARS ; i ++ ) { if ( freq [ i ] != 0 ) { freq1 = freq [ i ] ; count_freq1 = 1 ; break ; } }
int j , freq2 = 0 , count_freq2 = 0 ; for ( j = i + 1 ; j < CHARS ; j ++ ) { if ( freq [ j ] != 0 ) { if ( freq [ j ] == freq1 ) count_freq1 ++ ; else { count_freq2 = 1 ; freq2 = freq [ j ] ; break ; } } }
for ( int k = j + 1 ; k < CHARS ; k ++ ) { if ( freq [ k ] != 0 ) { if ( freq [ k ] == freq1 ) count_freq1 ++ ; if ( freq [ k ] == freq2 ) count_freq2 ++ ;
return false ; }
if ( count_freq1 > 1 && count_freq2 > 1 ) return false ; }
return true ; }
int main ( ) { char str [ ] = " abcbc " ; if ( isValidString ( str ) ) cout << " YES " << endl ; else cout << " NO " << endl ; return 0 ; }
bool checkForVariation ( string str ) { if ( str . empty ( ) || str . length ( ) != 0 ) { return true ; } map < char , int > mapp ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) { mapp [ str [ i ] ] ++ ; }
bool first = true , second = true ; int val1 = 0 , val2 = 0 ; int countOfVal1 = 0 , countOfVal2 = 0 ; map < char , int > :: iterator itr ; for ( itr = mapp . begin ( ) ; itr != mapp . end ( ) ; ++ itr ) { int i = itr -> first ;
if ( first ) { val1 = i ; first = false ; countOfVal1 ++ ; continue ; } if ( i == val1 ) { countOfVal1 ++ ; continue ; }
if ( second ) { val2 = i ; countOfVal2 ++ ; second = false ; continue ; } if ( i == val2 ) { countOfVal2 ++ ; continue ; } return false ; } if ( countOfVal1 > 1 && countOfVal2 > 1 ) { return false ; } else { return true ; } }
int main ( ) { if ( checkForVariation ( " abcbcvf " ) ) cout << " true " << endl ; else cout << " false " << endl ; return 0 ; }
int countCompletePairs ( string set1 [ ] , string set2 [ ] , int n , int m ) { int result = 0 ;
int con_s1 [ n ] , con_s2 [ m ] ;
for ( int i = 0 ; i < n ; i ++ ) {
con_s1 [ i ] = 0 ; for ( int j = 0 ; j < set1 [ i ] . length ( ) ; j ++ ) {
con_s1 [ i ] = con_s1 [ i ] | ( 1 << ( set1 [ i ] [ j ] - ' a ' ) ) ; } }
for ( int i = 0 ; i < m ; i ++ ) {
con_s2 [ i ] = 0 ; for ( int j = 0 ; j < set2 [ i ] . length ( ) ; j ++ ) {
con_s2 [ i ] = con_s2 [ i ] | ( 1 << ( set2 [ i ] [ j ] - ' a ' ) ) ; } }
long long complete = ( 1 << 26 ) - 1 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) {
if ( ( con_s1 [ i ] con_s2 [ j ] ) == complete ) result ++ ; } } return result ; }
int main ( ) { string set1 [ ] = { " abcdefgh " , " geeksforgeeks " , " lmnopqrst " , " abc " } ; string set2 [ ] = { " ijklmnopqrstuvwxyz " , " abcdefghijklmnopqrstuvwxyz " , " defghijklmnopqrstuvwxyz " } ; int n = sizeof ( set1 ) / sizeof ( set1 [ 0 ] ) ; int m = sizeof ( set2 ) / sizeof ( set2 [ 0 ] ) ; cout << countCompletePairs ( set1 , set2 , n , m ) ; return 0 ; }
string encodeString ( string str ) { unordered_map < char , int > map ; string res = " " ; int i = 0 ;
for ( char ch : str ) {
if ( map . find ( ch ) == map . end ( ) ) map [ ch ] = i ++ ;
res += to_string ( map [ ch ] ) ; } return res ; }
void findMatchedWords ( unordered_set < string > dict , string pattern ) {
int len = pattern . length ( ) ;
string hash = encodeString ( pattern ) ;
for ( string word : dict ) {
if ( word . length ( ) == len && encodeString ( word ) == hash ) cout << word << " ▁ " ; } }
int main ( ) { unordered_set < string > dict = { " abb " , " abc " , " xyz " , " xyy " } ; string pattern = " foo " ; findMatchedWords ( dict , pattern ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool check ( string pattern , string word ) { if ( pattern . length ( ) != word . length ( ) ) return false ; char ch [ 128 ] = { 0 } ; int len = word . length ( ) ; for ( int i = 0 ; i < len ; i ++ ) { if ( ch [ pattern [ i ] ] == 0 ) ch [ pattern [ i ] ] = word [ i ] ; else if ( ch [ pattern [ i ] ] != word [ i ] ) return false ; } return true ; }
void findMatchedWords ( unordered_set < string > dict , string pattern ) {
int len = pattern . length ( ) ;
for ( string word : dict ) { if ( check ( pattern , word ) ) cout << word << " ▁ " ; } }
int main ( ) { unordered_set < string > dict = { " abb " , " abc " , " xyz " , " xyy " } ; string pattern = " foo " ; findMatchedWords ( dict , pattern ) ; return 0 ; }
string RevString ( string s [ ] , int l ) {
if ( l % 2 == 0 ) {
int j = l / 2 ;
while ( j <= l - 1 ) { string temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } }
else {
int j = ( l / 2 ) + 1 ;
while ( j <= l - 1 ) { string temp ; temp = s [ l - j - 1 ] ; s [ l - j - 1 ] = s [ j ] ; s [ j ] = temp ; j += 1 ; } } string S = s [ 0 ] ;
for ( int i = 1 ; i < 9 ; i ++ ) { S = S + " ▁ " + s [ i ] ; } return S ; }
int main ( ) { string s = " getting ▁ good ▁ at ▁ coding ▁ " " needs ▁ a ▁ lot ▁ of ▁ practice " ; string words [ ] = { " getting " , " good " , " at " , " coding " , " needs " , " a " , " lot " , " of " , " practice " } ; cout << RevString ( words , 9 ) << endl ; return 0 ; }
void printPath ( vector < int > res , int nThNode , int kThNode ) {
if ( kThNode > nThNode ) return ;
res . push_back ( kThNode ) ;
for ( int i = 0 ; i < res . size ( ) ; i ++ ) cout << res [ i ] << " ▁ " ; cout << " STRNEWLINE " ;
printPath ( res , nThNode , kThNode * 2 ) ;
printPath ( res , nThNode , kThNode * 2 + 1 ) ; }
void printPathToCoverAllNodeUtil ( int nThNode ) {
vector < int > res ;
printPath ( res , nThNode , 1 ) ; }
int main ( ) {
int nThNode = 7 ;
printPathToCoverAllNodeUtil ( nThNode ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getMid ( int s , int e ) { return s + ( e - s ) / 2 ; }
bool isArmstrong ( int x ) { int n = to_string ( x ) . size ( ) ; int sum1 = 0 ; int temp = x ; while ( temp > 0 ) { int digit = temp % 10 ; sum1 += pow ( digit , n ) ; temp /= 10 ; } if ( sum1 == x ) return true ; return false ; }
int MaxUtil ( int * st , int ss , int se , int l , int r , int node ) {
if ( l <= ss && r >= se ) return st [ node ] ;
if ( se < l ss > r ) return -1 ;
int mid = getMid ( ss , se ) ; return max ( MaxUtil ( st , ss , mid , l , r , 2 * node + 1 ) , MaxUtil ( st , mid + 1 , se , l , r , 2 * node + 2 ) ) ; }
void updateValue ( int arr [ ] , int * st , int ss , int se , int index , int value , int node ) { if ( index < ss index > se ) { cout << " Invalid ▁ Input " << endl ; return ; } if ( ss == se ) {
arr [ index ] = value ; if ( isArmstrong ( value ) ) st [ node ] = value ; else st [ node ] = -1 ; } else { int mid = getMid ( ss , se ) ; if ( index >= ss && index <= mid ) updateValue ( arr , st , ss , mid , index , value , 2 * node + 1 ) ; else updateValue ( arr , st , mid + 1 , se , index , value , 2 * node + 2 ) ; st [ node ] = max ( st [ 2 * node + 1 ] , st [ 2 * node + 2 ] ) ; } return ; }
int getMax ( int * st , int n , int l , int r ) {
if ( l < 0 r > n - 1 l > r ) { printf ( " Invalid ▁ Input " ) ; return -1 ; } return MaxUtil ( st , 0 , n - 1 , l , r , 0 ) ; }
int constructSTUtil ( int arr [ ] , int ss , int se , int * st , int si ) {
if ( ss == se ) { if ( isArmstrong ( arr [ ss ] ) ) st [ si ] = arr [ ss ] ; else st [ si ] = -1 ; return st [ si ] ; }
int mid = getMid ( ss , se ) ; st [ si ] = max ( constructSTUtil ( arr , ss , mid , st , si * 2 + 1 ) , constructSTUtil ( arr , mid + 1 , se , st , si * 2 + 2 ) ) ; return st [ si ] ; }
int * constructST ( int arr [ ] , int n ) {
int x = ( int ) ( ceil ( log2 ( n ) ) ) ;
int max_size = 2 * ( int ) pow ( 2 , x ) - 1 ;
int * st = new int [ max_size ] ;
constructSTUtil ( arr , 0 , n - 1 , st , 0 ) ;
return st ; }
int main ( ) { int arr [ ] = { 192 , 113 , 535 , 7 , 19 , 111 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int * st = constructST ( arr , n ) ;
cout << " Maximum ▁ armstrong ▁ " << " number ▁ in ▁ given ▁ range ▁ = ▁ " << getMax ( st , n , 1 , 3 ) << endl ;
updateValue ( arr , st , 0 , n - 1 , 1 , 153 , 0 ) ;
cout << " Updated ▁ Maximum ▁ armstrong ▁ " << " number ▁ in ▁ given ▁ range ▁ = ▁ " << getMax ( st , n , 1 , 3 ) << endl ; return 0 ; }
void maxRegions ( int n ) { int num ; num = n * ( n + 1 ) / 2 + 1 ;
cout << num ; }
int main ( ) { int n = 10 ; maxRegions ( n ) ; return 0 ; }
void checkSolveable ( int n , int m ) {
if ( n == 1 or m == 1 ) cout << " YES " ;
else if ( m == 2 and n == 2 ) cout << " YES " ; else cout < < " NO " ; }
int main ( ) { int n = 1 , m = 3 ; checkSolveable ( n , m ) ; }
int GCD ( int a , int b ) {
if ( b == 0 ) return a ;
else return GCD ( b , a % b ) ; }
void check ( int x , int y ) {
if ( GCD ( x , y ) == 1 ) { cout << " Yes " ; } else { cout << " No " ; } }
int main ( ) {
int X = 2 , Y = 7 ;
check ( X , Y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define size  1000001
void seiveOfEratosthenes ( int * prime ) { prime [ 0 ] = 1 , prime [ 1 ] = 0 ; for ( int i = 2 ; i * i < 1000001 ; i ++ ) {
if ( prime [ i ] == 0 ) { for ( int j = i * i ; j < 1000001 ; j += i ) {
prime [ j ] = 1 ; } } } }
float probabiltyEuler ( int * prime , int L , int R , int M ) { int * arr = new int [ size ] { 0 } ; int * eulerTotient = new int [ size ] { 0 } ; int count = 0 ;
for ( int i = L ; i <= R ; i ++ ) {
eulerTotient [ i - L ] = i ; arr [ i - L ] = i ; } for ( int i = 2 ; i < 1000001 ; i ++ ) {
if ( prime [ i ] == 0 ) {
for ( int j = ( L / i ) * i ; j <= R ; j += i ) { if ( j - L >= 0 ) {
eulerTotient [ j - L ] = eulerTotient [ j - L ] / i * ( i - 1 ) ; while ( arr [ j - L ] % i == 0 ) { arr [ j - L ] /= i ; } } } } }
for ( int i = L ; i <= R ; i ++ ) { if ( arr [ i - L ] > 1 ) { eulerTotient [ i - L ] = ( eulerTotient [ i - L ] / arr [ i - L ] ) * ( arr [ i - L ] - 1 ) ; } } for ( int i = L ; i <= R ; i ++ ) {
if ( ( eulerTotient [ i - L ] % M ) == 0 ) { count ++ ; } }
return ( 1.0 * count / ( R + 1 - L ) ) ; }
int main ( ) { int * prime = new int [ size ] { 0 } ; seiveOfEratosthenes ( prime ) ; int L = 1 , R = 7 , M = 3 ; cout << probabiltyEuler ( prime , L , R , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findWinner ( int n , int k ) { int cnt = 0 ;
if ( n == 1 ) cout << " No " << endl ;
else if ( ( n & 1 ) or n == 2 ) cout << " Yes " << endl ; else { int tmp = n ; int val = 1 ;
while ( tmp > k and tmp % 2 == 0 ) { tmp /= 2 ; val *= 2 ; }
for ( int i = 3 ; i <= sqrt ( tmp ) ; i ++ ) { while ( tmp % i == 0 ) { cnt ++ ; tmp /= i ; } } if ( tmp > 1 ) cnt ++ ;
if ( val == n ) cout << " No " << endl ; else if ( n / tmp == 2 and cnt == 1 ) cout << " No " << endl ;
else cout < < " Yes " << endl ; } }
int main ( ) { long long n = 1 , k = 1 ; findWinner ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void pen_hex ( long long n ) { long long pn = 1 ; for ( long long int i = 1 ; ; i ++ ) {
pn = i * ( 3 * i - 1 ) / 2 ; if ( pn > n ) break ;
long double seqNum = ( 1 + sqrt ( 8 * pn + 1 ) ) / 4 ; if ( seqNum == long ( seqNum ) ) cout << pn << " , ▁ " ; } }
int main ( ) { long long int N = 1000000 ; pen_hex ( N ) ; return 0 ; }
bool isPal ( int a [ 3 ] [ 3 ] , int n , int m ) {
for ( int i = 0 ; i < n / 2 ; i ++ ) { for ( int j = 0 ; j < m - 1 ; j ++ ) { if ( a [ i ] [ j ] != a [ n - 1 - i ] [ m - 1 - j ] ) return false ; } } return true ; }
int main ( ) { int n = 3 , m = 3 ; int a [ 3 ] [ 3 ] = { { 1 , 2 , 3 } , { 4 , 5 , 4 } , { 3 , 2 , 1 } } ; if ( isPal ( a , n , m ) ) { cout << " YES " << endl ; } else { cout << " NO " << endl ; } }
int getSum ( int n ) { int sum = 0 ; while ( n != 0 ) { sum = sum + n % 10 ; n = n / 10 ; } return sum ; }
void smallestNumber ( int N ) { int i = 1 ; while ( 1 ) {
if ( getSum ( i ) == N ) { cout << i ; break ; } i ++ ; } }
int main ( ) { int N = 10 ; smallestNumber ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int reversDigits ( int num ) { int rev_num = 0 ; while ( num > 0 ) { rev_num = rev_num * 10 + num % 10 ; num = num / 10 ; } return rev_num ; }
bool isPerfectSquare ( long double x ) {
long double sr = sqrt ( x ) ;
return ( ( sr - floor ( sr ) ) == 0 ) ; }
bool isRare ( int N ) {
int reverseN = reversDigits ( N ) ;
if ( reverseN == N ) return false ; return isPerfectSquare ( N + reverseN ) && isPerfectSquare ( N - reverseN ) ; }
int main ( ) { int n = 65 ; if ( isRare ( n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
void calc_ans ( ll l , ll r ) { vector < ll > power2 , power3 ;
ll mul2 = 1 ; while ( mul2 <= r ) { power2 . push_back ( mul2 ) ; mul2 *= 2 ; }
ll mul3 = 1 ; while ( mul3 <= r ) { power3 . push_back ( mul3 ) ; mul3 *= 3 ; }
vector < ll > power23 ; for ( int x = 0 ; x < power2 . size ( ) ; x ++ ) { for ( int y = 0 ; y < power3 . size ( ) ; y ++ ) { ll mul = power2 [ x ] * power3 [ y ] ; if ( mul == 1 ) continue ;
if ( mul <= r ) power23 . push_back ( mul ) ; } }
ll ans = 0 ; for ( ll x : power23 ) { if ( x >= l && x <= r ) ans ++ ; }
cout << ans << endl ; }
int main ( ) { ll l = 1 , r = 10 ; calc_ans ( l , r ) ; return 0 ; }
int nCr ( int n , int r ) { if ( r > n ) return 0 ; return fact ( n ) / ( fact ( r ) * fact ( n - r ) ) ; }
int fact ( int n ) { int res = 1 ; for ( int i = 2 ; i <= n ; i ++ ) res = res * i ; return res ; }
int countSubsequences ( int arr [ ] , int n , int k ) { int countOdd = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] & 1 ) countOdd ++ ; } int ans = nCr ( n , k ) - nCr ( countOdd , k ) ; return ans ; }
int main ( ) { int arr [ ] = { 2 , 4 } ; int K = 1 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << countSubsequences ( arr , N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void first_digit ( int x , int y ) {
int length = log ( x ) / log ( y ) + 1 ;
int first_digit = x / pow ( y , length - 1 ) ; cout << first_digit ; }
int main ( ) { int X = 55 , Y = 3 ; first_digit ( X , Y ) ; return 0 ; }
void checkIfCurzonNumber ( int N ) { long int powerTerm , productTerm ;
powerTerm = pow ( 2 , N ) + 1 ;
productTerm = 2 * N + 1 ;
if ( powerTerm % productTerm == 0 ) cout << " Yes STRNEWLINE " ; else cout << " No STRNEWLINE " ; }
int main ( ) { long int N = 5 ; checkIfCurzonNumber ( N ) ; N = 10 ; checkIfCurzonNumber ( N ) ; return 0 ; }
int minCount ( int n ) {
int hasharr [ TEN ] = { 10 , 3 , 6 , 9 , 2 , 5 , 8 , 1 , 4 , 7 } ;
if ( n > 69 ) return hasharr [ n % TEN ] ; else {
if ( n >= hasharr [ n % TEN ] * 7 ) return ( hasharr [ n % TEN ] ) ; else return -1 ; } }
int main ( ) { int n = 38 ; cout << minCount ( n ) ; return 0 ; }
void modifiedBinaryPattern ( int n ) {
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) {
if ( j == 1 j == i ) cout << 1 ;
else cout < < 0 ; }
cout << endl ; } }
int main ( ) { int n = 7 ;
modifiedBinaryPattern ( n ) ; }
void findRealAndImag ( string s ) {
int l = s . length ( ) ;
int i ;
if ( s . find ( ' + ' ) < l ) { i = s . find ( ' + ' ) ; }
else { i = s . find ( ' - ' ) ; }
string real = s . substr ( 0 , i ) ;
string imaginary = s . substr ( i + 1 , l - i - 2 ) ; cout << " Real ▁ part : ▁ " << real << " STRNEWLINE " ; cout << " Imaginary ▁ part : ▁ " << imaginary << " STRNEWLINE " ; }
int main ( ) { string s = "3 + 4i " ; findRealAndImag ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int highestPower ( int n , int k ) { int i = 0 ; int a = pow ( n , i ) ;
while ( a <= k ) { i += 1 ; a = pow ( n , i ) ; } return i - 1 ; }
int b [ 50 ] = { 0 } ;
int PowerArray ( int n , int k ) { while ( k ) {
int t = highestPower ( n , k ) ;
if ( b [ t ] ) {
cout << -1 ; return 0 ; } else
b [ t ] = 1 ;
k -= pow ( n , t ) ; }
for ( int i = 0 ; i < 50 ; i ++ ) { if ( b [ i ] ) { cout << i << " , ▁ " ; } } }
int main ( ) { int N = 3 ; int K = 40 ; PowerArray ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  100005
void SieveOfEratosthenes ( vector < bool > & composite ) { for ( int i = 0 ; i < N ; i ++ ) composite [ i ] = false ; for ( int p = 2 ; p * p < N ; p ++ ) {
if ( ! composite [ p ] ) {
for ( int i = p * 2 ; i < N ; i += p ) composite [ i ] = true ; } } }
int sumOfElements ( int arr [ ] , int n ) { vector < bool > composite ( N ) ; SieveOfEratosthenes ( composite ) ;
unordered_map < int , int > m ; for ( int i = 0 ; i < n ; i ++ ) m [ arr [ i ] ] ++ ;
int sum = 0 ;
for ( auto it = m . begin ( ) ; it != m . end ( ) ; it ++ ) {
if ( composite [ it -> second ] ) { sum += ( it -> first ) ; } } return sum ; }
int main ( ) { int arr [ ] = { 1 , 2 , 1 , 1 , 1 , 3 , 3 , 2 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << sumOfElements ( arr , n ) ; return 0 ; }
void remove ( int arr [ ] , int n ) {
unordered_map < int , int > m ; for ( int i = 0 ; i < n ; i ++ ) { m [ arr [ i ] ] ++ ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( ( m [ arr [ i ] ] & 1 ) ) continue ; cout << arr [ i ] << " , ▁ " ; } }
int main ( ) { int arr [ ] = { 3 , 3 , 3 , 2 , 2 , 4 , 7 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
remove ( arr , n ) ; return 0 ; }
void getmax ( int arr [ ] , int n , int x ) {
int s = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { s = s + arr [ i ] ; }
cout << min ( s , x ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int x = 5 ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; getmax ( arr , arr_size , x ) ; return 0 ; }
void shortestLength ( int n , int x [ ] , int y [ ] ) { int answer = 0 ;
int i = 0 ; while ( n -- ) {
if ( x [ i ] + y [ i ] > answer ) answer = x [ i ] + y [ i ] ; i ++ ; }
cout << " Length ▁ - > ▁ " << answer << endl ; cout << " Path ▁ - > ▁ " << " ( ▁ 1 , ▁ " << answer << " ▁ ) " << " and ▁ ( ▁ " << answer << " , ▁ 1 ▁ ) " ; }
int main ( ) {
int n = 4 ;
int x [ n ] = { 1 , 4 , 2 , 1 } ; int y [ n ] = { 4 , 1 , 1 , 2 } ; shortestLength ( n , x , y ) ; return 0 ; }
void FindPoints ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 , int x4 , int y4 ) {
int x5 = max ( x1 , x3 ) ; int y5 = max ( y1 , y3 ) ;
int x6 = min ( x2 , x4 ) ; int y6 = min ( y2 , y4 ) ;
if ( x5 > x6 y5 > y6 ) { cout << " No ▁ intersection " ; return ; } cout << " ( " << x5 << " , ▁ " << y5 << " ) ▁ " ; cout << " ( " << x6 << " , ▁ " << y6 << " ) ▁ " ;
int x7 = x5 ; int y7 = y6 ; cout << " ( " << x7 << " , ▁ " << y7 << " ) ▁ " ;
int x8 = x6 ; int y8 = y5 ; cout << " ( " << x8 << " , ▁ " << y8 << " ) ▁ " ; }
int main ( ) {
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x3 = 2 , y3 = 3 , x4 = 7 , y4 = 9 ;
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Point { float x , y ; Point ( ) { x = y = 0 ; } Point ( float a , float b ) { x = a , y = b ; } } ;
void printCorners ( Point p , Point q , float l ) { Point a , b , c , d ;
if ( p . x == q . x ) { a . x = p . x - ( l / 2.0 ) ; a . y = p . y ; d . x = p . x + ( l / 2.0 ) ; d . y = p . y ; b . x = q . x - ( l / 2.0 ) ; b . y = q . y ; c . x = q . x + ( l / 2.0 ) ; c . y = q . y ; }
else if ( p . y == q . y ) { a . y = p . y - ( l / 2.0 ) ; a . x = p . x ; d . y = p . y + ( l / 2.0 ) ; d . x = p . x ; b . y = q . y - ( l / 2.0 ) ; b . x = q . x ; c . y = q . y + ( l / 2.0 ) ; c . x = q . x ; }
else {
float m = ( p . x - q . x ) / float ( q . y - p . y ) ;
float dx = ( l / sqrt ( 1 + ( m * m ) ) ) * 0.5 ; float dy = m * dx ; a . x = p . x - dx ; a . y = p . y - dy ; d . x = p . x + dx ; d . y = p . y + dy ; b . x = q . x - dx ; b . y = q . y - dy ; c . x = q . x + dx ; c . y = q . y + dy ; } cout << a . x << " , ▁ " << a . y << " ▁ n " << b . x << " , ▁ " << b . y << " n " ; << c . x << " , ▁ " << c . y << " ▁ n " << d . x << " , ▁ " << d . y << " nn " ; }
int main ( ) { Point p1 ( 1 , 0 ) , q1 ( 1 , 2 ) ; printCorners ( p1 , q1 , 2 ) ; Point p ( 1 , 1 ) , q ( -1 , -1 ) ; printCorners ( p , q , 2 * sqrt ( 2 ) ) ; return 0 ; }
int minimumCost ( int arr [ ] , int N , int X , int Y ) {
int even_count = 0 , odd_count = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( ( arr [ i ] & 1 ) && ( i % 2 == 0 ) ) { odd_count ++ ; }
if ( ( arr [ i ] % 2 ) == 0 && ( i & 1 ) ) { even_count ++ ; } }
int cost1 = X * min ( odd_count , even_count ) ;
int cost2 = Y * ( max ( odd_count , even_count ) - min ( odd_count , even_count ) ) ;
int cost3 = ( odd_count + even_count ) * Y ;
return min ( cost1 + cost2 , cost3 ) ; }
int main ( ) { int arr [ ] = { 5 , 3 , 7 , 2 , 1 } , X = 10 , Y = 2 ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minimumCost ( arr , N , X , Y ) ; return 0 ; }
int findMinMax ( vector < int > & a ) {
int min_val = 1000000000 ;
for ( int i = 1 ; i < a . size ( ) ; ++ i ) {
min_val = min ( min_val , a [ i ] * a [ i - 1 ] ) ; }
return min_val ; }
int main ( ) { vector < int > arr = { 6 , 4 , 5 , 6 , 2 , 4 , 1 } ; cout << findMinMax ( arr ) ; return 0 ; }
struct TreeNode { int data ; TreeNode * left ; TreeNode * right ;
TreeNode ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; } } ;
void kDistanceDownSum ( TreeNode * root , int k , int & sum ) {
if ( root == NULL k < 0 ) return ;
if ( k == 0 ) { sum += root -> data ; return ; }
kDistanceDownSum ( root -> left , k - 1 , sum ) ; kDistanceDownSum ( root -> right , k - 1 , sum ) ; }
int kDistanceSum ( TreeNode * root , int target , int k , int & sum ) {
if ( root == NULL ) return -1 ;
if ( root -> data == target ) { kDistanceDownSum ( root -> left , k - 1 , sum ) ; return 0 ; }
int dl = -1 ;
if ( target < root -> data ) { dl = kDistanceSum ( root -> left , target , k , sum ) ; }
if ( dl != -1 ) {
if ( dl + 1 == k ) sum += root -> data ;
return -1 ; }
int dr = -1 ; if ( target > root -> data ) { dr = kDistanceSum ( root -> right , target , k , sum ) ; } if ( dr != -1 ) {
if ( dr + 1 == k ) sum += root -> data ;
else kDistanceDownSum ( root -> left , k - dr - 2 , sum ) ; return 1 + dr ; }
return -1 ; }
TreeNode * insertNode ( int data , TreeNode * root ) {
if ( root == NULL ) { TreeNode * node = new TreeNode ( data ) ; return node ; }
else if ( data > root -> data ) { root -> right = insertNode ( data , root -> right ) ; }
else if ( data <= root -> data ) { root -> left = insertNode ( data , root -> left ) ; }
return root ; }
void findSum ( TreeNode * root , int target , int K ) {
int sum = 0 ; kDistanceSum ( root , target , K , sum ) ;
cout << sum ; }
int main ( ) { TreeNode * root = NULL ; int N = 11 ; int tree [ ] = { 3 , 1 , 7 , 0 , 2 , 5 , 10 , 4 , 6 , 9 , 8 } ;
for ( int i = 0 ; i < N ; i ++ ) { root = insertNode ( tree [ i ] , root ) ; } int target = 7 ; int K = 2 ; findSum ( root , target , K ) ; return 0 ; }
int itemType ( int n ) {
int count = 0 ;
for ( int day = 1 ; ; day ++ ) {
for ( int type = day ; type > 0 ; type -- ) { count += type ;
if ( count >= n ) return type ; } } }
int main ( ) { int N = 10 ; cout << itemType ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int FindSum ( int arr [ ] , int N ) {
int res = 0 ;
for ( int i = 0 ; i < N ; i ++ ) {
int power = log2 ( arr [ i ] ) ;
int LesserValue = pow ( 2 , power ) ;
int LargerValue = pow ( 2 , power + 1 ) ;
if ( ( arr [ i ] - LesserValue ) == ( LargerValue - arr [ i ] ) ) {
res += arr [ i ] ; } }
return res ; }
int main ( ) { int arr [ ] = { 10 , 24 , 17 , 3 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << FindSum ( arr , N ) ; return 0 ; }
void findLast ( int mat [ ] [ 3 ] ) { int m = 3 ; int n = 3 ;
set < int > rows ; set < int > cols ; for ( int i = 0 ; i < m ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( mat [ i ] [ j ] ) { rows . insert ( i ) ; cols . insert ( j ) ; } } }
int avRows = m - rows . size ( ) ; int avCols = n - cols . size ( ) ;
int choices = min ( avRows , avCols ) ;
if ( choices & 1 )
cout << " P1" ;
else cout < < " P2" ; }
int main ( ) { int mat [ ] [ 3 ] = { { 1 , 0 , 0 } , { 0 , 0 , 0 } , { 0 , 0 , 1 } } ; findLast ( mat ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MOD = 1e9 + 7 ;
void sumOfBinaryNumbers ( int n ) {
int ans = 0 ; int one = 1 ;
while ( 1 ) {
if ( n <= 1 ) { ans = ( ans + n ) % MOD ; break ; }
int x = log2 ( n ) ; int cur = 0 ; int add = ( one << ( x - 1 ) ) ;
for ( int i = 1 ; i <= x ; i ++ ) {
cur = ( cur + add ) % MOD ; add = ( add * 10 % MOD ) ; }
ans = ( ans + cur ) % MOD ;
int rem = n - ( one << x ) + 1 ;
int p = pow ( 10 , x ) ; p = ( p * ( rem % MOD ) ) % MOD ; ans = ( ans + p ) % MOD ;
n = rem - 1 ; }
cout << ans ; }
int main ( ) { int N = 3 ; sumOfBinaryNumbers ( N ) ; return 0 ; }
void nearestFibonacci ( int num ) {
if ( num == 0 ) { cout << 0 ; return ; }
int first = 0 , second = 1 ;
int third = first + second ;
while ( third <= num ) {
first = second ;
second = third ;
third = first + second ; }
int ans = ( abs ( third - num ) >= abs ( second - num ) ) ? second : third ;
cout << ans ; }
int main ( ) { int N = 17 ; nearestFibonacci ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkPermutation ( int ans [ ] , int a [ ] , int n ) {
int Max = INT_MIN ;
for ( int i = 0 ; i < n ; i ++ ) {
Max = max ( Max , ans [ i ] ) ;
if ( Max != a [ i ] ) return false ; }
return true ; }
void findPermutation ( int a [ ] , int n ) {
int ans [ n ] = { 0 } ;
unordered_map < int , int > um ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( um . find ( a [ i ] ) == um . end ( ) ) {
ans [ i ] = a [ i ] ; um [ a [ i ] ] = i ; } }
vector < int > v ; int j = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) {
if ( um . find ( i ) == um . end ( ) ) { v . push_back ( i ) ; } }
for ( int i = 0 ; i < n ; i ++ ) {
if ( ans [ i ] == 0 ) { ans [ i ] = v [ j ] ; j ++ ; } }
if ( checkPermutation ( ans , a , n ) ) {
for ( int i = 0 ; i < n ; i ++ ) { cout << ans [ i ] << " ▁ " ; } }
else cout < < " - 1" ; }
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 5 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
findPermutation ( arr , N ) ; return 0 ; }
void countEqualElementPairs ( int arr [ ] , int N ) {
unordered_map < int , int > mp ;
for ( int i = 0 ; i < N ; i ++ ) { mp [ arr [ i ] ] += 1 ; }
int total = 0 ;
for ( auto i : mp ) {
total += ( i . second * ( i . second - 1 ) ) / 2 ; }
for ( int i = 0 ; i < N ; i ++ ) {
cout << total - ( mp [ arr [ i ] ] - 1 ) << " ▁ " ; } }
int main ( ) {
int arr [ ] = { 1 , 1 , 2 , 1 , 2 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; countEqualElementPairs ( arr , N ) ; }
int count ( int N ) { int sum = 0 ;
for ( int i = 1 ; i <= N ; i ++ ) { sum += 7 * pow ( 8 , i - 1 ) ; } return sum ; }
int main ( ) { int N = 4 ; cout << count ( N ) ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE using namespace std ;
bool isPalindrome ( int n ) {
string str = to_string ( n ) ;
int s = 0 , e = str . length ( ) - 1 ; while ( s < e ) {
if ( str [ s ] != str [ e ] ) { return false ; } s ++ ; e -- ; } return true ; }
void palindromicDivisors ( int n ) {
vector < int > PalindromDivisors ; for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) {
if ( n % i == 0 ) {
if ( n / i == i ) {
if ( isPalindrome ( i ) ) { PalindromDivisors . push_back ( i ) ; } } else {
if ( isPalindrome ( i ) ) { PalindromDivisors . push_back ( i ) ; }
if ( isPalindrome ( n / i ) ) { PalindromDivisors . push_back ( n / i ) ; } } } }
sort ( PalindromDivisors . begin ( ) , PalindromDivisors . end ( ) ) ; for ( int i = 0 ; i < PalindromDivisors . size ( ) ; i ++ ) { cout << PalindromDivisors [ i ] << " ▁ " ; } }
int main ( ) { int n = 66 ;
palindromicDivisors ( n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinDel ( int * arr , int n ) {
int min_num = INT_MAX ;
for ( int i = 0 ; i < n ; i ++ ) min_num = min ( arr [ i ] , min_num ) ;
int cnt = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] == min_num ) cnt ++ ;
return n - cnt ; }
int main ( ) { int arr [ ] = { 3 , 3 , 2 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << findMinDel ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int cntSubArr ( int * arr , int n ) {
int ans = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int curr_gcd = 0 ;
for ( int j = i ; j < n ; j ++ ) { curr_gcd = __gcd ( curr_gcd , arr [ j ] ) ;
ans += ( curr_gcd == 1 ) ; } }
return ans ; }
int main ( ) { int arr [ ] = { 1 , 1 , 1 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << cntSubArr ( arr , n ) ; return 0 ; }
void print_primes_till_N ( int N ) {
int i , j , flag ;
cout << " Prime numbers between 1 and " << N < < " ▁ are : STRNEWLINE " ;
for ( i = 1 ; i <= N ; i ++ ) {
if ( i == 1 i == 0 ) continue ;
flag = 1 ; for ( j = 2 ; j <= i / 2 ; ++ j ) { if ( i % j == 0 ) { flag = 0 ; break ; } }
if ( flag == 1 ) cout << i << " ▁ " ; } }
int main ( ) { int N = 100 ; print_primes_till_N ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  32
int findX ( int A , int B ) { int X = 0 ;
for ( int bit = 0 ; bit < MAX ; bit ++ ) {
int tempBit = 1 << bit ;
int bitOfX = A & B & tempBit ;
X += bitOfX ; } return X ; }
int main ( ) { int A = 11 , B = 13 ; cout << findX ( A , B ) ; return 0 ; }
int cntSubSets ( int arr [ ] , int n ) {
int maxVal = * max_element ( arr , arr + n ) ;
int cnt = 0 ; for ( int i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == maxVal ) cnt ++ ; }
return ( pow ( 2 , cnt ) - 1 ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 1 , 2 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << cntSubSets ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float findProb ( int arr [ ] , int n ) {
long maxSum = INT_MIN , maxCount = 0 , totalPairs = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { for ( int j = i + 1 ; j < n ; j ++ ) {
int sum = arr [ i ] + arr [ j ] ;
if ( sum == maxSum ) {
maxCount ++ ; }
else if ( sum > maxSum ) {
maxSum = sum ; maxCount = 1 ; } totalPairs ++ ; } }
float prob = ( float ) maxCount / ( float ) totalPairs ; return prob ; }
int main ( ) { int arr [ ] = { 1 , 1 , 1 , 2 , 2 , 2 } ; int n = sizeof ( arr ) / sizeof ( int ) ; cout << findProb ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxCommonFactors ( int a , int b ) {
int gcd = __gcd ( a , b ) ;
int ans = 1 ;
for ( int i = 2 ; i * i <= gcd ; i ++ ) { if ( gcd % i == 0 ) { ans ++ ; while ( gcd % i == 0 ) gcd /= i ; } }
if ( gcd != 1 ) ans ++ ;
return ans ; }
int main ( ) { int a = 12 , b = 18 ; cout << maxCommonFactors ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int days [ ] = { 31 , 28 , 31 , 30 , 31 , 30 , 31 , 31 , 30 , 31 , 30 , 31 } ;
int dayOfYear ( string date ) {
int year = stoi ( date . substr ( 0 , 4 ) ) ; int month = stoi ( date . substr ( 5 , 2 ) ) ; int day = stoi ( date . substr ( 8 ) ) ;
if ( month > 2 && year % 4 == 0 && ( year % 100 != 0 year % 400 == 0 ) ) { ++ day ; }
while ( month -- > 0 ) { day = day + days [ month - 1 ] ; } return day ; }
int main ( ) { string date = "2019-01-09" ; cout << dayOfYear ( date ) ; return 0 ; }
int Cells ( int n , int x ) { int ans = 0 ; for ( int i = 1 ; i <= n ; i ++ ) if ( x % i == 0 && x / i <= n ) ans ++ ; return ans ; }
int main ( ) { int n = 6 , x = 12 ;
cout << Cells ( n , x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int nextPowerOfFour ( int n ) { int x = floor ( sqrt ( sqrt ( n ) ) ) ;
if ( pow ( x , 4 ) == n ) return n ; else { x = x + 1 ; return pow ( x , 4 ) ; } }
int main ( ) { int n = 122 ; cout << nextPowerOfFour ( n ) ; return 0 ; }
int minOperations ( int x , int y , int p , int q ) {
if ( y % x != 0 ) return -1 ; int d = y / x ;
int a = 0 ;
while ( d % p == 0 ) { d /= p ; a ++ ; }
int b = 0 ;
while ( d % q == 0 ) { d /= q ; b ++ ; }
if ( d != 1 ) return -1 ;
return ( a + b ) ; }
int main ( ) { int x = 12 , y = 2592 , p = 2 , q = 3 ; cout << minOperations ( x , y , p , q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int nCr ( int n ) {
if ( n < 4 ) return 0 ; int answer = n * ( n - 1 ) * ( n - 2 ) * ( n - 3 ) ; answer /= 24 ; return answer ; }
int countQuadruples ( int N , int K ) {
int M = N / K ; int answer = nCr ( M ) ;
for ( int i = 2 ; i < M ; i ++ ) { int j = i ;
int temp2 = M / i ;
int count = 0 ;
int check = 0 ; int temp = j ; while ( j % 2 == 0 ) { count ++ ; j /= 2 ; if ( count >= 2 ) break ; } if ( count >= 2 ) { check = 1 ; } for ( int k = 3 ; k <= sqrt ( temp ) ; k += 2 ) { int cnt = 0 ; while ( j % k == 0 ) { cnt ++ ; j /= k ; if ( cnt >= 2 ) break ; } if ( cnt >= 2 ) { check = 1 ; break ; } else if ( cnt == 1 ) count ++ ; } if ( j > 2 ) { count ++ ; }
if ( check ) continue ; else {
if ( count % 2 == 1 ) { answer -= nCr ( temp2 ) ; } else { answer += nCr ( temp2 ) ; } } } return answer ; }
int main ( ) { int N = 10 , K = 2 ; cout << countQuadruples ( N , K ) ; return 0 ; }
int getX ( int a , int b , int c , int d ) { int X = ( b * c - a * d ) / ( d - c ) ; return X ; }
int main ( ) { int a = 2 , b = 3 , c = 4 , d = 5 ; cout << getX ( a , b , c , d ) ; return 0 ; }
bool isVowel ( char ch ) { if ( ch == ' a ' ch == ' e ' ch == ' i ' ch == ' o ' ch == ' u ' ) return true ; else return false ; }
ll fact ( ll n ) { if ( n < 2 ) return 1 ; return n * fact ( n - 1 ) ; }
ll only_vowels ( map < char , int > & freq ) { ll denom = 1 ; ll cnt_vwl = 0 ;
for ( auto itr = freq . begin ( ) ; itr != freq . end ( ) ; itr ++ ) { if ( isVowel ( itr -> first ) ) { denom *= fact ( itr -> second ) ; cnt_vwl += itr -> second ; } } return fact ( cnt_vwl ) / denom ; }
ll all_vowels_together ( map < char , int > & freq ) {
ll vow = only_vowels ( freq ) ;
ll denom = 1 ;
ll cnt_cnst = 0 ; for ( auto itr = freq . begin ( ) ; itr != freq . end ( ) ; itr ++ ) { if ( ! isVowel ( itr -> first ) ) { denom *= fact ( itr -> second ) ; cnt_cnst += itr -> second ; } }
ll ans = fact ( cnt_cnst + 1 ) / denom ; return ( ans * vow ) ; }
ll total_permutations ( map < char , int > & freq ) {
ll cnt = 0 ;
ll denom = 1 ; for ( auto itr = freq . begin ( ) ; itr != freq . end ( ) ; itr ++ ) { denom *= fact ( itr -> second ) ; cnt += itr -> second ; }
return fact ( cnt ) / denom ; }
ll no_vowels_together ( string & word ) {
map < char , int > freq ;
for ( int i = 0 ; i < word . size ( ) ; i ++ ) { char ch = tolower ( word [ i ] ) ; freq [ ch ] ++ ; }
ll total = total_permutations ( freq ) ;
ll vwl_tgthr = all_vowels_together ( freq ) ;
ll res = total - vwl_tgthr ;
return res ; }
int main ( ) { string word = " allahabad " ; ll ans = no_vowels_together ( word ) ; cout << ans << endl ; word = " geeksforgeeks " ; ans = no_vowels_together ( word ) ; cout << ans << endl ; word = " abcd " ; ans = no_vowels_together ( word ) ; cout << ans << endl ; return 0 ; }
int numberOfMen ( int D , int m , int d ) { int Men = ( m * ( D - d ) ) / d ; return Men ; }
int main ( ) { int D = 5 , m = 4 , d = 4 ; cout << numberOfMen ( D , m , d ) ; return 0 ; }
double area ( double a , double b , double c ) { double d = fabs ( ( c * c ) / ( 2 * a * b ) ) ; return d ; }
int main ( ) { double a = -2 , b = 4 , c = 3 ; cout << area ( a , b , c ) ; return 0 ; }
vector < int > addToArrayForm ( vector < int > & A , int K ) {
vector < int > v , ans ;
int rem = 0 ; int i = 0 ;
for ( i = A . size ( ) - 1 ; i >= 0 ; i -- ) {
int my = A [ i ] + K % 10 + rem ; if ( my > 9 ) {
rem = 1 ;
v . push_back ( my % 10 ) ; } else { v . push_back ( my ) ; rem = 0 ; } K = K / 10 ; }
while ( K > 0 ) {
int my = K % 10 + rem ; v . push_back ( my % 10 ) ;
if ( my / 10 > 0 ) rem = 1 ; else rem = 0 ; K = K / 10 ; } if ( rem > 0 ) v . push_back ( rem ) ;
for ( int i = v . size ( ) - 1 ; i >= 0 ; i -- ) ans . push_back ( v [ i ] ) ; return ans ; }
int main ( ) { vector < int > A { 2 , 7 , 4 } ; int K = 181 ; vector < int > ans = addToArrayForm ( A , K ) ;
for ( int i = 0 ; i < ans . size ( ) ; i ++ ) cout << ans [ i ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define MAX  100005 NEW_LINE using namespace std ;
int kadaneAlgorithm ( const int * ar , int n ) { int sum = 0 , maxSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) { sum += ar [ i ] ; if ( sum < 0 ) sum = 0 ; maxSum = max ( maxSum , sum ) ; } return maxSum ; }
int maxFunction ( const int * arr , int n ) { int b [ MAX ] , c [ MAX ] ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { if ( i & 1 ) { b [ i ] = abs ( arr [ i + 1 ] - arr [ i ] ) ; c [ i ] = - b [ i ] ; } else { c [ i ] = abs ( arr [ i + 1 ] - arr [ i ] ) ; b [ i ] = - c [ i ] ; } }
int ans = kadaneAlgorithm ( b , n - 1 ) ; ans = max ( ans , kadaneAlgorithm ( c , n - 1 ) ) ; return ans ; }
int main ( ) { int arr [ ] = { 1 , 5 , 4 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << maxFunction ( arr , n ) ; return 0 ; }
int findThirdDigit ( int n ) {
if ( n < 3 ) return 0 ;
return n & 1 ? 1 : 6 ; }
int main ( ) { int n = 7 ; cout << findThirdDigit ( n ) ; return 0 ; }
double getProbability ( int a , int b , int c , int d ) {
double p = ( double ) a / ( double ) b ; double q = ( double ) c / ( double ) d ;
double ans = p * ( 1 / ( 1 - ( 1 - q ) * ( 1 - p ) ) ) ; return ans ; }
int main ( ) { int a = 1 , b = 2 , c = 10 , d = 11 ; cout << getProbability ( a , b , c , d ) ; return 0 ; }
bool isPalindrome ( int n ) {
int divisor = 1 ; while ( n / divisor >= 10 ) divisor *= 10 ; while ( n != 0 ) { int leading = n / divisor ; int trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = ( n % divisor ) / 10 ;
divisor = divisor / 100 ; } return true ; }
int largestPalindrome ( int A [ ] , int n ) { int currentMax = -1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( A [ i ] > currentMax && isPalindrome ( A [ i ] ) ) currentMax = A [ i ] ; }
return currentMax ; }
int main ( ) { int A [ ] = { 1 , 232 , 54545 , 999991 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
cout << largestPalindrome ( A , n ) ; return 0 ; }
long getFinalElement ( long n ) { long finalNum ; for ( finalNum = 2 ; finalNum * 2 <= n ; finalNum *= 2 ) ; return finalNum ; }
int main ( ) { int N = 12 ; cout << getFinalElement ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void SieveOfEratosthenes ( bool prime [ ] , int p_size ) {
prime [ 0 ] = false ; prime [ 1 ] = false ; for ( int p = 2 ; p * p <= p_size ; p ++ ) {
if ( prime [ p ] ) {
for ( int i = p * 2 ; i <= p_size ; i += p ) prime [ i ] = false ; } } }
int sumOfElements ( int arr [ ] , int n ) { bool prime [ n + 1 ] ; memset ( prime , true , sizeof ( prime ) ) ; SieveOfEratosthenes ( prime , n + 1 ) ; int i , j ;
unordered_map < int , int > m ; for ( i = 0 ; i < n ; i ++ ) m [ arr [ i ] ] ++ ; int sum = 0 ;
for ( auto it = m . begin ( ) ; it != m . end ( ) ; it ++ ) {
if ( prime [ it -> second ] ) { sum += ( it -> first ) ; } } return sum ; }
int main ( ) { int arr [ ] = { 5 , 4 , 6 , 5 , 4 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << sumOfElements ( arr , n ) ; return 0 ; }
bool isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
bool isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
int main ( ) { int L = 110 , R = 1130 ; cout << " ▁ " << sumOfAllPalindrome ( L , R ) << endl ; }
ll fact ( int n ) { ll f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f = f * i ; return f ; }
ll waysOfConsonants ( int size1 , int freq [ ] ) { ll ans = fact ( size1 ) ; for ( int i = 0 ; i < 26 ; i ++ ) {
if ( i == 0 i == 4 i == 8 i == 14 i == 20 ) continue ; else ans = ans / fact ( freq [ i ] ) ; } return ans ; }
ll waysOfVowels ( int size2 , int freq [ ] ) { return fact ( size2 ) / ( fact ( freq [ 0 ] ) * fact ( freq [ 4 ] ) * fact ( freq [ 8 ] ) * fact ( freq [ 14 ] ) * fact ( freq [ 20 ] ) ) ; }
ll countWays ( string str ) { int freq [ 26 ] = { 0 } ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
int vowel = 0 , consonant = 0 ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) { if ( str [ i ] != ' a ' && str [ i ] != ' e ' && str [ i ] != ' i ' && str [ i ] != ' o ' && str [ i ] != ' u ' ) consonant ++ ; else vowel ++ ; }
return waysOfConsonants ( consonant + 1 , freq ) * waysOfVowels ( vowel , freq ) ; }
int main ( ) { string str = " geeksforgeeks " ; cout << countWays ( str ) << endl ; return 0 ; }
int calculateAlternateSum ( int n ) { if ( n <= 0 ) return 0 ; int fibo [ n + 1 ] ; fibo [ 0 ] = 0 , fibo [ 1 ] = 1 ;
int sum = pow ( fibo [ 0 ] , 2 ) + pow ( fibo [ 1 ] , 2 ) ;
for ( int i = 2 ; i <= n ; i ++ ) { fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] ;
if ( i % 2 == 0 ) sum -= fibo [ i ] ;
else sum += fibo [ i ] ; }
return sum ; }
int main ( ) {
int n = 8 ;
cout << " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " << n << " ▁ terms : ▁ " << calculateAlternateSum ( n ) << endl ; return 0 ; }
int getValue ( int n ) { int i = 0 , k = 1 ; while ( i < n ) { i = i + k ; k = k * 2 ; } return k / 2 ; }
int n = 9 ;
cout << getValue ( n ) << endl ;
n = 1025 ;
cout << getValue ( n ) << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countDigits ( double val , long arr [ ] ) { while ( ( long ) val > 0 ) { long digit = ( long ) val % 10 ; arr [ ( int ) digit ] ++ ; val = ( long ) val / 10 ; } return ; } void countFrequency ( int x , int n ) {
long freq_count [ 10 ] = { 0 } ;
for ( int i = 1 ; i <= n ; i ++ ) {
double val = pow ( ( double ) x , ( double ) i ) ;
countDigits ( val , freq_count ) ; }
for ( int i = 0 ; i <= 9 ; i ++ ) { cout << freq_count [ i ] << " ▁ " ; } }
int main ( ) { int x = 15 , n = 3 ; countFrequency ( x , n ) ; }
int countSolutions ( int a ) { int count = 0 ;
for ( int i = 0 ; i <= a ; i ++ ) { if ( a == ( i + ( a ^ i ) ) ) count ++ ; } return count ; }
int main ( ) { int a = 3 ; cout << countSolutions ( a ) ; }
int countSolutions ( int a ) { int count = __builtin_popcount ( a ) ; count = pow ( 2 , count ) ; return count ; }
int main ( ) { int a = 3 ; cout << countSolutions ( a ) ; }
int calculateAreaSum ( int l , int b ) { int size = 1 ;
int maxSize = min ( l , b ) ; int totalArea = 0 ; for ( int i = 1 ; i <= maxSize ; i ++ ) {
int totalSquares = ( l - size + 1 ) * ( b - size + 1 ) ;
int area = totalSquares * size * size ;
totalArea += area ;
size ++ ; } return totalArea ; }
int main ( ) { int l = 4 , b = 3 ; cout << calculateAreaSum ( l , b ) ; return 0 ; }
ll boost_hyperfactorial ( ll num ) {
ll val = 1 ; for ( int i = 1 ; i <= num ; i ++ ) { val = val * pow ( i , i ) ; }
return val ; }
int main ( ) { int num = 5 ; cout << boost_hyperfactorial ( num ) ; return 0 ; }
int1024_t boost_hyperfactorial ( int num ) {
int1024_t val = 1 ; for ( int i = 1 ; i <= num ; i ++ ) { for ( int j = 1 ; j <= i ; j ++ ) {
val *= i ; } }
return val ; }
int main ( ) { int num = 5 ; cout << boost_hyperfactorial ( num ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int subtractOne ( int x ) { int m = 1 ;
while ( ! ( x & m ) ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
int main ( ) { cout << subtractOne ( 13 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define rows  3 NEW_LINE #define cols  3
void meanVector ( int mat [ rows ] [ cols ] ) { cout << " [ ▁ " ;
for ( int i = 0 ; i < rows ; i ++ ) {
double mean = 0.00 ;
int sum = 0 ; for ( int j = 0 ; j < cols ; j ++ ) sum += mat [ j ] [ i ] ; mean = sum / rows ; cout << mean << " ▁ " ; } cout << " ] " ; }
int main ( ) { int mat [ rows ] [ cols ] = { { 1 , 2 , 3 } , { 4 , 5 , 6 } , { 7 , 8 , 9 } } ; meanVector ( mat ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > primeFactors ( int n ) { vector < int > res ; if ( n % 2 == 0 ) { while ( n % 2 == 0 ) n = n / 2 ; res . push_back ( 2 ) ; }
for ( int i = 3 ; i <= sqrt ( n ) ; i = i + 2 ) {
if ( n % i == 0 ) { while ( n % i == 0 ) n = n / i ; res . push_back ( i ) ; } }
if ( n > 2 ) res . push_back ( n ) ; return res ; }
bool isHoax ( int n ) {
vector < int > pf = primeFactors ( n ) ;
if ( pf [ 0 ] == n ) return false ;
int all_pf_sum = 0 ; for ( int i = 0 ; i < pf . size ( ) ; i ++ ) {
int pf_sum ; for ( pf_sum = 0 ; pf [ i ] > 0 ; pf_sum += pf [ i ] % 10 , pf [ i ] /= 10 ) ; all_pf_sum += pf_sum ; }
int sum_n ; for ( sum_n = 0 ; n > 0 ; sum_n += n % 10 , n /= 10 ) ;
return sum_n == all_pf_sum ; }
int main ( ) { int n = 84 ; if ( isHoax ( n ) ) cout << " A ▁ Hoax ▁ Number STRNEWLINE " ; else cout << " Not ▁ a ▁ Hoax ▁ Number STRNEWLINE " ; return 0 ; }
void LucasLehmer ( int n ) {
unsigned long long current_val = 4 ;
vector < unsigned long long > series ;
series . push_back ( current_val ) ; for ( int i = 0 ; i < n ; i ++ ) { current_val = current_val * current_val - 2 ; series . push_back ( current_val ) ; }
for ( int i = 0 ; i <= n ; i ++ ) cout << " Term ▁ " << i << " : ▁ " << series [ i ] << endl ; }
int main ( ) { int n = 5 ; LucasLehmer ( n ) ; return 0 ; }
int modInverse ( int a , int prime ) { a = a % prime ; for ( int x = 1 ; x < prime ; x ++ ) if ( ( a * x ) % prime == 1 ) return x ; return -1 ; } void printModIverses ( int n , int prime ) { for ( int i = 1 ; i <= n ; i ++ ) cout << modInverse ( i , prime ) << " ▁ " ; }
int main ( ) { int n = 10 , prime = 17 ; printModIverses ( n , prime ) ; return 0 ; }
int minOp ( long long int num ) {
int rem ; int count = 0 ;
while ( num ) { rem = num % 10 ; if ( ! ( rem == 3 rem == 8 ) ) count ++ ; num /= 10 ; } return count ; }
int main ( ) { long long int num = 234198 ; cout << " Minimum ▁ Operations ▁ = " << minOp ( num ) ; return 0 ; }
int sumOfDigits ( int a ) { int sum = 0 ; while ( a ) { sum += a % 10 ; a /= 10 ; } return sum ; }
int findMax ( int x ) {
int b = 1 , ans = x ;
while ( x ) {
int cur = ( x - 1 ) * b + ( b - 1 ) ;
if ( sumOfDigits ( cur ) > sumOfDigits ( ans ) || ( sumOfDigits ( cur ) == sumOfDigits ( ans ) && cur > ans ) ) ans = cur ;
x /= 10 ; b *= 10 ; } return ans ; }
int main ( ) { int n = 521 ; cout << findMax ( n ) ; return 0 ; }
int median ( int * a , int l , int r ) { int n = r - l + 1 ; n = ( n + 1 ) / 2 - 1 ; return n + l ; }
int IQR ( int * a , int n ) { sort ( a , a + n ) ;
int mid_index = median ( a , 0 , n ) ;
int Q1 = a [ median ( a , 0 , mid_index ) ] ;
int Q3 = a [ mid_index + median ( a , mid_index + 1 , n ) ] ;
return ( Q3 - Q1 ) ; }
int main ( ) { int a [ ] = { 1 , 19 , 7 , 6 , 5 , 9 , 12 , 27 , 18 , 2 , 15 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << IQR ( a , n ) ; return 0 ; }
bool isPalindrome ( int n ) {
int divisor = 1 ; while ( n / divisor >= 10 ) divisor *= 10 ; while ( n != 0 ) { int leading = n / divisor ; int trailing = n % 10 ;
if ( leading != trailing ) return false ;
n = ( n % divisor ) / 10 ;
divisor = divisor / 100 ; } return true ; }
int largestPalindrome ( int A [ ] , int n ) {
sort ( A , A + n ) ; for ( int i = n - 1 ; i >= 0 ; -- i ) {
if ( isPalindrome ( A [ i ] ) ) return A [ i ] ; }
return -1 ; }
int main ( ) { int A [ ] = { 1 , 232 , 54545 , 999991 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ;
cout << largestPalindrome ( A , n ) ; return 0 ; }
int findSum ( int n , int a , int b ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
int main ( ) { int n = 10 , a = 3 , b = 5 ; cout << findSum ( n , a , b ) ; return 0 ; }
#include <stdio.h> NEW_LINE int subtractOne ( int x ) { return ( ( x << 1 ) + ( ~ x ) ) ; } int main ( ) { printf ( " % d " , subtractOne ( 13 ) ) ; return 0 ; }
int pell ( int n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
int main ( ) { int n = 4 ; cout << " ▁ " << pell ( n ) ; return 0 ; }
unsigned long long int LCM ( int arr [ ] , int n ) {
int max_num = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( max_num < arr [ i ] ) max_num = arr [ i ] ;
unsigned long long int res = 1 ;
while ( x <= max_num ) {
vector < int > indexes ; for ( int j = 0 ; j < n ; j ++ ) if ( arr [ j ] % x == 0 ) indexes . push_back ( j ) ;
if ( indexes . size ( ) >= 2 ) {
for ( int j = 0 ; j < indexes . size ( ) ; j ++ ) arr [ indexes [ j ] ] = arr [ indexes [ j ] ] / x ; res = res * x ; } else x ++ ; }
for ( int i = 0 ; i < n ; i ++ ) res = res * arr [ i ] ; return res ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 10 , 20 , 35 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << LCM ( arr , n ) << " STRNEWLINE " ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
int politness ( int n ) { int count = 0 ;
for ( int i = 2 ; i <= sqrt ( 2 * n ) ; i ++ ) { int a ; if ( ( 2 * n ) % i != 0 ) continue ; a = 2 * n ; a /= i ; a -= ( i - 1 ) ; if ( a % 2 != 0 ) continue ; a /= 2 ; if ( a > 0 ) { count ++ ; } } return count ; }
int main ( ) { int n = 90 ; cout << " Politness ▁ of ▁ " << n << " ▁ = ▁ " << politness ( n ) << " STRNEWLINE " ; n = 15 ; cout << " Politness ▁ of ▁ " << n << " ▁ = ▁ " << politness ( n ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 10000 ;
vector < int > primes ;
void sieveSundaram ( ) {
bool marked [ MAX / 2 + 100 ] = { 0 } ;
for ( int i = 1 ; i <= ( sqrt ( MAX ) - 1 ) / 2 ; i ++ ) for ( int j = ( i * ( i + 1 ) ) << 1 ; j <= MAX / 2 ; j = j + 2 * i + 1 ) marked [ j ] = true ;
primes . push_back ( 2 ) ;
for ( int i = 1 ; i <= MAX / 2 ; i ++ ) if ( marked [ i ] == false ) primes . push_back ( 2 * i + 1 ) ; }
void findPrimes ( int n ) {
if ( n <= 2 n % 2 != 0 ) { cout << " Invalid ▁ Input ▁ STRNEWLINE " ; return ; }
for ( int i = 0 ; primes [ i ] <= n / 2 ; i ++ ) {
int diff = n - primes [ i ] ;
if ( binary_search ( primes . begin ( ) , primes . end ( ) , diff ) ) {
cout << primes [ i ] << " ▁ + ▁ " << diff << " ▁ = ▁ " << n << endl ; return ; } } }
sieveSundaram ( ) ;
findPrimes ( 4 ) ; findPrimes ( 38 ) ; findPrimes ( 100 ) ; return 0 ; }
# include <bits/stdc++.h> NEW_LINE using namespace std ;
int kPrimeFactor ( int n , int k ) {
while ( n % 2 == 0 ) { k -- ; n = n / 2 ; if ( k == 0 ) return 2 ; }
for ( int i = 3 ; i <= sqrt ( n ) ; i = i + 2 ) {
while ( n % i == 0 ) { if ( k == 1 ) return i ; k -- ; n = n / i ; } }
if ( n > 2 && k == 1 ) return n ; return -1 ; }
int main ( ) { int n = 12 , k = 3 ; cout << kPrimeFactor ( n , k ) << endl ; n = 14 , k = 3 ; cout << kPrimeFactor ( n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; const int MAX = 10001 ;
void sieveOfEratosthenes ( int s [ ] ) {
vector < bool > prime ( MAX + 1 , false ) ;
for ( int i = 2 ; i <= MAX ; i += 2 ) s [ i ] = 2 ;
for ( int i = 3 ; i <= MAX ; i += 2 ) { if ( prime [ i ] == false ) {
s [ i ] = i ;
for ( int j = i ; j * i <= MAX ; j += 2 ) { if ( prime [ i * j ] == false ) { prime [ i * j ] = true ;
s [ i * j ] = i ; } } } } }
int kPrimeFactor ( int n , int k , int s [ ] ) {
while ( n > 1 ) { if ( k == 1 ) return s [ n ] ;
k -- ;
n /= s [ n ] ; } return -1 ; }
int s [ MAX + 1 ] ; memset ( s , -1 , sizeof ( s ) ) ; sieveOfEratosthenes ( s ) ; int n = 12 , k = 3 ; cout << kPrimeFactor ( n , k , s ) << endl ; n = 14 , k = 3 ; cout << kPrimeFactor ( n , k , s ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int sumDivisorsOfDivisors ( int n ) {
map < int , int > mp ; for ( int j = 2 ; j <= sqrt ( n ) ; j ++ ) { int count = 0 ; while ( n % j == 0 ) { n /= j ; count ++ ; } if ( count ) mp [ j ] = count ; }
if ( n != 1 ) mp [ n ] = 1 ;
int ans = 1 ; for ( auto it : mp ) { int pw = 1 ; int sum = 0 ; for ( int i = it . second + 1 ; i >= 1 ; i -- ) { sum += ( i * pw ) ; pw *= it . first ; } ans *= sum ; } return ans ; }
int main ( ) { int n = 10 ; cout << sumDivisorsOfDivisors ( n ) ; return 0 ; }
string fractionToDecimal ( int numr , int denr ) {
map < int , int > mp ; mp . clear ( ) ;
int rem = numr % denr ;
while ( ( rem != 0 ) && ( mp . find ( rem ) == mp . end ( ) ) ) {
mp [ rem ] = res . length ( ) ;
rem = rem * 10 ;
int res_part = rem / denr ; res += to_string ( res_part ) ;
rem = rem % denr ; } return ( rem == 0 ) ? " " : res . substr ( mp [ rem ] ) ; }
int main ( ) { int numr = 50 , denr = 22 ; string res = fractionToDecimal ( numr , denr ) ; if ( res == " " ) cout << " No ▁ recurring ▁ sequence " ; else cout << " Recurring ▁ sequence ▁ is ▁ " << res ; return 0 ; }
int has0 ( int x ) {
while ( x ) {
if ( x % 10 == 0 ) return 1 ; x /= 10 ; } return 0 ; }
int getCount ( int n ) {
int count = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) count += has0 ( i ) ; return count ; }
int main ( ) { int n = 107 ; cout << " Count ▁ of ▁ numbers ▁ from ▁ 1" << " ▁ to ▁ " << n << " ▁ is ▁ " << getCount ( n ) ; }
bool squareRootExists ( int n , int p ) { n = n % p ;
for ( int x = 2 ; x < p ; x ++ ) if ( ( x * x ) % p == n ) return true ; return false ; }
int main ( ) { int p = 7 ; int n = 2 ; squareRootExists ( n , p ) ? cout << " Yes " : cout << " No " ; return 0 ; }
int largestPower ( int n , int p ) {
int x = 0 ;
while ( n ) { n /= p ; x += n ; } return x ; }
int main ( ) { int n = 10 , p = 3 ; cout << " The ▁ largest ▁ power ▁ of ▁ " << p << " ▁ that ▁ divides ▁ " << n << " ! ▁ is ▁ " << largestPower ( n , p ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
int main ( ) { int num = 5 ; cout << " Factorial ▁ of ▁ " << num << " ▁ is ▁ " << factorial ( num ) ; return 0 ; }
boolean getBit ( int num , int i ) {
return ( ( num & ( 1 << i ) ) != 0 ) ; }
int clearBit ( int num , int i ) {
int mask = ~ ( 1 << i ) ;
return num & mask ; }
void Bitwise_AND_sum_i ( int arr1 [ ] , int arr2 [ ] , int M , int N ) {
int frequency [ 32 ] = { 0 } ;
for ( int i = 0 ; i < N ; i ++ ) {
int bit_position = 0 ; int num = arr1 [ i ] ;
while ( num ) {
if ( num & 1 ) {
frequency [ bit_position ] += 1 ; }
bit_position += 1 ;
num >>= 1 ; } }
for ( int i = 0 ; i < M ; i ++ ) { int num = arr2 [ i ] ;
int value_at_that_bit = 1 ;
int bitwise_AND_sum = 0 ;
for ( int bit_position = 0 ; bit_position < 32 ; bit_position ++ ) {
if ( num & 1 ) {
bitwise_AND_sum += frequency [ bit_position ] * value_at_that_bit ; }
num >>= 1 ;
value_at_that_bit <<= 1 ; }
cout << bitwise_AND_sum << ' ▁ ' ; } return ; }
int main ( ) {
int arr1 [ ] = { 1 , 2 , 3 } ;
int arr2 [ ] = { 1 , 2 , 3 } ;
int N = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ;
int M = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ;
Bitwise_AND_sum_i ( arr1 , arr2 , M , N ) ; return 0 ; }
void FlipBits ( int n ) { for ( int bit = 0 ; bit < 32 ; bit ++ ) {
if ( ( n >> bit ) & 1 ) {
n = n ^ ( 1ll << bit ) ; break ; } } cout << " The ▁ number ▁ after ▁ unsetting ▁ the " ; cout << " ▁ rightmost ▁ set ▁ bit ▁ " << n ; }
int main ( ) { int N = 12 ; FlipBits ( N ) ; return 0 ; }
int bitwiseAndOdd ( int n ) {
int result = 1 ;
for ( int i = 3 ; i <= n ; i = i + 2 ) { result = ( result & i ) ; } return result ; }
int main ( ) { int n = 10 ; cout << bitwiseAndOdd ( n ) ; return 0 ; }
int bitwiseAndOdd ( int n ) { return 1 ; }
int main ( ) { int n = 10 ; cout << bitwiseAndOdd ( n ) ; return 0 ; }
unsigned int reverseBits ( unsigned int n ) { unsigned int rev = 0 ;
while ( n > 0 ) {
rev <<= 1 ;
if ( n & 1 == 1 ) rev ^= 1 ;
n >>= 1 ; }
return rev ; }
int main ( ) { unsigned int n = 11 ; cout << reverseBits ( n ) ; return 0 ; }
int countgroup ( int a [ ] , int n ) { int xs = 0 ; for ( int i = 0 ; i < n ; i ++ ) xs = xs ^ a [ i ] ;
if ( xs == 0 ) return ( 1 << ( n - 1 ) ) - 1 ; return 0 ; }
int main ( ) { int a [ ] = { 1 , 2 , 3 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << countgroup ( a , n ) << endl ; return 0 ; }
int bitExtracted ( int number , int k , int p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
int main ( ) { int number = 171 , k = 5 , p = 2 ; cout << " The ▁ extracted ▁ number ▁ is ▁ " << bitExtracted ( number , k , p ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ull  unsigned long long int NEW_LINE ull findMax ( ull num ) { ull num_copy = num ;
int j = sizeof ( unsigned long long int ) * 8 - 1 ; int i = 0 ; while ( i < j ) {
int m = ( num_copy >> i ) & 1 ; int n = ( num_copy >> j ) & 1 ;
if ( m > n ) { int x = ( 1 << i 1 << j ) ; num = num ^ x ; } i ++ ; j -- ; } return num ; }
int main ( ) { ull num = 4 ; cout << findMax ( num ) ; return 0 ; }
string isAMultipleOf4 ( int n ) {
if ( ( n & 3 ) == 0 ) return " Yes " ;
return " No " ; }
int main ( ) { int n = 16 ; cout << isAMultipleOf4 ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int square ( int n ) {
if ( n < 0 ) n = - n ;
int res = n ;
for ( int i = 1 ; i < n ; i ++ ) res += n ; return res ; }
int main ( ) { for ( int n = 1 ; n <= 5 ; n ++ ) cout << " n ▁ = ▁ " << n << " , ▁ n ^ 2 ▁ = ▁ " << square ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int PointInKSquares ( int n , int a [ ] , int k ) { sort ( a , a + n ) ; return a [ n - k ] ; }
int main ( ) { int k = 2 ; int a [ ] = { 1 , 2 , 3 , 4 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int x = PointInKSquares ( n , a , k ) ; cout << " ( " << x << " , ▁ " << x << " ) " ; }
long long answer ( int n ) {
int dp [ 10 ] ;
int prev [ 10 ] ;
if ( n == 1 ) return 10 ;
for ( int j = 0 ; j <= 9 ; j ++ ) dp [ j ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= 9 ; j ++ ) { prev [ j ] = dp [ j ] ; } for ( int j = 0 ; j <= 9 ; j ++ ) {
if ( j == 0 ) dp [ j ] = prev [ j + 1 ] ;
else if ( j == 9 ) dp [ j ] = prev [ j - 1 ] ;
else dp [ j ] = prev [ j - 1 ] + prev [ j + 1 ] ; } }
long long sum = 0 ; for ( int j = 1 ; j <= 9 ; j ++ ) sum += dp [ j ] ; return sum ; }
int main ( ) { int n = 2 ; cout << answer ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100000 NEW_LINE #define ll  long long int
ll catalan [ MAX ] ;
void catalanDP ( ll n ) {
catalan [ 0 ] = catalan [ 1 ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { catalan [ i ] = 0 ; for ( int j = 0 ; j < i ; j ++ ) catalan [ i ] += catalan [ j ] * catalan [ i - j - 1 ] ; } }
int CatalanSequence ( int arr [ ] , int n ) {
catalanDP ( n ) ; unordered_multiset < int > s ;
int a = 1 , b = 1 ; int c ;
s . insert ( a ) ; if ( n >= 2 ) s . insert ( b ) ; for ( int i = 2 ; i < n ; i ++ ) { s . insert ( catalan [ i ] ) ; } unordered_multiset < int > :: iterator it ; for ( int i = 0 ; i < n ; i ++ ) {
it = s . find ( arr [ i ] ) ; if ( it != s . end ( ) ) s . erase ( it ) ; }
return s . size ( ) ; }
int main ( ) { int arr [ ] = { 1 , 1 , 2 , 5 , 41 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << CatalanSequence ( arr , n ) ; return 0 ; }
int composite ( int n ) { int flag = 0 ; int c = 0 ;
for ( int j = 1 ; j <= n ; j ++ ) { if ( n % j == 0 ) { c += 1 ; } }
if ( c >= 3 ) flag = 1 ; return flag ; }
void odd_indices ( int arr [ ] , int n ) { int sum = 0 ;
for ( int k = 0 ; k < n ; k += 2 ) { int check = composite ( arr [ k ] ) ;
if ( check == 1 ) sum += arr [ k ] ; }
cout << sum << endl ; }
int main ( ) { int arr [ ] = { 13 , 5 , 8 , 16 , 25 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; odd_indices ( arr , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void preprocess ( int p [ ] , int x [ ] , int y [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) p [ i ] = x [ i ] * x [ i ] + y [ i ] * y [ i ] ; sort ( p , p + n ) ; }
int query ( int p [ ] , int n , int rad ) { int start = 0 , end = n - 1 ; while ( ( end - start ) > 1 ) { int mid = ( start + end ) / 2 ; double tp = sqrt ( p [ mid ] ) ; if ( tp > ( rad * 1.0 ) ) end = mid - 1 ; else start = mid ; } double tp1 = sqrt ( p [ start ] ) , tp2 = sqrt ( p [ end ] ) ; if ( tp1 > ( rad * 1.0 ) ) return 0 ; else if ( tp2 <= ( rad * 1.0 ) ) return end + 1 ; else return start + 1 ; }
int main ( ) { int x [ ] = { 1 , 2 , 3 , -1 , 4 } ; int y [ ] = { 1 , 2 , 3 , -1 , 4 } ; int n = sizeof ( x ) / sizeof ( x [ 0 ] ) ;
int p [ n ] ; preprocess ( p , x , y , n ) ;
cout << query ( p , n , 3 ) << endl ;
cout << query ( p , n , 32 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int find_Numb_ways ( int n ) {
int odd_indices = n / 2 ;
int even_indices = ( n / 2 ) + ( n % 2 ) ;
int arr_odd = pow ( 4 , odd_indices ) ;
int arr_even = pow ( 5 , even_indices ) ;
return arr_odd * arr_even ; }
int main ( ) { int n = 4 ; cout << find_Numb_ways ( n ) << endl ; return 0 ; }
bool isSpiralSorted ( int arr [ ] , int n ) {
int start = 0 ;
int end = n - 1 ; while ( start < end ) {
if ( arr [ start ] > arr [ end ] ) { return false ; }
start ++ ;
if ( arr [ end ] > arr [ start ] ) { return false ; }
end -- ; } return true ; }
int main ( ) { int arr [ ] = { 1 , 10 , 14 , 20 , 18 , 12 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( isSpiralSorted ( arr , N ) ) cout << " YES " << endl ; else cout << " NO " << endl ; return 0 ; }
void findWordsSameRow ( vector < string > & arr ) {
unordered_map < char , int > mp { { ' q ' , 1 } , { ' w ' , 1 } , { ' e ' , 1 } , { ' r ' , 1 } , { ' t ' , 1 } , { ' y ' , 1 } , { ' u ' , 1 } , { ' o ' , 1 } , { ' p ' , 1 } , { ' i ' , 1 } , { ' a ' , 2 } , { ' s ' , 2 } , { ' d ' , 2 } , { ' f ' , 2 } , { ' g ' , 2 } , { ' h ' , 2 } , { ' j ' , 2 } , { ' k ' , 2 } , { ' l ' , 2 } , { ' z ' , 3 } , { ' x ' , 3 } , { ' c ' , 3 } , { ' v ' , 3 } , { ' b ' , 3 } , { ' n ' , 3 } , { ' m ' , 3 } } ;
for ( auto word : arr ) {
if ( ! word . empty ( ) ) {
bool flag = true ;
int rowNum = mp [ tolower ( word [ 0 ] ) ] ;
int M = word . length ( ) ;
for ( int i = 1 ; i < M ; i ++ ) {
if ( mp [ tolower ( word [ i ] ) ] != rowNum ) {
flag = false ; break ; } }
if ( flag ) {
cout << word << " ▁ " ; } } } }
int main ( ) { vector < string > words = { " Yeti " , " Had " , " GFG " , " comment " } ; findWordsSameRow ( words ) ; }
#include <iostream> NEW_LINE using namespace std ; const int maxN = 2002 ;
int countSubsequece ( int a [ ] , int n ) { int i , j , k , l ;
int answer = 0 ;
for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { for ( k = j + 1 ; k < n ; k ++ ) { for ( l = k + 1 ; l < n ; l ++ ) {
int main ( ) { int a [ 7 ] = { 1 , 2 , 3 , 2 , 1 , 3 , 2 } ; cout << countSubsequece ( a , 7 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
char minDistChar ( string s ) { int n = s . length ( ) ;
int * first = new int [ 26 ] ; int * last = new int [ 26 ] ;
for ( int i = 0 ; i < 26 ; i ++ ) { first [ i ] = -1 ; last [ i ] = -1 ; }
for ( int i = 0 ; i < n ; i ++ ) {
if ( first [ s [ i ] - ' a ' ] == -1 ) { first [ s [ i ] - ' a ' ] = i ; }
last [ s [ i ] - ' a ' ] = i ; }
int min = INT_MAX ; char ans = '1' ;
for ( int i = 0 ; i < 26 ; i ++ ) {
if ( last [ i ] == first [ i ] ) continue ;
if ( min > last [ i ] - first [ i ] ) { min = last [ i ] - first [ i ] ; ans = i + ' a ' ; } }
return ans ; }
int main ( ) { string str = " geeksforgeeks " ;
cout << minDistChar ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define n  3 NEW_LINE using namespace std ;
int minSteps ( int arr [ ] [ n ] ) {
bool v [ n ] [ n ] = { 0 } ;
queue < pair < int , int > > q ;
int depth = 0 ;
while ( q . size ( ) != 0 ) {
int x = q . size ( ) ; while ( x -- ) {
pair < int , int > y = q . front ( ) ;
int i = y . first , j = y . second ; q . pop ( ) ;
if ( v [ i ] [ j ] ) continue ;
if ( i == n - 1 && j == n - 1 ) return depth ;
v [ i ] [ j ] = 1 ;
if ( i + arr [ i ] [ j ] < n ) q . push ( { i + arr [ i ] [ j ] , j } ) ; if ( j + arr [ i ] [ j ] < n ) q . push ( { i , j + arr [ i ] [ j ] } ) ; } depth ++ ; } return -1 ; }
int main ( ) { int arr [ n ] [ n ] = { { 1 , 1 , 1 } , { 1 , 1 , 1 } , { 1 , 1 , 1 } } ; cout << minSteps ( arr ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int solve ( int a [ ] , int n ) { int max1 = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
int main ( ) { int arr [ ] = { -1 , 2 , 3 , -4 , -10 , 22 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Largest ▁ gap ▁ is ▁ : ▁ " << solve ( arr , size ) ; return 0 ; }
int solve ( int a [ ] , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return abs ( min1 - max1 ) ; }
int main ( ) { int arr [ ] = { -1 , 2 , 3 , 4 , -10 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Largest ▁ gap ▁ is ▁ : ▁ " << solve ( arr , size ) ; return 0 ; }
void replaceOriginal ( string s , int n ) {
string r ( n , ' ▁ ' ) ;
for ( int i = 0 ; i < n ; i ++ ) {
r [ i ] = s [ n - 1 - i ] ;
if ( s [ i ] != ' a ' && s [ i ] != ' e ' && s [ i ] != ' i ' && s [ i ] != ' o ' && s [ i ] != ' u ' ) { cout << r [ i ] ; } } cout << endl ; }
int main ( ) { string s = " geeksforgeeks " ; int n = s . length ( ) ; replaceOriginal ( s , n ) ; return 0 ; }
bool sameStrings ( string str1 , string str2 ) { int N = str1 . length ( ) ; int M = str2 . length ( ) ;
if ( N != M ) { return false ; }
int a [ 256 ] = { 0 } , b [ 256 ] = { 0 } ;
for ( int i = 0 ; i < N ; i ++ ) { a [ str1 [ i ] - ' a ' ] ++ ; b [ str2 [ i ] - ' a ' ] ++ ; }
int i = 0 ; while ( i < 256 ) { if ( ( a [ i ] == 0 && b [ i ] == 0 ) || ( a [ i ] != 0 && b [ i ] != 0 ) ) { i ++ ; }
else { return false ; } }
sort ( a , a + 256 ) ; sort ( b , b + 256 ) ;
for ( int i = 0 ; i < 256 ; i ++ ) {
if ( a [ i ] != b [ i ] ) return false ; }
return true ; }
int main ( ) { string S1 = " cabbba " , S2 = " abbccc " ; if ( sameStrings ( S1 , S2 ) ) cout << " YES " << endl ; else cout << " ▁ NO " << endl ; return 0 ; }
int solution ( int A , int B , int C ) { int arr [ 3 ] ;
arr [ 0 ] = A , arr [ 1 ] = B , arr [ 2 ] = C ;
sort ( arr , arr + 3 ) ;
if ( arr [ 2 ] < arr [ 0 ] + arr [ 1 ] ) return ( ( arr [ 0 ] + arr [ 1 ] + arr [ 2 ] ) / 2 ) ;
else return ( arr [ 0 ] + arr [ 1 ] ) ; }
int main ( ) {
int A = 8 , B = 1 , C = 5 ;
cout << solution ( A , B , C ) ; return 0 ; }
int search ( int arr [ ] , int l , int h , int key ) { if ( l > h ) return -1 ; int mid = ( l + h ) / 2 ; if ( arr [ mid ] == key ) return mid ;
if ( ( arr [ l ] == arr [ mid ] ) && ( arr [ h ] == arr [ mid ] ) ) { ++ l ; -- h ; return search ( arr , l , h , key ) ; }
if ( arr [ l ] <= arr [ mid ] ) {
if ( key >= arr [ l ] && key <= arr [ mid ] ) return search ( arr , l , mid - 1 , key ) ;
return search ( arr , mid + 1 , h , key ) ; }
if ( key >= arr [ mid ] && key <= arr [ h ] ) return search ( arr , mid + 1 , h , key ) ; return search ( arr , l , mid - 1 , key ) ; }
int main ( ) { int arr [ ] = { 3 , 3 , 1 , 2 , 3 , 3 } ; int n = sizeof ( arr ) / sizeof ( int ) ; int key = 3 ; cout << search ( arr , 0 , n - 1 , key ) ; return 0 ; }
string getSortedString ( string s , int n ) {
vector < char > v1 , v2 ; for ( int i = 0 ; i < n ; i ++ ) { if ( s [ i ] >= ' a ' && s [ i ] <= ' z ' ) v1 . push_back ( s [ i ] ) ; if ( s [ i ] >= ' A ' && s [ i ] <= ' Z ' ) v2 . push_back ( s [ i ] ) ; }
sort ( v1 . begin ( ) , v1 . end ( ) ) ; sort ( v2 . begin ( ) , v2 . end ( ) ) ; int i = 0 , j = 0 ; for ( int k = 0 ; k < n ; k ++ ) {
if ( s [ k ] >= ' a ' && s [ k ] <= ' z ' ) { s [ k ] = v1 [ i ] ; ++ i ; }
else if ( s [ k ] >= ' A ' && s [ k ] <= ' Z ' ) { s [ k ] = v2 [ j ] ; ++ j ; } }
return s ; }
int main ( ) { string s = " gEeksfOrgEEkS " ; int n = s . length ( ) ; cout << getSortedString ( s , n ) ; return 0 ; }
bool check ( string s ) {
int l = s . length ( ) ;
sort ( s . begin ( ) , s . end ( ) ) ;
for ( int i = 1 ; i < l ; i ++ ) {
if ( s [ i ] - s [ i - 1 ] != 1 ) return false ; } return true ; }
int main ( ) {
string str = " dcef " ; if ( check ( str ) ) cout << " Yes STRNEWLINE " ; else cout << " No STRNEWLINE " ;
str = " xyza " ; if ( check ( str ) ) cout << " Yes STRNEWLINE " ; else cout << " No STRNEWLINE " ; return 0 ; }
int minElements ( int arr [ ] , int n ) {
int halfSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = halfSum / 2 ;
sort ( arr , arr + n , greater < int > ( ) ) ; int res = 0 , curr_sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
int main ( ) { int arr [ ] = { 3 , 1 , 7 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minElements ( arr , n ) << endl ; return 0 ; }
void arrayElementEqual ( int arr [ ] , int N ) {
int sum = 0 ;
for ( int i = 0 ; i < N ; i ++ ) { sum += arr [ i ] ; }
if ( sum % N == 0 ) { cout << " Yes " ; }
else { cout << " No " << endl ; } }
int arr [ ] = { 1 , 5 , 6 , 4 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; arrayElementEqual ( arr , N ) ; }
int findMaxValByRearrArr ( int arr [ ] , int N ) {
int res = 0 ;
res = ( N * ( N + 1 ) ) / 2 ; return res ; }
int main ( ) { int arr [ ] = { 3 , 2 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMaxValByRearrArr ( arr , N ) ; return 0 ; }
int MaximumSides ( int n ) {
if ( n < 4 ) return -1 ;
return n % 2 == 0 ? n / 2 : -1 ; }
int main ( ) {
int N = 8 ;
cout << MaximumSides ( N ) ; return 0 ; }
float pairProductMean ( int arr [ ] , int N ) {
int suffixSumArray [ N ] ; suffixSumArray [ N - 1 ] = arr [ N - 1 ] ;
for ( int i = N - 2 ; i >= 0 ; i -- ) { suffixSumArray [ i ] = suffixSumArray [ i + 1 ] + arr [ i ] ; }
int length = ( N * ( N - 1 ) ) / 2 ;
float res = 0 ; for ( int i = 0 ; i < N - 1 ; i ++ ) { res += arr [ i ] * suffixSumArray [ i + 1 ] ; }
float mean ;
if ( length != 0 ) mean = res / length ; else mean = 0 ;
return mean ; }
int main ( ) {
int arr [ ] = { 1 , 2 , 4 , 8 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << fixed << setprecision ( 2 ) << pairProductMean ( arr , N ) ; return 0 ; }
int ncr ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
int countPath ( int N , int M , int K ) { int answer ; if ( K >= 2 ) answer = 0 ; else if ( K == 0 ) answer = ncr ( N + M - 2 , N - 1 ) ; else {
answer = ncr ( N + M - 2 , N - 1 ) ;
int X = ( N - 1 ) / 2 + ( M - 1 ) / 2 ; int Y = ( N - 1 ) / 2 ; int midCount = ncr ( X , Y ) ;
X = ( ( N - 1 ) - ( N - 1 ) / 2 ) + ( ( M - 1 ) - ( M - 1 ) / 2 ) ; Y = ( ( N - 1 ) - ( N - 1 ) / 2 ) ; midCount *= ncr ( X , Y ) ; answer -= midCount ; } return answer ; }
int main ( ) { int N = 3 ; int M = 3 ; int K = 1 ; cout << countPath ( N , M , K ) ; return 0 ; }
int find_max ( vector < pair < int , int > > v , int n ) {
int count = 0 ; if ( n >= 2 ) count = 2 ; else count = 1 ;
for ( int i = 1 ; i < n - 1 ; i ++ ) {
if ( v [ i - 1 ] . first < ( v [ i ] . first - v [ i ] . second ) ) count ++ ;
else if ( v [ i + 1 ] . first > ( v [ i ] . first + v [ i ] . second ) ) { count ++ ; v [ i ] . first = v [ i ] . first + v [ i ] . second ; }
else continue ; }
return count ; }
int main ( ) { int n = 3 ; vector < pair < int , int > > v ; v . push_back ( { 10 , 20 } ) ; v . push_back ( { 15 , 10 } ) ; v . push_back ( { 20 , 16 } ) ; cout << find_max ( v , n ) ; return 0 ; }
void numberofsubstrings ( string str , int k , char charArray [ ] ) { int N = str . length ( ) ;
bool available [ 26 ] = { 0 } ;
for ( int i = 0 ; i < k ; i ++ ) { available [ charArray [ i ] - ' a ' ] = 1 ; }
int lastPos = -1 ;
int ans = ( N * ( N + 1 ) ) / 2 ;
for ( int i = 0 ; i < N ; i ++ ) {
if ( available [ str [ i ] - ' a ' ] == 0 ) {
ans -= ( ( i - lastPos ) * ( N - i ) ) ;
lastPos = i ; } }
cout << ans << endl ; }
string str = " abcb " ; int k = 2 ;
char charArray [ k ] = { ' a ' , ' b ' } ;
numberofsubstrings ( str , k , charArray ) ; return 0 ; }
int minCost ( int N , int P , int Q ) {
int cost = 0 ;
while ( N > 0 ) { if ( N & 1 ) { cost += P ; N -- ; } else { int temp = N / 2 ;
if ( temp * P > Q ) cost += Q ;
else cost += P * temp ; N /= 2 ; } }
return cost ; }
void numberOfWays ( int n , int k ) {
int dp [ 1000 ] ;
dp [ 0 ] = 1 ;
for ( int i = 1 ; i <= k ; i ++ ) {
int numWays = 0 ;
for ( int j = 0 ; j < n ; j ++ ) { numWays += dp [ j ] ; }
for ( int j = 0 ; j < n ; j ++ ) { dp [ j ] = numWays - dp [ j ] ; } }
cout << dp [ 0 ] << endl ; }
int main ( ) {
int N = 5 , K = 3 ;
numberOfWays ( N , K ) ; return 0 ; }
int findMinCost ( pair < int , int > arr [ ] , int X , int n , int i = 0 ) {
if ( X <= 0 ) return 0 ; if ( i >= n ) return INT_MAX ;
int inc = findMinCost ( arr , X - arr [ i ] . first , n , i + 1 ) ; if ( inc != INT_MAX ) inc += arr [ i ] . second ;
int exc = findMinCost ( arr , X , n , i + 1 ) ;
return min ( inc , exc ) ; }
int main ( ) {
pair < int , int > arr [ ] = { { 4 , 3 } , { 3 , 2 } , { 2 , 4 } , { 1 , 3 } , { 4 , 2 } } ; int X = 7 ;
int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int ans = findMinCost ( arr , X , n ) ;
if ( ans != INT_MAX ) cout << ans ; else cout << -1 ; return 0 ; }
long double find ( int N , int sum ) {
if ( sum > 6 * N sum < N ) return 0 ; if ( N == 1 ) { if ( sum >= 1 && sum <= 6 ) return 1.0 / 6 ; else return 0 ; } long double s = 0 ; for ( int i = 1 ; i <= 6 ; i ++ ) s = s + find ( N - 1 , sum - i ) / 6 ; return s ; }
int main ( ) { int N = 4 , a = 13 , b = 17 ; long double probability = 0.0 ; for ( int sum = a ; sum <= b ; sum ++ ) probability = probability + find ( N , sum ) ;
cout << fixed << setprecision ( 6 ) << probability ; return 0 ; }
int minDays ( int n ) {
if ( n < 1 ) return n ;
int cnt = 1 + min ( n % 2 + minDays ( n / 2 ) , n % 3 + minDays ( n / 3 ) ) ;
return cnt ; }
int N = 6 ;
cout << minDays ( N ) ; return 0 ; }
