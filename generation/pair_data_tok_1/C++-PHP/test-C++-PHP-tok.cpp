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
int countWays ( int N ) {
int E = ( N * ( N - 1 ) ) / 2 ; if ( N == 1 ) return 0 ; return pow ( 2 , E - 1 ) ; }
int main ( ) { int N = 4 ; cout << countWays ( N ) ; return 0 ; }
int minAbsDiff ( int n ) { int mod = n % 4 ; if ( mod == 0 mod == 3 ) return 0 ; return 1 ; }
int main ( ) { int n = 5 ; cout << minAbsDiff ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool check ( int s ) {
int freq [ 10 ] = { 0 } , r ; while ( s != 0 ) {
r = s % 10 ;
s = int ( s / 10 ) ;
freq [ r ] += 1 ; } int xor__ = 0 ;
for ( int i = 0 ; i < 10 ; i ++ ) { xor__ = xor__ ^ freq [ i ] ; if ( xor__ == 0 ) return true ; else return false ; } }
int main ( ) { int s = 122233 ; if ( check ( s ) ) cout << " Yes " << endl ; else cout << " No " << endl ; }
void printLines ( int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) { cout << k * ( 6 * i + 1 ) << " ▁ " << k * ( 6 * i + 2 ) << " ▁ " << k * ( 6 * i + 3 ) << " ▁ " << k * ( 6 * i + 5 ) << endl ; } }
int main ( ) { int n = 2 , k = 2 ; printLines ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int calculateSum ( int n ) {
return ( pow ( 2 , n + 1 ) + n - 2 ) ; }
int n = 4 ;
cout << " Sum ▁ = ▁ " << calculateSum ( n ) ; return 0 ; }
long long partitions ( int n ) { vector < long long > p ( n + 1 , 0 ) ;
p [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; ++ i ) { int k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 ? 1 : -1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) k *= -1 ; else k = 1 - k ; } } return p [ n ] ; }
int main ( ) { int N = 20 ; cout << partitions ( N ) ; return 0 ; }
int countPaths ( int n , int m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
int main ( ) { int n = 3 , m = 2 ; cout << " ▁ Number ▁ of ▁ Paths ▁ " << countPaths ( n , m ) ; return 0 ; }
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
int countChar ( string str , char x ) { int count = 0 , n = 10 ; for ( int i = 0 ; i < str . size ( ) ; i ++ ) if ( str [ i ] == x ) count ++ ;
int repetitions = n / str . size ( ) ; count = count * repetitions ;
for ( int i = 0 ; i < n % str . size ( ) ; i ++ ) { if ( str [ i ] == x ) count ++ ; } return count ; }
int main ( ) { string str = " abcac " ; cout << countChar ( str , ' a ' ) ; return 0 ; }
bool check ( string s , int m ) {
int l = s . length ( ) ;
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( s [ i ] == '0' ) { c2 = 0 ;
c1 ++ ; } else { c1 = 0 ;
c2 ++ ; } if ( c1 == m c2 == m ) return true ; } return false ; }
int main ( ) { string s = "001001" ; int m = 2 ;
if ( check ( s , m ) ) cout << " YES " ; else cout << " NO " ; return 0 ; }
int productAtKthLevel ( string tree , int k ) { int level = -1 ;
int n = tree . length ( ) ; for ( int i = 0 ; i < n ; i ++ ) {
if ( tree [ i ] == ' ( ' ) level ++ ;
else if ( tree [ i ] == ' ) ' ) level -- ; else {
if ( level == k ) product *= ( tree [ i ] - '0' ) ; } }
return product ; }
int main ( ) { string tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; cout << productAtKthLevel ( tree , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isValidISBN ( string & isbn ) {
int n = isbn . length ( ) ; if ( n != 10 ) return false ;
int sum = 0 ; for ( int i = 0 ; i < 9 ; i ++ ) { int digit = isbn [ i ] - '0' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
char last = isbn [ 9 ] ; if ( last != ' X ' && ( last < '0' last > '9' ) ) return false ;
sum += ( ( last == ' X ' ) ? 10 : ( last - '0' ) ) ;
return ( sum % 11 == 0 ) ; }
int main ( ) { string isbn = "007462542X " ; if ( isValidISBN ( isbn ) ) cout << " Valid " ; else cout << " Invalid " ; return 0 ; }
int main ( ) { int d = 10 ; double a ;
a = ( double ) ( 360 - ( 6 * d ) ) / 4 ;
cout << a << " , ▁ " << a + d << " , ▁ " << a + ( 2 * d ) << " , ▁ " << a + ( 3 * d ) << endl ; return 0 ; }
void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = fabs ( ( c2 * z1 + d2 ) ) / ( sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; cout << " Perpendicular ▁ distance ▁ is ▁ " << d << endl ; } else cout << " Planes ▁ are ▁ not ▁ parallel " ; return ; }
int main ( ) { float a1 = 1 ; float b1 = 2 ; float c1 = -1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = -3 ; float d2 = -4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ; return 0 ; }
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
bool possibleToReach ( int a , int b ) {
int c = cbrt ( a * b ) ;
int re1 = a / c ; int re2 = b / c ;
if ( ( re1 * re1 * re2 == a ) && ( re2 * re2 * re1 == b ) ) return true ; else return false ; }
int main ( ) { int A = 60 , B = 450 ; if ( possibleToReach ( A , B ) ) cout << " yes " ; else cout << " no " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isUndulating ( string n ) {
if ( n . length ( ) <= 2 ) return false ;
for ( int i = 2 ; i < n . length ( ) ; i ++ ) if ( n [ i - 2 ] != n [ i ] ) false ; return true ; }
int main ( ) { string n = "1212121" ; if ( isUndulating ( n ) ) cout << " Yes " ; else cout << " No " ; }
int Series ( int n ) { int i ; int sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
int main ( ) { int n = 3 ; int res = Series ( n ) ; cout << res << endl ; }
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
int horner ( int poly [ ] , int n , int x ) {
for ( int i = 1 ; i < n ; i ++ ) result = result * x + poly [ i ] ; return result ; }
int findSign ( int poly [ ] , int n , int x ) { int result = horner ( poly , n , x ) ; if ( result > 0 ) return 1 ; else if ( result < 0 ) return -1 ; return 0 ; }
int poly [ ] = { 2 , -6 , 2 , -1 } ; int x = 3 ; int n = sizeof ( poly ) / sizeof ( poly [ 0 ] ) ; cout << " Sign ▁ of ▁ polynomial ▁ is ▁ " << findSign ( poly , n , x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100005
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
long int SubArraySum ( int arr [ ] , int n ) { long int result = 0 , temp = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
temp = 0 ; for ( int j = i ; j < n ; j ++ ) {
temp += arr [ j ] ; result += temp ; } } return result ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Sum ▁ of ▁ SubArray ▁ : ▁ " << SubArraySum ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int highestPowerof2 ( int n ) { int p = ( int ) log2 ( n ) ; return ( int ) pow ( 2 , p ) ; }
int main ( ) { int n = 10 ; cout << highestPowerof2 ( n ) ; return 0 ; }
unsigned int aModM ( string s , unsigned int mod ) { unsigned int number = 0 ; for ( unsigned int i = 0 ; i < s . length ( ) ; i ++ ) {
number = ( number * 10 + ( s [ i ] - '0' ) ) ; number %= mod ; } return number ; }
unsigned int ApowBmodM ( string & a , unsigned int b , unsigned int m ) {
unsigned int ans = aModM ( a , m ) ; unsigned int mul = ans ;
for ( unsigned int i = 1 ; i < b ; i ++ ) ans = ( ans * mul ) % m ; return ans ; }
int main ( ) { string a = "987584345091051645734583954832576" ; unsigned int b = 3 , m = 11 ; cout << ApowBmodM ( a , b , m ) ; return 0 ; }
int SieveOfSundaram ( int n ) {
int nNew = ( n - 1 ) / 2 ;
memset ( marked , false , sizeof ( marked ) ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) for ( int j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) cout << 2 << " ▁ " ;
for ( int i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) cout << 2 * i + 1 << " ▁ " ; }
int main ( void ) { int n = 20 ; SieveOfSundaram ( n ) ; return 0 ; }
int hammingDistance ( int n1 , int n2 ) { int x = n1 ^ n2 ; int setBits = 0 ; while ( x > 0 ) { setBits += x & 1 ; x >>= 1 ; } return setBits ; }
int main ( ) { int n1 = 9 , n2 = 14 ; cout << hammingDistance ( 9 , 14 ) << endl ; return 0 ; }
void printSubsets ( int n ) { for ( int i = 0 ; i <= n ; i ++ ) if ( ( n & i ) == i ) cout << i << " ▁ " ; }
int main ( ) { int n = 9 ; printSubsets ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int setBitNumber ( int n ) {
int k = ( int ) ( log2 ( n ) ) ;
return 1 << k ; }
int main ( ) { int n = 273 ; cout << setBitNumber ( n ) ; return 0 ; }
int subset ( int ar [ ] , int n ) {
int res = 0 ;
sort ( ar , ar + n ) ;
for ( int i = 0 ; i < n ; i ++ ) { int count = 1 ;
for ( ; i < n - 1 ; i ++ ) { if ( ar [ i ] == ar [ i + 1 ] ) count ++ ; else break ; }
res = max ( res , count ) ; } return res ; }
int main ( ) { int arr [ ] = { 5 , 6 , 9 , 3 , 4 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << subset ( arr , n ) ; return 0 ; }
int findLargestd ( int S [ ] , int n ) { bool found = false ;
sort ( S , S + n ) ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { for ( int j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( int k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( int l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return INT_MIN ; }
int main ( ) { int S [ ] = { 2 , 3 , 5 , 7 , 12 } ; int n = sizeof ( S ) / sizeof ( S [ 0 ] ) ; int ans = findLargestd ( S , n ) ; if ( ans == INT_MIN ) cout << " No ▁ Solution " << endl ; else cout << " Largest ▁ d ▁ such ▁ that ▁ a ▁ + ▁ b ▁ + ▁ " << " c ▁ = ▁ d ▁ is ▁ " << ans << endl ; return 0 ; }
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
void pushZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
int main ( ) { int arr [ ] = { 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; pushZerosToEnd ( arr , n ) ; cout << " Array ▁ after ▁ pushing ▁ all ▁ zeros ▁ to ▁ end ▁ of ▁ array ▁ : STRNEWLINE " ; for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
void RearrangePosNeg ( int arr [ ] , int n ) { int key , j ; for ( int i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
int main ( ) { int arr [ ] = { -12 , 11 , -13 , -5 , 6 , -7 , 5 , -3 , -6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findElements ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) cout << arr [ i ] << " ▁ " ; } }
int main ( ) { int arr [ ] = { 2 , -6 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findElements ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findElements ( int arr [ ] , int n ) { sort ( arr , arr + n ) ; for ( int i = 0 ; i < n - 2 ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 2 , -6 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findElements ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findElements ( int arr [ ] , int n ) { int first = INT_MIN , second = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 2 , -6 , 3 , 5 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findElements ( arr , n ) ; return 0 ; }
int findFirstMissing ( int array [ ] , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
int main ( ) { int arr [ ] = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Smallest ▁ missing ▁ element ▁ is ▁ " << findFirstMissing ( arr , 0 , n - 1 ) << endl ; }
int FindMaxSum ( vector < int > arr , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
int main ( ) { vector < int > arr = { 5 , 5 , 10 , 100 , 10 , 5 } ; cout << FindMaxSum ( arr , arr . size ( ) ) ; }
int findMaxAverage ( int arr [ ] , int n , int k ) {
if ( k > n ) return -1 ;
int * csum = new int [ n ] ; csum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) csum [ i ] = csum [ i - 1 ] + arr [ i ] ;
int max_sum = csum [ k - 1 ] , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { int curr_sum = csum [ i ] - csum [ i - k ] ; if ( curr_sum > max_sum ) { max_sum = curr_sum ; max_end = i ; } }
return max_end - k + 1 ; }
int main ( ) { int arr [ ] = { 1 , 12 , -5 , -6 , 50 , 3 } ; int k = 4 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " " length ▁ " << k << " ▁ begins ▁ at ▁ index ▁ " << findMaxAverage ( arr , n , k ) ; return 0 ; }
int findMaxAverage ( int arr [ ] , int n , int k ) {
if ( k > n ) return -1 ;
int sum = arr [ 0 ] ; for ( int i = 1 ; i < k ; i ++ ) sum += arr [ i ] ; int max_sum = sum , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { int sum = sum + arr [ i ] - arr [ i - k ] ; if ( sum > max_sum ) { max_sum = sum ; max_end = i ; } }
return max_end - k + 1 ; }
int main ( ) { int arr [ ] = { 1 , 12 , -5 , -6 , 50 , 3 } ; int k = 4 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " " length ▁ " << k << " ▁ begins ▁ at ▁ index ▁ " << findMaxAverage ( arr , n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isMajority ( int arr [ ] , int n , int x ) { int i ;
int last_index = n % 2 ? ( n / 2 + 1 ) : ( n / 2 ) ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + n / 2 ] == x ) return 1 ; } return 0 ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 4 , 4 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 4 ; if ( isMajority ( arr , n , x ) ) cout << x << " ▁ appears ▁ more ▁ than ▁ " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; else cout << x << " ▁ does ▁ not ▁ appear ▁ more ▁ than " << n / 2 << " ▁ times ▁ in ▁ arr [ ] " << endl ; return 0 ; }
int cutRod ( int price [ ] , int n ) { int val [ n + 1 ] ; val [ 0 ] = 0 ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { int max_val = INT_MIN ; for ( j = 0 ; j < i ; j ++ ) max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " << cutRod ( arr , size ) ; getchar ( ) ; return 0 ; }
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
int minValue ( int n , int x , int y ) {
float val = ( y * n ) / 100 ;
if ( x >= val ) return 0 ; else return ( ceil ( val ) - x ) ; }
int main ( ) { int n = 10 , x = 2 , y = 40 ; cout << minValue ( n , x , y ) ; }
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
bool isFactorialPrime ( long n ) {
if ( ! isPrime ( n ) ) return false ; long fact = 1 ; int i = 1 ; while ( fact <= n + 1 ) {
fact = fact * i ;
if ( n + 1 == fact n - 1 == fact ) return true ; i ++ ; }
return false ; }
int main ( ) { int n = 23 ; if ( isFactorialPrime ( n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
ll n = 5 ;
ll fac1 = 1 ; for ( int i = 2 ; i <= n - 1 ; i ++ ) fac1 = fac1 * i ;
ll fac2 = fac1 * n ;
ll totalWays = fac1 * fac2 ;
cout << totalWays << endl ; return 0 ; }
int nextPerfectCube ( int N ) { int nextN = floor ( cbrt ( N ) ) + 1 ; return nextN * nextN * nextN ; }
int main ( ) { int n = 35 ; cout << nextPerfectCube ( n ) ; return 0 ; }
#include <algorithm> NEW_LINE #include <iostream> NEW_LINE using namespace std ; int findpos ( string n ) { int pos = 0 ; for ( int i = 0 ; n [ i ] != ' \0' ; i ++ ) { switch ( n [ i ] ) {
case '2' : pos = pos * 4 + 1 ; break ;
case '3' : pos = pos * 4 + 2 ; break ;
case '5' : pos = pos * 4 + 3 ; break ;
case '7' : pos = pos * 4 + 4 ; break ; } } return pos ; }
int main ( ) { string n = "777" ; cout << findpos ( n ) ; }
#include <bits/stdc++.h> NEW_LINE #define mod  1000000007 NEW_LINE using namespace std ;
long long digitNumber ( long long n ) {
if ( n == 0 ) return 1 ;
if ( n == 1 ) return 9 ;
if ( n % 2 ) {
long long temp = digitNumber ( ( n - 1 ) / 2 ) % mod ; return ( 9 * ( temp * temp ) % mod ) % mod ; } else {
long long temp = digitNumber ( n / 2 ) % mod ; return ( temp * temp ) % mod ; } } int countExcluding ( int n , int d ) {
if ( d == 0 ) return ( 9 * digitNumber ( n - 1 ) ) % mod ; else return ( 8 * digitNumber ( n - 1 ) ) % mod ; }
long long d = 9 ; int n = 3 ; cout << countExcluding ( n , d ) << endl ; return 0 ; }
bool isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
bool isEmirp ( int n ) {
if ( isPrime ( n ) == false ) return false ;
int rev = 0 ; while ( n != 0 ) { int d = n % 10 ; rev = rev * 10 + d ; n /= 10 ; }
return isPrime ( rev ) ; }
int n = 13 ; if ( isEmirp ( n ) == true ) cout << " Yes " ; else cout << " No " ; }
double Convert ( double radian ) { double pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
int main ( ) { double radian = 5.0 ; double degree = Convert ( radian ) ; cout << degree ; return 0 ; }
int sn ( int n , int an ) { return ( n * ( 1 + an ) ) / 2 ; }
int trace ( int n , int m ) {
int an = 1 + ( n - 1 ) * ( m + 1 ) ;
int rowmajorSum = sn ( n , an ) ;
an = 1 + ( n - 1 ) * ( n + 1 ) ;
int colmajorSum = sn ( n , an ) ; return rowmajorSum + colmajorSum ; }
int main ( ) { int N = 3 , M = 3 ; cout << trace ( N , M ) << endl ; return 0 ; }
void max_area ( int n , int m , int k ) { if ( k > ( n + m - 2 ) ) cout << " Not ▁ possible " << endl ; else { int result ;
if ( k < max ( m , n ) - 1 ) { result = max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; }
else { result = max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; }
cout << result << endl ; } }
int main ( ) { int n = 3 , m = 4 , k = 1 ; max_area ( n , m , k ) ; }
int area_fun ( int side ) { int area = side * side ; return area ; }
int main ( ) { int side = 4 ; int area = area_fun ( side ) ; cout << area ; return 0 ; }
long int countConsecutive ( long int N ) {
long int count = 0 ; for ( long int L = 1 ; L * ( L + 1 ) < 2 * N ; L ++ ) { double a = ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) ; if ( a - ( int ) a == 0.0 ) count ++ ; } return count ; }
int main ( ) { long int N = 15 ; cout << countConsecutive ( N ) << endl ; N = 10 ; cout << countConsecutive ( N ) << endl ; return 0 ; }
bool isAutomorphic ( int N ) {
int sq = N * N ;
while ( N > 0 ) {
if ( N % 10 != sq % 10 ) return false ;
N /= 10 ; sq /= 10 ; } return true ; }
int main ( ) { int N = 5 ; isAutomorphic ( N ) ? cout << " Automorphic " : cout << " Not ▁ Automorphic " ; return 0 ; }
int maxPrimefactorNum ( int N ) {
bool arr [ N + 5 ] ; memset ( arr , true , sizeof ( arr ) ) ;
for ( int i = 3 ; i * i <= N ; i += 2 ) { if ( arr [ i ] ) for ( int j = i * i ; j <= N ; j += i ) arr [ j ] = false ; }
vector < int > prime ; prime . push_back ( 2 ) ; for ( int i = 3 ; i <= N ; i += 2 ) if ( arr [ i ] ) prime . push_back ( i ) ;
int i = 0 , ans = 1 ; while ( ans * prime [ i ] <= N && i < prime . size ( ) ) { ans *= prime [ i ] ; i ++ ; } return ans ; }
int main ( ) { int N = 40 ; cout << maxPrimefactorNum ( N ) << endl ; return 0 ; }
int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; }
int main ( ) { int num = 36 ; cout << divSum ( num ) ; return 0 ; }
int power ( int x , int y , int p ) {
while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
void squareRoot ( int n , int p ) { if ( p % 4 != 3 ) { cout << " Invalid ▁ Input " ; return ; }
n = n % p ; int x = power ( n , ( p + 1 ) / 4 , p ) ; if ( ( x * x ) % p == n ) { cout << " Square ▁ root ▁ is ▁ " << x ; return ; }
x = p - x ; if ( ( x * x ) % p == n ) { cout << " Square ▁ root ▁ is ▁ " << x ; return ; }
cout << " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ; }
int main ( ) { int p = 7 ; int n = 2 ; squareRoot ( n , p ) ; return 0 ; }
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
int main ( ) { int N = 6 ; int Even = N / 2 ; int Odd = N - Even ; cout << Even * Odd ; return 0 ; }
void steps ( string str , int n ) {
bool flag ; int x = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
if ( x == 0 ) flag = true ;
if ( x == n - 1 ) flag = false ;
for ( int j = 0 ; j < x ; j ++ ) cout << " * " ; cout << str [ i ] << " STRNEWLINE " ;
if ( flag == true ) x ++ ; else x -- ; } }
int n = 4 ; string str = " GeeksForGeeks " ; cout << " String : ▁ " << str << endl ; cout << " Max ▁ Length ▁ of ▁ Steps : ▁ " << n << endl ;
steps ( str , n ) ; return 0 ; }
bool isDivisible ( char str [ ] , int k ) { int n = strlen ( str ) ; int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) if ( str [ n - i - 1 ] == '0' ) c ++ ;
return ( c == k ) ; }
char str1 [ ] = "10101100" ; int k = 2 ; if ( isDivisible ( str1 , k ) ) cout << " Yes " << endl ; else cout << " No " << " STRNEWLINE " ;
char str2 [ ] = "111010100" ; k = 2 ; if ( isDivisible ( str2 , k ) ) cout << " Yes " << endl ; else cout << " No " << endl ; return 0 ; }
bool isNumber ( string s ) { for ( int i = 0 ; i < s . length ( ) ; i ++ ) if ( isdigit ( s [ i ] ) == false ) return false ; return true ; }
string str = "6790" ;
if ( isNumber ( str ) ) cout << " Integer " ;
else cout < < " String " ; }
void reverse ( string str ) { if ( str . size ( ) == 0 ) { return ; } reverse ( str . substr ( 1 ) ) ; cout << str [ 0 ] ; }
int main ( ) { string a = " Geeks ▁ for ▁ Geeks " ; reverse ( a ) ; return 0 ; }
float polyarea ( float n , float r ) {
if ( r < 0 && n < 0 ) return -1 ;
float A = ( ( r * r * n ) * sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
int main ( ) { float r = 9 , n = 6 ; cout << polyarea ( n , r ) << endl ; return 0 ; }
double findPCSlope ( double m ) { return -1.0 / m ; }
int main ( ) { double m = 2.0 ; cout << findPCSlope ( m ) ; return 0 ; }
float area_of_segment ( float radius , float angle ) {
float area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
float area_of_triangle = ( float ) 1 / 2 * ( radius * radius ) * sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
int main ( ) { float radius = 10.0 , angle = 90.0 ; cout << " Area ▁ of ▁ minor ▁ segment ▁ = ▁ " << area_of_segment ( radius , angle ) << endl ; cout << " Area ▁ of ▁ major ▁ segment ▁ = ▁ " << area_of_segment ( radius , ( 360 - angle ) ) ; }
#include <iostream> NEW_LINE using namespace std ; void SectorArea ( double radius , double angle ) { if ( angle >= 360 ) cout << " Angle ▁ not ▁ possible " ;
else { double sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; cout << sector ; } }
int main ( ) { double radius = 9 ; double angle = 60 ; SectorArea ( radius , angle ) ; return 0 ; }
void insertionSortRecursive ( int arr [ ] , int n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
int last = arr [ n - 1 ] ; int j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; insertionSortRecursive ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
bool isWaveArray ( int arr [ ] , int n ) { bool result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
int arr [ ] = { 1 , 3 , 2 , 4 } ; int n = sizeof ( arr ) / sizeof ( int ) ; if ( isWaveArray ( arr , n ) ) { cout << " YES " << endl ; } else { cout << " NO " << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define mod  1000000007
long long sumOddFibonacci ( int n ) { long long Sum [ n + 1 ] ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( int i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
int main ( ) { long long n = 6 ; cout << sumOddFibonacci ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int solve ( int N , int K ) {
int combo [ N + 1 ] = { 0 } ;
combo [ 0 ] = 1 ;
for ( int i = 1 ; i <= K ; i ++ ) {
for ( int j = 0 ; j <= N ; j ++ ) {
if ( j >= i ) {
combo [ j ] += combo [ j - i ] ; } } }
return combo [ N ] ; }
int N = 29 ; int K = 5 ; cout << solve ( N , K ) ; solve ( N , K ) ; return 0 ; }
int computeLIS ( int circBuff [ ] , int start , int end , int n ) { int LIS [ end - start ] ;
for ( int i = start ; i < end ; i ++ ) LIS [ i ] = 1 ;
for ( int i = start + 1 ; i < end ; i ++ )
for ( int j = start ; j < i ; j ++ ) if ( circBuff [ i ] > circBuff [ j ] && LIS [ i ] < LIS [ j ] + 1 ) LIS [ i ] = LIS [ j ] + 1 ;
int res = INT_MIN ; for ( int i = start ; i < end ; i ++ ) res = max ( res , LIS [ i ] ) ; return res ; }
int LICS ( int arr [ ] , int n ) {
int circBuff [ 2 * n ] ; for ( int i = 0 ; i < n ; i ++ ) circBuff [ i ] = arr [ i ] ; for ( int i = n ; i < 2 * n ; i ++ ) circBuff [ i ] = arr [ i - n ] ;
int res = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) res = max ( computeLIS ( circBuff , i , i + n , n ) , res ) ; return res ; }
int main ( ) { int arr [ ] = { 1 , 4 , 6 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Length ▁ of ▁ LICS ▁ is ▁ " << LICS ( arr , n ) ; return 0 ; }
int LCIS ( int arr1 [ ] , int n , int arr2 [ ] , int m ) {
int table [ m ] ; for ( int j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int current = 0 ;
for ( int j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
int result = 0 ; for ( int i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
int main ( ) { int arr1 [ ] = { 3 , 4 , 9 , 1 } ; int arr2 [ ] = { 5 , 3 , 8 , 9 , 10 , 2 , 1 } ; int n = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; int m = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; cout << " Length ▁ of ▁ LCIS ▁ is ▁ " << LCIS ( arr1 , n , arr2 , m ) ; return ( 0 ) ; }
string maxValue ( string a , string b ) {
sort ( b . begin ( ) , b . end ( ) ) ; int n = a . length ( ) ; int m = b . length ( ) ;
int j = m - 1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( j < 0 ) break ; if ( b [ j ] > a [ i ] ) { a [ i ] = b [ j ] ;
j -- ; } }
return a ; }
int main ( ) { string a = "1234" ; string b = "4321" ; cout << maxValue ( a , b ) ; return 0 ; }
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
bool is_possible ( string s ) {
int l = s . length ( ) ; int one = 0 , zero = 0 ; for ( int i = 0 ; i < l ; i ++ ) {
if ( s [ i ] == '0' ) zero ++ ;
else one ++ ; }
if ( l % 2 == 0 ) return ( one == zero ) ;
else return ( abs ( one - zero ) == 1 ) ; }
int main ( ) { string s = "100110" ; if ( is_possible ( s ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
string convert ( string s ) { int n = s . length ( ) ; s [ 0 ] = tolower ( s [ 0 ] ) ; for ( int i = 1 ; i < n ; i ++ ) {
if ( s [ i ] == ' ▁ ' && i < n ) {
s [ i + 1 ] = tolower ( s [ i + 1 ] ) ; i ++ ; }
else s [ i ] = toupper ( s [ i ] ) ; }
return s ; }
int main ( ) { string str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " ; cout << convert ( str ) ; return 0 ; }
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
#include <bits/stdc++.h> NEW_LINE using namespace std ; string findNthNo ( int n ) { string res = " " ; while ( n >= 1 ) {
if ( n & 1 ) { res = res + "3" ; n = ( n - 1 ) / 2 ; }
else { res = res + "5" ; n = ( n - 2 ) / 2 ; } }
reverse ( res . begin ( ) , res . end ( ) ) ; return res ; }
int main ( ) { int n = 5 ; cout << findNthNo ( n ) ; return 0 ; }
int findNthNonSquare ( int n ) {
long double x = ( long double ) n ;
long double ans = x + floor ( 0.5 + sqrt ( x ) ) ; return ( int ) ans ; }
int n = 16 ;
cout << " The ▁ " << n << " th ▁ Non - Square ▁ number ▁ is ▁ " ; cout << findNthNonSquare ( n ) ; return 0 ; }
int seiresSum ( int n , int a [ ] ) { return n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ; }
int main ( ) { int n = 2 ; int a [ ] = { 1 , 2 , 3 , 4 } ; cout << seiresSum ( n , a ) ; return 0 ; }
int checkdigit ( int n , int k ) { while ( n ) {
int rem = n % 10 ;
if ( rem == k ) return 1 ; n = n / 10 ; } return 0 ; }
int findNthNumber ( int n , int k ) {
for ( int i = k + 1 , count = 1 ; count < n ; i ++ ) {
if ( checkdigit ( i , k ) || ( i % k == 0 ) ) count ++ ; if ( count == n ) return i ; } return -1 ; }
int main ( ) { int n = 10 , k = 2 ; cout << findNthNumber ( n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int middleOfThree ( int a , int b , int c ) {
int middleOfThree ( int a , int b , int c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && a < c ) || ( c < a && a < b ) ) return a ; else return c ; }
int main ( ) { int a = 20 , b = 30 , c = 40 ; cout << middleOfThree ( a , b , c ) ; return 0 ; }
#include <iostream> NEW_LINE #include <climits> NEW_LINE using namespace std ; #define INF  INT_MAX NEW_LINE #define N  4
int minCost ( int cost [ ] [ N ] ) {
int dist [ N ] ; for ( int i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) dist [ j ] = dist [ i ] + cost [ i ] [ j ] ; return dist [ N - 1 ] ; }
int main ( ) { int cost [ N ] [ N ] = { { 0 , 15 , 80 , 90 } , { INF , 0 , 40 , 50 } , { INF , INF , 0 , 70 } , { INF , INF , INF , 0 } } ; cout << " The ▁ Minimum ▁ cost ▁ to ▁ reach ▁ station ▁ " << N << " ▁ is ▁ " << minCost ( cost ) ; return 0 ; }
int numOfways ( int n , int k ) { int p = 1 ; if ( k % 2 ) p = -1 ; return ( pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
int main ( ) { int n = 4 , k = 2 ; cout << numOfways ( n , k ) << endl ; return 0 ; }
void length_of_chord ( double r , double x ) { cout << " The ▁ length ▁ of ▁ the ▁ chord " << " ▁ of ▁ the ▁ circle ▁ is ▁ " << 2 * r * sin ( x * ( 3.14 / 180 ) ) << endl ; }
int main ( ) { double r = 4 , x = 63 ; length_of_chord ( r , x ) ; return 0 ; }
float area ( float a ) {
if ( a < 0 ) return -1 ;
float area = sqrt ( a ) / 6 ; return area ; }
int main ( ) { float a = 10 ; cout << area ( a ) << endl ; return 0 ; }
double longestRodInCuboid ( int length , int breadth , int height ) { double result ; int temp ;
temp = length * length + breadth * breadth + height * height ;
result = sqrt ( temp ) ; return result ; }
int main ( ) { int length = 12 , breadth = 9 , height = 8 ;
cout << longestRodInCuboid ( length , breadth , height ) ; return 0 ; }
bool LiesInsieRectangle ( int a , int b , int x , int y ) { if ( x - y - b <= 0 && x - y + b >= 0 && x + y - 2 * a + b <= 0 && x + y - b >= 0 ) return true ; return false ; }
int main ( ) { int a = 7 , b = 2 , x = 4 , y = 5 ; if ( LiesInsieRectangle ( a , b , x , y ) ) cout << " Given ▁ point ▁ lies ▁ inside ▁ the ▁ rectangle " ; else cout << " Given ▁ point ▁ does ▁ not ▁ lie ▁ on ▁ the ▁ rectangle " ; return 0 ; }
int maxvolume ( int s ) { int maxvalue = 0 ;
for ( int i = 1 ; i <= s - 2 ; i ++ ) {
for ( int j = 1 ; j <= s - 1 ; j ++ ) {
int k = s - i - j ;
maxvalue = max ( maxvalue , i * j * k ) ; } } return maxvalue ; }
int main ( ) { int s = 8 ; cout << maxvolume ( s ) << endl ; return 0 ; }
int maxvolume ( int s ) {
int length = s / 3 ; s -= length ;
int breadth = s / 2 ;
int height = s - breadth ; return length * breadth * height ; }
int main ( ) { int s = 8 ; cout << maxvolume ( s ) << endl ; return 0 ; }
double hexagonArea ( double s ) { return ( ( 3 * sqrt ( 3 ) * ( s * s ) ) / 2 ) ; }
double s = 4 ; cout << " Area ▁ : ▁ " << hexagonArea ( s ) ; return 0 ; }
int maxSquare ( int b , int m ) {
return ( b / m - 1 ) * ( b / m ) / 2 ; }
int main ( ) { int b = 10 , m = 2 ; cout << maxSquare ( b , m ) ; return 0 ; }
void findRightAngle ( int A , int H ) {
long D = pow ( H , 4 ) - 16 * A * A ; if ( D >= 0 ) {
long root1 = ( H * H + sqrt ( D ) ) / 2 ; long root2 = ( H * H - sqrt ( D ) ) / 2 ; long a = sqrt ( root1 ) ; long b = sqrt ( root2 ) ; if ( b >= a ) cout << a << " ▁ " << b << " ▁ " << H ; else cout << b << " ▁ " << a << " ▁ " << H ; } else cout << " - 1" ; }
int main ( ) { findRightAngle ( 6 , 5 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int numberOfSquares ( int base ) {
base = ( base - 2 ) ;
base = floor ( base / 2 ) ; return base * ( base + 1 ) / 2 ; }
int main ( ) { int base = 8 ; cout << numberOfSquares ( base ) ; return 0 ; }
int fib ( int n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
int findVertices ( int n ) {
return fib ( n + 2 ) ; }
int main ( ) { int n = 3 ; cout << findVertices ( n ) ; return 0 ; }
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
const int m = 3 ;
const int n = 2 ;
long long countSets ( int a [ n ] [ m ] ) {
long long res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < m ; j ++ ) a [ i ] [ j ] ? u ++ : v ++ ; res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 ; }
for ( int i = 0 ; i < m ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < n ; j ++ ) a [ j ] [ i ] ? u ++ : v ++ ; res += pow ( 2 , u ) - 1 + pow ( 2 , v ) - 1 ; }
return res - ( n * m ) ; }
int main ( ) { int a [ ] [ 3 ] = { ( 1 , 0 , 1 ) , ( 0 , 1 , 0 ) } ; cout << countSets ( a ) ; return 0 ; }
void transpose ( int mat [ ] [ MAX ] , int tr [ ] [ MAX ] , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) tr [ i ] [ j ] = mat [ j ] [ i ] ; }
bool isSymmetric ( int mat [ ] [ MAX ] , int N ) { int tr [ N ] [ MAX ] ; transpose ( mat , tr , N ) ; for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != tr [ i ] [ j ] ) return false ; return true ; }
int main ( ) { int mat [ ] [ MAX ] = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; const int MAX = 100 ;
bool isSymmetric ( int mat [ ] [ MAX ] , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i ] [ j ] != mat [ j ] [ i ] ) return false ; return true ; }
int main ( ) { int mat [ ] [ MAX ] = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
const int MAX = 100 ;
int findNormal ( int mat [ ] [ MAX ] , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) sum += mat [ i ] [ j ] * mat [ i ] [ j ] ; return sqrt ( sum ) ; }
int findTrace ( int mat [ ] [ MAX ] , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += mat [ i ] [ i ] ; return sum ; }
int main ( ) { int mat [ ] [ MAX ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; cout << " Trace ▁ of ▁ Matrix ▁ = ▁ " << findTrace ( mat , 5 ) << endl ; cout << " Normal ▁ of ▁ Matrix ▁ = ▁ " << findNormal ( mat , 5 ) << endl ; return 0 ; }
int maxDet ( int n ) { return ( 2 * n * n * n ) ; }
void resMatrix ( int n ) { for ( int i = 0 ; i < 3 ; i ++ ) { for ( int j = 0 ; j < 3 ; j ++ ) {
if ( i == 0 && j == 2 ) cout << "0 ▁ " ; else if ( i == 1 && j == 0 ) cout << "0 ▁ " ; else if ( i == 2 && j == 1 ) cout << "0 ▁ " ;
else cout < < n << " ▁ " ; } cout << " STRNEWLINE " ; } }
int main ( ) { int n = 15 ; cout << " Maximum ▁ Determinant ▁ = ▁ " << maxDet ( n ) ; cout << " Resultant Matrix : " resMatrix ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countNegative ( int M [ ] [ 4 ] , int n , int m ) { int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { if ( M [ i ] [ j ] < 0 ) count += 1 ;
else break ; } } return count ; }
int main ( ) { int M [ 3 ] [ 4 ] = { { -3 , -2 , -1 , 1 } , { -2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; cout << countNegative ( M , 3 , 4 ) ; return 0 ; }
int countNegative ( int M [ ] [ 4 ] , int n , int m ) {
int count = 0 ;
int i = 0 ; int j = m - 1 ;
while ( j >= 0 && i < n ) { if ( M [ i ] [ j ] < 0 ) {
count += j + 1 ;
i += 1 ; }
else j -= 1 ; } return count ; }
int main ( ) { int M [ 3 ] [ 4 ] = { { -3 , -2 , -1 , 1 } , { -2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; cout << countNegative ( M , 3 , 4 ) ; return 0 ; }
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
#define n  5
void printSumSimple ( int mat [ ] [ n ] , int k ) {
if ( k > n ) return ;
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
for ( int j = 0 ; j < n - k + 1 ; j ++ ) {
int sum = 0 ; for ( int p = i ; p < k + i ; p ++ ) for ( int q = j ; q < k + j ; q ++ ) sum += mat [ p ] [ q ] ; cout << sum << " ▁ " ; }
cout << endl ; } }
int main ( ) { int mat [ n ] [ n ] = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; int k = 3 ; printSumSimple ( mat , k ) ; return 0 ; }
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
void selectionSort ( int arr [ ] , int n ) { int i , j , min_idx ;
for ( i = 0 ; i < n - 1 ; i ++ ) {
min_idx = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
swap ( & arr [ min_idx ] , & arr [ i ] ) ; } }
int main ( ) { int arr [ ] = { 64 , 25 , 12 , 22 , 11 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; selectionSort ( arr , n ) ; cout << " Sorted ▁ array : ▁ STRNEWLINE " ; printArray ( arr , n ) ; return 0 ; }
void bubbleSort ( int arr [ ] , int n ) { int i , j ; bool swapped ; for ( i = 0 ; i < n - 1 ; i ++ ) { swapped = false ; for ( j = 0 ; j < n - i - 1 ; j ++ ) { if ( arr [ j ] > arr [ j + 1 ] ) {
swap ( & arr [ j ] , & arr [ j + 1 ] ) ; swapped = true ; } }
if ( swapped == false ) break ; } }
int main ( ) { int arr [ ] = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; bubbleSort ( arr , n ) ; printf ( " Sorted ▁ array : ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; return 0 ; }
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
int count ( int S [ ] , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
int main ( ) { int i , j ; int arr [ ] = { 1 , 2 , 3 } ; int m = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d ▁ " , count ( arr , m , 4 ) ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE int count ( int S [ ] , int m , int n ) {
int table [ n + 1 ] ; memset ( table , 0 , sizeof ( table ) ) ;
table [ 0 ] = 1 ;
for ( int i = 0 ; i < m ; i ++ ) for ( int j = S [ i ] ; j <= n ; j ++ ) table [ j ] += table [ j - S [ i ] ] ; return table [ n ] ; }
int MatrixChainOrder ( int p [ ] , int n ) {
int m [ n ] [ n ] ; int i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i ] [ i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; m [ i ] [ j ] = INT_MAX ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i ] [ j ] ) m [ i ] [ j ] = q ; } } } return m [ 1 ] [ n - 1 ] ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " << MatrixChainOrder ( arr , size ) ; getchar ( ) ; return 0 ; }
int cutRod ( int price [ ] , int n ) { if ( n <= 0 ) return 0 ; int max_val = INT_MIN ;
for ( int i = 0 ; i < n ; i ++ ) max_val = max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) ; return max_val ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ % dn " , cutRod ( arr , size ) ) ; getchar ( ) ; return 0 ; }
int cutRod ( int price [ ] , int n ) { int val [ n + 1 ] ; val [ 0 ] = 0 ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { int max_val = INT_MIN ; for ( j = 0 ; j < i ; j ++ ) max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ % dn " , cutRod ( arr , size ) ) ; getchar ( ) ; return 0 ; }
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
int binomialCoeff ( int n , int k ) ; int binomialCoeff ( int n , int k ) { int res = 1 ; if ( k > n - k ) k = n - k ; for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
void printPascal ( int n ) {
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) printf ( " % d ▁ " , binomialCoeff ( line , i ) ) ; printf ( " STRNEWLINE " ) ; } }
int main ( ) { int n = 7 ; printPascal ( n ) ; return 0 ; }
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
unsigned int getModulo ( unsigned int n , unsigned int d ) { return ( n & ( d - 1 ) ) ; }
int main ( ) { unsigned int n = 6 ;
unsigned int d = 4 ; printf ( " % u ▁ moduo ▁ % u ▁ is ▁ % u " , n , d , getModulo ( n , d ) ) ; getchar ( ) ; return 0 ; }
unsigned int countSetBits ( unsigned int n ) { unsigned int count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
int main ( ) { int i = 9 ; cout << countSetBits ( i ) ; return 0 ; }
int countSetBits ( int n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
int n = 9 ;
cout << countSetBits ( n ) ; return 0 ; }
using namespace std ; int main ( ) { cout << __builtin_popcount ( 4 ) << endl ; cout << __builtin_popcount ( 15 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int num_to_bits [ 16 ] = { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
unsigned int countSetBitsRec ( unsigned int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
int main ( ) { int num = 31 ; cout << countSetBitsRec ( num ) ; return 0 ; }
bool getParity ( unsigned int n ) { bool parity = 0 ; while ( n ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
int main ( ) { unsigned int n = 7 ; cout << " Parity ▁ of ▁ no ▁ " << n << " ▁ = ▁ " << ( getParity ( n ) ? " odd " : " even " ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes " << endl : cout << " No " << endl ; isPowerOfTwo ( 64 ) ? cout << " Yes " << endl : cout << " No " << endl ; return 0 ; }
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 2 != 0 ) return 0 ; n = n / 2 ; } return 1 ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerOfTwo ( 64 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerOfTwo ( 64 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
int maxRepeating ( int * arr , int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) arr [ arr [ i ] % k ] += k ;
int max = arr [ 0 ] , result = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; result = i ; } }
return result ; }
int main ( ) { int arr [ ] = { 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 8 ; cout << " The ▁ maximum ▁ repeating ▁ number ▁ is ▁ " << maxRepeating ( arr , n , k ) << endl ; return 0 ; }
int fun ( int x ) { int y = ( x / 4 ) * 4 ;
int ans = 0 ; for ( int i = y ; i <= x ; i ++ ) ans ^= i ; return ans ; }
int query ( int x ) {
if ( x == 0 ) return 0 ; int k = ( x + 1 ) / 2 ;
return ( x %= 2 ) ? 2 * fun ( k ) : ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) ; } void allQueries ( int q , int l [ ] , int r [ ] ) { for ( int i = 0 ; i < q ; i ++ ) cout << ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) << endl ; }
int main ( ) { int q = 3 ; int l [ ] = { 2 , 2 , 5 } ; int r [ ] = { 4 , 8 , 9 } ; allQueries ( q , l , r ) ; return 0 ; }
int findMinSwaps ( int arr [ ] , int n ) {
int noOfZeroes [ n ] ; memset ( noOfZeroes , 0 , sizeof ( noOfZeroes ) ) ; int i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
int main ( ) { int arr [ ] = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMinSwaps ( arr , n ) ; return 0 ; }
void printTwoOdd ( int arr [ ] , int size ) { int xor2 = arr [ 0 ] ;
int set_bit_no ;
int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } cout << " The ▁ two ▁ ODD ▁ elements ▁ are ▁ " << x << " ▁ & ▁ " << y ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoOdd ( arr , arr_size ) ; return 0 ; }
bool findPair ( int arr [ ] , int size , int n ) {
int i = 0 ; int j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { cout << " Pair ▁ Found : ▁ ( " << arr [ i ] << " , ▁ " << arr [ j ] << " ) " ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } cout << " No ▁ such ▁ pair " ; return false ; }
int main ( ) { int arr [ ] = { 1 , 8 , 30 , 40 , 100 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 60 ; findPair ( arr , size , n ) ; return 0 ; }
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
void findNumbers ( int arr [ ] , int n ) {
int sumN = ( n * ( n + 1 ) ) / 2 ;
int sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
int sum = 0 , sumSq = 0 , i ; for ( i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq = sumSq + ( pow ( arr [ i ] , 2 ) ) ; } int B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; int A = sum - sumN + B ; cout << " A ▁ = ▁ " ; cout << A << endl ; cout << " B ▁ = ▁ " ; cout << B << endl ; }
int main ( ) { int arr [ ] = { 1 , 2 , 2 , 3 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findNumbers ( arr , n ) ; return 0 ; }
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
int middleOfThree ( int a , int b , int c ) {
int x = a - b ;
int y = b - c ;
int z = a - c ;
if ( x * y > 0 ) return b ;
else if ( x * z > 0 ) return c ; else return a ; }
int main ( ) { int a = 20 , b = 30 , c = 40 ; cout << middleOfThree ( a , b , c ) ; return 0 ; }
void missing4 ( int arr [ ] , int n ) {
int helper [ 4 ] ;
for ( int i = 0 ; i < n ; i ++ ) { int temp = abs ( arr [ i ] ) ;
if ( temp <= n ) arr [ temp - 1 ] *= ( -1 ) ;
else if ( temp > n ) { if ( temp % n != 0 ) helper [ temp % n - 1 ] = -1 ; else helper [ ( temp % n ) + n - 1 ] = -1 ; } }
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] > 0 ) cout << ( i + 1 ) << " ▁ " ; for ( int i = 0 ; i < 4 ; i ++ ) if ( helper [ i ] >= 0 ) cout << ( n + i + 1 ) << " ▁ " ; return ; }
int main ( ) { int arr [ ] = { 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; missing4 ( arr , n ) ; return 0 ; }
int minMovesToSort ( int arr [ ] , int n ) { int moves = 0 ; int i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
} return moves ; }
int main ( ) { int arr [ ] = { 3 , 5 , 2 , 8 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minMovesToSort ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void findOptimalPairs ( int arr [ ] , int N ) { sort ( arr , arr + N ) ;
for ( int i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) cout << " ( " << arr [ i ] << " , ▁ " << arr [ j ] << " ) " << " ▁ " ; }
int main ( ) { int arr [ ] = { 9 , 6 , 5 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findOptimalPairs ( arr , N ) ; return 0 ; }
int minOperations ( int * arr , int n ) { int maxi , result = 0 ;
vector < int > freq ( 1000001 , 0 ) ; for ( int i = 0 ; i < n ; i ++ ) { int x = arr [ i ] ; freq [ x ] ++ ; }
maxi = * ( max_element ( arr , arr + n ) ) ; for ( int i = 1 ; i <= maxi ; i ++ ) { if ( freq [ i ] != 0 ) {
for ( int j = i * 2 ; j <= maxi ; j = j + i ) {
freq [ j ] = 0 ; }
result ++ ; } } return result ; }
int main ( ) { int arr [ ] = { 2 , 4 , 2 , 4 , 4 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minOperations ( arr , n ) ; return 0 ; }
int minGCD ( int arr [ ] , int n ) { int minGCD = 0 ;
for ( int i = 0 ; i < n ; i ++ ) minGCD = __gcd ( minGCD , arr [ i ] ) ; return minGCD ; }
int minLCM ( int arr [ ] , int n ) { int minLCM = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) minLCM = min ( minLCM , arr [ i ] ) ; return minLCM ; }
int main ( ) { int arr [ ] = { 2 , 66 , 14 , 521 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " LCM ▁ = ▁ " << minLCM ( arr , n ) << " , ▁ GCD ▁ = ▁ " << minGCD ( arr , n ) ; return 0 ; }
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
int findWaysToPair ( int p ) {
int dp [ p + 1 ] ; dp [ 1 ] = 1 ; dp [ 2 ] = 2 ;
for ( int i = 3 ; i <= p ; i ++ ) { dp [ i ] = dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ; } return dp [ p ] ; }
int main ( ) { int p = 3 ; cout << findWaysToPair ( p ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int CountWays ( int n ) {
if ( n == 0 ) { return 1 ; } if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 1 + 1 ; }
return CountWays ( n - 1 ) + CountWays ( n - 3 ) ; }
int main ( ) { int n = 10 ; cout << CountWays ( n ) ; return 0 ; }
int maxSubArraySumRepeated ( int a [ ] , int n , int k ) { int max_so_far = INT_MIN , max_ending_here = 0 ; for ( int i = 0 ; i < n * k ; i ++ ) {
max_ending_here = max_ending_here + a [ i % n ] ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; if ( max_ending_here < 0 ) max_ending_here = 0 ; } return max_so_far ; }
int main ( ) { int a [ ] = { 10 , 20 , -30 , -1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; int k = 3 ; cout << " Maximum ▁ contiguous ▁ sum ▁ is ▁ " << maxSubArraySumRepeated ( a , n , k ) ; return 0 ; }
int longOddEvenIncSeq ( int arr [ ] , int n ) {
int lioes [ n ] ;
int maxLen = 0 ;
for ( int i = 0 ; i < n ; i ++ ) lioes [ i ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && ( arr [ i ] + arr [ j ] ) % 2 != 0 && lioes [ i ] < lioes [ j ] + 1 ) lioes [ i ] = lioes [ j ] + 1 ;
for ( int i = 0 ; i < n ; i ++ ) if ( maxLen < lioes [ i ] ) maxLen = lioes [ i ] ;
return maxLen ; }
int main ( ) { int arr [ ] = { 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 } ; int n = sizeof ( arr ) / sizeof ( n ) ; cout << " Longest ▁ Increasing ▁ Odd ▁ Even ▁ " << " Subsequence : ▁ " << longOddEvenIncSeq ( arr , n ) ; return 0 ; }
int MatrixChainOrder ( int p [ ] , int i , int j ) { if ( i == j ) return 0 ; int k ; int min = INT_MAX ; int count ;
for ( k = i ; k < j ; k ++ ) { count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " << MatrixChainOrder ( arr , 1 , n - 1 ) ; }
int getCount ( string a , string b ) {
if ( b . length ( ) % a . length ( ) != 0 ) return -1 ; int count = b . length ( ) / a . length ( ) ;
string str = " " ; for ( int i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str == b ) return count ; return -1 ; }
int main ( ) { string a = " geeks " ; string b = " geeksgeeks " ; cout << ( getCount ( a , b ) ) ; return 0 ; }
int countPattern ( string str ) { int len = str . size ( ) ; bool oneSeen = 0 ;
for ( int i = 0 ; i < len ; i ++ ) {
if ( str [ i ] == '1' && oneSeen == 1 ) if ( str [ i - 1 ] == '0' ) count ++ ;
if ( str [ i ] == '1' && oneSeen == 0 ) { oneSeen = 1 ; continue ; }
if ( str [ i ] != '0' && str [ i ] != '1' ) oneSeen = 0 ; } return count ; }
int main ( ) { string str = "100001abc101" ; cout << countPattern ( str ) ; return 0 ; }
int minOperations ( string s , string t , int n ) { int ct0 = 0 , ct1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == t [ i ] ) continue ;
if ( s [ i ] == '0' ) ct0 ++ ;
else ct1 ++ ; } return max ( ct0 , ct1 ) ; }
int main ( ) { string s = "010" , t = "101" ; int n = s . length ( ) ; cout << minOperations ( s , t , n ) ; return 0 ; }
string decryptString ( string str , int n ) {
int i = 0 , jump = 1 ; string decryptedStr = " " ; while ( i < n ) { decryptedStr += str [ i ] ; i += jump ;
jump ++ ; } return decryptedStr ; }
int main ( ) { string str = " geeeeekkkksssss " ; int n = str . length ( ) ; cout << decryptString ( str , n ) ; return 0 ; }
char bitToBeFlipped ( string s ) {
char last = s [ s . length ( ) - 1 ] ; char first = s [ 0 ] ;
if ( last == first ) { if ( last == '0' ) { return '1' ; } else { return '0' ; } }
else if ( last != first ) { return last ; } }
int main ( ) { string s = "1101011000" ; cout << bitToBeFlipped ( s ) << endl ; return 0 ; }
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
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countOccurrences ( char * str , string word ) { char * p ;
vector < string > a ; p = strtok ( str , " ▁ " ) ; while ( p != NULL ) { a . push_back ( p ) ; p = strtok ( NULL , " ▁ " ) ; }
int c = 0 ; for ( int i = 0 ; i < a . size ( ) ; i ++ )
if ( word == a [ i ] ) c ++ ; return c ; }
int main ( ) { char str [ ] = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " ; string word = " portal " ; cout << countOccurrences ( str , word ) ; return 0 ; }
void permute ( string input ) { int n = input . length ( ) ;
int max = 1 << n ;
transform ( input . begin ( ) , input . end ( ) , input . begin ( ) , :: tolower ) ;
for ( int i = 0 ; i < max ; i ++ ) {
string combination = input ; for ( int j = 0 ; j < n ; j ++ ) if ( ( ( i >> j ) & 1 ) == 1 ) combination [ j ] = toupper ( input . at ( j ) ) ;
cout << combination << " ▁ " ; } }
int main ( ) { permute ( " ABC " ) ; return 0 ; }
bool isPalindrome ( const char * str ) {
int l = 0 ; int h = strlen ( str ) - 1 ;
while ( h > l ) if ( str [ l ++ ] != str [ h -- ] ) return false ; return true ; }
int minRemovals ( const char * str ) {
if ( str [ 0 ] == ' ' ) return 0 ;
if ( isPalindrome ( str ) ) return 1 ;
return 2 ; }
int main ( ) { cout << minRemovals ( "010010" ) << endl ; cout << minRemovals ( "0100101" ) << endl ; return 0 ; }
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
void lengtang ( double r1 , double r2 , double d ) { cout << " The ▁ length ▁ of ▁ the ▁ direct " << " ▁ common ▁ tangent ▁ is ▁ " << sqrt ( pow ( d , 2 ) - pow ( ( r1 - r2 ) , 2 ) ) << endl ; }
int main ( ) { double r1 = 4 , r2 = 6 , d = 3 ; lengtang ( r1 , r2 , d ) ; return 0 ; }
void rad ( double d , double h ) { cout << " The ▁ radius ▁ of ▁ the ▁ circle ▁ is ▁ " << ( ( d * d ) / ( 8 * h ) + h / 2 ) << endl ; }
int main ( ) { double d = 4 , h = 1 ; rad ( d , h ) ; return 0 ; }
void shortdis ( double r , double d ) { cout << " The ▁ shortest ▁ distance ▁ " << " from ▁ the ▁ chord ▁ to ▁ centre ▁ " << sqrt ( ( r * r ) - ( ( d * d ) / 4 ) ) << endl ; }
int main ( ) { double r = 4 , d = 3 ; shortdis ( r , d ) ; return 0 ; }
void lengtang ( double r1 , double r2 , double d ) { cout << " The ▁ length ▁ of ▁ the ▁ direct " << " ▁ common ▁ tangent ▁ is ▁ " << sqrt ( pow ( d , 2 ) - pow ( ( r1 - r2 ) , 2 ) ) << endl ; }
int main ( ) { double r1 = 4 , r2 = 6 , d = 12 ; lengtang ( r1 , r2 , d ) ; return 0 ; }
float square ( float a ) {
if ( a < 0 ) return -1 ;
float x = 0.464 * a ; return x ; }
int main ( ) { float a = 5 ; cout << square ( a ) << endl ; return 0 ; }
float polyapothem ( float n , float a ) {
if ( a < 0 && n < 0 ) return -1 ;
return a / ( 2 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; }
int main ( ) { float a = 9 , n = 6 ; cout << polyapothem ( n , a ) << endl ; return 0 ; }
float polyarea ( float n , float a ) {
if ( a < 0 && n < 0 ) return -1 ;
float A = ( a * a * n ) / ( 4 * tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; return A ; }
int main ( ) { float a = 9 , n = 6 ; cout << polyarea ( n , a ) << endl ; return 0 ; }
float calculateSide ( float n , float r ) { float theta , theta_in_radians ; theta = 360 / n ; theta_in_radians = theta * 3.14 / 180 ; return 2 * r * sin ( theta_in_radians / 2 ) ; }
float n = 3 ;
float r = 5 ; cout << calculateSide ( n , r ) ; }
float cyl ( float r , float R , float h ) {
if ( h < 0 && r < 0 && R < 0 ) return -1 ;
float r1 = r ;
float h1 = h ;
float V = 3.14 * pow ( r1 , 2 ) * h1 ; return V ; }
int main ( ) { float r = 7 , R = 11 , h = 6 ; cout << cyl ( r , R , h ) << endl ; return 0 ; }
float Perimeter ( float s , int n ) { float perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
int n = 5 ;
float s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; cout << " Perimeter ▁ of ▁ Regular ▁ Polygon " << " ▁ with ▁ " << n << " ▁ sides ▁ of ▁ length ▁ " << s << " ▁ = ▁ " << peri << endl ; return 0 ; }
float rhombusarea ( float l , float b ) {
if ( l < 0 b < 0 ) return -1 ;
return ( l * b ) / 2 ; }
int main ( ) { float l = 16 , b = 6 ; cout << rhombusarea ( l , b ) << endl ; return 0 ; }
bool FindPoint ( int x1 , int y1 , int x2 , int y2 , int x , int y ) { if ( x > x1 and x < x2 and y > y1 and y < y2 ) return true ; return false ; }
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x = 1 , y = 5 ;
if ( FindPoint ( x1 , y1 , x2 , y2 , x , y ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
void shortest_distance ( float x1 , float y1 , float z1 , float a , float b , float c , float d ) { d = fabs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; float e = sqrt ( a * a + b * b + c * c ) ; cout << " Perpendicular ▁ distance ▁ is ▁ " << ( d / e ) ; return ; }
int main ( ) { float x1 = 4 ; float y1 = -4 ; float z1 = 3 ; float a = 2 ; float b = -2 ; float c = 5 ; float d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ; }
float findVolume ( float l , float b , float h ) {
float volume = ( l * b * h ) / 2 ; return volume ; }
int main ( ) { float l = 18 , b = 12 , h = 9 ;
cout << " Volume ▁ of ▁ triangular ▁ prism : ▁ " << findVolume ( l , b , h ) ; return 0 ; }
void midpoint ( int x1 , int x2 , int y1 , int y2 ) { cout << ( float ) ( x1 + x2 ) / 2 << " ▁ , ▁ " << ( float ) ( y1 + y2 ) / 2 ; }
int main ( ) { int x1 = -1 , y1 = 2 ; int x2 = 3 , y2 = -6 ; midpoint ( x1 , x2 , y1 , y2 ) ; return 0 ; }
double arcLength ( double diameter , double angle ) { double pi = 22.0 / 7.0 ; double arc ; if ( angle >= 360 ) { cout << " Angle ▁ cannot " , " ▁ be ▁ formed " ; return 0 ; } else { arc = ( pi * diameter ) * ( angle / 360.0 ) ; return arc ; } }
int main ( ) { double diameter = 25.0 ; double angle = 45.0 ; double arc_len = arcLength ( diameter , angle ) ; cout << ( arc_len ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void checkCollision ( int a , int b , int c , int x , int y , int radius ) {
int dist = ( abs ( a * x + b * y + c ) ) / sqrt ( a * a + b * b ) ;
if ( radius == dist ) cout << " Touch " << endl ; else if ( radius > dist ) cout << " Intersect " << endl ; else cout << " Outside " << endl ; }
int main ( ) { int radius = 5 ; int x = 0 , y = 0 ; int a = 3 , b = 4 , c = 25 ; checkCollision ( a , b , c , x , y , radius ) ; return 0 ; }
double polygonArea ( double X [ ] , double Y [ ] , int n ) {
double area = 0.0 ;
int j = n - 1 ; for ( int i = 0 ; i < n ; i ++ ) { area += ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) ;
}
return abs ( area / 2.0 ) ; }
int main ( ) { double X [ ] = { 0 , 2 , 4 } ; double Y [ ] = { 1 , 3 , 7 } ; int n = sizeof ( X ) / sizeof ( X [ 0 ] ) ; cout << polygonArea ( X , Y , n ) ; }
int getAverage ( int x , int y ) {
int avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; return avg ; }
int main ( ) { int x = 10 , y = 9 ; cout << getAverage ( x , y ) ; return 0 ; }
int smallestIndex ( int a [ ] , int n ) {
int right1 = 0 , right0 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == 1 ) right1 = i ;
else right0 = i ; }
return min ( right1 , right0 ) ; }
int main ( ) { int a [ ] = { 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << smallestIndex ( a , n ) ; return 0 ; }
int countSquares ( int r , int c , int m ) {
int squares = 0 ;
for ( int i = 1 ; i <= 8 ; i ++ ) { for ( int j = 1 ; j <= 8 ; j ++ ) {
if ( max ( abs ( i - r ) , abs ( j - c ) ) <= m ) squares ++ ; } }
return squares ; }
int main ( ) { int r = 4 , c = 4 , m = 1 ; cout << countSquares ( r , c , m ) << endl ; return 0 ; }
int countNumbers ( int L , int R , int K ) { if ( K == 9 ) K = 0 ;
int totalnumbers = R - L + 1 ;
int factor9 = totalnumbers / 9 ;
int rem = totalnumbers % 9 ;
int ans = factor9 ;
for ( int i = R ; i > R - rem ; i -- ) { int rem1 = i % 9 ; if ( rem1 == K ) ans ++ ; } return ans ; }
int main ( ) { int L = 10 ; int R = 22 ; int K = 3 ; cout << countNumbers ( L , R , K ) ; return 0 ; }
void BalanceArray ( vector < int > & A , vector < vector < int > > & Q ) { vector < int > ANS ; int i , sum = 0 ; for ( i = 0 ; i < A . size ( ) ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; for ( i = 0 ; i < Q . size ( ) ; i ++ ) { int index = Q [ i ] [ 0 ] ; int value = Q [ i ] [ 1 ] ;
if ( A [ index ] % 2 == 0 ) sum = sum - A [ index ] ; A [ index ] = A [ index ] + value ;
if ( A [ index ] % 2 == 0 ) sum = sum + A [ index ] ;
ANS . push_back ( sum ) ; }
for ( i = 0 ; i < ANS . size ( ) ; i ++ ) cout << ANS [ i ] << " ▁ " ; }
int main ( ) { vector < int > A = { 1 , 2 , 3 , 4 } ; vector < vector < int > > Q = { { 0 , 1 } , { 1 , -3 } , { 0 , -4 } , { 3 , 2 } } ; BalanceArray ( A , Q ) ; return 0 ; }
int Cycles ( int N ) { int fact = 1 , result = 0 ; result = N - 1 ;
int i = result ; while ( i > 0 ) { fact = fact * i ; i -- ; } return fact / 2 ; }
int main ( ) { int N = 5 ; int Number = Cycles ( N ) ; cout << " Hamiltonian ▁ cycles ▁ = ▁ " << Number ; return 0 ; }
bool digitWell ( int n , int m , int k ) { int cnt = 0 ; while ( n > 0 ) { if ( n % 10 == m ) ++ cnt ; n /= 10 ; } return cnt == k ; }
int findInt ( int n , int m , int k ) { int i = n + 1 ; while ( true ) { if ( digitWell ( i , m , k ) ) return i ; i ++ ; } }
int main ( ) { int n = 111 , m = 2 , k = 2 ; cout << findInt ( n , m , k ) ; return 0 ; }
int countOdd ( int arr [ ] , int n ) {
int odd = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] % 2 == 1 ) odd ++ ; } return odd ; }
int countValidPairs ( int arr [ ] , int n ) { int odd = countOdd ( arr , n ) ; return ( odd * ( odd - 1 ) ) / 2 ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << countValidPairs ( arr , n ) ; return 0 ; }
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
int countDigitsToBeRemoved ( int N , int K ) {
string s = to_string ( N ) ;
int res = 0 ;
int f_zero = 0 ; for ( int i = s . size ( ) - 1 ; i >= 0 ; i -- ) { if ( K == 0 ) return res ; if ( s [ i ] == '0' ) {
f_zero = 1 ; K -- ; } else res ++ ; }
if ( ! K ) return res ; else if ( f_zero ) return s . size ( ) - 1 ; return -1 ; }
int main ( ) { int N = 10904025 , K = 2 ; cout << countDigitsToBeRemoved ( N , K ) << endl ; N = 1000 , K = 5 ; cout << countDigitsToBeRemoved ( N , K ) << endl ; N = 23985 , K = 2 ; cout << countDigitsToBeRemoved ( N , K ) << endl ; return 0 ; }
float getSum ( int a , int n ) {
float sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) {
sum += ( i / pow ( a , i ) ) ; } return sum ; }
int main ( ) { int a = 3 , n = 3 ;
cout << ( getSum ( a , n ) ) ; return 0 ; }
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
void isHalfReducible ( int arr [ ] , int n , int m ) { int frequencyHash [ m + 1 ] ; int i ; memset ( frequencyHash , 0 , sizeof ( frequencyHash ) ) ; for ( i = 0 ; i < n ; i ++ ) { frequencyHash [ arr [ i ] % ( m + 1 ) ] ++ ; } for ( i = 0 ; i <= m ; i ++ ) { if ( frequencyHash [ i ] >= n / 2 ) break ; } if ( i <= m ) cout << " Yes " << endl ; else cout << " No " << endl ; }
int main ( ) { int arr [ ] = { 8 , 16 , 32 , 3 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int m = 7 ; isHalfReducible ( arr , n , m ) ; return 0 ; }
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) { return false ; } } return true ; }
bool isPowerOfTwo ( int n ) { return ( n && ! ( n & ( n - 1 ) ) ) ; }
int main ( ) { int n = 43 ;
if ( isPrime ( n ) && ( isPowerOfTwo ( n * 3 - 1 ) ) ) { cout << " YES STRNEWLINE " ; } else { cout << " NO STRNEWLINE " ; } return 0 ; }
float area ( float a ) {
if ( a < 0 ) return -1 ;
float area = pow ( ( a * sqrt ( 3 ) ) / ( sqrt ( 2 ) ) , 2 ) ; return area ; }
int main ( ) { float a = 5 ; cout << area ( a ) << endl ; return 0 ; }
int nthTerm ( int n ) { return 3 * pow ( n , 2 ) - 4 * n + 2 ; }
int main ( ) { int N = 4 ; cout << nthTerm ( N ) << endl ; return 0 ; }
int calculateSum ( int n ) { return n * ( n + 1 ) / 2 + pow ( ( n * ( n + 1 ) / 2 ) , 2 ) ; }
int n = 3 ;
cout << " Sum ▁ = ▁ " << calculateSum ( n ) ; return 0 ; }
bool arePermutations ( int a [ ] , int b [ ] , int n , int m ) { int sum1 = 0 , sum2 = 0 , mul1 = 1 , mul2 = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { sum1 += a [ i ] ; mul1 *= a [ i ] ; }
for ( int i = 0 ; i < m ; i ++ ) { sum2 += b [ i ] ; mul2 *= b [ i ] ; }
return ( ( sum1 == sum2 ) && ( mul1 == mul2 ) ) ; }
int main ( ) { int a [ ] = { 1 , 3 , 2 } ; int b [ ] = { 3 , 1 , 2 } ; int n = sizeof ( a ) / sizeof ( int ) ; int m = sizeof ( b ) / sizeof ( int ) ; if ( arePermutations ( a , b , n , m ) ) cout << " Yes " << endl ; else cout << " No " << endl ; return 0 ; }
int Race ( int B , int C ) { int result = 0 ;
result = ( ( C * 100 ) / B ) ; return 100 - result ; }
int main ( ) { int B = 10 , C = 28 ;
B = 100 - B ; C = 100 - C ; cout << Race ( B , C ) << " ▁ meters " ; return 0 ; }
float Time ( float arr [ ] , int n , int Emptypipe ) { float fill = 0 ; for ( int i = 0 ; i < n ; i ++ ) fill += 1 / arr [ i ] ; fill = fill - ( 1 / ( float ) Emptypipe ) ; return 1 / fill ; }
int main ( ) { float arr [ ] = { 12 , 14 } ; float Emptypipe = 30 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << floor ( Time ( arr , n , Emptypipe ) ) << " ▁ Hours " ; return 0 ; }
int check ( int n ) { int sum = 0 ;
while ( n != 0 ) { sum += n % 10 ; n = n / 10 ; }
if ( sum % 7 == 0 ) return 1 ; else return 0 ; }
int n = 25 ; ( check ( n ) == 1 ) ? cout << " YES " : cout << " NO " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  1000005
bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
int SumOfPrimeDivisors ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { if ( n % i == 0 ) { if ( isPrime ( i ) ) sum += i ; } } return sum ; }
int main ( ) { int n = 60 ; cout << " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " << SumOfPrimeDivisors ( n ) << endl ; }
int Sum ( int N ) { int SumOfPrimeDivisors [ N + 1 ] = { 0 } ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( ! SumOfPrimeDivisors [ i ] ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
int main ( ) { int N = 60 ; cout << " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " << Sum ( N ) << endl ; }
ll power ( ll x , ll y , ll p ) {
x = x % p ; while ( y > 0 ) {
if ( y & 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
int main ( ) { ll a = 3 ;
string b = "100000000000000000000000000" ; ll remainderB = 0 ; ll MOD = 1000000007 ;
for ( int i = 0 ; i < b . length ( ) ; i ++ ) remainderB = ( remainderB * 10 + b [ i ] - '0' ) % ( MOD - 1 ) ; cout << power ( a , remainderB , MOD ) << endl ; return 0 ; }
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
int fact ( int n ) { if ( n == 0 n == 1 ) return 1 ; int ans = 1 ; for ( int i = 1 ; i <= n ; i ++ ) ans = ans * i ; return ans ; }
int nCr ( int n , int r ) { int Nr = n , Dr = 1 , ans = 1 ; for ( int i = 1 ; i <= r ; i ++ ) { ans = ( ans * Nr ) / ( Dr ) ; Nr -- ; Dr ++ ; } return ans ; }
int solve ( int n ) { int N = 2 * n - 2 ; int R = n - 1 ; return nCr ( N , R ) * fact ( n - 1 ) ; }
int main ( ) { int n = 6 ; cout << solve ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void pythagoreanTriplet ( int n ) {
for ( int i = 1 ; i <= n / 3 ; i ++ ) {
for ( int j = i + 1 ; j <= n / 2 ; j ++ ) { int k = n - i - j ; if ( i * i + j * j == k * k ) { cout << i << " , ▁ " << j << " , ▁ " << k ; return ; } } } cout << " No ▁ Triplet " ; }
int main ( ) { int n = 12 ; pythagoreanTriplet ( n ) ; return 0 ; }
int factorial ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
void series ( int A , int X , int n ) {
int nFact = factorial ( n ) ;
for ( int i = 0 ; i < n + 1 ; i ++ ) {
int niFact = factorial ( n - i ) ; int iFact = factorial ( i ) ;
int aPow = pow ( A , n - i ) ; int xPow = pow ( X , i ) ;
cout << ( nFact * aPow * xPow ) / ( niFact * iFact ) << " ▁ " ; } }
int main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; return 0 ; }
int seiresSum ( int n , int a [ ] ) { int res = 0 ; for ( int i = 0 ; i < 2 * n ; i ++ ) { if ( i % 2 == 0 ) res += a [ i ] * a [ i ] ; else res -= a [ i ] * a [ i ] ; } return res ; }
int main ( ) { int n = 2 ; int a [ ] = { 1 , 2 , 3 , 4 } ; cout << seiresSum ( n , a ) ; return 0 ; }
int power ( int n , int r ) {
int count = 0 ; for ( int i = r ; ( n / i ) >= 1 ; i = i * r ) count += n / i ; return count ; }
int main ( ) { int n = 6 , r = 3 ; printf ( " ▁ % d ▁ " , power ( n , r ) ) ; return 0 ; }
int avg_of_odd_num ( int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += ( 2 * i + 1 ) ;
return sum / n ; }
int main ( ) { int n = 20 ; cout << avg_of_odd_num ( n ) ; return 0 ; }
int avg_of_odd_num ( int n ) { return n ; }
int main ( ) { int n = 8 ; cout << avg_of_odd_num ( n ) ; return 0 ; }
void fib ( int f [ ] , int N ) {
f [ 1 ] = 1 ; f [ 2 ] = 1 ; for ( int i = 3 ; i <= N ; i ++ )
f [ i ] = f [ i - 1 ] + f [ i - 2 ] ; } void fiboTriangle ( int n ) {
int N = n * ( n + 1 ) / 2 ; int f [ N + 1 ] ; fib ( f , N ) ;
int fiboNum = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) cout << f [ fiboNum ++ ] << " ▁ " ; cout << endl ; } }
int main ( ) { int n = 5 ; fiboTriangle ( n ) ; return 0 ; }
int averageOdd ( int n ) { if ( n % 2 == 0 ) { printf ( " Invalid ▁ Input " ) ; return -1 ; } int sum = 0 , count = 0 ; while ( n >= 1 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
int main ( ) { int n = 15 ; printf ( " % d " , averageOdd ( n ) ) ; return 0 ; }
int TrinomialValue ( int n , int k ) {
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ; }
void printTrinomial ( int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = - i ; j <= 0 ; j ++ ) cout << TrinomialValue ( i , j ) << " ▁ " ;
for ( int j = 1 ; j <= i ; j ++ ) cout << TrinomialValue ( i , j ) << " ▁ " ; cout << endl ; } }
int main ( ) { int n = 4 ; printTrinomial ( n ) ; return 0 ; }
int sumOfLargePrimeFactor ( int n ) {
int prime [ n + 1 ] , sum = 0 ; memset ( prime , 0 , sizeof ( prime ) ) ; int max = n / 2 ; for ( int p = 2 ; p <= max ; p ++ ) {
if ( prime [ p ] == 0 ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = p ; } }
for ( int p = 2 ; p <= n ; p ++ ) {
if ( prime [ p ] ) sum += prime [ p ] ;
else sum += p ; }
return sum ; }
int main ( ) { int n = 12 ; cout << " Sum ▁ = ▁ " << sumOfLargePrimeFactor ( n ) ; return 0 ; }
int calculate_sum ( int a , int N ) {
int m = N / a ;
int sum = m * ( m + 1 ) / 2 ;
int ans = a * sum ; return ans ; }
int main ( ) { int a = 7 , N = 49 ; cout << " Sum ▁ of ▁ multiples ▁ of ▁ " << a << " ▁ up ▁ to ▁ " << N << " ▁ = ▁ " << calculate_sum ( a , N ) << endl ; return 0 ; }
bool ispowerof2 ( cpp_int num ) { if ( ( num & ( num - 1 ) ) == 0 ) return 1 ; return 0 ; }
int main ( ) { cpp_int num = 549755813888 ; cout << ispowerof2 ( num ) << endl ; return 0 ; }
int counDivisors ( int X ) {
int count = 0 ;
for ( int i = 1 ; i <= X ; ++ i ) { if ( X % i == 0 ) { count ++ ; } }
return count ; }
int countDivisorsMult ( int arr [ ] , int n ) {
int mul = 1 ; for ( int i = 0 ; i < n ; ++ i ) mul *= arr [ i ] ;
return counDivisors ( mul ) ; }
int main ( ) { int arr [ ] = { 2 , 4 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << countDivisorsMult ( arr , n ) << endl ; return 0 ; }
int freqPairs ( int arr [ ] , int n ) {
int max = * ( std :: max_element ( arr , arr + n ) ) ;
int freq [ max + 1 ] = { 0 } ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) freq [ arr [ i ] ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 2 * arr [ i ] ; j <= max ; j += arr [ i ] ) {
if ( freq [ j ] >= 1 ) count += freq [ j ] ; }
if ( freq [ arr [ i ] ] > 1 ) { count += freq [ arr [ i ] ] - 1 ; freq [ arr [ i ] ] -- ; } } return count ; }
int main ( ) { int arr [ ] = { 3 , 2 , 4 , 2 , 6 } ; int n = ( sizeof ( arr ) / sizeof ( arr [ 0 ] ) ) ; cout << freqPairs ( arr , n ) ; return 0 ; }
int Nth_Term ( int n ) { return ( 2 * pow ( n , 3 ) - 3 * pow ( n , 2 ) + n + 6 ) / 6 ; }
int main ( ) { int N = 8 ; cout << Nth_Term ( N ) ; }
int printNthElement ( int n ) {
int arr [ n + 1 ] ; arr [ 1 ] = 3 ; arr [ 2 ] = 5 ; for ( int i = 3 ; i <= n ; i ++ ) {
if ( i % 2 != 0 ) arr [ i ] = arr [ i / 2 ] * 10 + 3 ; else arr [ i ] = arr [ ( i / 2 ) - 1 ] * 10 + 5 ; } return arr [ n ] ; }
int main ( ) { int n = 6 ; cout << printNthElement ( n ) ; return 0 ; }
int nthTerm ( int N ) {
return ( N * ( ( N / 2 ) + ( ( N % 2 ) * 2 ) + N ) ) ; }
int N = 5 ;
cout << " Nth ▁ term ▁ for ▁ N ▁ = ▁ " << N << " ▁ : ▁ " << nthTerm ( N ) ; return 0 ; }
void series ( int A , int X , int n ) {
int term = pow ( A , n ) ; cout << term << " ▁ " ;
for ( int i = 1 ; i <= n ; i ++ ) {
term = term * X * ( n - i + 1 ) / ( i * A ) ; cout << term << " ▁ " ; } }
int main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; return 0 ; }
int Div_by_8 ( int n ) { return ( ( ( n >> 3 ) << 3 ) == n ) ; }
int main ( ) { int n = 16 ; if ( Div_by_8 ( n ) ) cout << " YES " << endl ; else cout << " NO " << endl ; return 0 ; }
int averageEven ( int n ) { if ( n % 2 != 0 ) { printf ( " Invalid ▁ Input " ) ; return -1 ; } int sum = 0 , count = 0 ; while ( n >= 2 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
int main ( ) { int n = 16 ; printf ( " % d " , averageEven ( n ) ) ; return 0 ; }
int averageEven ( int n ) { if ( n % 2 != 0 ) { cout << " Invalid ▁ Input " ; return -1 ; } return ( n + 2 ) / 2 ; }
int main ( ) { int n = 16 ; cout << averageEven ( n ) << endl ; return 0 ; }
int gcd ( int a , int b ) {
if ( a == 0 b == 0 ) return 0 ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
int cpFact ( int x , int y ) { while ( gcd ( x , y ) != 1 ) { x = x / gcd ( x , y ) ; } return x ; }
int main ( ) { int x = 15 ; int y = 3 ; cout << cpFact ( x , y ) << endl ; x = 14 ; y = 28 ; cout << cpFact ( x , y ) << endl ; x = 7 ; y = 3 ; cout << cpFact ( x , y ) ; return 0 ; }
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
int fact ( int n ) { if ( n == 0 ) return 1 ; return n * fact ( n - 1 ) ; }
int div ( int x ) { int ans = 0 ; for ( int i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
int sumFactDiv ( int n ) { return div ( fact ( n ) ) ; }
int main ( ) { int n = 4 ; cout << sumFactDiv ( n ) ; }
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
bool checkPandigital ( int b , char n [ ] ) {
if ( strlen ( n ) < b ) return false ; bool hash [ b ] ; memset ( hash , false , sizeof ( hash ) ) ;
for ( int i = 0 ; i < strlen ( n ) ; i ++ ) {
if ( n [ i ] >= '0' && n [ i ] <= '9' ) hash [ n [ i ] - '0' ] = true ;
else if ( n [ i ] - ' A ' <= b - 11 ) hash [ n [ i ] - ' A ' + 10 ] = true ; }
for ( int i = 0 ; i < b ; i ++ ) if ( hash [ i ] == false ) return false ; return true ; }
int main ( ) { int b = 13 ; char n [ ] = "1298450376ABC " ; ( checkPandigital ( b , n ) ) ? ( cout << " Yes " << endl ) : ( cout << " No " << endl ) ; return 0 ; }
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
int maxPrimefactorNum ( int N ) { int arr [ N + 5 ] ; memset ( arr , 0 , sizeof ( arr ) ) ;
for ( int i = 2 ; i * i <= N ; i ++ ) { if ( ! arr [ i ] ) for ( int j = 2 * i ; j <= N ; j += i ) arr [ j ] ++ ; arr [ i ] = 1 ; } int maxval = 0 , maxint = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( arr [ i ] > maxval ) { maxval = arr [ i ] ; maxint = i ; } } return maxint ; }
int main ( ) { int N = 40 ; cout << maxPrimefactorNum ( N ) << endl ; return 0 ; }
long int SubArraySum ( int arr [ ] , int n ) { long int result = 0 ;
for ( int i = 0 ; i < n ; i ++ ) result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) ;
return result ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Sum ▁ of ▁ SubArray ▁ : ▁ " << SubArraySum ( arr , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int highestPowerof2 ( int n ) { int res = 0 ; for ( int i = n ; i >= 1 ; i -- ) {
if ( ( i & ( i - 1 ) ) == 0 ) { res = i ; break ; } } return res ; }
int main ( ) { int n = 10 ; cout << highestPowerof2 ( n ) ; return 0 ; }
void findPairs ( int n ) {
int cubeRoot = pow ( n , 1.0 / 3.0 ) ;
int cube [ cubeRoot + 1 ] ;
for ( int i = 1 ; i <= cubeRoot ; i ++ ) cube [ i ] = i * i * i ;
int l = 1 ; int r = cubeRoot ; while ( l < r ) { if ( cube [ l ] + cube [ r ] < n ) l ++ ; else if ( cube [ l ] + cube [ r ] > n ) r -- ; else { cout << " ( " << l << " , ▁ " << r << " ) " << endl ; l ++ ; r -- ; } } }
int main ( ) { int n = 20683 ; findPairs ( n ) ; return 0 ; }
int gcd ( int a , int b ) { while ( b != 0 ) { int t = b ; b = a % b ; a = t ; } return a ; }
int findMinDiff ( int a , int b , int x , int y ) {
int g = gcd ( a , b ) ;
int diff = abs ( x - y ) % g ; return min ( diff , g - diff ) ; }
int main ( ) { int a = 20 , b = 52 , x = 5 , y = 7 ; cout << findMinDiff ( a , b , x , y ) << endl ; return 0 ; }
void printDivisors ( int n ) {
vector < int > v ; for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) printf ( " % d ▁ " , i ) ; else { printf ( " % d ▁ " , i ) ;
v . push_back ( n / i ) ; } } }
for ( int i = v . size ( ) - 1 ; i >= 0 ; i -- ) printf ( " % d ▁ " , v [ i ] ) ; }
int main ( ) { printf ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ n " ) ; printDivisors ( 100 ) ; return 0 ; }
void printDivisors ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) cout << " ▁ " << i ; }
int main ( ) { cout << " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ; printDivisors ( 100 ) ; return 0 ; }
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
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int reversDigits ( int num ) { static int rev_num = 0 ; static int base_pos = 1 ; if ( num > 0 ) { reversDigits ( num / 10 ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
int main ( ) { int num = 4562 ; cout << " Reverse ▁ of ▁ no . ▁ is ▁ " << reversDigits ( num ) ; return 0 ; }
void printSubsets ( int n ) { for ( int i = n ; i > 0 ; i = ( i - 1 ) & n ) cout << i << " ▁ " ; cout << 0 ; }
int main ( ) { int n = 9 ; printSubsets ( n ) ; return 0 ; }
bool isDivisibleby17 ( int n ) {
if ( n == 0 n == 17 ) return true ;
if ( n < 17 ) return false ;
return isDivisibleby17 ( ( int ) ( n >> 4 ) - ( int ) ( n & 15 ) ) ; }
int main ( ) { int n = 35 ; if ( isDivisibleby17 ( n ) ) cout << n << " ▁ is ▁ divisible ▁ by ▁ 17" ; else cout << n << " ▁ is ▁ not ▁ divisible ▁ by ▁ 17" ; return 0 ; }
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
float getMaxMedian ( int arr [ ] , int n , int k ) { int size = n + k ;
sort ( arr , arr + n ) ;
if ( size % 2 == 0 ) { float median = ( float ) ( arr [ ( size / 2 ) - 1 ] + arr [ size / 2 ] ) / 2 ; return median ; }
float median = arr [ size / 2 ] ; return median ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printSorted ( int a , int b , int c ) {
int get_max = max ( a , max ( b , c ) ) ;
int get_min = - max ( - a , max ( - b , - c ) ) ; int get_mid = ( a + b + c ) - ( get_max + get_min ) ; cout << get_min << " ▁ " << get_mid << " ▁ " << get_max ; }
int main ( ) { int a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ; return 0 ; }
void insertionSort ( int arr [ ] , int n ) { int i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
void printArray ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; insertionSort ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
int countPaths ( int n , int m ) { int dp [ n + 1 ] [ m + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) dp [ i ] [ 0 ] = 1 ; for ( int i = 0 ; i <= m ; i ++ ) dp [ 0 ] [ i ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) for ( int j = 1 ; j <= m ; j ++ ) dp [ i ] [ j ] = dp [ i - 1 ] [ j ] + dp [ i ] [ j - 1 ] ; return dp [ n ] [ m ] ; }
int main ( ) { int n = 3 , m = 2 ; cout << " ▁ Number ▁ of ▁ Paths ▁ " << countPaths ( n , m ) ; return 0 ; }
int count ( int S [ ] , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
int main ( ) { int i , j ; int arr [ ] = { 1 , 2 , 3 } ; int m = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " ▁ " << count ( arr , m , 4 ) ; return 0 ; }
bool isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
string encryptString ( string s , int n , int k ) { int countVowels = 0 ; int countConsonants = 0 ; string ans = " " ;
for ( int l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( int r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s [ r ] ) == true ) countVowels ++ ; else countConsonants ++ ; }
ans += to_string ( countVowels * countConsonants ) ; } return ans ; }
int main ( ) { string s = " hello " ; int n = s . length ( ) ; int k = 2 ; cout << encryptString ( s , n , k ) << endl ; return 0 ; }
float findVolume ( float a ) {
if ( a < 0 ) return -1 ;
float r = a / 2 ;
float h = a ;
float V = 3.14 * pow ( r , 2 ) * h ; return V ; }
int main ( ) { float a = 5 ; cout << findVolume ( a ) << endl ; return 0 ; }
float volumeTriangular ( int a , int b , int h ) { float vol = ( 0.1666 ) * a * b * h ; return vol ; }
float volumeSquare ( int b , int h ) { float vol = ( 0.33 ) * b * b * h ; return vol ; }
float volumePentagonal ( int a , int b , int h ) { float vol = ( 0.83 ) * a * b * h ; return vol ; }
float volumeHexagonal ( int a , int b , int h ) { float vol = a * b * h ; return vol ; }
int main ( ) { int b = 4 , h = 9 , a = 4 ; cout << " Volume ▁ of ▁ triangular " << " ▁ base ▁ pyramid ▁ is ▁ " << volumeTriangular ( a , b , h ) << endl ; cout << " Volume ▁ of ▁ square ▁ " << " ▁ base ▁ pyramid ▁ is ▁ " << volumeSquare ( b , h ) << endl ; cout << " Volume ▁ of ▁ pentagonal " << " ▁ base ▁ pyramid ▁ is ▁ " << volumePentagonal ( a , b , h ) << endl ; cout << " Volume ▁ of ▁ Hexagonal " << " ▁ base ▁ pyramid ▁ is ▁ " << volumeHexagonal ( a , b , h ) ; return 0 ; }
double Area ( int b1 , int b2 , int h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
int main ( ) { int base1 = 8 , base2 = 10 , height = 6 ; double area = Area ( base1 , base2 , height ) ; cout << " Area ▁ is : ▁ " << area ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int numberOfDiagonals ( int n ) { return n * ( n - 3 ) / 2 ; }
int main ( ) { int n = 5 ; cout << n << " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ; cout << numberOfDiagonals ( n ) << " ▁ diagonals " ; return 0 ; }
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
int LowerInsertionPoint ( int arr [ ] , int n , int X ) {
if ( X < arr [ 0 ] ) return 0 ; else if ( X > arr [ n - 1 ] ) return n ; int lowerPnt = 0 ; int i = 1 ; while ( i < n && arr [ i ] < X ) { lowerPnt = i ; i = i * 2 ; }
while ( lowerPnt < n && arr [ lowerPnt ] < X ) lowerPnt ++ ; return lowerPnt ; }
int main ( ) { int arr [ ] = { 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int X = 4 ; cout << LowerInsertionPoint ( arr , n , X ) ; return 0 ; }
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
int middleOfThree ( int a , int b , int c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
int main ( ) { int a = 20 , b = 30 , c = 40 ; cout << middleOfThree ( a , b , c ) ; return 0 ; }
void printArr ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] ; }
bool compare ( int num1 , int num2 ) {
string A = to_string ( num1 ) ;
string B = to_string ( num2 ) ;
return ( A + B ) <= ( B + A ) ; }
void printSmallest ( int N , int arr [ ] ) {
sort ( arr , arr + N , compare ) ;
printArr ( arr , N ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 2 , 9 , 21 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printSmallest ( N , arr ) ; return 0 ; }
bool isPossible ( int a [ ] , int b [ ] , int n , int k ) {
sort ( a , a + n ) ;
sort ( b , b + n , greater < int > ( ) ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
int main ( ) { int a [ ] = { 2 , 1 , 3 } ; int b [ ] = { 7 , 8 , 9 } ; int k = 10 ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; isPossible ( a , b , n , k ) ? cout << " Yes " : cout << " No " ; return 0 ; }
string encryptString ( string str , int n ) { int i = 0 , cnt = 0 ; string encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- ) encryptedStr += str [ i ] ; i ++ ; } return encryptedStr ; }
int main ( ) { string str = " geeks " ; int n = str . length ( ) ; cout << encryptString ( str , n ) ; return 0 ; }
int minDiff ( int n , int x , int A [ ] ) { int mn = A [ 0 ] , mx = A [ 0 ] ;
for ( int i = 0 ; i < n ; ++ i ) { mn = min ( mn , A [ i ] ) ; mx = max ( mx , A [ i ] ) ; }
return max ( 0 , mx - mn - 2 * x ) ; }
int main ( ) { int n = 3 , x = 3 ; int A [ ] = { 1 , 3 , 6 } ;
cout << minDiff ( n , x , A ) ; return 0 ; }
