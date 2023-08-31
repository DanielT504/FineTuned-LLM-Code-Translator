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
static int countWays ( int N ) {
int E = ( N * ( N - 1 ) ) / 2 ; if ( N == 1 ) return 0 ; return ( int ) Math . Pow ( 2 , E - 1 ) ; }
static public void Main ( ) { int N = 4 ; Console . WriteLine ( countWays ( N ) ) ; } }
static int minAbsDiff ( int n ) { int mod = n % 4 ; if ( mod == 0 mod == 3 ) { return 0 ; } return 1 ; }
static public void Main ( ) { int n = 5 ; Console . WriteLine ( minAbsDiff ( n ) ) ; } }
using System ; class GFG { static bool check ( int s ) {
int [ ] freq = new int [ 10 ] ; int r , i ; for ( i = 0 ; i < 10 ; i ++ ) { freq [ i ] = 0 ; } while ( s != 0 ) {
r = s % 10 ;
s = ( int ) ( s / 10 ) ;
freq [ r ] += 1 ; } int xor__ = 0 ;
for ( i = 0 ; i < 10 ; i ++ ) { xor__ = xor__ ^ freq [ i ] ; if ( xor__ == 0 ) return true ; else return false ; } return true ; }
public static void Main ( ) { int s = 122233 ; if ( check ( s ) ) Console . Write ( " Yes STRNEWLINE " ) ; else Console . Write ( " No STRNEWLINE " ) ; } }
static void printLines ( int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) { Console . WriteLine ( k * ( 6 * i + 1 ) + " ▁ " + k * ( 6 * i + 2 ) + " ▁ " + k * ( 6 * i + 3 ) + " ▁ " + k * ( 6 * i + 5 ) ) ; } }
public static void Main ( ) { int n = 2 , k = 2 ; printLines ( n , k ) ; } }
using System ; class GFG { static int calculateSum ( int n ) {
return ( ( int ) Math . Pow ( 2 , n + 1 ) + n - 2 ) ; }
int n = 4 ;
Console . WriteLine ( " Sum ▁ = ▁ " + calculateSum ( n ) ) ; } }
static long partitions ( int n ) { long [ ] p = new long [ n + 1 ] ;
p [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; ++ i ) { int k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 != 0 ? 1 : - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) { k *= - 1 ; } else { k = 1 - k ; } } } return p [ n ] ; }
public static void Main ( String [ ] args ) { int N = 20 ; Console . WriteLine ( partitions ( N ) ) ; } }
static int countPaths ( int n , int m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
public static void Main ( ) { int n = 3 , m = 2 ; Console . WriteLine ( " ▁ Number ▁ of " + " ▁ Paths ▁ " + countPaths ( n , m ) ) ; } }
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
static int countChar ( string str , char x ) { int count = 0 ; int n = 10 ; for ( int i = 0 ; i < str . Length ; i ++ ) if ( str [ i ] == x ) count ++ ;
int repetitions = n / str . Length ; count = count * repetitions ;
for ( int i = 0 ; i < n % str . Length ; i ++ ) { if ( str [ i ] == x ) count ++ ; } return count ; }
public static void Main ( ) { string str = " abcac " ; Console . WriteLine ( countChar ( str , ' a ' ) ) ; } }
static bool check ( string s , int m ) {
int l = s . Length ;
int c1 = 0 ;
int c2 = 0 ; for ( int i = 0 ; i < l ; i ++ ) { if ( s [ i ] == '0' ) { c2 = 0 ;
c1 ++ ; } else { c1 = 0 ;
c2 ++ ; } if ( c1 == m c2 == m ) return true ; } return false ; }
public static void Main ( ) { String s = "001001" ; int m = 2 ;
if ( check ( s , m ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
static int productAtKthLevel ( string tree , int k ) { int level = - 1 ;
int product = 1 ; int n = tree . Length ; for ( int i = 0 ; i < n ; i ++ ) {
if ( tree [ i ] == ' ( ' ) level ++ ;
else if ( tree [ i ] == ' ) ' ) -- ; else {
if ( level == k ) product *= ( tree [ i ] - '0' ) ; } }
return product ; }
static void Main ( ) { string tree = " ( 0(5(6 ( ) ( ) ) ( 4 ( ) (9 ( ) ( ) ) ) ) ( 7(1 ( ) ( ) ) ( 3 ( ) ( ) ) ) ) " ; int k = 2 ; Console . WriteLine ( productAtKthLevel ( tree , k ) ) ; } }
using System ; class GFG { static bool isValidISBN ( string isbn ) {
int n = isbn . Length ; if ( n != 10 ) return false ;
int sum = 0 ; for ( int i = 0 ; i < 9 ; i ++ ) { int digit = isbn [ i ] - '0' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
char last = isbn [ 9 ] ; if ( last != ' X ' && ( last < '0' last > '9' ) ) return false ;
sum += ( ( last == ' X ' ) ? 10 : ( last - '0' ) ) ;
return ( sum % 11 == 0 ) ; }
public static void Main ( ) { string isbn = "007462542X " ; if ( isValidISBN ( isbn ) ) Console . WriteLine ( " Valid " ) ; else Console . WriteLine ( " Invalid " ) ; } }
public static void Main ( ) { int d = 10 ; double a ;
a = ( double ) ( 360 - ( 6 * d ) ) / 4 ;
Console . WriteLine ( a + " , ▁ " + ( a + d ) + " , ▁ " + ( a + ( 2 * d ) ) + " , ▁ " + ( a + ( 3 * d ) ) ) ; } }
static void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { z1 = - d1 / c1 ; d = Math . Abs ( ( c2 * z1 + d2 ) ) / ( float ) ( Math . Sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; Console . Write ( " Perpendicular ▁ distance ▁ is ▁ " + d ) ; } else Console . Write ( " Planes ▁ are ▁ not ▁ parallel " ) ; }
public static void Main ( ) { float a1 = 1 ; float b1 = 2 ; float c1 = - 1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = - 3 ; float d2 = - 4 ; distance ( a1 , b1 , c1 , d1 ,
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
public static bool possibleToReach ( int a , int b ) {
int c = ( int ) Math . Pow ( a * b , ( double ) 1 / 3 ) ;
int re1 = a / c ; int re2 = b / c ;
if ( ( re1 * re1 * re2 == a ) && ( re2 * re2 * re1 == b ) ) return true ; else return false ; }
static public void Main ( String [ ] args ) { int A = 60 , B = 450 ; if ( possibleToReach ( A , B ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG { public static bool isUndulating ( string n ) {
if ( n . Length <= 2 ) return false ;
for ( int i = 2 ; i < n . Length ; i ++ ) if ( n [ i - 2 ] != n [ i ] ) return false ; return true ; }
public static void Main ( ) { string n = "1212121" ; if ( isUndulating ( n ) == true ) Console . WriteLine ( " yes " ) ; else Console . WriteLine ( " no " ) ; } }
static int Series ( int n ) { int i ; int sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
public static void Main ( ) { int n = 3 ; int res = Series ( n ) ; Console . Write ( res ) ; } }
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
static int horner ( int [ ] poly , int n , int x ) {
int result = poly [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) result = result * x + poly [ i ] ; return result ; }
static int findSign ( int [ ] poly , int n , int x ) { int result = horner ( poly , n , x ) ; if ( result > 0 ) return 1 ; else if ( result < 0 ) return - 1 ; return 0 ; }
int [ ] poly = { 2 , - 6 , 2 , - 1 } ; int x = 3 ; int n = poly . Length ; Console . Write ( " Sign ▁ of ▁ polynomial ▁ is ▁ " + findSign ( poly , n , x ) ) ; } }
class GFG { static int MAX = 100005 ;
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
public static long SubArraySum ( int [ ] arr , int n ) { long result = 0 , temp = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
temp = 0 ; for ( int j = i ; j < n ; j ++ ) {
temp += arr [ j ] ; result += temp ; } } return result ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int n = arr . Length ; Console . Write ( " Sum ▁ of ▁ SubArray ▁ : ▁ " + SubArraySum ( arr , n ) ) ; } }
using System ; class GFG { static int highestPowerof2 ( int n ) { int p = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) ; return ( int ) Math . Pow ( 2 , p ) ; }
static public void Main ( ) { int n = 10 ; Console . WriteLine ( highestPowerof2 ( n ) ) ; } }
static int aModM ( string s , int mod ) { int number = 0 ; for ( int i = 0 ; i < s . Length ; i ++ ) {
number = ( number * 10 ) ; int x = ( int ) ( s [ i ] - '0' ) ; number = number + x ; number %= mod ; } return number ; }
static int ApowBmodM ( string a , int b , int m ) {
int ans = aModM ( a , m ) ; int mul = ans ;
for ( int i = 1 ; i < b ; i ++ ) ans = ( ans * mul ) % m ; return ans ; }
public static void Main ( ) { string a = "987584345091051645734583954832576" ; int b = 3 , m = 11 ; Console . Write ( ApowBmodM ( a , b , m ) ) ; } }
static int SieveOfSundaram ( int n ) {
int nNew = ( n - 1 ) / 2 ;
for ( int i = 0 ; i < nNew + 1 ; i ++ ) marked [ i ] = false ;
for ( int i = 1 ; i <= nNew ; i ++ ) for ( int j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) Console . Write ( 2 + " ▁ " ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) Console . Write ( 2 * i + 1 + " ▁ " ) ; return - 1 ; }
public static void Main ( ) { int n = 20 ; SieveOfSundaram ( n ) ; } }
static int hammingDistance ( int n1 , int n2 ) { int x = n1 ^ n2 ; int setBits = 0 ; while ( x > 0 ) { setBits += x & 1 ; x >>= 1 ; } return setBits ; }
static void Main ( ) { int n1 = 9 , n2 = 14 ; System . Console . WriteLine ( hammingDistance ( n1 , n2 ) ) ; } }
static void printSubsets ( int n ) { for ( int i = 0 ; i <= n ; i ++ ) if ( ( n & i ) == i ) Console . Write ( i + " ▁ " ) ; }
public static void Main ( ) { int n = 9 ; printSubsets ( n ) ; } }
using System ; public class GFG { static int setBitNumber ( int n ) {
int k = ( int ) ( Math . Log ( n ) / Math . Log ( 2 ) ) ;
return 1 << k ; }
static public void Main ( ) { int n = 273 ; Console . WriteLine ( setBitNumber ( n ) ) ; } }
public static int subset ( int [ ] ar , int n ) {
int res = 0 ;
Array . Sort ( ar ) ;
for ( int i = 0 ; i < n ; i ++ ) { int count = 1 ;
for ( ; i < n - 1 ; i ++ ) { if ( ar [ i ] == ar [ i + 1 ] ) count ++ ; else break ; }
res = Math . Max ( res , count ) ; } return res ; }
public static void Main ( ) { int [ ] arr = { 5 , 6 , 9 , 3 , 4 , 3 , 4 } ; int n = 7 ; Console . WriteLine ( subset ( arr , n ) ) ; } }
static bool areElementsContiguous ( int [ ] arr , int n ) {
Array . Sort ( arr ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] > 1 ) return false ; return true ; }
public static void Main ( ) { int [ ] arr = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . Length ; if ( areElementsContiguous ( arr , n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
static int findLargestd ( int [ ] S , int n ) { bool found = false ;
Array . Sort ( S ) ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { for ( int j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( int k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( int l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return int . MaxValue ; return - 1 ; }
public static void Main ( ) { int [ ] S = new int [ ] { 2 , 3 , 5 , 7 , 12 } ; int n = S . Length ; int ans = findLargestd ( S , n ) ; if ( ans == int . MaxValue ) Console . WriteLine ( " No ▁ Solution " ) ; else Console . Write ( " Largest ▁ d ▁ such ▁ that ▁ a ▁ + ▁ " + " b ▁ + ▁ c ▁ = ▁ d ▁ is ▁ " + ans ) ; } }
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
static void pushZerosToEnd ( int [ ] arr , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
public static void Main ( ) { int [ ] arr = { 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = arr . Length ; pushZerosToEnd ( arr , n ) ; Console . WriteLine ( " Array ▁ after ▁ pushing ▁ all ▁ zeros ▁ to ▁ the ▁ back : ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
static void printArray ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
static void RearrangePosNeg ( int [ ] arr , int n ) { int key , j ; for ( int i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
public static void Main ( ) { int [ ] arr = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; int n = arr . Length ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ; } }
using System ; class GFG { static void findElements ( int [ ] arr , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . Length ; findElements ( arr , n ) ; } }
using System ; class GFG { static void findElements ( int [ ] arr , int n ) { Array . Sort ( arr ) ; for ( int i = 0 ; i < n - 2 ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . Length ; findElements ( arr , n ) ; } }
using System ; class GFG { static void findElements ( int [ ] arr , int n ) { int first = int . MinValue ; int second = int . MaxValue ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . Length ; findElements ( arr , n ) ; } }
static int findFirstMissing ( int [ ] array , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
public static void Main ( ) { int [ ] arr = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = arr . Length ; Console . Write ( " smallest ▁ Missing ▁ element ▁ is ▁ : ▁ " + findFirstMissing ( arr , 0 , n - 1 ) ) ; } }
static int FindMaxSum ( int [ ] arr , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 5 , 5 , 10 , 100 , 10 , 5 } ; Console . Write ( FindMaxSum ( arr , arr . Length ) ) ; } }
static int findMaxAverage ( int [ ] arr , int n , int k ) {
if ( k > n ) return - 1 ;
int [ ] csum = new int [ n ] ; csum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) csum [ i ] = csum [ i - 1 ] + arr [ i ] ;
int max_sum = csum [ k - 1 ] , max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { int curr_sum = csum [ i ] - csum [ i - k ] ; if ( curr_sum > max_sum ) { max_sum = curr_sum ; max_end = i ; } }
return max_end - k + 1 ; }
static public void Main ( ) { int [ ] arr = { 1 , 12 , - 5 , - 6 , 50 , 3 } ; int k = 4 ; int n = arr . Length ; Console . WriteLine ( " The ▁ maximum ▁ average ▁ subarray ▁ of ▁ " + " length ▁ " + k + " ▁ begins ▁ at ▁ index ▁ " + findMaxAverage ( arr , n , k ) ) ; } }
static int findMaxAverage ( int [ ] arr , int n , int k ) {
if ( k > n ) return - 1 ;
int sum = arr [ 0 ] ; for ( int i = 1 ; i < k ; i ++ ) sum += arr [ i ] ; int max_sum = sum ; int max_end = k - 1 ;
for ( int i = k ; i < n ; i ++ ) { sum = sum + arr [ i ] - arr [ i - k ] ; if ( sum > max_sum ) { max_sum = sum ; max_end = i ; } }
return max_end - k + 1 ; }
public static void Main ( ) { int [ ] arr = { 1 , 12 , - 5 , - 6 , 50 , 3 } ; int k = 4 ; int n = arr . Length ; Console . WriteLine ( " The ▁ maximum ▁ " + " average ▁ subarray ▁ of ▁ length ▁ " + k + " ▁ begins ▁ at ▁ index ▁ " + findMaxAverage ( arr , n , k ) ) ; } }
using System ; class GFG { static bool isMajority ( int [ ] arr , int n , int x ) { int i , last_index = 0 ;
last_index = ( n % 2 == 0 ) ? n / 2 : n / 2 + 1 ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + n / 2 ] == x ) return true ; } return false ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 4 , 4 , 4 } ; int n = arr . Length ; int x = 4 ; if ( isMajority ( arr , n , x ) == true ) Console . Write ( x + " ▁ appears ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; else Console . Write ( x + " ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; } }
static int cutRod ( int [ ] price , int n ) { int [ ] val = new int [ n + 1 ] ; val [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { int max_val = int . MinValue ; for ( int j = 0 ; j < i ; j ++ ) max_val = Math . Max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . Length ; Console . WriteLine ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
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
static int minValue ( int n , int x , int y ) {
float val = ( y * n ) / 100 ;
if ( x >= val ) return 0 ; else return ( int ) ( Math . Ceiling ( val ) - x ) ; }
public static void Main ( ) { int n = 10 , x = 2 , y = 40 ; Console . WriteLine ( ( int ) minValue ( n , x , y ) ) ; } }
static bool isPrime ( long n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static bool isFactorialPrime ( long n ) {
if ( ! isPrime ( n ) ) return false ; long fact = 1 ; int i = 1 ; while ( fact <= n + 1 ) {
fact = fact * i ;
if ( n + 1 == fact n - 1 == fact ) return true ; i ++ ; }
return false ; }
public static void Main ( ) { int n = 23 ; if ( isFactorialPrime ( n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
long n = 5 ;
long fac1 = 1 ; for ( int i = 2 ; i <= n - 1 ; i ++ ) fac1 = fac1 * i ;
long fac2 = fac1 * n ;
long totalWays = fac1 * fac2 ;
Console . WriteLine ( totalWays ) ; } }
static int nextPerfectCube ( int N ) { int nextN = ( int ) Math . Floor ( Math . Pow ( N , ( double ) 1 / 3 ) ) + 1 ; return nextN * nextN * nextN ; }
public static void Main ( ) { int n = 35 ; Console . Write ( nextPerfectCube ( n ) ) ; } }
using System ; class GFG { static int findpos ( String n ) { int pos = 0 ; for ( int i = 0 ; i < n . Length ; i ++ ) { switch ( n [ i ] ) {
'2' : pos = pos * 4 + 1 ; break ;
'3' : pos = pos * 4 + 2 ; break ;
'5' : pos = pos * 4 + 3 ; break ;
'7' : pos = pos * 4 + 4 ; break ; } } return pos ; }
public static void Main ( String [ ] args ) { String n = "777" ; Console . WriteLine ( findpos ( n ) ) ; } }
using System ; class GFG { static int mod = 1000000007 ;
static int digitNumber ( long n ) {
if ( n == 0 ) return 1 ;
if ( n == 1 ) return 9 ;
if ( n % 2 != 0 ) {
int temp = digitNumber ( ( n - 1 ) / 2 ) % mod ; return ( 9 * ( temp * temp ) % mod ) % mod ; } else {
int temp = digitNumber ( n / 2 ) % mod ; return ( temp * temp ) % mod ; } } static int countExcluding ( int n , int d ) {
if ( d == 0 ) return ( 9 * digitNumber ( n - 1 ) ) % mod ; else return ( 8 * digitNumber ( n - 1 ) ) % mod ; }
int d = 9 ; int n = 3 ; Console . WriteLine ( countExcluding ( n , d ) ) ; } }
public static bool isPrime ( int n ) {
if ( n <= 1 ) return false ;
for ( int i = 2 ; i < n ; i ++ ) if ( n % i == 0 ) return false ; return true ; }
public static bool isEmirp ( int n ) {
if ( isPrime ( n ) == false ) return false ;
int rev = 0 ; while ( n != 0 ) { int d = n % 10 ; rev = rev * 10 + d ; n /= 10 ; }
return isPrime ( rev ) ; }
int n = 13 ; if ( isEmirp ( n ) == true ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
static double Convert ( double radian ) { double pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
public static void Main ( ) { double radian = 5.0 ; double degree = Convert ( radian ) ; Console . Write ( " degree ▁ = ▁ " + degree ) ; } }
static int sn ( int n , int an ) { return ( n * ( 1 + an ) ) / 2 ; }
static int trace ( int n , int m ) {
int an = 1 + ( n - 1 ) * ( m + 1 ) ;
int rowmajorSum = sn ( n , an ) ;
an = 1 + ( n - 1 ) * ( n + 1 ) ;
int colmajorSum = sn ( n , an ) ; return rowmajorSum + colmajorSum ; }
static public void Main ( ) { int N = 3 , M = 3 ; Console . WriteLine ( trace ( N , M ) ) ; } }
static void max_area ( int n , int m , int k ) { if ( k > ( n + m - 2 ) ) Console . WriteLine ( " Not ▁ possible " ) ; else { int result ;
if ( k < Math . Max ( m , n ) - 1 ) { result = Math . Max ( m * ( n / ( k + 1 ) ) , n * ( m / ( k + 1 ) ) ) ; }
else { result = Math . Max ( m / ( k - n + 2 ) , n / ( k - m + 2 ) ) ; }
Console . WriteLine ( result ) ; } }
public static void Main ( ) { int n = 3 , m = 4 , k = 1 ; max_area ( n , m , k ) ; } }
static int area_fun ( int side ) { int area = side * side ; return area ; }
public static void Main ( ) { int side = 4 ; int area = area_fun ( side ) ; Console . WriteLine ( area ) ; } }
static int countConsecutive ( int N ) {
int count = 0 ; for ( int L = 1 ; L * ( L + 1 ) < 2 * N ; L ++ ) { double a = ( double ) ( ( 1.0 * N - ( L * ( L + 1 ) ) / 2 ) / ( L + 1 ) ) ; if ( a - ( int ) a == 0.0 ) count ++ ; } return count ; }
public static void Main ( ) { int N = 15 ; Console . WriteLine ( countConsecutive ( N ) ) ; N = 10 ; Console . Write ( countConsecutive ( N ) ) ; } }
static bool isAutomorphic ( int N ) {
int sq = N * N ;
while ( N > 0 ) {
if ( N % 10 != sq % 10 ) return false ;
N /= 10 ; sq /= 10 ; } return true ; }
public static void Main ( ) { int N = 5 ; Console . Write ( isAutomorphic ( N ) ? " Automorphic " : " Not ▁ Automorphic " ) ; } }
static int maxPrimefactorNum ( int N ) {
bool [ ] arr = new bool [ N + 5 ] ; int i ;
for ( i = 3 ; i * i <= N ; i += 2 ) { if ( ! arr [ i ] ) { for ( int j = i * i ; j <= N ; j += i ) { arr [ j ] = true ; } } }
ArrayList prime = new ArrayList ( ) ; prime . Add ( 2 ) ; for ( i = 3 ; i <= N ; i += 2 ) { if ( ! arr [ i ] ) { prime . Add ( i ) ; } }
int ans = 1 ; i = 0 ; while ( ans * ( int ) prime [ i ] <= N && i < prime . Count ) { ans *= ( int ) prime [ i ] ; i ++ ; } return ans ; }
public static void Main ( ) { int N = 40 ; Console . Write ( maxPrimefactorNum ( N ) ) ; } }
static int divSum ( int num ) {
int result = 0 ;
for ( int i = 2 ; i <= Math . Sqrt ( num ) ; i ++ ) {
if ( num % i == 0 ) {
if ( i == ( num / i ) ) result += i ; else result += ( i + num / i ) ; } }
return ( result + 1 ) ; }
public static void Main ( ) { int num = 36 ; Console . Write ( divSum ( num ) ) ; } }
static int power ( int x , int y , int p ) {
while ( y > 0 ) {
if ( y % 2 == 1 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
static void squareRoot ( int n , int p ) { if ( p % 4 != 3 ) { Console . Write ( " Invalid ▁ Input " ) ; return ; }
n = n % p ; int x = power ( n , ( p + 1 ) / 4 , p ) ; if ( ( x * x ) % p == n ) { Console . Write ( " Square ▁ root ▁ is ▁ " + x ) ; return ; }
x = p - x ; if ( ( x * x ) % p == n ) { Console . Write ( " Square ▁ root ▁ is ▁ " + x ) ; return ; }
Console . Write ( " Square ▁ root ▁ doesn ' t ▁ exist ▁ " ) ; }
static public void Main ( ) { int p = 7 ; int n = 2 ; squareRoot ( n , p ) ; } }
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
public static void Main ( ) { int N = 6 ; int Even = N / 2 ; int Odd = N - Even ; Console . WriteLine ( Even * Odd ) ; } }
public static void steps ( string str , int n ) {
bool flag = false ; int x = 0 ;
for ( int i = 0 ; i < str . Length ; i ++ ) {
if ( x == 0 ) { flag = true ; }
if ( x == n - 1 ) { flag = false ; }
for ( int j = 0 ; j < x ; j ++ ) { Console . Write ( " * " ) ; } Console . Write ( str [ i ] + " STRNEWLINE " ) ;
if ( flag == true ) { x ++ ; } else { x -- ; } } }
int n = 4 ; string str = " GeeksForGeeks " ; Console . WriteLine ( " String : ▁ " + str ) ; Console . WriteLine ( " Max ▁ Length ▁ of ▁ Steps : ▁ " + n ) ;
steps ( str , n ) ; } }
static bool isDivisible ( String str , int k ) { int n = str . Length ; int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) if ( str [ n - i - 1 ] == '0' ) c ++ ;
return ( c == k ) ; }
String str1 = "10101100" ; int k = 2 ; if ( isDivisible ( str1 , k ) == true ) Console . Write ( " Yes STRNEWLINE " ) ; else Console . Write ( " No " ) ;
String str2 = "111010100" ; k = 2 ; if ( isDivisible ( str2 , k ) == true ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
static bool isNumber ( string s ) { for ( int i = 0 ; i < s . Length ; i ++ ) if ( char . IsDigit ( s [ i ] ) == false ) return false ; return true ; }
string str = "6790" ;
if ( isNumber ( str ) ) Console . WriteLine ( " Integer " ) ;
else Console . ( " String " ) ; } }
static void reverse ( String str ) { if ( ( str == null ) || ( str . Length <= 1 ) ) Console . Write ( str ) ; else { Console . Write ( str [ str . Length - 1 ] ) ; reverse ( str . Substring ( 0 , ( str . Length - 1 ) ) ) ; } }
public static void Main ( ) { String str = " Geeks ▁ for ▁ Geeks " ; reverse ( str ) ; } }
static double polyarea ( double n , double r ) {
if ( r < 0 && n < 0 ) return - 1 ;
double A = ( ( r * r * n ) * Math . Sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
public static void Main ( ) { float r = 9 , n = 6 ; Console . WriteLine ( polyarea ( n , r ) ) ; } }
static double findPCSlope ( double m ) { return - 1.0 / m ; }
public static void Main ( ) { double m = 2.0 ; Console . Write ( findPCSlope ( m ) ) ; } }
static float area_of_segment ( float radius , float angle ) {
float area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
float area_of_triangle = ( float ) 1 / 2 * ( radius * radius ) * ( float ) Math . Sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
public static void Main ( ) { float radius = 10.0f , angle = 90.0f ; Console . WriteLine ( " Area ▁ of ▁ minor ▁ segment ▁ = ▁ " + area_of_segment ( radius , angle ) ) ; Console . WriteLine ( " Area ▁ of ▁ major ▁ segment ▁ = ▁ " + area_of_segment ( radius , ( 360 - angle ) ) ) ; } }
using System ; class GFG { static void SectorArea ( double radius , double angle ) { if ( angle >= 360 ) Console . WriteLine ( " Angle ▁ not ▁ possible " ) ;
else { double sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; Console . WriteLine ( sector ) ; } }
public static void Main ( ) { double radius = 9 ; double angle = 60 ; SectorArea ( radius , angle ) ; } }
static void insertionSortRecursive ( int [ ] arr , int n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
int last = arr [ n - 1 ] ; int j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
static void Main ( ) { int [ ] arr = { 12 , 11 , 13 , 5 , 6 } ; insertionSortRecursive ( arr , arr . Length ) ; for ( int i = 0 ; i < arr . Length ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; } }
static bool isWaveArray ( int [ ] arr , int n ) { bool result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
int [ ] arr = { 1 , 3 , 2 , 4 } ; int n = arr . Length ; if ( isWaveArray ( arr , n ) ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
using System ; public class GFG { static int mod = 1000000007 ;
static int sumOddFibonacci ( int n ) { int [ ] Sum = new int [ n + 1 ] ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( int i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
static public void Main ( ) { int n = 6 ; Console . WriteLine ( sumOddFibonacci ( n ) ) ; }
using System ; class GFG { static int solve ( int N , int K ) {
int [ ] combo ; combo = new int [ 50 ] ;
combo [ 0 ] = 1 ;
for ( int i = 1 ; i <= K ; i ++ ) {
for ( int j = 0 ; j <= N ; j ++ ) {
if ( j >= i ) {
combo [ j ] += combo [ j - i ] ; } } }
return combo [ N ] ; }
int N = 29 ; int K = 5 ; Console . WriteLine ( solve ( N , K ) ) ; solve ( N , K ) ; } }
static int computeLIS ( int [ ] circBuff , int start , int end , int n ) { int [ ] LIS = new int [ n + end - start ] ;
for ( int i = start ; i < end ; i ++ ) LIS [ i ] = 1 ;
for ( int i = start + 1 ; i < end ; i ++ )
for ( int j = start ; j < i ; j ++ ) if ( circBuff [ i ] > circBuff [ j ] && LIS [ i ] < LIS [ j ] + 1 ) LIS [ i ] = LIS [ j ] + 1 ;
int res = int . MinValue ; for ( int i = start ; i < end ; i ++ ) res = Math . Max ( res , LIS [ i ] ) ; return res ; }
static int LICS ( int [ ] arr , int n ) {
int [ ] circBuff = new int [ 2 * n ] ; for ( int i = 0 ; i < n ; i ++ ) circBuff [ i ] = arr [ i ] ; for ( int i = n ; i < 2 * n ; i ++ ) circBuff [ i ] = arr [ i - n ] ;
int res = int . MinValue ; for ( int i = 0 ; i < n ; i ++ ) res = Math . Max ( computeLIS ( circBuff , i , i + n , n ) , res ) ; return res ; }
public static void Main ( ) { int [ ] arr = { 1 , 4 , 6 , 2 , 3 } ; Console . Write ( " Length ▁ of ▁ LICS ▁ is ▁ " + LICS ( arr , arr . Length ) ) ; } }
static int LCIS ( int [ ] arr1 , int n , int [ ] arr2 , int m ) {
int [ ] table = new int [ m ] ; for ( int j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int current = 0 ;
for ( int j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
int result = 0 ; for ( int i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
public static void Main ( ) { int [ ] arr1 = { 3 , 4 , 9 , 1 } ; int [ ] arr2 = { 5 , 3 , 8 , 9 , 10 , 2 , 1 } ; int n = arr1 . Length ; int m = arr2 . Length ; Console . Write ( " Length ▁ of ▁ LCIS ▁ is ▁ " + LCIS ( arr1 , n , arr2 , m ) ) ; } }
static String maxValue ( char [ ] a , char [ ] b ) {
Array . Sort ( b ) ; int n = a . Length ; int m = b . Length ;
int j = m - 1 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( j < 0 ) break ; if ( b [ j ] > a [ i ] ) { a [ i ] = b [ j ] ;
j -- ; } }
return String . Join ( " " , a ) ; }
public static void Main ( String [ ] args ) { String a = "1234" ; String b = "4321" ; Console . Write ( maxValue ( a . ToCharArray ( ) , b . ToCharArray ( ) ) ) ; } }
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
public static bool is_possible ( String s ) {
int l = s . Length ; int one = 0 , zero = 0 ; for ( int i = 0 ; i < l ; i ++ ) {
if ( s [ i ] == '0' ) zero ++ ;
else one ++ ; }
if ( l % 2 == 0 ) return ( one == zero ) ;
else return ( Math . Abs ( one - zero ) == 1 ) ; }
public static void Main ( String [ ] args ) { String s = "100110" ; if ( is_possible ( s ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
static String convert ( String s ) { int n = s . Length ; String s1 = " " ; s1 = s1 + Char . ToLower ( s [ 0 ] ) ; for ( int i = 1 ; i < n ; i ++ ) {
if ( s [ i ] == ' ▁ ' && i < n ) {
s1 = s1 + " ▁ " + Char . ToLower ( s [ i + 1 ] ) ; i ++ ; }
else s1 = s1 + Char . ToUpper ( s [ i ] ) ; }
return s1 ; }
public static void Main ( ) { String str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " ; Console . Write ( convert ( str ) ) ; } }
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
if ( ( n & 1 ) == 1 ) { res = res + "3" ; n = ( n - 1 ) / 2 ; }
else { res = res + "5" ; n = ( n - 2 ) / 2 ; } }
string sb = Reverse ( res ) ; return sb ; }
static void Main ( ) { int n = 5 ; Console . WriteLine ( findNthNo ( n ) ) ; } }
static int findNthNonSquare ( int n ) {
double x = ( double ) n ;
double ans = x + Math . Floor ( 0.5 + Math . Sqrt ( x ) ) ; return ( int ) ans ; }
int n = 16 ;
Console . Write ( " The ▁ " + n + " th ▁ Non - Square ▁ " + " number ▁ is ▁ " ) ; Console . Write ( findNthNonSquare ( n ) ) ; } }
static int seiresSum ( int n , int [ ] a ) { return n * ( a [ 0 ] * a [ 0 ] - a [ 2 * n - 1 ] * a [ 2 * n - 1 ] ) / ( 2 * n - 1 ) ; }
public static void Main ( ) { int n = 2 ; int [ ] a = { 1 , 2 , 3 , 4 } ; Console . WriteLine ( seiresSum ( n , a ) ) ; } }
public static bool checkdigit ( int n , int k ) { while ( n != 0 ) {
int rem = n % 10 ;
if ( rem == k ) return true ; n = n / 10 ; } return false ; }
public static int findNthNumber ( int n , int k ) {
for ( int i = k + 1 , count = 1 ; count < n ; i ++ ) {
if ( checkdigit ( i , k ) || ( i % k == 0 ) ) count ++ ; if ( count == n ) return i ; } return - 1 ; }
public static void Main ( ) { int n = 10 , k = 2 ; Console . WriteLine ( findNthNumber ( n , k ) ) ; } }
using System ; class Middle {
public static int middleOfThree ( int a , int b , int c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && < c ) || ( c < a && < b ) ) return ; else return ; }
public static void Main ( ) { int a = 20 , b = 30 , c = 40 ; Console . WriteLine ( middleOfThree ( a , b , c ) ) ; } }
using System ; class GFG { static int INF = int . MaxValue , N = 4 ;
static int minCost ( int [ , ] cost ) {
int [ ] dist = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i , j ] ) dist [ j ] = dist [ i ] + cost [ i , j ] ; return dist [ N - 1 ] ; }
public static void Main ( ) { int [ , ] cost = { { 0 , 15 , 80 , 90 } , { INF , 0 , 40 , 50 } , { INF , INF , 0 , 70 } , { INF , INF , INF , 0 } } ; Console . WriteLine ( " The ▁ Minimum ▁ cost ▁ to " + " ▁ reach ▁ station ▁ " + N + " ▁ is ▁ " + minCost ( cost ) ) ; } }
static int numOfways ( int n , int k ) { int p = 1 ; if ( k % 2 != 0 ) p = - 1 ; return ( int ) ( Math . Pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
static void Main ( ) { int n = 4 , k = 2 ; Console . Write ( numOfways ( n , k ) ) ; } }
static void length_of_chord ( double r , double x ) { Console . WriteLine ( " The ▁ length ▁ of ▁ the ▁ chord " + " ▁ of ▁ the ▁ circle ▁ is ▁ " + 2 * r * Math . Sin ( x * ( 3.14 / 180 ) ) ) ; }
public static void Main ( String [ ] args ) { double r = 4 , x = 63 ; length_of_chord ( r , x ) ; } }
static float area ( float a ) {
if ( a < 0 ) return - 1 ;
float area = ( float ) Math . Sqrt ( a ) / 6 ; return area ; }
public static void Main ( ) { float a = 10 ; Console . WriteLine ( area ( a ) ) ; } }
static double longestRodInCuboid ( int length , int breadth , int height ) { double result ; int temp ;
temp = length * length + breadth * breadth + height * height ;
result = Math . Sqrt ( temp ) ; return result ; }
public static void Main ( ) { int length = 12 , breadth = 9 , height = 8 ;
Console . WriteLine ( ( int ) longestRodInCuboid ( length , breadth , height ) ) ; } }
static bool LiesInsieRectangle ( int a , int b , int x , int y ) { if ( x - y - b <= 0 && x - y + b >= 0 && x + y - 2 * a + b <= 0 && x + y - b >= 0 ) return true ; return false ; }
public static void Main ( ) { int a = 7 , b = 2 , x = 4 , y = 5 ; if ( LiesInsieRectangle ( a , b , x , y ) ) Console . Write ( " Given ▁ point ▁ lies ▁ " + " inside ▁ the ▁ rectangle " ) ; else Console . Write ( " Given ▁ point ▁ does ▁ not ▁ " + " lie ▁ on ▁ the ▁ rectangle " ) ; } }
static int maxvolume ( int s ) { int maxvalue = 0 ;
for ( int i = 1 ; i <= s - 2 ; i ++ ) {
for ( int j = 1 ; j <= s - 1 ; j ++ ) {
int k = s - i - j ;
maxvalue = Math . Max ( maxvalue , i * j * k ) ; } } return maxvalue ; }
public static void Main ( ) { int s = 8 ; Console . WriteLine ( maxvolume ( s ) ) ; } }
static int maxvolume ( int s ) {
int length = s / 3 ; s -= length ;
int breadth = s / 2 ;
int height = s - breadth ; return length * breadth * height ; }
public static void Main ( ) { int s = 8 ; Console . WriteLine ( maxvolume ( s ) ) ; } }
public static double hexagonArea ( double s ) { return ( ( 3 * Math . Sqrt ( 3 ) * ( s * s ) ) / 2 ) ; }
double s = 4 ; Console . WriteLine ( " Area : ▁ " + hexagonArea ( s ) ) ; } }
static int maxSquare ( int b , int m ) {
return ( b / m - 1 ) * ( b / m ) / 2 ; }
public static void Main ( ) { int b = 10 , m = 2 ; Console . WriteLine ( maxSquare ( b , m ) ) ; } }
static void findRightAngle ( double A , double H ) {
double D = Math . Pow ( H , 4 ) - 16 * A * A ; if ( D >= 0 ) {
double root1 = ( H * H + Math . Sqrt ( D ) ) / 2 ; double root2 = ( H * H - Math . Sqrt ( D ) ) / 2 ; double a = Math . Sqrt ( root1 ) ; double b = Math . Sqrt ( root2 ) ; if ( b >= a ) Console . WriteLine ( a + " ▁ " + b + " ▁ " + H ) ; else Console . WriteLine ( b + " ▁ " + a + " ▁ " + H ) ; } else Console . ( " - 1" ) ; }
public static void Main ( ) { findRightAngle ( 6 , 5 ) ; } }
using System ; class GFG { public static int numberOfSquares ( int _base ) {
_base = ( _base - 2 ) ;
_base = _base / 2 ; return _base * ( _base + 1 ) / 2 ; }
public static void Main ( ) { int _base = 8 ; Console . WriteLine ( numberOfSquares ( _base ) ) ; } }
static int fib ( int n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
static int findVertices ( int n ) {
return fib ( n + 2 ) ; }
static void Main ( ) { int n = 3 ; Console . Write ( findVertices ( n ) ) ; } }
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
static int m = 3 ;
static int n = 2 ;
static long countSets ( int [ , ] a ) {
long res = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < m ; j ++ ) { if ( a [ i , j ] == 1 ) u ++ ; else v ++ ; } res += ( long ) ( Math . Pow ( 2 , u ) - 1 + Math . Pow ( 2 , v ) ) - 1 ; }
for ( int i = 0 ; i < m ; i ++ ) { int u = 0 , v = 0 ; for ( int j = 0 ; j < n ; j ++ ) { if ( a [ j , i ] == 1 ) u ++ ; else v ++ ; } res += ( long ) ( Math . Pow ( 2 , u ) - 1 + Math . Pow ( 2 , v ) ) - 1 ; }
return res - ( n * m ) ; }
public static void Main ( ) { int [ , ] a = { { 1 , 0 , 1 } , { 0 , 1 , 0 } } ; Console . WriteLine ( countSets ( a ) ) ; } }
static void transpose ( int [ , ] mat , int [ , ] tr , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) tr [ i , j ] = mat [ j , i ] ; }
static bool isSymmetric ( int [ , ] mat , int N ) { int [ , ] tr = new int [ N , MAX ] ; transpose ( mat , tr , N ) ; for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i , j ] != tr [ i , j ] ) return false ; return true ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static bool isSymmetric ( int [ , ] mat , int N ) { for ( int i = 0 ; i < N ; i ++ ) for ( int j = 0 ; j < N ; j ++ ) if ( mat [ i , j ] != mat [ j , i ] ) return false ; return true ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 3 , 5 } , { 3 , 2 , 4 } , { 5 , 4 , 1 } } ; if ( isSymmetric ( mat , 3 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " NO " ) ; } }
static int findNormal ( int [ , ] mat , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) for ( int j = 0 ; j < n ; j ++ ) sum += mat [ i , j ] * mat [ i , j ] ; return ( int ) Math . Sqrt ( sum ) ; }
static int findTrace ( int [ , ] mat , int n ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += mat [ i , i ] ; return sum ; }
public static void Main ( ) { int [ , ] mat = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } , } ; Console . Write ( " Trace ▁ of ▁ Matrix ▁ = ▁ " + findTrace ( mat , 5 ) + " STRNEWLINE " ) ; Console . Write ( " Normal ▁ of ▁ Matrix ▁ = ▁ " + findNormal ( mat , 5 ) ) ; } }
static int maxDet ( int n ) { return ( 2 * n * n * n ) ; }
void resMatrix ( int n ) { for ( int i = 0 ; i < 3 ; i ++ ) { for ( int j = 0 ; j < 3 ; j ++ ) {
if ( i == 0 && j == 2 ) Console . Write ( "0 ▁ " ) ; else if ( i == 1 && j == 0 ) Console . Write ( "0 ▁ " ) ; else if ( i == 2 && j == 1 ) Console . Write ( "0 ▁ " ) ;
else Console . ( n + " " ) ; } Console . WriteLine ( " " ) ; } }
static public void Main ( String [ ] args ) { int n = 15 ; GFG geeks = new GFG ( ) ; Console . WriteLine ( " Maximum ▁ Determinant ▁ = ▁ " + maxDet ( n ) ) ; Console . WriteLine ( " Resultant ▁ Matrix ▁ : " ) ; geeks . resMatrix ( n ) ; } }
using System ; class GFG { static int countNegative ( int [ , ] M , int n , int m ) { int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < m ; j ++ ) { if ( M [ i , j ] < 0 ) count += 1 ;
else break ; } } return count ; }
public static void Main ( ) { int [ , ] M = { { - 3 , - 2 , - 1 , 1 } , { - 2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; Console . WriteLine ( countNegative ( M , 3 , 4 ) ) ; } }
static int countNegative ( int [ , ] M , int n , int m ) {
int count = 0 ;
int i = 0 ; int j = m - 1 ;
while ( j >= 0 && i < n ) { if ( M [ i , j ] < 0 ) {
count += j + 1 ;
i += 1 ; }
else j -= 1 ; } return count ; }
public static void Main ( ) { int [ , ] M = { { - 3 , - 2 , - 1 , 1 } , { - 2 , 2 , 3 , 4 } , { 4 , 5 , 7 , 8 } } ; Console . WriteLine ( countNegative ( M , 3 , 4 ) ) ; } }
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
static int n = 5 ;
static void printSumSimple ( int [ , ] mat , int k ) {
if ( k > n ) return ;
for ( int i = 0 ; i < n - k + 1 ; i ++ ) {
for ( int j = 0 ; j < n - k + 1 ; j ++ ) {
int sum = 0 ; for ( int p = i ; p < k + i ; p ++ ) for ( int q = j ; q < k + j ; q ++ ) sum += mat [ p , q ] ; Console . Write ( sum + " ▁ " ) ; }
Console . WriteLine ( ) ; } }
public static void Main ( ) { int [ , ] mat = { { 1 , 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 , 4 } , { 5 , 5 , 5 , 5 , 5 } } ; int k = 3 ; printSumSimple ( mat , k ) ; } }
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
static void sort ( int [ ] arr ) { int n = arr . Length ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min_idx = i ; for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
int temp = arr [ min_idx ] ; arr [ min_idx ] = arr [ i ] ; arr [ i ] = temp ; } }
public static void Main ( ) { int [ ] arr = { 64 , 25 , 12 , 22 , 11 } ; sort ( arr ) ; Console . WriteLine ( " Sorted ▁ array " ) ; printArray ( arr ) ; } }
static void bubbleSort ( int [ ] arr , int n ) { int i , j , temp ; bool swapped ; for ( i = 0 ; i < n - 1 ; i ++ ) { swapped = false ; for ( j = 0 ; j < n - i - 1 ; j ++ ) { if ( arr [ j ] > arr [ j + 1 ] ) {
temp = arr [ j ] ; arr [ j ] = arr [ j + 1 ] ; arr [ j + 1 ] = temp ; swapped = true ; } }
if ( swapped == false ) break ; } }
public static void Main ( ) { int [ ] arr = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; int n = arr . Length ; bubbleSort ( arr , n ) ; Console . WriteLine ( " Sorted ▁ array " ) ; printArray ( arr , n ) ; } }
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
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int m = arr . Length ; int n = 4 ; Console . Write ( count ( arr , m , n ) ) ; } }
static int MatrixChainOrder ( int [ ] p , int n ) {
int [ , ] m = new int [ n , n ] ; int i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i , i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; if ( j == n ) continue ; m [ i , j ] = int . MaxValue ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i , k ] + m [ k + 1 , j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i , j ] ) m [ i , j ] = q ; } } } return m [ 1 , n - 1 ] ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 2 , 3 , 4 } ; int size = arr . Length ; Console . Write ( " Minimum ▁ number ▁ of ▁ " + " multiplications ▁ is ▁ " + MatrixChainOrder ( arr , size ) ) ; } }
static int cutRod ( int [ ] price , int n ) { if ( n <= 0 ) return 0 ; int max_val = int . MinValue ;
for ( int i = 0 ; i < n ; i ++ ) max_val = Math . Max ( max_val , price [ i ] + cutRod ( price , n - i - 1 ) ) ; return max_val ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . Length ; Console . WriteLine ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
static int cutRod ( int [ ] price , int n ) { int [ ] val = new int [ n + 1 ] ; val [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { int max_val = int . MinValue ; for ( int j = 0 ; j < i ; j ++ ) max_val = Math . Max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . Length ; Console . WriteLine ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
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
static int binomialCoeff ( int n , int k ) { int res = 1 ; if ( k > n - k ) k = n - k ; for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
static void printPascal ( int n ) {
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) Console . Write ( binomialCoeff ( line , i ) + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( ) { int n = 7 ; printPascal ( n ) ; } }
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
static uint getModulo ( uint n , uint d ) { return ( n & ( d - 1 ) ) ; }
static public void Main ( ) { uint n = 6 ;
uint d = 4 ; Console . WriteLine ( n + " ▁ moduo ▁ " + d + " ▁ is ▁ " + getModulo ( n , d ) ) ; } }
static int countSetBits ( int n ) { int count = 0 ; while ( n > 0 ) { count += n & 1 ; n >>= 1 ; } return count ; }
public static void Main ( ) { int i = 9 ; Console . Write ( countSetBits ( i ) ) ; } }
public static int countSetBits ( int n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
int n = 9 ;
Console . WriteLine ( countSetBits ( n ) ) ; } }
public static void Main ( ) { Console . WriteLine ( Convert . ToString ( 4 , 2 ) . Count ( c = > c == '1' ) ) ; Console . WriteLine ( Convert . ToString ( 15 , 2 ) . Count ( c = > c == '1' ) ) ; } }
class GFG { static int [ ] num_to_bits = new int [ 16 ] { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
static int countSetBitsRec ( int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
static void Main ( ) { int num = 31 ; System . Console . WriteLine ( countSetBitsRec ( num ) ) ; } }
static bool getParity ( int n ) { bool parity = false ; while ( n != 0 ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
public static void Main ( ) { int n = 7 ; Console . Write ( " Parity ▁ of ▁ no ▁ " + n + " ▁ = ▁ " + ( getParity ( n ) ? " odd " : " even " ) ) ; } }
using System ; class GFG {
static bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( int ) ( Math . Ceiling ( ( Math . Log ( n ) / Math . Log ( 2 ) ) ) ) == ( int ) ( Math . Floor ( ( ( Math . Log ( n ) / Math . Log ( 2 ) ) ) ) ) ; }
public static void Main ( ) { if ( isPowerOfTwo ( 31 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; if ( isPowerOfTwo ( 64 ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
static bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; while ( n != 1 ) { if ( n % 2 != 0 ) return false ; n = n / 2 ; } return true ; }
public static void Main ( ) { Console . WriteLine ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; Console . WriteLine ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
static bool isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void Main ( ) { Console . WriteLine ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; Console . WriteLine ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
static int maxRepeating ( int [ ] arr , int n , int k ) {
for ( int i = 0 ; i < n ; i ++ ) arr [ ( arr [ i ] % k ) ] += k ;
int max = arr [ 0 ] , result = 0 ; for ( int i = 1 ; i < n ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; result = i ; } }
return result ; }
public static void Main ( ) { int [ ] arr = { 2 , 3 , 3 , 5 , 3 , 4 , 1 , 7 } ; int n = arr . Length ; int k = 8 ; Console . Write ( " Maximum ▁ repeating ▁ " + " element ▁ is : ▁ " + maxRepeating ( arr , n , k ) ) ; } }
static int fun ( int x ) { int y = ( x / 4 ) * 4 ;
int ans = 0 ; for ( int i = y ; i <= x ; i ++ ) ans ^= i ; return ans ; }
static int query ( int x ) {
if ( x == 0 ) return 0 ; int k = ( x + 1 ) / 2 ;
return ( ( x %= 2 ) != 0 ) ? 2 * fun ( k ) : ( ( fun ( k - 1 ) * 2 ) ^ ( k & 1 ) ) ; } static void allQueries ( int q , int [ ] l , int [ ] r ) { for ( int i = 0 ; i < q ; i ++ ) Console . WriteLine ( ( query ( r [ i ] ) ^ query ( l [ i ] - 1 ) ) ) ; }
public static void Main ( ) { int q = 3 ; int [ ] l = { 2 , 2 , 5 } ; int [ ] r = { 4 , 8 , 9 } ; allQueries ( q , l , r ) ; } }
static int findMinSwaps ( int [ ] arr , int n ) {
int [ ] noOfZeroes = new int [ n ] ; int i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
public static void Main ( ) { int [ ] ar = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; Console . WriteLine ( findMinSwaps ( ar , ar . Length ) ) ; } }
static void printTwoOdd ( int [ ] arr , int size ) {
int xor2 = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( ( arr [ i ] & set_bit_no ) > 0 ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } Console . WriteLine ( " The ▁ two ▁ ODD ▁ elements ▁ are ▁ " + x + " ▁ & ▁ " + y ) ; }
public static void Main ( ) { int [ ] arr = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = arr . Length ; printTwoOdd ( arr , arr_size ) ; } }
static bool findPair ( int [ ] arr , int n ) { int size = arr . Length ;
int i = 0 , j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { Console . Write ( " Pair ▁ Found : ▁ " + " ( ▁ " + arr [ i ] + " , ▁ " + arr [ j ] + " ▁ ) " ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } Console . Write ( " No ▁ such ▁ pair " ) ; return false ; }
public static void Main ( ) { int [ ] arr = { 1 , 8 , 30 , 40 , 100 } ; int n = 60 ; findPair ( arr , n ) ; } }
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
static void findNumbers ( int [ ] arr , int n ) {
int sumN = ( n * ( n + 1 ) ) / 2 ;
int sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
int sum = 0 , sumSq = 0 , i ; for ( i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq += ( int ) Math . Pow ( arr [ i ] , 2 ) ; } int B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; int A = sum - sumN + B ; Console . WriteLine ( " A ▁ = ▁ " + A + " B = " }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 2 , 3 , 4 } ; int n = arr . Length ; findNumbers ( arr , n ) ; } }
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
public static int middleOfThree ( int a , int b , int c ) {
int x = a - b ;
int y = b - c ;
int z = a - c ;
if ( x * y > 0 ) return b ;
else if ( x * z > 0 ) return c ; else return a ; }
public static void Main ( ) { int a = 20 , b = 30 , c = 40 ; Console . WriteLine ( middleOfThree ( a , b , c ) ) ; } }
public static void missing4 ( int [ ] arr ) {
int [ ] helper = new int [ 4 ] ;
for ( int i = 0 ; i < arr . Length ; i ++ ) { int temp = Math . Abs ( arr [ i ] ) ;
if ( temp <= arr . Length ) arr [ temp - 1 ] *= ( - 1 ) ;
else if ( temp > arr . Length ) { if ( temp % arr . Length != 0 ) helper [ temp % arr . Length - 1 ] = - 1 ; else helper [ ( temp % arr . Length ) + arr . Length - 1 ] = - 1 ; } }
for ( int i = 0 ; i < arr . Length ; i ++ ) if ( arr [ i ] > 0 ) Console . Write ( i + 1 + " ▁ " ) ; for ( int i = 0 ; i < helper . Length ; i ++ ) if ( helper [ i ] >= 0 ) Console . Write ( arr . Length + i + 1 + " ▁ " ) ; return ; }
public static void Main ( ) { int [ ] arr = { 1 , 7 , 3 , 12 , 5 , 10 , 8 , 4 , 9 } ; missing4 ( arr ) ; } }
static int minMovesToSort ( int [ ] arr , int n ) { int moves = 0 ; int i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
} return moves ; }
static public void Main ( ) { int [ ] arr = { 3 , 5 , 2 , 8 , 4 } ; int n = arr . Length ; Console . WriteLine ( minMovesToSort ( arr , n ) ) ; } }
using System ; public class GFG { static void findOptimalPairs ( int [ ] arr , int N ) { Array . Sort ( arr ) ;
for ( int i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) Console . Write ( " ( " + arr [ i ] + " , ▁ " + arr [ j ] + " ) " + " ▁ " ) ; }
static public void Main ( ) { int [ ] arr = { 9 , 6 , 5 , 1 } ; int N = arr . Length ; findOptimalPairs ( arr , N ) ; } }
static int minOperations ( int [ ] arr , int n ) { int maxi , result = 0 ;
int [ ] freq = new int [ 1000001 ] ; for ( int i = 0 ; i < n ; i ++ ) { int x = arr [ i ] ; freq [ x ] ++ ; }
maxi = arr . Max ( ) ; for ( int i = 1 ; i <= maxi ; i ++ ) { if ( freq [ i ] != 0 ) {
for ( int j = i * 2 ; j <= maxi ; j = j + i ) {
freq [ j ] = 0 ; }
result ++ ; } } return result ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 2 , 4 , 2 , 4 , 4 , 4 } ; int n = arr . Length ; Console . WriteLine ( minOperations ( arr , n ) ) ; } }
static int __gcd ( int a , int b ) { if ( a == 0 ) return b ; return __gcd ( b % a , a ) ; } static int minGCD ( int [ ] arr , int n ) { int minGCD = 0 ;
for ( int i = 0 ; i < n ; i ++ ) minGCD = __gcd ( minGCD , arr [ i ] ) ; return minGCD ; }
static int minLCM ( int [ ] arr , int n ) { int minLCM = arr [ 0 ] ;
for ( int i = 1 ; i < n ; i ++ ) minLCM = Math . Min ( minLCM , arr [ i ] ) ; return minLCM ; }
public static void Main ( ) { int [ ] arr = { 2 , 66 , 14 , 521 } ; int n = arr . Length ; Console . WriteLine ( " LCM ▁ = ▁ " + minLCM ( arr , n ) + " , ▁ GCD ▁ = ▁ " + minGCD ( arr , n ) ) ; } }
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
public static int findWaysToPair ( int p ) {
int [ ] dp = new int [ p + 1 ] ; dp [ 1 ] = 1 ; dp [ 2 ] = 2 ;
for ( int i = 3 ; i <= p ; i ++ ) { dp [ i ] = dp [ i - 1 ] + ( i - 1 ) * dp [ i - 2 ] ; } return dp [ p ] ; }
public static void Main ( string [ ] args ) { int p = 3 ; Console . WriteLine ( findWaysToPair ( p ) ) ; } }
using System ; class GFG { static int CountWays ( int n ) {
if ( n == 0 ) { return 1 ; } if ( n == 1 ) { return 1 ; } if ( n == 2 ) { return 1 + 1 ; }
return CountWays ( n - 1 ) + CountWays ( n - 3 ) ; }
static public void Main ( ) { int n = 5 ; Console . WriteLine ( CountWays ( n ) ) ; } }
static int maxSubArraySumRepeated ( int [ ] a , int n , int k ) { int max_so_far = 0 ; int max_ending_here = 0 ; for ( int i = 0 ; i < n * k ; i ++ ) {
max_ending_here = max_ending_here + a [ i % n ] ; if ( max_so_far < max_ending_here ) max_so_far = max_ending_here ; if ( max_ending_here < 0 ) max_ending_here = 0 ; } return max_so_far ; }
public static void Main ( ) { int [ ] a = { 10 , 20 , - 30 , - 1 } ; int n = a . Length ; int k = 3 ; Console . Write ( " Maximum ▁ contiguous ▁ sum ▁ is ▁ " + maxSubArraySumRepeated ( a , n , k ) ) ; } }
public static int longOddEvenIncSeq ( int [ ] arr , int n ) {
int [ ] lioes = new int [ n ] ;
int maxLen = 0 ;
for ( int i = 0 ; i < n ; i ++ ) lioes [ i ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) for ( int j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && ( arr [ i ] + arr [ j ] ) % 2 != 0 && lioes [ i ] < lioes [ j ] + 1 ) lioes [ i ] = lioes [ j ] + 1 ;
for ( int i = 0 ; i < n ; i ++ ) if ( maxLen < lioes [ i ] ) maxLen = lioes [ i ] ;
return maxLen ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 12 , 2 , 22 , 5 , 30 , 31 , 14 , 17 , 11 } ; int n = 10 ; Console . Write ( " Longest ▁ Increasing ▁ Odd " + " ▁ Even ▁ Subsequence : ▁ " + longOddEvenIncSeq ( arr , n ) ) ; } }
static int MatrixChainOrder ( int [ ] p , int i , int j ) { if ( i == j ) return 0 ; int min = int . MaxValue ;
for ( int k = i ; k < j ; k ++ ) { int count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 2 , 3 , 4 , 3 } ; int n = arr . Length ; Console . Write ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ; } }
static int getCount ( String a , String b ) {
if ( b . Length % a . Length != 0 ) return - 1 ; int count = b . Length / a . Length ;
String str = " " ; for ( int i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str . Equals ( b ) ) return count ; return - 1 ; }
public static void Main ( String [ ] args ) { String a = " geeks " ; String b = " geeksgeeks " ; Console . WriteLine ( getCount ( a , b ) ) ; } }
public static int countPattern ( string str ) { int len = str . Length ; bool oneSeen = false ;
for ( int i = 0 ; i < len ; i ++ ) { char getChar = str [ i ] ;
if ( getChar == '1' && oneSeen == true ) { if ( str [ i - 1 ] == '0' ) { count ++ ; } }
if ( getChar == '1' && oneSeen == false ) { oneSeen = true ; }
if ( getChar != '0' && str [ i ] != '1' ) { oneSeen = false ; } } return count ; }
public static void Main ( string [ ] args ) { string str = "100001abc101" ; Console . WriteLine ( countPattern ( str ) ) ; } }
static int minOperations ( string s , string t , int n ) { int ct0 = 0 , ct1 = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( s [ i ] == t [ i ] ) continue ;
if ( s [ i ] == '0' ) ct0 ++ ;
else ct1 ++ ; } return Math . Max ( ct0 , ct1 ) ; }
public static void Main ( ) { string s = "010" , t = "101" ; int n = s . Length ; Console . Write ( minOperations ( s , t , n ) ) ; } }
static string decryptString ( string str , int n ) {
int i = 0 , jump = 1 ; string decryptedStr = " " ; while ( i < n ) { decryptedStr += str [ i ] ; i += jump ;
jump ++ ; } return decryptedStr ; }
public static void Main ( ) { string str = " geeeeekkkksssss " ; int n = str . Length ; Console . Write ( decryptString ( str , n ) ) ; } }
static char bitToBeFlipped ( String s ) {
char last = s [ s . Length - 1 ] ; char first = s [ 0 ] ;
if ( last == first ) { if ( last == '0' ) { return '1' ; } else { return '0' ; } }
else if ( last != first ) { return last ; } return last ; }
public static void Main ( ) { string s = "1101011000" ; Console . WriteLine ( bitToBeFlipped ( s ) ) ; } }
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
using System ; class GFG { static int countOccurrences ( string str , string word ) {
string [ ] a = str . Split ( ' ▁ ' ) ;
int count = 0 ; for ( int i = 0 ; i < a . Length ; i ++ ) {
if ( word . Equals ( a [ i ] ) ) count ++ ; } return count ; }
public static void Main ( ) { string str = " GeeksforGeeks ▁ A ▁ computer ▁ science ▁ portal ▁ for ▁ geeks ▁ " ; string word = " portal " ; Console . Write ( countOccurrences ( str , word ) ) ; } }
static void permute ( String input ) { int n = input . Length ;
int max = 1 << n ;
input = input . ToLower ( ) ;
for ( int i = 0 ; i < max ; i ++ ) { char [ ] combination = input . ToCharArray ( ) ;
for ( int j = 0 ; j < n ; j ++ ) { if ( ( ( i >> j ) & 1 ) == 1 ) combination [ j ] = ( char ) ( combination [ j ] - 32 ) ; }
Console . Write ( combination ) ; Console . Write ( " ▁ " ) ; } }
public static void Main ( ) { permute ( " ABC " ) ; } }
static bool isPalindrome ( String str ) {
int l = 0 ; int h = str . Length - 1 ;
while ( h > l ) if ( str [ l ++ ] != str [ h -- ] ) return false ; return true ; }
static int minRemovals ( String str ) {
if ( str [ 0 ] == ' ' ) return 0 ;
if ( isPalindrome ( str ) ) return 1 ;
return 2 ; }
public static void Main ( ) { Console . WriteLine ( minRemovals ( "010010" ) ) ; Console . WriteLine ( minRemovals ( "0100101" ) ) ; } }
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
static void lengtang ( double r1 , double r2 , double d ) { Console . WriteLine ( " The ▁ length ▁ of ▁ the ▁ direct " + " ▁ common ▁ tangent ▁ is ▁ " + ( Math . Sqrt ( Math . Pow ( d , 2 ) - Math . Pow ( ( r1 - r2 ) , 2 ) ) ) ) ; }
public static void Main ( String [ ] args ) { double r1 = 4 , r2 = 6 , d = 3 ; lengtang ( r1 , r2 , d ) ; } }
static void rad ( double d , double h ) { Console . WriteLine ( " The ▁ radius ▁ of ▁ the ▁ circle ▁ is ▁ " + ( ( d * d ) / ( 8 * h ) + h / 2 ) ) ; }
public static void Main ( ) { double d = 4 , h = 1 ; rad ( d , h ) ; } }
static void shortdis ( double r , double d ) { Console . WriteLine ( " The ▁ shortest ▁ distance ▁ " + " from ▁ the ▁ chord ▁ to ▁ centre ▁ " + ( Math . Sqrt ( ( r * r ) - ( ( d * d ) / 4 ) ) ) ) ; }
public static void Main ( ) { double r = 4 , d = 3 ; shortdis ( r , d ) ; } }
static void lengtang ( double r1 , double r2 , double d ) { Console . WriteLine ( " The ▁ length ▁ of ▁ the ▁ direct " + " ▁ common ▁ tangent ▁ is ▁ " + ( Math . Sqrt ( Math . Pow ( d , 2 ) - Math . Pow ( ( r1 - r2 ) , 2 ) ) ) ) ; }
public static void Main ( ) { double r1 = 4 , r2 = 6 , d = 12 ; lengtang ( r1 , r2 , d ) ; } }
static double square ( double a ) {
if ( a < 0 ) return - 1 ;
double x = 0.464 * a ; return x ; }
public static void Main ( ) { double a = 5 ; Console . WriteLine ( square ( a ) ) ; } }
static double polyapothem ( double n , double a ) {
if ( a < 0 && n < 0 ) return - 1 ;
return ( a / ( 2 * Math . Tan ( ( 180 / n ) * 3.14159 / 180 ) ) ) ; }
public static void Main ( ) { double a = 9 , n = 6 ; Console . WriteLine ( Math . Round ( polyapothem ( n , a ) , 4 ) ) ; } }
static float polyarea ( float n , float a ) {
if ( a < 0 && n < 0 ) return - 1 ;
float A = ( a * a * n ) / ( float ) ( 4 * Math . Tan ( ( 180 / n ) * 3.14159 / 180 ) ) ; return A ; }
public static void Main ( ) { float a = 9 , n = 6 ; Console . WriteLine ( polyarea ( n , a ) ) ; } }
static double calculateSide ( double n , double r ) { double theta , theta_in_radians ; theta = 360 / n ; theta_in_radians = theta * 3.14 / 180 ; return Math . Round ( 2 * r * Math . Sin ( theta_in_radians / 2 ) , 4 ) ; }
double n = 3 ;
double r = 5 ; Console . WriteLine ( calculateSide ( n , r ) ) ; } }
static float cyl ( float r , float R , float h ) {
if ( h < 0 && r < 0 && R < 0 ) return - 1 ;
float r1 = r ;
float h1 = h ;
float V = ( float ) ( 3.14 * Math . Pow ( r1 , 2 ) * h1 ) ; return V ; }
public static void Main ( ) { float r = 7 , R = 11 , h = 6 ; Console . WriteLine ( cyl ( r , R , h ) ) ; } }
static double Perimeter ( double s , int n ) { double perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
int n = 5 ;
double s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; Console . WriteLine ( " Perimeter ▁ of ▁ Regular ▁ Polygon " + " ▁ with ▁ " + n + " ▁ sides ▁ of ▁ length ▁ " + s + " ▁ = ▁ " + peri ) ; } }
static float rhombusarea ( float l , float b ) {
if ( l < 0 b < 0 ) return - 1 ;
return ( l * b ) / 2 ; }
public static void Main ( ) { float l = 16 , b = 6 ; Console . WriteLine ( rhombusarea ( l , b ) ) ; } }
static bool FindPoint ( int x1 , int y1 , int x2 , int y2 , int x , int y ) { if ( x > x1 && x < x2 && y > y1 && y < y2 ) return true ; return false ; }
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x = 1 , y = 5 ;
if ( FindPoint ( x1 , y1 , x2 , y2 , x , y ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
static void shortest_distance ( float x1 , float y1 , float z1 , float a , float b , float c , float d ) { d = Math . Abs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; float e = ( float ) Math . Sqrt ( a * a + b * b + c * c ) ; Console . Write ( " Perpendicular ▁ distance ▁ " + " is ▁ " + d / e ) ; }
public static void Main ( ) { float x1 = 4 ; float y1 = - 4 ; float z1 = 3 ; float a = 2 ; float b = - 2 ; float c = 5 ; float d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ; } }
static float findVolume ( float l , float b , float h ) {
float volume = ( l * b * h ) / 2 ; return volume ; }
static public void Main ( ) { float l = 18 , b = 12 , h = 9 ;
Console . WriteLine ( " Volume ▁ of ▁ triangular ▁ prism : ▁ " + findVolume ( l , b , h ) ) ; } }
static void midpoint ( int x1 , int x2 , int y1 , int y2 ) { Console . WriteLine ( ( x1 + x2 ) / 2 + " ▁ , ▁ " + ( y1 + y2 ) / 2 ) ; }
public static void Main ( ) { int x1 = - 1 , y1 = 2 ; int x2 = 3 , y2 = - 6 ; midpoint ( x1 , x2 , y1 , y2 ) ; } }
static double arcLength ( double diameter , double angle ) { double pi = 22.0 / 7.0 ; double arc ; if ( angle >= 360 ) { Console . WriteLine ( " Angle ▁ cannot " + " ▁ be ▁ formed " ) ; return 0 ; } else { arc = ( pi * diameter ) * ( angle / 360.0 ) ; return arc ; } }
public static void Main ( ) { double diameter = 25.0 ; double angle = 45.0 ; double arc_len = arcLength ( diameter , angle ) ; Console . WriteLine ( arc_len ) ; } }
using System ; class GFG { static void checkCollision ( int a , int b , int c , int x , int y , int radius ) {
double dist = ( Math . Abs ( a * x + b * y + c ) ) / Math . Sqrt ( a * a + b * b ) ;
if ( radius == dist ) Console . WriteLine ( " Touch " ) ; else if ( radius > dist ) Console . WriteLine ( " Intersect " ) ; else Console . WriteLine ( " Outside " ) ; }
public static void Main ( ) { int radius = 5 ; int x = 0 , y = 0 ; int a = 3 , b = 4 , c = 25 ; checkCollision ( a , b , c , x , y , radius ) ; } }
static double polygonArea ( double [ ] X , double [ ] Y , int n ) {
double area = 0.0 ;
int j = n - 1 ; for ( int i = 0 ; i < n ; i ++ ) { area += ( X [ j ] + X [ i ] ) * ( Y [ j ] - Y [ i ] ) ;
j = i ; }
return Math . Abs ( area / 2.0 ) ; }
public static void Main ( ) { double [ ] X = { 0 , 2 , 4 } ; double [ ] Y = { 1 , 3 , 7 } ; int n = X . Length ; Console . WriteLine ( polygonArea ( X , Y , n ) ) ; } }
static int getAverage ( int x , int y ) {
int avg = ( x & y ) + ( ( x ^ y ) >> 1 ) ; return avg ; }
public static void Main ( ) { int x = 10 , y = 9 ; Console . WriteLine ( getAverage ( x , y ) ) ; } }
static int smallestIndex ( int [ ] a , int n ) {
int right1 = 0 , right0 = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( a [ i ] == 1 ) right1 = i ;
else right0 = i ; }
return Math . Min ( right1 , right0 ) ; }
public static void Main ( ) { int [ ] a = { 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int n = a . Length ; Console . Write ( smallestIndex ( a , n ) ) ; } }
static int countSquares ( int r , int c , int m ) {
int squares = 0 ;
for ( int i = 1 ; i <= 8 ; i ++ ) { for ( int j = 1 ; j <= 8 ; j ++ ) {
if ( Math . Max ( Math . Abs ( i - r ) , Math . Abs ( j - c ) ) <= m ) squares ++ ; } }
return squares ; }
public static void Main ( ) { int r = 4 , c = 4 , m = 1 ; Console . Write ( countSquares ( r , c , m ) ) ; } }
static int countNumbers ( int L , int R , int K ) { if ( K == 9 ) { K = 0 ; }
int totalnumbers = R - L + 1 ;
int factor9 = totalnumbers / 9 ;
int rem = totalnumbers % 9 ;
int ans = factor9 ;
for ( int i = R ; i > R - rem ; i -- ) { int rem1 = i % 9 ; if ( rem1 == K ) { ans ++ ; } } return ans ; }
public static void Main ( ) { int L = 10 ; int R = 22 ; int K = 3 ; Console . WriteLine ( countNumbers ( L , R , K ) ) ; } }
static void BalanceArray ( int [ ] A , int [ , ] Q ) { int [ ] ANS = new int [ A . Length ] ; int i , sum = 0 ; for ( i = 0 ; i < A . Length ; i ++ )
if ( A [ i ] % 2 == 0 ) sum = sum + A [ i ] ; for ( i = 0 ; i < Q . GetLength ( 0 ) ; i ++ ) { int index = Q [ i , 0 ] ; int value = Q [ i , 1 ] ;
if ( A [ index ] % 2 == 0 ) sum = sum - A [ index ] ; A [ index ] = A [ index ] + value ;
if ( A [ index ] % 2 == 0 ) sum = sum + A [ index ] ;
ANS [ i ] = sum ; }
for ( i = 0 ; i < ANS . Length ; i ++ ) Console . Write ( ANS [ i ] + " ▁ " ) ; }
public static void Main ( ) { int [ ] A = { 1 , 2 , 3 , 4 } ; int [ , ] Q = { { 0 , 1 } , { 1 , - 3 } , { 0 , - 4 } , { 3 , 2 } } ; BalanceArray ( A , Q ) ; } }
static int Cycles ( int N ) { int fact = 1 , result = 0 ; result = N - 1 ;
int i = result ; while ( i > 0 ) { fact = fact * i ; i -- ; } return fact / 2 ; }
public static void Main ( ) { int N = 5 ; int Number = Cycles ( N ) ; Console . Write ( " Hamiltonian ▁ cycles ▁ = ▁ " + Number ) ; } }
static bool digitWell ( int n , int m , int k ) { int cnt = 0 ; while ( n > 0 ) { if ( n % 10 == m ) ++ cnt ; n /= 10 ; } return cnt == k ; }
static int findInt ( int n , int m , int k ) { int i = n + 1 ; while ( true ) { if ( digitWell ( i , m , k ) ) return i ; i ++ ; } }
public static void Main ( ) { int n = 111 , m = 2 , k = 2 ; Console . WriteLine ( findInt ( n , m , k ) ) ; } }
static int countOdd ( int [ ] arr , int n ) {
int odd = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] % 2 == 1 ) odd ++ ; } return odd ; }
static int countValidPairs ( int [ ] arr , int n ) { int odd = countOdd ( arr , n ) ; return ( odd * ( odd - 1 ) ) / 2 ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 } ; int n = arr . Length ; Console . WriteLine ( countValidPairs ( arr , n ) ) ; } }
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
static int countDigitsToBeRemoved ( int N , int K ) {
string s = Convert . ToString ( N ) ;
int res = 0 ;
int f_zero = 0 ; for ( int i = s . Length - 1 ; i >= 0 ; i -- ) { if ( K == 0 ) return res ; if ( s [ i ] == '0' ) {
f_zero = 1 ; K -- ; } else res ++ ; }
if ( K == 0 ) return res ; else if ( f_zero == 1 ) return s . Length - 1 ; return - 1 ; }
public static void Main ( ) { int N = 10904025 ; int K = 2 ; Console . Write ( countDigitsToBeRemoved ( N , K ) + " STRNEWLINE " ) ; N = 1000 ; K = 5 ; Console . Write ( countDigitsToBeRemoved ( N , K ) + " STRNEWLINE " ) ; N = 23985 ; K = 2 ; Console . Write ( countDigitsToBeRemoved ( N , K ) + " STRNEWLINE " ) ; } }
public static double getSum ( int a , int n ) {
double sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) {
sum += ( i / Math . Pow ( a , i ) ) ; } return sum ; }
static public void Main ( ) { int a = 3 , n = 3 ;
Console . WriteLine ( getSum ( a , n ) ) ; } }
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
static void isHalfReducible ( int [ ] arr , int n , int m ) { int [ ] frequencyHash = new int [ m + 1 ] ; int i ; for ( i = 0 ; i < frequencyHash . Length ; i ++ ) frequencyHash [ i ] = 0 ; for ( i = 0 ; i < n ; i ++ ) { frequencyHash [ arr [ i ] % ( m + 1 ) ] ++ ; } for ( i = 0 ; i <= m ; i ++ ) { if ( frequencyHash [ i ] >= n / 2 ) break ; } if ( i <= m ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; }
public static void Main ( ) { int [ ] arr = { 8 , 16 , 32 , 3 , 12 } ; int n = arr . Length ; int m = 7 ; isHalfReducible ( arr , n , m ) ; } }
static bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) { if ( n % i == 0 || n % ( i + 2 ) == 0 ) { return false ; } } return true ; }
static bool isPowerOfTwo ( int n ) { return n != 0 && ( ( n & ( n - 1 ) ) == 0 ) ; }
public static void Main ( ) { int n = 43 ;
if ( isPrime ( n ) && ( isPowerOfTwo ( n * 3 - 1 ) ) ) { Console . WriteLine ( " YES " ) ; } else { Console . WriteLine ( " NO " ) ; } } }
static float area ( float a ) {
if ( a < 0 ) return - 1 ;
float area = ( float ) Math . Pow ( ( a * Math . Sqrt ( 3 ) ) / ( Math . Sqrt ( 2 ) ) , 2 ) ; return area ; }
public static void Main ( ) { float a = 5 ; Console . WriteLine ( area ( a ) ) ; } }
static int nthTerm ( int n ) { return 3 * ( int ) Math . Pow ( n , 2 ) - 4 * n + 2 ; }
public static void Main ( ) { int N = 4 ; Console . Write ( nthTerm ( N ) ) ; } }
public void calculateSum ( int n ) { double r = ( n * ( n + 1 ) / 2 + Math . Pow ( ( n * ( n + 1 ) / 2 ) , 2 ) ) ; Console . WriteLine ( " Sum ▁ = ▁ " + r ) ; }
int n = 3 ;
g . calculateSum ( n ) ; Console . Read ( ) ; return 0 ; } }
static bool arePermutations ( int [ ] a , int [ ] b , int n , int m ) { int sum1 = 0 , sum2 = 0 , mul1 = 1 , mul2 = 1 ;
for ( int i = 0 ; i < n ; i ++ ) { sum1 += a [ i ] ; mul1 *= a [ i ] ; }
for ( int i = 0 ; i < m ; i ++ ) { sum2 += b [ i ] ; mul2 *= b [ i ] ; }
return ( ( sum1 == sum2 ) && ( mul1 == mul2 ) ) ; }
public static void Main ( ) { int [ ] a = { 1 , 3 , 2 } ; int [ ] b = { 3 , 1 , 2 } ; int n = a . Length ; int m = b . Length ; if ( arePermutations ( a , b , n , m ) == true ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
static int Race ( int B , int C ) { int result = 0 ;
result = ( ( C * 100 ) / B ) ; return 100 - result ; }
public static void Main ( ) { int B = 10 ; int C = 28 ;
B = 100 - B ; C = 100 - C ; Console . Write ( Race ( B , C ) + " ▁ meters " ) ; } }
static float Time ( float [ ] arr , int n , float Emptypipe ) { float fill = 0 ; for ( int i = 0 ; i < n ; i ++ ) fill += 1 / arr [ i ] ; fill = fill - ( 1 / ( float ) Emptypipe ) ; return 1 / fill ; }
public static void Main ( ) { float [ ] arr = { 12 , 14 } ; float Emptypipe = 30 ; int n = arr . Length ; Console . WriteLine ( ( int ) ( Time ( arr , n , Emptypipe ) ) + " ▁ Hours " ) ; } }
static int check ( int n ) { int sum = 0 ;
while ( n != 0 ) { sum += n % 10 ; n = n / 10 ; }
if ( sum % 7 == 0 ) return 1 ; else return 0 ; }
int n = 25 ; String s = ( check ( n ) == 1 ) ? " YES " : " NO " ; Console . WriteLine ( s ) ; } }
using System ; class GFG {
static bool isPrime ( int n ) {
if ( n <= 1 ) return false ; if ( n <= 3 ) return true ;
if ( n % 2 == 0 n % 3 == 0 ) return false ; for ( int i = 5 ; i * i <= n ; i = i + 6 ) if ( n % i == 0 || n % ( i + 2 ) == 0 ) return false ; return true ; }
static int SumOfPrimeDivisors ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) { if ( n % i == 0 ) { if ( isPrime ( i ) ) sum += i ; } } return sum ; }
public static void Main ( ) { int n = 60 ; Console . WriteLine ( " Sum ▁ of ▁ prime ▁ divisors ▁ of ▁ 60 ▁ is ▁ " + SumOfPrimeDivisors ( n ) + " STRNEWLINE " ) ; } }
static int Sum ( int N ) { int [ ] SumOfPrimeDivisors = new int [ N + 1 ] ; for ( int i = 2 ; i <= N ; ++ i ) {
if ( SumOfPrimeDivisors [ i ] == 0 ) {
for ( int j = i ; j <= N ; j += i ) { SumOfPrimeDivisors [ j ] += i ; } } } return SumOfPrimeDivisors [ N ] ; }
public static void Main ( ) { int N = 60 ; Console . Write ( " Sum ▁ of ▁ prime ▁ " + " divisors ▁ of ▁ 60 ▁ is ▁ " + Sum ( N ) + " STRNEWLINE " ) ; } }
static long power ( long x , long y , long p ) {
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) > 0 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
public static void Main ( ) { long a = 3 ;
string b = "100000000000000000000000000" ; long remainderB = 0 ; long MOD = 1000000007 ;
for ( int i = 0 ; i < b . Length ; i ++ ) remainderB = ( remainderB * 10 + b [ i ] - '0' ) % ( MOD - 1 ) ; Console . WriteLine ( power ( a , remainderB , MOD ) ) ; } }
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
static int fact ( int n ) { if ( n == 0 n == 1 ) return 1 ; int ans = 1 ; for ( int i = 1 ; i <= n ; i ++ ) ans = ans * i ; return ans ; }
static int nCr ( int n , int r ) { int Nr = n , Dr = 1 , ans = 1 ; for ( int i = 1 ; i <= r ; i ++ ) { ans = ( ans * Nr ) / ( Dr ) ; Nr -- ; Dr ++ ; } return ans ; }
static int solve ( int n ) { int N = 2 * n - 2 ; int R = n - 1 ; return nCr ( N , R ) * fact ( n - 1 ) ; }
public static void Main ( ) { int n = 6 ; Console . WriteLine ( solve ( n ) ) ; } }
using System ; class GFG { static void pythagoreanTriplet ( int n ) {
for ( int i = 1 ; i <= n / 3 ; i ++ ) {
for ( int j = i + 1 ; j <= n / 2 ; j ++ ) { int k = n - i - j ; if ( i * i + j * j == k * k ) { Console . Write ( i + " , ▁ " + j + " , ▁ " + k ) ; return ; } } } Console . Write ( " No ▁ Triplet " ) ; }
public static void Main ( ) { int n = 12 ; pythagoreanTriplet ( n ) ; } }
static int factorial ( int n ) { int f = 1 ; for ( int i = 2 ; i <= n ; i ++ ) f *= i ; return f ; }
static void series ( int A , int X , int n ) {
int nFact = factorial ( n ) ;
for ( int i = 0 ; i < n + 1 ; i ++ ) {
int niFact = factorial ( n - i ) ; int iFact = factorial ( i ) ;
int aPow = ( int ) Math . Pow ( A , n - i ) ; int xPow = ( int ) Math . Pow ( X , i ) ;
Console . Write ( ( nFact * aPow * xPow ) / ( niFact * iFact ) + " ▁ " ) ; } }
public static void Main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; } }
static int seiresSum ( int n , int [ ] a ) { int res = 0 , i ; for ( i = 0 ; i < 2 * n ; i ++ ) { if ( i % 2 == 0 ) res += a [ i ] * a [ i ] ; else res -= a [ i ] * a [ i ] ; } return res ; }
public static void Main ( ) { int n = 2 ; int [ ] a = { 1 , 2 , 3 , 4 } ; Console . WriteLine ( seiresSum ( n , a ) ) ; } }
static int power ( int n , int r ) {
int count = 0 ; for ( int i = r ; ( n / i ) >= 1 ; i = i * r ) count += n / i ; return count ; }
public static void Main ( ) { int n = 6 , r = 3 ; Console . WriteLine ( power ( n , r ) ) ; } }
static int avg_of_odd_num ( int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += ( 2 * i + 1 ) ;
return sum / n ; }
public static void Main ( ) { int n = 20 ; avg_of_odd_num ( n ) ; Console . Write ( avg_of_odd_num ( n ) ) ; } }
static int avg_of_odd_num ( int n ) { return n ; }
public static void Main ( ) { int n = 8 ; Console . Write ( avg_of_odd_num ( n ) ) ; } }
static void fib ( int [ ] f , int N ) {
f [ 1 ] = 1 ; f [ 2 ] = 1 ; for ( int i = 3 ; i <= N ; i ++ )
f [ i ] = f [ i - 1 ] + f [ i - 2 ] ; } static void fiboTriangle ( int n ) {
int N = n * ( n + 1 ) / 2 ; int [ ] f = new int [ N + 1 ] ; fib ( f , N ) ;
int fiboNum = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = 1 ; j <= i ; j ++ ) Console . Write ( f [ fiboNum ++ ] + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( ) { int n = 5 ; fiboTriangle ( n ) ; } }
static int averageOdd ( int n ) { if ( n % 2 == 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } int sum = 0 , count = 0 ; while ( n >= 1 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
public static void Main ( ) { int n = 15 ; Console . Write ( averageOdd ( n ) ) ; } }
static int averageOdd ( int n ) { if ( n % 2 == 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } return ( n + 1 ) / 2 ; }
public static void Main ( ) { int n = 15 ; Console . Write ( averageOdd ( n ) ) ; } }
public static int TrinomialValue ( int n , int k ) {
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return TrinomialValue ( n - 1 , k - 1 ) + TrinomialValue ( n - 1 , k ) + TrinomialValue ( n - 1 , k + 1 ) ; }
public static void printTrinomial ( int n ) {
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = - i ; j <= 0 ; j ++ ) Console . Write ( TrinomialValue ( i , j ) + " ▁ " ) ;
for ( int j = 1 ; j <= i ; j ++ ) Console . Write ( TrinomialValue ( i , j ) + " ▁ " ) ; Console . WriteLine ( ) ; } }
public static void Main ( ) { int n = 4 ; printTrinomial ( n ) ; } }
using System ; class GFG { private static int MAX = 10 ;
public static int TrinomialValue ( int [ , ] dp , int n , int k ) {
if ( k < 0 ) k = - k ;
if ( dp [ n , k ] != 0 ) return dp [ n , k ] ;
if ( n == 0 && k == 0 ) return 1 ;
if ( k < - n k > n ) return 0 ;
return ( dp [ n , k ] = TrinomialValue ( dp , n - 1 , k - 1 ) + TrinomialValue ( dp , n - 1 , k ) + TrinomialValue ( dp , n - 1 , k + 1 ) ) ; }
public static void printTrinomial ( int n ) { int [ , ] dp = new int [ MAX , MAX ] ;
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = - i ; j <= 0 ; j ++ ) Console . Write ( TrinomialValue ( dp , i , j ) + " ▁ " ) ;
for ( int j = 1 ; j <= i ; j ++ ) Console . Write ( TrinomialValue ( dp , i , j ) + " ▁ " ) ; Console . WriteLine ( ) ; } }
static public void Main ( ) { int n = 4 ; printTrinomial ( n ) ; } }
static int sumOfLargePrimeFactor ( int n ) {
int [ ] prime = new int [ n + 1 ] ; int sum = 0 ; for ( int i = 1 ; i < n + 1 ; i ++ ) prime [ i ] = 0 ; int max = n / 2 ; for ( int p = 2 ; p <= max ; p ++ ) {
if ( prime [ p ] == 0 ) {
for ( int i = p * 2 ; i <= n ; i += p ) prime [ i ] = p ; } }
for ( int p = 2 ; p <= n ; p ++ ) {
if ( prime [ p ] != 0 ) sum += prime [ p ] ;
else sum += ; }
return sum ; }
public static void Main ( ) { int n = 12 ; Console . WriteLine ( " Sum ▁ = ▁ " + sumOfLargePrimeFactor ( n ) ) ; } }
static int calculate_sum ( int a , int N ) {
int m = N / a ;
int sum = m * ( m + 1 ) / 2 ;
int ans = a * sum ; return ans ; }
public static void Main ( ) { int a = 7 , N = 49 ; Console . WriteLine ( " Sum ▁ of ▁ multiples ▁ of ▁ " + a + " ▁ up ▁ to ▁ " + N + " ▁ = ▁ " + calculate_sum ( a , N ) ) ; } }
static int isPowerOf2 ( string s ) { char [ ] str = s . ToCharArray ( ) ; int len_str = str . Length ;
int num = 0 ;
if ( len_str == 1 && str [ len_str - 1 ] == '1' ) return 0 ;
while ( len_str != 1 str [ len_str - 1 ] != '1' ) {
if ( ( str [ len_str - 1 ] - '0' ) % 2 == 1 ) return 0 ;
int j = 0 ; for ( int i = 0 ; i < len_str ; i ++ ) { num = num * 10 + ( int ) str [ i ] - ( int ) '0' ;
if ( num < 2 ) {
if ( i != 0 ) str [ j ++ ] = '0' ;
continue ; } str [ j ++ ] = ( char ) ( ( int ) ( num / 2 ) + ( int ) '0' ) ; num = ( num ) - ( num / 2 ) * 2 ; } str [ j ] = ' \0' ;
len_str = j ; }
return 1 ; }
static void Main ( ) { string str1 = "124684622466842024680246842024662202000002" ; string str2 = "1" ; string str3 = "128" ; Console . Write ( isPowerOf2 ( str1 ) + " STRNEWLINE " + isPowerOf2 ( str2 ) + " STRNEWLINE " + isPowerOf2 ( str3 ) ) ; } }
static long ispowerof2 ( long num ) { if ( ( num & ( num - 1 ) ) == 0 ) return 1 ; return 0 ; }
public static void Main ( ) { long num = 549755813888 ; System . Console . WriteLine ( ispowerof2 ( num ) ) ; } }
static int counDivisors ( int X ) {
int count = 0 ;
for ( int i = 1 ; i <= X ; ++ i ) { if ( X % i == 0 ) { count ++ ; } }
return count ; }
static int countDivisorsMult ( int [ ] arr , int n ) {
int mul = 1 ; for ( int i = 0 ; i < n ; ++ i ) mul *= arr [ i ] ;
return counDivisors ( mul ) ; }
public static void Main ( ) { int [ ] arr = { 2 , 4 , 6 } ; int n = arr . Length ; Console . Write ( countDivisorsMult ( arr , n ) ) ; } }
static int freqPairs ( int [ ] arr , int n ) {
int max = arr . Max ( ) ;
int [ ] freq = new int [ max + 1 ] ;
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { freq [ arr [ i ] ] ++ ; }
for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 2 * arr [ i ] ; j <= max ; j += arr [ i ] ) {
if ( freq [ j ] >= 1 ) { count += freq [ j ] ; } }
if ( freq [ arr [ i ] ] > 1 ) { count += freq [ arr [ i ] ] - 1 ; freq [ arr [ i ] ] -- ; } } return count ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 3 , 2 , 4 , 2 , 6 } ; int n = arr . Length ; Console . WriteLine ( freqPairs ( arr , n ) ) ; } }
static double Nth_Term ( int n ) { return ( 2 * Math . Pow ( n , 3 ) - 3 * Math . Pow ( n , 2 ) + n + 6 ) / 6 ; }
static public void Main ( ) { int N = 8 ; Console . WriteLine ( Nth_Term ( N ) ) ; } }
static int printNthElement ( int n ) {
int [ ] arr = new int [ n + 1 ] ; arr [ 1 ] = 3 ; arr [ 2 ] = 5 ; for ( int i = 3 ; i <= n ; i ++ ) {
if ( i % 2 != 0 ) arr [ i ] = arr [ i / 2 ] * 10 + 3 ; else arr [ i ] = arr [ ( i / 2 ) - 1 ] * 10 + 5 ; } return arr [ n ] ; }
static void Main ( ) { int n = 6 ; Console . WriteLine ( printNthElement ( n ) ) ; } }
return ( N * ( ( N / 2 ) + ( ( N % 2 ) * 2 ) + N ) ) ; }
public static void Main ( ) {
GFG a = new GFG ( ) ;
Console . WriteLine ( " Nth ▁ term ▁ for ▁ N ▁ = ▁ " + N + " ▁ : ▁ " + a . nthTerm ( N ) ) ; } }
static void series ( int A , int X , int n ) {
int term = ( int ) Math . Pow ( A , n ) ; Console . Write ( term + " ▁ " ) ;
for ( int i = 1 ; i <= n ; i ++ ) {
term = term * X * ( n - i + 1 ) / ( i * A ) ; Console . Write ( term + " ▁ " ) ; } }
public static void Main ( ) { int A = 3 , X = 4 , n = 5 ; series ( A , X , n ) ; } }
static bool Div_by_8 ( int n ) { return ( ( ( n >> 3 ) << 3 ) == n ) ; }
public static void Main ( ) { int n = 16 ; if ( Div_by_8 ( n ) ) Console . WriteLine ( " YES " ) ; else Console . WriteLine ( " NO " ) ; } }
static int averageEven ( int n ) { if ( n % 2 != 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } int sum = 0 , count = 0 ; while ( n >= 2 ) {
count ++ ;
sum += n ; n = n - 2 ; } return sum / count ; }
public static void Main ( ) { int n = 16 ; Console . Write ( averageEven ( n ) ) ; } }
static int averageEven ( int n ) { if ( n % 2 != 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } return ( n + 2 ) / 2 ; }
public static void Main ( ) { int n = 16 ; Console . Write ( averageEven ( n ) ) ; } }
static int gcd ( int a , int b ) {
if ( a == 0 b == 0 ) return 0 ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
static int cpFact ( int x , int y ) { while ( gcd ( x , y ) != 1 ) { x = x / gcd ( x , y ) ; } return x ; }
public static void Main ( ) { int x = 15 ; int y = 3 ; Console . WriteLine ( cpFact ( x , y ) ) ; x = 14 ; y = 28 ; Console . WriteLine ( cpFact ( x , y ) ) ; x = 7 ; y = 3 ; Console . WriteLine ( cpFact ( x , y ) ) ; } }
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
static int fac ( int n ) { if ( n == 0 ) return 1 ; return n * fac ( n - 1 ) ; }
static int div ( int x ) { int ans = 0 ; for ( int i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
static int sumFactDiv ( int n ) { return div ( fac ( n ) ) ; }
public static void Main ( ) { int n = 4 ; Console . Write ( sumFactDiv ( n ) ) ; } }
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
static bool checkPandigital ( int b , string n ) {
if ( n . Length < b ) return false ; bool [ ] hash = new bool [ b ] ; for ( int i = 0 ; i < b ; i ++ ) hash [ i ] = false ;
for ( int i = 0 ; i < n . Length ; i ++ ) {
if ( n [ i ] >= '0' && n [ i ] <= '9' ) hash [ n [ i ] - '0' ] = true ;
else if ( n [ i ] - ' A ' <= b - 11 ) [ n [ i ] - ' A ' + 10 ] = true ; }
for ( int i = 0 ; i < b ; i ++ ) if ( hash [ i ] == false ) return false ; return true ; }
public static void Main ( ) { int b = 13 ; String n = "1298450376ABC " ; if ( checkPandigital ( b , n ) ) Console . Write ( " Yes " ) ; else Console . Write ( " No " ) ; } }
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
static int maxPrimefactorNum ( int N ) { int [ ] arr = new int [ N + 5 ] ;
for ( int i = 2 ; i * i <= N ; i ++ ) { if ( arr [ i ] == 0 ) { for ( int j = 2 * i ; j <= N ; j += i ) { arr [ j ] ++ ; } } arr [ i ] = 1 ; } int maxval = 0 , maxint = 1 ;
for ( int i = 1 ; i <= N ; i ++ ) { if ( arr [ i ] > maxval ) { maxval = arr [ i ] ; maxint = i ; } } return maxint ; }
public static void Main ( ) { int N = 40 ; Console . WriteLine ( maxPrimefactorNum ( N ) ) ; } }
public static long SubArraySum ( int [ ] arr , int n ) { long result = 0 ;
for ( int i = 0 ; i < n ; i ++ ) result += ( arr [ i ] * ( i + 1 ) * ( n - i ) ) ;
return result ; }
static public void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int n = arr . Length ; Console . WriteLine ( " Sum ▁ of ▁ SubArray : ▁ " + SubArraySum ( arr , n ) ) ; } }
using System ; class GFG { public static int highestPowerof2 ( int n ) { int res = 0 ; for ( int i = n ; i >= 1 ; i -- ) {
if ( ( i & ( i - 1 ) ) == 0 ) { res = i ; break ; } } return res ; }
static public void Main ( ) { int n = 10 ; Console . WriteLine ( highestPowerof2 ( n ) ) ; } }
static void findPairs ( int n ) {
int cubeRoot = ( int ) Math . Pow ( n , 1.0 / 3.0 ) ;
int [ ] cube = new int [ cubeRoot + 1 ] ;
for ( int i = 1 ; i <= cubeRoot ; i ++ ) cube [ i ] = i * i * i ;
int l = 1 ; int r = cubeRoot ; while ( l < r ) { if ( cube [ l ] + cube [ r ] < n ) l ++ ; else if ( cube [ l ] + cube [ r ] > n ) r -- ; else { Console . WriteLine ( " ( " + l + " , ▁ " + r + " ) " ) ; l ++ ; r -- ; } } }
public static void Main ( ) { int n = 20683 ; findPairs ( n ) ; } }
static int gcd ( int a , int b ) { while ( b != 0 ) { int t = b ; b = a % b ; a = t ; } return a ; }
static int findMinDiff ( int a , int b , int x , int y ) {
int g = gcd ( a , b ) ;
int diff = Math . Abs ( x - y ) % g ; return Math . Min ( diff , g - diff ) ; }
static void Main ( ) { int a = 20 , b = 52 , x = 5 , y = 7 ; Console . WriteLine ( findMinDiff ( a , b , x , y ) ) ; } }
static void printDivisors ( int n ) {
int [ ] v = new int [ n ] ; int t = 0 ; for ( int i = 1 ; i <= Math . Sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) Console . Write ( i + " ▁ " ) ; else { Console . Write ( i + " ▁ " ) ;
v [ t ++ ] = n / i ; } } }
for ( int i = t - 1 ; i >= 0 ; i -- ) Console . Write ( v [ i ] + " ▁ " ) ; }
public static void Main ( ) { Console . Write ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ) ; printDivisors ( 100 ) ; } }
static void printDivisors ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) Console . Write ( i + " ▁ " ) ; }
public static void Main ( ) { Console . Write ( " The ▁ divisors ▁ of " , " ▁ 100 ▁ are : ▁ " ) ; printDivisors ( 100 ) ; ; } }
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
static void printSubsets ( int n ) { for ( int i = n ; i > 0 ; i = ( i - 1 ) & n ) Console . Write ( i + " ▁ " ) ; Console . WriteLine ( "0" ) ; }
static public void Main ( ) { int n = 9 ; printSubsets ( n ) ; } }
static bool isDivisibleby17 ( int n ) {
if ( n == 0 n == 17 ) return true ;
if ( n < 17 ) return false ;
return isDivisibleby17 ( ( int ) ( n >> 4 ) - ( int ) ( n & 15 ) ) ; }
public static void Main ( ) { int n = 35 ; if ( isDivisibleby17 ( n ) == true ) Console . WriteLine ( n + " is ▁ divisible ▁ by ▁ 17" ) ; else Console . WriteLine ( n + " ▁ is ▁ not ▁ divisible ▁ by ▁ 17" ) ; } }
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
using System ; public class GFG { static int multiplyBySevenByEight ( int n ) {
return ( ( n << 3 ) - n ) >> 3 ; }
public static void Main ( ) { int n = 15 ; Console . WriteLine ( multiplyBySevenByEight ( n ) ) ; } }
using System ; using System . Linq ; class GFG {
static double getMaxMedian ( int [ ] arr , int n , int k ) { int size = n + k ;
Array . Sort ( arr ) ;
if ( size % 2 == 0 ) { double median = ( double ) ( arr [ ( size / 2 ) - 1 ] + arr [ size / 2 ] ) / 2 ; return median ; }
double median1 = arr [ size / 2 ] ; return median1 ; }
using System ; class GFG { static void printSorted ( int a , int b , int c ) {
int get_max = Math . Max ( a , Math . Max ( b , c ) ) ;
int get_min = - Math . Max ( - a , Math . Max ( - b , - c ) ) ; int get_mid = ( a + b + c ) - ( get_max + get_min ) ; Console . Write ( get_min + " ▁ " + get_mid + " ▁ " + get_max ) ; }
public static void Main ( ) { int a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ; } }
void sort ( int [ ] arr ) { int n = arr . Length ; for ( int i = 1 ; i < n ; ++ i ) { int key = arr [ i ] ; int j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
static void printArray ( int [ ] arr ) { int n = arr . Length ; for ( int i = 0 ; i < n ; ++ i ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . Write ( " STRNEWLINE " ) ; }
public static void Main ( ) { int [ ] arr = { 12 , 11 , 13 , 5 , 6 } ; InsertionSort ob = new InsertionSort ( ) ; ob . sort ( arr ) ; printArray ( arr ) ; } }
static int countPaths ( int n , int m ) { int [ , ] dp = new int [ n + 1 , m + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) dp [ i , 0 ] = 1 ; for ( int i = 0 ; i <= m ; i ++ ) dp [ 0 , i ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) for ( int j = 1 ; j <= m ; j ++ ) dp [ i , j ] = dp [ i - 1 , j ] + dp [ i , j - 1 ] ; return dp [ n , m ] ; }
public static void Main ( ) { int n = 3 , m = 2 ; Console . WriteLine ( " ▁ Number ▁ of " + " ▁ Paths ▁ " + countPaths ( n , m ) ) ; } }
static int count ( int [ ] S , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int m = arr . Length ; Console . Write ( count ( arr , m , 4 ) ) ; } }
static bool isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
static String encryptString ( String s , int n , int k ) { int countVowels = 0 ; int countConsonants = 0 ; String ans = " " ;
for ( int l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( int r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s [ r ] ) == true ) { countVowels ++ ; } else { countConsonants ++ ; } }
ans += Convert . ToString ( countVowels * countConsonants ) ; } return ans ; }
static public void Main ( ) { String s = " hello " ; int n = s . Length ; int k = 2 ; Console . Write ( encryptString ( s , n , k ) ) ; } }
static float findVolume ( float a ) {
if ( a < 0 ) return - 1 ;
float r = a / 2 ;
float h = a ;
float V = ( float ) ( 3.14 * Math . Pow ( r , 2 ) * h ) ; return V ; }
public static void Main ( ) { float a = 5 ; Console . WriteLine ( findVolume ( a ) ) ; } }
public static float volumeTriangular ( int a , int b , int h ) { float vol = ( float ) ( 0.1666 ) * a * b * h ; return vol ; }
public static float volumeSquare ( int b , int h ) { float vol = ( float ) ( 0.33 ) * b * b * h ; return vol ; }
public static float volumePentagonal ( int a , int b , int h ) { float vol = ( float ) ( 0.83 ) * a * b * h ; return vol ; }
public static float volumeHexagonal ( int a , int b , int h ) { float vol = ( float ) a * b * h ; return vol ; }
public static void Main ( ) { int b = 4 , h = 9 , a = 4 ; Console . WriteLine ( " Volume ▁ of ▁ triangular " + " ▁ base ▁ pyramid ▁ is ▁ " + volumeTriangular ( a , b , h ) ) ; Console . WriteLine ( " Volume ▁ of ▁ square ▁ " + " base ▁ pyramid ▁ is ▁ " + volumeSquare ( b , h ) ) ; Console . WriteLine ( " Volume ▁ of ▁ pentagonal " + " ▁ base ▁ pyramid ▁ is ▁ " + volumePentagonal ( a , b , h ) ) ; Console . WriteLine ( " Volume ▁ of ▁ Hexagonal " + " ▁ base ▁ pyramid ▁ is ▁ " + volumeHexagonal ( a , b , h ) ) ; } }
static double Area ( int b1 , int b2 , int h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
public static void Main ( ) { int base1 = 8 , base2 = 10 , height = 6 ; double area = Area ( base1 , base2 , height ) ; Console . WriteLine ( " Area ▁ is : ▁ " + area ) ; } }
using System ; class GFG { static int numberOfDiagonals ( int n ) { return n * ( n - 3 ) / 2 ; }
public static void Main ( ) { int n = 5 ; Console . Write ( n + " ▁ sided ▁ convex ▁ polygon ▁ have ▁ " ) ; Console . WriteLine ( numberOfDiagonals ( n ) + " ▁ diagonals " ) ; } }
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
static int LowerInsertionPoint ( int [ ] arr , int n , int X ) {
if ( X < arr [ 0 ] ) return 0 ; else if ( X > arr [ n - 1 ] ) return n ; int lowerPnt = 0 ; int i = 1 ; while ( i < n && arr [ i ] < X ) { lowerPnt = i ; i = i * 2 ; }
while ( lowerPnt < n && arr [ lowerPnt ] < X ) lowerPnt ++ ; return lowerPnt ; }
static public void Main ( ) { int [ ] arr = { 2 , 3 , 4 , 4 , 5 , 6 , 7 , 9 } ; int n = arr . Length ; int X = 4 ; Console . WriteLine ( LowerInsertionPoint ( arr , n , X ) ) ; } }
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
public static int middleOfThree ( int a , int b , int c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
public static void Main ( ) { int a = 20 , b = 30 , c = 40 ; Console . WriteLine ( middleOfThree ( a , b , c ) ) ; } }
public static void printArr ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) Console . Write ( arr [ i ] ) ; }
public static int compare ( int num1 , int num2 ) {
String A = num1 . ToString ( ) ;
String B = num2 . ToString ( ) ;
return ( A + B ) . CompareTo ( B + A ) ; }
public static void printSmallest ( int N , int [ ] arr ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) { if ( compare ( arr [ i ] , arr [ j ] ) > 0 ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } } }
printArr ( arr , N ) ; }
public static void Main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 2 , 9 , 21 , 1 } ; int N = arr . Length ; printSmallest ( N , arr ) ; } }
static bool isPossible ( int [ ] a , int [ ] b , int n , int k ) {
Array . Sort ( a ) ;
Array . Reverse ( b ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
public static void Main ( ) { int [ ] a = { 2 , 1 , 3 } ; int [ ] b = { 7 , 8 , 9 } ; int k = 10 ; int n = a . Length ; if ( isPossible ( a , b , n , k ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
static String encryptString ( String str , int n ) { int i = 0 , cnt = 0 ; String encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- > 0 ) encryptedStr += str [ i ] ; i ++ ; } return encryptedStr ; }
static public void Main ( ) { String str = " geeks " ; int n = str . Length ; Console . WriteLine ( encryptString ( str , n ) ) ; } }
static int minDiff ( int n , int x , int [ ] A ) { int mn = A [ 0 ] , mx = A [ 0 ] ;
for ( int i = 0 ; i < n ; ++ i ) { mn = Math . Min ( mn , A [ i ] ) ; mx = Math . Max ( mx , A [ i ] ) ; }
return Math . Max ( 0 , mx - mn - 2 * x ) ; }
public static void Main ( ) { int n = 3 , x = 3 ; int [ ] A = { 1 , 3 , 6 } ;
Console . WriteLine ( minDiff ( n , x , A ) ) ; } }
