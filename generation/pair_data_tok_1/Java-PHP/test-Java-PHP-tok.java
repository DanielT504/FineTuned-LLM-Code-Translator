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
static long partitions ( int n ) { long p [ ] = new long [ n + 1 ] ;
p [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; ++ i ) { int k = 1 ; while ( ( k * ( 3 * k - 1 ) ) / 2 <= i ) { p [ i ] += ( k % 2 != 0 ? 1 : - 1 ) * p [ i - ( k * ( 3 * k - 1 ) ) / 2 ] ; if ( k > 0 ) { k *= - 1 ; } else { k = 1 - k ; } } } return p [ n ] ; }
public static void main ( String [ ] args ) { int N = 20 ; System . out . println ( partitions ( N ) ) ; } }
static int countPaths ( int n , int m ) {
if ( n == 0 m == 0 ) return 1 ;
return ( countPaths ( n - 1 , m ) + countPaths ( n , m - 1 ) ) ; }
public static void main ( String [ ] args ) { int n = 3 , m = 2 ; System . out . println ( " ▁ Number ▁ of ▁ Paths ▁ " + countPaths ( n , m ) ) ; } }
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
static int countChar ( String str , char x ) { int count = 0 ; int n = 10 ; for ( int i = 0 ; i < str . length ( ) ; i ++ ) if ( str . charAt ( i ) == x ) count ++ ;
int repetitions = n / str . length ( ) ; count = count * repetitions ;
for ( int i = 0 ; i < n % str . length ( ) ; i ++ ) { if ( str . charAt ( i ) == x ) count ++ ; } return count ; }
public static void main ( String args [ ] ) { String str = " abcac " ; System . out . println ( countChar ( str , ' a ' ) ) ; } }
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
class GFG { static boolean isValidISBN ( String isbn ) {
int n = isbn . length ( ) ; if ( n != 10 ) return false ;
int sum = 0 ; for ( int i = 0 ; i < 9 ; i ++ ) { int digit = isbn . charAt ( i ) - '0' ; if ( 0 > digit 9 < digit ) return false ; sum += ( digit * ( 10 - i ) ) ; }
char last = isbn . charAt ( 9 ) ; if ( last != ' X ' && ( last < '0' last > '9' ) ) return false ;
sum += ( ( last == ' X ' ) ? 10 : ( last - '0' ) ) ;
return ( sum % 11 == 0 ) ; }
public static void main ( String [ ] args ) { String isbn = "007462542X " ; if ( isValidISBN ( isbn ) ) System . out . print ( " Valid " ) ; else System . out . print ( " Invalid " ) ; } }
public static void main ( String [ ] args ) { int d = 10 ; double a ;
a = ( double ) ( 360 - ( 6 * d ) ) / 4 ;
System . out . print ( a + " , ▁ " + ( a + d ) + " , ▁ " + ( a + ( 2 * d ) ) + " , ▁ " + ( a + ( 3 * d ) ) ) ; } }
static void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = Math . abs ( ( c2 * z1 + d2 ) ) / ( float ) ( Math . sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; System . out . println ( " Perpendicular ▁ distance ▁ is ▁ " + d ) ; } else System . out . println ( " Planes ▁ are ▁ not ▁ parallel " ) ; }
public static void main ( String [ ] args ) { float a1 = 1 ; float b1 = 2 ; float c1 = - 1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = - 3 ; float d2 = - 4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ; } }
static void PrintMinNumberForPattern ( String arr ) {
int curr_max = 0 ;
int last_entry = 0 ; int j ;
for ( int i = 0 ; i < arr . length ( ) ; i ++ ) {
int noOfNextD = 0 ; switch ( arr . charAt ( i ) ) { case ' I ' :
j = i + 1 ; while ( j < arr . length ( ) && arr . charAt ( j ) == ' D ' ) { noOfNextD ++ ; j ++ ; } if ( i == 0 ) { curr_max = noOfNextD + 2 ;
System . out . print ( " ▁ " + ++ last_entry ) ; System . out . print ( " ▁ " + curr_max ) ;
last_entry = curr_max ; } else {
curr_max = curr_max + noOfNextD + 1 ;
last_entry = curr_max ; System . out . print ( " ▁ " + last_entry ) ; }
for ( int k = 0 ; k < noOfNextD ; k ++ ) { System . out . print ( " ▁ " + -- last_entry ) ; i ++ ; } break ;
case ' D ' : if ( i == 0 ) {
j = i + 1 ; while ( j < arr . length ( ) && arr . charAt ( j ) == ' D ' ) { noOfNextD ++ ; j ++ ; }
curr_max = noOfNextD + 2 ;
System . out . print ( " ▁ " + curr_max + " ▁ " + ( curr_max - 1 ) ) ;
last_entry = curr_max - 1 ; } else {
System . out . print ( " ▁ " + ( last_entry - 1 ) ) ; last_entry -- ; } break ; } } System . out . println ( ) ; }
public static void main ( String [ ] args ) { PrintMinNumberForPattern ( " IDID " ) ; PrintMinNumberForPattern ( " I " ) ; PrintMinNumberForPattern ( " DD " ) ; PrintMinNumberForPattern ( " II " ) ; PrintMinNumberForPattern ( " DIDI " ) ; PrintMinNumberForPattern ( " IIDDD " ) ; PrintMinNumberForPattern ( " DDIDDIID " ) ; } }
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
static int sum ( int L , int R ) {
int p = R / 6 ;
int q = ( L - 1 ) / 6 ;
int sumR = 3 * ( p * ( p + 1 ) ) ;
int sumL = ( q * ( q + 1 ) ) * 3 ;
return sumR - sumL ; }
public static void main ( String [ ] args ) { int L = 1 , R = 20 ; System . out . println ( sum ( L , R ) ) ; } }
class GFG {
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
static int SieveOfSundaram ( int n ) {
int nNew = ( n - 1 ) / 2 ;
Arrays . fill ( marked , false ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) for ( int j = i ; ( i + j + 2 * i * j ) <= nNew ; j ++ ) marked [ i + j + 2 * i * j ] = true ;
if ( n > 2 ) System . out . print ( 2 + " ▁ " ) ;
for ( int i = 1 ; i <= nNew ; i ++ ) if ( marked [ i ] == false ) System . out . print ( 2 * i + 1 + " ▁ " ) ; return - 1 ; }
public static void main ( String [ ] args ) { int n = 20 ; SieveOfSundaram ( n ) ; } }
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
static boolean areElementsContiguous ( int arr [ ] , int n ) {
Arrays . sort ( arr ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] > 1 ) return false ; return true ; }
public static void main ( String [ ] args ) { int arr [ ] = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . length ; if ( areElementsContiguous ( arr , n ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static int findLargestd ( int [ ] S , int n ) { boolean found = false ;
Arrays . sort ( S ) ;
for ( int i = n - 1 ; i >= 0 ; i -- ) { for ( int j = 0 ; j < n ; j ++ ) {
if ( i == j ) continue ; for ( int k = j + 1 ; k < n ; k ++ ) { if ( i == k ) continue ; for ( int l = k + 1 ; l < n ; l ++ ) { if ( i == l ) continue ;
if ( S [ i ] == S [ j ] + S [ k ] + S [ l ] ) { found = true ; return S [ i ] ; } } } } } if ( found == false ) return Integer . MAX_VALUE ; return - 1 ; }
public static void main ( String [ ] args ) { int [ ] S = new int [ ] { 2 , 3 , 5 , 7 , 12 } ; int n = S . length ; int ans = findLargestd ( S , n ) ; if ( ans == Integer . MAX_VALUE ) System . out . println ( " No ▁ Solution " ) ; else System . out . println ( " Largest ▁ d ▁ such ▁ that ▁ " + " a ▁ + ▁ " + " b ▁ + ▁ c ▁ = ▁ d ▁ is ▁ " + ans ) ; } }
void leftRotatebyOne ( int arr [ ] , int n ) { int i , temp ; temp = arr [ 0 ] ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
void leftRotate ( int arr [ ] , int d , int n ) { for ( int i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String [ ] args ) { RotateArray rotate = new RotateArray ( ) ; int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; rotate . leftRotate ( arr , 2 , 7 ) ; rotate . printArray ( arr , 7 ) ; } }
import java . io . * ; import java . util . * ; import java . lang . * ; class GFG {
static void partSort ( int [ ] arr , int N , int a , int b ) {
int l = Math . min ( a , b ) ; int r = Math . max ( a , b ) ;
int [ ] temp = new int [ r - l + 1 ] ; int j = 0 ; for ( int i = l ; i <= r ; i ++ ) { temp [ j ] = arr [ i ] ; j ++ ; }
Arrays . sort ( temp ) ;
j = 0 ; for ( int i = l ; i <= r ; i ++ ) { arr [ i ] = temp [ j ] ; j ++ ; }
for ( int i = 0 ; i < N ; i ++ ) { System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int [ ] arr = { 7 , 8 , 4 , 5 , 2 } ; int a = 1 , b = 4 ;
int N = arr . length ; partSort ( arr , N , a , b ) ; } }
static void pushZerosToEnd ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] != 0 )
arr [ count ++ ] = arr [ i ] ;
while ( count < n ) arr [ count ++ ] = 0 ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 9 , 8 , 4 , 0 , 0 , 2 , 7 , 0 , 6 , 0 , 9 } ; int n = arr . length ; pushZerosToEnd ( arr , n ) ; System . out . println ( " Array ▁ after ▁ pushing ▁ zeros ▁ to ▁ the ▁ back : ▁ " ) ; for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
static void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; }
static void RearrangePosNeg ( int arr [ ] , int n ) { int key , j ; for ( int i = 1 ; i < n ; i ++ ) { key = arr [ i ] ;
if ( key > 0 ) continue ;
j = i - 1 ; while ( j >= 0 && arr [ j ] > 0 ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; }
arr [ j + 1 ] = key ; } }
public static void main ( String [ ] args ) { int arr [ ] = { - 12 , 11 , - 13 , - 5 , 6 , - 7 , 5 , - 3 , - 6 } ; int n = arr . length ; RearrangePosNeg ( arr , n ) ; printArray ( arr , n ) ; } }
import java . util . * ; import java . io . * ; class GFG { static void findElements ( int arr [ ] , int n ) {
for ( int i = 0 ; i < n ; i ++ ) { int count = 0 ; for ( int j = 0 ; j < n ; j ++ ) if ( arr [ j ] > arr [ i ] ) count ++ ; if ( count >= 2 ) System . out . print ( arr [ i ] + " ▁ " ) ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . length ; findElements ( arr , n ) ; } }
import java . util . * ; import java . io . * ; class GFG { static void findElements ( int arr [ ] , int n ) { Arrays . sort ( arr ) ; for ( int i = 0 ; i < n - 2 ; i ++ ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . length ; findElements ( arr , n ) ; } }
import java . util . * ; import java . io . * ; class GFG { static void findElements ( int arr [ ] , int n ) { int first = Integer . MIN_VALUE ; int second = Integer . MAX_VALUE ; for ( int i = 0 ; i < n ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second ) second = arr [ i ] ; } for ( int i = 0 ; i < n ; i ++ ) if ( arr [ i ] < second ) System . out . print ( arr [ i ] + " ▁ " ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 2 , - 6 , 3 , 5 , 1 } ; int n = arr . length ; findElements ( arr , n ) ; } }
int findFirstMissing ( int array [ ] , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
public static void main ( String [ ] args ) { SmallestMissing small = new SmallestMissing ( ) ; int arr [ ] = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = arr . length ; System . out . println ( " First ▁ Missing ▁ element ▁ is ▁ : ▁ " + small . findFirstMissing ( arr , 0 , n - 1 ) ) ; } }
int FindMaxSum ( int arr [ ] , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
public static void main ( String [ ] args ) { MaximumSum sum = new MaximumSum ( ) ; int arr [ ] = new int [ ] { 5 , 5 , 10 , 100 , 10 , 5 } ; System . out . println ( sum . FindMaxSum ( arr , arr . length ) ) ; } }
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
import java . io . * ; class Majority { static boolean isMajority ( int arr [ ] , int n , int x ) { int i , last_index = 0 ;
last_index = ( n % 2 == 0 ) ? n / 2 : n / 2 + 1 ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + n / 2 ] == x ) return true ; } return false ; }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 3 , 4 , 4 , 4 , 4 } ; int n = arr . length ; int x = 4 ; if ( isMajority ( arr , n , x ) == true ) System . out . println ( x + " ▁ appears ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; else System . out . println ( x + " ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ " + n / 2 + " ▁ times ▁ in ▁ arr [ ] " ) ; } }
static int cutRod ( int price [ ] , int n ) { int val [ ] = new int [ n + 1 ] ; val [ 0 ] = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { int max_val = Integer . MIN_VALUE ; for ( int j = 0 ; j < i ; j ++ ) max_val = Math . max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = arr . length ; System . out . println ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ " + cutRod ( arr , size ) ) ; } }
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
long n = 5 ;
long fac1 = 1 ; for ( int i = 2 ; i <= n - 1 ; i ++ ) fac1 = fac1 * i ;
long fac2 = fac1 * n ;
long totalWays = fac1 * fac2 ;
System . out . println ( totalWays ) ; } }
static int nextPerfectCube ( int N ) { int nextN = ( int ) Math . floor ( Math . cbrt ( N ) ) + 1 ; return nextN * nextN * nextN ; }
public static void main ( String args [ ] ) { int n = 35 ; System . out . print ( nextPerfectCube ( n ) ) ; } }
class GFG { static int findpos ( String n ) { int pos = 0 ; for ( int i = 0 ; i < n . length ( ) ; i ++ ) { switch ( n . charAt ( i ) ) {
case '2' : pos = pos * 4 + 1 ; break ;
case '3' : pos = pos * 4 + 2 ; break ;
case '5' : pos = pos * 4 + 3 ; break ;
case '7' : pos = pos * 4 + 4 ; break ; } } return pos ; }
public static void main ( String args [ ] ) { String n = "777" ; System . out . println ( findpos ( n ) ) ; } }
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
public static void main ( String args [ ] ) { int N = 6 ; int Even = N / 2 ; int Odd = N - Even ; System . out . println ( Even * Odd ) ; } }
static void steps ( String str , int n ) {
boolean flag = false ; int x = 0 ;
for ( int i = 0 ; i < str . length ( ) ; i ++ ) {
if ( x == 0 ) flag = true ;
if ( x == n - 1 ) flag = false ;
for ( int j = 0 ; j < x ; j ++ ) System . out . print ( " * " ) ; System . out . print ( str . charAt ( i ) + "NEW_LINE");
if ( flag == true ) x ++ ; else x -- ; } }
int n = 4 ; String str = " GeeksForGeeks " ; System . out . println ( " String : ▁ " + str ) ; System . out . println ( " Max ▁ Length ▁ of ▁ Steps : ▁ " + n ) ;
steps ( str , n ) ; } }
static boolean isDivisible ( String str , int k ) { int n = str . length ( ) ; int c = 0 ;
for ( int i = 0 ; i < k ; i ++ ) if ( str . charAt ( n - i - 1 ) == '0' ) c ++ ;
return ( c == k ) ; }
String str1 = "10101100" ; int k = 2 ; if ( isDivisible ( str1 , k ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ;
String str2 = "111010100" ; k = 2 ; if ( isDivisible ( str2 , k ) == true ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean isNumber ( String s ) { for ( int i = 0 ; i < s . length ( ) ; i ++ ) if ( Character . isDigit ( s . charAt ( i ) ) == false ) return false ; return true ; }
String str = "6790" ;
if ( isNumber ( str ) ) System . out . println ( " Integer " ) ;
else System . out . println ( " String " ) ; } }
void reverse ( String str ) { if ( ( str == null ) || ( str . length ( ) <= 1 ) ) System . out . println ( str ) ; else { System . out . print ( str . charAt ( str . length ( ) - 1 ) ) ; reverse ( str . substring ( 0 , str . length ( ) - 1 ) ) ; } }
public static void main ( String [ ] args ) { String str = " Geeks ▁ for ▁ Geeks " ; StringReverse obj = new StringReverse ( ) ; obj . reverse ( str ) ; } }
static double polyarea ( double n , double r ) {
if ( r < 0 && n < 0 ) return - 1 ;
double A = ( ( r * r * n ) * Math . sin ( ( 360 / n ) * 3.14159 / 180 ) ) / 2 ; return A ; }
public static void main ( String [ ] args ) { float r = 9 , n = 6 ; System . out . println ( polyarea ( n , r ) ) ; } }
static double findPCSlope ( double m ) { return - 1.0 / m ; }
public static void main ( String [ ] args ) { double m = 2.0 ; System . out . println ( findPCSlope ( m ) ) ; } }
area_of_segment ( float radius , float angle ) {
float area_of_sector = pi * ( radius * radius ) * ( angle / 360 ) ;
float area_of_triangle = ( float ) 1 / 2 * ( radius * radius ) * ( float ) Math . sin ( ( angle * pi ) / 180 ) ; return area_of_sector - area_of_triangle ; }
public static void main ( String [ ] args ) { float radius = 10.0f , angle = 90.0f ; System . out . println ( " Area ▁ of ▁ minor ▁ segment ▁ = ▁ " + area_of_segment ( radius , angle ) ) ; System . out . println ( " Area ▁ of ▁ major ▁ segment ▁ = ▁ " + area_of_segment ( radius , ( 360 - angle ) ) ) ; } }
class GFG { static void SectorArea ( double radius , double angle ) { if ( angle >= 360 ) System . out . println ( " Angle ▁ not ▁ possible " ) ;
else { double sector = ( ( 22 * radius * radius ) / 7 ) * ( angle / 360 ) ; System . out . println ( sector ) ; } }
public static void main ( String [ ] args ) { double radius = 9 ; double angle = 60 ; SectorArea ( radius , angle ) ; } }
static void insertionSortRecursive ( int arr [ ] , int n ) {
if ( n <= 1 ) return ;
insertionSortRecursive ( arr , n - 1 ) ;
int last = arr [ n - 1 ] ; int j = n - 2 ;
while ( j >= 0 && arr [ j ] > last ) { arr [ j + 1 ] = arr [ j ] ; j -- ; } arr [ j + 1 ] = last ; }
public static void main ( String [ ] args ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; insertionSortRecursive ( arr , arr . length ) ; System . out . println ( Arrays . toString ( arr ) ) ; } }
static boolean isWaveArray ( int arr [ ] , int n ) { boolean result = true ;
if ( arr [ 1 ] > arr [ 0 ] && arr [ 1 ] > arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] > arr [ i - 1 ] && arr [ i ] > arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] <= arr [ n - 2 ] ) { result = false ; } } } else if ( arr [ 1 ] < arr [ 0 ] && arr [ 1 ] < arr [ 2 ] ) { for ( int i = 1 ; i < n - 1 ; i += 2 ) { if ( arr [ i ] < arr [ i - 1 ] && arr [ i ] < arr [ i + 1 ] ) { result = true ; } else { result = false ; break ; } }
if ( result == true && n % 2 == 0 ) { if ( arr [ n - 1 ] >= arr [ n - 2 ] ) { result = false ; } } } return result ; }
int arr [ ] = { 1 , 3 , 2 , 4 } ; int n = arr . length ; if ( isWaveArray ( arr , n ) ) { System . out . println ( " YES " ) ; } else { System . out . println ( " NO " ) ; } } }
import java . io . * ; class GFG { static int mod = 1000000007 ;
static int sumOddFibonacci ( int n ) { int Sum [ ] = new int [ n + 1 ] ;
Sum [ 0 ] = 0 ; Sum [ 1 ] = 1 ; Sum [ 2 ] = 2 ; Sum [ 3 ] = 5 ; Sum [ 4 ] = 10 ; Sum [ 5 ] = 23 ; for ( int i = 6 ; i <= n ; i ++ ) { Sum [ i ] = ( ( Sum [ i - 1 ] + ( 4 * Sum [ i - 2 ] ) % mod - ( 4 * Sum [ i - 3 ] ) % mod + mod ) % mod + ( Sum [ i - 4 ] - Sum [ i - 5 ] + mod ) % mod ) % mod ; } return Sum [ n ] ; }
public static void main ( String [ ] args ) { int n = 6 ; System . out . println ( sumOddFibonacci ( n ) ) ; }
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
static int LCIS ( int arr1 [ ] , int n , int arr2 [ ] , int m ) {
int table [ ] = new int [ m ] ; for ( int j = 0 ; j < m ; j ++ ) table [ j ] = 0 ;
for ( int i = 0 ; i < n ; i ++ ) {
int current = 0 ;
for ( int j = 0 ; j < m ; j ++ ) {
if ( arr1 [ i ] == arr2 [ j ] ) if ( current + 1 > table [ j ] ) table [ j ] = current + 1 ;
if ( arr1 [ i ] > arr2 [ j ] ) if ( table [ j ] > current ) current = table [ j ] ; } }
int result = 0 ; for ( int i = 0 ; i < m ; i ++ ) if ( table [ i ] > result ) result = table [ i ] ; return result ; }
public static void main ( String [ ] args ) { int arr1 [ ] = { 3 , 4 , 9 , 1 } ; int arr2 [ ] = { 5 , 3 , 8 , 9 , 10 , 2 , 1 } ; int n = arr1 . length ; int m = arr2 . length ; System . out . println ( " Length ▁ of ▁ LCIS ▁ is ▁ " + LCIS ( arr1 , n , arr2 , m ) ) ; } }
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
static String convert ( String s ) { int n = s . length ( ) ; String s1 = " " ; s1 = s1 + Character . toLowerCase ( s . charAt ( 0 ) ) ; for ( int i = 1 ; i < n ; i ++ ) {
if ( s . charAt ( i ) == ' ▁ ' && i < n ) {
s1 = s1 + " ▁ " + Character . toLowerCase ( s . charAt ( i + 1 ) ) ; i ++ ; }
else s1 = s1 + Character . toUpperCase ( s . charAt ( i ) ) ; }
return s1 ; }
public static void main ( String [ ] args ) { String str = " I ▁ get ▁ intern ▁ at ▁ geeksforgeeks " ; System . out . println ( convert ( str ) ) ; } }
public static String change_case ( String a ) { String temp = " " ; int l = a . length ( ) ; for ( int i = 0 ; i < l ; i ++ ) { char ch = a . charAt ( i ) ;
if ( ch >= ' a ' && ch <= ' z ' ) ch = ( char ) ( 65 + ( int ) ( ch - ' a ' ) ) ;
else if ( ch >= ' A ' && ch <= ' Z ' ) ch = ( char ) ( 97 + ( int ) ( ch - ' A ' ) ) ; temp += ch ; } return temp ; }
public static String delete_vowels ( String a ) { String temp = " " ; int l = a . length ( ) ; for ( int i = 0 ; i < l ; i ++ ) { char ch = a . charAt ( i ) ;
if ( ch != ' a ' && ch != ' e ' && ch != ' i ' && ch != ' o ' && ch != ' u ' && ch != ' A ' && ch != ' E ' && ch != ' O ' && ch != ' U ' && ch != ' I ' ) temp += ch ; } return temp ; }
public static String insert_hash ( String a ) { String temp = " " ; int l = a . length ( ) ; char hash = ' # ' ; for ( int i = 0 ; i < l ; i ++ ) { char ch = a . charAt ( i ) ;
if ( ( ch >= ' a ' && ch <= ' z ' ) || ( ch >= ' A ' && ch <= ' Z ' ) ) temp = temp + hash + ch ; else temp = temp + ch ; } return temp ; }
public static void transformString ( String a ) { String b = delete_vowels ( a ) ; String c = change_case ( b ) ; String d = insert_hash ( c ) ; System . out . println ( d ) ; }
public static void main ( String args [ ] ) { String a = " SunshinE ! ! " ;
transformString ( a ) ; } }
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
import java . util . * ; class Middle {
public static int middleOfThree ( int a , int b , int c ) {
if ( ( a < b && b < c ) || ( c < b && b < a ) ) return b ;
else if ( ( b < a && a < c ) || ( c < a && a < b ) ) return a ; else return c ; }
public static void main ( String [ ] args ) { int a = 20 , b = 30 , c = 40 ; System . out . println ( middleOfThree ( a , b , c ) ) ; } }
class shortest_path { static int INF = Integer . MAX_VALUE , N = 4 ;
static int minCost ( int cost [ ] [ ] ) {
int dist [ ] = new int [ N ] ; for ( int i = 0 ; i < N ; i ++ ) dist [ i ] = INF ; dist [ 0 ] = 0 ;
for ( int i = 0 ; i < N ; i ++ ) for ( int j = i + 1 ; j < N ; j ++ ) if ( dist [ j ] > dist [ i ] + cost [ i ] [ j ] ) dist [ j ] = dist [ i ] + cost [ i ] [ j ] ; return dist [ N - 1 ] ; }
public static void main ( String args [ ] ) { int cost [ ] [ ] = { { 0 , 15 , 80 , 90 } , { INF , 0 , 40 , 50 } , { INF , INF , 0 , 70 } , { INF , INF , INF , 0 } } ; System . out . println ( " The ▁ Minimum ▁ cost ▁ to ▁ reach ▁ station ▁ " + N + " ▁ is ▁ " + minCost ( cost ) ) ; } }
static int numOfways ( int n , int k ) { int p = 1 ; if ( k % 2 != 0 ) p = - 1 ; return ( int ) ( Math . pow ( n - 1 , k ) + p * ( n - 1 ) ) / n ; }
public static void main ( String args [ ] ) { int n = 4 , k = 2 ; System . out . println ( numOfways ( n , k ) ) ; } }
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
static int fib ( int n ) { if ( n <= 1 ) return n ; return fib ( n - 1 ) + fib ( n - 2 ) ; }
static int findVertices ( int n ) {
return fib ( n + 2 ) ; } public static void main ( String args [ ] ) {
int n = 3 ; System . out . println ( findVertices ( n ) ) ; } }
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
import java . io . * ; import java . util . * ; class GFG {
static int findMaxValue ( int N , int mat [ ] [ ] ) {
int maxValue = Integer . MIN_VALUE ;
for ( int a = 0 ; a < N - 1 ; a ++ ) for ( int b = 0 ; b < N - 1 ; b ++ ) for ( int d = a + 1 ; d < N ; d ++ ) for ( int e = b + 1 ; e < N ; e ++ ) if ( maxValue < ( mat [ d ] [ e ] - mat [ a ] [ b ] ) ) maxValue = mat [ d ] [ e ] - mat [ a ] [ b ] ; return maxValue ; }
public static void main ( String [ ] args ) { int N = 5 ; int mat [ ] [ ] = { { 1 , 2 , - 1 , - 4 , - 20 } , { - 8 , - 3 , 4 , 2 , 1 } , { 3 , 8 , 6 , 1 , 3 } , { - 4 , - 1 , 1 , 7 , - 6 } , { 0 , - 4 , 10 , - 5 , 1 } } ; System . out . print ( " Maximum ▁ Value ▁ is ▁ " + findMaxValue ( N , mat ) ) ; } }
import java . io . * ; import java . util . * ; class GFG {
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
void sort ( int arr [ ] ) { int n = arr . length ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min_idx = i ; for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
int temp = arr [ min_idx ] ; arr [ min_idx ] = arr [ i ] ; arr [ i ] = temp ; } }
public static void main ( String args [ ] ) { SelectionSort ob = new SelectionSort ( ) ; int arr [ ] = { 64 , 25 , 12 , 22 , 11 } ; ob . sort ( arr ) ; System . out . println ( " Sorted ▁ array " ) ; ob . printArray ( arr ) ; } }
static void bubbleSort ( int arr [ ] , int n ) { int i , j , temp ; boolean swapped ; for ( i = 0 ; i < n - 1 ; i ++ ) { swapped = false ; for ( j = 0 ; j < n - i - 1 ; j ++ ) { if ( arr [ j ] > arr [ j + 1 ] ) {
temp = arr [ j ] ; arr [ j ] = arr [ j + 1 ] ; arr [ j + 1 ] = temp ; swapped = true ; } }
if ( swapped == false ) break ; } }
public static void main ( String args [ ] ) { int arr [ ] = { 64 , 34 , 25 , 12 , 22 , 11 , 90 } ; int n = arr . length ; bubbleSort ( arr , n ) ; System . out . println ( " Sorted ▁ array : ▁ " ) ; printArray ( arr , n ) ; } }
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
public static int count ( int S [ ] , int m , int n ) {
int table [ ] = new int [ n + 1 ] ;
table [ 0 ] = 1 ;
for ( int i = 0 ; i < m ; i ++ ) for ( int j = S [ i ] ; j <= n ; j ++ ) table [ j ] += table [ j - S [ i ] ] ; return table [ n ] ; }
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
static int getModulo ( int n , int d ) { return ( n & ( d - 1 ) ) ; }
public static void main ( String [ ] args ) { int n = 6 ;
int d = 4 ; System . out . println ( n + " ▁ moduo ▁ " + d + " ▁ is ▁ " + getModulo ( n , d ) ) ; } }
static int countSetBits ( int n ) { int count = 0 ; while ( n > 0 ) { count += n & 1 ; n >>= 1 ; } return count ; }
public static void main ( String args [ ] ) { int i = 9 ; System . out . println ( countSetBits ( i ) ) ; } }
public static int countSetBits ( int n ) {
if ( n == 0 ) return 0 ; else return 1 + countSetBits ( n & ( n - 1 ) ) ; }
int n = 9 ;
System . out . println ( countSetBits ( n ) ) ; } }
public static void main ( String [ ] args ) { System . out . println ( Integer . bitCount ( 4 ) ) ; System . out . println ( Integer . bitCount ( 15 ) ) ; } }
class GFG { static int [ ] num_to_bits = new int [ ] { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
static int countSetBitsRec ( int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
public static void main ( String [ ] args ) { int num = 31 ; System . out . println ( countSetBitsRec ( num ) ) ; } }
static boolean getParity ( int n ) { boolean parity = false ; while ( n != 0 ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
public static void main ( String [ ] args ) { int n = 12 ; System . out . println ( " Parity ▁ of ▁ no ▁ " + n + " ▁ = ▁ " + ( getParity ( n ) ? " odd " : " even " ) ) ; } }
class GFG {
static boolean isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( int ) ( Math . ceil ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) == ( int ) ( Math . floor ( ( ( Math . log ( n ) / Math . log ( 2 ) ) ) ) ) ; }
public static void main ( String [ ] args ) { if ( isPowerOfTwo ( 31 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; if ( isPowerOfTwo ( 64 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
static boolean isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; while ( n != 1 ) { if ( n % 2 != 0 ) return false ; n = n / 2 ; } return true ; }
public static void main ( String args [ ] ) { if ( isPowerOfTwo ( 31 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; if ( isPowerOfTwo ( 64 ) ) System . out . println ( " Yes " ) ; else System . out . println ( " No " ) ; } }
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
static int findMinSwaps ( int arr [ ] , int n ) {
int noOfZeroes [ ] = new int [ n ] ; int i , count = 0 ;
noOfZeroes [ n - 1 ] = 1 - arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) { noOfZeroes [ i ] = noOfZeroes [ i + 1 ] ; if ( arr [ i ] == 0 ) noOfZeroes [ i ] ++ ; }
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == 1 ) count += noOfZeroes [ i ] ; } return count ; }
public static void main ( String args [ ] ) { int ar [ ] = { 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 } ; System . out . println ( findMinSwaps ( ar , ar . length ) ) ; } }
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
static void findNumbers ( int arr [ ] , int n ) {
int sumN = ( n * ( n + 1 ) ) / 2 ;
int sumSqN = ( n * ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ;
int sum = 0 , sumSq = 0 , i ; for ( i = 0 ; i < n ; i ++ ) { sum += arr [ i ] ; sumSq += Math . pow ( arr [ i ] , 2 ) ; } int B = ( ( ( sumSq - sumSqN ) / ( sum - sumN ) ) + sumN - sum ) / 2 ; int A = sum - sumN + B ; System . out . println ( " A ▁ = ▁ " + A + " B = " + B); }
public static void main ( String [ ] args ) { int arr [ ] = { 1 , 2 , 2 , 3 , 4 } ; int n = arr . length ; findNumbers ( arr , n ) ; } }
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
static int minMovesToSort ( int arr [ ] , int n ) { int moves = 0 ; int i , mn = arr [ n - 1 ] ; for ( i = n - 2 ; i >= 0 ; i -- ) {
if ( arr [ i ] > mn ) moves += arr [ i ] - mn ;
} return moves ; }
public static void main ( String [ ] args ) { int arr [ ] = { 3 , 5 , 2 , 8 , 4 } ; int n = arr . length ; System . out . println ( minMovesToSort ( arr , n ) ) ; } }
import java . io . * ; import java . util . Arrays ; class GFG { static void findOptimalPairs ( int arr [ ] , int N ) { Arrays . sort ( arr ) ;
for ( int i = 0 , j = N - 1 ; i <= j ; i ++ , j -- ) System . out . print ( " ( " + arr [ i ] + " , ▁ " + arr [ j ] + " ) " + " ▁ " ) ; }
public static void main ( String [ ] args ) { int arr [ ] = { 9 , 6 , 5 , 1 } ; int N = arr . length ; findOptimalPairs ( arr , N ) ; } }
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
static int MatrixChainOrder ( int p [ ] , int i , int j ) { if ( i == j ) return 0 ; int min = Integer . MAX_VALUE ;
for ( int k = i ; k < j ; k ++ ) { int count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
public static void main ( String args [ ] ) { int arr [ ] = new int [ ] { 1 , 2 , 3 , 4 , 3 } ; int n = arr . length ; System . out . println ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ; } }
static int getCount ( String a , String b ) {
if ( b . length ( ) % a . length ( ) != 0 ) return - 1 ; int count = b . length ( ) / a . length ( ) ;
String str = " " ; for ( int i = 0 ; i < count ; i ++ ) { str = str + a ; } if ( str . equals ( b ) ) return count ; return - 1 ; }
public static void main ( String [ ] args ) { String a = " geeks " ; String b = " geeksgeeks " ; System . out . println ( getCount ( a , b ) ) ; } }
static int countPattern ( String str ) { int len = str . length ( ) ; boolean oneSeen = false ;
for ( int i = 0 ; i < len ; i ++ ) { char getChar = str . charAt ( i ) ;
if ( getChar == '1' && oneSeen == true ) { if ( str . charAt ( i - 1 ) == '0' ) count ++ ; }
if ( getChar == '1' && oneSeen == false ) oneSeen = true ;
if ( getChar != '0' && str . charAt ( i ) != '1' ) oneSeen = false ; } return count ; }
public static void main ( String [ ] args ) { String str = "100001abc101" ; System . out . println ( countPattern ( str ) ) ; } }
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
static int countNumbers ( int L , int R , int K ) { if ( K == 9 ) { K = 0 ; }
int totalnumbers = R - L + 1 ;
int factor9 = totalnumbers / 9 ;
int rem = totalnumbers % 9 ;
int ans = factor9 ;
for ( int i = R ; i > R - rem ; i -- ) { int rem1 = i % 9 ; if ( rem1 == K ) { ans ++ ; } } return ans ; }
public static void main ( String [ ] args ) { int L = 10 ; int R = 22 ; int K = 3 ; System . out . println ( countNumbers ( L , R , K ) ) ; } }
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
static long power ( long x , long y , long p ) {
x = x % p ; while ( y > 0 ) {
if ( ( y & 1 ) > 0 ) res = ( res * x ) % p ;
x = ( x * x ) % p ; } return res ; }
public static void main ( String [ ] args ) { long a = 3 ;
String b = "100000000000000000000000000" ; long remainderB = 0 ; long MOD = 1000000007 ;
for ( int i = 0 ; i < b . length ( ) ; i ++ ) remainderB = ( remainderB * 10 + b . charAt ( i ) - '0' ) % ( MOD - 1 ) ; System . out . println ( power ( a , remainderB , MOD ) ) ; } }
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
static int isPowerOf2 ( String s ) { char [ ] str = s . toCharArray ( ) ; int len_str = s . length ( ) ;
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
public static void main ( String [ ] args ) { String str1 = "124684622466842024680246842024662202000002" ; String str2 = "1" ; String str3 = "128" ; System . out . println ( isPowerOf2 ( str1 ) + "NEW_LINE"+isPowerOf2(str2) +NEW_LINE"NEW_LINE"+isPowerOf2(str3)); } }
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
import java . util . * ; class GFG {
static double getMaxMedian ( int [ ] arr , int n , int k ) { int size = n + k ;
Arrays . sort ( arr ) ;
if ( size % 2 == 0 ) { double median = ( double ) ( arr [ ( size / 2 ) - 1 ] + arr [ size / 2 ] ) / 2 ; return median ; }
double median1 = arr [ size / 2 ] ; return median1 ; }
class GFG { static void printSorted ( int a , int b , int c ) {
int get_max = Math . max ( a , Math . max ( b , c ) ) ;
int get_min = - Math . max ( - a , Math . max ( - b , - c ) ) ; int get_mid = ( a + b + c ) - ( get_max + get_min ) ; System . out . print ( get_min + " ▁ " + get_mid + " ▁ " + get_max ) ; }
public static void main ( String [ ] args ) { int a = 4 , b = 1 , c = 9 ; printSorted ( a , b , c ) ; } }
void sort ( int arr [ ] ) { int n = arr . length ; for ( int i = 1 ; i < n ; ++ i ) { int key = arr [ i ] ; int j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
static void printArray ( int arr [ ] ) { int n = arr . length ; for ( int i = 0 ; i < n ; ++ i ) System . out . print ( arr [ i ] + " ▁ " ) ; System . out . println ( ) ; }
public static void main ( String args [ ] ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; InsertionSort ob = new InsertionSort ( ) ; ob . sort ( arr ) ; printArray ( arr ) ; } }
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
static boolean isVowel ( char c ) { return ( c == ' a ' c == ' e ' c == ' i ' c == ' o ' c == ' u ' ) ; }
static String encryptString ( String s , int n , int k ) { int countVowels = 0 ; int countConsonants = 0 ; String ans = " " ;
for ( int l = 0 ; l <= n - k ; l ++ ) { countVowels = 0 ; countConsonants = 0 ;
for ( int r = l ; r <= l + k - 1 ; r ++ ) {
if ( isVowel ( s . charAt ( r ) ) == true ) { countVowels ++ ; } else { countConsonants ++ ; } }
ans += String . valueOf ( countVowels * countConsonants ) ; } return ans ; }
static public void main ( String [ ] args ) { String s = " hello " ; int n = s . length ( ) ; int k = 2 ; System . out . println ( encryptString ( s , n , k ) ) ; } }
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
public static int middleOfThree ( int a , int b , int c ) {
if ( a > b ) { if ( b > c ) return b ; else if ( a > c ) return c ; else return a ; } else {
if ( a > c ) return a ; else if ( b > c ) return c ; else return b ; } }
public static void main ( String [ ] args ) { int a = 20 , b = 30 , c = 40 ; System . out . println ( middleOfThree ( a , b , c ) ) ; } }
public static void printArr ( int [ ] arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) System . out . print ( arr [ i ] ) ; }
public static int compare ( int num1 , int num2 ) {
String A = Integer . toString ( num1 ) ;
String B = Integer . toString ( num2 ) ;
return ( A + B ) . compareTo ( B + A ) ; }
public static void printSmallest ( int N , int [ ] arr ) {
for ( int i = 0 ; i < N ; i ++ ) { for ( int j = i + 1 ; j < N ; j ++ ) { if ( compare ( arr [ i ] , arr [ j ] ) > 0 ) { int temp = arr [ i ] ; arr [ i ] = arr [ j ] ; arr [ j ] = temp ; } } }
printArr ( arr , N ) ; }
public static void main ( String [ ] args ) { int [ ] arr = { 5 , 6 , 2 , 9 , 21 , 1 } ; int N = arr . length ; printSmallest ( N , arr ) ; } }
static boolean isPossible ( Integer a [ ] , int b [ ] , int n , int k ) {
Arrays . sort ( a , Collections . reverseOrder ( ) ) ;
Arrays . sort ( b ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( a [ i ] + b [ i ] < k ) return false ; return true ; }
public static void main ( String [ ] args ) { Integer a [ ] = { 2 , 1 , 3 } ; int b [ ] = { 7 , 8 , 9 } ; int k = 10 ; int n = a . length ; if ( isPossible ( a , b , n , k ) ) System . out . print ( " Yes " ) ; else System . out . print ( " No " ) ; } }
static String encryptString ( String str , int n ) { int i = 0 , cnt = 0 ; String encryptedStr = " " ; while ( i < n ) {
cnt = i + 1 ;
while ( cnt -- > 0 ) encryptedStr += str . charAt ( i ) ; i ++ ; } return encryptedStr ; }
public static void main ( String [ ] args ) { String str = " geeks " ; int n = str . length ( ) ; System . out . println ( encryptString ( str , n ) ) ; } }
static int minDiff ( int n , int x , int A [ ] ) { int mn = A [ 0 ] , mx = A [ 0 ] ;
for ( int i = 0 ; i < n ; ++ i ) { mn = Math . min ( mn , A [ i ] ) ; mx = Math . max ( mx , A [ i ] ) ; }
return Math . max ( 0 , mx - mn - 2 * x ) ; }
public static void main ( String [ ] args ) { int n = 3 , x = 3 ; int A [ ] = { 1 , 3 , 6 } ;
System . out . println ( minDiff ( n , x , A ) ) ; } }
