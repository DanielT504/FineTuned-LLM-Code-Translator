using System ; class GFG {
static void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { z1 = - d1 / c1 ; d = Math . Abs ( ( c2 * z1 + d2 ) ) / ( float ) ( Math . Sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; Console . Write ( " Perpendicular ▁ distance ▁ is ▁ " + d ) ; } else Console . Write ( " Planes ▁ are ▁ not ▁ parallel " ) ; }
public static void Main ( ) { float a1 = 1 ; float b1 = 2 ; float c1 = - 1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = - 3 ; float d2 = - 4 ; distance ( a1 , b1 , c1 , d1 ,
using System ; class GFG {
static int Series ( int n ) { int i ; int sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
public static void Main ( ) { int n = 3 ; int res = Series ( n ) ; Console . Write ( res ) ; } }
using System ; class GFG {
static bool areElementsContiguous ( int [ ] arr , int n ) {
Array . Sort ( arr ) ;
for ( int i = 1 ; i < n ; i ++ ) if ( arr [ i ] - arr [ i - 1 ] > 1 ) return false ; return true ; }
public static void Main ( ) { int [ ] arr = { 5 , 2 , 3 , 6 , 4 , 4 , 6 , 6 } ; int n = arr . Length ; if ( areElementsContiguous ( arr , n ) ) Console . WriteLine ( " Yes " ) ; else Console . WriteLine ( " No " ) ; } }
using System ; class GFG {
static void leftRotatebyOne ( int [ ] arr , int n ) { int i , temp = arr [ 0 ] ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
static void leftRotate ( int [ ] arr , int d , int n ) { for ( int i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
static void printArray ( int [ ] arr , int size ) { for ( int i = 0 ; i < size ; i ++ ) Console . Write ( arr [ i ] + " ▁ " ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ; } }
using System ; class GFG {
static int findFirstMissing ( int [ ] array , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
public static void Main ( ) { int [ ] arr = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = arr . Length ; Console . Write ( " smallest ▁ Missing ▁ element ▁ is ▁ : ▁ " + findFirstMissing ( arr , 0 , n - 1 ) ) ; } }
using System ; class GFG {
static int FindMaxSum ( int [ ] arr , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 5 , 5 , 10 , 100 , 10 , 5 } ; Console . Write ( FindMaxSum ( arr , arr . Length ) ) ; } }
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
static double Convert ( double radian ) { double pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
public static void Main ( ) { double radian = 5.0 ; double degree = Convert ( radian ) ; Console . Write ( " degree ▁ = ▁ " + degree ) ; } }
using System ; class GFG { static int subtract ( int x , int y ) {
while ( y != 0 ) {
int borrow = ( ~ x ) & y ;
x = x ^ y ;
y = borrow << 1 ; } return x ; }
public static void Main ( ) { int x = 29 , y = 13 ; Console . WriteLine ( " x ▁ - ▁ y ▁ is ▁ " + subtract ( x , y ) ) ; } }
using System ; class GFG { static int subtract ( int x , int y ) { if ( y == 0 ) return x ; return subtract ( x ^ y , ( ~ x & y ) << 1 ) ; }
public static void Main ( ) { int x = 29 , y = 13 ; Console . WriteLine ( " x ▁ - ▁ y ▁ is ▁ " + subtract ( x , y ) ) ; } }
using System ; class GFG {
static void reverse ( String str ) { if ( ( str == null ) || ( str . Length <= 1 ) ) Console . Write ( str ) ; else { Console . Write ( str [ str . Length - 1 ] ) ; reverse ( str . Substring ( 0 , ( str . Length - 1 ) ) ) ; } }
public static void Main ( ) { String str = " Geeks ▁ for ▁ Geeks " ; reverse ( str ) ; } }
using System ; class GFG {
static int cola = 2 , rowa = 3 ; static int colb = 3 , rowb = 2 ;
static void Kroneckerproduct ( int [ , ] A , int [ , ] B ) { int [ , ] C = new int [ rowa * rowb , cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 , j + k + 1 ] = A [ i , j ] * B [ k , l ] ; Console . Write ( C [ i + l + 1 , j + k + 1 ] + " ▁ " ) ; } } Console . WriteLine ( ) ; } } }
public static void Main ( ) { int [ , ] A = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } ; int [ , ] B = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; } }
using System ; class GFG {
static void sort ( int [ ] arr ) { int n = arr . Length ;
for ( int i = 0 ; i < n - 1 ; i ++ ) {
int min_idx = i ; for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
int temp = arr [ min_idx ] ; arr [ min_idx ] = arr [ i ] ; arr [ i ] = temp ; } }
static void printArray ( int [ ] arr ) { int n = arr . Length ; for ( int i = 0 ; i < n ; ++ i ) Console . Write ( arr [ i ] + " ▁ " ) ; Console . WriteLine ( ) ; }
public static void Main ( ) { int [ ] arr = { 64 , 25 , 12 , 22 , 11 } ; sort ( arr ) ; Console . WriteLine ( " Sorted ▁ array " ) ; printArray ( arr ) ; } }
using System ; class GFG {
static int MatrixChainOrder ( int [ ] p , int n ) {
int [ , ] m = new int [ n , n ] ; int i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i , i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; if ( j == n ) continue ; m [ i , j ] = int . MaxValue ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i , k ] + m [ k + 1 , j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i , j ] ) m [ i , j ] = q ; } } } return m [ 1 , n - 1 ] ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 2 , 3 , 4 } ; int size = arr . Length ; Console . Write ( " Minimum ▁ number ▁ of ▁ " + " multiplications ▁ is ▁ " + MatrixChainOrder ( arr , size ) ) ; } }
using System ; class GFG {
static int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; return - 1 ; }
public static void Main ( ) { Console . WriteLine ( multiply ( 5 , - 11 ) ) ; } }
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
static int Add ( int x , int y ) { if ( y == 0 ) return x ; else return Add ( x ^ y , ( x & y ) << 1 ) ; }
using System ; class GFG {
static int countSetBits ( int n ) { int count = 0 ; while ( n > 0 ) { count += n & 1 ; n >>= 1 ; } return count ; }
public static void Main ( ) { int i = 9 ; Console . Write ( countSetBits ( i ) ) ; } }
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
static bool isPowerOfTwo ( int x ) {
return x != 0 && ( ( x & ( x - 1 ) ) == 0 ) ; }
public static void Main ( ) { Console . WriteLine ( isPowerOfTwo ( 31 ) ? " Yes " : " No " ) ; Console . WriteLine ( isPowerOfTwo ( 64 ) ? " Yes " : " No " ) ; } }
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
using System ; class GFG {
static int MatrixChainOrder ( int [ ] p , int i , int j ) { if ( i == j ) return 0 ; int min = int . MaxValue ;
for ( int k = i ; k < j ; k ++ ) { int count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
public static void Main ( ) { int [ ] arr = new int [ ] { 1 , 2 , 3 , 4 , 3 } ; int n = arr . Length ; Console . Write ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ " + MatrixChainOrder ( arr , 1 , n - 1 ) ) ; } }
using System ; class GFG {
static double Perimeter ( double s , int n ) { double perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
static public void Main ( ) {
int n = 5 ;
double s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; Console . WriteLine ( " Perimeter ▁ of ▁ Regular ▁ Polygon " + " ▁ with ▁ " + n + " ▁ sides ▁ of ▁ length ▁ " + s + " ▁ = ▁ " + peri ) ; } }
using System ; class GFG {
static void shortest_distance ( float x1 , float y1 , float z1 , float a , float b , float c , float d ) { d = Math . Abs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; float e = ( float ) Math . Sqrt ( a * a + b * b + c * c ) ; Console . Write ( " Perpendicular ▁ distance ▁ " + " is ▁ " + d / e ) ; }
public static void Main ( ) { float x1 = 4 ; float y1 = - 4 ; float z1 = 3 ; float a = 2 ; float b = - 2 ; float c = 5 ; float d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ; } }
using System ; class GFG {
static int averageOdd ( int n ) { if ( n % 2 == 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } return ( n + 1 ) / 2 ; }
public static void Main ( ) { int n = 15 ; Console . Write ( averageOdd ( n ) ) ; } }
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
using System ; class GFG {
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
using System ; class GFG {
static int averageEven ( int n ) { if ( n % 2 != 0 ) { Console . Write ( " Invalid ▁ Input " ) ; return - 1 ; } return ( n + 2 ) / 2 ; }
public static void Main ( ) { int n = 16 ; Console . Write ( averageEven ( n ) ) ; } }
using System ; class Division {
static int fac ( int n ) { if ( n == 0 ) return 1 ; return n * fac ( n - 1 ) ; }
static int div ( int x ) { int ans = 0 ; for ( int i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
static int sumFactDiv ( int n ) { return div ( fac ( n ) ) ; }
public static void Main ( ) { int n = 4 ; Console . Write ( sumFactDiv ( n ) ) ; } }
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
using System ; class GFG { static int rev_num = 0 ; static int base_pos = 1 ; static int reversDigits ( int num ) { if ( num > 0 ) { reversDigits ( num / 10 ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
public static void Main ( ) { int num = 4562 ; Console . WriteLine ( reversDigits ( num ) ) ; } }
using System ; public class GFG { static int multiplyBySevenByEight ( int n ) {
return ( n - ( n >> 3 ) ) ; }
public static void Main ( ) { int n = 9 ; Console . WriteLine ( multiplyBySevenByEight ( n ) ) ; } }
using System ; public class GFG { static int multiplyBySevenByEight ( int n ) {
return ( ( n << 3 ) - n ) >> 3 ; }
public static void Main ( ) { int n = 15 ; Console . WriteLine ( multiplyBySevenByEight ( n ) ) ; } }
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
using System ; class GFG {
static int count ( int [ ] S , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
public static void Main ( ) { int [ ] arr = { 1 , 2 , 3 } ; int m = arr . Length ; Console . Write ( count ( arr , m , 4 ) ) ; } }
using System ; class GFG {
static double Area ( int b1 , int b2 , int h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
public static void Main ( ) { int base1 = 8 , base2 = 10 , height = 6 ; double area = Area ( base1 , base2 , height ) ; Console . WriteLine ( " Area ▁ is : ▁ " + area ) ; } }
