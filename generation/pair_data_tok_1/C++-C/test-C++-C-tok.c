#include <stdio.h> NEW_LINE #include <math.h>
void distance ( float a1 , float b1 , float c1 , float d1 , float a2 , float b2 , float c2 , float d2 ) { float x1 , y1 , z1 , d ; if ( a1 / a2 == b1 / b2 && b1 / b2 == c1 / c2 ) { x1 = y1 = 0 ; z1 = - d1 / c1 ; d = fabs ( ( c2 * z1 + d2 ) ) / ( sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ) ; printf ( " Perpendicular ▁ distance ▁ is ▁ % f STRNEWLINE " , d ) ; } else printf ( " Planes ▁ are ▁ not ▁ parallel " ) ; return ; }
int main ( ) { float a1 = 1 ; float b1 = 2 ; float c1 = -1 ; float d1 = 1 ; float a2 = 3 ; float b2 = 6 ; float c2 = -3 ; float d2 = -4 ; distance ( a1 , b1 , c1 , d1 , a2 , b2 , c2 , d2 ) ; return 0 ; }
#include <stdio.h>
int Series ( int n ) { int i ; int sums = 0 ; for ( i = 1 ; i <= n ; i ++ ) sums += ( i * i ) ; return sums ; }
int main ( ) { int n = 3 ; int res = Series ( n ) ; printf ( " % d " , res ) ; }
#include <stdio.h>
void leftRotatebyOne ( int arr [ ] , int n ) ; void leftRotatebyOne ( int arr [ ] , int n ) { int temp = arr [ 0 ] , i ; for ( i = 0 ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; arr [ n - 1 ] = temp ; }
void leftRotate ( int arr [ ] , int d , int n ) { int i ; for ( i = 0 ; i < d ; i ++ ) leftRotatebyOne ( arr , n ) ; }
void printArray ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ; return 0 ; }
#include <stdio.h>
int findFirstMissing ( int array [ ] , int start , int end ) { if ( start > end ) return end + 1 ; if ( start != array [ start ] ) return start ; int mid = ( start + end ) / 2 ;
if ( array [ mid ] == mid ) return findFirstMissing ( array , mid + 1 , end ) ; return findFirstMissing ( array , start , mid ) ; }
int main ( ) { int arr [ ] = { 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Smallest ▁ missing ▁ element ▁ is ▁ % d " , findFirstMissing ( arr , 0 , n - 1 ) ) ; return 0 ; }
#include <stdio.h>
int FindMaxSum ( int arr [ ] , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
int main ( ) { int arr [ ] = { 5 , 5 , 10 , 100 , 10 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d ▁ n " , FindMaxSum ( arr , n ) ) ; return 0 ; }
# include <stdio.h> NEW_LINE # include <stdbool.h> NEW_LINE bool isMajority ( int arr [ ] , int n , int x ) { int i ;
int last_index = n % 2 ? ( n / 2 + 1 ) : ( n / 2 ) ;
for ( i = 0 ; i < last_index ; i ++ ) {
if ( arr [ i ] == x && arr [ i + n / 2 ] == x ) return 1 ; } return 0 ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 4 , 4 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 4 ; if ( isMajority ( arr , n , x ) ) printf ( " % d ▁ appears ▁ more ▁ than ▁ % d ▁ times ▁ in ▁ arr [ ] " , x , n / 2 ) ; else printf ( " % d ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ % d ▁ times ▁ in ▁ arr [ ] " , x , n / 2 ) ; return 0 ; }
# include <stdio.h> NEW_LINE # include <stdbool.h>
int _binarySearch ( int arr [ ] , int low , int high , int x ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( ( mid == 0 x > arr [ mid - 1 ] ) && ( arr [ mid ] == x ) ) return mid ; else if ( x > arr [ mid ] ) return _binarySearch ( arr , ( mid + 1 ) , high , x ) ; else return _binarySearch ( arr , low , ( mid - 1 ) , x ) ; } return -1 ; }
bool isMajority ( int arr [ ] , int n , int x ) {
int i = _binarySearch ( arr , 0 , n - 1 , x ) ;
if ( i == -1 ) return false ;
if ( ( ( i + n / 2 ) <= ( n - 1 ) ) && arr [ i + n / 2 ] == x ) return true ; else return false ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 3 ; if ( isMajority ( arr , n , x ) ) printf ( " % d ▁ appears ▁ more ▁ than ▁ % d ▁ times ▁ in ▁ arr [ ] " , x , n / 2 ) ; else printf ( " % d ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ % d ▁ times ▁ in ▁ arr [ ] " , x , n / 2 ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdbool.h> NEW_LINE bool isMajorityElement ( int arr [ ] , int n , int key ) { if ( arr [ n / 2 ] == key ) return true ; else return false ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 3 , 3 , 3 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 3 ; if ( isMajorityElement ( arr , n , x ) ) printf ( " % d ▁ appears ▁ more ▁ than ▁ % d ▁ times ▁ in ▁ arr [ ] " , x , n / 2 ) ; else printf ( " % d ▁ does ▁ not ▁ appear ▁ more ▁ than ▁ % d ▁ times ▁ in ▁ " " arr [ ] " , x , n / 2 ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h>
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int cutRod ( int price [ ] , int n ) { int val [ n + 1 ] ; val [ 0 ] = 0 ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { int max_val = INT_MIN ; for ( j = 0 ; j < i ; j ++ ) max_val = max ( max_val , price [ j ] + val [ i - j - 1 ] ) ; val [ i ] = max_val ; } return val [ n ] ; }
int main ( ) { int arr [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Maximum ▁ Obtainable ▁ Value ▁ is ▁ % d " , cutRod ( arr , size ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int t [ 9 ] [ 9 ] ;
int un_kp ( int price [ ] , int length [ ] , int Max_len , int n ) {
if ( n == 0 Max_len == 0 ) { return 0 ; }
if ( length [ n - 1 ] <= Max_len ) { t [ n ] [ Max_len ] = max ( price [ n - 1 ] + un_kp ( price , length , Max_len - length [ n - 1 ] , n ) , un_kp ( price , length , Max_len , n - 1 ) ) ; }
else { t [ n ] [ Max_len ] = un_kp ( price , length , Max_len , n - 1 ) ; }
return t [ n ] [ Max_len ] ; }
int main ( ) { int price [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int n = sizeof ( price ) / sizeof ( price [ 0 ] ) ; int length [ n ] ; for ( int i = 0 ; i < n ; i ++ ) { length [ i ] = i + 1 ; } int Max_len = n ;
printf ( " Maximum ▁ obtained ▁ value ▁ is ▁ % d ▁ STRNEWLINE " , un_kp ( price , length , n , Max_len ) ) ; }
#include <stdio.h>
double Convert ( double radian ) { double pi = 3.14159 ; return ( radian * ( 180 / pi ) ) ; }
int main ( ) { double radian = 5.0 ; double degree = Convert ( radian ) ; printf ( " % .5lf " , degree ) ; return 0 ; }
#include <stdio.h> NEW_LINE int subtract ( int x , int y ) {
while ( y != 0 ) {
int borrow = ( ~ x ) & y ;
x = x ^ y ;
y = borrow << 1 ; } return x ; }
int main ( ) { int x = 29 , y = 13 ; printf ( " x ▁ - ▁ y ▁ is ▁ % d " , subtract ( x , y ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int subtract ( int x , int y ) { if ( y == 0 ) return x ; return subtract ( x ^ y , ( ~ x & y ) << 1 ) ; }
int main ( ) { int x = 29 , y = 13 ; printf ( " x ▁ - ▁ y ▁ is ▁ % d " , subtract ( x , y ) ) ; return 0 ; }
# include <stdio.h>
void reverse ( char * str ) { if ( * str ) { reverse ( str + 1 ) ; printf ( " % c " , * str ) ; } }
int main ( ) { char a [ ] = " Geeks ▁ for ▁ Geeks " ; reverse ( a ) ; return 0 ; }
#include <stdio.h>
const int cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
void Kroneckerproduct ( int A [ ] [ cola ] , int B [ ] [ colb ] ) { int C [ rowa * rowb ] [ cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; printf ( " % d TABSYMBOL " , C [ i + l + 1 ] [ j + k + 1 ] ) ; } } printf ( " STRNEWLINE " ) ; } } }
int main ( ) { int A [ 3 ] [ 2 ] = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } , B [ 2 ] [ 3 ] = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; return 0 ; }
#include <stdio.h>
void swap ( int * xp , int * yp ) { int temp = * xp ; * xp = * yp ; * yp = temp ; }
void selectionSort ( int arr [ ] , int n ) { int i , j , min_idx ;
for ( i = 0 ; i < n - 1 ; i ++ ) {
min_idx = i ; for ( j = i + 1 ; j < n ; j ++ ) if ( arr [ j ] < arr [ min_idx ] ) min_idx = j ;
swap ( & arr [ min_idx ] , & arr [ i ] ) ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 64 , 25 , 12 , 22 , 11 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; selectionSort ( arr , n ) ; printf ( " Sorted ▁ array : ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h>
int MatrixChainOrder ( int p [ ] , int n ) {
int m [ n ] [ n ] ; int i , j , k , L , q ;
for ( i = 1 ; i < n ; i ++ ) m [ i ] [ i ] = 0 ;
for ( L = 2 ; L < n ; L ++ ) { for ( i = 1 ; i < n - L + 1 ; i ++ ) { j = i + L - 1 ; m [ i ] [ j ] = INT_MAX ; for ( k = i ; k <= j - 1 ; k ++ ) {
q = m [ i ] [ k ] + m [ k + 1 ] [ j ] + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( q < m [ i ] [ j ] ) m [ i ] [ j ] = q ; } } } return m [ 1 ] [ n - 1 ] ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ % d ▁ " , MatrixChainOrder ( arr , size ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int t [ 9 ] [ 9 ] ;
int un_kp ( int price [ ] , int length [ ] , int Max_len , int n ) {
if ( n == 0 Max_len == 0 ) { return 0 ; }
if ( length [ n - 1 ] <= Max_len ) { t [ n ] [ Max_len ] = max ( price [ n - 1 ] + un_kp ( price , length , Max_len - length [ n - 1 ] , n ) , un_kp ( price , length , Max_len , n - 1 ) ) ; }
else { t [ n ] [ Max_len ] = un_kp ( price , length , Max_len , n - 1 ) ; }
return t [ n ] [ Max_len ] ; }
int main ( ) { int price [ ] = { 1 , 5 , 8 , 9 , 10 , 17 , 17 , 20 } ; int n = sizeof ( price ) / sizeof ( price [ 0 ] ) ; int length [ n ] ; for ( int i = 0 ; i < n ; i ++ ) { length [ i ] = i + 1 ; } int Max_len = n ;
printf ( " Maximum ▁ obtained ▁ value ▁ is ▁ % d ▁ STRNEWLINE " , un_kp ( price , length , n , Max_len ) ) ; }
#include <stdio.h>
int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; }
int main ( ) { printf ( " % d " , multiply ( 5 , -11 ) ) ; getchar ( ) ; return 0 ; }
void printPascal ( int n ) {
int arr [ n ] [ n ] ;
for ( int line = 0 ; line < n ; line ++ ) {
for ( int i = 0 ; i <= line ; i ++ ) {
if ( line == i i == 0 ) arr [ line ] [ i ] = 1 ;
else arr [ line ] [ i ] = arr [ line - 1 ] [ i - 1 ] + arr [ line - 1 ] [ i ] ; printf ( " % d ▁ " , arr [ line ] [ i ] ) ; } printf ( " STRNEWLINE " ) ; } }
int main ( ) { int n = 5 ; printPascal ( n ) ; return 0 ; }
void printPascal ( int n ) { for ( int line = 1 ; line <= n ; line ++ ) {
int C = 1 ; for ( int i = 1 ; i <= line ; i ++ ) {
printf ( " % d ▁ " , C ) ; C = C * ( line - i ) / i ; } printf ( " STRNEWLINE " ) ; } }
int main ( ) { int n = 5 ; printPascal ( n ) ; return 0 ; }
#include <stdio.h> NEW_LINE int Add ( int x , int y ) {
while ( y != 0 ) {
int carry = x & y ;
x = x ^ y ;
y = carry << 1 ; } return x ; }
int main ( ) { printf ( " % d " , Add ( 15 , 32 ) ) ; return 0 ; }
#include <stdio.h>
unsigned int countSetBits ( unsigned int n ) { unsigned int count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
int main ( ) { int i = 9 ; printf ( " % d " , countSetBits ( i ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int num_to_bits [ 16 ] = { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
unsigned int countSetBitsRec ( unsigned int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
int main ( ) { int num = 31 ; printf ( " % d STRNEWLINE " , countSetBitsRec ( num ) ) ; }
#include <stdio.h>
int countSetBits ( int N ) { int count = 0 ;
for ( int i = 0 ; i < sizeof ( int ) * 8 ; i ++ ) { if ( N & ( 1 << i ) ) count ++ ; } return count ; }
int main ( ) { int N = 15 ; printf ( " % d " , countSetBits ( N ) ) ; return 0 ; }
# include <stdio.h> NEW_LINE # define bool  int
bool getParity ( unsigned int n ) { bool parity = 0 ; while ( n ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
int main ( ) { unsigned int n = 7 ; printf ( " Parity ▁ of ▁ no ▁ % d ▁ = ▁ % s " , n , ( getParity ( n ) ? " odd " : " even " ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdbool.h> NEW_LINE #include <math.h>
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdbool.h>
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 2 != 0 ) return 0 ; n = n / 2 ; } return 1 ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define bool  int
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
#include <stdio.h>
void printTwoOdd ( int arr [ ] , int size ) { int xor2 = arr [ 0 ] ;
int set_bit_no ;
int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } printf ( " The two ODD elements are % d & % d " }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoOdd ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
bool findPair ( int arr [ ] , int size , int n ) {
int i = 0 ; int j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { printf ( " Pair ▁ Found : ▁ ( % d , ▁ % d ) " , arr [ i ] , arr [ j ] ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } printf ( " No ▁ such ▁ pair " ) ; return false ; }
int main ( ) { int arr [ ] = { 1 , 8 , 30 , 40 , 100 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 60 ; findPair ( arr , size , n ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h>
int MatrixChainOrder ( int p [ ] , int i , int j ) { if ( i == j ) return 0 ; int k ; int min = INT_MAX ; int count ;
for ( k = i ; k < j ; k ++ ) { count = MatrixChainOrder ( p , i , k ) + MatrixChainOrder ( p , k + 1 , j ) + p [ i - 1 ] * p [ k ] * p [ j ] ; if ( count < min ) min = count ; }
return min ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Minimum ▁ number ▁ of ▁ multiplications ▁ is ▁ % d ▁ " , MatrixChainOrder ( arr , 1 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
float Perimeter ( float s , int n ) { float perimeter = 1 ;
perimeter = n * s ; return perimeter ; }
int main ( ) {
int n = 5 ;
float s = 2.5 , peri ;
peri = Perimeter ( s , n ) ; printf ( " Perimeter ▁ of ▁ Regular ▁ Polygon STRNEWLINE " " ▁ with ▁ % d ▁ sides ▁ of ▁ length ▁ % f ▁ = ▁ % f STRNEWLINE " , n , s , peri ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <math.h>
void shortest_distance ( float x1 , float y1 , float z1 , float a , float b , float c , float d ) { d = fabs ( ( a * x1 + b * y1 + c * z1 + d ) ) ; float e = sqrt ( a * a + b * b + c * c ) ; printf ( " Perpendicular ▁ distance ▁ is ▁ % f " , d / e ) ; return ; }
int main ( ) { float x1 = 4 ; float y1 = -4 ; float z1 = 3 ; float a = 2 ; float b = -2 ; float c = 5 ; float d = 8 ;
shortest_distance ( x1 , y1 , z1 , a , b , c , d ) ; }
#include <stdio.h>
int averageEven ( int n ) { if ( n % 2 != 0 ) { printf ( " Invalid ▁ Input " ) ; return -1 ; } return ( n + 2 ) / 2 ; }
int main ( ) { int n = 16 ; printf ( " % d " , averageEven ( n ) ) ; return 0 ; }
#include <stdio.h>
int fact ( int n ) { if ( n == 0 ) return 1 ; return n * fact ( n - 1 ) ; }
int div ( int x ) { int ans = 0 ; for ( int i = 1 ; i <= x ; i ++ ) if ( x % i == 0 ) ans += i ; return ans ; }
int sumFactDiv ( int n ) { return div ( fact ( n ) ) ; }
int main ( ) { int n = 4 ; printf ( " % d " , sumFactDiv ( n ) ) ; }
#include <stdio.h> NEW_LINE #include <math.h>
void printDivisors ( int n ) { int i ; for ( i = 1 ; i * i < n ; i ++ ) { if ( n % i == 0 ) printf ( " % d ▁ " , i ) ; } if ( i - ( n / i ) == 1 ) { i -- ; } for ( ; i >= 1 ; i -- ) { if ( n % i == 0 ) printf ( " % d ▁ " , n / i ) ; } }
int main ( ) { printf ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ) ; printDivisors ( 100 ) ; return 0 ; }
#include <stdio.h>
void printDivisors ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) if ( n % i == 0 ) printf ( " % d ▁ " , i ) ; }
int main ( ) { printf ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ) ; printDivisors ( 100 ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <math.h>
void printDivisors ( int n ) {
for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) { if ( n % i == 0 ) {
if ( n / i == i ) printf ( " % d ▁ " , i ) ;
printf ( " % d ▁ % d ▁ " , i , n / i ) ; } } }
int main ( ) { printf ( " The ▁ divisors ▁ of ▁ 100 ▁ are : ▁ STRNEWLINE " ) ; printDivisors ( 100 ) ; return 0 ; }
#include <stdio.h> ;
int reversDigits ( int num ) { static int rev_num = 0 ; static int base_pos = 1 ; if ( num > 0 ) { reversDigits ( num / 10 ) ; rev_num += ( num % 10 ) * base_pos ; base_pos *= 10 ; } return rev_num ; }
int main ( ) { int num = 4562 ; printf ( " Reverse ▁ of ▁ no . ▁ is ▁ % d " , reversDigits ( num ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int multiplyBySevenByEight ( unsigned int n ) {
return ( n - ( n >> 3 ) ) ; }
int main ( ) { unsigned int n = 9 ; printf ( " % d " , multiplyBySevenByEight ( n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int binarySearch ( int a [ ] , int item , int low , int high ) { while ( low <= high ) { int mid = low + ( high - low ) / 2 ; if ( item == a [ mid ] ) return mid + 1 ; else if ( item > a [ mid ] ) low = mid + 1 ; else high = mid - 1 ; } return low ; }
void insertionSort ( int a [ ] , int n ) { int i , loc , j , k , selected ; for ( i = 1 ; i < n ; ++ i ) { j = i - 1 ; selected = a [ i ] ;
loc = binarySearch ( a , selected , 0 , j ) ;
while ( j >= loc ) { a [ j + 1 ] = a [ j ] ; j -- ; } a [ j + 1 ] = selected ; } }
int main ( ) { int a [ ] = { 37 , 23 , 0 , 17 , 12 , 72 , 31 , 46 , 100 , 88 , 54 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) , i ; insertionSort ( a , n ) ; printf ( " Sorted ▁ array : ▁ STRNEWLINE " ) ; for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , a [ i ] ) ; return 0 ; }
#include <math.h> NEW_LINE #include <stdio.h>
void insertionSort ( int arr [ ] , int n ) { int i , key , j ; for ( i = 1 ; i < n ; i ++ ) { key = arr [ i ] ; j = i - 1 ;
while ( j >= 0 && arr [ j ] > key ) { arr [ j + 1 ] = arr [ j ] ; j = j - 1 ; } arr [ j + 1 ] = key ; } }
void printArray ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; insertionSort ( arr , n ) ; printArray ( arr , n ) ; return 0 ; }
#include <stdio.h>
int count ( int S [ ] , int m , int n ) {
if ( n == 0 ) return 1 ;
if ( n < 0 ) return 0 ;
if ( m <= 0 && n >= 1 ) return 0 ;
return count ( S , m - 1 , n ) + count ( S , m , n - S [ m - 1 ] ) ; }
int main ( ) { int i , j ; int arr [ ] = { 1 , 2 , 3 } ; int m = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d ▁ " , count ( arr , m , 4 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
double Area ( int b1 , int b2 , int h ) { return ( ( b1 + b2 ) / 2 ) * h ; }
int main ( ) { int base1 = 8 , base2 = 10 , height = 6 ; double area = Area ( base1 , base2 , height ) ; printf ( " Area ▁ is : ▁ % .1lf " , area ) ; return 0 ; }
