void search ( char pat [ ] , char txt [ ] ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
{ printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { char txt [ ] = " ABCEABCDABCEABCD " ; char pat [ ] = " ABCD " ; search ( pat , txt ) ; return 0 ; }
bool isPalRec ( char str [ ] , int s , int e ) {
if ( s == e ) return true ;
if ( str [ s ] != str [ e ] ) return false ;
if ( s < e + 1 ) return isPalRec ( str , s + 1 , e - 1 ) ; return true ; } bool isPalindrome ( char str [ ] ) { int n = strlen ( str ) ;
if ( n == 0 ) return true ; return isPalRec ( str , 0 , n - 1 ) ; }
int main ( ) { char str [ ] = " geeg " ; if ( isPalindrome ( str ) ) printf ( " Yes " ) ; else printf ( " No " ) ; return 0 ; }
void CalPeri ( ) { int s = 5 , Perimeter ; Perimeter = 10 * s ; printf ( " The ▁ Perimeter ▁ of ▁ Decagon ▁ is ▁ : ▁ % d " , Perimeter ) ; }
int main ( ) { CalPeri ( ) ; return 0 ; }
void distance ( float a1 , float b1 , float c1 , float a2 , float b2 , float c2 ) { float d = ( a1 * a2 + b1 * b2 + c1 * c2 ) ; float e1 = sqrt ( a1 * a1 + b1 * b1 + c1 * c1 ) ; float e2 = sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ; d = d / ( e1 * e2 ) ; float pi = 3.14159 ; float A = ( 180 / pi ) * ( acos ( d ) ) ; printf ( " Angle ▁ is ▁ % .2f ▁ degree " , A ) ; }
int main ( ) { float a1 = 1 ; float b1 = 1 ; float c1 = 2 ; float d1 = 1 ; float a2 = 2 ; float b2 = -1 ; float c2 = 1 ; float d2 = -4 ; distance ( a1 , b1 , c1 , a2 , b2 , c2 ) ; return 0 ; }
void mirror_point ( float a , float b , float c , float d , float x1 , float y1 , float z1 ) { float k = ( - a * x1 - b * y1 - c * z1 - d ) / ( float ) ( a * a + b * b + c * c ) ; float x2 = a * k + x1 ; float y2 = b * k + y1 ; float z2 = c * k + z1 ; float x3 = 2 * x2 - x1 ; float y3 = 2 * y2 - y1 ; float z3 = 2 * z2 - z1 ; printf ( " x3 ▁ = ▁ % .1f ▁ " , x3 ) ; printf ( " y3 ▁ = ▁ % .1f ▁ " , y3 ) ; printf ( " z3 ▁ = ▁ % .1f ▁ " , z3 ) ; }
int main ( ) { float a = 1 ; float b = -2 ; float c = 0 ; float d = 0 ; float x1 = -1 ; float y1 = 3 ; float z1 = 4 ;
mirror_point ( a , b , c , d , x1 , y1 , z1 ) ; }
void calculateSpan ( int price [ ] , int n , int S [ ] ) {
S [ 0 ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) {
S [ i ] = 1 ;
for ( int j = i - 1 ; ( j >= 0 ) && ( price [ i ] >= price [ j ] ) ; j -- ) S [ i ] ++ ; } }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; }
int main ( ) { int price [ ] = { 10 , 4 , 5 , 90 , 120 , 80 } ; int n = sizeof ( price ) / sizeof ( price [ 0 ] ) ; int S [ n ] ;
calculateSpan ( price , n , S ) ;
void printNGE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] < arr [ j ] ) { next = arr [ j ] ; break ; } } printf ( " % d ▁ - - ▁ % dn " , arr [ i ] , next ) ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNGE ( arr , n ) ; return 0 ; }
int gcd ( int a , int b ) {
if ( a == 0 && b == 0 ) return 0 ; if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
int main ( ) { int a = 0 , b = 56 ; printf ( " GCD ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d ▁ " , a , b , gcd ( a , b ) ) ; return 0 ; }
int msbPos ( int n ) { int pos = 0 ; while ( n != 0 ) { pos ++ ;
n = n >> 1 ; } return pos ; }
int josephify ( int n ) {
int position = msbPos ( n ) ;
int j = 1 << ( position - 1 ) ;
n = n ^ j ;
n = n << 1 ;
n = n | 1 ; return n ; }
int main ( ) { int n = 41 ; printf ( " % d STRNEWLINE " , josephify ( n ) ) ; return 0 ; }
int pairAndSum ( int arr [ ] , int n ) {
for ( int i = 0 ; i < 32 ; i ++ ) {
for ( int j = 0 ; j < n ; j ++ ) if ( ( arr [ j ] & ( 1 << i ) ) ) k ++ ;
ans += ( 1 << i ) * ( k * ( k - 1 ) / 2 ) ; } return ans ; }
int main ( ) { int arr [ ] = { 5 , 10 , 15 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << pairAndSum ( arr , n ) << endl ; return 0 ; }
function countSquares ( n ) {
return ( n * ( n + 1 ) / 2 ) * ( 2 * n + 1 ) / 3 ; }
let n = 4 ; document . write ( " Count ▁ of ▁ squares ▁ is ▁ " + countSquares ( n ) ) ;
int gcd ( int a , int b ) {
if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
int main ( ) { int a = 98 , b = 56 ; printf ( " GCD ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d ▁ " , a , b , gcd ( a , b ) ) ; return 0 ; }
int largest ( int arr [ ] , int n ) { int i ;
int max = arr [ 0 ] ;
for ( i = 1 ; i < n ; i ++ ) if ( arr [ i ] > max ) max = arr [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 10 , 324 , 45 , 90 , 9808 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Largest ▁ in ▁ given ▁ array ▁ is ▁ % d " , largest ( arr , n ) ) ; return 0 ; }
void print2largest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { printf ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = INT_MIN ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == INT_MIN ) printf ( " There ▁ is ▁ no ▁ second ▁ largest ▁ element STRNEWLINE " ) ; else printf ( " The ▁ second ▁ largest ▁ element ▁ is ▁ % dn " , second ) ; }
int main ( ) { int arr [ ] = { 12 , 35 , 1 , 10 , 34 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; print2largest ( arr , n ) ; return 0 ; }
int minJumps ( int arr [ ] , int l , int h ) {
if ( h == l ) return 0 ;
if ( arr [ l ] == 0 ) return INT_MAX ;
int min = INT_MAX ; for ( int i = l + 1 ; i <= h && i <= l + arr [ l ] ; i ++ ) { int jumps = minJumps ( arr , i , h ) ; if ( jumps != INT_MAX && jumps + 1 < min ) min = jumps + 1 ; } return min ; }
int main ( ) { int arr [ ] = { 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Minimum ▁ number ▁ of ▁ jumps ▁ to ▁ reach ▁ end ▁ is ▁ % d ▁ " , minJumps ( arr , 0 , n - 1 ) ) ; return 0 ; }
int smallestSubWithSum ( int arr [ ] , int n , int x ) {
int min_len = n + 1 ;
for ( int start = 0 ; start < n ; start ++ ) {
int curr_sum = arr [ start ] ;
if ( curr_sum > x ) return 1 ;
for ( int end = start + 1 ; end < n ; end ++ ) {
curr_sum += arr [ end ] ;
if ( curr_sum > x && ( end - start + 1 ) < min_len ) min_len = ( end - start + 1 ) ; } } return min_len ; }
int main ( ) { int arr1 [ ] = { 1 , 4 , 45 , 6 , 10 , 19 } ; int x = 51 ; int n1 = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; int res1 = smallestSubWithSum ( arr1 , n1 , x ) ; ( res1 == n1 + 1 ) ? cout << " Not ▁ possible STRNEWLINE " : cout << res1 << endl ; int arr2 [ ] = { 1 , 10 , 5 , 2 , 7 } ; int n2 = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; x = 9 ; int res2 = smallestSubWithSum ( arr2 , n2 , x ) ; ( res2 == n2 + 1 ) ? cout << " Not ▁ possible STRNEWLINE " : cout << res2 << endl ; int arr3 [ ] = { 1 , 11 , 100 , 1 , 0 , 200 , 3 , 2 , 1 , 250 } ; int n3 = sizeof ( arr3 ) / sizeof ( arr3 [ 0 ] ) ; x = 280 ; int res3 = smallestSubWithSum ( arr3 , n3 , x ) ; ( res3 == n3 + 1 ) ? cout << " Not ▁ possible STRNEWLINE " : cout << res3 << endl ; return 0 ; }
#define NA  -1
void moveToEnd ( int mPlusN [ ] , int size ) { int i = 0 , j = size - 1 ; for ( i = size - 1 ; i >= 0 ; i -- ) if ( mPlusN [ i ] != NA ) { mPlusN [ j ] = mPlusN [ i ] ; j -- ; } }
int merge ( int mPlusN [ ] , int N [ ] , int m , int n ) { int i = n ;
int j = 0 ;
int k = 0 ;
while ( k < ( m + n ) ) {
if ( ( j == n ) || ( i < ( m + n ) && mPlusN [ i ] <= N [ j ] ) ) { mPlusN [ k ] = mPlusN [ i ] ; k ++ ; i ++ ; }
else { mPlusN [ k ] = N [ j ] ; k ++ ; j ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int mPlusN [ ] = { 2 , 8 , NA , NA , NA , 13 , NA , 15 , 20 } ; int N [ ] = { 5 , 7 , 9 , 25 } ; int n = sizeof ( N ) / sizeof ( N [ 0 ] ) ; int m = sizeof ( mPlusN ) / sizeof ( mPlusN [ 0 ] ) - n ;
moveToEnd ( mPlusN , m + n ) ;
merge ( mPlusN , N , m , n ) ;
printArray ( mPlusN , m + n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE int getInvCount ( int arr [ ] , int n ) { int inv_count = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ i ] > arr [ j ] ) inv_count ++ ; return inv_count ; }
int main ( ) { int arr [ ] = { 1 , 20 , 6 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " ▁ Number ▁ of ▁ inversions ▁ are ▁ % d ▁ STRNEWLINE " , getInvCount ( arr , n ) ) ; return 0 ; }
# include <stdio.h> NEW_LINE # include <stdlib.h> NEW_LINE # include <math.h> NEW_LINE void minAbsSumPair ( int arr [ ] , int arr_size ) { int inv_count = 0 ; int l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { printf ( " Invalid ▁ Input " ) ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( abs ( min_sum ) > abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } printf ( " ▁ The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are ▁ % d ▁ and ▁ % d " , arr [ min_l ] , arr [ min_r ] ) ; }
int main ( ) { int arr [ ] = { 1 , 60 , -10 , 70 , -80 , 85 } ; minAbsSumPair ( arr , 6 ) ; getchar ( ) ; return 0 ; }
void printUnion ( int arr1 [ ] , int arr2 [ ] , int m , int n ) { int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( arr1 [ i ] < arr2 [ j ] ) printf ( " ▁ % d ▁ " , arr1 [ i ++ ] ) ; else if ( arr2 [ j ] < arr1 [ i ] ) printf ( " ▁ % d ▁ " , arr2 [ j ++ ] ) ; else { printf ( " ▁ % d ▁ " , arr2 [ j ++ ] ) ; i ++ ; } }
while ( i < m ) printf ( " ▁ % d ▁ " , arr1 [ i ++ ] ) ; while ( j < n ) printf ( " ▁ % d ▁ " , arr2 [ j ++ ] ) ; }
int main ( ) { int arr1 [ ] = { 1 , 2 , 4 , 5 , 6 } ; int arr2 [ ] = { 2 , 3 , 5 , 7 } ; int m = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; int n = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; printUnion ( arr1 , arr2 , m , n ) ; getchar ( ) ; return 0 ; }
void printIntersection ( int arr1 [ ] , int arr2 [ ] , int m , int n ) { int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( arr1 [ i ] < arr2 [ j ] ) i ++ ; else if ( arr2 [ j ] < arr1 [ i ] ) j ++ ; else { printf ( " ▁ % d ▁ " , arr2 [ j ++ ] ) ; i ++ ; } } }
int main ( ) { int arr1 [ ] = { 1 , 2 , 4 , 5 , 6 } ; int arr2 [ ] = { 2 , 3 , 5 , 7 } ; int m = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; int n = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ;
printIntersection ( arr1 , arr2 , m , n ) ; getchar ( ) ; return 0 ; }
void swap ( int * a , int * b ) { int temp = * a ; * a = * b ; * b = temp ; }
void sort012 ( int a [ ] , int arr_size ) { int lo = 0 ; int hi = arr_size - 1 ; int mid = 0 ; while ( mid <= hi ) { switch ( a [ mid ] ) { case 0 : swap ( & a [ lo ++ ] , & a [ mid ++ ] ) ; break ; case 1 : mid ++ ; break ; case 2 : swap ( & a [ mid ] , & a [ hi -- ] ) ; break ; } } }
void printArray ( int arr [ ] , int arr_size ) { int i ; for ( i = 0 ; i < arr_size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " n " ) ; }
int main ( ) { int arr [ ] = { 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int i ; sort012 ( arr , arr_size ) ; printf ( " array ▁ after ▁ segregation ▁ " ) ; printArray ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE void printUnsorted ( int arr [ ] , int n ) { int s = 0 , e = n - 1 , i , max , min ;
for ( s = 0 ; s < n - 1 ; s ++ ) { if ( arr [ s ] > arr [ s + 1 ] ) break ; } if ( s == n - 1 ) { printf ( " The ▁ complete ▁ array ▁ is ▁ sorted " ) ; return ; }
for ( e = n - 1 ; e > 0 ; e -- ) { if ( arr [ e ] < arr [ e - 1 ] ) break ; }
max = arr [ s ] ; min = arr [ s ] ; for ( i = s + 1 ; i <= e ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; if ( arr [ i ] < min ) min = arr [ i ] ; }
for ( i = 0 ; i < s ; i ++ ) { if ( arr [ i ] > min ) { s = i ; break ; } }
for ( i = n - 1 ; i >= e + 1 ; i -- ) { if ( arr [ i ] < max ) { e = i ; break ; } }
printf ( " ▁ The ▁ unsorted ▁ subarray ▁ which ▁ makes ▁ the ▁ given ▁ array ▁ " " ▁ sorted ▁ lies ▁ between ▁ the ▁ indees ▁ % d ▁ and ▁ % d " , s , e ) ; return ; } int main ( ) { int arr [ ] = { 10 , 12 , 20 , 30 , 25 , 40 , 32 , 31 , 35 , 50 , 60 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printUnsorted ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
int findNumberOfTriangles ( int arr [ ] , int n ) {
qsort ( arr , n , sizeof ( arr [ 0 ] ) , comp ) ;
int count = 0 ;
for ( int i = 0 ; i < n - 2 ; ++ i ) {
int k = i + 2 ;
for ( int j = i + 1 ; j < n ; ++ j ) {
while ( k < n && arr [ i ] + arr [ j ] > arr [ k ] ) ++ k ;
if ( k > j ) count += k - j - 1 ; } } return count ; }
int main ( ) { int arr [ ] = { 10 , 21 , 22 , 100 , 101 , 200 , 300 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Total ▁ number ▁ of ▁ triangles ▁ possible ▁ is ▁ % d ▁ " , findNumberOfTriangles ( arr , size ) ) ; return 0 ; }
int findElement ( int arr [ ] , int n , int key ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return -1 ; }
int main ( ) { int arr [ ] = { 12 , 34 , 10 , 6 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int key = 40 ; int position = findElement ( arr , n , key ) ; if ( position == - 1 ) printf ( " Element ▁ not ▁ found " ) ; else printf ( " Element ▁ Found ▁ at ▁ Position : ▁ % d " , position + 1 ) ; return 0 ; }
int insertSorted ( int arr [ ] , int n , int key , int capacity ) {
if ( n >= capacity ) return n ; arr [ n ] = key ; return ( n + 1 ) ; }
int main ( ) { int arr [ 20 ] = { 12 , 16 , 20 , 40 , 50 , 70 } ; int capacity = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 6 ; int i , key = 26 ; printf ( " Before Insertion : " for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ;
n = insertSorted ( arr , n , key , capacity ) ; printf ( " After Insertion : " for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; return 0 ; }
int findElement ( int arr [ ] , int n , int key ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return - 1 ; }
int deleteElement ( int arr [ ] , int n , int key ) {
int pos = findElement ( arr , n , key ) ; if ( pos == - 1 ) { printf ( " Element ▁ not ▁ found " ) ; return n ; }
int i ; for ( i = pos ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; return n - 1 ; }
int main ( ) { int i ; int arr [ ] = { 10 , 50 , 30 , 40 , 20 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int key = 30 ; printf ( " Array ▁ before ▁ deletion STRNEWLINE " ) ; for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; n = deleteElement ( arr , n , key ) ; printf ( " Array after deletion " for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; return 0 ; }
int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return -1 ; int mid = ( low + high ) / 2 ;
if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 7 , 8 , 9 , 10 } ; int n , key ; n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; key = 10 ; printf ( " Index : ▁ % d STRNEWLINE " , binarySearch ( arr , 0 , n - 1 , key ) ) ; return 0 ; }
int equilibrium ( int arr [ ] , int n ) { int i , j ; int leftsum , rightsum ;
for ( i = 0 ; i < n ; ++ i ) { leftsum = 0 ; rightsum = 0 ;
for ( j = 0 ; j < i ; j ++ ) leftsum += arr [ j ] ;
for ( j = i + 1 ; j < n ; j ++ ) rightsum += arr [ j ] ;
if ( leftsum == rightsum ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { -7 , 1 , 5 , 2 , -4 , 3 , 0 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d " , equilibrium ( arr , arr_size ) ) ; getchar ( ) ; return 0 ; }
int equilibrium ( int arr [ ] , int n ) {
int sum = 0 ;
int leftsum = 0 ;
for ( int i = 0 ; i < n ; ++ i ) sum += arr [ i ] ; for ( int i = 0 ; i < n ; ++ i ) {
sum -= arr [ i ] ; if ( leftsum == sum ) return i ; leftsum += arr [ i ] ; }
return -1 ; }
int main ( ) { int arr [ ] = { -7 , 1 , 5 , 2 , -4 , 3 , 0 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " First ▁ equilibrium ▁ index ▁ is ▁ % d " , equilibrium ( arr , arr_size ) ) ; getchar ( ) ; return 0 ; }
int ceilSearch ( int arr [ ] , int low , int high , int x ) { int i ;
if ( x <= arr [ low ] ) return low ;
for ( i = low ; i < high ; i ++ ) { if ( arr [ i ] == x ) return i ;
if ( arr [ i ] < x && arr [ i + 1 ] >= x ) return i + 1 ; }
return -1 ; }
int main ( ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 3 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == -1 ) printf ( " Ceiling ▁ of ▁ % d ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " , x ) ; else printf ( " ceiling ▁ of ▁ % d ▁ is ▁ % d " , x , arr [ index ] ) ; getchar ( ) ; return 0 ; }
int ceilSearch ( int arr [ ] , int low , int high , int x ) { int mid ;
if ( x <= arr [ low ] ) return low ;
if ( x > arr [ high ] ) return -1 ;
mid = ( low + high ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
else if ( arr [ mid ] < x ) { if ( mid + 1 <= high && x <= arr [ mid + 1 ] ) return mid + 1 ; else return ceilSearch ( arr , mid + 1 , high , x ) ; }
else { if ( mid - 1 >= low && x > arr [ mid - 1 ] ) return mid ; else return ceilSearch ( arr , low , mid - 1 , x ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 20 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == -1 ) printf ( " Ceiling ▁ of ▁ % d ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " , x ) ; else printf ( " ceiling ▁ of ▁ % d ▁ is ▁ % d " , x , arr [ index ] ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define NUM_LINE  2 NEW_LINE #define NUM_STATION  4
int min ( int a , int b ) { return a < b ? a : b ; } int carAssembly ( int a [ ] [ NUM_STATION ] , int t [ ] [ NUM_STATION ] , int * e , int * x ) { int T1 [ NUM_STATION ] , T2 [ NUM_STATION ] , i ;
T1 [ 0 ] = e [ 0 ] + a [ 0 ] [ 0 ] ;
T2 [ 0 ] = e [ 1 ] + a [ 1 ] [ 0 ] ;
for ( i = 1 ; i < NUM_STATION ; ++ i ) { T1 [ i ] = min ( T1 [ i - 1 ] + a [ 0 ] [ i ] , T2 [ i - 1 ] + t [ 1 ] [ i ] + a [ 0 ] [ i ] ) ; T2 [ i ] = min ( T2 [ i - 1 ] + a [ 1 ] [ i ] , T1 [ i - 1 ] + t [ 0 ] [ i ] + a [ 1 ] [ i ] ) ; }
return min ( T1 [ NUM_STATION - 1 ] + x [ 0 ] , T2 [ NUM_STATION - 1 ] + x [ 1 ] ) ; }
int main ( ) { int a [ ] [ NUM_STATION ] = { { 4 , 5 , 3 , 2 } , { 2 , 10 , 1 , 4 } } ; int t [ ] [ NUM_STATION ] = { { 0 , 7 , 4 , 5 } , { 0 , 9 , 2 , 8 } } ; int e [ ] = { 10 , 12 } , x [ ] = { 18 , 7 } ; printf ( " % d " , carAssembly ( a , t , e , x ) ) ; return 0 ; }
int minPalPartion ( char * str ) {
int n = strlen ( str ) ;
int C [ n ] [ n ] ; bool P [ n ] [ n ] ;
for ( i = 0 ; i < n ; i ++ ) { P [ i ] [ i ] = true ; C [ i ] [ i ] = 0 ; }
for ( L = 2 ; L <= n ; L ++ ) {
for ( i = 0 ; i < n - L + 1 ; i ++ ) {
j = i + L - 1 ;
if ( L == 2 ) P [ i ] [ j ] = ( str [ i ] == str [ j ] ) ; else P [ i ] [ j ] = ( str [ i ] == str [ j ] ) && P [ i + 1 ] [ j - 1 ] ;
if ( P [ i ] [ j ] == true ) C [ i ] [ j ] = 0 ; else {
C [ i ] [ j ] = INT_MAX ; for ( k = i ; k <= j - 1 ; k ++ ) C [ i ] [ j ] = min ( C [ i ] [ j ] , C [ i ] [ k ] + C [ k + 1 ] [ j ] + 1 ) ; } } }
return C [ 0 ] [ n - 1 ] ; }
int main ( ) { char str [ ] = " ababbbabbababa " ; printf ( " Min ▁ cuts ▁ needed ▁ for ▁ Palindrome ▁ Partitioning ▁ is ▁ % d " , minPalPartion ( str ) ) ; return 0 ; }
double sum ( int n ) { double i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
int main ( ) { int n = 5 ; printf ( " Sum ▁ is ▁ % f " , sum ( n ) ) ; return 0 ; }
int nthTermOfTheSeries ( int n ) {
if ( n % 2 == 0 ) nthTerm = pow ( n - 1 , 2 ) + n ;
else nthTerm = pow ( n + 1 , 2 ) + n ;
return nthTerm ; }
int main ( ) { int n ; n = 8 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 12 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 102 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 999 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 9999 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE unsigned int Log2n ( unsigned int n ) { return ( n > 1 ) ? 1 + Log2n ( n / 2 ) : 0 ; }
int main ( ) { unsigned int n = 32 ; printf ( " % u " , Log2n ( n ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE float findAmount ( float X , float W , float Y ) { return ( X * ( Y - W ) ) / ( 100 - Y ) ; }
int main ( ) { float X = 100 , W = 50 , Y = 60 ; printf ( " Water ▁ to ▁ be ▁ added ▁ = ▁ % .2f ▁ " , findAmount ( X , W , Y ) ) ; return 0 ; }
float AvgofSquareN ( int n ) { return ( float ) ( ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ; }
int main ( ) { int n = 10 ; printf ( " % f " , AvgofSquareN ( n ) ) ; return 0 ; }
void triangular_series ( int n ) { for ( int i = 1 ; i <= n ; i ++ ) printf ( " ▁ % d ▁ " , i * ( i + 1 ) / 2 ) ; }
int main ( ) { int n = 5 ; triangular_series ( n ) ; return 0 ; }
int divisorSum ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) sum += ( n / i ) * i ; return sum ; }
int main ( ) { int n = 4 ; printf ( " % d STRNEWLINE " , divisorSum ( n ) ) ; n = 5 ; printf ( " % d " , divisorSum ( n ) ) ; return 0 ; }
double sum ( int x , int n ) { double i , total = 1.0 ; for ( i = 1 ; i <= n ; i ++ ) total = total + ( pow ( x , i ) / i ) ; return total ; }
int main ( ) { int x = 2 ; int n = 5 ; printf ( " % .2f " , sum ( x , n ) ) ; return 0 ; }
bool check ( int n ) { if ( n <= 0 ) return false ;
return 1162261467 % n == 0 ; }
int main ( ) { int n = 9 ; if ( check ( n ) ) printf ( " Yes " ) ; else printf ( " No " ) ; return 0 ; }
#include <stdio.h> NEW_LINE int per ( int n ) { int a = 3 , b = 0 , c = 2 , i ; int m ; if ( n == 0 ) return a ; if ( n == 1 ) return b ; if ( n == 2 ) return c ; while ( n > 2 ) { m = a + b ; a = b ; b = c ; c = m ; n -- ; } return m ; }
int main ( ) { int n = 9 ; printf ( " % d " , per ( n ) ) ; return 0 ; }
void countDivisors ( int n ) {
int count = 0 ;
for ( int i = 1 ; i <= sqrt ( n ) + 1 ; i ++ ) { if ( n % i == 0 )
count += ( n / i == i ) ? 1 : 2 ; } if ( count % 2 == 0 ) printf ( " Even STRNEWLINE " ) ; else printf ( " Odd STRNEWLINE " ) ; }
int main ( ) { printf ( " The ▁ count ▁ of ▁ divisor : ▁ " ) ; countDivisors ( 10 ) ; return 0 ; }
int countSquares ( int m , int n ) { int temp ;
if ( n < m ) { temp = n ; n = m ; m = temp ; }
return m * ( m + 1 ) * ( 2 * m + 1 ) / 6 + ( n - m ) * m * ( m + 1 ) / 2 ; }
int main ( ) { int m = 4 , n = 3 ; printf ( " Count ▁ of ▁ squares ▁ is ▁ % d " , countSquares ( m , n ) ) ; }
double sum ( int n ) { double i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
int main ( ) { int n = 5 ; printf ( " Sum ▁ is ▁ % f " , sum ( n ) ) ; return 0 ; }
int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
int main ( ) { int a = 98 , b = 56 ; printf ( " GCD ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d ▁ " , a , b , gcd ( a , b ) ) ; return 0 ; }
void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; return ; }
void printSequencesRecur ( int arr [ ] , int n , int k , int index ) { int i ; if ( k == 0 ) { printArray ( arr , index ) ; } if ( k > 0 ) { for ( i = 1 ; i <= n ; ++ i ) { arr [ index ] = i ; printSequencesRecur ( arr , n , k - 1 , index + 1 ) ; } } }
void printSequences ( int n , int k ) { int * arr = new int [ k ] ; printSequencesRecur ( arr , n , k , 0 ) ; return ; }
int main ( ) { int n = 3 ; int k = 2 ; printSequences ( n , k ) ; return 0 ; }
bool isMultipleof5 ( int n ) { while ( n > 0 ) n = n - 5 ; if ( n == 0 ) return true ; return false ; }
int main ( ) { int n = 19 ; if ( isMultipleof5 ( n ) == true ) printf ( " % d ▁ is ▁ multiple ▁ of ▁ 5 STRNEWLINE " , n ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ multiple ▁ of ▁ 5 STRNEWLINE " , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE unsigned int countBits ( unsigned int n ) { unsigned int count = 0 ; while ( n ) { count ++ ; n >>= 1 ; } return count ; }
int main ( ) { int i = 65 ; printf ( " % d " , countBits ( i ) ) ; return 0 ; }
int isKthBitSet ( int x , int k ) { return ( x & ( 1 << ( k - 1 ) ) ) ? 1 : 0 ; }
int leftmostSetBit ( int x ) { int count = 0 ; while ( x ) { count ++ ; x = x >> 1 ; } return count ; }
int isBinPalindrome ( int x ) { int l = leftmostSetBit ( x ) ; int r = 1 ;
while ( l > r ) {
if ( isKthBitSet ( x , l ) != isKthBitSet ( x , r ) ) return 0 ; l -- ; r ++ ; } return 1 ; } int findNthPalindrome ( int n ) { int pal_count = 0 ;
int i = 0 ; for ( i = 1 ; i <= INT_MAX ; i ++ ) { if ( isBinPalindrome ( i ) ) { pal_count ++ ; }
if ( pal_count == n ) break ; } return i ; }
int main ( ) { int n = 9 ;
printf ( " % d " , findNthPalindrome ( n ) ) ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int exponentMod ( int A , int B , int C ) {
if ( A == 0 ) return 0 ; if ( B == 0 ) return 1 ;
long y ; if ( B % 2 == 0 ) { y = exponentMod ( A , B / 2 , C ) ; y = ( y * y ) % C ; }
else { y = A % C ; y = ( y * exponentMod ( A , B - 1 , C ) % C ) % C ; } return ( int ) ( ( y + C ) % C ) ; }
int main ( ) { int A = 2 , B = 5 , C = 13 ; printf ( " Power ▁ is ▁ % d " , exponentMod ( A , B , C ) ) ; return 0 ; }
void printknapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } }
int res = K [ n ] [ W ] ; printf ( " % d STRNEWLINE " , res ) ; w = W ; for ( i = n ; i > 0 && res > 0 ; i -- ) {
if ( res == K [ i - 1 ] [ w ] ) continue ; else {
printf ( " % d ▁ " , wt [ i - 1 ] ) ;
res = res - val [ i - 1 ] ; w = w - wt [ i - 1 ] ; } } }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printknapSack ( W , wt , val , n ) ; return 0 ; }
int eggDrop ( int n , int k ) {
int eggFloor [ n + 1 ] [ k + 1 ] ; int res ; int i , j , x ;
for ( i = 1 ; i <= n ; i ++ ) { eggFloor [ i ] [ 1 ] = 1 ; eggFloor [ i ] [ 0 ] = 0 ; }
for ( j = 1 ; j <= k ; j ++ ) eggFloor [ 1 ] [ j ] = j ;
for ( i = 2 ; i <= n ; i ++ ) { for ( j = 2 ; j <= k ; j ++ ) { eggFloor [ i ] [ j ] = INT_MAX ; for ( x = 1 ; x <= j ; x ++ ) { res = 1 + max ( eggFloor [ i - 1 ] [ x - 1 ] , eggFloor [ i ] [ j - x ] ) ; if ( res < eggFloor [ i ] [ j ] ) eggFloor [ i ] [ j ] = res ; } } }
return eggFloor [ n ] [ k ] ; }
int main ( ) { int n = 2 , k = 36 ; printf ( " Minimum number of trials " STRNEWLINE " in worst case with % d eggs and " STRNEWLINE " % d floors is % d " , n , k , eggDrop ( n , k ) ) ; return 0 ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
#define d  256
void search ( char pat [ ] , char txt [ ] , int q ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i , j ;
int h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat [ i ] ) % q ; t = ( d * t + txt [ i ] ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) {
for ( j = 0 ; j < M ; j ++ ) { if ( txt [ i + j ] != pat [ j ] ) break ; }
if ( j == M ) printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; }
if ( i < N - M ) { t = ( d * ( t - txt [ i ] * h ) + txt [ i + M ] ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
int main ( ) { char txt [ ] = " GEEKS ▁ FOR ▁ GEEKS " ; char pat [ ] = " GEEK " ;
int q = 101 ;
search ( pat , txt , q ) ; return 0 ; }
int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
int main ( ) { int a = 98 , b = 56 ; printf ( " GCD ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d ▁ " , a , b , gcd ( a , b ) ) ; return 0 ; }
int checkSemiprime ( int num ) { int cnt = 0 ; for ( int i = 2 ; cnt < 2 && i * i <= num ; ++ i ) while ( num % i == 0 ) num /= i , ++ cnt ;
if ( num > 1 ) ++ cnt ;
return cnt == 2 ; }
void semiprime ( int n ) { if ( checkSemiprime ( n ) ) printf ( " True STRNEWLINE " ) ; else printf ( " False STRNEWLINE " ) ; }
int main ( ) { int n = 6 ; semiprime ( n ) ; n = 8 ; semiprime ( n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE void indexedSequentialSearch ( int arr [ ] , int n , int k ) { int elements [ 20 ] , indices [ 20 ] , temp , i , set = 0 ; int j = 0 , ind = 0 , start , end ; for ( i = 0 ; i < n ; i += 3 ) {
elements [ ind ] = arr [ i ] ;
indices [ ind ] = i ; ind ++ ; } if ( k < elements [ 0 ] ) { printf ( " Not ▁ found " ) ; exit ( 0 ) ; } else { for ( i = 1 ; i <= ind ; i ++ ) if ( k <= elements [ i ] ) { start = indices [ i - 1 ] ; end = indices [ i ] ; set = 1 ; break ; } } if ( set == 0 ) { start = indices [ i - 1 ] ; end = n ; } for ( i = start ; i <= end ; i ++ ) { if ( k == arr [ i ] ) { j = 1 ; break ; } } if ( j == 1 ) printf ( " Found ▁ at ▁ index ▁ % d " , i ) ; else printf ( " Not ▁ found " ) ; }
void main ( ) { int arr [ ] = { 6 , 7 , 8 , 9 , 10 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int k = 8 ;
indexedSequentialSearch ( arr , n , k ) ; }
void printNSE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] > arr [ j ] ) { next = arr [ j ] ; break ; } } printf ( " % d ▁ - - ▁ % d STRNEWLINE " , arr [ i ] , next ) ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNSE ( arr , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE #define NO_OF_CHARS  256
int * getCharCountArray ( char * str ) { int * count = ( int * ) calloc ( sizeof ( int ) , NO_OF_CHARS ) ; int i ; for ( i = 0 ; * ( str + i ) ; i ++ ) count [ * ( str + i ) ] ++ ; return count ; }
int firstNonRepeating ( char * str ) { int * count = getCharCountArray ( str ) ; int index = -1 , i ; for ( i = 0 ; * ( str + i ) ; i ++ ) { if ( count [ * ( str + i ) ] == 1 ) { index = i ; break ; } }
int main ( ) { char str [ ] = " geeksforgeeks " ; int index = firstNonRepeating ( str ) ; if ( index == -1 ) printf ( " Either ▁ all ▁ characters ▁ are ▁ repeating ▁ or ▁ " " string ▁ is ▁ empty " ) ; else printf ( " First ▁ non - repeating ▁ character ▁ is ▁ % c " , str [ index ] ) ; getchar ( ) ; return 0 ; }
void divideString ( char * str , int n ) { int str_size = strlen ( str ) ; int i ; int part_size ;
if ( str_size % n != 0 ) { printf ( " Invalid ▁ Input : ▁ String ▁ size " ) ; printf ( " ▁ is ▁ not ▁ divisible ▁ by ▁ n " ) ; return ; }
part_size = str_size / n ; for ( i = 0 ; i < str_size ; i ++ ) { if ( i % part_size == 0 ) printf ( " STRNEWLINE " ) ; printf ( " % c " , str [ i ] ) ; } } int main ( ) {
char * str = " a _ simple _ divide _ string _ quest " ;
divideString ( str , 4 ) ; getchar ( ) ; return 0 ; }
void collinear ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 ) { if ( ( y3 - y2 ) * ( x2 - x1 ) == ( y2 - y1 ) * ( x3 - x2 ) ) printf ( " Yes " ) ; else printf ( " No " ) ; }
int main ( ) { int x1 = 1 , x2 = 1 , x3 = 0 , y1 = 1 , y2 = 6 , y3 = 9 ; collinear ( x1 , y1 , x2 , y2 , x3 , y3 ) ; return 0 ; }
void bestApproximate ( int x [ ] , int y [ ] , int n ) { int i , j ; float m , c , sum_x = 0 , sum_y = 0 , sum_xy = 0 , sum_x2 = 0 ; for ( i = 0 ; i < n ; i ++ ) { sum_x += x [ i ] ; sum_y += y [ i ] ; sum_xy += x [ i ] * y [ i ] ; sum_x2 += ( x [ i ] * x [ i ] ) ; } m = ( n * sum_xy - sum_x * sum_y ) / ( n * sum_x2 - ( sum_x * sum_x ) ) ; c = ( sum_y - m * sum_x ) / n ; printf ( " m ▁ = % ▁ f " , m ) ; printf ( " c = % f " , c ) ; }
int main ( ) { int x [ ] = { 1 , 2 , 3 , 4 , 5 } ; int y [ ] = { 14 , 27 , 40 , 55 , 68 } ; int n = sizeof ( x ) / sizeof ( x [ 0 ] ) ; bestApproximate ( x , y , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE void printSorted ( int arr [ ] , int start , int end ) { if ( start > end ) return ;
printSorted ( arr , start * 2 + 1 , end ) ;
printf ( " % d ▁ " , arr [ start ] ) ;
printSorted ( arr , start * 2 + 2 , end ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 5 , 1 , 3 } ; int arr_size = sizeof ( arr ) / sizeof ( int ) ; printSorted ( arr , 0 , arr_size - 1 ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int Identity ( int num ) { int row , col ; for ( row = 0 ; row < num ; row ++ ) { for ( col = 0 ; col < num ; col ++ ) {
if ( row == col ) printf ( " % d ▁ " , 1 ) ; else printf ( " % d ▁ " , 0 ) ; } printf ( " STRNEWLINE " ) ; } return 0 ; }
int main ( ) { int size = 5 ; identity ( size ) ; return 0 ; }
int search ( int mat [ 4 ] [ 4 ] , int n , int x ) { if ( n == 0 ) return -1 ; int smallest = mat [ 0 ] [ 0 ] , largest = mat [ n - 1 ] [ n - 1 ] ; if ( x < smallest x > largest ) return -1 ;
int i = 0 , j = n - 1 ; while ( i < n && j >= 0 ) { if ( mat [ i ] [ j ] == x ) { printf ( " Found at % d , % d " , i , j ) ; return 1 ; } if ( mat [ i ] [ j ] > x ) j -- ;
else i ++ ; } printf ( " n ▁ Element ▁ not ▁ found " ) ;
return 0 ; }
int main ( ) { int mat [ 4 ] [ 4 ] = { { 10 , 20 , 30 , 40 } , { 15 , 25 , 35 , 45 } , { 27 , 29 , 37 , 48 } , { 32 , 33 , 39 , 50 } , } ; search ( mat , 4 , 29 ) ; return 0 ; }
void fill0X ( int m , int n ) {
int i , k = 0 , l = 0 ;
int r = m , c = n ;
char x = ' X ' ;
while ( k < m && l < n ) {
for ( i = l ; i < n ; ++ i ) a [ k ] [ i ] = x ; k ++ ;
for ( i = k ; i < m ; ++ i ) a [ i ] [ n - 1 ] = x ; n -- ;
if ( k < m ) { for ( i = n - 1 ; i >= l ; -- i ) a [ m - 1 ] [ i ] = x ; m -- ; }
if ( l < n ) { for ( i = m - 1 ; i >= k ; -- i ) a [ i ] [ l ] = x ; l ++ ; }
x = ( x == '0' ) ? ' X ' : '0' ; }
for ( i = 0 ; i < r ; i ++ ) { for ( int j = 0 ; j < c ; j ++ ) printf ( " % c ▁ " , a [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } }
int main ( ) { puts ( " Output ▁ for ▁ m ▁ = ▁ 5 , ▁ n ▁ = ▁ 6" ) ; fill0X ( 5 , 6 ) ; puts ( " Output for m = 4 , n = 4 " ) ; fill0X ( 4 , 4 ) ; puts ( " Output for m = 3 , n = 4 " ) ; fill0X ( 3 , 4 ) ; return 0 ; }
int findPeakUtil ( int arr [ ] , int low , int high , int n ) {
int mid = low + ( high - low ) / 2 ;
if ( ( mid == 0 arr [ mid - 1 ] <= arr [ mid ] ) && ( mid == n - 1 arr [ mid + 1 ] <= arr [ mid ] ) ) return mid ;
else if ( mid > 0 && arr [ mid - 1 ] > arr [ mid ] ) return findPeakUtil ( arr , low , ( mid - 1 ) , n ) ;
else return findPeakUtil ( arr , ( mid + 1 ) , high , n ) ; }
int findPeak ( int arr [ ] , int n ) { return findPeakUtil ( arr , 0 , n - 1 , n ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 20 , 4 , 1 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Index ▁ of ▁ a ▁ peak ▁ point ▁ is ▁ % d " , findPeak ( arr , n ) ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) { int i , j ; printf ( " ▁ Repeating ▁ elements ▁ are ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) for ( j = i + 1 ; j < size ; j ++ ) if ( arr [ i ] == arr [ j ] ) printf ( " ▁ % d ▁ " , arr [ i ] ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) { int * count = ( int * ) calloc ( sizeof ( int ) , ( size - 2 ) ) ; int i ; printf ( " ▁ Repeating ▁ elements ▁ are ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) { if ( count [ arr [ i ] ] == 1 ) printf ( " ▁ % d ▁ " , arr [ i ] ) ; else count [ arr [ i ] ] ++ ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
int fact ( int n ) ; void printRepeating ( int arr [ ] , int size ) {
int S = 0 ;
int P = 1 ;
int x , y ;
int D ; int n = size - 2 , i ;
for ( i = 0 ; i < size ; i ++ ) { S = S + arr [ i ] ; P = P * arr [ i ] ; }
S = S - n * ( n + 1 ) / 2 ;
P = P / fact ( n ) ;
D = sqrt ( S * S - 4 * P ) ; x = ( D + S ) / 2 ; y = ( S - D ) / 2 ; printf ( " The ▁ two ▁ Repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d " , x , y ) ; }
int fact ( int n ) { return ( n == 0 ) ? 1 : n * fact ( n - 1 ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) {
int xor = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor ^= arr [ i ] ; for ( i = 1 ; i <= n ; i ++ ) xor ^= i ;
set_bit_no = xor & ~ ( xor - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ;
} for ( i = 1 ; i <= n ; i ++ ) { if ( i & set_bit_no ) x = x ^ i ;
else y = y ^ i ;
} printf ( " n ▁ The ▁ two ▁ repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d ▁ " , x , y ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) { int i ; printf ( " The repeating elements are " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) ] > 0 ) arr [ abs ( arr [ i ] ) ] = - arr [ abs ( arr [ i ] ) ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
int subArraySum ( int arr [ ] , int n , int sum ) { int curr_sum , i , j ;
for ( i = 0 ; i < n ; i ++ ) { curr_sum = arr [ i ] ;
for ( j = i + 1 ; j <= n ; j ++ ) { if ( curr_sum == sum ) { printf ( " Sum ▁ found ▁ between ▁ indexes ▁ % d ▁ and ▁ % d " , i , j - 1 ) ; return 1 ; } if ( curr_sum > sum j == n ) break ; curr_sum = curr_sum + arr [ j ] ; } } printf ( " No ▁ subarray ▁ found " ) ; return 0 ; }
int main ( ) { int arr [ ] = { 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int sum = 23 ; subArraySum ( arr , n , sum ) ; return 0 ; }
int subArraySum ( int arr [ ] , int n , int sum ) {
int curr_sum = arr [ 0 ] , start = 0 , i ;
for ( i = 1 ; i <= n ; i ++ ) {
while ( curr_sum > sum && start < i - 1 ) { curr_sum = curr_sum - arr [ start ] ; start ++ ; }
if ( curr_sum == sum ) { printf ( " Sum ▁ found ▁ between ▁ indexes ▁ % d ▁ and ▁ % d " , start , i - 1 ) ; return 1 ; }
if ( i < n ) curr_sum = curr_sum + arr [ i ] ; }
printf ( " No ▁ subarray ▁ found " ) ; return 0 ; }
int main ( ) { int arr [ ] = { 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int sum = 23 ; subArraySum ( arr , n , sum ) ; return 0 ; }
bool find3Numbers ( int A [ ] , int arr_size , int sum ) { int l , r ;
for ( int i = 0 ; i < arr_size - 2 ; i ++ ) {
for ( int j = i + 1 ; j < arr_size - 1 ; j ++ ) {
for ( int k = j + 1 ; k < arr_size ; k ++ ) { if ( A [ i ] + A [ j ] + A [ k ] == sum ) { printf ( " Triplet ▁ is ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] ) ; return true ; } } } }
return false ; }
int main ( ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int sum = 22 ; int arr_size = sizeof ( A ) / sizeof ( A [ 0 ] ) ; find3Numbers ( A , arr_size , sum ) ; return 0 ; }
int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == x ) return i ; } return -1 ; }
int main ( ) { int arr [ ] = { 1 , 10 , 30 , 15 } ; int x = 30 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d ▁ is ▁ present ▁ at ▁ index ▁ % d " , x , search ( arr , n , x ) ) ; getchar ( ) ; return 0 ; }
int binarySearch ( int arr [ ] , int l , int r , int x ) { if ( r >= l ) { int mid = l + ( r - l ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 10 ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
int binarySearch ( int arr [ ] , int l , int r , int x ) { while ( l <= r ) { int m = l + ( r - l ) / 2 ;
if ( arr [ m ] == x ) return m ;
if ( arr [ m ] < x ) l = m + 1 ;
else r = m - 1 ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 10 ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present " " ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ " " index ▁ % d " , result ) ; return 0 ; }
void swap ( int * a , int * b ) { int t = * a ; * a = * b ; * b = t ; }
int partition ( int arr [ ] , int l , int h ) { int x = arr [ h ] ; int i = ( l - 1 ) ; for ( int j = l ; j <= h - 1 ; j ++ ) { if ( arr [ j ] <= x ) { i ++ ; swap ( & arr [ i ] , & arr [ j ] ) ; } } swap ( & arr [ i + 1 ] , & arr [ h ] ) ; return ( i + 1 ) ; }
void quickSortIterative ( int arr [ ] , int l , int h ) {
int stack [ h - l + 1 ] ;
int top = -1 ;
stack [ ++ top ] = l ; stack [ ++ top ] = h ;
while ( top >= 0 ) {
h = stack [ top -- ] ; l = stack [ top -- ] ;
int p = partition ( arr , l , h ) ;
if ( p - 1 > l ) { stack [ ++ top ] = l ; stack [ ++ top ] = p - 1 ; }
if ( p + 1 < h ) { stack [ ++ top ] = p + 1 ; stack [ ++ top ] = h ; } } }
void printArr ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; ++ i ) printf ( " % d ▁ " , arr [ i ] ) ; }
int main ( ) { int arr [ ] = { 4 , 3 , 5 , 2 , 1 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( * arr ) ;
quickSortIterative ( arr , 0 , n - 1 ) ; printArr ( arr , n ) ; return 0 ; }
void printMaxActivities ( int s [ ] , int f [ ] , int n ) { int i , j ; printf ( " Following ▁ activities ▁ are ▁ selected ▁ n " ) ;
i = 0 ; printf ( " % d ▁ " , i ) ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { printf ( " % d ▁ " , j ) ; i = j ; } } }
int main ( ) { int s [ ] = { 1 , 3 , 0 , 5 , 8 , 5 } ; int f [ ] = { 2 , 4 , 6 , 7 , 9 , 9 } ; int n = sizeof ( s ) / sizeof ( s [ 0 ] ) ; printMaxActivities ( s , f , n ) ; return 0 ; }
int lcs ( char * X , char * Y , int m , int n ) { if ( m == 0 n == 0 ) return 0 ; if ( X [ m - 1 ] == Y [ n - 1 ] ) return 1 + lcs ( X , Y , m - 1 , n - 1 ) ; else return max ( lcs ( X , Y , m , n - 1 ) , lcs ( X , Y , m - 1 , n ) ) ; }
int main ( ) { char X [ ] = " AGGTAB " ; char Y [ ] = " GXTXAYB " ; int m = strlen ( X ) ; int n = strlen ( Y ) ; printf ( " Length ▁ of ▁ LCS ▁ is ▁ % d " , lcs ( X , Y , m , n ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE int max ( int a , int b ) ;
int lcs ( char * X , char * Y , int m , int n ) { int L [ m + 1 ] [ n + 1 ] ; int i , j ;
for ( i = 0 ; i <= m ; i ++ ) { for ( j = 0 ; j <= n ; j ++ ) { if ( i == 0 j == 0 ) L [ i ] [ j ] = 0 ; else if ( X [ i - 1 ] == Y [ j - 1 ] ) L [ i ] [ j ] = L [ i - 1 ] [ j - 1 ] + 1 ; else L [ i ] [ j ] = max ( L [ i - 1 ] [ j ] , L [ i ] [ j - 1 ] ) ; } }
return L [ m ] [ n ] ; }
int main ( ) { char X [ ] = " AGGTAB " ; char Y [ ] = " GXTXAYB " ; int m = strlen ( X ) ; int n = strlen ( Y ) ; printf ( " Length ▁ of ▁ LCS ▁ is ▁ % d " , lcs ( X , Y , m , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ;
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int minCost ( int cost [ R ] [ C ] , int m , int n ) { if ( n < 0 m < 0 ) return INT_MAX ; else if ( m == 0 && n == 0 ) return cost [ m ] [ n ] ; else return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ; int minCost ( int cost [ R ] [ C ] , int m , int n ) { int i , j ;
int tc [ R ] [ C ] ; tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] ;
for ( i = 1 ; i <= m ; i ++ ) tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] ;
for ( j = 1 ; j <= n ; j ++ ) tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] ;
for ( i = 1 ; i <= m ; i ++ ) for ( j = 1 ; j <= n ; j ++ ) tc [ i ] [ j ] = min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] ; return tc [ m ] [ n ] ; }
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
int count ( int n ) {
int table [ n + 1 ] , i ;
table [ 0 ] = 1 ;
for ( i = 3 ; i <= n ; i ++ ) table [ i ] += table [ i - 3 ] ; for ( i = 5 ; i <= n ; i ++ ) table [ i ] += table [ i - 5 ] ; for ( i = 10 ; i <= n ; i ++ ) table [ i ] += table [ i - 10 ] ; return table [ n ] ; }
int main ( void ) { int n = 20 ; printf ( " Count ▁ for ▁ % d ▁ is ▁ % d STRNEWLINE " , n , count ( n ) ) ; n = 13 ; printf ( " Count ▁ for ▁ % d ▁ is ▁ % d " , n , count ( n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h> NEW_LINE void search ( char * pat , char * txt ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ;
for ( int i = 0 ; i <= N - M ; i ++ ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; } }
int main ( ) { char txt [ ] = " AABAACAADAABAAABAA " ; char pat [ ] = " AABA " ; search ( pat , txt ) ; return 0 ; }
#define d  256
void search ( char pat [ ] , char txt [ ] , int q ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i , j ;
int p = 0 ;
int t = 0 ; int h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat [ i ] ) % q ; t = ( d * t + txt [ i ] ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) {
for ( j = 0 ; j < M ; j ++ ) { if ( txt [ i + j ] != pat [ j ] ) break ; }
if ( j == M ) printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; }
if ( i < N - M ) { t = ( d * ( t - txt [ i ] * h ) + txt [ i + M ] ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
int main ( ) { char txt [ ] = " GEEKS ▁ FOR ▁ GEEKS " ; char pat [ ] = " GEEK " ;
int q = 101 ;
search ( pat , txt , q ) ; return 0 ; }
void search ( char pat [ ] , char txt [ ] ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) { printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { char txt [ ] = " ABCEABCDABCEABCD " ; char pat [ ] = " ABCD " ; search ( pat , txt ) ; return 0 ; }
int getMedian ( int ar1 [ ] , int ar2 [ ] , int n ) { int i = 0 ; int j = 0 ; int count ; int m1 = -1 , m2 = -1 ;
for ( count = 0 ; count <= n ; count ++ ) {
if ( i == n ) { m1 = m2 ; m2 = ar2 [ 0 ] ; break ; }
else if ( j == n ) { m1 = m2 ; m2 = ar1 [ 0 ] ; break ; }
if ( ar1 [ i ] <= ar2 [ j ] ) { m1 = m2 ;
m2 = ar1 [ i ] ; i ++ ; } else { m1 = m2 ;
m2 = ar2 [ j ] ; j ++ ; } } return ( m1 + m2 ) / 2 ; }
int main ( ) { int ar1 [ ] = { 1 , 12 , 15 , 26 , 38 } ; int ar2 [ ] = { 2 , 13 , 17 , 30 , 45 } ; int n1 = sizeof ( ar1 ) / sizeof ( ar1 [ 0 ] ) ; int n2 = sizeof ( ar2 ) / sizeof ( ar2 [ 0 ] ) ; if ( n1 == n2 ) printf ( " Median ▁ is ▁ % d " , getMedian ( ar1 , ar2 , n1 ) ) ; else printf ( " Doesn ' t ▁ work ▁ for ▁ arrays ▁ of ▁ unequal ▁ size " ) ; getchar ( ) ; return 0 ; }
bool isLucky ( int n ) { static int counter = 2 ;
int next_position = n ; if ( counter > n ) return 1 ; if ( n % counter == 0 ) return 0 ;
next_position -= next_position / counter ; counter ++ ; return isLucky ( next_position ) ; }
int main ( ) { int x = 5 ; if ( isLucky ( x ) ) printf ( " % d ▁ is ▁ a ▁ lucky ▁ no . " , x ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ lucky ▁ no . " , x ) ; getchar ( ) ; }
int pow ( int a , int b ) { if ( b == 0 ) return 1 ; int answer = a ; int increment = a ; int i , j ; for ( i = 1 ; i < b ; i ++ ) { for ( j = 1 ; j < a ; j ++ ) { answer += increment ; } increment = answer ; } return answer ; }
int main ( ) { printf ( " % d " , pow ( 5 , 3 ) ) ; getchar ( ) ; return 0 ; }
int multiply ( int x , int y ) { if ( y ) return ( x + multiply ( x , y - 1 ) ) ; else return 0 ; }
int pow ( int a , int b ) { if ( b ) return multiply ( a , pow ( a , b - 1 ) ) ; else return 1 ; }
int main ( ) { printf ( " % d " , pow ( 5 , 3 ) ) ; getchar ( ) ; return 0 ; }
int count ( int n ) {
if ( n < 3 ) return n ; if ( n >= 3 && n < 10 ) return n - 1 ;
int po = 1 ; while ( n / po > 9 ) po = po * 10 ;
int msd = n / po ; if ( msd != 3 )
return count ( msd ) * count ( po - 1 ) + count ( msd ) + count ( n % po ) ; else
return count ( msd * po - 1 ) ; }
int main ( ) { printf ( " % d ▁ " , count ( 578 ) ) ; return 0 ; }
int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
int findSmallerInRight ( char * str , int low , int high ) { int countRight = 0 , i ; for ( i = low + 1 ; i <= high ; ++ i ) if ( str [ i ] < str [ low ] ) ++ countRight ; return countRight ; }
int findRank ( char * str ) { int len = strlen ( str ) ; int mul = fact ( len ) ; int rank = 1 ; int countRight ; int i ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
countRight = findSmallerInRight ( str , i , len - 1 ) ; rank += countRight * mul ; } return rank ; }
int main ( ) { char str [ ] = " string " ; printf ( " % d " , findRank ( str ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h> NEW_LINE #define MAX_CHAR  256
int count [ MAX_CHAR ] = { 0 } ;
int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
void populateAndIncreaseCount ( int * count , char * str ) { int i ; for ( i = 0 ; str [ i ] ; ++ i ) ++ count [ str [ i ] ] ; for ( i = 1 ; i < MAX_CHAR ; ++ i ) count [ i ] += count [ i - 1 ] ; }
void updatecount ( int * count , char ch ) { int i ; for ( i = ch ; i < MAX_CHAR ; ++ i ) -- count [ i ] ; }
int findRank ( char * str ) { int len = strlen ( str ) ; int mul = fact ( len ) ; int rank = 1 , i ;
populateAndIncreaseCount ( count , str ) ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
rank += count [ str [ i ] - 1 ] * mul ;
updatecount ( count , str [ i ] ) ; } return rank ; }
int main ( ) { char str [ ] = " string " ; printf ( " % d " , findRank ( str ) ) ; return 0 ; }
float exponential ( int n , float x ) {
float sum = 1.0f ; for ( int i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
int main ( ) { int n = 10 ; float x = 1.0f ; printf ( " e ^ x ▁ = ▁ % f " , exponential ( n , x ) ) ; return 0 ; }
int min ( int x , int y ) { return ( x < y ) ? x : y ; }
int calcAngle ( double h , double m ) {
if ( h < 0 m < 0 h > 12 m > 60 ) printf ( " Wrong ▁ input " ) ; if ( h == 12 ) h = 0 ; if ( m == 60 ) { m = 0 ; h += 1 ; if ( h > 12 ) h = h - 12 ; }
int hour_angle = 0.5 * ( h * 60 + m ) ; int minute_angle = 6 * m ;
int angle = abs ( hour_angle - minute_angle ) ;
angle = min ( 360 - angle , angle ) ; return angle ; }
int main ( ) { printf ( " % d ▁ n " , calcAngle ( 9 , 60 ) ) ; printf ( " % d ▁ n " , calcAngle ( 3 , 30 ) ) ; return 0 ; }
int getSingle ( int arr [ ] , int n ) { int ones = 0 , twos = 0 ; int common_bit_mask ; for ( int i = 0 ; i < n ; i ++ ) {
twos = twos | ( ones & arr [ i ] ) ;
ones = ones ^ arr [ i ] ;
common_bit_mask = ~ ( ones & twos ) ;
ones &= common_bit_mask ;
twos &= common_bit_mask ; } return ones ; }
int main ( ) { int arr [ ] = { 3 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define INT_SIZE  32 NEW_LINE int getSingle ( int arr [ ] , int n ) {
int result = 0 ; int x , sum ;
for ( int i = 0 ; i < INT_SIZE ; i ++ ) {
sum = 0 ; x = ( 1 << i ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] & x ) sum ++ ; }
if ( ( sum % 3 ) != 0 ) result |= x ; } return result ; }
int main ( ) { int arr [ ] = { 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int swapBits ( unsigned int x , unsigned int p1 , unsigned int p2 , unsigned int n ) {
unsigned int set1 = ( x >> p1 ) & ( ( 1U << n ) - 1 ) ;
unsigned int set2 = ( x >> p2 ) & ( ( 1U << n ) - 1 ) ;
unsigned int xor = ( set1 ^ set2 ) ;
xor = ( xor << p1 ) | ( xor << p2 ) ;
unsigned int result = x ^ xor ; return result ; }
int main ( ) { int res = swapBits ( 28 , 0 , 3 , 2 ) ; printf ( " Result = % d " , res ) ; return 0 ; }
#include <stdio.h> NEW_LINE int smallest ( int x , int y , int z ) { int c = 0 ; while ( x && y && z ) { x -- ; y -- ; z -- ; c ++ ; } return c ; }
int main ( ) { int x = 12 , y = 15 , z = 5 ; printf ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ % d " , smallest ( x , y , z ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int addOne ( int x ) { int m = 1 ;
while ( x & m ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
int main ( ) { printf ( " % d " , addOne ( 13 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int addOne ( int x ) { return ( - ( ~ x ) ) ; }
int main ( ) { printf ( " % d " , addOne ( 13 ) ) ; getchar ( ) ; return 0 ; }
int fun ( unsigned int n ) { return n & ( n - 1 ) ; }
int main ( ) { int n = 7 ; printf ( " The ▁ number ▁ after ▁ unsetting ▁ the " ) ; printf ( " ▁ rightmost ▁ set ▁ bit ▁ % d " , fun ( n ) ) ; return 0 ; }
bool isPowerOfFour ( int n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 4 != 0 ) return 0 ; n = n / 4 ; } return 1 ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) printf ( " % d ▁ is ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; getchar ( ) ; }
bool isPowerOfFour ( unsigned int n ) { int count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) {
while ( n > 1 ) { n >>= 1 ; count += 1 ; }
return ( count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) printf ( " % d ▁ is ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; getchar ( ) ; }
int min ( int x , int y ) { return y ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int max ( int x , int y ) { return x ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int main ( ) { int x = 15 ; int y = 6 ; printf ( " Minimum ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ " , x , y ) ; printf ( " % d " , min ( x , y ) ) ; printf ( " Maximum of % d and % d is " printf ( " % d " , max ( x , y ) ) ; getchar ( ) ; }
#include <math.h> NEW_LINE #include <stdio.h> NEW_LINE unsigned int getFirstSetBitPos ( int n ) { return log2 ( n & - n ) + 1 ; }
int main ( ) { int n = 12 ; printf ( " % u " , getFirstSetBitPos ( n ) ) ; getchar ( ) ; return 0 ; }
unsigned int swapBits ( unsigned int x ) {
unsigned int even_bits = x & 0xAAAAAAAA ;
unsigned int odd_bits = x & 0x55555555 ;
even_bits >>= 1 ;
odd_bits <<= 1 ;
return ( even_bits odd_bits ) ; }
unsigned int x = 23 ;
printf ( " % u ▁ " , swapBits ( x ) ) ; return 0 ; }
int isPowerOfTwo ( unsigned n ) { return n && ( ! ( n & ( n - 1 ) ) ) ; }
int findPosition ( unsigned n ) { if ( ! isPowerOfTwo ( n ) ) return -1 ; unsigned i = 1 , pos = 1 ;
while ( ! ( i & n ) ) {
i = i << 1 ;
++ pos ; } return pos ; }
int main ( void ) { int n = 16 ; int pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 12 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 128 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; return 0 ; }
int isPowerOfTwo ( unsigned n ) { return n && ( ! ( n & ( n - 1 ) ) ) ; }
int findPosition ( unsigned n ) { if ( ! isPowerOfTwo ( n ) ) return -1 ; unsigned count = 0 ;
while ( n ) { n = n >> 1 ;
++ count ; } return count ; }
int main ( void ) { int n = 0 ; int pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 12 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 128 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; return 0 ; }
#include <stdio.h> NEW_LINE int main ( ) { int x = 10 , y = 5 ;
x = x * y ;
y = x / y ;
x = x / y ; printf ( " After ▁ Swapping : ▁ x ▁ = ▁ % d , ▁ y ▁ = ▁ % d " , x , y ) ; return 0 ; }
#include <stdio.h> NEW_LINE int main ( ) { int x = 10 , y = 5 ;
x = x ^ y ;
y = x ^ y ;
x = x ^ y ; printf ( " After ▁ Swapping : ▁ x ▁ = ▁ % d , ▁ y ▁ = ▁ % d " , x , y ) ; return 0 ; }
void swap ( int * xp , int * yp ) { * xp = * xp ^ * yp ; * yp = * xp ^ * yp ; * xp = * xp ^ * yp ; }
int main ( ) { int x = 10 ; swap ( & x , & x ) ; printf ( " After ▁ swap ( & x , ▁ & x ) : ▁ x ▁ = ▁ % d " , x ) ; return 0 ; }
void nextGreatest ( int arr [ ] , int size ) {
int max_from_right = arr [ size - 1 ] ;
arr [ size - 1 ] = -1 ;
for ( int i = size - 2 ; i >= 0 ; i -- ) {
int temp = arr [ i ] ;
arr [ i ] = max_from_right ;
if ( max_from_right < temp ) max_from_right = temp ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 16 , 17 , 4 , 3 , 5 , 2 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; nextGreatest ( arr , size ) ; printf ( " The ▁ modified ▁ array ▁ is : ▁ STRNEWLINE " ) ; printArray ( arr , size ) ; return ( 0 ) ; }
int maxDiff ( int arr [ ] , int arr_size ) { int max_diff = arr [ 1 ] - arr [ 0 ] ; int i , j ; for ( i = 0 ; i < arr_size ; i ++ ) { for ( j = i + 1 ; j < arr_size ; j ++ ) { if ( arr [ j ] - arr [ i ] > max_diff ) max_diff = arr [ j ] - arr [ i ] ; } } return max_diff ; }
int main ( ) { int arr [ ] = { 1 , 2 , 90 , 10 , 110 } ;
printf ( " Maximum ▁ difference ▁ is ▁ % d " , maxDiff ( arr , 5 ) ) ; getchar ( ) ; return 0 ; }
int findMaximum ( int arr [ ] , int low , int high ) { int max = arr [ low ] ; int i ; for ( i = low + 1 ; i <= high ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; else break ; } return max ; }
int main ( ) { int arr [ ] = { 1 , 30 , 40 , 50 , 60 , 70 , 23 , 20 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ maximum ▁ element ▁ is ▁ % d " , findMaximum ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMaximum ( int arr [ ] , int low , int high ) {
if ( low == high ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] >= arr [ high ] ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] < arr [ high ] ) return arr [ high ] ; int mid = ( low + high ) / 2 ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] > arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] < arr [ mid - 1 ] ) return findMaximum ( arr , low , mid - 1 ) ;
else return findMaximum ( arr , mid + 1 , high ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 50 , 10 , 9 , 7 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ maximum ▁ element ▁ is ▁ % d " , findMaximum ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
int getMissingNo ( int a [ ] , int n ) { int i , total ; total = ( n + 1 ) * ( n + 2 ) / 2 ; for ( i = 0 ; i < n ; i ++ ) total -= a [ i ] ; return total ; }
int main ( ) { int a [ ] = { 1 , 2 , 4 , 5 , 6 } ; int miss = getMissingNo ( a , 5 ) ; printf ( " % d " , miss ) ; getchar ( ) ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE void printTwoElements ( int arr [ ] , int size ) { int i ; printf ( " The repeating element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } printf ( " and the missing element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) printf ( " % d " , i + 1 ) ; } }
int main ( ) { int arr [ ] = { 7 , 3 , 4 , 5 , 5 , 6 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoElements ( arr , n ) ; return 0 ; }
void getTwoElements ( int arr [ ] , int n , int * x , int * y ) {
int xor1 ;
int set_bit_no ; int i ; * x = 0 ; * y = 0 ; xor1 = arr [ 0 ] ;
for ( i = 1 ; i < n ; i ++ ) xor1 = xor1 ^ arr [ i ] ;
for ( i = 1 ; i <= n ; i ++ ) xor1 = xor1 ^ i ;
set_bit_no = xor1 & ~ ( xor1 - 1 ) ;
for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] & set_bit_no )
* x = * x ^ arr [ i ] ; else
* y = * y ^ arr [ i ] ; } for ( i = 1 ; i <= n ; i ++ ) { if ( i & set_bit_no )
* x = * x ^ i ; else
* y = * y ^ i ; }
}
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 5 , 5 , 6 , 2 } ; int * x = ( int * ) malloc ( sizeof ( int ) ) ; int * y = ( int * ) malloc ( sizeof ( int ) ) ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; getTwoElements ( arr , n , x , y ) ; printf ( " ▁ The ▁ missing ▁ element ▁ is ▁ % d " " ▁ and ▁ the ▁ repeating ▁ number " " ▁ is ▁ % d " , * x , * y ) ; getchar ( ) ; }
void findFourElements ( int A [ ] , int n , int X ) {
for ( int i = 0 ; i < n - 3 ; i ++ ) {
for ( int j = i + 1 ; j < n - 2 ; j ++ ) {
for ( int k = j + 1 ; k < n - 1 ; k ++ ) {
for ( int l = k + 1 ; l < n ; l ++ ) if ( A [ i ] + A [ j ] + A [ k ] + A [ l ] == X ) printf ( " % d , ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] , A [ l ] ) ; } } } }
int main ( ) { int A [ ] = { 10 , 20 , 30 , 40 , 1 , 2 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int X = 91 ; findFourElements ( A , n , X ) ; return 0 ; }
int minDistance ( int arr [ ] , int n ) { int maximum_element = arr [ 0 ] ; int min_dis = n ; int index = 0 ; for ( int i = 1 ; i < n ; i ++ ) {
if ( maximum_element == arr [ i ] ) { min_dis = min ( min_dis , ( i - index ) ) ; index = i ; }
else if ( maximum_element < arr [ i ] ) { maximum_element = arr [ i ] ; min_dis = n ; index = i ; }
else continue ; } return min_dis ; }
int main ( ) { int arr [ ] = { 6 , 3 , 1 , 3 , 6 , 4 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Minimum ▁ distance ▁ = ▁ " << minDistance ( arr , n ) ; return 0 ; }
int maxSumIS ( int arr [ ] , int n ) { int i , j , max = 0 ; int msis [ n ] ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 1 , 101 , 2 , 3 , 100 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ " " subsequence ▁ is ▁ % d STRNEWLINE " , maxSumIS ( arr , n ) ) ; return 0 ; }
float per ( float a , float b ) { return ( a + b ) ; }
float area ( float s ) { return ( s / 2 ) ; }
int main ( ) { float a = 7 , b = 8 , s = 10 ; printf ( " % f STRNEWLINE " , per ( a , b ) ) ; printf ( " % f " , area ( s ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define PI  3.14159265
float area_leaf ( float a ) { return ( a * a * ( PI / 2 - 1 ) ) ; }
int main ( ) { float a = 7 ; printf ( " % f " , area_leaf ( a ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define PI  3.14159265
float length_rope ( float r ) { return ( ( 2 * PI * r ) + 6 * r ) ; }
int main ( ) { float r = 7 ; printf ( " % f " , length_rope ( r ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define PI  3.14159265
float area_inscribed ( float P , float B , float H ) { return ( ( P + B - H ) * ( P + B - H ) * ( PI / 4 ) ) ; }
int main ( ) { float P = 3 , B = 4 , H = 5 ; printf ( " % f " , area_inscribed ( P , B , H ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define PI  3.14159265
float area_circumscribed ( float c ) { return ( c * c * ( PI / 4 ) ) ; }
int main ( ) { float c = 8 ; printf ( " % f " , area_circumscribed ( c ) ) ; return 0 ; }
#include <math.h> NEW_LINE #include <stdio.h> NEW_LINE #define PI  3.14159265
float area_inscribed ( float a ) { return ( a * a * ( PI / 12 ) ) ; }
float perm_inscribed ( float a ) { return ( PI * ( a / sqrt ( 3 ) ) ) ; }
int main ( ) { float a = 6 ; printf ( " Area ▁ of ▁ inscribed ▁ circle ▁ is ▁ : % f STRNEWLINE " , area_inscribed ( a ) ) ; printf ( " Perimeter ▁ of ▁ inscribed ▁ circle ▁ is ▁ : % f " , perm_inscribed ( a ) ) ; return 0 ; }
float area ( float r ) {
return ( 0.5 ) * ( 3.14 ) * ( r * r ) ; }
float perimeter ( float r ) {
return ( 3.14 ) * ( r ) ; }
float r = 10 ;
printf ( " The ▁ Area ▁ of ▁ Semicircle : ▁ % f STRNEWLINE " , area ( r ) ) ;
printf ( " The ▁ Perimeter ▁ of ▁ Semicircle : ▁ % f STRNEWLINE " , perimeter ( r ) ) ; return 0 ; }
void equation_plane ( float x1 , float y1 , float z1 , float x2 , float y2 , float z2 , float x3 , float y3 , float z3 ) { float a1 = x2 - x1 ; float b1 = y2 - y1 ; float c1 = z2 - z1 ; float a2 = x3 - x1 ; float b2 = y3 - y1 ; float c2 = z3 - z1 ; float a = b1 * c2 - b2 * c1 ; float b = a2 * c1 - a1 * c2 ; float c = a1 * b2 - b1 * a2 ; float d = ( - a * x1 - b * y1 - c * z1 ) ; printf ( " equation ▁ of ▁ plane ▁ is ▁ % .2f ▁ x ▁ + ▁ % .2f " " ▁ y ▁ + ▁ % .2f ▁ z ▁ + ▁ % .2f ▁ = ▁ 0 . " , a , b , c , d ) ; return ; }
int main ( ) { float x1 = -1 ; float y1 = 2 ; float z1 = 1 ; float x2 = 0 ; float y2 = -3 ; float z2 = 2 ; float x3 = 1 ; float y3 = 1 ; float z3 = -4 ; equation_plane ( x1 , y1 , z1 , x2 , y2 , z2 , x3 , y3 , z3 ) ; return 0 ; }
void shortest_distance ( float x1 , float y1 , float a , float b , float c ) { float d = fabs ( ( a * x1 + b * y1 + c ) ) / ( sqrt ( a * a + b * b ) ) ; printf ( " Perpendicular ▁ distance ▁ is ▁ % f STRNEWLINE " , d ) ; return ; }
int main ( ) { float x1 = 5 ; float y1 = 6 ; float a = -2 ; float b = 3 ; float c = 4 ; shortest_distance ( x1 , y1 , a , b , c ) ; return 0 ; }
void octant ( float x , float y , float z ) { if ( x >= 0 && y >= 0 && z >= 0 ) printf ( " Point ▁ lies ▁ in ▁ 1st ▁ octant STRNEWLINE " ) ; else if ( x < 0 && y >= 0 && z >= 0 ) printf ( " Point ▁ lies ▁ in ▁ 2nd ▁ octant STRNEWLINE " ) ; else if ( x < 0 && y < 0 && z >= 0 ) printf ( " Point ▁ lies ▁ in ▁ 3rd ▁ octant STRNEWLINE " ) ; else if ( x >= 0 && y < 0 && z >= 0 ) printf ( " Point ▁ lies ▁ in ▁ 4th ▁ octant STRNEWLINE " ) ; else if ( x >= 0 && y >= 0 && z < 0 ) printf ( " Point ▁ lies ▁ in ▁ 5th ▁ octant STRNEWLINE " ) ; else if ( x < 0 && y >= 0 && z < 0 ) printf ( " Point ▁ lies ▁ in ▁ 6th ▁ octant STRNEWLINE " ) ; else if ( x < 0 && y < 0 && z < 0 ) printf ( " Point ▁ lies ▁ in ▁ 7th ▁ octant STRNEWLINE " ) ; else if ( x >= 0 && y < 0 && z < 0 ) printf ( " Point ▁ lies ▁ in ▁ 8th ▁ octant STRNEWLINE " ) ; }
int main ( ) { float x = 2 , y = 3 , z = 4 ; octant ( x , y , z ) ; x = -4 , y = 2 , z = -8 ; octant ( x , y , z ) ; x = -6 , y = -2 , z = 8 ; octant ( x , y , z ) ; }
#include <stdio.h> NEW_LINE #include <math.h> NEW_LINE double maxArea ( double a , double b , double c , double d ) {
double semiperimeter = ( a + b + c + d ) / 2 ;
return sqrt ( ( semiperimeter - a ) * ( semiperimeter - b ) * ( semiperimeter - c ) * ( semiperimeter - d ) ) ; }
int main ( ) { double a = 1 , b = 2 , c = 1 , d = 2 ; printf ( " % .2f STRNEWLINE " , maxArea ( a , b , c , d ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE void midptellipse ( int rx , int ry , int xc , int yc ) { float dx , dy , d1 , d2 , x , y ; x = 0 ; y = ry ;
d1 = ( ry * ry ) - ( rx * rx * ry ) + ( 0.25 * rx * rx ) ; dx = 2 * ry * ry * x ; dy = 2 * rx * rx * y ;
while ( dx < dy ) {
printf ( " ( % f , ▁ % f ) STRNEWLINE " , x + xc , y + yc ) ; printf ( " ( % f , ▁ % f ) STRNEWLINE " , - x + xc , y + yc ) ; printf ( " ( % f , ▁ % f ) STRNEWLINE " , x + xc , - y + yc ) ; printf ( " ( % f , ▁ % f ) STRNEWLINE " , - x + xc , - y + yc ) ;
if ( d1 < 0 ) { x ++ ; dx = dx + ( 2 * ry * ry ) ; d1 = d1 + dx + ( ry * ry ) ; } else { x ++ ; y -- ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d1 = d1 + dx - dy + ( ry * ry ) ; } }
d2 = ( ( ry * ry ) * ( ( x + 0.5 ) * ( x + 0.5 ) ) ) + ( ( rx * rx ) * ( ( y - 1 ) * ( y - 1 ) ) ) - ( rx * rx * ry * ry ) ;
while ( y >= 0 ) {
printf ( " ( % f , ▁ % f ) STRNEWLINE " , x + xc , y + yc ) ; printf ( " ( % f , ▁ % f ) STRNEWLINE " , - x + xc , y + yc ) ; printf ( " ( % f , ▁ % f ) STRNEWLINE " , x + xc , - y + yc ) ; printf ( " ( % f , ▁ % f ) STRNEWLINE " , - x + xc , - y + yc ) ;
if ( d2 > 0 ) { y -- ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + ( rx * rx ) - dy ; } else { y -- ; x ++ ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + dx - dy + ( rx * rx ) ; } } }
midptellipse ( 10 , 15 , 50 , 50 ) ; return 0 ; }
void HexToBin ( char * hexdec ) { long int i = 0 ; while ( hexdec [ i ] ) { switch ( hexdec [ i ] ) { case '0' : printf ( "0000" ) ; break ; case '1' : printf ( "0001" ) ; break ; case '2' : printf ( "0010" ) ; break ; case '3' : printf ( "0011" ) ; break ; case '4' : printf ( "0100" ) ; break ; case '5' : printf ( "0101" ) ; break ; case '6' : printf ( "0110" ) ; break ; case '7' : printf ( "0111" ) ; break ; case '8' : printf ( "1000" ) ; break ; case '9' : printf ( "1001" ) ; break ; case ' A ' : case ' a ' : printf ( "1010" ) ; break ; case ' B ' : case ' b ' : printf ( "1011" ) ; break ; case ' C ' : case ' c ' : printf ( "1100" ) ; break ; case ' D ' : case ' d ' : printf ( "1101" ) ; break ; case ' E ' : case ' e ' : printf ( "1110" ) ; break ; case ' F ' : case ' f ' : printf ( "1111" ) ; break ; default : printf ( " Invalid hexadecimal digit % c " , hexdec [ i ] ) ; } i ++ ; } }
char hexdec [ 100 ] = "1AC5" ;
printf ( " Equivalent Binary value is : " HexToBin ( hexdec ) ; }
void distance ( float x1 , float y1 , float z1 , float x2 , float y2 , float z2 ) { float d = sqrt ( pow ( x2 - x1 , 2 ) + pow ( y2 - y1 , 2 ) + pow ( z2 - z1 , 2 ) * 1.0 ) ; printf ( " Distance ▁ is ▁ % f " , d ) ; return ; }
int main ( ) { float x1 = 2 ; float y1 = -5 ; float z1 = 7 ; float x2 = 3 ; float y2 = 4 ; float z2 = 5 ;
distance ( x1 , y1 , z1 , x2 , y2 , z2 ) ; return 0 ; }
int No_Of_Pairs ( int N ) { int i = 1 ;
while ( ( i * i * i ) + ( 2 * i * i ) + i <= N ) i ++ ; return ( i - 1 ) ; }
void print_pairs ( int pairs ) { int i = 1 , mul ; for ( i = 1 ; i <= pairs ; i ++ ) { mul = i * ( i + 1 ) ; printf ( " Pair ▁ no . ▁ % d ▁ - - > ▁ ( % d , ▁ % d ) STRNEWLINE " , i , ( mul * i ) , mul * ( i + 1 ) ) ; } }
int main ( ) { int N = 500 , pairs , mul , i = 1 ; pairs = No_Of_Pairs ( N ) ; printf ( " No . ▁ of ▁ pairs ▁ = ▁ % d ▁ STRNEWLINE " , pairs ) ; print_pairs ( pairs ) ; return 0 ; }
double findArea ( double d ) { return ( d * d ) / 2 ; }
int main ( ) { double d = 10 ; printf ( " % .2f " , findArea ( d ) ) ; return 0 ; }
float AvgofSquareN ( int n ) { float sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) sum += ( i * i ) ; return sum / n ; }
int main ( ) { int n = 2 ; printf ( " % f " , AvgofSquareN ( n ) ) ; return 0 ; }
double Series ( double x , int n ) { double sum = 1 , term = 1 , fct , j , y = 2 , m ;
int i ; for ( i = 1 ; i < n ; i ++ ) { fct = 1 ; for ( j = 1 ; j <= y ; j ++ ) { fct = fct * j ; } term = term * ( -1 ) ; m = term * pow ( x , y ) / fct ; sum = sum + m ; y += 2 ; } return sum ; }
int main ( ) { double x = 9 ; int n = 10 ; printf ( " % .4f " , Series ( x , n ) ) ; return 0 ; }
long long maxPrimeFactors ( long long n ) {
long long maxPrime = -1 ;
while ( n % 2 == 0 ) { maxPrime = 2 ;
}
while ( n % 3 == 0 ) { maxPrime = 3 ; n = n / 3 ; }
for ( int i = 5 ; i <= sqrt ( n ) ; i += 6 ) { while ( n % i == 0 ) { maxPrime = i ; n = n / i ; } while ( n % ( i + 2 ) == 0 ) { maxPrime = i + 2 ; n = n / ( i + 2 ) ; } }
if ( n > 4 ) maxPrime = n ; return maxPrime ; }
int main ( ) { long long n = 15 ; printf ( " % lld STRNEWLINE " , maxPrimeFactors ( n ) ) ; n = 25698751364526 ; printf ( " % lld " , maxPrimeFactors ( n ) ) ; return 0 ; }
double sum ( int x , int n ) { double i , total = 1.0 , multi = x ; for ( i = 1 ; i <= n ; i ++ ) { total = total + multi / i ; multi = multi * x ; } return total ; }
int main ( ) { int x = 2 ; int n = 5 ; printf ( " % .2f " , sum ( x , n ) ) ; return 0 ; }
void triangular_series ( int n ) { int i , j = 1 , k = 1 ;
for ( i = 1 ; i <= n ; i ++ ) { printf ( " ▁ % d ▁ " , k ) ;
j = j + 1 ;
k = k + j ; } }
int main ( ) { int n = 5 ; triangular_series ( n ) ; return 0 ; }
#include <stdio.h> NEW_LINE int countDigit ( long long n ) { if ( n / 10 == 0 ) return 1 ; return 1 + countDigit ( n / 10 ) ; }
int main ( void ) { long long n = 345289467 ; printf ( " Number ▁ of ▁ digits ▁ : ▁ % d " , countDigit ( n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMaxValue ( ) { int res = 2 ; long long int fact = 2 ; while ( 1 ) {
if ( fact < 0 ) break ; res ++ ; fact = fact * res ; } return res - 1 ; }
int main ( ) { printf ( " Maximum ▁ value ▁ of ▁ integer ▁ : ▁ % d STRNEWLINE " , findMaxValue ( ) ) ; return 0 ; }
long long firstkdigits ( int n , int k ) {
long double product = n * log10 ( n ) ;
long double decimal_part = product - floor ( product ) ;
decimal_part = pow ( 10 , decimal_part ) ;
long long digits = pow ( 10 , k - 1 ) , i = 0 ; return decimal_part * digits ; }
int main ( ) { int n = 1450 ; int k = 6 ; cout << firstkdigits ( n , k ) ; return 0 ; }
long long moduloMultiplication ( long long a , long long b , long long mod ) {
a %= mod ; while ( b ) {
if ( b & 1 ) res = ( res + a ) % mod ;
a = ( 2 * a ) % mod ;
} return res ; }
int main ( ) { long long a = 10123465234878998 ; long long b = 65746311545646431 ; long long m = 10005412336548794 ; printf ( " % lld " , moduloMultiplication ( a , b , m ) ) ; return 0 ; }
void findRoots ( int a , int b , int c ) {
if ( a == 0 ) { printf ( " Invalid " ) ; return ; } int d = b * b - 4 * a * c ; double sqrt_val = sqrt ( abs ( d ) ) ; if ( d > 0 ) { printf ( " Roots ▁ are ▁ real ▁ and ▁ different ▁ STRNEWLINE " ) ; printf ( " % f % f " , ( double ) ( - b + sqrt_val ) / ( 2 * a ) , ( double ) ( - b - sqrt_val ) / ( 2 * a ) ) ; } else if ( d == 0 ) { printf ( " Roots ▁ are ▁ real ▁ and ▁ same ▁ STRNEWLINE " ) ; printf ( " % f " , - ( double ) b / ( 2 * a ) ) ; }
{ printf ( " Roots ▁ are ▁ complex ▁ STRNEWLINE " ) ; printf ( " % f ▁ + ▁ i % f % f - i % f " , - ( double ) b / ( 2 * a ) , sqrt_val / ( 2 * a ) , - ( double ) b / ( 2 * a ) , sqrt_val / ( 2 * a ) ; } }
int main ( ) { int a = 1 , b = -7 , c = 12 ;
findRoots ( a , b , c ) ; return 0 ; }
int val ( char c ) { if ( c >= '0' && c <= '9' ) return ( int ) c - '0' ; else return ( int ) c - ' A ' + 10 ; }
int toDeci ( char * str , int base ) { int len = strlen ( str ) ;
int power = 1 ;
int num = 0 ; int i ;
for ( i = len - 1 ; i >= 0 ; i -- ) {
if ( val ( str [ i ] ) >= base ) { printf ( " Invalid ▁ Number " ) ; return -1 ; } num += val ( str [ i ] ) * power ; power = power * base ; } return num ; }
int main ( ) { char str [ ] = "11A " ; int base = 16 ; printf ( " Decimal ▁ equivalent ▁ of ▁ % s ▁ in ▁ base ▁ % d ▁ is ▁ " " ▁ % d STRNEWLINE " , str , base , toDeci ( str , base ) ) ; return 0 ; }
int seriesSum ( int calculated , int current , int N ) { int i , cur = 1 ;
if ( current == N + 1 ) return 0 ;
for ( i = calculated ; i < calculated + current ; i ++ ) cur *= i ;
return cur + seriesSum ( i , current + 1 , N ) ; }
int N = 5 ;
printf ( " % d STRNEWLINE " , seriesSum ( 1 , 1 , N ) ) ; return 0 ; }
int modInverse ( int a , int m ) { int m0 = m ; int y = 0 , x = 1 ; if ( m == 1 ) return 0 ; while ( a > 1 ) {
int q = a / m ; int t = m ;
m = a % m , a = t ; t = y ;
y = x - q * y ; x = t ; }
if ( x < 0 ) x += m0 ; return x ; }
int main ( ) { int a = 3 , m = 11 ;
printf ( " Modular ▁ multiplicative ▁ inverse ▁ is ▁ % d STRNEWLINE " , modInverse ( a , m ) ) ; return 0 ; }
int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
int phi ( unsigned int n ) { unsigned int result = 1 ; for ( int i = 2 ; i < n ; i ++ ) if ( gcd ( i , n ) == 1 ) result ++ ; return result ; }
int main ( ) { int n ; for ( n = 1 ; n <= 10 ; n ++ ) printf ( " phi ( % d ) ▁ = ▁ % d STRNEWLINE " , n , phi ( n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int phi ( int n ) {
for ( int p = 2 ; p * p <= n ; ++ p ) {
if ( n % p == 0 ) {
while ( n % p == 0 ) n /= p ; result *= ( 1.0 - ( 1.0 / ( float ) p ) ) ; } }
if ( n > 1 ) result *= ( 1.0 - ( 1.0 / ( float ) n ) ) ; return ( int ) result ; }
int main ( ) { int n ; for ( n = 1 ; n <= 10 ; n ++ ) printf ( " phi ( % d ) ▁ = ▁ % d STRNEWLINE " , n , phi ( n ) ) ; return 0 ; }
void printFibonacciNumbers ( int n ) { int f1 = 0 , f2 = 1 , i ; if ( n < 1 ) return ; printf ( " % d ▁ " , f1 ) ; for ( i = 1 ; i < n ; i ++ ) { printf ( " % d ▁ " , f2 ) ; int next = f1 + f2 ; f1 = f2 ; f2 = next ; } }
int main ( ) { printFibonacciNumbers ( 7 ) ; return 0 ; }
int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
int lcm ( int a , int b ) { return ( a / gcd ( a , b ) ) * b ; }
int main ( ) { int a = 15 , b = 20 ; printf ( " LCM ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d ▁ " , a , b , lcm ( a , b ) ) ; return 0 ; }
void convert_to_words ( char * num ) { int len = strlen (
if ( len == 0 ) { fprintf ( stderr , " empty ▁ string STRNEWLINE " ) ; return ; } if ( len > 4 ) { fprintf ( stderr , " Length ▁ more ▁ than ▁ 4 ▁ is ▁ not ▁ supported STRNEWLINE " ) ; return ; }
char * single_digits [ ] = { " zero " , " one " , " two " , " three " , " four " , " five " , " six " , " seven " , " eight " , " nine " } ;
char * two_digits [ ] = { " " , " ten " , " eleven " , " twelve " , " thirteen " , " fourteen " , " fifteen " , " sixteen " , " seventeen " , " eighteen " , " nineteen " } ;
char * tens_multiple [ ] = { " " , " " , " twenty " , " thirty " , " forty " , " fifty " , " sixty " , " seventy " , " eighty " , " ninety " } ; char * tens_power [ ] = { " hundred " , " thousand " } ;
printf ( " % s : " , num ) ;
if ( len == 1 ) { printf ( " % s STRNEWLINE " , single_digits [ * num - '0' ] ) ; return ; }
while ( * num != ' \0' ) {
if ( len >= 3 ) { if ( * num - '0' != 0 ) { printf ( " % s ▁ " , single_digits [ * num - '0' ] ) ; printf ( " % s ▁ " ,
} -- len ; }
else {
if ( * num == '1' ) { int sum = * num - '0' + * ( num + 1 ) - '0' ; printf ( " % s STRNEWLINE " , two_digits [ sum ] ) ; return ; }
else if ( * num = = '2' && * ( num + 1 ) == '0' ) { printf ( " twenty STRNEWLINE " ) ; return ; }
else { int i = * num - '0' ; printf ( " % s ▁ " , i ? tens_multiple [ i ] : " " ) ; ++ num ; if ( * num != '0' ) printf ( " % s ▁ " , single_digits [ * num - '0' ] ) ; } } ++ num ; } }
int main ( void ) { convert_to_words ( "9923" ) ; convert_to_words ( "523" ) ; convert_to_words ( "89" ) ; convert_to_words ( "8" ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE #include <string.h> NEW_LINE # define MAX  11 NEW_LINE bool isMultipleof5 ( int n ) { char str [ MAX ] ; int len = strlen ( str ) ;
if ( str [ len - 1 ] == '5' str [ len - 1 ] == '0' ) return true ; return false ; }
int main ( ) { int n = 19 ; if ( isMultipleof5 ( n ) == true ) printf ( " % d ▁ is ▁ multiple ▁ of ▁ 5 STRNEWLINE " , n ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ multiple ▁ of ▁ 5 STRNEWLINE " , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE int add ( int x , int y ) { int keep = ( x & y ) << 1 ; int res = x ^ y ;
if ( keep == 0 ) return res ; add ( keep , res ) ; }
int main ( ) { printf ( " % d " , add ( 15 , 38 ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <math.h> NEW_LINE unsigned countBits ( unsigned int number ) {
return ( int ) log2 ( number ) + 1 ; }
int main ( ) { unsigned int num = 65 ; printf ( " % d STRNEWLINE " , countBits ( num ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define INT_SIZE  32
int constructNthNumber ( int group_no , int aux_num , int op ) { int a [ INT_SIZE ] = { 0 } ; int num = 0 , len_f ; int i = 0 ;
if ( op == 2 ) {
len_f = 2 * group_no ;
a [ len_f - 1 ] = a [ 0 ] = 1 ;
while ( aux_num ) {
a [ group_no + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
else if ( op == 0 ) {
len_f = 2 * group_no + 1 ;
a [ len_f - 1 ] = a [ 0 ] = 1 ; a [ group_no ] = 0 ;
while ( aux_num ) {
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
{
len_f = 2 * group_no + 1 ;
a [ len_f - 1 ] = a [ 0 ] = 1 ; a [ group_no ] = 1 ;
while ( aux_num ) {
a [ group_no + 1 + i ] = a [ group_no - 1 - i ] = aux_num & 1 ; aux_num = aux_num >> 1 ; i ++ ; } }
for ( i = 0 ; i < len_f ; i ++ ) num += ( 1 << i ) * a [ i ] ; return num ; }
int getNthNumber ( int n ) { int group_no = 0 , group_offset ; int count_upto_group = 0 , count_temp = 1 ; int op , aux_num ;
while ( count_temp < n ) { group_no ++ ;
count_upto_group = count_temp ; count_temp += 3 * ( 1 << ( group_no - 1 ) ) ; }
group_offset = n - count_upto_group - 1 ;
if ( ( group_offset + 1 ) <= ( 1 << ( group_no - 1 ) ) ) {
aux_num = group_offset ; } else { if ( ( ( group_offset + 1 ) - ( 1 << ( group_no - 1 ) ) ) % 2 )
else
aux_num = ( ( group_offset ) - ( 1 << ( group_no - 1 ) ) ) / 2 ; } return constructNthNumber ( group_no , aux_num , op ) ; }
int main ( ) { int n = 9 ;
printf ( " % d " , getNthNumber ( n ) ) ; return 0 ; }
void flip ( int arr [ ] , int i ) { int temp , start = 0 ; while ( start < i ) { temp = arr [ start ] ; arr [ start ] = arr [ i ] ; arr [ i ] = temp ; start ++ ; i -- ; } }
int findMax ( int arr [ ] , int n ) { int mi , i ; for ( mi = 0 , i = 0 ; i < n ; ++ i ) if ( arr [ i ] > arr [ mi ] ) mi = i ; return mi ; }
void pancakeSort ( int * arr , int n ) {
for ( int curr_size = n ; curr_size > 1 ; -- curr_size ) {
int mi = findMax ( arr , curr_size ) ;
if ( mi != curr_size - 1 ) {
flip ( arr , mi ) ;
flip ( arr , curr_size - 1 ) ; } } }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; ++ i ) printf ( " % d ▁ " , arr [ i ] ) ; }
int main ( ) { int arr [ ] = { 23 , 10 , 20 , 11 , 12 , 6 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; pancakeSort ( arr , n ) ; puts ( " Sorted ▁ Array ▁ " ) ; printArray ( arr , n ) ; return 0 ; }
