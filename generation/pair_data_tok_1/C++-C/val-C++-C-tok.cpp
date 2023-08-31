#include <iostream> NEW_LINE using namespace std ;
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
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int permutationCoeff ( int n , int k ) { int fact [ n + 1 ] ;
fact [ 0 ] = 1 ;
for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = i * fact [ i - 1 ] ;
return fact [ n ] / fact [ n - k ] ; }
int main ( ) { int n = 10 , k = 2 ; cout << " Value ▁ of ▁ P ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << permutationCoeff ( n , k ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isSubsetSum ( int set [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 ) return false ;
if ( set [ n - 1 ] > sum ) return isSubsetSum ( set , n - 1 , sum ) ;
return isSubsetSum ( set , n - 1 , sum ) || isSubsetSum ( set , n - 1 , sum - set [ n - 1 ] ) ; }
int main ( ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = sizeof ( set ) / sizeof ( set [ 0 ] ) ; if ( isSubsetSum ( set , n , sum ) == true ) cout << " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else cout << " No ▁ subset ▁ with ▁ given ▁ sum " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int pell ( int n ) { if ( n <= 2 ) return n ; int a = 1 ; int b = 2 ; int c , i ; for ( i = 3 ; i <= n ; i ++ ) { c = 2 * b + a ; a = b ; b = c ; } return b ; }
int main ( ) { int n = 4 ; cout << pell ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
unsigned int factorial ( unsigned int n ) { if ( n == 0 ) return 1 ; return n * factorial ( n - 1 ) ; }
int main ( ) { int num = 5 ; cout << " Factorial ▁ of ▁ " << num << " ▁ is ▁ " << factorial ( num ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findSubArray ( int arr [ ] , int n ) { int sum = 0 ; int maxsize = -1 , startindex ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { sum = ( arr [ i ] == 0 ) ? -1 : 1 ;
for ( int j = i + 1 ; j < n ; j ++ ) { ( arr [ j ] == 0 ) ? ( sum += -1 ) : ( sum += 1 ) ;
if ( sum == 0 && maxsize < j - i + 1 ) { maxsize = j - i + 1 ; startindex = i ; } } } if ( maxsize == -1 ) cout << " No ▁ such ▁ subarray " ; else cout << startindex << " ▁ to ▁ " << startindex + maxsize - 1 ; return maxsize ; }
int main ( ) { int arr [ ] = { 1 , 0 , 0 , 1 , 0 , 1 , 1 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; findSubArray ( arr , size ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int ternarySearch ( int l , int r , int key , int ar [ ] ) { while ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
r = mid1 - 1 ; } else if ( key > ar [ mid2 ] ) {
l = mid2 + 1 ; } else {
l = mid1 + 1 ; r = mid2 - 1 ; } }
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
cout << " Index ▁ of ▁ " << key << " ▁ is ▁ " << p ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int findMin ( int arr [ ] , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
int main ( ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr1 , 0 , n1 - 1 ) << endl ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr2 , 0 , n2 - 1 ) << endl ; int arr3 [ ] = { 1 } ; int n3 = sizeof ( arr3 ) / sizeof ( arr3 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr3 , 0 , n3 - 1 ) << endl ; int arr4 [ ] = { 1 , 2 } ; int n4 = sizeof ( arr4 ) / sizeof ( arr4 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr4 , 0 , n4 - 1 ) << endl ; int arr5 [ ] = { 2 , 1 } ; int n5 = sizeof ( arr5 ) / sizeof ( arr5 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr5 , 0 , n5 - 1 ) << endl ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = sizeof ( arr6 ) / sizeof ( arr6 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr6 , 0 , n6 - 1 ) << endl ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = sizeof ( arr7 ) / sizeof ( arr7 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr7 , 0 , n7 - 1 ) << endl ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = sizeof ( arr8 ) / sizeof ( arr8 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr8 , 0 , n8 - 1 ) << endl ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = sizeof ( arr9 ) / sizeof ( arr9 [ 0 ] ) ; cout << " The ▁ minimum ▁ element ▁ is ▁ " << findMin ( arr9 , 0 , n9 - 1 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void print2Smallest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { cout << " ▁ Invalid ▁ Input ▁ " ; return ; } first = second = INT_MAX ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] < first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] < second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == INT_MAX ) cout << " There ▁ is ▁ no ▁ second ▁ smallest ▁ element STRNEWLINE " ; else cout << " The ▁ smallest ▁ element ▁ is ▁ " << first << " ▁ and ▁ second ▁ " " Smallest ▁ element ▁ is ▁ " << second << endl ; }
int main ( ) { int arr [ ] = { 12 , 13 , 1 , 10 , 34 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; print2Smallest ( arr , n ) ; return 0 ; }
bool isSubsetSum ( int arr [ ] , int n , int sum ) {
bool subset [ 2 ] [ sum + 1 ] ; for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) {
if ( j == 0 ) subset [ i % 2 ] [ j ] = true ;
else if ( i == 0 ) subset [ i % 2 ] [ j ] = false ; else if ( arr [ i - 1 ] <= j ) subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j - arr [ i - 1 ] ] || subset [ ( i + 1 ) % 2 ] [ j ] ; else subset [ i % 2 ] [ j ] = subset [ ( i + 1 ) % 2 ] [ j ] ; } } return subset [ n % 2 ] [ sum ] ; }
int main ( ) { int arr [ ] = { 6 , 2 , 5 } ; int sum = 7 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; if ( isSubsetSum ( arr , n , sum ) == true ) cout << " There ▁ exists ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else cout << " No ▁ subset ▁ exists ▁ with ▁ given ▁ sum " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; for ( int i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
bool isMajority ( int a [ ] , int size , int cand ) { int count = 0 ; for ( int i = 0 ; i < size ; i ++ ) if ( a [ i ] == cand ) count ++ ; if ( count > size / 2 ) return 1 ; else return 0 ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) cout << " ▁ " << cand << " ▁ " ; else cout << " No ▁ Majority ▁ Element " ; }
int main ( ) { int a [ ] = { 1 , 3 , 3 , 1 , 2 } ; int size = ( sizeof ( a ) ) / sizeof ( a [ 0 ] ) ;
printMajority ( a , size ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isSubsetSum ( int set [ ] , int n , int sum ) {
bool subset [ n + 1 ] [ sum + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ i ] [ 0 ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ 0 ] [ i ] = false ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) { if ( j < set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] ; if ( j >= set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] || subset [ i - 1 ] [ j - set [ i - 1 ] ] ; } }
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) printf ( " % 4d " , subset [ i ] [ j ] ) ; cout << " STRNEWLINE " ; } return subset [ n ] [ sum ] ; }
int main ( ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = sizeof ( set ) / sizeof ( set [ 0 ] ) ; if ( isSubsetSum ( set , n , sum ) == true ) cout << " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else cout << " No ▁ subset ▁ with ▁ given ▁ sum " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define bool  int
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerOfTwo ( 64 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; unsigned int nextPowerOf2 ( unsigned int n ) { unsigned count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( n != 0 ) { n >>= 1 ; count += 1 ; } return 1 << count ; }
int main ( ) { unsigned int n = 0 ; cout << nextPowerOf2 ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int countWays ( int n ) { int res [ n + 1 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
int main ( ) { int n = 4 ; cout << countWays ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ? x : y ) ; }
int maxTasks ( int high [ ] , int low [ ] , int n ) {
if ( n <= 0 ) return 0 ;
return max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
int main ( ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; cout << maxTasks ( high , low , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define OUT  0 NEW_LINE #define IN  1
unsigned countWords ( char * str ) { int state = OUT ;
unsigned wc = 0 ;
while ( * str ) {
if ( * str == ' ▁ ' * str == ' ' * str == ' TABSYMBOL ' ) state = OUT ;
else if ( state == OUT ) { state = IN ; ++ wc ; }
++ str ; } return wc ; }
int main ( void ) { char str [ ] = " One ▁ twothree STRNEWLINE ▁ four TABSYMBOL five ▁ " ; cout << " No ▁ of ▁ words ▁ : ▁ " << countWords ( str ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ? x : y ) ; }
int maxTasks ( int high [ ] , int low [ ] , int n ) {
int task_dp [ n + 1 ] ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( int i = 2 ; i <= n ; i ++ ) task_dp [ i ] = max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
int main ( ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; cout << maxTasks ( high , low , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool findPartiion ( int arr [ ] , int n ) { int sum = 0 ; int i , j ;
for ( i = 0 ; i < n ; i ++ ) sum += arr [ i ] ; if ( sum % 2 != 0 ) return false ; bool part [ sum / 2 + 1 ] [ n + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) part [ 0 ] [ i ] = true ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) part [ i ] [ 0 ] = false ;
for ( i = 1 ; i <= sum / 2 ; i ++ ) { for ( j = 1 ; j <= n ; j ++ ) { part [ i ] [ j ] = part [ i ] [ j - 1 ] ; if ( i >= arr [ j - 1 ] ) part [ i ] [ j ] = part [ i ] [ j ] || part [ i - arr [ j - 1 ] ] [ j - 1 ] ; } }
return part [ sum / 2 ] [ n ] ; }
int main ( ) { int arr [ ] = { 3 , 1 , 1 , 2 , 2 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) cout << " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ of ▁ equal ▁ " " sum " ; else cout << " Can ▁ not ▁ be ▁ divided ▁ into " << " ▁ two ▁ subsets ▁ of ▁ equal ▁ sum " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int dfa = 0 ;
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
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / pow ( i , i ) ; sums += ser ; } return sums ; }
int main ( ) { int n = 3 ; double res = Series ( n ) ; cout << res ; return 0 ; }
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
#include <iostream> NEW_LINE using namespace std ; int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == x ) return i ; return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int result = search ( arr , n , x ) ; ( result == -1 ) ? cout << " Element ▁ is ▁ not ▁ present ▁ in ▁ array " : cout << " Element ▁ is ▁ present ▁ at ▁ index ▁ " << result ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ; #define RANGE  255
void countSort ( char arr [ ] ) {
char output [ strlen ( arr ) ] ;
int count [ RANGE + 1 ] , i ; memset ( count , 0 , sizeof ( count ) ) ;
for ( i = 0 ; arr [ i ] ; ++ i ) ++ count [ arr [ i ] ] ;
for ( i = 1 ; i <= RANGE ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( i = 0 ; arr [ i ] ; ++ i ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( i = 0 ; arr [ i ] ; ++ i ) arr [ i ] = output [ i ] ; }
int main ( ) { char arr [ ] = " geeksforgeeks " ; countSort ( arr ) ; cout << " Sorted ▁ character ▁ array ▁ is ▁ " << arr ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
int main ( ) { int n = 5 , k = 2 ; cout << " Value ▁ of ▁ C ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << binomialCoeff ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) cout << " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " << N << " ▁ keystrokes ▁ is ▁ " << findoptimal ( N ) << endl ; }
#include <iostream> NEW_LINE using namespace std ;
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int screen [ N ] ;
int b ;
int n ; for ( n = 1 ; n <= 6 ; n ++ ) screen [ n - 1 ] = n ;
for ( n = 7 ; n <= N ; n ++ ) {
screen [ n - 1 ] = 0 ;
for ( b = n - 3 ; b >= 1 ; b -- ) {
int curr = ( n - b - 1 ) * screen [ b - 1 ] ; if ( curr > screen [ n - 1 ] ) screen [ n - 1 ] = curr ; } } return screen [ N - 1 ] ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) cout << " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ " << N << " ▁ keystrokes ▁ is ▁ " << findoptimal ( N ) << endl ; }
#include <iostream> NEW_LINE using namespace std ; class gfg {
public : int power ( int x , unsigned int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; } } ;
int main ( ) { gfg g ; int x = 2 ; unsigned int y = 3 ; cout << g . power ( x , y ) ; return 0 ; }
int power ( int x , unsigned int y ) { int temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else return x * temp * temp ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
int main ( ) { float x = 2 ; int y = -3 ; cout << power ( x , y ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; class gfg {
public : float squareRoot ( float n ) {
float x = n ; float y = 1 ; float e = 0.000001 ;
while ( x - y > e ) { x = ( x + y ) / 2 ; y = n / x ; } return x ; } } ;
int main ( ) { gfg g ; int n = 50 ; cout << " Square ▁ root ▁ of ▁ " << n << " ▁ is ▁ " << g . squareRoot ( n ) ; getchar ( ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float getAvg ( int x ) { static int sum , n ; sum += x ; return ( ( ( float ) sum ) / ++ n ) ; }
void streamAvg ( float arr [ ] , int n ) { float avg = 0 ; for ( int i = 0 ; i < n ; i ++ ) { avg = getAvg ( arr [ i ] ) ; cout << " Average ▁ of ▁ " << i + 1 << " ▁ numbers ▁ is ▁ " << fixed << setprecision ( 1 ) << avg << endl ; } return ; }
int main ( ) { float arr [ ] = { 10 , 20 , 30 , 40 , 50 , 60 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; streamAvg ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
#include <bits/stdc++.h> NEW_LINE using namespace std ; void combinationUtil ( int arr [ ] , int data [ ] , int start , int end , int index , int r ) ;
void printCombination ( int arr [ ] , int n , int r ) {
int data [ r ] ;
combinationUtil ( arr , data , 0 , n - 1 , 0 , r ) ; }
void combinationUtil ( int arr [ ] , int data [ ] , int start , int end , int index , int r ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) cout << data [ j ] << " ▁ " ; cout << endl ; return ; }
for ( int i = start ; i <= end && end - i + 1 >= r - index ; i ++ ) { data [ index ] = arr [ i ] ; combinationUtil ( arr , data , i + 1 , end , index + 1 , r ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printCombination ( arr , n , r ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) ;
void printCombination ( int arr [ ] , int n , int r ) {
int data [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) cout << data [ j ] << " ▁ " ; cout << endl ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printCombination ( arr , n , r ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unsigned int nextPowerOf2 ( unsigned int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
int main ( ) { unsigned int n = 5 ; cout << nextPowerOf2 ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
int main ( ) { int arr [ ] = { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; segregate0and1 ( arr , arr_size ) ; cout << " Array ▁ after ▁ segregation ▁ " ; for ( i = 0 ; i < 6 ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxIndexDiff ( int arr [ ] , int n ) { int maxDiff = -1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
int main ( ) { int arr [ ] = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int maxDiff = maxIndexDiff ( arr , n ) ; cout << " STRNEWLINE " << maxDiff ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; class GFG {
public : int findStep ( int n ) { if ( n == 1 n == 0 ) return 1 ; else if ( n == 2 ) return 2 ; else return findStep ( n - 3 ) + findStep ( n - 2 ) + findStep ( n - 1 ) ; } } ;
int main ( ) { GFG g ; int n = 4 ; cout << g . findStep ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
#include <bits/stdc++.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ; int findRepeatFirstN2 ( char * s ) {
int p = -1 , i , j ; for ( i = 0 ; i < strlen ( s ) ; i ++ ) { for ( j = i + 1 ; j < strlen ( s ) ; j ++ ) { if ( s [ i ] == s [ j ] ) { p = i ; break ; } } if ( p != -1 ) break ; } return p ; }
int main ( ) { char str [ ] = " geeksforgeeks " ; int pos = findRepeatFirstN2 ( str ) ; if ( pos == -1 ) cout << " Not ▁ found " ; else cout << str [ pos ] ; return 0 ; }
void reverseWords ( char * s ) { char * word_begin = NULL ;
char * temp = s ;
while ( * temp ) {
if ( ( word_begin == NULL ) && ( * temp != ' ▁ ' ) ) { word_begin = temp ; } if ( word_begin && ( ( * ( temp + 1 ) == ' ▁ ' ) || ( * ( temp + 1 ) == ' \0' ) ) ) { reverse ( word_begin , temp ) ; word_begin = NULL ; } temp ++ ; }
reverse ( s , temp - 1 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
bool isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
int main ( ) { int L = 110 , R = 1130 ; cout << " ▁ " << sumOfAllPalindrome ( L , R ) << endl ; }
#include <iostream> NEW_LINE using namespace std ; int subtractOne ( int x ) { int m = 1 ;
while ( ! ( x & m ) ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
int main ( ) { cout << subtractOne ( 13 ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int findSum ( int n , int a , int b ) { int sum = 0 ; for ( int i = 0 ; i < n ; i ++ )
if ( i % a == 0 i % b == 0 ) sum += i ; return sum ; }
int main ( ) { int n = 10 , a = 3 , b = 5 ; cout << findSum ( n , a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int pell ( int n ) { if ( n <= 2 ) return n ; return 2 * pell ( n - 1 ) + pell ( n - 2 ) ; }
int main ( ) { int n = 4 ; cout << " ▁ " << pell ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int largestPower ( int n , int p ) {
int x = 0 ;
while ( n ) { n /= p ; x += n ; } return x ; }
int main ( ) { int n = 10 , p = 3 ; cout << " The ▁ largest ▁ power ▁ of ▁ " << p << " ▁ that ▁ divides ▁ " << n << " ! ▁ is ▁ " << largestPower ( n , p ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int factorial ( int n ) {
return ( n == 1 n == 0 ) ? 1 : n * factorial ( n - 1 ) ; }
int main ( ) { int num = 5 ; cout << " Factorial ▁ of ▁ " << num << " ▁ is ▁ " << factorial ( num ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int bitExtracted ( int number , int k , int p ) { return ( ( ( 1 << k ) - 1 ) & ( number >> ( p - 1 ) ) ) ; }
int main ( ) { int number = 171 , k = 5 , p = 2 ; cout << " The ▁ extracted ▁ number ▁ is ▁ " << bitExtracted ( number , k , p ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int solve ( int a [ ] , int n ) { int max1 = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
int main ( ) { int arr [ ] = { -1 , 2 , 3 , -4 , -10 , 22 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Largest ▁ gap ▁ is ▁ : ▁ " << solve ( arr , size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int solve ( int a [ ] , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return abs ( min1 - max1 ) ; }
int main ( ) { int arr [ ] = { -1 , 2 , 3 , 4 , -10 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Largest ▁ gap ▁ is ▁ : ▁ " << solve ( arr , size ) ; return 0 ; }
