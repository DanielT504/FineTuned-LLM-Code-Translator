void Conversion ( double centi ) { double pixels = ( 96 * centi ) / 2.54 ; cout << fixed << setprecision ( 2 ) << pixels ; }
int main ( ) { double centi = 15 ; Conversion ( centi ) ; return 0 ; }
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
int printKDistinct ( int arr [ ] , int n , int k ) { int dist_count = 0 ; for ( int i = 0 ; i < n ; i ++ ) {
int j ; for ( j = 0 ; j < n ; j ++ ) if ( i != j && arr [ j ] == arr [ i ] ) break ;
if ( j == n ) dist_count ++ ; if ( dist_count == k ) return arr [ i ] ; } return -1 ; }
int main ( ) { int ar [ ] = { 1 , 2 , 1 , 3 , 4 , 2 } ; int n = sizeof ( ar ) / sizeof ( ar [ 0 ] ) ; int k = 2 ; cout << printKDistinct ( ar , n , k ) ; return 0 ; }
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
int findMaxSum ( int arr [ ] , int n ) { int res = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) { int prefix_sum = arr [ i ] ; for ( int j = 0 ; j < i ; j ++ ) prefix_sum += arr [ j ] ; int suffix_sum = arr [ i ] ; for ( int j = n - 1 ; j > i ; j -- ) suffix_sum += arr [ j ] ; if ( prefix_sum == suffix_sum ) res = max ( res , prefix_sum ) ; } return res ; }
int main ( ) { int arr [ ] = { -2 , 5 , 3 , 1 , 2 , 6 , -4 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << findMaxSum ( arr , n ) ; return 0 ; }
int findMaxSum ( int arr [ ] , int n ) {
int preSum [ n ] ;
int suffSum [ n ] ;
int ans = INT_MIN ;
preSum [ 0 ] = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) preSum [ i ] = preSum [ i - 1 ] + arr [ i ] ;
suffSum [ n - 1 ] = arr [ n - 1 ] ; if ( preSum [ n - 1 ] == suffSum [ n - 1 ] ) ans = max ( ans , preSum [ n - 1 ] ) ; for ( int i = n - 2 ; i >= 0 ; i -- ) { suffSum [ i ] = suffSum [ i + 1 ] + arr [ i ] ; if ( suffSum [ i ] == preSum [ i ] ) ans = max ( ans , preSum [ i ] ) ; } return ans ; }
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
bool isSubsetSum ( int set [ ] , int n , int sum ) {
bool subset [ n + 1 ] [ sum + 1 ] ;
for ( int i = 0 ; i <= n ; i ++ ) subset [ i ] [ 0 ] = true ;
for ( int i = 1 ; i <= sum ; i ++ ) subset [ 0 ] [ i ] = false ;
for ( int i = 1 ; i <= n ; i ++ ) { for ( int j = 1 ; j <= sum ; j ++ ) { if ( j < set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] ; if ( j >= set [ i - 1 ] ) subset [ i ] [ j ] = subset [ i - 1 ] [ j ] || subset [ i - 1 ] [ j - set [ i - 1 ] ] ; } }
for ( int i = 0 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= sum ; j ++ ) printf ( " % 4d " , subset [ i ] [ j ] ) ; cout << " STRNEWLINE " ; } return subset [ n ] [ sum ] ; }
int main ( ) { int set [ ] = { 3 , 34 , 4 , 12 , 5 , 2 } ; int sum = 9 ; int n = sizeof ( set ) / sizeof ( set [ 0 ] ) ; if ( isSubsetSum ( set , n , sum ) == true ) cout << " Found ▁ a ▁ subset ▁ with ▁ given ▁ sum " ; else cout << " No ▁ subset ▁ with ▁ given ▁ sum " ; return 0 ; }
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
SieveOfEratosthenes ( ) ;
int l = 3 , r = 9 ;
int c = ( sum [ r ] - sum [ l - 1 ] ) ;
cout << " Count : ▁ " << c << endl ; return 0 ; }
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
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; isPowerOfTwo ( 64 ) ? cout << " Yes STRNEWLINE " : cout << " No STRNEWLINE " ; return 0 ; }
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
void printTetra ( int n ) { if ( n < 0 ) return ;
int first = 0 , second = 1 ; int third = 1 , fourth = 2 ;
int curr ; if ( n == 0 ) cout << first ; else if ( n == 1 n == 2 ) cout << second ; else if ( n == 3 ) cout << fourth ; else {
for ( int i = 4 ; i <= n ; i ++ ) { curr = first + second + third + fourth ; first = second ; second = third ; third = fourth ; fourth = curr ; } cout << curr ; } }
int main ( ) { int n = 10 ; printTetra ( n ) ; return 0 ; }
int countWays ( int n ) { int res [ n + 1 ] ; res [ 0 ] = 1 ; res [ 1 ] = 1 ; res [ 2 ] = 2 ; for ( int i = 3 ; i <= n ; i ++ ) res [ i ] = res [ i - 1 ] + res [ i - 2 ] + res [ i - 3 ] ; return res [ n ] ; }
int main ( ) { int n = 4 ; cout << countWays ( n ) ; return 0 ; }
int maxTasks ( int high [ ] , int low [ ] , int n ) {
if ( n <= 0 ) return 0 ;
return max ( high [ n - 1 ] + maxTasks ( high , low , ( n - 2 ) ) , low [ n - 1 ] + maxTasks ( high , low , ( n - 1 ) ) ) ; }
int main ( ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; cout << maxTasks ( high , low , n ) ; return 0 ; }
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
int printTetraRec ( int n ) {
if ( n == 0 ) return 0 ;
if ( n == 1 n == 2 ) return 1 ;
if ( n == 3 ) return 2 ; else return printTetraRec ( n - 1 ) + printTetraRec ( n - 2 ) + printTetraRec ( n - 3 ) + printTetraRec ( n - 4 ) ; }
void printTetra ( int n ) { cout << printTetraRec ( n ) << " ▁ " ; }
int main ( ) { int n = 10 ; printTetra ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ? x : y ) ; }
int maxTasks ( int high [ ] , int low [ ] , int n ) {
int task_dp [ n + 1 ] ;
task_dp [ 0 ] = 0 ;
task_dp [ 1 ] = high [ 0 ] ;
for ( int i = 2 ; i <= n ; i ++ ) task_dp [ i ] = max ( high [ i - 1 ] + task_dp [ i - 2 ] , low [ i - 1 ] + task_dp [ i - 1 ] ) ; return task_dp [ n ] ; }
int main ( ) { int n = 5 ; int high [ ] = { 3 , 6 , 8 , 7 , 6 } ; int low [ ] = { 1 , 5 , 4 , 5 , 3 } ; cout << maxTasks ( high , low , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int PermutationCoeff ( int n , int k ) { int P = 1 ;
for ( int i = 0 ; i < k ; i ++ ) P *= ( n - i ) ; return P ; }
int main ( ) { int n = 10 , k = 2 ; cout << " Value ▁ of ▁ P ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << PermutationCoeff ( n , k ) ; return 0 ; }
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
int minOperations ( string str , int n ) {
int i , lastUpper = -1 , firstLower = -1 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { if ( isupper ( str [ i ] ) ) { lastUpper = i ; break ; } }
for ( i = 0 ; i < n ; i ++ ) { if ( islower ( str [ i ] ) ) { firstLower = i ; break ; } }
if ( lastUpper == -1 firstLower == -1 ) return 0 ;
int countUpper = 0 ; for ( i = firstLower ; i < n ; i ++ ) { if ( isupper ( str [ i ] ) ) { countUpper ++ ; } }
int countLower = 0 ; for ( i = 0 ; i < lastUpper ; i ++ ) { if ( islower ( str [ i ] ) ) { countLower ++ ; } }
return min ( countLower , countUpper ) ; }
int main ( ) { string str = " geEksFOrGEekS " ; int n = str . length ( ) ; cout << minOperations ( str , n ) << endl ; }
float rainDayProbability ( int a [ ] , int n ) { float count = 0 , m ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] == 1 ) count ++ ; }
m = count / n ; return m ; }
int main ( ) { int a [ ] = { 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << rainDayProbability ( a , n ) ; return 0 ; }
double Series ( int n ) { int i ; double sums = 0.0 , ser ; for ( i = 1 ; i <= n ; ++ i ) { ser = 1 / pow ( i , i ) ; sums += ser ; } return sums ; }
int main ( ) { int n = 3 ; double res = Series ( n ) ; cout << res ; return 0 ; }
int ternarySearch ( int l , int r , int key , int ar [ ] ) { if ( r >= l ) {
int mid1 = l + ( r - l ) / 3 ; int mid2 = r - ( r - l ) / 3 ;
if ( ar [ mid1 ] == key ) { return mid1 ; } if ( ar [ mid2 ] == key ) { return mid2 ; }
if ( key < ar [ mid1 ] ) {
return ternarySearch ( l , mid1 - 1 , key , ar ) ; } else if ( key > ar [ mid2 ] ) {
return ternarySearch ( mid2 + 1 , r , key , ar ) ; } else {
return ternarySearch ( mid1 + 1 , mid2 - 1 , key , ar ) ; } }
return -1 ; }
int ar [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 } ;
l = 0 ;
r = 9 ;
key = 5 ;
p = ternarySearch ( l , r , key , ar ) ;
cout << " Index ▁ of ▁ " << key << " ▁ is ▁ " << p << endl ;
key = 50 ;
p = ternarySearch ( l , r , key , ar ) ;
cout << " Index ▁ of ▁ " << key << " ▁ is ▁ " << p << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define SIZE  26
void printCharWithFreq ( string str ) {
int n = str . size ( ) ;
int freq [ SIZE ] ;
for ( int i = 0 ; i < n ; i ++ ) freq [ str [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; i < n ; i ++ ) {
if ( freq [ str [ i ] - ' a ' ] != 0 ) {
cout << str [ i ] << freq [ str [ i ] - ' a ' ] << " ▁ " ;
freq [ str [ i ] - ' a ' ] = 0 ; } } }
int main ( ) { string str = " geeksforgeeks " ; printCharWithFreq ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define MAX  1000 NEW_LINE using namespace std ; void checkHV ( int arr [ ] [ MAX ] , int N , int M ) {
bool horizontal = true , vertical = true ;
for ( int i = 0 , k = N - 1 ; i < N / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < M ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { horizontal = false ; break ; } } }
for ( int i = 0 , k = M - 1 ; i < M / 2 ; i ++ , k -- ) {
for ( int j = 0 ; j < N ; j ++ ) {
if ( arr [ i ] [ j ] != arr [ k ] [ j ] ) { vertical = false ; break ; } } } if ( ! horizontal && ! vertical ) cout << " NO STRNEWLINE " ; else if ( horizontal && ! vertical ) cout << " HORIZONTAL STRNEWLINE " ; else if ( vertical && ! horizontal ) cout << " VERTICAL STRNEWLINE " ; else cout << " BOTH STRNEWLINE " ; }
int main ( ) { int mat [ MAX ] [ MAX ] = { { 1 , 0 , 1 } , { 0 , 0 , 0 } , { 1 , 0 , 1 } } ; checkHV ( mat , 3 , 3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  4
void add ( int A [ ] [ N ] , int B [ ] [ N ] , int C [ ] [ N ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] + B [ i ] [ j ] ; }
int main ( ) { int A [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ N ] [ N ] ; int i , j ; add ( A , B , C ) ; cout << " Result ▁ matrix ▁ is ▁ " << endl ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) cout << C [ i ] [ j ] << " ▁ " ; cout << endl ; } return 0 ; }
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
#include <bits/stdc++.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ; #define RANGE  255
void countSort ( char arr [ ] ) {
char output [ strlen ( arr ) ] ;
int count [ RANGE + 1 ] , i ; memset ( count , 0 , sizeof ( count ) ) ;
for ( i = 0 ; arr [ i ] ; ++ i ) ++ count [ arr [ i ] ] ;
for ( i = 1 ; i <= RANGE ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( i = 0 ; arr [ i ] ; ++ i ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( i = 0 ; arr [ i ] ; ++ i ) arr [ i ] = output [ i ] ; }
int main ( ) { char arr [ ] = " geeksforgeeks " ; countSort ( arr ) ; cout << " Sorted ▁ character ▁ array ▁ is ▁ " << arr ; return 0 ; }
int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
int main ( ) { int n = 5 , k = 2 ; cout << " Value ▁ of ▁ C ( " << n << " , ▁ " << k << " ) ▁ is ▁ " << binomialCoeff ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int binomialCoeff ( int n , int k ) { int C [ k + 1 ] ; memset ( C , 0 , sizeof ( C ) ) ;
C [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) {
for ( int j = min ( i , k ) ; j > 0 ; j -- ) C [ j ] = C [ j ] + C [ j - 1 ] ; } return C [ k ] ; }
int main ( ) { int n = 5 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
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
public : int power ( int x , unsigned int y ) { if ( y == 0 ) return 1 ; else if ( y % 2 == 0 ) return power ( x , y / 2 ) * power ( x , y / 2 ) ; else return x * power ( x , y / 2 ) * power ( x , y / 2 ) ; } } ;
int main ( ) { gfg g ; int x = 2 ; unsigned int y = 3 ; cout << g . power ( x , y ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
int main ( ) { float x = 2 ; int y = -3 ; cout << power ( x , y ) ; return 0 ; }
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
int maxIndexDiff ( int arr [ ] , int n ) { int maxDiff = -1 ; int i , j ; for ( i = 0 ; i < n ; ++ i ) { for ( j = n - 1 ; j > i ; -- j ) { if ( arr [ j ] > arr [ i ] && maxDiff < ( j - i ) ) maxDiff = j - i ; } } return maxDiff ; }
int main ( ) { int arr [ ] = { 9 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 18 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int maxDiff = maxIndexDiff ( arr , n ) ; cout << " STRNEWLINE " << maxDiff ; return 0 ; }
int missingK ( int a [ ] , int k , int n ) { int difference = 0 , ans = 0 , count = k ; bool flag = 0 ;
for ( int i = 0 ; i < n - 1 ; i ++ ) { difference = 0 ;
if ( ( a [ i ] + 1 ) != a [ i + 1 ] ) {
difference += ( a [ i + 1 ] - a [ i ] ) - 1 ;
if ( difference >= count ) { ans = a [ i ] + count ; flag = 1 ; break ; } else count -= difference ; } }
if ( flag ) return ans ; else return -1 ; }
int a [ ] = { 1 , 5 , 11 , 19 } ;
int k = 11 ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ;
int missing = missingK ( a , k , n ) ; cout << missing << endl ; return 0 ; }
int findRotations ( string str ) {
string tmp = str + str ; int n = str . length ( ) ; for ( int i = 1 ; i <= n ; i ++ ) {
string substring = tmp . substr ( i , str . size ( ) ) ;
if ( str == substring ) return i ; } return n ; }
int main ( ) { string str = " abc " ; cout << findRotations ( str ) << endl ; return 0 ; }
int findKth ( int arr [ ] , int n , int k ) { unordered_set < int > missing ; int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) missing . insert ( arr [ i ] ) ;
int maxm = * max_element ( arr , arr + n ) ; int minm = * min_element ( arr , arr + n ) ;
for ( int i = minm + 1 ; i < maxm ; i ++ ) {
if ( missing . find ( i ) == missing . end ( ) ) count ++ ;
if ( count == k ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { 2 , 10 , 9 , 4 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int k = 5 ; cout << findKth ( arr , n , k ) ; return 0 ; }
int waysToKAdjacentSetBits ( int n , int k , int currentIndex , int adjacentSetBits , int lastBit ) {
if ( currentIndex == n ) {
if ( adjacentSetBits == k ) return 1 ; return 0 ; } int noOfWays = 0 ;
if ( lastBit == 1 ) {
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits + 1 , 1 ) ;
noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } else if ( ! lastBit ) { noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 1 ) ; noOfWays += waysToKAdjacentSetBits ( n , k , currentIndex + 1 , adjacentSetBits , 0 ) ; } return noOfWays ; }
int main ( ) { int n = 5 , k = 2 ;
int totalWays = waysToKAdjacentSetBits ( n , k , 1 , 0 , 1 ) + waysToKAdjacentSetBits ( n , k , 1 , 0 , 0 ) ; cout << " Number ▁ of ▁ ways ▁ = ▁ " << totalWays << " STRNEWLINE " ; return 0 ; }
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
#include <bits/stdc++.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ; int findRepeatFirstN2 ( char * s ) {
int p = -1 , i , j ; for ( i = 0 ; i < strlen ( s ) ; i ++ ) { for ( j = i + 1 ; j < strlen ( s ) ; j ++ ) { if ( s [ i ] == s [ j ] ) { p = i ; break ; } } if ( p != -1 ) break ; } return p ; }
int main ( ) { char str [ ] = " geeksforgeeks " ; int pos = findRepeatFirstN2 ( str ) ; if ( pos == -1 ) cout << " Not ▁ found " ; else cout << str [ pos ] ; return 0 ; }
int possibleStrings ( int n , int r , int b , int g ) {
int fact [ n + 1 ] ; fact [ 0 ] = 1 ; for ( int i = 1 ; i <= n ; i ++ ) fact [ i ] = fact [ i - 1 ] * i ;
int left = n - ( r + g + b ) ; int sum = 0 ;
for ( int i = 0 ; i <= left ; i ++ ) { for ( int j = 0 ; j <= left - i ; j ++ ) { int k = left - ( i + j ) ;
sum = sum + fact [ n ] / ( fact [ i + r ] * fact [ j + b ] * fact [ k + g ] ) ; } }
return sum ; }
int main ( ) { int n = 4 , r = 2 ; int b = 0 , g = 1 ; cout << possibleStrings ( n , r , b , g ) ; return 0 ; }
int remAnagram ( string str1 , string str2 ) {
int count1 [ CHARS ] = { 0 } , count2 [ CHARS ] = { 0 } ;
for ( int i = 0 ; str1 [ i ] != ' \0' ; i ++ ) count1 [ str1 [ i ] - ' a ' ] ++ ;
for ( int i = 0 ; str2 [ i ] != ' \0' ; i ++ ) count2 [ str2 [ i ] - ' a ' ] ++ ;
int result = 0 ; for ( int i = 0 ; i < 26 ; i ++ ) result += abs ( count1 [ i ] - count2 [ i ] ) ; return result ; }
int main ( ) { string str1 = " bcadeh " , str2 = " hea " ; cout << remAnagram ( str1 , str2 ) ; return 0 ; }
void printPath ( vector < int > res , int nThNode , int kThNode ) {
if ( kThNode > nThNode ) return ;
res . push_back ( kThNode ) ;
for ( int i = 0 ; i < res . size ( ) ; i ++ ) cout << res [ i ] << " ▁ " ; cout << " STRNEWLINE " ;
printPath ( res , nThNode , kThNode * 2 ) ;
printPath ( res , nThNode , kThNode * 2 + 1 ) ; }
void printPathToCoverAllNodeUtil ( int nThNode ) {
vector < int > res ;
printPath ( res , nThNode , 1 ) ; }
int nThNode = 7 ;
printPathToCoverAllNodeUtil ( nThNode ) ; return 0 ; }
void shortestLength ( int n , int x [ ] , int y [ ] ) { int answer = 0 ;
int i = 0 ; while ( n -- ) {
if ( x [ i ] + y [ i ] > answer ) answer = x [ i ] + y [ i ] ; i ++ ; }
cout << " Length ▁ - > ▁ " << answer << endl ; cout << " Path ▁ - > ▁ " << " ( ▁ 1 , ▁ " << answer << " ▁ ) " << " and ▁ ( ▁ " << answer << " , ▁ 1 ▁ ) " ; }
int n = 4 ;
int x [ n ] = { 1 , 4 , 2 , 1 } ; int y [ n ] = { 4 , 1 , 1 , 2 } ; shortestLength ( n , x , y ) ; return 0 ; }
void FindPoints ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 , int x4 , int y4 ) {
int x5 = max ( x1 , x3 ) ; int y5 = max ( y1 , y3 ) ;
int x6 = min ( x2 , x4 ) ; int y6 = min ( y2 , y4 ) ;
if ( x5 > x6 y5 > y6 ) { cout << " No ▁ intersection " ; return ; } cout << " ( " << x5 << " , ▁ " << y5 << " ) ▁ " ; cout << " ( " << x6 << " , ▁ " << y6 << " ) ▁ " ;
int x7 = x5 ; int y7 = y6 ; cout << " ( " << x7 << " , ▁ " << y7 << " ) ▁ " ;
int x8 = x6 ; int y8 = y5 ; cout << " ( " << x8 << " , ▁ " << y8 << " ) ▁ " ; }
int x1 = 0 , y1 = 0 , x2 = 10 , y2 = 8 ;
int x3 = 2 , y3 = 3 , x4 = 7 , y4 = 9 ;
FindPoints ( x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 ) ; return 0 ; }
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
bool isPalindrome ( int num ) { int reverse_num = 0 , remainder , temp ;
temp = num ; while ( temp != 0 ) { remainder = temp % 10 ; reverse_num = reverse_num * 10 + remainder ; temp /= 10 ; }
if ( reverse_num == num ) { return true ; } return false ; }
bool isOddLength ( int num ) { int count = 0 ; while ( num > 0 ) { num /= 10 ; count ++ ; } if ( count % 2 != 0 ) { return true ; } return false ; }
long sumOfAllPalindrome ( int L , int R ) { long sum = 0 ; if ( L <= R ) for ( int i = L ; i <= R ; i ++ ) {
if ( isPalindrome ( i ) && isOddLength ( i ) ) { sum += i ; } } return sum ; }
int main ( ) { int L = 110 , R = 1130 ; cout << " ▁ " << sumOfAllPalindrome ( L , R ) << endl ; }
int calculateAlternateSum ( int n ) { if ( n <= 0 ) return 0 ; int fibo [ n + 1 ] ; fibo [ 0 ] = 0 , fibo [ 1 ] = 1 ;
int sum = pow ( fibo [ 0 ] , 2 ) + pow ( fibo [ 1 ] , 2 ) ;
for ( int i = 2 ; i <= n ; i ++ ) { fibo [ i ] = fibo [ i - 1 ] + fibo [ i - 2 ] ;
if ( i % 2 == 0 ) sum -= fibo [ i ] ;
else sum += fibo [ i ] ; }
return sum ; }
int n = 8 ;
cout << " Alternating ▁ Fibonacci ▁ Sum ▁ upto ▁ " << n << " ▁ terms : ▁ " << calculateAlternateSum ( n ) << endl ; return 0 ; }
int getValue ( int n ) { int i = 0 , k = 1 ; while ( i < n ) { i = i + k ; k = k * 2 ; } return k / 2 ; }
int n = 9 ;
cout << getValue ( n ) << endl ;
n = 1025 ;
cout << getValue ( n ) << endl ; }
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
int solve ( int a [ ] , int n ) { int max1 = INT_MIN ; for ( int i = 0 ; i < n ; i ++ ) { for ( int j = 0 ; j < n ; j ++ ) { if ( abs ( a [ i ] - a [ j ] ) > max1 ) { max1 = abs ( a [ i ] - a [ j ] ) ; } } } return max1 ; }
int main ( ) { int arr [ ] = { -1 , 2 , 3 , -4 , -10 , 22 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Largest ▁ gap ▁ is ▁ : ▁ " << solve ( arr , size ) ; return 0 ; }
int solve ( int a [ ] , int n ) { int min1 = a [ 0 ] ; int max1 = a [ 0 ] ;
for ( int i = 0 ; i < n ; i ++ ) { if ( a [ i ] > max1 ) max1 = a [ i ] ; if ( a [ i ] < min1 ) min1 = a [ i ] ; } return abs ( min1 - max1 ) ; }
int main ( ) { int arr [ ] = { -1 , 2 , 3 , 4 , -10 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Largest ▁ gap ▁ is ▁ : ▁ " << solve ( arr , size ) ; return 0 ; }
int minElements ( int arr [ ] , int n ) {
int halfSum = 0 ; for ( int i = 0 ; i < n ; i ++ ) halfSum = halfSum + arr [ i ] ; halfSum = halfSum / 2 ;
sort ( arr , arr + n , greater < int > ( ) ) ; int res = 0 , curr_sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) { curr_sum += arr [ i ] ; res ++ ;
if ( curr_sum > halfSum ) return res ; } return res ; }
int main ( ) { int arr [ ] = { 3 , 1 , 7 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << minElements ( arr , n ) << endl ; return 0 ; }
int minCost ( int N , int P , int Q ) {
int cost = 0 ;
while ( N > 0 ) { if ( N & 1 ) { cost += P ; N -- ; } else { int temp = N / 2 ;
if ( temp * P > Q ) cost += Q ;
else cost += P * temp ; N /= 2 ; } }
return cost ; }
int main ( ) { int N = 9 , P = 5 , Q = 1 ; cout << minCost ( N , P , Q ) ; return 0 ; }
