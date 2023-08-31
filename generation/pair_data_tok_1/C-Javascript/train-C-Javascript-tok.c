void count_setbit ( int N ) {
int result = 0 ;
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( 1 << i ) & N ) {
result ++ ; } } printf ( " % d STRNEWLINE " , result ) ; }
int main ( ) { int N = 43 ; count_setbit ( N ) ; return 0 ; }
_Bool isPowerOfTwo ( int n ) { return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { int N = 8 ; if ( isPowerOfTwo ( N ) ) { printf ( " Yes " ) ; } else { printf ( " No " ) ; } }
typedef struct cantor { double start , end ; struct cantor * next ; } Cantor ;
Cantor * startList ( Cantor * head , double start_num , double end_num ) { if ( head == NULL ) { head = ( Cantor * ) malloc ( sizeof ( Cantor ) ) ; head -> start = start_num ; head -> end = end_num ; head -> next = NULL ; } return head ; }
Cantor * propagate ( Cantor * head ) { Cantor * temp = head ; if ( temp != NULL ) { Cantor * newNode = ( Cantor * ) malloc ( sizeof ( Cantor ) ) ; double diff = ( ( ( temp -> end ) - ( temp -> start ) ) / 3 ) ;
newNode -> end = temp -> end ; temp -> end = ( ( temp -> start ) + diff ) ; newNode -> start = ( newNode -> end ) - diff ;
newNode -> next = temp -> next ; temp -> next = newNode ;
propagate ( temp -> next -> next ) ; } return head ; }
void print ( Cantor * temp ) { while ( temp != NULL ) { printf ( " [ % lf ] ▁ - - ▁ [ % lf ] TABSYMBOL " , temp -> start , temp -> end ) ; temp = temp -> next ; } printf ( " STRNEWLINE " ) ; }
void buildCantorSet ( int A , int B , int L ) { Cantor * head = NULL ; head = startList ( head , A , B ) ; for ( int i = 0 ; i < L ; i ++ ) { printf ( " Level _ % d ▁ : ▁ " , i ) ; print ( head ) ; propagate ( head ) ; } printf ( " Level _ % d ▁ : ▁ " , L ) ; print ( head ) ; }
int main ( ) { int A = 0 ; int B = 9 ; int L = 2 ; buildCantorSet ( A , B , L ) ; return 0 ; }
void search ( char pat [ ] , char txt [ ] ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
{ printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { char txt [ ] = " ABCEABCDABCEABCD " ; char pat [ ] = " ABCD " ; search ( pat , txt ) ; return 0 ; }
void encrypt ( char input [ 100 ] ) {
char evenPos = ' @ ' , oddPos = ' ! ' ; int repeat , ascii ; for ( int i = 0 ; i <= strlen ( input ) ; i ++ ) {
ascii = input [ i ] ; repeat = ascii >= 97 ? ascii - 96 : ascii - 64 ; for ( int j = 0 ; j < repeat ; j ++ ) {
if ( i % 2 == 0 ) printf ( " % c " , oddPos ) ; else printf ( " % c " , evenPos ) ; } } }
void main ( ) { char input [ 100 ] = { ' A ' , ' b ' , ' C ' , ' d ' } ;
encrypt ( input ) ; }
bool isPalRec ( char str [ ] , int s , int e ) {
if ( s == e ) return true ;
if ( str [ s ] != str [ e ] ) return false ;
if ( s < e + 1 ) return isPalRec ( str , s + 1 , e - 1 ) ; return true ; } bool isPalindrome ( char str [ ] ) { int n = strlen ( str ) ;
if ( n == 0 ) return true ; return isPalRec ( str , 0 , n - 1 ) ; }
int main ( ) { char str [ ] = " geeg " ; if ( isPalindrome ( str ) ) printf ( " Yes " ) ; else printf ( " No " ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE int myAtoi ( const char * str ) { int sign = 1 , base = 0 , i = 0 ;
while ( str [ i ] == ' ▁ ' ) { i ++ ; }
if ( str [ i ] == ' - ' str [ i ] == ' + ' ) { sign = 1 - 2 * ( str [ i ++ ] == ' - ' ) ; }
while ( str [ i ] >= '0' && str [ i ] <= '9' ) {
if ( base > INT_MAX / 10 || ( base == INT_MAX / 10 && str [ i ] - '0' > 7 ) ) { if ( sign == 1 ) return INT_MAX ; else return INT_MIN ; } base = 10 * base + ( str [ i ++ ] - '0' ) ; } return base * sign ; }
int main ( ) { char str [ ] = " ▁ - 123" ;
int val = myAtoi ( str ) ; printf ( " % d ▁ " , val ) ; return 0 ; }
bool fillUtil ( int res [ ] , int curr , int n ) {
if ( curr == 0 ) return true ;
int i ; for ( i = 0 ; i < 2 * n - curr - 1 ; i ++ ) {
if ( res [ i ] == 0 && res [ i + curr + 1 ] == 0 ) {
res [ i ] = res [ i + curr + 1 ] = curr ;
if ( fillUtil ( res , curr - 1 , n ) ) return true ;
res [ i ] = res [ i + curr + 1 ] = 0 ; } } return false ; }
void fill ( int n ) {
int res [ 2 * n ] , i ; for ( i = 0 ; i < 2 * n ; i ++ ) res [ i ] = 0 ;
if ( fillUtil ( res , n , n ) ) { for ( i = 0 ; i < 2 * n ; i ++ ) printf ( " % d ▁ " , res [ i ] ) ; } else puts ( " Not ▁ Possible " ) ; }
int main ( ) { fill ( 7 ) ; return 0 ; }
int findNumberOfDigits ( int n , int base ) {
int dig = ( floor ( log ( n ) / log ( base ) ) + 1 ) ;
return ( dig ) ; }
int isAllKs ( int n , int b , int k ) { int len = findNumberOfDigits ( n , b ) ;
int sum = k * ( 1 - pow ( b , len ) ) / ( 1 - b ) ; if ( sum == n ) { return ( sum ) ; } }
int N = 13 ;
int B = 3 ;
int K = 1 ;
if ( isAllKs ( N , B , K ) ) { printf ( " Yes " ) ; } else { printf ( " No " ) ; } return 0 ; }
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
printArray ( S , n ) ; return 0 ; }
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
#define maxsize  10005
int xortree [ maxsize ] ;
void construct_Xor_Tree_Util ( int current [ ] , int start , int end , int x ) {
if ( start == end ) { xortree [ x ] = current [ start ] ;
return ; }
int left = x * 2 + 1 ;
int right = x * 2 + 2 ;
int mid = start + ( end - start ) / 2 ;
construct_Xor_Tree_Util ( current , start , mid , left ) ; construct_Xor_Tree_Util ( current , mid + 1 , end , right ) ;
xortree [ x ] = ( xortree [ left ] ^ xortree [ right ] ) ; }
void construct_Xor_Tree ( int arr [ ] , int n ) { int i = 0 ; for ( i = 0 ; i < maxsize ; i ++ ) xortree [ i ] = 0 ; construct_Xor_Tree_Util ( arr , 0 , n - 1 , 0 ) ; }
int leaf_nodes [ ] = { 40 , 32 , 12 , 1 , 4 , 3 , 2 , 7 } , i = 0 ; int n = sizeof ( leaf_nodes ) / sizeof ( leaf_nodes [ 0 ] ) ;
construct_Xor_Tree ( leaf_nodes , n ) ;
int x = ( int ) ( ceil ( log2 ( n ) ) ) ;
int max_size = 2 * ( int ) pow ( 2 , x ) - 1 ; printf ( " Nodes ▁ of ▁ the ▁ XOR ▁ tree STRNEWLINE " ) ; for ( i = 0 ; i < max_size ; i ++ ) { printf ( " % d ▁ " , xortree [ i ] ) ; }
int root = 0 ;
printf ( " Root : % d " }
#include <stdio.h> NEW_LINE int swapBits ( int n , int p1 , int p2 ) {
n ^= 1 << p1 ; n ^= 1 << p2 ; return n ; }
int main ( ) { printf ( " Result ▁ = ▁ % d " , swapBits ( 28 , 0 , 3 ) ) ; return 0 ; }
struct Node { int key ; struct Node * left , * right ; } ;
bool isFullTree ( struct Node * root ) {
if ( root == NULL ) return true ;
if ( root -> left == NULL && root -> right == NULL ) return true ;
if ( ( root -> left ) && ( root -> right ) ) return ( isFullTree ( root -> left ) && isFullTree ( root -> right ) ) ;
return false ; }
int main ( ) { struct Node * root = NULL ; root = newNode ( 10 ) ; root -> left = newNode ( 20 ) ; root -> right = newNode ( 30 ) ; root -> left -> right = newNode ( 40 ) ; root -> left -> left = newNode ( 50 ) ; root -> right -> left = newNode ( 60 ) ; root -> right -> right = newNode ( 70 ) ; root -> left -> left -> left = newNode ( 80 ) ; root -> left -> left -> right = newNode ( 90 ) ; root -> left -> right -> left = newNode ( 80 ) ; root -> left -> right -> right = newNode ( 90 ) ; root -> right -> left -> left = newNode ( 80 ) ; root -> right -> left -> right = newNode ( 90 ) ; root -> right -> right -> left = newNode ( 80 ) ; root -> right -> right -> right = newNode ( 90 ) ; if ( isFullTree ( root ) ) printf ( " The ▁ Binary ▁ Tree ▁ is ▁ full STRNEWLINE " ) ; else printf ( " The ▁ Binary ▁ Tree ▁ is ▁ not ▁ full STRNEWLINE " ) ; return ( 0 ) ; }
void printAlter ( int arr [ ] , int N ) {
for ( int currIndex = 0 ; currIndex < N ; currIndex += 2 ) {
printf ( " % d ▁ " , arr [ currIndex ] ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printAlter ( arr , N ) ; }
void printArray ( int arr [ ] , int size ) ; void swap ( int arr [ ] , int fi , int si , int d ) ; void leftRotate ( int arr [ ] , int d , int n ) {
if ( d == 0 d == n ) return ;
if ( n - d == d ) { swap ( arr , 0 , n - d , d ) ; return ; }
if ( d < n - d ) { swap ( arr , 0 , n - d , d ) ; leftRotate ( arr , d , n - d ) ; }
else { swap ( arr , 0 , d , n - d ) ; leftRotate ( arr + n - d , 2 * d - n , d ) ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE ▁ " ) ; }
void swap ( int arr [ ] , int fi , int si , int d ) { int i , temp ; for ( i = 0 ; i < d ; i ++ ) { temp = arr [ fi + i ] ; arr [ fi + i ] = arr [ si + i ] ; arr [ si + i ] = temp ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ; getchar ( ) ; return 0 ; }
void leftRotate ( int arr [ ] , int d , int n ) { int i , j ; if ( d == 0 d == n ) return ; i = d ; j = n - d ; while ( i != j ) {
if ( i < j ) { swap ( arr , d - i , d + j - i , i ) ; j -= i ; }
else { swap ( arr , d - i , d , j ) ; i -= j ; } }
swap ( arr , d - i , d , i ) ; }
void selectionSort ( char arr [ ] [ MAX_LEN ] , int n ) { int i , j , min_idx ;
char minStr [ MAX_LEN ] ; for ( i = 0 ; i < n - 1 ; i ++ ) {
int min_idx = i ; strcpy ( minStr , arr [ i ] ) ; for ( j = i + 1 ; j < n ; j ++ ) {
if ( strcmp ( minStr , arr [ j ] ) > 0 ) {
strcpy ( minStr , arr [ j ] ) ; min_idx = j ; } }
if ( min_idx != i ) { char temp [ MAX_LEN ] ; strcpy ( temp , arr [ i ] ) ; strcpy ( arr [ i ] , arr [ min_idx ] ) ; strcpy ( arr [ min_idx ] , temp ) ; } } }
int main ( ) { char arr [ ] [ MAX_LEN ] = { " GeeksforGeeks " , " Practice . GeeksforGeeks " , " GeeksQuiz " } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int i ; printf ( " Given ▁ array ▁ is STRNEWLINE " ) ;
for ( i = 0 ; i < n ; i ++ ) printf ( " % d : ▁ % s ▁ STRNEWLINE " , i , arr [ i ] ) ; selectionSort ( arr , n ) ; printf ( " Sorted array is "
for ( i = 0 ; i < n ; i ++ ) printf ( " % d : ▁ % s ▁ STRNEWLINE " , i , arr [ i ] ) ; return 0 ; }
void rearrangeNaive ( int arr [ ] , int n ) {
int temp [ n ] , i ;
for ( i = 0 ; i < n ; i ++ ) temp [ arr [ i ] ] = i ;
for ( i = 0 ; i < n ; i ++ ) arr [ i ] = temp [ i ] ; }
void printArray ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 0 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Given ▁ array ▁ is ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; rearrangeNaive ( arr , n ) ; printf ( " Modified ▁ array ▁ is ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; return 0 ; }
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
struct node { int data ; struct node * left ; struct node * right ; } ;
void printPostorder ( struct node * node ) { if ( node == NULL ) return ;
printPostorder ( node -> left ) ;
printPostorder ( node -> right ) ;
printf ( " % d ▁ " , node -> data ) ; }
void printInorder ( struct node * node ) { if ( node == NULL ) return ;
printInorder ( node -> left ) ;
printf ( " % d ▁ " , node -> data ) ;
printInorder ( node -> right ) ; }
void printPreorder ( struct node * node ) { if ( node == NULL ) return ;
printf ( " % d ▁ " , node -> data ) ;
printPreorder ( node -> left ) ;
printPreorder ( node -> right ) ; }
int main ( ) { struct node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; printf ( " Preorder traversal of binary tree is " printPreorder ( root ) ; printf ( " Inorder traversal of binary tree is " printInorder ( root ) ; printf ( " Postorder traversal of binary tree is " printPostorder ( root ) ; getchar ( ) ; return 0 ; }
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
int max ( int num1 , int num2 ) { return ( num1 > num2 ) ? num1 : num2 ; }
int min ( int num1 , int num2 ) { return ( num1 > num2 ) ? num2 : num1 ; }
int getCount ( int n , int k ) {
if ( n == 1 ) return 10 ;
int dp [ 11 ] = { 0 } ;
int next [ 11 ] = { 0 } ;
for ( int i = 1 ; i <= 9 ; i ++ ) dp [ i ] = 1 ;
for ( int i = 2 ; i <= n ; i ++ ) { for ( int j = 0 ; j <= 9 ; j ++ ) {
int l = max ( 0 , ( j - k ) ) ; int r = min ( 9 , ( j + k ) ) ;
next [ l ] += dp [ j ] ; next [ r + 1 ] -= dp [ j ] ; }
for ( int j = 1 ; j <= 9 ; j ++ ) next [ j ] += next [ j - 1 ] ;
for ( int j = 0 ; j < 10 ; j ++ ) { dp [ j ] = next [ j ] ; next [ j ] = 0 ; } }
int count = 0 ; for ( int i = 0 ; i <= 9 ; i ++ ) count += dp [ i ] ;
return count ; }
int main ( ) { int n = 2 , k = 1 ; printf ( " % d " , getCount ( n , k ) ) ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE int getInvCount ( int arr [ ] , int n ) { int inv_count = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ i ] > arr [ j ] ) inv_count ++ ; return inv_count ; }
int main ( ) { int arr [ ] = { 1 , 20 , 6 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " ▁ Number ▁ of ▁ inversions ▁ are ▁ % d ▁ STRNEWLINE " , getInvCount ( arr , n ) ) ; return 0 ; }
# include <stdio.h> NEW_LINE # include <stdlib.h> NEW_LINE # include <math.h> NEW_LINE void minAbsSumPair ( int arr [ ] , int arr_size ) { int inv_count = 0 ; int l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { printf ( " Invalid ▁ Input " ) ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( abs ( min_sum ) > abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } printf ( " ▁ The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are ▁ % d ▁ and ▁ % d " , arr [ min_l ] , arr [ min_r ] ) ; }
int main ( ) { int arr [ ] = { 1 , 60 , -10 , 70 , -80 , 85 } ; minAbsSumPair ( arr , 6 ) ; getchar ( ) ; return 0 ; }
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
int isPairSum ( int A [ ] , int N , int X ) {
int i = 0 ;
int j = N - 1 ; while ( i < j ) {
if ( A [ i ] + A [ j ] == X ) return 1 ;
else if ( A [ i ] + A [ j ] < X ) i ++ ;
else j -- ; } return 0 ; }
int arr [ ] = { 3 , 5 , 9 , 2 , 8 , 10 , 11 } ;
int val = 17 ;
int arrSize = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
printf ( " % d " , isPairSum ( arr , arrSize , val ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define NUM_LINE  2 NEW_LINE #define NUM_STATION  4
int min ( int a , int b ) { return a < b ? a : b ; } int carAssembly ( int a [ ] [ NUM_STATION ] , int t [ ] [ NUM_STATION ] , int * e , int * x ) { int T1 [ NUM_STATION ] , T2 [ NUM_STATION ] , i ;
T1 [ 0 ] = e [ 0 ] + a [ 0 ] [ 0 ] ;
T2 [ 0 ] = e [ 1 ] + a [ 1 ] [ 0 ] ;
for ( i = 1 ; i < NUM_STATION ; ++ i ) { T1 [ i ] = min ( T1 [ i - 1 ] + a [ 0 ] [ i ] , T2 [ i - 1 ] + t [ 1 ] [ i ] + a [ 0 ] [ i ] ) ; T2 [ i ] = min ( T2 [ i - 1 ] + a [ 1 ] [ i ] , T1 [ i - 1 ] + t [ 0 ] [ i ] + a [ 1 ] [ i ] ) ; }
return min ( T1 [ NUM_STATION - 1 ] + x [ 0 ] , T2 [ NUM_STATION - 1 ] + x [ 1 ] ) ; }
int main ( ) { int a [ ] [ NUM_STATION ] = { { 4 , 5 , 3 , 2 } , { 2 , 10 , 1 , 4 } } ; int t [ ] [ NUM_STATION ] = { { 0 , 7 , 4 , 5 } , { 0 , 9 , 2 , 8 } } ; int e [ ] = { 10 , 12 } , x [ ] = { 18 , 7 } ; printf ( " % d " , carAssembly ( a , t , e , x ) ) ; return 0 ; }
int findMinInsertionsDP ( char str [ ] , int n ) {
int table [ n ] [ n ] , l , h , gap ; memset ( table , 0 , sizeof ( table ) ) ;
for ( gap = 1 ; gap < n ; ++ gap ) for ( l = 0 , h = gap ; h < n ; ++ l , ++ h ) table [ l ] [ h ] = ( str [ l ] == str [ h ] ) ? table [ l + 1 ] [ h - 1 ] : ( min ( table [ l ] [ h - 1 ] , table [ l + 1 ] [ h ] ) + 1 ) ;
return table [ 0 ] [ n - 1 ] ; }
int main ( ) { char str [ ] = " geeks " ; printf ( " % d " , findMinInsertionsDP ( str , strlen ( str ) ) ) ; return 0 ; }
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
struct node { int data ; struct node * left , * right ; } ;
int LISS ( struct node * root ) { if ( root == NULL ) return 0 ;
int size_excl = LISS ( root -> left ) + LISS ( root -> right ) ;
int size_incl = 1 ; if ( root -> left ) size_incl += LISS ( root -> left -> left ) + LISS ( root -> left -> right ) ; if ( root -> right ) size_incl += LISS ( root -> right -> left ) + LISS ( root -> right -> right ) ;
return max ( size_incl , size_excl ) ; }
struct node * root = newNode ( 20 ) ; root -> left = newNode ( 8 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 12 ) ; root -> left -> right -> left = newNode ( 10 ) ; root -> left -> right -> right = newNode ( 14 ) ; root -> right = newNode ( 22 ) ; root -> right -> right = newNode ( 25 ) ; printf ( " Size ▁ of ▁ the ▁ Largest ▁ Independent ▁ Set ▁ is ▁ % d ▁ " , LISS ( root ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE struct pair { int a ; int b ; } ;
int maxChainLength ( struct pair arr [ ] , int n ) { int i , j , max = 0 ; int * mcl = ( int * ) malloc ( sizeof ( int ) * n ) ;
for ( i = 0 ; i < n ; i ++ ) mcl [ i ] = 1 ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] . a > arr [ j ] . b && mcl [ i ] < mcl [ j ] + 1 ) mcl [ i ] = mcl [ j ] + 1 ;
for ( i = 0 ; i < n ; i ++ ) if ( max < mcl [ i ] ) max = mcl [ i ] ; free ( mcl ) ; return max ; }
int main ( ) { struct pair arr [ ] = { { 5 , 24 } , { 15 , 25 } , { 27 , 40 } , { 50 , 60 } } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Length ▁ of ▁ maximum ▁ size ▁ chain ▁ is ▁ % d STRNEWLINE " , maxChainLength ( arr , n ) ) ; return 0 ; }
# include <limits.h> NEW_LINE # include <string.h> NEW_LINE # include <stdio.h> NEW_LINE # define NO_OF_CHARS  256
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
void badCharHeuristic ( char * str , int size , int badchar [ NO_OF_CHARS ] ) { int i ;
for ( i = 0 ; i < NO_OF_CHARS ; i ++ ) badchar [ i ] = -1 ;
for ( i = 0 ; i < size ; i ++ ) badchar [ ( int ) str [ i ] ] = i ; }
void search ( char * txt , char * pat ) { int m = strlen ( pat ) ; int n = strlen ( txt ) ; int badchar [ NO_OF_CHARS ] ;
badCharHeuristic ( pat , m , badchar ) ;
int s = 0 ; while ( s <= ( n - m ) ) { int j = m - 1 ;
while ( j >= 0 && pat [ j ] == txt [ s + j ] ) j -- ;
if ( j < 0 ) { printf ( " pattern occurs at shift = % d "
s += ( s + m < n ) ? m - badchar [ txt [ s + m ] ] : 1 ; } else
s += max ( 1 , j - badchar [ txt [ s + j ] ] ) ; } }
int main ( ) { char txt [ ] = " ABAAABCD " ; char pat [ ] = " ABC " ; search ( txt , pat ) ; return 0 ; }
int minChocolates ( int a [ ] , int n ) { int i = 0 , j = 0 ; int res = 0 , val = 1 ; while ( j < n - 1 ) { if ( a [ j ] > a [ j + 1 ] ) {
j += 1 ; continue ; } if ( i == j )
res += val ; else {
res += get_sum ( val , i , j ) ;
} if ( a [ j ] < a [ j + 1 ] )
val += 1 ; else
val = 1 ; j += 1 ; i = j ; }
if ( i == j ) res += val ; else res += get_sum ( val , i , j ) ; return res ; }
int get_sum ( int peak , int start , int end ) {
int count = end - start + 1 ;
peak = ( peak > count ) ? peak : count ;
int s = peak + ( ( ( count - 1 ) * count ) >> 1 ) ; return s ; }
int main ( ) { int a [ ] = { 5 , 5 , 4 , 3 , 2 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; printf ( " Minimum ▁ number ▁ of ▁ chocolates ▁ = ▁ % d " , minChocolates ( a , n ) ) ; return 0 ; }
double sum ( int n ) { double i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
int main ( ) { int n = 5 ; printf ( " Sum ▁ is ▁ % f " , sum ( n ) ) ; return 0 ; }
int nthTermOfTheSeries ( int n ) {
int nthTerm ;
if ( n % 2 == 0 ) nthTerm = pow ( n - 1 , 2 ) + n ;
else nthTerm = pow ( n + 1 , 2 ) + n ;
return nthTerm ; }
int main ( ) { int n ; n = 8 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 12 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 102 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 999 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; n = 9999 ; printf ( " % d STRNEWLINE " , nthTermOfTheSeries ( n ) ) ; return 0 ; }
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
#define ll  long long NEW_LINE ll multiply ( ll a , ll b , ll mod ) { return ( ( a % mod ) * ( b % mod ) ) % mod ; }
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
#include <stdio.h> NEW_LINE #define INT_MAX  2147483647
int isKthBitSet ( int x , int k ) { return ( x & ( 1 << ( k - 1 ) ) ) ? 1 : 0 ; }
int leftmostSetBit ( int x ) { int count = 0 ; while ( x ) { count ++ ; x = x >> 1 ; } return count ; }
int isBinPalindrome ( int x ) { int l = leftmostSetBit ( x ) ; int r = 1 ;
while ( l > r ) {
if ( isKthBitSet ( x , l ) != isKthBitSet ( x , r ) ) return 0 ; l -- ; r ++ ; } return 1 ; } int findNthPalindrome ( int n ) { int pal_count = 0 ;
int i = 0 ; for ( i = 1 ; i <= INT_MAX ; i ++ ) { if ( isBinPalindrome ( i ) ) { pal_count ++ ; }
if ( pal_count == n ) break ; } return i ; }
int main ( ) { int n = 9 ;
printf ( " % d " , findNthPalindrome ( n ) ) ; }
double temp_convert ( int F1 , int B1 , int F2 , int B2 , int T ) { float t2 ;
t2 = F2 + ( float ) ( B2 - F2 ) / ( B1 - F1 ) * ( T - F1 ) ; return t2 ; }
int main ( ) { int F1 = 0 , B1 = 100 ; int F2 = 32 , B2 = 212 ; int T = 37 ; float t2 ; printf ( " % .2f " , temp_convert ( F1 , B1 , F2 , B2 , T ) ) ; return 0 ; }
void findpath ( int N , int a [ ] ) {
if ( a [ 0 ] ) {
printf ( " % d ▁ " , N + 1 ) ; for ( int i = 1 ; i <= N ; i ++ ) printf ( " % d ▁ " , i ) ; return ; }
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( ! a [ i ] && a [ i + 1 ] ) {
for ( int j = 1 ; j <= i ; j ++ ) printf ( " % d ▁ " , j ) ; printf ( " % d ▁ " , N + 1 ) ; for ( int j = i + 1 ; j <= N ; j ++ ) printf ( " % d ▁ " , j ) ; return ; } }
for ( int i = 1 ; i <= N ; i ++ ) printf ( " % d ▁ " , i ) ; printf ( " % d ▁ " , N + 1 ) ; }
int N = 3 , arr [ ] = { 0 , 1 , 0 } ;
findpath ( N , arr ) ; }
void printArr ( int arr [ ] , int n ) {
qsort ( arr , n , sizeof ( int ) , compare ) ;
if ( arr [ 0 ] == arr [ n - 1 ] ) { printf ( " No STRNEWLINE " ) ; }
else { printf ( " Yes STRNEWLINE " ) ; for ( int i = 0 ; i < n ; i ++ ) { printf ( " % d ▁ " , arr [ i ] ) ; } } }
int arr [ ] = { 1 , 2 , 2 , 1 , 3 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
printArr ( arr , N ) ; return 0 ; }
int coins [ COINS ] = { 1 , 2 , 5 , 10 , 20 , 50 , 100 , 200 , 2000 } ; void findMin ( int cost ) {
int coinList [ MAX ] = { 0 } ;
int i , k = 0 ; for ( i = COINS - 1 ; i >= 0 ; i -- ) {
while ( cost >= coins [ i ] ) { cost -= coins [ i ] ; coinList [ k ++ ] = coins [ i ] ; } }
for ( i = 0 ; i < k ; i ++ ) { printf ( " % d ▁ " , coinList [ i ] ) ; } return ; }
int main ( void ) { int n = 93 ; printf ( " Following ▁ is ▁ minimal ▁ number " " of ▁ change ▁ for ▁ % d : ▁ " , n ) ; findMin ( n ) ; return 0 ; }
int findMinInsertions ( char str [ ] , int l , int h ) {
if ( l > h ) return INT_MAX ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) ; }
int main ( ) { char str [ ] = " geeks " ; printf ( " % d " , findMinInsertions ( str , 0 , strlen ( str ) - 1 ) ) ; return 0 ; }
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h> NEW_LINE #define NO_OF_CHARS  256 NEW_LINE int getNextState ( char * pat , int M , int state , int x ) {
if ( state < M && x == pat [ state ] ) return state + 1 ;
int ns , i ;
for ( ns = state ; ns > 0 ; ns -- ) { if ( pat [ ns - 1 ] == x ) { for ( i = 0 ; i < ns - 1 ; i ++ ) if ( pat [ i ] != pat [ state - ns + 1 + i ] ) break ; if ( i == ns - 1 ) return ns ; } } return 0 ; }
void computeTF ( char * pat , int M , int TF [ ] [ NO_OF_CHARS ] ) { int state , x ; for ( state = 0 ; state <= M ; ++ state ) for ( x = 0 ; x < NO_OF_CHARS ; ++ x ) TF [ state ] [ x ] = getNextState ( pat , M , state , x ) ; }
void search ( char * pat , char * txt ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int TF [ M + 1 ] [ NO_OF_CHARS ] ; computeTF ( pat , M , TF ) ;
int i , state = 0 ; for ( i = 0 ; i < N ; i ++ ) { state = TF [ state ] [ txt [ i ] ] ; if ( state == M ) printf ( " Pattern found at index % d " , i - M + 1 ) ; } }
int main ( ) { char * txt = " AABAACAADAABAAABAA " ; char * pat = " AABA " ; search ( pat , txt ) ; return 0 ; }
struct node { int data ; struct node * left , * right ; } ;
void printInoder ( struct node * root ) { if ( root != NULL ) { printInoder ( root -> left ) ; printf ( " % d ▁ " , root -> data ) ; printInoder ( root -> right ) ; } }
struct node * RemoveHalfNodes ( struct node * root ) { if ( root == NULL ) return NULL ; root -> left = RemoveHalfNodes ( root -> left ) ; root -> right = RemoveHalfNodes ( root -> right ) ; if ( root -> left == NULL && root -> right == NULL ) return root ;
if ( root -> left == NULL ) { struct node * new_root = root -> right ; free ( root ) ; return new_root ; }
if ( root -> right == NULL ) { struct node * new_root = root -> left ; free ( root ) ; return new_root ; } return root ; }
int main ( void ) { struct node * NewRoot = NULL ; struct node * root = newNode ( 2 ) ; root -> left = newNode ( 7 ) ; root -> right = newNode ( 5 ) ; root -> left -> right = newNode ( 6 ) ; root -> left -> right -> left = newNode ( 1 ) ; root -> left -> right -> right = newNode ( 11 ) ; root -> right -> right = newNode ( 9 ) ; root -> right -> right -> left = newNode ( 4 ) ; printf ( " Inorder ▁ traversal ▁ of ▁ given ▁ tree ▁ STRNEWLINE " ) ; printInoder ( root ) ; NewRoot = RemoveHalfNodes ( root ) ; printf ( " Inorder traversal of the modified tree " printInoder ( NewRoot ) ; return 0 ; }
#include <stdio.h> NEW_LINE void printSubstrings ( char str [ ] ) {
for ( int start = 0 ; str [ start ] != ' \0' ; start ++ ) {
for ( int end = start ; str [ end ] != ' \0' ; end ++ ) {
for ( int i = start ; i <= end ; i ++ ) { printf ( " % c " , str [ i ] ) ; }
printf ( " STRNEWLINE " ) ; } } }
int main ( ) {
char str [ ] = { ' a ' , ' b ' , ' c ' , ' d ' , ' \0' } ; printSubstrings ( str ) ; return 0 ; }
#define N  9
void print ( int arr [ N ] [ N ] ) { for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) printf ( " % d ▁ " , arr [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } }
int isSafe ( int grid [ N ] [ N ] , int row , int col , int num ) {
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ row ] [ x ] == num ) return 0 ;
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ x ] [ col ] == num ) return 0 ;
int startRow = row - row % 3 , startCol = col - col % 3 ; for ( int i = 0 ; i < 3 ; i ++ ) for ( int j = 0 ; j < 3 ; j ++ ) if ( grid [ i + startRow ] [ j + startCol ] == num ) return 0 ; return 1 ; }
int solveSuduko ( int grid [ N ] [ N ] , int row , int col ) {
if ( row == N - 1 && col == N ) return 1 ;
if ( col == N ) { row ++ ; col = 0 ; }
if ( grid [ row ] [ col ] > 0 ) return solveSuduko ( grid , row , col + 1 ) ; for ( int num = 1 ; num <= N ; num ++ ) {
if ( isSafe ( grid , row , col , num ) == 1 ) {
grid [ row ] [ col ] = num ;
if ( solveSuduko ( grid , row , col + 1 ) == 1 ) return 1 ; }
grid [ row ] [ col ] = 0 ; } return 0 ; } int main ( ) {
int grid [ N ] [ N ] = { { 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 } , { 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 } , { 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 } , { 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 } , { 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 } , { 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 } , { 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 } , { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 } , { 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 } } ; if ( solveSuduko ( grid , 0 , 0 ) == 1 ) print ( grid ) ; else printf ( " No ▁ solution ▁ exists " ) ; return 0 ; }
void printPairs ( int arr [ ] , int arr_size , int sum ) { int i , temp ; bool s [ MAX ] = { 0 } ; for ( i = 0 ; i < arr_size ; i ++ ) { temp = sum - arr [ i ] ;
if ( s [ temp ] == 1 ) printf ( " Pair ▁ with ▁ given ▁ sum ▁ % d ▁ is ▁ ( % d , ▁ % d ) ▁ n " , sum , arr [ i ] , temp ) ; s [ arr [ i ] ] = 1 ; } }
int main ( ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int n = 16 ; int arr_size = sizeof ( A ) / sizeof ( A [ 0 ] ) ; printPairs ( A , arr_size , n ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int exponentMod ( int A , int B , int C ) {
if ( A == 0 ) return 0 ; if ( B == 0 ) return 1 ;
long y ; if ( B % 2 == 0 ) { y = exponentMod ( A , B / 2 , C ) ; y = ( y * y ) % C ; }
else { y = A % C ; y = ( y * exponentMod ( A , B - 1 , C ) % C ) % C ; } return ( int ) ( ( y + C ) % C ) ; }
int main ( ) { int A = 2 , B = 5 , C = 13 ; printf ( " Power ▁ is ▁ % d " , exponentMod ( A , B , C ) ) ; return 0 ; }
int power ( int x , unsigned int y ) {
int res = 1 ; while ( y > 0 ) {
if ( y & 1 ) res = res * x ;
y = y >> 1 ;
x = x * x ; } return res ; }
int eggDrop ( int n , int k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; int min = INT_MAX , x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
int main ( ) { int n = 2 , k = 10 ; printf ( " nMinimum ▁ number ▁ of ▁ trials ▁ in ▁ " " worst ▁ case ▁ with ▁ % d ▁ eggs ▁ and ▁ " " % d ▁ floors ▁ is ▁ % d ▁ STRNEWLINE " , n , k , eggDrop ( n , k ) ) ; return 0 ; }
struct Node { int data ; struct Node * left , * right ; } ;
struct Node * extractLeafList ( struct Node * root , struct Node * * head_ref ) { if ( root == NULL ) return NULL ; if ( root -> left == NULL && root -> right == NULL ) { root -> right = * head_ref ; if ( * head_ref != NULL ) ( * head_ref ) -> left = root ; return NULL ; } root -> right = extractLeafList ( root -> right , head_ref ) ; root -> left = extractLeafList ( root -> left , head_ref ) ; return root ; }
void print ( struct Node * root ) { if ( root != NULL ) { print ( root -> left ) ; printf ( " % d ▁ " , root -> data ) ; print ( root -> right ) ; } }
void printList ( struct Node * head ) { while ( head ) { printf ( " % d ▁ " , head -> data ) ; head = head -> right ; } }
int main ( ) { struct Node * head = NULL ; struct Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> right = newNode ( 6 ) ; root -> left -> left -> left = newNode ( 7 ) ; root -> left -> left -> right = newNode ( 8 ) ; root -> right -> right -> left = newNode ( 9 ) ; root -> right -> right -> right = newNode ( 10 ) ; printf ( " Inorder ▁ Trvaersal ▁ of ▁ given ▁ Tree ▁ is : STRNEWLINE " ) ; print ( root ) ; root = extractLeafList ( root , & head ) ; printf ( " Extracted Double Linked list is : " printList ( head ) ; printf ( " Inorder traversal of modified tree is : " print ( root ) ; return 0 ; }
long long int countNumberOfStrings ( char * s ) {
int n = length - 1 ;
long long int count = pow ( 2 , n ) ; return count ; }
int main ( ) { char S [ ] = " ABCD " ; printf ( " % lld " , countNumberOfStrings ( S ) ) ; return 0 ; }
void makeArraySumEqual ( int a [ ] , int N ) {
int count_0 = 0 , count_1 = 0 ;
int odd_sum = 0 , even_sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( a [ i ] == 0 ) count_0 ++ ;
else count_1 ++ ;
if ( ( i + 1 ) % 2 == 0 ) even_sum += a [ i ] ; else if ( ( i + 1 ) % 2 > 0 ) odd_sum += a [ i ] ; }
if ( odd_sum == even_sum ) {
for ( int i = 0 ; i < N ; i ++ ) printf ( " % d ▁ " , a [ i ] ) ; }
else { if ( count_0 >= N / 2 ) {
for ( int i = 0 ; i < count_0 ; i ++ ) printf ( "0 ▁ " ) ; } else {
int is_Odd = count_1 % 2 ;
count_1 -= is_Odd ;
for ( int i = 0 ; i < count_1 ; i ++ ) printf ( "1 ▁ " ) ; } } }
int arr [ ] = { 1 , 1 , 1 , 0 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
makeArraySumEqual ( arr , N ) ; return 0 ; }
int countDigitSum ( int N , int K ) {
int l = ( int ) pow ( 10 , N - 1 ) , r = ( int ) pow ( 10 , N ) - 1 ; int count = 0 ; for ( int i = l ; i <= r ; i ++ ) { int num = i ;
int digits [ N ] ; for ( int j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num /= 10 ; } int sum = 0 , flag = 0 ;
for ( int j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( int j = K ; j < N ; j ++ ) { if ( sum - digits [ j - K ] + digits [ j ] != sum ) { flag = 1 ; break ; } } if ( flag == 0 ) count ++ ; } return count ; }
int N = 2 , K = 1 ; printf ( " % d " , countDigitSum ( N , K ) ) ; return 0 ; }
void findpath ( int N , int a [ ] ) {
if ( a [ 0 ] ) {
printf ( " % d ▁ " , N + 1 ) ; for ( int i = 1 ; i <= N ; i ++ ) printf ( " % d ▁ " , i ) ; return ; }
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( ! a [ i ] && a [ i + 1 ] ) {
for ( int j = 1 ; j <= i ; j ++ ) printf ( " % d ▁ " , j ) ; printf ( " % d ▁ " , N + 1 ) ; for ( int j = i + 1 ; j <= N ; j ++ ) printf ( " % d ▁ " , j ) ; return ; } }
for ( int i = 1 ; i <= N ; i ++ ) printf ( " % d ▁ " , i ) ; printf ( " % d ▁ " , N + 1 ) ; }
int N = 3 , arr [ ] = { 0 , 1 , 0 } ;
findpath ( N , arr ) ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
void printknapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } }
int res = K [ n ] [ W ] ; printf ( " % d STRNEWLINE " , res ) ; w = W ; for ( i = n ; i > 0 && res > 0 ; i -- ) {
if ( res == K [ i - 1 ] [ w ] ) continue ; else {
printf ( " % d ▁ " , wt [ i - 1 ] ) ;
res = res - val [ i - 1 ] ; w = w - wt [ i - 1 ] ; } } }
int optCost ( int freq [ ] , int i , int j ) {
return 0 ;
if ( j == i ) return freq [ i ] ;
int fsum = sum ( freq , i , j ) ;
int min = INT_MAX ;
for ( int r = i ; r <= j ; ++ r ) { int cost = optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ; if ( cost < min ) min = cost ; }
return min + fsum ; }
int optimalSearchTree ( int keys [ ] , int freq [ ] , int n ) {
return optCost ( freq , 0 , n - 1 ) ; }
int sum ( int freq [ ] , int i , int j ) { int s = 0 ; for ( int k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
int main ( ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = sizeof ( keys ) / sizeof ( keys [ 0 ] ) ; printf ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ % d ▁ " , optimalSearchTree ( keys , freq , n ) ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h> NEW_LINE #define INF  INT_MAX
int printSolution ( int p [ ] , int n ) ;
void solveWordWrap ( int l [ ] , int n , int M ) {
int extras [ n + 1 ] [ n + 1 ] ;
int lc [ n + 1 ] [ n + 1 ] ;
int c [ n + 1 ] ;
int p [ n + 1 ] ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( i = 1 ; i <= n ; i ++ ) { for ( j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = INF ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( j = 1 ; j <= n ; j ++ ) { c [ j ] = INF ; for ( i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != INF && lc [ i ] [ j ] != INF && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; } int printSolution ( int p [ ] , int n ) { int k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; printf ( " Line ▁ number ▁ % d : ▁ From ▁ word ▁ no . ▁ % d ▁ to ▁ % d ▁ STRNEWLINE " , k , p [ n ] , n ) ; return k ; }
int main ( ) { int l [ ] = { 3 , 2 , 2 , 5 } ; int n = sizeof ( l ) / sizeof ( l [ 0 ] ) ; int M = 6 ; solveWordWrap ( l , n , M ) ; return 0 ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int eggDrop ( int n , int k ) {
int eggFloor [ n + 1 ] [ k + 1 ] ; int res ; int i , j , x ;
for ( i = 1 ; i <= n ; i ++ ) { eggFloor [ i ] [ 1 ] = 1 ; eggFloor [ i ] [ 0 ] = 0 ; }
for ( j = 1 ; j <= k ; j ++ ) eggFloor [ 1 ] [ j ] = j ;
for ( i = 2 ; i <= n ; i ++ ) { for ( j = 2 ; j <= k ; j ++ ) { eggFloor [ i ] [ j ] = INT_MAX ; for ( x = 1 ; x <= j ; x ++ ) { res = 1 + max ( eggFloor [ i - 1 ] [ x - 1 ] , eggFloor [ i ] [ j - x ] ) ; if ( res < eggFloor [ i ] [ j ] ) eggFloor [ i ] [ j ] = res ; } } }
return eggFloor [ n ] [ k ] ; }
int main ( ) { int n = 2 , k = 36 ; printf ( " Minimum number of trials " STRNEWLINE " in worst case with % d eggs and " STRNEWLINE " % d floors is % d " , n , k , eggDrop ( n , k ) ) ; return 0 ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
let max_ref ;
int _lis ( int arr [ ] , int n , int * max_ref ) {
if ( n == 1 ) return 1 ;
int res , max_ending_here = 1 ;
for ( int i = 1 ; i < n ; i ++ ) { res = _lis ( arr , i , max_ref ) ; if ( arr [ i - 1 ] < arr [ n - 1 ] && res + 1 > max_ending_here ) max_ending_here = res + 1 ; }
if ( * max_ref < max_ending_here ) * max_ref = max_ending_here ;
return max_ending_here ; }
int lis ( int arr [ ] , int n ) {
int max = 1 ;
_lis ( arr , n , & max ) ;
return max ; }
int main ( ) { int arr [ ] = { 10 , 22 , 9 , 33 , 21 , 50 , 41 , 60 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Length ▁ of ▁ lis ▁ is ▁ % d " , lis ( arr , n ) ) ; return 0 ; }
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
#include <stdio.h> NEW_LINE #define N  8 NEW_LINE int solveKTUtil ( int x , int y , int movei , int sol [ N ] [ N ] , int xMove [ ] , int yMove [ ] ) ;
int isSafe ( int x , int y , int sol [ N ] [ N ] ) { return ( x >= 0 && x < N && y >= 0 && y < N && sol [ x ] [ y ] == -1 ) ; }
void printSolution ( int sol [ N ] [ N ] ) { for ( int x = 0 ; x < N ; x ++ ) { for ( int y = 0 ; y < N ; y ++ ) printf ( " ▁ % 2d ▁ " , sol [ x ] [ y ] ) ; printf ( " STRNEWLINE " ) ; } }
int solveKT ( ) { int sol [ N ] [ N ] ;
for ( int x = 0 ; x < N ; x ++ ) for ( int y = 0 ; y < N ; y ++ ) sol [ x ] [ y ] = -1 ;
int xMove [ 8 ] = { 2 , 1 , -1 , -2 , -2 , -1 , 1 , 2 } ; int yMove [ 8 ] = { 1 , 2 , 2 , 1 , -1 , -2 , -2 , -1 } ;
sol [ 0 ] [ 0 ] = 0 ;
if ( solveKTUtil ( 0 , 0 , 1 , sol , xMove , yMove ) == 0 ) { printf ( " Solution ▁ does ▁ not ▁ exist " ) ; return 0 ; } else printSolution ( sol ) ; return 1 ; }
int solveKTUtil ( int x , int y , int movei , int sol [ N ] [ N ] , int xMove [ N ] , int yMove [ N ] ) { int k , next_x , next_y ; if ( movei == N * N ) return 1 ;
for ( k = 0 ; k < 8 ; k ++ ) { next_x = x + xMove [ k ] ; next_y = y + yMove [ k ] ; if ( isSafe ( next_x , next_y , sol ) ) { sol [ next_x ] [ next_y ] = movei ; if ( solveKTUtil ( next_x , next_y , movei + 1 , sol , xMove , yMove ) == 1 ) return 1 ; else
sol [ next_x ] [ next_y ] = -1 ; } } return 0 ; }
solveKT ( ) ; return 0 ; }
#define V  4 NEW_LINE void printSolution ( int color [ ] ) ;
void printSolution ( int color [ ] ) { printf ( " Solution ▁ Exists : " " ▁ Following ▁ are ▁ the ▁ assigned ▁ colors ▁ STRNEWLINE " ) ; for ( int i = 0 ; i < V ; i ++ ) printf ( " ▁ % d ▁ " , color [ i ] ) ; printf ( " STRNEWLINE " ) ; }
bool isSafe ( bool graph [ V ] [ V ] , int color [ ] ) {
for ( int i = 0 ; i < V ; i ++ ) for ( int j = i + 1 ; j < V ; j ++ ) if ( graph [ i ] [ j ] && color [ j ] == color [ i ] ) return false ; return true ; }
bool graphColoring ( bool graph [ V ] [ V ] , int m , int i , int color [ V ] ) {
if ( i == V ) {
if ( isSafe ( graph , color ) ) {
printSolution ( color ) ; return true ; } return false ; }
for ( int j = 1 ; j <= m ; j ++ ) { color [ i ] = j ;
if ( graphColoring ( graph , m , i + 1 , color ) ) return true ; color [ i ] = 0 ; } return false ; }
bool graph [ V ] [ V ] = { { 0 , 1 , 1 , 1 } , { 1 , 0 , 1 , 0 } , { 1 , 1 , 0 , 1 } , { 1 , 0 , 1 , 0 } , } ;
int m = 3 ;
int color [ V ] ; for ( int i = 0 ; i < V ; i ++ ) color [ i ] = 0 ; if ( ! graphColoring ( graph , m , 0 , color ) ) printf ( " Solution ▁ does ▁ not ▁ exist " ) ; return 0 ; }
int prevPowerofK ( int n , int k ) { int p = ( int ) ( log ( n ) / log ( k ) ) ; return ( int ) pow ( k , p ) ; }
int nextPowerOfK ( int n , int k ) { return prevPowerofK ( n , k ) * k ; }
int main ( ) { int N = 7 ; int K = 2 ; printf ( " % d ▁ " , prevPowerofK ( N , K ) ) ; printf ( " % d STRNEWLINE " , nextPowerOfK ( N , K ) ) ; return 0 ; }
int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
int main ( ) { int a = 98 , b = 56 ; printf ( " GCD ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ % d ▁ " , a , b , gcd ( a , b ) ) ; return 0 ; }
int checkSemiprime ( int num ) { int cnt = 0 ; for ( int i = 2 ; cnt < 2 && i * i <= num ; ++ i ) while ( num % i == 0 ) num /= i , ++ cnt ;
if ( num > 1 ) ++ cnt ;
return cnt == 2 ; }
void semiprime ( int n ) { if ( checkSemiprime ( n ) ) printf ( " True STRNEWLINE " ) ; else printf ( " False STRNEWLINE " ) ; }
int main ( ) { int n = 6 ; semiprime ( n ) ; n = 8 ; semiprime ( n ) ; return 0 ; }
void printNSE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] > arr [ j ] ) { next = arr [ j ] ; break ; } } printf ( " % d ▁ - - ▁ % d STRNEWLINE " , arr [ i ] , next ) ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNSE ( arr , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE #define SIZE  100
char * base64Decoder ( char encoded [ ] , int len_str ) { char * decoded_string ; decoded_string = ( char * ) malloc ( sizeof ( char ) * SIZE ) ; int i , j , k = 0 ;
int num = 0 ;
int count_bits = 0 ;
for ( i = 0 ; i < len_str ; i += 4 ) { num = 0 , count_bits = 0 ; for ( j = 0 ; j < 4 ; j ++ ) {
if ( encoded [ i + j ] != ' = ' ) { num = num << 6 ; count_bits += 6 ; }
if ( encoded [ i + j ] >= ' A ' && encoded [ i + j ] <= ' Z ' ) num = num | ( encoded [ i + j ] - ' A ' ) ;
else if ( encoded [ i + j ] >= ' a ' && encoded [ i + j ] <= ' z ' ) num = num | ( encoded [ i + j ] - ' a ' + 26 ) ;
else if ( encoded [ i + j ] >= '0' && encoded [ i + j ] <= '9' ) num = num | ( encoded [ i + j ] - '0' + 52 ) ;
else if ( encoded [ i + j ] == ' + ' ) num = num | 62 ;
else if ( encoded [ i + j ] == ' / ' ) num = num | 63 ;
else { num = num >> 2 ; count_bits -= 2 ; } } while ( count_bits != 0 ) { count_bits -= 8 ;
decoded_string [ k ++ ] = ( num >> count_bits ) & 255 ; } } decoded_string [ k ] = ' \0' ; return decoded_string ; }
int main ( ) { char encoded_string [ ] = " TUVOT04 = " ; int len_str = sizeof ( encoded_string ) / sizeof ( encoded_string [ 0 ] ) ;
len_str -= 1 ; printf ( " Encoded ▁ string ▁ : ▁ % s STRNEWLINE " , encoded_string ) ; printf ( " Decoded _ string ▁ : ▁ % s STRNEWLINE " , base64Decoder ( encoded_string , len_str ) ) ; return 0 ; }
void divideString ( char * str , int n ) { int str_size = strlen ( str ) ; int i ; int part_size ;
if ( str_size % n != 0 ) { printf ( " Invalid ▁ Input : ▁ String ▁ size " ) ; printf ( " ▁ is ▁ not ▁ divisible ▁ by ▁ n " ) ; return ; }
part_size = str_size / n ; for ( i = 0 ; i < str_size ; i ++ ) { if ( i % part_size == 0 ) printf ( " STRNEWLINE " ) ; printf ( " % c " , str [ i ] ) ; } } int main ( ) {
char * str = " a _ simple _ divide _ string _ quest " ;
divideString ( str , 4 ) ; getchar ( ) ; return 0 ; }
void collinear ( int x1 , int y1 , int x2 , int y2 , int x3 , int y3 ) { if ( ( y3 - y2 ) * ( x2 - x1 ) == ( y2 - y1 ) * ( x3 - x2 ) ) printf ( " Yes " ) ; else printf ( " No " ) ; }
int main ( ) { int x1 = 1 , x2 = 1 , x3 = 0 , y1 = 1 , y2 = 6 , y3 = 9 ; collinear ( x1 , y1 , x2 , y2 , x3 , y3 ) ; return 0 ; }
void bestApproximate ( int x [ ] , int y [ ] , int n ) { int i , j ; float m , c , sum_x = 0 , sum_y = 0 , sum_xy = 0 , sum_x2 = 0 ; for ( i = 0 ; i < n ; i ++ ) { sum_x += x [ i ] ; sum_y += y [ i ] ; sum_xy += x [ i ] * y [ i ] ; sum_x2 += ( x [ i ] * x [ i ] ) ; } m = ( n * sum_xy - sum_x * sum_y ) / ( n * sum_x2 - ( sum_x * sum_x ) ) ; c = ( sum_y - m * sum_x ) / n ; printf ( " m ▁ = % ▁ f " , m ) ; printf ( " c = % f " , c ) ; }
int main ( ) { int x [ ] = { 1 , 2 , 3 , 4 , 5 } ; int y [ ] = { 14 , 27 , 40 , 55 , 68 } ; int n = sizeof ( x ) / sizeof ( x [ 0 ] ) ; bestApproximate ( x , y , n ) ; return 0 ; }
int findMinInsertions ( char str [ ] , int l , int h ) {
if ( l > h ) return INT_MAX ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) ; }
int main ( ) { char str [ ] = " geeks " ; printf ( " % d " , findMinInsertions ( str , 0 , strlen ( str ) - 1 ) ) ; return 0 ; }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void insertAfter ( struct Node * prev_node , int new_data ) {
if ( prev_node == NULL ) { printf ( " the ▁ given ▁ previous ▁ node ▁ cannot ▁ be ▁ NULL " ) ; return ; }
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = prev_node -> next ;
prev_node -> next = new_node ; }
struct Node { char data ; struct Node * next ; } ; void reverse ( struct Node * * ) ; bool compareLists ( struct Node * , struct Node * ) ;
bool isPalindrome ( struct Node * head ) { struct Node * slow_ptr = head , * fast_ptr = head ; struct Node * second_half , * prev_of_slow_ptr = head ;
struct Node * midnode = NULL ;
bool res = true ; if ( head != NULL && head -> next != NULL ) {
while ( fast_ptr != NULL && fast_ptr -> next != NULL ) { fast_ptr = fast_ptr -> next -> next ;
prev_of_slow_ptr = slow_ptr ; slow_ptr = slow_ptr -> next ; }
if ( fast_ptr != NULL ) { midnode = slow_ptr ; slow_ptr = slow_ptr -> next ; }
second_half = slow_ptr ;
prev_of_slow_ptr -> next = NULL ;
reverse ( & second_half ) ;
res = compareLists ( head , second_half ) ;
reverse ( & second_half ) ;
if ( midnode != NULL ) { prev_of_slow_ptr -> next = midnode ; midnode -> next = second_half ; } else prev_of_slow_ptr -> next = second_half ; } return res ; }
void reverse ( struct Node * * head_ref ) { struct Node * prev = NULL ; struct Node * current = * head_ref ; struct Node * next ; while ( current != NULL ) { next = current -> next ; current -> next = prev ; prev = current ; current = next ; } * head_ref = prev ; }
bool compareLists ( struct Node * head1 , struct Node * head2 ) { struct Node * temp1 = head1 ; struct Node * temp2 = head2 ; while ( temp1 && temp2 ) { if ( temp1 -> data == temp2 -> data ) { temp1 = temp1 -> next ; temp2 = temp2 -> next ; } else return 0 ; }
if ( temp1 == NULL && temp2 == NULL ) return 1 ;
return 0 ; }
void push ( struct Node * * head_ref , char new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( struct Node * ptr ) { while ( ptr != NULL ) { printf ( " % c - > " , ptr -> data ) ; ptr = ptr -> next ; } printf ( " NULL STRNEWLINE " ) ; }
struct Node * head = NULL ; char str [ ] = " abacaba " ; int i ; for ( i = 0 ; str [ i ] != ' \0' ; i ++ ) { push ( & head , str [ i ] ) ; printList ( head ) ; isPalindrome ( head ) ? printf ( " Is ▁ Palindrome STRNEWLINE STRNEWLINE " ) : printf ( " Not ▁ Palindrome STRNEWLINE STRNEWLINE " ) ; } return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE struct Node { int data ; struct Node * next ; } ;
void swapNodes ( struct Node * * head_ref , int x , int y ) {
if ( x == y ) return ;
struct Node * prevX = NULL , * currX = * head_ref ; while ( currX && currX -> data != x ) { prevX = currX ; currX = currX -> next ; }
struct Node * prevY = NULL , * currY = * head_ref ; while ( currY && currY -> data != y ) { prevY = currY ; currY = currY -> next ; }
if ( currX == NULL currY == NULL ) return ;
if ( prevX != NULL ) prevX -> next = currY ;
else * head_ref = currY ;
if ( prevY != NULL ) prevY -> next = currX ;
else * head_ref = currX ;
struct Node * temp = currY -> next ; currY -> next = currX -> next ; currX -> next = temp ; }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( struct Node * node ) { while ( node != NULL ) { printf ( " % d ▁ " , node -> data ) ; node = node -> next ; } }
push ( & start , 7 ) ; push ( & start , 6 ) ; push ( & start , 5 ) ; push ( & start , 4 ) ; push ( & start , 3 ) ; push ( & start , 2 ) ; push ( & start , 1 ) ; printf ( " Linked list before calling swapNodes ( ) " printList ( start ) ; swapNodes ( & start , 4 , 3 ) ; printf ( " Linked list after calling swapNodes ( ) " printList ( start ) ; return 0 ; }
struct Node { struct Node * prev ; int info ; struct Node * next ; } ;
void nodeInsetail ( struct Node * * head , struct Node * * tail , int key ) { struct Node * p = new Node ; p -> info = key ; p -> next = NULL ;
if ( ( * head ) == NULL ) { ( * head ) = p ; ( * tail ) = p ; ( * head ) -> prev = NULL ; return ; }
if ( ( p -> info ) < ( ( * head ) -> info ) ) { p -> prev = NULL ; ( * head ) -> prev = p ; p -> next = ( * head ) ; ( * head ) = p ; return ; }
if ( ( p -> info ) > ( ( * tail ) -> info ) ) { p -> prev = ( * tail ) ; ( * tail ) -> next = p ; ( * tail ) = p ; return ; }
temp = ( * head ) -> next ; while ( ( temp -> info ) < ( p -> info ) ) temp = temp -> next ;
( temp -> prev ) -> next = p ; p -> prev = temp -> prev ; temp -> prev = p ; p -> next = temp ; }
void printList ( struct Node * temp ) { while ( temp != NULL ) { printf ( " % d ▁ " , temp -> info ) ; temp = temp -> next ; } }
int main ( ) { struct Node * left = NULL , * right = NULL ; nodeInsetail ( & left , & right , 30 ) ; nodeInsetail ( & left , & right , 50 ) ; nodeInsetail ( & left , & right , 90 ) ; nodeInsetail ( & left , & right , 10 ) ; nodeInsetail ( & left , & right , 40 ) ; nodeInsetail ( & left , & right , 110 ) ; nodeInsetail ( & left , & right , 60 ) ; nodeInsetail ( & left , & right , 95 ) ; nodeInsetail ( & left , & right , 23 ) ; printf ( " Doubly linked list on printing " ▁ " from left to right " printList ( left ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
struct node { int data ; struct node * left ; struct node * right ; struct node * parent ; } ; struct node * minValue ( struct node * node ) ;
struct node * insert ( struct node * node , int data ) {
if ( node == NULL ) return ( newNode ( data ) ) ; else { struct node * temp ;
if ( data <= node -> data ) { temp = insert ( node -> left , data ) ; node -> left = temp ; temp -> parent = node ; } else { temp = insert ( node -> right , data ) ; node -> right = temp ; temp -> parent = node ; }
return node ; } } struct node * inOrderSuccessor ( struct node * root , struct node * n ) {
if ( n -> right != NULL ) return minValue ( n -> right ) ;
struct node * p = n -> parent ; while ( p != NULL && n == p -> right ) { n = p ; p = p -> parent ; } return p ; }
struct node * minValue ( struct node * node ) { struct node * current = node ;
while ( current -> left != NULL ) { current = current -> left ; } return current ; }
int main ( ) { struct node * root = NULL , * temp , * succ , * min ; root = insert ( root , 20 ) ; root = insert ( root , 8 ) ; root = insert ( root , 22 ) ; root = insert ( root , 4 ) ; root = insert ( root , 12 ) ; root = insert ( root , 10 ) ; root = insert ( root , 14 ) ; temp = root -> left -> right -> right ; succ = inOrderSuccessor ( root , temp ) ; if ( succ != NULL ) printf ( " Inorder Successor of % d is % d " , temp -> data , succ -> data ) ; else printf ( " Inorder Successor doesn ' exit " getchar ( ) ; return 0 ; }
struct node { int key ; struct node * left ; struct node * right ; } ;
void convertBSTtoDLL ( node * root , node * * head , node * * tail ) {
if ( root == NULL ) return ;
if ( root -> left ) convertBSTtoDLL ( root -> left , head , tail ) ;
root -> left = * tail ;
if ( * tail ) ( * tail ) -> right = root ; else * head = root ;
* tail = root ;
if ( root -> right ) convertBSTtoDLL ( root -> right , head , tail ) ; }
bool isPresentInDLL ( node * head , node * tail , int sum ) { while ( head != tail ) { int curr = head -> key + tail -> key ; if ( curr == sum ) return true ; else if ( curr > sum ) tail = tail -> left ; else head = head -> right ; } return false ; }
bool isTripletPresent ( node * root ) {
if ( root == NULL ) return false ;
node * head = NULL ; node * tail = NULL ; convertBSTtoDLL ( root , & head , & tail ) ;
while ( ( head -> right != tail ) && ( head -> key < 0 ) ) {
if ( isPresentInDLL ( head -> right , tail , -1 * head -> key ) ) return true ; else head = head -> right ; }
return false ; }
node * newNode ( int num ) { node * temp = new node ; temp -> key = num ; temp -> left = temp -> right = NULL ; return temp ; }
node * insert ( node * root , int key ) { if ( root == NULL ) return newNode ( key ) ; if ( root -> key > key ) root -> left = insert ( root -> left , key ) ; else root -> right = insert ( root -> right , key ) ; return root ; }
int main ( ) { node * root = NULL ; root = insert ( root , 6 ) ; root = insert ( root , -13 ) ; root = insert ( root , 14 ) ; root = insert ( root , -8 ) ; root = insert ( root , 15 ) ; root = insert ( root , 13 ) ; root = insert ( root , 7 ) ; if ( isTripletPresent ( root ) ) printf ( " Present " ) ; else printf ( " Not ▁ Present " ) ; return 0 ; }
#include <stdio.h> NEW_LINE void printSorted ( int arr [ ] , int start , int end ) { if ( start > end ) return ;
printSorted ( arr , start * 2 + 1 , end ) ;
printf ( " % d ▁ " , arr [ start ] ) ;
printSorted ( arr , start * 2 + 2 , end ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 5 , 1 , 3 } ; int arr_size = sizeof ( arr ) / sizeof ( int ) ; printSorted ( arr , 0 , arr_size - 1 ) ; getchar ( ) ; return 0 ; }
struct node { int key ; struct node * left ; struct node * right ; } ;
int Ceil ( struct node * root , int input ) {
if ( root == NULL ) return -1 ;
if ( root -> key == input ) return root -> key ;
if ( root -> key < input ) return Ceil ( root -> right , input ) ;
int ceil = Ceil ( root -> left , input ) ; return ( ceil >= input ) ? ceil : root -> key ; }
int main ( ) { struct node * root = newNode ( 8 ) ; root -> left = newNode ( 4 ) ; root -> right = newNode ( 12 ) ; root -> left -> left = newNode ( 2 ) ; root -> left -> right = newNode ( 6 ) ; root -> right -> left = newNode ( 10 ) ; root -> right -> right = newNode ( 14 ) ; for ( int i = 0 ; i < 16 ; i ++ ) printf ( " % d ▁ % d STRNEWLINE " , i , Ceil ( root , i ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE struct node { int key ; int count ; struct node * left , * right ; } ;
struct node * newNode ( int item ) { struct node * temp = ( struct node * ) malloc ( sizeof ( struct node ) ) ; temp -> key = item ; temp -> left = temp -> right = NULL ; temp -> count = 1 ; return temp ; }
void inorder ( struct node * root ) { if ( root != NULL ) { inorder ( root -> left ) ; printf ( " % d ( % d ) ▁ " , root -> key , root -> count ) ; inorder ( root -> right ) ; } }
struct node * insert ( struct node * node , int key ) {
if ( node == NULL ) return newNode ( key ) ;
if ( key == node -> key ) { ( node -> count ) ++ ; return node ; }
if ( key < node -> key ) node -> left = insert ( node -> left , key ) ; else node -> right = insert ( node -> right , key ) ;
return node ; }
struct node * minValueNode ( struct node * node ) { struct node * current = node ;
while ( current -> left != NULL ) current = current -> left ; return current ; }
struct node * deleteNode ( struct node * root , int key ) {
if ( root == NULL ) return root ;
if ( key < root -> key ) root -> left = deleteNode ( root -> left , key ) ;
else if ( key > root -> key ) root -> right = deleteNode ( root -> right , key ) ;
else {
if ( root -> count > 1 ) { ( root -> count ) -- ; return root ; }
if ( root -> left == NULL ) { struct node * temp = root -> right ; free ( root ) ; return temp ; } else if ( root -> right == NULL ) { struct node * temp = root -> left ; free ( root ) ; return temp ; }
struct node * temp = minValueNode ( root -> right ) ;
root -> key = temp -> key ;
root -> right = deleteNode ( root -> right , temp -> key ) ; } return root ; }
struct node * root = NULL ; root = insert ( root , 12 ) ; root = insert ( root , 10 ) ; root = insert ( root , 20 ) ; root = insert ( root , 9 ) ; root = insert ( root , 11 ) ; root = insert ( root , 10 ) ; root = insert ( root , 12 ) ; root = insert ( root , 12 ) ; printf ( " Inorder ▁ traversal ▁ of ▁ the ▁ given ▁ tree ▁ STRNEWLINE " ) ; inorder ( root ) ; printf ( " Delete 20 " root = deleteNode ( root , 20 ) ; printf ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ STRNEWLINE " ) ; inorder ( root ) ; printf ( " Delete 12 " root = deleteNode ( root , 12 ) ; printf ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ STRNEWLINE " ) ; inorder ( root ) ; printf ( " Delete 9 " root = deleteNode ( root , 9 ) ; printf ( " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ STRNEWLINE " ) ; inorder ( root ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE struct node { int key ; struct node * left , * right ; } ;
struct node * newNode ( int item ) { struct node * temp = ( struct node * ) malloc ( sizeof ( struct node ) ) ; temp -> key = item ; temp -> left = temp -> right = NULL ; return temp ; }
void inorder ( struct node * root ) { if ( root != NULL ) { inorder ( root -> left ) ; printf ( " % d ▁ " , root -> key ) ; inorder ( root -> right ) ; } }
struct node * insert ( struct node * node , int key ) {
if ( node == NULL ) return newNode ( key ) ;
if ( key < node -> key ) node -> left = insert ( node -> left , key ) ; else node -> right = insert ( node -> right , key ) ;
return node ; }
struct node * minValueNode ( struct node * node ) { struct node * current = node ;
while ( current -> left != NULL ) current = current -> left ; return current ; }
struct node * deleteNode ( struct node * root , int key ) {
if ( root == NULL ) return root ;
if ( key < root -> key ) root -> left = deleteNode ( root -> left , key ) ;
else if ( key > root -> key ) root -> right = deleteNode ( root -> right , key ) ;
else {
if ( root -> left == NULL ) { struct node * temp = root -> right ; free ( root ) ; return temp ; } else if ( root -> right == NULL ) { struct node * temp = root -> left ; free ( root ) ; return temp ; }
struct node * temp = minValueNode ( root -> right ) ;
root -> key = temp -> key ;
root -> right = deleteNode ( root -> right , temp -> key ) ; } return root ; }
struct node * changeKey ( struct node * root , int oldVal , int newVal ) {
root = deleteNode ( root , oldVal ) ;
root = insert ( root , newVal ) ;
return root ; }
struct node * root = NULL ; root = insert ( root , 50 ) ; root = insert ( root , 30 ) ; root = insert ( root , 20 ) ; root = insert ( root , 40 ) ; root = insert ( root , 70 ) ; root = insert ( root , 60 ) ; root = insert ( root , 80 ) ; printf ( " Inorder ▁ traversal ▁ of ▁ the ▁ given ▁ tree ▁ STRNEWLINE " ) ; inorder ( root ) ; root = changeKey ( root , 40 , 10 ) ;
printf ( " Inorder traversal of the modified tree " inorder ( root ) ; return 0 ; }
struct Node { struct Node * left ; int info ; struct Node * right ; } ;
void insert ( struct Node * * rt , int key ) {
if ( * rt == NULL ) { ( * rt ) = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; ( * rt ) -> left = NULL ; ( * rt ) -> right = NULL ; ( * rt ) -> info = key ; }
else if ( key < ( ( * rt ) -> info ) ) insert ( & ( ( * rt ) -> left ) , key ) ; else insert ( & ( * rt ) -> right , key ) ; }
int check ( int num ) { int sum = 0 , i = num , sum_of_digits , prod_of_digits ;
if ( num < 10 num > 99 ) return 0 ; else { sum_of_digits = ( i % 10 ) + ( i / 10 ) ; prod_of_digits = ( i % 10 ) * ( i / 10 ) ; sum = sum_of_digits + prod_of_digits ; } if ( sum == num ) return 1 ; else return 0 ; }
void countSpecialDigit ( struct Node * rt , int * c ) { int x ; if ( rt == NULL ) return ; else { x = check ( rt -> info ) ; if ( x == 1 ) * c = * c + 1 ; countSpecialDigit ( rt -> left , c ) ; countSpecialDigit ( rt -> right , c ) ; } }
int main ( ) { struct Node * root = NULL ; int count = 0 ; insert ( & root , 50 ) ; insert ( & root , 29 ) ; insert ( & root , 59 ) ; insert ( & root , 19 ) ; insert ( & root , 53 ) ; insert ( & root , 556 ) ; insert ( & root , 56 ) ; insert ( & root , 94 ) ; insert ( & root , 13 ) ;
countSpecialDigit ( root , & count ) ; printf ( " % d " , count ) ; return 0 ; }
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
char a [ m ] [ n ] ;
char x = ' X ' ;
while ( k < m && l < n ) {
for ( i = l ; i < n ; ++ i ) a [ k ] [ i ] = x ; k ++ ;
for ( i = k ; i < m ; ++ i ) a [ i ] [ n - 1 ] = x ; n -- ;
if ( k < m ) { for ( i = n - 1 ; i >= l ; -- i ) a [ m - 1 ] [ i ] = x ; m -- ; }
if ( l < n ) { for ( i = m - 1 ; i >= k ; -- i ) a [ i ] [ l ] = x ; l ++ ; }
x = ( x == '0' ) ? ' X ' : '0' ; }
for ( i = 0 ; i < r ; i ++ ) { for ( int j = 0 ; j < c ; j ++ ) printf ( " % c ▁ " , a [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } }
int main ( ) { puts ( " Output ▁ for ▁ m ▁ = ▁ 5 , ▁ n ▁ = ▁ 6" ) ; fill0X ( 5 , 6 ) ; puts ( " Output for m = 4 , n = 4 " ) ; fill0X ( 4 , 4 ) ; puts ( " Output for m = 3 , n = 4 " ) ; fill0X ( 3 , 4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  3
void interchangeDiagonals ( int array [ ] [ N ] ) {
for ( int i = 0 ; i < N ; ++ i ) if ( i != N / 2 ) swap ( array [ i ] [ i ] , array [ i ] [ N - i - 1 ] ) ; for ( int i = 0 ; i < N ; ++ i ) { for ( int j = 0 ; j < N ; ++ j ) printf ( " ▁ % d " , array [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } }
int main ( ) { int array [ N ] [ N ] = { 4 , 5 , 6 , 1 , 2 , 3 , 7 , 8 , 9 } ; interchangeDiagonals ( array ) ; return 0 ; }
struct node { int data ; node * left ; node * right ; } ;
node * bintree2listUtil ( node * root ) {
if ( root == NULL ) return root ;
if ( root -> left != NULL ) {
node * left = bintree2listUtil ( root -> left ) ;
for ( ; left -> right != NULL ; left = left -> right ) ;
left -> right = root ;
root -> left = left ; }
if ( root -> right != NULL ) {
node * right = bintree2listUtil ( root -> right ) ;
for ( ; right -> left != NULL ; right = right -> left ) ;
right -> left = root ;
root -> right = right ; } return root ; }
node * bintree2list ( node * root ) {
if ( root == NULL ) return root ;
root = bintree2listUtil ( root ) ;
while ( root -> left != NULL ) root = root -> left ; return ( root ) ; }
void printList ( node * node ) { while ( node != NULL ) { printf ( " % d ▁ " , node -> data ) ; node = node -> right ; } }
node * root = newNode ( 10 ) ; root -> left = newNode ( 12 ) ; root -> right = newNode ( 15 ) ; root -> left -> left = newNode ( 25 ) ; root -> left -> right = newNode ( 30 ) ; root -> right -> left = newNode ( 36 ) ;
node * head = bintree2list ( root ) ;
printList ( head ) ; return 0 ; }
#define M  4 NEW_LINE #define N  5
int findCommon ( int mat [ M ] [ N ] ) {
int column [ M ] ;
int min_row ;
int i ; for ( i = 0 ; i < M ; i ++ ) column [ i ] = N - 1 ;
min_row = 0 ;
while ( column [ min_row ] >= 0 ) {
for ( i = 0 ; i < M ; i ++ ) { if ( mat [ i ] [ column [ i ] ] < mat [ min_row ] [ column [ min_row ] ] ) min_row = i ; }
int eq_count = 0 ;
for ( i = 0 ; i < M ; i ++ ) {
if ( mat [ i ] [ column [ i ] ] > mat [ min_row ] [ column [ min_row ] ] ) { if ( column [ i ] == 0 ) return -1 ;
column [ i ] -= 1 ; } else eq_count ++ ; }
if ( eq_count == M ) return mat [ min_row ] [ column [ min_row ] ] ; } return -1 ; }
int main ( ) { int mat [ M ] [ N ] = { { 1 , 2 , 3 , 4 , 5 } , { 2 , 4 , 5 , 8 , 10 } , { 3 , 5 , 7 , 9 , 11 } , { 1 , 3 , 5 , 7 , 9 } , } ; int result = findCommon ( mat ) ; if ( result == -1 ) printf ( " No ▁ common ▁ element " ) ; else printf ( " Common ▁ element ▁ is ▁ % d " , result ) ; return 0 ; }
struct node { int data ; struct node * left , * right ; } ;
void inorder ( struct node * root ) { if ( root != NULL ) { inorder ( root -> left ) ; printf ( " TABSYMBOL % d " , root -> data ) ; inorder ( root -> right ) ; } }
void fixPrevPtr ( struct node * root ) { static struct node * pre = NULL ; if ( root != NULL ) { fixPrevPtr ( root -> left ) ; root -> left = pre ; pre = root ; fixPrevPtr ( root -> right ) ; } }
struct node * fixNextPtr ( struct node * root ) { struct node * prev = NULL ;
while ( root && root -> right != NULL ) root = root -> right ;
while ( root && root -> left != NULL ) { prev = root ; root = root -> left ; root -> right = prev ; }
return ( root ) ; }
struct node * BTToDLL ( struct node * root ) {
fixPrevPtr ( root ) ;
return fixNextPtr ( root ) ; }
void printList ( struct node * root ) { while ( root != NULL ) { printf ( " TABSYMBOL % d " , root -> data ) ; root = root -> right ; } }
struct node * root = newNode ( 10 ) ; root -> left = newNode ( 12 ) ; root -> right = newNode ( 15 ) ; root -> left -> left = newNode ( 25 ) ; root -> left -> right = newNode ( 30 ) ; root -> right -> left = newNode ( 36 ) ; printf ( " Inorder Tree Traversal " inorder ( root ) ; struct node * head = BTToDLL ( root ) ; printf ( " DLL Traversal " printList ( head ) ; return 0 ; }
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
int binarySearch ( int arr [ ] , int l , int r , int x ) { if ( r >= l ) { int mid = l + ( r - l ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 10 ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
int interpolationSearch ( int arr [ ] , int lo , int hi , int x ) { int pos ;
if ( lo <= hi && x >= arr [ lo ] && x <= arr [ hi ] ) {
pos = lo + ( ( ( double ) ( hi - lo ) / ( arr [ hi ] - arr [ lo ] ) ) * ( x - arr [ lo ] ) ) ;
if ( arr [ pos ] == x ) return pos ;
if ( arr [ pos ] < x ) return interpolationSearch ( arr , pos + 1 , hi , x ) ;
if ( arr [ pos ] > x ) return interpolationSearch ( arr , lo , pos - 1 , x ) ; } return -1 ; }
int arr [ ] = { 10 , 12 , 13 , 16 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 33 , 35 , 42 , 47 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int x = 18 ; int index = interpolationSearch ( arr , 0 , n - 1 , x ) ;
if ( index != -1 ) printf ( " Element ▁ found ▁ at ▁ index ▁ % d " , index ) ; else printf ( " Element ▁ not ▁ found . " ) ; return 0 ; }
void merge ( int arr [ ] , int l , int m , int r ) { int i , j , k ;
int n1 = m - l + 1 ; int n2 = r - m ;
int L [ n1 ] , R [ n2 ] ;
for ( i = 0 ; i < n1 ; i ++ ) L [ i ] = arr [ l + i ] ; for ( j = 0 ; j < n2 ; j ++ ) R [ j ] = arr [ m + 1 + j ] ;
i = 0 ; j = 0 ;
k = l ; while ( i < n1 && j < n2 ) { if ( L [ i ] <= R [ j ] ) { arr [ k ] = L [ i ] ; i ++ ; } else { arr [ k ] = R [ j ] ; j ++ ; } k ++ ; }
while ( i < n1 ) { arr [ k ] = L [ i ] ; i ++ ; k ++ ; }
while ( j < n2 ) { arr [ k ] = R [ j ] ; j ++ ; k ++ ; } }
void mergeSort ( int arr [ ] , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
mergeSort ( arr , l , m ) ; mergeSort ( arr , m + 1 , r ) ;
merge ( arr , l , m , r ) ; } }
void printArray ( int A [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , A [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 , 7 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Given ▁ array ▁ is ▁ STRNEWLINE " ) ; printArray ( arr , arr_size ) ; mergeSort ( arr , 0 , arr_size - 1 ) ; printf ( " Sorted array is " printArray ( arr , arr_size ) ; return 0 ; }
void printMaxActivities ( int s [ ] , int f [ ] , int n ) { int i , j ; printf ( " Following ▁ activities ▁ are ▁ selected ▁ n " ) ;
i = 0 ; printf ( " % d ▁ " , i ) ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { printf ( " % d ▁ " , j ) ; i = j ; } } }
int main ( ) { int s [ ] = { 1 , 3 , 0 , 5 , 8 , 5 } ; int f [ ] = { 2 , 4 , 6 , 7 , 9 , 9 } ; int n = sizeof ( s ) / sizeof ( s [ 0 ] ) ; printMaxActivities ( s , f , n ) ; return 0 ; }
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
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
int eggDrop ( int n , int k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; int min = INT_MAX , x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
int main ( ) { int n = 2 , k = 10 ; printf ( " nMinimum ▁ number ▁ of ▁ trials ▁ in ▁ " " worst ▁ case ▁ with ▁ % d ▁ eggs ▁ and ▁ " " % d ▁ floors ▁ is ▁ % d ▁ STRNEWLINE " , n , k , eggDrop ( n , k ) ) ; return 0 ; }
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h> NEW_LINE #define INF  INT_MAX
int printSolution ( int p [ ] , int n ) ; int printSolution ( int p [ ] , int n ) { int k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; printf ( " Line ▁ number ▁ % d : ▁ From ▁ word ▁ no . ▁ % d ▁ to ▁ % d ▁ STRNEWLINE " , k , p [ n ] , n ) ; return k ; }
void solveWordWrap ( int l [ ] , int n , int M ) {
int extras [ n + 1 ] [ n + 1 ] ;
int lc [ n + 1 ] [ n + 1 ] ;
int c [ n + 1 ] ;
int p [ n + 1 ] ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( i = 1 ; i <= n ; i ++ ) { for ( j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = INF ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( j = 1 ; j <= n ; j ++ ) { c [ j ] = INF ; for ( i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != INF && lc [ i ] [ j ] != INF && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; }
int main ( ) { int l [ ] = { 3 , 2 , 2 , 5 } ; int n = sizeof ( l ) / sizeof ( l [ 0 ] ) ; int M = 6 ; solveWordWrap ( l , n , M ) ; return 0 ; }
int sum ( int freq [ ] , int i , int j ) { int s = 0 ; for ( int k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
int optCost ( int freq [ ] , int i , int j ) {
if ( j < i ) return 0 ;
if ( j == i ) return freq [ i ] ;
int fsum = sum ( freq , i , j ) ;
int min = INT_MAX ;
for ( int r = i ; r <= j ; ++ r ) { int cost = optCost ( freq , i , r - 1 ) + optCost ( freq , r + 1 , j ) ; if ( cost < min ) min = cost ; }
return min + fsum ; }
int optimalSearchTree ( int keys [ ] , int freq [ ] , int n ) {
return optCost ( freq , 0 , n - 1 ) ; }
int main ( ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = sizeof ( keys ) / sizeof ( keys [ 0 ] ) ; printf ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ % d ▁ " , optimalSearchTree ( keys , freq , n ) ) ; return 0 ; }
int sum ( int freq [ ] , int i , int j ) { int s = 0 ; for ( int k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
int optimalSearchTree ( int keys [ ] , int freq [ ] , int n ) {
int cost [ n ] [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) cost [ i ] [ i ] = freq [ i ] ;
for ( int L = 2 ; L <= n ; L ++ ) {
for ( int i = 0 ; i <= n - L + 1 ; i ++ ) {
int j = i + L - 1 ; cost [ i ] [ j ] = INT_MAX ;
for ( int r = i ; r <= j ; r ++ ) {
int c = ( ( r > i ) ? cost [ i ] [ r - 1 ] : 0 ) + ( ( r < j ) ? cost [ r + 1 ] [ j ] : 0 ) + sum ( freq , i , j ) ; if ( c < cost [ i ] [ j ] ) cost [ i ] [ j ] = c ; } } } return cost [ 0 ] [ n - 1 ] ; }
int main ( ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = sizeof ( keys ) / sizeof ( keys [ 0 ] ) ; printf ( " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ % d ▁ " , optimalSearchTree ( keys , freq , n ) ) ; return 0 ; }
int getCount ( char keypad [ ] [ 3 ] , int n ) { if ( keypad == NULL n <= 0 ) return 0 ; if ( n == 1 ) return 10 ;
int odd [ 10 ] , even [ 10 ] ; int i = 0 , j = 0 , useOdd = 0 , totalCount = 0 ; for ( i = 0 ; i <= 9 ; i ++ )
odd [ i ] = 1 ;
for ( j = 2 ; j <= n ; j ++ ) { useOdd = 1 - useOdd ;
if ( useOdd == 1 ) { even [ 0 ] = odd [ 0 ] + odd [ 8 ] ; even [ 1 ] = odd [ 1 ] + odd [ 2 ] + odd [ 4 ] ; even [ 2 ] = odd [ 2 ] + odd [ 1 ] + odd [ 3 ] + odd [ 5 ] ; even [ 3 ] = odd [ 3 ] + odd [ 2 ] + odd [ 6 ] ; even [ 4 ] = odd [ 4 ] + odd [ 1 ] + odd [ 5 ] + odd [ 7 ] ; even [ 5 ] = odd [ 5 ] + odd [ 2 ] + odd [ 4 ] + odd [ 8 ] + odd [ 6 ] ; even [ 6 ] = odd [ 6 ] + odd [ 3 ] + odd [ 5 ] + odd [ 9 ] ; even [ 7 ] = odd [ 7 ] + odd [ 4 ] + odd [ 8 ] ; even [ 8 ] = odd [ 8 ] + odd [ 0 ] + odd [ 5 ] + odd [ 7 ] + odd [ 9 ] ; even [ 9 ] = odd [ 9 ] + odd [ 6 ] + odd [ 8 ] ; } else { odd [ 0 ] = even [ 0 ] + even [ 8 ] ; odd [ 1 ] = even [ 1 ] + even [ 2 ] + even [ 4 ] ; odd [ 2 ] = even [ 2 ] + even [ 1 ] + even [ 3 ] + even [ 5 ] ; odd [ 3 ] = even [ 3 ] + even [ 2 ] + even [ 6 ] ; odd [ 4 ] = even [ 4 ] + even [ 1 ] + even [ 5 ] + even [ 7 ] ; odd [ 5 ] = even [ 5 ] + even [ 2 ] + even [ 4 ] + even [ 8 ] + even [ 6 ] ; odd [ 6 ] = even [ 6 ] + even [ 3 ] + even [ 5 ] + even [ 9 ] ; odd [ 7 ] = even [ 7 ] + even [ 4 ] + even [ 8 ] ; odd [ 8 ] = even [ 8 ] + even [ 0 ] + even [ 5 ] + even [ 7 ] + even [ 9 ] ; odd [ 9 ] = even [ 9 ] + even [ 6 ] + even [ 8 ] ; } }
totalCount = 0 ; if ( useOdd == 1 ) { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += even [ i ] ; } else { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += odd [ i ] ; } return totalCount ; }
int main ( ) { char keypad [ 4 ] [ 3 ] = { { '1' , '2' , '3' } , { '4' , '5' , '6' } , { '7' , '8' , '9' } , { ' * ' , '0' , ' # ' } } ; printf ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ % d : ▁ % dn " , 1 , getCount ( keypad , 1 ) ) ; printf ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ % d : ▁ % dn " , 2 , getCount ( keypad , 2 ) ) ; printf ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ % d : ▁ % dn " , 3 , getCount ( keypad , 3 ) ) ; printf ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ % d : ▁ % dn " , 4 , getCount ( keypad , 4 ) ) ; printf ( " Count ▁ for ▁ numbers ▁ of ▁ length ▁ % d : ▁ % dn " , 5 , getCount ( keypad , 5 ) ) ; return 0 ; }
int count ( int n ) {
int table [ n + 1 ] , i ;
memset ( table , 0 , sizeof ( table ) ) ;
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
float exponential ( int n , float x ) {
float sum = 1.0f ; for ( int i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
int main ( ) { int n = 10 ; float x = 1.0f ; printf ( " e ^ x ▁ = ▁ % f " , exponential ( n , x ) ) ; return 0 ; }
int findCeil ( int arr [ ] , int r , int l , int h ) { int mid ; while ( l < h ) {
mid = l + ( ( h - l ) >> 1 ) ; ( r > arr [ mid ] ) ? ( l = mid + 1 ) : ( h = mid ) ; } return ( arr [ l ] >= r ) ? l : -1 ; }
int myRand ( int arr [ ] , int freq [ ] , int n ) {
int prefix [ n ] , i ; prefix [ 0 ] = freq [ 0 ] ; for ( i = 1 ; i < n ; ++ i ) prefix [ i ] = prefix [ i - 1 ] + freq [ i ] ;
int r = ( rand ( ) % prefix [ n - 1 ] ) + 1 ;
int indexc = findCeil ( prefix , r , 0 , n - 1 ) ; return arr [ indexc ] ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int freq [ ] = { 10 , 5 , 20 , 100 } ; int i , n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
for ( i = 0 ; i < 5 ; i ++ ) printf ( " % d STRNEWLINE " , myRand ( arr , freq , n ) ) ; return 0 ; }
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
#include <stdio.h> NEW_LINE #define CHAR_BIT  8
int min ( int x , int y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( sizeof ( int ) * CHAR_BIT - 1 ) ) ) ; }
int smallest ( int x , int y , int z ) { return min ( x , min ( y , z ) ) ; }
int main ( ) { int x = 12 , y = 15 , z = 5 ; printf ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ % d " , smallest ( x , y , z ) ) ; return 0 ; }
int smallest ( int x , int y , int z ) {
if ( ! ( y / x ) ) return ( ! ( y / z ) ) ? y : z ; return ( ! ( x / z ) ) ? x : z ; }
int main ( ) { int x = 78 , y = 88 , z = 68 ; printf ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ % d " , smallest ( x , y , z ) ) ; return 0 ; }
void changeToZero ( int a [ 2 ] ) { a [ a [ 1 ] ] = a [ ! a [ 1 ] ] ; }
int main ( ) { int a [ ] = { 1 , 0 } ; changeToZero ( a ) ; printf ( " ▁ arr [ 0 ] ▁ = ▁ % d ▁ STRNEWLINE " , a [ 0 ] ) ; printf ( " ▁ arr [ 1 ] ▁ = ▁ % d ▁ " , a [ 1 ] ) ; getchar ( ) ; return 0 ; }
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
#include <stdio.h> NEW_LINE #define bool  int NEW_LINE bool isPowerOfFour ( unsigned int n ) { return n != 0 && ( ( n & ( n - 1 ) ) == 0 ) && ! ( n & 0xAAAAAAAA ) ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) printf ( " % d ▁ is ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; getchar ( ) ; }
int min ( int x , int y ) { return y ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int max ( int x , int y ) { return x ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int main ( ) { int x = 15 ; int y = 6 ; printf ( " Minimum ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ " , x , y ) ; printf ( " % d " , min ( x , y ) ) ; printf ( " Maximum of % d and % d is " printf ( " % d " , max ( x , y ) ) ; getchar ( ) ; }
#include <stdio.h> NEW_LINE #define CHAR_BIT  8
int min ( int x , int y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( sizeof ( int ) * CHAR_BIT - 1 ) ) ) ; }
int max ( int x , int y ) { return x - ( ( x - y ) & ( ( x - y ) >> ( sizeof ( int ) * CHAR_BIT - 1 ) ) ) ; }
int main ( ) { int x = 15 ; int y = 6 ; printf ( " Minimum ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ " , x , y ) ; printf ( " % d " , min ( x , y ) ) ; printf ( " Maximum of % d and % d is " printf ( " % d " , max ( x , y ) ) ; getchar ( ) ; }
#include <math.h> NEW_LINE #include <stdio.h> NEW_LINE unsigned int getFirstSetBitPos ( int n ) { return log2 ( n & - n ) + 1 ; }
int main ( ) { int n = 12 ; printf ( " % u " , getFirstSetBitPos ( n ) ) ; getchar ( ) ; return 0 ; }
void bin ( unsigned n ) { unsigned i ; for ( i = 1 << 31 ; i > 0 ; i = i / 2 ) ( n & i ) ? printf ( "1" ) : printf ( "0" ) ; }
int main ( void ) { bin ( 7 ) ; printf ( " STRNEWLINE " ) ; bin ( 4 ) ; }
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
void constructLowerArray ( int * arr [ ] , int * countSmaller , int n ) { int i , j ;
for ( i = 0 ; i < n ; i ++ ) countSmaller [ i ] = 0 ; for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] < arr [ i ] ) countSmaller [ i ] ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 12 , 10 , 5 , 4 , 2 , 20 , 6 , 1 , 0 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int * low = ( int * ) malloc ( sizeof ( int ) * n ) ; constructLowerArray ( arr , low , n ) ; printArray ( low , n ) ; return 0 ; }
int segregate ( int arr [ ] , int size ) { int j = 0 , i ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] <= 0 ) { swap ( & arr [ i ] , & arr [ j ] ) ;
j ++ ; } } return j ; }
int findMissingPositive ( int arr [ ] , int size ) { int i ;
for ( i = 0 ; i < size ; i ++ ) { if ( abs ( arr [ i ] ) - 1 < size && arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; }
for ( i = 0 ; i < size ; i ++ ) if ( arr [ i ] > 0 )
return i + 1 ; return size + 1 ; }
int findMissing ( int arr [ ] , int size ) {
int shift = segregate ( arr , size ) ;
return findMissingPositive ( arr + shift , size - shift ) ; }
int main ( ) { int arr [ ] = { 0 , 10 , 2 , -10 , -20 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int missing = findMissing ( arr , arr_size ) ; printf ( " The ▁ smallest ▁ positive ▁ missing ▁ number ▁ is ▁ % d ▁ " , missing ) ; getchar ( ) ; return 0 ; }
int getMissingNo ( int a [ ] , int n ) { int i , total ; total = ( n + 1 ) * ( n + 2 ) / 2 ; for ( i = 0 ; i < n ; i ++ ) total -= a [ i ] ; return total ; }
int main ( ) { int a [ ] = { 1 , 2 , 4 , 5 , 6 } ; int miss = getMissingNo ( a , 5 ) ; printf ( " % d " , miss ) ; getchar ( ) ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE void printTwoElements ( int arr [ ] , int size ) { int i ; printf ( " The repeating element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } printf ( " and the missing element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) printf ( " % d " , i + 1 ) ; } }
int main ( ) { int arr [ ] = { 7 , 3 , 4 , 5 , 5 , 6 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoElements ( arr , n ) ; return 0 ; }
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
struct Node { int key ; struct Node * next ; } ;
void push ( struct Node * * head_ref , int new_key ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> key = new_key ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
bool search ( struct Node * head , int x ) {
if ( head == NULL ) return false ;
if ( head -> key == x ) return true ;
return search ( head -> next , x ) ; }
push ( & head , 10 ) ; push ( & head , 30 ) ; push ( & head , 11 ) ; push ( & head , 21 ) ; push ( & head , 14 ) ; search ( head , 21 ) ? printf ( " Yes " ) : printf ( " No " ) ; return 0 ; }
void deleteAlt ( struct Node * head ) { if ( head == NULL ) return ; struct Node * node = head -> next ; if ( node == NULL ) return ;
head -> next = node -> next ;
deleteAlt ( head -> next ) ; }
void AlternatingSplit ( struct Node * source , struct Node * * aRef , struct Node * * bRef ) { struct Node aDummy ; struct Node * aTail = & aDummy ;
struct Node bDummy ; struct Node * bTail = & bDummy ;
struct Node * current = source ; aDummy . next = NULL ; bDummy . next = NULL ; while ( current != NULL ) { MoveNode ( & ( aTail -> next ) , t ) ;
aTail = aTail -> next ;
if ( current != NULL ) { MoveNode ( & ( bTail -> next ) , t ) ; bTail = bTail -> next ; } } * aRef = aDummy . next ; * bRef = bDummy . next ; }
bool areIdentical ( struct Node * a , struct Node * b ) {
if ( a == NULL && b == NULL ) return true ;
if ( a != NULL && b != NULL ) return ( a -> data == b -> data ) && areIdentical ( a -> next , b -> next ) ;
return false ; }
struct Node { int data ; struct Node * next ; } ;
int count [ 3 ] = { 0 , 0 , 0 } ; struct Node * ptr = head ;
while ( ptr != NULL ) { count [ ptr -> data ] += 1 ; ptr = ptr -> next ; } int i = 0 ; ptr = head ;
while ( ptr != NULL ) { if ( count [ i ] == 0 ) ++ i ; else { ptr -> data = i ; -- count [ i ] ; ptr = ptr -> next ; } } }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( struct Node * node ) { while ( node != NULL ) { printf ( " % d ▁ " , node -> data ) ; node = node -> next ; } printf ( " n " ) ; }
struct Node * head = NULL ; push ( & head , 0 ) ; push ( & head , 1 ) ; push ( & head , 0 ) ; push ( & head , 2 ) ; push ( & head , 1 ) ; push ( & head , 1 ) ; push ( & head , 2 ) ; push ( & head , 1 ) ; push ( & head , 2 ) ; printf ( " Linked ▁ List ▁ Before ▁ Sorting STRNEWLINE " ) ; printList ( head ) ; sortList ( head ) ; printf ( " Linked ▁ List ▁ After ▁ Sorting STRNEWLINE " ) ; printList ( head ) ; return 0 ; }
struct List { int data ; struct List * next ; struct List * child ; } ;
struct Node { int data ; struct Node * next ; } ;
Node * newNode ( int key ) { Node * temp = new Node ; temp -> data = key ; temp -> next = NULL ; return temp ; }
Node * rearrangeEvenOdd ( Node * head ) {
if ( head == NULL ) return NULL ;
Node * odd = head ; Node * even = head -> next ;
Node * evenFirst = even ; while ( 1 ) {
if ( ! odd || ! even || ! ( even -> next ) ) { odd -> next = evenFirst ; break ; }
odd -> next = even -> next ; odd = even -> next ;
if ( odd -> next == NULL ) { even -> next = NULL ; odd -> next = evenFirst ; break ; }
even -> next = odd -> next ; even = odd -> next ; } return head ; }
void printlist ( Node * node ) { while ( node != NULL ) { cout << node -> data << " - > " ; node = node -> next ; } cout << " NULL " << endl ; }
int main ( void ) { Node * head = newNode ( 1 ) ; head -> next = newNode ( 2 ) ; head -> next -> next = newNode ( 3 ) ; head -> next -> next -> next = newNode ( 4 ) ; head -> next -> next -> next -> next = newNode ( 5 ) ; cout << " Given ▁ Linked ▁ List STRNEWLINE " ; printlist ( head ) ; head = rearrangeEvenOdd ( head ) ; cout << " Modified Linked List " ; printlist ( head ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
void deleteLast ( struct Node * head , int x ) { struct Node * temp = head , * ptr = NULL ; while ( temp ) {
if ( temp -> data == x ) ptr = temp ; temp = temp -> next ; }
if ( ptr != NULL && ptr -> next == NULL ) { temp = head ; while ( temp -> next != ptr ) temp = temp -> next ; temp -> next = NULL ; }
if ( ptr != NULL && ptr -> next != NULL ) { ptr -> data = ptr -> next -> data ; temp = ptr -> next ; ptr -> next = ptr -> next -> next ; free ( temp ) ; } }
struct Node * newNode ( int x ) { struct Node * node = malloc ( sizeof ( struct Node * ) ) ; node -> data = x ; node -> next = NULL ; return node ; }
void display ( struct Node * head ) { struct Node * temp = head ; if ( head == NULL ) { printf ( " NULL STRNEWLINE " ) ; return ; } while ( temp != NULL ) { printf ( " % d ▁ - - > ▁ " , temp -> data ) ; temp = temp -> next ; } printf ( " NULL STRNEWLINE " ) ; }
int main ( ) { struct Node * head = newNode ( 1 ) ; head -> next = newNode ( 2 ) ; head -> next -> next = newNode ( 3 ) ; head -> next -> next -> next = newNode ( 4 ) ; head -> next -> next -> next -> next = newNode ( 5 ) ; head -> next -> next -> next -> next -> next = newNode ( 4 ) ; head -> next -> next -> next -> next -> next -> next = newNode ( 4 ) ; printf ( " Created ▁ Linked ▁ list : ▁ " ) ; display ( head ) ; deleteLast ( head , 4 ) ; printf ( " List ▁ after ▁ deletion ▁ of ▁ 4 : ▁ " ) ; display ( head ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
int LinkedListLength ( struct Node * head ) { while ( head && head -> next ) { head = head -> next -> next ; } if ( ! head ) return 0 ; return 1 ; }
void push ( struct Node * * head , int info ) {
struct Node * node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ;
node -> data = info ;
node -> next = ( * head ) ;
( * head ) = node ; }
int main ( void ) { struct Node * head = NULL ;
push ( & head , 4 ) ; push ( & head , 5 ) ; push ( & head , 7 ) ; push ( & head , 2 ) ; push ( & head , 9 ) ; push ( & head , 6 ) ; push ( & head , 1 ) ; push ( & head , 2 ) ; push ( & head , 0 ) ; push ( & head , 5 ) ; push ( & head , 5 ) ; int check = LinkedListLength ( head ) ;
if ( check == 0 ) { printf ( " Even STRNEWLINE " ) ; } else { printf ( " Odd STRNEWLINE " ) ; } return 0 ; }
struct Node * SortedMerge ( struct Node * a , struct Node * b ) { struct Node * result = NULL ;
struct Node * * lastPtrRef = & result ; while ( 1 ) { if ( a == NULL ) { * lastPtrRef = b ; break ; } else if ( b == NULL ) { * lastPtrRef = a ; break ; } if ( a -> data <= b -> data ) { MoveNode ( lastPtrRef , & a ) ; } else { MoveNode ( lastPtrRef , & b ) ; }
lastPtrRef = & ( ( * lastPtrRef ) -> next ) ; } return ( result ) ; }
struct Node { int data ; struct Node * next ; } ;
void setMiddleHead ( struct Node * * head ) { if ( * head == NULL ) return ;
struct Node * one_node = ( * head ) ;
struct Node * two_node = ( * head ) ;
struct Node * prev = NULL ; while ( two_node != NULL && two_node -> next != NULL ) {
prev = one_node ;
two_node = two_node -> next -> next ;
one_node = one_node -> next ; }
prev -> next = prev -> next -> next ; one_node -> next = ( * head ) ; ( * head ) = one_node ; }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( struct Node * ptr ) { while ( ptr != NULL ) { printf ( " % d ▁ " , ptr -> data ) ; ptr = ptr -> next ; } printf ( " STRNEWLINE " ) ; }
struct Node * head = NULL ; int i ; for ( i = 5 ; i > 0 ; i -- ) push ( & head , i ) ; printf ( " ▁ list ▁ before : ▁ " ) ; printList ( head ) ; setMiddleHead ( & head ) ; printf ( " ▁ list ▁ After : ▁ " ) ; printList ( head ) ; return 0 ; }
void insertAfter ( struct Node * prev_node , int new_data ) {
if ( prev_node == NULL ) { printf ( " the ▁ given ▁ previous ▁ node ▁ cannot ▁ be ▁ NULL " ) ; return ; }
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = prev_node -> next ;
prev_node -> next = new_node ;
new_node -> prev = prev_node ;
if ( new_node -> next != NULL ) new_node -> next -> prev = new_node ; }
struct node { int data ; struct node * left ; struct node * right ; } ; void printKDistant ( struct node * root , int k ) { if ( root == NULL k < 0 ) return ; if ( k == 0 ) { printf ( " % d ▁ " , root -> data ) ; return ; } printKDistant ( root -> left , k - 1 ) ; printKDistant ( root -> right , k - 1 ) ; }
struct node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> left = newNode ( 8 ) ; printKDistant ( root , 2 ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <malloc.h> NEW_LINE #define COUNT  10
struct Node { int data ; struct Node * left , * right ; } ;
struct Node * newNode ( int data ) { struct Node * node = malloc ( sizeof ( struct Node ) ) ; node -> data = data ; node -> left = node -> right = NULL ; return node ; }
void print2DUtil ( struct Node * root , int space ) {
if ( root == NULL ) return ;
space += COUNT ;
print2DUtil ( root -> right , space ) ;
printf ( " STRNEWLINE " ) ; for ( int i = COUNT ; i < space ; i ++ ) printf ( " ▁ " ) ; printf ( " % d STRNEWLINE " , root -> data ) ;
print2DUtil ( root -> left , space ) ; }
void print2D ( struct Node * root ) {
print2DUtil ( root , 0 ) ; }
int main ( ) { struct Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> left = newNode ( 6 ) ; root -> right -> right = newNode ( 7 ) ; root -> left -> left -> left = newNode ( 8 ) ; root -> left -> left -> right = newNode ( 9 ) ; root -> left -> right -> left = newNode ( 10 ) ; root -> left -> right -> right = newNode ( 11 ) ; root -> right -> left -> left = newNode ( 12 ) ; root -> right -> left -> right = newNode ( 13 ) ; root -> right -> right -> left = newNode ( 14 ) ; root -> right -> right -> right = newNode ( 15 ) ; print2D ( root ) ; return 0 ; }
struct node * newNode ( int item ) { struct node * temp = ( struct node * ) malloc ( sizeof ( struct node ) ) ; temp -> data = item ; temp -> left = temp -> right = NULL ; return temp ; }
void leftViewUtil ( struct node * root , int level , int * max_level ) {
if ( root == NULL ) return ;
if ( * max_level < level ) { printf ( " % d TABSYMBOL " , root -> data ) ; * max_level = level ; }
leftViewUtil ( root -> left , level + 1 , max_level ) ; leftViewUtil ( root -> right , level + 1 , max_level ) ; }
void leftView ( struct node * root ) { int max_level = 0 ; leftViewUtil ( root , 1 , & max_level ) ; }
int main ( ) { struct node * root = newNode ( 12 ) ; root -> left = newNode ( 10 ) ; root -> right = newNode ( 30 ) ; root -> right -> left = newNode ( 25 ) ; root -> right -> right = newNode ( 40 ) ; leftView ( root ) ; return 0 ; }
int cntRotations ( char s [ ] , int n ) { int lh = 0 , rh = 0 , i , ans = 0 ;
for ( i = 0 ; i < n / 2 ; ++ i ) if ( s [ i ] == ' a ' s [ i ] == ' e ' s [ i ] == ' i ' s [ i ] == ' o ' s [ i ] == ' u ' ) { lh ++ ; }
for ( i = n / 2 ; i < n ; ++ i ) if ( s [ i ] == ' a ' s [ i ] == ' e ' s [ i ] == ' i ' s [ i ] == ' o ' s [ i ] == ' u ' ) { rh ++ ; }
if ( lh > rh ) ans ++ ;
for ( i = 1 ; i < n ; ++ i ) { if ( s [ i - 1 ] == ' a ' s [ i - 1 ] == ' e ' s [ i - 1 ] == ' i ' s [ i - 1 ] == ' o ' s [ i - 1 ] == ' u ' ) { rh ++ ; lh -- ; } if ( s [ ( i - 1 + n / 2 ) % n ] == ' a ' || s [ ( i - 1 + n / 2 ) % n ] == ' e ' || s [ ( i - 1 + n / 2 ) % n ] == ' i ' || s [ ( i - 1 + n / 2 ) % n ] == ' o ' || s [ ( i - 1 + n / 2 ) % n ] == ' u ' ) { rh -- ; lh ++ ; } if ( lh > rh ) ans ++ ; }
return ans ; }
int main ( ) { char s [ ] = " abecidft " ; int n = strlen ( s ) ;
printf ( " % d " , cntRotations ( s , n ) ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
struct Node * rotateHelper ( struct Node * blockHead , struct Node * blockTail , int d , struct Node * * tail , int k ) { if ( d == 0 ) return blockHead ;
if ( d > 0 ) { struct Node * temp = blockHead ; for ( int i = 1 ; temp -> next -> next && i < k - 1 ; i ++ ) temp = temp -> next ; blockTail -> next = blockHead ; * tail = temp ; return rotateHelper ( blockTail , temp , d - 1 , tail , k ) ; }
if ( d < 0 ) { blockTail -> next = blockHead ; * tail = blockHead ; return rotateHelper ( blockHead -> next , blockHead , d + 1 , tail , k ) ; } }
struct Node * rotateByBlocks ( struct Node * head , int k , int d ) {
if ( ! head ! head -> next ) return head ;
if ( d == 0 ) return head ; struct Node * temp = head , * tail = NULL ;
int i ; for ( i = 1 ; temp -> next && i < k ; i ++ ) temp = temp -> next ;
struct Node * nextBlock = temp -> next ;
if ( i < k ) head = rotateHelper ( head , temp , d % k , & tail , i ) ; else head = rotateHelper ( head , temp , d % k , & tail , k ) ;
tail -> next = rotateByBlocks ( nextBlock , k , d % k ) ;
return head ; }
void push ( struct Node * * head_ref , int new_data ) { struct Node * new_node = new Node ; new_node -> data = new_data ; new_node -> next = ( * head_ref ) ; ( * head_ref ) = new_node ; }
void printList ( struct Node * node ) { while ( node != NULL ) { printf ( " % d ▁ " , node -> data ) ; node = node -> next ; } }
struct Node * head = NULL ;
for ( int i = 9 ; i > 0 ; i -= 1 ) push ( & head , i ) ; printf ( " Given ▁ linked ▁ list ▁ STRNEWLINE " ) ; printList ( head ) ;
int k = 3 , d = 2 ; head = rotateByBlocks ( head , k , d ) ; printf ( " Rotated by blocks Linked list " printList ( head ) ; return ( 0 ) ; }
void DeleteFirst ( struct Node * * head ) { struct Node * previous = * head , * firstNode = * head ;
if ( * head == NULL ) { printf ( " List is empty " return ; }
if ( previous -> next == previous ) { * head = NULL ; return ; }
while ( previous -> next != * head ) { previous = previous -> next ; }
previous -> next = firstNode -> next ;
* head = previous -> next ; free ( firstNode ) ; return ; }
void DeleteLast ( struct Node * * head ) { struct Node * current = * head , * temp = * head , * previous ;
if ( * head == NULL ) { printf ( " List is empty " return ; }
if ( current -> next == current ) { * head = NULL ; return ; }
while ( current -> next != * head ) { previous = current ; current = current -> next ; } previous -> next = current -> next ; * head = previous -> next ; free ( current ) ; return ; }
void countSubarrays ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int sum = 0 ; for ( int j = i ; j < n ; j ++ ) {
if ( ( j - i ) % 2 == 0 ) sum += arr [ j ] ;
else sum -= arr [ j ] ;
if ( sum == 0 ) count ++ ; } }
printf ( " % d " , count ) ; }
int arr [ ] = { 2 , 4 , 6 , 4 , 2 } ;
int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
countSubarrays ( arr , n ) ; return 0 ; }
void printAlter ( int arr [ ] , int N ) {
for ( int currIndex = 0 ; currIndex < N ; currIndex ++ ) {
if ( currIndex % 2 == 0 ) { printf ( " % d ▁ " , arr [ currIndex ] ) ; } } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printAlter ( arr , N ) ; }
void reverse ( int arr [ ] , int start , int end ) {
int mid = ( end - start + 1 ) / 2 ;
for ( int i = 0 ; i < mid ; i ++ ) {
int temp = arr [ start + i ] ;
arr [ start + i ] = arr [ end - i ] ;
arr [ end - i ] = temp ; } return ; }
void shuffleArrayUtil ( int arr [ ] , int start , int end ) { int i ;
int l = end - start + 1 ;
if ( l == 2 ) return ;
int mid = start + l / 2 ;
if ( l % 4 ) {
mid -= 1 ; }
int mid1 = start + ( mid - start ) / 2 ; int mid2 = mid + ( end + 1 - mid ) / 2 ;
reverse ( arr , mid1 , mid2 - 1 ) ;
reverse ( arr , mid1 , mid - 1 ) ;
reverse ( arr , mid , mid2 - 1 ) ;
shuffleArrayUtil ( arr , start , mid - 1 ) ; shuffleArrayUtil ( arr , mid , end ) ; }
void shuffleArray ( int arr [ ] , int N , int start , int end ) {
shuffleArrayUtil ( arr , start , end ) ;
for ( int i = 0 ; i < N ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; }
int arr [ ] = { 1 , 3 , 5 , 2 , 4 , 6 } ;
int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
shuffleArray ( arr , N , 0 , N - 1 ) ; return 0 ; }
int canMadeEqual ( int A [ ] , int B [ ] , int n ) { int i ;
sort ( A , n ) ; sort ( B , n ) ;
for ( i = 0 ; i < n ; i ++ ) { if ( A [ i ] != B [ i ] ) { return ( 0 ) ; } } return ( 1 ) ; }
int main ( ) { int A [ ] = { 1 , 2 , 3 } ; int n ; int B [ ] = { 1 , 3 , 2 } ; n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; if ( canMadeEqual ( A , B , n ) ) { printf ( " Yes " ) ; } else { printf ( " No " ) ; } return 0 ; }
void merge ( int arr [ ] , int start , int mid , int end ) { int start2 = mid + 1 ;
if ( arr [ mid ] <= arr [ start2 ] ) { return ; }
while ( start <= mid && start2 <= end ) {
if ( arr [ start ] <= arr [ start2 ] ) { start ++ ; } else { int value = arr [ start2 ] ; int index = start2 ;
while ( index != start ) { arr [ index ] = arr [ index - 1 ] ; index -- ; } arr [ start ] = value ;
start ++ ; mid ++ ; start2 ++ ; } } }
void mergeSort ( int arr [ ] , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
mergeSort ( arr , l , m ) ; mergeSort ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } }
void printArray ( int A [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , A [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 , 7 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; mergeSort ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ; return 0 ; }
#include <stdio.h>
void constGraphWithCon ( int N , int K ) {
int Max = ( ( N - 1 ) * ( N - 2 ) ) / 2 ;
if ( K > Max ) { printf ( " - 1" ) ; return ; }
int count = 0 ;
for ( int i = 1 ; i < N ; i ++ ) { for ( int j = i + 1 ; j <= N ; j ++ ) { printf ( " % d ▁ % d STRNEWLINE " , i , j ) ;
count ++ ; if ( count == N * ( N - 1 ) / 2 - K ) break ; } if ( count == N * ( N - 1 ) / 2 - K ) break ; } }
int main ( ) { int N = 5 , K = 3 ; constGraphWithCon ( N , K ) ; return 0 ; }
void findArray ( int N , int K ) {
if ( N == 1 ) { printf ( " % d " , K ) ; return ; } if ( N == 2 ) { printf ( " % d ▁ % d " , 0 , K ) ; return ; }
int P = N - 2 ; int Q = N - 1 ;
int VAL = 0 ;
for ( int i = 1 ; i <= ( N - 3 ) ; i ++ ) { printf ( " % d ▁ " , i ) ;
VAL ^= i ; } if ( VAL == K ) { printf ( " % d ▁ % d ▁ % d " , P , Q , P ^ Q ) ; } else { printf ( " % d ▁ % d ▁ % d " , 0 , P , P ^ K ^ VAL ) ; } }
int main ( ) { int N = 4 , X = 6 ;
findArray ( N , X ) ; return 0 ; }
int countDigitSum ( int N , int K ) {
int l = ( int ) pow ( 10 , N - 1 ) , r = ( int ) pow ( 10 , N ) - 1 ; int count = 0 ; for ( int i = l ; i <= r ; i ++ ) { int num = i ;
int digits [ N ] ; for ( int j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num /= 10 ; } int sum = 0 , flag = 0 ;
for ( int j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( int j = 1 ; j < N - K + 1 ; j ++ ) { int curr_sum = 0 ; for ( int m = j ; m < j + K ; m ++ ) curr_sum += digits [ m ] ;
if ( sum != curr_sum ) { flag = 1 ; break ; } }
if ( flag == 0 ) { count ++ ; } } return count ; }
int N = 2 , K = 1 ;
printf ( " % d " , countDigitSum ( N , K ) ) ; return 0 ; }
bool arithmeticThree ( int set [ ] , int n ) {
for ( int j = 1 ; j < n - 1 ; j ++ ) {
int i = j - 1 , k = j + 1 ;
while ( i >= 0 && k <= n - 1 ) { if ( set [ i ] + set [ k ] == 2 * set [ j ] ) return true ; ( set [ i ] + set [ k ] < 2 * set [ j ] ) ? k ++ : i -- ; } } return false ; }
int maxSumIS ( int arr [ ] , int n ) { int i , j , max = 0 ; int msis [ n ] ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 1 , 101 , 2 , 3 , 100 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ " " subsequence ▁ is ▁ % d STRNEWLINE " , maxSumIS ( arr , n ) ) ; return 0 ; }
void reverse ( char str [ ] , int start , int end ) {
char temp ; while ( start <= end ) {
temp = str [ start ] ; str [ start ] = str [ end ] ; str [ end ] = temp ; start ++ ; end -- ; } }
void reverseletter ( char str [ ] , int start , int end ) { int wstart , wend ; for ( wstart = wend = start ; wend < end ; wend ++ ) { if ( str [ wend ] == ' ▁ ' ) continue ;
while ( str [ wend ] != ' ▁ ' && wend <= end ) wend ++ ; wend -- ;
reverse ( str , wstart , wend ) ; } }
int main ( ) { char str [ 1000 ] = " Ashish ▁ Yadav ▁ Abhishek ▁ Rajput ▁ Sunil ▁ Pundir " ; reverseletter ( str , 0 , strlen ( str ) - 1 ) ; printf ( " % s " , str ) ; return 0 ; }
#include <stdbool.h> NEW_LINE #include <stdio.h> NEW_LINE #include <string.h> NEW_LINE int min ( int a , int b ) { return a < b ? a : b ; } bool have_same_frequency ( int freq [ ] , int k ) { for ( int i = 0 ; i < 26 ; i ++ ) { if ( freq [ i ] != 0 && freq [ i ] != k ) { return false ; } } return true ; } int count_substrings ( char * s , int n , int k ) { int count = 0 ; int distinct = 0 ; bool have [ 26 ] = { false } ; for ( int i = 0 ; i < n ; i ++ ) { have [ s [ i ] - ' a ' ] = true ; } for ( int i = 0 ; i < 26 ; i ++ ) { if ( have [ i ] ) { distinct ++ ; } } for ( int length = 1 ; length <= distinct ; length ++ ) { int window_length = length * k ; int freq [ 26 ] = { 0 } ; int window_start = 0 ; int window_end = window_start + window_length - 1 ; for ( int i = window_start ; i <= min ( window_end , n - 1 ) ; i ++ ) { freq [ s [ i ] - ' a ' ] ++ ; } while ( window_end < n ) { if ( have_same_frequency ( freq , k ) ) { count ++ ; } freq [ s [ window_start ] - ' a ' ] -- ; window_start ++ ; window_end ++ ; if ( window_end < n ) { freq [ s [ window_end ] - ' a ' ] ++ ; } } } return count ; } int main ( ) { char * s = " aabbcc " ; int k = 2 ; printf ( " % d STRNEWLINE " , count_substrings ( s , 6 , k ) ) ; s = " aabbc " ; k = 2 ; printf ( " % d STRNEWLINE " , count_substrings ( s , 5 , k ) ) ; return 0 ; }
#include <stdio.h>
char * toggleCase ( char * a ) { for ( int i = 0 ; a [ i ] != ' \0' ; i ++ ) {
a [ i ] ^= 32 ; } return a ; }
int main ( ) { char str [ ] = " CheRrY " ; printf ( " Toggle ▁ case : ▁ % s STRNEWLINE " , toggleCase ( str ) ) ; printf ( " Original ▁ string : ▁ % s " , toggleCase ( str ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define NO_OF_CHARS  256
bool areAnagram ( char * str1 , char * str2 ) {
int count1 [ NO_OF_CHARS ] = { 0 } ; int count2 [ NO_OF_CHARS ] = { 0 } ; int i ;
for ( i = 0 ; str1 [ i ] && str2 [ i ] ; i ++ ) { count1 [ str1 [ i ] ] ++ ; count2 [ str2 [ i ] ] ++ ; }
if ( str1 [ i ] str2 [ i ] ) return false ;
for ( i = 0 ; i < NO_OF_CHARS ; i ++ ) if ( count1 [ i ] != count2 [ i ] ) return false ; return true ; }
int main ( ) { char str1 [ ] = " geeksforgeeks " ; char str2 [ ] = " forgeeksgeeks " ;
if ( areAnagram ( str1 , str2 ) ) printf ( " The ▁ two ▁ strings ▁ are ▁ anagram ▁ of ▁ each ▁ other " ) ; else printf ( " The ▁ two ▁ strings ▁ are ▁ not ▁ anagram ▁ of ▁ each ▁ " " other " ) ; return 0 ; }
int heptacontagonNum ( int n ) { return ( 68 * n * n - 66 * n ) / 2 ; }
int main ( ) { int N = 3 ; printf ( "3rd ▁ heptacontagon ▁ Number ▁ is ▁ = ▁ % d " , heptacontagonNum ( N ) ) ; return 0 ; }
void isEqualFactors ( lli N ) { if ( ( N % 2 == 0 ) && ( N % 4 != 0 ) ) printf ( " YES STRNEWLINE " ) ; else printf ( " NO STRNEWLINE " ) ; }
int main ( ) { lli N = 10 ; isEqualFactors ( N ) ; N = 125 ; isEqualFactors ( N ) ; return 0 ; }
int checkDivisibility ( int n , int digit ) {
return ( digit != 0 && n % digit == 0 ) ; }
int isAllDigitsDivide ( int n ) { int temp = n ; while ( temp > 0 ) {
int digit = temp % 10 ; if ( ! ( checkDivisibility ( n , digit ) ) ) return 0 ; temp /= 10 ; } return 1 ; }
int isAllDigitsDistinct ( int n ) {
int arr [ 10 ] , i , digit ; for ( i = 0 ; i < 10 ; i ++ ) arr [ i ] = 0 ;
while ( n > 0 ) {
digit = n % 10 ;
if ( arr [ digit ] ) return 0 ;
arr [ digit ] = 1 ;
n = n / 10 ; } return 1 ; }
int isLynchBell ( int n ) { return isAllDigitsDivide ( n ) && isAllDigitsDistinct ( n ) ; }
int N = 12 ;
if ( isLynchBell ( N ) ) printf ( " Yes " ) ; else printf ( " No " ) ; return 0 ; }
int maximumAND ( int L , int R ) { return R ; }
int main ( ) { int l = 3 ; int r = 7 ; printf ( " % d " , maximumAND ( l , r ) ) ; return 0 ; }
double findAverageOfCube ( int n ) {
double sum = 0 ;
int i ; for ( i = 1 ; i <= n ; i ++ ) { sum += i * i * i ; }
return sum / n ; }
int n = 3 ;
printf ( " % lf " , findAverageOfCube ( n ) ) ; return 0 ; }
_Bool isPower ( int N , int K ) {
int res1 = log ( N ) / log ( K ) ; double res2 = log ( N ) / log ( K ) ;
return ( res1 == res2 ) ; }
int main ( ) { int N = 8 ; int K = 2 ; if ( isPower ( N , K ) ) { printf ( " Yes " ) ; } else { printf ( " No " ) ; } return 0 ; }
float y ( float x ) { return ( 1 / ( 1 + x ) ) ; }
float BooleRule ( float a , float b ) {
int n = 4 ; int h ;
h = ( ( b - a ) / n ) ; float sum = 0 ;
float bl = ( 7 * y ( a ) + 32 * y ( a + h ) + 12 * y ( a + 2 * h ) + 32 * y ( a + 3 * h ) + 7 * y ( a + 4 * h ) ) * 2 * h / 45 ; sum = sum + bl ; return sum ; }
int main ( ) { float lowlimit = 0 ; float upplimit = 4 ; printf ( " f ( x ) ▁ = ▁ % .4f " , BooleRule ( 0 , 4 ) ) ; return 0 ; }
float y ( float x ) { float num = 1 ; float denom = 1.0 + x * x ; return num / denom ; }
float WeedleRule ( float a , float b ) {
double h = ( b - a ) / 6 ;
float sum = 0 ;
sum = sum + ( ( ( 3 * h ) / 10 ) * ( y ( a ) + y ( a + 2 * h ) + 5 * y ( a + h ) + 6 * y ( a + 3 * h ) + y ( a + 4 * h ) + 5 * y ( a + 5 * h ) + y ( a + 6 * h ) ) ) ;
return sum ; }
float a = 0 , b = 6 ;
printf ( " f ( x ) ▁ = ▁ % f " , WeedleRule ( a , b ) ) ; return 0 ; }
float dydx ( float x , float y ) { return ( x + y - 2 ) ; }
float rungeKutta ( float x0 , float y0 , float x , float h ) {
int n = ( int ) ( ( x - x0 ) / h ) ; float k1 , k2 ;
float y = y0 ; for ( int i = 1 ; i <= n ; i ++ ) {
k1 = h * dydx ( x0 , y ) ; k2 = h * dydx ( x0 + 0.5 * h , y + 0.5 * k1 ) ;
y = y + ( 1.0 / 6.0 ) * ( k1 + 2 * k2 ) ;
x0 = x0 + h ; } return y ; }
int main ( ) { float x0 = 0 , y = 1 , x = 2 , h = 0.2 ; printf ( " y ( x ) ▁ = ▁ % f " , rungeKutta ( x0 , y , x , h ) ) ; return 0 ; }
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
void addAP ( int A [ ] , int Q , int operations [ 2 ] [ 4 ] ) {
for ( int j = 0 ; j < 2 ; ++ j ) { int L = operations [ j ] [ 0 ] , R = operations [ j ] [ 1 ] , a = operations [ j ] [ 2 ] , d = operations [ j ] [ 3 ] ; int curr = a ;
for ( int i = L - 1 ; i < R ; i ++ ) {
A [ i ] += curr ;
curr += d ; } }
for ( int i = 0 ; i < 4 ; ++ i ) printf ( " % d ▁ " , A [ i ] ) ; }
int main ( ) { int A [ ] = { 5 , 4 , 2 , 8 } ; int Q = 2 ; int Query [ 2 ] [ 4 ] = { { 1 , 2 , 1 , 3 } , { 1 , 4 , 4 , 1 } } ;
addAP ( A , Q , Query ) ; return 0 ; }
#include <math.h> NEW_LINE #include <stdio.h> NEW_LINE int log_a_to_base_b ( int a , int b ) { return log ( a ) / log ( b ) ; }
int main ( ) { int a = 3 ; int b = 2 ; printf ( " % d STRNEWLINE " , log_a_to_base_b ( a , b ) ) ; a = 256 ; b = 4 ; printf ( " % d STRNEWLINE " , log_a_to_base_b ( a , b ) ) ; return 0 ; }
int log_a_to_base_b ( int a , int b ) { return ( a > b - 1 ) ? 1 + log_a_to_base_b ( a / b , b ) : 0 ; }
int main ( ) { int a = 3 ; int b = 2 ; printf ( " % d STRNEWLINE " , log_a_to_base_b ( a , b ) ) ; a = 256 ; b = 4 ; printf ( " % d STRNEWLINE " , log_a_to_base_b ( a , b ) ) ; return 0 ; }
int maximum ( int x , int y ) { return ( ( x + y + abs ( x - y ) ) / 2 ) ; }
int minimum ( int x , int y ) { return ( ( x + y - abs ( x - y ) ) / 2 ) ; }
void main ( ) { int x = 99 , y = 18 ;
printf ( " Maximum : ▁ % d STRNEWLINE " , maximum ( x , y ) ) ;
printf ( " Minimum : ▁ % d STRNEWLINE " , minimum ( x , y ) ) ; }
double e ( int x , int n ) { static double p = 1 , f = 1 ; double r ;
if ( n == 0 ) return 1 ;
r = e ( x , n - 1 ) ;
p = p * x ;
f = f * n ; return ( r + p / f ) ; }
int main ( ) { int x = 4 , n = 15 ; printf ( " % lf ▁ STRNEWLINE " , e ( x , n ) ) ; return 0 ; }
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
#include <stdio.h> NEW_LINE int main ( ) { int matrix [ 5 ] [ 5 ] , row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
printf ( " The ▁ matrix ▁ is STRNEWLINE " ) ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { printf ( " % d TABSYMBOL " , matrix [ row_index ] [ column_index ] ) ; } printf ( " STRNEWLINE " ) ; }
printf ( " Elements on Secondary diagonal : " for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) == size - 1 ) printf ( " % d , ▁ " , matrix [ row_index ] [ column_index ] ) ; } } return 0 ; }
#include <stdio.h> NEW_LINE int main ( ) { int matrix [ 5 ] [ 5 ] , row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
printf ( " The ▁ matrix ▁ is STRNEWLINE " ) ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { printf ( " % d TABSYMBOL " , matrix [ row_index ] [ column_index ] ) ; } printf ( " STRNEWLINE " ) ; }
printf ( " Elements above Secondary diagonal are : " for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) < size - 1 ) printf ( " % d , ▁ " , matrix [ row_index ] [ column_index ] ) ; } } return 0 ; }
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
double sum ( int x , int n ) { double i , total = 1.0 , multi = x ; for ( i = 1 ; i <= n ; i ++ ) { total = total + multi / i ; multi = multi * x ; } return total ; }
int main ( ) { int x = 2 ; int n = 5 ; printf ( " % .2f " , sum ( x , n ) ) ; return 0 ; }
int chiliagonNum ( int n ) { return ( 998 * n * n - 996 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ chiliagon ▁ Number ▁ is ▁ = ▁ % d " , chiliagonNum ( n ) ) ; return 0 ; }
int pentacontagonNum ( int n ) { return ( 48 * n * n - 46 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ pentacontagon ▁ Number ▁ is ▁ = ▁ % d " , pentacontagonNum ( n ) ) ; return 0 ; }
int lastElement ( vector < int > & arr ) {
priority_queue < int > pq ; for ( int i = 0 ; i < arr . size ( ) ; i ++ ) { pq . push ( arr [ i ] ) ; }
int m1 , m2 ;
while ( ! pq . empty ( ) ) {
if ( pq . size ( ) == 1 )
return pq . top ( ) ; m1 = pq . top ( ) ; pq . pop ( ) ; m2 = pq . top ( ) ; pq . pop ( ) ;
if ( m1 != m2 ) pq . push ( m1 - m2 ) ; }
return 0 ; }
int main ( ) { vector < int > arr = { 2 , 7 , 4 , 1 , 8 , 1 , 1 } ; cout << lastElement ( arr ) << endl ; return 0 ; }
int countDigit ( long long n ) { return ( floor ( log10 ( n ) + 1 ) ) ; }
int main ( ) { double N = 80 ; printf ( " % d " , countDigit ( N ) ) ; return 0 ; }
double sum ( int x , int n ) { double i , total = 1.0 , multi = x ;
printf ( "1 ▁ " ) ;
for ( i = 1 ; i < n ; i ++ ) { total = total + multi ; printf ( " % .1f ▁ " , multi ) ; multi = multi * x ; } printf ( " STRNEWLINE " ) ; return total ; }
int main ( ) { int x = 2 ; int n = 5 ; printf ( " % .2f " , sum ( x , n ) ) ; return 0 ; }
int findRemainder ( int n ) {
int x = n & 3 ;
return x ; }
int main ( ) { int N = 43 ; int ans = findRemainder ( N ) ; printf ( " % d " , ans ) ; return 0 ; }
void triangular_series ( int n ) { int i , j = 1 , k = 1 ;
for ( i = 1 ; i <= n ; i ++ ) { printf ( " ▁ % d ▁ " , k ) ;
j = j + 1 ;
k = k + j ; } }
int main ( ) { int n = 5 ; triangular_series ( n ) ; return 0 ; }
#include <stdio.h> NEW_LINE int countDigit ( long long n ) { if ( n / 10 == 0 ) return 1 ; return 1 + countDigit ( n / 10 ) ; }
int main ( void ) { long long n = 345289467 ; printf ( " Number ▁ of ▁ digits ▁ : ▁ % d " , countDigit ( n ) ) ; return 0 ; }
int x = 1234 ;
if ( x % 9 == 1 ) printf ( " Magic ▁ Number " ) ; else printf ( " Not ▁ a ▁ Magic ▁ Number " ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define MAX  100 NEW_LINE int main ( ) {
long long int arr [ MAX ] ; arr [ 0 ] = 0 ; arr [ 1 ] = 1 ; for ( int i = 2 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] + arr [ i - 2 ] ; printf ( " Fibonacci ▁ numbers ▁ divisible ▁ by ▁ " " their ▁ indexes ▁ are ▁ : STRNEWLINE " ) ; for ( int i = 1 ; i < MAX ; i ++ ) if ( arr [ i ] % i == 0 ) printf ( " % d ▁ " , i ) ; }
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
#define N  30
int fib [ N ] ;
int largestFiboLessOrEqual ( int n ) {
fib [ 0 ] = 1 ;
fib [ 1 ] = 2 ;
int i ; for ( i = 2 ; fib [ i - 1 ] <= n ; i ++ ) fib [ i ] = fib [ i - 1 ] + fib [ i - 2 ] ;
return ( i - 2 ) ; }
char * fibonacciEncoding ( int n ) { int index = largestFiboLessOrEqual ( n ) ;
char * codeword = ( char * ) malloc ( sizeof ( char ) * ( index + 3 ) ) ;
int i = index ; while ( n ) {
codeword [ i ] = '1' ;
n = n - fib [ i ] ;
i = i - 1 ;
while ( i >= 0 && fib [ i ] > n ) { codeword [ i ] = '0' ; i = i - 1 ; } }
codeword [ index + 1 ] = '1' ; codeword [ index + 2 ] = ' \0' ;
return codeword ; }
int main ( ) { int n = 143 ; printf ( " Fibonacci ▁ code ▁ word ▁ for ▁ % d ▁ is ▁ % s STRNEWLINE " , n , fibonacciEncoding ( n ) ) ; return 0 ; }
int countSquares ( int m , int n ) {
if ( n < m ) { int temp = m ; m = n ; n = temp ; }
return n * ( n + 1 ) * ( 3 * m - n + 1 ) / 6 ; }
int main ( ) { int m = 4 , n = 3 ; printf ( " Count ▁ of ▁ squares ▁ is ▁ % d " , countSquares ( m , n ) ) ; }
void simpleSieve ( int limit ) {
bool mark [ limit ] ; for ( int i = 0 ; i < limit ; i ++ ) { mark [ i ] = true ; }
for ( int p = 2 ; p * p < limit ; p ++ ) {
if ( mark [ p ] == true ) {
for ( int i = p * p ; i < limit ; i += p ) mark [ i ] = false ; } }
for ( int p = 2 ; p < limit ; p ++ ) if ( mark [ p ] == true ) cout << p << " ▁ " ; }
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
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE #include <string.h> NEW_LINE # define MAX  11 NEW_LINE bool isMultipleof5 ( int n ) { char str [ MAX ] ; int len = strlen ( str ) ;
if ( str [ len - 1 ] == '5' str [ len - 1 ] == '0' ) return true ; return false ; }
int main ( ) { int n = 19 ; if ( isMultipleof5 ( n ) == true ) printf ( " % d ▁ is ▁ multiple ▁ of ▁ 5 STRNEWLINE " , n ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ multiple ▁ of ▁ 5 STRNEWLINE " , n ) ; return 0 ; }
int toggleBit ( int n , int k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
int main ( ) { int n = 5 , k = 2 ; printf ( " % d STRNEWLINE " , toggleBit ( n , k ) ) ; return 0 ; }
int clearBit ( int n , int k ) { return ( n & ( ~ ( 1 << ( k - 1 ) ) ) ) ; }
int main ( ) { int n = 5 , k = 1 ; printf ( " % d STRNEWLINE " , clearBit ( n , k ) ) ; return 0 ; }
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
unsigned int toggleAllExceptK ( unsigned int n , unsigned int k ) {
return ~ ( n ^ ( 1 << k ) ) ; }
int main ( ) { unsigned int n = 4294967295 ; unsigned int k = 0 ; printf ( " % u " , toggleAllExceptK ( n , k ) ) ; return 0 ; }
int swapBits ( unsigned int n , unsigned int p1 , unsigned int p2 ) {
unsigned int bit1 = ( n >> p1 ) & 1 ;
unsigned int bit2 = ( n >> p2 ) & 1 ;
unsigned int x = ( bit1 ^ bit2 ) ;
x = ( x << p1 ) | ( x << p2 ) ;
unsigned int result = n ^ x ; }
int main ( ) { int res = swapBits ( 28 , 0 , 3 ) ; printf ( " Result ▁ = ▁ % d ▁ " , res ) ; return 0 ; }
int firstNonRepeating ( char * str ) {
int arr [ NO_OF_CHARS ] ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) arr [ i ] = -1 ;
for ( int i = 0 ; str [ i ] ; i ++ ) { if ( arr [ str [ i ] ] == -1 ) arr [ str [ i ] ] = i ; else arr [ str [ i ] ] = -2 ; } int res = INT_MAX ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ )
if ( arr [ i ] >= 0 ) res = min ( res , arr [ i ] ) ; return res ; }
int main ( ) { char str [ ] = " geeksforgeeks " ; int index = firstNonRepeating ( str ) ; if ( index == INT_MAX ) printf ( " Either ▁ all ▁ characters ▁ are ▁ " " repeating ▁ or ▁ string ▁ is ▁ empty " ) ; else printf ( " First ▁ non - repeating ▁ character " " ▁ is ▁ % c " , str [ index ] ) ; return 0 ; }
int triacontagonalNum ( int n ) { return ( 28 * n * n - 26 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ triacontagonal ▁ Number ▁ is ▁ = ▁ % d " , triacontagonalNum ( n ) ) ; return 0 ; }
int hexacontagonNum ( int n ) { return ( 58 * n * n - 56 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ hexacontagon ▁ Number ▁ is ▁ = ▁ % d " , hexacontagonNum ( n ) ) ; return 0 ; }
int enneacontagonNum ( int n ) { return ( 88 * n * n - 86 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ enneacontagon ▁ Number ▁ is ▁ = ▁ % d " , enneacontagonNum ( n ) ) ; return 0 ; }
int triacontakaidigonNum ( int n ) { return ( 30 * n * n - 28 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ triacontakaidigon ▁ Number ▁ is ▁ = ▁ % d " , triacontakaidigonNum ( n ) ) ; return 0 ; }
int IcosihexagonalNum ( int n ) { return ( 24 * n * n - 22 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ Icosihexagonal ▁ Number ▁ is ▁ = ▁ % d " , IcosihexagonalNum ( n ) ) ; return 0 ; }
int icosikaioctagonalNum ( int n ) { return ( 26 * n * n - 24 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ icosikaioctagonal ▁ Number ▁ is ▁ = ▁ % d " , icosikaioctagonalNum ( n ) ) ; return 0 ; }
int octacontagonNum ( int n ) { return ( 78 * n * n - 76 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ octacontagon ▁ Number ▁ is ▁ = ▁ % d " , octacontagonNum ( n ) ) ; return 0 ; }
int hectagonNum ( int n ) { return ( 98 * n * n - 96 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ hectagon ▁ Number ▁ is ▁ = ▁ % d " , hectagonNum ( n ) ) ; return 0 ; }
int tetracontagonNum ( int n ) { return ( 38 * n * n - 36 * n ) / 2 ; }
int main ( ) { int n = 3 ; printf ( "3rd ▁ tetracontagon ▁ Number ▁ is ▁ = ▁ % d " , tetracontagonNum ( n ) ) ; return 0 ; }
int binarySearch ( int arr [ ] , int N , int X ) {
int start = 0 ;
int end = N ; while ( start <= end ) {
int mid = start + ( end - start ) / 2 ;
if ( X == arr [ mid ] ) {
return mid ; }
else if ( X < arr [ mid ] ) {
start = mid + 1 ; } else {
end = mid - 1 ; } }
return -1 ; }
int main ( ) { int arr [ ] = { 5 , 4 , 3 , 2 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int X = 4 ; int res = binarySearch ( arr , N , X ) ; printf ( " ▁ % d ▁ " , res ) ; return 0 ; }
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
