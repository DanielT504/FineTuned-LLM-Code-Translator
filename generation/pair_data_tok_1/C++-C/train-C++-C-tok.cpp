#include <bits/stdc++.h> NEW_LINE using namespace std ;
void count_setbit ( int N ) {
int result = 0 ;
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( 1 << i ) & N ) {
result ++ ; } } cout << result << endl ; }
int main ( ) { int N = 43 ; count_setbit ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPowerOfTwo ( int n ) { return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { int N = 8 ; if ( isPowerOfTwo ( N ) ) { cout << " Yes " ; } else { cout << " No " ; } }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
typedef struct cantor { double start , end ; struct cantor * next ; } Cantor ;
Cantor * startList ( Cantor * head , double start_num , double end_num ) { if ( head == NULL ) { head = new Cantor ; head -> start = start_num ; head -> end = end_num ; head -> next = NULL ; } return head ; }
Cantor * propagate ( Cantor * head ) { Cantor * temp = head ; if ( temp != NULL ) { Cantor * newNode = new Cantor ; double diff = ( ( ( temp -> end ) - ( temp -> start ) ) / 3 ) ;
newNode -> end = temp -> end ; temp -> end = ( ( temp -> start ) + diff ) ; newNode -> start = ( newNode -> end ) - diff ;
newNode -> next = temp -> next ; temp -> next = newNode ;
propagate ( temp -> next -> next ) ; } return head ; }
void print ( Cantor * temp ) { while ( temp != NULL ) { printf ( " [ % lf ] ▁ - - ▁ [ % lf ] TABSYMBOL " , temp -> start , temp -> end ) ; temp = temp -> next ; } cout << endl ; }
void buildCantorSet ( int A , int B , int L ) { Cantor * head = NULL ; head = startList ( head , A , B ) ; for ( int i = 0 ; i < L ; i ++ ) { cout << " Level _ " << i << " ▁ : ▁ " ; print ( head ) ; propagate ( head ) ; } cout << " Level _ " << L << " ▁ : ▁ " ; print ( head ) ; }
int main ( ) { int A = 0 ; int B = 9 ; int L = 2 ; buildCantorSet ( A , B , L ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void search ( string pat , string txt ) { int M = pat . size ( ) ; int N = txt . size ( ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
{ cout << " Pattern ▁ found ▁ at ▁ index ▁ " << i << endl ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { string txt = " ABCEABCDABCEABCD " ; string pat = " ABCD " ; search ( pat , txt ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void encrypt ( char input [ 100 ] ) {
char evenPos = ' @ ' , oddPos = ' ! ' ; int repeat , ascii ; for ( int i = 0 ; i <= strlen ( input ) ; i ++ ) {
ascii = input [ i ] ; repeat = ascii >= 97 ? ascii - 96 : ascii - 64 ; for ( int j = 0 ; j < repeat ; j ++ ) {
if ( i % 2 == 0 ) cout << oddPos ; else cout << evenPos ; } } }
int main ( ) { char input [ 100 ] = { ' A ' , ' b ' , ' C ' , ' d ' } ;
encrypt ( input ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPalRec ( char str [ ] , int s , int e ) {
if ( s == e ) return true ;
if ( str [ s ] != str [ e ] ) return false ;
if ( s < e + 1 ) return isPalRec ( str , s + 1 , e - 1 ) ; return true ; } bool isPalindrome ( char str [ ] ) { int n = strlen ( str ) ;
if ( n == 0 ) return true ; return isPalRec ( str , 0 , n - 1 ) ; }
int main ( ) { char str [ ] = " geeg " ; if ( isPalindrome ( str ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int myAtoi ( const char * str ) { int sign = 1 , base = 0 , i = 0 ;
while ( str [ i ] == ' ▁ ' ) { i ++ ; }
if ( str [ i ] == ' - ' str [ i ] == ' + ' ) { sign = 1 - 2 * ( str [ i ++ ] == ' - ' ) ; }
while ( str [ i ] >= '0' && str [ i ] <= '9' ) {
if ( base > INT_MAX / 10 || ( base == INT_MAX / 10 && str [ i ] - '0' > 7 ) ) { if ( sign == 1 ) return INT_MAX ; else return INT_MIN ; } base = 10 * base + ( str [ i ++ ] - '0' ) ; } return base * sign ; }
int main ( ) { char str [ ] = " ▁ - 123" ;
int val = myAtoi ( str ) ; cout << " ▁ " << val ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool fillUtil ( int res [ ] , int curr , int n ) {
if ( curr == 0 ) return true ;
int i ; for ( i = 0 ; i < 2 * n - curr - 1 ; i ++ ) {
if ( res [ i ] == 0 && res [ i + curr + 1 ] == 0 ) {
res [ i ] = res [ i + curr + 1 ] = curr ;
if ( fillUtil ( res , curr - 1 , n ) ) return true ;
res [ i ] = res [ i + curr + 1 ] = 0 ; } } return false ; }
void fill ( int n ) {
int res [ 2 * n ] , i ; for ( i = 0 ; i < 2 * n ; i ++ ) res [ i ] = 0 ;
if ( fillUtil ( res , n , n ) ) { for ( i = 0 ; i < 2 * n ; i ++ ) cout << res [ i ] << " ▁ " ; } else cout << " Not ▁ Possible " ; }
int main ( ) { fill ( 7 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findNumberOfDigits ( int n , int base ) {
int dig = ( floor ( log ( n ) / log ( base ) ) + 1 ) ;
return ( dig ) ; }
int isAllKs ( int n , int b , int k ) { int len = findNumberOfDigits ( n , b ) ;
int sum = k * ( 1 - pow ( b , len ) ) / ( 1 - b ) ; if ( sum == n ) { return ( sum ) ; } }
int main ( ) {
int N = 13 ;
int B = 3 ;
int K = 1 ;
if ( isAllKs ( N , B , K ) ) { cout << " Yes " ; } else { cout << " No " ; } }
#include <iostream> NEW_LINE using namespace std ;
void CalPeri ( ) { int s = 5 , Perimeter ; Perimeter = 10 * s ; cout << " The ▁ Perimeter ▁ of ▁ Decagon ▁ is ▁ : ▁ " << Perimeter ; }
int main ( ) { CalPeri ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void distance ( float a1 , float b1 , float c1 , float a2 , float b2 , float c2 ) { float d = ( a1 * a2 + b1 * b2 + c1 * c2 ) ; float e1 = sqrt ( a1 * a1 + b1 * b1 + c1 * c1 ) ; float e2 = sqrt ( a2 * a2 + b2 * b2 + c2 * c2 ) ; d = d / ( e1 * e2 ) ; float pi = 3.14159 ; float A = ( 180 / pi ) * ( acos ( d ) ) ; cout << " Angle ▁ is ▁ " << A << " ▁ degree " ; }
int main ( ) { float a1 = 1 ; float b1 = 1 ; float c1 = 2 ; float d1 = 1 ; float a2 = 2 ; float b2 = -1 ; float c2 = 1 ; float d2 = -4 ; distance ( a1 , b1 , c1 , a2 , b2 , c2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE #include <iostream> NEW_LINE #include <iomanip> NEW_LINE using namespace std ;
void mirror_point ( float a , float b , float c , float d , float x1 , float y1 , float z1 ) { float k = ( - a * x1 - b * y1 - c * z1 - d ) / ( float ) ( a * a + b * b + c * c ) ; float x2 = a * k + x1 ; float y2 = b * k + y1 ; float z2 = c * k + z1 ; float x3 = 2 * x2 - x1 ; float y3 = 2 * y2 - y1 ; float z3 = 2 * z2 - z1 ; std :: cout << std :: fixed ; std :: cout << std :: setprecision ( 1 ) ; cout << " ▁ x3 ▁ = ▁ " << x3 ; cout << " ▁ y3 ▁ = ▁ " << y3 ; cout << " ▁ z3 ▁ = ▁ " << z3 ; }
int main ( ) { float a = 1 ; float b = -2 ; float c = 0 ; float d = 0 ; float x1 = -1 ; float y1 = 3 ; float z1 = 4 ;
mirror_point ( a , b , c , d , x1 , y1 , z1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left , * right ;
node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; } } ;
int updatetree ( node * root ) {
if ( ! root ) return 0 ; if ( root -> left == NULL && root -> right == NULL ) return root -> data ;
int leftsum = updatetree ( root -> left ) ; int rightsum = updatetree ( root -> right ) ;
root -> data += leftsum ;
return root -> data + rightsum ; }
void inorder ( node * node ) { if ( node == NULL ) return ; inorder ( node -> left ) ; cout << node -> data << " ▁ " ; inorder ( node -> right ) ; }
int main ( ) {
struct node * root = new node ( 1 ) ; root -> left = new node ( 2 ) ; root -> right = new node ( 3 ) ; root -> left -> left = new node ( 4 ) ; root -> left -> right = new node ( 5 ) ; root -> right -> right = new node ( 6 ) ; updatetree ( root ) ; cout << " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ is : ▁ STRNEWLINE " ; inorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void calculateSpan ( int price [ ] , int n , int S [ ] ) {
S [ 0 ] = 1 ;
for ( int i = 1 ; i < n ; i ++ ) {
S [ i ] = 1 ;
for ( int j = i - 1 ; ( j >= 0 ) && ( price [ i ] >= price [ j ] ) ; j -- ) S [ i ] ++ ; } }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int price [ ] = { 10 , 4 , 5 , 90 , 120 , 80 } ; int n = sizeof ( price ) / sizeof ( price [ 0 ] ) ; int S [ n ] ;
calculateSpan ( price , n , S ) ;
printArray ( S , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void printNGE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] < arr [ j ] ) { next = arr [ j ] ; break ; } } cout << arr [ i ] << " ▁ - - ▁ " << next << endl ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNGE ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * left ; struct Node * right ; } ;
struct Node * newNode ( int data ) { struct Node * node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; node -> data = data ; node -> left = NULL ; node -> right = NULL ; return ( node ) ; }
void mirror ( struct Node * node ) { if ( node == NULL ) return ; else { struct Node * temp ;
mirror ( node -> left ) ; mirror ( node -> right ) ;
temp = node -> left ; node -> left = node -> right ; node -> right = temp ; } }
void inOrder ( struct Node * node ) { if ( node == NULL ) return ; inOrder ( node -> left ) ; cout << node -> data << " ▁ " ; inOrder ( node -> right ) ; }
int main ( ) { struct Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ;
cout << " Inorder ▁ traversal ▁ of ▁ the ▁ constructed " << " ▁ tree ▁ is " << endl ; inOrder ( root ) ;
mirror ( root ) ;
cout << " Inorder traversal of the mirror tree " << " ▁ is ▁ STRNEWLINE " ; inOrder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define bool  int NEW_LINE #define true  1 NEW_LINE #define false  0
class node { public : int data ; node * left ; node * right ; } ;
bool IsFoldableUtil ( node * n1 , node * n2 ) ;
bool IsFoldable ( node * root ) { if ( root == NULL ) { return true ; } return IsFoldableUtil ( root -> left , root -> right ) ; }
bool IsFoldableUtil ( node * n1 , node * n2 ) {
if ( n1 == NULL && n2 == NULL ) { return true ; }
if ( n1 == NULL n2 == NULL ) { return false ; }
return IsFoldableUtil ( n1 -> left , n2 -> right ) && IsFoldableUtil ( n1 -> right , n2 -> left ) ; }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int main ( void ) {
node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> right = newNode ( 4 ) ; root -> right -> left = newNode ( 5 ) ; if ( IsFoldable ( root ) == true ) { cout << " Tree ▁ is ▁ foldable " ; } else { cout << " Tree ▁ is ▁ not ▁ foldable " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct node { int data ; struct node * left ; struct node * right ; } ;
int isSumProperty ( struct node * node ) {
int left_data = 0 , right_data = 0 ;
if ( node == NULL || ( node -> left == NULL && node -> right == NULL ) ) return 1 ; else {
if ( node -> left != NULL ) left_data = node -> left -> data ;
if ( node -> right != NULL ) right_data = node -> right -> data ;
if ( ( node -> data == left_data + right_data ) && isSumProperty ( node -> left ) && isSumProperty ( node -> right ) ) return 1 ; else return 0 ; } }
struct node * newNode ( int data ) { struct node * node = ( struct node * ) malloc ( sizeof ( struct node ) ) ; node -> data = data ; node -> left = NULL ; node -> right = NULL ; return ( node ) ; }
int main ( ) { struct node * root = newNode ( 10 ) ; root -> left = newNode ( 8 ) ; root -> right = newNode ( 2 ) ; root -> left -> left = newNode ( 3 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> right = newNode ( 2 ) ; if ( isSumProperty ( root ) ) cout << " The ▁ given ▁ tree ▁ satisfies ▁ " << " the ▁ children ▁ sum ▁ property ▁ " ; else cout << " The ▁ given ▁ tree ▁ does ▁ not ▁ satisfy ▁ " << " the ▁ children ▁ sum ▁ property ▁ " ; getchar ( ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int gcd ( int a , int b ) {
if ( a == 0 && b == 0 ) return 0 ; if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
int main ( ) { int a = 0 , b = 56 ; cout << " GCD ▁ of ▁ " << a << " ▁ and ▁ " << b << " ▁ is ▁ " << gcd ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int msbPos ( int n ) { int pos = 0 ; while ( n != 0 ) { pos ++ ;
n = n >> 1 ; } return pos ; }
int josephify ( int n ) {
int position = msbPos ( n ) ;
int j = 1 << ( position - 1 ) ;
n = n ^ j ;
n = n << 1 ;
n = n | 1 ; return n ; }
int main ( ) { int n = 41 ; cout << josephify ( n ) ; return 0 ;
#include <iostream> NEW_LINE using namespace std ;
int gcd ( int a , int b ) {
if ( a == 0 ) return b ; if ( b == 0 ) return a ;
if ( a == b ) return a ;
if ( a > b ) return gcd ( a - b , b ) ; return gcd ( a , b - a ) ; }
int main ( ) { int a = 98 , b = 56 ; cout << " GCD ▁ of ▁ " << a << " ▁ and ▁ " << b << " ▁ is ▁ " << gcd ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxsize = 100005 ;
vector < int > xor_tree ( maxsize ) ;
void construct_Xor_Tree_Util ( vector < int > current , int start , int end , int x ) {
if ( start == end ) { xor_tree [ x ] = current [ start ] ;
return ; }
int left = x * 2 + 1 ;
int right = x * 2 + 2 ;
int mid = start + ( end - start ) / 2 ;
construct_Xor_Tree_Util ( current , start , mid , left ) ; construct_Xor_Tree_Util ( current , mid + 1 , end , right ) ;
xor_tree [ x ] = ( xor_tree [ left ] ^ xor_tree [ right ] ) ; }
void construct_Xor_Tree ( vector < int > arr , int n ) { construct_Xor_Tree_Util ( arr , 0 , n - 1 , 0 ) ; }
int main ( ) {
vector < int > leaf_nodes = { 40 , 32 , 12 , 1 , 4 , 3 , 2 , 7 } ; int n = leaf_nodes . size ( ) ;
construct_Xor_Tree ( leaf_nodes , n ) ;
int x = ( int ) ( ceil ( log2 ( n ) ) ) ;
int max_size = 2 * ( int ) pow ( 2 , x ) - 1 ; cout << " Nodes ▁ of ▁ the ▁ XOR ▁ Tree : STRNEWLINE " ; for ( int i = 0 ; i < max_size ; i ++ ) { cout << xor_tree [ i ] << " ▁ " ; }
int root = 0 ;
cout << " Root : " }
#include <iostream> NEW_LINE using namespace std ; int swapBits ( int n , int p1 , int p2 ) {
n ^= 1 << p1 ; n ^= 1 << p2 ; return n ; }
int main ( ) { cout << " Result ▁ = ▁ " << swapBits ( 28 , 0 , 3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * left , * right ; } ;
struct Node * newNode ( int data ) { struct Node * node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; node -> data = data ; node -> left = node -> right = NULL ; return node ; }
bool checkUtil ( struct Node * root , int level , int * leafLevel ) {
if ( root == NULL ) return true ;
if ( root -> left == NULL && root -> right == NULL ) {
if ( * leafLevel == 0 ) {
* leafLevel = level ; return true ; }
return ( level == * leafLevel ) ; }
return checkUtil ( root -> left , level + 1 , leafLevel ) && checkUtil ( root -> right , level + 1 , leafLevel ) ; }
bool check ( struct Node * root ) { int level = 0 , leafLevel = 0 ; return checkUtil ( root , level , & leafLevel ) ; }
int main ( ) {
struct Node * root = newNode ( 12 ) ; root -> left = newNode ( 5 ) ; root -> left -> left = newNode ( 3 ) ; root -> left -> right = newNode ( 9 ) ; root -> left -> left -> left = newNode ( 1 ) ; root -> left -> right -> left = newNode ( 1 ) ; if ( check ( root ) ) cout << " Leaves ▁ are ▁ at ▁ same ▁ level STRNEWLINE " ; else cout << " Leaves ▁ are ▁ not ▁ at ▁ same ▁ level STRNEWLINE " ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int key ; struct Node * left , * right ; } ;
struct Node * newNode ( char k ) { struct Node * node = new Node ; node -> key = k ; node -> right = node -> left = NULL ; return node ; }
bool isFullTree ( struct Node * root ) {
if ( root == NULL ) return true ;
if ( root -> left == NULL && root -> right == NULL ) return true ;
if ( ( root -> left ) && ( root -> right ) ) return ( isFullTree ( root -> left ) && isFullTree ( root -> right ) ) ;
return false ; }
int main ( ) { struct Node * root = NULL ; root = newNode ( 10 ) ; root -> left = newNode ( 20 ) ; root -> right = newNode ( 30 ) ; root -> left -> right = newNode ( 40 ) ; root -> left -> left = newNode ( 50 ) ; root -> right -> left = newNode ( 60 ) ; root -> right -> right = newNode ( 70 ) ; root -> left -> left -> left = newNode ( 80 ) ; root -> left -> left -> right = newNode ( 90 ) ; root -> left -> right -> left = newNode ( 80 ) ; root -> left -> right -> right = newNode ( 90 ) ; root -> right -> left -> left = newNode ( 80 ) ; root -> right -> left -> right = newNode ( 90 ) ; root -> right -> right -> left = newNode ( 80 ) ; root -> right -> right -> right = newNode ( 90 ) ; if ( isFullTree ( root ) ) cout << " The ▁ Binary ▁ Tree ▁ is ▁ full STRNEWLINE " ; else cout << " The ▁ Binary ▁ Tree ▁ is ▁ not ▁ full STRNEWLINE " ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printAlter ( int arr [ ] , int N ) {
for ( int currIndex = 0 ; currIndex < N ; currIndex += 2 ) {
cout << arr [ currIndex ] << " ▁ " ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printAlter ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int identicalTrees ( node * a , node * b ) {
if ( a == NULL && b == NULL ) return 1 ;
if ( a != NULL && b != NULL ) { return ( a -> data == b -> data && identicalTrees ( a -> left , b -> left ) && identicalTrees ( a -> right , b -> right ) ) ; }
return 0 ; }
int main ( ) { node * root1 = newNode ( 1 ) ; node * root2 = newNode ( 1 ) ; root1 -> left = newNode ( 2 ) ; root1 -> right = newNode ( 3 ) ; root1 -> left -> left = newNode ( 4 ) ; root1 -> left -> right = newNode ( 5 ) ; root2 -> left = newNode ( 2 ) ; root2 -> right = newNode ( 3 ) ; root2 -> left -> left = newNode ( 4 ) ; root2 -> left -> right = newNode ( 5 ) ; if ( identicalTrees ( root1 , root2 ) ) cout << " Both ▁ tree ▁ are ▁ identical . " ; else cout << " Trees ▁ are ▁ not ▁ identical . " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; Node * left , * right ; } ;
Node * newNode ( int item ) { Node * temp = new Node ; temp -> data = item ; temp -> left = temp -> right = NULL ; return temp ; }
int getLevel ( Node * root , Node * node , int level ) {
if ( root == NULL ) return 0 ; if ( root == node ) return level ;
int downlevel = getLevel ( root -> left , node , level + 1 ) ; if ( downlevel != 0 ) return downlevel ;
return getLevel ( root -> right , node , level + 1 ) ; }
void printGivenLevel ( Node * root , Node * node , int level ) {
if ( root == NULL level < 2 ) return ;
if ( level == 2 ) { if ( root -> left == node root -> right == node ) return ; if ( root -> left ) cout << root -> left -> data << " ▁ " ; if ( root -> right ) cout << root -> right -> data ; }
else if ( level > 2 ) { printGivenLevel ( root -> left , node , level - 1 ) ; printGivenLevel ( root -> right , node , level - 1 ) ; } }
void printCousins ( Node * root , Node * node ) {
int level = getLevel ( root , node , 1 ) ;
printGivenLevel ( root , node , level ) ; }
int main ( ) { Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> left -> right -> right = newNode ( 15 ) ; root -> right -> left = newNode ( 6 ) ; root -> right -> right = newNode ( 7 ) ; root -> right -> left -> right = newNode ( 8 ) ; printCousins ( root , root -> left -> right ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
void printArray ( int [ ] , int ) ; void printPathsRecur ( node * , int [ ] , int ) ; node * newNode ( int ) ; void printPaths ( node * ) ;
void printPaths ( node * node ) { int path [ 1000 ] ; printPathsRecur ( node , path , 0 ) ; }
void printPathsRecur ( node * node , int path [ ] , int pathLen ) { if ( node == NULL ) return ;
path [ pathLen ] = node -> data ; pathLen ++ ;
if ( node -> left == NULL && node -> right == NULL ) { printArray ( path , pathLen ) ; } else {
printPathsRecur ( node -> left , path , pathLen ) ; printPathsRecur ( node -> right , path , pathLen ) ; } }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
void printArray ( int ints [ ] , int len ) { int i ; for ( i = 0 ; i < len ; i ++ ) { cout << ints [ i ] << " ▁ " ; } cout << endl ; }
int main ( ) { node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ;
printPaths ( root ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) {
int arr [ 10 ] ;
arr [ 0 ] = 5 ;
cout << arr [ 0 ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printArray ( int arr [ ] , int size ) ; void swap ( int arr [ ] , int fi , int si , int d ) ; void leftRotate ( int arr [ ] , int d , int n ) {
if ( d == 0 d == n ) return ;
if ( n - d == d ) { swap ( arr , 0 , n - d , d ) ; return ; }
if ( d < n - d ) { swap ( arr , 0 , n - d , d ) ; leftRotate ( arr , d , n - d ) ; }
else { swap ( arr , 0 , d , n - d ) ; leftRotate ( arr + n - d , 2 * d - n , d ) ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; }
void swap ( int arr [ ] , int fi , int si , int d ) { int i , temp ; for ( i = 0 ; i < d ; i ++ ) { temp = arr [ fi + i ] ; arr [ fi + i ] = arr [ si + i ] ; arr [ si + i ] = temp ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; leftRotate ( arr , 2 , 7 ) ; printArray ( arr , 7 ) ; return 0 ; }
void leftRotate ( int arr [ ] , int d , int n ) { int i , j ; if ( d == 0 d == n ) return ; i = d ; j = n - d ; while ( i != j ) {
if ( i < j ) { swap ( arr , d - i , d + j - i , i ) ; j -= i ; }
else { swap ( arr , d - i , d , j ) ; i -= j ; } }
swap ( arr , d - i , d , i ) ; }
#include <iostream> NEW_LINE using namespace std ;
void rotate ( int arr [ ] , int n ) { int i = 0 , j = n - 1 ; while ( i != j ) { swap ( arr [ i ] , arr [ j ] ) ; i ++ ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } , i ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Given ▁ array ▁ is ▁ STRNEWLINE " ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; rotate ( arr , n ) ; cout << " Rotated array is " ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ; #define MAX_LEN  100
void selectionSort ( char arr [ ] [ MAX_LEN ] , int n ) { int i , j , min_idx ;
char minStr [ MAX_LEN ] ; for ( i = 0 ; i < n - 1 ; i ++ ) {
int min_idx = i ; strcpy ( minStr , arr [ i ] ) ; for ( j = i + 1 ; j < n ; j ++ ) {
if ( strcmp ( minStr , arr [ j ] ) > 0 ) {
strcpy ( minStr , arr [ j ] ) ; min_idx = j ; } }
if ( min_idx != i ) { char temp [ MAX_LEN ] ; strcpy ( temp , arr [ i ] ) ; strcpy ( arr [ i ] , arr [ min_idx ] ) ; strcpy ( arr [ min_idx ] , temp ) ; } } }
int main ( ) { char arr [ ] [ MAX_LEN ] = { " GeeksforGeeks " , " Practice . GeeksforGeeks " , " GeeksQuiz " } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int i ; cout << " Given ▁ array ▁ is STRNEWLINE " ;
for ( i = 0 ; i < n ; i ++ ) cout << i << " : ▁ " << arr [ i ] << endl ; selectionSort ( arr , n ) ; cout << " Sorted array is " ;
for ( i = 0 ; i < n ; i ++ ) cout << i << " : ▁ " << arr [ i ] << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void rearrangeNaive ( int arr [ ] , int n ) {
int temp [ n ] , i ;
for ( i = 0 ; i < n ; i ++ ) temp [ arr [ i ] ] = i ;
for ( i = 0 ; i < n ; i ++ ) arr [ i ] = temp [ i ] ; }
void printArray ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) cout << ( " % d ▁ " , arr [ i ] ) ; cout << ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 0 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << ( " Given ▁ array ▁ is ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; rearrangeNaive ( arr , n ) ; cout << ( " Modified ▁ array ▁ is ▁ STRNEWLINE " ) ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int largest ( int arr [ ] , int n ) { int i ;
int max = arr [ 0 ] ;
for ( i = 1 ; i < n ; i ++ ) if ( arr [ i ] > max ) max = arr [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 10 , 324 , 45 , 90 , 9808 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Largest ▁ in ▁ given ▁ array ▁ is ▁ " << largest ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void print2largest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { cout << " ▁ Invalid ▁ Input ▁ " ; return ; } first = second = INT_MIN ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second && arr [ i ] != first ) { second = arr [ i ] ; } } if ( second == INT_MIN ) cout << " There ▁ is ▁ no ▁ second ▁ largest " " element STRNEWLINE " ; else cout << " The ▁ second ▁ largest ▁ element ▁ is ▁ " << second ; }
int main ( ) { int arr [ ] = { 12 , 35 , 1 , 10 , 34 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; print2largest ( arr , n ) ; return 0 ; }
struct Pair { int min ; int max ; } ; struct Pair getMinMax ( int arr [ ] , int n ) { struct Pair minmax ; int i ;
if ( n == 1 ) { minmax . max = arr [ 0 ] ; minmax . min = arr [ 0 ] ; return minmax ; }
if ( arr [ 0 ] > arr [ 1 ] ) { minmax . max = arr [ 0 ] ; minmax . min = arr [ 1 ] ; } else { minmax . max = arr [ 1 ] ; minmax . min = arr [ 0 ] ; } for ( i = 2 ; i < n ; i ++ ) { if ( arr [ i ] > minmax . max ) minmax . max = arr [ i ] ; else if ( arr [ i ] < minmax . min ) minmax . min = arr [ i ] ; } return minmax ; }
int main ( ) { int arr [ ] = { 1000 , 11 , 445 , 1 , 330 , 3000 } ; int arr_size = 6 ; struct Pair minmax = getMinMax ( arr , arr_size ) ; cout << " Minimum ▁ element ▁ is ▁ " << minmax . min << endl ; cout << " Maximum ▁ element ▁ is ▁ " << minmax . max ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
struct Pair { int min ; int max ; } ; struct Pair getMinMax ( int arr [ ] , int n ) { struct Pair minmax ; int i ;
if ( n % 2 == 0 ) { if ( arr [ 0 ] > arr [ 1 ] ) { minmax . max = arr [ 0 ] ; minmax . min = arr [ 1 ] ; } else { minmax . min = arr [ 0 ] ; minmax . max = arr [ 1 ] ; }
i = 2 ; }
else { minmax . min = arr [ 0 ] ; minmax . max = arr [ 0 ] ;
i = 1 ; }
while ( i < n - 1 ) { if ( arr [ i ] > arr [ i + 1 ] ) { if ( arr [ i ] > minmax . max ) minmax . max = arr [ i ] ; if ( arr [ i + 1 ] < minmax . min ) minmax . min = arr [ i + 1 ] ; } else { if ( arr [ i + 1 ] > minmax . max ) minmax . max = arr [ i + 1 ] ; if ( arr [ i ] < minmax . min ) minmax . min = arr [ i ] ; }
i += 2 ; } return minmax ; }
int main ( ) { int arr [ ] = { 1000 , 11 , 445 , 1 , 330 , 3000 } ; int arr_size = 6 ; Pair minmax = getMinMax ( arr , arr_size ) ; cout << " nMinimum ▁ element ▁ is ▁ " << minmax . min << endl ; cout << " nMaximum ▁ element ▁ is ▁ " << minmax . max ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minJumps ( int arr [ ] , int n ) {
if ( n == 1 ) return 0 ;
int res = INT_MAX ; for ( int i = n - 2 ; i >= 0 ; i -- ) { if ( i + arr [ i ] >= n - 1 ) { int sub_res = minJumps ( arr , i + 1 ) ; if ( sub_res != INT_MAX ) res = min ( res , sub_res + 1 ) ; } } return res ; }
int main ( ) { int arr [ ] = { 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Minimum ▁ number ▁ of ▁ jumps ▁ to " ; cout << " ▁ reach ▁ the ▁ end ▁ is ▁ " << minJumps ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * left , * right ; Node ( int data ) { this -> data = data ; left = right = NULL ; } } ;
void printPostorder ( struct Node * node ) { if ( node == NULL ) return ;
printPostorder ( node -> left ) ;
printPostorder ( node -> right ) ;
cout << node -> data << " ▁ " ; }
void printInorder ( struct Node * node ) { if ( node == NULL ) return ;
printInorder ( node -> left ) ;
cout << node -> data << " ▁ " ;
printInorder ( node -> right ) ; }
void printPreorder ( struct Node * node ) { if ( node == NULL ) return ;
cout << node -> data << " ▁ " ;
printPreorder ( node -> left ) ;
printPreorder ( node -> right ) ; }
int main ( ) { struct Node * root = new Node ( 1 ) ; root -> left = new Node ( 2 ) ; root -> right = new Node ( 3 ) ; root -> left -> left = new Node ( 4 ) ; root -> left -> right = new Node ( 5 ) ; cout << " Preorder traversal of binary tree is " ; printPreorder ( root ) ; cout << " Inorder traversal of binary tree is " ; printInorder ( root ) ; cout << " Postorder traversal of binary tree is " ; printPostorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * left , * right ; } ;
struct Node * newNode ( int data ) { struct Node * node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; node -> data = data ; node -> left = node -> right = NULL ; return node ; }
void print ( struct Node * root ) { if ( root != NULL ) { print ( root -> left ) ; cout << root -> data << " ▁ " ; print ( root -> right ) ; } }
struct Node * prune ( struct Node * root , int sum ) {
if ( root == NULL ) return NULL ;
root -> left = prune ( root -> left , sum - root -> data ) ; root -> right = prune ( root -> right , sum - root -> data ) ;
if ( root -> left == NULL && root -> right == NULL ) { if ( root -> data < sum ) { free ( root ) ; return NULL ; } } return root ; }
int main ( ) { int k = 45 ; struct Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> left = newNode ( 6 ) ; root -> right -> right = newNode ( 7 ) ; root -> left -> left -> left = newNode ( 8 ) ; root -> left -> left -> right = newNode ( 9 ) ; root -> left -> right -> left = newNode ( 12 ) ; root -> right -> right -> left = newNode ( 10 ) ; root -> right -> right -> left -> right = newNode ( 11 ) ; root -> left -> left -> right -> left = newNode ( 13 ) ; root -> left -> left -> right -> right = newNode ( 14 ) ; root -> left -> left -> right -> right -> left = newNode ( 15 ) ; cout << " Tree ▁ before ▁ truncation STRNEWLINE " ; print ( root ) ; root = prune ( root , k ) ; cout << " Tree after truncation " ; print ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define NA  -1
void moveToEnd ( int mPlusN [ ] , int size ) { int j = size - 1 ; for ( int i = size - 1 ; i >= 0 ; i -- ) if ( mPlusN [ i ] != NA ) { mPlusN [ j ] = mPlusN [ i ] ; j -- ; } }
int merge ( int mPlusN [ ] , int N [ ] , int m , int n ) { int i = n ;
int j = 0 ;
int k = 0 ;
while ( k < ( m + n ) ) {
if ( ( j == n ) || ( i < ( m + n ) && mPlusN [ i ] <= N [ j ] ) ) { mPlusN [ k ] = mPlusN [ i ] ; k ++ ; i ++ ; }
else { mPlusN [ k ] = N [ j ] ; k ++ ; j ++ ; } } }
void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; }
int main ( ) {
int mPlusN [ ] = { 2 , 8 , NA , NA , NA , 13 , NA , 15 , 20 } ; int N [ ] = { 5 , 7 , 9 , 25 } ; int n = sizeof ( N ) / sizeof ( N [ 0 ] ) ; int m = sizeof ( mPlusN ) / sizeof ( mPlusN [ 0 ] ) - n ;
moveToEnd ( mPlusN , m + n ) ;
merge ( mPlusN , N , m , n ) ;
printArray ( mPlusN , m + n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
int main ( ) { int n = 2 , k = 1 ; cout << getCount ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int getInvCount ( int arr [ ] , int n ) { int inv_count = 0 ; for ( int i = 0 ; i < n - 1 ; i ++ ) for ( int j = i + 1 ; j < n ; j ++ ) if ( arr [ i ] > arr [ j ] ) inv_count ++ ; return inv_count ; }
int main ( ) { int arr [ ] = { 1 , 20 , 6 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " ▁ Number ▁ of ▁ inversions ▁ are ▁ " << getInvCount ( arr , n ) ; return 0 ; }
# include <bits/stdc++.h> NEW_LINE # include <stdlib.h> NEW_LINE # include <math.h> NEW_LINE using namespace std ; void minAbsSumPair ( int arr [ ] , int arr_size ) { int inv_count = 0 ; int l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { cout << " Invalid ▁ Input " ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( abs ( min_sum ) > abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } cout << " The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are ▁ " << arr [ min_l ] << " ▁ and ▁ " << arr [ min_r ] ; }
int main ( ) { int arr [ ] = { 1 , 60 , -10 , 70 , -80 , 85 } ; minAbsSumPair ( arr , 6 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printUnion ( int arr1 [ ] , int arr2 [ ] , int m , int n ) { int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( arr1 [ i ] < arr2 [ j ] ) cout << arr1 [ i ++ ] << " ▁ " ; else if ( arr2 [ j ] < arr1 [ i ] ) cout << arr2 [ j ++ ] << " ▁ " ; else { cout << arr2 [ j ++ ] << " ▁ " ; i ++ ; } }
while ( i < m ) cout << arr1 [ i ++ ] << " ▁ " ; while ( j < n ) cout << arr2 [ j ++ ] << " ▁ " ; }
int main ( ) { int arr1 [ ] = { 1 , 2 , 4 , 5 , 6 } ; int arr2 [ ] = { 2 , 3 , 5 , 7 } ; int m = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; int n = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; printUnion ( arr1 , arr2 , m , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printIntersection ( int arr1 [ ] , int arr2 [ ] , int m , int n ) { int i = 0 , j = 0 ; while ( i < m && j < n ) { if ( arr1 [ i ] < arr2 [ j ] ) i ++ ; else if ( arr2 [ j ] < arr1 [ i ] ) j ++ ; else { cout << arr2 [ j ] << " ▁ " ; i ++ ; j ++ ; } } }
int main ( ) { int arr1 [ ] = { 1 , 2 , 4 , 5 , 6 } ; int arr2 [ ] = { 2 , 3 , 5 , 7 } ; int m = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; int n = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ;
printIntersection ( arr1 , arr2 , m , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
bool printPath ( node * root , node * target_leaf ) {
if ( root == NULL ) return false ;
if ( root == target_leaf || printPath ( root -> left , target_leaf ) || printPath ( root -> right , target_leaf ) ) { cout << root -> data << " ▁ " ; return true ; } return false ; }
void getTargetLeaf ( node * Node , int * max_sum_ref , int curr_sum , node * * target_leaf_ref ) { if ( Node == NULL ) return ;
curr_sum = curr_sum + Node -> data ;
if ( Node -> left == NULL && Node -> right == NULL ) { if ( curr_sum > * max_sum_ref ) { * max_sum_ref = curr_sum ; * target_leaf_ref = Node ; } }
getTargetLeaf ( Node -> left , max_sum_ref , curr_sum , target_leaf_ref ) ; getTargetLeaf ( Node -> right , max_sum_ref , curr_sum , target_leaf_ref ) ; }
int maxSumPath ( node * Node ) {
if ( Node == NULL ) return 0 ; node * target_leaf ; int max_sum = INT_MIN ;
getTargetLeaf ( Node , & max_sum , 0 , & target_leaf ) ;
printPath ( Node , target_leaf ) ;
return max_sum ; }
node * newNode ( int data ) { node * temp = new node ; temp -> data = data ; temp -> left = NULL ; temp -> right = NULL ; return temp ; }
int main ( ) { node * root = NULL ; root = newNode ( 10 ) ; root -> left = newNode ( -2 ) ; root -> right = newNode ( 7 ) ; root -> left -> left = newNode ( 8 ) ; root -> left -> right = newNode ( -4 ) ; cout << " Following ▁ are ▁ the ▁ nodes ▁ on ▁ the ▁ maximum ▁ " " sum ▁ path ▁ STRNEWLINE " ; int sum = maxSumPath ( root ) ; cout << " Sum of the nodes is " return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void sort012 ( int a [ ] , int arr_size ) { int lo = 0 ; int hi = arr_size - 1 ; int mid = 0 ; while ( mid <= hi ) { switch ( a [ mid ] ) { case 0 : swap ( a [ lo ++ ] , a [ mid ++ ] ) ; break ; case 1 : mid ++ ; break ; case 2 : swap ( a [ mid ] , a [ hi -- ] ) ; break ; } } }
void printArray ( int arr [ ] , int arr_size ) { for ( int i = 0 ; i < arr_size ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; sort012 ( arr , n ) ; cout << " array ▁ after ▁ segregation ▁ " ; printArray ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printUnsorted ( int arr [ ] , int n ) { int s = 0 , e = n - 1 , i , max , min ;
for ( s = 0 ; s < n - 1 ; s ++ ) { if ( arr [ s ] > arr [ s + 1 ] ) break ; } if ( s == n - 1 ) { cout << " The ▁ complete ▁ array ▁ is ▁ sorted " ; return ; }
for ( e = n - 1 ; e > 0 ; e -- ) { if ( arr [ e ] < arr [ e - 1 ] ) break ; }
max = arr [ s ] ; min = arr [ s ] ; for ( i = s + 1 ; i <= e ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; if ( arr [ i ] < min ) min = arr [ i ] ; }
for ( i = 0 ; i < s ; i ++ ) { if ( arr [ i ] > min ) { s = i ; break ; } }
for ( i = n - 1 ; i >= e + 1 ; i -- ) { if ( arr [ i ] < max ) { e = i ; break ; } }
cout << " The ▁ unsorted ▁ subarray ▁ which " << " ▁ makes ▁ the ▁ given ▁ array " << endl << " sorted ▁ lies ▁ between ▁ the ▁ indees ▁ " << s << " ▁ and ▁ " << e ; return ; } int main ( ) { int arr [ ] = { 10 , 12 , 20 , 30 , 25 , 40 , 32 , 31 , 35 , 50 , 60 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printUnsorted ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int comp ( const void * a , const void * b ) { return * ( int * ) a > * ( int * ) b ; }
int findNumberOfTriangles ( int arr [ ] , int n ) {
qsort ( arr , n , sizeof ( arr [ 0 ] ) , comp ) ;
int count = 0 ;
for ( int i = 0 ; i < n - 2 ; ++ i ) {
int k = i + 2 ;
for ( int j = i + 1 ; j < n ; ++ j ) {
while ( k < n && arr [ i ] + arr [ j ] > arr [ k ] ) ++ k ;
if ( k > j ) count += k - j - 1 ; } } return count ; }
int main ( ) { int arr [ ] = { 10 , 21 , 22 , 100 , 101 , 200 , 300 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Total ▁ number ▁ of ▁ triangles ▁ possible ▁ is ▁ " << findNumberOfTriangles ( arr , size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findElement ( int arr [ ] , int n , int key ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return -1 ; }
int main ( ) { int arr [ ] = { 12 , 34 , 10 , 6 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int key = 40 ; int position = findElement ( arr , n , key ) ; if ( position == - 1 ) cout << " Element ▁ not ▁ found " ; else cout << " Element ▁ Found ▁ at ▁ Position : ▁ " << position + 1 ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int insertSorted ( int arr [ ] , int n , int key , int capacity ) {
if ( n >= capacity ) return n ; arr [ n ] = key ; return ( n + 1 ) ; }
int main ( ) { int arr [ 20 ] = { 12 , 16 , 20 , 40 , 50 , 70 } ; int capacity = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 6 ; int i , key = 26 ; cout << " Before Insertion : " for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ;
n = insertSorted ( arr , n , key , capacity ) ; cout << " After Insertion : " for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int findElement ( int arr [ ] , int n , int key ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == key ) return i ; return - 1 ; }
int deleteElement ( int arr [ ] , int n , int key ) {
int pos = findElement ( arr , n , key ) ; if ( pos == - 1 ) { cout << " Element ▁ not ▁ found " ; return n ; }
int i ; for ( i = pos ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; return n - 1 ; }
int main ( ) { int i ; int arr [ ] = { 10 , 50 , 30 , 40 , 20 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int key = 30 ; cout << " Array ▁ before ▁ deletion STRNEWLINE " ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; n = deleteElement ( arr , n , key ) ; cout << " Array after deletion " ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return -1 ; int mid = ( low + high ) / 2 ;
if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 7 , 8 , 9 , 10 } ; int n , key ; n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; key = 10 ; cout << " Index : ▁ " << binarySearch ( arr , 0 , n - 1 , key ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int insertSorted ( int arr [ ] , int n , int key , int capacity ) {
if ( n >= capacity ) return n ; int i ; for ( i = n - 1 ; ( i >= 0 && arr [ i ] > key ) ; i -- ) arr [ i + 1 ] = arr [ i ] ; arr [ i + 1 ] = key ; return ( n + 1 ) ; }
int main ( ) { int arr [ 20 ] = { 12 , 16 , 20 , 40 , 50 , 70 } ; int capacity = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 6 ; int i , key = 26 ; cout << " Before Insertion : " for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ;
n = insertSorted ( arr , n , key , capacity ) ; cout << " After Insertion : " for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int binarySearch ( int arr [ ] , int low , int high , int key ) ; int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return -1 ; int mid = ( low + high ) / 2 ; if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
int deleteElement ( int arr [ ] , int n , int key ) {
int pos = binarySearch ( arr , 0 , n - 1 , key ) ; if ( pos == -1 ) { cout << " Element ▁ not ▁ found " ; return n ; }
int i ; for ( i = pos ; i < n - 1 ; i ++ ) arr [ i ] = arr [ i + 1 ] ; return n - 1 ; }
int main ( ) { int i ; int arr [ ] = { 10 , 20 , 30 , 40 , 50 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int key = 30 ; cout << " Array ▁ before ▁ deletion STRNEWLINE " ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; n = deleteElement ( arr , n , key ) ; cout << " Array after deletion " ; for ( i = 0 ; i < n ; i ++ ) cout << arr [ i ] << " ▁ " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int equilibrium ( int arr [ ] , int n ) { int i , j ; int leftsum , rightsum ;
for ( i = 0 ; i < n ; ++ i ) { leftsum = 0 ; rightsum = 0 ;
for ( j = 0 ; j < i ; j ++ ) leftsum += arr [ j ] ;
for ( j = i + 1 ; j < n ; j ++ ) rightsum += arr [ j ] ;
if ( leftsum == rightsum ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { -7 , 1 , 5 , 2 , -4 , 3 , 0 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << equilibrium ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int equilibrium ( int arr [ ] , int n ) {
int sum = 0 ;
int leftsum = 0 ;
for ( int i = 0 ; i < n ; ++ i ) sum += arr [ i ] ; for ( int i = 0 ; i < n ; ++ i ) {
sum -= arr [ i ] ; if ( leftsum == sum ) return i ; leftsum += arr [ i ] ; }
return -1 ; }
int main ( ) { int arr [ ] = { -7 , 1 , 5 , 2 , -4 , 3 , 0 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " First ▁ equilibrium ▁ index ▁ is ▁ " << equilibrium ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int ceilSearch ( int arr [ ] , int low , int high , int x ) { int i ;
if ( x <= arr [ low ] ) return low ;
for ( i = low ; i < high ; i ++ ) { if ( arr [ i ] == x ) return i ;
if ( arr [ i ] < x && arr [ i + 1 ] >= x ) return i + 1 ; }
return -1 ; }
int main ( ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 3 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == -1 ) cout << " Ceiling ▁ of ▁ " << x << " ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " ; else cout << " ceiling ▁ of ▁ " << x << " ▁ is ▁ " << arr [ index ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int ceilSearch ( int arr [ ] , int low , int high , int x ) { int mid ;
if ( x <= arr [ low ] ) return low ;
if ( x > arr [ high ] ) return -1 ;
mid = ( low + high ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
else if ( arr [ mid ] < x ) { if ( mid + 1 <= high && x <= arr [ mid + 1 ] ) return mid + 1 ; else return ceilSearch ( arr , mid + 1 , high , x ) ; }
else { if ( mid - 1 >= low && x > arr [ mid - 1 ] ) return mid ; else return ceilSearch ( arr , low , mid - 1 , x ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 20 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == -1 ) cout << " Ceiling ▁ of ▁ " << x << " ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " ; else cout << " ceiling ▁ of ▁ " << x << " ▁ is ▁ " << arr [ index ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isPairSum ( int A [ ] , int N , int X ) { for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) {
if ( i == j ) continue ;
if ( A [ i ] + A [ j ] == X ) return true ;
if ( A [ i ] + A [ j ] > X ) break ; } }
return false ; }
int main ( ) { int arr [ ] = { 3 , 5 , 9 , 2 , 8 , 10 , 11 } ; int val = 17 ; int arrSize = * ( & arr + 1 ) - arr ; sort ( arr , arr + arrSize ) ;
cout << isPairSum ( arr , arrSize , val ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int isPairSum ( int A [ ] , int N , int X ) {
int i = 0 ;
int j = N - 1 ; while ( i < j ) {
if ( A [ i ] + A [ j ] == X ) return 1 ;
else if ( A [ i ] + A [ j ] < X ) i ++ ;
else j -- ; } return 0 ; }
int main ( ) {
int arr [ ] = { 3 , 5 , 9 , 2 , 8 , 10 , 11 } ;
int val = 17 ;
int arrSize = * ( & arr + 1 ) - arr ;
cout << ( bool ) isPairSum ( arr , arrSize , val ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define NUM_LINE  2 NEW_LINE #define NUM_STATION  4
int min ( int a , int b ) { return a < b ? a : b ; } int carAssembly ( int a [ ] [ NUM_STATION ] , int t [ ] [ NUM_STATION ] , int * e , int * x ) { int T1 [ NUM_STATION ] , T2 [ NUM_STATION ] , i ;
T1 [ 0 ] = e [ 0 ] + a [ 0 ] [ 0 ] ;
T2 [ 0 ] = e [ 1 ] + a [ 1 ] [ 0 ] ;
for ( i = 1 ; i < NUM_STATION ; ++ i ) { T1 [ i ] = min ( T1 [ i - 1 ] + a [ 0 ] [ i ] , T2 [ i - 1 ] + t [ 1 ] [ i ] + a [ 0 ] [ i ] ) ; T2 [ i ] = min ( T2 [ i - 1 ] + a [ 1 ] [ i ] , T1 [ i - 1 ] + t [ 0 ] [ i ] + a [ 1 ] [ i ] ) ; }
return min ( T1 [ NUM_STATION - 1 ] + x [ 0 ] , T2 [ NUM_STATION - 1 ] + x [ 1 ] ) ; }
int main ( ) { int a [ ] [ NUM_STATION ] = { { 4 , 5 , 3 , 2 } , { 2 , 10 , 1 , 4 } } ; int t [ ] [ NUM_STATION ] = { { 0 , 7 , 4 , 5 } , { 0 , 9 , 2 , 8 } } ; int e [ ] = { 10 , 12 } , x [ ] = { 18 , 7 } ; cout << carAssembly ( a , t , e , x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinInsertionsDP ( char str [ ] , int n ) {
int table [ n ] [ n ] , l , h , gap ; memset ( table , 0 , sizeof ( table ) ) ;
for ( gap = 1 ; gap < n ; ++ gap ) for ( l = 0 , h = gap ; h < n ; ++ l , ++ h ) table [ l ] [ h ] = ( str [ l ] == str [ h ] ) ? table [ l + 1 ] [ h - 1 ] : ( min ( table [ l ] [ h - 1 ] , table [ l + 1 ] [ h ] ) + 1 ) ;
return table [ 0 ] [ n - 1 ] ; }
int main ( ) { char str [ ] = " geeks " ; cout << findMinInsertionsDP ( str , strlen ( str ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
class node { public : int data ; node * left , * right ; } ;
int LISS ( node * root ) { if ( root == NULL ) return 0 ;
int size_excl = LISS ( root -> left ) + LISS ( root -> right ) ;
int size_incl = 1 ; if ( root -> left ) size_incl += LISS ( root -> left -> left ) + LISS ( root -> left -> right ) ; if ( root -> right ) size_incl += LISS ( root -> right -> left ) + LISS ( root -> right -> right ) ;
return max ( size_incl , size_excl ) ; }
node * newNode ( int data ) { node * temp = new node ( ) ; temp -> data = data ; temp -> left = temp -> right = NULL ; return temp ; }
int main ( ) {
node * root = newNode ( 20 ) ; root -> left = newNode ( 8 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 12 ) ; root -> left -> right -> left = newNode ( 10 ) ; root -> left -> right -> right = newNode ( 14 ) ; root -> right = newNode ( 22 ) ; root -> right -> right = newNode ( 25 ) ; cout << " Size ▁ of ▁ the ▁ Largest " << " ▁ Independent ▁ Set ▁ is ▁ " << LISS ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class Pair { public : int a ; int b ; } ;
int maxChainLength ( Pair arr [ ] , int n ) { int i , j , max = 0 ; int * mcl = new int [ sizeof ( int ) * n ] ;
for ( i = 0 ; i < n ; i ++ ) mcl [ i ] = 1 ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] . a > arr [ j ] . b && mcl [ i ] < mcl [ j ] + 1 ) mcl [ i ] = mcl [ j ] + 1 ;
for ( i = 0 ; i < n ; i ++ ) if ( max < mcl [ i ] ) max = mcl [ i ] ; return max ; }
int main ( ) { Pair arr [ ] = { { 5 , 24 } , { 15 , 25 } , { 27 , 40 } , { 50 , 60 } } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Length ▁ of ▁ maximum ▁ size ▁ chain ▁ is ▁ " << maxChainLength ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int minPalPartion ( string str ) {
int n = str . length ( ) ;
int C [ n ] [ n ] ; bool P [ n ] [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) { P [ i ] [ i ] = true ; C [ i ] [ i ] = 0 ; }
for ( int L = 2 ; L <= n ; L ++ ) {
for ( int i = 0 ; i < n - L + 1 ; i ++ ) {
int j = i + L - 1 ;
if ( L == 2 ) P [ i ] [ j ] = ( str [ i ] == str [ j ] ) ; else P [ i ] [ j ] = ( str [ i ] == str [ j ] ) && P [ i + 1 ] [ j - 1 ] ;
if ( P [ i ] [ j ] == true ) C [ i ] [ j ] = 0 ; else {
C [ i ] [ j ] = INT_MAX ; for ( int k = i ; k <= j - 1 ; k ++ ) C [ i ] [ j ] = min ( C [ i ] [ j ] , C [ i ] [ k ] + C [ k + 1 ] [ j ] + 1 ) ; } } }
return C [ 0 ] [ n - 1 ] ; }
int main ( ) { string str = " ababbbabbababa " ; cout << " Min ▁ cuts ▁ needed ▁ for ▁ Palindrome " " ▁ Partitioning ▁ is ▁ " << minPalPartion ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; # define NO_OF_CHARS  256
void badCharHeuristic ( string str , int size , int badchar [ NO_OF_CHARS ] ) { int i ;
for ( i = 0 ; i < NO_OF_CHARS ; i ++ ) badchar [ i ] = -1 ;
for ( i = 0 ; i < size ; i ++ ) badchar [ ( int ) str [ i ] ] = i ; }
void search ( string txt , string pat ) { int m = pat . size ( ) ; int n = txt . size ( ) ; int badchar [ NO_OF_CHARS ] ;
badCharHeuristic ( pat , m , badchar ) ;
while ( s <= ( n - m ) ) { int j = m - 1 ;
while ( j >= 0 && pat [ j ] == txt [ s + j ] ) j -- ;
if ( j < 0 ) { cout << " pattern ▁ occurs ▁ at ▁ shift ▁ = ▁ " << s << endl ;
s += ( s + m < n ) ? m - badchar [ txt [ s + m ] ] : 1 ; } else
s += max ( 1 , j - badchar [ txt [ s + j ] ] ) ; } }
int main ( ) { string txt = " ABAAABCD " ; string pat = " ABC " ; search ( txt , pat ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left , * right ; } ;
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = Node -> right = NULL ; return ( Node ) ; }
int getLevelDiff ( node * root ) {
if ( root == NULL ) return 0 ;
return root -> data - getLevelDiff ( root -> left ) - getLevelDiff ( root -> right ) ; }
int main ( ) { node * root = newNode ( 5 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 6 ) ; root -> left -> left = newNode ( 1 ) ; root -> left -> right = newNode ( 4 ) ; root -> left -> right -> left = newNode ( 3 ) ; root -> right -> right = newNode ( 8 ) ; root -> right -> right -> right = newNode ( 9 ) ; root -> right -> right -> left = newNode ( 7 ) ; cout << getLevelDiff ( root ) << " ▁ is ▁ the ▁ required ▁ difference STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define bool  int
class node { public : int data ; node * left ; node * right ; } ;
bool hasPathSum ( node * Node , int sum ) { if ( Node == NULL ) { return ( sum == 0 ) ; } else { bool ans = 0 ; int subSum = sum - Node -> data ; if ( subSum == 0 && Node -> left == NULL && Node -> right == NULL ) return 1 ; if ( Node -> left ) ans = ans || hasPathSum ( Node -> left , subSum ) ; if ( Node -> right ) ans = ans || hasPathSum ( Node -> right , subSum ) ; return ans ; } }
node * newnode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int main ( ) { int sum = 21 ;
node * root = newnode ( 10 ) ; root -> left = newnode ( 8 ) ; root -> right = newnode ( 2 ) ; root -> left -> left = newnode ( 3 ) ; root -> left -> right = newnode ( 5 ) ; root -> right -> left = newnode ( 2 ) ; if ( hasPathSum ( root , sum ) ) cout << " There ▁ is ▁ a ▁ root - to - leaf ▁ path ▁ with ▁ sum ▁ " << sum ; else cout << " There ▁ is ▁ no ▁ root - to - leaf ▁ path ▁ with ▁ sum ▁ " << sum ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class node { public : int data ; node * left , * right ; } ;
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = Node -> right = NULL ; return ( Node ) ; }
int treePathsSumUtil ( node * root , int val ) {
if ( root == NULL ) return 0 ;
val = ( val * 10 + root -> data ) ;
if ( root -> left == NULL && root -> right == NULL ) return val ;
return treePathsSumUtil ( root -> left , val ) + treePathsSumUtil ( root -> right , val ) ; }
int treePathsSum ( node * root ) {
return treePathsSumUtil ( root , 0 ) ; }
int main ( ) { node * root = newNode ( 6 ) ; root -> left = newNode ( 3 ) ; root -> right = newNode ( 5 ) ; root -> left -> left = newNode ( 2 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> right = newNode ( 4 ) ; root -> left -> right -> left = newNode ( 7 ) ; root -> left -> right -> right = newNode ( 4 ) ; cout << " Sum ▁ of ▁ all ▁ paths ▁ is ▁ " << treePathsSum ( root ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
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
peak = max ( peak , count ) ;
int s = peak + ( ( ( count - 1 ) * count ) >> 1 ) ; return s ; }
int main ( ) { int a [ ] = { 5 , 5 , 4 , 3 , 2 , 1 } ; int n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; cout << " Minimum ▁ number ▁ of ▁ chocolates ▁ = ▁ " << minChocolates ( a , n ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double sum ( int n ) { double i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; }
int main ( ) { int n = 5 ; cout << " Sum ▁ is ▁ " << sum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int nthTermOfTheSeries ( int n ) {
int nthTerm ;
if ( n % 2 == 0 ) nthTerm = pow ( n - 1 , 2 ) + n ;
else nthTerm = pow ( n + 1 , 2 ) + n ;
return nthTerm ; }
int main ( ) { int n ; n = 8 ; cout << nthTermOfTheSeries ( n ) << endl ; n = 12 ; cout << nthTermOfTheSeries ( n ) << endl ; n = 102 ; cout << nthTermOfTheSeries ( n ) << endl ; n = 999 ; cout << nthTermOfTheSeries ( n ) << endl ; n = 9999 ; cout << nthTermOfTheSeries ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; unsigned int Log2n ( unsigned int n ) { return ( n > 1 ) ? 1 + Log2n ( n / 2 ) : 0 ; }
int main ( ) { unsigned int n = 32 ; cout << Log2n ( n ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float AvgofSquareN ( int n ) { return ( float ) ( ( n + 1 ) * ( 2 * n + 1 ) ) / 6 ; }
int main ( ) { int n = 10 ; cout << AvgofSquareN ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int divisorSum ( int n ) { int sum = 0 ; for ( int i = 1 ; i <= n ; ++ i ) sum += ( n / i ) * i ; return sum ; }
int main ( ) { int n = 4 ; cout << " ▁ " << divisorSum ( n ) << endl ; n = 5 ; cout << " ▁ " << divisorSum ( n ) << endl ; return 0 ; }
#include <math.h> NEW_LINE #include <iostream> NEW_LINE #include <boost/format.hpp> NEW_LINE class gfg { public :
double sum ( int x , int n ) { double i , total = 1.0 ; for ( i = 1 ; i <= n ; i ++ ) total = total + ( pow ( x , i ) / i ) ; return total ; } } ;
int main ( ) { gfg g ; int x = 2 ; int n = 5 ; std :: cout << boost :: format ( " % .2f " ) % g . sum ( x , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool check ( int n ) { if ( n <= 0 ) return false ;
return 1162261467 % n == 0 ; }
int main ( ) { int n = 9 ; if ( check ( n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int per ( int n ) { int a = 3 , b = 0 , c = 2 , i ; int m ; if ( n == 0 ) return a ; if ( n == 1 ) return b ; if ( n == 2 ) return c ; while ( n > 2 ) { m = a + b ; a = b ; b = c ; c = m ; n -- ; } return m ; }
int main ( ) { int n = 9 ; cout << per ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countDivisors ( int n ) {
int count = 0 ;
for ( int i = 1 ; i <= sqrt ( n ) + 1 ; i ++ ) { if ( n % i == 0 )
count += ( n / i == i ) ? 1 : 2 ; } if ( count % 2 == 0 ) cout << " Even " << endl ; else cout << " Odd " << endl ; }
int main ( ) { cout << " The ▁ count ▁ of ▁ divisor : ▁ " ; countDivisors ( 10 ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int countSquares ( int m , int n ) {
if ( n < m ) swap ( m , n ) ;
return m * ( m + 1 ) * ( 2 * m + 1 ) / 6 + ( n - m ) * m * ( m + 1 ) / 2 ; }
int main ( ) { int m = 4 , n = 3 ; cout << " Count ▁ of ▁ squares ▁ is ▁ " << countSquares ( m , n ) ; }
#include <iostream> NEW_LINE using namespace std ;
class gfg { public : double sum ( int n ) { double i , s = 0.0 ; for ( i = 1 ; i <= n ; i ++ ) s = s + 1 / i ; return s ; } } ;
int main ( ) { gfg g ; int n = 5 ; cout << " Sum ▁ is ▁ " << g . sum ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
int main ( ) { int a = 98 , b = 56 ; cout << " GCD ▁ of ▁ " << a << " ▁ and ▁ " << b << " ▁ is ▁ " << gcd ( a , b ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void printArray ( int arr [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) cout << " ▁ " << arr [ i ] ; cout << " STRNEWLINE " ; return ; }
void printSequencesRecur ( int arr [ ] , int n , int k , int index ) { int i ; if ( k == 0 ) { printArray ( arr , index ) ; } if ( k > 0 ) { for ( i = 1 ; i <= n ; ++ i ) { arr [ index ] = i ; printSequencesRecur ( arr , n , k - 1 , index + 1 ) ; } } }
void printSequences ( int n , int k ) { int * arr = new int [ k ] ; printSequencesRecur ( arr , n , k , 0 ) ; return ; }
int main ( ) { int n = 3 ; int k = 2 ; printSequences ( n , k ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
bool isMultipleof5 ( int n ) { while ( n > 0 ) n = n - 5 ; if ( n == 0 ) return true ; return false ; }
int main ( ) { int n = 19 ; if ( isMultipleof5 ( n ) == true ) cout << n << " ▁ is ▁ multiple ▁ of ▁ 5" ; else cout << n << " ▁ is ▁ not ▁ a ▁ multiple ▁ of ▁ 5" ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int isKthBitSet ( int x , int k ) { return ( x & ( 1 << ( k - 1 ) ) ) ? 1 : 0 ; }
int leftmostSetBit ( int x ) { int count = 0 ; while ( x ) { count ++ ; x = x >> 1 ; } return count ; }
int isBinPalindrome ( int x ) { int l = leftmostSetBit ( x ) ; int r = 1 ;
while ( l > r ) {
if ( isKthBitSet ( x , l ) != isKthBitSet ( x , r ) ) return 0 ; l -- ; r ++ ; } return 1 ; } int findNthPalindrome ( int n ) { int pal_count = 0 ;
int i = 0 ; for ( i = 1 ; i <= INT_MAX ; i ++ ) { if ( isBinPalindrome ( i ) ) { pal_count ++ ; }
if ( pal_count == n ) break ; } return i ; }
int main ( ) { int n = 9 ;
cout << findNthPalindrome ( n ) ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) {
int a = 5 , b = 9 ;
cout << " a ▁ = ▁ " << a << " , " << " ▁ b ▁ = ▁ " << b << endl ; cout << " a ▁ & ▁ b ▁ = ▁ " << ( a & b ) << endl ;
cout << " a ▁ | ▁ b ▁ = ▁ " << ( a b ) << endl ;
cout << " a ▁ ^ ▁ b ▁ = ▁ " << ( a ^ b ) << endl ;
cout << " ~ ( " << a << " ) ▁ = ▁ " << ( ~ a ) << endl ;
cout << " b ▁ < < ▁ 1" << " ▁ = ▁ " << ( b << 1 ) << endl ;
cout << " b ▁ > > ▁ 1 ▁ " << " = ▁ " << ( b >> 1 ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
double temp_convert ( int F1 , int B1 , int F2 , int B2 , int T ) { float t2 ;
t2 = F2 + ( float ) ( B2 - F2 ) / ( B1 - F1 ) * ( T - F1 ) ; return t2 ; }
int main ( ) { int F1 = 0 , B1 = 100 ; int F2 = 32 , B2 = 212 ; int T = 37 ; float t2 ; cout << temp_convert ( F1 , B1 , F2 , B2 , T ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
int maxDepth ( node * node ) { if ( node == NULL ) return 0 ; else {
int lDepth = maxDepth ( node -> left ) ; int rDepth = maxDepth ( node -> right ) ;
if ( lDepth > rDepth ) return ( lDepth + 1 ) ; else return ( rDepth + 1 ) ; } }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int main ( ) { node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; cout << " Height ▁ of ▁ tree ▁ is ▁ " << maxDepth ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
int height ( node * node ) ;
bool isBalanced ( node * root ) {
int lh ;
int rh ;
if ( root == NULL ) return 1 ;
lh = height ( root -> left ) ; rh = height ( root -> right ) ; if ( abs ( lh - rh ) <= 1 && isBalanced ( root -> left ) && isBalanced ( root -> right ) ) return 1 ;
return 0 ; }
int max ( int a , int b ) { return ( a >= b ) ? a : b ; }
int height ( node * node ) {
if ( node == NULL ) return 0 ;
return 1 + max ( height ( node -> left ) , height ( node -> right ) ) ; }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int main ( ) { node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> left -> left -> left = newNode ( 8 ) ; if ( isBalanced ( root ) ) cout << " Tree ▁ is ▁ balanced " ; else cout << " Tree ▁ is ▁ not ▁ balanced " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct node { int data ; struct node * left , * right ; } ;
struct node * newNode ( int data ) { struct node * node = ( struct node * ) malloc ( sizeof ( struct node ) ) ; node -> data = data ; node -> left = NULL ; node -> right = NULL ; return ( node ) ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int height ( struct node * node ) {
if ( node == NULL ) return 0 ;
return 1 + max ( height ( node -> left ) , height ( node -> right ) ) ; }
int diameter ( struct node * tree ) {
if ( tree == NULL ) return 0 ;
int lheight = height ( tree -> left ) ; int rheight = height ( tree -> right ) ;
int ldiameter = diameter ( tree -> left ) ; int rdiameter = diameter ( tree -> right ) ;
return max ( lheight + rheight + 1 , max ( ldiameter , rdiameter ) ) ; }
int main ( ) {
struct node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ;
cout << " Diameter ▁ of ▁ the ▁ given ▁ binary ▁ tree ▁ is ▁ " << diameter ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findpath ( int N , int a [ ] ) {
if ( a [ 0 ] ) {
cout << " ▁ " << N + 1 ; for ( int i = 1 ; i <= N ; i ++ ) cout << " ▁ " << i ; return ; }
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( ! a [ i ] && a [ i + 1 ] ) {
for ( int j = 1 ; j <= i ; j ++ ) cout << " ▁ " << j ; cout << " ▁ " << N + 1 ; for ( int j = i + 1 ; j <= N ; j ++ ) cout << " ▁ " << j ; return ; } }
for ( int i = 1 ; i <= N ; i ++ ) cout << " ▁ " << i ; cout << " ▁ " << N + 1 ; }
int main ( ) {
int N = 3 , arr [ ] = { 0 , 1 , 0 } ;
findpath ( N , arr ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
struct Node { int data ; struct Node * left , * right ; } ;
struct Node * newNode ( int data ) { struct Node * node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; node -> data = data ; node -> left = node -> right = NULL ; return node ; }
int depthOfOddLeafUtil ( struct Node * root , int level ) {
if ( root == NULL ) return 0 ;
if ( root -> left == NULL && root -> right == NULL && level & 1 ) return level ;
return max ( depthOfOddLeafUtil ( root -> left , level + 1 ) , depthOfOddLeafUtil ( root -> right , level + 1 ) ) ; }
int depthOfOddLeaf ( struct Node * root ) { int level = 1 , depth = 0 ; return depthOfOddLeafUtil ( root , level ) ; }
int main ( ) { struct Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> right -> left = newNode ( 5 ) ; root -> right -> right = newNode ( 6 ) ; root -> right -> left -> right = newNode ( 7 ) ; root -> right -> right -> right = newNode ( 8 ) ; root -> right -> left -> right -> left = newNode ( 9 ) ; root -> right -> right -> right -> right = newNode ( 10 ) ; root -> right -> right -> right -> right -> left = newNode ( 11 ) ; cout << depthOfOddLeaf ( root ) << " ▁ is ▁ the ▁ required ▁ depth " ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printArr ( int arr [ ] , int n ) {
sort ( arr , arr + n ) ;
if ( arr [ 0 ] == arr [ n - 1 ] ) { cout << " No " << endl ; }
else { cout << " Yes " << endl ; for ( int i = 0 ; i < n ; i ++ ) { cout << arr [ i ] << " ▁ " ; } } }
int main ( ) {
int arr [ ] = { 1 , 2 , 2 , 1 , 3 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
printArr ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
int getWidth ( node * root , int level ) ; int height ( node * node ) ; node * newNode ( int data ) ;
int getMaxWidth ( node * root ) { int maxWidth = 0 ; int width ; int h = height ( root ) ; int i ;
for ( i = 1 ; i <= h ; i ++ ) { width = getWidth ( root , i ) ; if ( width > maxWidth ) maxWidth = width ; } return maxWidth ; }
int getWidth ( node * root , int level ) { if ( root == NULL ) return 0 ; if ( level == 1 ) return 1 ; else if ( level > 1 ) return getWidth ( root -> left , level - 1 ) + getWidth ( root -> right , level - 1 ) ; }
int height ( node * node ) { if ( node == NULL ) return 0 ; else {
int lHeight = height ( node -> left ) ; int rHeight = height ( node -> right ) ;
return ( lHeight > rHeight ) ? ( lHeight + 1 ) : ( rHeight + 1 ) ; } }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int main ( ) {
node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> right = newNode ( 8 ) ; root -> right -> right -> left = newNode ( 6 ) ; root -> right -> right -> right = newNode ( 7 ) ;
cout << " Maximum ▁ width ▁ is ▁ " << getMaxWidth ( root ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
int height ( node * node ) ; node * newNode ( int data ) ; void getMaxWidthRecur ( node * root , int count [ ] , int level ) ; int getMax ( int arr [ ] , int n ) ;
int getMaxWidth ( node * root ) { int width ; int h = height ( root ) ;
int * count = new int [ h ] ; int level = 0 ;
getMaxWidthRecur ( root , count , level ) ;
return getMax ( count , h ) ; }
void getMaxWidthRecur ( node * root , int count [ ] , int level ) { if ( root ) { count [ level ] ++ ; getMaxWidthRecur ( root -> left , count , level + 1 ) ; getMaxWidthRecur ( root -> right , count , level + 1 ) ; } }
int height ( node * node ) { if ( node == NULL ) return 0 ; else {
int lHeight = height ( node -> left ) ; int rHeight = height ( node -> right ) ;
return ( lHeight > rHeight ) ? ( lHeight + 1 ) : ( rHeight + 1 ) ; } }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int getMax ( int arr [ ] , int n ) { int max = arr [ 0 ] ; int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; } return max ; }
int main ( ) {
node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> right = newNode ( 8 ) ; root -> right -> right -> left = newNode ( 6 ) ; root -> right -> right -> right = newNode ( 7 ) ; cout << " Maximum ▁ width ▁ is ▁ " << getMaxWidth ( root ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct node { int data ; struct node * left ; struct node * right ; } ;
unsigned int getLeafCount ( struct node * node ) { if ( node == NULL ) return 0 ; if ( node -> left == NULL && node -> right == NULL ) return 1 ; else return getLeafCount ( node -> left ) + getLeafCount ( node -> right ) ; }
struct node * newNode ( int data ) { struct node * node = ( struct node * ) malloc ( sizeof ( struct node ) ) ; node -> data = data ; node -> left = NULL ; node -> right = NULL ; return ( node ) ; }
int main ( ) {
struct node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ;
cout << " Leaf ▁ count ▁ of ▁ the ▁ tree ▁ is ▁ : ▁ " << getLeafCount ( root ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int deno [ ] = { 1 , 2 , 5 , 10 , 20 , 50 , 100 , 500 , 1000 } ; int n = sizeof ( deno ) / sizeof ( deno [ 0 ] ) ; void findMin ( int V ) { sort ( deno , deno + n ) ;
vector < int > ans ;
for ( int i = n - 1 ; i >= 0 ; i -- ) {
while ( V >= deno [ i ] ) { V -= deno [ i ] ; ans . push_back ( deno [ i ] ) ; } }
for ( int i = 0 ; i < ans . size ( ) ; i ++ ) cout << ans [ i ] << " ▁ " ; }
int main ( ) { int n = 93 ; cout << " Following ▁ is ▁ minimal " << " ▁ number ▁ of ▁ change ▁ for ▁ " << n << " : ▁ " ; findMin ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; node * nextRight ; node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; this -> nextRight = NULL ; } } ;
node * getNextRight ( node * p ) { node * temp = p -> nextRight ;
while ( temp != NULL ) { if ( temp -> left != NULL ) return temp -> left ; if ( temp -> right != NULL ) return temp -> right ; temp = temp -> nextRight ; }
return NULL ; }
void connectRecur ( node * p ) { node * temp ; if ( ! p ) return ;
p -> nextRight = NULL ;
while ( p != NULL ) { node * q = p ;
while ( q != NULL ) {
if ( q -> left ) {
if ( q -> right ) q -> left -> nextRight = q -> right ; else q -> left -> nextRight = getNextRight ( q ) ; } if ( q -> right ) q -> right -> nextRight = getNextRight ( q ) ;
q = q -> nextRight ; }
if ( p -> left ) p = p -> left ; else if ( p -> right ) p = p -> right ; else p = getNextRight ( p ) ; } }
int main ( ) {
node * root = new node ( 10 ) ; root -> left = new node ( 8 ) ; root -> right = new node ( 2 ) ; root -> left -> left = new node ( 3 ) ; root -> right -> right = new node ( 90 ) ;
connectRecur ( root ) ;
cout << " Following ▁ are ▁ populated ▁ nextRight ▁ pointers ▁ in ▁ the ▁ tree " " ▁ ( -1 ▁ is ▁ printed ▁ if ▁ there ▁ is ▁ no ▁ nextRight ) ▁ STRNEWLINE " ; cout << " nextRight ▁ of ▁ " << root -> data << " ▁ is ▁ " << ( root -> nextRight ? root -> nextRight -> data : -1 ) << endl ; cout << " nextRight ▁ of ▁ " << root -> left -> data << " ▁ is ▁ " << ( root -> left -> nextRight ? root -> left -> nextRight -> data : -1 ) << endl ; cout << " nextRight ▁ of ▁ " << root -> right -> data << " ▁ is ▁ " << ( root -> right -> nextRight ? root -> right -> nextRight -> data : -1 ) << endl ; cout << " nextRight ▁ of ▁ " << root -> left -> left -> data << " ▁ is ▁ " << ( root -> left -> left -> nextRight ? root -> left -> left -> nextRight -> data : -1 ) << endl ; cout << " nextRight ▁ of ▁ " << root -> right -> right -> data << " ▁ is ▁ " << ( root -> right -> right -> nextRight ? root -> right -> right -> nextRight -> data : -1 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; node * nextRight ; node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; this -> nextRight = NULL ; } } ; void connectRecur ( node * p ) ;
void connect ( node * p ) {
p -> nextRight = NULL ;
connectRecur ( p ) ; }
void connectRecur ( node * p ) {
if ( ! p ) return ;
if ( p -> left ) p -> left -> nextRight = p -> right ;
if ( p -> right ) p -> right -> nextRight = ( p -> nextRight ) ? p -> nextRight -> left : NULL ;
connectRecur ( p -> left ) ; connectRecur ( p -> right ) ; }
int main ( ) {
node * root = new node ( 10 ) ; root -> left = new node ( 8 ) ; root -> right = new node ( 2 ) ; root -> left -> left = new node ( 3 ) ;
connect ( root ) ;
cout << " Following ▁ are ▁ populated ▁ nextRight ▁ pointers ▁ in ▁ the ▁ tree " " ▁ ( -1 ▁ is ▁ printed ▁ if ▁ there ▁ is ▁ no ▁ nextRight ) STRNEWLINE " ; cout << " nextRight ▁ of ▁ " << root -> data << " ▁ is ▁ " << ( root -> nextRight ? root -> nextRight -> data : -1 ) << endl ; cout << " nextRight ▁ of ▁ " << root -> left -> data << " ▁ is ▁ " << ( root -> left -> nextRight ? root -> left -> nextRight -> data : -1 ) << endl ; cout << " nextRight ▁ of ▁ " << root -> right -> data << " ▁ is ▁ " << ( root -> right -> nextRight ? root -> right -> nextRight -> data : -1 ) << endl ; cout << " nextRight ▁ of ▁ " << root -> left -> left -> data << " ▁ is ▁ " << ( root -> left -> left -> nextRight ? root -> left -> left -> nextRight -> data : -1 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinInsertions ( char str [ ] , int l , int h ) {
if ( l > h ) return INT_MAX ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) ; }
int main ( ) { char str [ ] = " geeks " ; cout << findMinInsertions ( str , 0 , strlen ( str ) - 1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; cout << " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ " << lps ( seq , 0 , n - 1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct node { int data ; struct node * left ; struct node * right ; } ;
int getLevelUtil ( struct node * node , int data , int level ) { if ( node == NULL ) return 0 ; if ( node -> data == data ) return level ; int downlevel = getLevelUtil ( node -> left , data , level + 1 ) ; if ( downlevel != 0 ) return downlevel ; downlevel = getLevelUtil ( node -> right , data , level + 1 ) ; return downlevel ; }
int getLevel ( struct node * node , int data ) { return getLevelUtil ( node , data , 1 ) ; }
struct node * newNode ( int data ) { struct node * temp = new struct node ; temp -> data = data ; temp -> left = NULL ; temp -> right = NULL ; return temp ; }
int main ( ) { struct node * root = new struct node ; int x ;
root = newNode ( 3 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 5 ) ; root -> left -> left = newNode ( 1 ) ; root -> left -> right = newNode ( 4 ) ; for ( x = 1 ; x <= 5 ; x ++ ) { int level = getLevel ( root , x ) ; if ( level ) cout << " Level ▁ of ▁ " << x << " ▁ is ▁ " << getLevel ( root , x ) << endl ; else cout << x << " is ▁ not ▁ present ▁ in ▁ tree " << endl ; } getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define NO_OF_CHARS  256 NEW_LINE int getNextState ( string pat , int M , int state , int x ) {
if ( state < M && x == pat [ state ] ) return state + 1 ;
int ns , i ;
for ( ns = state ; ns > 0 ; ns -- ) { if ( pat [ ns - 1 ] == x ) { for ( i = 0 ; i < ns - 1 ; i ++ ) if ( pat [ i ] != pat [ state - ns + 1 + i ] ) break ; if ( i == ns - 1 ) return ns ; } } return 0 ; }
void computeTF ( string pat , int M , int TF [ ] [ NO_OF_CHARS ] ) { int state , x ; for ( state = 0 ; state <= M ; ++ state ) for ( x = 0 ; x < NO_OF_CHARS ; ++ x ) TF [ state ] [ x ] = getNextState ( pat , M , state , x ) ; }
void search ( string pat , string txt ) { int M = pat . size ( ) ; int N = txt . size ( ) ; int TF [ M + 1 ] [ NO_OF_CHARS ] ; computeTF ( pat , M , TF ) ;
int i , state = 0 ; for ( i = 0 ; i < N ; i ++ ) { state = TF [ state ] [ txt [ i ] ] ; if ( state == M ) cout << " ▁ Pattern ▁ found ▁ at ▁ index ▁ " << i - M + 1 << endl ; } }
int main ( ) { string txt = " AABAACAADAABAAABAA " ; string pat = " AABA " ; search ( pat , txt ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int key ; struct Node * left , * right ; } ;
struct Node * newNode ( int key ) { struct Node * n = ( struct Node * ) malloc ( sizeof ( struct Node * ) ) ; if ( n != NULL ) { n -> key = key ; n -> left = NULL ; n -> right = NULL ; return n ; } else { cout << " Memory ▁ allocation ▁ failed ! " << endl ; exit ( 1 ) ; } }
int findMirrorRec ( int target , struct Node * left , struct Node * right ) {
if ( left == NULL right == NULL ) return 0 ;
if ( left -> key == target ) return right -> key ; if ( right -> key == target ) return left -> key ;
int mirror_val = findMirrorRec ( target , left -> left , right -> right ) ; if ( mirror_val ) return mirror_val ;
findMirrorRec ( target , left -> right , right -> left ) ; }
int findMirror ( struct Node * root , int target ) { if ( root == NULL ) return 0 ; if ( root -> key == target ) return target ; return findMirrorRec ( target , root -> left , root -> right ) ; }
int main ( ) { struct Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> left -> right = newNode ( 7 ) ; root -> right = newNode ( 3 ) ; root -> right -> left = newNode ( 5 ) ; root -> right -> right = newNode ( 6 ) ; root -> right -> left -> left = newNode ( 8 ) ; root -> right -> left -> right = newNode ( 9 ) ;
int target = root -> left -> left -> key ; int mirror = findMirror ( root , target ) ; if ( mirror ) cout << " Mirror ▁ of ▁ Node ▁ " << target << " ▁ is ▁ Node ▁ " << mirror << endl ; else cout << " Mirror ▁ of ▁ Node ▁ " << target << " ▁ is ▁ NULL ! ▁ " << endl ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ;
node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; } } ;
bool iterativeSearch ( node * root , int x ) {
if ( root == NULL ) return false ;
queue < node * > q ;
q . push ( root ) ;
while ( q . empty ( ) == false ) {
node * node = q . front ( ) ; if ( node -> data == x ) return true ;
q . pop ( ) ; if ( node -> left != NULL ) q . push ( node -> left ) ; if ( node -> right != NULL ) q . push ( node -> right ) ; } return false ; }
int main ( ) { node * NewRoot = NULL ; node * root = new node ( 2 ) ; root -> left = new node ( 7 ) ; root -> right = new node ( 5 ) ; root -> left -> right = new node ( 6 ) ; root -> left -> right -> left = new node ( 1 ) ; root -> left -> right -> right = new node ( 11 ) ; root -> right -> right = new node ( 9 ) ; root -> right -> right -> left = new node ( 4 ) ; iterativeSearch ( root , 6 ) ? cout << " Found STRNEWLINE " : cout << " Not ▁ Found STRNEWLINE " ; iterativeSearch ( root , 12 ) ? cout << " Found STRNEWLINE " : cout << " Not ▁ Found STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class node { public : int data ; node * left ; node * right ; node * next ; } ;
void populateNext ( node * root ) {
node * next = NULL ; populateNextRecur ( root , & next ) ; }
void populateNextRecur ( node * p , node * * next_ref ) { if ( p ) {
populateNextRecur ( p -> right , next_ref ) ;
p -> next = * next_ref ;
* next_ref = p ;
populateNextRecur ( p -> left , next_ref ) ; } }
#include <iostream> NEW_LINE using namespace std ; void printSubstrings ( string str ) {
for ( int i = 0 ; i < n ; i ++ ) {
for ( int j = i ; j < n ; j ++ ) {
for ( int k = i ; k <= j ; k ++ ) { cout << str [ k ] ; }
cout << endl ; } } }
int main ( ) { string str = " abcd " ;
printSubstrings ( str ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
#define N  9
void print ( int arr [ N ] [ N ] ) { for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) cout << arr [ i ] [ j ] << " ▁ " ; cout << endl ; } }
bool isSafe ( int grid [ N ] [ N ] , int row , int col , int num ) {
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ row ] [ x ] == num ) return false ;
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ x ] [ col ] == num ) return false ;
int startRow = row - row % 3 , startCol = col - col % 3 ; for ( int i = 0 ; i < 3 ; i ++ ) for ( int j = 0 ; j < 3 ; j ++ ) if ( grid [ i + startRow ] [ j + startCol ] == num ) return false ; return true ; }
bool solveSuduko ( int grid [ N ] [ N ] , int row , int col ) {
if ( row == N - 1 && col == N ) return true ;
if ( col == N ) { row ++ ; col = 0 ; }
if ( grid [ row ] [ col ] > 0 ) return solveSuduko ( grid , row , col + 1 ) ; for ( int num = 1 ; num <= N ; num ++ ) {
if ( isSafe ( grid , row , col , num ) ) {
grid [ row ] [ col ] = num ;
if ( solveSuduko ( grid , row , col + 1 ) ) return true ; }
grid [ row ] [ col ] = 0 ; } return false ; }
int grid [ N ] [ N ] = { { 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 } , { 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 } , { 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 } , { 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 } , { 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 } , { 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 } , { 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 } , { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 } , { 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 } } ; if ( solveSuduko ( grid , 0 , 0 ) ) print ( grid ) ; else cout << " no ▁ solution ▁ exists ▁ " << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ARRAYSIZE ( a )  (sizeof(a))/(sizeof(a[0])) NEW_LINE static int total_nodes ;
void printSubset ( int A [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) { cout << " ▁ " << A [ i ] ; } cout << " STRNEWLINE " ; }
int comparator ( const void * pLhs , const void * pRhs ) { int * lhs = ( int * ) pLhs ; int * rhs = ( int * ) pRhs ; return * lhs > * rhs ; }
void subset_sum ( int s [ ] , int t [ ] , int s_size , int t_size , int sum , int ite , int const target_sum ) { total_nodes ++ ; if ( target_sum == sum ) {
printSubset ( t , t_size ) ;
if ( ite + 1 < s_size && sum - s [ ite ] + s [ ite + 1 ] <= target_sum ) {
subset_sum ( s , t , s_size , t_size - 1 , sum - s [ ite ] , ite + 1 , target_sum ) ; } return ; } else {
if ( ite < s_size && sum + s [ ite ] <= target_sum ) {
for ( int i = ite ; i < s_size ; i ++ ) { t [ t_size ] = s [ i ] ; if ( sum + s [ i ] <= target_sum ) {
subset_sum ( s , t , s_size , t_size + 1 , sum + s [ i ] , i + 1 , target_sum ) ; } } } } }
void generateSubsets ( int s [ ] , int size , int target_sum ) { int * tuplet_vector = ( int * ) malloc ( size * sizeof ( int ) ) ; int total = 0 ;
qsort ( s , size , sizeof ( int ) , & comparator ) ; for ( int i = 0 ; i < size ; i ++ ) { total += s [ i ] ; } if ( s [ 0 ] <= target_sum && total >= target_sum ) { subset_sum ( s , tuplet_vector , size , 0 , 0 , 0 , target_sum ) ; } free ( tuplet_vector ) ; }
int main ( ) { int weights [ ] = { 15 , 22 , 14 , 26 , 32 , 9 , 16 , 8 } ; int target = 53 ; int size = ARRAYSIZE ( weights ) ; generateSubsets ( weights , size , target ) ; cout << " Nodes ▁ generated ▁ " << total_nodes ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printPairs ( int arr [ ] , int arr_size , int sum ) { unordered_set < int > s ; for ( int i = 0 ; i < arr_size ; i ++ ) { int temp = sum - arr [ i ] ;
if ( s . find ( temp ) != s . end ( ) ) cout << " Pair ▁ with ▁ given ▁ sum ▁ " << sum << " ▁ is ▁ ( " << arr [ i ] << " , " << temp << " ) " << endl ; s . insert ( arr [ i ] ) ; } }
int main ( ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int n = 16 ; int arr_size = sizeof ( A ) / sizeof ( A [ 0 ] ) ; printPairs ( A , arr_size , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int exponentMod ( int A , int B , int C ) {
if ( A == 0 ) return 0 ; if ( B == 0 ) return 1 ;
long y ; if ( B % 2 == 0 ) { y = exponentMod ( A , B / 2 , C ) ; y = ( y * y ) % C ; }
else { y = A % C ; y = ( y * exponentMod ( A , B - 1 , C ) % C ) % C ; } return ( int ) ( ( y + C ) % C ) ; }
int main ( ) { int A = 2 , B = 5 , C = 13 ; cout << " Power ▁ is ▁ " << exponentMod ( A , B , C ) ; return 0 ; }
int power ( int x , int y ) {
int res = 1 ; while ( y ) {
if ( y % 2 == 1 ) res = ( res * x ) ;
y = y >> 1 ;
x = ( x * x ) ; } return res ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int eggDrop ( int n , int k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; int min = INT_MAX , x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
int main ( ) { int n = 2 , k = 10 ; cout << " Minimum ▁ number ▁ of ▁ trials ▁ " " in ▁ worst ▁ case ▁ with ▁ " << n << " ▁ eggs ▁ and ▁ " << k << " ▁ floors ▁ is ▁ " << eggDrop ( n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * left ; struct Node * right ; Node ( int val ) { data = val ; left = NULL ; right = NULL ; } } ;
int main ( ) { struct Node * root = new Node ( 1 ) ; root -> left = new Node ( 2 ) ; root -> right = new Node ( 3 ) ; root -> left -> left = new Node ( 4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
class Node { public : int data ; Node * left , * right ;
Node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; } } ;
int findMax ( Node * root ) {
if ( root == NULL ) return INT_MIN ;
int res = root -> data ; int lres = findMax ( root -> left ) ; int rres = findMax ( root -> right ) ; if ( lres > res ) res = lres ; if ( rres > res ) res = rres ; return res ; }
int main ( ) { Node * NewRoot = NULL ; Node * root = new Node ( 2 ) ; root -> left = new Node ( 7 ) ; root -> right = new Node ( 5 ) ; root -> left -> right = new Node ( 6 ) ; root -> left -> right -> left = new Node ( 1 ) ; root -> left -> right -> right = new Node ( 11 ) ; root -> right -> right = new Node ( 9 ) ; root -> right -> right -> left = new Node ( 4 ) ;
cout << " Maximum ▁ element ▁ is ▁ " << findMax ( root ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * left , * right ; } ;
Node * extractLeafList ( Node * root , Node * * head_ref ) { if ( root == NULL ) return NULL ; if ( root -> left == NULL && root -> right == NULL ) { root -> right = * head_ref ; if ( * head_ref != NULL ) ( * head_ref ) -> left = root ; * head_ref = root ; return NULL ; } root -> right = extractLeafList ( root -> right , head_ref ) ; root -> left = extractLeafList ( root -> left , head_ref ) ; return root ; }
Node * newNode ( int data ) { Node * node = new Node ( ) ; node -> data = data ; node -> left = node -> right = NULL ; return node ; }
void print ( Node * root ) { if ( root != NULL ) { print ( root -> left ) ; cout << root -> data << " ▁ " ; print ( root -> right ) ; } }
void printList ( Node * head ) { while ( head ) { cout << head -> data << " ▁ " ; head = head -> right ; } }
int main ( ) { Node * head = NULL ; Node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> right = newNode ( 6 ) ; root -> left -> left -> left = newNode ( 7 ) ; root -> left -> left -> right = newNode ( 8 ) ; root -> right -> right -> left = newNode ( 9 ) ; root -> right -> right -> right = newNode ( 10 ) ; cout << " Inorder ▁ Trvaersal ▁ of ▁ given ▁ Tree ▁ is : STRNEWLINE " ; print ( root ) ; root = extractLeafList ( root , & head ) ; cout << " Extracted Double Linked list is : " ; printList ( head ) ; cout << " Inorder traversal of modified tree is : " print ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long int countNumberOfStrings ( string s ) {
int length = s . length ( ) ;
int n = length - 1 ;
long long int count = pow ( 2 , n ) ; return count ; }
int main ( ) { string S = " ABCD " ; cout << countNumberOfStrings ( S ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void makeArraySumEqual ( int a [ ] , int N ) {
int count_0 = 0 , count_1 = 0 ;
int odd_sum = 0 , even_sum = 0 ; for ( int i = 0 ; i < N ; i ++ ) {
if ( a [ i ] == 0 ) count_0 ++ ;
else count_1 ++ ;
if ( ( i + 1 ) % 2 == 0 ) even_sum += a [ i ] ; else if ( ( i + 1 ) % 2 > 0 ) odd_sum += a [ i ] ; }
if ( odd_sum == even_sum ) {
for ( int i = 0 ; i < N ; i ++ ) cout << " ▁ " << a [ i ] ; }
else { if ( count_0 >= N / 2 ) {
for ( int i = 0 ; i < count_0 ; i ++ ) cout << "0 ▁ " ; } else {
int is_Odd = count_1 % 2 ;
count_1 -= is_Odd ;
for ( int i = 0 ; i < count_1 ; i ++ ) cout << "1 ▁ " ; } } }
int main ( ) {
int arr [ ] = { 1 , 1 , 1 , 0 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
makeArraySumEqual ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countDigitSum ( int N , int K ) {
int l = ( int ) pow ( 10 , N - 1 ) , r = ( int ) pow ( 10 , N ) - 1 ; int count = 0 ; for ( int i = l ; i <= r ; i ++ ) { int num = i ;
int digits [ N ] ; for ( int j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num /= 10 ; } int sum = 0 , flag = 0 ;
for ( int j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( int j = K ; j < N ; j ++ ) { if ( sum - digits [ j - K ] + digits [ j ] != sum ) { flag = 1 ; break ; } } if ( flag == 0 ) count ++ ; } return count ; }
int main ( ) {
int N = 2 , K = 1 ; cout << countDigitSum ( N , K ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findpath ( int N , int a [ ] ) {
if ( a [ 0 ] ) {
cout << " ▁ " << N + 1 ; for ( int i = 1 ; i <= N ; i ++ ) cout << " ▁ " << i ; return ; }
for ( int i = 0 ; i < N - 1 ; i ++ ) { if ( ! a [ i ] && a [ i + 1 ] ) {
for ( int j = 1 ; j <= i ; j ++ ) cout << " ▁ " << j ; cout << " ▁ " << N + 1 ; for ( int j = i + 1 ; j <= N ; j ++ ) cout << " ▁ " << j ; return ; } }
for ( int i = 1 ; i <= N ; i ++ ) cout << " ▁ " << i ; cout << " ▁ " << N + 1 ; }
int main ( ) {
int N = 3 , arr [ ] = { 0 , 1 , 0 } ;
findpath ( N , arr ) ; }
#include <bits/stdc++.h> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
void printknapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } }
int res = K [ n ] [ W ] ; cout << res << endl ; w = W ; for ( i = n ; i > 0 && res > 0 ; i -- ) {
if ( res == K [ i - 1 ] [ w ] ) continue ; else {
cout << " ▁ " << wt [ i - 1 ] ;
res = res - val [ i - 1 ] ; w = w - wt [ i - 1 ] ; } } }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printknapSack ( W , wt , val , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
int main ( ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = sizeof ( keys ) / sizeof ( keys [ 0 ] ) ; cout << " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ " << optimalSearchTree ( keys , freq , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define INF  INT_MAX
int printSolution ( int p [ ] , int n ) ;
void solveWordWrap ( int l [ ] , int n , int M ) {
int extras [ n + 1 ] [ n + 1 ] ;
int lc [ n + 1 ] [ n + 1 ] ;
int c [ n + 1 ] ;
int p [ n + 1 ] ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( i = 1 ; i <= n ; i ++ ) { for ( j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = INF ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( j = 1 ; j <= n ; j ++ ) { c [ j ] = INF ; for ( i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != INF && lc [ i ] [ j ] != INF && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; } int printSolution ( int p [ ] , int n ) { int k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; cout << " Line ▁ number ▁ " << k << " : ▁ From ▁ word ▁ no . ▁ " << p [ n ] << " ▁ to ▁ " << n << endl ; return k ; }
int main ( ) { int l [ ] = { 3 , 2 , 2 , 5 } ; int n = sizeof ( l ) / sizeof ( l [ 0 ] ) ; int M = 6 ; solveWordWrap ( l , n , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int eggDrop ( int n , int k ) {
int eggFloor [ n + 1 ] [ k + 1 ] ; int res ; int i , j , x ;
for ( i = 1 ; i <= n ; i ++ ) { eggFloor [ i ] [ 1 ] = 1 ; eggFloor [ i ] [ 0 ] = 0 ; }
for ( j = 1 ; j <= k ; j ++ ) eggFloor [ 1 ] [ j ] = j ;
for ( i = 2 ; i <= n ; i ++ ) { for ( j = 2 ; j <= k ; j ++ ) { eggFloor [ i ] [ j ] = INT_MAX ; for ( x = 1 ; x <= j ; x ++ ) { res = 1 + max ( eggFloor [ i - 1 ] [ x - 1 ] , eggFloor [ i ] [ j - x ] ) ; if ( res < eggFloor [ i ] [ j ] ) eggFloor [ i ] [ j ] = res ; } } }
return eggFloor [ n ] [ k ] ; }
int main ( ) { int n = 2 , k = 36 ; cout << " Minimum number of trials " STRNEWLINE " in worst case with " << n << " ▁ eggs ▁ and ▁ " << k << " ▁ floors ▁ is ▁ " << eggDrop ( n , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; cout << knapSack ( W , wt , val , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
static int max_ref ;
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
int main ( ) { int arr [ ] = { 10 , 22 , 9 , 33 , 21 , 50 , 41 , 60 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Length ▁ of ▁ lis ▁ is ▁ " << lis ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define d  256 
void search ( char pat [ ] , char txt [ ] , int q ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i , j ;
int h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat [ i ] ) % q ; t = ( d * t + txt [ i ] ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) { bool flag = true ;
for ( j = 0 ; j < M ; j ++ ) { if ( txt [ i + j ] != pat [ j ] ) { flag = false ; break ; } if ( flag ) cout << i << " ▁ " ; }
if ( j == M ) cout << " Pattern ▁ found ▁ at ▁ index ▁ " << i << endl ; }
if ( i < N - M ) { t = ( d * ( t - txt [ i ] * h ) + txt [ i + M ] ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
int main ( ) { char txt [ ] = " GEEKS ▁ FOR ▁ GEEKS " ; char pat [ ] = " GEEK " ;
int q = 101 ;
search ( pat , txt , q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define N  8 NEW_LINE int solveKTUtil ( int x , int y , int movei , int sol [ N ] [ N ] , int xMove [ ] , int yMove [ ] ) ;
int isSafe ( int x , int y , int sol [ N ] [ N ] ) { return ( x >= 0 && x < N && y >= 0 && y < N && sol [ x ] [ y ] == -1 ) ; }
void printSolution ( int sol [ N ] [ N ] ) { for ( int x = 0 ; x < N ; x ++ ) { for ( int y = 0 ; y < N ; y ++ ) cout << " ▁ " << setw ( 2 ) << sol [ x ] [ y ] << " ▁ " ; cout << endl ; } }
int solveKT ( ) { int sol [ N ] [ N ] ;
for ( int x = 0 ; x < N ; x ++ ) for ( int y = 0 ; y < N ; y ++ ) sol [ x ] [ y ] = -1 ;
int xMove [ 8 ] = { 2 , 1 , -1 , -2 , -2 , -1 , 1 , 2 } ; int yMove [ 8 ] = { 1 , 2 , 2 , 1 , -1 , -2 , -2 , -1 } ;
sol [ 0 ] [ 0 ] = 0 ;
if ( solveKTUtil ( 0 , 0 , 1 , sol , xMove , yMove ) == 0 ) { cout << " Solution ▁ does ▁ not ▁ exist " ; return 0 ; } else printSolution ( sol ) ; return 1 ; }
int solveKTUtil ( int x , int y , int movei , int sol [ N ] [ N ] , int xMove [ N ] , int yMove [ N ] ) { int k , next_x , next_y ; if ( movei == N * N ) return 1 ;
for ( k = 0 ; k < 8 ; k ++ ) { next_x = x + xMove [ k ] ; next_y = y + yMove [ k ] ; if ( isSafe ( next_x , next_y , sol ) ) { sol [ next_x ] [ next_y ] = movei ; if ( solveKTUtil ( next_x , next_y , movei + 1 , sol , xMove , yMove ) == 1 ) return 1 ; else
sol [ next_x ] [ next_y ] = -1 ; } } return 0 ; }
int main ( ) {
solveKT ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define V  4 NEW_LINE void printSolution ( int color [ ] ) ;
void printSolution ( int color [ ] ) { cout << " Solution ▁ Exists : " " ▁ Following ▁ are ▁ the ▁ assigned ▁ colors ▁ STRNEWLINE " ; for ( int i = 0 ; i < V ; i ++ ) cout << " ▁ " << color [ i ] ; cout << " STRNEWLINE " ; }
bool isSafe ( bool graph [ V ] [ V ] , int color [ ] ) {
for ( int i = 0 ; i < V ; i ++ ) for ( int j = i + 1 ; j < V ; j ++ ) if ( graph [ i ] [ j ] && color [ j ] == color [ i ] ) return false ; return true ; }
bool graphColoring ( bool graph [ V ] [ V ] , int m , int i , int color [ V ] ) {
if ( i == V ) {
if ( isSafe ( graph , color ) ) {
printSolution ( color ) ; return true ; } return false ; }
for ( int j = 1 ; j <= m ; j ++ ) { color [ i ] = j ;
if ( graphColoring ( graph , m , i + 1 , color ) ) return true ; color [ i ] = 0 ; } return false ; }
int main ( ) {
bool graph [ V ] [ V ] = { { 0 , 1 , 1 , 1 } , { 1 , 0 , 1 , 0 } , { 1 , 1 , 0 , 1 } , { 1 , 0 , 1 , 0 } , } ;
int m = 3 ;
int color [ V ] ; for ( int i = 0 ; i < V ; i ++ ) color [ i ] = 0 ; if ( ! graphColoring ( graph , m , 0 , color ) ) cout << " Solution ▁ does ▁ not ▁ exist " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int prevPowerofK ( int n , int k ) { int p = ( int ) ( log ( n ) / log ( k ) ) ; return ( int ) pow ( k , p ) ; }
int nextPowerOfK ( int n , int k ) { return prevPowerofK ( n , k ) * k ; }
int main ( ) { int N = 7 ; int K = 2 ; cout << prevPowerofK ( N , K ) << " ▁ " ; cout << nextPowerOfK ( N , K ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int gcd ( int a , int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
int main ( ) { int a = 98 , b = 56 ; cout << " GCD ▁ of ▁ " << a << " ▁ and ▁ " << b << " ▁ is ▁ " << gcd ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int checkSemiprime ( int num ) { int cnt = 0 ; for ( int i = 2 ; cnt < 2 && i * i <= num ; ++ i ) while ( num % i == 0 ) num /= i , ++ cnt ;
if ( num > 1 ) ++ cnt ;
return cnt == 2 ; }
void semiprime ( int n ) { if ( checkSemiprime ( n ) ) cout << " True STRNEWLINE " ; else cout << " False STRNEWLINE " ; }
int main ( ) { int n = 6 ; semiprime ( n ) ; n = 8 ; semiprime ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
void printGivenLevel ( node * root , int level ) ; int height ( node * node ) ; node * newNode ( int data ) ;
void reverseLevelOrder ( node * root ) { int h = height ( root ) ; int i ; for ( i = h ; i >= 1 ; i -- ) printGivenLevel ( root , i ) ; }
void printGivenLevel ( node * root , int level ) { if ( root == NULL ) return ; if ( level == 1 ) cout << root -> data << " ▁ " ; else if ( level > 1 ) { printGivenLevel ( root -> left , level - 1 ) ; printGivenLevel ( root -> right , level - 1 ) ; } }
int height ( node * node ) { if ( node == NULL ) return 0 ; else {
int lheight = height ( node -> left ) ; int rheight = height ( node -> right ) ;
if ( lheight > rheight ) return ( lheight + 1 ) ; else return ( rheight + 1 ) ; } }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int main ( ) {
node * root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; cout << " Level ▁ Order ▁ traversal ▁ of ▁ binary ▁ tree ▁ is ▁ STRNEWLINE " ; reverseLevelOrder ( root ) ; return 0 ; }
#include " bits / stdc + + . h " NEW_LINE using namespace std ;
void printNSE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] > arr [ j ] ) { next = arr [ j ] ; break ; } } cout << arr [ i ] << " ▁ - - ▁ " << next << endl ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNSE ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE #include <unordered_map> NEW_LINE using namespace std ;
void print ( int * arr , int n ) { for ( int i = 0 ; i < n ; i ++ ) { cout << arr [ i ] << " ▁ " ; } cout << endl ; }
int digit_at ( int x , int d ) { return ( int ) ( x / pow ( 10 , d - 1 ) ) % 10 ; }
void MSD_sort ( int * arr , int lo , int hi , int d ) {
if ( hi <= lo ) { return ; } int count [ 10 + 2 ] = { 0 } ;
unordered_map < int , int > temp ;
for ( int i = lo ; i <= hi ; i ++ ) { int c = digit_at ( arr [ i ] , d ) ; count ++ ; }
for ( int r = 0 ; r < 10 + 1 ; r ++ ) count [ r + 1 ] += count [ r ] ;
for ( int i = lo ; i <= hi ; i ++ ) { int c = digit_at ( arr [ i ] , d ) ; temp [ count ++ ] = arr [ i ] ; }
for ( int i = lo ; i <= hi ; i ++ ) arr [ i ] = temp [ i - lo ] ;
for ( int r = 0 ; r < 10 ; r ++ ) MSD_sort ( arr , lo + count [ r ] , lo + count [ r + 1 ] - 1 , d - 1 ) ; }
int getMax ( int arr [ ] , int n ) { int mx = arr [ 0 ] ; for ( int i = 1 ; i < n ; i ++ ) if ( arr [ i ] > mx ) mx = arr [ i ] ; return mx ; }
void radixsort ( int * arr , int n ) {
int m = getMax ( arr , n ) ;
int d = floor ( log10 ( abs ( m ) ) ) + 1 ;
MSD_sort ( arr , 0 , n - 1 , d ) ; }
int main ( ) {
int arr [ ] = { 9330 , 9950 , 718 , 8977 , 6790 , 95 , 9807 , 741 , 8586 , 5710 } ;
int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Unsorted ▁ array ▁ : ▁ " ) ;
print ( arr , n ) ;
radixsort ( arr , n ) ; printf ( " Sorted ▁ array ▁ : ▁ " ) ;
print ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #define __INSERT_THRESH  5 NEW_LINE #define __swap ( x , y )  (t = *(x), *(x) = *(y), *(y) = t)
static void merge ( int * a , size_t an , size_t bn ) { int * b = a + an , * e = b + bn , * s , t ;
if ( an == 0 || bn == 0 || ! ( * b < * ( b - 1 ) ) ) return ;
if ( an < __INSERT_THRESH && an <= bn ) { for ( int * p = b , * v ; p > a ;
for ( v = p , s = p - 1 ; v < e && * v < * s ; s = v , v ++ ) __swap ( s , v ) ; return ; } if ( bn < __INSERT_THRESH ) { for ( int * p = b , * v ; p < e ;
for ( s = p , v = p - 1 ; s > a && * s < * v ; s = v , v -- ) __swap ( s , v ) ; return ; }
int * pa = a , * pb = b ; for ( s = a ; s < b && pb < e ; s ++ ) if ( * pb < * pa ) pb ++ ; else pa ++ ; pa += b - s ;
for ( int * la = pa , * fb = b ; la < b ; la ++ , fb ++ ) __swap ( la , fb ) ;
merge ( a , pa - a , pb - b ) ; merge ( b , pb - b , e - pb ) ;
#undef  __swap NEW_LINE #undef  __INSERT_THRESH
void merge_sort ( int * a , size_t n ) { size_t m = ( n + 1 ) / 2 ;
if ( m > 1 ) merge_sort ( a , m ) ; if ( n - m > 1 ) merge_sort ( a + m , n - m ) ;
merge ( a , m , n - m ) ; }
void print_array ( int a [ ] , size_t n ) { if ( n > 0 ) { cout << " ▁ " << a [ 0 ] ; for ( size_t i = 1 ; i < n ; i ++ ) cout << " ▁ " << a [ i ] ; } cout << " STRNEWLINE " ; }
int main ( ) { int a [ ] = { 3 , 16 , 5 , 14 , 8 , 10 , 7 , 15 , 1 , 13 , 4 , 9 , 12 , 11 , 6 , 2 } ; size_t n = sizeof ( a ) / sizeof ( a [ 0 ] ) ; merge_sort ( a , n ) ; print_array ( a , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define SIZE  100
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
len_str -= 1 ; cout << " Encoded ▁ string ▁ : ▁ " << encoded_string << endl ; cout << " Decoded ▁ string ▁ : ▁ " << base64Decoder ( encoded_string , len_str ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <stdio.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ; # define NO_OF_CHARS  256
void print ( char list [ ] [ 50 ] , char * word , int list_size ) {
int * map = new int [ ( sizeof ( int ) * NO_OF_CHARS ) ] ; int i , j , count , word_size ;
for ( i = 0 ; * ( word + i ) ; i ++ ) map [ * ( word + i ) ] = 1 ;
word_size = strlen ( word ) ;
for ( i = 0 ; i < list_size ; i ++ ) { for ( j = 0 , count = 0 ; * ( list [ i ] + j ) ; j ++ ) { if ( map [ * ( list [ i ] + j ) ] ) { count ++ ;
map [ * ( list [ i ] + j ) ] = 0 ; } } if ( count == word_size ) cout << list [ i ] << endl ;
for ( j = 0 ; * ( word + j ) ; j ++ ) map [ * ( word + j ) ] = 1 ; } }
int main ( ) { char str [ ] = " sun " ; char list [ ] [ 50 ] = { " geeksforgeeks " , " unsorted " , " sunday " , " just " , " sss " } ; print ( list , str , 5 ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #define NO_OF_CHARS  256
int * getCharCountArray ( char * str ) { int * count = ( int * ) calloc ( sizeof ( int ) , NO_OF_CHARS ) ; int i ; for ( i = 0 ; * ( str + i ) ; i ++ ) count [ * ( str + i ) ] ++ ; return count ; }
int firstNonRepeating ( char * str ) { int * count = getCharCountArray ( str ) ; int index = -1 , i ; for ( i = 0 ; * ( str + i ) ; i ++ ) { if ( count [ * ( str + i ) ] == 1 ) { index = i ; break ; } }
free ( count ) ; return index ; }
int main ( ) { char str [ ] = " geeksforgeeks " ; int index = firstNonRepeating ( str ) ; if ( index == -1 ) cout << " Either ▁ all ▁ characters ▁ are ▁ repeating ▁ or ▁ " " string ▁ is ▁ empty " ; else cout << " First ▁ non - repeating ▁ character ▁ is ▁ " << str [ index ] ; getchar ( ) ; return 0 ; }
#include <iostream> NEW_LINE #include <string.h> NEW_LINE using namespace std ; class gfg {
public : void divideString ( char str [ ] , int n ) { int str_size = strlen ( str ) ; int i ; int part_size ;
if ( str_size % n != 0 ) { cout << " Invalid ▁ Input : ▁ String ▁ size " ; cout << " ▁ is ▁ not ▁ divisible ▁ by ▁ n " ; return ; }
part_size = str_size / n ; for ( i = 0 ; i < str_size ; i ++ ) { if ( i % part_size == 0 ) cout << endl ; cout << str [ i ] ; } } } ;
char str [ ] = " a _ simple _ divide _ string _ quest " ;
g . divideString ( str , 4 ) ; return 0 ; }
#include <cmath> NEW_LINE #include <iostream> NEW_LINE using namespace std ;
void bestApproximate ( int x [ ] , int y [ ] , int n ) { float m , c , sum_x = 0 , sum_y = 0 , sum_xy = 0 , sum_x2 = 0 ; for ( int i = 0 ; i < n ; i ++ ) { sum_x += x [ i ] ; sum_y += y [ i ] ; sum_xy += x [ i ] * y [ i ] ; sum_x2 += pow ( x [ i ] , 2 ) ; } m = ( n * sum_xy - sum_x * sum_y ) / ( n * sum_x2 - pow ( sum_x , 2 ) ) ; c = ( sum_y - m * sum_x ) / n ; cout << " m ▁ = " << m ; cout << " c = " }
int main ( ) { int x [ ] = { 1 , 2 , 3 , 4 , 5 } ; int y [ ] = { 14 , 27 , 40 , 55 , 68 } ; int n = sizeof ( x ) / sizeof ( x [ 0 ] ) ; bestApproximate ( x , y , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMinInsertions ( char str [ ] , int l , int h ) {
if ( l > h ) return INT_MAX ; if ( l == h ) return 0 ; if ( l == h - 1 ) return ( str [ l ] == str [ h ] ) ? 0 : 1 ;
return ( str [ l ] == str [ h ] ) ? findMinInsertions ( str , l + 1 , h - 1 ) : ( min ( findMinInsertions ( str , l , h - 1 ) , findMinInsertions ( str , l + 1 , h ) ) + 1 ) ; }
int main ( ) { char str [ ] = " geeks " ; cout << findMinInsertions ( str , 0 , strlen ( str ) - 1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class node { public : int data ; node * left , * right ; } ;
node * newNode ( int data ) { node * temp = new node ( ) ; temp -> data = data ; temp -> left = temp -> right = NULL ; return temp ; }
void morrisTraversalPreorder ( node * root ) { while ( root ) {
if ( root -> left == NULL ) { cout << root -> data << " ▁ " ; root = root -> right ; } else {
node * current = root -> left ; while ( current -> right && current -> right != root ) current = current -> right ;
if ( current -> right == root ) { current -> right = NULL ; root = root -> right ; }
else { cout << root -> data << " ▁ " ; current -> right = root ; root = root -> left ; } } } }
void preorder ( node * root ) { if ( root ) { cout << root -> data << " ▁ " ; preorder ( root -> left ) ; preorder ( root -> right ) ; } }
int main ( ) { node * root = NULL ; root = newNode ( 1 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 5 ) ; root -> right -> left = newNode ( 6 ) ; root -> right -> right = newNode ( 7 ) ; root -> left -> left -> left = newNode ( 8 ) ; root -> left -> left -> right = newNode ( 9 ) ; root -> left -> right -> left = newNode ( 10 ) ; root -> left -> right -> right = newNode ( 11 ) ; morrisTraversalPreorder ( root ) ; cout << endl ; preorder ( root ) ; return 0 ; }
void push ( Node * * head_ref , int new_data ) {
Node * new_node = new Node ( ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void insertAfter ( Node * prev_node , int new_data ) {
if ( prev_node == NULL ) { cout << " the ▁ given ▁ previous ▁ node ▁ cannot ▁ be ▁ NULL " ; return ; }
Node * new_node = new Node ( ) ; new_node -> data = new_data ;
new_node -> next = prev_node -> next ;
prev_node -> next = new_node ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ; void push ( Node * * head_ref , int new_data ) {
Node * new_node = new Node ( ) ;
new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; } int detectLoop ( Node * list ) { Node * slow_p = list , * fast_p = list ; while ( slow_p && fast_p && fast_p -> next ) { slow_p = slow_p -> next ; fast_p = fast_p -> next -> next ; if ( slow_p == fast_p ) { return 1 ; } } return 0 ; }
int main ( ) {
Node * head = NULL ; push ( & head , 20 ) ; push ( & head , 4 ) ; push ( & head , 15 ) ; push ( & head , 10 ) ;
head -> next -> next -> next -> next = head ; if ( detectLoop ( head ) ) cout << " Loop ▁ found " ; else cout << " No ▁ Loop " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class Node { public : int data ; Node * next ; } ;
void swapNodes ( Node * * head_ref , int x , int y ) {
if ( x == y ) return ;
Node * prevX = NULL , * currX = * head_ref ; while ( currX && currX -> data != x ) { prevX = currX ; currX = currX -> next ; }
Node * prevY = NULL , * currY = * head_ref ; while ( currY && currY -> data != y ) { prevY = currY ; currY = currY -> next ; }
if ( currX == NULL currY == NULL ) return ;
if ( prevX != NULL ) prevX -> next = currY ;
else * head_ref = currY ;
if ( prevY != NULL ) prevY -> next = currX ;
else * head_ref = currX ;
Node * temp = currY -> next ; currY -> next = currX -> next ; currX -> next = temp ; }
void push ( Node * * head_ref , int new_data ) {
Node * new_node = new Node ( ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( Node * node ) { while ( node != NULL ) { cout << node -> data << " ▁ " ; node = node -> next ; } }
int main ( ) { Node * start = NULL ;
push ( & start , 7 ) ; push ( & start , 6 ) ; push ( & start , 5 ) ; push ( & start , 4 ) ; push ( & start , 3 ) ; push ( & start , 2 ) ; push ( & start , 1 ) ; cout << " Linked ▁ list ▁ before ▁ calling ▁ swapNodes ( ) ▁ " ; printList ( start ) ; swapNodes ( & start , 4 , 3 ) ; cout << " Linked list after calling swapNodes ( ) " ; printList ( start ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : Node * prev ; int info ; Node * next ; } ;
void nodeInsetail ( Node * * head , Node * * tail , int key ) { Node * p = new Node ( ) ; p -> info = key ; p -> next = NULL ;
if ( ( * head ) == NULL ) { ( * head ) = p ; ( * tail ) = p ; ( * head ) -> prev = NULL ; return ; }
if ( ( p -> info ) < ( ( * head ) -> info ) ) { p -> prev = NULL ; ( * head ) -> prev = p ; p -> next = ( * head ) ; ( * head ) = p ; return ; }
if ( ( p -> info ) > ( ( * tail ) -> info ) ) { p -> prev = ( * tail ) ; ( * tail ) -> next = p ; ( * tail ) = p ; return ; }
Node * temp = ( * head ) -> next ; while ( ( temp -> info ) < ( p -> info ) ) temp = temp -> next ;
( temp -> prev ) -> next = p ; p -> prev = temp -> prev ; temp -> prev = p ; p -> next = temp ; }
void printList ( Node * temp ) { while ( temp != NULL ) { cout << temp -> info << " ▁ " ; temp = temp -> next ; } }
int main ( ) { Node * left = NULL , * right = NULL ; nodeInsetail ( & left , & right , 30 ) ; nodeInsetail ( & left , & right , 50 ) ; nodeInsetail ( & left , & right , 90 ) ; nodeInsetail ( & left , & right , 10 ) ; nodeInsetail ( & left , & right , 40 ) ; nodeInsetail ( & left , & right , 110 ) ; nodeInsetail ( & left , & right , 60 ) ; nodeInsetail ( & left , & right , 95 ) ; nodeInsetail ( & left , & right , 23 ) ; cout << " Doubly ▁ linked ▁ list ▁ on ▁ printing " " ▁ from ▁ left ▁ to ▁ right STRNEWLINE " ; printList ( left ) ; return 0 ; }
struct Node { int data ; struct Node * next ; } ;
void fun1 ( struct Node * head ) { if ( head == NULL ) return ; fun1 ( head -> next ) ; cout << head -> data << " ▁ " ; }
void fun2 ( struct Node * head ) { if ( head == NULL ) return ; cout << head -> data << " ▁ " ; if ( head -> next != NULL ) fun2 ( head -> next -> next ) ; cout << head -> data << " ▁ " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ;
void fun1 ( Node * head ) { if ( head == NULL ) return ; fun1 ( head -> next ) ; cout << head -> data << " ▁ " ; }
void fun2 ( Node * start ) { if ( start == NULL ) return ; cout << start -> data << " ▁ " ; if ( start -> next != NULL ) fun2 ( start -> next -> next ) ; cout << start -> data << " ▁ " ; }
void push ( Node * * head_ref , int new_data ) {
Node * new_node = new Node ( ) ;
new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
int main ( ) {
Node * head = NULL ;
push ( & head , 5 ) ; push ( & head , 4 ) ; push ( & head , 3 ) ; push ( & head , 2 ) ; push ( & head , 1 ) ; cout << " Output ▁ of ▁ fun1 ( ) ▁ for ▁ list ▁ 1 - > 2 - > 3 - > 4 - > 5 ▁ STRNEWLINE " ; fun1 ( head ) ; cout << " Output of fun2 ( ) for list 1 -> 2 -> 3 -> 4 -> 5 " ; fun2 ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ;
int printsqrtn ( Node * head ) { Node * sqrtn = NULL ; int i = 1 , j = 1 ;
while ( head != NULL ) {
if ( i == j * j ) {
if ( sqrtn == NULL ) sqrtn = head ; else sqrtn = sqrtn -> next ;
j ++ ; } i ++ ; head = head -> next ; }
return sqrtn -> data ; } void print ( Node * head ) { while ( head != NULL ) { cout << head -> data << " ▁ " ; head = head -> next ; } cout << endl ; }
void push ( Node * * head_ref , int new_data ) {
Node * new_node = new Node ( ) ;
new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
int main ( ) {
Node * head = NULL ; push ( & head , 40 ) ; push ( & head , 30 ) ; push ( & head , 20 ) ; push ( & head , 10 ) ; cout << " Given ▁ linked ▁ list ▁ is : " ; print ( head ) ; cout << " sqrt ( n ) th ▁ node ▁ is ▁ " << printsqrtn ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct node { int data ; struct node * left ; struct node * right ; } ;
struct node * newNode ( int data ) { struct node * node = ( struct node * ) malloc ( sizeof ( struct node ) ) ; node -> data = data ; node -> left = NULL ; node -> right = NULL ; return ( node ) ; }
struct node * insert ( struct node * node , int data ) {
if ( node == NULL ) return ( newNode ( data ) ) ; else {
if ( data <= node -> data ) node -> left = insert ( node -> left , data ) ; else node -> right = insert ( node -> right , data ) ;
return node ; } }
int minValue ( struct node * node ) { struct node * current = node ;
while ( current -> left != NULL ) { current = current -> left ; } return ( current -> data ) ; }
int main ( ) { struct node * root = NULL ; root = insert ( root , 4 ) ; insert ( root , 2 ) ; insert ( root , 1 ) ; insert ( root , 3 ) ; insert ( root , 6 ) ; insert ( root , 5 ) ; cout << " Minimum value in BST is " getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : char data ; node * left ; node * right ; } ;
int search ( char arr [ ] , int strt , int end , char value ) ; node * newNode ( char data ) ;
node * buildTree ( char in [ ] , char pre [ ] , int inStrt , int inEnd ) { static int preIndex = 0 ; if ( inStrt > inEnd ) return NULL ;
node * tNode = newNode ( pre [ preIndex ++ ] ) ;
if ( inStrt == inEnd ) return tNode ;
int inIndex = search ( in , inStrt , inEnd , tNode -> data ) ;
tNode -> left = buildTree ( in , pre , inStrt , inIndex - 1 ) ; tNode -> right = buildTree ( in , pre , inIndex + 1 , inEnd ) ; return tNode ; }
int search ( char arr [ ] , int strt , int end , char value ) { int i ; for ( i = strt ; i <= end ; i ++ ) { if ( arr [ i ] == value ) return i ; } }
node * newNode ( char data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
void printInorder ( node * node ) { if ( node == NULL ) return ;
printInorder ( node -> left ) ;
cout << node -> data << " ▁ " ;
printInorder ( node -> right ) ; }
int main ( ) { char in [ ] = { ' D ' , ' B ' , ' E ' , ' A ' , ' F ' , ' C ' } ; char pre [ ] = { ' A ' , ' B ' , ' D ' , ' E ' , ' C ' , ' F ' } ; int len = sizeof ( in ) / sizeof ( in [ 0 ] ) ; node * root = buildTree ( in , pre , 0 , len - 1 ) ;
cout << " Inorder ▁ traversal ▁ of ▁ the ▁ constructed ▁ tree ▁ is ▁ STRNEWLINE " ; printInorder ( root ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class node { public : int data ; node * left , * right ; } ;
struct node * lca ( struct node * root , int n1 , int n2 ) { while ( root != NULL ) {
if ( root -> data > n1 && root -> data > n2 ) root = root -> left ;
else if ( root -> data < n1 && root -> data < n2 ) root = root -> right ; else break ; } return root ; }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = Node -> right = NULL ; return ( Node ) ; }
int main ( ) {
node * root = newNode ( 20 ) ; root -> left = newNode ( 8 ) ; root -> right = newNode ( 22 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 12 ) ; root -> left -> right -> left = newNode ( 10 ) ; root -> left -> right -> right = newNode ( 14 ) ; int n1 = 10 , n2 = 14 ; node * t = lca ( root , n1 , n2 ) ; cout << " LCA ▁ of ▁ " << n1 << " ▁ and ▁ " << n2 << " ▁ is ▁ " << t -> data << endl ; n1 = 14 , n2 = 8 ; t = lca ( root , n1 , n2 ) ; cout << " LCA ▁ of ▁ " << n1 << " ▁ and ▁ " << n2 << " ▁ is ▁ " << t -> data << endl ; n1 = 10 , n2 = 22 ; t = lca ( root , n1 , n2 ) ; cout << " LCA ▁ of ▁ " << n1 << " ▁ and ▁ " << n2 << " ▁ is ▁ " << t -> data << endl ; return 0 ; }
int isBST ( struct node * node ) { if ( node == NULL ) return 1 ;
if ( node -> left != NULL && node -> left -> data > node -> data ) return 0 ;
if ( node -> right != NULL && node -> right -> data < node -> data ) return 0 ;
if ( ! isBST ( node -> left ) || ! isBST ( node -> right ) ) return 0 ;
return 1 ; }
int isBST ( struct node * node ) { if ( node == NULL ) return 1 ;
if ( node -> left != NULL && maxValue ( node -> left ) > node -> data ) return 0 ;
if ( node -> right != NULL && minValue ( node -> right ) < node -> data ) return 0 ;
if ( ! isBST ( node -> left ) || ! isBST ( node -> right ) ) return 0 ;
return 1 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int key ; node * left ; node * right ; } ;
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
node * newNode ( int num ) { node * temp = new node ( ) ; temp -> key = num ; temp -> left = temp -> right = NULL ; return temp ; }
node * insert ( node * root , int key ) { if ( root == NULL ) return newNode ( key ) ; if ( root -> key > key ) root -> left = insert ( root -> left , key ) ; else root -> right = insert ( root -> right , key ) ; return root ; }
int main ( ) { node * root = NULL ; root = insert ( root , 6 ) ; root = insert ( root , -13 ) ; root = insert ( root , 14 ) ; root = insert ( root , -8 ) ; root = insert ( root , 15 ) ; root = insert ( root , 13 ) ; root = insert ( root , 7 ) ; if ( isTripletPresent ( root ) ) cout << " Present " ; else cout << " Not ▁ Present " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX_SIZE  100
class node { public : int val ; node * left , * right ; } ;
class Stack { public : int size ; int top ; node * * array ; } ;
Stack * createStack ( int size ) { Stack * stack = new Stack ( ) ; stack -> size = size ; stack -> top = -1 ; stack -> array = new node * [ ( stack -> size * sizeof ( node * ) ) ] ; return stack ; }
int isFull ( Stack * stack ) { return stack -> top - 1 == stack -> size ; } int isEmpty ( Stack * stack ) { return stack -> top == -1 ; } void push ( Stack * stack , node * node ) { if ( isFull ( stack ) ) return ; stack -> array [ ++ stack -> top ] = node ; } node * pop ( Stack * stack ) { if ( isEmpty ( stack ) ) return NULL ; return stack -> array [ stack -> top -- ] ; }
bool isPairPresent ( node * root , int target ) {
Stack * s1 = createStack ( MAX_SIZE ) ; Stack * s2 = createStack ( MAX_SIZE ) ;
bool done1 = false , done2 = false ; int val1 = 0 , val2 = 0 ; node * curr1 = root , * curr2 = root ;
while ( 1 ) {
while ( done1 == false ) { if ( curr1 != NULL ) { push ( s1 , curr1 ) ; curr1 = curr1 -> left ; } else { if ( isEmpty ( s1 ) ) done1 = 1 ; else { curr1 = pop ( s1 ) ; val1 = curr1 -> val ; curr1 = curr1 -> right ; done1 = 1 ; } } }
while ( done2 == false ) { if ( curr2 != NULL ) { push ( s2 , curr2 ) ; curr2 = curr2 -> right ; } else { if ( isEmpty ( s2 ) ) done2 = 1 ; else { curr2 = pop ( s2 ) ; val2 = curr2 -> val ; curr2 = curr2 -> left ; done2 = 1 ; } } }
if ( ( val1 != val2 ) && ( val1 + val2 ) == target ) { cout << " Pair ▁ Found : ▁ " << val1 << " + ▁ " << val2 << " ▁ = ▁ " << target << endl ; return true ; }
else if ( ( val1 + val2 ) < target ) done1 = false ;
else if ( ( val1 + val2 ) > target ) done2 = false ;
if ( val1 >= val2 ) return false ; } }
node * NewNode ( int val ) { node * tmp = new node ( ) ; tmp -> val = val ; tmp -> right = tmp -> left = NULL ; return tmp ; }
int main ( ) {
node * root = NewNode ( 15 ) ; root -> left = NewNode ( 10 ) ; root -> right = NewNode ( 20 ) ; root -> left -> left = NewNode ( 8 ) ; root -> left -> right = NewNode ( 12 ) ; root -> right -> left = NewNode ( 16 ) ; root -> right -> right = NewNode ( 25 ) ; int target = 33 ; if ( isPairPresent ( root , target ) == false ) cout << " No such values are found " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
node * newNode ( int data ) { node * temp = new node ( ) ; temp -> data = data ; temp -> left = temp -> right = NULL ; return temp ; }
node * constructTreeUtil ( int pre [ ] , int post [ ] , int * preIndex , int l , int h , int size ) {
if ( * preIndex >= size l > h ) return NULL ;
node * root = newNode ( pre [ * preIndex ] ) ; ++ * preIndex ;
if ( l == h ) return root ;
int i ; for ( i = l ; i <= h ; ++ i ) if ( pre [ * preIndex ] == post [ i ] ) break ;
if ( i <= h ) { root -> left = constructTreeUtil ( pre , post , preIndex , l , i , size ) ; root -> right = constructTreeUtil ( pre , post , preIndex , i + 1 , h , size ) ; } return root ; }
node * constructTree ( int pre [ ] , int post [ ] , int size ) { int preIndex = 0 ; return constructTreeUtil ( pre , post , & preIndex , 0 , size - 1 , size ) ; }
void printInorder ( node * node ) { if ( node == NULL ) return ; printInorder ( node -> left ) ; cout << node -> data << " ▁ " ; printInorder ( node -> right ) ; }
int main ( ) { int pre [ ] = { 1 , 2 , 4 , 8 , 9 , 5 , 3 , 6 , 7 } ; int post [ ] = { 8 , 9 , 4 , 5 , 2 , 6 , 7 , 3 , 1 } ; int size = sizeof ( pre ) / sizeof ( pre [ 0 ] ) ; node * root = constructTree ( pre , post , size ) ; cout << " Inorder ▁ traversal ▁ of ▁ the ▁ constructed ▁ tree : ▁ STRNEWLINE " ; printInorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printSorted ( int arr [ ] , int start , int end ) { if ( start > end ) return ;
printSorted ( arr , start * 2 + 1 , end ) ;
cout << arr [ start ] << " ▁ " ;
printSorted ( arr , start * 2 + 2 , end ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 5 , 1 , 3 } ; int arr_size = sizeof ( arr ) / sizeof ( int ) ; printSorted ( arr , 0 , arr_size - 1 ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int key ; node * left ; node * right ; } ;
node * newNode ( int key ) { node * Node = new node ( ) ; Node -> key = key ; Node -> left = NULL ; Node -> right = NULL ; return ( Node ) ; }
int Ceil ( node * root , int input ) {
if ( root == NULL ) return -1 ;
if ( root -> key == input ) return root -> key ;
if ( root -> key < input ) return Ceil ( root -> right , input ) ;
int ceil = Ceil ( root -> left , input ) ; return ( ceil >= input ) ? ceil : root -> key ; }
int main ( ) { node * root = newNode ( 8 ) ; root -> left = newNode ( 4 ) ; root -> right = newNode ( 12 ) ; root -> left -> left = newNode ( 2 ) ; root -> left -> right = newNode ( 6 ) ; root -> right -> left = newNode ( 10 ) ; root -> right -> right = newNode ( 14 ) ; for ( int i = 0 ; i < 16 ; i ++ ) cout << i << " ▁ " << Ceil ( root , i ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; struct node { int key ; int count ; struct node * left , * right ; } ;
struct node * newNode ( int item ) { struct node * temp = ( struct node * ) malloc ( sizeof ( struct node ) ) ; temp -> key = item ; temp -> left = temp -> right = NULL ; temp -> count = 1 ; return temp ; }
void inorder ( struct node * root ) { if ( root != NULL ) { inorder ( root -> left ) ; cout << root -> key << " ( " << root -> count << " ) ▁ " ; inorder ( root -> right ) ; } }
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
int main ( ) {
struct node * root = NULL ; root = insert ( root , 12 ) ; root = insert ( root , 10 ) ; root = insert ( root , 20 ) ; root = insert ( root , 9 ) ; root = insert ( root , 11 ) ; root = insert ( root , 10 ) ; root = insert ( root , 12 ) ; root = insert ( root , 12 ) ; cout << " Inorder ▁ traversal ▁ of ▁ the ▁ given ▁ tree ▁ " << endl ; inorder ( root ) ; cout << " Delete 20 " root = deleteNode ( root , 20 ) ; cout << " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ STRNEWLINE " ; inorder ( root ) ; cout << " Delete 12 " root = deleteNode ( root , 12 ) ; cout << " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ STRNEWLINE " ; inorder ( root ) ; cout << " Delete 9 " root = deleteNode ( root , 9 ) ; cout << " Inorder ▁ traversal ▁ of ▁ the ▁ modified ▁ tree ▁ STRNEWLINE " ; inorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class node { public : int key ; node * left , * right ; } ;
node * newNode ( int item ) { node * temp = new node ; temp -> key = item ; temp -> left = temp -> right = NULL ; return temp ; }
void inorder ( node * root ) { if ( root != NULL ) { inorder ( root -> left ) ; cout << root -> key << " ▁ " ; inorder ( root -> right ) ; } }
node * insert ( node * node , int key ) {
if ( node == NULL ) return newNode ( key ) ;
if ( key < node -> key ) node -> left = insert ( node -> left , key ) ; else node -> right = insert ( node -> right , key ) ;
return node ; }
node * minValueNode ( node * Node ) { node * current = Node ;
while ( current -> left != NULL ) current = current -> left ; return current ; }
node * deleteNode ( node * root , int key ) {
if ( root == NULL ) return root ;
if ( key < root -> key ) root -> left = deleteNode ( root -> left , key ) ;
else if ( key > root -> key ) root -> right = deleteNode ( root -> right , key ) ;
else {
if ( root -> left == NULL ) { node * temp = root -> right ; free ( root ) ; return temp ; } else if ( root -> right == NULL ) { node * temp = root -> left ; free ( root ) ; return temp ; }
node * temp = minValueNode ( root -> right ) ;
root -> key = temp -> key ;
root -> right = deleteNode ( root -> right , temp -> key ) ; } return root ; }
node * changeKey ( node * root , int oldVal , int newVal ) {
root = deleteNode ( root , oldVal ) ;
root = insert ( root , newVal ) ;
return root ; }
int main ( ) {
node * root = NULL ; root = insert ( root , 50 ) ; root = insert ( root , 30 ) ; root = insert ( root , 20 ) ; root = insert ( root , 40 ) ; root = insert ( root , 70 ) ; root = insert ( root , 60 ) ; root = insert ( root , 80 ) ; cout << " Inorder ▁ traversal ▁ of ▁ the ▁ given ▁ tree ▁ STRNEWLINE " ; inorder ( root ) ; root = changeKey ( root , 40 , 10 ) ;
cout << " Inorder traversal of the modified tree " ; inorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
int max ( int inorder [ ] , int strt , int end ) ; node * newNode ( int data ) ;
node * buildTree ( int inorder [ ] , int start , int end ) { if ( start > end ) return NULL ;
int i = max ( inorder , start , end ) ;
node * root = newNode ( inorder [ i ] ) ;
if ( start == end ) return root ;
root -> left = buildTree ( inorder , start , i - 1 ) ; root -> right = buildTree ( inorder , i + 1 , end ) ; return root ; }
int max ( int arr [ ] , int strt , int end ) { int i , max = arr [ strt ] , maxind = strt ; for ( i = strt + 1 ; i <= end ; i ++ ) { if ( arr [ i ] > max ) { max = arr [ i ] ; maxind = i ; } } return maxind ; }
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = NULL ; Node -> right = NULL ; return Node ; }
void printInorder ( node * node ) { if ( node == NULL ) return ;
printInorder ( node -> left ) ;
cout << node -> data << " ▁ " ;
printInorder ( node -> right ) ; }
int main ( ) {
int inorder [ ] = { 5 , 10 , 40 , 30 , 28 } ; int len = sizeof ( inorder ) / sizeof ( inorder [ 0 ] ) ; node * root = buildTree ( inorder , 0 , len - 1 ) ;
cout << " Inorder ▁ traversal ▁ of ▁ the ▁ constructed ▁ tree ▁ is ▁ STRNEWLINE " ; printInorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int Identity ( int num ) { int row , col ; for ( row = 0 ; row < num ; row ++ ) { for ( col = 0 ; col < num ; col ++ ) {
if ( row == col ) cout << 1 << " ▁ " ; else cout << 0 << " ▁ " ; } cout << endl ; } return 0 ; }
int main ( ) { int size = 5 ; Identity ( size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int search ( int mat [ 4 ] [ 4 ] , int n , int x ) { if ( n == 0 ) return -1 ; int smallest = mat [ 0 ] [ 0 ] , largest = mat [ n - 1 ] [ n - 1 ] ; if ( x < smallest x > largest ) return -1 ;
int i = 0 , j = n - 1 ; while ( i < n && j >= 0 ) { if ( mat [ i ] [ j ] == x ) { cout << " n ▁ Found ▁ at ▁ " << i << " , ▁ " << j ; return 1 ; } if ( mat [ i ] [ j ] > x ) j -- ;
else i ++ ; } cout << " n ▁ Element ▁ not ▁ found " ;
return 0 ; }
int main ( ) { int mat [ 4 ] [ 4 ] = { { 10 , 20 , 30 , 40 } , { 15 , 25 , 35 , 45 } , { 27 , 29 , 37 , 48 } , { 32 , 33 , 39 , 50 } } ; search ( mat , 4 , 29 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define SIZE  50
class node { public : int data ; node * right , * left ; } ;
class Queue { public : int front , rear ; int size ; node * * array ; } ;
node * newNode ( int data ) { node * temp = new node ( ) ; temp -> data = data ; temp -> left = temp -> right = NULL ; return temp ; }
Queue * createQueue ( int size ) { Queue * queue = new Queue ( ) ; queue -> front = queue -> rear = -1 ; queue -> size = size ; queue -> array = new node * [ queue -> size * sizeof ( node * ) ] ; int i ; for ( i = 0 ; i < size ; ++ i ) queue -> array [ i ] = NULL ; return queue ; }
int isEmpty ( Queue * queue ) { return queue -> front == -1 ; } int isFull ( Queue * queue ) { return queue -> rear == queue -> size - 1 ; } int hasOnlyOneItem ( Queue * queue ) { return queue -> front == queue -> rear ; } void Enqueue ( node * root , Queue * queue ) { if ( isFull ( queue ) ) return ; queue -> array [ ++ queue -> rear ] = root ; if ( isEmpty ( queue ) ) ++ queue -> front ; } node * Dequeue ( Queue * queue ) { if ( isEmpty ( queue ) ) return NULL ; node * temp = queue -> array [ queue -> front ] ; if ( hasOnlyOneItem ( queue ) ) queue -> front = queue -> rear = -1 ; else ++ queue -> front ; return temp ; } node * getFront ( Queue * queue ) { return queue -> array [ queue -> front ] ; }
int hasBothChild ( node * temp ) { return temp && temp -> left && temp -> right ; }
void insert ( node * * root , int data , Queue * queue ) {
node * temp = newNode ( data ) ;
if ( ! * root ) * root = temp ; else {
node * front = getFront ( queue ) ;
if ( ! front -> left ) front -> left = temp ;
else if ( ! front -> right ) front -> right = temp ;
if ( hasBothChild ( front ) ) Dequeue ( queue ) ; }
Enqueue ( temp , queue ) ; }
void levelOrder ( node * root ) { Queue * queue = createQueue ( SIZE ) ; Enqueue ( root , queue ) ; while ( ! isEmpty ( queue ) ) { node * temp = Dequeue ( queue ) ; cout << temp -> data << " ▁ " ; if ( temp -> left ) Enqueue ( temp -> left , queue ) ; if ( temp -> right ) Enqueue ( temp -> right , queue ) ; } }
int main ( ) { node * root = NULL ; Queue * queue = createQueue ( SIZE ) ; int i ; for ( i = 1 ; i <= 12 ; ++ i ) insert ( & root , i , queue ) ; levelOrder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
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
node * newNode ( int data ) { node * new_node = new node ( ) ; new_node -> data = data ; new_node -> left = new_node -> right = NULL ; return ( new_node ) ; }
void printList ( node * node ) { while ( node != NULL ) { cout << node -> data << " ▁ " ; node = node -> right ; } }
int main ( ) {
node * root = newNode ( 10 ) ; root -> left = newNode ( 12 ) ; root -> right = newNode ( 15 ) ; root -> left -> left = newNode ( 25 ) ; root -> left -> right = newNode ( 30 ) ; root -> right -> left = newNode ( 36 ) ;
node * head = bintree2list ( root ) ;
printList ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ROW  4 NEW_LINE #define COL  5
class Node { public : bool isEndOfCol ;
Node * child [ 2 ] ; } ;
Node * newNode ( ) { Node * temp = new Node ( ) ; temp -> isEndOfCol = 0 ; temp -> child [ 0 ] = temp -> child [ 1 ] = NULL ; return temp ; }
bool insert ( Node * * root , int ( * M ) [ COL ] , int row , int col ) {
if ( * root == NULL ) * root = newNode ( ) ;
if ( col < COL ) return insert ( & ( ( * root ) -> child [ M [ row ] [ col ] ] ) , M , row , col + 1 ) ;
else {
if ( ! ( ( * root ) -> isEndOfCol ) ) return ( * root ) -> isEndOfCol = 1 ;
return 0 ; } }
void printRow ( int ( * M ) [ COL ] , int row ) { int i ; for ( i = 0 ; i < COL ; ++ i ) cout << M [ row ] [ i ] << " ▁ " ; cout << endl ; }
void findUniqueRows ( int ( * M ) [ COL ] ) {
Node * root = NULL ; int i ;
for ( i = 0 ; i < ROW ; ++ i )
if ( insert ( & root , M , i , 0 ) )
printRow ( M , i ) ; }
int main ( ) { int M [ ROW ] [ COL ] = { { 0 , 1 , 0 , 0 , 1 } , { 1 , 0 , 1 , 1 , 0 } , { 0 , 1 , 0 , 0 , 1 } , { 1 , 0 , 1 , 0 , 0 } } ; findUniqueRows ( M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
int main ( ) { int mat [ M ] [ N ] = { { 1 , 2 , 3 , 4 , 5 } , { 2 , 4 , 5 , 8 , 10 } , { 3 , 5 , 7 , 9 , 11 } , { 1 , 3 , 5 , 7 , 9 } , } ; int result = findCommon ( mat ) ; if ( result == -1 ) cout << " No ▁ common ▁ element " ; else cout << " Common ▁ element ▁ is ▁ " << result ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left , * right ; } ;
node * newNode ( int data ) { node * Node = new node ( ) ; Node -> data = data ; Node -> left = Node -> right = NULL ; return ( Node ) ; }
void inorder ( node * root ) { if ( root != NULL ) { inorder ( root -> left ) ; cout << " TABSYMBOL " << root -> data ; inorder ( root -> right ) ; } }
void fixPrevPtr ( node * root ) { static node * pre = NULL ; if ( root != NULL ) { fixPrevPtr ( root -> left ) ; root -> left = pre ; pre = root ; fixPrevPtr ( root -> right ) ; } }
node * fixNextPtr ( node * root ) { node * prev = NULL ;
while ( root && root -> right != NULL ) root = root -> right ;
while ( root && root -> left != NULL ) { prev = root ; root = root -> left ; root -> right = prev ; }
return ( root ) ; }
node * BTToDLL ( node * root ) {
fixPrevPtr ( root ) ;
return fixNextPtr ( root ) ; }
void printList ( node * root ) { while ( root != NULL ) { cout << " TABSYMBOL " << root -> data ; root = root -> right ; } }
int main ( void ) {
node * root = newNode ( 10 ) ; root -> left = newNode ( 12 ) ; root -> right = newNode ( 15 ) ; root -> left -> left = newNode ( 25 ) ; root -> left -> right = newNode ( 30 ) ; root -> right -> left = newNode ( 36 ) ; cout << " Inorder Tree Traversal " ; inorder ( root ) ; node * head = BTToDLL ( root ) ; cout << " DLL Traversal " printList ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; class node { public : int data ; node * left ; node * right ;
node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; } } ;
void increment ( node * node , int diff ) ;
void convertTree ( node * node ) { int left_data = 0 , right_data = 0 , diff ;
if ( node == NULL || ( node -> left == NULL && node -> right == NULL ) ) return ; else {
convertTree ( node -> left ) ; convertTree ( node -> right ) ;
if ( node -> left != NULL ) left_data = node -> left -> data ;
if ( node -> right != NULL ) right_data = node -> right -> data ;
diff = left_data + right_data - node -> data ;
if ( diff > 0 ) node -> data = node -> data + diff ;
if ( diff < 0 )
increment ( node , - diff ) ; } }
void increment ( node * node , int diff ) {
if ( node -> left != NULL ) { node -> left -> data = node -> left -> data + diff ;
increment ( node -> left , diff ) ; }
else if ( node -> right != NULL ) { node -> right -> data = node -> right -> data + diff ;
increment ( node -> right , diff ) ; } }
void printInorder ( node * node ) { if ( node == NULL ) return ;
printInorder ( node -> left ) ;
cout << node -> data << " ▁ " ;
printInorder ( node -> right ) ; }
int main ( ) { node * root = new node ( 50 ) ; root -> left = new node ( 7 ) ; root -> right = new node ( 2 ) ; root -> left -> left = new node ( 3 ) ; root -> left -> right = new node ( 5 ) ; root -> right -> left = new node ( 1 ) ; root -> right -> right = new node ( 30 ) ; cout << " Inorder traversal before conversion : " printInorder ( root ) ; convertTree ( root ) ; cout << " Inorder traversal after conversion : " printInorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; } ;
int toSumTree ( node * Node ) {
if ( Node == NULL ) return 0 ;
int old_val = Node -> data ;
Node -> data = toSumTree ( Node -> left ) + toSumTree ( Node -> right ) ;
return Node -> data + old_val ; }
void printInorder ( node * Node ) { if ( Node == NULL ) return ; printInorder ( Node -> left ) ; cout << " ▁ " << Node -> data ; printInorder ( Node -> right ) ; }
node * newNode ( int data ) { node * temp = new node ; temp -> data = data ; temp -> left = NULL ; temp -> right = NULL ; return temp ; }
int main ( ) { node * root = NULL ; int x ;
root = newNode ( 10 ) ; root -> left = newNode ( -2 ) ; root -> right = newNode ( 6 ) ; root -> left -> left = newNode ( 8 ) ; root -> left -> right = newNode ( -4 ) ; root -> right -> left = newNode ( 7 ) ; root -> right -> right = newNode ( 5 ) ; toSumTree ( root ) ;
cout << " Inorder ▁ Traversal ▁ of ▁ the ▁ resultant ▁ tree ▁ is : ▁ STRNEWLINE " ; printInorder ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findPeakUtil ( int arr [ ] , int low , int high , int n ) {
int mid = low + ( high - low ) / 2 ;
if ( ( mid == 0 arr [ mid - 1 ] <= arr [ mid ] ) && ( mid == n - 1 arr [ mid + 1 ] <= arr [ mid ] ) ) return mid ;
else if ( mid > 0 && arr [ mid - 1 ] > arr [ mid ] ) return findPeakUtil ( arr , low , ( mid - 1 ) , n ) ;
else return findPeakUtil ( arr , ( mid + 1 ) , high , n ) ; }
int findPeak ( int arr [ ] , int n ) { return findPeakUtil ( arr , 0 , n - 1 , n ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 20 , 4 , 1 , 0 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Index ▁ of ▁ a ▁ peak ▁ point ▁ is ▁ " << findPeak ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printRepeating ( int arr [ ] , int size ) { int i , j ; printf ( " ▁ Repeating ▁ elements ▁ are ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) for ( j = i + 1 ; j < size ; j ++ ) if ( arr [ i ] == arr [ j ] ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printRepeating ( int arr [ ] , int size ) { int * count = new int [ sizeof ( int ) * ( size - 2 ) ] ; int i ; cout << " ▁ Repeating ▁ elements ▁ are ▁ " ; for ( i = 0 ; i < size ; i ++ ) { if ( count [ arr [ i ] ] == 1 ) cout << arr [ i ] << " ▁ " ; else count [ arr [ i ] ] ++ ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int fact ( int n ) ; void printRepeating ( int arr [ ] , int size ) {
int S = 0 ;
int P = 1 ;
int x , y ;
int D ; int n = size - 2 , i ;
for ( i = 0 ; i < size ; i ++ ) { S = S + arr [ i ] ; P = P * arr [ i ] ; }
S = S - n * ( n + 1 ) / 2 ;
P = P / fact ( n ) ;
D = sqrt ( S * S - 4 * P ) ; x = ( D + S ) / 2 ; y = ( S - D ) / 2 ; cout << " The ▁ two ▁ Repeating ▁ elements ▁ are ▁ " << x << " ▁ & ▁ " << y ; }
int fact ( int n ) { return ( n == 0 ) ? 1 : n * fact ( n - 1 ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printRepeating ( int arr [ ] , int size ) {
int Xor = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) Xor ^= arr [ i ] ; for ( i = 1 ; i <= n ; i ++ ) Xor ^= i ;
set_bit_no = Xor & ~ ( Xor - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ;
} for ( i = 1 ; i <= n ; i ++ ) { if ( i & set_bit_no ) x = x ^ i ;
else y = y ^ i ;
} cout << " The ▁ two ▁ repeating ▁ elements ▁ are ▁ " << y << " ▁ " << x ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printRepeating ( int arr [ ] , int size ) { int i ; cout << " The ▁ repeating ▁ elements ▁ are " ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) ] > 0 ) arr [ abs ( arr [ i ] ) ] = - arr [ abs ( arr [ i ] ) ] ; else cout << " ▁ " << abs ( arr [ i ] ) << " ▁ " ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int subArraySum ( int arr [ ] , int n , int sum ) { int curr_sum , i , j ;
for ( i = 0 ; i < n ; i ++ ) { curr_sum = arr [ i ] ;
for ( j = i + 1 ; j <= n ; j ++ ) { if ( curr_sum == sum ) { cout << " Sum ▁ found ▁ between ▁ indexes ▁ " << i << " ▁ and ▁ " << j - 1 ; return 1 ; } if ( curr_sum > sum j == n ) break ; curr_sum = curr_sum + arr [ j ] ; } } cout << " No ▁ subarray ▁ found " ; return 0 ; }
int main ( ) { int arr [ ] = { 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int sum = 23 ; subArraySum ( arr , n , sum ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int subArraySum ( int arr [ ] , int n , int sum ) {
int curr_sum = arr [ 0 ] , start = 0 , i ;
for ( i = 1 ; i <= n ; i ++ ) {
while ( curr_sum > sum && start < i - 1 ) { curr_sum = curr_sum - arr [ start ] ; start ++ ; }
if ( curr_sum == sum ) { cout << " Sum ▁ found ▁ between ▁ indexes ▁ " << start << " ▁ and ▁ " << i - 1 ; return 1 ; }
if ( i < n ) curr_sum = curr_sum + arr [ i ] ; }
cout << " No ▁ subarray ▁ found " ; return 0 ; }
int main ( ) { int arr [ ] = { 15 , 2 , 4 , 8 , 9 , 5 , 10 , 23 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int sum = 23 ; subArraySum ( arr , n , sum ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool find3Numbers ( int A [ ] , int arr_size , int sum ) { int l , r ;
for ( int i = 0 ; i < arr_size - 2 ; i ++ ) {
for ( int j = i + 1 ; j < arr_size - 1 ; j ++ ) {
for ( int k = j + 1 ; k < arr_size ; k ++ ) { if ( A [ i ] + A [ j ] + A [ k ] == sum ) { cout << " Triplet ▁ is ▁ " << A [ i ] << " , ▁ " << A [ j ] << " , ▁ " << A [ k ] ; return true ; } } } }
return false ; }
int main ( ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int sum = 22 ; int arr_size = sizeof ( A ) / sizeof ( A [ 0 ] ) ; find3Numbers ( A , arr_size , sum ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) { if ( arr [ i ] == x ) return i ; } return -1 ; }
int main ( ) { int arr [ ] = { 1 , 10 , 30 , 15 } ; int x = 30 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << x << " ▁ is ▁ present ▁ at ▁ index ▁ " << search ( arr , n , x ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int binarySearch ( int arr [ ] , int l , int r , int x ) { if ( r >= l ) { int mid = l + ( r - l ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? cout << " Element ▁ is ▁ not ▁ present ▁ in ▁ array " : cout << " Element ▁ is ▁ present ▁ at ▁ index ▁ " << result ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int binarySearch ( int arr [ ] , int l , int r , int x ) { while ( l <= r ) { int m = l + ( r - l ) / 2 ;
if ( arr [ m ] == x ) return m ;
if ( arr [ m ] < x ) l = m + 1 ;
else r = m - 1 ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? cout << " Element ▁ is ▁ not ▁ present ▁ in ▁ array " : cout << " Element ▁ is ▁ present ▁ at ▁ index ▁ " << result ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int interpolationSearch ( int arr [ ] , int lo , int hi , int x ) { int pos ;
if ( lo <= hi && x >= arr [ lo ] && x <= arr [ hi ] ) {
pos = lo + ( ( ( double ) ( hi - lo ) / ( arr [ hi ] - arr [ lo ] ) ) * ( x - arr [ lo ] ) ) ;
if ( arr [ pos ] == x ) return pos ;
if ( arr [ pos ] < x ) return interpolationSearch ( arr , pos + 1 , hi , x ) ;
if ( arr [ pos ] > x ) return interpolationSearch ( arr , lo , pos - 1 , x ) ; } return -1 ; }
int main ( ) {
int arr [ ] = { 10 , 12 , 13 , 16 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 33 , 35 , 42 , 47 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int x = 18 ; int index = interpolationSearch ( arr , 0 , n - 1 , x ) ;
if ( index != -1 ) cout << " Element ▁ found ▁ at ▁ index ▁ " << index ; else cout << " Element ▁ not ▁ found . " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void merge ( int arr [ ] , int l , int m , int r ) {
int n1 = m - l + 1 ; int n2 = r - m ;
int L [ n1 ] , R [ n2 ] ;
for ( int i = 0 ; i < n1 ; i ++ ) L [ i ] = arr [ l + i ] ; for ( int j = 0 ; j < n2 ; j ++ ) R [ j ] = arr [ m + 1 + j ] ;
int i = 0 ; int j = 0 ;
int k = l ; while ( i < n1 && j < n2 ) { if ( L [ i ] <= R [ j ] ) { arr [ k ] = L [ i ] ; i ++ ; } else { arr [ k ] = R [ j ] ; j ++ ; } k ++ ; }
while ( i < n1 ) { arr [ k ] = L [ i ] ; i ++ ; k ++ ; }
while ( j < n2 ) { arr [ k ] = R [ j ] ; j ++ ; k ++ ; } }
void mergeSort ( int arr [ ] , int l , int r ) { if ( l >= r ) { return ; }
int m = l + ( r - l ) / 2 ;
mergeSort ( arr , l , m ) ; mergeSort ( arr , m + 1 , r ) ;
merge ( arr , l , m , r ) ; }
void printArray ( int A [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) cout << A [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 , 7 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Given ▁ array ▁ is ▁ STRNEWLINE " ; printArray ( arr , arr_size ) ; mergeSort ( arr , 0 , arr_size - 1 ) ; cout << " Sorted array is " ; printArray ( arr , arr_size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
void printArr ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; ++ i ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 4 , 3 , 5 , 2 , 1 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( * arr ) ;
quickSortIterative ( arr , 0 , n - 1 ) ; printArr ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printMaxActivities ( int s [ ] , int f [ ] , int n ) { int i , j ; cout << " Following ▁ activities ▁ are ▁ selected ▁ " << endl ;
i = 0 ; cout << " ▁ " << i ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { cout << " ▁ " << j ; i = j ; } } }
int main ( ) { int s [ ] = { 1 , 3 , 0 , 5 , 8 , 5 } ; int f [ ] = { 2 , 4 , 6 , 7 , 9 , 9 } ; int n = sizeof ( s ) / sizeof ( s [ 0 ] ) ; printMaxActivities ( s , f , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define MAX_TREE_HT  100
class QueueNode { public : char data ; unsigned freq ; QueueNode * left , * right ; } ;
class Queue { public : int front , rear ; int capacity ; QueueNode * * array ; } ;
QueueNode * newNode ( char data , unsigned freq ) { QueueNode * temp = new QueueNode [ ( sizeof ( QueueNode ) ) ] ; temp -> left = temp -> right = NULL ; temp -> data = data ; temp -> freq = freq ; return temp ; }
Queue * createQueue ( int capacity ) { Queue * queue = new Queue [ ( sizeof ( Queue ) ) ] ; queue -> front = queue -> rear = -1 ; queue -> capacity = capacity ; queue -> array = new QueueNode * [ ( queue -> capacity * sizeof ( QueueNode * ) ) ] ; return queue ; }
int isSizeOne ( Queue * queue ) { return queue -> front == queue -> rear && queue -> front != -1 ; }
int isEmpty ( Queue * queue ) { return queue -> front == -1 ; }
int isFull ( Queue * queue ) { return queue -> rear == queue -> capacity - 1 ; }
void enQueue ( Queue * queue , QueueNode * item ) { if ( isFull ( queue ) ) return ; queue -> array [ ++ queue -> rear ] = item ; if ( queue -> front == -1 ) ++ queue -> front ; }
QueueNode * deQueue ( Queue * queue ) { if ( isEmpty ( queue ) ) return NULL ; QueueNode * temp = queue -> array [ queue -> front ] ; if ( queue -> front == queue
- > rear ) queue -> front = queue -> rear = -1 ; else ++ queue -> front ; return temp ; }
QueueNode * getFront ( Queue * queue ) { if ( isEmpty ( queue ) ) return NULL ; return queue -> array [ queue -> front ] ; }
QueueNode * findMin ( Queue * firstQueue , Queue * secondQueue ) {
if ( isEmpty ( firstQueue ) ) return deQueue ( secondQueue ) ;
if ( isEmpty ( secondQueue ) ) return deQueue ( firstQueue ) ;
if ( getFront ( firstQueue ) -> freq < getFront ( secondQueue ) -> freq ) return deQueue ( firstQueue ) ; return deQueue ( secondQueue ) ; }
int isLeaf ( QueueNode * root ) { return ! ( root -> left ) && ! ( root -> right ) ; }
void printArr ( int arr [ ] , int n ) { int i ; for ( i = 0 ; i < n ; ++ i ) cout << arr [ i ] ; cout << endl ; }
QueueNode * buildHuffmanTree ( char data [ ] , int freq [ ] , int size ) { QueueNode * left , * right , * top ;
Queue * firstQueue = createQueue ( size ) ; Queue * secondQueue = createQueue ( size ) ;
for ( int i = 0 ; i < size ; ++ i ) enQueue ( firstQueue , newNode ( data [ i ] , freq [ i ] ) ) ;
while ( ! ( isEmpty ( firstQueue ) && isSizeOne ( secondQueue ) ) ) {
left = findMin ( firstQueue , secondQueue ) ; right = findMin ( firstQueue , secondQueue ) ;
top = newNode ( ' $ ' , left -> freq + right -> freq ) ; top -> left = left ; top -> right = right ; enQueue ( secondQueue , top ) ; } return deQueue ( secondQueue ) ; }
void printCodes ( QueueNode * root , int arr [ ] , int top ) {
if ( root -> left ) { arr [ top ] = 0 ; printCodes ( root -> left , arr , top + 1 ) ; }
if ( root -> right ) { arr [ top ] = 1 ; printCodes ( root -> right , arr , top + 1 ) ; }
if ( isLeaf ( root ) ) { cout << root -> data << " : ▁ " ; printArr ( arr , top ) ; } }
void HuffmanCodes ( char data [ ] , int freq [ ] , int size ) {
QueueNode * root = buildHuffmanTree ( data , freq , size ) ;
int arr [ MAX_TREE_HT ] , top = 0 ; printCodes ( root , arr , top ) ; }
int main ( ) { char arr [ ] = { ' a ' , ' b ' , ' c ' , ' d ' , ' e ' , ' f ' } ; int freq [ ] = { 5 , 9 , 12 , 13 , 16 , 45 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; HuffmanCodes ( arr , freq , size ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int max ( int a , int b ) ;
int lcs ( char * X , char * Y , int m , int n ) { if ( m == 0 n == 0 ) return 0 ; if ( X [ m - 1 ] == Y [ n - 1 ] ) return 1 + lcs ( X , Y , m - 1 , n - 1 ) ; else return max ( lcs ( X , Y , m , n - 1 ) , lcs ( X , Y , m - 1 , n ) ) ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int main ( ) { char X [ ] = " AGGTAB " ; char Y [ ] = " GXTXAYB " ; int m = strlen ( X ) ; int n = strlen ( Y ) ; cout << " Length ▁ of ▁ LCS ▁ is ▁ " << lcs ( X , Y , m , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int max ( int a , int b ) ;
int lcs ( char * X , char * Y , int m , int n ) { int L [ m + 1 ] [ n + 1 ] ; int i , j ;
for ( i = 0 ; i <= m ; i ++ ) { for ( j = 0 ; j <= n ; j ++ ) { if ( i == 0 j == 0 ) L [ i ] [ j ] = 0 ; else if ( X [ i - 1 ] == Y [ j - 1 ] ) L [ i ] [ j ] = L [ i - 1 ] [ j - 1 ] + 1 ; else L [ i ] [ j ] = max ( L [ i - 1 ] [ j ] , L [ i ] [ j - 1 ] ) ; } }
return L [ m ] [ n ] ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int main ( ) { char X [ ] = " AGGTAB " ; char Y [ ] = " GXTXAYB " ; int m = strlen ( X ) ; int n = strlen ( Y ) ; cout << " Length ▁ of ▁ LCS ▁ is ▁ " << lcs ( X , Y , m , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ;
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int minCost ( int cost [ R ] [ C ] , int m , int n ) { if ( n < 0 m < 0 ) return INT_MAX ; else if ( m == 0 && n == 0 ) return cost [ m ] [ n ] ; else return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; cout << minCost ( cost , 2 , 2 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE using namespace std ; int min ( int x , int y , int z ) ; int minCost ( int cost [ R ] [ C ] , int m , int n ) { int i , j ;
int tc [ R ] [ C ] ; tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] ;
for ( i = 1 ; i <= m ; i ++ ) tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] ;
for ( j = 1 ; j <= n ; j ++ ) tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] ;
for ( i = 1 ; i <= m ; i ++ ) for ( j = 1 ; j <= n ; j ++ ) tc [ i ] [ j ] = min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] ; return tc [ m ] [ n ] ; }
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; cout << " ▁ " << minCost ( cost , 2 , 2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) {
if ( n == 0 W == 0 ) return 0 ;
if ( wt [ n - 1 ] > W ) return knapSack ( W , wt , val , n - 1 ) ;
else return max ( val [ n - 1 ] + knapSack ( W - wt [ n - 1 ] , wt , val , n - 1 ) , knapSack ( W , wt , val , n - 1 ) ) ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; cout << knapSack ( W , wt , val , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; cout << knapSack ( W , wt , val , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int eggDrop ( int n , int k ) {
if ( k == 1 k == 0 ) return k ;
if ( n == 1 ) return k ; int min = INT_MAX , x , res ;
for ( x = 1 ; x <= k ; x ++ ) { res = max ( eggDrop ( n - 1 , x - 1 ) , eggDrop ( n , k - x ) ) ; if ( res < min ) min = res ; } return min + 1 ; }
int main ( ) { int n = 2 , k = 10 ; cout << " Minimum ▁ number ▁ of ▁ trials ▁ " " in ▁ worst ▁ case ▁ with ▁ " << n << " ▁ eggs ▁ and ▁ " << k << " ▁ floors ▁ is ▁ " << eggDrop ( n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; cout << " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ " << lps ( seq , 0 , n - 1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define INF  INT_MAX
int printSolution ( int p [ ] , int n ) ; int printSolution ( int p [ ] , int n ) { int k ; if ( p [ n ] == 1 ) k = 1 ; else k = printSolution ( p , p [ n ] - 1 ) + 1 ; cout << " Line ▁ number ▁ " << k << " : ▁ From ▁ word ▁ no . ▁ " << p [ n ] << " ▁ to ▁ " << n << endl ; return k ; }
void solveWordWrap ( int l [ ] , int n , int M ) {
int extras [ n + 1 ] [ n + 1 ] ;
int lc [ n + 1 ] [ n + 1 ] ;
int c [ n + 1 ] ;
int p [ n + 1 ] ; int i , j ;
for ( i = 1 ; i <= n ; i ++ ) { extras [ i ] [ i ] = M - l [ i - 1 ] ; for ( j = i + 1 ; j <= n ; j ++ ) extras [ i ] [ j ] = extras [ i ] [ j - 1 ] - l [ j - 1 ] - 1 ; }
for ( i = 1 ; i <= n ; i ++ ) { for ( j = i ; j <= n ; j ++ ) { if ( extras [ i ] [ j ] < 0 ) lc [ i ] [ j ] = INF ; else if ( j == n && extras [ i ] [ j ] >= 0 ) lc [ i ] [ j ] = 0 ; else lc [ i ] [ j ] = extras [ i ] [ j ] * extras [ i ] [ j ] ; } }
c [ 0 ] = 0 ; for ( j = 1 ; j <= n ; j ++ ) { c [ j ] = INF ; for ( i = 1 ; i <= j ; i ++ ) { if ( c [ i - 1 ] != INF && lc [ i ] [ j ] != INF && ( c [ i - 1 ] + lc [ i ] [ j ] < c [ j ] ) ) { c [ j ] = c [ i - 1 ] + lc [ i ] [ j ] ; p [ j ] = i ; } } } printSolution ( p , n ) ; }
int main ( ) { int l [ ] = { 3 , 2 , 2 , 5 } ; int n = sizeof ( l ) / sizeof ( l [ 0 ] ) ; int M = 6 ; solveWordWrap ( l , n , M ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
int main ( ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = sizeof ( keys ) / sizeof ( keys [ 0 ] ) ; cout << " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ " << optimalSearchTree ( keys , freq , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int sum ( int freq [ ] , int i , int j ) { int s = 0 ; for ( int k = i ; k <= j ; k ++ ) s += freq [ k ] ; return s ; }
int optimalSearchTree ( int keys [ ] , int freq [ ] , int n ) {
int cost [ n ] [ n ] ;
for ( int i = 0 ; i < n ; i ++ ) cost [ i ] [ i ] = freq [ i ] ;
for ( int L = 2 ; L <= n ; L ++ ) {
for ( int i = 0 ; i <= n - L + 1 ; i ++ ) {
int j = i + L - 1 ; cost [ i ] [ j ] = INT_MAX ;
for ( int r = i ; r <= j ; r ++ ) {
int c = ( ( r > i ) ? cost [ i ] [ r - 1 ] : 0 ) + ( ( r < j ) ? cost [ r + 1 ] [ j ] : 0 ) + sum ( freq , i , j ) ; if ( c < cost [ i ] [ j ] ) cost [ i ] [ j ] = c ; } } } return cost [ 0 ] [ n - 1 ] ; }
int main ( ) { int keys [ ] = { 10 , 12 , 20 } ; int freq [ ] = { 34 , 8 , 50 } ; int n = sizeof ( keys ) / sizeof ( keys [ 0 ] ) ; cout << " Cost ▁ of ▁ Optimal ▁ BST ▁ is ▁ " << optimalSearchTree ( keys , freq , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
class node { public : int data ; node * left , * right ; } ;
int LISS ( node * root ) { if ( root == NULL ) return 0 ;
int size_excl = LISS ( root -> left ) + LISS ( root -> right ) ;
int size_incl = 1 ; if ( root -> left ) size_incl += LISS ( root -> left -> left ) + LISS ( root -> left -> right ) ; if ( root -> right ) size_incl += LISS ( root -> right -> left ) + LISS ( root -> right -> right ) ;
return max ( size_incl , size_excl ) ; }
node * newNode ( int data ) { node * temp = new node ( ) ; temp -> data = data ; temp -> left = temp -> right = NULL ; return temp ; }
int main ( ) {
node * root = newNode ( 20 ) ; root -> left = newNode ( 8 ) ; root -> left -> left = newNode ( 4 ) ; root -> left -> right = newNode ( 12 ) ; root -> left -> right -> left = newNode ( 10 ) ; root -> left -> right -> right = newNode ( 14 ) ; root -> right = newNode ( 22 ) ; root -> right -> right = newNode ( 25 ) ; cout << " Size ▁ of ▁ the ▁ Largest " << " ▁ Independent ▁ Set ▁ is ▁ " << LISS ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getCount ( char keypad [ ] [ 3 ] , int n ) { if ( keypad == NULL n <= 0 ) return 0 ; if ( n == 1 ) return 10 ;
int odd [ 10 ] , even [ 10 ] ; int i = 0 , j = 0 , useOdd = 0 , totalCount = 0 ; for ( i = 0 ; i <= 9 ; i ++ )
odd [ i ] = 1 ;
for ( j = 2 ; j <= n ; j ++ ) { useOdd = 1 - useOdd ;
if ( useOdd == 1 ) { even [ 0 ] = odd [ 0 ] + odd [ 8 ] ; even [ 1 ] = odd [ 1 ] + odd [ 2 ] + odd [ 4 ] ; even [ 2 ] = odd [ 2 ] + odd [ 1 ] + odd [ 3 ] + odd [ 5 ] ; even [ 3 ] = odd [ 3 ] + odd [ 2 ] + odd [ 6 ] ; even [ 4 ] = odd [ 4 ] + odd [ 1 ] + odd [ 5 ] + odd [ 7 ] ; even [ 5 ] = odd [ 5 ] + odd [ 2 ] + odd [ 4 ] + odd [ 8 ] + odd [ 6 ] ; even [ 6 ] = odd [ 6 ] + odd [ 3 ] + odd [ 5 ] + odd [ 9 ] ; even [ 7 ] = odd [ 7 ] + odd [ 4 ] + odd [ 8 ] ; even [ 8 ] = odd [ 8 ] + odd [ 0 ] + odd [ 5 ] + odd [ 7 ] + odd [ 9 ] ; even [ 9 ] = odd [ 9 ] + odd [ 6 ] + odd [ 8 ] ; } else { odd [ 0 ] = even [ 0 ] + even [ 8 ] ; odd [ 1 ] = even [ 1 ] + even [ 2 ] + even [ 4 ] ; odd [ 2 ] = even [ 2 ] + even [ 1 ] + even [ 3 ] + even [ 5 ] ; odd [ 3 ] = even [ 3 ] + even [ 2 ] + even [ 6 ] ; odd [ 4 ] = even [ 4 ] + even [ 1 ] + even [ 5 ] + even [ 7 ] ; odd [ 5 ] = even [ 5 ] + even [ 2 ] + even [ 4 ] + even [ 8 ] + even [ 6 ] ; odd [ 6 ] = even [ 6 ] + even [ 3 ] + even [ 5 ] + even [ 9 ] ; odd [ 7 ] = even [ 7 ] + even [ 4 ] + even [ 8 ] ; odd [ 8 ] = even [ 8 ] + even [ 0 ] + even [ 5 ] + even [ 7 ] + even [ 9 ] ; odd [ 9 ] = even [ 9 ] + even [ 6 ] + even [ 8 ] ; } }
totalCount = 0 ; if ( useOdd == 1 ) { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += even [ i ] ; } else { for ( i = 0 ; i <= 9 ; i ++ ) totalCount += odd [ i ] ; } return totalCount ; }
int main ( ) { char keypad [ 4 ] [ 3 ] = { { '1' , '2' , '3' } , { '4' , '5' , '6' } , { '7' , '8' , '9' } , { ' * ' , '0' , ' # ' } } ; cout << " Count ▁ for ▁ numbers ▁ of ▁ length ▁ 1 : ▁ " << getCount ( keypad , 1 ) << endl ; cout << " Count ▁ for ▁ numbers ▁ of ▁ length ▁ 2 : ▁ " << getCount ( keypad , 2 ) << endl ; cout << " Count ▁ for ▁ numbers ▁ of ▁ length ▁ 3 : ▁ " << getCount ( keypad , 3 ) << endl ; cout << " Count ▁ for ▁ numbers ▁ of ▁ length ▁ 4 : ▁ " << getCount ( keypad , 4 ) << endl ; cout << " Count ▁ for ▁ numbers ▁ of ▁ length ▁ 5 : ▁ " << getCount ( keypad , 5 ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int count ( int n ) {
int table [ n + 1 ] , i ;
for ( int j = 0 ; j < n + 1 ; j ++ ) table [ j ] = 0 ;
table [ 0 ] = 1 ;
for ( i = 3 ; i <= n ; i ++ ) table [ i ] += table [ i - 3 ] ; for ( i = 5 ; i <= n ; i ++ ) table [ i ] += table [ i - 5 ] ; for ( i = 10 ; i <= n ; i ++ ) table [ i ] += table [ i - 10 ] ; return table [ n ] ; }
int main ( void ) { int n = 20 ; cout << " Count ▁ for ▁ " << n << " ▁ is ▁ " << count ( n ) << endl ; n = 13 ; cout << " Count ▁ for ▁ " << n << " ▁ is ▁ " << count ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void search ( char * pat , char * txt ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ;
for ( int i = 0 ; i <= N - M ; i ++ ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) cout << " Pattern ▁ found ▁ at ▁ index ▁ " << i << endl ; } }
int main ( ) { char txt [ ] = " AABAACAADAABAAABAA " ; char pat [ ] = " AABA " ; search ( pat , txt ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
#define d  256
void search ( char pat [ ] , char txt [ ] , int q ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i , j ;
int p = 0 ;
int t = 0 ; int h = 1 ;
for ( i = 0 ; i < M - 1 ; i ++ ) h = ( h * d ) % q ;
for ( i = 0 ; i < M ; i ++ ) { p = ( d * p + pat [ i ] ) % q ; t = ( d * t + txt [ i ] ) % q ; }
for ( i = 0 ; i <= N - M ; i ++ ) {
if ( p == t ) {
for ( j = 0 ; j < M ; j ++ ) { if ( txt [ i + j ] != pat [ j ] ) break ; }
if ( j == M ) cout << " Pattern ▁ found ▁ at ▁ index ▁ " << i << endl ; }
if ( i < N - M ) { t = ( d * ( t - txt [ i ] * h ) + txt [ i + M ] ) % q ;
if ( t < 0 ) t = ( t + q ) ; } } }
int main ( ) { char txt [ ] = " GEEKS ▁ FOR ▁ GEEKS " ; char pat [ ] = " GEEK " ;
int q = 101 ;
search ( pat , txt , q ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void search ( string pat , string txt ) { int M = pat . size ( ) ; int N = txt . size ( ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) { cout << " Pattern ▁ found ▁ at ▁ index ▁ " << i << endl ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { string txt = " ABCEABCDABCEABCD " ; string pat = " ABCD " ; search ( pat , txt ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define NO_OF_CHARS  256  NEW_LINE int getNextState ( string pat , int M , int state , int x ) {
if ( state < M && x == pat [ state ] ) return state + 1 ;
int ns , i ;
for ( ns = state ; ns > 0 ; ns -- ) { if ( pat [ ns - 1 ] == x ) { for ( i = 0 ; i < ns - 1 ; i ++ ) if ( pat [ i ] != pat [ state - ns + 1 + i ] ) break ; if ( i == ns - 1 ) return ns ; } } return 0 ; }
void computeTF ( string pat , int M , int TF [ ] [ NO_OF_CHARS ] ) { int state , x ; for ( state = 0 ; state <= M ; ++ state ) for ( x = 0 ; x < NO_OF_CHARS ; ++ x ) TF [ state ] [ x ] = getNextState ( pat , M , state , x ) ; }
void search ( string pat , string txt ) { int M = pat . size ( ) ; int N = txt . size ( ) ; int TF [ M + 1 ] [ NO_OF_CHARS ] ; computeTF ( pat , M , TF ) ;
int i , state = 0 ; for ( i = 0 ; i < N ; i ++ ) { state = TF [ state ] [ txt [ i ] ] ; if ( state == M ) cout << " ▁ Pattern ▁ found ▁ at ▁ index ▁ " << i - M + 1 << endl ; } }
int main ( ) { string txt = " AABAACAADAABAAABAA " ; string pat = " AABA " ; search ( pat , txt ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; # define NO_OF_CHARS  256
void badCharHeuristic ( string str , int size , int badchar [ NO_OF_CHARS ] ) { int i ;
for ( i = 0 ; i < NO_OF_CHARS ; i ++ ) badchar [ i ] = -1 ;
for ( i = 0 ; i < size ; i ++ ) badchar [ ( int ) str [ i ] ] = i ; }
void search ( string txt , string pat ) { int m = pat . size ( ) ; int n = txt . size ( ) ; int badchar [ NO_OF_CHARS ] ;
badCharHeuristic ( pat , m , badchar ) ;
int s = 0 ;
while ( s <= ( n - m ) ) { int j = m - 1 ;
while ( j >= 0 && pat [ j ] == txt [ s + j ] ) j -- ;
if ( j < 0 ) { cout << " pattern ▁ occurs ▁ at ▁ shift ▁ = ▁ " << s << endl ;
s += ( s + m < n ) ? m - badchar [ txt [ s + m ] ] : 1 ; } else
s += max ( 1 , j - badchar [ txt [ s + j ] ] ) ; } }
int main ( ) { string txt = " ABAAABCD " ; string pat = " ABC " ; search ( txt , pat ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ARRAYSIZE ( a )  (sizeof(a)) / (sizeof(a[0])) NEW_LINE #define mx  200 NEW_LINE static int total_nodes ;
void printSubset ( int A [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) { cout << A [ i ] << " ▁ " ; } cout << " STRNEWLINE ▁ " ; }
void subset_sum ( int s [ ] , int t [ ] , int s_size , int t_size , int sum , int ite , int const target_sum ) { total_nodes ++ ; if ( target_sum == sum ) {
printSubset ( t , t_size ) ;
subset_sum ( s , t , s_size , t_size - 1 , sum - s [ ite ] , ite + 1 , target_sum ) ; return ; } else {
for ( int i = ite ; i < s_size ; i ++ ) { t [ t_size ] = s [ i ] ;
subset_sum ( s , t , s_size , t_size + 1 , sum + s [ i ] , i + 1 , target_sum ) ; } } }
void generateSubsets ( int s [ ] , int size , int target_sum ) { int * tuplet_vector = new int [ mx ] ; subset_sum ( s , tuplet_vector , size , 0 , 0 , 0 , target_sum ) ; free ( tuplet_vector ) ; }
int main ( ) { int weights [ ] = { 10 , 7 , 5 , 18 , 12 , 20 , 15 } ; int size = ARRAYSIZE ( weights ) ; generateSubsets ( weights , size , 35 ) ; cout << " Nodes ▁ generated ▁ " << total_nodes << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ARRAYSIZE ( a )  (sizeof(a))/(sizeof(a[0])) NEW_LINE static int total_nodes ;
void printSubset ( int A [ ] , int size ) { for ( int i = 0 ; i < size ; i ++ ) { cout << " ▁ " << A [ i ] ; } cout << " STRNEWLINE " ; }
int comparator ( const void * pLhs , const void * pRhs ) { int * lhs = ( int * ) pLhs ; int * rhs = ( int * ) pRhs ; return * lhs > * rhs ; }
void subset_sum ( int s [ ] , int t [ ] , int s_size , int t_size , int sum , int ite , int const target_sum ) { total_nodes ++ ; if ( target_sum == sum ) {
printSubset ( t , t_size ) ;
if ( ite + 1 < s_size && sum - s [ ite ] + s [ ite + 1 ] <= target_sum ) {
subset_sum ( s , t , s_size , t_size - 1 , sum - s [ ite ] , ite + 1 , target_sum ) ; } return ; } else {
if ( ite < s_size && sum + s [ ite ] <= target_sum ) {
for ( int i = ite ; i < s_size ; i ++ ) { t [ t_size ] = s [ i ] ; if ( sum + s [ i ] <= target_sum ) {
subset_sum ( s , t , s_size , t_size + 1 , sum + s [ i ] , i + 1 , target_sum ) ; } } } } }
void generateSubsets ( int s [ ] , int size , int target_sum ) { int * tuplet_vector = ( int * ) malloc ( size * sizeof ( int ) ) ; int total = 0 ;
qsort ( s , size , sizeof ( int ) , & comparator ) ; for ( int i = 0 ; i < size ; i ++ ) { total += s [ i ] ; } if ( s [ 0 ] <= target_sum && total >= target_sum ) { subset_sum ( s , tuplet_vector , size , 0 , 0 , 0 , target_sum ) ; } free ( tuplet_vector ) ; }
int main ( ) { int weights [ ] = { 15 , 22 , 14 , 26 , 32 , 9 , 16 , 8 } ; int target = 53 ; int size = ARRAYSIZE ( weights ) ; generateSubsets ( weights , size , target ) ; cout << " Nodes ▁ generated ▁ " << total_nodes ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
#define N  9
void print ( int arr [ N ] [ N ] ) { for ( int i = 0 ; i < N ; i ++ ) { for ( int j = 0 ; j < N ; j ++ ) cout << arr [ i ] [ j ] << " ▁ " ; cout << endl ; } }
bool isSafe ( int grid [ N ] [ N ] , int row , int col , int num ) {
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ row ] [ x ] == num ) return false ;
for ( int x = 0 ; x <= 8 ; x ++ ) if ( grid [ x ] [ col ] == num ) return false ;
int startRow = row - row % 3 , startCol = col - col % 3 ; for ( int i = 0 ; i < 3 ; i ++ ) for ( int j = 0 ; j < 3 ; j ++ ) if ( grid [ i + startRow ] [ j + startCol ] == num ) return false ; return true ; }
bool solveSuduko ( int grid [ N ] [ N ] , int row , int col ) {
if ( row == N - 1 && col == N ) return true ;
if ( col == N ) { row ++ ; col = 0 ; }
if ( grid [ row ] [ col ] > 0 ) return solveSuduko ( grid , row , col + 1 ) ; for ( int num = 1 ; num <= N ; num ++ ) {
if ( isSafe ( grid , row , col , num ) ) {
grid [ row ] [ col ] = num ;
if ( solveSuduko ( grid , row , col + 1 ) ) return true ; }
grid [ row ] [ col ] = 0 ; } return false ; }
int main ( ) { int grid [ N ] [ N ] = { { 3 , 0 , 6 , 5 , 0 , 8 , 4 , 0 , 0 } , { 5 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 } , { 0 , 8 , 7 , 0 , 0 , 0 , 0 , 3 , 1 } , { 0 , 0 , 3 , 0 , 1 , 0 , 0 , 8 , 0 } , { 9 , 0 , 0 , 8 , 6 , 3 , 0 , 0 , 5 } , { 0 , 5 , 0 , 0 , 9 , 0 , 6 , 0 , 0 } , { 1 , 3 , 0 , 0 , 0 , 0 , 2 , 5 , 0 } , { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 7 , 4 } , { 0 , 0 , 5 , 2 , 0 , 6 , 3 , 0 , 0 } } ; if ( solveSuduko ( grid , 0 , 0 ) ) print ( grid ) ; else cout << " no ▁ solution ▁ exists ▁ " << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getMedian ( int ar1 [ ] , int ar2 [ ] , int n ) { int i = 0 ; int j = 0 ; int count ; int m1 = -1 , m2 = -1 ;
for ( count = 0 ; count <= n ; count ++ ) {
if ( i == n ) { m1 = m2 ; m2 = ar2 [ 0 ] ; break ; }
else if ( j == n ) { m1 = m2 ; m2 = ar1 [ 0 ] ; break ; }
if ( ar1 [ i ] <= ar2 [ j ] ) {
m1 = m2 ; m2 = ar1 [ i ] ; i ++ ; } else {
m1 = m2 ; m2 = ar2 [ j ] ; j ++ ; } } return ( m1 + m2 ) / 2 ; }
int main ( ) { int ar1 [ ] = { 1 , 12 , 15 , 26 , 38 } ; int ar2 [ ] = { 2 , 13 , 17 , 30 , 45 } ; int n1 = sizeof ( ar1 ) / sizeof ( ar1 [ 0 ] ) ; int n2 = sizeof ( ar2 ) / sizeof ( ar2 [ 0 ] ) ; if ( n1 == n2 ) cout << " Median ▁ is ▁ " << getMedian ( ar1 , ar2 , n1 ) ; else cout << " Doesn ' t ▁ work ▁ for ▁ arrays " << " ▁ of ▁ unequal ▁ size " ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int median ( int [ ] , int ) ;
int getMedian ( int ar1 [ ] , int ar2 [ ] , int n ) { if ( n <= 0 ) return -1 ; if ( n == 1 ) return ( ar1 [ 0 ] + ar2 [ 0 ] ) / 2 ; if ( n == 2 ) return ( max ( ar1 [ 0 ] , ar2 [ 0 ] ) + min ( ar1 [ 1 ] , ar2 [ 1 ] ) ) / 2 ;
int m1 = median ( ar1 , n ) ;
int m2 = median ( ar2 , n ) ;
if ( m1 == m2 ) return m1 ;
if ( m1 < m2 ) { if ( n % 2 == 0 ) return getMedian ( ar1 + n / 2 - 1 , ar2 , n - n / 2 + 1 ) ; return getMedian ( ar1 + n / 2 , ar2 , n - n / 2 ) ; }
if ( n % 2 == 0 ) return getMedian ( ar2 + n / 2 - 1 , ar1 , n - n / 2 + 1 ) ; return getMedian ( ar2 + n / 2 , ar1 , n - n / 2 ) ; }
int median ( int arr [ ] , int n ) { if ( n % 2 == 0 ) return ( arr [ n / 2 ] + arr [ n / 2 - 1 ] ) / 2 ; else return arr [ n / 2 ] ; }
int main ( ) { int ar1 [ ] = { 1 , 2 , 3 , 6 } ; int ar2 [ ] = { 4 , 6 , 8 , 10 } ; int n1 = sizeof ( ar1 ) / sizeof ( ar1 [ 0 ] ) ; int n2 = sizeof ( ar2 ) / sizeof ( ar2 [ 0 ] ) ; if ( n1 == n2 ) cout << " Median ▁ is ▁ " << getMedian ( ar1 , ar2 , n1 ) ; else cout << " Doesn ' t ▁ work ▁ for ▁ arrays ▁ " << " of ▁ unequal ▁ size " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Point { public : int x , y ; } ;
int compareX ( const void * a , const void * b ) { Point * p1 = ( Point * ) a , * p2 = ( Point * ) b ; return ( p1 -> x - p2 -> x ) ; }
int compareY ( const void * a , const void * b ) { Point * p1 = ( Point * ) a , * p2 = ( Point * ) b ; return ( p1 -> y - p2 -> y ) ; }
float dist ( Point p1 , Point p2 ) { return sqrt ( ( p1 . x - p2 . x ) * ( p1 . x - p2 . x ) + ( p1 . y - p2 . y ) * ( p1 . y - p2 . y ) ) ; }
float bruteForce ( Point P [ ] , int n ) { float min = FLT_MAX ; for ( int i = 0 ; i < n ; ++ i ) for ( int j = i + 1 ; j < n ; ++ j ) if ( dist ( P [ i ] , P [ j ] ) < min ) min = dist ( P [ i ] , P [ j ] ) ; return min ; }
float min ( float x , float y ) { return ( x < y ) ? x : y ; }
float stripClosest ( Point strip [ ] , int size , float d ) {
float min = d ; qsort ( strip , size , sizeof ( Point ) , compareY ) ;
for ( int i = 0 ; i < size ; ++ i ) for ( int j = i + 1 ; j < size && ( strip [ j ] . y - strip [ i ] . y ) < min ; ++ j ) if ( dist ( strip [ i ] , strip [ j ] ) < min ) min = dist ( strip [ i ] , strip [ j ] ) ; return min ; }
float closestUtil ( Point P [ ] , int n ) {
if ( n <= 3 ) return bruteForce ( P , n ) ;
int mid = n / 2 ; Point midPoint = P [ mid ] ;
float dl = closestUtil ( P , mid ) ; float dr = closestUtil ( P + mid , n - mid ) ;
float d = min ( dl , dr ) ;
Point strip [ n ] ; int j = 0 ; for ( int i = 0 ; i < n ; i ++ ) if ( abs ( P [ i ] . x - midPoint . x ) < d ) strip [ j ] = P [ i ] , j ++ ;
return min ( d , stripClosest ( strip , j , d ) ) ; }
float closest ( Point P [ ] , int n ) { qsort ( P , n , sizeof ( Point ) , compareX ) ;
return closestUtil ( P , n ) ; }
int main ( ) { Point P [ ] = { { 2 , 3 } , { 12 , 30 } , { 40 , 50 } , { 5 , 1 } , { 12 , 10 } , { 3 , 4 } } ; int n = sizeof ( P ) / sizeof ( P [ 0 ] ) ; cout << " The ▁ smallest ▁ distance ▁ is ▁ " << closest ( P , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define bool  int
bool isLucky ( int n ) { static int counter = 2 ;
int next_position = n ; if ( counter > n ) return 1 ; if ( n % counter == 0 ) return 0 ;
next_position -= next_position / counter ; counter ++ ; return isLucky ( next_position ) ; }
int main ( ) { int x = 5 ; if ( isLucky ( x ) ) cout << x << " ▁ is ▁ a ▁ lucky ▁ no . " ; else cout << x << " ▁ is ▁ not ▁ a ▁ lucky ▁ no . " ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int pow ( int a , int b ) { if ( b == 0 ) return 1 ; int answer = a ; int increment = a ; int i , j ; for ( i = 1 ; i < b ; i ++ ) { for ( j = 1 ; j < a ; j ++ ) { answer += increment ; } increment = answer ; } return answer ; }
int main ( ) { cout << pow ( 5 , 3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int multiply ( int x , int y ) { if ( y ) return ( x + multiply ( x , y - 1 ) ) ; else return 0 ; }
int pow ( int a , int b ) { if ( b ) return multiply ( a , pow ( a , b - 1 ) ) ; else return 1 ; }
int main ( ) { cout << pow ( 5 , 3 ) ; getchar ( ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int count ( int n ) {
if ( n < 3 ) return n ; if ( n >= 3 && n < 10 ) return n - 1 ;
int po = 1 ; while ( n / po > 9 ) po = po * 10 ;
int msd = n / po ; if ( msd != 3 )
return count ( msd ) * count ( po - 1 ) + count ( msd ) + count ( n % po ) ; else
return count ( msd * po - 1 ) ; }
int main ( ) { cout << count ( 578 ) << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <string.h> NEW_LINE using namespace std ;
int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
int findSmallerInRight ( char * str , int low , int high ) { int countRight = 0 , i ; for ( i = low + 1 ; i <= high ; ++ i ) if ( str [ i ] < str [ low ] ) ++ countRight ; return countRight ; }
int findRank ( char * str ) { int len = strlen ( str ) ; int mul = fact ( len ) ; int rank = 1 ; int countRight ; int i ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
countRight = findSmallerInRight ( str , i , len - 1 ) ; rank += countRight * mul ; } return rank ; }
int main ( ) { char str [ ] = " string " ; cout << findRank ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX_CHAR  256
int count [ MAX_CHAR ] = { 0 } ;
int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
void populateAndIncreaseCount ( int * count , char * str ) { int i ; for ( i = 0 ; str [ i ] ; ++ i ) ++ count [ str [ i ] ] ; for ( i = 1 ; i < MAX_CHAR ; ++ i ) count [ i ] += count [ i - 1 ] ; }
void updatecount ( int * count , char ch ) { int i ; for ( i = ch ; i < MAX_CHAR ; ++ i ) -- count [ i ] ; }
int findRank ( char * str ) { int len = strlen ( str ) ; int mul = fact ( len ) ; int rank = 1 , i ;
populateAndIncreaseCount ( count , str ) ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
rank += count [ str [ i ] - 1 ] * mul ;
updatecount ( count , str [ i ] ) ; } return rank ; }
int main ( ) { char str [ ] = " string " ; cout << findRank ( str ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int compare ( const void * a , const void * b ) { return ( * ( char * ) a - * ( char * ) b ) ; }
void swap ( char * a , char * b ) { char t = * a ; * a = * b ; * b = t ; }
int findCeil ( char str [ ] , char first , int l , int h ) {
int ceilIndex = l ;
for ( int i = l + 1 ; i <= h ; i ++ ) if ( str [ i ] > first && str [ i ] < str [ ceilIndex ] ) ceilIndex = i ; return ceilIndex ; }
void sortedPermutations ( char str [ ] ) {
int size = strlen ( str ) ;
qsort ( str , size , sizeof ( str [ 0 ] ) , compare ) ;
bool isFinished = false ; while ( ! isFinished ) {
cout << str << endl ;
int i ; for ( i = size - 2 ; i >= 0 ; -- i ) if ( str [ i ] < str [ i + 1 ] ) break ;
if ( i == -1 ) isFinished = true ; else {
int ceilIndex = findCeil ( str , str [ i ] , i + 1 , size - 1 ) ;
swap ( & str [ i ] , & str [ ceilIndex ] ) ;
qsort ( str + i + 1 , size - i - 1 , sizeof ( str [ 0 ] ) , compare ) ; } } }
int main ( ) { char str [ ] = " ABCD " ; sortedPermutations ( str ) ; return 0 ; }
void reverse ( char str [ ] , int l , int h ) { while ( l < h ) { swap ( & str [ l ] , & str [ h ] ) ; l ++ ; h -- ; } }
void swap ( char * a , char * b ) { char t = * a ; * a = * b ; * b = t ; } int compare ( const void * a , const void * b ) { return ( * ( char * ) a - * ( char * ) b ) ; }
int findCeil ( char str [ ] , char first , int l , int h ) { int ceilIndex = l ; for ( int i = l + 1 ; i <= h ; i ++ ) if ( str [ i ] > first && str [ i ] < str [ ceilIndex ] ) ceilIndex = i ; return ceilIndex ; }
void sortedPermutations ( char str [ ] ) {
int size = strlen ( str ) ;
qsort ( str , size , sizeof ( str [ 0 ] ) , compare ) ;
bool isFinished = false ; while ( ! isFinished ) {
cout << str << endl ;
int i ; for ( i = size - 2 ; i >= 0 ; -- i ) if ( str [ i ] < str [ i + 1 ] ) break ;
if ( i == -1 ) isFinished = true ; else {
int ceilIndex = findCeil ( str , str [ i ] , i + 1 , size - 1 ) ;
swap ( & str [ i ] , & str [ ceilIndex ] ) ;
reverse ( str , i + 1 , size - 1 ) ; } } }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float exponential ( int n , float x ) {
float sum = 1.0f ; for ( int i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
int main ( ) { int n = 10 ; float x = 1.0f ; cout << " e ^ x ▁ = ▁ " << fixed << setprecision ( 5 ) << exponential ( n , x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findCeil ( int arr [ ] , int r , int l , int h ) { int mid ; while ( l < h ) {
mid = l + ( ( h - l ) >> 1 ) ; ( r > arr [ mid ] ) ? ( l = mid + 1 ) : ( h = mid ) ; } return ( arr [ l ] >= r ) ? l : -1 ; }
int myRand ( int arr [ ] , int freq [ ] , int n ) {
int prefix [ n ] , i ; prefix [ 0 ] = freq [ 0 ] ; for ( i = 1 ; i < n ; ++ i ) prefix [ i ] = prefix [ i - 1 ] + freq [ i ] ;
int r = ( rand ( ) % prefix [ n - 1 ] ) + 1 ;
int indexc = findCeil ( prefix , r , 0 , n - 1 ) ; return arr [ indexc ] ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 } ; int freq [ ] = { 10 , 5 , 20 , 100 } ; int i , n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
srand ( time ( NULL ) ) ;
for ( i = 0 ; i < 5 ; i ++ ) cout << myRand ( arr , freq , n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int min ( int x , int y ) { return ( x < y ) ? x : y ; }
int calcAngle ( double h , double m ) {
if ( h < 0 m < 0 h > 12 m > 60 ) printf ( " Wrong ▁ input " ) ; if ( h == 12 ) h = 0 ; if ( m == 60 ) { m = 0 ; h += 1 ; if ( h > 12 ) h = h - 12 ; }
float hour_angle = 0.5 * ( h * 60 + m ) ; float minute_angle = 6 * m ;
float angle = abs ( hour_angle - minute_angle ) ;
angle = min ( 360 - angle , angle ) ; return angle ; }
int main ( ) { cout << calcAngle ( 9 , 60 ) << endl ; cout << calcAngle ( 3 , 30 ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getSingle ( int arr [ ] , int n ) { int ones = 0 , twos = 0 ; int common_bit_mask ; for ( int i = 0 ; i < n ; i ++ ) {
twos = twos | ( ones & arr [ i ] ) ;
ones = ones ^ arr [ i ] ;
common_bit_mask = ~ ( ones & twos ) ;
ones &= common_bit_mask ;
twos &= common_bit_mask ; } return ones ; }
int main ( ) { int arr [ ] = { 3 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ " << getSingle ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define INT_SIZE  32 NEW_LINE int getSingle ( int arr [ ] , int n ) {
int result = 0 ; int x , sum ;
for ( int i = 0 ; i < INT_SIZE ; i ++ ) {
sum = 0 ; x = ( 1 << i ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] & x ) sum ++ ; }
if ( ( sum % 3 ) != 0 ) result |= x ; } return result ; }
int main ( ) { int arr [ ] = { 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ " << getSingle ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unsigned int getLeftmostBit ( int n ) { int m = 0 ; while ( n > 1 ) { n = n >> 1 ; m ++ ; } return m ; }
unsigned int getNextLeftmostBit ( int n , int m ) { unsigned int temp = 1 << m ; while ( n < temp ) { temp = temp >> 1 ; m -- ; } return m ; }
unsigned int _countSetBits ( unsigned int n , int m ) ; unsigned int countSetBits ( unsigned int n ) {
int m = getLeftmostBit ( n ) ;
return _countSetBits ( n , m ) ; } unsigned int _countSetBits ( unsigned int n , int m ) {
if ( n == 0 ) return 0 ;
m = getNextLeftmostBit ( n , m ) ;
if ( n == ( ( unsigned int ) 1 << ( m + 1 ) ) - 1 ) return ( unsigned int ) ( m + 1 ) * ( 1 << m ) ;
n = n - ( 1 << m ) ; return ( n + 1 ) + countSetBits ( n ) + m * ( 1 << ( m - 1 ) ) ; }
int main ( ) { int n = 17 ; cout << " Total ▁ set ▁ bit ▁ count ▁ is ▁ " << countSetBits ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int swapBits ( unsigned int x , unsigned int p1 , unsigned int p2 , unsigned int n ) {
unsigned int set1 = ( x >> p1 ) & ( ( 1U << n ) - 1 ) ;
unsigned int set2 = ( x >> p2 ) & ( ( 1U << n ) - 1 ) ;
unsigned int Xor = ( set1 ^ set2 ) ;
Xor = ( Xor << p1 ) | ( Xor << p2 ) ;
unsigned int result = x ^ Xor ; return result ; }
int main ( ) { int res = swapBits ( 28 , 0 , 3 , 2 ) ; cout << " Result ▁ = ▁ " << res ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int smallest ( int x , int y , int z ) { int c = 0 ; while ( x && y && z ) { x -- ; y -- ; z -- ; c ++ ; } return c ; }
int main ( ) { int x = 12 , y = 15 , z = 5 ; cout << " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ " << smallest ( x , y , z ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define CHAR_BIT  8
int min ( int x , int y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( sizeof ( int ) * CHAR_BIT - 1 ) ) ) ; }
int smallest ( int x , int y , int z ) { return min ( x , min ( y , z ) ) ; }
int main ( ) { int x = 12 , y = 15 , z = 5 ; cout << " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ " << smallest ( x , y , z ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int smallest ( int x , int y , int z ) {
if ( ! ( y / x ) ) return ( ! ( y / z ) ) ? y : z ; return ( ! ( x / z ) ) ? x : z ; }
int main ( ) { int x = 78 , y = 88 , z = 68 ; cout << " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ " << smallest ( x , y , z ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void changeToZero ( int a [ 2 ] ) { a [ a [ 1 ] ] = a [ ! a [ 1 ] ] ; }
int main ( ) { int a [ ] = { 1 , 0 } ; changeToZero ( a ) ; cout << " arr [ 0 ] ▁ = ▁ " << a [ 0 ] << endl ; cout << " ▁ arr [ 1 ] ▁ = ▁ " << a [ 1 ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <time.h> NEW_LINE using namespace std ;
#define SIZE  (1 << 16) 
#define GROUP_A ( x )  x, x + 1, x + 1, x + 2 
#define GROUP_B ( x )  GROUP_A(x), GROUP_A(x+1), GROUP_A(x+1), GROUP_A(x+2) 
#define GROUP_C ( x )  GROUP_B(x), GROUP_B(x+1), GROUP_B(x+1), GROUP_B(x+2) 
#define META_LOOK_UP ( PARAMETER ) \NEW_LINEGROUP_##PARAMETER(0),\NEW_LINEGROUP_##PARAMETER(1),\NEW_LINEGROUP_##PARAMETER(1),\NEW_LINEGROUP_##PARAMETER(2)\NEW_LINEint countSetBits(int array[], size_t array_size) NEW_LINE { int count = 0 ;
static unsigned char const look_up [ ] = { META_LOOK_UP ( C ) } ;
unsigned char * pData = NULL ; for ( size_t index = 0 ; index < array_size ; index ++ ) {
pData = ( unsigned char * ) & array [ index ] ;
count += look_up [ pData [ 0 ] ] ; count += look_up [ pData [ 1 ] ] ; count += look_up [ pData [ 2 ] ] ; count += look_up [ pData [ 3 ] ] ; } return count ; }
int main ( ) { int index ; int random [ SIZE ] ;
srand ( ( unsigned ) time ( 0 ) ) ;
for ( index = 0 ; index < SIZE ; index ++ ) { random [ index ] = rand ( ) ; } cout << " Total ▁ number ▁ of ▁ bits ▁ = ▁ " << countSetBits ( random , SIZE ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int addOne ( int x ) { int m = 1 ;
while ( x & m ) { x = x ^ m ; m <<= 1 ; }
x = x ^ m ; return x ; }
int main ( ) { cout << addOne ( 13 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int addOne ( int x ) { return ( - ( ~ x ) ) ; }
int main ( ) { cout << addOne ( 13 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int fun ( unsigned int n ) { return n & ( n - 1 ) ; }
int main ( ) { int n = 7 ; cout << " The ▁ number ▁ after ▁ unsetting ▁ the " ; cout << " ▁ rightmost ▁ set ▁ bit ▁ " << fun ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #define bool  int NEW_LINE class GFG {
public : bool isPowerOfFour ( int n ) { if ( n == 0 ) return 0 ; while ( n != 1 ) { if ( n % 4 != 0 ) return 0 ; n = n / 4 ; } return 1 ; } } ;
int main ( ) { GFG g ; int test_no = 64 ; if ( g . isPowerOfFour ( test_no ) ) cout << test_no << " ▁ is ▁ a ▁ power ▁ of ▁ 4" ; else cout << test_no << " is ▁ not ▁ a ▁ power ▁ of ▁ 4" ; getchar ( ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPowerOfFour ( unsigned int n ) { int count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) {
while ( n > 1 ) { n >>= 1 ; count += 1 ; }
return ( count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) cout << test_no << " ▁ is ▁ a ▁ power ▁ of ▁ 4" ; else cout << test_no << " ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; bool isPowerOfFour ( unsigned int n ) { return n != 0 && ( ( n & ( n - 1 ) ) == 0 ) && ! ( n & 0xAAAAAAAA ) ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) cout << test_no << " ▁ is ▁ a ▁ power ▁ of ▁ 4" ; else cout << test_no << " ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" ; }
#include <iostream> NEW_LINE using namespace std ; class gfg {
public : int min ( int x , int y ) { return y ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int max ( int x , int y ) { return x ^ ( ( x ^ y ) & - ( x < y ) ) ; } } ;
int main ( ) { gfg g ; int x = 15 ; int y = 6 ; cout << " Minimum ▁ of ▁ " << x << " ▁ and ▁ " << y << " ▁ is ▁ " ; cout << g . min ( x , y ) ; cout << " Maximum of " ▁ < < ▁ x ▁ < < STRNEWLINE " and " ▁ < < ▁ y ▁ < < ▁ " is " cout << g . max ( x , y ) ; getchar ( ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define CHARBIT  8
int min ( int x , int y ) { return y + ( ( x - y ) & ( ( x - y ) >> ( sizeof ( int ) * CHARBIT - 1 ) ) ) ; }
int max ( int x , int y ) { return x - ( ( x - y ) & ( ( x - y ) >> ( sizeof ( int ) * CHARBIT - 1 ) ) ) ; }
int main ( ) { int x = 15 ; int y = 6 ; cout << " Minimum ▁ of ▁ " << x << " ▁ and ▁ " << y << " ▁ is ▁ " ; cout << min ( x , y ) ; cout << " Maximum of " < < x < < " and " < < y < < " is " cout << max ( x , y ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int addOvf ( int * result , int a , int b ) { * result = a + b ; if ( a > 0 && b > 0 && * result < 0 ) return -1 ; if ( a < 0 && b < 0 && * result > 0 ) return -1 ; return 0 ; }
int main ( ) { int * res = new int [ ( sizeof ( int ) ) ] ; int x = 2147483640 ; int y = 10 ; cout << addOvf ( res , x , y ) ; cout << " STRNEWLINE " << * res ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int addOvf ( int * result , int a , int b ) { if ( a > INT_MAX - b ) return -1 ; else { * result = a + b ; return 0 ; } } int main ( ) { int * res = new int [ ( sizeof ( int ) ) ] ; int x = 2147483640 ; int y = 10 ; cout << addOvf ( res , x , y ) << endl ; cout << * res ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { unsigned int i = 1 ; char * c = ( char * ) & i ; if ( * c ) cout << " Little ▁ endian " ; else cout << " Big ▁ endian " ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ; class gfg { public : unsigned int getFirstSetBitPos ( int n ) { return log2 ( n & - n ) + 1 ; } } ;
int main ( ) { gfg g ; int n = 12 ; cout << g . getFirstSetBitPos ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void bin ( long n ) { long i ; cout << "0" ; for ( i = 1 << 30 ; i > 0 ; i = i / 2 ) { if ( ( n & i ) != 0 ) { cout << "1" ; } else { cout << "0" ; } } }
int main ( void ) { bin ( 7 ) ; cout << endl ; bin ( 4 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
unsigned int swapBits ( unsigned int x ) {
unsigned int even_bits = x & 0xAAAAAAAA ;
unsigned int odd_bits = x & 0x55555555 ;
even_bits >>= 1 ;
odd_bits <<= 1 ;
return ( even_bits odd_bits ) ; }
int main ( ) {
unsigned int x = 23 ;
cout << swapBits ( x ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int isPowerOfTwo ( unsigned n ) { return n && ( ! ( n & ( n - 1 ) ) ) ; }
int findPosition ( unsigned n ) { if ( ! isPowerOfTwo ( n ) ) return -1 ; unsigned i = 1 , pos = 1 ;
while ( ! ( i & n ) ) {
i = i << 1 ;
++ pos ; } return pos ; }
int main ( void ) { int n = 16 ; int pos = findPosition ( n ) ; ( pos == -1 ) ? cout << " n ▁ = ▁ " << n << " , ▁ Invalid ▁ number " << endl : cout << " n ▁ = ▁ " << n << " , ▁ Position ▁ " << pos << endl ; n = 12 ; pos = findPosition ( n ) ; ( pos == -1 ) ? cout << " n ▁ = ▁ " << n << " , ▁ Invalid ▁ number " << endl : cout << " n ▁ = ▁ " << n << " , ▁ Position ▁ " << pos << endl ; n = 128 ; pos = findPosition ( n ) ; ( pos == -1 ) ? cout << " n ▁ = ▁ " << n << " , ▁ Invalid ▁ number " << endl : cout << " n ▁ = ▁ " << n << " , ▁ Position ▁ " << pos << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int isPowerOfTwo ( unsigned n ) { return n && ( ! ( n & ( n - 1 ) ) ) ; }
int findPosition ( unsigned n ) { if ( ! isPowerOfTwo ( n ) ) return -1 ; unsigned count = 0 ;
while ( n ) { n = n >> 1 ;
++ count ; } return count ; }
int main ( void ) { int n = 0 ; int pos = findPosition ( n ) ; ( pos == -1 ) ? cout << " n ▁ = ▁ " << n << " , ▁ Invalid ▁ number STRNEWLINE " : cout << " n ▁ = ▁ " << n << " , ▁ Position ▁ " << pos << endl ; n = 12 ; pos = findPosition ( n ) ; ( pos == -1 ) ? cout << " n ▁ = ▁ " << n << " , ▁ Invalid ▁ number STRNEWLINE " : cout << " n ▁ = ▁ " << n << " , ▁ Position ▁ " << pos << endl ; n = 128 ; pos = findPosition ( n ) ; ( pos == -1 ) ? cout << " n ▁ = ▁ " << n << " , ▁ Invalid ▁ number STRNEWLINE " : cout << " n ▁ = ▁ " << n << " , ▁ Position ▁ " << pos << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { int x = 10 , y = 5 ;
x = x * y ;
y = x / y ;
x = x / y ; cout << " After ▁ Swapping : ▁ x ▁ = " << x << " , ▁ y = " << y ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { int x = 10 , y = 5 ;
x = x ^ y ;
y = x ^ y ;
x = x ^ y ; cout << " After ▁ Swapping : ▁ x ▁ = " << x << " , ▁ y = " << y ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void swap ( int * xp , int * yp ) { * xp = * xp ^ * yp ; * yp = * xp ^ * yp ; * xp = * xp ^ * yp ; }
int main ( ) { int x = 10 ; swap ( & x , & x ) ; cout << " After ▁ swap ( & x , ▁ & x ) : ▁ x ▁ = ▁ " << x ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void nextGreatest ( int arr [ ] , int size ) {
int max_from_right = arr [ size - 1 ] ;
arr [ size - 1 ] = -1 ;
for ( int i = size - 2 ; i >= 0 ; i -- ) {
int temp = arr [ i ] ;
arr [ i ] = max_from_right ;
if ( max_from_right < temp ) max_from_right = temp ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; }
int main ( ) { int arr [ ] = { 16 , 17 , 4 , 3 , 5 , 2 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; nextGreatest ( arr , size ) ; cout << " The ▁ modified ▁ array ▁ is : ▁ STRNEWLINE " ; printArray ( arr , size ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxDiff ( int arr [ ] , int arr_size ) { int max_diff = arr [ 1 ] - arr [ 0 ] ; for ( int i = 0 ; i < arr_size ; i ++ ) { for ( int j = i + 1 ; j < arr_size ; j ++ ) { if ( arr [ j ] - arr [ i ] > max_diff ) max_diff = arr [ j ] - arr [ i ] ; } } return max_diff ; }
int main ( ) { int arr [ ] = { 1 , 2 , 90 , 10 , 110 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
cout << " Maximum ▁ difference ▁ is ▁ " << maxDiff ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findMaximum ( int arr [ ] , int low , int high ) { int max = arr [ low ] ; int i ; for ( i = low + 1 ; i <= high ; i ++ ) { if ( arr [ i ] > max ) max = arr [ i ] ; else break ; } return max ; }
int main ( ) { int arr [ ] = { 1 , 30 , 40 , 50 , 60 , 70 , 23 , 20 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ maximum ▁ element ▁ is ▁ " << findMaximum ( arr , 0 , n - 1 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int findMaximum ( int arr [ ] , int low , int high ) {
if ( low == high ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] >= arr [ high ] ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] < arr [ high ] ) return arr [ high ] ; int mid = ( low + high ) / 2 ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] > arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] < arr [ mid - 1 ] ) return findMaximum ( arr , low , mid - 1 ) ;
else return findMaximum ( arr , mid + 1 , high ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 50 , 10 , 9 , 7 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ maximum ▁ element ▁ is ▁ " << findMaximum ( arr , 0 , n - 1 ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; void constructLowerArray ( int arr [ ] , int * countSmaller , int n ) { int i , j ;
for ( i = 0 ; i < n ; i ++ ) countSmaller [ i ] = 0 ; for ( i = 0 ; i < n ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ j ] < arr [ i ] ) countSmaller [ i ] ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << " STRNEWLINE " ; }
int main ( ) { int arr [ ] = { 12 , 10 , 5 , 4 , 2 , 20 , 6 , 1 , 0 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int * low = ( int * ) malloc ( sizeof ( int ) * n ) ; constructLowerArray ( arr , low , n ) ; printArray ( low , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #include <stdio.h> NEW_LINE #include <stdlib.h>
struct node { int key ; struct node * left ; struct node * right ; int height ;
int size ; } ;
int max ( int a , int b ) ;
int height ( struct node * N ) { if ( N == NULL ) return 0 ; return N -> height ; }
int size ( struct node * N ) { if ( N == NULL ) return 0 ; return N -> size ; }
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
struct node * newNode ( int key ) { struct node * node = ( struct node * ) malloc ( sizeof ( struct node ) ) ; node -> key = key ; node -> left = NULL ; node -> right = NULL ;
node -> height = 1 ; node -> size = 1 ; return ( node ) ; }
struct node * rightRotate ( struct node * y ) { struct node * x = y -> left ; struct node * T2 = x -> right ;
x -> right = y ; y -> left = T2 ;
y -> height = max ( height ( y -> left ) , height ( y -> right ) ) + 1 ; x -> height = max ( height ( x -> left ) , height ( x -> right ) ) + 1 ;
y -> size = size ( y -> left ) + size ( y -> right ) + 1 ; x -> size = size ( x -> left ) + size ( x -> right ) + 1 ;
return x ; }
struct node * leftRotate ( struct node * x ) { struct node * y = x -> right ; struct node * T2 = y -> left ;
y -> left = x ; x -> right = T2 ;
x -> height = max ( height ( x -> left ) , height ( x -> right ) ) + 1 ; y -> height = max ( height ( y -> left ) , height ( y -> right ) ) + 1 ;
x -> size = size ( x -> left ) + size ( x -> right ) + 1 ; y -> size = size ( y -> left ) + size ( y -> right ) + 1 ;
return y ; }
int getBalance ( struct node * N ) { if ( N == NULL ) return 0 ; return height ( N -> left ) - height ( N -> right ) ; }
struct node * insert ( struct node * node , int key , int * count ) {
if ( node == NULL ) return ( newNode ( key ) ) ; if ( key < node -> key ) node -> left = insert ( node -> left , key , count ) ; else { node -> right = insert ( node -> right , key , count ) ;
* count = * count + size ( node -> left ) + 1 ; }
node -> height = max ( height ( node -> left ) , height ( node -> right ) ) + 1 ; node -> size = size ( node -> left ) + size ( node -> right ) + 1 ;
int balance = getBalance ( node ) ;
if ( balance > 1 && key < node -> left -> key ) return rightRotate ( node ) ;
if ( balance < -1 && key > node -> right -> key ) return leftRotate ( node ) ;
if ( balance > 1 && key > node -> left -> key ) { node -> left = leftRotate ( node -> left ) ; return rightRotate ( node ) ; }
if ( balance < -1 && key < node -> right -> key ) { node -> right = rightRotate ( node -> right ) ; return leftRotate ( node ) ; }
return node ; }
void constructLowerArray ( int arr [ ] , int countSmaller [ ] , int n ) { int i , j ; struct node * root = NULL ;
for ( i = 0 ; i < n ; i ++ ) countSmaller [ i ] = 0 ;
for ( i = n - 1 ; i >= 0 ; i -- ) { root = insert ( root , arr [ i ] , & countSmaller [ i ] ) ; } }
void printArray ( int arr [ ] , int size ) { int i ; cout << " STRNEWLINE " ; for ( i = 0 ; i < size ; i ++ ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 10 , 6 , 15 , 20 , 30 , 5 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int * low = ( int * ) malloc ( sizeof ( int ) * n ) ; constructLowerArray ( arr , low , n ) ; cout << " Following ▁ is ▁ the ▁ constructed ▁ smaller ▁ count ▁ array " ; printArray ( low , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void swap ( int * a , int * b ) { int temp ; temp = * a ; * a = * b ; * b = temp ; }
int segregate ( int arr [ ] , int size ) { int j = 0 , i ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] <= 0 ) { swap ( & arr [ i ] , & arr [ j ] ) ;
j ++ ; } } return j ; }
int findMissingPositive ( int arr [ ] , int size ) { int i ;
for ( i = 0 ; i < size ; i ++ ) { if ( abs ( arr [ i ] ) - 1 < size && arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; }
for ( i = 0 ; i < size ; i ++ ) if ( arr [ i ] > 0 )
return i + 1 ; return size + 1 ; }
int findMissing ( int arr [ ] , int size ) {
int shift = segregate ( arr , size ) ;
return findMissingPositive ( arr + shift , size - shift ) ; }
int main ( ) { int arr [ ] = { 0 , 10 , 2 , -10 , -20 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int missing = findMissing ( arr , arr_size ) ; cout << " The ▁ smallest ▁ positive ▁ missing ▁ number ▁ is ▁ " << missing ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int getMissingNo ( int a [ ] , int n ) { int total = ( n + 1 ) * ( n + 2 ) / 2 ; for ( int i = 0 ; i < n ; i ++ ) total -= a [ i ] ; return total ; }
int main ( ) { int arr [ ] = { 1 , 2 , 4 , 5 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int miss = getMissingNo ( arr , n ) ; cout << miss ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void printTwoElements ( int arr [ ] , int size ) { int i ; cout << " ▁ The ▁ repeating ▁ element ▁ is ▁ " ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; else cout << abs ( arr [ i ] ) << " STRNEWLINE " ; } cout << " and ▁ the ▁ missing ▁ element ▁ is ▁ " ; for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) cout << ( i + 1 ) ; } }
int main ( ) { int arr [ ] = { 7 , 3 , 4 , 5 , 5 , 6 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoElements ( arr , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
int main ( ) { int arr [ ] = { 1 , 3 , 4 , 5 , 5 , 6 , 2 } ; int * x = ( int * ) malloc ( sizeof ( int ) ) ; int * y = ( int * ) malloc ( sizeof ( int ) ) ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; getTwoElements ( arr , n , x , y ) ; cout << " ▁ The ▁ missing ▁ element ▁ is ▁ " << * x << " ▁ and ▁ the ▁ repeating " << " ▁ number ▁ is ▁ " << * y ; getchar ( ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findFourElements ( int A [ ] , int n , int X ) {
for ( int i = 0 ; i < n - 3 ; i ++ ) {
for ( int j = i + 1 ; j < n - 2 ; j ++ ) {
for ( int k = j + 1 ; k < n - 1 ; k ++ ) {
for ( int l = k + 1 ; l < n ; l ++ ) if ( A [ i ] + A [ j ] + A [ k ] + A [ l ] == X ) cout << A [ i ] << " , ▁ " << A [ j ] << " , ▁ " << A [ k ] << " , ▁ " << A [ l ] ; } } } }
int main ( ) { int A [ ] = { 10 , 20 , 30 , 40 , 1 , 2 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int X = 91 ; findFourElements ( A , n , X ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class pairSum { public :
int first ;
int sec ;
int sum ; } ;
int compare ( const void * a , const void * b ) { return ( ( * ( pairSum * ) a ) . sum - ( * ( pairSum * ) b ) . sum ) ; }
bool noCommon ( pairSum a , pairSum b ) { if ( a . first == b . first a . first == b . sec a . sec == b . first a . sec == b . sec ) return false ; return true ; }
void findFourElements ( int arr [ ] , int n , int X ) { int i , j ;
int size = ( n * ( n - 1 ) ) / 2 ; pairSum aux [ size ] ;
int k = 0 ; for ( i = 0 ; i < n - 1 ; i ++ ) { for ( j = i + 1 ; j < n ; j ++ ) { aux [ k ] . sum = arr [ i ] + arr [ j ] ; aux [ k ] . first = i ; aux [ k ] . sec = j ; k ++ ; } }
qsort ( aux , size , sizeof ( aux [ 0 ] ) , compare ) ;
i = 0 ; j = size - 1 ; while ( i < size && j >= 0 ) { if ( ( aux [ i ] . sum + aux [ j ] . sum == X ) && noCommon ( aux [ i ] , aux [ j ] ) ) { cout << arr [ aux [ i ] . first ] << " , ▁ " << arr [ aux [ i ] . sec ] << " , ▁ " << arr [ aux [ j ] . first ] << " , ▁ " << arr [ aux [ j ] . sec ] << endl ; return ; } else if ( aux [ i ] . sum + aux [ j ] . sum < X ) i ++ ; else j -- ; } }
int main ( ) { int arr [ ] = { 10 , 20 , 30 , 40 , 1 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int X = 91 ;
findFourElements ( arr , n , X ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int key ; struct Node * next ; } ;
void push ( struct Node * * head_ref , int new_key ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> key = new_key ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
bool search ( struct Node * head , int x ) {
if ( head == NULL ) return false ;
if ( head -> key == x ) return true ;
return search ( head -> next , x ) ; }
int main ( ) {
struct Node * head = NULL ; int x = 21 ;
push ( & head , 10 ) ; push ( & head , 30 ) ; push ( & head , 11 ) ; push ( & head , 21 ) ; push ( & head , 14 ) ; search ( head , 21 ) ? cout << " Yes " : cout << " No " ; return 0 ; }
void deleteAlt ( Node * head ) { if ( head == NULL ) return ; Node * node = head -> next ; if ( node == NULL ) return ;
head -> next = node -> next ;
free ( node ) ;
deleteAlt ( head -> next ) ; }
void AlternatingSplit ( Node * source , Node * * aRef , Node * * bRef ) { Node aDummy ;
Node * aTail = & aDummy ; Node bDummy ;
Node * bTail = & bDummy ; Node * current = source ; aDummy . next = NULL ; bDummy . next = NULL ; while ( current != NULL ) { MoveNode ( & ( aTail -> next ) , t ) ;
aTail = aTail -> next ;
if ( current != NULL ) { MoveNode ( & ( bTail -> next ) , t ) ; bTail = bTail -> next ; } } * aRef = aDummy . next ; * bRef = bDummy . next ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * next ; } ;
bool areIdentical ( struct Node * a , struct Node * b ) { while ( a != NULL && b != NULL ) { if ( a -> data != b -> data ) return false ;
a = a -> next ; b = b -> next ; }
return ( a == NULL && b == NULL ) ; }
void push ( struct Node * * head_ref , int new_data ) {
struct Node * new_node = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
int main ( ) {
struct Node * a = NULL ; struct Node * b = NULL ; push ( & a , 1 ) ; push ( & a , 2 ) ; push ( & a , 3 ) ; push ( & b , 1 ) ; push ( & b , 2 ) ; push ( & b , 3 ) ; if ( areIdentical ( a , b ) ) cout << " Identical " ; else cout << " Not ▁ identical " ; return 0 ; }
bool areIdentical ( Node * a , Node * b ) {
if ( a == NULL && b == NULL ) return true ;
if ( a != NULL && b != NULL ) return ( a -> data == b -> data ) && areIdentical ( a -> next , b -> next ) ;
return false ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ;
void sortList ( Node * head ) {
int count [ 3 ] = { 0 , 0 , 0 } ; Node * ptr = head ;
while ( ptr != NULL ) { count [ ptr -> data ] += 1 ; ptr = ptr -> next ; } int i = 0 ; ptr = head ;
while ( ptr != NULL ) { if ( count [ i ] == 0 ) ++ i ; else { ptr -> data = i ; -- count [ i ] ; ptr = ptr -> next ; } } }
void push ( Node * * head_ref , int new_data ) {
Node * new_node = new Node ( ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( Node * node ) { while ( node != NULL ) { cout << node -> data << " ▁ " ; node = node -> next ; } cout << endl ; }
int main ( void ) {
Node * head = NULL ; push ( & head , 0 ) ; push ( & head , 1 ) ; push ( & head , 0 ) ; push ( & head , 2 ) ; push ( & head , 1 ) ; push ( & head , 1 ) ; push ( & head , 2 ) ; push ( & head , 1 ) ; push ( & head , 2 ) ; cout << " Linked ▁ List ▁ Before ▁ Sorting STRNEWLINE " ; printList ( head ) ; sortList ( head ) ; cout << " Linked ▁ List ▁ After ▁ Sorting STRNEWLINE " ; printList ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ;
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
int main ( void ) { Node * head = newNode ( 1 ) ; head -> next = newNode ( 2 ) ; head -> next -> next = newNode ( 3 ) ; head -> next -> next -> next = newNode ( 4 ) ; head -> next -> next -> next -> next = newNode ( 5 ) ; cout << " Given ▁ Linked ▁ List STRNEWLINE " ; printlist ( head ) ; head = rearrangeEvenOdd ( head ) ; cout << " Modified ▁ Linked ▁ List STRNEWLINE " ; printlist ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * next ; } ;
void deleteLast ( struct Node * head , int x ) { struct Node * temp = head , * ptr = NULL ; while ( temp ) {
if ( temp -> data == x ) ptr = temp ; temp = temp -> next ; }
if ( ptr != NULL && ptr -> next == NULL ) { temp = head ; while ( temp -> next != ptr ) temp = temp -> next ; temp -> next = NULL ; }
if ( ptr != NULL && ptr -> next != NULL ) { ptr -> data = ptr -> next -> data ; temp = ptr -> next ; ptr -> next = ptr -> next -> next ; free ( temp ) ; } }
struct Node * newNode ( int x ) { Node * node = new Node ; node -> data = x ; node -> next = NULL ; return node ; }
void display ( struct Node * head ) { struct Node * temp = head ; if ( head == NULL ) { cout << " NULL STRNEWLINE " ; return ; } while ( temp != NULL ) { cout << " ▁ - - > ▁ " << temp -> data ; temp = temp -> next ; } cout << " NULL STRNEWLINE " ; }
int main ( ) { struct Node * head = newNode ( 1 ) ; head -> next = newNode ( 2 ) ; head -> next -> next = newNode ( 3 ) ; head -> next -> next -> next = newNode ( 4 ) ; head -> next -> next -> next -> next = newNode ( 5 ) ; head -> next -> next -> next -> next -> next = newNode ( 4 ) ; head -> next -> next -> next -> next -> next -> next = newNode ( 4 ) ; cout << " Created ▁ Linked ▁ list : ▁ " ; display ( head ) ; deleteLast ( head , 4 ) ; cout << " List ▁ after ▁ deletion ▁ of ▁ 4 : ▁ " ; display ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ;
int LinkedListLength ( Node * head ) { while ( head && head -> next ) { head = head -> next -> next ; } if ( ! head ) return 0 ; return 1 ; }
void push ( Node * * head , int info ) {
Node * node = new Node ( ) ;
node -> data = info ;
node -> next = ( * head ) ;
( * head ) = node ; }
int main ( void ) { Node * head = NULL ;
push ( & head , 4 ) ; push ( & head , 5 ) ; push ( & head , 7 ) ; push ( & head , 2 ) ; push ( & head , 9 ) ; push ( & head , 6 ) ; push ( & head , 1 ) ; push ( & head , 2 ) ; push ( & head , 0 ) ; push ( & head , 5 ) ; push ( & head , 5 ) ; int check = LinkedListLength ( head ) ;
if ( check == 0 ) { cout << " Even STRNEWLINE " ; } else { cout << " Odd STRNEWLINE " ; } return 0 ; }
Node * SortedMerge ( Node * a , Node * b ) { Node * result = NULL ;
Node * * lastPtrRef = & result ; while ( 1 ) { if ( a == NULL ) { * lastPtrRef = b ; break ; } else if ( b == NULL ) { * lastPtrRef = a ; break ; } if ( a -> data <= b -> data ) { MoveNode ( lastPtrRef , & a ) ; } else { MoveNode ( lastPtrRef , & b ) ; }
lastPtrRef = & ( ( * lastPtrRef ) -> next ) ; } return ( result ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ;
void setMiddleHead ( Node * * head ) { if ( * head == NULL ) return ;
Node * one_node = ( * head ) ;
Node * two_node = ( * head ) ;
Node * prev = NULL ; while ( two_node != NULL && two_node -> next != NULL ) {
prev = one_node ;
two_node = two_node -> next -> next ;
one_node = one_node -> next ; }
prev -> next = prev -> next -> next ; one_node -> next = ( * head ) ; ( * head ) = one_node ; }
void push ( Node * * head_ref , int new_data ) {
Node * new_node = new Node ( ) ; new_node -> data = new_data ;
new_node -> next = ( * head_ref ) ;
( * head_ref ) = new_node ; }
void printList ( Node * ptr ) { while ( ptr != NULL ) { cout << ptr -> data << " ▁ " ; ptr = ptr -> next ; } cout << endl ; }
int main ( ) {
Node * head = NULL ; int i ; for ( i = 5 ; i > 0 ; i -- ) push ( & head , i ) ; cout << " ▁ list ▁ before : ▁ " ; printList ( head ) ; setMiddleHead ( & head ) ; cout << " ▁ list ▁ After : ▁ " ; printList ( head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <cinttypes> NEW_LINE using namespace std ;
class Node { public : int data ; Node * npx ;
} ;
Node * XOR ( Node * a , Node * b ) { return reinterpret_cast < Node * > ( reinterpret_cast < uintptr_t > ( a ) ^ reinterpret_cast < uintptr_t > ( b ) ) ; }
void insert ( Node * * head_ref , int data ) {
Node * new_node = new Node ( ) ; new_node -> data = data ;
new_node -> npx = * head_ref ;
if ( * head_ref != NULL ) {
( * head_ref ) -> npx = XOR ( new_node , ( * head_ref ) -> npx ) ; }
* head_ref = new_node ; }
void printList ( Node * head ) { Node * curr = head ; Node * prev = NULL ; Node * next ; cout << " Following ▁ are ▁ the ▁ nodes ▁ of ▁ Linked ▁ List : ▁ STRNEWLINE " ; while ( curr != NULL ) {
cout << curr -> data << " ▁ " ;
next = XOR ( prev , curr -> npx ) ;
prev = curr ; curr = next ; } }
int main ( ) {
Node * head = NULL ; insert ( & head , 10 ) ; insert ( & head , 20 ) ; insert ( & head , 30 ) ; insert ( & head , 40 ) ;
printList ( head ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class node { public : int data ; node * left ; node * right ; node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; } } ; void printKDistant ( node * root , int k ) { if ( root == NULL k < 0 ) return ; if ( k == 0 ) { cout << root -> data << " ▁ " ; return ; } printKDistant ( root -> left , k - 1 ) ; printKDistant ( root -> right , k - 1 ) ; }
int main ( ) {
node * root = new node ( 1 ) ; root -> left = new node ( 2 ) ; root -> right = new node ( 3 ) ; root -> left -> left = new node ( 4 ) ; root -> left -> right = new node ( 5 ) ; root -> right -> left = new node ( 8 ) ; printKDistant ( root , 2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define COUNT  10
class Node { public : int data ; Node * left , * right ;
Node ( int data ) { this -> data = data ; this -> left = NULL ; this -> right = NULL ; } } ;
void print2DUtil ( Node * root , int space ) {
if ( root == NULL ) return ;
space += COUNT ;
print2DUtil ( root -> right , space ) ;
cout << endl ; for ( int i = COUNT ; i < space ; i ++ ) cout << " ▁ " ; cout << root -> data << " STRNEWLINE " ;
print2DUtil ( root -> left , space ) ; }
void print2D ( Node * root ) {
print2DUtil ( root , 0 ) ; }
int main ( ) { Node * root = new Node ( 1 ) ; root -> left = new Node ( 2 ) ; root -> right = new Node ( 3 ) ; root -> left -> left = new Node ( 4 ) ; root -> left -> right = new Node ( 5 ) ; root -> right -> left = new Node ( 6 ) ; root -> right -> right = new Node ( 7 ) ; root -> left -> left -> left = new Node ( 8 ) ; root -> left -> left -> right = new Node ( 9 ) ; root -> left -> right -> left = new Node ( 10 ) ; root -> left -> right -> right = new Node ( 11 ) ; root -> right -> left -> left = new Node ( 12 ) ; root -> right -> left -> right = new Node ( 13 ) ; root -> right -> right -> left = new Node ( 14 ) ; root -> right -> right -> right = new Node ( 15 ) ; print2D ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * left , * right ; } ; struct Node * newNode ( int item ) { struct Node * temp = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; temp -> data = item ; temp -> left = temp -> right = NULL ; return temp ; }
void leftViewUtil ( struct Node * root , int level , int * max_level ) {
if ( root == NULL ) return ;
if ( * max_level < level ) { cout << root -> data << " ▁ " ; * max_level = level ; }
leftViewUtil ( root -> left , level + 1 , max_level ) ; leftViewUtil ( root -> right , level + 1 , max_level ) ; }
void leftView ( struct Node * root ) { int max_level = 0 ; leftViewUtil ( root , 1 , & max_level ) ; }
int main ( ) { Node * root = newNode ( 10 ) ; root -> left = newNode ( 2 ) ; root -> right = newNode ( 3 ) ; root -> left -> left = newNode ( 7 ) ; root -> left -> right = newNode ( 8 ) ; root -> right -> right = newNode ( 15 ) ; root -> right -> left = newNode ( 12 ) ; root -> right -> right -> left = newNode ( 14 ) ; leftView ( root ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int cntRotations ( char s [ ] , int n ) { int lh = 0 , rh = 0 , i , ans = 0 ;
for ( i = 0 ; i < n / 2 ; ++ i ) if ( s [ i ] == ' a ' s [ i ] == ' e ' s [ i ] == ' i ' s [ i ] == ' o ' s [ i ] == ' u ' ) { lh ++ ; }
for ( i = n / 2 ; i < n ; ++ i ) if ( s [ i ] == ' a ' s [ i ] == ' e ' s [ i ] == ' i ' s [ i ] == ' o ' s [ i ] == ' u ' ) { rh ++ ; }
if ( lh > rh ) ans ++ ;
for ( i = 1 ; i < n ; ++ i ) { if ( s [ i - 1 ] == ' a ' s [ i - 1 ] == ' e ' s [ i - 1 ] == ' i ' s [ i - 1 ] == ' o ' s [ i - 1 ] == ' u ' ) { rh ++ ; lh -- ; } if ( s [ ( i - 1 + n / 2 ) % n ] == ' a ' || s [ ( i - 1 + n / 2 ) % n ] == ' e ' || s [ ( i - 1 + n / 2 ) % n ] == ' i ' || s [ ( i - 1 + n / 2 ) % n ] == ' o ' || s [ ( i - 1 + n / 2 ) % n ] == ' u ' ) { rh -- ; lh ++ ; } if ( lh > rh ) ans ++ ; }
return ans ; }
int main ( ) { char s [ ] = " abecidft " ; int n = strlen ( s ) ;
cout << " ▁ " << cntRotations ( s , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
class Node { public : int data ; Node * next ; } ;
Node * rotateHelper ( Node * blockHead , Node * blockTail , int d , Node * * tail , int k ) { if ( d == 0 ) return blockHead ;
if ( d > 0 ) { Node * temp = blockHead ; for ( int i = 1 ; temp -> next -> next && i < k - 1 ; i ++ ) temp = temp -> next ; blockTail -> next = blockHead ; * tail = temp ; return rotateHelper ( blockTail , temp , d - 1 , tail , k ) ; }
if ( d < 0 ) { blockTail -> next = blockHead ; * tail = blockHead ; return rotateHelper ( blockHead -> next , blockHead , d + 1 , tail , k ) ; } }
Node * rotateByBlocks ( Node * head , int k , int d ) {
if ( ! head ! head -> next ) return head ;
if ( d == 0 ) return head ; Node * temp = head , * tail = NULL ;
int i ; for ( i = 1 ; temp -> next && i < k ; i ++ ) temp = temp -> next ;
Node * nextBlock = temp -> next ;
if ( i < k ) head = rotateHelper ( head , temp , d % k , & tail , i ) ; else head = rotateHelper ( head , temp , d % k , & tail , k ) ;
tail -> next = rotateByBlocks ( nextBlock , k , d % k ) ;
return head ; }
void push ( Node * * head_ref , int new_data ) { Node * new_node = new Node ; new_node -> data = new_data ; new_node -> next = ( * head_ref ) ; ( * head_ref ) = new_node ; }
void printList ( Node * node ) { while ( node != NULL ) { cout << node -> data << " ▁ " ; node = node -> next ; } }
int main ( ) {
Node * head = NULL ;
for ( int i = 9 ; i > 0 ; i -= 1 ) push ( & head , i ) ; cout << " Given ▁ linked ▁ list ▁ STRNEWLINE " ; printList ( head ) ;
int k = 3 , d = 2 ; head = rotateByBlocks ( head , k , d ) ; cout << " Rotated by blocks Linked list " ; printList ( head ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE #include <stdio.h> NEW_LINE using namespace std ;
const int N = 5 ;
struct Matrix { int * A ; int size ; } ;
void Set ( struct Matrix * m , int i , int j , int x ) { if ( i >= j ) m -> A [ ( ( m -> size ) * ( j - 1 ) - ( ( ( j - 2 ) * ( j - 1 ) ) / 2 ) + ( i - j ) ) ] = x ; }
int Get ( struct Matrix m , int i , int j ) { if ( i >= j ) return m . A [ ( ( m . size ) * ( j - 1 ) - ( ( ( j - 2 ) * ( j - 1 ) ) / 2 ) + ( i - j ) ) ] ; else return 0 ; }
void Display ( struct Matrix m ) {
for ( int i = 1 ; i <= m . size ; i ++ ) { for ( int j = 1 ; j <= m . size ; j ++ ) { if ( i >= j ) cout << m . A [ ( ( m . size ) * ( j - 1 ) - ( ( ( j - 2 ) * ( j - 1 ) ) / 2 ) + ( i - j ) ) ] << " ▁ " ; else cout << "0 ▁ " ; } cout << endl ; } }
struct Matrix createMat ( int Mat [ N ] [ N ] ) {
struct Matrix mat ;
mat . size = N ; mat . A = ( int * ) malloc ( mat . size * ( mat . size + 1 ) / 2 * sizeof ( int ) ) ;
for ( int i = 1 ; i <= mat . size ; i ++ ) { for ( int j = 1 ; j <= mat . size ; j ++ ) { Set ( & mat , i , j , Mat [ i - 1 ] [ j - 1 ] ) ; } }
return mat ; }
int main ( ) {
int Mat [ 5 ] [ 5 ] = { { 1 , 0 , 0 , 0 , 0 } , { 1 , 2 , 0 , 0 , 0 } , { 1 , 2 , 3 , 0 , 0 } , { 1 , 2 , 3 , 4 , 0 } , { 1 , 2 , 3 , 4 , 5 } } ;
struct Matrix mat = createMat ( Mat ) ;
Display ( mat ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void countSubarrays ( int arr [ ] , int n ) {
int count = 0 ;
for ( int i = 0 ; i < n ; i ++ ) { int sum = 0 ; for ( int j = i ; j < n ; j ++ ) {
if ( ( j - i ) % 2 == 0 ) sum += arr [ j ] ;
else sum -= arr [ j ] ;
if ( sum == 0 ) count ++ ; } }
cout << " ▁ " << count ; }
int main ( ) {
int arr [ ] = { 2 , 4 , 6 , 4 , 2 } ;
int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
countSubarrays ( arr , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printAlter ( int arr [ ] , int N ) {
for ( int currIndex = 0 ; currIndex < N ; currIndex ++ ) {
if ( currIndex % 2 == 0 ) { cout << arr [ currIndex ] << " ▁ " ; } } }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printAlter ( arr , N ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool canMadeEqual ( int A [ ] , int B [ ] , int n ) {
sort ( A , A + n ) ; sort ( B , B + n ) ;
for ( int i = 0 ; i < n ; i ++ ) if ( A [ i ] != B [ i ] ) return false ; return true ; }
int main ( ) { int A [ ] = { 1 , 2 , 3 } ; int B [ ] = { 1 , 3 , 2 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; if ( canMadeEqual ( A , B , n ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
struct Node { int data ; struct Node * next ; } Node ;
struct Node * swap ( struct Node * ptr1 , struct Node * ptr2 ) { struct Node * tmp = ptr2 -> next ; ptr2 -> next = ptr1 ; ptr1 -> next = tmp ; return ptr2 ; }
int bubbleSort ( struct Node * * head , int count ) { struct Node * * h ; int i , j , swapped ; for ( i = 0 ; i <= count ; i ++ ) { h = head ; swapped = 0 ; for ( j = 0 ; j < count - i - 1 ; j ++ ) { struct Node * p1 = * h ; struct Node * p2 = p1 -> next ; if ( p1 -> data > p2 -> data ) {
* h = swap ( p1 , p2 ) ; swapped = 1 ; } h = & ( * h ) -> next ; }
if ( swapped == 0 ) break ; } }
void printList ( struct Node * n ) { while ( n != NULL ) { cout << n -> data << " ▁ - > ▁ " ; n = n -> next ; } cout << endl ; }
void insertAtTheBegin ( struct Node * * start_ref , int data ) { struct Node * ptr1 = ( struct Node * ) malloc ( sizeof ( struct Node ) ) ; ptr1 -> data = data ; ptr1 -> next = * start_ref ; * start_ref = ptr1 ; }
int main ( ) { int arr [ ] = { 78 , 20 , 10 , 32 , 1 , 5 } ; int list_size , i ;
struct Node * start = NULL ; list_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
for ( i = 0 ; i < list_size ; i ++ ) insertAtTheBegin ( & start , arr [ i ] ) ;
cout << " Linked ▁ list ▁ before ▁ sorting STRNEWLINE " ; printList ( start ) ;
bubbleSort ( & start , list_size ) ;
cout << " Linked ▁ list ▁ after ▁ sorting STRNEWLINE " ; printList ( start ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void merge ( int arr [ ] , int start , int mid , int end ) { int start2 = mid + 1 ;
if ( arr [ mid ] <= arr [ start2 ] ) { return ; }
while ( start <= mid && start2 <= end ) {
if ( arr [ start ] <= arr [ start2 ] ) { start ++ ; } else { int value = arr [ start2 ] ; int index = start2 ;
while ( index != start ) { arr [ index ] = arr [ index - 1 ] ; index -- ; } arr [ start ] = value ;
start ++ ; mid ++ ; start2 ++ ; } } }
void mergeSort ( int arr [ ] , int l , int r ) { if ( l < r ) {
int m = l + ( r - l ) / 2 ;
mergeSort ( arr , l , m ) ; mergeSort ( arr , m + 1 , r ) ; merge ( arr , l , m , r ) ; } }
void printArray ( int A [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) cout << " ▁ " << A [ i ] ; cout << " STRNEWLINE " ; }
int main ( ) { int arr [ ] = { 12 , 11 , 13 , 5 , 6 , 7 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; mergeSort ( arr , 0 , arr_size - 1 ) ; printArray ( arr , arr_size ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int partition ( int * arr , int low , int high , int * lp ) ; void swap ( int * a , int * b ) { int temp = * a ; * a = * b ; * b = temp ; } void DualPivotQuickSort ( int * arr , int low , int high ) { if ( low < high ) {
int lp , rp ; rp = partition ( arr , low , high , & lp ) ; DualPivotQuickSort ( arr , low , lp - 1 ) ; DualPivotQuickSort ( arr , lp + 1 , rp - 1 ) ; DualPivotQuickSort ( arr , rp + 1 , high ) ; } } int partition ( int * arr , int low , int high , int * lp ) { if ( arr [ low ] > arr [ high ] ) swap ( & arr [ low ] , & arr [ high ] ) ;
int j = low + 1 ; int g = high - 1 , k = low + 1 , p = arr [ low ] , q = arr [ high ] ; while ( k <= g ) {
if ( arr [ k ] < p ) { swap ( & arr [ k ] , & arr [ j ] ) ; j ++ ; }
else if ( arr [ k ] >= q ) { while ( arr [ g ] > q && k < g ) g -- ; swap ( & arr [ k ] , & arr [ g ] ) ; g -- ; if ( arr [ k ] < p ) { swap ( & arr [ k ] , & arr [ j ] ) ; j ++ ; } } k ++ ; } j -- ; g ++ ;
swap ( & arr [ low ] , & arr [ j ] ) ; swap ( & arr [ high ] , & arr [ g ] ) ;
return g ; }
int main ( ) { int arr [ ] = { 24 , 8 , 42 , 75 , 29 , 77 , 38 , 57 } ; DualPivotQuickSort ( arr , 0 , 7 ) ; cout << " Sorted ▁ array : ▁ " ; for ( int i = 0 ; i < 8 ; i ++ ) cout << arr [ i ] << " ▁ " ; cout << endl ; }
#include <iostream> NEW_LINE #include <vector> NEW_LINE using namespace std ;
void constGraphWithCon ( int N , int K ) {
int Max = ( ( N - 1 ) * ( N - 2 ) ) / 2 ;
if ( K > Max ) { cout << -1 << endl ; return ; }
vector < pair < int , int > > ans ;
for ( int i = 1 ; i < N ; i ++ ) { for ( int j = i + 1 ; j <= N ; j ++ ) { ans . emplace_back ( make_pair ( i , j ) ) ; } }
for ( int i = 0 ; i < ( N - 1 ) + Max - K ; i ++ ) { cout << ans [ i ] . first << " ▁ " << ans [ i ] . second << endl ; } }
int main ( ) { int N = 5 , K = 3 ; constGraphWithCon ( N , K ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findArray ( int N , int K ) {
if ( N == 1 ) { cout << " ▁ " << K ; return ; } if ( N == 2 ) { cout << 0 << " ▁ " << K ; return ; }
int P = N - 2 ; int Q = N - 1 ;
int VAL = 0 ;
for ( int i = 1 ; i <= ( N - 3 ) ; i ++ ) { cout << " ▁ " << i ;
VAL ^= i ; } if ( VAL == K ) { cout << P << " ▁ " << Q << " ▁ " << ( P ^ Q ) ; } else { cout << 0 << " ▁ " << P << " ▁ " << ( P ^ K ^ VAL ) ; } }
int main ( ) { int N = 4 , X = 6 ;
findArray ( N , X ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countDigitSum ( int N , int K ) {
int l = ( int ) pow ( 10 , N - 1 ) , r = ( int ) pow ( 10 , N ) - 1 ; int count = 0 ; for ( int i = l ; i <= r ; i ++ ) { int num = i ;
int digits [ N ] ; for ( int j = N - 1 ; j >= 0 ; j -- ) { digits [ j ] = num % 10 ; num /= 10 ; } int sum = 0 , flag = 0 ;
for ( int j = 0 ; j < K ; j ++ ) sum += digits [ j ] ;
for ( int j = 1 ; j < N - K + 1 ; j ++ ) { int curr_sum = 0 ; for ( int m = j ; m < j + K ; m ++ ) curr_sum += digits [ m ] ;
if ( sum != curr_sum ) { flag = 1 ; break ; } }
if ( flag == 0 ) { count ++ ; } } return count ; }
int main ( ) {
int N = 2 , K = 1 ;
cout << countDigitSum ( N , K ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void convert ( string s ) {
int num = 0 ; int n = s . length ( ) ;
for ( int i = 0 ; i < n ; i ++ )
num = num * 10 + ( int ( s [ i ] ) - 48 ) ;
cout << num ; }
int main ( ) {
char s [ ] = "123" ;
convert ( s ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maxSumIS ( int arr [ ] , int n ) { int i , j , max = 0 ; int msis [ n ] ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 1 , 101 , 2 , 3 , 100 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ " " subsequence ▁ is ▁ " << maxSumIS ( arr , n ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
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
int main ( ) { int arr [ ] = { 10 , 22 , 9 , 33 , 21 , 50 , 41 , 60 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " Length ▁ of ▁ lis ▁ is ▁ " << lis ( arr , n ) << " STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define ARRAY_SIZE ( a )  (sizeof(a) / sizeof(*a))
int internalSearch ( string needle , int row , int col , string hay [ ] , int row_max , int col_max , int xx ) { int found = 0 ; if ( row >= 0 && row <= row_max && col >= 0 && col <= col_max && needle [ xx ] == hay [ row ] [ col ] ) { char match = needle [ xx ] ; xx += 1 ; hay [ row ] [ col ] = 0 ; if ( needle [ xx ] == 0 ) { found = 1 ; } else {
found += internalSearch ( needle , row , col + 1 , hay , row_max , col_max , xx ) ; found += internalSearch ( needle , row , col - 1 , hay , row_max , col_max , xx ) ; found += internalSearch ( needle , row + 1 , col , hay , row_max , col_max , xx ) ; found += internalSearch ( needle , row - 1 , col , hay , row_max , col_max , xx ) ; } hay [ row ] [ col ] = match ; } return found ; }
int searchString ( string needle , int row , int col , string str [ ] , int row_count , int col_count ) { int found = 0 ; int r , c ; for ( r = 0 ; r < row_count ; ++ r ) { for ( c = 0 ; c < col_count ; ++ c ) { found += internalSearch ( needle , r , c , str , row_count - 1 , col_count - 1 , 0 ) ; } } return found ; }
int main ( ) { string needle = " MAGIC " ; string input [ ] = { " BBABBM " , " CBMBBA " , " IBABBG " , " GOZBBI " , " ABBBBC " , " MCIGAM " } ; string str [ ARRAY_SIZE ( input ) ] ; int i ; for ( i = 0 ; i < ARRAY_SIZE ( input ) ; ++ i ) { str [ i ] = input [ i ] ; } cout << " count : ▁ " << searchString ( needle , 0 , 0 , str , ARRAY_SIZE ( str ) , str [ 0 ] . size ( ) ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isBalanced ( string exp ) {
bool flag = true ; int count = 0 ;
for ( int i = 0 ; i < exp . length ( ) ; i ++ ) { if ( exp [ i ] == ' ( ' ) { count ++ ; } else {
count -- ; } if ( count < 0 ) {
flag = false ; break ; } }
if ( count != 0 ) { flag = false ; } return flag ; }
int main ( ) { string exp1 = " ( ( ( ) ) ) ( ) ( ) " ; if ( isBalanced ( exp1 ) ) cout << " Balanced ▁ STRNEWLINE " ; else cout << " Not ▁ Balanced ▁ STRNEWLINE " ; string exp2 = " ( ) ) ( ( ( ) ) " ; if ( isBalanced ( exp2 ) ) cout << " Balanced ▁ STRNEWLINE " ; else cout << " Not ▁ Balanced ▁ STRNEWLINE " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > shortestToChar ( string s , char c ) {
vector < int > list ;
vector < int > res ;
int len = s . length ( ) ;
for ( int i = 0 ; i < len ; i ++ ) { if ( s [ i ] == c ) { list . push_back ( i ) ; } } int p1 , p2 , v1 , v2 ;
int l = list . size ( ) - 1 ;
p1 = 0 ; p2 = l > 0 ? 1 : 0 ;
for ( int i = 0 ; i < len ; i ++ ) {
v1 = list [ p1 ] ; v2 = list [ p2 ] ;
if ( i <= v1 ) { res . push_back ( v1 - i ) ; }
else if ( i <= v2 ) {
if ( i - v1 < v2 - i ) { res . push_back ( i - v1 ) ; }
else { res . push_back ( v2 - i ) ;
p1 = p2 ; p2 = p2 < l ? ( p2 + 1 ) : p2 ; } }
else { res . push_back ( i - v2 ) ; } } return res ; }
int main ( ) { string s = " geeksforgeeks " ; char c = ' e ' ; vector < int > res = shortestToChar ( s , c ) ; for ( auto i : res ) cout << i << " ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void reverse ( char str [ ] , int start , int end ) {
char temp ; while ( start <= end ) {
temp = str [ start ] ; str [ start ] = str [ end ] ; str [ end ] = temp ; start ++ ; end -- ; } }
void reverseletter ( char str [ ] , int start , int end ) { int wstart , wend ; for ( wstart = wend = start ; wend < end ; wend ++ ) { if ( str [ wend ] == ' ▁ ' ) continue ;
while ( str [ wend ] != ' ▁ ' && wend <= end ) wend ++ ; wend -- ;
reverse ( str , wstart , wend ) ; } }
int main ( ) { char str [ 1000 ] = " Ashish ▁ Yadav ▁ Abhishek ▁ Rajput ▁ Sunil ▁ Pundir " ; reverseletter ( str , 0 , strlen ( str ) - 1 ) ; cout << str ; return 0 ; }
#include <iostream> NEW_LINE #include <map> NEW_LINE #include <set> NEW_LINE #include <string> NEW_LINE int min ( int a , int b ) { return a < b ? a : b ; } using namespace std ; bool have_same_frequency ( map < char , int > & freq , int k ) { for ( auto & pair : freq ) { if ( pair . second != k && pair . second != 0 ) { return false ; } } return true ; } int count_substrings ( string s , int k ) { int count = 0 ; int distinct = ( set < char > ( s . begin ( ) , s . end ( ) ) ) . size ( ) ; for ( int length = 1 ; length <= distinct ; length ++ ) { int window_length = length * k ; map < char , int > freq ; int window_start = 0 ; int window_end = window_start + window_length - 1 ; for ( int i = window_start ; i <= min ( window_end , s . length ( ) - 1 ) ; i ++ ) { freq [ s [ i ] ] ++ ; } while ( window_end < s . length ( ) ) { if ( have_same_frequency ( freq , k ) ) { count ++ ; } freq [ s [ window_start ] ] -- ; window_start ++ ; window_end ++ ; if ( window_length < s . length ( ) ) { freq [ s [ window_end ] ] ++ ; } } } return count ; } int main ( ) { string s = " aabbcc " ; int k = 2 ; cout << count_substrings ( s , k ) << endl ; s = " aabbc " ; k = 2 ; cout << count_substrings ( s , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
char * toggleCase ( char * a ) { for ( int i = 0 ; a [ i ] != ' \0' ; i ++ ) {
a [ i ] ^= 32 ; } return a ; }
int main ( ) { char str [ ] = " CheRrY " ; cout << " Toggle ▁ case : ▁ " << toggleCase ( str ) << endl ; cout << " Original ▁ string : ▁ " << toggleCase ( str ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { int val ; char strn1 [ ] = "12546" ; val = atoi ( strn1 ) ; cout << " String ▁ value ▁ = ▁ " << strn1 << endl ; cout << " Integer ▁ value ▁ = ▁ " << val << endl ; char strn2 [ ] = " GeeksforGeeks " ; val = atoi ( strn2 ) ; cout << " String ▁ value ▁ = ▁ " << strn2 << endl ; cout << " Integer ▁ value ▁ = ▁ " << val << endl ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int ic_strcmp ( string s1 , string s2 ) { int i ; for ( i = 0 ; s1 [ i ] && s2 [ i ] ; ++ i ) {
if ( s1 [ i ] == s2 [ i ] || ( s1 [ i ] ^ 32 ) == s2 [ i ] ) continue ; else break ; }
if ( s1 [ i ] == s2 [ i ] ) return 0 ;
if ( ( s1 [ i ] 32 ) < ( s2 [ i ] 32 ) ) return -1 ; return 1 ; }
int main ( ) { cout << " ret : ▁ " << ic_strcmp ( " Geeks " , " apple " ) << endl ; cout << " ret : ▁ " << ic_strcmp ( " " , " ABCD " ) << endl ; cout << " ret : ▁ " << ic_strcmp ( " ABCD " , " z " ) << endl ; cout << " ret : ▁ " << ic_strcmp ( " ABCD " , " abcdEghe " ) << endl ; cout << " ret : ▁ " << ic_strcmp ( " GeeksForGeeks " , " gEEksFORGeEKs " ) << endl ; cout << " ret : ▁ " << ic_strcmp ( " GeeksForGeeks " , " geeksForGeeks " ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define NO_OF_CHARS  256
bool areAnagram ( char * str1 , char * str2 ) {
int count1 [ NO_OF_CHARS ] = { 0 } ; int count2 [ NO_OF_CHARS ] = { 0 } ; int i ;
for ( i = 0 ; str1 [ i ] && str2 [ i ] ; i ++ ) { count1 [ str1 [ i ] ] ++ ; count2 [ str2 [ i ] ] ++ ; }
if ( str1 [ i ] str2 [ i ] ) return false ;
for ( i = 0 ; i < NO_OF_CHARS ; i ++ ) if ( count1 [ i ] != count2 [ i ] ) return false ; return true ; }
int main ( ) { char str1 [ ] = " geeksforgeeks " ; char str2 [ ] = " forgeeksgeeks " ;
if ( areAnagram ( str1 , str2 ) ) cout << " The ▁ two ▁ strings ▁ are ▁ anagram ▁ of ▁ each ▁ other " ; else cout << " The ▁ two ▁ strings ▁ are ▁ not ▁ anagram ▁ of ▁ each ▁ " " other " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int heptacontagonNum ( int n ) { return ( 68 * n * n - 66 * n ) / 2 ; }
int main ( ) { int N = 3 ; cout << "3rd ▁ heptacontagon ▁ Number ▁ is ▁ = ▁ " << heptacontagonNum ( N ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int findMax ( int a , int b ) { int z , i , max ;
z = a - b ;
i = ( z >> 31 ) & 1 ;
max = a - ( i * z ) ;
return max ; }
int main ( ) { int A = 40 , B = 54 ;
cout << findMax ( A , B ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int power ( int N , unsigned int D ) { if ( D == 0 ) return 1 ; if ( D % 2 == 0 ) return power ( N , D / 2 ) * power ( N , D / 2 ) ; return N * power ( N , D / 2 ) * power ( N , D / 2 ) ; }
int order ( int N ) { int r = 0 ;
while ( N ) { r ++ ; N = N / 10 ; } return r ; }
int isArmstrong ( int N ) {
int D = order ( N ) ; int temp = N , sum = 0 ;
while ( temp ) { int Ni = temp % 10 ; sum += power ( Ni , D ) ; temp = temp / 10 ; }
if ( sum == N ) return 1 ; else return 0 ; }
int main ( ) {
int N = 153 ;
if ( isArmstrong ( N ) == 1 ) cout << " True " ; else cout << " False " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void primeInRange ( int L , int R ) { int flag ;
for ( int i = L ; i <= R ; i ++ ) {
if ( i == 1 i == 0 ) continue ;
flag = 1 ;
for ( int j = 2 ; j <= i / 2 ; ++ j ) { if ( i % j == 0 ) { flag = 0 ; break ; } }
if ( flag == 1 ) cout << i << " ▁ " ; } }
int main ( ) {
int L = 1 ; int R = 10 ;
primeInRange ( L , R ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define lli  long long int
void isEqualFactors ( lli N ) { if ( ( N % 2 == 0 ) and ( N % 4 != 0 ) ) cout << " YES " << endl ; else cout << " NO " << endl ; }
int main ( ) { lli N = 10 ; isEqualFactors ( N ) ; N = 125 ; isEqualFactors ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool checkDivisibility ( int n , int digit ) {
return ( digit != 0 && n % digit == 0 ) ; }
bool isAllDigitsDivide ( int n ) { int temp = n ; while ( temp > 0 ) {
int digit = temp % 10 ; if ( ! ( checkDivisibility ( n , digit ) ) ) return false ; temp /= 10 ; } return true ; }
bool isAllDigitsDistinct ( int n ) {
bool arr [ 10 ] ; for ( int i = 0 ; i < 10 ; i ++ ) arr [ i ] = false ;
while ( n > 0 ) {
int digit = n % 10 ;
if ( arr [ digit ] ) return false ;
arr [ digit ] = true ;
n = n / 10 ; } return true ; }
bool isLynchBell ( int n ) { return isAllDigitsDivide ( n ) && isAllDigitsDistinct ( n ) ; }
int main ( ) {
int N = 12 ;
if ( isLynchBell ( N ) ) cout << " Yes " ; else cout << " No " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maximumAND ( int L , int R ) { return R ; }
int main ( ) { int l = 3 ; int r = 7 ; cout << maximumAND ( l , r ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double findAverageOfCube ( int n ) {
double sum = 0 ;
for ( int i = 1 ; i <= n ; i ++ ) { sum += i * i * i ; }
return sum / n ; }
int main ( ) {
int n = 3 ;
cout << findAverageOfCube ( n ) ; }
#include " iostream " NEW_LINE using namespace std ;
struct InchFeet {
int feet ; float inch ; } ;
void findSum ( InchFeet arr [ ] , int N ) {
int feet_sum = 0 ; float inch_sum = 0.0 ; int x ;
for ( int i = 0 ; i < N ; i ++ ) {
feet_sum += arr [ i ] . feet ; inch_sum += arr [ i ] . inch ; }
if ( inch_sum >= 12 ) {
int x = ( int ) inch_sum ;
inch_sum -= x ;
inch_sum += x % 12 ;
feet_sum += x / 12 ; }
cout << " Feet ▁ Sum : ▁ " << feet_sum << ' ' << " Inch ▁ Sum : ▁ " << inch_sum << endl ; }
int main ( ) {
InchFeet arr [ ] = { { 10 , 3.7 } , { 10 , 5.5 } , { 6 , 8.0 } } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
findSum ( arr , N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
bool isPower ( int N , int K ) {
int res1 = log ( N ) / log ( K ) ; double res2 = log ( N ) / log ( K ) ;
return ( res1 == res2 ) ; }
int main ( ) { int N = 8 ; int K = 2 ; if ( isPower ( N , K ) ) { cout << " Yes " ; } else { cout << " No " ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float y ( float x ) { return ( 1 / ( 1 + x ) ) ; }
float BooleRule ( float a , float b ) {
int n = 4 ; int h ;
h = ( ( b - a ) / n ) ; float sum = 0 ;
float bl = ( ( 7 * y ( a ) + 32 * y ( a + h ) + 12 * y ( a + 2 * h ) + 32 * y ( a + 3 * h ) + 7 * y ( a + 4 * h ) ) * 2 * h / 45 ) ; sum = sum + bl ; return sum ; }
int main ( ) { float lowlimit = 0 ; float upplimit = 4 ; cout << fixed << setprecision ( 4 ) << " f ( x ) ▁ = ▁ " << BooleRule ( 0 , 4 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float y ( float x ) { float num = 1 ; float denom = 1.0 + x * x ; return num / denom ; }
float WeedleRule ( float a , float b ) {
double h = ( b - a ) / 6 ;
float sum = 0 ;
sum = sum + ( ( ( 3 * h ) / 10 ) * ( y ( a ) + y ( a + 2 * h ) + 5 * y ( a + h ) + 6 * y ( a + 3 * h ) + y ( a + 4 * h ) + 5 * y ( a + 5 * h ) + y ( a + 6 * h ) ) ) ;
return sum ; }
int main ( ) {
float a = 0 , b = 6 ;
cout << " f ( x ) ▁ = ▁ " << fixed << WeedleRule ( a , b ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float dydx ( float x , float y ) { return ( x + y - 2 ) ; }
float rungeKutta ( float x0 , float y0 , float x , float h ) {
int n = ( int ) ( ( x - x0 ) / h ) ; float k1 , k2 ;
float y = y0 ; for ( int i = 1 ; i <= n ; i ++ ) {
k1 = h * dydx ( x0 , y ) ; k2 = h * dydx ( x0 + 0.5 * h , y + 0.5 * k1 ) ;
y = y + ( 1.0 / 6.0 ) * ( k1 + 2 * k2 ) ;
x0 = x0 + h ; } return y ; }
int main ( ) { float x0 = 0 , y = 1 , x = 2 , h = 0.2 ; cout << fixed << setprecision ( 6 ) << " y ( x ) ▁ = ▁ " << rungeKutta ( x0 , y , x , h ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define PI  3.14159265
float length_rope ( float r ) { return ( ( 2 * PI * r ) + 6 * r ) ; }
int main ( ) { float r = 7 ; cout << ceil ( length_rope ( r ) ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #define PI  3.14159265 NEW_LINE using namespace std ;
float area_circumscribed ( float c ) { return ( c * c * ( PI / 4 ) ) ; }
int main ( ) { float c = 8 ; cout << area_circumscribed ( c ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
float area ( float r ) {
return ( 0.5 ) * ( 3.14 ) * ( r * r ) ; }
float perimeter ( float r ) {
return ( 3.14 ) * ( r ) ; }
int main ( ) {
int r = 10 ;
cout << " The ▁ Area ▁ of ▁ Semicircle : ▁ " << area ( r ) << endl ;
cout << " The ▁ Perimeter ▁ of ▁ Semicircle : ▁ " << perimeter ( r ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE #include <iostream> NEW_LINE #include <iomanip> NEW_LINE using namespace std ;
void equation_plane ( float x1 , float y1 , float z1 , float x2 , float y2 , float z2 , float x3 , float y3 , float z3 ) { float a1 = x2 - x1 ; float b1 = y2 - y1 ; float c1 = z2 - z1 ; float a2 = x3 - x1 ; float b2 = y3 - y1 ; float c2 = z3 - z1 ; float a = b1 * c2 - b2 * c1 ; float b = a2 * c1 - a1 * c2 ; float c = a1 * b2 - b1 * a2 ; float d = ( - a * x1 - b * y1 - c * z1 ) ; std :: cout << std :: fixed ; std :: cout << std :: setprecision ( 2 ) ; cout << " equation ▁ of ▁ plane ▁ is ▁ " << a << " ▁ x ▁ + ▁ " << b << " ▁ y ▁ + ▁ " << c << " ▁ z ▁ + ▁ " << d << " ▁ = ▁ 0 . " ; }
int main ( ) { float x1 = -1 ; float y1 = 2 ; float z1 = 1 ; float x2 = 0 ; float y2 = -3 ; float z2 = 2 ; float x3 = 1 ; float y3 = 1 ; float z3 = -4 ; equation_plane ( x1 , y1 , z1 , x2 , y2 , z2 , x3 , y3 , z3 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void octant ( float x , float y , float z ) { if ( x >= 0 && y >= 0 && z >= 0 ) cout << " Point ▁ lies ▁ in ▁ 1st ▁ octant STRNEWLINE " ; else if ( x < 0 && y >= 0 && z >= 0 ) cout << " Point ▁ lies ▁ in ▁ 2nd ▁ octant STRNEWLINE " ; else if ( x < 0 && y < 0 && z >= 0 ) cout << " Point ▁ lies ▁ in ▁ 3rd ▁ octant STRNEWLINE " ; else if ( x >= 0 && y < 0 && z >= 0 ) cout << " Point ▁ lies ▁ in ▁ 4th ▁ octant STRNEWLINE " ; else if ( x >= 0 && y >= 0 && z < 0 ) cout << " Point ▁ lies ▁ in ▁ 5th ▁ octant STRNEWLINE " ; else if ( x < 0 && y >= 0 && z < 0 ) cout << " Point ▁ lies ▁ in ▁ 6th ▁ octant STRNEWLINE " ; else if ( x < 0 && y < 0 && z < 0 ) cout << " Point ▁ lies ▁ in ▁ 7th ▁ octant STRNEWLINE " ; else if ( x >= 0 && y < 0 && z < 0 ) cout << " Point ▁ lies ▁ in ▁ 8th ▁ octant STRNEWLINE " ; }
int main ( ) { float x = 2 , y = 3 , z = 4 ; octant ( x , y , z ) ; x = -4 , y = 2 , z = -8 ; octant ( x , y , z ) ; x = -6 , y = -2 , z = 8 ; octant ( x , y , z ) ; return 0 ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ; double maxArea ( double a , double b , double c , double d ) {
double semiperimeter = ( a + b + c + d ) / 2 ;
return sqrt ( ( semiperimeter - a ) * ( semiperimeter - b ) * ( semiperimeter - c ) * ( semiperimeter - d ) ) ; }
int main ( ) { double a = 1 , b = 2 , c = 1 , d = 2 ; cout << maxArea ( a , b , c , d ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void addAP ( int A [ ] , int Q , int operations [ 2 ] [ 4 ] ) {
for ( int j = 0 ; j < 2 ; ++ j ) { int L = operations [ j ] [ 0 ] , R = operations [ j ] [ 1 ] , a = operations [ j ] [ 2 ] , d = operations [ j ] [ 3 ] ; int curr = a ;
for ( int i = L - 1 ; i < R ; i ++ ) {
A [ i ] += curr ;
curr += d ; } }
for ( int i = 0 ; i < 4 ; ++ i ) cout << A [ i ] << " ▁ " ; }
int main ( ) { int A [ ] = { 5 , 4 , 2 , 8 } ; int Q = 2 ; int Query [ 2 ] [ 4 ] = { { 1 , 2 , 1 , 3 } , { 1 , 4 , 4 , 1 } } ;
addAP ( A , Q , Query ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
void monteCarlo ( int N , int K ) {
double x , y ;
double d ;
int pCircle = 0 ;
int pSquare = 0 ; int i = 0 ;
#pragma  omp parallel firstprivate(x, y, d, i) reduction(+ : pCircle, pSquare) num_threads(K) NEW_LINE {
srand48 ( ( int ) time ( NULL ) ) ; for ( i = 0 ; i < N ; i ++ ) {
x = ( double ) drand48 ( ) ;
y = ( double ) drand48 ( ) ;
d = ( ( x * x ) + ( y * y ) ) ;
if ( d <= 1 ) {
pCircle ++ ; }
pSquare ++ ; } }
double pi = 4.0 * ( ( double ) pCircle / ( double ) ( pSquare ) ) ;
cout << " Final ▁ Estimation ▁ of ▁ Pi ▁ = ▁ " << pi ; }
int main ( ) {
int N = 100000 ; int K = 8 ;
monteCarlo ( N , K ) ; }
#include <iostream> NEW_LINE using namespace std ; int main ( void ) {
int x , y ;
int result ; x = -3 ; y = 4 ; result = x % y ; cout << result << endl ; x = 4 ; y = -2 ; result = x % y ; cout << result << endl ; x = -3 ; y = -4 ; result = x % y ; cout << result ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int log_a_to_base_b ( int a , int b ) { return log ( a ) / log ( b ) ; }
int main ( ) { int a = 3 ; int b = 2 ; cout << log_a_to_base_b ( a , b ) << endl ; a = 256 ; b = 4 ; cout << log_a_to_base_b ( a , b ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int log_a_to_base_b ( int a , int b ) { return ( a > b - 1 ) ? 1 + log_a_to_base_b ( a / b , b ) : 0 ; }
int main ( ) { int a = 3 ; int b = 2 ; cout << log_a_to_base_b ( a , b ) << endl ; a = 256 ; b = 4 ; cout << log_a_to_base_b ( a , b ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int maximum ( int x , int y ) { return ( ( x + y + abs ( x - y ) ) / 2 ) ; }
int minimum ( int x , int y ) { return ( ( x + y - abs ( x - y ) ) / 2 ) ; }
int main ( ) { int x = 99 , y = 18 ;
cout << " Maximum : ▁ " << maximum ( x , y ) << endl ;
cout << " Minimum : ▁ " << minimum ( x , y ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
double e ( int x , int n ) { static double p = 1 , f = 1 ; double r ;
if ( n == 0 ) return 1 ;
r = e ( x , n - 1 ) ;
p = p * x ;
f = f * n ; return ( r + p / f ) ; }
int main ( ) { int x = 4 , n = 15 ; cout << " STRNEWLINE " << e ( x , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; void midptellipse ( int rx , int ry , int xc , int yc ) { float dx , dy , d1 , d2 , x , y ; x = 0 ; y = ry ;
d1 = ( ry * ry ) - ( rx * rx * ry ) + ( 0.25 * rx * rx ) ; dx = 2 * ry * ry * x ; dy = 2 * rx * rx * y ;
while ( dx < dy ) {
cout << x + xc << " ▁ , ▁ " << y + yc << endl ; cout << - x + xc << " ▁ , ▁ " << y + yc << endl ; cout << x + xc << " ▁ , ▁ " << - y + yc << endl ; cout << - x + xc << " ▁ , ▁ " << - y + yc << endl ;
if ( d1 < 0 ) { x ++ ; dx = dx + ( 2 * ry * ry ) ; d1 = d1 + dx + ( ry * ry ) ; } else { x ++ ; y -- ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d1 = d1 + dx - dy + ( ry * ry ) ; } }
d2 = ( ( ry * ry ) * ( ( x + 0.5 ) * ( x + 0.5 ) ) ) + ( ( rx * rx ) * ( ( y - 1 ) * ( y - 1 ) ) ) - ( rx * rx * ry * ry ) ;
while ( y >= 0 ) {
cout << x + xc << " ▁ , ▁ " << y + yc << endl ; cout << - x + xc << " ▁ , ▁ " << y + yc << endl ; cout << x + xc << " ▁ , ▁ " << - y + yc << endl ; cout << - x + xc << " ▁ , ▁ " << - y + yc << endl ;
if ( d2 > 0 ) { y -- ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + ( rx * rx ) - dy ; } else { y -- ; x ++ ; dx = dx + ( 2 * ry * ry ) ; dy = dy - ( 2 * rx * rx ) ; d2 = d2 + dx - dy + ( rx * rx ) ; } } }
int main ( ) {
midptellipse ( 10 , 15 , 50 , 50 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void HexToBin ( string hexdec ) { long int i = 0 ; while ( hexdec [ i ] ) { switch ( hexdec [ i ] ) { case '0' : cout << "0000" ; break ; case '1' : cout << "0001" ; break ; case '2' : cout << "0010" ; break ; case '3' : cout << "0011" ; break ; case '4' : cout << "0100" ; break ; case '5' : cout << "0101" ; break ; case '6' : cout << "0110" ; break ; case '7' : cout << "0111" ; break ; case '8' : cout << "1000" ; break ; case '9' : cout << "1001" ; break ; case ' A ' : case ' a ' : cout << "1010" ; break ; case ' B ' : case ' b ' : cout << "1011" ; break ; case ' C ' : case ' c ' : cout << "1100" ; break ; case ' D ' : case ' d ' : cout << "1101" ; break ; case ' E ' : case ' e ' : cout << "1110" ; break ; case ' F ' : case ' f ' : cout << "1111" ; break ; default : cout << " Invalid hexadecimal digit " << hexdec [ i ] ; } i ++ ; } }
int main ( ) {
char hexdec [ 100 ] = "1AC5" ;
cout << " Equivalent Binary value is : " HexToBin ( hexdec ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { int matrix [ 5 ] [ 5 ] , row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
cout << " The ▁ matrix ▁ is STRNEWLINE " ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { cout << matrix [ row_index ] [ column_index ] << " ▁ " ; } cout << endl ; }
cout << " Elements on Secondary diagonal : " for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) == size - 1 ) cout << matrix [ row_index ] [ column_index ] << " , ▁ " ; } } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { int matrix [ 5 ] [ 5 ] , row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
cout << " The ▁ matrix ▁ is STRNEWLINE " ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { cout << matrix [ row_index ] [ column_index ] << " ▁ " ; } cout << endl ; }
cout << " Elements above Secondary diagonal are : " ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index + column_index ) < size - 1 ) cout << matrix [ row_index ] [ column_index ] << " , ▁ " ; } } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int main ( ) { int matrix [ 5 ] [ 5 ] , row_index , column_index , x = 0 , size = 5 ;
for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { matrix [ row_index ] [ column_index ] = ++ x ; } }
cout << " The ▁ matrix ▁ is " << endl ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) { cout << " TABSYMBOL " << matrix [ row_index ] [ column_index ] ; } cout << endl ; }
cout << " Corner Elements are : " ; for ( row_index = 0 ; row_index < size ; row_index ++ ) { for ( column_index = 0 ; column_index < size ; column_index ++ ) {
if ( ( row_index == 0 row_index == size - 1 ) && ( column_index == 0 column_index == size - 1 ) ) cout << matrix [ row_index ] [ column_index ] << " , " ; } } return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <iomanip> NEW_LINE #include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ;
void distance ( float x1 , float y1 , float z1 , float x2 , float y2 , float z2 ) { float d = sqrt ( pow ( x2 - x1 , 2 ) + pow ( y2 - y1 , 2 ) + pow ( z2 - z1 , 2 ) * 1.0 ) ; std :: cout << std :: fixed ; std :: cout << std :: setprecision ( 2 ) ; cout << " ▁ Distance ▁ is ▁ " << d ; return ; }
int main ( ) { float x1 = 2 ; float y1 = -5 ; float z1 = 7 ; float x2 = 3 ; float y2 = 4 ; float z2 = 5 ;
distance ( x1 , y1 , z1 , x2 , y2 , z2 ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define MAX  100
void add ( char v1 [ ] , char v2 [ ] ) { int i , d , c = 0 ;
int l1 = strlen ( v1 ) ; int l2 = strlen ( v2 ) ;
for ( i = l1 ; i < l2 ; i ++ ) v1 [ i ] = '0' ; for ( i = l2 ; i < l1 ; i ++ ) v2 [ i ] = '0' ;
for ( i = 0 ; i < l1 i < l2 ; i ++ ) { d = ( v1 [ i ] - '0' ) + ( v2 [ i ] - '0' ) + c ; c = d / 10 ; d %= 10 ; v1 [ i ] = '0' + d ; }
while ( c ) { v1 [ i ] = '0' + ( c % 10 ) ; c /= 10 ; i ++ ; } v1 [ i ] = ' \0' ; v2 [ l2 ] = ' \0' ; }
void subs ( char v1 [ ] , char v2 [ ] ) { int i , d , c = 0 ;
int l1 = strlen ( v1 ) ; int l2 = strlen ( v2 ) ;
for ( i = l2 ; i < l1 ; i ++ ) v2 [ i ] = '0' ;
for ( i = 0 ; i < l1 ; i ++ ) { d = ( v1 [ i ] - '0' - c ) - ( v2 [ i ] - '0' ) ; if ( d < 0 ) { d += 10 ; c = 1 ; } else c = 0 ; v1 [ i ] = '0' + d ; } v2 [ l2 ] = ' \0' ; i = l1 - 1 ; while ( i > 0 && v1 [ i ] == '0' ) i -- ; v1 [ i + 1 ] = ' \0' ; }
int divi ( char v [ ] , int q ) { int i , l = strlen ( v ) ; int c = 0 , d ;
for ( i = l - 1 ; i >= 0 ; i -- ) { d = c * 10 + ( v [ i ] - '0' ) ; c = d % q ; d /= q ; v [ i ] = '0' + d ; } i = l - 1 ; while ( i > 0 && v [ i ] == '0' ) i -- ; v [ i + 1 ] = ' \0' ; return c ; }
void rev ( char v [ ] ) { int l = strlen ( v ) ; int i ; char cc ;
for ( i = 0 ; i < l - 1 - i ; i ++ ) { cc = v [ i ] ; v [ i ] = v [ l - 1 - i ] ; v [ l - i - 1 ] = cc ; } }
void divideWithDiffK ( char a [ ] , char k [ ] ) {
rev ( a ) ; rev ( k ) ;
add ( a , k ) ;
divi ( a , 2 ) ;
rev ( a ) ; cout << " ▁ " << a ;
rev ( a ) ; subs ( a , k ) ;
rev ( a ) ; cout << " ▁ " << a ; }
int main ( ) { char a [ MAX ] = "100" , k [ MAX ] = "20" ; divideWithDiffK ( a , k ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double findArea ( double d ) { return ( d * d ) / 2.0 ; }
int main ( ) { double d = 10 ; cout << ( findArea ( d ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
float AvgofSquareN ( int n ) { float sum = 0 ; for ( int i = 1 ; i <= n ; i ++ ) sum += ( i * i ) ; return sum / n ; }
int main ( ) { int n = 2 ; cout << AvgofSquareN ( n ) ; return 0 ; }
#include <iostream> NEW_LINE #include <bits/stdc++.h> NEW_LINE using namespace std ;
long long maxPrimeFactors ( long long n ) {
long long maxPrime = -1 ;
while ( n % 2 == 0 ) { maxPrime = 2 ;
}
while ( n % 3 == 0 ) { maxPrime = 3 ; n = n / 3 ; }
for ( int i = 5 ; i <= sqrt ( n ) ; i += 6 ) { while ( n % i == 0 ) { maxPrime = i ; n = n / i ; } while ( n % ( i + 2 ) == 0 ) { maxPrime = i + 2 ; n = n / ( i + 2 ) ; } }
if ( n > 4 ) maxPrime = n ; return maxPrime ; }
int main ( ) { long long n = 15 ; cout << maxPrimeFactors ( n ) << endl ; n = 25698751364526 ; cout << maxPrimeFactors ( n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double sum ( int x , int n ) { double i , total = 1.0 , multi = x ; for ( i = 1 ; i <= n ; i ++ ) { total = total + multi / i ; multi = multi * x ; } return total ; }
int main ( ) { int x = 2 ; int n = 5 ; cout << fixed << setprecision ( 2 ) << sum ( x , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int chiliagonNum ( int n ) { return ( 998 * n * n - 996 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ chiliagon ▁ Number ▁ is ▁ = ▁ " << chiliagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int pentacontagonNum ( int n ) { return ( 48 * n * n - 46 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ pentacontagon ▁ Number ▁ is ▁ = ▁ " << pentacontagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int countDigit ( long long n ) { return floor ( log10 ( n ) + 1 ) ; }
int main ( ) { double N = 80 ; cout << countDigit ( N ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
double sum ( int x , int n ) { double i , total = 1.0 , multi = x ;
cout << total << " ▁ " ;
for ( i = 1 ; i < n ; i ++ ) { total = total + multi ; cout << multi << " ▁ " ; multi = multi * x ; } cout << " STRNEWLINE " ; return total ; }
int main ( ) { int x = 2 ; int n = 5 ; cout << fixed << setprecision ( 2 ) << sum ( x , n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int findRemainder ( int n ) {
int x = n & 3 ;
return x ; }
int main ( ) { int N = 43 ; int ans = findRemainder ( N ) ; cout << ans << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int countDigit ( long long n ) { if ( n / 10 == 0 ) return 1 ; return 1 + countDigit ( n / 10 ) ; }
int main ( void ) { long long n = 345289467 ; cout << " Number ▁ of ▁ digits ▁ : " << countDigit ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) {
int x = 1234 ;
if ( x % 9 == 1 ) cout << ( " Magic ▁ Number " ) ; else cout << ( " Not ▁ a ▁ Magic ▁ Number " ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; #define MAX  100 NEW_LINE int main ( ) {
long long int arr [ MAX ] ; arr [ 0 ] = 0 ; arr [ 1 ] = 1 ; for ( int i = 2 ; i < MAX ; i ++ ) arr [ i ] = arr [ i - 1 ] + arr [ i - 2 ] ; cout << " Fibonacci ▁ numbers ▁ divisible ▁ by ▁ " " their ▁ indexes ▁ are ▁ : STRNEWLINE " ; for ( int i = 1 ; i < MAX ; i ++ ) if ( arr [ i ] % i == 0 ) cout << " ▁ " << i ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
long long moduloMultiplication ( long long a , long long b , long long mod ) {
a %= mod ; while ( b ) {
if ( b & 1 ) res = ( res + a ) % mod ;
a = ( 2 * a ) % mod ;
} return res ; }
int main ( ) { long long a = 426 ; long long b = 964 ; long long m = 235 ; cout << moduloMultiplication ( a , b , m ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; long long int canBeSumofConsec ( long long int n ) {
n = 2 * n ;
return ( ( n & ( n - 1 ) ) != 0 ) ; } int main ( ) { long long int n = 10 ; cout << canBeSumofConsec ( n ) << " STRNEWLINE " ; }
#include <iostream> NEW_LINE #include <math.h> NEW_LINE using namespace std ; #define COMPUTER  1 NEW_LINE #define HUMAN  2
struct move { int pile_index ; int stones_removed ; } ;
void showPiles ( int piles [ ] , int n ) { int i ; cout << " Current ▁ Game ▁ Status ▁ - > ▁ " ; for ( i = 0 ; i < n ; i ++ ) cout << " ▁ " << piles [ i ] ; cout << " STRNEWLINE " ; return ; }
bool gameOver ( int piles [ ] , int n ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( piles [ i ] != 0 ) return ( false ) ; return ( true ) ; }
void declareWinner ( int whoseTurn ) { if ( whoseTurn == COMPUTER ) cout << " HUMAN won " else cout << " COMPUTER won " return ; }
int calculateNimSum ( int piles [ ] , int n ) { int i , nimsum = piles [ 0 ] ; for ( i = 1 ; i < n ; i ++ ) nimsum = nimsum ^ piles [ i ] ; return ( nimsum ) ; }
void makeMove ( int piles [ ] , int n , struct move * moves ) { int i , nim_sum = calculateNimSum ( piles , n ) ;
if ( nim_sum != 0 ) { for ( i = 0 ; i < n ; i ++ ) {
if ( ( piles [ i ] ^ nim_sum ) < piles [ i ] ) { ( * moves ) . pile_index = i ; ( * moves ) . stones_removed = piles [ i ] - ( piles [ i ] ^ nim_sum ) ; piles [ i ] = ( piles [ i ] ^ nim_sum ) ; break ; } } }
else {
int non_zero_indices [ n ] , count ; for ( i = 0 , count = 0 ; i < n ; i ++ ) if ( piles [ i ] > 0 ) non_zero_indices [ count ++ ] = i ; ( * moves ) . pile_index = ( rand ( ) % ( count ) ) ; ( * moves ) . stones_removed = 1 + ( rand ( ) % ( piles [ ( * moves ) . pile_index ] ) ) ; piles [ ( * moves ) . pile_index ] = piles [ ( * moves ) . pile_index ] - ( * moves ) . stones_removed ; if ( piles [ ( * moves ) . pile_index ] < 0 ) piles [ ( * moves ) . pile_index ] = 0 ; } return ; }
void playGame ( int piles [ ] , int n , int whoseTurn ) { cout << " GAME STARTS " struct move moves ; while ( gameOver ( piles , n ) == false ) { showPiles ( piles , n ) ; makeMove ( piles , n , & moves ) ; if ( whoseTurn == COMPUTER ) { cout << " COMPUTER ▁ removes " << moves . stones_removed << " stones ▁ from ▁ pile ▁ at ▁ index ▁ " << moves . pile_index << endl ; whoseTurn = HUMAN ; } else { cout << " HUMAN ▁ removes " << moves . stones_removed << " stones ▁ from ▁ pile ▁ at ▁ index ▁ " << moves . pile_index << endl ; whoseTurn = COMPUTER ; } } showPiles ( piles , n ) ; declareWinner ( whoseTurn ) ; return ; } void knowWinnerBeforePlaying ( int piles [ ] , int n , int whoseTurn ) { cout << " Prediction ▁ before ▁ playing ▁ the ▁ game ▁ - > ▁ " ; if ( calculateNimSum ( piles , n ) != 0 ) { if ( whoseTurn == COMPUTER ) cout << " COMPUTER ▁ will ▁ win STRNEWLINE " ; else cout << " HUMAN ▁ will ▁ win STRNEWLINE " ; } else { if ( whoseTurn == COMPUTER ) cout << " HUMAN ▁ will ▁ win STRNEWLINE " ; else cout << " COMPUTER ▁ will ▁ win STRNEWLINE " ; } return ; }
int main ( ) {
int piles [ ] = { 3 , 4 , 5 } ; int n = sizeof ( piles ) / sizeof ( piles [ 0 ] ) ;
knowWinnerBeforePlaying ( piles , n , COMPUTER ) ;
playGame ( piles , n , COMPUTER ) ;
knowWinnerBeforePlaying ( piles , n , COMPUTER ) ;
return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void findRoots ( int a , int b , int c ) {
if ( a == 0 ) { cout << " Invalid " ; return ; } int d = b * b - 4 * a * c ; double sqrt_val = sqrt ( abs ( d ) ) ; if ( d > 0 ) { cout << " Roots ▁ are ▁ real ▁ and ▁ different ▁ STRNEWLINE " ; cout << ( double ) ( - b + sqrt_val ) / ( 2 * a ) << " STRNEWLINE " << ( double ) ( - b - sqrt_val ) / ( 2 * a ) ; } else if ( d == 0 ) { cout << " Roots ▁ are ▁ real ▁ and ▁ same ▁ STRNEWLINE " ; cout << - ( double ) b / ( 2 * a ) ; }
{ cout << " Roots ▁ are ▁ complex ▁ STRNEWLINE " ; cout << - ( double ) b / ( 2 * a ) << " ▁ + ▁ i " << sqrt_val << " STRNEWLINE " << - ( double ) b / ( 2 * a ) << " ▁ - ▁ i " << sqrt_val ; } }
int main ( ) { int a = 1 , b = -7 , c = 12 ;
findRoots ( a , b , c ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int seriesSum ( int calculated , int current , int N ) { int i , cur = 1 ;
if ( current == N + 1 ) return 0 ;
for ( i = calculated ; i < calculated + current ; i ++ ) cur *= i ;
return cur + seriesSum ( i , current + 1 , N ) ; }
int main ( ) {
int N = 5 ;
cout << seriesSum ( 1 , 1 , N ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
int main ( ) { int n = 143 ; cout << " Fibonacci ▁ code ▁ word ▁ for ▁ " << n << " ▁ is ▁ " << fibonacciEncoding ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int countSquares ( int m , int n ) {
if ( n < m ) { int temp = m ; m = n ; n = temp ; }
return n * ( n + 1 ) * ( 3 * m - n + 1 ) / 6 ; }
int main ( ) { int m = 4 , n = 3 ; cout << " Count ▁ of ▁ squares ▁ is ▁ " << countSquares ( m , n ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int modInverse ( int a , int m ) { int m0 = m ; int y = 0 , x = 1 ; if ( m == 1 ) return 0 ; while ( a > 1 ) {
int q = a / m ; int t = m ;
m = a % m , a = t ; t = y ;
y = x - q * y ; x = t ; }
if ( x < 0 ) x += m0 ; return x ; }
int main ( ) { int a = 3 , m = 11 ;
cout << " Modular ▁ multiplicative ▁ inverse ▁ is ▁ " << modInverse ( a , m ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int gcd ( int a , int b ) { if ( a == 0 ) return b ; return gcd ( b % a , a ) ; }
int phi ( unsigned int n ) { unsigned int result = 1 ; for ( int i = 2 ; i < n ; i ++ ) if ( gcd ( i , n ) == 1 ) result ++ ; return result ; }
int main ( ) { int n ; for ( n = 1 ; n <= 10 ; n ++ ) cout << " phi ( " << n << " ) ▁ = ▁ " << phi ( n ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; int phi ( int n ) {
for ( int p = 2 ; p * p <= n ; ++ p ) {
if ( n % p == 0 ) {
while ( n % p == 0 ) n /= p ; result *= ( 1.0 - ( 1.0 / ( float ) p ) ) ; } }
if ( n > 1 ) result *= ( 1.0 - ( 1.0 / ( float ) n ) ) ; return ( int ) result ; }
int main ( ) { int n ; for ( n = 1 ; n <= 10 ; n ++ ) { cout << " Phi " << " ( " << n << " ) " << " ▁ = ▁ " << phi ( n ) << endl ; } return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void printFibonacciNumbers ( int n ) { int f1 = 0 , f2 = 1 , i ; if ( n < 1 ) return ; cout << f1 << " ▁ " ; for ( i = 1 ; i < n ; i ++ ) { cout << f2 << " ▁ " ; int next = f1 + f2 ; f1 = f2 ; f2 = next ; } }
int main ( ) { printFibonacciNumbers ( 7 ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
long long gcd ( long long int a , long long int b ) { if ( b == 0 ) return a ; return gcd ( b , a % b ) ; }
long long lcm ( int a , int b ) { return ( a / gcd ( a , b ) ) * b ; }
int main ( ) { int a = 15 , b = 20 ; cout << " LCM ▁ of ▁ " << a << " ▁ and ▁ " << b << " ▁ is ▁ " << lcm ( a , b ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int add ( int x , int y ) { return printf ( " % * c % * c " , x , ' ▁ ' , y , ' ▁ ' ) ; }
int main ( ) { printf ( " Sum ▁ = ▁ % d " , add ( 3 , 4 ) ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; # define MAX  11 NEW_LINE bool isMultipleof5 ( int n ) { char str [ MAX ] ; int len = strlen ( str ) ;
if ( str [ len - 1 ] == '5' str [ len - 1 ] == '0' ) return true ; return false ; }
int main ( ) { int n = 19 ; if ( isMultipleof5 ( n ) == true ) cout << n << " ▁ is ▁ multiple ▁ of ▁ 5" << endl ; else cout << n << " ▁ is ▁ not ▁ multiple ▁ of ▁ 5" << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <inttypes.h> NEW_LINE using namespace std ;
struct Node {
int data ;
struct Node * nxp ; } ;
struct Node * XOR ( struct Node * a , struct Node * b ) { return ( struct Node * ) ( ( uintptr_t ) ( a ) ^ ( uintptr_t ) ( b ) ) ; }
struct Node * insert ( struct Node * * head , int value ) {
if ( * head == NULL ) {
struct Node * node = new Node ;
node -> data = value ;
node -> nxp = XOR ( NULL , NULL ) ;
* head = node ; }
else {
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * node = new Node ( ) ;
curr -> nxp = XOR ( node , XOR ( NULL , curr -> nxp ) ) ;
node -> nxp = XOR ( NULL , curr ) ;
* head = node ;
node -> data = value ; } return * head ; }
void printList ( struct Node * * head ) {
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * next ;
while ( curr != NULL ) {
cout << curr -> data << " ▁ " ;
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ; } }
struct Node * RevInGrp ( struct Node * * head , int K , int len ) {
struct Node * curr = * head ;
if ( curr == NULL ) return NULL ;
int count = 0 ;
struct Node * prev = NULL ;
struct Node * next ;
while ( count < K && count < len ) {
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ;
count ++ ; }
prev -> nxp = XOR ( NULL , XOR ( prev -> nxp , curr ) ) ;
if ( curr != NULL ) curr -> nxp = XOR ( XOR ( curr -> nxp , prev ) , NULL ) ;
if ( len < K ) { return prev ; } else {
len -= K ;
struct Node * dummy = RevInGrp ( & curr , K , len ) ;
( * head ) -> nxp = XOR ( XOR ( NULL , ( * head ) -> nxp ) , dummy ) ;
if ( dummy != NULL ) dummy -> nxp = XOR ( XOR ( dummy -> nxp , NULL ) , * head ) ; return prev ; } }
int main ( ) {
struct Node * head = NULL ; insert ( & head , 0 ) ; insert ( & head , 2 ) ; insert ( & head , 1 ) ; insert ( & head , 3 ) ; insert ( & head , 11 ) ; insert ( & head , 8 ) ; insert ( & head , 6 ) ; insert ( & head , 7 ) ;
head = RevInGrp ( & head , 3 , 8 ) ;
printList ( & head ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node {
int data ;
Node * nxp ; } ;
Node * XOR ( Node * a , Node * b ) { return ( Node * ) ( ( uintptr_t ) ( a ) ^ ( uintptr_t ) ( b ) ) ; }
Node * insert ( Node * * head , int value ) {
if ( * head == NULL ) {
Node * node = new Node ( ) ;
node -> data = value ;
node -> nxp = XOR ( NULL , NULL ) ;
* head = node ; }
else {
Node * curr = * head ;
Node * prev = NULL ;
Node * node = new Node ( ) ;
curr -> nxp = XOR ( node , XOR ( NULL , curr -> nxp ) ) ;
node -> nxp = XOR ( NULL , curr ) ;
* head = node ;
node -> data = value ; } return * head ; }
void printList ( Node * * head ) {
Node * curr = * head ;
Node * prev = NULL ;
Node * next ;
while ( curr != NULL ) {
cout << curr -> data << " ▁ " ;
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ; } cout << endl ; }
Node * reverse ( Node * * head ) {
Node * curr = * head ; if ( curr == NULL ) return NULL ; else {
Node * prev = NULL ;
Node * next ; while ( XOR ( prev , curr -> nxp ) != NULL ) {
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ; }
* head = curr ; return * head ; } }
int main ( ) {
Node * head = NULL ; insert ( & head , 10 ) ; insert ( & head , 20 ) ; insert ( & head , 30 ) ; insert ( & head , 40 ) ;
cout << " XOR ▁ linked ▁ list : ▁ " ; printList ( & head ) ; reverse ( & head ) ; cout << " Reversed ▁ XOR ▁ linked ▁ list : ▁ " ; printList ( & head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE #include <inttypes.h> NEW_LINE using namespace std ;
struct Node {
int data ;
struct Node * nxp ; } ;
struct Node * XOR ( struct Node * a , struct Node * b ) { return ( struct Node * ) ( ( uintptr_t ) ( a ) ^ ( uintptr_t ) ( b ) ) ; }
struct Node * insert ( struct Node * * head , int value ) {
if ( * head == NULL ) {
struct Node * node = new Node ;
node -> data = value ;
node -> nxp = XOR ( NULL , NULL ) ;
* head = node ; }
else {
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * node = new Node ( ) ;
curr -> nxp = XOR ( node , XOR ( NULL , curr -> nxp ) ) ;
node -> nxp = XOR ( NULL , curr ) ;
* head = node ;
node -> data = value ; } return * head ; }
void printList ( struct Node * * head ) {
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * next ;
while ( curr != NULL ) {
cout << curr -> data << " ▁ " ;
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ; } } struct Node * NthNode ( struct Node * * head , int N ) { int count = 0 ;
struct Node * curr = * head ; struct Node * curr1 = * head ;
struct Node * prev = NULL ; struct Node * prev1 = NULL ;
struct Node * next ; struct Node * next1 ; while ( count < N && curr != NULL ) {
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ; count ++ ; } if ( curr == NULL && count < N ) { cout << " Wrong ▁ Input STRNEWLINE " ; return ( uintptr_t ) 0 ; } else { while ( curr != NULL ) {
next = XOR ( prev , curr -> nxp ) ; next1 = XOR ( prev1 , curr1 -> nxp ) ;
prev = curr ; prev1 = curr1 ;
curr = next ; curr1 = next1 ; } cout << curr1 -> data << " ▁ " ; } }
int main ( ) {
struct Node * head = NULL ; insert ( & head , 0 ) ; insert ( & head , 2 ) ; insert ( & head , 1 ) ; insert ( & head , 3 ) ; insert ( & head , 11 ) ; insert ( & head , 8 ) ; insert ( & head , 6 ) ; insert ( & head , 7 ) ; NthNode ( & head , 3 ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE #include <inttypes.h> NEW_LINE using namespace std ;
struct Node {
int data ;
struct Node * nxp ; } ;
struct Node * XOR ( struct Node * a , struct Node * b ) { return ( struct Node * ) ( ( uintptr_t ) ( a ) ^ ( uintptr_t ) ( b ) ) ; }
struct Node * insert ( struct Node * * head , int value ) {
if ( * head == NULL ) {
struct Node * node = new Node ;
node -> data = value ;
node -> nxp = XOR ( NULL , NULL ) ;
* head = node ; }
else {
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * node = new Node ( ) ;
curr -> nxp = XOR ( node , XOR ( NULL , curr -> nxp ) ) ;
node -> nxp = XOR ( NULL , curr ) ;
* head = node ;
node -> data = value ; } return * head ; }
int printMiddle ( struct Node * * head , int len ) { int count = 0 ;
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * next ; int middle = ( int ) len / 2 ;
while ( count != middle ) {
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ; count ++ ; }
if ( len & 1 ) { cout << curr -> data << " ▁ " ; }
else { cout << prev -> data << " ▁ " << curr -> data << " ▁ " ; } }
int main ( ) {
struct Node * head = NULL ; insert ( & head , 4 ) ; insert ( & head , 7 ) ; insert ( & head , 5 ) ; printMiddle ( & head , 3 ) ; return ( 0 ) ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
struct Node {
int data ;
struct Node * nxp ; } ;
struct Node * XOR ( struct Node * a , struct Node * b ) { return ( struct Node * ) ( ( uintptr_t ) ( a ) ^ ( uintptr_t ) ( b ) ) ; }
struct Node * insert ( struct Node * * head , int value , int position ) {
if ( * head == NULL ) {
if ( position == 1 ) {
struct Node * node = new Node ( ) ;
node -> data = value ;
node -> nxp = XOR ( NULL , NULL ) ;
* head = node ; }
else { cout << " Invalid ▁ Position STRNEWLINE " ; } }
else {
int Pos = 1 ;
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * next = XOR ( prev , curr -> nxp ) ;
while ( next != NULL && Pos < position - 1 ) {
prev = curr ;
curr = next ;
next = XOR ( prev , curr -> nxp ) ;
Pos ++ ; }
if ( Pos == position - 1 ) {
struct Node * node = new Node ( ) ;
struct Node * temp = XOR ( curr -> nxp , next ) ;
curr -> nxp = XOR ( temp , node ) ;
if ( next != NULL ) {
next -> nxp = XOR ( node , XOR ( next -> nxp , curr ) ) ; }
node -> nxp = XOR ( curr , next ) ; node -> data = value ; }
else if ( position == 1 ) {
struct Node * node = new Node ( ) ;
curr -> nxp = XOR ( node , XOR ( NULL , curr -> nxp ) ) ;
node -> nxp = XOR ( NULL , curr ) ;
* head = node ;
node -> data = value ; } else { cout << " Invalid ▁ Position STRNEWLINE " ; } } return * head ; }
void printList ( struct Node * * head ) {
struct Node * curr = * head ;
struct Node * prev = NULL ;
struct Node * next ;
while ( curr != NULL ) {
cout << curr -> data << " ▁ " ;
next = XOR ( prev , curr -> nxp ) ;
prev = curr ;
curr = next ; } }
int main ( ) {
struct Node * head = NULL ; insert ( & head , 10 , 1 ) ; insert ( & head , 20 , 1 ) ; insert ( & head , 30 , 3 ) ; insert ( & head , 40 , 2 ) ;
printList ( & head ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int toggleBit ( int n , int k ) { return ( n ^ ( 1 << ( k - 1 ) ) ) ; }
int main ( ) { int n = 5 , k = 2 ; cout << toggleBit ( n , k ) << endl ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int clearBit ( int n , int k ) { return ( n & ( ~ ( 1 << ( k - 1 ) ) ) ) ; }
int main ( ) { int n = 5 , k = 1 ; cout << clearBit ( n , k ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) {
unsigned char a = 5 , b = 9 ;
cout << " a > > 1 ▁ = ▁ " << ( a >> 1 ) << endl ;
cout << " b > > 1 ▁ = ▁ " << ( b >> 1 ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) { int x = 19 ; unsigned long long y = 19 ; cout << " x ▁ < < ▁ 1 ▁ = ▁ " << ( x << 1 ) << endl ; cout << " x ▁ > > ▁ 1 ▁ = ▁ " << ( x >> 1 ) << endl ;
cout << " y ▁ < < ▁ 61 ▁ = ▁ " << ( y << 61 ) << endl ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) { int i = 3 ; cout << " pow ( 2 , ▁ " << i << " ) ▁ = ▁ " << ( 1 << i ) << endl ; i = 4 ; cout << " pow ( 2 , ▁ " << i << " ) ▁ = ▁ " << ( 1 << i ) << endl ; return 0 ; }
#include <iostream> NEW_LINE #include <cmath> NEW_LINE unsigned countBits ( unsigned int number ) {
return ( int ) log2 ( number ) + 1 ; }
int main ( ) { unsigned int num = 65 ; std :: cout << countBits ( num ) << ' ' ; return 0 ; }
#include <iostream> NEW_LINE #include <bits/stdc++.h> NEW_LINE using namespace std ;
int constructNthNumber ( int group_no , int aux_num , int op ) { int INT_SIZE = 32 ; int a [ INT_SIZE ] = { 0 } ; int num = 0 , len_f ; int i = 0 ;
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
else {
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
op = 0 ; else
op = 1 ; aux_num = ( ( group_offset ) - ( 1 << ( group_no - 1 ) ) ) / 2 ; } return constructNthNumber ( group_no , aux_num , op ) ; }
int main ( ) { int n = 9 ;
cout << getNthNumber ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
unsigned int toggleAllExceptK ( unsigned int n , unsigned int k ) {
return ~ ( n ^ ( 1 << k ) ) ; }
int main ( ) { unsigned int n = 4294967295 ; unsigned int k = 0 ; cout << toggleAllExceptK ( n , k ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) { int a = 10 , b = 4 ;
if ( a > b ) cout << " a ▁ is ▁ greater ▁ than ▁ b STRNEWLINE " ; else cout << " a ▁ is ▁ less ▁ than ▁ or ▁ equal ▁ to ▁ b STRNEWLINE " ;
if ( a >= b ) cout << " a ▁ is ▁ greater ▁ than ▁ or ▁ equal ▁ to ▁ b STRNEWLINE " ; else cout << " a ▁ is ▁ lesser ▁ than ▁ b STRNEWLINE " ;
if ( a < b ) cout << " a ▁ is ▁ less ▁ than ▁ b STRNEWLINE " ; else cout << " a ▁ is ▁ greater ▁ than ▁ or ▁ equal ▁ to ▁ b STRNEWLINE " ;
if ( a <= b ) cout << " a ▁ is ▁ lesser ▁ than ▁ or ▁ equal ▁ to ▁ b STRNEWLINE " ; else cout << " a ▁ is ▁ greater ▁ than ▁ b STRNEWLINE " ;
if ( a == b ) cout << " a ▁ is ▁ equal ▁ to ▁ b STRNEWLINE " ; else cout << " a ▁ and ▁ b ▁ are ▁ not ▁ equal STRNEWLINE " ;
if ( a != b ) cout << " a ▁ is ▁ not ▁ equal ▁ to ▁ b STRNEWLINE " ; else cout << " a ▁ is ▁ equal ▁ b STRNEWLINE " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) { int a = 10 , b = 4 , c = 10 , d = 20 ;
if ( a > b && c == d ) cout << " a ▁ is ▁ greater ▁ than ▁ b ▁ AND ▁ c ▁ is ▁ equal ▁ to ▁ d STRNEWLINE " ; else cout << " AND ▁ condition ▁ not ▁ satisfied STRNEWLINE " ;
if ( a > b c == d ) cout << " a ▁ is ▁ greater ▁ than ▁ b ▁ OR ▁ c ▁ is ▁ equal ▁ to ▁ d STRNEWLINE " ; else cout << " Neither ▁ a ▁ is ▁ greater ▁ than ▁ b ▁ nor ▁ c ▁ is ▁ equal ▁ " " ▁ to ▁ d STRNEWLINE " ;
if ( ! a ) cout << " a ▁ is ▁ zero STRNEWLINE " ; else cout << " a ▁ is ▁ not ▁ zero " ; return 0 ; }
#include <stdbool.h> NEW_LINE #include <stdio.h> NEW_LINE int main ( ) { int a = 10 , b = 4 ; bool res = ( ( a != b ) cout << " GeeksQuiz " ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int swapBits ( unsigned int n , unsigned int p1 , unsigned int p2 ) {
unsigned int bit1 = ( n >> p1 ) & 1 ;
unsigned int bit2 = ( n >> p2 ) & 1 ;
unsigned int x = ( bit1 ^ bit2 ) ;
x = ( x << p1 ) | ( x << p2 ) ;
unsigned int result = n ^ x ; }
int main ( ) { int res = swapBits ( 28 , 0 , 3 ) ; cout << " Result ▁ = ▁ " << res << " ▁ " ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int findOdd ( int arr [ ] , int n ) { int res = 0 , i ; for ( i = 0 ; i < n ; i ++ ) res ^= arr [ i ] ; return res ; }
int main ( void ) { int arr [ ] = { 12 , 12 , 14 , 90 , 14 , 14 , 14 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; cout << " The ▁ odd ▁ occurring ▁ element ▁ is ▁ " << findOdd ( arr , n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ; int main ( ) { int x = 2 , y = 5 ; ( x & y ) ? cout << " True ▁ " : cout << " False ▁ " ; ( x && y ) ? cout << " True ▁ " : cout << " False ▁ " ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
vector < int > generateHammingCode ( vector < int > msgBits , int m , int r ) {
vector < int > hammingCode ( r + m ) ;
for ( int i = 0 ; i < r ; ++ i ) {
hammingCode [ pow ( 2 , i ) - 1 ] = -1 ; } int j = 0 ;
for ( int i = 0 ; i < ( r + m ) ; i ++ ) {
if ( hammingCode [ i ] != -1 ) { hammingCode [ i ] = msgBits [ j ] ; j ++ ; } } for ( int i = 0 ; i < ( r + m ) ; i ++ ) {
if ( hammingCode [ i ] != -1 ) continue ; int x = log2 ( i + 1 ) ; int one_count = 0 ;
for ( int j = i + 2 ; j <= ( r + m ) ; ++ j ) { if ( j & ( 1 << x ) ) { if ( hammingCode [ j - 1 ] == 1 ) { one_count ++ ; } } }
if ( one_count % 2 == 0 ) { hammingCode [ i ] = 0 ; } else { hammingCode [ i ] = 1 ; } }
return hammingCode ; }
void findHammingCode ( vector < int > & msgBit ) {
int m = msgBit . size ( ) ;
int r = 1 ;
while ( pow ( 2 , r ) < ( m + r + 1 ) ) { r ++ ; }
vector < int > ans = generateHammingCode ( msgBit , m , r ) ;
cout << " Message ▁ bits ▁ are : ▁ " ; for ( int i = 0 ; i < msgBit . size ( ) ; i ++ ) cout << msgBit [ i ] << " ▁ " ; cout << " Hamming code is : " ; for ( int i = 0 ; i < ans . size ( ) ; i ++ ) cout << ans [ i ] << " ▁ " ; }
int main ( ) {
vector < int > msgBit = { 0 , 1 , 0 , 1 } ;
findHammingCode ( msgBit ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ; #define NO_OF_CHARS  256
int firstNonRepeating ( char * str ) {
int arr [ NO_OF_CHARS ] ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ ) arr [ i ] = -1 ;
for ( int i = 0 ; str [ i ] ; i ++ ) { if ( arr [ str [ i ] ] == -1 ) arr [ str [ i ] ] = i ; else arr [ str [ i ] ] = -2 ; } int res = INT_MAX ; for ( int i = 0 ; i < NO_OF_CHARS ; i ++ )
if ( arr [ i ] >= 0 ) res = min ( res , arr [ i ] ) ; return res ; }
int main ( ) { char str [ ] = " geeksforgeeks " ; int index = firstNonRepeating ( str ) ; if ( index == INT_MAX ) cout << " Either ▁ all ▁ characters ▁ are ▁ " " repeating ▁ or ▁ string ▁ is ▁ empty " ; else cout << " First ▁ non - repeating ▁ character " " ▁ is ▁ " << str [ index ] ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int triacontagonalNum ( int n ) { return ( 28 * n * n - 26 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ triacontagonal ▁ Number ▁ is ▁ = ▁ " << triacontagonalNum ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int hexacontagonNum ( int n ) { return ( 58 * n * n - 56 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ hexacontagon ▁ Number ▁ is ▁ = ▁ " << hexacontagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int enneacontagonNum ( int n ) { return ( 88 * n * n - 86 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ enneacontagon ▁ Number ▁ is ▁ = ▁ " << enneacontagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int triacontakaidigonNum ( int n ) { return ( 30 * n * n - 28 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ triacontakaidigon ▁ Number ▁ is ▁ = ▁ " << triacontakaidigonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int IcosihexagonalNum ( int n ) { return ( 24 * n * n - 22 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ Icosihexagonal ▁ Number ▁ is ▁ = ▁ " << IcosihexagonalNum ( n ) ; return 0 ; }
#include <iostream> NEW_LINE using namespace std ;
int icosikaioctagonalNum ( int n ) { return ( 26 * n * n - 24 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ icosikaioctagonal ▁ Number ▁ is ▁ = ▁ " << icosikaioctagonalNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int octacontagonNum ( int n ) { return ( 78 * n * n - 76 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ octacontagon ▁ Number ▁ is ▁ = ▁ " << octacontagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int hectagonNum ( int n ) { return ( 98 * n * n - 96 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ hectagon ▁ Number ▁ is ▁ = ▁ " << hectagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
int tetracontagonNum ( int n ) { return ( 38 * n * n - 36 * n ) / 2 ; }
int main ( ) { int n = 3 ; cout << "3rd ▁ tetracontagon ▁ Number ▁ is ▁ = ▁ " << tetracontagonNum ( n ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
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
int main ( ) { int arr [ ] = { 5 , 4 , 3 , 2 , 1 } ; int N = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int X = 5 ; cout << binarySearch ( arr , N , X ) ; return 0 ; }
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void flip ( int arr [ ] , int i ) { int temp , start = 0 ; while ( start < i ) { temp = arr [ start ] ; arr [ start ] = arr [ i ] ; arr [ i ] = temp ; start ++ ; i -- ; } }
int findMax ( int arr [ ] , int n ) { int mi , i ; for ( mi = 0 , i = 0 ; i < n ; ++ i ) if ( arr [ i ] > arr [ mi ] ) mi = i ; return mi ; }
void pancakeSort ( int * arr , int n ) {
for ( int curr_size = n ; curr_size > 1 ; -- curr_size ) {
int mi = findMax ( arr , curr_size ) ;
if ( mi != curr_size - 1 ) {
flip ( arr , mi ) ;
flip ( arr , curr_size - 1 ) ; } } }
void printArray ( int arr [ ] , int n ) { for ( int i = 0 ; i < n ; ++ i ) cout << arr [ i ] << " ▁ " ; }
int main ( ) { int arr [ ] = { 23 , 10 , 20 , 11 , 12 , 6 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; pancakeSort ( arr , n ) ; cout << " Sorted ▁ Array ▁ " << endl ; printArray ( arr , n ) ; return 0 ; }
